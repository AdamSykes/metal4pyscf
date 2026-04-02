"""
Layer 2: Chemistry integration tests.

Runs PySCF CPU calculations (RHF, DFT) and verifies that the same linear
algebra operations performed through each backend produce matching results.

This tests the critical numerical pathway — Fock matrix diagonalisation,
density matrix construction, energy evaluation — without requiring CUDA
integral kernels.
"""

import numpy as np
import scipy.linalg
import pytest

from .conftest import to_np, get_tolerance, _HAS_PYSCF

pytestmark = pytest.mark.skipif(not _HAS_PYSCF, reason='PySCF not installed')


# =========================================================================
# Fixtures: converged PySCF CPU calculations
# =========================================================================

@pytest.fixture(scope='module')
def h2o_rhf():
    """Converged RHF for H2O / STO-3G (7 basis functions)."""
    from pyscf import gto, scf
    mol = gto.M(
        atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470',
        basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged
    return mf, mol


@pytest.fixture(scope='module')
def h2_rhf():
    """Converged RHF for H2 / STO-3G (2 basis functions).

    Tiny molecule enabling full ERI contraction tests.
    """
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged
    return mf, mol


@pytest.fixture(scope='module')
def h2o_rks():
    """Converged RKS/LDA for H2O / def2-svp (24 basis functions)."""
    from pyscf import gto, dft
    mol = gto.M(
        atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470',
        basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='LDA')
    mf.kernel()
    assert mf.converged
    return mf, mol


# =========================================================================
# Helpers
# =========================================================================

def _lowdin_orthogonalise(S_np):
    """Compute S^{-1/2} via eigendecomposition (Löwdin orthogonalisation).

    Returns X such that X^T S X = I.
    """
    w, v = scipy.linalg.eigh(S_np)
    # Discard near-zero eigenvalues (linear dependence)
    mask = w > 1e-10
    X = v[:, mask] / np.sqrt(w[mask])
    return X


def _lowdin_via_backend(S_np, backend):
    """Same as _lowdin_orthogonalise but using the backend for eigh."""
    S = backend.to_device(S_np)
    w, v = backend.eigh(S)
    w_np = to_np(backend, w)
    v_np = to_np(backend, v)
    mask = w_np > 1e-10
    X = v_np[:, mask] / np.sqrt(w_np[mask])
    return X


def _diag_fock_via_backend(F_np, S_np, backend):
    """Diagonalise F in the orthogonal basis using the backend.

    Returns (mo_energy, mo_coeff) in the original AO basis.
    """
    X = _lowdin_via_backend(S_np, backend)
    F_ortho_np = X.T @ F_np @ X
    F_ortho = backend.to_device(F_ortho_np)
    w, v = backend.eigh(F_ortho)
    w_np = to_np(backend, w)
    v_np = to_np(backend, v)
    # Transform back to AO basis: C = X @ V
    C = X @ v_np
    return w_np, C


# =========================================================================
# Fock matrix diagonalisation (the core SCF operation)
# =========================================================================

class TestFockDiagonalisation:

    def test_mo_energies(self, backend, h2o_rhf):
        """MO energies from backend Fock diag match PySCF."""
        mf, mol = h2o_rhf
        F = mf.get_fock()
        S = mf.get_ovlp()
        mo_energy_be, _ = _diag_fock_via_backend(F, S, backend)
        ref = mf.mo_energy

        # Löwdin orthogonalisation introduces ~1e-6 numerical noise
        atol = get_tolerance(backend, tight=1e-5, loose=1e-3)
        np.testing.assert_allclose(
            sorted(mo_energy_be), sorted(ref), atol=atol,
            err_msg='MO energies from backend do not match PySCF')

    def test_mo_coefficients_overlap(self, backend, h2o_rhf):
        """MO coefficients span the same space (handles sign ambiguity)."""
        mf, mol = h2o_rhf
        F = mf.get_fock()
        S = mf.get_ovlp()
        _, C_be = _diag_fock_via_backend(F, S, backend)
        C_ref = mf.mo_coeff

        # Check overlap: |C_be^T @ S @ C_ref| should be close to
        # a permutation matrix (diagonal if orbitals are in the same order)
        overlap = np.abs(C_be.T @ S @ C_ref)
        # Each column of C_ref should overlap strongly with exactly one
        # column of C_be.  The max overlap per column should be ~1.
        max_overlaps = overlap.max(axis=0)
        atol = get_tolerance(backend, tight=1e-6, loose=1e-3)
        np.testing.assert_allclose(
            max_overlaps, 1.0, atol=atol,
            err_msg='MO coefficient spaces do not match')


class TestDensityMatrix:

    def test_reconstruction(self, backend, h2o_rhf):
        """Density matrix D = C_occ @ C_occ^T via backend matches PySCF."""
        mf, mol = h2o_rhf
        F = mf.get_fock()
        S = mf.get_ovlp()
        mo_energy_be, C_be = _diag_fock_via_backend(F, S, backend)

        # Occupy the lowest-energy orbitals
        nocc = mol.nelectron // 2
        # Sort by energy
        order = np.argsort(mo_energy_be)
        C_occ = C_be[:, order[:nocc]]

        # Build density matrix through the backend
        C_occ_dev = backend.to_device(C_occ)
        D_dev = backend.dot(C_occ_dev, backend.to_device(C_occ.T))
        D_be = to_np(backend, D_dev) * 2.0  # factor of 2 for RHF

        D_ref = mf.make_rdm1()
        atol = get_tolerance(backend, tight=1e-5, loose=1e-3)
        np.testing.assert_allclose(
            D_be, D_ref, atol=atol,
            err_msg='Backend density matrix does not match PySCF')


# =========================================================================
# Overlap matrix operations
# =========================================================================

class TestOverlapMatrix:

    def test_eigh_positive_definite(self, backend, h2o_rhf):
        """All eigenvalues of the overlap matrix should be positive."""
        mf, mol = h2o_rhf
        S_np = mf.get_ovlp()
        S = backend.to_device(S_np)
        w, _ = backend.eigh(S)
        w_np = to_np(backend, w)
        assert np.all(w_np > 0), \
            f'Non-positive eigenvalue in overlap matrix: {w_np.min()}'

    def test_eigh_matches_scipy(self, backend, h2o_rhf):
        """Overlap eigenvalues match SciPy reference."""
        mf, mol = h2o_rhf
        S_np = mf.get_ovlp()
        ref_w, _ = scipy.linalg.eigh(S_np)

        S = backend.to_device(S_np)
        w, _ = backend.eigh(S)
        w_np = to_np(backend, w)
        atol = get_tolerance(backend, tight=1e-10, loose=1e-5)
        np.testing.assert_allclose(w_np, ref_w, atol=atol)

    def test_inverse_via_solve(self, backend, h2o_rhf):
        """S^{-1} computed via solve satisfies S @ S^{-1} = I."""
        mf, mol = h2o_rhf
        S_np = mf.get_ovlp()
        n = S_np.shape[0]
        I_np = np.eye(n)

        S = backend.to_device(S_np)
        I_dev = backend.to_device(I_np)
        Sinv = backend.solve(S, I_dev)
        Sinv_np = to_np(backend, Sinv)

        product = S_np @ Sinv_np
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(product, I_np, atol=atol)


# =========================================================================
# Energy computation
# =========================================================================

class TestEnergy:

    def test_one_electron_energy(self, backend, h2o_rhf):
        """One-electron energy via backend einsum matches PySCF."""
        mf, mol = h2o_rhf
        h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        D = mf.make_rdm1()

        h1e_dev = backend.to_device(h1e)
        D_dev = backend.to_device(D)
        e1_dev = backend.einsum('ij,ji->', h1e_dev, D_dev)
        e1_be = float(to_np(backend, e1_dev))

        e1_ref = np.einsum('ij,ji->', h1e, D)
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(
            e1_be, e1_ref, atol=atol,
            err_msg='One-electron energy mismatch')

    def test_eri_coulomb_contraction(self, backend, h2_rhf):
        """Coulomb matrix J from ERI contraction matches PySCF (H2/STO-3G).

        For this tiny molecule (2 basis fns), the full (2,2,2,2) ERI tensor
        fits in memory, letting us test einsum('kl,ijkl->ij', D, eri).
        """
        mf, mol = h2_rhf
        D = mf.make_rdm1()
        eri = mol.intor('int2e')  # (2,2,2,2) tensor

        D_dev = backend.to_device(D)
        eri_dev = backend.to_device(eri)
        J_dev = backend.einsum('kl,ijkl->ij', D_dev, eri_dev)
        J_be = to_np(backend, J_dev)

        J_ref, _ = mf.get_jk()
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(
            J_be, J_ref, atol=atol,
            err_msg='Coulomb matrix J from backend does not match PySCF')


# =========================================================================
# DFT with larger basis
# =========================================================================

class TestDFT:

    def test_rks_mo_energies(self, backend, h2o_rks):
        """MO energies from backend Fock diag match PySCF DFT/LDA."""
        mf, mol = h2o_rks
        F = mf.get_fock()
        S = mf.get_ovlp()
        mo_energy_be, _ = _diag_fock_via_backend(F, S, backend)
        ref = mf.mo_energy

        atol = get_tolerance(backend, tight=1e-5, loose=1e-3)
        np.testing.assert_allclose(
            sorted(mo_energy_be), sorted(ref), atol=atol,
            err_msg='DFT MO energies from backend do not match PySCF')

    def test_rks_density_matrix(self, backend, h2o_rks):
        """Density matrix from backend Fock diag matches PySCF DFT."""
        mf, mol = h2o_rks
        F = mf.get_fock()
        S = mf.get_ovlp()
        mo_energy_be, C_be = _diag_fock_via_backend(F, S, backend)

        nocc = mol.nelectron // 2
        order = np.argsort(mo_energy_be)
        C_occ = C_be[:, order[:nocc]]

        C_occ_dev = backend.to_device(C_occ)
        D_dev = backend.dot(C_occ_dev, backend.to_device(C_occ.T))
        D_be = to_np(backend, D_dev) * 2.0

        D_ref = mf.make_rdm1()
        atol = get_tolerance(backend, tight=1e-5, loose=1e-3)
        np.testing.assert_allclose(
            D_be, D_ref, atol=atol,
            err_msg='DFT density matrix from backend does not match PySCF')


# =========================================================================
# Numerical stability edge cases
# =========================================================================

class TestStability:

    def test_near_singular_overlap(self, backend):
        """Backend eigh handles a near-singular matrix gracefully."""
        # Create a matrix with a very small eigenvalue
        n = 10
        w = np.array([1e-12] + list(np.linspace(0.1, 1.0, n - 1)))
        V = scipy.linalg.orth(np.random.randn(n, n))
        A_np = V @ np.diag(w) @ V.T

        A = backend.to_device(A_np)
        w_be, v_be = backend.eigh(A)
        w_be_np = to_np(backend, w_be)

        ref_w = scipy.linalg.eigvalsh(A_np)
        atol = get_tolerance(backend, tight=1e-6, loose=1e-3)
        np.testing.assert_allclose(
            sorted(w_be_np), sorted(ref_w), atol=atol,
            err_msg='Near-singular eigenvalues do not match')

    def test_large_matrix(self, backend):
        """100x100 eigh through backend matches SciPy."""
        A_np = np.random.randn(100, 100)
        A_np = A_np + A_np.T + 100 * np.eye(100)  # SPD
        ref_w, _ = scipy.linalg.eigh(A_np)

        A = backend.to_device(A_np)
        w, _ = backend.eigh(A)
        w_np = to_np(backend, w)
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(w_np, ref_w, atol=atol)
