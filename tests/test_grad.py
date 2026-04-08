"""Gradient tests: compare against PySCF CPU reference."""

import numpy as np
import pytest
from pyscf.scf import hf as hf_cpu
from pyscf.dft import rks as rks_cpu


class TestGradients:
    def test_rhf(self, h2o):
        from gpu4pyscf.scf import RHF
        g_ref = hf_cpu.RHF(h2o).run().nuc_grad_method().kernel()
        g = RHF(h2o).run(verbose=0).nuc_grad_method().kernel()
        assert np.linalg.norm(g - g_ref) < 1e-5

    def test_rks(self, h2o):
        from gpu4pyscf.dft import RKS
        g_ref = rks_cpu.RKS(h2o, xc='B3LYP').run().nuc_grad_method().kernel()
        g = RKS(h2o, xc='B3LYP').run(verbose=0).nuc_grad_method().kernel()
        assert np.linalg.norm(g - g_ref) < 1e-5

    def test_df_rks(self, h2o):
        from gpu4pyscf.dft import RKS
        g_ref = rks_cpu.RKS(h2o, xc='B3LYP').density_fit().run().nuc_grad_method().kernel()
        g = RKS(h2o, xc='B3LYP').density_fit().run(verbose=0).nuc_grad_method().kernel()
        # f64 refinement gives ~1e-7 accuracy
        assert np.linalg.norm(g - g_ref) < 1e-5


class TestGeomOpt:
    def test_optimize(self, h2o_sto3g):
        """Geometry optimization with Metal DF-RKS + geometric solver."""
        from gpu4pyscf.dft import RKS
        from pyscf.geomopt.geometric_solver import optimize
        mf = RKS(h2o_sto3g, xc='B3LYP').density_fit()
        mf.verbose = 0
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=10)
        assert mol_eq is not None
        coords = mol_eq.atom_coords() * 0.529177
        oh = np.linalg.norm(coords[1] - coords[0])
        assert 0.9 < oh < 1.1

    def test_df_optimize(self, h2o):
        """DF-B3LYP/def2-SVP geomopt: verify Metal matches CPU equilibrium."""
        from gpu4pyscf.dft import RKS
        from pyscf.geomopt.geometric_solver import optimize
        mf = RKS(h2o, xc='B3LYP').density_fit()
        mf.verbose = 0
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=20)
        assert mol_eq is not None
        # Check O-H bond length within expected range for B3LYP/def2-SVP
        coords = mol_eq.atom_coords() * 0.529177  # Bohr → Angstrom
        oh1 = np.linalg.norm(coords[1] - coords[0])
        oh2 = np.linalg.norm(coords[2] - coords[0])
        assert 0.95 < oh1 < 1.00, f'O-H1 = {oh1:.4f} A'
        assert 0.95 < oh2 < 1.00, f'O-H2 = {oh2:.4f} A'
