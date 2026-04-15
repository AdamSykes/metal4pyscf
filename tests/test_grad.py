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

    def test_df_uks(self, oh_radical):
        from gpu4pyscf.dft import UKS
        from pyscf.dft import uks as uks_cpu
        g_ref = uks_cpu.UKS(oh_radical, xc='B3LYP').density_fit().run().nuc_grad_method().kernel()
        g = UKS(oh_radical, xc='B3LYP').density_fit().run(verbose=0).nuc_grad_method().kernel()
        assert np.linalg.norm(g - g_ref) < 1e-4

    def test_df_rks(self, h2o):
        from gpu4pyscf.dft import RKS
        g_ref = rks_cpu.RKS(h2o, xc='B3LYP').density_fit().run().nuc_grad_method().kernel()
        g = RKS(h2o, xc='B3LYP').density_fit().run(verbose=0).nuc_grad_method().kernel()
        # f64 refinement gives ~1e-7 accuracy
        assert np.linalg.norm(g - g_ref) < 1e-5


class TestProductionPath:
    """Gradient regression tests that exercise the Metal int3c2e_ip1 path.

    The production dispatcher in df_grad_metal.py routes to the Metal kernel
    only for mol.nao >= 100 (and nroots <= 5). Every test in TestGradients
    above uses small molecules that fall through to CPU, which is why the
    Rys sqrt(pi/4) bug in the Metal kernel went undetected for months.
    These tests use benzene/def2-svp (nao = 114) so the Metal path is
    actually executed.

    The PySCF CPU reference is computed in a subprocess. Once gpu4pyscf
    is imported and a density_fit() Metal RKS is created in the current
    process, the global monkey-patch on pyscf.df.grad.rhf._int3c_wrapper
    routes every subsequent CPU gradient call through the Metal kernel
    too, so an in-process "CPU reference" would be polluted.
    """

    BENZENE_ATOM = (
        'C 0.000 1.396 0.000; C 1.209 0.698 0.000; C 1.209 -0.698 0.000; '
        'C 0.000 -1.396 0.000; C -1.209 -0.698 0.000; C -1.209 0.698 0.000; '
        'H 0.000 2.479 0.000; H 2.147 1.240 0.000; H 2.147 -1.240 0.000; '
        'H 0.000 -2.479 0.000; H -2.147 -1.240 0.000; H -2.147 1.240 0.000'
    )

    @staticmethod
    def _cpu_ref_gradient(atom, basis, xc='B3LYP'):
        """Compute a PySCF DF gradient in a clean subprocess."""
        import subprocess
        import sys
        import json
        script = (
            'import json, numpy as np\n'
            'from pyscf import gto\n'
            'from pyscf.dft import rks\n'
            f'mol = gto.M(atom={atom!r}, basis={basis!r}, verbose=0)\n'
            f'mf = rks.RKS(mol, xc={xc!r}).density_fit()\n'
            'mf.conv_tol = 1e-10\n'
            'mf.kernel()\n'
            'g = mf.nuc_grad_method().kernel()\n'
            'print(json.dumps({"nao": mol.nao, "g": g.tolist()}))\n'
        )
        r = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=180,
        )
        assert r.returncode == 0, f'CPU reference subprocess failed:\n{r.stderr}'
        data = json.loads(r.stdout.strip().splitlines()[-1])
        return data['nao'], np.array(data['g'])

    def test_df_rks_benzene(self):
        """Benzene/def2-svp DF-B3LYP gradient — exercises the Metal path."""
        from pyscf import gto
        from gpu4pyscf.dft import RKS

        nao_ref, g_ref = self._cpu_ref_gradient(self.BENZENE_ATOM, 'def2-svp')
        assert nao_ref >= 100, (
            f'test molecule must exceed the nao<100 gate in '
            f'df_grad_metal.py to exercise Metal int3c2e_ip1 (got nao={nao_ref})'
        )

        mol = gto.M(atom=self.BENZENE_ATOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='B3LYP').density_fit()
        mf.kernel()
        g = mf.nuc_grad_method().kernel()

        diff = np.linalg.norm(g - g_ref)
        max_diff = np.max(np.abs(g - g_ref))
        # Budget: Metal kernel runs in f32, SCF converges at conv_tol=1e-4.
        # On benzene the observed gradient norm difference is ~3e-5; allow
        # 2x headroom. A Rys precision bug typically pushes this to >1e-2.
        assert diff < 1e-4, (
            f'Metal gradient diverges from CPU: ||diff||={diff:.3e}, '
            f'max|diff|={max_diff:.3e}, CPU |g|={np.linalg.norm(g_ref):.3e}'
        )


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
