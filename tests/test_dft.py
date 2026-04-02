"""DFT tests: RKS, UKS with various functionals against PySCF CPU."""

import numpy as np
import pytest
import pyscf
from pyscf.dft import rks as rks_cpu, uks as uks_cpu


class TestRKS:
    @pytest.mark.parametrize('xc', ['LDA', 'PBE', 'B3LYP', 'PBE0', 'M06'])
    def test_energy(self, h2o, xc):
        from gpu4pyscf.dft import RKS
        e_ref = rks_cpu.RKS(h2o, xc=xc).run().e_tot
        e = RKS(h2o, xc=xc).run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-5  # f32 XC contraction precision

    @pytest.mark.parametrize('xc', ['B3LYP', 'PBE0'])
    def test_df_energy(self, h2o, xc):
        from gpu4pyscf.dft import RKS
        e_ref = rks_cpu.RKS(h2o, xc=xc).density_fit().run().e_tot
        e = RKS(h2o, xc=xc).density_fit().run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-4


class TestUKS:
    def test_energy(self, oh_radical):
        from gpu4pyscf.dft import UKS
        e_ref = uks_cpu.UKS(oh_radical, xc='B3LYP').run().e_tot
        e = UKS(oh_radical, xc='B3LYP').run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-5

    def test_df_energy(self, oh_radical):
        from gpu4pyscf.dft import UKS
        e_ref = uks_cpu.UKS(oh_radical, xc='PBE').density_fit().run().e_tot
        e = UKS(oh_radical, xc='PBE').density_fit().run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-4


class TestGKS:
    def test_accessible(self):
        from gpu4pyscf.dft import GKS
        assert GKS is not None


class TestROKS:
    def test_accessible(self):
        from gpu4pyscf.dft import ROKS
        assert ROKS is not None
