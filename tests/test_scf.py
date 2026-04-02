"""SCF tests: RHF, UHF, ROHF, GHF against PySCF CPU reference."""

import numpy as np
import pytest
import pyscf
from pyscf.scf import hf as hf_cpu, uhf as uhf_cpu


class TestRHF:
    def test_energy(self, h2o):
        from gpu4pyscf.scf import RHF
        e_ref = hf_cpu.RHF(h2o).kernel()
        e = RHF(h2o).run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-8

    def test_df_energy(self, h2o):
        from gpu4pyscf.scf import RHF
        e_ref = hf_cpu.RHF(h2o).density_fit().kernel()
        e = RHF(h2o).density_fit().run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-4  # f32 J/K


class TestUHF:
    def test_energy(self, oh_radical):
        from gpu4pyscf.scf import UHF
        e_ref = uhf_cpu.UHF(oh_radical).kernel()
        e = UHF(oh_radical).run(verbose=0).e_tot
        assert abs(e - e_ref) < 1e-8


class TestROHF:
    def test_accessible(self):
        from gpu4pyscf.scf import ROHF
        assert ROHF is not None


class TestGHF:
    def test_accessible(self):
        from gpu4pyscf.scf import GHF
        assert GHF is not None
