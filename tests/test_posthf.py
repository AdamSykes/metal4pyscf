"""Post-HF tests: MP2, CCSD."""

import numpy as np
import pytest
import pyscf


class TestMP2:
    def test_energy(self, h2o_sto3g):
        from gpu4pyscf.mp import MP2
        mf = pyscf.scf.RHF(h2o_sto3g).run()
        mp = MP2(mf)
        mp.kernel()
        assert mp.e_corr < 0  # correlation energy is negative
        # Compare with PySCF
        from pyscf.mp import MP2 as CPU_MP2
        mp_ref = CPU_MP2(mf).run()
        assert abs(mp.e_corr - mp_ref.e_corr) < 1e-10


class TestCCSD:
    def test_energy(self, h2o_sto3g):
        from gpu4pyscf.cc import CCSD
        mf = pyscf.scf.RHF(h2o_sto3g).run()
        cc = CCSD(mf)
        cc.kernel()
        assert cc.e_corr < 0
        from pyscf.cc import CCSD as CPU_CCSD
        cc_ref = CPU_CCSD(mf).run()
        assert abs(cc.e_corr - cc_ref.e_corr) < 1e-10
