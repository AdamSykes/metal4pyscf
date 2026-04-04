"""Solvent model tests."""

import numpy as np
import pytest
from pyscf.dft import rks as rks_cpu


class TestPCM:
    @pytest.mark.parametrize('method', ['C-PCM', 'IEF-PCM', 'SS(V)PE'])
    def test_energy(self, h2o, method):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP').PCM()
        mf_ref.with_solvent.method = method
        mf_ref.kernel()

        mf = RKS(h2o, xc='B3LYP').PCM()
        mf.with_solvent.method = method
        mf.verbose = 0
        mf.kernel()

        assert abs(mf.e_tot - mf_ref.e_tot) < 1e-8

    def test_gradient(self, h2o):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP').PCM()
        mf_ref.with_solvent.method = 'C-PCM'
        mf_ref.kernel()
        g_ref = mf_ref.nuc_grad_method().kernel()

        mf = RKS(h2o, xc='B3LYP').PCM()
        mf.with_solvent.method = 'C-PCM'
        mf.verbose = 0
        mf.kernel()
        g = mf.nuc_grad_method().kernel()

        assert np.linalg.norm(g - g_ref) < 1e-5
