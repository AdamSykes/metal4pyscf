"""Hessian and frequency tests."""

import numpy as np
import pytest
from pyscf.dft import rks as rks_cpu
from pyscf.hessian import thermo


class TestHessian:
    def test_df_rks(self, h2o):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP').density_fit()
        mf_ref.kernel()
        h_ref = mf_ref.Hessian().kernel()

        mf = RKS(h2o, xc='B3LYP').density_fit()
        mf.verbose = 0
        mf.kernel()
        h = mf.Hessian().kernel()

        assert np.max(np.abs(h - h_ref)) / np.max(np.abs(h_ref)) < 1e-5

    def test_frequencies(self, h2o):
        from gpu4pyscf.dft import RKS
        mf = RKS(h2o, xc='B3LYP').density_fit()
        mf.verbose = 0
        mf.kernel()
        h = mf.Hessian().kernel()
        freq = thermo.harmonic_analysis(h2o, h)['freq_wavenumber']
        # Should have 3 real vibrations for water
        real_freq = [f for f in freq if abs(f) > 100]
        assert len(real_freq) == 3
        # All positive (real minimum)
        assert all(f > 0 for f in real_freq)
