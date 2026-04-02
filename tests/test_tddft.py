"""TDDFT and TDA tests."""

import numpy as np
import pytest
from pyscf.dft import rks as rks_cpu


class TestTDDFT:
    def test_excitations(self, h2o):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP'); mf_ref.kernel()
        td_ref = mf_ref.TDDFT(); td_ref.nstates = 3; td_ref.kernel()

        mf = RKS(h2o, xc='B3LYP'); mf.verbose = 0; mf.kernel()
        td = mf.TDDFT(); td.nstates = 3; td.kernel()

        eV = 27.2114
        assert np.max(np.abs(td.e - td_ref.e)) * eV < 1e-4

    def test_df_excitations(self, h2o):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP').density_fit(); mf_ref.kernel()
        td_ref = mf_ref.TDDFT(); td_ref.nstates = 3; td_ref.kernel()

        mf = RKS(h2o, xc='B3LYP').density_fit(); mf.verbose = 0; mf.kernel()
        td = mf.TDDFT(); td.nstates = 3; td.kernel()

        eV = 27.2114
        assert np.max(np.abs(td.e - td_ref.e)) * eV < 1e-3


class TestTDA:
    def test_excitations(self, h2o):
        from gpu4pyscf.dft import RKS
        mf_ref = rks_cpu.RKS(h2o, xc='B3LYP').density_fit(); mf_ref.kernel()
        td_ref = mf_ref.TDA(); td_ref.nstates = 3; td_ref.kernel()

        mf = RKS(h2o, xc='B3LYP').density_fit(); mf.verbose = 0; mf.kernel()
        td = mf.TDA(); td.nstates = 3; td.kernel()

        eV = 27.2114
        assert np.max(np.abs(td.e - td_ref.e)) * eV < 1e-3
