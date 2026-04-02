"""Shared fixtures for gpu4pyscf-metal integration tests."""

import pytest
import numpy as np
import pyscf


@pytest.fixture(scope='session')
def h2o():
    return pyscf.M(
        atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470',
        basis='def2-svp', verbose=0)


@pytest.fixture(scope='session')
def h2o_sto3g():
    return pyscf.M(
        atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470',
        basis='sto-3g', verbose=0)


@pytest.fixture(scope='session')
def oh_radical():
    return pyscf.M(
        atom='O 0 0 0; H 0 0 0.97',
        basis='def2-svp', spin=1, verbose=0)
