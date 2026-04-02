"""Test that all gpu4pyscf modules import on Apple Silicon."""

import pytest


MODULES = [
    'gpu4pyscf',
    'gpu4pyscf.scf',
    'gpu4pyscf.dft',
    'gpu4pyscf.grad',
    'gpu4pyscf.hessian',
    'gpu4pyscf.solvent',
    'gpu4pyscf.tdscf',
    'gpu4pyscf.df',
    'gpu4pyscf.mp',
    'gpu4pyscf.cc',
    'gpu4pyscf.qmmm',
    'gpu4pyscf.pbc',
    'gpu4pyscf.properties',
    'gpu4pyscf.tools',
    'gpu4pyscf.md',
    'gpu4pyscf.nac',
    'gpu4pyscf.lib',
    'gpu4pyscf.lib.backend',
    'gpu4pyscf.lib.backends',
]


@pytest.mark.parametrize('module', MODULES)
def test_import(module):
    __import__(module)


CLASSES = [
    ('gpu4pyscf.scf', 'RHF'),
    ('gpu4pyscf.scf', 'UHF'),
    ('gpu4pyscf.scf', 'ROHF'),
    ('gpu4pyscf.scf', 'GHF'),
    ('gpu4pyscf.scf', 'HF'),
    ('gpu4pyscf.dft', 'RKS'),
    ('gpu4pyscf.dft', 'UKS'),
    ('gpu4pyscf.dft', 'GKS'),
    ('gpu4pyscf.dft', 'ROKS'),
    ('gpu4pyscf.dft', 'KS'),
    ('gpu4pyscf.dft', 'Grids'),
    ('gpu4pyscf.df', 'DF'),
    ('gpu4pyscf.mp', 'MP2'),
    ('gpu4pyscf.cc', 'CCSD'),
]


@pytest.mark.parametrize('module,cls', CLASSES)
def test_class_accessible(module, cls):
    mod = __import__(module, fromlist=[cls])
    assert hasattr(mod, cls)


def test_backend_name():
    import gpu4pyscf
    assert gpu4pyscf.BACKEND_NAME in ('cupy', 'mlx', 'numpy')
