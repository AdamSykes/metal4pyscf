"""
Fixtures for backend verification tests.

Loads backend modules via importlib to avoid triggering gpu4pyscf/__init__.py
(which imports cupy and would fail on non-CUDA machines).
"""

import os
import importlib.util
import numpy as np
import pytest

BACKENDS_DIR = os.path.join(os.path.dirname(__file__), '..')

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------

_HAS_MLX = False
try:
    import mlx.core  # noqa: F401
    _HAS_MLX = True
except ImportError:
    pass

_HAS_PYSCF = False
try:
    import pyscf  # noqa: F401
    _HAS_PYSCF = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Backend loader (bypasses gpu4pyscf.__init__)
# ---------------------------------------------------------------------------

def _load_backend_module(name):
    """Load a backend module directly from its file path."""
    path = os.path.join(BACKENDS_DIR, f'{name}_backend.py')
    spec = importlib.util.spec_from_file_location(
        f'gpu4pyscf.lib.backends.{name}_backend', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Cache loaded modules
_BACKEND_CACHE = {}

def get_backend(name):
    if name not in _BACKEND_CACHE:
        _BACKEND_CACHE[name] = _load_backend_module(name)
    return _BACKEND_CACHE[name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _backend_ids():
    ids = ['numpy']
    if _HAS_MLX:
        ids.append('mlx')
    return ids


@pytest.fixture(params=_backend_ids())
def backend(request):
    """Parametrized fixture yielding each available backend module."""
    return get_backend(request.param)


@pytest.fixture
def mlx_backend():
    """MLX backend only — skips if MLX is not installed."""
    if not _HAS_MLX:
        pytest.skip('MLX not available')
    return get_backend('mlx')


@pytest.fixture
def numpy_backend():
    """NumPy backend (always available)."""
    return get_backend('numpy')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(backend_mod, arr):
    """Convert a backend array to NumPy."""
    return backend_mod.to_host(arr)


def get_tolerance(backend_mod, tight=1e-10, loose=1e-5):
    """Return atol appropriate for the backend's precision.

    MLX truncates float64→float32, so it gets the looser tolerance.
    """
    if backend_mod.BACKEND_NAME == 'mlx':
        return loose
    return tight


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)
