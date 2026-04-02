# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Backend selection for gpu4pyscf.

Detects available GPU backends and selects the appropriate one:
  - 'cupy'  : NVIDIA CUDA via CuPy (original)
  - 'mlx'   : Apple Silicon Metal via MLX
  - 'numpy'  : CPU-only fallback via NumPy

Set the environment variable GPU4PYSCF_BACKEND to force a specific backend.
"""

import os
import sys
import warnings

AVAILABLE_BACKENDS = {}

# Probe for CuPy
try:
    import cupy  # noqa: F401
    AVAILABLE_BACKENDS['cupy'] = True
except ImportError:
    AVAILABLE_BACKENDS['cupy'] = False

# Probe for MLX
try:
    import mlx.core  # noqa: F401
    AVAILABLE_BACKENDS['mlx'] = True
except ImportError:
    AVAILABLE_BACKENDS['mlx'] = False

# NumPy is always available
AVAILABLE_BACKENDS['numpy'] = True


def _detect_backend():
    """Select the best available backend, or honor GPU4PYSCF_BACKEND env var."""
    forced = os.environ.get('GPU4PYSCF_BACKEND', '').lower().strip()
    if forced:
        if forced not in ('cupy', 'mlx', 'numpy'):
            raise ValueError(
                f"Unknown GPU4PYSCF_BACKEND={forced!r}. "
                f"Choose from: cupy, mlx, numpy")
        if not AVAILABLE_BACKENDS.get(forced):
            raise ImportError(
                f"Requested backend '{forced}' is not installed.")
        return forced

    # Auto-detect: prefer cupy (original), then mlx, then numpy
    if AVAILABLE_BACKENDS['cupy']:
        return 'cupy'
    if AVAILABLE_BACKENDS['mlx']:
        return 'mlx'
    warnings.warn(
        'No GPU backend found (cupy or mlx). Falling back to NumPy (CPU only).',
        stacklevel=3)
    return 'numpy'


BACKEND_NAME = _detect_backend()


def get_backend():
    """Return the active backend module (cupy_backend, mlx_backend, or numpy_backend)."""
    if BACKEND_NAME == 'cupy':
        from gpu4pyscf.lib.backends import cupy_backend
        return cupy_backend
    elif BACKEND_NAME == 'mlx':
        from gpu4pyscf.lib.backends import mlx_backend
        return mlx_backend
    else:
        from gpu4pyscf.lib.backends import numpy_backend
        return numpy_backend
