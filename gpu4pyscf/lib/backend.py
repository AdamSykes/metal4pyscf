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
Unified GPU backend API for gpu4pyscf.

Usage
-----
Replace direct CuPy usage with this module::

    # Old (CUDA-only):
    #   import cupy
    #   a = cupy.zeros((n, n))
    #   cupy.cuda.Device(0).synchronize()

    # New (backend-agnostic):
    from gpu4pyscf.lib.backend import xp, device, memory, stream

    a = xp.zeros((n, n))
    device.synchronize()

The actual backend (CuPy, MLX, or NumPy) is selected automatically at
import time — see ``gpu4pyscf.lib.backends`` for details.

Attributes
----------
xp : module
    The array-namespace module (cupy, mlx.core, or numpy).  Provides
    ``xp.zeros``, ``xp.dot``, ``xp.linalg.eigh``, etc.

device : DeviceManager
    Device enumeration, selection, and synchronisation.

memory : MemoryManager
    Memory pool queries and limits.

stream : StreamManager
    Execution stream / command-queue management.

BACKEND_NAME : str
    ``'cupy'``, ``'mlx'``, or ``'numpy'``.
"""

from gpu4pyscf.lib.backends import get_backend as _get_backend

_backend = _get_backend()

# ---- public API ----

#: Array namespace (cupy | mlx.core | numpy).
xp = _backend.xp

#: The GPU ndarray type for isinstance checks.
ndarray = _backend.ndarray

#: Device management.
device = _backend.device_manager

#: Memory management.
memory = _backend.memory_manager

#: Stream / command-queue management.
stream = _backend.stream_manager

#: Name of the active backend.
BACKEND_NAME = _backend.BACKEND_NAME

# ---- helpers re-exported from the backend ----

to_host = _backend.to_host
to_device = _backend.to_device
is_device_array = _backend.is_device_array
compile_kernel = _backend.compile_kernel

# ---- linear algebra (unified API) ----

eigh = _backend.eigh
eigvalsh = _backend.eigvalsh
solve = _backend.solve
svd = _backend.svd
inv = _backend.inv
norm = _backend.norm
dot = _backend.dot
einsum = _backend.einsum
