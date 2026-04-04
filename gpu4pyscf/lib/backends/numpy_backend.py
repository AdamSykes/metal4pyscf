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
NumPy (CPU-only) fallback backend.

Used when neither CuPy nor MLX is available.  All arrays are plain
NumPy arrays; "device" and "stream" APIs are no-ops.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Array namespace
# ---------------------------------------------------------------------------

xp = np
ndarray = np.ndarray


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class DeviceManager:
    def device_count(self):
        return 0

    def get_device_properties(self, device_id=0):
        return {
            'name': 'CPU',
            'total_memory': 0,
            'shared_memory_per_block': 0,
            'shared_memory_per_block_optin': 0,
            'backend': 'numpy',
        }

    def current_device_id(self):
        return 0

    class _NoopCtx:
        def __init__(self, device_id):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def set_device(self, device_id):
        return self._NoopCtx(device_id)

    def synchronize(self, device_id=None):
        pass

    def can_access_peer(self, src, dst):
        return True

device_manager = DeviceManager()


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

class MemoryManager:
    def get_mem_info(self):
        return (0, 0)

    def get_pool_used(self):
        return 0

    def get_pool_total(self):
        return 0

    def get_pool_limit(self):
        return 0

    def set_pool_limit(self, fraction=None, nbytes=None):
        pass

    def free_all_blocks(self):
        pass

    def alloc_pinned(self, nbytes):
        return np.empty(nbytes, dtype=np.uint8)

memory_manager = MemoryManager()


# ---------------------------------------------------------------------------
# Stream / synchronisation
# ---------------------------------------------------------------------------

class _NoopStream:
    ptr = 0
    def synchronize(self):
        pass

class StreamManager:
    _default = _NoopStream()
    def get_current_stream(self):
        return self._default
    def synchronize_stream(self, stream=None):
        pass
    def create_stream(self, non_blocking=True):
        return _NoopStream()

stream_manager = StreamManager()


# ---------------------------------------------------------------------------
# Data transfer helpers
# ---------------------------------------------------------------------------

def to_host(a):
    return np.asarray(a)

def to_device(a, dtype=None):
    if dtype is not None:
        return np.asarray(a, dtype=dtype)
    return np.asarray(a)

def is_device_array(a):
    return False


# ---------------------------------------------------------------------------
# Raw kernel compilation
# ---------------------------------------------------------------------------

def compile_kernel(source_code, kernel_name):
    raise NotImplementedError(
        'Raw GPU kernels are not available with the NumPy backend.')


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def eigh(a):
    return np.linalg.eigh(a)

def eigvalsh(a):
    return np.linalg.eigvalsh(a)

def solve(a, b):
    return np.linalg.solve(a, b)

def svd(a, full_matrices=True):
    return np.linalg.svd(a, full_matrices=full_matrices)

def inv(a):
    return np.linalg.inv(a)

def norm(a, *args, **kwargs):
    return np.linalg.norm(a, *args, **kwargs)

def dot(a, b, out=None):
    return np.dot(a, b, out=out)

def einsum(*args, **kwargs):
    return np.einsum(*args, **kwargs)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

BACKEND_NAME = 'numpy'
