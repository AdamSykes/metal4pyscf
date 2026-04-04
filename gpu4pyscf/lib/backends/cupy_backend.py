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
CuPy/CUDA backend — wraps the existing CuPy behaviour so that code
migrated to ``from gpu4pyscf.lib.backend import xp`` keeps working
identically on NVIDIA hardware.
"""

import numpy as np
import cupy

# ---------------------------------------------------------------------------
# Array namespace — drop-in for ``import cupy``
# ---------------------------------------------------------------------------

# Re-export the full cupy module as the array namespace
xp = cupy

# Convenience: the GPU ndarray type used for isinstance checks
ndarray = cupy.ndarray


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class DeviceManager:
    """Thin wrapper around cupy.cuda.Device and CUDA runtime."""

    def device_count(self):
        return cupy.cuda.runtime.getDeviceCount()

    def get_device_properties(self, device_id=0):
        props = cupy.cuda.runtime.getDeviceProperties(device_id)
        # Normalise to a dict with consistent keys across backends
        return {
            'name': (props.get('name', b'CUDA device').decode()
                     if isinstance(props.get('name'), bytes)
                     else props.get('name', 'CUDA device')),
            'total_memory': props['totalGlobalMem'],
            'shared_memory_per_block': props.get('sharedMemPerBlock', 0),
            'shared_memory_per_block_optin': props.get('sharedMemPerBlockOptin', 0),
            'backend': 'cupy',
        }

    def current_device_id(self):
        return cupy.cuda.Device().id

    def set_device(self, device_id):
        """Context manager for device selection."""
        return cupy.cuda.Device(device_id)

    def synchronize(self, device_id=None):
        if device_id is not None:
            cupy.cuda.Device(device_id).synchronize()
        else:
            cupy.cuda.Device().synchronize()

    def can_access_peer(self, src, dst):
        return cupy.cuda.runtime.deviceCanAccessPeer(src, dst)

device_manager = DeviceManager()


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

class MemoryManager:
    """Wraps CuPy memory pool and CUDA memory queries."""

    def get_mem_info(self):
        """Return (free, total) in bytes for current device."""
        return cupy.cuda.runtime.memGetInfo()

    def get_pool_used(self):
        return cupy.get_default_memory_pool().used_bytes()

    def get_pool_total(self):
        return cupy.get_default_memory_pool().total_bytes()

    def get_pool_limit(self):
        return cupy.get_default_memory_pool().get_limit()

    def set_pool_limit(self, fraction=None, nbytes=None):
        pool = cupy.get_default_memory_pool()
        if fraction is not None:
            pool.set_limit(fraction=fraction)
        elif nbytes is not None:
            pool.set_limit(size=nbytes)

    def free_all_blocks(self):
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    def alloc_pinned(self, nbytes):
        return cupy.cuda.alloc_pinned_memory(nbytes)

memory_manager = MemoryManager()


# ---------------------------------------------------------------------------
# Stream / synchronisation
# ---------------------------------------------------------------------------

class StreamManager:
    """Wraps CUDA stream operations."""

    def get_current_stream(self):
        return cupy.cuda.get_current_stream()

    def synchronize_stream(self, stream=None):
        if stream is None:
            cupy.cuda.Stream.null.synchronize()
        else:
            stream.synchronize()

    def create_stream(self, non_blocking=True):
        return cupy.cuda.Stream(non_blocking=non_blocking)

stream_manager = StreamManager()


# ---------------------------------------------------------------------------
# Data transfer helpers
# ---------------------------------------------------------------------------

def to_host(a):
    """GPU array -> NumPy array."""
    if isinstance(a, cupy.ndarray):
        return a.get()
    return np.asarray(a)


def to_device(a, dtype=None):
    """NumPy array -> GPU array."""
    if dtype is not None:
        return cupy.asarray(a, dtype=dtype)
    return cupy.asarray(a)


def is_device_array(a):
    """True if *a* lives on a GPU device."""
    return isinstance(a, cupy.ndarray)


# ---------------------------------------------------------------------------
# Raw kernel compilation
# ---------------------------------------------------------------------------

def compile_kernel(source_code, kernel_name):
    """Compile a raw GPU kernel from source.

    For CuPy this delegates to ``cupy.RawKernel``.
    """
    return cupy.RawKernel(source_code, kernel_name)


# ---------------------------------------------------------------------------
# Linear-algebra helpers (unified API)
# ---------------------------------------------------------------------------

def eigh(a):
    return cupy.linalg.eigh(a)

def eigvalsh(a):
    return cupy.linalg.eigvalsh(a)

def solve(a, b):
    return cupy.linalg.solve(a, b)

def svd(a, full_matrices=True):
    return cupy.linalg.svd(a, full_matrices=full_matrices)

def inv(a):
    return cupy.linalg.inv(a)

def norm(a, *args, **kwargs):
    return cupy.linalg.norm(a, *args, **kwargs)

def dot(a, b, out=None):
    return cupy.dot(a, b, out=out)

def einsum(*args, **kwargs):
    return cupy.einsum(*args, **kwargs)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

BACKEND_NAME = 'cupy'
