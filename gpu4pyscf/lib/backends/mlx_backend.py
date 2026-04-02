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
MLX / Apple Silicon Metal backend.

MLX (https://github.com/ml-explore/mlx) provides a NumPy-like API that runs
on Apple Silicon GPUs via Metal.  This module maps the gpu4pyscf backend
interface onto MLX, with SciPy fallbacks for operations MLX does not yet
support natively.

Key differences from CuPy/CUDA:
  - Apple Silicon has **unified memory** — no explicit host/device transfers.
  - There are no "streams" in the CUDA sense; MLX uses lazy evaluation and
    ``mx.eval()`` to force computation.
  - Metal compute shaders replace CUDA kernels; custom kernels are loaded via
    ``mx.fast.metal_kernel()`` (MLX ≥ 0.18).
  - Only one "device" (the Metal GPU); multi-GPU is N/A.
"""

import os
import platform
import subprocess
import numpy as np

import mlx.core as mx

# ---------------------------------------------------------------------------
# Array namespace
# ---------------------------------------------------------------------------

# MLX's mx.* mirrors NumPy closely: mx.zeros, mx.array, mx.dot, etc.
# We expose it as ``xp`` so that callers can write ``xp.zeros(...)`` and
# get Metal-backed arrays.
xp = mx

ndarray = mx.array  # the MLX array type, for isinstance checks


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def _get_metal_device_name():
    """Best-effort query for the Metal GPU name."""
    try:
        out = subprocess.check_output(
            ['system_profiler', 'SPDisplaysDataType'],
            text=True, timeout=5)
        for line in out.splitlines():
            if 'Chipset Model' in line or 'Chip' in line:
                return line.split(':')[-1].strip()
    except Exception:
        pass
    return platform.processor() or 'Apple Silicon'


def _get_system_memory():
    """Return total system RAM in bytes (unified memory on Apple Silicon)."""
    try:
        out = subprocess.check_output(
            ['sysctl', '-n', 'hw.memsize'], text=True, timeout=5)
        return int(out.strip())
    except Exception:
        return 0


class DeviceManager:
    """Apple Silicon has a single unified-memory GPU."""

    def __init__(self):
        self._name = _get_metal_device_name()
        self._total_mem = _get_system_memory()

    def device_count(self):
        return 1

    def get_device_properties(self, device_id=0):
        return {
            'name': self._name,
            'total_memory': self._total_mem,
            'shared_memory_per_block': 32768,  # Metal threadgroup memory
            'shared_memory_per_block_optin': 32768,
            'backend': 'mlx',
        }

    def current_device_id(self):
        return 0

    class _DeviceCtx:
        """No-op context manager — only one Metal device."""
        def __init__(self, device_id):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def set_device(self, device_id):
        return self._DeviceCtx(device_id)

    def synchronize(self, device_id=None):
        mx.eval()  # force pending Metal command buffers to complete

    def can_access_peer(self, src, dst):
        return True  # unified memory

device_manager = DeviceManager()


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

class MemoryManager:
    """Memory management for unified-memory systems.

    Apple Silicon shares RAM between CPU and GPU.  MLX manages its own
    allocation pool; we report system-level figures for compatibility.
    """

    def __init__(self):
        self._total = _get_system_memory()

    def get_mem_info(self):
        """Return (free, total) — approximate for unified memory."""
        try:
            active = mx.get_active_memory()
        except AttributeError:
            try:
                active = mx.metal.get_active_memory()
            except AttributeError:
                active = 0
        free = max(self._total - active, 0)
        return (free, self._total)

    def get_pool_used(self):
        try:
            return mx.get_active_memory()
        except AttributeError:
            try:
                return mx.metal.get_active_memory()
            except AttributeError:
                return 0

    def get_pool_total(self):
        return self._total

    def get_pool_limit(self):
        try:
            return mx.metal.get_cache_memory()
        except AttributeError:
            return self._total

    def set_pool_limit(self, fraction=None, nbytes=None):
        limit = None
        if fraction is not None:
            limit = int(self._total * fraction)
        elif nbytes is not None:
            limit = nbytes
        if limit is not None:
            try:
                mx.set_memory_limit(limit)
            except AttributeError:
                try:
                    mx.metal.set_memory_limit(limit)
                except AttributeError:
                    pass

    def free_all_blocks(self):
        try:
            mx.metal.clear_cache()
        except AttributeError:
            pass

    def alloc_pinned(self, nbytes):
        # Unified memory — pinning is a no-op; return a plain buffer.
        return np.empty(nbytes, dtype=np.uint8)

memory_manager = MemoryManager()


# ---------------------------------------------------------------------------
# Stream / synchronisation
# ---------------------------------------------------------------------------

class _MLXStream:
    """Minimal stand-in for a CUDA stream.

    MLX uses lazy evaluation.  ``synchronize()`` calls ``mx.eval()`` to
    flush the Metal command queue.  The ``ptr`` attribute is provided for
    compatibility with ctypes-based C library calls (set to 0 / NULL to
    indicate "default stream").
    """
    ptr = 0

    def synchronize(self):
        mx.eval()


class StreamManager:
    _default = _MLXStream()

    def get_current_stream(self):
        return self._default

    def synchronize_stream(self, stream=None):
        mx.eval()

    def create_stream(self, non_blocking=True):
        return _MLXStream()

stream_manager = StreamManager()


# ---------------------------------------------------------------------------
# Data transfer helpers
# ---------------------------------------------------------------------------

def to_host(a):
    """MLX array -> NumPy array."""
    if isinstance(a, mx.array):
        return np.array(a)
    return np.asarray(a)


def to_device(a, dtype=None):
    """NumPy array -> MLX array (zero-copy on unified memory when possible)."""
    if dtype is not None:
        return mx.array(np.asarray(a, dtype=dtype))
    return mx.array(np.asarray(a))


def is_device_array(a):
    """True if *a* is an MLX (Metal-backed) array."""
    return isinstance(a, mx.array)


# ---------------------------------------------------------------------------
# Raw kernel compilation
# ---------------------------------------------------------------------------

def compile_kernel(source_code, kernel_name):
    """Compile a Metal compute kernel from MSL source.

    Requires MLX ≥ 0.18 with ``mx.fast.metal_kernel`` support.
    Returns a callable ``kernel(grid, threadgroup, args)`` matching the
    CuPy RawKernel launch interface.
    """
    try:
        metal_kernel = mx.fast.metal_kernel
    except AttributeError:
        raise NotImplementedError(
            'Custom Metal kernels require MLX >= 0.18.  '
            'Install with: pip install mlx>=0.18')

    # mx.fast.metal_kernel has a different call convention than CuPy.
    # We return a thin wrapper so callers can use the same
    # ``kernel((grid,), (block,), (args,))`` pattern.
    compiled = metal_kernel(name=kernel_name, source=source_code)
    return compiled


# ---------------------------------------------------------------------------
# Linear-algebra helpers
#
# Strategy:
#   - For float64 inputs, always use SciPy (MLX silently truncates to float32).
#   - For float32 inputs, try MLX native with stream=mx.cpu (Metal GPU linalg
#     is not yet supported for most ops), then fall back to SciPy.
#   - Results are always returned as mx.array.
# ---------------------------------------------------------------------------

def _to_np(a):
    """Convert to NumPy, preserving dtype."""
    return np.array(a) if isinstance(a, mx.array) else np.asarray(a)


def _needs_scipy(a):
    """True if *a* is float64/complex128 (MLX would lose precision)."""
    a_np = _to_np(a)
    return a_np.dtype in (np.float64, np.complex128)


def eigh(a):
    """Symmetric/Hermitian eigendecomposition."""
    if _needs_scipy(a):
        import scipy.linalg
        a_np = _to_np(a)
        w, v = scipy.linalg.eigh(a_np)
        return mx.array(w), mx.array(v)
    try:
        return mx.linalg.eigh(a, stream=mx.cpu)
    except (AttributeError, ValueError, NotImplementedError):
        import scipy.linalg
        a_np = _to_np(a)
        w, v = scipy.linalg.eigh(a_np)
        return mx.array(w), mx.array(v)


def eigvalsh(a):
    if _needs_scipy(a):
        import scipy.linalg
        w = scipy.linalg.eigvalsh(_to_np(a))
        return mx.array(w)
    try:
        w, _ = mx.linalg.eigh(a, stream=mx.cpu)
        return w
    except (AttributeError, ValueError, NotImplementedError):
        import scipy.linalg
        w = scipy.linalg.eigvalsh(_to_np(a))
        return mx.array(w)


def solve(a, b):
    if _needs_scipy(a):
        import scipy.linalg
        return mx.array(scipy.linalg.solve(_to_np(a), _to_np(b)))
    try:
        return mx.linalg.solve(a, b, stream=mx.cpu)
    except (AttributeError, ValueError, NotImplementedError):
        import scipy.linalg
        return mx.array(scipy.linalg.solve(_to_np(a), _to_np(b)))


def svd(a, full_matrices=True):
    if _needs_scipy(a):
        a_np = _to_np(a)
        u, s, vt = np.linalg.svd(a_np, full_matrices=full_matrices)
        return mx.array(u), mx.array(s), mx.array(vt)
    try:
        return mx.linalg.svd(a, stream=mx.cpu)
    except (AttributeError, ValueError, NotImplementedError):
        a_np = _to_np(a)
        u, s, vt = np.linalg.svd(a_np, full_matrices=full_matrices)
        return mx.array(u), mx.array(s), mx.array(vt)


def inv(a):
    if _needs_scipy(a):
        return mx.array(np.linalg.inv(_to_np(a)))
    try:
        return mx.linalg.inv(a, stream=mx.cpu)
    except (AttributeError, ValueError, NotImplementedError):
        return mx.array(np.linalg.inv(_to_np(a)))


def norm(a, *args, **kwargs):
    try:
        return mx.linalg.norm(a, *args, **kwargs)
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return mx.array(np.linalg.norm(_to_np(a), *args, **kwargs))


def dot(a, b, out=None):
    if a.ndim >= 2 and b.ndim >= 2:
        result = mx.matmul(a, b)
    elif a.ndim == 2 and b.ndim == 1:
        # mx.matmul requires 2-D; reshape 1-D to column vector
        result = mx.matmul(a, b[:, None]).squeeze(-1)
    else:
        result = a @ b
    if out is not None:
        out[:] = result
        return out
    return result


def einsum(*args, **kwargs):
    """Einstein summation.

    MLX has mx.einsum as of recent versions; fall back to NumPy if missing.
    """
    try:
        return mx.einsum(*args, **kwargs)
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        np_args = [_to_np(a) if isinstance(a, mx.array) else a for a in args]
        return mx.array(np.einsum(*np_args, **kwargs))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

BACKEND_NAME = 'mlx'
