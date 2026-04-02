"""
Metal compute shader implementations of gpu4pyscf helper kernels.

These are direct translations of the CUDA kernels in lib/cupy_helper/*.cu
to Metal Shading Language (MSL), loaded via MLX's custom kernel API.

Note: Apple Silicon GPUs do NOT support IEEE float64 in hardware.
All Metal kernels operate in float32. For quantum chemistry applications,
a mixed-precision strategy is used:
  - float32 on Metal GPU for O(N^3)/O(N^4) operations (contractions, J/K build)
  - float64 on CPU for precision-critical operations (eigensolve, energy)
"""

import mlx.core as mx
import numpy as np

__all__ = [
    'dist_matrix',
    'pack_tril',
    'unpack_tril',
    'fill_triu',
]

# ---------------------------------------------------------------------------
# dist_matrix: Euclidean distance matrix between two point clouds
# Translated from: cupy_helper/dist_matrix.cu
# ---------------------------------------------------------------------------

_dist_matrix_source = '''
uint i = thread_position_in_grid.x;
uint j = thread_position_in_grid.y;
if (i >= m || j >= n) return;

float dx = x[3*i]   - y[3*j];
float dy = x[3*i+1] - y[3*j+1];
float dz = x[3*i+2] - y[3*j+2];
out[i*n+j] = sqrt(dx*dx + dy*dy + dz*dz);
'''

_dist_matrix_kernel = mx.fast.metal_kernel(
    name='dist_matrix',
    input_names=['x', 'y'],
    output_names=['out'],
    source=_dist_matrix_source,
)


def dist_matrix(x, y):
    """Compute pairwise Euclidean distance matrix on Metal GPU.

    Args:
        x: (m, 3) array of 3D coordinates
        y: (n, 3) array of 3D coordinates

    Returns:
        (m, n) distance matrix where out[i,j] = ||x[i] - y[j]||
    """
    x = mx.array(np.asarray(x, dtype=np.float32)).reshape(-1)
    y = mx.array(np.asarray(y, dtype=np.float32)).reshape(-1)
    m = x.size // 3
    n = y.size // 3

    THREADS = 32
    grid = (((m + THREADS - 1) // THREADS) * THREADS,
            ((n + THREADS - 1) // THREADS) * THREADS, 1)
    threadgroup = (THREADS, THREADS, 1)

    out = _dist_matrix_kernel(
        inputs=[x, y],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(m * n,)],
        output_dtypes=[mx.float32],
        template=[('m', m), ('n', n)],
    )
    return out[0].reshape(m, n)


# ---------------------------------------------------------------------------
# pack_tril: Pack lower triangular matrix into compact storage
# Translated from: cupy_helper/unpack.cu (_pack_tril)
# ---------------------------------------------------------------------------

_pack_tril_source = '''
uint j = thread_position_in_grid.x;
uint i = thread_position_in_grid.y;
if (i >= n || j >= n || i < j) return;

uint stride = ((n + 1) * n) / 2;
uint ptr = i*(i+1)/2 + j;
uint nao2 = n * n;
for (uint p = 0; p < counts; ++p) {
    out[ptr + p*stride] = a[p*nao2 + i*n + j];
}
'''

_pack_tril_kernel = mx.fast.metal_kernel(
    name='pack_tril',
    input_names=['a'],
    output_names=['out'],
    source=_pack_tril_source,
)


def pack_tril(a):
    """Pack lower triangular part of symmetric matrices into compact storage.

    Args:
        a: (counts, n, n) or (n, n) symmetric matrix/matrices

    Returns:
        (counts, n*(n+1)/2) or (n*(n+1)/2,) packed lower triangle
    """
    a = mx.array(np.asarray(a, dtype=np.float32))
    squeeze = a.ndim == 2
    if squeeze:
        a = a.reshape(1, a.shape[0], a.shape[1])
    counts, n, _ = a.shape
    tril_size = n * (n + 1) // 2

    THREADS = 16
    grid = (((n + THREADS - 1) // THREADS) * THREADS,
            ((n + THREADS - 1) // THREADS) * THREADS, 1)
    threadgroup = (THREADS, THREADS, 1)

    out = _pack_tril_kernel(
        inputs=[a.reshape(-1)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(counts * tril_size,)],
        output_dtypes=[mx.float32],
        template=[('n', n), ('counts', counts)],
    )
    result = out[0].reshape(counts, tril_size)
    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# unpack_tril: Unpack compact lower triangular back to full matrix
# Translated from: cupy_helper/unpack.cu (_unpack_tril + _dfill_triu)
# ---------------------------------------------------------------------------

_unpack_tril_source = '''
uint j = thread_position_in_grid.x;
uint i = thread_position_in_grid.y;
if (i >= nao || j >= nao) return;

uint stride = ((nao + 1) * nao) / 2;
uint nao2 = nao * nao;

// Read from tril for both triangles (no read-after-write hazard)
uint row = (i >= j) ? i : j;  // max(i,j)
uint col = (i >= j) ? j : i;  // min(i,j)
uint ptr = row*(row+1)/2 + col;

for (uint p = 0; p < counts; ++p) {
    float val = tril[ptr + p*stride];
    if (hermi == 2 && i < j) val = -val;
    out[p*nao2 + i*nao + j] = val;
}
'''

_unpack_tril_kernel = mx.fast.metal_kernel(
    name='unpack_tril',
    input_names=['tril'],
    output_names=['out'],
    source=_unpack_tril_source,
)


def unpack_tril(tril, hermi=1):
    """Unpack compact lower triangular to full symmetric matrix on Metal GPU.

    Args:
        tril: (counts, n*(n+1)/2) or (n*(n+1)/2,) packed lower triangle.
              Can be numpy array or mx.array.
        hermi: 1 for symmetric, 2 for anti-symmetric

    Returns:
        (counts, n, n) or (n, n) full matrix (mx.array)
    """
    if not isinstance(tril, mx.array):
        tril = mx.array(np.asarray(tril, dtype=np.float32))
    squeeze = tril.ndim == 1
    if squeeze:
        tril = tril.reshape(1, -1)
    counts = tril.shape[0]
    tril_size = tril.shape[1]
    nao = int((2 * tril_size) ** 0.5)
    assert nao * (nao + 1) // 2 == tril_size

    THREADS = 16
    grid = (((nao + THREADS - 1) // THREADS) * THREADS,
            ((nao + THREADS - 1) // THREADS) * THREADS, 1)
    threadgroup = (THREADS, THREADS, 1)
    # so we do this in a single pass with a barrier-like approach.
    # Actually, the lower triangle writes don't overlap with upper triangle reads,
    # so we need two passes.
    out = _unpack_tril_kernel(
        inputs=[tril.reshape(-1)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(counts * nao * nao,)],
        output_dtypes=[mx.float32],
        template=[('nao', nao), ('counts', counts), ('hermi', hermi)],
    )
    result = out[0].reshape(counts, nao, nao)
    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# fill_triu: Fill upper triangle from lower triangle (in-place symmetrize)
# Translated from: cupy_helper/unpack.cu (_dfill_triu)
# ---------------------------------------------------------------------------

_fill_triu_source = '''
uint j = thread_position_in_grid.x;
uint i = thread_position_in_grid.y;
if (i >= n || j >= n || i >= j) return;

uint n2 = n * n;
for (uint p = 0; p < counts; ++p) {
    uint off = p * n2;
    if (hermi == 1) {
        out[off + i*n + j] = a[off + j*n + i];
    } else {
        out[off + i*n + j] = -a[off + j*n + i];
    }
}
// Copy diagonal and lower triangle unchanged
if (i <= j) {
    // Already there from input
}
'''

# For fill_triu, we need a kernel that copies lower+diag and mirrors to upper
_fill_triu_full_source = '''
uint j = thread_position_in_grid.x;
uint i = thread_position_in_grid.y;
if (i >= n || j >= n) return;

uint n2 = n * n;
for (uint p = 0; p < counts; ++p) {
    uint off = p * n2;
    if (i >= j) {
        // Lower triangle + diagonal: copy as-is
        out[off + i*n + j] = a[off + i*n + j];
    } else {
        // Upper triangle: mirror from lower
        if (hermi == 1) {
            out[off + i*n + j] = a[off + j*n + i];
        } else {
            out[off + i*n + j] = -a[off + j*n + i];
        }
    }
}
'''

_fill_triu_kernel = mx.fast.metal_kernel(
    name='fill_triu',
    input_names=['a'],
    output_names=['out'],
    source=_fill_triu_full_source,
)


def fill_triu(a, hermi=1):
    """Fill upper triangle from lower triangle on Metal GPU.

    Args:
        a: (counts, n, n) or (n, n) matrix with lower triangle populated
        hermi: 1 for symmetric, 2 for anti-symmetric

    Returns:
        Full symmetric/anti-symmetric matrix
    """
    a = mx.array(np.asarray(a, dtype=np.float32))
    squeeze = a.ndim == 2
    if squeeze:
        a = a.reshape(1, a.shape[0], a.shape[1])
    counts, n, _ = a.shape

    THREADS = 16
    grid = (((n + THREADS - 1) // THREADS) * THREADS,
            ((n + THREADS - 1) // THREADS) * THREADS, 1)
    threadgroup = (THREADS, THREADS, 1)

    out = _fill_triu_kernel(
        inputs=[a.reshape(-1)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(counts * n * n,)],
        output_dtypes=[mx.float32],
        template=[('n', n), ('counts', counts), ('hermi', hermi)],
    )
    result = out[0].reshape(counts, n, n)
    if squeeze:
        result = result.squeeze(0)
    return result
