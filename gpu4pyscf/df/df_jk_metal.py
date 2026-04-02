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
Metal GPU-accelerated density-fitting J/K matrix construction.

Mixed-precision: f32 on Metal GPU, f64 accumulation on CPU.

Architecture:
  - CDERI precomputed once by PySCF on CPU, unpacked and stored on GPU as f32
  - J: packed-space gemvs (no unpack per cycle)
  - K: half-transform Y=cderi@L_occ then K=Y_t@Y_t.T (single GPU gemm)
  - Per-cycle cost is pure GPU BLAS — no CPU unpack in the hot path
"""

import numpy as np
import mlx.core as mx
from pyscf import lib as pyscf_lib
from gpu4pyscf.lib import logger


class MetalDFTensors:
    """Precomputed density-fitting tensors resident on Metal GPU.

    Built once from a PySCF DF object. Reused across SCF cycles.
    """

    def __init__(self, dfobj):
        nao = dfobj.mol.nao
        npairs = nao * (nao + 1) // 2

        # Collect all CDERI blocks
        tril_blocks = [np.asarray(b, dtype=np.float32) for b in dfobj.loop()]
        cderi_tril_np = np.concatenate(tril_blocks, axis=0)
        naux = cderi_tril_np.shape[0]

        # Store packed form on GPU for J
        self.cderi_tril_gpu = mx.array(cderi_tril_np)

        # Unpack to full form on CPU (once), store on GPU for K
        cderi_full_np = pyscf_lib.unpack_tril(cderi_tril_np)
        self.cderi_full_gpu = mx.array(cderi_full_np)

        self.nao = nao
        self.naux = naux
        self.npairs = npairs


# Cache per DF object so we don't rebuild every call
_tensor_cache = {}


def _get_tensors(dfobj):
    key = id(dfobj)
    if key not in _tensor_cache:
        _tensor_cache[key] = MetalDFTensors(dfobj)
    return _tensor_cache[key]


def clear_cache(dfobj=None):
    """Clear cached GPU tensors."""
    if dfobj is None:
        _tensor_cache.clear()
    else:
        _tensor_cache.pop(id(dfobj), None)


def _pack_dm(dm, nao):
    idx = np.tril_indices(nao)
    dm_packed = dm[idx].copy()
    off_diag = idx[0] != idx[1]
    dm_packed[off_diag] *= 2.0
    return dm_packed


def _get_occ_coeff(dm):
    import scipy.linalg
    w, v = scipy.linalg.eigh(dm)
    mask = w > 1e-10
    return v[:, mask] * np.sqrt(w[mask])


def get_jk_metal(dfobj, dm, hermi=1, with_j=True, with_k=True):
    """Compute DF J/K on Metal GPU.

    First call builds and caches GPU tensors. Subsequent calls reuse them.
    """
    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t0 = log.init_timer()

    tensors = _get_tensors(dfobj)
    nao = tensors.nao
    naux = tensors.naux

    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0]

    vj = vk = None

    if with_j:
        dm_packed_gpu = mx.array(_pack_dm(dm, nao).astype(np.float32))
        # J in packed space: two gemvs, pure GPU
        rho_q = tensors.cderi_tril_gpu @ dm_packed_gpu          # (naux,)
        vj_packed = mx.transpose(tensors.cderi_tril_gpu) @ rho_q  # (npairs,)
        mx.eval(vj_packed)
        vj_f64 = np.array(vj_packed).astype(np.float64)
        # Unpack to full
        vj = np.zeros((nao, nao))
        idx = np.tril_indices(nao)
        vj[idx] = vj_f64
        vj = vj + vj.T
        np.fill_diagonal(vj, np.diag(vj) * 0.5)

    if with_k:
        # Half-transform: Y[Q,i,a] = cderi[Q,i,j] @ L[j,a]
        L = _get_occ_coeff(dm)
        L_gpu = mx.array(L.astype(np.float32))
        nocc = L.shape[1]

        # cderi_full_gpu is (naux, nao, nao), already on GPU
        Y = tensors.cderi_full_gpu.reshape(naux * nao, nao) @ L_gpu  # (naux*nao, nocc)

        # K = sum_{Q,a} Y[Q,i,a]*Y[Q,j,a]
        Y_t = mx.transpose(Y.reshape(naux, nao, nocc), axes=(1, 0, 2)).reshape(nao, -1)
        vk_gpu = Y_t @ mx.transpose(Y_t)  # (nao, nao)
        mx.eval(vk_gpu)

        vk = np.array(vk_gpu).astype(np.float64)
        if hermi:
            vk = (vk + vk.T) * 0.5

    t0 = log.timer_debug1('DF-JK Metal GPU', *t0)
    return vj, vk
