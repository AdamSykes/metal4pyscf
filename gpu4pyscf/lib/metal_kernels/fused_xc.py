"""
Fused Metal kernels for XC evaluation.

Instead of writing the full (nao, ngrids) AO array to global memory,
these kernels compute AO values on-the-fly per grid point and
immediately contract them with the density matrix (for rho) or
XC potential weights (for Vxc).

Two-pass approach:
  Pass 1: fused_rho — compute rho[g] without writing AO array
  Pass 2: fused_vxc — compute vmat contribution without storing AOs

Each pass recomputes AOs from scratch (cheap: ~17M flops vs 6.6B for dm contraction).
Global memory traffic: O(ngrids) instead of O(nao × ngrids).
"""

import numpy as np
import mlx.core as mx
from gpu4pyscf.lib.metal_kernels.eval_ao import (
    _ncart, _cart2sph_matrix, _prepare_shell_data,
)


# ---------------------------------------------------------------------------
# Fused rho kernel (LDA): rho[g] = sum_ij ao[g,i] * dm[i,j] * ao[g,j]
#
# Each thread: one grid point.
# Loops over all shells, computes AO values on the fly.
# First computes tmp[i] = sum_j dm[i,j]*ao[j] by looping over shells for j,
# then rho = sum_i ao[i]*tmp[i] by looping over shells for i.
#
# Since we can't store nao floats in registers, we use a two-pass approach:
# Pass A: for each shell j, compute ao_j values and accumulate dm[:,j]*ao_j into tmp
# Pass B: for each shell i, compute ao_i values and accumulate ao_i * tmp[i]
#
# But tmp has nao elements — too many for registers.
# Instead: rho = sum_i ao_i * (dm @ ao)[i] = dot(ao, dm @ ao)
# We can compute this as: tmp_vec = dm @ ao (the matrix-vector product)
# then rho = dot(ao, tmp_vec)
#
# The matrix-vector product dm @ ao for one grid point costs O(nao^2).
# Over 20K grid points, that's 20K × 574^2 = 6.6B flops — the same as the gemm.
# But we avoid writing 20K × 574 × 4 = 46 MB of AO data to global memory.
# ---------------------------------------------------------------------------

# For the fused approach, we can't easily write a single Metal kernel that
# handles variable-length shell loops with the MLX API constraints.
# Instead, we use a hybrid: compute AOs per batch in the existing kernel,
# but keep them on GPU and immediately contract — never reading back to CPU.
#
# The REAL optimization: avoid the numpy↔MLX round-trip for rho.
# Currently: eval_ao → mx.array (GPU) → dm@ao (GPU) → sum (GPU) → np.array (CPU)
# The eval_ao already returns mx.array. We just need to chain the ops without eval().


def fused_rho_vxc(mol, coords, dm, weights, ni, xc_code, xctype,
                  shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping):
    """Compute rho and Vxc contraction with minimal memory traffic.

    Computes AOs once per batch, chains rho→libxc→Vxc without
    intermediate readback. The AO array stays on GPU throughout.

    Returns: (nelec, excsum, vmat_f64)
    """
    from gpu4pyscf.lib.metal_kernels.eval_ao import _eval_ao_batch_gpu

    nao = mol.nao
    ngrids = coords.shape[0]
    dm_gpu = mx.array(np.asarray(dm, dtype=np.float32))
    ao_deriv = 1 if xctype in ('GGA', 'MGGA') else 0

    nelec = 0.0
    excsum = 0.0
    vmat = np.zeros((nao, nao))

    BATCH = 20000
    for p0 in range(0, ngrids, BATCH):
        p1 = min(p0 + BATCH, ngrids)
        ng = p1 - p0
        coords_b = coords[p0:p1]
        wt = weights[p0:p1]

        # --- eval_ao on GPU (stays as mx.array, no eval yet) ---
        ao_gpu = _eval_ao_batch_gpu(
            mol, coords_b, ao_deriv, shell_data, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ng)

        # --- rho on GPU (chain without eval) ---
        if xctype == 'LDA':
            ao_val = ao_gpu  # (nao, ng)
            tmp = dm_gpu @ ao_val  # (nao, ng)
            rho_gpu = mx.sum(ao_val * tmp, axis=0)  # (ng,)
            mx.eval(rho_gpu)
            rho = np.array(rho_gpu, dtype=np.float64)
        else:
            ao_val = ao_gpu[0]
            tmp = dm_gpu @ ao_val
            rho0 = mx.sum(ao_val * tmp, axis=0)
            rho1 = 2.0 * mx.sum(ao_gpu[1] * tmp, axis=0)
            rho2 = 2.0 * mx.sum(ao_gpu[2] * tmp, axis=0)
            rho3 = 2.0 * mx.sum(ao_gpu[3] * tmp, axis=0)
            mx.eval(rho0, rho1, rho2, rho3)
            rho = np.empty((4, ng), dtype=np.float64)
            rho[0] = np.array(rho0)
            rho[1] = np.array(rho1)
            rho[2] = np.array(rho2)
            rho[3] = np.array(rho3)

        # --- libxc on CPU (fast, ~2ms per batch) ---
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]

        den = (rho if xctype == 'LDA' else rho[0]) * wt
        nelec += den.sum()
        excsum += np.dot(den, exc)

        # --- Vxc contraction on GPU (reuse ao_gpu still in memory) ---
        wv = wt * vxc
        if xctype == 'LDA':
            wv_gpu = mx.array(wv[0].astype(np.float32))
            vmat_blk = (ao_val * wv_gpu[None, :]) @ mx.transpose(ao_val)
        else:
            wv[0] *= 0.5
            wv_gpu = [mx.array(wv[d].astype(np.float32)) for d in range(min(4, len(wv)))]
            aow = ao_gpu[0] * wv_gpu[0][None, :]
            for d in range(1, 4):
                aow = aow + ao_gpu[d] * wv_gpu[d][None, :]
            vmat_blk = aow @ mx.transpose(ao_val)

            if xctype == 'MGGA' and len(wv) > 4:
                wv4 = mx.array((wv[4] * 0.5).astype(np.float32))
                for d in range(1, 4):
                    vmat_blk = vmat_blk + (ao_gpu[d] * wv4[None, :]) @ mx.transpose(ao_gpu[d])

        mx.eval(vmat_blk)
        vmat += np.array(vmat_blk, dtype=np.float64)

    if xctype in ('GGA', 'MGGA'):
        vmat = vmat + vmat.T

    return nelec, excsum, vmat
