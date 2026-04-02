"""
Fully Metal GPU-native XC numerical integration for DFT.

Replaces PySCF's CPU block_loop entirely. All steps on Metal GPU:
  1. eval_ao: Metal compute shader (our kernel)
  2. rho: GPU matmul  rho[g] = sum_ij ao[g,i] * dm[i,j] * ao[g,j]
  3. XC: CPU libxc (fast, ~25ms — not worth porting)
  4. Vxc contraction: GPU matmul  vmat += (ao * wv).T @ ao

Grid points processed in batches to control memory.
AO array stays on GPU — never copied to CPU except for rho→libxc→wv.
"""

import numpy as np
import mlx.core as mx
from gpu4pyscf.lib.metal_kernels.eval_ao import (
    eval_ao_metal, _prepare_shell_data, _eval_ao_batch_gpu,
)


def nr_rks_metal(ni, mol, grids, xc_code, dm, hermi=1, max_memory=2000):
    """Fully Metal GPU-accelerated nr_rks.

    All heavy computation on Metal GPU. Only libxc runs on CPU.
    """
    xctype = ni._xc_type(xc_code)
    nao = mol.nao
    ngrids_total = grids.coords.shape[0]

    nelec = 0.0
    excsum = 0.0
    vmat = np.zeros((nao, nao))

    dm_f32 = np.asarray(dm, dtype=np.float32)
    dm_gpu = mx.array(dm_f32)

    ao_deriv = {'LDA': 0, 'GGA': 1, 'MGGA': 1, 'HF': 0}.get(xctype, 1)

    # Precompute shell data for Metal eval_ao
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping = \
        _prepare_shell_data(mol)

    # Batch size: ~20K grid points is optimal for GPU cache utilization.
    # Smaller batches avoid large output allocations (4*ncart*batch*4 bytes)
    # that exceed the GPU's effective cache. 20K gives ~50-200 MB per batch.
    batch_size = 20000

    coords_np = np.asarray(grids.coords)
    weights_np = np.asarray(grids.weights)

    for p0 in range(0, ngrids_total, batch_size):
        p1 = min(p0 + batch_size, ngrids_total)
        ngrids_batch = p1 - p0
        coords_batch = coords_np[p0:p1]
        weights_batch = weights_np[p0:p1]

        # --- Step 1: eval_ao on Metal GPU ---
        ao_gpu = _eval_ao_batch_gpu(
            mol, coords_batch, ao_deriv,
            shell_data, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ngrids_batch)
        # ao_gpu: (nao, ngrids) for LDA or (4, nao, ngrids) for GGA
        # Data stays on GPU

        # --- Step 2: rho on Metal GPU ---
        if xctype == 'LDA':
            ao_val = ao_gpu  # (nao, ngrids)
            # rho[g] = sum_ij ao[i,g] * dm[i,j] * ao[j,g]
            # = sum_i ao[i,g] * (dm @ ao)[i,g]
            tmp = dm_gpu @ ao_val          # (nao, ngrids) — gemm
            rho_gpu = mx.sum(ao_val * tmp, axis=0)  # (ngrids,)
            mx.eval(rho_gpu)
            rho = np.array(rho_gpu, dtype=np.float64)
        else:
            ao_val = ao_gpu[0]  # (nao, ngrids)
            ao_dx = ao_gpu[1]
            ao_dy = ao_gpu[2]
            ao_dz = ao_gpu[3]
            # rho[0] = density, rho[1:4] = gradient
            tmp = dm_gpu @ ao_val
            rho0_gpu = mx.sum(ao_val * tmp, axis=0)
            # nabla rho: 2 * sum_i dao/dx[i,g] * (dm @ ao)[i,g]
            rhox_gpu = 2.0 * mx.sum(ao_dx * tmp, axis=0)
            rhoy_gpu = 2.0 * mx.sum(ao_dy * tmp, axis=0)
            rhoz_gpu = 2.0 * mx.sum(ao_dz * tmp, axis=0)
            mx.eval(rho0_gpu, rhox_gpu, rhoy_gpu, rhoz_gpu)
            rho = np.zeros((4, ngrids_batch), dtype=np.float64)
            rho[0] = np.array(rho0_gpu)
            rho[1] = np.array(rhox_gpu)
            rho[2] = np.array(rhoy_gpu)
            rho[3] = np.array(rhoz_gpu)

        # --- Step 3: XC on CPU (libxc, fast ~25ms) ---
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]

        # Accumulate nelec and exc
        if xctype == 'LDA':
            den = rho * weights_batch
        else:
            den = rho[0] * weights_batch
        nelec += den.sum()
        excsum += np.dot(den, exc)

        # --- Step 4: Vxc contraction on Metal GPU ---
        wv = weights_batch * vxc  # weighted XC potential

        if xctype == 'LDA':
            wv_gpu = mx.array(wv[0].astype(np.float32))
            # vmat += ao @ diag(wv) @ ao.T = (ao * wv) @ ao.T
            vmat_blk = (ao_val * wv_gpu[None, :]) @ mx.transpose(ao_val)
            mx.eval(vmat_blk)
            vmat += np.array(vmat_blk, dtype=np.float64)

        elif xctype in ('GGA', 'MGGA'):
            wv[0] *= 0.5  # for symmetrization
            wv_gpu = [mx.array(wv[d].astype(np.float32)) for d in range(len(wv))]
            # aow = sum_d ao[d] * wv[d]
            aow = ao_gpu[0] * wv_gpu[0][None, :]
            for d in range(1, min(4, len(wv_gpu))):
                aow = aow + ao_gpu[d] * wv_gpu[d][None, :]
            vmat_blk = aow @ mx.transpose(ao_val)

            if xctype == 'MGGA' and len(wv_gpu) > 4:
                wv_gpu[4] = wv_gpu[4] * 0.5  # tau factor
                for d in range(1, 4):
                    vmat_blk = vmat_blk + (ao_gpu[d] * wv_gpu[4][None, :]) @ mx.transpose(ao_gpu[d])

            mx.eval(vmat_blk)
            vmat += np.array(vmat_blk, dtype=np.float64)

    # Symmetrize for GGA/MGGA
    if xctype in ('GGA', 'MGGA'):
        vmat = vmat + vmat.T

    return nelec, excsum, vmat
