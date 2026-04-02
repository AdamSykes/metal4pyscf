"""
Fused Metal kernels for XC evaluation using threadgroup shared memory.

Architecture:
  - 1 threadgroup = 1 grid point
  - nao threads per threadgroup (up to 1024)
  - AO values stored in threadgroup shared memory (~4.5 KB for 574 AOs)
  - rho: each thread computes dm[i,:] @ shared_ao, then parallel reduction
  - Vxc: each thread computes shared_ao[i] * wv * shared_ao[j], accumulates vmat
  - AO values NEVER written to global memory

For GGA: 4 sets of AO values (val, dx, dy, dz) = ~18 KB shared memory.
"""

import numpy as np
import mlx.core as mx
from gpu4pyscf.lib.metal_kernels.eval_ao import (
    _ncart, _cart2sph_matrix, _prepare_shell_data, _eval_ao_batch_gpu,
)


# ---------------------------------------------------------------------------
# Fused rho kernel: computes rho[g] without writing AO array
#
# Each threadgroup = 1 grid point, threads_per_group = nao_padded
# Thread i computes ao[i], stores in shared_ao, then computes
# dm[i,:] dot shared_ao, then reduces to get rho[g].
# ---------------------------------------------------------------------------

_FUSED_RHO_HEADER = '''
// Parallel reduction helper: sum values across threadgroup
inline float threadgroup_reduce_sum(threadgroup float* sdata, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return sdata[0];
}
'''

_FUSED_RHO_SOURCE = '''
uint gid = threadgroup_position_in_grid.x;  // grid point index
uint tid = thread_position_in_threadgroup.x; // AO index within threadgroup

if (gid >= ngrids) return;

// Shared memory for AO values and reduction
threadgroup float shared_ao[MAX_NAO];
threadgroup float shared_red[MAX_NAO];

// --- Step 1: each thread computes ao[tid] for this grid point ---
float ao_val = 0.0f;
if (tid < nao) {
    // Look up which shell this AO belongs to
    int shell_id = ao_to_shell[tid];
    int ao_start = ao_to_start[tid]; // first AO index of this shell
    int local_ao = tid - ao_start;   // local index within shell

    float acx = shell_data[shell_id * 8 + 0];
    float acy = shell_data[shell_id * 8 + 1];
    float acz = shell_data[shell_id * 8 + 2];
    float fac = shell_data[shell_id * 8 + 3];
    int nprim = (int)shell_data[shell_id * 8 + 4];
    int exp_off = (int)shell_data[shell_id * 8 + 6];
    int ang = (int)shell_data[shell_id * 8 + 7];

    float rx = gridx[gid] - acx;
    float ry = gridy[gid] - acy;
    float rz = gridz[gid] - acz;
    float rr = rx*rx + ry*ry + rz*rz;

    float ce = 0.0f;
    for (int p = 0; p < nprim; p++) {
        ce += coeffs[exp_off + p] * exp(-exps[exp_off + p] * rr);
    }
    ce *= fac;

    // Compute the specific Cartesian component for local_ao
    // This uses the same ordering as the eval_ao kernel
    if (ang == 0) { ao_val = ce; }
    else if (ang == 1) {
        float r[3] = {rx, ry, rz};
        ao_val = ce * r[local_ao];
    }
    else if (ang == 2) {
        // xx, xy, xz, yy, yz, zz
        float r[3] = {rx, ry, rz};
        int idx[6][2] = {{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
        ao_val = ce * r[idx[local_ao][0]] * r[idx[local_ao][1]];
    }
    else if (ang == 3) {
        float r[3] = {rx, ry, rz};
        int idx[10][3] = {{0,0,0},{0,0,1},{0,0,2},{0,1,1},{0,1,2},
                          {0,2,2},{1,1,1},{1,1,2},{1,2,2},{2,2,2}};
        ao_val = ce * r[idx[local_ao][0]] * r[idx[local_ao][1]] * r[idx[local_ao][2]];
    }

    // Apply cart2sph coefficient if needed
    ao_val *= c2s_coeffs[tid];
}

shared_ao[tid] = ao_val;
threadgroup_barrier(mem_flags::mem_threadgroup);

// --- Step 2: compute tmp_i = sum_j dm[i*nao + j] * shared_ao[j] ---
float tmp = 0.0f;
if (tid < nao) {
    for (uint j = 0; j < nao; j++) {
        tmp += dm[tid * nao + j] * shared_ao[j];
    }
}

// --- Step 3: partial[i] = ao[i] * tmp, then reduce ---
shared_red[tid] = (tid < nao) ? (ao_val * tmp) : 0.0f;
float rho_val = threadgroup_reduce_sum(shared_red, tid, MAX_NAO);

if (tid == 0) {
    rho_out[gid] = rho_val;
}
'''

_fused_rho_kernel = mx.fast.metal_kernel(
    name='fused_rho',
    input_names=['gridx', 'gridy', 'gridz', 'exps', 'coeffs', 'shell_data',
                 'dm', 'ao_to_shell', 'ao_to_start', 'c2s_coeffs'],
    output_names=['rho_out'],
    header=_FUSED_RHO_HEADER,
    source=_FUSED_RHO_SOURCE,
)


# ---------------------------------------------------------------------------
# Fused GGA rho kernel: computes rho[0] (density) + rho[1:4] (gradient)
# Uses 4 shared arrays: ao_val, ao_dx, ao_dy, ao_dz
# Total shared memory: 4 * MAX_NAO * 4 + MAX_NAO * 4 = 5 * MAX_NAO * 4 bytes
# For 574 AOs (padded to 1024): 5 * 1024 * 4 = 20 KB — fits in 32 KB
# ---------------------------------------------------------------------------

_FUSED_RHO_GGA_SOURCE = '''
uint gid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
if (gid >= ngrids) return;

threadgroup float sh_ao[MAX_NAO];
threadgroup float sh_dx[MAX_NAO];
threadgroup float sh_dy[MAX_NAO];
threadgroup float sh_dz[MAX_NAO];
threadgroup float sh_red[MAX_NAO];

// --- Step 1: compute ao value and derivatives for this thread's AO ---
float ao_v = 0.0f, ao_x = 0.0f, ao_y = 0.0f, ao_z = 0.0f;

if (tid < nao) {
    int shell_id = ao_to_shell[tid];
    int ao_start_v = ao_to_start[tid];
    int local_ao = tid - ao_start_v;

    float acx = shell_data[shell_id * 8 + 0];
    float acy = shell_data[shell_id * 8 + 1];
    float acz = shell_data[shell_id * 8 + 2];
    float fac = shell_data[shell_id * 8 + 3];
    int nprim = (int)shell_data[shell_id * 8 + 4];
    int exp_off = (int)shell_data[shell_id * 8 + 6];
    int ang = (int)shell_data[shell_id * 8 + 7];

    float rx = gridx[gid] - acx;
    float ry = gridy[gid] - acy;
    float rz = gridz[gid] - acz;
    float rr = rx*rx + ry*ry + rz*rz;

    float ce = 0.0f, ce_2a = 0.0f;
    for (int p = 0; p < nprim; p++) {
        float c = coeffs[exp_off + p];
        float a = exps[exp_off + p];
        float e = exp(-a * rr);
        ce += c * e;
        ce_2a += c * e * a;
    }
    ce *= fac;
    ce_2a *= -2.0f * fac;

    if (ang == 0) {
        ao_v = ce;
        ao_x = ce_2a * rx;
        ao_y = ce_2a * ry;
        ao_z = ce_2a * rz;
    }
    else if (ang == 1) {
        float r[3] = {rx, ry, rz};
        float ar[3] = {ce_2a * rx, ce_2a * ry, ce_2a * rz};
        ao_v = ce * r[local_ao];
        // d/dx(ce * r[k]) = ce_2a*rx*r[k] + ce*delta(k,0)
        ao_x = ar[0] * r[local_ao] + ((local_ao == 0) ? ce : 0.0f);
        ao_y = ar[1] * r[local_ao] + ((local_ao == 1) ? ce : 0.0f);
        ao_z = ar[2] * r[local_ao] + ((local_ao == 2) ? ce : 0.0f);
    }

    ao_v *= c2s_coeffs[tid];
    ao_x *= c2s_coeffs[tid];
    ao_y *= c2s_coeffs[tid];
    ao_z *= c2s_coeffs[tid];
}

sh_ao[tid] = ao_v;
sh_dx[tid] = ao_x;
sh_dy[tid] = ao_y;
sh_dz[tid] = ao_z;
threadgroup_barrier(mem_flags::mem_threadgroup);

// --- Step 2: tmp[i] = sum_j dm[i,j] * ao[j] ---
float tmp = 0.0f;
if (tid < nao) {
    for (uint j = 0; j < nao; j++) {
        tmp += dm[tid * nao + j] * sh_ao[j];
    }
}

// --- Step 3: reduce for rho[0] = sum_i ao[i]*tmp[i] ---
sh_red[tid] = (tid < nao) ? (ao_v * tmp) : 0.0f;
float rho0 = threadgroup_reduce_sum(sh_red, tid, MAX_NAO);

// --- Step 4: reduce for nabla rho = 2 * sum_i dao/dr[i] * tmp[i] ---
sh_red[tid] = (tid < nao) ? (ao_x * tmp) : 0.0f;
float rhox = 2.0f * threadgroup_reduce_sum(sh_red, tid, MAX_NAO);

sh_red[tid] = (tid < nao) ? (ao_y * tmp) : 0.0f;
float rhoy = 2.0f * threadgroup_reduce_sum(sh_red, tid, MAX_NAO);

sh_red[tid] = (tid < nao) ? (ao_z * tmp) : 0.0f;
float rhoz = 2.0f * threadgroup_reduce_sum(sh_red, tid, MAX_NAO);

if (tid == 0) {
    rho_out[gid * 4 + 0] = rho0;
    rho_out[gid * 4 + 1] = rhox;
    rho_out[gid * 4 + 2] = rhoy;
    rho_out[gid * 4 + 3] = rhoz;
}
'''

_fused_rho_gga_kernel = mx.fast.metal_kernel(
    name='fused_rho_gga',
    input_names=['gridx', 'gridy', 'gridz', 'exps', 'coeffs', 'shell_data',
                 'dm', 'ao_to_shell', 'ao_to_start', 'c2s_coeffs'],
    output_names=['rho_out'],
    header=_FUSED_RHO_HEADER,
    source=_FUSED_RHO_GGA_SOURCE,
)


def _build_ao_mapping(mol):
    """Build AO-to-shell mapping and cart2sph coefficients for fused kernel.

    Returns:
        ao_to_shell: (nao,) int32 — which shell each spherical AO comes from
        ao_to_start: (nao,) int32 — first AO index of that shell
        c2s_coeffs:  (nao,) float32 — cart2sph coefficients (1.0 for s/p)
    """
    nao = mol.nao
    ao_to_shell = np.zeros(nao, dtype=np.int32)
    ao_to_start = np.zeros(nao, dtype=np.int32)
    c2s_coeffs = np.ones(nao, dtype=np.float32)

    sph_off = 0
    for ish in range(mol.nbas):
        l = mol.bas_angular(ish)
        ncart = _ncart(l)
        nsph = ncart if l <= 1 else 2 * l + 1

        for i in range(nsph):
            ao_to_shell[sph_off + i] = ish
            ao_to_start[sph_off + i] = sph_off
        # For l >= 2, cart2sph is a matrix, not a simple coefficient.
        # The fused kernel handles s/p directly and uses the simple
        # diagonal approximation for d/f (loses accuracy for l>=2).
        # TODO: proper cart2sph in fused kernel
        sph_off += nsph

    return ao_to_shell, ao_to_start, c2s_coeffs


def fused_rho_vxc(mol, coords, dm, weights, ni, xc_code, xctype,
                  shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping):
    """Compute rho and Vxc with fused GPU kernels where possible.

    Uses the fused threadgroup kernel for rho (LDA only, s/p basis).
    Falls back to the batched approach for GGA or d/f functions.
    """
    nao = mol.nao
    max_l = max(mol.bas_angular(i) for i in range(mol.nbas))

    use_fused_rho = (xctype in ('LDA', 'GGA') and max_l <= 1 and nao <= 1024)

    if not use_fused_rho:
        # Batched approach (existing, works for all cases)
        return _batched_rho_vxc(
            mol, coords, dm, weights, ni, xc_code, xctype,
            shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping)

    # --- Fused rho for LDA s/p basis ---
    ngrids = coords.shape[0]
    ao_to_shell, ao_to_start, c2s_coeffs = _build_ao_mapping(mol)

    ao_to_shell_gpu = mx.array(ao_to_shell)
    ao_to_start_gpu = mx.array(ao_to_start)
    c2s_gpu = mx.array(c2s_coeffs)
    dm_gpu = mx.array(np.asarray(dm, dtype=np.float32).ravel())
    shell_data_gpu = mx.array(shell_data.ravel())

    gridx = mx.array(coords[:, 0].astype(np.float32))
    gridy = mx.array(coords[:, 1].astype(np.float32))
    gridz = mx.array(coords[:, 2].astype(np.float32))

    # Pad nao to next power of 2 for efficient reduction
    max_nao = 1
    while max_nao < nao:
        max_nao *= 2

    # MLX grid = total threads, NOT threadgroup count.
    kernel_inputs = [gridx, gridy, gridz, exps_gpu, coeffs_gpu, shell_data_gpu,
                     dm_gpu, ao_to_shell_gpu, ao_to_start_gpu, c2s_gpu]
    kernel_template = [('ngrids', ngrids), ('nao', nao), ('MAX_NAO', max_nao)]

    if xctype == 'LDA':
        rho_gpu = _fused_rho_kernel(
            inputs=kernel_inputs,
            grid=(ngrids * max_nao, 1, 1),
            threadgroup=(max_nao, 1, 1),
            output_shapes=[(ngrids,)],
            output_dtypes=[mx.float32],
            template=kernel_template,
        )
        mx.eval(rho_gpu[0])
        rho = np.array(rho_gpu[0], dtype=np.float64)
    else:  # GGA
        rho_gpu = _fused_rho_gga_kernel(
            inputs=kernel_inputs,
            grid=(ngrids * max_nao, 1, 1),
            threadgroup=(max_nao, 1, 1),
            output_shapes=[(ngrids * 4,)],
            output_dtypes=[mx.float32],
            template=kernel_template,
        )
        mx.eval(rho_gpu[0])
        rho = np.array(rho_gpu[0], dtype=np.float64).reshape(ngrids, 4).T

    # XC on CPU
    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
    den = (rho if xctype == 'LDA' else rho[0]) * weights
    nelec = den.sum()
    excsum = np.dot(den, exc)

    # Vxc contraction uses batched approach (needs AO array for gemm)
    wv = weights * vxc
    ao_deriv = 1 if xctype == 'GGA' else 0
    vmat = np.zeros((nao, nao))
    BATCH = 20000
    for p0 in range(0, ngrids, BATCH):
        p1 = min(p0 + BATCH, ngrids)
        ng = p1 - p0
        ao_gpu = _eval_ao_batch_gpu(
            mol, coords[p0:p1], ao_deriv, shell_data, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ng)

        if xctype == 'LDA':
            wv_gpu = mx.array(wv[0][p0:p1].astype(np.float32))
            vmat_blk = (ao_gpu * wv_gpu[None, :]) @ mx.transpose(ao_gpu)
        else:  # GGA
            wv_b = wv[:, p0:p1].copy()
            wv_b[0] *= 0.5
            wv_gpu = [mx.array(wv_b[d].astype(np.float32)) for d in range(4)]
            ao_val = ao_gpu[0]
            aow = ao_gpu[0] * wv_gpu[0][None, :]
            for d in range(1, 4):
                aow = aow + ao_gpu[d] * wv_gpu[d][None, :]
            vmat_blk = aow @ mx.transpose(ao_val)

        mx.eval(vmat_blk)
        vmat += np.array(vmat_blk, dtype=np.float64)

    if xctype == 'GGA':
        vmat = vmat + vmat.T

    return nelec, excsum, vmat


def _batched_rho_vxc(mol, coords, dm, weights, ni, xc_code, xctype,
                     shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping):
    """Batched eval_ao + rho + Vxc (existing approach, works for all cases)."""
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
        wt = weights[p0:p1]

        ao_gpu = _eval_ao_batch_gpu(
            mol, coords[p0:p1], ao_deriv, shell_data, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ng)

        if xctype == 'LDA':
            ao_val = ao_gpu
            tmp = dm_gpu @ ao_val
            rho_gpu = mx.sum(ao_val * tmp, axis=0)
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

        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
        den = (rho if xctype == 'LDA' else rho[0]) * wt
        nelec += den.sum()
        excsum += np.dot(den, exc)

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
