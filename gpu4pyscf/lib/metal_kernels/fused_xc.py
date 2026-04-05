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
uint gid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
if (gid >= ngrids) return;

threadgroup float shared_ao[MAX_NAO];
threadgroup float shared_red[MAX_NAO];

// --- Step 1: compute spherical AO for this thread ---
// Each thread computes ALL Cartesian components for its shell,
// then applies the c2s row to get its spherical AO value.
float ao_val = 0.0f;
if (tid < nao) {
    int shell_id = ao_to_shell[tid];
    int ncart = c2s_ncart[tid];
    int c2s_off = c2s_offset[tid];

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

    // Compute Cartesian components and apply c2s row
    float cart[15]; // max 15 for l=4
    if (ang == 0) { cart[0] = ce; }
    else if (ang == 1) { cart[0]=ce*rx; cart[1]=ce*ry; cart[2]=ce*rz; }
    else if (ang == 2) {
        cart[0]=ce*rx*rx; cart[1]=ce*rx*ry; cart[2]=ce*rx*rz;
        cart[3]=ce*ry*ry; cart[4]=ce*ry*rz; cart[5]=ce*rz*rz;
    }
    else if (ang == 3) {
        cart[0]=ce*rx*rx*rx; cart[1]=ce*rx*rx*ry; cart[2]=ce*rx*rx*rz;
        cart[3]=ce*rx*ry*ry; cart[4]=ce*rx*ry*rz; cart[5]=ce*rx*rz*rz;
        cart[6]=ce*ry*ry*ry; cart[7]=ce*ry*ry*rz; cart[8]=ce*ry*rz*rz;
        cart[9]=ce*rz*rz*rz;
    }

    ao_val = 0.0f;
    for (int c = 0; c < ncart; c++) {
        ao_val += c2s_data[c2s_off + c] * cart[c];
    }
}

shared_ao[tid] = ao_val;
threadgroup_barrier(mem_flags::mem_threadgroup);

// --- Step 2: tmp[i] = dm[i,:] @ shared_ao ---
float tmp = 0.0f;
if (tid < nao) {
    for (uint j = 0; j < nao; j++) {
        tmp += dm[tid * nao + j] * shared_ao[j];
    }
}

// --- Step 3: reduce rho = sum_i ao[i]*tmp[i] ---
shared_red[tid] = (tid < nao) ? (ao_val * tmp) : 0.0f;
float rho_val = threadgroup_reduce_sum(shared_red, tid, MAX_NAO);

if (tid == 0) {
    rho_out[gid] = rho_val;
}
'''

_fused_rho_kernel = mx.fast.metal_kernel(
    name='fused_rho',
    input_names=['gridx', 'gridy', 'gridz', 'exps', 'coeffs', 'shell_data',
                 'dm', 'ao_to_shell', 'c2s_data', 'c2s_offset', 'c2s_ncart'],
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

    // Compute all Cartesian components + derivatives, apply c2s
    float cart_v[15], cart_x[15], cart_y[15], cart_z[15];
    float ax = ce_2a*rx, ay = ce_2a*ry, az = ce_2a*rz;

    if (ang == 0) {
        cart_v[0]=ce; cart_x[0]=ax; cart_y[0]=ay; cart_z[0]=az;
    }
    else if (ang == 1) {
        cart_v[0]=ce*rx;         cart_v[1]=ce*ry;         cart_v[2]=ce*rz;
        cart_x[0]=ax*rx+ce;      cart_x[1]=ax*ry;         cart_x[2]=ax*rz;
        cart_y[0]=ay*rx;         cart_y[1]=ay*ry+ce;       cart_y[2]=ay*rz;
        cart_z[0]=az*rx;         cart_z[1]=az*ry;          cart_z[2]=az*rz+ce;
    }
    else if (ang == 2) {
        cart_v[0]=ce*rx*rx;      cart_v[1]=ce*rx*ry;       cart_v[2]=ce*rx*rz;
        cart_v[3]=ce*ry*ry;      cart_v[4]=ce*ry*rz;       cart_v[5]=ce*rz*rz;
        cart_x[0]=(ax*rx+2*ce)*rx; cart_x[1]=(ax*rx+ce)*ry;  cart_x[2]=(ax*rx+ce)*rz;
        cart_x[3]=ax*ry*ry;       cart_x[4]=ax*ry*rz;       cart_x[5]=ax*rz*rz;
        cart_y[0]=ay*rx*rx;        cart_y[1]=(ay*ry+ce)*rx;   cart_y[2]=ay*rx*rz;
        cart_y[3]=(ay*ry+2*ce)*ry; cart_y[4]=(ay*ry+ce)*rz;   cart_y[5]=ay*rz*rz;
        cart_z[0]=az*rx*rx;        cart_z[1]=az*rx*ry;         cart_z[2]=(az*rz+ce)*rx;
        cart_z[3]=az*ry*ry;        cart_z[4]=(az*rz+ce)*ry;    cart_z[5]=(az*rz+2*ce)*rz;
    }
    else if (ang == 3) {
        cart_v[0]=ce*rx*rx*rx; cart_v[1]=ce*rx*rx*ry; cart_v[2]=ce*rx*rx*rz;
        cart_v[3]=ce*rx*ry*ry; cart_v[4]=ce*rx*ry*rz; cart_v[5]=ce*rx*rz*rz;
        cart_v[6]=ce*ry*ry*ry; cart_v[7]=ce*ry*ry*rz; cart_v[8]=ce*ry*rz*rz;
        cart_v[9]=ce*rz*rz*rz;
        cart_x[0]=(ax*rx+3*ce)*rx*rx; cart_x[1]=(ax*rx+2*ce)*rx*ry; cart_x[2]=(ax*rx+2*ce)*rx*rz;
        cart_x[3]=(ax*rx+ce)*ry*ry;   cart_x[4]=(ax*rx+ce)*ry*rz;   cart_x[5]=(ax*rx+ce)*rz*rz;
        cart_x[6]=ax*ry*ry*ry;        cart_x[7]=ax*ry*ry*rz;        cart_x[8]=ax*ry*rz*rz;
        cart_x[9]=ax*rz*rz*rz;
        cart_y[0]=ay*rx*rx*rx;        cart_y[1]=(ay*ry+ce)*rx*rx;   cart_y[2]=ay*rx*rx*rz;
        cart_y[3]=(ay*ry+2*ce)*rx*ry; cart_y[4]=(ay*ry+ce)*rx*rz;   cart_y[5]=ay*rx*rz*rz;
        cart_y[6]=(ay*ry+3*ce)*ry*ry; cart_y[7]=(ay*ry+2*ce)*ry*rz; cart_y[8]=(ay*ry+ce)*rz*rz;
        cart_y[9]=ay*rz*rz*rz;
        cart_z[0]=az*rx*rx*rx;        cart_z[1]=az*rx*rx*ry;        cart_z[2]=(az*rz+ce)*rx*rx;
        cart_z[3]=az*rx*ry*ry;        cart_z[4]=(az*rz+ce)*rx*ry;   cart_z[5]=(az*rz+2*ce)*rx*rz;
        cart_z[6]=az*ry*ry*ry;        cart_z[7]=(az*rz+ce)*ry*ry;   cart_z[8]=(az*rz+2*ce)*ry*rz;
        cart_z[9]=(az*rz+3*ce)*rz*rz;
    }

    int ncart = c2s_ncart[tid];
    int c2s_off = c2s_offset[tid];
    ao_v = 0; ao_x = 0; ao_y = 0; ao_z = 0;
    for (int c = 0; c < ncart; c++) {
        float coeff = c2s_data[c2s_off + c];
        ao_v += coeff * cart_v[c];
        ao_x += coeff * cart_x[c];
        ao_y += coeff * cart_y[c];
        ao_z += coeff * cart_z[c];
    }
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
                 'dm', 'ao_to_shell', 'c2s_data', 'c2s_offset', 'c2s_ncart'],
    output_names=['rho_out'],
    header=_FUSED_RHO_HEADER,
    source=_FUSED_RHO_GGA_SOURCE,
)


def _build_ao_mapping(mol):
    """Build AO-to-shell mapping and cart2sph data for fused kernel.

    Returns:
        ao_to_shell: (nao,) int32 — which shell each spherical AO comes from
        ao_to_start: (nao,) int32 — first spherical AO index of that shell
        c2s_data:    flat float32 — packed c2s rows for all AOs
        c2s_offset:  (nao,) int32 — offset into c2s_data for each AO's row
        c2s_ncart:   (nao,) int32 — number of Cartesian components for each AO's shell
    """
    nao = mol.nao
    ao_to_shell = np.zeros(nao, dtype=np.int32)
    ao_to_start = np.zeros(nao, dtype=np.int32)
    c2s_offset = np.zeros(nao, dtype=np.int32)
    c2s_ncart = np.zeros(nao, dtype=np.int32)
    c2s_rows = []
    data_offset = 0

    sph_off = 0
    for ish in range(mol.nbas):
        l = mol.bas_angular(ish)
        ncart = _ncart(l)
        nsph = ncart if l <= 1 else 2 * l + 1

        if l <= 1:
            # s/p: identity transform, one coefficient per AO
            for i in range(nsph):
                ao_to_shell[sph_off + i] = ish
                ao_to_start[sph_off + i] = sph_off
                c2s_offset[sph_off + i] = data_offset
                c2s_ncart[sph_off + i] = ncart
                # Identity row: 1.0 at position i, 0 elsewhere
                row = np.zeros(ncart, dtype=np.float32)
                row[i] = 1.0
                c2s_rows.append(row)
                data_offset += ncart
        else:
            # d/f/g: cart2sph matrix
            c2s = _cart2sph_matrix(l).astype(np.float32)  # (ncart, nsph)
            for i in range(nsph):
                ao_to_shell[sph_off + i] = ish
                ao_to_start[sph_off + i] = sph_off
                c2s_offset[sph_off + i] = data_offset
                c2s_ncart[sph_off + i] = ncart
                c2s_rows.append(c2s[:, i].copy())  # column i = row for sph AO i
                data_offset += ncart

        sph_off += nsph

    c2s_data = np.concatenate(c2s_rows).astype(np.float32)
    return ao_to_shell, ao_to_start, c2s_data, c2s_offset, c2s_ncart


def fused_rho_vxc(mol, coords, dm, weights, ni, xc_code, xctype,
                  shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
                  shell_data_gpu):
    """Compute rho and Vxc with fused GPU kernels where possible.

    Uses the fused threadgroup kernel for rho (LDA only, s/p basis).
    Falls back to the batched approach for GGA or d/f functions.
    """
    nao = mol.nao
    max_l = max(mol.bas_angular(i) for i in range(mol.nbas))

    # Fused kernel is faster for small molecules (nao <= ~128).
    # For larger nao, the batched approach with GPU gemm wins because
    # the dm@ao inner loop (O(nao) per thread) serializes badly,
    # and nao must be padded to power-of-2 threadgroup size.
    use_fused_rho = (xctype in ('LDA', 'GGA') and max_l <= 3 and nao <= 128)

    if not use_fused_rho:
        # Batched approach (existing, works for all cases)
        return _batched_rho_vxc(
            mol, coords, dm, weights, ni, xc_code, xctype,
            shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
            shell_data_gpu)

    # --- Fused rho for LDA s/p basis ---
    ngrids = coords.shape[0]
    ao_to_shell, ao_to_start, c2s_data, c2s_off, c2s_nc = _build_ao_mapping(mol)

    ao_to_shell_gpu = mx.array(ao_to_shell)
    c2s_data_gpu = mx.array(c2s_data)
    c2s_off_gpu = mx.array(c2s_off)
    c2s_nc_gpu = mx.array(c2s_nc)
    dm_gpu = mx.array(np.asarray(dm, dtype=np.float32).ravel())

    gridx = mx.array(coords[:, 0].astype(np.float32))
    gridy = mx.array(coords[:, 1].astype(np.float32))
    gridz = mx.array(coords[:, 2].astype(np.float32))

    # Pad nao to next power of 2 for efficient reduction
    max_nao = 1
    while max_nao < nao:
        max_nao *= 2

    # MLX grid = total threads, NOT threadgroup count.
    kernel_inputs = [gridx, gridy, gridz, exps_gpu, coeffs_gpu, shell_data_gpu,
                     dm_gpu, ao_to_shell_gpu, c2s_data_gpu, c2s_off_gpu, c2s_nc_gpu]
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
    # Pre-upload wv to GPU once (avoid per-batch numpy→MLX conversion)
    wv = weights * vxc
    ao_deriv = 1 if xctype == 'GGA' else 0
    if xctype == 'LDA':
        wv_gpu_all = mx.array(wv[0].astype(np.float32))
    else:  # GGA
        wv_all = wv.copy()
        wv_all[0] *= 0.5
        wv_gpu_all = mx.array(wv_all.astype(np.float32))

    # Grid coords already on GPU (gridx/gridy/gridz); slice per batch (MLX view)
    vmat_gpu = mx.zeros((nao, nao), dtype=mx.float32)
    BATCH = 20000
    for p0 in range(0, ngrids, BATCH):
        p1 = min(p0 + BATCH, ngrids)
        ng = p1 - p0
        ao_gpu = _eval_ao_batch_gpu(
            mol, gridx[p0:p1], gridy[p0:p1], gridz[p0:p1],
            ao_deriv, shell_data_gpu, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ng)

        if xctype == 'LDA':
            wv_blk = wv_gpu_all[p0:p1]
            vmat_gpu = vmat_gpu + (ao_gpu * wv_blk[None, :]) @ mx.transpose(ao_gpu)
        else:  # GGA
            wv_blk = wv_gpu_all[:, p0:p1]
            ao_val = ao_gpu[0]
            aow = ao_gpu[0] * wv_blk[0][None, :]
            for d in range(1, 4):
                aow = aow + ao_gpu[d] * wv_blk[d][None, :]
            vmat_gpu = vmat_gpu + aow @ mx.transpose(ao_val)

    mx.eval(vmat_gpu)
    vmat = np.array(vmat_gpu, dtype=np.float64)

    if xctype == 'GGA':
        vmat = vmat + vmat.T

    return nelec, excsum, vmat


def _batched_rho_vxc(mol, coords, dm, weights, ni, xc_code, xctype,
                     shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
                     shell_data_gpu):
    """Batched eval_ao + rho + Vxc (existing approach, works for all cases)."""
    nao = mol.nao
    ngrids = coords.shape[0]
    dm_gpu = mx.array(np.asarray(dm, dtype=np.float32))
    ao_deriv = 1 if xctype in ('GGA', 'MGGA') else 0

    # Pre-upload grid coords once (avoid per-batch numpy→MLX conversion)
    gridx_all = mx.array(coords[:, 0].astype(np.float32))
    gridy_all = mx.array(coords[:, 1].astype(np.float32))
    gridz_all = mx.array(coords[:, 2].astype(np.float32))

    nelec = 0.0
    excsum = 0.0
    vmat_gpu = mx.zeros((nao, nao), dtype=mx.float32)

    BATCH = 20000
    for p0 in range(0, ngrids, BATCH):
        p1 = min(p0 + BATCH, ngrids)
        ng = p1 - p0
        wt = weights[p0:p1]

        ao_gpu = _eval_ao_batch_gpu(
            mol, gridx_all[p0:p1], gridy_all[p0:p1], gridz_all[p0:p1],
            ao_deriv, shell_data_gpu, exps_gpu, coeffs_gpu,
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
            vmat_gpu = vmat_gpu + (ao_val * wv_gpu[None, :]) @ mx.transpose(ao_val)
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
            vmat_gpu = vmat_gpu + vmat_blk

        mx.eval(vmat_gpu)

    vmat = np.array(vmat_gpu, dtype=np.float64)

    if xctype in ('GGA', 'MGGA'):
        vmat = vmat + vmat.T

    return nelec, excsum, vmat


def _batched_rho_vxc_uks(mol, coords, dm_a, dm_b, weights, ni, xc_code, xctype,
                         shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
                         shell_data_gpu):
    """Batched UKS XC: separate rho for alpha/beta, spin-coupled libxc, separate Vxc."""
    nao = mol.nao
    ngrids = coords.shape[0]
    dma_gpu = mx.array(np.asarray(dm_a, dtype=np.float32))
    dmb_gpu = mx.array(np.asarray(dm_b, dtype=np.float32))
    ao_deriv = 1 if xctype in ('GGA', 'MGGA') else 0

    # Pre-upload grid coords once
    gridx_all = mx.array(coords[:, 0].astype(np.float32))
    gridy_all = mx.array(coords[:, 1].astype(np.float32))
    gridz_all = mx.array(coords[:, 2].astype(np.float32))

    nelec_a = 0.0
    nelec_b = 0.0
    excsum = 0.0
    vmat_a_gpu = mx.zeros((nao, nao), dtype=mx.float32)
    vmat_b_gpu = mx.zeros((nao, nao), dtype=mx.float32)

    BATCH = 20000
    for p0 in range(0, ngrids, BATCH):
        p1 = min(p0 + BATCH, ngrids)
        ng = p1 - p0
        wt = weights[p0:p1]

        ao_gpu = _eval_ao_batch_gpu(
            mol, gridx_all[p0:p1], gridy_all[p0:p1], gridz_all[p0:p1],
            ao_deriv, shell_data_gpu, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ng)

        # Compute rho for both spins
        if xctype == 'LDA':
            ao_val = ao_gpu
            tmp_a = dma_gpu @ ao_val
            tmp_b = dmb_gpu @ ao_val
            rho_a_gpu = mx.sum(ao_val * tmp_a, axis=0)
            rho_b_gpu = mx.sum(ao_val * tmp_b, axis=0)
            mx.eval(rho_a_gpu, rho_b_gpu)
            rho_a = np.array(rho_a_gpu, dtype=np.float64)
            rho_b = np.array(rho_b_gpu, dtype=np.float64)
        else:
            ao_val = ao_gpu[0]
            tmp_a = dma_gpu @ ao_val
            tmp_b = dmb_gpu @ ao_val
            ra0 = mx.sum(ao_val * tmp_a, axis=0)
            ra1 = 2.0 * mx.sum(ao_gpu[1] * tmp_a, axis=0)
            ra2 = 2.0 * mx.sum(ao_gpu[2] * tmp_a, axis=0)
            ra3 = 2.0 * mx.sum(ao_gpu[3] * tmp_a, axis=0)
            rb0 = mx.sum(ao_val * tmp_b, axis=0)
            rb1 = 2.0 * mx.sum(ao_gpu[1] * tmp_b, axis=0)
            rb2 = 2.0 * mx.sum(ao_gpu[2] * tmp_b, axis=0)
            rb3 = 2.0 * mx.sum(ao_gpu[3] * tmp_b, axis=0)
            mx.eval(ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3)
            rho_a = np.empty((4, ng), dtype=np.float64)
            rho_b = np.empty((4, ng), dtype=np.float64)
            rho_a[0] = np.array(ra0); rho_a[1] = np.array(ra1)
            rho_a[2] = np.array(ra2); rho_a[3] = np.array(ra3)
            rho_b[0] = np.array(rb0); rho_b[1] = np.array(rb1)
            rho_b[2] = np.array(rb2); rho_b[3] = np.array(rb3)

        # Spin-coupled XC (single libxc call)
        rho = (rho_a, rho_b)
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[:2]
        # vxc shape: (2, ncomp, ngrids) — alpha, beta
        if xctype == 'LDA':
            den_a = rho_a * wt; den_b = rho_b * wt
        else:
            den_a = rho_a[0] * wt; den_b = rho_b[0] * wt
        nelec_a += den_a.sum()
        nelec_b += den_b.sum()
        excsum += np.dot(den_a + den_b, exc)

        # Vxc contraction: separate alpha/beta
        wv = wt * vxc  # shape (2, ncomp, ngrids)
        if xctype == 'LDA':
            wv_a_gpu = mx.array(wv[0, 0].astype(np.float32))
            wv_b_gpu = mx.array(wv[1, 0].astype(np.float32))
            vmat_a_gpu = vmat_a_gpu + (ao_val * wv_a_gpu[None, :]) @ mx.transpose(ao_val)
            vmat_b_gpu = vmat_b_gpu + (ao_val * wv_b_gpu[None, :]) @ mx.transpose(ao_val)
        else:
            wv_a = wv[0].copy(); wv_a[0] *= 0.5
            wv_b = wv[1].copy(); wv_b[0] *= 0.5
            wv_a_gpu = [mx.array(wv_a[d].astype(np.float32)) for d in range(min(4, len(wv_a)))]
            wv_b_gpu = [mx.array(wv_b[d].astype(np.float32)) for d in range(min(4, len(wv_b)))]
            aow_a = ao_gpu[0] * wv_a_gpu[0][None, :]
            aow_b = ao_gpu[0] * wv_b_gpu[0][None, :]
            for d in range(1, 4):
                aow_a = aow_a + ao_gpu[d] * wv_a_gpu[d][None, :]
                aow_b = aow_b + ao_gpu[d] * wv_b_gpu[d][None, :]
            vmat_a_gpu = vmat_a_gpu + aow_a @ mx.transpose(ao_val)
            vmat_b_gpu = vmat_b_gpu + aow_b @ mx.transpose(ao_val)
            if xctype == 'MGGA' and wv.shape[1] > 4:
                wv4a = mx.array((wv[0, 4] * 0.5).astype(np.float32))
                wv4b = mx.array((wv[1, 4] * 0.5).astype(np.float32))
                for d in range(1, 4):
                    vmat_a_gpu = vmat_a_gpu + (ao_gpu[d] * wv4a[None, :]) @ mx.transpose(ao_gpu[d])
                    vmat_b_gpu = vmat_b_gpu + (ao_gpu[d] * wv4b[None, :]) @ mx.transpose(ao_gpu[d])

        mx.eval(vmat_a_gpu, vmat_b_gpu)

    vmat_a = np.array(vmat_a_gpu, dtype=np.float64)
    vmat_b = np.array(vmat_b_gpu, dtype=np.float64)
    if xctype in ('GGA', 'MGGA'):
        vmat_a = vmat_a + vmat_a.T
        vmat_b = vmat_b + vmat_b.T

    vmat = np.stack([vmat_a, vmat_b])
    nelec = np.array([nelec_a, nelec_b])
    return nelec, excsum, vmat
