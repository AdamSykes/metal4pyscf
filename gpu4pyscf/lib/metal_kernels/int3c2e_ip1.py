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
Metal GPU computation of int3c2e_ip1 — 3-center 2-electron integral
first-center derivatives: nabla_A (mu nu | P).

Architecture (follows rys_jk_metal.py proven pattern):
  Phase A (CPU, f64): enumerate primitive triples, compute Boys Fm(T),
    Rys roots/weights. Pack per-primitive task data.
  Phase B (Metal, f32): TRR vertical recursion → HRR horizontal transfer
    → derivative via raised angular momentum → Cartesian output.
  Phase C (CPU, f64): accumulate primitives per shell triple, write
    into (3, nao, nao, naux_slice) output tensor.

Supports l_orb <= 3 (f), l_aux <= 4 (g), nroots <= 6.
"""

import numpy as np
import mlx.core as mx
from math import pi, sqrt
from gpu4pyscf.lib.metal_kernels.rys_jk_metal import (
    _boys_function_vec,
)
from gpu4pyscf.lib.metal_kernels.rys_jk import _rys_from_boys
from pyscf.gto.moleintor import getints, make_cintopt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NCART_LUT = np.array([1, 3, 6, 10, 15], dtype=np.int32)
PI_5_2 = pi ** 2.5  # 34.986...

# CINTcommon_fac_sp: per-shell normalization correction between
# _libcint_ctr_coeff (radial gto_norm) and the standard integral formula.
# For s-type (l=0): extra sqrt(4pi) in gto_norm, corrected by 1/sqrt(4pi).
# For p-type (l=1): extra sqrt(4pi/3) in gto_norm, corrected by sqrt(3/(4pi)).
# For l>=2: gto_norm matches the integral formula directly, no correction.
_FAC_L = {0: sqrt(1.0 / (4 * pi)),
          1: sqrt(3.0 / (4 * pi)),
          2: 1.0, 3: 1.0, 4: 1.0}

# Task layout (36 floats per primitive triple)
TASK_STRIDE = 36
#  [0]     aij (combined bra exponent)
#  [1]     ak  (aux exponent)
#  [2:5]   PA  (P - A, 3 floats)
#  [5:8]   PC  (P - C, 3 floats; also used as Rpq since Q=C for 3-center)
#  [8:11]  AB  (A - B, 3 floats)
#  [11]    prefac
#  [12]    ai  (primitive exponent on center A, for derivative)
#  [13:16] li, lj, lk  (angular momenta as int-in-float)
#  [16]    out_offset  (offset into per-task output buffer)
#  [17]    nroots
#  [18:24] roots[0..5]   (6 floats, zero-padded)
#  [24:30] weights[0..5] (6 floats, zero-padded)
#  [30:33] Rpq (= P-C, duplicated for kernel convenience)
#  [33:36] reserved


# ---------------------------------------------------------------------------
# Cartesian index encoding (l=0..4, base-25: ix*25 + iy*5 + iz)
# ---------------------------------------------------------------------------

def _cart_powers(l):
    """Cartesian (ix,iy,iz) with ix+iy+iz=l in PySCF ordering."""
    return [(ix, iy, l - ix - iy)
            for ix in range(l, -1, -1)
            for iy in range(l - ix, -1, -1)]


def _build_cart_table():
    ncart = [1, 3, 6, 10, 15]
    cart_all = []
    cart_off = [0]
    for l in range(5):
        for ix, iy, iz in _cart_powers(l):
            cart_all.append(ix * 25 + iy * 5 + iz)
        cart_off.append(cart_off[-1] + ncart[l])
    return ncart, cart_all, cart_off[:-1]


_NCART_TBL, _CART_ALL_TBL, _CART_OFF_TBL = _build_cart_table()


# ---------------------------------------------------------------------------
# Metal kernel: TRR + HRR + derivative contraction (3-center)
# ---------------------------------------------------------------------------

_HEADER_3C = '''
constant int NCART[] = {1, 3, 6, 10, 15};
constant int CART_ALL[] = {
''' + ','.join(str(v) for v in _CART_ALL_TBL) + '''
};
constant int CART_OFF[] = {0, 1, 4, 10, 20};
'''

_SOURCE_3C = '''
uint tid = thread_position_in_grid.x;
if (tid >= n_tasks) return;

int off = tid * TASK_STRIDE;
float aij     = task_data[off + 0];
float ak      = task_data[off + 1];
float PA_x    = task_data[off + 2];
float PA_y    = task_data[off + 3];
float PA_z    = task_data[off + 4];
float AB_x    = task_data[off + 8];
float AB_y    = task_data[off + 9];
float AB_z    = task_data[off + 10];
float prefac  = task_data[off + 11];
float ai_exp  = task_data[off + 12];
int li        = (int)task_data[off + 13];
int lj        = (int)task_data[off + 14];
int lk        = (int)task_data[off + 15];
int out_off   = (int)task_data[off + 16];
int nroots    = (int)task_data[off + 17];
float rys_r[6] = {task_data[off+18], task_data[off+19], task_data[off+20],
                   task_data[off+21], task_data[off+22], task_data[off+23]};
float rys_w[6] = {task_data[off+24], task_data[off+25], task_data[off+26],
                   task_data[off+27], task_data[off+28], task_data[off+29]};
float Rpq_x   = task_data[off + 30];
float Rpq_y   = task_data[off + 31];
float Rpq_z   = task_data[off + 32];

int li1 = li + 1;
int lij1 = li1 + lj;
int ni = NCART[li], nj = NCART[lj], nk = NCART[lk];
int ci_off = CART_OFF[li], cj_off = CART_OFF[lj], ck_off = CART_OFF[lk];

// Zero output: 3 * ni * nj * nk
int comp_stride = ni * nj * nk;
int n_out = 3 * comp_stride;
for (int i = 0; i < n_out; i++) int_out[out_off + i] = 0.0f;

float PA[3] = {PA_x, PA_y, PA_z};
float AB[3] = {AB_x, AB_y, AB_z};
float Rpq[3] = {Rpq_x, Rpq_y, Rpq_z};
float aijk = aij + ak;

for (int iroot = 0; iroot < nroots; iroot++) {
    float rt = rys_r[iroot];
    float wt = rys_w[iroot];

    // Recursion coefficients (identical to 4c with akl=ak, lkl=lk)
    float rt_aa  = rt / aijk;
    float rt_aij = rt_aa * ak;
    float b10 = 0.5f / aij * (1.0f - rt_aij);
    float b01 = 0.5f / ak  * (1.0f - rt_aa * aij);
    float b00 = 0.5f * rt_aa;

    // Per-direction 1D recursion arrays
    // g_ij[a][j][k]: a=0..lij1(8), j=0..lj(3), k=0..lk(4) → max 9*4*5=180
    float g_ij[3][9][4][5];

    for (int dir = 0; dir < 3; dir++) {
        float c0 = PA[dir] - rt_aij * Rpq[dir];       // bra TRR center
        float cp = (rt_aa * aij) * Rpq[dir];            // ket TRR center (QC=0)

        // TRR: build g[a][k] for a=0..lij1, k=0..lk
        float g[9][5];
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++) g[a][c] = 0.0f;
        g[0][0] = 1.0f;

        // Grow bra at k=0
        if (lij1 > 0) g[1][0] = c0;
        for (int a = 1; a < lij1; a++)
            g[a+1][0] = c0 * g[a][0] + float(a) * b10 * g[a-1][0];

        // Grow ket with bra coupling
        for (int c = 0; c < lk; c++) {
            for (int a = 0; a <= lij1; a++) {
                float val = cp * g[a][c];
                if (c > 0) val += float(c) * b01 * g[a][c-1];
                if (a > 0) val += float(a) * b00 * g[a-1][c];
                g[a][c+1] = val;
            }
        }

        // HRR: g_ij[a][0][k] = g[a][k], then
        //   g_ij[i][j+1][k] = g_ij[i+1][j][k] + AB[dir] * g_ij[i][j][k]
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++)
                g_ij[dir][a][0][c] = g[a][c];

        float ab = AB[dir];
        for (int j = 0; j < lj; j++)
            for (int c = 0; c <= lk; c++)
                for (int i = 0; i <= lij1 - j - 1; i++)
                    g_ij[dir][i][j+1][c] = g_ij[dir][i+1][j][c] + ab * g_ij[dir][i][j][c];
    }

    // Contract over this root with derivative application
    float pf_wt = prefac * wt;

    for (int ii = 0; ii < ni; ii++) {
        int ci_v = CART_ALL[ci_off + ii];
        int ix = ci_v / 25, iy = (ci_v / 5) % 5, iz = ci_v % 5;
        for (int jj = 0; jj < nj; jj++) {
            int cj_v = CART_ALL[cj_off + jj];
            int jx = cj_v / 25, jy = (cj_v / 5) % 5, jz = cj_v % 5;
            for (int kk = 0; kk < nk; kk++) {
                int ck_v = CART_ALL[ck_off + kk];
                int kx = ck_v / 25, ky = (ck_v / 5) % 5, kz = ck_v % 5;

                float vx = g_ij[0][ix][jx][kx];
                float vy = g_ij[1][iy][jy][ky];
                float vz = g_ij[2][iz][jz][kz];

                // d/dA_x: 2*ai * g[ix+1,jx,kx] - ix * g[ix-1,jx,kx]
                float dx = 2.0f * ai_exp * g_ij[0][ix+1][jx][kx];
                if (ix > 0) dx -= float(ix) * g_ij[0][ix-1][jx][kx];

                float dy = 2.0f * ai_exp * g_ij[1][iy+1][jy][ky];
                if (iy > 0) dy -= float(iy) * g_ij[1][iy-1][jy][ky];

                float dz = 2.0f * ai_exp * g_ij[2][iz+1][jz][kz];
                if (iz > 0) dz -= float(iz) * g_ij[2][iz-1][jz][kz];

                int base = out_off + (ii * nj + jj) * nk + kk;
                int_out[base]                  += pf_wt * dx * vy * vz;
                int_out[base + comp_stride]    += pf_wt * vx * dy * vz;
                int_out[base + 2*comp_stride]  += pf_wt * vx * vy * dz;
            }
        }
    }
} // end root loop
'''

_int3c2e_ip1_kernel = mx.fast.metal_kernel(
    name='int3c2e_ip1_trr',
    input_names=['task_data'],
    output_names=['int_out'],
    header=_HEADER_3C,
    source=_SOURCE_3C,
    atomic_outputs=False,
)


# ---------------------------------------------------------------------------
# CPU Phase A: build tasks (Boys + Rys on CPU in f64)
# ---------------------------------------------------------------------------

def _rys_from_boys_general_batch(nroots, fm):
    """Batch Rys roots/weights for general nroots (1..6).

    Args:
        nroots: int
        fm: (2*nroots, N) array of Boys moments
    Returns:
        roots: (N, 6) zero-padded
        weights: (N, 6) zero-padded
    """
    N = fm.shape[1]
    roots = np.zeros((N, 6), dtype=np.float64)
    weights = np.zeros((N, 6), dtype=np.float64)
    if N == 0:
        return roots, weights
    # Process each point individually (safe for all nroots)
    for i in range(N):
        rt, wt = _rys_from_boys(nroots, fm[:, i])
        roots[i, :nroots] = rt[:nroots]
        weights[i, :nroots] = wt[:nroots]
    return roots, weights


def _build_3c_tasks(mol, auxmol, shl0_aux, shl1_aux):
    """Build task array for int3c2e_ip1 Metal kernel.

    Enumerates all primitive triples (pi, pj, pk) for orbital shell pairs
    (ish, jsh) and aux shells ksh in [shl0_aux, shl1_aux). Computes Boys
    function and Rys roots/weights on CPU in f64. Packs into TASK_STRIDE
    float32 task array for GPU.

    Returns:
        tasks: (n_tasks, TASK_STRIDE) float32
        meta: list of (ish, jsh, ksh, task_start, task_count, nfi, nfj, nfk)
        total_out: total output floats needed
    """
    nbas = mol.nbas
    tasks_list = []
    meta = []
    out_offset = 0

    for ish in range(nbas):
        li = mol.bas_angular(ish)
        if li > 3:
            continue
        ai_arr = mol.bas_exp(ish)
        ci_arr = mol._libcint_ctr_coeff(ish).flatten()
        Ri = mol.atom_coord(mol.bas_atom(ish))
        npi = len(ai_arr)
        nfi = _NCART_LUT[li]

        for jsh in range(nbas):
            lj = mol.bas_angular(jsh)
            if lj > 3:
                continue
            aj_arr = mol.bas_exp(jsh)
            cj_arr = mol._libcint_ctr_coeff(jsh).flatten()
            Rj = mol.atom_coord(mol.bas_atom(jsh))
            npj = len(aj_arr)
            nfj = _NCART_LUT[lj]
            rr_ab = np.dot(Ri - Rj, Ri - Rj)
            AB = Ri - Rj

            for ksh_idx in range(shl0_aux, shl1_aux):
                lk = auxmol.bas_angular(ksh_idx)
                if lk > 4:
                    continue
                ak_arr = auxmol.bas_exp(ksh_idx)
                ck_arr = auxmol._libcint_ctr_coeff(ksh_idx).flatten()
                Rk = auxmol.atom_coord(auxmol.bas_atom(ksh_idx))
                npk = len(ak_arr)
                nfk = _NCART_LUT[lk]

                li1 = li + 1
                nroots = (li1 + lj + lk) // 2 + 1
                n_out_triple = 3 * nfi * nfj * nfk
                task_start = len(tasks_list)

                for pi in range(npi):
                    ai = float(ai_arr[pi])
                    ci = float(ci_arr[pi])
                    for pj in range(npj):
                        aj = float(aj_arr[pj])
                        cj = float(cj_arr[pj])
                        aij = ai + aj
                        eij = np.exp(-ai * aj / aij * rr_ab)
                        if eij < 1e-14:
                            continue
                        P = (ai * Ri + aj * Rj) / aij
                        PA = P - Ri
                        coeff_ij = ci * cj * eij

                        for pk in range(npk):
                            ak = float(ak_arr[pk])
                            ck = float(ck_arr[pk])
                            aijk = aij + ak
                            a0 = aij * ak / aijk
                            PC = P - Rk
                            T = a0 * np.dot(PC, PC)
                            prefac = coeff_ij * ck * 2.0 * PI_5_2 / (aij * ak * sqrt(aijk))
                            # Don't apply fac_l here; it's applied per-shell in Phase C

                            # Boys function (CPU f64)
                            fm = _boys_function_vec(2 * nroots - 1, np.array([T]))
                            rt, wt = _rys_from_boys(nroots, fm[:, 0])

                            task = np.zeros(TASK_STRIDE, dtype=np.float32)
                            task[0] = aij
                            task[1] = ak
                            task[2:5] = PA.astype(np.float32)
                            task[5:8] = PC.astype(np.float32)
                            task[8:11] = AB.astype(np.float32)
                            task[11] = prefac
                            task[12] = ai
                            task[13] = li
                            task[14] = lj
                            task[15] = lk
                            # Each primitive gets its OWN output region
                            # (no atomics, no race conditions)
                            task[16] = out_offset
                            task[17] = nroots
                            task[18:18 + nroots] = rt[:nroots].astype(np.float32)
                            task[24:24 + nroots] = wt[:nroots].astype(np.float32)
                            task[30:33] = PC.astype(np.float32)
                            tasks_list.append(task)
                            out_offset += n_out_triple

                task_count = len(tasks_list) - task_start
                if task_count > 0:
                    meta.append((ish, jsh, ksh_idx, task_start, task_count,
                                 nfi, nfj, nfk, n_out_triple))

    if not tasks_list:
        return np.zeros((0, TASK_STRIDE), dtype=np.float32), meta, 0
    return np.array(tasks_list, dtype=np.float32), meta, out_offset


# ---------------------------------------------------------------------------
# CPU Phase C: accumulate primitives → output tensor
# ---------------------------------------------------------------------------

def _accumulate_output(int_buf, meta, mol, auxmol, shl0_aux, shl1_aux):
    """Sum primitive contributions per shell triple into (3, nao, nao, naux_slice)."""
    nao = mol.nao_cart()
    aux_loc = auxmol.ao_loc_nr(cart=True)
    naux_slice = aux_loc[shl1_aux] - aux_loc[shl0_aux]
    p0_aux = aux_loc[shl0_aux]
    ao_loc = mol.ao_loc_nr(cart=True)

    out = np.zeros((3, nao, nao, naux_slice), dtype=np.float64)
    for ish, jsh, ksh, t_start, t_count, nfi, nfj, nfk, n_out in meta:
        if t_count == 0:
            continue
        i0 = ao_loc[ish]
        j0 = ao_loc[jsh]
        k0 = aux_loc[ksh] - p0_aux
        # Sum all primitive contributions for this shell triple
        # Each primitive wrote n_out floats starting at its out_offset
        block = np.zeros(n_out, dtype=np.float64)
        for t in range(t_count):
            off = int(meta[meta.index((ish, jsh, ksh, t_start, t_count,
                                       nfi, nfj, nfk, n_out))][3])
            # Actually the offset is stored in the task data
            pass
        # Simpler: iterate tasks and accumulate
        pass

    # Re-implementation: just index by meta
    out = np.zeros((3, nao, nao, naux_slice), dtype=np.float64)
    for ish, jsh, ksh, t_start, t_count, nfi, nfj, nfk, n_out in meta:
        i0 = ao_loc[ish]
        j0 = ao_loc[jsh]
        k0 = aux_loc[ksh] - p0_aux
        comp_stride = nfi * nfj * nfk
        # All primitives for this triple share the same out_offset (set to
        # the FIRST task's offset).  Actually no -- each primitive has its
        # OWN output region. We need to sum them.
        for t_idx in range(t_start, t_start + t_count):
            t_off = int(round(float(0)))  # need to track offsets properly
            pass

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_int3c2e_ip1_metal(mol, auxmol, shls_slice=None):
    """Compute int3c2e_ip1 on Metal GPU.

    Args:
        mol: orbital basis Mole
        auxmol: auxiliary basis Mole
        shls_slice: (i0, i1, j0, j1, k0_aux, k1_aux) in aux-shell indices

    Returns:
        (3, nao_cart, nao_cart, naux_slice_cart) float64 array in Cartesian
        basis, matching PySCF getints('int3c2e_ip1', aosym='s1').
    """
    nbas = mol.nbas
    if shls_slice is None:
        shl0_aux, shl1_aux = 0, auxmol.nbas
        orb_slice = (0, nbas, 0, nbas)
    else:
        orb_slice = shls_slice[:4]
        shl0_aux = shls_slice[4]
        shl1_aux = shls_slice[5]

    # Check if all orbital shells are supported
    max_l_orb = max(mol.bas_angular(i) for i in range(nbas))
    max_l_aux = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))
    if max_l_orb > 3 or max_l_aux > 4:
        return _cpu_fallback(mol, auxmol, shls_slice)

    # Phase A: build tasks on CPU
    tasks, meta, total_out = _build_3c_tasks(mol, auxmol, shl0_aux, shl1_aux)

    if len(tasks) == 0 or total_out == 0:
        return _cpu_fallback(mol, auxmol, shls_slice)

    # Phase B: run Metal kernel
    task_gpu = mx.array(tasks.ravel())
    n_tasks = tasks.shape[0]
    THREADS = 256
    grid_size = ((n_tasks + THREADS - 1) // THREADS) * THREADS

    result = _int3c2e_ip1_kernel(
        inputs=[task_gpu],
        grid=(grid_size, 1, 1),
        threadgroup=(min(THREADS, n_tasks), 1, 1),
        output_shapes=[(total_out,)],
        output_dtypes=[mx.float32],
        template=[('n_tasks', n_tasks), ('TASK_STRIDE', TASK_STRIDE)],
    )
    mx.eval(result[0])
    int_buf = np.array(result[0]).astype(np.float64)

    # Phase C: accumulate into output tensor
    nao_cart = mol.nao_cart()
    aux_loc = auxmol.ao_loc_nr(cart=True)
    naux_slice = aux_loc[shl1_aux] - aux_loc[shl0_aux]
    p0_aux = aux_loc[shl0_aux]
    ao_loc = mol.ao_loc_nr(cart=True)

    out = np.zeros((3, nao_cart, nao_cart, naux_slice), dtype=np.float64)
    for ish, jsh, ksh, t_start, t_count, nfi, nfj, nfk, n_out in meta:
        i0 = ao_loc[ish]
        j0 = ao_loc[jsh]
        k0 = aux_loc[ksh] - p0_aux
        comp_stride = nfi * nfj * nfk
        # Sum all primitives for this shell triple
        shell_block = np.zeros(n_out, dtype=np.float64)
        for t in range(t_count):
            t_off = int(tasks[t_start + t, 16])  # out_offset from task data
            shell_block += int_buf[t_off:t_off + n_out]
        # Unpack into output tensor with per-shell normalization
        # fac_l = CINTcommon_fac_sp: 1/sqrt(4pi) for s, sqrt(3/(4pi)) for p, 1.0 for l>=2
        li = mol.bas_angular(ish)
        lj = mol.bas_angular(jsh)
        lk = auxmol.bas_angular(ksh)
        fac = _FAC_L[li] * _FAC_L[lj] * _FAC_L[lk]
        for comp in range(3):
            block = shell_block[comp*comp_stride:(comp+1)*comp_stride].reshape(nfi, nfj, nfk)
            # Negate: int3c2e_ip1 = -d/dA (PySCF convention)
            out[comp, i0:i0+nfi, j0:j0+nfj, k0:k0+nfk] -= block * fac

    return out


def _cpu_fallback(mol, auxmol, shls_slice):
    """Fall back to PySCF CPU for int3c2e_ip1."""
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix('int3c2e_ip1')
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    if shls_slice is None:
        s = (0, nbas, 0, nbas, nbas, pmol.nbas)
    else:
        s = shls_slice[:4] + (nbas + shls_slice[4], nbas + shls_slice[5])
    return getints(intor, pmol._atm, pmol._bas, pmol._env, s,
                   aosym='s1', cintopt=opt)
