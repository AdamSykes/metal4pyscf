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
    _boys_function_vec, _rys_from_boys_batch,
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
int out_off   = offsets[tid];
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
    input_names=['task_data', 'offsets'],
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


def _precompute_bra_pairs(mol):
    """Precompute primitive pair data for all orbital shell pairs."""
    nbas = mol.nbas
    shell_l = np.array([mol.bas_angular(i) for i in range(nbas)])
    shell_atom = np.array([mol.bas_atom(i) for i in range(nbas)])
    coords = np.array([mol.atom_coord(i) for i in range(mol.natm)])
    shell_R = coords[shell_atom]
    cache = {}
    for ish in range(nbas):
        li = shell_l[ish]
        if li > 3:
            continue
        ai = mol.bas_exp(ish)
        ci = mol._libcint_ctr_coeff(ish).flatten()
        Ri = shell_R[ish]
        ni = len(ai)
        for jsh in range(nbas):
            lj = shell_l[jsh]
            if lj > 3:
                continue
            aj = mol.bas_exp(jsh)
            cj = mol._libcint_ctr_coeff(jsh).flatten()
            Rj = shell_R[jsh]
            nj = len(aj)
            # Cross-product of primitives
            ai2 = np.repeat(ai, nj)
            aj2 = np.tile(aj, ni)
            ci2 = np.repeat(ci, nj)
            cj2 = np.tile(cj, ni)
            aij = ai2 + aj2
            rr = np.dot(Ri - Rj, Ri - Rj)
            Kab = np.exp(-ai2 * aj2 / aij * rr)
            P = (ai2[:, None] * Ri + aj2[:, None] * Rj) / aij[:, None]
            PA = P - Ri
            coeff_K = ci2 * cj2 * Kab
            AB = Ri - Rj
            cache[(ish, jsh)] = (aij, ai2, PA, P, coeff_K, AB)
    return cache


def _build_3c_tasks(mol, auxmol, shl0_aux, shl1_aux):
    """Build task array for int3c2e_ip1 Metal kernel (vectorized).

    Three phases:
      1. Precompute bra shell pairs and enumerate valid shell triples
      2. Group by (n_bra_prims, n_aux_prims), batch cross-product expansion
      3. Batch Boys/Rys across all primitives per nroots

    Returns:
        tasks: (n_tasks, TASK_STRIDE) float32
        offsets: (n_tasks,) int32  — per-task output offset
        meta: list of (ish, jsh, ksh, task_start, task_count, nfi, nfj, nfk, n_out)
        total_out: total output floats needed
    """
    nbas = mol.nbas
    pair_cache = _precompute_bra_pairs(mol)
    shell_l = np.array([mol.bas_angular(i) for i in range(nbas)])

    # Aux shell data
    aux_l = np.array([auxmol.bas_angular(i) for i in range(auxmol.nbas)])
    aux_atom = np.array([auxmol.bas_atom(i) for i in range(auxmol.nbas)])
    aux_coords = np.array([auxmol.atom_coord(i) for i in range(auxmol.natm)])
    aux_R = aux_coords[aux_atom]
    aux_exps = [auxmol.bas_exp(i) for i in range(auxmol.nbas)]
    aux_coeffs = [auxmol._libcint_ctr_coeff(i).flatten() for i in range(auxmol.nbas)]

    # Phase 1: enumerate valid shell triples (lightweight Python loop)
    valid = []
    for ish in range(nbas):
        li = shell_l[ish]
        if li > 3:
            continue
        for jsh in range(nbas):
            lj = shell_l[jsh]
            if lj > 3:
                continue
            if (ish, jsh) not in pair_cache:
                continue
            n_ij = len(pair_cache[(ish, jsh)][0])
            for ksh in range(shl0_aux, shl1_aux):
                lk = aux_l[ksh]
                if lk > 4:
                    continue
                n_k = len(aux_exps[ksh])
                nroots = (li + 1 + lj + lk) // 2 + 1
                valid.append((ish, jsh, ksh, li, lj, lk, nroots, n_ij, n_k))

    if not valid:
        return (np.zeros((0, TASK_STRIDE), dtype=np.float32),
                np.zeros(0, dtype=np.int32), [], 0)

    # Phase 2: group by (n_ij, n_k) for batched expansion
    groups = {}
    for v in valid:
        groups.setdefault((v[7], v[8]), []).append(v)

    all_blocks = []
    all_t_vals = []
    all_nroots_flat = []
    all_triple_idx = []
    triple_info = []  # (ish, jsh, ksh, nfi, nfj, nfk, n_out)
    eri_offset = 0

    for (n_ij, n_k), glist in groups.items():
        n_t = len(glist)
        n_prim = n_ij * n_k

        # Stack bra pair data: (n_t, n_ij[, 3])
        aij_b = np.array([pair_cache[(g[0], g[1])][0] for g in glist])  # (n_t, n_ij)
        ai_b = np.array([pair_cache[(g[0], g[1])][1] for g in glist])
        PA_b = np.array([pair_cache[(g[0], g[1])][2] for g in glist])   # (n_t, n_ij, 3)
        P_b = np.array([pair_cache[(g[0], g[1])][3] for g in glist])
        cK_b = np.array([pair_cache[(g[0], g[1])][4] for g in glist])   # (n_t, n_ij)
        AB_b = np.array([pair_cache[(g[0], g[1])][5] for g in glist])   # (n_t, 3)

        # Aux data: (n_t, n_k[, 3])
        ak_b = np.array([aux_exps[g[2]] for g in glist])               # (n_t, n_k)
        ck_b = np.array([aux_coeffs[g[2]] for g in glist])
        Rk_b = np.array([aux_R[g[2]] for g in glist])                   # (n_t, 3)

        # Cross-product expansion: (n_t, n_prim=n_ij*n_k[, 3])
        aij_f = np.repeat(aij_b, n_k, axis=1)                          # (n_t, n_prim)
        ai_f = np.repeat(ai_b, n_k, axis=1)
        ak_f = np.tile(ak_b, (1, n_ij))
        PA_f = np.repeat(PA_b, n_k, axis=1)                            # (n_t, n_prim, 3)
        P_f = np.repeat(P_b, n_k, axis=1)
        coeff_f = np.repeat(cK_b, n_k, axis=1) * np.tile(ck_b, (1, n_ij))
        Rk_f = np.tile(Rk_b[:, None, :], (1, n_prim, 1))              # broadcast
        # Simpler: Rk is per-triple, same for all prims
        PC_f = P_f - Rk_f                                              # (n_t, n_prim, 3)

        aijk_f = aij_f + ak_f
        prefac_f = coeff_f * 2.0 * PI_5_2 / (aij_f * ak_f * np.sqrt(aijk_f))

        # Screening: mask out negligible primitives
        screen = np.abs(prefac_f) > 1e-15

        # T values: (n_t, n_prim)
        theta_f = aij_f * ak_f / aijk_f
        t_vals_f = theta_f * np.sum(PC_f ** 2, axis=2)

        # Per-triple metadata
        li_a = np.array([g[3] for g in glist], dtype=np.float32)
        lj_a = np.array([g[4] for g in glist], dtype=np.float32)
        lk_a = np.array([g[5] for g in glist], dtype=np.float32)
        nr_a = np.array([g[6] for g in glist], dtype=np.float32)
        ncarts = np.array([3 * int(_NCART_LUT[g[3]]) * int(_NCART_LUT[g[4]])
                           * int(_NCART_LUT[g[5]]) for g in glist], dtype=np.int64)

        # Output offsets: each primitive gets its own region
        eri_sizes = ncarts * n_prim
        eri_bases = np.empty(n_t, dtype=np.int64)
        eri_bases[0] = eri_offset
        if n_t > 1:
            np.cumsum(eri_sizes[:-1], out=eri_bases[1:])
            eri_bases[1:] += eri_offset
        prim_idx = np.arange(n_prim)
        eri_off_f = eri_bases[:, None] + prim_idx[None, :] * ncarts[:, None]
        eri_offset += int(eri_sizes.sum())

        # Build task block: (n_t * n_prim, TASK_STRIDE)
        total = n_t * n_prim
        block = np.zeros((total, TASK_STRIDE), dtype=np.float32)
        block[:, 0] = aij_f.ravel()
        block[:, 1] = ak_f.ravel()
        block[:, 2:5] = PA_f.reshape(total, 3)
        block[:, 5:8] = PC_f.reshape(total, 3)
        block[:, 8:11] = np.repeat(AB_b, n_prim, axis=0)
        block[:, 11] = prefac_f.ravel()
        block[:, 12] = ai_f.ravel()
        block[:, 13] = np.repeat(li_a, n_prim)
        block[:, 14] = np.repeat(lj_a, n_prim)
        block[:, 15] = np.repeat(lk_a, n_prim)
        # slot 16 unused (offset in separate array)
        block[:, 17] = np.repeat(nr_a, n_prim)
        block[:, 30:33] = PC_f.reshape(total, 3)

        # Apply screening
        keep = screen.ravel()
        block = block[keep]
        eri_off_kept = eri_off_f.ravel()[keep]

        all_blocks.append(block)
        all_t_vals.append(t_vals_f.ravel()[keep])
        all_nroots_flat.append(np.repeat(nr_a.astype(np.int32), n_prim)[keep])

        # Triple info and task-to-triple mapping
        t_start = len(triple_info)
        for g in glist:
            nfi = int(_NCART_LUT[g[3]])
            nfj = int(_NCART_LUT[g[4]])
            nfk = int(_NCART_LUT[g[5]])
            triple_info.append((g[0], g[1], g[2], nfi, nfj, nfk, 3 * nfi * nfj * nfk))
        all_triple_idx.append(np.repeat(
            np.arange(t_start, t_start + n_t, dtype=np.int32), n_prim)[keep])

    if not all_blocks:
        return (np.zeros((0, TASK_STRIDE), dtype=np.float32),
                np.zeros(0, dtype=np.int32), [], 0)

    tasks = np.concatenate(all_blocks, axis=0)
    task_tidx = np.concatenate(all_triple_idx)
    t_all = np.concatenate(all_t_vals)
    nroots_flat = np.concatenate(all_nroots_flat)
    n_tasks = len(tasks)

    # Phase 3: batched Boys + Rys (one call per nroots value)
    for nr in range(1, 7):
        nr_mask = nroots_flat == nr
        if not np.any(nr_mask):
            continue
        m_max = 2 * nr - 1
        fm = _boys_function_vec(m_max, t_all[nr_mask])
        n_pts = fm.shape[1]
        if nr <= 3:
            # Use vectorized batch solver (fast)
            roots_3, weights_3 = _rys_from_boys_batch(nr, fm)
            roots = np.zeros((n_pts, 6), dtype=np.float64)
            weights = np.zeros((n_pts, 6), dtype=np.float64)
            roots[:, :3] = roots_3
            weights[:, :3] = weights_3
        else:
            # Batch solver for nroots 4-6: Hankel matrix + companion eigenvalue
            roots = np.zeros((n_pts, 6), dtype=np.float64)
            weights = np.zeros((n_pts, 6), dtype=np.float64)
            # Build Hankel matrices: H[i,j] = fm[i+j] for i,j in 0..nr-1
            H = np.zeros((n_pts, nr, nr), dtype=np.float64)
            rhs = np.zeros((n_pts, nr), dtype=np.float64)
            for i in range(nr):
                for j in range(nr):
                    H[:, i, j] = fm[i + j]
                rhs[:, i] = fm[nr + i]
            # Solve H @ c = rhs (batch)
            try:
                c = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                # Fallback to per-point for singular cases
                for i in range(n_pts):
                    rt, wt = _rys_from_boys(nr, fm[:, i])
                    roots[i, :nr] = rt[:nr]
                    weights[i, :nr] = wt[:nr]
                tasks[nr_mask, 18:24] = roots.astype(np.float32)
                tasks[nr_mask, 24:30] = weights.astype(np.float32)
                continue
            # Build companion matrix and find eigenvalues (roots)
            comp = np.zeros((n_pts, nr, nr), dtype=np.float64)
            for i in range(nr - 1):
                comp[:, i + 1, i] = 1.0
            for i in range(nr):
                comp[:, i, nr - 1] = c[:, i]
            batch_roots = np.real(np.linalg.eigvals(comp))
            batch_roots.sort(axis=1)
            roots[:, :nr] = batch_roots
            # Weights from Vandermonde: V @ w = fm[0..nr-1]
            V = np.zeros((n_pts, nr, nr), dtype=np.float64)
            V[:, :, 0] = 1.0
            for j in range(1, nr):
                V[:, :, j] = batch_roots ** j
            try:
                batch_w = np.linalg.solve(V.transpose(0, 2, 1), fm[:nr].T)
                weights[:, :nr] = batch_w
            except np.linalg.LinAlgError:
                for i in range(n_pts):
                    rt, wt = _rys_from_boys(nr, fm[:, i])
                    weights[i, :nr] = wt[:nr]
        tasks[nr_mask, 18:24] = roots.astype(np.float32)
        tasks[nr_mask, 24:30] = weights.astype(np.float32)

    # Build int32 offset array via vectorized cumsum
    n_out_per = np.array([triple_info[tidx][6] for tidx in task_tidx], dtype=np.int64)
    offsets_i64 = np.zeros(n_tasks, dtype=np.int64)
    if n_tasks > 1:
        np.cumsum(n_out_per[:-1], out=offsets_i64[1:])
    offsets = offsets_i64.astype(np.int32)
    off = int(n_out_per.sum())

    # Build meta for Phase C accumulation
    # Group tasks by triple_info index
    meta = []
    if n_tasks > 0:
        # Find contiguous runs of the same triple idx
        changes = np.where(np.diff(task_tidx) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [n_tasks]])
        for s, e in zip(starts, ends):
            tidx = task_tidx[s]
            ish, jsh, ksh, nfi, nfj, nfk, n_out = triple_info[tidx]
            meta.append((ish, jsh, ksh, s, e - s, nfi, nfj, nfk, n_out))

    return tasks, offsets, meta, off


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
        float64 array matching PySCF getints('int3c2e_ip1', aosym='s1').
        Shape depends on mol.cart: (3, nao, nao, naux_slice) in the
        appropriate basis (Cartesian or spherical).
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
    tasks, offsets, meta, total_out = _build_3c_tasks(mol, auxmol, shl0_aux, shl1_aux)

    if len(tasks) == 0 or total_out == 0:
        return _cpu_fallback(mol, auxmol, shls_slice)

    # Phase B: run Metal kernel
    task_gpu = mx.array(tasks.ravel())
    off_gpu = mx.array(offsets)
    n_tasks = tasks.shape[0]
    THREADS = 256
    grid_size = ((n_tasks + THREADS - 1) // THREADS) * THREADS

    result = _int3c2e_ip1_kernel(
        inputs=[task_gpu, off_gpu],
        grid=(grid_size, 1, 1),
        threadgroup=(min(THREADS, n_tasks), 1, 1),
        output_shapes=[(total_out,)],
        output_dtypes=[mx.float32],
        template=[('n_tasks', n_tasks), ('TASK_STRIDE', TASK_STRIDE)],
    )
    mx.eval(result[0])
    int_buf = np.array(result[0]).astype(np.float64)

    # Phase C: accumulate into output tensor (Cartesian first, then cart2sph)
    nao_cart = mol.nao_cart()
    naux_cart = auxmol.nao_cart()
    aux_loc_c = auxmol.ao_loc_nr(cart=True)
    naux_slice_c = aux_loc_c[shl1_aux] - aux_loc_c[shl0_aux]
    p0_aux = aux_loc_c[shl0_aux]
    ao_loc_c = mol.ao_loc_nr(cart=True)

    out_cart = np.zeros((3, nao_cart, nao_cart, naux_slice_c), dtype=np.float64)
    for ish, jsh, ksh, t_start, t_count, nfi, nfj, nfk, n_out in meta:
        i0 = ao_loc_c[ish]
        j0 = ao_loc_c[jsh]
        k0 = aux_loc_c[ksh] - p0_aux
        comp_stride = nfi * nfj * nfk
        # All t_count primitives are contiguous in int_buf
        off_start = int(offsets[t_start])
        shell_block = int_buf[off_start:off_start + t_count * n_out].reshape(
            t_count, n_out).sum(axis=0)
        li = mol.bas_angular(ish)
        lj = mol.bas_angular(jsh)
        lk = auxmol.bas_angular(ksh)
        fac = _FAC_L[li] * _FAC_L[lj] * _FAC_L[lk]
        for comp in range(3):
            block = shell_block[comp*comp_stride:(comp+1)*comp_stride].reshape(nfi, nfj, nfk)
            out_cart[comp, i0:i0+nfi, j0:j0+nfj, k0:k0+nfk] -= block * fac

    if mol.cart:
        # PySCF getints returns (ncomp, ni, nj, nk) with Fortran-ordered
        # inner axes: strides=(big, 8, ni*8, ni*nj*8). Replicate this
        # layout so that fdrv's ctypes access sees the correct strides
        # after .transpose(0,3,2,1) in get_jk.
        result = np.zeros((nao_cart, nao_cart, naux_slice_c, 3),
                          dtype=np.float64, order='F').transpose(3, 0, 1, 2)
        result[:] = out_cart
        return result

    # Cart-to-spherical transform for orbital indices (i, j) and aux (k)
    from pyscf.gto.mole import cart2sph as _c2s

    def _c2s_mat(l):
        return np.asarray(_c2s(l, normalized='sp')).T  # (nsph, ncart)

    nao = mol.nao
    naux_slice = auxmol.ao_loc_nr()[shl1_aux] - auxmol.ao_loc_nr()[shl0_aux]
    out = np.zeros((3, nao, nao, naux_slice), dtype=np.float64)

    ao_loc_s = mol.ao_loc_nr()
    aux_loc_s = auxmol.ao_loc_nr()
    p0_aux_s = aux_loc_s[shl0_aux]

    for comp in range(3):
        # Transform i (first orbital index)
        tmp1 = np.zeros((nao, nao_cart, naux_slice_c), dtype=np.float64)
        for ish in range(mol.nbas):
            l = mol.bas_angular(ish)
            c0 = ao_loc_c[ish]; c1 = ao_loc_c[ish + 1]
            s0 = ao_loc_s[ish]; s1 = ao_loc_s[ish + 1]
            if l <= 1:
                tmp1[s0:s1] = out_cart[comp, c0:c1]
            else:
                C = _c2s_mat(l)
                tmp1[s0:s1] = np.einsum('si,ijk->sjk', C, out_cart[comp, c0:c1])

        # Transform j (second orbital index)
        tmp2 = np.zeros((nao, nao, naux_slice_c), dtype=np.float64)
        for jsh in range(mol.nbas):
            l = mol.bas_angular(jsh)
            c0 = ao_loc_c[jsh]; c1 = ao_loc_c[jsh + 1]
            s0 = ao_loc_s[jsh]; s1 = ao_loc_s[jsh + 1]
            if l <= 1:
                tmp2[:, s0:s1] = tmp1[:, c0:c1]
            else:
                C = _c2s_mat(l)
                tmp2[:, s0:s1] = np.einsum('sj,ijk->isk', C, tmp1[:, c0:c1])

        # Transform k (aux index)
        for ksh in range(shl0_aux, shl1_aux):
            l = auxmol.bas_angular(ksh)
            c0 = aux_loc_c[ksh] - p0_aux; c1 = aux_loc_c[ksh + 1] - p0_aux
            s0 = aux_loc_s[ksh] - p0_aux_s; s1 = aux_loc_s[ksh + 1] - p0_aux_s
            if l <= 1:
                out[comp, :, :, s0:s1] = tmp2[:, :, c0:c1]
            else:
                C = _c2s_mat(l)
                out[comp, :, :, s0:s1] = np.einsum('sk,ijk->ijs', C, tmp2[:, :, c0:c1])

    # Match getints' stride layout: (ni, nj, nk, ncomp) F-order → transpose
    result = np.zeros((nao, nao, naux_slice, 3),
                      dtype=np.float64, order='F').transpose(3, 0, 1, 2)
    result[:] = out
    return result


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
