"""
Metal GPU direct J/K engine via Rys polynomial quadrature.

Phase 1 (CPU f64): enumerate primitives, Boys function, Rys roots/weights
Phase 2 (Metal f32): TRR + HRR + Cartesian ERI assembly per primitive
Phase 3 (CPU f64): sum primitives per quartet, cart2sph, DM contraction

Rys roots/weights are computed on CPU in f64 (the f32 Hankel solve is
numerically unstable for nroots>=3 at large t). The GPU kernel receives
pre-computed roots/weights and only does the TRR/HRR recurrences,
which are stable in f32.

Each primitive gets its OWN output region — no atomic adds, fully
deterministic. All accumulation happens in f64 on CPU.

Supports s/p/d/f shells (l=0,1,2,3), nroots up to 3.
"""

import numpy as np
import mlx.core as mx
import numba as nb
from math import sqrt, pi
from scipy.special import erf as _erf
from gpu4pyscf.lib.metal_kernels.eval_ao import _cart2sph_matrix
from gpu4pyscf.lib.metal_kernels.rys_jk import _rys_from_boys

# Precomputed lookup tables (avoid 35K+ function calls)
_NCART_LUT = np.array([1, 3, 6, 10], dtype=np.int64)   # l -> ncart
_NSPH_LUT = np.array([1, 3, 5, 7], dtype=np.int64)      # l -> nsph

def _ncart(l):
    return int(_NCART_LUT[l])

def _nsph(l):
    return int(_NSPH_LUT[l])

# ---------------------------------------------------------------------------
# Metal kernel: TRR + HRR + Cartesian ERI output
#
# Each thread receives pre-computed Rys roots/weights from CPU (f64),
# builds the 1D g-values via TRR+HRR, assembles Cartesian ERI block.
# ---------------------------------------------------------------------------

# Task layout (TASK_STRIDE = 30 floats):
#   [0]  aij
#   [1]  akl
#   [2:5]  PA (3 floats)
#   [5:8]  QC (3 floats)
#   [8:11] AB (3 floats)
#   [11:14] CD (3 floats)
#   [14] prefac
#   [15:19] li, lj, lk, ll
#   [19] eri_offset
#   [20] nroots
#   [21:24] roots (3 floats, padded with 0)
#   [24:27] weights (3 floats, padded with 0)
#   [27:30] Rpq (3 floats)
TASK_STRIDE = 30

_RYS_TRR_HEADER = '''
constant int NCART[] = {1, 3, 6, 10};
constant int CART_ALL[] = {
    0,
    16, 4, 1,
    32, 20, 17, 8, 5, 2,
    48, 36, 33, 24, 21, 18, 12, 9, 6, 3
};
constant int CART_OFF[] = {0, 1, 4, 10};
'''

_RYS_TRR_SOURCE = '''
uint tid = thread_position_in_grid.x;
if (tid >= n_tasks) return;

int off = tid * TASK_STRIDE;
float aij    = task_data[off + 0];
float akl    = task_data[off + 1];
float PA_x   = task_data[off + 2];
float PA_y   = task_data[off + 3];
float PA_z   = task_data[off + 4];
float QC_x   = task_data[off + 5];
float QC_y   = task_data[off + 6];
float QC_z   = task_data[off + 7];
float AB_x   = task_data[off + 8];
float AB_y   = task_data[off + 9];
float AB_z   = task_data[off + 10];
float CD_x   = task_data[off + 11];
float CD_y   = task_data[off + 12];
float CD_z   = task_data[off + 13];
float prefac = task_data[off + 14];
int li       = (int)task_data[off + 15];
int lj       = (int)task_data[off + 16];
int lk       = (int)task_data[off + 17];
int ll       = (int)task_data[off + 18];
int eri_off  = (int)task_data[off + 19];
int nroots   = (int)task_data[off + 20];
float rys_roots[3]   = {task_data[off+21], task_data[off+22], task_data[off+23]};
float rys_weights[3] = {task_data[off+24], task_data[off+25], task_data[off+26]};

int lij = li + lj;
int lkl = lk + ll;

int ni = NCART[li], nj = NCART[lj], nk = NCART[lk], nl = NCART[ll];
int ci_off = CART_OFF[li], cj_off = CART_OFF[lj];
int ck_off = CART_OFF[lk], cl_off = CART_OFF[ll];

// Zero this task's ERI region (MLX may reuse output buffers)
int n_eri = ni * nj * nk * nl;
for (int i = 0; i < n_eri; i++)
    eri_out[eri_off + i] = 0.0f;

for (int iroot = 0; iroot < nroots; iroot++) {
float rt = rys_roots[iroot];
float wt = rys_weights[iroot];

float rt_aa  = rt / (aij + akl);
float rt_aij = rt_aa * akl;
float rt_akl = rt_aa * aij;
float b10 = 0.5f / aij * (1.0f - rt_aij);
float b01 = 0.5f / akl * (1.0f - rt_akl);
float b00 = 0.5f * rt_aa;

float PA[3] = {PA_x, PA_y, PA_z};
float QC[3] = {QC_x, QC_y, QC_z};
float AB[3] = {AB_x, AB_y, AB_z};
float CD[3] = {CD_x, CD_y, CD_z};
float Rpq[3] = {PA_x - QC_x + AB_x*0.0f, PA_y - QC_y, PA_z - QC_z};

float g1d_final[3][4][4][4][4];

for (int dir = 0; dir < 3; dir++) {
    float Rpq_dir = task_data[off + 27 + dir];

    float c0 = PA[dir] - rt_aij * Rpq_dir;
    float cp = QC[dir] + rt_akl * Rpq_dir;

    float g[8][8];
    for (int a = 0; a <= lij; a++)
        for (int c = 0; c <= lkl; c++)
            g[a][c] = 0.0f;
    g[0][0] = 1.0f;

    for (int a = 0; a < lij; a++) {
        g[a+1][0] = c0 * g[a][0];
        if (a > 0) g[a+1][0] += a * b10 * g[a-1][0];
    }
    for (int c = 0; c < lkl; c++) {
        for (int a = 0; a <= lij; a++) {
            g[a][c+1] = cp * g[a][c];
            if (c > 0) g[a][c+1] += c * b01 * g[a][c-1];
            if (a > 0) g[a][c+1] += a * b00 * g[a-1][c];
        }
    }

    float g_ij[8][4][8];
    for (int a = 0; a <= lij; a++)
        for (int c = 0; c <= lkl; c++)
            g_ij[a][0][c] = g[a][c];
    for (int j = 0; j < lj; j++)
        for (int c = 0; c <= lkl; c++)
            for (int i = 0; i < lij-j; i++)
                g_ij[i][j+1][c] = g_ij[i+1][j][c] + AB[dir] * g_ij[i][j][c];

    float g_full[4][4][8][4];
    for (int i = 0; i <= li; i++)
        for (int j = 0; j <= lj; j++) {
            for (int c = 0; c <= lkl; c++)
                g_full[i][j][c][0] = g_ij[i][j][c];
            for (int l = 0; l < ll; l++)
                for (int k = 0; k < lkl-l; k++)
                    g_full[i][j][k][l+1] = g_full[i][j][k+1][l] + CD[dir] * g_full[i][j][k][l];
        }

    for (int i = 0; i <= li; i++)
        for (int j = 0; j <= lj; j++)
            for (int k = 0; k <= lk; k++)
                for (int l = 0; l <= ll; l++)
                    g1d_final[dir][i][j][k][l] = g_full[i][j][k][l];
}

float pf_wt = prefac * wt;

for (int ii = 0; ii < ni; ii++) {
    int ci_v = CART_ALL[ci_off + ii];
    int ix_i = ci_v / 16, iy_i = (ci_v / 4) % 4, iz_i = ci_v % 4;
    for (int jj = 0; jj < nj; jj++) {
        int cj_v = CART_ALL[cj_off + jj];
        int ix_j = cj_v / 16, iy_j = (cj_v / 4) % 4, iz_j = cj_v % 4;
        for (int kk = 0; kk < nk; kk++) {
            int ck_v = CART_ALL[ck_off + kk];
            int ix_k = ck_v / 16, iy_k = (ck_v / 4) % 4, iz_k = ck_v % 4;
            for (int ll_idx = 0; ll_idx < nl; ll_idx++) {
                int cl_v = CART_ALL[cl_off + ll_idx];
                int ix_l = cl_v / 16, iy_l = (cl_v / 4) % 4, iz_l = cl_v % 4;
                float eri_val = g1d_final[0][ix_i][ix_j][ix_k][ix_l]
                              * g1d_final[1][iy_i][iy_j][iy_k][iy_l]
                              * g1d_final[2][iz_i][iz_j][iz_k][iz_l]
                              * pf_wt;
                int idx = eri_off + ((ii * nj + jj) * nk + kk) * nl + ll_idx;
                eri_out[idx] += eri_val;
            }
        }
    }
}
} // end root loop
'''

_rys_eri_kernel = mx.fast.metal_kernel(
    name='rys_trr_eri',
    input_names=['task_data'],
    output_names=['eri_out'],
    header=_RYS_TRR_HEADER,
    source=_RYS_TRR_SOURCE,
    atomic_outputs=False,
)


# ---------------------------------------------------------------------------
# Vectorized Boys function and Rys roots
# ---------------------------------------------------------------------------

def _boys_function_vec(m_max, t_values):
    """Vectorized Boys function via erf + upward recursion.

    F_0(t) = sqrt(pi/4) / sqrt(t) * erf(sqrt(t))
    F_{m+1}(t) = ((2m+1)*F_m(t) - exp(-t)) / (2t)
    """
    t = np.asarray(t_values, dtype=np.float64).ravel()
    npts = len(t)
    if npts == 0:
        return np.zeros((m_max + 1, 0))
    result = np.zeros((m_max + 1, npts))
    small = t < 1e-15
    big = ~small
    for m in range(m_max + 1):
        result[m, small] = 1.0 / (2 * m + 1)
    if np.any(big):
        tb = t[big]
        sqt = np.sqrt(tb)
        exp_neg = np.exp(-tb)
        result[0, big] = np.sqrt(np.pi * 0.25) / sqt * _erf(sqt)
        inv2t = 0.5 / tb
        for m in range(m_max):
            result[m + 1, big] = ((2 * m + 1) * result[m, big] - exp_neg) * inv2t
    return result


def _rys_from_boys_batch(nroots, fm):
    """Vectorized Rys roots/weights from Boys moments for nroots=1,2,3.

    Args:
        nroots: int (1, 2, or 3)
        fm: (m_max+1, N) array of Boys function values

    Returns:
        roots: (N, 3) zero-padded
        weights: (N, 3) zero-padded
    """
    N = fm.shape[1]
    roots = np.zeros((N, 3), dtype=np.float64)
    weights = np.zeros((N, 3), dtype=np.float64)
    if N == 0:
        return roots, weights

    if nroots == 1:
        F0, F1 = fm[0], fm[1]
        safe = F0 > 1e-30
        roots[:, 0] = np.where(safe, F1 / np.maximum(F0, 1e-30), 0.0)
        weights[:, 0] = F0

    elif nroots == 2:
        f0, f1, f2, f3 = fm[0], fm[1], fm[2], fm[3]
        det = f0 * f2 - f1 * f1
        safe = np.abs(det) > 1e-30
        det_s = np.where(safe, det, 1.0)
        c0 = np.where(safe, (f2 * f2 - f1 * f3) / det_s, 0.0)
        c1 = np.where(safe, (f0 * f3 - f1 * f2) / det_s, 0.0)
        disc = np.maximum(c1**2 + 4 * c0, 0.0)
        sq = np.sqrt(disc)
        t0 = (c1 - sq) / 2
        t1 = (c1 + sq) / 2
        sep = np.abs(t1 - t0)
        sep_s = np.where(sep > 1e-30, sep, 1.0)
        w1 = np.where(sep > 1e-30, (f1 - f0 * t0) / sep_s, f0 / 2)
        w0 = f0 - w1
        t_fb = np.where(f0 > 1e-30, f1 / np.maximum(f0, 1e-30), 0.0)
        roots[:, 0] = np.where(safe, t0, t_fb)
        roots[:, 1] = np.where(safe, t1, t_fb)
        weights[:, 0] = np.where(safe, w0, f0 / 2)
        weights[:, 1] = np.where(safe, w1, f0 / 2)

    elif nroots == 3:
        H = np.empty((N, 3, 3))
        H[:, 0, 0] = fm[0]
        H[:, 0, 1] = fm[1]
        H[:, 0, 2] = fm[2]
        H[:, 1, 0] = fm[1]
        H[:, 1, 1] = fm[2]
        H[:, 1, 2] = fm[3]
        H[:, 2, 0] = fm[2]
        H[:, 2, 1] = fm[3]
        H[:, 2, 2] = fm[4]
        rhs_h = np.stack([fm[3], fm[4], fm[5]], axis=1)
        dets = np.linalg.det(H)
        good = np.abs(dets) > 1e-30
        bad = ~good
        if np.any(bad):
            t_fb = np.where(fm[0, bad] > 1e-30,
                            fm[1, bad] / np.maximum(fm[0, bad], 1e-30), 0.0)
            roots[bad, 0] = roots[bad, 1] = roots[bad, 2] = t_fb
            weights[bad, 0] = weights[bad, 1] = weights[bad, 2] = fm[0, bad] / 3
        if np.any(good):
            c = np.linalg.solve(H[good], rhs_h[good])
            n_g = c.shape[0]
            comp = np.zeros((n_g, 3, 3))
            comp[:, 1, 0] = 1.0
            comp[:, 2, 1] = 1.0
            comp[:, 0, 2] = c[:, 0]
            comp[:, 1, 2] = c[:, 1]
            comp[:, 2, 2] = c[:, 2]
            r_g = np.real(np.linalg.eigvals(comp))
            r_g.sort(axis=1)
            roots[good, :3] = r_g
            V = np.zeros((n_g, 3, 3))
            V[:, :, 0] = 1.0
            V[:, :, 1] = r_g
            V[:, :, 2] = r_g ** 2
            VT = V.transpose(0, 2, 1)
            fm_rhs = fm[:3, good].T
            vdets = np.linalg.det(VT)
            vgood = np.abs(vdets) > 1e-30
            good_idx = np.where(good)[0]
            if np.any(vgood):
                weights[good_idx[vgood], :3] = np.linalg.solve(
                    VT[vgood], fm_rhs[vgood])
            if np.any(~vgood):
                for gi in good_idx[~vgood]:
                    rt, wt = _rys_from_boys(3, fm[:, gi])
                    roots[gi, :3] = rt
                    weights[gi, :3] = wt

    return roots, weights


# ---------------------------------------------------------------------------
# Schwarz screening (cached on mol)
# ---------------------------------------------------------------------------

def _schwarz_bounds(mol, nbas):
    """Schwarz screening bounds Q[ish,jsh] = sqrt(max|(ij|ij)|), cached."""
    if hasattr(mol, '_schwarz_Q'):
        return mol._schwarz_Q
    Q = np.zeros((nbas, nbas))
    for ish in range(nbas):
        for jsh in range(ish, nbas):
            eri = mol.intor('int2e', shls_slice=(
                ish, ish + 1, jsh, jsh + 1, ish, ish + 1, jsh, jsh + 1))
            Q[ish, jsh] = Q[jsh, ish] = np.sqrt(np.max(np.abs(eri)))
    mol._schwarz_Q = Q
    return Q


# ---------------------------------------------------------------------------
# Shell pair pre-computation
# ---------------------------------------------------------------------------

def _precompute_shell_pairs(nbas, shell_l, shell_R, shell_exps, shell_coeffs):
    """Pre-compute primitive pair data for all unique shell pairs."""
    pair_cache = {}
    for ish in range(nbas):
        li = shell_l[ish]
        if li > 3:
            continue
        Ri = shell_R[ish]
        ai = shell_exps[ish]
        ci = shell_coeffs[ish]
        for jsh in range(ish + 1):
            lj = shell_l[jsh]
            if lj > 3:
                continue
            Rj = shell_R[jsh]
            aj = shell_exps[jsh]
            cj = shell_coeffs[jsh]
            ni, nj = len(ai), len(aj)
            ai2 = np.repeat(ai, nj)
            aj2 = np.tile(aj, ni)
            ci2 = np.repeat(ci, nj)
            cj2 = np.tile(cj, ni)
            aij = ai2 + aj2
            rr = np.dot(Ri - Rj, Ri - Rj)
            Kab = np.exp(-ai2 * aj2 / aij * rr)
            Pij = (ai2[:, None] * Ri + aj2[:, None] * Rj) / aij[:, None]
            PA = Pij - Ri
            coeff_K = ci2 * cj2 * Kab
            AB = Ri - Rj
            pair_cache[(ish, jsh)] = (aij, PA, Pij, coeff_K, AB)
    return pair_cache


# ---------------------------------------------------------------------------
# Task generation with 8-fold symmetry and batched Rys roots
# ---------------------------------------------------------------------------

def _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, schwarz_Q,
                            schwarz_thresh=1e-10, max_nroots=3):
    """Build per-primitive GPU tasks with 8-fold symmetry and grouped batching.

    Three phases:
      1. Light enumeration of valid quartets (Python loop, minimal work)
      2. Group by (n_ij, n_kl) and batch cross-product expansion per group
      3. Batch Boys/Rys across all primitives (3 calls total)
    """
    shell_l = np.array([mol.bas_angular(i) for i in range(nbas)])
    shell_atom = np.array([mol.bas_atom(i) for i in range(nbas)])
    atom_coords = np.array([mol.atom_coord(i) for i in range(mol.natm)])
    shell_R = atom_coords[shell_atom]
    shell_i0 = np.array([ao_loc[i] for i in range(nbas)])
    shell_exps = [mol.bas_exp(i) for i in range(nbas)]
    shell_coeffs = [mol._libcint_ctr_coeff(i).flatten() for i in range(nbas)]
    Q = schwarz_Q

    pair_cache = _precompute_shell_pairs(
        nbas, shell_l, shell_R, shell_exps, shell_coeffs)

    # Phase 1: enumerate valid quartets (lightweight)
    valid_q = []
    cpu_quartets = []
    for ish in range(nbas):
        li = shell_l[ish]
        if li > 3:
            continue
        for jsh in range(ish + 1):
            lj = shell_l[jsh]
            if lj > 3:
                continue
            ij = ish * (ish + 1) // 2 + jsh
            for ksh in range(nbas):
                lk = shell_l[ksh]
                if lk > 3:
                    continue
                for lsh in range(ksh + 1):
                    ll = shell_l[lsh]
                    if ll > 3:
                        continue
                    kl = ksh * (ksh + 1) // 2 + lsh
                    if kl > ij:
                        continue
                    if Q[ish, jsh] * Q[ksh, lsh] < schwarz_thresh:
                        continue
                    nroots = (li + lj + lk + ll) // 2 + 1
                    if nroots > max_nroots:
                        cpu_quartets.append((ish, jsh, ksh, lsh))
                        continue
                    valid_q.append((ish, jsh, ksh, lsh,
                                    li, lj, lk, ll, nroots))

    if not valid_q:
        return (np.zeros((0, TASK_STRIDE), dtype=np.float32),
                np.zeros(0, dtype=np.int32), [], 0, cpu_quartets)

    # Phase 2: group by (n_ij, n_kl) for batched expansion
    groups = {}
    for q in valid_q:
        n_ij = len(pair_cache[(q[0], q[1])][0])
        n_kl = len(pair_cache[(q[2], q[3])][0])
        groups.setdefault((n_ij, n_kl), []).append(q)

    all_blocks = []
    all_t_vals = []
    all_nroots_flat = []
    all_qidx = []
    quartet_info = []
    eri_offset = 0

    for (n_ij, n_kl), gq_list in groups.items():
        n_q = len(gq_list)
        n_prim = n_ij * n_kl

        ij_keys = [(g[0], g[1]) for g in gq_list]
        kl_keys = [(g[2], g[3]) for g in gq_list]

        # Stack pair data: (n_q, n_pair_prim[, 3])
        aij_b = np.array([pair_cache[k][0] for k in ij_keys])
        PA_b  = np.array([pair_cache[k][1] for k in ij_keys])
        Pij_b = np.array([pair_cache[k][2] for k in ij_keys])
        cK_ij = np.array([pair_cache[k][3] for k in ij_keys])
        AB_b  = np.array([pair_cache[k][4] for k in ij_keys])

        akl_b = np.array([pair_cache[k][0] for k in kl_keys])
        QC_b  = np.array([pair_cache[k][1] for k in kl_keys])
        Pkl_b = np.array([pair_cache[k][2] for k in kl_keys])
        cK_kl = np.array([pair_cache[k][3] for k in kl_keys])
        CD_b  = np.array([pair_cache[k][4] for k in kl_keys])

        # Batched cross-product: (n_q, n_prim[, 3])
        aij_f  = np.repeat(aij_b, n_kl, axis=1)
        akl_f  = np.tile(akl_b, (1, n_ij))
        PA_f   = np.repeat(PA_b, n_kl, axis=1)
        QC_f   = np.tile(QC_b, (1, n_ij, 1))
        Rpq_f  = np.repeat(Pij_b, n_kl, axis=1) - np.tile(Pkl_b, (1, n_ij, 1))
        coeff_f = np.repeat(cK_ij, n_kl, axis=1) * np.tile(cK_kl, (1, n_ij))

        # Prefactor: (n_q, n_prim)
        prefac_f = coeff_f * 2.0 * pi**2.5 / (
            aij_f * akl_f * np.sqrt(aij_f + akl_f))
        fac_arr = np.array([
            fac_l.get(g[4], 1.0) * fac_l.get(g[5], 1.0)
            * fac_l.get(g[6], 1.0) * fac_l.get(g[7], 1.0)
            for g in gq_list])
        prefac_f *= fac_arr[:, None]

        # t values: (n_q, n_prim)
        theta = aij_f * akl_f / (aij_f + akl_f)
        t_vals_f = theta * np.sum(Rpq_f**2, axis=2)

        # Per-quartet metadata
        li_a = np.array([g[4] for g in gq_list], dtype=np.float32)
        lj_a = np.array([g[5] for g in gq_list], dtype=np.float32)
        lk_a = np.array([g[6] for g in gq_list], dtype=np.float32)
        ll_a = np.array([g[7] for g in gq_list], dtype=np.float32)
        nr_a = np.array([g[8] for g in gq_list], dtype=np.float32)
        ncarts = np.array([_ncart(g[4]) * _ncart(g[5])
                           * _ncart(g[6]) * _ncart(g[7])
                           for g in gq_list], dtype=np.int64)

        # ERI offsets: (n_q, n_prim)
        eri_sizes = ncarts * n_prim
        eri_bases = np.empty(n_q, dtype=np.int64)
        eri_bases[0] = eri_offset
        if n_q > 1:
            np.cumsum(eri_sizes[:-1], out=eri_bases[1:])
            eri_bases[1:] += eri_offset
        prim_idx = np.arange(n_prim)
        eri_off_f = eri_bases[:, None] + prim_idx[None, :] * ncarts[:, None]
        eri_offset += int(eri_sizes.sum())

        # Build task block: (n_q * n_prim, TASK_STRIDE)
        total = n_q * n_prim
        block = np.zeros((total, TASK_STRIDE), dtype=np.float32)
        block[:, 0]    = aij_f.ravel()
        block[:, 1]    = akl_f.ravel()
        block[:, 2:5]  = PA_f.reshape(total, 3)
        block[:, 5:8]  = QC_f.reshape(total, 3)
        block[:, 8:11] = np.repeat(AB_b, n_prim, axis=0)
        block[:, 11:14] = np.repeat(CD_b, n_prim, axis=0)
        block[:, 14]   = prefac_f.ravel()
        block[:, 15]   = np.repeat(li_a, n_prim)
        block[:, 16]   = np.repeat(lj_a, n_prim)
        block[:, 17]   = np.repeat(lk_a, n_prim)
        block[:, 18]   = np.repeat(ll_a, n_prim)
        block[:, 19]   = eri_off_f.ravel().astype(np.float32)
        block[:, 20]   = np.repeat(nr_a, n_prim)
        block[:, 27:30] = Rpq_f.reshape(total, 3).astype(np.float32)

        all_blocks.append(block)
        all_t_vals.append(t_vals_f.ravel())
        all_nroots_flat.append(
            np.repeat(nr_a.astype(np.int32), n_prim))

        # Quartet info and task-to-quartet mapping
        q_start = len(quartet_info)
        for g in gq_list:
            ish, jsh, ksh, lsh, li, lj, lk, ll, _nr = g
            quartet_info.append((
                shell_i0[ish], shell_i0[jsh],
                shell_i0[ksh], shell_i0[lsh],
                li, lj, lk, ll,
                ish, jsh, ksh, lsh,
            ))
        all_qidx.append(np.repeat(
            np.arange(q_start, q_start + n_q, dtype=np.int32), n_prim))

    tasks = np.concatenate(all_blocks, axis=0)
    task_qidx = np.concatenate(all_qidx)
    t_all = np.concatenate(all_t_vals)
    nroots_flat = np.concatenate(all_nroots_flat)

    # Sort by quartet index so accumulation can use contiguous ranges
    sort_idx = np.argsort(task_qidx, kind='mergesort')
    tasks = tasks[sort_idx]
    task_qidx = task_qidx[sort_idx]
    t_all = t_all[sort_idx]
    nroots_flat = nroots_flat[sort_idx]

    # Phase 3: batched Boys + Rys (3 calls instead of ~3000)
    for nr in (1, 2, 3):
        nr_mask = nroots_flat == nr
        if not np.any(nr_mask):
            continue
        m_max = 2 * nr - 1
        fm = _boys_function_vec(m_max, t_all[nr_mask])
        roots, weights = _rys_from_boys_batch(nr, fm)
        tasks[nr_mask, 21:24] = roots.astype(np.float32)
        tasks[nr_mask, 24:27] = weights.astype(np.float32)

    return tasks, task_qidx, quartet_info, eri_offset, cpu_quartets


# ---------------------------------------------------------------------------
# Cart-to-spherical with caching
# ---------------------------------------------------------------------------

_c2s_cache = {}


def _cart2sph_eri(eri_cart, li, lj, lk, ll):
    """Transform a Cartesian ERI block to spherical harmonics."""
    eri = eri_cart
    if li >= 2:
        if li not in _c2s_cache:
            _c2s_cache[li] = _cart2sph_matrix(li)
        eri = np.einsum('ip,ijkl->pjkl', _c2s_cache[li], eri)
    if lj >= 2:
        if lj not in _c2s_cache:
            _c2s_cache[lj] = _cart2sph_matrix(lj)
        eri = np.einsum('jq,pjkl->pqkl', _c2s_cache[lj], eri)
    if lk >= 2:
        if lk not in _c2s_cache:
            _c2s_cache[lk] = _cart2sph_matrix(lk)
        eri = np.einsum('kr,pqkl->pqrl', _c2s_cache[lk], eri)
    if ll >= 2:
        if ll not in _c2s_cache:
            _c2s_cache[ll] = _cart2sph_matrix(ll)
        eri = np.einsum('ls,pqrl->pqrs', _c2s_cache[ll], eri)
    return eri


@nb.njit(cache=True)
def _sum_primitives_numba(eri_buf, task_eri_offsets,
                          q_starts, q_counts, ncarts, n_q):
    """Numba-compiled primitive ERI summation per quartet."""
    total = 0
    for q in range(n_q):
        total += ncarts[q]
    eri_flat = np.zeros(total, dtype=np.float64)
    out_off = 0
    for q in range(n_q):
        nc = ncarts[q]
        s = q_starts[q]
        cnt = q_counts[q]
        for t in range(cnt):
            off = task_eri_offsets[s + t]
            for x in range(nc):
                eri_flat[out_off + x] += eri_buf[off + x]
        out_off += nc
    return eri_flat


# ---------------------------------------------------------------------------
# 8-fold symmetry J/K accumulation (Numba JIT)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _accumulate_jk_numba(vj, vk, dm, eri_flat, eri_off,
                         q_i0, q_j0, q_k0, q_l0,
                         q_ni, q_nj, q_nk, q_nl,
                         q_ish, q_jsh, q_ksh, q_lsh,
                         do_j, do_k):
    """Numba-compiled J/K accumulation with 8-fold symmetry unfolding."""
    n_q = len(q_i0)
    for q in range(n_q):
        i0 = q_i0[q]
        j0 = q_j0[q]
        k0 = q_k0[q]
        l0 = q_l0[q]
        ni = q_ni[q]
        nj = q_nj[q]
        nk = q_nk[q]
        nl = q_nl[q]
        ish = q_ish[q]
        jsh = q_jsh[q]
        ksh = q_ksh[q]
        lsh = q_lsh[q]
        ij = ish * (ish + 1) // 2 + jsh
        kl = ksh * (ksh + 1) // 2 + lsh
        off = eri_off[q]
        fkl = 2 if ksh != lsh else 1
        fij = 2 if ish != jsh else 1

        if do_j:
            for i in range(ni):
                for j in range(nj):
                    s = 0.0
                    for k in range(nk):
                        for l in range(nl):
                            s += dm[k0+k, l0+l] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                    g = fkl * s
                    vj[i0+i, j0+j] += g
                    if ish != jsh:
                        vj[j0+j, i0+i] += g
            if ij != kl:
                for k in range(nk):
                    for l in range(nl):
                        s = 0.0
                        for i in range(ni):
                            for j in range(nj):
                                s += dm[i0+i, j0+j] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                        g = fij * s
                        vj[k0+k, l0+l] += g
                        if ksh != lsh:
                            vj[l0+l, k0+k] += g

        if do_k:
            for i in range(ni):
                for k in range(nk):
                    s = 0.0
                    for j in range(nj):
                        for l in range(nl):
                            s += dm[j0+j, l0+l] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                    vk[i0+i, k0+k] += s
                    if ij != kl:
                        vk[k0+k, i0+i] += s
            if ksh != lsh:
                for i in range(ni):
                    for l in range(nl):
                        s = 0.0
                        for j in range(nj):
                            for k in range(nk):
                                s += dm[j0+j, k0+k] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                        vk[i0+i, l0+l] += s
                        if ij != kl:
                            vk[l0+l, i0+i] += s
            if ish != jsh:
                for j in range(nj):
                    for k in range(nk):
                        s = 0.0
                        for i in range(ni):
                            for l in range(nl):
                                s += dm[i0+i, l0+l] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                        vk[j0+j, k0+k] += s
                        if ij != kl:
                            vk[k0+k, j0+j] += s
                if ksh != lsh:
                    for j in range(nj):
                        for l in range(nl):
                            s = 0.0
                            for i in range(ni):
                                for k in range(nk):
                                    s += dm[i0+i, k0+k] * eri_flat[off+((i*nj+j)*nk+k)*nl+l]
                            vk[j0+j, l0+l] += s
                            if ij != kl:
                                vk[l0+l, j0+j] += s


def _accumulate_jk_sym(vj, vk, dm, eri,
                       i0, ni, j0, nj, k0, nk, l0, nl,
                       ish, jsh, ksh, lsh, with_j, with_k):
    """Python fallback for _accumulate_jk_numba (used by CPU quartets)."""
    ie, je, ke, le = i0 + ni, j0 + nj, k0 + nk, l0 + nl
    ij = ish * (ish + 1) // 2 + jsh
    kl = ksh * (ksh + 1) // 2 + lsh

    if with_j:
        fkl = 2 if ksh != lsh else 1
        gij = fkl * np.einsum('kl,ijkl->ij', dm[k0:ke, l0:le], eri)
        vj[i0:ie, j0:je] += gij
        if ish != jsh:
            vj[j0:je, i0:ie] += gij.T
        if ij != kl:
            fij = 2 if ish != jsh else 1
            gkl = fij * np.einsum('ij,ijkl->kl', dm[i0:ie, j0:je], eri)
            vj[k0:ke, l0:le] += gkl
            if ksh != lsh:
                vj[l0:le, k0:ke] += gkl.T

    if with_k:
        k_ik = np.einsum('jl,ijkl->ik', dm[j0:je, l0:le], eri)
        k_il = (np.einsum('jk,ijkl->il', dm[j0:je, k0:ke], eri)
                if ksh != lsh else None)
        k_jk = (np.einsum('il,ijkl->jk', dm[i0:ie, l0:le], eri)
                if ish != jsh else None)
        k_jl = (np.einsum('ik,ijkl->jl', dm[i0:ie, k0:ke], eri)
                if ish != jsh and ksh != lsh else None)

        vk[i0:ie, k0:ke] += k_ik
        if k_il is not None:
            vk[i0:ie, l0:le] += k_il
        if k_jk is not None:
            vk[j0:je, k0:ke] += k_jk
        if k_jl is not None:
            vk[j0:je, l0:le] += k_jl

        if ij != kl:
            vk[k0:ke, i0:ie] += k_ik.T
            if k_il is not None:
                vk[l0:le, i0:ie] += k_il.T
            if k_jk is not None:
                vk[k0:ke, j0:je] += k_jk.T
            if k_jl is not None:
                vk[l0:le, j0:je] += k_jl.T


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_jk_rys_metal(mol, dm, with_j=True, with_k=True):
    """Direct J/K via Rys quadrature: f64 roots + f32 GPU TRR + f64 accumulation.

    Uses 8-fold ERI symmetry to reduce work ~8x.
    """
    nao = mol.nao
    dm = np.asarray(dm, dtype=np.float64)
    nbas = mol.nbas
    ao_loc = mol.ao_loc_nr()
    fac_l = {0: sqrt(1.0 / (4 * pi)), 1: sqrt(3.0 / (4 * pi))}

    Q = _schwarz_bounds(mol, nbas)

    tasks, task_qidx, quartet_info, total_eri_size, cpu_quartets = \
        _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, Q,
                                schwarz_thresh=1e-10)

    vj = np.zeros((nao, nao), dtype=np.float64) if with_j else None
    vk = np.zeros((nao, nao), dtype=np.float64) if with_k else None

    # Phase 2: Metal GPU kernel for nroots<=3 tasks
    if len(tasks) > 0:
        if total_eri_size == 0:
            total_eri_size = 1

        task_gpu = mx.array(tasks.ravel())
        n_tasks = tasks.shape[0]

        THREADS = 256
        grid_size = ((n_tasks + THREADS - 1) // THREADS) * THREADS

        result = _rys_eri_kernel(
            inputs=[task_gpu],
            grid=(grid_size, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(total_eri_size,)],
            output_dtypes=[mx.float32],
            template=[('n_tasks', n_tasks), ('TASK_STRIDE', TASK_STRIDE)],
        )
        mx.eval(result[0])

        eri_buf = np.array(result[0]).astype(np.float64)

        # Phase 3a: Numba primitive sum + cart2sph + Numba J/K accumulation
        task_eri_offsets = tasks[:, 19].astype(np.int64)

        n_q = len(quartet_info)
        q_counts = np.bincount(task_qidx, minlength=n_q)
        q_starts = np.empty(n_q + 1, dtype=np.int64)
        q_starts[0] = 0
        np.cumsum(q_counts, out=q_starts[1:])

        # Pre-compute per-quartet metadata arrays
        q_meta = np.empty((n_q, 12), dtype=np.int64)
        q_li = np.empty(n_q, dtype=np.int64)
        q_lj = np.empty(n_q, dtype=np.int64)
        q_lk = np.empty(n_q, dtype=np.int64)
        q_ll = np.empty(n_q, dtype=np.int64)
        for q_idx in range(n_q):
            qi = quartet_info[q_idx]
            q_meta[q_idx] = qi
            q_li[q_idx] = qi[4]
            q_lj[q_idx] = qi[5]
            q_lk[q_idx] = qi[6]
            q_ll[q_idx] = qi[7]
        ncarts = (_NCART_LUT[q_li] * _NCART_LUT[q_lj]
                  * _NCART_LUT[q_lk] * _NCART_LUT[q_ll])
        nsphs = (_NSPH_LUT[q_li] * _NSPH_LUT[q_lj]
                 * _NSPH_LUT[q_lk] * _NSPH_LUT[q_ll])

        # Numba primitive summing (replaces per-quartet fancy indexing)
        eri_summed = _sum_primitives_numba(
            eri_buf, task_eri_offsets, q_starts[:-1],
            q_counts.astype(np.int64), ncarts, n_q)

        # Cart2sph + pack into pre-allocated flat array
        total_sph = int(nsphs.sum())
        eri_sph_flat = np.empty(total_sph, dtype=np.float64)
        eri_sph_off = np.zeros(n_q, dtype=np.int64)
        if n_q > 1:
            np.cumsum(nsphs[:-1], out=eri_sph_off[1:])

        cart_off = 0
        for q_idx in range(n_q):
            li, lj, lk, ll = int(q_li[q_idx]), int(q_lj[q_idx]), \
                              int(q_lk[q_idx]), int(q_ll[q_idx])
            nc = int(ncarts[q_idx])
            ni_c = int(_NCART_LUT[li])
            nj_c = int(_NCART_LUT[lj])
            nk_c = int(_NCART_LUT[lk])
            nl_c = int(_NCART_LUT[ll])
            eri_cart = eri_summed[cart_off:cart_off + nc].reshape(
                ni_c, nj_c, nk_c, nl_c)
            eri = _cart2sph_eri(eri_cart, li, lj, lk, ll)
            ns = int(nsphs[q_idx])
            eri_sph_flat[eri_sph_off[q_idx]:eri_sph_off[q_idx] + ns] = \
                eri.ravel()
            cart_off += nc
            q_meta[q_idx, 4:8] = [_NSPH_LUT[li], _NSPH_LUT[lj],
                                   _NSPH_LUT[lk], _NSPH_LUT[ll]]

        _vj = vj if vj is not None else np.zeros((nao, nao))
        _vk = vk if vk is not None else np.zeros((nao, nao))
        _accumulate_jk_numba(
            _vj, _vk, dm, eri_sph_flat, eri_sph_off,
            q_meta[:, 0], q_meta[:, 1], q_meta[:, 2], q_meta[:, 3],
            q_meta[:, 4], q_meta[:, 5], q_meta[:, 6], q_meta[:, 7],
            q_meta[:, 8], q_meta[:, 9], q_meta[:, 10], q_meta[:, 11],
            with_j, with_k)
        if with_j:
            vj[:] = _vj
        if with_k:
            vk[:] = _vk

    # Phase 3b: CPU fallback for nroots > max_nroots (with symmetry)
    if cpu_quartets:
        _accumulate_cpu_quartets(mol, dm, ao_loc, cpu_quartets,
                                 vj, vk, with_j, with_k)

    return vj, vk


def _accumulate_cpu_quartets(mol, dm, ao_loc, cpu_quartets,
                             vj, vk, with_j, with_k):
    """Compute ERIs via PySCF libcint for shell quartets needing nroots > 3."""
    for (ish, jsh, ksh, lsh) in cpu_quartets:
        eri = mol.intor('int2e', shls_slice=(
            ish, ish + 1, jsh, jsh + 1, ksh, ksh + 1, lsh, lsh + 1))

        i0 = ao_loc[ish]
        j0 = ao_loc[jsh]
        k0 = ao_loc[ksh]
        l0 = ao_loc[lsh]
        ni = ao_loc[ish + 1] - i0
        nj = ao_loc[jsh + 1] - j0
        nk = ao_loc[ksh + 1] - k0
        nl = ao_loc[lsh + 1] - l0
        eri = eri.reshape(ni, nj, nk, nl)

        _accumulate_jk_sym(vj, vk, dm, eri,
                           i0, ni, j0, nj, k0, nk, l0, nl,
                           ish, jsh, ksh, lsh, with_j, with_k)
