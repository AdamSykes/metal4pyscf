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
from math import sqrt, pi
from gpu4pyscf.lib.metal_kernels.eval_ao import _ncart, _cart2sph_matrix
from gpu4pyscf.lib.metal_kernels.rys_jk import boys_function, _rys_from_boys

# ---------------------------------------------------------------------------
# Metal kernel: TRR + HRR + Cartesian ERI output
#
# Each thread receives pre-computed Rys roots/weights from CPU (f64),
# builds the 1D g-values via TRR+HRR, assembles Cartesian ERI block.
# ---------------------------------------------------------------------------

# Task layout (TASK_STRIDE = 29 floats):
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
    // Rys-modified recurrence centres
    // P-A already in PA. Need Rpq = P - Q.
    // We reconstruct Rpq from PA and QC:
    //   Rpq = P - Q, PA = P - A, QC = Q - C
    //   We need Rpq_dir for TRR, but it was removed from task data.
    //   Actually Rpq is implicit: c0 = PA - rt_aij * Rpq, cp = QC + rt_akl * Rpq.
    //   We need Rpq. But we removed it to save task stride.
    //   Actually I can pass Rpq via the reserved slots!
    // WORKAROUND: reconstruct from PA, QC, AB, CD
    // P = A + PA, Q = C + QC
    // A = arbitrary reference, C = arbitrary reference
    // Rpq = P - Q = A + PA - C - QC
    // But we don't have A or C separately!
    // The correct approach: store Rpq in the task data.

    // Actually, we DO need Rpq. Let me use the reserved slots [27:30].
    // For now, I'll read them from task_data:
    float Rpq_dir = task_data[off + 27 + dir];  // Rpq stored at [27:30]

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


def _schwarz_bounds(mol, nbas):
    """Precompute Schwarz screening bounds Q[ish,jsh] = sqrt(max|(ij|ij)|)."""
    Q = np.zeros((nbas, nbas))
    for ish in range(nbas):
        for jsh in range(ish, nbas):
            eri = mol.intor('int2e', shls_slice=(
                ish, ish + 1, jsh, jsh + 1, ish, ish + 1, jsh, jsh + 1))
            Q[ish, jsh] = Q[jsh, ish] = np.sqrt(np.max(np.abs(eri)))
    return Q


def _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, schwarz_thresh=1e-10,
                            max_nroots=3):
    """Build per-primitive tasks with CPU-computed Rys roots/weights.

    Returns:
        tasks: (N, TASK_STRIDE) float32 array for GPU
        task_quartet_idx: (N,) int array mapping task → quartet
        quartet_info: list of (i0, j0, k0, l0, li, lj, lk, ll) tuples
        total_eri_size: total floats in the ERI output buffer
        cpu_quartets: shell quartets needing nroots > max_nroots
    """
    shell_l = np.array([mol.bas_angular(i) for i in range(nbas)])
    shell_atom = np.array([mol.bas_atom(i) for i in range(nbas)])
    atom_coords = np.array([mol.atom_coord(i) for i in range(mol.natm)])
    shell_R = atom_coords[shell_atom]
    shell_i0 = np.array([ao_loc[i] for i in range(nbas)])

    shell_exps = [mol.bas_exp(i) for i in range(nbas)]
    shell_coeffs = [mol._libcint_ctr_coeff(i).flatten() for i in range(nbas)]

    Q = _schwarz_bounds(mol, nbas)

    all_tasks = []
    all_quartet_idx = []
    quartet_info = []
    cpu_quartets = []
    eri_offset = 0

    for ish in range(nbas):
        li = shell_l[ish]
        if li > 3: continue
        Ri = shell_R[ish]
        ai_arr = shell_exps[ish]
        ci_arr = shell_coeffs[ish]

        for jsh in range(nbas):
            lj = shell_l[jsh]
            if lj > 3: continue
            Rj = shell_R[jsh]
            aj_arr = shell_exps[jsh]
            cj_arr = shell_coeffs[jsh]
            rr_ij = np.dot(Ri - Rj, Ri - Rj)
            AB = Ri - Rj

            ni_p, nj_p = len(ai_arr), len(aj_arr)
            ai_2d = np.repeat(ai_arr, nj_p)
            aj_2d = np.tile(aj_arr, ni_p)
            ci_2d = np.repeat(ci_arr, nj_p)
            cj_2d = np.tile(cj_arr, ni_p)
            aij = ai_2d + aj_2d
            Kab = np.exp(-ai_2d * aj_2d / aij * rr_ij)
            Pij = (ai_2d[:, None] * Ri + aj_2d[:, None] * Rj) / aij[:, None]
            PA = Pij - Ri
            cicj_Kab = ci_2d * cj_2d * Kab

            for ksh in range(nbas):
                lk = shell_l[ksh]
                if lk > 3: continue
                Rk = shell_R[ksh]
                ak_arr = shell_exps[ksh]
                ck_arr = shell_coeffs[ksh]

                for lsh in range(nbas):
                    ll = shell_l[lsh]
                    if ll > 3: continue

                    if Q[ish, jsh] * Q[ksh, lsh] < schwarz_thresh:
                        continue

                    nroots = (li + lj + lk + ll) // 2 + 1
                    if nroots > max_nroots:
                        cpu_quartets.append((ish, jsh, ksh, lsh))
                        continue

                    Rl = shell_R[lsh]
                    al_arr = shell_exps[lsh]
                    cl_arr = shell_coeffs[lsh]
                    rr_kl = np.dot(Rk - Rl, Rk - Rl)
                    CD = Rk - Rl

                    nk_p, nl_p = len(ak_arr), len(al_arr)
                    ak_2d = np.repeat(ak_arr, nl_p)
                    al_2d = np.tile(al_arr, nk_p)
                    ck_2d = np.repeat(ck_arr, nl_p)
                    cl_2d = np.tile(cl_arr, nk_p)
                    akl = ak_2d + al_2d
                    Kcd = np.exp(-ak_2d * al_2d / akl * rr_kl)
                    Pkl = (ak_2d[:, None] * Rk + al_2d[:, None] * Rl) / akl[:, None]
                    QC_arr = Pkl - Rk
                    ckcl_Kcd = ck_2d * cl_2d * Kcd

                    n_ij = ni_p * nj_p
                    n_kl = nk_p * nl_p
                    aij_all = np.repeat(aij, n_kl)
                    akl_all = np.tile(akl, n_ij)
                    PA_all = np.repeat(PA, n_kl, axis=0)
                    QC_all = np.tile(QC_arr, (n_ij, 1))
                    Pij_all = np.repeat(Pij, n_kl, axis=0)
                    Pkl_all = np.tile(Pkl, (n_ij, 1))
                    Rpq_all = Pij_all - Pkl_all
                    coeff_all = np.repeat(cicj_Kab, n_kl) * np.tile(ckcl_Kcd, n_ij)

                    mask = np.abs(coeff_all) > 1e-14
                    if not np.any(mask):
                        continue

                    aij_m = aij_all[mask]
                    akl_m = akl_all[mask]
                    PA_m = PA_all[mask]
                    QC_m = QC_all[mask]
                    Rpq_m = Rpq_all[mask]
                    coeff_m = coeff_all[mask]

                    prefac = coeff_m * 2.0 * pi**2.5 / (aij_m * akl_m * np.sqrt(aij_m + akl_m))
                    fac = fac_l.get(li, 1.0) * fac_l.get(lj, 1.0) * fac_l.get(lk, 1.0) * fac_l.get(ll, 1.0)
                    prefac *= fac

                    # Compute Rys roots/weights on CPU in f64
                    theta = aij_m * akl_m / (aij_m + akl_m)
                    rr_pq = np.sum(Rpq_m**2, axis=1)
                    t_vals = theta * rr_pq
                    m_max = max(li + lj + lk + ll, 2 * nroots - 1)
                    fm_all = boys_function(m_max, t_vals)  # (m_max+1, ntasks)

                    ntasks = len(prefac)
                    roots_arr = np.zeros((ntasks, 3), dtype=np.float64)
                    weights_arr = np.zeros((ntasks, 3), dtype=np.float64)
                    for t_idx in range(ntasks):
                        rt, wt = _rys_from_boys(nroots, fm_all[:, t_idx])
                        roots_arr[t_idx, :nroots] = rt
                        weights_arr[t_idx, :nroots] = wt

                    ncart = _ncart(li) * _ncart(lj) * _ncart(lk) * _ncart(ll)
                    q_idx = len(quartet_info)
                    quartet_info.append((
                        shell_i0[ish], shell_i0[jsh],
                        shell_i0[ksh], shell_i0[lsh],
                        li, lj, lk, ll,
                    ))

                    block = np.zeros((ntasks, TASK_STRIDE), dtype=np.float32)
                    block[:, 0] = aij_m
                    block[:, 1] = akl_m
                    block[:, 2:5] = PA_m
                    block[:, 5:8] = QC_m
                    block[:, 8] = AB[0]
                    block[:, 9] = AB[1]
                    block[:, 10] = AB[2]
                    block[:, 11] = CD[0]
                    block[:, 12] = CD[1]
                    block[:, 13] = CD[2]
                    block[:, 14] = prefac
                    block[:, 15] = li
                    block[:, 16] = lj
                    block[:, 17] = lk
                    block[:, 18] = ll
                    block[:, 19] = np.arange(ntasks) * ncart + eri_offset
                    block[:, 20] = nroots
                    block[:, 21:24] = roots_arr.astype(np.float32)
                    block[:, 24:27] = weights_arr.astype(np.float32)
                    block[:, 27:30] = Rpq_m.astype(np.float32)  # Rpq for TRR
                    all_tasks.append(block)
                    all_quartet_idx.append(np.full(ntasks, q_idx, dtype=np.int32))

                    eri_offset += ntasks * ncart

    if all_tasks:
        tasks = np.concatenate(all_tasks, axis=0)
        task_qidx = np.concatenate(all_quartet_idx)
    else:
        tasks = np.zeros((0, TASK_STRIDE), dtype=np.float32)
        task_qidx = np.zeros(0, dtype=np.int32)
    return tasks, task_qidx, quartet_info, eri_offset, cpu_quartets


def _cart2sph_eri(eri_cart, li, lj, lk, ll):
    """Transform a Cartesian ERI block to spherical harmonics.

    _cart2sph_matrix(l) returns shape (ncart, nsph).
    """
    eri = eri_cart
    if li >= 2:
        c2s = _cart2sph_matrix(li)          # (ncart_i, nsph_i)
        eri = np.einsum('ip,ijkl->pjkl', c2s, eri)
    if lj >= 2:
        c2s = _cart2sph_matrix(lj)
        eri = np.einsum('jq,pjkl->pqkl', c2s, eri)
    if lk >= 2:
        c2s = _cart2sph_matrix(lk)
        eri = np.einsum('kr,pqkl->pqrl', c2s, eri)
    if ll >= 2:
        c2s = _cart2sph_matrix(ll)
        eri = np.einsum('ls,pqrl->pqrs', c2s, eri)
    return eri


def _nsph(l):
    """Number of spherical harmonics for angular momentum l."""
    return 2 * l + 1 if l >= 2 else _ncart(l)


def get_jk_rys_metal(mol, dm, with_j=True, with_k=True):
    """Direct J/K via Rys quadrature: f64 roots + f32 GPU TRR + f64 accumulation.

    Phase 1 (CPU f64): primitives, Boys function, Rys roots/weights
    Phase 2 (Metal f32): TRR + HRR + per-primitive Cartesian ERI
    Phase 3 (CPU f64): sum primitives per quartet, cart2sph, DM contraction
    """
    nao = mol.nao
    dm = np.asarray(dm, dtype=np.float64)
    nbas = mol.nbas
    ao_loc = mol.ao_loc_nr()
    fac_l = {0: sqrt(1.0 / (4 * pi)), 1: sqrt(3.0 / (4 * pi))}

    tasks, task_qidx, quartet_info, total_eri_size, cpu_quartets = \
        _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, schwarz_thresh=1e-10)

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

        # Phase 3a: CPU f64 accumulation for GPU quartets
        task_eri_offsets = tasks[:, 19].astype(np.int64)

        for q_idx in range(len(quartet_info)):
            i0, j0, k0, l0, li, lj, lk, ll = quartet_info[q_idx]
            ni_c = _ncart(li)
            nj_c = _ncart(lj)
            nk_c = _ncart(lk)
            nl_c = _ncart(ll)
            ncart = ni_c * nj_c * nk_c * nl_c

            task_mask = task_qidx == q_idx
            offsets = task_eri_offsets[task_mask]

            eri_sum = np.zeros(ncart, dtype=np.float64)
            for off in offsets:
                eri_sum += eri_buf[off:off + ncart]

            eri_cart = eri_sum.reshape(ni_c, nj_c, nk_c, nl_c)
            eri = _cart2sph_eri(eri_cart, li, lj, lk, ll)

            ni_s = _nsph(li)
            nj_s = _nsph(lj)
            nk_s = _nsph(lk)
            nl_s = _nsph(ll)

            if with_j:
                dm_kl = dm[k0:k0+nk_s, l0:l0+nl_s]
                vj[i0:i0+ni_s, j0:j0+nj_s] += np.einsum('kl,ijkl->ij', dm_kl, eri)
            if with_k:
                dm_jl = dm[j0:j0+nj_s, l0:l0+nl_s]
                vk[i0:i0+ni_s, k0:k0+nk_s] += np.einsum('jl,ijkl->ik', dm_jl, eri)

    # Phase 3b: CPU fallback for nroots > max_nroots
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

        if with_j:
            dm_kl = dm[k0:k0+nk, l0:l0+nl]
            vj[i0:i0+ni, j0:j0+nj] += np.einsum('kl,ijkl->ij', dm_kl, eri)
        if with_k:
            dm_jl = dm[j0:j0+nj, l0:l0+nl]
            vk[i0:i0+ni, k0:k0+nk] += np.einsum('jl,ijkl->ik', dm_jl, eri)
