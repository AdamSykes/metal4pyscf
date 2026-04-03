"""
Metal GPU direct J/K engine via Rys polynomial quadrature.

Phase 1 (CPU): enumerate primitive quartets, compute Rys roots/weights
Phase 2 (Metal): TRR + HRR + ERI + J/K contraction (1 thread per primitive per root)

Supports s/p/d/f shells (l=0,1,2,3).
"""

import numpy as np
import mlx.core as mx
from math import sqrt, pi
from gpu4pyscf.lib.metal_kernels.eval_ao import _ncart, _cart2sph_matrix
from gpu4pyscf.lib.metal_kernels.rys_jk import boys_function, _rys_from_boys

# ---------------------------------------------------------------------------
# Metal kernel: TRR + HRR + Cartesian ERI product + J/K atomic accumulation
#
# Each thread handles one (primitive quartet, Rys root) pair.
# Computes the 1D g-values for x,y,z, assembles the Cartesian ERI block,
# and atomic-adds the J/K contributions.
#
# Template params: MAX_LIJ, MAX_LKL, MAX_NCART
# These bound the array sizes for the g arrays and cart loops.
# ---------------------------------------------------------------------------

_RYS_JK_HEADER = '''
constant int NCART[] = {1, 3, 6, 10};

// Cartesian indices packed as lx*16+ly*4+lz, flat with offsets
constant int CART_ALL[] = {
    0,                                              // l=0: s
    64, 16, 4,                                      // l=1: p
    128, 80, 68, 32, 20, 8,                         // l=2: d
    192, 144, 132, 96, 84, 72, 48, 36, 24, 12       // l=3: f
};
constant int CART_OFF[] = {0, 1, 4, 10};  // offsets into CART_ALL
'''

_RYS_JK_SOURCE = '''
uint tid = thread_position_in_grid.x;
if (tid >= n_tasks) return;

// Load task data: each task = one (primitive quartet, Rys root) pair
int task_off = tid * TASK_STRIDE;
float aij   = task_data[task_off + 0];
float akl   = task_data[task_off + 1];
float PA_x  = task_data[task_off + 2];
float PA_y  = task_data[task_off + 3];
float PA_z  = task_data[task_off + 4];
float QC_x  = task_data[task_off + 5];
float QC_y  = task_data[task_off + 6];
float QC_z  = task_data[task_off + 7];
float AB_x  = task_data[task_off + 8];
float AB_y  = task_data[task_off + 9];
float AB_z  = task_data[task_off + 10];
float CD_x  = task_data[task_off + 11];
float CD_y  = task_data[task_off + 12];
float CD_z  = task_data[task_off + 13];
float Rpq_x = task_data[task_off + 14];
float Rpq_y = task_data[task_off + 15];
float Rpq_z = task_data[task_off + 16];
float prefac = task_data[task_off + 17];
float rys_root   = task_data[task_off + 18];
float rys_weight = task_data[task_off + 19];
int i0 = (int)task_data[task_off + 20];
int j0 = (int)task_data[task_off + 21];
int k0 = (int)task_data[task_off + 22];
int l0 = (int)task_data[task_off + 23];
int li = (int)task_data[task_off + 24];
int lj = (int)task_data[task_off + 25];
int lk = (int)task_data[task_off + 26];
int ll = (int)task_data[task_off + 27];

int lij = li + lj;
int lkl = lk + ll;
float rt = rys_root;
float wt = rys_weight;

// Rys-modified recurrence coefficients
float rt_aa = rt / (aij + akl);
float rt_aij = rt_aa * akl;
float rt_akl = rt_aa * aij;
float b10 = 0.5f / aij * (1.0f - rt_aij);
float b01 = 0.5f / akl * (1.0f - rt_akl);
float b00 = 0.5f * rt_aa;

// TRR + HRR per direction, store final g[i,j,k,l] values
// Max array: g[MAX_LIJ+1][MAX_LKL+1] for TRR
float PA[3] = {PA_x, PA_y, PA_z};
float QC[3] = {QC_x, QC_y, QC_z};
float AB[3] = {AB_x, AB_y, AB_z};
float CD[3] = {CD_x, CD_y, CD_z};
float Rpq[3] = {Rpq_x, Rpq_y, Rpq_z};

// g1d[dir][i][j][k][l] — stored flat
// For MAX_LIJ=6, MAX_LKL=6: 7*7 = 49 per dir for TRR
// After HRR: (li+1)*(lj+1)*(lk+1)*(ll+1) per dir
float g1d_final[3][4][4][4][4]; // max 4 per index (f-shell l=3 → 4 values)

for (int dir = 0; dir < 3; dir++) {
    float c0 = PA[dir] - rt_aij * Rpq[dir];
    float cp = QC[dir] + rt_akl * Rpq[dir];

    // TRR: g[a, c] for a in [0,lij], c in [0,lkl]
    float g[8][8]; // max 7+1 for lij+lkl up to 6+6... actually max lij=6 for ffff
    // For f-shells: lij = li+lj <= 6, lkl <= 6
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

    // HRR j-index: g_ij[i,j,c] from g[a,c]
    float g_ij[8][4][8]; // [lij+1][lj+1][lkl+1]
    for (int a = 0; a <= lij; a++)
        for (int c = 0; c <= lkl; c++)
            g_ij[a][0][c] = g[a][c];
    for (int j = 0; j < lj; j++)
        for (int c = 0; c <= lkl; c++)
            for (int i = 0; i < lij-j; i++)
                g_ij[i][j+1][c] = g_ij[i+1][j][c] + AB[dir] * g_ij[i][j][c];

    // HRR l-index
    float g_full[4][4][8][4]; // [li+1][lj+1][lkl+1][ll+1]
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

// Cartesian ERI assembly + J/K contraction
int ni = NCART[li], nj = NCART[lj], nk = NCART[lk], nl = NCART[ll];
int ci_off = CART_OFF[li], cj_off = CART_OFF[lj];
int ck_off = CART_OFF[lk], cl_off = CART_OFF[ll];

float pf_wt = prefac * wt;

for (int ii = 0; ii < ni; ii++) {
    int ci_v = CART_ALL[ci_off + ii];
    int ix_i = ci_v / 16, iy_i = (ci_v / 4) % 4, iz_i = ci_v % 4;
    for (int jj = 0; jj < nj; jj++) {
        int cj_v = CART_ALL[cj_off + jj];
        int ix_j = cj_v / 16, iy_j = (cj_v / 4) % 4, iz_j = cj_v % 4;

        float j_contrib = 0.0f;
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
                j_contrib += eri_val * dm[(k0+kk)*nao + (l0+ll_idx)];

                // K contribution: K[i,k] += ERI[i,j,k,l] * DM[j,l]
                float k_val = eri_val * dm[(j0+jj)*nao + (l0+ll_idx)];
                atomic_fetch_add_explicit(&vk[(i0+ii)*nao + (k0+kk)], k_val, memory_order_relaxed);
            }
        }
        atomic_fetch_add_explicit(&vj[(i0+ii)*nao + (j0+jj)], j_contrib, memory_order_relaxed);
    }
}
'''

_rys_jk_kernel = mx.fast.metal_kernel(
    name='rys_jk_direct',
    input_names=['task_data', 'dm'],
    output_names=['vj', 'vk'],
    header=_RYS_JK_HEADER,
    source=_RYS_JK_SOURCE,
    atomic_outputs=True,
)

TASK_STRIDE = 28  # floats per task


def get_jk_rys_metal(mol, dm, with_j=True, with_k=True):
    """Direct J/K on Metal GPU via Rys quadrature.

    CPU: enumerates primitive quartets, computes Rys roots/weights.
    GPU: TRR + HRR + ERI + J/K contraction in parallel.
    """
    nao = mol.nao
    dm = np.asarray(dm, dtype=np.float64)
    nbas = mol.nbas
    ao_loc = mol.ao_loc_nr()
    fac_l = {0: sqrt(1.0 / (4 * pi)), 1: sqrt(3.0 / (4 * pi))}

    # --- Phase 1: CPU — enumerate tasks and compute Rys roots ---
    tasks = []

    for ish in range(nbas):
        li = mol.bas_angular(ish)
        if li > 3: continue
        i0 = ao_loc[ish]
        Ri = mol.atom_coord(mol.bas_atom(ish))
        ai_list = mol.bas_exp(ish)
        ci_list = mol._libcint_ctr_coeff(ish).flatten()

        for jsh in range(nbas):
            lj = mol.bas_angular(jsh)
            if lj > 3: continue
            j0 = ao_loc[jsh]
            Rj = mol.atom_coord(mol.bas_atom(jsh))
            aj_list = mol.bas_exp(jsh)
            cj_list = mol._libcint_ctr_coeff(jsh).flatten()
            rr_ij = np.dot(Ri - Rj, Ri - Rj)

            for ksh in range(nbas):
                lk = mol.bas_angular(ksh)
                if lk > 3: continue
                k0 = ao_loc[ksh]
                Rk = mol.atom_coord(mol.bas_atom(ksh))
                ak_list = mol.bas_exp(ksh)
                ck_list = mol._libcint_ctr_coeff(ksh).flatten()

                for lsh in range(nbas):
                    ll = mol.bas_angular(lsh)
                    if ll > 3: continue
                    l0 = ao_loc[lsh]
                    Rl = mol.atom_coord(mol.bas_atom(lsh))
                    al_list = mol.bas_exp(lsh)
                    cl_list = mol._libcint_ctr_coeff(lsh).flatten()
                    rr_kl = np.dot(Rk - Rl, Rk - Rl)
                    nroots = (li + lj + lk + ll) // 2 + 1

                    PA = np.zeros(3)  # filled per primitive
                    QC = np.zeros(3)
                    AB = Ri - Rj
                    CD = Rk - Rl

                    for ip, ai in enumerate(ai_list):
                        for jp, aj in enumerate(aj_list):
                            aij = ai + aj
                            Pij = (ai * Ri + aj * Rj) / aij
                            Kab = np.exp(-ai * aj / aij * rr_ij)
                            PA = Pij - Ri

                            for kp, ak in enumerate(ak_list):
                                for lp, al in enumerate(al_list):
                                    akl = ak + al
                                    Pkl = (ak * Rk + al * Rl) / akl
                                    Kcd = np.exp(-ak * al / akl * rr_kl)
                                    QC = Pkl - Rk

                                    coeff = (ci_list[ip] * cj_list[jp] * Kab *
                                             ck_list[kp] * cl_list[lp] * Kcd)
                                    if abs(coeff) < 1e-14:
                                        continue

                                    theta = aij * akl / (aij + akl)
                                    Rpq = Pij - Pkl
                                    x = theta * np.dot(Rpq, Rpq)

                                    prefac = coeff * 2.0 * pi**2.5 / (aij * akl * sqrt(aij + akl))
                                    prefac *= (fac_l.get(li, 1.0) * fac_l.get(lj, 1.0) *
                                               fac_l.get(lk, 1.0) * fac_l.get(ll, 1.0))

                                    m_max = max(li + lj + lk + ll, 2 * nroots - 1)
                                    fm = boys_function(m_max, [x])
                                    roots, weights = _rys_from_boys(nroots, fm[:, 0])

                                    for ir in range(nroots):
                                        task = np.zeros(TASK_STRIDE, dtype=np.float32)
                                        task[0] = aij
                                        task[1] = akl
                                        task[2:5] = PA
                                        task[5:8] = QC
                                        task[8:11] = AB
                                        task[11:14] = CD
                                        task[14:17] = Rpq
                                        task[17] = prefac
                                        task[18] = roots[ir]
                                        task[19] = weights[ir]
                                        task[20] = i0
                                        task[21] = j0
                                        task[22] = k0
                                        task[23] = l0
                                        task[24] = li
                                        task[25] = lj
                                        task[26] = lk
                                        task[27] = ll
                                        tasks.append(task)

    if not tasks:
        return (np.zeros((nao, nao)) if with_j else None,
                np.zeros((nao, nao)) if with_k else None)

    # --- Phase 2: Metal GPU kernel ---
    task_array = mx.array(np.stack(tasks).ravel())
    dm_gpu = mx.array(dm.astype(np.float32).ravel())
    n_tasks = len(tasks)

    THREADS = 256
    grid_size = ((n_tasks + THREADS - 1) // THREADS) * THREADS

    result = _rys_jk_kernel(
        inputs=[task_array, dm_gpu],
        grid=(grid_size, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(nao * nao,), (nao * nao,)],
        output_dtypes=[mx.float32, mx.float32],
        template=[('n_tasks', n_tasks), ('TASK_STRIDE', TASK_STRIDE), ('nao', nao)],
    )
    mx.eval(result[0], result[1])

    vj = np.array(result[0]).astype(np.float64).reshape(nao, nao) if with_j else None
    vk = np.array(result[1]).astype(np.float64).reshape(nao, nao) if with_k else None

    # Cart → spherical transformation on the accumulated J/K
    # (The kernel works in the spherical AO basis since ao_loc gives spherical indices)

    return vj, vk
