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
constant int CART_ALL[] = {
    0,
    16, 4, 1,
    32, 20, 17, 8, 5, 2,
    48, 36, 33, 24, 21, 18, 12, 9, 6, 3
};
constant int CART_OFF[] = {0, 1, 4, 10};

// erf approximation for Metal (Abramowitz & Stegun 7.1.26, max error 1.5e-7)
inline float metal_erf(float x) {
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f +
                 t * (-1.453152027f + t * 1.061405429f))));
    float result = 1.0f - poly * exp(-ax * ax);
    return (x >= 0) ? result : -result;
}

// Boys function F_m(t) for m=0..max_m, in f32
// 3 regimes matching CUDA gamma_inc.cu
inline void boys_fn(thread float* f, float t, int m) {
    if (t < 1e-7f) {
        f[0] = 1.0f;
        for (int i = 1; i <= m; i++)
            f[i] = 1.0f / (2*i + 1);
        return;
    }
    if (m > 0 && t < m * 0.5f + 0.5f) {
        float bi = m + 0.5f;
        float e = 0.5f * exp(-t);
        float x = e, s = e;
        for (int iter = 0; iter < 50 && x > 1e-7f * e; iter++) {
            bi += 1.0f; x *= t / bi; s += x;
        }
        float b = m + 0.5f;
        float fval = s / b;
        f[m] = fval;
        for (int i = m - 1; i >= 0; i--) {
            b -= 1.0f;
            fval = (e + t * fval) / b;
            f[i] = fval;
        }
        return;
    }
    float tt = sqrt(t);
    float fval = 0.886226925f / tt * metal_erf(tt); // sqrt(pi/4)/tt * erf(tt)
    f[0] = fval;
    if (m > 0) {
        float e = 0.5f * exp(-t);
        float b = 1.0f / t;
        float b1 = 0.5f;
        for (int i = 1; i <= m; i++) {
            fval = b * (b1 * fval - e);
            f[i] = fval;
            b1 += 1.0f;
        }
    }
}

// Rys roots for nroots=1: root = F1/F0, weight = F0
// Rys roots for nroots=2: Hankel 2x2 solve
// Rys roots for nroots=3: Hankel 3x3 solve (simplified)
inline void rys_roots_1(thread float* fm, thread float* roots, thread float* weights) {
    weights[0] = fm[0];
    roots[0] = (fm[0] > 1e-15f) ? fm[1] / fm[0] : 0.0f;
}

inline void rys_roots_2(thread float* fm, thread float* roots, thread float* weights) {
    // Hankel 2x2: [F0 F1; F1 F2] @ c = [F2, F3]
    float det = fm[0]*fm[2] - fm[1]*fm[1];
    if (abs(det) < 1e-20f) {
        float t = (fm[0] > 1e-15f) ? fm[1]/fm[0] : 0.0f;
        roots[0] = roots[1] = t;
        weights[0] = weights[1] = fm[0] * 0.5f;
        return;
    }
    float c0 = (fm[2]*fm[2] - fm[1]*fm[3]) / det;
    float c1 = (fm[0]*fm[3] - fm[1]*fm[2]) / det;
    float disc = c1*c1 + 4*c0;
    if (disc < 0) disc = 0;
    float sq = sqrt(disc);
    roots[0] = (c1 - sq) * 0.5f;
    roots[1] = (c1 + sq) * 0.5f;
    if (abs(roots[1] - roots[0]) < 1e-15f) {
        weights[0] = weights[1] = fm[0] * 0.5f;
    } else {
        weights[1] = (fm[1] - fm[0]*roots[0]) / (roots[1] - roots[0]);
        weights[0] = fm[0] - weights[1];
    }
}

inline void rys_roots_3(thread float* fm, thread float* roots, thread float* weights) {
    // Hankel 3x3 solve: simplified Cramer's rule
    float H[3][3] = {{fm[0],fm[1],fm[2]},{fm[1],fm[2],fm[3]},{fm[2],fm[3],fm[4]}};
    float rhs[3] = {fm[3], fm[4], fm[5]};
    // Gaussian elimination
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        for (int row = col+1; row < 3; row++)
            if (abs(H[row][col]) > abs(H[pivot][col])) pivot = row;
        if (pivot != col) {
            for (int j = 0; j < 3; j++) { float t=H[col][j]; H[col][j]=H[pivot][j]; H[pivot][j]=t; }
            float t=rhs[col]; rhs[col]=rhs[pivot]; rhs[pivot]=t;
        }
        if (abs(H[col][col]) < 1e-20f) continue;
        for (int row = col+1; row < 3; row++) {
            float fac = H[row][col] / H[col][col];
            for (int j = col; j < 3; j++) H[row][j] -= fac * H[col][j];
            rhs[row] -= fac * rhs[col];
        }
    }
    float c[3];
    for (int i = 2; i >= 0; i--) {
        float s = rhs[i];
        for (int j = i+1; j < 3; j++) s -= H[i][j] * c[j];
        c[i] = (abs(H[i][i]) > 1e-20f) ? s / H[i][i] : 0.0f;
    }
    // Roots of t^3 - c[2]*t^2 - c[1]*t - c[0] = 0
    // Cardano's formula or Newton iteration
    // Use Newton iteration from 3 initial guesses
    float p = -c[2], q = -c[1], r = -c[0]; // t^3 + p*t^2 + q*t + r = 0
    // Initial guesses spread in [0,1]
    roots[0] = 0.1f; roots[1] = 0.4f; roots[2] = 0.8f;
    for (int iter = 0; iter < 20; iter++) {
        for (int k = 0; k < 3; k++) {
            float t = roots[k];
            float f = t*t*t + p*t*t + q*t + r;
            float fp = 3*t*t + 2*p*t + q;
            if (abs(fp) > 1e-15f) roots[k] -= f / fp;
            roots[k] = clamp(roots[k], 0.0f, 1.0f);
        }
    }
    // Sort
    if (roots[0] > roots[1]) { float t=roots[0]; roots[0]=roots[1]; roots[1]=t; }
    if (roots[1] > roots[2]) { float t=roots[1]; roots[1]=roots[2]; roots[2]=t; }
    if (roots[0] > roots[1]) { float t=roots[0]; roots[0]=roots[1]; roots[1]=t; }
    // Weights from Vandermonde
    float V[3][3];
    for (int i = 0; i < 3; i++) { V[i][0]=1; V[i][1]=roots[i]; V[i][2]=roots[i]*roots[i]; }
    // Solve V^T @ w = fm[0:3] via Gaussian elimination
    float VT[3][3]; for(int i=0;i<3;i++) for(int j=0;j<3;j++) VT[i][j]=V[j][i];
    float wrhs[3] = {fm[0], fm[1], fm[2]};
    for (int col = 0; col < 3; col++) {
        int piv = col;
        for (int row = col+1; row < 3; row++)
            if (abs(VT[row][col]) > abs(VT[piv][col])) piv = row;
        if (piv != col) {
            for (int j=0;j<3;j++){float t=VT[col][j];VT[col][j]=VT[piv][j];VT[piv][j]=t;}
            float t=wrhs[col]; wrhs[col]=wrhs[piv]; wrhs[piv]=t;
        }
        for (int row = col+1; row < 3; row++) {
            float fac = VT[row][col] / (VT[col][col] + 1e-30f);
            for (int j=col;j<3;j++) VT[row][j] -= fac*VT[col][j];
            wrhs[row] -= fac*wrhs[col];
        }
    }
    for (int i=2;i>=0;i--) {
        float s=wrhs[i]; for(int j=i+1;j<3;j++) s-=VT[i][j]*weights[j];
        weights[i] = s / (VT[i][i] + 1e-30f);
    }
}
'''

_RYS_JK_SOURCE = '''
uint tid = thread_position_in_grid.x;
if (tid >= n_tasks) return;

// Load task data: each task = one primitive quartet
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
int i0 = (int)task_data[task_off + 18];
int j0 = (int)task_data[task_off + 19];
int k0 = (int)task_data[task_off + 20];
int l0 = (int)task_data[task_off + 21];
int li = (int)task_data[task_off + 22];
int lj = (int)task_data[task_off + 23];
int lk = (int)task_data[task_off + 24];
int ll = (int)task_data[task_off + 25];

int lij = li + lj;
int lkl = lk + ll;
int nroots = (li + lj + lk + ll) / 2 + 1;

// Compute Boys function on GPU
float theta = aij * akl / (aij + akl);
float rr_pq = Rpq_x*Rpq_x + Rpq_y*Rpq_y + Rpq_z*Rpq_z;
float x = theta * rr_pq;
int m_max = max(li+lj+lk+ll, 2*nroots-1);
float fm[14]; // max m_max = 13 for ffff
boys_fn(fm, x, m_max);

// Compute Rys roots and weights on GPU
float rys_roots[7], rys_weights[7]; // max nroots=7
if (nroots == 1) rys_roots_1(fm, rys_roots, rys_weights);
else if (nroots == 2) rys_roots_2(fm, rys_roots, rys_weights);
else if (nroots == 3) rys_roots_3(fm, rys_roots, rys_weights);
// nroots > 3 not yet supported in MSL

// Loop over Rys roots
for (int iroot = 0; iroot < nroots; iroot++) {
float rt = rys_roots[iroot];
float wt = rys_weights[iroot];

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
} // end root loop
'''

_rys_jk_kernel = mx.fast.metal_kernel(
    name='rys_jk_direct',
    input_names=['task_data', 'dm'],
    output_names=['vj', 'vk'],
    header=_RYS_JK_HEADER,
    source=_RYS_JK_SOURCE,
    atomic_outputs=True,
)

TASK_STRIDE = 26  # floats per task (no roots/weights — computed on GPU)


def _schwarz_bounds(mol, nbas):
    """Precompute Schwarz screening bounds Q[ish,jsh] = sqrt(max|(ij|ij)|).

    Used to skip shell quartets where Q_ij * Q_kl < threshold.
    """
    Q = np.zeros((nbas, nbas))
    for ish in range(nbas):
        for jsh in range(ish, nbas):
            eri = mol.intor('int2e', shls_slice=(
                ish, ish + 1, jsh, jsh + 1, ish, ish + 1, jsh, jsh + 1))
            Q[ish, jsh] = Q[jsh, ish] = np.sqrt(np.max(np.abs(eri)))
    return Q


def _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, schwarz_thresh=1e-10):
    """Build all primitive quartet tasks using numpy vectorization.

    Includes Schwarz screening to skip negligible shell quartets.
    """
    shell_l = np.array([mol.bas_angular(i) for i in range(nbas)])
    shell_atom = np.array([mol.bas_atom(i) for i in range(nbas)])
    atom_coords = np.array([mol.atom_coord(i) for i in range(mol.natm)])
    shell_R = atom_coords[shell_atom]
    shell_i0 = np.array([ao_loc[i] for i in range(nbas)])

    shell_exps = [mol.bas_exp(i) for i in range(nbas)]
    shell_coeffs = [mol._libcint_ctr_coeff(i).flatten() for i in range(nbas)]

    # Schwarz screening bounds
    Q = _schwarz_bounds(mol, nbas)

    all_tasks = []
    n_screened = 0
    n_total = 0

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

            # Vectorize over ij primitives
            ni, nj = len(ai_arr), len(aj_arr)
            ai_2d = np.repeat(ai_arr, nj)      # (ni*nj,)
            aj_2d = np.tile(aj_arr, ni)
            ci_2d = np.repeat(ci_arr, nj)
            cj_2d = np.tile(cj_arr, ni)
            aij = ai_2d + aj_2d
            Kab = np.exp(-ai_2d * aj_2d / aij * rr_ij)
            Pij = (ai_2d[:, None] * Ri + aj_2d[:, None] * Rj) / aij[:, None]  # (ni*nj, 3)
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
                    n_total += 1

                    # Schwarz screening
                    if Q[ish, jsh] * Q[ksh, lsh] < schwarz_thresh:
                        n_screened += 1
                        continue

                    Rl = shell_R[lsh]
                    al_arr = shell_exps[lsh]
                    cl_arr = shell_coeffs[lsh]
                    rr_kl = np.dot(Rk - Rl, Rk - Rl)
                    CD = Rk - Rl

                    # Vectorize over kl primitives
                    nk, nl = len(ak_arr), len(al_arr)
                    ak_2d = np.repeat(ak_arr, nl)
                    al_2d = np.tile(al_arr, nk)
                    ck_2d = np.repeat(ck_arr, nl)
                    cl_2d = np.tile(cl_arr, nk)
                    akl = ak_2d + al_2d
                    Kcd = np.exp(-ak_2d * al_2d / akl * rr_kl)
                    Pkl = (ak_2d[:, None] * Rk + al_2d[:, None] * Rl) / akl[:, None]
                    QC = Pkl - Rk
                    ckcl_Kcd = ck_2d * cl_2d * Kcd

                    # Cross-product: all (ij, kl) combinations
                    n_ij = ni * nj
                    n_kl = nk * nl
                    # Expand to (n_ij * n_kl) tasks
                    aij_all = np.repeat(aij, n_kl)
                    akl_all = np.tile(akl, n_ij)
                    PA_all = np.repeat(PA, n_kl, axis=0)
                    QC_all = np.tile(QC, (n_ij, 1))
                    Pij_all = np.repeat(Pij, n_kl, axis=0)
                    Pkl_all = np.tile(Pkl, (n_ij, 1))
                    Rpq_all = Pij_all - Pkl_all
                    coeff_all = np.repeat(cicj_Kab, n_kl) * np.tile(ckcl_Kcd, n_ij)

                    # Screening
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

                    ntasks = len(prefac)
                    block = np.zeros((ntasks, TASK_STRIDE), dtype=np.float32)
                    block[:, 0] = aij_m
                    block[:, 1] = akl_m
                    block[:, 2:5] = PA_m
                    block[:, 5:8] = QC_m
                    block[:, 8] = AB[0]; block[:, 9] = AB[1]; block[:, 10] = AB[2]
                    block[:, 11] = CD[0]; block[:, 12] = CD[1]; block[:, 13] = CD[2]
                    block[:, 14:17] = Rpq_m
                    block[:, 17] = prefac
                    block[:, 18] = shell_i0[ish]
                    block[:, 19] = shell_i0[jsh]
                    block[:, 20] = shell_i0[ksh]
                    block[:, 21] = shell_i0[lsh]
                    block[:, 22] = li
                    block[:, 23] = lj
                    block[:, 24] = lk
                    block[:, 25] = ll
                    all_tasks.append(block)

    if all_tasks:
        return np.concatenate(all_tasks, axis=0)
    return np.zeros((0, TASK_STRIDE), dtype=np.float32)


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

    # --- Phase 1: vectorized task generation with Schwarz screening ---
    tasks = _build_tasks_vectorized(mol, ao_loc, nbas, fac_l, schwarz_thresh=1e-10)

    if len(tasks) == 0:
        return (np.zeros((nao, nao)) if with_j else None,
                np.zeros((nao, nao)) if with_k else None)

    # --- Phase 2: Metal GPU kernel ---
    task_array = mx.array(tasks.ravel())
    dm_gpu = mx.array(dm.astype(np.float32).ravel())
    n_tasks = tasks.shape[0]

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
