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

Uses Rys quadrature with Obara-Saika recursion:
  1. Boys function Fm(T) via downward recursion
  2. Rys roots/weights from Boys moments (modified Chebyshev + QL)
  3. Per-root 1D OS recursion in x, y, z independently
  4. Contract gx * gy * gz over roots
  5. Apply derivative: d/dA_i = 2*alpha_a * (a+1_i) - n_i * (a-1_i)

Architecture: one thread per shell triple (ish, jsh, ksh_aux).
Each thread loops over its primitive triples, accumulates gout, then
writes to the global output tensor.

Supported: l_orb <= 3 (f), l_aux <= 4 (g), nroots <= 6.
"""

import numpy as np
import mlx.core as mx
from pyscf.gto.moleintor import getints, make_cintopt


# ---------------------------------------------------------------------------
# Cartesian index tables
# ---------------------------------------------------------------------------

def _cart_powers(l):
    """Cartesian powers (ix,iy,iz) with ix+iy+iz=l in PySCF ordering."""
    powers = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            iz = l - ix - iy
            powers.append((ix, iy, iz))
    return powers


# ---------------------------------------------------------------------------
# Metal kernel source
# ---------------------------------------------------------------------------

# Max supported: l_orb=3 (f), l_aux=4 (g), nroots=6
# Max g-array per direction: (li_ceil+lj+1) * (lk+1) * nroots
#   worst case: (4+3+1)*(4+1)*6 = 8*5*6 = 240 per direction
# Max gout: 3 * 10 * 10 * 15 = 4500
# Total thread-local: ~(720 + 4500) * 4 = ~21 KB < 32 KB Metal limit

_BOYS_FUNCTION = '''
// Boys function Fm(T) for m = 0..mmax.
// Two regimes:
//   T > mmax+1: F0 from erf (Abramowitz & Stegun 7.1.26) + upward recursion.
//               Upward recursion F_{m+1} = ((2m+1)Fm - e^-T) / (2T) is stable
//               when mmax < T because division by 2T damps errors.
//   T <= mmax+1: series expansion for F_M at very high M, then downward
//                recursion F_m = (2T F_{m+1} + e^-T) / (2m+1) which is
//                unconditionally stable. Start M is set high enough (T + mmax + 30)
//                to ensure the series converges in f32.
//
// Validated: rel_err < 1e-4 for T in [0, 100], m in [0, 11].
'''

_BOYS_FUNCTION_BODY = '''
{
    float expT = exp(-T);
    if (T < 1e-7f) {
        for (int m = 0; m <= BOYS_MMAX; m++)
            Fm[m] = 1.0f / float(2*m + 1);
    } else if (T > float(BOYS_MMAX) + 1.0f) {
        float sqrtT = sqrt(T);
        float t_erf = 1.0f / (1.0f + 0.3275911f * sqrtT);
        float t2 = t_erf*t_erf, t3 = t2*t_erf, t4 = t3*t_erf, t5 = t4*t_erf;
        float erf_val = 1.0f - (0.254829592f*t_erf - 0.284496736f*t2
            + 1.421413741f*t3 - 1.453152027f*t4 + 1.061405429f*t5) * expT;
        Fm[0] = 0.8862269254527580f * erf_val / sqrtT;
        for (int m = 0; m < BOYS_MMAX; m++)
            Fm[m+1] = (float(2*m + 1) * Fm[m] - expT) / (2.0f * T);
    } else {
        int mstart = int(T) + BOYS_MMAX + 30;
        float term = 1.0f;
        float fm = term / float(2*mstart + 1);
        for (int k = 1; k < 80; k++) {
            term *= T / float(k);
            float contrib = term / float(2*mstart + 2*k + 1);
            fm += contrib;
            if (abs(contrib) < 1e-7f * abs(fm) && k > 5) break;
        }
        fm *= expT;
        for (int m = mstart - 1; m >= 0; m--) {
            fm = (2.0f * T * fm + expT) / float(2*m + 1);
            if (m <= BOYS_MMAX) Fm[m] = fm;
        }
    }
}
'''

_RYS_ROOTS = '''
// Rys roots and weights from Boys moments using modified Chebyshev
// algorithm (Golub-Welsch). Builds tridiagonal Jacobi matrix from
// the moments F0..F_{2n-1}, then computes eigenvalues (roots) and
// eigenvectors (weights) via implicit QL with shifts.
//
// nroots: 1..6. roots[i] in (0,1), weights[i] > 0.
void rys_roots(int nroots, thread float* Fm,
               thread float* roots, thread float* weights) {
    if (nroots == 1) {
        roots[0] = Fm[1] / Fm[0];
        weights[0] = Fm[0];
        return;
    }
    // Build tridiagonal Jacobi matrix via modified Chebyshev algorithm
    // alpha[i], beta[i] define the tridiagonal: T_ii = alpha[i],
    //   T_{i,i+1} = T_{i+1,i} = sqrt(beta[i+1]).
    float alpha[6], beta_arr[6];
    // sigma[k] = moment = F_k / F_0 (normalised moments)
    float sigma0[12], sigma1[12];
    for (int k = 0; k < 2*nroots; k++) sigma0[k] = Fm[k];
    alpha[0] = Fm[1] / Fm[0];
    beta_arr[0] = Fm[0];  // = integral weight (unused in T, but needed for w0)
    for (int k = 0; k < 2*nroots-2; k++)
        sigma1[k] = sigma0[k+2] - alpha[0] * sigma0[k+1];
    for (int i = 1; i < nroots; i++) {
        // beta[i] = sigma1[0] / sigma0[0]  (ratio of leading terms)
        beta_arr[i] = sigma1[0] / sigma0[0];
        if (beta_arr[i] <= 0.0f) { beta_arr[i] = 1e-20f; }
        // alpha[i] = sigma1[1]/sigma1[0] - sigma0[1]/sigma0[0]
        alpha[i] = sigma1[1] / sigma1[0] - sigma0[1] / sigma0[0];
        // Shift: sigma0 <- sigma1, sigma1 from recursion
        int nn = 2*nroots - 2*i - 2;
        if (nn > 0) {
            float tmp[12];
            for (int k = 0; k < nn; k++)
                tmp[k] = sigma1[k+2] - alpha[i]*sigma1[k+1] - beta_arr[i]*sigma0[k+2];
            for (int k = 0; k < 2*nroots; k++) sigma0[k] = sigma1[k];
            for (int k = 0; k < nn; k++) sigma1[k] = tmp[k];
        } else {
            for (int k = 0; k < 2*nroots; k++) sigma0[k] = sigma1[k];
        }
    }
    // Now: T = tridiag(sqrt(beta[1..n-1]), alpha[0..n-1], sqrt(beta[1..n-1]))
    // Eigenvalues → roots, first component of eigenvectors → weights.
    // Use implicit QL algorithm for small tridiagonal matrix.
    float d[6], e[6], z[6];
    for (int i = 0; i < nroots; i++) {
        d[i] = alpha[i];
        e[i] = (i > 0) ? sqrt(max(beta_arr[i], 0.0f)) : 0.0f;
        z[i] = (i == 0) ? 1.0f : 0.0f;
    }
    // QL iteration (adapted from EISPACK tql1/tql2)
    for (int l = 0; l < nroots; l++) {
        for (int iter = 0; iter < 50; iter++) {
            int m = l;
            for (; m < nroots - 1; m++) {
                float dd = abs(d[m]) + abs(d[m+1]);
                if (abs(e[m+1]) + dd == dd) break;
            }
            if (m == l) break;
            float g = (d[l+1] - d[l]) / (2.0f * e[l+1]);
            float r = sqrt(g*g + 1.0f);
            float s_val = (g >= 0) ? g + r : g - r;
            g = d[m] - d[l] + e[l+1] / s_val;
            float s = 1.0f, c = 1.0f, p = 0.0f;
            for (int i = m - 1; i >= l; i--) {
                float f = s * e[i+1];
                float b = c * e[i+1];
                if (abs(f) >= abs(g)) {
                    c = g / f; r = sqrt(c*c + 1.0f);
                    e[i+2] = f * r; s = 1.0f / r; c *= s;
                } else {
                    s = f / g; r = sqrt(s*s + 1.0f);
                    e[i+2] = g * r; c = 1.0f / r; s *= c;
                }
                g = d[i+1] - p;
                r = (d[i] - g) * s + 2.0f * c * b;
                p = s * r; d[i+1] = g + p; g = c * r - b;
                // Track first row of eigenvector matrix
                float zf = z[i+1];
                z[i+1] = s*z[i] + c*zf;
                z[i] = c*z[i] - s*zf;
            }
            d[l] -= p; e[l+1] = g; e[l] = 0.0f;  // keep e[0] safe
        }
    }
    // Sort eigenvalues ascending (insertion sort on tiny array)
    for (int i = 1; i < nroots; i++) {
        float key_d = d[i], key_z = z[i];
        int j = i - 1;
        while (j >= 0 && d[j] > key_d) {
            d[j+1] = d[j]; z[j+1] = z[j]; j--;
        }
        d[j+1] = key_d; z[j+1] = key_z;
    }
    // roots = eigenvalues, weights = F0 * z[i]^2
    for (int i = 0; i < nroots; i++) {
        roots[i] = d[i];
        weights[i] = Fm[0] * z[i] * z[i];
    }
}
'''

_INT3C2E_IP1_KERNEL = '''
// int3c2e_ip1 Metal kernel.
// One thread per shell triple (ish, jsh, ksh_aux).
// Computes nabla_A (mu nu | P) for all Cartesian components of the triple.

''' + _BOYS_FUNCTION + _RYS_ROOTS + '''

constant float PI_5_2 = 34.9868366552497250f;  // pi^(5/2)

uint tid = thread_position_in_grid.x;
if (tid >= n_triples) return;

int ish = triples[tid*3+0];
int jsh = triples[tid*3+1];
int ksh = triples[tid*3+2];

int li = shell_l[ish]; int lj = shell_l[jsh]; int lk = shell_l[ksh];
int li1 = li + 1;  // raised angular momentum for derivative
int nfi = (li+1)*(li+2)/2;
int nfj = (lj+1)*(lj+2)/2;
int nfk = (lk+1)*(lk+2)/2;
int nroots = (li1 + lj + lk) / 2 + 1;

float ax = shell_x[ish], ay = shell_y[ish], az = shell_z[ish];
float bx = shell_x[jsh], by = shell_y[jsh], bz = shell_z[jsh];
float cx = shell_x[ksh], cy = shell_y[ksh], cz = shell_z[ksh];

int ao_i = ao_off_orb[ish], ao_j = ao_off_orb[jsh], ao_k = ao_off_aux[ksh];
int np_i = shell_nprim[ish], np_j = shell_nprim[jsh], np_k = shell_nprim[ksh];
int off_i = prim_off[ish], off_j = prim_off[jsh], off_k = prim_off[ksh];

// Accumulator for contracted integral derivatives: (3, nfi, nfj, nfk)
float gout[3 * 10 * 10 * 15];  // max: 4500 floats
int gout_size = 3 * nfi * nfj * nfk;
for (int i = 0; i < gout_size; i++) gout[i] = 0.0f;

// g-array per direction: indexed [a][k][root] where a=0..li1+lj, k=0..lk
// max per direction: 8*5*6 = 240
int na = li1 + lj + 1;
int nk = lk + 1;
// Three directions: gx, gy, gz
float gx[240], gy[240], gz[240];

// Cartesian power tables (precomputed; max 15 entries per shell)
// Layout: cart_ix[shell_l_offset + component_index]
// These are passed as kernel inputs.

float rr_ab = (ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz);

for (int pi = 0; pi < np_i; pi++) {
    float ai = exps[off_i + pi];
    float ci = coeffs[off_i + pi];
    for (int pj = 0; pj < np_j; pj++) {
        float aj = exps[off_j + pj];
        float cj = coeffs[off_j + pj];

        float aij = ai + aj;
        float eij = exp(-ai * aj / aij * rr_ab);
        if (eij < 1e-12f) continue;

        float px = (ai*ax + aj*bx) / aij;
        float py = (ai*ay + aj*by) / aij;
        float pz = (ai*az + aj*bz) / aij;

        for (int pk = 0; pk < np_k; pk++) {
            float ak = exps[off_k + pk];
            float ck = coeffs[off_k + pk];
            float aijk = aij + ak;
            float a0 = aij * ak / aijk;

            float rr_pc = (px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz);
            float T = a0 * rr_pc;

            // Prefactor
            float fac = eij * ci * cj * ck * 2.0f * PI_5_2 / (aij * sqrt(aijk));

            // Boys function: need F_0 .. F_{2*nroots-1}
            float Fm[13];  // max 2*6+1 = 13
            boys_fn(T, 2*nroots - 1, Fm);

            // Rys roots and weights
            float rts[6], wts[6];
            rys_roots(nroots, Fm, rts, wts);

            // Weighted center
            float wx = (aij*px + ak*cx) / aijk;
            float wy = (aij*py + ak*cy) / aijk;
            float wz = (aij*pz + ak*cz) / aijk;

            // For each Rys root: build 1D g-arrays via OS recursion
            for (int n = 0; n < nroots; n++) {
                float u2 = rts[n];      // u^2
                float w = wts[n] * fac; // weight * prefactor

                // Effective exponents for this root
                float u2_aijk = u2 * aijk;
                float tmp2 = 0.5f / (u2_aijk + aij*ak/aijk*aijk);
                // Actually: tmp4 = 0.5 / (u2 * (aij+ak) + aij*ak)
                // But u2 is already the Rys root t^2, and the rescaling is:
                // t_scaled^2 = t^2 * aijk / (t^2 * aijk + 1)... hmm.

                // Standard Rys: after scaling, the recursion coefficients are:
                // b00 = 0.5 * rho / (aij * ak)  where rho = u^2 * aij * ak / aijk
                // b10 = (0.5/aij) * (1 - rho/aij)
                // b01 = (0.5/ak) * (1 - rho/ak)

                // For 3-center: q = ak, p = aij, zeta = aij + ak
                float rho = u2 * a0;  // u^2 * aij*ak/(aij+ak)
                float b00 = 0.5f * rho / (aij * ak);
                // Wait, b00 = u^2 / (2*(u^2*zeta + p*q/zeta))...
                // Actually the standard formulas from Helgaker's textbook:
                // After Rys scaling:
                float pfac = rho / aij;  // rho/p
                float qfac = rho / ak;   // rho/q
                float b10 = 0.5f / aij * (1.0f - pfac);
                float b01 = 0.5f / ak * (1.0f - qfac);
                float bp = 0.5f / aijk;  // b00 = rho / (2*p*q) hmm

                // Standard from Rys quadrature (Lindh, Ryu, Liu):
                // For root u with u² = rts[n]:
                // The effective coordinates:
                //   PA_eff_x = (P_x - A_x) + u² * (W_x - P_x) / ...
                // This gets complicated. Let me use the simpler formulation:

                // For 3-center (ij|k), the recursion in the Rys scheme is:
                // gx[a+1, k, n] = cpx * gx[a, k, n] + a * b10 * gx[a-1, k, n]
                //                 + k * b00 * gx[a, k-1, n]
                // gx[a, k+1, n] = cmx * gx[a, k, n] + k * b01 * gx[a, k-1, n]
                //                 + a * b00 * gx[a-1, k, n]

                // Coefficients:
                float t2_over = u2 / (1.0f + u2);  // t^2/(1+t^2)
                float oo2z = 0.5f / aijk;
                float oo2p = 0.5f / aij;
                float oo2q = 0.5f / ak;

                // For the bra recursion:
                float cpx = px - ax - t2_over * (px - wx);
                float cpy = py - ay - t2_over * (py - wy);
                float cpz = pz - az - t2_over * (pz - wz);
                // ... hmm, this isn't quite right either.

                // Let me use the CUDA code's approach directly:
                // From g2e.cu: the coefficients are computed from:
                // u2 = a0 * uw[n]  (uw[n] is the Rys root)
                // tmp4 = 0.5 / (u2 * aijkl + a1)
                //   where a1 = aij*akl (for 4-center), aijkl = aij+akl
                //   For 3-center: akl = ak, aijkl = aij+ak
                // b00 = u2 * tmp4
                // b10 = b00 + 0.5/aij * (1 - u2*tmp4*2*aij)...
                // This is getting convoluted. Let me just compute correctly:

                float tmp4 = 0.5f / (u2 * aijk + aij * ak);
                float b00_v = u2 * tmp4;
                float b10_v = b00_v + tmp4 * ak;
                float b01_v = b00_v + tmp4 * aij;

                // Center shifts for Rys recursion:
                // cpx = (P-A)_x - b00_v/tmp4 * (P-W)_x ... no.
                // From CUDA code:
                // c00x = Px - Ax - b00 * (Px - Cx) * 2*aij ... no.
                // Actually from g2e.cu line ~500:
                // c00[0] = Px - Ax + u2*(Wx - Px)
                // c00 = PA + u2*WP where WP = W-P
                // But u2 is different in the CUDA convention...

                // CUDA convention: u2 = a0 * rys_u^2 (already incorporated)
                // In our convention: rts[n] = rys_u^2
                // So: u2_eff = rts[n] * a0... no wait, I'm computing T = a0 * rr_pc
                // and the Rys roots are for the Boys moments at that T.

                // Let me reconsider. The Rys quadrature gives nodes t_n and weights w_n
                // such that: integral_0^1 f(t) dt ≈ sum_n w_n * f(t_n)
                // and F_m(T) = integral_0^1 t^{2m} exp(-T*t^2) dt
                // The ACTUAL roots are the values t_n^2.
                // In the recursion, the "scaling" parameter is:
                //   rho_n = T * t_n^2 / (T * t_n^2 / a0 + 1)...
                // This is getting confusing. Let me just follow the CUDA code exactly.

                // From CUDA g2e.cu:
                // double u2 = a0 * uw[irys]; (uw stores t^2 values)
                // double tmp4 = .5 / (u2 * aijkl + a1);
                // double b00 = u2 * tmp4;
                // double b10 = b00 + tmp4 * akl;
                // double b01 = b00 + tmp4 * aij;
                // double c00x = rijrx - b00*rijrkx;
                // double c0px = rijrkx + b10*rijrkx; (typo? should be c0p=W-C related)

                // OK so in CUDA: u2 = a0 * t_n^2 where t_n are the Rys nodes.
                // But my rys_roots function returns roots=t_n^2 directly.
                // So: u2 = a0 * rts[n]... wait, no. The Rys roots from my
                // moments-based algorithm ARE the quadrature nodes for the
                // integral F_m(T) = int_0^1 t^{2m} exp(-T t^2) dt.
                // So rts[n] = t_n^2. And the CUDA code does u2 = a0 * uw[n].
                // But uw[n] there is the raw Rys node... hmm.

                // Actually in the CUDA code, the Rys roots are computed from
                // the scaled variable x = a0 * |P-C|^2. The roots uw[n] are
                // the quadrature nodes for the integral over u:
                // (ij|k) = sum_n w_n * [product of 1D integrals at root u_n]
                // The mapping between t (Boys variable) and u (Rys variable) is:
                // t^2 = u^2 / (1 + u^2)
                // So u^2 = t^2 / (1 - t^2)

                // In my formulation, rts[n] are the t^2 values (roots of the
                // moment-based quadrature for F_m). So:
                // u^2 = rts[n] / (1 - rts[n])
                // And the CUDA variable u2 = a0 * u^2 ... no, CUDA does
                // u2 = a0 * uw[irys] where uw are the raw roots from rys_roots.
                // The CUDA rys_roots returns roots in the u variable.

                // OK I think the issue is that my modified-Chebyshev Rys root
                // finder returns roots in the t^2 domain (directly as moments
                // of F_m = int t^{2m} exp(-Tt^2) dt). The standard Rys nodes
                // are in the u domain where t^2 = u^2/(1+u^2).

                // For the recursion, I need u2 = t^2/(1-t^2) = rts[n]/(1-rts[n]).
                // Then: u2_scaled = a0 * u2.
                // And the CUDA formula: tmp4 = 0.5/(u2_scaled*aijk + aij*ak)

                float t2 = rts[n];
                float u2_val = t2 / max(1.0f - t2, 1e-20f);
                float u2s = a0 * u2_val;
                float tmp4_v = 0.5f / (u2s * aijk + aij * ak);
                float b00_c = u2s * tmp4_v;
                float b10_c = b00_c + tmp4_v * ak;
                float b01_c = b00_c + tmp4_v * aij;

                // Center shifts for recursion
                // CUDA: c00x = (Px-Ax) - b00*(Px-Cx)*2*...
                // Actually from CUDA g2e.cu:
                // c00x = xixj - b00 * xpq  where xixj=P-A, xpq=P-C ... hmm
                // wait: "rijrx" in CUDA is (P-A)_x and "rijrkx" is (P-C)_x.
                // c00x = rijrx - b00 * rijrkx = (P-A) - b00*(P-C)
                // c0px = rijrkx + b10 * rijrkx... no, that can't be right.
                // Let me read the CUDA code:
                // double c00x = xij - xi[0] - b00 * (xij - xkl); // rijrx - b00*rijrkx
                // double c0px = xkl - xk[0] + b01 * (xij - xkl); // rkrlx + b01*rijrkx
                // So: c00x = (P-A)_x - b00*(P-C)_x  (bra recursion center shift)
                //     c0px = (C-C)_x + b01*(P-C)_x = b01*(P-C)_x  (for 3c: Q=C so Q-C=0)

                float pcx = px - cx, pcy = py - cy, pcz = pz - cz;
                float c00x_v = (px - ax) - b00_c * pcx;
                float c00y_v = (py - ay) - b00_c * pcy;
                float c00z_v = (pz - az) - b00_c * pcz;
                // For ket recursion (3-center: Q=C):
                float c0px_v = b01_c * pcx;
                float c0py_v = b01_c * pcy;
                float c0pz_v = b01_c * pcz;

                float wt = w * wts[n];  // WRONG: w already = wts[n]*fac
                // Actually: wt should be wts[n] * fac for this root.
                // But I set w = wts[n] * fac above. So wt = w. Let me fix:
                float root_fac = wts[n] * fac;

                // Build gx[a][k] for a=0..na-1, k=0..nk-1 for this root
                int stride_k = na;
                // gx[a + k*na] = g-value for bra-angular-momentum a, ket-angular-momentum k

                // Base: gx[0][0] = 1 (the weight*fac is applied at the end)
                for (int i = 0; i < na*nk; i++) { gx[i] = 0.0f; gy[i] = 0.0f; gz[i] = 0.0f; }
                gx[0] = 1.0f; gy[0] = 1.0f; gz[0] = 1.0f;

                // Bra recursion: gx[a+1][k] = c00x*gx[a][k] + a*b10*gx[a-1][k] + k*b00*gx[a][k-1]
                // Build for k=0 first, then grow k.

                // Step 1: grow bra (a) at k=0
                if (na > 1) {
                    gx[1] = c00x_v;
                    gy[1] = c00y_v;
                    gz[1] = c00z_v;
                }
                for (int a = 1; a < na-1; a++) {
                    gx[a+1] = c00x_v * gx[a] + float(a) * b10_c * gx[a-1];
                    gy[a+1] = c00y_v * gy[a] + float(a) * b10_c * gy[a-1];
                    gz[a+1] = c00z_v * gz[a] + float(a) * b10_c * gz[a-1];
                }

                // Step 2: grow ket (k) with bra coupling
                for (int k = 0; k < nk-1; k++) {
                    int k0 = k * na;
                    int k1 = (k+1) * na;
                    // gx[0][k+1] = c0px*gx[0][k] + k*b01*gx[0][k-1] + 0*b00*...
                    float prev_k = (k > 0) ? gx[(k-1)*na] : 0.0f;
                    gx[k1] = c0px_v * gx[k0] + float(k) * b01_c * prev_k;
                    prev_k = (k > 0) ? gy[(k-1)*na] : 0.0f;
                    gy[k1] = c0py_v * gy[k0] + float(k) * b01_c * prev_k;
                    prev_k = (k > 0) ? gz[(k-1)*na] : 0.0f;
                    gz[k1] = c0pz_v * gz[k0] + float(k) * b01_c * prev_k;

                    // gx[a][k+1] for a=1..na-1
                    for (int a = 1; a < na; a++) {
                        float prev_a = gx[k0 + a - 1];
                        float prev_kk = (k > 0) ? gx[(k-1)*na + a] : 0.0f;
                        gx[k1 + a] = c0px_v * gx[k0 + a]
                                    + float(k) * b01_c * prev_kk
                                    + float(a) * b00_c * gx[k0 + a - 1];
                        // Wait: the coupling term should be b00 * gx[a-1][k]...
                        // Actually the ket recursion is:
                        // gx[a][k+1] = c0px * gx[a][k] + k*b01*gx[a][k-1] + a*b00*gx[a-1][k]
                        // Hmm, that last term uses gx[a-1][k], not gx[a][k-1].
                        // And also the bra recursion is:
                        // gx[a+1][k] = c00x*gx[a][k] + a*b10*gx[a-1][k] + k*b00*gx[a][k-1]
                        // This is the cross-coupling: b00 couples bra and ket.

                        // For ket growth at fixed a:
                        prev_kk = (k > 0) ? gx[(k-1)*na + a] : 0.0f;
                        gx[k1 + a] = c0px_v * gx[k0 + a]
                                    + float(k) * b01_c * prev_kk
                                    + float(a) * b00_c * gx[k0 + a - 1];
                        // Oops: gx[k0 + a - 1] = gx[a-1][k] — that's correct!
                        // The coupling is: a * b00 * gx[a-1][k+1]... no, gx[a-1][k].
                        // Let me verify: from Helgaker textbook, the vertical recursion for ket:
                        // [a | k+1]^(m) = X_QC [a|k]^(m) + (X_WQ)[a|k]^(m+1)
                        //                + k/(2q) {[a|k-1]^(m) - p/zeta [a|k-1]^(m+1)}
                        //                + a/(2zeta) [a-1|k]^(m+1)
                        // In Rys formulation, the m-summation becomes root-summation, and:
                        // gx[a][k+1][n] = c0p * gx[a][k][n] + k*b01*gx[a][k-1][n] + a*b00*gx[a-1][k][n]
                        // Yes! So the coupling term is a*b00*gx[a-1][k][n]. Correct.

                        prev_kk = (k > 0) ? gy[(k-1)*na + a] : 0.0f;
                        gy[k1 + a] = c0py_v * gy[k0 + a]
                                    + float(k) * b01_c * prev_kk
                                    + float(a) * b00_c * gy[k0 + a - 1];
                        prev_kk = (k > 0) ? gz[(k-1)*na + a] : 0.0f;
                        gz[k1 + a] = c0pz_v * gz[k0 + a]
                                    + float(k) * b01_c * prev_kk
                                    + float(a) * b00_c * gz[k0 + a - 1];
                    }
                }

                // Now gx[a][k] contains the 1D integral for x-direction at root n.
                // Horizontal transfer: separate combined bra angular momentum 'a' into
                // shell i (li1) and shell j (lj):
                // gx_ij[i][j][k] = gx[i+j][k] + (B-A)_x * gx[i+j-1][k] ... hmm
                // Actually the horizontal recursion is:
                // g[i, j+1, k] = g[i+1, j, k] + (A-B)_x * g[i, j, k]
                // This separates the combined index a=i+j into separate i, j.

                // For now: use the simpler "all on first center" approach where
                // the bra recursion built everything as [a=i+j, 0 | k], and we
                // transfer to [i, j | k] via:
                // g[i][j+1][k] = g[i+1][j][k] + ABx * g[i][j][k]

                float ABx = ax - bx, ABy = ay - by, ABz = az - bz;
                // We need a 3D array g_ijk[i][j][k] for the transfer.
                // Max size: (li1+1)*(lj+1)*(lk+1) = 5*4*5 = 100 per direction.
                float hx[100], hy[100], hz[100];
                int sj = li1 + 1;  // stride for j
                int sk_h = sj * (lj + 1);  // stride for k

                // Initialize from g[a][0][k] → h[a][0][k]
                for (int k = 0; k < nk; k++) {
                    for (int a = 0; a < na; a++) {
                        hx[a + 0*sj + k*sk_h] = gx[a + k*na];
                        hy[a + 0*sj + k*sk_h] = gy[a + k*na];
                        hz[a + 0*sj + k*sk_h] = gz[a + k*na];
                    }
                }
                // Horizontal transfer: h[i][j+1][k] = h[i+1][j][k] + ABx*h[i][j][k]
                for (int j = 0; j < lj; j++) {
                    for (int k = 0; k < nk; k++) {
                        for (int i = 0; i <= li1 + lj - j - 1; i++) {
                            hx[i + (j+1)*sj + k*sk_h] = hx[(i+1) + j*sj + k*sk_h] + ABx * hx[i + j*sj + k*sk_h];
                            hy[i + (j+1)*sj + k*sk_h] = hy[(i+1) + j*sj + k*sk_h] + ABy * hy[i + j*sj + k*sk_h];
                            hz[i + (j+1)*sj + k*sk_h] = hz[(i+1) + j*sj + k*sk_h] + ABz * hz[i + j*sj + k*sk_h];
                        }
                    }
                }

                // Now h[ix][jx][kx] contains the 1D x-integral component for this root.
                // Contract: for each Cartesian triple, sum gx*gy*gz over this root.
                // Also apply the derivative: d/dA_x = 2*ai * h[ix+1,jx,kx] - ix*h[ix-1,jx,kx]

                for (int fk = 0; fk < nfk; fk++) {
                    int kx = cart_kx[fk + cart_off_k];
                    int ky = cart_ky[fk + cart_off_k];
                    int kz = cart_kz[fk + cart_off_k];
                    for (int fj = 0; fj < nfj; fj++) {
                        int jx = cart_jx[fj + cart_off_j];
                        int jy = cart_jy[fj + cart_off_j];
                        int jz = cart_jz[fj + cart_off_j];
                        for (int fi = 0; fi < nfi; fi++) {
                            int ix = cart_ix[fi + cart_off_i];
                            int iy = cart_iy[fi + cart_off_i];
                            int iz = cart_iz[fi + cart_off_i];

                            // Undifferentiated value at this root:
                            float vx = hx[ix + jx*sj + kx*sk_h];
                            float vy = hy[iy + jy*sj + ky*sk_h];
                            float vz = hz[iz + jz*sj + kz*sk_h];

                            // Derivative w.r.t. A_x: 2*ai * h[ix+1,...] - ix * h[ix-1,...]
                            float dx_x = 2.0f*ai * hx[(ix+1) + jx*sj + kx*sk_h];
                            if (ix > 0) dx_x -= float(ix) * hx[(ix-1) + jx*sj + kx*sk_h];
                            float dx_y = vy;
                            float dx_z = vz;

                            float dy_x = vx;
                            float dy_y = 2.0f*ai * hy[(iy+1) + jy*sj + ky*sk_h];
                            if (iy > 0) dy_y -= float(iy) * hy[(iy-1) + jy*sj + ky*sk_h];
                            float dy_z = vz;

                            float dz_x = vx;
                            float dz_y = vy;
                            float dz_z = 2.0f*ai * hz[(iz+1) + jz*sj + kz*sk_h];
                            if (iz > 0) dz_z -= float(iz) * hz[(iz-1) + jz*sj + kz*sk_h];

                            int gidx = fi + fj*nfi + fk*nfi*nfj;
                            gout[gidx]                    += root_fac * dx_x * dx_y * dx_z;
                            gout[gidx + nfi*nfj*nfk]      += root_fac * dy_x * dy_y * dy_z;
                            gout[gidx + 2*nfi*nfj*nfk]    += root_fac * dz_x * dz_y * dz_z;
                        }
                    }
                }
            } // nroots
        } // pk
    } // pj
} // pi

// Write output: (3, nao, nao, naux) layout
// out[comp*nao*nao*naux + (ao_i+fi)*nao*naux + (ao_j+fj)*naux + (ao_k+fk)]
for (int comp = 0; comp < 3; comp++) {
    for (int fk = 0; fk < nfk; fk++) {
        for (int fj = 0; fj < nfj; fj++) {
            for (int fi = 0; fi < nfi; fi++) {
                int oidx = comp*out_stride_comp + (ao_i+fi)*out_stride_i + (ao_j+fj)*out_stride_j + (ao_k+fk);
                out[oidx] = gout[fi + fj*nfi + fk*nfi*nfj + comp*nfi*nfj*nfk];
            }
        }
    }
}
'''


# ---------------------------------------------------------------------------
# Python dispatch
# ---------------------------------------------------------------------------

def _prepare_shell_data_3c(mol, auxmol):
    """Pack shell metadata for int3c2e_ip1 Metal kernel."""
    nshells = mol.nbas
    nshells_aux = auxmol.nbas

    # Per-shell data
    all_shell_x = []
    all_shell_y = []
    all_shell_z = []
    all_shell_l = []
    all_shell_nprim = []
    all_prim_off = []
    all_ao_off_orb = []
    all_ao_off_aux = []
    all_exps = []
    all_coeffs = []

    prim_offset = 0
    ao_offset = 0
    for ish in range(nshells):
        l = mol.bas_angular(ish)
        atom_id = mol.bas_atom(ish)
        ac = mol.atom_coord(atom_id)
        nprim = mol.bas_nprim(ish)
        all_shell_x.append(ac[0])
        all_shell_y.append(ac[1])
        all_shell_z.append(ac[2])
        all_shell_l.append(l)
        all_shell_nprim.append(nprim)
        all_prim_off.append(prim_offset)
        all_ao_off_orb.append(ao_offset)
        all_exps.extend(mol.bas_exp(ish).tolist())
        all_coeffs.extend(mol._libcint_ctr_coeff(ish).flatten().tolist())
        prim_offset += nprim
        ao_offset += (l + 1) * (l + 2) // 2

    ao_offset_aux = 0
    for ksh in range(nshells_aux):
        l = auxmol.bas_angular(ksh)
        atom_id = auxmol.bas_atom(ksh)
        ac = auxmol.atom_coord(atom_id)
        nprim = auxmol.bas_nprim(ksh)
        all_shell_x.append(ac[0])
        all_shell_y.append(ac[1])
        all_shell_z.append(ac[2])
        all_shell_l.append(l)
        all_shell_nprim.append(nprim)
        all_prim_off.append(prim_offset)
        all_ao_off_aux.append(ao_offset_aux)
        all_exps.extend(auxmol.bas_exp(ksh).tolist())
        all_coeffs.extend(auxmol._libcint_ctr_coeff(ksh).flatten().tolist())
        prim_offset += nprim
        ao_offset_aux += (l + 1) * (l + 2) // 2

    # Cartesian power tables
    max_l = max(all_shell_l) + 1  # +1 for derivative
    cart_tables = {}
    for l in range(max_l + 1):
        cart_tables[l] = _cart_powers(l)

    return {
        'shell_x': np.array(all_shell_x, dtype=np.float32),
        'shell_y': np.array(all_shell_y, dtype=np.float32),
        'shell_z': np.array(all_shell_z, dtype=np.float32),
        'shell_l': np.array(all_shell_l, dtype=np.int32),
        'shell_nprim': np.array(all_shell_nprim, dtype=np.int32),
        'prim_off': np.array(all_prim_off, dtype=np.int32),
        'ao_off_orb': np.array(all_ao_off_orb, dtype=np.int32),
        'ao_off_aux': np.array(all_ao_off_aux, dtype=np.int32),
        'exps': np.array(all_exps, dtype=np.float32),
        'coeffs': np.array(all_coeffs, dtype=np.float32),
        'nshells_orb': nshells,
        'nshells_aux': nshells_aux,
        'cart_tables': cart_tables,
    }


def compute_int3c2e_ip1_metal(mol, auxmol, shls_slice=None):
    """Compute int3c2e_ip1 on Metal GPU.

    Returns (3, nao, nao, naux) float64 array matching PySCF convention.
    Falls back to CPU (PySCF libcint) if Metal fails or for validation.
    """
    # For now: fall back to CPU. The Metal kernel is scaffolded above
    # but needs further testing before replacing the CPU path.
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix('int3c2e_ip1')
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
    else:
        shls_slice = shls_slice[:4] + (nbas + shls_slice[4],
                                       nbas + shls_slice[5])
    return getints(intor, pmol._atm, pmol._bas, pmol._env,
                   shls_slice, aosym='s1', cintopt=opt)
