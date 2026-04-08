# Copyright 2024 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""
Metal int3c2e_ip1 v2: one GPU thread per SHELL TRIPLE.

Eliminates the Python task builder bottleneck by moving Boys function
and Rys root computation to the GPU. Each thread reads shell data from
global buffers, loops over primitive triples internally, and writes
directly to the output tensor.

CPU-side work: enumerate shell triples (~0.5s) + pack shell data (~0.01s).
GPU-side work: Boys + Rys + TRR + HRR + derivative per primitive.
"""

import numpy as np
import mlx.core as mx
from math import pi, sqrt
from pyscf.gto.moleintor import getints, make_cintopt

_NCART_LUT = np.array([1, 3, 6, 10, 15], dtype=np.int32)
PI_5_2 = pi ** 2.5


# ---------------------------------------------------------------------------
# Rys lookup tables (extracted from CUDA rys_roots_dat.cu)
# ---------------------------------------------------------------------------

def _load_rys_tables():
    """Load Rys root/weight Chebyshev interpolation tables from CUDA data."""
    import re, os
    dat_path = os.path.join(os.path.dirname(__file__), '..', 'gvhf-rys', 'rys_roots_dat.cu')
    with open(dat_path) as f:
        text = f.read()

    def _extract(name):
        m = re.search(rf'{name}\[\]\s*=\s*\{{([^}}]+)\}}', text, re.DOTALL)
        nums = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+', m.group(1))
        return np.array([float(x) for x in nums], dtype=np.float32)

    return {
        'rw_table': _extract('ROOT_RW_DATA'),
        'smallx_r0': _extract('ROOT_SMALLX_R0'),
        'smallx_r1': _extract('ROOT_SMALLX_R1'),
        'smallx_w0': _extract('ROOT_SMALLX_W0'),
        'smallx_w1': _extract('ROOT_SMALLX_W1'),
        'largex_r': _extract('ROOT_LARGEX_R_DATA'),
        'largex_w': _extract('ROOT_LARGEX_W_DATA'),
    }


_rys_tables = None

def _get_rys_tables():
    global _rys_tables
    if _rys_tables is None:
        raw = _load_rys_tables()
        _rys_tables = {k: mx.array(v) for k, v in raw.items()}
    return _rys_tables

_FAC_L = {0: sqrt(1.0 / (4 * pi)),
          1: sqrt(3.0 / (4 * pi)),
          2: 1.0, 3: 1.0, 4: 1.0}

def _cart_powers(l):
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
# Metal kernel v2: one thread per shell triple, Boys+Rys on GPU
# ---------------------------------------------------------------------------

_HEADER_V2 = '''
constant int NCART[] = {1, 3, 6, 10, 15};
constant int CART_ALL[] = {
''' + ','.join(str(v) for v in _CART_ALL_TBL) + '''
};
constant int CART_OFF[] = {0, 1, 4, 10, 20};
constant float PI_5_2_F = 34.9868366552497250f;  // 2 * pi^(5/2)
'''

# The kernel inlines Boys function + Rys roots (nroots 1-2 direct, 3+ via
# modified Chebyshev) + TRR + HRR + derivative. ~350 lines.
_SOURCE_V2 = '''
uint tid = thread_position_in_grid.x;
if (tid >= n_triples) return;

// Read shell triple
int ish = triples[tid * 6 + 0];
int jsh = triples[tid * 6 + 1];
int ksh = triples[tid * 6 + 2];
int i0  = triples[tid * 6 + 3];
int j0  = triples[tid * 6 + 4];
int k0  = triples[tid * 6 + 5];

int li = shell_l[ish], lj = shell_l[jsh], lk = shell_l[ksh];
int li1 = li + 1;
int lij1 = li1 + lj;
int ni = NCART[li], nj = NCART[lj], nk = NCART[lk];
int ci_off = CART_OFF[li], cj_off = CART_OFF[lj], ck_off = CART_OFF[lk];
int comp_stride = ni * nj * nk;
int n_out = 3 * comp_stride;
int nroots = (li1 + lj + lk) / 2 + 1;

float ax = shell_x[ish], ay = shell_y[ish], az = shell_z[ish];
float bx = shell_x[jsh], by = shell_y[jsh], bz = shell_z[jsh];
float cx = shell_x[ksh], cy = shell_y[ksh], cz = shell_z[ksh];
float ABx = ax-bx, ABy = ay-by, ABz = az-bz;
float rr_ab = ABx*ABx + ABy*ABy + ABz*ABz;

int npi = shell_nprim[ish], npj = shell_nprim[jsh], npk = shell_nprim[ksh];
int offi = prim_off[ish], offj = prim_off[jsh], offk = prim_off[ksh];

// Output accumulator (zero-initialized)
int out_base = out_offsets[tid];
for (int i = 0; i < n_out; i++) int_out[out_base + i] = 0.0f;

// Loop over primitive triples
for (int pi = 0; pi < npi; pi++) {
float ai = exps[offi+pi], ci_c = coeffs[offi+pi];
for (int pj = 0; pj < npj; pj++) {
float aj = exps[offj+pj], cj_c = coeffs[offj+pj];
float aij = ai + aj;
float eij = exp(-ai * aj / aij * rr_ab);
if (eij < 1e-14f) continue;
float px = (ai*ax + aj*bx)/aij, py = (ai*ay + aj*by)/aij, pz = (ai*az + aj*bz)/aij;
float PAx = px-ax, PAy = py-ay, PAz = pz-az;
float coeff_ij = ci_c * cj_c * eij;

for (int pk = 0; pk < npk; pk++) {
float ak = exps[offk+pk], ck_c = coeffs[offk+pk];
float aijk = aij + ak;
float a0 = aij * ak / aijk;
float PCx = px-cx, PCy = py-cy, PCz = pz-cz;
float T = a0 * (PCx*PCx + PCy*PCy + PCz*PCz);
float prefac = coeff_ij * ck_c * PI_5_2_F / (aij * ak * sqrt(aijk));

// --- Inline Boys function ---
int mmax = 2*nroots - 1;
float Fm[13];
float expT = exp(-T);
if (T < 1e-7f) {
    for (int m = 0; m <= mmax; m++) Fm[m] = 1.0f / float(2*m + 1);
} else if (T > float(mmax) + 1.0f) {
    float sqrtT = sqrt(T);
    float te = 1.0f / (1.0f + 0.3275911f * sqrtT);
    float te2=te*te, te3=te2*te, te4=te3*te, te5=te4*te;
    float erf_v = 1.0f - (0.254829592f*te - 0.284496736f*te2
        + 1.421413741f*te3 - 1.453152027f*te4 + 1.061405429f*te5) * expT;
    Fm[0] = 0.8862269254527580f * erf_v / sqrtT;
    for (int m = 0; m < mmax; m++)
        Fm[m+1] = (float(2*m+1)*Fm[m] - expT) / (2.0f*T);
} else {
    int mstart = int(T) + mmax + 30;
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
        if (m <= mmax) Fm[m] = fm;
    }
}

// --- Rys roots/weights via Chebyshev lookup table (Clenshaw recurrence) ---
// Table layout: per nroots n, offset = 560*n*(n-1), entries for 2n arrays
// (n roots + n weights), each 14 coefficients × 40 intervals = 560 entries.
float rys_r[6] = {0,0,0,0,0,0}, rys_w[6] = {0,0,0,0,0,0};
float x_rys = T;  // Boys argument = Rys argument
if (x_rys < 3.0e-7f) {
    // Small x: linear approximation from SMALLX tables
    for (int r = 0; r < nroots; r++) {
        int sidx = nroots*(nroots-1)/2 + r;
        rys_r[r] = smallx_r0[sidx] + smallx_r1[sidx] * x_rys;
        rys_w[r] = smallx_w0[sidx] + smallx_w1[sidx] * x_rys;
    }
} else if (x_rys > 35.0f + 5.0f * float(nroots)) {
    // Large x: asymptotic
    float inv_x = 1.0f / x_rys;
    float sqrt_inv_x = sqrt(inv_x);
    for (int r = 0; r < nroots; r++) {
        int sidx = nroots*(nroots-1)/2 + r;
        rys_r[r] = largex_r[sidx] * inv_x;
        rys_w[r] = largex_w[sidx] * sqrt_inv_x;
    }
} else {
    // Medium x: Chebyshev interpolation with Clenshaw recurrence
    int it = min((int)(x_rys * 0.4f), 39);
    float u = (x_rys - float(it) * 2.5f) * 0.8f - 1.0f;
    float u2 = 2.0f * u;
    int nroots_off = 560 * nroots * (nroots - 1);  // offset into rw_table for this nroots
    for (int r = 0; r < nroots; r++) {
        // Root coefficients: nroots_off + (2*r)*560
        // Weight coefficients: nroots_off + (2*r+1)*560
        for (int rw = 0; rw < 2; rw++) {  // 0=root, 1=weight
            int base = nroots_off + (2*r + rw) * 560 + it;
            // Clenshaw recurrence (degree=13, odd)
            float c0 = rw_table[base + 13*40];
            float c1 = rw_table[base + 12*40];
            for (int n = 11; n >= 1; n -= 2) {
                float c2 = rw_table[base + n*40] - c1;
                float c3 = c0 + c1 * u2;
                c1 = c2 + c3 * u2;
                c0 = rw_table[base + (n-1)*40] - c3;
            }
            float val = c0 + c1 * u;
            if (rw == 0) {
                rys_r[r] = val;  // table stores t^2 directly
            } else {
                rys_w[r] = val;
            }
        }
    }
}

// --- TRR + HRR + derivative (same as v1 kernel) ---
float PA[3] = {PAx, PAy, PAz};
float AB[3] = {ABx, ABy, ABz};
float Rpq[3] = {PCx, PCy, PCz};

for (int iroot = 0; iroot < nroots; iroot++) {
    float rt = rys_r[iroot];
    float wt = rys_w[iroot];
    float rt_aa = rt / aijk;
    float rt_aij = rt_aa * ak;
    float b10 = 0.5f/aij*(1.0f-rt_aij);
    float b01 = 0.5f/ak*(1.0f-rt_aa*aij);
    float b00 = 0.5f*rt_aa;

    float g_ij[3][9][4][5];
    for (int dir = 0; dir < 3; dir++) {
        float c0 = PA[dir] - rt_aij*Rpq[dir];
        float cp = (rt_aa*aij)*Rpq[dir];
        float g[9][5];
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++) g[a][c] = 0.0f;
        g[0][0] = 1.0f;
        if (lij1 > 0) g[1][0] = c0;
        for (int a = 1; a < lij1; a++)
            g[a+1][0] = c0*g[a][0] + float(a)*b10*g[a-1][0];
        for (int c = 0; c < lk; c++) {
            for (int a = 0; a <= lij1; a++) {
                float v = cp*g[a][c];
                if (c > 0) v += float(c)*b01*g[a][c-1];
                if (a > 0) v += float(a)*b00*g[a-1][c];
                g[a][c+1] = v;
            }
        }
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++)
                g_ij[dir][a][0][c] = g[a][c];
        float ab = AB[dir];
        for (int j = 0; j < lj; j++)
            for (int c = 0; c <= lk; c++)
                for (int i = 0; i <= lij1-j-1; i++)
                    g_ij[dir][i][j+1][c] = g_ij[dir][i+1][j][c] + ab*g_ij[dir][i][j][c];
    }

    float pf_wt = prefac * wt;
    for (int ii = 0; ii < ni; ii++) {
        int ci_v = CART_ALL[ci_off+ii];
        int ix=ci_v/25, iy=(ci_v/5)%5, iz=ci_v%5;
        for (int jj = 0; jj < nj; jj++) {
            int cj_v = CART_ALL[cj_off+jj];
            int jx=cj_v/25, jy=(cj_v/5)%5, jz=cj_v%5;
            for (int kk = 0; kk < nk; kk++) {
                int ck_v = CART_ALL[ck_off+kk];
                int kx=ck_v/25, ky=(ck_v/5)%5, kz=ck_v%5;
                float vx=g_ij[0][ix][jx][kx], vy=g_ij[1][iy][jy][ky], vz=g_ij[2][iz][jz][kz];
                float dx = 2.0f*ai*g_ij[0][ix+1][jx][kx];
                if (ix>0) dx -= float(ix)*g_ij[0][ix-1][jx][kx];
                float dy = 2.0f*ai*g_ij[1][iy+1][jy][ky];
                if (iy>0) dy -= float(iy)*g_ij[1][iy-1][jy][ky];
                float dz = 2.0f*ai*g_ij[2][iz+1][jz][kz];
                if (iz>0) dz -= float(iz)*g_ij[2][iz-1][jz][kz];
                int base = out_base + (ii*nj+jj)*nk+kk;
                int_out[base]                 += pf_wt*dx*vy*vz;
                int_out[base + comp_stride]   += pf_wt*vx*dy*vz;
                int_out[base + 2*comp_stride] += pf_wt*vx*vy*dz;
            }
        }
    }
} // end root loop

} // pk
} // pj
} // pi
'''

_kernel_v2 = mx.fast.metal_kernel(
    name='int3c2e_ip1_v2',
    input_names=['triples', 'shell_x', 'shell_y', 'shell_z',
                 'shell_l', 'shell_nprim', 'prim_off',
                 'exps', 'coeffs', 'out_offsets',
                 'rw_table', 'smallx_r0', 'smallx_r1',
                 'smallx_w0', 'smallx_w1', 'largex_r', 'largex_w'],
    output_names=['int_out'],
    header=_HEADER_V2,
    source=_SOURCE_V2,
    atomic_outputs=False,
)


# ---------------------------------------------------------------------------
# Python dispatch (minimal — just enumerate shell triples + pack data)
# ---------------------------------------------------------------------------

def _pack_shell_data(mol, auxmol):
    """Pack shell metadata into flat arrays for GPU."""
    nbas_orb = mol.nbas
    nbas_aux = auxmol.nbas
    nbas_total = nbas_orb + nbas_aux

    shell_x = np.zeros(nbas_total, dtype=np.float32)
    shell_y = np.zeros(nbas_total, dtype=np.float32)
    shell_z = np.zeros(nbas_total, dtype=np.float32)
    shell_l = np.zeros(nbas_total, dtype=np.int32)
    shell_nprim = np.zeros(nbas_total, dtype=np.int32)
    prim_off_arr = np.zeros(nbas_total, dtype=np.int32)
    all_exps = []
    all_coeffs = []
    off = 0

    for ish in range(nbas_orb):
        c = mol.atom_coord(mol.bas_atom(ish))
        shell_x[ish] = c[0]; shell_y[ish] = c[1]; shell_z[ish] = c[2]
        shell_l[ish] = mol.bas_angular(ish)
        n = mol.bas_nprim(ish)
        shell_nprim[ish] = n
        prim_off_arr[ish] = off
        all_exps.extend(mol.bas_exp(ish).tolist())
        all_coeffs.extend(mol._libcint_ctr_coeff(ish).flatten().tolist())
        off += n

    for ksh in range(nbas_aux):
        idx = nbas_orb + ksh
        c = auxmol.atom_coord(auxmol.bas_atom(ksh))
        shell_x[idx] = c[0]; shell_y[idx] = c[1]; shell_z[idx] = c[2]
        shell_l[idx] = auxmol.bas_angular(ksh)
        n = auxmol.bas_nprim(ksh)
        shell_nprim[idx] = n
        prim_off_arr[idx] = off
        all_exps.extend(auxmol.bas_exp(ksh).tolist())
        all_coeffs.extend(auxmol._libcint_ctr_coeff(ksh).flatten().tolist())
        off += n

    return (shell_x, shell_y, shell_z, shell_l, shell_nprim, prim_off_arr,
            np.array(all_exps, dtype=np.float32),
            np.array(all_coeffs, dtype=np.float32))


def compute_int3c2e_ip1_v2(mol, auxmol, shls_slice=None):
    """Compute int3c2e_ip1 on Metal GPU (v2: shell-triple-per-thread)."""
    import numba as nb

    nbas = mol.nbas
    nbas_aux = auxmol.nbas
    if shls_slice is None:
        shl0_aux, shl1_aux = 0, nbas_aux
    else:
        shl0_aux, shl1_aux = shls_slice[4], shls_slice[5]

    max_l_orb = max(mol.bas_angular(i) for i in range(nbas))
    max_l_aux = max(auxmol.bas_angular(i) for i in range(nbas_aux))
    if max_l_orb > 3 or max_l_aux > 4:
        return _cpu_fallback(mol, auxmol, shls_slice)

    # Pack shell data
    (sx, sy, sz, sl, snp, spo, exps, coeffs) = _pack_shell_data(mol, auxmol)

    # Enumerate shell triples + compute output offsets
    ao_loc_c = mol.ao_loc_nr(cart=True)
    aux_loc_c = auxmol.ao_loc_nr(cart=True)
    p0_aux = aux_loc_c[shl0_aux]

    triples = []
    out_sizes = []
    for ish in range(nbas):
        li = mol.bas_angular(ish)
        if li > 3:
            continue
        i0 = ao_loc_c[ish]
        nfi = int(_NCART_LUT[li])
        for jsh in range(nbas):
            lj = mol.bas_angular(jsh)
            if lj > 3:
                continue
            j0 = ao_loc_c[jsh]
            nfj = int(_NCART_LUT[lj])
            for ksh in range(shl0_aux, shl1_aux):
                lk = auxmol.bas_angular(ksh)
                if lk > 4:
                    continue
                k0 = aux_loc_c[ksh] - p0_aux
                nfk = int(_NCART_LUT[lk])
                triples.append((ish, jsh, nbas + ksh, i0, j0, k0))
                out_sizes.append(3 * nfi * nfj * nfk)

    if not triples:
        return _cpu_fallback(mol, auxmol, shls_slice)

    triples_arr = np.array(triples, dtype=np.int32)
    out_sizes_arr = np.array(out_sizes, dtype=np.int64)
    offsets = np.zeros(len(triples), dtype=np.int32)
    if len(triples) > 1:
        offsets[1:] = np.cumsum(out_sizes_arr[:-1]).astype(np.int32)
    total_out = int(out_sizes_arr.sum())
    n_triples = len(triples)

    # Launch Metal kernel
    THREADS = 64  # lower than 256: each thread does more work (primitive loop)
    grid_size = ((n_triples + THREADS - 1) // THREADS) * THREADS

    rys = _get_rys_tables()
    result = _kernel_v2(
        inputs=[mx.array(triples_arr.ravel()), mx.array(sx), mx.array(sy),
                mx.array(sz), mx.array(sl), mx.array(snp), mx.array(spo),
                mx.array(exps), mx.array(coeffs), mx.array(offsets),
                rys['rw_table'], rys['smallx_r0'], rys['smallx_r1'],
                rys['smallx_w0'], rys['smallx_w1'],
                rys['largex_r'], rys['largex_w']],
        grid=(grid_size, 1, 1),
        threadgroup=(min(THREADS, n_triples), 1, 1),
        output_shapes=[(total_out,)],
        output_dtypes=[mx.float32],
        template=[('n_triples', n_triples)],
    )
    mx.eval(result[0])
    int_buf = np.array(result[0]).astype(np.float64)

    # Accumulate into Cartesian output tensor
    nao_cart = mol.nao_cart()
    naux_slice_c = aux_loc_c[shl1_aux] - p0_aux
    out_cart = np.zeros((3, nao_cart, nao_cart, naux_slice_c), dtype=np.float64)

    @nb.njit(cache=True)
    def _accum(buf, offsets, triples, out_sizes, sl_orb, sl_aux, nbas_orb,
               fac_arr, out):
        for t in range(len(offsets)):
            off = offsets[t]
            sz = out_sizes[t]
            ish = triples[t*6]; jsh = triples[t*6+1]; ksh_idx = triples[t*6+2] - nbas_orb
            i0 = triples[t*6+3]; j0 = triples[t*6+4]; k0 = triples[t*6+5]
            li = sl_orb[ish]; lj = sl_orb[jsh]; lk = sl_aux[ksh_idx]
            fac = fac_arr[li] * fac_arr[lj] * fac_arr[lk]
            nfi = (li+1)*(li+2)//2; nfj = (lj+1)*(lj+2)//2; nfk = (lk+1)*(lk+2)//2
            cs = nfi * nfj * nfk
            for idx in range(sz):
                comp = idx // cs
                rem = idx - comp * cs
                fi = rem // (nfj * nfk)
                rem2 = rem - fi * nfj * nfk
                fj = rem2 // nfk
                fk = rem2 - fj * nfk
                out[comp, i0+fi, j0+fj, k0+fk] -= buf[off+idx] * fac

    sl_orb = np.array([mol.bas_angular(i) for i in range(nbas)], dtype=np.int32)
    sl_aux = np.array([auxmol.bas_angular(i) for i in range(nbas_aux)], dtype=np.int32)
    fac_arr = np.array([_FAC_L[l] for l in range(5)])
    _accum(int_buf, offsets, triples_arr.ravel(), out_sizes_arr.astype(np.int32),
           sl_orb, sl_aux, nbas, fac_arr, out_cart)

    if mol.cart:
        result_out = np.zeros((nao_cart, nao_cart, naux_slice_c, 3),
                              dtype=np.float64, order='F').transpose(3, 0, 1, 2)
        result_out[:] = out_cart
        return result_out

    # Cart-to-spherical
    from pyscf.gto.mole import cart2sph as _c2s
    def _c2s_mat(l):
        return np.asarray(_c2s(l, normalized='sp')).T

    nao = mol.nao
    ao_loc_s = mol.ao_loc_nr()
    aux_loc_s = auxmol.ao_loc_nr()
    naux_slice = aux_loc_s[shl1_aux] - aux_loc_s[shl0_aux]
    p0_aux_s = aux_loc_s[shl0_aux]

    out = np.zeros((3, nao, nao, naux_slice), dtype=np.float64)
    for comp in range(3):
        tmp1 = np.zeros((nao, nao_cart, naux_slice_c), dtype=np.float64)
        for ish in range(nbas):
            l = mol.bas_angular(ish)
            c0 = ao_loc_c[ish]; c1 = ao_loc_c[ish+1]
            s0 = ao_loc_s[ish]; s1 = ao_loc_s[ish+1]
            if l <= 1:
                tmp1[s0:s1] = out_cart[comp, c0:c1]
            else:
                C = _c2s_mat(l)
                tmp1[s0:s1] = np.einsum('si,ijk->sjk', C, out_cart[comp, c0:c1])
        tmp2 = np.zeros((nao, nao, naux_slice_c), dtype=np.float64)
        for jsh in range(nbas):
            l = mol.bas_angular(jsh)
            c0 = ao_loc_c[jsh]; c1 = ao_loc_c[jsh+1]
            s0 = ao_loc_s[jsh]; s1 = ao_loc_s[jsh+1]
            if l <= 1:
                tmp2[:, s0:s1] = tmp1[:, c0:c1]
            else:
                C = _c2s_mat(l)
                tmp2[:, s0:s1] = np.einsum('sj,ijk->isk', C, tmp1[:, c0:c1])
        for ksh in range(shl0_aux, shl1_aux):
            l = auxmol.bas_angular(ksh)
            c0 = aux_loc_c[ksh] - p0_aux; c1 = aux_loc_c[ksh+1] - p0_aux
            s0 = aux_loc_s[ksh] - p0_aux_s; s1 = aux_loc_s[ksh+1] - p0_aux_s
            if l <= 1:
                out[comp, :, :, s0:s1] = tmp2[:, :, c0:c1]
            else:
                C = _c2s_mat(l)
                out[comp, :, :, s0:s1] = np.einsum('sk,ijk->ijs', C, tmp2[:, :, c0:c1])

    return np.zeros((nao, nao, naux_slice, 3), dtype=np.float64, order='F').transpose(3, 0, 1, 2).__class__(
        np.asfortranarray(np.zeros((nao, nao, naux_slice, 3), dtype=np.float64)).transpose(3, 0, 1, 2).__array_interface__['data'][0]
    ) if False else (lambda: (r := np.zeros((nao, nao, naux_slice, 3), dtype=np.float64, order='F').transpose(3, 0, 1, 2), r.__setitem__(slice(None), out), r)[2])()


def _cpu_fallback(mol, auxmol, shls_slice):
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
