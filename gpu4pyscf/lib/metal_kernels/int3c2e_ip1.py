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
import numba as nb


@nb.njit(cache=True)
def _accumulate_cart_nb(int_buf, offsets, m_i0, m_j0, m_k0, m_ts, m_tc,
                        m_nfi, m_nfj, m_nfk, m_nout, m_li, m_lj, m_lk,
                        fac_tbl, out):
    """Numba-JIT Phase C: sum primitives per shell triple into output tensor."""
    n_meta = len(m_i0)
    for m in range(n_meta):
        i0 = m_i0[m]
        j0 = m_j0[m]
        k0 = m_k0[m]
        t_start = m_ts[m]
        t_count = m_tc[m]
        nfi = m_nfi[m]
        nfj = m_nfj[m]
        nfk = m_nfk[m]
        n_out = m_nout[m]
        fac = fac_tbl[m_li[m]] * fac_tbl[m_lj[m]] * fac_tbl[m_lk[m]]
        comp_stride = nfi * nfj * nfk
        off_start = offsets[t_start]
        # Sum all primitive contributions
        for idx in range(n_out):
            val = 0.0
            for t in range(t_count):
                val += int_buf[off_start + t * n_out + idx]
            comp = idx // comp_stride
            rem = idx - comp * comp_stride
            fi = rem // (nfj * nfk)
            rem2 = rem - fi * nfj * nfk
            fj = rem2 // nfk
            fk = rem2 - fj * nfk
            out[comp, i0 + fi, j0 + fj, k0 + fk] -= val * fac


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


# (id(mol), shl0, shl1) -> cart2sph plan
# {
#   'copy':   list of (s0, s1, c0, c1) contiguous slice copies for l<=1,
#   'groups': dict l -> {'cart': int[n*ncart], 'sph': int[n*nsph],
#                         'C': (nsph, ncart), 'n': n, 'ncart', 'nsph'}
# }
# l<=1 shells stay as per-shell slice copies (contiguous memcpy is fast).
# l>=2 shells group by l so a single matmul handles every shell with that
# angular momentum (dropping ~50 einsum calls per axis to ~3).
_C2S_PLAN_CACHE = {}


def _cart2sph_plan(mol, ao_loc_c, ao_loc_s, shl0=None, shl1=None):
    """Build / fetch cart->sph plan for a shell slice, cached by mol+range.

    Returns both CPU-side arrays and GPU-side MLX arrays (mx.array). The
    MLX variants stay resident on GPU across calls so the hot path pays
    zero upload cost for the transform matrices and index arrays.
    """
    if shl0 is None:
        shl0 = 0
    if shl1 is None:
        shl1 = mol.nbas
    key = (id(mol), shl0, shl1)
    hit = _C2S_PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    from pyscf.gto.mole import cart2sph as _c2s
    c_start = ao_loc_c[shl0]
    s_start = ao_loc_s[shl0]
    copies = []
    group_cart, group_sph = {}, {}
    for ish in range(shl0, shl1):
        l = mol.bas_angular(ish)
        c0 = int(ao_loc_c[ish] - c_start)
        c1 = int(ao_loc_c[ish + 1] - c_start)
        s0 = int(ao_loc_s[ish] - s_start)
        s1 = int(ao_loc_s[ish + 1] - s_start)
        if l <= 1:
            copies.append((s0, s1, c0, c1))
        else:
            group_cart.setdefault(l, []).append(np.arange(c0, c1, dtype=np.int32))
            group_sph.setdefault(l, []).append(np.arange(s0, s1, dtype=np.int32))
    plan = {'copy': copies, 'groups': {}, 'groups_mx': {}}
    for l, cart_list in group_cart.items():
        C = np.asarray(_c2s(l, normalized='sp')).T  # (nsph, ncart)
        cart_idx = np.concatenate(cart_list)
        sph_idx = np.concatenate(group_sph[l])
        plan['groups'][l] = {
            'cart': cart_idx,
            'sph': sph_idx,
            'C': C,
            'n': len(cart_list),
            'ncart': C.shape[1],
            'nsph': C.shape[0],
        }
        # Pre-upload GPU copies (f32) — MLX stays resident across calls.
        C_T_mx = mx.array(C.T.astype(np.float32))
        cart_idx_mx = mx.array(cart_idx)
        sph_idx_mx = mx.array(sph_idx)
        mx.eval(C_T_mx, cart_idx_mx, sph_idx_mx)
        plan['groups_mx'][l] = {
            'cart': cart_idx_mx,
            'sph': sph_idx_mx,
            'C_T': C_T_mx,
            'n': len(cart_list),
            'ncart': C.shape[1],
            'nsph': C.shape[0],
        }
    _C2S_PLAN_CACHE[key] = plan
    return plan


def _cart2sph_gpu(out_cart_np, orb_plan, aux_plan, nao, nao_cart,
                   naux_slice_c, naux_slice):
    """Apply the three cart->sph axis transforms on GPU via MLX.

    Input is (3, nao_cart, nao_cart, naux_slice_c) f64 on CPU; output is
    (3, nao, nao, naux_slice) f64 on CPU. All three axis transforms run
    in f32 on GPU and the result is cast back to f64 — matches the f32
    precision of the upstream int_buf, so there is no extra accuracy
    loss beyond what the kernel already produced.
    """
    oc = mx.array(out_cart_np.astype(np.float32))

    tmp1 = mx.zeros((3, nao, nao_cart, naux_slice_c), dtype=mx.float32)
    for s0, s1, c0, c1 in orb_plan['copy']:
        tmp1[:, s0:s1] = oc[:, c0:c1]
    for g in orb_plan['groups_mx'].values():
        sub = oc[:, g['cart']].reshape(3, g['n'], g['ncart'], nao_cart, naux_slice_c)
        tr = mx.tensordot(sub, g['C_T'], axes=([2], [0]))
        tmp1[:, g['sph']] = mx.transpose(tr, (0, 1, 4, 2, 3)).reshape(
            3, g['n'] * g['nsph'], nao_cart, naux_slice_c)

    tmp2 = mx.zeros((3, nao, nao, naux_slice_c), dtype=mx.float32)
    for s0, s1, c0, c1 in orb_plan['copy']:
        tmp2[:, :, s0:s1] = tmp1[:, :, c0:c1]
    for g in orb_plan['groups_mx'].values():
        sub = tmp1[:, :, g['cart']].reshape(3, nao, g['n'], g['ncart'], naux_slice_c)
        tr = mx.tensordot(sub, g['C_T'], axes=([3], [0]))
        tmp2[:, :, g['sph']] = mx.transpose(tr, (0, 1, 2, 4, 3)).reshape(
            3, nao, g['n'] * g['nsph'], naux_slice_c)

    out_mx = mx.zeros((3, nao, nao, naux_slice), dtype=mx.float32)
    for s0, s1, c0, c1 in aux_plan['copy']:
        out_mx[:, :, :, s0:s1] = tmp2[:, :, :, c0:c1]
    for g in aux_plan['groups_mx'].values():
        sub = tmp2[:, :, :, g['cart']].reshape(3, nao, nao, g['n'], g['ncart'])
        tr = sub @ g['C_T']
        out_mx[:, :, :, g['sph']] = tr.reshape(3, nao, nao, g['n'] * g['nsph'])
    mx.eval(out_mx)
    return np.asarray(out_mx).astype(np.float64)

# Task layout (36 floats per primitive triple).
# f32 inputs with f64-accurate Rys roots (truncated to f32 after Clenshaw).
# Chebyshev table accurate to 1e-7 for nroots ≤ 5 (threshold=99).
TASK_STRIDE = 36
#  [0:2]    aij (f64e: hi, lo)
#  [2:4]    ak  (f64e: hi, lo)
#  [4:10]   PA  (3 × f64e: hi,lo pairs)
#  [10:16]  PC  (3 × f64e)
#  [16:22]  AB  (3 × f64e)
#  [22:24]  prefac (f64e)
#  [24:26]  ai  (f64e: primitive exponent for derivative)
#  [26:29]  li, lj, lk  (int-in-float, single)
#  [29]     nroots (int-in-float, single)
#  [30:42]  roots[0..5]  (6 × f64e: hi,lo pairs)
#  [42:54]  weights[0..5] (6 × f64e)
#  [54:60]  Rpq (3 × f64e, = PC)
#  [60:64]  reserved


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

// Emulated f64 arithmetic using (hi, lo) float pairs.
// Achieves ~15 digits of precision (full IEEE f64 equivalent).
// Based on Knuth two-sum and Dekker splitting.
// Performance: ~1/18 of native f32 throughput.
struct f64e { float hi; float lo; };

f64e f64e_from_f32(float a) { return {a, 0.0f}; }
f64e f64e_from_pair(float hi, float lo) { return {hi, lo}; }
float f64e_to_f32(f64e a) { return a.hi + a.lo; }

// Knuth two-sum: exact addition
f64e f64e_add(f64e a, f64e b) {
    float s = a.hi + b.hi;
    float v = s - a.hi;
    float e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    float s2 = s + e;
    return {s2, e - (s2 - s)};
}

// FMA-based exact multiplication
f64e f64e_mul(f64e a, f64e b) {
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    float s = p + e;
    return {s, e - (s - p)};
}

// Scalar * f64e (when scalar is exact in f32)
f64e f64e_mul_s(float a, f64e b) {
    float p = a * b.hi;
    float e = fma(a, b.hi, -p) + a * b.lo;
    float s = p + e;
    return {s, e - (s - p)};
}

// Approximate sqrt (sufficient for prefac computation)
f64e f64e_sqrt(f64e a) {
    float s = sqrt(a.hi);
    float e = (a.hi - s*s + a.lo) / (2.0f * s);
    return {s + e, 0.0f - ((s + e) - s - e)};
}

// Approximate reciprocal
f64e f64e_inv(f64e a) {
    float r = 1.0f / a.hi;
    float e = fma(-a.hi, r, 1.0f) * r - a.lo * r * r;
    return {r + e, 0.0f - ((r + e) - r - e)};
}

// Division via multiply by reciprocal
f64e f64e_div(f64e a, f64e b) { return f64e_mul(a, f64e_inv(b)); }

// Negation
f64e f64e_neg(f64e a) { return {-a.hi, -a.lo}; }

// Subtraction
f64e f64e_sub(f64e a, f64e b) { return f64e_add(a, f64e_neg(b)); }
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
    float rt_aa  = rt / aijk;
    float rt_aij = rt_aa * ak;
    float b10 = 0.5f / aij * (1.0f - rt_aij);
    float b01 = 0.5f / ak  * (1.0f - rt_aa * aij);
    float b00 = 0.5f * rt_aa;

    float g_ij[3][9][4][5];
    for (int dir = 0; dir < 3; dir++) {
        float c0 = PA[dir] - rt_aij * Rpq[dir];
        float cp = (rt_aa * aij) * Rpq[dir];
        float g[9][5];
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++) g[a][c] = 0.0f;
        g[0][0] = 1.0f;
        if (lij1 > 0) g[1][0] = c0;
        for (int a = 1; a < lij1; a++)
            g[a+1][0] = c0 * g[a][0] + float(a) * b10 * g[a-1][0];
        for (int c = 0; c < lk; c++) {
            for (int a = 0; a <= lij1; a++) {
                float val = cp * g[a][c];
                if (c > 0) val += float(c) * b01 * g[a][c-1];
                if (a > 0) val += float(a) * b00 * g[a-1][c];
                g[a][c+1] = val;
            }
        }
        for (int a = 0; a <= lij1; a++)
            for (int c = 0; c <= lk; c++)
                g_ij[dir][a][0][c] = g[a][c];
        float ab = AB[dir];
        for (int j = 0; j < lj; j++)
            for (int c = 0; c <= lk; c++)
                for (int i = 0; i <= lij1 - j - 1; i++)
                    g_ij[dir][i][j+1][c] = g_ij[dir][i+1][j][c] + ab * g_ij[dir][i][j][c];
    }

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
    name='int3c2e_ip1_f32_rysfix',  # f32 TRR + fixed Rys threshold (99)
    input_names=['task_data', 'offsets'],
    output_names=['int_out'],
    header=_HEADER_3C,
    source=_SOURCE_3C,
    atomic_outputs=False,
)


# ---------------------------------------------------------------------------
# CPU Phase A: build tasks (Boys + Rys on CPU in f64)
# ---------------------------------------------------------------------------

def _load_rys_tables():
    """Load Rys Chebyshev interpolation tables from CUDA data file."""
    import re, os
    dat_path = os.path.join(os.path.dirname(__file__), '..', 'gvhf-rys', 'rys_roots_dat.cu')
    with open(dat_path) as f:
        text = f.read()
    def _extract(name):
        m = re.search(rf'{name}\[\]\s*=\s*\{{([^}}]+)\}}', text, re.DOTALL)
        nums = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+', m.group(1))
        return np.array([float(x) for x in nums], dtype=np.float64)
    return {k: _extract(v) for k, v in [
        ('rw', 'ROOT_RW_DATA'), ('sr0', 'ROOT_SMALLX_R0'), ('sr1', 'ROOT_SMALLX_R1'),
        ('sw0', 'ROOT_SMALLX_W0'), ('sw1', 'ROOT_SMALLX_W1'),
        ('lr', 'ROOT_LARGEX_R_DATA'), ('lw', 'ROOT_LARGEX_W_DATA')]}

_rys_tbl = None
def _get_rys_tbl():
    global _rys_tbl
    if _rys_tbl is None:
        _rys_tbl = _load_rys_tables()
    return _rys_tbl


@nb.njit
def _clenshaw_rys(T, nroots, rw, sr0, sr1, sw0, sw1, lr, lw, roots, weights, base_idx):
    """Compute Rys roots/weights for one primitive via f64 Clenshaw."""
    if T < 3e-7:
        for r in range(nroots):
            s = nroots * (nroots - 1) // 2 + r
            roots[base_idx + r] = sr0[s] + sr1[s] * T
            weights[base_idx + r] = sw0[s] + sw1[s] * T
    elif T > 99.0:
        # Asymptotic only for T beyond Chebyshev table range [0, 100).
        # CUDA reference (rys_roots.cu:43-48) applies t = sqrt(PIE4/x) to the
        # weight, where PIE4 = pi/4. ROOT_LARGEX_W_DATA is normalized assuming
        # this external factor; missing it gives a 2/sqrt(pi) = 1.1284 error
        # on F_0 for every large-T primitive.
        ix = 1.0 / T
        sqix = np.sqrt(0.7853981633974483 * ix)   # sqrt(pi/4 * 1/T)
        for r in range(nroots):
            s = nroots * (nroots - 1) // 2 + r
            roots[base_idx + r] = lr[s] * ix
            weights[base_idx + r] = lw[s] * sqix
    else:
        it = min(int(T * 0.4), 39)
        u = (T - it * 2.5) * 0.8 - 1.0
        u2 = 2.0 * u
        noff = 560 * nroots * (nroots - 1)
        for r in range(nroots):
            for rwidx in range(2):
                b = noff + (2 * r + rwidx) * 560 + it
                c0 = rw[b + 520]  # 13*40
                c1 = rw[b + 480]  # 12*40
                for nn in range(11, 0, -2):
                    c2 = rw[b + nn * 40] - c1
                    c3 = c0 + c1 * u2
                    c1 = c2 + c3 * u2
                    c0 = rw[b + (nn - 1) * 40] - c3
                v = c0 + c1 * u
                if rwidx == 0:
                    roots[base_idx + r] = v
                else:
                    weights[base_idx + r] = v


@nb.njit
def _build_3c_tasks_nb(
    nbas, shl0_aux, shl1_aux,
    shell_l, shell_nprim, prim_off_arr, exps, coeffs,
    shell_x, shell_y, shell_z,
    nbas_orb, ncart_lut,
    rw, sr0, sr1, sw0, sw1, lr, lw,
    # Output (pre-allocated, oversized):
    tasks, roots_flat, weights_flat, offsets,
    meta_ish, meta_jsh, meta_ksh, meta_ts, meta_tc,
    meta_nfi, meta_nfj, meta_nfk, meta_nout):
    """Numba-JIT task builder: enumerate + T + Clenshaw + pack."""
    PI52 = 34.986836655249725  # 2 * pi^2.5
    task_count = 0
    triple_count = 0
    out_offset = 0

    for ish in range(nbas_orb):
        li = shell_l[ish]
        if li > 3:
            continue
        nfi = ncart_lut[li]
        for jsh in range(nbas_orb):
            lj = shell_l[jsh]
            if lj > 3:
                continue
            nfj = ncart_lut[lj]
            dx = shell_x[ish] - shell_x[jsh]
            dy = shell_y[ish] - shell_y[jsh]
            dz = shell_z[ish] - shell_z[jsh]
            rr_ab = dx * dx + dy * dy + dz * dz

            for ksh in range(shl0_aux, shl1_aux):
                ksh_g = nbas_orb + ksh  # global shell index
                lk = shell_l[ksh_g]
                if lk > 4:
                    continue
                nfk = ncart_lut[lk]
                nroots = (li + 1 + lj + lk) // 2 + 1
                n_out = 3 * nfi * nfj * nfk
                npi = shell_nprim[ish]
                npj = shell_nprim[jsh]
                npk = shell_nprim[ksh_g]
                t_start = task_count

                for p_i in range(npi):
                    ai = exps[prim_off_arr[ish] + p_i]
                    ci = coeffs[prim_off_arr[ish] + p_i]
                    for p_j in range(npj):
                        aj = exps[prim_off_arr[jsh] + p_j]
                        cj = coeffs[prim_off_arr[jsh] + p_j]
                        aij = ai + aj
                        eij = np.exp(-ai * aj / aij * rr_ab)
                        px = (ai * shell_x[ish] + aj * shell_x[jsh]) / aij
                        py = (ai * shell_y[ish] + aj * shell_y[jsh]) / aij
                        pz = (ai * shell_z[ish] + aj * shell_z[jsh]) / aij
                        coeff_ij = ci * cj * eij

                        for p_k in range(npk):
                            ak = exps[prim_off_arr[ksh_g] + p_k]
                            ck = coeffs[prim_off_arr[ksh_g] + p_k]
                            aijk = aij + ak
                            PCx = px - shell_x[ksh_g]
                            PCy = py - shell_y[ksh_g]
                            PCz = pz - shell_z[ksh_g]
                            T = aij * ak / aijk * (PCx*PCx + PCy*PCy + PCz*PCz)
                            prefac = coeff_ij * ck * PI52 / (aij * ak * np.sqrt(aijk))

                            # Clenshaw Rys in f64
                            _clenshaw_rys(T, nroots, rw, sr0, sr1, sw0, sw1,
                                          lr, lw, roots_flat, weights_flat,
                                          task_count * 6)

                            # Pack task (f32, compact TASK_STRIDE=36)
                            tc = task_count
                            base = tc * 36
                            tasks[base + 0] = aij
                            tasks[base + 1] = ak
                            tasks[base + 2] = px - shell_x[ish]  # PA
                            tasks[base + 3] = py - shell_y[ish]
                            tasks[base + 4] = pz - shell_z[ish]
                            tasks[base + 5] = PCx
                            tasks[base + 6] = PCy
                            tasks[base + 7] = PCz
                            tasks[base + 8] = dx  # AB
                            tasks[base + 9] = dy
                            tasks[base + 10] = dz
                            tasks[base + 11] = prefac
                            tasks[base + 12] = ai
                            tasks[base + 13] = li
                            tasks[base + 14] = lj
                            tasks[base + 15] = lk
                            tasks[base + 17] = nroots
                            for r in range(nroots):
                                tasks[base + 18 + r] = float(roots_flat[tc * 6 + r])
                                tasks[base + 24 + r] = float(weights_flat[tc * 6 + r])
                            tasks[base + 30] = PCx
                            tasks[base + 31] = PCy
                            tasks[base + 32] = PCz

                            offsets[tc] = out_offset
                            out_offset += n_out
                            task_count += 1

                tc_now = task_count - t_start
                if tc_now > 0:
                    meta_ish[triple_count] = ish
                    meta_jsh[triple_count] = jsh
                    meta_ksh[triple_count] = ksh
                    meta_ts[triple_count] = t_start
                    meta_tc[triple_count] = tc_now
                    meta_nfi[triple_count] = nfi
                    meta_nfj[triple_count] = nfj
                    meta_nfk[triple_count] = nfk
                    meta_nout[triple_count] = n_out
                    triple_count += 1

    return task_count, triple_count, out_offset


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
    else:
        shl0_aux = shls_slice[4]
        shl1_aux = shls_slice[5]

    # Check if all orbital shells are supported
    max_l_orb = max(mol.bas_angular(i) for i in range(nbas))
    max_l_aux = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))
    if max_l_orb > 3 or max_l_aux > 4:
        return _cpu_fallback(mol, auxmol, shls_slice)

    # Phase A: build tasks on CPU using Numba (f64 Clenshaw Rys)
    from gpu4pyscf.lib.metal_kernels.int3c2e_ip1_v2 import _pack_shell_data
    (sx, sy, sz, sl_arr, snp_arr, spo_arr, exps_arr, coeffs_arr) = _pack_shell_data(mol, auxmol)
    tbl = _get_rys_tbl()

    # Pre-allocate oversized buffers (Numba writes into them). Use np.empty —
    # the Numba task builder writes sequentially and never reads uninitialized
    # slots, and zeroing 1.4 GB costs ~55ms per call.
    max_tasks = 10_000_000  # generous upper bound
    max_triples = 2_000_000
    tasks_flat = np.empty(max_tasks * TASK_STRIDE, dtype=np.float32)
    roots_flat = np.empty(max_tasks * 6, dtype=np.float64)
    weights_flat = np.empty(max_tasks * 6, dtype=np.float64)
    offsets_buf = np.empty(max_tasks, dtype=np.int32)
    m_ish = np.empty(max_triples, dtype=np.int32)
    m_jsh = np.empty(max_triples, dtype=np.int32)
    m_ksh = np.empty(max_triples, dtype=np.int32)
    m_ts = np.empty(max_triples, dtype=np.int32)
    m_tc = np.empty(max_triples, dtype=np.int32)
    m_nfi = np.empty(max_triples, dtype=np.int32)
    m_nfj = np.empty(max_triples, dtype=np.int32)
    m_nfk = np.empty(max_triples, dtype=np.int32)
    m_nout = np.empty(max_triples, dtype=np.int32)

    n_tasks, n_meta, total_out = _build_3c_tasks_nb(
        nbas, shl0_aux, shl1_aux,
        sl_arr, snp_arr, spo_arr,
        exps_arr.astype(np.float64), coeffs_arr.astype(np.float64),
        sx.astype(np.float64), sy.astype(np.float64), sz.astype(np.float64),
        nbas, _NCART_LUT,
        tbl['rw'], tbl['sr0'], tbl['sr1'], tbl['sw0'], tbl['sw1'],
        tbl['lr'], tbl['lw'],
        tasks_flat, roots_flat, weights_flat, offsets_buf,
        m_ish, m_jsh, m_ksh, m_ts, m_tc, m_nfi, m_nfj, m_nfk, m_nout)

    if n_tasks == 0 or total_out == 0 or n_tasks >= max_tasks:
        return _cpu_fallback(mol, auxmol, shls_slice)

    tasks = tasks_flat[:n_tasks * TASK_STRIDE].reshape(n_tasks, TASK_STRIDE)
    offsets = offsets_buf[:n_tasks]

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
    # Keep int_buf in f32 — the GPU kernel runs in f32, so f64 cast here is
    # cosmetic and costs ~24ms per call (~108 MB conversion on benzene).
    # _accumulate_cart_nb sums into an f64 output, which promotes automatically.
    int_buf = np.asarray(result[0])

    # Phase C: accumulate into output tensor (Cartesian first, then cart2sph)
    nao_cart = mol.nao_cart()
    aux_loc_c = auxmol.ao_loc_nr(cart=True)
    naux_slice_c = aux_loc_c[shl1_aux] - aux_loc_c[shl0_aux]
    p0_aux = aux_loc_c[shl0_aux]
    ao_loc_c = mol.ao_loc_nr(cart=True)

    out_cart = np.zeros((3, nao_cart, nao_cart, naux_slice_c), dtype=np.float64)
    if n_meta > 0:
        # Slice directly from the Numba-filled arrays — avoids a Python list
        # round-trip that previously cost ~260ms per call on benzene.
        ish_s = m_ish[:n_meta]
        jsh_s = m_jsh[:n_meta]
        ksh_s = m_ksh[:n_meta]          # aux-local index
        m_i0 = ao_loc_c[ish_s].astype(np.int32, copy=False)
        m_j0 = ao_loc_c[jsh_s].astype(np.int32, copy=False)
        m_k0 = (aux_loc_c[ksh_s] - p0_aux).astype(np.int32, copy=False)
        fac_tbl = np.array([_FAC_L[0], _FAC_L[1], _FAC_L[2], _FAC_L[3], _FAC_L[4]])
        # sl_arr is packed: [0, nbas) are orbital shells, [nbas, nbas+nbas_aux) are aux
        m_li = sl_arr[ish_s].astype(np.int32, copy=False)
        m_lj = sl_arr[jsh_s].astype(np.int32, copy=False)
        m_lk = sl_arr[ksh_s + nbas].astype(np.int32, copy=False)
        _accumulate_cart_nb(int_buf, offsets, m_i0, m_j0, m_k0,
                            m_ts[:n_meta], m_tc[:n_meta],
                            m_nfi[:n_meta], m_nfj[:n_meta], m_nfk[:n_meta],
                            m_nout[:n_meta], m_li, m_lj, m_lk,
                            fac_tbl, out_cart)

    if mol.cart:
        # PySCF getints returns (ncomp, ni, nj, nk) with Fortran-ordered
        # inner axes: strides=(big, 8, ni*8, ni*nj*8). Replicate this
        # layout so that fdrv's ctypes access sees the correct strides
        # after .transpose(0,3,2,1) in get_jk.
        result = np.zeros((nao_cart, nao_cart, naux_slice_c, 3),
                          dtype=np.float64, order='F').transpose(3, 0, 1, 2)
        result[:] = out_cart
        return result

    # Cart-to-spherical transform for orbital indices (i, j) and aux (k).
    # Most shells have l <= 1 where cart == sph (copy), and only high-l
    # shells need an actual matrix multiply. Cache per-mol shell plans so
    # the second/third calls skip recomputing ranges and cart2sph matrices,
    # and batch over the 3 derivative components (one einsum per high-l
    # shell instead of three) to amortise Python/einsum overhead.
    nao = mol.nao
    ao_loc_s = mol.ao_loc_nr()
    aux_loc_s = auxmol.ao_loc_nr()
    naux_slice = aux_loc_s[shl1_aux] - aux_loc_s[shl0_aux]

    orb_plan = _cart2sph_plan(mol, ao_loc_c, ao_loc_s)
    aux_plan = _cart2sph_plan(auxmol, aux_loc_c, aux_loc_s, shl0_aux, shl1_aux)

    # All three axis transforms run on GPU via MLX (~6x faster than CPU
    # numpy tensordot, including CPU<->GPU transfer). Plan index arrays
    # and transform matrices stay resident on GPU across calls.
    out = _cart2sph_gpu(out_cart, orb_plan, aux_plan,
                         nao, nao_cart, naux_slice_c, naux_slice)

    # Match getints' stride layout: (ni, nj, nk, ncomp) F-order then transpose
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
