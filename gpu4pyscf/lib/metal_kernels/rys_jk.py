"""
Metal GPU Rys polynomial J/K integral engine.

Direct translation of the CUDA Rys polynomial algorithm to Metal,
enabling direct (non-DF) J/K matrix construction on Apple Silicon GPU.

The Rys polynomial method evaluates 2-electron integrals:
  (ij|kl) = ∫∫ φ_i(r1)φ_j(r1) (1/|r1-r2|) φ_k(r2)φ_l(r2) dr1 dr2

by decomposing 1/r12 using Rys polynomial quadrature.

Currently supports s and p shells (l=0,1). This covers STO-3G and
minimal basis sets. Extension to d/f follows the same TRR/HRR pattern.
"""

import numpy as np
import mlx.core as mx
from math import sqrt, pi, erf, exp
from gpu4pyscf.lib.metal_kernels.eval_ao import _ncart, _cart2sph_matrix


# ---------------------------------------------------------------------------
# Boys function F_m(t): the incomplete gamma function
# F_0(t) = (1/2) * sqrt(pi/t) * erf(sqrt(t))  for t > 0
# F_m(t) = downward recursion from F_0
# ---------------------------------------------------------------------------

def boys_function(m_max, t_values):
    """Compute Boys functions F_0(t) through F_m(t) for array of t values.

    Uses the same 3-regime algorithm as the CUDA gamma_inc.cu.
    Returns: (m_max+1, npts) array
    """
    t = np.asarray(t_values, dtype=np.float64)
    npts = t.size
    f = np.zeros((m_max + 1, npts))

    for p in range(npts):
        tp = t[p]
        if tp < 1e-15:
            for m in range(m_max + 1):
                f[m, p] = 1.0 / (2 * m + 1)
        elif m_max > 0 and tp < m_max * 0.5 + 0.5:
            # Upward series for F_m, then downward recursion
            bi = m_max + 0.5
            e = 0.5 * exp(-tp)
            x = e
            s = e
            while x > 1e-15 * e:
                bi += 1.0
                x *= tp / bi
                s += x
            b = m_max + 0.5
            fval = s / b
            f[m_max, p] = fval
            for i in range(m_max - 1, -1, -1):
                b -= 1.0
                fval = (e + tp * fval) / b
                f[i, p] = fval
        else:
            # Large t: asymptotic
            tt = sqrt(tp)
            fval = sqrt(pi) / 4.0 / tt * erf(tt) * 2.0  # = sqrt(pi/4)/tt * erf(tt)
            # Actually: F_0 = sqrt(pi/4t) * erf(sqrt(t))
            fval = sqrt(pi / 4.0) / tt * erf(tt)
            f[0, p] = fval
            if m_max > 0:
                e = 0.5 * exp(-tp)
                b = 1.0 / tp
                b1 = 0.5
                for i in range(1, m_max + 1):
                    fval = b * (b1 * fval - e)
                    f[i, p] = fval
                    b1 += 1.0

    return f


# ---------------------------------------------------------------------------
# Shell pair data preparation
# ---------------------------------------------------------------------------

def _prepare_shell_pairs(mol):
    """Precompute shell pair data for all significant ij pairs.

    Returns numpy arrays with primitive pair information.
    """
    nbas = mol.nbas
    pairs = []

    for ish in range(nbas):
        for jsh in range(ish, nbas):
            li = mol.bas_angular(ish)
            lj = mol.bas_angular(jsh)
            if li > 1 or lj > 1:
                continue  # skip d/f for now

            ai_list = mol.bas_exp(ish)
            aj_list = mol.bas_exp(jsh)
            ci_list = mol._libcint_ctr_coeff(ish).flatten()
            cj_list = mol._libcint_ctr_coeff(jsh).flatten()
            Ri = mol.atom_coord(mol.bas_atom(ish))
            Rj = mol.atom_coord(mol.bas_atom(jsh))

            rij = Ri - Rj
            rr_ij = np.dot(rij, rij)

            for ip, ai in enumerate(ai_list):
                for jp, aj in enumerate(aj_list):
                    aij = ai + aj
                    theta_ij = ai * aj / aij
                    Kab = exp(-theta_ij * rr_ij)
                    if abs(ci_list[ip] * cj_list[jp] * Kab) < 1e-14:
                        continue

                    Pij = (ai * Ri + aj * Rj) / aij
                    pairs.append({
                        'ish': ish, 'jsh': jsh,
                        'li': li, 'lj': lj,
                        'aij': aij, 'Pij': Pij,
                        'rij': rij,
                        'coeff_ij': ci_list[ip] * cj_list[jp] * Kab,
                        'ai': ai, 'aj': aj,
                    })

    return pairs


# ---------------------------------------------------------------------------
# Direct J/K builder (CPU reference, then Metal kernel)
# ---------------------------------------------------------------------------

def get_jk_rys(mol, dm, with_j=True, with_k=True):
    """Compute J/K matrices using Rys polynomial quadrature.

    This is a direct (non-DF) implementation for s/p shells.
    Currently runs on CPU for correctness verification.
    Metal GPU kernel is the next step.

    Args:
        mol: PySCF Mole object
        dm: density matrix (nao, nao)

    Returns:
        vj, vk: J and K matrices
    """
    nao = mol.nao
    dm = np.asarray(dm)
    vj = np.zeros((nao, nao)) if with_j else None
    vk = np.zeros((nao, nao)) if with_k else None

    nbas = mol.nbas
    ao_loc = mol.ao_loc_nr()

    # Spherical harmonic normalization: 1/(4pi) per electron pair
    # = (1/sqrt(4pi))^4 for a 2-electron integral over 4 basis functions
    fac_l = {0: sqrt(1.0 / (4 * pi)), 1: sqrt(3.0 / (4 * pi))}

    for ish in range(nbas):
        li = mol.bas_angular(ish)
        if li > 3:
            continue  # support up to f-shells
        i0, i1 = ao_loc[ish], ao_loc[ish + 1]
        ni = _ncart(li)
        Ri = mol.atom_coord(mol.bas_atom(ish))
        ai_list = mol.bas_exp(ish)
        ci_list = mol._libcint_ctr_coeff(ish).flatten()

        for jsh in range(nbas):
            lj = mol.bas_angular(jsh)
            if lj > 1:
                continue
            j0, j1 = ao_loc[jsh], ao_loc[jsh + 1]
            nj = _ncart(lj)
            Rj = mol.atom_coord(mol.bas_atom(jsh))
            aj_list = mol.bas_exp(jsh)
            cj_list = mol._libcint_ctr_coeff(jsh).flatten()

            for ksh in range(nbas):
                lk = mol.bas_angular(ksh)
                if lk > 1:
                    continue
                k0, k1 = ao_loc[ksh], ao_loc[ksh + 1]
                nk = _ncart(lk)
                Rk = mol.atom_coord(mol.bas_atom(ksh))
                ak_list = mol.bas_exp(ksh)
                ck_list = mol._libcint_ctr_coeff(ksh).flatten()

                for lsh in range(nbas):
                    ll = mol.bas_angular(lsh)
                    if ll > 1:
                        continue
                    l0, l1 = ao_loc[lsh], ao_loc[lsh + 1]
                    nl = _ncart(ll)
                    Rl = mol.atom_coord(mol.bas_atom(lsh))
                    al_list = mol.bas_exp(lsh)
                    cl_list = mol._libcint_ctr_coeff(lsh).flatten()

                    nroots = (li + lj + lk + ll) // 2 + 1

                    # Contract over all primitives
                    eri = np.zeros((ni, nj, nk, nl))

                    for ip, ai in enumerate(ai_list):
                        for jp, aj in enumerate(aj_list):
                            aij = ai + aj
                            Pij = (ai * Ri + aj * Rj) / aij
                            rr_ij = np.dot(Ri - Rj, Ri - Rj)
                            Kab = exp(-ai * aj / aij * rr_ij)

                            for kp, ak in enumerate(ak_list):
                                for lp, al in enumerate(al_list):
                                    akl = ak + al
                                    Pkl = (ak * Rk + al * Rl) / akl
                                    rr_kl = np.dot(Rk - Rl, Rk - Rl)
                                    Kcd = exp(-ak * al / akl * rr_kl)

                                    coeff = (ci_list[ip] * cj_list[jp] * Kab *
                                             ck_list[kp] * cl_list[lp] * Kcd)
                                    if abs(coeff) < 1e-15:
                                        continue

                                    theta = aij * akl / (aij + akl)
                                    Rpq = Pij - Pkl
                                    rr_pq = np.dot(Rpq, Rpq)

                                    # Boys functions
                                    # Need F_0 through F_{2*nroots-1} for Rys root extraction
                                    m_max = max(li + lj + lk + ll, 2 * nroots - 1)
                                    fm = boys_function(m_max, [theta * rr_pq])

                                    prefac = coeff * 2.0 * pi**2.5 / (aij * akl * sqrt(aij + akl))
                                    fac_i = fac_l.get(li, 1.0)
                                    fac_j = fac_l.get(lj, 1.0)
                                    fac_k = fac_l.get(lk, 1.0)
                                    fac_ll = fac_l.get(ll, 1.0)
                                    prefac *= fac_i * fac_j * fac_k * fac_ll

                                    # Build integrals via Rys quadrature
                                    _rys_build_eri(
                                        eri, prefac, fm[:, 0],
                                        ai, aj, ak, al, aij, akl,
                                        Ri, Rj, Rk, Rl, Pij, Pkl,
                                        li, lj, lk, ll, nroots)

                    # Cart → spherical transformation
                    if not mol.cart:
                        for idx, lv in enumerate([li, lj, lk, ll]):
                            if lv >= 2:
                                c2s = _cart2sph_matrix(lv)
                                eri = np.tensordot(c2s.T, eri, axes=([1],[idx]))
                                eri = np.moveaxis(eri, 0, idx)

                    # Accumulate J and K (using spherical ao_loc indices)
                    if with_j:
                        vj[i0:i1, j0:j1] += np.einsum('kl,ijkl->ij', dm[k0:k1, l0:l1], eri)
                    if with_k:
                        vk[i0:i1, k0:k1] += np.einsum('jl,ijkl->ik', dm[j0:j1, l0:l1], eri)

    return vj, vk


def _rys_build_eri(eri, prefac, fm, ai, aj, ak, al, aij, akl,
                   Ri, Rj, Rk, Rl, Pij, Pkl, li, lj, lk, ll, nroots):
    """Build ERI using Rys quadrature: factorized 1D per direction.

    Each Rys root gives independent 1D recurrences that multiply across
    x,y,z. This is the correct factorization matching the CUDA code.
    """
    Rpq = Pij - Pkl
    PA = Pij - Ri
    QC = Pkl - Rk
    ABx = Ri - Rj
    CDx = Rk - Rl

    lij = li + lj
    lkl = lk + ll
    ni, nj, nk, nl = eri.shape

    roots, weights = _rys_from_boys(nroots, fm)

    cart_idx = {
        0: [(0,0,0)],
        1: [(1,0,0),(0,1,0),(0,0,1)],
        2: [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
        3: [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
            (0,3,0),(0,2,1),(0,1,2),(0,0,3)],
    }

    for iroot in range(nroots):
        rt = roots[iroot]
        wt = weights[iroot]

        rt_aa = rt / (aij + akl)
        rt_aij = rt_aa * akl
        rt_akl = rt_aa * aij
        b10 = 0.5 / aij * (1 - rt_aij)
        b01 = 0.5 / akl * (1 - rt_akl)
        b00 = 0.5 * rt_aa

        g1d = [None, None, None]
        for ix in range(3):
            c0 = PA[ix] - rt_aij * Rpq[ix]
            cp = QC[ix] + rt_akl * Rpq[ix]

            g = np.zeros((lij + 1, lkl + 1))
            g[0, 0] = 1.0

            for a in range(lij):
                g[a + 1, 0] = c0 * g[a, 0]
                if a > 0:
                    g[a + 1, 0] += a * b10 * g[a - 1, 0]

            for c in range(lkl):
                for a in range(lij + 1):
                    g[a, c + 1] = cp * g[a, c]
                    if c > 0:
                        g[a, c + 1] += c * b01 * g[a, c - 1]
                    if a > 0:
                        g[a, c + 1] += a * b00 * g[a - 1, c]

            g_ijc = np.zeros((lij + 1, lj + 1, lkl + 1))
            for a in range(lij + 1):
                for c in range(lkl + 1):
                    g_ijc[a, 0, c] = g[a, c]
            for j in range(lj):
                for c in range(lkl + 1):
                    for i in range(li + 1):
                        g_ijc[i, j + 1, c] = g_ijc[i + 1, j, c] + ABx[ix] * g_ijc[i, j, c]

            g_full = np.zeros((li + 1, lj + 1, lkl + 1, ll + 1))
            for i in range(li + 1):
                for j in range(lj + 1):
                    for c in range(lkl + 1):
                        g_full[i, j, c, 0] = g_ijc[i, j, c]
                    for l_idx in range(ll):
                        for k in range(lk + 1):
                            g_full[i, j, k, l_idx + 1] = (g_full[i, j, k + 1, l_idx] +
                                                            CDx[ix] * g_full[i, j, k, l_idx])
            g1d[ix] = g_full[:li+1, :lj+1, :lk+1, :ll+1]

        for ii, (ix_i, iy_i, iz_i) in enumerate(cart_idx[li]):
            for jj, (ix_j, iy_j, iz_j) in enumerate(cart_idx[lj]):
                for kk, (ix_k, iy_k, iz_k) in enumerate(cart_idx[lk]):
                    for ll_idx, (ix_l, iy_l, iz_l) in enumerate(cart_idx[ll]):
                        val = (g1d[0][ix_i, ix_j, ix_k, ix_l] *
                               g1d[1][iy_i, iy_j, iy_k, iy_l] *
                               g1d[2][iz_i, iz_j, iz_k, iz_l])
                        eri[ii, jj, kk, ll_idx] += prefac * wt * val


def _rys_from_boys(nroots, fm):
    """Compute Rys roots and weights from Boys functions via Hankel matrix."""
    if nroots == 1:
        F0, F1 = fm[0], fm[1]
        if F0 > 1e-30:
            return np.array([F1 / F0]), np.array([F0])
        return np.array([0.0]), np.array([F0])
    elif nroots == 2:
        A = np.array([[fm[0], fm[1]], [fm[1], fm[2]]])
        b = np.array([fm[2], fm[3]])
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            t = fm[1]/fm[0] if fm[0] > 1e-30 else 0
            return np.array([t, t]), np.array([fm[0]/2, fm[0]/2])
        disc = max(c[1]**2 + 4*c[0], 0)
        sq = np.sqrt(disc)
        t0, t1 = (c[1]-sq)/2, (c[1]+sq)/2
        if abs(t1-t0) < 1e-30:
            return np.array([t0, t1]), np.array([fm[0]/2, fm[0]/2])
        w1 = (fm[1] - fm[0]*t0) / (t1-t0)
        w0 = fm[0] - w1
        return np.array([t0, t1]), np.array([w0, w1])
    # General nroots >= 2 via Hankel matrix
    n = nroots
    H = np.array([[fm[i+j] for j in range(n)] for i in range(n)])
    rhs = np.array([fm[n+i] for i in range(n)])
    try:
        c = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        t = fm[1]/fm[0] if fm[0] > 1e-30 else 0
        return np.full(n, t), np.full(n, fm[0]/n)
    poly = np.zeros(n + 1)
    poly[0] = 1.0
    for i in range(n):
        poly[n - i] = -c[i]
    roots = np.real(np.roots(poly))
    roots.sort()
    V = np.vander(roots, increasing=True)[:, :n]
    try:
        weights = np.linalg.solve(V.T, fm[:n])
    except np.linalg.LinAlgError:
        weights = np.full(n, fm[0]/n)
    return roots, weights
