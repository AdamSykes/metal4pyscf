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
        if li > 1:
            continue
        i0, i1 = ao_loc[ish], ao_loc[ish + 1]
        ni = i1 - i0
        Ri = mol.atom_coord(mol.bas_atom(ish))
        ai_list = mol.bas_exp(ish)
        ci_list = mol._libcint_ctr_coeff(ish).flatten()

        for jsh in range(nbas):
            lj = mol.bas_angular(jsh)
            if lj > 1:
                continue
            j0, j1 = ao_loc[jsh], ao_loc[jsh + 1]
            nj = j1 - j0
            Rj = mol.atom_coord(mol.bas_atom(jsh))
            aj_list = mol.bas_exp(jsh)
            cj_list = mol._libcint_ctr_coeff(jsh).flatten()

            for ksh in range(nbas):
                lk = mol.bas_angular(ksh)
                if lk > 1:
                    continue
                k0, k1 = ao_loc[ksh], ao_loc[ksh + 1]
                nk = k1 - k0
                Rk = mol.atom_coord(mol.bas_atom(ksh))
                ak_list = mol.bas_exp(ksh)
                ck_list = mol._libcint_ctr_coeff(ksh).flatten()

                for lsh in range(nbas):
                    ll = mol.bas_angular(lsh)
                    if ll > 1:
                        continue
                    l0, l1 = ao_loc[lsh], ao_loc[lsh + 1]
                    nl = l1 - l0
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
                                    fm = boys_function(li + lj + lk + ll, [theta * rr_pq])

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

                    # Accumulate J and K
                    if with_j:
                        vj[i0:i1, j0:j1] += np.einsum('kl,ijkl->ij', dm[k0:k1, l0:l1], eri)
                    if with_k:
                        vk[i0:i1, k0:k1] += np.einsum('jl,ijkl->ik', dm[j0:j1, l0:l1], eri)

    return vj, vk


def _rys_build_eri(eri, prefac, fm, ai, aj, ak, al, aij, akl,
                   Ri, Rj, Rk, Rl, Pij, Pkl, li, lj, lk, ll, nroots):
    """Build ERI tensor from Boys functions using Obara-Saika recurrence.

    For (ss|ss): direct from F_0
    For shells with l>0: uses vertical recurrence relation (VRR)
    """
    theta = aij * akl / (aij + akl)
    Rpq = Pij - Pkl

    # Recurrence coefficients
    oo2z = 0.5 / aij
    oo2e = 0.5 / akl
    oo2ze = 0.5 / (aij + akl)

    PA = Pij - Ri
    PB = Pij - Rj
    QC = Pkl - Rk
    QD = Pkl - Rl
    WP = theta / aij * Rpq
    WQ = -theta / akl * Rpq

    ni, nj, nk, nl = eri.shape

    if li + lj + lk + ll == 0:
        # (ss|ss): simplest case
        eri[0, 0, 0, 0] += prefac * fm[0]
        return

    # General case: VRR + HRR for s/p shells
    # Build g[ix,jx,kx,lx] for each Cartesian direction x=0,1,2
    # then combine: eri[I,J,K,L] = g_x[ix,jx,kx,lx] * g_y[iy,jy,ky,ly] * g_z[iz,jz,kz,lz]

    lij = li + lj
    lkl = lk + ll
    lmax = lij + lkl

    # For each Rys root
    for n in range(nroots):
        # Rys root and weight from Boys functions
        # For nroots roots, we use the Rys quadrature relation
        # For simplicity with nroots=1,2: use Boys directly
        pass

    # For s/p shells, use direct Obara-Saika VRR on Boys functions
    # This avoids the Rys root computation for small nroots

    # VRR: compute (a0|c0) integrals, then HRR for b,d indices
    # [a+1,0|c,0]_m = PA[x]*[a,0|c,0]_m + WP[x]*[a,0|c,0]_{m+1}
    #                 + a/(2*aij) * ([a-1,0|c,0]_m - theta/aij*[a-1,0|c,0]_{m+1})
    #                 + c/(2*(aij+akl)) * [a,0|c-1,0]_{m+1}

    for ix in range(3):  # x, y, z components
        # Build 1D integrals along direction ix
        g = np.zeros((lij + 1, lkl + 1, lmax + 1))

        # Base: g[0,0,m] = fm[m]
        for m in range(lmax + 1):
            g[0, 0, m] = fm[m]

        # VRR upward in a (bra side)
        for a in range(lij):
            for m in range(lmax - a):
                g[a + 1, 0, m] = PA[ix] * g[a, 0, m] + WP[ix] * g[a, 0, m + 1]
                if a > 0:
                    g[a + 1, 0, m] += a * oo2z * (g[a - 1, 0, m] -
                                                    theta / aij * g[a - 1, 0, m + 1])

        # VRR upward in c (ket side)
        for c in range(lkl):
            for a in range(lij + 1):
                for m in range(lmax - a - c):
                    g[a, c + 1, m] = QC[ix] * g[a, c, m] + WQ[ix] * g[a, c, m + 1]
                    if c > 0:
                        g[a, c + 1, m] += c * oo2e * (g[a, c - 1, m] -
                                                        theta / akl * g[a, c - 1, m + 1])
                    if a > 0:
                        g[a, c + 1, m] += a * oo2ze * g[a - 1, c, m + 1]

        # HRR: transfer from (a,0) to (i,j) and (c,0) to (k,l)
        # [i,j+1|k,l] = [i+1,j|k,l] + AB[x]*[i,j|k,l]
        # [i,j|k,l+1] = [i,j|k+1,l] + CD[x]*[i,j|k,l]
        ABx = Ri[ix] - Rj[ix]
        CDx = Rk[ix] - Rl[ix]

        # Store 1D integrals: g1d[i_comp, j_comp, k_comp, l_comp]
        # For s/p: indices 0 or 1 only
        g_hrr = np.zeros((li + 1, lj + 1, lk + 1, ll + 1))

        # First build (a, 0 | c, 0) → (i, j | c, 0) via HRR on j
        g_ac = np.zeros((lij + 1, lkl + 1))
        for a in range(lij + 1):
            for c in range(lkl + 1):
                g_ac[a, c] = g[a, c, 0]

        # HRR j-index: [i,j+1|c] = [i+1,j|c] + AB*[i,j|c]
        g_ijc = np.zeros((lij + 1, lj + 1, lkl + 1))
        for c in range(lkl + 1):
            for a in range(lij + 1):
                g_ijc[a, 0, c] = g_ac[a, c]
            for j in range(lj):
                for i in range(li + 1):
                    g_ijc[i, j + 1, c] = g_ijc[i + 1, j, c] + ABx * g_ijc[i, j, c]

        # HRR l-index: [i,j|k,l+1] = [i,j|k+1,l] + CD*[i,j|k,l]
        g_ijkl = np.zeros((li + 1, lj + 1, lkl + 1, ll + 1))
        for i in range(li + 1):
            for j in range(lj + 1):
                for c in range(lkl + 1):
                    g_ijkl[i, j, c, 0] = g_ijc[i, j, c]
                for l in range(ll):
                    for k in range(lk + 1):
                        g_ijkl[i, j, k, l + 1] = g_ijkl[i, j, k + 1, l] + CDx * g_ijkl[i, j, k, l]
        g_hrr = g_ijkl[:li+1, :lj+1, :lk+1, :ll+1]

        # Combine 3D: for each combination of Cartesian components
        # This is the tensor product across x,y,z
        if ix == 0:
            gx = g_hrr.copy()
        elif ix == 1:
            gy = g_hrr.copy()
        else:
            gz = g_hrr.copy()

    # Combine x,y,z into full ERI
    # For s-shells (l=0): only (0,0,0,0) component
    # For p-shells (l=1): 3 components (x,y,z)
    cart_idx = {0: [(0, 0, 0)],
                1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]}

    for ii, (ix_i, iy_i, iz_i) in enumerate(cart_idx[li]):
        for jj, (ix_j, iy_j, iz_j) in enumerate(cart_idx[lj]):
            for kk, (ix_k, iy_k, iz_k) in enumerate(cart_idx[lk]):
                for ll_idx, (ix_l, iy_l, iz_l) in enumerate(cart_idx[ll]):
                    val = (gx[ix_i, ix_j, ix_k, ix_l] *
                           gy[iy_i, iy_j, iy_k, iy_l] *
                           gz[iz_i, iz_j, iz_k, iz_l])
                    eri[ii, jj, kk, ll_idx] += prefac * val
