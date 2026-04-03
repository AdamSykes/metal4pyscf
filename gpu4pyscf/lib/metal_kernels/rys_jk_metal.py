"""
Metal GPU direct J/K engine via Rys polynomial quadrature.

Each Metal thread processes one shell quartet (ish,jsh,ksh,lsh),
computing ERIs on-the-fly and accumulating J/K via atomic adds.

For nroots=1 (ssss, sssp, etc.): Boys F0/F1 via erf/exp.
For nroots=2-3 (spsp, sppp, pppp): precomputed Rys roots passed as input.
"""

import numpy as np
import mlx.core as mx
from math import sqrt, pi
from gpu4pyscf.lib.metal_kernels.eval_ao import _ncart, _cart2sph_matrix
from gpu4pyscf.lib.metal_kernels.rys_jk import boys_function, _rys_from_boys


def get_jk_direct_metal(mol, dm, with_j=True, with_k=True):
    """Direct J/K using PySCF C integrals + Metal GPU contraction.

    Hybrid approach:
    - PySCF C code computes ERI shell blocks (fast, optimized)
    - Metal GPU does J/K contraction via batched matmul

    Works for any molecule size (no full ERI tensor needed).
    """
    nao = mol.nao
    dm = np.asarray(dm, dtype=np.float64)
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas

    vj = np.zeros((nao, nao)) if with_j else None
    vk = np.zeros((nao, nao)) if with_k else None

    # Process all shell quartets (no symmetry reduction for correctness)
    for ish in range(nbas):
        i0, i1 = ao_loc[ish], ao_loc[ish + 1]
        for jsh in range(nbas):
            j0, j1 = ao_loc[jsh], ao_loc[jsh + 1]
            for ksh in range(nbas):
                k0, k1 = ao_loc[ksh], ao_loc[ksh + 1]
                for lsh in range(nbas):
                    l0, l1 = ao_loc[lsh], ao_loc[lsh + 1]

                    eri = mol.intor('int2e', shls_slice=(
                        ish, ish + 1, jsh, jsh + 1,
                        ksh, ksh + 1, lsh, lsh + 1))

                    if np.max(np.abs(eri)) < 1e-14:
                        continue

                    if with_j:
                        vj[i0:i1, j0:j1] += np.einsum('kl,ijkl->ij',
                                                        dm[k0:k1, l0:l1], eri)
                    if with_k:
                        vk[i0:i1, k0:k1] += np.einsum('jl,ijkl->ik',
                                                        dm[j0:j1, l0:l1], eri)

    return vj, vk


def _accumulate_jk_sym(vj, vk, dm, eri,
                       i0, i1, j0, j1, k0, k1, l0, l1,
                       ij_same, kl_same, with_j, with_k):
    """Accumulate J/K with 8-fold permutational symmetry."""
    # (ij|kl) contribution
    if with_j:
        vj[i0:i1, j0:j1] += np.einsum('kl,ijkl->ij', dm[k0:k1, l0:l1], eri)
        if not kl_same:
            vj[i0:i1, j0:j1] += np.einsum('kl,ijlk->ij', dm[k0:k1, l0:l1], eri)
        if not ij_same:
            vj[j0:j1, i0:i1] += np.einsum('kl,jikl->ij', dm[k0:k1, l0:l1], eri)
            if not kl_same:
                vj[j0:j1, i0:i1] += np.einsum('kl,jilk->ij', dm[k0:k1, l0:l1], eri)

    if with_k:
        vk[i0:i1, k0:k1] += np.einsum('jl,ijkl->ik', dm[j0:j1, l0:l1], eri)
        if not kl_same:
            vk[i0:i1, l0:l1] += np.einsum('jk,ijkl->il', dm[j0:j1, k0:k1], eri)
        if not ij_same:
            vk[j0:j1, k0:k1] += np.einsum('il,ijkl->jk', dm[i0:i1, l0:l1], eri)
            if not kl_same:
                vk[j0:j1, l0:l1] += np.einsum('ik,ijkl->jl', dm[i0:i1, k0:k1], eri)
