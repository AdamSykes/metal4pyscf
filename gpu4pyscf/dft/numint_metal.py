"""
Metal GPU-accelerated XC numerical integration for DFT.

Fused architecture: eval_ao + rho + Vxc contraction per grid batch on GPU.
Never materializes the full (ngrids, nao) AO array.
"""

import numpy as np
import mlx.core as mx
from pyscf.dft import numint as numint_cpu
from gpu4pyscf.lib.metal_kernels.eval_ao import eval_ao_metal


def nr_rks_metal(ni, mol, grids, xc_code, dm, hermi=1, max_memory=2000):
    """Metal GPU-accelerated nr_rks with fused batched evaluation.

    For each grid batch:
      1. eval_ao on Metal GPU (batch only, ~60K points)
      2. rho + XC on CPU (libxc, fast)
      3. Vxc contraction on Metal GPU (gemm)
      4. Accumulate Vxc in f64

    No full AO array is ever materialized.
    """
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))

    ao_deriv = {'LDA': 0, 'GGA': 1, 'MGGA': 1, 'HF': 0}.get(xctype, 1)

    # Precompute shell data for Metal eval_ao (reused across batches)
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping = \
        _prepare_shell_data(mol)

    for ao_np, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv,
                                                      max_memory=max_memory):
        ngrids_batch = coords.shape[0]

        # --- eval_ao on Metal GPU for this batch ---
        ao_gpu = _eval_ao_batch_gpu(
            mol, coords, ao_deriv, shell_data, exps_gpu, coeffs_gpu,
            ncart_total, shell_mapping, ngrids_batch)
        # ao_gpu: mx.array, shape (nao, ngrids) for LDA or (4, nao, ngrids) for GGA

        for i in range(nset):
            # --- rho on CPU (uses PySCF's make_rho with screening) ---
            rho = make_rho(i, ao_np, mask, xctype)

            # --- XC on CPU (libxc) ---
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1,
                                       xctype=xctype, spin=0)[:2]

            if xctype == 'LDA':
                den = rho * weight
            else:
                den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += np.dot(den, exc)

            # --- Vxc contraction on Metal GPU ---
            wv = weight * vxc

            if xctype == 'LDA':
                # ao_gpu: (nao, ngrids)
                wv_gpu = mx.array(wv[0].astype(np.float32))
                # vmat += ao @ diag(wv) @ ao.T = (ao * wv) @ ao.T
                vmat_blk = (ao_gpu * wv_gpu[None, :]) @ ao_gpu.T
                mx.eval(vmat_blk)
                vmat[i] += np.array(vmat_blk).astype(np.float64)

            elif xctype == 'GGA':
                wv[0] *= 0.5
                # ao_gpu: (4, nao, ngrids) — [val, dx, dy, dz]
                # aow = sum_d ao[d] * wv[d]
                wv_gpu = [mx.array(wv[d].astype(np.float32)) for d in range(4)]
                aow = ao_gpu[0] * wv_gpu[0][None, :]
                for d in range(1, 4):
                    aow = aow + ao_gpu[d] * wv_gpu[d][None, :]
                # vmat += aow @ ao[0].T
                vmat_blk = aow @ ao_gpu[0].T
                mx.eval(vmat_blk)
                vmat[i] += np.array(vmat_blk).astype(np.float64)

            elif xctype == 'MGGA':
                wv[0] *= 0.5
                wv[4] *= 0.5
                wv_gpu = [mx.array(wv[d].astype(np.float32)) for d in range(5)]
                aow = ao_gpu[0] * wv_gpu[0][None, :]
                for d in range(1, 4):
                    aow = aow + ao_gpu[d] * wv_gpu[d][None, :]
                vmat_blk = aow @ ao_gpu[0].T
                for d in range(1, 4):
                    vmat_blk = vmat_blk + (ao_gpu[d] * wv_gpu[4][None, :]) @ ao_gpu[d].T
                mx.eval(vmat_blk)
                vmat[i] += np.array(vmat_blk).astype(np.float64)

    if xctype in ('GGA', 'MGGA'):
        vmat = vmat + vmat.transpose(0, 2, 1)

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    return nelec, excsum, vmat


def _ncart(l):
    return (l + 1) * (l + 2) // 2


def _cart2sph_matrix(l):
    from pyscf.gto.mole import cart2sph
    return np.asarray(cart2sph(l, normalized='sp'))


def _prepare_shell_data(mol):
    """Precompute shell metadata arrays for Metal eval_ao (reusable)."""
    from gpu4pyscf.lib.metal_kernels.eval_ao import _eval_ao_kernel, _eval_ao_deriv1_kernel

    nshells = mol.nbas
    ncart_total = sum(_ncart(mol.bas_angular(i)) for i in range(nshells))

    shell_data = np.zeros((nshells, 8), dtype=np.float32)
    all_exps = []
    all_coeffs = []
    exp_offset = 0
    cart_ao_off = 0
    shell_mapping = []  # (l, cart_start, ncart_l, sph_start, nsph_l)
    sph_off = 0

    for ish in range(nshells):
        l = mol.bas_angular(ish)
        atom_id = mol.bas_atom(ish)
        ac = mol.atom_coord(atom_id)
        nprim = mol.bas_nprim(ish)
        ncart_l = _ncart(l)
        nsph_l = ncart_l if l <= 1 else 2 * l + 1

        fac = {0: 0.282094791773878143, 1: 0.488602511902919921}.get(l, 1.0)
        shell_data[ish] = [ac[0], ac[1], ac[2], fac, nprim,
                           cart_ao_off, exp_offset, l]
        shell_mapping.append((l, cart_ao_off, ncart_l, sph_off, nsph_l))

        all_exps.append(mol.bas_exp(ish).astype(np.float32))
        all_coeffs.append(mol._libcint_ctr_coeff(ish).flatten().astype(np.float32))
        exp_offset += nprim
        cart_ao_off += ncart_l
        sph_off += nsph_l

    exps_gpu = mx.array(np.concatenate(all_exps))
    coeffs_gpu = mx.array(np.concatenate(all_coeffs))

    return shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping


def _eval_ao_batch_gpu(mol, coords, deriv, shell_data, exps_gpu, coeffs_gpu,
                       ncart_total, shell_mapping, ngrids):
    """Evaluate AOs for a single grid batch on Metal GPU.

    Returns: mx.array of shape (nao, ngrids) for deriv=0
             or (4, nao, ngrids) for deriv=1
    Data stays on GPU (no readback to CPU).
    """
    from gpu4pyscf.lib.metal_kernels.eval_ao import _eval_ao_kernel, _eval_ao_deriv1_kernel

    coords = np.asarray(coords)
    gridx = mx.array(coords[:, 0].astype(np.float32))
    gridy = mx.array(coords[:, 1].astype(np.float32))
    gridz = mx.array(coords[:, 2].astype(np.float32))
    shell_data_gpu = mx.array(shell_data.ravel())
    nshells = mol.nbas
    nao = mol.nao
    cart = mol.cart

    THREADS_X = 256
    grid_x = ((ngrids + THREADS_X - 1) // THREADS_X) * THREADS_X
    ncomp = 4 if deriv == 1 else 1

    if deriv == 0:
        kernel = _eval_ao_kernel
        template = [('ngrids', ngrids), ('nshells', nshells)]
    else:
        kernel = _eval_ao_deriv1_kernel
        template = [('ngrids', ngrids), ('nshells', nshells),
                    ('ncart_total', ncart_total)]

    result = kernel(
        inputs=[gridx, gridy, gridz, exps_gpu, coeffs_gpu, shell_data_gpu],
        grid=(grid_x, nshells, 1),
        threadgroup=(THREADS_X, 1, 1),
        output_shapes=[(ncomp * ncart_total * ngrids,)],
        output_dtypes=[mx.float32],
        template=template,
    )

    if deriv == 0:
        ao_cart = result[0].reshape(ncart_total, ngrids)
    else:
        ao_cart = result[0].reshape(4, ncart_total, ngrids)

    # Cart-to-spherical on GPU (stays as mx.array)
    if not cart and ncart_total != nao:
        def _apply_c2s(ao_2d):
            parts = []
            for l, c0, ncart_l, s0, nsph_l in shell_mapping:
                block = ao_2d[c0:c0 + ncart_l]
                if l <= 1:
                    parts.append(block)
                else:
                    c2s_gpu = mx.array(_cart2sph_matrix(l).T.astype(np.float32))
                    parts.append(c2s_gpu @ block)
            return mx.concatenate(parts, axis=0)

        if deriv == 0:
            return _apply_c2s(ao_cart)  # (nao, ngrids)
        else:
            return mx.stack([_apply_c2s(ao_cart[c]) for c in range(4)])

    return ao_cart
