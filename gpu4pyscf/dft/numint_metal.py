"""
Fully Metal GPU-native XC numerical integration for DFT.

Uses fused eval_ao + rho + Vxc contraction per grid batch.
AO array stays on GPU — never copied to CPU. Only rho values
go to CPU for libxc, then wv comes back for Vxc contraction.
"""

import numpy as np
from gpu4pyscf.lib.metal_kernels.eval_ao import _prepare_shell_data
from gpu4pyscf.lib.metal_kernels.fused_xc import (
    fused_rho_vxc, _batched_rho_vxc_uks, _batched_vxc_grad,
    _batched_vxc_grad_uks,
)


def nr_rks_metal(ni, mol, grids, xc_code, dm, hermi=1, max_memory=2000):
    """Fully Metal GPU-accelerated nr_rks."""
    xctype = ni._xc_type(xc_code)
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping, shell_data_gpu = \
        _prepare_shell_data(mol)

    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)

    return fused_rho_vxc(
        mol, coords, dm, weights, ni, xc_code, xctype,
        shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
        shell_data_gpu)


def nr_uks_metal(ni, mol, grids, xc_code, dms, hermi=1, max_memory=2000):
    """Fully Metal GPU-accelerated nr_uks (unrestricted DFT).

    dms: (2, nao, nao) or tuple (dm_a, dm_b) — alpha/beta density matrices.
    Returns (nelec, excsum, vmat) where nelec is (2,) and vmat is (2, nao, nao).
    """
    xctype = ni._xc_type(xc_code)
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping, shell_data_gpu = \
        _prepare_shell_data(mol)

    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)

    dms = np.asarray(dms)
    if dms.ndim != 3 or dms.shape[0] != 2:
        raise ValueError(f'nr_uks_metal expects (2,nao,nao) dm, got shape {dms.shape}')
    dm_a, dm_b = dms[0], dms[1]

    return _batched_rho_vxc_uks(
        mol, coords, dm_a, dm_b, weights, ni, xc_code, xctype,
        shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
        shell_data_gpu)


def nr_rks_grad_metal(ni, mol, grids, xc_code, dm, max_memory=2000):
    """Metal GPU XC nuclear-gradient contraction (RKS).

    Drop-in replacement for pyscf.grad.rks.get_vxc(ni, mol, grids, xc_code, dm).
    Returns (exc=None, -vmat) with vmat shape (3, nao, nao) matching PySCF.
    Supports LDA and GGA xctypes; MGGA falls through to the CPU path.
    """
    xctype = ni._xc_type(xc_code)
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping, shell_data_gpu = \
        _prepare_shell_data(mol)

    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)

    return _batched_vxc_grad(
        mol, coords, dm, weights, ni, xc_code, xctype,
        shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
        shell_data_gpu)


def nr_uks_grad_metal(ni, mol, grids, xc_code, dms, max_memory=2000):
    """Metal GPU XC nuclear-gradient contraction (UKS).

    Drop-in replacement for pyscf.grad.uks.get_vxc with spin=1.
    dms: (2, nao, nao) — alpha/beta density matrices.
    Returns (exc=None, -vmat) with vmat shape (2, 3, nao, nao).
    """
    xctype = ni._xc_type(xc_code)
    shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping, shell_data_gpu = \
        _prepare_shell_data(mol)

    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    dms = np.asarray(dms)
    dm_a, dm_b = dms[0], dms[1]

    return _batched_vxc_grad_uks(
        mol, coords, dm_a, dm_b, weights, ni, xc_code, xctype,
        shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping,
        shell_data_gpu)
