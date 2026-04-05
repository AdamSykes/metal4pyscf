"""
Fully Metal GPU-native XC numerical integration for DFT.

Uses fused eval_ao + rho + Vxc contraction per grid batch.
AO array stays on GPU — never copied to CPU. Only rho values
go to CPU for libxc, then wv comes back for Vxc contraction.
"""

import numpy as np
from gpu4pyscf.lib.metal_kernels.eval_ao import _prepare_shell_data
from gpu4pyscf.lib.metal_kernels.fused_xc import fused_rho_vxc


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
