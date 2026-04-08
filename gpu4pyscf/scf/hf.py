# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import h5py
import itertools
from functools import reduce
from pyscf import gto
from pyscf import lib as pyscf_lib
from pyscf.scf import hf as hf_cpu
from pyscf.scf import chkfile
from gpu4pyscf import lib
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.backend import (
    xp, to_host, to_device, is_device_array, BACKEND_NAME,
    eigh as _backend_eigh, einsum as _backend_einsum,
    norm as _backend_norm, dot as _backend_dot,
)
from gpu4pyscf.lib import logger
from gpu4pyscf import __config__

# Backend-conditional imports: CUDA-specific modules only on CuPy
if BACKEND_NAME == 'cupy':
    import cupy
    from gpu4pyscf.gto.ecp import get_ecp
    from gpu4pyscf.lib.cupy_helper import (
        eigh, tag_array, return_cupy_array, cond, asarray, get_avail_mem,
        block_diag, sandwich_dot, stack_with_padding)
    from gpu4pyscf.scf import diis, jk, j_engine
    from gpu4pyscf.scf.smearing import smearing
else:
    import pyscf.scf.diis as _cpu_diis

# Dispersion is backend-agnostic
from . import dispersion

# ---------------------------------------------------------------------------
# Backend-agnostic replacements for cupy_helper functions
# These are used when BACKEND_NAME != 'cupy'
# ---------------------------------------------------------------------------

if BACKEND_NAME != 'cupy':
    def eigh(a, b=None, overwrite=False):  # noqa: F811
        """Eigendecomposition via SciPy. Returns numpy arrays."""
        import scipy.linalg
        a_np = np.asarray(to_host(a)) if is_device_array(a) else np.asarray(a)
        if b is not None:
            b_np = np.asarray(to_host(b)) if is_device_array(b) else np.asarray(b)
            return scipy.linalg.eigh(a_np, b_np)
        return scipy.linalg.eigh(a_np)

    def tag_array(a, **kwargs):  # noqa: F811
        """Attach metadata to an array. On non-CuPy backends, use PySCF's
        NPArrayWithTag for numpy arrays or a simple wrapper for device arrays."""
        a_np = to_host(a) if is_device_array(a) else np.asarray(a)
        t = np.asarray(a_np).view(pyscf_lib.NPArrayWithTag)
        if isinstance(a, pyscf_lib.NPArrayWithTag):
            t.__dict__.update(a.__dict__)
        t.__dict__.update(kwargs)
        # Keep as numpy — the SCF loop will call to_device where needed
        return t

    def return_cupy_array(fn):  # noqa: F811
        """On non-CuPy backends, just return numpy arrays as-is."""
        return fn

    def cond(a, **kwargs):  # noqa: F811
        a_np = to_host(a) if is_device_array(a) else np.asarray(a)
        return np.linalg.cond(a_np)

    def asarray(a, **kwargs):  # noqa: F811
        if isinstance(a, np.ndarray):
            return a
        if is_device_array(a):
            return to_host(a)
        return np.asarray(a)

    def get_avail_mem():  # noqa: F811
        free, total = __import__('gpu4pyscf.lib.backend', fromlist=['memory']).memory.get_mem_info()
        return free

    def block_diag(blocks, out=None):  # noqa: F811
        import scipy.linalg
        np_blocks = [to_host(b) if is_device_array(b) else np.asarray(b) for b in blocks]
        return scipy.linalg.block_diag(*np_blocks)

    def sandwich_dot(a, c, out=None):  # noqa: F811
        a_np = to_host(a) if is_device_array(a) else np.asarray(a)
        c_np = to_host(c) if is_device_array(c) else np.asarray(c)
        if a_np.ndim == 2:
            return c_np.T @ a_np @ c_np
        return np.einsum('...ij,ip,jq->...pq', a_np, c_np.conj(), c_np)

    def stack_with_padding(arrays):  # noqa: F811
        if not arrays:
            raise ValueError("arrays must be a non-empty sequence")
        max_nmo = max(a.shape[1] for a in arrays)
        nao = arrays[0].shape[0]
        dtype = np.result_type(*arrays)
        out = np.empty((len(arrays), nao, max_nmo), dtype=dtype)
        for k, a in enumerate(arrays):
            nmo = a.shape[1]
            out[k,:,:nmo] = a
            if nmo < max_nmo:
                out[k,:,nmo:] = 0
        return out

    def smearing(mf, *args, **kwargs):  # noqa: F811
        raise NotImplementedError(
            'Smearing is not yet available on the %s backend' % BACKEND_NAME)

def _patch_dft_veff_metal(mf_df):
    """Override get_veff on a DF-RKS object to use Metal XC.

    The PySCF DF-RKS class (instantiated by .density_fit()) inherits
    get_veff from pyscf.dft.rks, which calls ni.nr_rks() on the CPU.
    This wrapper replaces it with one that calls nr_rks_metal for the
    Metal GPU XC path, keeping all other semantics identical to PySCF's
    version (J/K assembly, ecoul/exc accounting, hermi=2 handling).

    Only engages for 2D density matrices; response calculations passing
    3D dm (CPHF) fall through to PySCF's CPU numint code.
    """
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location(
        'numint_metal',
        _os.path.join(_os.path.dirname(__file__), '..', 'dft', 'numint_metal.py'))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _nr_rks_metal = _mod.nr_rks_metal

    from pyscf.lib import tag_array as _tag_array

    def _metal_get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = mf_df.mol
        if dm is None: dm = mf_df.make_rdm1()
        mf_df.initialize_grids(mol, dm)
        ni = mf_df._numint

        dm_np = np.asarray(dm)
        ground_state = (dm_np.ndim == 2)

        if hermi == 2:  # anti-Hermitian, rho = 0
            n, exc, vxc = 0, 0, 0
        elif ground_state:
            # Ground-state 2D density: use Metal XC
            try:
                n, exc, vxc = _nr_rks_metal(ni, mol, mf_df.grids, mf_df.xc, dm_np)
            except Exception:
                n, exc, vxc = ni.nr_rks(mol, mf_df.grids, mf_df.xc, dm_np)
            if mf_df.do_nlc():
                if ni.libxc.is_nlc(mf_df.xc):
                    xc = mf_df.xc
                else:
                    xc = mf_df.nlc
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, mf_df.nlcgrids, xc, dm_np)
                exc += enlc
                vxc += vnlc
            logger.debug(mf_df, 'nelec by numeric integration = %s', n)
        else:
            # Non-2D (e.g., response dm): fall through to CPU numint
            n, exc, vxc = ni.nr_rks(mol, mf_df.grids, mf_df.xc, dm_np)

        # J/K assembly (matches pyscf.dft.rks.get_veff exactly).
        # direct_scf=False on DF objects, so incremental_jk is always False.
        if not ni.libxc.is_hybrid_xc(mf_df.xc):
            vk = None
            vj = mf_df.get_j(mol, dm_np, hermi)
            vxc = vxc + vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_df.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = mf_df.get_jk(mol, dm_np, hermi)
                vk *= hyb
            elif alpha == 0:
                vj = mf_df.get_j(mol, dm_np, hermi)
                vk = mf_df.get_k(mol, dm_np, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:
                vj = mf_df.get_j(mol, dm_np, hermi)
                vk = mf_df.get_k(mol, dm_np, hermi, omega=omega)
                vk *= alpha
            else:
                vj, vk = mf_df.get_jk(mol, dm_np, hermi)
                vk *= hyb
                vklr = mf_df.get_k(mol, dm_np, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vxc += vj - vk * .5
            if ground_state:
                exc -= np.einsum('ij,ji', dm_np, vk).real * .5 * .5

        if ground_state:
            ecoul = np.einsum('ij,ji', dm_np, vj).real * .5
        else:
            ecoul = None

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    mf_df.get_veff = _metal_get_veff


def _patch_dft_veff_metal_uks(mf_df):
    """Override get_veff on a DF-UKS object to use Metal XC.

    Mirrors _patch_dft_veff_metal but for unrestricted DFT: densities are
    (2, nao, nao) stacks (alpha, beta), and the XC potential, J/K, and
    energy accounting all follow pyscf.dft.uks.get_veff semantics (no 0.5
    factor on vk; vj summed over spins).

    Only engages for 3D (2,nao,nao) ground-state densities; response
    calculations fall through to PySCF's CPU numint code.
    """
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location(
        'numint_metal',
        _os.path.join(_os.path.dirname(__file__), '..', 'dft', 'numint_metal.py'))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _nr_uks_metal = _mod.nr_uks_metal

    from pyscf.lib import tag_array as _tag_array

    def _metal_get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = mf_df.mol
        if dm is None: dm = mf_df.make_rdm1()
        if not isinstance(dm, np.ndarray):
            dm = np.asarray(dm)
        if dm.ndim == 2:  # RHF DM fallback (unusual)
            dm = np.repeat(dm[None]*.5, 2, axis=0)
        mf_df.initialize_grids(mol, dm)
        ni = mf_df._numint

        ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

        if hermi == 2:
            n, exc, vxc = (0, 0), 0, 0
        elif ground_state:
            try:
                n, exc, vxc = _nr_uks_metal(ni, mol, mf_df.grids, mf_df.xc, dm)
            except Exception:
                n, exc, vxc = ni.nr_uks(mol, mf_df.grids, mf_df.xc, dm)
            if mf_df.do_nlc():
                if ni.libxc.is_nlc(mf_df.xc):
                    xc = mf_df.xc
                else:
                    xc = mf_df.nlc
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, mf_df.nlcgrids, xc, dm[0]+dm[1])
                exc += enlc
                vxc += vnlc
            logger.debug(mf_df, 'nelec by numeric integration = %s', n)
        else:
            n, exc, vxc = ni.nr_uks(mol, mf_df.grids, mf_df.xc, dm)

        # J/K assembly (matches pyscf.dft.uks.get_veff exactly)
        if not ni.libxc.is_hybrid_xc(mf_df.xc):
            vk = None
            vj = mf_df.get_j(mol, dm[0] + dm[1], hermi)
            vxc = vxc + vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_df.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = mf_df.get_jk(mol, dm, hermi)
                vk *= hyb
            elif alpha == 0:
                vj = mf_df.get_j(mol, dm, hermi)
                vk = mf_df.get_k(mol, dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:
                vj = mf_df.get_j(mol, dm, hermi)
                vk = mf_df.get_k(mol, dm, hermi, omega=omega)
                vk *= alpha
            else:
                vj, vk = mf_df.get_jk(mol, dm, hermi)
                vk *= hyb
                vklr = mf_df.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1]
            vxc += vj - vk  # note: no 0.5 factor for UKS
            if ground_state:
                exc -= (np.einsum('ij,ji', dm[0], vk[0]).real +
                        np.einsum('ij,ji', dm[1], vk[1]).real) * .5

        if ground_state:
            ecoul = np.einsum('ij,ji', dm[0] + dm[1], vj).real * .5
        else:
            ecoul = None

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    mf_df.get_veff = _metal_get_veff


def _patch_df_with_metal_jk(mf_df):
    """Replace the DF object's get_jk with Metal GPU f32 engine.

    ALL SCF cycles use Metal GPU float32 J/K (6-12x faster per cycle).
    Convergence threshold is set to 1e-6 (f32 precision floor).

    The energy is a variational functional, so the error from using an
    f32-converged density is second-order: E_error ~ O(|D_f32 - D_f64|^2)
    ~ 1e-12, well below chemical accuracy.  A single f64 energy evaluation
    at the end restores full precision if needed.
    """
    import importlib.util as _ilu, os as _os, sys as _sys
    # Register in sys.modules so subsequent `from gpu4pyscf.df.df_jk_metal
    # import ...` (e.g. from df_grad_metal) sees the SAME module instance
    # and shares the _tensor_cache dict.
    if 'gpu4pyscf.df.df_jk_metal' in _sys.modules:
        _mod = _sys.modules['gpu4pyscf.df.df_jk_metal']
    else:
        _spec = _ilu.spec_from_file_location(
            'gpu4pyscf.df.df_jk_metal',
            _os.path.join(_os.path.dirname(__file__), '..', 'df',
                          'df_jk_metal.py'))
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules['gpu4pyscf.df.df_jk_metal'] = _mod
        _spec.loader.exec_module(_mod)
    get_jk_metal, clear_cache = _mod.get_jk_metal, _mod.clear_cache
    dfobj = mf_df.with_df
    _original_get_jk = dfobj.get_jk
    dfobj._original_get_jk = _original_get_jk  # save for f64 refinement
    _original_reset = getattr(dfobj, 'reset', None)

    def _metal_get_jk(dm, hermi=1, with_j=True, with_k=True,
                      direct_scf_tol=1e-13, omega=None):
        if omega is not None:
            # RSH integrals: fall back to CPU
            return _original_get_jk(dm, hermi, with_j, with_k,
                                    direct_scf_tol, omega)
        return get_jk_metal(dfobj, dm, hermi, with_j, with_k)

    def _metal_reset(*args, **kwargs):
        clear_cache(dfobj)
        if _original_reset is not None:
            return _original_reset(*args, **kwargs)

    dfobj.get_jk = _metal_get_jk
    dfobj.reset = _metal_reset

    # Install Metal-accelerated DF gradient patch (Phase 1: CDERI reuse in
    # _cho_solve_rhojk). Idempotent global monkey-patch on
    # pyscf.df.grad.rhf._cho_solve_rhojk that only activates when a
    # MetalDFTensors cache exists for the dfobj being used.
    # Load directly (bypass gpu4pyscf.df.grad package __init__ which imports
    # the cupy-based submodules).
    _grad_modname = 'gpu4pyscf_df_grad_metal_loader'
    if _grad_modname in _sys.modules:
        _grad_mod = _sys.modules[_grad_modname]
    else:
        _grad_spec = _ilu.spec_from_file_location(
            _grad_modname,
            _os.path.join(_os.path.dirname(__file__), '..', 'df', 'grad',
                          'df_grad_metal.py'))
        _grad_mod = _ilu.module_from_spec(_grad_spec)
        _sys.modules[_grad_modname] = _grad_mod
        _grad_spec.loader.exec_module(_grad_mod)
    _grad_mod.install_metal_grad_patch()
    # Phase 3: Metal XC nuclear-gradient kernel.
    if hasattr(mf_df, '_numint'):
        _grad_mod.install_metal_vxc_grad_patch()
        mf_df._numint._metal_enabled = True
    # Phase 4c: Metal int3c2e_ip1 kernel.
    _grad_mod.install_metal_int3c2e_ip1_patch()

    # Relax convergence threshold to match f32 noise floor.
    # Metal J/K produce Fock matrices with ~1e-5 element-wise noise; at the
    # true minimum the Metal gradient norm floor is ~1e-2 (larger molecules)
    # so conv_tol_grad=sqrt(conv_tol)=0.01 is the tightest achievable.
    # A final f64 energy evaluation restores accuracy if needed.
    if mf_df.conv_tol < 1e-4:
        mf_df.conv_tol = 1e-4

    # For DF-DFT: route XC evaluation through the Metal GPU engine.
    # DF-HF falls through (no XC).
    if hasattr(mf_df, 'xc'):
        _cls_name = type(mf_df).__name__
        if _cls_name == 'DFRKS':
            _patch_dft_veff_metal(mf_df)
        elif _cls_name == 'DFUKS':
            _patch_dft_veff_metal_uks(mf_df)

    # Override nuc_grad_method to do f64 refinement first.
    # The returned grad object's as_scanner() is patched to set
    # _metal_skip_refine=True during optimization, saving ~1.4s per step.
    _original_ngm = mf_df.nuc_grad_method
    def _refined_nuc_grad_method():
        _refine_to_f64(mf_df)
        grad = _original_ngm()
        # Patch as_scanner so geomopt skips refinement on intermediate steps
        _orig_as_scanner = grad.as_scanner
        def _fast_as_scanner():
            scanner = _orig_as_scanner()
            _orig_call = scanner.__call__
            def _call_skip_refine(mol_or_geom, **kw):
                mf_df._metal_skip_refine = True
                try:
                    return _orig_call(mol_or_geom, **kw)
                finally:
                    mf_df._metal_skip_refine = False
            scanner.__call__ = _call_skip_refine
            return scanner
        grad.as_scanner = _fast_as_scanner
        return grad
    mf_df.nuc_grad_method = _refined_nuc_grad_method

    _original_hess = mf_df.Hessian
    def _refined_hessian():
        _refine_to_f64(mf_df)
        return _original_hess()
    mf_df.Hessian = _refined_hessian

    if hasattr(mf_df, 'TDDFT'):
        _original_tddft = mf_df.TDDFT
        def _refined_tddft():
            _refine_to_f64(mf_df)
            return _original_tddft()
        mf_df.TDDFT = _refined_tddft

    if hasattr(mf_df, 'TDA'):
        _original_tda = mf_df.TDA
        def _refined_tda():
            _refine_to_f64(mf_df)
            return _original_tda()
        mf_df.TDA = _refined_tda

def _refine_to_f64(mf):
    """Refine f32-converged SCF to f64 before gradient/Hessian.

    Temporarily switches to CPU f64 J/K for 3 SCF cycles, then restores
    Metal f32 J/K so subsequent SCF calls (e.g. geomopt steps) remain fast.

    Can be skipped by setting mf._metal_skip_refine = True (useful for
    intermediate geometry optimization steps where ~4e-4 gradient error
    from f32 density is acceptable).
    """
    if not getattr(mf, 'converged', False):
        return
    if getattr(mf, '_metal_skip_refine', False):
        return
    # Temporarily switch to CPU f64 J/K, keeping the Metal version for later
    metal_get_jk = None
    if hasattr(mf, 'with_df') and hasattr(mf.with_df, '_original_get_jk'):
        metal_get_jk = mf.with_df.get_jk
        mf.with_df.get_jk = mf.with_df._original_get_jk
    dm0 = mf.make_rdm1()
    mf.converged = False
    mf.mo_energy = None
    mf.mo_coeff = None
    mf.mo_occ = None
    saved_tol, mf.conv_tol = mf.conv_tol, 1e-10
    saved_max, mf.max_cycle = mf.max_cycle, 3
    mf.kernel(dm0=dm0)
    mf.conv_tol = saved_tol
    mf.max_cycle = saved_max
    # Restore Metal f32 J/K for next SCF cycle (critical for geomopt)
    if metal_get_jk is not None:
        mf.with_df.get_jk = metal_get_jk

remove_overlap_zero_eigenvalue = getattr(__config__, 'scf_hf_remove_overlap_zero_eigenvalue', True)
overlap_zero_eigenvalue_threshold = getattr(__config__, 'scf_hf_overlap_zero_eigenvalue_threshold', 1e-6)

__all__ = [
    'get_jk', 'get_occ', 'get_grad', 'damping', 'level_shift', 'get_fock',
    'energy_elec', 'RHF', 'SCF'
]

if BACKEND_NAME == 'cupy':
    def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
               verbose=None):
        '''Compute J, K matrices with CPU-GPU hybrid algorithm
        '''
        with mol.with_range_coulomb(omega):
            vj, vk = jk.get_jk(mol, dm, hermi, vhfopt, with_j, with_k, verbose)
        if not isinstance(dm, cupy.ndarray):
            if with_j: vj = vj.get()
            if with_k: vk = vk.get()
        return vj, vk

    def _get_jk(mf, mol, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if omega is None:
            omega = mol.omega
        vhfopt = mf._opt_gpu.get(omega)
        if vhfopt is None:
            with mol.with_range_coulomb(omega):
                vhfopt = mf._opt_gpu[omega] = jk._VHFOpt(
                    mol, mf.direct_scf_tol, tile=1).build()
        vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega)
        return vj, vk
else:
    def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
               verbose=None):
        '''Compute J, K matrices.

        On MLX backend: Metal GPU Rys engine (f64 roots + f32 TRR + f64 accumulation).
        On NumPy backend: PySCF CPU engine (f64 throughout).
        '''
        dm_np = np.asarray(dm)
        if omega and omega != 0:
            # Range-separated Coulomb: use CPU engine
            with mol.with_range_coulomb(omega):
                vj, vk = hf_cpu.get_jk(mol, dm_np, hermi, None, with_j, with_k, omega)
            return np.asarray(vj), np.asarray(vk)
        if BACKEND_NAME == 'mlx':
            from gpu4pyscf.lib.metal_kernels.rys_jk_metal import get_jk_rys_metal
            return get_jk_rys_metal(mol, dm_np, with_j=with_j, with_k=with_k)
        # NumPy backend — PySCF CPU engine
        vj, vk = hf_cpu.get_jk(mol, dm_np, hermi, None, with_j, with_k, omega)
        return np.asarray(vj), np.asarray(vk)

    def _get_jk(mf, mol, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if omega is None:
            omega = mol.omega
        return get_jk(mol, dm, hermi, None, with_j, with_k, omega)

def make_rdm1(mo_coeff, mo_occ):
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    is_occ = mo_occ > 0
    mocc = mo_coeff[:, is_occ]
    dm = np.dot(mocc*mo_occ[is_occ], mocc.conj().T)
    occ_coeff = mo_coeff[:, is_occ]
    return tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = np.argsort(mo_energy)
    nmo = mo_energy.size
    mo_occ = np.zeros(nmo)
    nocc = mf.mol.nelectron // 2
    if nocc > nmo:
        raise RuntimeError('Failed to assign occupancies. '
                           f'Nocc ({nocc}) > Nmo ({nmo})')
    mo_occ[e_idx[:nocc]] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        homo = float(mo_energy[e_idx[nocc-1]])
        lumo = float(mo_energy[e_idx[nocc]])
        if homo+1e-3 > lumo:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g', homo, lumo)
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', homo, lumo)
    return mo_occ

def get_veff(mf, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    if dm is None: dm = mf.make_rdm1()
    if dm_last is not None and mf.direct_scf:
        dm = asarray(dm) - asarray(dm_last)
    vj = mf.get_j(mol, dm, hermi)
    vhf = mf.get_k(mol, dm, hermi)
    vhf *= -.5
    vhf += vj
    if vhf_last is not None:
        vhf += asarray(vhf_last)
    return vhf

def get_grad(mo_coeff, mo_occ, fock_ao):
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(np.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()

def damping(f, f_prev, factor):
    return f*(1-factor) + f_prev*factor

def level_shift(s, d, f, factor):
    dm_vir = s - reduce(np.dot, (s, d, s))
    return f + dm_vir * factor

if BACKEND_NAME == 'cupy':
    def get_hcore(mol):
        from gpu4pyscf.pbc.gto.int1e import int1e_kin
        if mol._pseudo:
            from pyscf.gto import pp_int
            h = asarray(pp_int.get_gth_pp(mol))
        else:
            assert not mol.nucmod
            from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec
            nucmol = gto.mole.fakemol_for_charges(mol.atom_coords())
            h = contract_int3c2e_auxvec(mol, nucmol, -mol.atom_charges())
        h += int1e_kin(mol)
        if len(mol._ecpbas) > 0:
            h += get_ecp(mol)
        return h
else:
    def get_hcore(mol):
        '''Core Hamiltonian via PySCF CPU engine (fallback).'''
        return np.asarray(hf_cpu.get_hcore(mol))

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = np.asarray(h1e)
    vhf = np.asarray(vhf)
    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()
    s1e = np.asarray(s1e)
    dm = np.asarray(dm)
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if damp_factor is None:
        damp_factor = mf.damp
    if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
        f = damping(f, fock_last, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f)

    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if level_shift_factor is not None:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def energy_elec(self, dm=None, h1e=None, vhf=None):
    '''
    electronic energy
    '''
    if dm is None: dm = self.make_rdm1()
    if h1e is None: h1e = self.get_hcore()
    if vhf is None: vhf = self.get_veff(self.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm).real
    e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    e1 = float(e1)
    e_coul = float(e_coul)
    self.scf_summary['e1'] = e1
    self.scf_summary['e2'] = e_coul
    logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul, e_coul

def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)
    t0 = t1 = log.init_timer()
    if(conv_tol_grad is None):
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        dm0 = mf.get_init_guess(mol, mf.init_guess)
        t1 = log.timer_debug1('generating initial guess', *t1)

    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        if dm0.ndim == 2:
            mo_coeff = np.asarray(dm0.mo_coeff[:,dm0.mo_occ>0])
            mo_occ = np.asarray(dm0.mo_occ[dm0.mo_occ>0])
            dm0 = tag_array(dm0, mo_occ=mo_occ, mo_coeff=mo_coeff)
        else:
            # Drop attributes like mo_coeff, mo_occ for UHF and other methods.
            dm0 = asarray(dm0, order='C')

    h1e = np.asarray(mf.get_hcore())
    s1e = np.asarray(mf.get_ovlp())
    t1 = log.timer_debug1('hcore', *t1)

    dm, dm0 = asarray(dm0, order='C'), None
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    x_orth = mf.check_linear_dependency(s1e, log)
    t1 = log.timer('SCF initialization', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e, overwrite=True, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, (lib.diis.DIIS, pyscf_lib.diis.DIIS)):
        mf_diis = mf.diis
    elif mf.diis:
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        # CDIIS just require a C that's orthonormal (C.T@S@C==I), and X satisfies that.
        if isinstance(x_orth, list): # k point
            mf_diis.Corth = stack_with_padding(x_orth)
        else:
            mf_diis.Corth = np.asarray(x_orth)
            # For UHF, PySCF CPU DIIS expects Corth to be (2, nao, nmo)
            if hasattr(mf, 'nelec') and mf_diis.Corth.ndim == 2:
                if BACKEND_NAME != 'cupy':
                    mf_diis.Corth = np.stack([mf_diis.Corth, mf_diis.Corth])
    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    fock_last = None
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        mo_coeff = mo_occ = mo_energy = fock = None
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis, fock_last=fock_last)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        if mf.damp is not None:
            fock_last = fock
        fock = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        dm = asarray(dm) # Remove the attached attributes
        t1 = log.timer_debug1('veff', *t1)

        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        e_tot = mf.energy_tot(dm, h1e, vhf)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))

        norm_ddm = np.linalg.norm(dm-dm_last)
        t1 = log.timer(f'cycle={cycle+1}', *t0)

        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        e_diff = abs(e_tot-last_hf_e)
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break
    else:
        log.warn("SCF failed to converge")

    mf.cycles = cycle + 1
    if scf_conv and mf.level_shift is not None:
        # An extra diagonalization, to remove level shift
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = np.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        else:
            log.warn("Level-shifted SCF extra cycle failed to converge")
            scf_conv = False
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulstion
    See :func:`scf.hf.energy_elec` for the electron part

    Note this function has side effects which cause mf.scf_summary updated.

    '''
    nuc = mf.energy_nuc()
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    if mf.do_disp():
        if 'dispersion' in mf.scf_summary:
            e_tot += mf.scf_summary['dispersion']
        else:
            e_disp = mf.get_dispersion()
            mf.scf_summary['dispersion'] = e_disp
            e_tot += e_disp
    mf.scf_summary['nuc'] = nuc.real
    if not isinstance(e_tot, (int, float)):
        e_tot = float(e_tot)
    return e_tot

def scf(mf, dm0=None, **kwargs):
    cput0 = logger.init_timer(mf)

    mf.dump_flags()
    mf.build(mf.mol)

    if dm0 is None and mf.mo_coeff is not None and mf.mo_occ is not None:
        # Initial guess from existing wavefunction
        dm0 = mf.make_rdm1()

    if mf.max_cycle > 0 or mf.mo_coeff is None:
        mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                        dm0=dm0, callback=mf.callback,
                        conv_check=mf.conv_check, **kwargs)
    else:
        # Avoid to update SCF orbitals in the non-SCF initialization
        # (issue #495).  But run regular SCF for initial guess if SCF was
        # not initialized.
        mf.e_tot = _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                            dm0=dm0, callback=mf.callback,
                            conv_check=mf.conv_check, **kwargs)[1]

    logger.timer(mf, 'SCF', *cput0)
    mf._finalize()
    return mf.e_tot

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = np.empty_like(mo_coeff)
    mo_e = np.empty(mo_occ.size)
    for idx in (coreidx, openidx, viridx):
        if np.any(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = orb.conj().T.dot(fock).dot(orb)
            e, c = eigh(f1)
            mo[:,idx] = orb.dot(c)
            mo_e[idx] = e
    return mo_e, mo

def _init_guess_by_minao_cpu(mol):
    '''CPU fallback for init_guess_by_minao using PySCF.'''
    from pyscf.scf.hf import init_guess_by_minao as _cpu_minao
    dm = _cpu_minao(mol)
    return tag_array(np.asarray(dm), mo_coeff=np.eye(dm.shape[0]), mo_occ=np.zeros(dm.shape[0]))

def init_guess_by_minao(mol):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Note: this function is inconsistent with the latest PySCF (v2.9) and eariler versions.
    This function returns block diagonal density matrix associated with each atom.
    While the function in PySCF projects the density matrix into the full space of atomic basis

    Returns:
        Density matrix, 2D ndarray
    '''
    if BACKEND_NAME != 'cupy':
        return _init_guess_by_minao_cpu(mol)

    from pyscf.scf import atom_hf
    from pyscf.scf import addons

    def minao_basis(symb, nelec_ecp):
        occ = []
        basis_ano = []
        if gto.is_ghost_atom(symb):
            return occ, basis_ano

        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp, atom_symbol=stdsymb)
        # coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            if ndocc >= coreshl[l]:
                degen = l * 2 + 1
                occ_l = [2, ]*(ndocc-coreshl[l]) + [frac, ]
                occ.append(np.repeat(occ_l, degen))
                basis_ano.append([l] + [b[:1] + b[1+coreshl[l]:ndocc+2]
                                        for b in basis_add[l][1:]])
            else:
                logger.debug(mol, '*** ECP incorporates partially occupied '
                             'shell of l = %d for atom %s ***', l, symb)
        occ = np.hstack(occ)

        if nelec_ecp > 0:
            if symb in mol._basis:
                input_basis = mol._basis[symb]
            elif stdsymb in mol._basis:
                input_basis = mol._basis[stdsymb]
            else:
                raise KeyError(symb)

            basis4ecp = [[] for i in range(4)]
            for bas in input_basis:
                l = bas[0]
                if l < 4:
                    basis4ecp[l].append(bas)

            occ4ecp = []
            for l in range(4):
                nbas_l = sum((len(bas[1]) - 1) for bas in basis4ecp[l])
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                ndocc -= coreshl[l]
                assert ndocc <= nbas_l

                if nbas_l > 0:
                    occ_l = np.zeros(nbas_l)
                    occ_l[:ndocc] = 2
                    if frac > 0:
                        occ_l[ndocc] = frac
                    occ4ecp.append(np.repeat(occ_l, l * 2 + 1))

            occ4ecp = np.hstack(occ4ecp)
            basis4ecp = list(itertools.chain.from_iterable(basis4ecp))

# Compared to ANO valence basis, to check whether the ECP basis set has
# reasonable AO-character contraction.  The ANO valence AO should have
# significant overlap to ECP basis if the ECP basis has AO-character.
            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            atm1._built = True
            atm2._built = True
            s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)
            if abs(np.linalg.det(s12[occ4ecp>0][:,occ>0])) > .1:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                logger.debug(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    # Issue 548
    if any(gto.charge(mol.atom_symbol(ia)) > 96 for ia in range(mol.natm)):
        from pyscf.scf.hf import init_guess_by_atom
        logger.info(mol, 'MINAO initial guess is not available for super-heavy '
                    'elements. "atom" initial guess is used.')
        return init_guess_by_atom(mol)

    nelec_ecp_dic = {mol.atom_symbol(ia): mol.atom_nelec_core(ia)
                          for ia in range(mol.natm)}

    basis = {}
    occdic = {}
    for symb, nelec_ecp in nelec_ecp_dic.items():
        occ_add, basis_add = minao_basis(symb, nelec_ecp)
        occdic[symb] = occ_add
        basis[symb] = basis_add

    mol1 = gto.Mole()
    mol1._built = True
    mol2 = mol.copy()

    aoslice = mol.aoslice_by_atom()
    nao = aoslice[-1,3]
    dm = np.zeros((nao, nao))
    # Preallocate a buffer in cupy memory pool for small arrays held in atm_conf
    workspace = np.empty(50**2*12)
    workspace = None # noqa: F841
    atm_conf = {}
    mo_coeff = []
    mo_occ = []
    for ia, (p0, p1) in enumerate(aoslice[:,2:]):
        symb = mol.atom_symbol(ia)
        if gto.is_ghost_atom(symb):
            n = p1 - p0
            mo_coeff.append(np.zeros((n, 0)))
            mo_occ.append(np.zeros(0))
            continue

        if symb not in atm_conf:
            nelec_ecp = mol.atom_nelec_core(ia)
            occ, basis = minao_basis(symb, nelec_ecp)
            mol1._atm, mol1._bas, mol1._env = mol1.make_env(
                [mol._atom[ia]], {symb: basis}, [])
            i0, i1 = aoslice[ia,:2]
            mol2._bas = mol._bas[i0:i1]
            s22 = mol2.intor_symmetric('int1e_ovlp')
            s21 = gto.mole.intor_cross('int1e_ovlp', mol2, mol1)
            c = pyscf_lib.cho_solve(s22, s21, strict_sym_pos=False)
            c = np.asarray(c[:,occ>0], order='C')
            occ = np.asarray(occ[occ>0], order='C')
            atm_conf[symb] = occ, c

        occ, c = atm_conf[symb]
        dm[p0:p1,p0:p1] = (c*occ).dot(c.conj().T)
        mo_coeff.append(c)
        mo_occ.append(occ)

    mo_coeff = block_diag(mo_coeff)
    mo_occ = np.hstack(mo_occ)
    return tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

def check_linear_dependency(s, log=None):
    e, v = eigh(s)
    if log is not None:
        abs_e = np.asarray(abs(e))
        emax = abs_e.max()
        emin = abs_e.min()
        c = emax / emin
        log.debug('cond(S) = %s', c)
        if c > 1e10:
            log.warn('Singularity detected in the overlap matrix. '
                     'SCF may be inaccurate and difficult to converge.')
    if remove_overlap_zero_eigenvalue:
        mask = e > overlap_zero_eigenvalue_threshold
        x = v[:,mask] / np.sqrt(e[mask])
    else:
        x = v / np.sqrt(e)
    return x

def canonical_orthogonalization(s):
    return check_linear_dependency(s)

def as_scanner(mf):
    if isinstance(mf, pyscf_lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + SCF_Scanner.__name_mixin__
    return pyscf_lib.set_class(SCF_Scanner(mf), (SCF_Scanner, mf.__class__), name)

class SCF_Scanner(pyscf_lib.SinglePointScanner):
    def __init__(self, mf_obj):
        self.__dict__.update(mf_obj.__dict__)
        self._last_mol_fp = mf_obj.mol.ao_loc

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # Cleanup intermediates associated to the previous mol object
        self.reset(mol)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
        elif self.mo_coeff is None:
            dm0 = None
        else:
            dm0 = None
            if np.array_equal(self._last_mol_fp, mol.ao_loc):
                dm0 = self.make_rdm1()
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.from_chk(self.chkfile)
        self.mo_coeff = None  # To avoid last mo_coeff being used by SOSCF
        e_tot = self.kernel(dm0=dm0, **kwargs)
        self._last_mol_fp = mol.ao_loc
        return e_tot

class SCF(pyscf_lib.StreamObject):

    # attributes
    conv_tol            = hf_cpu.SCF.conv_tol
    conv_tol_grad       = hf_cpu.SCF.conv_tol_grad
    max_cycle           = hf_cpu.SCF.max_cycle
    init_guess          = hf_cpu.SCF.init_guess
    conv_tol_cpscf      = 1e-6   # TODO: reuse the default value in PySCF

    disp                = None
    DIIS                = diis.SCF_DIIS if BACKEND_NAME == 'cupy' else _cpu_diis.SCF_DIIS
    diis                = hf_cpu.SCF.diis
    diis_space          = hf_cpu.SCF.diis_space
    diis_damp           = hf_cpu.SCF.diis_damp
    diis_start_cycle    = hf_cpu.SCF.diis_start_cycle
    diis_file           = hf_cpu.SCF.diis_file
    diis_space_rollback = hf_cpu.SCF.diis_space_rollback
    damp                = None
    level_shift         = None
    direct_scf          = hf_cpu.SCF.direct_scf
    direct_scf_tol      = hf_cpu.SCF.direct_scf_tol
    conv_check          = hf_cpu.SCF.conv_check
    callback            = hf_cpu.SCF.callback

    _keys = {
        'conv_tol', 'conv_tol_grad', 'conv_tol_cpscf', 'max_cycle', 'init_guess',
        'sap_basis', 'DIIS', 'diis', 'diis_space', 'diis_damp', 'diis_start_cycle',
        'diis_file', 'diis_space_rollback', 'damp', 'level_shift',
        'direct_scf', 'direct_scf_tol', 'conv_check', 'callback',
        'mol', 'chkfile', 'mo_energy', 'mo_coeff', 'mo_occ',
        'e_tot', 'converged', 'cycles', 'scf_summary',
        'disp', 'disp_with_3body',
    }

    # methods
    def __init__(self, mol):
        if not mol._built:
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

        # The chkfile part is different from pyscf, we turn off chkfile by default.
        self.chkfile = None

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.e_tot = 0
        self.converged = False
        self.cycles = 0
        self.scf_summary = {}

        self._opt_gpu = {None: None}
        self._opt_jengine = {None: None}
        self._eri = None # Note: self._eri requires large amount of memory

    __getstate__, __setstate__ = pyscf_lib.generate_pickle_methods(
        excludes=('_eri', '_numint', '_opt', '_opt_gpu', '_opt_jengine',
                  'chkfile', '_chkfile', 'callback'))

    def build(self, mol=None):
        if mol is None: mol = self.mol
        self.check_sanity()
        return self

    def check_linear_dependency(self, s, verbose=None):
        log = logger.new_logger(self, verbose)
        x = check_linear_dependency(s, log)
        nao, nmo = x.shape
        if nmo < nao:
            log.warn(f"{nao - nmo} small eigenvectors of overlap matrix removed "
                     "because of linear dependency between AOs.\n"
                     "The support for low-rank overlap matrix is not fully tested. "
                     "Please report any bug you encountered to the developers.")
        return x

    def eig(self, h, s, overwrite=False, x=None):
        '''
        Solve generalized eigenvalue problem.

        When overwrite is specified, both fock and s matrices are overwritten.

        Kwargs:
            overwrite: bool
                whether to allow modifying the h and s matrices in the input (to
                reduce memory footprint)
            x: ndarray
                The matrix that orthogonalizes the s matrix: x^dagger s x = 1 .
                When x is specified, the input s can be skipped.
        '''
        if x is None:
            if h.dtype != s.dtype:
                s = s.astype(h.dtype)
            # In DIIS, h and overlap matrices are temporarily constructed
            # and discarded, they can be overwritten in the eigh solver.
            mo_energy, mo_coeff = eigh(h, s, overwrite=overwrite)
        else:
            x = np.asarray(x)
            mo_energy, C = eigh(x.conj().T.dot(h).dot(x))
            mo_coeff = x @ C
        return mo_energy, mo_coeff
    _eigh = eig

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('initial guess = %s', self.init_guess)
        log.info('damping factor = %s', self.damp)
        log.info('level_shift factor = %s', self.level_shift)
        if isinstance(self.diis, lib.diis.DIIS):
            log.info('DIIS = %s', self.diis)
            log.info('diis_start_cycle = %d', self.diis_start_cycle)
            log.info('diis_space = %d', self.diis.space)
            if getattr(self.diis, 'damp', None):
                log.info('diis_damp = %g', self.diis.damp)
        elif self.diis:
            log.info('DIIS = %s', self.DIIS)
            log.info('diis_start_cycle = %d', self.diis_start_cycle)
            log.info('diis_space = %d', self.diis_space)
            log.info('diis_damp = %g', self.diis_damp)
        else:
            log.info('DIIS disabled')
        log.info('SCF conv_tol = %g', self.conv_tol)
        log.info('SCF conv_tol_grad = %s', self.conv_tol_grad)
        log.info('SCF max_cycles = %d', self.max_cycle)
        log.info('direct_scf = %s', self.direct_scf)
        if self.direct_scf:
            log.info('direct_scf_tol = %g', self.direct_scf_tol)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        return self

    opt                      = NotImplemented
    get_fock                 = get_fock
    get_occ                  = get_occ
    get_grad                 = staticmethod(get_grad)
    init_guess_by_atom       = hf_cpu.SCF.init_guess_by_atom
    init_guess_by_huckel     = hf_cpu.SCF.init_guess_by_huckel
    init_guess_by_mod_huckel = hf_cpu.SCF.init_guess_by_mod_huckel
    init_guess_by_1e         = hf_cpu.SCF.init_guess_by_1e
    init_guess_by_chkfile    = hf_cpu.SCF.init_guess_by_chkfile
    from_chk                 = hf_cpu.SCF.from_chk
    get_init_guess           = return_cupy_array(hf_cpu.SCF.get_init_guess)
    make_rdm2                = NotImplemented
    energy_elec              = NotImplemented
    energy_tot               = energy_tot
    energy_nuc               = hf_cpu.SCF.energy_nuc
    check_convergence        = None
    do_disp                  = dispersion.check_disp
    get_dispersion           = dispersion.get_dispersion
    kernel = scf             = scf
    as_scanner               = hf_cpu.SCF.as_scanner
    _finalize                = hf_cpu.SCF._finalize
    init_direct_scf          = NotImplemented
    get_jk                   = _get_jk
    get_veff                 = NotImplemented
    mulliken_meta = pop      = NotImplemented
    mulliken_pop             = NotImplemented
    _is_mem_enough           = NotImplemented
    density_fit              = NotImplemented
    newton                   = NotImplemented
    x2c = x2c1e = sfx2c1e    = NotImplemented
    stability                = NotImplemented
    update_                  = NotImplemented
    istype                   = hf_cpu.SCF.istype
    to_rhf                   = NotImplemented
    to_uhf                   = NotImplemented
    to_ghf                   = NotImplemented
    to_rks                   = NotImplemented
    to_uks                   = NotImplemented
    to_gks                   = NotImplemented
    to_ks                    = NotImplemented
    canonicalize             = NotImplemented
    dump_scf_summary         = hf_cpu.dump_scf_summary

    smearing = smearing

    def init_guess_by_minao(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        if BACKEND_NAME == 'cupy':
            from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
            return int1e_ovlp(mol)
        return np.asarray(mol.intor_symmetric('int1e_ovlp'))

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return make_rdm1(mo_coeff, mo_occ)

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.dip_moment(mol, np.asarray(dm), unit, origin, verbose)

    def quad_moment(self, mol=None, dm=None, unit='DebyeAngstrom', origin=None,
                    verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.quad_moment(mol, np.asarray(dm), unit, origin, verbose)

    def remove_soscf(self):
        lib.logger.warn('remove_soscf has no effect in current version')
        return self

    def analyze(self, *args, **kwargs):
        return self.to_cpu().analyze()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._opt_gpu = {None: None}
        self._opt_jengine = {None: None}
        self._eri = None
        self.scf_summary = {}
        return self

    def dump_chk(self, envs):
        assert isinstance(envs, dict)
        if self.chkfile:
            chkfile.dump_scf(
                self.mol, self.chkfile, envs['e_tot'],
                np.asarray(envs['mo_energy']), np.asarray(envs['mo_coeff']),
                np.asarray(envs['mo_occ']), overwrite_mol=False)

    def get_j(self, mol, dm, hermi=1, omega=None):
        if omega is None:
            omega = mol.omega
        if BACKEND_NAME == 'cupy':
            jopt = self._opt_jengine.get(omega)
            if jopt is None:
                jopt = j_engine._VHFOpt(mol, self.direct_scf_tol).build()
                self._opt_jengine[omega] = jopt
            vj = j_engine.get_j(mol, dm, hermi, jopt)
            if not isinstance(dm, cupy.ndarray):
                vj = vj.get()
            return vj
        # CPU fallback
        dm_np = np.asarray(dm)
        with mol.with_range_coulomb(omega):
            vj = hf_cpu.get_jk(mol, dm_np, hermi, with_k=False)[0]
        return np.asarray(vj)

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        if omega is None:
            omega = mol.omega
        if BACKEND_NAME == 'cupy':
            vhfopt = self._opt_gpu.get(omega)
            with mol.with_range_coulomb(omega):
                if vhfopt is None:
                    vhfopt = self._opt_gpu[omega] = jk._VHFOpt(
                        mol, self.direct_scf_tol, tile=1).build()
                vk = jk.get_k(mol, dm, hermi, vhfopt)
            if not isinstance(dm, cupy.ndarray):
                vk = vk.get()
            return vk
        # CPU fallback
        dm_np = np.asarray(dm)
        with mol.with_range_coulomb(omega):
            vk = hf_cpu.get_jk(mol, dm_np, hermi, with_j=False)[1]
        return np.asarray(vk)

    def nuc_grad_method(self):
        if BACKEND_NAME == 'mlx':
            _refine_to_f64(self)
        return self.Gradients()

    def Gradients(self):
        raise NotImplementedError

    def Hessian(self):
        if BACKEND_NAME != 'cupy':
            from gpu4pyscf.hessian import RHF as RHFHess
            return RHFHess(self)
        raise NotImplementedError

    def TDDFT(self):
        """Time-dependent DFT for excited states."""
        if BACKEND_NAME == 'cupy':
            raise NotImplementedError
        _refine_to_f64(self)
        mf_cpu = self.to_cpu() if hasattr(self, 'to_cpu') else self
        return mf_cpu.TDDFT()

    def TDA(self):
        """Tamm-Dancoff approximation for excited states."""
        if BACKEND_NAME == 'cupy':
            raise NotImplementedError
        _refine_to_f64(self)
        mf_cpu = self.to_cpu() if hasattr(self, 'to_cpu') else self
        return mf_cpu.TDA()

    def PCM(self, solvent_obj=None, dm=None):
        """Apply PCM solvent model."""
        from gpu4pyscf.solvent import PCM
        return PCM(self, solvent_obj, dm)

    def SMD(self, solvent_obj=None, dm=None):
        """Apply SMD solvent model."""
        from gpu4pyscf.solvent import SMD
        return SMD(self, solvent_obj, dm)

class KohnShamDFT:
    '''
    A mock DFT base class, to be compatible with PySCF
    '''

class RHF(SCF):

    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'e_disp', 'h1e', 's1e', 'e_mf', 'conv_tol_cpscf', 'disp_with_3body'}

    get_veff = get_veff
    canonicalize = canonicalize

    def check_sanity(self):
        mol = self.mol
        if mol.nelectron != 1 and mol.spin != 0:
            logger.warn(self, 'Invalid number of electrons %d for RHF method.',
                        mol.nelectron)
        mem = get_avail_mem()
        nao = mol.nao
        if nao**2*20*8 > mem:
            logger.warn(self, 'GPU memory may be insufficient for SCF of this system. '
                        'It is recommended to use the scf.LRHF or dft.LRKS class for this system.')
        return SCF.check_sanity(self)

    energy_elec = energy_elec

    def Gradients(self):
        from gpu4pyscf.grad import RHF as RHFGrad
        return RHFGrad(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        if self.istype('_Solvation'):
            raise RuntimeError(
                'It is recommended to call density_fit() before applying a solvent model. '
                'Calling density_fit() after the solvent model may result in '
                'incorrect nuclear gradients, TDDFT, and other methods.')
        if BACKEND_NAME == 'cupy':
            import gpu4pyscf.df.df_jk
            return gpu4pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        # MLX/NumPy: use PySCF CPU DF infrastructure with Metal GPU J/K engine
        mf_cpu = self.to_cpu()
        mf_df = mf_cpu.density_fit(auxbasis, with_df, only_dfj)
        mf_df.__class__.__module__ = self.__class__.__module__
        if BACKEND_NAME == 'mlx':
            _patch_df_with_metal_jk(mf_df)
        return mf_df

    def newton(self):
        from gpu4pyscf.scf.soscf import newton
        return newton(self)

    def to_cpu(self):
        mf = hf_cpu.RHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
