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

# modified by Xiaojie Wu (wxj6000@gmail.com)

import numpy as np
from pyscf.dft import rks
from pyscf import dft as pyscf_dft
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf
from gpu4pyscf.lib.backend import BACKEND_NAME
from pyscf import __config__

if BACKEND_NAME == 'cupy':
    import cupy
    from gpu4pyscf.dft import numint, gen_grid
    from gpu4pyscf.scf import j_engine
    from gpu4pyscf.lib.cupy_helper import tag_array, asarray
else:
    from pyscf.dft import numint as numint_cpu, gen_grid as gen_grid_cpu
    from gpu4pyscf.scf.hf import tag_array, asarray

__all__ = [
    'get_veff', 'RKS', 'KohnShamDFT',
]


def _initialize_grids_cpu(ks, mol, dm):
    """Build grids using PySCF CPU grid generation."""
    if ks.grids.coords is None:
        t0 = logger.init_timer(ks)
        ks.grids.build()
        t0 = logger.timer_debug1(ks, 'setting up grids', *t0)


if BACKEND_NAME == 'cupy':
    def prune_small_rho_grids_(ks, mol, dm, grids):
        '''Prune grids if the electron density on the grid is small'''
        threshold = ks.small_rho_cutoff
        rho = ks._numint.get_rho(mol, dm, grids, ks.max_memory, verbose=ks.verbose)
        return grids.prune_by_density_(rho, threshold)

    def initialize_grids(ks, mol=None, dm=None):
        if mol is None: mol = ks.mol
        if ks.grids.coords is None:
            t0 = logger.init_timer(ks)
            ks.grids.build()
            ks.grids.weights = asarray(ks.grids.weights)
            ks.grids.coords = asarray(ks.grids.coords)
            ground_state = getattr(dm, 'ndim', 0) == 2
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.grids = prune_small_rho_grids_(ks, ks.mol, dm, ks.grids)
            t0 = logger.timer_debug1(ks, 'setting up grids', *t0)

        if ks.do_nlc() and ks.nlcgrids.coords is None:
            t0 = logger.init_timer(ks)
            ks.nlcgrids.build()
            ks.nlcgrids.weights = asarray(ks.nlcgrids.weights)
            ks.nlcgrids.coords = asarray(ks.nlcgrids.coords)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.nlcgrids = prune_small_rho_grids_(ks, ks.mol, dm, ks.nlcgrids)
            t0 = logger.timer_debug1(ks, 'setting up nlc grids', *t0)
        return ks

    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functionals (GPU path)'''
        if mol is None: mol = ks.mol
        if dm is None: dm = ks.make_rdm1()
        t0 = logger.init_timer(ks)
        initialize_grids(ks, mol, dm)

        ni = ks._numint
        if hermi == 2:
            n, exc, vxc = 0, 0, 0
        else:
            n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm)
            if ks.do_nlc():
                if ni.libxc.is_nlc(ks.xc):
                    xc = ks.xc
                else:
                    assert ni.libxc.is_nlc(ks.nlc)
                    xc = ks.nlc
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm)
                exc += enlc
                vxc += vnlc
            logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

        dm_orig = dm
        vj_last = getattr(vhf_last, 'vj', None)
        if vj_last is not None:
            dm = asarray(dm) - asarray(dm_last)
        vj = ks.get_j(mol, dm, hermi)
        if vj_last is not None:
            vj += asarray(vj_last)
        vxc += vj
        ecoul = float(cupy.einsum('ij,ij', dm_orig, vj).real) * .5

        vk = None
        if ni.libxc.is_hybrid_xc(ks.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
            if omega == 0:
                vk = ks.get_k(mol, dm, hermi)
                vk *= hyb
            elif alpha == 0:
                vk = ks.get_k(mol, dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:
                vk = ks.get_k(mol, dm, hermi, omega=omega)
                vk *= alpha
            else:
                vk = ks.get_k(mol, dm, hermi)
                vk *= hyb
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vk *= .5
            if vj_last is not None:
                vk += asarray(vhf_last.vk)
            vxc -= vk
            exc -= float(cupy.einsum('ij,ij', dm_orig, vk).real) * .5
        t0 = logger.timer(ks, 'veff', *t0)
        vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    def energy_elec(ks, dm=None, h1e=None, vhf=None):
        if dm is None: dm = ks.make_rdm1()
        if h1e is None: h1e = ks.get_hcore()
        if vhf is None: vhf = ks.get_veff(ks.mol, dm)
        e1 = cupy.einsum('ij,ji->', h1e, dm).get()[()].real
        ecoul = vhf.ecoul.real
        exc = vhf.exc.real
        if isinstance(ecoul, cupy.ndarray):
            ecoul = ecoul.get()[()]
        if isinstance(exc, cupy.ndarray):
            exc = exc.get()[()]
        e2 = ecoul + exc
        ks.scf_summary['e1'] = e1
        ks.scf_summary['coul'] = ecoul
        ks.scf_summary['exc'] = exc
        logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
        return e1+e2, e2

else:
    # ---------------------------------------------------------------------------
    # Non-CuPy backend: delegate XC evaluation to PySCF CPU
    # ---------------------------------------------------------------------------

    def initialize_grids(ks, mol=None, dm=None):
        if mol is None: mol = ks.mol
        _initialize_grids_cpu(ks, mol, dm)
        return ks

    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functionals (Metal GPU accelerated path).

        Uses Metal GPU for Vxc contraction via numint_metal.
        '''
        if mol is None: mol = ks.mol
        if dm is None: dm = ks.make_rdm1()
        t0 = logger.init_timer(ks)
        initialize_grids(ks, mol, dm)

        dm_np = np.asarray(dm)
        ni = ks._numint

        if hermi == 2:
            n, exc, vxc = 0, 0, np.zeros_like(dm_np)
        else:
            if BACKEND_NAME == 'mlx':
                try:
                    import importlib.util as _ilu, os as _os
                    _spec = _ilu.spec_from_file_location(
                        'numint_metal',
                        _os.path.join(_os.path.dirname(__file__), 'numint_metal.py'))
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    n, exc, vxc = _mod.nr_rks_metal(ni, mol, ks.grids, ks.xc, dm_np)
                except Exception:
                    n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm_np)
            else:
                n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm_np)
            if ks.do_nlc():
                if ni.libxc.is_nlc(ks.xc):
                    xc = ks.xc
                else:
                    assert ni.libxc.is_nlc(ks.nlc)
                    xc = ks.nlc
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm_np)
                exc += enlc
                vxc += vnlc
            logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

        dm_orig = dm_np
        vj_last = getattr(vhf_last, 'vj', None)
        if vj_last is not None:
            dm_np = np.asarray(dm) - np.asarray(dm_last)
        vj = ks.get_j(mol, dm_np, hermi)
        if vj_last is not None:
            vj = np.asarray(vj) + np.asarray(vj_last)
        vxc = np.asarray(vxc) + np.asarray(vj)
        ecoul = float(np.einsum('ij,ij', dm_orig, vj).real) * .5

        vk = None
        if ni.libxc.is_hybrid_xc(ks.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
            if omega == 0:
                vk = ks.get_k(mol, dm_np, hermi)
                vk = np.asarray(vk) * hyb
            elif alpha == 0:
                vk = ks.get_k(mol, dm_np, hermi, omega=-omega)
                vk = np.asarray(vk) * hyb
            elif hyb == 0:
                vk = ks.get_k(mol, dm_np, hermi, omega=omega)
                vk = np.asarray(vk) * alpha
            else:
                vk = ks.get_k(mol, dm_np, hermi)
                vk = np.asarray(vk) * hyb
                vklr = ks.get_k(mol, dm_np, hermi, omega=omega)
                vk += np.asarray(vklr) * (alpha - hyb)
            vk *= .5
            if vj_last is not None:
                vk = np.asarray(vk) + np.asarray(vhf_last.vk)
            vxc = vxc - vk
            exc -= float(np.einsum('ij,ij', dm_orig, vk).real) * .5
        t0 = logger.timer(ks, 'veff', *t0)
        vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    def energy_elec(ks, dm=None, h1e=None, vhf=None):
        if dm is None: dm = ks.make_rdm1()
        if h1e is None: h1e = ks.get_hcore()
        if vhf is None: vhf = ks.get_veff(ks.mol, dm)
        e1 = float(np.einsum('ij,ji->', h1e, dm).real)
        ecoul = float(vhf.ecoul)
        exc = float(vhf.exc)
        e2 = ecoul + exc
        ks.scf_summary['e1'] = e1
        ks.scf_summary['coul'] = ecoul
        ks.scf_summary['exc'] = exc
        logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
        return e1+e2, e2


# ---------------------------------------------------------------------------
# KohnShamDFT mixin and RKS class
# ---------------------------------------------------------------------------

class KohnShamDFT(rks.KohnShamDFT):

    _keys = {'cphf_grids', *rks.KohnShamDFT._keys}

    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_hf  = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented

    small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 0)

    def __init__(self, xc='LDA,VWN'):
        self.xc = xc
        self.disp = None
        self.disp_with_3body = None
        self.nlc = ''

        if BACKEND_NAME == 'cupy':
            self.grids = gen_grid.Grids(self.mol)
        else:
            self.grids = pyscf_dft.gen_grid.Grids(self.mol)

        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)

        if BACKEND_NAME == 'cupy':
            self.nlcgrids = gen_grid.Grids(self.mol)
            self.cphf_grids = gen_grid.Grids(self.mol)
        else:
            self.nlcgrids = pyscf_dft.gen_grid.Grids(self.mol)
            self.cphf_grids = pyscf_dft.gen_grid.Grids(self.mol)

        self.nlcgrids.level = getattr(
            __config__, 'dft_rks_RKS_nlcgrids_level', self.nlcgrids.level)
        self.cphf_grids.prune = pyscf_dft.gen_grid.sg1_prune
        self.cphf_grids.atom_grid = (50,194)

        if BACKEND_NAME == 'cupy':
            self._numint = numint.NumInt()
        else:
            self._numint = pyscf_dft.numint.NumInt()

    @property
    def omega(self):
        return self._numint.omega
    @omega.setter
    def omega(self, v):
        self._numint.omega = float(v)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('XC library %s version %s\n    %s',
                 self._numint.libxc.__name__,
                 self._numint.libxc.__version__,
                 self._numint.libxc.__reference__)
        log.info('XC functionals = %s', self.xc)
        self.grids.dump_flags(verbose)

        if self.do_nlc():
            log.info('** Following is NLC and NLC Grids **')
            if self.nlc:
                log.info('NLC functional = %s', self.nlc)
            else:
                log.info('NLC functional = %s', self.xc)
            self.nlcgrids.dump_flags(verbose)

        log.info('small_rho_cutoff = %g', self.small_rho_cutoff)
        return self

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        self._numint.reset()
        if hasattr(self, 'cphf_grids'):
            self.cphf_grids.reset(self.mol)
        else:
            if BACKEND_NAME == 'cupy':
                cphf_grids = self.cphf_grids = gen_grid.Grids(self.mol)
            else:
                cphf_grids = self.cphf_grids = pyscf_dft.gen_grid.Grids(self.mol)
            cphf_grids.prune = pyscf_dft.gen_grid.sg1_prune
            cphf_grids.atom_grid = (50,194)
        return self

    do_nlc = rks.KohnShamDFT.do_nlc

hf.KohnShamDFT = KohnShamDFT
from gpu4pyscf.lib import utils

class RKS(KohnShamDFT, hf.RHF):

    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mol, xc='LDA,VWN'):
        hf.RHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf.RHF.dump_flags(self, verbose)
        return KohnShamDFT.dump_flags(self, verbose)

    def Gradients(self):
        from gpu4pyscf.grad import RKS as RKSGrad
        return RKSGrad(self)

    def to_cpu(self):
        mf = rks.RKS(self.mol)
        utils.to_cpu(self, out=mf)
        return mf

    energy_elec = energy_elec
    energy_tot = hf.RHF.energy_tot
    get_veff = get_veff
    to_hf = NotImplemented
    init_guess_by_vsap = rks.init_guess_by_vsap
