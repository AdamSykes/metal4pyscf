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
CPU-fallback Hessian implementations for non-CUDA backends.

Delegates to PySCF CPU Hessian code after f64 refinement of the
Metal-converged wavefunction.
"""

import numpy as np
from pyscf import lib as pyscf_lib
from pyscf.hessian import rhf as rhf_hess_cpu
from pyscf.hessian import rks as rks_hess_cpu
from pyscf.hessian import uhf as uhf_hess_cpu
from pyscf.hessian import uks as uks_hess_cpu


class _HessianMixin:
    """Common Hessian functionality for CPU fallback."""

    def kernel(self):
        """Compute Hessian with f64 refinement."""
        from gpu4pyscf.scf.hf import _refine_to_f64
        _refine_to_f64(self.base)

        # Build CPU Hessian from refined mf
        hess_cpu = self._make_cpu_hess(self.base)

        if hasattr(self, 'auxbasis_response'):
            hess_cpu.auxbasis_response = self.auxbasis_response

        self.de = hess_cpu.kernel()
        return self.de

    hess = kernel

    def _make_cpu_hess(self, mf_cpu):
        raise NotImplementedError


class RHFHessian(_HessianMixin, pyscf_lib.StreamObject):
    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.auxbasis_response = 2
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_hess(self, mf_cpu):
        return rhf_hess_cpu.Hessian(mf_cpu)


class RKSHessian(_HessianMixin, pyscf_lib.StreamObject):
    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.auxbasis_response = 2
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_hess(self, mf_cpu):
        return rks_hess_cpu.Hessian(mf_cpu)


class UHFHessian(_HessianMixin, pyscf_lib.StreamObject):
    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.auxbasis_response = 2
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_hess(self, mf_cpu):
        return uhf_hess_cpu.Hessian(mf_cpu)


class UKSHessian(_HessianMixin, pyscf_lib.StreamObject):
    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.auxbasis_response = 2
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_hess(self, mf_cpu):
        return uks_hess_cpu.Hessian(mf_cpu)
