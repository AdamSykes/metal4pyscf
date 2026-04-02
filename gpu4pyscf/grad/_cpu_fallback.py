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
CPU-fallback gradient implementations for non-CUDA backends.

Converts the gpu4pyscf SCF object to PySCF CPU, computes gradients using
PySCF's CPU gradient code, and returns the result. The SCF wavefunction
(MO coefficients, density matrix) computed on Metal GPU is reused —
only the integral derivatives run on CPU.
"""

import numpy as np
from pyscf import lib as pyscf_lib
from pyscf.grad import rhf as rhf_grad_cpu
from pyscf.grad import rks as rks_grad_cpu
from pyscf.grad import uhf as uhf_grad_cpu
from pyscf.grad import uks as uks_grad_cpu


class _GradientsMixin:
    """Common gradient functionality for CPU fallback."""

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        """Compute gradients with f64 refinement.

        The Metal SCF converges with f32 J/K (~1e-5 energy precision).
        Before computing gradients, we run a few f64 SCF cycles on CPU
        starting from the Metal-converged density matrix. This restores
        full f64 precision in the wavefunction, giving gradients accurate
        to ~1e-6 instead of ~1e-5.
        """
        # Get a CPU SCF object
        if hasattr(self.base, 'to_cpu'):
            mf_cpu = self.base.to_cpu()
        else:
            mf_cpu = self.base

        if mo_energy is not None:
            mf_cpu.mo_energy = np.asarray(mo_energy)
        if mo_coeff is not None:
            mf_cpu.mo_coeff = np.asarray(mo_coeff)
        if mo_occ is not None:
            mf_cpu.mo_occ = np.asarray(mo_occ)

        # --- f64 refinement ---
        # The Metal SCF may have converged with f32 J/K (conv_tol=1e-5).
        # Run a few f64 SCF cycles starting from the Metal density to
        # restore full precision before computing the gradient.
        dm0 = mf_cpu.make_rdm1()
        # Unpatch Metal J/K if present — restore original PySCF f64 J/K
        if hasattr(mf_cpu, 'with_df') and hasattr(mf_cpu.with_df, '_original_get_jk'):
            mf_cpu.with_df.get_jk = mf_cpu.with_df._original_get_jk
        mf_cpu.converged = False
        mf_cpu.mo_energy = None
        mf_cpu.mo_coeff = None
        mf_cpu.mo_occ = None
        mf_cpu.conv_tol = 1e-10
        mf_cpu.max_cycle = 10
        mf_cpu.kernel(dm0=dm0)

        # Create the CPU gradient object
        grad_cpu = self._make_cpu_grad(mf_cpu)

        if atmlst is not None:
            grad_cpu.atmlst = atmlst
        if hasattr(self, 'grid_response'):
            grad_cpu.grid_response = self.grid_response

        self.de = grad_cpu.kernel()
        return self.de

    grad = kernel

    def _make_cpu_grad(self, mf_cpu):
        raise NotImplementedError


class RHFGradients(_GradientsMixin, pyscf_lib.StreamObject):
    """RHF analytical gradients via PySCF CPU fallback."""

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.atmlst = None
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_grad(self, mf_cpu):
        return rhf_grad_cpu.Gradients(mf_cpu)


class RKSGradients(_GradientsMixin, pyscf_lib.StreamObject):
    """RKS analytical gradients via PySCF CPU fallback."""

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.atmlst = None
        self.grid_response = False
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_grad(self, mf_cpu):
        return rks_grad_cpu.Gradients(mf_cpu)


class UHFGradients(_GradientsMixin, pyscf_lib.StreamObject):
    """UHF analytical gradients via PySCF CPU fallback."""

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.atmlst = None
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_grad(self, mf_cpu):
        return uhf_grad_cpu.Gradients(mf_cpu)


class UKSGradients(_GradientsMixin, pyscf_lib.StreamObject):
    """UKS analytical gradients via PySCF CPU fallback."""

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.de = None
        self.atmlst = None
        self.grid_response = False
        self.verbose = mf.verbose
        self.stdout = mf.stdout

    def _make_cpu_grad(self, mf_cpu):
        return uks_grad_cpu.Gradients(mf_cpu)
