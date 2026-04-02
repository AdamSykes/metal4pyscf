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

import os
import numpy
from gpu4pyscf.lib import diis
from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from gpu4pyscf.lib import cupy_helper
    from gpu4pyscf.lib import cutensor
else:
    # On non-CUDA backends, cupy_helper and cutensor are not available.
    # Provide a lazy-import fallback so that code importing these names
    # gets a clear error rather than crashing at package init.
    import importlib as _importlib

    class _LazyModule:
        """Proxy that raises ImportError on attribute access."""
        def __init__(self, name, backend):
            self._name = name
            self._backend = backend
        def __getattr__(self, attr):
            raise ImportError(
                f'gpu4pyscf.lib.{self._name} requires CuPy (CUDA). '
                f'Current backend: {self._backend}')

    cupy_helper = _LazyModule('cupy_helper', BACKEND_NAME)
    cutensor = _LazyModule('cutensor', BACKEND_NAME)

from gpu4pyscf.lib import utils

from pyscf import lib

# Only override format_sys_info if it doesn't need CUDA runtime info
if BACKEND_NAME == 'cupy':
    lib.misc.format_sys_info = utils.format_sys_info
