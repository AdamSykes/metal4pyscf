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

__version__ = '1.6.1'

from . import _patch_pyscf

from gpu4pyscf.lib.backends import BACKEND_NAME

from . import lib

# Backend-specific memory pool setup
if BACKEND_NAME == 'cupy':
    # Overwrite the cupy memory allocator. Make memory pool manage small-sized
    # arrays only.
    lib.cupy_helper.set_conditional_mempool_malloc()

# Import submodules. On non-CUDA backends some may fail due to unported
# cupy dependencies — import them lazily so the package itself stays usable.
import importlib as _importlib

def __getattr__(name):
    _submodules = {
        'grad', 'hessian', 'solvent', 'scf', 'dft', 'tdscf', 'nac',
        'df', 'mp', 'cc', 'qmmm', 'pbc', 'properties', 'tools', 'md',
    }
    if name in _submodules:
        try:
            return _importlib.import_module(f'.{name}', __name__)
        except ImportError as e:
            raise ImportError(
                f'gpu4pyscf.{name} is not available with the {BACKEND_NAME} '
                f'backend. The module requires components that have not been '
                f'ported yet. Original error: {e}'
            ) from e
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
