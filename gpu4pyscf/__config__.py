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

from gpu4pyscf.lib.backends import BACKEND_NAME
from gpu4pyscf.lib.backend import device, memory

num_devices = device.device_count()

_device_props = device.get_device_properties(0) if num_devices > 0 else {}

GB = 1024*1024*1024
min_ao_blksize = 256        # maximum batch size of AOs
min_grid_blksize = 64*64    # maximum batch size of grids for DFT
ao_aligned = 32             # global AO alignment for slicing
grid_aligned = 256          # 256 alignment for grids globally

total_mem = _device_props.get('total_memory', 0)

# Use smaller blksize for old gaming GPUs / small-memory devices
if total_mem < 16 * GB:
    min_ao_blksize = 64
    min_grid_blksize = 64*64

# Use 90% of the global memory for the memory pool
mem_fraction = 0.9
memory.set_pool_limit(fraction=mem_fraction)

# Shared memory per block (CUDA) or threadgroup memory (Metal)
shm_per_block_optin = _device_props.get('shared_memory_per_block_optin', 0)
shm_per_block = _device_props.get('shared_memory_per_block', 0)
if shm_per_block_optin > 65536:
    shm_size = shm_per_block_optin
else:
    shm_size = shm_per_block

# Provide a 'props' dict that matches the old cupy format for downstream code
# that imports `from gpu4pyscf.__config__ import props`
props = {
    'totalGlobalMem': total_mem,
    'sharedMemPerBlock': shm_per_block,
    'sharedMemPerBlockOptin': shm_per_block_optin,
    'name': _device_props.get('name', 'Unknown'),
}

# Check P2P data transfer is available
_p2p_access = True
if num_devices > 1:
    for src in range(num_devices):
        for dst in range(num_devices):
            if src != dst:
                can_access_peer = device.can_access_peer(src, dst)
                _p2p_access &= can_access_peer

# Overwrite the above settings using the global pyscf configs
from pyscf.__config__ import * # noqa
