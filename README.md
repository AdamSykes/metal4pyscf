# gpu4pyscf-metal

GPU-accelerated quantum chemistry on Apple Silicon, via Metal.

A fork of [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) that replaces NVIDIA CUDA/CuPy with Apple Silicon Metal/MLX, enabling GPU-accelerated DFT calculations on Mac.

## Installation

Requires macOS on Apple Silicon (M1 or later), Python 3.10+.

```bash
pip install -e .
```

Dependencies (installed automatically): `pyscf`, `mlx`, `numpy`, `scipy`, `h5py`, `geometric`.

## Quick Start

```python
import pyscf
from gpu4pyscf.dft import RKS

mol = pyscf.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='def2-tzvpp')

mf = RKS(mol, xc='B3LYP').density_fit()
mf.kernel()
print(f'Energy: {mf.e_tot:.10f} Ha')
```

## Features

All standard PySCF quantum chemistry methods are accessible:

| Feature | Metal GPU accelerated | CPU fallback |
|---------|----------------------|-------------|
| RHF / UHF | Energy (DF-J/K on GPU) | ROHF, GHF |
| RKS / UKS (any XC functional) | Energy (DF-J/K + XC on GPU) | GKS, ROKS |
| Density fitting | J/K contractions (6-12x/cycle) | Integral precomputation |
| Analytical gradients | — | Via PySCF (f64 refinement) |
| Geometry optimization | — | Via geomeTRIC / berny |
| Analytical Hessians | — | Via PySCF (f64 refinement) |
| Frequency analysis | — | Via PySCF |
| TDDFT / TDA | — | Via PySCF (f64 refinement) |
| PCM / IEF-PCM / SS(V)PE | — | Via PySCF |
| MP2 | — | Via PySCF |
| CCSD | — | Via PySCF |
| PBC (periodic) | — | Via PySCF |
| QM/MM | — | Via PySCF |

## Performance

Benchmarked on Apple M4 Pro (48 GB), DF-B3LYP, cached DF tensors:

| System | AOs | PySCF CPU | Metal GPU | Speedup |
|--------|-----|-----------|-----------|---------|
| H₂O / def2-TZVPP | 59 | 0.5s | 0.2s | **3.9x** |
| Caffeine / def2-SVP | 246 | 14s | 10s | **1.4x** |
| Caffeine / def2-TZVPP | 574 | 35s | 24s | **1.5x** |

Per-component speedups:
- DF J/K contractions: **6-12x** per SCF cycle
- XC grid integration: **3.6x** (Metal eval_ao + GPU contraction)
- DF tensor build: **4.4x** (GPU unpack_tril)

## How It Works

The fork introduces a backend abstraction layer (`gpu4pyscf.lib.backend`) that auto-detects CuPy (NVIDIA), MLX (Apple Silicon), or NumPy (CPU fallback).

On Apple Silicon:
- **DF J/K contractions** run on Metal GPU in float32 via MLX, with float64 accumulation on CPU
- **XC grid integration** uses a custom Metal compute shader (`eval_ao`) for basis function evaluation, with GPU matmul for density/potential contractions
- **Eigensolve, DIIS, energy** run on CPU in float64 via Accelerate/LAPACK
- **Gradients, Hessians, TDDFT** use f64 refinement: the Metal-converged density is refined with a few CPU f64 SCF cycles before computing properties

Apple Silicon GPUs lack float64 hardware. The mixed-precision strategy (f32 GPU + f64 CPU accumulation) follows [Codina et al., JCTC 2025](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01800).

## Testing

```bash
# Run all tests
pytest

# Backend unit + chemistry tests (116 tests)
pytest gpu4pyscf/lib/backends/tests/ -v

# Smoke tests for all features
pytest tests/ -v
```

## Documentation

See [METAL_PORT.md](METAL_PORT.md) for the full technical documentation, including:
- Architecture and file manifest
- Metal kernel implementations
- Mixed-precision strategy and prior art
- CUDA-to-Metal translation notes
- Benchmark details

## License

Apache 2.0 (same as gpu4pyscf and PySCF).

## Acknowledgments

- [PySCF](https://pyscf.org) — the quantum chemistry engine
- [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) — the original CUDA GPU acceleration layer
- [MLX](https://github.com/ml-explore/mlx) — Apple's GPU array framework
- [metal-float64](https://github.com/philipturner/metal-float64) — research on emulated double precision on Apple GPUs
