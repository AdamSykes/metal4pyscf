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
| RKS (any XC functional) | Energy (DF-J/K + XC on GPU) | — |
| UKS (any XC functional) | Energy (DF-J/K + XC on GPU, batched α/β) | ROKS, GKS |
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

End-to-end DF-B3LYP SCF on Apple M4 Pro (48 GB), warm GPU, cached CDERI:

**RKS (closed-shell):**

| System | AOs | PySCF CPU | Metal GPU | Speedup |
|--------|-----|-----------|-----------|---------|
| H₂O / def2-TZVP | 34 | 0.4s | 0.1s | **3.7x** |
| CH₄ / def2-TZVPP | 84 | 1.1s | 0.4s | **2.8x** |
| Benzene / def2-SVP | 114 | 2.9s | 0.9s | **3.2x** |
| Benzene / def2-TZVP | 228 | 4.6s | 1.9s | **2.5x** |
| Naphthalene / def2-TZVP | 358 | 12.4s | 4.5s | **2.8x** |
| Aspirin / def2-TZVP | 451 | 21.6s | 6.7s | **3.2x** |
| Anthracene / def2-TZVP | 494 | 18.7s | 7.7s | **2.4x** |
| Caffeine / def2-TZVPP | 557 | 40.3s | 22.2s | **1.8x** |

**UKS (open-shell):**

| System | PySCF CPU | Metal GPU | Speedup |
|--------|-----------|-----------|---------|
| NO₂ radical / def2-TZVP | 1.4s | 0.5s | **3.0x** |
| Phenyl radical / def2-TZVPP | 9.3s | 2.3s | **4.0x** |
| Phenyl radical / def2-TZVP | 10.7s | 1.6s | **6.5x** |

Energies match PySCF CPU reference to ≤1e-5 Ha across all systems (well within chemical accuracy).

**Per-component speedups:**
- DF J/K contractions: **6-12x** per SCF cycle (batched over spins for UKS)
- XC grid integration: **3.2x** (Metal `eval_ao` + fused rho/Vxc kernels)
- DF tensor build: **4.4x** (GPU `unpack_tril`)

**Analytical gradients and Hessians** give correct results (f64-refined, 1e-7 to 1e-6 accuracy) but do not show end-to-end speedups — the SCF savings are consumed by the CPU-bound gradient/Hessian code and the f64 refinement overhead needed for tight accuracy.

## How It Works

The fork introduces a backend abstraction layer (`gpu4pyscf.lib.backend`) that auto-detects CuPy (NVIDIA), MLX (Apple Silicon), or NumPy (CPU fallback).

On Apple Silicon:
- **DF J/K contractions** run on Metal GPU in float32 via MLX, with float64 accumulation on CPU. For UKS, alpha and beta densities are batched into single GPU calls (one half-transform, one matmul-pair for J).
- **XC grid integration** uses a custom Metal compute shader (`eval_ao`) for basis function evaluation, with GPU contraction of `Vxc` and on-GPU accumulation of `vmat`. Grid coords, weights, and shell data are pre-uploaded once per SCF cycle.
- **Eigensolve, DIIS, energy** run on CPU in float64 via Accelerate/LAPACK
- **Gradients, Hessians, TDDFT** use f64 refinement: the Metal-converged density is refined with a few CPU f64 SCF cycles before computing properties

Apple Silicon GPUs lack float64 hardware. The mixed-precision strategy (f32 GPU + f64 CPU accumulation) follows [Codina et al., JCTC 2025](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01800).

**Convergence:** Metal SCF uses `conv_tol = 1e-4` by default (instead of 1e-9) because f32 J/K noise gives a gradient-norm floor of ~1e-2 for larger molecules — tighter targets are unreachable. Energy is variational in the density, so f32-converged densities give energies accurate to ≤1e-5 Ha regardless, well within chemical accuracy. A final f64 refinement step is applied automatically before property calculations.

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
