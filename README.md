# gpu4pyscf-metal

GPU-accelerated quantum chemistry on Apple Silicon, via Metal.

A fork of [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) that replaces NVIDIA CUDA/CuPy with Apple Silicon Metal/MLX, enabling GPU-accelerated DFT calculations on Mac.

## Installation

Requires macOS on Apple Silicon (M1 or later), Python 3.10+.

```bash
pip install -e .
```

Dependencies (installed automatically): `pyscf`, `mlx`, `numpy`, `scipy`, `numba`, `h5py`, `geometric`.

## Quick Start

**Single-point energy:**

```python
import pyscf
from gpu4pyscf.dft import RKS

mol = pyscf.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='def2-tzvpp')

mf = RKS(mol, xc='B3LYP').density_fit()
mf.kernel()  # Metal-accelerated SCF
```

**Geometry optimization:**

```python
from pyscf.geomopt.geometric_solver import optimize

mol_eq = optimize(mf)  # Metal-accelerated SCF + gradient per step
```

**Open-shell (UKS):**

```python
from gpu4pyscf.dft import UKS

mol = pyscf.M(atom='O 0 0 0; H 0 0 0.97', spin=1, basis='def2-svp')
mf = UKS(mol, xc='B3LYP').density_fit()
mf.kernel()
g = mf.nuc_grad_method().kernel()  # Metal-accelerated gradient
```

Metal acceleration activates automatically via `density_fit()`. No code changes needed compared to PySCF — just import from `gpu4pyscf` instead of `pyscf`.

## Features

| Feature | Metal GPU accelerated | CPU fallback |
|---------|----------------------|-------------|
| RHF / UHF | Energy (DF-J/K on GPU) | ROHF, GHF |
| RKS / UKS (any XC) | Energy (DF-J/K + XC on GPU) | ROKS, GKS |
| Density fitting | J/K contractions (6-12x/cycle) | Integral precomputation |
| Analytical gradients | XC gradient (4.4x), DF cho_solve (13x) | int3c2e_ip1 (CPU libcint) |
| Geometry optimization | 1.8x per step (Metal SCF + gradient) | — |
| Analytical Hessians | — | Via PySCF (f64 refinement) |
| Frequency analysis | — | Via PySCF |
| TDDFT / TDA | — | Via PySCF (f64 refinement) |
| PCM / IEF-PCM / SS(V)PE | — | Via PySCF |
| MP2, CCSD, PBC, QM/MM | — | Via PySCF |

## Performance

Benchmarks on Apple M4 Pro (48 GB), warm GPU, cached CDERI.

**DF-B3LYP SCF energy:**

| System | AOs | PySCF CPU | Metal GPU | Speedup |
|--------|-----|-----------|-----------|---------|
| H2O / def2-TZVP | 34 | 0.4s | 0.1s | **3.7x** |
| Benzene / def2-SVP | 114 | 2.9s | 0.9s | **3.2x** |
| Benzene / def2-TZVP | 228 | 4.6s | 1.9s | **2.5x** |
| Caffeine / def2-SVP | 246 | 11.4s | 4.5s | **2.5x** |
| Aspirin / def2-TZVP | 451 | 21.6s | 6.7s | **3.2x** |

**DF-B3LYP single-point + gradient (caffeine/def2-SVP):**

| Component | CPU | Metal | Speedup |
|-----------|-----|-------|---------|
| SCF energy | 11.4s | 4.5s | **2.5x** |
| XC gradient (`get_vxc`) | 0.82s | 0.21s | **4.4x** |
| DF Cholesky solve | 0.51s | 0.03s | **13x** |
| int3c2e_ip1 integrals | 3.1s | 3.1s | 1.0x (CPU) |
| **Total gradient** | **9.0s** | **7.9s** | **1.14x** |
| **Total SCF + gradient** | **20.4s** | **12.4s** | **1.65x** |

**Geometry optimization per step (caffeine/def2-SVP):**

| Component | CPU | Metal | Speedup |
|-----------|-----|-------|---------|
| SCF | ~11s | ~3.2s | **3.4x** |
| Gradient | ~9s | ~7.7s | **1.17x** |
| **Per step** | **~20s** | **~10.9s** | **1.84x** |

Energies match PySCF CPU reference to ≤1e-5 Ha. Gradients match to ≤1e-5 (norm).

## How It Works

The fork introduces a backend abstraction layer (`gpu4pyscf.lib.backend`) that auto-detects CuPy (NVIDIA), MLX (Apple Silicon), or NumPy (CPU fallback).

On Apple Silicon:
- **DF J/K contractions** run on Metal GPU in float32 via MLX, with float64 accumulation on CPU. For UKS, alpha and beta densities are batched into single GPU calls.
- **XC grid integration** uses custom Metal compute shaders (`eval_ao` for basis functions up to deriv=2, fused `rho`/`Vxc` contraction). Grid data stays on GPU across batches.
- **XC gradient** (`get_vxc`) uses Metal `eval_ao` deriv=2 + on-GPU `_gga_grad_sum_` contraction for both RKS and UKS. 4.4x faster than CPU.
- **DF gradient Cholesky solve** reuses cached CDERI from the SCF via `L^{-T} @ (cderi @ D)` on GPU. 13x faster.
- **Geometry optimization** uses `as_scanner()` with automatic refinement skipping on intermediate steps (f32 gradient error ~4e-4, within 4.5e-4 convergence threshold).
- **Eigensolve, DIIS, energy** run on CPU in float64 via Accelerate/LAPACK.
- **f64 refinement**: 3 f64 SCF cycles from the f32 density, applied before gradient/Hessian for precision. Metal get_jk is restored afterward for the next optimization step.

Apple Silicon GPUs lack float64 hardware. The mixed-precision strategy (f32 GPU + f64 CPU accumulation) follows [Codina et al., JCTC 2025](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01800).

## Testing

```bash
pytest tests/ -v       # 30 tests covering SCF, DFT, gradients, geomopt, Hessian
```

## Known Limitations

- **int3c2e_ip1 integrals** remain on CPU (f32 TRR precision gives ~0.1 gradient error; needs f64 GPU compute that Apple Silicon doesn't expose). A complete Metal kernel exists but is disabled.
- **Hessian precision**: ~3% relative error from f32 CDERI in the SCF density. Gradients are not affected (different code path).
- **MGGA functionals**: Metal XC gradient supports LDA and GGA only. MGGA (e.g. TPSS, SCAN) falls through to CPU.
- **Large molecules**: CDERI disk cache grows with basis size (~300 MB for caffeine/def2-TZVPP). Stored in `~/.cache/gpu4pyscf/`.

## License

Apache 2.0 (same as gpu4pyscf and PySCF).

## Acknowledgments

- [PySCF](https://pyscf.org) — the quantum chemistry engine
- [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) — the original CUDA GPU acceleration layer
- [MLX](https://github.com/ml-explore/mlx) — Apple's GPU array framework
