# gpu4pyscf — Apple Silicon Metal Port

This document tracks the porting of gpu4pyscf from NVIDIA CUDA/CuPy to Apple Silicon Metal via MLX.

## Architecture

### Backend Abstraction Layer

The original codebase has ~319 files importing CuPy and 156 CUDA kernel files. Rather than rewriting everything at once, we introduced a backend abstraction layer that lets the package run on CuPy (NVIDIA), MLX (Apple Silicon), or NumPy (CPU fallback).

```
gpu4pyscf/lib/backend.py          — Public API: xp, device, memory, stream
gpu4pyscf/lib/backends/
    __init__.py                    — Auto-detection: CuPy > MLX > NumPy
    cupy_backend.py                — CuPy/CUDA wrapper (original behavior)
    mlx_backend.py                 — MLX/Metal wrapper (Apple Silicon)
    numpy_backend.py               — CPU-only fallback
```

**Usage:**

```python
from gpu4pyscf.lib.backend import xp, device, memory, stream

a = xp.zeros((n, n))          # backend array (cupy, mlx.core, or numpy)
device.synchronize()           # flush GPU commands
free, total = memory.get_mem_info()
```

**Selection:** Automatic, or force via `GPU4PYSCF_BACKEND=mlx` (or `cupy`, `numpy`).

### Unified API Surface

Every backend exports the same interface:

| Component | CuPy | MLX | NumPy |
|-----------|------|-----|-------|
| `xp` | `cupy` | `mlx.core` | `numpy` |
| `device_manager` | CUDA runtime | Apple Metal (1 device, unified memory) | no-op |
| `memory_manager` | CuPy memory pool | `mx.get_active_memory()` | no-op |
| `stream_manager` | CUDA streams | lazy eval + `mx.eval()` | no-op |
| `eigh`, `solve`, `svd`, `inv` | cupy.linalg | scipy (float64) / MLX CPU stream (float32) | numpy.linalg |
| `to_host` / `to_device` | `.get()` / `cupy.asarray()` | `np.array()` / `mx.array()` | identity |
| `compile_kernel` | `cupy.RawKernel` | `mx.fast.metal_kernel` | N/A |

### MLX Float64 Limitation

MLX silently casts float64 to float32. Quantum chemistry requires float64. The MLX backend detects dtype and routes float64 operations through SciPy (running on CPU via Accelerate framework). Float32 operations use native MLX with `stream=mx.cpu` for linalg ops not yet supported on Metal GPU.

### CPU Fallback Strategy

On non-CuPy backends, CUDA-specific operations fall back to PySCF CPU:

| Operation | CUDA Path | Fallback Path |
|-----------|-----------|---------------|
| J/K matrices | `libgvhf_rys` CUDA kernels | `pyscf.scf.hf.get_jk()` |
| Core Hamiltonian | GPU integral engine | `pyscf.scf.hf.get_hcore()` |
| Overlap matrix | GPU `int1e_ovlp` | `mol.intor_symmetric('int1e_ovlp')` |
| XC evaluation | GPU `numint.nr_rks()` | `pyscf.dft.numint.NumInt().nr_rks()` |
| Density fitting | `gpu4pyscf.df.df_jk` | `pyscf.df.density_fit()` |
| Initial guess | GPU MINAO with CuPy arrays | `pyscf.scf.hf.init_guess_by_minao()` |
| DIIS | `gpu4pyscf.scf.diis.CDIIS` | `pyscf.scf.diis.CDIIS` |
| Grids | `gpu4pyscf.dft.gen_grid` | `pyscf.dft.gen_grid` |

## What Works

### Calculations Verified on Apple M4 Pro (48 GB unified memory)

| Method | Basis | Energy (Hartree) | vs PySCF CPU |
|--------|-------|-----------------|--------------|
| RHF | STO-3G | -74.9630631297 | 7.1e-12 |
| RHF | def2-SVP | -75.9609751670 | 1.3e-11 |
| RHF | def2-TZVPP | -76.0624634523 | 2.3e-12 |
| DF-RHF | def2-TZVPP | -76.0624583327 | 0.0 |
| LDA | def2-TZVPP | -75.2427579117 | 1.7e-10 |
| PBE | def2-TZVPP | -76.3800182267 | 1.9e-10 |
| B3LYP | def2-TZVPP | -76.4666495581 | 1.6e-10 |
| PBE0 | def2-TZVPP | -76.3808230213 | 1.5e-10 |
| M06 | def2-TZVPP | -76.4265805640 | 1.6e-10 |
| DF-B3LYP | def2-TZVPP | -76.4666819936 | 0.0 |
| DF-PBE0 | def2-TZVPP | -76.3808521773 | 0.0 |

All closed-shell calculations on water (H₂O). All match PySCF CPU to machine precision or better.

### Unrestricted (Open-Shell) Calculations

| Method | System | Basis | Energy (Hartree) | vs PySCF CPU |
|--------|--------|-------|-----------------|--------------|
| UHF | OH radical | def2-SVP | -75.3251000880 | 2.5e-12 |
| UHF | H₂O⁺ | def2-SVP | -75.5622721229 | 3.1e-11 |
| UKS B3LYP | OH radical | def2-SVP | -75.6673261832 | 1.9e-07 |
| UKS PBE | H₂O⁺ | def2-SVP | -75.8170437570 | 1.4e-14 |
| DF-UKS B3LYP | H₂O⁺ | def2-SVP | -75.9019164338 | 0.0 |

### Backend Verification Tests

116 tests in `gpu4pyscf/lib/backends/tests/`, run on both MLX and NumPy backends:

- **Layer 1 (92 tests):** Array creation, data transfer, eigh, eigvalsh, solve, SVD, inv, norm, dot, einsum, device/memory/stream managers
- **Layer 2 (24 tests):** Fock diagonalization vs PySCF, density matrix reconstruction, overlap matrix eigendecomposition, one-electron energy, ERI Coulomb contraction, DFT MO energies, near-singular matrix handling

**Tolerances:**

| Category | NumPy backend | MLX backend |
|----------|--------------|-------------|
| Array roundtrip | exact | exact |
| Eigenvalues (linalg) | 1e-10 | 1e-5 (float32) |
| Chemistry eigenvalues | 1e-5 | 1e-3 |
| Energy (Hartree) | 1e-8 | 1e-3 |

## Files Modified from Upstream

### Package initialization (removes hard CuPy dependency)

| File | Change |
|------|--------|
| `__init__.py` | Lazy submodule imports via `__getattr__`; memory pool setup only on CuPy |
| `__config__.py` | Uses `backend.device`/`backend.memory` instead of `cupy.cuda.runtime` |
| `_patch_pyscf.py` | `cupy.asarray()` → `to_device()`; deferred `gto.mole` import |

### Library layer

| File | Change |
|------|--------|
| `lib/__init__.py` | Conditional import of `cupy_helper`/`cutensor`; lazy error stubs on non-CuPy |
| `lib/logger.py` | Backend-agnostic GPU timing (`cupy.cuda.Event` → wall-clock fallback) |
| `lib/diis.py` | `cupy` → `backend` imports (`to_device`, `to_host`, `is_device_array`) |
| `lib/utils.py` | Removed `import cupy`; `to_cpu()` uses `is_device_array`/`to_host`; `format_sys_info()` backend-aware |

### SCF module

| File | Change |
|------|--------|
| `scf/__init__.py` | Imports RHF and UHF on all backends; guards GHF/ROHF behind CuPy |
| `scf/hf.py` | All `cupy.*` → `np.*`; CPU fallback for J/K, hcore, ovlp, init_guess, DIIS, density_fit; backend-agnostic `eigh`, `tag_array`, `asarray`, etc. |
| `scf/uhf.py` | All `cupy.*` → `np.*`; inherits CPU fallbacks from hf.py |

### DFT module

| File | Change |
|------|--------|
| `dft/__init__.py` | Imports RKS and UKS on all backends; guards GKS/ROKS behind CuPy |
| `dft/rks.py` | GPU path (cupy) vs CPU fallback (PySCF numint) for XC evaluation; PySCF grids on non-CuPy |
| `dft/uks.py` | GPU path (cupy) vs CPU fallback (PySCF numint) for unrestricted XC; mirrors rks.py pattern |

## Files Created

| File | Purpose |
|------|---------|
| `lib/backend.py` | Public API for backend-agnostic code |
| `lib/backends/__init__.py` | Backend detection and selection |
| `lib/backends/cupy_backend.py` | CuPy/CUDA wrapper |
| `lib/backends/mlx_backend.py` | MLX/Metal wrapper with float64 handling |
| `lib/backends/numpy_backend.py` | CPU-only fallback |
| `lib/backends/tests/__init__.py` | Test package |
| `lib/backends/tests/conftest.py` | Test fixtures: backend loader, tolerance helpers |
| `lib/backends/tests/test_backend_unit.py` | 92 unit tests for backend API |
| `lib/backends/tests/test_backend_chemistry.py` | 24 chemistry integration tests |

## MLX Backend Details

### Bug Fixes Applied

1. **`except AttributeError` → `except (AttributeError, ValueError, NotImplementedError)`** — MLX linalg ops raise `ValueError` ("not yet supported on the GPU"), not `AttributeError`
2. **`stream=mx.cpu`** for native MLX linalg (solve, eigh, inv, svd) — Metal GPU doesn't support these yet
3. **Float64 detection** — Routes to SciPy when MLX would silently truncate to float32
4. **`dot()` for 2D×1D** — `mx.matmul` requires reshaping 1D to column vector
5. **`mx.metal.set_memory_limit` deprecation** — Updated to `mx.set_memory_limit`

### Apple Silicon Unified Memory

Apple Silicon shares RAM between CPU and GPU. This means:
- No explicit host↔device transfer overhead
- `to_host()` / `to_device()` are lightweight (data stays in the same physical memory)
- The memory pool limit is set to 90% of system RAM by default

## What's Not Yet Ported

| Feature | Status | Blocker |
|---------|--------|---------|
| UHF | Ported | Working on MLX backend |
| UKS | Ported | Working on MLX backend |
| RHF/RKS/UHF/UKS gradients | Ported | CPU fallback with f64 refinement (grad error ~1e-7) |
| Geometry optimization | Working | Via geomeTRIC or PySCF berny with gradients |
| ROHF, GHF | Ported | CPU fallback via PySCF |
| Analytical Hessians | Ported | CPU fallback with f64 refinement (freq error < 0.01 cm⁻¹) |
| Solvent models (PCM/SMD) | Ported | CPU fallback via PySCF (exact for non-DF, ~5e-6 for DF) |
| TDDFT / TDA | Ported | CPU fallback with f64 refinement (error < 3e-6 eV) |
| Metal GPU acceleration | Active | DF-J/K 6-12x, eval_ao 1.1x, XC contraction 3.6x, end-to-end 1.5-3.9x |
| Rys integral engine (CPU) | Working | Direct J/K for s/p/d/f, machine precision on def2-TZVPP |
| Rys integral engine (Metal GPU) | **SCF-integrated** | 102x optimised, 30ms for H₂O/def2-SVP (libcint 1.8ms) |
| QM/MM | Ported | CPU fallback via PySCF |
| PBC (periodic) | Ported | CPU fallback via PySCF |

## Roadmap for Metal GPU Acceleration

### Phase 1: CPU Fallback (DONE)

Everything works via PySCF CPU. No Metal GPU acceleration but functionally correct.

### Phase 2: Mixed-Precision Metal Acceleration (in progress)

**Critical finding:** Apple Silicon GPUs have NO float64 hardware. Metal Shading Language has no `double` type. However, Metal GPU is **20x faster than CPU** for float32 matrix operations at typical quantum chemistry sizes (2000x2000).

#### Prior Art: Float64 on Hardware Without Native FP64

This is a solved problem in other disciplines. Three approaches exist:

**1. Emulated Float64 via `metal-float64` ([github.com/philipturner/metal-float64](https://github.com/philipturner/metal-float64))**
Philip Turner's library implements IEEE-compliant double precision using pairs of float32 ops in MSL. Throughput is ~1/36–1/68 of float32 — competitive with NVIDIA consumer GPU FP64 rates (1/32–1/64). Redefines the `double` keyword in Metal shaders via compiler macro.

**2. Mixed-Precision Iterative Refinement ([Dongarra et al., Royal Society A, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/))**
Factor `A` in float32 on GPU (O(N³), fast), compute residual `r = b - Ax` in float64 on CPU (O(N²), cheap), solve correction using low-precision factors, repeat. Delivers float64 accuracy at **4x speedup** over pure float64. The expensive factorization runs in low precision; the cheap refinement restores full precision.

**3. Mixed-Precision SCF for Quantum Chemistry ([Codina et al., JCTC 2025](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01800))**
Directly relevant to our problem. Most of the SCF cycle (J/K build, XC integration, Fock assembly) runs in float32. Only the eigensolve and energy accumulation need float64. One refinement step in double precision per SCF cycle restores accuracy. Result: **4–7x speedup** with no degradation in convergence. The [gpu4pyscf paper](https://arxiv.org/html/2404.09452v2) itself confirms mixed precision schemes "remain stable and accurate for molecules with d functions."

**Strategy for this port (combining approaches 1 and 3):**
- **J/K integral contractions** → float32 on Metal GPU (dominant cost, 14–20x speedup)
- **DFT grid integration** → float32 on Metal GPU
- **Fock matrix eigensolve** → float64 on CPU via Accelerate (or emulated via metal-float64)
- **Energy accumulation** → float64 on CPU (cheap, O(N²))
- **Final SCF cycle** → one double-precision refinement pass to restore full accuracy

This is essentially what NVIDIA's AI-focused GPUs (H100 with weak FP64) are already forcing the field to adopt. Apple Silicon makes it mandatory rather than optional.

**Benchmark results (Apple M4 Pro):**

| Operation | Size | CPU f64 | Metal GPU f32 | Speedup |
|-----------|------|---------|---------------|---------|
| Matrix multiply | 2000×2000 | 323 ms | 16 ms | **20x** |
| DF-J/K (H₂O) | 59 AO, 113 aux | 20 ms | 2 ms | **11.5x** |
| DF-J/K (caffeine/SVP) | 246 AO, 1242 aux | 210 ms | 17 ms | **12.3x** |
| DF-J/K (caffeine/TZVPP) | 574 AO, 1242 aux | 545 ms | 87 ms | **6.3x** |
| dist_matrix | 5000×5000 | 449 ms | 32 ms | **14x** |
| Eigendecomposition | 500×500 | 20 ms | 9 ms | 2.2x |

**DF-J/K implementation** (`gpu4pyscf/df/df_jk_metal.py`):
- CDERI precomputed once by PySCF, unpacked and cached on GPU as f32 MLX arrays
- J: packed-space gemvs — two matrix-vector products, no unpack per cycle
- K: half-transform Y = cderi @ L_occ (nocc << nao), then K = Y_t @ Y_t.T as single GPU gemm
- Per-cycle cost is pure GPU BLAS with zero CPU data movement
- One-time build cost (unpack + transfer): ~0.5s for 246 AOs, ~4s for 574 AOs
- Build cost is amortized over 10-30 SCF iterations
- Accuracy: relative error ~1e-7 (J) to ~2e-6 (K) from f32 precision

### Phase 3: Metal Compute Shaders (first kernels done)

Ported CUDA kernels to Metal via `mx.fast.metal_kernel()`:

| Kernel | CUDA source | Metal status | Correctness | GPU speedup |
|--------|-------------|-------------|-------------|-------------|
| `dist_matrix` | `dist_matrix.cu` | **Done** | error < 1e-7 | **14x** |
| `pack_tril` | `unpack.cu` | **Done** | exact | (small matrix overhead) |
| `unpack_tril` | `unpack.cu` | **Done** | exact | — |
| `fill_triu` | `unpack.cu` | **Done** | exact | — |
| `transpose_sum` | `transpose.cu` | Not started | — | Needs threadgroup memory |
| `block_diag` | `block_diag.cu` | Not started | — | Easy |
| `add_sparse` | `add_sparse.cu` | Not started | — | Needs atomics |
| `cart2sph` | `cart2sph.cu` | Not started | — | Medium |
| `grouped_gemm` | `grouped_gemm.cu` | Not started | — | Needs CUTLASS replacement |

Metal kernels are in `gpu4pyscf/lib/metal_kernels/__init__.py`.

### Phase 4: End-to-End Metal-Accelerated DF-DFT SCF (working)

The Metal DF-J/K engine and Metal XC kernels are both integrated into the SCF loop via `density_fit()`. `_patch_df_with_metal_jk()` (in `gpu4pyscf/scf/hf.py`) replaces `with_df.get_jk` with the Metal f32 engine, and also installs `_patch_dft_veff_metal()` / `_patch_dft_veff_metal_uks()` which override the DFT `get_veff` to route XC evaluation through `nr_rks_metal` / `nr_uks_metal`.

Cached `MetalDFTensors` (CDERI on GPU) persist across SCF calls via `gpu4pyscf/df/df_jk_metal.py`.

**Convergence strategy:** Default `conv_tol = 1e-4` for Metal DF objects (instead of 1e-9). Rationale: at the true minimum, f32 J/K noise gives a gradient-norm floor of ~1e-2 for larger molecules, so `conv_tol_grad = sqrt(conv_tol) = 0.01` is the tightest achievable. Energies are variational in the density → f32-converged densities give energies accurate to ≤1e-5 Ha. A f64 refinement step is run automatically before any gradient / Hessian / TDDFT calculation.

**End-to-end SCF benchmarks (DF-B3LYP, M4 Pro, cached GPU tensors):**

| System | AOs | CPU (s) | Metal (s) | Speedup | dE (Ha) |
|--------|-----|---------|-----------|---------|---------|
| H₂O / def2-TZVP | 34 | 0.4 | 0.1 | **3.7x** | 2e-6 |
| Benzene / def2-TZVP | 228 | 4.6 | 1.9 | **2.5x** | 7e-7 |
| Naphthalene / def2-TZVP | 358 | 12.4 | 4.5 | **2.8x** | 3e-5 |
| Aspirin / def2-TZVP | 451 | 21.6 | 6.7 | **3.2x** | 2e-6 |
| Anthracene / def2-TZVP | 494 | 18.7 | 7.7 | **2.4x** | 4e-5 |
| Caffeine / def2-TZVPP | 557 | 40.3 | 22.2 | **1.8x** | 6e-6 |

Median RKS speedup across 13 systems spanning 3–36 atoms: **~2.8x**.

### UKS (Unrestricted DFT)

UKS (open-shell) calculations go through `nr_uks_metal` (`gpu4pyscf/dft/numint_metal.py`) which computes alpha and beta densities in the same batch loop, calls libxc once with `spin=1` (spin-coupled XC), then contracts wv_alpha / wv_beta into separate vmat matrices on the GPU. The J/K side is batched in `_batched_jk` (`gpu4pyscf/df/df_jk_metal.py`): one matmul-pair for J across all spins, one half-transform for K.

| System | CPU | Metal | Speedup |
|--------|-----|-------|---------|
| NO₂ radical / def2-TZVP | 1.4s | 0.5s | **3.0x** |
| Phenyl radical / def2-TZVPP | 9.3s | 2.3s | **4.0x** |
| Phenyl radical / def2-TZVP | 10.7s | 1.6s | **6.5x** |

### XC Numerical Integration (accelerated)

The XC path evaluates AOs on the grid via `eval_ao` Metal kernels, computes rho and Vxc on GPU using fused kernels (small molecules) or batched gemm (large molecules), and contracts the XC potential into `vmat` entirely on GPU. Only the libxc functional evaluation itself runs on CPU (single rho→libxc→wv roundtrip per SCF cycle).

**XC-only speedup on caffeine/def2-TZVPP (574 AOs):** 1029ms CPU → 318ms Metal = **3.2x**.

Key optimizations in `gpu4pyscf/lib/metal_kernels/fused_xc.py`:
- Grid coords pre-uploaded once per SCF cycle (not per batch)
- wv (weights × vxc) pre-uploaded once per SCF cycle
- shell_data cached on GPU (set up in `_prepare_shell_data`)
- vmat accumulated on GPU, single download at end of loop
- `_eval_ao_batch_gpu` accepts pre-uploaded MLX arrays directly

The earlier naive implementation was 0.7x (slower) due to ~60 per-batch numpy↔MLX conversions dominating the GPU savings. Eliminating those gives the current 3.2x speedup.

### Analytical Gradients and Hessians (correct, not faster)

`.Gradients().kernel()` and `.Hessian().kernel()` work and give correct results (1e-7 to 1e-6 accuracy on typical molecules) via automatic f64 refinement: the Metal-converged density seeds a short PySCF CPU f64 SCF which provides the tight density needed for analytical derivatives.

End-to-end wall time is roughly equal to CPU — the SCF savings (2-5x) are consumed by (a) the f64 refinement pass (~40% of gradient wall time) and (b) PySCF's CPU-bound gradient/Hessian analytical integral code which is not Metal-accelerated. The current `_refine_to_f64` uses `conv_tol=1e-10, max_cycle=10`; lighter refinement fails the 1e-5 gradient accuracy threshold on larger aromatics.

### Metal eval_ao (working)

**File:** `gpu4pyscf/lib/metal_kernels/eval_ao.py`

Translates PySCF's C-level `eval_gto` (Gaussian basis evaluation) to a Metal compute shader. Single kernel launch processes all (grid_point, shell) pairs in parallel. Supports l=0 through l=3 (s, p, d, f) with deriv=0 and deriv=1 (GGA gradients). Cart-to-spherical transformation on GPU.

### Fused XC kernel with threadgroup shared memory

**File:** `gpu4pyscf/lib/metal_kernels/fused_xc.py`

Computes rho (and nabla rho for GGA) entirely in threadgroup shared memory — AO values never written to global memory. One threadgroup per grid point, nao threads. Includes analytical AO derivatives for s/p/d/f shells with full cart-to-spherical transform via precomputed lookup tables.

Used for small molecules (nao ≤ 128) where the dm@ao inner loop fits in threadgroup parallelism. For larger nao, the batched gemm approach is faster.

### Metal Rys Polynomial Integral Engine

**Files:** `gpu4pyscf/lib/metal_kernels/rys_jk.py` (CPU reference), `gpu4pyscf/lib/metal_kernels/rys_jk_metal.py` (Metal GPU)

Complete implementation of the Rys polynomial quadrature for 2-electron integrals, translated from the CUDA `gvhf-rys` engine.

**CPU reference** (`rys_jk.py`):
- Boys function F_m(t): 3-regime computation (series, asymptotic, erf)
- Rys root extraction via Hankel matrix + Vandermonde solve (nroots 1-7)
- Obara-Saika TRR with Rys-factorized 1D recurrence
- HRR for j and l angular momentum transfer
- Cart-to-spherical ERI transformation for d/f shells
- Verified to machine precision on STO-3G, def2-SVP, def2-TZVPP

**Metal GPU kernel** (`rys_jk_metal.py`):

Three-phase architecture for f64-accurate direct J/K with GPU acceleration:
1. **CPU f64:** Boys function (vectorised erf + upward recursion) + Rys roots/weights via batched Hankel matrix solve (3 batched calls, grouped by nroots). The f32 Hankel solve diverges for nroots>=3 at large t, so f64 is mandatory.
2. **Metal f32:** TRR + HRR + Cartesian ERI assembly. Each primitive gets its OWN output slot (no atomic adds, fully deterministic). MLX reuses output buffers, so the kernel explicitly zeros each slot before writing.
3. **CPU f64:** Numba-compiled primitive summation per shell quartet, cart-to-spherical transformation for d/f shells, Numba-compiled J/K accumulation with 8-fold symmetry unfolding.

Optimisations (102x speedup over initial implementation):
- **8-fold ERI symmetry** — ish≥jsh, ksh≥lsh, ij≥kl reduces quartets ~8x
- **Grouped batch expansion** — shell quartets grouped by (n_ij, n_kl); one batched repeat/tile per group eliminates per-quartet Python loops
- **Vectorised Boys function** — scipy.special.erf + upward recursion (7x faster than gammainc)
- **Batched Rys roots** — nroots=1 (closed-form), nroots=2 (Cramer's rule + quadratic), nroots=3 (batched 3×3 Hankel solve + companion matrix eigenvalues + Vandermonde). All primitives processed in 3 calls, not 3000.
- **Numba JIT accumulation** — explicit compiled loops replace 17K numpy.einsum calls on tiny arrays (0.3ms vs 60ms)
- **Numba primitive summation** — compiled per-quartet reduction replaces fancy indexing
- **Pre-computed shell pairs** — pair data (aij, PA, Pij, Kab) cached once, avoids redundant kl-pair recomputation
- **Schwarz bounds cached** on mol object; cart2sph matrices cached at module level

**Performance (H₂O, Apple M4 Pro):**

| Basis | Metal Rys | libcint (C) | Ratio | Original (unoptimised) |
|-------|-----------|-------------|-------|------------------------|
| STO-3G | **2.9 ms** | 0.3 ms | 9.7x | 233 ms (80x faster) |
| def2-SVP | **29.9 ms** | 1.8 ms | 16.6x | 3054 ms (102x faster) |

The GPU kernel itself is <1ms. The remaining gap to libcint is Python overhead in the build phase (~18ms: quartet enumeration, grouped expansion, Boys/Rys batching, sort) and Phase 3a (~8ms: cart2sph loop, metadata preparation). Closing this requires either Numba-compiling the entire pipeline or a monolithic Metal kernel — diminishing returns for small molecules.

**Where Metal Rys sits in the performance landscape:**

For small molecules, libcint (pure C, highly optimised, decades of development) is unbeatable. Metal Rys serves a different purpose: it provides a **correct, density-fitting-free J/K** on Apple Silicon that is fast enough for routine SCF convergence. The practical fast path is the DF engine (see below), which achieves real end-to-end speedups. Metal Rys is the fallback for methods that cannot use density fitting (e.g. exact exchange in hybrid DFT without DF, or testing/validation).

**Accuracy:** dE < 1e-9 Ha vs PySCF CPU reference. J error ~4e-7, K error ~7e-8 (f32 TRR/HRR precision, all summation in f64). SCF converges at conv_tol=1e-8 without DIIS issues.

**SCF integration:** The Metal Rys engine is the default `get_jk` on the MLX backend. No density fitting required.
```python
from gpu4pyscf.scf import hf
mf = hf.RHF(mol)  # uses Metal Rys automatically on Apple Silicon
mf.kernel()
```

### Metal eval_ao (working)

**File:** `gpu4pyscf/lib/metal_kernels/eval_ao.py`

Translates PySCF's C-level `eval_gto` (Gaussian basis evaluation) to a Metal compute shader. Single kernel launch processes all (grid_point, shell) pairs in parallel.

| Component | CPU (ms) | Metal (ms) | Speedup |
|-----------|----------|------------|---------|
| AO evaluation (deriv=0) | 394 | 344 | **1.1x** |
| Cart-to-spherical transform | (in CPU time) | (on GPU) | — |

Correctness: ~1e-7 relative error (f32 precision) across STO-3G, def2-SVP, def2-TZVPP basis sets.

**Full XC pipeline (LDA, caffeine/def2-TZVPP, 574 AOs, 295K grids):**

| Step | CPU (ms) | Metal (ms) |
|------|----------|------------|
| eval_ao | 394 | 344 |
| rho computation | ~200 | 182 (GPU gemm) |
| libxc XC evaluation | 2 | 2 (CPU) |
| Vxc contraction | ~260 | 98 (GPU gemm) |
| **Total** | **~860** | **~630** |
| **Speedup** | | **1.4x** |

Supports l=0 through l=4 (s, p, d, f, g). Spherical harmonics handled via cart-to-sph transform on GPU.

**GGA derivative support (deriv=1):** Implemented. Computes d/dx, d/dy, d/dz of all basis functions using analytical formulas (product rule on polynomial × Gaussian). Correctness verified against PySCF CPU:

| System | val rel err | dx rel err | dy rel err | dz rel err |
|--------|-----------|-----------|-----------|-----------|
| H₂O/def2-SVP | 1.6e-7 | 2.6e-7 | 2.6e-7 | 1.1e-6 |
| H₂O/def2-TZVPP | 2.3e-7 | 5.5e-7 | 5.5e-7 | 3.6e-6 |
| Caffeine/def2-TZVPP | 4.6e-6 | 1.1e-4 | 1.4e-4 | 2.9e-5 |

Higher derivative errors for caffeine are from f-function polynomial evaluation in f32 — acceptable for DFT energies.

**Performance note:** For large systems, materializing the full (4, ngrids, ncart) output buffer (~3 GB for caffeine/TZVPP) makes the deriv=1 kernel slower than CPU. The proper architecture is a **fused eval_ao + Vxc contraction kernel** that processes grid batches and contracts immediately, never materializing the full AO array. This is the CUDA code's approach (`_nr_rks_task` loops over grid batches).

### Roadmap: where performance comes from on Metal

**The density-fitting path is the practical Metal equivalent of what CUDA gives PySCF.** DF-J/K reduces to large GEMM (3-index tensor contractions), which maps directly to Metal's matmul hardware. This is where real end-to-end speedups come from — 1.5–3.5x already, scaling better with molecule size.

**Direct integrals (Rys) cannot match CUDA's approach from Python.** CUDA's `libgvhf_rys` runs the entire J/K pipeline — task generation, Rys roots, TRR/HRR, contraction, accumulation — in a single kernel launch with zero Python round-trips. Our Metal Rys engine has a fast GPU kernel (<1ms) but ~30ms of Python orchestration. To match CUDA would require a monolithic Metal kernel that takes `(mol, dm)` → `(J, K)` in one dispatch — essentially writing a new ERI engine in Metal Shading Language.

**Priorities for maximum return:**

| Action | Effort | Expected gain |
|--------|--------|---------------|
| **Optimise DF-CDERI build** (move to Metal/Accelerate) | Medium | 2–3x end-to-end for DF-DFT |
| **Reduce XC conversion overhead** in fused Metal kernels | Low | 1.2–1.5x for DFT |
| **Larger molecules** where GPU parallelism dominates | None | Current code already benefits |
| **Monolithic Metal J/K kernel** (full ERI in MSL) | High | Would match CUDA Rys for direct J/K |
| Translate remaining CUDA kernels to Metal | High | Unlocks gradient/Hessian GPU paths |

**Remaining CUDA kernel translation effort:**

| Module | CUDA files | Purpose | Effort |
|--------|-----------|---------|--------|
| `gvhf-rys/` | 26 files | J/K matrix via Rys polynomials (monolithic kernel) | Months |
| `gint/` | 38 files | Gaussian integral evaluation | Months |
| `gdft/` | 6 files | DFT grid integration | Weeks |
| `multigrid/` | 13 files | Multi-grid DFT | Weeks |
| `ecp/` | 13 files | Effective core potentials | Weeks |

### Key CUDA→Metal Translation Notes

| CUDA Concept | Metal Equivalent |
|-------------|-----------------|
| `__global__` kernel | `kernel` function in MSL |
| `threadIdx.x` | `thread_position_in_threadgroup` |
| `blockIdx.x` | `threadgroup_position_in_grid` |
| `__shared__` memory | `threadgroup` memory |
| Warp (32 threads) | SIMD group (32 threads on Apple GPU) |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| cuBLAS | Accelerate BLAS (via scipy/numpy) |
| cuSOLVER | Accelerate LAPACK (via scipy) |
| cuTENSOR | No equivalent; use einsum or custom Metal |
| CUTLASS | No equivalent; must rewrite |

## Running the Tests

```bash
# Backend verification tests (MLX + NumPy)
python3 -m pytest gpu4pyscf/lib/backends/tests/ -v --import-mode=importlib

# Quick smoke test — single-point energy
python3 -c "
from gpu4pyscf.dft import RKS
import pyscf
mol = pyscf.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='def2-svp')
mf = RKS(mol, xc='B3LYP').density_fit()
mf.kernel()
print('Energy:', mf.e_tot)
"

# Full workflow test — energy + gradient + Hessian + TDDFT + PCM
python3 -c "
import pyscf
from gpu4pyscf.dft import RKS
from pyscf.hessian import thermo

mol = pyscf.M(atom='O 0 0 0.117; H -0.757 0 -0.470; H 0.757 0 -0.470',
              basis='def2-svp', verbose=0)
mf = RKS(mol, xc='B3LYP').density_fit()
mf.kernel()
print(f'Energy: {mf.e_tot:.10f} Ha')

g = mf.nuc_grad_method().kernel()
print(f'Gradient max: {abs(g).max():.6f} Ha/Bohr')

h = mf.Hessian().kernel()
freq = thermo.harmonic_analysis(mol, h)['freq_wavenumber']
print(f'Frequencies: {[f\"{f:.0f}\" for f in freq if abs(f)>10]} cm-1')

td = mf.TDDFT(); td.nstates = 3; td.kernel()
print(f'Excitations: {[f\"{e*27.2114:.2f}\" for e in td.e]} eV')

mf_pcm = RKS(mol, xc=\"B3LYP\").density_fit().PCM()
mf_pcm.verbose = 0; mf_pcm.kernel()
print(f'PCM energy: {mf_pcm.e_tot:.10f} Ha')
"
```

## Complete File Manifest

### Files Created (21 files)

| File | Purpose |
|------|---------|
| `METAL_PORT.md` | This document |
| `README.md` | User-facing documentation |
| `pyproject.toml` | Package metadata for `pip install` |
| `gpu4pyscf/lib/backend.py` | Public API: `xp`, `device`, `memory`, `stream`, linalg |
| `gpu4pyscf/lib/backends/__init__.py` | Auto-detect CuPy → MLX → NumPy |
| `gpu4pyscf/lib/backends/cupy_backend.py` | CuPy/CUDA wrapper |
| `gpu4pyscf/lib/backends/mlx_backend.py` | MLX/Metal wrapper with f64 handling |
| `gpu4pyscf/lib/backends/numpy_backend.py` | CPU-only fallback |
| `gpu4pyscf/lib/backends/tests/__init__.py` | Test package |
| `gpu4pyscf/lib/backends/tests/conftest.py` | Fixtures: backend loader, tolerance helpers |
| `gpu4pyscf/lib/backends/tests/test_backend_unit.py` | 92 unit tests |
| `gpu4pyscf/lib/backends/tests/test_backend_chemistry.py` | 24 chemistry tests |
| `gpu4pyscf/lib/metal_kernels/__init__.py` | Metal compute shaders: dist_matrix, pack/unpack_tril, fill_triu |
| `gpu4pyscf/lib/metal_kernels/eval_ao.py` | Metal eval_ao kernel (deriv=0,1) with cart2sph |
| `gpu4pyscf/lib/metal_kernels/fused_xc.py` | Fused eval_ao+rho+Vxc with threadgroup shared memory |
| `gpu4pyscf/lib/metal_kernels/rys_jk.py` | CPU Rys polynomial integral engine (reference) |
| `gpu4pyscf/lib/metal_kernels/rys_jk_metal.py` | Metal GPU Rys integral engine (f64 roots + f32 TRR/HRR + f64 accumulation) |
| `gpu4pyscf/df/df_jk_metal.py` | Metal GPU DF-J/K engine with cached tensors |
| `gpu4pyscf/dft/numint_metal.py` | Metal GPU XC integration (fused eval_ao + contraction) |
| `gpu4pyscf/grad/_cpu_fallback.py` | RHF/RKS/UHF/UKS gradient CPU fallback |
| `gpu4pyscf/hessian/_cpu_fallback.py` | RHF/RKS/UHF/UKS Hessian CPU fallback |

### Files Modified (29 files)

| File | Change |
|------|--------|
| `gpu4pyscf/__init__.py` | Lazy submodule imports, backend-aware memory pool |
| `gpu4pyscf/__config__.py` | Backend device/memory managers instead of cupy.cuda |
| `gpu4pyscf/_patch_pyscf.py` | `to_device()` instead of `cupy.asarray()`, deferred imports |
| `gpu4pyscf/lib/__init__.py` | Conditional cupy_helper/cutensor imports |
| `gpu4pyscf/lib/logger.py` | Backend-agnostic GPU timing |
| `gpu4pyscf/lib/diis.py` | Backend data transfer functions |
| `gpu4pyscf/lib/utils.py` | `is_device_array`/`to_host`, backend-aware `format_sys_info` |
| `gpu4pyscf/scf/__init__.py` | Imports RHF+UHF on all backends |
| `gpu4pyscf/scf/hf.py` | Backend-agnostic SCF: CPU fallback J/K, Metal DF patch, f64 refinement, TDDFT/PCM/SMD methods |
| `gpu4pyscf/scf/uhf.py` | Backend-agnostic UHF |
| `gpu4pyscf/dft/__init__.py` | Imports RKS+UKS on all backends |
| `gpu4pyscf/dft/rks.py` | Backend-aware XC evaluation with Metal numint |
| `gpu4pyscf/dft/uks.py` | Backend-aware unrestricted XC |
| `gpu4pyscf/grad/__init__.py` | Routes to CPU fallback on non-CUDA |
| `gpu4pyscf/hessian/__init__.py` | Routes to CPU fallback on non-CUDA |
| `gpu4pyscf/solvent/__init__.py` | Delegates to PySCF CPU solvent on non-CUDA |
| `gpu4pyscf/tdscf/__init__.py` | Guards CUDA imports on non-CUDA |
| `gpu4pyscf/df/__init__.py` | Guards cupy imports, falls back to PySCF DF |
| `gpu4pyscf/mp/__init__.py` | Falls back to PySCF MP2/RMP2 |
| `gpu4pyscf/cc/__init__.py` | Falls back to PySCF CCSD |
| `gpu4pyscf/qmmm/__init__.py` | Falls back to PySCF QM/MM |
| `gpu4pyscf/pbc/__init__.py` | Falls back to PySCF PBC |
| `gpu4pyscf/properties/__init__.py` | Guards CUDA-only properties |
| `gpu4pyscf/tools/__init__.py` | Guards CUDA-only tools |
| `gpu4pyscf/md/__init__.py` | Guards CUDA-only MD |
| `gpu4pyscf/nac/__init__.py` | Guards CUDA-only NAC |
| `setup.py` → `setup_cuda.py` | Renamed to avoid nvcc requirement |
| `tests/` (9 files) | Integration tests: SCF, DFT, grad, Hessian, TDDFT, solvent, MP2, CCSD, imports |
