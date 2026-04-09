# Seminumerical Exchange (SGX) Gradient for Metal GPU

## Motivation

The current DF (density-fitting) gradient pipeline has a hard precision barrier:
the `int3c2e_ip1` 3-center 2-electron derivative integrals cannot be computed
accurately on Apple Silicon because f32 GPU compute gives ~1e-4 relative error
in the integral values, which compounds to ~0.1 absolute error in the gradient.

This is NOT fixable with better algorithms (double-single arithmetic was tried
and doesn't help — the precision limit is in the f32 INPUT data, not the f32
computation). Apple Silicon GPUs have no f64 hardware.

## The Alternative: Seminumerical Exchange (SGX)

### What changes

The standard DF gradient for hybrid DFT (B3LYP, PBE0) has this structure:

```
dE/dA = dE_J/dA + dE_K/dA + dE_XC/dA + dE_nuc/dA + ...
```

where:
- `dE_J/dA` (Coulomb) uses `int3c2e_ip1` for ∂(μν|P)/∂A — **BLOCKED by f32**
- `dE_K/dA` (exact exchange) uses `int3c2e_ip1` for ∂(μν|P)/∂A — **BLOCKED by f32**
- `dE_XC/dA` (XC functional) uses eval_ao + grid quadrature — **WORKS in f32**

The seminumerical exchange approach (Neese 2009, Izsák & Neese 2011) replaces
the 3c2e integrals for exchange with:

```
K_μν ≈ Σ_g w_g Σ_λ D_λσ φ_μ(r_g) (λσ|r_g) φ_ν(r_g)
```

where `(λσ|r_g)` is a **3-center 1-electron** integral (nuclear attraction type).

### Why 3c1e integrals are f32-safe

The 3c2e integral `(μν|P) = ∫∫ χ_μ(r1) χ_ν(r1) (1/r12) χ_P(r2) dr1 dr2` is
a 6-dimensional integral requiring Rys quadrature with up to 6 quadrature roots.
The Obara-Saika recursion has 8+ levels, and f32 errors compound.

The 3c1e integral `(μν|r_g) = ∫ χ_μ(r) χ_ν(r) (1/|r-r_g|) dr` is a
3-dimensional integral that:
- Uses the Boys function directly (no Rys root-finding)
- Has simpler recursion (fewer levels, less cancellation)
- Has been shown to be accurate in f32 (Laqua et al., JCP 2021)

### The gradient structure with SGX

For the K contribution to the gradient:
```
dE_K/dA ≈ Σ_g w_g Σ_μνλσ D_μν D_λσ [
    (∂φ_μ/∂A)(r_g) (λσ|r_g) φ_ν(r_g)
  + φ_μ(r_g) (∂(λσ)/∂A|r_g) φ_ν(r_g)
  + φ_μ(r_g) (λσ|r_g) (∂φ_ν/∂A)(r_g)
]
```

The components:
1. `∂φ/∂A` — AO derivatives. **Already on Metal GPU** (eval_ao deriv=1,2).
2. `(λσ|r_g)` — 3c1e integrals at grid points. **f32-safe**, simpler than 3c2e.
3. `∂(λσ)/∂A|r_g` — 3c1e derivative integrals. **f32-safe**, same as above.
4. Grid weights `w_g`. Already available from XC grid.

The Coulomb (J) contribution stays as DF (already working on Metal).

### What we reuse

| Component | Current status | Needed for SGX gradient |
|---|---|---|
| Metal eval_ao (deriv 0,1,2) | Working | Reuse directly |
| Metal fused XC grid | Working | Reuse grid infrastructure |
| Metal DF J/K (SCF) | Working | J stays DF, K switches to SGX |
| DFT grid (coords, weights) | Available from SCF | Reuse for SGX quadrature |
| Metal cho_solve gradient | Working | Reuse for J gradient |

### What we need to build

1. **3c1e integral evaluation on Metal GPU** — `(μν|r_g)` at grid points.
   This is a nuclear-attraction-type integral. For each grid point r_g and
   shell pair (μ,ν): `(μν|r_g) = Σ_prim N_p exp(-ζ(r-P)²) F0(ζ|P-r_g|²)`.
   Uses Boys function (already validated in f32 on Metal) + simple OS recursion.

2. **3c1e derivative integrals** — `∂(μν|r_g)/∂A` using the same derivative
   formula as int3c2e_ip1: `2α[(μ+1_x ν|r_g)] - n_x[(μ-1_x ν|r_g)]`.

3. **SGX gradient contraction kernel** — contract 3c1e integrals with density
   matrix and AO values on the grid. This is a batched GEMM on GPU.

4. **Wiring**: integrate SGX gradient into the `get_jk` path as an alternative
   to the DF int3c2e_ip1 path.

## Precision analysis

| Quantity | DF path (current) | SGX path (proposed) |
|---|---|---|
| J gradient integrals | int3c2e_ip1 (f32: 1e-4 error) | Same DF path (CPU f64) |
| K gradient integrals | int3c2e_ip1 (f32: 1e-4 error) | **3c1e (f32: ~1e-7 error)** |
| XC gradient | Metal eval_ao + grid (f32: ~1e-6) | Same |
| Overall gradient error | **0.108** (from K integrals) | **~1e-5** (expected) |

The key insight: the K gradient is the ONLY component that needs int3c2e_ip1.
By replacing it with SGX (3c1e + grid), we bypass the f32 precision barrier
entirely.

## Performance estimate

SGX exchange scales as O(N² × N_grid) vs DF's O(N² × N_aux). For typical
molecules:
- N_grid ~ 50,000-200,000 grid points (depends on grid level)
- N_aux ~ 500-2000 aux functions

SGX is ~10-100x more work than DF for the exchange part. BUT:
- The work is all GPU-friendly (GEMM, eval_ao — things we're good at)
- No int3c2e_ip1 (no Rys roots, no TRR precision issue)
- The grid quadrature parallelizes perfectly on GPU

Expected timing for caffeine gradient K contribution:
- Current (CPU int3c2e_ip1): ~3s
- SGX on Metal GPU: ~1-3s (depends on grid size)
- Net gradient time: similar or faster, with MUCH better precision

## Feasibility assessment (completed)

### Finding 1: PySCF SGX does NOT support analytical gradients

```python
mf_sgx = sgx.sgx_fit(RKS(mol, xc='B3LYP'))
mf_sgx.kernel()            # Works: energy matches DF to 4.5e-5 Ha
mf_sgx.nuc_grad_method()   # Raises NotImplementedError
```

The SGX module (`pyscf/sgx/`) contains only `sgx.py` and `sgx_jk.py` — no
gradient code. SGX gradients would need to be written from scratch.

### Finding 2: SGX only replaces K, not J

The DF gradient uses `int3c2e_ip1` for BOTH J and K contributions:

```
J gradient: vj = rhoj @ ∂(μν|P)/∂A    ← still needs int3c2e_ip1
K gradient: vk = rhok @ ∂(μν|P)/∂A    ← SGX can replace this
```

For non-hybrid DFT (PBE, BLYP): no K gradient term at all, and J still
needs int3c2e_ip1. SGX provides NO benefit.

For hybrid DFT (B3LYP, PBE0): SGX replaces K only. J stays on CPU.

### Finding 3: Cost-benefit assessment

| Approach | Effort | Gradient speedup |
|----------|--------|------------------|
| SGX K gradient from scratch | ~2 weeks | ~1.2-1.5x (hybrid only) |
| Current production path | Done | 1.14x (all functionals) |

### Conclusion

SGX is a viable **future direction** when either:
1. PySCF adds SGX gradient support upstream
2. A fully grid-based J gradient eliminates int3c2e_ip1 for both J and K
3. Apple exposes f64 Metal compute (making the DF path viable directly)

## References

1. Neese, F. "An improvement of the resolution of the identity approximation
   for the formation of the Coulomb matrix." J. Comput. Chem. 24, 1740 (2003).

2. Izsák, R. & Neese, F. "An overlap fitted chain of spheres exchange method."
   J. Chem. Phys. 135, 144105 (2011).

3. Laqua, H., Kussmann, J., & Ochsenfeld, C. "Accelerating seminumerical
   Fock-exchange calculations using mixed single- and double-precision
   arithmetic." J. Chem. Phys. 154, 214116 (2021).
   **Key finding: 3c1e integrals are f32-safe.**

4. PySCF SGX module: https://pyscf.org/user/sgx.html

5. Helmich-Paris, B. "An improved chain of spheres for exchange algorithm."
   J. Chem. Phys. 155, 104109 (2021).
