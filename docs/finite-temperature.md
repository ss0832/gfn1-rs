# Finite temperature and the charge-space response solver

Fermi smearing (`ElectronicOptions::electronic_temperature`, default 300 K) makes
the SCC solve the **Mermin free energy** `E − T·S_elec`, and every derivative of it
must carry the occupation response. This page describes the v0.5.0 response
machinery that makes that work, and which derivative orders it currently reaches.

The reported `total_free` is the Mermin free energy; `total_internal` is the plain
internal energy; `electronic_entropy` is the `−T·S` piece. `EnergyTerms::named_values()`
returns **15** terms (see [rust-api.md](rust-api.md#single-point)).

`constants::KB_HARTREE_PER_K` is the one Boltzmann constant used by every
finite-temperature path — see [parameters.md](parameters.md#unit-constants-reporting-vs-model).

---

## 1. The charge-space dielectric solver

Module: `gfn1_rs::response::charge_space` (not glob-re-exported; import the full
path). Sibling `gfn1_rs::response::cpxtb` is the historical MO-pair-space CPXTB
solver, still reachable as `gfn1_rs::cphf` through a compat shim.

**The idea.** The SCC self-consistency couples the density only through the `nsh`
Mulliken shell charges, so any response splits into a **bare** part (the density
response at frozen SCC potential, evaluated spectrally from the reference MOs) and
a **screening** part mediated by the response kernel `K`. With the static
shell-charge susceptibility `χ⁰` (charges induced by a unit shell potential at
frozen skeleton), the self-consistent shell-charge response of *any* perturbation
solves one `nsh × nsh` linear system

```text
(I − χ⁰ K) δq = δq_bare
```

whose **dielectric matrix is LU-factored once and reused for every right-hand
side** — all `3N` first-order geometric perturbations and, for the quartic force
constants, all `O((3N)²)` second-order ones. This replaces both the MO-pair-space
iterative/dense CPXTB solves and the 50-iteration fixed-point loop of the
finite-temperature branch with one exact solve.

### The MO-pair CPXTB solver uses the same reduction

`response::cpxtb` still has to produce the MO-pair amplitudes `x` themselves (the
Z-vector adjoint, the 2n+1 ladder and TD all consume them), so it cannot simply
call the charge-space solver. It applies the **same dielectric reduction directly
to its own operator** instead. The pair-space Jacobian is a diagonal plus a
rank-`nsh` term,

```text
A = D_g + T K Tᵀ D_s
```

(`D_g` = occupied–virtual gaps, `D_s = ½(f_i − f_a)`, `T` = the `npair × nsh`
Mulliken transition shell charges), and `npair ~ n²/4 ≫ nsh ~ n`. Eliminating the
diagonal in closed form gives

```text
y ≡ Tᵀ D_s x,   X ≡ Tᵀ D_s D_g⁻¹ T          (nsh × nsh)
(I + X K) y = (D_s D_g⁻¹ T)ᵀ b
x = D_g⁻¹ (b − T K y)
```

— again **one `nsh × nsh` factorization for the whole right-hand-side family**,
with `O(npair·nsh)` GEMM work per RHS instead of an `O(npair³)` dense
factorization or a Krylov loop. `CpxtbSolution::route` reports which route ran
(`ChargeSpace` / `Dense` / `Pcg` / `PcgDenseFallback`); the dense LU and the
preconditioned CG remain as fallbacks and are not reached in practice.

Measured against the routes it replaces (`cargo test --profile reltest --lib
response::cpxtb::tests -- --ignored --nocapture`):

| fixture | `npair` | previous route | charge-space route | speedup |
|---|---|---|---|---|
| distorted Ni(CO)₄, 3000 K | 679 | dense LU 0.23–0.26 s | 0.002–0.015 s | **16–132×** |
| water 2×2×2 | 1024 | dense LU 0.45–0.67 s | 0.011–0.026 s | **18–62×** |
| water 3×3×2 | 5184 | dense LU 22.1–39.7 s | 0.18–0.27 s | **126–147×** |
| water 3×3×3 | 11664 | PCG 16.9–22.1 s | 0.56–0.85 s | **26–30×** |

(Ranges are two runs on a machine with varying background load; the *dense/PCG*
columns are the stable ones, so the low end of each ratio is the safe figure.)

The Z-vector / adjoint solve the quartic 2n+1 ladder issues
(`response_stage.rs`, `tol = 1e-11`) inherits the same factorization —
`CpxtbSetup` builds the reduction once and `solve_adjoint` reuses it. On
distorted Ni(CO)₄ at 3000 K that solve went from `0.120 s` (a CG that bails out
and is rescued by a fresh dense operator build) to below the timer resolution;
on well-gapped fixtures, where the CG converges in ~9 iterations, it is a wash.

Accuracy is unchanged — both direct routes sit at round-off, and the reduction
and the dense LU agree on the amplitudes to `~1e-15` relative. What changes is
**robustness where finite temperature bites**. On distorted Ni(CO)₄ at 3000 K the
near-degenerate occupied–virtual pairs (gaps down to `3.6e-7`) defeat the Krylov
route outright: the preconditioned CG averages **356 iterations for 679
unknowns** and hits its cap, so it never reaches `1e-9` on its own and survives
only because the dense fallback rescues it. Its amplitudes are ~9 digits, against
round-off for either direct route. The reduction needs **zero** iterations.

Two conditioning ideas were implemented, measured and **rejected** — both made
the CG *worse* on every fixture, and both are kept only as documented negative
results in the test module:

* the **true** Jacobi diagonal `gap + ½(f_i − f_a)·(q^{ia})ᵀK q^{ia}` instead of
  the bare gap: `+11 %` to `+75 %` iterations;
* seeding each geometric RHS with the previous DOF's amplitudes: `+5 %` to
  `+90 %` iterations (the `x/y/z` amplitude vectors of one atom are near
  orthogonal, so a neighbour's solution is a worse start than zero **and** it
  discards the Krylov space).

### API

```rust
use gfn1_rs::response::charge_space::ChargeSpaceContext;

let ctx = ChargeSpaceContext::build(&system, &params, &electronic)?;
ctx.nshell();                  // nsh
ctx.is_finite_temperature();   // whether the fractional-occupation channels are live
ctx.kernel;                    // pub: K = γ + ∂²E_onsite/∂q²   (nsh × nsh)
ctx.chi0;                      // pub: static shell-charge susceptibility χ⁰
```

First order (one perturbation `x`):

```rust
pub fn solve_first_order(&self, fock_skeleton: &Matrix, overlap_deriv: &Matrix)
    -> Result<FirstOrderBundle>
pub fn first_order_field(&self, fock_skeleton: Matrix, overlap_deriv: Matrix)
    -> Result<FirstOrderField>
```

`FirstOrderBundle` carries the fully screened `{ density: P^x, energy_weighted: W^x,
shell_charges: q^x, occupation_response: f^x, screened_potential: K q^x }`.
`FirstOrderField` wraps that bundle plus the cached MO-side representations
(`fock_skeleton`, `overlap_deriv`, `h_tilde`, `s_tilde`, `eps_response`,
`u_rotation`) that the second-order solve needs.

Second order (a perturbation **pair** `x, y`):

```rust
pub fn solve_second_order(&self, x: &FirstOrderField, y: &FirstOrderField,
                          fock_skeleton_xy: &Matrix, overlap_xy: &Matrix,
                          dgamma_y_qx: &[f64], dgamma_x_qy: &[f64])
    -> Result<SecondOrderBundle>
pub fn second_order_field(&self, /* same arguments */) -> Result<SecondOrderField>
```

with the **same factored dielectric** serving the new right-hand side:

```text
(I − χ⁰ K) q^xy = q̃^xy
q̃^xy = −Tr_s(P^xy_ext S) − Tr_s(P^x S^y) − Tr_s(P^y S^x) − Tr_s(P₀ S^xy)
```

`P^xy` assembles the frame rotations `(U^y c + c U^yᵀ)`, the coefficient formula
applied to the derivative MO representations (`h_dot`, `s_dot`), its explicit-ε
correction (including the gauge-invariant in-block average for the energy-weighted
degenerate branch), the on-site `dK/dq` chain (full anharmonic `E'''` at reference
charges, `charge_order`-aware), the geometric kernel derivatives, and **both**
symmetry-mirrored reference-potential/screening cross channels
`RF_S((d_y γ)q^x) + RF_S((d_x γ)q^y) + RF_{S^y}(K q^x) + RF_{S^x}(K q^y)` —
omitting the mirrored pair was the initial 7e-3 FD failure, caught by the gate.

`SecondOrderBundle` = `{ density: P^xy, energy_weighted: W^xy, shell_charges: q^xy,
occupation_response: f^xy }`; `SecondOrderField` adds `h_dot`, `s_dot`,
`eps_second`, `u_second`, and the FIXED-basis dots `h_dot_fixed` / `s_dot_fixed`
(the third-order solve's ingredients).

Third order (directional, `x = y = z = v`):

```rust
pub fn solve_third_order_directional(&self,
    v_field: &FirstOrderField, vv_field: &SecondOrderField,
    fock_skeleton_vvv: &Matrix, overlap_vvv: &Matrix, overlap_vv: &Matrix,
    v_pot_geo: &[f64], dgamma_v_qv: &[f64], dgamma_v_qvv: &[f64],
    d2gamma_vv_qv: &[f64]) -> Result<ThirdOrderBundle>
```

the exact `λ`-derivative of the second-order field along the same `v`, again on
the SAME factored dielectric. Ingredients: the MO-representation recursion
`s̈ = C†S³C + frame(U, C†S²C) + frame(U̇, s̃) + frame(U, ṡ)` (same for `ḧ`), the
occupation third response `f''' δε³ + 3f'' δε δε² + f'(ε^{vvv} − μ^{vvv})` with
third-order particle-number conservation, the onsite `E''''` chain, and the
coefficient third derivative `base(ḧ, s̈, f^{vvv}) + 2Δ𝒞(ḣ, ṡ, f^{vv})
+ Δ𝒞(ref'') + Δ²𝒞_quad` — the last term computed EXACTLY by dual-number
lifting of the reference-motion correction (no hand-derived second chains).
The caller supplies the frozen directional third skeleton (geo legs) plus the
CN cache motion of the bare-H0 second (the affine self-energy trick); the
solver adds the symmetric-D³ response-potential channels including the
`RF_{S^{vv}}(V^v_geo)` completion.

Gates (`run_third_order_gate`): FD of the second-order bundle along `v`,
everything reconverged, `h²` ladder — T = 0 water `1.6e-10 / 3.2e-10 / 1.9e-10`
(ratios 4.01/4.00/4.01); Fermi-smeared distorted Ni(CO)₄ at 3000 K
`2.1e-7 / 2.1e-7 / 3.1e-7` at `h = 2e-3` (ratios 4.00/4.00/4.00 — the larger
step keeps the reconvergence noise floor, which grows as `1/h`, below the
truncation).

Gates: first order at `T = 0` vs the MO-pair CPXTB bundles `≤ 2e-9` on `P`/`W`/`q`
for all 9 DOFs of water; second order vs a central FD of the screened first-order
fields on non-equilibrium water over 5 `(x,y)` pairs: `3.4e-9 / 5.4e-9 / 3.0e-9`.

---

## 2. Native finite temperature

Fractional occupations are not a special case bolted on — the unified
response-coefficient formula covers them:

- orbital rotations with `(f_p − f_q)/(ε_p − ε_q)` weights,
- the metric channel,
- the grand-canonical **occupation channel** with the chemical-potential shift
  fixed by particle-number conservation.

At `T = 0` with integer occupations the occupation channel vanishes and the same
formulas reduce **exactly** to classic CPXTB; the `T = 0` code path stays
bit-identical.

### First order: the `f'` channel

`fermi_occupation_response` supplies `f^x`, with `μ^x` from particle-number
conservation. A (near-)zero occupied–virtual gap with **integer** occupations is
rejected at build time (the response really is singular there — enable Fermi
smearing), mirroring the CPXTB guard.

### Second order: the `f''` chain and `μ^xy`

Outside degenerate blocks the scalar chain is

```text
f^(xy)_p = f'' · δε^x_p · δε^y_p + f' · (ε^(xy)_p − μ^(xy)) ,    f''_p = w_p(1 − f_p)/kT
```

with the **second-order chemical-potential shift `μ^(xy)`** fixed by
particle-number conservation (gated to `4e-16`). On top of that sits the `Δ𝒞_T`
reference-motion correction of the finite-T coefficient formula (diagonal,
quotient and degenerate-slope branches) — the exact companion of the base formula
applied to the derivative inputs.

A branch-consistency fix landed with it: `u_rotation` / `u_second` now use the
finite-T coefficient formula's `1e-10` degeneracy threshold when smeared. The old
`1e-6` / `1e-10` mismatch left near-degenerate pairs (gaps `2e-8 .. 4e-7`) with a
broken frame-rotation/coefficient cancellation — a **flat** `1.5e-6` residual that
the `h`-ladder correctly flagged as a missing term. With the branches aligned the
FD gate closes at `P 5.5e-9 / W 7.0e-9 / q 9.7e-9` on distorted Ni(CO)₄ at 3000 K.

### The bug this replaced

The old finite-temperature CPXTB branch was a **50-iteration, 0.35-damped fixed
point with no convergence check after the loop**. Under strong screening it
**silently returned unconverged iterates**: on Ni(CO)₄ at 3000 K its shell-charge
responses were off by `O(1)` (max 2.28) against the reconverged-SCC finite
difference, while the direct dielectric solve agrees with that FD ground truth to
`1.0e-8` and satisfies its own fixed point to `3e-16`. The branch now routes
through the charge-space solver, and the three-way comparison is a permanent
regression test.

The **periodic** Gamma and k-point Hessians carried the identical pattern and are
now fixed the same way: `PeriodicChargeDielectric` in `src/pbc/hessian.rs` builds
`χ⁰` from `nsh` unit-shell-potential responses (Γ, or the k-summed complex
response — the susceptibility is real `nsh × nsh` either way) and LU-factors
`I − χ⁰K` once per Hessian call for all `3N` DOFs. The periodic legacy iteration
was off by up to `2e22` on a Ni cell at 3000 K, where its damped iteration matrix
had spectral radius 4.55; the direct solve now matches the FD of the analytic
periodic gradient to `5.3e-11` (Γ) and `6.2e-12` (`2×2×2` k-mesh). See
[pbc.md](pbc.md#5-periodic-finite-temperature-fermi-smearing) and
[limitations.md](limitations.md#periodic-finite-temperature-response-singular-dielectric--rejected).

---

## 3. Analytic finite-temperature FC3

```rust
gfn1_rs::third_derivative::finite_t::directional_third_finite_t(
    &system, &params, &options, coordination_cutoff, &v,
) -> Result<f64>
```

The full directional cubic `e³[v] = D_v[H[v,v]]`, assembled by the product rule
over the directional Hessian's composition:

```text
D_v[r2[v,v]] = g(X^vv)                    (response motion — the second-order legs)
             + path_hessian(X^v)[v,v]      (geometric motion of g)
             + background_motion(X^v, X^v) (reference-state motion)
```

The background motion collects the non-geometric reference dependencies of `g`:
`P₀ → P^v` under the screening shift, `V₀ → V^v` under the response density, the
kernel motion `∂K/∂q·q^v` (on-site `E'''` chain), and the shell-charge motion
`q₀ → q^v` in the kernel-gradient bilinear.

Every ingredient is **occupation-agnostic** — the frozen blocks read the
fractional `P/W/q` reference and the response legs come from the finite-temperature
charge-space solver — so one code path serves `T = 0` and smeared systems alike.

Gates:

- **`T = 0` equality** against the adjoint-assembled FC3
  (`third_derivative_analytic_vector`): `2.3e-15`, for both the response-derivative
  alone and the full cubic;
- **Fermi-smeared** distorted Ni(CO)₄ at 3000 K vs the FD of the finite-T analytic
  Hessian: `δ(h) = 6.25e-10`, `δ(h/2) = 1.57e-10`, ratio 3.98 (pure `h²`). This
  test is `#[ignore]`d by default — four finite-T Hessians cost ~10 min; the rerun
  command is in the test doc comment.

### Dense / block finite-temperature FC3

```rust
gfn1_rs::third_derivative::finite_t::third_derivative_finite_t_dense(
    &system, &params, &options, coordination_cutoff,
) -> Result<SymmetricThird>
gfn1_rs::third_derivative::finite_t::third_derivative_finite_t_block(
    &system, &params, &options, coordination_cutoff, &dofs,
) -> Result<SymmetricThird>   // indexed by POSITION in `dofs`
```

The full packed tensor, recovered from shared-reference directional
evaluations by the cubic polarization identity
`T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]` — the same
pattern as the mixed-index FC4 driver. One SCF/CPXTB/charge-space
factorization and the direction-independent frozen third slabs are computed
once ([`FiniteTThirdReference`]); the ~`C(n+2,3)` deduplicated subset
directions are evaluated in parallel (rayon). For large systems prefer the
block or directional modes. Gates: `T = 0` element-wise equality against
`third_derivative_analytic_dense` (`2.2e-13`); Fermi-smeared scalene H₃
(fractional occupations, no symmetry degeneracy) against the seminumerical
dense reference with the `h²` ladder (`ratio 4.00`); block-vs-dense
consistency.

**Not covered:** exactly degenerate *and* fractionally occupied blocks, which the
second-order solver rejects explicitly (see below). The strict closed-form
dense/vector/block FC3 (`third_derivative_analytic_*`) still rejects fractional
occupations outright — use the `finite_t` drivers above for smeared systems.

---

## 3b. Analytic finite-temperature FC4

```rust
gfn1_rs::third_derivative::finite_t::directional_fourth_finite_t(
    &system, &params, &options, coordination_cutoff, &v,
) -> Result<f64>
gfn1_rs::third_derivative::finite_t::fourth_derivative_finite_t_dense(
    &system, &params, &options, coordination_cutoff,
) -> Result<SymmetricFourth>
gfn1_rs::third_derivative::finite_t::fourth_derivative_finite_t_block(
    &system, &params, &options, coordination_cutoff, &dofs,
) -> Result<SymmetricFourth>   // indexed by POSITION in `dofs`
```

The directional quartic is the product rule over the finite-T cubic's
composition, `e⁴[v] = D_v[e³[v]]`, with the third-order response `X^{vvv}`
supplying the top leg and the `(d)`-motion assembled as

```text
D_v[(d)] = g(X^vvv) + 2·B(X^vv) + 3·G(X^vv, X^v) + ∂B(X^v) + ∂G(X^v, X^v)
```

where `B` is the six-block coefficient motion, `G` the background families,
and the eigen-motion inventories `∂B`/`∂G` (block cache motions — the cn/CN
caches, the s2 first-slot, the pulay potential/CN caches, the `so_q` density
slot — and the per-family background eigen-motions) were pinned block-by-block
and family-by-family with dedicated FD split diagnostics (kept `#[ignore]`d
in the test module). The dense/block drivers reuse the cubic pattern one
order up: the quartic polarization identity
`Q(x₁..x₄) = (1/24) Σ_{∅≠S⊆{1..4}} (−1)^{4−|S|} e⁴[Σ_{i∈S} x_i]` over one
shared reference with ~`C(n+3,4)` deduplicated directions in parallel.

Gates:

- **`T = 0` equality** vs the validated five-stage quartic
  (`directional_fourth_derivative`) on non-eq water **and** skew HF:
  `2.9e-15` / `1.5e-16`;
- **Fermi-smeared** scalene H₃ at 3000 K (fractional occupations) vs the
  Richardson-extrapolated FD of `directional_third_finite_t`: `6.6e-12`
  with the `h²` ladder ratio at 3.99;
- dense `T = 0` element-wise equality vs `fourth_derivative_analytic_dense`
  (`7.5e-14`) and a smeared 4-dof block contraction vs the directional value
  (`9.8e-13`).

Like the FC3 stack, every ingredient is occupation-agnostic; the same
exact-degeneracy exclusion applies.

---

## 4. What finite temperature does *not* reach yet

| Path | Smearing |
| --- | --- |
| SCC energy, gradient / forces (molecular and periodic) | ✅ |
| Analytic Hessian (molecular) | ✅ |
| Seminumerical FC3 (`third_derivative_seminumerical_*`) | ✅ (free-energy cubic constants) |
| Analytic directional FC3 (`directional_third_finite_t`) | ✅ (except exact degeneracy) |
| Analytic dense/block FC3 (`third_derivative_finite_t_dense` / `_block`) | ✅ (except exact degeneracy) |
| Analytic FC3 dense/vector/block, `T = 0` closed form (`third_derivative_analytic_*`) | ❌ explicit error — use the `finite_t` drivers |
| Analytic directional/dense/block FC4 (`directional_fourth_finite_t`, `fourth_derivative_finite_t_dense` / `_block`) | ✅ (except exact degeneracy) |
| Analytic FC4, `T = 0` closed form (`directional_fourth_derivative`, `fourth_derivative_analytic_*`) | ❌ explicit error — use the `finite_t` drivers |
| Analytic polarizability (`static_polarizability`) | ❌ — use `static_polarizability_finite_field` |
| Periodic Hessian at `T > 0` (Γ and k-point) | ✅ direct charge-space dielectric solve — see [pbc.md §5](pbc.md#5-periodic-finite-temperature-fermi-smearing) |
| Periodic FD stack at `T > 0` (`pbc_third_derivative_seminumerical_*`, `pbc_strain_hessian_derivative`, `pbc_gruneisen`) | ✅ — tighten `charge_tolerance` first, see [pbc.md §5](pbc.md#5-periodic-finite-temperature-fermi-smearing) |

The exact error strings are quoted in [limitations.md](limitations.md).
