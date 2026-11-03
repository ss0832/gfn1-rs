# Nuclear derivatives: gradient → Hessian → FC3 → FC4

This page covers the non-periodic analytic derivative ladder. The periodic
counterparts live in [pbc.md](pbc.md); Fermi smearing and the response solver that
makes it work live in [finite-temperature.md](finite-temperature.md); the honest
list of what is *not* implemented is in [limitations.md](limitations.md).

---

## 0. The term registry: `require_order`

`src/terms.rs` is the **single source of truth** for which analytic derivative
order each energy term implements. Every derivative driver calls
`terms::require_order(&electronic_options, &params, order, context)` before doing
any work, so an option set whose terms are not implemented at the requested order
fails fast with a uniform message instead of silently returning derivatives of a
*different* energy expression (the pre-v0.5.0 failure mode of the analytic
Hessian).

```rust
pub fn require_order(
    options: &ElectronicOptions,
    params: &Gfn1Parameters,
    order: u8,
    context: &str,
) -> Result<()>
```

Current coverage (`terms::active_terms`):

The order-4 capability check additionally requires
`hamiltonian.enable_cn_hamiltonian = true`. Both strict and finite-temperature
analytic FC4 reject CN-disabled calculations before SCF, because their frozen
CN stages have not been derived for that modified Hamiltonian. This restriction
does not lower the order-1/2/3 capability checks.

| Term | Active when | `max_analytic_order` |
| --- | --- | --- |
| `repulsion` | always | **4** |
| `H0 band + Pulay + CN chain` | always | **4** |
| `isotropic SCC (2nd/3rd/higher charge orders)` | always | **4** |
| `D3(BJ) two-body dispersion` | `enable_dispersion && !experimental_d4` | **4** |
| `D3 ATM (three-body) dispersion` | above **and** `s9 != 0` | **4** |
| `halogen bond` | always | **4** |
| `experimental D4 dispersion` | `experimental_d4` | 1 |
| `multipole (mDFTB2/CAMM) electrostatics` | `multipole` | 1 |
| `long-range Fock exchange (MFX/OFX)` | `lr_exchange` | 1 |
| `DFT+U/+U+V` | `plus_u` | 1 |
| `spin polarization (spGFN1)` | `spin_polarization` | 1 |
| `external electric field` | `external_field.electric_field.is_some()` | 1 |
| `GFN1 ALPB/GBSA` | `solvation` | 4 |
| `structural ML Hamiltonian` | `ml_model` | **3** molecular canonical free energy; **2** periodic DFPT |

So **stock GFN1 passes `require_order(4)`** (and fails at order 5 — nothing
implements order 5). Most experimental flags cap the ladder at the gradient;
the separately dispatched molecular ML path supports Hessian and FC3.

Fermi smearing is deliberately **not** a registry row. Standard molecular
FC3/FC4 entry points select canonical analytic free-energy response at positive
electronic temperature, including fractional/degenerate occupations; the T=0
integer-occupation backend is retained. Temperature is never changed by dispatch.

Call sites: `hessian` (order 2), `third_derivative` (order 3),
`third_derivative::finite_t` (order 3), `fourth_derivative` (order 4).

---

## 1. Gradient

`analytic_gradient(&system, &params, AnalyticGradientOptions)` →
`AnalyticGradientResult { gradient, forces, electronic_result }`. See
[rust-api.md](rust-api.md#entry-points-by-task).

## 2. Hessian

### Reduced two-body D3 chain

For stock GFN1 (`s9=0`), the two-body D3 Hessian is assembled exactly in
(Cartesian R, atomic CN) space. Each interpolated C6 pair stores six scalar
partials in two CN variables; normalized reference weights retain their full
first/second derivatives. The final chain is
`H_RR + J_CN^T H_CN,CN J_CN + H_R,CN J_CN + transpose + sum_i E_CN_i H_CN_i`.
Radial pair and self-image counting match the original energy. This avoids
the full-Cartesian `nat*nat` C6 jet table, which alone would store about
10.5 GB of Hessian doubles at 110 atoms. CN jets and compact C6 partials
replace that table; this is an allocation estimate, not an RSS measurement.
Independent full-geometry jets remain the ATM route and regression reference.
Mixed-element molecules, periodic water and self-images agree with that
reference, and the existing gradient/Hessian/FC3/FC4 D3 gates remain active.

`analytic_hessian(&system, &params, AnalyticHessianOptions)` →
`AnalyticHessianResult`. `AnalyticHessianOptions` toggles each block
(`include_repulsion`, `include_fixed_scc`, `include_fixed_pulay`,
`include_fixed_cn_h0`, `include_electronic`, `include_dispersion`,
`include_halogen`) and carries the nested `electronic_options`. The same struct is
the options type for the third **and** fourth derivative drivers.

Since v0.5.0 `analytic_hessian` / `analytic_hessian_from_result` **reject** option
sets whose second-derivative terms are unimplemented (multipole, `lr_exchange`,
`plus_u`, `spin_polarization`, `experimental_d4`, external electric field) rather
than silently dropping them.

---

## 3. Third derivative (cubic force constants, FC3)

`T_abc = ∂³E/∂R_a∂R_b∂R_c`. Two routes, both non-periodic.

### Shared-reference direct contractions and structural ML

`third_derivative::CubicReference::build(system, params, electronic, budget)`
builds one converged canonical reference and charge-space response
factorization. `directional(v)` returns `E'''[v,v,v]`; `mixed(u,v,w)` returns
`E'''[u,v,w]` by polarization; `directional_derivatives(v)` returns E through
E''' along v. At T=0 it evaluates an overlap-normalized occupied trial space.
At positive T it uses all metric-orthonormalized MOs, the full first
occupation-matrix response and analytic entropy-trace divided differences.
Off-diagonal occupation response is retained inside fractional degenerate
blocks. Both use first-order CP-SCC and the variational 2n+1 rule. The direct route does not
materialize Cartesian FC3 slabs; its workspace estimate is checked separately
from tensor storage. The legacy non-ML vector API retains its slab-based
electronic assembly, so use `CubicReference` for direct scalar contractions.

Structure-local ML level jets, their H0/CN coupling and the correction energy
are included consistently through order three. `analytic_hessian` and standard
analytic FC3 dense/vector/block entry points dispatch to this complete ML
energy path, including fractional occupations. Partial-term ML Hessian options
and ML FC4 remain rejected. `analytic_hessian_from_result` does not reconstruct this ML
reference and retains its guard. Python exposes `third_derivative_contractions`
for batched directional cubics with a shared reference.

The reusable `tests/profiling/profile_contracted_fc3.rs` driver compares full
FC3 contraction with the shared directional reference on water/NH3/CH4 and
reports stage timings, allocation counts and numerical differences. Run
`cargo test --profile reltest --test profile_contracted_fc3` with both
`RAYON_NUM_THREADS=1` and `GFN1_FAER_THREADS=1` for a serial comparison.
Memory counters measure additional live Rust allocations per stage; a reused
reference's retained storage is separate from a direction's extra bytes.
The old timing table's external receipt is not included in this source base,
so it is not presented as a v0.7.0 performance measurement.

### 3.1 Strict closed form (2n+1, no finite differences)

Crate-root re-exports (`use gfn1_rs::{...}`):

```rust
third_derivative_analytic       (system, params, options, coordination_cutoff) -> Vec<Matrix>
third_derivative_analytic_dense (system, params, options, coordination_cutoff) -> SymmetricThird
third_derivative_analytic_vector(system, params, options, coordination_cutoff, v: &[f64]) -> Matrix
third_derivative_analytic_block (system, params, options, coordination_cutoff, atoms: &[usize])
                                                             -> (Vec<usize>, Vec<Matrix>)
```

- **Dense (`third_derivative_analytic`)** — `ndof` dense slabs, `slab[c][(a,b)] = T_abc`.
- **Dense packed (`_dense`)** — the same data as a `SymmetricThird`
  (`n(n+1)(n+2)/6` entries, ~1/6 the memory).
- **Vector (`_vector`)** — the directional contraction `K[a][b] = Σ_c v_c T_abc`
  as one `3N×3N` matrix; never materialises `ndof³`.
- **Block (`_block`)** — the `O(|block|³)` sub-tensor over the DOFs of chosen atoms.

`SymmetricThird` methods: `zeros`, `n`, `len`, `is_empty`, `add`, `get`, `scale`,
`add_from`, `to_dense_slabs`, `contract_vvv`, `contract_last`, `block`.

The response slabs are computed in parallel over the shared `rayon` pool.

**Two bugs fixed in v0.5.0** (both silently wrong before, both now regression-gated):

1. **`dK/dq` kernel chain.** The response kernel `K = γ + 2Γ_A q_A` is
   charge-dependent, but `D_c` of a kernel action on a fixed response-charge
   vector only carried `(dγ/dR_c)·u + K·(D_c u)`; the on-site anharmonicity piece
   `2Γ_A q_A^(c) Σ_{shells of A} u` was missing at four sites. Measured
   improvement vs the seminumerical reference: methane `1.1e-5 → 2.0e-8`, HF dimer
   `3.0e-5 → 3.4e-7`, stretched water `1.7e-6 → 5.6e-8`.
2. **Degenerate orbitals.** `mo_coefficient_derivatives` left degenerate
   same-block rotations at zero, violating first-order orthonormality
   `U_pq + U_qp = −S̃_pq`, and a gauge-dependent per-orbital `ε^(c)` leaked into
   four contractions. Fixed with the symmetric gauge `U_pq = −½S̃_pq` plus the
   gauge-invariant in-block matrix `Λ^c_pq = F̃^c_pq − ε S̃^c_pq`. Symmetric
   molecules were ~2e-2 relative off: NH₃ `2.4e-2 → 7.8e-8`, T_d CH₄
   `2.9e-2 → 1.9e-8`.

Additionally, the CPXTB response kernel used to truncate the on-site block at the
DFTB3 `2Γq` term, so with `charge_order >= 4` every response property (Hessian,
polarizability, TDA, dipole derivatives) was silently inconsistent with the
energy. `response_shell_scc_kernel` now adds the full `∂²E/∂q²` on-site block and
the third-derivative `dK/dq` chain uses the full `∂³E/∂q³` (reducing to `2Γ` at
`charge_order = 3`).

**Accuracy.** Analytic vs seminumerical: ~5e-7 on non-equilibrium stretched+bent
water with the third-order on-site Γ active; ~1e-4 vs a double-gradient FD at
strongly non-equilibrium geometries. Use a tight SCF (`energy_tolerance 1e-11`,
`charge_tolerance 1e-9`).

**Frozen blocks** are individually exported and FD-isolatable:
`third_derivative_geometric`, `third_derivative_dispersion`,
`third_derivative_frozen`, `third_derivative_frozen_full`,
`third_derivative_frozen_complete`, plus
`gfn1_rs::third_derivative::third_derivative_frozen_electronic` (this one is
**not** re-exported at the crate root).

### 3.2 Seminumerical (production fallback, smearing-capable)

```rust
third_derivative_seminumerical_dense (system, params, options, step) -> SymmetricThird
third_derivative_seminumerical_vector(system, params, options, v, step) -> Matrix
third_derivative_seminumerical_block (system, params, options, atoms, step)
                                                             -> (Vec<usize>, Vec<Matrix>)
```

A central finite difference of the FD-validated analytic Hessian. The **Vector**
mode is the cheapest — a directional cubic constant along `v` costs exactly **two**
Hessian evaluations. Block mode computes only `|dofs|` Hessian pairs and is
bit-for-bit the corresponding sub-block of the Dense result.

This route **supports Fermi smearing transparently**: with
`electronic_temperature > 0` the Hessian it differentiates is the electronic
free-energy Hessian, so the result is the free-energy cubic force constant.

### 3.3 Finite-temperature analytic FC3

```rust
gfn1_rs::third_derivative::finite_t::directional_third_finite_t(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,   // note: by reference
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64>                        // the scalar e³[v] = Σ_abc T_abc v_a v_b v_c
```

Fully analytic, **natively Fermi-smeared**, directional. Route:
`e³[v] = D_v[H[v,v]]` assembled by the product rule over the directional Hessian's
two halves. Because the directional mode can use the second-order screened bundle
`X^vv` from the charge-space solver directly, no adjoint/Z-vector machinery is
needed and the T = 0 orbital algebra is bypassed entirely. See
[finite-temperature.md](finite-temperature.md#3-analytic-finite-temperature-fc3).

Not re-exported at the crate root, and no CLI or Python binding.

### 3.4 Front ends

- CLI: `--third-derivative` (alias `--cubic`) prints Cartesian slabs in Hartree/bohr³ for molecules and supported periodic references.
  **Non-periodic only** — a lattice-bearing input is rejected.
- Python: `third_derivative`, `third_derivative_vector`, `third_derivative_block`
  (closed form) and `third_derivative_along` (seminumerical directional);
  ASE: `get_third_derivative*`.

---

## 4. Fourth derivative (quartic force constants, FC4)

`--fourth-derivative` (alias `--quartic`) prints dense molecular components in
Hartree/bohr⁴ using zero-based Cartesian indices.

`Q_abcd = ∂⁴F/∂R_a∂R_b∂R_c∂R_d`. Non-periodic; standard entry points
support both integer T=0 occupations and the canonical thermal ensemble.

The names are **not** re-exported at the crate root; import them from the module:

```rust
use gfn1_rs::fourth_derivative::{
    directional_fourth_derivative, directional_fourth_with_reference,
    directional_fourth_seminumerical, fourth_derivative_analytic_dense,
    fourth_derivative_analytic_block, QuarticReference, SymmetricFourth,
};
```

### 4.1 Directional five-stage 2n+1 assembly (the flagship mode)

```rust
pub fn directional_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64>            // e⁗[v] = Σ_abcd Q_abcd v_a v_b v_c v_d
```

Each stage is the exact `λ`-derivative of one ingredient of the (already
validated) analytic third derivative along `R + λv`, so their sum is the
`λ`-derivative of the full third derivative contracted `vvv`:

1. **geometric** — repulsion + halogen + frozen SCC2 + D3, `L_λλλλ`
   (`directional::directional_fourth_geometric_with`), plus the frozen-SCC2
   bilinear charge path `d(scc2_third·vvv)/dq · q^v`;
2. **frozen-density Hamiltonian blocks** — Pulay / CN-H0 / scalar-overlap fourths
   plus the density path of their thirds and the Pulay-third CN response
   (`directional::directional_fourth_frozen_density`);
3. **FC3 density-path derivative** — needs the **second-order** screened bundle
   `X^vv` from the charge-space solver
   (`directional::directional_fourth_hessian_path_stage`);
4. **FC3 Pulay CN-response derivative** — needs the second directional
   coordination response `CN^vv`
   (`directional::directional_fourth_cn_response_stage`);
5. **2n+1 quartic response stage**
   (`response_stage::directional_response_fourth`).

By the 2n+1 rule the whole assembly needs only the first- and second-order
responses **along `v`** — one charge-space first-order solve and one second-order
solve — never the full `O(ndof)` / `O(ndof²)` response sets. Total cost per
direction: one SCF, one CPXTB solve, one first-order and one second-order
charge-space solve.

Gate result: `delta 2.4e-10` (dispersion off) / `6.9e-10` (on) vs the
seminumerical reference, with an exact `h²` ratio of 4.00.

#### Performance

Set `GFN1_PROFILE=1` for a per-stage `gfn1_profile_ms` breakdown
(`fc4.reference.*`, `fc4.legs.*`, `fc4.stage1..5.*`);
`cargo test --profile reltest --test profile_fc4` times the directional and
dense drivers on the gate geometries.

Every stage is now **one-pass directional**: no step of the directional quartic
materialises an `O(ndof³)`-or-larger derivative tensor just to contract it
against `v`. Three changes did that, each gated element-wise/scalar against the
code it replaced (agreement `1e-17`–`1e-19` absolute, i.e. summation-order
roundoff):

| what | why it was slow | now |
| --- | --- | --- |
| stage 5, skeleton second derivatives | `ndof²` per-`(b,c)` blocks × 7 matrix builds, each rebuilding the CN and shell-potential ladders | one AO-pair sweep per matrix, both legs contracted inside it |
| stage 2, frozen fourth blocks | nested `out[c][d][(a,b)]` store of `ndof⁴` doubles, then a `vvvv` contraction | `directional_fixed_density_{pulay,cn_h0,scalar_overlap}_fourth` return the scalar directly |
| stage 1, D3 + halogen | full-space `Jet4` (`ndof⁴` doubles per jet) | univariate `Jet1` along `v` (5 doubles per jet) |

Wall-clock per direction (`reltest`, non-equilibrium water and the CH3Br···OH2
stage-1 gate complex, dispersion on), reading the `gfn1_profile_ms` scopes each
change touched:

| scope | water, 9 DOF | CH3Br···OH2, 24 DOF |
| --- | --- | --- |
| `fc4.stage1.geometric` (D3 + halogen) | 34 → 0.4 ms | 3 715 → 4.8 ms |
| `fc4.stage2.{pulay,cn_h0,scalar_overlap}_fourth` | 372 → 234 ms | 3 947 → 3 275 ms |
| `fc4.stage5.skeleton_second` | 7 761 → 87 ms | 609 435 → 2 085 ms |
| **`fc4.direction.total`** | **9 713 → ≈1 100 ms** | **634 193 → ≈24 700 ms** |

i.e. **≈9×** on water and **≈26×** on the 8-atom complex. The wall-clock totals
were taken on a shared machine, so treat ±30 % on any single number as noise —
the changed scopes moved by 1.6×–292×, far outside it, while the stages nothing
touched (3 and 4) stayed put within that band.

The stage-2 speedup is the *least* interesting of the three in wall clock and
the most important in reach: dropping the nested store is what removes the
`O(ndof⁴)` (and, for CN-H0, `O(nat·ndof⁴)`) working set. At 45 DOF the nested
CN-H0 route alone would allocate ≈1 GB of jets before doing any arithmetic.

What is left is pair-integral bound: the leaders are now the skeleton THIRD
derivatives of stage 5 and the third-derivative block builders shared by stages
2 and 3, both of which already run one AO-pair sweep per object.

The one DIRECTION-INDEPENDENT object among them — the undoctored Pulay third
block, which stages 2 and 3 each subtract as the baseline of the Pulay `V`-shift
— is built once per geometry in `QuarticReference` instead of twice per
direction (`fc4.reference.pulay_third`). On the 24-DOF fixture that moves
`fc4.stage2.third_density_path` 3 650 → 2 717 ms and
`fc4.stage3.geometric_path` 3 200 → 2 177 ms for a one-off 775 ms in the
reference, i.e. ≈2.0 s off every direction; the mixed-index drivers amortise the
build over hundreds of directions. Everything else that is direction-independent
was measured and left alone as sub-10 %: `ResponseGradientContext` (40 ms), the
stage-5 adjoint sector (108 ms), the shell-potential third ladder (2.6 ms) and
the duplicate `ChargeSpaceContext` build inside stage 5 (1.7 ms).

`fourth_derivative_analytic_dense` on water (714 deduplicated directions over
rayon, one shared `QuarticReference`) measures **85 s**. Its directions carry at
most four non-zero components, so the per-`(b,c)` screens made them much cheaper
than a generic direction to begin with — the dense driver gains far less from
the one-pass work than the flagship directional mode does.

### 4.2 Mixed-index tensor via the polarization identity

```rust
pub fn fourth_derivative_analytic_dense(system, params, options, coordination_cutoff)
    -> Result<SymmetricFourth>
pub fn fourth_derivative_analytic_block(system, params, options, coordination_cutoff, dofs: &[usize])
    -> Result<SymmetricFourth>
```

The full mixed-index `Q_abcd` is recovered **element by element from the
directional quartic**, using the polarization identity for symmetric quartic
forms:

```text
Q(x₁,x₂,x₃,x₄) = (1/24) Σ_{∅ ≠ S ⊆ {1,2,3,4}} (−1)^(4−|S|) e⁗[Σ_{i∈S} x_i]
```

with `x₁..x₄` the Cartesian basis vectors of the quadruple (repeats allowed; a
repeat gives that DOF weight 2, 3 or 4). So the mixed-index tensor inherits the
directional assembly's correctness by construction — no separate derivation and
**no nuclear finite differences**. The 15 subset directions are deduplicated
across the whole build (a water tensor needs 714 distinct directions rather than
495·15 = 7425) and evaluated in parallel against one shared reference.

> **Signature asymmetry to watch.** `third_derivative_analytic_block` takes
> **atom** indices and returns `(Vec<usize>, Vec<Matrix>)`;
> `fourth_derivative_analytic_block` takes **DOF** indices and returns a
> `SymmetricFourth`.

`QuarticReference` is that shared, `v`-independent reference (SCF + CPXTB Hessian
response + charge-space context). Build it once and reuse it:

```rust
let reference = QuarticReference::build(&system, &params, &options, cutoff)?;
let e4 = directional_fourth_with_reference(&system, &params, &options, cutoff, &reference, &v)?;
let scf = reference.electronic();
```

The order-4 registry check and temperature-specific backend selection live in
`QuarticReference::build`, so scalar, dense, block, semidiagonal and reused
reference entry points all support smearing. The thermal backend includes the
third chemical-potential response and confluent spectral limits. The five-stage
description above applies to the retained T=0 backend; positive T uses the
analytic product-rule spectral assembly described in [finite-temperature.md](finite-temperature.md).

Gate results: reconstructed-tensor `vvvv` vs the directional quartic `6.8e-16`
(skew HF) / `4.5e-17` (water block); element-wise vs the seminumerical reference
`3.32e-6 → 8.31e-7` on halving `h`, ratio 4.00.

### 4.3 `SymmetricFourth` storage

Packed fully-symmetric rank-4 storage, mirroring `SymmetricThird` one order up.
Public methods: `zeros`, `n`, `len`, `is_empty`, `index`, `add`, `get`, `scale`,
`add_from`, `contract_last(v) -> SymmetricThird`, `contract_last2(v, w) -> Matrix`,
`contract_vvvv(v) -> f64`, `block(dofs) -> Vec<Matrix>`.

Memory for 300 DOF: **64.8 GB dense** (`8*n^4`) and **2.7543306 GB packed**
(`8*binomial(n+3,4)`), in decimal bytes. Packed output is not scratch memory.
Fallible `try_zeros` rejects size/allocator overflow. Analytic assembly uses
`QuarticAssemblyOptions`: a default 512 MiB budget, batches of 2,048
quadruples, up to four concurrent directions and an explicit workspace reserve.
Available threads and the workspace budget can lower the concurrency. The
assembly retains the packed output and lower-order directional values, with
bounded worker scratch. For FC4 the additional directional cache is cubic in
the number of selected coordinates. The estimate does not represent process RSS.
`fourth_derivative_analytic_block_with_limits` exposes custom limits.
`fourth_derivative_analytic_semidiagonal` computes mode-pair quartics directly,
without a full Cartesian FC4. References carry a stable electronic-model and
geometry identity; incompatible reuse is rejected.

### 4.4 Seminumerical fallback

```rust
pub fn directional_fourth_seminumerical(
    system, params, options, coordination_cutoff, v: &[f64], step: f64,
) -> Result<f64>
```

A central FD of the analytic third derivative along `v`. It is both the
verification reference for the analytic assembly, including Fermi-smeared
systems. The standard analytic FC4 routes also support smearing without FD.

### 4.5 Dispersion / halogen quartics

`dispersion_fourth_derivative(system, params, reference_path)` →
`DispersionFourthResult` (dense `ndof⁴`, two-body always plus ATM when
`s9 != 0`). The D3 ATM triple energy is written once against a shared jet op set
(`src/jets.rs`, `Jet2`/`Jet3`/`Jet4`) and instantiated at second, third and fourth
order, so `s9 != 0` no longer restricts dispersion to energy + gradient.

Both **full-tensor** quartics are capped at `MAX_FOURTH_DERIVATIVE_NDOF = 30` DOF
(**10 atoms**) with an explicit error, because a full-space `Jet4` stores `ndof⁴`
doubles and the D3 assembly keeps `O(nat)` of them alive.

```rust
pub fn dispersion_fourth_directional(system, params, reference_path, v: &[f64]) -> Result<f64>
pub fn halogen_fourth_directional(system, params, v: &[f64]) -> Result<f64>
```

carry **no cap**. A directional fourth derivative is the fourth Taylor
coefficient of `E(R + t·v)`, so one differentiation variable suffices: these run
the *same* generic pipelines — the same D3 coordination jets, reference weights,
streaming `C6`, two-body BJ term and ATM triples; the same per-triple halogen
expression — instantiated on `jets::Jet1` (value + four `t`-derivatives, five
doubles) instead of `Jet4`. The per-DOF geometry seeds contract against a
direction installed for the call by `jets::DirectionScope` (a thread-local, so
the polarization driver's rayon fan-out over directions is safe by
construction). Gated against `contract_vvvv` of the full tensor at `1e-12`
relative on systems small enough for both (measured `1e-19`).

**Consequence: `directional_fourth_derivative` is no longer capped.** Stage 1
uses the directional entry points and no other stage builds an `ndof⁴` object,
so the directional quartic — and therefore
`fourth_derivative_analytic_dense`/`_block`, which are assembled from it —
run at any system size the SCF itself can handle.
`tests/fourth_derivative_nocap.rs` pins this on a 12-atom (36 DOF)
bromoethanol···OH₂ halogen-bonded complex with dispersion and halogen active:
the full-tensor entry points still refuse it, while the directional quartic
matches the seminumerical reference with an `h²` ratio of 4.00
(`delta(h) 2.07e-8 → delta(h/2) 5.17e-9`). Measured further up, 15-atom
n-butanol (45 DOF) evaluates in ≈61 s.

What remains capped: the full-tensor `dispersion_fourth_derivative` /
`halogen_fourth_derivative` themselves, and the Python/ASE `n⁴`-expansion guard
that reuses the same constant — see [limitations.md](limitations.md).

### 4.6 Front ends

None. FC4 is a Rust-library API only — no CLI flag, no Python binding.

---

## 5. The verification-gate philosophy

### Full Cartesian revalidation

`tests/nuclear_derivative_audit.rs` checks every Cartesian component on its
declared molecular, periodic, SOC/ML, field, solvent, spin, experimental-model
and classical-term fixtures. Run the extended audit explicitly:

```bash
cargo test --profile reltest --test nuclear_derivative_audit -- --ignored --nocapture --test-threads=4
```

Set `GFN1_FAER_THREADS=1` and `RAYON_NUM_THREADS=2` for the recorded controls.
The default output directory is `target/validation/nuclear-derivatives`;
`GFN1_VALIDATION_DIR` overrides it. `GFN1_DERIVATIVE_AUDIT_CASE` selects one
diagnostic case within the selected test group; zero matched cases fail.

Order n is compared with central differences of the analytic order n-1 for
every displaced coordinate, at three halved steps. Thus a d-DOF quartic audit
compares all d^4 entries, including zeros and every index permutation. Numerical
tensors are neither symmetrized nor projected. Fine-step and Richardson values
must each satisfy the local per-component absolute/relative tolerance. Raw
analytic tensors, all three numerical tensors, error metrics, units, SCF controls
and the exact audit-driver hash are retained in each case's JSON record.
Unsupported higher orders are reported separately from successful comparisons.

The distributed summary is `tests/data/nuclear-derivatives-report.json`; its
SHA-256-linked `nuclear-derivatives-raw.json.gz` contains all numerical arrays
and the earlier failed measurements. Regenerate both from full-group output
directories, in replacement order, with `tests/validation/collect_derivative_audits.py
--input <initial-output> --input <periodic-output> --input <extended-output>`.
Diagnostic subsets cannot replace a complete group. The separate
`multipole-rank4-ph3-diagnostic.json` records the repaired 0 K PH3 state, all
12 gradient components and the earlier failed fallback attempt.

The v0.6.0 collection covers 42 cases and 48,345 Cartesian components. All
four groups are rerun with `private-gemm-x86 0.1.22`, including all 23
extended-model cases and PH3. Each case records the dependency lockfile hash;
the collector's `--require-current-lockfile` option rejects stale results.
Earlier failed measurements remain separate historical evidence. The backend
update comparison, independent matrix checks and complete regression gates
are in `tests/data/backend-update-audit.json`.

This explains the shape of the test suite, and is the standard any new derivative
order must meet before it lands.

**The ladder rule.** An analytic derivative of order `n` is gated against a
**central finite difference of the already-validated analytic derivative of order
`n−1`** — never against a double finite difference of order `n−2`, and never
against an energy FD. Concretely:

| Order | Gated against |
| --- | --- |
| gradient | FD of the energy |
| Hessian | FD of the analytic gradient |
| FC3 (`third_derivative_analytic_*`) | FD of the analytic Hessian (= `third_derivative_seminumerical_*`) |
| FC4 (`directional_fourth_derivative`) | FD of the analytic third (= `directional_fourth_seminumerical`) |

Each step reuses a reference that is itself gated one rung down, so an error is
localised to the rung that introduced it.

**The `h²` ladder.** A central difference has `O(h²)` truncation error, so the
residual between analytic and FD must **drop by a factor of 4 when `h` is
halved**. This is the load-bearing diagnostic:

- residual falling as `h²` (ratio ≈ 4.00) ⇒ the analytic expression is complete
  and the remaining gap is pure FD truncation;
- a **flat** residual across `h` and `h/2` ⇒ a **missing or double-counted term**,
  not numerical noise.

Real examples from the log: the finite-T FC3 gate closes at
`δ(h) = 6.25e-10`, `δ(h/2) = 1.57e-10`, ratio 3.98; the FC4 polarization gate at
`3.32e-6 → 8.31e-7`, ratio 4.00. Conversely, a *flat* `1.5e-6` residual is what
flagged the mismatched degeneracy thresholds (`1e-6` vs `1e-10`) in the
finite-temperature second-order response.

**Exact invariances** are gated separately and to machine precision, because they
have no truncation error at all:

- translational invariance / acoustic sum rule (a uniform shift gives zero) —
  e.g. `1.7e-12` on a `1.39 Eh/Bohr³` scale for the periodic FC3;
- full permutation symmetry of the dense tensor;
- contraction identities (vector mode vs dense contraction: exactly 0);
- α-independence of the periodic generalised-Ewald energy.

**Equality gates** pin two independent implementations against each other where
they must agree exactly. The finite-temperature FC3 at `T = 0` must equal the
`T = 0` adjoint-assembled FC3 (measured `2.3e-15`); the charge-space first-order
bundles at `T = 0` must equal the MO-pair CPXTB bundles (`≤ 2e-9` on `P`/`W`/`q`
for all 9 DOFs of water).

**Component isolation.** Every frozen (response-free) block is exported
individually so its own third derivative can be gated against the FD of its own
analytic Hessian. That is why `third_derivative_geometric`,
`third_derivative_dispersion`, `third_derivative_frozen_full` and the five FC4
stage functions are public.

**Where the gates live.** Fine-grained stage gates are `#[cfg(test)]` modules
inside `src/` (`src/third_derivative/tests.rs`, `src/hessian/tests.rs`, the
in-module tests of `src/fourth_derivative/`). The compact always-on integration
versions are in `tests/`:

| File | Gates |
| --- | --- |
| `tests/gradient_fd_probe.rs` | FD gradient probe (`GFN1_FD_TIGHT=1` tightens it) |
| `tests/hessian.rs` | analytic Hessian vs gradient FD |
| `tests/third_derivative.rs` | analytic FC3 vs seminumerical: non-eq water, degenerate NH₃, vector mode |
| `tests/fourth_derivative.rs` | directional FC4 vs seminumerical; mixed-index tensor self-consistency + element-wise |
| `tests/pbc_third_derivative.rs` | periodic seminumerical FC3, `dH/dlnV`, Grüneisen (diamond) |
| `tests/tblite_parity.rs` | opt-in external parity suite (`GFN1_TBLITE_BIN`) |

Tests resolve parameters with `Gfn1Parameters::resolve(None)`, so they run against
the bundled parametrization and **no longer silently no-op** when
`GFN1_XTB_PARAM` is unset.

### Independent full-component field audit

The extended `cotton_mouton` test compares every existing Cotton-Mouton component
for M0 and M1 with an independent mixed fourth derivative of total energy.
All 27 components per model passed on asymmetric water using three electric and
magnetic field steps and Richardson extrapolation. The maximum extrapolated
absolute discrepancies were 6.66e-4 (M0) and 3.35e-4 (M1) atomic units in the
updated Linux audit. The finest mixed energy differences amplify floating-point
roundoff; all three steps and the unchanged component tolerances are recorded.
This is a
finite-field response of the Mulliken dipole, not an analytic fourth-order
electronic response. The current `[k][i][j]` API covers
`d² alpha_ij / d B_k²`; it does not supply the full mixed-`B_k B_l` tensor or
an orientationally averaged experimental birefringence coefficient.
All components and step-ladder values are in
`tests/data/cotton-mouton-audit.json`.

Run the reproducible independent energy audit with:

```text
cargo test --profile reltest --test cotton_mouton -- --include-ignored --nocapture
```

### Real molecular VPT2 execution

`cargo test --profile reltest --test validate_vpt2` optimizes H2, D2, HF and
water and evaluates PES-only VPT2. It independently checks every normal-mode
cubic and semidiagonal quartic force-field component using three steps of
Hessian differences. The one-mode cases also match the closed perturbation
formula `omega + Phi4/(8 omega^2) - 5 Phi3^2/(24 omega^4)`.
Raw force fields, finite-difference arrays, masses and optimized geometries are
in `tests/data/vpt2-molecular-audit.json`. Fundamental wavenumbers (cm^-1):

| System | Harmonic | PES-only VPT2 |
| --- | --- | --- |
| H2 | 5014.81 | 4395.37 |
| D2 | 3547.68 | 3237.67 |
| HF | 3834.89 | 3604.79 |
| H2O | 1438.12, 3675.15, 3786.82 | 1381.96, 3531.11, 3612.96 |

The public CLI was checked with `examples/water.xyz --opt --opt-gtol 1e-8
--etemp 0 --vpt2-pes`, giving the same spectrum and a PES-only zero-point energy
of 0.02007480167455 Eh. These checks validate execution, units and perturbation
algebra within the GFN1 potential; Watson/Coriolis terms, anharmonic intensities
and experimental frequency accuracy remain outside this calculation.



### VPT2 reference geometry and solvation

`--vpt2-pes` tests stationarity on the electronic free-energy surface
selected for that invocation. Use the same charge, electronic temperature,
Hamiltonian and model when optimizing the reference geometry. An ALPB/GBSA
minimum generally has a nonzero gas-phase gradient. Repeat the solvent flags
when running the analytic VPT2 calculation.

```sh
gfn1-rs examples/water.xyz --alpb water --opt --opt-gtol 1e-7 --opt-output water_alpb_opt.xyz
gfn1-rs water_alpb_opt.xyz --alpb water --vpt2-pes
```

Molecular ALPB and GBSA support the total canonical free-energy Hessian, FC3
and FC4, including fractional electronic occupations. The variational
response differentiates the Born, SASA, H-bond, shape and CM5 terms with the
same geometry and SCC charges as the energy. The fourth derivative uses
second-order charge and density response in a normalized trial density.
Cartesian tensors and VPT2 mode force constants follow by polarization of
analytic directional derivatives. An exact switch of a Born or SASA branch
can be non-differentiable at higher order and is diagnosed when the selected
direction encounters the switch. The VPT2 stationarity and positive-curvature
checks also apply to the solvated surface.

The Cartesian audit uses 81 Hessian, 729 cubic and 6,561 quartic components
per condition. With a central displacement of `1e-4 bohr`, the maximum
absolute differences from the next-lower analytic derivative are:

| Water model | Electronic temperature | Hessian (Ha/bohr²) | FC3 (Ha/bohr³) | FC4 (Ha/bohr⁴) |
| --- | ---: | ---: | ---: | ---: |
| ALPB | 0 K | 5.29e-9 | 2.43e-8 | 1.14e-7 |
| ALPB | 30,000 K | 4.91e-9 | 2.52e-8 | 1.39e-7 |
| GBSA | 0 K | 5.30e-9 | 2.43e-8 | 1.14e-7 |
| GBSA | 30,000 K | 4.92e-9 | 2.51e-8 | 1.39e-7 |

At the supplied Ni(CO)₄ ALPB/water optimized geometry and 300 K electronic
temperature, the full 27×27 analytic Hessian agrees with differences of
analytic gradients to `2.31e-9 Hartree/bohr²` in WSL. Its stationary point
passes the projected Hessian check. The nonresonant PES-only VPT2 driver
then encounters an overtone/combination coupling with a denominator of
`-3.83e-5 Hartree` (magnitude about `8.4 cm^-1`). The default resonance
threshold is `10 cm^-1`. Treating those states together requires a GVPT2
polyad.


### Cartesian FC3 and FC4 assembly in v0.7.2

For complete molecular energies with matching coordination cutoffs, packed
thermal FC3 uses a stationary trial density through first order. Gas-phase FC4
up to 30 Cartesian coordinates and solvated FC4 use it through second order. The free-energy functional, including the
canonical entropy when active, is expanded analytically at the reference
geometry. The stage-by-stage response route serves larger gas-phase FC4
references and supported component selections. Zero-temperature gas-phase Cartesian FC3 uses the
direct closed-form tensor. Molecular thermal and periodic Cartesian FC3 share
the bounded packed polarization driver.

The cubic and quartic CLI print directly from the packed tensor in the existing
Cartesian index order. Thread and workspace limits apply to the assembly.
The analytic periodic nuclear derivative routes in this release reach FC3.


The variational reference tightens the SCC thresholds to at most `1e-12`
Hartree and `1e-11` in the charge residual. Near-degenerate thermal orbitals
are transported together with the Fermi matrix response.

Ubuntu WSL verification at 300 K used Richardson-extrapolated differences of
the analytic Hessian. The largest absolute difference includes every ordered
Cartesian component.

| System | FC3 components | Max FC3 difference (Hartree/bohr³) | FC4 components | Max FC4 difference (Hartree/bohr⁴) |
| --- | ---: | ---: | ---: | ---: |
| Water, gas | 729 | 4.27e-12 | 6561 | 2.37e-9 |
| Water, ALPB/water | 729 | 1.87e-12 | 6561 | 2.33e-9 |
| Ni(CO)4, gas | 19683 | 8.44e-8 | 531441 | 2.48e-5 |

With four workers, the water CLI benchmark changed from 1.68 s to 0.11 s
for gas FC3 and 78.39 s to 0.14 s for gas FC4. ALPB FC4 changed from
2.70 s to 0.15 s. Process peak RSS for these water runs was about 15 MiB.
The Ni(CO)4 FC3 run took 2.82 s and about 18 MiB peak RSS. For its FC4
check, a second Richardson extrapolation combined Hessian differences at
`2e-3`, `1e-3` and `5e-4` bohr. The maximum error across the 531,441
components passed the `1e-4 Hartree/bohr^4` test threshold.

A tighter WSL audit at the same geometry and 300 K used SCC energy and
charge tolerances of 1e-14 Hartree and 3e-14 e. It displaced all 27
Cartesian coordinates independently and differentiated analytic FC3.
Three Richardson levels with steps 0.008, 0.004, 0.002 and 0.001 bohr
gave a maximum absolute difference of 4.69e-6 Hartree/bohr⁴ and RMS
difference of 2.93e-7 across all 531,441 entries. A separate two-level
extrapolation of Hessian differences at 0.002, 0.001 and 0.0005 bohr
gave 5.04e-6 maximum difference with symmetry-related displacement pairs
reconstructed. On this fixture, the analytic reference itself moved by
1.22e-5 Hartree/bohr⁴ when the SCC tolerances were tightened from 1e-12
Hartree and 1e-11 e. The maximum difference stayed above 1e-8 across
the smaller steps and extra extrapolation levels.


### File-backed Cartesian derivatives

```sh
gfn1-rs examples/ni_co4.xyz --cubic-file ni-fc3.bin
gfn1-rs examples/ni_co4.xyz --quartic-file ni-fc4.bin
```

The file options stream only the canonical symmetry entries. File-backed FC4
accepts more than 30 Cartesian degrees of freedom. The terminal and expanded
Python-list output retain their 30-DOF limit. The configured memory budget
still covers the electronic reference, worker scratch and lower-order
direction caches. The file route reduces the batch size and worker count when
needed to fit that budget. Full FC4 still has quartic evaluation and file-size scaling.

A WSL run on propane (11 atoms, 33 Cartesian degrees of freedom) used the ASE
G2 geometry, ALPB water and a zero electronic temperature. With eight threads,
the CLI wrote all 58,905 packed entries in 35.58 s. Its process peak RSS was
18,132 KiB and the output was 471,264 bytes including the header.
The ALPB water surface uses a 230-point Lebedev grid. Displacements of 0.01
and 0.005 bohr crossed SASA branches for some coordinates. All positive
and negative Cartesian displacements up to 0.0025 bohr stayed in the
reference branches, so only those steps were used for local extrapolation.
At 0.0025, 0.00125 and 0.000625 bohr, central differences of independently
converged analytic FC3 along all 33 coordinates followed by two Richardson
levels gave a maximum absolute difference of 3.96e-9 Hartree/bohr⁴ and an
RMS difference of 5.56e-11 across all 1,185,921 ordered FC4 entries.

In Rust, pass `DerivativeStorageOptions { disk_storage: true, ..Default::default() }`
to `third_derivative_with_storage` or `fourth_derivative_with_storage`.
Both return `DerivativeOutput::File` with the path and shape. Use
`PackedDerivativeFile::open` and `get(&[a,b,c,d])` to read a component. Use
three indices for FC3. With `disk_storage: false`, the result is
`DerivativeOutput::Memory`.

Python native `third_derivative` and `fourth_derivative`, and ASE
`get_third_derivative` and `get_fourth_derivative`, take the keyword
`disk_storage=True`. Pass `output_path` to name the file or omit it for
a generated path in the operating system's temporary directory. Delete the
file when finished. Existing paths are protected. An incomplete write has
an invalid header and normal calculation failures remove the partial file.

```python
meta = atoms.calc.get_fourth_derivative(atoms=atoms, disk_storage=True)
packed = np.memmap(meta["path"], mode="r", dtype="<f8",
                   offset=meta["offset_bytes"], shape=(meta["packed_entries"],))
```

The file has an 8-byte magic (`GFN1FC3` or `GFN1FC4` followed by NUL), a
little-endian u64 DOF count, a little-endian u64 packed-entry count, and
little-endian f64 values starting at byte 24. Units are Hartree/bohr³ for
FC3 and Hartree/bohr⁴ for FC4, including files produced through ASE.
For sorted zero-based indices `a <= b <= c <= d`, the FC4 offset in the
payload is `a + C(b+1,2) + C(c+2,3) + C(d+3,4)`. FC3 uses the first three
terms. Here `C(n,k)` is a binomial coefficient.
