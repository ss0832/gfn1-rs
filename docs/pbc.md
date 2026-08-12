# Periodic systems (PBC)

The periodic code lives under `src/pbc/` and is isolated from the molecular
modules. All periodic entry points share the signature
`(system, params, options, pbc)`, where `PbcOptions` selects the k-mesh
(`KMesh::gamma` / `KMesh::monkhorst_pack`), the generalised-Ewald controls
(`EwaldOptions`), and the AO image cutoff.

For the SCC / gradient / stress / multipole details see
[rust-api.md](rust-api.md#pbc). This page collects the **periodic derivative
ladder**, including everything new in v0.5.0.

---

## 1. What exists

| Property | Entry point | Status |
| --- | --- | --- |
| SCC energy | `run_pbc_scc`, `run_electronic_pbc`, `pbc_electronic_result` | Γ and Monkhorst-Pack k-points |
| Gradient / forces | `pbc_analytic_gradient` | analytic, FD-gated |
| Stress | `pbc_stress` | component stress, analytic |
| Hessian | `pbc_gamma_hessian`, `pbc_kpoint_hessian` | analytic; k-point CPXTB via preconditioned CG |
| **Third derivative (Γ), seminumerical** | `pbc_third_derivative_seminumerical_dense` / `_vector` | v0.5.0; supports smearing and every option the Hessian does |
| **Third derivative (Γ), analytic** | `pbc_gamma_third_analytic_dense` / `_block` / `_vector`, `GammaThirdReference` | **v0.5.0, §2c** — no finite differences; Γ only, integer occupations, `~1e-7` residual |
| **Third derivative (k-point), seminumerical** | `pbc_kpoint_third_derivative_seminumerical_dense` / `_vector` | v0.5.0, §2b; supports smearing |
| **Third derivative (k-point), analytic** | `pbc_kpoint_third_analytic_dense` / `_block` / `_vector`, `KpointThirdReference` | **v0.5.0, §2d** — no finite differences; **any Monkhorst–Pack mesh**, integer occupations |
| **Strain-mixed `dH/dlnV`** | `pbc_strain_hessian_derivative`, `pbc_kpoint_strain_hessian_derivative` | v0.5.0 |
| **Grüneisen parameters** | `pbc_gruneisen` | v0.5.0, `q = 0`, frozen-ion; Γ or k-point Hessian |
| **Second-order Grüneisen** | `pbc_gruneisen` + `GruneisenOptions::second_order` | v0.5.0, same call; own step `delta_second` (2 extra Hessians) |
| Cell scaling helper | `scale_lattice_isotropic` | v0.5.0 |
| **Bulk polarization** | `pbc_berry_polarization` | v0.5.0, Berry phase (KSV k-strings + Resta single point) |

All the v0.5.0 names above — plus `SecondOrderStencil`,
`pbc_gamma_third_with_reference`, `GammaThirdReference`,
`pbc_kpoint_third_with_reference` and `KpointThirdReference` — are re-exported at
the crate root (`use gfn1_rs::{...}`), and also available as
`gfn1_rs::pbc::third_derivative::*` / `gfn1_rs::pbc::gruneisen::*`.

**Python bindings** exist for the periodic third derivatives, the strain
derivative and Grüneisen: `third_derivative_periodic`,
`third_derivative_periodic_vector`, `third_derivative_periodic_analytic`,
`third_derivative_periodic_analytic_vector`,
`third_derivative_periodic_kpoint_analytic`,
`third_derivative_periodic_kpoint_analytic_vector`, `strain_hessian_derivative`,
`gruneisen`. The analytic FC3 **block** modes, the *seminumerical* k-point FC3
and `pbc_berry_polarization` remain Rust-only, and **none** of these has a CLI
flag.

---

## 2. Seminumerical periodic third derivative

```rust
pub fn pbc_third_derivative_seminumerical_dense(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, step: f64,
) -> Result<Vec<Matrix>>              // 3N slabs, slab[c][(a,b)] = dH_ab/dR_c

pub fn pbc_third_derivative_seminumerical_vector(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, step: f64, v: &[f64],
) -> Result<Matrix>                   // Σ_c v_c dH_ab/dR_c
```

Everything here is a **central finite difference of the analytic Γ-point PBC
Hessian** (`pbc_gamma_hessian`): each `±` displacement re-runs the full periodic
SCC *and* the full analytic Hessian, so the result inherits the FD-validated
analytic Hessian rather than double-differencing energies.

The dense slabs are deliberately **not** re-symmetrised across `c`, so invariance
checks stay genuine. The vector mode is an exact contraction of the same per-DOF
differences, skipping zero components, so it is **bit-for-bit equal** to the dense
contraction.

Cost: dense needs `2·3N` analytic periodic Hessians; vector needs `2·nnz(v)`; the
strain derivative needs exactly 2.

> **A closed-form Γ-point third derivative also exists** — see §2c. It carries
> the periodic CPXTB response chain one derivative further through the
> Ewald/Klopman–Ohno partitioning, needs no finite differences, and is gated
> against *this* route. Prefer it at Γ with integer occupations; stay here for
> Fermi smearing, or when you need better than `~1e-7` absolute. For a k-mesh
> there is now a closed form too — §2d, `pbc_kpoint_third_analytic_*`, gated
> against §2b. The CLI rejects `--third-derivative`
> on any lattice-bearing input with
> `"third derivative (cubic force constants) supports non-periodic systems only"`.

---

## 2b. Seminumerical **k-point** third derivative

```rust
pub fn pbc_kpoint_third_derivative_seminumerical_dense(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, step: f64,
) -> Result<Vec<Matrix>>              // 3N slabs, slab[c][(a,b)] = dH_ab/dR_c

pub fn pbc_kpoint_third_derivative_seminumerical_vector(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, step: f64, v: &[f64],
) -> Result<Matrix>                   // Σ_c v_c dH_ab/dR_c

pub fn pbc_kpoint_strain_hessian_derivative(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, delta: f64,
) -> Result<Matrix>                   // dH_ab/d(ln V), Hartree/Bohr²
```

Identical machinery to §2/§3 — same displacement convention, same exact
`Δ ln V = ln((1+δ)/(1−δ))` denominator, same "no re-symmetrisation across `c`",
same bit-exact vector/dense contraction — but each analytic Hessian is
`pbc_kpoint_hessian` instead of `pbc_gamma_hessian`. There is one shared FD
implementation parameterised by the Hessian evaluator; nothing is duplicated.

**The k-mesh comes from `pbc.kmesh`**, exactly as `pbc_kpoint_hessian` takes it.
A `KMesh::gamma()` mesh reduces these to the Γ entry points.

**What converges with the mesh, and what does not.** The nuclear displacement
pattern is still one atom of one cell, so these are the `q = 0` cubic force
constants — *not* phonon FC3s at finite `q`. What the mesh converges is the
**electronic** Brillouin-zone sum behind each Hessian. That is not a small effect
on a primitive cell: on 2-atom diamond the slab `dH/dR_{0x}` moves by **127% of
its own magnitude** between `[1,1,1]` and `[2,2,2]` (see the gate numbers below).
A Γ-only FC3 on a primitive cell is a Γ-only *electronic* FC3, and should be read
as such.

Cost: `2·3N` k-point Hessians (dense), `2·nnz(v)` (vector), exactly 2 (strain) —
each roughly `n_k` times dearer than its Γ counterpart, and complex. The vector
mode with a single non-zero component (2 Hessians) is the affordable way to run a
k-mesh study.

Under isotropic strain the mesh is held **fixed in fractional reciprocal
coordinates**, so it scales with the reciprocal lattice and both strained cells
are sampled at the same fractional `k`. That is what makes the central difference
well defined at fixed `pbc.kmesh`.

> **Gated at `T = 0`, integer occupations, gapped.** Every k-point gate below
> pins `electronic_temperature = 0.0` explicitly, rather than relying on a gapped
> system's 300 K occupations rounding to integers. The periodic
> finite-temperature response (`kpoint_finite_temperature_response_dof` and its Γ
> analogue) is a direct dielectric solve as of v0.5.0 — see §5 — so a smeared
> k-point run is not *unsound*; it is simply not gated here. Differencing a
> reconverged metallic Hessian needs a far tighter SCC than the defaults, and
> that is the finite-temperature workstream's ground, not this one's.

### Why this exists: the verification base for the analytic k-point FC3

These entry points differentiate an already FD-gated analytic k-point Hessian, so
they isolate exactly the one derivative the **analytic** k-point FC3 has to add.
That analytic path landed in v0.5.0 (§2d) and is gated against these — the same
relationship the molecular analytic FC3 has to `third_derivative_seminumerical`.
The seminumerical route remains the supported one for Fermi-smeared cells and for
option sets the term registry caps below analytic order 3.

### Gates (`tests/pbc_third_derivative.rs`, 6 tests, 275 s total)

Primitive diamond with **one carbon nudged 0.03 Å along x** (see the caveat
below for why), `T = 0`, lean cutoffs (AO 12 / Ewald real 18 / sr 8 Bohr),
FD step `h = 10⁻³` Bohr, strain `δ = 5·10⁻³`.

| Gate | Result |
| --- | --- |
| Γ-only mesh vs the Γ FC3, same `h` | `max\|Δ\| = 7.2e-12` on a `1.396` Eh/Bohr³ scale (rel `5.2e-12`); **7 of 216 entries bit-identical** — close, not bit-exact |
| Acoustic sum rule, `[2,2,2]` mesh | `5.6e-13` on a `0.599` Eh/Bohr³ scale (rel `9.3e-13`) |
| vector vs dense contraction, `[2,2,2]` | **exactly `0`** |
| k-mesh trend | `\|[1,1,1]−[2,2,2]\| = 7.97e-1` (rel `1.265`), `\|[2,2,2]−[3,3,3]\| = 8.78e-2` (rel `0.140`) — settling, factor ~9 |
| `dH/dlnV`: Γ-only vs Γ path | `6.2e-13` on a `2.164` Eh/Bohr² scale (rel `2.9e-13`) |
| `dH/dlnV`: `[1,1,1]` vs `[2,2,2]` | `1.400` (rel `0.647`) — the mesh reaches the strain derivative |

The Γ-only collapse is **not** bit-for-bit, and should not be expected to be: the
k-point path back-transforms real-space image densities and solves the *complex*
CPXTB by preconditioned CG, where the Γ path uses the real density and a direct
real solve. `kpoint_hessian_reduces_to_gamma_at_gamma_only` gates the Hessians
themselves at `1e-8`; dividing that by `2h = 2·10⁻³` would permit a `1e-5` slab
difference, and what is measured is seven orders tighter.

### ⚠ Caveat: `pbc_kpoint_hessian` is wrong on exactly degenerate frontier orbitals

This is a **pre-existing bug in `src/pbc/hessian.rs`**, not in the FD layer, but
it decides which cells the k-point derivatives above can be run on.

Perfect diamond has a triply degenerate `t2` HOMO *and* LUMO at Γ. Inside an
exactly degenerate block the complex generalized eigensolver returns an arbitrary
unitary basis, and the k-point CPXTB is **not invariant** under that choice. At a
Γ-only mesh the correct answer is exactly `pbc_gamma_hessian` (itself verified
against the analytic-gradient FD to `2.3e-9`); measured
`max |H_Γ − H_kpoint|` on undistorted diamond, scale `1.2 – 1.5` Eh/Bohr²:

| `a` (Å) \ `ao_cutoff` (Bohr) | 10 | 12 | 14 | 16 | 20 |
| --- | --- | --- | --- | --- | --- |
| 3.567 | 2e-15 | **1.90** | **3.71e-1** | 5e-15 | **7.91e-1** |
| 3.500 | **4.15e-1** | **7.58e-1** | **1.26** | 3e-15 | 8e-15 |
| 3.400 | **2.07e-1** | 3e-15 | 3e-15 | 5e-15 | 1e-15 |
| 3.567004 | **7.88e-1** | 1e-15 | **5.79e-1** | **1.50e-1** | 2e-15 |

i.e. up to a **100% error**, appearing and vanishing with a `4·10⁻⁶ Å` change in
the lattice constant. With the 0.03 Å distortion **all twenty** configurations
collapse to `4e-16 … 3e-14`.

Consequences, and they are not symmetric across the entry points:

- The **FC3** entry points are only exposed through evaluations at the
  reference geometry, which a central difference over atomic displacements never
  makes — a displaced high-symmetry cell has its degeneracy lifted.
- The **strain derivative** and **Grüneisen** paths evaluate isotropically scaled
  cells, which preserve the point group *exactly*. They hit the bug head-on on a
  perfect crystal.

So until the degenerate-block handling in the k-point CPXTB is fixed, run the
k-point derivatives on cells whose frontier orbitals are non-degenerate at every
sampled `k`. `kpoint_hessian_degenerate_frontier_orbital_bug_is_confined` pins
both halves of this as a negative gate: the distorted fixture must stay clean
(`< 1e-12` relative) and the perfect cell must stay broken (`> 1e-3`), so the day
the bug is fixed the test fails loudly instead of silently.

---

## 2c. Analytic Γ-point third derivative

```rust
// One-shot entry points (each builds its own shared reference).
pub fn pbc_gamma_third_analytic_vector(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, v: &[f64],
) -> Result<f64>                              // e³[v] = Σ_abc T_abc v_a v_b v_c

pub fn pbc_gamma_third_analytic_dense(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions,
) -> Result<SymmetricThird>                   // packed T_abc, Hartree/Bohr³

pub fn pbc_gamma_third_analytic_block(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, dofs: &[usize],
) -> Result<SymmetricThird>                   // |dofs|³ sub-tensor, indexed by POSITION in dofs

// Shared-reference form, for sweeping many directions on one geometry.
pub struct GammaThirdReference { /* … */ }
impl GammaThirdReference {
    pub fn build(system, params, options, pbc) -> Result<Self>;
    pub fn scc(&self) -> &PbcSccResult;       // the converged periodic SCC
    pub fn mos(&self) -> &GammaMos;           // Γ coefficients / energies / occupations / S
}
pub fn pbc_gamma_third_with_reference(
    system, params, options, pbc, reference: &GammaThirdReference, v: &[f64],
) -> Result<f64>
```

**No finite differences anywhere.** §2 differences the analytic periodic
Hessian; this differentiates the periodic CPXTB response chain itself, once
more, through the Ewald-split Klopman–Ohno γ. Python:
`third_derivative_periodic_analytic` and
`third_derivative_periodic_analytic_vector` (atomic units, Γ-only).

### What is assembled

The directional third is a sum of five groups. Writing `X⁰` for the ground
`(P₀, W₀, q₀)`, `X¹` for the first-order response along `v` and `X²` for the
second-order response:

```text
e³[v] = frozen third                        band/Pulay + CN + SCC2(real) + SCC2(Ewald)
                                            + repulsion / halogen / D3 (already image-summed)
      + density path   ∂frozen²/∂X⁰ · X¹    the frozen Hessian's own motion with the reference
      + g(X²)·v                             the periodic response gradient at the 2nd-order slots
      + B6(X¹)[v,v]                         geometric motion of the response gradient
      + 2·bg4(X¹, X¹)·v                     the bilinear self-term, differentiated into BOTH slots
```

Three of these are worth spelling out, because they are exactly what a naive
mirror of the molecular assembly gets wrong.

**The density path is not optional, and it is not in the response gradient.**
The periodic `response_gradient` has a *narrower* scope than its molecular
counterpart — no potential legs, one-sided CN — so `∂frozen²/∂X⁰ · X¹` is a
separate contribution rather than something `B6` already covers. It is four
pieces: the band/Pulay Hessian block with `X¹` in the frozen density slot
*including* its `v₁`/`v₂` potential legs, the CN block in its two-sided Hessian
convention, twice the SCC2 charge-path bilinear `(q₀, q¹)`, and the
`V(q₀)`-cache motion. That last one is evaluated by a Δ-trick rather than a new
builder: re-evaluate the band/Pulay block on an SCC record whose shell potential
is shifted by `K·q¹` and whose potential legs are shifted by `∂γ_v·q¹` and
`∂²γ_vv·q¹`, then subtract the unshifted evaluation. Omitting the density path
was the single largest error in the first working assembly — `1.29e-4`, i.e.
**1500x** the current residual.

**`bg4` enters with a factor 2.** The response gradient's bilinear self-terms
differentiate into both slots (`d/dλ[−P¹(Kq¹)∇S]` feeds the mixed `(X², X¹)`
pairs from either side) while `g(X²)` supplies only the diagonal; the two mixed
completions are precisely the `bg4` families.

**The frozen thirds use a V-fixed convention.** Per AO image pair,
`t = C·S₃ + 2P·S·H₃ + P(6H₁−V₁)·S₂ + P(6H₂−V₂)·S₁`, with the whole `dV/dR`
routed through the density path exactly as in the molecular assembly. The three
classical terms (repulsion, halogen, D3) needed no new code at all: their
existing third derivatives are image-summed already.

### Gates

Fixtures are two **distorted** 2-atom cells — skewed diamond and skewed
zincblende BN — for the degeneracy reason of §2b, plus BN to exercise the
heteronuclear charge path. Lean cutoffs (AO 12, Ewald sr 8 Bohr), `T = 0`.

| Gate | diamond-skew | BN-skew | Where |
| --- | --- | --- | --- |
| **Total** vs Richardson-extrapolated seminumerical, `\|Δ\|` | `8.53e-8` | `8.03e-8` | `gamma_analytic_third_matches_seminumerical` |
| Acoustic sum rule, worst of 4 rigid translations | `1.0e-15` | `4.2e-16` | `..._obeys_the_acoustic_sum_rule` |
| Block (4 DOF) `contract_vvv` vs directional, rel | `9.6e-16` | `4.1e-15` | `..._block_matches_directional` |
| Dense `contract_vvv` vs directional, 3 independent `v`, worst rel | `2.9e-16` | `1.6e-15` | `..._dense_matches_directional` |
| `q^vv` vs 2nd central FD of reconverged SCC charges | `2.7e-9` | `4.9e-9` | `gamma_second_order_charges_match_reconverged_fd` |
| 1st-order field: dielectric route vs occ-virt PCG (`dP`/`dW`/`dq`) | `1.7e-15` / `5.3e-16` / `2.3e-16` | `4.8e-14` / `3.6e-14` / `4.5e-15` | `gamma_context_first_order_matches_pcg_route` |

Rows 2–4 land at **machine precision**, not merely inside tolerance. That is the
expected outcome and it is worth saying why each is still informative:

- The **acoustic sum rule** is an exact invariance of the assembled tensor, and
  the `~1e-15` result says every block — including the Ewald reciprocal sums and
  the CN chain — is translation-covariant to the last bit. It is the one gate
  that would catch a sign or image-offset error in a block that happens to be
  small on these fixtures. (The `1e-10` tolerance in the test is deliberate
  slack for other cells, not a description of what is measured.)
- The **polarization identities** confirm the dense/block drivers recover the
  directional evaluator exactly. They reach the same contraction through 7
  signed evaluations per canonical triple at *integer*-weighted directions,
  recombined — so agreement at `1e-15` with a single fractional-weight direction
  also certifies that the directional evaluator is exactly cubic-homogeneous
  in `v`, with no quadratic or quartic contamination.

The last two rows are the load-bearing intermediate gates. The first-order one is
an **equivalence** gate — two independently implemented solvers of the same fixed
point (a charge-space dielectric factorization vs occupied-virtual preconditioned
CG) agreeing to machine precision — which is a far stronger statement than any
finite difference.

### ⚠ The `~1e-7` residual is real

The total gate's `8.5e-8` / `8.0e-8` is **not** finite-difference truncation. The
`h`-ladder ratio is `0.99` / `1.10` (a truncation error would show `~4`), so the
seminumerical reference is converged and the assembly is genuinely missing a term
of that size. On a `~10⁻³` Eh/Bohr³ scale that is a `~1e-4` relative error.

What is already excluded, block by block, by the standing diagnostics:

- **CN closes exactly** (`1e-13`, two-sided convention).
- **SCC2 real-space + Ewald close together**: the bilinear plus `3.79e-7` matches
  the sum of both blocks' charge motion.
- The remainder is **entirely band/Pulay**: `+1.06e-7` (diamond), `+1.45e-8`
  (BN). It appears on diamond, where the `dV`-cache term is ≈ 0, so it is
  *charge-independent* — a geometric, not electrostatic, omission.

Leading hypothesis: the `se(CN)` self-energy cache motion crossed with `X¹`. The
`bp(X¹)` block freezes `se₀`, but the true `D_v` includes `se(CN(λ))` moving
against the `P¹` term — a cross between the CN builder's `dse/dCN` and the band
geometry's own motion. A per-block FD of the reconverged `scf(λ)` cannot settle
it directly, because `se(λ)` contaminates the band/Pulay and CN attributions of
each other.

Four `#[ignore]`d diagnostics stand permanently in `src/pbc/gamma_response.rs`
for this: `gamma_response_split_diagnostic` (FD of the true `D_v[r₂]` vs the
assembly's response side, and by subtraction the frozen side),
`gamma_frozen_block_split_diagnostic` (per-block `D_v` at the reconverged
`scf(λ)`, pinning the density-path inventory block by block),
`gamma_second_order_charges_match_reconverged_fd`, and the total gate itself.
`GFN1_G3_DEBUG=1` prints the five-group breakdown plus the `b6` blocks, the
`bg4` families and the individual frozen thirds.

**Practical consequence — and be clear-eyed about it: the analytic route is
currently the *less* accurate of the two.** The gate prints both the raw
`h = 10⁻³` difference and the Richardson extrapolation, and they differ by only
`6.2e-10` (diamond) / `1.1e-8` (BN). So the seminumerical FD is already
converged to `~10⁻⁹`, an order of magnitude *inside* the analytic assembly's
`8e-8` residual. Accuracy is not the reason to reach for this path.

The reasons that do hold:

- **Vector mode is dramatically cheaper.** `pbc_gamma_third_analytic_vector`
  costs one SCC and one assembly; `pbc_third_derivative_seminumerical_vector`
  costs `2·nnz(v)` full periodic Hessians. For a directional anharmonicity along
  one normal mode that is a large constant-factor win, and it grows with the
  number of moving atoms. (Dense mode is *not* a win at small `N` — see Cost.)
- **No step size to choose.** No `h` to tune, no cancellation floor, and nothing
  that silently degrades when the SCC tolerance is loosened.
- **It composes.** Being a closed form, it can be differentiated further and
  reused as a building block; a finite difference cannot.

If you need better than `~1e-7` absolute on periodic cubic force constants
today, use `pbc_third_derivative_seminumerical_*`. For phonon anharmonicity and
Grüneisen work, where the FC3 scale is `10⁻³` and the physics tolerance is
percent-level, `8e-8` is far below anything that matters and the vector-mode
cost advantage decides it.

### Cost

One directional evaluation costs a periodic SCC, **two** image-summed skeleton
builds, one charge-space factorization, and the frozen/response/density-path
assemblies. `GammaThirdReference` hoists everything that depends only on the
reference geometry — the SCC, the Γ MOs, the primary skeleton, the charge-space
factorization, the real-space ground densities, the response band-pair table, the
screening kernel and the coordination derivatives — so a dense sweep pays those
once.

What genuinely *cannot* be hoisted is the second skeleton build: `dγ_v·q¹` needs
the geometric kernel motion evaluated **at the response charges**, and the only
way to reach it is a full `gamma_skeleton_derivatives` call on an SCC record
doctored to carry `q¹` — which depends on the direction. That is why a
direction still costs ~2x a single skeleton, and it is the obvious target if
this path ever needs to get faster (a charge-slot-generic skeleton entry point
would remove it).

Dense mode needs `~C(3N+2, 3)` distinct directions after deduplicating the
polarization subsets — 56 for a 2-atom cell, 816 for 8 atoms — evaluated in
parallel with `rayon`. The direction count alone is `N³`, on top of a
per-direction cost that itself grows with `N`.

**Dense mode is therefore not a cost win at small `N`, and the measured numbers
say so.** On the 2-atom fixtures (lean cutoffs, `rayon` across all cores): a
4-DOF `_block` sweep runs `142 – 158 s`, and the full 6-DOF `_dense` sweep runs
`302 s` (BN) / `333 s` (diamond). `pbc_third_derivative_seminumerical_dense`
produces the same-shaped tensor from only `2·3N = 12` periodic Hessians, in a
fraction of that. The crossover moves with `N` — and it moves the wrong way,
since the seminumerical Hessian count is linear in `N` while the polarization
direction count is cubic.

So: `_vector` when one direction suffices (that is where the analytic path wins
outright), `_block` for a localized sub-tensor, and `_dense` only when you
specifically want the closed form rather than the cheapest route to the tensor.

### Not covered

- **k-point meshes.** Γ only; a Monkhorst–Pack mesh is rejected by name. Use
  the analytic k-point path `pbc_kpoint_third_analytic_*` (§2d), which has no
  mesh restriction, or the seminumerical
  `pbc_kpoint_third_derivative_seminumerical_*` (§2b).
- **Fermi smearing.** Fractional occupations are rejected. The seminumerical
  routes support them.
- **Order-1 model options.** multipole, long-range exchange, DFT+U, spin
  polarization, external field and experimental D4 are rejected by
  `terms::require_order(_, _, 3, _)` — the first PBC entry point to consult the
  term registry.

---

## 2d. Analytic **k-point** third derivative

```rust
// One-shot entry points (each builds its own shared reference).
pub fn pbc_kpoint_third_analytic_vector(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, v: &[f64],
) -> Result<f64>                              // e³[v] = Σ_abc T_abc v_a v_b v_c

pub fn pbc_kpoint_third_analytic_dense(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions,
) -> Result<SymmetricThird>                   // packed T_abc, Hartree/Bohr³

pub fn pbc_kpoint_third_analytic_block(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, dofs: &[usize],
) -> Result<SymmetricThird>                   // |dofs|³ sub-tensor, indexed by POSITION in dofs

// Shared-reference form, for sweeping many directions on one geometry.
pub struct KpointThirdReference { /* … */ }
impl KpointThirdReference {
    pub fn build(system, params, options, pbc) -> Result<Self>;
    pub fn scc(&self) -> &PbcSccResult;       // the converged k-mesh SCC
}
pub fn pbc_kpoint_third_with_reference(
    system, params, options, pbc, reference: &KpointThirdReference, v: &[f64],
) -> Result<f64>
```

**No finite differences anywhere, and no mesh restriction.** `pbc.kmesh` is taken
as given: `KMesh::gamma()` collapses onto the Γ path (to `~1e-17` relative, see
the gates), any `KMesh::monkhorst_pack([n1,n2,n3])` samples the zone. Python:
`third_derivative_periodic_kpoint_analytic` and
`third_derivative_periodic_kpoint_analytic_vector` (atomic units, `kgrid`
argument).

As in §2b, the nuclear displacement pattern is still one atom of one cell, so
these are the `q = 0` cubic force constants; what the mesh converges is the
**electronic** Brillouin-zone sum inside them.

### What is assembled, and why it is the §2c assembly verbatim

```text
e³[v] = frozen third + density path + g(X²)·v + B6(X¹)[v,v] + 2·bg4(X¹, X¹)·v
```

Term for term, the same five groups as §2c, evaluated by the same builders. That
is not an implementation shortcut, it is the structural claim of this section:

**Every consumer in that assembly reads the electronic state through exactly two
objects — real-space density images `P(T)`, `W(T)`, and real shell charges `q`.**
Both are Brillouin-zone invariants. The inverse Bloch transform
`M(T) = Σ_k w_k Re[M(k) e^{−ik·T}]` is a real matrix per image, and Mulliken
charges are real by construction, so the frozen blocks, `response_gradient` and
the `B6`/`bg4` paths cannot tell a k-mesh from Γ. Concretely:

- every classical block (repulsion / halogen / D3) sees only geometry;
- `gamma_realspace_densities` was already a general `Σ_k` transform, not a Γ
  special case, and is reused unchanged;
- `shell_potential_second_directional` reads only `scf.shell_charges` and the
  lattice — the scalar SCC potential is a property of the converged density, not
  of the Bloch phase.

So the k-dependence is confined to the **two response solves** that produce those
images, and they are what `src/pbc/kpoint_third.rs` adds:

| new object | what it is |
| --- | --- |
| `kpoint_shell_potential_first_directional` | `V₁` sourced from `shell_potential_derivatives` directly, so no Γ skeleton is needed |
| `kpoint_frozen_third_directional` | the §2c frozen bundle driven off the mesh-summed density images |
| `kpoint_first_order_directional` | `X¹ = (P¹(T), W¹(T), q¹)` from the complex k-point CPXTB |
| `kpoint_directional_second_matrices` | `(F^vv(k), S^vv(k))`, the complex Bloch transcription of the Γ second-order skeleton pair |
| `kpoint_second_order_directional` | `X² = (P^vv(T), W^vv(T), q^vv)` from the complex resolvent form |

**The second-order response is where the genuinely new mathematics is.** It is
built from the Daleckii–Krein resolvent expansion
`d²G = G Bˣ G Bʸ G + G Bʸ G Bˣ G − G Bˣʸ G`, `B = z dS − dH`, contour-integrated
into divided differences of `z^L f(z)` — the same formulation the molecular
degenerate-response work uses. Two things about the complexification are worth
stating because getting either backwards is the likely transcription error:

- **Only the matrices become complex.** The pencil is Hermitian at every `k`, so
  the band energies are real and every weight `f^{[1]}, f^{[2]}, g, k` is a real
  divided difference of a real function. Complexifying the *weights* would be
  wrong.
- **`μ` is one scalar for the whole zone, and the dielectric coupling is real.**
  The particle-number condition generalises to the weighted mesh sum
  `d² Σ_k w_k Tr[S(k)P(k)] = 0` — one equation, one unknown — not one condition
  per k-point, which would conserve each k-point's occupancy separately and is
  not a physical constraint. And the only object coupling the k-points is the
  shell-charge vector, which is real: `q^vv` solves the same `nsh × nsh` system
  `(I − χ⁰K) q^vv = q̃^vv` the Γ path solves, with `χ⁰` accumulated over the mesh.
  For a gapped reference every Fermi weight vanishes and `μ` drops out entirely.

Hermiticity of `P^vv(k)` / `W^vv(k)` is **structural**, not imposed: divided
differences are symmetric in their nodes and every bilinear term is written in
both orderings, so nothing symmetrises and the gate measures the consequence.

### Gates

Fixtures are the same two **distorted** 2-atom cells as §2c — skewed diamond and
skewed zincblende BN — for the degeneracy reason of §2b.

The pair is not redundant, and the reason is mesh-dependent in a way worth being
precise about. On a **Γ-only** mesh the homonuclear diamond cell has an
essentially vanishing charge response — measured `max|q¹| = 9.05e-16`,
`max|q^vv| = 1.60e-14` — so the electrostatic channel and the dielectric solve are
silent there, and **BN is what makes the Γ-limit gate load-bearing**. Over the
`2 x 2 x 2` mesh the other gates use, diamond's charge response is *not* small
(`max|q¹| = 1.15e-2`, against BN's `2.35e-2`), so both fixtures drive the charge
path. Every gate reports the scale of what it compares and asserts it is non-zero,
so neither statement is relied on silently.

Integration gates, `tests/pbc_kpoint_third_analytic.rs`, `T = 0`, AO cutoff 12,
Ewald `sr` 8 Bohr, `2 x 2 x 2` Monkhorst–Pack unless stated:

| Gate | diamond-skew | BN-skew |
| --- | --- | --- |
| **Total** vs Richardson-extrapolated seminumerical k-point FC3, `\|Δ\|` | `1.43e-7` | `2.32e-8` |
| **Γ-limit**: `[1,1,1]` mesh vs `pbc_gamma_third_analytic_vector`, rel | `9.4e-18` | `2.9e-17` |
| Acoustic sum rule, worst of 4 rigid translations | `2.5e-16` | `7.2e-17` |
| Block (4 DOF) `contract_vvv` vs directional, rel | `8.1e-16` | `8.9e-16` |

Scale for the total row: `e³[v] = +2.363e-4` (diamond) / `+2.148e-4` (BN) over the
`2 x 2 x 2` mesh, so `1.43e-7` is `6e-4` relative. The **ladder ratio is `1.00`
on both fixtures** — the `h = 10⁻³` and `h = 5·10⁻⁴` references differ from the
analytic value by the same amount — which says the seminumerical reference is
converged and the discrepancy is `h`-independent, i.e. a genuinely missing term
rather than FD truncation. That is the §2c residual, inherited (see below).

Timings from the same run (2-atom cells, `2 x 2 x 2` mesh, lean cutoffs): the
analytic directional evaluation takes `35 – 38 s` against `79 – 85 s` for the two
seminumerical references it is checked against, and the 4-DOF block sweep takes
`144 – 161 s`.

Unit gates alongside the builders in `src/pbc/kpoint_third.rs` (13 tests) pin the
pieces individually: `V₁` against the production k-point skeleton at every `k`
and against a frozen-charge FD; the frozen bundle and each of its components
against their own `h`-ladders; `q¹` against a reconverged-SCC FD; `(F^vv, S^vv)`
against the Γ builder element-for-element at `k = 0` (`0.0e0`), against a
refreshed skeleton FD at a live `k`, and for Hermiticity; the divided-difference
table against its confluent limits and the classical symmetric form; `X²` against
the Γ charge-space path (`3.1e-15`) and against a reconverged second central
difference (`q^vv`, `9.2e-9`).

**The Γ-limit row is the sharpest gate in the file, and the cheapest.** On a
Γ-only mesh the two paths share the frozen half but reach `X¹` and `X²` through
completely different algebra — complex k-point CPXTB plus the complex resolvent
form, versus the real molecular charge-space solver with its
coefficient-and-frame second-order machinery. Agreement at `~1e-17` relative is a
statement about the derivation, not about shared code, and it is what to run
first when anything moves.

### ⚠ Same `~1e-7` residual as §2c, for the same reason

The k-point path reuses the §2c assembly, so it inherits the §2c open residual
exactly — see "The `~1e-7` residual is real" there for the block-by-block
attribution (band/Pulay; CN and both SCC2 blocks close exactly) and the leading
`se(CN)`-cache hypothesis. The total-gate tolerance here is set from that, not
from anything k-specific. `GFN1_K3_DEBUG=1` prints the five-group breakdown.

Two observations support the "inherited, not k-specific" reading rather than
merely asserting it. First, the **Γ-limit row is `1e-17`**: on a Γ-only mesh the
k-point assembly reproduces the Γ number bit-for-bit in practice, so the k-point
machinery adds nothing of its own at the one mesh where the two are comparable.
Second, the residual **does not grow with sampling** — `1.43e-7` / `2.32e-8` over
`2 x 2 x 2` sits either side of the Γ values `8.53e-8` / `8.03e-8`, i.e. it moves
the way a fixed missing term does when the quantity it is missing from changes,
not the way a per-k error accumulating over the zone would.

### Cost

`KpointThirdReference` hoists everything that depends only on the reference
geometry, and for the k-point path that is more than at Γ:

- the k-mesh SCC and the frozen-charge `∂V/∂R` table;
- the real-space ground densities, response band pairs, screening kernel,
  coordination derivatives;
- **the full per-DOF CPXTB response sweep** `(∂P/∂R_y, ∂W/∂R_y, ∂q/∂R_y)` over
  the whole mesh — `3N` coupled complex solves. Every direction is then an exact
  contraction of those, because each stage between the skeleton pair and
  `(P¹, W¹, q¹)` is linear;
- **the per-k skeleton derivative table** `(∂F(k)/∂R_y, ∂S(k)/∂R_y)` for every
  k-point, which is the single most expensive per-`k` object the second-order
  response needs.

What stays per-direction: the frozen bundle, the `V₁/V₂` ladders, the `X¹`
contraction, the whole second-order response (`nk` second-order skeleton pairs,
`nk` MO transforms, the `nsh`-column susceptibility, the dielectric solve), the
response gradient and both paths.

One thing is *cheaper* than at Γ. The Γ assembly needs a second full
`gamma_skeleton_derivatives` build per direction to reach `dγ_v·q¹` (the
geometric kernel motion at the response charges) because that is the only entry
point exposing it. The k-point path reaches the same object through
`kpoint_shell_potential_first_directional`, a pure charge/lattice call — so there
is no second skeleton build per direction here.

Measured on the 2-atom fixtures (lean cutoffs, `rayon` across all cores,
`2 x 2 x 2` mesh): the shared reference plus 4 rigid directional evaluations run
`132 – 140 s` per fixture.

Dense mode needs `~C(3N+2, 3)` distinct directions after deduplication — 56 for a
2-atom cell, 816 for 8 atoms — on top of a per-direction cost that is itself
roughly `n_k` times the Γ one. **So the same advice as §2c applies, more
strongly:** `_vector` when one direction suffices (that is where this path wins
outright — one SCC and one assembly against `2·nnz(v)` full k-point Hessians),
`_block` for a localized sub-tensor, and `_dense` only when you specifically want
the closed form rather than the cheapest route to the tensor.

### Not covered

- **Fermi smearing.** Fractional occupations at *any* k-point are rejected; use
  `pbc_kpoint_third_derivative_seminumerical_*` (§2b).
- **Order-1 model options.** Same `terms::require_order(_, _, 3, _)` guard as
  §2c: multipole, long-range exchange, DFT+U, spin polarization, external field
  and experimental D4 are rejected.
- **The degenerate-frontier caveat of §2b still applies**, since the first-order
  response comes from the same k-point CPXTB. Run on cells whose frontier
  orbitals are non-degenerate at every sampled `k` — which is why every fixture
  here is distorted.

---

## 3. Strain-mixed derivative `dH/d(ln V)`

```rust
pub fn pbc_strain_hessian_derivative(
    system: &PeriodicSystem, params: &Gfn1Parameters,
    options: &ElectronicOptions, pbc: &PbcOptions, delta: f64,
) -> Result<Matrix>                   // Hartree/Bohr² (ln V is dimensionless)
```

A central difference of the analytic Γ-point Hessian under **isotropic frozen-ion
volumetric strain**: the three lattice vectors are multiplied by `(1 ± δ)^(1/3)` so
that `V → (1 ± δ)V`, and the atoms follow with **frozen fractional coordinates**
(which under isotropic scaling is exactly `r → s·r`). `scale_lattice_isotropic`
performs that scaling and is public.

The denominator is the **exact** log-volume separation
`Δ ln V = ln((1+δ)/(1−δ))`, not the leading-order `2δ`, so the estimator is
`O(δ²)`. Gate: Richardson `δ` vs `δ/2` agrees to `2.7e-5` relative on diamond.

Contracting this matrix with a mass-weighted normal mode gives that mode's
Grüneisen parameter directly; the `gruneisen` module instead re-diagonalises at
both volumes, which additionally resolves mode crossings. Cross-check on diamond:
first-order perturbation theory on `dH/dlnV` gives `γ = 0.90545` against the
re-diagonalised `0.90542`.

---

## 4. Grüneisen parameters

```rust
pub fn pbc_gruneisen(
    system: &PeriodicSystem, params: &Gfn1Parameters, options: &GruneisenOptions,
) -> Result<GruneisenResult>
```

**Mode Grüneisen parameter** `γ_i = −d ln ω_i / d ln V` — the microscopic origin of
thermal expansion, via the Grüneisen relation `α_V = γ_th · C_V · κ_T / V`.

Three analytic Γ-point PBC Hessians are evaluated (at `V₀` and `V₀(1 ± δ)`), each
mass-weighted and diagonalised. The two strained mode sets are matched back onto
the reference set by greedy **maximum eigenvector overlap**, then

```text
γ_i = −(ln ω_i(V₊) − ln ω_i(V₋)) / (ln V₊ − ln V₋)
    = −(ln λ_i(V₊) − ln λ_i(V₋)) / (2 · ln((1+δ)/(1−δ)))
```

Degenerate subspaces get a trace-averaged `γ`. The **thermodynamic Grüneisen
parameter** weights each mode by its Einstein heat capacity:

```text
γ_th(T) = Σ_i γ_i c_i(T) / Σ_i c_i(T) ,   c_i(T) = k_B x² eˣ/(eˣ−1)² ,  x = ħω_i/(k_B T)
```

### Second order: γ⁽²⁾, and the literature `q`

`GruneisenOptions::second_order` turns on the **curvature of `ln ω` in `ln V`**:

```text
γ⁽²⁾_i = ∂²ln ω_i / ∂(ln V)²
```

**Sign convention — read before comparing with anything.** `γ⁽²⁾` is the *plain*
second log-derivative; unlike `γ` it carries **no** leading minus sign. Hence

```text
γ⁽²⁾_i = −∂γ_i/∂ln V = q_i · γ_i ,      q_i = −∂ln γ_i/∂ln V = γ⁽²⁾_i / γ_i
```

`q` is the exponent of the Mie–Grüneisen thermal-EOS model `γ(V) = γ₀(V/V₀)^q`;
`GruneisenResult::mode_q()` returns it per mode. A positive `γ⁽²⁾` (positive `q`)
means `γ` *grows* on compression.

The estimator fits `ln λ_i` (= `2 ln ω_i`) against `ln V` through the matched mode
sets, with **Fornberg weights for the actual, non-uniformly spaced nodes**
`ln(1 ± δ)`. That is not a nicety. The nodes are *not* symmetric about `ln V₀`:
`ln(1+δ) + ln(1−δ) = ln(1−δ²) ≈ −δ²`, and feeding them to the textbook
`(f₊ − 2f₀ + f₋)/h²` leaks a term `f′·(a−b)/h² ≈ −f′` — **not** a small
correction but the *first* derivative at full size. On diamond that is `+1.81`
against a true `f″ = 2γ⁽²⁾ = −0.077`: it would report `γ⁽²⁾ ≈ +0.87`, `q ≈ 0.96` —
a number that lands squarely in the literature range and is entirely an artefact.
Every strained set, including the outer pair of the five-point stencil, is matched
onto the **central** volume's modes, and degenerate subspaces are trace-averaged
before the fit, exactly as at first order.

| stencil | nodes | extra Hessians | truncation |
| --- | --- | --- | --- |
| `ThreePoint` (default) | `V(1−δ₂), V₀, V(1+δ₂)` | 2 (**0** if `δ₂ = δ`) | `O(δ₂²)` |
| `FivePoint` | adds `V(1 ± 2δ₂)` | 4 (2 if `δ₂ = δ`) | `O(δ₂⁴)` |

The same fit also returns the first derivative as `mode_gamma_refit`, which must
reproduce the independent two-point `mode_gamma` — the internal consistency check
on the whole path.

### The two orders do not share a step (`delta_second`)

`γ⁽²⁾` is a **second** difference of `ln λ(ln V)`, so a residual noise `ε` in the
phonons reaches it as `ε/δ²`, where the first-order `γ` only suffers `ε/δ`. One
shared step cannot serve both, and the second-order node set therefore has its
own `GruneisenOptions::delta_second` — default `2·10⁻²`, against `delta = 5·10⁻³`
for the first order. `delta_second = None` puts the nodes back on the
first-order ones (free, since those Hessians exist anyway, but 16x more exposed
to noise).

**The step alone was not the whole story.** The real-space cutoffs (`ao_cutoff`,
Ewald `real_cutoff` / `sr_cutoff`) are radii in Bohr and every lattice sum runs
over the *integer* images inside them, so a cell breathing under a **fixed**
radius crosses image shells at discrete volumes: `ln λ(ln V)` picks up jumps of
`ε ≈ 5·10⁻⁷` (`≈ 6·10⁻⁴ cm⁻¹` on a 2292 cm⁻¹ mode) instead of being smooth.
Invisible in `γ`; fatal once a second difference divides it by `δ²`. Every
cutoff now travels with the node's linear factor `(V/V₀)^{1/3}`, which freezes
the integer image set across the whole stencil. Measured on the primitive cell
at the lean test cutoffs (12/18/8 Bohr), `γ⁽²⁾(300 K)` at `δ₂ = 5·10⁻³`:

| | three-point | five-point | relative gap |
| --- | --- | --- | --- |
| fixed cutoffs (before) | −0.037186 | **+0.067365** | 1.55 |
| scaled cutoffs (now) | −0.037186 | −0.037458 | 7.3·10⁻³ |

The three-point value never moved — at `δ = 5·10⁻³` the inner nodes do not reach
a shell — while the five-point outer nodes at `±10⁻²` did, which is why the two
stencils disagreed by more than the value itself and with opposite signs.

**Two thermodynamic conventions, and they are not the same thing:**

```text
γ⁽²⁾_th(T)      = Σ_i γ⁽²⁾_i c_i / Σ_i c_i                            (mode average)
γ⁽²⁾_th,full(T) = −∂γ_th(T,V)/∂ln V = γ⁽²⁾_th(T) − Σ_i w_i D_i (γ_i − γ_th)
```

with `w_i = c_i/Σc` and `D_i = ∂ln c_i/∂ln V`. The first is the heat-capacity
weighted mode average — the direct analogue of `γ_th`. The second is the honest
volume derivative of `γ_th(T,V)`: the **weights move with volume too**, because
`c_i` depends on `V` only through `ω_i(V)`, so the correction term is closed-form,

```text
D_i = −γ_i · x_i · (d ln c/dx)(x_i) ,   (d ln c/dx)(x) = 2/x − coth(x/2)
```

(series `−x/6 + x³/360` below `x = 10⁻³`, where the closed form is two diverging
terms cancelling). The correction vanishes identically when all modes share one
`γ` — which is exactly diamond's degenerate optical triplet, so on diamond the two
numbers agree to machine precision. The term is therefore gated by a
synthetic-model unit test in `src/pbc/gruneisen.rs`, against a numerical
`∂γ_th/∂ln V` of a model with a known `ln ω_i(V)`.

### k-point routing (`GruneisenOptions::kpoint`)

`kpoint: true` routes all three (or five) strained Hessians through
`pbc_kpoint_hessian` on `GruneisenOptions::pbc`'s mesh instead of
`pbc_gamma_hessian`. It converges the **electronic** Brillouin-zone sum behind
each Hessian; it does **not** turn this into a phonon-dispersion Grüneisen
average — the displacement pattern is still one cell's Cartesian DOFs, i.e. the
`q = 0` dynamical matrix, so the acoustic-branch caveat below is untouched. Cost
scales with the number of k-points; `KMesh::gamma()` reproduces the Γ path.

Measured on distorted primitive diamond (lean cutoffs, `δ = 5·10⁻³`,
`gruneisen_kpoint_routing`):

| run | ω(optical) | γ | γ_th(300 K) |
| --- | --- | --- | --- |
| `kpoint: false`, Γ | 2367.302 cm⁻¹ | 0.900819 | 0.907012 |
| `kpoint: true`, `[1,1,1]` | 2367.302 cm⁻¹ | 0.900819 | 0.907012 |
| `kpoint: true`, `[2,2,2]` | 1492.655 cm⁻¹ | 0.965533 | 1.073457 |

The option is inert at a Γ-only mesh (identical to six figures, gated at `1e-6`
relative on γ and `1e-4` cm⁻¹ on ω) and the real mesh moves the optical frequency
by ~875 cm⁻¹ — another reading of how Γ-only a primitive-cell Γ-only calculation
really is.

> Read the ⚠ caveat in §2b before using this on a high-symmetry crystal: the
> strained cells preserve the point group exactly, so `kpoint: true` on perfect
> diamond hits the degenerate-frontier-orbital bug in `pbc_kpoint_hessian`.

### Options and result

```rust
pub struct GruneisenOptions {
    pub delta: f64,                     // default 5.0e-3   (first order)
    pub temperatures: Vec<f64>,         // default vec![300.0]
    pub electronic: ElectronicOptions,  // default ElectronicOptions::default()
    pub pbc: PbcOptions,                // default PbcOptions::default()
    pub kpoint: bool,                   // default false  (see "k-point routing" below)
    pub acoustic_modes: usize,          // default 3
    pub degeneracy_tolerance_cm1: f64,  // default 1.0
    pub second_order: bool,             // default false  (see "Second order" above)
    pub second_order_stencil: SecondOrderStencil,  // default ThreePoint
    pub delta_second: Option<f64>,      // default Some(2.0e-2); None reuses `delta`
}

pub const DEFAULT_GRUNEISEN_DELTA_SECOND: f64 = 2.0e-2;

pub enum SecondOrderStencil { ThreePoint, FivePoint }

pub struct GruneisenResult {
    pub volume: f64,                            // Bohr³, reference geometry
    pub delta: f64,
    pub delta_second: Option<f64>,              // None unless second_order
    pub frequencies_cm1: Vec<f64>,              // reference, ascending; imaginary reported negative
    pub frequencies_cm1_expanded: Vec<f64>,     // at V(1+δ), permuted onto the reference ordering
    pub frequencies_cm1_compressed: Vec<f64>,   // at V(1−δ), permuted onto the reference ordering
    pub mode_gamma: Vec<f64>,                   // first `acoustic_modes` entries are NaN
    pub mode_gamma2: Vec<f64>,                  // γ⁽²⁾_i; all-NaN unless second_order
    pub mode_gamma_refit: Vec<f64>,             // γ_i from the same fit; all-NaN unless second_order
    pub thermodynamic_gamma: Vec<(f64, f64)>,   // (T, γ_th(T))
    pub thermodynamic_gamma2: Vec<(f64, f64)>,      // (T, γ⁽²⁾_th(T)); empty unless second_order
    pub thermodynamic_gamma2_full: Vec<(f64, f64)>, // (T, −∂γ_th/∂lnV);  empty unless second_order
    pub second_order_stencil: Option<SecondOrderStencil>,
    pub match_overlaps: Vec<f64>,               // worst over every strained volume evaluated
    pub acoustic_modes: usize,
    pub degenerate_groups: Vec<(usize, usize)>, // (start, len)
}

impl GruneisenResult {
    pub fn min_optical_overlap(&self) -> f64;
    pub fn gamma_at(&self, temperature: f64) -> Option<f64>;
    pub fn gamma2_at(&self, temperature: f64) -> Option<f64>;       // mode average
    pub fn gamma2_full_at(&self, temperature: f64) -> Option<f64>;  // −∂γ_th/∂lnV
    pub fn mode_q(&self) -> Vec<f64>;                               // q_i = γ⁽²⁾_i / γ_i
}
```

`min_optical_overlap()` is the quality metric for the mode matching — a value well
below 1 means modes crossed and the `γ_i` assignment is suspect.

### Validation

The 2-atom primitive fcc cell of diamond at `a = 3.567 Å`: mode `γ = 0.90542` for
the triply degenerate optical branch at `2292.5 cm⁻¹`, `γ_th(300 K) = 0.90542`,
`δ` vs `δ/2` agreement `5.5e-7` relative. The translational acoustic sum rule on
the dense FC3 slabs holds to `1.7e-12` on a `1.39 Eh/Bohr³` scale, and the
vector-vs-dense contraction is exactly 0.

**Second order, same cell** (δ = 5·10⁻³, three-point, lean test cutoffs
16/24/10 Bohr):

```text
γ⁽²⁾ = −0.03834      q = γ⁽²⁾/γ = −0.04234      (optical triplet)
γ⁽²⁾_th(100 K) = γ⁽²⁾_th(300 K) = γ⁽²⁾_th(1000 K) = −0.03834   (one degenerate
       triplet ⇒ the T weighting is trivial and the full derivative coincides)
γ_refit = 0.905418 against the two-point γ = 0.905418
       (agree to <1e-6 relative; the predicted gap is (q/2)δ² ≈ 5e-7)
```

So GFN1-xTB makes diamond's `γ` **almost volume-independent**: `|q| ≈ 0.04`,
where the experimental/DFT literature for diamond puts `q` at O(1). The sign says
`γ` grows slightly with volume. This is a *reported* number, not a tuned one — the
first-order `γ = 0.905` already sits at the bottom of the experimental
`0.9 – 1.2` window.

**`delta_second` ladder** — how the default was chosen. Primitive diamond, lean
cutoffs (12/18/8 Bohr), `delta = 5·10⁻³` throughout, so the first-order `γ` is
bit-for-bit identical (`0.905609595`) on every row and only the second-order node
set moves (`gruneisen_second_order_delta_calibration`):

| δ₂ | 5e-3 | 1e-2 | **2e-2** | 3e-2 | 4e-2 | 6e-2 | 1e-1 | 1.4e-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| γ⁽²⁾ 3-point | −0.037186 | −0.036372 | −0.037083 | −0.037207 | −0.037299 | −0.037508 | −0.038132 | −0.039088 |
| γ⁽²⁾ 5-point | −0.037458 | −0.036135 | −0.037011 | −0.037107 | −0.037140 | −0.037168 | −0.037177 | −0.037155 |
| relative gap | 7.3e-3 | 6.6e-3 | **1.9e-3** | 2.7e-3 | 4.3e-3 | 9.2e-3 | 2.6e-2 | 5.2e-2 |

A clean U: to the left the residual `ε/δ₂²` noise amplification wins, to the
right the `O(δ₂²)` truncation of the three-point stencil does — and the
five-point value, which is `O(δ₂⁴)`, sits flat at `−0.03717` from `4·10⁻²` all
the way to `1.4·10⁻¹`, so it is the three-point estimate that walks away. The
minimum is `δ₂ = 2·10⁻²`, which is the default. Running both stencils and reading
their gap is the honest convergence check.

The same ladder at the **production** cutoffs (30/40/10 Bohr) puts the minimum in
exactly the same place — relative gap `7.0·10⁻³` at `δ₂ = 5·10⁻³` against
`1.9·10⁻³` at `2·10⁻²`, landing on `γ⁽²⁾ = −0.03824` / `−0.03816` — so the
recommendation is not an artefact of the lean test cutoffs.

Two notes on reading this table against older numbers:

- The pre-repair ladder was measured with **fixed** cutoffs and a **shared** step,
  and it showed `γ⁽²⁾` swinging over `−0.037, −0.351, −0.121, −0.061` across
  `δ = 5e-3 … 4e-2` while `γ` held three digits. That swing was the image-list
  step, not physics; it is gone.
- The old advice — "the plateau is `δ ≤ 5·10⁻³`, so the first-order step serves
  the second order too" — was reading the *left* wall of that artefact. With the
  cutoffs scaled the curve has an ordinary noise/truncation minimum, and it sits
  a factor of four higher than the first-order step, exactly as the `ε/δ²`
  argument predicts.

### Caveats

- **Frozen-ion convention.** The **relaxed-ion** variant — re-optimising the
  internal coordinates at each strained volume and adding the `−H⁻¹ dF/d(ln V)`
  internal-strain coupling — is not implemented. This applies to `γ⁽²⁾` too: it is
  the curvature of the *frozen-ion* `ω(V)`.
- **`γ⁽²⁾` is a small number, and small numbers are exposed.** On diamond it comes
  out two orders of magnitude below `γ`, so an absolute error that is invisible in
  `γ` is a percent-level error in `γ⁽²⁾` — the δ₂ ladder above is a measurement of
  exactly that. Read `|q| ≈ 0.04` as "`γ` is volume-independent to within what
  this model resolves", not as four significant figures. The residual cutoff
  sensitivity is real but small now that the cutoffs scale with the strain: lean
  (12/18/8) gives `−0.0371`, production (30/40/10) `−0.0382`, a 3% spread.
- **`delta_second` costs Hessians.** With the default `δ₂ ≠ δ` the second-order
  nodes are their own volumes: two extra analytic Hessians for `ThreePoint`, four
  for `FivePoint`. Set `delta_second: Some(options.delta)` to put them back on the
  first-order nodes and pay nothing — at the price of `16x` more noise
  amplification (`(2e-2 / 5e-3)² = 16`), which on this fixture widens the
  stencil gap from `1.9·10⁻³` to `7.3·10⁻³`.
- **The two `γ⁽²⁾_th` conventions.** `thermodynamic_gamma2` is the mode average;
  `thermodynamic_gamma2_full` is `−∂γ_th/∂lnV` including the `∂c_i/∂lnV`
  reweighting. They coincide only when the modes share one `γ`. Neither is a
  Brillouin-zone integral.
- **Γ-only.** The three acoustic branches are the `ω → 0` modes of the cell and
  carry no meaningful `γ`; they are excluded from both the reported optical set
  and the thermodynamic average. A physically converged `γ_th(T)` at low `T` needs
  a Brillouin-zone sum over acoustic branches, which a Γ-only Hessian cannot
  provide.
- **Fermi smearing is supported** (§5), but a smeared Grüneisen run differences a
  reconverged Hessian, so tighten `charge_tolerance` first — see §5.

---

## 5. Periodic finite temperature (Fermi smearing)

Smeared periodic systems take a dedicated response branch: as soon as any band is
genuinely fractional (outside `[1e-10, 2 − 1e-10]`) the integer occupied–virtual
CPXTB cannot carry the occupation response `df/dR`, so the Γ and k-point paths
switch to the full-band finite-temperature response with a single global Fermi
level (Brillouin-zone-wide, `Σ_k w_k Σ_i df_ik = 0`).

**How the self-consistency is closed.** At a fixed reference state the
finite-temperature response map is *linear* in `(H^1, S^1, RF)`, so the SCC
feedback through the `nsh` Mulliken shell charges reduces to one charge-space
dielectric system, exactly as in the molecular
[charge-space solver](finite-temperature.md#1-the-charge-space-dielectric-solver):

```text
(I − χ⁰K) δq = δq_bare
```

`χ⁰` (the shell charges induced by a unit potential on each shell, at frozen
skeleton) costs `nsh` response evaluations, the dielectric is LU-factored **once
per Hessian call**, and every Cartesian DOF reuses the factorization. The k-point
susceptibility is a real `nsh × nsh` matrix as well: the perturbing potential and
the Brillouin-zone-summed Mulliken charges are real even though each per-k
response operator is complex. Every solve verifies its own residual and errors out
on a singular dielectric rather than returning a number.

This **replaced** a hard-capped 50-round damped fixed point that had no
post-loop convergence check — see
[limitations.md](limitations.md#periodic-finite-temperature-response-singular-dielectric--rejected)
for the measured size of that defect (up to `1e22` on a Ni cell at 3000 K) and
for the current gate numbers.

**What this means for the derivative stack.** `pbc_gamma_hessian`,
`pbc_kpoint_hessian` and everything built on them —
`pbc_third_derivative_seminumerical_*`, `pbc_strain_hessian_derivative`,
`pbc_gruneisen` — are now correct at `T > 0`; the earlier "run these at `T = 0`"
convention is retired. Two practical cautions remain, and neither is a guard:

- **The FD layers on top still need a tight SCC.** The seminumerical third
  derivative, the strain derivative and Grüneisen all difference a *reconverged*
  quantity, and metallic reconvergence noise enters as `noise / 2h`. Measured on
  the Ni₂ fixture: at the default `charge_tolerance = 1e-9` the shell-charge FD
  gate is noise-limited at `1e-6`; at `1e-12` the same gate reaches `3.1e-10`.
  Tighten `charge_tolerance` / `energy_tolerance` before reading a smeared FD.
- **Band reordering across the stencil.** A metal whose occupations change
  character between the `±h` geometries makes the differenced quantity
  non-smooth. This is a property of the fixture, not of the response.

Gapped insulators are unaffected: their Fermi occupations at 300 K are integer to
within the `1e-10` fractional-occupancy epsilon, so they stay on the integer
occupied–virtual path and reproduce the `T = 0` Hessian bit-identically (gated).

---

## 6. External field under PBC

Since v0.5.0 `pbc_stress` includes the **external electric field** contribution,
reported separately as `PbcStressResult::external_field_stress`. The stress
previously omitted it while `total_free` included it.

Because the SCC free energy is stationary at convergence, the field enters the
strain derivative only explicitly, through two channels:

1. the position-scaling term `σ_ab = −(1/V) E_a Σ_A q_A R_{A,b}` of the
   reference-cell sawtooth field energy (positions scale under strain; the field
   and its origin stay lab-fixed). This is generally an **asymmetric** tensor —
   the antisymmetric part is the field torque on the cell dipole;
2. the field-overlap (Pulay) strain coupling, which was already carried by
   `band_and_cn_stress` because `shell_scc_potential` folds the external site
   potential in.

Gate: all 9 stress components against the strain FD of the field-inclusive free
energy on a polar cell with an oblique field (`5e-6`).

The PBC **field** model itself remains the reference-cell dipole-coupling
approximation (a finite field is still applied as a sawtooth site potential, not
as a Berry-phase enthalpy). The **polarization** itself, however, is no longer
missing — see section 7.

---

## 7. Berry-phase bulk polarization

```rust
pub fn pbc_berry_polarization(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
) -> Result<BerryPolarizationResult>
```

`gfn1_rs::pbc::polarization` (re-exported at the crate root as
`pbc_berry_polarization`, `BerryPolarizationOptions`, `BerryPolarizationResult`,
`BerryPolarizationMethod`, `BerryMethodSelector`, `POLARIZATION_AU_TO_C_PER_M2`).

The cell dipole `∫ r ρ(r)` is **not** a bulk observable — it moves when you move
the cell boundary. The bulk polarization is a Berry phase of the occupied Bloch
manifold, defined only modulo a polarization quantum. Both discretisations of the
modern theory are implemented:

- **King-Smith–Vanderbilt** (`BerryPolarizationMethod::KingSmithVanderbilt`):
  a product of link determinants along a k-string, averaged over the
  perpendicular mesh,

  ```text
  phi_d = Im ln prod_j det M^(j),
  M^(j)_mn = <psi_{m,k_{j+1}} | e^{i dk . r} | psi_{n,k_j}>,   dk = b_d / N_d
  ```

- **Resta** (`BerryPolarizationMethod::Resta`): the Γ-only single-point form,
  `phi_d = Im ln det <psi_m | e^{i b_d . r} | psi_n>` — the `N_d = 1` member of the
  same family.

### The AO ingredient is a *boosted overlap*, not new integral code

The link matrix needs `<chi_mu(k)| e^{-i dk . r} |chi_nu(k + dk)>`, which reduces
to Bloch sums of the **boosted AO overlap**
`<chi_mu | e^{i q . r} | chi_nu>`. Completing the square in
`exp(-zeta |r-P|^2 + i q . r)` gives a complex product centre
`Pbar = P + (i/2 zeta) q` and prefactor `exp(i P.q - q.q/(4 zeta))` — which is
*exactly* the complex Gaussian product theorem the London/GIAO overlap in
`src/magnetic.rs` already implements, at `chi = -q`. The Berry code therefore
calls `gfn1_rs::magnetic::boosted_overlap_pair`, a thin public wrapper over the
shared kernel (`complex_boost_overlap_pair`) that `lao_overlap_matrix` also uses.
No integral code was duplicated.

Because all links along one direction share the same boost `b_d / N_d`, the
boosted image blocks are built **once per direction** and every link is then a
cheap phase-weighted Bloch sum — the same "expensive integrals once, phases per
k" structure as `BlochBuilder`.

### The wrap-around link is free

In the LCAO cell gauge used throughout `src/pbc`
(`phi_mu(k) = sum_T e^{i k.T} chi_mu(r - tau_mu - T)`, phase on the *cell*
translation only), `H(k + G) = H(k)` and `C(k + G) = C(k)` **exactly**. The
closing link `k_{N-1} -> k_0 + b_d` therefore needs no periodic-gauge correction:
it reuses `C(k_0)` verbatim. Likewise no band gauge fixing is needed — each
`C(k_j)` appears once as bra and once as ket around the closed string, so any
per-band phase and any `U(n_occ)` mixing of degenerate bands cancels from the
product of determinants.

### Sign and quantum conventions

```text
Phi_d = 2 pi sum_A z_A s_{A,d}  -  n_spin * phi_d
P     = (e / (2 pi V)) sum_d a_d Phi_d
```

`z_A` are the GFN1 valence (core) charges (`BasisSet::reference_electrons`) and
`s_{A,d}` the **unwrapped** fractional coordinates. The signs are fixed by the
molecular limit, not by a literature transcription: for a localized closed-shell
density `phi_d -> b_d · sum_n <r>_n`, so `-n_spin phi_d` reproduces the electronic
dipole `-2 sum_n <r>_n`.

`phi_d` is only recovered modulo `2 pi` and enters multiplied by `n_spin = 2`, so
the **spin-restricted polarization quantum is `2 e a_d / V`** — a restricted
calculation moves electrons in pairs. `total_phase_raw` is therefore reduced into
`(-2 pi, 2 pi]` (one quantum wide) to give `total_phase_reduced` /
`polarization` / `dipole`; `polarization_raw` keeps the unreduced branch, and
`quantum[d]` is the quantum vector itself. A **centrosymmetric** crystal has
`P = 0` modulo *half* that quantum (`e a_d / V`), i.e. `Phi_d ≡ 0 (mod 2 pi)`.

### Options

| Field | Default | Meaning |
| --- | --- | --- |
| `mesh` | `[1, 1, 1]` | `mesh[d]` = k-points per string along `d`; the other two entries enumerate the parallel strings. `BerryPolarizationOptions::from_kmesh(kmesh)` takes it from a `KMesh`. |
| `method` | `Auto` | `Auto` = Resta for a Γ-only mesh, KSV otherwise. `Resta` / `KingSmithVanderbilt` force it. |
| `directions` | `[true; 3]` | Which lattice directions to evaluate (non-periodic axes are skipped anyway). Restricting this skips the corresponding boosted-block builds. |
| `occupation_tolerance` | `1e-6` | Largest tolerated deviation of a band occupation from an integer. |
| `min_band_gap` | `1e-6` Ha | Smallest tolerated gap at any Berry k-point. |
| `ao_cutoff` | `None` | Image cutoff for the boosted sums; `None` reuses `PbcOptions::ao_cutoff`. |

```rust
use gfn1_rs::{
    pbc_berry_polarization, BerryMethodSelector, BerryPolarizationOptions,
    ElectronicOptions, KMesh, PbcOptions, POLARIZATION_AU_TO_C_PER_M2,
};

let options = ElectronicOptions { electronic_temperature: 0.0, ..Default::default() };
let pbc = PbcOptions { kmesh: KMesh::monkhorst_pack([4, 4, 4]), ..PbcOptions::default() };
let berry = BerryPolarizationOptions {
    mesh: [4, 4, 4],
    method: BerryMethodSelector::KingSmithVanderbilt,
    ..BerryPolarizationOptions::default()
};
let p = pbc_berry_polarization(&system, &params, &options, &pbc, &berry)?;
println!("P = {:?} e/a0^2 = {:?} C/m^2",
         p.polarization,
         p.polarization.map(|x| x * POLARIZATION_AU_TO_C_PER_M2));
println!("quantum along a3 = {:?}", p.quantum[2]);
```

### Validation (`tests/pbc_polarization.rs`, whole file 6.1 s)

**Molecular limit.** HF centred in a cubic box, Γ-only Resta, against the *exact*
quantum-mechanical dipole `Σ_A z_A R_A − Tr[P r]` built from the same
`lao_dipole_matrix` integrals at `B = 0` (not the Mulliken point-charge dipole —
a Berry phase reproduces the true `<r>`):

| `L` (Å) | `V` (a₀³) | Berry `μ_z` | exact `μ_z` | residual |
| --- | --- | --- | --- | --- |
| 8 | 3455.147 | +0.95341603 | +0.93981799 | 1.3598e-2 |
| 12 | 11661.122 | +0.94469428 | +0.93868816 | 6.0061e-3 |
| 16 | 27641.178 | +0.94179837 | +0.93842739 | 3.3710e-3 |

The residual falls as **1/L²** — the Resta single-point form is exact only as
`b = 2π/L → 0`, its leading error being `O(b²)` times the second moment of the
occupied orbitals. Measured ratios **2.264** (predicted 2.250) and **1.782**
(predicted 1.778).

**KSV string refinement** at fixed box (`L = 8` Å, string along z):

| string points | `μ_z` | residual | ratio |
| --- | --- | --- | --- |
| 1 (= Resta) | +0.95341603 | 1.3598e-2 | — |
| 2 | +0.94318786 | 3.3699e-3 | 4.04 |
| 4 | +0.94065864 | 8.4065e-4 | 4.01 |

i.e. a clean `1/N²` convergence of the string discretisation onto the exact
dipole, which is the gate on the string/link bookkeeping itself.

**Quantum consistency.** Translating *every* HF atom by `a₁` leaves the electronic
Berry phase unchanged to `<1e-12` (proved structurally: each link determinant
picks up `e^{i q·a₁ n_occ}` and the `N` links multiply to `e^{-2πi n_occ} = 1`)
and shifts the raw total phase by **exactly `8.000000 × 2π = 2π N_el`** — whole
quanta — so `total_phase_reduced` and `polarization` are bit-stable.

**Inversion symmetry.** Diamond (the 8-atom cubic fixture) gives
`Φ_d ≡ 0 (mod 2π)` with residual `≤ 2.8e-14` on all three axes, for **both**
Resta and KSV `[2,2,2]`. Rock-salt NaCl likewise gives a reduced phase of exactly
0.

**Negative control + Born charge.** Because diamond's Born charges vanish by
symmetry, the "is the phase actually moving?" control uses heteropolar NaCl:
pushing the whole Na sublattice along z by 0.02 / 0.04 Å gives
`Φ_z = 0.088306693 / 0.176609641` — a linearity ratio of **2.0000** — and
`P_z = 7.08e-3 / 1.4158e-2` C·m⁻². The slope is the Born effective charge
**`Z*(Na) = +0.99 e`**, next to the nominal ionic `+1`.

**KSV ↔ Resta.** On a `[1,1,1]` mesh the forced-KSV string reproduces the forced
Resta value **bit-for-bit** (`|diff| = 0.000e0` on all three axes), exercising the
wrap-around link and the degenerate single-string average.

### Caveats

- **Integer occupations only.** Fractional Fermi occupations or a closed gap at
  any Berry k-point are **rejected**, not approximated: metallic polarization is
  ill-defined. Both error texts are in [limitations.md](limitations.md).
- **Closed shell, restricted.** `n_spin = 2` is hard-wired; the quantum is
  correspondingly `2 e R / V`, not `e R / V`.
- **`ElectronicOptions::multipole` is rejected.** The on-site multipole Fock is not
  carried by `PbcSccResult`, so the states re-diagonalised at the Berry k-points
  would not be the SCC states. Everything else the periodic SCC supports
  (`charge_order`, external electric field, dispersion) works.
- **Charged cells are rejected** — their polarization depends on the coordinate
  origin.
- Only `P` is implemented (level 1). Born-charge tensors are *derivable* from it
  by finite differences (as the NaCl gate does by hand) but there is no analytic
  `dP/dR` entry point, and no finite-field Berry enthalpy.
- No Python or CLI binding: Rust-library API only.
