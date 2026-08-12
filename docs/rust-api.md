# Rust API

The library target is `gfn1_rs` (Cargo package `gfn1-rs`).

**Parameters are bundled.** The official GFN1-xTB parametrization ships inside the
crate, so [`Gfn1Parameters::resolve`]`(None)` just works. The resolution order is
explicit spec (`--param` / the argument) > `GFN1_XTB_PARAM` > builtin; see
[parameters.md](parameters.md) for the specifiers, provenance and
`ParamSource` reporting. (The older [`resolve_param_path`] is a *path-only*
resolver with **no** builtin fallback — it still errors when neither an explicit
path nor the environment variable is set.)

Subsystem deep-dives that this page links into:
[derivatives.md](derivatives.md) (gradient → Hessian → FC3 → FC4 and the
verification gates), [finite-temperature.md](finite-temperature.md) (the
charge-space response solver), [pbc.md](pbc.md) (periodic derivatives), and
[limitations.md](limitations.md).

## Single point

Parse the GFN1 parameters, build a [`PeriodicSystem`], run [`run_electronic`], and
read component energies from [`ElectronicResult::energy_terms`]:

```rust
use gfn1_rs::{run_electronic, ElectronicOptions, Gfn1Parameters, PeriodicSystem};

let params = Gfn1Parameters::resolve(None)?;   // bundled; or Some("path") / Some("builtin:si")
let system = PeriodicSystem::from_xyz_file("examples/water.xyz", 0.0, false)?;

let mut options = ElectronicOptions::default();
options.charge = Some(0.0);
options.spin_multiplicity = None;

let result = run_electronic(&system, &params, options)?;
for (name, value_hartree) in result.energy_terms().named_values() {
    println!("{name}\t{value_hartree:.16}");
}
```

`named_values()` returns the **fifteen** reported terms (all in Hartree, in this
order): `total_free`, `total_internal`, `repulsion`, `electronic`,
`isotropic_scc`, `third_order`, `higher_order`, `dispersion`, `halogen`,
`multipole`, `exchange`, `spin_polarization`, `plus_u`, `external_field`,
`electronic_entropy`.

Before v0.5.0 the breakdown omitted `higher_order`, `multipole`, `exchange`,
`spin_polarization` and `plus_u` even though all five entered `total_internal`, so
`named_values()` could not be reconciled with the total. [`EnergyTerms::component_sum`]
now returns the sum of the component terms (everything except the two totals and
the entropy, which belongs to `total_free`) and is gated to equal `total_internal`
to 1e-12 across the default / multipole / `lr_exchange` / `charge_order = 4` and
spin-polarized paths.

Mulliken atomic and shell charges are on `ElectronicResult::atomic_charges` /
`shell_charges`, and the Mulliken dipole on `ElectronicResult::dipole`.

`run_electronic` (and `analytic_gradient`) auto-dispatch to the periodic path when
the system carries a lattice (extXYZ `Lattice="..."`); see [PBC](#pbc) below.

## SCC convergence controls

Convergence and model controls live on [`ElectronicOptions`] (defaults follow the
tblite GFN1 conventions: 250 iterations, `1e-6` energy, `2e-5` density/mixer RMS,
0.4 damping, 300 K electronic temperature):

```rust
let mut options = ElectronicOptions::default();
options.max_scc = 250;
options.energy_tolerance = 1.0e-6;
options.charge_tolerance = 2.0e-5;     // density / mixer RMS
options.mixing = 0.4;
options.scc_broyden = true;            // Broyden mixing; false -> linear mixing
options.scc_broyden_size = 250;
options.electronic_temperature = 300.0; // K; Fermi/Mermin smearing
options.enable_dispersion = true;
options.hamiltonian.enable_cn_hamiltonian = true;
options.spin_polarization = false;      // spGFN1: collinear spin term (open-shell only, non-PBC);
                                        // with a multiplicity set, `run_electronic` dispatches to
                                        // `crate::spin`. Closed-shell singlet == GFN1 (byte-identical).
```

### Experimental D4 dispersion (v0.4.1)

Set `options.experimental_d4 = true` to replace the default D3(BJ) term with the
experimental non-PBC, self-consistent D4 term, including the ATM three-body
contribution. The D4 scalar potential is included in SCC, and
`analytic_gradient` includes the fixed-charge D4 geometry/CN derivative. `a1`,
`a2`, and `s8` are read from the active `param_gfn1-xtb.txt`; `d4_s9` is an API
override. If `d4_s9` is `None`, D4 calculations use the GFN2-xTB value `5.0`,
while non-D4 calculations resolve to `s9 = 0`. The saved upstream DFT-D4
reference data lives under `third_party/dftd4`.

```rust
let mut options = ElectronicOptions::default();
options.experimental_d4 = true;
options.d4_cutoff = 60.0;
options.d4_cn_cutoff = 30.0;
options.d4_s9 = Some(5.0); // optional; None uses 5.0 only when D4 is active
let result = run_electronic(&system, &params, options)?;
```

### Convergence accelerators (v0.1.2)

`options.scc_accelerator` ([`SccAccelerator`]) selects the SCC charge accelerator
and `options.level_shift` adds a virtual level shift (Hartree):

```rust
options.scc_accelerator = gfn1_rs::SccAccelerator::Cdiis; // or Newton / Broyden / Linear
options.level_shift = 0.1;                                // damp small-gap oscillations
```

- `Broyden` (default) and `Linear` are the historical charge-mixing schemes (with
  the default accelerator, the legacy `scc_broyden = false` still selects linear).
- `Cdiis` is Pulay DIIS on the SCC charge residual.
- `Newton` is a second-order step using the transition-charge susceptibility
  `chi`; the Jacobian `dq_out/dq_in = chi K` (with the full `dv/dq` kernel `K`) is
  finite-difference verified, and the step falls back to Broyden when the local
  solve is singular.

All accelerators converge to the same energy; level shifting leaves the converged
result unchanged.

### Multipole rank-continuation (v0.2.2)

A cold high-rank multipole SCC (`multipole_order >= 4`, i.e. 16-pole and beyond) can be hard
to converge because the monopole↔high-multipole coupling oscillates. [`run_electronic_rank_ladder`]
converges it one rank at a time, warm-starting each rank's shell charges from the previous
(lower-rank) converged result (the warm-start seeds [`ElectronicOptions::scc_initial_shell_charges`]):

```rust
use gfn1_rs::{run_electronic_rank_ladder, ElectronicOptions};

let mut options = ElectronicOptions::default();
options.multipole = true;
// octupole (rank 3) -> 16-pole (4) -> 32-pole (5), staged and warm-started:
let result = run_electronic_rank_ladder(&system, &params, &options, /*base*/ 3, /*target*/ 5)?;
```

It reaches the same SCF solution as a direct `run_electronic` with `multipole_order = 5`, just
more robustly. Ranks `<= 3` use the legacy dipole+quad(+octupole) path; ranks `>= 4` use the
generic arbitrary-rank path. Non-periodic. (The atomic moments are themselves SCF variables,
jointly mixed with the shell charges — the tblite/GFN2 joint-mixing scheme — at every rank.)

## Entry points by task

- **Gradients / forces**: [`analytic_gradient`] returns
  [`AnalyticGradientResult`] (`gradient`, `forces`, `electronic_result`) for the
  H0 / SCC / repulsion / D3 / halogen terms, validated against finite differences.
- **Hessian**: [`analytic_hessian`] returns [`AnalyticHessianResult`]
  (`hessian`, `electronic_result`). [`AnalyticHessianOptions`] toggles each block
  (`include_repulsion`, `include_fixed_scc`, `include_fixed_pulay`,
  `include_fixed_cn_h0`, `include_electronic`, `include_dispersion`,
  `include_halogen`). Lower-level pieces are also public:
  [`analytic_repulsion_hessian`], [`fixed_shell_charge_scc_hessian`],
  [`fixed_density_pulay_hessian`], [`fixed_density_cn_h0_hessian`],
  [`fixed_density_cn_h0_third_derivative`] (the CN-H0 frozen block one order up, `Vec<Matrix>`
  slabs — symmetric in `(b,c)` only, which is why the frozen bundle is summed dense), and
  [`analytic_hessian_from_result`] (reuse a converged SCC).
  Since v0.5.0 both drivers call [`crate::terms::require_order`]`(.., 2, ..)` and **reject**
  option sets whose second-derivative terms are unimplemented (`multipole`, `lr_exchange`,
  `plus_u`, `spin_polarization`, `experimental_d4`, external electric field) instead of
  silently dropping them and returning the Hessian of a different energy expression.
  Both also reject a lattice-bearing system — use [`pbc_gamma_hessian`] / [`pbc_kpoint_hessian`].
  See [derivatives.md](derivatives.md#0-the-term-registry-require_order).
- **Optimization**: [`optimize_geometry`] runs the Rust-native L-BFGS
  ([`GeometryOptimizationOptions`]: `max_iterations`, `gradient_tolerance`,
  `step_tolerance`, `history`, `initial_step`, `max_atom_step`) and returns the
  relaxed [`PeriodicSystem`] with energy, gradient, forces, and convergence flags.
  It accepts **periodic** systems too: a lattice-bearing `PeriodicSystem` relaxes
  the atomic positions at **fixed cell** (the gradient auto-routes to the Γ /
  k-point PBC path and the lattice is preserved). For variable-cell relaxation,
  drive the lattice externally from [`pbc_stress`] (e.g. an ASE cell filter).
- **Vibrational analysis**: [`vibrational_analysis`]`(&hessian, &atomic_numbers)`
  mass-weights a Cartesian Hessian (Hartree/Bohr², ordered `3*atom + axis`) and
  returns [`VibrationalModes`] (`wavenumbers` in cm⁻¹, `eigenvalues`, `modes`).

```rust
use gfn1_rs::{analytic_hessian, vibrational_analysis, AnalyticHessianOptions};

let hess = analytic_hessian(&system, &params, AnalyticHessianOptions::default())?;
let numbers: Vec<u8> = system.atoms.iter().map(|a| a.z).collect();
let modes = vibrational_analysis(&hess.hessian, &numbers)?;
println!("{:?}", modes.wavenumbers); // ascending; negative = imaginary
```

## PBC

The periodic code lives under `src/pbc/` and is isolated from the molecular
modules. All periodic entry points share the signature
`(system, params, options, pbc)`, where [`PbcOptions`] selects the k-mesh
([`KMesh::gamma`] or [`KMesh::monkhorst_pack`]), the generalised-Ewald controls
([`EwaldOptions`]), and the AO image cutoff. [`PbcOptions::for_boundary`] picks a
mesh from the [`ElectronicOptions::boundary`] (Gamma-only for a bare lattice, a
default Monkhorst-Pack mesh for explicit k-point boundaries).

```rust
use gfn1_rs::{
    pbc_analytic_gradient, pbc_gamma_hessian, pbc_kpoint_hessian, pbc_stress, run_pbc_scc,
    ElectronicOptions, KMesh, PbcOptions,
};

let options = ElectronicOptions::default();
let pbc = PbcOptions { kmesh: KMesh::monkhorst_pack([4, 4, 4]), ..PbcOptions::default() };

let scf  = run_pbc_scc(&system, &params, &options, &pbc)?;          // PbcSccResult
let grad = pbc_analytic_gradient(&system, &params, &options, &pbc)?; // gradient + forces
let strs = pbc_stress(&system, &params, &options, &pbc)?;            // component stress
let hess = pbc_gamma_hessian(&system, &params, &options, &pbc)?;     // Gamma-point Hessian
let hk   = pbc_kpoint_hessian(&system, &params, &options, &pbc)?;    // k-point Hessian
```

The k-point Hessian's coupled complex CPXTB is solved with preconditioned conjugate
gradient (v0.1.2); the previous fixed-point iteration could diverge when the SCC
coupling outweighed the orbital gaps (it produced unphysical Hessian elements on
bromoethanol off Gamma). It now matches the k-point gradient finite difference.

Implementation notes:

- `H0(k)` / `S(k)` are built from Bloch sums of the per-image `H0(T)` / `S(T)` and
  solved through a real `2n` embedding of the complex Hermitian generalized
  problem, reusing the molecular Löwdin solver and Broyden mixer.
- The Klopman–Ohno second-order electrostatics use a QCore-style **generalised
  Ewald** partitioning (Buccheri et al. 2025): the standard `1/R` Ewald sum, the
  `R⁻³` binomial Ewald term, and a rapidly decaying short-range residual. The
  result is alpha-independent and reduces to the molecular limit (both checked).
- A single global Fermi level is filled across the Brillouin zone, so the
  analytical force is the gradient of the Mermin free energy `E − TS`. Validated by
  finite-difference gradient tests at Gamma and for a fractionally-occupied metal.
- The Gamma-point [`pbc_gamma_hessian`] path also assembles the periodic
  repulsion, D3(BJ), and halogen-bond Cartesian Hessian blocks.
- [`run_electronic_pbc`] / [`pbc_electronic_result`] project the periodic SCC into
  the molecular-shaped [`ElectronicResult`] so periodic and molecular callers share
  the same energy-term and charge accessors.
- The higher-order **on-site charge expansion** [`ElectronicOptions::charge_order`]
  (`4` = quartic Breathing-Radius, `5+` = higher) is honoured by the periodic SCC too
  (v0.2.2): it is a per-atom local term (no lattice sum), added to the k-point SCC
  energy and shell potential exactly as in the molecular path. Set `options.charge_order
  = 4` and it applies to `run_pbc_scc` / `pbc_analytic_gradient` as well.
- The **angular multipole correction** [`ElectronicOptions::multipole`] is now periodic at
  **arbitrary rank** (v0.2.2): `run_pbc_scc` runs the mDFTB2 multipole SCC with the atomic moments
  (ranks `1..=L`) mixed **jointly** with the shell charges in one Broyden vector (tblite/GFN2
  style). `L` follows the molecular convention — explicit `multipole_order ≥ 1` sets the rank; the
  bare `multipole` flag defaults to **dipole+quadrupole** (rank 2). The periodic field is the QCore
  generalized-Ewald `periodic_multipole_fields_generic` (every rank-pair real-space + reciprocal +
  rank-diagonal self), rebuilt from the mixed state each iteration; it enters through two Fock
  routes — the rank-0 charge potential folds into the shell potential, the rank-≥1 moment operator
  into the on-site Fock added to every `H(k)`. The converged total energy is **α-independent** (the
  Ewald correctness property, exercised through the whole loop), and the quadrupole SCC differs
  from dipole-only. **Energy, analytic gradient, and analytic stress are all complete**, FD/α-gated
  at rank 2 (dipole+quadrupole):
  - **Gradient** (`pbc_analytic_gradient`): the inter-atomic kernel force
    (`periodic_multipole_forces_generic`, all rank pairs, periodic images) + the on-site
    overlap-Pulay term `∂E_mp/∂S·dS/dR` (`multipole_weight_from_fields` contracted with the
    reference-cell overlap derivative).
  - **Stress** (`pbc_stress`): the kernel strain (the geometric field's strain derivative, via an
    α-independent central difference at fixed moments — no basis rebuild) + the overlap-Pulay strain
    (`2·W·dS/dε` virial). On `PbcStressResult::multipole_stress`.
  - Converged moments on `PbcSccResult::atomic_moments`; the high-rank field/force use
    moment-active-mask screening (`O(N·n_active)`). **Periodic multipole Part A is complete.**

### Periodic third derivative, seminumerical (v0.5.0)

Central finite differences of the **analytic** PBC Hessian — each `±` displacement re-runs the
full periodic SCC and Hessian, so the result inherits the FD-validated analytic Hessian rather
than double-differencing energies. Γ-point and Brillouin-zone-sampled variants share every
convention and differ only in which Hessian they differentiate:

```rust
use gfn1_rs::{
    pbc_kpoint_strain_hessian_derivative, pbc_kpoint_third_derivative_seminumerical_dense,
    pbc_kpoint_third_derivative_seminumerical_vector, pbc_strain_hessian_derivative,
    pbc_third_derivative_seminumerical_dense, pbc_third_derivative_seminumerical_vector,
    scale_lattice_isotropic,
};

// Γ-point dense: `3N` slabs `slabs[c][(a, b)] = d(H_ab)/dR_c`; cost `2*3N` analytic Hessians.
let slabs = pbc_third_derivative_seminumerical_dense(&system, &params, &options, &pbc, 1.0e-3)?;
// Γ-point vector: `K_ab = Σ_c v_c d(H_ab)/dR_c` in one `3N×3N` matrix. DOFs with `v_c == 0.0`
// are skipped (cost `2*nnz(v)`), and it is bit-for-bit the contraction of the dense mode.
let kv    = pbc_third_derivative_seminumerical_vector(&system, &params, &options, &pbc, 1.0e-3, &v)?;
// The same two against `pbc_kpoint_hessian` over `pbc.kmesh` (a 1×1×1 mesh reduces to Γ):
let kslabs =
    pbc_kpoint_third_derivative_seminumerical_dense(&system, &params, &options, &pbc, 1.0e-3)?;
let kkv =
    pbc_kpoint_third_derivative_seminumerical_vector(&system, &params, &options, &pbc, 1.0e-3, &v)?;
// Strain-mixed `d(H_ab)/d(ln V)` under isotropic frozen-ion strain — exactly 2 Hessians each.
let dhdlnv  = pbc_strain_hessian_derivative(&system, &params, &options, &pbc, 5.0e-3)?;
let dhdlnvk = pbc_kpoint_strain_hessian_derivative(&system, &params, &options, &pbc, 5.0e-3)?;
```

Nuclear displacements are *absolute Cartesian* shifts with the lattice held fixed; the volumetric
strain scales the three lattice vectors by `(1 ± delta)^(1/3)` with **frozen fractional**
coordinates ([`scale_lattice_isotropic`], which errors on a non-finite or non-positive scale), and
the denominator is the exact log-volume separation `Δ ln V = ln((1 + δ)/(1 − δ))`, not `2δ`. The
dense slabs are deliberately **not** re-symmetrised across `c`, so the acoustic sum rule stays a
genuine test.

**Errors.** Every entry point above (including [`scale_lattice_isotropic`]) rejects a system with
no lattice; the third-derivative entry points reject a non-finite or non-positive `step` and
(vector mode) a direction whose length is not `3N`; the two strain routines reject a non-finite
`delta` or one outside `(0, 1)`. **Nothing here rejects a nonzero
`electronic_temperature`.** The periodic finite-temperature response is a *direct dielectric
solve* that verifies its own residual as of v0.5.0 — it errors instead of silently returning an
unconverged response — so a smeared run is no longer unsound, but every estimator here differences
a **reconverged** quantity, so metallic SCC noise enters divided by `2·step`. Tighten
`charge_tolerance` / `energy_tolerance` well past their defaults first, and watch for band
reordering between the `+`/`−` geometries. Full details in [pbc.md](pbc.md).

### Periodic third derivative, analytic (v0.5.0)

Closed-form periodic cubic force constants, at **Γ** and across a **k-mesh**, in the same
Vector / Dense / Block trio as the molecular path. Both assemble the directional scalar
`e³[v] = Σ_abc T_abc v_a v_b v_c`; Dense and Block recover the packed [`SymmetricThird`] from
directional evaluations by the cubic polarization identity
`T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]`, deduplicating the subset
directions and evaluating them in parallel against **one** shared reference:

```rust
use gfn1_rs::{
    pbc_gamma_third_analytic_block, pbc_gamma_third_analytic_dense,
    pbc_gamma_third_analytic_vector, pbc_gamma_third_with_reference,
    pbc_kpoint_third_analytic_block, pbc_kpoint_third_analytic_dense,
    pbc_kpoint_third_analytic_vector, pbc_kpoint_third_with_reference,
    GammaThirdReference, KpointThirdReference,
};

// Γ-point: cheapest mode first.
let e3    = pbc_gamma_third_analytic_vector(&system, &params, &options, &pbc, &v)?;   // f64
let dense = pbc_gamma_third_analytic_dense(&system, &params, &options, &pbc)?;        // SymmetricThird
let block = pbc_gamma_third_analytic_block(&system, &params, &options, &pbc, &[0, 3])?;

// Many directions sharing one SCC + MO solve + skeleton build + charge-space factorization:
let reference = GammaThirdReference::build(&system, &params, &options, &pbc)?;
let e3b = pbc_gamma_third_with_reference(&system, &params, &options, &pbc, &reference, &v)?;
reference.scc();   // &PbcSccResult the reference was built on
reference.mos();   // &GammaMos (coefficients, energies, occupations, overlap)

// k-point: the same four entry points over `pbc.kmesh`. The reference additionally hoists the
// `3N` coupled-perturbed solves, so each further direction is a contraction.
let ke3 = pbc_kpoint_third_analytic_vector(&system, &params, &options, &pbc, &v)?;
let kref = KpointThirdReference::build(&system, &params, &options, &pbc)?;
let ke3b = pbc_kpoint_third_with_reference(&system, &params, &options, &pbc, &kref, &v)?;
```

Note the signature difference from the molecular path: these take `&ElectronicOptions` and
`&PbcOptions`, not an `AnalyticHessianOptions` + coordination cutoff.

**Errors**, in the order they fire:

- [`crate::terms::require_order`]`(.., 3, ..)` — `multipole`, `lr_exchange`, `plus_u`,
  `spin_polarization`, `experimental_d4` and an external electric field are all capped at analytic
  order 1 and are rejected.
- **Γ only**: a non-Gamma `pbc.kmesh` is rejected with a pointer to the k-point or seminumerical
  routes. The k-point path deliberately has **no** mesh restriction — a Gamma-only mesh is accepted
  there and reproduces the Γ path (gated to rel `1e-9`).
- No lattice; a periodic SCC that did not converge.
- **Integer (gapped) occupations only** — a Fermi-smeared filling is rejected (at *every* k-point
  for the k-point path) with a pointer to `pbc_*_third_derivative_seminumerical_*`. There is no
  periodic finite-temperature analytic FC3.
- Direction length ≠ `3N` (vector / with-reference), or a block DOF ≥ `3N`.

**Accuracy caveat.** The assembly agrees with the Richardson-extrapolated seminumerical reference
to `~1e-7` absolute (`8.5e-8` skew diamond, `8.0e-8` skew BN, on a `~1e-3` Eh/Bohr³ scale). That
residual is `h`-independent — a genuinely missing term localised to the band/Pulay block, not FD
truncation — and the k-point path inherits it exactly, since it reuses the Γ assembly. If you need
better than `1e-7` absolute, use the seminumerical route. Exact invariances are clean: the
acoustic sum rule holds to `<1e-10` and the polarization identity reproduces the directional
evaluator to rel `1e-9`. Full details in [pbc.md](pbc.md) and
[limitations.md](limitations.md).

### Mode and thermodynamic Grüneisen parameters (v0.5.0)

```rust
use gfn1_rs::{pbc_gruneisen, GruneisenOptions, GruneisenResult, SecondOrderStencil};

let mut gopts = GruneisenOptions::default();  // delta 5e-3, T = [300 K], acoustic_modes 3
gopts.second_order = true;                    // also fit gamma2_i = d² ln ω_i / d(ln V)²
gopts.second_order_stencil = SecondOrderStencil::FivePoint;  // default: ::ThreePoint
gopts.delta_second = Some(2.0e-2);            // the SECOND-order node set's own strain
gopts.kpoint = true;                          // strained Hessians via `pbc_kpoint_hessian`
let g: GruneisenResult = pbc_gruneisen(&system, &params, &gopts)?;

println!("gamma_th(300 K)  = {:?}", g.gamma_at(300.0));        // Option<f64>
println!("gamma2_th(300 K) = {:?}", g.gamma2_at(300.0));       // mode average
println!("full            = {:?}", g.gamma2_full_at(300.0));   // + the dc_i/dlnV reweighting
let q: Vec<f64> = g.mode_q();                                  // q_i = gamma2_i / gamma_i
println!("worst optical mode match = {}", g.min_optical_overlap());
```

Three analytic PBC Hessians (`V0`, `V0(1 ± delta)`) under isotropic frozen-ion volumetric strain,
with modes matched across volumes by eigenvector overlap and degenerate subspaces averaged. It is
**frozen-ion** (no internal relaxation at the strained volumes) and `q = 0`, but **no longer
Γ-only**: with `GruneisenOptions::kpoint` the strained Hessians go through
[`pbc_kpoint_hessian`] on `options.pbc.kmesh`, which converges the *electronic* Brillouin-zone sum
behind each Hessian (it does **not** make this a phonon-dispersion average — that would need
dynamical matrices at finite `q`). The real-space cutoffs are scaled with the strain so the
integer image set is identical at every node.

`delta_second` is **new in v0.5.0**: the volumetric strain of the *second-order* node set,
independent of `delta`, default `Some(`[`DEFAULT_GRUNEISEN_DELTA_SECOND`]`)` = `2e-2`, `None`
reusing `delta`. `gamma2` is a *second* difference of `ln λ(ln V)`, so noise enters it as `ε/δ²`
where the first-order `gamma` only suffers `ε/δ`; one shared step cannot serve both. At
`delta = 5e-3` the three- and five-point stencils return `-0.0372` and `+0.0674` for the same
diamond `gamma2` — opposite signs — while `gamma` agrees to three digits. Splitting the step costs
two extra Hessians (four with [`SecondOrderStencil::FivePoint`]); setting it to `Some(delta)` or
`None` restores the historical zero-extra-cost behaviour together with its noise.
`DEFAULT_GRUNEISEN_DELTA_SECOND` lives in `gfn1_rs::pbc::gruneisen`; [`SecondOrderStencil`] is
re-exported at the crate root.

With `second_order = false` (the default) the second-order fields of [`GruneisenResult`] are
`NaN` / empty and the run is bit-for-bit what it was before the option existed.

**Errors:** no lattice; a `delta` that is non-finite or outside `(0, 1)`; `acoustic_modes` greater
than `3N`; and — only when `second_order` is on — the same range check on the effective
`delta_second`, plus a rejection of `delta_second >= 0.5` under the five-point stencil. Keep
`electronic.electronic_temperature = 0`: as with the seminumerical estimators above, nothing here
rejects smearing.

### Berry-phase bulk polarization (v0.5.0)

```rust
use gfn1_rs::{
    pbc_berry_polarization, BerryMethodSelector, BerryPolarizationMethod,
    BerryPolarizationOptions, BerryPolarizationResult, ElectronicOptions, KMesh,
    PbcOptions, POLARIZATION_AU_TO_C_PER_M2,
};

let options = ElectronicOptions { electronic_temperature: 0.0, ..Default::default() };
let pbc = PbcOptions { kmesh: KMesh::monkhorst_pack([4, 4, 4]), ..PbcOptions::default() };
let berry = BerryPolarizationOptions {
    mesh: [4, 4, 4],                                   // k-points per string along each axis
    method: BerryMethodSelector::KingSmithVanderbilt,  // or ::Resta / ::Auto
    ..BerryPolarizationOptions::default()
};
let p: BerryPolarizationResult =
    pbc_berry_polarization(&system, &params, &options, &pbc, &berry)?;

assert_eq!(p.method, BerryPolarizationMethod::KingSmithVanderbilt);
p.polarization;        // [f64; 3] e/a0^2, from the branch-reduced phase
p.polarization_raw;    // [f64; 3] from the raw (unreduced) phase
p.dipole;              // P * V (e a0) — comparable to a molecular dipole
p.quantum;             // [[f64; 3]; 3]: n_spin * e * a_d / V, one vector per direction
p.electronic_phase;    // per-spin-channel Berry phase phi_d, in (-pi, pi]
p.ionic_phase;         // 2 pi sum_A z_A s_{A,d}
p.total_phase_raw;     // ionic - n_spin * electronic, unwrapped
p.total_phase_reduced; // the same, reduced into (-2 pi, 2 pi]
p.string_phases;       // [Vec<f64>; 3] per-string phases (perpendicular-mesh diagnostic)
p.min_band_gap;        // smallest gap seen over the Berry k-points
```

The module is `gfn1_rs::pbc::polarization`; all six names above
(`pbc_berry_polarization`, `BerryPolarizationOptions`, `BerryPolarizationResult`,
`BerryPolarizationMethod`, `BerryMethodSelector`, `POLARIZATION_AU_TO_C_PER_M2`)
are re-exported at the crate root. `BerryPolarizationOptions::from_kmesh(kmesh)`
takes the Berry mesh from an existing [`KMesh`].

The AO ingredient — the boosted overlap `<chi_mu | e^{i q . r} | chi_nu>` — is
[`crate::magnetic::boosted_overlap_pair`], a thin public wrapper over the same
complex Gaussian product theorem [`lao_overlap_matrix`] uses, at `chi = -q`. It is
also re-exported at the crate root as `boosted_overlap_pair` and is useful on its
own for any plane-wave-boosted AO matrix element.

Three further knobs on [`BerryPolarizationOptions`]: `directions: [bool; 3]` restricts which
lattice directions are evaluated (the others report zero and `evaluated[d] == false`),
`occupation_tolerance` is how far a Fermi occupation may sit from `0`/`2`, and `min_band_gap` is
the smallest gap accepted at any Berry k-point. `ao_cutoff: Option<f64>` overrides the AO image
cutoff for the boosted-overlap build.

**Errors** — every one of these is an explicit `Err`, never a silently wrong number: no lattice; a
non-neutral cell; an SCC that did not converge; a non-integer / open-shell band filling; a band gap
below `min_band_gap` or a Fermi occupation outside `occupation_tolerance` at any Berry k-point;
`ElectronicOptions::multipole` (the multipole Fock is not carried by `PbcSccResult`, so the Berry
states would not be the SCC states); and a vanishing Berry link determinant, which asks for a finer
mesh. Messages are quoted in [limitations.md](limitations.md). No Python or CLI binding.
Conventions, gate numbers and caveats: [pbc.md](pbc.md#7-berry-phase-bulk-polarization).

## External electric field

A uniform field is set on [`ElectronicOptions::external_field`]
([`ExternalFieldOptions`], atomic units). It is coupled to the GFN1 Mulliken
monopoles as a site potential `v_ext_i = -E·(R_i - origin)` folded into the
self-consistent shell potential, so the density polarizes and the analytic
gradient/stress carry the field. The energy term is `external_field` and the
Mulliken dipole `mu = sum_A q_A (R_A - origin)` is on `ElectronicResult::dipole`.
Works for both non-periodic and periodic (`run_pbc_scc` / `pbc_analytic_gradient`)
systems; FD-verified for energy↔gradient and `mu = -dE/dE`.

```rust
use gfn1_rs::{run_electronic, ElectronicOptions, ExternalFieldOptions};
use gfn1_rs::math::Vec3;

let mut options = ElectronicOptions::default();
options.external_field = ExternalFieldOptions::electric(Vec3::new(0.0, 0.01, 0.0));
let result = run_electronic(&system, &params, options)?;
println!("dipole = {:?}", result.dipole);
```

## Cubic force constants (nuclear third derivative)

The non-PBC nuclear **third derivative** `T_abc = ∂³E/∂R_a∂R_b∂R_c` (cubic force constants —
for anharmonic vibrational corrections and mode coupling) is available three ways: a strict
`T = 0` closed form, a seminumerical FD of the analytic Hessian, and (v0.5.0) a natively
Fermi-smeared analytic path.

**(1) Strict closed form (v0.3.0).** [`third_derivative_analytic`] assembles the full tensor by
the **2n+1 rule with no finite differences anywhere**: `T_abc = D_c H_frozen + D_c R_static +
D_c R_orbital`, where the response derivative is the analytic Z-vector route (the orbital-amplitude
derivative is closed by the self-adjoint Z-vector `y_a = A⁻¹ L_a`, never a second-order CPHF solve).
It matches `FD(full analytic Hessian)` to ~7e-5 at **equilibrium** water, and — after the v0.4.4
Pulay coordination-number-response term (`fixed_density_pulay_cn_h0_response`; the frozen-density
`2P·∂h0/∂CN·∂CN/∂R` piece the P/W/V density-path omitted, since `h0` reads a CN cached in the
electronic result) — to **~1e-4 at strongly non-equilibrium** geometries too (stretched+bent water:
analytic vs seminumerical 7.4e-7, vs double-gradient FD 9.2e-6; was 6.1e-4 / 0.2% before v0.4.4).
Every component is independently FD-gated. Use a tight SCF (`energy_tolerance 1e-11`,
`charge_tolerance 1e-9`) for best accuracy.

The response slabs are computed in **parallel** over the shared `rayon` pool (v0.3.3), so the
wall-clock cost scales down with cores. It has **three output modes** (v0.3.1) so callers never need
to *return/hold* the full `ndof³` tensor when a direction or a local block is all that is wanted:

```rust
use gfn1_rs::{third_derivative_analytic, third_derivative_analytic_dense,
              third_derivative_analytic_vector, third_derivative_analytic_block,
              third_derivative_seminumerical_vector, AnalyticHessianOptions};

let opts = AnalyticHessianOptions::default();
let cutoff = opts.electronic_options.hamiltonian.coordination_cutoff;

// Dense — `ndof` dense slabs `slab[c][(a, b)] = T_abc` (backward-compatible):
let slabs = third_derivative_analytic(&system, &params, opts.clone(), cutoff)?;
// Dense, symmetric-packed — the same data in a `SymmetricThird` (~1/6 the memory; `.get(a,b,c)`,
// `.contract_last(v)`, `.block(dofs)`, `.contract_vvv(v)`):
let packed = third_derivative_analytic_dense(&system, &params, opts.clone(), cutoff)?;
// Vector (recommended when you need a direction) — the directional `K[a][b] = Σ_c v_c T_abc` returned
// as ONE `3N×3N` matrix (returns only the contraction, not the full `ndof³` tensor):
let kv = third_derivative_analytic_vector(&system, &params, opts.clone(), cutoff, &v)?;
// Block — the `O(|block|³)` sub-tensor over the DOFs of the chosen atoms (local anharmonicity):
let (dofs, block) = third_derivative_analytic_block(&system, &params, opts.clone(), cutoff, &[0, 2])?;
```

**Errors (all four).** [`crate::terms::require_order`]`(.., 3, ..)` rejects any active term capped
below analytic order 3 (`multipole`, `lr_exchange`, `plus_u`, `spin_polarization`,
`experimental_d4`, an external electric field), and **fractional (Fermi-smeared) occupations are
rejected outright** — the closed-form response-derivative algebra assumes integer `0/2`
occupations. Use `third_derivative_seminumerical_*` or the `finite_t` drivers below for smeared
systems. The Vector mode additionally rejects a direction whose length is not `3N`. Note that
`third_derivative_analytic_block` / `third_derivative_seminumerical_block` take **atom** indices
(each contributing its 3 Cartesian DOFs), while the newer `finite_t` / PBC / FC4 block entry points
take **DOF** indices and validate their range explicitly.

**(2) Semi-numerical (cheap production path).** A finite difference of the FD-validated analytic
Hessian, packed into a symmetric-packed [`SymmetricThird`] (`n(n+1)(n+2)/6` entries) with
`Dense`/`Block`/`Vector` output. The **Vector** mode is the cheapest — the directional cubic
constant along `v` (e.g. a normal mode) needs only **two** Hessian evaluations:

```rust
// Directional (cheap): K[a][b] = Σ_c v_c T_abc, the derivative of the Hessian along `v`.
let k = third_derivative_seminumerical_vector(&system, &params, opts.clone(), &v, 1.0e-3)?;
// Full tensor (2·ndof Hessian evals), symmetric-packed:
let t = third_derivative_seminumerical_dense(&system, &params, opts.clone(), 1.0e-3)?; // SymmetricThird
let _scalar = t.contract_vvv(&v);   // T[v,v,v]: cubic anharmonicity along one mode

// Block mode (OOM-saving): only the sub-tensor over a chosen atom subset, computed WITHOUT the
// full ndof³ tensor — only |dofs| Hessian pairs (along the in-block axes). Bit-for-bit the Dense
// sub-block (same canonical FD path), so memory and compute scale with the subset, not N.
let (dofs, slabs) =
    third_derivative_seminumerical_block(&system, &params, opts, &[0, 3], 1.0e-3)?; // atoms 0,3
```

Both are validated by translational invariance (a uniform shift gives zero) and full permutation
symmetry. The CLI exposes the closed form as `gfn1_rs … --third-derivative` (alias `--cubic`;
add `--matrices` to print each `T_abc` slab), and rejects periodic systems there.

The seminumerical route carries **no occupation guard of its own** — it inherits whatever
[`analytic_hessian`] accepts, including its [`crate::terms::require_order`]`(.., 2, ..)` check —
so fractional occupations are fully supported (see below). It also does not validate `v` or the
`atoms` list, so pass a full `3N` direction and in-range atom indices.

**(3) Analytic with Fermi smearing (v0.5.0)** — Vector, Dense and Block.

```rust
use gfn1_rs::{directional_third_finite_t, third_derivative_finite_t_block,
              third_derivative_finite_t_dense, FiniteTThirdReference};
use gfn1_rs::third_derivative::finite_t::directional_third_with_reference;

// Vector — e³[v] = Σ_abc T_abc v_a v_b v_c, one scalar; `options` is passed BY REFERENCE here.
let e3 = directional_third_finite_t(&system, &params, &opts, cutoff, &v)?;
// Dense / Block — the packed `SymmetricThird`, recovered from shared-reference directional
// evaluations by the cubic polarization identity
// `T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]`, deduplicated and
// evaluated in parallel. Block is indexed by POSITION in `dofs`.
let t  = third_derivative_finite_t_dense(&system, &params, &opts, cutoff)?;
let tb = third_derivative_finite_t_block(&system, &params, &opts, cutoff, &[0, 3])?;

// Many directions against one SCF + CPXTB + charge-space factorization + frozen slabs:
let reference = FiniteTThirdReference::build(&system, &params, &opts, cutoff)?;
let e3b = directional_third_with_reference(&system, &params, cutoff, &reference, &v)?;
```

Fully analytic and **natively Fermi-smeared**: `e³[v] = D_v[H[v,v]]` assembled by the product
rule over the directional Hessian's composition, using the second-order screened bundle `X^vv`
from the charge-space solver directly, so no adjoint/Z-vector machinery is needed. Every
ingredient is occupation-agnostic, so one code path serves `T = 0` and smeared systems alike.
Gates: `T = 0` equality vs the adjoint-assembled [`third_derivative_analytic_vector`] at 2.3e-15;
smeared distorted Ni(CO)₄ at 3000 K vs the FD of the finite-T analytic Hessian at
`δ(h) = 6.25e-10`, `δ(h/2) = 1.57e-10`, ratio 3.98; block-vs-dense consistency.

**Errors.** [`FiniteTThirdReference::build`] runs [`crate::terms::require_order`]`(.., 3, ..)`, so
the experimental model flags (`multipole`, `lr_exchange`, `plus_u`, `spin_polarization`,
`experimental_d4`, an external electric field) are rejected; the directional entry points reject a
direction whose length is not `3N`, and the block entry point a DOF index `>= 3N`. **Fractional
occupations are not an error here** — that is the whole point of this path — and neither are
*exactly degenerate* fractionally occupied blocks any more: the second-order charge-space solve
switches to the frame-free Daleckii–Krein resolvent form for exactly those references, and the
guard that used to refuse them is gone. (The remaining exact-degeneracy rejection is one order up,
in the **third**-order response used by the finite-T FC4 — see below.)

`directional_third_finite_t`, `third_derivative_finite_t_dense`, `third_derivative_finite_t_block`
and `FiniteTThirdReference` are re-exported at the crate root;
`directional_third_with_reference` needs the full path
`gfn1_rs::third_derivative::finite_t::directional_third_with_reference`. There is no CLI flag;
Python has `third_derivative_finite_t_directional` / `third_derivative_finite_t` /
`third_derivative_finite_t_block`. See
[finite-temperature.md](finite-temperature.md#3-analytic-finite-temperature-fc3).

**Smearing on the other two routes.** The **semi-numerical** path supports it transparently: with
`options.electronic_options.electronic_temperature > 0` the analytic Hessian it differentiates is
the electronic **free-energy** Hessian, so the result is the free-energy cubic force constant
(e.g. Ni(CO)₄ at 3000 K has fractional frontier occupations, yet the directional third derivative
along a rigid translation still vanishes). The **strict closed form**
(`third_derivative_analytic*`) rejects fractional occupations with an explicit error — its
response-derivative algebra assumes integer (0/2) occupations and would otherwise be silently
wrong. See [limitations.md](limitations.md#finite-temperature--fermi-smearing).

### Analytic frozen `L_abc` blocks

The frozen (response-free) third-derivative blocks of the fully-analytic path are individually
exported and FD-isolatable (each carries no electronic response, so its third derivative matches
the finite difference of its *own* analytic Hessian):

```rust
use gfn1_rs::{third_derivative_geometric, third_derivative_dispersion,
              dispersion_third_derivative};

// Repulsion + halogen (classical central + Jet3), symmetric-packed:
let geo = third_derivative_geometric(&system, &params)?;          // SymmetricThird
// D3-BJ dispersion, full many-body C6(CN(R)) chain rule via Jet3 forward-AD:
let disp = third_derivative_dispersion(&system, &params, None)?;  // SymmetricThird
// ...or the dense tensor + energy directly:
let dense = dispersion_third_derivative(&system, &params, None)?; // DispersionThirdResult

// The full frozen L_abc bundle = repulsion + halogen + frozen SCC2 + frozen Pulay + dispersion:
use gfn1_rs::third_derivative_frozen_full;
let frozen = third_derivative_frozen_full(&system, &params, &electronic, None)?;
```

`third_derivative_frozen_full` merges all five response-free blocks into one `SymmetricThird` and
FD-validates against the sum of their analytic Hessians. It is the complete frozen part of the
2n+1 third derivative bar the CN-H0 frozen block; the CPHF response cross-terms
(`L_abx`/`L_axx`/`L_xxx`) are added on top for the fully-analytic path.

Two more bundle drivers are re-exported at the crate root.
[`third_derivative_frozen`]`(&system, &params, &electronic)` is the same sum **without**
dispersion (geometric + frozen SCC2 + frozen Pulay), and
[`third_derivative_frozen_complete`]`(&system, &params, &electronic, reference_path,
coordination_cutoff, include_dispersion)` adds the **CN-H0** block and returns **dense** slabs
(`Vec<Matrix>`, `slab[c][(a,b)]`) rather than a packed `SymmetricThird` — deliberately, because the
frozen bundle is only `(b,c)`-symmetric once CN-H0 is in it, and packing it alone would discard the
asymmetry the CPHF response has to cancel. Its `include_dispersion` flag is load-bearing: D3's
third derivative is nonzero even for `reference_path = None`, so passing `true` when the caller
disabled dispersion adds a spurious term. (The response-free per-block builder
`third_derivative_frozen_electronic` is public under `gfn1_rs::third_derivative` but not at the
crate root.)

The dispersion third derivative is the `Jet2 → Jet3` promotion of the analytic dispersion
Hessian — the same energy expression carried one AD order higher, so the many-body coordination
chain `C6(CN(R))`, the reference-weight softmax, and the BJ radial term all propagate their third
derivative automatically (no hand-coded Faà di Bruno). It is validated against the finite
difference of the analytic dispersion Hessian, and its dense tensor is fully permutation-symmetric
(`∂³E` invariance), packing exactly into the canonical `SymmetricThird`. Dense `ndof³` storage ⇒
small molecules / reference checks.

The **D3 ATM (three-body)** term rides the same promotion. Its triple energy is written once
against a shared jet op set and instantiated at second, third and fourth order, so `s9 != 0` no
longer restricts dispersion to energy + gradient: the analytic Hessian, third and fourth
derivatives all include it (molecular `i<j<k` and lattice-summed loops alike). Stock GFN1 has
`s9 = 0`, so the two-body results are unchanged.

### Analytic dispersion fourth derivative

```rust
use gfn1_rs::{dispersion_fourth_derivative, MAX_FOURTH_DERIVATIVE_NDOF};

// Dense ndof⁴ quartic force constants (two-body always, ATM when s9 != 0):
let q = dispersion_fourth_derivative(&system, &params, None)?; // DispersionFourthResult
let value = q.fourth[((a * q.ndof + b) * q.ndof + c) * q.ndof + d];
```

A full-space `Jet4` stores `ndof⁴` doubles, so this driver — and its halogen sibling
[`crate::halogen::halogen_fourth_derivative`] — reject systems above
`MAX_FOURTH_DERIVATIVE_NDOF` (30 DOF / 10 atoms) with an explicit error rather than exhausting
memory. **These two are the only Rust entry points that still check the cap** (see the FC4 section
below). Within that cap it keeps only `O(nat)` jets alive: coordination-number jets are dropped
once the reference weights are built, pair `C6` jets are rebuilt on demand from those weights with
a one-row cache, and squared interatomic distances — quadratic in the coordinates — are seeded in
closed form instead of formed as jet products. Validated against the finite difference of
`dispersion_third_derivative`, plus the fourth-order acoustic sum rule and full permutation
symmetry.

## Quartic force constants (nuclear fourth derivative)

`Q_abcd = ∂⁴E/∂R_a∂R_b∂R_c∂R_d`, fully analytic with **no nuclear finite differences**
(v0.5.0), non-PBC. The `T = 0` closed form below is **integer occupations only**; a natively
Fermi-smeared analytic quartic landed alongside it and is documented at the end of this section.
`directional_fourth_derivative`, `directional_fourth_seminumerical`,
`fourth_derivative_analytic_dense`, `fourth_derivative_analytic_block`, `QuarticReference` and
`SymmetricFourth` **are** re-exported at the crate root; `directional_fourth_with_reference` and
the individual stage functions need the `gfn1_rs::fourth_derivative::…` path:

```rust
use gfn1_rs::fourth_derivative::{
    directional_fourth_derivative, directional_fourth_with_reference,
    directional_fourth_seminumerical, fourth_derivative_analytic_dense,
    fourth_derivative_analytic_block, QuarticReference, SymmetricFourth,
};
use gfn1_rs::hessian::AnalyticHessianOptions;

let opts = AnalyticHessianOptions::default();
let cutoff = opts.electronic_options.hamiltonian.coordination_cutoff;

// Directional (the memory-lean flagship mode): the scalar e⁗[v] = Σ_abcd Q_abcd v_a v_b v_c v_d.
let e4 = directional_fourth_derivative(&system, &params, &opts, cutoff, &v)?;

// Many directions sharing one SCF + CPXTB + charge-space reference:
let reference = QuarticReference::build(&system, &params, &opts, cutoff)?;
let e4 = directional_fourth_with_reference(&system, &params, &opts, cutoff, &reference, &v)?;

// The full mixed-index tensor (packed), or a sub-tensor over chosen DOFs:
let q: SymmetricFourth = fourth_derivative_analytic_dense(&system, &params, &opts, cutoff)?;
let qb = fourth_derivative_analytic_block(&system, &params, &opts, cutoff, &[0, 1, 2, 5])?;
let scalar = q.contract_vvvv(&v)?;
let third  = q.contract_last(&v)?;          // -> SymmetricThird
let matrix = q.contract_last2(&v, &w)?;     // -> Matrix

// Verification reference / smeared-occupation fallback (central FD of the analytic third):
let e4_fd = directional_fourth_seminumerical(&system, &params, &opts, cutoff, &v, 1.0e-3)?;
```

[`directional_fourth_derivative`] sums **five FD-gated 2n+1 stages**, each the exact
`λ`-derivative of one ingredient of the validated analytic third derivative along `R + λv`:
(1) geometric `L_λλλλ` (repulsion + halogen + frozen SCC2 + D3) plus the frozen-SCC2 bilinear
charge path; (2) the frozen-density Hamiltonian blocks; (3) the FC3 density-path derivative
(needs the second-order screened bundle `X^vv`); (4) the FC3 Pulay CN-response derivative (needs
`CN^vv`); (5) the 2n+1 quartic response stage. By the 2n+1 rule the whole assembly needs only the
first- and second-order responses **along `v`** — one charge-space first-order solve and one
second-order solve — never the `O(ndof)` / `O(ndof²)` response sets.

`fourth_derivative_analytic_dense` / `_block` recover the mixed-index tensor **element by element
from that directional quartic** through the polarization identity for symmetric quartic forms,
`Q(x₁,x₂,x₃,x₄) = (1/24) Σ_{∅≠S⊆{1,2,3,4}} (−1)^(4−|S|) e⁗[Σ_{i∈S} x_i]`, so they inherit its
correctness by construction. The 15 subset directions are deduplicated across the whole build
(714 distinct directions for water rather than 495·15 = 7425) and evaluated in parallel against
one shared [`QuarticReference`].

`SymmetricFourth` stores the packed fully-symmetric tensor (300 DOF: ≈2.8 GB dense vs ≈120 MB
packed) with `zeros`, `n`, `len`, `is_empty`, `index`, `add`, `get`, `scale`, `add_from`,
`contract_last`, `contract_last2`, `contract_vvvv`, `block`.

**Errors.** Both guards of the analytic quartic live in `QuarticReference::build`, so every entry
point that goes through a reference inherits them:

- `terms::require_order(..., 4, ...)` — stock GFN1 passes; any experimental model flag
  (`multipole`, `lr_exchange`, `plus_u`, `spin_polarization`, `experimental_d4`, an external
  electric field) fails, since all of those are capped at analytic order 1.
- Fractional (Fermi-smeared) occupations are rejected — use the finite-T quartic below, or
  `directional_fourth_seminumerical`, whose differenced reference is the occupation-agnostic
  finite-T cubic and which therefore serves smeared systems too.

The directional entry points additionally reject a direction whose length is not `3N`;
`directional_fourth_seminumerical` rejects a non-finite or non-positive `step`; and
`fourth_derivative_analytic_block` rejects a DOF index `>= 3N` (it takes **DOF** indices, not atom
indices).

**No system-size cap on the analytic quartic.** `MAX_FOURTH_DERIVATIVE_NDOF = 30` is checked in
exactly two Rust functions — [`dispersion_fourth_derivative`] and
[`crate::halogen::halogen_fourth_derivative`], the full-tensor `Jet4` builders — plus the
`n⁴`-expanded Python front ends (`fourth_derivative` / `fourth_derivative_block`, where it bounds
`3N` and `|dofs|` respectively). Neither `directional_fourth_derivative` nor
`fourth_derivative_analytic_dense` / `_block` checks it: their D3 and halogen legs go through the
*directional* entry points, which carry a univariate `Jet1` instead of a full-space `Jet4`. A
36-DOF halogen-bonded bromoethanol·H₂O complex — above the cap, with D3, halogen, CN-Hamiltonian
and third-order onsite channels all live — is gated in `tests/fourth_derivative_nocap.rs`, which
also pins that the two full-tensor builders *do* still refuse it. See
[limitations.md](limitations.md#higher-derivatives).

The individual stage functions are public for isolated FD gating, under
`gfn1_rs::fourth_derivative::directional::{directional_fourth_geometric_with,
directional_fourth_frozen_density, directional_fourth_hessian_path_stage,
directional_fourth_cn_response_stage}` and
`gfn1_rs::fourth_derivative::response_stage::{directional_response_third,
directional_response_fourth}`.

There is **no CLI flag** for FC4. Python *does* bind it: `fourth_derivative_directional`,
`fourth_derivative_directional_seminumerical`, `fourth_derivative` (dense, `n⁴`-expanded) and
`fourth_derivative_block`.

### Quartic with Fermi smearing (v0.5.0)

The finite-temperature quartic lives beside the finite-T cubic, in
`gfn1_rs::third_derivative::finite_t` (**not** in `fourth_derivative`, and **not** re-exported at
the crate root), because it is the product rule over the finite-T cubic's composition,
`e⁴[v] = D_v[e³[v]]`, with the third-order charge-space response `X^{vvv}` supplying the top leg:

```rust
use gfn1_rs::third_derivative::finite_t::{
    directional_fourth_finite_t, fourth_derivative_finite_t_block, fourth_derivative_finite_t_dense,
};

let e4 = directional_fourth_finite_t(&system, &params, &opts, cutoff, &v)?;          // f64
let q  = fourth_derivative_finite_t_dense(&system, &params, &opts, cutoff)?;         // SymmetricFourth
let qb = fourth_derivative_finite_t_block(&system, &params, &opts, cutoff, &[0, 3])?; // by POSITION in dofs
```

Dense and block use the same quartic polarization identity as
`fourth_derivative_analytic_dense`, over ~`C(n+3,4)` deduplicated directions evaluated in parallel
against one shared [`FiniteTThirdReference`]. Gates: `T = 0` equality vs the validated five-stage
quartic on non-eq water and skew HF (`2.9e-15` / `1.5e-16`); Fermi-smeared scalene H₃ at 3000 K
against the Richardson-extrapolated FD of `directional_third_finite_t` (`6.6e-12`, `h²` ladder
ratio 3.99); dense `T = 0` element-wise equality vs `fourth_derivative_analytic_dense` (`7.5e-14`).

**Errors.** `terms::require_order(..., 4, ...)` as above; a direction whose length is not `3N`; a
block DOF `>= 3N`. Fractional occupations are supported — that is the point — but **exactly
degenerate *and* fractionally occupied orbital pairs are rejected here**, at *third* order:

> `third-order finite-temperature response: orbitals {p} and {q} are exactly degenerate (gap …)
> with fractional occupation — the third-order assembly is still frame-based and the frame is not
> defined inside such a block (the second order takes the frame-free resolvent form instead);
> break the symmetry or use a seminumerical path`

A *near*-degenerate reference (gap ≥ `1e-10`) is fully supported. This is the one exact-degeneracy
rejection left in the stack: the second order was closed by the Daleckii–Krein resolvent form, so
the finite-T **cubic** no longer refuses these references. No CLI flag and no Python binding.

## Charge-space response solver

`gfn1_rs::response::charge_space` (v0.5.0) is the response engine behind the finite-temperature
Hessian, the finite-T FC3 and the whole FC4 assembly. The SCC couples the density only through
the `nsh` Mulliken shell charges, so every response solves one `nsh × nsh` system
`(I − χ⁰K) δq = δq_bare` whose dielectric matrix is LU-factored **once** and reused for every
right-hand side.

```rust
use gfn1_rs::response::charge_space::{ChargeSpaceContext, directional_third_order_inputs};

let ctx = ChargeSpaceContext::build(&system, &params, &electronic)?;
let x = ctx.first_order_field(fock_skeleton_x, overlap_deriv_x)?;   // FirstOrderField
let y = ctx.first_order_field(fock_skeleton_y, overlap_deriv_y)?;
let xy = ctx.solve_second_order(&x, &y, &fock_xy, &overlap_xy, &dgamma_y_qx, &dgamma_x_qy)?;

// Third order, directional only (x = y = z = v), against the same factored dielectric:
let xyf = ctx.second_order_field(&x, &x, &fock_vv, &overlap_vv, &dgamma_v_qv, &dgamma_v_qv)?;
let inp = directional_third_order_inputs(&system, &params, &electronic, cutoff, &x, &xyf, &v)?;
let vvv = ctx.solve_third_order_directional(
    &x, &xyf, &inp.fock_skeleton_vvv, &inp.overlap_vvv, &inp.overlap_vv,
    &inp.v_pot_geo, &inp.dgamma_v_qv, &inp.dgamma_v_qvv, &inp.d2gamma_vv_qv,
)?;                                                                // ThirdOrderBundle
```

`solve_first_order` returns a `FirstOrderBundle` (`density`, `energy_weighted`, `shell_charges`,
`occupation_response`, `screened_potential`); `first_order_field` wraps it with the cached MO-side
representations the second-order solve needs. `solve_second_order` / `second_order_field` produce
the `SecondOrderBundle` / `SecondOrderField`, and `solve_third_order_directional` the
`ThirdOrderBundle` (its caller-side inputs are bundled in `ThirdOrderInputs`, built by
`directional_third_order_inputs`). `ctx.kernel` (`K = γ + ∂²E_onsite/∂q²`) and `ctx.chi0` are
public fields; `ctx.nshell()` and `ctx.is_finite_temperature()` are the accessors.

Finite temperature is native — see [finite-temperature.md](finite-temperature.md), which also
documents the silently-unconverged fixed point this replaced.

**Errors.**

- `build` rejects a periodic system ("the charge-space response solver is non-PBC only") and, at
  **integer** occupations, a (near-)degenerate occupied↔virtual pair — the response is singular
  there and the message suggests enabling Fermi smearing.
- The **second** order handles exactly degenerate *fractionally occupied* blocks: `build` detects
  them and switches that solve to the frame-free **Daleckii–Krein resolvent** form
  (`P = (2πi)⁻¹ ∮ f(z)(zS − H)⁻¹ dz`, each MO element a weighted sum of divided differences of `f`
  against the reference spectrum), where degeneracy is just the confluent limit — no frame
  rotation `U`, no gauge choice, no in-block case split. Gated element-wise against the validated
  coefficient/rotation algebra on a non-degenerate reference, and against a reconverged finite
  difference on exactly degenerate T_d Ni(CO)₄.
- The **third** order is still frame-based and therefore **rejects** an exactly degenerate
  fractionally occupied reference (see the finite-T FC4 section above for the message). Near
  degeneracy (gap ≥ `1e-10`) is supported.

The degeneracy check itself only fires at finite temperature and only for orbitals carrying a
non-negligible Fermi weight, so a gapped `T = 0` reference never trips it. The periodic Γ and
k-point analytic FC3 paths reuse this same solver through a crate-internal constructor that feeds
it the periodic MOs, so their response algebra is literally this code.

The historical MO-pair-space CPXTB solver moved to `gfn1_rs::response::cpxtb`;
`gfn1_rs::cphf` is kept as a re-export shim, so `crate::cphf::AoDerivativeOptions` and
`crate::response::cpxtb::AoDerivativeOptions` are the same type.

## DFT+U and DFT+U+V (non-empirical)

An on-site Hubbard **+U** (Dudarev fully-localised-limit) correction on the correlated
transition-metal **d** subspace, plus an optional inter-site **+V** (metal–ligand hybridisation)
term, layered on the spGFN1 spin machinery (non-periodic; requires the spin path, so it acts on the
open-shell density and is byte-identical to GFN1 for closed shells with `U=0`). It is off by default
and driven entirely through [`ElectronicOptions`] (see [`crate::plus_u`], [`crate::plus_u_dudr`],
[`crate::spin`]):

- `plus_u` — enable +U on the correlated d shell.
- `hubbard_u: Vec<(u8, f64)>` — **fixed** on-site `U` (Hartree) per element, e.g. `(28, 0.12)` for Ni.
- `plus_u_v` + `hubbard_v: Vec<(u8, u8, f64)>` — the inter-site `+V` per element pair, e.g.
  `(28, 7, 0.04)` for Ni–N; neighbours within `hubbard_v_cutoff` (bohr, default 10).
- `hubbard_u_linear_response` — compute `U` (**and `V`**) **NON-EMPIRICALLY** by the Cococcioni–de
  Gironcoli linear response `K = χ₀⁻¹ − χ⁻¹` (bare minus screened susceptibility, via the SCC-CPHF
  charge response) — **no fitted parameters**. Auto-selects the TM d shells; overrides `hubbard_u`.
- `plus_u_all_d` — with linear response, apply +U to **every** atom with a d shell (incl. main-group
  d polarisation, e.g. S/P/Cl), not just transition metals.

**Analytic forces.** The non-empirical `+U(+V)` gradient is fully analytic — including the
geometry response `dU/dR` of the self-consistently-determined `U` (the SCC-CPHF susceptibility
derivative), solved once per structure by a direct linear solve and reused across all `3N`
coordinates ([`crate::plus_u_dudr`]); FD-verified. So `--linear-response-u` geometry optimizations
relax on a consistent PES.

```rust
let mut eo = ElectronicOptions { spin_polarization: true, ..Default::default() };
eo.plus_u = true;
eo.hubbard_u_linear_response = true;   // non-empirical U (+V); no fitted parameters
eo.plus_u_v = true;                    // add inter-site +V (metal–ligand)
// eo.plus_u_all_d = true;             // +U on all d-shell atoms, not just TM
let result = run_electronic(&system, &params, eo)?;
```

CLI: `--plus-u` / `--hubbard-u Fe:0.15` / `--plus-u-v --hubbard-v Fe:N:0.04` / `--linear-response-u`
/ `--plus-u-all-d` (see `gfn1_rs_cli --help`). Python: the same names as `Gfn1NativeCalculator` /
`GFN1RSCalculator` kwargs (`plus_u`, `hubbard_u=[(28, 0.12)]`, `plus_u_v`, `hubbard_v`,
`hubbard_u_linear_response`, `plus_u_all_d`).

## Spectroscopy: dipole derivatives, polarizability, IR, Raman

Non-periodic response properties live in `properties` and reuse the CPXTB machinery:

- [`dipole_derivatives`] — **analytic** Cartesian `dmu/dR` from the CPXTB nuclear
  charge response (`DipoleDerivatives::ddipole_dr`, the raw IR tensor).
- [`static_polarizability`] — **analytic** `alpha = dmu/dE` from the CPXTB field
  response ([`crate::cphf::solve_field_response`]); requires gapped occupations.
  [`static_polarizability_finite_field`] is the numerical fallback.
- [`polarizability_derivatives`] — `alpha` plus the raw `dalpha/dR`
  (`PolarizabilityDerivatives::dpolarizability_dr`).
- [`ir_spectrum`] / [`raman_spectrum`] — wavenumbers with IR intensities (km/mol
  and a.u.) / Raman activities; each result also carries the raw derivative
  tensors and the per-mode `dmu/dQ` / `dalpha/dQ`.

```rust
use gfn1_rs::{ir_spectrum, raman_spectrum, static_polarizability,
              AnalyticHessianOptions, run_electronic, ElectronicOptions};
use gfn1_rs::math::Vec3;

let electronic = run_electronic(&system, &params, ElectronicOptions::default())?;
let alpha = static_polarizability(&system, &params, &electronic)?;     // analytic
let ir = ir_spectrum(&system, &params, AnalyticHessianOptions::default(), Vec3::zero())?;
let raman = raman_spectrum(&system, &params, AnalyticHessianOptions::default(), Vec3::zero(), 1.0e-3)?;
```

The dipole/polarizability use the GFN1 point-charge (monopole) electrostatics, so
intra-atomic polarization is absent; treat absolute intensities as qualitative.

## Parameters: round-trip and finite-difference derivatives

[`Gfn1Parameters::to_param_string`] / [`Gfn1Parameters::write_param_file`] emit the
`param_gfn1-xtb.txt` format deterministically with value-exact round-trip
(`from_str(&p.to_param_string())` reproduces the parsed values). Individual scalars
are addressed with [`ParameterTarget`] (`glob:<key>`, `elem:<Z>:<KEY>[:idx]`,
`pair:<ZA>:<ZB>`); [`Gfn1Parameters::parameter_value`] /
[`Gfn1Parameters::with_parameter`] read/set them consistently (the element
derivation is rebuilt). [`parameter_finite_difference`] takes central differences
of the energy (and optionally forces) per target, and
[`active_targets_for_system`] expands a default target set for a structure.

```rust
use gfn1_rs::{parameter_finite_difference, ParamDerivativeOptions, ParameterTarget};

let targets = vec![ParameterTarget::parse("glob:ks")?, ParameterTarget::parse("elem:1:GAM")?];
let derivs = parameter_finite_difference(&system, &params, &targets,
                                         &ParamDerivativeOptions::default())?;
for d in &derivs { println!("{}\t{:.10e}", d.target.label(), d.energy_derivative); }
```

`ParamDerivativeOptions` also has `include_forces` (`dF/dp`) and `include_stress`
(`dsigma/dp`, periodic only). [`select_target_chunk`]`(targets, i, n)` takes the `Vec` by value and
restricts the work / output to the `i`-th of `n` contiguous target slices — `i` is **1-based**, and
`i = 0` or `i > n` is an error (the CLI exposes this as `--target-chunk i/n`, with
`--param-forces` / `--param-stress`). [`active_targets_for_system`]`(&params, &system)` (note the
argument order) expands the default set for a structure.

Two further central-difference drivers are exported at the crate root. They do **not** take a
`ParamDerivativeOptions` — each takes the options object of the quantity it differentiates plus an
explicit `step`, and both reject a non-finite or non-positive step:

```rust
parameter_dipole_derivatives(&system, &params, &targets, &electronic_options, step)?;
//   -> Vec<(ParameterTarget, [f64; 3])>   dmu/dp in e*a0 per unit parameter
parameter_hessian_derivatives(&system, &params, &targets, &hessian_options, step)?;
//   -> Vec<(ParameterTarget, Matrix)>     dH/dp, a 3N x 3N matrix per target
```

## TD-GFN1 (TDA excited states)

[`solve_tda`] computes closed-shell TD-GFN1 excited states in the Tamm-Dancoff
approximation (Niehaus transition-charge TD-DFTB) for a non-periodic converged
[`ElectronicResult`]. The TDA matrix is built from the GFN1 transition shell
charges and the SCC response kernel; `TdaSpin::Singlet` uses the `2 q K q`
coupling, `TdaSpin::Triplet` reduces to bare orbital-energy gaps. Oscillator
strengths use the Mulliken (monopole) transition dipole.

```rust
use gfn1_rs::{run_electronic, solve_tda, ElectronicOptions, TdaOptions, TdaSpin};

let electronic = run_electronic(&system, &params, ElectronicOptions::default())?;
let tda = solve_tda(&system, &params, &electronic,
                    TdaOptions { n_states: 5, spin: TdaSpin::Singlet })?;
for st in &tda.states {
    println!("{:.4} eV  f={:.4}", st.excitation_energy * 27.2114, st.oscillator_strength);
}
```

[`solve_tda_pbc_gamma`] runs the Gamma-point periodic TDA (periodic Bloch MOs +
the Ewald Klopman-Ohno response kernel). [`solve_tda_gradient_method`] returns the
excited-state gradient of a chosen state by the requested [`TdaGradientMethod`]:

- **`SemiNumerical`** (default, [`solve_tda_gradient_seminumerical`]) — the analytic
  ground-state gradient plus a central finite difference of the *frozen-amplitude*
  excitation energy. By amplitude stationarity (`∂ω/∂X = 0` at the eigenstate) the
  frozen finite difference equals the true `dω/dR` to FD precision for a tracked
  adiabatic state, while skipping the per-displacement TDA re-diagonalisation.
  Non-periodic. Since v0.5.0 every displaced solve is **phase-aligned to the
  reference MO gauge** — without that alignment this method silently returned
  garbage (diverging as `1/h`) for any state with non-vanishing transition charge;
  see [td.md](td.md#4-the-mo-phase-gauge-fixed-in-this-revision).
- **`FiniteDifference`** ([`solve_tda_gradient`]) — full central finite difference
  of the re-diagonalised excitation energy with amplitude-overlap root tracking.
  Non-periodic **and** Γ-point periodic. Robust for well-separated roots; the
  tracking is ill-posed inside a near-degenerate pair.
- **`Analytic`** ([`solve_tda_gradient_analytic`]) — the **direct-CPHF**
  excited-state gradient: the excitation-energy derivative is closed by a single
  CPHF solve for the orbital-relaxation response (replacing the earlier
  Lagrangian/Z-vector route). Non-periodic **and** Γ-point periodic (a lattice-bearing
  system is routed to the periodic path automatically). This is the recommended
  route — it is the cheapest, and it never re-diagonalises at a displaced geometry,
  so it is the only method that stays well defined near a crossing.

```rust
use gfn1_rs::{solve_tda_gradient_method, TdaGradientMethod, ElectronicOptions, TdaOptions, TdaSpin};
let g = solve_tda_gradient_method(&system, &params, &ElectronicOptions::default(), 0,
                                  TdaOptions { n_states: 5, spin: TdaSpin::Singlet },
                                  1.0e-3, TdaGradientMethod::SemiNumerical)?;
println!("excited force[0] = {:?}", g.forces[0]);
```

`step` is used by `SemiNumerical` and `FiniteDifference` and **ignored** by `Analytic`.
[`TdaGradientMethod`] implements `Default` (= `SemiNumerical`) and
[`TdaGradientMethod::parse`] accepts `semi_numerical`/`semi`, `finite_difference`/`fd`/`numerical`
and `analytic`/`lagrangian`/`z_vector`, case- and hyphen-insensitively.

**Errors.** `solve_tda` rejects a periodic system (use [`solve_tda_pbc_gamma`]) and a non-positive
occupied–virtual gap; [`solve_tda_pbc_gamma`] / [`solve_tda_kpoint`] reject a non-periodic one, an
unconverged periodic SCC, a non-integer closed-shell band filling, and a mesh with no
positive-gap occupied→virtual transitions; the gradient drivers reject a non-positive `step` and a
`state_index` beyond `n_states`; [`tda_rotatory_strengths`] is non-periodic.

`TdaGradientResult::gradient` is the gradient of the **total** excited-state energy
`E_ground(free) + ω`, so it includes repulsion, D3 dispersion and the halogen
correction; those are state independent and cancel in `dω/dR`, which you get by
subtracting `analytic_gradient(...).gradient`.

[`tda_frozen_excitation_energy`] evaluates the excitation energy with a **fixed**
amplitude vector at the current geometry (`X^T A(R) X`); it reproduces the
variational TDA energy at the reference geometry and is the per-displacement kernel
of the `SemiNumerical` gradient (and the root-tracking-free FD reference). Its last
argument is `reference: Option<&ElectronicResult>` — **pass the converged SCC the
amplitudes came from for any cross-geometry use**; `None` only fixes the MO phase
gauge correctly at the reference geometry itself.

[`solve_tda_kpoint`] runs the off-Gamma **k-point** periodic TDA: it diagonalises
the complex Bloch Fock `F(k)` at every Monkhorst-Pack point and assembles the
optical (`q = 0`) closed-shell TDA over all occupied->virtual band pairs across the
mesh. The transition shell charges are the real optical Mulliken populations
weighted by `sqrt(w_k)`, so the matrix is real symmetric and at a single Gamma
point it reduces to [`solve_tda_pbc_gamma`] (verified). Integer (gapped) band
occupations are required.

```rust
use gfn1_rs::{solve_tda_kpoint, KMesh, ElectronicOptions, TdaOptions, TdaSpin};
let exc = solve_tda_kpoint(&system, &params, &ElectronicOptions::default(),
    KMesh::monkhorst_pack([2, 2, 2]),
    TdaOptions { n_states: 5, spin: TdaSpin::Singlet })?;
```

[`solve_tda_kpoint_gradient`] / [`solve_tda_kpoint_gradient_analytic`] give the
**periodic** excited-state gradient — analytic at the Γ-point **and across a
k-mesh**, made gauge-invariant by working in the natural CPHF gauge (the max-AO
phase fixing that pins the eigenvectors is never differentiated). FD-verified to
~`4e-7` Hartree/bohr.

The full working equations, the term-by-term breakdown of the analytic gradient,
every gate number, and the limitations (spin, integer occupations only, root
tracking near degeneracies) are collected in **[td.md](td.md)**.

## External magnetic field (GFN1-xTB-M0 / M1)

The `magnetic` module implements the closed-shell GFN1-xTB-M method of Cheng &
Wibowo-Teale (*J. Chem. Theory Comput.* **19**, 6226 (2023)) in London atomic
orbitals (LAOs), non-periodic. Building blocks:

- [`crate::magnetic::lao_overlap_matrix`] — the exact complex LAO overlap `S(B)`
  via the complex Gaussian product theorem (complex product centre `Pbar`).
- [`crate::magnetic::lao_kinetic_matrix`] — the complex LAO kinetic integral
  `<omega|1/2 pi^2|omega>` (`pi = p + A`); the field-dependent kinetic-energy
  correction is `<omega|1/2 pi^2|omega> - e^{i f} <phi|1/2 p^2|phi>`.
- [`crate::magnetic::magnetic_h0_overlap`] — assemble `(H0(B), S(B))` for a field;
  the entry point for taking field/nuclear derivatives.
- [`crate::magnetic::london_phase_angle`] / `spin_zeeman_blocks` — the London phase
  and the (closed-shell-cancelling) spin-Zeeman blocks.

[`crate::magnetic::run_magnetic_scc`] (M0, single basis) and
[`crate::magnetic::run_magnetic_scc_m1`] (M1, node-correct dual basis for the
kinetic-energy correction; see [`crate::secondary_basis`]) solve the complex
Hermitian SCC and return the total energy. Both reduce exactly to the field-free
GFN1 energy at `B = 0` and are gauge-origin invariant (tested). `K^KE = K^SZ = 1`;
for a closed shell the spin-Zeeman term cancels, so the only new physics over
field-free GFN1 is the kinetic-energy correction (M0 over the primary basis, M1
over the secondary basis).

```rust
use gfn1_rs::{run_magnetic_scc, run_magnetic_scc_m1, parse_secondary_basis,
              ElectronicOptions, ExternalFieldOptions};
use gfn1_rs::math::Vec3;
let mut options = ElectronicOptions::default();
options.external_field = ExternalFieldOptions { magnetic_field: Some(Vec3::new(0.0, 0.0, 0.05)), ..Default::default() };
let e_m0 = run_magnetic_scc(&system, &params, &options)?.energy;
let secondary = parse_secondary_basis(&std::fs::read_to_string("GFN1-xTB-cc-pVDZ.txt")?)?;
let e_m1 = run_magnetic_scc_m1(&system, &params, &options, &secondary)?.energy;
```

[`crate::magnetic::magnetizability_isotropic`] returns the isotropic
magnetizability `xi_iso = -1/3 Tr d^2E/dB^2` (eq 26) by central finite field; in
atomic units, scale by [`crate::magnetic::MAGNETIZABILITY_AU_TO_SI`] for
`10^-30 J/T^2`. Pass `Some(&secondary)` for M1 (recommended — M0 is unreliable for
lone-pair / heavier-element systems), `None` for M0.

[`crate::magnetic::magnetizability_isotropic_analytic`],
[`crate::magnetic::magnetizability_diagonal_analytic`] and
[`crate::magnetic::magnetizability_tensor_analytic`] return the **analytic** McWeeny
density-matrix CP-SCC magnetizability (one SCC plus the analytic first-order
response — diamagnetic, paramagnetic, and per-AO charge-overlap terms — instead of
the `6+1` finite-field SCCs). Only the LAO `H0(B)` / `S(B)` *integrals* are differenced, never the
SCF; the isotropic variant is the trace of the diagonal one and matches the finite-field
[`crate::magnetic::magnetizability_isotropic`] to `<0.1%`.

Two v0.5.0 changes make these **frame independent**, and both change what `step` means:

- The molecule is rigidly **recentred on its atom centroid** before the field derivatives are
  taken (the field/gauge origin travels with it). That is an exact gauge transformation for the
  London basis, but the *finite differences* are not invariant under it — their effective
  expansion parameter grows with the molecule's distance from the coordinate origin. Measured on
  non-eq water at `step = 4e-3`, the tensor moved by rel `6.3e-6` under a 2-bohr rigid shift and
  `1.2e-3` under a 9.4-bohr one; after recentring the residual is rel `1.8e-12` / `7.5e-12`, i.e.
  SCC noise. Periodic inputs are returned untouched (the LAO path is molecular).
- Each integral derivative is **Richardson extrapolated** over `(step, step/2)` as
  `(4·D(h/2) − D(h))/3`, so `step` is the **coarse** node of a pair and the truncation is
  `O(step⁴)`, not `O(step²)`. Useful steps are `4e-3 … 1.6e-2`; the extrapolation is at its best
  around `4e-3`–`8e-3` and degrades below `2e-3`, where `1/h²` amplification of builder rounding
  takes over. Both effects are gated to rel `1e-9` in `tests/magnetizability_frame_invariance.rs`.

The finite-field [`crate::magnetic::magnetizability_isotropic`] is **not** recentred and takes a
plain central-difference `step`.

The magneto-optical raw differential tensors:
[`crate::td::tda_rotatory_strengths`] (ECD `R_n = Im(mu_0n . m_n0)` with the raw
magnetic transition dipoles + [`crate::magnetic::angular_momentum_matrix`]),
[`crate::td::tda_optical_rotation`] (Rosenfeld `beta(omega)`),
[`crate::magnetic::magnetic_polarizability`] (`alpha(B)`),
[`crate::magnetic::cotton_mouton_tensor`] (`d^2 alpha/dB^2`),
[`crate::magnetic::mcd_tensor`] (`d alpha/dB`, Faraday/MCD), and
[`crate::magnetic::lao_dipole_matrix`] (raw LAO electric-dipole integrals). In the
GFN1 point-charge electric model the static optical-rotation `G`-tensor and the MCD
vanish by time reversal (`dq/dB = 0`); the length-gauge `lao_dipole_matrix` is the
route to their orbital-current versions. The Cotton-Mouton tensor is nonzero.

[`crate::magnetic::magnetic_analytic_gradient`] returns the **analytic** nuclear
gradient by the Hellmann-Feynman contraction `Re Tr(P dH0(B)/dR) - Re Tr(W dS(B)/dR)
+ shift-Pulay + 1/2 q dgamma/dR q + classical` (one SCC plus cheap integral-builder
derivatives, M0 and M1). At `B = 0` it reproduces the field-free analytic gradient,
and at finite `B` it matches the `6N+1`-SCC finite-difference
[`crate::magnetic::magnetic_gradient`] (both tested).

[`crate::magnetic::nmr_shielding_tensor`] returns the NMR nuclear magnetic shielding
tensor `sigma_{A,ab} = d^2E/dB_a dm_{A,b}` of one nucleus (closed-shell, non-periodic;
[`crate::magnetic::NmrShielding`] with `.isotropic()`, dimensionless atomic units —
`x1e6` for ppm). The diamagnetic part is the ground-state expectation of the operator
`[(r_O.r_A) delta_ab - r_{A,a} r_{O,b}] / r_A^3`; the paramagnetic part contracts the
CP-SCC orbital-Zeeman density response `dP/dB_a` with the nuclear operator
`(r_A x grad)_b / r_A^3`. The magnetic-dipole `1/r^3` integrals are built from scratch
in [`crate::nmr`] (Boys + McMurchie-Davidson), and the assembly is FD-validated against
`d^2E/dB dm` of the operator-injected magnetic SCC. The common gauge origin is the
shielded nucleus; the prefactor is `alpha^2/2 = 1/(2 c^2)`. The GFN1 valence-only basis
omits core electrons, so absolute shieldings track within-method trends rather than
all-electron references.

```rust
use gfn1_rs::{magnetic_analytic_gradient, magnetizability_isotropic, MAGNETIZABILITY_AU_TO_SI};
let g = magnetic_analytic_gradient(&system, &params, &options, Some(&secondary), 1.0e-3)?;
let xi = magnetizability_isotropic(&system, &params, &options, Some(&secondary), 0.02)? * MAGNETIZABILITY_AU_TO_SI;
```

Still to do: open-shell spin-Zeeman / NMR shieldings and the length-gauge
(orbital-current) MCD / optical-rotation `G`-tensor. The ordinary `run_electronic`
errors if a magnetic field is set — the magnetic path is the dedicated
`run_magnetic_scc[_m1]`.

## Multipole electrostatics (experimental mDFTB2)

A parameter-free **atomic dipole + quadrupole** correction on top of the GFN1
monopole electrostatics (Vuong, Aradi, Niklasson, Cui, Irle, *JCTC* **2023**, 19,
7592), **non-PBC only**, providing a self-consistent **energy and analytic gradient**.
It is off by default (the result is then byte-for-byte GFN1); enable it with the
`multipole` flag on [`ElectronicOptions`]:

```rust
use gfn1_rs::{analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions};

let mut options = ElectronicOptions::default();
options.multipole = true;                       // mDFTB2 dipole+quadrupole correction
let scf = run_electronic(&system, &params, options.clone())?;   // energy

let mut grad = AnalyticGradientOptions::default();
grad.electronic = options;                      // threads the flag into the gradient
let forces = analytic_gradient(&system, &params, grad)?;        // analytic forces
```

The multipole interaction tensors are spatial derivatives of the *same* Klopman–Ohno
`γ` profile used for the monopole term (no new fitted parameters); the atomic moments
come from the on-site AO dipole/quadrupole integrals. The correction is variational —
the gradient matches a finite difference of the energy. When it is on, the SCC mixes
the atomic dipole/quadrupole moments **jointly with the shell charges** in the Broyden
vector (the tblite-style multipole SCF) so the self-consistency converges on hard,
polarizable systems (e.g. heavy-element catalyst complexes). The general
`scripts/optimize.py` optimizes any XYZ with `--multipole` (and `--compare` for on-vs-off).
The correction is *not* wired into the PBC, Hessian, or property routines.

Four further **experimental, parameter-free, off-by-default** electrostatics knobs build
on this (all non-PBC, self-consistent energy + analytic gradient, each FD-gated):

```rust
options.multipole = true;
options.multipole_octupole = true;             // add the atomic rank-3 octupole (needs d fns)
options.field_multipole = true;                // with a field on: couple the atomic dipoles
options.multipole_third_order = true;          // on-site charge·dipole² / charge·quad² terms
options.multipole_secondary_basis = Some(sec); // evaluate the moments over a cc-pVnZ basis
options.charge_order = 4;                       // on-site charge expansion to order n (≥3)
options.multipole_order = 4;                    // arbitrary-rank multipole path (n≥4); ≤3 ≡ legacy
options.multipole_charge_order = vec![6, 4];    // per-rank multipole×charge cross terms (dipole→6, quad→4)
```

- `multipole_octupole` extends the mDFTB2 ladder to the traceless atomic octupole (only
  nonzero for atoms with d functions); the octupole joins the same joint-Broyden moment
  vector.
- `field_multipole` (requires `multipole` **and** an external electric field) adds the
  first-order field–dipole coupling `E_field += -E·Σ_A d_A`, makes the reported dipole the
  physically complete `Σ_A q_A R_A + Σ_A d_A` (so `dipole = -∂E/∂E_field`), and enriches the
  field response.
- `multipole_third_order` adds the on-site charge·dipole² / charge·quad² cross terms
  `E³ = Σ_A [α_A Δq_A(d_A·d_A) + β_A Δq_A(Q_A:Q_A)]`, with `α,β` fixed by the hardness
  charge-derivative (the angular generalization of the monopole `(1/3)ΓΔq³`).
- `multipole_secondary_basis = Some(sec)` evaluates the on-site dipole/quadrupole moment
  *integrals* over a node-correct secondary `SecondaryBasis` (e.g. `parse_secondary_basis` or
  `builtin_secondary("cc-pVTZ")`) instead of the minimal primary basis (the Mulliken
  population stays primary) — better-resolved moments for every moment-based term above.
- `charge_order = n` adds the on-site charge terms `E_k = Σ_A (1/k) X_k Δq_A^k` for
  `4 ≤ k ≤ n`, with the deterministic Linear Breathing-Radius coefficients
  `X_k = (γ_A/(k−1))(2Γ_A/γ_A)^(k−2)` (default `3` ≡ stock GFN1).
- `multipole_order = n` generalises the whole moment ladder to **arbitrary rank**. For
  `n ≥ 4` a single parameter-free generic path (`crate::multipole::multipole_fock_generic`)
  self-consistently mixes the atomic moments of ranks `1..=n`, superseding the
  dipole/quadrupole/octupole blocks; `n ≤ 3` keeps the speed-optimised legacy paths
  **byte-for-byte**. The interaction tensors `f^(la,lb) = ∇^(la+lb)γ` are fully symmetric, so
  only their `(r+1)(r+2)/2` unique Cartesian components are formed (not `3^(la+lb)`), and a
  per-element on-site moment cache plus active-moment screening keep it tractable; energy +
  analytic gradient are FD-verified at `n = 4`. High rank is experimental (ill-conditioned at
  short range); requires `multipole`.
- `multipole_charge_order = vec![o1, o2, …]` adds per-rank **multipole×charge cross terms**: entry
  `l−1` is the highest on-site charge order coupled to the rank-`l` (`2^l`-pole) atomic multipole,
  via the breathing-radius Taylor expansion of `½ g_l(η_A(q))(m_l·m_l)`. Empty (default) ⇒ none.
  Each entry must satisfy `order ≤ 2l+3` (the rank-`l` self-energy terminates there) — an
  out-of-range order is a **hard error** from `run_electronic`, never silently truncated. Requires
  `multipole` and `multipole_order ≥` the highest rank carrying a cross term (it forces the generic
  path on). Generalises `multipole_third_order` (the `l ∈ {1,2}`, order-3 special case it reproduces
  exactly; see `crate::multipole::multipole_charge_cross_fields`). Parameter-free; self-consistent
  energy + analytic gradient, FD-gated.

## Range-separated Fock exchange (experimental MFX / OFX)

A parameter-free **long-range exact (Fock) exchange** correction on top of the GFN1
mean field, in the spirit of LC-DFTB (Niehaus & Della Sala, *PSSB* **249**, 237 (2012);
Lutsker, Aradi & Niehaus, *JCP* **143**, 184107 (2015)). **Non-PBC**, **off by default**
(then byte-for-byte GFN1), self-consistent **energy + analytic gradient**, every term
finite-difference-gated. Lives in `crate::exchange` (kernels) and `crate::coulomb` (the
range-separated `γ^lr` and the `ω` schemes).

```rust
use gfn1_rs::{analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions};

let mut options = ElectronicOptions::default();
options.lr_exchange = true;      // MFX: long-range Mulliken exact exchange
options.onsite_exchange = true;  // OFX: exact one-center exchange on top of MFX (optional)
let scf = run_electronic(&system, &params, options.clone())?;

let mut grad = AnalyticGradientOptions::default();
grad.electronic = options;       // threads the flags into the gradient
let forces = analytic_gradient(&system, &params, grad)?;
```

- **`lr_exchange` (MFX).** The Mulliken long-range exchange energy
  `E_x = ½Tr[ΔP·K[ΔP]]` over the density fluctuation `ΔP = P − P0`
  ([`crate::exchange::neutral_atom_reference_density`]), with
  `K[ΔP]_{μν} = −⅛ Σ_{σλ} ΔP_{σλ} S_{μσ}S_{νλ}(γ^lr_{μν}+γ^lr_{μλ}+γ^lr_{σν}+γ^lr_{σλ})`
  ([`crate::exchange::mfx_kernel`]; [`crate::exchange::mfx_energy_fock`]). The kernel is
  assembled by a symmetric **GEMM factorisation** (`Γ∘(SΔPS)` style products — no AO
  four-index loop) and is self-adjoint, so the Fock contribution is exactly `K[ΔP]`. The
  long-range operator is a hardness-derived Gaussian charge-cloud `γ^lr = erf(R/τ)/R`
  (finite at `R = 0`; [`crate::coulomb::lr_gamma_exchange`]), so the exchange is
  overlap-weighted and **size-consistent** at dissociation.
- **`onsite_exchange` (OFX).** Layered on MFX (requires `lr_exchange`), it upgrades the
  *same-atom* exchange from the Mulliken approximation to the **exact one-center
  two-electron integrals** via the difference kernel
  `K_OFX = K_onsite,refined^lr − K_onsite,Mulliken^lr`
  ([`crate::exchange::onsite_fock_exchange_kernel`]) — no double count, since the Mulliken
  on-site part MFX already applies is subtracted. The one-center ERIs are real STO-nG
  integrals (McMurchie–Davidson) and are **geometry-independent**, so they are built once
  per element and cached ([`crate::exchange::OnsiteExchangeCache`]). With a *static* `ω`, OFX adds
  **no explicit force** (the integrals are one-center / translation-invariant) — its effect flows
  through the relaxed density; with `dynamic_omega` it gains the `∂ω/∂R` term below (the screening
  `ω_AA` moves with the coordination number).
- **`ω` schemes.** The crossover `ω` is parameter-free, built from existing GFN1 quantities
  via [`crate::coulomb::OmegaScheme`]: `HardnessPairwise` (default, `ω_A = η_A`, pairs by
  harmonic mean) and `Fixed(ω)` (reference baseline).
- **`dynamic_omega` (LocalGeometry — geometry-adaptive `ω`).** Sets `ω_A = η_A / s_A` with the
  parameter-free size factor `s_A = (1+CN_A)^(−1/3)` from the GFN1-Hamiltonian coordination number
  ([`crate::coulomb::local_size_factor_from_cn`]); a more-coordinated atom screens at shorter range,
  and `CN = 0` recovers `HardnessPairwise`. The **analytic gradient gains the `∂ω/∂R` reorganisation
  force** for *both* MFX and OFX — `∂E_x/∂R_C = Σ_A (∂E_x/∂CN_A)(∂CN_A/∂R_C)` with
  `∂E_x/∂CN_A = (∂E_x/∂ω_A)(∂ω_A/∂s_A)(∂s_A/∂CN_A)` (including the onsite `γ^lr_AA(R=0)` term and the
  OFX one-center ERIs' `ω`-dependence), all FD-gated. Because the OFX `ω` now changes with geometry,
  the one-center ERIs are factored as `(μκ|νλ)^lr(ω) = Σ_k c_k β^{(2k+1)/2}` (`β = ω²/(ω²+α)`): an
  ω-independent per-element **skeleton** (`OnsiteEriSkeleton`, an internal type in
  `crate::exchange`: `(p,q)`-grouped + even-order, memory-efficient) carries the heavy contraction (built once per process) and is re-evaluated cheaply at
  each step's `ω`, so a dynamic-ω optimisation never rebuilds the d/f-element ERIs.
- **Convergence.** The off-diagonal exchange Fock defeats charge-vector mixing, so the exchange path
  runs a **density-matrix SCF** — a one-way ADIIS → C-DIIS pipeline with a κ-trust-region virtual
  level shift and an ADIIS fallback on a limit-cycling metal, finished by a **TRAH continuation** from
  the near-converged density (below). *Small-gap / metallic systems:* full long-range exact exchange
  is variationally unbounded against GFN1's cubic on-site term, so pair it with **`charge_order = 4`**
  — the convex quartic bounds the density fluctuation and the SCF converges to a physical state (e.g.
  dppf-PdCl₂, a 68-atom Fe/Pd complex, converges and optimises with MFX/OFX + `charge_order = 4`).
- **Performance / scaling.** OFX's one-center ERIs are geometry-independent, so each unique
  element's tensor is built **once per process** (a global `Arc` memo) and reused across all
  atoms, SCC iterations, and geometry steps; the build exploits the **8-fold permutational
  ERI symmetry** (≈8× fewer quartet evaluations) and the per-SCC contraction is `O(N)` (a
  small per-atom `nao⁴`). MFX's kernel is the symmetric **GEMM factorization** (5 matrix
  products via `ΔPS=(SΔP)ᵀ`, `K3=K2ᵀ`), and the commutator uses `SPF=(FPS)ᵀ`. The per-SCC MFX
  cost is `O(N³)` (dense, like the eigensolve); genuinely sub-cubic large-scale MFX (exploiting
  the short-range sparsity of `S`) is a future item.

The correction is wired to the CLI (`--lr-exchange`, `--onsite-exchange`, `--dynamic-omega`) and to
Python/ASE (`lr_exchange`, `onsite_exchange`, `dynamic_omega`). It is *not* wired into the PBC,
Hessian, or magnetic routines.

## Trust-Region Augmented Hessian SCF (experimental)

`crate::trah` is a matrix-free **second-order SCF** for the exchange-augmented SCC:
it minimises the electronic energy directly over orbital rotations `C → C·exp(κ)` (with `κ`
real, antisymmetric, occupied–virtual only) instead of mixing densities — robust where DIIS
on the off-diagonal exchange Fock stalls. The orbital Hessian is never assembled; only
Hessian–vector products are formed from the linear Fock response (the second-order charge
kernel + MFX + OFX), and an augmented-Hessian / trust-region level shift globalises the
Newton step. The module is pure orbital-rotation algebra (gradient
`g = 2Δn·F^MO`, [`crate::trah::orbital_gradient`]; Hessian–vector
[`crate::trah::hessian_vector`]; trust-region step
[`crate::trah::trust_region_newton_step`]; driver [`crate::trah::run_trah_scf`]), validated
against an analytic model functional and the real exchange functional (FD-gated).

In the SCC it is selected for the exchange path automatically as **AutoTRAH** — DIIS runs
first and TRAH takes over only if DIIS fails to converge — or forced directly with
`options.scf_trah = true` (CLI `--scf-trah`, Python/ASE `scf_trah`). Closed-shell / gapped
systems, integer occupations, non-periodic.

```rust
let mut options = ElectronicOptions::default();
options.lr_exchange = true;
options.scf_trah = true;   // force the TRAH second-order driver (else AutoTRAH: DIIS then TRAH)
let scf = run_electronic(&system, &params, options)?;
```

## Notes

- Dense linear algebra uses `faer`; BLAS/LAPACK are not required. `linalg` uses
  zero-copy strided `faer` views, and [`crate::linalg::symmetric_eigen`]`(input)`
  replaces the old `symmetric_eigen_jacobi(input, tol, max_sweeps)` whose
  `tol`/`max_sweeps` were silently ignored (the old name remains as a deprecated
  wrapper).
- The crate builds both an `rlib` and a `cdylib`; the CLI binary is `gfn1_rs_cli`.
- Set `GFN1_PROFILE=1` to emit one line per scoped region to stderr; scopes are
  created with `gfn1_rs::profile::scope(...)`.
- [`crate::terms`] is the single source of truth for each energy term's highest
  implemented analytic derivative order; every derivative driver calls
  [`crate::terms::require_order`] first. Update `max_analytic_order` in exactly
  one place when a new order lands, and update the table in
  [derivatives.md](derivatives.md#0-the-term-registry-require_order).
- GFN2 parameter files are rejected at parse time.
- `constants::HARTREE_TO_EV` is CODATA-2018 and is for **reporting only**; the
  model-side `basis::EV_TO_HARTREE` deliberately stays on the legacy value GFN1
  was parametrized against. `constants::KB_HARTREE_PER_K` is the one Boltzmann
  constant every finite-temperature path uses.
