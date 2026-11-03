# Known limitations

An honest inventory of what does **not** work, what errors explicitly, and — most
importantly — what can still be silently wrong. Kept next to the feature docs on
purpose: a feature is not documented until its gaps are.

Convention used below:

- **Rejected** — the code returns an explicit `Err` with a message. Safe.
- **Silent** — the code returns a number that may be wrong. Read the note.

---

## v0.7.0 scope and further research work

Native sTDA vertical spectra and local isolated-root derivatives do not imply
a globally smooth excited-state PES. Its fixed occupied/virtual orbital model
has a direction-dependent limit at final frontier degeneracy. The ordinary
signed-sheet ethylene searches now find independently audited force-zero
saddles, but did not establish a smooth minimum. A separately defined
uncoupled upper-envelope crossing has converged
KKT stationarity and positive tangent curvature; its individual-sheet force
remains nonzero. These outcomes and complete checks are distinguished in
[stda-pes.md](stda-pes.md).

Version `0.7.0` covers the implemented features in `scope.md`. Local regression
tests do not establish completion of the broader revision-2 research plan:

- The original exporter did not record its BSE version. All 720 inherited
  contractions have now been independently reconstructed against BSE 0.12
  and the GFN1 hydrogen functions, without changing their numerical bodies.
  Several higher-zeta files retain lower-zeta polarization subsets, so they
  are not complete Dunning bases or a certified published-SI reconstruction.
  See `tests/data/secondary-basis-provenance.json` and
  `third_party/upstream-byte-review.json` for separate parameter/source checks.
- SOC requires explicit zeta values and provenance. The current synthetic
  angular-momentum/force/response gates do not certify an adopted literature
  parameter set or the requested heme/metal-complex literature benchmarks.
- M0/M1 remains closed-shell. Solvent total derivatives stop at the analytic
  gradient; solvated Hessian/FC3/FC4 and solvated VPT2 are rejected. Composite
  solvent/SOC/spin/magnetic/ML derivatives are not inferred from separate gates.
- VPT2/GVPT2 is PES-only; Watson/Coriolis terms and anharmonic intensities are
  outside its implementation. The complete requested five-molecule comparison
  has not been established by the H2/water molecular tests.
- A primitive-cell arbitrary-q Sternheimer backend, periodic ML analytic FC3,
  ML FC4 remains
  separate unfinished work.

The pinned official xtb `v6.7.1` binary has two independently investigated
reference limitations. Its tblite frontend's energy was unchanged when
`--etemp` was swept over 0/100/300/600 K; spin comparisons therefore use 300 K,
matching the observed default ensemble without energy offsets. Water-dimer
and zwitterionic-glycine analytic gradients disagree with the same reference's
energy finite differences at transverse zero-overlap pairs. The missing
`H0*dS` contribution predicts that difference. Direct gradient parity remains
a failed comparison; independent reference-energy derivative audits are
reported separately, with two steps for each of ten molecule/solvent cases.
The complete raw comparison counts, fixed binary/archive hashes, source
fingerprints and temperature sweep are retained in
[`tests/data/xtb-validation-report.json`](../tests/data/xtb-validation-report.json).

## Structural ML boundary coverage

The independent radial Hamiltonian supports molecular energies, analytic
forces, static scalar response, and canonical analytic Hessian/FC3, including
fractional thermal occupations and degenerate occupation blocks.
Gamma/k-point **periodic energies, analytic forces, stress and DFPT Hessians**
include periodic descriptor images and the full first/second structural and
electronic response chains. Commensurate finite-q ML phonons use analytic
supercell response. Periodic **analytic** ML FC3 and ML TDA excited-state
gradients remain rejected; a seminumerical derivative of the ML periodic
Hessian is available through the existing fallback. ML FC4,
spin/SOC/magnetic/solvent composite models are also rejected. Model-specific
training domains and held-out/extrapolation measurements are in
[parameters.md](parameters.md#independent-structural-ml-models).

## Finite temperature / Fermi smearing

### Analytic molecular FC3 and FC4 — **smearing supported**

The standard `third_derivative_analytic{,_dense,_vector,_block}` and
`fourth_derivative` entry points select the analytic canonical free-energy
backend at positive electronic temperature. `QuarticReference` and
`CubicReference` retain the same temperature and electron count. Degenerate
fractionally occupied blocks use analytic occupation-matrix response; they do
not require a symmetry-breaking displacement or a finite-difference fallback.
The explicit `third_derivative::finite_t` APIs remain available. Their block
indices are Cartesian DOFs, whereas the standard molecular FC3 block API takes
atom indices.

Memory budgets are checked before building the reference. The thermal
`CubicReference` stores complete orbital and occupation response matrices, so
its workspace is larger than the occupied-space zero-temperature reference.
ML supports thermal FC3; ML FC4 and options capped below the requested
derivative order remain explicit errors. CN-disabled FC4 is also rejected.
Seminumerical routines remain independent verification routes.

### ~~Exactly degenerate + fractionally occupied second order~~ — **resolved**

*This was a rejection until commit `c524656`. It is no longer a limitation; the
entry is kept because the error message it describes was public API.*

The second-order charge-space solver used to refuse when a fractionally occupied
orbital pair was **exactly** degenerate (gap `< 1e-10`). **That guard is gone**,
and the string it raised no longer exists in the source.

**What replaced it.** `ChargeSpaceContext` now builds the second-order response in
the **Daleckii–Krein resolvent form**. Starting from

```text
P = (2πi)⁻¹ ∮ f(z) (zS − H)⁻¹ dz
```

the second derivative is a pure resolvent product,

```text
d²G = G Bˣ G Bʸ G + G Bʸ G Bˣ G − G Bˣʸ G ,   B = z dS − dH
```

and contour integration turns each element of the MO representation into a
weighted sum of **divided differences of `f` against the reference spectrum**
(`f^[1]`, `f^[2]`, and their `z`-lifted partners `g = zf`, `k = z²f`, built by the
Leibniz recursion). `build()` selects this path (`dk_second`) exactly when it
detects a degenerate fractionally occupied reference.

**There is no frame rotation `U`, no gauge choice, and no in-block case split.**
Degeneracy is not special-cased at all: it is the **confluent limit** of the same
smooth expression — pinched (`p ≈ q`, `r` apart) and fully confluent
(`p ≈ r ≈ q`, giving `½f″`). That is the whole reason the problem dissolved.

Measured:

| Gate | Result |
| --- | --- |
| Non-degenerate equivalence vs the validated frame path (`P` / `W` / `q`) | `7.7e-12` / `1.4e-10` / `4.6e-15` |
| **T_d Ni(CO)₄** (exactly degenerate, fractional) FD gate (`P` / `W` / `q`) | `8.2e-8` / `6.4e-8` / `6.2e-8` |

The old pin test — which asserted the guard fired — has been replaced by that
real finite-difference gate. Near-degenerate pairs (gap ≥ `1e-10`) remain
supported at `5.5e-9`, unchanged.

**Historical note, since it explains three failed attempts.** The previous
formulation wrote the in-block occupation response as a matrix
`F^(xy)_B = (f''/2){δΛ^y, δΛ^x} + f'(Λ̇^xy − μ^(xy) I)` and tried to close it with
in-block quantities. It cannot be closed that way: in that algebra the in-block
second order depends on the arbitrary choice of eigenbasis *within* the block, so
every dot convention tried failed the T_d gate (frame-included dots stuck at
`4.5e-2`, fixed-basis dots at `2.2e1`, a third scalar-chain variant at the same
order). The resolvent form has no such freedom to fix. The reasoning survives in
comments in `src/response/charge_space.rs`.

### Exactly degenerate fractional third-order response — resolved in v0.7.0 development

The third-order electronic response now uses a fixed-frame spectral resolvent
with confluent Fermi divided differences. It includes the overlap derivatives,
the canonical chemical-potential response through third order, and the screened
shell-charge response. No orbital eigenvector derivative is assigned inside a
degenerate block. Molecular analytic free-energy FC3 and FC4 therefore support
exactly degenerate fractional occupations.

The Ni(CO)4 regression at 3000 K compares every density, energy-weighted-density
and shell-charge component against displaced second responses. Halving the step
from 0.004 to 0.002 reduces the respective maximum errors from
7.27e-7 / 8.21e-7 / 1.18e-6 to 1.81e-7 / 2.06e-7 / 2.96e-7.
The HF audit at 30000 K checks all 1554 nuclear components through order four
against three-step finite differences; its quartic maximum error falls from
1.06e-5 to 2.65e-6 to 6.62e-7 before extrapolation.

The zero-temperature occupied/virtual gap requirement remains. Periodic
analytic FC4 is not implemented. Only a small-spectrum divided-difference table
is cached (at most about 11 MiB at 48 orbitals); larger spectra calculate weights
on demand instead of allocating an unrestricted fourth-power table.

### Analytic polarizability — *rejected*

> `analytic field response requires gapped (integer) occupations; use a finite-field polarizability for fractional occupations`
>
> — `src/response/cpxtb.rs` (`solve_field_response`)

Use `static_polarizability_finite_field`.

### Near-degenerate gap with integer occupations — *rejected*

At `T = 0` the response operator is genuinely singular:

> `charge-space response is singular: occupied orbital {i} and virtual orbital {a} are (near-)degenerate (gap {…} Ha) with integer occupations; enable Fermi smearing`
>
> — `src/response/charge_space.rs`

The MO-pair CPXTB carries the same guard. Before v0.5.0 this returned ~1e42
garbage without erroring (observed on a symmetry-broken aufbau filling of a
degenerate open shell).

### Periodic finite-temperature response: singular dielectric — *rejected*

**The silent-nonconvergence defect here is fixed.** Both periodic finite-T
response sites — `gamma_finite_temperature_response` and the k-point analogue
`kpoint_finite_temperature_response_dof` in `src/pbc/hessian.rs` — used to close
their SCC self-consistency with a hard-capped `for _ in 0..50` damped
(mixing 0.35) fixed point whose convergence was only used to `break` early: no
convergence flag, no post-loop residual check, no error return. They now take the
same **direct charge-space dielectric solve** the molecular path took in v0.5.0,
`(I − χ⁰K) δq = δq_bare`, with `χ⁰` built once per Hessian call from `nsh`
unit-shell-potential responses and one LU factorization reused by every Cartesian
DOF (`PeriodicChargeDielectric`). The k-point susceptibility is *also* a real
`nsh × nsh` matrix — the perturbing potential and the Brillouin-zone-summed
Mulliken charges are real even though every per-k response operator is complex.

How wrong the old path could be: the damped iteration converges only for `χ⁰K`
eigenvalues inside `(−4.71, 1)`. Alkali cells sit comfortably inside that window
(bcc Li at 30000 K: spectral radius `ρ(M) = 0.6504`, legacy result correct to
`5.8e-18`), which is why the pre-existing bcc-Li gates never caught it.
Transition-metal cells do not:

| fixture | `T` (K) | fractional bands | `ρ(M)` | legacy − exact (shell-charge response) |
| --- | --- | --- | --- | --- |
| bcc Li, `a = 3.51 Å` | 30000 | 4 | 0.650 | `5.8e-18` |
| Li₄ chain, `d = 2.6 Å` | 3000 | 5 | 0.650 | `5.3e-15` |
| Ni₂, `a = 3.52 Å` | 3000 | 10 | 4.55 | **`2.0e22`** |
| Ni₂, `a = 3.52 Å` | 1000 | 10 | 14.1 | **`2.0e43`** |
| Ti₂, `a = 3.30 Å` | 3000 | 10 | 3.13 | **`6.4e12`** |
| Fe₂, `a = 2.87 Å` | 3000 | 11 | 34.8 | **`4.6e65`** |
| Ni + 4 C, 9 Å cell | 30000 | 25 | 1.05 | **`4.9e-1`** |

The direct solve is exact for all of these. What is left is the *conditioning* of
the dielectric itself, and that is now an explicit error rather than a wrong
number — `DenseLu` does not detect a zero pivot, so the residual of every solve is
verified against its own equation:

> `periodic finite-temperature dielectric solve produced a non-finite shell-charge response (the charge-space dielectric I - chi0*K is singular)`
>
> `periodic finite-temperature dielectric solve did not satisfy its own equation: residual {…} > tolerance {…} (the charge-space dielectric I - chi0*K is singular or severely ill-conditioned)`
>
> — `src/pbc/hessian.rs` (`PeriodicChargeDielectric::solve`)

Gates (`tests/pbc_finite_t.rs`, fixture Ni₂ at 3000 K — ten fractional bands and
the `ρ(M) = 4.55` divergence above):

| gate | measured |
| --- | --- |
| Γ shell-charge response vs central FD of the reconverged SCC shell charges | `3.1e-10` |
| Γ finite-T Hessian vs FD of the analytic periodic gradient | `5.3e-11` |
| k-point (`2×2×2`) finite-T Hessian vs FD of the analytic gradient | `6.2e-12` |
| gapped insulator at 300 K vs the `T = 0` path | `0.0` (bit-identical) |

**What this does *not* cover.** Exact degeneracy is not a problem for the periodic
Hessian: it needs only the **first**-order response, whose divided difference has
a well-defined `kT`-slope limit at zero gap. (The **second**-order molecular-stack
gap that used to sit next to this — the Λ-covariant in-block channel — was closed
by the resolvent form; see the resolved section above.) The one genuinely
uncovered case is a singular `I − χ⁰K`, and that is now an explicit rejection
rather than a silent fallback.

### Berry-phase polarization: fractional occupations — *rejected*

Unlike the paths above, `pbc_berry_polarization` **does** enforce it. A metallic
manifold has no well-defined Berry phase, so both the smeared-occupation and the
closed-gap cases error out rather than returning a number:

> `pbc_berry_polarization requires integer band occupations: band {band} at k = ({…}, {…}, {…}) has Fermi occupation {f} (deviation {…} > {…}). Rerun at ElectronicOptions::electronic_temperature = 0.0, or on a gapped insulator`
>
> `pbc_berry_polarization requires integer band occupations: the band gap at k = ({…}, {…}, {…}) is {gap} Hartree, below the {min_gap} threshold. A metallic (fractionally occupied) manifold has no well-defined Berry phase`
>
> — `src/pbc/polarization.rs` (`string_berry_phase`)

Note that `ElectronicOptions::default()` carries `electronic_temperature = 300.0`,
which is fine for a gapped insulator (its Fermi occupations are integer to well
under the `1e-6` default `occupation_tolerance`) and trips immediately on anything
soft. Thresholds are `BerryPolarizationOptions::occupation_tolerance` and
`::min_band_gap`.

Three more `pbc_berry_polarization` guards, all *rejected*:

> `pbc_berry_polarization requires an integer closed-shell band filling (got {…} occupied bands out of {n}); open-shell and fractional fillings have no well-defined Berry phase`
>
> `pbc_berry_polarization requires a neutral cell (net charge {…}); the polarization of a charged periodic cell depends on the coordinate origin`
>
> `pbc_berry_polarization does not support the on-site multipole SCC (the multipole Fock is not carried by PbcSccResult, so the Berry states would not be the SCC states); rerun with ElectronicOptions::multipole = false`

and one numerical guard, which means the Berry mesh is too coarse for the system
(neighbouring occupied manifolds have gone orthogonal):

> `pbc_berry_polarization: a Berry link determinant vanished (the occupied manifolds at neighbouring k-points are orthogonal); refine the Berry mesh`

---

## Higher derivatives

### System-size cap on the FULL-TENSOR FC4 — *rejected*

`MAX_FOURTH_DERIVATIVE_NDOF = 30` (**10 atoms**). A full-space `Jet4` stores
`ndof⁴` doubles and the assembly keeps `O(nat)` of them alive, so the working set
grows as `ndof⁵`.

> `analytic halogen-bond fourth derivative is capped at 30 degrees of freedom (10 atoms); got {ndof} ({nat} atoms). …`
>
> — `src/halogen.rs`; the analogous message for D3 is in `src/dispersion.rs`

**This no longer constrains the directional quartic.** D3 and halogen supply
their geometric fourth derivative through 1-D jets along the requested direction
(`Jet1`: value plus four `t`-derivatives, direction installed per rayon worker),
so `directional_fourth_derivative` scales with `nat`, not with `ndof⁵`;
`tests/fourth_derivative_nocap.rs` gates a 36-DOF halogen-bonded complex
(ratio 4.00 against the seminumerical reference) and a 45-DOF alcohol runs in
about a minute. The cap still applies to

* the **full-tensor** `dispersion_fourth_derivative` / `halogen_fourth_derivative`
  entry points (the `ndof⁵` working set is real there), and
* the **`n⁴`-expanded** Python bindings, where it additionally bounds the
  `8·n⁴`-byte nested-list expansion — applied to `|dofs|` in block mode, so a
  small block of a large molecule is fine.

### `enable_cn_hamiltonian = false` at fourth order — *rejected*

Both strict and finite-temperature analytic FC4 front ends reject this
combination through `terms::require_order`, before SCF:

> `{context}: analytic order-{order} derivatives with enable_cn_hamiltonian=false are not implemented`

The quartic frozen-density stages retain CN-dependent blocks. Disabling CN in
only the other stages would mix different Hamiltonians. Earlier versions
silently returned such a value (a recorded water discrepancy was about `2e-5`).
The default CN-enabled model is supported. Gradient, Hessian and FC3 capability
checks are unaffected. Directional, dense and block entry points in both FC4
families are checked by
`cn_disabled_quartics_are_rejected_by_strict_and_finite_temperature_drivers`.

### The *Γ* analytic FC3 entry points are Γ-only — but an analytic k-point FC3 exists

`pbc_gamma_third_analytic_dense/_block/_vector` reject a Monkhorst–Pack mesh
rather than silently returning the Γ answer:

> `the analytic Gamma-point periodic third derivative requires a Gamma-only k-mesh (got a [2, 2, 2] Monkhorst-Pack grid); use pbc_kpoint_third_derivative_seminumerical_* for k-point sampling`
>
> — `src/pbc/gamma_response.rs`

That message names only the seminumerical alternative and is now incomplete:
since v0.5.0 the closed form also exists **on an arbitrary mesh** as
`pbc_kpoint_third_analytic_dense/_block/_vector` +
`KpointThirdReference` (`docs/pbc.md` §2d, Python
`third_derivative_periodic_kpoint_analytic{,_vector}`). It carries the response
chain through the complex CPXTB per k-point and builds the second-order response
from the complex resolvent (Daleckii–Krein) form, then hands both back to the
Γ assembly as real-space image densities. Driven on `KMesh::gamma()` it
reproduces the Γ path to `~1e-17` relative.

So the remaining limitation is narrow: the *Γ-named* entry points stay Γ-only
(they are the cheaper specialisation, not a subset of the k-point path's
coverage). The seminumerical
`pbc_kpoint_third_derivative_seminumerical_dense/_vector` remains an independent
reference and supports option sets capped below analytic order 3 when their
analytic Hessian is available.

### Analytic periodic FC3 smearing and reference-motion residual — **resolved**

Both Γ (`pbc_gamma_third_analytic_*`) and k-mesh
(`pbc_kpoint_third_analytic_*`) APIs support canonical fractional occupations,
including exact degeneracy. A k-mesh uses one globally conserved electron count
and chemical potential. No finite differences enter those drivers.

The earlier `~1e-7` diamond/BN residual came from missing CN/SCC frozen-cache and
reference-motion chains. After correcting these, the T=0 directional recheck
differs by `9.9e-16` / `5.5e-16` from Richardson-extrapolated Hessian differences.
All 216 components of each H2/HF/B2 finite-temperature fixture are below
`8.5e-11` at Γ and `7.2e-12` on the k-mesh. The independently reconverged FD
checks retain both temperatures and canonical electron count; see [pbc.md](pbc.md).

The CLI rejects the molecular flag on periodic input:

> `third derivative (cubic force constants) supports non-periodic systems only`
>
> — `src/bin/gfn1_rs.rs`

### Nothing implements analytic order 5

`terms::require_order(_, _, 5, _)` fails for stock GFN1 — every core row blocks it.

### Experimental model flags cap the ladder at the gradient — *rejected*

`experimental_d4`, `multipole` (mDFTB2 / CAMM), `lr_exchange` (MFX/OFX), `plus_u`,
`spin_polarization` and an external **electric field** all carry
`max_analytic_order: 1`. Enabling any of them and asking for a Hessian, FC3 or FC4
fails fast:

> `{context} requires analytic order-{order} derivatives, but the active option set includes terms without them: `{term}` (max analytic order {n}). Disable those options or use a lower-order / finite-difference path`
>
> — `src/terms.rs`

Before v0.5.0 the analytic Hessian **silently dropped** these terms and returned
the Hessian of a different energy expression.

### FC4 front ends and remaining boundaries

The CLI exposes `--fourth-derivative` for molecular FC4. Python provides
`fourth_derivative_directional`, `fourth_derivative_directional_seminumerical`,
`fourth_derivative` (dense) and `fourth_derivative_block`, gated by
`tests/python/test_v050_derivatives.py`. The dense and block bindings go through
`check_fourth_dense_size`, so they carry the `n⁴`-expansion cap described above
(applied to `|dofs|` in block mode); the directional binding does not.

The standard Python/ASE and CLI quartic APIs select analytic canonical response
at positive electronic temperature, including exact fractional degeneracy. The
explicit `third_derivative::finite_t` names remain available in Rust. Periodic
FC4, ML FC4 and CN-disabled FC4 remain unavailable and return explicit errors.

The periodic derivative entry points *do* have Python bindings — seminumerical
Γ FC3 (`third_derivative_periodic`, `third_derivative_periodic_vector`), analytic
Γ FC3 (`third_derivative_periodic_analytic`,
`third_derivative_periodic_analytic_vector`), analytic k-point FC3
(`third_derivative_periodic_kpoint_analytic`,
`third_derivative_periodic_kpoint_analytic_vector`), `strain_hessian_derivative`
and `gruneisen` — but no CLI flag. Three gaps remain on the Python side: the
analytic FC3 **block** modes (`pbc_gamma_third_analytic_block`,
`pbc_kpoint_third_analytic_block`), the *seminumerical* k-point FC3
(`pbc_kpoint_third_derivative_seminumerical_dense/_vector`) and Berry-phase
polarization (`pbc_berry_polarization`) are Rust-only.

### Crate-root re-export gaps

The main FC4 drivers (`directional_fourth_derivative`,
`directional_fourth_seminumerical`, `fourth_derivative_analytic_dense/_block`,
`QuarticReference`, `SymmetricFourth`) and the finite-temperature FC3 drivers
(`directional_third_finite_t`, `third_derivative_finite_t_dense/_block`,
`FiniteTThirdReference`) ARE re-exported at the crate root. Still
module-path-only:

- `gfn1_rs::third_derivative::third_derivative_frozen_electronic`
- `gfn1_rs::params::{BUILTIN_GFN1_PARAM_TEXT, BUILTIN_GFN1_SI_PARAM_TEXT}`
- the fine-grained FC4 stage functions (`fourth_derivative::directional`,
  `fourth_derivative::response_stage`)

---

## TD-GFN1 (TDA) excited states

Full treatment in [td.md](td.md); the gaps a user can actually hit:

### `tda_frozen_excitation_energy` without a reference SCC (silent, gauge)

The frozen-amplitude Rayleigh quotient depends on the eigensolver's arbitrary
per-orbital sign, which is not continuous in the geometry. Called at a geometry
other than the one the amplitudes came from and with `reference = None`, it can be
off by a finite step, and a central difference of it then **diverges as `1/h`**
(measured: `14 Hartree/bohr` on the water `S3` root at `h = 1e-3`, against a true
gradient of `~1e-2`). Pass `Some(&reference_scc)` for any cross-geometry use. Dark
states (zero transition charge) are accidentally immune.

Until v0.5.0 the **default** `TdaGradientMethod::SemiNumerical` inherited this
exactly; it now phase-aligns internally.

### Thermal occupations and periodic sampling

Molecular and Γ-point TDA support fractional canonical occupations. The
off-Γ TDA k-mesh backend requires gapped integer bands. The optical
TD-DFPT k-mesh route supports fractional bands at momentum transfer q=0.

### Root tracking near degeneracies

`solve_tda_gradient` follows the requested root by amplitude overlap. Inside a
near-degenerate pair the two amplitude vectors span the same subspace and the
tracking is ill-posed: on the jittered-formaldehyde `S2`/`S3` pair (split
`1.07e-3` Hartree) its disagreement with the analytic gradient is `1.0e-4 -> 1.9e-5`
Hartree/bohr instead of the `~5e-7` it reaches on well-separated roots. Use
`solve_tda_gradient_analytic` there. The **periodic** branch of
`solve_tda_gradient` additionally does not phase-align its displaced solves.

### No magnetic (spin) kernel for triplets

`TdaSpin::Triplet` sets the coupling to zero, so the triplet spectrum is *exactly*
the bare orbital-energy gaps. Spin-restricted TD-DFTB without the spin-constant `W`
term — a model gap, not a bug.

### The TDA coupling kernel is the plain GFN1 shell-charge kernel

Experimental model layers (multipole/CAMM, Fock exchange, +U, spin polarization,
external field) are **not** in the TDA response kernel, and the TDA entry points do
not reject those option sets.

---

## Finite-difference steps that are not free parameters

Two entry points still take a finite-difference step as an argument, and in both
the *default* used to hand back a number the caller could not trust. Both are
repaired; what remains is the honest residual, recorded here.

### The magnetizability's field step is a Richardson pair, not a bare step

`magnetizability_tensor_analytic` / `magnetizability_diagonal_analytic` are
analytic in the orbital response but **not** in the LAO integrals: `H0^a`, `S^a`,
`H0^aa`, `S^aa` and the mixed `H0^ab`, `S^ab` come from central differences of the
London builder along the *fixed global* field axes. A finite difference taken
along global axes is not a tensor, so its truncation error is frame dependent —
it moves when the molecule is translated or rotated even though the physics is
exactly invariant (London orbitals make `χ` gauge-origin independent as an
identity).

Measured on non-eq water at the commonly used `step = 4e-3`, **before** the
repair: a 2-bohr rigid translation moved `χ` by rel `6.3e-6`, a 9.4-bohr one by
rel `1.3e-3` (the effective FD parameter carries the LAO phase area, so the error
grows like `|d|⁴`), and `χ(Rr) = R χ(r) Rᵀ` broke by rel `2.7e-6`.

The repair is two-part, and each part is needed:

- **Recentring.** The molecule is moved onto its centroid before the field
  derivatives are taken (the gauge origin travels with it, so a simultaneous
  electric field is unaffected). Translating a molecule is an exact gauge
  transformation of `S(B)`/`H0(B)`, so this changes no physics — it just stops
  the FD parameter from depending on where the caller put the molecule.
- **Richardson.** Every difference is built at `step` *and* `step/2` and combined
  as `(4D(h/2) − D(h))/3`, leaving `O(step⁴)`. **`step` is therefore the coarse
  node of a pair**, and the routines cost twice the builder evaluations (still
  one SCC).

Shrinking the step instead does not work, and that is the interesting part: the
bare rotation residual falls exactly `4.00×` per halving down to rel `~1e-8` near
`step = 2.5e-4`, then **rises again** as `1/h²` amplification of builder rounding
takes over. There is no single step that reaches rel `1e-9`.

Residuals now (gated in `tests/magnetizability_frame_invariance.rs`): translation
rel `2.6e-11 … 1.3e-10`, rotation rel `2.7e-11 … 8.6e-11`, both at the SCC
convergence floor. Useful steps are `4e-3 … 1.6e-2`; below `2e-3` the
extrapolation degrades.

### `γ⁽²⁾` needs its own volumetric step, and cutoffs that scale

`GruneisenOptions::delta_second` (default `2e-2`) carries the second-order node
set, separately from `delta` (default `5e-3`). `γ⁽²⁾` is a second difference, so
phonon noise reaches it as `ε/δ²` against `ε/δ` for the first-order `γ`; one
shared step cannot serve both.

The noise it was amplifying was not intrinsic: the real-space cutoffs were held
at a **fixed radius** while the cell was strained, so the integer image lists
stepped and `ln λ(ln V)` acquired `~5e-7` jumps. `pbc_gruneisen` now scales every
cutoff by the node's linear factor `(V/V₀)^{1/3}`, freezing the image set across
the stencil. Before both fixes, the three- and five-point stencils returned
`−0.0372` and `+0.0674` for the same diamond `γ⁽²⁾` — opposite signs, gap 1.5x the
value; after, `−0.0371` and `−0.0370` (rel `1.9e-3`).

What remains: `γ⁽²⁾` is two orders of magnitude below `γ`, so it is a
percent-level quantity — lean cutoffs give `−0.0371`, production cutoffs
`−0.0382` (3% apart). Quote it as "`γ` is volume-independent to within what GFN1 resolves",
not to four figures. The separate step also costs two extra Hessians (four with
`FivePoint`); `delta_second: Some(options.delta)` puts them back on the
first-order nodes for free, with 16x the noise amplification. See
[pbc.md §4](pbc.md#4-grüneisen-parameters).

### A too-lean periodic AO cutoff breaks the overlap, not just the accuracy — *rejected*

`ao_cutoff` looks like a pure speed/accuracy dial. It is not: truncating the Bloch
image sum part-way through a shell of self-images can leave the Γ-point overlap
matrix **indefinite**, and then the SCC does not converge slightly worse — it does
not run at all.

> `overlap matrix is not positive definite; eigenvalue 0 = -6.197e-1`
>
> — `src/linalg.rs`

The in-tree case is **rocksalt NaCl** (8 atoms, `a = 5.64 Å ≈ 10.66 bohr`) under
the "lean" periodic settings used by most of the fast gates (`ao_cutoff = 12`,
`real_cutoff = 18`, `sr_cutoff = 8` bohr): `run_pbc_scc` errors out with the line
above, which is why the Berry-polarization gate runs NaCl at `ao_cutoff = 30`,
`real_cutoff = 40` instead (`tests/physical_consistency.rs`). Rocksalt LiH was
rejected as a Γ-FC3 fixture for the same reason (`eigenvalue 0 = -1.8e-1`).

The failure is loud, so nothing silently wrong ships — but it means **a cutoff that
works for a covalent cell is not transferable to an ionic one**. If a new periodic
system errors here, raise `ao_cutoff` before suspecting the physics.

---

## Model / physics scope

- **D3 ATM higher derivatives under PBC.** The ATM term has energy + analytic
  forces + analytic stress for periodic systems and the full Jet2/3/4 ladder for
  molecules. Official GFN1-xTB has `s9 = 0`, so stock GFN1 is unaffected.
- **Open-shell magnetic properties** (spin-Zeeman, NMR shieldings) are not
  implemented; the closed-shell magnetics are complete.
- **Optical MCD/Faraday response** requires frequency-dependent complex
  response and is not supplied by the legacy `mcd_tensor` static API. Its
  `d alpha/dB` vanishes at zero field for a time-reversal-symmetric closed
  shell, including a length-gauge dipole model. Raw LAO integrals alone do not
  provide the frequency-dependent response.
- **Variable-cell relaxation** is available through `pbc_relax`, native
  `relax_periodic` and ASE `get_relax_periodic`; the older
  `optimize_geometry` / `optimize_native` retains its fixed-cell behavior.
  See [mechanical-properties.md](mechanical-properties.md). Convergence of
  forces/stress does not certify positive phonon curvatures.
- **Relaxed-ion Grüneisen** is not implemented (frozen-ion only).
- **PBC external-field SCC** retains its reference-cell site-potential model.
  The separate analytic bulk dielectric response differentiates the discrete
  Berry electric enthalpy. Analytic Born charges and proper piezoelectricity
  are available for scalar/structural-ML 3D insulators at their requested
  temperature; appreciably fractional occupations are rejected. It is not a
  nonlinear finite-field SCC driver for thermal metals. The spin-restricted
  polarization quantum is `2 e R / V`; see
  [electromechanical.md](electromechanical.md).
- The dipole and polarizability use GFN1 point-charge (monopole) electrostatics,
  so intra-atomic polarization is absent — treat absolute intensities as
  qualitative.
- **NMR shieldings use a common gauge origin, so they are genuinely
  origin-dependent.** `nmr_shielding_tensor` takes `gauge_origin` as a required
  argument and builds the orbital-Zeeman perturbation as `-i L_O` with **no
  London phase** (`s_a = 0`) — unlike the magnetizability, which is gauge-origin
  invariant *structurally* because it is built in London atomic orbitals. With a
  finite valence-only basis the CGO result therefore moves when the origin moves,
  and there is no cancelling term. Every in-tree caller (CLI `--nmr`, the Python
  binding, the tests) pins the origin at the **shielded nucleus's own position**;
  if you call the Rust API directly, keep that convention or your numbers will not
  be comparable to anything else in the project. Against the published GIAO/London
  reference the absolute values differ by roughly **1–2×, worst for ¹H**, which is
  why the gate asserts only correlation (`r² > 0.9`, positive slope) across the
  −420 … +60 ppm range rather than agreement. *There is currently no test that
  translates the gauge origin and measures the drift.*
- The GFN1 valence-only basis omits core electrons, so NMR shieldings track
  within-method trends rather than all-electron references.
- `pauling_en` rejects `Z > 86` — entries above Rn were 1.50 placeholders and the
  GFN1 parametrization ends at Rn.
- **Stiff experimental combinations** (high-order multipoles + long-range exchange
  on transition-metal complexes) can still fail to converge. That is a
  model/numerical limit, not a flag bug.

### PH3 high-rank 0 K convergence — resolved

The former joint charge/moment mixer oscillated on asymmetric PH3 at rank 4,
then changed the requested 0 K to 3000 K. High-rank, non-solvated 0 K models
now minimize the unchanged restricted, integer-occupation energy directly over
orbital rotations. The quadratic multipole response enters the Newton step;
unit-amplitude response evaluation avoids cancellation in small Krylov vectors.
Near machine energy resolution, a step must reduce the physical gradient.

This functional can have a non-Aufbau orbital minimum: globally refilling its
lowest Fock eigenvalues discards the minimum. Canonicalization therefore acts
within equal-occupation subspaces and retains their occupations and density.
Returned energies are sorted with the occupation labels attached. This is a
restricted determinant minimum, not a fractional-occupation ensemble result.
The actual temperature, full Cartesian gradient differences and stationary
Pulay relation are recorded in `tests/data/multipole-rank4-ph3-diagnostic.json`.
High-rank nuclear derivatives remain implemented through gradients only;
chemical accuracy and finite-temperature ensemble behavior require separate
validation. The earlier 3000 K fallback measurements remain historical failures.

