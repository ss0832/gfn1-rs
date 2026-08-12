# Known limitations

An honest inventory of what does **not** work, what errors explicitly, and — most
importantly — what can still be silently wrong. Kept next to the feature docs on
purpose: a feature is not documented until its gaps are.

Convention used below:

- **Rejected** — the code returns an explicit `Err` with a message. Safe.
- **Silent** — the code returns a number that may be wrong. Read the note.

---

## Finite temperature / Fermi smearing

### Analytic FC3 (dense / vector / block) — *rejected*

`third_derivative_analytic`, `_dense`, `_vector`, `_block` reject fractional
occupations. The closed-form response-derivative algebra assumes integer (0/2)
occupations; with smearing it would silently return wrong numbers.

> `analytic third derivative with fractional (Fermi-smeared) occupations is not yet supported; use third_derivative_seminumerical_* until the finite-temperature analytic path lands`
>
> — `src/third_derivative/mod.rs`

**Workarounds:** `third_derivative_seminumerical_*` (full smearing support), or
`third_derivative::finite_t::directional_third_finite_t` for the directional
contraction.

### Analytic FC4 — *rejected*

`QuarticReference::build` — and therefore `directional_fourth_derivative`,
`directional_fourth_with_reference`, `fourth_derivative_analytic_dense` and
`fourth_derivative_analytic_block` — rejects fractional occupations.

> `analytic fourth derivative with fractional (Fermi-smeared) occupations is not yet supported; use directional_fourth_seminumerical until the finite-temperature analytic path lands`
>
> — `src/fourth_derivative/assemble.rs`

**Workarounds.** There are now two, and the analytic one is easy to miss because
of where it lives:

- `third_derivative::finite_t::directional_fourth_finite_t`,
  `fourth_derivative_finite_t_dense` and `_block` are the **analytic**
  finite-temperature quartic. They sit in the `third_derivative::finite_t`
  module beside the finite-T FC3 — *not* under `fourth_derivative` — because
  they extend the same occupation-agnostic product-rule chain. Rust-only (no
  Python binding).
- `directional_fourth_seminumerical` (central FD of the analytic third
  derivative) remains the verification reference and the front-end fallback.

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

**The fix stops at second order — see the next entry.**

### Exactly degenerate + fractionally occupied third-order **response** — *rejected*

The resolvent form above closed the **second**-order channel. The third-order
*response* chain did **not** move with it:
`ChargeSpaceContext::solve_third_order_directional` is still written in the frame
algebra (its ingredient recursion is built from `frame(U, M) = UᵀM + MU` and
`u_second`), which is precisely the formulation whose in-block second order
depends on the arbitrary within-block eigenbasis. Measured on T_d Ni(CO)₄ at
3000 K, that chain lands at `3.5e3`.

Since v0.5.0 it is **guarded** rather than silent — `solve_third_order_directional`
rejects up front:

> `third-order finite-temperature response: orbitals {p} and {q} are exactly degenerate (gap {…}) with fractional occupation — the third-order assembly is still frame-based and the frame is not defined inside such a block (the second order takes the frame-free resolvent form instead); break the symmetry or use a seminumerical path`
>
> — `src/response/charge_space.rs` (`solve_third_order_directional`)

It fires only at finite temperature, for a fractionally occupied pair with an
orbital gap below `1e-10`.

**Which entry points actually inherit it — the FC4 ones, not the FC3 ones.** This
is the part that is easy to get backwards:

- **Rejected:** `directional_fourth_finite_t`, `fourth_derivative_finite_t_dense`
  and `_block`. They are the only production callers of
  `solve_third_order_directional` (through `directional_fourth_with_reference`).
- **Not affected — and *not* a limitation:** `directional_third_finite_t`,
  `third_derivative_finite_t_dense` and `_block`. They never reach the frame-based
  third-order chain; they use only `solve_second_order`, which *is* the frame-free
  resolvent form, so exact degeneracy is the confluent limit of a smooth
  expression rather than a special case. Verified directly on an exactly
  degenerate, fractionally occupied NiO reference (minimum level spacing
  `1.1e-16`) at 3000 K against a central FD of the smeared analytic Hessian
  contracted `vv`:

  | `h` | `\|analytic − FD\|` | ratio |
  | --- | --- | --- |
  | `2e-3` | `1.24e-9` | — |
  | `1e-3` | `3.10e-10` | 4.00 |
  | `5e-4` | `7.77e-11` | 4.00 |

  A clean `O(h²)` ladder onto the analytic value — the residual is FD truncation,
  not a missing term.

**Workaround for the FC4 case:** break the symmetry (a near-degenerate reference,
gap ≥ `1e-10`, is fully supported and gated at `5.5e-9`), or use
`directional_fourth_seminumerical`. The fix is to rewrite the third-order chain in
the same resolvent form — the second-order case is the worked precedent.

> **Stale rustdoc, tracked here so it is not mistaken for behaviour.**
> `src/third_derivative/finite_t.rs` still tells
> `directional_third_finite_t`'s reader that exactly degenerate blocks "are
> rejected by the second-order solver's guard", and `src/python.rs` repeats it.
> That guard was removed when the resolvent form landed; the measurements above
> are the current behaviour. Likewise `tests/smearing.rs`'s
> `exactly_degenerate_smeared_third_is_rejected` still asserts the FC3 path
> errors, and therefore **fails** — it pins a guard the codebase deliberately
> replaced.

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

### `enable_cn_hamiltonian = false` is not supported at fourth order — *silent*

Turning the coordination-number Hamiltonian off is honoured by the SCC, the
gradient and the Hessian, but the analytic quartic only honours it in **some** of
its stages. `QuarticReference::build` has exactly two guards —
`terms::require_order(…, 4, …)` and the fractional-occupation rejection — and
neither looks at this flag, so nothing errors.

| FC4 stage | honours `include_cn_h0`? |
| --- | --- |
| second-order legs (`cn_block`) | yes |
| stage 4, CN response | yes (returns `0.0`) |
| stage 5, response | yes |
| **stage 2, frozen density** | **no** — CN/H0 blocks built unconditionally |
| **stage 3, Hessian path** | **no** — same |

The result is the quartic of *neither* energy expression: a finite, plausible
number assembled from two different Hamiltonians. The `T = 0` equality against the
seminumerical reference widens from `~3e-15` to `~2e-5` when the flag is cleared.

**Workaround:** leave `enable_cn_hamiltonian = true` (the default, and the actual
GFN1 model) for any fourth-derivative work. There is no reason to clear it except
for term-isolation experiments, and those should use the seminumerical route.

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
coverage), and the seminumerical
`pbc_kpoint_third_derivative_seminumerical_dense/_vector` remains the route for
Fermi-smeared cells and for option sets the term registry caps below analytic
order 3.

### The analytic periodic FC3 carries a `~1e-7` residual, and no smearing

Two separate caveats, and they apply to **both** the Γ
(`pbc_gamma_third_analytic_*`) and the k-point (`pbc_kpoint_third_analytic_*`)
entry points — the k-point path reuses the Γ assembly, so it inherits the Γ
assembly's open residual exactly:

- **Accuracy.** The assembly agrees with the Richardson-extrapolated
  seminumerical reference to `~1e-7` absolute (`8.5e-8` diamond, `8.0e-8` BN on
  a `~10⁻³` Eh/Bohr³ scale). That residual is `h`-independent — a genuinely
  missing term, not FD truncation — and is localised to the band/Pulay block
  (CN and the two SCC2 blocks close exactly). It is ~1500x smaller than the
  residual the assembly had before the density path was added, but it has not
  been driven to zero; if you need better than `1e-7` absolute on the cubic
  force constants, use the seminumerical route. Tracked in `docs/pbc.md` §2c.
- **Integer occupations only.** Fermi-smeared periodic systems are rejected (at
  *every* k-point, for the k-point path). The molecular FC3 has a native
  finite-temperature path (`third_derivative_finite_t_*`); the periodic one does
  not.

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

### FC4 has no CLI front end

No CLI flag. It **does** have Python bindings as of v0.5.0 —
`fourth_derivative_directional`, `fourth_derivative_directional_seminumerical`,
`fourth_derivative` (dense) and `fourth_derivative_block`, gated by
`tests/python/test_v050_derivatives.py`. The dense and block bindings go through
`check_fourth_dense_size`, so they carry the `n⁴`-expansion cap described above
(applied to `|dofs|` in block mode); the directional binding does not.

What is still Rust-only on the quartic side is the **finite-temperature** FC4
(`third_derivative::finite_t::directional_fourth_finite_t`,
`fourth_derivative_finite_t_dense` / `_block`) — see the note below the FC4
rejection entry.

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

### Integer occupations only — *rejected*

Every TDA path needs a gapped closed shell at `T = 0`:

> `TD-GFN1 requires a positive occupied-virtual gap (gapped closed shell)`

> `PBC TDA analytic gradient requires integer (gapped) occupations`

> `k-point TD-GFN1 requires an integer closed-shell band filling (gapped insulator)`

> `k-point TD-GFN1 found no positive-gap occupied->virtual transitions (metallic or non-integer occupations are not supported)`

### Semi-numerical gradients are non-periodic — *rejected*

> `solve_tda_gradient_seminumerical is non-periodic; use solve_tda_gradient (finite difference) for periodic (Gamma-point) systems`

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
- **Length-gauge orbital-current** optical-rotation `G`-tensor and MCD vanish
  identically in the GFN1 point-charge model (`dq/dB = 0`). The
  `lao_dipole_matrix` integrals are the route to the orbital-current versions.
- **Variable-cell relaxation** is not native; drive the lattice externally from
  `pbc_stress` (e.g. an ASE cell filter).
- **Relaxed-ion Grüneisen** is not implemented (frozen-ion only).
- **PBC external field** is the reference-cell dipole-coupling approximation (a
  finite field is a sawtooth site potential, not a Berry-phase enthalpy). The bulk
  **polarization** itself is implemented — `pbc_berry_polarization`, see
  [pbc.md](pbc.md#7-berry-phase-bulk-polarization) — but only the polarization:
  there is no analytic `dP/dR` (Born-charge) entry point and no finite-field Berry
  enthalpy, and the spin-restricted quantum is `2 e R / V`, not `e R / V`.
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
