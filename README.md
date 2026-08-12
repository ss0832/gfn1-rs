# gfn1-rs

A Rust implementation of GFN1-xTB energies, analytical nuclear gradients,
stress, Hessians, cubic (FC3) and quartic (FC4) force constants, and periodic
(Gamma / Monkhorst-Pack k-point) electrostatics, with external electric fields,
IR/Raman spectroscopy, parameter tooling, and an optional Python/ASE interface.
The official parametrization is **bundled** — nothing to download. Dense linear
algebra uses `faer`; no BLAS/LAPACK required. **Unofficial — use with caution.**

Names: Cargo package `gfn1-rs`; Rust library target `gfn1_rs`; CLI binary
`gfn1_rs_cli`; Python package `gfn1_rs` (PyO3 extension module `gfn1_rs._native`).

## Scope

Headlines for **v0.5.0** — everything below is analytic and finite-difference
gated unless it says otherwise:

- **Energy, gradient, stress** — molecular and periodic (Γ and Monkhorst–Pack
  k-points), with generalised-Ewald Klopman–Ohno electrostatics.
- **Hessian** — molecular, periodic Γ, and periodic k-point.
- **Cubic force constants (FC3)** — the strict closed form for molecules, a
  native **finite-temperature** analytic path, and — new in v0.5.0 — the closed
  form under PBC at **Γ *and* on an arbitrary k-mesh**, plus seminumerical
  references for every one of them.
- **Quartic force constants (FC4)** — fully analytic, no nuclear finite
  differences; the **directional route has no degrees-of-freedom cap** as of
  v0.5.0, and a finite-temperature analytic path exists alongside it.
- **Charge-space response solver** — one LU-factored dielectric serves every
  first- and second-order right-hand side; Fermi smearing is native, and the
  exactly-degenerate second order is closed in Daleckii–Krein resolvent form.
- **Periodic thermal expansion** — strain-mixed `dH/dlnV` and mode/thermodynamic
  Grüneisen parameters (with a separate second-order volumetric step).
- **Bulk polarization** — Berry phase, King-Smith–Vanderbilt and Resta forms.
- **Excited states** — TD-GFN1 (TDA) singlet/triplet, non-periodic, Γ and
  k-point, with analytic excited-state gradients.
- **Fields and magnetics** — external electric field (molecular and periodic),
  closed-shell magnetic SCC in London orbitals (GFN1-xTB-M0/M1), magnetizability
  (finite-field and analytic), NMR shieldings, and magneto-optical tensors.
- **Spectroscopy and tooling** — IR/Raman, polarizability, geometry
  optimization, parameter round-trip and finite-difference parameter
  derivatives, Python/ASE bindings.

Experimental, off by default (stock GFN1 is byte-identical with them off):
multipole/CAMM electrostatics, higher-order on-site charge, range-separated Fock
exchange, DFT+U/+V, collinear spin polarization, and self-consistent D4.

**The full feature matrix — including the FC3/FC4 coverage tables — lives in
[docs/scope.md](docs/scope.md); the honest list of gaps is in
[docs/limitations.md](docs/limitations.md).**

## Documentation

The docs are organised one page per subsystem — see the index at
[`docs/README.md`](docs/README.md):

- [`docs/scope.md`](docs/scope.md) — the full feature matrix, including the
  FC3/FC4 coverage tables (molecular / PBC Γ / PBC k-mesh × `T = 0` / finite `T`).
- [`docs/parameters.md`](docs/parameters.md) — bundled official parameters,
  `Gfn1Parameters::resolve` precedence, `ParamSource` provenance reporting,
  reference data, unit constants.
- [`docs/derivatives.md`](docs/derivatives.md) — the nuclear derivative ladder
  (gradient → Hessian → FC3 → FC4), the `terms::require_order` registry, and the
  verification-gate philosophy (analytic order `n` vs FD of order `n−1`, `h²`
  ladders).
- [`docs/finite-temperature.md`](docs/finite-temperature.md) — the charge-space
  dielectric response solver (first and second order, natively Fermi-smeared) and
  the analytic finite-temperature FC3.
- [`docs/pbc.md`](docs/pbc.md) — periodic derivatives, the **analytic** periodic
  third derivative (Γ and k-point) and its seminumerical reference, strain-mixed
  `dH/dlnV`, Grüneisen parameters, and Berry-phase bulk polarization.
- [`docs/td.md`](docs/td.md) — TD-GFN1 (TDA) excited states: working equations,
  the analytic excited-state gradients, the MO phase gauge, and the gate numbers.
- [`docs/limitations.md`](docs/limitations.md) — **known gaps and silent failure
  modes**, with the exact error messages. Read this one.
- [`docs/rust-api.md`](docs/rust-api.md) — Rust library API (`run_electronic`,
  `analytic_gradient`, `analytic_hessian`, `optimize_geometry`, the periodic
  `run_pbc_scc` / `pbc_analytic_gradient` / `pbc_stress` / `pbc_gamma_hessian`
  entry points, `vibrational_analysis`, the external-field / spectroscopy API
  (`ExternalFieldOptions`, `dipole_derivatives`, `static_polarizability`,
  `ir_spectrum`, `raman_spectrum`), the third/fourth-derivative APIs, and the
  parameter tooling (`to_param_string`, `ParameterTarget`,
  `parameter_finite_difference`)).
- [`docs/python-api.md`](docs/python-api.md) — Python (`Gfn1NativeCalculator`)
  and ASE (`GFN1RSCalculator`) API, including charge/spin and SCC controls.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — project discipline: **docs are
  updated in the same commit as the feature that lands.**

## Build

```bash
cargo build --release          # CLI -> target/release/gfn1_rs_cli
```

Python extension (Python 3.9–3.14):

```bash
python -m pip install maturin "ase>=3.22" "numpy>=1.22"
maturin develop --release --features python
```

For a Python newer than PyO3's support window, set
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `cargo`/`maturin`.

### Performance

The dense linear algebra (`faer`) runs **multi-threaded** by default, using the shared
rayon pool — the O(N³) SCF eigensolve and the analytic-gradient assembly scale across
all cores (≈1.9× on a 300-atom cluster vs single-threaded). The closed-form cubic
force constants additionally compute their independent per-slab response in parallel over
the same pool, and offer Dense / Vector / Block output so you only build what you need;
the quartic force constants evaluate their deduplicated polarization directions in
parallel against one shared reference. Diagnostics:

- `GFN1_PROFILE=1` — emit per-region wall-clock timings to stderr.
- `GFN1_FAER_THREADS=1` — force single-threaded `faer` (reproducible benchmarking).

- `GFN1_PBC_MULTIPOLE_FIELD_CACHE_MB=256` controls the periodic multipole field-cache cap;
  larger cells automatically use the lower-memory streaming path instead of risking OOM.

## Parameters

The official GFN1-xTB parametrization is **bundled** — everything works out of
the box with no downloads. The bundled files (`param_gfn1-xtb.txt` and the
GFN1(Si)-xTB silicon reparametrization `param_gfn1-si-xtb.txt`) are verbatim
copies from [`grimme-lab/xtb`](https://github.com/grimme-lab/xtb)
(LGPL-3.0-or-later; see `third_party/xtb/PROVENANCE.md` and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

Resolution order (`Gfn1Parameters::resolve`): `--param` / `param_path` >
`GFN1_XTB_PARAM` environment variable > builtin. Every run reports which
parametrization is active (CLI banner `parameters: ...`, Python
`calc.param_source()`). Details in
[`docs/parameters.md`](docs/parameters.md).

- `--param builtin` selects the bundled GFN1-xTB set explicitly;
  `--param builtin:si` selects the bundled GFN1(Si)-xTB set.
- To use your own file: `--param PATH` or

```bash
export GFN1_XTB_PARAM=/path/to/param_gfn1-xtb.txt   # (PowerShell: $env:GFN1_XTB_PARAM = "...")
```

GFN2 parameter files are rejected intentionally.

### D3 reference data

D3(BJ) uses the `s-dftd3` reference table. A minimal LGPL-3.0-or-later
`simple-dftd3` data copy is bundled under `third_party/simple-dftd3` (see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)); an explicit
`--d3-reference PATH` / `GFN1_D3_REFERENCE` still takes precedence.

### Experimental D4 dispersion

`--experimental-d4` replaces the default D3 term with an experimental
non-periodic, self-consistent D4 term, including the Axilrod-Teller-Muto
three-body contribution. The D4 charge potential is fed back into SCC as an
atomic scalar shift, and non-PBC analytic gradients include the fixed-charge D4
geometry and coordination-number response. The GFN1 damping constants `a1`,
`a2`, and `s8` are read from the active `param_gfn1-xtb.txt`; `s9` is an
API/CLI option that defaults to the GFN2-xTB value `5.0` only when D4 is active
(ordinary non-D4 calculations use `s9 = 0`). The saved upstream DFT-D4 reference
data and provenance live under `third_party/dftd4`.

### Charge and spin multiplicity

Formal charge (`--charge`) and spin multiplicity (`--multiplicity`) are passed
through `ElectronicOptions`. Multiplicity constrains the alpha/beta occupations of
the spin-independent GFN1 Hamiltonian.

**Optional spin polarization (spGFN1).** Plain GFN1 is restricted-open-shell (the
energy does not depend on the spin density). Setting `--spinpol` (CLI) /
`spin_polarization=True` (Python / ASE) / `ElectronicOptions::spin_polarization`
adds the collinear spin-DFTB spin-polarization energy from the tabulated atomic
spin constants `W` ("spGFN1"): non-PBC, **energy + analytic forces**, off by
default. It only affects **open-shell** configurations — a closed-shell singlet is
**byte-identical** to plain GFN1. The `W` constants are transcribed byte-exact from
tblite (LGPL-3.0; provenance under `third_party/tblite/` and `THIRD_PARTY_NOTICES.md`).
v1 is the bare GFN1 model (not combinable with the experimental multipole / CAMM /
exchange / D4 / field paths). See [scope.md](docs/scope.md).

**Extended Hubbard correction (DFT+U / +U+V), non-empirical.** GFN1 describes the
on-site charge fluctuation only at the monopole (shell-charge) level and carries no
orbital-resolved penalty against the self-interaction / over-delocalisation of a
localised transition-metal `d` shell — the diagnosed root cause of its poor TM
spin-state energetics. `--plus-u` (CLI) / `plus_u=True` (Python / ASE) adds the
rotationally-invariant FLL `+U` energy (and, with `--plus-u-v`, the inter-site `+V`
that restores metal–ligand hybridisation), built from the dual population of the
correlated subspace; **energy + analytic forces** (overlap-Pulay term, frozen-`U`),
on the open-shell spin path (requires `--spinpol`), non-PBC, off by default. With
`--linear-response-u` the `U` (and `V`) are computed **non-empirically** from the
occupation response (Cococcioni–de Gironcoli, `U = χ0⁻¹ − χ⁻¹`) and the correlated
`d` shells are auto-selected — **no fitted parameters** (the FD step and `+V` cutoff
are numerical/geometric settings, not fits). Experimental; in the screened
semiempirical SCC the bare/screened response separation is approximate. See
[scope.md](docs/scope.md).

## CLI

Coordinates are XYZ/extXYZ in Angstrom (use `--bohr` for Bohr). A periodic cell is
read from the extXYZ `Lattice="..."` header and auto-dispatches the periodic path.
See `--help` for the full option set. **No `--param` is needed** — the official
parametrization is compiled in, so the examples below run as written from a fresh
checkout; pass `--param builtin:si` for the GFN1(Si) set or `--param PATH` for
your own file. Common workflows:

```bash
# single-point energy (+ Mulliken charges)
gfn1_rs_cli examples/water.xyz --charges

# energy + analytical gradient / forces
gfn1_rs_cli --gradient examples/ammonia.xyz

# analytical Hessian (non-periodic)
gfn1_rs_cli --hessian examples/water.xyz

# periodic single point + component stress (Lattice from the extXYZ header)
gfn1_rs_cli --stress examples/diamond.xyz

# L-BFGS optimization (non-periodic; analytical gradient). --opt-output writes the final
# geometry, --opt-traj writes the full multi-frame XYZ trajectory.
gfn1_rs_cli --optimize --opt-output opt.xyz --opt-traj opt_traj.xyz examples/ammonia.xyz

# mDFTB2 multipole-corrected single point / optimization (experimental, non-PBC)
gfn1_rs_cli --multipole --optimize --opt-output opt.xyz examples/ammonia.xyz

# CAMM-on-mDFTB2: GFN2-style anisotropic electrostatics (q-mu/q-Theta/mu-mu) on cumulative
# atomic multipoles, replacing the mDFTB2 off-site term (experimental, non-PBC, v0.4.2).
# camm_damp (kappa) is the primary range-selective lever; camm_aes_scale (s_AES) the secondary amplitude.
gfn1_rs_cli --multipole --multipole-model camm_on_mdftb2 --camm-damp 1.0 examples/water48.xyz

# experimental self-consistent D4 single point / gradient (non-PBC only)
gfn1_rs_cli --experimental-d4 --gradient examples/water.xyz

# range-separated Fock exchange single point (experimental, non-PBC); --onsite-exchange adds
# the exact one-center (OFX) term, --scf-trah forces the TRAH second-order SCF
gfn1_rs_cli --lr-exchange --onsite-exchange --charges examples/water.xyz

# charged / open-shell single point
gfn1_rs_cli --charge 1 --multiplicity 2 examples/h2.xyz

# spin-polarized GFN1 ("spGFN1"): collinear spin term for open-shell systems (non-PBC, energy +
# analytic forces). Off by default; a closed-shell singlet stays byte-identical to GFN1.
gfn1_rs_cli --multiplicity 3 --spinpol --gradient examples/h2.xyz

# TD-GFN1 (TDA) excited states; --tda-grad adds the excited-state gradient. The CLI default is
# --tda-gradient-method semi-numerical; pass `analytic` for the fully analytic direct-CPHF gradient
# (or `fd` for the root-tracked full finite difference).
gfn1_rs_cli --tda examples/water.xyz

# single point in a uniform external electric field (atomic units, Ex Ey Ez)
gfn1_rs_cli --field 0.0 0.01 0.0 --charges examples/water.xyz

# static polarizability, IR, and Raman (each also prints the raw derivative tensors)
gfn1_rs_cli --polarizability examples/water.xyz
gfn1_rs_cli --ir examples/water.xyz
gfn1_rs_cli --raman examples/water.xyz

# NMR nuclear magnetic shielding tensor of atom 0 (ppm; isotropic + 3x3)
gfn1_rs_cli --nmr 0 examples/water.xyz

# finite-difference parameter derivatives (dE/dp, add --param-forces for dF/dp)
gfn1_rs_cli --param-deriv \
  --targets glob:ks,elem:1:GAM,pair:1:1 examples/h2.xyz
```

`cargo run --bin gfn1_rs_cli -- ...` works too if you have not built the binary.

IR/Raman are taken at the input geometry, so optimize first (`--optimize`) for a
clean spectrum free of spurious low-frequency modes.

## Python / ASE

Periodic **single points** (energy / forces / stress) dispatch automatically for
ASE `Atoms` with a cell/pbc (Monkhorst-Pack via the `kgrid` parameter), and
periodic **k-point TD** is exposed (`get_tda_kpoint`); the property/spectroscopy
methods (polarizability, IR/Raman, magnetic) are non-periodic.

```python
from ase import Atoms
from ase.optimize import LBFGS
from gfn1_rs.ase import GFN1RSCalculator      # reads GFN1_XTB_PARAM, or pass param_path=...

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
atoms.calc = GFN1RSCalculator(charge=0.0, multiplicity=None)
print(atoms.get_potential_energy())            # eV
print(atoms.get_forces())                      # eV/Angstrom
print(atoms.get_dipole_moment())               # e*Angstrom
print(atoms.calc.results["native_energy_terms_hartree"])

# ASE-driven optimization, or the Rust-native L-BFGS:
LBFGS(atoms).run(fmax=0.05)
atoms.calc.optimize_native(atoms, gradient_tolerance=1e-4)
```

### Units: two layers, two conventions

The package has exactly one unit boundary, and it is `gfn1_rs.ase`:

| Layer | Convention |
| --- | --- |
| `gfn1_rs.ase.GFN1RSCalculator` (ASE) | **ASE units** — Angstrom, eV, eV/Angstrom, eV/Angstrom³, e·Angstrom, e |
| `gfn1_rs.Gfn1NativeCalculator`, `gfn1_rs.torch_interop`, the CLI, the Rust API | **atomic units** — bohr, Hartree, Hartree/bohr, Hartree/bohr³, e·bohr |

Everything the ASE calculator returns is in ASE units, and **every conversion is
taken from `ase.units`** (`Bohr`, `Hartree`, `invcm`) — there are no hand-typed
factors in the ASE layer, and it deliberately does not use the engine's own
CODATA constants (they differ at the 1e-8 level). Concretely:

- `get_potential_energy()` eV · `get_forces()` eV/Å · `get_stress()` eV/Å³ in the
  ASE Voigt order `(xx, yy, zz, yz, xz, xy)` with the ASE sign convention
  `sigma = (1/V) dE/d(strain)` · `get_dipole_moment()` e·Å · `get_charges()` e
- `get_hessian()` eV/Å², `get_third_derivative*()` eV/Å³,
  `get_polarizability()` / `get_magnetic_polarizability()` e²Å²/eV,
  `get_magnetic_energy()` eV
- gauge/multipole `origin` arguments are read in **Angstrom**
- dicts follow a suffix rule: an unsuffixed key (`forces`, `total_energy`) is in
  ASE units, a suffixed one (`*_hartree`, `*_hartree_per_bohr`, `*_au`, `*_ev`)
  is in the unit its name states — so `results["native_energy_terms_hartree"]`
  and `get_tda_gradient()["forces_hartree_per_bohr"]` still give you the raw
  atomic-unit numbers
- observables with a universal unit of their own keep it, exactly as ASE's own
  `ase.vibrations` does: cm⁻¹ wavenumbers, km/mol IR intensities, ppm NMR
  shieldings, 1e-30 J/T² magnetizabilities; raw AO-integral and magnetic-response
  tensors stay in atomic units and say so in their docstrings
- the calculator's **construction parameters** are model knobs passed straight to
  the engine, so they stay in native units (`electric_field` a.u.,
  `hubbard_u`/`hubbard_v`/`level_shift`/`energy_tolerance` Hartree,
  `hubbard_v_cutoff` and the `d4_*` cutoffs bohr, `electronic_temperature` K), as
  do the numerical `step` / `e_step` / `b_step` / `field_step` arguments

The low-level `gfn1_rs.Gfn1NativeCalculator` exposes the same controls without
ASE, in atomic units. Set `experimental_d4=True` on either interface to use the
non-PBC D4 path. Use `d4_s9=0.0` / `--d4-s9 0` or `--no-d4-atm` to disable the D4
three-body term explicitly.

## Validation

Tests resolve parameters through `Gfn1Parameters::resolve(None)`, so they run
against the **bundled** parametrization with no environment variable — and, unlike
before, no longer silently no-op when `GFN1_XTB_PARAM` is unset:

```bash
cargo test --profile reltest              # use reltest: these are numerical gates
cargo test --profile reltest -- --ignored # long-running (FD probes, finite-T ladders)
maturin develop --release --features python && python -m pytest -q
```

The `reltest` profile is release-grade optimization without fat LTO or
single-codegen-unit linking, so the numerical gates run at production speed
without re-linking every test binary at full LTO cost. A plain debug-profile
`cargo test` is correct but *much* slower on this suite.

Set `GFN1_XTB_PARAM=/path/to/param_gfn1-xtb.txt` only to test against a
*different* parametrization.

Tests cover finite-difference gradient / Hessian checks
(`tests/gradient_fd_probe.rs`, `tests/hessian.rs`), the analytic third and fourth
derivatives against their seminumerical references (`tests/third_derivative.rs`,
`tests/fourth_derivative.rs`), the periodic path (`tests/pbc_integration.rs`) and
its third derivative / Grüneisen module (`tests/pbc_third_derivative.rs`), the
optimizer (`tests/optimizer.rs`), and an opt-in external `tblite` parity suite
(`tests/tblite_parity.rs`, enabled by setting `GFN1_TBLITE_BIN`; `tblite` is not a
Cargo dependency). `GFN1_FD_TIGHT=1` tightens the finite-difference probe and
`GFN1_PROFILE=1` emits per-scope native timings.

The gate design — analytic order `n` against a central FD of the *validated*
analytic order `n−1`, with the `h²` truncation ladder as the completeness
diagnostic — is described in
[`docs/derivatives.md`](docs/derivatives.md#5-the-verification-gate-philosophy).

The Python suite lives in `tests/python/` (collected by the `pytest` config in
`pyproject.toml`). `tests/python/test_ase_units.py` pins the unit contract above:
every ASE-calculator output is asserted to equal the corresponding
`Gfn1NativeCalculator` (atomic-unit) output times the exact `ase.units` factor.
Its pure-Python half mocks the native layer, so it runs even without the compiled
extension.

## Limitations

The full, honest list — including the failure modes that are *silent* — lives in
[`docs/limitations.md`](docs/limitations.md). The headline items:

- **Fermi smearing reaches everything except the strict `T = 0` closed forms.**
  The `third_derivative::finite_t` drivers carry smearing through the analytic
  FC3 *and* FC4 (directional, dense and block); the strict closed forms
  (`third_derivative_analytic_*`, `fourth_derivative_analytic_*`) still reject
  fractional occupations with an explicit error pointing at them.
- **The analytic periodic FC3 carries a `~1e-7` residual and rejects smearing.**
  Both the Γ path (`pbc_gamma_third_analytic_*`) and the k-point path
  (`pbc_kpoint_third_analytic_*`, any Monkhorst–Pack mesh, vector / dense /
  block) are complete and agree with the seminumerical reference to `≈8e-8`; the
  two share one assembly, so they share that open residual. Fermi-smeared cells
  and order-1 model options still need
  `pbc_kpoint_third_derivative_seminumerical_*`.
- **The `n⁴`-expanded FC4 front ends are capped at 30 DOF**
  (`MAX_FOURTH_DERIVATIVE_NDOF`, the `8·n⁴`-byte expansion and the full-space
  `Jet4` working set of the D3/halogen full-tensor routes). The **directional**
  quartic has no such cap — 45 DOF is exercised in the test suite — and the
  block mode applies the cap to `|dofs|`, not to `3N`.
- **`pbc_kpoint_hessian` is wrong on exactly degenerate frontier orbitals**
  (perfect-symmetry cells; up to 100 % error, vanishing under a 0.03 Å
  distortion). The Γ path and every FC3 route are structurally immune; see
  [`docs/pbc.md`](docs/pbc.md) §2b.
- **Experimental model flags cap the derivative ladder at the gradient.** D4,
  multipole/CAMM, MFX/OFX exchange, +U, spGFN1 and an external electric field all
  carry `max_analytic_order: 1`; asking for a Hessian/FC3/FC4 with any of them on
  now fails fast instead of silently returning the derivative of a different
  energy expression.
- **Open-shell** magnetic properties (spin-Zeeman / NMR shieldings) and the
  **length-gauge orbital-current** optical-rotation `G`-tensor / MCD (both vanish in
  the GFN1 point-charge model since `dq/dB = 0`; the `lao_dipole_matrix` integrals
  are the route to the orbital-current versions). The closed-shell magnetics —
  analytic magnetizability and the FD-verified magneto-optical suite — are already
  complete; see [scope.md](docs/scope.md).

## References

1. A. Buccheri, R. Li, J. E. Deustua, S. M. Moosavi, P. J. Bygrave, F. R. Manby,
   "Periodic GFN1-xTB Tight Binding: A Generalized Ewald Partitioning Scheme for
   the Klopman–Ohno Function," *J. Chem. Theory Comput.* **21**, 1615–1625 (2025).
   DOI: [10.1021/acs.jctc.4c01234](https://doi.org/10.1021/acs.jctc.4c01234).
2. S. Grimme, C. Bannwarth, P. Shushkov, "A Robust and Accurate Tight-Binding
   Quantum Chemical Method for Structures, Vibrational Frequencies, and
   Noncovalent Interactions of Large Molecular Systems Parametrized for All
   spd-Block Elements (Z = 1–86)," *J. Chem. Theory Comput.* **13**, 1989–2009
   (2017). DOI: [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).
3. A. Hjorth Larsen et al., "The Atomic Simulation Environment — A Python library
   for working with atoms," *J. Phys.: Condens. Matter* **29**, 273002 (2017).
   DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
4. T. A. Niehaus, S. Suhai, F. Della Sala, P. Lugli, M. Elstner, G. Seifert, T.
   Frauenheim, "Tight-binding approach to time-dependent density-functional
   response theory," *Phys. Rev. B* **63**, 085108 (2001). DOI:
   [10.1103/PhysRevB.63.085108](https://doi.org/10.1103/PhysRevB.63.085108).
   (The TD-DFTB transition-charge response model behind TD-GFN1.)
5. C. Y. Cheng, A. M. Wibowo-Teale, "Semiempirical Methods for Molecular Systems
   in Strong Magnetic Fields," *J. Chem. Theory Comput.* **19**, 6226–6241 (2023).
   DOI: [10.1021/acs.jctc.3c00671](https://doi.org/10.1021/acs.jctc.3c00671).
   (The GFN1-xTB London-orbital magnetic-field method behind the `magnetic` module.)

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
