# gfn1-rs

A Rust implementation of GFN1-xTB energies, analytical nuclear gradients,
stress, Hessians, and periodic (Gamma / Monkhorst-Pack k-point) electrostatics,
with external electric fields, IR/Raman spectroscopy, parameter tooling, and an
optional Python/ASE interface. Dense linear algebra uses `faer`; no BLAS/LAPACK
required. **Unofficial — use with caution.**

Names: Cargo package `gfn1-rs`; Rust library target `gfn1_rs`; CLI binary
`gfn1_rs_cli`; Python package `gfn1_rs` (PyO3 extension module `gfn1_rs._native`).

## Scope


**The full feature matrix — lives in [docs/scope.md](docs/scope.md).**

## Documentation

Full API references live in `docs/`:

- [`docs/rust-api.md`](docs/rust-api.md) — Rust library API (`run_electronic`,
  `analytic_gradient`, `analytic_hessian`, `optimize_geometry`, the periodic
  `run_pbc_scc` / `pbc_analytic_gradient` / `pbc_stress` / `pbc_gamma_hessian`
  entry points, `vibrational_analysis`, the external-field / spectroscopy API
  (`ExternalFieldOptions`, `dipole_derivatives`, `static_polarizability`,
  `ir_spectrum`, `raman_spectrum`), and the parameter tooling
  (`to_param_string`, `ParameterTarget`, `parameter_finite_difference`)).
- [`docs/python-api.md`](docs/python-api.md) — Python (`Gfn1NativeCalculator`)
  and ASE (`GFN1RSCalculator`) API, including charge/spin and SCC controls.

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
the same pool, and offer Dense / Vector / Block output so you only build what you need.
Diagnostics:

- `GFN1_PROFILE=1` — emit per-region wall-clock timings to stderr.
- `GFN1_FAER_THREADS=1` — force single-threaded `faer` (reproducible benchmarking).

- `GFN1_PBC_MULTIPOLE_FIELD_CACHE_MB=256` controls the periodic multipole field-cache cap;
  larger cells automatically use the lower-memory streaming path instead of risking OOM.

## Parameters

The GFN1-xTB parameter file is **not** distributed. Obtain `param_gfn1-xtb.txt`
from the `xtb` repository and point to it via `--param PATH` or:

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
See `--help` for the full option set; common workflows:

```bash
# single-point energy (+ Mulliken charges)
gfn1_rs_cli --param param_gfn1-xtb.txt examples/water.xyz --charges

# energy + analytical gradient / forces
gfn1_rs_cli --param param_gfn1-xtb.txt --gradient examples/ammonia.xyz

# analytical Hessian (non-periodic)
gfn1_rs_cli --param param_gfn1-xtb.txt --hessian examples/water.xyz

# periodic single point + component stress (Lattice from the extXYZ header)
gfn1_rs_cli --param param_gfn1-xtb.txt --stress examples/diamond.xyz

# L-BFGS optimization (non-periodic; analytical gradient). --opt-output writes the final
# geometry, --opt-traj writes the full multi-frame XYZ trajectory.
gfn1_rs_cli --param param_gfn1-xtb.txt --optimize --opt-output opt.xyz --opt-traj opt_traj.xyz examples/ammonia.xyz

# mDFTB2 multipole-corrected single point / optimization (experimental, non-PBC)
gfn1_rs_cli --param param_gfn1-xtb.txt --multipole --optimize --opt-output opt.xyz examples/ammonia.xyz

# CAMM-on-mDFTB2: GFN2-style anisotropic electrostatics (q-mu/q-Theta/mu-mu) on cumulative
# atomic multipoles, replacing the mDFTB2 off-site term (experimental, non-PBC, v0.4.2).
# camm_damp (kappa) is the primary range-selective lever; camm_aes_scale (s_AES) the secondary amplitude.
gfn1_rs_cli --param param_gfn1-xtb.txt --multipole --multipole-model camm_on_mdftb2 --camm-damp 1.0 examples/water48.xyz

# experimental self-consistent D4 single point / gradient (non-PBC only)
gfn1_rs_cli --param param_gfn1-xtb.txt --experimental-d4 --gradient examples/water.xyz

# range-separated Fock exchange single point (experimental, non-PBC); --onsite-exchange adds
# the exact one-center (OFX) term, --scf-trah forces the TRAH second-order SCF
gfn1_rs_cli --param param_gfn1-xtb.txt --lr-exchange --onsite-exchange --charges examples/water.xyz

# charged / open-shell single point
gfn1_rs_cli --param param_gfn1-xtb.txt --charge 1 --multiplicity 2 examples/h2.xyz

# spin-polarized GFN1 ("spGFN1"): collinear spin term for open-shell systems (non-PBC, energy +
# analytic forces). Off by default; a closed-shell singlet stays byte-identical to GFN1.
gfn1_rs_cli --param param_gfn1-xtb.txt --multiplicity 3 --spinpol --gradient examples/h2.xyz

# TD-GFN1 (TDA) excited states; --tda-grad adds the excited-state gradient (analytic by default)
gfn1_rs_cli --param param_gfn1-xtb.txt --tda examples/water.xyz

# single point in a uniform external electric field (atomic units, Ex Ey Ez)
gfn1_rs_cli --param param_gfn1-xtb.txt --field 0.0 0.01 0.0 --charges examples/water.xyz

# static polarizability, IR, and Raman (each also prints the raw derivative tensors)
gfn1_rs_cli --param param_gfn1-xtb.txt --polarizability examples/water.xyz
gfn1_rs_cli --param param_gfn1-xtb.txt --ir examples/water.xyz
gfn1_rs_cli --param param_gfn1-xtb.txt --raman examples/water.xyz

# NMR nuclear magnetic shielding tensor of atom 0 (ppm; isotropic + 3x3)
gfn1_rs_cli --param param_gfn1-xtb.txt --nmr 0 examples/water.xyz

# finite-difference parameter derivatives (dE/dp, add --param-forces for dF/dp)
gfn1_rs_cli --param param_gfn1-xtb.txt --param-deriv \
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
print(atoms.calc.results["native_energy_terms_hartree"])

# ASE-driven optimization, or the Rust-native L-BFGS:
LBFGS(atoms).run(fmax=0.05)
atoms.calc.optimize_native(atoms, gradient_tolerance=1e-4)
```

ASE units throughout: Angstrom, eV, eV/Angstrom. The low-level
`gfn1_rs.Gfn1NativeCalculator` exposes the same controls without ASE.
Set `experimental_d4=True` on either interface to use the non-PBC D4 path.
Use `d4_s9=0.0` / `--d4-s9 0` or `--no-d4-atm` to disable the D4 three-body
term explicitly.

## Validation

Tests load the external parameter file from `GFN1_XTB_PARAM` and skip cleanly when
it is unset:

```bash
GFN1_XTB_PARAM=/path/to/param_gfn1-xtb.txt cargo test
GFN1_XTB_PARAM=/path/to/param_gfn1-xtb.txt cargo test -- --ignored   # FD gradient probe
GFN1_XTB_PARAM=/path/to/param_gfn1-xtb.txt maturin develop --release --features python \
  && python -m pytest -q
```

Tests cover finite-difference gradient / Hessian checks
(`tests/gradient_fd_probe.rs`, `tests/hessian.rs`), the periodic path
(`tests/pbc_integration.rs`), the optimizer (`tests/optimizer.rs`), and an opt-in
external `tblite` parity suite (`tests/tblite_parity.rs`, enabled by setting
`GFN1_TBLITE_BIN`; `tblite` is not a Cargo dependency). `GFN1_FD_TIGHT=1` tightens
the finite-difference probe and `GFN1_PROFILE=1` emits per-scope native timings.

## Limitations

- D3 ATM (three-body Axilrod–Teller–Muto) **higher derivatives**. The ATM term is now
  implemented for both **non-PBC** (energy + analytic forces) and **periodic**
  (energy + analytic forces + analytic stress) systems, all FD-verified, so a custom
  param with `s9 != 0` works for molecules and crystals. Only the ATM **Hessian** and
  **third derivative** (`s9 != 0`) remain unimplemented and error with a clear message.
  Official GFN1-xTB has `s9 = 0`, so stock GFN1 is unaffected.
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
