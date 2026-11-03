# SOC and density-functional perturbation theory APIs

The native Python API accepts coordinates in angstrom by default (`unit="bohr"`
selects bohr) and reports energies and response tensors in atomic units. The ASE
adapter converts its unsuffixed energy, forces and stress to ASE units. Cell
vectors are rows. Cartesian Hessian indices are `3 * atom + axis`.

## Spin-orbit coupling

### Rust

The following molecular example uses explicit **zero** SOC to demonstrate the
API and its scalar limit. Replace the input by sourced nonzero parameters for a
physical SOC calculation; zero is not a recommended oxygen constant.

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use gfn1_rs::soc::{SocParameters, soc_analytic_gradient, soc_static_polarizability};

fn main() -> gfn1_rs::Result<()> {
let params = Gfn1Parameters::builtin()?;
let molecule = PeriodicSystem::from_xyz_str(
    "3\nwater; coordinates in angstrom\nO 0 0 0\nH .958 0 0\nH -.25 .925 .03\n",
    0.0, false)?;
let options = ElectronicOptions {
    electronic_temperature: 0.0,
    energy_tolerance: 1e-12, charge_tolerance: 1e-10,
    ..Default::default()
};
let soc = SocParameters::from_zeta_str(
    "units hartree\nprovenance zero-SOC scalar-limit regression\n8 1 0.0\n")?;
let (state, gradient) = soc_analytic_gradient(&molecule, &params, &options, &soc)?;
let response = soc_static_polarizability(&molecule, &params, &state)?;
println!("F={} Hartree, gradient={:?} Hartree/bohr", state.total_free, gradient);
println!("alpha={:?}, residual={}", response.tensor, response.max_normalized_residual);
Ok(())
}
```

For a system with `lattice: Some(...)`, periodic physical-spinor entry points are
`pbc::pbc_soc_analytic_gradient(&system, &params, &options, &pbc, &soc)`,
`pbc::pbc_soc_stress(...)`, and `pbc::pbc_soc_hessian(...)`, all taking the same
five arguments. `pbc` is `pbc::PbcOptions`; select `pbc.kmesh` explicitly. The
Hessian result contains `.hessian`, `.raw_max_asymmetry` and
`.max_response_residual`. Internal Rust coordinates and cell vectors are bohr.

### Python native

SOC uses complex physical spinors with occupation limit one and a self-consistent
charge response. GFN1 does not provide a fitted SOC parameter set: callers must
provide each `(atomic_number, angular_momentum_l, zeta_hartree)` and a nonempty
provenance string. A citation, source table identifier, sign convention and any
unit conversion belong in that provenance. Duplicate element/shell entries are
errors. Missing fitted constants must not be replaced by guessed values.

```python
import json
from gfn1_rs import Gfn1NativeCalculator

calc = Gfn1NativeCalculator(electronic_temperature=0.0,
                           energy_tolerance=1e-12, charge_tolerance=1e-10)
nums = [8, 1, 1]
pos = [[0., 0., 0.], [.958, 0., 0.], [-.25, .925, .03]]
with open("soc-parameters.json", encoding="utf-8") as stream:
    parameters = json.load(stream)
result = calc.soc_calculate(
    nums, pos, zeta_hartree=parameters["zeta_hartree"],
    parameter_provenance=parameters["provenance"],
    compute_forces=True, compute_response=True)
print(result["energy_hartree"], result["gradient_hartree_per_bohr"])
print(result["polarizability_au"], result["response_normalized_residual"])
```

An executable zero-SOC regression input is
`{"zeta_hartree": [[8, 1, 0.0]], "provenance": "zero-SOC regression input"}`.
It tests the scalar limit; it is not an element parameter recommendation.
`soc_energy_hartree` is the SOC contribution, and `occupations` and
`orbital_energies_hartree` describe physical spinor states.

For a periodic calculation, use the same explicit parameters:

```python
cell = [[12., 0., 0.], [0., 12., 0.], [0., 0., 12.]]
result = calc.soc_calculate_periodic(
    nums, pos, cell, parameters["zeta_hartree"], parameters["provenance"],
    kgrid=(2, 1, 1), compute_forces=True, compute_stress=True,
    compute_hessian=True)
gradient = result["gradient_hartree_per_bohr"]
stress = result["stress_hartree_per_bohr3"]
hessian = result["hessian_hartree_per_bohr2"]
print(result["max_response_residual"])
```

The Hessian differentiates nuclear positions at fixed cell; stress differentiates
the cell at fixed fractional coordinates. Stress is positive for increasing
energy under tensile strain. `raw_hessian_asymmetry_hartree_per_bohr2` and the
response residual permit checking the raw calculation. SOC nuclear FC3/FC4 and
combined noncollinear spin/SOC are outside the implemented response scope.

### ASE

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

atoms = Atoms("OH2", positions=[[0, 0, 0], [.958, 0, 0], [-.25, .925, .03]])
atoms.calc = GFN1RSCalculator(electronic_temperature=0.0)
# Use the same sourced parameters dictionary as in the native example above.
soc = atoms.calc.get_soc(parameters["zeta_hartree"], parameters["provenance"],
                         atoms=atoms, compute_forces=True, compute_response=True)
print(soc["energy"], soc["forces"])

crystal = atoms.copy()
crystal.set_cell([12., 12., 12.])
crystal.pbc = True
crystal.calc = GFN1RSCalculator(electronic_temperature=0.0, kgrid=(2, 1, 1))
soc_periodic = crystal.calc.get_soc(
    parameters["zeta_hartree"], parameters["provenance"], atoms=crystal,
    compute_forces=True, compute_stress=True, compute_hessian=True)
print(soc_periodic["stress"], soc_periodic["hessian"])
```

`GFN1RSCalculator.get_soc(...)` accepts the same explicit parameter
arguments. Its unsuffixed `energy`, `forces`, `stress`, and `hessian` entries use
eV, eV/angstrom, eV/angstrom cubed, and eV/angstrom squared; suffixed native
entries retain atomic units.

## Analytic nuclear response (DFPT / CP-SCC)

Actual 300 K calculations for diamond C/Si, SiC, LiF, NaCl and MgO, including
all Hessian component checks and imaginary modes, are reported in
[crystal-dfpt-results.md](crystal-dfpt-results.md).

The induced shell charges solve `(I - chi0 K) dq = dq_bare`. Molecular response
reuses a factored charge-space dielectric. Periodic response uses complex Bloch
states at the specified k mesh. These are analytic density responses to nuclear
perturbations, rather than displaced-geometry force calculations.

### Rust

For the molecular `molecule`, `params` and `options` constructed above:

```rust
use gfn1_rs::{analytic_hessian, AnalyticHessianOptions};
let response = analytic_hessian(&molecule, &params, AnalyticHessianOptions {
    electronic_options: options.clone(), ..Default::default()
})?;
let hessian = response.hessian; // 3N x 3N, Hartree/bohr^2
```

For periodic `system` with a lattice in bohr:

```rust
use gfn1_rs::pbc::{PbcOptions, KMesh, pbc_gamma_hessian, pbc_kpoint_hessian};
let gamma = PbcOptions::default();
let h_gamma = pbc_gamma_hessian(&system, &params, &options, &gamma)?.hessian;
let kmesh = PbcOptions { kmesh: KMesh::monkhorst_pack([2, 1, 1]), ..gamma };
let h_k = pbc_kpoint_hessian(&system, &params, &options, &kmesh)?.hessian;

// For the cubic-crystal report, explicitly use a Gamma-centred electronic mesh:
let cubic_mesh = PbcOptions { kmesh: KMesh {
    size: [2, 2, 2], gamma_centered: true, fold_time_reversal: true,
}, ..Default::default() };
let h_cubic = pbc_kpoint_hessian(&system, &params, &options, &cubic_mesh)?.hessian;

use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
let phonons = AnalyticSupercellPhonons::build(
    &system, &params, &options, &gamma, &[15.999, 1.008, 1.008],
    PhononSupercellOptions { repetitions: [2, 1, 1], ..Default::default() })?;
let zone_edge = phonons.at_q([0.5, 0.0, 0.0])?;
println!("{:?} cm^-1", zone_edge.wavenumbers_cm1);
```

The mass array must match the primitive atom order; the array above is for
O,H,H only. `build_soc` adds a final `&soc` argument for SOC phonons.

### Python native

```python
hmol = calc.hessian(nums, pos)
hgamma = calc.hessian_periodic(nums, pos, cell, kgrid=(1, 1, 1))
hk = calc.hessian_periodic(nums, pos, cell, kgrid=(2, 1, 1))
# Explicit sampling used by the cubic-crystal report:
hk_centered = calc.hessian_periodic(nums, pos, cell, kgrid=(2, 2, 2), gamma_centered=True)
```

Each returns a `3N` by `3N` list in Hartree/bohr squared. Electronic temperature
is set on the calculator; fractional occupations differentiate the electronic
free energy with the electron-number constraint retained. A converged SCC and
response are required. Test raw force differences before interpreting small
phonon eigenvalues at an unoptimized structure.

Finite-q modes use commensurate analytic supercell response:

```python
modes = calc.phonons_commensurate(
    nums, pos, cell, repetitions=(2, 1, 1),
    qpoints=[(0., 0., 0.), (.5, 0., 0.)])
```

Q points are fractional reciprocal coordinates. Each component multiplied by
its supercell repetition must be integral. This API forms an analytic supercell
Hessian; it is not a primitive-cell arbitrary-q Sternheimer implementation.
`memory_budget_bytes` and `max_atoms` bound allocation before work starts.
Explicit SOC parameters can also be passed through `zeta_hartree` and
`parameter_provenance`. The ASE counterpart is `get_phonons_commensurate`.

### ASE

```python
# Molecular Cartesian Hessian, eV/angstrom^2:
h_molecule = atoms.calc.get_hessian(atoms=atoms)
# Periodic fixed-cell Hessian, same units; k mesh comes from the calculator:
h_periodic = crystal.calc.get_hessian(atoms=crystal)
# Optional per-property centring override; the calculator supplies kgrid:
h_centered = crystal.calc.get_hessian(atoms=crystal, gamma_centered=True)
# Commensurate q modes use supercell Gamma sampling:
crystal.calc = GFN1RSCalculator(electronic_temperature=0.0, kgrid=(1, 1, 1))
modes = crystal.calc.get_phonons_commensurate(
    repetitions=(2, 1, 1), qpoints=[(0., 0., 0.), (.5, 0., 0.)], atoms=crystal)
```

The ASE adapter supplies isotope masses from `Atoms.get_masses()`. Mode
wavenumbers are cm^-1; Hessian entries are eV/angstrom^2. These property methods
return the response directly and do not populate the ordinary `Atoms` force
cache with a different Hamiltonian.

The optional `gamma_centered` Hessian keyword defaults to `False` for backwards
compatibility. Its ASE override does not change ordinary ASE energy/force
sampling. To check a centred-mesh Hessian against matching forces, use native
`calculate_periodic(..., kgrid=..., gamma_centered=True, compute_gradient=True)`.
Electronic k sampling and phonon q points describe different quantities.

Rust entry points are `analytic_hessian`, `pbc_gamma_hessian`,
`pbc_kpoint_hessian`, `soc::run_soc_scc`, `soc::soc_analytic_gradient`,
`soc::soc_static_polarizability`, `pbc::pbc_soc_hessian`, and
`pbc::phonon::AnalyticSupercellPhonons::{build, build_soc, at_q}`; see their
argument types in the Rust API.

## SOC data redistribution and conventions

SOC data are not categorically excluded by licensing. The DFTB parameter
project distributes the Jha–Heine constants with an explicit **CC BY 4.0**
notice in its [SOC parameter file](https://www.dftb.org/_downloads/4bcfc14e194bcbbaa8154eb7b76430d2/soc_parameters.txt).
That source can be redistributed with attribution, the license and any changes
identified, according to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Its paper is Jha and Heine, JCTC **18**, 4472–4481 (2022),
[doi:10.1021/acs.jctc.2c00376](https://doi.org/10.1021/acs.jctc.2c00376).
This is a separate parameter-data license from any DFTB program's source license.

License permission does not establish model compatibility. This API uses the
explicit convention `H_atom = zeta * L.S` in Hartree; an isolated l shell has
eigenvalues `zeta*l/2` and `-zeta*(l+1)/2`. Published renormalized constants,
different radial shells, or a different operator normalization require a
documented conversion and atomic-splitting check before use. The public table
also contains mismatches between some element symbols and printed atomic
numbers, so importing rows solely by that number is unsafe. No unverified
conversion is silently installed as a default GFN1 SOC fit. The current API
therefore requires caller-supplied constants and provenance, as shown above.

## References

- GFN1-xTB: Grimme, Bannwarth and Shushkov, JCTC 13, 1989–2009,
  [doi:10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).
- Density-functional perturbation theory: Baroni, de Gironcoli, Dal Corso and
  Giannozzi, Reviews of Modern Physics 73, 515,
  [doi:10.1103/RevModPhys.73.515](https://doi.org/10.1103/RevModPhys.73.515).
- Model implementation and executable checks: `src/soc.rs`, `src/pbc/soc.rs`,
  `src/pbc/soc_response.rs`, `src/response/charge_space.rs`, `tests/soc.rs`,
  `tests/pbc_soc.rs`, and `tests/pbc_integration.rs`. Element-specific SOC
  constants remain caller-supplied and must carry their own references.
