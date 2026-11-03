# Public API organization

This is the entry map for v0.7.0. Existing public names and argument spellings
remain compatible; the subsystem guides linked below contain complete recipes.
Install the local distribution wheel, or follow the
[source build instructions](../README.md#build) and
[Python reference](python-api.md). The examples below assume that installation.

## Four entry points

| Interface | Public entry | Input/result convention |
| --- | --- | --- |
| Rust | Cargo package `gfn1-rs`, library `gfn1_rs`; typed functions/options | `PeriodicSystem` in internal bohr; typed results/options |
| Python native | `from gfn1_rs import Gfn1NativeCalculator` | Numbers and coordinates/cell arrays; native result objects/dictionaries |
| ASE | `from gfn1_rs import GFN1RSCalculator` | Attach to `ase.Atoms`; ordinary ASE methods or explicit `get_*` methods |
| CLI | `gfn1-rs`, `gfn1-rs-train` | Cargo binaries and pip console commands share the Rust drivers |

`gfn1_rs.native` is the no-ASE Python surface; ASE is an optional dependency.
Cell arrays contain lattice vectors **as rows**. Cartesian derivative indices
are `3 * atom + axis`. Python state and nucleus indices are zero based.

An installed native calculator can start with a molecular energy and gradient:

```python
from gfn1_rs import Gfn1NativeCalculator

numbers = [8, 1, 1]
positions = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]
calc = Gfn1NativeCalculator(charge=0, electronic_temperature=300.0,
    threads=2, memory_mb=1024)
result = calc.calculate(numbers, positions, compute_gradient=True)
print(result.energy_hartree, result.gradient_hartree_per_bohr)
```

The same molecule through ASE:

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

numbers = [8, 1, 1]
positions = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]
atoms = Atoms(numbers=numbers, positions=positions)
atoms.calc = GFN1RSCalculator(charge=0, electronic_temperature=300.0,
    threads=2, memory_mb=1024)
print(atoms.get_potential_energy(), atoms.get_forces())  # eV, eV/angstrom
```

The typed Rust equivalent:

```rust
use gfn1_rs::{run_electronic, ElectronicOptions, Gfn1Parameters, PeriodicSystem};

fn main() -> gfn1_rs::Result<()> {
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater; angstrom\nO 0 0 0\nH .757 .586 0\nH -.757 .586 0\n",
        0.0, false)?;
    let options = ElectronicOptions {
        charge: Some(0.0), electronic_temperature: 300.0,
        ..Default::default()
    };
    let result = run_electronic(&system, &Gfn1Parameters::resolve(None)?, options)?;
    println!("{} Hartree", result.total_free);
    Ok(())
}
```

## Choose a capability group

Names in the Rust column are module/function starting points, not a promise
that every option combination is implemented. The term/response guards reject
unsupported combinations instead of silently dropping their derivatives.

| Need | Rust starting point | Python native starting point | ASE starting point | Detailed guide |
| --- | --- | --- | --- | --- | 
| Molecular scalar energy/gradient/Hessian and FC3/FC4 | `electronic`, `gradient`, `hessian`, `third_derivative`, `fourth_derivative` | `calculate`, `hessian`, `third_derivative`, `fourth_derivative` | ordinary energy/forces, `get_hessian`, `get_third_derivative`, `get_fourth_derivative_directional` | [derivatives](derivatives.md), [finite temperature](finite-temperature.md) |
| Periodic scalar/ML ground state and DFPT | `pbc::scf`, `pbc::gradient`, `pbc::stress`, `pbc::hessian` | `calculate_periodic`, `hessian_periodic` | ordinary periodic energy/forces/stress, `get_hessian` | [PBC](pbc.md), [SOC/DFPT](soc-dfpt.md) |
| Cell/ion relaxation, EOS, elastic/compliance/moduli/acoustics | `pbc::mechanical`, `pbc::elastic_response` | `relax_periodic`, `equation_of_state`, `elastic_constants`, `elastic_properties` | `get_relax_periodic`, `get_equation_of_state`, `get_elastic_constants`, `get_elastic_properties` | [mechanical](mechanical-properties.md) |
| Born charges, proper piezoelectricity, dielectric, IR/LO–TO | `pbc::polarization::response`, `pbc::electromechanical` | `born_effective_charges`, `dielectric_tensor`, `electromechanical_response`, `lo_to_modes` | `get_born_effective_charges`, `get_dielectric_tensor`, `get_electromechanical_response`, `get_lo_to_modes` | [electromechanical](electromechanical.md) |
| Phonons/DOS/thermal/QHA and bands/electronic DOS | `pbc::phonon`, `pbc::spectra` | `phonon_mesh`, `harmonic_thermodynamics`, `quasiharmonic_zero_pressure`, `band_structure`, `electronic_dos` | `get_phonon_mesh`, `get_harmonic_thermodynamics`, `get_quasiharmonic_zero_pressure`, `get_band_structure`, `get_electronic_dos` | [spectra and thermodynamics](phonon-band-properties.md) |
| Mode Grüneisen parameters | `pbc::gruneisen` | `gruneisen` | `get_gruneisen` | [PBC](pbc.md) |
| Explicit-parameter SOC | `soc`, `pbc::soc` | `soc_calculate`, `soc_calculate_periodic` | `get_soc` | [SOC/DFPT](soc-dfpt.md) |
| FMO and divide-and-conquer | `fragment::fmo_xtb`, `fragment::divide_and_conquer` | `fmo_xtb`, `divide_and_conquer` | `get_fmo_xtb`, `get_divide_and_conquer` | [fragments](fragments.md) |
| TD-GFN1 TDA or independent native sTDA-xTB | `td`, `stda_xtb` | `tda`, `tda_gradient`, `tda_hessian`; `stda_xtb`, `stda_xtb_gradient`, `stda_xtb_hessian` | corresponding `get_tda*` and `get_stda_xtb*` methods | [TD-GFN1](td.md), [sTDA](stda-xtb.md), [excited Hessians](excited-hessians.md) |
| Molecular full-TD RPA | `solve_rpa`, `solve_rpa_gradient_analytic`, `solve_rpa_hessian_analytic` and manifold variants | `rpa`, `rpa_gradient`, `rpa_hessian` | `get_rpa`, `get_rpa_gradient`, `get_rpa_hessian` | [TD-GFN1](td.md) |
| Closed-shell magnetics | `magnetic`, `nmr` | `magnetic_energy`, `magnetizability`, `cotton_mouton`, `nmr_shielding` | `get_magnetic_energy`, `get_magnetizability`, `get_cotton_mouton`, `get_nmr_shielding` | [native/ASE reference](python-api.md), [scope](scope.md) |
| Molecular PES-only VPT2 | `vpt2::molecular_vpt2_pes` | `vpt2_pes` | `get_vpt2_pes` | [derivatives](derivatives.md), [limitations](limitations.md) |
| User ML weights, training and property fine-tuning | `ml`, `training`, `property_training` | constructor `ml_model_path`, `fine_tune_properties` | constructor `ml_model_path`, `fine_tune_properties` | [ML training](ml-training.md), [property fitting](ml-property-training.md) |
| Optional solvent/multipole/exchange/+U/spin models | `ElectronicOptions` and their model modules | calculator constructor options | calculator constructor options | [parameters](parameters.md), [CAMM preset](sigma_hole_preset.md), [scope](scope.md) |

The shared crystal CLI has a separate, exact
[21-operation Rust/native/ASE/CLI coverage table](crystal-cli.md). It supports
unit-tagged JSON requests and earlier-result references:

```sh
gfn1-rs molecule.xyz --gradient --etemp 300 --threads 2 --memory-mb 1024
gfn1-rs properties --list
gfn1-rs properties --request examples/crystal-response-request.json --output response.json --threads 2 --memory-mb 2048
gfn1-rs-train --help
```

The legacy calculation CLI selects a primary property action. Use separate
gradient/stress invocations, or `properties` operation `ground_state` for their
shared-SCC combined result. The CLI catalog/help, rather than a library method
name, determines which CLI routes exist. Training is the separate
`gfn1-rs-train` workflow; [its guide](ml-property-training.md) gives complete
dataset, user-weight and save/reload recipes.

## Compatibility and state changes

| Existing spelling | Correct use |
| --- | --- |
| `kgrid=(nx,ny,nz)` | Native `calculate_periodic`, `hessian_periodic`, explicit SOC periodic calls and the older analytic k-point FC3 wrappers; also the ASE calculator's default electronic mesh |
| `kmesh=(nx,ny,nz)` | New crystal property helpers and periodic TDA; ASE crystal helpers inherit `kgrid` when this per-call override is omitted |
| `gamma_centered` | Set explicitly where exposed when matching calculations. Older ground/Hessian and mechanical helpers default to the legacy uncentred choice; Berry/Born/dielectric helpers default to centred complete meshes |
| `relax_cell=True` | Native `relax_periodic` and ASE `get_relax_periodic`; the keyword is not `variable_cell` |

Do not rename `kgrid` globally to `kmesh`; use the documented signature of the
method being called. For Gamma, the compatible older argument is `kgrid=None`
or `(1,1,1)`. Some older k-point derivative wrappers retain a fixed legacy mesh
construction rather than exposing `gamma_centered`; their detailed signatures
in [the Python reference](python-api.md) remain authoritative.
Phonon supercell Gamma folds a primitive electronic grid set by
the repetitions, so it is not the same calculation as primitive-cell Gamma.

Native relaxation returns the new geometry. ASE `get_relax_periodic` defaults
to `update_atoms=False`; set it true to update the attached object only after
convergence. A later native calculation must receive the returned geometry
explicitly. `fine_tune_properties` saves new weights and retains the inference
calculator's original weights until the user explicitly loads the saved model.

## Units, conditions and controls

- Scalar calculators default to **300 K electronic temperature** and neutral
  charge. Set `charge` and, when needed, `multiplicity` explicitly; Rust uses
  `ElectronicOptions::charge` and `spin_multiplicity`. Excitation `spin` or
  `triplet` selects a separate response channel, not a substitute for the
  ground-state electron count or spin model.
- Energies/forces/stresses differentiate the canonical electronic **free
  energy**, not the bare internal energy, at nonzero electronic temperature.
  Vibrational lattice temperatures are separate explicit arguments.
- Rust geometry is internally bohr. Native positions/cell default to angstrom
  input (`unit="bohr"` selects bohr), while output is atomic units unless a key
  states otherwise. For example, native `CalculationResult` also provides
  `energy_ev` and `forces_ev_per_angstrom`; optional fields may be `None` when
  that result surface does not supply them. Native periodic forces use the
  explicitly tagged force field, not an assumed populated raw-gradient field.
- ASE uses eV/angstrom/e with `ase.units` conversions and retains explicit raw
  unit-tagged fields. Native CODATA constants and ASE constants differ slightly;
  do not mix them when checking strict numerical equality. Wavenumbers are
  `cm^-1`, NMR shielding is ppm and relative dielectric tensors are dimensionless.
- [Parameters](parameters.md) resolve explicit spec/path, environment, then
  builtin. `param_source()` reports the active source. `ml_model_path` accepts
  bundled or user weights; `ml_model_info()` reports source/canonical hashes,
  provenance and license. This identity is also available in model logs/results.

| Control | Rust | Native/ASE | CLI |
| --- | --- | --- | --- |
| Workers/workspace | `resources::ResourceLimits` / `ResourceContext` | `threads`, `memory_mb` | `--threads`, `--memory-mb` |
| SCC convergence | `ElectronicOptions` | `max_scc`, `energy_tolerance`, `charge_tolerance`, `mixing`, `scc_accelerator` | legacy flags in help; complete controls in the `properties` JSON `electronic` object |
| CP/DFPT response residuals | `resources::CpSettings` | `cp_tolerance`, `cp_relative_tolerance`, `cp_max_iter` | `--cp-tolerance`, `--cp-relative-tolerance`, `--cp-max-iter` |

Workspace budgets are checked allocation estimates, not process RSS caps.
`auto` SCC uses Broyden with an analytic Newton fallback for supported scalar/ML
PBC without changing the requested temperature/tolerances; the molecular driver
retains its existing safeguards. Joint multipole/unequal-spin variables do not
become charge-only Newton variables implicitly. CP controls and SCC controls
govern different equations; direct CP solves verify their own residuals.
See [resources](resources.md), [CP settings](cp-settings.md) and
[periodic SCC](pbc-scc.md) for exact defaults, scopes and diagnostics.

## Calculation CLI properties

The Cargo executable and the pip console script share the same flags. Run
`gfn1-rs --help` for the calculation flags and `gfn1-rs properties --list` for
structured crystal workflows. The table covers physical observables and
derivatives. Rust and Python also expose lower-level matrix and response tools.

| Property | Calculation CLI |
| --- | --- |
| Ground-state gradients, Hessians and force constants | `--gradient`, `--hessian --matrices`, `--cubic`, `--quartic`. FC3 supports molecules and admitted periodic references. FC4 uses the molecular dense backend. |
| Periodic analytic Hessian and SOC | `--dfpt [--kmesh nx,ny,nz]` prints Cartesian components. Use `--soc-zeta-file PATH` for explicit SOC, including periodic SOC DFPT and commensurate phonons. |
| Embedded ML-xTB weights | `--list-ml-models` and `--ml-xtb builtin:water-clusters`. The singular `builtin:water-cluster` resolves to the same model and identity. |
| TDA, sTDA and RPA | `--tda`, `--tda-grad`, `--tda-hessian`, `--stda-xtb`, `--stda-grad`, `--stda-hessian`, `--rpa`, `--rpa-grad`, `--rpa-hessian`. |
| Periodic optical full TD | `--td-dfpt --kmesh nx,ny,nz`, with optional `--soc-zeta-file PATH`. |
| Static magnetic tensors | `--magnetizability`, `--magnetic-polarizability`, `--cotton-mouton`, `--mcd`. M1 accepts `--m1-basis NAME|FILE`. |
| Magnetic forces and NMR | `--bfield Bx By Bz --gradient` and `--nmr ATOM`. |
| ECD and electronic optical rotation | `--ecd` or `--optical-rotation E` with photon energy in Hartree. `--n-states` selects the root count. |
| Berry polarization, Born charges, proper piezoelectricity and dielectric response | `--berry-polarization`, `--piezoelectric` / `--born-charges`, `--dielectric`, `--electromechanical`. Use the [crystal JSON workflow](crystal-cli.md) for custom relaxation, stability tolerances and LO–TO results. |
| Harmonic modes, IR/Raman and phonons | `--frequencies [--matrices]`, `--ir`, `--raman`, `--phonon-q qx,qy,qz --phonon-supercell nx,ny,nz`. |
| Grüneisen parameters and volumetric Hessian response | `--gruneisen`, `--gruneisen-second-order`, `--strain-hessian-derivative H` with dimensionless relative volume step. |
| Fragment free energies, forces and periodic stress | `--fmo-fragments PARTITION.json` and `--dc-fragments PARTITION.json`. |

The existing `--mcd` backend evaluates the static `d alpha_ij / dB_k` tensor.
At zero field it vanishes for the closed-shell model. Cotton–Mouton evaluates
`d² alpha_ij / dB_k²` at zero field. Both tensors use atomic units and order
`[k][i][j]`. The CLI accepts `--field-step` and `--magnetic-field-step` in
atomic units. The current magnetic response does not include a
frequency-dependent optical MCD spectrum.

For a periodic insulating cell, the piezoelectric command returns the proper
clamped-ion tensor in atomic units and C/m² alongside the Born charges. These
response flags use an unfolded Γ-centered mesh of the requested size. The
coupled electromechanical result includes relaxed-ion quantities when the
reference meets the backend's stationarity and stability requirements. The
`properties` operations `born_effective_charges` and
`electromechanical_response` expose the same backends.

A fragment partition starts with zero-based atom lists. A two-fragment FMO3
request can use `{"fragments":[[0,1],[2,3]],"fragment_charges":[0,0],"order":3}`.
FMO also accepts `cuts` objects with `bda`, `baa`, and optional three-integer
`image`. The divide-and-conquer request accepts `buffers` in place of
`cuts` and `order`. Both accept `max_ao` and `compute_derivatives`, which
defaults to true.

## Supported routes and research boundaries

The APIs above are implemented and tested on their documented domains. That
does not establish experimental accuracy for arbitrary crystals or every
combination of optional Hamiltonians. In particular:

- Relaxed elastic/e/d/static-dielectric results require the documented ionic
  stationarity, curvature and stress conditions. A null response and its
  reason remain null; it is not replaced with a clamped or zero tensor.
- Berry/Born/dielectric and LO–TO need the supported insulating, effectively
  integer-occupation reference at its actual temperature. Mechanical finite-T
  response does not imply a general mixed-state metallic Berry polarization.
- Analytic and explicitly seminumerical derivative names remain distinct.
  Molecular scalar FC3/FC4 includes canonical finite-T degeneracy response;
  supported structural-ML derivatives stop at their stated order. Periodic
  ML analytic FC3, ML FC4 and unimplemented composite chains are not inferred.
- EOS and QHA sample volumes and fit a stated model. Commensurate phonons do
  not supply arbitrary-q interpolation; mode and k-mesh convergence must be checked.
- TD-GFN1 TDA and native Rust sTDA-xTB are separate models. Local analytic
  excitation gradients/Hessians do not certify a smooth global sTDA PES.
  Original-sheet ethylene force-zero saddles and the unresolved smooth minimum
  are distinguished in [the PES guide](stda-pes.md) and
  [local topology](stda-pes-local-topology.md).
- SOC needs explicit parameters and provenance. Closed-shell static magnetics
  does not supply frequency-dependent optical MCD. PES-only VPT2 omits the
  additional Watson/Coriolis/intensity terms. The exact boundaries remain in
  [scope](scope.md) and [limitations](limitations.md).

Actual crystal executions and sampling/stability limits are documented in
[DFPT results](crystal-dfpt-results.md),
[coupled crystal response](crystal-properties-results.md) and
[the sourced-CIF property survey](real-crystal-property-survey.md).
The [documentation index](README.md) links the remaining model studies.
User calculation inputs live in `examples/`; reusable tests/physical audits
live in `tests/`. Retained measured data is evidence, not an extra public API.
One-off recording, packaging and capture helpers are workspace material and
are not required to use these interfaces.
