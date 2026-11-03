# Crystal properties through Rust, native Python, ASE and the CLI

The new crystal calculations have one Rust implementation. Native Python and
ASE call it in process; the `gfn1-rs properties` command now exposes the same
backends. The Cargo executable and the pip-installed command share this Rust
dispatcher. No external electronic-structure program or official sTDA driver
is invoked.

```sh
gfn1-rs properties --list
gfn1-rs properties --request examples/crystal-response-request.json --output response.json --threads 2 --memory-mb 2048
gfn1-rs properties --request examples/crystal-phonon-request.json --output phonons.json --threads 2 --memory-mb 2048
```

Run these examples from the source root. A Cargo checkout can equivalently use
`cargo run --release --bin gfn1-rs -- properties ...`. The installed pip command
uses the same request files and arguments. The supplied examples are declared
ideal NaCl/diamond prototypes, not newly acquired experimental CIFs. Actual
crystallographic structures and their model applicability are discussed in
[the crystal results](crystal-properties-results.md).

## Interface coverage

The CLI `operation` strings below select the listed typed Rust API; module
names in the Rust column are relative to `gfn1_rs::pbc`. Every
geometry calculation supports the active structural ML Hamiltonian where its
physical backend does; unsupported composite models are not inferred from
separate tests. Rust and native output use atomic units with explicitly named
exceptions. ASE converts its convenient unsuffixed fields and preserves raw
unit-tagged fields.

| Quantity | Rust | Native method | ASE method | CLI operation |
| --- | --- | --- | --- | --- |
| Energy, forces, stress, pressure | `run_pbc_scc`, gradient/stress from the same SCC | `calculate_periodic` | ordinary energy/forces/stress | `ground_state` |
| Nuclear Hessian | `pbc_gamma_hessian` / `pbc_kpoint_hessian` | `hessian_periodic` | `get_hessian` on periodic atoms | `hessian` |
| Ionic/cell relaxation | `mechanical::pbc_relax` | `relax_periodic` | `get_relax_periodic` | `relax_periodic` |
| EOS sampling and fit | `mechanical::pbc_equation_of_state` | `equation_of_state` | `get_equation_of_state` | `equation_of_state` |
| Fit precomputed EOS points | `mechanical::fit_birch_murnaghan` | `fit_equation_of_state` | `get_fit_equation_of_state` | `fit_equation_of_state` |
| Analytic elastic/internal-strain tensors | `elastic_response::pbc_elastic_response` | `elastic_constants` | `get_elastic_constants` | `elastic_constants` |
| Compliance, bulk/shear/Young/Poisson, sound | `mechanical::elastic_properties` | `elastic_properties` | `get_elastic_properties` | `elastic_properties` |
| Directional elastic/acoustic properties | `mechanical::directional_elastic_response` | `directional_elastic_properties` | `get_directional_elastic_properties` | `directional_elastic_properties` |
| Berry polarization | `pbc_berry_polarization` | `berry_polarization` | `get_berry_polarization` | `berry_polarization` |
| Born charges/proper clamped piezoelectricity | `polarization::response::pbc_polarization_derivatives` | `born_effective_charges` | `get_born_effective_charges` | `born_effective_charges` |
| Electronic relative dielectric | `polarization::response::pbc_electronic_dielectric` | `dielectric_tensor` | `get_dielectric_tensor` | `dielectric_tensor` |
| Coupled elastic/Born/e/d/static dielectric/IR | `electromechanical::pbc_electromechanical_response` | `electromechanical_response` | `get_electromechanical_response` | `electromechanical_response` |
| Mode effective charges and IR oscillator tensors | `electromechanical::polar_mode_properties` | `polar_mode_properties` | `get_polar_mode_properties` | `polar_mode_properties` |
| Directional Gamma LO–TO | `electromechanical::lo_to_modes` | `lo_to_modes` | `get_lo_to_modes` | `lo_to_modes` |
| Commensurate phonon mesh | `spectra::sample_phonon_mesh` with `AnalyticSupercellPhonons` | `phonon_mesh` | `get_phonon_mesh` | `phonon_mesh` |
| Chosen commensurate q points | `spectra::sample_phonon_qpoints` | `phonons_commensurate` | `get_phonons_commensurate` | `phonon_qpoints` |
| Phonon DOS | `spectra::phonon_dos` | `phonon_dos` | `get_phonon_dos` | `phonon_dos` |
| ZPE, vibrational F/U/S/Cv | `spectra::harmonic_thermodynamics` | `harmonic_thermodynamics` | `get_harmonic_thermodynamics` | `harmonic_thermodynamics` |
| Volume-sampled QHA | `spectra::quasiharmonic_zero_pressure` | `quasiharmonic_zero_pressure` | `get_quasiharmonic_zero_pressure` | `quasiharmonic_zero_pressure` |
| Bands and sampled gap | `spectra::pbc_band_structure` | `band_structure` | `get_band_structure` | `band_structure` |
| Electronic DOS | `spectra::electronic_dos_from_energies[_ev]` | `electronic_dos` | `get_electronic_dos` | `electronic_dos` |

The complete three-API examples, definitions and physical guards remain in
[mechanical properties](mechanical-properties.md),
[electromechanical response](electromechanical.md) and
[phonons/bands/thermodynamics](phonon-band-properties.md).
Property-objective model training uses the separate
[`gfn1-rs-train fit-properties` workflow](ml-property-training.md).
It has 17 explicitly registered targets; a postprocessor in this table is not
automatically a supported training target.

## Request envelope and conditions

```json
{
  "schema": "gfn1-rs-crystal-properties-v1",
  "geometry": {
    "numbers": [11, 17],
    "positions": [[0, 0, 0], [2.82, 0, 0]],
    "cell": [[0, 2.82, 2.82], [2.82, 0, 2.82], [2.82, 2.82, 0]],
    "coordinate_units": "angstrom",
    "charge": 0
  },
  "electronic": {
    "temperature_kelvin": 300,
    "energy_tolerance_hartree": 1e-13,
    "charge_tolerance": 1e-11,
    "max_scc": 1000,
    "scc_accelerator": "auto"
  },
  "periodic": {
    "kmesh": [2, 2, 2],
    "gamma_centered": true,
    "fold_time_reversal": false,
    "ao_cutoff_bohr": 30
  },
  "tasks": [{"id": "response", "operation": "electromechanical_response"}]
}
```

The three `cell` rows are the lattice vectors a,b,c, not the transpose of a
Cartesian component table. `coordinate_units` is required and is either
`angstrom` or `bohr`. Optional `pbc` defaults to `[true,true,true]`. Bulk
elastic/electrical properties require all three directions periodic.
Instead of arrays, geometry may contain `xyz_file`, `coordinate_units` and
`charge`; the file must carry its extended-XYZ lattice/PBC header. File and
array geometries cannot be mixed.

Electronic defaults are 300 K, energy tolerance 1e−12 Hartree, charge tolerance
1e−10, and the normal default SCC iteration/mixing controls. Optional keys are
`max_scc`, `mixing`, `scc_accelerator` (`auto`, `broyden`, `linear`, `cdiis`,
`newton`), `broyden_history`, `enable_dispersion`, `nprim`, `charge_order`,
`coordination_cutoff_bohr` and `enable_cn_hamiltonian`. The output `conditions`
records the requested controls. As in the typed APIs, analytic response
backends can tighten loose SCC thresholds; tighter user thresholds and the
requested temperature are preserved. Actual response thresholds are returned
where the underlying result supplies them. Electronic and vibrational lattice
temperatures are distinct.

Periodic defaults are Gamma `[1,1,1]`, Gamma-centred, unfolded, AO cutoff
30 bohr and the normal Ewald settings. Optional `ewald` contains `k_split`,
`real_cutoff_bohr`, `reciprocal_cutoff_inverse_bohr`,
`short_range_cutoff_bohr` and `short_range_width_bohr`.
Berry responses require a matching complete unfolded Gamma-centred reference
mesh and effectively integer occupations at the actual temperature. No
automatic cooling or occupation-guard relaxation is performed.

Optional top-level `parameter_file` selects the GFN1 parameter file or builtin
spec. If omitted, the normal parameter/environment/builtin resolution applies,
and the resolved source is returned. `ml_model_path` selects user weights or an explicitly named bundled model.
Relative parameter/model/XYZ paths resolve against the **request file's
directory**, independently of the shell's working directory. The loaded model
path, source SHA, canonical SHA, provenance and license are printed to stderr
and retained in the report. Only physical calculation tasks carry an active
model identity; pure postprocessing does not invent the origin of caller arrays.

Optional logging is
`"logging":{"file":"run.jsonl","level":"info","console":false}`;
the log path also resolves relative to the request file. `--threads`,
`--memory-mb`, `--cp-tolerance`, `--cp-relative-tolerance` and `--cp-max-iter`
apply to the whole request through the existing Rust resource scope. Memory
is a checked workspace budget, not a process RSS limit. JSON results go to
`--output` or clean stdout; model/resource diagnostics go to stderr.

## Task chaining and failures

Each task has a unique ASCII `id`, an `operation`, optional `arguments`, and
optional replacement `geometry`. Arguments can reference an earlier completed
result using a JSON Pointer:

```json
{
  "id": "lo100",
  "operation": "lo_to_modes",
  "arguments": {
    "hessian_hartree_per_bohr2": {"$ref":"response#/elastic/nuclear_hessian_hartree_per_bohr2"},
    "born_effective_charges": {"$ref":"response#/berry/born_effective_charges"},
    "electronic_relative_dielectric": {"$ref":"response#/electronic/electronic_relative_dielectric"},
    "volume_bohr3": {"$ref":"response#/berry/volume_bohr3"},
    "masses_amu": [22.989769, 35.45],
    "direction": [1, 0, 0]
  }
}
```

References address the task's `result`, not its wrapper. Forward/missing
references and references into failed tasks are errors. A `$ref` object must
contain only that key. A successful ionic/cell relaxation does **not** silently
replace the default geometry of later tasks. Set that explicitly:

```json
{
  "id": "relaxed_elastic",
  "operation": "elastic_constants",
  "geometry": {
    "numbers": {"$ref":"relax#/numbers"},
    "positions": {"$ref":"relax#/positions_bohr"},
    "cell": {"$ref":"relax#/cell_bohr"},
    "coordinate_units": "bohr",
    "charge": 0
  }
}
```

Schema/configuration errors fail before calculation. A failed physical task
is recorded as `error`; an unconverged relaxation retains its geometry/history
and is `not_converged`. Later tasks are `skipped` by default. Top-level
`continue_on_error:true` allows independent later tasks to run, while bad
dependencies remain errors. The CLI writes the partial report and exits with
code 1. Successful request execution exits 0. Status `completed` means the
requested function returned: it does **not** certify a stable crystal,
converged k sampling, or an available relaxed-ion response. Those flags,
negative modes, null tensors and reasons remain in the result.

## Operation arguments and units

All array and tolerance names below are native units. The CLI never infers
ASE units or converts an unlabelled eV/angstrom Hessian as if it were atomic.
Unknown keys are rejected rather than ignored.

| Operation | Required arguments | Optional arguments |
| --- | --- | --- |
| `ground_state`, `hessian`, `berry_polarization`, `born_effective_charges`, `dielectric_tensor` | none | none |
| `relax_periodic` | none | `relax_cell`, `max_iterations`, `force_tolerance_hartree_per_bohr`, `stress_tolerance_hartree_per_bohr3`, `target_pressure_hartree_per_bohr3`, `max_atom_step_bohr`, `max_strain_step`, `history` |
| `equation_of_state` | `volume_ratios` | `relax_ions` and fixed-cell ionic relaxation controls above; `relax_cell:true` is rejected |
| `elastic_constants` | none | `relax_reference`, `force_tolerance_hartree_per_bohr`, `stability_tolerance_hartree_per_bohr2`, fixed-cell ionic relaxation controls |
| `electromechanical_response` | none | elastic controls above, `masses_amu`, `zero_curvature_tolerance_hartree_per_bohr2_amu`, `zero_stress_tolerance_hartree_per_bohr3` |
| `phonon_mesh` | `repetitions:[nx,ny,nz]` | `masses_amu`, `memory_budget_bytes`, `max_atoms` |
| `phonon_qpoints` | repetitions, `q_fractional`, `q_weights` | same phonon controls |
| `band_structure` | exactly one of `kpoints_fractional` or `sampling_mesh` | `sampling_gamma_centered` with mesh sampling |
| `fit_equation_of_state` | `volumes_bohr3`, `energies_hartree` | none |
| `elastic_properties` | `stiffness_hartree_per_bohr3` (6x6), `density_kg_m3` | none |
| `directional_elastic_properties` | stiffness, density, Cartesian `direction`, Cartesian `transverse` | none |
| `polar_mode_properties` | `hessian_hartree_per_bohr2`, `born_effective_charges`, `masses_amu`, `volume_bohr3` | `zero_tolerance_hartree_per_bohr2_amu` |
| `lo_to_modes` | Hessian/Born/masses/volume, `electronic_relative_dielectric` (3x3), Cartesian `direction` | none |
| `phonon_dos` | `wavenumbers_cm1` (q,mode), `weights`, `grid_cm1` | `sigma_cm1` |
| `harmonic_thermodynamics` | wavenumbers, weights, `temperatures_kelvin` | `zero_tolerance_cm1` |
| `electronic_dos` | exactly one of `energies_hartree` or `energies_ev` (k,band), `weights`, `grid_ev` | `sigma_ev` |
| `quasiharmonic_zero_pressure` | `volumes_bohr3`, `static_energies_hartree`, `phonons_by_volume_cm1` (V,q,mode), `q_weights`, `temperatures_kelvin` | `zero_tolerance_cm1` |

The phonon supercell API requires `periodic.kmesh:[1,1,1]`: its Gamma point
folds a primitive electronic grid determined by the supercell repetitions.
This is different from primitive-cell Gamma sampling. QHA requires at least
five compatible volume points with the same physical/reference conventions;
the API is a volume-fit workflow, not a closed-form strain derivative of the
phonon spectrum. It rejects significant imaginary modes.

Band paths do not supply Brillouin-zone integration weights. Use
`sampling_mesh` to generate a complete integration grid and obtain the output
`integration_weights`; this resamples the same reference SCC potential.
Alternatively supply an independently defined full-grid energy/weight table.
Do not use a symmetry path as a physical electronic DOS integration grid.
Inspect DOS normalization/tail diagnostics and converge grids and broadening.
No arbitrary-q phonon interpolation is introduced by this command.

For elastic postprocessing choose the available clamped or relaxed tensor
explicitly. A null relaxed tensor is not replaced with the clamped one. Supply
the physical density for the same cell and isotope masses. Coupled response
continues to require stationary stable ions for relaxed e/static dielectric,
and stable zero-stress references for d/free-stress dielectric. The native
units, proper piezoelectric convention and mode normalization are defined in
the linked subsystem guides.

## Verification

`tests/crystal_cli.rs` compares nonzero user-ML CLI request outputs to the typed
Rust APIs, exercises all 21 operations, checks explicit relaxed-geometry
chaining and an actual LiF EOS, and preserves unstable/nonconverged failures.
`tests/python/test_crystal_cli.py` runs the installed pip command, checks
relative user-weight resolution and hashes, compares native energy/force/stress
availability, and requires nonzero exit plus retained JSON for a failed physical
task. These gates do not make the prototype examples experimental benchmarks.
Both distributed example request files were also executed unchanged with the
installed command: the NaCl coupled/LO–TO/band tasks and the diamond
phonon/thermodynamic/full-grid electronic-DOS tasks returned successfully.
The one-cell phonon example is a Gamma demonstration, not a converged bulk
heat capacity; refine the repetitions before drawing that physical conclusion.
