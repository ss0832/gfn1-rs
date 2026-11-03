// SPDX-License-Identifier: GPL-3.0-or-later
//! Structured crystal-property requests shared by Rust and pip commands.
//! Physical calculations call the same analytic backends as the typed APIs.
use crate::electronic::SccAccelerator;
use crate::linalg::Matrix;
use crate::pbc::elastic_response::{pbc_elastic_response, AnalyticElasticOptions};
use crate::pbc::electromechanical::{self, ElectromechanicalOptions};
use crate::pbc::mechanical::{self, PeriodicRelaxOptions};
use crate::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
use crate::pbc::polarization::response::{pbc_electronic_dielectric, pbc_polarization_derivatives};
use crate::pbc::polarization::{pbc_berry_polarization, BerryPolarizationOptions};
use crate::pbc::{spectra, KMesh, PbcOptions};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SCHEMA: &str = "gfn1-rs-crystal-properties-v1";

fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{label} must be a JSON object")))
}
fn keys(value: &Map<String, Value>, allowed: &[&str], label: &str) -> Result<()> {
    if let Some(key) = value.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(format!("unknown {label} field `{key}`")));
    }
    Ok(())
}
fn required<T: DeserializeOwned>(value: &Map<String, Value>, key: &str) -> Result<T> {
    let source = value
        .get(key)
        .ok_or_else(|| invalid(format!("missing `{key}`")))?;
    serde_json::from_value(source.clone())
        .map_err(|error| invalid(format!("invalid `{key}`: {error}")))
}
fn optional<T: DeserializeOwned>(value: &Map<String, Value>, key: &str, default: T) -> Result<T> {
    value.get(key).map_or(Ok(default), |source| {
        serde_json::from_value(source.clone())
            .map_err(|error| invalid(format!("invalid `{key}`: {error}")))
    })
}
fn path(base: &Path, value: &str) -> PathBuf {
    let value = Path::new(value);
    if value.is_absolute() {
        value.to_path_buf()
    } else {
        base.join(value)
    }
}
fn rows(matrix: &Matrix) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|i| (0..matrix.cols()).map(|j| matrix[(i, j)]).collect())
        .collect()
}
fn matrix(args: &Map<String, Value>, name: &str, size: usize) -> Result<Matrix> {
    let data: Vec<Vec<f64>> = required(args, name)?;
    if data.len() != size
        || data
            .iter()
            .any(|row| row.len() != size || row.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(format!(
            "{name} must be a finite {size}x{size} matrix"
        )));
    }
    crate::resources::check_dense_workspace(size, size, 1, "CLI matrix input")?;
    Matrix::from_vec(size, size, data.into_iter().flatten().collect())
}
fn born(args: &Map<String, Value>) -> Result<Vec<Matrix>> {
    let data: Vec<Vec<Vec<f64>>> = required(args, "born_effective_charges")?;
    data.into_iter()
        .map(|tensor| {
            if tensor.len() != 3
                || tensor
                    .iter()
                    .any(|row| row.len() != 3 || row.iter().any(|v| !v.is_finite()))
            {
                return Err(invalid(
                    "Born tensors must be finite [atom][3][3] arrays in e",
                ));
            }
            Matrix::from_vec(3, 3, tensor.into_iter().flatten().collect())
        })
        .collect()
}

fn geometry(value: &Value, base: &Path) -> Result<PeriodicSystem> {
    let input = object(value, "geometry")?;
    keys(
        input,
        &[
            "numbers",
            "positions",
            "cell",
            "pbc",
            "coordinate_units",
            "charge",
            "xyz_file",
        ],
        "geometry",
    )?;
    let unit: String = required(input, "coordinate_units")?;
    let bohr = match unit.as_str() {
        "angstrom" => false,
        "bohr" => true,
        _ => return Err(invalid("coordinate_units must be angstrom or bohr")),
    };
    let charge: f64 = optional(input, "charge", 0.)?;
    if !charge.is_finite() {
        return Err(invalid("charge must be finite"));
    }
    let system = if let Some(file) = input.get("xyz_file") {
        if ["numbers", "positions", "cell", "pbc"]
            .iter()
            .any(|key| input.contains_key(*key))
        {
            return Err(invalid(
                "xyz_file cannot be combined with explicit geometry arrays",
            ));
        }
        let file = file
            .as_str()
            .ok_or_else(|| invalid("xyz_file must be a path string"))?;
        PeriodicSystem::from_xyz_file(path(base, file), charge, bohr)?
    } else {
        let numbers: Vec<u8> = required(input, "numbers")?;
        let positions: Vec<[f64; 3]> = required(input, "positions")?;
        let cell: [[f64; 3]; 3] = required(input, "cell")?;
        let periodic: [bool; 3] = optional(input, "pbc", [true; 3])?;
        if numbers.is_empty()
            || numbers.len() != positions.len()
            || positions
                .iter()
                .flatten()
                .chain(cell.iter().flatten())
                .any(|x| !x.is_finite())
        {
            return Err(invalid(
                "geometry needs matching nonempty numbers/positions and finite cell coordinates",
            ));
        }
        let scale = if bohr {
            1.
        } else {
            crate::constants::ANGSTROM_TO_BOHR
        };
        let vector = |r: [f64; 3]| crate::math::Vec3::new(r[0], r[1], r[2]) * scale;
        let lattice = crate::lattice::Lattice::new(
            crate::math::Mat3::from_columns(vector(cell[0]), vector(cell[1]), vector(cell[2])),
            periodic,
        )?;
        PeriodicSystem::new(
            numbers
                .into_iter()
                .zip(positions)
                .map(|(z, r)| crate::system::Atom {
                    z,
                    position: vector(r),
                })
                .collect(),
            Some(lattice),
        )
        .with_charge(charge)
    };
    if system.lattice.is_none() {
        return Err(invalid(
            "crystal properties require a lattice-bearing periodic geometry",
        ));
    }
    if system.atoms.iter().any(|atom| atom.z == 0 || atom.z > 86) {
        return Err(invalid("GFN1 crystal species require supported Z=1..86"));
    }
    Ok(system)
}

fn electronic(value: Option<&Value>) -> Result<ElectronicOptions> {
    let empty = json!({});
    let input = object(value.unwrap_or(&empty), "electronic")?;
    keys(
        input,
        &[
            "temperature_kelvin",
            "max_scc",
            "energy_tolerance_hartree",
            "charge_tolerance",
            "mixing",
            "scc_accelerator",
            "broyden_history",
            "enable_dispersion",
            "nprim",
            "charge_order",
            "coordination_cutoff_bohr",
            "enable_cn_hamiltonian",
        ],
        "electronic",
    )?;
    let mut options = ElectronicOptions::default();
    options.electronic_temperature = optional(input, "temperature_kelvin", 300.)?;
    options.max_scc = optional(input, "max_scc", options.max_scc)?;
    options.energy_tolerance = optional(input, "energy_tolerance_hartree", 1e-12)?;
    options.charge_tolerance = optional(input, "charge_tolerance", 1e-10)?;
    options.mixing = optional(input, "mixing", options.mixing)?;
    options.scc_broyden_size = optional(input, "broyden_history", options.scc_broyden_size)?;
    options.enable_dispersion = optional(input, "enable_dispersion", options.enable_dispersion)?;
    options.nprim = optional(input, "nprim", options.nprim)?;
    options.charge_order = optional(input, "charge_order", options.charge_order)?;
    options.hamiltonian.coordination_cutoff = optional(
        input,
        "coordination_cutoff_bohr",
        options.hamiltonian.coordination_cutoff,
    )?;
    options.hamiltonian.enable_cn_hamiltonian = optional(
        input,
        "enable_cn_hamiltonian",
        options.hamiltonian.enable_cn_hamiltonian,
    )?;
    let accelerator: String = optional(input, "scc_accelerator", "auto".into())?;
    options.scc_accelerator = match accelerator.as_str() {
        "auto" => SccAccelerator::Auto,
        "linear" => SccAccelerator::Linear,
        "broyden" => SccAccelerator::Broyden,
        "cdiis" => SccAccelerator::Cdiis,
        "newton" => SccAccelerator::Newton,
        _ => {
            return Err(invalid(
                "scc_accelerator must be auto/linear/broyden/cdiis/newton",
            ))
        }
    };
    if !options.electronic_temperature.is_finite()
        || options.electronic_temperature < 0.
        || options.max_scc == 0
        || [
            options.energy_tolerance,
            options.charge_tolerance,
            options.mixing,
            options.hamiltonian.coordination_cutoff,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
        || options.mixing > 1.
        || options.charge_order < 3
        || options.scc_broyden_size < 2
    {
        return Err(invalid(
            "invalid electronic temperature, SCC controls, mixing, charge order or CN cutoff",
        ));
    }
    Ok(options)
}
fn periodic(value: Option<&Value>) -> Result<PbcOptions> {
    let empty = json!({});
    let input = object(value.unwrap_or(&empty), "periodic")?;
    keys(
        input,
        &[
            "kmesh",
            "gamma_centered",
            "fold_time_reversal",
            "ao_cutoff_bohr",
            "ewald",
        ],
        "periodic",
    )?;
    let mut periodic = PbcOptions::default();
    periodic.kmesh = KMesh {
        size: optional(input, "kmesh", [1; 3])?,
        gamma_centered: optional(input, "gamma_centered", true)?,
        fold_time_reversal: optional(input, "fold_time_reversal", false)?,
    };
    periodic.ao_cutoff = optional(input, "ao_cutoff_bohr", periodic.ao_cutoff)?;
    if periodic.kmesh.size.contains(&0)
        || periodic
            .kmesh
            .size
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .is_none()
        || !periodic.ao_cutoff.is_finite()
        || periodic.ao_cutoff <= 0.
    {
        return Err(invalid(
            "periodic mesh and AO cutoff must be positive and finite",
        ));
    }
    if let Some(value) = input.get("ewald") {
        let e = object(value, "Ewald")?;
        keys(
            e,
            &[
                "k_split",
                "real_cutoff_bohr",
                "reciprocal_cutoff_inverse_bohr",
                "short_range_cutoff_bohr",
                "short_range_width_bohr",
            ],
            "Ewald",
        )?;
        periodic.ewald.k_split = optional(e, "k_split", periodic.ewald.k_split)?;
        periodic.ewald.real_cutoff = optional(e, "real_cutoff_bohr", periodic.ewald.real_cutoff)?;
        periodic.ewald.recip_cutoff = optional(
            e,
            "reciprocal_cutoff_inverse_bohr",
            periodic.ewald.recip_cutoff,
        )?;
        periodic.ewald.sr_cutoff =
            optional(e, "short_range_cutoff_bohr", periodic.ewald.sr_cutoff)?;
        periodic.ewald.sr_width = optional(e, "short_range_width_bohr", periodic.ewald.sr_width)?;
        if [
            periodic.ewald.real_cutoff,
            periodic.ewald.sr_cutoff,
            periodic.ewald.sr_width,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
            || !periodic.ewald.recip_cutoff.is_finite()
            || periodic.ewald.recip_cutoff < 0.
            || periodic
                .ewald
                .k_split
                .is_some_and(|x| !x.is_finite() || x <= 0.)
        {
            return Err(invalid("invalid Ewald cutoffs or splitting parameter"));
        }
    }
    Ok(periodic)
}

fn relax_options(input: &Map<String, Value>) -> Result<PeriodicRelaxOptions> {
    let mut settings = PeriodicRelaxOptions::default();
    settings.relax_cell = optional(input, "relax_cell", settings.relax_cell)?;
    settings.max_iterations = optional(input, "max_iterations", settings.max_iterations)?;
    settings.force_tolerance = optional(
        input,
        "force_tolerance_hartree_per_bohr",
        settings.force_tolerance,
    )?;
    settings.stress_tolerance = optional(
        input,
        "stress_tolerance_hartree_per_bohr3",
        settings.stress_tolerance,
    )?;
    settings.target_pressure = optional(
        input,
        "target_pressure_hartree_per_bohr3",
        settings.target_pressure,
    )?;
    settings.max_atom_step = optional(input, "max_atom_step_bohr", settings.max_atom_step)?;
    settings.max_strain_step = optional(input, "max_strain_step", settings.max_strain_step)?;
    settings.history = optional(input, "history", settings.history)?;
    Ok(settings)
}
fn elastic_options(input: &Map<String, Value>) -> Result<AnalyticElasticOptions> {
    let mut settings = AnalyticElasticOptions::default();
    settings.relax_reference = optional(input, "relax_reference", settings.relax_reference)?;
    settings.force_tolerance = optional(
        input,
        "force_tolerance_hartree_per_bohr",
        settings.force_tolerance,
    )?;
    settings.stability_tolerance = optional(
        input,
        "stability_tolerance_hartree_per_bohr2",
        settings.stability_tolerance,
    )?;
    settings.relax = relax_options(input)?;
    settings.relax.force_tolerance = settings.force_tolerance;
    Ok(settings)
}

/// Public operation names and their underlying APIs. Array arguments use the
/// unit indicated in their key; no ASE unit inference occurs in the CLI.
pub fn operation_catalog() -> Value {
    let rows = json!([
        [
            "ground_state",
            true,
            "run_pbc_scc + analytic gradient/stress"
        ],
        ["hessian", true, "pbc_gamma_hessian / pbc_kpoint_hessian"],
        ["relax_periodic", true, "pbc_relax"],
        ["equation_of_state", true, "pbc_equation_of_state"],
        ["elastic_constants", true, "pbc_elastic_response"],
        ["berry_polarization", true, "pbc_berry_polarization"],
        [
            "born_effective_charges",
            true,
            "pbc_polarization_derivatives"
        ],
        ["dielectric_tensor", true, "pbc_electronic_dielectric"],
        [
            "electromechanical_response",
            true,
            "pbc_electromechanical_response"
        ],
        [
            "phonon_mesh",
            true,
            "AnalyticSupercellPhonons + sample_phonon_mesh"
        ],
        [
            "phonon_qpoints",
            true,
            "AnalyticSupercellPhonons + sample_phonon_qpoints"
        ],
        ["band_structure", true, "pbc_band_structure"],
        ["fit_equation_of_state", false, "fit_birch_murnaghan"],
        ["elastic_properties", false, "elastic_properties"],
        [
            "directional_elastic_properties",
            false,
            "directional_elastic_response"
        ],
        ["polar_mode_properties", false, "polar_mode_properties"],
        ["lo_to_modes", false, "lo_to_modes"],
        ["phonon_dos", false, "phonon_dos"],
        ["harmonic_thermodynamics", false, "harmonic_thermodynamics"],
        ["electronic_dos", false, "electronic_dos_from_energies[_ev]"],
        [
            "quasiharmonic_zero_pressure",
            false,
            "quasiharmonic_zero_pressure"
        ]
    ]);
    Value::Array(
        rows.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                json!({
                    "operation":row[0],"geometry_required":row[1],"rust_backend":row[2]
                })
            })
            .collect(),
    )
}
fn needs_geometry(operation: &str) -> Result<bool> {
    operation_catalog()
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["operation"] == operation)
        .map(|row| row["geometry_required"].as_bool().unwrap())
        .ok_or_else(|| {
            invalid(format!(
                "unknown crystal operation `{operation}`; use properties --list"
            ))
        })
}

struct Context {
    params: Gfn1Parameters,
    electronic: ElectronicOptions,
    periodic: PbcOptions,
    ml_identity: Option<crate::ml::MlModelIdentity>,
}
fn masses(input: &Map<String, Value>, system: &PeriodicSystem) -> Result<Vec<f64>> {
    optional(
        input,
        "masses_amu",
        system
            .atoms
            .iter()
            .map(|a| crate::data_tables::relative_atomic_mass(a.z))
            .collect(),
    )
}
const RELAX_KEYS: &[&str] = &[
    "relax_cell",
    "max_iterations",
    "force_tolerance_hartree_per_bohr",
    "stress_tolerance_hartree_per_bohr3",
    "target_pressure_hartree_per_bohr3",
    "max_atom_step_bohr",
    "max_strain_step",
    "history",
];
fn combined_keys<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
    RELAX_KEYS
        .iter()
        .copied()
        .chain(extra.iter().copied())
        .collect()
}

fn calculate(
    context: &Context,
    operation: &str,
    system: Option<&PeriodicSystem>,
    input: &Map<String, Value>,
) -> Result<Value> {
    let params = &context.params;
    let electronic = &context.electronic;
    let pbc = &context.periodic;
    let system = || system.ok_or_else(|| invalid(format!("{operation} requires geometry")));
    let berry = BerryPolarizationOptions::from_kmesh(pbc.kmesh);
    match operation {
        "ground_state" => {
            keys(input, &[], operation)?;
            let system = system()?;
            let scf = crate::pbc::run_pbc_scc(system, params, electronic, pbc)?;
            let gradient = crate::pbc::pbc_gradient_from_scc(system, params, scf, electronic, pbc)?;
            let stress = crate::pbc::pbc_stress_from_scc(
                system,
                params,
                gradient.scf,
                electronic,
                pbc,
                &system.lattice.unwrap(),
            )?;
            Ok(
                json!({"energy_hartree":stress.total_energy,"internal_energy_hartree":stress.scf.total_internal,
                "gradient_hartree_per_bohr":gradient.gradient.iter().map(|r|r.to_array()).collect::<Vec<_>>(),
                "forces_hartree_per_bohr":gradient.forces.iter().map(|r|r.to_array()).collect::<Vec<_>>(),
                "stress_hartree_per_bohr3":rows(&stress.stress),"atomic_charges":stress.scf.atomic_charges,
                "pressure_gpa":-(0..3).map(|i|stress.stress[(i,i)]).sum::<f64>()/3.*mechanical::STRESS_AU_TO_GPA,
                "scc_iterations":stress.scf.iterations,"scc_diagnostics":stress.scf.scc_diagnostics.as_json()}),
            )
        }
        "hessian" => {
            keys(input, &[], operation)?;
            let result = if pbc.kmesh.is_gamma_only() {
                crate::pbc::pbc_gamma_hessian(system()?, params, electronic, pbc)?
            } else {
                crate::pbc::pbc_kpoint_hessian(system()?, params, electronic, pbc)?
            };
            Ok(
                json!({"energy_hartree":result.scf.total_free,"hessian_hartree_per_bohr2":rows(&result.hessian),"raw_max_asymmetry":result.raw_max_asymmetry,"scc_diagnostics":result.scf.scc_diagnostics.as_json()}),
            )
        }
        "relax_periodic" => {
            keys(input, RELAX_KEYS, operation)?;
            Ok(
                mechanical::pbc_relax(system()?, params, electronic, pbc, &relax_options(input)?)?
                    .as_json(),
            )
        }
        "equation_of_state" => {
            keys(
                input,
                &combined_keys(&["volume_ratios", "relax_ions"]),
                operation,
            )?;
            if optional(input, "relax_cell", false)? {
                return Err(invalid(
                    "EOS relax_ions uses fixed-volume cells; relax_cell must be false",
                ));
            }
            let ratios: Vec<f64> = required(input, "volume_ratios")?;
            Ok(mechanical::pbc_equation_of_state(
                system()?,
                params,
                electronic,
                pbc,
                &ratios,
                optional(input, "relax_ions", false)?,
                &relax_options(input)?,
            )?
            .as_json())
        }
        "elastic_constants" => {
            keys(
                input,
                &combined_keys(&["relax_reference", "stability_tolerance_hartree_per_bohr2"]),
                operation,
            )?;
            if optional(input, "relax_cell", false)? {
                return Err(invalid("elastic reference relaxation is ionic; perform cell relaxation in a preceding task"));
            }
            Ok(
                pbc_elastic_response(system()?, params, electronic, pbc, &elastic_options(input)?)?
                    .as_json(),
            )
        }
        "berry_polarization" => {
            keys(input, &[], operation)?;
            let result = pbc_berry_polarization(system()?, params, electronic, pbc, &berry)?;
            Ok(
                json!({"polarization_au":result.polarization,"polarization_raw_au":result.polarization_raw,"polarization_quantum_au":result.quantum,"volume_bohr3":result.volume,"mesh":result.mesh,"minimum_band_gap_hartree":result.min_band_gap}),
            )
        }
        "born_effective_charges" => {
            keys(input, &[], operation)?;
            Ok(pbc_polarization_derivatives(system()?, params, electronic, pbc, &berry)?.as_json())
        }
        "dielectric_tensor" => {
            keys(input, &[], operation)?;
            Ok(pbc_electronic_dielectric(system()?, params, electronic, pbc, &berry)?.as_json())
        }
        "electromechanical_response" => {
            keys(
                input,
                &combined_keys(&[
                    "relax_reference",
                    "stability_tolerance_hartree_per_bohr2",
                    "masses_amu",
                    "zero_curvature_tolerance_hartree_per_bohr2_amu",
                    "zero_stress_tolerance_hartree_per_bohr3",
                ]),
                operation,
            )?;
            if optional(input, "relax_cell", false)? {
                return Err(invalid("electromechanical reference relaxation is ionic; perform cell relaxation in a preceding task"));
            }
            let mut settings = ElectromechanicalOptions {
                berry,
                elastic: elastic_options(input)?,
                ..Default::default()
            };
            settings.zero_curvature_tolerance = optional(
                input,
                "zero_curvature_tolerance_hartree_per_bohr2_amu",
                settings.zero_curvature_tolerance,
            )?;
            settings.zero_stress_tolerance = optional(
                input,
                "zero_stress_tolerance_hartree_per_bohr3",
                settings.zero_stress_tolerance,
            )?;
            Ok(electromechanical::pbc_electromechanical_response(
                system()?,
                params,
                electronic,
                pbc,
                &masses(input, system()?)?,
                &settings,
            )?
            .as_json())
        }
        "phonon_mesh" | "phonon_qpoints" => {
            keys(
                input,
                &[
                    "repetitions",
                    "masses_amu",
                    "memory_budget_bytes",
                    "max_atoms",
                    "q_fractional",
                    "q_weights",
                ],
                operation,
            )?;
            if !pbc.kmesh.is_gamma_only() {
                return Err(invalid("phonon supercell uses Gamma of the supercell; periodic.kmesh must be [1,1,1], with primitive electronic folding set by repetitions"));
            }
            if operation == "phonon_mesh"
                && (input.contains_key("q_fractional") || input.contains_key("q_weights"))
            {
                return Err(invalid("phonon_mesh generates its full commensurate mesh; use phonon_qpoints for explicit q points"));
            }
            let settings = PhononSupercellOptions {
                repetitions: required(input, "repetitions")?,
                memory_budget_bytes: optional(input, "memory_budget_bytes", 536870912usize)?,
                max_atoms: optional(input, "max_atoms", 128usize)?,
            };
            let reference = AnalyticSupercellPhonons::build(
                system()?,
                params,
                electronic,
                pbc,
                &masses(input, system()?)?,
                settings,
            )?;
            let result = if operation == "phonon_mesh" {
                spectra::sample_phonon_mesh(&reference)?
            } else {
                spectra::sample_phonon_qpoints(
                    &reference,
                    &required::<Vec<[f64; 3]>>(input, "q_fractional")?,
                    &required::<Vec<f64>>(input, "q_weights")?,
                )?
            };
            let mut out = result.as_json();
            out["electronic_sampling"] =
                json!("Gamma of supercell; primitive electronic folding follows repetitions");
            Ok(out)
        }
        "band_structure" => {
            keys(
                input,
                &[
                    "kpoints_fractional",
                    "sampling_mesh",
                    "sampling_gamma_centered",
                ],
                operation,
            )?;
            if input.contains_key("kpoints_fractional") == input.contains_key("sampling_mesh") {
                return Err(invalid(
                    "band_structure requires exactly one of kpoints_fractional or sampling_mesh",
                ));
            }
            let (points, weights, mesh) = if input.contains_key("sampling_mesh") {
                let mesh: [usize; 3] = required(input, "sampling_mesh")?;
                let count = mesh
                    .iter()
                    .try_fold(1usize, |n, &m| n.checked_mul(m))
                    .filter(|_| !mesh.contains(&0))
                    .ok_or_else(|| invalid("sampling_mesh must be positive and not overflow"))?;
                crate::resources::check_dense_workspace(count, 3, 2, "band sampling mesh")?;
                let points = crate::pbc::kpoints::monkhorst_pack(
                    mesh,
                    system()?.lattice.unwrap().periodic,
                    optional(input, "sampling_gamma_centered", true)?,
                );
                (
                    points.iter().map(|p| p.fractional).collect::<Vec<_>>(),
                    Some(points.iter().map(|p| p.weight).collect::<Vec<_>>()),
                    Some(mesh),
                )
            } else {
                if input.contains_key("sampling_gamma_centered") {
                    return Err(invalid("sampling_gamma_centered requires sampling_mesh"));
                }
                (
                    required::<Vec<[f64; 3]>>(input, "kpoints_fractional")?,
                    None,
                    None,
                )
            };
            let mut result =
                spectra::pbc_band_structure(system()?, params, electronic, pbc, &points)?.as_json();
            result["integration_weights"] = json!(weights);
            result["sampling_mesh"] = json!(mesh);
            result["sampling_kind"] = json!(if mesh.is_some() {
                "complete integration mesh at the reference SCC potential"
            } else {
                "caller points/path; no integration weights inferred"
            });
            Ok(result)
        }
        "fit_equation_of_state" => {
            keys(input, &["volumes_bohr3", "energies_hartree"], operation)?;
            Ok(json!(mechanical::fit_birch_murnaghan(
                &required::<Vec<f64>>(input, "volumes_bohr3")?,
                &required::<Vec<f64>>(input, "energies_hartree")?
            )?))
        }
        "elastic_properties" | "directional_elastic_properties" => {
            keys(
                input,
                &[
                    "stiffness_hartree_per_bohr3",
                    "density_kg_m3",
                    "direction",
                    "transverse",
                ],
                operation,
            )?;
            let stiffness = matrix(input, "stiffness_hartree_per_bohr3", 6)?;
            let density = required(input, "density_kg_m3")?;
            if operation == "elastic_properties" {
                if input.contains_key("direction") || input.contains_key("transverse") {
                    return Err(invalid(
                        "direction/transverse require directional_elastic_properties",
                    ));
                }
                Ok(json!(mechanical::elastic_properties(&stiffness, density)?))
            } else {
                Ok(json!(mechanical::directional_elastic_response(
                    &stiffness,
                    density,
                    required(input, "direction")?,
                    required(input, "transverse")?
                )?))
            }
        }
        "polar_mode_properties" | "lo_to_modes" => {
            keys(
                input,
                &[
                    "hessian_hartree_per_bohr2",
                    "born_effective_charges",
                    "masses_amu",
                    "volume_bohr3",
                    "zero_tolerance_hartree_per_bohr2_amu",
                    "electronic_relative_dielectric",
                    "direction",
                ],
                operation,
            )?;
            let masses: Vec<f64> = required(input, "masses_amu")?;
            let hessian = matrix(input, "hessian_hartree_per_bohr2", 3 * masses.len())?;
            let born = born(input)?;
            let volume = required(input, "volume_bohr3")?;
            if operation == "polar_mode_properties" {
                if input.contains_key("electronic_relative_dielectric")
                    || input.contains_key("direction")
                {
                    return Err(invalid("dielectric/direction require lo_to_modes"));
                }
                Ok(electromechanical::polar_mode_properties(
                    &hessian,
                    &born,
                    &masses,
                    volume,
                    optional(input, "zero_tolerance_hartree_per_bohr2_amu", 1e-10)?,
                )?
                .as_json())
            } else {
                if input.contains_key("zero_tolerance_hartree_per_bohr2_amu") {
                    return Err(invalid(
                        "lo_to_modes does not use an ionic-static zero-mode cutoff",
                    ));
                }
                Ok(electromechanical::lo_to_modes(
                    &hessian,
                    &born,
                    &matrix(input, "electronic_relative_dielectric", 3)?,
                    volume,
                    &masses,
                    required(input, "direction")?,
                )?
                .as_json())
            }
        }
        "phonon_dos" => {
            keys(
                input,
                &["wavenumbers_cm1", "weights", "grid_cm1", "sigma_cm1"],
                operation,
            )?;
            Ok(spectra::phonon_dos(
                &required::<Vec<Vec<f64>>>(input, "wavenumbers_cm1")?,
                &required::<Vec<f64>>(input, "weights")?,
                &required::<Vec<f64>>(input, "grid_cm1")?,
                optional(input, "sigma_cm1", 10.)?,
            )?
            .as_json())
        }
        "harmonic_thermodynamics" => {
            keys(
                input,
                &[
                    "wavenumbers_cm1",
                    "weights",
                    "temperatures_kelvin",
                    "zero_tolerance_cm1",
                ],
                operation,
            )?;
            Ok(json!(spectra::harmonic_thermodynamics(
                &required::<Vec<Vec<f64>>>(input, "wavenumbers_cm1")?,
                &required::<Vec<f64>>(input, "weights")?,
                &required::<Vec<f64>>(input, "temperatures_kelvin")?,
                optional(input, "zero_tolerance_cm1", 1e-3)?
            )?
            .iter()
            .map(|v| v.as_json())
            .collect::<Vec<_>>()))
        }
        "electronic_dos" => {
            keys(
                input,
                &[
                    "energies_hartree",
                    "energies_ev",
                    "weights",
                    "grid_ev",
                    "sigma_ev",
                ],
                operation,
            )?;
            if input.contains_key("energies_hartree") == input.contains_key("energies_ev") {
                return Err(invalid(
                    "electronic_dos requires exactly one of energies_hartree / energies_ev",
                ));
            }
            let weights: Vec<f64> = required(input, "weights")?;
            let grid: Vec<f64> = required(input, "grid_ev")?;
            let sigma = optional(input, "sigma_ev", 0.1)?;
            let result = if input.contains_key("energies_hartree") {
                spectra::electronic_dos_from_energies(
                    &required::<Vec<Vec<f64>>>(input, "energies_hartree")?,
                    &weights,
                    &grid,
                    sigma,
                )?
            } else {
                spectra::electronic_dos_from_energies_ev(
                    &required::<Vec<Vec<f64>>>(input, "energies_ev")?,
                    &weights,
                    &grid,
                    sigma,
                )?
            };
            Ok(result.as_json())
        }
        "quasiharmonic_zero_pressure" => {
            keys(
                input,
                &[
                    "volumes_bohr3",
                    "static_energies_hartree",
                    "phonons_by_volume_cm1",
                    "q_weights",
                    "temperatures_kelvin",
                    "zero_tolerance_cm1",
                ],
                operation,
            )?;
            Ok(json!(spectra::quasiharmonic_zero_pressure(
                &required::<Vec<f64>>(input, "volumes_bohr3")?,
                &required::<Vec<f64>>(input, "static_energies_hartree")?,
                &required::<Vec<Vec<Vec<f64>>>>(input, "phonons_by_volume_cm1")?,
                &required::<Vec<f64>>(input, "q_weights")?,
                &required::<Vec<f64>>(input, "temperatures_kelvin")?,
                optional(input, "zero_tolerance_cm1", 1e-3)?
            )?
            .iter()
            .map(|v| v.as_json())
            .collect::<Vec<_>>()))
        }
        _ => Err(invalid(format!(
            "unhandled crystal operation `{operation}`"
        ))),
    }
}

fn resolve(value: &Value, results: &BTreeMap<String, usize>, completed: &[Value]) -> Result<Value> {
    match value {
        Value::Object(input) if input.contains_key("$ref") => {
            if input.len() != 1 {
                return Err(invalid("a $ref object cannot contain other fields"));
            }
            let spec = input["$ref"]
                .as_str()
                .ok_or_else(|| invalid("$ref must be a task-id#JSON-pointer string"))?;
            let (id, pointer) = spec
                .split_once('#')
                .ok_or_else(|| invalid("$ref must contain # before its JSON pointer"))?;
            results
                .get(id)
                .and_then(|&index| completed[index]["result"].pointer(pointer))
                .cloned()
                .ok_or_else(|| invalid(format!("unavailable result reference `{spec}`")))
        }
        Value::Object(input) => Ok(Value::Object(
            input
                .iter()
                .map(|(key, value)| Ok((key.clone(), resolve(value, results, completed)?)))
                .collect::<Result<_>>()?,
        )),
        Value::Array(input) => Ok(Value::Array(
            input
                .iter()
                .map(|value| resolve(value, results, completed))
                .collect::<Result<_>>()?,
        )),
        _ => Ok(value.clone()),
    }
}

/// Execute a validated request. Relative model/XYZ/log paths use `base`.
/// Tasks never silently change the electronic temperature or tensor units.
pub fn execute_request(request: &Value, base: &Path) -> Result<Value> {
    let input = object(request, "request")?;
    keys(
        input,
        &[
            "schema",
            "geometry",
            "electronic",
            "periodic",
            "parameter_file",
            "ml_model_path",
            "logging",
            "continue_on_error",
            "tasks",
        ],
        "request",
    )?;
    if required::<String>(input, "schema")? != SCHEMA {
        return Err(invalid(format!("request schema must be {SCHEMA}")));
    }
    let parameter: Option<String> = optional(input, "parameter_file", None)?;
    let parameter = parameter.map(|s| {
        if s == "builtin" || s.starts_with("builtin:") {
            s
        } else {
            path(base, &s).to_string_lossy().into_owned()
        }
    });
    let params = Gfn1Parameters::resolve(parameter.as_deref())?;
    let mut electronic = electronic(input.get("electronic"))?;
    let periodic = periodic(input.get("periodic"))?;
    if let Some(value) = input.get("logging") {
        let log = object(value, "logging")?;
        keys(log, &["file", "level", "console"], "logging")?;
        let file: String = required(log, "file")?;
        let level: String = optional(log, "level", "info".into())?;
        electronic.logger = Some(crate::CalculationLogger::append(
            path(base, &file),
            crate::LogLevel::parse(&level)?,
            optional(log, "console", false)?,
        )?);
    }
    let model: Option<String> = optional(input, "ml_model_path", None)?;
    let ml_identity = model
        .map(|s| {
            let spec = if s.starts_with("builtin:") {
                s
            } else {
                path(base, &s).to_string_lossy().into_owned()
            };
            let (model, identity) = crate::ml::MlHamiltonian::resolve_with_identity(&spec)?;
            electronic.ml_model = Some(Arc::new(model));
            if let Some(logger) = &electronic.logger {
                logger.event(crate::LogLevel::Info, "ml_model_loaded", identity.as_json())?;
            }
            eprintln!("[gfn1-rs ML model] {}", identity.description());
            Ok::<_, Gfn1Error>(identity)
        })
        .transpose()?;
    let context = Context {
        params,
        electronic,
        periodic,
        ml_identity,
    };
    let default_geometry = input
        .get("geometry")
        .map(|v| geometry(v, base))
        .transpose()?;
    let tasks: Vec<Value> = required(input, "tasks")?;
    if tasks.is_empty() {
        return Err(invalid("tasks must be nonempty"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for task in &tasks {
        let task = object(task, "task")?;
        keys(task, &["id", "operation", "arguments", "geometry"], "task")?;
        let id: String = required(task, "id")?;
        if id.is_empty()
            || id
                .chars()
                .any(|c| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
            || !seen.insert(id)
        {
            return Err(invalid(
                "task ids must be unique nonempty ASCII letters/digits/_/-",
            ));
        }
        needs_geometry(&required::<String>(task, "operation")?)?;
    }
    let continuation = optional(input, "continue_on_error", false)?;
    let mut results = BTreeMap::new();
    let mut output = Vec::new();
    let mut failures = 0;
    let mut stopped = false;
    let mut skipped = 0;
    for task in tasks {
        let task = task.as_object().unwrap();
        let id: String = required(task, "id")?;
        let operation: String = required(task, "operation")?;
        if stopped {
            skipped += 1;
            output.push(json!({"id":id,"operation":operation,"status":"skipped","reason":"earlier task did not complete; continue_on_error was false"}));
            continue;
        }
        let uses_geometry = needs_geometry(&operation)?;
        let calculation: Result<(Value, Option<f64>)> = (|| {
            let args = resolve(
                task.get("arguments").unwrap_or(&json!({})),
                &results,
                &output,
            )?;
            let args = object(&args, "arguments")?;
            let override_geometry = task
                .get("geometry")
                .map(|value| geometry(&resolve(value, &results, &output)?, base))
                .transpose()?;
            if !uses_geometry && override_geometry.is_some() {
                return Err(invalid(
                    "postprocessing task cannot override unused geometry",
                ));
            }
            let system = override_geometry.as_ref().or(default_geometry.as_ref());
            let result = calculate(&context, &operation, system, args)?;
            Ok((result, system.map(|s| s.charge)))
        })();
        match calculation {
            Ok((value, charge)) => {
                let provenance = if uses_geometry {
                    json!({"kind":"analytic Rust physical calculation","electronic_temperature_kelvin":context.electronic.electronic_temperature,"charge":charge,"ml_model_identity":context.ml_identity.as_ref().map(|m|m.as_json())})
                } else {
                    json!({"kind":"postprocessing of explicitly supplied arrays or prior result references; no new SCC or inferred ML identity"})
                };
                let not_converged = operation == "relax_periodic" && value["converged"] == false;
                if not_converged {
                    failures += 1;
                    stopped = !continuation;
                }
                output.push(json!({"id":id,"operation":operation,"status":if not_converged{"not_converged"}else{"completed"},"provenance":provenance,"result":value}));
                results.insert(id, output.len() - 1);
            }
            Err(error) => {
                failures += 1;
                stopped = !continuation;
                output.push(json!({"id":id,"operation":operation,"status":"error","error":error.to_string()}));
            }
        }
    }
    let mut report = json!({"schema":SCHEMA,"package_version":env!("CARGO_PKG_VERSION"),"status":if failures==0{"completed"}else{"partial"},"failed_tasks":failures,"skipped_tasks":skipped,
        "parameters":context.params.source_description(),"ml_model_identity":context.ml_identity.as_ref().map(|m|m.as_json()),
        "conditions":{"electronic_temperature_kelvin":context.electronic.electronic_temperature,"energy_tolerance_hartree":context.electronic.energy_tolerance,
            "charge_tolerance":context.electronic.charge_tolerance,"max_scc":context.electronic.max_scc,"mixing":context.electronic.mixing,
            "scc_accelerator":format!("{:?}",context.electronic.scc_accelerator).to_lowercase(),"kmesh":context.periodic.kmesh.size,"gamma_centered":context.periodic.kmesh.gamma_centered,
            "fold_time_reversal":context.periodic.kmesh.fold_time_reversal,"ao_cutoff_bohr":context.periodic.ao_cutoff,
            "ewald":format!("{:?}",context.periodic.ewald),"nprim":context.electronic.nprim,"coordination_cutoff_bohr":context.electronic.hamiltonian.coordination_cutoff,
            "charge_order":context.electronic.charge_order,"enable_dispersion":context.electronic.enable_dispersion},
        "resources":{"threads":crate::resources::effective_threads(),"workspace_budget_bytes":crate::resources::current().memory_budget_bytes,"cp":crate::resources::current_cp().as_json()},
        "unit_policy":"Geometry uses its explicit coordinate_units; every array/property argument uses the native unit named in its key. Electronic and lattice temperatures are independent.",
        "tasks":[]});
    report["tasks"] = Value::Array(output);
    Ok(report)
}

pub(super) fn run(arguments: Vec<String>) -> Result<()> {
    let mut input = None;
    let mut output = None;
    let mut list = false;
    let mut args = arguments.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("Usage: gfn1-rs properties --request REQUEST.json [--output RESULT.json]\n       gfn1-rs properties --list\n\nShared Rust/pip crystal calculations and native-unit postprocessing.\nUse --threads, --memory-mb and --cp-* for the outer calculation scope.\nSee docs/crystal-cli.md for the request schema, references, units and every operation.");
                return Ok(());
            }
            "--list" => list = true,
            "--request" | "--output" => {
                let target = if arg == "--request" {
                    &mut input
                } else {
                    &mut output
                };
                if target.is_some() {
                    return Err(invalid(format!("duplicate {arg}")));
                }
                *target = Some(
                    args.next()
                        .ok_or_else(|| invalid(format!("{arg} requires a path")))?,
                );
            }
            _ => return Err(invalid(format!("unknown properties argument `{arg}`"))),
        }
    }
    if list {
        if input.is_some() || output.is_some() {
            return Err(invalid("--list cannot be combined with --request/--output"));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&operation_catalog())
                .map_err(|e| invalid(e.to_string()))?
        );
        return Ok(());
    }
    let input = PathBuf::from(input.ok_or_else(|| {
        invalid("properties requires --request REQUEST.json; use --help or --list")
    })?);
    let raw = std::fs::read(&input)?;
    crate::resources::check_workspace(raw.len(), "CLI request bytes")?;
    let request: Value =
        serde_json::from_slice(&raw).map_err(|e| invalid(format!("invalid request JSON: {e}")))?;
    let mut result = execute_request(&request, input.parent().unwrap_or_else(|| Path::new(".")))?;
    result["request_sha256"] = json!(format!("{:x}", Sha256::digest(&raw)));
    let bytes = serde_json::to_vec_pretty(&result).map_err(|e| invalid(e.to_string()))?;
    if let Some(output) = output {
        std::fs::write(output, [bytes.as_slice(), b"\n"].concat())?;
    } else {
        println!(
            "{}",
            String::from_utf8(bytes).map_err(|e| invalid(e.to_string()))?
        );
    }
    if result["status"] == "partial" {
        return Err(invalid(
            "some crystal-property tasks failed; partial results and failure reasons were written",
        ));
    }
    Ok(())
}
