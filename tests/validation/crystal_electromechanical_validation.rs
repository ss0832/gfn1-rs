// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer diagnostic of the complete analytic, same-SCC bulk response.
//! A completed response is not a claim of mechanical stability or accuracy.
use gfn1_rs::linalg::{symmetric_eigen, Matrix};
use gfn1_rs::pbc::electromechanical::{
    lo_to_modes, pbc_electromechanical_response, ElectromechanicalOptions,
    ElectromechanicalResponse,
};
use gfn1_rs::pbc::polarization::BerryPolarizationOptions;
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::resources::{ResourceContext, ResourceLimits};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        mixing: 0.4,
        ..Default::default()
    }
}

fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
}

fn max_abs(m: &Matrix) -> f64 {
    m.as_slice().iter().map(|x| x.abs()).fold(0., f64::max)
}

fn summary(value: &ElectromechanicalResponse) -> Value {
    let elastic = &value.elastic;
    let stress = max_abs(&elastic.reference_stress);
    let pressure = -(0..3)
        .map(|i| elastic.reference_stress[(i, i)])
        .sum::<f64>()
        / 3.
        * gfn1_rs::pbc::mechanical::STRESS_AU_TO_GPA;
    let curvature = &value.polar_modes.eigenvalues_hartree_per_bohr2_amu;
    let relaxed_curvatures = elastic
        .relaxed_stiffness
        .as_ref()
        .and_then(|c| symmetric_eigen(c).ok())
        .map(|e| e.values);
    json!({
        "scc_converged":elastic.scf.converged,
        "scc_iterations":elastic.scf.iterations,
        "electronic_temperature_kelvin":elastic.scf.electronic_temperature,
        "primitive_volume_bohr3":elastic.reference_system.lattice.unwrap().volume(),
        "free_energy_hartree_per_primitive_cell":elastic.reference_free_energy_hartree,
        "sampled_minimum_band_gap_ev":value.berry.polarization.min_band_gap * gfn1_rs::constants::HARTREE_TO_EV,
        "reference_max_gradient_hartree_per_bohr":elastic.reference_max_gradient,
        "reference_max_stress_hartree_per_bohr3":stress,
        "reference_hydrostatic_pressure_gpa":pressure,
        "stationary_ions_at_default_tolerance":elastic.reference_max_gradient < 1e-6,
        "zero_stress_at_default_tolerance":stress < 1e-6,
        "born_tensors":value.berry.born_effective_charges.iter().map(rows).collect::<Vec<_>>(),
        "born_acoustic_sum_max_e":max_abs(&value.berry.born_acoustic_sum_residual),
        "clamped_piezoelectric_max_c_per_m2":max_abs(&value.berry.clamped_piezoelectric) * gfn1_rs::pbc::polarization::POLARIZATION_AU_TO_C_PER_M2,
        "electronic_relative_dielectric":rows(&value.electronic.relative_dielectric),
        "dielectric_raw_asymmetry":value.electronic.raw_asymmetry,
        "minimum_susceptibility_eigenvalue":value.electronic.minimum_susceptibility_eigenvalue,
        "gamma_wavenumbers_cm1":value.polar_modes.wavenumbers_cm1,
        "negative_gamma_modes_below_curvature_tolerance":curvature.iter().filter(|&&x|x < -1e-10).count(),
        "zero_gamma_modes_within_curvature_tolerance":curvature.iter().filter(|&&x|x.abs() <= 1e-10).count(),
        "ionic_dielectric_available":value.polar_modes.ionic_dielectric.is_some(),
        "ionic_dielectric_error":value.polar_modes.ionic_dielectric_error,
        "relaxed_elastic_curvatures_hartree_per_bohr3":relaxed_curvatures,
        "relaxed_piezoelectric_available":value.relaxed_piezoelectric.is_some(),
        "piezoelectric_strain_available":value.piezoelectric_strain.is_some(),
        "static_dielectric_fixed_strain_available":value.static_dielectric_fixed_strain.is_some(),
        "static_dielectric_free_stress_available":value.static_dielectric_free_stress.is_some(),
        "relaxed_response_error":value.relaxed_error,
        "elastic_raw_max_asymmetry":elastic.raw_max_asymmetry,
        "elastic_translation_coupling_residual":elastic.translation_coupling_residual
    })
}

fn evaluate(
    system: &PeriodicSystem,
    masses: &[f64],
    mesh: [usize; 3],
    cutoff: f64,
    params: &Gfn1Parameters,
) -> Value {
    let started = Instant::now();
    let mut electronic = options();
    let pbc = PbcOptions {
        kmesh: KMesh {
            size: mesh,
            gamma_centered: true,
            fold_time_reversal: false,
        },
        ao_cutoff: cutoff,
        ..Default::default()
    };
    let settings = ElectromechanicalOptions {
        berry: BerryPolarizationOptions::from_kmesh(pbc.kmesh),
        ..Default::default()
    };
    let mut attempts = Vec::new();
    for mixing in [0.4, 0.1, 0.05] {
        electronic.mixing = mixing;
        let attempt = Instant::now();
        match pbc_electromechanical_response(system, params, &electronic, &pbc, masses, &settings) {
            Ok(value) => {
                attempts.push(json!({"mixing":mixing,"status":"completed","elapsed_seconds":attempt.elapsed().as_secs_f64()}));
                let mut nonanalytic = Vec::new();
                for direction in [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]] {
                    match lo_to_modes(
                        &value.elastic.nuclear_hessian,
                        &value.berry.born_effective_charges,
                        &value.electronic.relative_dielectric,
                        value.elastic.reference_system.lattice.unwrap().volume(),
                        masses,
                        direction,
                    ) {
                        Ok(modes) => nonanalytic.push(json!({"status":"completed","direction_cartesian":direction,"response":modes.as_json()})),
                        Err(error) => nonanalytic.push(json!({"status":"failed","direction_cartesian":direction,"error":error.to_string()})),
                    }
                }
                return json!({"status":"analytic_response_completed","attempts":attempts,
                    "elapsed_seconds":started.elapsed().as_secs_f64(),"summary":summary(&value),
                    "response":value.as_json(),"directional_gamma_lo_to":nonanalytic});
            }
            Err(error) => {
                let text = error.to_string();
                attempts.push(json!({"mixing":mixing,"status":"failed","error":text,"elapsed_seconds":attempt.elapsed().as_secs_f64()}));
                // Only the SCC mixer is retried. Geometry, temperature, model,
                // k grid, tolerances and physical guards are never changed.
                if !text.contains("SCC did not converge") {
                    break;
                }
            }
        }
    }
    json!({"status":"failed","attempts":attempts,"elapsed_seconds":started.elapsed().as_secs_f64()})
}

fn representative(case: &Value) -> Result<PeriodicSystem, Box<dyn std::error::Error>> {
    let numbers = case["numbers"].as_array().ok_or("missing numbers")?;
    let positions = case["positions_angstrom"]
        .as_array()
        .ok_or("missing positions")?;
    if numbers.len() != positions.len() {
        return Err("representative atom/position mismatch".into());
    }
    let mut xyz = format!("{}\nLattice=\"", numbers.len());
    for row in case["cell_angstrom"].as_array().ok_or("missing cell")? {
        for x in row.as_array().ok_or("invalid cell row")? {
            xyz.push_str(&format!("{} ", x.as_f64().ok_or("invalid cell entry")?));
        }
    }
    xyz.push_str("\" pbc=\"T T T\"\n");
    for (z, r) in numbers.iter().zip(positions) {
        let symbol = gfn1_rs::z_to_symbol(z.as_u64().ok_or("invalid atomic number")? as u8)
            .ok_or("unknown atomic number")?;
        let r = r.as_array().ok_or("invalid position")?;
        xyz.push_str(&format!("{symbol} {} {} {}\n", r[0], r[1], r[2]));
    }
    Ok(PeriodicSystem::from_xyz_str(&xyz, 0., false)?)
}

fn numbers(values: &Value) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    values
        .as_array()
        .ok_or("missing numeric array")?
        .iter()
        .map(|value| value.as_f64().ok_or_else(|| "invalid number".into()))
        .collect()
}

fn save(path: &Path, report: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)? + "\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let output = args
        .iter()
        .find(|arg| !arg.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("target/validation/v070/crystal-electromechanical-raw.json");
    let selected: Vec<_> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--case="))
        .collect();
    let choose = |name: &str| selected.is_empty() || selected.contains(&name);
    let meshes: Vec<[usize; 3]> = if args.iter().any(|a| a == "--mesh=gamma") {
        vec![[1; 3]]
    } else if args.iter().any(|a| a == "--mesh=2") {
        vec![[2; 3]]
    } else if args.iter().any(|a| a == "--mesh=4") {
        vec![[4; 3]]
    } else {
        vec![[2; 3], [1; 3]]
    };
    let context = ResourceContext::new(ResourceLimits::from_megabytes(Some(2), Some(2048))?)?;
    let params = Gfn1Parameters::builtin()?;
    let source = std::fs::read("tests/data/crystal-dfpt-results.json")?;
    let prior: Value = serde_json::from_slice(&source)?;
    let experimental_source = std::fs::read("tests/data/real_crystals.json")?;
    let fixtures: Value = serde_json::from_slice(&experimental_source)?;
    let mut report = if args.iter().any(|a| a == "--append") && Path::new(output).exists() {
        serde_json::from_slice(&std::fs::read(output)?)?
    } else {
        json!({"schema":"gfn1-rs-crystal-electromechanical-v1","package_version":env!("CARGO_PKG_VERSION"),
            "electronic_temperature_kelvin":300.,"model":"builtin scalar GFN1-xTB; no ML weights in this applicability survey",
            "resources":{"threads":2,"workspace_budget_bytes":2147483648_u64,"limit_scope":"checked workspace estimates, not process RSS"},
            "scc":{"energy_tolerance_hartree":1e-13,"charge_tolerance_e":1e-11,"max_iterations":1000,"mixing_retry_sequence":[0.4,0.1,0.05]},
            "cp_controls":"route defaults; inherited scoped CP settings are unchanged",
            "sampling":"same full Gamma-centred unfolded SCC/Berry grid for every response component; Γ phonons, no q integration",
            "field_functional":"variational discretized Berry electric enthalpy F-Omega*E.P; no reference-cell dipole substitution",
            "geometry_policy":"no relaxation, hidden projection, refit or completion",
            "scope":"actual model calculations; a completed tensor is not a mechanical stability or experimental accuracy pass",
            "representative_input_sha256":format!("{:x}",Sha256::digest(&source)),
            "experimental_fixture_sha256":format!("{:x}",Sha256::digest(&experimental_source)),"cases":[]})
    };
    let mut inputs = Vec::new();
    for case in prior["cases"]
        .as_array()
        .ok_or("missing representative cases")?
    {
        if case["kmesh"] != json!([1, 1, 1]) {
            continue;
        }
        let name = case["name"].as_str().ok_or("missing name")?;
        if ![
            "C-diamond",
            "SiC-zincblende",
            "NaCl-rocksalt",
            "LiF-rocksalt",
        ]
        .contains(&name)
            || !choose(name)
        {
            continue;
        }
        inputs.push((name.to_string(), representative(case)?, numbers(&case["masses_amu"])?, 30.,
            json!({"kind":"specified ideal representative, not acquired experimental CIF","input_fixture":"tests/data/crystal-dfpt-results.json",
                "cell_angstrom":case["cell_angstrom"],"numbers":case["numbers"],"positions_angstrom":case["positions_angstrom"]})));
    }
    if choose("Na2MoO4-experimental") && !args.iter().any(|a| a == "--no-experimental") {
        let case = fixtures["cases"]
            .as_array()
            .ok_or("missing experimental fixtures")?
            .iter()
            .find(|c| c["id"] == "na2moo4")
            .ok_or("missing na2moo4")?;
        inputs.push(("Na2MoO4-experimental".into(),PeriodicSystem::from_xyz_str(case["xyz"].as_str().ok_or("missing experimental XYZ")?,0.,false)?,
            numbers(&case["masses_amu"])?,case["ao_cutoff_bohr"].as_f64().ok_or("missing experimental cutoff")?,
            json!({"kind":"experimental CIF geometry transformed to primitive cell without idealization or relaxation",
                "input_fixture":"tests/data/real_crystals.json","fixture_id":"na2moo4","source":case["source"],"xyz":case["xyz"]})));
    }
    for (name, system, masses, cutoff, provenance) in inputs {
        for &mesh in &meshes {
            let key = format!("{name}-{}{}{}", mesh[0], mesh[1], mesh[2]);
            if args.iter().any(|a| a == "--append")
                && !args.iter().any(|a| a == "--rerun")
                && report["cases"]
                    .as_array()
                    .ok_or("missing output cases")?
                    .iter()
                    .any(|c| c["key"] == key && c["status"] != "running")
            {
                continue;
            }
            let metadata = json!({"key":key,"name":name,"mesh":mesh,"gamma_centered":true,"fold_time_reversal":false,
                "ao_cutoff_bohr":cutoff,"atoms_in_primitive":system.atoms.len(),"masses_amu":masses,"source":provenance,
                "electronic_temperature_kelvin":300.,"status":"running"});
            let previous = report["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["key"] == key && c["status"] != "running")
                .cloned();
            if let Some(old) = previous {
                if report.get("previous_runs").is_none() {
                    report["previous_runs"] = json!([]);
                }
                report["previous_runs"].as_array_mut().unwrap().push(old);
            }
            let index = if let Some(i) = report["cases"]
                .as_array()
                .unwrap()
                .iter()
                .position(|c| c["key"] == key)
            {
                i
            } else {
                report["cases"]
                    .as_array_mut()
                    .unwrap()
                    .push(metadata.clone());
                report["cases"].as_array().unwrap().len() - 1
            };
            report["cases"][index] = metadata;
            save(Path::new(output), &report)?;
            eprintln!(
                "starting {key}, {}atoms, cutoff{cutoff}bohr, Tel300K",
                system.atoms.len()
            );
            let calculation =
                context.run(|| Ok(evaluate(&system, &masses, mesh, cutoff, &params)))?;
            let fields = calculation
                .as_object()
                .ok_or("invalid calculation result")?;
            for (field, value) in fields {
                report["cases"][index][field] = value.clone();
            }
            save(Path::new(output), &report)?;
            eprintln!(
                "finished {key}: {}, {:.1}s, summary={}",
                calculation["status"],
                calculation["elapsed_seconds"].as_f64().unwrap_or(0.),
                calculation
                    .get("summary")
                    .unwrap_or(&calculation["attempts"])
            );
        }
    }
    Ok(())
}
