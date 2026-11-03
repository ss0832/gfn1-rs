// SPDX-License-Identifier: GPL-3.0-or-later
//! Actual executions of the analytic mechanical API. This diagnostic retains
//! failed model/sampling cases as failures; writing a report is not a pass gate.
use gfn1_rs::pbc::elastic_response::{
    pbc_elastic_response, pbc_elastic_response_with_guess, AnalyticElasticOptions,
};
use gfn1_rs::pbc::mechanical::{
    elastic_properties, pbc_equation_of_state_with_guess, pbc_relax, PeriodicRelaxOptions,
};
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

fn parameters() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        mixing: 0.4,
        logger: std::env::args().find_map(|argument| {
            argument.strip_prefix("--profile=").map(|path| {
                gfn1_rs::logging::CalculationLogger::append(path, gfn1_rs::LogLevel::Info, false)
                    .unwrap()
            })
        }),
        ..Default::default()
    }
}
fn mixing_candidates() -> Vec<f64> {
    std::env::args()
        .find_map(|a| {
            a.strip_prefix("--mixing=")
                .map(|v| vec![v.parse::<f64>().expect("numeric mixing")])
        })
        .unwrap_or_else(|| vec![0.4, 0.1, 0.05])
}
fn representative(case: &Value) -> PeriodicSystem {
    let mut xyz = String::from("2\nLattice=\"");
    for row in case["cell_angstrom"].as_array().unwrap() {
        for x in row.as_array().unwrap() {
            xyz.push_str(&format!("{} ", x.as_f64().unwrap()));
        }
    }
    xyz.push_str("\" pbc=\"T T T\"\n");
    for (z, r) in case["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["positions_angstrom"].as_array().unwrap())
    {
        let sym = gfn1_rs::z_to_symbol(z.as_u64().unwrap() as u8).unwrap();
        let r = r.as_array().unwrap();
        xyz.push_str(&format!("{sym} {} {} {}\n", r[0], r[1], r[2]));
    }
    PeriodicSystem::from_xyz_str(&xyz, 0., false).unwrap()
}
fn execute(
    name: &str,
    system: &PeriodicSystem,
    masses: &[f64],
    mesh: [usize; 3],
    params: &Gfn1Parameters,
    relax_cell: bool,
    cutoff: f64,
) -> Value {
    if std::env::args().any(|a| a == "--forces-only") {
        return force_supplement(name, system, mesh, params, cutoff);
    }
    let start = Instant::now();
    let mut electronic = parameters();
    let pbc = PbcOptions {
        kmesh: KMesh {
            size: mesh,
            gamma_centered: true,
            fold_time_reversal: true,
        },
        ao_cutoff: cutoff,
        ..Default::default()
    };
    let mut attempts = Vec::new();
    let mut solved = None;
    for mixing in mixing_candidates() {
        electronic.mixing = mixing;
        match pbc_elastic_response(
            system,
            params,
            &electronic,
            &pbc,
            &AnalyticElasticOptions::default(),
        ) {
            Ok(value) => {
                attempts.push(json!({"mixing":mixing,"status":"converged"}));
                solved = Some(value);
                break;
            }
            Err(error) => {
                let text = error.to_string();
                attempts.push(json!({"mixing":mixing,"error":text}));
                if !text.contains("SCC did not converge") {
                    break;
                }
            }
        }
    }
    let Some(reference) = solved else {
        return json!({"name":name,"mesh":mesh,"status":"failed","attempts":attempts,"elapsed_seconds":start.elapsed().as_secs_f64()});
    };
    let mut result = json!({"name":name,"mesh":mesh,"status":"analytic_response_completed","temperature_kelvin":300.,"ao_cutoff_bohr":cutoff,"attempts":attempts,"reference":reference.as_json()});
    eprintln!(
        "{name} {mesh:?}: analytic C/Lambda done in {:.1}s, raw asym {}, P={} GPa",
        start.elapsed().as_secs_f64(),
        reference.raw_max_asymmetry,
        -(0..3)
            .map(|i| reference.reference_stress[(i, i)])
            .sum::<f64>()
            / 3.
            * gfn1_rs::pbc::mechanical::STRESS_AU_TO_GPA
    );
    if relax_cell {
        let options = PeriodicRelaxOptions {
            relax_cell: true,
            max_iterations: 400,
            ..Default::default()
        };
        match pbc_relax(system, params, &electronic, &pbc, &options) {
            Err(error) => result["relaxation_error"] = json!(error.to_string()),
            Ok(relaxed) => {
                result["relaxation"] = relaxed.as_json();
                eprintln!(
                    "{name}: variable-cell {} {}steps E={} V={}",
                    relaxed.termination,
                    relaxed.iterations,
                    relaxed.energy_hartree,
                    relaxed.system.lattice.unwrap().volume()
                );
                if relaxed.converged {
                    match pbc_elastic_response_with_guess(
                        &relaxed.system,
                        params,
                        &electronic,
                        &pbc,
                        &AnalyticElasticOptions::default(),
                        Some(&relaxed.scf_guess),
                    ) {
                        Err(error) => {
                            result["equilibrium_elastic_error"] = json!(error.to_string())
                        }
                        Ok(elastic) => {
                            result["equilibrium_elastic"] = elastic.as_json();
                            let volume = relaxed.system.lattice.unwrap().volume();
                            let bohr_m = gfn1_rs::constants::BOHR_TO_ANGSTROM * 1e-10;
                            let density = masses.iter().sum::<f64>() * 1.660_539_066_60e-27
                                / (volume * bohr_m.powi(3));
                            result["equilibrium_density_kg_m3"] = json!(density);
                            if let Some(c) = &elastic.relaxed_stiffness {
                                match elastic_properties(c, density) {
                                    Ok(value) => result["equilibrium_moduli"] = json!(value),
                                    Err(error) => {
                                        result["equilibrium_moduli_error"] =
                                            json!(error.to_string())
                                    }
                                }
                            }
                        }
                    }
                    match pbc_equation_of_state_with_guess(
                        &relaxed.system,
                        params,
                        &electronic,
                        &pbc,
                        &[0.94, 0.97, 1., 1.03, 1.06],
                        true,
                        &PeriodicRelaxOptions::default(),
                        Some(&relaxed.scf_guess),
                    ) {
                        Ok(value) => result["equilibrium_eos"] = value.as_json(),
                        Err(error) => result["equilibrium_eos_error"] = json!(error.to_string()),
                    }
                }
            }
        }
    }
    result["elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    result
}
fn force_supplement(
    name: &str,
    system: &PeriodicSystem,
    mesh: [usize; 3],
    params: &Gfn1Parameters,
    cutoff: f64,
) -> Value {
    let start = Instant::now();
    let mut electronic = parameters();
    let pbc = PbcOptions {
        kmesh: KMesh {
            size: mesh,
            gamma_centered: true,
            fold_time_reversal: true,
        },
        ao_cutoff: cutoff,
        ..Default::default()
    };
    let mut attempts = Vec::new();
    for mixing in mixing_candidates() {
        electronic.mixing = mixing;
        match gfn1_rs::pbc::pbc_analytic_gradient(system, params, &electronic, &pbc) {
            Ok(gradient) => {
                return json!({"name":name,"mesh":mesh,"status":"force_supplement_completed","temperature_kelvin":300.,"ao_cutoff_bohr":cutoff,"mixing":mixing,"attempts":attempts,
                "reference_free_energy_hartree":gradient.total_energy,"reference_max_gradient_hartree_per_bohr":gradient.max_gradient,
                "reference_gradient_hartree_per_bohr":gradient.gradient.iter().map(|g|g.to_array()).collect::<Vec<_>>(),
                "reference_forces_hartree_per_bohr":gradient.gradient.iter().map(|g|g.to_array().map(|x|-x)).collect::<Vec<_>>(),
                "scf_iterations":gradient.scf.iterations,"elapsed_seconds":start.elapsed().as_secs_f64()})
            }
            Err(error) => {
                let text = error.to_string();
                attempts.push(json!({"mixing":mixing,"error":text}));
                if !text.contains("SCC did not converge") {
                    break;
                }
            }
        }
    }
    json!({"name":name,"mesh":mesh,"status":"failed","attempts":attempts})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let no_relax = std::env::args().any(|a| a == "--no-relax");
    let selected: Vec<String> = std::env::args()
        .filter_map(|a| a.strip_prefix("--case=").map(str::to_string))
        .collect();
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "target/validation/v070/crystal-mechanics-raw.json".into()),
    );
    let actual_only = std::env::args().any(|a| a == "--actual-only");
    let representatives_only = std::env::args().any(|a| a == "--representatives-only");
    let context = gfn1_rs::resources::ResourceContext::new(
        gfn1_rs::resources::ResourceLimits::from_megabytes(Some(2), Some(2048))?,
    )?;
    let params = Gfn1Parameters::builtin()?;
    let source = std::fs::read("tests/data/crystal-dfpt-results.json")?;
    let prior: Value = serde_json::from_slice(&source)?;
    let mut report = json!({"schema":"gfn1-rs-real-crystal-mechanics-v1","package_version":env!("CARGO_PKG_VERSION"),"temperature_kelvin":300.,"representative_input_sha256":format!("{:x}",Sha256::digest(&source)),"representative_inputs":"Specified ideal archetypes with representative lattice constants; not newly acquired experimental CIFs","cases":[]});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for case in prior["cases"].as_array().unwrap() {
        if actual_only {
            break;
        }
        if case["kmesh"] != json!([1, 1, 1]) {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        if !selected.is_empty() && !selected.iter().any(|value| value == name) {
            continue;
        }
        let system = representative(case);
        let masses: Vec<_> = case["masses_amu"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        for mesh in [[1, 1, 1], [2, 2, 2]] {
            if std::env::args().any(|argument| argument == "--gamma-only") && mesh != [1, 1, 1] {
                continue;
            }
            let mut value = context.run(|| {
                Ok(execute(
                    name,
                    &system,
                    &masses,
                    mesh,
                    &params,
                    mesh == [1, 1, 1] && !no_relax,
                    30.,
                ))
            })?;
            value["source_kind"] = json!("specified ideal representative");
            report["cases"].as_array_mut().unwrap().push(value);
            std::fs::write(&output, serde_json::to_string_pretty(&report)? + "\n")?;
        }
    }
    if representatives_only {
        return Ok(());
    }
    let real_bytes = std::fs::read("tests/data/real_crystals.json")?;
    let real: Value = serde_json::from_slice(&real_bytes)?;
    let entries = real
        .as_array()
        .or_else(|| real["cases"].as_array())
        .or_else(|| real["crystals"].as_array())
        .or_else(|| real["structures"].as_array())
        .ok_or("unknown real-crystal fixture layout")?;
    let case = entries
        .iter()
        .find(|c| c["id"] == "na2moo4")
        .ok_or("missing na2moo4 fixture")?;
    let xyz = case["xyz"].as_str().ok_or("missing actual crystal XYZ")?;
    let system = PeriodicSystem::from_xyz_str(xyz, 0., false)?;
    let masses: Vec<_> = case["masses_amu"]
        .as_array()
        .ok_or("missing crystal masses")?
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let mut value = context.run(|| {
        Ok(execute(
            "Na2MoO4 COD2240655",
            &system,
            &masses,
            [1, 1, 1],
            &params,
            false,
            24.,
        ))
    })?;
    value["source_kind"] = json!("experimental COD structure");
    value["source_fixture"] = case.clone();
    report["cases"].as_array_mut().unwrap().push(value);
    report["experimental_input_sha256"] = json!(format!("{:x}", Sha256::digest(&real_bytes)));
    std::fs::write(&output, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "Wrote {} actual execution records to {}",
        report["cases"].as_array().unwrap().len(),
        output.display()
    );
    Ok(())
}
