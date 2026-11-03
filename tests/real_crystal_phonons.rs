// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended real-material gate. Run explicitly with --ignored --nocapture.
//! Experimental geometries are not silently relaxed or declared stable; raw
//! residual forces, imaginary modes, ASR, symmetry and force-wave FDs are saved.
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
use gfn1_rs::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
const DATA: &str = include_str!("data/real_crystals.json");
fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn array3(value: &Value) -> [usize; 3] {
    std::array::from_fn(|i| value[i].as_u64().unwrap() as usize)
}
fn save(path: &std::path::Path, report: &Value) {
    fs::write(path, serde_json::to_vec_pretty(report).unwrap()).unwrap();
}

fn calculate(case: &Value, params: &Gfn1Parameters) -> Result<Value> {
    let started = Instant::now();
    let system = PeriodicSystem::from_xyz_str(case["xyz"].as_str().unwrap(), 0., false)?;
    let masses: Vec<f64> = case["masses_amu"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let repetitions = array3(&case["repetitions"]);
    let options = ElectronicOptions {
        electronic_temperature: case["electronic_temperature_K"].as_f64().unwrap(),
        energy_tolerance: 1e-11,
        charge_tolerance: 1e-9,
        max_scc: 600,
        mixing: 0.25,
        scc_broyden_size: 200,
        ..Default::default()
    };
    let pbc = PbcOptions {
        ao_cutoff: case["ao_cutoff_bohr"].as_f64().unwrap(),
        ..Default::default()
    };
    println!(
        "real_crystal_stage {} analytic_supercell_hessian",
        case["id"]
    );
    let reference = AnalyticSupercellPhonons::build(
        &system,
        params,
        &options,
        &pbc,
        &masses,
        PhononSupercellOptions {
            repetitions,
            max_atoms: 128,
            memory_budget_bytes: case["memory_budget_bytes"].as_u64().unwrap() as usize,
        },
    )?;
    println!(
        "real_crystal_stage {} reference_gradient elapsed_seconds {}",
        case["id"],
        started.elapsed().as_secs_f64()
    );
    let gradient = pbc_analytic_gradient(&reference.supercell, params, &options, &pbc)?;
    if !gradient.scf.converged
        || gradient
            .gradient
            .iter()
            .any(|v| v.to_array().iter().any(|x| !x.is_finite()))
    {
        return Err(Gfn1Error::InvalidInput(
            "reference force calculation is unconverged or non-finite".into(),
        ));
    }
    let gamma = reference.at_q([0.; 3])?;
    let d = 3 * system.atoms.len();
    let mut acoustic = 0_f64;
    for row in 0..d {
        for axis in 0..3 {
            let sum: f64 = (0..system.atoms.len())
                .map(|a| gamma.dynamical_matrix.re[(row, 3 * a + axis)] * masses[a].sqrt())
                .sum();
            acoustic = acoustic.max(sum.abs());
        }
    }
    let mut translations = Vec::new();
    for x in 0..repetitions[0] {
        for y in 0..repetitions[1] {
            for z in 0..repetitions[2] {
                translations.push([x, y, z]);
            }
        }
    }
    let neighborhood = gfn1_rs::smoothness::NuclearNeighborhood::periodic_scalar(
        &reference.supercell,
        &options,
        &pbc,
    )?;
    let mut q_results = Vec::new();
    let mut largest_error = 0_f64;
    let mut largest_conjugacy = 0_f64;
    let mut largest_hermiticity = 0_f64;
    for q in case["qpoints"].as_array().unwrap() {
        let q: [f64; 3] = std::array::from_fn(|a| q[a].as_f64().unwrap());
        let modes = reference.at_q(q)?;
        let minus = reference.at_q(q.map(|x| -x))?;
        println!(
            "real_crystal_stage {} force_waves q={q:?} elapsed_seconds {}",
            case["id"],
            started.elapsed().as_secs_f64()
        );
        largest_hermiticity = largest_hermiticity.max(modes.raw_hermiticity_error);
        for i in 0..d {
            for j in 0..d {
                largest_conjugacy = largest_conjugacy
                    .max(
                        (modes.dynamical_matrix.re[(i, j)] - minus.dynamical_matrix.re[(i, j)])
                            .abs(),
                    )
                    .max(
                        (modes.dynamical_matrix.im[(i, j)] + minus.dynamical_matrix.im[(i, j)])
                            .abs(),
                    );
            }
        }
        let mut selected = Vec::new();
        for mode in [0, d / 2, d - 1] {
            let mut direction = vec![Vec3::zero(); reference.supercell.atoms.len()];
            let mut norm = 0.;
            let mut imaginary_direction = direction.clone();
            let mut imaginary_norm = 0.;
            for (cell, t) in translations.iter().enumerate() {
                let phase =
                    2. * std::f64::consts::PI * (0..3).map(|a| q[a] * t[a] as f64).sum::<f64>();
                let (sin, cos) = phase.sin_cos();
                for a in 0..system.atoms.len() {
                    let components: [f64; 3] = std::array::from_fn(|k| {
                        cos * modes.mass_weighted_eigenvectors.re[(3 * a + k, mode)]
                            - sin * modes.mass_weighted_eigenvectors.im[(3 * a + k, mode)]
                    });
                    norm += components.iter().map(|v| v * v).sum::<f64>();
                    direction[cell * system.atoms.len() + a] =
                        Vec3::new(components[0], components[1], components[2]) / masses[a].sqrt();
                    let imaginary: [f64; 3] = std::array::from_fn(|k| {
                        sin * modes.mass_weighted_eigenvectors.re[(3 * a + k, mode)]
                            + cos * modes.mass_weighted_eigenvectors.im[(3 * a + k, mode)]
                    });
                    imaginary_norm += imaginary.iter().map(|v| v * v).sum::<f64>();
                    imaginary_direction[cell * system.atoms.len() + a] =
                        Vec3::new(imaginary[0], imaginary[1], imaginary[2]) / masses[a].sqrt();
                }
            }
            // A numerical complex eigenvector may differ by a global phase;
            // select the nonzero real/imaginary wave without fixing that gauge.
            if imaginary_norm > norm {
                direction = imaginary_direction;
                norm = imaginary_norm;
            }
            if norm < 1e-20 {
                return Err(Gfn1Error::InvalidInput(
                    "phonon eigenvector has no real force-wave component".into(),
                ));
            }
            for v in &mut direction {
                *v = *v / norm.sqrt();
            }
            let mut rejected = Vec::new();
            let mut coarse = 2e-3;
            let mut stable = false;
            for _ in 0..12 {
                let mut changed = false;
                for step in [coarse, coarse / 2.] {
                    for sign in [-1., 1.] {
                        let mut trial = reference.supercell.clone();
                        for (atom, v) in trial.atoms.iter_mut().zip(&direction) {
                            atom.position += *v * (sign * step);
                        }
                        let terms = neighborhood.changed_terms(&trial)?;
                        if !terms.is_empty() {
                            changed = true;
                            rejected.push(
                                json!({"step_bohr":step,"sign":sign,"changed_cutoff_terms":terms}),
                            );
                        }
                    }
                }
                if !changed {
                    stable = true;
                    break;
                }
                coarse *= 0.5;
            }
            if !stable {
                return Err(Gfn1Error::InvalidInput(
                    "could not find two finite-displacement steps on the reference cutoff branch"
                        .into(),
                ));
            }
            let mut algebraic = 0.;
            let flat: Vec<_> = direction.iter().flat_map(|v| v.to_array()).collect();
            for x in 0..flat.len() {
                for y in 0..flat.len() {
                    algebraic += flat[x] * reference.cartesian_hessian[(x, y)] * flat[y];
                }
            }
            if (algebraic - modes.eigenvalues_hartree_per_bohr2_amu[mode]).abs() > 1e-8 {
                return Err(Gfn1Error::InvalidInput(
                    "real phonon wave does not reproduce the analytic mode eigenvalue".into(),
                ));
            }
            let mut comparisons = Vec::new();
            for step in [coarse, coarse / 2.] {
                let force = |sign: f64| -> Result<f64> {
                    let mut changed = reference.supercell.clone();
                    for (atom, v) in changed.atoms.iter_mut().zip(&direction) {
                        atom.position += *v * (sign * step);
                    }
                    let result = pbc_analytic_gradient(&changed, params, &options, &pbc)?;
                    if !result.scf.converged {
                        return Err(Gfn1Error::InvalidInput(
                            "force-wave SCF did not converge".into(),
                        ));
                    }
                    let projected: f64 = result
                        .gradient
                        .iter()
                        .zip(&direction)
                        .map(|(g, v)| g.dot(*v))
                        .sum();
                    if !projected.is_finite() {
                        return Err(Gfn1Error::InvalidInput(
                            "non-finite force-wave projection".into(),
                        ));
                    }
                    Ok(projected)
                };
                let fd = (force(1.)? - force(-1.)?) / (2. * step);
                let error = (fd - modes.eigenvalues_hartree_per_bohr2_amu[mode]).abs();
                largest_error = largest_error.max(error);
                comparisons.push(json!({"step_bohr":step,"force_wave_eigenvalue":fd,"absolute_error_hartree_per_bohr2_amu":error}));
            }
            selected.push(json!({"mode":mode,"analytic_eigenvalue_hartree_per_bohr2_amu":modes.eigenvalues_hartree_per_bohr2_amu[mode],"analytic_real_wave_rayleigh":algebraic,
                "smooth_cutoff_branch_verified":stable,"rejected_cutoff_steps":rejected,"normalized_real_force_wave":direction.iter().map(|v|v.to_array()).collect::<Vec<_>>(),"comparisons":comparisons}));
        }
        let imaginary = modes.wavenumbers_cm1.iter().filter(|&&v| v < -1.).count();
        q_results.push(json!({"q_fractional":q,"wavenumbers_cm1":modes.wavenumbers_cm1,"imaginary_modes_below_minus_1_cm1":imaginary,
            "raw_hermiticity_error":modes.raw_hermiticity_error,"selected_modes":selected,"backend":modes.backend}));
    }
    let passed = largest_error < 2e-5
        && reference.raw_hessian_asymmetry < 1e-6
        && acoustic < 1e-6
        && largest_hermiticity < 1e-10
        && largest_conjugacy < 1e-10;
    Ok(
        json!({"id":case["id"],"status":if passed{"pass"}else{"validation_fail"},"primitive_atoms":system.atoms.len(),"supercell_atoms":reference.supercell.atoms.len(),
        "crystal_class":case["crystal_class"],"source":case["source"],"seconds":started.elapsed().as_secs_f64(),"scc_converged":gradient.scf.converged,"ao_cutoff_bohr":pbc.ao_cutoff,
        "electronic_temperature_K":gradient.scf.electronic_temperature,"total_free_hartree":gradient.total_energy,"residual_gradient_max_hartree_per_bohr":gradient.max_gradient,
        "model_stationary_at_1e_minus_5_hartree_per_bohr":gradient.max_gradient<1e-5,"raw_hessian_asymmetry_hartree_per_bohr2":reference.raw_hessian_asymmetry,
        "raw_acoustic_mass_weighted_contraction":acoustic,"max_hermiticity_error":largest_hermiticity,"max_q_minus_q_conjugacy_error":largest_conjugacy,
        "max_force_wave_fd_error_hartree_per_bohr2_amu":largest_error,"q_results":q_results}),
    )
}

#[test]
#[ignore = "extended real-crystal gate; includes a 110-atom organometallic cell with a 6 GiB workspace estimate budget"]
fn ten_real_crystals_have_consistent_analytic_phonons_and_force_waves() {
    let data: Value = serde_json::from_str(DATA).unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    let selected = std::env::var("GFN1_CRYSTAL_ID").ok();
    if let Some(id) = &selected {
        assert!(
            cases.iter().any(|c| c["id"] == *id),
            "unknown diagnostic crystal ID"
        );
    }
    assert!(
        std::env::var_os("GFN1_D3_REFERENCE").is_none(),
        "real-crystal reference gate requires bundled dispersion data"
    );
    let params = Gfn1Parameters::builtin().unwrap();
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"));
    fs::create_dir_all(&root).unwrap();
    let report_path = root.join(selected.as_ref().map_or_else(
        || "real-crystal-phonons-report.json".into(),
        |id| format!("real-crystal-phonons-diagnostic-{id}.json"),
    ));
    let mut report = json!({"schema":"gfn1-rs-crystal-phonon-report-v1","package_version":env!("CARGO_PKG_VERSION"),"status":"running",
        "scope":if selected.is_some(){"diagnostic_subset"}else{"ten_real_crystal_gate"},"dataset_sha256":hash(DATA.as_bytes()),"parameter_sha256":hash(params.to_param_string().as_bytes()),
        "source_fingerprints":{"driver_sha256":hash(include_bytes!("real_crystal_phonons.rs")),"hessian_sha256":hash(include_bytes!("../src/pbc/hessian.rs")),
            "native_response_sha256":hash(include_bytes!("../src/pbc/soc_response.rs")),"reduced_d3_hessian_sha256":hash(include_bytes!("../src/dispersion/reduced_hessian.rs")),
            "neighborhood_sha256":hash(include_bytes!("../src/smoothness.rs"))},
        "model":"standard scalar periodic GFN1; no fitted crystal force constants","sampling":"Gamma analytic supercells; only explicitly commensurate q",
        "density_response_backend":"native-complex max-two scalar matrix-function response with one shared shell-space LU",
        "dispersion_hessian_backend":"exact reduced (R,CN) two-body D3 chain; no full-Cartesian C6 pair-jet table",
        "imaginary_mode_interpretation":"Unrelaxed input structures; preserve negative frequencies and report residual forces rather than assuming GFN1 minima",
        "finite_displacement_branch":"Start at 2e-3/1e-3 bohr, halve together until both signed steps preserve exact reference cutoff-pair identities; record every rejected cutoff step",
        "controls":{"electronic_temperature_K":300.,"ao_cutoff_bohr":"explicit per-fixture value; diffuse Sr requires a longer converged AO sum","energy_tolerance":1e-11,"charge_tolerance":1e-9,"max_scc":600,"mixing":0.25},
        "proposed_thresholds":{"force_wave_error_hartree_per_bohr2_amu":2e-5,"raw_hessian_asymmetry":1e-6,"raw_acoustic_contraction":1e-6,"hermiticity":1e-10,"q_minus_q_conjugacy":1e-10},"cases":[]});
    save(&report_path, &report);
    for case in cases
        .iter()
        .filter(|c| selected.as_ref().is_none_or(|id| c["id"] == *id))
    {
        println!("real_crystal_start {}", case["id"]);
        let result = calculate(case, &params).unwrap_or_else(|error| {
            eprintln!("real_crystal_error {}: {error}", case["id"]);
            // Detailed terminal errors may contain runtime paths. Keep the
            // distributable report free of those strings.
            json!({"id":case["id"],"status":"calculation_error","source":case["source"],"error":"calculation failed; inspect execution output"})
        });
        println!("real_crystal_end {} {}", case["id"], result["status"]);
        report["cases"].as_array_mut().unwrap().push(result);
        save(&report_path, &report);
    }
    let results = report["cases"].as_array().unwrap();
    let executed = results.len();
    let passed = results.iter().filter(|c| c["status"] == "pass").count();
    let expected = if selected.is_some() { 1 } else { 10 };
    report["executed_cases"] = json!(executed);
    report["passed_cases"] = json!(passed);
    report["status"] = json!(if executed != expected || passed != expected {
        "fail"
    } else if selected.is_some() {
        "diagnostic_pass"
    } else {
        "pass"
    });
    save(&report_path, &report);
    assert_eq!(
        executed, expected,
        "zero/missing cases do not constitute validation"
    );
    assert_eq!(
        passed, expected,
        "one or more real-crystal calculations or derivative gates failed"
    );
}
