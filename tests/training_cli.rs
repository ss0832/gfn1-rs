// SPDX-License-Identifier: GPL-3.0-or-later
//! End-to-end observable fitting, split isolation and relative-energy metrics.
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

fn fixture(name: &str) -> PathBuf {
    let dir = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"))
        .join("training-cli")
        .join(format!(
            "{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    fs::create_dir_all(&dir).unwrap();
    let mut teacher = MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0);
    teacher.heads.get_mut(&8).unwrap().levels[1].weights[0] = 0.006;
    teacher.heads.get_mut(&1).unwrap().correction.weights[2] = 0.001;
    let params = Gfn1Parameters::builtin().unwrap();
    let mut samples = Vec::new();
    let mut energies = Vec::new();
    for (id, split, shift) in [
        ("train", "train", 0.0),
        ("anchor", "reference", 0.12),
        ("held-out", "validation", -0.07),
    ] {
        let xyz = format!(
            "3\nsynthetic observable test\nO .1 .2 0\nH {} .73 .03\nH -.64 .86 -.04\n",
            0.95 + shift
        );
        let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
        let mut options = AnalyticGradientOptions::default();
        options.electronic.electronic_temperature = 0.0;
        options.electronic.energy_tolerance = 1e-12;
        options.electronic.charge_tolerance = 1e-10;
        options.electronic.ml_model = Some(Arc::new(teacher.clone()));
        let result = analytic_gradient(&system, &params, options).unwrap();
        energies.push(result.total_energy);
        samples.push(json!({"id":id,"split":split,"xyz":xyz,"energy_hartree":result.total_energy,
            "forces_hartree_per_bohr":result.gradient.iter().map(|g|(-*g).to_array()).collect::<Vec<_>>(),
            "dipole_e_bohr":result.electronic_result.dipole.to_array()}));
    }
    // Labels on a shared reference must never become direct training targets.
    samples[1]["energy_hartree"] = json!(-999.0);
    let data = json!({"schema":"gfn1-rs-observable-training-v2","energy_units":"hartree","coordinate_units":"angstrom",
        "energy_definition":"mermin_free","electronic_temperature_K":0.0,"provenance":"Independent synthetic CLI test",
        "license_spdx":"GPL-3.0-or-later","reference_method":"synthetic Hamiltonian","independent_reference":false,
        "samples":samples,"relative_energies":[
            {"split":"train","group":"reaction","energy_hartree":energies[0]-energies[1],"terms":[{"sample":"train","coefficient":1},{"sample":"anchor","coefficient":-1}]},
            {"split":"validation","group":"reaction","energy_hartree":energies[2]-energies[1],"terms":[{"sample":"held-out","coefficient":1},{"sample":"anchor","coefficient":-1}]}]});
    fs::write(
        dir.join("data.json"),
        serde_json::to_vec_pretty(&data).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("initial.model"),
        MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0)
            .to_model_string()
            .unwrap(),
    )
    .unwrap();
    dir
}
fn fit(dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_gfn1-rs-train"))
        .arg("fit")
        .arg("--model")
        .arg(dir.join("initial.model"))
        .arg("--data")
        .arg(dir.join("data.json"))
        .arg("--output")
        .arg(dir.join("trained.model"))
        .arg("--report")
        .arg(dir.join("report.json"))
        .args([
            "--epochs",
            "3",
            "--learning-rate",
            "0.0001",
            "--train-correction",
            "--max-model-bytes",
            "5000000",
        ])
        .args(extra)
        .output()
        .unwrap()
}
#[test]
fn observable_cli_trains_and_reports_all_targets_without_reference_label_leakage() {
    let dir = fixture("observables");
    let output = fit(&dir, &[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "complete");
    assert_eq!(report["train"]["structures_evaluated"], 2);
    assert_eq!(report["train"]["absolute_energy_cases"], 1);
    assert_eq!(report["train"]["gradient_components"], 9);
    assert_eq!(report["train"]["dipole_components"], 3);
    assert_eq!(report["weighting"]["normalizers"]["absolute_energy"], 1.0);
    for key in ["energy", "gradient", "dipole", "relative_energy"] {
        assert!(report["training_loss"][0][key].as_f64().unwrap() > 0.0);
    }
    assert!(
        report["training_loss"][2]["total"].as_f64().unwrap()
            < report["training_loss"][0]["total"].as_f64().unwrap()
    );
    assert_eq!(report["validation"]["absolute_energy_cases"], 1);
    assert!(report["held_out_independent_shape_improved"].is_null());
    assert!(report["serialized_model_bytes"].as_u64().unwrap() < 5_000_000);
}
#[test]
fn wtmad2_cli_honors_subset_ratios_and_rejects_training_on_held_out_structures() {
    let dir = fixture("wtmad2");
    let output = fit(
        &dir,
        &[
            "--wtmad2",
            "--group-ratios",
            "reaction:0.25",
            "--gradient-weight",
            "0",
            "--dipole-weight",
            "0",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["weighting"]["energy_weight"], 0.0);
    assert_eq!(report["weighting"]["groups"]["reaction"]["ratio"], 0.25);
    assert_eq!(
        report["validation"]["model_wtmad2_kcal_mol"]["status"],
        "measured"
    );
    let mut data: Value =
        serde_json::from_slice(&fs::read(dir.join("data.json")).unwrap()).unwrap();
    data["relative_energies"][0]["terms"][0]["sample"] = json!("held-out");
    fs::write(dir.join("data.json"), serde_json::to_vec(&data).unwrap()).unwrap();
    let rejected = fit(&dir, &[]);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("cannot use held-out structures"));
}
