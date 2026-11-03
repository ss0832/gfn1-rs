// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::*;
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
fn path(name: &str) -> PathBuf {
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"))
        .join("logging");
    fs::create_dir_all(&root).unwrap();
    root.join(format!(
        "{name}-{}.jsonl",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
fn records(path: &PathBuf) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
#[test]
fn api_logging_records_progress_and_keeps_the_same_energy_and_forces() {
    let system =
        PeriodicSystem::from_xyz_str(include_str!("../examples/water.xyz"), 0.0, false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let plain = analytic_gradient(&system, &params, Default::default()).unwrap();
    let file = path("water");
    let mut options = AnalyticGradientOptions::default();
    options.electronic.logger =
        Some(CalculationLogger::append(&file, LogLevel::Info, false).unwrap());
    let logged = analytic_gradient(&system, &params, options).unwrap();
    assert!((plain.total_energy - logged.total_energy).abs() < 1e-13);
    for (a, b) in plain.gradient.iter().zip(&logged.gradient) {
        assert!((*a - *b).norm() < 1e-12);
    }
    let events = records(&file);
    assert!(events
        .iter()
        .any(|r| r["event"] == "scf_iteration" && r["data"]["charge_rms"].is_number()));
    let id = logged.electronic_result.calculation_id.unwrap();
    assert!(events.iter().any(|r| r["operation_id"] == id
        && r["event"] == "operation_end"
        && r["data"]["status"] == "success"));
    assert!(
        events
            .iter()
            .any(|r| r["data"]["operation"] == "analytic_gradient"
                && r["data"]["status"] == "success")
    );
    let text = fs::read_to_string(file).unwrap();
    if let Ok(home) = std::env::var("USERPROFILE") {
        assert!(!text.contains(&home));
    }
}
#[test]
fn failures_are_structured_and_sensitive_error_tokens_are_redacted() {
    let file = path("failure");
    let logger = CalculationLogger::append(&file, LogLevel::Debug, false).unwrap();
    let span = logger.begin("input_test", serde_json::json!({})).unwrap();
    span.failure(&Gfn1Error::InvalidInput(
        "cannot load Z:\\example-private\\input.xyz".into(),
    ))
    .unwrap();
    let events = records(&file);
    let error = events
        .iter()
        .find(|r| r["data"]["status"] == "error")
        .unwrap();
    assert_eq!(error["data"]["error_kind"], "invalid_input");
    let text = fs::read_to_string(file).unwrap();
    assert!(!text.contains("example-private"));
    assert!(text.contains("<redacted>"));
}
#[test]
fn cli_writes_machine_readable_calculation_events() {
    let file = path("cli");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_gfn1-rs"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "examples/h2.xyz",
            "--gradient",
            "--log-level",
            "debug",
            "--log-file",
        ])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = records(&file);
    assert!(events.iter().any(|r| r["event"] == "scf_iteration"));
    assert!(events
        .iter()
        .any(|r| r["event"] == "operation_end" && r["data"]["operation"] == "analytic_gradient"));
    assert!(!fs::read_to_string(file)
        .unwrap()
        .contains("examples/h2.xyz"));
}
#[test]
fn numerical_api_failure_is_logged_before_returning_the_error() {
    let file = path("invalid");
    let params = Gfn1Parameters::builtin().unwrap();
    let result = run_electronic(
        &PeriodicSystem::new(vec![], None),
        &params,
        ElectronicOptions {
            logger: Some(CalculationLogger::append(&file, LogLevel::Info, false).unwrap()),
            ..Default::default()
        },
    );
    assert!(result.is_err());
    let events = records(&file);
    assert!(events
        .iter()
        .any(|r| r["event"] == "operation_end" && r["data"]["status"] == "error"));
    assert!(!events.iter().any(|r| r["event"] == "operation_unfinished"));
}
#[test]
fn concurrent_api_calls_append_complete_json_lines_with_unique_ids() {
    let file = path("parallel");
    let logger = CalculationLogger::append(&file, LogLevel::Info, false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let system =
        PeriodicSystem::from_xyz_str(include_str!("../examples/h2.xyz"), 0.0, false).unwrap();
    let ids = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let logger = logger.clone();
                let p = &params;
                let s = &system;
                scope.spawn(move || {
                    run_electronic(
                        s,
                        p,
                        ElectronicOptions {
                            logger: Some(logger),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .calculation_id
                    .unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_ne!(ids[0], ids[1]);
    let events = records(&file);
    for id in ids {
        assert!(events
            .iter()
            .any(|r| r["operation_id"] == id && r["data"]["status"] == "success"));
    }
}
