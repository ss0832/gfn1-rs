// SPDX-License-Identifier: GPL-3.0-or-later
//! Calculation CLI routes periodic ML/SOC properties to their actual backends.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
fn fixture(name: &str) -> PathBuf {
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"))
        .join(format!(
            "periodic-cli-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("water.xyz"),"3\nLattice=\"6 0 0 .2 6.3 0 0 .1 6.6\" pbc=\"T T T\"\nO 0 0 0\nH .958 0 0\nH -.25 .925 .03\n").unwrap();
    root
}
#[test]
fn cli_periodic_ml_hessian_and_ao_control_are_available() {
    let root = fixture("ml");
    let output = Command::new(env!("CARGO_BIN_EXE_gfn1-rs"))
        .arg(root.join("water.xyz"))
        .args([
            "--ml-xtb",
            "builtin:water-cluster",
            "--dfpt",
            "--pbc-ao-cutoff",
            "18",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("hessian_size"));
    assert!(text.contains("hessian_raw_asymmetry"));
    assert!(text.contains("ml_model builtin:water-clusters"));
    assert!(text.contains("hessian 9 9"));
}
#[test]
fn cli_periodic_soc_force_stress_dfpt_and_logging_are_connected() {
    let root = fixture("soc");
    fs::write(root.join("zeta.txt"),"units hartree\nprovenance Independent synthetic CLI test; not a literature atomic constant\n8 1 0.005\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gfn1-rs"))
        .arg(root.join("water.xyz"))
        .arg("--soc-zeta-file")
        .arg(root.join("zeta.txt"))
        .args([
            "--hessian",
            "--gradient",
            "--stress",
            "--citations",
            "--pbc-ao-cutoff",
            "18",
        ])
        .arg("--log-file")
        .arg(root.join("soc.jsonl"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    for key in [
        "spinor_count_per_k",
        "occupation_limit",
        "hessian_hartree_per_bohr2",
        "gradient",
        "stress_hartree_per_bohr3",
        "Jha2023SOC",
    ] {
        assert!(text.contains(key), "missing {key}");
    }
    let records: Vec<serde_json::Value> = fs::read_to_string(root.join("soc.jsonl"))
        .unwrap()
        .lines()
        .map(|v| serde_json::from_str(v).unwrap())
        .collect();
    assert!(records
        .iter()
        .any(|r| r["data"]["operation"] == "periodic_soc_dfpt_hessian"
            && r["data"]["status"] == "success"));
    let optical = Command::new(env!("CARGO_BIN_EXE_gfn1-rs"))
        .arg(root.join("water.xyz"))
        .args(["--td-dfpt", "--n-states", "2", "--soc"])
        .arg(root.join("zeta.txt"))
        .args([
            "--pbc-ao-cutoff",
            "18",
            "--kmesh",
            "2,1,1",
            "--threads",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        optical.status.success(),
        "{}",
        String::from_utf8_lossy(&optical.stderr)
    );
    let text = String::from_utf8(optical.stdout).unwrap();
    assert!(text.contains("td_dfpt_kpoints 2"));
    assert!(text.contains("excitation_Hartree"));
}
