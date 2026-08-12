// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use gfn1_rs::{
    analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem,
};

struct TestMolecule {
    name: &'static str,
    xyz: &'static str,
}

const TEST_MOLECULES: &[TestMolecule] = &[
    TestMolecule {
        name: "hydrogen",
        xyz: "2\nhydrogen\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n",
    },
    TestMolecule {
        name: "hydrogen fluoride",
        xyz: "2\nhydrogen fluoride\nH 0.000000 0.000000 0.000000\nF 0.917000 0.000000 0.000000\n",
    },
    TestMolecule {
        name: "water",
        xyz: "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
    },
    TestMolecule {
        name: "phosphine",
        xyz: "4\nphosphine\nP 0.000000 0.000000 0.000000\nH 1.193000 0.000000 0.768000\nH -0.596500 1.033300 0.768000\nH -0.596500 -1.033300 0.768000\n",
    },
    TestMolecule {
        name: "ferrocene",
        xyz: "21\nferrocene staggered test geometry\nFe 0.000000 0.000000 0.000000\nC 1.430000 0.000000 1.650000\nC 0.441908 1.360370 1.650000\nC -1.156908 0.840788 1.650000\nC -1.156908 -0.840788 1.650000\nC 0.441908 -1.360370 1.650000\nH 2.510000 0.000000 1.650000\nH 0.775615 2.386978 1.650000\nH -2.030615 1.475161 1.650000\nH -2.030615 -1.475161 1.650000\nH 0.775615 -2.386978 1.650000\nC 1.156908 0.840788 -1.650000\nC -0.441908 1.360370 -1.650000\nC -1.430000 0.000000 -1.650000\nC -0.441908 -1.360370 -1.650000\nC 1.156908 -0.840788 -1.650000\nH 2.030615 1.475161 -1.650000\nH -0.775615 2.386978 -1.650000\nH -2.510000 0.000000 -1.650000\nH -0.775615 -2.386978 -1.650000\nH 2.030615 -1.475161 -1.650000\n",
    },
    TestMolecule {
        name: "borane",
        xyz: "4\nborane\nB 0.000000 0.000000 0.000000\nH 1.190000 0.000000 0.000000\nH -0.595000 1.030570 0.000000\nH -0.595000 -1.030570 0.000000\n",
    },
    TestMolecule {
        name: "caffeine",
        xyz: "24\ncaffeine fixed test geometry\nN 0.000000 0.000000 0.000000\nC 1.250000 0.000000 0.000000\nN 2.000000 1.100000 0.000000\nC 1.250000 2.200000 0.000000\nC 0.000000 2.200000 0.000000\nC -0.700000 1.100000 0.000000\nN 1.750000 3.350000 0.000000\nC 0.750000 4.250000 0.000000\nN -0.350000 3.350000 0.000000\nO 1.900000 -1.050000 0.000000\nO -1.950000 1.100000 0.000000\nC -0.800000 -1.200000 0.250000\nH -1.830000 -0.880000 0.250000\nH -0.550000 -1.780000 1.140000\nH -0.550000 -1.820000 -0.620000\nC 3.450000 1.100000 0.250000\nH 3.800000 2.130000 0.250000\nH 3.780000 0.580000 1.150000\nH 3.850000 0.540000 -0.600000\nC 3.100000 3.900000 0.250000\nH 3.060000 4.990000 0.250000\nH 3.640000 3.580000 1.140000\nH 3.700000 3.520000 -0.580000\nH 0.780000 5.330000 0.000000\n",
    },
    TestMolecule {
        name: "methyl bromide water",
        xyz: "9\nCH3Br water halogen-bond probe\nC 0.000000 0.000000 0.000000\nBr 1.940000 0.000000 0.000000\nH -0.360000 1.020000 0.000000\nH -0.360000 -0.510000 0.883000\nH -0.360000 -0.510000 -0.883000\nO 4.650000 0.120000 0.000000\nH 5.080000 0.740000 0.600000\nH 5.080000 -0.720000 0.250000\nH 4.700000 0.150000 0.960000\n",
    },
];

struct TbliteReference {
    energy: f64,
    gradient: Vec<[f64; 3]>,
    terms: HashMap<&'static str, f64>,
}

#[test]
fn requested_molecules_match_tblite_reference() {
    let Ok(tblite_bin) = std::env::var("GFN1_TBLITE_BIN") else {
        return;
    };

    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let run_root = root.join(".tblite_runs").join("parity");
    fs::create_dir_all(&run_root).unwrap();

    let mut failures = Vec::new();
    for molecule in TEST_MOLECULES {
        let workdir = run_root.join(sanitize_name(molecule.name));
        fs::create_dir_all(&workdir).unwrap();
        let xyz_path = workdir.join("input.xyz");
        fs::write(&xyz_path, molecule.xyz).unwrap();

        let reference = run_tblite(&tblite_bin, &root, &workdir, &xyz_path);
        let system = PeriodicSystem::from_xyz_str(molecule.xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, ElectronicOptions::default())
            .unwrap_or_else(|err| panic!("{} Rust singlepoint failed: {err}", molecule.name));
        let gradient = analytic_gradient(&system, &params, AnalyticGradientOptions::default())
            .unwrap_or_else(|err| panic!("{} Rust gradient failed: {err}", molecule.name));

        check_delta(
            &mut failures,
            molecule.name,
            "total",
            electronic.total_internal,
            reference.energy,
            1.0e-6,
        );
        check_delta(
            &mut failures,
            molecule.name,
            "electronic",
            electronic.electronic_energy
                + electronic.isotropic_scc_energy
                + electronic.third_order_energy,
            reference.terms["electronic"],
            1.0e-6,
        );
        check_delta(
            &mut failures,
            molecule.name,
            "repulsion",
            electronic.repulsion_energy,
            reference.terms["repulsion"],
            1.0e-6,
        );
        check_delta(
            &mut failures,
            molecule.name,
            "dispersion",
            electronic.dispersion_energy,
            reference.terms["dispersion"],
            1.0e-6,
        );
        check_delta(
            &mut failures,
            molecule.name,
            "halogen",
            electronic.halogen_energy,
            reference.terms["halogen"],
            1.0e-6,
        );

        for (iat, (actual, expected)) in gradient
            .gradient
            .iter()
            .zip(reference.gradient.iter())
            .enumerate()
        {
            check_delta(
                &mut failures,
                molecule.name,
                &format!("grad[{iat}].x"),
                actual.x,
                expected[0],
                1.0e-6,
            );
            check_delta(
                &mut failures,
                molecule.name,
                &format!("grad[{iat}].y"),
                actual.y,
                expected[1],
                1.0e-6,
            );
            check_delta(
                &mut failures,
                molecule.name,
                &format!("grad[{iat}].z"),
                actual.z,
                expected[2],
                1.0e-6,
            );
        }
    }

    assert!(
        failures.is_empty(),
        "tblite parity mismatches:\n{}",
        failures.join("\n")
    );
}

fn run_tblite(tblite_bin: &str, root: &Path, workdir: &Path, xyz_path: &Path) -> TbliteReference {
    let output = Command::new(tblite_bin)
        .current_dir(workdir)
        .env("PATH", tblite_path(root))
        .args([
            "run",
            xyz_path.to_str().unwrap(),
            "--method",
            "gfn1",
            "--no-restart",
            "--json",
            "tblite.json",
            "--grad",
            "tblite.grad",
            "-v",
        ])
        .output()
        .unwrap_or_else(|err| panic!("failed to run tblite: {err}"));

    if !output.status.success() {
        panic!(
            "tblite failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = fs::read_to_string(workdir.join("tblite.json")).unwrap();
    TbliteReference {
        energy: json_scalar(&json, "energy"),
        gradient: json_gradient(&json),
        terms: parse_terms(&stdout),
    }
}

fn tblite_path(root: &Path) -> String {
    let mut entries = vec![
        root.join(".tblite_alias"),
        PathBuf::from("C:\\TDM-GCC-64\\bin"),
    ];
    if let Ok(path) = std::env::var("PATH") {
        entries.extend(std::env::split_paths(&path));
    }
    std::env::join_paths(entries)
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

fn parse_terms(stdout: &str) -> HashMap<&'static str, f64> {
    let mut terms = HashMap::new();
    for line in stdout.lines() {
        if line.contains("halogen bonding energy") {
            terms.insert("halogen", parse_first_float(line));
        } else if line.contains("repulsion energy") {
            terms.insert("repulsion", parse_first_float(line));
        } else if line.contains("dispersion energy") {
            terms.insert("dispersion", parse_first_float(line));
        } else if line.contains("electronic energy") {
            terms.insert("electronic", parse_first_float(line));
        }
    }
    for key in ["halogen", "repulsion", "dispersion", "electronic"] {
        assert!(
            terms.contains_key(key),
            "tblite output did not contain {key}"
        );
    }
    terms
}

fn parse_first_float(line: &str) -> f64 {
    line.split_whitespace()
        .find_map(|token| token.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no float in tblite line `{line}`"))
}

fn json_scalar(json: &str, key: &str) -> f64 {
    let label = format!("\"{key}\"");
    let start = json
        .find(&label)
        .unwrap_or_else(|| panic!("missing key {key}"));
    let rest = &json[start + label.len()..];
    let colon = rest.find(':').unwrap();
    parse_leading_float(&rest[colon + 1..])
}

fn json_gradient(json: &str) -> Vec<[f64; 3]> {
    let label = "\"gradient\"";
    let start = json
        .find(label)
        .unwrap_or_else(|| panic!("missing key gradient"));
    let rest = &json[start + label.len()..];
    let open = rest.find('[').unwrap();
    let close = rest[open + 1..].find(']').unwrap() + open + 1;
    let values = rest[open + 1..close]
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            (!trimmed.is_empty()).then(|| trimmed.parse::<f64>().unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(values.len() % 3, 0);
    values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn parse_leading_float(input: &str) -> f64 {
    let mut end = 0;
    for (idx, ch) in input.char_indices() {
        if ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E') {
            end = idx + ch.len_utf8();
        } else if end > 0 {
            break;
        }
    }
    input[..end].trim().parse::<f64>().unwrap()
}

fn check_delta(
    failures: &mut Vec<String>,
    molecule: &str,
    term: &str,
    actual: f64,
    expected: f64,
    tolerance: f64,
) {
    let delta = actual - expected;
    if delta.abs() > tolerance {
        failures.push(format!(
            "{molecule} {term}: rust={actual:.12e} tblite={expected:.12e} delta={delta:.3e}"
        ));
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}
