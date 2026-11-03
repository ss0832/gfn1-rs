// SPDX-License-Identifier: GPL-3.0-or-later
//! Explicit external-oracle gate, including WSL-hosted official Linux xtb.
use gfn1_rs::*;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn read_engrad(dir: &Path) -> (f64, Vec<Vec<f64>>) {
    let text = fs::read_to_string(dir.join("input.engrad")).expect("MISSING oracle input.engrad");
    let values: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .collect();
    let n = values[0].parse::<usize>().unwrap();
    let energy = values[1].parse::<f64>().unwrap();
    let gradient = values[2..2 + 3 * n]
        .chunks_exact(3)
        .map(|v| {
            v.iter()
                .map(|x| x.replace(['D', 'd'], "E").parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    (energy, gradient)
}

#[test]
#[ignore = "external oracle: set GFN1_XTB_BIN, GFN1_XTB_REVISION and optionally GFN1_XTB_WSL_DISTRO"]
fn real_molecule_energy_charge_gradient_parity() {
    let bin = std::env::var("GFN1_XTB_BIN").expect("MISSING GFN1_XTB_BIN");
    let revision =
        std::env::var("GFN1_XTB_REVISION").expect("MISSING pinned reference build revision");
    assert!(!revision.trim().is_empty());
    let distro = std::env::var("GFN1_XTB_WSL_DISTRO").ok();
    let command = || {
        let mut cmd = if let Some(d) = &distro {
            let mut c = Command::new("wsl");
            c.args([
                "-d",
                d,
                "--",
                "env",
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                &bin,
            ]);
            c
        } else {
            Command::new(&bin)
        };
        cmd.env("OMP_NUM_THREADS", "1").env("MKL_NUM_THREADS", "1");
        cmd
    };
    let version = command()
        .arg("--version")
        .output()
        .expect("MISSING reference executable");
    assert!(
        version.status.success(),
        "reference --version failed: {}",
        String::from_utf8_lossy(&version.stderr)
    );
    // Official binaries can print the builder's login and hostname. Keep only
    // the scientific program version/revision in distributable records.
    let version = String::from_utf8_lossy(&version.stdout)
        .lines()
        .find(|line| line.contains("xtb version"))
        .expect("reference did not report an xtb version")
        .split(" compiled")
        .next()
        .unwrap()
        .trim()
        .trim_start_matches('*')
        .trim()
        .to_string();
    assert!(
        version.contains(&revision),
        "reference version did not match pinned revision: {version}"
    );
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"));
    let cases=[
        ("h2",include_str!("../examples/h2.xyz"),0.0,0),
        ("water",include_str!("../examples/water.xyz"),0.0,0),
        ("ammonia",include_str!("../examples/ammonia.xyz"),0.0,0),
        ("methane",include_str!("../examples/methane.xyz"),0.0,0),
        ("nh4_plus","5\nammonium\nN 0 0 0\nH .6 .6 .6\nH -.6 -.6 .6\nH -.6 .6 -.6\nH .6 -.6 -.6\n",1.0,0),
        ("oh_minus","2\nhydroxide\nO 0 0 0\nH .98 0 0\n",-1.0,0),
        ("water_dimer","6\nhydrogen-bonded water dimer\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\nO 2.75 .1 0\nH 3.1 .9 0\nH 3.3 -.65 .1\n",0.0,0),
        ("glycine_zwitterion","10\nglycine zwitterionic geometry\nN -1.2 0 0\nC .25 0 0\nC .95 1.32 0\nO .4 2.4 0\nO 2.2 1.25 0\nH -1.55 .5 .8\nH -1.55 .5 -.8\nH -1.55 -.95 0\nH .55 -.53 .9\nH .55 -.53 -.9\n",0.0,0),
        ("oh_radical","2\nOH radical\nO 0 0 0\nH .98 0 0\n",0.0,1),
        ("no_radical","2\nNO radical\nN 0 0 0\nO 1.15 0 0\n",0.0,1),
        ("o2_triplet","2\ntriplet O2\nO 0 0 0\nO 1.21 0 0\n",0.0,2),
    ];
    let models = [
        None,
        Some((SolvationModel::Alpb, "water")),
        Some((SolvationModel::Gbsa, "water")),
        Some((SolvationModel::Alpb, "benzene")),
        Some((SolvationModel::Gbsa, "acetonitrile")),
    ];
    let params = Gfn1Parameters::builtin().unwrap();
    let mut records = Vec::new();
    let mut failures = Vec::new();
    fs::create_dir_all(root.join(".xtb_runs")).unwrap();
    fs::write(
        root.join(".xtb_runs/validation.json"),
        "{\"status\":\"running\",\"executed_cases\":0}",
    )
    .unwrap();
    for (name, xyz, charge, uhf) in cases {
        for model in models {
            if uhf > 0 && model.is_some() {
                continue;
            }
            let id = format!(
                "{name}-{}",
                model.map_or("vacuum".to_string(), |(m, s)| format!("{m:?}-{s}"))
            );
            let dir = root.join(".xtb_runs").join(&id);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("input.xyz"), xyz).unwrap();
            for f in [
                "gradient",
                "charges",
                "xtbout.json",
                "xtbrestart",
                "input.engrad",
            ] {
                let p = dir.join(f);
                if p.exists() {
                    fs::remove_file(p).unwrap();
                }
            }
            // The pinned xtb tblite frontend keeps its default 300 K when
            // --etemp is changed (independently checked at 0/100/300/600 K).
            // Compare spin cases at that actual ensemble, without energy offsets.
            let temperature = if uhf > 0 { 300.0 } else { 0.0 };
            let mut cmd = command();
            cmd.current_dir(&dir).args([
                "input.xyz",
                "--gfn",
                "1",
                "--grad",
                "--json",
                "--norestart",
                "--etemp",
                &format!("{temperature:.0}"),
                "--acc",
                "0.01",
                "--chrg",
                &format!("{charge:.0}"),
                "--uhf",
                &uhf.to_string(),
            ]);
            if let Some((m, s)) = model {
                cmd.args([
                    if m == SolvationModel::Alpb {
                        "--alpb"
                    } else {
                        "--gbsa"
                    },
                    s,
                ]);
            }
            if uhf > 0 {
                cmd.args(["--tblite", "--spinpol"]);
            }
            let output = cmd.output().expect("reference launch failed");
            fs::write(dir.join("stdout.txt"), &output.stdout).unwrap();
            assert!(
                output.status.success(),
                "{id} reference failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if model.is_some() {
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains("gsolv [1 M gas/solution]"),
                    "reference state must match gsolv"
                );
            }
            let (energy, gradient) = read_engrad(&dir);
            let charges: Option<Vec<f64>> = fs::read_to_string(dir.join("charges"))
                .ok()
                .map(|s| s.split_whitespace().map(|x| x.parse().unwrap()).collect());
            assert!(
                uhf > 0 || charges.is_some(),
                "MISSING required scalar oracle charges"
            );
            let sys = PeriodicSystem::from_xyz_str(xyz, charge, false).unwrap();
            let mut opts = ElectronicOptions::default();
            opts.charge = Some(charge);
            opts.electronic_temperature = temperature;
            opts.energy_tolerance = 1e-12;
            opts.charge_tolerance = 1e-10;
            opts.spin_polarization = uhf > 0;
            if uhf > 0 {
                opts.spin_multiplicity = Some(uhf + 1);
            }
            opts.solvation = model.map(|(m, s)| SolvationOptions::new(m, s));
            let rust = analytic_gradient(
                &sys,
                &params,
                AnalyticGradientOptions {
                    electronic: opts,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(rust.electronic_result.converged);
            if let Some(q) = &charges {
                assert_eq!(q.len(), sys.atoms.len());
            }
            assert_eq!(gradient.len(), sys.atoms.len());
            let de = (rust.total_energy - energy).abs();
            let dq = charges.as_ref().map(|charges| {
                rust.electronic_result
                    .atomic_charges
                    .iter()
                    .zip(charges.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
            });
            let dg = rust
                .gradient
                .iter()
                .zip(&gradient)
                .flat_map(|(a, b)| {
                    [a.x, a.y, a.z]
                        .into_iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0.0, f64::max);
            let pass = de.is_finite()
                && dq.is_none_or(f64::is_finite)
                && dg.is_finite()
                && de < 1e-6
                && dq.is_none_or(|x| x < 2e-5)
                && dg < 2e-6;
            eprintln!("{id}: energy={de:.3e} charge={dq:?} gradient={dg:.3e}");
            if !pass {
                if let Some(s) = &rust.electronic_result.solvation {
                    eprintln!(
                        "{id}: solvent at oracle charges {:?}; Born {:?}",
                        s.energy(charges.as_ref().unwrap()).unwrap(),
                        s.born_radii
                    );
                }
                failures.push(id.clone());
            }
            records.push(serde_json::json!({"validation_id":id,"status":if pass {"pass"}else{"fail"},
            "input_sha256":format!("{:x}",Sha256::digest(xyz.as_bytes())),"reference_version":version,
            "reference_archive_sha256":std::env::var("GFN1_XTB_ARCHIVE_SHA256").ok(),
            "parameter_sha256":format!("{:x}",Sha256::digest(params.to_param_string().as_bytes())),
             "charge":charge,"unpaired":uhf,"electronic_temperature_K":temperature,"solvent_temperature_K":298.15,
             "charge_reference_status":if charges.is_some(){"measured"}else{"not_emitted_by_xtb_tblite_backend"},
            "energy_error_hartree":de,"max_charge_error_e":dq,"max_gradient_error_hartree_per_bohr":dg,
            "thresholds":{"energy":1e-6,"charge":2e-5,"gradient":2e-6}}));
        }
    }
    fs::write(root.join(".xtb_runs/validation.json"),serde_json::to_string_pretty(&serde_json::json!({
        "schema":"gfn1-rs-xtb-parity-v1","package_version":env!("CARGO_PKG_VERSION"),
        "status":if failures.is_empty(){"pass"}else{"fail"},"executed_cases":records.len(),"cases":records})).unwrap()).unwrap();
    assert!(!records.is_empty(), "zero executed reference cases");
    assert!(failures.is_empty(), "xtb parity failed: {failures:?}");
}
