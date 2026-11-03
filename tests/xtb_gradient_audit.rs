// SPDX-License-Identifier: GPL-3.0-or-later
//! Independently audit the oracle's gradient against its own energy surface.
use gfn1_rs::*;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn reference(
    system: &PeriodicSystem,
    dir: &Path,
    solvent: Option<(SolvationModel, &str)>,
) -> (f64, Vec<f64>) {
    fs::create_dir_all(dir).unwrap();
    let mut xyz = format!("{}\ngradient oracle audit\n", system.atoms.len());
    for atom in &system.atoms {
        let r = atom.position * gfn1_rs::constants::BOHR_TO_ANGSTROM;
        xyz.push_str(&format!(
            "{} {:.16} {:.16} {:.16}\n",
            z_to_symbol(atom.z).unwrap(),
            r.x,
            r.y,
            r.z
        ));
    }
    fs::write(dir.join("input.xyz"), xyz).unwrap();
    let path = dir.join("input.engrad");
    if path.exists() {
        fs::remove_file(&path).unwrap();
    }
    let bin = std::env::var("GFN1_XTB_BIN").expect("MISSING GFN1_XTB_BIN");
    let mut command = if let Ok(distro) = std::env::var("GFN1_XTB_WSL_DISTRO") {
        let mut c = Command::new("wsl");
        c.args([
            "-d",
            &distro,
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
    command.current_dir(dir).args([
        "input.xyz",
        "--gfn",
        "1",
        "--grad",
        "--norestart",
        "--etemp",
        "0",
        "--acc",
        "0.01",
    ]);
    if let Some((model, name)) = solvent {
        command.args([
            if model == SolvationModel::Alpb {
                "--alpb"
            } else {
                "--gbsa"
            },
            name,
        ]);
    }
    let output = command.output().expect("oracle launch failed");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::write(dir.join("stdout.txt"), &output.stdout).unwrap();
    let text = fs::read_to_string(path).expect("MISSING oracle engrad");
    let values: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .collect();
    let n = values[0].parse::<usize>().unwrap();
    assert_eq!(n, system.atoms.len());
    (
        values[1].parse().unwrap(),
        values[2..2 + 3 * n]
            .iter()
            .map(|x| x.replace(['D', 'd'], "E").parse().unwrap())
            .collect(),
    )
}

#[test]
#[ignore = "external reference self-consistency audit; configure pinned GFN1_XTB_BIN and optional WSL distro"]
fn legacy_reference_transverse_gradients_are_checked_against_reference_energy_fd() {
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gfn1-rs-validation"))
        .join("gradient-audit");
    let params = Gfn1Parameters::builtin().unwrap();
    let mut records = Vec::new();
    let cases=[("water_dimer","6\nwater dimer\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\nO 2.75 .1 0\nH 3.1 .9 0\nH 3.3 -.65 .1\n"),
        ("glycine","10\nglycine\nN -1.2 0 0\nC .25 0 0\nC .95 1.32 0\nO .4 2.4 0\nO 2.2 1.25 0\nH -1.55 .5 .8\nH -1.55 .5 -.8\nH -1.55 -.95 0\nH .55 -.53 .9\nH .55 -.53 -.9\n")];
    for (name, xyz) in cases {
        for solvent in [
            None,
            Some((SolvationModel::Alpb, "water")),
            Some((SolvationModel::Gbsa, "water")),
            Some((SolvationModel::Alpb, "benzene")),
            Some((SolvationModel::Gbsa, "acetonitrile")),
        ] {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let mut options = ElectronicOptions {
                electronic_temperature: 0.0,
                energy_tolerance: 1e-12,
                charge_tolerance: 1e-10,
                ..Default::default()
            };
            options.solvation = solvent.map(|(model, name)| SolvationOptions::new(model, name));
            let rust = analytic_gradient(
                &system,
                &params,
                AnalyticGradientOptions {
                    electronic: options.clone(),
                    ..Default::default()
                },
            )
            .unwrap();
            let dir = root.join(format!(
                "{name}-{}",
                solvent.map_or("vacuum".into(), |(model, name)| format!("{model:?}-{name}"))
            ));
            let (reference_energy, reference_gradient) =
                reference(&system, &dir.join("base"), solvent);
            let gradient: Vec<f64> = rust.gradient.iter().flat_map(|g| g.to_array()).collect();
            let coordinate = (0..gradient.len())
                .max_by(|&a, &b| {
                    (gradient[a] - reference_gradient[a])
                        .abs()
                        .total_cmp(&(gradient[b] - reference_gradient[b]).abs())
                })
                .unwrap();
            // Upstream tracking: xtb 6.7.1 gradient discrepancy due to S=0 screening omission.
            // Pinned legacy scf_module.F90 excludes |S|<neglect from matlist2,
            // then reconstructs H0/S only on that list. The noreset gradient still
            // evaluates dS for every pair, so these H0*dS terms are lost.
            let e = &rust.electronic_result;
            let basis = &e.basis;
            let ds = gfn1_rs::response::cpxtb::overlap_first_derivative_matrix(
                &system, basis, coordinate,
            )
            .unwrap();
            let mut missing_h0_ds = 0.0;
            for i in 0..basis.len() {
                let a = &basis.shells[basis.aos[i].shell_index];
                for j in 0..i {
                    let b = &basis.shells[basis.aos[j].shell_index];
                    if a.atom_index == b.atom_index || e.integrals.overlap[(i, j)].abs() >= 1e-10 {
                        continue;
                    }
                    let self_a = a.hdiag_ha
                        - a.kcn_raw.unwrap_or(0.0) * e.coordination_numbers[a.atom_index];
                    let self_b = b.hdiag_ha
                        - b.kcn_raw.unwrap_or(0.0) * e.coordination_numbers[b.atom_index];
                    let r = (system.atoms[a.atom_index].position
                        - system.atoms[b.atom_index].position)
                        .norm();
                    let rad = gfn1_rs::data_tables::atomic_radius_bohr(a.z).unwrap()
                        + gfn1_rs::data_tables::atomic_radius_bohr(b.z).unwrap();
                    let hij = 0.5
                        * (self_a + self_b)
                        * gfn1_rs::hamiltonian::hscale(a, b, &params).unwrap()
                        * gfn1_rs::hamiltonian::shell_polynomial(a, b, (r / rad).sqrt());
                    missing_h0_ds += 2.0 * e.density[(i, j)] * hij * ds[(i, j)];
                }
            }
            eprintln!("{name} solvent={solvent:?}: observed gradient difference={:e}, source-predicted missing H0*dS={missing_h0_ds:e}",gradient[coordinate]-reference_gradient[coordinate]);
            assert!(
                (gradient[coordinate] - reference_gradient[coordinate] - missing_h0_ds).abs()
                    < 5e-7,
                "the identified source omission does not explain the discrepancy"
            );
            let mut derivatives = Vec::new();
            for (index, h) in [4e-4, 2e-4].into_iter().enumerate() {
                let mut energies = [0.0; 2];
                let mut rust_energies = [0.0; 2];
                for (i, sign) in [1.0, -1.0].into_iter().enumerate() {
                    let mut displaced = system.clone();
                    let r = &mut displaced.atoms[coordinate / 3].position;
                    match coordinate % 3 {
                        0 => r.x += sign * h,
                        1 => r.y += sign * h,
                        _ => r.z += sign * h,
                    }
                    energies[i] = reference(
                        &displaced,
                        &dir.join(format!("step-{index}-side-{i}")),
                        solvent,
                    )
                    .0;
                    rust_energies[i] = run_electronic(&displaced, &params, options.clone())
                        .unwrap()
                        .total_free;
                }
                let fd = (energies[0] - energies[1]) / (2.0 * h);
                let own = (rust_energies[0] - rust_energies[1]) / (2.0 * h);
                eprintln!("{name} solvent={solvent:?} dof={coordinate} h={h:e}: Rust analytic={:e}, official analytic={:e}, official energy FD={fd:e}, Rust energy FD={own:e}",gradient[coordinate],reference_gradient[coordinate]);
                derivatives.push(json!({"step_bohr":h,"official_energy_fd":fd,"rust_energy_fd":own,
                "rust_analytic_error":(gradient[coordinate]-fd).abs(),"official_analytic_error":(reference_gradient[coordinate]-fd).abs()}));
                assert!(
                    (gradient[coordinate] - own).abs() < 5e-7,
                    "Rust energy/gradient inconsistency"
                );
                assert!(
                    (own - fd).abs() < 5e-7,
                    "the energy surfaces differ in the perturbed direction"
                );
            }
            records.push(json!({"case":name,"solvent":solvent.map_or("vacuum".into(), |(model, name)| format!("{model:?}-{name}")),"coordinate":coordinate,
            "base_energy_error_hartree":(rust.total_energy-reference_energy).abs(),"rust_analytic":gradient[coordinate],
            "source_predicted_missing_h0_ds":missing_h0_ds,
            "official_analytic":reference_gradient[coordinate],"finite_differences":derivatives}));
        }
    }
    assert_eq!(records.len(), 10, "missing reference gradient audit cases");
    fs::write(
        root.join("validation.json"),
        serde_json::to_vec_pretty(&json!({"schema":"gfn1-rs-xtb-gradient-audit-v1","package_version":env!("CARGO_PKG_VERSION"),"status":"pass","executed_cases":records.len(),"cases":records})).unwrap(),
    )
    .unwrap();
}
