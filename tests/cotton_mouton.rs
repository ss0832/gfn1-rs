// SPDX-License-Identifier: GPL-3.0-or-later
//! Independent mixed electric/magnetic total-energy differences, including zeros.
use gfn1_rs::math::Vec3;
use gfn1_rs::*;
use serde_json::json;
use sha2::Digest;

const WATER: &str = "3\nasymmetric water\nO .1 .2 0\nH .97 .74 .04\nH -.69 .83 -.03\n";

fn field(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn energy_alpha(
    system: &PeriodicSystem,
    parameters: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    b: Vec3,
    step: f64,
) -> [[f64; 3]; 3] {
    let energy = |e: [f64; 3]| {
        let mut o = options.clone();
        o.external_field.electric_field = Some(field(e));
        o.external_field.magnetic_field = Some(b);
        let r = if let Some(basis) = secondary {
            run_magnetic_scc_m1(system, parameters, &o, basis)
        } else {
            run_magnetic_scc(system, parameters, &o)
        }
        .unwrap();
        assert!(r.converged, "mixed energy reference did not converge");
        r.energy
    };
    let e0 = energy([0.; 3]);
    let mut alpha = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            alpha[i][j] = if i == j {
                let mut e = [0.; 3];
                e[i] = step;
                let ep = energy(e);
                e[i] = -step;
                -(ep - 2. * e0 + energy(e)) / step.powi(2)
            } else {
                let mut sum = 0.;
                for si in [-1., 1.] {
                    for sj in [-1., 1.] {
                        let mut e = [0.; 3];
                        e[i] = si * step;
                        e[j] = sj * step;
                        sum += si * sj * energy(e);
                    }
                }
                -sum / (4. * step.powi(2))
            };
        }
    }
    alpha
}

#[test]
#[ignore = "all 27 Cotton-Mouton components from independent E^2 B^2 energy differences, M0 and M1"]
fn cotton_mouton_all_components_match_energy_differences() {
    let system = PeriodicSystem::from_xyz_str(WATER, 0., false).unwrap();
    let parameters = Gfn1Parameters::builtin().unwrap();
    let options = ElectronicOptions {
        electronic_temperature: 0.,
        energy_tolerance: 1e-14,
        charge_tolerance: 1e-12,
        max_scc: 1000,
        ..Default::default()
    };
    let secondary = builtin_secondary("cc-pVDZ").unwrap().unwrap();
    let mut cases = Vec::new();
    for (name, basis) in [("M0", None), ("M1", Some(&secondary))] {
        let mut levels = Vec::new();
        let mut actual = Vec::new();
        let mut reference = Vec::new();
        for (e_step, b_step) in [(0.004, 0.04), (0.002, 0.02), (0.001, 0.01)] {
            let cm = cotton_mouton_tensor(&system, &parameters, &options, basis, e_step, b_step)
                .unwrap();
            let a0 = energy_alpha(&system, &parameters, &options, basis, Vec3::zero(), e_step);
            let mut expected = [[[0.; 3]; 3]; 3];
            for k in 0..3 {
                let mut b = [0.; 3];
                b[k] = b_step;
                let plus = energy_alpha(&system, &parameters, &options, basis, field(b), e_step);
                b[k] = -b_step;
                let minus = energy_alpha(&system, &parameters, &options, basis, field(b), e_step);
                for i in 0..3 {
                    for j in 0..3 {
                        expected[k][i][j] =
                            (plus[i][j] - 2. * a0[i][j] + minus[i][j]) / b_step.powi(2);
                    }
                }
            }
            let a: Vec<f64> = cm.into_iter().flatten().flatten().collect();
            let r: Vec<f64> = expected.into_iter().flatten().flatten().collect();
            let max_error = a
                .iter()
                .zip(&r)
                .map(|(a, r)| (a - r).abs())
                .fold(0., f64::max);
            println!(
                "Cotton-Mouton {name} E={e_step:e} B={b_step:e} max energy-FD error={max_error:e}"
            );
            levels.push(json!({"electric_step_au":e_step,"magnetic_step_au":b_step,
                "dipole_field_difference":a,"total_energy_mixed_difference":r,"max_abs_error":max_error}));
            actual.push(a);
            reference.push(r);
        }
        let mut extrapolated = Vec::new();
        let mut passed = 0;
        for component in 0..27 {
            let a = (4. * actual[2][component] - actual[1][component]) / 3.;
            let r = (4. * reference[2][component] - reference[1][component]) / 3.;
            let tolerance = 2e-3 + 2e-3 * a.abs().max(r.abs());
            if (a - r).abs() <= tolerance {
                passed += 1;
            }
            extrapolated.push(
                json!({"flat_index":component,"response":a,"energy_reference":r,
                "abs_error":(a-r).abs(),"tolerance":tolerance}),
            );
        }
        cases.push(
            json!({"model":name,"components":27,"passed_components":passed,"levels":levels,
            "richardson":extrapolated,"status":if passed==27{"pass"}else{"fail"}}),
        );
    }
    let report = json!({"schema":"gfn1-rs-cotton-mouton-energy-audit-v1",
        "package_version":env!("CARGO_PKG_VERSION"),"dependency_lockfile_sha256":format!("{:x}",sha2::Sha256::digest(include_bytes!("../Cargo.lock"))),"system_xyz_angstrom":WATER,
        "scope":"all existing diagonal magnetic-field slices: d2 alpha_ij / d B_k^2; not the 81-component mixed-B tensor",
        "reference":"negative fourth derivative of converged total energy with two electric and two magnetic field legs",
        "implementation":"nested finite fields of the Mulliken dipole; not an analytic fourth-order response",
        "electronic_temperature_K":0,"energy_tolerance":options.energy_tolerance,
        "charge_tolerance":options.charge_tolerance,"cases":cases});
    let root = std::path::PathBuf::from(
        std::env::var("GFN1_VALIDATION_DIR")
            .unwrap_or_else(|_| "target/validation/cotton-mouton".into()),
    );
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("cotton-mouton.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    assert!(
        cases.iter().all(|c| c["status"] == "pass"),
        "Cotton-Mouton energy audit failed"
    );
}

#[test]
fn mixed_magnetic_responses_reject_invalid_electric_steps() {
    let s = PeriodicSystem::from_xyz_str(WATER, 0., false).unwrap();
    let p = Gfn1Parameters::builtin().unwrap();
    let o = ElectronicOptions::default();
    for e in [0., -0.001, f64::NAN, f64::INFINITY] {
        assert!(cotton_mouton_tensor(&s, &p, &o, None, e, 0.01).is_err());
        assert!(mcd_tensor(&s, &p, &o, None, Vec3::zero(), e, 0.01).is_err());
    }
}
