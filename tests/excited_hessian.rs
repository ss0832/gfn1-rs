// SPDX-License-Identifier: GPL-3.0-or-later
//! Independent Cartesian checks of native analytic excited-state Hessians.
//! Finite differences occur only in these tests, at fixed model/temperature/root.

use gfn1_rs::stda_xtb::{
    solve_stda_xtb_gradient, solve_stda_xtb_hessian, solve_stda_xtb_total_hessian,
    NativeStdaOptions,
};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, TdaOptions};

const WATER: &str = "3\nwater\nO .02 -.01 .03\nH .958 .02 -.01\nH -.25 .925 .04\n";
const ETHYLENE: &str = "6\nethylene\nC -.67 .01 -.02\nC .67 -.01 .02\nH -1.23 .92 .03\nH -1.24 -.93 -.01\nH 1.24 .93 -.02\nH 1.23 -.92 .04\n";

fn displaced(system: &PeriodicSystem, coordinate: usize, step: f64) -> PeriodicSystem {
    let mut result = system.clone();
    let r = &mut result.atoms[coordinate / 3].position;
    match coordinate % 3 {
        0 => r.x += step,
        1 => r.y += step,
        _ => r.z += step,
    }
    result
}

fn rotated_reference(name: &str) -> PeriodicSystem {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/stda-ground-reference.json")).unwrap();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap();
    let atoms = case["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["positions_bohr"].as_array().unwrap())
        .map(|(z, r)| {
            let (x, y, zc) = (
                r[0].as_f64().unwrap(),
                r[1].as_f64().unwrap(),
                r[2].as_f64().unwrap(),
            );
            gfn1_rs::system::Atom {
                z: z.as_u64().unwrap() as u8,
                position: gfn1_rs::math::Vec3::new(
                    0.8 * x - 0.6 * zc,
                    0.8 * y - 0.6 * (0.6 * x + 0.8 * zc),
                    0.6 * y + 0.8 * (0.6 * x + 0.8 * zc),
                ),
            }
        })
        .collect();
    PeriodicSystem::new(atoms, None)
}

#[test]
fn stda_analytic_hessian_all_cartesian_components_match_gradient_differences() {
    let mut failures = Vec::new();
    for (name, xyz, state) in [("water", WATER, 0), ("ethylene", ETHYLENE, 2)] {
        let system = if name == "ethylene" {
            rotated_reference(name)
        } else {
            PeriodicSystem::from_xyz_str(xyz, 0., false).unwrap()
        };
        let n = 3 * system.atoms.len();
        for temperature in [0., 300.] {
            for triplet in [false, true] {
                let options = NativeStdaOptions {
                    electronic_temperature: temperature,
                    // Match the independently validated analytic-gradient model.
                    energy_window_ev: 10.,
                    n_states: 6,
                    triplet,
                    ..Default::default()
                };
                let result = solve_stda_xtb_hessian(&system, state, &options)
                    .unwrap_or_else(|e| panic!("{name} T={temperature} triplet={triplet}: {e}"));
                let mut maximum = 0.0_f64;
                let mut coarse_maximum = 0.0_f64;
                let mut first_failure = None;
                for column in 0..n {
                    let mut differences = Vec::new();
                    // The singlet's CI curvature needs a smaller stencil;
                    // the triplet and water ladders converge at 1e-3 bohr.
                    let steps = if name == "ethylene" && !triplet {
                        [1e-4, 5e-5]
                    } else {
                        [1e-3, 5e-4]
                    };
                    for step in steps {
                        let plus = solve_stda_xtb_gradient(
                            &displaced(&system, column, step),
                            state,
                            &options,
                        )
                        .unwrap();
                        let minus = solve_stda_xtb_gradient(
                            &displaced(&system, column, -step),
                            state,
                            &options,
                        )
                        .unwrap();
                        let configuration_set = |pairs: &[(usize, usize)]| {
                            pairs
                                .iter()
                                .copied()
                                .collect::<std::collections::BTreeSet<_>>()
                        };
                        assert_eq!(
                            configuration_set(&plus.spectrum.configurations),
                            configuration_set(&result.spectrum.configurations),
                            "{name}: configuration boundary crossed in positive stencil"
                        );
                        assert_eq!(
                            configuration_set(&minus.spectrum.configurations),
                            configuration_set(&result.spectrum.configurations),
                            "{name}: configuration boundary crossed in negative stencil"
                        );
                        differences.push(
                            plus.excitation_gradient
                                .iter()
                                .zip(&minus.excitation_gradient)
                                .flat_map(|(a, b)| ((*a - *b) * (0.5 / step)).to_array())
                                .collect::<Vec<_>>(),
                        );
                    }
                    for row in 0..n {
                        let reference = (4. * differences[1][row] - differences[0][row]) / 3.;
                        let analytic = result.excitation_hessian[(row, column)];
                        maximum = maximum.max((analytic - reference).abs());
                        coarse_maximum = coarse_maximum.max((analytic - differences[0][row]).abs());
                        if (analytic - reference).abs() >= 2e-8 + 1e-7 * reference.abs()
                            && first_failure.is_none()
                        {
                            first_failure=Some(format!("{name} T={temperature} triplet={triplet} ({row},{column}): analytic={analytic:e} FD={reference:e}"));
                        }
                        assert!(
                            (analytic - result.excitation_hessian[(column, row)]).abs() < 1e-12
                        );
                    }
                }
                let mut translation = 0.0_f64;
                for row in 0..n {
                    for axis in 0..3 {
                        translation = translation.max(
                            (0..system.atoms.len())
                                .map(|atom| result.excitation_hessian[(row, 3 * atom + axis)])
                                .sum::<f64>()
                                .abs(),
                        );
                    }
                }
                assert!(
                    translation < 1e-8,
                    "{name}: translational ASR {translation:e}"
                );
                let gradient = solve_stda_xtb_gradient(&system, state, &options).unwrap();
                assert!((gradient.excitation_energy - result.excitation_energy).abs() < 1e-12);
                for (a, b) in gradient
                    .excitation_gradient
                    .iter()
                    .zip(&result.excitation_gradient)
                {
                    assert!((*a - *b).norm() < 1e-11);
                }
                eprintln!("sTDA Hessian {name} root={state} T={temperature} triplet={triplet}: all {} components; coarse={coarse_maximum:e}; Richardson={maximum:e}; translation={translation:e}",n*n);
                if let Some(failure) = first_failure {
                    failures.push(failure);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn stda_total_hessian_matches_independent_ground_and_excitation() {
    let params = Gfn1Parameters::builtin().unwrap();
    for (name, xyz, state) in [("water", WATER, 0), ("ethylene", ETHYLENE, 2)] {
        let system = PeriodicSystem::from_xyz_str(xyz, 0., false).unwrap();
        let electronic = ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            max_scc: 1000,
            ..Default::default()
        };
        let options = NativeStdaOptions {
            energy_window_ev: 10.,
            n_states: 6,
            ..Default::default()
        };
        let total =
            solve_stda_xtb_total_hessian(&system, &params, &electronic, state, &options).unwrap();
        let excitation = solve_stda_xtb_hessian(&system, state, &options).unwrap();
        let ground = gfn1_rs::analytic_hessian(
            &system,
            &params,
            gfn1_rs::AnalyticHessianOptions {
                electronic_options: electronic.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..total.hessian.rows() {
            for j in 0..total.hessian.cols() {
                assert!(
                    (total.hessian[(i, j)]
                        - ground.hessian[(i, j)]
                        - excitation.excitation_hessian[(i, j)])
                        .abs()
                        < 1e-12,
                    "{name}: composite Hessian mismatch ({i},{j})"
                );
                assert!((total.ground_hessian[(i, j)] - ground.hessian[(i, j)]).abs() < 1e-12);
                assert!(
                    (total.excitation_hessian[(i, j)] - excitation.excitation_hessian[(i, j)])
                        .abs()
                        < 1e-12
                );
            }
        }
        assert!((total.total_energy - total.ground_energy - total.excitation_energy).abs() < 1e-12);
        assert!((total.excitation_energy - excitation.excitation_energy).abs() < 1e-12);
        assert!((total.ground_energy - ground.electronic_result.unwrap().total_free).abs() < 1e-12);
        let mismatched = NativeStdaOptions {
            electronic_temperature: 0.,
            ..options
        };
        assert!(
            solve_stda_xtb_total_hessian(&system, &params, &electronic, state, &mismatched)
                .unwrap_err()
                .to_string()
                .contains("matching electronic temperatures")
        );
        eprintln!("sTDA composite Hessian {name}: energy and all matrix components agree");
    }
}

#[test]
fn excited_hessian_budget_rejects_before_orbital_or_scf_work() {
    let system = PeriodicSystem::from_xyz_str(WATER, 0., false).unwrap();
    let options = NativeStdaOptions {
        memory_budget_bytes: 1,
        ..Default::default()
    };
    let error = solve_stda_xtb_hessian(&system, 0, &options)
        .unwrap_err()
        .to_string();
    assert!(error.contains("before orbital allocation"), "{error}");
    let params = Gfn1Parameters::builtin().unwrap();
    let electronic = ElectronicOptions {
        max_scc: 0,
        ..Default::default()
    };
    let error = gfn1_rs::td::solve_tda_hessian_analytic_with_budget(
        &system,
        &params,
        &electronic,
        0,
        TdaOptions::default(),
        1,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("before SCF"), "{error}");
}

#[test]
#[ignore = "diagnostic step ladder for the strongly curved ethylene response"]
fn stda_hessian_ethylene_step_ladder() {
    let system = PeriodicSystem::from_xyz_str(ETHYLENE, 0., false).unwrap();
    for triplet in [false, true] {
        let options = NativeStdaOptions {
            electronic_temperature: 0.,
            energy_window_ev: 200.,
            triplet,
            ..Default::default()
        };
        let result = solve_stda_xtb_hessian(&system, 2, &options).unwrap();
        for step in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5] {
            let plus = solve_stda_xtb_gradient(&displaced(&system, 0, step), 2, &options).unwrap();
            let minus =
                solve_stda_xtb_gradient(&displaced(&system, 0, -step), 2, &options).unwrap();
            eprintln!(
                "ethylene ladder triplet={triplet} h={step:e} analytic={:e} FD={:e}",
                result.excitation_hessian[(0, 0)],
                (plus.excitation_gradient[0].x - minus.excitation_gradient[0].x) / (2. * step)
            );
        }
    }
}

#[test]
#[ignore = "all-component FD conditioning audit of the wide-window ethylene model"]
fn stda_hessian_wide_window_all_component_ladder() {
    let system = rotated_reference("ethylene");
    let options = NativeStdaOptions {
        electronic_temperature: 300.,
        energy_window_ev: 200.,
        ..Default::default()
    };
    let result = solve_stda_xtb_hessian(&system, 2, &options).unwrap();
    let energies = &result.spectrum.orbital_energies_hartree;
    let mo_gap = energies
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(f64::INFINITY, f64::min);
    let root_gap = result
        .spectrum
        .states
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != 2)
        .map(|(_, state)| (state.excitation_hartree - result.excitation_energy).abs())
        .fold(f64::INFINITY, f64::min);
    eprintln!("wide-window conditioning: MOgap={mo_gap:e}, CIgap={root_gap:e}, configurations={}, primary={}",result.spectrum.configurations.len(),result.spectrum.primary_configuration_count);
    let mut previous: Option<(f64, Vec<f64>)> = None;
    for step in [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5] {
        let mut values = vec![0.; 18 * 18];
        for column in 0..18 {
            let plus =
                solve_stda_xtb_gradient(&displaced(&system, column, step), 2, &options).unwrap();
            let minus =
                solve_stda_xtb_gradient(&displaced(&system, column, -step), 2, &options).unwrap();
            for row in 0..18 {
                values[row * 18 + column] = (plus.excitation_gradient[row / 3].to_array()[row % 3]
                    - minus.excitation_gradient[row / 3].to_array()[row % 3])
                    / (2. * step);
            }
        }
        let maximum = values
            .iter()
            .enumerate()
            .map(|(index, &fd)| (fd - result.excitation_hessian[(index / 18, index % 18)]).abs())
            .fold(0., f64::max);
        let rich = previous.as_ref().map(|(old_step, old)| {
            let ratio = (old_step / step).powi(2);
            values
                .iter()
                .zip(old)
                .enumerate()
                .map(|(index, (&fine, &coarse))| {
                    ((ratio * fine - coarse) / (ratio - 1.)
                        - result.excitation_hessian[(index / 18, index % 18)])
                        .abs()
                })
                .fold(0., f64::max)
        });
        eprintln!("wide-window all 324 h={step:e}: raw_error={maximum:e} Richardson={rich:?}, component10_FD={:e}, component10_analytic={:e}",values[18],result.excitation_hessian[(1,0)]);
        previous = Some((step, values));
    }
}

#[test]
fn fractional_stda_ensemble_matches_all_gradient_and_hessian_components() {
    let system = PeriodicSystem::from_xyz_str(WATER, 0., false).unwrap();
    let options = NativeStdaOptions {
        electronic_temperature: 30000.,
        thermal_response: Some(true),
        energy_window_ev: 20.,
        ..Default::default()
    };
    let reference = solve_stda_xtb_hessian(&system, 0, &options).unwrap();
    assert!(reference.spectrum.thermal_response);
    assert!(reference.spectrum.maximum_occupation_deviation > 1e-4);
    let n = 3 * system.atoms.len();
    let mut max_gradient = 0.0_f64;
    let mut max_hessian = 0.0_f64;
    for column in 0..n {
        let steps = [2e-4, 1e-4];
        let mut estimates = Vec::new();
        for step in steps {
            let plus = solve_stda_xtb_gradient(&displaced(&system, column, step), 0, &options).unwrap();
            let minus = solve_stda_xtb_gradient(&displaced(&system, column, -step), 0, &options).unwrap();
            assert_eq!(plus.spectrum.configurations, reference.spectrum.configurations);
            assert_eq!(minus.spectrum.configurations, reference.spectrum.configurations);
            let d1 = (plus.excitation_energy - minus.excitation_energy) / (2. * step);
            let d2: Vec<_> = plus.excitation_gradient.iter().zip(&minus.excitation_gradient)
                .flat_map(|(a, b)| ((*a - *b) * (0.5 / step)).to_array()).collect();
            estimates.push((d1, d2));
        }
        let d1 = (4. * estimates[1].0 - estimates[0].0) / 3.;
        max_gradient = max_gradient.max((d1 - reference.excitation_gradient[column/3].to_array()[column%3]).abs());
        for row in 0..n {
            let d2 = (4. * estimates[1].1[row] - estimates[0].1[row]) / 3.;
            max_hessian = max_hessian.max((d2 - reference.excitation_hessian[(row,column)]).abs());
        }
    }
    eprintln!("sTDA fractional 30000 K: all {n} gradient entries max {max_gradient:.6e}, all {} Hessian entries max {max_hessian:.6e}", n*n);
    assert!(max_gradient < 5e-7);
    assert!(max_hessian < 5e-6);
}
