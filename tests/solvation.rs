// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::*;

fn water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.1 0.2 0\nH 0.87 0.75 0.04\nH -0.63 0.81 -0.02\n",
        0.0,
        false,
    )
    .unwrap()
}

#[test]
fn solvent_quadratic_potential_kernel_and_geometric_derivative() {
    let sys = water();
    let charges = [-0.5, 0.2, 0.4];
    for model in [SolvationModel::Alpb, SolvationModel::Gbsa] {
        for name in ["water", "benzene", "acetonitrile"] {
            let opts = SolvationOptions::new(model, name);
            let cache = SolvationCache::build(&sys, &opts).unwrap();
            assert!(cache.born_radii.iter().all(|x| x.is_finite() && *x > 0.0));
            let potential = cache.potential(&charges).unwrap();
            for i in 0..3 {
                let mut plus = charges;
                let mut minus = charges;
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let fd = (cache.energy(&plus).unwrap().total()
                    - cache.energy(&minus).unwrap().total())
                    / 2e-5;
                assert!((fd - potential[i]).abs() < 1e-10);
                let vp = cache.potential(&plus).unwrap();
                let vm = cache.potential(&minus).unwrap();
                for j in 0..3 {
                    assert!(((vp[j] - vm[j]) / 2e-5 - cache.matrix[(j, i)]).abs() < 1e-10);
                }
            }
            let gradient = cache.gradient(&charges).unwrap();
            for h in [1e-4, 5e-5] {
                let mut plus = sys.clone();
                let mut minus = sys.clone();
                plus.atoms[1].position.x += h;
                minus.atoms[1].position.x -= h;
                let fd = (SolvationCache::build(&plus, &opts)
                    .unwrap()
                    .energy(&charges)
                    .unwrap()
                    .total()
                    - SolvationCache::build(&minus, &opts)
                        .unwrap()
                        .energy(&charges)
                        .unwrap()
                        .total())
                    / (2.0 * h);
                assert!(
                    (fd - gradient[1].x).abs() < 2e-8,
                    "{model:?} {name}: analytic {} FD {fd}",
                    gradient[1].x
                );
            }
            let sum = gradient
                .iter()
                .fold(gfn1_rs::math::Vec3::zero(), |a, b| a + *b);
            assert!(sum.norm() < 1e-12);
        }
    }
}

#[test]
fn solvent_scf_total_force_and_response_include_same_kernel() {
    let params = Gfn1Parameters::builtin().unwrap();
    let sys = water();
    for model in [SolvationModel::Alpb, SolvationModel::Gbsa] {
        let mut opts = ElectronicOptions::default();
        opts.solvation = Some(SolvationOptions::new(model, "water"));
        opts.energy_tolerance = 1e-12;
        opts.charge_tolerance = 1e-10;
        let gradient = analytic_gradient(
            &sys,
            &params,
            AnalyticGradientOptions {
                electronic: opts.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let e = &gradient.electronic_result;
        assert!((e.energy_terms().component_sum() - e.total_internal).abs() < 2e-14);
        assert!(e.method_citations.contains(&"Ehlert2021ALPB"));
        for h in [2e-4, 1e-4] {
            let mut plus = sys.clone();
            let mut minus = sys.clone();
            plus.atoms[1].position.x += h;
            minus.atoms[1].position.x -= h;
            let fd = (run_electronic(&plus, &params, opts.clone())
                .unwrap()
                .total_free
                - run_electronic(&minus, &params, opts.clone())
                    .unwrap()
                    .total_free)
                / (2.0 * h);
            assert!(
                (fd - gradient.gradient[1].x).abs() < 5e-7,
                "{model:?} full force {} FD {fd}",
                gradient.gradient[1].x
            );
        }
        let polar = static_polarizability(&sys, &params, e).unwrap();
        let finite = static_polarizability_finite_field(&sys, &params, &opts, 1e-4).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((polar.tensor[i][j] - finite.tensor[i][j]).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn analytic_solvent_hessian_fc3_and_fc4_match_all_cartesian_differences() {
    let params = Gfn1Parameters::builtin().unwrap();
    let system = water();
    let ndof = 3 * system.atoms.len();
    let displaced = |coordinate: usize, step: f64| {
        let mut geometry = system.clone();
        let mut shift = [0.; 3];
        shift[coordinate % 3] = step;
        geometry.atoms[coordinate / 3].position += math::Vec3::new(shift[0], shift[1], shift[2]);
        geometry
    };
    for model in [SolvationModel::Alpb, SolvationModel::Gbsa] {
        for temperature in [0., 30000.] {
            let electronic = ElectronicOptions {
                solvation: Some(SolvationOptions::new(model, "water")),
                electronic_temperature: temperature,
                energy_tolerance: 1e-13,
                charge_tolerance: 1e-11,
                ..Default::default()
            };
            let options = AnalyticHessianOptions {
                electronic_options: electronic.clone(),
                ..Default::default()
            };
            let cutoff = electronic.hamiltonian.coordination_cutoff;
            let gradient = |geometry: &PeriodicSystem| {
                analytic_gradient(
                    geometry,
                    &params,
                    AnalyticGradientOptions {
                        electronic: electronic.clone(),
                        ..Default::default()
                    },
                )
                .unwrap()
                .gradient
            };
            let hessian = |geometry: &PeriodicSystem| {
                analytic_hessian(geometry, &params, options.clone())
                    .unwrap()
                    .hessian
            };
            let cubic = |geometry: &PeriodicSystem| {
                third_derivative_analytic_dense(geometry, &params, options.clone(), cutoff).unwrap()
            };
            let analytic_h = hessian(&system);
            let analytic_c = cubic(&system);
            let analytic_q =
                fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap();
            let step = 1e-4;
            let scale = 0.5 / step;
            let mut worst = [0.0_f64; 3];
            let mut locations = [[0usize; 4]; 3];
            let mut measure = |order: usize, error: f64, indices: [usize; 4]| {
                assert!(error.is_finite(), "{model:?} {temperature} K {indices:?}");
                if error.abs() > worst[order] {
                    worst[order] = error.abs();
                    locations[order] = indices;
                }
            };
            for d in 0..ndof {
                let plus = displaced(d, step);
                let minus = displaced(d, -step);
                let (gp, gm) = (gradient(&plus), gradient(&minus));
                let (hp, hm) = (hessian(&plus), hessian(&minus));
                let (cp, cm) = (cubic(&plus), cubic(&minus));
                for a in 0..ndof {
                    let dh = (gp[a / 3] - gm[a / 3]).to_array()[a % 3] * scale;
                    measure(0, analytic_h[(a, d)] - dh, [a, d, 0, 0]);
                    for b in 0..ndof {
                        let dc = (hp[(a, b)] - hm[(a, b)]) * scale;
                        measure(1, analytic_c.get(a, b, d) - dc, [a, b, d, 0]);
                        for c in 0..ndof {
                            let dq = (cp.get(a, b, c) - cm.get(a, b, c)) * scale;
                            measure(2, analytic_q.get(a, b, c, d) - dq, [a, b, c, d]);
                        }
                    }
                }
            }
            eprintln!("{model:?} {temperature} K maximum errors H={:.3e} FC3={:.3e} FC4={:.3e} at {locations:?}",
                worst[0], worst[1], worst[2]);
            assert!(
                worst[0] < 2e-7,
                "{model:?} {temperature} K Hessian {worst:?}"
            );
            assert!(worst[1] < 1e-6, "{model:?} {temperature} K FC3 {worst:?}");
            assert!(worst[2] < 1e-5, "{model:?} {temperature} K FC4 {worst:?}");
        }
    }
}

#[test]
fn gapped_nickel_carbonyl_thermal_solvent_hessian_matches_all_components() {
    let params = Gfn1Parameters::builtin().unwrap();
    let mut system =
        PeriodicSystem::from_xyz_str(include_str!("../examples/ni_co4.xyz"), 0., false).unwrap();
    for atom in &mut system.atoms {
        match atom.z {
            6 => atom.position = atom.position * (1.0352710931 / 1.82),
            8 => atom.position = atom.position * (1.6899361262 / 2.48),
            _ => {}
        }
    }
    let options = ElectronicOptions {
        solvation: Some(SolvationOptions::new(SolvationModel::Alpb, "water")),
        electronic_temperature: 300.,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-11,
        ..Default::default()
    };
    let gradient = |geometry: &PeriodicSystem| {
        analytic_gradient(
            geometry,
            &params,
            AnalyticGradientOptions {
                electronic: options.clone(),
                ..Default::default()
            },
        )
        .unwrap()
        .gradient
    };
    let hessian = analytic_hessian(
        &system,
        &params,
        AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..Default::default()
        },
    )
    .unwrap()
    .hessian;
    let step = 1e-4;
    let mut maximum = 0.0_f64;
    for column in 0..hessian.cols() {
        let mut delta = [0.; 3];
        delta[column % 3] = step;
        let delta = math::Vec3::new(delta[0], delta[1], delta[2]);
        let mut plus = system.clone();
        let mut minus = system.clone();
        plus.atoms[column / 3].position += delta;
        minus.atoms[column / 3].position -= delta;
        let gp = gradient(&plus);
        let gm = gradient(&minus);
        for row in 0..hessian.rows() {
            let fd = (gp[row / 3] - gm[row / 3]).to_array()[row % 3] / (2. * step);
            let error = (fd - hessian[(row, column)]).abs();
            assert!(error.is_finite());
            maximum = maximum.max(error);
        }
    }
    eprintln!("Ni(CO)4 ALPB Hessian 27x27 maximum FD difference {maximum:.3e} Ha/bohr²");
    assert!(maximum < 2e-6);
}

#[test]
fn no_parameter_fallback_and_invalid_grid_errors() {
    let sys = water();
    assert!(
        SolvationCache::build(&sys, &SolvationOptions::new(SolvationModel::Gbsa, "dmf")).is_err()
    );
    let mut opts = SolvationOptions::new(SolvationModel::Alpb, "water");
    opts.grid_points = 17;
    assert!(SolvationCache::build(&sys, &opts).is_err());
}
