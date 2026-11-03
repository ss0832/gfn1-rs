// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::*;
use std::sync::Arc;
fn fixture() -> (PeriodicSystem, Gfn1Parameters, ElectronicOptions) {
    let system = PeriodicSystem::from_xyz_str(
        "3\nnonstationary water\nO .1 .2 .0\nH .97 .74 .04\nH -.69 .83 -.03\n",
        0.0,
        false,
    )
    .unwrap();
    let mut model = MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0);
    for (&z, h) in &mut model.heads {
        for (l, p) in h.levels.iter_mut().enumerate() {
            p.bias = 0.0001 * (l + 1) as f64;
            for (k, w) in p.weights.iter_mut().enumerate() {
                *w = 0.0003 * ((k + z as usize + l) % 3) as f64;
            }
        }
        h.correction.weights[0] = 0.0002;
    }
    let options = ElectronicOptions {
        ml_model: Some(Arc::new(model)),
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-11,
        ..Default::default()
    };
    (system, Gfn1Parameters::builtin().unwrap(), options)
}
fn displaced(system: &PeriodicSystem, v: &[f64], step: f64) -> PeriodicSystem {
    let mut out = system.clone();
    for (a, atom) in out.atoms.iter_mut().enumerate() {
        atom.position.x += step * v[3 * a];
        atom.position.y += step * v[3 * a + 1];
        atom.position.z += step * v[3 * a + 2];
    }
    out
}
#[test]
fn structural_ml_hessian_matches_gradient_fd_and_fc3_matches_hessian_fd() {
    let (system, params, electronic) = fixture();
    let d = 3 * system.atoms.len();
    let options = AnalyticHessianOptions {
        electronic_options: electronic.clone(),
        ..Default::default()
    };
    let v: Vec<f64> = (0..d).map(|i| 0.17 - 0.08 * (i % 4) as f64).collect();
    let h = analytic_hessian(&system, &params, options.clone())
        .unwrap()
        .hessian;
    let reference = gfn1_rs::third_derivative::CubicReference::build(
        &system,
        &params,
        &electronic,
        512 * 1024 * 1024,
    )
    .unwrap();
    let direct = reference.directional(&v).unwrap();
    let vector = third_derivative_analytic_vector(
        &system,
        &params,
        options.clone(),
        electronic.hamiltonian.coordination_cutoff,
        &v,
    )
    .unwrap();
    for step in [1e-3, 5e-4] {
        let plus = displaced(&system, &v, step);
        let minus = displaced(&system, &v, -step);
        let gp = analytic_gradient(
            &plus,
            &params,
            AnalyticGradientOptions {
                electronic: electronic.clone(),
                ..Default::default()
            },
        )
        .unwrap()
        .gradient;
        let gm = analytic_gradient(
            &minus,
            &params,
            AnalyticGradientOptions {
                electronic: electronic.clone(),
                ..Default::default()
            },
        )
        .unwrap()
        .gradient;
        let hp = analytic_hessian(&plus, &params, options.clone())
            .unwrap()
            .hessian;
        let hm = analytic_hessian(&minus, &params, options.clone())
            .unwrap()
            .hessian;
        let mut fd_cubic = 0.0;
        for i in 0..d {
            let fd = (gp[i / 3].to_array()[i % 3] - gm[i / 3].to_array()[i % 3]) / (2.0 * step);
            let analytic = (0..d).map(|j| h[(i, j)] * v[j]).sum::<f64>();
            assert!(
                (fd - analytic).abs() < 2e-6,
                "ML Hessian analytic {analytic:e}, FD {fd:e}"
            );
            for j in 0..d {
                let fd = (hp[(i, j)] - hm[(i, j)]) / (2.0 * step);
                fd_cubic += v[i] * v[j] * fd;
                assert!(
                    (fd - vector[(i, j)]).abs() < 5e-6,
                    "ML FC3 analytic {:e}, FD {fd:e}",
                    vector[(i, j)]
                );
            }
        }
        assert!(
            (fd_cubic - direct).abs() < 1e-6,
            "ML directional FC3 {direct:e}, Hessian FD {fd_cubic:e}"
        );
    }
    assert!(fourth_derivative_analytic_dense(
        &system,
        &params,
        &options,
        electronic.hamiltonian.coordination_cutoff
    )
    .is_err());
}
#[test]
fn independently_fitted_h2_model_has_full_analytic_cubic_derivatives() {
    let params = Gfn1Parameters::builtin().unwrap();
    let system = PeriodicSystem::from_xyz_str("2\nH2\nH 0 0 0\nH .8 0 0\n", 0.0, false).unwrap();
    let electronic = ElectronicOptions {
        ml_model: Some(Arc::new(MlHamiltonian::builtin_h2_fci().unwrap())),
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-11,
        ..Default::default()
    };
    let options = AnalyticHessianOptions {
        electronic_options: electronic.clone(),
        ..Default::default()
    };
    let tensor = third_derivative_analytic_dense(
        &system,
        &params,
        options.clone(),
        electronic.hamiltonian.coordination_cutoff,
    )
    .unwrap();
    let h = 1e-3;
    let mut plus = system.clone();
    let mut minus = system.clone();
    plus.atoms[1].position.x += h;
    minus.atoms[1].position.x -= h;
    let hp = analytic_hessian(&plus, &params, options.clone())
        .unwrap()
        .hessian;
    let hm = analytic_hessian(&minus, &params, options).unwrap().hessian;
    assert!((tensor.get(3, 3, 3) - (hp[(3, 3)] - hm[(3, 3)]) / (2.0 * h)).abs() < 2e-5);
}
