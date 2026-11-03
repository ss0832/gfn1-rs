// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::ml::*;
use gfn1_rs::*;
use std::sync::Arc;
fn water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nwater\nO .1 0 0\nH .9 .55 .01\nH -.7 .62 -.03\n",
        0.0,
        false,
    )
    .unwrap()
}
fn options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ..Default::default()
    }
}
#[test]
fn zero_heads_exactly_restore_gfn1_and_structure_local_parameters_are_independent() {
    let params = Gfn1Parameters::builtin().unwrap();
    let system = water();
    let plain = options();
    let base = run_electronic(&system, &params, plain.clone()).unwrap();
    let model = Arc::new(MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0));
    let mut opts = plain;
    opts.ml_model = Some(Arc::clone(&model));
    let zero = run_electronic(&system, &params, opts).unwrap();
    assert_eq!(base.total_free, zero.total_free);
    assert_eq!(base.atomic_charges, zero.atomic_charges);
    let mut nonzero = (*model).clone();
    nonzero.heads.get_mut(&1).unwrap().levels[0].weights = vec![0.0002, -0.0001, 0.0005, -0.0003];
    let nonzero = Arc::new(nonzero);
    let view = nonzero.evaluate(&system).unwrap();
    assert_ne!(
        view.delta_levels_hartree[1][0],
        view.delta_levels_hartree[2][0]
    );
    let mut changed = system.clone();
    changed.atoms[1].position.x += 0.2;
    let other = nonzero.evaluate(&changed).unwrap();
    assert_ne!(view.delta_levels_hartree, other.delta_levels_hartree);
    assert_eq!(
        view.delta_levels_hartree,
        nonzero.evaluate(&system).unwrap().delta_levels_hartree
    );
}
#[test]
fn ml_hamiltonian_force_chain_and_stationary_training_gradient_match_fd() {
    let params = Gfn1Parameters::builtin().unwrap();
    let system = water();
    let mut model = MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0);
    for (&z, heads) in &mut model.heads {
        for (l, h) in heads.levels.iter_mut().enumerate() {
            h.bias = 0.0001 * (l + 1) as f64;
            for (k, w) in h.weights.iter_mut().enumerate() {
                *w = 0.0002 * ((k + z as usize + l) % 3) as f64;
            }
        }
        heads.correction.weights[0] = 0.0001;
    }
    let mut opts = options();
    opts.ml_model = Some(Arc::new(model.clone()));
    let result = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: opts.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        (result.electronic_result.energy_terms().component_sum() - result.total_energy).abs()
            < 1e-13
    );
    for h in [2e-4, 1e-4] {
        let mut plus = system.clone();
        let mut minus = system.clone();
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
            (fd - result.gradient[1].x).abs() < 2e-7,
            "analytic {} FD {fd}",
            result.gradient[1].x
        );
    }
    let sensitivity = level_sensitivities(&system, &params, &result.electronic_result).unwrap();
    let grad = result
        .electronic_result
        .ml_parameters
        .as_ref()
        .unwrap()
        .energy_head_gradient(&sensitivity)
        .unwrap();
    let h = 1e-6;
    let mut plus = model.clone();
    let mut minus = model.clone();
    plus.heads.get_mut(&1).unwrap().levels[0].weights[2] += h;
    minus.heads.get_mut(&1).unwrap().levels[0].weights[2] -= h;
    let energy = |m| {
        let mut o = options();
        o.ml_model = Some(Arc::new(m));
        run_electronic(&system, &params, o).unwrap().total_free
    };
    let fd = (energy(plus) - energy(minus)) / (2.0 * h);
    assert!((fd - grad[&1].levels[0].weights[2]).abs() < 1e-7);
    let analytic = static_polarizability(&system, &params, &result.electronic_result).unwrap();
    let finite = static_polarizability_finite_field(&system, &params, &opts, 1e-4).unwrap();
    assert!((analytic.isotropic - finite.isotropic).abs() < 1e-4);
    assert!(analytic_hessian(
        &system,
        &params,
        AnalyticHessianOptions {
            electronic_options: opts,
            ..Default::default()
        }
    )
    .is_ok());
    let text = model.to_model_string().unwrap();
    assert_eq!(
        MlHamiltonian::from_model_str(&text)
            .unwrap()
            .to_model_string()
            .unwrap(),
        text
    );
}
