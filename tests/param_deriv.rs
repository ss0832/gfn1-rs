// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-difference parameter-derivative checks (require `GFN1_XTB_PARAM`).

use gfn1_rs::{
    parameter_dipole_derivatives, parameter_finite_difference, parameter_hessian_derivatives,
    run_electronic, AnalyticHessianOptions, ElectronicOptions, Gfn1Parameters,
    ParamDerivativeOptions, ParameterTarget, PeriodicSystem,
};

fn h2() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "2\nH2\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n",
        0.0,
        false,
    )
    .unwrap()
}

#[test]
fn param_energy_derivative_matches_independent_central_difference() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    let system = h2();
    let electronic = ElectronicOptions::default();

    let target = ParameterTarget::parse("glob:ks").unwrap();
    let options = ParamDerivativeOptions {
        step: 1.0e-4,
        electronic: electronic.clone(),
        include_forces: false,
        include_stress: false,
    };
    let driver =
        parameter_finite_difference(&system, &params, &[target.clone()], &options).unwrap();
    let driver_deriv = driver[0].energy_derivative;

    // Independent central difference at a different step.
    let v0 = params.parameter_value(&target).unwrap();
    let h = 5.0e-4;
    let ep = run_electronic(
        &system,
        &params.with_parameter(&target, v0 + h).unwrap(),
        electronic.clone(),
    )
    .unwrap()
    .total_free;
    let em = run_electronic(
        &system,
        &params.with_parameter(&target, v0 - h).unwrap(),
        electronic,
    )
    .unwrap()
    .total_free;
    let manual = (ep - em) / (2.0 * h);

    assert!(driver_deriv.is_finite());
    assert!(
        driver_deriv.abs() > 1.0e-3,
        "ks derivative should be non-trivial for H2: {driver_deriv}"
    );
    assert!(
        (driver_deriv - manual).abs() < 1.0e-4,
        "driver {driver_deriv} vs independent CD {manual}"
    );
}

#[test]
fn dipole_and_hessian_parameter_derivatives_are_consistent() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    // Off-symmetric water so the dipole responds to parameters.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.80 0.55 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let targets = vec![
        ParameterTarget::parse("glob:ks").unwrap(),
        ParameterTarget::parse("elem:1:GAM").unwrap(),
    ];
    let electronic = ElectronicOptions::default();

    // dmu/dp: finite, and at least one component responds to `glob:ks`.
    let dmu =
        parameter_dipole_derivatives(&system, &params, &targets, &electronic, 1.0e-4).unwrap();
    assert_eq!(dmu.len(), 2);
    for (_, d) in &dmu {
        assert!(d.iter().all(|v| v.is_finite()));
    }
    let ks_norm: f64 = dmu[0].1.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        ks_norm > 1.0e-4,
        "dmu/d(ks) should be non-trivial: {ks_norm}"
    );

    // dH/dp: finite and symmetric (the Hessian is symmetric, so is its derivative).
    let dh = parameter_hessian_derivatives(
        &system,
        &params,
        &targets[..1],
        &AnalyticHessianOptions::default(),
        1.0e-4,
    )
    .unwrap();
    let mat = &dh[0].1;
    assert_eq!(mat.rows(), 9);
    let mut max_asym = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..mat.rows() {
        for j in 0..mat.cols() {
            assert!(mat[(i, j)].is_finite());
            max_asym = max_asym.max((mat[(i, j)] - mat[(j, i)]).abs());
            max_abs = max_abs.max(mat[(i, j)].abs());
        }
    }
    assert!(max_asym < 1.0e-6, "dH/dp not symmetric: {max_asym:.3e}");
    assert!(max_abs > 1.0e-4, "dH/d(ks) is unexpectedly zero");
}

#[test]
fn symmetric_h2_hardness_derivative_is_negligible() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    let system = h2();
    let target = ParameterTarget::parse("elem:1:GAM").unwrap();
    let derivs = parameter_finite_difference(
        &system,
        &params,
        &[target],
        &ParamDerivativeOptions::default(),
    )
    .unwrap();
    // Neutral, symmetric H2 has ~zero Mulliken charges, so the second-order SCC
    // energy (and hence its hardness derivative) is negligible.
    assert!(
        derivs[0].energy_derivative.abs() < 1.0e-6,
        "expected ~0 hardness derivative, got {}",
        derivs[0].energy_derivative
    );
}
