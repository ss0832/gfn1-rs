// SPDX-License-Identifier: GPL-3.0-or-later
//! IR / Raman / polarizability checks.

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    dipole_derivatives, ir_spectrum, raman_spectrum, run_electronic, static_polarizability,
    AnalyticHessianOptions, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters,
    PeriodicSystem,
};

fn load_params() -> Option<Gfn1Parameters> {
    Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
}

fn water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\n\
         H 0.000000 -0.757200 -0.469200\n",
        0.0,
        false,
    )
    .unwrap()
}

fn shift(system: &mut PeriodicSystem, atom: usize, axis: usize, delta: f64) {
    match axis {
        0 => system.atoms[atom].position.x += delta,
        1 => system.atoms[atom].position.y += delta,
        _ => system.atoms[atom].position.z += delta,
    }
}

fn tight() -> ElectronicOptions {
    ElectronicOptions {
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    }
}

fn dipole(system: &PeriodicSystem, params: &Gfn1Parameters) -> Vec3 {
    run_electronic(system, params, tight()).unwrap().dipole
}

#[test]
fn analytic_dipole_derivatives_match_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    let system = water();
    let electronic = run_electronic(&system, &params, tight()).unwrap();
    let analytic = dipole_derivatives(&system, &params, &electronic, Vec3::zero()).unwrap();

    let h = 1.0e-4;
    let mut max_err = 0.0_f64;
    for atom in 0..system.atoms.len() {
        for axis in 0..3 {
            let coord = 3 * atom + axis;
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, atom, axis, h);
            shift(&mut minus, atom, axis, -h);
            let dmu = (dipole(&plus, &params) - dipole(&minus, &params)) * (1.0 / (2.0 * h));
            let fd = dmu.to_array();
            for alpha in 0..3 {
                max_err = max_err.max((fd[alpha] - analytic.ddipole_dr[coord][alpha]).abs());
            }
        }
    }
    assert!(
        max_err < 1.0e-5,
        "analytic dmu/dR vs finite difference max error {max_err:.3e}"
    );
}

#[test]
fn polarizability_matches_energy_second_derivative() {
    let Some(params) = load_params() else {
        return;
    };
    let system = water();
    let electronic = run_electronic(&system, &params, ElectronicOptions::default()).unwrap();
    let pol = static_polarizability(&system, &params, &electronic).unwrap();

    assert!(
        pol.isotropic > 0.0,
        "isotropic polarizability must be positive"
    );
    // Symmetric tensor.
    for a in 0..3 {
        for b in 0..3 {
            assert!((pol.tensor[a][b] - pol.tensor[b][a]).abs() < 1.0e-8);
        }
    }

    // alpha_zz = -d^2 E / dE_z^2 by finite field of the energy (independent route).
    let d = 2.0e-3;
    let energy = |field: Vec3| {
        let mut opts = ElectronicOptions::default();
        opts.external_field = ExternalFieldOptions::electric(field);
        run_electronic(&system, &params, opts).unwrap().total_free
    };
    let e0 = energy(Vec3::zero());
    let ep = energy(Vec3::new(0.0, 0.0, d));
    let em = energy(Vec3::new(0.0, 0.0, -d));
    let alpha_zz_energy = -(ep + em - 2.0 * e0) / (d * d);
    assert!(
        (alpha_zz_energy - pol.tensor[2][2]).abs() < 1.0e-2 * pol.tensor[2][2].abs().max(1.0),
        "alpha_zz energy route {alpha_zz_energy} vs dipole route {}",
        pol.tensor[2][2]
    );
}

#[test]
fn ir_and_raman_spectra_run() {
    let Some(params) = load_params() else {
        return;
    };
    let system = water();
    let ndim = 3 * system.atoms.len();

    let ir = ir_spectrum(
        &system,
        &params,
        AnalyticHessianOptions::default(),
        Vec3::zero(),
    )
    .unwrap();
    assert_eq!(ir.modes.len(), ndim);
    assert!(ir
        .modes
        .iter()
        .all(|m| m.intensity_au.is_finite() && m.intensity_au >= 0.0));
    // Water has IR-active stretching/bending modes.
    let max_ir = ir.modes.iter().map(|m| m.intensity_au).fold(0.0, f64::max);
    assert!(max_ir > 0.0, "expected a non-zero IR intensity");

    let raman = raman_spectrum(
        &system,
        &params,
        AnalyticHessianOptions::default(),
        Vec3::zero(),
        1.0e-3,
    )
    .unwrap();
    assert_eq!(raman.modes.len(), ndim);
    assert!(raman
        .modes
        .iter()
        .all(|m| m.activity.is_finite() && m.activity >= 0.0));
    let max_raman = raman.modes.iter().map(|m| m.activity).fold(0.0, f64::max);
    assert!(max_raman > 0.0, "expected a non-zero Raman activity");
}
