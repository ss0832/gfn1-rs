// SPDX-License-Identifier: GPL-3.0-or-later
//! External electric-field energy/gradient/dipole checks.

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions,
    ExternalFieldOptions, Gfn1Parameters, PeriodicSystem,
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

fn water_cell() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
         O 0.000000 0.000000 0.117300\n\
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

fn component(v: Vec3, axis: usize) -> f64 {
    v.to_array()[axis]
}

fn field_options(field: Vec3) -> ElectronicOptions {
    ElectronicOptions {
        external_field: ExternalFieldOptions::electric(field),
        ..ElectronicOptions::default()
    }
}

fn energy(system: &PeriodicSystem, params: &Gfn1Parameters, field: Vec3) -> f64 {
    run_electronic(system, params, field_options(field))
        .unwrap()
        .total_free
}

fn gradient_fd_error(system: &PeriodicSystem, params: &Gfn1Parameters, field: Vec3, h: f64) -> f64 {
    let grad_opts = AnalyticGradientOptions {
        electronic: field_options(field),
        ..AnalyticGradientOptions::default()
    };
    let analytic = analytic_gradient(system, params, grad_opts).unwrap();
    let mut max_err = 0.0_f64;
    for atom in 0..system.atoms.len() {
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, atom, axis, h);
            shift(&mut minus, atom, axis, -h);
            let fd = (energy(&plus, params, field) - energy(&minus, params, field)) / (2.0 * h);
            let an = component(analytic.gradient[atom], axis);
            max_err = max_err.max((fd - an).abs());
        }
    }
    max_err
}

#[test]
fn nonpbc_field_gradient_matches_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    let field = Vec3::new(0.01, -0.006, 0.004);
    let max_err = gradient_fd_error(&water(), &params, field, 1.0e-4);
    assert!(
        max_err < 1.0e-6,
        "non-PBC field gradient vs FD max error {max_err:.3e}"
    );
}

#[test]
fn pbc_gamma_field_gradient_matches_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    let field = Vec3::new(0.008, 0.0, -0.005);
    let max_err = gradient_fd_error(&water_cell(), &params, field, 1.0e-4);
    assert!(
        max_err < 1.0e-5,
        "PBC Gamma field gradient vs FD max error {max_err:.3e}"
    );
}

#[test]
fn dipole_is_negative_field_energy_derivative() {
    let Some(params) = load_params() else {
        return;
    };
    let system = water();
    let mu = run_electronic(&system, &params, ElectronicOptions::default())
        .unwrap()
        .dipole;
    assert!(mu.norm() > 1.0e-3, "water should have a non-zero dipole");

    let d = 1.0e-4;
    for axis in 0..3 {
        let mut plus = Vec3::zero();
        let mut minus = Vec3::zero();
        match axis {
            0 => {
                plus.x = d;
                minus.x = -d;
            }
            1 => {
                plus.y = d;
                minus.y = -d;
            }
            _ => {
                plus.z = d;
                minus.z = -d;
            }
        }
        let de = (energy(&system, &params, plus) - energy(&system, &params, minus)) / (2.0 * d);
        // E(F) = E0 - mu·F - ... so -dE/dF = mu at zero field.
        assert!(
            (-de - component(mu, axis)).abs() < 1.0e-5,
            "axis {axis}: -dE/dF {} vs dipole {}",
            -de,
            component(mu, axis)
        );
    }
}
