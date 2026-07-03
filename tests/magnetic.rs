// SPDX-License-Identifier: GPL-3.0-or-later
//! Magnetic (GFN1-xTB-M0) SCC checks (need `GFN1_XTB_PARAM`).

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    analytic_gradient, magnetic_analytic_gradient, magnetic_gradient, magnetizability_isotropic,
    parse_secondary_basis, run_electronic, run_magnetic_scc, run_magnetic_scc_m1,
    AnalyticGradientOptions, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters,
    PeriodicSystem, SecondaryBasis, MAGNETIZABILITY_AU_TO_SI,
};

/// Load the GFN1-xTB-M1 secondary basis from the path in `GFN1_M1_BASIS` (the
/// paper's `$Basis = GFN1-xTB-cc-pVDZ` file). Tests no-op when it is absent.
fn load_m1_basis() -> Option<SecondaryBasis> {
    let path = std::env::var("GFN1_M1_BASIS").ok()?;
    let text = std::fs::read_to_string(path).ok()?;
    parse_secondary_basis(&text).ok()
}

fn load_params() -> Option<Gfn1Parameters> {
    let path = std::env::var("GFN1_XTB_PARAM").ok()?;
    Gfn1Parameters::from_file(path).ok()
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";

fn opts_with_field(b: Vec3) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(b),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

#[test]
fn magnetic_scc_reduces_to_field_free_and_responds_to_field() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

    // Field-free GFN1 reference (T = 0 internal energy: band + SCC + rep/disp/hal).
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let e_ref = run_electronic(&system, &params, ref_opts)
        .unwrap()
        .total_internal;

    // B = 0 magnetic SCC must reproduce the field-free energy exactly.
    let m0 = run_magnetic_scc(&system, &params, &opts_with_field(Vec3::zero())).unwrap();
    assert!(m0.converged);
    assert!(
        (m0.energy - e_ref).abs() < 1.0e-8,
        "B=0 magnetic energy {} vs field-free {} (diff {:.3e})",
        m0.energy,
        e_ref,
        (m0.energy - e_ref).abs()
    );

    // A finite field perpendicular to the molecular plane: real, finite, converged,
    // and with a measurable effect on the energy. (The sign of the M0 magnetic
    // response is not guaranteed without the kinetic-energy correction of the
    // dual-basis M1 variant, so only the magnitude is checked here.)
    let mb = run_magnetic_scc(
        &system,
        &params,
        &opts_with_field(Vec3::new(0.0, 0.0, 0.05)),
    )
    .unwrap();
    assert!(mb.converged && mb.energy.is_finite());
    assert!(
        (mb.energy - m0.energy).abs() > 1.0e-7,
        "the magnetic field has no effect on the energy: B {} vs 0 {}",
        mb.energy,
        m0.energy
    );

    // The field couples through the gauge-origin-dependent London phase but the
    // energy is gauge-origin invariant for a neutral closed shell: shifting the
    // origin must not change the energy.
    let shifted = ExternalFieldOptions {
        magnetic_field: Some(Vec3::new(0.0, 0.0, 0.05)),
        origin: Vec3::new(1.3, -0.7, 0.4),
        ..ExternalFieldOptions::default()
    };
    let opts_shift = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: shifted,
        ..ElectronicOptions::default()
    };
    let ms = run_magnetic_scc(&system, &params, &opts_shift).unwrap();
    assert!(
        (ms.energy - mb.energy).abs() < 1.0e-7,
        "magnetic energy is not gauge-origin invariant: {} vs {}",
        ms.energy,
        mb.energy
    );
}

#[test]
fn m1_reduces_to_field_free_differs_from_m0_and_is_gauge_invariant() {
    let Some(params) = load_params() else {
        return;
    };
    let Some(secondary) = load_m1_basis() else {
        return; // needs GFN1_M1_BASIS
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

    // B = 0: M1 reproduces the field-free GFN1 energy (the KE correction vanishes).
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let e_ref = run_electronic(&system, &params, ref_opts)
        .unwrap()
        .total_internal;
    let m1_0 =
        run_magnetic_scc_m1(&system, &params, &opts_with_field(Vec3::zero()), &secondary).unwrap();
    assert!(
        (m1_0.energy - e_ref).abs() < 1.0e-8,
        "M1 B=0 energy {} vs field-free {}",
        m1_0.energy,
        e_ref
    );

    // Finite field: M1 must differ from M0 (the secondary basis changes the KE term).
    let bz = Vec3::new(0.0, 0.0, 0.08);
    let m0_b = run_magnetic_scc(&system, &params, &opts_with_field(bz)).unwrap();
    let m1_b = run_magnetic_scc_m1(&system, &params, &opts_with_field(bz), &secondary).unwrap();
    assert!(
        (m1_b.energy - m0_b.energy).abs() > 1.0e-7,
        "M1 energy equals M0 — secondary basis appears inactive ({} vs {})",
        m1_b.energy,
        m0_b.energy
    );

    // Gauge-origin invariance of the M1 energy.
    let shifted = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(bz),
            origin: Vec3::new(1.3, -0.7, 0.4),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    };
    let m1_s = run_magnetic_scc_m1(&system, &params, &shifted, &secondary).unwrap();
    assert!(
        (m1_s.energy - m1_b.energy).abs() < 1.0e-7,
        "M1 energy is not gauge-origin invariant: {} vs {}",
        m1_s.energy,
        m1_b.energy
    );

    // Isotropic magnetizabilities (printed for comparison with the paper / literature).
    let base = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    };
    let xi_m0 = magnetizability_isotropic(&system, &params, &base, None, 0.02).unwrap();
    let xi_m1 = magnetizability_isotropic(&system, &params, &base, Some(&secondary), 0.02).unwrap();
    eprintln!(
        "H2O isotropic magnetizability (10^-30 J/T^2): M0 = {:.3}, M1 = {:.3}",
        xi_m0 * MAGNETIZABILITY_AU_TO_SI,
        xi_m1 * MAGNETIZABILITY_AU_TO_SI
    );
    assert!(xi_m0.is_finite() && xi_m1.is_finite());
    assert!(
        (xi_m1 - xi_m0).abs() > 1.0e-12,
        "M1 magnetizability equals M0"
    );
}

#[test]
fn magnetic_gradient_matches_field_free_at_zero_and_responds_to_field() {
    let Some(params) = load_params() else {
        return;
    };
    // Off-equilibrium water so the gradient is non-trivial.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let step = 1.0e-3;

    // At B = 0 the magnetic (M0) gradient is the finite difference of the
    // field-free internal energy and must reproduce the field-free analytic
    // nuclear gradient.
    let g0 = magnetic_gradient(&system, &params, &opts_with_field(Vec3::zero()), step).unwrap();
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let ana = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: ref_opts,
            ..AnalyticGradientOptions::default()
        },
    )
    .unwrap();
    let mut max_diff = 0.0_f64;
    for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
        max_diff = max_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_diff < 5.0e-5,
        "B=0 magnetic gradient vs field-free analytic gradient: max diff {max_diff:.3e}"
    );

    // forces = -gradient.
    for (g, f) in g0.gradient.iter().zip(g0.forces.iter()) {
        assert!((g.x + f.x).abs() < 1.0e-14 && (g.y + f.y).abs() < 1.0e-14);
    }

    // A finite field changes the forces (real, finite) and the gradient remains
    // step-consistent between two finite-difference steps.
    let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
    let gb1 = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
    let gb2 = magnetic_gradient(&system, &params, &field, 2.0e-3).unwrap();
    let mut step_diff = 0.0_f64;
    let mut field_diff = 0.0_f64;
    for ((a, b), z) in gb1
        .gradient
        .iter()
        .zip(gb2.gradient.iter())
        .zip(g0.gradient.iter())
    {
        assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite());
        step_diff = step_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
        field_diff = field_diff
            .max((a.x - z.x).abs())
            .max((a.y - z.y).abs())
            .max((a.z - z.z).abs());
    }
    assert!(
        step_diff < 1.0e-4,
        "magnetic gradient not step-consistent: {step_diff:.3e}"
    );
    assert!(
        field_diff > 1.0e-7,
        "the field did not change the forces: {field_diff:.3e}"
    );
}

#[test]
fn magnetic_analytic_gradient_matches_field_free_and_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    // Off-equilibrium water so the gradient is non-trivial.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();

    // At B = 0 the analytic (Hellmann-Feynman) magnetic gradient must reproduce the
    // field-free GFN1 analytic nuclear gradient.
    let g0 = magnetic_analytic_gradient(
        &system,
        &params,
        &opts_with_field(Vec3::zero()),
        None,
        1.0e-3,
    )
    .unwrap();
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let ana = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: ref_opts,
            ..AnalyticGradientOptions::default()
        },
    )
    .unwrap();
    let mut max_b0 = 0.0_f64;
    for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
        max_b0 = max_b0
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_b0 < 5.0e-5,
        "B=0 analytic magnetic gradient vs field-free analytic gradient: {max_b0:.3e}"
    );

    // At a finite field the analytic gradient must match the finite-difference of the
    // converged magnetic energy ([`magnetic_gradient`]) — the key correctness check
    // of the Hellmann-Feynman density/Pulay contraction.
    let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
    let ga = magnetic_analytic_gradient(&system, &params, &field, None, 1.0e-3).unwrap();
    let gfd = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
    let mut max_fd = 0.0_f64;
    for (a, b) in ga.gradient.iter().zip(gfd.gradient.iter()) {
        max_fd = max_fd
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_fd < 1.0e-4,
        "analytic vs finite-difference magnetic gradient at finite B: {max_fd:.3e}"
    );

    // forces = -gradient.
    for (g, f) in ga.gradient.iter().zip(ga.forces.iter()) {
        assert!((g.x + f.x).abs() < 1.0e-14 && (g.z + f.z).abs() < 1.0e-14);
    }
}
