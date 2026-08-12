// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gate for the **uncapped** directional quartic.
//!
//! The full-tensor D3 and halogen fourth derivatives keep their
//! `MAX_FOURTH_DERIVATIVE_NDOF = 30` guard, because a full-space `Jet4` stores `ndof⁴` doubles
//! per jet. The DIRECTIONAL quartic does not need those tensors at all: a directional fourth
//! derivative is the fourth Taylor coefficient of `E(R + t·v)`, so the geometric stage carries a
//! univariate `Jet1` (five doubles) through the very same expressions.
//!
//! This gate pins the consequence end to end on a system ABOVE the cap: the full-tensor entry
//! points must still refuse it, and `directional_fourth_derivative` — with dispersion AND halogen
//! active — must both run and agree with the seminumerical reference (central FD along `v` of the
//! analytic third derivative), with the `h²` truncation scaling that separates FD noise from a
//! missing analytic term.

use gfn1_rs::fourth_derivative::{directional_fourth_derivative, directional_fourth_seminumerical};
use gfn1_rs::hessian::AnalyticHessianOptions;
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, MAX_FOURTH_DERIVATIVE_NDOF};

/// `examples/bromoethanol.xyz` halogen-bonded to a water placed on the extension of the C–Br
/// axis: 12 atoms / **36 DOF**, above the 30-DOF full-tensor cap, with the D3 (two-body + ATM),
/// halogen-bond, CN-Hamiltonian and third-order onsite Γ channels all live.
fn bromoethanol_water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "12\nbromoethanol...OH2 halogen bond\n\
         C 0.000000 0.000000 0.000000\n\
         C 1.520000 0.000000 0.000000\n\
         O 2.160000 1.220000 0.000000\n\
         Br -1.940000 0.000000 0.000000\n\
         H 0.220000 1.020000 0.000000\n\
         H 0.220000 -0.510000 0.884000\n\
         H 1.880000 -0.510000 -0.884000\n\
         H 1.880000 -0.510000 0.884000\n\
         H 2.960000 1.100000 0.500000\n\
         O -5.140000 0.000000 0.000000\n\
         H -5.500000 0.700000 0.480000\n\
         H -5.500000 -0.700000 0.480000\n",
        0.0,
        false,
    )
    .unwrap()
}

fn tight_options() -> AnalyticHessianOptions {
    AnalyticHessianOptions {
        electronic_options: ElectronicOptions {
            enable_dispersion: true,
            energy_tolerance: 1.0e-12,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        },
        ..AnalyticHessianOptions::default()
    }
}

fn skew_direction(ndof: usize) -> Vec<f64> {
    (0..ndof)
        .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
        .collect()
}

/// The full-tensor quartics still refuse a system above the cap — the guard this gate's
/// counterpart is measured against is real, not vestigial.
#[test]
fn full_tensor_geometric_quartics_still_refuse_systems_above_the_cap() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = bromoethanol_water();
    let ndof = 3 * system.atoms.len();
    assert!(
        ndof > MAX_FOURTH_DERIVATIVE_NDOF,
        "fixture must be above the cap: {ndof} DOF vs {MAX_FOURTH_DERIVATIVE_NDOF}"
    );
    assert!(
        gfn1_rs::dispersion_fourth_derivative(&system, &params, None).is_err(),
        "the full-tensor D3 quartic must still reject {ndof} DOF"
    );
    assert!(
        gfn1_rs::halogen::halogen_fourth_derivative(&system, &params).is_err(),
        "the full-tensor halogen quartic must still reject {ndof} DOF"
    );
}

/// **The uncapped directional gate.** With dispersion and halogen on, the analytic directional
/// quartic must run at 36 DOF and match the central FD of the analytic third derivative along the
/// same direction. Two FD steps assert the `h²` truncation scaling.
#[test]
fn directional_quartic_runs_and_matches_fd_above_the_cap() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = bromoethanol_water();
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    assert!(ndof > MAX_FOURTH_DERIVATIVE_NDOF);
    assert!(options.include_dispersion && options.include_halogen);
    let v = skew_direction(ndof);

    let analytic = directional_fourth_derivative(&system, &params, &options, cutoff, &v)
        .expect("the directional quartic must not be capped");
    let fd_at = |h: f64| {
        directional_fourth_seminumerical(&system, &params, &options, cutoff, &v, h).unwrap()
    };
    let h1 = 1.0e-3;
    let fd1 = fd_at(h1);
    let delta1 = (analytic - fd1).abs();
    let fd2 = fd_at(0.5 * h1);
    let delta2 = (analytic - fd2).abs();
    eprintln!(
        "uncapped directional quartic ({ndof} DOF): analytic {analytic:.10e} fd(h) {fd1:.10e} \
         fd(h/2) {fd2:.10e} delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
        delta1 / delta2.max(1.0e-300)
    );
    assert!(
        delta1 < 1.0e-6 * (1.0 + fd1.abs()),
        "uncapped directional quartic vs FD(analytic third): analytic {analytic:.10e} \
         fd {fd1:.10e} delta {delta1:.3e}"
    );
    assert!(
        delta2 < 0.4 * delta1,
        "residual does not scale as h² (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
         suspect a missing or double-counted analytic term"
    );
}
