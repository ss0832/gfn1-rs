// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration coverage for the periodic GFN1-xTB public API surface: the
//! `run_electronic` / `analytic_gradient` auto-dispatch to the PBC path, and the
//! agreement of the dispatched result with a direct `run_pbc_scc` call.
//!
//! These tests require an external `param_gfn1-xtb.txt` via `GFN1_XTB_PARAM` and
//! quietly no-op when it is absent (matching the rest of the suite).

use gfn1_rs::{
    analytic_gradient, run_electronic, run_pbc_scc, AnalyticGradientOptions, ElectronicOptions,
    Gfn1Parameters, PbcOptions, PeriodicSystem,
};

fn params() -> Option<Gfn1Parameters> {
    let path = std::env::var("GFN1_XTB_PARAM").ok()?;
    Gfn1Parameters::from_file(path).ok()
}

const DIAMOND: &str = "8\n\
Lattice=\"3.567 0 0 0 3.567 0 0 0 3.567\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.891750 0.891750 0.891750\n\
C 0.000000 1.783500 1.783500\n\
C 0.891750 2.675250 2.675250\n\
C 1.783500 0.000000 1.783500\n\
C 2.675250 0.891750 2.675250\n\
C 1.783500 1.783500 0.000000\n\
C 2.675250 2.675250 0.891750\n";

#[test]
fn run_electronic_dispatches_to_periodic_path() {
    let Some(params) = params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(DIAMOND, 0.0, false).unwrap();

    // The molecular entry point must auto-route a lattice-bearing system to the
    // Gamma-point PBC path and converge.
    let dispatched = run_electronic(&system, &params, ElectronicOptions::default()).unwrap();
    assert!(dispatched.converged);
    assert!(dispatched.total_free.is_finite());

    // It must agree with a direct Gamma-point periodic SCC call.
    let direct = run_pbc_scc(
        &system,
        &params,
        &ElectronicOptions::default(),
        &PbcOptions::default(),
    )
    .unwrap();
    assert!(
        (dispatched.total_free - direct.total_free).abs() < 1.0e-10,
        "dispatch {:.10} vs direct {:.10}",
        dispatched.total_free,
        direct.total_free
    );
    // Charge neutrality of the diamond cell.
    let qsum: f64 = dispatched.atomic_charges.iter().sum();
    assert!(qsum.abs() < 1.0e-6, "net charge {qsum:.2e}");
}

#[test]
fn analytic_gradient_dispatches_to_periodic_path() {
    let Some(params) = params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(DIAMOND, 0.0, false).unwrap();
    let result = analytic_gradient(&system, &params, AnalyticGradientOptions::default()).unwrap();
    assert!(result.electronic_result.converged);
    assert_eq!(result.gradient.len(), 8);
    assert!(result.max_gradient.is_finite());
    // The ideal diamond lattice is a symmetry equilibrium: forces vanish.
    assert!(
        result.max_gradient < 1.0e-4,
        "ideal diamond max gradient {:.3e} should vanish by symmetry",
        result.max_gradient
    );
}
