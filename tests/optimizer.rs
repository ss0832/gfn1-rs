// SPDX-License-Identifier: GPL-3.0-or-later

use gfn1_rs::{optimize_geometry, GeometryOptimizationOptions, Gfn1Parameters, PeriodicSystem};

#[test]
fn h2_native_lbfgs_optimization_converges() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    let system = PeriodicSystem::from_xyz_str(
        "2\nH2 stretched\nH 0.000000 0.000000 0.000000\nH 1.100000 0.000000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let options = GeometryOptimizationOptions {
        max_iterations: 40,
        gradient_tolerance: 2.0e-4,
        ..GeometryOptimizationOptions::default()
    };
    let result = optimize_geometry(&system, &params, options).unwrap();
    assert!(result.converged, "native L-BFGS did not converge");
    assert!(result.max_gradient < 2.0e-4);
}

/// v0.2.1 Feature B (Stage 1): the L-BFGS optimizer now accepts **periodic** systems and relaxes
/// the atomic positions at fixed cell (the Γ-point PBC gradient routes through `analytic_gradient`,
/// and `system_with_positions` preserves the lattice). A diamond cell with one atom displaced from
/// its ideal site must relax: the energy drops, the max gradient shrinks, and the lattice survives.
#[test]
fn pbc_gamma_optimization_lowers_energy() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    // Diamond cubic cell (8 C) with atom 0 displaced ~0.15 Å off its ideal site.
    let perturbed = "8\n\
Lattice=\"3.567 0 0 0 3.567 0 0 0 3.567\" pbc=\"T T T\"\n\
C 0.150000 0.080000 0.030000\n\
C 0.891750 0.891750 0.891750\n\
C 0.000000 1.783500 1.783500\n\
C 0.891750 2.675250 2.675250\n\
C 1.783500 0.000000 1.783500\n\
C 2.675250 0.891750 2.675250\n\
C 1.783500 1.783500 0.000000\n\
C 2.675250 2.675250 0.891750\n";
    let system = PeriodicSystem::from_xyz_str(perturbed, 0.0, false).unwrap();
    let options = GeometryOptimizationOptions {
        max_iterations: 20,
        gradient_tolerance: 1.0e-3,
        ..GeometryOptimizationOptions::default()
    };
    let result = optimize_geometry(&system, &params, options).unwrap();

    // Fixed-cell: the lattice must be preserved through the optimization.
    assert!(
        result.system.lattice.is_some(),
        "periodic optimization dropped the lattice"
    );
    let e_initial = result.trajectory.first().unwrap().energy;
    let g_initial = result.trajectory.first().unwrap().max_gradient;
    assert!(
        result.energy < e_initial - 1.0e-6,
        "PBC optimization did not lower the energy: {e_initial:.8} -> {:.8}",
        result.energy
    );
    assert!(
        result.max_gradient < g_initial,
        "PBC optimization did not reduce the max gradient: {g_initial:.3e} -> {:.3e}",
        result.max_gradient
    );
}
