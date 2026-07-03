// SPDX-License-Identifier: GPL-3.0-or-later
//! Wall-clock profiling harness for the periodic GFN1-xTB SCC / gradient /
//! Gamma-point Hessian. The binary `gfn1_rs --hessian` is the non-PBC path, so
//! this example drives the periodic functions directly.
//!
//! Run with phase breakdown:
//!   GFN1_PROFILE=1 cargo run --release --example profile_pbc -- examples/diamond.xyz
//! (set GFN1_XTB_PARAM to the parameter file path).

use gfn1_rs::pbc::hessian::pbc_gamma_hessian;
use gfn1_rs::pbc::pbc_stress_from_scc;
use gfn1_rs::{
    pbc_analytic_gradient, run_pbc_scc, ElectronicOptions, Gfn1Parameters, PbcOptions,
    PeriodicSystem,
};
use std::time::Instant;

fn main() {
    let xyz = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/diamond.xyz".to_string());
    let param = std::env::var("GFN1_XTB_PARAM").expect("set GFN1_XTB_PARAM to param_gfn1-xtb.txt");
    let params = Gfn1Parameters::from_file(param).unwrap();
    let system = PeriodicSystem::from_xyz_file(&xyz, 0.0, false).unwrap();

    let opts = ElectronicOptions {
        enable_dispersion: false,
        energy_tolerance: 1.0e-9,
        charge_tolerance: 1.0e-8,
        max_scc: 500,
        ..ElectronicOptions::default()
    };
    let pbc = PbcOptions::default(); // Gamma point

    let n = system.atoms.len();
    println!("system   {xyz} ({n} atoms, {} DOF)", 3 * n);

    let t = Instant::now();
    let scf = run_pbc_scc(&system, &params, &opts, &pbc).unwrap();
    println!(
        "scf      {:8.1} ms ({} iters)",
        t.elapsed().as_secs_f64() * 1e3,
        scf.iterations
    );

    let lattice = system
        .lattice
        .as_ref()
        .expect("profile input must be periodic");
    let t = Instant::now();
    let _s = pbc_stress_from_scc(&system, &params, scf.clone(), &opts, &pbc, lattice).unwrap();
    println!(
        "stress   {:8.1} ms (from converged SCC)",
        t.elapsed().as_secs_f64() * 1e3
    );

    let t = Instant::now();
    let _g = pbc_analytic_gradient(&system, &params, &opts, &pbc).unwrap();
    println!("gradient {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);

    // Pass a second arg ("nohess") to skip the (slow at scale) Hessian.
    if std::env::args().nth(2).as_deref() == Some("nohess") {
        return;
    }
    let t = Instant::now();
    let _h = pbc_gamma_hessian(&system, &params, &opts, &pbc).unwrap();
    println!("hessian  {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
}
