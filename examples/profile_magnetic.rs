// SPDX-License-Identifier: GPL-3.0-or-later
//! Wall-clock benchmark of the magnetic (v0.1.5/v0.1.6) property stack, to localize
//! optimization targets. `cargo run --release --example profile_magnetic`
//! (`GFN1_XTB_PARAM` is optional; the builtin parameters are used when unset).

use std::time::Instant;

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    cotton_mouton_tensor, magnetic_analytic_gradient, magnetic_h0_overlap, magnetic_polarizability,
    magnetizability_tensor_analytic, run_magnetic_scc, ElectronicOptions, ExternalFieldOptions,
    Gfn1Parameters, PeriodicSystem,
};

fn base_opts() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

fn bench<F: FnMut()>(name: &str, reps: usize, mut f: F) {
    // one warm-up
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / (reps as f64);
    println!("{name:48} {ms:9.2} ms/call");
}

fn main() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let xyz = std::fs::read_to_string("examples/bromoethanol.xyz").unwrap();
    let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
    let base = base_opts();
    let (h0, _) = magnetic_h0_overlap(&system, &params, &base, None).unwrap();
    println!("bromoethanol: {} atoms, {} AOs\n", system.atoms.len(), h0.n);

    bench("magnetic_h0_overlap (basis+core+LAO assemble)", 30, || {
        magnetic_h0_overlap(&system, &params, &base, None).unwrap();
    });
    bench("run_magnetic_scc (one full SCC)", 15, || {
        run_magnetic_scc(&system, &params, &base).unwrap();
    });
    bench("magnetizability_tensor_analytic", 5, || {
        magnetizability_tensor_analytic(&system, &params, &base, None, 0.004).unwrap();
    });
    bench("magnetic_polarizability (6 SCC)", 5, || {
        magnetic_polarizability(&system, &params, &base, None, 0.002).unwrap();
    });
    bench("magnetic_analytic_gradient (12N+1 assemble)", 3, || {
        magnetic_analytic_gradient(&system, &params, &base, None, 1.0e-3).unwrap();
    });
    bench("cotton_mouton_tensor (42 SCC)", 1, || {
        cotton_mouton_tensor(&system, &params, &base, None, 0.002, 0.02).unwrap();
    });
}
