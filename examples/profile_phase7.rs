// SPDX-License-Identifier: GPL-3.0-or-later
//! Phase-7 optimization benchmark: the four wall-clock numbers the Phase-7 work
//! is measured against — SCF, analytic gradient, analytic Hessian and the
//! finite-temperature directional FC3 (the measured hotspot).
//!
//! Reports the MEDIAN of `reps` timed calls (after one untimed warm-up), so the
//! numbers are stable enough to compare across commits.
//!
//! ```text
//! cargo run --profile reltest --example profile_phase7            # all stages
//! cargo run --profile reltest --example profile_phase7 -- fc3     # only the FC3 hotspot
//! ```
//!
//! Fixtures (no fabricated geometries — everything comes from `examples/`):
//!
//! * `w8`   — the first 8 waters of `examples/water48.xyz` (24 atoms, caffeine
//!   sized): SCF / gradient / Hessian;
//! * `nico4` — the distorted Ni(CO)₄ of the finite-temperature FC3 gate
//!   (9 atoms / 27 DOF, 3000 K Fermi smearing): the directional FC3 hotspot.

use std::time::Instant;

use gfn1_rs::{
    analytic_gradient, analytic_hessian, directional_third_finite_t, fixed_density_cn_h0_hessian,
    fixed_density_pulay_hessian, run_electronic, AnalyticGradientOptions, AnalyticHessianOptions,
    ElectronicOptions, Gfn1Parameters, PeriodicSystem,
};

/// Median of `reps` timed calls in milliseconds, after one untimed warm-up.
/// `reps == 1` skips the warm-up — for the minutes-scale stages where a warm-up
/// would double the measurement for no statistical gain.
fn bench<F: FnMut()>(name: &str, reps: usize, mut f: F) {
    if reps > 1 {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let (lo, hi) = (samples[0], samples[samples.len() - 1]);
    println!("{name:44} median {median:11.2} ms   [min {lo:.2}, max {hi:.2}]");
}

/// The first `nat` atoms of `examples/water48.xyz` as a standalone molecule.
fn water_cluster(nat: usize) -> PeriodicSystem {
    let raw = std::fs::read_to_string("examples/water48.xyz")
        .expect("examples/water48.xyz (run from the crate root)");
    let lines: Vec<&str> = raw.lines().collect();
    let mut xyz = format!("{nat}\nwater cluster ({nat} atoms)\n");
    for line in lines.iter().skip(2).take(nat) {
        xyz.push_str(line);
        xyz.push('\n');
    }
    PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap()
}

/// The distorted Ni(CO)₄ of `directional_third_finite_t_matches_hessian_fd_smeared`.
fn ni_co4() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "9\ndistorted Ni(CO)4\n\
         Ni 0.020000 -0.030000 0.010000\n\
         C 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\n\
         C -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\n\
         C -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\n\
         C 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n",
        0.0,
        false,
    )
    .unwrap()
}

fn tight_options() -> ElectronicOptions {
    ElectronicOptions {
        enable_dispersion: false,
        energy_tolerance: 1.0e-12,
        charge_tolerance: 1.0e-10,
        ..ElectronicOptions::default()
    }
}

fn main() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let run_core = which == "all" || which == "core";
    let run_fc3 = which == "all" || which == "fc3";
    let run_blocks = which == "all" || which == "blocks";

    if run_blocks {
        // The two fixed-density AO-pair sweeps in isolation, on a SPATIALLY
        // EXTENDED cluster — the regime where the `P`/`W` screen inside those
        // sweeps actually fires, unlike the compact fixtures above. No CPXTB
        // response, so the sweep cost is not buried under an O(n³) solve.
        let nat: usize = std::env::var("BLOCK_ATOMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(144);
        let system = water_cluster(nat);
        let opts = tight_options();
        let cutoff = opts.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, opts).unwrap();
        println!(
            "-- water cluster: {} atoms, {} AOs (fixed-density blocks only) --",
            system.atoms.len(),
            electronic.basis.len()
        );
        bench("fixed_density_pulay_hessian", 3, || {
            fixed_density_pulay_hessian(&system, &params, &electronic).unwrap();
        });
        bench("fixed_density_cn_h0_hessian", 3, || {
            fixed_density_cn_h0_hessian(&system, &params, &electronic, cutoff).unwrap();
        });
    }

    if run_core {
        let system = water_cluster(24);
        let opts = tight_options();
        println!("-- water cluster: {} atoms --", system.atoms.len());
        bench("scf (run_electronic)", 5, || {
            run_electronic(&system, &params, opts.clone()).unwrap();
        });
        let grad_opts = AnalyticGradientOptions {
            electronic: opts.clone(),
            ..AnalyticGradientOptions::default()
        };
        bench("gradient (analytic_gradient)", 5, || {
            analytic_gradient(&system, &params, grad_opts.clone()).unwrap();
        });
        let hess_opts = AnalyticHessianOptions {
            electronic_options: opts.clone(),
            include_dispersion: false,
            ..AnalyticHessianOptions::default()
        };
        bench("hessian (analytic_hessian)", 3, || {
            analytic_hessian(&system, &params, hess_opts.clone()).unwrap();
        });
    }

    if which == "scf" {
        // Fermi-SMEARED SCF in isolation: the only path that reaches
        // `fermi_occupations`, whose chemical-potential bisection runs once per
        // SCC iteration. `run_electronic` at T = 0 takes the aufbau shortcut.
        let system = ni_co4();
        let opts = ElectronicOptions {
            enable_dispersion: false,
            electronic_temperature: 3000.0,
            energy_tolerance: 1.0e-14,
            charge_tolerance: 1.0e-12,
            ..ElectronicOptions::default()
        };
        let e = run_electronic(&system, &params, opts.clone()).unwrap();
        println!(
            "-- distorted Ni(CO)4, 3000 K: {} AOs, smeared SCF --",
            e.basis.len()
        );
        bench("scf (run_electronic, 3000 K)", 25, || {
            run_electronic(&system, &params, opts.clone()).unwrap();
        });
    }

    if run_fc3 {
        // The finite-temperature FC3 gate's own fixture and settings.
        let system = ni_co4();
        let mut options = AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.electronic_temperature = 3000.0;
        options.electronic_options.energy_tolerance = 1.0e-14;
        options.electronic_options.charge_tolerance = 1.0e-12;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        println!("-- distorted Ni(CO)4, 3000 K: {ndof} DOF --");
        // `FC3_REPS` (default 3) — set it to 1 for the slow pre-optimization
        // baseline, where a warm-up plus three reps would cost the better part
        // of an hour.
        let reps: usize = std::env::var("FC3_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        bench("hessian (analytic_hessian, smeared)", 3, || {
            analytic_hessian(&system, &params, options.clone()).unwrap();
        });
        let mut value = 0.0;
        bench("finite-T directional FC3", reps, || {
            value = directional_third_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        });
        // Printed so the optimization can be checked for value-preservation.
        println!("directional_third_finite_t = {value:.14e}");
    }
}
