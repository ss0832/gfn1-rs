// SPDX-License-Identifier: GPL-3.0-or-later
//! Wall-clock benchmark of the molecular analytic **fourth** derivative (FC4).
//!
//! Reports the MEDIAN of `reps` timed calls (after one untimed warm-up) so the
//! numbers are comparable across commits, and — with `GFN1_PROFILE=1` — the
//! per-stage `gfn1_profile_ms` breakdown of the directional assembly.
//!
//! ```text
//! cargo run --profile reltest --example profile_fc4              # directional, both fixtures
//! cargo run --profile reltest --example profile_fc4 -- dir       # directional only
//! cargo run --profile reltest --example profile_fc4 -- dense     # water dense FC4
//! cargo run --profile reltest --example profile_fc4 -- big       # the >10-atom no-cap probe
//! GFN1_PROFILE=1 cargo run --profile reltest --example profile_fc4 -- dir 1
//! ```
//!
//! Fixtures (the gate geometries, so the timings describe gated code paths):
//!
//! * `water` — the non-equilibrium water of `fourth_derivative::assemble`'s
//!   integration gate (3 atoms / 9 DOF);
//! * `ch3br` — the CH3Br···OH2 complex of the stage-1 geometric gate
//!   (8 atoms / 24 DOF), the largest system the full-tensor D3/halogen fourths
//!   used to allow;
//! * `c4h10o` — n-butanol (15 atoms / 45 DOF), ABOVE the historical 30-DOF cap:
//!   the demonstration fixture for the directional 1D-jet geometric stage.

use std::time::Instant;

use gfn1_rs::{
    directional_fourth_derivative, fourth_derivative_analytic_dense, AnalyticHessianOptions,
    ElectronicOptions, Gfn1Parameters, PeriodicSystem,
};

/// Median of `reps` timed calls in milliseconds, after one untimed warm-up.
/// `reps == 1` skips the warm-up (the minutes-scale runs gain nothing from it).
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
    println!("{name:40} median {median:11.2} ms   [min {lo:.2}, max {hi:.2}]");
}

/// The non-equilibrium water of the FC4 integration gate.
fn water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
        0.0,
        false,
    )
    .unwrap()
}

/// The CH3Br···OH2 complex of the stage-1 geometric gate (8 atoms).
fn ch3br_oh2() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "8\nCH3Br...OH2\n\
         C 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\n\
         H 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\n\
         H -0.515000 -0.892000 -0.330000\nO 0.000000 0.100000 4.900000\n\
         H 0.760000 0.100000 5.470000\nH -0.760000 0.100000 5.470000\n",
        0.0,
        false,
    )
    .unwrap()
}

/// n-butanol, 15 atoms / 45 DOF — above the historical 30-DOF fourth-derivative cap.
fn butanol() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "15\nn-butanol\n\
         C -1.940000 -0.470000 0.060000\nC -0.690000 0.390000 -0.070000\n\
         C 0.590000 -0.430000 0.070000\nC 1.850000 0.410000 -0.060000\n\
         O 2.980000 -0.400000 0.090000\n\
         H -2.840000 0.090000 -0.040000\nH -1.970000 -1.230000 -0.730000\n\
         H -1.960000 -0.990000 1.020000\n\
         H -0.680000 1.170000 0.700000\nH -0.690000 0.900000 -1.040000\n\
         H 0.590000 -1.210000 -0.700000\nH 0.590000 -0.940000 1.040000\n\
         H 1.880000 1.190000 0.710000\nH 1.870000 0.920000 -1.030000\n\
         H 3.780000 0.130000 0.010000\n",
        0.0,
        false,
    )
    .unwrap()
}

fn tight_options(dispersion: bool) -> AnalyticHessianOptions {
    let electronic_options = ElectronicOptions {
        enable_dispersion: dispersion,
        energy_tolerance: 1.0e-12,
        charge_tolerance: 1.0e-10,
        ..ElectronicOptions::default()
    };
    AnalyticHessianOptions {
        electronic_options,
        ..AnalyticHessianOptions::default()
    }
}

/// The gate's fixed skew direction for a system of `ndof` degrees of freedom.
fn skew_direction(ndof: usize) -> Vec<f64> {
    (0..ndof)
        .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map(String::as_str).unwrap_or("all");
    let reps: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");

    if which == "all" || which == "dir" {
        for (name, system, dispersion) in [
            ("water(3at/9dof)", water(), false),
            ("water(3at/9dof,disp)", water(), true),
            ("ch3br_oh2(8at/24dof)", ch3br_oh2(), true),
        ] {
            let options = tight_options(dispersion);
            let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
            let v = skew_direction(3 * system.atoms.len());
            bench(&format!("fc4 directional {name}"), reps, || {
                directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap();
            });
        }
    }

    if which == "big" {
        let system = butanol();
        let options = tight_options(true);
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v = skew_direction(ndof);
        println!(
            "butanol: {ndof} DOF (the full-tensor D3/halogen cap is \
             MAX_FOURTH_DERIVATIVE_NDOF = 30)"
        );
        let t = Instant::now();
        match directional_fourth_derivative(&system, &params, &options, cutoff, &v) {
            Ok(value) => println!(
                "fc4 directional butanol(15at/{ndof}dof) = {value:.10e}   ({:.2} ms)",
                t.elapsed().as_secs_f64() * 1000.0
            ),
            Err(err) => println!("fc4 directional butanol(15at/{ndof}dof) FAILED: {err}"),
        }
        let t = Instant::now();
        match gfn1_rs::directional_fourth_seminumerical(
            &system, &params, &options, cutoff, &v, 1.0e-3,
        ) {
            Ok(value) => println!(
                "fc4 seminumerical butanol(15at/{ndof}dof) = {value:.10e}   ({:.2} ms)",
                t.elapsed().as_secs_f64() * 1000.0
            ),
            Err(err) => println!("fc4 seminumerical butanol(15at/{ndof}dof) FAILED: {err}"),
        }
    }

    // The DIRECTION-INDEPENDENT objects the per-direction stages rebuild, timed on their own:
    // the item-4 inventory. Anything here that is a large share of a stage is worth hoisting into
    // `QuarticReference`; anything that is not is reported and left alone.
    if which == "inventory" {
        let system = ch3br_oh2();
        let options = tight_options(true);
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let electronic = gfn1_rs::run_electronic(
            &system,
            &params,
            options.electronic_options.clone(),
        )
        .unwrap();
        let basis = &electronic.basis;
        bench("shell_scalar_potential_third_derivatives", 3, || {
            gfn1_rs::hessian::shell_scalar_potential_third_derivatives(
                &system,
                basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap();
        });
        bench("shell_scalar_potential_second_derivatives", 3, || {
            gfn1_rs::hessian::shell_scalar_potential_second_derivatives(
                &system,
                basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap();
        });
        bench("fixed_density_pulay_third_derivative", 3, || {
            gfn1_rs::hessian::fixed_density_pulay_third_derivative(&system, &params, &electronic)
                .unwrap();
        });
        bench("fixed_density_cn_h0_third_derivative", 3, || {
            gfn1_rs::hessian::fixed_density_cn_h0_third_derivative(
                &system,
                &params,
                &electronic,
                cutoff,
            )
            .unwrap();
        });
        bench("fixed_density_scalar_overlap_third_derivative", 3, || {
            gfn1_rs::hessian::fixed_density_scalar_overlap_third_derivative(
                &system,
                &params,
                &electronic,
            )
            .unwrap();
        });
        bench("fixed_density_pulay_hessian", 3, || {
            gfn1_rs::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap();
        });
    }

    if which == "all" || which == "dense" {
        let system = water();
        let options = tight_options(false);
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        bench("fc4 dense water(3at/9dof)", 1, || {
            fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap();
        });
    }
}
