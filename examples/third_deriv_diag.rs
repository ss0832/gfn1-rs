// SPDX-License-Identifier: GPL-3.0-or-later
//! Third-derivative diagnostic bisection harness (v0.5.0 Phase 1).
//!
//! Compares the closed-form analytic third derivative against the
//! semi-numerical reference (central FD of the analytic Hessian) over a matrix
//! of molecules × option toggles, to localize any residual analytic error.
//!
//! Run: `cargo run --profile reltest --example third_deriv_diag [-- name-filter]`
//! (`GFN1_XTB_PARAM` optional — bundled parameters used when unset.)

use gfn1_rs::hessian::AnalyticHessianOptions;
use gfn1_rs::third_derivative::{
    third_derivative_analytic_dense, third_derivative_seminumerical_dense,
};
use gfn1_rs::{Gfn1Parameters, PeriodicSystem};

struct Case {
    name: &'static str,
    xyz: &'static str,
    charge: f64,
}

const CASES: &[Case] = &[
    Case {
        name: "water_eq",
        xyz: "3\nwater\nO 0.000000 0.000000 0.119262\nH 0.000000 0.763239 -0.477047\nH 0.000000 -0.763239 -0.477047\n",
        charge: 0.0,
    },
    Case {
        name: "water_stretched",
        xyz: "3\nwater OH+0.16A\nO 0.000000 0.000000 0.119262\nH 0.000000 0.895239 -0.580047\nH 0.000000 -0.763239 -0.477047\n",
        charge: 0.0,
    },
    Case {
        name: "water_bent",
        xyz: "3\nwater bent\nO 0.000000 0.000000 0.119262\nH 0.000000 0.680000 -0.590000\nH 0.000000 -0.763239 -0.477047\n",
        charge: 0.0,
    },
    Case {
        name: "ammonia",
        xyz: "4\nnh3\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\nH 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n",
        charge: 0.0,
    },
    Case {
        name: "h3o_plus",
        xyz: "4\nh3o+\nO 0.000000 0.000000 0.074000\nH 0.000000 0.939000 -0.198000\nH 0.813000 -0.469000 -0.198000\nH -0.813000 -0.469000 -0.198000\n",
        charge: 1.0,
    },
    Case {
        name: "hf_dimer",
        xyz: "4\n(HF)2\nF 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.922000\nF 0.000000 0.000000 2.720000\nH 0.000000 0.760000 3.220000\n",
        charge: 0.0,
    },
    Case {
        name: "methane_td",
        xyz: "5\nmethane (exact Td, triply degenerate t2 HOMO)\nC 0.000000 0.000000 0.000000\nH 0.629118 0.629118 0.629118\nH -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\nH 0.629118 -0.629118 -0.629118\n",
        charge: 0.0,
    },
    Case {
        name: "methane_dist",
        xyz: "5\nmethane (symmetry broken, no degeneracy)\nC 0.010000 -0.020000 0.005000\nH 0.649118 0.619118 0.639118\nH -0.629118 -0.609118 0.629118\nH -0.619118 0.629118 -0.649118\nH 0.629118 -0.639118 -0.619118\n",
        charge: 0.0,
    },
    Case {
        name: "ch3br_water",
        xyz: "8\nCH3Br...OH2 (halogen bond, closed shell)\nC 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\nH 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\nH -0.515000 -0.892000 -0.330000\nO 0.000000 0.100000 4.900000\nH 0.760000 0.100000 5.470000\nH -0.760000 0.100000 5.470000\n",
        charge: 0.0,
    },
];

/// Clone `params` with every element's DFTB3 third-order onsite Γ zeroed
/// (re-derived through the canonical text round-trip so all derived fields
/// stay consistent, mirroring `with_parameter`).
fn zero_gamma3(params: &Gfn1Parameters) -> Gfn1Parameters {
    let mut clone = params.clone();
    for elem in clone.elements.values_mut() {
        elem.gam3_raw = 0.0;
        if let Some(v) = elem.raw.get_mut("GAM3") {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
    }
    Gfn1Parameters::from_str(&clone.to_param_string()).expect("gamma3-zeroed params reparse")
}

fn run_case(
    label: &str,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    dispersion: bool,
    step: f64,
) {
    let mut options = AnalyticHessianOptions::default();
    options.include_dispersion = dispersion;
    options.electronic_options.enable_dispersion = dispersion;
    options.electronic_options.charge = Some(system.charge);
    // Tight SCF so the FD reference is clean.
    options.electronic_options.energy_tolerance = 1.0e-11;
    options.electronic_options.charge_tolerance = 1.0e-9;

    let coordination_cutoff = 30.0;
    let ana = match third_derivative_analytic_dense(system, params, options.clone(), coordination_cutoff)
    {
        Ok(t) => t,
        Err(err) => {
            println!("{label:<44} ANALYTIC ERROR: {err}");
            return;
        }
    };
    let semi = match third_derivative_seminumerical_dense(system, params, options, step) {
        Ok(t) => t,
        Err(err) => {
            println!("{label:<44} SEMINUMERICAL ERROR: {err}");
            return;
        }
    };

    let ndof = 3 * system.atoms.len();
    let mut max_abs = 0.0_f64;
    let mut max_at = (0usize, 0usize, 0usize);
    let mut tensor_scale = 0.0_f64;
    for c in 0..ndof {
        for b in 0..=c {
            for a in 0..=b {
                let va = ana.get(a, b, c);
                let vs = semi.get(a, b, c);
                tensor_scale = tensor_scale.max(vs.abs());
                let d = (va - vs).abs();
                if d > max_abs {
                    max_abs = d;
                    max_at = (a, b, c);
                }
            }
        }
    }
    let rel = if tensor_scale > 0.0 {
        max_abs / tensor_scale
    } else {
        0.0
    };
    let (a, b, c) = max_at;
    println!(
        "{label:<44} max|Δ| {max_abs:10.3e}  rel {rel:9.3e}  scale {tensor_scale:9.3e}  argmax ({a},{b},{c})"
    );
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    println!("parameters: {}", params.source_description());
    let params_g0 = zero_gamma3(&params);
    let step = 2.0e-4;
    println!("seminumerical FD step: {step:.1e} bohr\n");

    for case in CASES {
        if !filter.is_empty() && !case.name.contains(&filter) {
            continue;
        }
        let system = PeriodicSystem::from_xyz_str(case.xyz, case.charge, false)
            .unwrap_or_else(|e| panic!("bad xyz for {}: {e}", case.name));
        run_case(
            &format!("{} [disp,g3]", case.name),
            &system,
            &params,
            true,
            step,
        );
        run_case(
            &format!("{} [nodisp,g3]", case.name),
            &system,
            &params,
            false,
            step,
        );
        run_case(
            &format!("{} [disp,g3=0]", case.name),
            &system,
            &params_g0,
            true,
            step,
        );
        run_case(
            &format!("{} [nodisp,g3=0]", case.name),
            &system,
            &params_g0,
            false,
            step,
        );
        println!();
    }
}
