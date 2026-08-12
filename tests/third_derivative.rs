// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gates for the analytic nuclear third derivative (cubic force
//! constants): analytic vs seminumerical (central FD of the analytic Hessian)
//! on a non-equilibrium geometry, a degenerate-orbital symmetric top, and the
//! memory-lean vector output mode. These are the compact, always-on versions of
//! the fine-grained in-module gates in `src/third_derivative.rs`.

use gfn1_rs::hessian::AnalyticHessianOptions;
use gfn1_rs::third_derivative::{
    third_derivative_analytic_dense, third_derivative_analytic_vector,
    third_derivative_seminumerical_dense,
};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};

fn tight_options() -> AnalyticHessianOptions {
    AnalyticHessianOptions {
        electronic_options: ElectronicOptions {
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        },
        ..AnalyticHessianOptions::default()
    }
}

fn max_error_vs_seminumerical(system: &PeriodicSystem, params: &Gfn1Parameters) -> (f64, f64) {
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    let ana = third_derivative_analytic_dense(system, params, options.clone(), cutoff).unwrap();
    let semi = third_derivative_seminumerical_dense(system, params, options, 1.0e-4).unwrap();
    let mut err = 0.0_f64;
    let mut scale = 0.0_f64;
    for c in 0..ndof {
        for b in 0..=c {
            for a in 0..=b {
                err = err.max((ana.get(a, b, c) - semi.get(a, b, c)).abs());
                scale = scale.max(semi.get(a, b, c).abs());
            }
        }
    }
    (err, scale)
}

/// Non-equilibrium stretched+bent water with the third-order onsite Γ active —
/// the regression gate for the historical missing ∂K/∂q kernel-chain term.
#[test]
fn analytic_third_derivative_matches_seminumerical_noneq_water() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
        0.0,
        false,
    )
    .unwrap();
    let (err, scale) = max_error_vs_seminumerical(&system, &params);
    assert!(
        err < 5.0e-7,
        "non-eq water analytic vs seminumerical third derivative: err={err:.3e} (scale {scale:.3e})"
    );
}

/// C3v ammonia with exactly degenerate e-orbitals — the regression gate for the
/// historical degenerate-orbital bug (~2e-2 relative error before v0.5.0).
#[test]
fn analytic_third_derivative_matches_seminumerical_degenerate_ammonia() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "4\nnh3 C3v\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\n\
         H 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n",
        0.0,
        false,
    )
    .unwrap();
    let (err, scale) = max_error_vs_seminumerical(&system, &params);
    assert!(
        err < 5.0e-7,
        "degenerate NH3 analytic vs seminumerical third derivative: err={err:.3e} (scale {scale:.3e})"
    );
}

/// v0.5.0 regression: with `charge_order = 4` the third derivative's ∂K/∂q
/// chain must use the full anharmonic ∂³E_onsite/∂q³ (not just the DFTB3 2Γ),
/// consistent with the extended response kernel. The seminumerical reference
/// differentiates the (now-consistent) analytic Hessian, so agreement here
/// validates the whole charge_order ≥ 4 response chain end to end.
#[test]
fn analytic_third_derivative_matches_seminumerical_charge_order_4() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
        0.0,
        false,
    )
    .unwrap();
    let mut options = tight_options();
    options.electronic_options.charge_order = 4;
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    let ana = third_derivative_analytic_dense(&system, &params, options.clone(), cutoff).unwrap();
    let semi = third_derivative_seminumerical_dense(&system, &params, options, 1.0e-4).unwrap();
    let mut err = 0.0_f64;
    let mut scale = 0.0_f64;
    for c in 0..ndof {
        for b in 0..=c {
            for a in 0..=b {
                err = err.max((ana.get(a, b, c) - semi.get(a, b, c)).abs());
                scale = scale.max(semi.get(a, b, c).abs());
            }
        }
    }
    assert!(
        err < 5.0e-7,
        "charge_order=4 analytic vs seminumerical third derivative: err={err:.3e} (scale {scale:.3e})"
    );
}

/// The memory-lean vector mode `K[a][b] = Σ_c v_c T_abc` must agree with the
/// dense tensor contracted along the same direction.
#[test]
fn analytic_vector_mode_matches_dense_contraction() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.119262\nH 0.0 0.763239 -0.477047\nH 0.0 -0.763239 -0.477047\n",
        0.0,
        false,
    )
    .unwrap();
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    let v: Vec<f64> = (0..ndof).map(|k| 0.3 + 0.1 * k as f64).collect();
    let dense = third_derivative_analytic_dense(&system, &params, options.clone(), cutoff).unwrap();
    let kmat = third_derivative_analytic_vector(&system, &params, options, cutoff, &v).unwrap();
    let mut err = 0.0_f64;
    for a in 0..ndof {
        for b in 0..ndof {
            let mut want = 0.0;
            for (c, &vc) in v.iter().enumerate() {
                want += vc * dense.get(a, b, c);
            }
            err = err.max((kmat[(a, b)] - want).abs());
        }
    }
    assert!(
        err < 1.0e-10,
        "vector mode vs dense contraction: err={err:.3e}"
    );
}
