// SPDX-License-Identifier: GPL-3.0-or-later
//! Frame invariance of the analytic magnetizability tensor.
//!
//! Calibration ladders (`#[ignore]`d) plus the always-on regression gates.

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    magnetizability_tensor_analytic, Atom, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters,
    PeriodicSystem,
};

const NONEQ_WATER: &str = "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n";
const NEAR_TD_METHANE: &str = "5\nnear-Td methane\nC 0.0 0.0 0.0\nH 0.640000 0.640000 0.645000\n\
     H -0.640000 -0.645000 0.640000\nH -0.645000 0.640000 -0.640000\nH 0.640000 -0.640000 -0.641000\n";

type Mat3x3 = [[f64; 3]; 3];

fn params() -> Gfn1Parameters {
    Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
}

fn system(xyz: &str) -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture parse")
}

fn magnetic_options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

fn rotation_matrix(axis: [f64; 3], angle: f64) -> Mat3x3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (c, s) = (angle.cos(), angle.sin());
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

fn rotate_system(sys: &PeriodicSystem, r: &Mat3x3) -> PeriodicSystem {
    let atoms = sys
        .atoms
        .iter()
        .map(|a| {
            let p = [a.position.x, a.position.y, a.position.z];
            let mut out = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i] += r[i][j] * p[j];
                }
            }
            Atom {
                z: a.z,
                position: Vec3::new(out[0], out[1], out[2]),
            }
        })
        .collect();
    PeriodicSystem::new(atoms, None).with_charge(sys.charge)
}

fn translate_system(sys: &PeriodicSystem, shift: [f64; 3]) -> PeriodicSystem {
    let atoms = sys
        .atoms
        .iter()
        .map(|a| Atom {
            z: a.z,
            position: Vec3::new(
                a.position.x + shift[0],
                a.position.y + shift[1],
                a.position.z + shift[2],
            ),
        })
        .collect();
    PeriodicSystem::new(atoms, None).with_charge(sys.charge)
}

/// Worst absolute difference between two tensors, and the scale of the first.
fn worst(a: &Mat3x3, b: &Mat3x3) -> (f64, f64) {
    let mut d = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            d = d.max((a[i][j] - b[i][j]).abs());
            scale = scale.max(a[i][j].abs());
        }
    }
    (d, scale)
}

fn rotate_tensor(r: &Mat3x3, x: &Mat3x3) -> Mat3x3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for ip in 0..3 {
                for jp in 0..3 {
                    acc += r[i][ip] * r[j][jp] * x[ip][jp];
                }
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Recentre a molecule on its unweighted centroid (what the fixed
/// `magnetizability_tensor_analytic` does internally).
fn recentered(sys: &PeriodicSystem) -> PeriodicSystem {
    let n = sys.atoms.len() as f64;
    let mut c = [0.0; 3];
    for a in &sys.atoms {
        c[0] += a.position.x;
        c[1] += a.position.y;
        c[2] += a.position.z;
    }
    translate_system(sys, [-c[0] / n, -c[1] / n, -c[2] / n])
}

fn richardson(coarse: &Mat3x3, fine: &Mat3x3) -> Mat3x3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (4.0 * fine[i][j] - coarse[i][j]) / 3.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Calibration ladders (diagnostics; run with `-- --ignored --nocapture`)
// ---------------------------------------------------------------------------

/// **Calibration for the field-step choice.** Prints, for a ladder of steps,
/// the translation-invariance and rotation-covariance residuals of the raw
/// tensor and of four candidate repairs, so the design choice is made on
/// measured numbers rather than on the `O(step²)` argument alone.
#[test]
#[ignore]
fn magnetizability_frame_residual_calibration() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    let r = rotation_matrix([1.0, 2.0, 3.0], 0.7);

    println!("[M0] variants: raw = as called; rc = geometry recentred on the centroid first;");
    println!("[M0]           +R = Richardson (4 chi(h/2) - chi(h))/3.");
    for shift in [[0.0_f64, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, -4.0, 3.0]] {
        let moved = translate_system(&sys, shift);
        let dist = (shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2]).sqrt();
        println!("[M0] shift = {shift:?} (|d| = {dist:.3} bohr)");
        for step in [1.6e-2_f64, 8.0e-3, 4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.0e-4] {
            let call = |s: &PeriodicSystem, h: f64| {
                magnetizability_tensor_analytic(s, &params, &opts, None, h).unwrap()
            };
            let base = call(&sys, step);
            let mv = call(&moved, step);
            let (dt, scale) = worst(&mv, &base);

            let base_r = richardson(&base, &call(&sys, 0.5 * step));
            let mv_r = richardson(&mv, &call(&moved, 0.5 * step));
            let (dt_r, _) = worst(&mv_r, &base_r);

            let rc_base = call(&recentered(&sys), step);
            let rc_mv = call(&recentered(&moved), step);
            let (dt_rc, _) = worst(&rc_mv, &rc_base);

            // Rotation covariance about the coordinate origin.
            let rot = call(&rotate_system(&sys, &r), step);
            let (dr, _) = worst(&rot, &rotate_tensor(&r, &base));
            let rot_r = richardson(
                &rot,
                &call(&rotate_system(&sys, &r), 0.5 * step),
            );
            let (dr_r, _) = worst(&rot_r, &rotate_tensor(&r, &base_r));

            println!(
                "[M0]   h = {step:.2e}: trans raw {:.3e}  rc {:.3e}  +R {:.3e} | \
                 rot raw {:.3e}  +R {:.3e}   (chi scale {scale:.4e})",
                dt / scale,
                dt_rc / scale,
                dt_r / scale,
                dr / scale,
                dr_r / scale
            );
        }
    }
}

/// **Roundoff floor of the cross finite difference.** The mixed second field
/// derivative divides by `step²`, so below some step the `~1e-16` rounding of
/// the LAO builder dominates and the tensor stops improving. Prints the drift
/// of `chi` against the smallest step, which is what sets the lower bound on
/// any "just use a smaller step" repair.
#[test]
#[ignore]
fn magnetizability_step_roundoff_floor() {
    let params = params();
    let opts = magnetic_options();
    let sys = recentered(&system(NONEQ_WATER));
    let reference = magnetizability_tensor_analytic(&sys, &params, &opts, None, 4.0e-3).unwrap();
    let fine = magnetizability_tensor_analytic(&sys, &params, &opts, None, 2.0e-3).unwrap();
    let exact = richardson(&reference, &fine);
    println!("[M1] |chi(h) - chi_Richardson(4e-3)| / scale, water recentred:");
    for step in [
        1.6e-2_f64, 4.0e-3, 1.0e-3, 2.5e-4, 1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6,
    ] {
        let x = magnetizability_tensor_analytic(&sys, &params, &opts, None, step).unwrap();
        let (d, scale) = worst(&x, &exact);
        println!("[M1]   h = {step:.2e}: |dchi| = {d:.4e} (rel {:.3e})", d / scale);
    }
}

// ---------------------------------------------------------------------------
// Regression gates
// ---------------------------------------------------------------------------

/// **The magnetizability may not depend on where the molecule sits.** London
/// orbitals make `xi` gauge-origin independent as an identity, so a rigid
/// translation is a pure gauge transformation of `H0(B)` / `S(B)`.
///
/// The finite differences that supply the LAO integral derivatives are *not*
/// invariant under that transformation, though, because they run along fixed
/// global field axes with an effective parameter that grows with the molecule's
/// distance from the coordinate origin. Before the repair the raw tensor moved
/// by rel `6.3e-6` under a 2-bohr shift and rel `1.3e-3` under a 9.4-bohr one,
/// at the commonly used `step = 4e-3`. `magnetizability_tensor_analytic` now
/// differentiates in the molecule's own centroid frame, so the residual is
/// SCC-convergence noise.
#[test]
fn magnetizability_is_translation_invariant() {
    let params = params();
    let opts = magnetic_options();
    for (name, xyz) in [("water", NONEQ_WATER), ("near-Td methane", NEAR_TD_METHANE)] {
        let sys = system(xyz);
        let base = magnetizability_tensor_analytic(&sys, &params, &opts, None, 4.0e-3).unwrap();
        for shift in [[2.0, 0.0, 0.0], [0.0, -3.5, 1.25], [8.0, -4.0, 3.0]] {
            let moved = translate_system(&sys, shift);
            let x = magnetizability_tensor_analytic(&moved, &params, &opts, None, 4.0e-3).unwrap();
            let (d, scale) = worst(&x, &base);
            println!(
                "[M3] {name} shifted by {shift:?} bohr: |dchi|_max = {d:.3e} \
                 (chi scale {scale:.4e}, rel {:.3e})",
                d / scale
            );
            assert!(
                d / scale < 1.0e-9,
                "{name}: magnetizability moved by rel {:.3e} under a rigid translation by \
                 {shift:?} — the field-derivative frame dependence is back",
                d / scale
            );
        }
    }
}

/// **`xi(R r) = R xi(r) R^T`.** Rotating the molecule while the field axes stay
/// put changes which truncation error the cross finite difference samples, so
/// this is the gate the Richardson extrapolation exists for: the bare central
/// difference broke covariance by rel `2.7e-6` at `step = 4e-3`, and shrinking
/// the step could not do better than rel `~1e-8` (it bottoms out at
/// `step ~ 2.5e-4` and rises again on rounding — see
/// `magnetizability_step_roundoff_floor`).
#[test]
fn magnetizability_is_rotationally_covariant() {
    let params = params();
    let opts = magnetic_options();
    for (name, xyz) in [("water", NONEQ_WATER), ("near-Td methane", NEAR_TD_METHANE)] {
        let sys = system(xyz);
        let base = magnetizability_tensor_analytic(&sys, &params, &opts, None, 4.0e-3).unwrap();
        for (rname, axis, angle) in [
            ("(1,2,3) 0.7 rad", [1.0, 2.0, 3.0], 0.7),
            ("(0,0,1) 90 deg", [0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2),
            ("(-2,1,4) 2.4 rad", [-2.0, 1.0, 4.0], 2.4),
        ] {
            let r = rotation_matrix(axis, angle);
            let x =
                magnetizability_tensor_analytic(&rotate_system(&sys, &r), &params, &opts, None, 4.0e-3)
                    .unwrap();
            let (d, scale) = worst(&x, &rotate_tensor(&r, &base));
            println!(
                "[M4] {name} rotated {rname}: |chi(Rr) - R chi R^T|_max = {d:.3e} \
                 (chi scale {scale:.4e}, rel {:.3e})",
                d / scale
            );
            assert!(
                d / scale < 1.0e-9,
                "{name}: rotational covariance broken by rel {:.3e} under {rname}",
                d / scale
            );
        }
    }
}

/// The Richardson pair must actually deliver `O(step^4)`: halving the (coarse)
/// step may not move the tensor by more than the frame gates tolerate. A bare
/// central difference moves by a factor `~1.4e-5` between these two steps.
#[test]
fn magnetizability_is_insensitive_to_the_field_step() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    let coarse = magnetizability_tensor_analytic(&sys, &params, &opts, None, 8.0e-3).unwrap();
    let fine = magnetizability_tensor_analytic(&sys, &params, &opts, None, 4.0e-3).unwrap();
    let (d, scale) = worst(&coarse, &fine);
    println!(
        "[M5] water: |chi(8e-3) - chi(4e-3)|_max = {d:.3e} (chi scale {scale:.4e}, rel {:.3e})",
        d / scale
    );
    assert!(
        d / scale < 1.0e-8,
        "magnetizability still carries a visible O(step^2) term: rel {:.3e} between \
         step = 8e-3 and 4e-3",
        d / scale
    );
}

/// Wall-clock cost of the tensor, to price the doubled builder evaluations of
/// an internal Richardson extrapolation.
#[test]
#[ignore]
fn magnetizability_cost_probe() {
    let params = params();
    let opts = magnetic_options();
    for (name, xyz) in [("water", NONEQ_WATER), ("methane", NEAR_TD_METHANE)] {
        let sys = system(xyz);
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            magnetizability_tensor_analytic(&sys, &params, &opts, None, 4.0e-3).unwrap();
        }
        println!("[M2] {name}: {:.1} ms / call", t0.elapsed().as_secs_f64() * 1000.0 / 3.0);
    }
}
