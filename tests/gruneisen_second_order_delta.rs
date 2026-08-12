// SPDX-License-Identifier: GPL-3.0-or-later
//! Convergence of the second-order Gruneisen parameter in the volumetric step.

use gfn1_rs::pbc::gruneisen::DEFAULT_GRUNEISEN_DELTA_SECOND;
use gfn1_rs::pbc::{
    pbc_gruneisen, EwaldOptions, GruneisenOptions, PbcOptions, SecondOrderStencil,
};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};

const DIAMOND_PRIMITIVE: &str = "2\n\
Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.891750 0.891750 0.891750\n";

fn params() -> Gfn1Parameters {
    Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
}

fn pbc_electronic() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

fn lean_pbc() -> PbcOptions {
    PbcOptions {
        ao_cutoff: 12.0,
        ewald: EwaldOptions {
            real_cutoff: 18.0,
            sr_cutoff: 8.0,
            ..EwaldOptions::default()
        },
        ..PbcOptions::default()
    }
}

/// **Calibration ladder for the second-order volumetric step.** `gamma2` is a
/// second difference of `ln lambda(ln V)`, so phonon noise enters as
/// `eps / delta^2`: at a small `delta` the two stencils disagree by more than
/// the value itself. Prints both stencils, their relative gap, and the
/// first-order `gamma` from the same run so the cost of raising `delta` to the
/// first order can be read off directly.
#[test]
#[ignore]
fn gruneisen_second_order_delta_calibration() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    println!("[G0] diamond gamma2(300 K) vs second-order volumetric step (lean cutoffs):");
    for delta in [
        5.0e-3_f64, 1.0e-2, 2.0e-2, 3.0e-2, 4.0e-2, 6.0e-2, 8.0e-2, 1.0e-1, 1.4e-1,
    ] {
        let base = GruneisenOptions {
            delta: 5.0e-3,
            delta_second: Some(delta),
            temperatures: vec![300.0],
            electronic: pbc_electronic(),
            pbc: lean_pbc(),
            second_order: true,
            ..GruneisenOptions::default()
        };
        let three = pbc_gruneisen(
            &sys,
            &params,
            &GruneisenOptions {
                second_order_stencil: SecondOrderStencil::ThreePoint,
                ..base.clone()
            },
        )
        .unwrap();
        let five = pbc_gruneisen(
            &sys,
            &params,
            &GruneisenOptions {
                second_order_stencil: SecondOrderStencil::FivePoint,
                ..base.clone()
            },
        )
        .unwrap();
        let g3 = three.gamma2_at(300.0).unwrap_or(f64::NAN);
        let g5 = five.gamma2_at(300.0).unwrap_or(f64::NAN);
        let rel = (g3 - g5).abs() / g5.abs().max(1.0e-30);
        println!(
            "[G0]   delta = {delta:.2e}: 3pt gamma2 = {g3:+.6}, 5pt gamma2 = {g5:+.6}, \
             |d| = {:.3e} (rel {rel:.3e}) | gamma 3pt {:.9} 5pt {:.9} | refit-gamma gap {:.3e}",
            (g3 - g5).abs(),
            three.gamma_at(300.0).unwrap_or(f64::NAN),
            five.gamma_at(300.0).unwrap_or(f64::NAN),
            three
                .mode_gamma
                .iter()
                .zip(&three.mode_gamma_refit)
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
        );
    }
}

// ---------------------------------------------------------------------------
// Regression gates
// ---------------------------------------------------------------------------

/// **The two second-order stencils must agree on `gamma2` at the default step.**
///
/// This is the gate the whole repair exists for. Before it, at the shared
/// `delta = 5e-3`, the three-point stencil returned `-0.0372` and the five-point
/// one `+0.0674` — opposite signs, with a gap 1.5x the value itself — so a
/// `gamma2` quoted from a default call was not trustworthy to its leading digit.
///
/// Two independent defects fed that: the real-space cutoffs were held at a fixed
/// radius while the cell breathed (so the integer image lists stepped, putting
/// `~5e-7` jumps into `ln lambda`), and the second difference inherited the
/// first-order step even though it amplifies noise by `delta^-2` instead of
/// `delta^-1`. The cutoffs now travel with the strain and the second-order node
/// set has its own [`DEFAULT_GRUNEISEN_DELTA_SECOND`] = `2e-2`.
#[test]
fn gruneisen_second_order_stencils_agree_at_the_default_step() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let base = GruneisenOptions {
        delta: 5.0e-3,
        temperatures: vec![300.0],
        electronic: pbc_electronic(),
        pbc: lean_pbc(),
        second_order: true,
        ..GruneisenOptions::default()
    };
    assert_eq!(
        base.delta_second,
        Some(DEFAULT_GRUNEISEN_DELTA_SECOND),
        "the default second-order step must not silently fall back to `delta`"
    );
    let three = pbc_gruneisen(
        &sys,
        &params,
        &GruneisenOptions {
            second_order_stencil: SecondOrderStencil::ThreePoint,
            ..base.clone()
        },
    )
    .unwrap();
    let five = pbc_gruneisen(
        &sys,
        &params,
        &GruneisenOptions {
            second_order_stencil: SecondOrderStencil::FivePoint,
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(three.delta_second, Some(DEFAULT_GRUNEISEN_DELTA_SECOND));

    let g3 = three.gamma2_at(300.0).unwrap();
    let g5 = five.gamma2_at(300.0).unwrap();
    let rel = (g3 - g5).abs() / g5.abs();
    println!(
        "[G2] diamond gamma2(300 K): 3-point {g3:+.6}, 5-point {g5:+.6}, rel gap {rel:.3e}"
    );
    assert!(
        g3 < 0.0 && g5 < 0.0,
        "gamma2 must not flip sign between stencils: 3-point {g3:+.6}, 5-point {g5:+.6}"
    );
    assert!(
        rel < 1.0e-2,
        "three- and five-point gamma2 disagree by rel {rel:.3e} at the default second-order \
         step: 3-point {g3:+.6}, 5-point {g5:+.6}"
    );

    // Every optical mode, not just the thermodynamic average.
    let mut worst = 0.0_f64;
    for (a, b) in three.mode_gamma2.iter().zip(&five.mode_gamma2) {
        if a.is_finite() && b.is_finite() {
            worst = worst.max((a - b).abs() / b.abs().max(1.0e-30));
        }
    }
    println!("[G2] worst per-mode relative stencil gap: {worst:.3e}");
    assert!(worst < 1.0e-2, "per-mode gamma2 stencil gap rel {worst:.3e}");
}

/// **`gamma2` must be stable against its own step.** A quantity that moves when
/// the step moves is not a derivative. Gated across a factor of 4 in
/// `delta_second`, from the default upwards — the recommended window. (Below the
/// default the residual `eps / delta^2` amplification is still visible: the
/// calibration ladder puts `delta_second = 1e-2` at `-0.03613` against `-0.03701`
/// at the default, a 2.4% offset that is noise, not curvature.)
///
/// Also pins the *separation* of the two orders: changing `delta_second` may not
/// move the first-order `gamma` at all (it is bit-for-bit the same first-order
/// central difference), which is the property that lets the second order take a
/// coarser step without paying for it at first order.
#[test]
fn gruneisen_second_order_is_stable_across_the_step_ladder() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let mut values = Vec::new();
    let mut gammas = Vec::new();
    for delta_second in [2.0e-2_f64, 4.0e-2, 8.0e-2] {
        let g = pbc_gruneisen(
            &sys,
            &params,
            &GruneisenOptions {
                delta: 5.0e-3,
                delta_second: Some(delta_second),
                temperatures: vec![300.0],
                electronic: pbc_electronic(),
                pbc: lean_pbc(),
                second_order: true,
                second_order_stencil: SecondOrderStencil::FivePoint,
                ..GruneisenOptions::default()
            },
        )
        .unwrap();
        values.push(g.gamma2_at(300.0).unwrap());
        gammas.push(g.gamma_at(300.0).unwrap());
    }
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = (hi - lo) / hi.abs().max(lo.abs());
    println!("[G3] diamond gamma2(300 K) over delta_second = 2e-2 .. 8e-2: {values:?}");
    println!("[G3] relative spread {spread:.3e}; first-order gamma {gammas:?}");
    assert!(
        spread < 1.0e-2,
        "gamma2 moves by rel {spread:.3e} across a factor of 8 in delta_second: {values:?}"
    );
    for g in &gammas {
        assert!(
            (g - gammas[0]).abs() < 1.0e-12,
            "the second-order step leaked into the first-order gamma: {gammas:?}"
        );
    }
}

/// The second-order step must not perturb a first-order-only run: asking for
/// `second_order` may add numbers, never change the ones already reported.
#[test]
fn gruneisen_second_order_leaves_the_first_order_untouched() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let base = GruneisenOptions {
        delta: 5.0e-3,
        temperatures: vec![300.0],
        electronic: pbc_electronic(),
        pbc: lean_pbc(),
        ..GruneisenOptions::default()
    };
    let plain = pbc_gruneisen(&sys, &params, &base).unwrap();
    let second = pbc_gruneisen(
        &sys,
        &params,
        &GruneisenOptions {
            second_order: true,
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(plain.delta_second, None);
    for (a, b) in plain.mode_gamma.iter().zip(&second.mode_gamma) {
        assert!(
            (a - b).abs() < 1.0e-14 || (a.is_nan() && b.is_nan()),
            "second_order changed the first-order mode gamma: {a} vs {b}"
        );
    }
    println!(
        "[G4] first-order gamma_th(300 K): plain {:.12}, with second order {:.12}",
        plain.gamma_at(300.0).unwrap(),
        second.gamma_at(300.0).unwrap()
    );
}

/// The same ladder at the production (default) cutoffs, to confirm the step
/// recommendation is not an artefact of the lean test cutoffs.
#[test]
#[ignore]
fn gruneisen_second_order_delta_calibration_default_cutoffs() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    println!("[G1] diamond gamma2(300 K) vs second-order volumetric step (default cutoffs):");
    for delta in [5.0e-3_f64, 2.0e-2, 6.0e-2, 1.0e-1] {
        let base = GruneisenOptions {
            delta: 5.0e-3,
            delta_second: Some(delta),
            temperatures: vec![300.0],
            electronic: pbc_electronic(),
            second_order: true,
            ..GruneisenOptions::default()
        };
        let three = pbc_gruneisen(
            &sys,
            &params,
            &GruneisenOptions {
                second_order_stencil: SecondOrderStencil::ThreePoint,
                ..base.clone()
            },
        )
        .unwrap();
        let five = pbc_gruneisen(
            &sys,
            &params,
            &GruneisenOptions {
                second_order_stencil: SecondOrderStencil::FivePoint,
                ..base.clone()
            },
        )
        .unwrap();
        let g3 = three.gamma2_at(300.0).unwrap_or(f64::NAN);
        let g5 = five.gamma2_at(300.0).unwrap_or(f64::NAN);
        println!(
            "[G1]   delta = {delta:.2e}: 3pt gamma2 = {g3:+.6}, 5pt gamma2 = {g5:+.6}, \
             rel gap {:.3e} | gamma 3pt {:.9}",
            (g3 - g5).abs() / g5.abs().max(1.0e-30),
            three.gamma_at(300.0).unwrap_or(f64::NAN)
        );
    }
}
