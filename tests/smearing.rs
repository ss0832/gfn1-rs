// SPDX-License-Identifier: GPL-3.0-or-later
//! **Fermi-smearing regression gates for the analytic derivative ladder.**
//!
//! Every gate here runs on a FRACTIONALLY occupied reference, where the T = 0
//! orbital algebra (a clean occupied/virtual split, integer occupations) does
//! not apply:
//!
//! * order 2 — [`analytic_hessian`] (internally a finite-temperature
//!   charge-space response) vs central FD of the analytic gradient, on three
//!   representative DOF columns of Fermi-smeared Ni(CO)₄;
//! * order 3 — [`directional_third_finite_t`] vs central FD along `v` of the
//!   smeared analytic Hessian contracted `vv`;
//! * order 4 — [`directional_fourth_seminumerical`], which finite-differences
//!   that occupation-agnostic third derivative and therefore RUNS on smeared
//!   systems (the analytic quartic still does not); gated for step consistency;
//! * the two guards that must stay closed: the ANALYTIC quartic rejects
//!   fractional occupations, and an EXACTLY degenerate fractionally occupied
//!   reference is rejected by the second-order charge-space solver.
//!
//! **Fixtures and cost.** Fermi-smeared Ni(CO)₄ (3000 K, 18 of 41 orbitals
//! fractional) is the reference smeared system and carries the order-2 gate and
//! both guards, all of which are cheap. Orders 3 and 4 use non-equilibrium
//! formaldehyde at 10000 K (10 of 12 orbitals fractional, including a
//! 1.79/0.23 pair) because the analytic directional FC3 costs ~855 s on
//! Ni(CO)₄'s 27 DOF versus ~22 s on formaldehyde's 12 — the full Ni(CO)₄ FC3
//! `h`-ladder is the `#[ignore]`d gate in `src/third_derivative/finite_t.rs`,
//! and duplicating it here would cost a quarter of an hour per run.
//!
//! **Option-set pairing (easy to get wrong).** The analytic third derivative's
//! frozen bundle ALWAYS carries repulsion and halogen — it has no
//! `include_repulsion` switch — so its FD reference must be the FULL analytic
//! Hessian with only dispersion gated off. Pairing it with an electronic-only
//! Hessian leaves a temperature-independent constant residual (the repulsion
//! third derivative) that looks exactly like a missing smearing term. The
//! order-2 gate is the opposite case: it pairs the ELECTRONIC-only Hessian with
//! the gradient's `electronic_gradient`, as `tests/hessian.rs` does.

use gfn1_rs::fourth_derivative::{
    directional_fourth_derivative, directional_fourth_seminumerical,
    fourth_derivative_analytic_dense,
};
use gfn1_rs::hessian::AnalyticHessianOptions;
use gfn1_rs::third_derivative::finite_t::{
    directional_fourth_finite_t, directional_third_finite_t,
};
use gfn1_rs::{
    analytic_gradient, analytic_hessian, run_electronic, AnalyticGradientOptions,
    ElectronicOptions, Gfn1Parameters, PeriodicSystem,
};

/// Fermi-smeared Ni(CO)₄ with the Td symmetry broken: 18 of 41 orbitals
/// fractionally occupied (1.94 … 0.0003) and accidental near-degenerate pairs
/// (gaps 1.9e-8 … 3.6e-7) — fractional occupations WITHOUT exact degeneracy.
const DISTORTED_NI_CO4: &str = "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n";

/// The cheap strongly smeared fixture for orders 3 and 4: non-equilibrium
/// formaldehyde at 10000 K puts 10 of 12 orbitals at fractional occupation
/// (1.99999 … 1.79 … 0.23 … 0.0002) with the smallest level spacing at 1.1e-2,
/// i.e. heavy smearing and no degeneracy.
const NONEQ_HCHO: &str =
    "4\nnon-eq formaldehyde\nC 0.0 0.0 0.0\nO 1.28 0.10 0.05\nH -0.60 0.95 0.10\nH -0.62 -0.90 0.12\n";

/// The rejection fixture: a Ni–O diatomic at 3000 K. Cylindrical symmetry makes
/// its π manifold EXACTLY degenerate (gap 5.6e-16, i.e. symmetry-exact rather
/// than accidental) and the partly filled Ni d/π levels are fractionally
/// occupied (9 of 13 orbitals) — the combination the second-order solver must
/// refuse. Same physics as Td-symmetric Ni(CO)₄, two orders of magnitude
/// cheaper (see the doc comment on the guard test).
const NI_O_DIATOMIC: &str = "2\nNiO\nNi 0.0 0.0 0.0\nO 1.65 0.0 0.0\n";

const NI_CO4_TEMPERATURE: f64 = 3000.0;
const HCHO_TEMPERATURE: f64 = 10000.0;

/// Tight SCF at the requested electronic temperature: the FD references below
/// need the electronic state converged far past the FD truncation level.
fn smeared_electronic_options(etemp: f64) -> ElectronicOptions {
    ElectronicOptions {
        enable_dispersion: false,
        electronic_temperature: etemp,
        energy_tolerance: 1.0e-14,
        charge_tolerance: 1.0e-12,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

/// The FULL model minus dispersion — the option set the analytic third and
/// fourth derivatives assemble (repulsion and halogen included), and the one
/// the `#[ignore]`d Ni(CO)₄ FC3 ladder uses.
fn full_options(etemp: f64) -> AnalyticHessianOptions {
    AnalyticHessianOptions {
        electronic_options: smeared_electronic_options(etemp),
        ..AnalyticHessianOptions::default()
    }
}

/// The ELECTRONIC subset of the analytic Hessian (fixed blocks + relaxed
/// response), which matches the gradient result's `electronic_gradient` term by
/// term — the pairing the T = 0 Hessian gates in `tests/hessian.rs` use.
fn electronic_only_hessian_options(etemp: f64) -> AnalyticHessianOptions {
    AnalyticHessianOptions {
        include_repulsion: false,
        include_fixed_scc: true,
        include_fixed_pulay: true,
        include_fixed_cn_h0: true,
        include_electronic: true,
        include_dispersion: false,
        include_halogen: false,
        electronic_options: smeared_electronic_options(etemp),
    }
}

fn electronic_only_gradient_options(etemp: f64) -> AnalyticGradientOptions {
    AnalyticGradientOptions {
        electronic: smeared_electronic_options(etemp),
        include_repulsion: false,
        include_dispersion: false,
        include_hamiltonian: true,
        include_scc: true,
        include_halogen: false,
    }
}

/// The skew probe direction shared with the third/fourth derivative gates.
fn skew_direction(ndof: usize) -> Vec<f64> {
    (0..ndof)
        .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
        .collect()
}

fn displaced_dof(system: &PeriodicSystem, dof: usize, step: f64) -> PeriodicSystem {
    let mut out = system.clone();
    let atom = &mut out.atoms[dof / 3];
    match dof % 3 {
        0 => atom.position.x += step,
        1 => atom.position.y += step,
        _ => atom.position.z += step,
    }
    out
}

fn displaced_along(system: &PeriodicSystem, v: &[f64], step: f64) -> PeriodicSystem {
    let mut out = system.clone();
    for (atom_idx, atom) in out.atoms.iter_mut().enumerate() {
        atom.position.x += step * v[3 * atom_idx];
        atom.position.y += step * v[3 * atom_idx + 1];
        atom.position.z += step * v[3 * atom_idx + 2];
    }
    out
}

fn component(values: &[gfn1_rs::math::Vec3], dof: usize) -> f64 {
    let atom = dof / 3;
    match dof % 3 {
        0 => values[atom].x,
        1 => values[atom].y,
        _ => values[atom].z,
    }
}

/// Guard on the fixtures themselves: assert the reference really IS fractionally
/// occupied. Without this a future parameter or default change could silently
/// turn every gate below into a T = 0 gate that still passes.
fn assert_fractional(system: &PeriodicSystem, params: &Gfn1Parameters, etemp: f64, label: &str) {
    let electronic = run_electronic(system, params, smeared_electronic_options(etemp)).unwrap();
    let fractional: Vec<f64> = electronic
        .occupations
        .iter()
        .copied()
        .filter(|&f| f > 1.0e-8 && (f - 2.0).abs() > 1.0e-8)
        .collect();
    let widest = fractional
        .iter()
        .map(|f| (f - 2.0).abs().min(*f))
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{label} @ {etemp} K: {} of {} orbitals fractional, widest deviation from integer {widest:.3e}",
        fractional.len(),
        electronic.occupations.len()
    );
    assert!(
        fractional.len() >= 2 && widest > 1.0e-4,
        "{label}: the fixture is not meaningfully Fermi-smeared ({} fractional, widest \
         deviation {widest:.3e}) — the smearing gates would be vacuous",
        fractional.len()
    );
}

/// **Order 2.** The smeared analytic Hessian must differentiate the smeared
/// analytic gradient. Three columns (Ni `x`, a carbonyl `y`, the last oxygen
/// `z`) instead of all 27 keep the FD count small; each is compared over ALL 27
/// rows, so the fractional-occupation response channel is probed in every
/// direction.
///
/// Honest first run: max deviation 6.99e-10 at `h = 1e-4` (3.72e-10 at `h =
/// 1e-5`) against column entries of order 1e-1 — pure FD truncation. The
/// assertion sits at 1e-7, two orders above that and an order tighter than the
/// 1e-6 the T = 0 Hessian gates in `tests/hessian.rs` use.
#[test]
fn smeared_analytic_hessian_matches_gradient_fd_columns() {
    let started = std::time::Instant::now();
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(DISTORTED_NI_CO4, 0.0, false).unwrap();
    assert_fractional(&system, &params, NI_CO4_TEMPERATURE, "distorted Ni(CO)4");

    let analytic = analytic_hessian(
        &system,
        &params,
        electronic_only_hessian_options(NI_CO4_TEMPERATURE),
    )
    .unwrap();
    assert!(
        analytic
            .cpxtb_response
            .as_ref()
            .is_some_and(|r| r.converged),
        "finite-temperature CPXTB response did not converge"
    );

    let grad_options = electronic_only_gradient_options(NI_CO4_TEMPERATURE);
    let step = 1.0e-4;
    let ndof = 3 * system.atoms.len();
    let columns = [0usize, 13, 26];
    let mut max_delta = 0.0_f64;
    let mut max_entry = (0usize, 0usize, 0.0_f64, 0.0_f64);
    for &col in &columns {
        let gp = analytic_gradient(
            &displaced_dof(&system, col, step),
            &params,
            grad_options.clone(),
        )
        .unwrap()
        .electronic_gradient;
        let gm = analytic_gradient(
            &displaced_dof(&system, col, -step),
            &params,
            grad_options.clone(),
        )
        .unwrap()
        .electronic_gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            let delta = (analytic.hessian[(row, col)] - fd).abs();
            if delta > max_delta {
                max_delta = delta;
                max_entry = (row, col, analytic.hessian[(row, col)], fd);
            }
        }
    }
    eprintln!(
        "smeared Hessian vs gradient FD (h {step:.0e}, columns {columns:?}): max delta \
         {max_delta:.3e} at (row, col, analytic, fd) {max_entry:?} [{:?}]",
        started.elapsed()
    );
    assert!(
        max_delta < 1.0e-7,
        "smeared analytic Hessian vs gradient FD: max delta {max_delta:.3e} at \
         (row, col, analytic, fd) {max_entry:?}"
    );
}

/// **Order 3.** The occupation-agnostic directional third derivative vs the
/// central FD along `v` of the smeared analytic Hessian contracted `vv` — the
/// always-on companion of the `h`-ladder gate that lives (as `#[ignore]`,
/// because its Ni(CO)₄ FC3 alone costs ~855 s) in
/// `src/third_derivative/finite_t.rs`.
///
/// Honest first run on this fixture: `delta(h) = 4.45e-10`, `delta(h/2) =
/// 1.11e-10`, ratio 4.00 on a value of 3.15e-2 — textbook `h²` truncation, no
/// missing-term floor. That matches the validated Ni(CO)₄ ladder (`delta(h)
/// 6.25e-10` on values ~1.4e-2, ratio 3.98), so the shared `1e-7·(1 + |fd|)`
/// bound is ~200× above truncation and far below any missing-term signal. Only
/// the coarse step is taken here — the `h²` scaling is the ignored ladder's job,
/// this gate's job is to catch a term that disappears at finite temperature.
#[test]
fn smeared_directional_third_matches_hessian_fd() {
    let started = std::time::Instant::now();
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(NONEQ_HCHO, 0.0, false).unwrap();
    assert_fractional(&system, &params, HCHO_TEMPERATURE, "non-eq HCHO");
    let options = full_options(HCHO_TEMPERATURE);
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    let v = skew_direction(ndof);

    let analytic = directional_third_finite_t(&system, &params, &options, cutoff, &v).unwrap();
    let hessian_vv = |sys: &PeriodicSystem| -> f64 {
        let h = analytic_hessian(sys, &params, options.clone()).unwrap().hessian;
        let mut acc = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                acc += v[a] * v[b] * h[(a, b)];
            }
        }
        acc
    };
    let h = 1.0e-3;
    let fd = (hessian_vv(&displaced_along(&system, &v, h))
        - hessian_vv(&displaced_along(&system, &v, -h)))
        / (2.0 * h);
    let delta = (analytic - fd).abs();
    eprintln!(
        "smeared directional FC3: analytic {analytic:.12e} fd(h={h:.0e}) {fd:.12e} delta \
         {delta:.3e} [{:?}]",
        started.elapsed()
    );
    assert!(
        delta < 1.0e-7 * (1.0 + fd.abs()),
        "finite-temperature directional FC3 vs Hessian FD: analytic {analytic:.12e} fd \
         {fd:.12e} delta {delta:.3e}"
    );
}

/// **Order 4.** The seminumerical directional quartic finite-differences the
/// finite-temperature third derivative, so it must RUN on a fractionally
/// occupied reference and return a finite value. Before the reference was
/// re-routed it inherited `third_derivative_analytic_vector`'s
/// integer-occupation rejection, and there was no FC4 route at all for smeared
/// systems (the analytic quartic rejects them too — see the guard below).
///
/// It is a second-order-accurate central FD of an ANALYTIC quantity, so the two
/// steps must agree to `O(h²)`: the `h` → `h/2` change IS the truncation error.
/// Honest first run: `fd(h) = 1.3180092955e-2`, `fd(h/2) = 1.3180093448e-2`,
/// relative change 4.87e-10 — five orders below the `1e-5` bound, which is
/// itself far tighter than the "few percent" a smoothness check needs. A
/// blow-up or a `NaN` would mean the smeared third derivative is not smooth
/// along `v`.
#[test]
fn smeared_directional_fourth_seminumerical_is_step_consistent() {
    let started = std::time::Instant::now();
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(NONEQ_HCHO, 0.0, false).unwrap();
    let options = full_options(HCHO_TEMPERATURE);
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let v = skew_direction(3 * system.atoms.len());

    let h = 1.0e-3;
    let coarse =
        directional_fourth_seminumerical(&system, &params, &options, cutoff, &v, h).unwrap();
    let fine =
        directional_fourth_seminumerical(&system, &params, &options, cutoff, &v, 0.5 * h).unwrap();
    let rel = (coarse - fine).abs() / (1.0 + fine.abs());
    eprintln!(
        "smeared seminumerical FC4: fd(h={h:.0e}) {coarse:.10e} fd(h/2) {fine:.10e} rel \
         {rel:.3e} richardson {:.10e} [{:?}]",
        (4.0 * fine - coarse) / 3.0,
        started.elapsed()
    );
    assert!(
        coarse.is_finite() && fine.is_finite(),
        "smeared seminumerical quartic returned a non-finite value: {coarse} / {fine}"
    );
    assert!(
        rel < 1.0e-5,
        "smeared seminumerical quartic is not step-consistent: fd(h) {coarse:.10e} vs \
         fd(h/2) {fine:.10e} (rel {rel:.3e}) — the FD'd third derivative is not smooth"
    );
}

/// **Guard 1.** The ANALYTIC quartic (directional and dense) must keep rejecting
/// fractional occupations with the documented message: silently returning the
/// integer-occupation expression for a smeared reference would be a wrong
/// answer, not a slow one. The message names
/// [`directional_fourth_seminumerical`] as the fallback, which — since that
/// route now finite-differences the finite-temperature third derivative — is
/// actually available for these systems (the test above runs it).
#[test]
fn smeared_analytic_fourth_rejects_fractional_occupations() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(DISTORTED_NI_CO4, 0.0, false).unwrap();
    let options = full_options(NI_CO4_TEMPERATURE);
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let v = skew_direction(3 * system.atoms.len());

    let err = directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("fractional") && msg.contains("Fermi-smeared"),
        "expected the fractional-occupation rejection from the directional quartic, got: {msg}"
    );
    let err = fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("fractional") && msg.contains("Fermi-smeared"),
        "expected the fractional-occupation rejection from the dense quartic, got: {msg}"
    );
}

/// **Guard 2, end to end — the order boundary at exact degeneracy.**
///
/// An EXACTLY degenerate fractionally occupied reference used to be rejected
/// by the second-order charge-space solver, because the in-block occupation
/// channel is matrix-valued (Daleckii–Krein) and the scalar `f''` chain could
/// not represent it. The solver now assembles the second order in the
/// frame-free **resolvent** form, where the degenerate case is the confluent
/// limit of the same divided differences, so the guard was retired.
///
/// That has an order-dependent consequence this test pins:
///
/// * [`directional_third_finite_t`] consumes the second-order response only,
///   so it **inherited the fix and now succeeds**. Verified against the
///   central difference of the smeared analytic Hessian on this very fixture
///   (min level spacing `1.1e-16`): `1.24e-9 → 3.10e-10 → 7.77e-11` for
///   `h = 2e-3 → 1e-3 → 5e-4`, a clean `O(h²)` ladder at ratio 4.00.
/// * `solve_third_order_directional` — reached only by the finite-T **FC4**
///   routes — is still frame-based, and a frame is not defined inside such a
///   block (measured 3.5e3 against its FD gate). It still refuses, so
///   [`directional_fourth_finite_t`] must error here.
#[test]
fn exactly_degenerate_smeared_third_succeeds_but_fourth_is_rejected() {
    let started = std::time::Instant::now();
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(NI_O_DIATOMIC, 0.0, false).unwrap();
    let options = full_options(NI_CO4_TEMPERATURE);
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let v = skew_direction(3 * system.atoms.len());

    let electronic =
        run_electronic(&system, &params, options.electronic_options.clone()).unwrap();
    let min_spacing = electronic
        .orbital_energies
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_spacing < 1.0e-10,
        "NiO is no longer exactly degenerate (min level spacing {min_spacing:.3e}) — the \
         order boundary would be tested on the wrong regime"
    );
    assert_fractional(&system, &params, NI_CO4_TEMPERATURE, "NiO");

    let third = directional_third_finite_t(&system, &params, &options, cutoff, &v)
        .expect("the resolvent second order makes the finite-T FC3 valid at exact degeneracy");
    assert!(
        third.is_finite(),
        "exactly degenerate smeared FC3 returned a non-finite value: {third}"
    );

    let err = directional_fourth_finite_t(&system, &params, &options, cutoff, &v).unwrap_err();
    let msg = format!("{err}");
    eprintln!(
        "exactly degenerate smeared FC3 = {third:.10e}; FC4 rejected [{:?}]: {msg}",
        started.elapsed()
    );
    assert!(
        msg.contains("exactly degenerate") && msg.contains("fractional occupation"),
        "expected the exact-degeneracy rejection from the frame-based third-order solver, \
         got: {msg}"
    );
}
