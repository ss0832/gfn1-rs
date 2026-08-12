// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gates for the analytic nuclear fourth derivative (quartic force
//! constants):
//!
//! * the DIRECTIONAL five-stage 2n+1 assembly vs the seminumerical reference
//!   (central FD along `v` of the analytic third derivative), full stock model
//!   (dispersion + halogen + CN Hamiltonian + the third-order onsite Γ), on a
//!   non-equilibrium geometry;
//! * the MIXED-INDEX tensor built from that directional quartic by the
//!   polarization identity — self-consistency (`Q·vvvv` vs the directional
//!   value) and element-wise agreement with the seminumerical reference
//!   (central FD of the analytic third-derivative tensor).
//!
//! These are the compact, always-on versions of the fine-grained in-module
//! stage gates in `src/fourth_derivative/`.

use gfn1_rs::fourth_derivative::{
    directional_fourth_derivative, directional_fourth_seminumerical,
    fourth_derivative_analytic_block, fourth_derivative_analytic_dense, SymmetricFourth,
};
use gfn1_rs::hessian::AnalyticHessianOptions;
use gfn1_rs::third_derivative::{third_derivative_analytic_dense, SymmetricThird};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};

fn tight_options() -> AnalyticHessianOptions {
    AnalyticHessianOptions {
        electronic_options: ElectronicOptions {
            energy_tolerance: 1.0e-12,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        },
        ..AnalyticHessianOptions::default()
    }
}

fn noneq_water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
        0.0,
        false,
    )
    .unwrap()
}

/// A stretched HF placed off-axis so that every Cartesian component of every
/// DOF is active (a diatomic on a coordinate axis would zero out most mixed
/// elements and hide index-composition errors).
fn skew_hf() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "2\nstretched skew HF\nF 0.0 0.0 0.0\nH 0.70 0.54 0.40\n",
        0.0,
        false,
    )
    .unwrap()
}

/// The skew probe direction of the directional gate, truncated to `ndof`.
fn skew_direction(ndof: usize) -> Vec<f64> {
    (0..ndof)
        .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
        .collect()
}

/// Non-equilibrium stretched+bent water, full stock model, skew direction:
/// the analytic directional quartic must match the FD of the analytic cubic
/// with `h²` truncation scaling (a flat residual would mean a missing or
/// double-counted composition term).
#[test]
fn directional_fourth_matches_seminumerical_noneq_water() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = noneq_water();
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let v = skew_direction(3 * system.atoms.len());

    let analytic = directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap();
    let fd_at = |h: f64| -> f64 {
        directional_fourth_seminumerical(&system, &params, &options, cutoff, &v, h).unwrap()
    };
    let h1 = 1.0e-3;
    let fd1 = fd_at(h1);
    let delta1 = (analytic - fd1).abs();
    let fd2 = fd_at(0.5 * h1);
    let delta2 = (analytic - fd2).abs();
    eprintln!(
        "directional quartic: analytic {analytic:.10e} fd(h) {fd1:.10e} fd(h/2) {fd2:.10e} \
         delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
        delta1 / delta2.max(1.0e-300)
    );
    assert!(
        delta1 < 1.0e-6 * (1.0 + fd1.abs()),
        "directional fourth vs seminumerical: analytic {analytic:.10e} fd {fd1:.10e} \
         delta {delta1:.3e}"
    );
    assert!(
        delta2 < 0.4 * delta1,
        "residual does not scale as h² (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e})"
    );
}

/// The registry guard: an option set with a term that has no analytic quartic
/// (multipole electrostatics) must fail fast with the uniform message instead
/// of silently returning derivatives of a different energy expression.
#[test]
fn directional_fourth_rejects_unsupported_terms() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.95 0.0 0.0\nH -0.24 0.92 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let mut options = tight_options();
    options.electronic_options.multipole = true;
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let v = vec![0.1; 9];
    let err = directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap_err();
    assert!(
        format!("{err}").contains("multipole"),
        "expected the multipole registry row to block order 4, got: {err}"
    );
}

/// The SAME registry guard on the mixed-index driver — it must not be possible
/// to sidestep the order-4 check by asking for the dense tensor. (Cheap: the
/// guard fires before the SCF.)
#[test]
fn dense_fourth_rejects_unsupported_terms() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.95 0.0 0.0\nH -0.24 0.92 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let mut options = tight_options();
    options.electronic_options.multipole = true;
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let err = fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap_err();
    assert!(
        format!("{err}").contains("multipole"),
        "expected the multipole registry row to block the dense quartic, got: {err}"
    );
    let err =
        fourth_derivative_analytic_block(&system, &params, &options, cutoff, &[0, 1]).unwrap_err();
    assert!(
        format!("{err}").contains("multipole"),
        "expected the multipole registry row to block the quartic block driver, got: {err}"
    );
    // ... and an out-of-range DOF is rejected too.
    let mut ok_options = tight_options();
    ok_options.electronic_options.multipole = false;
    let err = fourth_derivative_analytic_block(&system, &params, &ok_options, cutoff, &[0, 9])
        .unwrap_err();
    assert!(
        format!("{err}").contains("out of range"),
        "expected an out-of-range DOF rejection, got: {err}"
    );
}

/// `system` displaced by `delta` along nuclear DOF `dof` (positions are in
/// bohr, matching the analytic derivative conventions).
fn displaced(system: &PeriodicSystem, dof: usize, delta: f64) -> PeriodicSystem {
    let mut out = system.clone();
    let atom = &mut out.atoms[dof / 3];
    match dof % 3 {
        0 => atom.position.x += delta,
        1 => atom.position.y += delta,
        _ => atom.position.z += delta,
    }
    out
}

/// The central-FD slab `D^{(c)}_abc' = (T_abc'(R + h e_c) − T_abc'(R − h e_c)) / 2h`
/// of the analytic third-derivative tensor — an estimate of `Q_abc'c` that is
/// symmetric in its three ANALYTIC indices but singles out `c` as the FD index.
fn fd_third_slab(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    cutoff: f64,
    dof: usize,
    h: f64,
) -> SymmetricThird {
    let mut acc = third_derivative_analytic_dense(
        &displaced(system, dof, h),
        params,
        options.clone(),
        cutoff,
    )
    .unwrap();
    let mut minus = third_derivative_analytic_dense(
        &displaced(system, dof, -h),
        params,
        options.clone(),
        cutoff,
    )
    .unwrap();
    minus.scale(-1.0);
    acc.add_from(&minus);
    acc.scale(1.0 / (2.0 * h));
    acc
}

/// The seminumerical quartic over `dofs`, packed exactly like
/// [`fourth_derivative_analytic_block`] (index = position in `dofs`).
///
/// Each FD slab is symmetric in only three of the four indices, so the
/// reference is the honest average over the four choices of WHICH index is
/// differentiated numerically — the rank-4 analogue of the explicit
/// 6-permutation mean used to symmetrize the third-derivative slabs. (Averaging
/// over the four slots is correct for repeated indices too: a repeat simply
/// makes several of the four terms coincide.)
fn seminumerical_quartic_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    cutoff: f64,
    dofs: &[usize],
    h: f64,
) -> SymmetricFourth {
    let slabs: Vec<SymmetricThird> = dofs
        .iter()
        .map(|&dof| fd_third_slab(system, params, options, cutoff, dof, h))
        .collect();
    let m = dofs.len();
    let mut store = SymmetricFourth::zeros(m);
    for di in 0..m {
        for ci in 0..=di {
            for bi in 0..=ci {
                for ai in 0..=bi {
                    let (a, b, c, d) = (dofs[ai], dofs[bi], dofs[ci], dofs[di]);
                    let value = (slabs[ai].get(b, c, d)
                        + slabs[bi].get(a, c, d)
                        + slabs[ci].get(a, b, d)
                        + slabs[di].get(a, b, c))
                        / 4.0;
                    store.add(ai, bi, ci, di, value);
                }
            }
        }
    }
    store
}

/// Max element deviation (and the reference scale) between two packed quartics
/// of equal size, over the canonical quadruples.
fn max_element_error(a: &SymmetricFourth, b: &SymmetricFourth) -> (f64, f64) {
    let m = a.n();
    let mut err = 0.0_f64;
    let mut scale = 0.0_f64;
    for d in 0..m {
        for c in 0..=d {
            for bb in 0..=c {
                for aa in 0..=bb {
                    err = err.max((a.get(aa, bb, c, d) - b.get(aa, bb, c, d)).abs());
                    scale = scale.max(b.get(aa, bb, c, d).abs());
                }
            }
        }
    }
    (err, scale)
}

/// Assert the FD residual ladder: the coarse-step deviation is small, and the
/// halved-step deviation either shrinks like `h²` (~4×) or has already hit the
/// SCF/CPXTB noise floor. A residual that stays FLAT and well above the floor
/// means a composition error in the mixed-index assembly, not FD truncation.
///
/// The floor is set an order of magnitude BELOW the observed `h/2` truncation
/// residuals on purpose: at the tight SCF tolerances used here the FD noise is
/// nowhere near it, so it is the `h²` ratio — not the escape hatch — that
/// actually gates these tests.
fn assert_h2_ladder(label: &str, delta1: f64, delta2: f64, tol: f64, floor: f64) {
    eprintln!(
        "{label}: delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
        delta1 / delta2.max(1.0e-300)
    );
    assert!(delta1 < tol, "{label}: delta(h) {delta1:.3e} exceeds {tol:.3e}");
    assert!(
        delta2 < 0.4 * delta1 || delta2 < floor,
        "{label}: residual neither scales as h² nor sits on the noise floor \
         (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}, floor {floor:.3e}) — \
         suspect a composition error in the polarization assembly"
    );
}

/// **The mixed-index gate.** The full dense quartic of a skew stretched HF
/// (`ndof = 6`, 209 distinct polarization directions) vs the seminumerical
/// reference, element by element, with the `h²` ladder — plus the
/// self-consistency check that contracting the reconstructed tensor `vvvv`
/// reproduces the directional quartic it was built from.
///
/// HF rather than water only for runtime: the full water tensor needs 714
/// directional evaluations (~100 s); the water coverage is the block gate
/// below.
#[test]
fn dense_fourth_matches_seminumerical_skew_hf() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = skew_hf();
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let ndof = 3 * system.atoms.len();
    let dofs: Vec<usize> = (0..ndof).collect();

    let started = std::time::Instant::now();
    let dense = fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap();
    eprintln!("hf dense quartic ({ndof} dof) built in {:?}", started.elapsed());

    // Gate 1 — self-consistency: the polarization reconstruction contracted
    // `vvvv` must reproduce the directional quartic for the same `v`. The
    // 15-term alternating sum cancels ~2 orders of magnitude, so this is a
    // relative-1e-8 check, not machine precision.
    let v = skew_direction(ndof);
    let contracted = dense.contract_vvvv(&v).unwrap();
    let directional = directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap();
    let rel = (contracted - directional).abs() / (1.0 + directional.abs());
    eprintln!(
        "hf polarization consistency: Q·vvvv {contracted:.12e} directional {directional:.12e} \
         rel {rel:.3e}"
    );
    assert!(
        rel < 1.0e-8,
        "reconstructed tensor contracted vvvv disagrees with the directional quartic: \
         {contracted:.12e} vs {directional:.12e} (rel {rel:.3e})"
    );

    // Gate 2 — element-wise vs the seminumerical reference, two FD steps.
    let h = 1.0e-3;
    let semi1 = seminumerical_quartic_block(&system, &params, &options, cutoff, &dofs, h);
    let (delta1, scale) = max_element_error(&dense, &semi1);
    let semi2 = seminumerical_quartic_block(&system, &params, &options, cutoff, &dofs, 0.5 * h);
    let (delta2, _) = max_element_error(&dense, &semi2);
    eprintln!("hf element-wise reference scale {scale:.3e}");
    assert_h2_ladder(
        "hf dense quartic vs seminumerical",
        delta1,
        delta2,
        5.0e-5 * (1.0 + scale),
        1.0e-7 * (1.0 + scale),
    );
}

/// **The water coverage gate.** The `4⁴` sub-block of the non-equilibrium water
/// quartic over DOFs spanning all three atoms — 35 canonical quadruples
/// covering every index pattern (`aaaa`, `aaab`, `aabb`, `aabc`, `abcd`) — vs
/// the seminumerical reference, plus the same `Q·vvvv` self-consistency check
/// on a skew direction supported inside the block.
///
/// This is the full-model, full-response system (dispersion + halogen + CN
/// Hamiltonian + third-order onsite Γ); the block restriction is a runtime
/// budget, not a physics restriction — the polarization directions are exactly
/// those the full tensor would use for these quadruples.
#[test]
fn block_fourth_matches_seminumerical_noneq_water() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = noneq_water();
    let options = tight_options();
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    // O_x, H1_y, H1_z, H2_y — three atoms, mixed Cartesian components.
    let dofs = [0usize, 4, 5, 7];

    let started = std::time::Instant::now();
    let block =
        fourth_derivative_analytic_block(&system, &params, &options, cutoff, &dofs).unwrap();
    eprintln!("water quartic block {dofs:?} built in {:?}", started.elapsed());

    // Gate 1 — self-consistency on a direction supported inside the block:
    // `Σ_{abcd ∈ block} Q_abcd v_a v_b v_c v_d` is then the FULL `Q·vvvv`.
    let mut v_full = vec![0.0_f64; 3 * system.atoms.len()];
    let v_block = skew_direction(dofs.len());
    for (slot, &dof) in dofs.iter().enumerate() {
        v_full[dof] = v_block[slot];
    }
    let contracted = block.contract_vvvv(&v_block).unwrap();
    let directional =
        directional_fourth_derivative(&system, &params, &options, cutoff, &v_full).unwrap();
    let rel = (contracted - directional).abs() / (1.0 + directional.abs());
    eprintln!(
        "water polarization consistency: Q·vvvv {contracted:.12e} directional {directional:.12e} \
         rel {rel:.3e}"
    );
    assert!(
        rel < 1.0e-8,
        "reconstructed water block contracted vvvv disagrees with the directional quartic: \
         {contracted:.12e} vs {directional:.12e} (rel {rel:.3e})"
    );

    // Gate 2 — element-wise vs the seminumerical reference, two FD steps.
    let h = 1.0e-3;
    let semi1 = seminumerical_quartic_block(&system, &params, &options, cutoff, &dofs, h);
    let (delta1, scale) = max_element_error(&block, &semi1);
    let semi2 = seminumerical_quartic_block(&system, &params, &options, cutoff, &dofs, 0.5 * h);
    let (delta2, _) = max_element_error(&block, &semi2);
    eprintln!("water element-wise reference scale {scale:.3e}");
    assert_h2_ladder(
        "water quartic block vs seminumerical",
        delta1,
        delta2,
        5.0e-5 * (1.0 + scale),
        1.0e-7 * (1.0 + scale),
    );
}
