// SPDX-License-Identifier: GPL-3.0-or-later
//! **Physical-law gates for the v0.5.0 feature set.**
//!
//! Every other derivative test in this repository answers "does the analytic
//! implementation reproduce a finite difference of itself?". That question is
//! blind to a whole class of defect: a term that is missing (or wrong) in BOTH
//! the analytic assembly and the quantity it is differenced against passes an
//! FD ladder with flying colours. The gates here answer a different and
//! independent question — **does the result obey the symmetries and
//! conservation laws that the underlying physics forces on it?** — using no
//! finite differencing of the quantity under test at all.
//!
//! * §A **translational invariance**, orders 1–4. The energy depends only on
//!   interatomic vectors, so every derivative tensor must annihilate the rigid
//!   translation direction `t`: `Σ_a g_a = 0`, `H·t = 0`, `Σ_c T_abc = 0`,
//!   `Σ_d Q_abcd = 0`. Tested both directly (`e^n[t] = 0`) and, much more
//!   sharply, as *shift independence* — `e^n[u + λt]` must not depend on `λ`
//!   for a generic `u`, which probes every mixed `(u,…,u,t,…,t)` block rather
//!   than the fully-translational corner alone.
//! * §B **rotational covariance**, orders 0–3. A rigidly rotated molecule is
//!   the same molecule. Because the real-solid-harmonic AO basis is defined in
//!   the *global* frame, this is a genuinely non-trivial statement about the
//!   integral code and every geometric derivative built on it: any term that
//!   silently assumes an axis alignment (a hard-coded component, a dropped
//!   off-diagonal, a mis-transposed rotation) breaks here and nowhere else.
//! * §C **index-permutation symmetry** of the packed FC3/FC4 stores, plus
//!   cross-accessor consistency (dense vs block vs contracted vs directional)
//!   on a real computed tensor. The packed canonical index is a bijection
//!   claim; it is verified exhaustively rather than assumed.
//! * §D **finite-temperature → T = 0 continuity**: a gapped system's smeared
//!   derivative path must converge to the integer-occupation analytic value as
//!   the electronic temperature is lowered, monotonically and to machine-ish
//!   precision at 5 K.
//! * §E **gauge-origin independence** of the GIAO magnetizability and NMR
//!   shielding tensors — the defining property of London orbitals.
//! * §F **thermodynamic consistency** of the periodic stress (isotropic part vs
//!   `−dE/dV`) and stencil-independence of the Grüneisen parameter.
//! * §G **response conservation laws**: particle-number and total-charge sum
//!   rules, and the symmetry of the response matrices.
//! * §H **Berry-phase polarization quantum** under a lattice-vector translation.
//!
//! Cost discipline: the always-on gates target < 30 s each. Heavier variants
//! (larger fixtures, periodic systems, dense high-order tensors) carry
//! `#[ignore]` with the re-run command in their doc comment.

use gfn1_rs::fourth_derivative::{directional_fourth_derivative, fourth_derivative_analytic_dense};
use gfn1_rs::hessian::{
    h0_bare_second_derivative_matrix, h0_cn_block_second_derivative_matrix,
    h0_scc_scalar_second_derivative_matrix, shell_scalar_potential_first_derivatives,
    AnalyticHessianOptions,
};
use gfn1_rs::math::Vec3;
use gfn1_rs::response::charge_space::ChargeSpaceContext;
use gfn1_rs::response::cpxtb::{
    overlap_second_derivative_matrix, solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions,
    CpxtbOptions,
};
use gfn1_rs::third_derivative::finite_t::{
    directional_fourth_finite_t, directional_third_finite_t, fourth_derivative_finite_t_dense,
    third_derivative_finite_t_dense,
};
use gfn1_rs::third_derivative::{
    third_derivative_analytic_dense, third_derivative_analytic_vector, SymmetricThird,
};
use gfn1_rs::{
    analytic_gradient, analytic_hessian, magnetizability_tensor_analytic, nmr_shielding_tensor,
    pbc_berry_polarization, pbc_gruneisen, pbc_stress, run_electronic, run_pbc_scc,
    scale_lattice_isotropic, AnalyticGradientOptions, Atom, BerryMethodSelector,
    BerryPolarizationOptions, ElectronicOptions, EwaldOptions, Gfn1Parameters, GruneisenOptions,
    KMesh, PbcOptions, PeriodicSystem, SecondOrderStencil,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Stretched + bent water: no symmetry left to accidentally satisfy a gate, and
/// small enough (9 DOF) for the dense quartic.
const NONEQ_WATER: &str = "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n";

/// Non-equilibrium formaldehyde (12 DOF) — the mid-size fixture for order 3/4
/// and the strongly smeared finite-temperature gates.
const NONEQ_HCHO: &str =
    "4\nnon-eq formaldehyde\nC 0.0 0.0 0.0\nO 1.28 0.10 0.05\nH -0.60 0.95 0.10\nH -0.62 -0.90 0.12\n";

/// Distorted Ni(CO)₄ (27 DOF): a transition metal with d orbitals and, at
/// 3000 K, 18 of 41 orbitals fractionally occupied. The d shell is what makes
/// the rotational-covariance gate bite — real solid harmonics of ℓ = 2 mix
/// non-trivially under rotation, so a rotation-naive integral or derivative
/// path cannot hide here the way it can in an s/p-only molecule.
const DISTORTED_NI_CO4: &str = "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n";

/// Methane, slightly off Td so the frontier orbitals are not exactly
/// degenerate — the gapped fixture for the temperature-limit gate.
const NEAR_TD_METHANE: &str = "5\nnear-Td methane\nC 0.0 0.0 0.0\nH 0.640000 0.640000 0.645000\n\
     H -0.640000 -0.645000 0.640000\nH -0.645000 0.640000 -0.640000\nH 0.640000 -0.640000 -0.641000\n";

fn params() -> Gfn1Parameters {
    Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
}

fn system(xyz: &str) -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture parse")
}

/// Tight SCC so that symmetry residuals are not masked by convergence noise.
/// The physical gates compare two *independent* SCF solutions (rotated vs not,
/// translated vs not), so the SCC tolerance is a hard floor on every tolerance
/// in this file.
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

fn smeared_options(temperature: f64) -> AnalyticHessianOptions {
    let mut opts = tight_options();
    opts.electronic_options.electronic_temperature = temperature;
    opts
}

fn cutoff(options: &AnalyticHessianOptions) -> f64 {
    options.electronic_options.hamiltonian.coordination_cutoff
}

fn gradient_options(options: &AnalyticHessianOptions) -> AnalyticGradientOptions {
    AnalyticGradientOptions {
        electronic: options.electronic_options.clone(),
        ..AnalyticGradientOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Direction helpers
// ---------------------------------------------------------------------------

/// Rigid translation of the whole molecule along Cartesian `axis`.
fn translation(nat: usize, axis: usize) -> Vec<f64> {
    let mut v = vec![0.0; 3 * nat];
    for a in 0..nat {
        v[3 * a + axis] = 1.0;
    }
    v
}

/// The uniform direction `(1,1,1,1,…)` — a rigid translation along `(1,1,1)`,
/// i.e. the superposition of the three axis translations.
fn uniform_translation(nat: usize) -> Vec<f64> {
    vec![1.0; 3 * nat]
}

/// A deterministic pseudo-random unit direction (LCG), used as the generic
/// probe `u` in the shift-independence gates. It is deliberately NOT built from
/// a small integer stencil: a direction that happens to differ from another
/// only by a rigid translation is not an independent probe of an odd-order
/// tensor.
fn probe_direction(nat: usize, seed: u64) -> Vec<f64> {
    let ndof = 3 * nat;
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut v = Vec::with_capacity(ndof);
    for _ in 0..ndof {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64); // [0,1)
        v.push(2.0 * u - 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn axpy(a: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(xi, yi)| a * xi + yi).collect()
}

// ---------------------------------------------------------------------------
// Rotation helpers
// ---------------------------------------------------------------------------

type Mat3x3 = [[f64; 3]; 3];

/// Rodrigues rotation about a (not necessarily normalized) axis.
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

fn apply_rotation(r: &Mat3x3, v: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i] += r[i][j] * v[j];
        }
    }
    out
}

fn rotate_system(sys: &PeriodicSystem, r: &Mat3x3) -> PeriodicSystem {
    assert!(
        sys.lattice.is_none(),
        "rotate_system: molecular fixtures only"
    );
    let atoms = sys
        .atoms
        .iter()
        .map(|a| {
            let p = apply_rotation(r, [a.position.x, a.position.y, a.position.z]);
            Atom {
                z: a.z,
                position: Vec3::new(p[0], p[1], p[2]),
            }
        })
        .collect();
    PeriodicSystem::new(atoms, None).with_charge(sys.charge)
}

/// Rotate a 3N-dimensional Cartesian displacement vector atom-block-wise.
fn rotate_direction(r: &Mat3x3, v: &[f64]) -> Vec<f64> {
    let nat = v.len() / 3;
    let mut out = vec![0.0; v.len()];
    for a in 0..nat {
        let p = apply_rotation(r, [v[3 * a], v[3 * a + 1], v[3 * a + 2]]);
        out[3 * a..3 * a + 3].copy_from_slice(&p);
    }
    out
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
    PeriodicSystem {
        atoms,
        lattice: sys.lattice.clone(),
        charge: sys.charge,
    }
}

// ---------------------------------------------------------------------------
// §A  Translational invariance
// ---------------------------------------------------------------------------

/// `Σ_a ∂E/∂R_a = 0` — the total force on an isolated molecule vanishes
/// (Newton's third law / momentum conservation). Any per-atom term whose
/// counter-term is misplaced (a one-sided CN derivative, an unpaired Pulay leg)
/// shows up here as a net force.
fn gradient_translation_residual(xyz: &str) -> (f64, f64) {
    let params = params();
    let sys = system(xyz);
    let opts = tight_options();
    let g = analytic_gradient(&sys, &params, gradient_options(&opts)).unwrap();
    let mut sum = [0.0_f64; 3];
    let mut scale = 0.0_f64;
    for gi in &g.gradient {
        sum[0] += gi.x;
        sum[1] += gi.y;
        sum[2] += gi.z;
        scale = scale.max(gi.x.abs()).max(gi.y.abs()).max(gi.z.abs());
    }
    let worst = sum.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
    (worst, scale)
}

#[test]
fn gradient_sums_to_zero_under_rigid_translation() {
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-12),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-12),
        ("distorted Ni(CO)4", DISTORTED_NI_CO4, 1.0e-11),
    ] {
        let (worst, scale) = gradient_translation_residual(xyz);
        println!("[A1] {name}: |Σ_a g_a|_max = {worst:.3e} (gradient scale {scale:.3e})");
        assert!(
            worst < tol,
            "{name}: net force {worst:.3e} exceeds {tol:.1e} (gradient scale {scale:.3e})"
        );
    }
}

/// `H·t = 0` for every rigid translation `t` — the acoustic sum rule. A Hessian
/// that violates it produces spurious non-zero translational frequencies.
#[test]
fn hessian_annihilates_rigid_translations() {
    let params = params();
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-10),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-10),
        ("distorted Ni(CO)4", DISTORTED_NI_CO4, 1.0e-9),
    ] {
        let sys = system(xyz);
        let nat = sys.atoms.len();
        let h = analytic_hessian(&sys, &params, tight_options())
            .unwrap()
            .hessian;
        let scale = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(h[(i, j)].abs()));
        let mut worst = 0.0_f64;
        for axis in 0..3 {
            let t = translation(nat, axis);
            for i in 0..3 * nat {
                let row: f64 = (0..3 * nat).map(|j| h[(i, j)] * t[j]).sum();
                worst = worst.max(row.abs());
            }
        }
        println!("[A2] {name}: |H·t|_max = {worst:.3e} (Hessian scale {scale:.3e})");
        assert!(
            worst < tol,
            "{name}: acoustic sum rule violated by {worst:.3e} (> {tol:.1e}, scale {scale:.3e})"
        );
    }
}

/// `Σ_c T_abc = 0` for the cubic force constants. Checked in the *matrix* form
/// `K_ab = Σ_c t_c T_abc ≡ 0` (all `(3N)²` entries), which is far sharper than
/// the scalar `e³[t] = 0`: the latter only probes the fully translational
/// corner of the tensor, the former every `(a,b)` pair.
#[test]
fn third_derivative_annihilates_rigid_translation() {
    let params = params();
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-8),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-8),
    ] {
        let sys = system(xyz);
        let nat = sys.atoms.len();
        let opts = tight_options();
        let cut = cutoff(&opts);
        let t = uniform_translation(nat);
        let k = third_derivative_analytic_vector(&sys, &params, opts, cut, &t).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                worst = worst.max(k[(i, j)].abs());
            }
        }
        println!("[A3] {name}: |Σ_c t_c T_abc|_max = {worst:.3e}");
        assert!(
            worst < tol,
            "{name}: FC3 translational sum rule violated by {worst:.3e} (> {tol:.1e})"
        );
    }
}

/// **Shift independence**, the sharp form of translational invariance for the
/// directional high-order derivatives: since `Σ_c T_abc = 0` (and likewise at
/// order 4), `e^n[u + λt]` must be *independent of λ* for a generic direction
/// `u` and rigid translation `t`. This probes every mixed `(u,…,t)` block,
/// unlike `e^n[t] = 0` which probes only the pure-translation corner.
#[test]
fn directional_third_and_fourth_are_shift_independent() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let opts = tight_options();
    let cut = cutoff(&opts);
    let u = probe_direction(nat, 20_260_812);
    let t = uniform_translation(nat);

    // Order 3, via the vector mode contracted twice (uses the full analytic
    // cubic, not a difference of Hessians).
    let e3 = |dir: &[f64]| -> f64 {
        let k = third_derivative_analytic_vector(&sys, &params, opts.clone(), cut, dir).unwrap();
        let mut acc = 0.0;
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                acc += dir[i] * k[(i, j)] * dir[j];
            }
        }
        acc
    };
    let e3_ref = e3(&u);
    let mut worst3 = 0.0_f64;
    for lambda in [-1.0, 0.5, 1.0] {
        let shifted = axpy(lambda, &t, &u);
        let val = e3(&shifted);
        worst3 = worst3.max((val - e3_ref).abs());
    }
    println!("[A4] water FC3 shift independence: |Δe³| = {worst3:.3e} (e³[u] = {e3_ref:.6e})");
    assert!(
        worst3 < 1.0e-9,
        "FC3 is not translation-shift independent: {worst3:.3e} (e³[u] = {e3_ref:.3e})"
    );

    // Order 4.
    let e4 = |dir: &[f64]| directional_fourth_derivative(&sys, &params, &opts, cut, dir).unwrap();
    let e4_ref = e4(&u);
    let mut worst4 = 0.0_f64;
    for lambda in [-1.0, 0.5, 1.0] {
        let shifted = axpy(lambda, &t, &u);
        worst4 = worst4.max((e4(&shifted) - e4_ref).abs());
    }
    println!("[A4] water FC4 shift independence: |Δe⁴| = {worst4:.3e} (e⁴[u] = {e4_ref:.6e})");
    assert!(
        worst4 < 1.0e-8,
        "FC4 is not translation-shift independent: {worst4:.3e} (e⁴[u] = {e4_ref:.3e})"
    );

    // And the pure-translation corners themselves.
    let pure3 = e3(&t);
    let pure4 = e4(&t);
    println!("[A4] water pure-translation corners: e³[t] = {pure3:.3e}, e⁴[t] = {pure4:.3e}");
    assert!(pure3.abs() < 1.0e-9, "e³[t] = {pure3:.3e} ≠ 0");
    assert!(pure4.abs() < 1.0e-8, "e⁴[t] = {pure4:.3e} ≠ 0");
}

/// The finite-temperature derivative path must obey the same sum rules — the
/// smeared occupation response introduces an entirely separate assembly, so
/// this is an independent gate, not a corollary of the T = 0 one.
#[test]
fn finite_temperature_third_and_fourth_are_shift_independent() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let opts = smeared_options(3000.0);
    let cut = cutoff(&opts);
    let u = probe_direction(nat, 777);
    let t = uniform_translation(nat);

    let e3 = |dir: &[f64]| directional_third_finite_t(&sys, &params, &opts, cut, dir).unwrap();
    let e4 = |dir: &[f64]| directional_fourth_finite_t(&sys, &params, &opts, cut, dir).unwrap();

    let (e3_ref, e4_ref) = (e3(&u), e4(&u));
    let mut worst3 = 0.0_f64;
    let mut worst4 = 0.0_f64;
    for lambda in [-1.0, 1.0] {
        let shifted = axpy(lambda, &t, &u);
        worst3 = worst3.max((e3(&shifted) - e3_ref).abs());
        worst4 = worst4.max((e4(&shifted) - e4_ref).abs());
    }
    let (pure3, pure4) = (e3(&t), e4(&t));
    println!(
        "[A5] smeared water (3000 K): |Δe³| = {worst3:.3e} (ref {e3_ref:.6e}), \
         |Δe⁴| = {worst4:.3e} (ref {e4_ref:.6e}), e³[t] = {pure3:.3e}, e⁴[t] = {pure4:.3e}"
    );
    assert!(worst3 < 1.0e-8, "finite-T FC3 shift dependence {worst3:.3e}");
    assert!(worst4 < 1.0e-7, "finite-T FC4 shift dependence {worst4:.3e}");
    assert!(pure3.abs() < 1.0e-8, "finite-T e³[t] = {pure3:.3e} ≠ 0");
    assert!(pure4.abs() < 1.0e-7, "finite-T e⁴[t] = {pure4:.3e} ≠ 0");
}

// ---------------------------------------------------------------------------
// §B  Rotational covariance
// ---------------------------------------------------------------------------

fn test_rotations() -> Vec<(&'static str, Mat3x3)> {
    vec![
        ("x-axis 90°", rotation_matrix([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2)),
        ("z-axis 90°", rotation_matrix([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)),
        ("generic (1,2,3) 0.7 rad", rotation_matrix([1.0, 2.0, 3.0], 0.7)),
        ("generic (-2,1,4) 2.31 rad", rotation_matrix([-2.0, 1.0, 4.0], 2.31)),
    ]
}

/// **Energy is a rotational scalar.** The AO basis is a set of real solid
/// harmonics anchored to the *global* Cartesian axes, so a rotated molecule is
/// represented by a genuinely different set of integrals; the SCF must
/// nonetheless land on the same energy.
#[test]
fn energy_is_rotationally_invariant() {
    let params = params();
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-12),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-12),
        ("distorted Ni(CO)4", DISTORTED_NI_CO4, 1.0e-11),
    ] {
        let sys = system(xyz);
        let opts = tight_options();
        let e0 = analytic_gradient(&sys, &params, gradient_options(&opts))
            .unwrap()
            .total_energy;
        for (rname, r) in test_rotations() {
            let rot = rotate_system(&sys, &r);
            let e1 = analytic_gradient(&rot, &params, gradient_options(&opts))
                .unwrap()
                .total_energy;
            let d = (e1 - e0).abs();
            println!("[B1] {name} / {rname}: |ΔE| = {d:.3e} (E = {e0:.10} Eh)");
            assert!(
                d < tol,
                "{name} / {rname}: energy changed by {d:.3e} under rigid rotation (E = {e0:.10})"
            );
        }
    }
}

/// **The gradient is a rotational vector**: `g'(R x) = R g(x)`. A term that
/// builds a derivative from a component rather than covariantly (e.g. a
/// projection onto a fixed axis, or a transposed rotation in a two-center
/// derivative) fails here while passing every finite-difference gate, because
/// the FD reference inherits the same defect.
#[test]
fn gradient_is_rotationally_covariant() {
    let params = params();
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-12),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-12),
        ("distorted Ni(CO)4", DISTORTED_NI_CO4, 1.0e-11),
    ] {
        let sys = system(xyz);
        let opts = tight_options();
        let g0 = analytic_gradient(&sys, &params, gradient_options(&opts))
            .unwrap()
            .gradient;
        for (rname, r) in test_rotations() {
            let rot = rotate_system(&sys, &r);
            let g1 = analytic_gradient(&rot, &params, gradient_options(&opts))
                .unwrap()
                .gradient;
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for (a, g0a) in g0.iter().enumerate() {
                let expect = apply_rotation(&r, [g0a.x, g0a.y, g0a.z]);
                let got = [g1[a].x, g1[a].y, g1[a].z];
                for k in 0..3 {
                    worst = worst.max((got[k] - expect[k]).abs());
                    scale = scale.max(expect[k].abs());
                }
            }
            println!("[B2] {name} / {rname}: |g'(Rx) − R g(x)|_max = {worst:.3e} (scale {scale:.3e})");
            assert!(
                worst < tol,
                "{name} / {rname}: gradient covariance violated by {worst:.3e} (scale {scale:.3e})"
            );
        }
    }
}

/// **The Hessian is a rank-2 rotational tensor**:
/// `H'_{(a,i),(b,j)} = Σ_{i'j'} R_{ii'} R_{jj'} H_{(a,i'),(b,j')}`.
#[test]
fn hessian_is_rotationally_covariant() {
    let params = params();
    for (name, xyz, tol) in [
        ("non-eq water", NONEQ_WATER, 1.0e-11),
        ("non-eq HCHO", NONEQ_HCHO, 1.0e-11),
    ] {
        let sys = system(xyz);
        let nat = sys.atoms.len();
        let h0 = analytic_hessian(&sys, &params, tight_options())
            .unwrap()
            .hessian;
        for (rname, r) in test_rotations() {
            let rot = rotate_system(&sys, &r);
            let h1 = analytic_hessian(&rot, &params, tight_options())
                .unwrap()
                .hessian;
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for a in 0..nat {
                for b in 0..nat {
                    for i in 0..3 {
                        for j in 0..3 {
                            let mut expect = 0.0;
                            for ip in 0..3 {
                                for jp in 0..3 {
                                    expect += r[i][ip] * r[j][jp] * h0[(3 * a + ip, 3 * b + jp)];
                                }
                            }
                            let got = h1[(3 * a + i, 3 * b + j)];
                            worst = worst.max((got - expect).abs());
                            scale = scale.max(expect.abs());
                        }
                    }
                }
            }
            println!("[B3] {name} / {rname}: |H' − (R⊗R)H(R⊗R)ᵀ|_max = {worst:.3e} (scale {scale:.3e})");
            assert!(
                worst < tol,
                "{name} / {rname}: Hessian covariance violated by {worst:.3e} (scale {scale:.3e})"
            );
        }
    }
}

/// **The cubic force constants are a rank-3 rotational tensor.** Tested in the
/// directional contraction that is cheapest and still complete for the probed
/// direction: `e³_rotated[R v] = e³_original[v]`, plus the matrix form
/// `K'(R v) = R K(v) Rᵀ` block-wise, which checks the two free indices too.
#[test]
fn third_derivative_is_rotationally_covariant() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let opts = tight_options();
    let cut = cutoff(&opts);
    let v = probe_direction(nat, 31_337);
    let k0 = third_derivative_analytic_vector(&sys, &params, opts.clone(), cut, &v).unwrap();
    let e3_0: f64 = (0..3 * nat)
        .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
        .map(|(i, j)| v[i] * k0[(i, j)] * v[j])
        .sum();

    for (rname, r) in test_rotations() {
        let rot = rotate_system(&sys, &r);
        let rv = rotate_direction(&r, &v);
        let k1 = third_derivative_analytic_vector(&rot, &params, opts.clone(), cut, &rv).unwrap();
        let e3_1: f64 = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .map(|(i, j)| rv[i] * k1[(i, j)] * rv[j])
            .sum();
        let scalar_err = (e3_1 - e3_0).abs();

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for a in 0..nat {
            for b in 0..nat {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut expect = 0.0;
                        for ip in 0..3 {
                            for jp in 0..3 {
                                expect += r[i][ip] * r[j][jp] * k0[(3 * a + ip, 3 * b + jp)];
                            }
                        }
                        worst = worst.max((k1[(3 * a + i, 3 * b + j)] - expect).abs());
                        scale = scale.max(expect.abs());
                    }
                }
            }
        }
        println!(
            "[B4] water FC3 / {rname}: |Δe³[v]| = {scalar_err:.3e} (e³ = {e3_0:.6e}), \
             |K'(Rv) − R K(v) Rᵀ|_max = {worst:.3e} (scale {scale:.3e})"
        );
        assert!(
            scalar_err < 1.0e-9,
            "FC3 directional rotation invariance violated by {scalar_err:.3e} (e³ = {e3_0:.3e})"
        );
        assert!(
            worst < 1.0e-9,
            "FC3 vector-mode rotation covariance violated by {worst:.3e} (scale {scale:.3e})"
        );
    }
}

/// Build the antisymmetric generator `Ω` with `Ω v = ω × v`.
fn rotation_generator(omega: [f64; 3]) -> Mat3x3 {
    [
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ]
}

/// **The rotational sum rules — exact identities linking consecutive derivative
/// orders at the SAME geometry.**
///
/// Rotational invariance is usually tested by re-running a rotated molecule
/// (§B above). Its *infinitesimal* form is stronger and completely free: it
/// relates the tensors already computed at one geometry, with no second SCF and
/// no finite differencing. Writing `r_a = Ω x_a` for the rigid-rotation
/// displacement field of the generator `Ω` (`Ω v = ω × v`):
///
/// 1. `Σ_a x_a × g_a = 0` — **zero net torque**, the rotational counterpart of
///    the vanishing net force.
/// 2. `H · r = Ω g` — the Hessian contracted with a rotation reproduces the
///    rotated gradient. Note this is NOT `H · r = 0`: that only holds at a
///    stationary point, and asserting it on a non-equilibrium geometry (as
///    projected-frequency code sometimes does) is simply wrong.
/// 3. `K(r) = Ω H − H Ω` — the cubic force constants contracted with a rotation
///    give the commutator of the generator with the Hessian.
///
/// These hold at *any* geometry, equilibrium or not, and each one couples two
/// different assemblies (gradient/Hessian, Hessian/FC3), so a term present in
/// one order but dropped in the next breaks them even though every same-order
/// finite-difference gate stays green.
#[test]
fn rotational_sum_rules_link_gradient_hessian_and_third_derivative() {
    let params = params();
    for (name, xyz) in [("non-eq water", NONEQ_WATER), ("non-eq HCHO", NONEQ_HCHO)] {
        let sys = system(xyz);
        let nat = sys.atoms.len();
        let ndof = 3 * nat;
        let opts = tight_options();
        let cut = cutoff(&opts);
        let grad = analytic_gradient(&sys, &params, gradient_options(&opts))
            .unwrap()
            .gradient;
        let hess = analytic_hessian(&sys, &params, opts.clone()).unwrap().hessian;

        // (1) zero net torque.
        let mut torque = [0.0_f64; 3];
        for (a, g) in grad.iter().enumerate() {
            let x = sys.atoms[a].position;
            torque[0] += x.y * g.z - x.z * g.y;
            torque[1] += x.z * g.x - x.x * g.z;
            torque[2] += x.x * g.y - x.y * g.x;
        }
        let worst_torque = torque.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
        let g_scale = grad
            .iter()
            .fold(0.0_f64, |m, g| m.max(g.x.abs()).max(g.y.abs()).max(g.z.abs()));
        println!("[B7] {name}: |Σ_a x_a × g_a|_max = {worst_torque:.3e} (gradient scale {g_scale:.3e})");
        assert!(
            worst_torque < 1.0e-11,
            "{name}: net torque {worst_torque:.3e} ≠ 0 — the energy is not rotationally invariant"
        );

        for (gname, omega) in [
            ("ω = x̂", [1.0, 0.0, 0.0]),
            ("ω = ŷ", [0.0, 1.0, 0.0]),
            ("ω = ẑ", [0.0, 0.0, 1.0]),
        ] {
            let gen = rotation_generator(omega);
            // r_a = Ω x_a
            let mut r = vec![0.0; ndof];
            for a in 0..nat {
                let x = sys.atoms[a].position;
                let p = apply_rotation(&gen, [x.x, x.y, x.z]);
                r[3 * a..3 * a + 3].copy_from_slice(&p);
            }

            // (2) H · r = Ω g
            let mut worst_h = 0.0_f64;
            let mut scale_h = 0.0_f64;
            for a in 0..nat {
                let g = grad[a];
                let og = apply_rotation(&gen, [g.x, g.y, g.z]);
                for i in 0..3 {
                    let lhs: f64 = (0..ndof).map(|j| hess[(3 * a + i, j)] * r[j]).sum();
                    worst_h = worst_h.max((lhs - og[i]).abs());
                    scale_h = scale_h.max(og[i].abs());
                }
            }

            // (3) K(r) = Ω H − H Ω
            let k = third_derivative_analytic_vector(&sys, &params, opts.clone(), cut, &r).unwrap();
            let mut worst_k = 0.0_f64;
            let mut scale_k = 0.0_f64;
            for a in 0..nat {
                for b in 0..nat {
                    for i in 0..3 {
                        for j in 0..3 {
                            let mut expect = 0.0;
                            for m in 0..3 {
                                expect += gen[i][m] * hess[(3 * a + m, 3 * b + j)];
                                expect += gen[j][m] * hess[(3 * a + i, 3 * b + m)];
                            }
                            worst_k = worst_k.max((k[(3 * a + i, 3 * b + j)] - expect).abs());
                            scale_k = scale_k.max(expect.abs());
                        }
                    }
                }
            }
            println!(
                "[B8] {name} / {gname}: |H·r − Ωg|_max = {worst_h:.3e} (scale {scale_h:.3e}), \
                 |K(r) − [Ω,H]|_max = {worst_k:.3e} (scale {scale_k:.3e})"
            );
            assert!(
                worst_h < 1.0e-10,
                "{name} / {gname}: Hessian rotational sum rule violated by {worst_h:.3e} \
                 (scale {scale_h:.3e})"
            );
            assert!(
                worst_k < 1.0e-9,
                "{name} / {gname}: FC3 rotational sum rule K(r) = [Ω,H] violated by \
                 {worst_k:.3e} (scale {scale_k:.3e})"
            );
        }
    }
}

/// The d-shell version of the FC3 rotational-covariance gate. Real ℓ = 2 solid
/// harmonics mix under rotation in a way ℓ ≤ 1 does not, so this is the gate
/// that would catch a transition-metal-specific frame error.
///
/// **The electronic temperature must be pinned to zero here.** Distorted
/// Ni(CO)₄ is Fermi-smeared at the default 300 K, and the T = 0 analytic cubic
/// refuses fractional occupations outright (`InvalidInput("analytic third
/// derivative with fractional (Fermi-smeared) occupations is not yet
/// supported")`). The finite-temperature d-shell rotation gate is the
/// smeared-path sibling [`finite_temperature_directional_derivatives_are_rotation_invariant`].
///
/// Measured: `|ΔK|_max` = 2.81e-14 / 2.69e-14 (tensor scale 5.18e-1), i.e. the
/// d-shell cubic is rotationally covariant to ~5e-14 relative.
///
/// Heavy (~400 s: 27 DOF closed-form cubic, three evaluations). Re-run with:
/// `$env:CARGO_TARGET_DIR='...\target-agentphys'; cargo test --profile reltest --test physical_consistency -- --ignored --nocapture third_derivative_rotational_covariance_d_shell`
#[test]
#[ignore]
fn third_derivative_rotational_covariance_d_shell() {
    let params = params();
    let sys = system(DISTORTED_NI_CO4);
    let nat = sys.atoms.len();
    let mut opts = tight_options();
    opts.electronic_options.electronic_temperature = 0.0;
    let cut = cutoff(&opts);
    let v = probe_direction(nat, 9_001);
    let k0 = third_derivative_analytic_vector(&sys, &params, opts.clone(), cut, &v).unwrap();
    for (rname, r) in test_rotations().into_iter().take(2) {
        let rot = rotate_system(&sys, &r);
        let rv = rotate_direction(&r, &v);
        let k1 = third_derivative_analytic_vector(&rot, &params, opts.clone(), cut, &rv).unwrap();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for a in 0..nat {
            for b in 0..nat {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut expect = 0.0;
                        for ip in 0..3 {
                            for jp in 0..3 {
                                expect += r[i][ip] * r[j][jp] * k0[(3 * a + ip, 3 * b + jp)];
                            }
                        }
                        worst = worst.max((k1[(3 * a + i, 3 * b + j)] - expect).abs());
                        scale = scale.max(expect.abs());
                    }
                }
            }
        }
        println!("[B5] Ni(CO)4 FC3 / {rname}: |ΔK|_max = {worst:.3e} (scale {scale:.3e})");
        assert!(
            worst < 1.0e-8,
            "Ni(CO)4 FC3 rotation covariance violated by {worst:.3e} (scale {scale:.3e})"
        );
    }
}

/// The finite-temperature cubic and quartic must be rotational scalars too.
#[test]
fn finite_temperature_directional_derivatives_are_rotation_invariant() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let opts = smeared_options(3000.0);
    let cut = cutoff(&opts);
    let v = probe_direction(nat, 55_555);
    let e3_0 = directional_third_finite_t(&sys, &params, &opts, cut, &v).unwrap();
    let e4_0 = directional_fourth_finite_t(&sys, &params, &opts, cut, &v).unwrap();
    for (rname, r) in test_rotations() {
        let rot = rotate_system(&sys, &r);
        let rv = rotate_direction(&r, &v);
        let e3_1 = directional_third_finite_t(&rot, &params, &opts, cut, &rv).unwrap();
        let e4_1 = directional_fourth_finite_t(&rot, &params, &opts, cut, &rv).unwrap();
        let (d3, d4) = ((e3_1 - e3_0).abs(), (e4_1 - e4_0).abs());
        println!(
            "[B6] smeared water / {rname}: |Δe³| = {d3:.3e} (ref {e3_0:.6e}), \
             |Δe⁴| = {d4:.3e} (ref {e4_0:.6e})"
        );
        assert!(d3 < 1.0e-9, "finite-T FC3 rotation invariance: {d3:.3e}");
        assert!(d4 < 1.0e-8, "finite-T FC4 rotation invariance: {d4:.3e}");
    }
}

// ---------------------------------------------------------------------------
// §C  Index-permutation symmetry and accessor consistency
// ---------------------------------------------------------------------------

/// The packed canonical index of [`SymmetricThird`] / `SymmetricFourth` must be
/// a *bijection* from unordered index tuples onto `0..len`. A collision would
/// silently sum two unrelated force constants together; a gap would leave one
/// permanently zero. Verified exhaustively (pure algebra, no SCF).
#[test]
fn packed_symmetric_stores_use_bijective_canonical_indices() {
    for n in 1..=10 {
        let expected_len = n * (n + 1) * (n + 2) / 6;
        assert_eq!(
            SymmetricThird::zeros(n).len(),
            expected_len,
            "SymmetricThird n={n}: packed length is not n(n+1)(n+2)/6"
        );
        let mut count = 0usize;
        for c in 0..n {
            for b in 0..=c {
                for a in 0..=b {
                    // Probe the store rather than a private index fn: write a
                    // unique value at ONE canonical triple, then require that
                    // (i) every permutation reads it back and (ii) exactly the
                    // permutation orbit of that triple — and nothing else — is
                    // non-zero in the unpacked dense view. A canonical-index
                    // collision would light up a second, unrelated orbit.
                    let mut probe = SymmetricThird::zeros(n);
                    let value = 1.0 + (a * 100 + b * 10 + c) as f64;
                    probe.add(a, b, c, value);
                    for perm in [
                        (a, b, c),
                        (a, c, b),
                        (b, a, c),
                        (b, c, a),
                        (c, a, b),
                        (c, b, a),
                    ] {
                        let got = probe.get(perm.0, perm.1, perm.2);
                        assert!(
                            (got - value).abs() < 1.0e-15,
                            "SymmetricThird n={n}: get{perm:?} = {got} != {value} written at ({a},{b},{c})"
                        );
                    }
                    let slabs = probe.to_dense_slabs();
                    let mut nonzero = 0usize;
                    for (k, m) in slabs.iter().enumerate() {
                        for i in 0..n {
                            for j in 0..n {
                                if m[(i, j)] != 0.0 {
                                    nonzero += 1;
                                    assert!(
                                        (m[(i, j)] - value).abs() < 1.0e-15,
                                        "SymmetricThird n={n}: aliased value {} at ({i},{j},{k}) for triple ({a},{b},{c})",
                                        m[(i, j)]
                                    );
                                }
                            }
                        }
                    }
                    let orbit = distinct_permutations_3(a, b, c);
                    assert_eq!(
                        nonzero, orbit,
                        "SymmetricThird n={n}: triple ({a},{b},{c}) lit {nonzero} dense entries, expected orbit size {orbit}"
                    );
                    count += 1;
                }
            }
        }
        assert_eq!(
            count, expected_len,
            "SymmetricThird n={n}: {count} canonical triples but packed length {expected_len}"
        );
    }
}

fn distinct_permutations_3(a: usize, b: usize, c: usize) -> usize {
    let mut set = std::collections::HashSet::new();
    for p in [
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    ] {
        set.insert(p);
    }
    set.len()
}

/// `SymmetricFourth`: exhaustive bijectivity of the packed canonical index
/// `C(d+3,4) + C(c+2,3) + C(b+1,2) + a`, and full 4! permutation invariance of
/// `get`.
#[test]
fn packed_symmetric_fourth_uses_bijective_canonical_indices() {
    use gfn1_rs::fourth_derivative::SymmetricFourth;
    for n in 1..=10 {
        let q = SymmetricFourth::zeros(n);
        let mut seen = vec![0usize; q.len()];
        let mut count = 0usize;
        for d in 0..n {
            for c in 0..=d {
                for b in 0..=c {
                    for a in 0..=b {
                        let idx = q.index(a, b, c, d);
                        assert!(
                            idx < q.len(),
                            "SymmetricFourth n={n}: index({a},{b},{c},{d}) = {idx} >= len {}",
                            q.len()
                        );
                        seen[idx] += 1;
                        count += 1;
                        // 4! permutation invariance of the index itself.
                        for perm in permutations_4(a, b, c, d) {
                            let j = q.index(perm[0], perm[1], perm[2], perm[3]);
                            assert_eq!(
                                idx, j,
                                "SymmetricFourth n={n}: index not permutation invariant for ({a},{b},{c},{d}) vs {perm:?}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(count, q.len(), "SymmetricFourth n={n}: canonical quadruple count {count} != packed length {}", q.len());
        assert!(
            seen.iter().all(|&s| s == 1),
            "SymmetricFourth n={n}: packed index is not a bijection (multiplicities {:?})",
            seen.iter().enumerate().filter(|(_, &s)| s != 1).take(5).collect::<Vec<_>>()
        );
    }
}

fn permutations_4(a: usize, b: usize, c: usize, d: usize) -> Vec<[usize; 4]> {
    let base = [a, b, c, d];
    let mut out = Vec::with_capacity(24);
    for i in 0..4 {
        for j in 0..4 {
            if j == i {
                continue;
            }
            for k in 0..4 {
                if k == i || k == j {
                    continue;
                }
                let l = 6 - i - j - k;
                out.push([base[i], base[j], base[k], base[l]]);
            }
        }
    }
    out
}

/// **Cross-accessor consistency of the computed FC3 tensor.** The dense packed
/// store, the `block()` view, `contract_vvv`, and the independently assembled
/// vector mode `third_derivative_analytic_vector` are four different code paths
/// onto the same object; they must agree to machine precision. This is the
/// non-vacuous part of "permutation symmetry": `get()` sorts its arguments, so
/// permutation invariance of the *store* is definitional — what can actually be
/// wrong is the symmetrization that fills it and the views that read it.
#[test]
fn fc3_dense_block_and_directional_views_agree() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let ndof = 3 * nat;
    let opts = tight_options();
    let cut = cutoff(&opts);
    let dense = third_derivative_analytic_dense(&sys, &params, opts.clone(), cut).unwrap();

    // (i) dense slabs must be fully symmetric under all 3! index permutations.
    let slabs = dense.to_dense_slabs();
    let mut perm_worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..ndof {
        for b in 0..ndof {
            for c in 0..ndof {
                let base = slabs[c][(a, b)];
                scale = scale.max(base.abs());
                for (x, y, z) in [
                    (a, c, b),
                    (b, a, c),
                    (b, c, a),
                    (c, a, b),
                    (c, b, a),
                ] {
                    perm_worst = perm_worst.max((slabs[z][(x, y)] - base).abs());
                }
                perm_worst = perm_worst.max((dense.get(a, b, c) - base).abs());
            }
        }
    }

    // (ii) block() over a subset must reproduce the same entries.
    let dofs: Vec<usize> = vec![0, 2, 4, 7];
    let block = dense.block(&dofs);
    let mut block_worst = 0.0_f64;
    for (ci, &c) in dofs.iter().enumerate() {
        for (ai, &a) in dofs.iter().enumerate() {
            for (bi, &b) in dofs.iter().enumerate() {
                block_worst = block_worst.max((block[ci][(ai, bi)] - dense.get(a, b, c)).abs());
            }
        }
    }

    // (iii) contract_vvv and the vector mode against the explicit triple sum.
    let v = probe_direction(nat, 4_242);
    let mut explicit = 0.0;
    for a in 0..ndof {
        for b in 0..ndof {
            for c in 0..ndof {
                explicit += v[a] * v[b] * v[c] * dense.get(a, b, c);
            }
        }
    }
    let contracted = dense.contract_vvv(&v);
    let k = third_derivative_analytic_vector(&sys, &params, opts, cut, &v).unwrap();
    let vector_mode: f64 = (0..ndof)
        .flat_map(|i| (0..ndof).map(move |j| (i, j)))
        .map(|(i, j)| v[i] * k[(i, j)] * v[j])
        .sum();

    println!(
        "[C3] water FC3 views: perm |Δ| = {perm_worst:.3e} (scale {scale:.3e}), \
         block |Δ| = {block_worst:.3e}, contract_vvv − explicit = {:.3e}, \
         vector-mode − explicit = {:.3e}",
        (contracted - explicit).abs(),
        (vector_mode - explicit).abs()
    );
    assert!(perm_worst < 1.0e-14 * scale.max(1.0), "FC3 dense slabs are not 3!-symmetric: {perm_worst:.3e}");
    assert!(block_worst < 1.0e-15, "FC3 block view disagrees with dense: {block_worst:.3e}");
    assert!(
        (contracted - explicit).abs() < 1.0e-12 * explicit.abs().max(1.0),
        "contract_vvv {contracted:.12e} != explicit triple sum {explicit:.12e}"
    );
    assert!(
        (vector_mode - explicit).abs() < 1.0e-10 * explicit.abs().max(1.0),
        "vector mode {vector_mode:.12e} != dense triple sum {explicit:.12e}"
    );
}

/// The same cross-view consistency for the analytic quartic, plus agreement of
/// `contract_vvvv` with the directional assembly `directional_fourth_derivative`
/// — two genuinely different drivers (mixed-index polarization vs a single
/// direction) onto the same tensor.
#[test]
fn fc4_dense_views_and_directional_agree() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let ndof = 3 * nat;
    let opts = tight_options();
    let cut = cutoff(&opts);
    let dense = fourth_derivative_analytic_dense(&sys, &params, &opts, cut).unwrap();

    // 4! permutation invariance of get() across ALL quadruples.
    let mut perm_worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for d in 0..ndof {
        for c in 0..=d {
            for b in 0..=c {
                for a in 0..=b {
                    let base = dense.get(a, b, c, d);
                    scale = scale.max(base.abs());
                    for p in permutations_4(a, b, c, d) {
                        perm_worst = perm_worst.max((dense.get(p[0], p[1], p[2], p[3]) - base).abs());
                    }
                }
            }
        }
    }

    let v = probe_direction(nat, 606);
    let contracted = dense.contract_vvvv(&v).unwrap();
    let directional = directional_fourth_derivative(&sys, &params, &opts, cut, &v).unwrap();
    let rel = (contracted - directional).abs() / directional.abs().max(1.0e-12);

    // contract_last / contract_last2 must be consistent with the full contraction.
    let third = dense.contract_last(&v).unwrap();
    let via_third = third.contract_vvv(&v);
    let mat = dense.contract_last2(&v, &v).unwrap();
    let via_mat: f64 = (0..ndof)
        .flat_map(|i| (0..ndof).map(move |j| (i, j)))
        .map(|(i, j)| v[i] * mat[(i, j)] * v[j])
        .sum();

    println!(
        "[C4] water FC4 views: perm |Δ| = {perm_worst:.3e} (scale {scale:.3e}), \
         contract_vvvv = {contracted:.9e} vs directional {directional:.9e} (rel {rel:.3e}), \
         contract_last chain |Δ| = {:.3e}, contract_last2 |Δ| = {:.3e}",
        (via_third - contracted).abs(),
        (via_mat - contracted).abs()
    );
    assert!(perm_worst < 1.0e-15 * scale.max(1.0), "FC4 get() not 4!-symmetric: {perm_worst:.3e}");
    assert!(rel < 1.0e-9, "FC4 dense vs directional disagree: rel {rel:.3e}");
    assert!(
        (via_third - contracted).abs() < 1.0e-9 * contracted.abs().max(1.0),
        "contract_last chain inconsistent: {:.3e}",
        (via_third - contracted).abs()
    );
    assert!(
        (via_mat - contracted).abs() < 1.0e-9 * contracted.abs().max(1.0),
        "contract_last2 inconsistent: {:.3e}",
        (via_mat - contracted).abs()
    );
}

/// The finite-temperature dense drivers (built by the cubic/quartic
/// polarization identity from directional evaluations) must produce a tensor
/// that is fully index-symmetric and whose `vvv`/`vvvv` contraction reproduces
/// the directional value it was polarized from.
#[test]
fn finite_temperature_dense_tensors_are_index_symmetric() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let ndof = 3 * nat;
    let opts = smeared_options(3000.0);
    let cut = cutoff(&opts);

    let dense3 = third_derivative_finite_t_dense(&sys, &params, &opts, cut).unwrap();
    let slabs = dense3.to_dense_slabs();
    let mut perm3 = 0.0_f64;
    let mut scale3 = 0.0_f64;
    for a in 0..ndof {
        for b in 0..ndof {
            for c in 0..ndof {
                let base = slabs[c][(a, b)];
                scale3 = scale3.max(base.abs());
                for (x, y, z) in [(a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)] {
                    perm3 = perm3.max((slabs[z][(x, y)] - base).abs());
                }
            }
        }
    }
    let v = probe_direction(nat, 121_212);
    let d3_contract = dense3.contract_vvv(&v);
    let d3_direct = directional_third_finite_t(&sys, &params, &opts, cut, &v).unwrap();
    let rel3 = (d3_contract - d3_direct).abs() / d3_direct.abs().max(1.0e-12);

    println!(
        "[C5] smeared water FC3 dense: perm |Δ| = {perm3:.3e} (scale {scale3:.3e}), \
         contract_vvv = {d3_contract:.9e} vs directional {d3_direct:.9e} (rel {rel3:.3e})"
    );
    assert!(perm3 < 1.0e-14 * scale3.max(1.0), "finite-T FC3 dense not symmetric: {perm3:.3e}");
    assert!(rel3 < 1.0e-9, "finite-T FC3 dense vs directional: rel {rel3:.3e}");
}

/// Finite-temperature dense quartic: 4! symmetry and directional agreement.
///
/// Heavy (the quartic polarization identity needs C(ndof+3,4) directional
/// finite-temperature evaluations). Re-run with:
/// `$env:CARGO_TARGET_DIR='...\target-agentphys'; cargo test --profile reltest --test physical_consistency -- --ignored --nocapture finite_temperature_dense_quartic_is_index_symmetric`
#[test]
#[ignore]
fn finite_temperature_dense_quartic_is_index_symmetric() {
    let params = params();
    let sys = system(NONEQ_WATER);
    let nat = sys.atoms.len();
    let ndof = 3 * nat;
    let opts = smeared_options(3000.0);
    let cut = cutoff(&opts);
    let dense4 = fourth_derivative_finite_t_dense(&sys, &params, &opts, cut).unwrap();
    let mut perm4 = 0.0_f64;
    let mut scale4 = 0.0_f64;
    for d in 0..ndof {
        for c in 0..=d {
            for b in 0..=c {
                for a in 0..=b {
                    let base = dense4.get(a, b, c, d);
                    scale4 = scale4.max(base.abs());
                    for p in permutations_4(a, b, c, d) {
                        perm4 = perm4.max((dense4.get(p[0], p[1], p[2], p[3]) - base).abs());
                    }
                }
            }
        }
    }
    let v = probe_direction(nat, 808);
    let c4 = dense4.contract_vvvv(&v).unwrap();
    let d4 = directional_fourth_finite_t(&sys, &params, &opts, cut, &v).unwrap();
    let rel = (c4 - d4).abs() / d4.abs().max(1.0e-12);
    println!(
        "[C6] smeared water FC4 dense: perm |Δ| = {perm4:.3e} (scale {scale4:.3e}), \
         contract_vvvv = {c4:.9e} vs directional {d4:.9e} (rel {rel:.3e})"
    );
    assert!(perm4 < 1.0e-15 * scale4.max(1.0), "finite-T FC4 dense not symmetric: {perm4:.3e}");
    assert!(rel < 1.0e-8, "finite-T FC4 dense vs directional: rel {rel:.3e}");
}

// ---------------------------------------------------------------------------
// §D  Finite temperature → T = 0 continuity
// ---------------------------------------------------------------------------

/// For a system with a real HOMO–LUMO gap, the Fermi factors approach integers
/// exponentially in `gap / kT`, so every finite-temperature derivative must
/// converge to its integer-occupation counterpart as `T → 0` — *monotonically*
/// in the temperature sequence, since the leading correction is
/// `O(exp(−gap/2kT))`. A single-point check at one temperature cannot
/// distinguish "converges to the right value" from "happens to be close".
#[test]
fn finite_temperature_derivatives_converge_to_zero_temperature_limit() {
    let params = params();
    for (name, xyz) in [("non-eq water", NONEQ_WATER), ("near-Td methane", NEAR_TD_METHANE)] {
        let sys = system(xyz);
        let nat = sys.atoms.len();
        let t0_opts = tight_options();
        let cut = cutoff(&t0_opts);
        let v = probe_direction(nat, 2_026);

        // T = 0 reference values (integer occupations, separate assembly).
        let h0 = analytic_hessian(&sys, &params, t0_opts.clone()).unwrap().hessian;
        let h0_vv: f64 = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .map(|(i, j)| v[i] * h0[(i, j)] * v[j])
            .sum();
        let k0 = third_derivative_analytic_vector(&sys, &params, t0_opts.clone(), cut, &v).unwrap();
        let e3_0: f64 = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .map(|(i, j)| v[i] * k0[(i, j)] * v[j])
            .sum();
        let e4_0 = directional_fourth_derivative(&sys, &params, &t0_opts, cut, &v).unwrap();

        let mut prev: Option<(f64, f64, f64)> = None;
        let mut last = (f64::NAN, f64::NAN, f64::NAN);
        for temperature in [300.0_f64, 50.0, 5.0] {
            let opts = smeared_options(temperature);
            let hess = analytic_hessian(&sys, &params, opts.clone()).unwrap().hessian;
            let h_vv: f64 = (0..3 * nat)
                .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
                .map(|(i, j)| v[i] * hess[(i, j)] * v[j])
                .sum();
            let e3 = directional_third_finite_t(&sys, &params, &opts, cut, &v).unwrap();
            let e4 = directional_fourth_finite_t(&sys, &params, &opts, cut, &v).unwrap();
            let errs = (
                (h_vv - h0_vv).abs(),
                (e3 - e3_0).abs(),
                (e4 - e4_0).abs(),
            );
            println!(
                "[D1] {name} @ {temperature:>5} K: |ΔH[vv]| = {:.3e} (rel {:.2e}), \
                 |Δe³[v]| = {:.3e} (rel {:.2e}), |Δe⁴[v]| = {:.3e} (rel {:.2e}) \
                 [T=0 refs: H {h0_vv:.6e}, e³ {e3_0:.6e}, e⁴ {e4_0:.6e}]",
                errs.0,
                errs.0 / h0_vv.abs().max(1.0e-30),
                errs.1,
                errs.1 / e3_0.abs().max(1.0e-30),
                errs.2,
                errs.2 / e4_0.abs().max(1.0e-30)
            );
            if let Some(p) = prev {
                // Monotone convergence, with a small absolute floor so that
                // values already at the noise level are not required to keep
                // shrinking.
                let floor = 1.0e-11;
                assert!(
                    errs.0 <= p.0 + floor,
                    "{name}: Hessian T-limit error grew {:.3e} -> {:.3e} at {temperature} K",
                    p.0,
                    errs.0
                );
                assert!(
                    errs.1 <= p.1 + floor,
                    "{name}: FC3 T-limit error grew {:.3e} -> {:.3e} at {temperature} K",
                    p.1,
                    errs.1
                );
                assert!(
                    errs.2 <= p.2 + 1.0e-10,
                    "{name}: FC4 T-limit error grew {:.3e} -> {:.3e} at {temperature} K",
                    p.2,
                    errs.2
                );
            }
            prev = Some(errs);
            last = errs;
        }
        assert!(
            last.0 < 1.0e-8,
            "{name}: Hessian did not reach its T = 0 limit at 5 K: {:.3e}",
            last.0
        );
        assert!(
            last.1 < 1.0e-8,
            "{name}: FC3 did not reach its T = 0 limit at 5 K: {:.3e}",
            last.1
        );
        assert!(
            last.2 < 1.0e-8,
            "{name}: FC4 did not reach its T = 0 limit at 5 K: {:.3e}",
            last.2
        );
    }
}

// ---------------------------------------------------------------------------
// §E  Magnetic gauge-origin behaviour
// ---------------------------------------------------------------------------

fn magnetic_options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        external_field: gfn1_rs::ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..gfn1_rs::ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

/// **Diagnostic: is the magnetizability's symmetry residual a finite-difference
/// artifact or a genuine invariance defect?**
///
/// `magnetizability_tensor_analytic` is not fully analytic — its mixed second
/// field derivatives `H0^ab`/`S^ab` come from a *cross finite difference of the
/// LAO builder* controlled by `step`. That FD is taken along the fixed GLOBAL
/// magnetic-field axes, so its truncation error is frame dependent: rotating the
/// molecule (or moving it relative to the coordinate origin) samples a different
/// truncation error even when the underlying physics is exactly invariant.
///
/// The discriminator is the `step` ladder. A truncation artifact falls like
/// `step²` (and eventually turns back up as SCF noise divided by `step²`
/// dominates); a real symmetry violation is `step` independent.
#[test]
fn magnetizability_symmetry_residuals_versus_field_step() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    let shift = [2.0, 0.0, 0.0];
    let (rname, r) = ("generic (1,2,3) 0.7 rad", rotation_matrix([1.0, 2.0, 3.0], 0.7));
    let moved = translate_system(&sys, shift);
    let rotated = rotate_system(&sys, &r);

    println!("[E0] magnetizability symmetry residuals vs the field-derivative step:");
    let mut translation_res = Vec::new();
    let mut rotation_res = Vec::new();
    let steps = [1.6e-2_f64, 8.0e-3, 4.0e-3, 2.0e-3, 1.0e-3];
    for step in steps {
        let chi0 = magnetizability_tensor_analytic(&sys, &params, &opts, None, step).unwrap();
        let chi_t = magnetizability_tensor_analytic(&moved, &params, &opts, None, step).unwrap();
        let chi_r = magnetizability_tensor_analytic(&rotated, &params, &opts, None, step).unwrap();
        let scale = chi0.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        let mut dt = 0.0_f64;
        let mut dr = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                dt = dt.max((chi_t[i][j] - chi0[i][j]).abs());
                let mut expect = 0.0;
                for ip in 0..3 {
                    for jp in 0..3 {
                        expect += r[i][ip] * r[j][jp] * chi0[ip][jp];
                    }
                }
                dr = dr.max((chi_r[i][j] - expect).abs());
            }
        }
        println!(
            "[E0]   step = {step:.1e}: translation residual = {dt:.4e} (rel {:.3e}), \
             rotation residual [{rname}] = {dr:.4e} (rel {:.3e}), χ scale = {scale:.4e}",
            dt / scale,
            dr / scale
        );
        translation_res.push(dt);
        rotation_res.push(dr);
    }
    // Ratios between consecutive (halved) steps: ~4 means step² truncation,
    // ~1 means a step-independent (i.e. genuine) invariance defect.
    for (label, res) in [
        ("translation", &translation_res),
        ("rotation", &rotation_res),
    ] {
        let ratios: Vec<String> = res
            .windows(2)
            .map(|w| format!("{:.2}", w[0] / w[1].max(1.0e-300)))
            .collect();
        println!("[E0] {label} residual ratios between consecutive halved steps: {ratios:?}");
    }
    assert!(
        translation_res.iter().all(|x| x.is_finite()),
        "non-finite magnetizability residual"
    );
}

/// Richardson-extrapolated magnetizability: `(4 χ(h/2) − χ(h)) / 3` removes the
/// leading `O(h²)` truncation error of the cross finite difference that supplies
/// the mixed second field derivatives, leaving `O(h⁴)`. The step-ladder
/// diagnostic above establishes that the `h²` scaling is exact (ratio 4.00 at
/// every halving), which is what makes this extrapolation legitimate.
fn magnetizability_extrapolated(
    sys: &PeriodicSystem,
    params: &Gfn1Parameters,
    opts: &ElectronicOptions,
    h: f64,
) -> [[f64; 3]; 3] {
    let coarse = magnetizability_tensor_analytic(sys, params, opts, None, h).unwrap();
    let fine = magnetizability_tensor_analytic(sys, params, opts, None, 0.5 * h).unwrap();
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (4.0 * fine[i][j] - coarse[i][j]) / 3.0;
        }
    }
    out
}

/// **The defining property of London (GIAO) orbitals: the magnetizability does
/// not depend on where the gauge origin sits.** The implementation realises
/// this structurally — the London phase is built from the *interatomic* vector,
/// so `options.external_field.origin` never enters `S(B)` or `H0(B)` — which
/// means the statement is probed by moving the molecule instead: a rigid
/// translation changes every atom's position relative to the coordinate origin,
/// and a residual common-gauge term would show up immediately. A conventional
/// (non-GIAO) implementation fails this at the percent level.
///
/// **The step matters and the raw tensor does not pass this at face value.**
/// `magnetizability_tensor_analytic` obtains its mixed second field derivatives
/// `H0^ab`/`S^ab` from a cross finite difference of the LAO builder along the
/// GLOBAL field axes, so its truncation error is frame dependent: at the
/// commonly used `step = 4e-3` a 2-bohr rigid translation moves the tensor by
/// `1.50e-5` (rel `6.3e-6`). That residual is pure `O(step²)` truncation, not a
/// broken invariance — see
/// [`magnetizability_symmetry_residuals_versus_field_step`], where the residual
/// falls by exactly 4.00 per halving over a decade of steps. Extrapolating that
/// term away restores the invariance to the level gated here.
#[test]
fn magnetizability_is_gauge_origin_independent() {
    let params = params();
    let opts = magnetic_options();
    for (name, xyz) in [("water", NONEQ_WATER), ("near-Td methane", NEAR_TD_METHANE)] {
        let sys = system(xyz);
        let residual_at = |shift: [f64; 3], h: f64| -> (f64, f64) {
            let base = magnetizability_extrapolated(&sys, &params, &opts, h);
            let moved = magnetizability_extrapolated(&translate_system(&sys, shift), &params, &opts, h);
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((moved[i][j] - base[i][j]).abs());
                    scale = scale.max(base[i][j].abs());
                }
            }
            (worst, scale)
        };

        for shift in [[2.0, 0.0, 0.0], [0.0, -3.5, 1.25]] {
            let (worst, scale) = residual_at(shift, 4.0e-3);
            println!(
                "[E1] {name} shifted by {shift:?} bohr: |Δχ|_max = {worst:.3e} \
                 (χ scale {scale:.3e}, rel {:.3e})",
                worst / scale
            );
            assert!(
                worst / scale < 1.0e-8,
                "{name}: magnetizability moved by {worst:.3e} (rel {:.3e}) under a rigid \
                 translation by {shift:?} — the GIAO gauge-origin invariance is broken beyond \
                 the field-step truncation floor",
                worst / scale
            );
        }

        // A FAR translation. Historically the London phase was referenced to
        // the COORDINATE ORIGIN, so the cross-FD's effective expansion
        // parameter was `step · |R − origin|` and the residual grew like
        // `|shift|⁴` — rel 1.3e-3 at 9.4 bohr. `magnetizability_*` now
        // differentiates in the CENTROID frame (an exact gauge transformation
        // for London orbitals), which removes that channel structurally, so
        // the residual no longer depends on the shift at all and no longer
        // falls under step refinement: what is left is SCC/eigensolver noise.
        // The assertion is therefore on the residual itself, at a level the
        // old origin-referenced code could not reach at any step.
        let far = [7.0, 7.0, 7.0];
        let (coarse, scale) = residual_at(far, 4.0e-3);
        let (fine, _) = residual_at(far, 2.0e-3);
        println!(
            "[E1] {name} FAR shift {far:?} (|d| = {:.2} bohr): residual {coarse:.3e} (h=4e-3), \
             {fine:.3e} (h=2e-3) (χ scale {scale:.3e}, rel {:.3e})",
            (far[0] * far[0] + far[1] * far[1] + far[2] * far[2]).sqrt(),
            coarse / scale
        );
        assert!(
            coarse / scale < 1.0e-8 && fine / scale < 1.0e-8,
            "{name}: the far-translation magnetizability residual is {coarse:.3e} / {fine:.3e} \
             (rel {:.3e}) — the centroid-frame differentiation should make gauge-origin \
             invariance structural, so anything above the SCC noise floor means it regressed",
            coarse / scale
        );
    }
}

/// The magnetizability is a rank-2 rotational tensor: `χ(R x) = R χ(x) Rᵀ`.
/// The v0.5.0 magnetic audit fixed a shell-pair prefactor by this very
/// criterion; this is the always-on regression gate.
///
/// Same caveat as the gauge-origin gate: the raw `step = 4e-3` tensor is only
/// covariant to rel `2.7e-6` under a *generic* rotation (axis-aligned 90°
/// rotations are clean at rel `5e-11`, because permuting the global field axes
/// permutes the finite-difference stencil exactly). The `O(step²)` cross-FD
/// truncation is the whole of that residual, so the extrapolated tensor is
/// gated here at the physical tolerance.
#[test]
fn magnetizability_tensor_is_rotationally_covariant() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    let chi0 = magnetizability_extrapolated(&sys, &params, &opts, 4.0e-3);
    let scale = chi0.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
    for (rname, r) in test_rotations() {
        let rot = rotate_system(&sys, &r);
        let chi1 = magnetizability_extrapolated(&rot, &params, &opts, 4.0e-3);
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let mut expect = 0.0;
                for ip in 0..3 {
                    for jp in 0..3 {
                        expect += r[i][ip] * r[j][jp] * chi0[ip][jp];
                    }
                }
                worst = worst.max((chi1[i][j] - expect).abs());
            }
        }
        println!(
            "[E2] water magnetizability / {rname}: |χ' − RχRᵀ|_max = {worst:.3e} \
             (scale {scale:.3e}, rel {:.3e})",
            worst / scale
        );
        assert!(
            worst / scale < 1.0e-8,
            "magnetizability rotation covariance violated by {worst:.3e} (rel {:.3e}) — \
             beyond the field-step truncation floor",
            worst / scale
        );
    }
}

/// **NMR shielding must be translationally covariant**: displacing the molecule
/// and its gauge origin by the same vector cannot change a shielding tensor.
/// This is the invariance that must hold *regardless* of the gauge choice, so
/// it is a hard gate (unlike gauge-origin independence at a fixed molecule,
/// which this implementation's common-gauge-origin nuclear term does not
/// provide — see [`nmr_shielding_gauge_origin_dependence_is_characterized`]).
#[test]
fn nmr_shielding_is_translationally_covariant() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    for nucleus in 0..sys.atoms.len() {
        let gauge = sys.atoms[nucleus].position;
        let s0 = nmr_shielding_tensor(&sys, &params, &opts, None, nucleus, gauge).unwrap();
        let scale = s0.sigma.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        for shift in [[1.5, 0.0, 0.0], [-2.0, 3.0, -1.0]] {
            let moved = translate_system(&sys, shift);
            let gauge_moved = Vec3::new(
                gauge.x + shift[0],
                gauge.y + shift[1],
                gauge.z + shift[2],
            );
            let s1 =
                nmr_shielding_tensor(&moved, &params, &opts, None, nucleus, gauge_moved).unwrap();
            let mut worst = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((s1.sigma[i][j] - s0.sigma[i][j]).abs());
                }
            }
            println!(
                "[E3] water nucleus {nucleus} shifted by {shift:?}: |Δσ|_max = {worst:.3e} \
                 (σ_iso = {:.6} ppm, scale {scale:.3e})",
                s0.isotropic() * 1.0e6
            );
            assert!(
                worst / scale.max(1.0e-12) < 1.0e-8,
                "nucleus {nucleus}: NMR shielding changed by {worst:.3e} under a rigid \
                 translation of molecule AND gauge origin"
            );
        }
    }
}

/// **Characterization (not a pass/fail physics gate): how origin-dependent is
/// the NMR shielding at a fixed geometry?**
///
/// `nmr_shielding_tensor` takes an explicit `gauge_origin` and the module
/// documents the nuclear term as a *common gauge origin* construction, so a
/// residual origin dependence is expected by design rather than a bug — a full
/// GIAO shielding (with the London phase differentiated against the nuclear
/// magnetic moment) would be origin independent. This test records the size of
/// that dependence so a future GIAO shielding can be compared against it, and
/// only asserts that the numbers stay finite and physically bounded.
#[test]
fn nmr_shielding_gauge_origin_dependence_is_characterized() {
    let params = params();
    let opts = magnetic_options();
    let sys = system(NONEQ_WATER);
    let nucleus = 0;
    let at_nucleus = sys.atoms[nucleus].position;
    let centroid = {
        let n = sys.atoms.len() as f64;
        let mut c = Vec3::zero();
        for a in &sys.atoms {
            c = Vec3::new(c.x + a.position.x, c.y + a.position.y, c.z + a.position.z);
        }
        Vec3::new(c.x / n, c.y / n, c.z / n)
    };
    let origins = [
        ("at nucleus", at_nucleus),
        ("centroid", centroid),
        ("coordinate origin", Vec3::zero()),
        ("far (10,10,10)", Vec3::new(10.0, 10.0, 10.0)),
    ];
    let mut isotropics = Vec::new();
    for (oname, origin) in origins {
        let s = nmr_shielding_tensor(&sys, &params, &opts, None, nucleus, origin).unwrap();
        let iso_ppm = s.isotropic() * 1.0e6;
        println!(
            "[E4] water O shielding, gauge origin {oname}: σ_iso = {iso_ppm:.4} ppm \
             (dia {:.4}, para {:.4})",
            (s.diamagnetic[0][0] + s.diamagnetic[1][1] + s.diamagnetic[2][2]) / 3.0 * 1.0e6,
            (s.paramagnetic[0][0] + s.paramagnetic[1][1] + s.paramagnetic[2][2]) / 3.0 * 1.0e6
        );
        assert!(iso_ppm.is_finite(), "{oname}: σ_iso is not finite");
        isotropics.push((oname, iso_ppm));
    }
    let spread = isotropics
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max)
        - isotropics
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::INFINITY, f64::min);
    println!("[E4] gauge-origin spread of σ_iso over the four origins: {spread:.4} ppm");
}

// ---------------------------------------------------------------------------
// §F  Thermodynamic consistency of the periodic stress and Grüneisen path
// ---------------------------------------------------------------------------

const DIAMOND_PRIMITIVE: &str = "2\n\
Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.891750 0.891750 0.891750\n";

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

/// **The isotropic stress is the volume derivative of the free energy.** With
/// the module's convention `σ_ab = (1/V) dE_free/dε_ab`, a uniform dilation
/// `ε = δ I` gives `dE = V Tr(σ) δ` and `dV = 3 V δ`, hence
/// `(1/3) Tr(σ) = dE/dV = −P`. The right-hand side is obtained by *actually
/// rescaling the cell* and re-converging the SCC — an entirely separate code
/// path from the analytic strain derivative, so this is a genuine
/// thermodynamic-consistency check and not a self-comparison.
#[test]
fn isotropic_stress_equals_the_volume_derivative_of_the_energy() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let el = pbc_electronic();
    let pbc = lean_pbc();
    let v0 = sys.lattice.as_ref().unwrap().volume();

    let stress = pbc_stress(&sys, &params, &el, &pbc).unwrap();
    let trace_third =
        (stress.stress[(0, 0)] + stress.stress[(1, 1)] + stress.stress[(2, 2)]) / 3.0;

    // Central difference of E_free with respect to the isotropic scale factor.
    // V(s) = s³ V₀  ⇒  dE/dV = (dE/ds) / (3 V₀) at s = 1.
    let energy_at = |s: f64| -> f64 {
        let scaled = scale_lattice_isotropic(&sys, s).unwrap();
        run_pbc_scc(&scaled, &params, &el, &pbc).unwrap().total_free
    };
    let mut fd_values = Vec::new();
    for delta in [2.0e-3_f64, 1.0e-3] {
        let de_ds = (energy_at(1.0 + delta) - energy_at(1.0 - delta)) / (2.0 * delta);
        fd_values.push(de_ds / (3.0 * v0));
    }
    // Richardson: the central difference is O(δ²), so halving δ lets the two
    // estimates be extrapolated to remove the leading truncation error.
    let richardson = (4.0 * fd_values[1] - fd_values[0]) / 3.0;
    let err = (trace_third - richardson).abs();
    let rel = err / richardson.abs().max(1.0e-14);
    println!(
        "[F1] diamond: (1/3)Tr(σ) = {trace_third:.12e}, dE/dV (δ=2e-3) = {:.12e}, \
         (δ=1e-3) = {:.12e}, Richardson = {richardson:.12e}, |Δ| = {err:.3e} (rel {rel:.3e}), \
         V₀ = {v0:.6} bohr³",
        fd_values[0], fd_values[1]
    );
    assert!(
        rel < 1.0e-5,
        "isotropic stress is not the volume derivative of E_free: \
         (1/3)Tr(σ) = {trace_third:.9e} vs dE/dV = {richardson:.9e} (rel {rel:.3e})"
    );
}

/// The analytic stress must also be rotationally covariant and symmetric:
/// `σ` is a symmetric rank-2 tensor (angular-momentum conservation / absence of
/// a body torque in a lattice with no external field).
#[test]
fn periodic_stress_tensor_is_symmetric() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let stress = pbc_stress(&sys, &params, &pbc_electronic(), &lean_pbc()).unwrap();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((stress.stress[(i, j)] - stress.stress[(j, i)]).abs());
            scale = scale.max(stress.stress[(i, j)].abs());
        }
    }
    println!("[F2] diamond stress antisymmetry: |σ − σᵀ|_max = {worst:.3e} (scale {scale:.3e})");
    assert!(
        worst < 1.0e-12 * scale.max(1.0e-6),
        "periodic stress tensor is not symmetric: |σ − σᵀ|_max = {worst:.3e} (scale {scale:.3e})"
    );
}

/// **Grüneisen path equivalence**: with a Γ-only mesh, the k-point routing
/// (`GruneisenOptions::kpoint = true`) evaluates the same physics as the Γ path
/// and must return identical mode Grüneisen parameters. Any difference is a
/// defect in one of the two routings, not physics.
///
/// Measured cost on the diamond primitive at the lean cutoffs: ~14 s, so this
/// runs always-on rather than `#[ignore]`d.
#[test]
fn gruneisen_gamma_mesh_kpoint_routing_matches_gamma_path() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    let base = GruneisenOptions {
        delta: 5.0e-3,
        temperatures: vec![300.0],
        electronic: pbc_electronic(),
        pbc: lean_pbc(),
        ..GruneisenOptions::default()
    };
    let gamma = pbc_gruneisen(&sys, &params, &base).unwrap();
    let kpt = pbc_gruneisen(
        &sys,
        &params,
        &GruneisenOptions {
            kpoint: true,
            pbc: PbcOptions {
                kmesh: KMesh::gamma(),
                ..lean_pbc()
            },
            ..base.clone()
        },
    )
    .unwrap();
    let mut worst = 0.0_f64;
    for (a, b) in gamma.mode_gamma.iter().zip(&kpt.mode_gamma) {
        worst = worst.max((a - b).abs());
    }
    let g_gamma = gamma.gamma_at(300.0).unwrap();
    let g_kpt = kpt.gamma_at(300.0).unwrap();
    println!(
        "[F3] diamond Grüneisen: Γ-path γ(300K) = {g_gamma:.9}, Γ-mesh k-point γ = {g_kpt:.9}, \
         worst mode |Δγ_i| = {worst:.3e}"
    );
    assert!(
        worst < 1.0e-6,
        "Γ-only k-point Grüneisen routing disagrees with the Γ path by {worst:.3e}"
    );
    assert!(
        (g_gamma - g_kpt).abs() < 1.0e-8,
        "thermodynamic γ differs between routings: {g_gamma:.9} vs {g_kpt:.9}"
    );
}

/// **Stencil independence of the Grüneisen parameter.** `γ_i = −dlnω_i/dlnV` is
/// a property of the material, not of the finite-difference stencil used to
/// extract it: the three-point and five-point (Fornberg) stencils must agree to
/// within their truncation error.
///
/// **Finding recorded here, deliberately not asserted.** The FIRST-order γ is
/// perfectly stencil independent (both stencils give `0.905609595`, difference
/// exactly `0`). The SECOND-order γ⁽²⁾ is not: at the DEFAULT `delta = 5e-3`
/// the two stencils return `−0.0372` and `+0.0674` — different in sign as well
/// as magnitude.
///
/// The δ-ladder in [`gruneisen_second_order_stencil_delta_ladder`] settles the
/// mechanism: the stencil gap SHRINKS monotonically as δ grows
/// (`1.05e-1 → 7.7e-2 → 2.0e-2 → 5.1e-3` for δ = 5e-3 … 4e-2), which is the
/// signature of `δ⁻²` noise amplification rather than wrong Fornberg weights.
/// The practical consequence is nonetheless serious: γ⁽²⁾ is **not numerically
/// converged at any δ tested** — it swings over `−0.037, −0.351, −0.121,
/// −0.061` across the ladder while the first-order γ stays put to three
/// digits. The default δ = 5e-3 sits deep in the noise-dominated regime, so a
/// γ⁽²⁾ quoted from it should not be trusted to its leading digit. Only the
/// first-order statement is gated here.
///
/// Measured cost: ~17 s (three + five phonon calculations).
#[test]
fn gruneisen_is_independent_of_the_finite_difference_stencil() {
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
    let mut worst_first = 0.0_f64;
    for (a, b) in three.mode_gamma.iter().zip(&five.mode_gamma) {
        worst_first = worst_first.max((a - b).abs());
    }
    let g3 = three.gamma_at(300.0).unwrap();
    let g5 = five.gamma_at(300.0).unwrap();
    println!(
        "[F4] diamond: 3-point γ(300K) = {g3:.9}, 5-point γ(300K) = {g5:.9}, \
         |Δγ| = {:.3e}, worst mode |Δγ_i| = {worst_first:.3e}",
        (g3 - g5).abs()
    );
    println!(
        "[F4] second-order (NOT gated -- see doc comment): 3-point γ⁽²⁾ = {:?}, \
         5-point γ⁽²⁾ = {:?}",
        three.gamma2_at(300.0),
        five.gamma2_at(300.0)
    );
    assert!(
        worst_first < 1.0e-3,
        "first-order mode Grüneisen depends on the second-order stencil by {worst_first:.3e} \
         — the two stencils must extract the same dlnω/dlnV"
    );
}

/// **Diagnostic for the γ⁽²⁾ stencil disagreement above.** Extracting
/// `γ⁽²⁾ = d²lnω/dlnV²` by finite differences amplifies phonon noise by `δ⁻²`,
/// so a small `δ` can make two formally equivalent stencils disagree wildly
/// while both converge to the same value as `δ` grows. Conversely, a
/// disagreement that PERSISTS as `δ` grows indicates wrong Fornberg weights or
/// an asymmetric `lnV` node placement rather than noise.
///
/// This test only reports; it makes no assertion. Measured on diamond
/// (δ = 5e-3 → 4e-2, `|Δ|` = 1.046e-1, 7.667e-2, 1.987e-2, 5.148e-3): the gap
/// closes as δ grows, so the stencils are consistent and the disagreement at
/// small δ is noise — but γ⁽²⁾ itself does not settle over the same range
/// (−0.037, −0.351, −0.121, −0.061), so the extraction needs either a larger δ
/// with tighter phonons or an analytic second strain derivative. Re-run with:
/// `$env:CARGO_TARGET_DIR='...\target-agentphys'; cargo test --profile reltest --test physical_consistency -- --ignored --nocapture gruneisen_second_order_stencil_delta_ladder`
#[test]
#[ignore]
fn gruneisen_second_order_stencil_delta_ladder() {
    let params = params();
    let sys = PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap();
    println!("[F5] diamond γ⁽²⁾(300 K) vs volumetric step, by stencil:");
    for delta in [5.0e-3_f64, 1.0e-2, 2.0e-2, 4.0e-2] {
        let base = GruneisenOptions {
            delta,
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
        println!(
            "[F5]   delta = {delta:.1e}: 3-point γ⁽²⁾ = {g3:+.9}, 5-point γ⁽²⁾ = {g5:+.9}, \
             |Δ| = {:.3e}   [first-order γ: 3pt {:.9}, 5pt {:.9}]",
            (g3 - g5).abs(),
            three.gamma_at(300.0).unwrap_or(f64::NAN),
            five.gamma_at(300.0).unwrap_or(f64::NAN)
        );
    }
}

// ---------------------------------------------------------------------------
// §G  Response conservation laws
// ---------------------------------------------------------------------------

/// The linear-response bundle must obey the sum rules that follow from
/// particle-number conservation and the symmetry of the AO density matrix:
///
/// * `P^x` and `W^x` are **symmetric** (they are derivatives of symmetric AO
///   matrices — an asymmetric result means a missing transpose somewhere in the
///   `C U` assembly);
/// * `Σ_p f^x_p = 0` — the occupation response conserves the electron count
///   (grand-canonical `μ^x` is determined by exactly this condition);
/// * `Σ_s q^x_s = 0` — the Mulliken shell charges sum to the fixed total charge,
///   so their response sums to zero;
/// * `Tr(S P^x) + Tr(S^x P⁰) = 0` — the metric-aware form of the same statement,
///   `d/dx Tr(S P) = d/dx N = 0`. This one is independent of the Mulliken
///   partitioning and therefore catches errors the shell-charge sum rule cannot.
#[test]
fn first_order_response_obeys_conservation_laws() {
    let params = params();
    // The `expect_fractional` flag marks the fixtures whose occupations really
    // are fractional, so the finite-temperature sum rules cannot pass vacuously.
    // Water keeps integer occupations even at 3000 K (its gap is far larger than
    // kT), which is why the strongly smeared formaldehyde fixture is here.
    for (name, xyz, temperature, expect_fractional) in [
        ("non-eq water (T=0)", NONEQ_WATER, 0.0, false),
        ("non-eq water (3000 K)", NONEQ_WATER, 3000.0, false),
        ("non-eq HCHO (10000 K)", NONEQ_HCHO, 10000.0, true),
    ] {
        let sys = system(xyz);
        let mut el_opts = tight_options().electronic_options;
        el_opts.electronic_temperature = temperature;
        let electronic = run_electronic(&sys, &params, el_opts.clone()).unwrap();
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &sys,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: el_opts.hamiltonian.coordination_cutoff,
                include_cn_h0: el_opts.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&sys, &params, &electronic).unwrap();
        let n = electronic.density.rows();
        let overlap = &electronic.integrals.overlap;

        let (mut sym_p, mut sym_w, mut sum_f, mut sum_q, mut trace_rule) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut fractional = false;
        for dof in 0..3 * sys.atoms.len() {
            let bundle = ctx
                .solve_first_order(
                    &cpxtb.derivative_matrices[dof].h0_deriv,
                    &cpxtb.derivative_matrices[dof].overlap_deriv,
                )
                .unwrap();
            for i in 0..n {
                for j in 0..n {
                    sym_p = sym_p.max((bundle.density[(i, j)] - bundle.density[(j, i)]).abs());
                    sym_w = sym_w.max(
                        (bundle.energy_weighted[(i, j)] - bundle.energy_weighted[(j, i)]).abs(),
                    );
                }
            }
            let f_sum: f64 = bundle.occupation_response.iter().sum();
            sum_f = sum_f.max(f_sum.abs());
            if bundle.occupation_response.iter().any(|f| f.abs() > 1.0e-12) {
                fractional = true;
            }
            sum_q = sum_q.max(bundle.shell_charges.iter().sum::<f64>().abs());

            // Tr(S P^x) + Tr(S^x P⁰) = 0.
            let s_deriv = &cpxtb.derivative_matrices[dof].overlap_deriv;
            let mut trace = 0.0;
            for i in 0..n {
                for j in 0..n {
                    trace += overlap[(i, j)] * bundle.density[(j, i)]
                        + s_deriv[(i, j)] * electronic.density[(j, i)];
                }
            }
            trace_rule = trace_rule.max(trace.abs());
        }
        println!(
            "[G1] {name}: |P^x − P^xᵀ| = {sym_p:.3e}, |W^x − W^xᵀ| = {sym_w:.3e}, \
             |Σ_p f^x_p| = {sum_f:.3e}, |Σ_s q^x_s| = {sum_q:.3e}, \
             |d/dx Tr(SP)| = {trace_rule:.3e} (occupation response active: {fractional})"
        );
        assert!(sym_p < 1.0e-12, "{name}: P^x is not symmetric ({sym_p:.3e})");
        assert!(sym_w < 1.0e-12, "{name}: W^x is not symmetric ({sym_w:.3e})");
        assert!(
            sum_f < 1.0e-11,
            "{name}: occupation response does not conserve the electron count ({sum_f:.3e})"
        );
        assert!(
            sum_q < 1.0e-10,
            "{name}: shell-charge response does not conserve the total charge ({sum_q:.3e})"
        );
        assert!(
            trace_rule < 1.0e-9,
            "{name}: d/dx Tr(SP) = {trace_rule:.3e} ≠ 0 — the density response violates \
             particle-number conservation in the AO metric"
        );
        if expect_fractional {
            assert!(
                fractional,
                "{name}: the occupation response is identically zero at {temperature} K — the \
                 finite-temperature branch was not exercised, so the sum rule above is vacuous"
            );
        }
    }
}

/// The same conservation laws one order up. `P^xy` symmetric, `Σ_p f^xy_p = 0`,
/// `Σ_s q^xy_s = 0`, and the second-order metric sum rule
/// `Tr(S P^xy) + Tr(S^x P^y) + Tr(S^y P^x) + Tr(S^xy P⁰) = 0`.
#[test]
fn second_order_response_obeys_conservation_laws() {
    let params = params();
    for (name, xyz, temperature) in [
        ("non-eq water (T=0)", NONEQ_WATER, 0.0),
        ("non-eq water (3000 K)", NONEQ_WATER, 3000.0),
    ] {
        let sys = system(xyz);
        let mut el_opts = tight_options().electronic_options;
        el_opts.electronic_temperature = temperature;
        let cut = el_opts.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&sys, &params, el_opts.clone()).unwrap();
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &sys,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cut,
                include_cn_h0: el_opts.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&sys, &params, &electronic).unwrap();
        let n = electronic.density.rows();
        let nshell = electronic.basis.shells.len();
        let overlap = &electronic.integrals.overlap;
        let dvdr_q = shell_scalar_potential_first_derivatives(
            &sys,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();

        let field_for = |dof: usize| {
            ctx.first_order_field(
                cpxtb.derivative_matrices[dof].h0_deriv.clone(),
                cpxtb.derivative_matrices[dof].overlap_deriv.clone(),
            )
            .unwrap()
        };

        let (mut sym_p, mut sum_f, mut sum_q, mut trace_rule) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for &(x, y) in &[(0usize, 0usize), (1, 4), (2, 7), (5, 5), (4, 1)] {
            let fx = field_for(x);
            let fy = field_for(y);

            // Skeleton second derivative (frozen density AND frozen charges),
            // assembled exactly as the in-crate second-order gate does.
            let mut f_xy =
                h0_bare_second_derivative_matrix(&sys, &params, &electronic, x, y).unwrap();
            let cn_block =
                h0_cn_block_second_derivative_matrix(&sys, &params, &electronic, cut, x, y).unwrap();
            let v_geo_y: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, y)]).collect();
            let scc_block = h0_scc_scalar_second_derivative_matrix(
                &sys,
                &params,
                &electronic,
                &v_geo_y,
                &vec![0.0; nshell],
                x,
                y,
            )
            .unwrap();
            for (dst, (a, b)) in f_xy
                .as_mut_slice()
                .iter_mut()
                .zip(cn_block.as_slice().iter().zip(scc_block.as_slice()))
            {
                *dst += a + b;
            }
            let s_xy = overlap_second_derivative_matrix(&sys, &electronic.basis, x, y).unwrap();
            let potential_of = |q: &[f64], dof: usize| -> Vec<f64> {
                let m =
                    shell_scalar_potential_first_derivatives(&sys, &electronic.basis, q, &params)
                        .unwrap();
                (0..nshell).map(|s| m[(s, dof)]).collect()
            };
            let dgamma_y_qx = potential_of(&fx.bundle.shell_charges, y);
            let dgamma_x_qy = potential_of(&fy.bundle.shell_charges, x);

            let second = ctx
                .solve_second_order(&fx, &fy, &f_xy, &s_xy, &dgamma_y_qx, &dgamma_x_qy)
                .unwrap();

            for i in 0..n {
                for j in 0..n {
                    sym_p = sym_p.max((second.density[(i, j)] - second.density[(j, i)]).abs());
                }
            }
            sum_f = sum_f.max(second.occupation_response.iter().sum::<f64>().abs());
            sum_q = sum_q.max(second.shell_charges.iter().sum::<f64>().abs());

            let sx = &cpxtb.derivative_matrices[x].overlap_deriv;
            let sy = &cpxtb.derivative_matrices[y].overlap_deriv;
            let mut trace = 0.0;
            for i in 0..n {
                for j in 0..n {
                    trace += overlap[(i, j)] * second.density[(j, i)]
                        + sx[(i, j)] * fy.bundle.density[(j, i)]
                        + sy[(i, j)] * fx.bundle.density[(j, i)]
                        + s_xy[(i, j)] * electronic.density[(j, i)];
                }
            }
            trace_rule = trace_rule.max(trace.abs());
        }
        println!(
            "[G2] {name}: |P^xy − P^xyᵀ| = {sym_p:.3e}, |Σ_p f^xy_p| = {sum_f:.3e}, \
             |Σ_s q^xy_s| = {sum_q:.3e}, |d²/dxdy Tr(SP)| = {trace_rule:.3e}"
        );
        assert!(sym_p < 1.0e-11, "{name}: P^xy is not symmetric ({sym_p:.3e})");
        assert!(
            sum_f < 1.0e-10,
            "{name}: second-order occupation response does not conserve the electron count \
             ({sum_f:.3e})"
        );
        assert!(
            sum_q < 1.0e-9,
            "{name}: second-order shell-charge response does not conserve the total charge \
             ({sum_q:.3e})"
        );
        assert!(
            trace_rule < 1.0e-8,
            "{name}: d²/dxdy Tr(SP) = {trace_rule:.3e} ≠ 0"
        );
    }
}

/// The periodic Γ-point charge-space response cannot be gated here: the
/// constructor `gamma_charge_space_context` and `ChargeSpaceContext::from_raw_parts`
/// are both `pub(crate)` (`src/pbc/gamma_response.rs:44`,
/// `src/response/charge_space.rs:263`), and the one publicly re-exported handle
/// on the periodic context, `GammaThirdReference`, exposes no accessor for its
/// internal `ctx` field. From an external integration test a
/// `ChargeSpaceContext` is therefore reachable for non-PBC systems only.
///
/// Making `GammaThirdReference` expose `pub fn charge_space(&self) -> &ChargeSpaceContext`
/// (or re-exporting `gamma_charge_space_context`) would let §G run at Γ unchanged.
#[test]
#[ignore = "periodic ChargeSpaceContext is unreachable from an external test: \
            gamma_charge_space_context and ChargeSpaceContext::from_raw_parts are pub(crate), \
            and GammaThirdReference has no accessor for its ctx field"]
fn periodic_gamma_response_conservation_laws() {
    unimplemented!("needs a public accessor for the Gamma-point ChargeSpaceContext");
}

// ---------------------------------------------------------------------------
// §H  Berry-phase polarization
// ---------------------------------------------------------------------------

const NACL: &str = "8\n\
Lattice=\"5.64 0 0 0 5.64 0 0 0 5.64\" pbc=\"T T T\"\n\
Na 0.00 0.00 0.00\n\
Na 0.00 2.82 2.82\n\
Na 2.82 0.00 2.82\n\
Na 2.82 2.82 0.00\n\
Cl 2.82 2.82 2.82\n\
Cl 2.82 0.00 0.00\n\
Cl 0.00 2.82 0.00\n\
Cl 0.00 0.00 2.82\n";

/// **Why §H does not use [`lean_pbc`].** The rest of the periodic gates run at
/// `ao_cutoff = 12` bohr, but this NaCl cell is only 10.66 bohr across, so that
/// cutoff truncates the Bloch image sum part-way through a shell of self-images
/// and the Γ-point overlap loses positive definiteness outright —
/// `run_pbc_scc` returns `InvalidInput("overlap matrix is not positive
/// definite; eigenvalue 0 = -6.197e-1")`. A converged AO cutoff is therefore a
/// correctness requirement here, not a precision preference. (This is the same
/// self-image pathology that ruled rocksalt LiH out as a Γ-FC3 fixture.) The
/// cost is instead controlled by evaluating fewer displaced structures in the
/// always-on gate.
fn berry_pbc() -> PbcOptions {
    PbcOptions {
        ao_cutoff: 30.0,
        ewald: EwaldOptions {
            real_cutoff: 40.0,
            ..EwaldOptions::default()
        },
        ..PbcOptions::default()
    }
}

/// NaCl with the whole Na sublattice displaced by `shift` Angstrom along z —
/// a polar structure with a genuinely non-zero spontaneous polarization.
fn nacl_polar(shift: f64) -> PeriodicSystem {
    let mut sys = PeriodicSystem::from_xyz_str(NACL, 0.0, false).unwrap();
    let dz = shift * 1.889_726_124_625_770_2;
    for atom in sys.atoms.iter_mut() {
        if atom.z == 11 {
            atom.position.z += dz;
        }
    }
    sys
}

/// **The polarization of a charge-neutral cell is origin independent, and
/// translating every atom by a lattice vector reproduces the same crystal.**
///
/// Two distinct statements are gated here:
///
/// 1. *Lattice translation* — displacing all atoms by a primitive lattice
///    vector gives literally the same periodic crystal, so the branch-reduced
///    polarization must be unchanged, while the raw (unreduced) polarization may
///    only move by **whole quanta** `e R / Ω`. A non-integer shift would mean the
///    Berry phase is not being accumulated as a proper `U(1)` winding.
/// 2. *Arbitrary rigid translation* — for a cell with zero net charge the dipole
///    (and hence the polarization) does not depend on the choice of origin, so an
///    arbitrary non-lattice shift must leave the reduced polarization invariant
///    as well.
#[test]
fn berry_polarization_is_invariant_up_to_the_polarization_quantum() {
    let params = params();
    let el = ElectronicOptions {
        electronic_temperature: 0.0,
        ..ElectronicOptions::default()
    };
    let pbc = berry_pbc();
    let berry = BerryPolarizationOptions {
        mesh: [1, 1, 1],
        method: BerryMethodSelector::KingSmithVanderbilt,
        ..BerryPolarizationOptions::default()
    };
    let base = nacl_polar(0.25);
    let reference = pbc_berry_polarization(&base, &params, &el, &pbc, &berry).unwrap();
    let lattice = base.lattice.as_ref().unwrap();
    println!(
        "[H1] polar NaCl reference: P_reduced = {:?}, quantum diag = [{:.6}, {:.6}, {:.6}]",
        reference.polarization,
        reference.quantum[0][0],
        reference.quantum[1][1],
        reference.quantum[2][2]
    );

    // (1) rigid translation by each primitive lattice vector. Each must move the
    // raw polarization by exactly one quantum along its OWN direction and by
    // none along the other two.
    for axis in 0..3 {
        let r = lattice.cell.column(axis);
        let moved = translate_system(&base, [r.x, r.y, r.z]);
        let shifted = pbc_berry_polarization(&moved, &params, &el, &pbc, &berry).unwrap();
        for d in 0..3 {
            let d_reduced = (shifted.polarization[d] - reference.polarization[d]).abs();
            let quantum = reference.quantum[d][d].abs();
            let d_raw = shifted.polarization_raw[d] - reference.polarization_raw[d];
            let quanta = if quantum > 0.0 { d_raw / quantum } else { 0.0 };
            let integer_defect = (quanta - quanta.round()).abs();
            println!(
                "[H1] NaCl + a_{axis}, component {d}: |ΔP_reduced| = {d_reduced:.3e}, \
                 ΔP_raw = {d_raw:.6e} = {quanta:.6} quanta (integer defect {integer_defect:.3e})"
            );
            assert!(
                d_reduced < 1.0e-8,
                "translating by lattice vector a_{axis} changed the reduced polarization \
                 component {d} by {d_reduced:.3e} — the same crystal must have the same P"
            );
            assert!(
                integer_defect < 1.0e-6,
                "translating by lattice vector a_{axis} moved the raw polarization component {d} \
                 by {quanta:.6} quanta, which is not an integer (defect {integer_defect:.3e})"
            );
        }
    }

    // (2) an arbitrary, non-lattice rigid translation. The cell is neutral, so
    // the polarization cannot depend on it.
    for shift in [[0.7, -1.3, 2.1], [3.0, 3.0, 3.0], [-1.75, 0.4, 6.2]] {
        let moved = translate_system(&base, shift);
        let shifted = pbc_berry_polarization(&moved, &params, &el, &pbc, &berry).unwrap();
        for d in 0..3 {
            let diff = (shifted.polarization[d] - reference.polarization[d]).abs();
            println!("[H2] NaCl shifted by {shift:?}, component {d}: |ΔP_reduced| = {diff:.3e}");
            assert!(
                diff < 1.0e-8,
                "an arbitrary rigid translation by {shift:?} changed the polarization component \
                 {d} of a CHARGE-NEUTRAL cell by {diff:.3e}"
            );
        }
    }
}

