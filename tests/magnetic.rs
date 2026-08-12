// SPDX-License-Identifier: GPL-3.0-or-later
//! Magnetic (GFN1-xTB-M0) SCC checks.

use gfn1_rs::magnetic::{lao_kinetic_matrix, lao_overlap_matrix, london_dress_ao_matrix};
use gfn1_rs::math::Vec3;
use gfn1_rs::{
    analytic_gradient, magnetic_analytic_gradient, magnetic_gradient, magnetic_h0_overlap,
    magnetizability_isotropic, magnetizability_tensor_analytic, parse_secondary_basis,
    run_electronic, run_magnetic_scc, run_magnetic_scc_m1, AnalyticGradientOptions, BasisOptions,
    BasisSet, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters, PeriodicSystem,
    SecondaryBasis, MAGNETIZABILITY_AU_TO_SI,
};

/// Load the GFN1-xTB-M1 secondary basis from the path in `GFN1_M1_BASIS` (the
/// paper's `$Basis = GFN1-xTB-cc-pVDZ` file). Tests no-op when it is absent.
fn load_m1_basis() -> Option<SecondaryBasis> {
    let path = std::env::var("GFN1_M1_BASIS").ok()?;
    let text = std::fs::read_to_string(path).ok()?;
    parse_secondary_basis(&text).ok()
}

fn load_params() -> Option<Gfn1Parameters> {
    Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";

fn opts_with_field(b: Vec3) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(b),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

#[test]
fn magnetic_scc_reduces_to_field_free_and_responds_to_field() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

    // Field-free GFN1 reference (T = 0 internal energy: band + SCC + rep/disp/hal).
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let e_ref = run_electronic(&system, &params, ref_opts)
        .unwrap()
        .total_internal;

    // B = 0 magnetic SCC must reproduce the field-free energy exactly.
    let m0 = run_magnetic_scc(&system, &params, &opts_with_field(Vec3::zero())).unwrap();
    assert!(m0.converged);
    assert!(
        (m0.energy - e_ref).abs() < 1.0e-8,
        "B=0 magnetic energy {} vs field-free {} (diff {:.3e})",
        m0.energy,
        e_ref,
        (m0.energy - e_ref).abs()
    );

    // A finite field perpendicular to the molecular plane: real, finite, converged,
    // and with a measurable effect on the energy. (The sign of the M0 magnetic
    // response is not guaranteed without the kinetic-energy correction of the
    // dual-basis M1 variant, so only the magnitude is checked here.)
    let mb = run_magnetic_scc(
        &system,
        &params,
        &opts_with_field(Vec3::new(0.0, 0.0, 0.05)),
    )
    .unwrap();
    assert!(mb.converged && mb.energy.is_finite());
    assert!(
        (mb.energy - m0.energy).abs() > 1.0e-7,
        "the magnetic field has no effect on the energy: B {} vs 0 {}",
        mb.energy,
        m0.energy
    );

    // The field couples through the gauge-origin-dependent London phase but the
    // energy is gauge-origin invariant for a neutral closed shell: shifting the
    // origin must not change the energy.
    let shifted = ExternalFieldOptions {
        magnetic_field: Some(Vec3::new(0.0, 0.0, 0.05)),
        origin: Vec3::new(1.3, -0.7, 0.4),
        ..ExternalFieldOptions::default()
    };
    let opts_shift = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: shifted,
        ..ElectronicOptions::default()
    };
    let ms = run_magnetic_scc(&system, &params, &opts_shift).unwrap();
    assert!(
        (ms.energy - mb.energy).abs() < 1.0e-7,
        "magnetic energy is not gauge-origin invariant: {} vs {}",
        ms.energy,
        mb.energy
    );
}

#[test]
fn m1_reduces_to_field_free_differs_from_m0_and_is_gauge_invariant() {
    let Some(params) = load_params() else {
        return;
    };
    let Some(secondary) = load_m1_basis() else {
        return; // needs GFN1_M1_BASIS
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

    // B = 0: M1 reproduces the field-free GFN1 energy (the KE correction vanishes).
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let e_ref = run_electronic(&system, &params, ref_opts)
        .unwrap()
        .total_internal;
    let m1_0 =
        run_magnetic_scc_m1(&system, &params, &opts_with_field(Vec3::zero()), &secondary).unwrap();
    assert!(
        (m1_0.energy - e_ref).abs() < 1.0e-8,
        "M1 B=0 energy {} vs field-free {}",
        m1_0.energy,
        e_ref
    );

    // Finite field: M1 must differ from M0 (the secondary basis changes the KE term).
    let bz = Vec3::new(0.0, 0.0, 0.08);
    let m0_b = run_magnetic_scc(&system, &params, &opts_with_field(bz)).unwrap();
    let m1_b = run_magnetic_scc_m1(&system, &params, &opts_with_field(bz), &secondary).unwrap();
    assert!(
        (m1_b.energy - m0_b.energy).abs() > 1.0e-7,
        "M1 energy equals M0 — secondary basis appears inactive ({} vs {})",
        m1_b.energy,
        m0_b.energy
    );

    // Gauge-origin invariance of the M1 energy.
    let shifted = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(bz),
            origin: Vec3::new(1.3, -0.7, 0.4),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    };
    let m1_s = run_magnetic_scc_m1(&system, &params, &shifted, &secondary).unwrap();
    assert!(
        (m1_s.energy - m1_b.energy).abs() < 1.0e-7,
        "M1 energy is not gauge-origin invariant: {} vs {}",
        m1_s.energy,
        m1_b.energy
    );

    // Isotropic magnetizabilities (printed for comparison with the paper / literature).
    let base = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    };
    let xi_m0 = magnetizability_isotropic(&system, &params, &base, None, 0.02).unwrap();
    let xi_m1 = magnetizability_isotropic(&system, &params, &base, Some(&secondary), 0.02).unwrap();
    eprintln!(
        "H2O isotropic magnetizability (10^-30 J/T^2): M0 = {:.3}, M1 = {:.3}",
        xi_m0 * MAGNETIZABILITY_AU_TO_SI,
        xi_m1 * MAGNETIZABILITY_AU_TO_SI
    );
    assert!(xi_m0.is_finite() && xi_m1.is_finite());
    assert!(
        (xi_m1 - xi_m0).abs() > 1.0e-12,
        "M1 magnetizability equals M0"
    );
}

#[test]
fn magnetic_gradient_matches_field_free_at_zero_and_responds_to_field() {
    let Some(params) = load_params() else {
        return;
    };
    // Off-equilibrium water so the gradient is non-trivial.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let step = 1.0e-3;

    // At B = 0 the magnetic (M0) gradient is the finite difference of the
    // field-free internal energy and must reproduce the field-free analytic
    // nuclear gradient.
    let g0 = magnetic_gradient(&system, &params, &opts_with_field(Vec3::zero()), step).unwrap();
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let ana = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: ref_opts,
            ..AnalyticGradientOptions::default()
        },
    )
    .unwrap();
    let mut max_diff = 0.0_f64;
    for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
        max_diff = max_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_diff < 5.0e-5,
        "B=0 magnetic gradient vs field-free analytic gradient: max diff {max_diff:.3e}"
    );

    // forces = -gradient.
    for (g, f) in g0.gradient.iter().zip(g0.forces.iter()) {
        assert!((g.x + f.x).abs() < 1.0e-14 && (g.y + f.y).abs() < 1.0e-14);
    }

    // A finite field changes the forces (real, finite) and the gradient remains
    // step-consistent between two finite-difference steps.
    let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
    let gb1 = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
    let gb2 = magnetic_gradient(&system, &params, &field, 2.0e-3).unwrap();
    let mut step_diff = 0.0_f64;
    let mut field_diff = 0.0_f64;
    for ((a, b), z) in gb1
        .gradient
        .iter()
        .zip(gb2.gradient.iter())
        .zip(g0.gradient.iter())
    {
        assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite());
        step_diff = step_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
        field_diff = field_diff
            .max((a.x - z.x).abs())
            .max((a.y - z.y).abs())
            .max((a.z - z.z).abs());
    }
    assert!(
        step_diff < 1.0e-4,
        "magnetic gradient not step-consistent: {step_diff:.3e}"
    );
    assert!(
        field_diff > 1.0e-7,
        "the field did not change the forces: {field_diff:.3e}"
    );
}

#[test]
fn magnetic_analytic_gradient_matches_field_free_and_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    // Off-equilibrium water so the gradient is non-trivial.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();

    // At B = 0 the analytic (Hellmann-Feynman) magnetic gradient must reproduce the
    // field-free GFN1 analytic nuclear gradient.
    let g0 = magnetic_analytic_gradient(
        &system,
        &params,
        &opts_with_field(Vec3::zero()),
        None,
        1.0e-3,
    )
    .unwrap();
    let ref_opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let ana = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: ref_opts,
            ..AnalyticGradientOptions::default()
        },
    )
    .unwrap();
    let mut max_b0 = 0.0_f64;
    for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
        max_b0 = max_b0
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_b0 < 5.0e-5,
        "B=0 analytic magnetic gradient vs field-free analytic gradient: {max_b0:.3e}"
    );

    // At a finite field the analytic gradient must match the finite-difference of the
    // converged magnetic energy ([`magnetic_gradient`]) — the key correctness check
    // of the Hellmann-Feynman density/Pulay contraction.
    let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
    let ga = magnetic_analytic_gradient(&system, &params, &field, None, 1.0e-3).unwrap();
    let gfd = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
    let mut max_fd = 0.0_f64;
    for (a, b) in ga.gradient.iter().zip(gfd.gradient.iter()) {
        max_fd = max_fd
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_fd < 1.0e-4,
        "analytic vs finite-difference magnetic gradient at finite B: {max_fd:.3e}"
    );

    // forces = -gradient.
    for (g, f) in ga.gradient.iter().zip(ga.forces.iter()) {
        assert!((g.x + f.x).abs() < 1.0e-14 && (g.z + f.z).abs() < 1.0e-14);
    }
}

// ---------------------------------------------------------------------------
// Physical-invariance audit gates.
//
// Every observable the GFN1-xTB-M (London/GIAO) machinery produces has to obey a
// short list of exact symmetries. They are cheap, they are independent of any
// reference implementation, and each one pins down a different part of the
// assembly:
//
//   * rigid rotation of (molecule, B)     -> the Cartesian-Gaussian LAO integrals
//                                            and the H0(B) band prefactor,
//   * rigid translation / gauge-origin    -> the London-orbital construction,
//   * B -> -B (time reversal)             -> complex-conjugation consistency,
//   * Hermiticity of H0(B), S(B)          -> the eigenproblem is well posed.
// ---------------------------------------------------------------------------

/// Right-handed rotation matrix `Rz(c) Ry(b) Rx(a)` (radians).
fn rotation(a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let (sc, cc) = c.sin_cos();
    [
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb, cb * sa, cb * ca],
    ]
}

fn apply_rotation(r: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    )
}

fn rotated_system(system: &PeriodicSystem, r: &[[f64; 3]; 3]) -> PeriodicSystem {
    let mut out = system.clone();
    for atom in out.atoms.iter_mut() {
        atom.position = apply_rotation(r, atom.position);
    }
    out
}

fn translated_system(system: &PeriodicSystem, d: Vec3) -> PeriodicSystem {
    let mut out = system.clone();
    for atom in out.atoms.iter_mut() {
        atom.position += d;
    }
    out
}

fn field_opts(b: Vec3, origin: Vec3) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(b),
            origin,
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

/// **Rotational invariance.** Rotating the molecule and the magnetic field together
/// is an exact symmetry of the GFN1-xTB-M energy: the London (GIAO) overlap, the
/// `pi^2` kinetic correction and the GFN1 `H0` band prefactor are all built from
/// rotation-covariant Cartesian-Gaussian integrals over a basis (s + p here) that
/// spans complete angular shells.
///
/// This is the sharpest structural gate available for the field-dressed `H0(B)`,
/// because the *field-free* overlap `S` has exact symmetry zeros in an axis-aligned
/// frame (water in the `xy` plane has `<O 2p_z|H 1s> = 0`) while the London overlap
/// `S(B)` at an in-plane field does **not** — any assembly that infers the band
/// prefactor from `H0/S` per AO pair silently drops those elements and becomes
/// orientation dependent. The `B = 0` control isolates the failure to the magnetic
/// terms (the field-free GFN1 energy is trivially rotation invariant).
#[test]
fn magnetic_energy_is_rotation_invariant() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    // Generic rotation: no AO-pair overlap vanishes by symmetry in the rotated frame.
    let r = rotation(0.37, -0.52, 0.83);
    let rotated = rotated_system(&system, &r);

    // Control: at B = 0 this is just the field-free GFN1 energy.
    let e0 = run_magnetic_scc(&system, &params, &opts_with_field(Vec3::zero()))
        .unwrap()
        .energy;
    let e0r = run_magnetic_scc(&rotated, &params, &opts_with_field(Vec3::zero()))
        .unwrap()
        .energy;
    assert!(
        (e0 - e0r).abs() < 1.0e-9,
        "B=0 energy is not rotation invariant ({e0} vs {e0r}) - the test rotation is wrong"
    );

    // In-plane field: the case that exercises the symmetry-zero AO pairs.
    for b in [
        Vec3::new(0.05, 0.03, 0.0),
        Vec3::new(0.0, 0.0, 0.05),
        Vec3::new(0.03, -0.02, 0.04),
    ] {
        let e = run_magnetic_scc(&system, &params, &opts_with_field(b))
            .unwrap()
            .energy;
        let er = run_magnetic_scc(&rotated, &params, &opts_with_field(apply_rotation(&r, b)))
            .unwrap()
            .energy;
        eprintln!("B = {b:?}: E = {e:.12}, E(rotated) = {er:.12}, d = {:.3e}", e - er);
        assert!(
            (e - er).abs() < 1.0e-9,
            "magnetic energy is not rotation invariant at B = {b:?}: {e} vs {er} \
             (diff {:.3e})",
            (e - er).abs()
        );
    }
}

/// **Rotational covariance of the magnetizability tensor**: `xi(R r) = R xi(r) R^T`.
/// A stronger statement than the scalar energy invariance because it also pins the
/// tensor's index handling (the off-diagonal cross-field finite differences).
#[test]
fn magnetizability_tensor_is_rotationally_covariant() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let base = field_opts(Vec3::zero(), Vec3::zero());
    let step = 0.006;
    let xi = magnetizability_tensor_analytic(&system, &params, &base, None, step).unwrap();

    let r = rotation(0.41, 0.28, -0.63);
    let rotated = rotated_system(&system, &r);
    let xi_rot = magnetizability_tensor_analytic(&rotated, &params, &base, None, step).unwrap();

    // Expected = R xi R^T.
    let mut expected = [[0.0_f64; 3]; 3];
    for (a, row) in expected.iter_mut().enumerate() {
        for (b, slot) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, xi_i) in xi.iter().enumerate() {
                for (j, &xi_ij) in xi_i.iter().enumerate() {
                    acc += r[a][i] * xi_ij * r[b][j];
                }
            }
            *slot = acc;
        }
    }
    let scale = xi
        .iter()
        .flatten()
        .fold(0.0_f64, |m, &x| m.max(x.abs()))
        .max(1.0e-3);
    let mut max_diff = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            max_diff = max_diff.max((xi_rot[a][b] - expected[a][b]).abs());
        }
    }
    eprintln!("magnetizability rotational covariance: max |xi(R) - R xi R^T| = {max_diff:.3e} (|xi|max = {scale:.4})");
    assert!(
        max_diff < 2.0e-3 * scale,
        "magnetizability tensor is not rotationally covariant: {max_diff:.3e} vs |xi| {scale:.3e}"
    );
}

/// **Translational / gauge-origin invariance.** London orbitals make the energy
/// independent of the gauge origin `O`; equivalently, rigidly translating the
/// molecule at fixed `O` (and fixed `B`) cannot change the energy. Both directions
/// are checked, plus the two-at-once case.
#[test]
fn magnetic_energy_is_translation_and_gauge_origin_invariant() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let b = Vec3::new(0.04, -0.02, 0.05);
    let d = Vec3::new(1.7, -0.9, 2.3);

    let e_ref = run_magnetic_scc(&system, &params, &field_opts(b, Vec3::zero()))
        .unwrap()
        .energy;
    // (a) move the gauge origin, keep the molecule
    let e_gauge = run_magnetic_scc(&system, &params, &field_opts(b, d))
        .unwrap()
        .energy;
    // (b) move the molecule, keep the gauge origin
    let e_trans = run_magnetic_scc(
        &translated_system(&system, d),
        &params,
        &field_opts(b, Vec3::zero()),
    )
    .unwrap()
    .energy;
    // (c) move both, incoherently
    let e_both = run_magnetic_scc(
        &translated_system(&system, d),
        &params,
        &field_opts(b, Vec3::new(-0.6, 2.1, 0.4)),
    )
    .unwrap()
    .energy;
    for (name, e) in [("gauge", e_gauge), ("translation", e_trans), ("both", e_both)] {
        assert!(
            (e - e_ref).abs() < 1.0e-9,
            "magnetic energy is not {name} invariant: {e} vs {e_ref} (diff {:.3e})",
            (e - e_ref).abs()
        );
    }
}

/// The gauge-origin invariance above only means something if the observable *could*
/// have been origin dependent. This gate proves the test discriminates: the exact
/// London (GIAO) overlap `S(B)` from the complex Gaussian product theorem is
/// bit-for-bit independent of `O` (the London wave vectors enter only through the
/// difference `A_mu - A_nu = 1/2 B x (R_mu - R_nu)`), whereas the naive Peierls-style
/// phase dressing `M_munu exp(i/2 B.[(R_mu-O) x (R_nu-O)])` of
/// [`london_dress_ao_matrix`] is strongly origin dependent and is therefore *not* a
/// substitute for the real integrals.
#[test]
fn lao_overlap_is_gauge_origin_free_but_phase_dressing_is_not() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
    let n = basis.len();
    let b = Vec3::new(0.03, -0.05, 0.07);
    let o1 = ExternalFieldOptions {
        magnetic_field: Some(b),
        ..ExternalFieldOptions::default()
    };
    let o2 = ExternalFieldOptions {
        magnetic_field: Some(b),
        origin: Vec3::new(1.9, -0.8, 1.1),
        ..ExternalFieldOptions::default()
    };

    let s1 = lao_overlap_matrix(&system, &basis, &o1);
    let s2 = lao_overlap_matrix(&system, &basis, &o2);
    let mut lao_diff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            lao_diff = lao_diff
                .max((s1.re[(i, j)] - s2.re[(i, j)]).abs())
                .max((s1.im[(i, j)] - s2.im[(i, j)]).abs());
        }
    }
    assert!(
        lao_diff < 1.0e-14,
        "the exact LAO overlap must not depend on the gauge origin: {lao_diff:.3e}"
    );

    // Discriminator: the same comparison on the naive phase dressing must fail badly.
    let real = &s1.re;
    let d1 = london_dress_ao_matrix(real, &system, &basis, &o1).unwrap();
    let d2 = london_dress_ao_matrix(real, &system, &basis, &o2).unwrap();
    let mut dress_diff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            dress_diff = dress_diff
                .max((d1.re[(i, j)] - d2.re[(i, j)]).abs())
                .max((d1.im[(i, j)] - d2.im[(i, j)]).abs());
        }
    }
    eprintln!("gauge-origin shift: |dS(LAO)| = {lao_diff:.3e}, |dS(phase dressing)| = {dress_diff:.3e}");
    assert!(
        dress_diff > 1.0e-3,
        "the phase-dressing control did not move with the gauge origin ({dress_diff:.3e}) - \
         the invariance gate above would not discriminate"
    );
}

/// The London kinetic integral `<omega_mu|1/2 pi^2|omega_nu>` (`pi = p + A`,
/// `A = 1/2 B x (r - O)`) is gauge-origin invariant even though every ingredient of
/// its evaluation — the ket London wave vector `k_b = 1/2 B x (R_nu - O)`, the
/// multipoles `<a|(x-O)^n|b>` — is origin dependent. Shifting `O` multiplies both
/// orbitals by the same plane wave `exp(i chi_0.r)` and shifts `pi` by `-chi_0`, so
/// the matrix element is unchanged. This is a complete, reference-free check of the
/// operator decomposition in `lao_kinetic_pair` and of the 1D `S/D1/D2/M1/M2` block
/// recursions: a wrong coefficient, a dropped `-i k_b` term or a mismatched
/// multipole would break it immediately.
#[test]
fn lao_kinetic_is_gauge_origin_invariant_and_hermitian() {
    let Some(params) = load_params() else {
        return;
    };
    // Low-symmetry water so every 1D block carries a nonzero contribution.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.62 0.51 0.30\nH -0.51 0.59 0.22\n",
        0.0,
        false,
    )
    .unwrap();
    let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
    let n = basis.len();
    let b = Vec3::new(0.06, -0.04, 0.09);
    let make = |origin: Vec3| ExternalFieldOptions {
        magnetic_field: Some(b),
        origin,
        ..ExternalFieldOptions::default()
    };
    let k0 = lao_kinetic_matrix(&system, &basis, &make(Vec3::zero()));
    let k1 = lao_kinetic_matrix(&system, &basis, &make(Vec3::new(1.3, -0.7, 0.4)));
    let k2 = lao_kinetic_matrix(&system, &basis, &make(Vec3::new(-2.6, 3.1, -1.8)));
    let (mut diff, mut mag, mut herm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            diff = diff
                .max((k0.re[(i, j)] - k1.re[(i, j)]).abs())
                .max((k0.im[(i, j)] - k1.im[(i, j)]).abs())
                .max((k0.re[(i, j)] - k2.re[(i, j)]).abs())
                .max((k0.im[(i, j)] - k2.im[(i, j)]).abs());
            mag = mag.max(k0.re[(i, j)].abs()).max(k0.im[(i, j)].abs());
            herm = herm
                .max((k2.re[(i, j)] - k2.re[(j, i)]).abs())
                .max((k2.im[(i, j)] + k2.im[(j, i)]).abs());
        }
    }
    eprintln!("LAO kinetic: max |K(O1) - K(O2)| = {diff:.3e} (|K|max = {mag:.4})");
    assert!(mag > 1.0e-3, "degenerate LAO kinetic matrix");
    assert!(
        diff < 1.0e-9,
        "the LAO kinetic integral must be gauge-origin invariant: {diff:.3e}"
    );
    assert!(
        herm < 1.0e-9,
        "the LAO kinetic matrix is not Hermitian at an off-centre gauge origin: {herm:.3e}"
    );
}

/// **Time reversal.** For a closed shell with no spin-Zeeman term the GFN1-xTB-M
/// Hamiltonian satisfies `H0(-B) = H0(B)*`, `S(-B) = S(B)*`, so the spectrum and
/// hence the energy are exactly even in `B`. Also gates the Hermiticity of the
/// assembled `H0(B)` / `S(B)` that the complex generalized eigensolver assumes.
#[test]
fn magnetic_energy_is_even_in_field_and_matrices_are_hermitian() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let b = Vec3::new(0.04, -0.03, 0.06);
    let ep = run_magnetic_scc(&system, &params, &field_opts(b, Vec3::zero()))
        .unwrap()
        .energy;
    let em = run_magnetic_scc(
        &system,
        &params,
        &field_opts(-b, Vec3::zero()),
    )
    .unwrap()
    .energy;
    assert!(
        (ep - em).abs() < 1.0e-9,
        "the closed-shell magnetic energy must be even in B: E(B) = {ep}, E(-B) = {em}"
    );

    // H0(B), S(B) Hermitian; H0(-B) = conj(H0(B)), S(-B) = conj(S(B)).
    let opt_p = field_opts(b, Vec3::new(0.3, -0.2, 0.1));
    let opt_m = field_opts(-b, Vec3::new(0.3, -0.2, 0.1));
    let (h0p, sp) = magnetic_h0_overlap(&system, &params, &opt_p, None).unwrap();
    let (h0m, sm) = magnetic_h0_overlap(&system, &params, &opt_m, None).unwrap();
    let n = h0p.n;
    let (mut herm, mut conj, mut im_max) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            herm = herm
                .max((h0p.re[(i, j)] - h0p.re[(j, i)]).abs())
                .max((h0p.im[(i, j)] + h0p.im[(j, i)]).abs())
                .max((sp.re[(i, j)] - sp.re[(j, i)]).abs())
                .max((sp.im[(i, j)] + sp.im[(j, i)]).abs());
            conj = conj
                .max((h0p.re[(i, j)] - h0m.re[(i, j)]).abs())
                .max((h0p.im[(i, j)] + h0m.im[(i, j)]).abs())
                .max((sp.re[(i, j)] - sm.re[(i, j)]).abs())
                .max((sp.im[(i, j)] + sm.im[(i, j)]).abs());
            im_max = im_max.max(h0p.im[(i, j)].abs());
        }
    }
    assert!(im_max > 1.0e-6, "the field produced no imaginary H0(B)");
    assert!(herm < 1.0e-10, "H0(B)/S(B) are not Hermitian: {herm:.3e}");
    assert!(
        conj < 1.0e-10,
        "H0(-B) != conj(H0(B)) (time-reversal broken): {conj:.3e}"
    );
}

/// **Sign / magnitude convention.** The code uses `xi_ab = -d^2 E / dB_a dB_b`, i.e.
/// `E(B) = E(0) - 1/2 xi_ab B_a B_b`. In that (standard) convention a closed-shell,
/// diamagnetic molecule is *repelled* by the field, so `d^2E/dB^2 > 0` and the
/// isotropic magnetizability is **negative**. Water and methane are both firmly
/// diamagnetic (experiment / GIAO-CCSD put `xi_iso(H2O)` near
/// `-230 x 10^-30 J T^-2`), so the sign is not a free convention here. Checked for
/// the finite-field route and the analytic CP-SCC route, which must agree.
#[test]
fn diamagnetic_molecules_have_negative_isotropic_magnetizability() {
    let Some(params) = load_params() else {
        return;
    };
    let base = field_opts(Vec3::zero(), Vec3::zero());
    let cases = [
        ("water", WATER),
        (
            "methane",
            "5\nCH4\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH 0.6276 -0.6276 -0.6276\n\
             H -0.6276 0.6276 -0.6276\nH -0.6276 -0.6276 0.6276\n",
        ),
    ];
    for (name, xyz) in cases {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let xi = magnetizability_isotropic(&system, &params, &base, None, 0.004).unwrap();
        let si = xi * MAGNETIZABILITY_AU_TO_SI;
        eprintln!("{name}: xi_iso = {xi:.6} a.u. = {si:.2} x 10^-30 J T^-2");
        assert!(
            xi < 0.0,
            "{name} is diamagnetic: xi_iso must be negative in the xi = -d2E/dB2 \
             convention, got {xi} a.u."
        );
        // Order of magnitude: a tight-binding valence model lands within a factor of a
        // few of the correlated value; anything outside this window is a unit/scale bug.
        assert!(
            (-2000.0..-5.0).contains(&si),
            "{name}: isotropic magnetizability {si:.2} x 10^-30 J T^-2 is outside any \
             physically plausible range for a small diamagnetic molecule"
        );
    }
}

/// **Matrix-level translation covariance** — the sharpest localisation of the
/// invariances above. Translating the molecule by `d` at a fixed gauge origin turns
/// each London orbital into `exp(i A_mu.d)` times the original, so both `H0(B)` and
/// `S(B)` must transform by the *same* diagonal unitary,
/// `M'_{mu nu} = exp(i (A_mu - A_nu).d) M_{mu nu}`. Two consequences are checked
/// element by element: every modulus `|M_{mu nu}|` is unchanged, and the phase that
/// `H0(B)` picks up equals the phase that `S(B)` picks up.
///
/// This is what fails first if the band prefactor in `H0(B) = hij S(B) + KE` is
/// recovered per AO pair as `H0_real/S_real`: for the symmetry-zero pairs of a planar
/// molecule the denominator is a hard zero at one position and a `1e-16` rounding
/// artefact at another, so `|H0(B)|` moves while `|S(B)|` does not.
#[test]
fn magnetic_h0_and_overlap_are_translation_covariant() {
    let Some(params) = load_params() else {
        return;
    };
    // Water in the xy plane: <O 2p_z | H 1s> vanishes at B = 0 but not at an in-plane B.
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let moved = translated_system(&system, Vec3::new(1.7, -0.9, 2.3));
    let opt = field_opts(Vec3::new(0.04, -0.02, 0.05), Vec3::zero());
    let (h0a, sa) = magnetic_h0_overlap(&system, &params, &opt, None).unwrap();
    let (h0b, sb) = magnetic_h0_overlap(&moved, &params, &opt, None).unwrap();
    let n = sa.n;
    let modulus = |m: &gfn1_rs::pbc::complex::CMatrix, i: usize, j: usize| {
        (m.re[(i, j)].powi(2) + m.im[(i, j)].powi(2)).sqrt()
    };
    let phase = |m: &gfn1_rs::pbc::complex::CMatrix, i: usize, j: usize| {
        m.im[(i, j)].atan2(m.re[(i, j)])
    };
    let wrap = |mut x: f64| {
        while x > std::f64::consts::PI {
            x -= 2.0 * std::f64::consts::PI;
        }
        while x < -std::f64::consts::PI {
            x += 2.0 * std::f64::consts::PI;
        }
        x
    };
    let (mut dmod_s, mut dmod_h, mut dphase) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            dmod_s = dmod_s.max((modulus(&sa, i, j) - modulus(&sb, i, j)).abs());
            dmod_h = dmod_h.max((modulus(&h0a, i, j) - modulus(&h0b, i, j)).abs());
            // Only compare phases where both matrices carry a resolvable magnitude.
            if modulus(&sa, i, j) > 1.0e-6 && modulus(&h0a, i, j) > 1.0e-6 {
                let ds = wrap(phase(&sb, i, j) - phase(&sa, i, j));
                let dh = wrap(phase(&h0b, i, j) - phase(&h0a, i, j));
                dphase = dphase.max(wrap(ds - dh).abs());
            }
        }
    }
    eprintln!(
        "translation covariance: max d|S| = {dmod_s:.3e}, max d|H0| = {dmod_h:.3e}, \
         max phase mismatch = {dphase:.3e}"
    );
    assert!(dmod_s < 1.0e-12, "|S(B)| is not translation invariant: {dmod_s:.3e}");
    assert!(
        dmod_h < 1.0e-12,
        "|H0(B)| is not translation invariant: {dmod_h:.3e} - the band prefactor \
         hij is position dependent"
    );
    assert!(
        dphase < 1.0e-9,
        "H0(B) and S(B) pick up different London phases under translation: {dphase:.3e}"
    );
}

/// Every off-diagonal of the analytic magnetizability tensor must reproduce the mixed
/// energy finite difference `-d^2 E / dB_a dB_b`. The in-tensor symmetry assertion is
/// vacuous (the routine symmetrises by construction), so this checks the physics
/// instead, on a low-symmetry geometry where all three off-diagonals are nonzero.
#[test]
fn magnetizability_off_diagonals_match_mixed_energy_fd() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.62 0.51 0.30\nH -0.51 0.59 0.22\n",
        0.0,
        false,
    )
    .unwrap();
    let base = field_opts(Vec3::zero(), Vec3::zero());
    let step = 0.005;
    let xi = magnetizability_tensor_analytic(&system, &params, &base, None, step).unwrap();
    let energy = |b: Vec3| -> f64 {
        let mut o = base.clone();
        o.external_field.magnetic_field = Some(b);
        run_magnetic_scc(&system, &params, &o).unwrap().energy
    };
    let axis = |k: usize, s: f64| match k {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    };
    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let f = |sa: f64, sb: f64| axis(a, sa * step) + axis(b, sb * step);
        let d2 = (energy(f(1.0, 1.0)) - energy(f(1.0, -1.0)) - energy(f(-1.0, 1.0))
            + energy(f(-1.0, -1.0)))
            / (4.0 * step * step);
        let fd = -d2;
        eprintln!("xi[{a}][{b}]: analytic = {:.6}, FD = {fd:.6}", xi[a][b]);
        assert!(
            (xi[a][b] - fd).abs() < 5.0e-3 * fd.abs().max(0.01),
            "xi[{a}][{b}] analytic {} != finite-field {fd}",
            xi[a][b]
        );
    }
}

/// [`MAGNETIZABILITY_AU_TO_SI`] must be the CODATA-2018 atomic unit of
/// magnetizability `e^2 a_0^2 / m_e = 7.8910366008e-29 J T^-2`, expressed in
/// `10^-30 J T^-2`. Mirrors the `constants::kb_hartree_matches_si_exact_ratio`
/// pattern: derive the constant from the underlying SI values instead of trusting a
/// hand-typed literal.
#[test]
fn magnetizability_au_to_si_matches_codata() {
    // CODATA 2018 / SI-2019: e (exact), a_0, m_e.
    let e = 1.602_176_634e-19_f64;
    let a0 = 5.291_772_109_03e-11_f64;
    let me = 9.109_383_701_5e-31_f64;
    let au = e * e * a0 * a0 / me; // J T^-2
    let expected = au * 1.0e30;
    eprintln!("e^2 a_0^2 / m_e = {expected:.9} x 10^-30 J T^-2 (constant: {MAGNETIZABILITY_AU_TO_SI})");
    assert!(
        (MAGNETIZABILITY_AU_TO_SI - expected).abs() < 1.0e-9 * expected,
        "MAGNETIZABILITY_AU_TO_SI = {MAGNETIZABILITY_AU_TO_SI} but e^2 a_0^2 / m_e = \
         {expected} x 10^-30 J T^-2"
    );
}
