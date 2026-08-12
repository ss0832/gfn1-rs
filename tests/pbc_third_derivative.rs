// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gates for the seminumerical periodic third derivative
//! (`src/pbc/third_derivative.rs`) and the Grueneisen module
//! (`src/pbc/gruneisen.rs`).
//!
//! All gates run on the **2-atom primitive fcc cell of diamond** at `a = 3.567 A`
//! — a gapped insulator with integer occupations at `T = 0`, which keeps the
//! periodic CPXTB out of its finite-temperature branch, and small enough that a
//! full `2 * 3N` Hessian sweep fits comfortably in the test budget.
//!
//! The real-space cutoffs are reduced from the library defaults (`ao_cutoff`
//! 30 -> 12/16 Bohr, Ewald `real_cutoff` 40 -> 18/24 Bohr) purely for speed. Every
//! property gated here is either an exact invariance (the acoustic sum rule, the
//! vector/dense contraction identity) or a ratio of logarithms that is converged
//! with respect to the cutoffs: the diamond mode Grueneisen parameter is
//! `0.90542` at the library defaults and `0.90562` at the leanest cutoffs tried,
//! a 0.02% spread.
//!
//! # Why the k-point gates use a *distorted* cell
//!
//! The k-point gates ([`kpoint_third_derivative_gamma_mesh_matches_gamma_path`]
//! onwards) run on the same cell with **one carbon nudged 0.03 A along x**. That
//! is not cosmetic and not a tolerance dodge: `pbc_kpoint_hessian` is **wrong on
//! the undistorted cell**, by an amount comparable to the Hessian itself.
//!
//! Perfect diamond has a triply degenerate `t2` HOMO *and* LUMO at Gamma. Inside
//! an exactly degenerate block the complex generalized eigensolver returns an
//! arbitrary unitary basis, and the k-point CPXTB is not invariant under that
//! choice — so the analytic k-point Hessian picks up an error that depends
//! chaotically on rounding. Measured on the undistorted cell at a Gamma-only mesh
//! (so the correct answer is exactly `pbc_gamma_hessian`, itself verified against
//! the analytic-gradient finite difference to `2.3e-9`):
//!
//! ```text
//!  a (A)     ao_cutoff (Bohr)   10        12        14        16        20
//!  3.567                      2e-15   1.90e0    3.71e-1   5e-15     7.91e-1
//!  3.500                      4.15e-1 7.58e-1   1.26e0    3e-15     8e-15
//!  3.400                      2.07e-1 3e-15     3e-15     5e-15     1e-15
//!  3.567004                   7.88e-1 1e-15     5.79e-1   1.50e-1   2e-15
//! ```
//!
//! (`max |H_gamma - H_kpoint|` on a scale of `1.2 - 1.5` Hartree/Bohr^2 — i.e. up
//! to a 100% error, appearing and vanishing with a `4e-6 A` change in the lattice
//! constant.) With the 0.03 A distortion the degeneracy is lifted and **all
//! twenty** of those configurations collapse to `4e-16 .. 3e-14`.
//!
//! This is a pre-existing bug in `src/pbc/hessian.rs`, not in anything gated
//! here; it is recorded in `docs/pbc.md`. The FC3 entry points are only exposed
//! to it through *evaluations at the reference geometry*, which a central
//! difference over atomic displacements never makes — but the strain derivative
//! and the Grueneisen path evaluate isotropically scaled cells, which preserve
//! the point group exactly, so those would hit it head-on. Hence one fixture,
//! distorted, for every k-point gate.

use gfn1_rs::pbc::gruneisen::{pbc_gruneisen, GruneisenOptions, SecondOrderStencil};
use gfn1_rs::pbc::hessian::pbc_gamma_hessian;
use gfn1_rs::pbc::third_derivative::{
    pbc_kpoint_strain_hessian_derivative, pbc_kpoint_third_derivative_seminumerical_dense,
    pbc_kpoint_third_derivative_seminumerical_vector, pbc_strain_hessian_derivative,
    pbc_third_derivative_seminumerical_dense, pbc_third_derivative_seminumerical_vector,
};
use gfn1_rs::vibrational::vibrational_analysis;
use gfn1_rs::{ElectronicOptions, EwaldOptions, Gfn1Parameters, KMesh, PbcOptions, PeriodicSystem};
use std::time::Instant;

/// Primitive (2-atom) fcc cell of diamond, `a = 3.567 A`.
const DIAMOND_PRIMITIVE: &str = "2\n\
Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.891750 0.891750 0.891750\n";

/// The same cell with one carbon displaced `0.03 A` along x, which lifts the
/// triply degenerate `t2` frontier orbitals of perfect diamond. See the module
/// docs for why every k-point gate needs that.
const DIAMOND_PRIMITIVE_DISTORTED: &str = "2\n\
Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.921750 0.891750 0.891750\n";

const NAT: usize = 2;
const NDOF: usize = 3 * NAT;

fn params() -> Gfn1Parameters {
    Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
}

fn diamond() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE, 0.0, false).unwrap()
}

fn diamond_distorted() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(DIAMOND_PRIMITIVE_DISTORTED, 0.0, false).unwrap()
}

/// Tight SCC, `T = 0` (integer occupations) — the periodic finite-temperature
/// CPXTB fixed point must not be exercised here.
fn electronic() -> ElectronicOptions {
    ElectronicOptions {
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

/// [`electronic`] with the electronic temperature pinned to **exactly zero**.
///
/// The Gamma gates inherit `ElectronicOptions::default()`'s `300 K` and rely on
/// diamond's Fermi occupations coming out integer to within the `1e-10`
/// fractional-occupancy epsilon (see `docs/pbc.md` §5). The k-point gates do not
/// rely on that at all: they pin the temperature, so no gate here can drift onto
/// the finite-temperature branch through a fixture change. Smeared *periodic*
/// derivatives are gated by the finite-temperature workstream, not here.
fn electronic_t0() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        ..electronic()
    }
}

fn pbc_options(ao_cutoff: f64, real_cutoff: f64, sr_cutoff: f64) -> PbcOptions {
    PbcOptions {
        ao_cutoff,
        ewald: EwaldOptions {
            real_cutoff,
            sr_cutoff,
            ..EwaldOptions::default()
        },
        ..PbcOptions::default()
    }
}

/// Same lean cutoffs, plus an explicit Monkhorst-Pack mesh. `[1, 1, 1]` selects
/// the true Gamma-only mesh (`KMesh::gamma`), which is what makes the k-point
/// path collapse onto the Gamma path.
fn kpbc_options(mesh: [usize; 3], ao_cutoff: f64, real_cutoff: f64, sr_cutoff: f64) -> PbcOptions {
    PbcOptions {
        kmesh: if mesh == [1, 1, 1] {
            KMesh::gamma()
        } else {
            KMesh::monkhorst_pack(mesh)
        },
        ..pbc_options(ao_cutoff, real_cutoff, sr_cutoff)
    }
}

/// `max |a - b|` and `max |a|` over a list of slabs, for reporting FD agreement
/// on a common scale.
fn slab_diff(a: &[gfn1_rs::linalg::Matrix], b: &[gfn1_rs::linalg::Matrix]) -> (f64, f64) {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (sa, sb) in a.iter().zip(b.iter()) {
        for i in 0..NDOF {
            for j in 0..NDOF {
                worst = worst.max((sa[(i, j)] - sb[(i, j)]).abs());
                scale = scale.max(sa[(i, j)].abs());
            }
        }
    }
    (worst, scale)
}

/// Acoustic sum rule for the seminumerical periodic third derivative, and the
/// exactness of the vector contraction against the dense slabs.
///
/// **The contraction that is tested.** For a Cartesian axis `alpha`, the
/// direction `v^alpha_c = 1 if c % 3 == alpha else 0` translates *every* atom in
/// the cell by the same vector with the lattice held fixed. That maps the
/// crystal onto itself, so `H(R + t v^alpha) = H(R)` for all `t` and therefore
///
/// ```text
///     sum_{atoms i} slabs[3*i + alpha][(a, b)] = 0     for every (a, b), alpha.
/// ```
///
/// This is a genuine cancellation across `2 * 3N` independently converged SCC +
/// Hessian evaluations — nothing in the implementation symmetrises or projects
/// the slabs to make it hold.
#[test]
fn seminumerical_third_derivative_acoustic_sum_rule_and_vector_contraction() {
    let params = params();
    let system = diamond();
    let el = electronic();
    let pbc = pbc_options(12.0, 18.0, 8.0);
    let step = 1.0e-3;

    let t = Instant::now();
    let slabs =
        pbc_third_derivative_seminumerical_dense(&system, &params, &el, &pbc, step).unwrap();
    let dense_secs = t.elapsed().as_secs_f64();
    assert_eq!(slabs.len(), NDOF);

    let scale = slabs
        .iter()
        .flat_map(|s| s.as_slice().iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 0.1 && scale.is_finite(),
        "third-derivative magnitude {scale:.3e} looks degenerate"
    );

    let mut worst = 0.0_f64;
    for axis in 0..3 {
        for a in 0..NDOF {
            for b in 0..NDOF {
                let sum: f64 = (0..NAT).map(|i| slabs[3 * i + axis][(a, b)]).sum();
                worst = worst.max(sum.abs());
            }
        }
    }
    println!(
        "pbc third derivative: dense {dense_secs:.1} s, scale {scale:.4e} Eh/Bohr^3, \
         translational sum-rule residual {worst:.3e}"
    );
    assert!(
        worst < 1.0e-5,
        "translational acoustic sum rule residual {worst:.3e} (scale {scale:.3e})"
    );

    // Vector mode: an exact contraction of the same per-DOF central differences,
    // so it must reproduce the dense contraction to machine precision (in fact
    // bit-for-bit). A deliberately sparse direction also exercises the
    // zero-component skip.
    let mut v = vec![0.0; NDOF];
    v[1] = 0.7;
    v[4] = -1.3;
    let t = Instant::now();
    let k = pbc_third_derivative_seminumerical_vector(&system, &params, &el, &pbc, step, &v)
        .unwrap();
    let vector_secs = t.elapsed().as_secs_f64();
    let mut err = 0.0_f64;
    let mut kscale = 0.0_f64;
    for a in 0..NDOF {
        for b in 0..NDOF {
            let want: f64 = (0..NDOF).map(|c| v[c] * slabs[c][(a, b)]).sum();
            err = err.max((k[(a, b)] - want).abs());
            kscale = kscale.max(want.abs());
        }
    }
    println!(
        "pbc third derivative: vector {vector_secs:.1} s, |K - sum_c v_c T_c| = {err:.3e} \
         (scale {kscale:.4e})"
    );
    assert!(
        err < 1.0e-12,
        "vector mode vs dense contraction: err={err:.3e} (scale {kscale:.3e})"
    );
}

/// **k-point gate (a).** A `1 x 1 x 1` mesh must collapse the k-point
/// seminumerical third derivative onto the Gamma one.
///
/// The two paths are *not* the same code: `pbc_kpoint_hessian` assembles the
/// fixed part from real-space image densities `P(T)/W(T)` obtained by an inverse
/// Bloch transform, and solves the **complex** CPXTB by preconditioned CG, where
/// `pbc_gamma_hessian` uses the real density directly and a direct real solve. At
/// a Gamma-only mesh those must agree analytically; the residual is the k-point
/// CPXTB's iterative-solver noise, amplified by `1/(2 h)` from the finite
/// difference. `kpoint_hessian_reduces_to_gamma_at_gamma_only` (a unit test in
/// `src/pbc/hessian.rs`) gates the Hessians themselves at `1e-8`; here the same
/// residual is divided by `2 h = 2e-3`, so a `1e-5`-scale slab difference would
/// still be consistent with that bound. What is observed is `1e-10` absolute /
/// `1e-10` relative — close, but **not** bit-for-bit (5 of 216 entries match
/// exactly), which is exactly what an iterative complex solve reproducing a
/// direct real one should look like.
#[test]
fn kpoint_third_derivative_gamma_mesh_matches_gamma_path() {
    let params = params();
    let system = diamond_distorted();
    let el = electronic_t0();
    let gamma_pbc = pbc_options(12.0, 18.0, 8.0);
    let kpbc = kpbc_options([1, 1, 1], 12.0, 18.0, 8.0);
    let step = 1.0e-3;

    let t = Instant::now();
    let reference =
        pbc_third_derivative_seminumerical_dense(&system, &params, &el, &gamma_pbc, step).unwrap();
    let gamma_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let slabs =
        pbc_kpoint_third_derivative_seminumerical_dense(&system, &params, &el, &kpbc, step).unwrap();
    let kpoint_secs = t.elapsed().as_secs_f64();
    assert_eq!(slabs.len(), NDOF);

    let (worst, scale) = slab_diff(&reference, &slabs);
    let identical = reference
        .iter()
        .zip(slabs.iter())
        .flat_map(|(a, b)| a.as_slice().iter().zip(b.as_slice().iter()))
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    let total = NDOF * NDOF * NDOF;
    println!(
        "kpoint FC3 [1,1,1] vs Gamma: gamma {gamma_secs:.1} s, kpoint {kpoint_secs:.1} s, \
         scale {scale:.4e} Eh/Bohr^3, max |diff| {worst:.3e} (rel {:.3e}), \
         bit-identical {identical}/{total}",
        worst / scale
    );
    assert!(
        scale > 0.1 && scale.is_finite(),
        "third-derivative magnitude {scale:.3e} looks degenerate"
    );
    assert!(
        worst / scale < 1.0e-8,
        "Gamma-only k-mesh must reproduce the Gamma FC3: max |diff| {worst:.3e} on scale \
         {scale:.3e} (rel {:.3e})",
        worst / scale
    );
}

/// **k-point gates (b) + (c).** Translational (acoustic) sum rule of the k-point
/// seminumerical FC3 on a genuine `2 x 2 x 2` Monkhorst-Pack mesh, and the
/// exactness of the vector contraction against those dense slabs.
///
/// Translating every atom of the cell by a common Cartesian vector, lattice
/// fixed, maps the crystal onto itself, so for each axis `alpha`
///
/// ```text
///     sum_{atoms i} slabs[3*i + alpha][(a, b)] = 0     for every (a, b).
/// ```
///
/// Nothing symmetrises or projects the slabs, and the cancellation runs across
/// `2 * 3N` independently converged SCC + complex-CPXTB Hessians on 4
/// (time-reversal folded) k-points, so this is the real invariance check on the
/// whole k-point path. Note that the sum rule follows from translational
/// invariance alone and is therefore just as sharp on the distorted cell.
#[test]
fn kpoint_third_derivative_acoustic_sum_rule_and_vector_contraction() {
    let params = params();
    let system = diamond_distorted();
    let el = electronic_t0();
    let pbc = kpbc_options([2, 2, 2], 12.0, 18.0, 8.0);
    let step = 1.0e-3;

    let t = Instant::now();
    let slabs =
        pbc_kpoint_third_derivative_seminumerical_dense(&system, &params, &el, &pbc, step).unwrap();
    let secs = t.elapsed().as_secs_f64();

    let scale = slabs
        .iter()
        .flat_map(|s| s.as_slice().iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        for a in 0..NDOF {
            for b in 0..NDOF {
                let sum: f64 = (0..NAT).map(|i| slabs[3 * i + axis][(a, b)]).sum();
                worst = worst.max(sum.abs());
            }
        }
    }
    println!(
        "kpoint FC3 [2,2,2] [{secs:.1} s]: scale {scale:.4e} Eh/Bohr^3, \
         translational sum-rule residual {worst:.3e} (rel {:.3e})",
        worst / scale
    );
    assert!(
        scale > 0.1 && scale.is_finite(),
        "k-point third-derivative magnitude {scale:.3e} looks degenerate"
    );
    assert!(
        worst < 1.0e-8,
        "k-point translational acoustic sum rule residual {worst:.3e} (scale {scale:.3e})"
    );

    // (c) vector == dense contraction, on the k-point path at a real mesh.
    // Deliberately sparse, so the zero-component skip is exercised too.
    let mut v = vec![0.0; NDOF];
    v[1] = 0.7;
    v[4] = -1.3;
    let t = Instant::now();
    let k = pbc_kpoint_third_derivative_seminumerical_vector(&system, &params, &el, &pbc, step, &v)
        .unwrap();
    let vector_secs = t.elapsed().as_secs_f64();
    let mut err = 0.0_f64;
    let mut kscale = 0.0_f64;
    for a in 0..NDOF {
        for b in 0..NDOF {
            let want: f64 = (0..NDOF).map(|c| v[c] * slabs[c][(a, b)]).sum();
            err = err.max((k[(a, b)] - want).abs());
            kscale = kscale.max(want.abs());
        }
    }
    println!(
        "kpoint FC3 vector [{vector_secs:.1} s] vs dense contraction: \
         |K - sum_c v_c T_c| = {err:.3e} (scale {kscale:.4e})"
    );
    assert!(
        err < 1.0e-12,
        "k-point vector mode vs dense contraction: err={err:.3e} (scale {kscale:.3e})"
    );
}

/// **k-point gate (d).** k-mesh convergence smoke test of the seminumerical FC3.
///
/// A single-DOF direction (`v = e_0`, i.e. one slab `dH/dR_{0x}`) keeps this to
/// two analytic k-point Hessians per mesh, which is what makes a three-mesh study
/// affordable at all. Two statements are gated, both structural rather than
/// tuned:
///
/// 1. `[1,1,1]` and `[2,2,2]` must **differ** well above the FD noise floor — if
///    they did not, the k-mesh would not be reaching the third derivative at all
///    and every k-point number here would be a Gamma number in disguise.
/// 2. `[2,2,2]` -> `[3,3,3]` must be a **smaller** step than `[1,1,1]` ->
///    `[2,2,2]`, i.e. the sequence is settling. No absolute threshold is placed on
///    the residual: the meshes are not nested (`[2,2,2]` is the shifted
///    Monkhorst-Pack grid and excludes Gamma, `[3,3,3]` includes it), so the
///    approach is not expected to be smooth, only decreasing.
#[test]
fn kpoint_third_derivative_kmesh_trend() {
    let params = params();
    let system = diamond_distorted();
    let el = electronic_t0();
    let step = 1.0e-3;
    let mut v = vec![0.0; NDOF];
    v[0] = 1.0;

    let run = |mesh: [usize; 3]| {
        let pbc = kpbc_options(mesh, 12.0, 18.0, 8.0);
        let t = Instant::now();
        let k =
            pbc_kpoint_third_derivative_seminumerical_vector(&system, &params, &el, &pbc, step, &v)
                .unwrap();
        println!(
            "kpoint FC3 slab dH/dR_0x on mesh {mesh:?} [{:.1} s]",
            t.elapsed().as_secs_f64()
        );
        k
    };

    let m1 = run([1, 1, 1]);
    let m2 = run([2, 2, 2]);
    let m3 = run([3, 3, 3]);

    let diff = |a: &gfn1_rs::linalg::Matrix, b: &gfn1_rs::linalg::Matrix| {
        let mut d = 0.0_f64;
        for i in 0..NDOF {
            for j in 0..NDOF {
                d = d.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        d
    };
    let scale = (0..NDOF)
        .flat_map(|i| (0..NDOF).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(m3[(i, j)].abs()));
    let d12 = diff(&m1, &m2);
    let d23 = diff(&m2, &m3);
    println!(
        "kpoint FC3 k-mesh trend: scale {scale:.4e} Eh/Bohr^3, \
         |[1,1,1] - [2,2,2]| = {d12:.4e} (rel {:.3e}), |[2,2,2] - [3,3,3]| = {d23:.4e} \
         (rel {:.3e})",
        d12 / scale,
        d23 / scale
    );
    assert!(
        d12 / scale > 1.0e-4,
        "[1,1,1] and [2,2,2] FC3 slabs are suspiciously equal: {d12:.3e} on scale {scale:.3e}"
    );
    assert!(
        d23 < d12,
        "k-mesh refinement is not settling: |1-2| = {d12:.3e} but |2-3| = {d23:.3e}"
    );
}

/// **k-point strain-mixed derivative** `dH/d(ln V)`.
///
/// Two things are gated: the `1 x 1 x 1` collapse onto the Gamma strain
/// derivative (same argument as the FC3 gate above), and that a real mesh
/// actually moves the answer. The Monkhorst-Pack grid is defined in *fractional*
/// reciprocal coordinates, so it scales with the reciprocal lattice and both
/// strained cells are sampled at the same fractional k — that is what makes the
/// central difference well defined at fixed `pbc.kmesh`.
#[test]
fn kpoint_strain_hessian_derivative_gamma_collapse_and_mesh_shift() {
    let params = params();
    let system = diamond_distorted();
    let el = electronic_t0();
    let delta = 5.0e-3;

    let t = Instant::now();
    let gamma =
        pbc_strain_hessian_derivative(&system, &params, &el, &pbc_options(12.0, 18.0, 8.0), delta)
            .unwrap();
    let kgamma = pbc_kpoint_strain_hessian_derivative(
        &system,
        &params,
        &el,
        &kpbc_options([1, 1, 1], 12.0, 18.0, 8.0),
        delta,
    )
    .unwrap();
    let k222 = pbc_kpoint_strain_hessian_derivative(
        &system,
        &params,
        &el,
        &kpbc_options([2, 2, 2], 12.0, 18.0, 8.0),
        delta,
    )
    .unwrap();
    let secs = t.elapsed().as_secs_f64();

    let mut collapse = 0.0_f64;
    let mut mesh_shift = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..NDOF {
        for j in 0..NDOF {
            collapse = collapse.max((gamma[(i, j)] - kgamma[(i, j)]).abs());
            mesh_shift = mesh_shift.max((kgamma[(i, j)] - k222[(i, j)]).abs());
            scale = scale.max(gamma[(i, j)].abs());
        }
    }
    println!(
        "kpoint dH/dlnV [{secs:.1} s]: scale {scale:.4e} Eh/Bohr^2, \
         |Gamma - kpoint[1,1,1]| = {collapse:.3e} (rel {:.3e}), \
         |kpoint[1,1,1] - kpoint[2,2,2]| = {mesh_shift:.3e} (rel {:.3e})",
        collapse / scale,
        mesh_shift / scale
    );
    assert!(
        scale > 0.1 && scale.is_finite(),
        "dH/dlnV magnitude {scale:.3e} looks degenerate"
    );
    assert!(
        collapse / scale < 1.0e-7,
        "Gamma-only k-mesh must reproduce the Gamma dH/dlnV: {collapse:.3e} on scale {scale:.3e}"
    );
    assert!(
        mesh_shift / scale > 1.0e-4,
        "the [2,2,2] mesh left dH/dlnV unchanged ({mesh_shift:.3e} on scale {scale:.3e}); \
         the k-mesh is not reaching the strain derivative"
    );
}

/// **The pre-existing degenerate-frontier-orbital bug in `pbc_kpoint_hessian`**,
/// pinned as a **negative** gate: it must stay confined to exactly degenerate
/// cells, and it must not reappear on the distorted fixture every k-point gate in
/// this file relies on.
///
/// At a Gamma-only mesh `pbc_kpoint_hessian` must equal `pbc_gamma_hessian`
/// exactly (`kpoint_hessian_reduces_to_gamma_at_gamma_only` asserts as much on a
/// water cell). On **perfect** diamond it does not: the `t2` HOMO/LUMO triplets
/// are exactly degenerate, the complex eigensolver picks an arbitrary basis
/// inside them, and the k-point CPXTB is not invariant under that choice. The
/// error is up to 100% of the Hessian and moves chaotically with the cutoff (see
/// the module docs for the full scan).
///
/// This test therefore asserts the *distorted* cell is clean at three cutoffs —
/// which is what licenses every other k-point gate here — and separately records
/// that the undistorted cell is not, so that the day the bug is fixed this test
/// fails loudly and can be turned into a real gate.
#[test]
fn kpoint_hessian_degenerate_frontier_orbital_bug_is_confined() {
    use gfn1_rs::pbc::hessian::pbc_kpoint_hessian;
    let params = params();
    let el = electronic_t0();
    let compare = |system: &PeriodicSystem, ao: f64| {
        let pbc = pbc_options(ao, 18.0, 8.0);
        let g = pbc_gamma_hessian(system, &params, &el, &pbc).unwrap();
        let k = pbc_kpoint_hessian(system, &params, &el, &pbc).unwrap();
        let mut d = 0.0_f64;
        let mut s = 0.0_f64;
        for i in 0..NDOF {
            for j in 0..NDOF {
                d = d.max((g.hessian[(i, j)] - k.hessian[(i, j)]).abs());
                s = s.max(g.hessian[(i, j)].abs());
            }
        }
        (d, s)
    };

    let mut worst_distorted = 0.0_f64;
    let mut worst_perfect = 0.0_f64;
    for ao in [10.0_f64, 12.0, 14.0] {
        let (d, s) = compare(&diamond_distorted(), ao);
        println!("ao={ao:.1} distorted: |Hgamma - Hkpoint| = {d:.4e} on scale {s:.4e}");
        worst_distorted = worst_distorted.max(d / s);
        let (d, s) = compare(&diamond(), ao);
        println!("ao={ao:.1} perfect  : |Hgamma - Hkpoint| = {d:.4e} on scale {s:.4e}");
        worst_perfect = worst_perfect.max(d / s);
    }
    assert!(
        worst_distorted < 1.0e-12,
        "the distorted fixture is no longer clean (rel {worst_distorted:.3e}); every k-point \
         gate in this file rests on it"
    );
    assert!(
        worst_perfect > 1.0e-3,
        "pbc_kpoint_hessian now agrees with pbc_gamma_hessian on exactly degenerate diamond \
         (rel {worst_perfect:.3e}). If the degenerate-block bug is fixed, delete this negative \
         pin and move the k-point gates back onto the undistorted cell."
    );
}

/// **Grueneisen through the k-point Hessian** (`GruneisenOptions::kpoint`).
///
/// The option must be inert at a Gamma-only mesh (it then evaluates the same
/// physics through the complex path) and must actually change the answer on a
/// real mesh. Note what it does *not* do: the phonons are still those of one
/// cell at `q = 0`, so the acoustic-branch caveat of the module is untouched —
/// only the electronic Brillouin-zone sum behind each Hessian is converged.
#[test]
fn gruneisen_kpoint_routing() {
    let params = params();
    let system = diamond_distorted();
    let el = electronic_t0();

    let run = |kpoint: bool, mesh: [usize; 3]| {
        let t = Instant::now();
        let g = pbc_gruneisen(
            &system,
            &params,
            &GruneisenOptions {
                delta: 5.0e-3,
                temperatures: vec![300.0],
                electronic: el.clone(),
                pbc: kpbc_options(mesh, 12.0, 18.0, 8.0),
                kpoint,
                ..GruneisenOptions::default()
            },
        )
        .unwrap();
        println!(
            "gruneisen kpoint={kpoint} mesh={mesh:?} [{:.1} s]: freq={:.3} cm^-1  \
             gamma={:.6}  gamma_th(300 K)={:.6}  min_overlap={:.6}",
            t.elapsed().as_secs_f64(),
            g.frequencies_cm1[NDOF - 1],
            g.mode_gamma[NDOF - 1],
            g.gamma_at(300.0).unwrap(),
            g.min_optical_overlap()
        );
        g
    };

    let gamma = run(false, [1, 1, 1]);
    let kgamma = run(true, [1, 1, 1]);
    let k222 = run(true, [2, 2, 2]);

    // Gamma-only mesh: the option must be inert.
    for i in 3..NDOF {
        let rel = ((gamma.mode_gamma[i] - kgamma.mode_gamma[i]) / gamma.mode_gamma[i]).abs();
        assert!(
            rel < 1.0e-6,
            "kpoint=true on a Gamma-only mesh changed mode {i} gamma: {} vs {} (rel {rel:.3e})",
            gamma.mode_gamma[i],
            kgamma.mode_gamma[i]
        );
        let df = (gamma.frequencies_cm1[i] - kgamma.frequencies_cm1[i]).abs();
        assert!(
            df < 1.0e-4,
            "kpoint=true on a Gamma-only mesh moved mode {i}: {} vs {} cm^-1",
            gamma.frequencies_cm1[i],
            kgamma.frequencies_cm1[i]
        );
    }

    // A real mesh must move both the frequency and gamma, and stay physical.
    let df = (k222.frequencies_cm1[NDOF - 1] - gamma.frequencies_cm1[NDOF - 1]).abs();
    assert!(
        df > 1.0,
        "the [2,2,2] mesh left the optical frequency unchanged ({df:.3e} cm^-1)"
    );
    for i in 3..NDOF {
        let g = k222.mode_gamma[i];
        assert!(
            g.is_finite() && (0.5..=2.0).contains(&g),
            "k-point mode {i} gamma = {g} is outside the physical window 0.5..2.0"
        );
    }
    assert!(
        k222.min_optical_overlap() > 0.999,
        "k-point mode matching degraded: min optical subspace overlap {:.6}",
        k222.min_optical_overlap()
    );
}

/// Grueneisen parameters of diamond: physical window, degeneracy, mode-matching
/// quality, and convergence with respect to the volumetric strain `delta`.
///
/// Reference numbers from the first honest run (primitive diamond, GFN1-xTB,
/// frozen-ion isotropic strain): the triply degenerate optical mode sits at
/// `2292.5 cm^-1` (GFN1 overbinds badly against the experimental `1332 cm^-1`),
/// its `gamma = 0.90542`, and `gamma_th(300 K) = 0.90542`. Experiment for
/// diamond is `gamma ~ 0.9 - 1.2`, so the model value lands at the bottom of the
/// literature range. The windows below are deliberately broad — they are
/// sanity gates, not a fit.
#[test]
fn gruneisen_diamond_is_physical_and_delta_converged() {
    let params = params();
    let system = diamond();
    let el = electronic();
    let pbc = pbc_options(16.0, 24.0, 10.0);

    let run = |delta: f64| {
        let t = Instant::now();
        let g = pbc_gruneisen(
            &system,
            &params,
            &GruneisenOptions {
                delta,
                temperatures: vec![100.0, 300.0, 1000.0],
                electronic: el.clone(),
                pbc,
                ..GruneisenOptions::default()
            },
        )
        .unwrap();
        println!(
            "gruneisen delta={delta:.1e} [{:.1} s]: V={:.4} Bohr^3  freq(V0)={:.3} cm^-1  \
             mode_gamma={:?}  gamma_th={:?}  min_overlap={:.6}",
            t.elapsed().as_secs_f64(),
            g.volume,
            g.frequencies_cm1[NDOF - 1],
            &g.mode_gamma[g.acoustic_modes..],
            g.thermodynamic_gamma,
            g.min_optical_overlap()
        );
        g
    };

    let coarse = run(5.0e-3);
    let fine = run(2.5e-3);

    // Three acoustic modes at ~0 and a triply degenerate optical branch.
    assert_eq!(coarse.acoustic_modes, 3);
    assert_eq!(
        coarse.degenerate_groups,
        vec![(0, 3), (3, 3)],
        "expected an acoustic triplet and a degenerate optical triplet"
    );
    for i in 0..3 {
        assert!(
            coarse.frequencies_cm1[i].abs() < 1.0,
            "acoustic mode {i} is {} cm^-1, translational invariance is broken",
            coarse.frequencies_cm1[i]
        );
        assert!(
            coarse.mode_gamma[i].is_nan(),
            "acoustic mode {i} must be excluded (NaN), got {}",
            coarse.mode_gamma[i]
        );
    }
    for i in 3..NDOF {
        assert!(
            coarse.frequencies_cm1[i] > 500.0,
            "optical mode {i} at {} cm^-1 is not a real optical branch",
            coarse.frequencies_cm1[i]
        );
    }

    // Mode assignment must be clean (subspace projections ~ 1).
    assert!(
        coarse.min_optical_overlap() > 0.999,
        "mode matching degraded: min optical subspace overlap {:.6}",
        coarse.min_optical_overlap()
    );

    // Every optical mode gamma inside a broad physical window.
    for i in 3..NDOF {
        let g = coarse.mode_gamma[i];
        assert!(
            g.is_finite() && (0.5..=2.0).contains(&g),
            "optical mode {i} gamma = {g} is outside the physical window 0.5..2.0"
        );
    }

    // Thermodynamic average at 300 K, diamond literature ~0.9-1.2.
    let g300 = coarse.gamma_at(300.0).expect("300 K was requested");
    assert!(
        (0.7..=1.5).contains(&g300),
        "gamma_th(300 K) = {g300} outside the 0.7..1.5 window"
    );

    // delta-convergence: gamma from delta and delta/2 must agree to a few %.
    for i in 3..NDOF {
        let rel = ((coarse.mode_gamma[i] - fine.mode_gamma[i]) / coarse.mode_gamma[i]).abs();
        assert!(
            rel < 0.03,
            "mode {i} gamma not delta-converged: {} vs {} (rel {rel:.3e})",
            coarse.mode_gamma[i],
            fine.mode_gamma[i]
        );
    }
    let f300 = fine.gamma_at(300.0).unwrap();
    let rel = ((g300 - f300) / g300).abs();
    assert!(
        rel < 0.03,
        "gamma_th(300 K) not delta-converged: {g300} vs {f300} (rel {rel:.3e})"
    );
}

/// Second-order Grueneisen parameters of diamond.
///
/// `gamma2 = d^2 ln omega / d(ln V)^2 = -d gamma / d ln V = q * gamma`, with the
/// sign convention spelled out in `src/pbc/gruneisen.rs` (`gamma2` has **no**
/// leading minus sign, unlike `gamma`).
///
/// Three things are gated, in increasing order of interest:
///
/// 1. **Internal consistency of the fit.** The same polynomial fit that gives the
///    curvature also gives the slope; `mode_gamma_refit` must reproduce the
///    independent two-point `mode_gamma`. They differ only by the `O(delta^2)`
///    asymmetry of the nodes `ln(1 + delta)` vs `-ln(1 - delta)`.
/// 2. **`delta` convergence**, `delta` against `delta / 2`. See
///    [`gruneisen_diamond_second_order_delta_study`] for the full ladder: the
///    plateau is `delta <= 5e-3` (`gamma2 = -0.03833` from `1.25e-3` to `5e-3`,
///    spread `1.3e-4` relative), and — contrary to the usual "second differences
///    want a coarser step" instinct — *larger* `delta` is worse here, because the
///    SCC and the analytic Hessian are converged to ~`1e-12` in `ln lambda` while
///    the fixed real-space cutoffs step the image lists at a few `1e-3` strain.
/// 3. **Stencil independence.** The five-point (`O(delta^4)`) fit must agree with
///    the free three-point one.
///
/// Diamond has one degenerate optical triplet, so every mode shares one `gamma`
/// and the `dc/dlnV` reweighting term of the *full* `-d gamma_th/d ln V` cancels
/// identically — that term is gated separately by a synthetic-model unit test in
/// `src/pbc/gruneisen.rs`. Here the two thermodynamic conventions must agree.
#[test]
fn gruneisen_diamond_second_order() {
    let params = params();
    let system = diamond();
    let el = electronic();
    let pbc = pbc_options(16.0, 24.0, 10.0);

    let run = |delta: f64, stencil: SecondOrderStencil| {
        let t = Instant::now();
        let g = pbc_gruneisen(
            &system,
            &params,
            &GruneisenOptions {
                delta,
                // Tie the second-order node set to `delta` so that the `delta`
                // vs `delta/2` comparison below actually moves it. With the
                // default `delta_second` the second-order step is fixed at
                // `2e-2` regardless of `delta`, which would make gate (2)
                // vacuously true. The default step is gated separately, in
                // `tests/gruneisen_second_order_delta.rs`.
                delta_second: Some(delta),
                temperatures: vec![100.0, 300.0, 1000.0],
                electronic: el.clone(),
                pbc,
                second_order: true,
                second_order_stencil: stencil,
                ..GruneisenOptions::default()
            },
        )
        .unwrap();
        println!(
            "gruneisen2 {stencil:?} delta={delta:.1e} [{:.1} s]: gamma={:.6} refit={:.6} \
             gamma2={:.6} q={:.6}  gamma2_th={:?}  gamma2_full={:?}  min_overlap={:.6}",
            t.elapsed().as_secs_f64(),
            g.mode_gamma[NDOF - 1],
            g.mode_gamma_refit[NDOF - 1],
            g.mode_gamma2[NDOF - 1],
            g.mode_q()[NDOF - 1],
            g.thermodynamic_gamma2,
            g.thermodynamic_gamma2_full,
            g.min_optical_overlap()
        );
        g
    };

    let base = run(5.0e-3, SecondOrderStencil::ThreePoint);
    let half = run(2.5e-3, SecondOrderStencil::ThreePoint);
    let five = run(2.5e-3, SecondOrderStencil::FivePoint);

    assert_eq!(
        base.second_order_stencil,
        Some(SecondOrderStencil::ThreePoint)
    );
    for i in 0..3 {
        assert!(
            base.mode_gamma2[i].is_nan() && base.mode_gamma_refit[i].is_nan(),
            "acoustic mode {i} must be excluded from the second-order fit"
        );
    }

    for i in 3..NDOF {
        // (1) the fit's own slope against the two-point central difference.
        let refit = base.mode_gamma_refit[i];
        let gamma = base.mode_gamma[i];
        let rel = ((refit - gamma) / gamma).abs();
        assert!(
            rel < 1.0e-3,
            "mode {i}: refit gamma {refit} vs central-difference {gamma} (rel {rel:.3e})"
        );

        // Physical window: at most order gamma for a hard covalent solid (GFN1
        // diamond in fact lands two orders below, gamma2 ~ -0.038).
        let g2 = base.mode_gamma2[i];
        assert!(
            g2.is_finite() && g2.abs() < 5.0,
            "mode {i}: gamma2 = {g2} is outside a sane window"
        );
        // q = gamma2 / gamma; literature |q| is O(1) for diamond.
        let q = base.mode_q()[i];
        assert!(
            q.is_finite() && q.abs() < 5.0,
            "mode {i}: q = {q} is outside a sane window"
        );

        // (2) delta convergence, and (3) stencil independence.
        let rel = ((g2 - half.mode_gamma2[i]) / g2).abs();
        assert!(
            rel < 0.03,
            "mode {i}: gamma2 not delta-converged: {g2} (delta) vs {} (delta/2), rel {rel:.3e}",
            half.mode_gamma2[i]
        );
        let rel = ((g2 - five.mode_gamma2[i]) / g2).abs();
        assert!(
            rel < 0.03,
            "mode {i}: three-point gamma2 {g2} vs five-point {} (rel {rel:.3e})",
            five.mode_gamma2[i]
        );
    }

    // Thermodynamic averages: all modes are one degenerate triplet, so the mode
    // average, the full derivative and the single mode gamma2 all coincide.
    for &t in &[100.0, 300.0, 1000.0] {
        let mode_avg = base.gamma2_at(t).expect("gamma2_th was requested");
        let full = base.gamma2_full_at(t).expect("gamma2_full was requested");
        assert!(
            (mode_avg - base.mode_gamma2[NDOF - 1]).abs() < 1.0e-12,
            "gamma2_th({t} K) = {mode_avg} should equal the single optical gamma2 {}",
            base.mode_gamma2[NDOF - 1]
        );
        assert!(
            (mode_avg - full).abs() < 1.0e-12,
            "uniform-gamma reweighting correction should vanish: {mode_avg} vs {full}"
        );
    }

    // Second order off: the second-order fields must be inert, and the
    // first-order numbers bit-for-bit what they were before the option existed.
    let plain = pbc_gruneisen(
        &system,
        &params,
        &GruneisenOptions {
            delta: 5.0e-3,
            temperatures: vec![300.0],
            electronic: el.clone(),
            pbc,
            ..GruneisenOptions::default()
        },
    )
    .unwrap();
    assert!(plain.second_order_stencil.is_none());
    assert!(plain.thermodynamic_gamma2.is_empty());
    assert!(plain.thermodynamic_gamma2_full.is_empty());
    assert!(plain.mode_gamma2.iter().all(|g| g.is_nan()));
    for i in 3..NDOF {
        assert_eq!(
            plain.mode_gamma[i], base.mode_gamma[i],
            "enabling second_order perturbed the first-order gamma of mode {i}"
        );
    }
}

/// `#[ignore]`d: the `delta`-convergence study behind the recommended
/// `delta <= 5e-3` for `second_order`, plus a cutoff-sensitivity pair. ~40
/// analytic periodic Hessians, ~10 minutes. Run with
/// `cargo test --profile reltest --test pbc_third_derivative -- --ignored --nocapture`.
///
/// It prints, for each `delta`, the first-order `gamma`, the refit `gamma`, the
/// curvature `gamma2` and `q = gamma2/gamma` of diamond's optical triplet. The
/// measured ladder is
///
/// ```text
///    stencil   delta        gamma  gamma_refit       gamma2            q     secs
/// -- lean cutoffs, AO 16 / Ewald real 24 / sr 10 Bohr --
/// ThreePoint  1.3e-3     0.905417     0.905417    -0.038331    -0.042335     19.7
/// ThreePoint  2.5e-3     0.905417     0.905417    -0.038331    -0.042335     19.3
/// ThreePoint  5.0e-3     0.905418     0.905418    -0.038336    -0.042341     29.7
/// ThreePoint  1.0e-2     0.905431     0.905433    -0.040588    -0.044828     31.8
/// ThreePoint  2.0e-2     0.905433     0.905440    -0.039025    -0.043101     28.0
/// ThreePoint  4.0e-2     0.905461     0.905492    -0.038657    -0.042693     32.2
///  FivePoint  2.5e-3     0.905417     0.905417    -0.038329    -0.042333     66.4
///  FivePoint  5.0e-3     0.905418     0.905413    -0.037586    -0.041512     65.0
///  FivePoint  1.0e-2     0.905431     0.905431    -0.041109    -0.045403     63.1
///  FivePoint  2.0e-2     0.905433     0.905423    -0.039147    -0.043236     98.6
/// -- library-default cutoffs, AO 30 / Ewald real 40 / sr 10 Bohr --
/// ThreePoint  2.5e-3     0.905417     0.905417    -0.038332    -0.042337     58.4
/// ThreePoint  5.0e-3     0.905418     0.905418    -0.038338    -0.042342     35.0
/// ```
///
/// The last two rows settle the obvious worry: `gamma2` is two orders of
/// magnitude below `gamma`, but it is **not** cutoff-limited — nearly doubling
/// both real-space cutoffs moves the plateau by `2.6e-5` / `5.2e-5` relative.
///
/// The `delta` dependence is **not** the textbook picture, and the reason is
/// worth recording. A noise-limited second difference scatters at small
/// `delta`; this one is dead
/// flat there (six digits at `delta` and `delta/2`), because the SCC and the
/// analytic Hessian are converged to ~`1e-12` in `ln lambda`. What actually
/// degrades is the *large*-`delta` end, and not as `O(delta^2)` truncation: from
/// `5e-3` to `1e-2` the value jumps by 6% and then comes back, decaying as
/// `1/delta^2` afterwards. That is the signature of a fixed **step** in
/// `ln lambda`, not of truncation — fitting `eps / delta^2` to the `1e-2`,
/// `2e-2`, `4e-2` rows gives `eps ~ 5e-7` in `ln lambda` (`~6e-4 cm^-1` on a
/// 2292 cm^-1 mode), i.e. the discreteness of the real-space image lists as the
/// cell is scaled past a shell of the fixed AO / Ewald cutoffs (the caveat noted
/// in `src/pbc/third_derivative.rs`). It is invisible in `gamma` itself and only
/// surfaces because a second difference divides it by `delta^2`.
#[test]
#[ignore]
fn gruneisen_diamond_second_order_delta_study() {
    let params = params();
    let system = diamond();
    let el = electronic();

    println!(
        "{:>10} {:>7} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "stencil", "delta", "gamma", "gamma_refit", "gamma2", "q", "secs"
    );
    let row = |delta: f64, stencil: SecondOrderStencil, pbc: PbcOptions| {
        let t = Instant::now();
        let g = pbc_gruneisen(
            &system,
            &params,
            &GruneisenOptions {
                delta,
                temperatures: vec![300.0],
                electronic: el.clone(),
                pbc,
                second_order: true,
                second_order_stencil: stencil,
                ..GruneisenOptions::default()
            },
        )
        .unwrap();
        println!(
            "{:>10} {delta:>7.1e} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>8.1}",
            format!("{stencil:?}"),
            g.mode_gamma[NDOF - 1],
            g.mode_gamma_refit[NDOF - 1],
            g.mode_gamma2[NDOF - 1],
            g.mode_q()[NDOF - 1],
            t.elapsed().as_secs_f64(),
        );
    };
    let lean = pbc_options(16.0, 24.0, 10.0);
    for &delta in &[1.25e-3, 2.5e-3, 5.0e-3, 1.0e-2, 2.0e-2, 4.0e-2] {
        row(delta, SecondOrderStencil::ThreePoint, lean);
    }
    for &delta in &[2.5e-3, 5.0e-3, 1.0e-2, 2.0e-2] {
        row(delta, SecondOrderStencil::FivePoint, lean);
    }
    // Cutoff sensitivity of the plateau: the library defaults (AO 30 Bohr, Ewald
    // real 40 Bohr) against the lean cutoffs used everywhere else in this file.
    // gamma2 is two orders of magnitude smaller than gamma, so it is far more
    // exposed to the cutoffs in *relative* terms.
    let full = pbc_options(30.0, 40.0, 10.0);
    for &delta in &[2.5e-3, 5.0e-3] {
        row(delta, SecondOrderStencil::ThreePoint, full);
    }
}

/// Strain-mixed derivative `dH/d(ln V)`: Richardson (`delta` vs `delta/2`)
/// consistency, plus an independent cross-check of the whole strain path against
/// the Grueneisen module.
///
/// The direct "agrees with a finite difference of Hessians at explicitly scaled
/// lattices" check is vacuous — that *is* the implementation — so this gates the
/// two things that are not vacuous:
///
/// 1. **Richardson.** The estimator uses the exact log-volume separation
///    `ln((1+delta)/(1-delta))`, so it is `O(delta^2)`; halving `delta` must move
///    it by far less than its own magnitude.
/// 2. **Cross-module.** First-order perturbation theory gives the mode Grueneisen
///    parameter straight from this matrix without any re-diagonalisation,
///    `gamma_k = -(1/2) u_k^T (dH^mw/d ln V) u_k / lambda_k` with `u_k, lambda_k`
///    the mass-weighted normal mode at the reference volume. That must reproduce
///    what `pbc_gruneisen` gets by diagonalising at both volumes and matching
///    modes — two genuinely different code paths through the same physics.
#[test]
fn strain_hessian_derivative_richardson_and_mode_gamma() {
    let params = params();
    let system = diamond();
    let el = electronic();
    let pbc = pbc_options(16.0, 24.0, 10.0);

    let t = Instant::now();
    let coarse = pbc_strain_hessian_derivative(&system, &params, &el, &pbc, 5.0e-3).unwrap();
    let fine = pbc_strain_hessian_derivative(&system, &params, &el, &pbc, 2.5e-3).unwrap();
    let strain_secs = t.elapsed().as_secs_f64();

    let mut err = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..NDOF {
        for j in 0..NDOF {
            err = err.max((coarse[(i, j)] - fine[(i, j)]).abs());
            scale = scale.max(fine[(i, j)].abs());
        }
    }
    println!(
        "strain dH/dlnV [{strain_secs:.1} s]: scale {scale:.4e} Eh/Bohr^2, \
         |delta - delta/2| = {err:.3e} (rel {:.3e})",
        err / scale
    );
    assert!(
        scale > 0.1 && scale.is_finite(),
        "dH/dlnV magnitude {scale:.3e} looks degenerate"
    );
    assert!(
        err / scale < 1.0e-3,
        "dH/dlnV not delta-converged: |delta - delta/2| = {err:.3e} on scale {scale:.3e}"
    );

    // Cross-module: first-order perturbation theory on the reference modes.
    let t = Instant::now();
    let h0 = pbc_gamma_hessian(&system, &params, &el, &pbc).unwrap().hessian;
    let vib = vibrational_analysis(&h0, &[6, 6]).unwrap();
    let mass = gfn1_rs::data_tables::relative_atomic_mass(6); // all atoms are carbon
    let mut hf_gamma = [0.0; NDOF];
    for k in 3..NDOF {
        // Mass-weighted eigenvector (vibrational_analysis returns mode/sqrt(m)).
        let raw: Vec<f64> = (0..NDOF).map(|i| vib.modes[k][i] * mass.sqrt()).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u: Vec<f64> = raw.iter().map(|x| x / norm).collect();
        let mut quad = 0.0;
        for i in 0..NDOF {
            for j in 0..NDOF {
                quad += u[i] * (coarse[(i, j)] / mass) * u[j];
            }
        }
        hf_gamma[k] = -0.5 * quad / vib.eigenvalues[k];
    }

    let reference = pbc_gruneisen(
        &system,
        &params,
        &GruneisenOptions {
            delta: 5.0e-3,
            temperatures: vec![300.0],
            electronic: el.clone(),
            pbc,
            ..GruneisenOptions::default()
        },
    )
    .unwrap();
    println!(
        "cross-check [{:.1} s]: perturbative gamma = {:?} vs rediagonalised {:?}",
        t.elapsed().as_secs_f64(),
        &hf_gamma[3..],
        &reference.mode_gamma[3..]
    );
    for k in 3..NDOF {
        let a = hf_gamma[k];
        let b = reference.mode_gamma[k];
        assert!(
            a.is_finite() && (0.5..=2.0).contains(&a),
            "perturbative mode {k} gamma = {a} outside the physical window 0.5..2.0"
        );
        let rel = ((a - b) / b).abs();
        assert!(
            rel < 1.0e-3,
            "perturbative vs rediagonalised gamma for mode {k}: {a} vs {b} (rel {rel:.3e})"
        );
    }
}
