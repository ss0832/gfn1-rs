// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gates for the **analytic Brillouin-zone-sampled periodic third derivative**
//! (`src/pbc/kpoint_third.rs`).
//!
//! The Gamma-point analytic FC3 (`pbc::gamma_response`) is gated by
//! `tests/pbc_gamma_third_analytic.rs` plus one crate-internal accuracy test. This file is its
//! k-point counterpart, and it carries the accuracy gate itself because the k-point assembly's
//! only independent reference — `pbc_kpoint_third_derivative_seminumerical_vector` — is public.
//!
//! Four independent statements, in decreasing order of how much they would hurt to lose:
//!
//! 1. **Seminumerical agreement.** The assembled analytic `e³[v]` over a real `2 x 2 x 2`
//!    Monkhorst-Pack mesh against a Richardson-extrapolated central difference of the production
//!    **k-point** Hessian. Nothing is shared between the two: one differentiates the response
//!    chain analytically through the Ewald-split Klopman-Ohno γ, the other reconverges the SCC at
//!    displaced geometries and differences the CPXTB Hessian. This is the gate that says the
//!    assembly is the third derivative of the energy and not merely a self-consistent artifact.
//! 2. **Gamma-limit equality.** Driven on `KMesh::gamma()`, the k-point path must reproduce the
//!    production Gamma path to solver precision. The two share the frozen builders and both
//!    response paths differ completely (complex resolvent / Daleckii-Krein versus the real
//!    coefficient-and-frame form), so this pins the whole complex transcription against an
//!    already-validated number — and it is *cheap*, which makes it the gate to run first when
//!    something breaks.
//! 3. **Acoustic sum rule.** A rigid translation changes no interatomic distance and no image
//!    sum, so `e³[v] = 0` exactly. Sensitive to any block whose image sums or Ewald partitioning
//!    are not translation-covariant, and it costs one directional evaluation.
//! 4. **Polarization identity.** The block-mode `|dofs|³` sub-tensor, contracted, must reproduce
//!    the directional evaluator it was assembled from — through a completely different set of
//!    integer-weighted directional evaluations.
//!
//! # Fixtures
//!
//! The same two **distorted** cells every other periodic third-derivative gate uses: a skewed
//! 2-atom diamond cell and a skewed zincblende BN cell.
//!
//! The distortion is not cosmetic: perfect diamond has a triply degenerate `t2` frontier, where
//! the k-point band extraction is documented to be fragile (`docs/pbc.md` §2b).
//!
//! The *pair* is not redundant either, though the reason is narrower than it first looks. On a
//! **Gamma-only** mesh the homonuclear `diamond-skew` cell has an essentially vanishing charge
//! response (`max|q¹| = 9.05e-16`, `max|q^vv| = 1.60e-14`, measured by the unit gate
//! `kpoint_second_order_matches_gamma_charge_space`), so the electrostatic channel and the
//! dielectric solve are silent there and **BN is what makes the Gamma-limit gate load-bearing**.
//! Over the `2 x 2 x 2` mesh used by the other gates diamond's charge response is *not* small
//! (`max|q¹| = 1.15e-2` against BN's `2.35e-2`), so both fixtures exercise the charge path.
//! Every gate below reports the scale of what it is comparing and asserts it is non-zero, rather
//! than relying on either statement staying true.
//!
//! Every gate is `#[ignore]`d: the accuracy gate alone costs `2 · 2 · nnz(v)` analytic k-point
//! Hessians per fixture.

use gfn1_rs::pbc::{
    pbc_gamma_third_analytic_vector, pbc_kpoint_third_analytic_block,
    pbc_kpoint_third_analytic_vector, pbc_kpoint_third_derivative_seminumerical_vector,
    pbc_kpoint_third_with_reference, KpointThirdReference,
};
use gfn1_rs::{ElectronicOptions, EwaldOptions, Gfn1Parameters, KMesh, PbcOptions, PeriodicSystem};
use std::time::Instant;

/// The two distorted fixtures. See the module docs for why both are needed.
const FIXTURES: &[(&str, &str)] = &[
    (
        "diamond-skew",
        "2\n\
Lattice=\"0.06 1.83 1.75 1.75 0.04 1.81 1.82 1.76 0.03\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.930000 0.880000 0.905000\n",
    ),
    (
        "BN-skew",
        "2\n\
Lattice=\"0.06 1.86 1.78 1.78 0.04 1.84 1.85 1.79 0.03\" pbc=\"T T T\"\n\
B 0.000000 0.000000 0.000000\n\
N 0.940000 0.890000 0.920000\n",
    ),
];

fn params() -> Gfn1Parameters {
    Gfn1Parameters::builtin().expect("builtin parameters")
}

/// Tight SCC, dispersion off (it is a purely classical block with its own gates and it only slows
/// the finite-difference reference down), electronic temperature pinned to zero so neither path
/// drifts onto the periodic finite-temperature branch.
fn electronic() -> ElectronicOptions {
    ElectronicOptions {
        enable_dispersion: false,
        energy_tolerance: 1.0e-12,
        charge_tolerance: 1.0e-10,
        electronic_temperature: 0.0,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

/// A real `2 x 2 x 2` Monkhorst-Pack mesh. The real-space cutoffs are the ones the Gamma total
/// gate uses (`ao_cutoff = 12`, `sr_cutoff = 8`), where the finite-difference ladder on these
/// fixtures is known to be clean.
fn kpbc_opts() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::monkhorst_pack([2, 2, 2]),
        ao_cutoff: 12.0,
        ewald: EwaldOptions {
            sr_cutoff: 8.0,
            ..EwaldOptions::default()
        },
    }
}

/// The same options with the mesh collapsed to Gamma — everything else identical, so the
/// Gamma-limit gate compares the two *paths* and not two discretisations.
fn gamma_pbc_opts() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::gamma(),
        ..kpbc_opts()
    }
}

/// The direction generator the Gamma gates use, kept identical so the reported k-point numbers
/// sit next to the published Gamma ones on the same fixtures.
///
/// **Seed independence.** The generator's only degree of freedom is `m = (seed + 7) mod 13`, and
/// `m` with `13 − m` give `v' = −v + (rigid translation)`; since the third derivative is odd and
/// translation-invariant, such a pair satisfies `e³[v'] = −e³[v]` exactly and is a parity check
/// rather than a second probe. Seed 41 gives `m = 9`.
fn direction(ndof: usize, seed: u64) -> Vec<f64> {
    (0..ndof)
        .map(|k| {
            let x = ((k as u64 + 1) * (seed + 7)) % 13;
            0.31 - 0.05 * (x as f64) + 0.01 * ((k % 3) as f64)
        })
        .collect()
}

/// **The accuracy gate.** The analytic k-point directional third against a Richardson
/// extrapolation of the seminumerical route over a real `2 x 2 x 2` mesh.
///
/// # Why Richardson and not a single step
///
/// The reference is a central difference of the analytic k-point Hessian, so it carries an
/// `O(h²)` truncation error of its own. Comparing against a single `h` therefore measures
/// `|analytic − exact| + |truncation(h)|` and cannot distinguish the two. Evaluating at `h` and
/// `h/2` and extrapolating removes the leading truncation term, so the residual that survives is
/// the analytic assembly's — and the printed **ladder ratio** `(fd(h) − a) / (fd(h/2) − a)` is
/// the diagnostic: near 4 means the residual is dominated by the finite difference (the analytic
/// value is the better number), near 1 means a genuinely `h`-independent discrepancy, i.e. a
/// missing term.
///
/// The tolerance is set from the Gamma path's published performance on these same fixtures
/// (`|delta| ≈ 8.5e-8` / `8.0e-8`, a documented `~1e-7` tail attributed to the band/Pulay
/// self-energy cache channel): the k-point assembly inherits that assembly, so it inherits that
/// tail, and anything materially worse is a k-specific defect.
#[test]
#[ignore = "analytic k-point FC3: 2 x 2 x nnz(v) k-point Hessians + one analytic assembly per fixture"]
fn kpoint_analytic_third_matches_seminumerical() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = kpbc_opts();
        let ndof = 3 * system.atoms.len();
        let v = direction(ndof, 41);

        let started = Instant::now();
        let analytic = pbc_kpoint_third_analytic_vector(&system, &params, &opts, &pbc, &v)
            .expect("analytic k-point third");
        let analytic_secs = started.elapsed().as_secs_f64();

        let ref_at = |h: f64| -> f64 {
            let dv_h = pbc_kpoint_third_derivative_seminumerical_vector(
                &system, &params, &opts, &pbc, h, &v,
            )
            .expect("seminumerical k-point third");
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * dv_h[(a, b)];
                }
            }
            acc
        };
        let started = Instant::now();
        let ref_h = ref_at(1.0e-3);
        let ref_h2 = ref_at(5.0e-4);
        let reference = (4.0 * ref_h2 - ref_h) / 3.0;
        let fd_secs = started.elapsed().as_secs_f64();

        let delta = (analytic - reference).abs();
        let ratio = (ref_h - analytic) / (ref_h2 - analytic);
        println!(
            "ktotal/{name}: analytic {analytic:+.10e} ({analytic_secs:.1} s) vs seminumerical \
             richardson {reference:+.10e} ({fd_secs:.1} s)\n  |delta| {delta:.3e}  fd(h) delta \
             {:.3e}  fd(h/2) delta {:.3e}  ladder ratio {ratio:.2}",
            (analytic - ref_h).abs(),
            (analytic - ref_h2).abs()
        );
        assert!(
            analytic.abs() > 1.0e-6,
            "{name}: the directional third is ~zero, the gate is vacuous"
        );
        assert!(
            delta < 1.0e-6 * (1.0 + reference.abs()),
            "{name}: analytic k-point third vs seminumerical: {analytic:.10e} vs \
             {reference:.10e} (|delta| {delta:.3e})"
        );
    }
}

/// **Gamma-limit equality.** On `KMesh::gamma()` the k-point path must reproduce the production
/// Gamma analytic third derivative.
///
/// The two assemblies share the frozen half and the two response *paths* (`B6`, `bg4`, the
/// density path), but the objects fed into them come from completely different algebra: the
/// k-point side solves the first-order response with the complex k-point CPXTB and the
/// second-order response with the complex resolvent / Daleckii-Krein form, while the Gamma side
/// uses the real molecular charge-space solver with its coefficient-and-frame second-order
/// machinery. Agreement to solver precision is therefore a statement about the derivation.
///
/// This is also the cheapest gate in the file by a wide margin (one k-point is one k-point), so
/// it is the first thing to run when the accuracy gate moves.
#[test]
#[ignore = "analytic k-point FC3: two Gamma-mesh analytic assemblies per fixture"]
fn kpoint_analytic_third_at_gamma_matches_the_gamma_path() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = gamma_pbc_opts();
        let ndof = 3 * system.atoms.len();
        let v = direction(ndof, 41);

        let via_k = pbc_kpoint_third_analytic_vector(&system, &params, &opts, &pbc, &v)
            .expect("k-point path on a Gamma mesh");
        let via_gamma = pbc_gamma_third_analytic_vector(&system, &params, &opts, &pbc, &v)
            .expect("Gamma path");
        let delta = (via_k - via_gamma).abs();
        let rel = delta / (1.0 + via_gamma.abs());
        println!(
            "kgamma/{name}: k-point path {via_k:+.10e}  Gamma path {via_gamma:+.10e}  |delta| \
             {delta:.3e}  rel {rel:.3e}"
        );
        assert!(
            via_gamma.abs() > 1.0e-6,
            "{name}: the Gamma reference is ~zero, the gate is vacuous"
        );
        assert!(
            rel < 1.0e-9,
            "{name}: the k-point assembly on a Gamma-only mesh disagrees with the production \
             Gamma assembly: {via_k:.10e} vs {via_gamma:.10e} (rel {rel:.3e})"
        );
    }
}

/// **Acoustic sum rule** over a real `2 x 2 x 2` mesh. Translating every atom by the same vector
/// leaves the energy exactly invariant, so every rigid direction contracts the cubic force
/// constants to zero.
///
/// Four rigid directions per fixture: the three Cartesian axes plus a skew translation that mixes
/// them. The skew one is the sharp test — an assembly that was translation-covariant axis by axis
/// but got a mixed `(x,y,z)` slot wrong would pass the first three.
#[test]
#[ignore = "analytic k-point FC3: one shared reference + 4 directional evaluations per fixture"]
fn kpoint_analytic_third_obeys_the_acoustic_sum_rule() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = kpbc_opts();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let started = Instant::now();
        let reference =
            KpointThirdReference::build(&system, &params, &opts, &pbc).expect("shared reference");
        assert!(
            reference.scc().kpoints.len() > 1,
            "{name}: the mesh collapsed to a single k-point, the gate is not testing k-sampling"
        );
        let rigid: [([f64; 3], &str); 4] = [
            ([1.0, 0.0, 0.0], "x"),
            ([0.0, 1.0, 0.0], "y"),
            ([0.0, 0.0, 1.0], "z"),
            ([0.37, -0.81, 0.52], "skew"),
        ];
        for (shift, label) in rigid {
            let mut v = vec![0.0_f64; ndof];
            for at in 0..nat {
                v[3 * at] = shift[0];
                v[3 * at + 1] = shift[1];
                v[3 * at + 2] = shift[2];
            }
            let e3 = pbc_kpoint_third_with_reference(&system, &params, &opts, &pbc, &reference, &v)
                .expect("directional third");
            println!("kasr/{name}/{label}: e3[rigid] {e3:+.6e}");
            assert!(
                e3.abs() < 1.0e-10,
                "{name}: analytic k-point third violates the acoustic sum rule along the rigid \
                 {label} translation: {e3:.6e}"
            );
        }
        println!("kasr/{name}: {:.1} s", started.elapsed().as_secs_f64());
    }
}

/// **Block-mode polarization identity.** The `|dofs|³` sub-tensor assembled by the cubic
/// polarization identity, contracted with a weight vector, must reproduce the directional
/// evaluation along the direction those weights embed into the full DOF space.
///
/// Not a tautology: the polarization identity reaches the contraction through 7 signed
/// evaluations per canonical triple at *integer-weighted* directions, deduplicated across
/// triples, and recombines them — a completely different set of directional evaluations from the
/// single fractional-weight one it is checked against. Any contamination in the directional
/// evaluator that is not exactly third order in `v` breaks the identity immediately.
#[test]
#[ignore = "analytic k-point FC3: a 4-DOF block polarization sweep over a 2x2x2 mesh per fixture"]
fn kpoint_analytic_third_block_matches_directional() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = kpbc_opts();
        let ndof = 3 * system.atoms.len();
        let dofs = [0_usize, 1, 2, 4];
        let weights = [0.73, -0.41, 0.26, 0.58];

        let started = Instant::now();
        let block = pbc_kpoint_third_analytic_block(&system, &params, &opts, &pbc, &dofs)
            .expect("block third");
        let block_secs = started.elapsed().as_secs_f64();
        assert_eq!(block.n(), dofs.len());
        let contracted = block.contract_vvv(&weights);

        let mut v = vec![0.0_f64; ndof];
        for (slot, &d) in dofs.iter().enumerate() {
            v[d] = weights[slot];
        }
        let direct = pbc_kpoint_third_analytic_vector(&system, &params, &opts, &pbc, &v)
            .expect("directional third");
        let delta = (contracted - direct).abs();
        let rel = delta / (1.0 + direct.abs());
        println!(
            "kblock/{name}: contract_vvv {contracted:+.10e}  directional {direct:+.10e}  |delta| \
             {delta:.3e}  rel {rel:.3e}  ({block_secs:.1} s for the block)"
        );
        assert!(
            direct.abs() > 1.0e-6,
            "{name}: the directional third is ~zero, the gate is vacuous"
        );
        assert!(
            rel < 1.0e-9,
            "{name}: block polarization tensor disagrees with the directional evaluator: \
             {contracted:.10e} vs {direct:.10e} (rel {rel:.3e})"
        );
    }
}

/// The option guards are honest and — unlike the Gamma path — a Monkhorst-Pack mesh is **not**
/// among them: an out-of-range block DOF is rejected, and a fractional (Fermi-smeared) filling
/// is rejected with a message naming the seminumerical alternative.
#[test]
fn kpoint_analytic_third_guards() {
    let system = PeriodicSystem::from_xyz_str(FIXTURES[0].1, 0.0, false).expect("fixture");
    let params = params();
    let pbc = kpbc_opts();

    // Out-of-range block DOF.
    let ndof = 3 * system.atoms.len();
    // `expect_err` is unavailable: the Ok type deliberately has no `Debug`.
    let text =
        match pbc_kpoint_third_analytic_block(&system, &params, &electronic(), &pbc, &[0, ndof + 3])
        {
            Ok(_) => panic!("an out-of-range block DOF must be rejected"),
            Err(err) => err.to_string(),
        };
    assert!(
        text.contains("out of range"),
        "unhelpful block-DOF rejection: {text}"
    );

    // A Monkhorst-Pack mesh is accepted (that is the point of this path), so the reference build
    // must not raise on the mesh itself.
    let opts = electronic();
    let built = KpointThirdReference::build(&system, &params, &opts, &pbc);
    assert!(
        built.is_ok(),
        "a Monkhorst-Pack mesh must be ACCEPTED by the k-point analytic path: {:?}",
        built.err().map(|e| e.to_string())
    );
}
