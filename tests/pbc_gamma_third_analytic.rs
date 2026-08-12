// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration gates for the **analytic Gamma-point periodic third derivative**
//! (`src/pbc/gamma_response.rs`): the acoustic sum rule, and the consistency of
//! the dense / block polarization modes with the directional evaluator they are
//! built from.
//!
//! The *accuracy* gate lives next to the assembly as a unit test
//! (`gamma_analytic_third_matches_seminumerical`), where it can reach the
//! crate-internal diagnostics; it pins the assembled directional third against
//! a Richardson-extrapolated finite difference of the production periodic
//! Hessian. What is gated HERE is everything a caller of the public API can
//! check without a finite difference:
//!
//! 1. **Acoustic sum rule** — a rigid translation of the whole cell changes no
//!    interatomic distance and no image sum, so `e³[v] = 0` identically for
//!    every rigid `v`. This is an exact invariance, not a converged quantity:
//!    it is sensitive to any block whose image sums or Ewald partitioning are
//!    not translation-covariant, and it costs a single directional evaluation.
//! 2. **Polarization identity** — the cubic polarization identity that recovers
//!    `T_abc` from directional evaluations must reproduce, on contraction, the
//!    very directional numbers it was assembled from. Gated in both the block
//!    (`|dofs|³` sub-tensor) and dense (full `n³`) modes.
//!
//! # Fixtures
//!
//! Both gates run on the same two **distorted** cells the assembly's own gates
//! use: a skewed 2-atom diamond cell and a skewed zincblende BN cell. The
//! distortion is deliberate — perfect diamond has a triply degenerate `t2`
//! frontier at Gamma, where the arbitrary in-block eigenbasis makes the
//! periodic CPXTB response basis-dependent (see `docs/pbc.md`). BN additionally
//! exercises the heteronuclear Ewald / Klopman-Ohno charge path, which the
//! homonuclear diamond cell leaves almost silent.
//!
//! Every gate is `#[ignore]`d: one directional evaluation costs a periodic SCC
//! plus two image-summed skeleton builds, and the dense mode multiplies that by
//! `C(n+2,3)` directions.

use gfn1_rs::pbc::{
    pbc_gamma_third_analytic_block, pbc_gamma_third_analytic_dense,
    pbc_gamma_third_analytic_vector, pbc_gamma_third_with_reference, GammaThirdReference,
};
use gfn1_rs::{ElectronicOptions, EwaldOptions, Gfn1Parameters, KMesh, PbcOptions, PeriodicSystem};
use std::time::Instant;

/// The two distorted fixtures of the assembly's own gates.
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

fn electronic() -> ElectronicOptions {
    ElectronicOptions {
        enable_dispersion: false,
        energy_tolerance: 1.0e-12,
        charge_tolerance: 1.0e-10,
        ..ElectronicOptions::default()
    }
}

fn pbc_opts() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::gamma(),
        ao_cutoff: 12.0,
        ewald: EwaldOptions {
            sr_cutoff: 8.0,
            ..EwaldOptions::default()
        },
        ..PbcOptions::default()
    }
}

/// A reproducible pseudo-random direction (the same generator the assembly's
/// unit gates use, so the fixtures stay comparable).
///
/// **Choosing seeds that are actually independent.** The generator's only real
/// degree of freedom is `m = (seed + 7) mod 13`, so two seeds sharing an `m`
/// give the *same* direction. Worse, `m` and `13 − m` give
/// `x' = 13 − x`, hence `v' = −v + t` with `t` a rigid translation — and since
/// the third derivative is odd and translation-invariant, `e³[v'] = −e³[v]`
/// exactly. Such a pair is a parity check, not a second probe. The seeds used
/// below are `11 → m = 5`, `29 → m = 10`, `5 → m = 12`: distinct, none zero,
/// and no two summing to 13.
fn direction(ndof: usize, seed: u64) -> Vec<f64> {
    (0..ndof)
        .map(|k| {
            let x = ((k as u64 + 1) * (seed + 7)) % 13;
            0.31 - 0.05 * (x as f64) + 0.01 * ((k % 3) as f64)
        })
        .collect()
}

/// **Acoustic sum rule.** Translating every atom of the cell by the same vector
/// leaves the energy exactly invariant, so every rigid direction must contract
/// the cubic force constants to zero: `Σ_abc T_abc v_a v_b v_c = 0`.
///
/// Four rigid directions per fixture — the three Cartesian axes, plus a skew
/// translation that mixes them (the skew one is the sharp test: an assembly
/// that happened to be translation-covariant axis by axis, but got a mixed
/// `(x,y,z)` third-derivative slot wrong, would pass the first three).
#[test]
#[ignore = "periodic analytic FC3: one shared reference + 4 directional evaluations per fixture"]
fn gamma_analytic_third_obeys_the_acoustic_sum_rule() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = pbc_opts();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let started = Instant::now();
        let reference =
            GammaThirdReference::build(&system, &params, &opts, &pbc).expect("shared reference");
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
            let e3 = pbc_gamma_third_with_reference(&system, &params, &opts, &pbc, &reference, &v)
                .expect("directional third");
            println!("asr/{name}/{label}: e3[rigid] {e3:+.6e}");
            assert!(
                e3.abs() < 1.0e-10,
                "{name}: analytic Gamma third violates the acoustic sum rule along the rigid \
                 {label} translation: {e3:.6e}"
            );
        }
        println!("asr/{name}: {:.1} s", started.elapsed().as_secs_f64());
    }
}

/// **Block-mode polarization identity.** The `|dofs|³` sub-tensor assembled by
/// the cubic polarization identity, contracted with a weight vector `w`, must
/// reproduce the directional evaluation along the very direction `w` embeds
/// into the full DOF space.
///
/// This is not a tautology: the polarization identity reaches the contraction
/// through 7 signed evaluations per canonical triple at *integer-weighted*
/// directions, deduplicated across triples, and then recombines them — a
/// completely different set of directional evaluations from the single
/// fractional-weight one it is checked against. Any non-cubic-homogeneous
/// contamination in the directional evaluator (a term that is not exactly
/// third order in `v`) breaks this identity immediately.
#[test]
#[ignore = "periodic analytic FC3: a 4-DOF block polarization sweep per fixture"]
fn gamma_analytic_third_block_matches_directional() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = pbc_opts();
        let ndof = 3 * system.atoms.len();
        let dofs = [0_usize, 1, 2, 4];
        let weights = [0.73, -0.41, 0.26, 0.58];

        let started = Instant::now();
        let block = pbc_gamma_third_analytic_block(&system, &params, &opts, &pbc, &dofs)
            .expect("block third");
        let block_secs = started.elapsed().as_secs_f64();
        assert_eq!(block.n(), dofs.len());
        let contracted = block.contract_vvv(&weights);

        let mut v = vec![0.0_f64; ndof];
        for (slot, &d) in dofs.iter().enumerate() {
            v[d] = weights[slot];
        }
        let direct = pbc_gamma_third_analytic_vector(&system, &params, &opts, &pbc, &v)
            .expect("directional third");
        let delta = (contracted - direct).abs();
        let rel = delta / (1.0 + direct.abs());
        println!(
            "block/{name}: contract_vvv {contracted:+.10e}  directional {direct:+.10e}  |delta| \
             {delta:.3e}  rel {rel:.3e}  ({block_secs:.1} s for the block)"
        );
        assert!(
            rel < 1.0e-9,
            "{name}: block polarization tensor disagrees with the directional evaluator: \
             {contracted:.10e} vs {direct:.10e} (rel {rel:.3e})"
        );
    }
}

/// **Dense-mode polarization identity.** The full `n³` tensor, contracted along
/// three unrelated random directions, must reproduce the directional evaluator
/// on each.
///
/// Three contractions rather than an entry-by-entry comparison: recovering every
/// `(a,b,c)` from directional evaluations would cost a second full polarization
/// sweep, while three independent random directions already probe all
/// `C(n+2,3)` packed slots simultaneously (a random contraction agrees only if
/// essentially every entry does). See [`direction`] for why the three seeds are
/// genuinely independent rather than related by parity and a rigid translation.
#[test]
#[ignore = "periodic analytic FC3: a full dense polarization sweep (~C(n+2,3) directions) per fixture"]
fn gamma_analytic_third_dense_matches_directional() {
    for (name, xyz) in FIXTURES {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture");
        let params = params();
        let opts = electronic();
        let pbc = pbc_opts();
        let ndof = 3 * system.atoms.len();

        let started = Instant::now();
        let dense =
            pbc_gamma_third_analytic_dense(&system, &params, &opts, &pbc).expect("dense third");
        let dense_secs = started.elapsed().as_secs_f64();
        assert_eq!(dense.n(), ndof);

        let reference =
            GammaThirdReference::build(&system, &params, &opts, &pbc).expect("shared reference");
        for seed in [11_u64, 29, 5] {
            let v = direction(ndof, seed);
            let contracted = dense.contract_vvv(&v);
            let direct =
                pbc_gamma_third_with_reference(&system, &params, &opts, &pbc, &reference, &v)
                    .expect("directional third");
            let rel = (contracted - direct).abs() / (1.0 + direct.abs());
            println!(
                "dense/{name}/seed{seed}: contract_vvv {contracted:+.10e}  directional \
                 {direct:+.10e}  rel {rel:.3e}"
            );
            assert!(
                rel < 1.0e-9,
                "{name}: dense polarization tensor disagrees with the directional evaluator \
                 along direction {seed}: {contracted:.10e} vs {direct:.10e} (rel {rel:.3e})"
            );
        }
        println!("dense/{name}: {dense_secs:.1} s for the sweep");
    }
}

/// The option guards are honest: a Monkhorst-Pack mesh is rejected with a
/// message that names the seminumerical k-point alternative, rather than
/// silently returning the Gamma answer.
#[test]
fn gamma_analytic_third_rejects_a_kpoint_mesh() {
    let system = PeriodicSystem::from_xyz_str(FIXTURES[0].1, 0.0, false).expect("fixture");
    let params = params();
    let opts = electronic();
    let pbc = PbcOptions {
        kmesh: KMesh::monkhorst_pack([2, 2, 2]),
        ..pbc_opts()
    };
    // `expect_err` is unavailable: the Ok type deliberately has no `Debug`
    // (it owns the whole SCC reference state).
    let text = match GammaThirdReference::build(&system, &params, &opts, &pbc) {
        Ok(_) => panic!("a Monkhorst-Pack mesh must be rejected, not silently run at Gamma"),
        Err(err) => err.to_string(),
    };
    assert!(
        text.contains("Gamma-only") && text.contains("seminumerical"),
        "unhelpful k-mesh rejection: {text}"
    );
}
