// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates for the periodic FINITE-TEMPERATURE (Fermi-smearing) coupled-perturbed
//! response.
//!
//! The smeared periodic response used to close its SCC self-consistency with a
//! damped fixed-point iteration under a hard 50-round cap, with no convergence
//! flag and no post-loop residual check: when the iteration did not converge, the
//! unconverged charge response flowed silently into the density, the
//! energy-weighted density and the Hessian. It is now ONE direct charge-space
//! dielectric solve `(I − χ⁰K) δq = δq_bare` (see
//! `pbc::hessian::PeriodicChargeDielectric`), which is exact and self-verified.
//!
//! Fixture rationale. The old iteration's convergence factor is the spectral
//! radius of `M = (1 − m) I + m χ⁰K` with mixing `m = 0.35`, so it converges only
//! for `χ⁰K` eigenvalues in `(−4.71, 1)`. Alkali cells are comfortably inside
//! that window (bcc Li at 30000 K: `ρ(M) = 0.6504`), which is why the pre-existing
//! bcc-Li gates never caught the defect. A 2-atom Ni cell at 3000 K has genuinely
//! fractional d-band occupations AND `ρ(M) = 4.55`: the old iteration diverged
//! there by 22 orders of magnitude (measured max |legacy − exact| shell-charge
//! response `1.98e22`) while reporting nothing. That is the fixture used here,
//! deliberately displaced off the ideal lattice site: the cubic site symmetry
//! would impose exact band degeneracies, which are a separate (second-order)
//! open problem and would confound this gate.

use gfn1_rs::linalg::Matrix;
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::hessian::{
    gamma_cpxtb_density_responses, gamma_mos, gamma_skeleton_derivatives, GammaMos,
    GammaSkeletonDerivatives,
};
use gfn1_rs::pbc::scf::PbcSccResult;
use gfn1_rs::pbc::KMesh;
use gfn1_rs::{
    pbc_analytic_gradient, pbc_gamma_hessian, pbc_kpoint_hessian, run_pbc_scc, ElectronicOptions,
    Gfn1Parameters, PbcOptions, PeriodicSystem,
};

/// 2-atom Ni cell (bcc-like, `a = 3.52 Å`), one atom displaced off the ideal site.
/// Metallic-in-character: at 3000 K it carries ten genuinely fractional bands.
const NI2: &str = "2\nLattice=\"3.52 0 0 0 3.52 0 0 0 3.52\" pbc=\"T T T\"\n\
     Ni 0.120000 0.040000 0.000000\n\
     Ni 1.760000 1.760000 1.760000\n";

/// Gapped molecular-crystal chain of H2 units: a wide-gap insulator whose Fermi
/// occupations are integer to well below `f64` resolution at 300 K.
const H2_CHAIN: &str = "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
     H 0.000000 0.300000 0.000000\n\
     H 0.950000 -0.200000 0.000000\n";

fn params() -> Option<Gfn1Parameters> {
    Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
}

fn smeared(temperature: f64) -> ElectronicOptions {
    ElectronicOptions {
        enable_dispersion: false,
        electronic_temperature: temperature,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

fn shift(system: &mut PeriodicSystem, dof: usize, delta: f64) {
    let atom = dof / 3;
    match dof % 3 {
        0 => system.atoms[atom].position.x += delta,
        1 => system.atoms[atom].position.y += delta,
        _ => system.atoms[atom].position.z += delta,
    }
}

fn component(v: Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Number of genuinely fractional Gamma-point band occupations.
fn fractional_band_count(mos: &GammaMos) -> usize {
    mos.occupations
        .iter()
        .filter(|&&f| f > 1.0e-6 && f < 2.0 - 1.0e-6)
        .count()
}

/// Mulliken shell-charge response `dq_s/dR` from the AO density response and the
/// overlap derivative: `dq_s = -sum_{nu in s} [ (dP S)_nu,nu + (P0 dS)_nu,nu ]`,
/// the same contraction the Hessian assembly applies internally.
fn mulliken_shell_charge_response(
    scf: &PbcSccResult,
    mos: &GammaMos,
    density_response: &Matrix,
    overlap_deriv: &Matrix,
) -> Vec<f64> {
    let n = scf.basis.aos.len();
    let mut ground = Matrix::zeros(n, n);
    for (ik, kp) in scf.kpoints.iter().enumerate() {
        for i in 0..n {
            for j in 0..n {
                ground[(i, j)] += kp.weight * scf.density_k[ik].re[(i, j)];
            }
        }
    }
    let mut out = vec![0.0_f64; scf.basis.shells.len()];
    for nu in 0..n {
        let mut population = 0.0;
        for kappa in 0..n {
            population += density_response[(nu, kappa)] * mos.overlap[(kappa, nu)]
                + ground[(nu, kappa)] * overlap_deriv[(kappa, nu)];
        }
        out[scf.basis.aos[nu].shell_index] -= population;
    }
    out
}

struct GammaResponseSetup {
    scf: PbcSccResult,
    mos: GammaMos,
    skeleton: GammaSkeletonDerivatives,
    density_response: Vec<Matrix>,
}

fn gamma_response_setup(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    opts: &ElectronicOptions,
    pbc: &PbcOptions,
) -> GammaResponseSetup {
    let scf = run_pbc_scc(system, params, opts, pbc).unwrap();
    let skeleton = gamma_skeleton_derivatives(system, params, &scf, opts, pbc).unwrap();
    let mos = gamma_mos(&scf, scf.nelec).unwrap();
    let (density_response, _weighted) =
        gamma_cpxtb_density_responses(&scf, &skeleton, &mos).unwrap();
    GammaResponseSetup {
        scf,
        mos,
        skeleton,
        density_response,
    }
}

// GATE (a). The smeared Gamma-point shell-charge response dq/dR must reproduce the
// central finite difference of the RECONVERGED periodic SCC shell charges. This is
// the direct gate on the dielectric solve itself: dq is exactly the vector the
// solve returns (screened by K and fed back into the density), so a wrong
// self-consistent charge response shows up here undiluted.
#[test]
fn gamma_finite_t_shell_charge_response_matches_scc_fd() {
    let Some(params) = params() else {
        return;
    };
    let base = PeriodicSystem::from_xyz_str(NI2, 0.0, false).unwrap();
    let opts = ElectronicOptions {
        charge_tolerance: 1.0e-12,
        energy_tolerance: 1.0e-12,
        max_scc: 800,
        ..smeared(3000.0)
    };
    let pbc = PbcOptions::default();
    let setup = gamma_response_setup(&base, &params, &opts, &pbc);

    // The fixture must actually be smeared, or the test silently degenerates into
    // a T = 0 gate on the integer occ-virt path.
    let fractional = fractional_band_count(&setup.mos);
    assert!(
        fractional >= 4,
        "fixture is not genuinely smeared: only {fractional} fractional bands"
    );
    assert!(
        setup.scf.electronic_entropy_term.abs() > 1.0e-3,
        "entropy term too small to stress finite-T: {}",
        setup.scf.electronic_entropy_term
    );

    let nsh = setup.scf.basis.shells.len();
    let ndof = 3 * base.atoms.len();
    let charges = |system: &PeriodicSystem| -> Vec<f64> {
        run_pbc_scc(system, &params, &opts, &pbc)
            .unwrap()
            .shell_charges
    };
    // Single documented step. The reference is a RECONVERGED SCC, so its noise
    // floor (~1e-12 in the shell charges at this tolerance) enters the difference
    // quotient as noise/2h; the measured h-ladder is
    //   h      4e-4      2e-4       1e-4       5e-5      2.5e-5
    //   diff   1.03e-9   3.43e-10   3.09e-10   8.14e-10  1.46e-9
    // i.e. truncation-limited above h = 2e-4 and noise-limited below h = 1e-4,
    // with the optimum ~3e-10 in between. h = 1e-4 sits at that optimum. (At the
    // default charge_tolerance = 1e-9 the whole ladder is noise, 1e-6 to 1e-5,
    // which is why this gate tightens the SCC instead of loosening its bound.)
    let h = 1.0e-4;
    let mut max_diff = 0.0_f64;
    for y in 0..ndof {
        let dq = mulliken_shell_charge_response(
            &setup.scf,
            &setup.mos,
            &setup.density_response[y],
            &setup.skeleton.overlap[y],
        );
        let mut plus = base.clone();
        let mut minus = base.clone();
        shift(&mut plus, y, h);
        shift(&mut minus, y, -h);
        let qp = charges(&plus);
        let qm = charges(&minus);
        for s in 0..nsh {
            let fd = (qp[s] - qm[s]) / (2.0 * h);
            max_diff = max_diff.max((dq[s] - fd).abs());
        }
    }
    assert!(
        max_diff < 1.0e-8,
        "smeared Gamma shell-charge response vs SCC FD max diff {max_diff:.3e}"
    );
}

// GATE (b). The smeared Gamma-point Hessian must match the central finite
// difference of the (independently FD-verified, occupation-stationary) analytic
// periodic free-energy gradient on the same fixture. The response enters every
// Hessian column, so this is the end-to-end gate.
#[test]
fn gamma_finite_t_hessian_matches_gradient_fd() {
    let Some(params) = params() else {
        return;
    };
    let base = PeriodicSystem::from_xyz_str(NI2, 0.0, false).unwrap();
    let opts = smeared(3000.0);
    let pbc = PbcOptions::default();
    let result = pbc_gamma_hessian(&base, &params, &opts, &pbc).unwrap();
    assert!(
        result.scf.electronic_entropy_term.abs() > 1.0e-3,
        "entropy term too small to stress finite-T: {}",
        result.scf.electronic_entropy_term
    );
    let nat = base.atoms.len();
    let h = 1.0e-4;
    let grad = |system: &PeriodicSystem| {
        pbc_analytic_gradient(system, &params, &opts, &pbc)
            .unwrap()
            .gradient
    };
    let mut max_diff = 0.0_f64;
    for y in 0..3 * nat {
        let mut plus = base.clone();
        let mut minus = base.clone();
        shift(&mut plus, y, h);
        shift(&mut minus, y, -h);
        let gp = grad(&plus);
        let gm = grad(&minus);
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
            }
        }
    }
    // Measured 5.32e-11 (the old fixed point produced garbage of order 1e22 on
    // this fixture, see the module header).
    assert!(
        max_diff < 1.0e-9,
        "smeared Gamma Hessian vs gradient FD max diff {max_diff:.3e}"
    );
}

// GATE (b'), k-point. The same fixture on a [2,2,2] Monkhorst-Pack mesh exercises
// the complex k-point finite-T response, whose charge-space dielectric is built
// from the k-SUMMED susceptibility (real nsh x nsh, because the perturbing
// potential and the Brillouin-zone-summed Mulliken charges are real even though
// every per-k response operator is complex).
#[test]
fn kpoint_finite_t_hessian_matches_gradient_fd() {
    let Some(params) = params() else {
        return;
    };
    let base = PeriodicSystem::from_xyz_str(NI2, 0.0, false).unwrap();
    let opts = smeared(3000.0);
    let pbc = PbcOptions {
        kmesh: KMesh::monkhorst_pack([2, 2, 2]),
        ..PbcOptions::default()
    };
    let result = pbc_kpoint_hessian(&base, &params, &opts, &pbc).unwrap();
    assert!(
        result.scf.electronic_entropy_term.abs() > 1.0e-3,
        "entropy term too small to stress finite-T: {}",
        result.scf.electronic_entropy_term
    );
    let nat = base.atoms.len();
    let h = 1.0e-4;
    let grad = |system: &PeriodicSystem| {
        pbc_analytic_gradient(system, &params, &opts, &pbc)
            .unwrap()
            .gradient
    };
    let mut max_diff = 0.0_f64;
    for y in 0..3 * nat {
        let mut plus = base.clone();
        let mut minus = base.clone();
        shift(&mut plus, y, h);
        shift(&mut minus, y, -h);
        let gp = grad(&plus);
        let gm = grad(&minus);
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
            }
        }
    }
    // Measured 6.20e-12.
    assert!(
        max_diff < 1.0e-9,
        "smeared k-point Hessian vs gradient FD max diff {max_diff:.3e}"
    );
}

// GATE (c). No regression on gapped systems. A wide-gap insulator at 300 K has
// integer occupations to below f64 resolution, so it must take the T = 0 integer
// occ-virt CPXTB path and produce a BIT-IDENTICAL Hessian. The dielectric route is
// only reached when a band is genuinely fractional.
#[test]
fn gapped_insulator_at_300k_is_identical_to_zero_temperature() {
    let Some(params) = params() else {
        return;
    };
    let base = PeriodicSystem::from_xyz_str(H2_CHAIN, 0.0, false).unwrap();
    let pbc = PbcOptions::default();
    let cold = ElectronicOptions {
        enable_dispersion: false,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        max_scc: 500,
        ..ElectronicOptions::default()
    };
    let warm = ElectronicOptions {
        electronic_temperature: 300.0,
        ..cold.clone()
    };

    let scf_warm = run_pbc_scc(&base, &params, &warm, &pbc).unwrap();
    let mos_warm = gamma_mos(&scf_warm, scf_warm.nelec).unwrap();
    assert_eq!(
        fractional_band_count(&mos_warm),
        0,
        "the insulator fixture must have integer occupations at 300 K"
    );

    let h_cold = pbc_gamma_hessian(&base, &params, &cold, &pbc).unwrap();
    let h_warm = pbc_gamma_hessian(&base, &params, &warm, &pbc).unwrap();
    let ndof = 3 * base.atoms.len();
    let mut max_diff = 0.0_f64;
    for i in 0..ndof {
        for j in 0..ndof {
            max_diff = max_diff.max((h_cold.hessian[(i, j)] - h_warm.hessian[(i, j)]).abs());
        }
    }
    // Measured 0.0 exactly: same branch, same arithmetic, bit-identical.
    assert_eq!(
        max_diff, 0.0,
        "gapped 300 K Hessian differs from the T = 0 Hessian by {max_diff:.3e}"
    );
}
