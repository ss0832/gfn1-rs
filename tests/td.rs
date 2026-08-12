// SPDX-License-Identifier: GPL-3.0-or-later
//! TD-GFN1 (TDA) excited-state checks.

use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::KMesh;
use gfn1_rs::{
    analytic_gradient, run_electronic, solve_tda, solve_tda_gradient, solve_tda_gradient_analytic,
    solve_tda_gradient_method, solve_tda_gradient_seminumerical, solve_tda_kpoint,
    solve_tda_kpoint_gradient, solve_tda_kpoint_gradient_analytic, solve_tda_pbc_gamma,
    tda_frozen_excitation_energy, AnalyticGradientOptions, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem, TdaGradientMethod, TdaOptions, TdaSpin,
};

fn load_params() -> Option<Gfn1Parameters> {
    Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
}

// ---------------------------------------------------------------------------
// Shared helpers for the analytic-gradient audit gates.
// ---------------------------------------------------------------------------

/// Tight T = 0 SCC settings: integer occupations (the analytic TDA gradient
/// requires a gapped closed shell) and converged tightly enough that the
/// finite-difference oracles are truncation- rather than noise-limited.
fn tight_options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-12,
        charge_tolerance: 1.0e-11,
        max_scc: 500,
        ..ElectronicOptions::default()
    }
}

fn displaced(system: &PeriodicSystem, atom: usize, axis: usize, delta: f64) -> PeriodicSystem {
    let mut out = system.clone();
    match axis {
        0 => out.atoms[atom].position.x += delta,
        1 => out.atoms[atom].position.y += delta,
        _ => out.atoms[atom].position.z += delta,
    }
    out
}

fn max_component_diff(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| {
        m.max((x.x - y.x).abs())
            .max((x.y - y.y).abs())
            .max((x.z - y.z).abs())
    })
}

fn max_norm(a: &[Vec3]) -> f64 {
    a.iter()
        .fold(0.0_f64, |m, v| m.max(v.x.abs()).max(v.y.abs()).max(v.z.abs()))
}

/// `sum_A dE/dR_A` — zero for any translationally invariant energy.
fn gradient_sum(g: &[Vec3]) -> Vec3 {
    g.iter().fold(Vec3::zero(), |acc, v| acc + *v)
}

/// Central finite difference of a scalar geometry function, per atom and axis.
fn central_fd<F>(system: &PeriodicSystem, h: f64, f: F) -> Vec<Vec3>
where
    F: Fn(&PeriodicSystem) -> f64,
{
    let mut out = vec![Vec3::zero(); system.atoms.len()];
    for atom in 0..system.atoms.len() {
        for axis in 0..3 {
            let plus = displaced(system, atom, axis, h);
            let minus = displaced(system, atom, axis, -h);
            let d = (f(&plus) - f(&minus)) / (2.0 * h);
            match axis {
                0 => out[atom].x = d,
                1 => out[atom].y = d,
                _ => out[atom].z = d,
            }
        }
    }
    out
}

/// Analytic ground-state (state-independent) total-energy gradient, used to peel
/// the excitation-energy gradient `domega/dR` out of the total excited gradient.
fn ground_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    opts: &ElectronicOptions,
) -> Vec<Vec3> {
    analytic_gradient(
        system,
        params,
        AnalyticGradientOptions {
            electronic: opts.clone(),
            ..AnalyticGradientOptions::default()
        },
    )
    .unwrap()
    .gradient
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";

const FORMALDEHYDE: &str = "4\nformaldehyde\n\
     C 0.000 0.000 0.000\n\
     O 0.000 0.000 1.205\n\
     H 0.000 0.943 -0.588\n\
     H 0.000 -0.943 -0.588\n";

/// Jittered water: no residual symmetry, so every Cartesian component of the
/// excited-state gradient is nonzero and no state is accidentally decoupled.
const WATER_JITTER: &str = "3\nwater\n\
     O 0.021 0.014 0.103\n\
     H 0.792 0.553 -0.041\n\
     H -0.741 0.581 0.033\n";

/// Jittered formaldehyde (out of the Cs plane) for the same reason.
const FORMALDEHYDE_JITTER: &str = "4\nformaldehyde\n\
     C 0.013 -0.021 0.004\n\
     O 0.032 0.017 1.208\n\
     H -0.041 0.951 -0.583\n\
     H 0.022 -0.937 -0.594\n";

const BUTADIENE: &str = "10\ns-trans-1,3-butadiene\n\
     C -1.850 -0.180 0.000\n\
     C -0.610 0.310 0.000\n\
     C 0.610 -0.310 0.000\n\
     C 1.850 0.180 0.000\n\
     H -1.950 -1.260 0.000\n\
     H -2.730 0.450 0.000\n\
     H -0.500 1.390 0.000\n\
     H 0.500 -1.390 0.000\n\
     H 1.950 1.260 0.000\n\
     H 2.730 -0.450 0.000\n";

fn check_tda_molecule(params: &Gfn1Parameters, xyz: &str, label: &str) {
    let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        max_scc: 500,
        ..ElectronicOptions::default()
    };
    let electronic = run_electronic(&system, params, opts).unwrap();
    let occ: Vec<usize> = (0..electronic.occupations.len())
        .filter(|&i| electronic.occupations[i] > 1.0e-8)
        .collect();
    let virt: Vec<usize> = (0..electronic.occupations.len())
        .filter(|&a| electronic.occupations[a] <= 1.0e-8)
        .collect();
    let mut gaps = Vec::new();
    for &i in &occ {
        for &a in &virt {
            gaps.push(electronic.orbital_energies[a] - electronic.orbital_energies[i]);
        }
    }
    gaps.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = gaps.len().min(6);

    let triplet = solve_tda(
        &system,
        params,
        &electronic,
        TdaOptions {
            n_states: n,
            spin: TdaSpin::Triplet,
        },
    )
    .unwrap();
    let singlet = solve_tda(
        &system,
        params,
        &electronic,
        TdaOptions {
            n_states: n,
            spin: TdaSpin::Singlet,
        },
    )
    .unwrap();
    for k in 0..n {
        assert!(
            (triplet.states[k].excitation_energy - gaps[k]).abs() < 1.0e-9,
            "{label}: triplet[{k}] != orbital gap"
        );
        assert!(
            singlet.states[k].excitation_energy >= triplet.states[k].excitation_energy - 1.0e-8,
            "{label}: singlet[{k}] below triplet[{k}]"
        );
        assert!(singlet.states[k].excitation_energy > 0.0);
    }
    let max_osc = singlet
        .states
        .iter()
        .map(|s| s.oscillator_strength)
        .fold(0.0, f64::max);
    assert!(max_osc > 0.0, "{label}: expected a bright singlet state");
}

#[test]
fn tda_formaldehyde_and_butadiene() {
    let Some(params) = load_params() else {
        return;
    };
    check_tda_molecule(&params, FORMALDEHYDE, "formaldehyde");
    check_tda_molecule(&params, BUTADIENE, "butadiene");
}

#[test]
fn tda_triplet_matches_orbital_gaps_and_singlet_is_higher() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0, // clean integer occupations
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let electronic = run_electronic(&system, &params, opts).unwrap();

    // Bare occupied->virtual orbital-energy gaps (the triplet TDA spectrum).
    let occ: Vec<usize> = (0..electronic.occupations.len())
        .filter(|&i| electronic.occupations[i] > 1.0e-8)
        .collect();
    let virt: Vec<usize> = (0..electronic.occupations.len())
        .filter(|&a| electronic.occupations[a] <= 1.0e-8)
        .collect();
    let mut gaps = Vec::new();
    for &i in &occ {
        for &a in &virt {
            gaps.push(electronic.orbital_energies[a] - electronic.orbital_energies[i]);
        }
    }
    gaps.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let n = gaps.len();
    let triplet = solve_tda(
        &system,
        &params,
        &electronic,
        TdaOptions {
            n_states: n,
            spin: TdaSpin::Triplet,
        },
    )
    .unwrap();
    let singlet = solve_tda(
        &system,
        &params,
        &electronic,
        TdaOptions {
            n_states: n,
            spin: TdaSpin::Singlet,
        },
    )
    .unwrap();

    assert_eq!(triplet.states.len(), n);
    // Triplet TDA (zero coupling) is exactly the sorted orbital-energy gaps.
    let mut max_gap_err = 0.0_f64;
    for k in 0..n {
        max_gap_err = max_gap_err.max((triplet.states[k].excitation_energy - gaps[k]).abs());
    }
    assert!(
        max_gap_err < 1.0e-9,
        "triplet TDA vs orbital gaps max error {max_gap_err:.3e}"
    );

    // The singlet Coulomb coupling kernel is positive semidefinite, so each singlet
    // eigenvalue is >= the corresponding triplet eigenvalue (Weyl monotonicity).
    for k in 0..n {
        assert!(
            singlet.states[k].excitation_energy >= triplet.states[k].excitation_energy - 1.0e-8,
            "singlet[{k}]={} below triplet[{k}]={}",
            singlet.states[k].excitation_energy,
            triplet.states[k].excitation_energy
        );
    }

    // Excitation energies positive, oscillator strengths finite and non-negative.
    for st in &singlet.states {
        assert!(st.excitation_energy > 0.0);
        assert!(st.oscillator_strength.is_finite() && st.oscillator_strength >= 0.0);
    }
    let max_osc = singlet
        .states
        .iter()
        .map(|s| s.oscillator_strength)
        .fold(0.0, f64::max);
    assert!(max_osc > 0.0, "expected a bright singlet state");
}

#[test]
fn tda_gradient_is_finite_and_step_consistent() {
    let Some(params) = load_params() else {
        return;
    };
    // Off-equilibrium water so the excited-state gradient is non-trivial.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.10\nH 0.80 0.60 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    let g1 = solve_tda_gradient(&system, &params, &opts, 0, tda, 1.0e-3).unwrap();
    let g2 = solve_tda_gradient(&system, &params, &opts, 0, tda, 2.0e-3).unwrap();

    // forces = -gradient.
    for (g, f) in g1.gradient.iter().zip(g1.forces.iter()) {
        assert!((g.x + f.x).abs() < 1.0e-14 && (g.y + f.y).abs() < 1.0e-14);
    }
    // total energy = ground free energy + excitation energy.
    let ground = run_electronic(&system, &params, opts.clone()).unwrap();
    assert!((g1.total_energy - (ground.total_free + g1.excitation_energy)).abs() < 1.0e-10);
    // Central differences at two steps agree to FD truncation order.
    let mut max_diff = 0.0_f64;
    for (a, b) in g1.gradient.iter().zip(g2.gradient.iter()) {
        max_diff = max_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_diff < 1.0e-4,
        "TD gradient not step-consistent: max diff {max_diff:.3e}"
    );
    // The excited-state gradient differs from the ground-state gradient.
    assert!(
        g1.gradient
            .iter()
            .any(|g| g.x.abs() + g.y.abs() + g.z.abs() > 1.0e-4),
        "excited-state gradient is unexpectedly zero"
    );
}

#[test]
fn tda_frozen_excitation_energy_reproduces_tda_energy() {
    let Some(params) = load_params() else {
        return;
    };
    // The frozen-amplitude Rayleigh quotient X^T A(R) X must reproduce the
    // variational TDA excitation energy at the reference geometry; its central
    // finite difference is the root-tracking-free reference for excited-state
    // gradients.
    let system = PeriodicSystem::from_xyz_str(
        "4\nformaldehyde\nC 0.02 0.00 0.00\nO 0.00 0.00 1.21\nH 0.00 0.94 -0.59\nH 0.00 -0.94 -0.59\n",
        0.0,
        false,
    )
    .unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-9,
        max_scc: 500,
        ..ElectronicOptions::default()
    };
    let electronic = run_electronic(&system, &params, opts.clone()).unwrap();
    for spin in [TdaSpin::Triplet, TdaSpin::Singlet] {
        let tda = TdaOptions { n_states: 3, spin };
        let td = solve_tda(&system, &params, &electronic, tda).unwrap();
        for state in 0..3 {
            let x = &td.states[state].amplitudes;
            // At the reference geometry both gauges coincide, so `None` and
            // `Some(&electronic)` must give the variational excitation energy.
            for reference in [None, Some(&electronic)] {
                let frozen =
                    tda_frozen_excitation_energy(&system, &params, &opts, x, spin, reference)
                        .unwrap();
                assert!(
                    (frozen - td.states[state].excitation_energy).abs() < 1.0e-8,
                    "{} state {state}: frozen energy {frozen} != TDA omega {}",
                    spin.label(),
                    td.states[state].excitation_energy
                );
            }
        }

        // The fully analytic excited-state gradient (direct-CPHF derivative) must
        // produce consistent forces = -gradient, the correct total energy, and agree
        // with the root-tracking finite-difference reference to FD precision.
        let g = solve_tda_gradient_analytic(&system, &params, &opts, 0, tda).unwrap();
        assert!((g.total_energy - (electronic.total_free + g.excitation_energy)).abs() < 1.0e-9);
        for (gr, f) in g.gradient.iter().zip(g.forces.iter()) {
            assert!((gr.x + f.x).abs() < 1.0e-14 && (gr.y + f.y).abs() < 1.0e-14);
            assert!(gr.x.is_finite() && gr.y.is_finite() && gr.z.is_finite());
        }
        assert!(
            g.gradient
                .iter()
                .any(|gr| gr.x.abs() + gr.y.abs() + gr.z.abs() > 1.0e-4),
            "{}: analytic excited-state gradient unexpectedly zero",
            spin.label()
        );
        let fdref = solve_tda_gradient(&system, &params, &opts, 0, tda, 1.0e-4).unwrap();
        let mut maxdiff = 0.0_f64;
        for (ga, gf) in g.gradient.iter().zip(fdref.gradient.iter()) {
            maxdiff = maxdiff
                .max((ga.x - gf.x).abs())
                .max((ga.y - gf.y).abs())
                .max((ga.z - gf.z).abs());
        }
        assert!(
            maxdiff < 1.0e-5,
            "{}: analytic vs finite-difference excited-state gradient max diff {maxdiff:.3e} Ha/bohr",
            spin.label()
        );
    }
}

#[test]
fn pbc_gamma_tda_matches_molecular_limit_and_singlet_above_triplet() {
    let Some(params) = load_params() else {
        return;
    };
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    let mol = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
    let mol_e = run_electronic(&mol, &params, opts.clone()).unwrap();
    let mol_td = solve_tda(&mol, &params, &mol_e, tda).unwrap();

    // Water in a large box: the Gamma-point periodic TDA should approach the
    // molecular excitation energies.
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"13 0 0 0 13 0 0 0 13\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let pbc_s = solve_tda_pbc_gamma(&cell, &params, &opts, tda).unwrap();
    for k in 0..3 {
        let d = (mol_td.states[k].excitation_energy - pbc_s.states[k].excitation_energy).abs();
        assert!(
            d < 5.0e-3,
            "PBC Gamma TDA state {k} differs from molecular limit by {d:.3e} Hartree"
        );
    }
    let pbc_t = solve_tda_pbc_gamma(
        &cell,
        &params,
        &opts,
        TdaOptions {
            n_states: 3,
            spin: TdaSpin::Triplet,
        },
    )
    .unwrap();
    for k in 0..3 {
        assert!(
            pbc_s.states[k].excitation_energy >= pbc_t.states[k].excitation_energy - 1.0e-8,
            "PBC singlet below triplet at state {k}"
        );
    }
}

#[test]
fn kpoint_tda_reduces_to_gamma_and_runs_on_a_mesh() {
    let Some(params) = load_params() else {
        return;
    };
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 4,
        spin: TdaSpin::Singlet,
    };
    // Water in a box (gapped molecular crystal limit).
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
        0.0,
        false,
    )
    .unwrap();

    // A single Gamma point through the k-point path must reproduce the dedicated
    // Gamma-point TDA exactly.
    let gamma = solve_tda_pbc_gamma(&cell, &params, &opts, tda).unwrap();
    let kp_gamma = solve_tda_kpoint(&cell, &params, &opts, KMesh::gamma(), tda).unwrap();
    for k in 0..4 {
        let d = (gamma.states[k].excitation_energy - kp_gamma.states[k].excitation_energy).abs();
        assert!(
            d < 1.0e-8,
            "k-point Gamma TDA state {k} differs from solve_tda_pbc_gamma by {d:.3e}"
        );
    }

    // A 2x2x2 Monkhorst-Pack mesh must run and give positive, finite, sorted
    // singlet excitations >= the triplet (bare-gap) spectrum.
    let mp = KMesh::monkhorst_pack([2, 2, 2]);
    let s = solve_tda_kpoint(&cell, &params, &opts, mp, tda).unwrap();
    let t = solve_tda_kpoint(
        &cell,
        &params,
        &opts,
        mp,
        TdaOptions {
            n_states: 4,
            spin: TdaSpin::Triplet,
        },
    )
    .unwrap();
    for k in 0..4 {
        assert!(s.states[k].excitation_energy > 0.0 && s.states[k].excitation_energy.is_finite());
        assert!(
            s.states[k].excitation_energy >= t.states[k].excitation_energy - 1.0e-8,
            "k-point singlet[{k}] below triplet"
        );
    }
}

#[test]
fn pbc_gamma_tda_gradient_step_consistent() {
    let Some(params) = load_params() else {
        return;
    };
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"12 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.10\nH 0.80 0.60 0.0\nH -0.74 0.58 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    let g1 = solve_tda_gradient(&cell, &params, &opts, 0, tda, 1.0e-3).unwrap();
    let g2 = solve_tda_gradient(&cell, &params, &opts, 0, tda, 2.0e-3).unwrap();
    let mut max_diff = 0.0_f64;
    for (a, b) in g1.gradient.iter().zip(g2.gradient.iter()) {
        max_diff = max_diff
            .max((a.x - b.x).abs())
            .max((a.y - b.y).abs())
            .max((a.z - b.z).abs());
    }
    assert!(
        max_diff < 1.0e-4,
        "PBC TD gradient not step-consistent: {max_diff:.3e}"
    );
}

#[test]
fn pbc_gamma_tda_gradient_analytic_matches_fd() {
    let Some(params) = load_params() else {
        return;
    };
    // Water in an 11 A box (gapped molecular-crystal limit), slightly off the
    // molecular-plane symmetry so all three Cartesian components are nonzero.
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    for state in 0..3 {
        let analytic = solve_tda_gradient_analytic(&cell, &params, &opts, state, tda).unwrap();
        let fd = solve_tda_gradient(&cell, &params, &opts, state, tda, 1.0e-3).unwrap();
        let mut max_diff = 0.0_f64;
        for (a, b) in analytic.gradient.iter().zip(fd.gradient.iter()) {
            max_diff = max_diff
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
        assert!(
            max_diff < 1.0e-5,
            "PBC Gamma analytic TDA gradient state {state} disagrees with FD: {max_diff:.3e}"
        );
    }
}

// The general complex k-mesh analytic TDA excitation gradient must match the
// finite-difference k-mesh gradient (the FD ground truth) to FD precision, and the
// Gamma-mesh special case must reduce to the verified Gamma path.
#[test]
fn kmesh_tda_gradient_analytic_matches_fd() {
    let Some(params) = load_params() else {
        return;
    };
    let opts = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        ..ElectronicOptions::default()
    };
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };

    let kmesh_max_diff = |cell: &PeriodicSystem, mesh: KMesh, state: usize| -> f64 {
        let an =
            solve_tda_kpoint_gradient_analytic(cell, &params, &opts, mesh, state, tda).unwrap();
        let fd = solve_tda_kpoint_gradient(cell, &params, &opts, mesh, state, tda, 1.0e-3).unwrap();
        let mut md = 0.0_f64;
        for (a, b) in an.gradient.iter().zip(fd.gradient.iter()) {
            md = md
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
        md
    };

    // Water in an 11 A box: Gamma-mesh reduction and a genuine 2x1x1 complex mesh,
    // both for the single-pair (0) and mixed (1) roots.
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    for &(mesh, name) in &[
        (KMesh::gamma(), "gamma"),
        (KMesh::monkhorst_pack([2, 1, 1]), "2x1x1"),
    ] {
        for state in 0..2 {
            let md = kmesh_max_diff(&cell, mesh, state);
            assert!(
                md < 1.0e-5,
                "k-mesh ({name}) analytic TDA gradient state {state} vs FD: {md:.3e}"
            );
        }
    }

    // Dispersive HF cell (short b axis) with a 1x2x1 mesh: genuinely dispersive
    // complex Bloch bands (not near-degenerate), stressing the complex per-k path.
    let disp = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"10 0 0 0 3.2 0 0 0 10\" pbc=\"T T T\"\n\
         F 0.0 0.0 0.0\nH 0.0 0.92 0.10\n",
        0.0,
        false,
    )
    .unwrap();
    let md = kmesh_max_diff(&disp, KMesh::monkhorst_pack([1, 2, 1]), 0);
    assert!(
        md < 1.0e-5,
        "dispersive k-mesh analytic TDA gradient vs FD: {md:.3e}"
    );
}

// ===========================================================================
// Analytic TD-GFN1 nuclear-gradient audit gates (non-PBC + PBC).
//
// Every gate below follows the project convention of
// docs/derivatives.md#5-the-verification-gate-philosophy: the analytic quantity
// is compared to a central finite difference of the validated lower order, the
// `h^2` ladder is demonstrated, and the exact invariances (translational sum
// rule, path-to-path consistency) are asserted separately.
// ===========================================================================

/// **Gate (a) — molecular analytic vs finite difference, with the `h^2` ladder.**
///
/// For each fixture and each requested root the analytic total excited-state
/// gradient is checked against a central FD of the *total* excited energy
/// `E_ground(free) + omega` (re-diagonalised, amplitude-overlap root tracking —
/// [`solve_tda_gradient`]), and the excitation-only part
/// `analytic_total - analytic_ground` against a central FD of the excitation
/// energy alone ([`tda_frozen_excitation_energy`], which is root-flip immune and
/// whose derivative equals `domega/dR` exactly by amplitude stationarity).
///
/// Both fixtures are jittered off their symmetry so no state is accidentally
/// decoupled: on symmetric water the lowest roots carry *exactly zero* transition
/// charge (`|mu| ~ 1e-15`), which silently switches off the whole
/// `c d(P^T K P)/dR` coupling derivative and makes an FD gate on those roots
/// vacuous for two of the three gradient terms. The gate therefore also asserts
/// that a genuinely bright root is covered.
///
/// Roots closer than `5e-3` Hartree to a neighbour are skipped: the
/// re-diagonalised oracle follows the root by amplitude overlap, which is
/// ill-posed inside a near-degenerate pair. Those are covered separately by
/// [`tda_near_degenerate_root_gradient`].
#[test]
fn tda_analytic_gradient_matches_fd_with_h_ladder() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 6,
        spin: TdaSpin::Singlet,
    };
    let mut brightest = 0.0_f64;
    for (label, xyz) in [
        ("water", WATER_JITTER),
        ("formaldehyde", FORMALDEHYDE_JITTER),
    ] {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, eo.clone()).unwrap();
        let td = solve_tda(&system, &params, &electronic, tda).unwrap();
        let gnd = ground_gradient(&system, &params, &eo);
        let mut gated = 0usize;
        for state in 0..5 {
            let omega = td.states[state].excitation_energy;
            let sep = td
                .states
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != state)
                .fold(f64::MAX, |m, (_, s)| {
                    m.min((s.excitation_energy - omega).abs())
                });
            if sep < 5.0e-3 {
                eprintln!("{label} S{state}: skipped, nearest root {sep:.3e} Ha away");
                continue;
            }
            gated += 1;
            let mu = td.states[state].transition_dipole;
            brightest =
                brightest.max((mu.x * mu.x + mu.y * mu.y + mu.z * mu.z).sqrt());

            let ana = solve_tda_gradient_analytic(&system, &params, &eo, state, tda).unwrap();
            assert!(
                (ana.total_energy - (electronic.total_free + ana.excitation_energy)).abs() < 1.0e-9,
                "{label} state {state}: total energy is not E_ground(free) + omega"
            );
            for (g, f) in ana.gradient.iter().zip(ana.forces.iter()) {
                assert!((g.x + f.x).abs() < 1.0e-14 && (g.y + f.y).abs() < 1.0e-14);
            }

            // (i) total excited energy, re-diagonalised FD oracle, h ladder.
            let amps = td.states[state].amplitudes.clone();
            let exc_ana: Vec<Vec3> = ana
                .gradient
                .iter()
                .zip(gnd.iter())
                .map(|(a, g)| *a - *g)
                .collect();
            let mut res_total = [0.0_f64; 2];
            let mut res_omega = [0.0_f64; 2];
            for (idx, &h) in [2.0e-3_f64, 1.0e-3].iter().enumerate() {
                let fd_total = solve_tda_gradient(&system, &params, &eo, state, tda, h).unwrap();
                res_total[idx] = max_component_diff(&ana.gradient, &fd_total.gradient);
                // (ii) excitation energy alone, frozen-amplitude FD oracle. The
                // reference SCC pins the MO phase gauge the amplitudes live in.
                let fd_omega = central_fd(&system, h, |s| {
                    tda_frozen_excitation_energy(
                        s,
                        &params,
                        &eo,
                        &amps,
                        tda.spin,
                        Some(&electronic),
                    )
                    .unwrap()
                });
                res_omega[idx] = max_component_diff(&exc_ana, &fd_omega);
            }
            eprintln!(
                "{label} S{state} (f={:.2e}): |ana-FD_total| {:.3e} -> {:.3e} | \
                 |ana-FD_omega| {:.3e} -> {:.3e}",
                td.states[state].oscillator_strength,
                res_total[0],
                res_total[1],
                res_omega[0],
                res_omega[1]
            );
            assert!(
                res_total[1] < 5.0e-6,
                "{label} state {state}: analytic vs FD(total energy) = {:.3e} Ha/bohr",
                res_total[1]
            );
            assert!(
                res_omega[1] < 5.0e-6,
                "{label} state {state}: analytic vs FD(excitation energy) = {:.3e} Ha/bohr",
                res_omega[1]
            );
            // h^2 ladder: halving h must quarter the residual, unless it has
            // already reached the SCC/FD noise floor.
            for (name, r) in [("total", res_total), ("omega", res_omega)] {
                if r[1] > 5.0e-9 {
                    let ratio = r[0] / r[1];
                    assert!(
                        (2.5..=6.5).contains(&ratio),
                        "{label} state {state} ({name}): h^2 ladder ratio {ratio:.2} \
                         (residuals {:.3e} -> {:.3e}) — the residual is not pure FD truncation",
                        r[0],
                        r[1]
                    );
                }
            }
        }
        assert!(gated >= 2, "{label}: only {gated} roots were gated");
    }
    // At least one gated root must carry a substantial transition dipole, or the
    // kernel-derivative and transition-charge-derivative terms are never exercised.
    assert!(
        brightest > 1.0e-1,
        "no gated root is bright (max |mu| = {brightest:.3e} e*a0); the \
         c d(P^T K P)/dR coupling derivative would be untested"
    );
}

/// **Gate (a2) — near-degenerate roots / root-flipping robustness.**
///
/// Jittered formaldehyde has a near-degenerate singlet pair (`S2`, `S3`, split by
/// about `1e-3` Hartree). This gate pins what happens there:
///
///  * the **frozen-amplitude** oracle never re-diagonalises, so it is immune to
///    root flipping and the analytic gradient still tracks it;
///  * the **re-diagonalised, amplitude-overlap-tracked** oracle
///    ([`solve_tda_gradient`]) is ill-posed inside the pair — the two eigenvectors
///    span the same two-dimensional subspace — and its disagreement with the
///    analytic gradient is orders of magnitude larger.
///
/// Both numbers are asserted so a regression in either direction is visible.
#[test]
fn tda_near_degenerate_root_gradient() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 6,
        spin: TdaSpin::Singlet,
    };
    let system = PeriodicSystem::from_xyz_str(FORMALDEHYDE_JITTER, 0.0, false).unwrap();
    let electronic = run_electronic(&system, &params, eo.clone()).unwrap();
    let td = solve_tda(&system, &params, &electronic, tda).unwrap();
    // Locate the closest adjacent pair; the fixture is chosen so one exists.
    let mut pair = (0usize, f64::MAX);
    for k in 1..td.states.len() {
        let d = td.states[k].excitation_energy - td.states[k - 1].excitation_energy;
        if d < pair.1 {
            pair = (k - 1, d);
        }
    }
    let (lo, gap) = pair;
    eprintln!("formaldehyde near-degenerate pair S{lo}/S{}: gap {gap:.3e} Ha", lo + 1);
    assert!(
        gap < 5.0e-3,
        "fixture no longer has a near-degenerate pair (closest gap {gap:.3e} Ha)"
    );

    let gnd = ground_gradient(&system, &params, &eo);
    for state in [lo, lo + 1] {
        let ana = solve_tda_gradient_analytic(&system, &params, &eo, state, tda).unwrap();
        let amps = td.states[state].amplitudes.clone();
        let exc_ana: Vec<Vec3> = ana
            .gradient
            .iter()
            .zip(gnd.iter())
            .map(|(a, g)| *a - *g)
            .collect();
        let mut frozen = [0.0_f64; 2];
        for (idx, &h) in [2.0e-3_f64, 1.0e-3].iter().enumerate() {
            let fd = central_fd(&system, h, |s| {
                tda_frozen_excitation_energy(s, &params, &eo, &amps, tda.spin, Some(&electronic))
                    .unwrap()
            });
            frozen[idx] = max_component_diff(&exc_ana, &fd);
        }
        let rediag = max_component_diff(
            &ana.gradient,
            &solve_tda_gradient(&system, &params, &eo, state, tda, 1.0e-3)
                .unwrap()
                .gradient,
        );
        eprintln!(
            "  S{state}: |ana-FDfrozen| {:.3e} -> {:.3e}   |ana-FDrediag(h=1e-3)| {rediag:.3e}",
            frozen[0], frozen[1]
        );
        // The frozen oracle stays usable through the near degeneracy.
        assert!(
            frozen[1] < 1.0e-5,
            "near-degenerate S{state}: analytic vs frozen-amplitude FD = {:.3e} Ha/bohr",
            frozen[1]
        );
        // The re-diagonalised oracle degrades but must not diverge.
        assert!(
            rediag < 1.0e-3,
            "near-degenerate S{state}: analytic vs re-diagonalised FD = {rediag:.3e} Ha/bohr"
        );
    }
}

/// **Gate (b) — translational invariance.** The excited-state total energy is
/// invariant under a rigid translation, so `sum_A dE/dR_A = 0` exactly. Checked
/// for every analytic entry point (molecular, semi-numerical, Gamma-point PBC,
/// k-mesh PBC) and for each of several roots.
#[test]
fn tda_gradient_translational_invariance() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 4,
        spin: TdaSpin::Singlet,
    };
    let mol = PeriodicSystem::from_xyz_str(FORMALDEHYDE_JITTER, 0.0, false).unwrap();
    for state in 0..3 {
        let ana = solve_tda_gradient_analytic(&mol, &params, &eo, state, tda).unwrap();
        let s = gradient_sum(&ana.gradient);
        let scale = max_norm(&ana.gradient).max(1.0);
        eprintln!(
            "translational sum, molecular analytic S{state}: ({:.2e},{:.2e},{:.2e})",
            s.x, s.y, s.z
        );
        assert!(
            s.x.abs().max(s.y.abs()).max(s.z.abs()) < 1.0e-8 * scale,
            "molecular analytic TDA gradient state {state} breaks translational invariance: \
             ({:.3e},{:.3e},{:.3e})",
            s.x,
            s.y,
            s.z
        );
    }
    let semi = solve_tda_gradient_seminumerical(&mol, &params, &eo, 0, tda, 1.0e-3).unwrap();
    let s = gradient_sum(&semi.gradient);
    assert!(
        s.x.abs().max(s.y.abs()).max(s.z.abs()) < 1.0e-6,
        "semi-numerical TDA gradient breaks translational invariance: \
         ({:.3e},{:.3e},{:.3e})",
        s.x,
        s.y,
        s.z
    );

    // Periodic: a rigid translation of the whole basis inside a fixed cell.
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    for state in 0..2 {
        let g = solve_tda_gradient_analytic(&cell, &params, &eo, state, tda).unwrap();
        let s = gradient_sum(&g.gradient);
        eprintln!(
            "translational sum, PBC Gamma analytic S{state}: ({:.2e},{:.2e},{:.2e})",
            s.x, s.y, s.z
        );
        assert!(
            s.x.abs().max(s.y.abs()).max(s.z.abs()) < 1.0e-7,
            "PBC Gamma analytic TDA gradient state {state} breaks translational invariance: \
             ({:.3e},{:.3e},{:.3e})",
            s.x,
            s.y,
            s.z
        );
    }
    let g = solve_tda_kpoint_gradient_analytic(
        &cell,
        &params,
        &eo,
        KMesh::monkhorst_pack([2, 1, 1]),
        0,
        tda,
    )
    .unwrap();
    let s = gradient_sum(&g.gradient);
    eprintln!(
        "translational sum, k-mesh analytic S0: ({:.2e},{:.2e},{:.2e})",
        s.x, s.y, s.z
    );
    assert!(
        s.x.abs().max(s.y.abs()).max(s.z.abs()) < 1.0e-7,
        "k-mesh analytic TDA gradient breaks translational invariance: \
         ({:.3e},{:.3e},{:.3e})",
        s.x,
        s.y,
        s.z
    );
}

/// **Gate (c) — dispatch and method consistency.** `solve_tda_gradient_method`
/// must return exactly what the concrete entry point returns for each variant,
/// and the semi-numerical hybrid must agree with the fully analytic gradient to
/// the finite-difference tolerance of its `domega/dR` step.
#[test]
fn tda_gradient_method_dispatch_and_cross_consistency() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 4,
        spin: TdaSpin::Singlet,
    };
    let system = PeriodicSystem::from_xyz_str(WATER_JITTER, 0.0, false).unwrap();
    let h = 1.0e-3;
    for state in 0..2 {
        let ana = solve_tda_gradient_analytic(&system, &params, &eo, state, tda).unwrap();
        let semi = solve_tda_gradient_seminumerical(&system, &params, &eo, state, tda, h).unwrap();
        let fd = solve_tda_gradient(&system, &params, &eo, state, tda, h).unwrap();

        let d_ana = solve_tda_gradient_method(
            &system,
            &params,
            &eo,
            state,
            tda,
            h,
            TdaGradientMethod::Analytic,
        )
        .unwrap();
        let d_semi = solve_tda_gradient_method(
            &system,
            &params,
            &eo,
            state,
            tda,
            h,
            TdaGradientMethod::SemiNumerical,
        )
        .unwrap();
        let d_fd = solve_tda_gradient_method(
            &system,
            &params,
            &eo,
            state,
            tda,
            h,
            TdaGradientMethod::FiniteDifference,
        )
        .unwrap();
        assert_eq!(max_component_diff(&d_ana.gradient, &ana.gradient), 0.0);
        assert_eq!(max_component_diff(&d_semi.gradient, &semi.gradient), 0.0);
        assert_eq!(max_component_diff(&d_fd.gradient, &fd.gradient), 0.0);

        let semi_vs_ana = max_component_diff(&semi.gradient, &ana.gradient);
        let fd_vs_ana = max_component_diff(&fd.gradient, &ana.gradient);
        eprintln!(
            "water S{state}: |semi-ana| = {semi_vs_ana:.3e}  |fd-ana| = {fd_vs_ana:.3e}"
        );
        assert!(
            semi_vs_ana < 1.0e-6,
            "semi-numerical vs analytic state {state}: {semi_vs_ana:.3e} Ha/bohr"
        );
        assert!(
            fd_vs_ana < 1.0e-6,
            "full FD vs analytic state {state}: {fd_vs_ana:.3e} Ha/bohr"
        );
    }
    // The periodic dispatch: `Analytic` routes to the Gamma-point analytic path,
    // `SemiNumerical` is non-periodic only and must say so.
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let pbc_disp = solve_tda_gradient_method(
        &cell,
        &params,
        &eo,
        0,
        tda,
        h,
        TdaGradientMethod::Analytic,
    )
    .unwrap();
    let pbc_direct = solve_tda_gradient_analytic(&cell, &params, &eo, 0, tda).unwrap();
    assert_eq!(
        max_component_diff(&pbc_disp.gradient, &pbc_direct.gradient),
        0.0
    );
    let err = solve_tda_gradient_method(
        &cell,
        &params,
        &eo,
        0,
        tda,
        h,
        TdaGradientMethod::SemiNumerical,
    );
    assert!(
        err.is_err(),
        "semi-numerical TDA gradient must reject periodic systems"
    );
}

/// **Gate (d2) — large-box Gamma-point PBC reduces to the molecular limit.**
/// Both the excitation energies and the excited-state gradients converge to the
/// molecular values as the box grows; the residual is the `O(1/L)` image
/// interaction of the periodic electrostatics, so it is checked at two box sizes
/// and required to shrink.
#[test]
fn pbc_gamma_tda_gradient_matches_molecular_limit() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    let geom = "O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n";
    let mol = PeriodicSystem::from_xyz_str(&format!("3\nwater\n{geom}"), 0.0, false).unwrap();
    let mol_e = run_electronic(&mol, &params, eo.clone()).unwrap();
    let mol_td = solve_tda(&mol, &params, &mol_e, tda).unwrap();
    let mol_g = solve_tda_gradient_analytic(&mol, &params, &eo, 0, tda).unwrap();

    let mut prev_energy = f64::MAX;
    let mut prev_grad = f64::MAX;
    for l in [11.0_f64, 16.0] {
        let cell = PeriodicSystem::from_xyz_str(
            &format!("3\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"\n{geom}"),
            0.0,
            false,
        )
        .unwrap();
        let td = solve_tda_pbc_gamma(&cell, &params, &eo, tda).unwrap();
        let de = (0..3).fold(0.0_f64, |m, k| {
            m.max((td.states[k].excitation_energy - mol_td.states[k].excitation_energy).abs())
        });
        let g = solve_tda_gradient_analytic(&cell, &params, &eo, 0, tda).unwrap();
        let dg = max_component_diff(&g.gradient, &mol_g.gradient);
        eprintln!("L={l} A: max|domega| = {de:.3e} Ha,  max|dgrad| = {dg:.3e} Ha/bohr");
        assert!(
            de < 5.0e-3,
            "L={l}: Gamma-point TDA excitation energies differ from the molecular limit by {de:.3e}"
        );
        assert!(
            dg < 5.0e-3,
            "L={l}: Gamma-point TDA gradient differs from the molecular limit by {dg:.3e}"
        );
        assert!(
            de < prev_energy && dg < prev_grad,
            "L={l}: the periodic-to-molecular residual did not shrink with the box"
        );
        prev_energy = de;
        prev_grad = dg;
    }
}

/// **Gate (e2) — a Gamma-only k-mesh reduces to the dedicated Gamma path**, for
/// the excitation energies *and* the analytic gradient, and the k-mesh analytic
/// gradient reproduces the k-mesh FD gradient on a 1D-periodic chain.
#[test]
fn kmesh_gamma_reduces_to_pbc_gamma_gradient_and_matches_fd_on_a_chain() {
    let Some(params) = load_params() else {
        return;
    };
    let eo = tight_options();
    let tda = TdaOptions {
        n_states: 3,
        spin: TdaSpin::Singlet,
    };
    let cell = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
         O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    for state in 0..2 {
        let gamma = solve_tda_gradient_analytic(&cell, &params, &eo, state, tda).unwrap();
        let kg =
            solve_tda_kpoint_gradient_analytic(&cell, &params, &eo, KMesh::gamma(), state, tda)
                .unwrap();
        let d = max_component_diff(&gamma.gradient, &kg.gradient);
        let dw = (gamma.excitation_energy - kg.excitation_energy).abs();
        eprintln!("Gamma-mesh reduction S{state}: |dgrad| = {d:.3e}, |domega| = {dw:.3e}");
        assert!(
            dw < 1.0e-9,
            "Gamma-only k-mesh excitation energy differs from solve_tda_pbc_gamma: {dw:.3e}"
        );
        assert!(
            d < 1.0e-7,
            "Gamma-only k-mesh analytic gradient differs from the Gamma path: {d:.3e}"
        );
    }

    // A genuinely 1D-periodic chain (long a, vacuum in b and c) with a 3x1x1
    // mesh: the analytic BZ-summed gradient against the k-mesh FD gradient.
    let chain = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"3.4 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
         F 0.0 0.0 0.0\nH 0.93 0.08 0.05\n",
        0.0,
        false,
    )
    .unwrap();
    let mesh = KMesh::monkhorst_pack([3, 1, 1]);
    let an = solve_tda_kpoint_gradient_analytic(&chain, &params, &eo, mesh, 0, tda).unwrap();
    let fd = solve_tda_kpoint_gradient(&chain, &params, &eo, mesh, 0, tda, 1.0e-3).unwrap();
    let d = max_component_diff(&an.gradient, &fd.gradient);
    eprintln!("1D chain 3x1x1 k-mesh: |analytic - FD| = {d:.3e} Ha/bohr");
    assert!(
        d < 1.0e-5,
        "1D-chain k-mesh analytic TDA gradient vs FD: {d:.3e} Ha/bohr"
    );
    let s = gradient_sum(&an.gradient);
    assert!(
        s.x.abs().max(s.y.abs()).max(s.z.abs()) < 1.0e-7,
        "1D-chain k-mesh analytic TDA gradient breaks translational invariance: \
         ({:.3e},{:.3e},{:.3e})",
        s.x,
        s.y,
        s.z
    );
}
