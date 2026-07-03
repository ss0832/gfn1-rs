// SPDX-License-Identifier: GPL-3.0-or-later
//! TD-GFN1 (TDA) excited-state checks (need `GFN1_XTB_PARAM`).

use gfn1_rs::pbc::KMesh;
use gfn1_rs::{
    run_electronic, solve_tda, solve_tda_gradient, solve_tda_gradient_analytic,
    solve_tda_kpoint, solve_tda_kpoint_gradient, solve_tda_kpoint_gradient_analytic,
    solve_tda_pbc_gamma, tda_frozen_excitation_energy, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem, TdaOptions, TdaSpin,
};

fn load_params() -> Option<Gfn1Parameters> {
    let path = std::env::var("GFN1_XTB_PARAM").ok()?;
    Gfn1Parameters::from_file(path).ok()
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";

const FORMALDEHYDE: &str = "4\nformaldehyde\n\
     C 0.000 0.000 0.000\n\
     O 0.000 0.000 1.205\n\
     H 0.000 0.943 -0.588\n\
     H 0.000 -0.943 -0.588\n";

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
            let frozen = tda_frozen_excitation_energy(&system, &params, &opts, x, spin).unwrap();
            assert!(
                (frozen - td.states[state].excitation_energy).abs() < 1.0e-8,
                "{} state {state}: frozen energy {frozen} != TDA omega {}",
                spin.label(),
                td.states[state].excitation_energy
            );
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
