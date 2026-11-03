// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic ML stationarity, image chains and strain/nuclear derivative gates.
use gfn1_rs::lattice::Lattice;
use gfn1_rs::math::{Mat3, Vec3};
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::pbc::*;
use gfn1_rs::*;
use std::sync::Arc;

fn hydrogen() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(
        "2\nLattice=\"5.6 .3 0 0 6.1 .2 .1 0 6.4\" pbc=\"T T T\"\nH .2 .3 .1\nH 1.6 .45 .2\n",
        0.,
        true,
    )
    .unwrap()
}
fn settings() -> (ElectronicOptions, PbcOptions) {
    (
        ElectronicOptions {
            electronic_temperature: 0.,
            energy_tolerance: 1e-12,
            charge_tolerance: 1e-11,
            ..Default::default()
        },
        PbcOptions {
            ao_cutoff: 20.,
            ..Default::default()
        },
    )
}
fn model() -> MlHamiltonian {
    let mut m = MlHamiltonian::zero(vec![1], vec![1.2, 3.5, 5.8], 9.);
    m.heads.get_mut(&1).unwrap().levels[0].weights = vec![0.001, -0.0015, 0.002];
    m.heads.get_mut(&1).unwrap().correction.weights = vec![0.0004, 0.0002, -0.0003];
    m
}
fn strain(system: &PeriodicSystem, a: usize, b: usize, step: f64) -> PeriodicSystem {
    let lattice = system.lattice.unwrap();
    let transform = |r: Vec3| {
        let mut v = r.to_array();
        v[a] += step * v[b];
        Vec3::new(v[0], v[1], v[2])
    };
    let mut changed = system.clone();
    changed.lattice = Some(
        Lattice::new(
            Mat3::from_columns(
                transform(lattice.cell.col[0]),
                transform(lattice.cell.col[1]),
                transform(lattice.cell.col[2]),
            ),
            lattice.periodic,
        )
        .unwrap(),
    );
    for atom in &mut changed.atoms {
        atom.position = transform(atom.position);
    }
    changed
}
#[test]
fn zero_heads_restore_gamma_and_kpoint_results_including_dfpt() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (plain, mut pbc) = settings();
    let mut zero = plain.clone();
    zero.ml_model = Some(Arc::new(MlHamiltonian::zero(
        vec![1],
        vec![1.2, 3.5, 5.8],
        9.,
    )));
    for mesh in [KMesh::gamma(), KMesh::monkhorst_pack([2, 2, 2])] {
        pbc.kmesh = mesh;
        let base = pbc_analytic_gradient(&system, &params, &plain, &pbc).unwrap();
        let actual = pbc_analytic_gradient(&system, &params, &zero, &pbc).unwrap();
        assert_eq!(base.total_energy, actual.total_energy);
        assert_eq!(base.gradient, actual.gradient);
        assert_eq!(base.scf.atomic_charges, actual.scf.atomic_charges);
        assert_eq!(
            pbc_stress(&system, &params, &plain, &pbc)
                .unwrap()
                .stress
                .as_slice(),
            pbc_stress(&system, &params, &zero, &pbc)
                .unwrap()
                .stress
                .as_slice()
        );
        let projected = pbc_electronic_result(actual.scf.clone(), &system, pbc.ao_cutoff).unwrap();
        assert!(projected.ml_parameters.is_some());
        assert!(projected.method_citations.contains(&"Li2026MLxTB"));
        assert!((projected.energy_terms().component_sum() - actual.total_energy).abs() < 1e-13);
        let hessian = |o| {
            if mesh.is_gamma_only() {
                pbc_gamma_hessian(&system, &params, o, &pbc).unwrap()
            } else {
                pbc_kpoint_hessian(&system, &params, o, &pbc).unwrap()
            }
        };
        assert_eq!(
            hessian(&plain).hessian.as_slice(),
            hessian(&zero).hessian.as_slice()
        );
        let error = if mesh.is_gamma_only() {
            GammaThirdReference::build(&system, &params, &zero, &pbc)
                .err()
                .unwrap()
        } else {
            KpointThirdReference::build(&system, &params, &zero, &pbc)
                .err()
                .unwrap()
        };
        assert!(error
            .to_string()
            .contains("third-order structural response"));
    }
}

#[test]
fn periodic_ml_dfpt_hessian_includes_frozen_mixed_and_density_response_chains() {
    let water = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"3.8 0 0 .2 4 0 0 .1 4.2\" pbc=\"T T T\"\nO 0 0 0\nH .958 0 0\nH -.25 .925 .03\n", 0., false,
    ).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let (mut options, mut pbc) = settings();
    for (system, model) in [
        (hydrogen(), model()),
        (water, MlHamiltonian::builtin_water_clusters().unwrap()),
    ] {
        options.ml_model = Some(Arc::new(model));
        for (mesh, temperature) in [
            (KMesh::gamma(), 0.),
            (KMesh::monkhorst_pack([2, 1, 1]), 0.),
            (KMesh::monkhorst_pack([2, 1, 1]), 10000.),
        ] {
            pbc.kmesh = mesh;
            options.electronic_temperature = temperature;
            let actual = if mesh.is_gamma_only() {
                pbc_gamma_hessian(&system, &params, &options, &pbc)
            } else {
                pbc_kpoint_hessian(&system, &params, &options, &pbc)
            }
            .unwrap();
            assert!(
                actual.raw_max_asymmetry < 2e-7,
                "unprojected Hessian asymmetry {}",
                actual.raw_max_asymmetry
            );
            let n = system.atoms.len();
            let d = 3 * n;
            for step in [2e-4, 1e-4] {
                let mut largest = 0_f64;
                for y in 0..d {
                    let direction = match y % 3 {
                        0 => Vec3::new(step, 0., 0.),
                        1 => Vec3::new(0., step, 0.),
                        _ => Vec3::new(0., 0., step),
                    };
                    let mut plus = system.clone();
                    let mut minus = system.clone();
                    plus.atoms[y / 3].position += direction;
                    minus.atoms[y / 3].position -= direction;
                    let gp = pbc_analytic_gradient(&plus, &params, &options, &pbc)
                        .unwrap()
                        .gradient;
                    let gm = pbc_analytic_gradient(&minus, &params, &options, &pbc)
                        .unwrap()
                        .gradient;
                    for a in 0..n {
                        let fd = ((gp[a] - gm[a]) / (2. * step)).to_array();
                        for k in 0..3 {
                            largest = largest.max((fd[k] - actual.hessian[(3 * a + k, y)]).abs());
                        }
                    }
                }
                assert!(
                    largest < 3e-6,
                    "atoms={n}, k={:?}, T={temperature}, h={step}, Hessian FD error {largest:e}",
                    mesh.size
                );
            }
            for x in 0..d {
                for k in 0..3 {
                    let translation: f64 = (0..n).map(|a| actual.hessian[(x, 3 * a + k)]).sum();
                    assert!(
                        translation.abs() < 1e-7,
                        "raw acoustic residual {translation:e}"
                    );
                }
            }
            if n == 3 && mesh.is_gamma_only() {
                let mut higher = options.clone();
                higher.charge_order = 4;
                let projected = pbc_electronic_result(
                    run_pbc_scc(&system, &params, &higher, &pbc).unwrap(),
                    &system,
                    pbc.ao_cutoff,
                )
                .unwrap();
                assert!(
                    (projected.energy_terms().component_sum() - projected.total_free).abs() < 1e-12
                );
            }
        }
    }
}

#[test]
fn finite_q_ml_phonons_match_a_frozen_supercell_force_wave() {
    use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (mut options, pbc) = settings();
    options.ml_model = Some(Arc::new(model()));
    let reference = AnalyticSupercellPhonons::build(
        &system,
        &params,
        &options,
        &pbc,
        &[1., 1.],
        PhononSupercellOptions {
            repetitions: [3, 1, 1],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.raw_hessian_asymmetry < 2e-7);
    let positive = reference.at_q([1. / 3., 0., 0.]).unwrap();
    let negative = reference.at_q([-1. / 3., 0., 0.]).unwrap();
    assert!(positive.method_citations.contains(&"Li2026MLxTB"));
    assert!(positive.raw_hermiticity_error < 1e-11);
    for i in 0..6 {
        for j in 0..6 {
            assert!(
                (positive.dynamical_matrix.re[(i, j)] - negative.dynamical_matrix.re[(i, j)]).abs()
                    < 1e-12
            );
            assert!(
                (positive.dynamical_matrix.im[(i, j)] + negative.dynamical_matrix.im[(i, j)]).abs()
                    < 1e-12
            );
        }
    }
    let mut direction = vec![Vec3::zero(); 6];
    for cell in 0..3 {
        direction[2 * cell].x =
            (2. / 3_f64).sqrt() * (2. * std::f64::consts::PI * cell as f64 / 3.).cos();
    }
    for step in [2e-4, 1e-4] {
        let force = |sign| {
            let mut changed = reference.supercell.clone();
            for (atom, v) in changed.atoms.iter_mut().zip(&direction) {
                atom.position += *v * (sign * step);
            }
            pbc_analytic_gradient(&changed, &params, &options, &pbc)
                .unwrap()
                .gradient
                .iter()
                .zip(&direction)
                .map(|(g, v)| g.dot(*v))
                .sum::<f64>()
        };
        let fd = (force(1.) - force(-1.)) / (2. * step);
        assert!(
            (fd - positive.dynamical_matrix.re[(0, 0)]).abs() < 3e-6,
            "finite-q Hessian {}, force FD {fd}",
            positive.dynamical_matrix.re[(0, 0)]
        );
    }
}

#[test]
fn explicit_ao_cutoff_is_preserved_by_high_level_periodic_entry_points() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut options = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-11,
        charge_tolerance: 1e-9,
        pbc_ao_cutoff: Some(6.),
        ..Default::default()
    };
    let pbc = PbcOptions::for_electronic(&options).unwrap();
    assert_eq!(pbc.ao_cutoff, 6.);
    let explicit = run_pbc_scc(&system, &params, &options, &pbc).unwrap();
    let automatic = run_electronic(&system, &params, options.clone()).unwrap();
    assert!((explicit.total_free - automatic.total_free).abs() < 1e-11);
    let default = run_pbc_scc(&system, &params, &options, &PbcOptions::default()).unwrap();
    assert!(
        (explicit.total_free - default.total_free).abs() > 1e-7,
        "cutoff probe must distinguish the explicit value from the default"
    );
    options.pbc_ao_cutoff = Some(-1.);
    assert!(run_electronic(&system, &params, options).is_err());
}
#[test]
fn periodic_heads_include_self_images_and_match_force_stress_and_parameter_fd() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (mut options, mut pbc) = settings();
    let model = model();
    options.ml_model = Some(Arc::new(model.clone()));
    for (mesh, temperature) in [
        (KMesh::gamma(), 0.),
        (KMesh::monkhorst_pack([2, 2, 2]), 0.),
        (KMesh::monkhorst_pack([2, 2, 2]), 10000.),
    ] {
        pbc.kmesh = mesh;
        options.electronic_temperature = temperature;
        let actual = pbc_analytic_gradient(&system, &params, &options, &pbc).unwrap();
        let stress = pbc_stress(&system, &params, &options, &pbc).unwrap();
        assert!(stress.ml_stress.as_slice().iter().any(|x| x.abs() > 1e-8));
        for h in [2e-4, 1e-4] {
            let mut plus = system.clone();
            let mut minus = system.clone();
            plus.atoms[1].position.x += h;
            minus.atoms[1].position.x -= h;
            let fd = (run_pbc_scc(&plus, &params, &options, &pbc)
                .unwrap()
                .total_free
                - run_pbc_scc(&minus, &params, &options, &pbc)
                    .unwrap()
                    .total_free)
                / (2. * h);
            assert!(
                (actual.gradient[1].x - fd).abs() < 3e-7,
                "k={:?}, T={temperature}, gradient {}, FD {fd}",
                mesh.size,
                actual.gradient[1].x
            );
        }
        let volume = system.lattice.unwrap().volume();
        for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)] {
            let h = 1e-5;
            let fd = (run_pbc_scc(&strain(&system, a, b, h), &params, &options, &pbc)
                .unwrap()
                .total_free
                - run_pbc_scc(&strain(&system, a, b, -h), &params, &options, &pbc)
                    .unwrap()
                    .total_free)
                / (2. * h * volume);
            assert!(
                (stress.stress[(a, b)] - fd).abs() < 2e-8,
                "k={:?}, T={temperature}, stress {a}{b}: {}, FD {fd}",
                mesh.size,
                stress.stress[(a, b)]
            );
        }
        let sensitivity =
            gfn1_rs::pbc::ml::level_sensitivities(&system, &params, &actual.scf).unwrap();
        let parameter_gradient = actual
            .scf
            .ml_parameters
            .as_ref()
            .unwrap()
            .energy_head_gradient(&sensitivity)
            .unwrap();
        let h = 1e-6;
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.heads.get_mut(&1).unwrap().levels[0].weights[2] += h;
        minus.heads.get_mut(&1).unwrap().levels[0].weights[2] -= h;
        let energy = |model| {
            let mut o = options.clone();
            o.ml_model = Some(Arc::new(model));
            run_pbc_scc(&system, &params, &o, &pbc).unwrap().total_free
        };
        let fd = (energy(plus) - energy(minus)) / (2. * h);
        assert!((fd - parameter_gradient[&1].levels[0].weights[2]).abs() < 2e-7);
        let mut wrapped = system.clone();
        wrapped.atoms[1].position += wrapped.lattice.unwrap().cell.col[0];
        assert!(
            (run_pbc_scc(&wrapped, &params, &options, &pbc)
                .unwrap()
                .total_free
                - actual.total_energy)
                .abs()
                < 1e-10
        );
    }
}
#[test]
fn independently_trained_water_model_has_consistent_periodic_energy_and_forces() {
    let system = PeriodicSystem::from_xyz_str(
        "3\nLattice=\"6 0 0 .2 6.3 0 0 .1 6.6\" pbc=\"T T T\"\nO 0 0 0\nH .958 0 0\nH -.25 .925 .03\n", 0., false,
    ).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let (mut options, mut pbc) = settings();
    options.ml_model = Some(Arc::new(MlHamiltonian::builtin_water_clusters().unwrap()));
    pbc.kmesh = KMesh::monkhorst_pack([2, 1, 1]);
    let gradient = pbc_analytic_gradient(&system, &params, &options, &pbc).unwrap();
    let h = 1e-4;
    let mut plus = system.clone();
    let mut minus = system.clone();
    plus.atoms[1].position.y += h;
    minus.atoms[1].position.y -= h;
    let fd = (run_pbc_scc(&plus, &params, &options, &pbc)
        .unwrap()
        .total_free
        - run_pbc_scc(&minus, &params, &options, &pbc)
            .unwrap()
            .total_free)
        / (2. * h);
    assert!((fd - gradient.gradient[1].y).abs() < 3e-7);
    let stress = pbc_stress(&system, &params, &options, &pbc).unwrap();
    let h = 1e-5;
    let fd = (run_pbc_scc(&strain(&system, 0, 1, h), &params, &options, &pbc)
        .unwrap()
        .total_free
        - run_pbc_scc(&strain(&system, 0, 1, -h), &params, &options, &pbc)
            .unwrap()
            .total_free)
        / (2. * h * system.lattice.unwrap().volume());
    assert!((fd - stress.stress[(0, 1)]).abs() < 2e-8);
}
