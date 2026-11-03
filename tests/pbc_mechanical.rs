// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::linalg::Matrix;
use gfn1_rs::pbc::elastic_response::{pbc_elastic_response, AnalyticElasticOptions};
use gfn1_rs::pbc::mechanical::{
    directional_elastic_response, elastic_properties, fit_birch_murnaghan, pbc_relax,
    PeriodicRelaxOptions, STRESS_AU_TO_GPA,
};
use gfn1_rs::pbc::mechanical::{homogeneous_deformation, strain_deformation};
use gfn1_rs::pbc::{pbc_analytic_gradient, pbc_stress_from_scc, KMesh};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PbcOptions, PeriodicSystem};

#[test]
fn birch_murnaghan_recovers_independent_energy_and_pressure_parameters() {
    let (v0, e0, bulk, derivative) = (140.0_f64, -9.0, 0.006, 4.7);
    let volumes: Vec<_> = [0.88, 0.92, 0.965, 0.99, 1.025, 1.055, 1.10, 1.14]
        .iter()
        .map(|x| v0 * x)
        .collect();
    let energies: Vec<_> = volumes
        .iter()
        .map(|v| {
            let x = (v0 / v).powf(2. / 3.);
            e0 + 9. * v0 * bulk / 16.
                * ((x - 1.).powi(3) * derivative + (x - 1.).powi(2) * (6. - 4. * x))
        })
        .collect();
    let fit = fit_birch_murnaghan(&volumes, &energies).unwrap();
    assert!((fit.equilibrium_volume_bohr3 - v0).abs() < 1e-9);
    assert!((fit.minimum_energy_hartree - e0).abs() < 1e-12);
    assert!((fit.bulk_modulus_hartree_per_bohr3 - bulk).abs() < 1e-12);
    assert!((fit.bulk_modulus_pressure_derivative - derivative).abs() < 1e-8);
    assert!(fit.maximum_energy_error_hartree < 1e-12);
    let monotone: Vec<_> = volumes.iter().map(|v| -v).collect();
    assert!(fit_birch_murnaghan(&volumes, &monotone).is_err());
}

fn isotropic_stiffness(lambda: f64, mu: f64) -> Matrix {
    let mut c = Matrix::zeros(6, 6);
    for i in 0..3 {
        for j in 0..3 {
            c[(i, j)] = (lambda + if i == j { 2. * mu } else { 0. }) / STRESS_AU_TO_GPA;
        }
    }
    for i in 3..6 {
        c[(i, i)] = mu / STRESS_AU_TO_GPA;
    }
    c
}

#[test]
fn isotropic_voigt_reuss_hill_directional_moduli_and_acoustic_speeds() {
    let c = isotropic_stiffness(120., 80.);
    let density = 3000.;
    let result = elastic_properties(&c, density).unwrap();
    assert!((result.bulk_hill_gpa - 173.3333333333333).abs() < 1e-10);
    assert!((result.shear_hill_gpa - 80.).abs() < 1e-10);
    assert!((result.young_hill_gpa - 208.).abs() < 1e-10);
    assert!((result.poisson_hill - 0.3).abs() < 1e-12);
    assert!(result.universal_anisotropy.abs() < 1e-12);
    for (n, m) in [([1., 0., 0.], [0., 1., 0.]), ([1., 2., 3.], [-2., 1., 0.])] {
        let d = directional_elastic_response(&c, density, n, m).unwrap();
        assert!((d.young_gpa - 208.).abs() < 1e-10);
        assert!((d.poisson_ratio - 0.3).abs() < 1e-12);
        assert!((d.shear_gpa - 80.).abs() < 1e-10);
        for i in 0..2 {
            assert!((d.sound_speeds_m_per_s[i] - (80e9_f64 / density).sqrt()).abs() < 1e-8);
        }
        assert!((d.sound_speeds_m_per_s[2] - (280e9_f64 / density).sqrt()).abs() < 1e-8);
    }
    let mut unstable = c.clone();
    unstable[(3, 3)] = -c[(3, 3)];
    assert!(elastic_properties(&unstable, density).is_err());
    assert!(directional_elastic_response(&c, density, [1., 0., 0.], [1., 1., 0.]).is_err());
    let mut orthotropic = Matrix::zeros(6, 6);
    for (i, value) in [200., 170., 140., 50., 60., 70.].into_iter().enumerate() {
        orthotropic[(i, i)] = value / STRESS_AU_TO_GPA;
    }
    for (i, j, value) in [(0, 1, 60.), (0, 2, 50.), (1, 2, 40.)] {
        orthotropic[(i, j)] = value / STRESS_AU_TO_GPA;
        orthotropic[(j, i)] = value / STRESS_AU_TO_GPA;
    }
    let principal =
        directional_elastic_response(&orthotropic, density, [1., 0., 0.], [0., 1., 0.]).unwrap();
    let determinant = 200. * (170. * 140. - 40. * 40.) - 60. * (60. * 140. - 40. * 50.)
        + 50. * (60. * 40. - 170. * 50.);
    assert!((principal.young_gpa - determinant / (170. * 140. - 40. * 40.)).abs() < 1e-10);
    assert!(
        (principal.poisson_ratio - (60. * 140. - 50. * 40.) / (170. * 140. - 40. * 40.)).abs()
            < 1e-12
    );
    assert!((principal.shear_gpa - 70.).abs() < 1e-10);
    let rotated =
        directional_elastic_response(&orthotropic, density, [1., 1., 0.], [-1., 1., 0.]).unwrap();
    let expected = [[135., 65., 0.], [65., 120., 0.], [0., 0., 55.]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((rotated.christoffel_gpa[i][j] - expected[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn fixed_cell_relaxation_converges_without_changing_sampling_or_temperature() {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\"\nH 0 0 0\nH 1.1 0.03 -0.02\n",
        0.,
        false,
    )
    .unwrap();
    let parameters = Gfn1Parameters::builtin().unwrap();
    let electronic = ElectronicOptions {
        electronic_temperature: 300.,
        ..Default::default()
    };
    let pbc = PbcOptions::default();
    let result = pbc_relax(
        &system,
        &parameters,
        &electronic,
        &pbc,
        &PeriodicRelaxOptions::default(),
    )
    .unwrap();
    assert!(
        result.converged,
        "{} at {:?}",
        result.termination,
        result.history.last()
    );
    assert_eq!(
        result.system.lattice.unwrap().cell,
        system.lattice.unwrap().cell
    );
    assert_eq!(result.electronic_temperature_kelvin, 300.);
    assert!(result.gradient.iter().map(|v| v.norm()).fold(0., f64::max) < 1e-7);
    assert!(result.energy_hartree < result.history[0].energy_hartree);
    let limited = pbc_relax(
        &system,
        &parameters,
        &electronic,
        &pbc,
        &PeriodicRelaxOptions {
            max_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        !limited.converged,
        "an iteration cap must not be called convergence"
    );
    assert!(result
        .history
        .windows(2)
        .all(|w| w[1].enthalpy_hartree <= w[0].enthalpy_hartree));
    assert!(result.work.scf_evaluations >= result.iterations + 1);
    let checked = gfn1_rs::pbc::run_pbc_scc_with_guess(
        &result.system,
        &parameters,
        &electronic,
        &pbc,
        Some(&result.scf_guess),
    )
    .unwrap();
    assert!((checked.total_free - result.energy_hartree).abs() < 2e-11);
}

#[test]
fn ionic_variable_cell_relaxation_reuses_converged_scc_without_relaxing_criteria() {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"0 2.013 2.013 2.013 0 2.013 2.013 2.013 0\" pbc=\"T T T\"\nLi 0 0 0\nF 2.013 0 0\n",0.,false,
    ).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        mixing: 0.1,
        max_scc: 1000,
        ..Default::default()
    };
    let pbc = PbcOptions {
        ao_cutoff: 30.,
        ..Default::default()
    };
    let controls = PeriodicRelaxOptions {
        relax_cell: true,
        max_iterations: 40,
        ..Default::default()
    };
    let result = pbc_relax(&system, &params, &options, &pbc, &controls).unwrap();
    assert!(
        result.converged,
        "{} {:?}",
        result.termination, result.last_rejected_trial
    );
    assert!(result.iterations <= 35);
    assert!(
        result
            .history
            .last()
            .unwrap()
            .maximum_gradient_hartree_per_bohr
            <= controls.force_tolerance
    );
    assert!(
        result
            .history
            .last()
            .unwrap()
            .maximum_stress_residual_hartree_per_bohr3
            <= controls.stress_tolerance
    );
    assert_eq!(result.work.failed_scf_evaluations, 0);
    assert!(
        result.work.scf_iterations < 1200,
        "charge continuation should avoid repeated cold SCC failures"
    );
    assert!(result
        .history
        .windows(2)
        .all(|w| w[1].enthalpy_hartree <= w[0].enthalpy_hartree));
    let state = gfn1_rs::pbc::run_pbc_scc_with_guess(
        &result.system,
        &params,
        &options,
        &pbc,
        Some(&result.scf_guess),
    )
    .unwrap();
    assert!((state.total_free - result.energy_hartree).abs() < 2e-11);
    let energy = state.total_free;
    let gradient =
        gfn1_rs::pbc::pbc_gradient_from_scc(&result.system, &params, state, &options, &pbc)
            .unwrap();
    assert!(gradient.max_gradient <= controls.force_tolerance);
    let stress = pbc_stress_from_scc(
        &result.system,
        &params,
        gradient.scf,
        &options,
        &pbc,
        &result.system.lattice.unwrap(),
    )
    .unwrap();
    assert!(stress
        .stress
        .as_slice()
        .iter()
        .all(|v| v.abs() <= controls.stress_tolerance));
    eprintln!("LiF warm SCC relaxation: {}iterations {}SCFevaluations {}SCCiterations {:.4}s finalE{energy:.15}Ha",result.iterations,result.work.scf_evaluations,result.work.scf_iterations,result.work.elapsed_seconds);
    // An arbitrary origin cannot turn homogeneous strain into a huge atomic
    // optimization step. The former total-displacement cap throttled this
    // physically identical problem, even with good electronic guesses.
    let mut translated = system.clone();
    let offset = gfn1_rs::math::Vec3::new(1000., -800., 500.);
    for atom in &mut translated.atoms {
        atom.position += offset;
    }
    let shifted = pbc_relax(&translated, &params, &options, &pbc, &controls).unwrap();
    assert!(
        shifted.converged,
        "origin-shifted periodic optimization: {}",
        shifted.termination
    );
    assert!(
        shifted.iterations <= 35,
        "origin must not throttle the cell step"
    );
    assert!((shifted.energy_hartree - result.energy_hartree).abs() < 2e-9);
    assert!(
        (shifted.system.lattice.unwrap().volume() - result.system.lattice.unwrap().volume()).abs()
            < 2e-3
    );
    eprintln!(
        "origin-shifted LiF: {}steps {}SCCiterations {:.4}s",
        shifted.iterations, shifted.work.scf_iterations, shifted.work.elapsed_seconds
    );
}

fn stressed_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    strain: [f64; 6],
) -> (Vec<f64>, [f64; 6]) {
    let deformed = homogeneous_deformation(system, strain).unwrap();
    let f = strain_deformation(strain).unwrap();
    let inv = gfn1_rs::linalg::DenseLu::factor(&f)
        .unwrap()
        .solve_matrix(&Matrix::identity(3))
        .unwrap();
    let gradient = pbc_analytic_gradient(&deformed, params, options, pbc).unwrap();
    assert!(gradient.scf.converged);
    let general: Vec<_> = gradient
        .gradient
        .iter()
        .flat_map(|g| {
            let r = g.to_array();
            (0..3)
                .map(|a| (0..3).map(|b| f[(b, a)] * r[b]).sum::<f64>())
                .collect::<Vec<_>>()
        })
        .collect();
    let stress = pbc_stress_from_scc(
        &deformed,
        params,
        gradient.scf,
        options,
        pbc,
        &deformed.lattice.unwrap(),
    )
    .unwrap()
    .stress;
    let pulled = inv
        .matmul(&stress)
        .unwrap()
        .matmul(&inv.transpose())
        .unwrap();
    let jac = deformed.lattice.unwrap().volume() / system.lattice.unwrap().volume();
    let pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    (
        general,
        std::array::from_fn(|s| {
            let (a, b) = pairs[s];
            0.5 * jac * (pulled[(a, b)] + pulled[(b, a)])
        }),
    )
}

#[test]
fn all_analytic_elastic_and_internal_strain_components_include_structural_ml() {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"5.6 .3 0 0 6.1 .2 .1 0 6.4\" pbc=\"T T T\"\nH .2 .3 .1\nH 1.6 .45 .2\n",
        0.,
        true,
    )
    .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let topology = |s: &PeriodicSystem, cutoff: f64| {
        gfn1_rs::pairlist::unique_short_range_pairs(s, cutoff)
            .unwrap()
            .into_iter()
            .map(|p| (p.i, p.j, p.offset.n))
            .collect::<std::collections::BTreeSet<_>>()
    };
    let mut coarse_crossings = 0;
    for column in 0..6 {
        for cutoff in [20., 25., 30., 50.] {
            let base = topology(&system, cutoff);
            for sign in [-1., 1.] {
                let mut strain = [0.; 6];
                strain[column] = sign * 2e-4;
                if topology(&homogeneous_deformation(&system, strain).unwrap(), cutoff) != base {
                    coarse_crossings += 1;
                    eprintln!(
                        "coarse strain stencil crosses cutoff {cutoff} column {column} sign {sign}"
                    );
                }
                strain[column] = sign * 2e-5;
                assert_eq!(
                    topology(&homogeneous_deformation(&system, strain).unwrap(), cutoff),
                    base,
                    "fine strain oracle must preserve the retained image pairs at cutoff {cutoff}"
                );
            }
        }
    }
    eprintln!(
        "coarse stencil cutoff crossings: {coarse_crossings}; fine stencil pair sets unchanged"
    );
    for (ml, temperature, mesh, heteronuclear) in [
        (false, 300., KMesh::gamma(), false),
        (true, 300., KMesh::monkhorst_pack([2, 1, 1]), false),
        (true, 30000., KMesh::gamma(), false),
        (false, 30000., KMesh::gamma(), true),
    ] {
        let mut system = system.clone();
        if heteronuclear {
            system.atoms[1].z = 9;
        }
        let mut options = ElectronicOptions {
            electronic_temperature: temperature,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            ..Default::default()
        };
        if ml {
            let mut model = gfn1_rs::ml::MlHamiltonian::zero(vec![1], vec![1.2, 3.5, 5.8], 9.);
            model.heads.get_mut(&1).unwrap().levels[0].weights = vec![0.001, -0.0015, 0.002];
            model.heads.get_mut(&1).unwrap().correction.weights = vec![0.0004, 0.0002, -0.0003];
            options.ml_model = Some(std::sync::Arc::new(model));
        }
        let pbc = PbcOptions {
            kmesh: mesh,
            ao_cutoff: 20.,
            ..Default::default()
        };
        let analytic = pbc_elastic_response(
            &system,
            &params,
            &options,
            &pbc,
            &AnalyticElasticOptions::default(),
        )
        .unwrap();
        let nuclear = gfn1_rs::pbc::pbc_kpoint_hessian(&system, &params, &options, &pbc).unwrap();
        let nuclear_error = analytic.nuclear_hessian.max_abs_diff(&nuclear.hessian);
        let mut elastic_error = 0.0_f64;
        let mut mixed_error = 0.0_f64;
        for j in 0..6 {
            let evaluate = |h: f64| {
                let mut eps = [0.; 6];
                eps[j] = h;
                stressed_gradient(&system, &params, &options, &pbc, eps)
            };
            let step = 2e-5;
            let p = evaluate(step);
            let m = evaluate(-step);
            let ph = evaluate(step / 2.);
            let mh = evaluate(-step / 2.);
            for i in 0..6 {
                let coarse = (p.1[i] - m.1[i]) / (2. * step);
                let fine = (ph.1[i] - mh.1[i]) / step;
                let fd = (4. * fine - coarse) / 3.;
                elastic_error = elastic_error.max((fd - analytic.clamped_stiffness[(i, j)]).abs());
                let coarse = (p.0[i] - m.0[i]) / (2. * step);
                let fine = (ph.0[i] - mh.0[i]) / step;
                let fd = (4. * fine - coarse) / 3.;
                if (fd - analytic.internal_strain_coupling[(i, j)]).abs() > mixed_error {
                    eprintln!(
                        "mixed maximum ({i},{j}) analytic {} FD {} coarse {} fine {}",
                        analytic.internal_strain_coupling[(i, j)],
                        fd,
                        coarse,
                        fine
                    );
                }
                mixed_error =
                    mixed_error.max((fd - analytic.internal_strain_coupling[(i, j)]).abs());
            }
        }
        eprintln!("analytic elastic HF={heteronuclear} ML={ml} T={temperature} mesh={:?}: K error {nuclear_error:e}, all36C {elastic_error:e}, all36Lambda {mixed_error:e}, raw asym {}",mesh.size,analytic.raw_max_asymmetry);
        assert!(
            nuclear_error < 2e-8,
            "nuclear block differs from existing analytic Hessian {nuclear_error:e}"
        );
        assert!(elastic_error < 2e-7, "elastic error {elastic_error:e}");
        assert!(mixed_error < 2e-7, "mixed response error {mixed_error:e}");
        assert!(
            analytic.raw_max_asymmetry < 2e-7,
            "raw joint Hessian asymmetric"
        );
    }
}

#[test]
fn relaxed_ionic_ml_response_matches_independently_relaxed_strained_cells() {
    // A non-centrosymmetric crystal gives a nonzero internal shear response.
    // These small artificial learned heads test derivatives, not model accuracy.
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"0 2.18 2.18 2.18 0 2.18 2.18 2.18 0\" pbc=\"T T T\"\nSi 0 0 0\nC 1.09 1.09 1.09\n",
        0., false,
    ).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut model = gfn1_rs::ml::MlHamiltonian::zero(vec![6, 14], vec![2., 4., 6.], 9.);
    for (z, scale) in [(6, 1.), (14, -0.7)] {
        let head = model.heads.get_mut(&z).unwrap();
        for (l, level) in head.levels.iter_mut().enumerate() {
            level.weights = vec![
                0.0003 * scale,
                -0.0002 * scale,
                0.0001 * (l + 1) as f64,
                -0.0001 * scale,
                0.00015 * scale,
                -0.00005,
            ];
        }
        head.correction.weights = vec![
            0.0001 * scale,
            0.0002,
            -0.0001,
            -0.0002 * scale,
            0.0001,
            -0.00005,
        ];
    }
    let electronic = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        ml_model: Some(std::sync::Arc::new(model)),
        ..Default::default()
    };
    let pbc = PbcOptions {
        ao_cutoff: 23.713,
        ..Default::default()
    };
    let analytic = pbc_elastic_response(
        &system,
        &params,
        &electronic,
        &pbc,
        &AnalyticElasticOptions::default(),
    )
    .unwrap();
    let du = analytic
        .internal_displacement_strain
        .as_ref()
        .unwrap_or_else(|| {
            panic!(
                "stable stationary SiC ionic response: {:?}",
                analytic.relaxed_response_error
            )
        });
    let relaxed_c = analytic.relaxed_stiffness.as_ref().unwrap();
    assert!(
        du.as_slice().iter().map(|x| x.abs()).fold(0., f64::max) > 0.1,
        "nontrivial internal strain response required"
    );
    let relax = PeriodicRelaxOptions {
        force_tolerance: 1e-10,
        max_iterations: 100,
        ..Default::default()
    };
    let mut displacement_error = 0.0_f64;
    let mut elastic_error = 0.0_f64;
    for column in 0..6 {
        let evaluate = |h: f64| {
            let mut eta = [0.; 6];
            eta[column] = h;
            let deformed = homogeneous_deformation(&system, eta).unwrap();
            let result = pbc_relax(&deformed, &params, &electronic, &pbc, &relax).unwrap();
            assert!(
                result.converged,
                "independent strained-cell relaxation: {} {:?}",
                result.termination, result.last_rejected_trial
            );
            let inv = gfn1_rs::linalg::DenseLu::factor(&strain_deformation(eta).unwrap())
                .unwrap()
                .solve_matrix(&Matrix::identity(3))
                .unwrap();
            let mut u = vec![0.; 6];
            for atom in 0..2 {
                for i in 0..3 {
                    u[3 * atom + i] = (0..3)
                        .map(|j| inv[(i, j)] * result.system.atoms[atom].position.to_array()[j])
                        .sum::<f64>()
                        - system.atoms[atom].position.to_array()[i];
                }
            }
            for i in 0..3 {
                let mean = 0.5 * (u[i] + u[i + 3]);
                u[i] -= mean;
                u[i + 3] -= mean;
            }
            let pulled = inv
                .matmul(&result.stress)
                .unwrap()
                .matmul(&inv.transpose())
                .unwrap();
            let jac = result.system.lattice.unwrap().volume() / system.lattice.unwrap().volume();
            let pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
            let stress: [f64; 6] = std::array::from_fn(|i| jac * pulled[pairs[i]]);
            (u, stress)
        };
        let h = 1e-4;
        let plus = evaluate(h);
        let minus = evaluate(-h);
        for i in 0..6 {
            displacement_error = displacement_error
                .max(((plus.0[i] - minus.0[i]) / (2. * h) - du[(i, column)]).abs());
            elastic_error = elastic_error
                .max(((plus.1[i] - minus.1[i]) / (2. * h) - relaxed_c[(i, column)]).abs());
        }
    }
    eprintln!("independent relaxed-ion ML SiC: all36du {displacement_error:e} bohr, all36Crel {elastic_error:e} Ha/bohr³");
    assert!(
        displacement_error < 2e-5,
        "relaxed displacement derivative error {displacement_error:e}"
    );
    assert!(
        elastic_error < 2e-6,
        "relaxed stiffness error {elastic_error:e}"
    );
}
