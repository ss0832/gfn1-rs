// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::constants::{HARTREE_TO_EV, HARTREE_TO_WAVENUMBER, KB_HARTREE_PER_K};
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
use gfn1_rs::pbc::spectra::*;
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
fn diamond() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str("2\nLattice=\"0 1.7835 1.7835 1.7835 0 1.7835 1.7835 1.7835 0\" pbc=\"T T T\"\nC 0 0 0\nC .89175 .89175 .89175\n",0.,false).unwrap()
}
fn settings() -> (ElectronicOptions, PbcOptions) {
    (
        ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            ..Default::default()
        },
        PbcOptions {
            ao_cutoff: 20.,
            ..Default::default()
        },
    )
}
fn learned_hydrogen() -> MlHamiltonian {
    let mut m = MlHamiltonian::zero(vec![1], vec![1.2, 3.5, 5.8], 9.);
    m.heads.get_mut(&1).unwrap().levels[0].weights = vec![0.003, -0.0025, 0.004];
    m.heads.get_mut(&1).unwrap().correction.weights = vec![0.0004, 0.0002, -0.0003];
    m
}
#[test]
fn harmonic_partition_functions_obey_thermodynamic_identities_and_limits() {
    let frequencies = vec![vec![0., 1000., 2000.], vec![0., 1200., 1800.]];
    let weights = vec![0.25, 0.75];
    let h = 0.02;
    let t = harmonic_thermodynamics(
        &frequencies,
        &weights,
        &[0., 300. - h, 300., 300. + h, 1e9],
        1e-3,
    )
    .unwrap();
    let zpe = (0.25 * 3000. + 0.75 * 3000.) / (2. * HARTREE_TO_WAVENUMBER);
    assert!((t[0].zero_point_energy_hartree - zpe).abs() < 1e-16);
    assert_eq!(t[0].free_energy_hartree, t[0].internal_energy_hartree);
    assert_eq!(t[0].entropy_hartree_per_kelvin, 0.);
    assert_eq!(t[0].heat_capacity_hartree_per_kelvin, 0.);
    assert_eq!(t[0].excluded_zero_modes_per_primitive_cell, 1.);
    assert_eq!(t[0].included_modes_per_primitive_cell, 2.);
    let c = &t[2];
    assert!(
        (c.internal_energy_hartree - c.free_energy_hartree - 300. * c.entropy_hartree_per_kelvin)
            .abs()
            < 1e-16
    );
    assert!(
        ((t[3].free_energy_hartree - t[1].free_energy_hartree) / (2. * h)
            + c.entropy_hartree_per_kelvin)
            .abs()
            < 2e-14
    );
    assert!(
        ((t[3].internal_energy_hartree - t[1].internal_energy_hartree) / (2. * h)
            - c.heat_capacity_hartree_per_kelvin)
            .abs()
            < 2e-14
    );
    assert!((t[4].heat_capacity_hartree_per_kelvin / (2. * KB_HARTREE_PER_K) - 1.).abs() < 1e-10);
    assert!((t[4].internal_energy_hartree / (2. * KB_HARTREE_PER_K * 1e9) - 1.).abs() < 1e-10);
}
#[test]
fn unstable_or_invalid_thermal_inputs_are_rejected_without_clipping() {
    let err = harmonic_thermodynamics(&[vec![-10., 100.]], &[1.], &[300.], 1e-3).unwrap_err();
    assert!(err.to_string().contains("imaginary mode"));
    assert!(harmonic_thermodynamics(&[vec![100.]], &[0.8], &[300.], 1e-3).is_err());
    assert!(harmonic_thermodynamics(&[vec![100.]], &[1.], &[-1.], 1e-3).is_err());
    let tiny = harmonic_thermodynamics(&[vec![-1e-4, 100.]], &[1.], &[1e-10, 1e8], 1e-3).unwrap();
    assert_eq!(tiny[0].excluded_zero_modes_per_primitive_cell, 1.);
    assert!(tiny.iter().all(
        |t| t.free_energy_hartree.is_finite() && t.heat_capacity_hartree_per_kelvin.is_finite()
    ));
}
#[test]
fn invalid_band_directions_are_rejected_before_the_reference_scf() {
    let mut system = hydrogen();
    system.lattice.as_mut().unwrap().periodic[2] = false;
    let params = Gfn1Parameters::builtin().unwrap();
    let (options, pbc) = settings();
    let error = pbc_band_structure(&system, &params, &options, &pbc, &[[0., 0., 0.2]]).unwrap_err();
    assert!(error.to_string().contains("non-periodic direction"));
    assert!(pbc_band_structure(&system, &params, &options, &pbc, &[]).is_err());
}
#[test]
fn gaussian_phonon_dos_retains_signed_modes_and_state_normalization() {
    let values = vec![vec![-100., 500., 900.], vec![-90., 450., 1000.]];
    let grid: Vec<f64> = (0..2501).map(|i| -200. + i as f64 * 0.5).collect();
    let dos = phonon_dos(&values, &[0.4, 0.6], &grid, 5.).unwrap();
    assert_eq!(dos.expected_states, 3.);
    assert!((dos.integrated_states - 3.).abs() < 2e-13);
    assert!(dos.density[200] > 0.02); // -100 cm^-1 remains in the signed DOS
    let narrow = phonon_dos(&values, &[0.4, 0.6], &[-1., 0., 1.], 5.).unwrap();
    assert!(narrow.integrated_states < 0.01); // no artificial finite-grid renormalization
    assert!(phonon_dos(&values, &[0.4, 0.6], &[0., 0., 1.], 5.).is_err());
}
#[test]
fn same_scc_bands_obey_periodicity_and_physical_spin_state_count() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (options, mut pbc) = settings();
    pbc.kmesh = KMesh::monkhorst_pack([2, 2, 2]);
    let reference = run_pbc_scc(&system, &params, &options, &pbc).unwrap();
    let points: Vec<_> = reference.kpoints.iter().map(|k| k.fractional).collect();
    let bands = band_structure_from_scc(&system, &reference, &options, &points).unwrap();
    let kt = KB_HARTREE_PER_K * options.electronic_temperature;
    let from_eigenvalues: f64 = bands
        .eigenvalues_hartree
        .iter()
        .zip(&reference.kpoints)
        .map(|(row, k)| {
            row.iter()
                .map(|&e| {
                    let x = (e - reference.fermi_level) / kt;
                    let f = if x > 40. {
                        0.
                    } else if x < -40. {
                        1.
                    } else {
                        1. / (1. + x.exp())
                    };
                    2. * f * e
                })
                .sum::<f64>()
                * k.weight
        })
        .sum();
    let from_density: f64 = reference
        .ew_density_k
        .iter()
        .zip(&reference.hs_k)
        .zip(&reference.kpoints)
        .map(|((w, (_, s)), k)| w.real_trace_product(s) * k.weight)
        .sum();
    assert!((from_eigenvalues - from_density).abs() < 2e-12);
    let periodic = band_structure_from_scc(
        &system,
        &reference,
        &options,
        &[
            [0.07, 0.13, -0.04],
            [1.07, 0.13, -0.04],
            [-0.07, -0.13, 0.04],
        ],
    )
    .unwrap();
    for b in 0..reference.basis.len() {
        assert!(
            (periodic.eigenvalues_hartree[0][b] - periodic.eigenvalues_hartree[1][b]).abs() < 1e-12
        );
        assert!(
            (periodic.eigenvalues_hartree[0][b] - periodic.eigenvalues_hartree[2][b]).abs() < 1e-12
        );
    }
    let low = bands
        .eigenvalues_hartree
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min)
        * HARTREE_TO_EV
        - 2.;
    let high = bands
        .eigenvalues_hartree
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        * HARTREE_TO_EV
        + 2.;
    let grid: Vec<f64> = (0..4001)
        .map(|i| low + (high - low) * i as f64 / 4000.)
        .collect();
    let weights: Vec<_> = reference.kpoints.iter().map(|k| k.weight).collect();
    let dos = electronic_dos(&bands, &weights, &grid, 0.15).unwrap();
    assert!((dos.expected_states - 2. * reference.basis.len() as f64).abs() < 1e-12);
    assert!((dos.integrated_states - dos.expected_states).abs() < 1e-10);
    assert_eq!(bands.electronic_temperature_kelvin, 300.);
}
#[test]
fn ml_bands_keep_learned_levels_and_distinguish_scalar_energy_corrections() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (plain, pbc) = settings();
    let path = [[0., 0., 0.], [0.2, 0.1, 0.]];
    let base = pbc_band_structure(&system, &params, &plain, &pbc, &path).unwrap();
    let mut zero = plain.clone();
    zero.ml_model = Some(Arc::new(MlHamiltonian::zero(
        vec![1],
        vec![1.2, 3.5, 5.8],
        9.,
    )));
    let restore = pbc_band_structure(&system, &params, &zero, &pbc, &path).unwrap();
    assert_eq!(base.eigenvalues_hartree, restore.eigenvalues_hartree);
    let model = learned_hydrogen();
    let mut learned = plain.clone();
    learned.ml_model = Some(Arc::new(model.clone()));
    let reference = run_pbc_scc(&system, &params, &learned, &pbc).unwrap();
    let bands = band_structure_from_scc(&system, &reference, &learned, &path).unwrap();
    assert_eq!(
        bands.ml_model_sha256,
        Some(model.canonical_sha256().unwrap())
    );
    assert!(bands
        .eigenvalues_hartree
        .iter()
        .flatten()
        .zip(base.eigenvalues_hartree.iter().flatten())
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let mut corrected = model.clone();
    corrected.heads.get_mut(&1).unwrap().correction.bias += 0.002;
    let mut shifted = plain.clone();
    shifted.ml_model = Some(Arc::new(corrected));
    let scalar = pbc_band_structure(&system, &params, &shifted, &pbc, &path).unwrap();
    assert_eq!(bands.eigenvalues_hartree, scalar.eigenvalues_hartree);
    assert!(
        (scalar.ml_correction_energy_hartree - bands.ml_correction_energy_hartree - 0.004).abs()
            < 1e-12
    );
    assert!(band_structure_from_scc(&system, &reference, &plain, &path).is_err());
    assert!(band_structure_from_scc(&system, &reference, &shifted, &path).is_err());
}
#[test]
fn analytic_ml_supercell_mesh_reuses_the_actual_hessian() {
    let system = hydrogen();
    let params = Gfn1Parameters::builtin().unwrap();
    let (mut options, pbc) = settings();
    options.ml_model = Some(Arc::new(learned_hydrogen()));
    let reference = AnalyticSupercellPhonons::build(
        &system,
        &params,
        &options,
        &pbc,
        &[1., 1.],
        PhononSupercellOptions {
            repetitions: [2, 1, 1],
            ..Default::default()
        },
    )
    .unwrap();
    let mesh = sample_phonon_mesh(&reference).unwrap();
    assert_eq!(mesh.q_fractional, vec![[0., 0., 0.], [0.5, 0., 0.]]);
    assert_eq!(mesh.weights, vec![0.5, 0.5]);
    assert_eq!(mesh.electronic_temperature_kelvin, 300.);
    assert!(mesh.reference_energy_hartree_per_primitive_cell.is_finite());
    assert_eq!(
        mesh.ml_model_sha256,
        Some(learned_hydrogen().canonical_sha256().unwrap())
    );
    assert!(mesh.method_citations.contains(&"Li2026MLxTB"));
    for (q, row) in mesh.q_fractional.iter().zip(&mesh.wavenumbers_cm1) {
        assert_eq!(*row, reference.at_q(*q).unwrap().wavenumbers_cm1);
        assert_eq!(row.len(), 6);
    }
    assert!(sample_phonon_qpoints(&reference, &[[0.25, 0., 0.]], &[1.]).is_err());
}
#[test]
fn real_diamond_structure_spectrum_has_expected_symmetry_and_mode_count() {
    let system = diamond();
    let params = Gfn1Parameters::builtin().unwrap();
    let (options, mut pbc) = settings();
    pbc.ao_cutoff = 30.;
    let bands = pbc_band_structure(
        &system,
        &params,
        &options,
        &pbc,
        &[[0., 0., 0.], [0.5, 0., 0.], [0.5, 0.5, 0.5]],
    )
    .unwrap();
    assert_eq!(bands.eigenvalues_hartree[0].len(), 8);
    assert!(bands
        .sampled_gap
        .as_ref()
        .unwrap()
        .indirect_gap_hartree
        .is_finite());
    let reference = AnalyticSupercellPhonons::build(
        &system,
        &params,
        &options,
        &pbc,
        &[12.0, 12.0],
        PhononSupercellOptions::default(),
    )
    .unwrap();
    let mesh = sample_phonon_mesh(&reference).unwrap();
    let frequencies = &mesh.wavenumbers_cm1[0];
    assert!(frequencies[..3].iter().all(|x| x.abs() < 0.01));
    assert!((frequencies[5] - frequencies[3]).abs() < 1e-5);
    let thermo =
        harmonic_thermodynamics(&mesh.wavenumbers_cm1, &mesh.weights, &[0., 300.], 0.01).unwrap();
    assert_eq!(thermo[0].included_modes_per_primitive_cell, 3.);
    println!("diamond representative a=3.567 A, isotope masses12.0amu, Gamma only: optic {:?} cm^-1, ZPE={} Ha, Cv300={} Ha/K; this smoke is not a bulk q-integrated thermodynamic result",&frequencies[3..],thermo[0].zero_point_energy_hartree,thermo[1].heat_capacity_hartree_per_kelvin);
}
#[test]
fn qha_thermal_expansion_uses_entropy_derivatives_and_vanishing_zero_t_limits() {
    let v0 = 100.;
    let bulk = 0.01;
    let bp = 4.;
    let volumes: Vec<f64> = (0..9).map(|i| v0 * (0.92 + 0.02 * i as f64)).collect();
    let static_energy: Vec<f64> = volumes
        .iter()
        .map(|&v| {
            let f = 0.5 * ((v0 / v).powf(2. / 3.) - 1.);
            -2. + 4.5 * bulk * v0 * f * f * (1. + (bp - 4.) * f)
        })
        .collect();
    let gamma = 1.2;
    let frequencies: Vec<Vec<Vec<f64>>> = volumes
        .iter()
        .map(|&v| {
            vec![vec![
                1000. * (v / v0).powf(-gamma),
                1500. * (v / v0).powf(-gamma),
                2000. * (v / v0).powf(-gamma),
            ]]
        })
        .collect();
    let qha = quasiharmonic_zero_pressure(
        &volumes,
        &static_energy,
        &frequencies,
        &[1.],
        &[0., 300., 900.],
        1e-3,
    )
    .unwrap();
    assert_eq!(qha[0].thermal_expansion_per_kelvin, 0.);
    assert_eq!(qha[0].heat_capacity_cp_hartree_per_kelvin, 0.);
    assert!(qha[0].equilibrium_volume_bohr3 > v0); // positive zero-point pressure
    assert!(qha[2].equilibrium_volume_bohr3 > qha[1].equilibrium_volume_bohr3);
    for point in &qha[1..] {
        let expected = gamma * point.heat_capacity_cv_hartree_per_kelvin
            / (point.bulk_modulus_hartree_per_bohr3 * point.equilibrium_volume_bohr3);
        assert!((point.thermal_expansion_per_kelvin / expected - 1.).abs() < 0.003);
        assert!(
            point.heat_capacity_cp_hartree_per_kelvin >= point.heat_capacity_cv_hartree_per_kelvin
        );
    }
}
