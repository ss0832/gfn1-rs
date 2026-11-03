// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::linalg::Matrix;
use gfn1_rs::math::Vec3;
use gfn1_rs::*;

#[test]
fn rigid_rank_is_geometric_and_isotope_masses_are_explicit() {
    let co2 = [
        Vec3::new(-2.2, 0.0, 0.0),
        Vec3::zero(),
        Vec3::new(2.2, 0.0, 0.0),
    ];
    let linear = molecular_vibrational_analysis(
        &Matrix::identity(9),
        &co2,
        &[16.0, 12.0, 16.0],
        Default::default(),
    )
    .unwrap();
    assert_eq!(linear.rigid_directions.len(), 5);
    assert_eq!(linear.vibrations.modes.len(), 4);
    let positions = [
        Vec3::zero(),
        Vec3::new(1.4, 1.1, 0.0),
        Vec3::new(-1.4, 1.1, 0.0),
    ];
    let modes = molecular_vibrational_analysis(
        &Matrix::identity(9),
        &positions,
        &[16.0, 1.0, 2.0],
        Default::default(),
    )
    .unwrap();
    assert_eq!(modes.rigid_directions.len(), 6);
    assert_eq!(modes.vibrations.modes.len(), 3);
    for (i, u) in modes.vibrations.modes.iter().enumerate() {
        for (j, v) in modes.vibrations.modes.iter().enumerate() {
            let inner: f64 = u
                .iter()
                .zip(v)
                .enumerate()
                .map(|(k, (a, b))| a * b * [16.0, 1.0, 2.0][k / 3])
                .sum();
            assert!((inner - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    let mut asymmetric = Matrix::identity(9);
    asymmetric[(0, 1)] = 1e-3;
    assert!(molecular_vibrational_analysis(
        &asymmetric,
        &positions,
        &[16.0, 1.0, 1.0],
        Default::default()
    )
    .is_err());
    assert!(molecular_vibrational_analysis(
        &Matrix::identity(9),
        &positions,
        &[16.0, -1.0, 1.0],
        Default::default()
    )
    .is_err());
}

#[test]
fn optimized_h2_and_water_have_finite_analytic_pes_only_vpt2() {
    let params = Gfn1Parameters::builtin().unwrap();
    for xyz in [
        include_str!("../examples/h2.xyz"),
        include_str!("../examples/water.xyz"),
    ] {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let mut options = ElectronicOptions::default();
        options.electronic_temperature = 0.0;
        options.energy_tolerance = 1e-12;
        options.charge_tolerance = 1e-10;
        let optimized = optimize_geometry(
            &system,
            &params,
            GeometryOptimizationOptions {
                gradient_tolerance: 1e-7,
                gradient_options: AnalyticGradientOptions {
                    electronic: options.clone(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert!(optimized.converged);
        let masses: Vec<f64> = optimized
            .system
            .atoms
            .iter()
            .map(|a| gfn1_rs::data_tables::relative_atomic_mass(a.z))
            .collect();
        let result = molecular_vpt2_pes(
            &optimized.system,
            &params,
            options.clone(),
            &masses,
            Default::default(),
            1e-6,
        )
        .unwrap();
        eprintln!(
            "{} atoms: harmonic {:?}, PES-only VPT2 {:?}",
            masses.len(),
            result.normal_modes.vibrations.wavenumbers,
            result.spectrum.fundamentals_cm1
        );
        assert_eq!(
            result.spectrum.fundamentals_cm1.len(),
            if masses.len() == 2 { 1 } else { 3 }
        );
        assert!(result
            .spectrum
            .fundamentals_cm1
            .iter()
            .all(|x| x.is_finite() && *x > 0.0));
        assert!(result.residual_gradient_hartree_per_bohr < 1e-6);
        if masses.len() == 2 {
            let deuterium = molecular_vpt2_pes(
                &optimized.system,
                &params,
                options,
                &[2.01410177812; 2],
                Default::default(),
                1e-6,
            )
            .unwrap();
            let ratio = deuterium.normal_modes.vibrations.wavenumbers[0]
                / result.normal_modes.vibrations.wavenumbers[0];
            assert!((ratio - (masses[0] / 2.01410177812).sqrt()).abs() < 1e-12);
            assert!(deuterium.spectrum.fundamentals_cm1[0] < result.spectrum.fundamentals_cm1[0]);
        }
    }
}
