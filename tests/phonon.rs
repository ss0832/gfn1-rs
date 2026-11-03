// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::phonon::*;
use gfn1_rs::*;

fn diamond() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str("2\nLattice=\"0 1.7835 1.7835 1.7835 0 1.7835 1.7835 1.7835 0\" pbc=\"T T T\"\nC 0 0 0\nC .89175 .89175 .89175\n",0.0,false).unwrap()
}
#[test]
fn commensurate_diamond_phonons_are_hermitian_and_match_supercell_force_fd() {
    let primitive = diamond();
    let params = Gfn1Parameters::builtin().unwrap();
    let electronic = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ..Default::default()
    };
    let pbc = PbcOptions::default();
    let reference = AnalyticSupercellPhonons::build(
        &primitive,
        &params,
        &electronic,
        &pbc,
        &[12.0, 12.0],
        PhononSupercellOptions {
            repetitions: [3, 1, 1],
            ..Default::default()
        },
    )
    .unwrap();
    let positive = reference.at_q([1.0 / 3.0, 0.0, 0.0]).unwrap();
    let negative = reference.at_q([-1.0 / 3.0, 0.0, 0.0]).unwrap();
    assert!(positive.raw_hermiticity_error < 1e-12);
    assert_eq!(positive.wavenumbers_cm1.len(), 6);
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
    let gamma = reference.at_q([0.0; 3]).unwrap();
    for axis in 0..3 {
        for row in 0..6 {
            let sum =
                gamma.dynamical_matrix.re[(row, axis)] + gamma.dynamical_matrix.re[(row, 3 + axis)];
            assert!(sum.abs() < 1e-7, "raw acoustic contraction {sum:e}");
        }
    }
    let mut direction = vec![Vec3::zero(); 6];
    for cell in 0..3 {
        direction[2 * cell].x = (2.0 / 3.0_f64).sqrt()
            * (2.0 * std::f64::consts::PI * cell as f64 / 3.0).cos()
            / 12.0_f64.sqrt();
    }
    for h in [2e-3, 1e-3] {
        let force = |sign: f64| {
            let mut system = reference.supercell.clone();
            for (atom, v) in system.atoms.iter_mut().zip(&direction) {
                atom.position += *v * (sign * h);
            }
            let gradient = pbc_analytic_gradient(&system, &params, &electronic, &pbc)
                .unwrap()
                .gradient;
            gradient
                .iter()
                .zip(&direction)
                .map(|(g, v)| g.dot(*v))
                .sum::<f64>()
        };
        let fd = (force(1.0) - force(-1.0)) / (2.0 * h);
        assert!(
            (fd - positive.dynamical_matrix.re[(0, 0)]).abs() < 2e-6,
            "finite-q analytic {} supercell FD {fd}",
            positive.dynamical_matrix.re[(0, 0)]
        );
    }
    assert!(reference.at_q([0.25, 0.0, 0.0]).is_err());
}
#[test]
fn supercell_size_is_rejected_before_scf() {
    let error = AnalyticSupercellPhonons::build(
        &diamond(),
        &Gfn1Parameters::builtin().unwrap(),
        &Default::default(),
        &Default::default(),
        &[12.0; 2],
        PhononSupercellOptions {
            repetitions: [usize::MAX, 2, 2],
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("overflow"));
}
