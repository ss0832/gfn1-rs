// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::lattice::Lattice;
use gfn1_rs::math::{Mat3, Vec3};
use gfn1_rs::pbc::complex::{native_generalized_eigen, CMatrix};
use gfn1_rs::pbc::*;
use gfn1_rs::soc::SocParameters;
use gfn1_rs::*;
use std::collections::BTreeMap;

fn water() -> PeriodicSystem {
    PeriodicSystem::from_xyz_str("3\nLattice=\"3.8 0 0 .2 4 0 0 .1 4.2\" pbc=\"T T T\"\nO 0 0 0\nH .958 0 0\nH -.25 .925 .03\n", 0., false).unwrap()
}
fn options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-11,
        ..Default::default()
    }
}
fn parameters(value: f64) -> SocParameters {
    SocParameters {
        zeta: BTreeMap::from([((8, 1), value)]),
        provenance:
            "Independent synthetic zeta L.S derivative test; not a literature atomic constant"
                .into(),
    }
}
fn pbc(mesh: KMesh) -> PbcOptions {
    PbcOptions {
        kmesh: mesh,
        ao_cutoff: 20.,
        ..Default::default()
    }
}
fn strain(system: &PeriodicSystem, a: usize, b: usize, h: f64) -> PeriodicSystem {
    let change = |r: Vec3| {
        let mut v = r.to_array();
        v[a] += h * v[b];
        Vec3::new(v[0], v[1], v[2])
    };
    let mut out = system.clone();
    let l = system.lattice.unwrap();
    out.lattice = Some(
        Lattice::new(
            Mat3::from_columns(
                change(l.cell.col[0]),
                change(l.cell.col[1]),
                change(l.cell.col[2]),
            ),
            l.periodic,
        )
        .unwrap(),
    );
    for atom in &mut out.atoms {
        atom.position = change(atom.position);
    }
    out
}
#[test]
fn native_complex_generalized_solver_preserves_physical_dimension_and_metric() {
    let mut h = CMatrix::zeros(2);
    h.re[(0, 0)] = -0.4;
    h.re[(1, 1)] = 0.3;
    h.re[(0, 1)] = 0.2;
    h.re[(1, 0)] = 0.2;
    h.im[(0, 1)] = 0.1;
    h.im[(1, 0)] = -0.1;
    let mut s = CMatrix::zeros(2);
    s.re[(0, 0)] = 1.;
    s.re[(1, 1)] = 1.;
    s.im[(0, 1)] = 0.15;
    s.im[(1, 0)] = -0.15;
    let result = native_generalized_eigen(&h, &s, 1e-12).unwrap();
    assert_eq!(result.values.len(), 2);
    let metric = result
        .vectors
        .adjoint()
        .matmul(&s.matmul(&result.vectors).unwrap())
        .unwrap();
    let hc = h.matmul(&result.vectors).unwrap();
    let sc = s.matmul(&result.vectors).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!((metric.re[(i, j)] - if i == j { 1. } else { 0. }).abs() < 1e-12);
            assert!(metric.im[(i, j)].abs() < 1e-12);
            assert!((hc.re[(i, j)] - sc.re[(i, j)] * result.values[j]).abs() < 1e-12);
            assert!((hc.im[(i, j)] - sc.im[(i, j)] * result.values[j]).abs() < 1e-12);
        }
    }
    s.re[(0, 0)] = -1.;
    assert!(native_generalized_eigen(&h, &s, 1e-12).is_err());
}
#[test]
fn periodic_zero_soc_recovers_scalar_energy_force_stress_and_physical_state_count() {
    let system = water();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = options();
    for mesh in [KMesh::gamma(), KMesh::monkhorst_pack([2, 1, 1])] {
        let pbc = pbc(mesh);
        let plain = pbc_analytic_gradient(&system, &params, &options, &pbc).unwrap();
        let (soc, gradient) =
            pbc_soc_analytic_gradient(&system, &params, &options, &pbc, &parameters(0.)).unwrap();
        assert!((plain.total_energy - soc.total_free).abs() < 2e-11);
        for (g, r) in gradient.iter().zip(&plain.gradient) {
            assert!((*g - *r).norm() < 2e-9);
        }
        let charge: f64 = soc
            .kpoints
            .iter()
            .zip(&soc.occupations_k)
            .map(|(k, o)| k.weight * o.iter().sum::<f64>())
            .sum();
        assert!((charge - plain.scf.nelec).abs() < 1e-10);
        assert!(soc
            .occupations_k
            .iter()
            .flatten()
            .all(|&o| (0. ..=1.).contains(&o)));
        for eigen in &soc.eigen_k {
            assert_eq!(eigen.values.len(), 2 * plain.scf.basis.len());
        }
        let plain_stress = pbc_stress(&system, &params, &options, &pbc).unwrap();
        let (_, stress) =
            pbc_soc_stress(&system, &params, &options, &pbc, &parameters(0.)).unwrap();
        for (a, b) in stress.as_slice().iter().zip(plain_stress.stress.as_slice()) {
            assert!((a - b).abs() < 2e-10);
        }
    }
}
#[test]
fn periodic_nonzero_soc_force_stress_and_time_reversal_are_variational() {
    let system = water();
    let params = Gfn1Parameters::builtin().unwrap();
    let coupling = parameters(0.005);
    let mut options = options();
    for (mesh, temperature) in [
        (KMesh::gamma(), 0.),
        (KMesh::monkhorst_pack([2, 1, 1]), 0.),
        (KMesh::monkhorst_pack([2, 1, 1]), 10000.),
    ] {
        options.electronic_temperature = temperature;
        let pbc = pbc(mesh);
        let (result, gradient) =
            pbc_soc_analytic_gradient(&system, &params, &options, &pbc, &coupling).unwrap();
        assert!(result.soc_energy.abs() > 1e-8);
        if mesh.is_gamma_only() {
            for values in result.eigen_k[0].values.chunks_exact(2) {
                assert!((values[0] - values[1]).abs() < 1e-10);
            }
        }
        for h in [2e-4, 1e-4] {
            let mut plus = system.clone();
            let mut minus = system.clone();
            plus.atoms[1].position.y += h;
            minus.atoms[1].position.y -= h;
            let fd = (run_pbc_soc_scc(&plus, &params, &options, &pbc, &coupling)
                .unwrap()
                .total_free
                - run_pbc_soc_scc(&minus, &params, &options, &pbc, &coupling)
                    .unwrap()
                    .total_free)
                / (2. * h);
            assert!(
                (fd - gradient[1].y).abs() < 3e-7,
                "SOC force {}, FD {fd}",
                gradient[1].y
            );
        }
        let (_, stress) = pbc_soc_stress(&system, &params, &options, &pbc, &coupling).unwrap();
        for (a, b) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)] {
            let h = 1e-5;
            let fd = (run_pbc_soc_scc(
                &strain(&system, a, b, h),
                &params,
                &options,
                &pbc,
                &coupling,
            )
            .unwrap()
            .total_free
                - run_pbc_soc_scc(
                    &strain(&system, a, b, -h),
                    &params,
                    &options,
                    &pbc,
                    &coupling,
                )
                .unwrap()
                .total_free)
                / (2. * h * system.lattice.unwrap().volume());
            assert!(
                (fd - stress[(a, b)]).abs() < 3e-8,
                "SOC stress {a}{b}: {}, FD {fd}",
                stress[(a, b)]
            );
        }
        if !mesh.is_gamma_only() {
            let mut full = pbc;
            full.kmesh.fold_time_reversal = false;
            let unfolded = run_pbc_soc_scc(&system, &params, &options, &full, &coupling).unwrap();
            assert!((unfolded.total_free - result.total_free).abs() < 1e-10);
            for (a, b) in unfolded.atomic_charges.iter().zip(&result.atomic_charges) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }
}
#[test]
fn periodic_soc_energy_is_invariant_under_joint_cell_molecule_rotation() {
    let system = water();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = options();
    let pbc = pbc(KMesh::monkhorst_pack([2, 1, 1]));
    let original = run_pbc_soc_scc(&system, &params, &options, &pbc, &parameters(0.005)).unwrap();
    let rotate = |r: Vec3| Vec3::new(-r.y, r.x, r.z);
    let mut rotated = system.clone();
    let l = system.lattice.unwrap();
    rotated.lattice = Some(
        Lattice::new(
            Mat3::from_columns(
                rotate(l.cell.col[0]),
                rotate(l.cell.col[1]),
                rotate(l.cell.col[2]),
            ),
            l.periodic,
        )
        .unwrap(),
    );
    for atom in &mut rotated.atoms {
        atom.position = rotate(atom.position);
    }
    let changed = run_pbc_soc_scc(&rotated, &params, &options, &pbc, &parameters(0.005)).unwrap();
    assert!((original.total_free - changed.total_free).abs() < 1e-10);
}

#[test]
fn physical_spinor_dfpt_hessian_matches_force_fd_and_zero_soc_scalar_response() {
    let system = water();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut options = options();
    for (mesh, temperature) in [
        (KMesh::gamma(), 0.),
        (KMesh::monkhorst_pack([2, 1, 1]), 0.),
        (KMesh::monkhorst_pack([2, 1, 1]), 10000.),
    ] {
        let pbc = pbc(mesh);
        options.electronic_temperature = temperature;
        let actual = pbc_soc_hessian(&system, &params, &options, &pbc, &parameters(0.005)).unwrap();
        assert!(actual.max_response_residual < 1e-9);
        assert!(
            actual.raw_max_asymmetry < 2e-7,
            "raw spinor Hessian asymmetry {}",
            actual.raw_max_asymmetry
        );
        for h in [2e-4, 1e-4] {
            let mut error = 0_f64;
            for y in 0..9 {
                let vector = match y % 3 {
                    0 => Vec3::new(h, 0., 0.),
                    1 => Vec3::new(0., h, 0.),
                    _ => Vec3::new(0., 0., h),
                };
                let mut plus = system.clone();
                let mut minus = system.clone();
                plus.atoms[y / 3].position += vector;
                minus.atoms[y / 3].position -= vector;
                let gp =
                    pbc_soc_analytic_gradient(&plus, &params, &options, &pbc, &parameters(0.005))
                        .unwrap()
                        .1;
                let gm =
                    pbc_soc_analytic_gradient(&minus, &params, &options, &pbc, &parameters(0.005))
                        .unwrap()
                        .1;
                for a in 0..3 {
                    let fd = ((gp[a] - gm[a]) / (2. * h)).to_array();
                    for k in 0..3 {
                        error = error.max((fd[k] - actual.hessian[(3 * a + k, y)]).abs());
                    }
                }
            }
            assert!(
                error < 3e-6,
                "physical spinor Hessian vs force FD {error:e}, k={:?}, T={temperature}",
                mesh.size
            );
        }
        for row in 0..9 {
            for k in 0..3 {
                assert!(
                    (0..3)
                        .map(|a| actual.hessian[(row, 3 * a + k)])
                        .sum::<f64>()
                        .abs()
                        < 2e-7
                );
            }
        }
        let zero = pbc_soc_hessian(&system, &params, &options, &pbc, &parameters(0.)).unwrap();
        let scalar = if mesh.is_gamma_only() {
            pbc_gamma_hessian(&system, &params, &options, &pbc).unwrap()
        } else {
            pbc_kpoint_hessian(&system, &params, &options, &pbc).unwrap()
        };
        let difference = zero
            .hessian
            .as_slice()
            .iter()
            .zip(scalar.hessian.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(
            difference < 3e-8,
            "zero-SOC scalar CP-SCC limit {difference:e}"
        );
    }
}

#[test]
fn finite_q_soc_supercell_phonons_match_their_analytic_force_wave() {
    use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
    let system = water();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = options();
    let pbc = pbc(KMesh::gamma());
    let soc = parameters(0.005);
    let reference = AnalyticSupercellPhonons::build_soc(
        &system,
        &params,
        &options,
        &pbc,
        &[16., 1., 1.],
        PhononSupercellOptions {
            repetitions: [3, 1, 1],
            ..Default::default()
        },
        &soc,
    )
    .unwrap();
    assert!(reference.max_spinor_response_residual.unwrap() < 1e-9);
    assert!(reference.raw_hessian_asymmetry < 2e-7);
    let positive = reference.at_q([1. / 3., 0., 0.]).unwrap();
    let negative = reference.at_q([-1. / 3., 0., 0.]).unwrap();
    assert!(positive.raw_hermiticity_error < 1e-11);
    assert!(positive.method_citations.contains(&"Jha2023SOC"));
    for i in 0..9 {
        for j in 0..9 {
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
    let mut direction = vec![Vec3::zero(); 9];
    for cell in 0..3 {
        direction[3 * cell + 1].y =
            (2. / 3_f64).sqrt() * (2. * std::f64::consts::PI * cell as f64 / 3.).cos();
    }
    for h in [2e-4, 1e-4] {
        let force = |sign| {
            let mut changed = reference.supercell.clone();
            for (atom, v) in changed.atoms.iter_mut().zip(&direction) {
                atom.position += *v * (sign * h);
            }
            pbc_soc_analytic_gradient(&changed, &params, &options, &pbc, &soc)
                .unwrap()
                .1
                .iter()
                .zip(&direction)
                .map(|(g, v)| g.dot(*v))
                .sum::<f64>()
        };
        let fd = (force(1.) - force(-1.)) / (2. * h);
        assert!((fd - positive.dynamical_matrix.re[(4, 4)]).abs() < 3e-6);
    }
}
