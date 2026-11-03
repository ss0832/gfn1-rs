// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::linalg::Matrix;
use gfn1_rs::soc::*;
use gfn1_rs::*;
use std::collections::BTreeMap;

#[test]
fn real_spherical_angular_momentum_algebra() {
    for (l, angular) in [
        (0, AngularMomentum::S),
        (1, AngularMomentum::P),
        (2, AngularMomentum::D),
    ] {
        let matrices = shell_angular_momentum(angular).unwrap();
        let n = 2 * l + 1;
        for a in &matrices {
            for i in 0..n {
                for j in 0..n {
                    assert!((a.im[(i, j)] + a.im[(j, i)]).abs() < 1e-14);
                }
            }
        }
        let xy = matrices[0].im.matmul(&matrices[1].im).unwrap();
        let yx = matrices[1].im.matmul(&matrices[0].im).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!((xy[(i, j)] - yx[(i, j)] - matrices[2].im[(i, j)]).abs() < 1e-13);
            }
        }
        let mut casimir = Matrix::zeros(n, n);
        for a in &matrices {
            let square = a.im.matmul(&a.im).unwrap();
            for i in 0..n {
                for j in 0..n {
                    casimir[(i, j)] -= square[(i, j)];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { (l * (l + 1)) as f64 } else { 0.0 };
                assert!((casimir[(i, j)] - expected).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn atom_j_splitting_uses_physical_spinors() {
    let params = Gfn1Parameters::builtin().unwrap();
    for (symbol, z, l) in [("F", 9, 1), ("Fe", 26, 2)] {
        let sys = PeriodicSystem::from_xyz_str(&format!("1\natom\n{symbol} 0 0 0\n"), 0.0, false)
            .unwrap();
        let basis = BasisSet::build(&sys, &params, Default::default()).unwrap();
        let mut couplings = BTreeMap::new();
        for sh in &basis.shells {
            couplings.insert(
                (z, sh.angular.as_index()),
                if sh.angular.as_index() == l { 0.2 } else { 0.0 },
            );
        }
        let soc = SocParameters {
            zeta: couplings,
            provenance: "independent synthetic atomic LS fixture; not literature parameters".into(),
        };
        let h = onsite_soc(&basis, &soc).unwrap();
        let eig = spinor_eigen(&h, &Matrix::identity(h.n), 1e-12).unwrap();
        let lower = -0.2 * ((l + 1) as f64) / 2.0;
        let upper = 0.2 * (l as f64) / 2.0;
        assert_eq!(eig.values.len(), 2 * basis.len());
        assert_eq!(
            eig.values
                .iter()
                .filter(|&&e| (e - lower).abs() < 1e-12)
                .count(),
            2 * l
        );
        assert_eq!(
            eig.values
                .iter()
                .filter(|&&e| (e - upper).abs() < 1e-12)
                .count(),
            2 * l + 2
        );
    }
}

#[test]
fn water_zero_soc_limit_nonzero_forces_and_rotation() {
    let params = Gfn1Parameters::builtin().unwrap();
    let sys = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.1 0.0 0\nH 0.9 0.5 0.02\nH -0.7 0.6 -0.03\n",
        0.0,
        false,
    )
    .unwrap();
    let mut opts = ElectronicOptions::default();
    opts.energy_tolerance = 1e-12;
    opts.charge_tolerance = 1e-11;
    opts.electronic_temperature = 0.0;
    let mut soc = SocParameters {
        zeta: BTreeMap::from([((8, 1), 0.0)]),
        provenance: "independent synthetic water test, not fitted SOC parameters".into(),
    };
    let scalar = run_electronic(&sys, &params, opts.clone()).unwrap();
    let zero = run_soc_scc(&sys, &params, &opts, &soc).unwrap();
    assert!((zero.total_free - scalar.total_free).abs() < 1e-11);
    assert!((zero.occupations.iter().sum::<f64>() - 8.0).abs() < 1e-12);
    assert!(zero.occupations.iter().all(|&x| (0.0..=1.0).contains(&x)));
    soc.zeta.insert((8, 1), 0.02);
    let (result, gradient) = soc_analytic_gradient(&sys, &params, &opts, &soc).unwrap();
    let response = soc_static_polarizability(&sys, &params, &result).unwrap();
    assert!(response.max_normalized_residual < 1e-13);
    let configured = gfn1_rs::resources::CpSettings {
        absolute_tolerance: Some(1e-10),
        relative_tolerance: Some(1e-10),
        max_iterations: Some(1), // direct LU: no Krylov iteration
    }
    .run(|| soc_static_polarizability(&sys, &params, &result))
    .unwrap();
    assert_eq!(configured.tensor, response.tensor);
    let strict = gfn1_rs::resources::CpSettings {
        absolute_tolerance: Some(1e-30),
        relative_tolerance: Some(1e-30),
        ..Default::default()
    }
    .run(|| soc_static_polarizability(&sys, &params, &result))
    .err()
    .expect("an unattainable configured CP tolerance must fail");
    assert!(strict
        .to_string()
        .contains("molecular spinor CP-SCC residual"));
    for axis in 0..3 {
        let h = 1e-4;
        let dipole = |sign: f64| {
            let mut o = opts.clone();
            let mut field = [0.0; 3];
            field[axis] = sign * h;
            o.external_field.electric_field =
                Some(gfn1_rs::math::Vec3::new(field[0], field[1], field[2]));
            run_soc_scc(&sys, &params, &o, &soc)
                .unwrap()
                .dipole
                .to_array()
        };
        let plus = dipole(1.0);
        let minus = dipole(-1.0);
        for a in 0..3 {
            assert!(((plus[a] - minus[a]) / (2.0 * h) - response.tensor[a][axis]).abs() < 1e-4);
        }
    }
    for pair in result.eigen.values.chunks_exact(2) {
        assert!((pair[0] - pair[1]).abs() < 1e-10, "Kramers pair");
    }
    for h in [2e-4, 1e-4] {
        let mut plus = sys.clone();
        let mut minus = sys.clone();
        plus.atoms[1].position.x += h;
        minus.atoms[1].position.x -= h;
        let fd = (run_soc_scc(&plus, &params, &opts, &soc).unwrap().total_free
            - run_soc_scc(&minus, &params, &opts, &soc)
                .unwrap()
                .total_free)
            / (2.0 * h);
        assert!(
            (fd - gradient[1].x).abs() < 2e-7,
            "SOC full force analytic {} FD {fd}",
            gradient[1].x
        );
    }
    let mut rotated = sys.clone();
    let (s, c) = 0.37_f64.sin_cos();
    for a in &mut rotated.atoms {
        let r = a.position;
        a.position.x = c * r.x - s * r.y;
        a.position.y = s * r.x + c * r.y;
    }
    assert!(
        (run_soc_scc(&rotated, &params, &opts, &soc)
            .unwrap()
            .total_free
            - result.total_free)
            .abs()
            < 1e-10
    );
}
