// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::third_derivative::CubicReference;
use gfn1_rs::*;

#[test]
fn constrained_variational_cubic_matches_cartesian_tensor_and_hessian_fd() {
    let cases = [
        include_str!("../examples/water.xyz"),
        include_str!("../examples/ammonia.xyz"),
        "4\nCBrOH probe\nC 0 0 0\nBr 1.9 0 0\nO 4.6 0.2 0\nH 5.0 0.8 0\n",
    ];
    for xyz in cases {
        for (atm, order) in [(false, 3), (true, 3), (false, 5), (true, 5)] {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let d = 3 * system.atoms.len();
            let mut params = Gfn1Parameters::builtin().unwrap();
            params
                .globpar
                .insert("s9".into(), if atm { 1.0 } else { 0.0 });
            let mut options = ElectronicOptions::default();
            options.electronic_temperature = 0.0;
            options.energy_tolerance = 1e-12;
            options.charge_tolerance = 1e-11;
            options.charge_order = order;
            let hopt = AnalyticHessianOptions {
                electronic_options: options.clone(),
                ..Default::default()
            };
            let v: Vec<f64> = (0..d).map(|i| 0.15 - 0.07 * (i % 4) as f64).collect();
            let reference =
                CubicReference::build(&system, &params, &options, 512 * 1024 * 1024).unwrap();
            let direct = reference.directional(&v).unwrap();
            let tensor = third_derivative_analytic_dense(
                &system,
                &params,
                hopt.clone(),
                options.hamiltonian.coordination_cutoff,
            )
            .unwrap();
            let legacy = tensor.contract_vvv(&v);
            eprintln!(
                "{} atoms, ATM={atm}, order={order}: direct {direct:e}, legacy {legacy:e}",
                system.atoms.len()
            );
            assert!(
                (direct - legacy).abs() < 1e-8,
                "direct {direct:e} legacy {legacy:e}"
            );
            let twice: Vec<f64> = v.iter().map(|x| 2.0 * x).collect();
            assert!((reference.directional(&twice).unwrap() - 8.0 * direct).abs() < 1e-10);
            for h in [1e-3, 5e-4] {
                let evaluate = |sign: f64| {
                    let mut displaced = system.clone();
                    for (a, atom) in displaced.atoms.iter_mut().enumerate() {
                        atom.position.x += sign * h * v[3 * a];
                        atom.position.y += sign * h * v[3 * a + 1];
                        atom.position.z += sign * h * v[3 * a + 2];
                    }
                    let hessian = analytic_hessian(&displaced, &params, hopt.clone())
                        .unwrap()
                        .hessian;
                    (0..d)
                        .map(|i| (0..d).map(|j| v[i] * hessian[(i, j)] * v[j]).sum::<f64>())
                        .sum::<f64>()
                };
                let fd = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * h);
                assert!(
                    (fd - direct).abs() < 1e-7,
                    "FC3 direct {direct:e}, Hessian FD {fd:e}"
                );
            }
            if xyz.contains("CBrOH") {
                // This branch engages the robust SCC ladder. Independent energy
                // differences exposed a retained level shift in the old Fock.
                let energy = |t: f64| {
                    let mut x = system.clone();
                    for (a, atom) in x.atoms.iter_mut().enumerate() {
                        atom.position.x += t * v[3 * a];
                        atom.position.y += t * v[3 * a + 1];
                        atom.position.z += t * v[3 * a + 2];
                    }
                    let e = run_electronic(&x, &params, options.clone()).unwrap();
                    let physical = gfn1_rs::electronic::fock_from_shell_potential(
                        &e.basis,
                        &e.integrals.overlap,
                        &e.h0,
                        &e.shell_scc_potential,
                    );
                    assert!(
                        physical.max_abs_diff(&e.fock) < 1e-9,
                        "final response Fock must be unshifted"
                    );
                    e.total_free
                };
                for h in [0.01, 0.005] {
                    let fd = (energy(2.0 * h) - 2.0 * energy(h) + 2.0 * energy(-h)
                        - energy(-2.0 * h))
                        / (2.0 * h * h * h);
                    assert!(
                        (fd - direct).abs() < 2e-6,
                        "independent energy FC3 stencil {fd:e} vs {direct:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn cubic_budget_precedes_scf_and_fractional_branch_is_analytic() {
    let params = Gfn1Parameters::builtin().unwrap();
    let sys =
        PeriodicSystem::from_xyz_str(include_str!("../examples/water.xyz"), 0.0, false).unwrap();
    let error = CubicReference::build(&sys, &params, &Default::default(), 1)
        .err()
        .unwrap();
    assert!(error.to_string().contains("budget"));
    let mut options = ElectronicOptions::default();
    options.electronic_temperature = 100_000.0;
    options.max_scc = 1000;
    let reference = CubicReference::build(&sys, &params, &options, 512 * 1024 * 1024).unwrap();
    assert!(reference
        .electronic()
        .occupations
        .iter()
        .any(|&f| f > 1e-5 && f < 2. - 1e-5));
    let direction = vec![0.1; 9];
    assert!(reference.directional(&direction).unwrap().abs() < 1e-9);
}

#[test]
fn canonical_thermal_cubic_trial_matches_response_and_hessian_differences() {
    for xyz in [
        "2\nskew HF\nH .1 -.2 .03\nF 1.0 .12 -.11\n",
        "3\nequilateral H3 exact fractional doublet\nH 0 0 0\nH .8 0 0\nH .4 .6928203230275509 0\n",
        "2\nB2 exact fractional pi blocks\nB 0 0 0\nB 0 0 1.6\n",
    ] {
        let sys = PeriodicSystem::from_xyz_str(xyz, 0., false).unwrap();
        let params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions {
            electronic_temperature: 30000.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            max_scc: 1000,
            ..Default::default()
        };
        let reference = CubicReference::build(&sys, &params, &options, 512 * 1024 * 1024).unwrap();
        assert!(reference
            .electronic()
            .occupations
            .iter()
            .any(|&f| f > 1e-5 && f < 2. - 1e-5));
        let d = 3 * sys.atoms.len();
        let v: Vec<_> = (0..d).map(|i| 0.13 - 0.047 * (i % 5) as f64).collect();
        let values = reference.directional_derivatives(&v).unwrap();
        let hopt = AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..Default::default()
        };
        let hessian = analytic_hessian(&sys, &params, hopt.clone())
            .unwrap()
            .hessian;
        let hh = (0..d)
            .map(|i| (0..d).map(|j| v[i] * hessian[(i, j)] * v[j]).sum::<f64>())
            .sum::<f64>();
        assert!(
            (hh - values[2]).abs() < 1e-9,
            "thermal trial H {0:e} ref {hh:e}",
            values[2]
        );
        let expected = gfn1_rs::third_derivative::finite_t::directional_third_finite_t(
            &sys,
            &params,
            &hopt,
            options.hamiltonian.coordination_cutoff,
            &v,
        )
        .unwrap();
        eprintln!(
            "thermal trial {} atoms cubic {:e}, spectral {:e}",
            sys.atoms.len(),
            values[3],
            expected
        );
        assert!((expected - values[3]).abs() < 1e-8);
        let curvature = |shift: f64| {
            let mut moved = sys.clone();
            for (i, a) in moved.atoms.iter_mut().enumerate() {
                a.position +=
                    gfn1_rs::math::Vec3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]) * shift;
            }
            let h = analytic_hessian(&moved, &params, hopt.clone())
                .unwrap()
                .hessian;
            (0..d)
                .map(|i| (0..d).map(|j| v[i] * h[(i, j)] * v[j]).sum::<f64>())
                .sum::<f64>()
        };
        let coarse = (curvature(1e-3) - curvature(-1e-3)) / 2e-3;
        let fine = (curvature(5e-4) - curvature(-5e-4)) / 1e-3;
        assert!(
            (values[3] - (4. * fine - coarse) / 3.).abs() < 1e-8,
            "thermal cubic {:?}, FD {}",
            values,
            (4. * fine - coarse) / 3.
        );
        if sys.atoms.len() == 3 {
            let fourth = directional_fourth_derivative(
                &sys,
                &params,
                &hopt,
                options.hamiltonian.coordination_cutoff,
                &v,
            )
            .unwrap();
            let cubic = |shift: f64| {
                let mut moved = sys.clone();
                for (i, a) in moved.atoms.iter_mut().enumerate() {
                    a.position +=
                        gfn1_rs::math::Vec3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]) * shift;
                }
                CubicReference::build(&moved, &params, &options, 512 * 1024 * 1024)
                    .unwrap()
                    .directional(&v)
                    .unwrap()
            };
            let coarse = (cubic(1e-3) - cubic(-1e-3)) / 2e-3;
            let fine = (cubic(5e-4) - cubic(-5e-4)) / 1e-3;
            assert!(
                (fourth - (4. * fine - coarse) / 3.).abs() < 1e-8,
                "exact-degenerate canonical FC4 {fourth:e}, FD {:e}",
                (4. * fine - coarse) / 3.
            );
        }
    }
}

#[test]
fn thermal_public_fc3_fc4_routes_match_every_cartesian_component() {
    use gfn1_rs::fourth_derivative::{
        directional_fourth_with_reference, fourth_derivative_analytic_block_with_limits,
        fourth_derivative_analytic_semidiagonal, QuarticAssemblyOptions, QuarticReference,
    };
    let system = PeriodicSystem::from_xyz_str(
        "2\nthermal skew HF\nH .1 -.2 .03\nF 1.0 .12 -.11\n",
        0.,
        false,
    )
    .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let electronic = ElectronicOptions {
        electronic_temperature: 30000.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    };
    let options = AnalyticHessianOptions {
        electronic_options: electronic.clone(),
        ..Default::default()
    };
    let cutoff = electronic.hamiltonian.coordination_cutoff;
    let n = 6;
    let v = vec![0.13, -0.07, 0.21, -0.11, 0.09, 0.03];
    let reference =
        CubicReference::build(&system, &params, &electronic, 512 * 1024 * 1024).unwrap();
    let cubic = third_derivative_analytic_dense(&system, &params, options.clone(), cutoff).unwrap();
    let slabs = third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
    let vector =
        third_derivative_analytic_vector(&system, &params, options.clone(), cutoff, &v).unwrap();
    let (dofs, block) =
        third_derivative_analytic_block(&system, &params, options.clone(), cutoff, &[1, 0])
            .unwrap();
    let mut axes = vec![vec![0.; n]; n];
    for i in 0..n {
        axes[i][i] = 1.;
    }
    let mut cubic_alias_error = 0f64;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                cubic_alias_error = cubic_alias_error.max(
                    (reference.mixed(&axes[a], &axes[b], &axes[c]).unwrap() - cubic.get(a, b, c))
                        .abs(),
                );
                assert!((slabs[c][(a, b)] - cubic.get(a, b, c)).abs() < 1e-10);
                assert!((block[c][(a, b)] - cubic.get(dofs[a], dofs[b], dofs[c])).abs() < 1e-10);
            }
        }
    }
    assert!(
        cubic_alias_error < 2e-9,
        "cubic canonical variational vs spectral {cubic_alias_error:e}"
    );
    assert!(vector.max_abs_diff(&cubic.contract_last(&v)) < 1e-10);
    let quartic = fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap();
    let qref = QuarticReference::build(&system, &params, &options, cutoff).unwrap();
    assert_eq!(qref.electronic().electronic_temperature, 30000.);
    let directional =
        directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap();
    assert!((directional - quartic.contract_vvvv(&v).unwrap()).abs() < 1e-10);
    assert!(
        (directional
            - directional_fourth_with_reference(&system, &params, &options, cutoff, &qref, &v)
                .unwrap())
        .abs()
            < 1e-12
    );
    let selection = [5, 1, 3];
    let qblock =
        fourth_derivative_analytic_block(&system, &params, &options, cutoff, &selection).unwrap();
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    assert!(
                        (qblock.get(a, b, c, d)
                            - quartic.get(selection[a], selection[b], selection[c], selection[d]))
                        .abs()
                            < 1e-10
                    );
                }
            }
        }
    }
    let modepairs = fourth_derivative_analytic_semidiagonal(
        &system,
        &params,
        &options,
        cutoff,
        &[v.clone(), axes[1].clone()],
    )
    .unwrap();
    assert!((modepairs[(0, 0)] - directional).abs() < 1e-10);
    let limits = QuarticAssemblyOptions {
        memory_budget_bytes: 1,
        ..Default::default()
    };
    assert!(fourth_derivative_analytic_block_with_limits(
        &system, &params, &options, cutoff, &selection, &limits
    )
    .unwrap_err()
    .to_string()
    .contains("budget"));
    let shifted = |coordinate: usize, step: f64| {
        let mut out = system.clone();
        out.atoms[coordinate / 3].position += gfn1_rs::math::Vec3::new(
            if coordinate % 3 == 0 { step } else { 0. },
            if coordinate % 3 == 1 { step } else { 0. },
            if coordinate % 3 == 2 { step } else { 0. },
        );
        out
    };
    let mut cubic_error = 0f64;
    let mut quartic_error = 0f64;
    for d in 0..n {
        let mut hd = Vec::new();
        let mut td = Vec::new();
        for h in [1e-3, 5e-4] {
            let plus = shifted(d, h);
            let minus = shifted(d, -h);
            let hp = analytic_hessian(&plus, &params, options.clone())
                .unwrap()
                .hessian;
            let hm = analytic_hessian(&minus, &params, options.clone())
                .unwrap()
                .hessian;
            let tp =
                third_derivative_analytic_dense(&plus, &params, options.clone(), cutoff).unwrap();
            let tm =
                third_derivative_analytic_dense(&minus, &params, options.clone(), cutoff).unwrap();
            hd.push((hp, hm, h));
            td.push((tp, tm, h));
        }
        for a in 0..n {
            for b in 0..n {
                let hcoarse = (hd[0].0[(a, b)] - hd[0].1[(a, b)]) / (2. * hd[0].2);
                let hfine = (hd[1].0[(a, b)] - hd[1].1[(a, b)]) / (2. * hd[1].2);
                cubic_error =
                    cubic_error.max((cubic.get(a, b, d) - (4. * hfine - hcoarse) / 3.).abs());
                for c in 0..n {
                    let coarse = (td[0].0.get(a, b, c) - td[0].1.get(a, b, c)) / (2. * td[0].2);
                    let fine = (td[1].0.get(a, b, c) - td[1].1.get(a, b, c)) / (2. * td[1].2);
                    quartic_error = quartic_error
                        .max((quartic.get(a, b, c, d) - (4. * fine - coarse) / 3.).abs());
                }
            }
        }
    }
    eprintln!("thermal public routes: all216 cubic FD {cubic_error:e}, all1296 quartic FD {quartic_error:e}, canonical trial {cubic_alias_error:e}");
    assert!(cubic_error < 1e-8);
    assert!(quartic_error < 1e-7);
}

#[test]
fn structural_ml_fractional_cubic_matches_reconverged_hessian() {
    use std::sync::Arc;
    let system =
        PeriodicSystem::from_xyz_str("2\nthermal ML H2\nH .1 .02 0\nH .95 .13 -.07\n", 0., false)
            .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut model = gfn1_rs::ml::MlHamiltonian::zero(vec![1], vec![1.5, 3.0], 7.);
    model.heads.get_mut(&1).unwrap().levels[0].weights[0] = 0.003;
    model.heads.get_mut(&1).unwrap().correction.weights[0] = 0.002;
    let electronic = ElectronicOptions {
        ml_model: Some(Arc::new(model)),
        electronic_temperature: 30000.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    };
    let options = AnalyticHessianOptions {
        electronic_options: electronic.clone(),
        ..Default::default()
    };
    let reference =
        CubicReference::build(&system, &params, &electronic, 512 * 1024 * 1024).unwrap();
    assert!(reference
        .electronic()
        .occupations
        .iter()
        .any(|&f| f > 1e-5 && f < 2. - 1e-5));
    let v = [0.13, -0.07, 0.21, -0.11, 0.09, 0.03];
    let value = reference.directional(&v).unwrap();
    let explicit = gfn1_rs::third_derivative::finite_t::directional_third_finite_t(
        &system,
        &params,
        &options,
        electronic.hamiltonian.coordination_cutoff,
        &v,
    )
    .unwrap();
    assert!((explicit - value).abs() < 1e-12);
    let standard = third_derivative_analytic_dense(
        &system,
        &params,
        options.clone(),
        electronic.hamiltonian.coordination_cutoff,
    )
    .unwrap();
    let explicit = gfn1_rs::third_derivative::finite_t::third_derivative_finite_t_dense(
        &system,
        &params,
        &options,
        electronic.hamiltonian.coordination_cutoff,
    )
    .unwrap();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                assert!((standard.get(i, j, k) - explicit.get(i, j, k)).abs() < 1e-10);
            }
        }
    }
    let curvature = |step: f64| {
        let mut moved = system.clone();
        for (i, a) in moved.atoms.iter_mut().enumerate() {
            a.position += gfn1_rs::math::Vec3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]) * step;
        }
        let h = analytic_hessian(&moved, &params, options.clone())
            .unwrap()
            .hessian;
        (0..6)
            .map(|i| (0..6).map(|j| v[i] * h[(i, j)] * v[j]).sum::<f64>())
            .sum::<f64>()
    };
    let coarse = (curvature(1e-3) - curvature(-1e-3)) / 2e-3;
    let fine = (curvature(5e-4) - curvature(-5e-4)) / 1e-3;
    assert!((value - (4. * fine - coarse) / 3.).abs() < 1e-8);
}


#[test]
fn symmetric_water_cubic_and_quartic_match_all_numerical_components() {
    let system = PeriodicSystem::from_xyz_str(include_str!("../examples/water.xyz"), 0., false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = AnalyticHessianOptions {
        electronic_options: ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    };
    let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
    let cubic = third_derivative_analytic_dense(&system, &params, options.clone(), cutoff).unwrap();
    let quartic = fourth_derivative_analytic_dense(&system, &params, &options, cutoff).unwrap();
    let n = 3 * system.atoms.len();
    let mut max_cubic = 0_f64;
    let mut max_quartic = 0_f64;
    for d in 0..n {
        let mut differences = Vec::new();
        for step in [1e-3, 5e-4] {
            let evaluate = |sign: f64| {
                let mut displaced = system.clone();
                let atom = &mut displaced.atoms[d/3];
                match d % 3 {
                    0 => atom.position.x += sign * step,
                    1 => atom.position.y += sign * step,
                    _ => atom.position.z += sign * step,
                }
                let h = analytic_hessian(&displaced, &params, options.clone()).unwrap().hessian;
                let c = third_derivative_analytic_dense(&displaced, &params, options.clone(), cutoff).unwrap();
                (h,c)
            };
            differences.push((evaluate(1.), evaluate(-1.), 0.5 / step));
        }
        for a in 0..n {
            for b in 0..n {
                let hdiff = |i: usize| {
                    let (p,m,scale) = &differences[i];
                    scale * (p.0[(a,b)] - m.0[(a,b)])
                };
                max_cubic = max_cubic.max((cubic.get(a,b,d) - (4.*hdiff(1) - hdiff(0))/3.).abs());
                for c in 0..n {
                    let cdiff = |i: usize| {
                        let (p,m,scale) = &differences[i];
                        scale * (p.1.get(a,b,c) - m.1.get(a,b,c))
                    };
                    max_quartic = max_quartic.max((quartic.get(a,b,c,d) - (4.*cdiff(1) - cdiff(0))/3.).abs());
                }
            }
        }
    }
    eprintln!("symmetric gas water at 300 K: all {} FC3 max {max_cubic:e}, all {} FC4 max {max_quartic:e}", n.pow(3), n.pow(4));
    assert!(max_cubic < 1e-7 && max_quartic < 2e-7);
}


#[test]
#[ignore = "full 19,683-component Ni(CO)4 FC3 validation"]
fn nickel_carbonyl_cubic_matches_all_numerical_components() {
    let system = PeriodicSystem::from_xyz_str(include_str!("../examples/ni_co4.xyz"), 0., false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let reference_options = AnalyticHessianOptions::default();
    let cutoff = reference_options.electronic_options.hamiltonian.coordination_cutoff;
    let cubic = third_derivative_analytic_dense(&system, &params, reference_options, cutoff).unwrap();
    let options = AnalyticHessianOptions {
        electronic_options: ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-12,
            charge_tolerance: 1e-11,
            max_scc: 1500,
            ..Default::default()
        },
        ..Default::default()
    };
    let n = 3 * system.atoms.len();
    let mut maximum = 0_f64;
    for d in 0..n {
        let hessian = |shift: f64| {
            let mut displaced = system.clone();
            let p = &mut displaced.atoms[d / 3].position;
            match d % 3 {
                0 => p.x += shift,
                1 => p.y += shift,
                _ => p.z += shift,
            }
            analytic_hessian(&displaced, &params, options.clone()).unwrap().hessian
        };
        let coarse_p = hessian(2e-4);
        let coarse_m = hessian(-2e-4);
        let fine_p = hessian(1e-4);
        let fine_m = hessian(-1e-4);
        for a in 0..n {
            for b in 0..n {
                let coarse = (coarse_p[(a,b)] - coarse_m[(a,b)]) / 4e-4;
                let fine = (fine_p[(a,b)] - fine_m[(a,b)]) / 2e-4;
                maximum = maximum.max((cubic.get(a,b,d) - (4.*fine - coarse)/3.).abs());
            }
        }
    }
    eprintln!("Ni(CO)4 300 K gas FC3, all {} entries maximum FD difference {maximum:e}", n.pow(3));
    assert!(maximum < 1e-6);
}
