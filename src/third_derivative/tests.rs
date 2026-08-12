    // Physics-notation identifiers (nonEq, dV_err, err_pP, ...) are deliberate in
    // this diagnostic-heavy test module.
    #![allow(non_snake_case)]

    use super::*;

    // SymmetricThird stores n(n+1)(n+2)/6 canonical entries; permutations of an index triple
    // accumulate into the same slot, distinct sorted triples map to distinct slots, and the
    // index is a bijection onto 0..len (so no entry is lost or aliased).
    #[test]
    fn symmetric_third_packing_and_symmetry() {
        let n = 5;
        let mut s = SymmetricThird::zeros(n);
        assert_eq!(s.len(), n * (n + 1) * (n + 2) / 6); // 35

        // Permuted indices hit the same slot.
        s.add(1, 3, 2, 4.0);
        s.add(2, 1, 3, 1.0);
        assert!((s.get(3, 2, 1) - 5.0).abs() < 1.0e-15);
        assert!((s.get(2, 3, 1) - 5.0).abs() < 1.0e-15);

        // A distinct triple is independent.
        s.add(0, 0, 0, 7.0);
        assert!((s.get(0, 0, 0) - 7.0).abs() < 1.0e-15);
        assert!((s.get(1, 2, 3) - 5.0).abs() < 1.0e-15);

        // The canonical index is a bijection onto 0..len.
        let mut seen = vec![false; s.len()];
        for c in 0..n {
            for b in 0..=c {
                for a in 0..=b {
                    let idx = SymmetricThird::index(a, b, c);
                    assert!(idx < seen.len(), "index {idx} out of range");
                    assert!(!seen[idx], "collision at ({a},{b},{c})");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&x| x), "index not surjective");
    }

    // The symmetric-packed central block must reproduce the dense block exactly, at 6x less
    // memory -- the symmetry cost/memory reduction is lossless.
    #[test]
    fn add_radial_third_block_sym_matches_dense() {
        let ndof = 6;
        let rel = Vec3::new(1.6, 0.7, 0.2);
        let (g, f3, scale) = (0.3_f64, -1.1_f64, 0.5_f64);
        let mut dense = vec![Matrix::zeros(ndof, ndof); ndof];
        add_radial_third_block(&mut dense, 0, 1, rel, g, f3, scale);
        let mut sym = SymmetricThird::zeros(ndof);
        add_radial_third_block_sym(&mut sym, 0, 1, rel, g, f3, scale);
        for a in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    assert!(
                        (sym.get(a, b, c) - dense[c][(a, b)]).abs() < 1.0e-12,
                        "({a},{b},{c}): sym {} vs dense {}",
                        sym.get(a, b, c),
                        dense[c][(a, b)]
                    );
                }
            }
        }
    }

    // The geometric driver (repulsion + halogen, in the symmetric store) FD-validates as a
    // bundle against the sum of their analytic Hessians (both response-free).
    #[test]
    fn third_derivative_geometric_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let store = third_derivative_geometric(&system, &params).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let mut max_delta = 0.0_f64;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += hal[(r, c)];
                }
            }
            m
        };
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((store.get(row, col, slab) - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "geometric third-derivative bundle FD max delta {max_delta:.3e}"
        );
    }

    // The full GEOMETRIC frozen bundle (repulsion + halogen + D3 dispersion) depends only on
    // interatomic distances/angles, so -- carrying no electronic response and no fixed-density
    // artefact -- its third derivative obeys the acoustic sum rule `Σ_A T_{Aα,bc} = 0` exactly.
    // (The electronic frozen blocks do NOT: with a held-fixed density a rigid shift is not a
    // symmetry, so only the full stationary bundle -- incl. response -- satisfies the rule there.)
    #[test]
    fn geometric_bundle_acoustic_sum_rule() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let mut store = third_derivative_geometric(&system, &params).unwrap();
        let disp = third_derivative_dispersion(&system, &params, None).unwrap();
        store.add_from(&disp);
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let mut max = 0.0_f64;
        for alpha in 0..3 {
            for b in 0..ndof {
                for c in 0..ndof {
                    let sum: f64 = (0..nat).map(|atom| store.get(3 * atom + alpha, b, c)).sum();
                    max = max.max(sum.abs());
                }
            }
        }
        assert!(
            max < 1.0e-7,
            "geometric bundle acoustic sum rule violated: max {max:.3e}"
        );
    }

    // The packed dispersion store reproduces the dense Jet3 tensor under EVERY index permutation
    // simultaneously validating (a) the dense third derivative is fully permutation-symmetric (the
    // physical requirement on `∂³E/∂R³`) and (b) the canonical `i≤j≤k` packing reads the right
    // entries. This is the dispersion analog of the plan's permutation-residual bookkeeping check.
    #[test]
    fn third_derivative_dispersion_packing_is_permutation_symmetric() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let dense = crate::dispersion::dispersion_third_derivative(&system, &params, None).unwrap();
        let store = third_derivative_dispersion(&system, &params, None).unwrap();
        let n = dense.ndof;
        let mut max_delta = 0.0_f64;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let d = dense.third[(a * n + b) * n + c];
                    max_delta = max_delta.max((store.get(a, b, c) - d).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-12,
            "dispersion packed/dense permutation mismatch {max_delta:.3e}"
        );
    }

    // The frozen-electronic driver (SCC2 + Pulay, in the symmetric store) FD-validates as a
    // bundle against the sum of their frozen analytic Hessians (charges/density held fixed).
    #[test]
    fn third_derivative_frozen_electronic_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let electronic = crate::electronic::run_electronic(&system, &params, options).unwrap();
        let store = third_derivative_frozen_electronic(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::hessian::fixed_shell_charge_scc_hessian(
                sys,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let pulay = crate::hessian::fixed_density_pulay_hessian(sys, &params, &electronic)
                .unwrap()
                .hessian;
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += pulay[(r, c)];
                }
            }
            m
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((store.get(row, col, slab) - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "frozen-electronic third-derivative bundle FD max delta {max_delta:.3e}"
        );
    }

    // The complete frozen bundle (geometric + frozen-electronic, merged in the symmetric
    // store) FD-validates against the sum of all four frozen Hessian blocks.
    #[test]
    fn third_derivative_frozen_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let electronic = crate::electronic::run_electronic(&system, &params, options).unwrap();
        let store = third_derivative_frozen(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let scc = crate::hessian::fixed_shell_charge_scc_hessian(
                sys,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let pulay = crate::hessian::fixed_density_pulay_hessian(sys, &params, &electronic)
                .unwrap()
                .hessian;
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += hal[(r, c)] + scc[(r, c)] + pulay[(r, c)];
                }
            }
            m
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((store.get(row, col, slab) - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "frozen bundle third-derivative FD max delta {max_delta:.3e}"
        );
    }

    // The FULL frozen bundle (geometric + frozen-electronic + D3 dispersion) FD-validates against
    // the sum of all five frozen Hessian blocks -- confirming the dispersion third derivative
    // composes correctly (sign/packing) into the assembled symmetric store alongside the others.
    #[test]
    fn third_derivative_frozen_full_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let electronic = crate::electronic::run_electronic(&system, &params, options).unwrap();
        let store = third_derivative_frozen_full(&system, &params, &electronic, None).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let scc = crate::hessian::fixed_shell_charge_scc_hessian(
                sys,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let pulay = crate::hessian::fixed_density_pulay_hessian(sys, &params, &electronic)
                .unwrap()
                .hessian;
            let disp = crate::dispersion::dispersion_energy_gradient_hessian(sys, &params, None)
                .unwrap()
                .hessian;
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += hal[(r, c)] + scc[(r, c)] + pulay[(r, c)] + disp[(r, c)];
                }
            }
            m
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((store.get(row, col, slab) - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "full frozen bundle (with dispersion) third-derivative FD max delta {max_delta:.3e}"
        );
    }

    // The COMPLETE frozen bundle (incl. CN-H0, dense-summed then packed) FD-validates against the
    // sum of ALL frozen Hessian blocks -- repulsion + halogen + SCC2 + Pulay + D3 + CN-H0 + cross.
    // This is the entire frozen `L_abc` part of the 2n+1 third derivative.
    #[test]
    fn third_derivative_frozen_complete_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = crate::electronic::run_electronic(&system, &params, options).unwrap();
        let store =
            third_derivative_frozen_complete(&system, &params, &electronic, None, cutoff, true).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let scc = crate::hessian::fixed_shell_charge_scc_hessian(
                sys,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let pulay = crate::hessian::fixed_density_pulay_hessian(sys, &params, &electronic)
                .unwrap()
                .hessian;
            let disp = crate::dispersion::dispersion_energy_gradient_hessian(sys, &params, None)
                .unwrap()
                .hessian;
            let cnh0 =
                crate::hessian::fixed_density_cn_h0_hessian(sys, &params, &electronic, cutoff)
                    .unwrap()
                    .hessian;
            let cnh0x = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
                sys,
                &params,
                &electronic,
                cutoff,
            )
            .unwrap();
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += hal[(r, c)]
                        + scc[(r, c)]
                        + pulay[(r, c)]
                        + disp[(r, c)]
                        + cnh0[(r, c)]
                        + cnh0x[(r, c)];
                }
            }
            m
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((store[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "complete frozen bundle (with CN-H0) third-derivative FD max delta {max_delta:.3e}"
        );
    }

    // B0 keystone-input check: the CPHF/CPXTB first-order response (the per-DOF density,
    // shell-charge, and energy-weighted-density responses) -- which the 2n+1 response cross-terms
    // (L_abx/L_axx/L_xxx) will contract -- is available, converged, and correctly shaped (one
    // response per nuclear DOF, density n×n, shell charges per shell).
    #[test]
    fn cphf_first_order_responses_available_for_2n1() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let electronic =
            crate::electronic::run_electronic(&system, &params, options.clone()).unwrap();
        let response = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            crate::cphf::AoDerivativeOptions {
                coordination_cutoff: options.hamiltonian.coordination_cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        assert!(response.converged, "CPHF response did not converge");
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let nsh = electronic.basis.shells.len();
        assert_eq!(response.density_responses.len(), ndof);
        assert_eq!(response.energy_weighted_density_responses.len(), ndof);
        assert_eq!(response.shell_charge_responses.len(), ndof);
        for dr in &response.density_responses {
            assert_eq!((dr.rows(), dr.cols()), (n, n));
        }
        for sc in &response.shell_charge_responses {
            assert_eq!(sc.len(), nsh);
        }
    }

    // The analytic 2n+1 third derivative vs the central FD of the FULL analytic Hessian.
    // Since the v0.5.0 fixes (∂K/∂q kernel chain + degenerate-orbital Λ covariance) the
    // analytic path is production-accurate: it matches the FD reference to the FD noise
    // floor (~1e-7 at h=1e-4). The frozen baseline is also measured to document how much
    // the response terms contribute.
    #[test]
    fn third_derivative_analytic_improves_substantially_over_frozen() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        // Disable dispersion to isolate the electronic response (D3 is a separately-validated,
        // response-free geometric block). Tight SCF: the FD-of-Hessian reference amplifies SCF
        // noise by 1/(2h), so default tolerances would put the noise floor at ~1e-4.
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: crate::electronic::ElectronicOptions {
                enable_dispersion: false,
                energy_tolerance: 1.0e-11,
                charge_tolerance: 1.0e-9,
                ..crate::electronic::ElectronicOptions::default()
            },
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let electronic =
            crate::electronic::run_electronic(&system, &params, options.electronic_options.clone())
                .unwrap();
        let store = third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
        // Dispersion-free frozen baseline to match this test's dispersion-off FD reference.
        let frozen =
            third_derivative_frozen_complete(&system, &params, &electronic, None, cutoff, false).unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-4;
        let full_hess = |sys: &PeriodicSystem| -> Matrix {
            crate::hessian::analytic_hessian(sys, &params, options.clone())
                .unwrap()
                .hessian
        };
        let mut err_analytic = 0.0_f64;
        let mut err_frozen = 0.0_f64;
        let mut max_ref = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let (atom, ax) = (slab / 3, slab % 3);
            match ax {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let hp = full_hess(&plus);
            let hm = full_hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_ref = max_ref.max(fd.abs());
                    err_analytic = err_analytic.max((store[slab][(row, col)] - fd).abs());
                    err_frozen = err_frozen.max((frozen[slab][(row, col)] - fd).abs());
                }
            }
        }
        eprintln!(
            "STRICT closed-form 2n+1 third derivative: err_analytic={err_analytic:.3e} \
             err_frozen={err_frozen:.3e} (max |T| ~= {max_ref:.3e})"
        );
        // STRICT CLOSED FORM (no finite differences anywhere): D_c H_frozen (L_abc + scalar_overlap + L_abx)
        // + D_c(hessian_response) via the analytic Z-vector assembly (closed_form_response_hessian_derivative).
        // Since the v0.5.0 fixes this matches FD(full Hessian) to the FD noise floor (~1e-7 at
        // h=1e-4 on water). Gate at 1e-6 (10× margin).
        assert!(
            err_analytic < 1.0e-6,
            "strict closed-form 2n+1 third-derivative error too large: analytic={err_analytic:.3e} frozen={err_frozen:.3e}"
        );
    }

    fn load_params_diag() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    /// **NON-EQ third-derivative gate** (the deliverable). At a stretched+bent water geometry (far
    /// from equilibrium — where the equilibrium-only gate is blind) the analytic third derivative
    /// [`third_derivative_analytic`] must match the validated seminumerical path
    /// [`third_derivative_seminumerical_dense`] (central FD of the analytic Hessian). This gate
    /// caught the **dispersion-gating bug**: `third_derivative_frozen_complete` used to add the D3
    /// 3rd derivative unconditionally (nonzero even for `reference_path = None`), so with dispersion
    /// DISABLED the analytic path carried a spurious D3 term while the seminumerical ground truth
    /// excluded it — reaching O(100%) at compressed geometries where the D3 3rd derivative blows up.
    /// Fixed by gating dispersion on `include_dispersion && enable_dispersion` (as the Hessian does).
    ///
    /// The former ~6e-4 (rel 0.2%) residual — the Pulay overlap-coefficient's coordination-number response
    /// omitted by the P/W/V density-path (`h0` reads a CN cached in `electronic`) — is now supplied by
    /// [`crate::hessian::fixed_density_pulay_cn_h0_response`] (a frozen-density, first-order ∂CN/∂R term,
    /// consistent with 2n+1). With the v0.5.0 fixes (∂K/∂q kernel chain + degenerate-orbital Λ
    /// covariance) the analytic 3rd derivative matches the seminumerical to ~6e-8 here; gate at 5e-7.
    #[test]
    fn third_derivative_nonEq_matches_seminumerical() {
        let Some(params) = load_params_diag() else { return };
        // NON-EQ: stretched + bent water (O–H ~1.15 Å, angle far from equilibrium).
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0, false,
        ).unwrap();
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: crate::electronic::ElectronicOptions {
                enable_dispersion: false,
                energy_tolerance: 1.0e-11,
                charge_tolerance: 1.0e-9,
                ..crate::electronic::ElectronicOptions::default()
            },
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let store = third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
        let semi = third_derivative_seminumerical_dense(&system, &params, options.clone(), 1.0e-4).unwrap();
        let mut err = 0.0_f64;
        let mut refm = 0.0_f64;
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            err = err.max((store[c][(a, b)] - semi.get(a, b, c)).abs());
            refm = refm.max(semi.get(a, b, c).abs());
        }}}
        // Since the v0.5.0 fixes (∂K/∂q kernel chain + degenerate-orbital Λ covariance) the
        // analytic path matches the seminumerical to ~6e-8 at this non-EQ geometry. Gate at
        // 5e-7 (~8× margin over the seminumerical's own FD noise); the historical bugs were
        // O(100%) (dispersion gating) and ~1e-6 (kernel chain) of `refm`.
        assert!(
            err < 5.0e-7,
            "non-EQ analytic 3rd derivative vs seminumerical: err={err:.3e} (ref {refm:.3e})"
        );
    }

    /// **Degenerate-orbital regression** (v0.5.0). Symmetric molecules with exactly
    /// degenerate MOs (NH₃ e-levels, CH₄ t₂ HOMO) historically had ~2e-2 relative errors
    /// in the analytic third derivative: `mo_coefficient_derivatives` left degenerate
    /// same-block rotations at zero (violating first-order orthonormality) and the
    /// gauge-dependent per-orbital ε^{(c)} entered four contractions. Fixed by the
    /// symmetric gauge `U_pq = −½S̃_pq` plus the gauge-invariant in-block matrix
    /// `Λ^c_pq = F̃^c_pq − ε S̃^c_pq`. This gate holds both systems at the FD noise floor.
    #[test]
    fn third_derivative_analytic_matches_seminumerical_degenerate_orbitals() {
        let Some(params) = load_params_diag() else { return };
        let cases: [(&str, &str); 2] = [
            (
                "ammonia C3v",
                "4\nnh3\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\n\
                 H 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n",
            ),
            (
                "methane Td",
                "5\nch4\nC 0.0 0.0 0.0\nH 0.629118 0.629118 0.629118\n\
                 H -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\n\
                 H 0.629118 -0.629118 -0.629118\n",
            ),
        ];
        for (label, xyz) in cases {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let options = crate::hessian::AnalyticHessianOptions {
                electronic_options: crate::electronic::ElectronicOptions {
                    energy_tolerance: 1.0e-11,
                    charge_tolerance: 1.0e-9,
                    ..crate::electronic::ElectronicOptions::default()
                },
                ..crate::hessian::AnalyticHessianOptions::default()
            };
            let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
            let ndof = 3 * system.atoms.len();
            let store =
                third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
            let semi =
                third_derivative_seminumerical_dense(&system, &params, options, 1.0e-4).unwrap();
            let mut err = 0.0_f64;
            let mut refm = 0.0_f64;
            for a in 0..ndof {
                for b in a..ndof {
                    for c in b..ndof {
                        err = err.max((store[c][(a, b)] - semi.get(a, b, c)).abs());
                        refm = refm.max(semi.get(a, b, c).abs());
                    }
                }
            }
            assert!(
                err < 5.0e-7,
                "{label}: degenerate-orbital analytic 3rd derivative vs seminumerical: \
                 err={err:.3e} (ref {refm:.3e}; pre-v0.5.0 bug was ~2e-2·ref)"
            );
        }
    }

    #[test]
    #[ignore]
    fn diag_nonEq_third_derivative_decompose() {
        let Some(params) = load_params_diag() else { return };
        // NON-EQ: stretched + bent water (O–H ~1.15 Å, angle far from equilibrium).
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0, false,
        ).unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: eo.clone(),
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();

        // (0) Total analytic vs seminumerical (ground truth).
        let store = third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
        let semi = third_derivative_seminumerical_dense(&system, &params, options.clone(), 1.0e-4).unwrap();
        let mut err_total = 0.0_f64;
        let mut ref_total = 0.0_f64;
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            let e = (store[c][(a, b)] - semi.get(a, b, c)).abs();
            err_total = err_total.max(e);
            ref_total = ref_total.max(semi.get(a, b, c).abs());
        }}}
        eprintln!("DIAG total analytic vs semi: err={err_total:.3e} (ref {ref_total:.3e}, rel {:.1}%)",
            100.0 * err_total / ref_total.max(1e-30));

        // === Bug-2 DEFINITIVE localization v4 (2026-07-01 — the CHANNEL-SWAP settles it: PULAY, not CN-H0) ===
        // Two DISTINCT residuals were being conflated:
        //   (A) the a↔c ASYMMETRY of the un-symmetrized raw = 6.243e-4, which lives ENTIRELY in cnh03
        //       (fixed_density_cn_h0_third_derivative; geo3/scc3/pulay3 asym ≈ 0). Symmetrization REMOVES
        //       this for free — it is NOT the surviving error. Implementing the expert's CN-H0 density-
        //       response `T_cn_h0_rho = (L(a,b|c)+L(a,c|b)+L(b,c|a))/3` is 6-perm invariant (rho-3=8.7e-19)
        //       but leaves the symmetric total UNCHANGED (SYMMETRIZED-NEW vs semi=6.141e-4). So CN-H0 is a
        //       red herring for the accuracy residual.
        //   (B) the SYMMETRIC deficit that SURVIVES full symmetrization = 6.107e-4 (store vs GG-FD and vs
        //       semi). The DECISIVE channel-swap: substituting each channel's analytic density-path by its
        //       RECONVERGED-FD density-path and re-symmetrizing —
        //          SWAP pulay->reconFD: sym vs GG = 9.185e-6, vs SEMINUMERICAL = 7.449e-7  (CLOSES it)
        //          SWAP cnh0 ->reconFD: sym vs GG = 6.107e-4                                (NO effect)
        //       Two INDEPENDENT 3rd-deriv references (double-gradient FD and canonical seminumerical,
        //       different stencils/steps) agree the pulay swap closes the residual by ~3 orders. So the
        //       missing SYMMETRIC term is in the PULAY DENSITY-PATH — the geometry×density-response cross
        //       the linearized `pulay(P^(c),W^(c),V) + [pulay(V+V^(c))−pulay(V)]` omits (|miss|=7.134e-4;
        //       the earlier reconverged-pulay 7.1e-4 was REAL, not an FD artifact — GATE3=4.5e-8 tests the
        //       gradient-level path, a different object than the Hessian-level density-response).
        // The naive candidate `fixed_density_pulay_third_derivative(P^(c),W^(c))[slab c]` is WRONG (1.08e-1,
        // it double-differentiates geometrically). The correct analytic term is still to be derived.
        // See the gates below (SWAP*/rho-3/asym[*]). Fix requires the PULAY geometry×density-response cross.
        //
        // === v5 (2026-07-02): the "V_geo_c missing" hypothesis is REFUTED for THIS codebase ===
        // The proposed fix (feed V_total_c = V_geo_c + E_qq·q_c to the pulay coeff instead of E_qq·q_c only)
        // is a NO-OP here: the production density-path in `frozen_hessian_density_path` (and `pulay_ana`)
        // ALREADY builds v_c = dscalar[(s,c)] + Σ shell_kernel·q_c, and P0-check proves dscalar (hessian::
        // shell_scalar_potential_first_derivatives) == vgeo (cphf::shell_scalar_potential_derivatives) EXACTLY
        // (err=0.000e0). Rigorous FD disproof of the V-channel being the gap:
        //   P1  V_geo_c analytic vs FD@fixed-q0                                    = 1.06e-11 (V_geo exact)
        //   P3  pulay density-path with V_total vs reconverged truth               = 7.134e-4 (UNCHANGED)
        //   P3b pulay density-path fed the EXACT reconverged (dP,dW,dV) vs truth   = 7.134e-4 (|dV−v_c|=7.2e-11)
        //   P3c coordinator closed form −P0·V_geo_c:S_ab vs miss[c]                 = 5.87e-3  (wrong object)
        // ⇒ P,W,V responses are ALL exact; the 7.134e-4 miss SURVIVES even with exact reconverged (dP,dW,dV)
        // fed to the linearized `h1(dP,dW,V0)+[h2(V0+dV)−h0]` split. So the gap is NOT a missing/wrong response
        // field — it is the GEOMETRY×DENSITY-RESPONSE bilinear the linearized Hessian-level split cannot see:
        // recon_dpath differentiates `pulay_hess` GEOMETRICALLY (slab c on S_ab→S_abc and h0-poly) while the
        // coefficient carries the density response, and holds base CN/V in the fixed reference — a coupling
        // absent from the base-geometry linear path. The C:S_ab channel carries 6.90e-4 of it, h0-channel
        // 2.40e-5. Deriving the correct analytic cross (NOT the 1.08e-1 full-∂³S) remains open. Reported to
        // coordinator; the v5 scratchpad fix does not apply because V_geo is already present here.
        //
        // === v6 (2026-07-02): the miss is REAL & STABLE, but resists ALL closed forms (accept-0.2% territory) ===
        // FD-step invariance: |miss| = 7.1341e-4 at h ∈ {2e-5, 5e-5, 1e-4, 2e-4} — IDENTICAL to 5 s.f. ⇒ a
        // genuine, well-defined 3rd-derivative term, NOT an FD artifact and NOT FD noise. Yet every derived
        // ordered closed form fails its FD gate:
        //   - C^(c)·S_abc (ordered, c=slab):  norm 7.4e-2, but miss_csab=5.9e-4 (125× too big — NOT present).
        //   - P^(c)·(2h0_c − V_geo_c)·S_ab:   proj(miss_csab onto 2h0_c·S_ab)=0.054, onto (−V_geo·S_ab)=0.043
        //                                     ⇒ miss_csab is ORTHOGONAL to both — neither is the term.
        // Paradox: a first-principles linear-response expansion of recon_dpath's (recon−fixed) FD predicts the
        // C:S_ab miss should CANCEL to O(h²) [ana = C^(c)·S_ab exactly = recon leading term], yet the measured,
        // FD-step-stable residual is 5.9e-4 and orthogonal to the obvious geometry-of-coefficient terms. The
        // surviving structure is some higher/mixed coupling of the pulay bilinear (−P·V and the 2p·h0-poly)
        // with the reconverged density that the analytic first-order Hessian-split does not reproduce. After a
        // genuine multi-form derivation effort (C_a:S_b, CN-H0/t_rho, V_geo-total, −P0·V_geo:S_ab, ordered
        // C·S_abc, coefficient-geometry P^(c)(2h0_c−V_geo_c):S_ab — all FD-gated, all failed), the closest
        // clean analytic candidate (ordered coefficient-geometry) still leaves 1.99e-3 on the C:S_ab channel.
        // === v7 (2026-07-02): RESOLVED — the miss is the Pulay overlap-coefficient's CN-response ==========
        // The v6 "analytically-elusive" verdict was WRONG. STEP1 (C:S_ab sub-channel split) shows the entire
        // 6.903e-4 lives in the [P·2h0] sub-block, with [−P·V] and [−2W] EXACT (~1e-10) — refuting the W/EWD
        // and V hypotheses. Because [−P·V] uses the same density `p`, the density response ∂p/∂c==P^(c) is
        // confirmed; the miss is the OTHER field in P·2h0: `h0` reads a CN cached in `electronic`
        // (h0_prefactor_second → electronic.coordination_numbers), so the reconverged path differentiates CN
        // while the base-CN linearized density-path does not. The exact closed form (both pulay channels) is
        // fixed_density_pulay_cn_h0_response(cn_grad_c), cn_grad_c[at]=∂CN_at/∂R_c: STEP5 matches miss[c] to
        // 1.4e-10, STEP6 (add to raw, symmetrize) closes the total to 9.2e-6 vs GG / 7.4e-7 vs semi — NO
        // double-count (it closes, not overshoots). Wired into third_derivative_closed_form_total (frozen,
        // first-order ∂CN/∂R — consistent with 2n+1); the non-EQ gate is now tightened 2e-3 → 1e-5.
        let nshell = electronic.shell_charges.len();
        let cutoff2 = cutoff;
        let shell_kernel = crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dscalar = crate::hessian::shell_scalar_potential_first_derivatives(&system, &electronic.basis, &electronic.shell_charges, &params).unwrap();
        let ao_opts = crate::cphf::AoDerivativeOptions { coordination_cutoff: cutoff, include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian };
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(&system, &params, &electronic, ao_opts, crate::cphf::CpxtbOptions::default()).unwrap();
        let displace = |c: usize, sign: f64| -> PeriodicSystem {
            let (atom, ax) = (c / 3, c % 3);
            let mut s = system.clone();
            let h = 5.0e-5; match ax { 0 => s.atoms[atom].position.x += sign * h, 1 => s.atoms[atom].position.y += sign * h, _ => s.atoms[atom].position.z += sign * h };
            s
        };
        let h = 5.0e-5;
        let recon = |sys: &PeriodicSystem| crate::electronic::run_electronic(sys, &params, eo.clone()).unwrap();
        // The RECONVERGED pulay density-path (reconv − fixed FD) vs the analytic
        // `pulay(P^(c),W^(c),V) + [pulay(V+V^(c)) − pulay(V)]`. This is the ~6e-4 residual: the analytic
        // form is EXACT as a fixed-geometry directional derivative (verified ~6e-12) but MISSES a
        // geometry-integral × density-response cross term unique to the pulay bilinear (its `overlap_coeff`
        // couples the CN-dependent h0 and the second-derivative overlap to the density). s2/scalar_overlap/
        // cn_h0 density-paths are exact to ~1e-11. Fixing this closes the last ~6e-4 to reach ~1e-4.
        let mut pulay_err = 0.0_f64;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
            let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
            let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            let sp = displace(c, 1.0); let sm = displace(c, -1.0);
            let ep = recon(&sp); let em = recon(&sm);
            let rp = crate::hessian::fixed_density_pulay_hessian(&sp, &params, &ep).unwrap().hessian;
            let rm = crate::hessian::fixed_density_pulay_hessian(&sm, &params, &em).unwrap().hessian;
            let fp = crate::hessian::fixed_density_pulay_hessian(&sp, &params, &electronic).unwrap().hessian;
            let fm = crate::hessian::fixed_density_pulay_hessian(&sm, &params, &electronic).unwrap().hessian;
            for a in 0..ndof { for b in 0..ndof {
                let fd_path = (rp[(a, b)] - rm[(a, b)]) / (2.0 * h) - (fp[(a, b)] - fm[(a, b)]) / (2.0 * h);
                let ana = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)];
                pulay_err = pulay_err.max((ana - fd_path).abs());
            }}
        }
        let _ = cutoff2;
        eprintln!("DIAG   PULAY reconverged density-path: analytic vs FD err={pulay_err:.3e} (the Bug-2 residual)");

        // === GATE 2 (anti-trap): is `fixed_density_pulay_hessian.hessian` == FD of its OWN gradient? ===
        // H_ab should equal [g_a(R+h_b) − g_a(R−h_b)]/2h at FIXED density. If NOT, the Hessian function
        // itself is incomplete — which the reconverged density-path FD (Gate 4) exposes but a Hessian-vs-
        // Hessian FD hides (shared omission). This is the guidance's key anti-trap.
        let pulay_g = |sys: &PeriodicSystem| crate::hessian::fixed_density_pulay_hessian(sys, &params, &electronic).unwrap().gradient;
        let mut gate2 = 0.0_f64;
        let hg = 1.0e-5;
        for b in 0..ndof {
            let (atom, ax) = (b / 3, b % 3);
            let mut sp = system.clone(); let mut sm = system.clone();
            match ax { 0 => { sp.atoms[atom].position.x += hg; sm.atoms[atom].position.x -= hg; },
                1 => { sp.atoms[atom].position.y += hg; sm.atoms[atom].position.y -= hg; },
                _ => { sp.atoms[atom].position.z += hg; sm.atoms[atom].position.z -= hg; } };
            let gp = pulay_g(&sp); let gm = pulay_g(&sm);
            let hess = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            for a in 0..ndof {
                let gpa = [gp[a/3].x, gp[a/3].y, gp[a/3].z][a%3];
                let gma = [gm[a/3].x, gm[a/3].y, gm[a/3].z][a%3];
                let fd = (gpa - gma) / (2.0 * hg);
                gate2 = gate2.max((hess[(a, b)] - fd).abs());
            }
        }
        eprintln!("DIAG   GATE2 pulay Hessian vs FD-of-gradient: err={gate2:.3e}  (if >>1e-9, the Hessian fn is incomplete)");

        // === GATE 3 (the true density-path reference): MIXED gradient FD ∂²g_a/∂z∂R_b along Δz_c ===
        // L_ab|c = d/dλ [ dg_a/dR_b at (z + λΔz_c) ] |_0, Δz_c = {P^(c),W^(c),V^(c)}. Compare to the
        // analytic pulay density-path (h1 + h2 − h0). If they match (~1e-11), the analytic IS correct and
        // the reconverged-FD residual is a geometry×density SECOND-order artifact, not a missing term.
        let lam = 1.0e-4;
        let pulay_g_at = |sys: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian(sys, &params, e).unwrap().gradient;
        let mut gate3 = 0.0_f64;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            // z ± λΔz_c electronic states (fixed geometry, density fields shifted).
            let mut ezp = electronic.clone(); let mut ezm = electronic.clone();
            for r in 0..electronic.density.rows() { for k in 0..electronic.density.cols() {
                ezp.density[(r, k)] += lam * p_c[(r, k)]; ezm.density[(r, k)] -= lam * p_c[(r, k)];
                ezp.energy_weighted_density[(r, k)] += lam * w_c[(r, k)]; ezm.energy_weighted_density[(r, k)] -= lam * w_c[(r, k)];
            }}
            for s in 0..nshell { ezp.shell_scc_potential[s] += lam * v_c[s]; ezm.shell_scc_potential[s] -= lam * v_c[s]; }
            // analytic pulay density-path for this slab c.
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
            let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
            let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            for b in 0..ndof {
                let (atom, ax) = (b / 3, b % 3);
                let mut sp = system.clone(); let mut sm = system.clone();
                match ax { 0 => { sp.atoms[atom].position.x += hg; sm.atoms[atom].position.x -= hg; },
                    1 => { sp.atoms[atom].position.y += hg; sm.atoms[atom].position.y -= hg; },
                    _ => { sp.atoms[atom].position.z += hg; sm.atoms[atom].position.z -= hg; } };
                let gzp_p = pulay_g_at(&sp, &ezp); let gzp_m = pulay_g_at(&sm, &ezp);
                let gzm_p = pulay_g_at(&sp, &ezm); let gzm_m = pulay_g_at(&sm, &ezm);
                for a in 0..ndof {
                    let idx = a % 3; let at = a / 3;
                    let g = |v: &[crate::math::Vec3]| [v[at].x, v[at].y, v[at].z][idx];
                    // mixed FD: [ (g(z+,R+)−g(z+,R−)) − (g(z−,R+)−g(z−,R−)) ] / (4 λ hg)
                    let mixed = ((g(&gzp_p) - g(&gzp_m)) - (g(&gzm_p) - g(&gzm_m))) / (4.0 * lam * hg);
                    let ana = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)];
                    gate3 = gate3.max((ana - mixed).abs());
                }
            }
        }
        eprintln!("DIAG   GATE3 pulay density-path vs MIXED gradient FD: err={gate3:.3e}  (the TRUE anti-trap reference)");

        // === Is the analytic HESSIAN itself correct at non-EQ? Compare to FD of the analytic GRADIENT. ===
        // If the analytic Hessian has a small non-EQ error, both `store` and the seminumerical (FD of the
        // Hessian) inherit it → the 6.1e-4 could be a HESSIAN bug, not a 3rd-derivative bug.
        let grad_at = |sys: &PeriodicSystem| -> Vec<f64> {
            let gopt = crate::gradient::AnalyticGradientOptions { electronic: eo.clone(), include_dispersion: false, ..Default::default() };
            let g = crate::gradient::analytic_gradient(sys, &params, gopt).unwrap().gradient;
            (0..ndof).map(|i| [g[i/3].x, g[i/3].y, g[i/3].z][i%3]).collect()
        };
        let hess_ref = crate::hessian::analytic_hessian(&system, &params, options.clone()).unwrap().hessian;
        let mut hess_vs_grad = 0.0_f64;
        for b in 0..ndof {
            let (atom, ax) = (b / 3, b % 3);
            let mut sp = system.clone(); let mut sm = system.clone();
            match ax { 0 => { sp.atoms[atom].position.x += hg; sm.atoms[atom].position.x -= hg; },
                1 => { sp.atoms[atom].position.y += hg; sm.atoms[atom].position.y -= hg; },
                _ => { sp.atoms[atom].position.z += hg; sm.atoms[atom].position.z -= hg; } };
            let gp = grad_at(&sp); let gm = grad_at(&sm);
            for a in 0..ndof {
                let fd = (gp[a] - gm[a]) / (2.0 * hg);
                hess_vs_grad = hess_vs_grad.max((hess_ref[(a, b)] - fd).abs());
            }
        }
        eprintln!("DIAG   analytic HESSIAN vs FD-of-analytic-GRADIENT (non-EQ): err={hess_vs_grad:.3e}");

        // === Independent 3rd-deriv reference: DOUBLE FD of the analytic GRADIENT (no Hessian fn). ===
        // T_abc = ∂²g_a/∂R_b∂R_c. Compare `store` (analytic 3rd deriv) AND the un-symmetrized sum of
        // exactly-validated components to this clean reference and to the seminumerical.
        let resp = closed_form_response_hessian_derivative(&system, &params, &electronic, &cphf, ao_opts, cutoff).unwrap();
        let l_abc_geo = third_derivative_frozen_complete(&system, &params, &electronic, None, cutoff, false).unwrap();
        let scalar3 = crate::hessian::fixed_density_scalar_overlap_third_derivative(&system, &params, &electronic).unwrap();
        // double gradient FD: g_a(R + h_b + h_c) etc. (4-point mixed for b≠c; forward-central for the diagonal).
        let g_disp = |db: usize, dc: usize| -> Vec<f64> {
            let mut s = system.clone();
            for (d, sign) in [(db, 1.0_f64), (dc, 1.0_f64)] {
                let (at, ax) = (d / 3, d % 3);
                match ax { 0 => s.atoms[at].position.x += sign * hg, 1 => s.atoms[at].position.y += sign * hg, _ => s.atoms[at].position.z += sign * hg };
            }
            grad_at(&s)
        };
        // Use the standard mixed 2nd derivative of g_a: [g(+b,+c) − g(+b,−c) − g(−b,+c) + g(−b,−c)]/(4h²).
        let gmix = |db: usize, dc: usize| -> Vec<f64> {
            let disp = |sb: f64, sc: f64| -> Vec<f64> {
                let mut s = system.clone();
                for (d, sign) in [(db, sb), (dc, sc)] { let (at, ax) = (d / 3, d % 3); match ax { 0 => s.atoms[at].position.x += sign * hg, 1 => s.atoms[at].position.y += sign * hg, _ => s.atoms[at].position.z += sign * hg }; }
                grad_at(&s)
            };
            let pp = disp(1.0, 1.0); let pm = disp(1.0, -1.0); let mp = disp(-1.0, 1.0); let mm = disp(-1.0, -1.0);
            (0..ndof).map(|a| (pp[a] - pm[a] - mp[a] + mm[a]) / (4.0 * hg * hg)).collect()
        };
        let _ = g_disp;
        // Precompute l_abx per slab c (density-path).
        let l_abx_all: Vec<Matrix> = (0..ndof).map(|c| {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            frozen_hessian_density_path(&system, &params, &electronic, cutoff, p_c, w_c, q_c, &v_c).unwrap()
        }).collect();
        let (mut store_vs_gg, mut unsym_vs_gg, mut store_vs_semi2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c); // gg[a] = ∂²g_a/∂b∂c = T_abc
            for a in 0..ndof {
                let unsym_full = resp[c][(a, b)] + l_abc_geo[c][(a, b)] + scalar3[c][(a, b)] + l_abx_all[c][(a, b)];
                store_vs_gg = store_vs_gg.max((store[c][(a, b)] - gg[a]).abs());
                unsym_vs_gg = unsym_vs_gg.max((unsym_full - gg[a]).abs());
                store_vs_semi2 = store_vs_semi2.max((store[c][(a, b)] - semi.get(a, b, c)).abs());
            }
        }}
        eprintln!("DIAG   store vs DOUBLE-GRADIENT-FD: err={store_vs_gg:.3e};  un-symmetrized-sum vs GG-FD: err={unsym_vs_gg:.3e};  store vs semi: {store_vs_semi2:.3e}");

        // Does `store` (symmetrized driver) == raw un-symmetrized sum of components? And does the raw
        // un-symmetrized sum, SYMMETRIZED by hand over 6 perms, == store?
        let mut store_vs_rawsum = 0.0_f64;
        let raw = |a: usize, b: usize, c: usize| resp[c][(a, b)] + l_abc_geo[c][(a, b)] + scalar3[c][(a, b)] + l_abx_all[c][(a, b)];
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            store_vs_rawsum = store_vs_rawsum.max((store[c][(a, b)] - raw(a, b, c)).abs());
        }}}
        // hand-symmetrized raw over 6 permutations of (a,b,c).
        let mut symraw_vs_gg = 0.0_f64;
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let s = (raw(a, b, c) + raw(a, c, b) + raw(b, a, c) + raw(b, c, a) + raw(c, a, b) + raw(c, b, a)) / 6.0;
                symraw_vs_gg = symraw_vs_gg.max((s - gg[a]).abs());
            }
        }}
        eprintln!("DIAG   store vs raw-unsym-sum: err={store_vs_rawsum:.3e};  hand-symmetrized-raw vs GG-FD: err={symraw_vs_gg:.3e}");
        // Per-component un-symmetrized asymmetry: how non-symmetric is `raw(a,b,c)` under a↔c and b↔c?
        let mut asym_bc = 0.0_f64; let mut asym_ac = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            asym_bc = asym_bc.max((raw(a, b, c) - raw(a, c, b)).abs());
            asym_ac = asym_ac.max((raw(a, b, c) - raw(c, b, a)).abs());
        }}}
        eprintln!("DIAG   raw-sum asymmetry: |raw(a,b,c)−raw(a,c,b)|={asym_bc:.3e}  |raw(a,b,c)−raw(c,b,a)|={asym_ac:.3e}");

        // Per-component a↔c asymmetry: which piece carries the 6.2e-4 asymmetry?
        let comp_asym = |name: &str, m: &dyn Fn(usize, usize, usize) -> f64| {
            let mut e = 0.0_f64;
            for a in 0..ndof { for b in 0..ndof { for c in 0..ndof { e = e.max((m(a, b, c) - m(c, b, a)).abs()); }}}
            eprintln!("DIAG   asym[{name}] |m(a,b,c)−m(c,b,a)|={e:.3e}");
        };
        comp_asym("resp", &|a, b, c| resp[c][(a, b)]);
        comp_asym("l_abc_geo", &|a, b, c| l_abc_geo[c][(a, b)]);
        comp_asym("scalar3", &|a, b, c| scalar3[c][(a, b)]);
        comp_asym("l_abx", &|a, b, c| l_abx_all[c][(a, b)]);
        // And: raw is symmetric in a↔b (Hessian symmetry)?
        let mut ab_asym = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof { ab_asym = ab_asym.max((raw(a, b, c) - raw(b, a, c)).abs()); }}}
        eprintln!("DIAG   raw a↔b (Hessian) asymmetry: {ab_asym:.3e}");

        // Which GEOMETRIC block of l_abc_geo carries the uncancelled 6.243e-4 a↔c asymmetry?
        let geo3 = third_derivative_geometric(&system, &params).unwrap().to_dense_slabs();
        let scc3 = crate::hessian::fixed_shell_charge_scc_third_derivative(&system, &electronic.basis, &electronic.shell_charges, &params).unwrap();
        let pulay3 = crate::hessian::fixed_density_pulay_third_derivative(&system, &params, &electronic).unwrap();
        let cnh03 = crate::hessian::fixed_density_cn_h0_third_derivative(&system, &params, &electronic, cutoff).unwrap();
        comp_asym("geo3(rep+hal)", &|a, b, c| geo3[c][(a, b)]);
        comp_asym("scc3", &|a, b, c| scc3[c][(a, b)]);
        comp_asym("pulay3", &|a, b, c| pulay3[c][(a, b)]);
        comp_asym("cnh03", &|a, b, c| cnh03[c][(a, b)]);

        // === CN-H0 density-response symmetrizing term prototype (v3 fix) ===============
        // The 6.243e-4 residual is the a↔c asymmetry of the CN-H0 DENSITY-PATH block. The current
        // density-path `l_abx` uses `∂_c(H_bc)` via P^(c) fed into fixed_density_cn_h0_hessian+cross,
        // which is the ASYMMETRIC `∂_c(CN_bc·e + CN_b·e_c + e_b·CN_c)` — it omits the `CN·de_bc`
        // density-response symmetrizing term. Replace it by the 3-perm average of
        //   L(i,j|k) = Σ_I [ e_I^[k]·N_{I,ij} + ½ e_{I,i}^[k]·N_{I,j} + ½ e_{I,j}^[k]·N_{I,i} ]
        // where e_I^[k], e_{I,i}^[k] = cn_h0_dedcn_jets evaluated with the RESPONSE density P^(k).
        // FIRST-ORDER-RESPONSE ONLY (feed P^(k) into existing d_edcn — no 2nd-order CPHF).
        //
        // cn jet (geometric): N_{I,a}=grad, N_{I,ab}=hess. de^[k] jet (response density P^(k)).
        let cn_jet = crate::hessian::cn_h0_cn_jets(&system, cutoff).unwrap();
        let de_resp: Vec<Vec<crate::hessian::DedcnJet>> = (0..ndof)
            .map(|k| {
                let mut e = electronic.clone();
                e.density = cphf.density_responses[k].clone();
                crate::hessian::cn_h0_dedcn_jets(&system, &params, &e).unwrap()
            })
            .collect();
        // L(i,j|k) = Σ_I [ e_I^[k]·N_{I,ij} + ½ e_{I,i}^[k]·N_{I,j} + ½ e_{I,j}^[k]·N_{I,i} ].
        let ell = |i: usize, j: usize, k: usize| -> f64 {
            let mut t = 0.0;
            for at in 0..system.atoms.len() {
                let (n, ek) = (&cn_jet[at], &de_resp[k][at]);
                t += ek.value * n.hess[i * ndof + j]
                    + 0.5 * ek.grad[i] * n.grad[j]
                    + 0.5 * ek.grad[j] * n.grad[i];
            }
            t
        };
        // rho-2: de_sym[i,j] = ½(e_{I,i}^[j] + e_{I,j}^[i]) symmetric under i↔j (structurally exact).
        // rho-3: T_cn_h0_rho(a,b,c) = (L(a,b|c)+L(a,c|b)+L(b,c|a))/3 — 6-perm invariant.
        let t_rho = |a: usize, b: usize, c: usize| (ell(a, b, c) + ell(a, c, b) + ell(b, c, a)) / 3.0;
        let mut rho3_asym = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            rho3_asym = rho3_asym.max((t_rho(a, b, c) - t_rho(c, b, a)).abs());
            rho3_asym = rho3_asym.max((t_rho(a, b, c) - t_rho(a, c, b)).abs());
            rho3_asym = rho3_asym.max((t_rho(a, b, c) - t_rho(b, a, c)).abs());
        }}}
        eprintln!("DIAG   rho-3: T_cn_h0_rho 6-perm asymmetry (should be ~0)={rho3_asym:.3e}");

        // The CN-H0 density-path currently in `l_abx_all[c]` is fixed_density_cn_h0_hessian(P^(c)) +
        // fixed_density_cn_h0_pulay_cross_hessian(P^(c)). Its value at (a,b) is the OLD block. Compare
        // the replacement: raw_new = raw − cnh0_dpath_old(a,b,c) + t_rho(a,b,c), and check its a↔c
        // asymmetry drops to ~0 and the symmetrized total matches the GG-FD ground truth.
        let cnh0_dpath_old = |c: usize| -> Matrix {
            let mut e = electronic.clone();
            e.density = cphf.density_responses[c].clone();
            let a = crate::hessian::fixed_density_cn_h0_hessian(&system, &params, &e, cutoff).unwrap().hessian;
            let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&system, &params, &e, cutoff).unwrap();
            let mut m = a; for r in 0..ndof { for cc in 0..ndof { m[(r, cc)] += cr[(r, cc)]; } } m
        };
        let old_dpath: Vec<Matrix> = (0..ndof).map(cnh0_dpath_old).collect();
        // (i) Is the existing CN-H0 density-path contribution == L(a,b|c) = ell(a,b,c)?
        let mut dpath_vs_ell = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            dpath_vs_ell = dpath_vs_ell.max((old_dpath[c][(a, b)] - ell(a, b, c)).abs());
        }}}
        eprintln!("DIAG   old CN-H0 density-path[c](a,b) vs ell(a,b|c): err={dpath_vs_ell:.3e}");
        // (ii) EXPLICIT-CORRECTION form: replace L(a,b|c) by the symmetric average T_sym.
        //   raw_corr = raw − ell(a,b,c) + t_rho(a,b,c)   [add T_sym − L_ab|c on top].
        let raw_corr = |a: usize, b: usize, c: usize| raw(a, b, c) - ell(a, b, c) + t_rho(a, b, c);
        let mut asym_ac_new = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            asym_ac_new = asym_ac_new.max((raw_corr(a, b, c) - raw_corr(c, b, a)).abs());
        }}}
        eprintln!("DIAG   raw_corr a↔c asymmetry (should be ~0)={asym_ac_new:.3e}");
        // Symmetrized correction total vs the double-gradient FD ground truth AND vs semi.
        let mut new_vs_gg = 0.0_f64; let mut new_vs_semi = 0.0_f64;
        let mut store_corr_vs_gg = 0.0_f64;
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let s = (raw_corr(a, b, c) + raw_corr(a, c, b) + raw_corr(b, a, c)
                    + raw_corr(b, c, a) + raw_corr(c, a, b) + raw_corr(c, b, a)) / 6.0;
                new_vs_gg = new_vs_gg.max((s - gg[a]).abs());
                new_vs_semi = new_vs_semi.max((s - semi.get(a, b, c)).abs());
                // Also: store (already symmetric) + symmetric correction (t_rho−ell averaged).
                let corr = ((t_rho(a, b, c) - ell(a, b, c)) + (t_rho(a, c, b) - ell(a, c, b))
                    + (t_rho(b, a, c) - ell(b, a, c)) + (t_rho(b, c, a) - ell(b, c, a))
                    + (t_rho(c, a, b) - ell(c, a, b)) + (t_rho(c, b, a) - ell(c, b, a))) / 6.0;
                store_corr_vs_gg = store_corr_vs_gg.max((store[c][(a, b)] + corr - gg[a]).abs());
            }
        }}
        eprintln!("DIAG   store+symcorr vs GG-FD={store_corr_vs_gg:.3e}");

        // === Directly identify the missing SYMMETRIC term ===
        // residual R_abc = gg[a] − store[c][(a,b)] (symmetric ground-truth minus current symmetric store).
        // Compare against candidate missing terms:
        //   sym(t_rho)               (= t_rho, already symmetric)
        //   sym(cnh03)               (the frozen block, symmetrized)
        //   t_rho − sym(cnh03)       (the density-response CN·de_bc that the frozen block omits)
        let sym3 = |m: &dyn Fn(usize, usize, usize) -> f64, a: usize, b: usize, c: usize| {
            (m(a, b, c) + m(a, c, b) + m(b, a, c) + m(b, c, a) + m(c, a, b) + m(c, b, a)) / 6.0
        };
        let cnh03f = |a: usize, b: usize, c: usize| cnh03[c][(a, b)];
        let (mut r_vs_trho, mut r_vs_trho_m_cnh, mut ref_r) = (0.0_f64, 0.0_f64, 0.0_f64);
        // ratio probe: mean(R / t_rho) over large entries.
        let (mut ratio_num, mut ratio_den) = (0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let r_abc = gg[a] - store[c][(a, b)];
                ref_r = ref_r.max(r_abc.abs());
                r_vs_trho = r_vs_trho.max((r_abc - t_rho(a, b, c)).abs());
                let cand = t_rho(a, b, c) - sym3(&cnh03f, a, b, c);
                r_vs_trho_m_cnh = r_vs_trho_m_cnh.max((r_abc - cand).abs());
                if t_rho(a, b, c).abs() > 1e-6 { ratio_num += r_abc * t_rho(a, b, c); ratio_den += t_rho(a, b, c) * t_rho(a, b, c); }
            }
        }}
        let ratio = if ratio_den > 0.0 { ratio_num / ratio_den } else { 0.0 };
        eprintln!("DIAG   missing-symmetric residual |R|={ref_r:.3e}");
        eprintln!("DIAG   R vs t_rho: err={r_vs_trho:.3e}  | R vs (t_rho−sym(cnh03)): err={r_vs_trho_m_cnh:.3e}  | proj ratio R·trho/|trho|²={ratio:.3}");

        // === Candidate: FULL frozen product 3rd-deriv (8-term Leibniz) minus cnh03 ===
        // E = Σ_I N_I·e_I with FROZEN jets (de from `electronic`). cnh03 uses only the 6 terms of
        // ∂_c(CN_bc·e + CN_b·e_c + e_b·CN_c) i.e. it OMITS the `CN·e_bc` group. The full symmetric
        // product 3rd-deriv adds N_a·e_bc + N_b·e_ac + N_c·e_ab + N·e_abc (frozen e). Test whether the
        // omitted frozen group equals the residual R.
        let de_froz = crate::hessian::cn_h0_dedcn_jets(&system, &params, &electronic).unwrap();
        let full_prod = |a: usize, b: usize, c: usize| -> f64 {
            let mut t = 0.0;
            for at in 0..system.atoms.len() {
                let (n, d) = (&cn_jet[at], &de_froz[at]);
                t += n.third[(a * ndof + b) * ndof + c] * d.value
                    + n.hess[a * ndof + b] * d.grad[c]
                    + n.hess[a * ndof + c] * d.grad[b]
                    + n.hess[b * ndof + c] * d.grad[a]
                    + n.grad[a] * d.hess[b * ndof + c]
                    + n.grad[b] * d.hess[a * ndof + c]
                    + n.grad[c] * d.hess[a * ndof + b]
                    + n.value * d.third[(a * ndof + b) * ndof + c];
            }
            t
        };
        // Omitted frozen group = full_prod − cnh03 (both un-symmetrized). Symmetrize and compare to R.
        let omitted_froz = |a: usize, b: usize, c: usize| full_prod(a, b, c) - cnh03[c][(a, b)];
        let (mut r_vs_omit, mut r_vs_omit_sym) = (0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let r_abc = gg[a] - store[c][(a, b)];
                r_vs_omit = r_vs_omit.max((r_abc - omitted_froz(a, b, c)).abs());
                r_vs_omit_sym = r_vs_omit_sym.max((r_abc - sym3(&omitted_froz, a, b, c)).abs());
            }
        }}
        eprintln!("DIAG   R vs omitted-frozen(unsym): err={r_vs_omit:.3e}  | R vs sym(omitted-frozen): err={r_vs_omit_sym:.3e}");
        // Also: does full_prod (fully symmetric already?) match R directly as the whole missing block?
        let mut full_asym = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof {
            full_asym = full_asym.max((full_prod(a, b, c) - full_prod(c, b, a)).abs());
        }}}
        eprintln!("DIAG   full_prod a↔c asymmetry (should be ~0 if truly symmetric)={full_asym:.3e}");

        // === DECISIVE channel attribution: swap each channel's analytic density-path for its
        // RECONVERGED-FD density-path in `raw`, symmetrize, and see which swap closes R to ~0. ===
        // reconverged density-path for a channel `block`: ∂_c[block(reconv@R±c)] − ∂_c[block(fixed)].
        let recon_dpath = |block: &dyn Fn(&PeriodicSystem, &ElectronicResult) -> Matrix, c: usize| -> Matrix {
            let sp = displace(c, 1.0); let sm = displace(c, -1.0);
            let ep = recon(&sp); let em = recon(&sm);
            let rp = block(&sp, &ep); let rm = block(&sm, &em);
            let fp = block(&sp, &electronic); let fm = block(&sm, &electronic);
            let mut m = crate::linalg::Matrix::zeros(ndof, ndof);
            for a in 0..ndof { for b in 0..ndof {
                m[(a, b)] = (rp[(a, b)] - rm[(a, b)]) / (2.0 * h) - (fp[(a, b)] - fm[(a, b)]) / (2.0 * h);
            }}
            m
        };
        let pulay_block = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian(s, &params, e).unwrap().hessian;
        let cnh0_block = |s: &PeriodicSystem, e: &ElectronicResult| {
            let a = crate::hessian::fixed_density_cn_h0_hessian(s, &params, e, cutoff).unwrap().hessian;
            let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(s, &params, e, cutoff).unwrap();
            let mut m = a; for r in 0..ndof { for cc in 0..ndof { m[(r, cc)] += cr[(r, cc)]; } } m
        };
        // Analytic pulay density-path per c (the h1+h2−h0 form) to subtract when swapping.
        let pulay_ana_dpath = |c: usize| -> Matrix {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
            let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
            let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            let mut m = crate::linalg::Matrix::zeros(ndof, ndof);
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)]; } }
            m
        };
        let pulay_recon: Vec<Matrix> = (0..ndof).map(|c| recon_dpath(&pulay_block, c)).collect();
        let pulay_ana: Vec<Matrix> = (0..ndof).map(pulay_ana_dpath).collect();
        let cnh0_recon: Vec<Matrix> = (0..ndof).map(|c| recon_dpath(&cnh0_block, c)).collect();
        let cnh0_ana: Vec<Matrix> = (0..ndof).map(|c| old_dpath[c].clone()).collect();
        // raw with pulay density-path swapped to reconverged.
        let raw_swap_pulay = |a: usize, b: usize, c: usize| raw(a, b, c) - pulay_ana[c][(a, b)] + pulay_recon[c][(a, b)];
        let raw_swap_cnh0 = |a: usize, b: usize, c: usize| raw(a, b, c) - cnh0_ana[c][(a, b)] + cnh0_recon[c][(a, b)];
        let raw_swap_both = |a: usize, b: usize, c: usize| raw(a, b, c) - pulay_ana[c][(a, b)] + pulay_recon[c][(a, b)] - cnh0_ana[c][(a, b)] + cnh0_recon[c][(a, b)];
        let (mut sp_gg, mut sc_gg, mut sboth_gg) = (0.0_f64, 0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let sp = sym3(&raw_swap_pulay, a, b, c);
                let sc = sym3(&raw_swap_cnh0, a, b, c);
                let sb = sym3(&raw_swap_both, a, b, c);
                sp_gg = sp_gg.max((sp - gg[a]).abs());
                sc_gg = sc_gg.max((sc - gg[a]).abs());
                sboth_gg = sboth_gg.max((sb - gg[a]).abs());
            }
        }}
        eprintln!("DIAG   SWAP pulay->reconFD: sym vs GG={sp_gg:.3e}  | SWAP cnh0->reconFD: {sc_gg:.3e}  | SWAP both: {sboth_gg:.3e}");

        // Robustness: compare SWAP-pulay symmetrized total against the independent SEMINUMERICAL
        // (different stencil & step than gg) to rule out shared-FD-noise coincidence.
        let mut sp_semi = 0.0_f64;
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            let s = sym3(&raw_swap_pulay, a, b, c);
            sp_semi = sp_semi.max((s - semi.get(a, b, c)).abs());
        }}}
        eprintln!("DIAG   SWAP pulay->reconFD: sym vs SEMINUMERICAL={sp_semi:.3e}  (indep. reference)");

        // === Identify the missing PULAY analytic term ===
        // Hypothesis: the missing symmetric term is the GEOMETRY×DENSITY-RESPONSE cross — the geometric
        // slab-c derivative of the pulay Hessian evaluated at the RESPONSE density (P^(c),W^(c)), which
        // the linearized density-path (h1+[h2−h0]) omits. Candidate = slab c of
        // fixed_density_pulay_third_derivative evaluated with density=P^(c),W^(c).
        let miss = |c: usize| -> Matrix {
            let mut m = pulay_recon[c].clone();
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] -= pulay_ana[c][(a, b)]; } }
            m
        };
        let cand_geomxdens = |c: usize| -> Matrix {
            let mut e = electronic.clone();
            e.density = cphf.density_responses[c].clone();
            e.energy_weighted_density = cphf.energy_weighted_density_responses[c].clone();
            let slabs = crate::hessian::fixed_density_pulay_third_derivative(&system, &params, &e).unwrap();
            slabs[c].clone()
        };
        let mut miss_vs_cand = 0.0_f64; let mut miss_norm = 0.0_f64;
        for c in 0..ndof {
            let m = miss(c); let cand = cand_geomxdens(c);
            for a in 0..ndof { for b in 0..ndof {
                miss_norm = miss_norm.max(m[(a, b)].abs());
                miss_vs_cand = miss_vs_cand.max((m[(a, b)] - cand[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   pulay missing term |miss|={miss_norm:.3e}  vs geom×dens candidate: err={miss_vs_cand:.3e}");
        // If the candidate matches, adding it (symmetrized) into the total should also close the residual.
        let cand_slabs: Vec<Matrix> = (0..ndof).map(cand_geomxdens).collect();
        let raw_add_cand = |a: usize, b: usize, c: usize| raw(a, b, c) + cand_slabs[c][(a, b)];
        let (mut addc_gg, mut addc_semi) = (0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                let s = sym3(&raw_add_cand, a, b, c);
                addc_gg = addc_gg.max((s - gg[a]).abs());
            }
        }}
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            addc_semi = addc_semi.max((sym3(&raw_add_cand, a, b, c) - semi.get(a, b, c)).abs());
        }}}
        eprintln!("DIAG   raw+geom×dens-candidate: sym vs GG={addc_gg:.3e}  vs SEMI={addc_semi:.3e}");

        // === Attribute miss[c] to the two pulay geometric channels (C:S_ab vs h0-derivative) ===
        // reconverged density-path of EACH channel separately: which one carries the 7.13e-4?
        let csab_block = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian_parts(s, &params, e).unwrap().0;
        let h0ch_block = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian_parts(s, &params, e).unwrap().1;
        // analytic linearized density-path of each channel: block(P^(c),W^(c),V) + [block(V+V^(c))−block(V)].
        let chan_ana_dpath = |c: usize, block: &dyn Fn(&PeriodicSystem, &ElectronicResult) -> Matrix| -> Matrix {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let h1 = block(&system, &e1);
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
            let h2 = block(&system, &e2);
            let h0b = block(&system, &electronic);
            let mut m = crate::linalg::Matrix::zeros(ndof, ndof);
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] = h1[(a, b)] + h2[(a, b)] - h0b[(a, b)]; } }
            m
        };
        let (mut csab_miss, mut h0_miss) = (0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let csab_recon = recon_dpath(&csab_block, c);
            let csab_ana = chan_ana_dpath(c, &csab_block);
            let h0_recon = recon_dpath(&h0ch_block, c);
            let h0_ana = chan_ana_dpath(c, &h0ch_block);
            for a in 0..ndof { for b in 0..ndof {
                csab_miss = csab_miss.max((csab_recon[(a, b)] - csab_ana[(a, b)]).abs());
                h0_miss = h0_miss.max((h0_recon[(a, b)] - h0_ana[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   pulay miss by channel: C:S_ab miss={csab_miss:.3e}  h0-deriv miss={h0_miss:.3e}  (sum≈7.13e-4)");

        // === PLAN STEP 1: split the C:S_ab channel into [P·2h0], [−P·V], [−2W] sub-blocks and feed each
        // into the SAME recon_dpath (same displaced geometries, same reconverged fields). Which sub-block
        // carries the 6.903e-4? The −2W (EWD/W) sub-block's analytic linearized density-path is exactly
        // −2·W^(c)·d²S_ab (chan_ana_dpath with a W-only block: h2−h0=0 since it ignores V). ===
        let sub_p2h0_block = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian_csab_subparts(s, &params, e).unwrap().0;
        let sub_npv_block  = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian_csab_subparts(s, &params, e).unwrap().1;
        let sub_n2w_block  = |s: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian_csab_subparts(s, &params, e).unwrap().2;
        let (mut miss_p2h0, mut miss_npv, mut miss_n2w) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut recon_p2h0_n, mut recon_npv_n, mut recon_n2w_n) = (0.0_f64, 0.0_f64, 0.0_f64);
        // cross-check: analytic W-only via chan_ana_dpath vs the direct closed form −2·W^(c)·d²S_ab.
        let mut wonly_ana_selfcheck = 0.0_f64;
        for c in 0..ndof {
            let rp = recon_dpath(&sub_p2h0_block, c);
            let ap = chan_ana_dpath(c, &sub_p2h0_block);
            let rn = recon_dpath(&sub_npv_block, c);
            let an = chan_ana_dpath(c, &sub_npv_block);
            let rw = recon_dpath(&sub_n2w_block, c);
            let aw = chan_ana_dpath(c, &sub_n2w_block);
            // closed-form analytic W-only: −2·W^(c)·d²S_ab (build via sub_n2w_block fed W^(c) as w).
            let mut ew = electronic.clone();
            ew.energy_weighted_density = cphf.energy_weighted_density_responses[c].clone();
            let aw_direct = sub_n2w_block(&system, &ew);
            for a in 0..ndof { for b in 0..ndof {
                miss_p2h0 = miss_p2h0.max((rp[(a, b)] - ap[(a, b)]).abs());
                miss_npv = miss_npv.max((rn[(a, b)] - an[(a, b)]).abs());
                miss_n2w = miss_n2w.max((rw[(a, b)] - aw[(a, b)]).abs());
                recon_p2h0_n = recon_p2h0_n.max(rp[(a, b)].abs());
                recon_npv_n = recon_npv_n.max(rn[(a, b)].abs());
                recon_n2w_n = recon_n2w_n.max(rw[(a, b)].abs());
                wonly_ana_selfcheck = wonly_ana_selfcheck.max((aw[(a, b)] - aw_direct[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   STEP1 C:S_ab sub-channel recon miss:  [P·2h0]={miss_p2h0:.3e}  [−P·V]={miss_npv:.3e}  [−2W]={miss_n2w:.3e}");
        eprintln!("DIAG   STEP1 sub-channel |recon| magnitudes: [P·2h0]={recon_p2h0_n:.3e}  [−P·V]={recon_npv_n:.3e}  [−2W]={recon_n2w_n:.3e}  (W-only ana self-check {wonly_ana_selfcheck:.1e})");

        // === PLAN STEP 5 GATE: the complete closed-form CN-response term vs the FULL pulay miss[c]. ===
        // `h0` in fixed_density_pulay_hessian reads a CN cached in `electronic`, so the reconverged path
        // differentiates CN while the base-CN analytic density-path holds it fixed. The exact missing piece
        // (both pulay geometric channels) is fixed_density_pulay_cn_h0_response(cn_grad_c), cn_grad_c[at]=
        // ∂CN_at/∂R_c. If this closes miss[c] (=pulay_recon−pulay_ana) to ~1e-11, the term is fully identified.
        let nat_s5 = system.atoms.len();
        let cn_grad_mat = crate::hessian::cn_gradient_matrix(&system, cutoff).unwrap();
        let cn_resp_slabs: Vec<Matrix> = (0..ndof).map(|c| {
            let cn_grad_c: Vec<f64> = (0..nat_s5).map(|at| cn_grad_mat[at][c]).collect();
            crate::hessian::fixed_density_pulay_cn_h0_response(&system, &params, &electronic, &cn_grad_c).unwrap()
        }).collect();
        let (mut cnresp_err, mut cnresp_norm, mut miss_norm5) = (0.0_f64, 0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let m = miss(c); // full pulay recon − ana density-path miss for slab c (both channels).
            let cand = &cn_resp_slabs[c];
            for a in 0..ndof { for b in 0..ndof {
                miss_norm5 = miss_norm5.max(m[(a, b)].abs());
                cnresp_norm = cnresp_norm.max(cand[(a, b)].abs());
                cnresp_err = cnresp_err.max((m[(a, b)] - cand[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   STEP5 full CN-response vs pulay miss[c]: err={cnresp_err:.3e}  (|miss|={miss_norm5:.3e}, |cand|={cnresp_norm:.3e})");

        // === PLAN STEP 6 pre-check: add the CN-response to `raw` (simulating production wiring into the
        // pulay density-path), symmetrize over the 6 (a,b,c) perms, and compare to the double-gradient FD
        // AND the independent seminumerical. If it CLOSES (not overshoots), there is no double-count. ===
        let raw_plus_cn = |a: usize, b: usize, c: usize| raw(a, b, c) + cn_resp_slabs[c][(a, b)];
        let (mut cn_gg, mut cn_semi) = (0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof {
                cn_gg = cn_gg.max((sym3(&raw_plus_cn, a, b, c) - gg[a]).abs());
            }
        }}
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            cn_semi = cn_semi.max((sym3(&raw_plus_cn, a, b, c) - semi.get(a, b, c)).abs());
        }}}
        eprintln!("DIAG   STEP6 raw+CN-response: sym vs GG={cn_gg:.3e}  vs SEMI={cn_semi:.3e}  (target ~1e-4)");

        // ================= v5 fix (VALIDATED): Pulay's V_c must be the TOTAL dV/dR_c =================
        // The density-path feeds the pulay overlap-coefficient only V_c^rho = E_qq·q_c (density response),
        // but the true reconverged path needs V_c^total = V_c^geo + V_c^rho, where V_c^geo = ∂V/∂R_c|_{q0}
        // (geometric, fixed charge). Missing term = −P0·V_geo_c : S_ab. Gates P1→P4.
        let vgeo_all = crate::cphf::shell_scalar_potential_derivatives(&system, &electronic.basis, &params, &electronic.shell_charges).unwrap();
        // P1: V_geo_c (analytic, fixed q0) vs geometry FD of shell_scc_potential at FIXED charge q0.
        // Build V(R) at fixed q0 = Σ_t γ_st(R)·q0_t via the same kernel the potential uses.
        let vshell_at = |sys: &PeriodicSystem| -> Vec<f64> {
            // V_s(R) = Σ_t γ_st(R)·q0_t at FIXED charge q0 (the reconverged model rebuilt per geometry).
            let model = crate::coulomb::ShellChargeModel::build(sys, &electronic.basis, &params).unwrap();
            let amat = crate::coulomb::effective_coulomb_matrix(sys, &electronic.basis, &model);
            (0..nshell).map(|s| (0..nshell).map(|t| amat[(s, t)] * electronic.shell_charges[t]).sum::<f64>()).collect()
        };
        let mut p1_err = 0.0_f64;
        for c in 0..ndof {
            let vp = vshell_at(&displace(c, 1.0)); let vm = vshell_at(&displace(c, -1.0));
            for s in 0..nshell {
                let fd = (vp[s] - vm[s]) / (2.0 * h);
                p1_err = p1_err.max((vgeo_all[c][s] - fd).abs());
            }
        }
        eprintln!("DIAG   P1 V_geo_c analytic vs FD(shell_scc @ fixed q0): err={p1_err:.3e}");

        // P2: −P0·V_geo_c : S_ab  vs  FD of fixed_density_pulay_hessian with ONLY shell_scc_potential
        // perturbed by ±λ·V_geo_c (geometry & density fixed). = the pulay Hessian's V-channel directional deriv.
        let mut p2_err = 0.0_f64;
        let lam = 1.0e-5;
        for c in 0..ndof {
            // analytic candidate slab: pulay_hessian with shell_scc_potential = V_geo_c, density = P0, W=0
            // ⇒ overlap_coeff = P0·(2h0 − 0) ... NO: we need the V-CHANNEL only: ∂(overlap_coeff)/∂V·V_geo_c
            // = −P0·V_geo_c (per pair, lifted), contracted with the SAME body as pulay_hessian → S_ab term.
            // Cleanest: [pulay_hess(V0+λ·V_geo_c) − pulay_hess(V0−λ·V_geo_c)]/2λ at fixed P0,W0 (this IS the
            // −P0·V_geo_c:S_ab channel, exactly as fixed_density_pulay_hessian lifts V). Analytic vs this FD:
            let mut ep2 = electronic.clone(); let mut em2 = electronic.clone();
            for s in 0..nshell { ep2.shell_scc_potential[s] += lam * vgeo_all[c][s]; em2.shell_scc_potential[s] -= lam * vgeo_all[c][s]; }
            let hp = crate::hessian::fixed_density_pulay_hessian(&system, &params, &ep2).unwrap().hessian;
            let hm = crate::hessian::fixed_density_pulay_hessian(&system, &params, &em2).unwrap().hessian;
            // analytic: the same directional derivative via one pulay_hessian call with shell_scc_potential=V_geo_c
            // and density=P0, energy_weighted=0, then take ONLY the overlap_coeff·d2s part change: it equals
            // −P0·V_geo_c:S_ab because overlap_coeff is LINEAR in V (coefficient −P per pair). Use parts.0 with
            // a coefficient-only eval: set density=P0, ew=0, and V = V_geo_c gives overlap_coeff = P0(2h0 − Vgeo);
            // subtract the h0 term (P0·2h0:contribution) which is captured elsewhere — so use the pure FD as ref
            // and the analytic candidate = −(pulay parts.0 with density=P0,ew=0,V=Vgeo) minus (…,V=0). Do FD-FD:
            for a in 0..ndof { for b in 0..ndof {
                let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * lam);
                // analytic = pulay_density_path_geom_cross's V-only piece is not separable here; compare to the
                // direct −P0·V_geo_c:S_ab via a second FD with only overlap_coeff perturbed → same as fd. So P2
                // measures FD self-consistency (noise floor) as the reference the production term must match.
                p2_err = p2_err.max(fd.abs()).min(1.0e30); // captured magnitude; err computed in P3 vs miss.
            }}
        }
        let _ = p2_err;

        // P3 (the real gate): does adding the V_geo_c channel to the analytic pulay density-path make it
        // reproduce the exact reconverged miss[c]? analytic_total = pulay(P_c,W_c, V0+ (Vgeo_c+Vrho_c)) − ...
        // We compute the pulay density-path with V_total_c and compare its residual-to-miss.
        let pulay_dpath_vtotal = |c: usize| -> Matrix {
            let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
            let v_rho_c: Vec<f64> = (0..nshell).map(|s| (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let v_total_c: Vec<f64> = (0..nshell).map(|s| vgeo_all[c][s] + v_rho_c[s]).collect();
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_total_c[s]; }
            let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
            let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            let mut m = crate::linalg::Matrix::zeros(ndof, ndof);
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)]; } }
            m
        };
        let mut p3_err = 0.0_f64;
        for c in 0..ndof {
            let ana = pulay_dpath_vtotal(c);
            // the TRUE pulay density-path (reconverged) = pulay_ana[c] + miss[c]. So ana should match that.
            let m = miss(c);
            for a in 0..ndof { for b in 0..ndof {
                let truth = pulay_ana[c][(a, b)] + m[(a, b)];
                p3_err = p3_err.max((ana[(a, b)] - truth).abs());
            }}
        }
        eprintln!("DIAG   P3 pulay density-path(V_total) vs reconverged truth: err={p3_err:.3e}  (target ~1e-6)");

        // P3b: is v_c (=dscalar+shell_kernel·q_c) the TRUE reconverged dV/dR_c? And does the pulay density-path
        // with TRUE (dP,dW,dV) reproduce miss? Isolates whether the gap is a WRONG response or a MISSING cross.
        let mut dV_err = 0.0_f64; let mut p3b_err = 0.0_f64;
        for c in 0..ndof {
            let ep = recon(&displace(c, 1.0)); let em = recon(&displace(c, -1.0));
            let dv_true: Vec<f64> = (0..nshell).map(|s| (ep.shell_scc_potential[s] - em.shell_scc_potential[s]) / (2.0 * h)).collect();
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            for s in 0..nshell { dV_err = dV_err.max((dv_true[s] - v_c[s]).abs()); }
            // pulay density-path with TRUE reconverged dP,dW,dV:
            let mut dp = crate::linalg::Matrix::zeros(electronic.density.rows(), electronic.density.cols());
            let mut dw = dp.clone();
            for r in 0..dp.rows() { for k in 0..dp.cols() {
                dp[(r, k)] = (ep.density[(r, k)] - em.density[(r, k)]) / (2.0 * h);
                dw[(r, k)] = (ep.energy_weighted_density[(r, k)] - em.energy_weighted_density[(r, k)]) / (2.0 * h);
            }}
            let mut e1 = electronic.clone(); e1.density = dp; e1.energy_weighted_density = dw;
            let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += dv_true[s]; }
            let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
            let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            let m = miss(c);
            for a in 0..ndof { for b in 0..ndof {
                let ana = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)];
                let truth = pulay_ana[c][(a, b)] + m[(a, b)];
                p3b_err = p3b_err.max((ana - truth).abs());
            }}
        }
        eprintln!("DIAG   P3b |dV_true−v_c|={dV_err:.3e}  pulay-path(TRUE dP,dW,dV) vs reconverged truth: err={p3b_err:.3e}");

        // P3c: coordinator's EXACT closed form miss = −P0·V_geo_c : S_ab (P0=BASE density, NOT P^(c)).
        // Directional derivative of the C:S_ab channel along V_geo_c at base density = −P0·V_geo_c:S_ab.
        let mut p3c_err = 0.0_f64; let mut p3c_norm = 0.0_f64;
        for c in 0..ndof {
            let mut ep = electronic.clone(); let mut em = electronic.clone();
            for s in 0..nshell { ep.shell_scc_potential[s] += lam * vgeo_all[c][s]; em.shell_scc_potential[s] -= lam * vgeo_all[c][s]; }
            let cp = crate::hessian::fixed_density_pulay_hessian_parts(&system, &params, &ep).unwrap().0;
            let cm = crate::hessian::fixed_density_pulay_hessian_parts(&system, &params, &em).unwrap().0;
            let m = miss(c);
            for a in 0..ndof { for b in 0..ndof {
                let cand = (cp[(a, b)] - cm[(a, b)]) / (2.0 * lam); // = −P0·V_geo_c : S_ab
                p3c_norm = p3c_norm.max(cand.abs());
                p3c_err = p3c_err.max((cand - m[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   P3c coordinator closed-form (−P0·V_geo_c:S_ab) |cand|={p3c_norm:.3e}  vs miss[c]: err={p3c_err:.3e}");

        // P0-check: the PRODUCTION density-path already feeds v_c = dscalar + shell_kernel·q_c. Confirm
        // dscalar (hessian::shell_scalar_potential_first_derivatives) == vgeo (cphf::…_derivatives).
        let mut dscalar_vs_vgeo = 0.0_f64;
        for c in 0..ndof { for s in 0..nshell { dscalar_vs_vgeo = dscalar_vs_vgeo.max((dscalar[(s, c)] - vgeo_all[c][s]).abs()); } }
        eprintln!("DIAG   P0-check dscalar(prod) vs vgeo(cphf): err={dscalar_vs_vgeo:.3e}  (if ~0, V_geo ALREADY in prod path)");

        // === DERIVED ORDERED candidate: pulay_density_path_geom_cross_ordered vs miss[c] per channel ===
        // Build per-channel miss split via reconverged FD of the two Hessian parts (C:S_ab vs h0).
        let csab_recon_all: Vec<Matrix> = (0..ndof).map(|c| recon_dpath(&csab_block, c)).collect();
        let csab_ana_all: Vec<Matrix> = (0..ndof).map(|c| chan_ana_dpath(c, &csab_block)).collect();
        let h0_recon_all: Vec<Matrix> = (0..ndof).map(|c| recon_dpath(&h0ch_block, c)).collect();
        let h0_ana_all: Vec<Matrix> = (0..ndof).map(|c| chan_ana_dpath(c, &h0ch_block)).collect();
        let miss_csab = |c: usize| -> Matrix { let mut m = csab_recon_all[c].clone(); for a in 0..ndof { for b in 0..ndof { m[(a, b)] -= csab_ana_all[c][(a, b)]; } } m };
        let miss_h0 = |c: usize| -> Matrix { let mut m = h0_recon_all[c].clone(); for a in 0..ndof { for b in 0..ndof { m[(a, b)] -= h0_ana_all[c][(a, b)]; } } m };
        let (mut cand_csab_err, mut cand_h0_err, mut cand_full_err) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut cc_norm, mut ch_norm) = (0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect(); // V^(c) response part; V_geo handled by geometry deriv
            let cand_cs = crate::hessian::pulay_density_path_geom_cross_ordered(&system, &params, &electronic, p_c, w_c, &v_c, Some(0)).unwrap();
            let cand_h = crate::hessian::pulay_density_path_geom_cross_ordered(&system, &params, &electronic, p_c, w_c, &v_c, Some(1)).unwrap();
            let sub2 = crate::hessian::pulay_density_path_geom_cross_ordered(&system, &params, &electronic, p_c, w_c, &v_c, Some(2)).unwrap(); // C·S_abc only
            let sub3 = crate::hessian::pulay_density_path_geom_cross_ordered(&system, &params, &electronic, p_c, w_c, &v_c, Some(3)).unwrap(); // pc·2h_c·S_ab only
            let mcs = miss_csab(c); let mh = miss_h0(c); let mfull = miss(c);
            let (mut s2n, mut s3n) = (0.0_f64, 0.0_f64);
            for a in 0..ndof { for b in 0..ndof {
                cc_norm = cc_norm.max(cand_cs[c][(a, b)].abs());
                ch_norm = ch_norm.max(cand_h[c][(a, b)].abs());
                s2n = s2n.max(sub2[c][(a, b)].abs()); s3n = s3n.max(sub3[c][(a, b)].abs());
                cand_csab_err = cand_csab_err.max((cand_cs[c][(a, b)] - mcs[(a, b)]).abs());
                cand_h0_err = cand_h0_err.max((cand_h[c][(a, b)] - mh[(a, b)]).abs());
                cand_full_err = cand_full_err.max(((cand_cs[c][(a, b)] + cand_h[c][(a, b)]) - mfull[(a, b)]).abs());
            }}
            if c == 0 {
                let mn = { let mut n=0.0_f64; for a in 0..ndof { for b in 0..ndof { n=n.max(mcs[(a,b)].abs()); }} n };
                eprintln!("DIAG   [c=0] sub-term norms: term2={s2n:.3e}  term3={s3n:.3e}  miss_csab={mn:.3e}");
                // Probe: is miss_csab ∝ term2 (2h0_c·S_ab), term3 (−Vgeo·S_ab), or their sum? projections.
                let (mut n2,mut d2,mut n3,mut d3)=(0.0,0.0,0.0,0.0);
                for a in 0..ndof { for b in 0..ndof {
                    n2 += mcs[(a,b)]*sub2[c][(a,b)]; d2 += sub2[c][(a,b)]*sub2[c][(a,b)];
                    n3 += mcs[(a,b)]*sub3[c][(a,b)]; d3 += sub3[c][(a,b)]*sub3[c][(a,b)];
                }}
                eprintln!("DIAG   [c=0] proj miss_csab onto term2={:.3}  onto term3={:.3}", if d2>0.0 {n2/d2} else {0.0}, if d3>0.0 {n3/d3} else {0.0});
            }
        }
        eprintln!("DIAG   ORDERED cand C:S_ab |{cc_norm:.3e}| vs miss_csab err={cand_csab_err:.3e}  (target ~1e-11)");
        eprintln!("DIAG   ORDERED cand h0    |{ch_norm:.3e}| vs miss_h0    err={cand_h0_err:.3e}  (target ~1e-11)");
        eprintln!("DIAG   ORDERED cand FULL vs miss[c] err={cand_full_err:.3e}");

        // === Is miss[c] a real analytic term or an FD artifact? Vary the recon FD step; a REAL term is
        // stable, an artifact scales with h. Also compare to the MIXED-gradient FD reference (∂²g/∂z∂R,
        // the anti-trap that has no S_ab): if the mixed-gradient path ALSO shows the miss, it's real. ===
        let recon_dpath_h = |block: &dyn Fn(&PeriodicSystem, &ElectronicResult) -> Matrix, c: usize, hh: f64| -> Matrix {
            let disp = |sign: f64| { let (at, ax) = (c / 3, c % 3); let mut s = system.clone(); match ax { 0 => s.atoms[at].position.x += sign * hh, 1 => s.atoms[at].position.y += sign * hh, _ => s.atoms[at].position.z += sign * hh }; s };
            let sp = disp(1.0); let sm = disp(-1.0);
            let ep = recon(&sp); let em = recon(&sm);
            let (rp, rm) = (block(&sp, &ep), block(&sm, &em));
            let (fp, fm) = (block(&sp, &electronic), block(&sm, &electronic));
            let mut m = crate::linalg::Matrix::zeros(ndof, ndof);
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] = (rp[(a, b)] - rm[(a, b)]) / (2.0 * hh) - (fp[(a, b)] - fm[(a, b)]) / (2.0 * hh); } }
            m
        };
        for &hh in &[2.0e-5_f64, 5.0e-5, 1.0e-4, 2.0e-4] {
            let mut n = 0.0_f64;
            for c in 0..ndof {
                let mr = recon_dpath_h(&pulay_block, c, hh);
                let ma = pulay_ana[c].clone();
                for a in 0..ndof { for b in 0..ndof { n = n.max((mr[(a, b)] - ma[(a, b)]).abs()); } }
            }
            eprintln!("DIAG   |miss| (pulay recon−ana) at h={hh:.0e}: {n:.4e}");
        }

        // P4: full total with the V_total pulay density-path swapped in, symmetrized, vs GG & semi.
        let p4_slabs: Vec<Matrix> = (0..ndof).map(pulay_dpath_vtotal).collect();
        let raw_p4 = |a: usize, b: usize, c: usize| raw(a, b, c) - pulay_ana[c][(a, b)] + p4_slabs[c][(a, b)];
        let (mut p4_gg, mut p4_semi) = (0.0_f64, 0.0_f64);
        for b in 0..ndof { for c in b..ndof {
            let gg = gmix(b, c);
            for a in 0..ndof { p4_gg = p4_gg.max((sym3(&raw_p4, a, b, c) - gg[a]).abs()); }
        }}
        for a in 0..ndof { for b in a..ndof { for c in b..ndof {
            p4_semi = p4_semi.max((sym3(&raw_p4, a, b, c) - semi.get(a, b, c)).abs());
        }}}
        eprintln!("DIAG   P4 full total (V_total pulay path): sym vs GG={p4_gg:.3e}  vs SEMI={p4_semi:.3e}  (target ~1e-4)");

        eprintln!("DIAG   SYMMETRIZED-NEW total vs GG-FD={new_vs_gg:.3e}  vs semi={new_vs_semi:.3e}");
        return;
        #[allow(unreachable_code)]
        for &hs in &[5.0e-5_f64, 2.0e-4] {
            let sem = third_derivative_seminumerical_dense(&system, &params, options.clone(), hs).unwrap();
            let mut e = 0.0_f64;
            for a in 0..ndof { for b in a..ndof { for c in b..ndof { e = e.max((store[c][(a, b)] - sem.get(a, b, c)).abs()); }}}
            eprintln!("DIAG   analytic vs semi(h={hs:.0e}): err={e:.3e}");
        }
        // Direct: analytic slab[c][(a,b)] vs central FD of the analytic Hessian along axis c (the EQ
        // gate's methodology — full ndof×ndof per slab, not the canonical-packed seminumerical).
        {
            let hh = 1.0e-4;
            let mut e = 0.0_f64;
            for c in 0..ndof {
                let (atom, ax) = (c / 3, c % 3);
                let mut sp = system.clone(); let mut sm = system.clone();
                match ax { 0 => { sp.atoms[atom].position.x += hh; sm.atoms[atom].position.x -= hh; },
                    1 => { sp.atoms[atom].position.y += hh; sm.atoms[atom].position.y -= hh; },
                    _ => { sp.atoms[atom].position.z += hh; sm.atoms[atom].position.z -= hh; } };
                let hp = crate::hessian::analytic_hessian(&sp, &params, options.clone()).unwrap().hessian;
                let hm = crate::hessian::analytic_hessian(&sm, &params, options.clone()).unwrap().hessian;
                for a in 0..ndof { for b in 0..ndof {
                    let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * hh);
                    e = e.max((store[c][(a, b)] - fd).abs());
                }}
            }
            eprintln!("DIAG   analytic slab vs FD-of-Hessian (full, EQ-gate style): err={e:.3e}");
        }

        // (1) Closed-form response part vs FD of hessian_response (re-solved at ±c).
        let ao_opts = crate::cphf::AoDerivativeOptions { coordination_cutoff: cutoff, include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian };
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(&system, &params, &electronic, ao_opts, crate::cphf::CpxtbOptions::default()).unwrap();
        let resp = closed_form_response_hessian_derivative(&system, &params, &electronic, &cphf, ao_opts, cutoff).unwrap();
        let h = 5.0e-5;
        let mut err_resp = 0.0_f64;
        let mut ref_resp = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone(); let mut sm = system.clone();
            let (dp, dm) = match ax { 0 => (&mut sp.atoms[atom].position.x, &mut sm.atoms[atom].position.x),
                1 => (&mut sp.atoms[atom].position.y, &mut sm.atoms[atom].position.y),
                _ => (&mut sp.atoms[atom].position.z, &mut sm.atoms[atom].position.z) };
            *dp += h; *dm -= h;
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let cp = crate::cphf::solve_nonpbc_cpxtb_hessian_response(&sp, &params, &ep, ao_opts, crate::cphf::CpxtbOptions::default()).unwrap();
            let cm = crate::cphf::solve_nonpbc_cpxtb_hessian_response(&sm, &params, &em, ao_opts, crate::cphf::CpxtbOptions::default()).unwrap();
            for a in 0..ndof { for b in 0..ndof {
                let fd = (cp.hessian_response[(a, b)] - cm.hessian_response[(a, b)]) / (2.0 * h);
                err_resp = err_resp.max((resp[c][(a, b)] - fd).abs());
                ref_resp = ref_resp.max(fd.abs());
            }}
        }
        eprintln!("DIAG D_c(hessian_response) closed-form vs FD: err={err_resp:.3e} (ref {ref_resp:.3e})");

        // (2) Analytic D_c H_frozen (= store − D_c hessian_response) vs FD of H_frozen (re-converged).
        // H_frozen(s) = analytic_hessian(s) − hessian_response(s). FD it at ±c and compare to
        // store − resp (which is the analytic D_c H_frozen).
        let hfrozen_at = |sys: &PeriodicSystem| -> Matrix {
            let e = crate::electronic::run_electronic(sys, &params, eo.clone()).unwrap();
            let full = crate::hessian::analytic_hessian_from_result(sys, &params, Some(e.clone()), options.clone()).unwrap().hessian;
            let cp = crate::cphf::solve_nonpbc_cpxtb_hessian_response(sys, &params, &e, ao_opts, crate::cphf::CpxtbOptions::default()).unwrap();
            let mut hf = full.clone();
            for r in 0..ndof { for cc in 0..ndof { hf[(r, cc)] -= cp.hessian_response[(r, cc)]; } }
            hf
        };
        let mut err_frz = 0.0_f64;
        let mut ref_frz = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone(); let mut sm = system.clone();
            let (dp, dm) = match ax { 0 => (&mut sp.atoms[atom].position.x, &mut sm.atoms[atom].position.x),
                1 => (&mut sp.atoms[atom].position.y, &mut sm.atoms[atom].position.y),
                _ => (&mut sp.atoms[atom].position.z, &mut sm.atoms[atom].position.z) };
            *dp += h; *dm -= h;
            let hfp = hfrozen_at(&sp);
            let hfm = hfrozen_at(&sm);
            for a in 0..ndof { for b in 0..ndof {
                let fd = (hfp[(a, b)] - hfm[(a, b)]) / (2.0 * h);
                let ana_frozen = store[c][(a, b)] - resp[c][(a, b)];
                err_frz = err_frz.max((ana_frozen - fd).abs());
                ref_frz = ref_frz.max(fd.abs());
            }}
        }
        eprintln!("DIAG D_c(H_frozen) analytic vs FD: err={err_frz:.3e} (ref {ref_frz:.3e})");

        // (3) PER-BLOCK geometric FD: each frozen block's analytic 3rd derivative vs the FD of that
        // block's fixed-density Hessian (FIXED reference density on the displaced geometry).
        let displace = |c: usize, sign: f64| -> PeriodicSystem {
            let (atom, ax) = (c / 3, c % 3);
            let mut s = system.clone();
            match ax { 0 => s.atoms[atom].position.x += sign * h, 1 => s.atoms[atom].position.y += sign * h, _ => s.atoms[atom].position.z += sign * h };
            s
        };
        // analytic 3rd-derivative slabs and the FD-Hessian closure per block.
        let scc3 = crate::hessian::fixed_shell_charge_scc_third_derivative(&system, &electronic.basis, &electronic.shell_charges, &params).unwrap();
        let pulay3 = crate::hessian::fixed_density_pulay_third_derivative(&system, &params, &electronic).unwrap();
        let cnh03 = crate::hessian::fixed_density_cn_h0_third_derivative(&system, &params, &electronic, cutoff).unwrap();
        let scalar3 = crate::hessian::fixed_density_scalar_overlap_third_derivative(&system, &params, &electronic).unwrap();
        let block_err = |name: &str, ana: &Vec<Matrix>, hess: &dyn Fn(&PeriodicSystem) -> Matrix| {
            let mut e = 0.0_f64;
            for c in 0..ndof {
                let hp = hess(&displace(c, 1.0));
                let hm = hess(&displace(c, -1.0));
                for a in 0..ndof { for b in 0..ndof {
                    let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * h);
                    e = e.max((ana[c][(a, b)] - fd).abs());
                }}
            }
            eprintln!("DIAG   block {name}: analytic 3rd vs fixed-density FD err={e:.3e}");
        };
        block_err("scc", &scc3, &|s| crate::hessian::fixed_shell_charge_scc_hessian(s, &electronic.basis, &electronic.shell_charges, &params).unwrap().hessian);
        block_err("pulay", &pulay3, &|s| crate::hessian::fixed_density_pulay_hessian(s, &params, &electronic).unwrap().hessian);
        block_err("scalar_overlap", &scalar3, &|s| crate::hessian::fixed_density_scalar_overlap_hessian(s, &params, &electronic).unwrap());
        // cn_h0 block = fixed_density_cn_h0_hessian + fixed_density_cn_h0_pulay_cross_hessian (per its doc).
        block_err("cn_h0(+cross)", &cnh03, &|s| {
            let a = crate::hessian::fixed_density_cn_h0_hessian(s, &params, &electronic, cutoff).unwrap().hessian;
            let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(s, &params, &electronic, cutoff).unwrap();
            let mut m = a; for r in 0..ndof { for cc in 0..ndof { m[(r, cc)] += cr[(r, cc)]; } } m
        });
        // cn_h0 block WITHOUT the cross (to isolate whether the cross is the culprit).
        block_err("cn_h0(no cross)", &cnh03, &|s| crate::hessian::fixed_density_cn_h0_hessian(s, &params, &electronic, cutoff).unwrap().hessian);
        // geo = repulsion + halogen (purely geometric, no density). Analytic via third_derivative_geometric.
        let geo3 = third_derivative_geometric(&system, &params).unwrap().to_dense_slabs();
        block_err("geo(rep+hal)", &geo3, &|s| {
            let r = crate::repulsion::repulsion_energy_gradient_hessian(s, &params).unwrap().hessian;
            let hh = crate::halogen::halogen_energy_gradient_hessian(s, &params).unwrap().hessian;
            let mut m = r; for a in 0..ndof { for b in 0..ndof { m[(a, b)] += hh[(a, b)]; } } m
        });
        // The FULL analytic geometric sum vs the FULL frozen fixed-density Hessian FD.
        let mut opt_frozen = options.clone();
        opt_frozen.include_electronic = false;
        let l_abc_geo = third_derivative_frozen_complete(&system, &params, &electronic, None, cutoff, false).unwrap();
        let mut err_sum = 0.0_f64;
        for c in 0..ndof {
            let hp = crate::hessian::analytic_hessian_from_result(&displace(c, 1.0), &params, Some(electronic.clone()), opt_frozen.clone()).unwrap().hessian;
            let hm = crate::hessian::analytic_hessian_from_result(&displace(c, -1.0), &params, Some(electronic.clone()), opt_frozen.clone()).unwrap().hessian;
            for a in 0..ndof { for b in 0..ndof {
                let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * h);
                let ana = l_abc_geo[c][(a, b)] + scalar3[c][(a, b)];
                err_sum = err_sum.max((ana - fd).abs());
            }}
        }
        eprintln!("DIAG   FULL geo sum (l_abc_geo+scalar3) vs frozen fixed-density FD err={err_sum:.3e}");

        // Reference-geometry check: does the sum of per-block fixed-density Hessians equal
        // analytic_hessian_from_result(include_electronic=false)? If not, I mis-enumerated the blocks.
        let hfrozen_full_ref = crate::hessian::analytic_hessian_from_result(&system, &params, Some(electronic.clone()), opt_frozen.clone()).unwrap().hessian;
        let blocks_sum_ref = {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(&system, &params).unwrap().hessian;
            let hh = crate::halogen::halogen_energy_gradient_hessian(&system, &params).unwrap().hessian;
            let scc = crate::hessian::fixed_shell_charge_scc_hessian(&system, &electronic.basis, &electronic.shell_charges, &params).unwrap().hessian;
            let pl = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
            let cnh = crate::hessian::fixed_density_cn_h0_hessian(&system, &params, &electronic, cutoff).unwrap().hessian;
            let so = crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &electronic).unwrap();
            let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&system, &params, &electronic, cutoff).unwrap();
            for a in 0..ndof { for b in 0..ndof { m[(a, b)] += hh[(a, b)] + scc[(a, b)] + pl[(a, b)] + cnh[(a, b)] + so[(a, b)] + cr[(a, b)]; } }
            m
        };
        let mut ref_block_err = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { ref_block_err = ref_block_err.max((hfrozen_full_ref[(a, b)] - blocks_sum_ref[(a, b)]).abs()); } }
        eprintln!("DIAG   ref: sum-of-blocks vs analytic_hessian(frozen) err={ref_block_err:.3e}");

        // cn_pulay_cross block ALONE: analytic 3rd derivative? third_derivative_frozen_complete folds
        // the cross INTO cnh03. So FD the cross Hessian alone and see its magnitude / whether cnh03
        // − cn_h0_only_3rd matches it.
        let mut cross_fd_mag = 0.0_f64;
        for c in 0..ndof {
            let hp = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&displace(c, 1.0), &params, &electronic, cutoff).unwrap();
            let hm = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&displace(c, -1.0), &params, &electronic, cutoff).unwrap();
            for a in 0..ndof { for b in 0..ndof { cross_fd_mag = cross_fd_mag.max(((hp[(a, b)] - hm[(a, b)]) / (2.0 * h)).abs()); } }
        }
        eprintln!("DIAG   cn_pulay_cross 3rd-deriv FD magnitude = {cross_fd_mag:.3e}");

        // Sum of the INDIVIDUALLY-VALIDATED 3rd-deriv slabs (geo3+scc3+pulay3+cnh03+scalar3) vs the
        // frozen fixed-density FD. If THIS matches but `l_abc_geo+scalar3` doesn't, then
        // `third_derivative_frozen_complete` computes cn_h0 differently than my direct `cnh03`.
        let mut err_myblocks = 0.0_f64;
        let mut err_lvs_my = 0.0_f64;
        for c in 0..ndof {
            let hp = crate::hessian::analytic_hessian_from_result(&displace(c, 1.0), &params, Some(electronic.clone()), opt_frozen.clone()).unwrap().hessian;
            let hm = crate::hessian::analytic_hessian_from_result(&displace(c, -1.0), &params, Some(electronic.clone()), opt_frozen.clone()).unwrap().hessian;
            for a in 0..ndof { for b in 0..ndof {
                let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * h);
                let myblocks = geo3[c][(a, b)] + scc3[c][(a, b)] + pulay3[c][(a, b)] + cnh03[c][(a, b)] + scalar3[c][(a, b)];
                err_myblocks = err_myblocks.max((myblocks - fd).abs());
                err_lvs_my = err_lvs_my.max((l_abc_geo[c][(a, b)] + scalar3[c][(a, b)] - myblocks).abs());
            }}
        }
        eprintln!("DIAG   sum-of-my-3rd-slabs vs frozen FD err={err_myblocks:.3e};  l_abc_geo+scalar3 minus my-slabs={err_lvs_my:.3e}");
        // Hypothesis: l_abc_geo includes an unconditional dispersion 3rd derivative. Print its magnitude.
        let disp3 = crate::dispersion::dispersion_third_derivative(&system, &params, None).unwrap();
        let mut disp_mag = 0.0_f64;
        for a in 0..ndof { for b in 0..ndof { for c in 0..ndof { disp_mag = disp_mag.max(disp3.third[(a * ndof + b) * ndof + c].abs()); } } }
        eprintln!("DIAG   dispersion_third_derivative(None) magnitude = {disp_mag:.3e}");

        // (4) PULAY density-path analytic vs FD, split P-channel vs V-channel.
        let nshell = electronic.shell_charges.len();
        let shell_kernel = crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dscalar = crate::hessian::shell_scalar_potential_first_derivatives(&system, &electronic.basis, &electronic.shell_charges, &params).unwrap();
        let pulay_at = |sys: &PeriodicSystem, e: &ElectronicResult| crate::hessian::fixed_density_pulay_hessian(sys, &params, e).unwrap().hessian;
        let (mut err_pP, mut err_pV, mut err_pTot) = (0.0_f64, 0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            // analytic P-channel = pulay(P^(c), W^(c), V);  V-channel = pulay(V+V^(c)) − pulay(V).
            let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
            let ana_p = pulay_at(&system, &e1);
            let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
            let h2 = pulay_at(&system, &e2);
            let h0p = pulay_at(&system, &electronic);
            // CLEAN directional-derivative FD at FIXED geometry: d/dt pulay(P+t·P^(c), W+t·W^(c), V+t·V^(c))|_0.
            // This is the EXACT density-path the analytic formula must reproduce (no geometry change).
            let t = 1.0e-5;
            let mut e_pp = electronic.clone(); let mut e_mm = electronic.clone();
            for r in 0..electronic.density.rows() { for k in 0..electronic.density.cols() {
                e_pp.density[(r, k)] += t * p_c[(r, k)]; e_mm.density[(r, k)] -= t * p_c[(r, k)];
                e_pp.energy_weighted_density[(r, k)] += t * w_c[(r, k)]; e_mm.energy_weighted_density[(r, k)] -= t * w_c[(r, k)];
            }}
            for s in 0..nshell { e_pp.shell_scc_potential[s] += t * v_c[s]; e_mm.shell_scc_potential[s] -= t * v_c[s]; }
            let dd_p = pulay_at(&system, &e_pp); let dd_m = pulay_at(&system, &e_mm);
            // V-only directional FD (P,W held at ref): isolates the V-channel.
            let mut e_vp = electronic.clone(); let mut e_vm = electronic.clone();
            for s in 0..nshell { e_vp.shell_scc_potential[s] += t * v_c[s]; e_vm.shell_scc_potential[s] -= t * v_c[s]; }
            let dv_p = pulay_at(&system, &e_vp); let dv_m = pulay_at(&system, &e_vm);
            for a in 0..ndof { for b in 0..ndof {
                let fd_path = (dd_p[(a, b)] - dd_m[(a, b)]) / (2.0 * t);
                let ana_tot = ana_p[(a, b)] + h2[(a, b)] - h0p[(a, b)];
                err_pTot = err_pTot.max((ana_tot - fd_path).abs());
                // V-channel: analytic (h2−h0) as a linear-in-V derivative vs the V-only directional FD.
                let fd_v = (dv_p[(a, b)] - dv_m[(a, b)]) / (2.0 * t);
                err_pV = err_pV.max((h2[(a, b)] - h0p[(a, b)] - fd_v).abs());
                // P-channel: analytic ana_p vs (full − V-only) directional FD.
                let fd_p = fd_path - fd_v;
                err_pP = err_pP.max((ana_p[(a, b)] - fd_p).abs());
            }}
        }
        eprintln!("DIAG   PULAY density-path(fixed-geom directional) analytic vs FD err={err_pTot:.3e}  (Pchan err {err_pP:.2e}  Vchan err {err_pV:.2e})");

        // (5) Is the CPHF first-order density response consistent with the true SCF density derivative?
        // FD d(P)/dR_c (re-converged) vs cphf.density_responses[c], and d(W)/dR_c, d(q)/dR_c, d(V)/dR_c.
        let (mut ep_err, mut ew_err, mut eq_err, mut ev_err) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let hd = 5.0e-5;
        for c in 0..ndof {
            let sp = { let (at, ax) = (c/3, c%3); let mut s = system.clone(); match ax {0=>s.atoms[at].position.x+=hd,1=>s.atoms[at].position.y+=hd,_=>s.atoms[at].position.z+=hd}; s };
            let sm = { let (at, ax) = (c/3, c%3); let mut s = system.clone(); match ax {0=>s.atoms[at].position.x-=hd,1=>s.atoms[at].position.y-=hd,_=>s.atoms[at].position.z-=hd}; s };
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let pc_ana = &cphf.density_responses[c];
            let wc_ana = &cphf.energy_weighted_density_responses[c];
            let qc_ana = &cphf.shell_charge_responses[c];
            for r in 0..electronic.density.rows() { for k in 0..electronic.density.cols() {
                ep_err = ep_err.max((pc_ana[(r, k)] - (ep.density[(r, k)] - em.density[(r, k)]) / (2.0 * hd)).abs());
                ew_err = ew_err.max((wc_ana[(r, k)] - (ep.energy_weighted_density[(r, k)] - em.energy_weighted_density[(r, k)]) / (2.0 * hd)).abs());
            }}
            for s in 0..nshell {
                eq_err = eq_err.max((qc_ana[s] - (ep.shell_charges[s] - em.shell_charges[s]) / (2.0 * hd)).abs());
                let vc = dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * qc_ana[t]).sum::<f64>();
                ev_err = ev_err.max((vc - (ep.shell_scc_potential[s] - em.shell_scc_potential[s]) / (2.0 * hd)).abs());
            }
        }
        eprintln!("DIAG   CPHF response vs true SCF deriv: P^(c) err={ep_err:.3e}  W^(c) err={ew_err:.3e}  q^(c) err={eq_err:.3e}  V^(c) err={ev_err:.3e}");

        // (6) DECISIVE: does the full L_abc + L_abx (production) equal D_c H_frozen split as
        // (geometric FD, fixed density) + (directional density-path FD, fixed geometry)?
        // If yes, the analytic is correct and the 6.1e-4 is a FD-reference artifact of `reconv−fixed`.
        let frozen_full = |sys: &PeriodicSystem, e: &ElectronicResult| {
            crate::hessian::analytic_hessian_from_result(sys, &params, Some(e.clone()), opt_frozen.clone()).unwrap().hessian
        };
        let mut err_split = 0.0_f64;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            // Geometric FD (fixed ref density).
            let hgp = frozen_full(&displace(c, 1.0), &electronic);
            let hgm = frozen_full(&displace(c, -1.0), &electronic);
            // Directional density-path FD (fixed geometry, along the response fields).
            let t = 1.0e-5;
            let mut epp = electronic.clone(); let mut emm = electronic.clone();
            for r in 0..electronic.density.rows() { for k in 0..electronic.density.cols() {
                epp.density[(r, k)] += t * p_c[(r, k)]; emm.density[(r, k)] -= t * p_c[(r, k)];
                epp.energy_weighted_density[(r, k)] += t * w_c[(r, k)]; emm.energy_weighted_density[(r, k)] -= t * w_c[(r, k)];
            }}
            for s in 0..nshell {
                epp.shell_charges[s] += t * q_c[s]; emm.shell_charges[s] -= t * q_c[s];
                epp.shell_scc_potential[s] += t * v_c[s]; emm.shell_scc_potential[s] -= t * v_c[s];
            }
            let hdp = frozen_full(&system, &epp); let hdm = frozen_full(&system, &emm);
            // Compare (geo FD + directional density-path FD) to the analytic store slab − response.
            for a in 0..ndof { for b in 0..ndof {
                let split_fd = (hgp[(a, b)] - hgm[(a, b)]) / (2.0 * h) + (hdp[(a, b)] - hdm[(a, b)]) / (2.0 * t);
                let ana_frozen = store[c][(a, b)] - resp[c][(a, b)];
                err_split = err_split.max((ana_frozen - split_fd).abs());
            }}
        }
        eprintln!("DIAG   D_c(H_frozen) analytic vs (geoFD + directional-density-path FD) err={err_split:.3e}");

        // (7) PER-CHANNEL density-path: analytic (frozen_hessian_density_path channel) vs directional FD.
        // scalar_overlap channel: analytic = so(P^(c),q) + so(P,q^(c)); directional FD along (P,q).
        let so_at = |e: &ElectronicResult| crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, e).unwrap();
        let s2_at = |e: &ElectronicResult| crate::hessian::fixed_shell_charge_scc_hessian(&system, &e.basis, &e.shell_charges, &params).unwrap().hessian;
        let cnh_at = |e: &ElectronicResult| {
            let a = crate::hessian::fixed_density_cn_h0_hessian(&system, &params, e, cutoff).unwrap().hessian;
            let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&system, &params, e, cutoff).unwrap();
            let mut m = a; for r in 0..ndof { for cc in 0..ndof { m[(r, cc)] += cr[(r, cc)]; } } m
        };
        let (mut so_err, mut s2_err, mut cnh_err) = (0.0_f64, 0.0_f64, 0.0_f64);
        let t = 1.0e-5;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            // analytic scalar_overlap channel.
            let mut epc = electronic.clone(); epc.density = p_c.clone();
            let mut eqc = electronic.clone(); eqc.shell_charges = q_c.to_vec();
            let so_ana = { let a = so_at(&epc); let b = so_at(&eqc); let mut m = a; for r in 0..ndof { for cc in 0..ndof { m[(r, cc)] += b[(r, cc)]; } } m };
            // directional FD of scalar_overlap along (P, q).
            let mut epp = electronic.clone(); let mut emm = electronic.clone();
            for r in 0..electronic.density.rows() { for k in 0..electronic.density.cols() { epp.density[(r, k)] += t * p_c[(r, k)]; emm.density[(r, k)] -= t * p_c[(r, k)]; }}
            for s in 0..nshell { epp.shell_charges[s] += t * q_c[s]; emm.shell_charges[s] -= t * q_c[s]; }
            let so_fd = { let p = so_at(&epp); let m = so_at(&emm); let mut r = crate::linalg::Matrix::zeros(ndof, ndof); for a in 0..ndof { for b in 0..ndof { r[(a, b)] = (p[(a, b)] - m[(a, b)]) / (2.0 * t); } } r };
            // s2 analytic charge-path vs directional FD along q.
            let s2_ana = crate::hessian::fixed_shell_charge_scc_hessian_charge_path(&system, &electronic.basis, &electronic.shell_charges, q_c, &params).unwrap();
            let s2_fd = { let p = s2_at(&epp); let m = s2_at(&emm); let mut r = crate::linalg::Matrix::zeros(ndof, ndof); for a in 0..ndof { for b in 0..ndof { r[(a, b)] = (p[(a, b)] - m[(a, b)]) / (2.0 * t); } } r };
            // cn_h0+cross analytic (P-channel) vs directional FD along P.
            let cnh_ana = cnh_at(&epc);
            let cnh_fd = { let p = cnh_at(&epp); let m = cnh_at(&emm); let mut r = crate::linalg::Matrix::zeros(ndof, ndof); for a in 0..ndof { for b in 0..ndof { r[(a, b)] = (p[(a, b)] - m[(a, b)]) / (2.0 * t); } } r };
            for a in 0..ndof { for b in 0..ndof {
                so_err = so_err.max((so_ana[(a, b)] - so_fd[(a, b)]).abs());
                s2_err = s2_err.max((s2_ana[(a, b)] - s2_fd[(a, b)]).abs());
                cnh_err = cnh_err.max((cnh_ana[(a, b)] - cnh_fd[(a, b)]).abs());
            }}
        }
        eprintln!("DIAG   channel density-path err: scalar_overlap={so_err:.3e}  s2={s2_err:.3e}  cn_h0+cross={cnh_err:.3e}");

        // (8) UN-SYMMETRIZED total (resp + l_abc_geo + scalar3 + l_abx) vs ∂_c H_ab (FD of full Hessian).
        // If this matches but `store` (symmetrized) doesn't → symmetrization bug. If both 6e-4 → assembly.
        let mut err_unsym = 0.0_f64;
        let hh = 1.0e-4;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let l_abx = frozen_hessian_density_path(&system, &params, &electronic, cutoff, p_c, w_c, q_c, &v_c).unwrap();
            let sp = { let (at, ax) = (c/3, c%3); let mut s = system.clone(); match ax {0=>s.atoms[at].position.x+=hh,1=>s.atoms[at].position.y+=hh,_=>s.atoms[at].position.z+=hh}; s };
            let sm = { let (at, ax) = (c/3, c%3); let mut s = system.clone(); match ax {0=>s.atoms[at].position.x-=hh,1=>s.atoms[at].position.y-=hh,_=>s.atoms[at].position.z-=hh}; s };
            let hp = crate::hessian::analytic_hessian(&sp, &params, options.clone()).unwrap().hessian;
            let hm = crate::hessian::analytic_hessian(&sm, &params, options.clone()).unwrap().hessian;
            for a in 0..ndof { for b in 0..ndof {
                let unsym = resp[c][(a, b)] + l_abc_geo[c][(a, b)] + scalar3[c][(a, b)] + l_abx[(a, b)];
                let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * hh);
                err_unsym = err_unsym.max((unsym - fd).abs());
            }}
        }
        eprintln!("DIAG   UN-SYMMETRIZED total vs ∂_c H_ab (FD): err={err_unsym:.3e}");

        // (9) CLEAN: (l_abc_geo + scalar3 + l_abx) vs the TRUE ∂_c H_frozen (FD of RE-CONVERGED H_frozen).
        let mut err_frozen_clean = 0.0_f64;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
            let l_abx = frozen_hessian_density_path(&system, &params, &electronic, cutoff, p_c, w_c, q_c, &v_c).unwrap();
            let hfp = hfrozen_at(&displace(c, 1.0)); // re-converged frozen (analytic_hessian − hessian_response)
            let hfm = hfrozen_at(&displace(c, -1.0));
            for a in 0..ndof { for b in 0..ndof {
                let ana = l_abc_geo[c][(a, b)] + scalar3[c][(a, b)] + l_abx[(a, b)];
                let fd = (hfp[(a, b)] - hfm[(a, b)]) / (2.0 * h);
                err_frozen_clean = err_frozen_clean.max((ana - fd).abs());
            }}
        }
        eprintln!("DIAG   CLEAN (l_abc+scalar3+l_abx) vs FD(reconv H_frozen): err={err_frozen_clean:.3e}");

        // (10) Per-channel RECONVERGED density-path (reconv−fixed FD) vs the analytic channel density-path.
        // This is the true density-path each channel must reproduce. The mismatched channel is the bug.
        let recon = |sys: &PeriodicSystem| crate::electronic::run_electronic(sys, &params, eo.clone()).unwrap();
        let chan_dpath = |name: &str, ana: &dyn Fn(usize) -> Matrix, block: &dyn Fn(&PeriodicSystem, &ElectronicResult) -> Matrix| {
            let mut e = 0.0_f64;
            for c in 0..ndof {
                let sp = displace(c, 1.0); let sm = displace(c, -1.0);
                let ep = recon(&sp); let em = recon(&sm);
                let rp = block(&sp, &ep); let rm = block(&sm, &em);
                let fp = block(&sp, &electronic); let fm = block(&sm, &electronic);
                let a = ana(c);
                for i in 0..ndof { for j in 0..ndof {
                    let fd_path = (rp[(i, j)] - rm[(i, j)]) / (2.0 * h) - (fp[(i, j)] - fm[(i, j)]) / (2.0 * h);
                    e = e.max((a[(i, j)] - fd_path).abs());
                }}
            }
            eprintln!("DIAG   recon-density-path[{name}]: analytic vs FD err={e:.3e}");
        };
        // s2 analytic charge-path per slab.
        chan_dpath("s2",
            &|c| crate::hessian::fixed_shell_charge_scc_hessian_charge_path(&system, &electronic.basis, &electronic.shell_charges, &cphf.shell_charge_responses[c], &params).unwrap(),
            &|s, e| crate::hessian::fixed_shell_charge_scc_hessian(s, &e.basis, &e.shell_charges, &params).unwrap().hessian);
        // scalar_overlap analytic: so(P^(c),q) + so(P,q^(c)).
        chan_dpath("scalar_overlap",
            &|c| { let mut epc = electronic.clone(); epc.density = cphf.density_responses[c].clone();
                   let mut eqc = electronic.clone(); eqc.shell_charges = cphf.shell_charge_responses[c].clone();
                   let a = crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &epc).unwrap();
                   let b = crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &eqc).unwrap();
                   let mut m = a; for i in 0..ndof { for j in 0..ndof { m[(i, j)] += b[(i, j)]; } } m },
            &|s, e| crate::hessian::fixed_density_scalar_overlap_hessian(s, &params, e).unwrap());
        // pulay analytic density-path: pulay(P^(c),W^(c),V) + [pulay(V+V^(c)) − pulay(V)]  (CURRENT, buggy).
        chan_dpath("pulay(current)",
            &|c| { let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
                   let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
                   let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
                   let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
                   let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
                   let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
                   let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
                   let mut m = h1; for i in 0..ndof { for j in 0..ndof { m[(i, j)] += h2[(i, j)] - h0p[(i, j)]; } } m },
            &|s, e| crate::hessian::fixed_density_pulay_hessian(s, &params, e).unwrap().hessian);
        // Hypothesis test: is the 7e-4 residual the pulay 3rd-derivative at density=P^(c) (geometry×P^(c) cross)?
        {
            let mut e = 0.0_f64;
            for c in 0..ndof {
                let p_c = &cphf.density_responses[c]; let w_c = &cphf.energy_weighted_density_responses[c]; let q_c = &cphf.shell_charge_responses[c];
                let v_c: Vec<f64> = (0..nshell).map(|s| dscalar[(s, c)] + (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum::<f64>()).collect();
                let mut e1 = electronic.clone(); e1.density = p_c.clone(); e1.energy_weighted_density = w_c.clone();
                let h1 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1).unwrap().hessian;
                let mut e2 = electronic.clone(); for s in 0..nshell { e2.shell_scc_potential[s] += v_c[s]; }
                let h2 = crate::hessian::fixed_density_pulay_hessian(&system, &params, &e2).unwrap().hessian;
                let h0p = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic).unwrap().hessian;
                // Candidate correction: pulay 3rd-derivative with density=P^(c) (∂_c kernel × P^(c)).
                let mut epc = electronic.clone(); epc.density = p_c.clone(); epc.energy_weighted_density = w_c.clone();
                let cross = crate::hessian::fixed_density_pulay_third_derivative(&system, &params, &epc).unwrap();
                let sp = displace(c, 1.0); let sm = displace(c, -1.0);
                let ep = recon(&sp); let em = recon(&sm);
                let rp = crate::hessian::fixed_density_pulay_hessian(&sp, &params, &ep).unwrap().hessian;
                let rm = crate::hessian::fixed_density_pulay_hessian(&sm, &params, &em).unwrap().hessian;
                let fp = crate::hessian::fixed_density_pulay_hessian(&sp, &params, &electronic).unwrap().hessian;
                let fm = crate::hessian::fixed_density_pulay_hessian(&sm, &params, &electronic).unwrap().hessian;
                for a in 0..ndof { for b in 0..ndof {
                    let fd_path = (rp[(a, b)] - rm[(a, b)]) / (2.0 * h) - (fp[(a, b)] - fm[(a, b)]) / (2.0 * h);
                    let ana = h1[(a, b)] + h2[(a, b)] - h0p[(a, b)] + cross[c][(a, b)];
                    e = e.max((ana - fd_path).abs());
                }}
            }
            eprintln!("DIAG   pulay(current + pulay_3rd(P^(c)) cross) vs recon FD: err={e:.3e}");
        }
        // cn_h0 + cross analytic density-path: linear in P → block(density=P^(c)).
        chan_dpath("cn_h0+cross",
            &|c| { let mut epc = electronic.clone(); epc.density = cphf.density_responses[c].clone();
                   let a = crate::hessian::fixed_density_cn_h0_hessian(&system, &params, &epc, cutoff).unwrap().hessian;
                   let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(&system, &params, &epc, cutoff).unwrap();
                   let mut m = a; for i in 0..ndof { for j in 0..ndof { m[(i, j)] += cr[(i, j)]; } } m },
            &|s, e| { let a = crate::hessian::fixed_density_cn_h0_hessian(s, &params, e, cutoff).unwrap().hessian;
                      let cr = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(s, &params, e, cutoff).unwrap();
                      let mut m = a; for i in 0..ndof { for j in 0..ndof { m[(i, j)] += cr[(i, j)]; } } m });
    }

    // The Dense / Vector / Block output modes must be mutually consistent: the dense slabs equal the
    // packed store, the vector mode equals `dense.contract_last(v)`, and the block equals the dense
    // sub-tensor. (The streaming `(1/3)(A+B+Bᵀ)` vector formula must reproduce the symmetric contraction.)
    #[test]
    fn closed_form_output_modes_consistent() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: crate::electronic::ElectronicOptions {
                enable_dispersion: false,
                ..crate::electronic::ElectronicOptions::default()
            },
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let dense =
            third_derivative_analytic_dense(&system, &params, options.clone(), cutoff).unwrap();
        let slabs = third_derivative_analytic(&system, &params, options.clone(), cutoff).unwrap();
        let mut e1 = 0.0_f64;
        for c in 0..ndof {
            for a in 0..ndof {
                for b in 0..ndof {
                    e1 = e1.max((slabs[c][(a, b)] - dense.get(a, b, c)).abs());
                }
            }
        }
        assert!(e1 < 1.0e-12, "dense slabs vs packed store: {e1:.3e}");
        let v: Vec<f64> = (0..ndof)
            .map(|i| (((i * 7 + 3) % 5) as f64) - 2.0)
            .collect();
        let kv = third_derivative_analytic_vector(&system, &params, options.clone(), cutoff, &v)
            .unwrap();
        let kref = dense.contract_last(&v);
        let mut e2 = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                e2 = e2.max((kv[(a, b)] - kref[(a, b)]).abs());
            }
        }
        assert!(e2 < 1.0e-9, "vector mode vs dense.contract_last: {e2:.3e}");
        let atoms = [0usize, 2usize];
        let (dofs, bslabs) =
            third_derivative_analytic_block(&system, &params, options.clone(), cutoff, &atoms)
                .unwrap();
        let mut e3 = 0.0_f64;
        for (ci, &c) in dofs.iter().enumerate() {
            for (ai, &a) in dofs.iter().enumerate() {
                for (bi, &b) in dofs.iter().enumerate() {
                    e3 = e3.max((bslabs[ci][(ai, bi)] - dense.get(a, b, c)).abs());
                }
            }
        }
        assert!(e3 < 1.0e-12, "block mode vs dense sub-tensor: {e3:.3e}");
        eprintln!("output modes consistent: slabs={e1:.1e} vector={e2:.1e} block={e3:.1e}");
    }

    // Localization diagnostic for the strict-analytic frozen density-path `L_abx·x_c`: compares
    // `frozen_hessian_density_path` to the EXACT directional FD of the density-dependent frozen blocks
    // along the SAME cphf responses (central FD of a multilinear function is exact). Per-block max diffs
    // pinpoint which block's density-path is wrong.
    #[test]
    fn frozen_density_path_localization() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let electronic = crate::electronic::run_electronic(&system, &params, eo).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let nshell = electronic.shell_charges.len();
        let n = electronic.basis.len();
        // density-dependent frozen blocks summed (pulay + s2 + cn_h0 + cross + scalar_overlap)
        let blocks = |e: &ElectronicResult| -> Matrix {
            let mut h = crate::hessian::fixed_density_pulay_hessian(&system, &params, e)
                .unwrap()
                .hessian;
            let s2 = crate::hessian::fixed_shell_charge_scc_hessian(
                &system,
                &e.basis,
                &e.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let cn = crate::hessian::fixed_density_cn_h0_hessian(&system, &params, e, cutoff)
                .unwrap()
                .hessian;
            let cx = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
                &system, &params, e, cutoff,
            )
            .unwrap();
            let so =
                crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, e).unwrap();
            for r in 0..ndof {
                for c in 0..ndof {
                    h[(r, c)] += s2[(r, c)] + cn[(r, c)] + cx[(r, c)] + so[(r, c)];
                }
            }
            h
        };
        let d = 1.0e-3;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let p_c = &cphf.density_responses[c];
            let w_c = &cphf.energy_weighted_density_responses[c];
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| (0..nshell).map(|t| shell_kernel[(s, t)] * q_c[t]).sum())
                .collect();
            let analytic = super::frozen_hessian_density_path(
                &system,
                &params,
                &electronic,
                cutoff,
                p_c,
                w_c,
                q_c,
                &v_c,
            )
            .unwrap();
            let mut ep = electronic.clone();
            let mut em = electronic.clone();
            for i in 0..n {
                for j in 0..n {
                    ep.density[(i, j)] += d * p_c[(i, j)];
                    em.density[(i, j)] -= d * p_c[(i, j)];
                    ep.energy_weighted_density[(i, j)] += d * w_c[(i, j)];
                    em.energy_weighted_density[(i, j)] -= d * w_c[(i, j)];
                }
            }
            for s in 0..nshell {
                ep.shell_charges[s] += d * q_c[s];
                em.shell_charges[s] -= d * q_c[s];
                ep.shell_scc_potential[s] += d * v_c[s];
                em.shell_scc_potential[s] -= d * v_c[s];
            }
            let hp = blocks(&ep);
            let hm = blocks(&em);
            for a in 0..ndof {
                for b in 0..ndof {
                    let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * d);
                    max_err = max_err.max((analytic[(a, b)] - fd).abs());
                }
            }
        }
        eprintln!("frozen density-path vs directional FD: max_err = {max_err:.3e}");
        assert!(
            max_err < 1.0e-6,
            "frozen_hessian_density_path mismatch: {max_err:.3e}"
        );
    }

    // Response ladder Step 2: the SCC-scalar block of F_bc (`h0_scc_scalar_second_derivative_matrix`)
    // matches the re-converged FD of the ISOLATED SCC contribution to h0_deriv[b]
    // (`h0_scc_scalar_first_derivative_matrix`) — no H0/CN entanglement. V_c = TOTAL dV/dR_c.
    #[test]
    fn f_bc_scc_block_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let nshell = electronic.shell_charges.len();
        let beps = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += beps;
                    sm.atoms[atom].position.x -= beps;
                }
                1 => {
                    sp.atoms[atom].position.y += beps;
                    sm.atoms[atom].position.y -= beps;
                }
                _ => {
                    sp.atoms[atom].position.z += beps;
                    sm.atoms[atom].position.z -= beps;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| {
                    dvdr_q[(s, c)]
                        + (0..nshell)
                            .map(|t| shell_kernel[(s, t)] * q_c[t])
                            .sum::<f64>()
                })
                .collect();
            for b in 0..ndof {
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_c,
                    q_c,
                    b,
                    c,
                )
                .unwrap();
                let fp =
                    crate::hessian::h0_scc_scalar_first_derivative_matrix(&sp, &params, &ep, b)
                        .unwrap();
                let fm =
                    crate::hessian::h0_scc_scalar_first_derivative_matrix(&sm, &params, &em, b)
                        .unwrap();
                for mu in 0..n {
                    for nu in 0..n {
                        let fd = (fp[(mu, nu)] - fm[(mu, nu)]) / (2.0 * beps);
                        max_err = max_err.max((scc[(mu, nu)] - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step2 F_bc SCC block: max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 1.0e-5, "F_bc SCC block mismatch: {max_err:.3e}");
    }

    // Response ladder Step 3+4: the FULL F_bc = H0-bare + CN-block + SCC-scalar matches the re-converged FD
    // of `AoDerivativeMatrices[b].h0_deriv` (CN-H0 ON). Tight SCF; V_c = TOTAL dV/dR_c.
    #[test]
    fn f_bc_full_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let nshell = electronic.shell_charges.len();
        let solve = |sys: &PeriodicSystem, el: &ElectronicResult| {
            crate::cphf::solve_nonpbc_cpxtb_hessian_response(
                sys,
                &params,
                el,
                ao_opts,
                crate::cphf::CpxtbOptions::default(),
            )
            .unwrap()
        };
        let beps = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += beps;
                    sm.atoms[atom].position.x -= beps;
                }
                1 => {
                    sp.atoms[atom].position.y += beps;
                    sm.atoms[atom].position.y -= beps;
                }
                _ => {
                    sp.atoms[atom].position.z += beps;
                    sm.atoms[atom].position.z -= beps;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let cp = solve(&sp, &ep);
            let cm = solve(&sm, &em);
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| {
                    dvdr_q[(s, c)]
                        + (0..nshell)
                            .map(|t| shell_kernel[(s, t)] * q_c[t])
                            .sum::<f64>()
                })
                .collect();
            for b in 0..ndof {
                let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    b,
                    c,
                )
                .unwrap();
                let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    cutoff,
                    b,
                    c,
                )
                .unwrap();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_c,
                    q_c,
                    b,
                    c,
                )
                .unwrap();
                for mu in 0..n {
                    for nu in 0..n {
                        let analytic = h0b[(mu, nu)] + cnb[(mu, nu)] + scc[(mu, nu)];
                        let fd = (cp.derivative_matrices[b].h0_deriv[(mu, nu)]
                            - cm.derivative_matrices[b].h0_deriv[(mu, nu)])
                            / (2.0 * beps);
                        max_err = max_err.max((analytic - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step3 full F_bc: max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 2.0e-5, "full F_bc mismatch: {max_err:.3e}");
    }

    // Step 4: assemble the gauge-aware observable D_c(Cᵀ F_b C), where F_b is the effective skeleton-Fock
    // derivative (= cphf.derivative_matrices[b].h0_deriv) and F_bc its FULL nuclear derivative (the Step-3
    // block ladder h0_bare_second + CN-block + SCC-scalar, validated to ~1e-10 by f_bc_full_matches_fd).
    // Same outer-transform rule as the validated D_c(CᵀS_bC):
    //   D_c(Cᵀ F_b C) = C^(c)ᵀ F_b C + Cᵀ F_bc C + Cᵀ F_b C^(c).
    // FD reference: the AO-basis F_b(±c) (gauge-free) transformed by the SIGN-ALIGNED mos (build_cpxtb_setup
    // align_to=cphf.mos). Tight SCF (1e-11/1e-9) so the gap-amplified MO-rep terms are not SCF-floor-limited.
    #[test]
    fn d_c_fock_mo_derivative_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nshell = electronic.shell_charges.len();
        let f_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].h0_deriv.clone())
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| {
                    dvdr_q[(s, c)]
                        + (0..nshell)
                            .map(|t| shell_kernel[(s, t)] * q_c[t])
                            .sum::<f64>()
                })
                .collect();
            for b in 0..ndof {
                // F_bc = h0_bare_second + CN-block + SCC-scalar (Step-3 block ladder)
                let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    b,
                    c,
                )
                .unwrap();
                let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    cutoff,
                    b,
                    c,
                )
                .unwrap();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_c,
                    q_c,
                    b,
                    c,
                )
                .unwrap();
                let mut f_bc = h0b;
                for i in 0..n {
                    for j in 0..n {
                        f_bc[(i, j)] += cnb[(i, j)] + scc[(i, j)];
                    }
                }
                // analytic D_c(Cᵀ F_b C) = C^(c)ᵀ F_b C + Cᵀ F_bc C + Cᵀ F_b C^(c)
                let t1 = c_analytic[c]
                    .transpose()
                    .matmul(&f_b_ref[b].matmul(&mos).unwrap())
                    .unwrap();
                let t2 = motrans(&f_bc, &mos);
                let t3 = mos
                    .transpose()
                    .matmul(&f_b_ref[b].matmul(&c_analytic[c]).unwrap())
                    .unwrap();
                // FD reference: AO-basis F_b(±) (gauge-free) transformed by the aligned mos
                let ft_p = motrans(&setp.derivative_matrices[b].h0_deriv, &setp.mos);
                let ft_m = motrans(&setm.derivative_matrices[b].h0_deriv, &setm.mos);
                for p in 0..n {
                    for q in 0..n {
                        let analytic = t1[(p, q)] + t2[(p, q)] + t3[(p, q)];
                        let fd = (ft_p[(p, q)] - ft_m[(p, q)]) / (2.0 * h);
                        max_err = max_err.max((analytic - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step4 D_c(CᵀF_bC): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "D_c(Cᵀ F_b C) does not match FD: {max_err:.3e}"
        );
    }

    // Step 5a: the NON-metric part of D_c rhs_b. `cpxtb_rhs_vector` gives rhs0_b[ia] = −(CᵀF_bC)_ia +
    // ε_i·(CᵀS_bC)_ia (occ-virt pairs). Differentiating:
    //   D_c rhs0_b[ia] = −D_c(CᵀF_bC)_ia + ε_i^(c)·(CᵀS_bC)_ia + ε_i·D_c(CᵀS_bC)_ia,
    // built entirely from already-validated observables: D_c(CᵀF_bC) (Step 4), D_c(CᵀS_bC) (Z5), and
    // ε^(c) (diag of the relaxed F̃_c). FD reference = cpxtb_rhs_vector at ±c with the SIGN-ALIGNED mos and
    // the displaced AO derivative matrices (NO metric-SCC — that is Step 5b). Tight SCF.
    #[test]
    fn d_c_rhs_nonmetric_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nshell = electronic.shell_charges.len();
        let f_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].h0_deriv.clone())
            .collect();
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        // ε^(c)[p] = (h0_mo + resp_mo)[p,p] − ε_p·s_tilde[p,p]  (diag of the relaxed F̃_c)
        let eps_c: Vec<Vec<f64>> = (0..ndof)
            .map(|c| {
                let (h0_mo, resp_mo, s_tilde) = &cand[c];
                (0..n)
                    .map(|p| (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)])
                    .collect()
            })
            .collect();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let q_c = &cphf.shell_charge_responses[c];
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| {
                    dvdr_q[(s, c)]
                        + (0..nshell)
                            .map(|t| shell_kernel[(s, t)] * q_c[t])
                            .sum::<f64>()
                })
                .collect();
            for b in 0..ndof {
                // analytic D_c(CᵀF_bC) and D_c(CᵀS_bC) via the validated outer-transform rule
                let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    b,
                    c,
                )
                .unwrap();
                let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    cutoff,
                    b,
                    c,
                )
                .unwrap();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_c,
                    q_c,
                    b,
                    c,
                )
                .unwrap();
                let mut f_bc = h0b;
                for i in 0..n {
                    for j in 0..n {
                        f_bc[(i, j)] += cnb[(i, j)] + scc[(i, j)];
                    }
                }
                let s_bc =
                    crate::cphf::overlap_second_derivative_matrix(&system, &electronic.basis, b, c)
                        .unwrap();
                let df = {
                    let t1 = c_analytic[c]
                        .transpose()
                        .matmul(&f_b_ref[b].matmul(&mos).unwrap())
                        .unwrap();
                    let t2 = motrans(&f_bc, &mos);
                    let t3 = mos
                        .transpose()
                        .matmul(&f_b_ref[b].matmul(&c_analytic[c]).unwrap())
                        .unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                let ds = {
                    let t1 = c_analytic[c]
                        .transpose()
                        .matmul(&s_b_ref[b].matmul(&mos).unwrap())
                        .unwrap();
                    let t2 = motrans(&s_bc, &mos);
                    let t3 = mos
                        .transpose()
                        .matmul(&s_b_ref[b].matmul(&c_analytic[c]).unwrap())
                        .unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                let s_tilde_b = motrans(&s_b_ref[b], &mos);
                // FD reference: non-metric rhs0_b at ±c (aligned mos, displaced AO derivs)
                let rhs0_p = crate::cphf::cpxtb_rhs_vector(
                    &ep.basis,
                    &setp.mos,
                    &occ,
                    &setp.derivative_matrices[b].h0_deriv,
                    &setp.derivative_matrices[b].overlap_deriv,
                    &setp.orbital_energies,
                )
                .unwrap();
                let rhs0_m = crate::cphf::cpxtb_rhs_vector(
                    &em.basis,
                    &setm.mos,
                    &occ,
                    &setm.derivative_matrices[b].h0_deriv,
                    &setm.derivative_matrices[b].overlap_deriv,
                    &setm.orbital_energies,
                )
                .unwrap();
                for (idx, &(i, a)) in space.pairs.iter().enumerate() {
                    let analytic =
                        -df[(i, a)] + eps_c[c][i] * s_tilde_b[(i, a)] + eps[i] * ds[(i, a)];
                    let fd = (rhs0_p[idx] - rhs0_m[idx]) / (2.0 * h);
                    max_err = max_err.max((analytic - fd).abs());
                }
            }
        }
        eprintln!("Step5a D_c rhs0 (non-metric): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "D_c rhs0 (non-metric) mismatch: {max_err:.3e}"
        );
    }

    // Step 5b: the METRIC-SCC part of D_c rhs_b. add_metric_scc_rhs adds m_b[ia] = −(Cᵀ RF_b C)_ia, where
    //   ΔP_b      = C·B_b^P·Cᵀ,  B_b^P[i,j] = −½(n_i+n_j)·S̃_b[i,j]   (occ-occ, add_occupied_metric_density)
    //   q_b       = response_shell_charges_from_density(S, P, ΔP_b, S_b)   (Mulliken, with explicit S_b)
    //   w_b       = K·q_b,   RF_b = scalar_response_fock_matrix(S, w_b) = −½(w_μ+w_ν)·S_μν.
    // Differentiate the whole chain (META-PRINCIPLE: differentiate the exact helpers, both AO-geometric and
    // ground-state/MO-gauge paths). KEY identity: (D_c K)·q = shell_scalar_potential_first_derivatives(q)[:,c].
    // FD reference isolates the metric term: m_b = setp.rhs_vectors[b] − cpxtb_rhs_vector(non-metric). Tight SCF.
    #[test]
    fn d_c_rhs_metric_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let p_mat = electronic.density.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nshell = basis.shells.len();
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        // Mulliken population of a density `dens` with overlap-like `ov`: out[shell(ν)] -= Σ_κ dens[ν,κ]·ov[κ,ν]
        let population = |dens: &crate::linalg::Matrix, ov: &crate::linalg::Matrix| -> Vec<f64> {
            let mut out = vec![0.0_f64; nshell];
            for nu in 0..n {
                let mut acc = 0.0;
                for kappa in 0..n {
                    acc += dens[(nu, kappa)] * ov[(kappa, nu)];
                }
                out[basis.aos[nu].shell_index] -= acc;
            }
            out
        };
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        // ΔP_b = C·B_b^P·Cᵀ for B_b^P[i,j] = −½(n_i+n_j)·S̃_b[i,j] (occ-occ only)
        let bmat_of = |s_tilde_b: &crate::linalg::Matrix| -> crate::linalg::Matrix {
            let mut b = crate::linalg::Matrix::zeros(n, n);
            for i in 0..n {
                if occ[i] <= 1.0e-8 {
                    continue;
                }
                for j in 0..n {
                    if occ[j] <= 1.0e-8 {
                        continue;
                    }
                    b[(i, j)] = -0.5 * (occ[i] + occ[j]) * s_tilde_b[(i, j)];
                }
            }
            b
        };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let cc = &c_analytic[c];
            let s_c = &s_b_ref[c];
            let p_c = &cphf.density_responses[c];
            for b in 0..ndof {
                let s_b = &s_b_ref[b];
                let s_bc =
                    crate::cphf::overlap_second_derivative_matrix(&system, basis, b, c).unwrap();
                // --- reference metric chain ---
                let s_tilde_b = motrans(s_b, &mos);
                let bmat = bmat_of(&s_tilde_b);
                let dp_b = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &bmat).unwrap();
                let q_b = crate::cphf::response_shell_charges_from_density(
                    basis, &s_mat, &p_mat, &dp_b, s_b,
                )
                .unwrap();
                let w_b = kvec(&q_b);
                let rf_b = crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &w_b).unwrap();
                // --- D_c of the chain ---
                // D_c S̃_b = C^(c)ᵀ S_b C + Cᵀ S_bc C + Cᵀ S_b C^(c)
                let ds_tilde_b = {
                    let t1 = cc.transpose().matmul(&s_b.matmul(&mos).unwrap()).unwrap();
                    let t2 = motrans(&s_bc, &mos);
                    let t3 = mos.transpose().matmul(&s_b.matmul(cc).unwrap()).unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                let dbmat = bmat_of(&ds_tilde_b);
                // D_c ΔP_b = C^(c)·B·Cᵀ + C·(D_cB)·Cᵀ + C·B·C^(c)ᵀ
                let d_dp_b = {
                    let a1 = cc.matmul(&bmat.matmul(&mos.transpose()).unwrap()).unwrap();
                    let a2 = mos
                        .matmul(&dbmat.matmul(&mos.transpose()).unwrap())
                        .unwrap();
                    let a3 = mos.matmul(&bmat.matmul(&cc.transpose()).unwrap()).unwrap();
                    let mut m = a1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += a2[(i, j)] + a3[(i, j)];
                        }
                    }
                    m
                };
                // D_c q_b = population(D_cΔP_b, S) + population(ΔP_b, S_c) + population(P^(c), S_b) + population(P, S_bc)
                let dq_b = {
                    let a = population(&d_dp_b, &s_mat);
                    let b2 = population(&dp_b, s_c);
                    let d = population(p_c, s_b);
                    let e = population(&p_mat, &s_bc);
                    (0..nshell)
                        .map(|s| a[s] + b2[s] + d[s] + e[s])
                        .collect::<Vec<f64>>()
                };
                // D_c w_b = (D_cK)·q_b + K·(D_c q_b);  (D_cK)·q_b = shell_scalar_potential_first_derivatives(q_b)[:,c]
                let dk_q = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, &q_b, &params,
                )
                .unwrap();
                let k_dq = kvec(&dq_b);
                let w_bc: Vec<f64> = (0..nshell).map(|s| dk_q[(s, c)] + k_dq[s]).collect();
                // D_c RF_b[μν] = −½(w_bc_μ+w_bc_ν)·S_μν − ½(w_b_μ+w_b_ν)·S_c_μν
                let d_rf_b = {
                    let t1 =
                        crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &w_bc).unwrap();
                    let mut m = t1;
                    for mu in 0..n {
                        let wmu = w_b[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let wnu = w_b[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (wmu + wnu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                // D_c m_b[ia] = −[C^(c)ᵀ RF_b C + Cᵀ D_cRF_b C + Cᵀ RF_b C^(c)]_ia
                let dm_mo = {
                    let t1 = cc.transpose().matmul(&rf_b.matmul(&mos).unwrap()).unwrap();
                    let t2 = motrans(&d_rf_b, &mos);
                    let t3 = mos.transpose().matmul(&rf_b.matmul(cc).unwrap()).unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                // FD reference: metric part m_b = full rhs − non-metric rhs0, at ±c
                let m_p = {
                    let r0 = crate::cphf::cpxtb_rhs_vector(
                        &ep.basis,
                        &setp.mos,
                        &occ,
                        &setp.derivative_matrices[b].h0_deriv,
                        &setp.derivative_matrices[b].overlap_deriv,
                        &setp.orbital_energies,
                    )
                    .unwrap();
                    setp.rhs_vectors[b]
                        .iter()
                        .zip(r0.iter())
                        .map(|(f, z)| f - z)
                        .collect::<Vec<f64>>()
                };
                let m_m = {
                    let r0 = crate::cphf::cpxtb_rhs_vector(
                        &em.basis,
                        &setm.mos,
                        &occ,
                        &setm.derivative_matrices[b].h0_deriv,
                        &setm.derivative_matrices[b].overlap_deriv,
                        &setm.orbital_energies,
                    )
                    .unwrap();
                    setm.rhs_vectors[b]
                        .iter()
                        .zip(r0.iter())
                        .map(|(f, z)| f - z)
                        .collect::<Vec<f64>>()
                };
                for (idx, &(i, a)) in space.pairs.iter().enumerate() {
                    let analytic = -dm_mo[(i, a)];
                    let fd = (m_p[idx] - m_m[idx]) / (2.0 * h);
                    max_err = max_err.max((analytic - fd).abs());
                }
            }
        }
        eprintln!("Step5b D_c rhs metric-SCC: max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "D_c rhs metric-SCC mismatch: {max_err:.3e}"
        );
    }

    // Step 6: the CP operator derivative-ACTION (D_c A)·x_b (x_b FIXED). The CP Jacobian acts as
    //   (A u)[ia] = gap_ia·u_ia + Σ_s q_ia[s]·pot[s],  pot = K·g,  g[s] = Σ_jb q_jb[s]·scale_jb·u_jb,
    // with gap_ia=ε_a−ε_i, scale_jb=½(n_j−n_b) (kt=0 → geometry-independent), q the transition shell charges.
    // With u=x_b held fixed:
    //   (D_c A x_b)[ia] = (ε_a^(c)−ε_i^(c))·x_b[ia] + Σ_s [ (D_c q_ia[s])·pot[s] + q_ia[s]·D_c pot[s] ],
    //   D_c pot = (D_cK)·g + K·D_c g,  D_c g[s] = Σ_jb (D_c q_jb[s])·scale_jb·x_b[jb],
    //   (D_cK)·g = shell_scalar_potential_first_derivatives(g)[:,c],
    //   D_c q_ia[s] = −Σ_{μ∈s}( C^(c)[μa]·SC[μi] + C[μa]·(D_cSC)[μi] + C^(c)[μi]·SC[μa] + C[μi]·(D_cSC)[μa] ),
    //   D_c SC = S_c·C + S·C^(c).
    // FD reference: A at ±c acting on the FIXED reference amplitude — setp/setm.matvec(x_b), aligned gauge. Tight SCF.
    #[test]
    fn d_c_operator_action_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let npair = space.pairs.len();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let q_ref = crate::cphf::transition_shell_charges(basis, &mos, &occ, &s_mat).unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nshell = basis.shells.len();
        let scale: Vec<f64> = space
            .pairs
            .iter()
            .map(|&(i, a)| 0.5 * (occ[i] - occ[a]))
            .collect();
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let sc = s_mat.matmul(&mos).unwrap();
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        let eps_c: Vec<Vec<f64>> = (0..ndof)
            .map(|c| {
                let (h0_mo, resp_mo, s_tilde) = &cand[c];
                (0..n)
                    .map(|p| (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)])
                    .collect()
            })
            .collect();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let cc = &c_analytic[c];
            // D_c SC = S_c·C + S·C^(c)
            let dsc = {
                let a = s_b_ref[c].matmul(&mos).unwrap();
                let b = s_mat.matmul(cc).unwrap();
                let mut m = a;
                for i in 0..n {
                    for j in 0..n {
                        m[(i, j)] += b[(i, j)];
                    }
                }
                m
            };
            // D_c q_ia[s] for every occ-virt pair
            let dq: Vec<Vec<f64>> = space
                .pairs
                .iter()
                .map(|&(i, a)| {
                    let mut q = vec![0.0_f64; nshell];
                    for (shell_idx, shell) in basis.shells.iter().enumerate() {
                        let end = shell.first_ao + shell.nao;
                        for mu in shell.first_ao..end {
                            q[shell_idx] -= cc[(mu, a)] * sc[(mu, i)]
                                + mos[(mu, a)] * dsc[(mu, i)]
                                + cc[(mu, i)] * sc[(mu, a)]
                                + mos[(mu, i)] * dsc[(mu, a)];
                        }
                    }
                    q
                })
                .collect();
            for b in 0..ndof {
                let x_b = &cphf.solutions[b].amplitudes;
                // reference g, pot
                let mut g = vec![0.0_f64; nshell];
                for p in 0..npair {
                    for s in 0..nshell {
                        g[s] += q_ref[p][s] * scale[p] * x_b[p];
                    }
                }
                let pot = kvec(&g);
                // D_c g, D_c pot
                let mut dg = vec![0.0_f64; nshell];
                for p in 0..npair {
                    for s in 0..nshell {
                        dg[s] += dq[p][s] * scale[p] * x_b[p];
                    }
                }
                let dk_g = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, &g, &params,
                )
                .unwrap();
                let k_dg = kvec(&dg);
                let dpot: Vec<f64> = (0..nshell).map(|s| dk_g[(s, c)] + k_dg[s]).collect();
                // analytic (D_c A x_b)[ia]
                let analytic: Vec<f64> = space
                    .pairs
                    .iter()
                    .enumerate()
                    .map(|(p, &(i, a))| {
                        let dgap = eps_c[c][a] - eps_c[c][i];
                        let mut v = dgap * x_b[p];
                        for s in 0..nshell {
                            v += dq[p][s] * pot[s] + q_ref[p][s] * dpot[s];
                        }
                        v
                    })
                    .collect();
                // FD reference: A(±c)·x_b via the displaced operator (matvec), aligned gauge
                let ap = setp.matvec(x_b).unwrap();
                let am = setm.matvec(x_b).unwrap();
                for p in 0..npair {
                    let fd = (ap[p] - am[p]) / (2.0 * h);
                    max_err = max_err.max((analytic[p] - fd).abs());
                }
            }
        }
        eprintln!("Step6 (D_c A)x_b: max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 1.0e-4, "(D_c A)x_b mismatch: {max_err:.3e}");
    }

    // Step 7B (Group B of D_c L_a): the orbital-bundle derivative G_a[(D_c B)·x_b]. The bundle
    //   B x_b = (ΔP_orb, ΔW_orb, Δq_orb):  ΔP_orb = C·coeffP·Cᵀ (coeffP[a,i]=coeffP[i,a]=(n_i−n_a)x_b),
    //   Δq_orb = pop(ΔP_orb,S) (implicit), shell_pot = K·Δq_orb, RF = scalar_response_fock(S,shell_pot),
    //   ΔW_orb = C·coeffW1·Cᵀ (coeffW1[a,i]=(n_i−n_a)ε_i x_b) + C·coeffW2·Cᵀ (coeffW2[i,j]=½(n_i+n_j)(CᵀRF C)_ij).
    // Differentiate w.r.t. R_c at FIXED x_b (C→C^(c), ε→ε^(c), S/K geometric), then apply the REFERENCE
    // gradient functional G_a. FD reference: build the bundle at ±c (displaced electronic, aligned mos, fixed
    // x_b) and apply the REFERENCE-geometry G_a — isolates the bundle variation. Tight SCF.
    #[test]
    fn d_c_orbital_bundle_derivative_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ref_ctx = crate::cphf::ResponseGradientContext::new(
            &system,
            basis,
            &params,
            &electronic,
            cutoff,
            incn,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nshell = basis.shells.len();
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let eps_c: Vec<Vec<f64>> = (0..ndof)
            .map(|c| {
                let (h0_mo, resp_mo, s_tilde) = &cand[c];
                (0..n)
                    .map(|p| (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)])
                    .collect()
            })
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        let population = |dens: &crate::linalg::Matrix, ov: &crate::linalg::Matrix| -> Vec<f64> {
            let mut out = vec![0.0_f64; nshell];
            for nu in 0..n {
                let mut acc = 0.0;
                for kappa in 0..n {
                    acc += dens[(nu, kappa)] * ov[(kappa, nu)];
                }
                out[basis.aos[nu].shell_index] -= acc;
            }
            out
        };
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        // C^(c)·coeff·Cᵀ + C·dcoeff·Cᵀ + C·coeff·C^(c)ᵀ
        let triple = |cc: &crate::linalg::Matrix,
                      coeff: &crate::linalg::Matrix,
                      dcoeff: &crate::linalg::Matrix|
         -> crate::linalg::Matrix {
            let a1 = cc.matmul(&coeff.matmul(&mos.transpose()).unwrap()).unwrap();
            let a2 = mos
                .matmul(&dcoeff.matmul(&mos.transpose()).unwrap())
                .unwrap();
            let a3 = mos.matmul(&coeff.matmul(&cc.transpose()).unwrap()).unwrap();
            let mut m = a1;
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += a2[(i, j)] + a3[(i, j)];
                }
            }
            m
        };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let kp = crate::cphf::response_shell_scc_kernel(&sp, &params, &ep).unwrap();
            let km = crate::cphf::response_shell_scc_kernel(&sm, &params, &em).unwrap();
            let cc = &c_analytic[c];
            let s_c = &s_b_ref[c];
            for b in 0..ndof {
                let x_b = &cphf.solutions[b].amplitudes;
                // reference bundle pieces
                let mut coeff_p = crate::linalg::Matrix::zeros(n, n);
                let mut coeff_w1 = crate::linalg::Matrix::zeros(n, n);
                let mut dcoeff_w1 = crate::linalg::Matrix::zeros(n, n);
                for (pi, &(i, a)) in space.pairs.iter().enumerate() {
                    let w = (occ[i] - occ[a]) * x_b[pi];
                    coeff_p[(a, i)] += w;
                    coeff_p[(i, a)] += w;
                    let w1 = w * eps[i];
                    coeff_w1[(a, i)] += w1;
                    coeff_w1[(i, a)] += w1;
                    let dw1 = w * eps_c[c][i];
                    dcoeff_w1[(a, i)] += dw1;
                    dcoeff_w1[(i, a)] += dw1;
                }
                let dp_orb = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &coeff_p).unwrap();
                let q_orb = population(&dp_orb, &s_mat);
                let sp_vec = kvec(&q_orb);
                let rf = crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &sp_vec).unwrap();
                let rf_mo = motrans(&rf, &mos);
                let mut coeff_w2 = crate::linalg::Matrix::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1.0e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1.0e-8 {
                            continue;
                        }
                        coeff_w2[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo[(i, j)];
                    }
                }
                // --- derivatives ---
                let zero = crate::linalg::Matrix::zeros(n, n);
                let d_dp_orb = triple(cc, &coeff_p, &zero);
                let d_q_orb = {
                    let a = population(&d_dp_orb, &s_mat);
                    let b2 = population(&dp_orb, s_c);
                    (0..nshell).map(|s| a[s] + b2[s]).collect::<Vec<f64>>()
                };
                let dk_q = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, &q_orb, &params,
                )
                .unwrap();
                let k_dq = kvec(&d_q_orb);
                let d_sp_vec: Vec<f64> = (0..nshell).map(|s| dk_q[(s, c)] + k_dq[s]).collect();
                let d_rf = {
                    let t1 =
                        crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &d_sp_vec).unwrap();
                    let mut m = t1;
                    for mu in 0..n {
                        let smu = sp_vec[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let snu = sp_vec[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (smu + snu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                let d_rf_mo = {
                    let t1 = cc.transpose().matmul(&rf.matmul(&mos).unwrap()).unwrap();
                    let t2 = motrans(&d_rf, &mos);
                    let t3 = mos.transpose().matmul(&rf.matmul(cc).unwrap()).unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                let mut dcoeff_w2 = crate::linalg::Matrix::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1.0e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1.0e-8 {
                            continue;
                        }
                        dcoeff_w2[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo[(i, j)];
                    }
                }
                let d_w1 = triple(cc, &coeff_w1, &dcoeff_w1);
                let d_w2 = triple(cc, &coeff_w2, &dcoeff_w2);
                let mut d_w_orb = d_w1;
                for i in 0..n {
                    for j in 0..n {
                        d_w_orb[(i, j)] += d_w2[(i, j)];
                    }
                }
                // analytic Group B = G_a[(D_cΔP, D_cΔP, D_cΔW, D_cΔq)]
                let gb = crate::cphf::response_electronic_gradient(
                    &system,
                    &electronic,
                    &shell_kernel,
                    &ref_ctx,
                    &d_dp_orb,
                    &d_dp_orb,
                    &d_w_orb,
                    &d_q_orb,
                )
                .unwrap();
                // FD reference: REFERENCE G_a applied to displaced bundle (aligned mos, fixed x_b)
                let bundle_p = crate::cphf::orbital_response_bundle_from_amplitudes(
                    &ep.basis,
                    &ep.integrals.overlap,
                    &ep.density,
                    &setp.mos,
                    &occ,
                    &setp.orbital_energies,
                    &space,
                    &kp,
                    x_b,
                )
                .unwrap();
                let bundle_m = crate::cphf::orbital_response_bundle_from_amplitudes(
                    &em.basis,
                    &em.integrals.overlap,
                    &em.density,
                    &setm.mos,
                    &occ,
                    &setm.orbital_energies,
                    &space,
                    &km,
                    x_b,
                )
                .unwrap();
                let gp = crate::cphf::response_electronic_gradient(
                    &system,
                    &electronic,
                    &shell_kernel,
                    &ref_ctx,
                    &bundle_p.density,
                    &bundle_p.density,
                    &bundle_p.weighted,
                    &bundle_p.shell_charges,
                )
                .unwrap();
                let gm = crate::cphf::response_electronic_gradient(
                    &system,
                    &electronic,
                    &shell_kernel,
                    &ref_ctx,
                    &bundle_m.density,
                    &bundle_m.density,
                    &bundle_m.weighted,
                    &bundle_m.shell_charges,
                )
                .unwrap();
                for at in 0..system.atoms.len() {
                    for axc in 0..3 {
                        let an = match axc {
                            0 => gb[at].x,
                            1 => gb[at].y,
                            _ => gb[at].z,
                        };
                        let fp = match axc {
                            0 => gp[at].x,
                            1 => gp[at].y,
                            _ => gp[at].z,
                        };
                        let fm = match axc {
                            0 => gm[at].x,
                            1 => gm[at].y,
                            _ => gm[at].z,
                        };
                        let fd = (fp - fm) / (2.0 * h);
                        max_err = max_err.max((an - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step7B Group B (D_c bundle): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "D_c orbital-bundle mismatch: {max_err:.3e}"
        );
    }

    // Step 7A-pulay (first term of Group A = (D_c G_a)[bundle FIXED]): the Pulay term of the response
    // gradient functional is `pulay[atom_μ] += d_bra·(−2·dw)`, `pulay[atom_ν] += d_ket·(−2·dw)` (dw = the
    // FIXED bundle energy-weighted density). With the bundle held fixed, D_c only hits the overlap-gradient
    // vectors d_bra/d_ket → their nuclear derivatives are the bra/ket overlap SECOND derivatives:
    //   D_c(d_bra)[α] = [atom_c==atom_μ]·h_bra_bra[α][γ] + [atom_c==atom_ν]·h_bra_ket[α][γ],
    //   D_c(d_ket)[α] = [atom_c==atom_ν]·h_ket_ket[α][γ] + [atom_c==atom_μ]·h_bra_ket[γ][α],   γ=axis_c.
    // FD reference: the `.pulay` field of response_electronic_gradient_terms at ±c (displaced context) with
    // the FIXED reference bundle — isolates the functional's geometric derivative (Group A) per term.
    #[test]
    fn d_c_groupa_pulay_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let n = mos.rows();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        // pick a representative fixed bundle per DOF b (reference geometry)
        for b in 0..ndof {
            let x_b = &cphf.solutions[b].amplitudes;
            let ref_bundle = crate::cphf::orbital_response_bundle_from_amplitudes(
                basis,
                &s_mat,
                &electronic.density,
                &mos,
                &occ,
                &eps,
                &space,
                &shell_kernel,
                x_b,
            )
            .unwrap();
            let dw = &ref_bundle.weighted;
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                // analytic D_c of the pulay term, with dw FIXED
                let mut grad = vec![[0.0_f64; 3]; nat];
                for mu in 0..n {
                    let atom_mu = basis.aos[mu].atom_index;
                    let rmu = system.atoms[atom_mu].position;
                    for nu in 0..mu {
                        let atom_nu = basis.aos[nu].atom_index;
                        if atom_mu == atom_nu {
                            continue;
                        }
                        let rnu = system.atoms[atom_nu].position;
                        if (rmu - rnu).norm2() <= 1.0e-18 {
                            continue;
                        }
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let fac = -2.0 * dw[(mu, nu)];
                        for alpha in 0..3 {
                            let mut d_bra = 0.0;
                            let mut d_ket = 0.0;
                            if atom_c == atom_mu {
                                d_bra += pair.h_bra_bra[0][alpha][axis_c];
                                d_ket += pair.h_bra_ket[0][axis_c][alpha];
                            }
                            if atom_c == atom_nu {
                                d_bra += pair.h_bra_ket[0][alpha][axis_c];
                                d_ket += pair.h_ket_ket[0][alpha][axis_c];
                            }
                            grad[atom_mu][alpha] += d_bra * fac;
                            grad[atom_nu][alpha] += d_ket * fac;
                        }
                    }
                }
                // FD reference: displaced functional's .pulay with FIXED bundle
                let mut sp = system.clone();
                let mut sm = system.clone();
                match axis_c {
                    0 => {
                        sp.atoms[atom_c].position.x += h;
                        sm.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        sp.atoms[atom_c].position.y += h;
                        sm.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        sp.atoms[atom_c].position.z += h;
                        sm.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
                let kp = crate::cphf::response_shell_scc_kernel(&sp, &params, &ep).unwrap();
                let km = crate::cphf::response_shell_scc_kernel(&sm, &params, &em).unwrap();
                let ctx_p = crate::cphf::ResponseGradientContext::new(
                    &sp, &ep.basis, &params, &ep, cutoff, incn,
                )
                .unwrap();
                let ctx_m = crate::cphf::ResponseGradientContext::new(
                    &sm, &em.basis, &params, &em, cutoff, incn,
                )
                .unwrap();
                let tp = crate::cphf::response_electronic_gradient_terms(
                    &sp,
                    &ep,
                    &kp,
                    &ctx_p,
                    &ref_bundle.density,
                    &ref_bundle.density,
                    &ref_bundle.weighted,
                    &ref_bundle.shell_charges,
                )
                .unwrap();
                let tm = crate::cphf::response_electronic_gradient_terms(
                    &sm,
                    &em,
                    &km,
                    &ctx_m,
                    &ref_bundle.density,
                    &ref_bundle.density,
                    &ref_bundle.weighted,
                    &ref_bundle.shell_charges,
                )
                .unwrap();
                for at in 0..nat {
                    for alpha in 0..3 {
                        let fp = match alpha {
                            0 => tp.pulay[at].x,
                            1 => tp.pulay[at].y,
                            _ => tp.pulay[at].z,
                        };
                        let fm = match alpha {
                            0 => tm.pulay[at].x,
                            1 => tm.pulay[at].y,
                            _ => tm.pulay[at].z,
                        };
                        let fd = (fp - fm) / (2.0 * h);
                        max_err = max_err.max((grad[at][alpha] - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step7A-pulay (D_c G_a pulay term): max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 1.0e-4, "Group A pulay mismatch: {max_err:.3e}");
    }

    // Step 7A-H0CN (band + polynomial + cn terms of Group A, via F_bc reuse): these three terms together are
    // the H0(no-SCC) band-energy gradient with the FIXED bundle density ΔP, i.e. `G_a^{H0+CN}[ΔP] =
    // ∂_a Tr[ΔP·H0^{noSCC}] = Σ_μν ΔP[μν]·∂H0[μν]/∂R_a`. Holding ΔP fixed, D_c is
    //   D_c(band+poly+cn)[a] = Σ_μν ΔP[μν]·∂²H0[μν]/∂R_a∂R_c = Tr[ΔP·(h0_bare_second(a,c)+cn_block(a,c))],
    // reusing the already-validated F_bc H0+CN blocks (f_bc_full_matches_fd, 6.9e-10). FD reference: the SUM
    // of the .band/.polynomial/.cn fields of response_electronic_gradient_terms at ±c (FIXED bundle).
    #[test]
    fn d_c_groupa_h0cn_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let n = mos.rows();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for b in 0..ndof {
            let x_b = &cphf.solutions[b].amplitudes;
            let ref_bundle = crate::cphf::orbital_response_bundle_from_amplitudes(
                basis,
                &s_mat,
                &electronic.density,
                &mos,
                &occ,
                &eps,
                &space,
                &shell_kernel,
                x_b,
            )
            .unwrap();
            let dp = &ref_bundle.density;
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                // analytic: for each a, Tr[ΔP·(h0_bare_second(a,c)+cn_block(a,c))]
                let mut grad = vec![0.0_f64; ndof];
                for a in 0..ndof {
                    let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        a,
                        c,
                    )
                    .unwrap();
                    let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        cutoff,
                        a,
                        c,
                    )
                    .unwrap();
                    let mut acc = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += dp[(mu, nu)] * (h0b[(mu, nu)] + cnb[(mu, nu)]);
                        }
                    }
                    grad[a] = acc;
                }
                // FD reference: SUM of band+polynomial+cn at ±c with FIXED bundle
                let mut sp = system.clone();
                let mut sm = system.clone();
                match axis_c {
                    0 => {
                        sp.atoms[atom_c].position.x += h;
                        sm.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        sp.atoms[atom_c].position.y += h;
                        sm.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        sp.atoms[atom_c].position.z += h;
                        sm.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
                let kp = crate::cphf::response_shell_scc_kernel(&sp, &params, &ep).unwrap();
                let km = crate::cphf::response_shell_scc_kernel(&sm, &params, &em).unwrap();
                let ctx_p = crate::cphf::ResponseGradientContext::new(
                    &sp, &ep.basis, &params, &ep, cutoff, incn,
                )
                .unwrap();
                let ctx_m = crate::cphf::ResponseGradientContext::new(
                    &sm, &em.basis, &params, &em, cutoff, incn,
                )
                .unwrap();
                let tp = crate::cphf::response_electronic_gradient_terms(
                    &sp,
                    &ep,
                    &kp,
                    &ctx_p,
                    dp,
                    dp,
                    &ref_bundle.weighted,
                    &ref_bundle.shell_charges,
                )
                .unwrap();
                let tm = crate::cphf::response_electronic_gradient_terms(
                    &sm,
                    &em,
                    &km,
                    &ctx_m,
                    dp,
                    dp,
                    &ref_bundle.weighted,
                    &ref_bundle.shell_charges,
                )
                .unwrap();
                for at in 0..nat {
                    let sp_v = tp.band[at] + tp.polynomial[at] + tp.cn[at];
                    let sm_v = tm.band[at] + tm.polynomial[at] + tm.cn[at];
                    for alpha in 0..3 {
                        let fp = match alpha {
                            0 => sp_v.x,
                            1 => sp_v.y,
                            _ => sp_v.z,
                        };
                        let fm = match alpha {
                            0 => sm_v.x,
                            1 => sm_v.y,
                            _ => sm_v.z,
                        };
                        let fd = (fp - fm) / (2.0 * h);
                        max_err = max_err.max((grad[3 * at + alpha] - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step7A-H0CN (band+poly+cn via F_bc reuse): max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 1.0e-4, "Group A H0+CN mismatch: {max_err:.3e}");
    }

    // Step 7A-scc_overlap (5th Group A term): `scc[atom_μ] += d_bra·F`, `scc[atom_ν] += d_ket·F`, with
    //   F = −(dp·scalar_shift + p0·scalar_response),  scalar_shift = V_μ+V_ν (ground shell potentials),
    //   scalar_response = (K·Δq)_μ + (K·Δq)_ν.  dp, Δq FIXED (bundle); p0 = ground density.
    // D_c (dp, Δq fixed; p0→P^(c)):
    //   D_c scc[atom_μ] = D_c(d_bra)·F + d_bra·D_cF,
    //   D_cF = −( dp·(v_c_μ+v_c_ν) + P^(c)[μν]·scalar_response + p0·(dk_dq_μ+dk_dq_ν) ),
    //   v_c = TOTAL dV/dR_c = ∂V/∂R_c|_q + (K·q^(c)),  dk_dq = (D_cK)·Δq = shell_scalar_potential_first_derivatives(Δq)[:,c].
    // FD reference: the .scc_overlap field at ±c with FIXED bundle.
    #[test]
    fn d_c_groupa_scc_overlap_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let p_mat = electronic.density.clone();
        let v_pot = electronic.shell_scc_potential.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let n = mos.rows();
        let nshell = basis.shells.len();
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for b in 0..ndof {
            let x_b = &cphf.solutions[b].amplitudes;
            let ref_bundle = crate::cphf::orbital_response_bundle_from_amplitudes(
                basis,
                &s_mat,
                &p_mat,
                &mos,
                &occ,
                &eps,
                &space,
                &shell_kernel,
                x_b,
            )
            .unwrap();
            let dp = &ref_bundle.density;
            let dq = &ref_bundle.shell_charges;
            let sp_resp = kvec(dq); // scalar_response per shell = K·Δq
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                let q_c = &cphf.shell_charge_responses[c];
                let p_c = &cphf.density_responses[c];
                let v_c: Vec<f64> = (0..nshell)
                    .map(|s| {
                        dvdr_q[(s, c)]
                            + (0..nshell)
                                .map(|t| shell_kernel[(s, t)] * q_c[t])
                                .sum::<f64>()
                    })
                    .collect();
                let dk_dq = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, dq, &params,
                )
                .unwrap();
                let mut grad = vec![[0.0_f64; 3]; nat];
                for mu in 0..n {
                    let atom_mu = basis.aos[mu].atom_index;
                    let shell_mu = basis.aos[mu].shell_index;
                    let rmu = system.atoms[atom_mu].position;
                    for nu in 0..mu {
                        let atom_nu = basis.aos[nu].atom_index;
                        if atom_mu == atom_nu {
                            continue;
                        }
                        let shell_nu = basis.aos[nu].shell_index;
                        let rnu = system.atoms[atom_nu].position;
                        if (rmu - rnu).norm2() <= 1.0e-18 {
                            continue;
                        }
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let scalar_shift = v_pot[shell_mu] + v_pot[shell_nu];
                        let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                        let dp_v = dp[(mu, nu)];
                        let p0_v = p_mat[(mu, nu)];
                        let f = -(dp_v * scalar_shift + p0_v * scalar_response);
                        let dcf = -(dp_v * (v_c[shell_mu] + v_c[shell_nu])
                            + p_c[(mu, nu)] * scalar_response
                            + p0_v * (dk_dq[(shell_mu, c)] + dk_dq[(shell_nu, c)]));
                        let dbra0 = pair.d_bra[0].to_array();
                        let dket0 = pair.d_ket[0].to_array();
                        for alpha in 0..3 {
                            let mut dc_dbra = 0.0;
                            let mut dc_dket = 0.0;
                            if atom_c == atom_mu {
                                dc_dbra += pair.h_bra_bra[0][alpha][axis_c];
                                dc_dket += pair.h_bra_ket[0][axis_c][alpha];
                            }
                            if atom_c == atom_nu {
                                dc_dbra += pair.h_bra_ket[0][alpha][axis_c];
                                dc_dket += pair.h_ket_ket[0][alpha][axis_c];
                            }
                            grad[atom_mu][alpha] += dc_dbra * f + dbra0[alpha] * dcf;
                            grad[atom_nu][alpha] += dc_dket * f + dket0[alpha] * dcf;
                        }
                    }
                }
                // FD reference
                let mut sp = system.clone();
                let mut sm = system.clone();
                match axis_c {
                    0 => {
                        sp.atoms[atom_c].position.x += h;
                        sm.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        sp.atoms[atom_c].position.y += h;
                        sm.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        sp.atoms[atom_c].position.z += h;
                        sm.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
                let kp = crate::cphf::response_shell_scc_kernel(&sp, &params, &ep).unwrap();
                let km = crate::cphf::response_shell_scc_kernel(&sm, &params, &em).unwrap();
                let ctx_p = crate::cphf::ResponseGradientContext::new(
                    &sp, &ep.basis, &params, &ep, cutoff, incn,
                )
                .unwrap();
                let ctx_m = crate::cphf::ResponseGradientContext::new(
                    &sm, &em.basis, &params, &em, cutoff, incn,
                )
                .unwrap();
                let tp = crate::cphf::response_electronic_gradient_terms(
                    &sp,
                    &ep,
                    &kp,
                    &ctx_p,
                    dp,
                    dp,
                    &ref_bundle.weighted,
                    dq,
                )
                .unwrap();
                let tm = crate::cphf::response_electronic_gradient_terms(
                    &sm,
                    &em,
                    &km,
                    &ctx_m,
                    dp,
                    dp,
                    &ref_bundle.weighted,
                    dq,
                )
                .unwrap();
                for at in 0..nat {
                    for alpha in 0..3 {
                        let fp = match alpha {
                            0 => tp.scc_overlap[at].x,
                            1 => tp.scc_overlap[at].y,
                            _ => tp.scc_overlap[at].z,
                        };
                        let fm = match alpha {
                            0 => tm.scc_overlap[at].x,
                            1 => tm.scc_overlap[at].y,
                            _ => tm.scc_overlap[at].z,
                        };
                        let fd = (fp - fm) / (2.0 * h);
                        max_err = max_err.max((grad[at][alpha] - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step7A-scc_overlap (D_c G_a scc term): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "Group A scc_overlap mismatch: {max_err:.3e}"
        );
    }

    // Step 7A-scc_kernel (6th/last Group A term): `scc_kernel[atom_i] += dkernel·(Δq_i q_j + q_i Δq_j)`
    // (shell pairs i>j) = `Σ_{st} (∂γ_st/∂R_a)·Δq_s·q_t` (full, γ symmetric). Δq FIXED, q→q^(c):
    //   D_c scc_kernel[a] = Σ_st Δq_s (∂²γ_st/∂R_a∂R_c) q_t + Σ_st Δq_s (∂γ_st/∂R_a) q_t^(c)
    //                     = Σ_s Δq_s·( d2vdr_q[s][(a,c)] + dvdr_qc[(s,a)] ),
    // reusing the validated shell-scalar-potential 2nd derivs (kernel Hessian × q) and 1st derivs (× q^(c)).
    // FD reference: the .scc_kernel field at ±c with FIXED bundle.
    #[test]
    fn d_c_groupa_scc_kernel_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let nshell = basis.shells.len();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for b in 0..ndof {
            let x_b = &cphf.solutions[b].amplitudes;
            let ref_bundle = crate::cphf::orbital_response_bundle_from_amplitudes(
                basis,
                &s_mat,
                &electronic.density,
                &mos,
                &occ,
                &eps,
                &space,
                &shell_kernel,
                x_b,
            )
            .unwrap();
            let dq = &ref_bundle.shell_charges;
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                let q_c = &cphf.shell_charge_responses[c];
                let dvdr_qc = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, q_c, &params,
                )
                .unwrap();
                let mut grad = vec![0.0_f64; ndof];
                for a in 0..ndof {
                    let mut acc = 0.0;
                    for s in 0..nshell {
                        acc += dq[s] * (d2vdr_q[s][(a, c)] + dvdr_qc[(s, a)]);
                    }
                    grad[a] = acc;
                }
                // FD reference
                let mut sp = system.clone();
                let mut sm = system.clone();
                match axis_c {
                    0 => {
                        sp.atoms[atom_c].position.x += h;
                        sm.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        sp.atoms[atom_c].position.y += h;
                        sm.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        sp.atoms[atom_c].position.z += h;
                        sm.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
                let kp = crate::cphf::response_shell_scc_kernel(&sp, &params, &ep).unwrap();
                let km = crate::cphf::response_shell_scc_kernel(&sm, &params, &em).unwrap();
                let ctx_p = crate::cphf::ResponseGradientContext::new(
                    &sp, &ep.basis, &params, &ep, cutoff, incn,
                )
                .unwrap();
                let ctx_m = crate::cphf::ResponseGradientContext::new(
                    &sm, &em.basis, &params, &em, cutoff, incn,
                )
                .unwrap();
                let tp = crate::cphf::response_electronic_gradient_terms(
                    &sp,
                    &ep,
                    &kp,
                    &ctx_p,
                    &ref_bundle.density,
                    &ref_bundle.density,
                    &ref_bundle.weighted,
                    dq,
                )
                .unwrap();
                let tm = crate::cphf::response_electronic_gradient_terms(
                    &sm,
                    &em,
                    &km,
                    &ctx_m,
                    &ref_bundle.density,
                    &ref_bundle.density,
                    &ref_bundle.weighted,
                    dq,
                )
                .unwrap();
                for at in 0..nat {
                    for alpha in 0..3 {
                        let fp = match alpha {
                            0 => tp.scc_kernel[at].x,
                            1 => tp.scc_kernel[at].y,
                            _ => tp.scc_kernel[at].z,
                        };
                        let fm = match alpha {
                            0 => tm.scc_kernel[at].x,
                            1 => tm.scc_kernel[at].y,
                            _ => tm.scc_kernel[at].z,
                        };
                        let fd = (fp - fm) / (2.0 * h);
                        max_err = max_err.max((grad[3 * at + alpha] - fd).abs());
                    }
                }
            }
        }
        eprintln!("Step7A-scc_kernel (D_c G_a kernel term): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "Group A scc_kernel mismatch: {max_err:.3e}"
        );
    }

    // Step 7 capstone: (D_c L_a)·x_b = Group A [(D_c G_a)[bundle fixed]] + Group B [G_a[(D_c B)·x_b]], the
    // complete product-rule split D_c(G_a[B x_b]) = (D_c G_a)[B x_b] + G_a[D_c(B x_b)]. Group A assembled from
    // its 6 FD-validated terms (band+poly+cn via F_bc reuse + pulay + scc_overlap + scc_kernel); Group B from
    // the bundle derivative. Validated against the TOTAL FD of orbital_sector_response_hessian at ±c with the
    // FIXED reference amplitudes — confirms the split is complete (no missing cross-term). Tight SCF.
    #[test]
    fn d_c_orbital_adjoint_total_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let p_mat = electronic.density.clone();
        let v_pot = electronic.shell_scc_potential.clone();
        let space = crate::cphf::CpxtbSpace::from_occupations(&occ).unwrap();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ref_ctx = crate::cphf::ResponseGradientContext::new(
            &system,
            basis,
            &params,
            &electronic,
            cutoff,
            incn,
        )
        .unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let n = mos.rows();
        let nshell = basis.shells.len();
        let amps: Vec<Vec<f64>> = (0..ndof)
            .map(|b| cphf.solutions[b].amplitudes.clone())
            .collect();
        let eps_c: Vec<Vec<f64>> = (0..ndof)
            .map(|c| {
                let (h0_mo, resp_mo, s_tilde) = &cand[c];
                (0..n)
                    .map(|p| (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)])
                    .collect()
            })
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        let population = |dens: &crate::linalg::Matrix, ov: &crate::linalg::Matrix| -> Vec<f64> {
            let mut out = vec![0.0_f64; nshell];
            for nu in 0..n {
                let mut a = 0.0;
                for k in 0..n {
                    a += dens[(nu, k)] * ov[(k, nu)];
                }
                out[basis.aos[nu].shell_index] -= a;
            }
            out
        };
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        let triple = |cc: &crate::linalg::Matrix,
                      coeff: &crate::linalg::Matrix,
                      dcoeff: &crate::linalg::Matrix|
         -> crate::linalg::Matrix {
            let a1 = cc.matmul(&coeff.matmul(&mos.transpose()).unwrap()).unwrap();
            let a2 = mos
                .matmul(&dcoeff.matmul(&mos.transpose()).unwrap())
                .unwrap();
            let a3 = mos.matmul(&coeff.matmul(&cc.transpose()).unwrap()).unwrap();
            let mut m = a1;
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += a2[(i, j)] + a3[(i, j)];
                }
            }
            m
        };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for b in 0..ndof {
            let x_b = &cphf.solutions[b].amplitudes;
            let ref_bundle = crate::cphf::orbital_response_bundle_from_amplitudes(
                basis,
                &s_mat,
                &p_mat,
                &mos,
                &occ,
                &eps,
                &space,
                &shell_kernel,
                x_b,
            )
            .unwrap();
            let dp_orb = ref_bundle.density.clone();
            let dq_orb = ref_bundle.shell_charges.clone();
            let dw_orb = ref_bundle.weighted.clone();
            let sp_resp = kvec(&dq_orb);
            // reference bundle coeffs for Group B
            let mut coeff_p = crate::linalg::Matrix::zeros(n, n);
            let mut coeff_w1 = crate::linalg::Matrix::zeros(n, n);
            for (pi, &(i, a)) in space.pairs.iter().enumerate() {
                let w = (occ[i] - occ[a]) * x_b[pi];
                coeff_p[(a, i)] += w;
                coeff_p[(i, a)] += w;
                let w1 = w * eps[i];
                coeff_w1[(a, i)] += w1;
                coeff_w1[(i, a)] += w1;
            }
            let rf = crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &sp_resp).unwrap();
            let rf_mo = motrans(&rf, &mos);
            let mut coeff_w2 = crate::linalg::Matrix::zeros(n, n);
            for i in 0..n {
                if occ[i] <= 1e-8 {
                    continue;
                }
                for j in 0..n {
                    if occ[j] <= 1e-8 {
                        continue;
                    }
                    coeff_w2[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo[(i, j)];
                }
            }
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                let cc = &c_analytic[c];
                let s_c = &cphf.derivative_matrices[c].overlap_deriv;
                let q_c = &cphf.shell_charge_responses[c];
                let p_c = &cphf.density_responses[c];
                let v_c: Vec<f64> = (0..nshell)
                    .map(|s| {
                        dvdr_q[(s, c)]
                            + (0..nshell)
                                .map(|t| shell_kernel[(s, t)] * q_c[t])
                                .sum::<f64>()
                    })
                    .collect();
                let dvdr_qc = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, q_c, &params,
                )
                .unwrap();
                let dk_dq = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, &dq_orb, &params,
                )
                .unwrap();
                let mut total = vec![0.0_f64; ndof];
                // --- Group A: H0+CN (band+poly+cn) via F_bc reuse + scc_kernel ---
                for a in 0..ndof {
                    let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        a,
                        c,
                    )
                    .unwrap();
                    let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        cutoff,
                        a,
                        c,
                    )
                    .unwrap();
                    let mut acc = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += dp_orb[(mu, nu)] * (h0b[(mu, nu)] + cnb[(mu, nu)]);
                        }
                    }
                    let mut kern = 0.0;
                    for s in 0..nshell {
                        kern += dq_orb[s] * (d2vdr_q[s][(a, c)] + dvdr_qc[(s, a)]);
                    }
                    total[a] += acc + kern;
                }
                // --- Group A: pulay + scc_overlap (ao_pair loop) ---
                for mu in 0..n {
                    let atom_mu = basis.aos[mu].atom_index;
                    let shell_mu = basis.aos[mu].shell_index;
                    let rmu = system.atoms[atom_mu].position;
                    for nu in 0..mu {
                        let atom_nu = basis.aos[nu].atom_index;
                        if atom_mu == atom_nu {
                            continue;
                        }
                        let shell_nu = basis.aos[nu].shell_index;
                        let rnu = system.atoms[atom_nu].position;
                        if (rmu - rnu).norm2() <= 1e-18 {
                            continue;
                        }
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let dw = dw_orb[(mu, nu)];
                        let scalar_shift = v_pot[shell_mu] + v_pot[shell_nu];
                        let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                        let dp_v = dp_orb[(mu, nu)];
                        let p0_v = p_mat[(mu, nu)];
                        let f_scc = -(dp_v * scalar_shift + p0_v * scalar_response);
                        let dcf = -(dp_v * (v_c[shell_mu] + v_c[shell_nu])
                            + p_c[(mu, nu)] * scalar_response
                            + p0_v * (dk_dq[(shell_mu, c)] + dk_dq[(shell_nu, c)]));
                        let dbra0 = pair.d_bra[0].to_array();
                        let dket0 = pair.d_ket[0].to_array();
                        for alpha in 0..3 {
                            let mut db = 0.0;
                            let mut dk = 0.0;
                            if atom_c == atom_mu {
                                db += pair.h_bra_bra[0][alpha][axis_c];
                                dk += pair.h_bra_ket[0][axis_c][alpha];
                            }
                            if atom_c == atom_nu {
                                db += pair.h_bra_ket[0][alpha][axis_c];
                                dk += pair.h_ket_ket[0][alpha][axis_c];
                            }
                            // pulay: d_bra·(-2 dw); scc_overlap: d_bra·F
                            total[3 * atom_mu + alpha] +=
                                db * (-2.0 * dw) + db * f_scc + dbra0[alpha] * dcf;
                            total[3 * atom_nu + alpha] +=
                                dk * (-2.0 * dw) + dk * f_scc + dket0[alpha] * dcf;
                        }
                    }
                }
                // --- Group B: G_a[(D_c B)·x_b] ---
                let zero = crate::linalg::Matrix::zeros(n, n);
                let d_dp = triple(cc, &coeff_p, &zero);
                let d_q = {
                    let a = population(&d_dp, &s_mat);
                    let b2 = population(&dp_orb, s_c);
                    (0..nshell).map(|s| a[s] + b2[s]).collect::<Vec<f64>>()
                };
                let d_sp: Vec<f64> = (0..nshell).map(|s| dk_dq[(s, c)] + kvec(&d_q)[s]).collect();
                let d_rf = {
                    let t1 =
                        crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &d_sp).unwrap();
                    let mut m = t1;
                    for mu in 0..n {
                        let smu = sp_resp[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let snu = sp_resp[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (smu + snu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                let d_rf_mo = {
                    let t1 = cc.transpose().matmul(&rf.matmul(&mos).unwrap()).unwrap();
                    let t2 = motrans(&d_rf, &mos);
                    let t3 = mos.transpose().matmul(&rf.matmul(cc).unwrap()).unwrap();
                    let mut m = t1;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += t2[(i, j)] + t3[(i, j)];
                        }
                    }
                    m
                };
                let mut dcoeff_w1 = crate::linalg::Matrix::zeros(n, n);
                let mut dcoeff_w2 = crate::linalg::Matrix::zeros(n, n);
                for (pi, &(i, a)) in space.pairs.iter().enumerate() {
                    let dw1 = (occ[i] - occ[a]) * eps_c[c][i] * x_b[pi];
                    dcoeff_w1[(a, i)] += dw1;
                    dcoeff_w1[(i, a)] += dw1;
                }
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        dcoeff_w2[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo[(i, j)];
                    }
                }
                let d_w1 = triple(cc, &coeff_w1, &dcoeff_w1);
                let d_w2 = triple(cc, &coeff_w2, &dcoeff_w2);
                let mut d_w = d_w1;
                for i in 0..n {
                    for j in 0..n {
                        d_w[(i, j)] += d_w2[(i, j)];
                    }
                }
                let gb = crate::cphf::response_electronic_gradient(
                    &system,
                    &electronic,
                    &shell_kernel,
                    &ref_ctx,
                    &d_dp,
                    &d_dp,
                    &d_w,
                    &d_q,
                )
                .unwrap();
                for at in 0..nat {
                    total[3 * at] += gb[at].x;
                    total[3 * at + 1] += gb[at].y;
                    total[3 * at + 2] += gb[at].z;
                }
                // FD reference: total D_c(L_a·x_b) via orbital_sector_response_hessian at ±c (fixed amplitudes)
                let mut spos = system.clone();
                let mut sneg = system.clone();
                match axis_c {
                    0 => {
                        spos.atoms[atom_c].position.x += h;
                        sneg.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        spos.atoms[atom_c].position.y += h;
                        sneg.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        spos.atoms[atom_c].position.z += h;
                        sneg.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&spos, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sneg, &params, eo.clone()).unwrap();
                let setp = crate::cphf::build_cpxtb_setup(&spos, &params, &ep, ao_opts, Some(&mos))
                    .unwrap();
                let setm = crate::cphf::build_cpxtb_setup(&sneg, &params, &em, ao_opts, Some(&mos))
                    .unwrap();
                let rp = crate::cphf::orbital_sector_response_hessian(
                    &spos,
                    &params,
                    &ep,
                    ao_opts,
                    &setp.mos,
                    &setp.orbital_energies,
                    &amps,
                )
                .unwrap();
                let rm = crate::cphf::orbital_sector_response_hessian(
                    &sneg,
                    &params,
                    &em,
                    ao_opts,
                    &setm.mos,
                    &setm.orbital_energies,
                    &amps,
                )
                .unwrap();
                for a in 0..ndof {
                    let fd = (rp[(a, b)] - rm[(a, b)]) / (2.0 * h);
                    max_err = max_err.max((total[a] - fd).abs());
                }
            }
        }
        eprintln!("Step7 capstone (D_c L_a)·x_b: max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "(D_c L_a)·x_b total mismatch: {max_err:.3e}"
        );
    }

    // Step 8a: D_c R_static. R_static_ab = G_a[static_b], static_b the x-INDEPENDENT metric bundle from
    // static_metric_response_sector: ΔP^stat=C·B_b^P·Cᵀ (B_b^P[i,j]=−½(n_i+n_j)S̃_b), Δq^stat=charges(ΔP^stat,S_b)
    // [implicit+explicit S_b], RF_b=scalar_response_fock(S,K·Δq^stat), ΔW^stat = metric_W(F_b,S_b)+metric_W(RF_b,0)
    // with coeffWa[i,j]=½(n_i+n_j)(F̃_b−(ε_i+ε_j)S̃_b), coeffWb[i,j]=½(n_i+n_j)(CᵀRF_b C). D_c R_static =
    // (D_c G_a)[static_b] (Group A reuse, ground path P^(c)/q^(c)/v_c) + G_a[D_c static_b] (Group B). D_c static_b
    // reuses D_c S̃_b (Z5), D_c F̃_b (Step 4), D_c RF_b (Step 5b), ε^(c). FD ref: gauge-invariant
    // static_metric_response_sector at ±c. Tight SCF.
    #[test]
    fn d_c_static_sector_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let incn = eo.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: incn,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let basis = &electronic.basis;
        let mos = cphf.mos.clone();
        let occ = electronic.occupations.clone();
        let eps = cphf.orbital_energies.clone();
        let s_mat = electronic.integrals.overlap.clone();
        let p_mat = electronic.density.clone();
        let v_pot = electronic.shell_scc_potential.clone();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ref_ctx = crate::cphf::ResponseGradientContext::new(
            &system,
            basis,
            &params,
            &electronic,
            cutoff,
            incn,
        )
        .unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let n = mos.rows();
        let nshell = basis.shells.len();
        let f_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].h0_deriv.clone())
            .collect();
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let eps_c: Vec<Vec<f64>> = (0..ndof)
            .map(|c| {
                let (h0_mo, resp_mo, s_tilde) = &cand[c];
                (0..n)
                    .map(|p| (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)])
                    .collect()
            })
            .collect();
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        let population = |dens: &crate::linalg::Matrix, ov: &crate::linalg::Matrix| -> Vec<f64> {
            let mut out = vec![0.0_f64; nshell];
            for nu in 0..n {
                let mut a = 0.0;
                for k in 0..n {
                    a += dens[(nu, k)] * ov[(k, nu)];
                }
                out[basis.aos[nu].shell_index] -= a;
            }
            out
        };
        let kvec = |v: &[f64]| -> Vec<f64> {
            (0..nshell)
                .map(|s| {
                    (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * v[t])
                        .sum::<f64>()
                })
                .collect()
        };
        let triple = |cc: &crate::linalg::Matrix,
                      coeff: &crate::linalg::Matrix,
                      dcoeff: &crate::linalg::Matrix|
         -> crate::linalg::Matrix {
            let a1 = cc.matmul(&coeff.matmul(&mos.transpose()).unwrap()).unwrap();
            let a2 = mos
                .matmul(&dcoeff.matmul(&mos.transpose()).unwrap())
                .unwrap();
            let a3 = mos.matmul(&coeff.matmul(&cc.transpose()).unwrap()).unwrap();
            let mut m = a1;
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += a2[(i, j)] + a3[(i, j)];
                }
            }
            m
        };
        // MO-transform helper for D_c(CᵀMC) = C^(c)ᵀ M C + Cᵀ M_c C + Cᵀ M C^(c)
        let d_motrans = |cc: &crate::linalg::Matrix,
                         m: &crate::linalg::Matrix,
                         m_c: &crate::linalg::Matrix|
         -> crate::linalg::Matrix {
            let t1 = cc.transpose().matmul(&m.matmul(&mos).unwrap()).unwrap();
            let t2 = motrans(m_c, &mos);
            let t3 = mos.transpose().matmul(&m.matmul(cc).unwrap()).unwrap();
            let mut r = t1;
            for i in 0..n {
                for j in 0..n {
                    r[(i, j)] += t2[(i, j)] + t3[(i, j)];
                }
            }
            r
        };
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for b in 0..ndof {
            let s_b = &s_b_ref[b];
            let f_b = &f_b_ref[b];
            let s_tilde_b = motrans(s_b, &mos);
            let f_tilde_b = motrans(f_b, &mos);
            // static bundle
            let mut bmat = crate::linalg::Matrix::zeros(n, n);
            for i in 0..n {
                if occ[i] <= 1e-8 {
                    continue;
                }
                for j in 0..n {
                    if occ[j] <= 1e-8 {
                        continue;
                    }
                    bmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * s_tilde_b[(i, j)];
                }
            }
            let dp_s = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &bmat).unwrap();
            let dq_s =
                crate::cphf::response_shell_charges_from_density(basis, &s_mat, &p_mat, &dp_s, s_b)
                    .unwrap();
            let sp_resp = kvec(&dq_s);
            let rf = crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &sp_resp).unwrap();
            let rf_mo = motrans(&rf, &mos);
            let mut cwa = crate::linalg::Matrix::zeros(n, n);
            let mut cwb = crate::linalg::Matrix::zeros(n, n);
            for i in 0..n {
                if occ[i] <= 1e-8 {
                    continue;
                }
                for j in 0..n {
                    if occ[j] <= 1e-8 {
                        continue;
                    }
                    cwa[(i, j)] = 0.5
                        * (occ[i] + occ[j])
                        * (f_tilde_b[(i, j)] - (eps[i] + eps[j]) * s_tilde_b[(i, j)]);
                    cwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo[(i, j)];
                }
            }
            let dw_s = {
                let a = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwa).unwrap();
                let b2 = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwb).unwrap();
                let mut m = a;
                for i in 0..n {
                    for j in 0..n {
                        m[(i, j)] += b2[(i, j)];
                    }
                }
                m
            };
            for c in 0..ndof {
                let (atom_c, axis_c) = (c / 3, c % 3);
                let cc = &c_analytic[c];
                let s_c = &s_b_ref[c];
                let q_c = &cphf.shell_charge_responses[c];
                let p_c = &cphf.density_responses[c];
                let v_c: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, c)] + kvec(q_c)[s]).collect();
                let dvdr_qc = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, q_c, &params,
                )
                .unwrap();
                let dk_dq = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system, basis, &dq_s, &params,
                )
                .unwrap();
                // F_bc, S_bc
                let h0b = crate::hessian::h0_bare_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    b,
                    c,
                )
                .unwrap();
                let cnb = crate::hessian::h0_cn_block_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    cutoff,
                    b,
                    c,
                )
                .unwrap();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_c,
                    q_c,
                    b,
                    c,
                )
                .unwrap();
                let mut f_bc = h0b.clone();
                for i in 0..n {
                    for j in 0..n {
                        f_bc[(i, j)] += cnb[(i, j)] + scc[(i, j)];
                    }
                }
                let s_bc =
                    crate::cphf::overlap_second_derivative_matrix(&system, basis, b, c).unwrap();
                // D_c S̃_b, D_c F̃_b
                let d_s_tilde = d_motrans(cc, s_b, &s_bc);
                let d_f_tilde = d_motrans(cc, f_b, &f_bc);
                // D_c static bundle
                let mut dbmat = crate::linalg::Matrix::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        dbmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * d_s_tilde[(i, j)];
                    }
                }
                let d_dp_s = triple(cc, &bmat, &dbmat);
                let d_dq_s = {
                    let a = population(&d_dp_s, &s_mat);
                    let b2 = population(&dp_s, s_c);
                    let d = population(p_c, s_b);
                    let e = population(&p_mat, &s_bc);
                    (0..nshell)
                        .map(|s| a[s] + b2[s] + d[s] + e[s])
                        .collect::<Vec<f64>>()
                };
                let d_sp: Vec<f64> = (0..nshell)
                    .map(|s| dk_dq[(s, c)] + kvec(&d_dq_s)[s])
                    .collect();
                let d_rf = {
                    let t1 =
                        crate::cphf::scalar_response_fock_matrix(basis, &s_mat, &d_sp).unwrap();
                    let mut m = t1;
                    for mu in 0..n {
                        let smu = sp_resp[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let snu = sp_resp[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (smu + snu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                let d_rf_mo = d_motrans(cc, &rf, &d_rf);
                let mut dcwa = crate::linalg::Matrix::zeros(n, n);
                let mut dcwb = crate::linalg::Matrix::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        dcwa[(i, j)] = 0.5
                            * (occ[i] + occ[j])
                            * (d_f_tilde[(i, j)]
                                - (eps_c[c][i] + eps_c[c][j]) * s_tilde_b[(i, j)]
                                - (eps[i] + eps[j]) * d_s_tilde[(i, j)]);
                        dcwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo[(i, j)];
                    }
                }
                let d_dw_s = {
                    let a = triple(cc, &cwa, &dcwa);
                    let b2 = triple(cc, &cwb, &dcwb);
                    let mut m = a;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += b2[(i, j)];
                        }
                    }
                    m
                };
                // Group A (static bundle) + Group B
                let mut total = vec![0.0_f64; ndof];
                for a in 0..ndof {
                    let h0a = crate::hessian::h0_bare_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        a,
                        c,
                    )
                    .unwrap();
                    let cna = crate::hessian::h0_cn_block_second_derivative_matrix(
                        &system,
                        &params,
                        &electronic,
                        cutoff,
                        a,
                        c,
                    )
                    .unwrap();
                    let mut acc = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += dp_s[(mu, nu)] * (h0a[(mu, nu)] + cna[(mu, nu)]);
                        }
                    }
                    let mut kern = 0.0;
                    for s in 0..nshell {
                        kern += dq_s[s] * (d2vdr_q[s][(a, c)] + dvdr_qc[(s, a)]);
                    }
                    total[a] += acc + kern;
                }
                for mu in 0..n {
                    let atom_mu = basis.aos[mu].atom_index;
                    let shell_mu = basis.aos[mu].shell_index;
                    let rmu = system.atoms[atom_mu].position;
                    for nu in 0..mu {
                        let atom_nu = basis.aos[nu].atom_index;
                        if atom_mu == atom_nu {
                            continue;
                        }
                        let shell_nu = basis.aos[nu].shell_index;
                        let rnu = system.atoms[atom_nu].position;
                        if (rmu - rnu).norm2() <= 1e-18 {
                            continue;
                        }
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let dw = dw_s[(mu, nu)];
                        let scalar_shift = v_pot[shell_mu] + v_pot[shell_nu];
                        let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                        let dp_v = dp_s[(mu, nu)];
                        let p0_v = p_mat[(mu, nu)];
                        let f_scc = -(dp_v * scalar_shift + p0_v * scalar_response);
                        let dcf = -(dp_v * (v_c[shell_mu] + v_c[shell_nu])
                            + p_c[(mu, nu)] * scalar_response
                            + p0_v * (dk_dq[(shell_mu, c)] + dk_dq[(shell_nu, c)]));
                        let dbra0 = pair.d_bra[0].to_array();
                        let dket0 = pair.d_ket[0].to_array();
                        for alpha in 0..3 {
                            let mut db = 0.0;
                            let mut dk = 0.0;
                            if atom_c == atom_mu {
                                db += pair.h_bra_bra[0][alpha][axis_c];
                                dk += pair.h_bra_ket[0][axis_c][alpha];
                            }
                            if atom_c == atom_nu {
                                db += pair.h_bra_ket[0][alpha][axis_c];
                                dk += pair.h_ket_ket[0][alpha][axis_c];
                            }
                            total[3 * atom_mu + alpha] +=
                                db * (-2.0 * dw) + db * f_scc + dbra0[alpha] * dcf;
                            total[3 * atom_nu + alpha] +=
                                dk * (-2.0 * dw) + dk * f_scc + dket0[alpha] * dcf;
                        }
                    }
                }
                let gb = crate::cphf::response_electronic_gradient(
                    &system,
                    &electronic,
                    &shell_kernel,
                    &ref_ctx,
                    &d_dp_s,
                    &d_dp_s,
                    &d_dw_s,
                    &d_dq_s,
                )
                .unwrap();
                for at in 0..nat {
                    total[3 * at] += gb[at].x;
                    total[3 * at + 1] += gb[at].y;
                    total[3 * at + 2] += gb[at].z;
                }
                // FD reference
                let mut spos = system.clone();
                let mut sneg = system.clone();
                match axis_c {
                    0 => {
                        spos.atoms[atom_c].position.x += h;
                        sneg.atoms[atom_c].position.x -= h;
                    }
                    1 => {
                        spos.atoms[atom_c].position.y += h;
                        sneg.atoms[atom_c].position.y -= h;
                    }
                    _ => {
                        spos.atoms[atom_c].position.z += h;
                        sneg.atoms[atom_c].position.z -= h;
                    }
                }
                let ep = crate::electronic::run_electronic(&spos, &params, eo.clone()).unwrap();
                let em = crate::electronic::run_electronic(&sneg, &params, eo.clone()).unwrap();
                let rp = crate::cphf::static_metric_response_sector(&spos, &params, &ep, ao_opts)
                    .unwrap();
                let rm = crate::cphf::static_metric_response_sector(&sneg, &params, &em, ao_opts)
                    .unwrap();
                for a in 0..ndof {
                    let fd = (rp[(a, b)] - rm[(a, b)]) / (2.0 * h);
                    max_err = max_err.max((total[a] - fd).abs());
                }
            }
        }
        eprintln!("Step8a D_c R_static: max|analytic - FD| = {max_err:.3e}");
        assert!(max_err < 1.0e-4, "D_c R_static mismatch: {max_err:.3e}");
    }

    // Step 8b: the assembled closed-form response derivative D_c(cphf.hessian_response) = D_c R_static +
    // D_c R_orbital must match the FD of cphf.hessian_response (gauge-invariant, re-solved at ±c). Validates
    // the full Z-vector assembly (static + orbital + y_a·[D_c rhs − D_c A x]) in production form.
    #[test]
    fn closed_form_response_matches_hessian_response_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let resp = closed_form_response_hessian_derivative(
            &system,
            &params,
            &electronic,
            &cphf,
            ao_opts,
            cutoff,
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let cp = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
                &sp,
                &params,
                &ep,
                ao_opts,
                crate::cphf::CpxtbOptions::default(),
            )
            .unwrap();
            let cm = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
                &sm,
                &params,
                &em,
                ao_opts,
                crate::cphf::CpxtbOptions::default(),
            )
            .unwrap();
            for a in 0..ndof {
                for b in 0..ndof {
                    let fd =
                        (cp.hessian_response[(a, b)] - cm.hessian_response[(a, b)]) / (2.0 * h);
                    max_err = max_err.max((resp[c][(a, b)] - fd).abs());
                }
            }
        }
        eprintln!("Step8b closed-form D_c(hessian_response): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "closed-form response derivative mismatch: {max_err:.3e}"
        );
    }

    // Bisection: split the frozen derivative into geometric (FD at fixed reference density) and density-path
    // (re-converged FD − geometric), and compare each to the analytic `L_abc` and `L_abx` SEPARATELY to
    // localize the 3e-3 closed-form residual.
    #[test]
    fn frozen_split_bisection() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let cutoff = eo.hamiltonian.coordination_cutoff;
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let nshell = electronic.shell_charges.len();
        let frozen = |sys: &PeriodicSystem, e: &ElectronicResult| -> Matrix {
            let mut h = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let pulay = crate::hessian::fixed_density_pulay_hessian(sys, &params, e)
                .unwrap()
                .hessian;
            let s2 = crate::hessian::fixed_shell_charge_scc_hessian(
                sys,
                &e.basis,
                &e.shell_charges,
                &params,
            )
            .unwrap()
            .hessian;
            let cn = crate::hessian::fixed_density_cn_h0_hessian(sys, &params, e, cutoff)
                .unwrap()
                .hessian;
            let cx =
                crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(sys, &params, e, cutoff)
                    .unwrap();
            let so = crate::hessian::fixed_density_scalar_overlap_hessian(sys, &params, e).unwrap();
            for r in 0..ndof {
                for c in 0..ndof {
                    h[(r, c)] += hal[(r, c)]
                        + pulay[(r, c)]
                        + s2[(r, c)]
                        + cn[(r, c)]
                        + cx[(r, c)]
                        + so[(r, c)];
                }
            }
            h
        };
        let l_abc_geo =
            third_derivative_frozen_complete(&system, &params, &electronic, None, cutoff, false).unwrap();
        let scalar_overlap_3rd = crate::hessian::fixed_density_scalar_overlap_third_derivative(
            &system,
            &params,
            &electronic,
        )
        .unwrap();
        let dscalar = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let beps = 1.0e-4;
        let (mut geo_err, mut dens_err, mut tot_err) = (0.0_f64, 0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += beps;
                    sm.atoms[atom].position.x -= beps;
                }
                1 => {
                    sp.atoms[atom].position.y += beps;
                    sm.atoms[atom].position.y -= beps;
                }
                _ => {
                    sp.atoms[atom].position.z += beps;
                    sm.atoms[atom].position.z -= beps;
                }
            }
            // geometric FD at FIXED reference density:
            let geo_p = frozen(&sp, &electronic);
            let geo_m = frozen(&sm, &electronic);
            // re-converged total FD:
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let tot_p = frozen(&sp, &ep);
            let tot_m = frozen(&sm, &em);
            // analytic pieces:
            let q_c = &cphf.shell_charge_responses[c];
            // TOTAL dV/dR_c = ∂V/∂R_c|_q (geometric) + E_qq·q^(c) (density).
            let v_c: Vec<f64> = (0..nshell)
                .map(|s| {
                    dscalar[(s, c)]
                        + (0..nshell)
                            .map(|t| shell_kernel[(s, t)] * q_c[t])
                            .sum::<f64>()
                })
                .collect();
            let l_abx = super::frozen_hessian_density_path(
                &system,
                &params,
                &electronic,
                cutoff,
                &cphf.density_responses[c],
                &cphf.energy_weighted_density_responses[c],
                q_c,
                &v_c,
            )
            .unwrap();
            for a in 0..ndof {
                for b in 0..ndof {
                    let geo_fd = (geo_p[(a, b)] - geo_m[(a, b)]) / (2.0 * beps);
                    let tot_fd = (tot_p[(a, b)] - tot_m[(a, b)]) / (2.0 * beps);
                    let dens_fd = tot_fd - geo_fd;
                    let l_abc_an = l_abc_geo[c][(a, b)] + scalar_overlap_3rd[c][(a, b)];
                    geo_err = geo_err.max((l_abc_an - geo_fd).abs());
                    dens_err = dens_err.max((l_abx[(a, b)] - dens_fd).abs());
                    tot_err = tot_err.max(((l_abc_an + l_abx[(a, b)]) - tot_fd).abs());
                }
            }
        }
        eprintln!("frozen split bisection: geo_err(L_abc vs geomFD)={geo_err:.3e} dens_err(L_abx vs densFD)={dens_err:.3e} total_err={tot_err:.3e}");
        // Strict-analytic frozen `D_c H_frozen = L_abc + L_abx` matches the re-converged FD to FD precision
        // (V_c carries the TOTAL dV/dR_c). Locks both halves of the split.
        assert!(geo_err < 5.0e-4, "L_abc != geometric FD: {geo_err:.3e}");
        assert!(
            dens_err < 5.0e-4,
            "L_abx != density-path FD: {dens_err:.3e}"
        );
        assert!(
            tot_err < 5.0e-4,
            "analytic frozen != re-converged FD: {tot_err:.3e}"
        );
    }

    // Pinpoint which cphf response input ≠ the true total dρ/dR_c (re-converged FD of the ground-state
    // P/W/q). Diagnostic only.
    #[test]
    fn cphf_responses_vs_reconverged_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let nshell = electronic.shell_charges.len();
        let h = 1.0e-4;
        let (mut ep_p, mut ep_w, mut ep_q) = (0.0_f64, 0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let dp = (ep.density[(i, j)] - em.density[(i, j)]) / (2.0 * h);
                    ep_p = ep_p.max((dp - cphf.density_responses[c][(i, j)]).abs());
                    let dw = (ep.energy_weighted_density[(i, j)]
                        - em.energy_weighted_density[(i, j)])
                        / (2.0 * h);
                    ep_w = ep_w.max((dw - cphf.energy_weighted_density_responses[c][(i, j)]).abs());
                }
            }
            for s in 0..nshell {
                let dq = (ep.shell_charges[s] - em.shell_charges[s]) / (2.0 * h);
                ep_q = ep_q.max((dq - cphf.shell_charge_responses[c][s]).abs());
            }
        }
        eprintln!("cphf response vs reconverged FD: dP={ep_p:.3e} dW={ep_w:.3e} dq={ep_q:.3e}");
        // The cphf first-order responses ARE the true total dρ/dR_c (re-converged) — locks that the
        // frozen density-path's response inputs are correct (the 3e-3 closed-form residual is NOT here).
        assert!(
            ep_p < 1.0e-6,
            "density response != reconverged dP/dR: {ep_p:.3e}"
        );
        assert!(
            ep_w < 1.0e-6,
            "energy-weighted response != reconverged dW/dR: {ep_w:.3e}"
        );
        assert!(
            ep_q < 1.0e-6,
            "shell-charge response != reconverged dq/dR: {ep_q:.3e}"
        );
    }

    // Regression for the root cause of the old ~4.8e-3 third-derivative "floor": the bridge differentiates
    // its fixed-density Hessian reconstruction PLUS `cphf.hessian_response`, which MUST equal the reference
    // `analytic_hessian` EXACTLY — otherwise the third derivative inherits the derivative of the Hessian
    // mismatch. The bug was a MISSING block (`fixed_density_scalar_overlap_hessian`, the SCC-scalar×overlap
    // coupling). This test locks the full block set so a future missing/extra Hessian block can't silently
    // reintroduce the floor.
    #[test]
    fn bridge_frozen_reconstruction_equals_analytic_hessian() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: crate::electronic::ElectronicOptions {
                enable_dispersion: false,
                ..crate::electronic::ElectronicOptions::default()
            },
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let electronic =
            crate::electronic::run_electronic(&system, &params, options.electronic_options.clone())
                .unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            crate::cphf::AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.electronic_options.hamiltonian.enable_cn_hamiltonian,
            },
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        // The SAME fixed-density block set the bridge's `frozen_hess_at` uses, + scalar-overlap + response.
        let rep = crate::repulsion::repulsion_energy_gradient_hessian(&system, &params)
            .unwrap()
            .hessian;
        let hal = crate::halogen::halogen_energy_gradient_hessian(&system, &params)
            .unwrap()
            .hessian;
        let scc = crate::hessian::fixed_shell_charge_scc_hessian(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap()
        .hessian;
        let pulay = crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic)
            .unwrap()
            .hessian;
        let cnh0 =
            crate::hessian::fixed_density_cn_h0_hessian(&system, &params, &electronic, cutoff)
                .unwrap()
                .hessian;
        let cross = crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
            &system,
            &params,
            &electronic,
            cutoff,
        )
        .unwrap();
        let scalar_overlap =
            crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &electronic)
                .unwrap();
        let h_analytic = crate::hessian::analytic_hessian(&system, &params, options.clone())
            .unwrap()
            .hessian;
        let mut gap = 0.0_f64;
        for r in 0..ndof {
            for c in 0..ndof {
                let recon = rep[(r, c)]
                    + hal[(r, c)]
                    + scc[(r, c)]
                    + pulay[(r, c)]
                    + cnh0[(r, c)]
                    + cross[(r, c)]
                    + scalar_overlap[(r, c)]
                    + cphf.hessian_response[(r, c)];
                gap = gap.max((recon - h_analytic[(r, c)]).abs());
            }
        }
        eprintln!("bridge frozen+response reconstruction vs analytic_hessian: max gap = {gap:.3e}");
        assert!(
            gap < 1.0e-9,
            "bridge Hessian reconstruction != analytic_hessian (missing/extra block?): {gap:.3e}"
        );
    }

    // Stage Z0–Z2: sector decomposition of the response Hessian R^code = G[full] = R_static + R_orbital,
    // plus the density-gradient adjoint L_a = B^T G_a^* by basis-vector projection.
    //   * LINEARITY (machine precision): R^code == R_static + R_orbital. Proves the static sector
    //     (`static_metric_response_sector`) and the b-independent orbital bundle exactly partition G[full].
    //   * ADJOINT (Stage-Z2 decisive): dot(L_a, x_b) == R_orbital_ab. Proves L_a (built by projecting
    //     u ↦ G_a[orbital_bundle(u)] onto CP unit vectors) is the correct adjoint — the Z-vector RHS.
    //   * INTERCHANGE != 0: R_orbital ≠ −rhs·x (~0.125). Records that L_a != −rhs_a, which is WHY the
    //     Z-vector route is required (NOT a failure of 2n+1). Locks all three so the route can't regress.
    #[test]
    fn response_hessian_sector_decomposition() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let diag = crate::cphf::response_hessian_sector_diagnostic(
            &system,
            &params,
            &electronic,
            ao_opts,
            &cphf,
        )
        .unwrap();
        eprintln!(
            "sector: linearity(R^code vs R_static+R_orbital) = {:.3e} | adjoint(L·x vs R_orbital) = {:.3e} | interchange(R_orbital vs -rhs·x) = {:.3e}",
            diag.linearity_max, diag.adjoint_max, diag.interchange_max
        );
        // The static sector + orbital bundle exactly partition the response Hessian (solid identity).
        assert!(
            diag.linearity_max < 1.0e-10,
            "R^code != R_static + R_orbital: {:.3e}",
            diag.linearity_max
        );
        // Stage-Z2 decisive: the projected adjoint reproduces the orbital-sector Hessian.
        assert!(
            diag.adjoint_max < 1.0e-10,
            "L_a is not the orbital-bundle adjoint (dot(L,x) != R_orbital): {:.3e}",
            diag.adjoint_max
        );
        // L_a != -rhs_a (records WHY the Z-vector route is needed; just a lock that this is not ~0).
        assert!(
            diag.interchange_max > 1.0e-3,
            "interchange unexpectedly ~0 (L_a would equal -rhs_a): {:.3e}",
            diag.interchange_max
        );
    }

    // Stage Z3: the Z-vector solve `A y_a = L_a` converges (residual ~0), and the solution is genuinely
    // different from -x_a — i.e. L_a != -rhs_a, so y_a != -A⁻¹ rhs_a = -x_a. Locks that the adjoint solve
    // works and is non-trivial.
    #[test]
    fn z_vector_adjoint_solve_converges() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let setup =
            crate::cphf::build_cpxtb_setup(&system, &params, &electronic, ao_opts, Some(&cphf.mos))
                .unwrap();
        let l_vectors = crate::cphf::density_gradient_adjoint_vectors(
            &system,
            &params,
            &electronic,
            ao_opts,
            &cphf.mos,
            &cphf.orbital_energies,
        )
        .unwrap();
        let opts = crate::cphf::CpxtbOptions::default();
        let ndof = 3 * system.atoms.len();
        let mut max_resid = 0.0_f64;
        let mut max_y_plus_x = 0.0_f64;
        for a in 0..ndof {
            let y = setup
                .solve_adjoint(&l_vectors[a], opts.tol, opts.max_iter)
                .unwrap();
            let ay = setup.matvec(&y.amplitudes).unwrap();
            let resid: f64 = ay
                .iter()
                .zip(l_vectors[a].iter())
                .map(|(v, l)| (v - l).abs())
                .fold(0.0, f64::max);
            max_resid = max_resid.max(resid);
            // y_a vs -x_a (would be equal iff L_a == -rhs_a)
            let dev: f64 = y
                .amplitudes
                .iter()
                .zip(cphf.solutions[a].amplitudes.iter())
                .map(|(yi, xi)| (yi + xi).abs())
                .fold(0.0, f64::max);
            max_y_plus_x = max_y_plus_x.max(dev);
        }
        eprintln!(
            "Z3: max|A y_a - L_a| = {max_resid:.3e} | max|y_a - (-x_a)| = {max_y_plus_x:.3e}"
        );
        assert!(
            max_resid < 1.0e-8,
            "Z-vector solve did not converge: {max_resid:.3e}"
        );
        assert!(
            max_y_plus_x > 1.0e-3,
            "y_a == -x_a (would mean L_a == -rhs_a): {max_y_plus_x:.3e}"
        );
    }

    // Stage Z5 keystone (VALIDATED to ~1e-6): analytic MO-coefficient derivatives C^(c) = C U^c vs FD of
    // aligned canonical mos, block-by-block on U = Cᵀ S C^(c), with TIGHT SCF convergence (so the FD floor
    // is not SCF-noise-limited). All blocks match: diagonal (= -1/2 S̃_c), ov (= x), vo (= -S̃ - x), and the
    // oo/vv canonical block [F̃_c = Cᵀ(h0_deriv + RF(γ·q_c))C]. The FD back-solve F̃_needed = (ε_q-ε_p)U_FD +
    // ε_q S̃_c additionally confirms F_c = h0_deriv + RF(γ·q_c) (F+) over skeleton (F0) / wrong-sign (F-).
    #[test]
    fn mo_coefficient_derivatives_match_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        // Tight SCF convergence so the FD reference is not limited by SCF noise (mos noise / 2h floor).
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let overlap = electronic.integrals.overlap.clone();
        let eps = cphf.orbital_energies.clone();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let nocc = electronic
            .occupations
            .iter()
            .filter(|&&o| o > 1.0e-8)
            .count();
        let h = 5.0e-5;
        let umat = |cder: &crate::linalg::Matrix| -> crate::linalg::Matrix {
            mos.transpose()
                .matmul(&overlap.matmul(cder).unwrap())
                .unwrap()
        };
        let (mut e_diag, mut e_ov, mut e_vo, mut e_oo, mut e_vv) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        // same-block F̃ candidate errors vs back-solved F̃_needed
        let (mut f0_oo, mut fp_oo, mut fm_oo) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut f0_vv, mut fp_vv, mut fm_vv) = (0.0_f64, 0.0_f64, 0.0_f64);
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let mut cfd = crate::linalg::Matrix::zeros(n, n);
            for mu in 0..n {
                for p in 0..n {
                    cfd[(mu, p)] = (setp.mos[(mu, p)] - setm.mos[(mu, p)]) / (2.0 * h);
                }
            }
            let u_fd = umat(&cfd);
            let u_an = umat(&c_analytic[c]);
            let (h0_mo, resp_mo, s_tilde) = &cand[c];
            for p in 0..n {
                for q in 0..n {
                    let e = (u_fd[(p, q)] - u_an[(p, q)]).abs();
                    let p_occ = p < nocc;
                    let q_occ = q < nocc;
                    if p == q {
                        e_diag = e_diag.max(e);
                    } else if p_occ != q_occ {
                        if p_occ {
                            e_ov = e_ov.max(e);
                        } else {
                            e_vo = e_vo.max(e);
                        }
                    } else {
                        // back-solve the needed relaxed Fock derivative element from FD
                        let f_needed = (eps[q] - eps[p]) * u_fd[(p, q)] + eps[q] * s_tilde[(p, q)];
                        let d0 = (f_needed - h0_mo[(p, q)]).abs();
                        let dp = (f_needed - (h0_mo[(p, q)] + resp_mo[(p, q)])).abs();
                        let dm = (f_needed - (h0_mo[(p, q)] - resp_mo[(p, q)])).abs();
                        if p_occ {
                            e_oo = e_oo.max(e);
                            f0_oo = f0_oo.max(d0);
                            fp_oo = fp_oo.max(dp);
                            fm_oo = fm_oo.max(dm);
                        } else {
                            e_vv = e_vv.max(e);
                            f0_vv = f0_vv.max(d0);
                            fp_vv = fp_vv.max(dp);
                            fm_vv = fm_vv.max(dm);
                        }
                    }
                }
            }
        }
        eprintln!(
            "Z5 U-blocks: diag={e_diag:.3e} ov={e_ov:.3e} vo={e_vo:.3e} oo={e_oo:.3e} vv={e_vv:.3e}"
        );
        eprintln!(
            "Z5 F-needed vs candidates  oo: F0={f0_oo:.3e} F+={fp_oo:.3e} F-={fm_oo:.3e} | vv: F0={f0_vv:.3e} F+={fp_vv:.3e} F-={fm_vv:.3e}"
        );
        // With tight SCF convergence the FD floor collapses and ALL blocks of U = Cᵀ S C^(c) match FD to
        // ~1e-6: diagonal (-1/2 S̃), ov (= x), vo (= -S̃ - x), and the oo/vv canonical block. (The earlier
        // 2.5e-3/7.4e-3 oo/vv "errors" were SCF-convergence noise amplified by 1/Δε, not a formula error.)
        assert!(e_diag < 1.0e-5, "U diagonal mismatch: {e_diag:.3e}");
        assert!(e_ov < 1.0e-5, "U ov mismatch: {e_ov:.3e}");
        assert!(e_vo < 1.0e-5, "U vo mismatch: {e_vo:.3e}");
        assert!(e_oo < 1.0e-5, "U oo mismatch: {e_oo:.3e}");
        assert!(e_vv < 1.0e-4, "U vv mismatch: {e_vv:.3e}");
        // The FD back-solve also confirms F_c = h0_deriv + RF(γ·q_c): F+ matches F̃_needed to ~1e-7 and
        // beats the skeleton (F0) and wrong-sign (F-) candidates by orders of magnitude.
        assert!(
            fp_oo < 1.0e-6,
            "F+ (oo) does not match F_needed: {fp_oo:.3e}"
        );
        assert!(
            fp_vv < 1.0e-5,
            "F+ (vv) does not match F_needed: {fp_vv:.3e}"
        );
        assert!(
            fp_oo < 0.1 * f0_oo && fp_oo < 0.1 * fm_oo,
            "F+ is not the best oo candidate: F0={f0_oo:.3e} F+={fp_oo:.3e} F-={fm_oo:.3e}"
        );
    }

    // Stage Z5: orbital-energy derivatives ε^(c)_p = F̃_c_pp - ε_p S̃_c_pp, where F̃_c = Cᵀ(h0_deriv + RF(γ q_c))C
    // is the relaxed effective-Fock derivative. Validates the DIAGONAL of F̃_c (the back-solve confirmed the
    // off-diagonal). Needed for D_c rhs (the ε_i·(CᵀS_bC) term). Tight SCF FD of the SCF orbital energies.
    #[test]
    fn orbital_energy_derivatives_match_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let eps = cphf.orbital_energies.clone();
        let cand =
            crate::cphf::relaxed_fock_derivative_candidates(&system, &params, &electronic, &cphf)
                .unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            let (h0_mo, resp_mo, s_tilde) = &cand[c];
            for p in 0..n {
                let eps_c_analytic = (h0_mo[(p, p)] + resp_mo[(p, p)]) - eps[p] * s_tilde[(p, p)];
                let eps_c_fd = (setp.orbital_energies[p] - setm.orbital_energies[p]) / (2.0 * h);
                max_err = max_err.max((eps_c_analytic - eps_c_fd).abs());
            }
        }
        eprintln!("Z5 ε^(c): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-5,
            "orbital-energy derivative mismatch: {max_err:.3e}"
        );
    }

    // Stage Z5: gauge-aware observable D_c(Cᵀ S_b C) = C^(c)ᵀ S_b C + Cᵀ S_bc C + Cᵀ S_b C^(c), validated
    // against tight-SCF FD of the aligned-gauge S̃_b. Exercises the analytic C^(c) AND the second
    // AO-overlap derivative S_bc (overlap_second_derivative_matrix) — the first piece of the D_c rhs / D_c A
    // plumbing (pure overlap, no SCC, so it isolates the second-derivative machinery).
    #[test]
    fn d_c_overlap_mo_derivative_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let c_analytic =
            crate::cphf::mo_coefficient_derivatives(&system, &params, &electronic, &cphf).unwrap();
        let ndof = 3 * system.atoms.len();
        let n = mos.rows();
        // reference S_b (MO transform helper)
        let motrans =
            |m: &crate::linalg::Matrix, u: &crate::linalg::Matrix| -> crate::linalg::Matrix {
                u.transpose().matmul(&m.matmul(u).unwrap()).unwrap()
            };
        let s_b_ref: Vec<crate::linalg::Matrix> = (0..ndof)
            .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
            .collect();
        let h = 5.0e-5;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            for b in 0..ndof {
                let s_b_p =
                    crate::cphf::overlap_first_derivative_matrix(&sp, &ep.basis, b).unwrap();
                let s_b_m =
                    crate::cphf::overlap_first_derivative_matrix(&sm, &em.basis, b).unwrap();
                let st_p = motrans(&s_b_p, &setp.mos);
                let st_m = motrans(&s_b_m, &setm.mos);
                let s_bc =
                    crate::cphf::overlap_second_derivative_matrix(&system, &electronic.basis, b, c)
                        .unwrap();
                // analytic D_c(Cᵀ S_b C) = C^(c)ᵀ S_b C + Cᵀ S_bc C + Cᵀ S_b C^(c)
                let t1 = c_analytic[c]
                    .transpose()
                    .matmul(&s_b_ref[b].matmul(&mos).unwrap())
                    .unwrap();
                let t2 = motrans(&s_bc, &mos);
                let t3 = mos
                    .transpose()
                    .matmul(&s_b_ref[b].matmul(&c_analytic[c]).unwrap())
                    .unwrap();
                for p in 0..n {
                    for q in 0..n {
                        let analytic = t1[(p, q)] + t2[(p, q)] + t3[(p, q)];
                        let fd = (st_p[(p, q)] - st_m[(p, q)]) / (2.0 * h);
                        max_err = max_err.max((analytic - fd).abs());
                    }
                }
            }
        }
        eprintln!("Z5 D_c(CᵀS_bC): max|analytic - FD| = {max_err:.3e}");
        assert!(
            max_err < 1.0e-4,
            "D_c(Cᵀ S_b C) does not match FD: {max_err:.3e}"
        );
    }

    // Stage Z4: the Z-vector bridge for the orbital-sector nuclear derivative
    //   D_c R_orbital_ab = (D_c L_a)·x_b + y_a·[D_c rhs_b − (D_c A) x_b],   y_a = A⁻¹ L_a
    // must reproduce the finite-difference of R_orbital (which DOES re-solve CPHF at ±h to get x_b(±h)).
    // Closing this proves the interchange L_a^T x_bc = y_a^T[D_c rhs_b − (D_c A)x_b] — the orbital sector's
    // second-order response is captured WITHOUT a second-order solve. L_a, rhs_b, A are FD'd here (Stage Z4
    // scaffold); Stage Z5 makes them analytic. Aligned-gauge (to cphf.mos) for L/rhs/A; R_orbital is
    // gauge-invariant so its FD reference needs no alignment.
    #[test]
    fn z_vector_bridge_matches_orbital_sector_derivative() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf.mos.clone();
        let ndof = 3 * system.atoms.len();
        let xref: Vec<Vec<f64>> = (0..ndof)
            .map(|a| cphf.solutions[a].amplitudes.clone())
            .collect();
        let npair = xref[0].len();
        let opts = crate::cphf::CpxtbOptions::default();
        // Reference adjoint and Z-vectors.
        let setup_ref =
            crate::cphf::build_cpxtb_setup(&system, &params, &electronic, ao_opts, Some(&mos))
                .unwrap();
        let l_ref = crate::cphf::density_gradient_adjoint_vectors(
            &system,
            &params,
            &electronic,
            ao_opts,
            &mos,
            &cphf.orbital_energies,
        )
        .unwrap();
        let yvec: Vec<Vec<f64>> = (0..ndof)
            .map(|a| {
                setup_ref
                    .solve_adjoint(&l_ref[a], opts.tol, opts.max_iter)
                    .unwrap()
                    .amplitudes
            })
            .collect();

        let mdot = |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(a, b)| a * b).sum() };
        let solve = |sys: &PeriodicSystem, el: &ElectronicResult| {
            crate::cphf::solve_nonpbc_cpxtb_hessian_response(
                sys,
                &params,
                el,
                ao_opts,
                crate::cphf::CpxtbOptions::default(),
            )
            .unwrap()
        };
        let h = 1.0e-4;
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            if setp.space.pairs.len() != npair || setm.space.pairs.len() != npair {
                continue;
            }
            // D_c L (aligned gauge).
            let lp = crate::cphf::density_gradient_adjoint_vectors(
                &sp,
                &params,
                &ep,
                ao_opts,
                &setp.mos,
                &setp.orbital_energies,
            )
            .unwrap();
            let lm = crate::cphf::density_gradient_adjoint_vectors(
                &sm,
                &params,
                &em,
                ao_opts,
                &setm.mos,
                &setm.orbital_energies,
            )
            .unwrap();
            // FD reference: R_orbital with re-solved amplitudes at ±h (gauge-invariant, self-consistent).
            let cp = solve(&sp, &ep);
            let cm = solve(&sm, &em);
            let xp: Vec<Vec<f64>> = (0..ndof)
                .map(|b| cp.solutions[b].amplitudes.clone())
                .collect();
            let xm: Vec<Vec<f64>> = (0..ndof)
                .map(|b| cm.solutions[b].amplitudes.clone())
                .collect();
            let rorb_p = crate::cphf::orbital_sector_response_hessian(
                &sp,
                &params,
                &ep,
                ao_opts,
                &cp.mos,
                &cp.orbital_energies,
                &xp,
            )
            .unwrap();
            let rorb_m = crate::cphf::orbital_sector_response_hessian(
                &sm,
                &params,
                &em,
                ao_opts,
                &cm.mos,
                &cm.orbital_energies,
                &xm,
            )
            .unwrap();
            for a in 0..ndof {
                let dl: Vec<f64> = (0..npair)
                    .map(|p| (lp[a][p] - lm[a][p]) / (2.0 * h))
                    .collect();
                for b in 0..ndof {
                    let drhs: Vec<f64> = (0..npair)
                        .map(|p| (setp.rhs_vectors[b][p] - setm.rhs_vectors[b][p]) / (2.0 * h))
                        .collect();
                    let axp = setp.matvec(&xref[b]).unwrap();
                    let axm = setm.matvec(&xref[b]).unwrap();
                    let dax: Vec<f64> = (0..npair).map(|p| (axp[p] - axm[p]) / (2.0 * h)).collect();
                    let rhs_minus_ax: Vec<f64> = (0..npair).map(|p| drhs[p] - dax[p]).collect();
                    let zbridge = mdot(&dl, &xref[b]) + mdot(&yvec[a], &rhs_minus_ax);
                    let fd_rorb = (rorb_p[(a, b)] - rorb_m[(a, b)]) / (2.0 * h);
                    max_err = max_err.max((zbridge - fd_rorb).abs());
                }
            }
        }
        eprintln!("Z4: max|Z_bridge_orbital − FD(R_orbital)| = {max_err:.3e}");
        assert!(
            max_err < 5.0e-4,
            "Z-vector orbital bridge does not match FD(R_orbital): {max_err:.3e}"
        );
    }

    // Stage-1 floor-free gate: the bridge's RESPONSE part (D_c R^orb + D_c M, i.e. the orbital CP-residual
    // bridge + the metric FD bridge) must reproduce the EXACT geometric derivative of the response Hessian
    // D_c(cphf.hessian_response) = D_c R^code, independent of the `frozen_hess_at` FD floor. Since
    // R^code = R^orb + M, this is D_c R^orb + D_c M ?= D_c R^code — the localization that `M = R^code − R^orb`
    // is the only remaining response term.
    #[test]
    fn metric_bridge_response_matches_response_hessian_derivative() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let ao_opts = crate::cphf::AoDerivativeOptions {
            coordination_cutoff: eo.hamiltonian.coordination_cutoff,
            include_cn_h0: eo.hamiltonian.enable_cn_hamiltonian,
        };
        let electronic = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        let cphf_ref = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let mos = cphf_ref.mos.clone();
        let ndof = 3 * system.atoms.len();
        let xresp: Vec<Vec<f64>> = (0..ndof)
            .map(|a| cphf_ref.solutions[a].amplitudes.clone())
            .collect();
        let npair = xresp[0].len();
        let h = 1.0e-4;
        let mdot = |rhs: &[f64], x: &[f64]| -> f64 { rhs.iter().zip(x).map(|(r, y)| r * y).sum() };
        let solve = |sys: &PeriodicSystem, el: &ElectronicResult| {
            crate::cphf::solve_nonpbc_cpxtb_hessian_response(
                sys,
                &params,
                el,
                ao_opts,
                crate::cphf::CpxtbOptions::default(),
            )
            .unwrap()
        };
        let mut max_err = 0.0_f64;
        for c in 0..ndof {
            let (atom, ax) = (c / 3, c % 3);
            let mut sp = system.clone();
            let mut sm = system.clone();
            match ax {
                0 => {
                    sp.atoms[atom].position.x += h;
                    sm.atoms[atom].position.x -= h;
                }
                1 => {
                    sp.atoms[atom].position.y += h;
                    sm.atoms[atom].position.y -= h;
                }
                _ => {
                    sp.atoms[atom].position.z += h;
                    sm.atoms[atom].position.z -= h;
                }
            }
            let ep = crate::electronic::run_electronic(&sp, &params, eo.clone()).unwrap();
            let em = crate::electronic::run_electronic(&sm, &params, eo.clone()).unwrap();
            let cp = solve(&sp, &ep);
            let cm = solve(&sm, &em);
            let setp =
                crate::cphf::build_cpxtb_setup(&sp, &params, &ep, ao_opts, Some(&mos)).unwrap();
            let setm =
                crate::cphf::build_cpxtb_setup(&sm, &params, &em, ao_opts, Some(&mos)).unwrap();
            if setp.space.pairs.len() != npair || setm.space.pairs.len() != npair {
                continue;
            }
            let mut drhs = vec![vec![0.0_f64; npair]; ndof];
            let mut rbc = vec![vec![0.0_f64; npair]; ndof];
            for a in 0..ndof {
                for p in 0..npair {
                    drhs[a][p] = (setp.rhs_vectors[a][p] - setm.rhs_vectors[a][p]) / (2.0 * h);
                }
            }
            for b in 0..ndof {
                let avp = setp.matvec(&xresp[b]).unwrap();
                let avm = setm.matvec(&xresp[b]).unwrap();
                for p in 0..npair {
                    rbc[b][p] = ((avp[p] - setp.rhs_vectors[b][p])
                        - (avm[p] - setm.rhs_vectors[b][p]))
                        / (2.0 * h);
                }
            }
            for a in 0..ndof {
                for b in 0..ndof {
                    let mut orb = 0.0_f64;
                    for p in 0..npair {
                        orb += -drhs[a][p] * xresp[b][p] + xresp[a][p] * rbc[b][p];
                    }
                    let m_p = cp.hessian_response[(a, b)]
                        + mdot(&cp.rhs_vectors[a], &cp.solutions[b].amplitudes);
                    let m_m = cm.hessian_response[(a, b)]
                        + mdot(&cm.rhs_vectors[a], &cm.solutions[b].amplitudes);
                    let metric_fd = (m_p - m_m) / (2.0 * h);
                    let fd_code =
                        (cp.hessian_response[(a, b)] - cm.hessian_response[(a, b)]) / (2.0 * h);
                    max_err = max_err.max((orb + metric_fd - fd_code).abs());
                }
            }
        }
        eprintln!(
            "Stage-1 floor-free: max|D_cR^orb + D_cM − D_c(cphf.hessian_response)| = {max_err:.3e}"
        );
        // Exact to FD precision (measured ~1.4e-8 on water): the response side is fully closed; the only
        // remaining error in the whole tensor is the `frozen_hess_at` FD-path floor (removed in Stage 4).
        assert!(
            max_err < 1.0e-6,
            "metric bridge response does not match D_c R^code: {max_err:.3e}"
        );
    }

    // Vector mode end-to-end: contract_last(v) over a real assembled bundle equals the
    // directional derivative of the combined Hessian along v (Σ_c v_c T_abc = ∂H/∂v).
    #[test]
    fn geometric_vector_mode_matches_directional_hessian_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let store = third_derivative_geometric(&system, &params).unwrap();
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|i| 0.1 * ((i * 7 + 3) % 11) as f64 - 0.5)
            .collect();
        let k = store.contract_last(&v);
        let hess = |sys: &PeriodicSystem| -> Matrix {
            let mut m = crate::repulsion::repulsion_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            let hal = crate::halogen::halogen_energy_gradient_hessian(sys, &params)
                .unwrap()
                .hessian;
            for r in 0..ndof {
                for c in 0..ndof {
                    m[(r, c)] += hal[(r, c)];
                }
            }
            m
        };
        let h = 1.0e-5;
        let mut plus = system.clone();
        let mut minus = system.clone();
        for atom in 0..system.atoms.len() {
            plus.atoms[atom].position.x += h * v[3 * atom];
            plus.atoms[atom].position.y += h * v[3 * atom + 1];
            plus.atoms[atom].position.z += h * v[3 * atom + 2];
            minus.atoms[atom].position.x -= h * v[3 * atom];
            minus.atoms[atom].position.y -= h * v[3 * atom + 1];
            minus.atoms[atom].position.z -= h * v[3 * atom + 2];
        }
        let hp = hess(&plus);
        let hm = hess(&minus);
        let mut max_delta = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                let fd = (hp[(a, b)] - hm[(a, b)]) / (2.0 * h);
                max_delta = max_delta.max((k[(a, b)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "Vector-mode directional FD max delta {max_delta:.3e}"
        );
    }

    // The scalar T[v,v,v] mode equals the dense triple contraction.
    #[test]
    fn symmetric_third_contract_vvv_matches_dense() {
        let ndof = 6;
        let rel = Vec3::new(1.6, 0.7, 0.2);
        let mut dense = vec![Matrix::zeros(ndof, ndof); ndof];
        add_radial_third_block(&mut dense, 0, 1, rel, 0.3, -1.1, 0.5);
        let mut sym = SymmetricThird::zeros(ndof);
        add_radial_third_block_sym(&mut sym, 0, 1, rel, 0.3, -1.1, 0.5);
        let v = [0.3_f64, -0.7, 1.1, 0.2, -0.5, 0.9];
        let s = sym.contract_vvv(&v);
        let mut dense_s = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    dense_s += dense[c][(a, b)] * v[a] * v[b] * v[c];
                }
            }
        }
        assert!((s - dense_s).abs() < 1.0e-12, "{s} vs {dense_s}");
    }

    // The semi-numerical Vector-mode third derivative (FD of the analytic Hessian along a
    // direction) must vanish along a uniform translation -- the Hessian is translation-invariant,
    // so its directional derivative along a rigid shift is zero. Validates the production path.
    #[test]
    fn seminumerical_vector_translational_invariance() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let mut eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        eo.energy_tolerance = 1.0e-9;
        eo.charge_tolerance = 1.0e-9;
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: eo,
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        // Uniform translation along x.
        let mut v = vec![0.0_f64; ndof];
        for atom in 0..nat {
            v[3 * atom] = 1.0;
        }
        let k =
            third_derivative_seminumerical_vector(&system, &params, options, &v, 1.0e-3).unwrap();
        let mut max = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                max = max.max(k[(a, b)].abs());
                // Symmetric, since the analytic Hessian is.
                assert!((k[(a, b)] - k[(b, a)]).abs() < 1.0e-8);
            }
        }
        assert!(
            max < 1.0e-4,
            "semi-numerical third deriv not translation-invariant: max {max:.3e}"
        );
    }

    // Fermi-smearing (finite electronic temperature) support for the production third derivative.
    // At kt > 0 the analytic Hessian the semi-numerical path differentiates is the FREE-ENERGY
    // Hessian -- its CPHF carries the occupation/finite-T responses (cphf.rs
    // `finite_temperature_density_response`), and `third_derivative_seminumerical_*` route
    // `electronic_options.electronic_temperature` straight through to both Hessian evaluations. On
    // the small-gap Ni(CO)4 complex at elevated T the SCF genuinely smears (nonzero electronic
    // entropy => fractional frontier occupations), so the finite-T branch is exercised; the
    // directional third derivative of the free energy along a RIGID TRANSLATION must still vanish
    // (the free-energy Hessian is translation-invariant) -- a physical correctness gate confirming
    // the finite-T third derivative is well-formed.
    #[test]
    fn seminumerical_third_derivative_handles_fermi_smearing() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n",
            0.0,
            false,
        )
        .unwrap();
        let mut eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        eo.electronic_temperature = 3000.0;
        eo.energy_tolerance = 1.0e-11;
        eo.charge_tolerance = 1.0e-10;
        eo.max_scc = 500;

        // Fermi smearing is genuinely active: nonzero electronic entropy => fractional occupations.
        let elec = crate::electronic::run_electronic(&system, &params, eo.clone()).unwrap();
        assert!(
            elec.electronic_entropy_term.abs() > 1.0e-7,
            "Ni(CO)4 at 3000 K should smear (entropy {:.3e})",
            elec.electronic_entropy_term
        );
        let fractional = elec
            .occupations
            .iter()
            .any(|&o| (o - o.round()).abs() > 1.0e-4);
        assert!(
            fractional,
            "expected fractional frontier occupations at finite T"
        );

        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            include_halogen: false,
            electronic_options: eo,
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        // Rigid +x translation.
        let mut v = vec![0.0_f64; ndof];
        for atom in 0..nat {
            v[3 * atom] = 1.0;
        }
        let k =
            third_derivative_seminumerical_vector(&system, &params, options, &v, 1.0e-2).unwrap();
        let mut max = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                max = max.max(k[(a, b)].abs());
            }
        }
        assert!(
            max < 1.0e-4,
            "finite-T (smeared) third derivative not translation-invariant: max {max:.3e}"
        );
    }

    // The semi-numerical Dense mode (per-axis FD, symmetric-packed) contracted with a direction
    // must agree with the direct Vector mode (FD along that direction) -- validates the Dense
    // packing/canonical indexing against the (separately gated) Vector mode.
    #[test]
    fn seminumerical_dense_contract_matches_vector() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system =
            PeriodicSystem::from_xyz_str("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0.0, false)
                .unwrap();
        let mut eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        eo.energy_tolerance = 1.0e-9;
        eo.charge_tolerance = 1.0e-9;
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: eo,
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof).map(|i| 0.2 * (i as f64) - 0.3).collect();
        let dense = third_derivative_seminumerical_dense(&system, &params, options.clone(), 1.0e-3)
            .unwrap();
        let vector =
            third_derivative_seminumerical_vector(&system, &params, options, &v, 1.0e-3).unwrap();
        let k = dense.contract_last(&v);
        let mut max = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                max = max.max((k[(a, b)] - vector[(a, b)]).abs());
            }
        }
        assert!(
            max < 1.0e-4,
            "Dense.contract(v) vs Vector(v) max delta {max:.3e}"
        );
    }

    // The full semi-numerical Dense third-derivative tensor must obey the acoustic sum rule
    // Σ_A T_{Aα,bc} = 0 (the Hessian is translation-invariant, so its derivative sums to zero
    // over atoms) -- a physical check on the whole tensor.
    #[test]
    fn seminumerical_dense_acoustic_sum_rule() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system =
            PeriodicSystem::from_xyz_str("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0.0, false)
                .unwrap();
        let mut eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        eo.energy_tolerance = 1.0e-9;
        eo.charge_tolerance = 1.0e-9;
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: eo,
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let dense =
            third_derivative_seminumerical_dense(&system, &params, options, 1.0e-3).unwrap();
        let mut max = 0.0_f64;
        for alpha in 0..3 {
            for b in 0..ndof {
                for c in 0..ndof {
                    let sum: f64 = (0..nat).map(|atom| dense.get(3 * atom + alpha, b, c)).sum();
                    max = max.max(sum.abs());
                }
            }
        }
        assert!(max < 1.0e-4, "acoustic sum rule violated: max {max:.3e}");
    }

    // The per-DOF atomic charge responses (Σ over shells of each atom) must obey the
    // charge-conservation sum rule Σ_atom ∂q_atom/∂R = 0 (total charge is fixed) -- a physical
    // check on the CPHF responses the L_xxx term contracts.
    #[test]
    fn atomic_charge_responses_conserve_charge() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        let electronic =
            crate::electronic::run_electronic(&system, &params, options.clone()).unwrap();
        let response = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            crate::cphf::AoDerivativeOptions {
                coordination_cutoff: options.hamiltonian.coordination_cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            crate::cphf::CpxtbOptions::default(),
        )
        .unwrap();
        let nat = system.atoms.len();
        for d in 0..3 * nat {
            let mut atom_resp = vec![0.0_f64; nat];
            for (ish, shell) in electronic.basis.shells.iter().enumerate() {
                atom_resp[shell.atom_index] += response.shell_charge_responses[d][ish];
            }
            let total: f64 = atom_resp.iter().sum();
            assert!(
                total.abs() < 1.0e-6,
                "atomic charge response for DOF {d} violates charge conservation: {total:.3e}"
            );
        }
    }

    // Block mode: the sub-tensor for a DOF subset matches the corresponding dense entries.
    #[test]
    fn symmetric_third_block_matches_dense() {
        let ndof = 6;
        let rel = Vec3::new(1.6, 0.7, 0.2);
        let mut sym = SymmetricThird::zeros(ndof);
        add_radial_third_block_sym(&mut sym, 0, 1, rel, 0.3, -1.1, 0.5);
        let dofs = [1usize, 3, 4];
        let blk = sym.block(&dofs);
        for a in 0..dofs.len() {
            for b in 0..dofs.len() {
                for c in 0..dofs.len() {
                    assert!((blk[c][(a, b)] - sym.get(dofs[a], dofs[b], dofs[c])).abs() < 1.0e-15);
                }
            }
        }
    }

    // Semi-numerical Block mode (compute-saving): the sub-tensor computed by FD-ing the Hessian
    // only along the in-block axes equals the corresponding entries of the full Dense tensor
    // realizing the OOM-saving Block path without materializing the ndof³ tensor.
    #[test]
    fn seminumerical_block_matches_dense_subset() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let mut eo = crate::electronic::ElectronicOptions {
            enable_dispersion: false,
            ..crate::electronic::ElectronicOptions::default()
        };
        eo.energy_tolerance = 1.0e-10;
        eo.charge_tolerance = 1.0e-9;
        let options = crate::hessian::AnalyticHessianOptions {
            include_dispersion: false,
            electronic_options: eo,
            ..crate::hessian::AnalyticHessianOptions::default()
        };
        let step = 1.0e-3;
        // Block over the O atom (index 0): DOFs {0,1,2}.
        let (dofs, slabs) =
            third_derivative_seminumerical_block(&system, &params, options.clone(), &[0], step)
                .unwrap();
        assert_eq!(dofs, vec![0, 1, 2]);
        // Reference: the full Dense tensor restricted to the same DOFs.
        let dense = third_derivative_seminumerical_dense(&system, &params, options, step).unwrap();
        let dense_block = dense.block(&dofs);
        let m = dofs.len();
        let mut max = 0.0_f64;
        for c in 0..m {
            for a in 0..m {
                for b in 0..m {
                    max = max.max((slabs[c][(a, b)] - dense_block[c][(a, b)]).abs());
                }
            }
        }
        assert!(
            max < 1.0e-9,
            "semi-numerical Block vs Dense subset mismatch: max {max:.3e}"
        );
    }

    // Dense materialization round-trips: the symmetric store → dense slabs equals the
    // original dense block.
    #[test]
    fn symmetric_third_to_dense_slabs_matches() {
        let ndof = 6;
        let rel = Vec3::new(1.6, 0.7, 0.2);
        let (g, f3, scale) = (0.3_f64, -1.1_f64, 0.5_f64);
        let mut dense = vec![Matrix::zeros(ndof, ndof); ndof];
        add_radial_third_block(&mut dense, 0, 1, rel, g, f3, scale);
        let mut sym = SymmetricThird::zeros(ndof);
        add_radial_third_block_sym(&mut sym, 0, 1, rel, g, f3, scale);
        let slabs = sym.to_dense_slabs();
        for a in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    assert!((slabs[c][(a, b)] - dense[c][(a, b)]).abs() < 1.0e-15);
                }
            }
        }
    }

    // The Vector-mode contraction over the symmetric store equals the dense directional
    // contraction K[a][b] = Σ_c T_abc v_c.
    #[test]
    fn symmetric_third_contract_last_matches_dense() {
        let ndof = 6;
        let rel = Vec3::new(1.6, 0.7, 0.2);
        let (g, f3, scale) = (0.3_f64, -1.1_f64, 0.5_f64);
        let mut dense = vec![Matrix::zeros(ndof, ndof); ndof];
        add_radial_third_block(&mut dense, 0, 1, rel, g, f3, scale);
        let mut sym = SymmetricThird::zeros(ndof);
        add_radial_third_block_sym(&mut sym, 0, 1, rel, g, f3, scale);
        let v = [0.3_f64, -0.7, 1.1, 0.2, -0.5, 0.9];
        let k = sym.contract_last(&v);
        for a in 0..ndof {
            for b in 0..ndof {
                let mut s = 0.0;
                for c in 0..ndof {
                    s += dense[c][(a, b)] * v[c];
                }
                assert!(
                    (k[(a, b)] - s).abs() < 1.0e-12,
                    "({a},{b}): {} vs {s}",
                    k[(a, b)]
                );
            }
        }
    }
