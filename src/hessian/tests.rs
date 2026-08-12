    use super::*;
    use crate::basis::{BasisOptions, BasisSet};

    // The frozen-shell-charge SCC2 third derivative (slab c = ∂H/∂R_c) must match the
    // central FD of its own analytic Hessian. The charges are held fixed, so this isolated
    // FD is valid (no electronic response in the frozen block).
    #[test]
    fn fixed_scc_third_derivative_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let nsh = basis.shells.len();
        // Arbitrary but fixed shell charges (the block is bilinear in them).
        let q: Vec<f64> = (0..nsh)
            .map(|i| 0.13 * ((i % 3) as f64 - 1.0) + 0.05)
            .collect();

        let third = fixed_shell_charge_scc_third_derivative(&system, &basis, &q, &params).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = fixed_shell_charge_scc_hessian(&plus, &basis, &q, &params)
                .unwrap()
                .hessian;
            let hm = fixed_shell_charge_scc_hessian(&minus, &basis, &q, &params)
                .unwrap()
                .hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "frozen SCC third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    /// Water monomer with converged shell charges, shared by the frozen-SCC quartic tests.
    fn scc_fourth_probe() -> (Gfn1Parameters, PeriodicSystem, ElectronicResult) {
        let params = Gfn1Parameters::builtin().expect("builtin GFN1 parameters");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        (params, system, electronic)
    }

    // The frozen-shell-charge SCC2 fourth derivative must reproduce the central FD of its own
    // analytic third derivative: `Q(a,b,c,d) = ∂ T[c][(a,b)] / ∂R_d`. The shell charges are
    // frozen at the converged values of the reference geometry (the block carries no
    // electronic response), so this isolated FD is the valid gate.
    #[test]
    fn fixed_scc_fourth_derivative_matches_third_finite_difference() {
        let (params, system, electronic) = scc_fourth_probe();
        let basis = &electronic.basis;
        let q = &electronic.shell_charges;
        let fourth = fixed_shell_charge_scc_fourth_derivative(&system, basis, q, &params).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, step);
            displace(&mut minus, d, -step);
            let tp = fixed_shell_charge_scc_third_derivative(&plus, basis, q, &params).unwrap();
            let tm = fixed_shell_charge_scc_third_derivative(&minus, basis, q, &params).unwrap();
            for c in 0..ndof {
                for a in 0..ndof {
                    for b in 0..ndof {
                        let fd = (tp[c][(a, b)] - tm[c][(a, b)]) / (2.0 * step);
                        max_delta = max_delta.max((fourth.get(a, b, c, d) - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "frozen SCC fourth-derivative FD-vs-third max delta {max_delta:.3e}"
        );
    }

    // Translational invariance: summing any one index over all atoms (fixed Cartesian
    // component) must annihilate the frozen-SCC quartic tensor.
    #[test]
    fn fixed_scc_fourth_derivative_acoustic_sum_rule() {
        let (params, system, electronic) = scc_fourth_probe();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let fourth = fixed_shell_charge_scc_fourth_derivative(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let mut max_sum = 0.0_f64;
        for alpha in 0..3 {
            for b in 0..ndof {
                for c in 0..ndof {
                    for d in 0..ndof {
                        let mut sum = 0.0;
                        for atom in 0..nat {
                            sum += fourth.get(3 * atom + alpha, b, c, d);
                        }
                        max_sum = max_sum.max(sum.abs());
                    }
                }
            }
        }
        assert!(
            max_sum < 1.0e-9,
            "frozen SCC fourth-derivative acoustic sum rule residual {max_sum:.3e}"
        );
    }

    // Scalar gate on the Klopman–Ohno radial ladder itself: the newly added `γ''''` against
    // (i) the central FD of the analytic `γ'''` and (ii) the independent closed form
    // `γ'''' = 105 r⁴ s^(−9/2) − 90 r² s^(−7/2) + 9 s^(−5/2)`, `s = r² + 1/γ_h²`.
    #[test]
    fn effective_kernel_radial_ladder_fourth_matches_third_finite_difference() {
        let step = 1.0e-5;
        let mut max_rel_fd = 0.0_f64;
        let mut max_rel_closed = 0.0_f64;
        for &gamma_h in &[0.2_f64, 0.47, 1.0, 2.3] {
            for &r in &[0.8_f64, 1.5, 2.5, 4.0, 7.0] {
                let f4 = effective_kernel_derivatives(r, gamma_h).radial_fourth_derivative;
                let f3_plus =
                    effective_kernel_derivatives(r + step, gamma_h).radial_third_derivative;
                let f3_minus =
                    effective_kernel_derivatives(r - step, gamma_h).radial_third_derivative;
                let fd = (f3_plus - f3_minus) / (2.0 * step);
                max_rel_fd = max_rel_fd.max((f4 - fd).abs() / f4.abs().max(1.0e-300));

                let s = r * r + 1.0 / (gamma_h * gamma_h);
                let closed = 105.0 * r.powi(4) * s.powf(-4.5) - 90.0 * r * r * s.powf(-3.5)
                    + 9.0 * s.powf(-2.5);
                max_rel_closed =
                    max_rel_closed.max((f4 - closed).abs() / f4.abs().max(1.0e-300));
            }
        }
        assert!(
            max_rel_fd < 1.0e-8,
            "kernel radial ladder γ'''' vs FD(γ''') max relative delta {max_rel_fd:.3e}"
        );
        assert!(
            max_rel_closed < 1.0e-12,
            "kernel radial ladder γ'''' vs closed form max relative delta {max_rel_closed:.3e}"
        );
    }

    // The frozen-density Pulay/overlap+H0 third derivative (slab c = ∂H/∂R_c) must match the
    // central FD of its own analytic Hessian. P, W, the shell potential and CN are all held
    // fixed (the `electronic` result is reused at displaced geometries), so this isolated FD
    // is valid — it carries no electronic response. This block is the first consumer of the
    // B1 third-derivative AO integrals.
    #[test]
    fn fixed_pulay_third_derivative_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third = fixed_density_pulay_third_derivative(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = fixed_density_pulay_hessian(&plus, &params, &electronic)
                .unwrap()
                .hessian;
            let hm = fixed_density_pulay_hessian(&minus, &params, &electronic)
                .unwrap()
                .hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "frozen Pulay third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // The `d_edcn` jet (∂E/∂CN_A and its 1st/2nd/3rd nuclear derivatives at frozen density) must
    // satisfy its own derivative ladder: grad = FD(value), hess = FD(grad), third = FD(hess). This
    // validates the `scale·overlap` Leibniz (the error-prone core of the CN-H0 third derivative) in
    // isolation, reusing the same `electronic` (frozen density) at displaced geometries.
    #[test]
    fn cn_h0_dedcn_jet_derivative_ladder_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let jets = cn_h0_dedcn_jets(&system, &params, &electronic).unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let step = 1.0e-5;
        let (mut max_g, mut max_h, mut max_t) = (0.0_f64, 0.0_f64, 0.0_f64);
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, step);
            displace(&mut minus, d, -step);
            let jp = cn_h0_dedcn_jets(&plus, &params, &electronic).unwrap();
            let jm = cn_h0_dedcn_jets(&minus, &params, &electronic).unwrap();
            for atom in 0..nat {
                let fd_g = (jp[atom].value - jm[atom].value) / (2.0 * step);
                max_g = max_g.max((jets[atom].grad[d] - fd_g).abs());
                for a in 0..ndof {
                    let fd_h = (jp[atom].grad[a] - jm[atom].grad[a]) / (2.0 * step);
                    max_h = max_h.max((jets[atom].hess[a * ndof + d] - fd_h).abs());
                    for b in 0..ndof {
                        let fd_t = (jp[atom].hess[a * ndof + b] - jm[atom].hess[a * ndof + b])
                            / (2.0 * step);
                        max_t =
                            max_t.max((jets[atom].third[(a * ndof + b) * ndof + d] - fd_t).abs());
                    }
                }
            }
        }
        assert!(max_g < 1.0e-6, "d_edcn grad vs FD: {max_g:.3e}");
        assert!(max_h < 1.0e-5, "d_edcn hess vs FD: {max_h:.3e}");
        assert!(max_t < 1.0e-4, "d_edcn third vs FD: {max_t:.3e}");
    }

    // The SCC-scalar × overlap-derivative coupling block's third derivative matches the central FD of
    // `fixed_density_scalar_overlap_hessian` at FIXED density (reference `electronic` reused at displaced
    // geometries). This is the block `analytic_hessian` adds but `third_derivative_frozen_complete` lacked.
    #[test]
    fn fixed_density_scalar_overlap_third_derivative_matches_hessian_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third =
            fixed_density_scalar_overlap_third_derivative(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            fixed_density_scalar_overlap_hessian(sys, &params, &electronic).unwrap()
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "scalar-overlap third-derivative FD max delta {max_delta:.3e}"
        );
    }

    // Response-side block ladder, Step 1: the bare-H0 SECOND nuclear derivative (at fixed CN) matches the
    // central FD of the bare-H0 first derivative. Establishes the AO pair mapping + signs + the
    // scale·overlap 2nd-derivative machinery before adding SCC-scalar and CN-H0 blocks to `F_bc`.
    #[test]
    fn h0_bare_second_derivative_matches_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for b in 0..ndof {
            for c in 0..ndof {
                let analytic =
                    h0_bare_second_derivative_matrix(&system, &params, &electronic, b, c).unwrap();
                let (atom, ax) = (c / 3, c % 3);
                let mut sp = system.clone();
                let mut sm = system.clone();
                displace(&mut sp, 3 * atom + ax, step);
                displace(&mut sm, 3 * atom + ax, -step);
                // bare-H0 first deriv at displaced geometry, fixed reference CN (pass reference electronic).
                let fp = h0_bare_first_derivative_matrix(&sp, &params, &electronic, b).unwrap();
                let fm = h0_bare_first_derivative_matrix(&sm, &params, &electronic, b).unwrap();
                for mu in 0..n {
                    for nu in 0..n {
                        let fd = (fp[(mu, nu)] - fm[(mu, nu)]) / (2.0 * step);
                        max_delta = max_delta.max((analytic[(mu, nu)] - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "bare-H0 second derivative FD max delta {max_delta:.3e}"
        );
    }

    // The CN-H0 frozen third derivative (slab a = ∂H_bc/∂R_a) matches the central FD of the analytic
    // CN-H0 Hessian block (`fixed_density_cn_h0_hessian` + `fixed_density_cn_h0_pulay_cross_hessian`),
    // with the frozen `electronic` reused at displaced geometries. The last frozen `L_abc` block.
    #[test]
    fn fixed_density_cn_h0_third_derivative_matches_hessian_fd() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third =
            fixed_density_cn_h0_third_derivative(&system, &params, &electronic, cutoff).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let cn_h0_hess = |sys: &PeriodicSystem| -> Matrix {
            let mut h = fixed_density_cn_h0_hessian(sys, &params, &electronic, cutoff)
                .unwrap()
                .hessian;
            let cross =
                fixed_density_cn_h0_pulay_cross_hessian(sys, &params, &electronic, cutoff).unwrap();
            for r in 0..ndof {
                for c in 0..ndof {
                    h[(r, c)] += cross[(r, c)];
                }
            }
            h
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = cn_h0_hess(&plus);
            let hm = cn_h0_hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "CN-H0 third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // The CN counting-function third radial derivative (sigmoid chain rule) must match the
    // central FD of its analytic second derivative. Prerequisite for the CN-H0 and D3
    // many-body third-derivative chain rules. Self-contained (no params).
    #[test]
    fn coordination_third_derivative_matches_second_finite_difference() {
        let kcn = CoordinationOptions::default().kcn;
        let h = 1.0e-6;
        for &rc in &[3.0_f64, 4.5] {
            for &r in &[1.2_f64, 2.5, 4.0, 6.0] {
                let d = coordination_value_derivatives(kcn, r, rc);
                let fd = (coordination_value_derivatives(kcn, r + h, rc).second
                    - coordination_value_derivatives(kcn, r - h, rc).second)
                    / (2.0 * h);
                assert!(
                    (d.third - fd).abs() < 1.0e-5 * (1.0 + d.third.abs()),
                    "CN''' at r={r}, rc={rc}: analytic {} vs FD {fd}",
                    d.third
                );
            }
        }
    }

    // The CN pair counting function's Cartesian third-derivative tensor (CN radials fed into
    // the shared central rank-3 block) must match the FD of the analytic CN pair Hessian.
    // This is the per-pair kernel the CN-H0 and D3 many-body assemblies build on.
    #[test]
    fn cn_pair_third_block_matches_hessian_finite_difference() {
        let kcn = CoordinationOptions::default().kcn;
        let rc = 3.2_f64;
        let pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.6, 0.7, 0.2)];
        let ndof = 6;
        let cn_hess = |p: &[Vec3; 2]| -> Matrix {
            let rel = p[0] - p[1];
            let r = rel.norm();
            let u = (rel / r).to_array();
            let d = coordination_value_derivatives(kcn, r, rc);
            let signs = [1.0_f64, -1.0];
            let mut hm = Matrix::zeros(ndof, ndof);
            for xi in 0..2 {
                for yi in 0..2 {
                    for a in 0..3 {
                        for b in 0..3 {
                            let dab = if a == b { 1.0 } else { 0.0 };
                            let hrel = d.second * u[a] * u[b] + (d.first / r) * (dab - u[a] * u[b]);
                            hm[(3 * xi + a, 3 * yi + b)] += signs[xi] * signs[yi] * hrel;
                        }
                    }
                }
            }
            hm
        };
        let rel = pos[0] - pos[1];
        let r = rel.norm();
        let d = coordination_value_derivatives(kcn, r, rc);
        let g = d.second / r - d.first / (r * r);
        let mut third = vec![Matrix::zeros(ndof, ndof); ndof];
        crate::third_derivative::add_radial_third_block(&mut third, 0, 1, rel, g, d.third, 1.0);
        let h = 1.0e-6;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = pos;
            let mut minus = pos;
            let atom = slab / 3;
            match slab % 3 {
                0 => {
                    plus[atom].x += h;
                    minus[atom].x -= h;
                }
                1 => {
                    plus[atom].y += h;
                    minus[atom].y -= h;
                }
                _ => {
                    plus[atom].z += h;
                    minus[atom].z -= h;
                }
            }
            let hp = cn_hess(&plus);
            let hmn = cn_hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hmn[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "CN pair third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // =======================================================================
    // Phase 4 — frozen (geometry-only) FOURTH-derivative blocks.
    //
    // Every gate below is the SAME statement: the fourth-order block equals the
    // central finite difference of its OWN third-order block,
    //     (third(R + h e_d)[c][(a,b)] − third(R − h e_d)[c][(a,b)]) / 2h
    //       ≈ fourth[c][d][(a,b)].
    // That is the whole correctness contract — it guarantees the Phase-6 quartic
    // assembly can consume these as the exact `∂_d` of the existing third-order
    // objects, asymmetries and deliberate omissions included.
    //
    // There is deliberately NO acoustic-sum-rule test for these blocks. They are
    // frozen-density objects: `P`, `W`, the shell potential and the CN are held at
    // the reference geometry's converged values and are NOT translated with the
    // nuclei, so a rigid translation genuinely changes the block. The third-order
    // blocks have exactly the same property. ASR belongs to the fully assembled,
    // response-complete tensor, not to any frozen ingredient.
    // =======================================================================

    /// Equilibrium water, the standard frozen-block probe geometry.
    const FOURTH_PROBE_EQ: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";
    /// Stretched, fully asymmetric water: no symmetry plane and no equal bond lengths, so every
    /// bra/ket centre pattern of the fourth-order integral ladder carries a distinct value.
    const FOURTH_PROBE_NONEQ: &str =
        "3\nwater-noneq\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.6 0.95 0.1\n";

    /// Builtin params + a tightly converged, dispersion-free `ElectronicResult` for the frozen
    /// fourth-derivative gates. Returns the CN cutoff alongside, since the CN-H0 block needs it.
    fn frozen_fourth_probe(
        xyz: &str,
    ) -> (Gfn1Parameters, PeriodicSystem, ElectronicResult, f64) {
        let params = Gfn1Parameters::builtin().expect("builtin GFN1 parameters");
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, options).unwrap();
        (params, system, electronic, cutoff)
    }

    /// Shared driver for the three block gates: FD the supplied third-derivative closure along
    /// every DOF `d` and compare with `fourth[c][d][(a,b)]`. Returns `(max |Δ|, max |analytic|)`.
    fn fourth_vs_third_fd(
        system: &PeriodicSystem,
        fourth: &[Vec<Matrix>],
        step: f64,
        third: impl Fn(&PeriodicSystem) -> Vec<Matrix>,
    ) -> (f64, f64) {
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        let mut max_value = 0.0_f64;
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, step);
            displace(&mut minus, d, -step);
            let tp = third(&plus);
            let tm = third(&minus);
            for c in 0..ndof {
                for a in 0..ndof {
                    for b in 0..ndof {
                        let fd = (tp[c][(a, b)] - tm[c][(a, b)]) / (2.0 * step);
                        let analytic = fourth[c][d][(a, b)];
                        max_delta = max_delta.max((analytic - fd).abs());
                        max_value = max_value.max(analytic.abs());
                    }
                }
            }
        }
        (max_delta, max_value)
    }

    // -- Task 1a: the distance-polynomial ladder ---------------------------------------------

    // `shell_poly_fourth` must (i) embed `shell_poly_third` bit-for-bit and (ii) have its five
    // fourth-order centre patterns equal the FD of the third-order tensors. This isolates the
    // radial rung (φ'''' via the truncated Faà di Bruno) and the angular rung (the rank-4 hat
    // block + the (−1)^{#bra} sign law) before any integral or density enters.
    #[test]
    fn shell_poly_fourth_matches_third_derivative_fd() {
        let system = PeriodicSystem::from_xyz_str(
            "2\npair\nO 0.0 0.0 0.0\nH 0.83 0.41 0.22\n",
            0.0,
            false,
        )
        .unwrap();
        let (zi, zj) = (8_u8, 1_u8);
        let (pi, pj) = (Some(2.35_f64), Some(-1.17_f64));
        let quad = shell_poly_fourth(&system, 0, 1, zi, zj, pi, pj).unwrap();

        // (i) embedded lower orders are identical to the third-order builder.
        let cubic = shell_poly_third(&system, 0, 1, zi, zj, pi, pj).unwrap();
        let emb = quad.third();
        let mut embed_delta = (emb.value - cubic.value).abs();
        for a in 0..3 {
            embed_delta = embed_delta
                .max((emb.d_bra.to_array()[a] - cubic.d_bra.to_array()[a]).abs())
                .max((emb.d_ket.to_array()[a] - cubic.d_ket.to_array()[a]).abs());
            for b in 0..3 {
                embed_delta = embed_delta
                    .max((emb.h_bra_bra[a][b] - cubic.h_bra_bra[a][b]).abs())
                    .max((emb.h_bra_ket[a][b] - cubic.h_bra_ket[a][b]).abs())
                    .max((emb.h_ket_ket[a][b] - cubic.h_ket_ket[a][b]).abs());
                for c in 0..3 {
                    embed_delta = embed_delta
                        .max((emb.t_bra_bra_bra[a][b][c] - cubic.t_bra_bra_bra[a][b][c]).abs())
                        .max((emb.t_bra_bra_ket[a][b][c] - cubic.t_bra_bra_ket[a][b][c]).abs())
                        .max((emb.t_bra_ket_ket[a][b][c] - cubic.t_bra_ket_ket[a][b][c]).abs())
                        .max((emb.t_ket_ket_ket[a][b][c] - cubic.t_ket_ket_ket[a][b][c]).abs());
                }
            }
        }
        assert!(
            embed_delta == 0.0,
            "shell_poly_fourth does not embed shell_poly_third exactly: {embed_delta:.3e}"
        );

        // (ii) FD of every third-order entry along every DOF.
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        let mut max_value = 0.0_f64;
        for dofd in 0..6 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, dofd, h);
            displace(&mut minus, dofd, -h);
            let tp = shell_poly_third(&plus, 0, 1, zi, zj, pi, pj).unwrap();
            let tm = shell_poly_third(&minus, 0, 1, zi, zj, pi, pj).unwrap();
            // atom 0 is the bra centre, atom 1 the ket centre.
            let cd = if dofd < 3 { Center::Bra } else { Center::Ket };
            let axd = dofd % 3;
            for pattern in 0..4 {
                let centers3 = match pattern {
                    0 => [Center::Bra, Center::Bra, Center::Bra],
                    1 => [Center::Bra, Center::Bra, Center::Ket],
                    2 => [Center::Bra, Center::Ket, Center::Ket],
                    _ => [Center::Ket, Center::Ket, Center::Ket],
                };
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            let pick = |t: &H0Third| -> f64 {
                                third_select(
                                    &t.t_bra_bra_bra,
                                    &t.t_bra_bra_ket,
                                    &t.t_bra_ket_ket,
                                    &t.t_ket_ket_ket,
                                    centers3,
                                    [a, b, c],
                                )
                            };
                            let fd = (pick(&tp) - pick(&tm)) / (2.0 * h);
                            let analytic = fourth_select(
                                &quad.q_bbbb,
                                &quad.q_bbbk,
                                &quad.q_bbkk,
                                &quad.q_bkkk,
                                &quad.q_kkkk,
                                [centers3[0], centers3[1], centers3[2], cd],
                                [a, b, c, axd],
                            );
                            max_delta = max_delta.max((analytic - fd).abs());
                            max_value = max_value.max(analytic.abs());
                        }
                    }
                }
            }
        }
        let rel = max_delta / max_value.max(1.0e-300);
        println!("[phase4] shell_poly_fourth vs FD(third): abs {max_delta:.3e} rel {rel:.3e}");
        assert!(
            rel < 1.0e-7,
            "shell_poly_fourth vs FD(shell_poly_third) relative delta {rel:.3e} (abs {max_delta:.3e})"
        );
    }

    // -- Task 1b/1c: the H0 geometric-scale and EHT-prefactor ladders -------------------------

    // `h0_scale_fourth` (hscale·poly) and `h0_prefactor_fourth` (½(self_i+self_j)·hscale·poly) are
    // the polynomial ladder times a geometry-CONSTANT base, so their fourth-order patterns must
    // FD-match their third-order counterparts on a real basis/shell pair. Runs over every
    // inter-atomic shell pair of water so all `hscale`/`poly_raw` combinations are hit.
    #[test]
    fn h0_scale_and_prefactor_fourth_match_third_fd() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_EQ);
        let basis = &electronic.basis;
        let nsh = basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let h = 1.0e-5;
        let mut worst = [0.0_f64; 2]; // [scale, prefactor] relative deltas
        for (kind, worst_slot) in worst.iter_mut().enumerate() {
            let mut max_delta = 0.0_f64;
            let mut max_value = 0.0_f64;
            for i in 0..nsh {
                for j in 0..i {
                    let (ai, aj) = (basis.shells[i].atom_index, basis.shells[j].atom_index);
                    if ai == aj {
                        continue;
                    }
                    let quad = if kind == 0 {
                        h0_scale_fourth(&system, &params, i, j, basis).unwrap()
                    } else {
                        h0_prefactor_fourth(&system, &params, &electronic, i, j).unwrap()
                    };
                    for dofd in 0..ndof {
                        let atom_d = dofd / 3;
                        let cd = if atom_d == ai {
                            Center::Bra
                        } else if atom_d == aj {
                            Center::Ket
                        } else {
                            continue; // the pair polynomial does not depend on other atoms
                        };
                        let mut plus = system.clone();
                        let mut minus = system.clone();
                        displace(&mut plus, dofd, h);
                        displace(&mut minus, dofd, -h);
                        let third_at = |s: &PeriodicSystem| -> H0Third {
                            if kind == 0 {
                                h0_scale_third(s, &params, i, j, basis).unwrap()
                            } else {
                                h0_prefactor_third(s, &params, &electronic, i, j).unwrap()
                            }
                        };
                        let tp = third_at(&plus);
                        let tm = third_at(&minus);
                        for pattern in 0..4 {
                            let centers3 = match pattern {
                                0 => [Center::Bra, Center::Bra, Center::Bra],
                                1 => [Center::Bra, Center::Bra, Center::Ket],
                                2 => [Center::Bra, Center::Ket, Center::Ket],
                                _ => [Center::Ket, Center::Ket, Center::Ket],
                            };
                            for a in 0..3 {
                                for b in 0..3 {
                                    for c in 0..3 {
                                        let pick = |t: &H0Third| -> f64 {
                                            third_select(
                                                &t.t_bra_bra_bra,
                                                &t.t_bra_bra_ket,
                                                &t.t_bra_ket_ket,
                                                &t.t_ket_ket_ket,
                                                centers3,
                                                [a, b, c],
                                            )
                                        };
                                        let fd = (pick(&tp) - pick(&tm)) / (2.0 * h);
                                        let analytic = fourth_select(
                                            &quad.q_bbbb,
                                            &quad.q_bbbk,
                                            &quad.q_bbkk,
                                            &quad.q_bkkk,
                                            &quad.q_kkkk,
                                            [centers3[0], centers3[1], centers3[2], cd],
                                            [a, b, c, dofd % 3],
                                        );
                                        max_delta = max_delta.max((analytic - fd).abs());
                                        max_value = max_value.max(analytic.abs());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            *worst_slot = max_delta / max_value.max(1.0e-300);
        }
        println!(
            "[phase4] h0_scale_fourth rel {:.3e}, h0_prefactor_fourth rel {:.3e}",
            worst[0], worst[1]
        );
        assert!(
            worst[0] < 1.0e-7,
            "h0_scale_fourth vs FD(h0_scale_third) relative delta {:.3e}",
            worst[0]
        );
        assert!(
            worst[1] < 1.0e-7,
            "h0_prefactor_fourth vs FD(h0_prefactor_third) relative delta {:.3e}",
            worst[1]
        );
    }

    // -- Task 1d: the coordination-number ladders --------------------------------------------

    // Scalar rung: the CN counting function's FOURTH radial derivative (sigmoid chain rule with
    // `σ₄ = σ(1−σ)(1 − 14σ + 36σ² − 24σ³)`) versus the central FD of its third derivative.
    #[test]
    fn coordination_fourth_derivative_matches_third_finite_difference() {
        let kcn = CoordinationOptions::default().kcn;
        let h = 1.0e-6;
        let mut max_rel = 0.0_f64;
        for &rc in &[3.0_f64, 4.5] {
            for &r in &[1.2_f64, 2.5, 4.0, 6.0] {
                let d = coordination_value_derivatives(kcn, r, rc);
                let fd = (coordination_value_derivatives(kcn, r + h, rc).third
                    - coordination_value_derivatives(kcn, r - h, rc).third)
                    / (2.0 * h);
                max_rel = max_rel.max((d.fourth - fd).abs() / d.fourth.abs().max(1.0e-300));
            }
        }
        println!("[phase4] CN counting f'''' vs FD(f'''): rel {max_rel:.3e}");
        assert!(
            max_rel < 1.0e-7,
            "CN counting-function fourth derivative vs FD relative delta {max_rel:.3e}"
        );
    }

    // Jet rung: both fourth-order flat jets must embed their third-order builders exactly and
    // have `.fourth` equal the FD of `.third`. Frozen density (the reference `electronic` is
    // reused at displaced geometries), so the `d_edcn` jet's FD is a pure geometry derivative.
    #[test]
    fn cn_h0_fourth_jets_match_third_jets_fd() {
        let (params, system, electronic, cutoff) = frozen_fourth_probe(FOURTH_PROBE_EQ);
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let de4 = cn_h0_dedcn_jets_fourth(&system, &params, &electronic).unwrap();
        let cn4 = cn_h0_cn_jets_fourth(&system, cutoff).unwrap();
        let de3 = cn_h0_dedcn_jets(&system, &params, &electronic).unwrap();
        let cn3 = cn_h0_cn_jets(&system, cutoff).unwrap();

        // Embedding: the fourth-order builders reproduce the third-order ones exactly.
        let mut embed = 0.0_f64;
        for atom in 0..nat {
            for (a, b) in [(de4[atom].to_third(), &de3[atom]), (cn4[atom].to_third(), &cn3[atom])] {
                embed = embed.max((a.value - b.value).abs());
                for k in 0..ndof {
                    embed = embed.max((a.grad[k] - b.grad[k]).abs());
                }
                for k in 0..ndof * ndof {
                    embed = embed.max((a.hess[k] - b.hess[k]).abs());
                }
                for k in 0..ndof * ndof * ndof {
                    embed = embed.max((a.third[k] - b.third[k]).abs());
                }
            }
        }
        assert!(
            embed == 0.0,
            "fourth-order jets do not embed the third-order jets exactly: {embed:.3e}"
        );

        // FD ladder: `.fourth` == ∂_d `.third`.
        let h = 1.0e-5;
        let (mut d_delta, mut d_value) = (0.0_f64, 0.0_f64);
        let (mut c_delta, mut c_value) = (0.0_f64, 0.0_f64);
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, h);
            displace(&mut minus, d, -h);
            let dep = cn_h0_dedcn_jets(&plus, &params, &electronic).unwrap();
            let dem = cn_h0_dedcn_jets(&minus, &params, &electronic).unwrap();
            let cnp = cn_h0_cn_jets(&plus, cutoff).unwrap();
            let cnm = cn_h0_cn_jets(&minus, cutoff).unwrap();
            for atom in 0..nat {
                for t in 0..ndof * ndof * ndof {
                    let fd_de = (dep[atom].third[t] - dem[atom].third[t]) / (2.0 * h);
                    let an_de = de4[atom].fourth[t * ndof + d];
                    d_delta = d_delta.max((an_de - fd_de).abs());
                    d_value = d_value.max(an_de.abs());
                    let fd_cn = (cnp[atom].third[t] - cnm[atom].third[t]) / (2.0 * h);
                    let an_cn = cn4[atom].fourth[t * ndof + d];
                    c_delta = c_delta.max((an_cn - fd_cn).abs());
                    c_value = c_value.max(an_cn.abs());
                }
            }
        }
        let d_rel = d_delta / d_value.max(1.0e-300);
        let c_rel = c_delta / c_value.max(1.0e-300);
        println!(
            "[phase4] dedcn jet4 rel {d_rel:.3e} (abs {d_delta:.3e}); cn jet4 rel {c_rel:.3e} (abs {c_delta:.3e})"
        );
        assert!(d_rel < 1.0e-7, "d_edcn `.fourth` vs FD(`.third`) rel {d_rel:.3e}");
        assert!(c_rel < 1.0e-7, "CN `.fourth` vs FD(`.third`) rel {c_rel:.3e}");
    }

    // The per-shell SCC scalar-potential THIRD derivative (new ladder rung consumed by the
    // scalar-overlap quartic block) versus the central FD of its own second derivative, at fixed
    // charges.
    #[test]
    fn shell_scalar_potential_third_derivatives_match_second_fd() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let basis = &electronic.basis;
        let q = &electronic.shell_charges;
        let nsh = basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let third = shell_scalar_potential_third_derivatives(&system, basis, q, &params).unwrap();
        let h = 1.0e-5;
        let (mut max_delta, mut max_value) = (0.0_f64, 0.0_f64);
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, h);
            displace(&mut minus, d, -h);
            let sp = shell_scalar_potential_second_derivatives(&plus, basis, q, &params).unwrap();
            let sm = shell_scalar_potential_second_derivatives(&minus, basis, q, &params).unwrap();
            for sh in 0..nsh {
                for b in 0..ndof {
                    for c in 0..ndof {
                        let fd = (sp[sh][(b, c)] - sm[sh][(b, c)]) / (2.0 * h);
                        let analytic = third[sh][(b * ndof + c) * ndof + d];
                        max_delta = max_delta.max((analytic - fd).abs());
                        max_value = max_value.max(analytic.abs());
                    }
                }
            }
        }
        let rel = max_delta / max_value.max(1.0e-300);
        println!("[phase4] shell scalar potential ∂³V rel {rel:.3e} (abs {max_delta:.3e})");
        assert!(
            rel < 1.0e-7,
            "shell scalar-potential third derivative vs FD relative delta {rel:.3e}"
        );
    }

    // -- Task 2: the frozen Pulay/band/H0/overlap quartic block -------------------------------

    fn pulay_fourth_gate(xyz: &str) -> (f64, f64) {
        let (params, system, electronic, _) = frozen_fourth_probe(xyz);
        let fourth = fixed_density_pulay_fourth_derivative(&system, &params, &electronic).unwrap();
        fourth_vs_third_fd(&system, &fourth, 1.0e-4, |s| {
            fixed_density_pulay_third_derivative(s, &params, &electronic).unwrap()
        })
    }

    #[test]
    fn fixed_density_pulay_fourth_derivative_matches_third_fd_equilibrium() {
        let (delta, value) = pulay_fourth_gate(FOURTH_PROBE_EQ);
        println!("[phase4] pulay⁴ eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen Pulay fourth-derivative FD-vs-third max delta {delta:.3e}"
        );
    }

    #[test]
    fn fixed_density_pulay_fourth_derivative_matches_third_fd_stretched() {
        let (delta, value) = pulay_fourth_gate(FOURTH_PROBE_NONEQ);
        println!("[phase4] pulay⁴ non-eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen Pulay fourth-derivative FD-vs-third max delta (stretched) {delta:.3e}"
        );
    }

    // The Pulay quartic block is the true fourth derivative of a scalar (P, W, V and the frozen
    // self-energies are geometry constants), so unlike the other two blocks it must be fully
    // index-symmetric. Cheap structural check that the `distinct_perms4` scatter is complete.
    #[test]
    fn fixed_density_pulay_fourth_derivative_is_index_symmetric() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let q = fixed_density_pulay_fourth_derivative(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let mut max_asym = 0.0_f64;
        let mut scale = 0.0_f64;
        for a in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    for d in 0..ndof {
                        let base = q[c][d][(a, b)];
                        scale = scale.max(base.abs());
                        max_asym = max_asym
                            .max((base - q[c][d][(b, a)]).abs())
                            .max((base - q[d][c][(a, b)]).abs())
                            .max((base - q[a][d][(c, b)]).abs())
                            .max((base - q[c][a][(d, b)]).abs());
                    }
                }
            }
        }
        assert!(
            max_asym <= 1.0e-9 * scale.max(1.0),
            "Pulay quartic block is not index-symmetric: {max_asym:.3e} (scale {scale:.3e})"
        );
    }

    // -- Task 3: the frozen CN-H0 quartic block ----------------------------------------------

    fn cn_h0_fourth_gate(xyz: &str) -> (f64, f64) {
        let (params, system, electronic, cutoff) = frozen_fourth_probe(xyz);
        let fourth =
            fixed_density_cn_h0_fourth_derivative(&system, &params, &electronic, cutoff).unwrap();
        fourth_vs_third_fd(&system, &fourth, 1.0e-4, |s| {
            fixed_density_cn_h0_third_derivative(s, &params, &electronic, cutoff).unwrap()
        })
    }

    #[test]
    fn fixed_density_cn_h0_fourth_derivative_matches_third_fd_equilibrium() {
        let (delta, value) = cn_h0_fourth_gate(FOURTH_PROBE_EQ);
        println!("[phase4] cn-h0⁴ eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen CN-H0 fourth-derivative FD-vs-third max delta {delta:.3e}"
        );
    }

    #[test]
    fn fixed_density_cn_h0_fourth_derivative_matches_third_fd_stretched() {
        let (delta, value) = cn_h0_fourth_gate(FOURTH_PROBE_NONEQ);
        println!("[phase4] cn-h0⁴ non-eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen CN-H0 fourth-derivative FD-vs-third max delta (stretched) {delta:.3e}"
        );
    }

    // -- Task 4: the frozen SCC-scalar × dS quartic block -------------------------------------

    fn scalar_overlap_fourth_gate(xyz: &str) -> (f64, f64) {
        let (params, system, electronic, _) = frozen_fourth_probe(xyz);
        let fourth =
            fixed_density_scalar_overlap_fourth_derivative(&system, &params, &electronic).unwrap();
        fourth_vs_third_fd(&system, &fourth, 1.0e-4, |s| {
            fixed_density_scalar_overlap_third_derivative(s, &params, &electronic).unwrap()
        })
    }

    #[test]
    fn fixed_density_scalar_overlap_fourth_derivative_matches_third_fd_equilibrium() {
        let (delta, value) = scalar_overlap_fourth_gate(FOURTH_PROBE_EQ);
        println!("[phase4] scalar-overlap⁴ eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen scalar-overlap fourth-derivative FD-vs-third max delta {delta:.3e}"
        );
    }

    #[test]
    fn fixed_density_scalar_overlap_fourth_derivative_matches_third_fd_stretched() {
        let (delta, value) = scalar_overlap_fourth_gate(FOURTH_PROBE_NONEQ);
        println!("[phase4] scalar-overlap⁴ non-eq: abs {delta:.3e} (max |Q| {value:.3e})");
        assert!(
            delta < 2.0e-6,
            "frozen scalar-overlap fourth-derivative FD-vs-third max delta (stretched) {delta:.3e}"
        );
    }

    fn displace(system: &mut PeriodicSystem, dof: usize, step: f64) {
        let atom = dof / 3;
        match dof % 3 {
            0 => system.atoms[atom].position.x += step,
            1 => system.atoms[atom].position.y += step,
            _ => system.atoms[atom].position.z += step,
        }
    }

    // =======================================================================
    // Directional THIRD-derivative AO-matrix builders (quartic response stage)
    //
    // Every gate central-differences ALONG `v` the corresponding SECOND-derivative matrix
    // built DIRECTIONALLY at the displaced geometry,
    //     `M2_dir(sys) = Σ_bc v_b v_c second_matrix(sys, b, c)`,
    // at a FROZEN electronic reference (these are skeleton objects — nothing is reconverged),
    // for `h` and `h/2`, and asserts (i) the analytic directional third matrix reproduces it
    // and (ii) the residual is truncation-dominated, `delta(h/2) < 0.4·delta(h)` (h² scaling).
    // Probe geometry: the stretched, fully asymmetric water (`FOURTH_PROBE_NONEQ`), so every
    // bra/ket centre pattern of the third-order ladder carries a distinct value.
    // =======================================================================

    /// Fixed pseudo-random unit direction — no symmetry alignment, every DOF active.
    fn probe_direction(ndof: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..ndof)
            .map(|i| {
                let x = i as f64;
                (0.7371 * x + 0.31).sin() + 0.37 * (1.911 * x + 1.07).cos()
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn displaced_along(system: &PeriodicSystem, v: &[f64], step: f64) -> PeriodicSystem {
        let mut out = system.clone();
        for (dof, &vb) in v.iter().enumerate() {
            displace(&mut out, dof, step * vb);
        }
        out
    }

    fn max_abs_matrix(m: &Matrix) -> f64 {
        let mut out = 0.0_f64;
        for i in 0..m.rows() {
            for j in 0..m.cols() {
                out = out.max(m[(i, j)].abs());
            }
        }
        out
    }

    /// `(delta(h), delta(h/2))`: max-abs elementwise difference between the analytic
    /// directional third matrix and the central FD along `v` of the directional second matrix.
    fn directional_third_fd(
        system: &PeriodicSystem,
        v: &[f64],
        third: &Matrix,
        h: f64,
        second_dir: impl Fn(&PeriodicSystem) -> Matrix,
    ) -> (f64, f64) {
        let mut deltas = [0.0_f64; 2];
        for (slot, step) in [h, 0.5 * h].into_iter().enumerate() {
            let plus = second_dir(&displaced_along(system, v, step));
            let minus = second_dir(&displaced_along(system, v, -step));
            let mut worst = 0.0_f64;
            for i in 0..third.rows() {
                for j in 0..third.cols() {
                    let fd = (plus[(i, j)] - minus[(i, j)]) / (2.0 * step);
                    worst = worst.max((third[(i, j)] - fd).abs());
                }
            }
            deltas[slot] = worst;
        }
        (deltas[0], deltas[1])
    }

    /// `Σ_bc v_b v_c block(sys, b, c)` for an n×n per-DOF-pair builder.
    fn contract_second_matrix_vv(
        n: usize,
        ndof: usize,
        v: &[f64],
        block: impl Fn(usize, usize) -> Matrix,
    ) -> Matrix {
        let mut out = Matrix::zeros(n, n);
        for b in 0..ndof {
            if v[b] == 0.0 {
                continue;
            }
            for c in 0..ndof {
                let w = v[b] * v[c];
                if w == 0.0 {
                    continue;
                }
                let m = block(b, c);
                for i in 0..n {
                    for j in 0..n {
                        out[(i, j)] += w * m[(i, j)];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn directional_h0_bare_third_matches_directional_second_fd() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let v = probe_direction(ndof);
        let third = directional_h0_bare_third_matrix(&system, &params, &electronic, &v).unwrap();
        let (d1, d2) = directional_third_fd(&system, &v, &third, 1.0e-4, |sys| {
            contract_second_matrix_vv(n, ndof, &v, |b, c| {
                h0_bare_second_derivative_matrix(sys, &params, &electronic, b, c).unwrap()
            })
        });
        let scale = max_abs_matrix(&third);
        println!(
            "[phase6] h0-bare³ directional: abs(h) {d1:.3e} abs(h/2) {d2:.3e} \
             ratio {:.3} (max |M3| {scale:.3e})",
            d2 / d1.max(1.0e-300)
        );
        assert!(
            d1 < 1.0e-7 && d2 < 1.0e-7,
            "directional bare-H0 third vs FD(second) deltas {d1:.3e} / {d2:.3e}"
        );
        assert!(
            d2 < 0.4 * d1,
            "directional bare-H0 third FD residual does not scale like h²: {d1:.3e} -> {d2:.3e}"
        );
    }

    #[test]
    fn directional_h0_cn_block_third_matches_directional_second_fd() {
        let (params, system, electronic, cutoff) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let v = probe_direction(ndof);
        let third =
            directional_h0_cn_block_third_matrix(&system, &params, &electronic, cutoff, &v).unwrap();
        let (d1, d2) = directional_third_fd(&system, &v, &third, 1.0e-4, |sys| {
            contract_second_matrix_vv(n, ndof, &v, |b, c| {
                h0_cn_block_second_derivative_matrix(sys, &params, &electronic, cutoff, b, c)
                    .unwrap()
            })
        });
        let scale = max_abs_matrix(&third);
        println!(
            "[phase6] h0-cn-block³ directional: abs(h) {d1:.3e} abs(h/2) {d2:.3e} \
             ratio {:.3} (max |M3| {scale:.3e})",
            d2 / d1.max(1.0e-300)
        );
        assert!(
            d1 < 1.0e-7 && d2 < 1.0e-7,
            "directional CN-block third vs FD(second) deltas {d1:.3e} / {d2:.3e}"
        );
        assert!(
            d2 < 0.4 * d1,
            "directional CN-block third FD residual does not scale like h²: {d1:.3e} -> {d2:.3e}"
        );
    }

    /// The pure-GEOMETRIC directional potential legs at `sys` (zero charge response):
    /// `v_c = ∂V/∂R|_q·v` and `v_cc = ∂²V/∂R²|_q:vv`, both at the frozen reference charges.
    /// Rebuilt from each displaced geometry so that `v_cc` really is `D v_c`.
    fn geometric_potential_legs(
        sys: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &ElectronicResult,
        v: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let basis = &electronic.basis;
        let nsh = basis.shells.len();
        let ndof = v.len();
        let d1 =
            shell_scalar_potential_first_derivatives(sys, basis, &electronic.shell_charges, params)
                .unwrap();
        let d2 =
            shell_scalar_potential_second_derivatives(sys, basis, &electronic.shell_charges, params)
                .unwrap();
        let mut v_c = vec![0.0; nsh];
        let mut v_cc = vec![0.0; nsh];
        for (s, (leg1, leg2)) in v_c.iter_mut().zip(v_cc.iter_mut()).enumerate() {
            for b in 0..ndof {
                *leg1 += v[b] * d1[(s, b)];
                for c in 0..ndof {
                    *leg2 += v[b] * v[c] * d2[s][(b, c)];
                }
            }
        }
        (v_c, v_cc)
    }

    // Pure-geometric channel of the SCC-scalar third: the caller-supplied response legs are set
    // to the GEOMETRIC directional potential derivatives (`q_c = q_cc = 0`) and rebuilt from each
    // displaced geometry, so the FD reference is a self-consistent function of geometry alone.
    //
    // The directional second reference cannot be a plain `Σ_bc v_b v_c M2(b,c)`: the supplied
    // `v_c`/`q_c` legs are already contracted and do NOT carry the `c` index, so summing them
    // over `c` would multiply them by `Σ_c v_c`. Since `M2` is AFFINE in the legs and its leg
    // part is independent of `c`, the exact directional object is the leg-free part contracted
    // `vv` plus the leg part (a `c`-independent difference at any fixed `c`, here `c = 0`)
    // contracted over `b` only.
    #[test]
    fn directional_h0_scc_scalar_third_matches_directional_second_fd_geometric_channel() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let nsh = electronic.basis.shells.len();
        let v = probe_direction(ndof);
        let zero = vec![0.0; nsh];
        let (v_c, v_cc) = geometric_potential_legs(&system, &params, &electronic, &v);
        let third = directional_h0_scc_scalar_third_matrix(
            &system,
            &params,
            &electronic,
            &v,
            &v_c,
            &v_cc,
            &zero,
            &zero,
        )
        .unwrap();
        let (d1, d2) = directional_third_fd(&system, &v, &third, 1.0e-4, |sys| {
            let (leg_v_c, _) = geometric_potential_legs(sys, &params, &electronic, &v);
            let mut out = contract_second_matrix_vv(n, ndof, &v, |b, c| {
                h0_scc_scalar_second_derivative_matrix(
                    sys,
                    &params,
                    &electronic,
                    &zero,
                    &zero,
                    b,
                    c,
                )
                .unwrap()
            });
            for b in 0..ndof {
                if v[b] == 0.0 {
                    continue;
                }
                let with = h0_scc_scalar_second_derivative_matrix(
                    sys, &params, &electronic, &leg_v_c, &zero, b, 0,
                )
                .unwrap();
                let without = h0_scc_scalar_second_derivative_matrix(
                    sys, &params, &electronic, &zero, &zero, b, 0,
                )
                .unwrap();
                for i in 0..n {
                    for j in 0..n {
                        out[(i, j)] += v[b] * (with[(i, j)] - without[(i, j)]);
                    }
                }
            }
            out
        });
        let scale = max_abs_matrix(&third);
        println!(
            "[phase6] h0-scc-scalar³ directional (geometric channel): abs(h) {d1:.3e} \
             abs(h/2) {d2:.3e} ratio {:.3} (max |M3| {scale:.3e})",
            d2 / d1.max(1.0e-300)
        );
        assert!(
            d1 < 1.0e-7 && d2 < 1.0e-7,
            "directional SCC-scalar third vs FD(second) deltas {d1:.3e} / {d2:.3e}"
        );
        assert!(
            d2 < 0.4 * d1,
            "directional SCC-scalar third FD residual does not scale like h²: {d1:.3e} -> {d2:.3e}"
        );
    }

    // The response legs enter LINEARLY, so the FD gate above (which exercises only the
    // geometric channel) is completed by an affinity check on all four supplied legs:
    // `M3(x+y) = M3(x) + M3(y) − M3(0)`.
    #[test]
    fn directional_h0_scc_scalar_third_is_affine_in_the_supplied_legs() {
        let (params, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let ndof = 3 * system.atoms.len();
        let nsh = electronic.basis.shells.len();
        let v = probe_direction(ndof);
        let zero = vec![0.0; nsh];
        let leg = |seed: f64| -> Vec<f64> {
            (0..nsh)
                .map(|i| 0.21 * (seed * (i as f64 + 1.3)).sin())
                .collect()
        };
        let add = |a: &[f64], b: &[f64]| -> Vec<f64> {
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        };
        let build = |v_c: &[f64], v_cc: &[f64], q_c: &[f64], q_cc: &[f64]| {
            directional_h0_scc_scalar_third_matrix(
                &system,
                &params,
                &electronic,
                &v,
                v_c,
                v_cc,
                q_c,
                q_cc,
            )
            .unwrap()
        };
        let (x1, x2, x3, x4) = (leg(1.7), leg(2.3), leg(3.1), leg(4.7));
        let (y1, y2, y3, y4) = (leg(5.3), leg(6.1), leg(7.9), leg(8.3));
        let m0 = build(&zero, &zero, &zero, &zero);
        let mx = build(&x1, &x2, &x3, &x4);
        let my = build(&y1, &y2, &y3, &y4);
        let mxy = build(
            &add(&x1, &y1),
            &add(&x2, &y2),
            &add(&x3, &y3),
            &add(&x4, &y4),
        );
        let mut worst = 0.0_f64;
        let mut spread = 0.0_f64;
        for i in 0..m0.rows() {
            for j in 0..m0.cols() {
                let lhs = mxy[(i, j)];
                let rhs = mx[(i, j)] + my[(i, j)] - m0[(i, j)];
                worst = worst.max((lhs - rhs).abs());
                spread = spread.max((mx[(i, j)] - m0[(i, j)]).abs());
            }
        }
        println!("[phase6] h0-scc-scalar³ leg affinity: {worst:.3e} (leg swing {spread:.3e})");
        assert!(
            spread > 1.0e-4,
            "the supplied legs barely move the matrix ({spread:.3e}) — affinity check is vacuous"
        );
        assert!(
            worst < 1.0e-12,
            "directional SCC-scalar third is not affine in the supplied legs: {worst:.3e}"
        );
    }

    #[test]
    fn directional_overlap_third_matches_directional_second_fd() {
        let (_, system, electronic, _) = frozen_fourth_probe(FOURTH_PROBE_NONEQ);
        let basis = &electronic.basis;
        let ndof = 3 * system.atoms.len();
        let n = basis.len();
        let v = probe_direction(ndof);
        let third = directional_overlap_third_matrix(&system, basis, &v).unwrap();
        let (d1, d2) = directional_third_fd(&system, &v, &third, 1.0e-4, |sys| {
            contract_second_matrix_vv(n, ndof, &v, |b, c| {
                crate::response::cpxtb::overlap_second_derivative_matrix(sys, basis, b, c).unwrap()
            })
        });
        let scale = max_abs_matrix(&third);
        println!(
            "[phase6] overlap³ directional: abs(h) {d1:.3e} abs(h/2) {d2:.3e} \
             ratio {:.3} (max |M3| {scale:.3e})",
            d2 / d1.max(1.0e-300)
        );
        assert!(
            d1 < 1.0e-7 && d2 < 1.0e-7,
            "directional overlap third vs FD(second) deltas {d1:.3e} / {d2:.3e}"
        );
        assert!(
            d2 < 0.4 * d1,
            "directional overlap third FD residual does not scale like h²: {d1:.3e} -> {d2:.3e}"
        );
    }
