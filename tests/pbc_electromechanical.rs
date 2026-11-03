// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::magnetic::{boosted_overlap_pair, boosted_overlap_pair_derivatives};
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::mechanical::homogeneous_deformation;
use gfn1_rs::pbc::polarization::response::pbc_polarization_derivatives;
use gfn1_rs::pbc::polarization::{pbc_berry_polarization, BerryPolarizationOptions};
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::{BasisOptions, BasisSet, ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use std::f64::consts::PI;
use std::sync::Arc;

#[test]
fn boosted_overlap_exact_centres_and_reciprocal_gradient() {
    let sys =
        PeriodicSystem::from_xyz_str("2\nSiH\nSi .1 -.2 .3\nH .9 .2 -.1\n", 0., false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let basis = BasisSet::build(&sys, &params, BasisOptions::default()).unwrap();
    let ca = Vec3::new(0.2, -0.3, 0.1);
    let cb = Vec3::new(1.7, 0.4, -0.6);
    let q = Vec3::new(0.2, -0.13, 0.07);
    let h = 2e-5;
    for a in &basis.aos {
        for b in basis.aos.iter().step_by(3) {
            let (value, da, db, dq) = boosted_overlap_pair_derivatives(a, b, ca, cb, q);
            let primal = boosted_overlap_pair(a, b, ca, cb, q);
            assert!((value.0 - primal.0).abs().max((value.1 - primal.1).abs()) < 1e-13);
            for axis in 0..3 {
                let mut r = [0.; 3];
                r[axis] = h;
                let delta = Vec3::new(r[0], r[1], r[2]);
                for (slot, expected) in [(0, da[axis]), (1, db[axis]), (2, dq[axis])] {
                    let plus = boosted_overlap_pair(
                        a,
                        b,
                        ca + if slot == 0 { delta } else { Vec3::zero() },
                        cb + if slot == 1 { delta } else { Vec3::zero() },
                        q + if slot == 2 { delta } else { Vec3::zero() },
                    );
                    let minus = boosted_overlap_pair(
                        a,
                        b,
                        ca - if slot == 0 { delta } else { Vec3::zero() },
                        cb - if slot == 1 { delta } else { Vec3::zero() },
                        q - if slot == 2 { delta } else { Vec3::zero() },
                    );
                    assert!(
                        ((plus.0 - minus.0) / (2. * h) - expected.0).abs() < 2e-8,
                        "real boost derivative slot{slot} axis{axis}"
                    );
                    assert!(
                        ((plus.1 - minus.1) / (2. * h) - expected.1).abs() < 2e-8,
                        "imag boost derivative slot{slot} axis{axis}"
                    );
                }
                let q = q.to_array()[axis];
                assert!((da[axis].0 + db[axis].0 + q * value.1).abs() < 2e-12);
                assert!((da[axis].1 + db[axis].1 - q * value.0).abs() < 2e-12);
            }
        }
    }
}

fn audit(learned: bool) {
    let system=PeriodicSystem::from_xyz_str("2\nLattice=\"3.4 .1 0 .2 3.6 .1 0 .2 3.7\" pbc=\"T T T\"\nF .12 -.07 .11\nH 1.01 .23 -.04\n",0.,false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut options = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-14,
        charge_tolerance: 1e-12,
        max_scc: 500,
        ..Default::default()
    };
    if learned {
        let mut m = gfn1_rs::ml::MlHamiltonian::zero(vec![1, 9], vec![1.3, 3.0, 5.0], 8.);
        for (z, s) in [(1, 1.), (9, -1.)] {
            for l in 0..3 {
                m.heads.get_mut(&z).unwrap().levels[l].weights[0] = s * 0.008 * (l + 1) as f64;
                m.heads.get_mut(&z).unwrap().levels[l].weights[4] = -s * 0.006;
            }
        }
        options.ml_model = Some(Arc::new(m));
    }
    let pbc = PbcOptions {
        ao_cutoff: 9.,
        kmesh: KMesh {
            size: [2, 1, 1],
            gamma_centered: true,
            fold_time_reversal: false,
        },
        ..Default::default()
    };
    let berry = BerryPolarizationOptions {
        mesh: [2, 1, 1],
        ..Default::default()
    };
    let exact = pbc_polarization_derivatives(&system, &params, &options, &pbc, &berry).unwrap();
    assert_eq!(exact.ml_model_sha256.is_some(), learned);
    let lattice = system.lattice.unwrap();
    let h = 1e-4;
    let mut max_born = 0.0_f64;
    let mut max_piezo = 0.0_f64;
    for y in 0..12 {
        let mut numerical = [[0.; 3]; 2];
        for (ih, step) in [h, h / 2.].into_iter().enumerate() {
            let mut shifted = Vec::new();
            for sign in [1., -1.] {
                let trial = if y < 6 {
                    let mut s = system.clone();
                    let p = s.atoms[y / 3].position.to_array();
                    let mut p = p;
                    p[y % 3] += sign * step;
                    s.atoms[y / 3].position = Vec3::new(p[0], p[1], p[2]);
                    s
                } else {
                    let mut strain = [0.; 6];
                    strain[y - 6] = sign * step;
                    homogeneous_deformation(&system, strain).unwrap()
                };
                shifted
                    .push(pbc_berry_polarization(&trial, &params, &options, &pbc, &berry).unwrap());
            }
            for d in 0..3 {
                let mut dp = shifted[0].total_phase_raw[d] - shifted[1].total_phase_raw[d];
                dp -= 4. * PI * (dp / (4. * PI)).round();
                for i in 0..3 {
                    numerical[ih][i] += lattice.cell.col[d].to_array()[i] * dp / (4. * PI * step);
                }
            }
        }
        for i in 0..3 {
            let fd = (4. * numerical[1][i] - numerical[0][i]) / 3.;
            if y < 6 {
                max_born =
                    max_born.max((fd - exact.born_effective_charges[y / 3][(i, y % 3)]).abs());
            } else {
                max_piezo = max_piezo
                    .max((fd / lattice.volume() - exact.clamped_piezoelectric[(i, y - 6)]).abs());
            }
        }
    }
    eprintln!("Berry analytic ML={learned}: all 18 Born +18 proper piezo entries max errors {max_born:e} / {max_piezo:e}");
    assert!(max_born < 5e-7, "Born {max_born:e}");
    assert!(max_piezo < 2e-8, "piezo {max_piezo:e}");
    assert!(exact
        .born_acoustic_sum_residual
        .as_slice()
        .iter()
        .all(|x| x.abs() < 1e-8));
}
#[test]
fn analytic_berry_all_components_scalar() {
    audit(false);
}
#[test]
fn analytic_berry_all_components_structural_ml() {
    audit(true);
}

#[test]
fn polar_modes_lo_to_and_ionic_dielectric_obey_lst() {
    use gfn1_rs::linalg::Matrix;
    use gfn1_rs::pbc::electromechanical::{lo_to_modes, polar_mode_properties};
    let masses = [1., 2.];
    let volume = 100.;
    let spring = 0.1;
    let mut h = Matrix::zeros(6, 6);
    for i in 0..3 {
        h[(i, i)] = spring;
        h[(i + 3, i + 3)] = spring;
        h[(i, i + 3)] = -spring;
        h[(i + 3, i)] = -spring;
    }
    let z0 = Matrix::identity(3);
    let mut z1 = z0.clone();
    for v in z1.as_mut_slice() {
        *v = -*v;
    }
    let born = [z0, z1];
    let mut eps = Matrix::identity(3);
    for i in 0..3 {
        eps[(i, i)] = 2.;
    }
    let modes = polar_mode_properties(&h, &born, &masses, volume, 1e-10).unwrap();
    let ionic = modes.ionic_dielectric.unwrap();
    let expected = 4. * PI / (volume * spring);
    for i in 0..3 {
        for j in 0..3 {
            assert!((ionic[(i, j)] - if i == j { expected } else { 0. }).abs() < 1e-12);
        }
    }
    let lo = lo_to_modes(&h, &born, &eps, volume, &masses, [2., 0., 0.]).unwrap();
    let to = spring * (1. / masses[0] + 1. / masses[1]);
    assert!((lo.eigenvalues[5] / to - (1. + expected / 2.)).abs() < 1e-12);
    assert!((lo.eigenvalues[3] - to).abs() < 1e-12 && (lo.eigenvalues[4] - to).abs() < 1e-12);
    let mut unstable = h.clone();
    for v in unstable.as_mut_slice() {
        *v = -*v;
    }
    let unstable = polar_mode_properties(&unstable, &born, &masses, volume, 1e-10).unwrap();
    assert!(unstable.ionic_dielectric.is_none() && unstable.ionic_dielectric_error.is_some());
}

#[test]
fn berry_dielectric_keeps_reciprocity_and_learned_reference() {
    use gfn1_rs::pbc::polarization::response::pbc_electronic_dielectric;
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"10 0 0 0 11 0 0 0 12\" pbc=\"T T T\"\nH .1 .2 .3\nH .83 .36 .41\n",
        0.,
        false,
    )
    .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut options = ElectronicOptions::default();
    options.energy_tolerance = 1e-14;
    options.charge_tolerance = 1e-12;
    let pbc = PbcOptions {
        ao_cutoff: 9.,
        ..Default::default()
    };
    let berry = BerryPolarizationOptions::default();
    let plain = pbc_electronic_dielectric(&system, &params, &options, &pbc, &berry).unwrap();
    assert!(plain.raw_asymmetry < 1e-10);
    assert!(plain.minimum_susceptibility_eigenvalue > -1e-10);
    let mut m = gfn1_rs::ml::MlHamiltonian::zero(vec![1], vec![1.2, 2.5], 6.);
    m.heads.get_mut(&1).unwrap().levels[0].weights[0] = 0.03;
    options.ml_model = Some(Arc::new(m));
    let learned = pbc_electronic_dielectric(&system, &params, &options, &pbc, &berry).unwrap();
    assert!(learned.raw_asymmetry < 1e-10);
    assert!(learned.minimum_susceptibility_eigenvalue > -1e-10);
    assert!(learned.ml_model_sha256.is_some());
    assert!(plain
        .relative_dielectric
        .as_slice()
        .iter()
        .zip(learned.relative_dielectric.as_slice())
        .any(|(a, b)| (a - b).abs() > 1e-8));
    eprintln!(
        "Berry dielectric plain {:?}, ML {:?}, asym {} / {}",
        plain.relative_dielectric,
        learned.relative_dielectric,
        plain.raw_asymmetry,
        learned.raw_asymmetry
    );
}

#[test]
fn all_dielectric_entries_match_independent_finite_field_variational_minima() {
    use gfn1_rs::coulomb::coulomb_energy_potential_from_matrix;
    use gfn1_rs::linalg::{DenseLu, Matrix};
    use gfn1_rs::pbc::hessian::kpoint_complex_mos;
    use gfn1_rs::pbc::polarization::response::pbc_electronic_dielectric;
    use gfn1_rs::pbc::run_pbc_scc;
    let sys = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"10 0 0 0 11 0 0 0 12\" pbc=\"T T T\"\nH .1 .2 .3\nH .83 .36 .41\n",
        0.,
        false,
    )
    .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions {
        ao_cutoff: 9.,
        ..Default::default()
    };
    let berry = BerryPolarizationOptions::default();
    let lattice = sys.lattice.unwrap();
    let volume = lattice.volume();
    for learned in [false, true] {
        let mut opts = ElectronicOptions {
            energy_tolerance: 1e-14,
            charge_tolerance: 1e-13,
            ..Default::default()
        };
        if learned {
            let mut m = gfn1_rs::ml::MlHamiltonian::zero(vec![1], vec![1.2, 2.5], 6.);
            m.heads.get_mut(&1).unwrap().levels[0].weights[0] = 0.03;
            opts.ml_model = Some(Arc::new(m));
        }
        let analytic = pbc_electronic_dielectric(&sys, &params, &opts, &pbc, &berry).unwrap();
        let scf = run_pbc_scc(&sys, &params, &opts, &pbc).unwrap();
        let mos = kpoint_complex_mos(&scf, 0).unwrap();
        let n = scf.basis.len();
        let nv = n - 1;
        assert!(
            mos.occupations[0] > 2. - 1e-12 && mos.occupations[1..].iter().all(|x| x.abs() < 1e-12)
        );
        let mut c = Matrix::zeros(n, n);
        for b in 0..n {
            let p = (0..n)
                .max_by(|&i, &j| {
                    let v = |k| mos.coeff.re[(k, b)].powi(2) + mos.coeff.im[(k, b)].powi(2);
                    v(i).total_cmp(&v(j))
                })
                .unwrap();
            let re = mos.coeff.re[(p, b)];
            let im = mos.coeff.im[(p, b)];
            let norm = re.hypot(im);
            for i in 0..n {
                c[(i, b)] = (mos.coeff.re[(i, b)] * re + mos.coeff.im[(i, b)] * im) / norm;
                assert!(
                    (mos.coeff.im[(i, b)] * re - mos.coeff.re[(i, b)] * im).abs() / norm < 1e-12
                );
            }
        }
        let co = |u: &[f64]| -> Vec<f64> {
            let mut y: Vec<f64> = (0..n)
                .map(|i| c[(i, 0)] + (0..nv).map(|a| c[(i, a + 1)] * u[a]).sum::<f64>())
                .collect();
            let norm = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| y[i] * scf.hs_k[0].1.re[(i, j)] * y[j])
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();
            for x in &mut y {
                *x /= norm;
            }
            y
        };
        let original = co(&vec![0.; nv]);
        let boosts: Vec<Vec<(f64, f64)>> = lattice
            .reciprocal_vectors_2pi()
            .iter()
            .map(|&q| {
                (0..n)
                    .flat_map(|i| (0..n).map(move |j| (i, j)))
                    .map(|(i, j)| {
                        let a = &scf.basis.aos[i];
                        let b = &scf.basis.aos[j];
                        boosted_overlap_pair(
                            a,
                            b,
                            sys.atoms[a.atom_index].position,
                            sys.atoms[b.atom_index].position,
                            q,
                        )
                    })
                    .collect()
            })
            .collect();
        let phases = |v: &[f64]| -> [f64; 3] {
            std::array::from_fn(|d| {
                let re = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| v[i] * v[j] * boosts[d][i * n + j].0)
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                let im = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| v[i] * v[j] * boosts[d][i * n + j].1)
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                im.atan2(re)
            })
        };
        let phase0 = phases(&original);
        let delta_p = |v: &[f64]| -> [f64; 3] {
            let ph = phases(v);
            std::array::from_fn(|i| {
                (0..3)
                    .map(|d| {
                        let x = ph[d] - phase0[d];
                        let x = x - 2. * PI * (x / (2. * PI)).round();
                        -lattice.cell.col[d].to_array()[i] * x / (PI * volume)
                    })
                    .sum()
            })
        };
        let charges = |v: &[f64]| -> Vec<f64> {
            scf.basis
                .shells
                .iter()
                .map(|s| {
                    s.reference_occ
                        - (s.first_ao..s.first_ao + s.nao)
                            .map(|i| {
                                (0..n)
                                    .map(|j| 2. * v[i] * v[j] * scf.hs_k[0].1.re[(j, i)])
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                })
                .collect()
        };
        let coulomb = |v: &[f64]| -> f64 {
            let e = coulomb_energy_potential_from_matrix(
                &scf.basis,
                &scf.shell_model,
                &charges(v),
                &scf.gamma,
            )
            .unwrap();
            e.second_order + e.third_order + e.higher_order
        };
        let coulomb0 = coulomb(&original);
        // Independent finite-field variational oracle: occupied-orbital Rayleigh
        // energy + full SCC charge functional - Ω E.P. Only validation uses
        // finite differences, both in field and orbital minimization.
        let objective = |u: &[f64], field: [f64; 3]| -> f64 {
            let v = co(u);
            let p = delta_p(&v);
            let band = 2.
                * (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                (v[i] - original[i])
                                    * scf.hs_k[0].0.re[(i, j)]
                                    * (v[j] + original[j])
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
            band + coulomb(&v) - coulomb0 - volume * (0..3).map(|i| field[i] * p[i]).sum::<f64>()
        };
        let dh = 1e-3;
        let mut h = Matrix::zeros(nv, nv);
        for a in 0..nv {
            for b in 0..nv {
                if a == b {
                    let mut p = vec![0.; nv];
                    let mut m = p.clone();
                    p[a] = dh;
                    m[a] = -dh;
                    h[(a, b)] = (objective(&p, [0.; 3]) + objective(&m, [0.; 3])) / (dh * dh);
                } else {
                    let mut sum = 0.;
                    for sa in [1., -1.] {
                        for sb in [1., -1.] {
                            let mut u = vec![0.; nv];
                            u[a] = sa * dh;
                            u[b] = sb * dh;
                            sum += sa * sb * objective(&u, [0.; 3]);
                        }
                    }
                    h[(a, b)] = sum / (4. * dh * dh);
                }
            }
        }
        let lu = DenseLu::factor(&h).unwrap();
        let mut fd = [[[0.; 3]; 3]; 2];
        for (step_index, field_step) in [1e-4, 5e-5].into_iter().enumerate() {
            for axis in 0..3 {
                let mut polar = [[0.; 3]; 2];
                for (sign_index, sign) in [1., -1.].into_iter().enumerate() {
                    let mut field = [0.; 3];
                    field[axis] = sign * field_step;
                    let mut u = vec![0.; nv];
                    let mut converged = false;
                    for _ in 0..50 {
                        let mut g = vec![0.; nv];
                        for a in 0..nv {
                            let mut p = u.clone();
                            let mut m = u.clone();
                            p[a] += 1e-5;
                            m[a] -= 1e-5;
                            g[a] = (objective(&p, field) - objective(&m, field)) / 2e-5;
                        }
                        if g.iter().map(|x| x.abs()).fold(0., f64::max) < 2e-10 {
                            converged = true;
                            break;
                        }
                        let step = lu.solve_vec(&g).unwrap();
                        for a in 0..nv {
                            u[a] -= step[a];
                        }
                    }
                    assert!(
                        converged,
                        "finite-field orbital minimization did not converge"
                    );
                    polar[sign_index] = delta_p(&co(&u));
                }
                for i in 0..3 {
                    fd[step_index][i][axis] = (polar[0][i] - polar[1][i]) / (2. * field_step);
                }
            }
        }
        let mut error = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let chi = (4. * fd[1][i][j] - fd[0][i][j]) / 3.;
                error = error.max((chi - analytic.susceptibility[(i, j)]).abs());
            }
        }
        eprintln!("all9 dielectric tensor components against independent finite-field minima, ML={learned}: {error:e}");
        assert!(error < 1e-8);
    }
}
