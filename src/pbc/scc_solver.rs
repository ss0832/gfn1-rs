// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic Jacobian and globalization of the periodic shell-charge fixed point.
use super::*;
use crate::linalg::DenseLu;

/// Canonical finite-temperature derivative dq_out/dV at fixed geometry.
/// The real embedding carries each physical spatial orbital twice, with max-1
/// occupations. Consequently the trace below needs no extra spin factor.
pub(super) fn susceptibility(
    basis: &BasisSet,
    hs: &[(CMatrix, CMatrix)],
    kpoints: &[KPoint],
    eigs: &[crate::pbc::complex::KEigen],
    occupations: &[Vec<f64>],
    kt: f64,
) -> Result<Matrix> {
    let n = basis.len();
    let m = 2 * n;
    let ns = basis.shells.len();
    let scratch = m
        .checked_mul(m)
        .and_then(|x| x.checked_mul(2 * ns + 6))
        .and_then(|x| x.checked_add(4 * ns * ns))
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| Gfn1Error::InvalidInput("periodic Newton workspace overflow".into()))?;
    let retained = n
        .checked_mul(n)
        .and_then(|x| x.checked_mul(32 * kpoints.len()))
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| Gfn1Error::InvalidInput("periodic Newton workspace overflow".into()))?;
    crate::resources::check_workspace(
        scratch.saturating_add(retained),
        "periodic analytic charge Newton",
    )?;
    let mut chi = Matrix::zeros(ns, ns);
    let mut canonical = vec![0.; ns];
    let mut slope_sum = 0.;
    for (ik, eig) in eigs.iter().enumerate() {
        let sk = &hs[ik].1;
        let mut overlap = Matrix::zeros(m, m);
        for i in 0..n {
            for j in 0..n {
                overlap[(i, j)] = sk.re[(i, j)];
                overlap[(n + i, n + j)] = sk.re[(i, j)];
                overlap[(i, n + j)] = -sk.im[(i, j)];
                overlap[(n + i, j)] = sk.im[(i, j)];
            }
        }
        let sc = overlap.matmul(&eig.vectors)?;
        let mut t = Matrix::zeros(ns, m * m);
        for (s, shell) in basis.shells.iter().enumerate() {
            for i in shell.first_ao..shell.first_ao + shell.nao {
                for mu in [i, n + i] {
                    for a in 0..m {
                        for b in 0..m {
                            t[(s, a * m + b)] += 0.5
                                * (eig.vectors[(mu, a)] * sc[(mu, b)]
                                    + sc[(mu, a)] * eig.vectors[(mu, b)]);
                        }
                    }
                }
            }
        }
        let f = &occupations[ik];
        let slopes: Vec<f64> = f
            .iter()
            .map(|g| if kt > 0. { -g * (1. - g) / kt } else { 0. })
            .collect();
        let weight = kpoints[ik].weight;
        let mut weighted = Matrix::zeros(m * m, ns);
        for a in 0..m {
            slope_sum += weight * slopes[a];
            for s in 0..ns {
                canonical[s] += weight * slopes[a] * t[(s, a * m + a)];
            }
            for b in 0..m {
                let gap = eig.values[a] - eig.values[b];
                // Analytic Fermi divided difference. Its coincident-energy limit
                // is f', including a partially occupied degenerate subspace.
                let divided = if a == b || gap.abs() <= 1e-10_f64.min(kt * 1e-4).max(1e-14) {
                    if kt <= 0. && (f[a] - f[b]).abs() > 1e-9 {
                        return Err(Gfn1Error::InvalidInput(
                            "periodic Newton at zero temperature needs a differentiable occupation map; use finite temperature for a fractionally occupied degeneracy".into()));
                    }
                    0.5 * (slopes[a] + slopes[b])
                } else {
                    (f[a] - f[b]) / gap
                };
                for s in 0..ns {
                    weighted[(a * m + b, s)] = weight * divided * t[(s, a * m + b)];
                }
            }
        }
        let block = t.matmul(&weighted)?;
        for s in 0..ns {
            for u in 0..ns {
                chi[(s, u)] += block[(s, u)];
            }
        }
    }
    // dmu enforces ONE fixed total electron count across the complete k mesh.
    if slope_sum.abs() > 1e-200 {
        for s in 0..ns {
            for t in 0..ns {
                chi[(s, t)] -= canonical[s] * canonical[t] / slope_sum;
            }
        }
    }
    Ok(chi)
}

pub(super) fn total_charge(q: &mut [f64], charge: f64) {
    let shift = (charge - q.iter().sum::<f64>()) / q.len() as f64;
    for value in q {
        *value += shift;
    }
}

/// Solve the exact Newton equation; if its direction is unsuitable, use the
/// negative gradient of ||q_out(q)-q||². Both use residual-decreasing backtracking.
#[allow(clippy::too_many_arguments)]
pub(super) fn globalized_step(
    basis: &BasisSet,
    model: &ShellChargeModel,
    gamma: &Matrix,
    hs: &[(CMatrix, CMatrix)],
    kpoints: &[KPoint],
    state: &SccState,
    q: &[f64],
    residual: &[f64],
    kt: f64,
    charge: f64,
    diagnostics: &mut PbcSccDiagnostics,
    mut evaluate: impl FnMut(&[f64]) -> Result<SccState>,
) -> Result<Option<(Vec<f64>, SccState)>> {
    let (eigs, occ) = state.response.as_ref().ok_or_else(|| {
        Gfn1Error::InvalidInput("periodic Newton response eigensystem was not retained".into())
    })?;
    let chi = susceptibility(basis, hs, kpoints, eigs, occ, kt)?;
    let kernel = crate::electronic::scc_response_kernel(gamma, model, basis, q);
    let product = chi.matmul(&kernel)?;
    let ns = q.len();
    let mut dielectric = Matrix::identity(ns);
    for i in 0..ns {
        for j in 0..ns {
            dielectric[(i, j)] -= product[(i, j)];
        }
    }
    let newton = DenseLu::factor(&dielectric)
        .and_then(|lu| lu.solve_vec(residual))
        .ok()
        .filter(|v| v.iter().all(|x| x.is_finite()))
        .filter(|v| {
            (0..ns).all(|i| {
                let error = (0..ns).map(|j| dielectric[(i, j)] * v[j]).sum::<f64>() - residual[i];
                error.abs() < 1e-8 * (1. + residual[i].abs())
            })
        });
    let gradient: Vec<f64> = (0..ns)
        .map(|j| (0..ns).map(|i| dielectric[(i, j)] * residual[i]).sum())
        .collect();
    let current = residual.iter().map(|x| x * x).sum::<f64>();
    for (is_newton, direction) in newton
        .into_iter()
        .map(|d| (true, d))
        .chain(std::iter::once((false, gradient)))
    {
        diagnostics.newton_directions += usize::from(is_newton);
        let max = direction.iter().fold(0_f64, |v, x| v.max(x.abs()));
        if !max.is_finite() || max == 0. {
            continue;
        }
        let mut scale = (0.5 / max).min(1.);
        for _ in 0..14 {
            let mut trial: Vec<f64> = q
                .iter()
                .zip(&direction)
                .map(|(q, d)| q + scale * d)
                .collect();
            total_charge(&mut trial, charge);
            if trusted_charge_vector(&trial) {
                diagnostics.map_evaluations += 1;
                if let Ok(next) = evaluate(&trial) {
                    let value = next
                        .shell_charges
                        .iter()
                        .zip(&trial)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>();
                    if value <= current * (1. - 1e-4 * scale) || value < 1e-28 {
                        diagnostics.newton_accepted += usize::from(is_newton);
                        diagnostics.merit_gradient_accepted += usize::from(!is_newton);
                        return Ok(Some((trial, next)));
                    }
                }
            }
            diagnostics.backtracks += 1;
            scale *= 0.5;
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_exact_degeneracy_is_gauge_invariant_and_charge_conserving() {
        let system = PeriodicSystem::from_xyz_str(
            "1\nH anion synthetic degenerate shell space\nH 0 0 0\n",
            -1.,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(
            &system,
            &Gfn1Parameters::builtin().unwrap(),
            BasisOptions::default(),
        )
        .unwrap();
        assert_eq!(basis.len(), 2);
        let mut overlap = CMatrix::zeros(2);
        overlap.re = Matrix::identity(2);
        let hs = vec![(CMatrix::zeros(2), overlap)];
        let points = gamma_only();
        let mut eig = crate::pbc::complex::KEigen {
            values: vec![0.; 4],
            vectors: Matrix::identity(4),
            n: 2,
        };
        let occ = vec![vec![0.5; 4]];
        let original = susceptibility(&basis, &hs, &points, &[eig.clone()], &occ, 0.001).unwrap();
        let (c, s) = (0.731_f64.cos(), 0.731_f64.sin());
        eig.vectors[(0, 0)] = c;
        eig.vectors[(3, 3)] = c;
        eig.vectors[(0, 3)] = -s;
        eig.vectors[(3, 0)] = s;
        let rotated = susceptibility(&basis, &hs, &points, &[eig], &occ, 0.001).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((original[(i, j)] - rotated[(i, j)]).abs() < 1e-10);
                assert!((original[(i, j)] - if i == j { -250. } else { 250. }).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn canonical_jacobian_all_shells_matches_potential_differences() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4 .2 .1 0 4.5 .3 .1 0 5\" pbc=\"T T T\"\nH .2 .3 .1\nF 1.1 .6 .4\n",
            0.,
            false,
        )
        .unwrap();
        let params = Gfn1Parameters::builtin().unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let bloch = BlochBuilder::build(&system, &basis, &params, 9., 20., true).unwrap();
        let points = build_kpoints(
            &system,
            crate::pbc::KMesh {
                size: [2, 2, 2],
                gamma_centered: false,
                fold_time_reversal: false,
            },
        );
        let hs: Vec<_> = points
            .iter()
            .map(|p| bloch.h_s_at_k(p.fractional))
            .collect();
        let orth: Vec<_> = hs
            .iter()
            .map(|(_, s)| embedded_overlap_orthogonalizer(s, 1e-12).unwrap())
            .collect();
        let potential: Vec<_> = (0..basis.shells.len())
            .map(|i| 0.03 * (i as f64 - 1.))
            .collect();
        for temperature in [0., 300., 30000.] {
            let kt = temperature * BOLTZMANN_HARTREE_PER_K;
            let evaluate = |v: &[f64]| {
                scc_step(
                    &basis,
                    &hs,
                    &orth,
                    &points,
                    v,
                    basis.total_reference_electrons,
                    None,
                    kt,
                    1e-12,
                    None,
                    true,
                )
                .unwrap()
            };
            let state = evaluate(&potential);
            let (eigs, occ) = state.response.as_ref().unwrap();
            let chi = susceptibility(&basis, &hs, &points, eigs, occ, kt).unwrap();
            let mut error = 0_f64;
            for t in 0..potential.len() {
                let derivative = |h: f64| {
                    let mut p = potential.clone();
                    p[t] += h;
                    let mut m = potential.clone();
                    m[t] -= h;
                    let plus = evaluate(&p);
                    let minus = evaluate(&m);
                    plus.shell_charges
                        .iter()
                        .zip(minus.shell_charges)
                        .map(|(a, b)| (a - b) / (2. * h))
                        .collect::<Vec<_>>()
                };
                let coarse = derivative(2e-5);
                let fine = derivative(1e-5);
                for s in 0..potential.len() {
                    error = error.max(((4. * fine[s] - coarse[s]) / 3. - chi[(s, t)]).abs());
                    assert!((chi[(s, t)] - chi[(t, s)]).abs() < 1e-10);
                }
                assert!((0..potential.len()).map(|s| chi[(s, t)]).sum::<f64>().abs() < 1e-9);
            }
            eprintln!("canonical periodic charge Jacobian T={temperature} all{} components error {error:e}",potential.len().pow(2));
            assert!(error < 2e-7, "charge response error {error:e}");
        }
    }

    #[test]
    fn complete_onsite_charge_kernel_matches_all_potential_derivatives() {
        let system = PeriodicSystem::from_xyz_str("2\nHF\nH 0 0 0\nF .9 0 0\n", 0., false).unwrap();
        let params = Gfn1Parameters::builtin().unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let mut model = ShellChargeModel::build(&system, &basis, &params).unwrap();
        model.charge_order = 6;
        let gamma = Matrix::identity(basis.shells.len());
        let q: Vec<_> = (0..basis.shells.len())
            .map(|i| 0.15 * (i as f64 - 1.))
            .collect();
        let kernel = crate::electronic::scc_response_kernel(&gamma, &model, &basis, &q);
        for t in 0..q.len() {
            let mut p = q.clone();
            p[t] += 1e-5;
            let mut m = q.clone();
            m[t] -= 1e-5;
            let plus = coulomb_energy_potential_from_matrix(&basis, &model, &p, &gamma).unwrap();
            let minus = coulomb_energy_potential_from_matrix(&basis, &model, &m, &gamma).unwrap();
            for s in 0..q.len() {
                assert!(
                    ((plus.shell_potential[s] - minus.shell_potential[s]) / 2e-5 - kernel[(s, t)])
                        .abs()
                        < 1e-9
                );
            }
        }
    }
}
