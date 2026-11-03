// SPDX-License-Identifier: GPL-3.0-or-later
//! Optical (q=0) periodic linearized SCC density-matrix response.

use super::*;
use crate::pbc::complex::{native_generalized_eigen, native_hermitian_eigen, CMatrix};
use crate::pbc::scf::{fock_at_k, spinor_occupations};
use crate::pbc::PbcOptions;
use crate::soc::{SocParameters, SpinorEigen};
use faer::c64;

/// Positive-frequency optical response poles at the zone centre, in Hartree.
#[derive(Clone, Debug)]
pub struct OpticalTdDfptResult {
    pub ground_free_energy: f64,
    pub excitation_energies: Vec<f64>,
    pub transitions: usize,
    pub kpoints: usize,
}

/// Solve the full resonant and antiresonant periodic SCC response at optical
/// momentum q=0. SOC, when supplied, uses complex physical spinors with a
/// max-one occupation. The kernel couples the spin trace of the density.
pub fn solve_optical_td_dfpt(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    pbc: &PbcOptions,
    soc: Option<&SocParameters>,
    n_states: usize,
) -> Result<OpticalTdDfptResult> {
    if system.lattice.is_none() || n_states == 0 {
        return Err(Gfn1Error::InvalidInput(
            "optical TD-DFPT requires a periodic system and a positive root count".into(),
        ));
    }
    if electronic.multipole
        || electronic.spin_polarization
        || electronic.plus_u
        || electronic.lr_exchange
        || electronic.external_field.electric_field.is_some()
        || electronic.external_field.magnetic_field.is_some()
    {
        return Err(Gfn1Error::InvalidInput(
            "optical TD-DFPT requires the scalar SCC charge kernel without external fields".into(),
        ));
    }

    // Complex resonant/antiresonant charge response needs both partners of
    // each time-reversal pair. Use their original independent BZ weights.
    let mut pbc = pbc.clone();
    pbc.kmesh.fold_time_reversal = false;
    let (scf, bands, occupations) = if let Some(soc) = soc {
        let result = crate::pbc::run_pbc_soc_scc(system, params, electronic, &pbc, soc)?;
        (result.scalar_proxy, result.eigen_k, result.occupations_k)
    } else {
        let scf = crate::pbc::run_pbc_scc(system, params, electronic, &pbc)?;
        if !scf.converged {
            return Err(Gfn1Error::InvalidInput(
                "optical TD-DFPT requires converged SCC".into(),
            ));
        }
        let mut potential = vec![0.0; scf.basis.len()];
        for (shell, &value) in scf.basis.shells.iter().zip(&scf.shell_scc_potential) {
            potential[shell.first_ao..shell.first_ao + shell.nao].fill(value);
        }
        let bands = scf
            .hs_k
            .iter()
            .map(|(h, s)| {
                let eig = native_generalized_eigen(
                    &fock_at_k(h, s, &potential),
                    s,
                    electronic.eigen_tolerance,
                )?;
                Ok(SpinorEigen {
                    values: eig.values,
                    vectors: eig.vectors,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let energies = bands
            .iter()
            .map(|band| band.values.clone())
            .collect::<Vec<_>>();
        let weights = scf.kpoints.iter().map(|k| k.weight).collect::<Vec<_>>();
        let mut occupations = spinor_occupations(
            &energies,
            &weights,
            scf.nelec / 2.,
            scf.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
        )?
        .occupations;
        for f in occupations.iter_mut().flatten() {
            *f *= 2.;
        }
        (scf, bands, occupations)
    };
    let kernel = crate::pbc::hessian::periodic_response_kernel(&scf);
    let mut gaps = Vec::new();
    let mut charges = Vec::new();
    let n_ao = scf.basis.len();
    let nshell = scf.basis.shells.len();
    for (k, (band, occupations)) in bands.iter().zip(&occupations).enumerate() {
        let overlap = &scf.hs_k[k].1;
        let c = &band.vectors;
        let nspin = c.n / n_ao;
        let mut sc = CMatrix::zeros(c.n);
        for spin in 0..nspin {
            for mu in 0..n_ao {
                for nu in 0..n_ao {
                    let s = at(overlap, mu, nu);
                    for p in 0..c.n {
                        let value = s * at(c, spin * n_ao + nu, p);
                        sc.accumulate(spin * n_ao + mu, p, value.re, value.im);
                    }
                }
            }
        }
        for i in 0..c.n {
            for a in i + 1..c.n {
                let df = occupations[i] - occupations[a];
                let gap = band.values[a] - band.values[i];
                if df <= 1e-10 || gap <= 1e-10 {
                    continue;
                }
                let weight = (scf.kpoints[k].weight * df).sqrt();
                let mut q = vec![c64::new(0., 0.); nshell];
                for (charge, shell) in q.iter_mut().zip(&scf.basis.shells) {
                    for spin in 0..nspin {
                        for mu in shell.first_ao..shell.first_ao + shell.nao {
                            let mu = spin * n_ao + mu;
                            *charge -= (at(c, mu, i).conj() * at(&sc, mu, a)
                                + at(&sc, mu, i).conj() * at(c, mu, a))
                                * weight;
                        }
                    }
                }
                gaps.push(gap);
                charges.push(q);
            }
        }
    }

    let n = gaps.len();
    if n == 0 {
        return Err(Gfn1Error::InvalidInput(
            "optical TD-DFPT found no positive interband transitions".into(),
        ));
    }
    let excitation_energies = optical_poles(&gaps, &kernel, &charges, n_states)?;
    Ok(OpticalTdDfptResult {
        ground_free_energy: scf.total_free,
        excitation_energies,
        transitions: n,
        kpoints: scf.kpoints.len(),
    })
}

fn at(matrix: &CMatrix, i: usize, j: usize) -> c64 {
    c64::new(matrix.re[(i, j)], matrix.im[(i, j)])
}

fn optical_poles(
    gaps: &[f64],
    kernel: &Matrix,
    charges: &[Vec<c64>],
    n_states: usize,
) -> Result<Vec<f64>> {
    // Hessian of the response energy H = [[A,B],[B*,A*]].
    // A = D + Q† K Q, B = Q† K Q*. Solve J H through its Hermitian
    // representative H^(1/2) J H^(1/2), J = diag(I,-I).
    let n = gaps.len();
    let dim = n
        .checked_mul(2)
        .ok_or_else(|| Gfn1Error::InvalidInput("optical response dimension overflow".into()))?;
    let estimate = dim
        .checked_mul(dim)
        .and_then(|count| count.checked_mul(128))
        .ok_or_else(|| {
            Gfn1Error::InvalidInput("optical TD-DFPT workspace size overflows".into())
        })?;
    let budget = crate::resources::effective_memory_budget(512 * 1024 * 1024);
    if estimate > budget {
        return Err(Gfn1Error::InvalidInput(format!(
            "optical TD-DFPT workspace estimate {estimate} exceeds {budget} bytes"
        )));
    }
    let mut h = CMatrix::zeros(dim);
    for (j, charge) in charges.iter().enumerate() {
        let potential: Vec<c64> = (0..kernel.rows())
            .map(|s| {
                charge
                    .iter()
                    .enumerate()
                    .map(|(t, &q)| q * kernel[(s, t)])
                    .sum()
            })
            .collect();
        for i in j..n {
            let mut a: c64 = charges[i]
                .iter()
                .zip(&potential)
                .map(|(&qi, &p)| qi.conj() * p)
                .sum();
            let b: c64 = charges[i]
                .iter()
                .zip(&potential)
                .map(|(&qi, &p)| (qi * p).conj())
                .sum();
            if i == j {
                a.re += gaps[i];
            } else {
                h.accumulate(j, i, a.re, -a.im);
                h.accumulate(j + n, i + n, a.re, a.im);
                h.accumulate(j, i + n, b.re, b.im);
                h.accumulate(i + n, j, b.re, -b.im);
            }
            h.accumulate(i, j, a.re, a.im);
            h.accumulate(i + n, j + n, a.re, -a.im);
            h.accumulate(i, j + n, b.re, b.im);
            h.accumulate(j + n, i, b.re, -b.im);
        }
    }
    h.hermitianize();
    let eigen = native_hermitian_eigen(&h)?;
    if eigen.values.iter().any(|&value| value <= 0.) {
        return Err(Gfn1Error::InvalidInput(
            "unstable optical TD-DFPT reference: response Hessian is not positive definite".into(),
        ));
    }
    let mut scaled = eigen.vectors.clone();
    for (j, value) in eigen.values.iter().enumerate() {
        let factor = value.sqrt();
        for i in 0..dim {
            scaled.re[(i, j)] *= factor;
            scaled.im[(i, j)] *= factor;
        }
    }
    let root = scaled.matmul(&eigen.vectors.adjoint())?;
    let mut signed = root.clone();
    for i in n..dim {
        for j in 0..dim {
            signed.re[(i, j)] *= -1.;
            signed.im[(i, j)] *= -1.;
        }
    }
    let mut response = root.matmul(&signed)?;
    response.hermitianize();
    let values = native_hermitian_eigen(&response)?.values;
    if values[n] <= 0. || values[n - 1] >= 0. {
        return Err(Gfn1Error::InvalidInput(
            "optical TD-DFPT positive and negative frequencies could not be resolved".into(),
        ));
    }
    Ok(values.into_iter().skip(n).take(n_states).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complex_optical_poles_are_invariant_under_transition_phases() {
        let k = Matrix::from_vec(2, 2, vec![0.7, 0.11, 0.11, 0.5]).unwrap();
        let gaps = [0.4, 0.7];
        let q = vec![
            vec![c64::new(0.17, 0.2), c64::new(-0.23, 0.3)],
            vec![c64::new(0.12, -0.1), c64::new(0.32, 0.1)],
        ];
        let expected = optical_poles(&gaps, &k, &q, 2).unwrap();
        let transformed: Vec<_> = q
            .iter()
            .zip([0.37_f64, 1.21])
            .map(|(row, angle)| {
                let phase = c64::new(angle.cos(), angle.sin());
                row.iter().map(|&charge| charge * phase).collect()
            })
            .collect();
        let actual = optical_poles(&gaps, &k, &transformed, 2).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
        let [x, y] = q[0][..] else { unreachable!() };
        let resonant = gaps[0]
            + k[(0, 0)] * x.norm_sqr()
            + 2.0 * k[(0, 1)] * (x.conj() * y).re
            + k[(1, 1)] * y.norm_sqr();
        let antiresonant = k[(0, 0)] * x * x + 2.0 * k[(0, 1)] * x * y + k[(1, 1)] * y * y;
        let exact = (resonant.powi(2) - antiresonant.norm_sqr()).sqrt();
        let pole = optical_poles(&gaps[..1], &k, &q[..1], 1).unwrap()[0];
        assert!((pole - exact).abs() < 1e-12);
    }
}
