// SPDX-License-Identifier: GPL-3.0-or-later
use super::cluster::{subset, Cluster};
use super::geometry::Geometry;
use super::spectral::{fermi, trace, Spectrum};
use super::{finish, invalid, validate, FragmentPartition, FragmentResult};
use crate::coulomb::coulomb_energy_potential_from_matrix;
use crate::electronic::{BroydenMixer, ElectronicOptions};
use crate::error::{Gfn1Error, Result};
use crate::linalg::{
    lowdin_solve_generalized, matrix_vector_product, DenseLu, EigenDecomposition, Matrix,
};
use crate::params::Gfn1Parameters;
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct DivideConquerOptions {
    /// One frozen core+buffer atom list per partition fragment. Empty selects
    /// the core atoms only. Lists must contain all atoms of the associated core.
    pub buffers: Vec<Vec<usize>>,
    pub compute_derivatives: bool,
    pub max_ao: usize,
}
impl Default for DivideConquerOptions {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            compute_derivatives: true,
            max_ao: 512,
        }
    }
}

fn weighted(cluster: &Cluster, matrix: &Matrix) -> Matrix {
    let n = cluster.aos.len();
    let mut result = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = 0.5 * (cluster.core[i] + cluster.core[j]) * matrix[(i, j)];
        }
    }
    result
}
fn population(cluster: &Cluster, matrix: &Matrix, overlap: &Matrix, charges: &mut [f64]) {
    let n = cluster.aos.len();
    for i in 0..n {
        for j in 0..n {
            charges[cluster.shells[cluster.ao_shell[i]]] -=
                0.5 * (cluster.core[i] + cluster.core[j]) * matrix[(i, j)] * overlap[(j, i)];
        }
    }
}
fn orbital_weights(cluster: &Cluster, eigen: &EigenDecomposition) -> Vec<f64> {
    let c = &eigen.vectors;
    let n = c.rows();
    (0..c.cols())
        .map(|k| {
            (0..n)
                .map(|i| {
                    cluster.core[i]
                        * c[(i, k)]
                        * (0..n).map(|j| cluster.s[(i, j)] * c[(j, k)]).sum::<f64>()
                })
                .sum()
        })
        .collect()
}
fn common_mu(
    eigen: &[EigenDecomposition],
    weights: &[Vec<f64>],
    electrons: f64,
    kt: f64,
) -> Result<f64> {
    // Individual Mulliken core contributions can be negative even with complete
    // buffers. Electron conservation is imposed on their sum, not on each
    // arbitrarily partitioned orbital weight.
    let mut low = eigen
        .iter()
        .flat_map(|e| e.values.iter())
        .copied()
        .fold(f64::INFINITY, f64::min)
        - 100. * kt
        - 1.;
    let mut high = eigen
        .iter()
        .flat_map(|e| e.values.iter())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        + 100. * kt
        + 1.;
    let number = |mu: f64| {
        eigen
            .iter()
            .zip(weights)
            .map(|(e, w)| {
                e.values
                    .iter()
                    .zip(w)
                    .map(|(&energy, &weight)| {
                        weight
                            * if kt > 0. {
                                fermi(energy, mu, kt)
                            } else if energy < mu {
                                2.
                            } else {
                                0.
                            }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    };
    if number(high) < electrons - 1e-8 || number(low) > electrons + 1e-8 {
        return Err(invalid(
            "DC buffers cannot accommodate the requested total electron count",
        ));
    }
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if kt == 0. && (number(mid) - electrons).abs() < 1e-10 {
            return Ok(mid);
        }
        if mid == low || mid == high {
            break;
        }
        if number(mid) < electrons {
            low = mid;
        } else {
            high = mid;
        }
    }
    let mu = 0.5 * (low + high);
    if (number(mu) - electrons).abs() > 1e-7 {
        return Err(invalid("zero-temperature DC cannot conserve electrons for this finite buffer; use positive electronic temperature or complete buffers"));
    }
    Ok(mu)
}

/// Overlapping core/buffer DC density with symmetrized core weights and a
/// common chemical potential. Forces differentiate this finite-buffer energy,
/// including the density, entropy and global SCC/chemical-potential response.
/// Frozen atom lists keep the partition differentiable during the calculation.
pub fn divide_and_conquer(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    pbc: &PbcOptions,
    partition: &FragmentPartition,
    options: &DivideConquerOptions,
) -> Result<FragmentResult> {
    validate(system, electronic, partition)?;
    if !partition.cuts.is_empty() {
        return Err(invalid(
            "DC uses overlapping buffers across covalent boundaries; HOP cuts apply to FMO only",
        ));
    }
    if system.lattice.is_some() && !pbc.kmesh.is_gamma_only() {
        return Err(invalid("periodic DC currently uses Gamma reference-cell orbitals; use supercells for k-space convergence"));
    }
    let nat = system.atoms.len();
    let nf = partition.atoms.len();
    let buffers = if options.buffers.is_empty() {
        &partition.atoms
    } else {
        &options.buffers
    };
    if buffers.len() != nf {
        return Err(invalid("DC requires one buffer list per core fragment"));
    }
    for (i, atoms) in buffers.iter().enumerate() {
        let mut seen = vec![false; nat];
        for &a in atoms {
            if a >= nat || seen[a] {
                return Err(invalid("DC buffer atoms must be valid and unique"));
            }
            seen[a] = true;
        }
        if partition.atoms[i].iter().any(|&a| !seen[a]) {
            return Err(invalid("each DC buffer must contain its complete core"));
        }
    }
    let basis = crate::basis::BasisSet::build(
        system,
        params,
        crate::basis::BasisOptions {
            nprim: electronic.nprim,
        },
    )?;
    if basis.len() > options.max_ao {
        return Err(invalid("DC reference-cell AO count exceeds max_ao"));
    }
    let geometry = Geometry::build(system, params, electronic, pbc)?;
    let clusters: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, a)| Cluster::new(&geometry, system, a, &partition.atoms[i], 0., &[], 0.))
        .collect::<Result<_>>()?;
    let nsh = geometry.gamma.rows();
    let n = geometry.basis.len();
    let kt = electronic.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
    let electrons =
        geometry.basis.total_reference_electrons - electronic.charge.unwrap_or(system.charge);
    if electrons < 0. || electrons > 2. * n as f64 {
        return Err(invalid("DC electron count exceeds the AO capacity"));
    }
    let mut q = vec![0.; nsh];
    let mut mixer = BroydenMixer::new(
        nsh,
        electronic.scc_broyden_size.max(2),
        electronic.mixing.clamp(0.01, 1.),
    );
    let mut spectra = Vec::new();
    let mut final_potential = Vec::new();
    let mut iterations = 0;
    let mut rms = f64::INFINITY;
    for iteration in 1..=electronic.max_scc {
        let potential = coulomb_energy_potential_from_matrix(
            &geometry.basis,
            &geometry.model,
            &q,
            &geometry.gamma,
        )?
        .shell_potential;
        let eigens: Vec<_> = clusters
            .iter()
            .map(|c| {
                let v: Vec<_> = c.shells.iter().map(|&s| potential[s]).collect();
                lowdin_solve_generalized(&c.fock(&v)?, &c.s, electronic.eigen_tolerance)
            })
            .collect::<Result<_>>()?;
        let weights: Vec<_> = clusters
            .iter()
            .zip(&eigens)
            .map(|(c, e)| orbital_weights(c, e))
            .collect();
        let mu = common_mu(&eigens, &weights, electrons, kt)?;
        spectra = eigens
            .into_iter()
            .map(|e| Spectrum::from_eigen(e, kt, mu))
            .collect::<Result<_>>()?;
        let mut new_q: Vec<_> = geometry
            .basis
            .shells
            .iter()
            .map(|s| s.reference_occ)
            .collect();
        for (c, s) in clusters.iter().zip(&spectra) {
            population(c, &s.density, &c.s, &mut new_q);
        }
        let residual: Vec<_> = new_q.iter().zip(&q).map(|(a, b)| a - b).collect();
        rms = (residual.iter().map(|x| x * x).sum::<f64>() / nsh as f64).sqrt();
        iterations = iteration;
        final_potential = potential;
        if rms < electronic.charge_tolerance.min(1e-10) {
            q = new_q;
            break;
        }
        q = mixer
            .next(&q, &residual)
            .filter(|q| q.iter().all(|x| x.is_finite() && x.abs() < 20.))
            .unwrap_or_else(|| q.iter().zip(&residual).map(|(a, b)| a + 0.2 * b).collect());
    }
    if rms >= electronic.charge_tolerance.min(1e-10) {
        return Err(Gfn1Error::SccNotConverged { iterations, rms });
    }
    let coulomb = coulomb_energy_potential_from_matrix(
        &geometry.basis,
        &geometry.model,
        &q,
        &geometry.gamma,
    )?;
    let mut energy = geometry.classical + coulomb.second_order + coulomb.third_order;
    for (c, s) in clusters.iter().zip(&spectra) {
        energy +=
            trace(&weighted(c, &s.density), &c.h) + trace(&weighted(c, &s.entropy_density), &c.s);
    }
    let mut derivatives = vec![0.; geometry.ndof];
    if options.compute_derivatives {
        let mut kernel = geometry.gamma.clone();
        let atomic = geometry.model.atomic_charges(&geometry.basis, &q);
        for i in 0..nsh {
            for j in 0..nsh {
                let a = geometry.basis.shells[i].atom_index;
                if a == geometry.basis.shells[j].atom_index {
                    kernel[(i, j)] += 2. * geometry.model.hubbard_derivs[i] * atomic[a];
                }
            }
        }
        let mut jacobian = Matrix::zeros(nsh + 1, nsh + 1);
        let mut use_mu = true;
        for j in 0..=nsh {
            let v: Vec<_> = (0..nsh)
                .map(|i| if j < nsh { kernel[(i, j)] } else { 0. })
                .collect();
            let mut response = vec![0.; nsh];
            for (c, s) in clusters.iter().zip(&spectra) {
                let potential: Vec<_> = c.shells.iter().map(|&g| v[g]).collect();
                let ds = Matrix::zeros(c.aos.len(), c.aos.len());
                let dp = s.response(
                    &c.shift(&potential, &c.s),
                    &ds,
                    if j == nsh { 1. } else { 0. },
                    false,
                )?;
                population(c, &dp, &c.s, &mut response);
            }
            for i in 0..nsh {
                jacobian[(i, j)] = if i == j {
                    1. - response[i]
                } else {
                    -response[i]
                };
            }
            jacobian[(nsh, j)] = -response.iter().sum::<f64>();
            if j == nsh && jacobian[(nsh, j)].abs() < 1e-14 {
                use_mu = false;
            }
        }
        if !use_mu {
            for i in 0..=nsh {
                jacobian[(i, nsh)] = 0.;
                jacobian[(nsh, i)] = 0.;
            }
            jacobian[(nsh, nsh)] = 1.;
        }
        let lu = DenseLu::factor(&jacobian)?;
        for (y, value) in derivatives.iter_mut().enumerate() {
            let direction = geometry.direction(y);
            let dv_global = matrix_vector_product(&direction.gamma, &q)?;
            let mut rhs = vec![0.; nsh + 1];
            for (c, s) in clusters.iter().zip(&spectra) {
                let potential: Vec<_> = c.shells.iter().map(|&g| final_potential[g]).collect();
                let dv: Vec<_> = c.shells.iter().map(|&g| dv_global[g]).collect();
                let (dh, ds) =
                    c.fixed_fock_derivative(&direction, &potential, &dv, y, 3 * nat, 0.)?;
                let dp = s.response(&dh, &ds, 0., false)?;
                population(c, &dp, &c.s, &mut rhs[..nsh]);
                population(c, &s.density, &ds, &mut rhs[..nsh]);
            }
            rhs[nsh] = if use_mu { rhs[..nsh].iter().sum() } else { 0. };
            let response = lu.solve_vec(&rhs)?;
            let dmu = response[nsh];
            let dq = &response[..nsh];
            let dynamic = matrix_vector_product(&kernel, dq)?;
            let mut derivative = direction.classical
                + dq.iter()
                    .zip(&coulomb.shell_potential)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                + 0.5
                    * q.iter()
                        .zip(matrix_vector_product(&direction.gamma, &q)?)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
            for (c, s) in clusters.iter().zip(&spectra) {
                let potential: Vec<_> = c.shells.iter().map(|&g| final_potential[g]).collect();
                let dv: Vec<_> = c
                    .shells
                    .iter()
                    .map(|&g| dv_global[g] + dynamic[g])
                    .collect();
                let (dh, ds) =
                    c.fixed_fock_derivative(&direction, &potential, &dv, y, 3 * nat, 0.)?;
                let dp = s.response(&dh, &ds, dmu, false)?;
                let de = s.response(&dh, &ds, dmu, true)?;
                let h0 = subset(&direction.h, &c.aos, &c.aos);
                derivative += trace(&weighted(c, &dp), &c.h)
                    + trace(&weighted(c, &s.density), &h0)
                    + trace(&weighted(c, &de), &c.s)
                    + trace(&weighted(c, &s.entropy_density), &ds);
            }
            *value = derivative;
        }
    }
    let largest = clusters.iter().map(|c| c.aos.len()).max().unwrap_or(0);
    if !options.compute_derivatives {
        derivatives.clear();
    }
    finish(
        energy,
        derivatives,
        q,
        iterations,
        iterations * nf,
        nf,
        largest,
        &geometry,
    )
}
