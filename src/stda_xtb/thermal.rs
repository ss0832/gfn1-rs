// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-temperature simplified response in complete interband blocks.
use super::dual::Dual as D;
use super::gradient::sqrt_and_derivatives;
use super::ground::{ground_directional, tables, Orbitals, EV_PER_HARTREE, VTB_KB_EV};
use super::{NativeStdaOptions, NativeStdaResult, NativeStdaState};
use crate::error::{Gfn1Error, Result};
use crate::integrals::contracted_pair;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::PeriodicSystem;
use crate::td::hessian::{diagonal_jet_sqrt, occupation_jet, orbital_blocks};

const INTERBAND_WIDTH: f64 = 1e-6;
const FRAME_TOL: f64 = 1e-8;
const KIA: f64 = 0.036749320;

fn invalid(message: &str) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}

pub(super) fn use_ensemble(orbital: &Orbitals, options: &NativeStdaOptions) -> Result<bool> {
    if options.electronic_temperature <= 0. {
        return if options.thermal_response == Some(true) {
            Err(invalid(
                "ensemble sTDA requires a positive electronic temperature",
            ))
        } else {
            Ok(false)
        };
    }
    Ok(options.thermal_response.unwrap_or_else(|| {
        let range = 2.8 * options.energy_window_ev / EV_PER_HARTREE;
        let nocc = orbital.nocc;
        let active = |i: usize| if i < nocc {
            orbital.energies[i] > orbital.energies[nocc] - range
        } else {
            orbital.energies[i] <= 3.
                && orbital.energies[i] < orbital.energies[nocc - 1] + range
        };
        orbital.occupations.iter().any(|&f| f > 1e-8 && f < 2. - 1e-8)
            || orbital.unshifted_energies.windows(2).enumerate().any(|(i, e)| {
                e[1] - e[0] < INTERBAND_WIDTH && (active(i) || active(i + 1))
            })
    }))
}

pub(super) struct Ensemble<'a> {
    system: &'a PeriodicSystem,
    orbital: &'a Orbitals,
    options: &'a NativeStdaOptions,
    pairs: Vec<(usize, usize)>,
    active: Vec<usize>,
    mean_blocks: Vec<Vec<usize>>,
    frame_blocks: Vec<Vec<usize>>,
    metric: crate::linalg::EigenDecomposition,
    lowdin: Matrix,
    workers: usize,
}

impl<'a> Ensemble<'a> {
    pub(super) fn new(
        system: &'a PeriodicSystem,
        orbital: &'a Orbitals,
        options: &'a NativeStdaOptions,
    ) -> Result<Self> {
        let e = &orbital.unshifted_energies;
        let f = &orbital.occupations;
        let band = orbital_blocks(e, INTERBAND_WIDTH);
        let frame = orbital_blocks(e, FRAME_TOL);
        let limit = 2.8 * options.energy_window_ev / EV_PER_HARTREE;
        let mut active = Vec::new();
        let mut pairs = Vec::new();
        let mut mean_blocks = std::collections::BTreeMap::<_, Vec<_>>::new();
        let mut frame_blocks = std::collections::BTreeMap::<_, Vec<_>>::new();
        // Band decisions use mean occupations and mean energies so a band
        // is admitted as a complete tensor product, including near a cutoff.
        let mut band_energy = vec![0.; e.len()];
        let mut band_occupation = vec![0.; e.len()];
        let mut band_count = vec![0.; e.len()];
        for (i, &b) in band.iter().enumerate() {
            band_energy[b] += e[i];
            band_occupation[b] += f[i];
            band_count[b] += 1.;
        }
        for i in 0..e.len() {
            for a in i + 1..e.len() {
                let (bi, ba) = (band[i], band[a]);
                if bi == ba
                    || band_occupation[bi] / band_count[bi] - band_occupation[ba] / band_count[ba]
                        <= 1e-10
                    || band_energy[ba] / band_count[ba] - band_energy[bi] / band_count[bi] > limit
                {
                    continue;
                }
                let p = pairs.len();
                pairs.push((i, a));
                active.extend([i, a]);
                mean_blocks.entry((bi, ba)).or_default().push(p);
                frame_blocks
                    .entry((frame[i], frame[a]))
                    .or_default()
                    .push(p);
            }
        }
        if pairs.is_empty() || pairs.len() > options.max_configurations {
            return Err(invalid("ensemble sTDA interband configuration count is empty or exceeds max_configurations"));
        }
        active.sort_unstable();
        active.dedup();
        let size = pairs.len() as u128;
        let orbitals = e.len() as u128;
        let charges = active.len() as u128;
        let per_worker = 1024 * orbitals * orbitals
            + 72 * charges * (charges + 1) / 2 * system.atoms.len() as u128;
        let shared = 48 * size * size;
        let need = shared + per_worker;
        if need > options.memory_budget_bytes as u128 {
            return Err(invalid("ensemble sTDA exceeds memory_budget_bytes"));
        }
        let metric = symmetric_eigen(&orbital.overlap)?;
        if metric.values.iter().any(|&x| x <= 1e-10) {
            return Err(invalid("sTDA Lowdin metric is singular"));
        }
        let mut scaled = metric.vectors.clone();
        for i in 0..scaled.rows() {
            for j in 0..scaled.cols() {
                scaled[(i, j)] *= metric.values[j].sqrt();
            }
        }
        let lowdin = scaled
            .matmul(&metric.vectors.transpose())?
            .matmul(&orbital.coefficients)?;
        Ok(Self {
            system,
            orbital,
            options,
            pairs,
            active,
            lowdin,
            metric,
            workers: (((options.memory_budget_bytes as u128 - shared) / per_worker.max(1))
                .min(crate::resources::effective_threads() as u128)
                .clamp(1, 8)) as usize,
            mean_blocks: mean_blocks.into_values().collect(),
            frame_blocks: frame_blocks.into_values().collect(),
        })
    }

    pub(super) fn solve(&self, all: bool) -> Result<NativeStdaResult> {
        let response = self.matrix(None, false)?;
        let n = self.pairs.len();
        let mut matrix = Matrix::zeros(n, n);
        // At the reference W and the gap are diagonal in the canonical MOs.
        for i in 0..n {
            for j in 0..=i {
                matrix[(i, j)] = response.reference_entry(i, j);
                matrix[(j, i)] = matrix[(i, j)];
            }
        }
        let eigen = symmetric_eigen(&matrix)?;
        let c = &self.orbital.coefficients;
        let nao = c.rows();
        let mut dipoles = [
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
            Matrix::zeros(nao, nao),
        ];
        if !self.options.triplet {
            let atoms = &self.system.atoms;
            let mut ao = dipoles.clone();
            for (i, left) in self.orbital.aos.iter().enumerate() {
                let ri = atoms[left.atom_index].position;
                for (j, right) in self.orbital.aos[..=i].iter().enumerate() {
                    let rj = atoms[right.atom_index].position;
                    let integral = contracted_pair(left, right, ri, rj);
                    for (axis, value) in
                        [integral.1, integral.2, integral.3].into_iter().enumerate()
                    {
                        ao[axis][(i, j)] = -(value + rj.to_array()[axis] * integral.0);
                        ao[axis][(j, i)] = ao[axis][(i, j)];
                    }
                }
            }
            for axis in 0..3 {
                dipoles[axis] = c.transpose().matmul(&ao[axis])?.matmul(c)?;
            }
        }
        let mut states = Vec::new();
        for (root, &energy) in eigen.values.iter().enumerate() {
            if !all && (energy <= 0. || energy * EV_PER_HARTREE > self.options.energy_window_ev) {
                continue;
            }
            if !all && states.len() >= self.options.n_states {
                break;
            }
            let amplitudes = eigen.vectors.column(root);
            let dipole: [f64; 3] = std::array::from_fn(|axis| {
                self.pairs
                    .iter()
                    .zip(&amplitudes)
                    .enumerate()
                    .map(|(p, (&(i, a), &x))| {
                        std::f64::consts::SQRT_2
                            * x
                            * response.diagonal_weight[p]
                            * dipoles[axis][(i, a)]
                    })
                    .sum()
            });
            states.push(NativeStdaState {
                excitation_hartree: energy,
                excitation_ev: energy * EV_PER_HARTREE,
                oscillator_strength: 2. / 3.
                    * energy.max(0.)
                    * dipole.iter().map(|v| v * v).sum::<f64>(),
                transition_dipole: dipole,
                amplitudes,
            });
        }
        if states.is_empty() {
            return Err(invalid(
                "no positive ensemble sTDA excitation in the requested window",
            ));
        }
        Ok(NativeStdaResult {
            thermal_response: true,
            electronic_temperature_kelvin: self.orbital.electronic_temperature_kelvin,
            maximum_occupation_deviation: self.orbital.maximum_occupation_deviation,
            lowest_ci_eigenvalue_hartree: eigen.values[0],
            nonpositive_ci_roots: eigen.values.iter().filter(|&&v| v <= 0.).count(),
            ci_eigenvalues_hartree: eigen.values,
            states,
            configurations: self.pairs.clone(),
            primary_configuration_count: n,
            promoted_configuration_count: 0,
            orbital_energies_hartree: self.orbital.unshifted_energies.clone(),
            vtb_charges: self.orbital.vtb_charges.clone(),
            basis_size: nao,
        })
    }

    pub(super) fn directional(
        &self,
        v: &[f64],
        second: bool,
        roots: std::ops::Range<usize>,
        spectrum: &NativeStdaResult,
    ) -> Result<(f64, f64)> {
        let matrix = self.matrix(Some(v), second)?;
        let mut derivative = 0.;
        let mut curvature = 0.;
        let states = &spectrum.states[roots.clone()];
        let actions = matrix.actions(states);
        for (state, action) in states.iter().zip(actions) {
            for (&x, a) in state.amplitudes.iter().zip(&action) {
                derivative += x * a.derivative;
                curvature += x * a.second;
            }
            if second {
                for (other, state2) in spectrum.states.iter().enumerate() {
                    if !roots.contains(&other) {
                        let mix: f64 = state2
                            .amplitudes
                            .iter()
                            .zip(&action)
                            .map(|(&x, a)| x * a.derivative)
                            .sum();
                        curvature += 2. * mix.powi(2)
                            / (state.excitation_hartree - state2.excitation_hartree);
                    }
                }
            }
        }
        let n = roots.len() as f64;
        Ok((derivative / n, curvature / n))
    }

    fn matrix(&self, direction: Option<&[f64]>, second: bool) -> Result<ResponseMatrix<'_>> {
        let orbital = self.orbital;
        let nao = orbital.aos.len();
        let nat = self.system.atoms.len();
        let n = self.pairs.len();
        let eps = &orbital.unshifted_energies;
        let mut low1 = Matrix::zeros(nao, nao);
        let mut low2 = Matrix::zeros(nao, nao);
        let mut f1 = Matrix::zeros(nao, nao);
        let mut f2 = Matrix::zeros(nao, nao);
        if let Some(v) = direction {
            let response = ground_directional(self.system, orbital, v, second)?;
            let (sqrt, dsqrt, d2sqrt) =
                sqrt_and_derivatives(&self.metric, &response.ds, second.then_some(&response.d2s))?;
            let c = &orbital.coefficients;
            let low1a = dsqrt.matmul(c)?;
            let low1b = sqrt.matmul(&response.dc)?;
            for (r, (&a, &b)) in low1
                .as_mut_slice()
                .iter_mut()
                .zip(low1a.as_slice().iter().zip(low1b.as_slice()))
            {
                *r = a + b;
            }
            if second {
                let a = d2sqrt.matmul(c)?;
                let b = dsqrt.matmul(&response.dc)?;
                let c = sqrt.matmul(&response.d2c)?;
                for (r, ((&a, &b), &c)) in low2
                    .as_mut_slice()
                    .iter_mut()
                    .zip(a.as_slice().iter().zip(b.as_slice()).zip(c.as_slice()))
                {
                    *r = a + 2. * b + c;
                }
                f2 = response.fock_second;
            }
            f1 = response.fock_first;
        }
        let occ = occupation_jet(
            eps,
            &orbital.occupations,
            self.options.electronic_temperature * VTB_KB_EV / EV_PER_HARTREE,
            &f1,
            second.then_some(&f2),
            FRAME_TOL,
            direction.is_some(),
        );
        let mut gaps = Vec::new();
        let mut weights = Vec::new();
        let mut diagonal_weight = vec![0.; n];
        let mut diagonal_gap = vec![0.; n];
        for block in &self.frame_blocks {
            let len = block.len();
            let mut gap = vec![vec![D::default(); len]; len];
            let mut df = gap.clone();
            let fock =
                |i, j| D::seed_second(if i == j { eps[i] } else { 0. }, f1[(i, j)], f2[(i, j)]);
            for (p, &row) in block.iter().enumerate() {
                let (i, a) = self.pairs[row];
                for (q, &col) in block.iter().enumerate() {
                    let (j, b) = self.pairs[col];
                    if i == j {
                        gap[p][q] += fock(a, b);
                        df[p][q] -= 0.5 * occ[a][b];
                    }
                    if a == b {
                        gap[p][q] -= fock(i, j);
                        df[p][q] += 0.5 * occ[i][j];
                    }
                    gap[p][q] += orbital.virtual_level_shift * df[p][q];
                }
                diagonal_weight[row] = df[p][p].value.sqrt();
                diagonal_gap[row] = gap[p][p].value;
            }
            weights.push(diagonal_jet_sqrt(&df));
            gaps.push(gap);
        }
        let mut index = vec![usize::MAX; nao];
        let active = self.active.len();
        let mut charges = vec![vec![D::default(); nat]; active * (active + 1) / 2];
        for (i, &p) in self.active.iter().enumerate() {
            index[p] = i;
            for (j, &q) in self.active[..=i].iter().enumerate() {
                for (mu, ao) in orbital.aos.iter().enumerate() {
                    let c = |k| D::seed_second(self.lowdin[(mu, k)], low1[(mu, k)], low2[(mu, k)]);
                    charges[packed(i, j)][ao.atom_index] += c(p) * c(q);
                }
            }
        }
        let mut j_kernel = vec![vec![D::default(); nat]; nat];
        let mut k_kernel = j_kernel.clone();
        for i in 0..nat {
            for j in 0..=i {
                let hardness = |a: usize| {
                    tables()["default_hardness"][usize::from(self.system.atoms[a].z) - 1]
                        .as_f64()
                        .expect("hardness")
                };
                let eta = 0.5 * (hardness(i) + hardness(j));
                let dr = self.system.atoms[i].position - self.system.atoms[j].position;
                let squared = (0..3).fold(D::default(), |sum, axis| {
                    let step = direction.map_or(0., |v| v[3 * i + axis] - v[3 * j + axis]);
                    sum + D::seed(dr.to_array()[axis], step).powi(2)
                });
                let jv = (squared.powi(2) + (0.5 * eta).powi(-4)).powf(-0.25);
                let kv = (squared + eta.powi(-2)).powf(-0.5);
                j_kernel[i][j] = jv;
                j_kernel[j][i] = jv;
                k_kernel[i][j] = kv;
                k_kernel[j][i] = kv;
            }
        }
        let j_potential = charges
            .iter()
            .map(|q| j_kernel.iter().map(|row| dot(row, q)).collect())
            .collect();
        let k_potential: Vec<Vec<D>> = self
            .pairs
            .iter()
            .map(|&(i, a)| {
                let q = &charges[packed(index[i], index[a])];
                k_kernel.iter().map(|row| dot(row, q)).collect()
            })
            .collect();
        let mut shifts = vec![D::default(); n];
        for block in &self.mean_blocks {
            let exchange = block.iter().fold(D::default(), |sum, &p| {
                let (i, a) = self.pairs[p];
                sum + dot(&charges[packed(index[i], index[a])], &k_potential[p])
            }) / block.len() as f64;
            let shift = (0.5 * KIA) / (1. + (exchange / (0.1 * KIA)).powi(4));
            for &p in block {
                shifts[p] = shift;
            }
        }
        Ok(ResponseMatrix {
            pairs: self
                .pairs
                .iter()
                .map(|&(i, a)| (index[i], index[a]))
                .collect(),
            blocks: &self.frame_blocks,
            gaps,
            weights,
            shifts,
            diagonal_weight,
            diagonal_gap,
            charges,
            j_potential,
            k_potential,
            singlet: !self.options.triplet,
        })
    }
}

struct ResponseMatrix<'a> {
    pairs: Vec<(usize, usize)>,
    blocks: &'a [Vec<usize>],
    gaps: Vec<Vec<Vec<D>>>,
    weights: Vec<Vec<Vec<D>>>,
    shifts: Vec<D>,
    diagonal_weight: Vec<f64>,
    diagonal_gap: Vec<f64>,
    charges: Vec<Vec<D>>,
    j_potential: Vec<Vec<D>>,
    k_potential: Vec<Vec<D>>,
    singlet: bool,
}

impl ResponseMatrix<'_> {
    fn interaction(&self, p: usize, q: usize) -> D {
        let (i, a) = self.pairs[p];
        let (j, b) = self.pairs[q];
        let mut value = -dot(&self.charges[packed(i, j)], &self.j_potential[packed(a, b)]);
        if self.singlet {
            value += 2. * dot(&self.charges[packed(i, a)], &self.k_potential[q]);
        }
        if p == q {
            value += self.shifts[p];
        }
        value
    }

    fn reference_entry(&self, p: usize, q: usize) -> f64 {
        let mut value =
            self.diagonal_weight[p] * self.interaction(p, q).value * self.diagonal_weight[q];
        if p == q {
            value += self.diagonal_gap[p];
        }
        value
    }

    fn actions(&self, states: &[NativeStdaState]) -> Vec<Vec<D>> {
        let mut results = Vec::with_capacity(states.len());
        let mut weights = Vec::with_capacity(states.len());
        for state in states {
            let x: Vec<_> = state.amplitudes.iter().copied().map(D::constant).collect();
            results.push(self.block_action(&self.gaps, &x));
            weights.push(self.block_action(&self.weights, &x));
        }
        let n = self.pairs.len();
        let mut coupled = vec![vec![D::default(); n]; states.len()];
        for i in 0..n {
            for j in 0..=i {
                let entry = self.interaction(i, j);
                for (output, weight) in coupled.iter_mut().zip(&weights) {
                    output[i] += entry * weight[j];
                    if i != j {
                        output[j] += entry * weight[i];
                    }
                }
            }
        }
        for (result, coupled) in results.iter_mut().zip(coupled) {
            for (out, update) in result.iter_mut().zip(self.block_action(&self.weights, &coupled)) {
                *out += update;
            }
        }
        results
    }

    fn block_action(&self, matrices: &[Vec<Vec<D>>], x: &[D]) -> Vec<D> {
        let mut result = vec![D::default(); x.len()];
        for (block, matrix) in self.blocks.iter().zip(matrices) {
            for (&p, row) in block.iter().zip(matrix) {
                result[p] = row
                    .iter()
                    .zip(block)
                    .fold(D::default(), |sum, (&a, &q)| sum + a * x[q]);
            }
        }
        result
    }
}

fn packed(i: usize, j: usize) -> usize {
    let hi = i.max(j);
    hi * (hi + 1) / 2 + i.min(j)
}

fn dot(a: &[D], b: &[D]) -> D {
    a.iter()
        .zip(b)
        .fold(D::default(), |sum, (&a, &b)| sum + a * b)
}

pub(super) fn spectrum(
    system: &PeriodicSystem,
    orbital: &Orbitals,
    options: &NativeStdaOptions,
    all: bool,
) -> Result<NativeStdaResult> {
    Ensemble::new(system, orbital, options)?.solve(all)
}

pub(super) fn derivatives(
    ensemble: &Ensemble<'_>,
    roots: std::ops::Range<usize>,
    spectrum: &NativeStdaResult,
    second: bool,
) -> Result<(Vec<Vec3>, Option<Matrix>)> {
    let ndof = 3 * ensemble.system.atoms.len();
    let evaluate =
        |direction: &[f64]| ensemble.directional(direction, second, roots.clone(), spectrum);
    let (values, hessian) = if second {
        let (values, hessian) =
            crate::td::hessian::directional_hessian(ndof, ensemble.workers, evaluate)?;
        (values, Some(hessian))
    } else {
        let mut direction = vec![0.; ndof];
        let mut values = Vec::with_capacity(ndof);
        for i in 0..ndof {
            direction[i] = 1.;
            values.push(evaluate(&direction)?.0);
            direction[i] = 0.;
        }
        (values, None)
    };
    let gradient = values
        .chunks_exact(3)
        .map(|v| Vec3::new(v[0], v[1], v[2]))
        .collect();
    Ok((gradient, hessian))
}
