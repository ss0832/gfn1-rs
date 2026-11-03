// SPDX-License-Identifier: GPL-3.0-or-later
//! PES-only vibrational perturbation theory (Piccardo2015GVPT).
//! Q is mass weighted in ELECTRON masses; hbar=1. Frequencies are angular
//! frequencies in atomic units, and Phi are derivatives (not Taylor coefficients):
//! V3 = sum_ijk Phi_ijk Q_i Q_j Q_k / 3!, V4 analogously / 4!.
//! Watson/Coriolis kinetic terms and anharmonic IR/Raman intensities are absent.
//! Small denominators are errors, never clamped or silently omitted.

use crate::constants::HARTREE_TO_WAVENUMBER;
use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::third_derivative::SymmetricThird;
use std::collections::BTreeMap;

pub struct NormalModeForceField {
    pub omega: Vec<f64>,
    pub cubic: SymmetricThird,
    /// Phi_iijj, in the same atomic-unit mass-weighted coordinates.
    pub quartic_semidiagonal: Matrix,
}

#[derive(Clone, Copy, Debug)]
pub struct Vpt2Options {
    pub resonance_threshold_hartree: f64,
    pub minimum_frequency_hartree: f64,
    pub max_intermediate_states: usize,
}
impl Default for Vpt2Options {
    fn default() -> Self {
        Self {
            resonance_threshold_hartree: 10.0 / HARTREE_TO_WAVENUMBER,
            minimum_frequency_hartree: 20.0 / HARTREE_TO_WAVENUMBER,
            max_intermediate_states: 100_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VibrationalLevel {
    pub quanta: Vec<u16>,
    pub harmonic_hartree: f64,
    pub cubic_correction_hartree: f64,
    pub quartic_correction_hartree: f64,
}
impl VibrationalLevel {
    pub fn energy_hartree(&self) -> f64 {
        self.harmonic_hartree + self.cubic_correction_hartree + self.quartic_correction_hartree
    }
}

#[derive(Clone, Debug)]
pub struct Vpt2Result {
    pub scope: &'static str,
    pub method_citations: Vec<&'static str>,
    pub ground: VibrationalLevel,
    pub fundamentals_cm1: Vec<f64>,
    pub excited: Vec<VibrationalLevel>,
}

impl NormalModeForceField {
    fn validate(&self, o: &Vpt2Options) -> Result<()> {
        let n = self.omega.len();
        if n == 0
            || self.cubic.n() != n
            || self.quartic_semidiagonal.rows() != n
            || self.quartic_semidiagonal.cols() != n
            || !o.resonance_threshold_hartree.is_finite()
            || o.resonance_threshold_hartree <= 0.0
            || !o.minimum_frequency_hartree.is_finite()
            || o.minimum_frequency_hartree <= 0.0
            || o.max_intermediate_states == 0
            || self
                .omega
                .iter()
                .any(|x| !x.is_finite() || *x < o.minimum_frequency_hartree)
            || self
                .quartic_semidiagonal
                .as_slice()
                .iter()
                .any(|x| !x.is_finite())
        {
            return Err(Gfn1Error::InvalidInput(
                "invalid VPT2 force field, low/imaginary mode, or budget".into(),
            ));
        }
        for i in 0..n {
            for j in 0..n {
                if (self.quartic_semidiagonal[(i, j)] - self.quartic_semidiagonal[(j, i)]).abs()
                    > 1e-12
                {
                    return Err(Gfn1Error::InvalidInput(
                        "asymmetric mode-pair quartic force field".into(),
                    ));
                }
            }
        }
        for k in 0..n {
            for j in 0..=k {
                for i in 0..=j {
                    if !self.cubic.get(i, j, k).is_finite() {
                        return Err(Gfn1Error::InvalidInput(
                            "non-finite cubic force constant".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn harmonic(&self, state: &[u16]) -> f64 {
        state
            .iter()
            .zip(&self.omega)
            .map(|(&n, &w)| (f64::from(n) + 0.5) * w)
            .sum()
    }

    fn quartic_diagonal(&self, state: &[u16]) -> f64 {
        let mut e = 0.0;
        for i in 0..self.omega.len() {
            let ni = f64::from(state[i]);
            let wi = self.omega[i];
            e += self.quartic_semidiagonal[(i, i)] * (2.0 * ni * ni + 2.0 * ni + 1.0)
                / (32.0 * wi * wi);
            for j in 0..i {
                e += self.quartic_semidiagonal[(i, j)] * (ni + 0.5) * (f64::from(state[j]) + 0.5)
                    / (4.0 * wi * self.omega[j]);
            }
        }
        e
    }

    /// Sparse V3|n>. Contributions to the SAME intermediate state are summed
    /// before squaring; this retains interference between different monomials.
    fn cubic_action(&self, state: &[u16], limit: usize) -> Result<BTreeMap<Vec<u16>, f64>> {
        let n = self.omega.len();
        let mut out = BTreeMap::new();
        for k in 0..n {
            for j in 0..=k {
                for i in 0..=j {
                    let phi = self.cubic.get(i, j, k);
                    if phi == 0.0 {
                        continue;
                    }
                    let divisor = if i == k {
                        6.0
                    } else if i == j || j == k {
                        2.0
                    } else {
                        1.0
                    };
                    let mut terms = vec![(state.to_vec(), phi / divisor)];
                    for mode in [i, j, k] {
                        let mut next = Vec::with_capacity(terms.len() * 2);
                        for (s, a) in terms {
                            let quanta = s[mode];
                            let scale = (2.0 * self.omega[mode]).sqrt();
                            if quanta > 0 {
                                let mut t = s.clone();
                                t[mode] -= 1;
                                next.push((t, a * f64::from(quanta).sqrt() / scale));
                            }
                            let mut t = s;
                            t[mode] = quanta.checked_add(1).ok_or_else(|| {
                                Gfn1Error::InvalidInput("vibrational quantum overflow".into())
                            })?;
                            next.push((t, a * f64::from(quanta + 1).sqrt() / scale));
                        }
                        terms = next;
                    }
                    for (s, a) in terms {
                        *out.entry(s).or_insert(0.0) += a;
                    }
                    if out.len() > limit {
                        return Err(Gfn1Error::InvalidInput(
                            "VPT2 intermediate-state budget exceeded".into(),
                        ));
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn level(&self, state: &[u16], options: &Vpt2Options) -> Result<VibrationalLevel> {
        self.validate(options)?;
        if state.len() != self.omega.len() {
            return Err(Gfn1Error::InvalidInput(
                "vibrational state dimension mismatch".into(),
            ));
        }
        let mut cubic = 0.0;
        for (other, amplitude) in self.cubic_action(state, options.max_intermediate_states)? {
            if other == state || amplitude == 0.0 {
                continue;
            }
            let delta: f64 = state
                .iter()
                .zip(&other)
                .zip(&self.omega)
                .map(|((&a, &b), &w)| (f64::from(a) - f64::from(b)) * w)
                .sum();
            if delta.abs() < options.resonance_threshold_hartree {
                return Err(Gfn1Error::InvalidInput(format!(
                    "PES-only VPT2 resonance: {state:?} <-> {other:?}, denominator {delta:e} Eh, coupling {amplitude:e} Eh; supply an explicit GVPT2 polyad")));
            }
            cubic += amplitude * amplitude / delta;
        }
        let level = VibrationalLevel {
            quanta: state.to_vec(),
            harmonic_hartree: self.harmonic(state),
            cubic_correction_hartree: cubic,
            quartic_correction_hartree: self.quartic_diagonal(state),
        };
        if !level.energy_hartree().is_finite() {
            return Err(Gfn1Error::InvalidInput("non-finite VPT2 energy".into()));
        }
        Ok(level)
    }
}

pub fn vpt2_pes(force_field: &NormalModeForceField, options: Vpt2Options) -> Result<Vpt2Result> {
    let n = force_field.omega.len();
    let ground = force_field.level(&vec![0; n], &options)?;
    let mut excited = Vec::with_capacity(n);
    let mut fundamentals_cm1 = Vec::with_capacity(n);
    for i in 0..n {
        let mut quanta = vec![0; n];
        quanta[i] = 1;
        let level = force_field.level(&quanta, &options)?;
        fundamentals_cm1
            .push((level.energy_hartree() - ground.energy_hartree()) * HARTREE_TO_WAVENUMBER);
        excited.push(level);
    }
    Ok(Vpt2Result {
        scope: "PES-only nonresonant VPT2",
        method_citations: vec!["Piccardo2015GVPT"],
        ground,
        fundamentals_cm1,
        excited,
    })
}

#[derive(Clone, Debug)]
pub struct Gvpt2Result {
    pub scope: &'static str,
    pub method_citations: Vec<&'static str>,
    pub states: Vec<Vec<u16>>,
    pub energies_hartree: Vec<f64>,
    pub mixing: Matrix,
    pub effective_hamiltonian_hartree: Matrix,
    /// These states were excluded from perturbative denominators and included
    /// exactly once in the variational polyad Hamiltonian.
    pub excluded_internal_intermediates: Vec<Vec<u16>>,
}

fn coordinate_power_element(initial: u16, final_state: u16, power: usize, omega: f64) -> f64 {
    let mut states = BTreeMap::from([(initial, 1.0)]);
    for _ in 0..power {
        let mut next = BTreeMap::new();
        for (n, a) in states {
            if n > 0 {
                *next.entry(n - 1).or_insert(0.0) += a * (f64::from(n) / (2.0 * omega)).sqrt();
            }
            *next.entry(n + 1).or_insert(0.0) += a * (f64::from(n + 1) / (2.0 * omega)).sqrt();
        }
        states = next;
    }
    states.get(&final_state).copied().unwrap_or(0.0)
}

/// Explicit-polyad PES-only generalized VPT2, using the Hermitian second-order
/// Van Vleck effective Hamiltonian. Internal resonant intermediate states are
/// removed from perturbation sums and their direct V3/V4 couplings are restored
/// once. Off-diagonal quartics are requested lazily from `extra_quartic` when
/// not determined by Phi_iijj; a missing component must be an error, not zero.
pub fn gvpt2_pes(
    force: &NormalModeForceField,
    states: &[Vec<u16>],
    options: Vpt2Options,
    mut extra_quartic: impl FnMut([usize; 4]) -> Result<f64>,
) -> Result<Gvpt2Result> {
    force.validate(&options)?;
    let n = force.omega.len();
    let p = states.len();
    if p == 0
        || p > 256
        || states
            .iter()
            .any(|s| s.len() != n || s.iter().any(|&q| q > u16::MAX - 4))
        || states
            .iter()
            .enumerate()
            .any(|(i, s)| states[..i].contains(s))
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid, duplicate or oversized GVPT2 polyad".into(),
        ));
    }
    let energy: Vec<f64> = states.iter().map(|s| force.harmonic(s)).collect();
    let actions: Vec<_> = states
        .iter()
        .map(|s| force.cubic_action(s, options.max_intermediate_states))
        .collect::<Result<_>>()?;
    let mut effective = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..=i {
            let mut value = if i == j {
                energy[i] + force.quartic_diagonal(&states[i])
            } else {
                actions[i].get(&states[j]).copied().unwrap_or(0.0)
            };
            if i != j
                && states[i]
                    .iter()
                    .zip(&states[j])
                    .map(|(&a, &b)| usize::from(a.abs_diff(b)))
                    .sum::<usize>()
                    <= 4
                && (states[i].iter().map(|&q| usize::from(q)).sum::<usize>()
                    + states[j].iter().map(|&q| usize::from(q)).sum::<usize>())
                    % 2
                    == 0
            {
                for d in 0..n {
                    for c in 0..=d {
                        for b in 0..=c {
                            for a in 0..=b {
                                let indices = [a, b, c, d];
                                let mut count = vec![0usize; n];
                                for &k in &indices {
                                    count[k] += 1;
                                }
                                if (0..n).any(|k| {
                                    usize::from(states[i][k].abs_diff(states[j][k])) > count[k]
                                        || (usize::from(states[i][k].abs_diff(states[j][k]))
                                            + count[k])
                                            % 2
                                            != 0
                                }) {
                                    continue;
                                }
                                let mut matrix_element = 1.0;
                                let mut factorial = 1.0;
                                for k in 0..n {
                                    if count[k] > 0 {
                                        matrix_element *= coordinate_power_element(
                                            states[j][k],
                                            states[i][k],
                                            count[k],
                                            force.omega[k],
                                        );
                                        factorial *= [1.0, 1.0, 2.0, 6.0, 24.0][count[k]];
                                    }
                                }
                                if matrix_element == 0.0 {
                                    continue;
                                }
                                let phi = if a == b && c == d {
                                    force.quartic_semidiagonal[(a, c)]
                                } else {
                                    extra_quartic(indices)?
                                };
                                if !phi.is_finite() {
                                    return Err(Gfn1Error::InvalidInput(
                                        "non-finite off-diagonal quartic".into(),
                                    ));
                                }
                                value += phi * matrix_element / factorial;
                            }
                        }
                    }
                }
            }
            for (intermediate, &left) in &actions[i] {
                if left == 0.0 || states.contains(intermediate) {
                    continue;
                }
                let right = actions[j].get(intermediate).copied().unwrap_or(0.0);
                if right == 0.0 {
                    continue;
                }
                let ek = force.harmonic(intermediate);
                let di = energy[i] - ek;
                let dj = energy[j] - ek;
                if di.abs() < options.resonance_threshold_hartree
                    || dj.abs() < options.resonance_threshold_hartree
                {
                    return Err(Gfn1Error::InvalidInput(format!(
                        "GVPT2 polyad is incomplete: resonant external state {intermediate:?}"
                    )));
                }
                value += 0.5 * left * right * (1.0 / di + 1.0 / dj);
            }
            effective[(i, j)] = value;
            effective[(j, i)] = value;
        }
    }
    let eigen = crate::linalg::symmetric_eigen(&effective)?;
    Ok(Gvpt2Result {
        scope: "PES-only explicit-polyad GVPT2",
        method_citations: vec!["Piccardo2015GVPT"],
        states: states.to_vec(),
        energies_hartree: eigen.values,
        mixing: eigen.vectors,
        effective_hamiltonian_hartree: effective,
        excluded_internal_intermediates: states.to_vec(),
    })
}

/// Contract a cubic form into selected modes with the degree-3 polarization
/// identity. The electronic reference belongs to `evaluate` and is reused.
pub fn cubic_from_directional(
    directions: &[Vec<f64>],
    evaluate: impl FnMut(&[f64]) -> Result<f64>,
) -> Result<SymmetricThird> {
    cubic_from_directional_with_budget(directions, evaluate, 512 * 1024 * 1024)
}

pub fn cubic_from_directional_with_budget(
    directions: &[Vec<f64>],
    mut evaluate: impl FnMut(&[f64]) -> Result<f64>,
    budget: usize,
) -> Result<SymmetricThird> {
    let n = directions.len();
    let d = directions.first().map_or(0, Vec::len);
    if directions
        .iter()
        .any(|v| v.len() != d || v.iter().any(|x| !x.is_finite()))
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid cubic contraction directions".into(),
        ));
    }
    let mut out = SymmetricThird::try_zeros(n, budget)?;
    let mut diag = Vec::with_capacity(n);
    for v in directions {
        diag.push(evaluate(v)?);
    }
    let mut pairs = Matrix::zeros(n, n);
    let mut v = vec![0.0; d];
    for i in 0..n {
        pairs[(i, i)] = 8.0 * diag[i];
        for j in 0..i {
            for c in 0..d {
                v[c] = directions[i][c] + directions[j][c];
            }
            let q = evaluate(&v)?;
            pairs[(i, j)] = q;
            pairs[(j, i)] = q;
        }
    }
    for k in 0..n {
        for j in 0..=k {
            for i in 0..=j {
                let phi = if i == k {
                    diag[i]
                } else {
                    for c in 0..d {
                        v[c] = directions[i][c] + directions[j][c] + directions[k][c];
                    }
                    (evaluate(&v)? - pairs[(i, j)] - pairs[(i, k)] - pairs[(j, k)]
                        + diag[i]
                        + diag[j]
                        + diag[k])
                        / 6.0
                };
                if !phi.is_finite() {
                    return Err(Gfn1Error::InvalidInput(
                        "non-finite cubic contraction".into(),
                    ));
                }
                out.add(i, j, k, phi);
            }
        }
    }
    Ok(out)
}

pub struct MolecularVpt2Result {
    pub normal_modes: crate::vibrational::MolecularVibrations,
    pub force_field: NormalModeForceField,
    pub spectrum: Vpt2Result,
    pub residual_gradient_hartree_per_bohr: f64,
}

/// Molecular analytic GFN1 PES-only VPT2. No Cartesian FC4 is allocated.
/// Masses are per-atom isotope masses in amu. A stationary minimum is required;
/// optimization is a separate explicit operation.
pub fn molecular_vpt2_pes(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    electronic: crate::ElectronicOptions,
    masses_amu: &[f64],
    options: Vpt2Options,
    gradient_tolerance: f64,
) -> Result<MolecularVpt2Result> {
    crate::terms::require_order(&electronic, params, 4, "molecular PES-only VPT2")?;
    if system.lattice.is_some() || !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
        return Err(Gfn1Error::InvalidInput(
            "VPT2 requires a molecule and positive stationarity tolerance".into(),
        ));
    }
    let gradient = crate::analytic_gradient(
        system,
        params,
        crate::AnalyticGradientOptions {
            electronic: electronic.clone(),
            ..Default::default()
        },
    )?;
    if !gradient.electronic_result.converged || gradient.max_gradient > gradient_tolerance {
        return Err(Gfn1Error::InvalidInput(format!(
            "VPT2 requires a stationary minimum: max gradient {:e} > {:e}",
            gradient.max_gradient, gradient_tolerance
        )));
    }
    let hoptions = crate::AnalyticHessianOptions {
        electronic_options: electronic,
        ..Default::default()
    };
    let hessian = crate::analytic_hessian_from_result(
        system,
        params,
        Some(gradient.electronic_result),
        hoptions.clone(),
    )?;
    let positions: Vec<_> = system.atoms.iter().map(|a| a.position).collect();
    let normal_modes = crate::vibrational::molecular_vibrational_analysis(
        &hessian.hessian,
        &positions,
        masses_amu,
        Default::default(),
    )?;
    let omega = normal_modes.angular_frequencies_au()?;
    let directions = normal_modes.atomic_unit_directions();
    let cutoff = hoptions.electronic_options.hamiltonian.coordination_cutoff;
    let reference = crate::third_derivative::CubicReference::build(
        system,
        params,
        &hoptions.electronic_options,
        512 * 1024 * 1024,
    )?;
    let cubic = cubic_from_directional(&directions, |v| reference.directional(v))?;
    drop(reference);
    let quartic_semidiagonal = crate::fourth_derivative_analytic_semidiagonal(
        system,
        params,
        &hoptions,
        cutoff,
        &directions,
    )?;
    let force_field = NormalModeForceField {
        omega,
        cubic,
        quartic_semidiagonal,
    };
    let spectrum = vpt2_pes(&force_field, options)?;
    Ok(MolecularVpt2Result {
        normal_modes,
        force_field,
        spectrum,
        residual_gradient_hartree_per_bohr: gradient.max_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::symmetric_eigen;
    fn model(w: f64, g: f64, h: f64) -> NormalModeForceField {
        let mut cubic = SymmetricThird::zeros(1);
        cubic.add(0, 0, 0, g);
        NormalModeForceField {
            omega: vec![w],
            cubic,
            quartic_semidiagonal: Matrix::from_vec(1, 1, vec![h]).unwrap(),
        }
    }
    #[test]
    fn one_mode_matches_closed_form_and_morse() {
        let (w, g, h) = (0.7, 0.03, 0.02);
        let f = model(w, g, h);
        for n in 0..4 {
            let got = f.level(&[n], &Default::default()).unwrap();
            let n = f64::from(n);
            assert!(
                (got.cubic_correction_hartree
                    + g * g * (30.0 * n * n + 30.0 * n + 11.0) / (288.0 * w.powi(4)))
                .abs()
                    < 1e-15
            );
            assert!(
                (got.quartic_correction_hartree
                    - h * (2.0 * n * n + 2.0 * n + 1.0) / (32.0 * w * w))
                    .abs()
                    < 1e-15
            );
        }
        let (depth, a) = (1.5_f64, 0.12_f64);
        let w = (2.0 * depth * a * a).sqrt();
        let f = model(w, -6.0 * depth * a.powi(3), 14.0 * depth * a.powi(4));
        let result = vpt2_pes(&f, Default::default()).unwrap();
        assert!((result.ground.energy_hartree() - (0.5 * w - a * a / 8.0)).abs() < 1e-14);
        assert!((result.fundamentals_cm1[0] / HARTREE_TO_WAVENUMBER - (w - a * a)).abs() < 1e-14);
    }
    #[test]
    fn polynomial_ho_diagonalization_has_the_same_lambda_squared_limit() {
        let f = model(0.8, 0.05, 0.03);
        let expected = f.level(&[0], &Default::default()).unwrap();
        let mut q = Matrix::zeros(16, 16);
        for i in 0..15 {
            let x = ((i + 1) as f64 / (2.0 * f.omega[0])).sqrt();
            q[(i, i + 1)] = x;
            q[(i + 1, i)] = x;
        }
        let q2 = q.matmul(&q).unwrap();
        let q3 = q2.matmul(&q).unwrap();
        let q4 = q2.matmul(&q2).unwrap();
        let coefficient = |lambda: f64| {
            let mut h = Matrix::zeros(16, 16);
            for i in 0..16 {
                for j in 0..16 {
                    h[(i, j)] = lambda * 0.05 * q3[(i, j)] / 6.0
                        + lambda * lambda * 0.03 * q4[(i, j)] / 24.0;
                }
                h[(i, i)] += (i as f64 + 0.5) * 0.8;
            }
            (symmetric_eigen(&h).unwrap().values[0] - 0.4) / (lambda * lambda)
        };
        let extrapolated = (4.0 * coefficient(0.01) - coefficient(0.02)) / 3.0;
        assert!(
            (extrapolated
                - expected.cubic_correction_hartree
                - expected.quartic_correction_hartree)
                .abs()
                < 1e-9
        );
    }
    #[test]
    fn resonance_is_not_hidden_by_denominator_clamping() {
        let mut cubic = SymmetricThird::zeros(2);
        cubic.add(0, 0, 1, 0.1);
        let f = NormalModeForceField {
            omega: vec![1.0, 2.0],
            cubic,
            quartic_semidiagonal: Matrix::zeros(2, 2),
        };
        let err = f
            .level(&[0, 1], &Default::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("resonance") && err.contains("[2, 0]"));
    }
    #[test]
    fn fermi_polyad_matches_weak_coupling_ho_diagonalization() {
        let compare = |lambda: f64| {
            let mut cubic = SymmetricThird::zeros(2);
            cubic.add(0, 0, 1, 0.1 * lambda);
            let force = NormalModeForceField {
                omega: vec![1.0, 2.0],
                cubic,
                quartic_semidiagonal: Matrix::zeros(2, 2),
            };
            let gvpt = gvpt2_pes(
                &force,
                &[vec![2, 0], vec![0, 1]],
                Default::default(),
                |_| Err(Gfn1Error::InvalidInput("unexpected extra quartic".into())),
            )
            .unwrap();
            let size = 8;
            let mut q0 = Matrix::zeros(size, size);
            let mut q1 = Matrix::zeros(size, size);
            for i in 0..size - 1 {
                q0[(i, i + 1)] = ((i + 1) as f64 / 2.0).sqrt();
                q0[(i + 1, i)] = q0[(i, i + 1)];
                q1[(i, i + 1)] = ((i + 1) as f64 / 4.0).sqrt();
                q1[(i + 1, i)] = q1[(i, i + 1)];
            }
            let q02 = q0.matmul(&q0).unwrap();
            let mut h = Matrix::zeros(size * size, size * size);
            for i in 0..size * size {
                for j in 0..size * size {
                    h[(i, j)] =
                        lambda * 0.05 * q02[(i / size, j / size)] * q1[(i % size, j % size)];
                }
                h[(i, i)] += (i / size) as f64 + 0.5 + 2.0 * ((i % size) as f64 + 0.5);
            }
            let exact = symmetric_eigen(&h).unwrap().values;
            (gvpt.energies_hartree[0] - exact[2])
                .abs()
                .max((gvpt.energies_hartree[1] - exact[3]).abs())
        };
        let large = compare(0.04);
        let small = compare(0.02);
        assert!(
            large < 1e-6 && small < 0.2 * large,
            "GVPT2 lambda^3 remainder {large:e} -> {small:e}"
        );
    }
}
