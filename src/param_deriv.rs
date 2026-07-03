// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-difference derivatives of GFN1-xTB observables with respect to model
//! parameters.
//!
//! Parameters are addressed with [`ParameterTarget`]; each derivative is a
//! central finite difference that perturbs one scalar by `±step`, rebuilds the
//! (consistent) parameter set through [`Gfn1Parameters::with_parameter`], and
//! re-runs the electronic structure. This is the GFN1 analogue of the parameter
//! refit/sensitivity tooling and is the basis for external parameter optimizers.

use crate::electronic::{run_electronic, ElectronicOptions};
use crate::error::{Gfn1Error, Result};
use crate::gradient::{analytic_gradient, AnalyticGradientOptions};
use crate::hessian::{analytic_hessian, AnalyticHessianOptions};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::{Gfn1Parameters, ParameterTarget};
use crate::pbc::{pbc_analytic_gradient, pbc_stress, PbcOptions};
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct ParamDerivativeOptions {
    /// Central-difference step in the (absolute) parameter value.
    pub step: f64,
    /// Electronic-structure options used for every perturbed evaluation.
    pub electronic: ElectronicOptions,
    /// Also differentiate the analytic forces (`dF/dp`, Hartree/Bohr per unit
    /// parameter).
    pub include_forces: bool,
    /// Also differentiate the periodic stress tensor (`dsigma/dp`); requires a
    /// periodic system.
    pub include_stress: bool,
}

impl Default for ParamDerivativeOptions {
    fn default() -> Self {
        Self {
            step: 1.0e-4,
            electronic: ElectronicOptions::default(),
            include_forces: false,
            include_stress: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamDerivative {
    pub target: ParameterTarget,
    /// Unperturbed value of the parameter.
    pub value: f64,
    /// `d(total free energy)/dp` (Hartree per unit parameter).
    pub energy_derivative: f64,
    /// `dF_atom/dp` (Hartree/Bohr per unit parameter), present iff
    /// [`ParamDerivativeOptions::include_forces`] was set.
    pub force_derivatives: Option<Vec<Vec3>>,
    /// `dsigma/dp` (3x3) when [`ParamDerivativeOptions::include_stress`] was set.
    pub stress_derivative: Option<[[f64; 3]; 3]>,
}

struct Observables {
    energy: f64,
    forces: Option<Vec<Vec3>>,
    stress: Option<[[f64; 3]; 3]>,
}

fn evaluate_observables(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ParamDerivativeOptions,
) -> Result<Observables> {
    let periodic = system.lattice.is_some();
    if options.include_stress {
        if !periodic {
            return Err(Gfn1Error::InvalidInput(
                "parameter stress derivatives require a periodic system".to_string(),
            ));
        }
        let pbc = PbcOptions::for_boundary(options.electronic.boundary);
        let st = pbc_stress(system, params, &options.electronic, &pbc)?;
        let gr = pbc_analytic_gradient(system, params, &options.electronic, &pbc)?;
        let mut stress = [[0.0_f64; 3]; 3];
        for (i, row) in stress.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = st.stress[(i, j)];
            }
        }
        return Ok(Observables {
            energy: gr.total_energy,
            forces: options.include_forces.then_some(gr.forces),
            stress: Some(stress),
        });
    }
    if options.include_forces {
        let gradient = if periodic {
            let pbc = PbcOptions::for_boundary(options.electronic.boundary);
            pbc_analytic_gradient(system, params, &options.electronic, &pbc)?.forces
        } else {
            analytic_gradient(system, params, gradient_options(options))?.forces
        };
        let energy = run_electronic(system, params, options.electronic.clone())?.total_free;
        return Ok(Observables {
            energy,
            forces: Some(gradient),
            stress: None,
        });
    }
    let energy = run_electronic(system, params, options.electronic.clone())?.total_free;
    Ok(Observables {
        energy,
        forces: None,
        stress: None,
    })
}

/// Central finite-difference derivatives of the energy (and optionally forces /
/// periodic stress) with respect to each requested parameter target.
pub fn parameter_finite_difference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    targets: &[ParameterTarget],
    options: &ParamDerivativeOptions,
) -> Result<Vec<ParamDerivative>> {
    let _profile = crate::profile::scope("param_deriv.total");
    let h = options.step;
    if !(h.is_finite() && h > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "parameter finite-difference step must be positive".to_string(),
        ));
    }
    let inv = 1.0 / (2.0 * h);
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let v0 = params.parameter_value(target)?;
        let plus = params.with_parameter(target, v0 + h)?;
        let minus = params.with_parameter(target, v0 - h)?;
        let op = evaluate_observables(system, &plus, options)?;
        let om = evaluate_observables(system, &minus, options)?;
        let energy_derivative = (op.energy - om.energy) * inv;
        let force_derivatives = match (op.forces, om.forces) {
            (Some(fp), Some(fm)) => Some(
                fp.iter()
                    .zip(fm.iter())
                    .map(|(a, b)| (*a - *b) * inv)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };
        let stress_derivative = match (op.stress, om.stress) {
            (Some(sp), Some(sm)) => {
                let mut d = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        d[i][j] = (sp[i][j] - sm[i][j]) * inv;
                    }
                }
                Some(d)
            }
            _ => None,
        };
        out.push(ParamDerivative {
            target: target.clone(),
            value: v0,
            energy_derivative,
            force_derivatives,
            stress_derivative,
        });
    }
    Ok(out)
}

fn gradient_options(options: &ParamDerivativeOptions) -> AnalyticGradientOptions {
    AnalyticGradientOptions {
        electronic: options.electronic.clone(),
        ..AnalyticGradientOptions::default()
    }
}

/// Central finite-difference derivative of the Mulliken dipole moment with
/// respect to each parameter target: `dmu/dp` (atomic units, `e*a0` per unit
/// parameter), returned as `(target, [dmu_x, dmu_y, dmu_z])`.
pub fn parameter_dipole_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    targets: &[ParameterTarget],
    electronic: &ElectronicOptions,
    step: f64,
) -> Result<Vec<(ParameterTarget, [f64; 3])>> {
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "parameter dipole-derivative step must be positive".to_string(),
        ));
    }
    let inv = 1.0 / (2.0 * step);
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let v0 = params.parameter_value(target)?;
        let plus = params.with_parameter(target, v0 + step)?;
        let minus = params.with_parameter(target, v0 - step)?;
        let mp = run_electronic(system, &plus, electronic.clone())?.dipole;
        let mm = run_electronic(system, &minus, electronic.clone())?.dipole;
        out.push((
            target.clone(),
            [
                (mp.x - mm.x) * inv,
                (mp.y - mm.y) * inv,
                (mp.z - mm.z) * inv,
            ],
        ));
    }
    Ok(out)
}

/// Central finite-difference derivative of the (Cartesian) Hessian with respect
/// to each parameter target: `dH/dp` (Hartree/Bohr^2 per unit parameter), returned
/// as `(target, dHessian)` with `dHessian` a `3N x 3N` matrix.
pub fn parameter_hessian_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    targets: &[ParameterTarget],
    hessian_options: &AnalyticHessianOptions,
    step: f64,
) -> Result<Vec<(ParameterTarget, Matrix)>> {
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "parameter Hessian-derivative step must be positive".to_string(),
        ));
    }
    let inv = 1.0 / (2.0 * step);
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let v0 = params.parameter_value(target)?;
        let plus = params.with_parameter(target, v0 + step)?;
        let minus = params.with_parameter(target, v0 - step)?;
        let hp = analytic_hessian(system, &plus, hessian_options.clone())?.hessian;
        let hm = analytic_hessian(system, &minus, hessian_options.clone())?.hessian;
        if hp.rows() != hm.rows() || hp.cols() != hm.cols() {
            return Err(Gfn1Error::InvalidInput(
                "parameter Hessian-derivative shape mismatch".to_string(),
            ));
        }
        let mut dh = Matrix::zeros(hp.rows(), hp.cols());
        for i in 0..hp.rows() {
            for j in 0..hp.cols() {
                dh[(i, j)] = (hp[(i, j)] - hm[(i, j)]) * inv;
            }
        }
        out.push((target.clone(), dh));
    }
    Ok(out)
}

/// Select the `chunk_index`-th slice (1-based) of `targets` split into
/// `chunk_count` contiguous, roughly equal chunks. Used to suppress / restrict
/// the output to a subset of targets (e.g. for parallel refits).
pub fn select_target_chunk(
    targets: Vec<ParameterTarget>,
    chunk_index: usize,
    chunk_count: usize,
) -> Result<Vec<ParameterTarget>> {
    if chunk_count == 0 || chunk_index == 0 || chunk_index > chunk_count {
        return Err(Gfn1Error::InvalidInput(format!(
            "invalid target chunk {chunk_index}/{chunk_count} (use 1..=count)"
        )));
    }
    let n = targets.len();
    let base = n / chunk_count;
    let rem = n % chunk_count;
    // Chunks 1..=rem get one extra element.
    let start = (chunk_index - 1) * base + (chunk_index - 1).min(rem);
    let len = base + if chunk_index <= rem { 1 } else { 0 };
    Ok(targets.into_iter().skip(start).take(len).collect())
}

/// Build a default ("active") target list for a structure: every global
/// parameter, every element scalar entry for the elements present, and every
/// pair-scaling entry whose two elements are both present.
pub fn active_targets_for_system(
    params: &Gfn1Parameters,
    system: &PeriodicSystem,
) -> Vec<ParameterTarget> {
    let mut present = system.atoms.iter().map(|a| a.z).collect::<Vec<_>>();
    present.sort_unstable();
    present.dedup();

    let mut targets = Vec::new();
    let mut globals = params.globpar.keys().cloned().collect::<Vec<_>>();
    globals.sort();
    for key in globals {
        targets.push(ParameterTarget::Global(key));
    }
    for &z in &present {
        if let Ok(elem) = params.element(z) {
            let mut keys = elem.raw.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                let len = elem.raw[&key].len();
                for index in 0..len {
                    targets.push(ParameterTarget::Element {
                        z,
                        key: key.clone(),
                        index,
                    });
                }
            }
        }
    }
    let mut pairs = params.pairpar.keys().copied().collect::<Vec<_>>();
    pairs.sort_unstable();
    for (za, zb) in pairs {
        if present.contains(&za) && present.contains(&zb) {
            targets.push(ParameterTarget::Pair(za, zb));
        }
    }
    targets
}
