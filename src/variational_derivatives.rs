// SPDX-License-Identifier: GPL-3.0-or-later
//! Complete molecular structural-head and solvent derivatives using the variational
//! canonical CP-SCC reference, including thermal occupation-matrix entropy.
use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::third_derivative::{CubicReference, SymmetricThird};
use crate::{AnalyticHessianOptions, AnalyticHessianResult, Gfn1Parameters, PeriodicSystem};

fn complete_energy(options: &AnalyticHessianOptions) -> bool {
    options.include_repulsion
        && options.include_fixed_scc
        && options.include_fixed_pulay
        && options.include_fixed_cn_h0
        && options.include_electronic
        && options.include_dispersion
        && options.include_halogen
}

pub(crate) fn supports_cartesian(options: &AnalyticHessianOptions, cutoff: f64) -> bool {
    complete_energy(options)
        && cutoff.is_finite()
        && cutoff > 0.
        && (!options.electronic_options.hamiltonian.enable_cn_hamiltonian
            || (cutoff - options.electronic_options.hamiltonian.coordination_cutoff).abs() <= 1e-12)
}

fn complete(options: &AnalyticHessianOptions) -> Result<()> {
    if !complete_energy(options) {
        return Err(Gfn1Error::InvalidInput("variational nuclear derivatives require the complete active energy; component-only assembly is not implemented".into()));
    }
    Ok(())
}
pub(crate) fn hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: Option<crate::ElectronicResult>,
    options: AnalyticHessianOptions,
) -> Result<AnalyticHessianResult> {
    complete(&options)?;
    let d =
        system.atoms.len().checked_mul(3).ok_or_else(|| {
            Gfn1Error::InvalidInput("variational Hessian dimension overflow".into())
        })?;
    if d.checked_mul(d)
        .and_then(|x| x.checked_mul(8))
        .is_none_or(|x| x > 512 * 1024 * 1024)
    {
        return Err(Gfn1Error::InvalidInput(
            "variational Hessian output exceeds the 512 MiB budget".into(),
        ));
    }
    let reference = CubicReference::build_from_result(
        system,
        params,
        &options.electronic_options,
        512 * 1024 * 1024,
        electronic,
    )?;
    let mut out = Matrix::zeros(d, d);
    let mut diagonal = vec![0.0; d];
    let mut v = vec![0.0; d];
    for i in 0..d {
        v[i] = 1.0;
        diagonal[i] = reference.directional_second(&v)?;
        out[(i, i)] = diagonal[i];
        v[i] = 0.0;
    }
    for j in 0..d {
        v[j] = 1.0;
        for i in 0..j {
            v[i] = 1.0;
            let value = 0.5 * (reference.directional_second(&v)? - diagonal[i] - diagonal[j]);
            out[(i, j)] = value;
            out[(j, i)] = value;
            v[i] = 0.0;
        }
        v[j] = 0.0;
        if let Some(logger) = &options.electronic_options.logger {
            logger.event(
                crate::LogLevel::Info,
                "ml_hessian_progress",
                serde_json::json!({"completed_rows":j+1,"rows":d}),
            )?;
        }
    }
    Ok(AnalyticHessianResult {
        hessian: out,
        repulsion: None,
        fixed_scc: None,
        fixed_pulay: None,
        fixed_cn_h0: None,
        dispersion: None,
        halogen: None,
        cpxtb_response: None,
        electronic_result: Some(reference.electronic().clone()),
    })
}
pub(crate) fn validate_cutoff(options: &AnalyticHessianOptions, cutoff: f64) -> Result<()> {
    complete(options)?;
    if !cutoff.is_finite()
        || cutoff <= 0.0
        || (options.electronic_options.hamiltonian.enable_cn_hamiltonian
            && (cutoff - options.electronic_options.hamiltonian.coordination_cutoff).abs() > 1e-12)
    {
        return Err(Gfn1Error::InvalidInput(
            "variational derivative coordination cutoff must match its reference energy".into(),
        ));
    }
    Ok(())
}
pub(crate) fn cubic_packed(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: AnalyticHessianOptions,
    cutoff: f64,
    dofs: &[usize],
    budget: usize,
) -> Result<SymmetricThird> {
    validate_cutoff(&options, cutoff)?;
    let workspace =
        CubicReference::estimate_workspace(system, params, &options.electronic_options)?;
    let plan = crate::derivative_assembly::CartesianAssembly::new(
        3,
        dofs,
        3 * system.atoms.len(),
        budget,
        workspace,
        0,
        8,
        128,
    )?;
    if dofs.is_empty() {
        return SymmetricThird::try_zeros(0, budget);
    }
    let reference = CubicReference::build(system, params, &options.electronic_options, budget)?;
    let count = std::sync::atomic::AtomicUsize::new(0);
    let event = if options.electronic_options.ml_model.is_some() {
        "ml_fc3_progress"
    } else {
        "fc3_progress"
    };
    plan.cubic(|v| {
        if let Some(logger) = &options.electronic_options.logger {
            let completed = count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if completed % 100 == 0 {
                logger.event(
                    crate::LogLevel::Info,
                    event,
                    serde_json::json!({"directional_solves":completed}),
                )?;
            }
        }
        reference.directional(v)
    })
}
pub(crate) fn cubic_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: AnalyticHessianOptions,
    cutoff: f64,
    v: &[f64],
) -> Result<Matrix> {
    validate_cutoff(&options, cutoff)?;
    let d = 3 * system.atoms.len();
    if v.len() != d || v.iter().any(|x| !x.is_finite()) {
        return Err(Gfn1Error::InvalidInput(
            "invalid variational FC3 contraction direction".into(),
        ));
    }
    if d.checked_mul(d)
        .and_then(|x| x.checked_mul(8))
        .is_none_or(|x| x > 512 * 1024 * 1024)
    {
        return Err(Gfn1Error::InvalidInput(
            "variational FC3 contraction output exceeds memory budget".into(),
        ));
    }
    let reference = CubicReference::build(
        system,
        params,
        &options.electronic_options,
        512 * 1024 * 1024,
    )?;
    crate::derivative_assembly::cubic_contraction(v, |direction| reference.directional(direction))
}
