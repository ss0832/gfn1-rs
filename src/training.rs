// SPDX-License-Identifier: GPL-3.0-or-later
//! Energy/gradient/dipole fitting with stationary SCF derivatives. Gradient
//! and dipole losses use implicit CP-SCC response, without SCF unrolling.
//! WTMAD-2 subset scaling applies only to explicitly relative/reaction energies.
use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::ml::{
    level_sensitivities, level_sensitivity_derivative, update_parameter, ElementHeads,
    EnergyOptimizer, EnergyTrainingOptions, MlHamiltonian, SelfGradient,
};
use crate::third_derivative::CubicReference;
use crate::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};

pub const WTMAD2_REFERENCE_KCAL_MOL: f64 = 56.84;
pub const HARTREE_TO_KCAL_MOL: f64 = 627.50947428;
#[derive(Clone, Debug)]
pub struct TrainingExample {
    pub system: PeriodicSystem,
    pub group: String,
    pub energy_hartree: Option<f64>,
    pub gradient_hartree_per_bohr: Option<Vec<Vec3>>,
    pub dipole_e_bohr: Option<Vec3>,
}
impl TrainingExample {
    pub fn new(system: PeriodicSystem) -> Self {
        Self {
            system,
            group: "default".into(),
            energy_hartree: None,
            gradient_hartree_per_bohr: None,
            dipole_e_bohr: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct RelativeEnergyTarget {
    pub group: String,
    /// Linear combination of predicted total energies, indexed into examples.
    pub terms: Vec<(usize, f64)>,
    pub reference_hartree: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct GroupWeight {
    pub ratio: f64,
    pub mean_abs_reference_energy_kcal_mol: Option<f64>,
}
impl Default for GroupWeight {
    fn default() -> Self {
        Self {
            ratio: 1.0,
            mean_abs_reference_energy_kcal_mol: None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyWeighting {
    Uniform,
    Wtmad2,
}
#[derive(Clone, Copy, Debug)]
pub enum EnergyLoss {
    Squared,
    Absolute,
    Huber { delta_hartree: f64 },
}
impl EnergyLoss {
    fn value_derivative(self, r: f64) -> (f64, f64) {
        match self {
            Self::Squared => (0.5 * r * r, r),
            Self::Absolute => (r.abs(), if r == 0.0 { 0.0 } else { r.signum() }),
            Self::Huber { delta_hartree: d } => {
                if r.abs() <= d {
                    (0.5 * r * r, r)
                } else {
                    (d * (r.abs() - 0.5 * d), d * r.signum())
                }
            }
        }
    }
}
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub optimizer: EnergyTrainingOptions,
    pub energy_weight: f64,
    pub gradient_weight: f64,
    pub dipole_weight: f64,
    pub relative_energy_weight: f64,
    pub energy_loss: EnergyLoss,
    pub energy_weighting: EnergyWeighting,
    pub groups: BTreeMap<String, GroupWeight>,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            optimizer: Default::default(),
            energy_weight: 1.0,
            gradient_weight: 1.0,
            dipole_weight: 1.0,
            relative_energy_weight: 1.0,
            energy_loss: EnergyLoss::Squared,
            energy_weighting: EnergyWeighting::Uniform,
            groups: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LossComponents {
    pub total: f64,
    pub energy: f64,
    pub gradient: f64,
    pub dipole: f64,
    pub relative_energy: f64,
}
pub struct TrainingLossGradient {
    pub loss: LossComponents,
    pub gradient: BTreeMap<u8, ElementHeads>,
    pub weighting_metadata: Value,
}
pub struct TrainingHistory {
    pub epochs: Vec<LossComponents>,
    pub weighting_metadata: Value,
}

fn invalid(s: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(s.into())
}
fn ratio(config: &TrainingConfig, group: &str) -> f64 {
    config.groups.get(group).map_or(1.0, |g| g.ratio)
}
fn validate(
    examples: &[TrainingExample],
    relative: &[RelativeEnergyTarget],
    config: &TrainingConfig,
) -> Result<()> {
    if config.energy_weighting == EnergyWeighting::Wtmad2
        && config.relative_energy_weight > 0.0
        && relative.is_empty()
    {
        return Err(invalid(
            "WTMAD-2 training requires explicit relative/reaction energy cases",
        ));
    }
    if examples.is_empty()
        || [
            config.energy_weight,
            config.gradient_weight,
            config.dipole_weight,
            config.relative_energy_weight,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(invalid(
            "invalid observable training weights or empty examples",
        ));
    }
    for group in config.groups.values() {
        if !group.ratio.is_finite()
            || group.ratio < 0.0
            || group
                .mean_abs_reference_energy_kcal_mol
                .is_some_and(|x| !x.is_finite() || x <= 0.0)
        {
            return Err(invalid("invalid group ratio or reference energy scale"));
        }
    }
    if let EnergyLoss::Huber { delta_hartree: d } = config.energy_loss {
        if !d.is_finite() || d <= 0.0 {
            return Err(invalid("Huber width must be positive"));
        }
    }
    for e in examples {
        if e.system.atoms.is_empty()
            || e.group.trim().is_empty()
            || e.energy_hartree.is_some_and(|x| !x.is_finite())
            || e.dipole_e_bohr
                .is_some_and(|v| v.to_array().iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("invalid observable training example"));
        }
        if let Some(g) = &e.gradient_hartree_per_bohr {
            if g.len() != e.system.atoms.len()
                || g.iter()
                    .any(|v| v.to_array().iter().any(|x| !x.is_finite()))
            {
                return Err(invalid("invalid nuclear-gradient labels"));
            }
        }
        if config.energy_weighting == EnergyWeighting::Wtmad2
            && config.energy_weight > 0.0
            && e.energy_hartree.is_some()
        {
            return Err(invalid("WTMAD-2 is defined for relative/reaction energies; disable absolute energy loss and provide relative targets"));
        }
    }
    for r in relative {
        if r.group.trim().is_empty()
            || !r.reference_hartree.is_finite()
            || r.terms.is_empty()
            || r.terms
                .iter()
                .any(|&(i, c)| i >= examples.len() || !c.is_finite())
            || r.terms.iter().all(|&(_, c)| c == 0.0)
        {
            return Err(invalid("invalid relative-energy target"));
        }
    }
    Ok(())
}
fn relative_scales(
    relative: &[RelativeEnergyTarget],
    config: &TrainingConfig,
) -> Result<BTreeMap<String, f64>> {
    let mut statistics: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for r in relative {
        let s = statistics.entry(r.group.clone()).or_default();
        s.0 += r.reference_hartree.abs() * HARTREE_TO_KCAL_MOL;
        s.1 += 1;
    }
    statistics
        .into_iter()
        .map(|(group, (sum, n))| {
            if ratio(config, &group) == 0.0 {
                return Ok((group, 0.0));
            }
            let mean = config
                .groups
                .get(&group)
                .and_then(|g| g.mean_abs_reference_energy_kcal_mol)
                .unwrap_or(sum / n as f64);
            if !mean.is_finite() || mean <= 0.0 {
                return Err(invalid(format!(
                    "WTMAD-2 reference mean is zero/invalid for subset {group}"
                )));
            }
            Ok((group, WTMAD2_REFERENCE_KCAL_MOL / mean))
        })
        .collect()
}
/// Exact WTMAD-2 MAE in kcal/mol, with optional explicitly reported subset
/// mixture ratios. With all ratios 1, N_i counts enter the standard definition.
pub fn wtmad2(
    predicted_total_energies: &[f64],
    relative: &[RelativeEnergyTarget],
    config: &TrainingConfig,
) -> Result<f64> {
    if relative.is_empty() {
        return Err(invalid("WTMAD-2 requires relative/reaction energy cases"));
    }
    if config
        .groups
        .values()
        .any(|g| !g.ratio.is_finite() || g.ratio < 0.0)
        || relative.iter().any(|r| !r.reference_hartree.is_finite())
    {
        return Err(invalid("invalid WTMAD-2 subset ratio or reference"));
    }
    let scales = relative_scales(relative, config)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for r in relative {
        let mut predicted = 0.0;
        for &(i, c) in &r.terms {
            let e = *predicted_total_energies
                .get(i)
                .ok_or_else(|| invalid("WTMAD-2 energy index out of range"))?;
            if !e.is_finite() || !c.is_finite() {
                return Err(invalid("non-finite WTMAD-2 input"));
            }
            predicted += c * e;
        }
        let weight = ratio(config, &r.group);
        numerator += weight
            * scales[&r.group]
            * (predicted - r.reference_hartree).abs()
            * HARTREE_TO_KCAL_MOL;
        denominator += weight;
    }
    if denominator <= 0.0 {
        return Err(invalid("WTMAD-2 has no positively weighted cases"));
    }
    Ok(numerator / denominator)
}

struct Forward {
    energy: f64,
    charges: Vec<f64>,
    energy_adjoint: f64,
    gradient_direction: Vec<f64>,
    dipole_direction: Vec3,
}
pub fn loss_gradient(
    model: &MlHamiltonian,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    examples: &[TrainingExample],
    relative: &[RelativeEnergyTarget],
    config: &TrainingConfig,
) -> Result<TrainingLossGradient> {
    validate(examples, relative, config)?;
    model.validate()?;
    let ze: f64 = examples
        .iter()
        .filter(|e| e.energy_hartree.is_some())
        .map(|e| ratio(config, &e.group))
        .sum();
    let zg: f64 = examples
        .iter()
        .filter(|e| e.gradient_hartree_per_bohr.is_some())
        .map(|e| ratio(config, &e.group))
        .sum();
    let zm: f64 = examples
        .iter()
        .filter(|e| e.dipole_e_bohr.is_some())
        .map(|e| ratio(config, &e.group))
        .sum();
    let zr: f64 = relative.iter().map(|r| ratio(config, &r.group)).sum();
    if config.energy_weight * ze
        + config.gradient_weight * zg
        + config.dipole_weight * zm
        + config.relative_energy_weight * zr
        <= 0.0
    {
        return Err(invalid("no active positively weighted training targets"));
    }
    let scales = if config.energy_weighting == EnergyWeighting::Wtmad2 {
        relative_scales(relative, config)?
    } else {
        BTreeMap::new()
    };
    let mut options = electronic.clone();
    options.ml_model = Some(Arc::new(model.clone()));
    let mut loss = LossComponents::default();
    let mut forward = Vec::with_capacity(examples.len());
    // Keep only scalar/residual/charge checkpoints, not every sample's AO state
    // or SCF iteration history. The backward pass recomputes a converged state.
    for example in examples {
        let n = example.system.atoms.len();
        let sample_ratio = ratio(config, &example.group);
        let need_gradient = example.gradient_hartree_per_bohr.is_some()
            && config.gradient_weight > 0.0
            && sample_ratio > 0.0;
        let (e, gradient) = if need_gradient {
            let g = crate::analytic_gradient(
                &example.system,
                params,
                crate::AnalyticGradientOptions {
                    electronic: options.clone(),
                    ..Default::default()
                },
            )?;
            (g.electronic_result, Some(g.gradient))
        } else {
            (
                crate::run_electronic(&example.system, params, options.clone())?,
                None,
            )
        };
        if (e.electronic_temperature - options.electronic_temperature.max(0.0)).abs() > 1e-6 {
            return Err(invalid("training SCF changed electronic temperature; labels and prediction must use the same free-energy definition"));
        }
        let mut checkpoint = Forward {
            energy: e.total_free,
            charges: e.shell_charges,
            energy_adjoint: 0.0,
            gradient_direction: vec![0.0; 3 * n],
            dipole_direction: Vec3::zero(),
        };
        if let Some(target) = example.energy_hartree {
            if ze > 0.0 && config.energy_weight > 0.0 && sample_ratio > 0.0 {
                let (value, derivative) =
                    config.energy_loss.value_derivative(e.total_free - target);
                let weight = config.energy_weight * sample_ratio / ze;
                loss.energy += weight * value;
                checkpoint.energy_adjoint = weight * derivative;
            }
        }
        if let (Some(target), Some(prediction)) = (&example.gradient_hartree_per_bohr, gradient) {
            let weight = config.gradient_weight * sample_ratio / (zg * (3 * n) as f64);
            for a in 0..n {
                let residual = (prediction[a] - target[a]).to_array();
                for k in 0..3 {
                    loss.gradient += 0.5 * weight * residual[k] * residual[k];
                    checkpoint.gradient_direction[3 * a + k] = weight * residual[k];
                }
            }
        }
        if let Some(target) = example.dipole_e_bohr {
            if zm > 0.0 && config.dipole_weight > 0.0 && sample_ratio > 0.0 {
                let residual = e.dipole - target;
                let weight = config.dipole_weight * sample_ratio / (3.0 * zm);
                loss.dipole += 0.5 * weight * residual.norm2();
                checkpoint.dipole_direction = residual * weight;
            }
        }
        forward.push(checkpoint);
    }
    if zr > 0.0 && config.relative_energy_weight > 0.0 {
        for r in relative {
            if ratio(config, &r.group) == 0.0 {
                continue;
            }
            let predicted: f64 = r.terms.iter().map(|&(i, c)| c * forward[i].energy).sum();
            let (value, derivative) = config
                .energy_loss
                .value_derivative(predicted - r.reference_hartree);
            let scale = if config.energy_weighting == EnergyWeighting::Wtmad2 {
                scales[&r.group]
            } else {
                1.0
            };
            let weight = config.relative_energy_weight * ratio(config, &r.group) * scale / zr;
            loss.relative_energy += weight * value;
            for &(i, c) in &r.terms {
                forward[i].energy_adjoint += weight * derivative * c;
            }
        }
    }
    loss.total = loss.energy + loss.gradient + loss.dipole + loss.relative_energy;
    if !loss.total.is_finite() {
        return Err(invalid("non-finite observable loss"));
    }
    let mut accumulated = SelfGradient::zero(model);
    for (i, example) in examples.iter().enumerate() {
        let checkpoint = &forward[i];
        let nuclear = checkpoint.gradient_direction.iter().any(|&x| x != 0.0);
        let electric = checkpoint.dipole_direction.norm2() != 0.0;
        if !nuclear && !electric && checkpoint.energy_adjoint == 0.0 {
            continue;
        }
        let response = if nuclear || electric {
            Some(CubicReference::build(
                &example.system,
                params,
                &options,
                512 * 1024 * 1024,
            )?)
        } else {
            None
        };
        let owned = if response.is_none() {
            Some(crate::run_electronic(
                &example.system,
                params,
                options.clone(),
            )?)
        } else {
            None
        };
        let e = response
            .as_ref()
            .map(CubicReference::electronic)
            .or(owned.as_ref())
            .unwrap();
        let charge_error = e
            .shell_charges
            .iter()
            .zip(&checkpoint.charges)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if (e.total_free - checkpoint.energy).abs() > 1e-8 || charge_error > 1e-7 {
            return Err(invalid(
                "training checkpoint recomputation reached a different SCF branch",
            ));
        }
        let view = e
            .ml_parameters
            .as_ref()
            .ok_or_else(|| invalid("missing structural ML parameter view"))?;
        let sensitivity = level_sensitivities(&example.system, params, e)?;
        if checkpoint.energy_adjoint != 0.0 {
            accumulated.accumulate(
                &view.energy_head_gradient(&sensitivity)?,
                checkpoint.energy_adjoint,
            );
        }
        if nuclear {
            let (bundle, s1) = response
                .as_ref()
                .unwrap()
                .nuclear_response(&checkpoint.gradient_direction)?;
            let derivative = level_sensitivity_derivative(
                &example.system,
                params,
                e,
                &bundle.density,
                &s1,
                &checkpoint.gradient_direction,
            )?;
            accumulated.accumulate(
                &view.directional_head_gradient(
                    &sensitivity,
                    &derivative,
                    &checkpoint.gradient_direction,
                )?,
                1.0,
            );
        }
        if electric {
            let bundle = response
                .as_ref()
                .unwrap()
                .dipole_response(checkpoint.dipole_direction)?;
            let n = e.basis.len();
            let derivative = level_sensitivity_derivative(
                &example.system,
                params,
                e,
                &bundle.density,
                &Matrix::zeros(n, n),
                &vec![0.0; 3 * example.system.atoms.len()],
            )?;
            // E_field=-mu.F. The QM-only correction energy has no field
            // dependence and must not receive a spurious dipole gradient.
            accumulated.accumulate(&view.level_head_gradient(&derivative)?, -1.0);
        }
    }
    let group_metadata:BTreeMap<String,Value>=examples.iter().map(|e|e.group.clone()).chain(relative.iter().map(|r|r.group.clone())).map(|group|{
        let configured=config.groups.get(&group).copied().unwrap_or_default();let cases=relative.iter().filter(|r|r.group==group).count();
        let data=json!({"ratio":configured.ratio,"relative_cases":cases,"relative_energy_scale":scales.get(&group),
            "scale_source":if config.energy_weighting==EnergyWeighting::Uniform{"uniform"}else if configured.mean_abs_reference_energy_kcal_mol.is_some(){"provided_reference_subset_mean"}else{"training_relative_reference_mean"}});(group,data)
    }).collect();
    let metadata = json!({"energy_weight":config.energy_weight,"gradient_weight":config.gradient_weight,"dipole_weight":config.dipole_weight,"relative_energy_weight":config.relative_energy_weight,
        "energy_weighting":format!("{:?}",config.energy_weighting),"energy_loss":format!("{:?}",config.energy_loss),"wtmad2_reference_kcal_mol":WTMAD2_REFERENCE_KCAL_MOL,
        "groups":group_metadata,"normalizers":{"absolute_energy":ze,"gradient":zg,"dipole":zm,"relative_energy":zr},
        "component_averaging":"gradients averaged over 3N; dipoles over 3; each target family normalized by active group-ratio sum",
        "gradient_units":"hartree/bohr","dipole_units":"e*bohr","dipole_definition":"Mulliken monopoles at the configured field origin",
        "derivative_method":"stationary energy gradient and implicit CP-SCC mixed response; no SCF unrolling"});
    Ok(TrainingLossGradient {
        loss,
        gradient: accumulated.0,
        weighting_metadata: metadata,
    })
}

pub fn fit_observables(
    model: &mut MlHamiltonian,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    examples: &[TrainingExample],
    relative: &[RelativeEnergyTarget],
    config: &TrainingConfig,
) -> Result<TrainingHistory> {
    let o = config.optimizer;
    if o.epochs == 0
        || !o.learning_rate.is_finite()
        || o.learning_rate <= 0.0
        || !o.weight_decay.is_finite()
        || o.weight_decay < 0.0
    {
        return Err(invalid("invalid observable optimizer settings"));
    }
    if let EnergyOptimizer::AdamW {
        beta1,
        beta2,
        epsilon,
    } = o.optimizer
    {
        if !(beta1.is_finite()
            && beta2.is_finite()
            && (0.0..1.0).contains(&beta1)
            && (0.0..1.0).contains(&beta2)
            && epsilon.is_finite()
            && epsilon > 0.0)
        {
            return Err(invalid("invalid AdamW parameters"));
        }
    }
    let span=electronic.logger.as_ref().map(|logger|logger.begin("ml_observable_training",json!({"samples":examples.len(),"relative_cases":relative.len(),"epochs":o.epochs,"optimizer":format!("{:?}",o.optimizer)}))).transpose()?;
    let result = (|| {
        let mut m = SelfGradient::zero(model);
        let mut v = SelfGradient::zero(model);
        let mut history = Vec::new();
        history
            .try_reserve_exact(o.epochs)
            .map_err(|e| invalid(format!("training history allocation failed: {e}")))?;
        let mut metadata = Value::Null;
        for epoch in 1..=o.epochs {
            let result = loss_gradient(model, params, electronic, examples, relative, config)?;
            metadata = result.weighting_metadata;
            if let Some(logger) = &electronic.logger {
                logger.event(crate::LogLevel::Info,"observable_training_epoch",json!({"epoch":epoch,"epochs":o.epochs,"loss":result.loss.total,"energy_loss":result.loss.energy,"gradient_loss":result.loss.gradient,"dipole_loss":result.loss.dipole,"relative_energy_loss":result.loss.relative_energy}))?;
            }
            history.push(result.loss);
            for (&z, heads) in &mut model.heads {
                let gradient = &result.gradient[&z];
                let first = m.0.get_mut(&z).unwrap();
                let second = v.0.get_mut(&z).unwrap();
                for l in 0..if o.train_correction { 4 } else { 3 } {
                    let (head, g, m, v) = if l < 3 {
                        (
                            &mut heads.levels[l],
                            &gradient.levels[l],
                            &mut first.levels[l],
                            &mut second.levels[l],
                        )
                    } else {
                        (
                            &mut heads.correction,
                            &gradient.correction,
                            &mut first.correction,
                            &mut second.correction,
                        )
                    };
                    update_parameter(
                        &mut head.bias,
                        g.bias,
                        &mut m.bias,
                        &mut v.bias,
                        epoch,
                        &o,
                        o.decay_bias,
                    );
                    for k in 0..head.weights.len() {
                        update_parameter(
                            &mut head.weights[k],
                            g.weights[k],
                            &mut m.weights[k],
                            &mut v.weights[k],
                            epoch,
                            &o,
                            true,
                        );
                    }
                }
            }
            model.validate()?;
        }
        Ok(TrainingHistory {
            epochs: history,
            weighting_metadata: metadata,
        })
    })();
    crate::logging::finish(
        span,
        result,
        |h| json!({"epochs":h.epochs.len(),"last_evaluated_loss":h.epochs.last().map(|x|x.total),"weighting":h.weighting_metadata}),
    )
}
