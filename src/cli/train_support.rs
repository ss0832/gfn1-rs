// SPDX-License-Identifier: GPL-3.0-or-later
//! Versioned training-data parsing, split projection and independent metrics.
use super::invalid;
use crate::math::Vec3;
use crate::ml::MlHamiltonian;
use crate::training::{wtmad2, GroupWeight, RelativeEnergyTarget, TrainingConfig, TrainingExample};
use crate::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub struct Dataset {
    samples: Vec<TrainingExample>,
    splits: Vec<String>,
    relative: Vec<(RelativeEnergyTarget, String)>,
    pub temperature: f64,
    pub origin: Vec3,
    pub provenance: String,
    pub license: String,
    pub reference_method: String,
    pub independent: bool,
    pub sha256: String,
    pub groups: BTreeMap<String, GroupWeight>,
}
pub struct Batch {
    pub examples: Vec<TrainingExample>,
    pub relative: Vec<RelativeEnergyTarget>,
}
fn vector(value: &Value, name: &str) -> Result<Vec3> {
    let v = value
        .as_array()
        .filter(|v| v.len() == 3)
        .ok_or_else(|| invalid(format!("{name} must be a three-component vector")))?;
    let values = v
        .iter()
        .map(|x| {
            x.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid(format!("non-finite {name}")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Vec3::new(values[0], values[1], values[2]))
}
fn number(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| invalid(format!("invalid {name}")))
}
fn split(value: &Value) -> Result<String> {
    match value.as_str() {
        Some(s @ ("train" | "validation" | "extrapolation" | "reference")) => Ok(s.into()),
        _ => Err(invalid(
            "sample split must be train, validation, extrapolation or reference",
        )),
    }
}
pub fn dataset(value: &Value, sha256: String) -> Result<Dataset> {
    let text = |key: &str| {
        value[key]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| invalid(format!("dataset requires {key}")))
    };
    if ![
        "gfn1-rs-energy-training-v1",
        "gfn1-rs-observable-training-v2",
    ]
    .contains(&text("schema")?)
        || text("energy_units")? != "hartree"
        || text("energy_definition")? != "mermin_free"
    {
        return Err(invalid(
            "unsupported training schema, energy units or energy definition",
        ));
    }
    let bohr = match text("coordinate_units")? {
        "angstrom" => false,
        "bohr" => true,
        _ => return Err(invalid("coordinate units must be angstrom or bohr")),
    };
    let temperature = number(&value["electronic_temperature_K"], "electronic temperature")?;
    if temperature < 0.0 {
        return Err(invalid("negative electronic temperature"));
    }
    let origin = value
        .get("dipole_origin_bohr")
        .map(|v| vector(v, "dipole origin"))
        .transpose()?
        .unwrap_or(Vec3::zero());
    let mut out = Dataset {
        samples: vec![],
        splits: vec![],
        relative: vec![],
        temperature,
        origin,
        provenance: text("provenance")?.into(),
        license: text("license_spdx")?.into(),
        reference_method: text("reference_method")?.into(),
        independent: value["independent_reference"]
            .as_bool()
            .ok_or_else(|| invalid("declare independent_reference"))?,
        sha256,
        groups: BTreeMap::new(),
    };
    if let Some(groups) = value.get("groups") {
        for (name, g) in groups
            .as_object()
            .ok_or_else(|| invalid("groups must be an object"))?
        {
            out.groups.insert(
                name.clone(),
                GroupWeight {
                    ratio: g
                        .get("ratio")
                        .map(|v| number(v, "group ratio"))
                        .transpose()?
                        .unwrap_or(1.0),
                    mean_abs_reference_energy_kcal_mol: g
                        .get("mean_abs_reference_energy_kcal_mol")
                        .map(|v| number(v, "reference subset mean"))
                        .transpose()?,
                },
            );
        }
    }
    let mut ids = BTreeMap::new();
    for (i, sample) in value["samples"]
        .as_array()
        .ok_or_else(|| invalid("dataset requires samples"))?
        .iter()
        .enumerate()
    {
        let id = sample["id"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("sample-{i}"));
        if ids.insert(id, i).is_some() {
            return Err(invalid("duplicate sample id"));
        }
        let xyz = sample["xyz"]
            .as_str()
            .ok_or_else(|| invalid("each sample requires inline xyz"))?;
        let charge = sample
            .get("charge")
            .map(|x| number(x, "charge"))
            .transpose()?
            .unwrap_or(0.0);
        let mut e = TrainingExample::new(PeriodicSystem::from_xyz_str(xyz, charge, bohr)?);
        e.group = sample["group"].as_str().unwrap_or("default").into();
        e.energy_hartree = sample
            .get("energy_hartree")
            .filter(|v| !v.is_null())
            .map(|v| number(v, "reference energy"))
            .transpose()?;
        let gradient = sample
            .get("gradient_hartree_per_bohr")
            .filter(|v| !v.is_null());
        let forces = sample
            .get("forces_hartree_per_bohr")
            .filter(|v| !v.is_null());
        if gradient.is_some() && forces.is_some() {
            return Err(invalid("provide gradient or forces, not both"));
        }
        if let Some(values) = gradient.or(forces) {
            let mut g = values
                .as_array()
                .ok_or_else(|| invalid("nuclear labels must be an array"))?
                .iter()
                .map(|v| vector(v, "nuclear label"))
                .collect::<Result<Vec<_>>>()?;
            if g.len() != e.system.atoms.len() {
                return Err(invalid("nuclear label atom count mismatch"));
            }
            if forces.is_some() {
                for v in &mut g {
                    *v = *v * (-1.0);
                }
            }
            e.gradient_hartree_per_bohr = Some(g);
        }
        e.dipole_e_bohr = sample
            .get("dipole_e_bohr")
            .filter(|v| !v.is_null())
            .map(|v| vector(v, "dipole label"))
            .transpose()?;
        out.samples.push(e);
        out.splits.push(split(&sample["split"])?);
    }
    if let Some(relative) = value.get("relative_energies") {
        for case in relative
            .as_array()
            .ok_or_else(|| invalid("relative_energies must be an array"))?
        {
            let mut terms = Vec::new();
            for term in case["terms"]
                .as_array()
                .ok_or_else(|| invalid("relative case requires terms"))?
            {
                let id = term["sample"]
                    .as_str()
                    .ok_or_else(|| invalid("relative term requires sample id"))?;
                let index = *ids
                    .get(id)
                    .ok_or_else(|| invalid("unknown relative-energy sample id"))?;
                terms.push((index, number(&term["coefficient"], "reaction coefficient")?));
            }
            let selected = split(&case["split"])?;
            if selected == "reference" {
                return Err(invalid("relative cases must belong to a measured split"));
            }
            if selected == "train"
                && terms
                    .iter()
                    .any(|&(i, _)| !matches!(out.splits[i].as_str(), "train" | "reference"))
            {
                return Err(invalid("training reactions cannot use held-out structures; mark shared anchors as reference"));
            }
            out.relative.push((
                RelativeEnergyTarget {
                    group: case["group"].as_str().unwrap_or("default").into(),
                    terms,
                    reference_hartree: number(
                        &case["energy_hartree"],
                        "relative reference energy",
                    )?,
                },
                selected,
            ));
        }
    }
    if !out.splits.iter().any(|s| s == "train") && !out.relative.iter().any(|(_, s)| s == "train") {
        return Err(invalid("training split is empty"));
    }
    Ok(out)
}
impl Dataset {
    pub fn batch(&self, selected: &str) -> Batch {
        self.project(selected, None)
    }
    pub fn group_batches(&self, selected: &str) -> BTreeMap<String, Batch> {
        let groups: BTreeSet<_> = self
            .samples
            .iter()
            .zip(&self.splits)
            .filter(|(_, s)| s.as_str() == selected)
            .map(|(e, _)| e.group.clone())
            .collect();
        groups
            .into_iter()
            .map(|g| {
                let batch = self.project(selected, Some(&g));
                (g, batch)
            })
            .collect()
    }
    fn project(&self, selected: &str, group: Option<&str>) -> Batch {
        let primary = |i: usize| {
            self.splits[i] == selected && group.is_none_or(|g| self.samples[i].group == g)
        };
        let measured = |r: &RelativeEnergyTarget, s: &str| {
            s == selected
                && group.is_none_or(|_| r.terms.iter().any(|&(i, c)| c != 0.0 && primary(i)))
        };
        let mut needed: BTreeSet<usize> = (0..self.samples.len()).filter(|&i| primary(i)).collect();
        for (r, s) in &self.relative {
            if measured(r, s) {
                needed.extend(r.terms.iter().map(|&(i, _)| i));
            }
        }
        let mut map = BTreeMap::new();
        let mut examples = Vec::new();
        for i in needed {
            map.insert(i, examples.len());
            let mut e = self.samples[i].clone();
            if !primary(i) {
                e.energy_hartree = None;
                e.gradient_hartree_per_bohr = None;
                e.dipole_e_bohr = None;
            }
            examples.push(e);
        }
        let relative = self
            .relative
            .iter()
            .filter(|(r, s)| measured(r, s))
            .map(|(r, _)| RelativeEnergyTarget {
                group: r.group.clone(),
                terms: r.terms.iter().map(|&(i, c)| (map[&i], c)).collect(),
                reference_hartree: r.reference_hartree,
            })
            .collect();
        Batch { examples, relative }
    }
}
#[derive(Default)]
pub struct Evaluation {
    pub absolute: Vec<f64>,
    pub relative: Vec<f64>,
    pub energies: Vec<f64>,
    gradient_sum: f64,
    gradient_count: usize,
    dipole_sum: f64,
    dipole_count: usize,
}
pub fn evaluate(
    batch: &Batch,
    params: &Gfn1Parameters,
    settings: &ElectronicOptions,
    model: Option<&MlHamiltonian>,
) -> Result<Evaluation> {
    let mut out = Evaluation::default();
    let mut options = settings.clone();
    options.ml_model = model.map(|m| Arc::new(m.clone()));
    for sample in &batch.examples {
        let (e, g) = if sample.gradient_hartree_per_bohr.is_some() {
            let result = crate::analytic_gradient(
                &sample.system,
                params,
                crate::AnalyticGradientOptions {
                    electronic: options.clone(),
                    ..Default::default()
                },
            )?;
            (result.electronic_result, Some(result.gradient))
        } else {
            (
                crate::run_electronic(&sample.system, params, options.clone())?,
                None,
            )
        };
        out.energies.push(e.total_free);
        if let Some(reference) = sample.energy_hartree {
            out.absolute.push(e.total_free - reference);
        }
        if let (Some(reference), Some(g)) = (&sample.gradient_hartree_per_bohr, g) {
            for (a, b) in g.iter().zip(reference) {
                out.gradient_sum += (*a - *b).norm2();
                out.gradient_count += 3;
            }
        }
        if let Some(reference) = sample.dipole_e_bohr {
            out.dipole_sum += (e.dipole - reference).norm2();
            out.dipole_count += 3;
        }
    }
    out.relative = batch
        .relative
        .iter()
        .map(|r| {
            r.terms
                .iter()
                .map(|&(i, c)| c * out.energies[i])
                .sum::<f64>()
                - r.reference_hartree
        })
        .collect();
    Ok(out)
}
pub fn statistics(
    base: &Evaluation,
    model: &Evaluation,
    batch: &Batch,
    config: &TrainingConfig,
    offsets: Option<[f64; 2]>,
) -> Value {
    if batch.examples.is_empty() {
        return Value::Null;
    }
    let rmse = |values: &[f64], offset: f64| {
        if values.is_empty() {
            None
        } else {
            Some(
                (values.iter().map(|x| (x - offset).powi(2)).sum::<f64>() / values.len() as f64)
                    .sqrt(),
            )
        }
    };
    let component = |sum: f64, count: usize| {
        if count == 0 {
            None
        } else {
            Some((sum / count as f64).sqrt())
        }
    };
    let metric = |energies: &[f64]| {
        if batch.relative.is_empty() {
            json!({"value":null,"status":"no_relative_cases"})
        } else {
            match wtmad2(energies, &batch.relative, config) {
                Ok(value) => json!({"value":value,"status":"measured"}),
                Err(error) => json!({"value":null,"status":"undefined","reason":error.to_string()}),
            }
        }
    };
    json!({"structures_evaluated":batch.examples.len(),"absolute_energy_cases":base.absolute.len(),"relative_energy_cases":base.relative.len(),
        "baseline_rmse_hartree":rmse(&base.absolute,0.0),"model_rmse_hartree":rmse(&model.absolute,0.0),
        "baseline_shape_rmse_hartree":offsets.and_then(|o|rmse(&base.absolute,o[0])),"model_shape_rmse_hartree":offsets.and_then(|o|rmse(&model.absolute,o[1])),
        "baseline_relative_rmse_hartree":rmse(&base.relative,0.0),"model_relative_rmse_hartree":rmse(&model.relative,0.0),
        "baseline_gradient_rmse_hartree_per_bohr":component(base.gradient_sum,base.gradient_count),"model_gradient_rmse_hartree_per_bohr":component(model.gradient_sum,model.gradient_count),
        "gradient_components":base.gradient_count,"baseline_dipole_rmse_e_bohr":component(base.dipole_sum,base.dipole_count),"model_dipole_rmse_e_bohr":component(model.dipole_sum,model.dipole_count),"dipole_components":base.dipole_count,
        "baseline_wtmad2_kcal_mol":metric(&base.energies),"model_wtmad2_kcal_mol":metric(&model.energies),
        "wtmad2_scale_source":"provided reference subset means, otherwise reference means of this evaluated split; no held-out statistics enter fitting"})
}
