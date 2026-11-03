// SPDX-License-Identifier: GPL-3.0-or-later
//! Independent smooth radial Hamiltonian heads inspired by Li2026MLxTB.
//! Only QM geometry/species enter the heads. The basis exponents and global
//! GFN1 tables remain fixed. This small native architecture is NOT MACE-OFF24
//! and does not claim the paper's trained model/accuracy. No restricted weights
//! are distributed. Structure-local level shifts enter H0 before SCC, rather
//! than correcting only its final energy. Analytic forces include dtheta/dR.
use crate::error::{Gfn1Error, Result};
use crate::jets::Jet1;
use crate::math::Vec3;
use crate::{BasisSet, ElectronicResult, Gfn1Parameters, PeriodicSystem};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct LinearHead {
    pub bias: f64,
    pub weights: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct ElementHeads {
    pub levels: [LinearHead; 3],
    pub correction: LinearHead,
}
#[derive(Clone, Debug)]
pub struct MlHamiltonian {
    pub species: Vec<u8>,
    pub radial_centers_bohr: Vec<f64>,
    pub radial_width_bohr: f64,
    pub cutoff_bohr: f64,
    pub maximum_level_shift_hartree: f64,
    pub heads: BTreeMap<u8, ElementHeads>,
    pub training_provenance: String,
    pub license_spdx: String,
}

/// Load-time identity: original file bytes and canonical parsed weights are
/// distinguished so formatting-only edits do not look like different physics.
#[derive(Clone, Debug)]
pub struct MlModelIdentity {
    pub source: String,
    pub source_sha256: String,
    pub canonical_sha256: String,
    pub training_provenance: String,
    pub license_spdx: String,
}
impl MlModelIdentity {
    pub fn description(&self) -> String {
        format!(
            "{}; source_sha256={}; canonical_sha256={}; provenance={}; license={}",
            self.source,
            self.source_sha256,
            self.canonical_sha256,
            self.training_provenance,
            self.license_spdx
        )
    }
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({"source":self.source,"source_sha256":self.source_sha256,"canonical_sha256":self.canonical_sha256,"training_provenance":self.training_provenance,"license_spdx":self.license_spdx,"architecture":"gfn1-rs-radial-hamiltonian-v1"})
    }
}

#[derive(Clone, Debug)]
pub struct AtomicParameterView {
    pub delta_levels_hartree: Vec<[f64; 3]>,
    pub correction_energy_hartree: f64,
    pub model_sha256: String,
    features: Vec<Vec<f64>>,
    level_slopes: Vec<[f64; 3]>,
    model: Arc<MlHamiltonian>,
    system: PeriodicSystem,
}

pub(crate) struct DirectionalParameters {
    pub levels: Vec<[Jet1; 3]>,
    pub correction: Jet1,
}


const BUILTIN_MODELS: &[(&str, &str)] = &[
    ("builtin:h2-demo", include_str!("../models/h2-radial-demo.model")),
    ("builtin:h2-fci", include_str!("../models/h2-fci-trained.model")),
    ("builtin:water-clusters", include_str!("../models/water-clusters-trained.model")),
];

impl MlHamiltonian {
    /// Canonical identifiers of the models embedded in the native build.
    pub fn builtin_model_names() -> impl Iterator<Item = &'static str> {
        BUILTIN_MODELS.iter().map(|&(name, _)| name)
    }

    pub fn resolve(spec: &str) -> Result<Self> {
        Ok(Self::resolve_with_identity(spec)?.0)
    }
    pub fn canonical_sha256(&self) -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(self.to_model_string()?.as_bytes())
        ))
    }
    pub fn resolve_with_identity(spec: &str) -> Result<(Self, MlModelIdentity)> {

        let spec = match spec {
            "builtin:water-cluster" => "builtin:water-clusters",
            spec => spec,
        };
        let (text, source) = if let Some(&(name, text)) = BUILTIN_MODELS.iter().find(|&&(name, _)| name == spec) {
            (text.to_string(), name.to_string())
        } else if spec.starts_with("builtin:") {
            return Err(Gfn1Error::InvalidInput(format!(
                "unknown builtin ML model {spec}; available: {}",
                Self::builtin_model_names().collect::<Vec<_>>().join(", ")
            )));
        } else {
            if std::fs::metadata(spec)?.len() > 16 * 1024 * 1024 {
                return Err(Gfn1Error::InvalidInput("ML model file exceeds 16 MiB".into()));
            }
            (std::fs::read_to_string(spec)?, std::fs::canonicalize(spec)?.display().to_string())
        };
        let model = Self::from_model_str(&text)?;
        let identity = MlModelIdentity {
            source,
            source_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            canonical_sha256: model.canonical_sha256()?,
            training_provenance: model.training_provenance.clone(),
            license_spdx: model.license_spdx.clone(),
        };
        Ok((model, identity))
    }
    pub fn builtin_h2_fci() -> Result<Self> {
        Self::from_model_str(include_str!("../models/h2-fci-trained.model"))
    }
    /// Small independent PBE0/def2-SVP water-cluster demonstration model.
    pub fn builtin_water_clusters() -> Result<Self> {
        Self::from_model_str(include_str!("../models/water-clusters-trained.model"))
    }
    pub fn builtin_h2_demo() -> Result<Self> {
        Self::from_model_str(include_str!("../models/h2-radial-demo.model"))
    }
    /// Portable, unit-tagged native model format. No external inference runtime.
    pub fn to_model_string(&self) -> Result<String> {
        self.validate()?;
        if self.training_provenance.contains(['\n', '\r'])
            || self.license_spdx.contains(char::is_whitespace)
        {
            return Err(Gfn1Error::InvalidInput(
                "ML provenance/license must fit one model line".into(),
            ));
        }
        let mut out=format!("gfn1-rs-radial-hamiltonian-v1\nunits hartree bohr\nspecies {}\ncenters {}\nwidth {}\ncutoff {}\nlimit {}\nlicense {}\nprovenance {}\n",
            self.species.iter().map(u8::to_string).collect::<Vec<_>>().join(" "),
            self.radial_centers_bohr.iter().map(f64::to_string).collect::<Vec<_>>().join(" "),
            self.radial_width_bohr,self.cutoff_bohr,self.maximum_level_shift_hartree,self.license_spdx,self.training_provenance);
        for (&z, h) in &self.heads {
            for (l, head) in h
                .levels
                .iter()
                .chain(std::iter::once(&h.correction))
                .enumerate()
            {
                out.push_str(&format!(
                    "head {z} {l} {} {}\n",
                    head.bias,
                    head.weights
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            }
        }
        Ok(out)
    }
    pub fn from_model_str(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some("gfn1-rs-radial-hamiltonian-v1")
            || lines.next() != Some("units hartree bohr")
        {
            return Err(Gfn1Error::InvalidInput(
                "unsupported ML model version or units".into(),
            ));
        }
        let fields: Vec<&str> = lines.collect();
        let value = |key: &str| {
            fields
                .iter()
                .find_map(|line| line.strip_prefix(&format!("{key} ")))
                .ok_or_else(|| Gfn1Error::InvalidInput(format!("missing ML field {key}")))
        };
        let numbers = |s: &str| {
            s.split_whitespace()
                .map(|x| {
                    x.parse::<f64>()
                        .map_err(|_| Gfn1Error::InvalidInput("invalid ML numeric value".into()))
                })
                .collect::<Result<Vec<_>>>()
        };
        let species = value("species")?
            .split_whitespace()
            .map(|x| {
                x.parse::<u8>()
                    .map_err(|_| Gfn1Error::InvalidInput("invalid ML species".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut model = Self::zero(species, numbers(value("centers")?)?, 0.0);
        model.radial_width_bohr = value("width")?
            .parse()
            .map_err(|_| Gfn1Error::InvalidInput("invalid ML width".into()))?;
        model.cutoff_bohr = value("cutoff")?
            .parse()
            .map_err(|_| Gfn1Error::InvalidInput("invalid ML cutoff".into()))?;
        model.maximum_level_shift_hartree = value("limit")?
            .parse()
            .map_err(|_| Gfn1Error::InvalidInput("invalid ML limit".into()))?;
        model.license_spdx = value("license")?.to_string();
        model.training_provenance = value("provenance")?.to_string();
        let mut seen = std::collections::BTreeSet::new();
        for line in &fields {
            if let Some(line) = line.strip_prefix("head ") {
                let mut words = line.split_whitespace();
                let z = words
                    .next()
                    .and_then(|x| x.parse::<u8>().ok())
                    .ok_or_else(|| Gfn1Error::InvalidInput("invalid ML head species".into()))?;
                let l = words
                    .next()
                    .and_then(|x| x.parse::<usize>().ok())
                    .ok_or_else(|| Gfn1Error::InvalidInput("invalid ML head index".into()))?;
                if !seen.insert((z, l)) {
                    return Err(Gfn1Error::InvalidInput("duplicate ML head".into()));
                }
                let p = numbers(&words.collect::<Vec<_>>().join(" "))?;
                if p.is_empty() {
                    return Err(Gfn1Error::InvalidInput("empty ML head".into()));
                }
                let heads = model
                    .heads
                    .get_mut(&z)
                    .ok_or_else(|| Gfn1Error::InvalidInput("unknown ML head element".into()))?;
                let head = if l < 3 {
                    &mut heads.levels[l]
                } else if l == 3 {
                    &mut heads.correction
                } else {
                    return Err(Gfn1Error::InvalidInput("ML head index out of range".into()));
                };
                *head = LinearHead {
                    bias: p[0],
                    weights: p[1..].to_vec(),
                };
            }
        }
        if seen.len() != 4 * model.species.len() {
            return Err(Gfn1Error::InvalidInput(
                "ML model must specify every head".into(),
            ));
        }
        model.validate()?;
        Ok(model)
    }
    pub fn zero(species: Vec<u8>, centers: Vec<f64>, cutoff: f64) -> Self {
        let n = species.len() * centers.len();
        let head = || LinearHead {
            bias: 0.0,
            weights: vec![0.0; n],
        };
        let heads = species
            .iter()
            .map(|&z| {
                (
                    z,
                    ElementHeads {
                        levels: std::array::from_fn(|_| head()),
                        correction: head(),
                    },
                )
            })
            .collect();
        Self {
            species,
            radial_centers_bohr: centers,
            radial_width_bohr: 1.0,
            cutoff_bohr: cutoff,
            maximum_level_shift_hartree: 0.05,
            heads,
            training_provenance: "untrained zero initialization".into(),
            license_spdx: "GPL-3.0-or-later".into(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.species.len() * self.radial_centers_bohr.len();
        if n == 0
            || self.species.iter().any(|&z| z == 0 || z > 86)
            || self
                .species
                .iter()
                .enumerate()
                .any(|(i, z)| self.species[..i].contains(z))
            || self
                .radial_centers_bohr
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
            || !self.radial_width_bohr.is_finite()
            || self.radial_width_bohr <= 0.0
            || !self.cutoff_bohr.is_finite()
            || self.cutoff_bohr <= 0.0
            || !self.maximum_level_shift_hartree.is_finite()
            || self.maximum_level_shift_hartree <= 0.0
            || self.training_provenance.trim().is_empty()
            || self.license_spdx.trim().is_empty()
        {
            return Err(Gfn1Error::InvalidInput(
                "invalid ML Hamiltonian shape, smooth descriptor, bounds or provenance".into(),
            ));
        }
        for z in &self.species {
            let h = self
                .heads
                .get(z)
                .ok_or_else(|| Gfn1Error::InvalidInput("missing ML element heads".into()))?;
            for head in h.levels.iter().chain(std::iter::once(&h.correction)) {
                if head.weights.len() != n
                    || !head.bias.is_finite()
                    || head.weights.iter().any(|x| !x.is_finite())
                {
                    return Err(Gfn1Error::InvalidInput(
                        "invalid ML head dimensions or weights".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn radial(&self, r: f64, center: f64) -> (f64, f64) {
        if r >= self.cutoff_bohr {
            return (0.0, 0.0);
        }
        let c2 = self.cutoff_bohr * self.cutoff_bohr;
        let gap = c2 - r * r;
        // Compact-support C-infinity cutoff, including all endpoint derivatives.
        let x = r - center;
        let width2 = self.radial_width_bohr * self.radial_width_bohr;
        let value = (-x * x / width2 - r * r / gap).exp();
        (
            value,
            value * (-2.0 * x / width2 - 2.0 * r * c2 / (gap * gap)),
        )
    }
    fn directional_features(&self, system: &PeriodicSystem, v: &[f64]) -> Result<Vec<Vec<Jet1>>> {
        self.validate()?;
        let n = system.atoms.len();
        let nr = self.radial_centers_bohr.len();
        let nf = self.species.len() * nr;
        if n == 0
            || v.len() != 3 * n
            || v.iter().any(|x| !x.is_finite())
            || system.atoms.iter().any(|a| !self.species.contains(&a.z))
        {
            return Err(Gfn1Error::InvalidInput(
                "invalid ML geometry, species or direction".into(),
            ));
        }
        let bytes = n
            .checked_mul(nf)
            .and_then(|x| x.checked_mul(std::mem::size_of::<Jet1>()))
            .ok_or_else(|| Gfn1Error::InvalidInput("ML feature size overflow".into()))?;
        if bytes > 512 * 1024 * 1024 {
            return Err(Gfn1Error::InvalidInput(
                "ML feature jets exceed the 512 MiB budget".into(),
            ));
        }
        let mut features = vec![vec![Jet1::constant(0.0); nf]; n];
        for pair in crate::pairlist::unique_short_range_pairs(system, self.cutoff_bohr)? {
            let mut r2 = Jet1::constant(0.0);
            let dr = pair.dr.to_array();
            for k in 0..3 {
                let x = Jet1 {
                    value: dr[k],
                    d1: v[3 * pair.j + k] - v[3 * pair.i + k],
                    ..Default::default()
                };
                r2 = r2.add(&x.mul(&x));
            }
            let r = r2.sqrt();
            if !r.value.is_finite() || r.value < 1e-10 {
                return Err(Gfn1Error::InvalidInput(
                    "invalid ML neighbor distance".into(),
                ));
            }
            if r.value >= self.cutoff_bohr {
                continue;
            }
            let si = self
                .species
                .iter()
                .position(|&z| z == system.atoms[pair.i].z)
                .unwrap();
            let sj = self
                .species
                .iter()
                .position(|&z| z == system.atoms[pair.j].z)
                .unwrap();
            let gap = r2
                .scale(-1.0)
                .add_scalar(self.cutoff_bohr * self.cutoff_bohr);
            for (k, &center) in self.radial_centers_bohr.iter().enumerate() {
                let x = r.add_scalar(-center);
                let feature = x
                    .mul(&x)
                    .scale(-1.0 / (self.radial_width_bohr * self.radial_width_bohr))
                    .sub(&r2.div(&gap))
                    .exp();
                features[pair.i][sj * nr + k] = features[pair.i][sj * nr + k].add(&feature);
                features[pair.j][si * nr + k] = features[pair.j][si * nr + k].add(&feature);
            }
        }
        Ok(features)
    }
    pub(crate) fn directional_parameters(
        &self,
        system: &PeriodicSystem,
        v: &[f64],
    ) -> Result<DirectionalParameters> {
        let features = self.directional_features(system, v)?;
        let mut levels = Vec::with_capacity(features.len());
        let mut correction = Jet1::constant(0.0);
        for (i, feature) in features.iter().enumerate() {
            let heads = &self.heads[&system.atoms[i].z];
            let raw = |h: &LinearHead| {
                let mut out = Jet1::constant(h.bias);
                for (f, &w) in feature.iter().zip(&h.weights) {
                    out = out.add(&f.scale(w));
                }
                out
            };
            let row = std::array::from_fn(|l| {
                let x = raw(&heads.levels[l]).scale(1.0 / self.maximum_level_shift_hartree);
                let t = x.value.tanh();
                let s = 1.0 - t * t;
                x.compose(
                    t,
                    s,
                    -2.0 * t * s,
                    -2.0 * s * (1.0 - 3.0 * t * t),
                    8.0 * t * s * (2.0 - 3.0 * t * t),
                )
                .scale(self.maximum_level_shift_hartree)
            });
            levels.push(row);
            correction = correction.add(&raw(&heads.correction));
        }
        Ok(DirectionalParameters { levels, correction })
    }
    pub fn evaluate(self: &Arc<Self>, system: &PeriodicSystem) -> Result<AtomicParameterView> {
        self.validate()?;
        if system.atoms.is_empty() || system.atoms.iter().any(|a| !self.heads.contains_key(&a.z)) {
            return Err(Gfn1Error::InvalidInput(
                "ML model lacks a supported molecular element/geometry".into(),
            ));
        }
        let n = system.atoms.len();
        let features: Vec<Vec<f64>> = self
            .directional_features(system, &vec![0.0; 3 * n])?
            .into_iter()
            .map(|v| v.into_iter().map(|x| x.value).collect())
            .collect();
        let mut levels = vec![[0.0; 3]; n];
        let mut slopes = vec![[0.0; 3]; n];
        let mut correction = 0.0;
        for i in 0..n {
            let h = &self.heads[&system.atoms[i].z];
            let evaluate = |head: &LinearHead| {
                head.bias
                    + head
                        .weights
                        .iter()
                        .zip(&features[i])
                        .map(|(w, f)| w * f)
                        .sum::<f64>()
            };
            for l in 0..3 {
                let t = (evaluate(&h.levels[l]) / self.maximum_level_shift_hartree).tanh();
                levels[i][l] = self.maximum_level_shift_hartree * t;
                slopes[i][l] = 1.0 - t * t;
            }
            correction += evaluate(&h.correction);
        }
        if !correction.is_finite() || levels.iter().any(|row| row.iter().any(|x| !x.is_finite())) {
            return Err(Gfn1Error::InvalidInput(
                "non-finite structural ML output".into(),
            ));
        }
        let hash = format!("{:x}", Sha256::digest(self.to_model_string()?.as_bytes()));
        Ok(AtomicParameterView {
            delta_levels_hartree: levels,
            correction_energy_hartree: correction,
            model_sha256: hash,
            features,
            level_slopes: slopes,
            model: Arc::clone(self),
            system: system.clone(),
        })
    }
}

impl AtomicParameterView {
    pub(crate) fn level_head_gradient(
        &self,
        sensitivity: &[[f64; 3]],
    ) -> Result<BTreeMap<u8, ElementHeads>> {
        let mut out = self.energy_head_gradient(sensitivity)?;
        for h in out.values_mut() {
            h.correction.bias = 0.0;
            h.correction.weights.fill(0.0);
        }
        Ok(out)
    }

    pub(crate) fn directional_head_gradient(
        &self,
        sensitivity: &[[f64; 3]],
        sensitivity_derivative: &[[f64; 3]],
        v: &[f64],
    ) -> Result<BTreeMap<u8, ElementHeads>> {
        if sensitivity.len() != self.system.atoms.len()
            || sensitivity_derivative.len() != sensitivity.len()
        {
            return Err(Gfn1Error::InvalidInput(
                "ML mixed head sensitivity size mismatch".into(),
            ));
        }
        let features = self.model.directional_features(&self.system, v)?;
        let nf = features[0].len();
        let mut out = BTreeMap::new();
        for (i, atom) in self.system.atoms.iter().enumerate() {
            let heads = &self.model.heads[&atom.z];
            let g = out.entry(atom.z).or_insert_with(|| ElementHeads {
                levels: std::array::from_fn(|_| LinearHead {
                    bias: 0.0,
                    weights: vec![0.0; nf],
                }),
                correction: LinearHead {
                    bias: 0.0,
                    weights: vec![0.0; nf],
                },
            });
            for l in 0..3 {
                let t = self.delta_levels_hartree[i][l] / self.model.maximum_level_shift_hartree;
                let slope = self.level_slopes[i][l];
                let raw_derivative: f64 = heads.levels[l]
                    .weights
                    .iter()
                    .zip(&features[i])
                    .map(|(w, f)| w * f.d1)
                    .sum();
                let slope_derivative =
                    -2.0 * t * slope * raw_derivative / self.model.maximum_level_shift_hartree;
                let bias =
                    sensitivity_derivative[i][l] * slope + sensitivity[i][l] * slope_derivative;
                g.levels[l].bias += bias;
                for k in 0..nf {
                    g.levels[l].weights[k] +=
                        bias * features[i][k].value + sensitivity[i][l] * slope * features[i][k].d1;
                }
            }
            for k in 0..nf {
                g.correction.weights[k] += features[i][k].d1;
            }
        }
        Ok(out)
    }
    /// Analytic reverse contraction of structural head derivatives with energy
    /// sensitivities. Stores no SCF iteration history and mutates no element table.
    pub fn gradient(&self, sensitivity: &[[f64; 3]]) -> Result<Vec<Vec3>> {
        let n = self.system.atoms.len();
        if sensitivity.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "ML sensitivity size mismatch".into(),
            ));
        }
        let model = &self.model;
        let nr = model.radial_centers_bohr.len();
        let mut g = vec![Vec3::zero(); n];
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, model.cutoff_bohr)? {
            if pair.i == pair.j {
                continue;
            }
            let mut derivative = 0.0;
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let heads = &model.heads[&self.system.atoms[central].z];
                let species = model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                for (k, &center) in model.radial_centers_bohr.iter().enumerate() {
                    let index = species * nr + k;
                    let mut adjoint = heads.correction.weights[index];
                    for l in 0..3 {
                        adjoint += sensitivity[central][l]
                            * self.level_slopes[central][l]
                            * heads.levels[l].weights[index];
                    }
                    derivative += adjoint * model.radial(pair.r, center).1;
                }
            }
            let force = pair.dr * (-derivative / pair.r);
            g[pair.i] += force;
            g[pair.j] -= force;
        }
        Ok(g)
    }
    fn raw_level_jacobian(&self) -> Result<Vec<Vec<[f64; 3]>>> {
        let n = self.system.atoms.len();
        let d = 3 * n;
        if n.checked_mul(d)
            .and_then(|v| v.checked_mul(24))
            .is_none_or(|b| b > 512 * 1024 * 1024)
        {
            return Err(Gfn1Error::InvalidInput(
                "ML level Jacobian exceeds the 512 MiB budget".into(),
            ));
        }
        let mut jac = vec![vec![[0.; 3]; n]; d];
        let nr = self.model.radial_centers_bohr.len();
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, self.model.cutoff_bohr)?
        {
            if pair.i == pair.j {
                continue;
            }
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let species = self
                    .model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                let heads = &self.model.heads[&self.system.atoms[central].z];
                for (k, &center) in self.model.radial_centers_bohr.iter().enumerate() {
                    let radial = pair.dr * (self.model.radial(pair.r, center).1 / pair.r);
                    for l in 0..3 {
                        let gradient =
                            (radial * heads.levels[l].weights[species * nr + k]).to_array();
                        for a in 0..3 {
                            jac[3 * pair.i + a][central][l] -= gradient[a];
                            jac[3 * pair.j + a][central][l] += gradient[a];
                        }
                    }
                }
            }
        }
        Ok(jac)
    }

    pub(crate) fn level_jacobian(&self) -> Result<Vec<Vec<[f64; 3]>>> {
        let mut jac = self.raw_level_jacobian()?;
        for row in &mut jac {
            for (i, levels) in row.iter_mut().enumerate() {
                for l in 0..3 {
                    levels[l] *= self.level_slopes[i][l];
                }
            }
        }
        Ok(jac)
    }

    /// Exact homogeneous engineering-strain derivative of the learned levels.
    /// Unlike a nuclear-Jacobian contraction, this includes self-image pairs.
    pub(crate) fn affine_level_jacobian(&self) -> Result<Vec<Vec<[f64; 3]>>> {
        let mut out = vec![vec![[0.; 3]; self.system.atoms.len()]; 6];
        let axes = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
        let nr = self.model.radial_centers_bohr.len();
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, self.model.cutoff_bohr)?
        {
            let r = pair.dr.to_array();
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let species = self
                    .model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                let heads = &self.model.heads[&self.system.atoms[central].z];
                for (k, &center) in self.model.radial_centers_bohr.iter().enumerate() {
                    let radial = self.model.radial(pair.r, center).1 / pair.r;
                    for (j, &(a, b)) in axes.iter().enumerate() {
                        for l in 0..3 {
                            out[j][central][l] += radial
                                * r[a]
                                * r[b]
                                * heads.levels[l].weights[species * nr + k]
                                * self.level_slopes[central][l];
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Exact engineering-strain Hessian of the learned levels and correction
    /// at fixed electronic level sensitivities. Includes self-image descriptor
    /// derivatives and the nonlocal tanh outer-product response.
    pub(crate) fn affine_fixed_parameter_hessian(
        &self,
        sensitivity: &[[f64; 3]],
    ) -> Result<crate::linalg::Matrix> {
        let nat = self.system.atoms.len();
        if sensitivity.len() != nat {
            return Err(Gfn1Error::InvalidInput(
                "ML affine sensitivity size mismatch".into(),
            ));
        }
        crate::resources::check_dense_workspace(nat, 18, 3, "ML affine descriptor derivatives")?;
        let model = &self.model;
        let nr = model.radial_centers_bohr.len();
        let axes = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
        let mut raw = vec![vec![[0.; 3]; nat]; 6];
        let mut hessian = crate::linalg::Matrix::zeros(6, 6);
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, model.cutoff_bohr)? {
            let r = pair.dr.to_array();
            let velocities: [[f64; 3]; 6] = std::array::from_fn(|s| {
                let (a, b) = axes[s];
                let mut v = [0.; 3];
                if a == b {
                    v[a] = r[b];
                } else {
                    v[a] = 0.5 * r[b];
                    v[b] = 0.5 * r[a];
                }
                v
            });
            let first_r: [f64; 6] = std::array::from_fn(|s| r[axes[s].0] * r[axes[s].1] / pair.r);
            let mut first = 0.;
            let mut second = 0.;
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let heads = &model.heads[&self.system.atoms[central].z];
                let species = model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                for (k, &center) in model.radial_centers_bohr.iter().enumerate() {
                    let index = species * nr + k;
                    let (value, derivative) = model.radial(pair.r, center);
                    if value == 0. {
                        continue;
                    }
                    let mut adjoint = heads.correction.weights[index];
                    for l in 0..3 {
                        adjoint += sensitivity[central][l]
                            * self.level_slopes[central][l]
                            * heads.levels[l].weights[index];
                        for s in 0..6 {
                            raw[s][central][l] +=
                                heads.levels[l].weights[index] * derivative * first_r[s];
                        }
                    }
                    let gap = model.cutoff_bohr.powi(2) - pair.r.powi(2);
                    let log_first = -2. * (pair.r - center) / model.radial_width_bohr.powi(2)
                        - 2. * pair.r * model.cutoff_bohr.powi(2) / gap.powi(2);
                    let log_second = -2. / model.radial_width_bohr.powi(2)
                        - 2. * model.cutoff_bohr.powi(2) / gap.powi(2)
                        - 8. * pair.r.powi(2) * model.cutoff_bohr.powi(2) / gap.powi(3);
                    first += adjoint * derivative;
                    second += adjoint * value * (log_first * log_first + log_second);
                }
            }
            for x in 0..6 {
                for y in 0..6 {
                    let vv = (0..3)
                        .map(|a| velocities[x][a] * velocities[y][a])
                        .sum::<f64>();
                    hessian[(x, y)] += second * first_r[x] * first_r[y]
                        + first * (vv - first_r[x] * first_r[y]) / pair.r;
                }
            }
        }
        for a in 0..nat {
            for l in 0..3 {
                let t = self.delta_levels_hartree[a][l] / model.maximum_level_shift_hartree;
                let coefficient = -2. * sensitivity[a][l] * t * self.level_slopes[a][l]
                    / model.maximum_level_shift_hartree;
                for x in 0..6 {
                    for y in 0..6 {
                        hessian[(x, y)] += coefficient * raw[x][a][l] * raw[y][a][l];
                    }
                }
            }
        }
        Ok(hessian)
    }

    /// Joint nuclear/affine fixed-sensitivity Hessian in the coordinate map
    /// R'=F(R+u). Nuclear coordinates precede nine Cartesian affine components.
    /// Descriptor self images, mixed coordinate acceleration and tanh curvature
    /// are all retained; no displaced geometry is evaluated.
    pub(crate) fn affine_nuclear_fixed_parameter_hessian(
        &self,
        sensitivity: &[[f64; 3]],
    ) -> Result<crate::linalg::Matrix> {
        let nat = self.system.atoms.len();
        let d = 3 * nat + 9;
        if sensitivity.len() != nat {
            return Err(Gfn1Error::InvalidInput(
                "ML joint affine sensitivity mismatch".into(),
            ));
        }
        crate::resources::check_dense_workspace(d, d, 10, "ML joint affine descriptor Hessian")?;
        let model = &self.model;
        let nr = model.radial_centers_bohr.len();
        let mut raw = vec![vec![[0.; 3]; nat]; d];
        let mut out = crate::linalg::Matrix::zeros(d, d);
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, model.cutoff_bohr)? {
            let v =
                crate::pbc::elastic_response::affine_nuclear_vector_jet(pair.dr, d, pair.j, pair.i);
            let mut r2 = crate::jets::Jet2::constant(0., d);
            for x in v {
                r2 = r2.add(&x.mul(&x));
            }
            let r = r2.powf(0.5);
            let mut first = 0.;
            let mut second = 0.;
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let heads = &model.heads[&self.system.atoms[central].z];
                let species = model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                for (k, &center) in model.radial_centers_bohr.iter().enumerate() {
                    let index = species * nr + k;
                    let (value, derivative) = model.radial(pair.r, center);
                    if value == 0. {
                        continue;
                    }
                    let mut adjoint = heads.correction.weights[index];
                    for l in 0..3 {
                        adjoint += sensitivity[central][l]
                            * self.level_slopes[central][l]
                            * heads.levels[l].weights[index];
                        for x in 0..d {
                            raw[x][central][l] +=
                                heads.levels[l].weights[index] * derivative * r.gradient[x];
                        }
                    }
                    let gap = model.cutoff_bohr.powi(2) - pair.r.powi(2);
                    let l1 = -2. * (pair.r - center) / model.radial_width_bohr.powi(2)
                        - 2. * pair.r * model.cutoff_bohr.powi(2) / gap.powi(2);
                    let l2 = -2. / model.radial_width_bohr.powi(2)
                        - 2. * model.cutoff_bohr.powi(2) / gap.powi(2)
                        - 8. * pair.r.powi(2) * model.cutoff_bohr.powi(2) / gap.powi(3);
                    first += adjoint * derivative;
                    second += adjoint * value * (l1 * l1 + l2);
                }
            }
            for x in 0..d {
                for y in 0..d {
                    out[(x, y)] +=
                        second * r.gradient[x] * r.gradient[y] + first * r.hessian[x * d + y];
                }
            }
        }
        for a in 0..nat {
            for l in 0..3 {
                let t = self.delta_levels_hartree[a][l] / model.maximum_level_shift_hartree;
                let coefficient = -2. * sensitivity[a][l] * t * self.level_slopes[a][l]
                    / model.maximum_level_shift_hartree;
                for x in 0..d {
                    for y in 0..d {
                        out[(x, y)] += coefficient * raw[x][a][l] * raw[y][a][l];
                    }
                }
            }
        }
        Ok(out)
    }

    /// d2/dR2 [sum_A,l sensitivity_A,l * delta_h_A,l(R) + E_corr(R)]
    /// with sensitivity fixed. Includes the nonlocal tanh outer-product chain;
    /// a pair-only Hessian would omit that term. No finite differences/jets.
    pub(crate) fn fixed_parameter_hessian(
        &self,
        sensitivity: &[[f64; 3]],
    ) -> Result<crate::linalg::Matrix> {
        let n = self.system.atoms.len();
        let d = 3 * n;
        if sensitivity.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "ML Hessian sensitivity size mismatch".into(),
            ));
        }
        let bytes = d
            .checked_mul(d)
            .and_then(|b| b.checked_mul(8))
            .and_then(|b| {
                n.checked_mul(d)
                    .and_then(|j| j.checked_mul(24))
                    .and_then(|j| b.checked_add(j))
            });
        if bytes.is_none_or(|b| b > 512 * 1024 * 1024) {
            return Err(Gfn1Error::InvalidInput(
                "ML structural Hessian exceeds the 512 MiB budget".into(),
            ));
        }
        let raw = self.raw_level_jacobian()?;
        let mut hessian = crate::linalg::Matrix::zeros(d, d);
        for i in 0..n {
            for l in 0..3 {
                let t = self.delta_levels_hartree[i][l] / self.model.maximum_level_shift_hartree;
                let coefficient = -2. * sensitivity[i][l] * t * self.level_slopes[i][l]
                    / self.model.maximum_level_shift_hartree;
                for x in 0..d {
                    for y in 0..d {
                        hessian[(x, y)] += coefficient * raw[x][i][l] * raw[y][i][l];
                    }
                }
            }
        }
        let model = &self.model;
        let nr = model.radial_centers_bohr.len();
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, model.cutoff_bohr)? {
            if pair.i == pair.j {
                continue;
            }
            let mut first = 0.;
            let mut second = 0.;
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let heads = &model.heads[&self.system.atoms[central].z];
                let species = model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                for (k, &center) in model.radial_centers_bohr.iter().enumerate() {
                    let index = species * nr + k;
                    let mut adjoint = heads.correction.weights[index];
                    for l in 0..3 {
                        adjoint += sensitivity[central][l]
                            * self.level_slopes[central][l]
                            * heads.levels[l].weights[index];
                    }
                    let (value, derivative) = model.radial(pair.r, center);
                    if value == 0. {
                        continue;
                    }
                    let gap = model.cutoff_bohr.powi(2) - pair.r.powi(2);
                    let log_first = -2. * (pair.r - center) / model.radial_width_bohr.powi(2)
                        - 2. * pair.r * model.cutoff_bohr.powi(2) / gap.powi(2);
                    let log_second = -2. / model.radial_width_bohr.powi(2)
                        - 2. * model.cutoff_bohr.powi(2) / gap.powi(2)
                        - 8. * pair.r.powi(2) * model.cutoff_bohr.powi(2) / gap.powi(3);
                    first += adjoint * derivative;
                    second += adjoint * value * (log_first * log_first + log_second);
                }
            }
            let unit = (pair.dr / pair.r).to_array();
            for a in 0..3 {
                for b in 0..3 {
                    let value = (second - first / pair.r) * unit[a] * unit[b]
                        + if a == b { first / pair.r } else { 0. };
                    for (i, j, sign) in [
                        (pair.i, pair.i, 1.),
                        (pair.j, pair.j, 1.),
                        (pair.i, pair.j, -1.),
                        (pair.j, pair.i, -1.),
                    ] {
                        hessian[(3 * i + a, 3 * j + b)] += sign * value;
                    }
                }
            }
        }
        Ok(hessian)
    }

    /// Analytic structural-head/correction stress at fixed electronic density.
    pub fn stress(&self, sensitivity: &[[f64; 3]]) -> Result<crate::linalg::Matrix> {
        let lattice =
            self.system.lattice.as_ref().ok_or_else(|| {
                Gfn1Error::InvalidInput("ML stress requires a periodic cell".into())
            })?;
        if sensitivity.len() != self.system.atoms.len() {
            return Err(Gfn1Error::InvalidInput(
                "ML stress sensitivity size mismatch".into(),
            ));
        }
        let model = &self.model;
        let nr = model.radial_centers_bohr.len();
        let mut out = crate::linalg::Matrix::zeros(3, 3);
        for pair in crate::pairlist::unique_short_range_pairs(&self.system, model.cutoff_bohr)? {
            let mut derivative = 0.0;
            for (central, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
                let heads = &model.heads[&self.system.atoms[central].z];
                let species = model
                    .species
                    .iter()
                    .position(|&z| z == self.system.atoms[neighbor].z)
                    .unwrap();
                for (k, &center) in model.radial_centers_bohr.iter().enumerate() {
                    let index = species * nr + k;
                    let mut adj = heads.correction.weights[index];
                    for l in 0..3 {
                        adj += sensitivity[central][l]
                            * self.level_slopes[central][l]
                            * heads.levels[l].weights[index];
                    }
                    derivative += adj * model.radial(pair.r, center).1;
                }
            }
            let r = pair.dr.to_array();
            for a in 0..3 {
                for b in 0..3 {
                    out[(a, b)] += derivative * r[a] * r[b] / (pair.r * lattice.volume());
                }
            }
        }
        Ok(out)
    }
    /// Exact stationary-energy gradient w.r.t. level-head weights/biases. The
    /// density response cancels by SCF stationarity; no unrolled iterations.
    pub fn energy_head_gradient(
        &self,
        sensitivity: &[[f64; 3]],
    ) -> Result<BTreeMap<u8, ElementHeads>> {
        if sensitivity.len() != self.features.len() {
            return Err(Gfn1Error::InvalidInput(
                "ML training sensitivity size mismatch".into(),
            ));
        }
        let nf = self.features[0].len();
        let mut out = BTreeMap::new();
        for (i, atom) in self.system.atoms.iter().enumerate() {
            let heads = out.entry(atom.z).or_insert_with(|| ElementHeads {
                levels: std::array::from_fn(|_| LinearHead {
                    bias: 0.0,
                    weights: vec![0.0; nf],
                }),
                correction: LinearHead {
                    bias: 0.0,
                    weights: vec![0.0; nf],
                },
            });
            for l in 0..3 {
                let weight = sensitivity[i][l] * self.level_slopes[i][l];
                heads.levels[l].bias += weight;
                for k in 0..nf {
                    heads.levels[l].weights[k] += weight * self.features[i][k];
                }
            }
            heads.correction.bias += 1.0;
            for k in 0..nf {
                heads.correction.weights[k] += self.features[i][k];
            }
        }
        Ok(out)
    }
}

/// Hellmann-Feynman dE/d(delta h_A,l) for the actual GFN1 H0 expression,
/// including the off-site Hückel scaling/polynomial and on-site overlap.
pub fn level_sensitivities(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
) -> Result<Vec<[f64; 3]>> {
    let basis = &result.basis;
    let mut out = vec![[0.0; 3]; system.atoms.len()];
    for i in 0..basis.len() {
        let a = &basis.shells[basis.aos[i].shell_index];
        for j in 0..basis.len() {
            let b = &basis.shells[basis.aos[j].shell_index];
            let factor = if a.atom_index == b.atom_index {
                1.0
            } else {
                let r = (system.atoms[a.atom_index].position - system.atoms[b.atom_index].position)
                    .norm();
                let rad = crate::data_tables::atomic_radius_bohr(a.z)?
                    + crate::data_tables::atomic_radius_bohr(b.z)?;
                crate::hamiltonian::hscale(a, b, params)?
                    * crate::hamiltonian::shell_polynomial(a, b, (r / rad).sqrt())
            };
            out[a.atom_index][a.angular.as_index()] +=
                result.density[(i, j)] * result.integrals.overlap[(i, j)] * factor;
        }
    }
    Ok(out)
}

pub(crate) fn level_sensitivity_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
    density_derivative: &crate::linalg::Matrix,
    overlap_derivative: &crate::linalg::Matrix,
    v: &[f64],
) -> Result<Vec<[f64; 3]>> {
    let basis = &result.basis;
    let n = basis.len();
    if v.len() != 3 * system.atoms.len()
        || density_derivative.rows() != n
        || density_derivative.cols() != n
        || overlap_derivative.rows() != n
        || overlap_derivative.cols() != n
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid ML sensitivity derivative dimensions".into(),
        ));
    }
    let mut out = vec![[0.0; 3]; system.atoms.len()];
    for i in 0..n {
        let a = &basis.shells[basis.aos[i].shell_index];
        for j in 0..n {
            let b = &basis.shells[basis.aos[j].shell_index];
            let (factor, derivative) = if a.atom_index == b.atom_index {
                (1.0, 0.0)
            } else {
                let dr = system.atoms[a.atom_index].position - system.atoms[b.atom_index].position;
                let r = dr.norm();
                let velocity = Vec3::new(
                    v[3 * a.atom_index] - v[3 * b.atom_index],
                    v[3 * a.atom_index + 1] - v[3 * b.atom_index + 1],
                    v[3 * a.atom_index + 2] - v[3 * b.atom_index + 2],
                );
                let rad = crate::data_tables::atomic_radius_bohr(a.z)?
                    + crate::data_tables::atomic_radius_bohr(b.z)?;
                let rr = (r / rad).sqrt();
                let rr1 = 0.5 * rr * dr.dot(velocity) / (r * r);
                let pi = a.poly_raw.unwrap_or(0.0);
                let pj = b.poly_raw.unwrap_or(0.0);
                let scale = crate::hamiltonian::hscale(a, b, params)?;
                (
                    scale * (1.0 + pi * rr) * (1.0 + pj * rr),
                    scale * (pi + pj + 2.0 * pi * pj * rr) * rr1,
                )
            };
            out[a.atom_index][a.angular.as_index()] +=
                density_derivative[(i, j)] * result.integrals.overlap[(i, j)] * factor
                    + result.density[(i, j)]
                        * (overlap_derivative[(i, j)] * factor
                            + result.integrals.overlap[(i, j)] * derivative);
        }
    }
    Ok(out)
}

pub(crate) fn apply_parameter_view(basis: &mut BasisSet, view: &AtomicParameterView) {
    for shell in &mut basis.shells {
        shell.hdiag_ha += view.delta_levels_hartree[shell.atom_index][shell.angular.as_index()];
    }
}

pub(crate) fn validate_scalar_options(options: &crate::ElectronicOptions) -> Result<()> {
    if options.ml_model.is_some()
        && (options.spin_polarization
            || options.plus_u
            || options.multipole
            || options.lr_exchange
            || options.experimental_d4
            || options.solvation.is_some()
            || options.scf_trah
            || options.external_field.magnetic_field.is_some())
    {
        return Err(Gfn1Error::InvalidInput(
            "ML Hamiltonian heads support scalar GFN1 and electric fields; requested composite model is not implemented".into(),
        ));
    }
    Ok(())
}

pub struct EnergyTrainingExample {
    pub system: PeriodicSystem,
    pub energy_hartree: f64,
}
#[derive(Clone, Copy, Debug)]
pub enum EnergyOptimizer {
    GradientDescent,
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct EnergyTrainingOptions {
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub optimizer: EnergyOptimizer,
    pub decay_bias: bool,
    pub train_correction: bool,
}
impl Default for EnergyTrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            weight_decay: 1e-8,
            optimizer: EnergyOptimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            decay_bias: false,
            train_correction: false,
        }
    }
}
/// Small-model energy fitting using the stationary SCF gradient. Force training
/// and non-stationary/unrolled SCF gradients are separate capabilities.
pub fn fit_energy_heads(
    model: &mut MlHamiltonian,
    params: &Gfn1Parameters,
    electronic: &crate::ElectronicOptions,
    data: &[EnergyTrainingExample],
    options: EnergyTrainingOptions,
) -> Result<Vec<f64>> {
    let span=electronic.logger.as_ref().map(|logger|logger.begin("ml_energy_training",serde_json::json!({"samples":data.len(),"epochs":options.epochs,"optimizer":format!("{:?}",options.optimizer),"learning_rate":options.learning_rate,"weight_decay":options.weight_decay}))).transpose()?;
    crate::logging::finish(
        span,
        fit_energy_heads_impl(model, params, electronic, data, options),
        |loss| serde_json::json!({"epochs":loss.len(),"first_evaluated_half_mse":loss.first(),"last_evaluated_half_mse":loss.last()}),
    )
}
fn fit_energy_heads_impl(
    model: &mut MlHamiltonian,
    params: &Gfn1Parameters,
    electronic: &crate::ElectronicOptions,
    data: &[EnergyTrainingExample],
    options: EnergyTrainingOptions,
) -> Result<Vec<f64>> {
    if data.is_empty()
        || data.iter().any(|x| !x.energy_hartree.is_finite())
        || options.epochs == 0
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || !options.weight_decay.is_finite()
        || options.weight_decay < 0.0
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid ML training data or optimizer".into(),
        ));
    }
    if let EnergyOptimizer::AdamW {
        beta1,
        beta2,
        epsilon,
    } = options.optimizer
    {
        if !(beta1.is_finite()
            && beta2.is_finite()
            && (0.0..1.0).contains(&beta1)
            && (0.0..1.0).contains(&beta2)
            && epsilon.is_finite()
            && epsilon > 0.0)
        {
            return Err(Gfn1Error::InvalidInput(
                "invalid AdamW moments or epsilon".into(),
            ));
        }
    }
    model.validate()?;
    let mut history = Vec::new();
    history.try_reserve_exact(options.epochs).map_err(|e| {
        Gfn1Error::InvalidInput(format!("ML training history allocation failed: {e}"))
    })?;
    let mut first_moment = SelfGradient::zero(model);
    let mut second_moment = SelfGradient::zero(model);
    for epoch in 1..=options.epochs {
        let mut derivative = SelfGradient::zero(model);
        let mut loss = 0.0;
        let mut opts = electronic.clone();
        opts.ml_model = Some(Arc::new(model.clone()));
        for example in data {
            let result = crate::run_electronic(&example.system, params, opts.clone())?;
            let residual = result.total_free - example.energy_hartree;
            loss += 0.5 * residual * residual / data.len() as f64;
            let sensitivities = level_sensitivities(&example.system, params, &result)?;
            let gradient = result
                .ml_parameters
                .as_ref()
                .unwrap()
                .energy_head_gradient(&sensitivities)?;
            derivative.accumulate(&gradient, residual / data.len() as f64);
        }
        history.push(loss);
        if let Some(logger) = &electronic.logger {
            logger.event(crate::LogLevel::Info,"training_epoch",serde_json::json!({"epoch":epoch,"epochs":options.epochs,"half_mse_hartree2":loss,"learning_rate":options.learning_rate}))?;
        }
        for (&z, heads) in &mut model.heads {
            let grad = &derivative.0[&z];
            let first = first_moment.0.get_mut(&z).unwrap();
            let second = second_moment.0.get_mut(&z).unwrap();
            for l in 0..if options.train_correction { 4 } else { 3 } {
                let (head, g) = if l < 3 {
                    (&mut heads.levels[l], &grad.levels[l])
                } else {
                    (&mut heads.correction, &grad.correction)
                };
                let (m, v) = if l < 3 {
                    (&mut first.levels[l], &mut second.levels[l])
                } else {
                    (&mut first.correction, &mut second.correction)
                };
                update_parameter(
                    &mut head.bias,
                    g.bias,
                    &mut m.bias,
                    &mut v.bias,
                    epoch,
                    &options,
                    options.decay_bias,
                );
                for k in 0..head.weights.len() {
                    update_parameter(
                        &mut head.weights[k],
                        g.weights[k],
                        &mut m.weights[k],
                        &mut v.weights[k],
                        epoch,
                        &options,
                        true,
                    );
                }
            }
        }
    }
    model.validate()?;
    Ok(history)
}

pub(crate) fn update_parameter(
    w: &mut f64,
    g: f64,
    m: &mut f64,
    v: &mut f64,
    step: usize,
    options: &EnergyTrainingOptions,
    decay: bool,
) {
    let rate = options.learning_rate;
    let lambda = if decay { options.weight_decay } else { 0.0 };
    match options.optimizer {
        EnergyOptimizer::GradientDescent => *w -= rate * (g + lambda * *w),
        EnergyOptimizer::AdamW {
            beta1,
            beta2,
            epsilon,
        } => {
            *m = beta1 * *m + (1.0 - beta1) * g;
            *v = beta2 * *v + (1.0 - beta2) * g * g;
            let mhat = *m / (1.0 - beta1.powf(step as f64));
            let vhat = *v / (1.0 - beta2.powf(step as f64));
            *w = *w * (1.0 - rate * lambda) - rate * mhat / (vhat.sqrt() + epsilon);
        }
    }
}
pub(crate) struct SelfGradient(pub(crate) BTreeMap<u8, ElementHeads>);
impl SelfGradient {
    pub(crate) fn zero(model: &MlHamiltonian) -> Self {
        let n = model.species.len() * model.radial_centers_bohr.len();
        Self(
            model
                .species
                .iter()
                .map(|&z| {
                    (
                        z,
                        ElementHeads {
                            levels: std::array::from_fn(|_| LinearHead {
                                bias: 0.0,
                                weights: vec![0.0; n],
                            }),
                            correction: LinearHead {
                                bias: 0.0,
                                weights: vec![0.0; n],
                            },
                        },
                    )
                })
                .collect(),
        )
    }
    pub(crate) fn accumulate(&mut self, gradient: &BTreeMap<u8, ElementHeads>, weight: f64) {
        for (&z, h) in gradient {
            let target = self.0.get_mut(&z).unwrap();
            for l in 0..4 {
                let (a, b) = if l < 3 {
                    (&mut target.levels[l], &h.levels[l])
                } else {
                    (&mut target.correction, &h.correction)
                };
                a.bias += weight * b.bias;
                for (w, d) in a.weights.iter_mut().zip(&b.weights) {
                    *w += weight * d;
                }
            }
        }
    }
}

#[cfg(test)]
mod optimizer_tests {
    use super::*;
    #[test]
    fn adamw_matches_constant_gradient_solution_with_decoupled_decay() {
        let options = EnergyTrainingOptions {
            learning_rate: 0.1,
            weight_decay: 0.01,
            optimizer: EnergyOptimizer::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-12,
            },
            ..Default::default()
        };
        let (mut w, mut m, mut v) = (2.0, 0.0, 0.0);
        for step in 1..=5 {
            update_parameter(&mut w, 1.0, &mut m, &mut v, step, &options, true);
            let factor = (1.0 - options.learning_rate * options.weight_decay).powi(step as i32);
            let expected = 2.0 * factor - (1.0 - factor) / options.weight_decay;
            assert!((w - expected).abs() < 1e-11);
        }
    }
}
