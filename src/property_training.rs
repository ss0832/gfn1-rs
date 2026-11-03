// SPDX-License-Identifier: GPL-3.0-or-later
//! Warm-start property-objective fitting of native ML Hamiltonian heads.
//!
//! Nuclear/strain/field observables are evaluated by their analytic production
//! backends. General objective gradients use central differences of **model
//! parameters**, never differences of nuclear positions, cells or fields.
use crate::linalg::{symmetric_eigen, Matrix};
use crate::ml::{LinearHead, MlHamiltonian};
use crate::pbc::elastic_response::{pbc_elastic_response, AnalyticElasticOptions};
use crate::pbc::electromechanical::{pbc_electromechanical_response, ElectromechanicalOptions};
use crate::pbc::polarization::response::{pbc_electronic_dielectric, pbc_polarization_derivatives};
use crate::pbc::{BerryPolarizationOptions, KMesh, PbcOptions};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn require_scalar_options(e: &ElectronicOptions) -> Result<()> {
    if e.multipole
        || e.lr_exchange
        || e.spin_polarization
        || e.plus_u
        || e.solvation.is_some()
        || e.external_field.is_active()
        || e.spin_multiplicity.is_some_and(|m| m != 1)
        || e.experimental_d4
    {
        return Err(invalid("property-objective training currently requires the scalar GFN1/ML Hamiltonian without composite terms, solvation or external fields; unsupported physical derivatives are not replaced by differences"));
    }
    Ok(())
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn rows(m: &Matrix) -> Value {
    json!((0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Backend {
    Ground,
    Hessian,
    Elastic,
    Berry,
    Dielectric,
    Coupled,
    Bands,
}

/// Each name fixes both the physical definition and its native output unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PropertyObservable {
    Energy,
    Forces,
    Stress,
    Hessian,
    PhononEigenvalues,
    PhononWavenumbers,
    ElasticClamped,
    ElasticRelaxed,
    BornEffectiveCharges,
    PiezoelectricClamped,
    ElectronicDielectric,
    PiezoelectricRelaxed,
    PiezoelectricStrain,
    StaticDielectricFixedStrain,
    StaticDielectricFreeStress,
    BandEnergies,
    BandGap,
}
impl PropertyObservable {
    fn specification(self) -> (&'static str, &'static str, Backend, &'static str) {
        use PropertyObservable::*;
        match self {
            Energy => ("energy", "hartree", Backend::Ground, "/energy_hartree"),
            Forces => (
                "forces",
                "hartree/bohr",
                Backend::Ground,
                "/forces_hartree_per_bohr",
            ),
            Stress => (
                "stress",
                "hartree/bohr^3",
                Backend::Ground,
                "/stress_hartree_per_bohr3",
            ),
            Hessian => (
                "hessian",
                "hartree/bohr^2",
                Backend::Hessian,
                "/hessian_hartree_per_bohr2",
            ),
            PhononEigenvalues => (
                "phonon_eigenvalues",
                "hartree/bohr^2/amu",
                Backend::Hessian,
                "/phonon_eigenvalues_hartree_per_bohr2_amu",
            ),
            PhononWavenumbers => (
                "phonon_wavenumbers",
                "cm^-1",
                Backend::Hessian,
                "/phonon_wavenumbers_cm1",
            ),
            ElasticClamped => (
                "elastic_clamped",
                "hartree/bohr^3",
                Backend::Elastic,
                "/clamped_stiffness_hartree_per_bohr3",
            ),
            ElasticRelaxed => (
                "elastic_relaxed",
                "hartree/bohr^3",
                Backend::Elastic,
                "/relaxed_stiffness_hartree_per_bohr3",
            ),
            BornEffectiveCharges => (
                "born_effective_charges",
                "e",
                Backend::Berry,
                "/born_effective_charges",
            ),
            PiezoelectricClamped => (
                "piezoelectric_clamped",
                "e/bohr^2",
                Backend::Berry,
                "/clamped_piezoelectric_au",
            ),
            ElectronicDielectric => (
                "dielectric_electronic",
                "dimensionless",
                Backend::Dielectric,
                "/electronic_relative_dielectric",
            ),
            PiezoelectricRelaxed => (
                "piezoelectric_relaxed",
                "e/bohr^2",
                Backend::Coupled,
                "/relaxed_piezoelectric_au",
            ),
            PiezoelectricStrain => (
                "piezoelectric_strain",
                "e*bohr/hartree",
                Backend::Coupled,
                "/piezoelectric_strain_au",
            ),
            StaticDielectricFixedStrain => (
                "dielectric_static_fixed_strain",
                "dimensionless",
                Backend::Coupled,
                "/static_dielectric_fixed_strain",
            ),
            StaticDielectricFreeStress => (
                "dielectric_static_free_stress",
                "dimensionless",
                Backend::Coupled,
                "/static_dielectric_free_stress",
            ),
            BandEnergies => (
                "band_energies",
                "hartree",
                Backend::Bands,
                "/energies_hartree",
            ),
            BandGap => (
                "band_gap",
                "hartree",
                Backend::Bands,
                "/sampled_gap/indirect_gap_hartree",
            ),
        }
    }
    pub fn name(self) -> &'static str {
        self.specification().0
    }
    pub fn unit(self) -> &'static str {
        self.specification().1
    }
    pub fn parse(name: &str) -> Result<Self> {
        Self::all()
            .iter()
            .copied()
            .find(|p| p.name() == name)
            .ok_or_else(|| invalid(format!("unknown property objective {name}")))
    }
    pub fn all() -> &'static [Self] {
        use PropertyObservable::*;
        &[
            Energy,
            Forces,
            Stress,
            Hessian,
            PhononEigenvalues,
            PhononWavenumbers,
            ElasticClamped,
            ElasticRelaxed,
            BornEffectiveCharges,
            PiezoelectricClamped,
            ElectronicDielectric,
            PiezoelectricRelaxed,
            PiezoelectricStrain,
            StaticDielectricFixedStrain,
            StaticDielectricFreeStress,
            BandEnergies,
            BandGap,
        ]
    }
    fn correction_sensitive(self) -> bool {
        !matches!(
            self,
            Self::BornEffectiveCharges
                | Self::PiezoelectricClamped
                | Self::ElectronicDielectric
                | Self::BandEnergies
                | Self::BandGap
        )
    }
}

pub fn property_registry() -> Value {
    json!(PropertyObservable::all().iter().map(|p| {
        let (name,unit,backend,pointer)=p.specification();
        json!({"name":name,"unit":unit,"backend":format!("{backend:?}"),"output_pointer":pointer,
            "physical_derivatives":"analytic","periodic_only":!matches!(p,PropertyObservable::Energy|PropertyObservable::Forces|PropertyObservable::Hessian|PropertyObservable::PhononEigenvalues|PropertyObservable::PhononWavenumbers),
            "phonon_scope":if matches!(p,PropertyObservable::PhononEigenvalues|PropertyObservable::PhononWavenumbers) {Some("Gamma mass-weighted 3N eigenvalues, including signed unstable branches; no acoustic projection")}else{None}})
    }).collect::<Vec<_>>())
}

#[derive(Clone, Debug)]
pub struct PropertyTarget {
    pub observable: PropertyObservable,
    /// A scalar or nested rectangular array, with the registry's exact unit.
    pub values: Value,
    /// Same shape as `values`, containing booleans. None selects all entries.
    pub mask: Option<Value>,
    pub weight: f64,
    /// Residual divisor in the observable's native unit, strictly positive.
    pub scale: f64,
}
impl PropertyTarget {
    pub fn new(observable: PropertyObservable, values: Value, scale: f64) -> Self {
        Self {
            observable,
            values,
            mask: None,
            weight: 1.,
            scale,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertySplit {
    Train,
    Validation,
}
impl PropertySplit {
    fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
        }
    }
}
#[derive(Clone, Debug)]
pub struct PropertySample {
    pub id: String,
    pub split: PropertySplit,
    pub system: PeriodicSystem,
    pub electronic: ElectronicOptions,
    /// Mandatory for periodic samples; never inferred from a hidden grid.
    pub periodic: Option<PbcOptions>,
    pub masses_amu: Vec<f64>,
    pub band_kpoints: Vec<[f64; 3]>,
    pub targets: Vec<PropertyTarget>,
}
impl PropertySample {
    pub fn new(id: impl Into<String>, split: PropertySplit, system: PeriodicSystem) -> Self {
        Self {
            id: id.into(),
            split,
            system,
            electronic: ElectronicOptions {
                electronic_temperature: 300.,
                energy_tolerance: 1e-12,
                charge_tolerance: 1e-10,
                ..Default::default()
            },
            periodic: None,
            masses_amu: Vec::new(),
            band_kpoints: Vec::new(),
            targets: Vec::new(),
        }
    }
    pub fn conditions_json(&self) -> Value {
        json!({"id":self.id,"split":self.split.name(),"charge":self.system.charge,
            "positions_bohr":self.system.atoms.iter().map(|a|a.position.to_array()).collect::<Vec<_>>(),
            "atomic_numbers":self.system.atoms.iter().map(|a|a.z).collect::<Vec<_>>(),
            "cell_bohr":self.system.lattice.map(|l|l.cell.col.map(|r|r.to_array())),
            "periodic_axes":self.system.lattice.map(|l|l.periodic),
            "electronic_temperature_kelvin":self.electronic.electronic_temperature,
            "energy_definition":"mermin_free","scc_energy_tolerance_hartree":self.electronic.energy_tolerance.min(1e-12),
            "scc_charge_tolerance_e":self.electronic.charge_tolerance.min(1e-10),
            "requested_scc_energy_tolerance_hartree":self.electronic.energy_tolerance,
            "requested_scc_charge_tolerance_e":self.electronic.charge_tolerance,"scc_max_iterations":self.electronic.max_scc,
            "mixing":self.electronic.mixing,"enable_dispersion":self.electronic.enable_dispersion,
            "scc_accelerator":format!("{:?}",self.electronic.scc_accelerator),"scc_broyden":self.electronic.scc_broyden,
            "basis_nprim":self.electronic.nprim,"eigen_tolerance":self.electronic.eigen_tolerance,
            "charge_order":self.electronic.charge_order,"kmesh":self.periodic.map(|p|p.kmesh.size),
            "gamma_centered":self.periodic.map(|p|p.kmesh.gamma_centered),"fold_time_reversal":self.periodic.map(|p|p.kmesh.fold_time_reversal),
            "ao_cutoff_bohr":self.periodic.map(|p|p.ao_cutoff),"masses_amu":self.masses_amu,"band_kpoints":self.band_kpoints,
            "ewald":self.periodic.map(|p|json!({"k_split":p.ewald.k_split,"real_cutoff_bohr":p.ewald.real_cutoff,"reciprocal_cutoff_inverse_bohr":p.ewald.recip_cutoff,"short_range_cutoff_bohr":p.ewald.sr_cutoff})),
            "hamiltonian":format!("{:?}",self.electronic.hamiltonian),
            "targets":self.targets.iter().map(|t|json!({"observable":t.observable.name(),"unit":t.observable.unit(),"values":t.values,"mask":t.mask,"weight":t.weight,"scale":t.scale})).collect::<Vec<_>>(),
            "cp_controls":crate::resources::current_cp().as_json()})
    }
}

#[derive(Clone, Debug)]
pub struct PropertyDataset {
    pub samples: Vec<PropertySample>,
    pub provenance: String,
    pub license_spdx: String,
    pub reference_method: String,
    pub independent_reference: bool,
    pub source_sha256: Option<String>,
}
impl PropertyDataset {
    /// Strict, unit-tagged portable data format used by Rust, Python and CLI.
    pub fn from_json(value: &Value) -> Result<Self> {
        let text = |v: &Value, key: &str| -> Result<String> {
            v[key]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("missing text field {key}")))
        };
        if text(value, "schema")? != "gfn1-rs-property-training-v1" {
            return Err(invalid("unsupported property-training schema"));
        }
        let bohr = match text(value, "coordinate_units")?.as_str() {
            "bohr" => true,
            "angstrom" => false,
            _ => return Err(invalid("coordinate_units must be bohr or angstrom")),
        };
        let mut out = Self {
            samples: Vec::new(),
            provenance: text(value, "provenance")?,
            license_spdx: text(value, "license_spdx")?,
            reference_method: text(value, "reference_method")?,
            independent_reference: value["independent_reference"]
                .as_bool()
                .ok_or_else(|| invalid("declare independent_reference"))?,
            source_sha256: Some(sha(
                &serde_json::to_vec(value).map_err(|e| invalid(e.to_string()))?
            )),
        };
        for v in value["samples"]
            .as_array()
            .ok_or_else(|| invalid("samples must be an array"))?
        {
            let split = match text(v, "split")?.as_str() {
                "train" => PropertySplit::Train,
                "validation" => PropertySplit::Validation,
                _ => return Err(invalid("split must explicitly be train or validation")),
            };
            let charge = number(&v["charge"], "sample charge")?;
            let mut s = PropertySample::new(
                text(v, "id")?,
                split,
                PeriodicSystem::from_xyz_str(&text(v, "xyz")?, charge, bohr)?,
            );
            s.electronic.electronic_temperature = number(
                &v["electronic_temperature_kelvin"],
                "electronic temperature",
            )?;
            if let Some(x) = v.get("scc_energy_tolerance_hartree") {
                s.electronic.energy_tolerance = number(x, "SCC energy tolerance")?;
            }
            if let Some(x) = v.get("scc_charge_tolerance_e") {
                s.electronic.charge_tolerance = number(x, "SCC charge tolerance")?;
            }
            if let Some(x) = v.get("scc_max_iterations") {
                s.electronic.max_scc = x
                    .as_u64()
                    .ok_or_else(|| invalid("invalid SCC max iterations"))?
                    as usize;
            }
            if let Some(x) = v.get("mixing") {
                s.electronic.mixing = number(x, "mixing")?;
            }
            if let Some(x) = v.get("enable_dispersion") {
                s.electronic.enable_dispersion = x
                    .as_bool()
                    .ok_or_else(|| invalid("invalid enable_dispersion"))?;
            }
            if s.system.lattice.is_some() {
                let a = v["kmesh"]
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .ok_or_else(|| invalid("periodic samples require explicit kmesh[3]"))?;
                let mut mesh = [0; 3];
                for i in 0..3 {
                    mesh[i] = a[i]
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|&n| n > 0)
                        .ok_or_else(|| invalid("kmesh entries must be positive integers"))?;
                }
                s.periodic = Some(PbcOptions {
                    kmesh: KMesh {
                        size: mesh,
                        gamma_centered: v["gamma_centered"]
                            .as_bool()
                            .ok_or_else(|| invalid("declare gamma_centered"))?,
                        fold_time_reversal: v["fold_time_reversal"]
                            .as_bool()
                            .ok_or_else(|| invalid("declare fold_time_reversal"))?,
                    },
                    ao_cutoff: number(&v["ao_cutoff_bohr"], "periodic AO cutoff")?,
                    ..Default::default()
                });
            }
            if let Some(a) = v.get("masses_amu") {
                s.masses_amu = numeric_vector(a, "masses")?;
            }
            if let Some(a) = v.get("band_kpoints") {
                for k in a
                    .as_array()
                    .ok_or_else(|| invalid("band_kpoints must be an array"))?
                {
                    let k = numeric_vector(k, "band k point")?;
                    s.band_kpoints.push(
                        k.try_into()
                            .map_err(|_| invalid("band k point requires 3 coordinates"))?,
                    );
                }
            }
            for t in v["targets"]
                .as_array()
                .ok_or_else(|| invalid("sample requires targets"))?
            {
                let observable = PropertyObservable::parse(&text(t, "observable")?)?;
                if text(t, "unit")? != observable.unit() {
                    return Err(invalid(format!(
                        "{} requires unit {}",
                        observable.name(),
                        observable.unit()
                    )));
                }
                s.targets.push(PropertyTarget {
                    observable,
                    values: t
                        .get("values")
                        .ok_or_else(|| invalid("target requires values"))?
                        .clone(),
                    mask: t.get("mask").filter(|v| !v.is_null()).cloned(),
                    weight: t
                        .get("weight")
                        .map(|x| number(x, "target weight"))
                        .transpose()?
                        .unwrap_or(1.),
                    scale: number(&t["scale"], "target scale")?,
                });
            }
            out.samples.push(s);
        }
        out.validate()?;
        Ok(out)
    }
    pub fn validate(&self) -> Result<()> {
        if self.samples.is_empty()
            || [
                self.provenance.as_str(),
                self.license_spdx.as_str(),
                self.reference_method.as_str(),
            ]
            .iter()
            .any(|s| s.trim().is_empty())
        {
            return Err(invalid("property dataset requires nonempty samples, provenance, license and reference method"));
        }
        let mut ids = BTreeSet::new();
        for s in &self.samples {
            if s.id.trim().is_empty()
                || !ids.insert(&s.id)
                || s.system.atoms.is_empty()
                || s.targets.is_empty()
            {
                return Err(invalid(
                    "property samples require unique nonempty ids, atoms and targets",
                ));
            }
            if !s.electronic.electronic_temperature.is_finite()
                || s.electronic.electronic_temperature < 0.
                || !s.electronic.energy_tolerance.is_finite()
                || s.electronic.energy_tolerance <= 0.
                || !s.electronic.charge_tolerance.is_finite()
                || s.electronic.charge_tolerance <= 0.
                || s.electronic.max_scc == 0
            {
                return Err(invalid("invalid electronic temperature or SCC controls"));
            }
            if s.electronic
                .charge
                .is_some_and(|charge| charge != s.system.charge)
            {
                return Err(invalid(
                    "sample system/electronic charge disagree; refusing a label-condition mismatch",
                ));
            }
            require_scalar_options(&s.electronic)?;
            if s.system.lattice.is_some() != s.periodic.is_some() {
                return Err(invalid(
                    "periodic samples must supply explicit PbcOptions, molecular samples must not",
                ));
            }
            if let Some(p) = s.periodic {
                if p.kmesh.size.contains(&0) || !p.ao_cutoff.is_finite() || p.ao_cutoff <= 0. {
                    return Err(invalid("invalid periodic sampling or cutoff"));
                }
            }
            let mut properties = BTreeSet::new();
            for t in &s.targets {
                if !properties.insert(t.observable)
                    || !t.weight.is_finite()
                    || t.weight < 0.
                    || !t.scale.is_finite()
                    || t.scale <= 0.
                {
                    return Err(invalid(
                        "duplicate target, negative weight or nonpositive scale",
                    ));
                }
                let mut selected = Vec::new();
                selected_values(&t.values, &t.values, t.mask.as_ref(), &mut selected)?;
                if selected.is_empty() {
                    return Err(invalid("target mask selects no components"));
                }
            }
        }
        if !self
            .samples
            .iter()
            .any(|s| s.split == PropertySplit::Train && s.targets.iter().any(|t| t.weight > 0.))
            || !self.samples.iter().any(|s| {
                s.split == PropertySplit::Validation && s.targets.iter().any(|t| t.weight > 0.)
            })
        {
            return Err(invalid(
                "explicit nonempty training and held-out validation splits are required",
            ));
        }
        Ok(())
    }
}
fn number(v: &Value, name: &str) -> Result<f64> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| invalid(format!("invalid {name}")))
}
fn numeric_vector(v: &Value, name: &str) -> Result<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| invalid(format!("{name} must be an array")))?
        .iter()
        .map(|x| number(x, name))
        .collect()
}

fn selected_values(
    pred: &Value,
    target: &Value,
    mask: Option<&Value>,
    out: &mut Vec<(f64, f64)>,
) -> Result<()> {
    if let Some(reference) = target.as_array() {
        let prediction = pred
            .as_array()
            .filter(|a| a.len() == reference.len())
            .ok_or_else(|| invalid("property prediction/target tensor shape mismatch"))?;
        let masked = mask
            .map(|m| {
                m.as_array()
                    .filter(|a| a.len() == reference.len())
                    .ok_or_else(|| invalid("property mask tensor shape mismatch"))
            })
            .transpose()?;
        for i in 0..reference.len() {
            selected_values(&prediction[i], &reference[i], masked.map(|m| &m[i]), out)?;
        }
    } else {
        let p = number(pred, "property prediction")?;
        let r = number(target, "property target")?;
        if mask
            .map(|m| {
                m.as_bool()
                    .ok_or_else(|| invalid("mask entries must be booleans"))
            })
            .transpose()?
            .unwrap_or(true)
        {
            out.push((p, r));
        }
    }
    Ok(())
}

fn ensure_temperature(actual: f64, sample: &PropertySample) -> Result<()> {
    if actual != sample.electronic.electronic_temperature {
        return Err(invalid(format!(
            "sample {} changed electronic temperature from {} to {actual}; refusing label mismatch",
            sample.id, sample.electronic.electronic_temperature
        )));
    }
    Ok(())
}

/// Predict only requested observables. Backend JSON paths are fixed by the
/// typed registry; null/unstable/unsupported properties are errors, never zeros.
pub fn evaluate_property_sample(
    model: &MlHamiltonian,
    params: &Gfn1Parameters,
    sample: &PropertySample,
) -> Result<Value> {
    model.validate()?;
    require_scalar_options(&sample.electronic)?;
    if sample
        .electronic
        .charge
        .is_some_and(|charge| charge != sample.system.charge)
    {
        return Err(invalid(
            "sample system/electronic charge disagree; refusing a label-condition mismatch",
        ));
    }
    if sample.system.lattice.is_some() != sample.periodic.is_some() {
        return Err(invalid("property evaluation requires explicit PbcOptions for a periodic sample, and no PbcOptions for a molecule"));
    }
    let mut electronic = sample.electronic.clone();
    electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
    electronic.charge_tolerance = electronic.charge_tolerance.min(1e-10);
    electronic.ml_model = Some(Arc::new(model.clone()));
    let requested: BTreeSet<_> = sample
        .targets
        .iter()
        .filter(|t| t.weight > 0.)
        .map(|t| t.observable.specification().2)
        .collect();
    let pbc = sample.periodic;
    let mut cache = BTreeMap::new();
    for backend in requested {
        let periodic = || {
            pbc.ok_or_else(|| {
                invalid(format!(
                    "sample {} requires an explicit periodic cell/options for {backend:?}",
                    sample.id
                ))
            })
        };
        let value = match backend {
            Backend::Ground => {
                let need_force = sample
                    .targets
                    .iter()
                    .any(|t| t.weight > 0. && t.observable == PropertyObservable::Forces);
                let need_stress = sample
                    .targets
                    .iter()
                    .any(|t| t.weight > 0. && t.observable == PropertyObservable::Stress);
                if let Some(pbc) = pbc {
                    let g = crate::pbc::pbc_analytic_gradient(
                        &sample.system,
                        params,
                        &electronic,
                        &pbc,
                    )?;
                    ensure_temperature(g.scf.electronic_temperature, sample)?;
                    let mut v = json!({"energy_hartree":g.total_energy,"forces_hartree_per_bohr":g.forces.iter().map(|f|f.to_array()).collect::<Vec<_>>()});
                    if need_stress {
                        let s = crate::pbc::pbc_stress_from_scc(
                            &sample.system,
                            params,
                            g.scf,
                            &electronic,
                            &pbc,
                            &sample.system.lattice.unwrap(),
                        )?;
                        v["stress_hartree_per_bohr3"] = rows(&s.stress);
                    }
                    v
                } else if need_stress {
                    return Err(invalid("stress objective requires a periodic cell"));
                } else if need_force {
                    let g = crate::analytic_gradient(
                        &sample.system,
                        params,
                        crate::AnalyticGradientOptions {
                            electronic: electronic.clone(),
                            ..Default::default()
                        },
                    )?;
                    ensure_temperature(g.electronic_result.electronic_temperature, sample)?;
                    json!({"energy_hartree":g.total_energy,"forces_hartree_per_bohr":g.forces.iter().map(|f|f.to_array()).collect::<Vec<_>>()})
                } else {
                    let e = crate::run_electronic(&sample.system, params, electronic.clone())?;
                    ensure_temperature(e.electronic_temperature, sample)?;
                    json!({"energy_hartree":e.total_free})
                }
            }
            Backend::Hessian => {
                let h = if let Some(pbc) = pbc {
                    let h = if pbc.kmesh.is_gamma_only() {
                        crate::pbc_gamma_hessian(&sample.system, params, &electronic, &pbc)?
                    } else {
                        crate::pbc_kpoint_hessian(&sample.system, params, &electronic, &pbc)?
                    };
                    ensure_temperature(h.scf.electronic_temperature, sample)?;
                    h.hessian
                } else {
                    let h = crate::analytic_hessian(
                        &sample.system,
                        params,
                        crate::AnalyticHessianOptions {
                            electronic_options: electronic.clone(),
                            ..Default::default()
                        },
                    )?;
                    if let Some(e) = &h.electronic_result {
                        ensure_temperature(e.electronic_temperature, sample)?;
                    }
                    h.hessian
                };
                let mut v = json!({"hessian_hartree_per_bohr2":rows(&h)});
                if sample.targets.iter().any(|t| {
                    t.weight > 0.
                        && matches!(
                            t.observable,
                            PropertyObservable::PhononEigenvalues
                                | PropertyObservable::PhononWavenumbers
                        )
                }) {
                    if sample.masses_amu.len() != sample.system.atoms.len()
                        || sample.masses_amu.iter().any(|m| !m.is_finite() || *m <= 0.)
                    {
                        return Err(invalid(
                            "phonon objectives require one positive finite mass_amu per atom",
                        ));
                    }
                    let mut d = h;
                    for i in 0..d.rows() {
                        for j in 0..d.cols() {
                            d[(i, j)] /=
                                (sample.masses_amu[i / 3] * sample.masses_amu[j / 3]).sqrt();
                        }
                    }
                    let eig = symmetric_eigen(&d)?;
                    v["phonon_eigenvalues_hartree_per_bohr2_amu"] = json!(eig.values);
                    v["phonon_wavenumbers_cm1"] = json!(eig
                        .values
                        .iter()
                        .map(|&x| x.signum()
                            * (x.abs() / crate::constants::AMU_TO_ELECTRON_MASS).sqrt()
                            * crate::constants::HARTREE_TO_WAVENUMBER)
                        .collect::<Vec<_>>());
                }
                v
            }
            Backend::Elastic => {
                let pbc = periodic()?;
                let r = pbc_elastic_response(
                    &sample.system,
                    params,
                    &electronic,
                    &pbc,
                    &AnalyticElasticOptions::default(),
                )?;
                ensure_temperature(r.electronic_temperature_kelvin, sample)?;
                r.as_json()
            }
            Backend::Berry => {
                let pbc = periodic()?;
                pbc_polarization_derivatives(
                    &sample.system,
                    params,
                    &electronic,
                    &pbc,
                    &BerryPolarizationOptions::from_kmesh(pbc.kmesh),
                )?
                .as_json()
            }
            Backend::Dielectric => {
                let pbc = periodic()?;
                pbc_electronic_dielectric(
                    &sample.system,
                    params,
                    &electronic,
                    &pbc,
                    &BerryPolarizationOptions::from_kmesh(pbc.kmesh),
                )?
                .as_json()
            }
            Backend::Coupled => {
                let pbc = periodic()?;
                pbc_electromechanical_response(
                    &sample.system,
                    params,
                    &electronic,
                    &pbc,
                    &sample.masses_amu,
                    &ElectromechanicalOptions {
                        berry: BerryPolarizationOptions::from_kmesh(pbc.kmesh),
                        ..Default::default()
                    },
                )?
                .as_json()
            }
            Backend::Bands => {
                let pbc = periodic()?;
                crate::pbc::spectra::pbc_band_structure(
                    &sample.system,
                    params,
                    &electronic,
                    &pbc,
                    &sample.band_kpoints,
                )?
                .as_json()
            }
        };
        cache.insert(backend, value);
    }
    let mut prediction = serde_json::Map::new();
    for target in sample.targets.iter().filter(|t| t.weight > 0.) {
        let (_, _, backend, pointer) = target.observable.specification();
        let backend_value = &cache[&backend];
        let value=backend_value.pointer(pointer).filter(|v|!v.is_null()).ok_or_else(||invalid(format!("sample {}: {} unavailable from {backend:?} at {pointer}; backend diagnostics: {}",sample.id,target.observable.name(),backend_value.get("relaxed_response_error").or_else(||backend_value.get("ionic_dielectric_error")).unwrap_or(&Value::Null))))?;
        let mut check = Vec::new();
        selected_values(value, &target.values, target.mask.as_ref(), &mut check).map_err(|e| {
            invalid(format!(
                "sample {} target {}: {e}",
                sample.id,
                target.observable.name()
            ))
        })?;
        prediction.insert(target.observable.name().into(), value.clone());
    }
    Ok(
        json!({"sample_id":sample.id,"predictions":prediction,"conditions":sample.conditions_json(),"ml_model_sha256":model.canonical_sha256()?,"physical_derivatives":"analytic"}),
    )
}

/// Head 0/1/2 selects s/p/d levels; head 3 is the scalar correction.
/// coordinate 0 is its bias; coordinate 1.. selects radial weights.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrainableParameter {
    pub atomic_number: u8,
    pub head: usize,
    pub coordinate: usize,
}
impl TrainableParameter {
    pub fn parse(text: &str) -> Result<Self> {
        let a: Vec<_> = text.split(':').collect();
        if a.len() != 3 {
            return Err(invalid(
                "parameter must be Z:HEAD:COORDINATE; coordinate 0=bias, 1..=radial weights",
            ));
        }
        Ok(Self {
            atomic_number: a[0]
                .parse()
                .ok()
                .or_else(|| crate::symbol_to_z(a[0]))
                .ok_or_else(|| invalid("invalid trainable species"))?,
            head: match a[1] {
                "s" => 0,
                "p" => 1,
                "d" => 2,
                "correction" => 3,
                x => x.parse().map_err(|_| invalid("invalid trainable head"))?,
            },
            coordinate: a[2]
                .parse()
                .map_err(|_| invalid("invalid trainable coordinate"))?,
        })
    }
    pub fn name(&self) -> String {
        format!("{}:{}:{}", self.atomic_number, self.head, self.coordinate)
    }
    fn head<'a>(&self, m: &'a MlHamiltonian) -> Result<&'a LinearHead> {
        let h = m
            .heads
            .get(&self.atomic_number)
            .ok_or_else(|| invalid("trainable species missing from warm-start model"))?;
        match self.head {
            0..=2 => Ok(&h.levels[self.head]),
            3 => Ok(&h.correction),
            _ => Err(invalid("trainable head must be 0..3")),
        }
    }
    fn get(&self, m: &MlHamiltonian) -> Result<f64> {
        let h = self.head(m)?;
        if self.coordinate == 0 {
            Ok(h.bias)
        } else {
            h.weights
                .get(self.coordinate - 1)
                .copied()
                .ok_or_else(|| invalid("trainable radial coordinate out of range"))
        }
    }
    fn set(&self, m: &mut MlHamiltonian, value: f64) -> Result<()> {
        self.get(m)?;
        let h = m.heads.get_mut(&self.atomic_number).unwrap();
        let h = if self.head == 3 {
            &mut h.correction
        } else {
            &mut h.levels[self.head]
        };
        if self.coordinate == 0 {
            h.bias = value
        } else {
            h.weights[self.coordinate - 1] = value;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyGradientMethod {
    Auto,
    CentralParameterDifference,
}
#[derive(Clone, Debug)]
pub struct PropertyTrainingOptions {
    pub parameters: Vec<TrainableParameter>,
    pub max_epochs: usize,
    pub learning_rate: f64,
    pub parameter_difference_step: f64,
    pub maximum_parameter_step: f64,
    pub maximum_backtracks: usize,
    pub gradient_tolerance: f64,
    pub loss_tolerance: f64,
    pub regularization_to_initial: f64,
    pub gradient_method: PropertyGradientMethod,
}
impl Default for PropertyTrainingOptions {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
            max_epochs: 50,
            learning_rate: 0.1,
            parameter_difference_step: 1e-5,
            maximum_parameter_step: 0.05,
            maximum_backtracks: 16,
            gradient_tolerance: 1e-8,
            loss_tolerance: 1e-14,
            regularization_to_initial: 0.,
            gradient_method: PropertyGradientMethod::Auto,
        }
    }
}
impl PropertyTrainingOptions {
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("property optimizer settings must be an object"))?;
        let mut out = Self::default();
        for (key, v) in object {
            match key.as_str() {
                "parameters" => {
                    out.parameters =
                        v.as_array()
                            .ok_or_else(|| invalid("parameters must be a string array"))?
                            .iter()
                            .map(|v| {
                                TrainableParameter::parse(v.as_str().ok_or_else(|| {
                                    invalid("parameter selection must be a string")
                                })?)
                            })
                            .collect::<Result<Vec<_>>>()?
                }
                "max_epochs" => {
                    out.max_epochs = v
                        .as_u64()
                        .and_then(|x| usize::try_from(x).ok())
                        .ok_or_else(|| invalid("invalid max_epochs"))?
                }
                "maximum_backtracks" => {
                    out.maximum_backtracks = v
                        .as_u64()
                        .and_then(|x| usize::try_from(x).ok())
                        .ok_or_else(|| invalid("invalid maximum_backtracks"))?
                }
                "learning_rate" => out.learning_rate = number(v, key)?,
                "parameter_difference_step" => out.parameter_difference_step = number(v, key)?,
                "maximum_parameter_step" => out.maximum_parameter_step = number(v, key)?,
                "gradient_tolerance" => out.gradient_tolerance = number(v, key)?,
                "loss_tolerance" => out.loss_tolerance = number(v, key)?,
                "regularization_to_initial" => out.regularization_to_initial = number(v, key)?,
                "gradient_method" => {
                    out.gradient_method = match v.as_str() {
                        Some("auto") => PropertyGradientMethod::Auto,
                        Some("parameter-central") => {
                            PropertyGradientMethod::CentralParameterDifference
                        }
                        _ => {
                            return Err(invalid(
                                "gradient_method must be auto or parameter-central",
                            ))
                        }
                    }
                }
                _ => return Err(invalid(format!("unknown property optimizer setting {key}"))),
            }
        }
        Ok(out)
    }
}
#[derive(Clone, Debug)]
pub struct PropertyTrainingReport {
    pub json: Value,
}
impl PropertyTrainingReport {
    pub fn as_json(&self) -> Value {
        self.json.clone()
    }
}

struct Evaluation {
    loss: f64,
    report: Value,
    predictions: Vec<f64>,
}
fn evaluate_split(
    model: &MlHamiltonian,
    params: &Gfn1Parameters,
    data: &PropertyDataset,
    split: PropertySplit,
) -> Result<Evaluation> {
    let mut loss = 0.;
    let mut statistics: BTreeMap<String, (usize, f64, f64, f64)> = BTreeMap::new();
    let mut predictions = Vec::new();
    for sample in data.samples.iter().filter(|s| s.split == split) {
        let result = evaluate_property_sample(model, params, sample)?;
        for target in sample.targets.iter().filter(|t| t.weight > 0.) {
            let mut selected = Vec::new();
            selected_values(
                &result["predictions"][target.observable.name()],
                &target.values,
                target.mask.as_ref(),
                &mut selected,
            )?;
            let n = selected.len() as f64;
            let component = statistics
                .entry(target.observable.name().into())
                .or_default();
            for (prediction, reference) in selected {
                let residual = prediction - reference;
                let value = 0.5 * target.weight / n * (residual / target.scale).powi(2);
                loss += value;
                component.0 += 1;
                component.1 += residual * residual;
                component.2 = component.2.max(residual.abs());
                component.3 += value;
                predictions.push(prediction / target.scale);
            }
        }
    }
    if !loss.is_finite() {
        return Err(invalid("nonfinite scaled property loss"));
    }
    let by_property: BTreeMap<_,_>=statistics.into_iter().map(|(name,(n,square,max,weighted))|{
        let unit=PropertyObservable::parse(&name).unwrap().unit();
        (name,json!({"components":n,"rmse_native":(square/n as f64).sqrt(),"maximum_absolute_error_native":max,"native_unit":unit,"weighted_scaled_loss":weighted}))
    }).collect();
    Ok(Evaluation {
        loss,
        report: json!({"loss":loss,"by_property":by_property}),
        predictions,
    })
}

fn fast_analytic_eligible(data: &PropertyDataset) -> bool {
    data.samples
        .iter()
        .filter(|s| s.split == PropertySplit::Train)
        .all(|s| {
            s.system.lattice.is_none()
                && s.targets.iter().all(|t| {
                    t.weight == 0.
                        || (t.mask.is_none()
                            && matches!(
                                t.observable,
                                PropertyObservable::Energy | PropertyObservable::Forces
                            ))
                })
        })
}
fn analytic_weight_gradient(
    model: &MlHamiltonian,
    params: &Gfn1Parameters,
    data: &PropertyDataset,
    selected: &[TrainableParameter],
) -> Result<Vec<f64>> {
    let mut gradient = vec![0.; selected.len()];
    for sample in data
        .samples
        .iter()
        .filter(|s| s.split == PropertySplit::Train)
    {
        let mut example = crate::training::TrainingExample::new(sample.system.clone());
        let mut config = crate::training::TrainingConfig {
            energy_weight: 0.,
            gradient_weight: 0.,
            dipole_weight: 0.,
            relative_energy_weight: 0.,
            ..Default::default()
        };
        for t in sample.targets.iter().filter(|t| t.weight > 0.) {
            match t.observable {
                PropertyObservable::Energy => {
                    example.energy_hartree = Some(number(&t.values, "energy")?);
                    config.energy_weight = t.weight / t.scale.powi(2);
                }
                PropertyObservable::Forces => {
                    let a = t
                        .values
                        .as_array()
                        .ok_or_else(|| invalid("force target must be Nx3"))?;
                    let mut force = Vec::new();
                    for row in a {
                        let r = numeric_vector(row, "force")?;
                        if r.len() != 3 {
                            return Err(invalid("force target must be Nx3"));
                        }
                        force.push(crate::math::Vec3::new(-r[0], -r[1], -r[2]));
                    }
                    example.gradient_hartree_per_bohr = Some(force);
                    config.gradient_weight = t.weight / t.scale.powi(2);
                }
                _ => return Err(invalid("internal fast-gradient admission error")),
            }
        }
        if config.energy_weight + config.gradient_weight == 0. {
            continue;
        }
        let mut electronic = sample.electronic.clone();
        electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
        electronic.charge_tolerance = electronic.charge_tolerance.min(1e-10);
        let result =
            crate::training::loss_gradient(model, params, &electronic, &[example], &[], &config)?;
        for (i, p) in selected.iter().enumerate() {
            let h = &result.gradient[&p.atomic_number];
            let h = if p.head == 3 {
                &h.correction
            } else {
                &h.levels[p.head]
            };
            gradient[i] += if p.coordinate == 0 {
                h.bias
            } else {
                h.weights[p.coordinate - 1]
            };
        }
    }
    Ok(gradient)
}

/// Fine-tune selected coordinates of the supplied, actual model. The held-out
/// split is reported but never participates in gradients or line search.
pub fn fit_properties(
    model: &mut MlHamiltonian,
    params: &Gfn1Parameters,
    data: &PropertyDataset,
    settings: &PropertyTrainingOptions,
) -> Result<PropertyTrainingReport> {
    model.validate()?;
    data.validate()?;
    if settings.parameters.is_empty()
        || settings.max_epochs == 0
        || settings.maximum_backtracks == 0
        || [
            settings.learning_rate,
            settings.parameter_difference_step,
            settings.maximum_parameter_step,
            settings.gradient_tolerance,
            settings.loss_tolerance,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
        || !settings.regularization_to_initial.is_finite()
        || settings.regularization_to_initial < 0.
    {
        return Err(invalid(
            "invalid property optimizer options or no selected parameters",
        ));
    }
    let mut unique = BTreeSet::new();
    for p in &settings.parameters {
        p.get(model)?;
        if !unique.insert(p) {
            return Err(invalid("duplicate selected parameter"));
        }
    }
    if settings.parameters.iter().all(|p| p.head == 3)
        && data
            .samples
            .iter()
            .filter(|s| s.split == PropertySplit::Train)
            .flat_map(|s| &s.targets)
            .any(|t| t.weight > 0. && !t.observable.correction_sensitive())
    {
        return Err(invalid("correction-only heads cannot fit fixed-geometry Born charges, clamped proper piezoelectricity, electronic dielectric or band energies/gaps; select level-head parameters"));
    }
    let initial = model.clone();
    let initial_sha = model.canonical_sha256()?;
    let reference = settings
        .parameters
        .iter()
        .map(|p| p.get(model))
        .collect::<Result<Vec<_>>>()?;
    let use_analytic =
        settings.gradient_method == PropertyGradientMethod::Auto && fast_analytic_eligible(data);
    let method = if use_analytic {
        "analytic stationary energy/implicit CP-SCC force parameter gradient"
    } else {
        "central model-parameter difference of analytic physical-property losses"
    };
    let regularization = |m: &MlHamiltonian| -> Result<f64> {
        Ok(0.5
            * settings.regularization_to_initial
            * settings
                .parameters
                .iter()
                .zip(&reference)
                .map(|(p, r)| p.get(m).map(|v| (v - r).powi(2)))
                .collect::<Result<Vec<_>>>()?
                .iter()
                .sum::<f64>())
    };
    let mut current = evaluate_split(model, params, data, PropertySplit::Train)?;
    let initial_validation = evaluate_split(model, params, data, PropertySplit::Validation)?;
    let initial_train = current.report.clone();
    let mut epochs = Vec::new();
    let mut termination = "maximum_epochs";
    let mut insensitive = BTreeSet::new();
    for epoch in 0..settings.max_epochs {
        let before = current.loss + regularization(model)?;
        if current.loss <= settings.loss_tolerance {
            termination = "loss_tolerance";
            break;
        }
        let mut gradient = if use_analytic {
            analytic_weight_gradient(model, params, data, &settings.parameters)?
        } else {
            let mut out = Vec::new();
            for p in &settings.parameters {
                let value = p.get(model)?;
                let step = settings.parameter_difference_step * value.abs().max(1.);
                let mut plus = model.clone();
                let mut minus = model.clone();
                p.set(&mut plus, value + step)?;
                p.set(&mut minus, value - step)?;
                let plus =
                    evaluate_split(&plus, params, data, PropertySplit::Train).map_err(|e| {
                        invalid(format!(
                            "positive parameter perturbation {} failed: {e}",
                            p.name()
                        ))
                    })?;
                let minus =
                    evaluate_split(&minus, params, data, PropertySplit::Train).map_err(|e| {
                        invalid(format!(
                            "negative parameter perturbation {} failed: {e}",
                            p.name()
                        ))
                    })?;
                out.push((plus.loss - minus.loss) / (2. * step));
                let sensitivity = plus
                    .predictions
                    .iter()
                    .zip(&minus.predictions)
                    .map(|(a, b)| ((a - b) / (2. * step)).abs())
                    .fold(0., f64::max);
                if sensitivity < 1e-11 {
                    insensitive.insert(p.name());
                } else {
                    insensitive.remove(&p.name());
                }
            }
            out
        };
        for (i, p) in settings.parameters.iter().enumerate() {
            gradient[i] += settings.regularization_to_initial * (p.get(model)? - reference[i]);
        }
        if gradient.iter().any(|g| !g.is_finite()) {
            return Err(invalid("nonfinite property parameter gradient"));
        }
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm <= settings.gradient_tolerance {
            termination = if !use_analytic && insensitive.len() == settings.parameters.len() {
                "zero_selected_parameter_sensitivity"
            } else {
                "gradient_tolerance"
            };
            break;
        }
        let largest = gradient.iter().map(|g| g.abs()).fold(0., f64::max);
        let mut rate = settings
            .learning_rate
            .min(settings.maximum_parameter_step / largest);
        let mut accepted = None;
        let mut rejected = Vec::new();
        for backtrack in 0..settings.maximum_backtracks {
            let mut trial = model.clone();
            for (p, g) in settings.parameters.iter().zip(&gradient) {
                p.set(&mut trial, p.get(model)? - rate * g)?;
            }
            match evaluate_split(&trial, params, data, PropertySplit::Train) {
                Ok(result) if result.loss + regularization(&trial)? < before => {
                    accepted = Some((trial, result, rate, backtrack));
                    break;
                }
                Ok(_) => {}
                Err(e) => rejected.push(e.to_string()),
            }
            rate *= 0.5;
        }
        let Some((trial, result, rate, backtracks)) = accepted else {
            termination = "line_search_no_improvement";
            epochs.push(json!({"epoch":epoch+1,"accepted":false,"train":current.report,"rejected_trial_errors":rejected,"gradient_norm":norm}));
            break;
        };
        // Validation is evaluated before mutating the caller's model. Invalid
        // validation properties cannot yield a silently saved "trained" model.
        let held_out = evaluate_split(&trial, params, data, PropertySplit::Validation)?;
        let reg = regularization(&trial)?;
        epochs.push(json!({"epoch":epoch+1,"accepted":true,"train":result.report,"validation":held_out.report,"regularization":reg,"objective":result.loss+reg,"gradient_norm":norm,"learning_rate_used":rate,"backtracks":backtracks,"rejected_trial_errors":rejected}));
        *model = trial;
        current = result;
    }
    let final_validation = evaluate_split(model, params, data, PropertySplit::Validation)?;
    let parent_provenance = model.training_provenance.clone();
    model.training_provenance = format!(
        "property fine-tune; parent {initial_sha}; reference {}; {}",
        data.reference_method, data.provenance
    )
    .replace(['\n', '\r'], " ");
    let final_sha = model.canonical_sha256()?;
    let changes = settings
        .parameters
        .iter()
        .zip(&reference)
        .map(|(p, old)| Ok(json!({"parameter":p.name(),"initial":old,"final":p.get(model)?})))
        .collect::<Result<Vec<_>>>()?;
    Ok(PropertyTrainingReport {
        json: json!({"schema":"gfn1-rs-property-training-report-v1","package_version":env!("CARGO_PKG_VERSION"),"termination":termination,
        "initial_model_canonical_sha256":initial_sha,"final_model_canonical_sha256":final_sha,"initial_training_provenance":parent_provenance,
        "model_license_spdx":initial.license_spdx,"dataset_provenance":data.provenance,"dataset_license_spdx":data.license_spdx,"reference_method":data.reference_method,
        "independent_reference":data.independent_reference,"dataset_sha256":data.source_sha256,"parameter_sha256":sha(params.to_param_string().as_bytes()),
        "gradient_method":method,"parameter_difference_step":if use_analytic{None}else{Some(settings.parameter_difference_step)},
        "optimizer_settings":{"max_epochs":settings.max_epochs,"learning_rate":settings.learning_rate,"parameter_difference_step":settings.parameter_difference_step,
            "maximum_parameter_step":settings.maximum_parameter_step,"maximum_backtracks":settings.maximum_backtracks,
            "gradient_tolerance":settings.gradient_tolerance,"loss_tolerance":settings.loss_tolerance,"regularization_to_initial":settings.regularization_to_initial},
        "physical_derivatives":"analytic Rust backends; no finite nuclear, strain or field differences",
        "loss_definition":"sum over targets of weight/(2*Nselected) * sum(((prediction-reference)/scale)^2), plus initial-coordinate L2 regularization",
        "regularization_to_initial":settings.regularization_to_initial,"selected_parameters":changes,"insensitive_selected_parameters":insensitive,
        "validation_policy":"explicit held-out split; never used in optimizer gradients, line search or target alignment",
        "initial_train":initial_train,"initial_validation":initial_validation.report,"final_train":current.report,"final_validation":final_validation.report,
        "epochs":epochs,"conditions":data.samples.iter().map(PropertySample::conditions_json).collect::<Vec<_>>(),"observable_registry":property_registry()}),
    })
}
