// SPDX-License-Identifier: GPL-3.0-or-later
//! PyO3 bindings for the native Rust GFN1-xTB implementation.

use crate::constants::{
    ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, HARTREE_TO_EV,
    HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2,
};
use crate::electronic::{run_electronic, ElectronicOptions, SccAccelerator};
use crate::error::Gfn1Error;
use crate::field::ExternalFieldOptions;
use crate::gradient::{analytic_gradient, AnalyticGradientOptions};
use crate::hessian::AnalyticHessianOptions;
use crate::lattice::Lattice;
use crate::math::{Mat3, Vec3};
use crate::optimizer::{
    optimize_geometry, GeometryOptimizationOptions, GeometryOptimizationResult,
};
use crate::param_deriv::{
    parameter_dipole_derivatives, parameter_finite_difference, parameter_hessian_derivatives,
    ParamDerivativeOptions,
};
use crate::params::{Gfn1Parameters, ParameterTarget};
use crate::pbc::{
    pbc_electronic_result, pbc_gamma_hessian, pbc_gradient_from_scc, pbc_gruneisen,
    pbc_kpoint_hessian, pbc_stress_from_scc, run_pbc_scc_with_guess, GruneisenOptions, KMesh,
    PbcOptions, PbcSccGuess, SecondOrderStencil,
};
use crate::properties::{dipole_derivatives, ir_spectrum, raman_spectrum, static_polarizability};
use crate::system::{Atom, PeriodicSystem};
use crate::td::{
    solve_tda_gradient_method, solve_tda_kpoint, solve_tda_kpoint_gradient,
    solve_tda_kpoint_gradient_analytic, tda_optical_rotation, tda_rotatory_strengths,
    TdaGradientMethod, TdaOptions, TdaSpin,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, Mutex};

fn parse_accelerator(name: Option<&str>) -> PyResult<SccAccelerator> {
    match name.map(|s| s.to_ascii_lowercase()) {
        None => Ok(SccAccelerator::Auto),
        Some(s) => match s.as_str() {
            "auto" => Ok(SccAccelerator::Auto),
            "broyden" => Ok(SccAccelerator::Broyden),
            "linear" => Ok(SccAccelerator::Linear),
            "cdiis" => Ok(SccAccelerator::Cdiis),
            "newton" => Ok(SccAccelerator::Newton),
            other => Err(PyValueError::new_err(format!(
                "unknown scc_accelerator `{other}` (use auto/broyden/linear/cdiis/newton)"
            ))),
        },
    }
}

/// Load the GFN1-xTB-M1 secondary (dual) basis, returning `None` when nothing is given
/// (selecting the single-basis M0 variant). The argument is either a **built-in basis
/// name** (`"cc-pVDZ"`, `"cc-pVTZ"`, `"cc-pVQZ"`, `"cc-pV5Z"`; bundled, Z=1..36) or a
/// path to a secondary-basis file.
fn load_m1_basis(path: Option<&str>) -> PyResult<Option<crate::secondary_basis::SecondaryBasis>> {
    match path {
        None => Ok(None),
        Some(p) => {
            if let Some(res) = crate::secondary_bases::builtin_secondary(p) {
                return Ok(Some(res.map_err(to_py_err)?));
            }
            let text = std::fs::read_to_string(p).map_err(|e| {
                PyValueError::new_err(format!(
                    "GFN1_M1 basis `{p}` is not a built-in name (cc-pVDZ/TZ/QZ/5Z) and \
                     cannot be read as a file: {e}"
                ))
            })?;
            let basis = crate::secondary_basis::parse_secondary_basis(&text).map_err(to_py_err)?;
            Ok(Some(basis))
        }
    }
}

/// Native GFN1-xTB calculator (single point, gradient, Hessian; non-periodic + PBC Γ/k).
///
/// Beyond stock GFN1 it exposes two **independent** experimental electrostatics "orders" (both
/// parameter-free, both default to the stock value, both enter the SCC self-consistently):
///
///   • ``charge_order`` — highest order of the *isotropic* on-site charge (monopole Δq) expansion.
///       3 = stock GFN1 (2nd-order Klopman–Ohno + 3rd-order DFTB3); n≥4 adds the Linear
///       Breathing-Radius terms ``Σ_A (1/k) X_k Δq_A^k`` for 4≤k≤n. **Use ``charge_order=4`` to
///       stabilise long-range exchange on small-gap/metallic systems** (the convex quartic bounds
///       the otherwise-unbounded cubic). No ``multipole`` flag needed.
///
///   • ``multipole_order`` — highest *angular* atomic multipole rank (needs ``multipole=True``).
///       0 (default) = the legacy rank-1/2 (dipole+quadrupole) path; add ``multipole_octupole=True``
///       for rank 3; set ``multipole_order=n`` (n≥4) for the unified arbitrary-rank path (rank 1..n).
///
/// Long-range Fock exchange (LC-DFTB style, parameter-free): ``lr_exchange=True`` adds the
/// Mulliken-approximated long-range exact exchange ``K[ΔP]``; ``onsite_exchange=True`` (requires
/// ``lr_exchange``) upgrades the same-atom part to exact one-center integrals (OFX). On small-gap
/// systems pair these with ``charge_order=4``; the SCF then runs the robust ADIIS→C-DIIS pipeline
/// with a second-order TRAH polish automatically (or force it with ``scf_trah=True``).
///
/// All of the above are off / at the stock value by default, so the default constructor reproduces
/// plain GFN1-xTB. See the Rust ``ElectronicOptions`` docs for the full per-field reference.
#[pyclass(name = "Gfn1NativeCalculator")]
#[derive(Clone)]
pub struct PyGfn1NativeCalculator {
    params: Gfn1Parameters,
    electronic: ElectronicOptions,
    d3_reference_path: Option<String>,
    /// When set (and `multipole_order ≥ 4`, non-periodic), the multipole SCC is run by the
    /// **rank-continuation ladder** from this base rank up to `multipole_order`, warm-starting
    /// each rank from the previous — robust for high (16-pole+) multipole SCC. `None` = direct.
    rank_ladder_base: Option<usize>,
    pbc_guess: Arc<Mutex<Option<PbcSccGuess>>>,
    ml_identity: Option<crate::ml::MlModelIdentity>,
}

#[pymethods]
impl PyGfn1NativeCalculator {
    #[new]
    #[pyo3(signature = (
        param_path = None,
        charge = 0.0,
        multiplicity = None,
        spin_polarization = false,
        max_scc = 250,
        energy_tolerance = 1.0e-6,
        charge_tolerance = 2.0e-5,
        mixing = 0.4,
        scc_broyden = true,
        scc_broyden_size = 250,
        electronic_temperature = 300.0,
        nprim = 0,
        eigen_tolerance = 1.0e-12,
        enable_dispersion = true,
        d3_reference_path = None,
        experimental_d4 = false,
        d4_cutoff = None,
        d4_cn_cutoff = None,
        d4_atm = true,
        d4_atm_cutoff = None,
        d4_s9 = None,
        enable_cn_hamiltonian = true,
        electric_field = None,
        level_shift = 0.0,
        scc_accelerator = None,
        multipole = false,
        multipole_octupole = false,
        field_multipole = false,
        multipole_third_order = false,
        multipole_secondary_basis = None,
        multipole_order = 0,
        multipole_charge_order = vec![],
        lr_exchange = false,
        onsite_exchange = false,
        dynamic_omega = false,
        scf_trah = false,
        charge_order = 3,
        multipole_rank_ladder_base = None,
        multipole_model = None,
        camm_damp = 1.0,
        camm_aes_scale = 1.0,
        camm_onsite_scale = 1.0,
        camm_preset = None,
        plus_u = false,
        hubbard_u = vec![],
        plus_u_v = false,
        hubbard_v = vec![],
        hubbard_v_cutoff = 10.0,
        hubbard_u_linear_response = false,
        plus_u_all_d = false,
        solvation_model = None,
        solvent = None,
        solvent_state = "gsolv",
        solvent_temperature = 298.15,
        solvent_grid = 230,
        ml_model_path = None,
        log_file = None,
        log_level = "info",
        log_console = false,
        pbc_ao_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        param_path: Option<String>,
        charge: f64,
        multiplicity: Option<usize>,
        spin_polarization: bool,
        max_scc: usize,
        energy_tolerance: f64,
        charge_tolerance: f64,
        mixing: f64,
        scc_broyden: bool,
        scc_broyden_size: usize,
        electronic_temperature: f64,
        nprim: usize,
        eigen_tolerance: f64,
        enable_dispersion: bool,
        d3_reference_path: Option<String>,
        experimental_d4: bool,
        d4_cutoff: Option<f64>,
        d4_cn_cutoff: Option<f64>,
        d4_atm: bool,
        d4_atm_cutoff: Option<f64>,
        d4_s9: Option<f64>,
        enable_cn_hamiltonian: bool,
        electric_field: Option<(f64, f64, f64)>,
        level_shift: f64,
        scc_accelerator: Option<String>,
        multipole: bool,
        multipole_octupole: bool,
        field_multipole: bool,
        multipole_third_order: bool,
        multipole_secondary_basis: Option<String>,
        multipole_order: usize,
        multipole_charge_order: Vec<usize>,
        lr_exchange: bool,
        onsite_exchange: bool,
        dynamic_omega: bool,
        scf_trah: bool,
        charge_order: usize,
        multipole_rank_ladder_base: Option<usize>,
        multipole_model: Option<String>,
        camm_damp: f64,
        camm_aes_scale: f64,
        camm_onsite_scale: f64,
        camm_preset: Option<String>,
        plus_u: bool,
        hubbard_u: Vec<(u8, f64)>,
        plus_u_v: bool,
        hubbard_v: Vec<(u8, u8, f64)>,
        hubbard_v_cutoff: f64,
        hubbard_u_linear_response: bool,
        plus_u_all_d: bool,
        solvation_model: Option<String>,
        solvent: Option<String>,
        solvent_state: &str,
        solvent_temperature: f64,
        solvent_grid: usize,
        ml_model_path: Option<String>,
        log_file: Option<String>,
        log_level: &str,
        log_console: bool,
        pbc_ao_cutoff: Option<f64>,
    ) -> PyResult<Self> {
        // Resolution order: explicit param_path (a file path or the builtin
        // specs "builtin" / "builtin:si") > GFN1_XTB_PARAM > bundled builtin.
        let params = Gfn1Parameters::resolve(param_path.as_deref()).map_err(to_py_err)?;
        let mut electronic = ElectronicOptions::default();
        if pbc_ao_cutoff.is_some_and(|v| !v.is_finite() || v <= 0.) {
            return Err(PyValueError::new_err(
                "pbc_ao_cutoff must be positive and finite (bohr)",
            ));
        }
        electronic.pbc_ao_cutoff = pbc_ao_cutoff;
        if let Some(path) = log_file {
            electronic.logger = Some(
                crate::CalculationLogger::append(
                    path,
                    crate::LogLevel::parse(log_level).map_err(to_py_err)?,
                    log_console,
                )
                .map_err(to_py_err)?,
            );
        }
        if let Some(model) = solvation_model.as_deref() {
            if model != "none" {
                let kind = match model {
                    "alpb" => crate::SolvationModel::Alpb,
                    "gbsa" => crate::SolvationModel::Gbsa,
                    _ => {
                        return Err(PyValueError::new_err(
                            "solvation_model must be none, alpb or gbsa",
                        ))
                    }
                };
                let name = solvent
                    .ok_or_else(|| PyValueError::new_err("solvent is required with ALPB/GBSA"))?;
                let mut s = crate::SolvationOptions::new(kind, name);
                s.state = match solvent_state {
                    "gsolv" => crate::SolutionState::Gsolv,
                    "bar1mol" => crate::SolutionState::Bar1Mol,
                    "reference" => crate::SolutionState::Reference,
                    _ => return Err(PyValueError::new_err("invalid solvent_state")),
                };
                s.temperature = solvent_temperature;
                s.grid_points = solvent_grid;
                electronic.solvation = Some(s);
            } else if solvent.is_some() {
                return Err(PyValueError::new_err(
                    "solvent requires an active solvation_model",
                ));
            }
        } else if solvent.is_some() {
            return Err(PyValueError::new_err("solvent requires solvation_model"));
        }
        let mut ml_identity = None;
        if let Some(path) = ml_model_path {
            let (model, identity) =
                crate::ml::MlHamiltonian::resolve_with_identity(&path).map_err(to_py_err)?;
            if let Some(logger) = &electronic.logger {
                logger
                    .event(crate::LogLevel::Info, "ml_model_loaded", identity.as_json())
                    .map_err(to_py_err)?;
            } else if log_console {
                eprintln!("[gfn1-rs ML model] {}", identity.description());
            }
            electronic.ml_model = Some(Arc::new(model));
            ml_identity = Some(identity);
        }
        electronic.charge = Some(charge);
        electronic.spin_multiplicity = multiplicity;
        electronic.spin_polarization = spin_polarization;
        electronic.plus_u = plus_u;
        electronic.hubbard_u = hubbard_u;
        electronic.plus_u_v = plus_u_v;
        electronic.hubbard_v = hubbard_v;
        electronic.hubbard_v_cutoff = hubbard_v_cutoff;
        electronic.hubbard_u_linear_response = hubbard_u_linear_response;
        electronic.plus_u_all_d = plus_u_all_d;
        electronic.max_scc = max_scc;
        electronic.energy_tolerance = energy_tolerance;
        electronic.charge_tolerance = charge_tolerance;
        electronic.mixing = mixing;
        electronic.scc_broyden = scc_broyden;
        electronic.scc_broyden_size = scc_broyden_size;
        electronic.electronic_temperature = electronic_temperature;
        if nprim > 0 {
            electronic.nprim = nprim;
        }
        electronic.eigen_tolerance = eigen_tolerance;
        electronic.enable_dispersion = enable_dispersion;
        electronic.d3_reference_path = d3_reference_path.clone();
        electronic.experimental_d4 = experimental_d4;
        if let Some(value) = d4_cutoff {
            electronic.d4_cutoff = value;
        }
        if let Some(value) = d4_cn_cutoff {
            electronic.d4_cn_cutoff = value;
        }
        electronic.d4_atm = d4_atm;
        if let Some(value) = d4_atm_cutoff {
            electronic.d4_atm_cutoff = value;
        }
        electronic.d4_s9 = d4_s9;
        electronic.hamiltonian.enable_cn_hamiltonian = enable_cn_hamiltonian;
        electronic.level_shift = level_shift;
        electronic.multipole = multipole;
        electronic.multipole_octupole = multipole_octupole;
        electronic.field_multipole = field_multipole;
        electronic.multipole_third_order = multipole_third_order;
        electronic.multipole_secondary_basis = load_m1_basis(multipole_secondary_basis.as_deref())?;
        electronic.multipole_order = multipole_order;
        electronic.multipole_charge_order = multipole_charge_order;
        electronic.multipole_model = match multipole_model.as_deref() {
            None | Some("mdftb2") => crate::electronic::MultipoleModel::Mdftb2,
            Some("camm_on_mdftb2") | Some("camm") => {
                crate::electronic::MultipoleModel::CammOnMdftb2
            }
            Some(other) => {
                return Err(to_py_err(crate::error::Gfn1Error::InvalidInput(format!(
                    "multipole_model `{other}` (want mdftb2 | camm_on_mdftb2)"
                ))))
            }
        };
        electronic.camm_damp = camm_damp;
        electronic.camm_aes_scale = camm_aes_scale;
        electronic.camm_onsite_scale = camm_onsite_scale;
        // A named CAMM preset fills the per-element κ + s_onsite (the only way to reach the
        // element-specific κ from Python); explicit non-default kwargs still win.
        if let Some(name) = camm_preset.as_deref() {
            let (gk, elems, aes, onsite, onsite_elem) =
                crate::electronic::camm_preset(name).ok_or_else(|| {
                    to_py_err(crate::error::Gfn1Error::InvalidInput(format!(
                        "unknown camm_preset `{name}` (valid: polar | halogen | halogen-v1 | halogen-allgrad | sigma-hole)"
                    )))
                })?;
            electronic.multipole = true;
            electronic.multipole_model = crate::electronic::MultipoleModel::CammOnMdftb2;
            if camm_damp == 1.0 {
                electronic.camm_damp = gk;
            }
            if camm_aes_scale == 1.0 {
                electronic.camm_aes_scale = aes;
            }
            if camm_onsite_scale == 1.0 {
                electronic.camm_onsite_scale = onsite;
            }
            electronic.camm_damp_elem = elems;
            electronic.camm_onsite_scale_elem = onsite_elem;
        }
        electronic.lr_exchange = lr_exchange;
        electronic.onsite_exchange = onsite_exchange;
        electronic.dynamic_omega = dynamic_omega;
        electronic.scf_trah = scf_trah;
        electronic.charge_order = charge_order;
        electronic.scc_accelerator = parse_accelerator(scc_accelerator.as_deref())?;
        if let Some((ex, ey, ez)) = electric_field {
            electronic.external_field = ExternalFieldOptions::electric(Vec3::new(ex, ey, ez));
        }
        Ok(Self {
            params,
            electronic,
            d3_reference_path,
            rank_ladder_base: multipole_rank_ladder_base,
            pbc_guess: Arc::new(Mutex::new(None)),
            ml_identity,
        })
    }

    /// One-line description of the loaded parametrization and where it came
    /// from, e.g. ``"GFN1-xTB (builtin, grimme-lab/xtb@2b5cd48, LGPL-3.0-or-later)"``
    /// or ``"GFN1-xTB (C:\\path\\param_gfn1-xtb.txt (via GFN1_XTB_PARAM))"``.
    fn param_source(&self) -> String {
        self.params.source_description()
    }

    /// Exact source and canonical-weight hashes, or None when ML is inactive.
    fn ml_model_source(&self) -> Option<String> {
        self.ml_identity.as_ref().map(|m| m.description())
    }
    fn ml_model_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(info) = &self.ml_identity else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("source", &info.source)?;
        dict.set_item("source_sha256", &info.source_sha256)?;
        dict.set_item("canonical_sha256", &info.canonical_sha256)?;
        dict.set_item("training_provenance", &info.training_provenance)?;
        dict.set_item("license_spdx", &info.license_spdx)?;
        dict.set_item("architecture", "gfn1-rs-radial-hamiltonian-v1")?;
        Ok(Some(dict.into()))
    }

    /// Fine-tune the selected user model against explicitly conditioned property
    /// labels. The saved model is returned by identity; this calculator retains
    /// its original inference model until the caller loads the new file.
    #[pyo3(signature=(dataset,settings,output_model="trained.model"))]
    fn fine_tune_properties(
        &self,
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        settings: &Bound<'_, PyAny>,
        output_model: &str,
    ) -> PyResult<Py<PyAny>> {
        let mut model = self
            .electronic
            .ml_model
            .as_ref()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "fine_tune_properties requires an explicitly loaded ML model (ml_model_path)",
                )
            })?
            .as_ref()
            .clone();
        let json = py.import("json")?;
        let data_text: String = json.call_method1("dumps", (dataset,))?.extract()?;
        let settings_text: String = json.call_method1("dumps", (settings,))?.extract()?;
        let value =
            serde_json::from_str(&data_text).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut data =
            crate::property_training::PropertyDataset::from_json(&value).map_err(to_py_err)?;
        let settings_value = serde_json::from_str(&settings_text)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let settings =
            crate::property_training::PropertyTrainingOptions::from_json(&settings_value)
                .map_err(to_py_err)?;
        for sample in &mut data.samples {
            sample.electronic.logger = self.electronic.logger.clone();
        }
        if let Some(identity) = &self.ml_identity {
            eprintln!(
                "[gfn1-rs property training warm-start] {}",
                identity.description()
            );
        }
        let mut report = py
            .allow_threads(|| {
                crate::property_training::fit_properties(&mut model, &self.params, &data, &settings)
            })
            .map_err(to_py_err)?
            .as_json();
        let text = model.to_model_string().map_err(to_py_err)?;
        std::fs::write(output_model, text).map_err(|e| {
            PyValueError::new_err(format!("cannot save trained model {output_model}: {e}"))
        })?;
        let (_, identity) =
            crate::ml::MlHamiltonian::resolve_with_identity(output_model).map_err(to_py_err)?;
        report["initial_model_identity"] = self
            .ml_identity
            .as_ref()
            .map(|m| m.as_json())
            .unwrap_or(serde_json::Value::Null);
        report["saved_model_path"] = serde_json::json!(output_model);
        report["saved_model_identity"] = identity.as_json();
        report["saved_model_file_sha256"] = serde_json::json!(identity.source_sha256);
        if let Some(logger) = &self.electronic.logger {
            logger
                .event(crate::LogLevel::Info, "property_training", report.clone())
                .map_err(to_py_err)?;
        }
        eprintln!(
            "[gfn1-rs property training saved] {}",
            identity.description()
        );
        json_value_to_py(py, report)
    }

    fn property_training_observables(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_value_to_py(py, crate::property_training::property_registry())
    }

    fn __repr__(&self) -> String {
        format!(
            "Gfn1NativeCalculator(parameters: {})",
            self.params.source_description()
        )
    }

    #[pyo3(signature = (numbers, positions, unit = "angstrom", compute_gradient = false))]
    fn calculate(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        compute_gradient: bool,
    ) -> PyResult<PyCalculationResult> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        if compute_gradient {
            let gradient = analytic_gradient(&system, &self.params, self.gradient_options())
                .map_err(to_py_err)?;
            Ok(PyCalculationResult::from_gradient(gradient))
        } else {
            // Rank-continuation ladder for a robust high-rank multipole SCC (non-periodic).
            let result = match self.rank_ladder_base {
                Some(base) if self.electronic.multipole && self.electronic.multipole_order >= 4 => {
                    crate::run_electronic_rank_ladder(
                        &system,
                        &self.params,
                        &self.electronic,
                        base,
                        self.electronic.multipole_order,
                    )
                    .map_err(to_py_err)?
                }
                _ => crate::run_electronic(&system, &self.params, self.electronic.clone())
                    .map_err(to_py_err)?,
            };
            Ok(PyCalculationResult::from_electronic(result))
        }
    }

    /// Periodic single point. `cell` is 3 lattice vectors (rows a, b, c) in the
    /// chosen unit; `pbc` is the per-axis periodicity; `kgrid` is the
    /// Monkhorst-Pack mesh (`None` or [1,1,1] -> Gamma). Returns energy, forces
    /// (when requested), charges, dipole, and (optionally) the stress tensor.
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), kgrid = None,
        unit = "angstrom", compute_gradient = false, compute_stress = false, gamma_centered = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn calculate_periodic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        unit: &str,
        compute_gradient: bool,
        compute_stress: bool,
        gamma_centered: bool,
    ) -> PyResult<PyCalculationResult> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let kmesh = match kgrid {
            None | Some((1, 1, 1)) => KMesh::gamma(),
            Some((a, b, c)) => KMesh::monkhorst_pack([a, b, c]),
        };
        let mut pbc_options = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        pbc_options.kmesh = kmesh;
        pbc_options.kmesh.gamma_centered = gamma_centered;
        let options = self.electronic.clone();

        let guess = self.pbc_guess.lock().ok().and_then(|guard| guard.clone());
        let scf = run_pbc_scc_with_guess(
            &system,
            &self.params,
            &options,
            &pbc_options,
            guess.as_ref(),
        )
        .map_err(to_py_err)?;

        let (electronic, forces, stress, next_guess) = if compute_stress {
            let lattice = system.lattice.as_ref().copied().ok_or_else(|| {
                to_py_err(Gfn1Error::InvalidInput(
                    "stress requires a lattice".to_string(),
                ))
            })?;
            let st =
                pbc_stress_from_scc(&system, &self.params, scf, &options, &pbc_options, &lattice)
                    .map_err(to_py_err)?;
            let stress_rows: Vec<Vec<f64>> = (0..3)
                .map(|i| (0..3).map(|j| st.stress[(i, j)]).collect())
                .collect();
            let gr = pbc_gradient_from_scc(&system, &self.params, st.scf, &options, &pbc_options)
                .map_err(to_py_err)?;
            let next_guess = PbcSccGuess::from(&gr.scf);
            let forces = gr.forces;
            let er =
                pbc_electronic_result(gr.scf, &system, pbc_options.ao_cutoff).map_err(to_py_err)?;
            (er, Some(forces), Some(stress_rows), next_guess)
        } else if compute_gradient {
            let gr = pbc_gradient_from_scc(&system, &self.params, scf, &options, &pbc_options)
                .map_err(to_py_err)?;
            let next_guess = PbcSccGuess::from(&gr.scf);
            let forces = gr.forces;
            let er =
                pbc_electronic_result(gr.scf, &system, pbc_options.ao_cutoff).map_err(to_py_err)?;
            (er, Some(forces), None, next_guess)
        } else {
            let next_guess = PbcSccGuess::from(&scf);
            let er =
                pbc_electronic_result(scf, &system, pbc_options.ao_cutoff).map_err(to_py_err)?;
            (er, None, None, next_guess)
        };
        if let Ok(mut guard) = self.pbc_guess.lock() {
            *guard = Some(next_guess);
        }

        let terms = energy_terms(&electronic);
        let forces_ev = forces.map(|f| {
            f.iter()
                .map(|v| {
                    vec![
                        v.x * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                        v.y * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                        v.z * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                    ]
                })
                .collect::<Vec<_>>()
        });
        Ok(PyCalculationResult {
            calculation_id: electronic.calculation_id.clone(),
            solvation_options: electronic.solvation.as_ref().map(|s| s.options().clone()),
            ml_model_sha256: electronic
                .ml_parameters
                .as_ref()
                .map(|m| m.model_sha256.clone()),
            method_citations: electronic
                .method_citations
                .iter()
                .map(|s| s.to_string())
                .collect(),
            energy_hartree: electronic.total_free,
            energy_ev: electronic.total_free * HARTREE_TO_EV,
            gradient_hartree_per_bohr: None,
            forces_ev_per_angstrom: forces_ev,
            charges: electronic.atomic_charges,
            dipole: (
                electronic.dipole.x,
                electronic.dipole.y,
                electronic.dipole.z,
            ),
            stress,
            iterations: electronic.iterations,
            converged: electronic.converged,
            terms,
        })
    }

    /// Rust-native L-BFGS geometry optimization. Pass `cell` (3 lattice vectors, rows a/b/c, in the
    /// chosen unit) to run a **fixed-cell periodic (Γ-point) optimization** — the atomic positions
    /// relax while the lattice is held fixed (the gradient routes through the PBC path); `pbc` is the
    /// per-axis periodicity. With `cell=None` it is the non-periodic (molecular) optimizer (unchanged).
    #[pyo3(signature = (
        numbers,
        positions,
        unit = "angstrom",
        cell = None,
        pbc = (true, true, true),
        max_iterations = 250,
        gradient_tolerance = 1.0e-4,
        step_tolerance = 1.0e-7,
        history = 12,
        max_atom_step = 0.30,
        trajectory_path = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn optimize(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        cell: Option<Vec<Vec<f64>>>,
        pbc: (bool, bool, bool),
        max_iterations: usize,
        gradient_tolerance: f64,
        step_tolerance: f64,
        history: usize,
        max_atom_step: f64,
        trajectory_path: Option<String>,
    ) -> PyResult<PyOptimizationResult> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let mut gradient_options = self.gradient_options();
        let system = match cell {
            Some(cell) => {
                // Fixed-cell periodic (Γ-point) optimization: mark the boundary periodic so the
                // L-BFGS gradient routes through the PBC path; the lattice is preserved across steps.
                gradient_options.electronic.boundary =
                    crate::model::BoundaryCondition::GammaPointPbc;
                build_periodic_system(numbers, positions, cell, pbc, unit, charge)?
            }
            None => build_system(numbers, positions, unit, charge)?,
        };
        let options = GeometryOptimizationOptions {
            max_iterations,
            gradient_tolerance,
            step_tolerance,
            history,
            max_atom_step,
            gradient_options,
            // When set, the L-BFGS trajectory is streamed live to this XYZ file.
            trajectory_path: trajectory_path.map(std::path::PathBuf::from),
            ..GeometryOptimizationOptions::default()
        };
        let result = optimize_geometry(&system, &self.params, options).map_err(to_py_err)?;
        Ok(PyOptimizationResult::from_result(result))
    }

    /// Static dipole polarizability (analytic CPXTB field response). Returns a dict
    /// with `tensor` (3x3, a.u.), `isotropic`, and `anisotropy`.
    #[pyo3(signature = (numbers, positions, unit = "angstrom"))]
    fn polarizability(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let electronic =
            run_electronic(&system, &self.params, self.electronic.clone()).map_err(to_py_err)?;
        let pol = static_polarizability(&system, &self.params, &electronic).map_err(to_py_err)?;
        let dict = PyDict::new(py);
        let tensor: Vec<Vec<f64>> = pol.tensor.iter().map(|r| r.to_vec()).collect();
        dict.set_item("tensor", tensor)?;
        dict.set_item("isotropic", pol.isotropic)?;
        dict.set_item("anisotropy", pol.anisotropy)?;
        Ok(dict.into())
    }

    /// Analytic Cartesian dipole derivatives `dmu/dR` (the raw IR tensor). Returns
    /// a dict with `dipole` (a.u.) and `ddipole_dr` (`[coord][alpha]`).
    #[pyo3(signature = (numbers, positions, unit = "angstrom", origin = None))]
    fn dipole_derivatives(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let electronic =
            run_electronic(&system, &self.params, self.electronic.clone()).map_err(to_py_err)?;
        let dd = dipole_derivatives(&system, &self.params, &electronic, origin_vec(origin))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("dipole", vec![dd.dipole.x, dd.dipole.y, dd.dipole.z])?;
        let ddr: Vec<Vec<f64>> = dd.ddipole_dr.iter().map(|r| r.to_vec()).collect();
        dict.set_item("ddipole_dr", ddr)?;
        Ok(dict.into())
    }

    /// Harmonic IR spectrum. Returns a dict of lists: `wavenumbers` (cm^-1),
    /// `intensities_km_per_mol`, `intensities_au`, and `dipole_gradients`.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", origin = None))]
    fn ir_spectrum(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let ir = ir_spectrum(
            &system,
            &self.params,
            self.hessian_options(),
            origin_vec(origin),
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item(
            "wavenumbers",
            ir.modes.iter().map(|m| m.wavenumber).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "intensities_km_per_mol",
            ir.modes
                .iter()
                .map(|m| m.intensity_km_per_mol)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "intensities_au",
            ir.modes.iter().map(|m| m.intensity_au).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "dipole_gradients",
            ir.modes
                .iter()
                .map(|m| m.dipole_gradient.to_vec())
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// Harmonic Raman spectrum. Returns a dict of lists: `wavenumbers` (cm^-1),
    /// `activities`, `mean_polarizability_derivative`, `anisotropy_squared`.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", origin = None, field_step = 1.0e-3))]
    fn raman_spectrum(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        origin: Option<(f64, f64, f64)>,
        field_step: f64,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let raman = raman_spectrum(
            &system,
            &self.params,
            self.hessian_options(),
            origin_vec(origin),
            field_step,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item(
            "wavenumbers",
            raman.modes.iter().map(|m| m.wavenumber).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "activities",
            raman.modes.iter().map(|m| m.activity).collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "mean_polarizability_derivative",
            raman
                .modes
                .iter()
                .map(|m| m.mean_polarizability_derivative)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "anisotropy_squared",
            raman
                .modes
                .iter()
                .map(|m| m.anisotropy_squared)
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// Closed-shell magnetic (GFN1-xTB-M0) SCC total energy (Hartree) for a
    /// uniform magnetic field `b_field` (atomic units). Non-periodic; reduces to
    /// the field-free energy at B = 0. The kinetic-energy correction (M1) is not
    /// included.
    /// Electronic-CD **rotatory strengths** of the TD-GFN1 (TDA) excited states,
    /// `R_n = Im(mu_0n . m_n0)` (atomic units; `m = -1/2 (r - O) x p` about `origin`,
    /// default the coordinate origin). Returns a dict with `excitation_energies_ev` /
    /// `_hartree` and `rotatory_strengths`. Non-periodic; every value vanishes for an
    /// achiral molecule.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", n_states = 5, spin = "singlet", origin = None))]
    fn rotatory_strengths(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        n_states: usize,
        spin: &str,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let electronic =
            run_electronic(&system, &self.params, self.electronic.clone()).map_err(to_py_err)?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let o = origin
            .map(|(x, y, z)| Vec3::new(x, y, z))
            .unwrap_or(Vec3::zero());
        let states = tda_rotatory_strengths(
            &system,
            &self.params,
            &electronic,
            TdaOptions { n_states, spin },
            o,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item(
            "excitation_energies_hartree",
            states
                .iter()
                .map(|s| s.excitation_energy)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "excitation_energies_ev",
            states
                .iter()
                .map(|s| s.excitation_energy * HARTREE_TO_EV)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "rotatory_strengths",
            states
                .iter()
                .map(|s| s.rotatory_strength)
                .collect::<Vec<_>>(),
        )?;
        // Raw magnetic transition dipoles m_n0 = i * h_n (stored as h_n = Im part) per
        // state, as `[x, y, z]` rows — the raw vectors behind R_n = mu_0n . m_n0.
        dict.set_item(
            "magnetic_transition_dipoles",
            states
                .iter()
                .map(|s| {
                    let h = s.magnetic_transition_dipole;
                    vec![h.x, h.y, h.z]
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// Frequency-dependent electronic **optical rotation** (isotropic Rosenfeld `beta`,
    /// atomic units) from the TD-GFN1 (TDA) sum over states:
    /// `beta(w) = (2/3) sum_n R_n w_n/(w_n^2 - w^2)`. `frequencies_ev` are photon
    /// energies in eV (`0.0` = static; for a wavelength use `E[eV] = 1239.84 / lambda[nm]`).
    /// Returns a dict with `frequencies_ev` and the corresponding `beta`. Achiral
    /// molecules give 0; the enantiomer negates `beta`. Frequencies near an excitation
    /// are undamped poles. Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", n_states = 10, spin = "singlet", frequencies_ev = vec![0.0], origin = None))]
    #[allow(clippy::too_many_arguments)]
    fn optical_rotation(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        n_states: usize,
        spin: &str,
        frequencies_ev: Vec<f64>,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let electronic =
            run_electronic(&system, &self.params, self.electronic.clone()).map_err(to_py_err)?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let o = origin
            .map(|(x, y, z)| Vec3::new(x, y, z))
            .unwrap_or(Vec3::zero());
        let freqs_ha: Vec<f64> = frequencies_ev.iter().map(|e| e / HARTREE_TO_EV).collect();
        let beta = tda_optical_rotation(
            &system,
            &self.params,
            &electronic,
            TdaOptions { n_states, spin },
            o,
            &freqs_ha,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("frequencies_ev", frequencies_ev)?;
        dict.set_item("beta", beta)?;
        Ok(dict.into())
    }

    /// Embedded FMO2/FMO3 with optional covalent HOP cuts and analytic response
    /// gradients. `cell` selects the coherent Gamma-periodic extension; stress
    /// is dF/dstrain divided by volume. Atom and cut indices are zero-based.
    #[pyo3(signature=(numbers, positions, fragments, fragment_charges=None, cuts=None, cut_images=None, order=2, cell=None, unit="angstrom", compute_derivatives=true, embedding_tolerance=1e-9, max_embedding_iterations=200, hop_penalty=1e6, max_ao=512))]
    #[allow(clippy::too_many_arguments)]
    fn fmo_xtb(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        fragments: Vec<Vec<usize>>,
        fragment_charges: Option<Vec<f64>>,
        cuts: Option<Vec<(usize, usize)>>,
        cut_images: Option<Vec<(i32, i32, i32)>>,
        order: usize,
        cell: Option<Vec<Vec<f64>>>,
        unit: &str,
        compute_derivatives: bool,
        embedding_tolerance: f64,
        max_embedding_iterations: usize,
        hop_penalty: f64,
        max_ao: usize,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.);
        let system = if let Some(cell) = cell {
            build_periodic_system(numbers, positions, cell, (true, true, true), unit, charge)?
        } else {
            build_system(numbers, positions, unit, charge)?
        };
        let cuts = cuts.unwrap_or_default();
        let images = cut_images.unwrap_or_else(|| vec![(0, 0, 0); cuts.len()]);
        if cuts.len() != images.len() {
            return Err(PyValueError::new_err(
                "cuts and cut_images must have equal lengths",
            ));
        }
        let charges = fragment_charges.unwrap_or_else(|| vec![0.; fragments.len()]);
        let partition = crate::fragment::FragmentPartition {
            atoms: fragments,
            charges,
            cuts: cuts
                .into_iter()
                .zip(images)
                .map(|((bda, baa), (a, b, c))| crate::fragment::CovalentCut {
                    bda,
                    baa,
                    image: [a, b, c],
                })
                .collect(),
        };
        let options = crate::fragment::FmoOptions {
            order,
            compute_derivatives,
            embedding_tolerance,
            max_embedding_iterations,
            hop_penalty,
            max_ao,
            ..Default::default()
        };
        let pbc = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        let result = py
            .allow_threads(|| {
                crate::fragment::fmo_xtb(
                    &system,
                    &self.params,
                    &self.electronic,
                    &pbc,
                    &partition,
                    &options,
                )
            })
            .map_err(to_py_err)?;
        fragment_result_dict(
            py,
            &result,
            &format!("FMO{order}-GFN1-xTB"),
            compute_derivatives,
            self.electronic.electronic_temperature,
        )
    }

    /// Overlapping density-matrix divide-and-conquer with a common chemical
    /// potential and analytic derivatives of its finite-buffer free energy.
    /// Frozen buffers contain complete atom indices, including their core.
    #[pyo3(signature=(numbers, positions, fragments, buffers=None, fragment_charges=None, cell=None, unit="angstrom", compute_derivatives=true, max_ao=512))]
    #[allow(clippy::too_many_arguments)]
    fn divide_and_conquer(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        fragments: Vec<Vec<usize>>,
        buffers: Option<Vec<Vec<usize>>>,
        fragment_charges: Option<Vec<f64>>,
        cell: Option<Vec<Vec<f64>>>,
        unit: &str,
        compute_derivatives: bool,
        max_ao: usize,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.);
        let system = if let Some(cell) = cell {
            build_periodic_system(numbers, positions, cell, (true, true, true), unit, charge)?
        } else {
            build_system(numbers, positions, unit, charge)?
        };
        let charges = fragment_charges.unwrap_or_else(|| vec![0.; fragments.len()]);
        let partition = crate::fragment::FragmentPartition {
            atoms: fragments,
            charges,
            cuts: vec![],
        };
        let options = crate::fragment::DivideConquerOptions {
            buffers: buffers.unwrap_or_default(),
            compute_derivatives,
            max_ao,
        };
        let pbc = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        let result = py
            .allow_threads(|| {
                crate::fragment::divide_and_conquer(
                    &system,
                    &self.params,
                    &self.electronic,
                    &pbc,
                    &partition,
                    &options,
                )
            })
            .map_err(to_py_err)?;
        fragment_result_dict(
            py,
            &result,
            "DC-GFN1-xTB",
            compute_derivatives,
            self.electronic.electronic_temperature,
        )
    }

    /// Native Rust sTDA-xTB with dedicated VTB/xTB orbitals, distinct from TD-GFN1.
    /// Fitted parameters are from the pinned official xtb4stda repository.
    #[pyo3(signature=(numbers,positions,n_states=10,energy_window_ev=10.0,triplet=false,pt_threshold=1e-4,unit="angstrom",max_ao=512,max_configurations=4096,memory_budget_bytes=536870912,*,thermal_response=None))]
    #[allow(clippy::too_many_arguments)]
    fn stda_xtb(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        n_states: usize,
        energy_window_ev: f64,
        triplet: bool,
        pt_threshold: f64,
        unit: &str,
        max_ao: usize,
        max_configurations: usize,
        memory_budget_bytes: usize,
        thermal_response: Option<bool>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let options = crate::stda_xtb::NativeStdaOptions {
            electronic_temperature: self.electronic.electronic_temperature,
            thermal_response,
            manifold_average: false,
            n_states,
            energy_window_ev,
            triplet,
            pt_threshold,
            max_ao,
            max_configurations,
            memory_budget_bytes,
        };
        let result = py
            .allow_threads(|| crate::stda_xtb::solve_stda_xtb_native(&system, &options))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("model", "sTDA-xTB")?;
        dict.set_item("implementation", "native-rust")?;
        dict.set_item(
            "electronic_temperature_kelvin",
            result.electronic_temperature_kelvin,
        )?;
        dict.set_item(
            "maximum_occupation_deviation",
            result.maximum_occupation_deviation,
        )?;
        dict.set_item(
            "lowest_ci_eigenvalue_hartree",
            result.lowest_ci_eigenvalue_hartree,
        )?;
        dict.set_item("nonpositive_ci_roots", result.nonpositive_ci_roots)?;
        dict.set_item(
            "excitation_energies_hartree",
            result
                .states
                .iter()
                .map(|s| s.excitation_hartree)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "excitation_energies_ev",
            result
                .states
                .iter()
                .map(|s| s.excitation_ev)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "oscillator_strengths",
            result
                .states
                .iter()
                .map(|s| s.oscillator_strength)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "transition_dipoles",
            result
                .states
                .iter()
                .map(|s| s.transition_dipole.to_vec())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "amplitudes",
            result
                .states
                .iter()
                .map(|s| s.amplitudes.clone())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("configurations", result.configurations)?;
        dict.set_item(
            "primary_configuration_count",
            result.primary_configuration_count,
        )?;
        dict.set_item(
            "promoted_configuration_count",
            result.promoted_configuration_count,
        )?;
        dict.set_item("orbital_energies_hartree", result.orbital_energies_hartree)?;
        dict.set_item("vtb_charges", result.vtb_charges)?;
        dict.set_item("basis_size", result.basis_size)?;
        dict.set_item("ci_eigenvalues_hartree", result.ci_eigenvalues_hartree)?;
        dict.set_item(
            "occupation_model",
            stda_occupation_model(result.thermal_response),
        )?;
        dict.set_item(
            "parameter_sha256",
            crate::stda_xtb::parameter_sha256().to_vec(),
        )?;
        dict.set_item("parameter_source_commit", crate::stda_xtb::XTB4STDA_COMMIT)?;
        dict.set_item(
            "response_reference_commit",
            crate::stda_xtb::STD2_REFERENCE_COMMIT,
        )?;
        Ok(dict.into())
    }

    /// Fully analytic native sTDA excitation and GFN1+sTDA composite PES gradient.
    /// State indices are zero based; every gradient is in Hartree/bohr.
    #[pyo3(signature=(numbers,positions,unit="angstrom",state=0,n_states=5,energy_window_ev=10.0,triplet=false,pt_threshold=1e-4,max_ao=512,max_configurations=4096,memory_budget_bytes=536870912,*,thermal_response=None,manifold_average=false))]
    #[allow(clippy::too_many_arguments)]
    fn stda_xtb_gradient(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        state: usize,
        n_states: usize,
        energy_window_ev: f64,
        triplet: bool,
        pt_threshold: f64,
        max_ao: usize,
        max_configurations: usize,
        memory_budget_bytes: usize,
        thermal_response: Option<bool>,
        manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let options = crate::stda_xtb::NativeStdaOptions {
            electronic_temperature: self.electronic.electronic_temperature,
            thermal_response,
            manifold_average,
            n_states,
            energy_window_ev,
            triplet,
            pt_threshold,
            max_ao,
            max_configurations,
            memory_budget_bytes,
        };
        let result = py
            .allow_threads(|| {
                crate::stda_xtb::solve_stda_xtb_total_gradient(
                    &system,
                    &self.params,
                    &self.electronic,
                    state,
                    &options,
                )
            })
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("model", "GFN1+sTDA-xTB")?;
        dict.set_item(
            "state_indexing",
            "signed energy-ordered CI roots before absorption filtering",
        )?;
        dict.set_item("nonpositive_ci_roots", result.nonpositive_ci_roots)?;
        dict.set_item(
            "lowest_ci_eigenvalue_hartree",
            result.lowest_ci_eigenvalue_hartree,
        )?;
        dict.set_item(
            "electronic_temperature_kelvin",
            self.electronic.electronic_temperature,
        )?;
        dict.set_item("implementation", "native-rust")?;
        dict.set_item("method", "analytic")?;
        dict.set_item("state", state)?;
        dict.set_item("ground_reference", "GFN1 free energy")?;
        dict.set_item(
            "occupation_model",
            stda_occupation_model(result.thermal_response),
        )?;
        if manifold_average {
            set_manifold_metadata(&dict, &result.roots)?;
        }
        dict.set_item("total_energy_hartree", result.total_energy)?;
        dict.set_item("ground_energy_hartree", result.ground_energy)?;
        dict.set_item("excitation_energy_hartree", result.excitation_energy)?;
        dict.set_item(
            "excitation_gradient",
            vec3_list(&result.excitation_gradient),
        )?;
        dict.set_item("gradient", vec3_list(&result.gradient))?;
        dict.set_item("forces", vec3_list(&result.forces))?;
        Ok(dict.into())
    }

    /// Analytic molecular Hessian of the explicit GFN1-free-energy plus sTDA
    /// excitation surface. Tensor entries are Hartree/bohr^2.
    #[pyo3(signature=(numbers,positions,unit="angstrom",state=0,n_states=5,energy_window_ev=10.0,triplet=false,pt_threshold=1e-4,max_ao=512,max_configurations=4096,memory_budget_bytes=536870912,*,thermal_response=None,manifold_average=false))]
    #[allow(clippy::too_many_arguments)]
    fn stda_xtb_hessian(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        state: usize,
        n_states: usize,
        energy_window_ev: f64,
        triplet: bool,
        pt_threshold: f64,
        max_ao: usize,
        max_configurations: usize,
        memory_budget_bytes: usize,
        thermal_response: Option<bool>,
        manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let options = crate::stda_xtb::NativeStdaOptions {
            electronic_temperature: self.electronic.electronic_temperature,
            thermal_response,
            manifold_average,
            n_states,
            energy_window_ev,
            triplet,
            pt_threshold,
            max_ao,
            max_configurations,
            memory_budget_bytes,
        };
        let result = py
            .allow_threads(|| {
                crate::stda_xtb::solve_stda_xtb_total_hessian(
                    &system,
                    &self.params,
                    &self.electronic,
                    state,
                    &options,
                )
            })
            .map_err(to_py_err)?;
        let dict = excited_hessian_dict(
            py,
            "GFN1+sTDA-xTB",
            state,
            self.electronic.electronic_temperature,
            result.ground_energy,
            result.excitation_energy,
            &result.ground_hessian,
            &result.excitation_hessian,
            &result.hessian,
        )?;
        dict.bind(py).set_item(
            "state_indexing",
            "signed energy-ordered CI roots before absorption filtering",
        )?;
        dict.bind(py)
            .set_item("nonpositive_ci_roots", result.nonpositive_ci_roots)?;
        dict.bind(py).set_item(
            "lowest_ci_eigenvalue_hartree",
            result.lowest_ci_eigenvalue_hartree,
        )?;
        dict.bind(py).set_item("occupation_model", stda_occupation_model(result.thermal_response))?;
        if manifold_average {
            set_manifold_metadata(dict.bind(py), &result.roots)?;
        }
        Ok(dict)
    }

    /// Closed-shell magnetic (GFN1-xTB-M) SCC total energy (Hartree) in a uniform
    /// field `b_field` (atomic units). Single-basis **M0** by default; pass
    /// `m1_basis_path` to a `GFN1-xTB-cc-pVDZ` secondary-basis file for the
    /// node-correct **M1** kinetic-energy correction. Non-periodic.
    /// Cheng & Wibowo-Teale, *J. Chem. Theory Comput.* **19**, 6226 (2023).
    #[pyo3(signature = (numbers, positions, b_field, unit = "angstrom", m1_basis_path = None))]
    fn magnetic_energy(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        b_field: (f64, f64, f64),
        unit: &str,
        m1_basis_path: Option<String>,
    ) -> PyResult<f64> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::new(b_field.0, b_field.1, b_field.2));
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let result = match &secondary {
            Some(sec) => crate::magnetic::run_magnetic_scc_m1(&system, &self.params, &options, sec),
            None => crate::magnetic::run_magnetic_scc(&system, &self.params, &options),
        }
        .map_err(to_py_err)?;
        Ok(result.energy)
    }

    /// Isotropic magnetizability `xi_iso = -1/3 Tr d^2E/dB^2` (Cheng & Wibowo-Teale
    /// eq 26), returned in SI units of `10^-30 J/T^2`. `analytic = True` (default)
    /// uses the McWeeny density-matrix CP-SCC response (one magnetic SCC + cheap LAO
    /// integral derivatives, no extra SCF); `analytic = False` central-differences the
    /// energy (`6+1` SCCs). **M0** by default; pass `m1_basis_path` for the node-correct
    /// **M1** variant (recommended — M0 is unreliable for lone-pair/heavier elements).
    /// Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", step = 0.02, analytic = true, m1_basis_path = None))]
    fn magnetizability(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        step: f64,
        analytic: bool,
        m1_basis_path: Option<String>,
    ) -> PyResult<f64> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let xi = if analytic {
            crate::magnetic::magnetizability_isotropic_analytic(
                &system,
                &self.params,
                &options,
                secondary.as_ref(),
                step,
            )
        } else {
            crate::magnetic::magnetizability_isotropic(
                &system,
                &self.params,
                &options,
                secondary.as_ref(),
                step,
            )
        }
        .map_err(to_py_err)?;
        Ok(xi * crate::magnetic::MAGNETIZABILITY_AU_TO_SI)
    }

    /// Diagonal magnetizability tensor `[xi_xx, xi_yy, xi_zz]` (`10^-30 J/T^2`) from
    /// the analytic CP-SCC response ([`magnetizability`] returns its mean). Useful for
    /// the diagonal anisotropy. `m1_basis_path` selects M1. Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", step = 0.02, m1_basis_path = None))]
    fn magnetizability_diagonal(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        step: f64,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<f64>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let diag = crate::magnetic::magnetizability_diagonal_analytic(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            step,
        )
        .map_err(to_py_err)?;
        Ok(diag
            .iter()
            .map(|x| x * crate::magnetic::MAGNETIZABILITY_AU_TO_SI)
            .collect())
    }

    /// Full symmetric magnetizability tensor `xi_ab` as a `3x3` nested list
    /// (`10^-30 J/T^2`) from the analytic CP-SCC response. The diagonal matches
    /// [`magnetizability_diagonal`]; the off-diagonals give the anisotropy.
    /// `m1_basis_path` selects M1. Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", step = 0.02, m1_basis_path = None))]
    fn magnetizability_tensor(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        step: f64,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let xi = crate::magnetic::magnetizability_tensor_analytic(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            step,
        )
        .map_err(to_py_err)?;
        Ok(xi
            .iter()
            .map(|row| {
                row.iter()
                    .map(|x| x * crate::magnetic::MAGNETIZABILITY_AU_TO_SI)
                    .collect()
            })
            .collect())
    }

    /// NMR nuclear magnetic shielding tensor of nucleus `nucleus` (0-based),
    /// `sigma_ab = d^2E/dB_a dm_b`, returned in ppm (`x1e6`) as a `3x3` nested list;
    /// the isotropic shielding is `trace/3`. Closed-shell, non-periodic, with the
    /// common gauge origin (CGO) at the shielded nucleus. The analytic CP-SCC magnetic
    /// response gives the paramagnetic part and a ground-state expectation the
    /// diamagnetic part; `m1_basis_path` selects the M1 kinetic-energy basis. Note: the
    /// GFN1 valence-only basis omits core electrons, so absolute shieldings are not
    /// comparable to all-electron references (use for within-method trends).
    #[pyo3(signature = (numbers, positions, nucleus, unit = "angstrom", m1_basis_path = None))]
    fn nmr_shielding(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        nucleus: usize,
        unit: &str,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        if nucleus >= system.atoms.len() {
            return Err(to_py_err(crate::error::Gfn1Error::InvalidInput(format!(
                "nmr_shielding: nucleus index {nucleus} out of range ({} atoms)",
                system.atoms.len()
            ))));
        }
        let gauge = system.atoms[nucleus].position; // common gauge origin at the nucleus
        let sh = crate::magnetic::nmr_shielding_tensor(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            nucleus,
            gauge,
        )
        .map_err(to_py_err)?;
        Ok(sh
            .sigma
            .iter()
            .map(|row| row.iter().map(|s| s * 1.0e6).collect())
            .collect())
    }

    /// Electric dipole polarizability `alpha_ij(B) = dmu_i/dE_j` (atomic units) in a
    /// uniform magnetic field `b_field` (atomic units), from the combined electric+
    /// magnetic SCC. Returns a `3x3` nested list. At `B = 0` it reduces to the field-
    /// free GFN1 polarizability. `e_step` is the electric-field step; `m1_basis_path`
    /// selects M1. Non-periodic.
    #[pyo3(signature = (numbers, positions, b_field = (0.0, 0.0, 0.0), unit = "angstrom", e_step = 0.002, m1_basis_path = None))]
    fn magnetic_polarizability(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        b_field: (f64, f64, f64),
        unit: &str,
        e_step: f64,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::new(b_field.0, b_field.1, b_field.2));
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let a = crate::magnetic::magnetic_polarizability(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            e_step,
        )
        .map_err(to_py_err)?;
        Ok(a.iter().map(|r| r.to_vec()).collect())
    }

    /// Cotton-Mouton tensor `d^2 alpha_ij / d B_k^2` (atomic units), indexed
    /// `[k][i][j]`, by finite difference of the magnetic-field polarizability. Returns
    /// a `3x3x3` nested list. Even in `B` (the symmetric, in `i,j`, part survives) — it
    /// drives the magnetic-field-induced birefringence. NOTE: the first derivative
    /// `d alpha / d B` (MCD/Faraday) is identically zero in the GFN1 monopole model
    /// (`dq/dB = 0`), so only this second derivative is a nonzero observable here.
    /// `e_step`/`b_step` are the field steps; `m1_basis_path` selects M1. Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", e_step = 0.002, b_step = 0.02, m1_basis_path = None))]
    fn cotton_mouton(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        e_step: f64,
        b_step: f64,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let cm = crate::magnetic::cotton_mouton_tensor(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            e_step,
            b_step,
        )
        .map_err(to_py_err)?;
        Ok(cm
            .iter()
            .map(|m| m.iter().map(|r| r.to_vec()).collect())
            .collect())
    }

    /// Static tensor `d alpha_ij / d B_k` (atomic units), indexed `[k][i][j]`, by
    /// finite difference of the magnetic-field polarizability about `B = 0`. Returns a
    /// `3x3x3` nested list. NOTE: this is **identically zero in the GFN1 monopole model**
    /// at zero field by time reversal. Optical MCD/Faraday spectra require a
    /// frequency-dependent complex response and are not implemented by this
    /// legacy-named static API. `e_step`/`b_step` are field steps; `m1_basis_path` selects M1.
    /// Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", e_step = 0.002, b_step = 0.01, m1_basis_path = None))]
    fn mcd(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        e_step: f64,
        b_step: f64,
        m1_basis_path: Option<String>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::zero());
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let mcd = crate::magnetic::mcd_tensor(
            &system,
            &self.params,
            &options,
            secondary.as_ref(),
            Vec3::zero(),
            e_step,
            b_step,
        )
        .map_err(to_py_err)?;
        Ok(mcd
            .iter()
            .map(|m| m.iter().map(|r| r.to_vec()).collect())
            .collect())
    }

    /// Raw orbital angular-momentum AO integral matrices used by the CD / magnetic-
    /// dipole response: `out[a]` is the real antisymmetric coefficient with
    /// `<mu|L_a|nu> = -i * out[a][mu][nu]`, `L = (r - O) x p` about `origin` (default the
    /// coordinate origin). The orbital magnetic dipole is `m = -1/2 L`. Returns a
    /// `3 x nAO x nAO` nested list (atomic units). Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", origin = None))]
    fn angular_momentum(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        origin: Option<(f64, f64, f64)>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let basis = crate::basis::BasisSet::build(
            &system,
            &self.params,
            crate::basis::BasisOptions {
                nprim: self.electronic.nprim,
            },
        )
        .map_err(to_py_err)?;
        let o = origin
            .map(|(x, y, z)| Vec3::new(x, y, z))
            .unwrap_or(Vec3::zero());
        let l = crate::magnetic::angular_momentum_matrix(&system, &basis, o);
        Ok(l.iter().map(matrix_to_nested).collect())
    }

    /// Raw London (GIAO) electric-dipole integral matrices `D_c(B)_{mu nu} =
    /// <om_mu|(r_c - O)|om_nu>` (`c = x, y, z`; `O` = origin) in a uniform magnetic field
    /// `b_field` (atomic units) — the length-gauge dipole behind MCD / optical rotation.
    /// Returns a dict with `re` and `im`, each a `3 x nAO x nAO` nested list (real and
    /// Hermitian at `B = 0`). Non-periodic.
    #[pyo3(signature = (numbers, positions, b_field = (0.0, 0.0, 0.0), unit = "angstrom"))]
    fn lao_dipole(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        b_field: (f64, f64, f64),
        unit: &str,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let basis = crate::basis::BasisSet::build(
            &system,
            &self.params,
            crate::basis::BasisOptions {
                nprim: self.electronic.nprim,
            },
        )
        .map_err(to_py_err)?;
        let field = crate::field::ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(b_field.0, b_field.1, b_field.2)),
            ..self.electronic.external_field
        };
        let d = crate::magnetic::lao_dipole_matrix(&system, &basis, &field);
        let dict = PyDict::new(py);
        dict.set_item(
            "re",
            d.iter()
                .map(|m| matrix_to_nested(&m.re))
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "im",
            d.iter()
                .map(|m| matrix_to_nested(&m.im))
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// Nuclear gradient / forces of the magnetic (GFN1-xTB-M) energy in a uniform
    /// field `b_field` (atomic units). `analytic = True` (default) uses the
    /// stationary-density contraction (one SCC with finite-differenced integral
    /// and fixed-charge builders; not a fully analytic integral derivative);
    /// `analytic = False` uses the `6N+1`-SCC finite difference of the energy.
    /// `m1_basis_path` selects the GFN1-xTB-M1 dual basis. Returns a dict with
    /// `energy_hartree`, `gradient` and `forces` (Hartree/bohr). Non-periodic.
    #[pyo3(signature = (
        numbers, positions, b_field, unit = "angstrom", step = 1.0e-3,
        analytic = true, m1_basis_path = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn magnetic_forces(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        b_field: (f64, f64, f64),
        unit: &str,
        step: f64,
        analytic: bool,
        m1_basis_path: Option<String>,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let mut options = self.electronic.clone();
        options.external_field.magnetic_field = Some(Vec3::new(b_field.0, b_field.1, b_field.2));
        let secondary = load_m1_basis(m1_basis_path.as_deref())?;
        let g = if analytic {
            crate::magnetic::magnetic_analytic_gradient(
                &system,
                &self.params,
                &options,
                secondary.as_ref(),
                step,
            )
        } else if secondary.is_some() {
            return Err(PyValueError::new_err(
                "M1 magnetic forces require analytic=True (the finite-difference path is M0 only)",
            ));
        } else {
            crate::magnetic::magnetic_gradient(&system, &self.params, &options, step)
        }
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("energy_hartree", g.energy)?;
        dict.set_item("gradient", vec3_list(&g.gradient))?;
        dict.set_item("forces", vec3_list(&g.forces))?;
        Ok(dict.into())
    }

    /// Periodic TD-GFN1 (TDA) excited states sampled over a Monkhorst-Pack k-mesh
    /// (optical `q = 0` transitions). `kmesh` is the mesh size `(n1, n2, n3)`;
    /// `gamma_centered` selects a Gamma-centred mesh. Returns the same dict shape as
    /// [`tda`] (`excitation_energies_hartree/ev`, `oscillator_strengths`). Requires
    /// integer (gapped) occupations.
    #[pyo3(signature = (
        numbers, positions, cell, kmesh = (2, 2, 2), unit = "angstrom",
        n_states = 5, spin = "singlet", pbc = (true, true, true), gamma_centered = true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn tda_kpoint(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        kmesh: (usize, usize, usize),
        unit: &str,
        n_states: usize,
        spin: &str,
        pbc: (bool, bool, bool),
        gamma_centered: bool,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let km = KMesh {
            size: [kmesh.0.max(1), kmesh.1.max(1), kmesh.2.max(1)],
            gamma_centered,
            fold_time_reversal: !gamma_centered,
        };
        let result = solve_tda_kpoint(
            &system,
            &self.params,
            &self.electronic,
            km,
            TdaOptions { n_states, spin },
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item(
            "excitation_energies_hartree",
            result
                .states
                .iter()
                .map(|s| s.excitation_energy)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "excitation_energies_ev",
            result
                .states
                .iter()
                .map(|s| s.excitation_energy * HARTREE_TO_EV)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "oscillator_strengths",
            result
                .states
                .iter()
                .map(|s| s.oscillator_strength)
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// Periodic TD-GFN1 (TDA) excited-state **gradient** over a Monkhorst-Pack
    /// k-mesh. `method` is `analytic` (default; the exact direct-CPHF gradient) or
    /// `fd` (central finite difference of the matched k-mesh excitation energy).
    /// Returns a dict with `total_energy_hartree`, `excitation_energy_hartree`,
    /// `gradient`, and `forces` (atomic units). Requires integer (gapped) occupations.
    #[pyo3(signature = (
        numbers, positions, cell, kmesh = (2, 2, 2), state = 0, unit = "angstrom",
        n_states = 5, spin = "singlet", pbc = (true, true, true), gamma_centered = true,
        method = "analytic", step = 1.0e-3
    ))]
    #[allow(clippy::too_many_arguments)]
    fn tda_kpoint_gradient(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        kmesh: (usize, usize, usize),
        state: usize,
        unit: &str,
        n_states: usize,
        spin: &str,
        pbc: (bool, bool, bool),
        gamma_centered: bool,
        method: &str,
        step: f64,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let km = KMesh {
            size: [kmesh.0.max(1), kmesh.1.max(1), kmesh.2.max(1)],
            gamma_centered,
            fold_time_reversal: !gamma_centered,
        };
        let opts = TdaOptions {
            n_states: n_states.max(state + 1),
            spin,
        };
        let g = match method
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "analytic" | "" => solve_tda_kpoint_gradient_analytic(
                &system,
                &self.params,
                &self.electronic,
                km,
                state,
                opts,
            ),
            "fd" | "finite_difference" | "numerical" => solve_tda_kpoint_gradient(
                &system,
                &self.params,
                &self.electronic,
                km,
                state,
                opts,
                step,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA k-mesh gradient method `{other}` (use analytic or fd)"
                )))
            }
        }
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("total_energy_hartree", g.total_energy)?;
        dict.set_item("excitation_energy_hartree", g.excitation_energy)?;
        dict.set_item("gradient", vec3_list(&g.gradient))?;
        dict.set_item("forces", vec3_list(&g.forces))?;
        Ok(dict.into())
    }

    /// TD-GFN1 (TDA) excited states. Returns a dict with `excitation_energies_ev`,
    /// `excitation_energies_hartree`, `oscillator_strengths`, and
    /// `transition_dipoles` (atomic units). Non-periodic only. `transition_dipole`
    /// selects legacy Mulliken strengths or full AO-integral `"ao_length"`.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", n_states = 5, spin = "singlet", transition_dipole = "mulliken"))]
    #[allow(clippy::too_many_arguments)]
    fn tda(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        n_states: usize,
        spin: &str,
        transition_dipole: &str,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let electronic =
            run_electronic(&system, &self.params, self.electronic.clone()).map_err(to_py_err)?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let operator = match transition_dipole {
            "mulliken" => crate::td::TdaTransitionDipole::Mulliken,
            "ao_length" => crate::td::TdaTransitionDipole::AoLength,
            _ => {
                return Err(PyValueError::new_err(
                    "transition_dipole must be mulliken or ao_length",
                ))
            }
        };
        let result = crate::td::solve_tda_with_dipole(
            &system,
            &self.params,
            &electronic,
            TdaOptions { n_states, spin },
            operator,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("transition_dipole_method", transition_dipole)?;
        dict.set_item(
            "electronic_temperature_kelvin",
            electronic.electronic_temperature,
        )?;
        dict.set_item(
            "excitation_energies_hartree",
            result
                .states
                .iter()
                .map(|s| s.excitation_energy)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "excitation_energies_ev",
            result
                .states
                .iter()
                .map(|s| s.excitation_energy * HARTREE_TO_EV)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "oscillator_strengths",
            result
                .states
                .iter()
                .map(|s| s.oscillator_strength)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "transition_dipoles",
            result
                .states
                .iter()
                .map(|s| {
                    vec![
                        s.transition_dipole.x,
                        s.transition_dipole.y,
                        s.transition_dipole.z,
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }

    /// TD-GFN1 total excited-state gradient in Hartree/bohr.
    /// The default `analytic` uses direct SCC response. `semi_numerical` adds a
    /// numerical excitation derivative to the analytic ground gradient. `fd`
    /// differences the entire energy. These methods support molecules and Gamma.
    /// Degenerate roots require the explicit manifold_average mean surface.
    #[pyo3(signature = (
        numbers, positions, unit = "angstrom", state = 0, n_states = 5,
        spin = "singlet", step = 1.0e-3, method = "analytic",
        cell = None, pbc = (true, true, true), *, manifold_average = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn tda_gradient(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        state: usize,
        n_states: usize,
        spin: &str,
        step: f64,
        method: &str,
        cell: Option<Vec<Vec<f64>>>,
        pbc: (bool, bool, bool),
        manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = match cell {
            Some(cell) => build_periodic_system(numbers, positions, cell, pbc, unit, charge)?,
            None => build_system(numbers, positions, unit, charge)?,
        };
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown TDA spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let grad_method = TdaGradientMethod::parse(method).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown TDA gradient method `{method}` (use semi_numerical, fd, or analytic)"
            ))
        })?;

        let options = TdaOptions { n_states: n_states.max(state.saturating_add(1)), spin };
        let dict = PyDict::new(py);
        let g = if manifold_average {
            let result = crate::solve_tda_manifold_gradient(
                &system, &self.params, &self.electronic, state, options, step, grad_method,
            ).map_err(to_py_err)?;
            dict.set_item("manifold_first_root", result.roots.start)?;
            dict.set_item("manifold_size", result.roots.len())?;
            dict.set_item("manifold_status", tda_manifold_status(result.roots.len()))?;
            dict.set_item("effective_step_bohr", result.effective_step_bohr)?;
            result.mean
        } else {
            solve_tda_gradient_method(
                &system, &self.params, &self.electronic, state, options, step, grad_method,
            ).map_err(to_py_err)?
        };
        dict.set_item("total_energy_hartree", g.total_energy)?;
        dict.set_item("excitation_energy_hartree", g.excitation_energy)?;
        dict.set_item("gradient", vec3_list(&g.gradient))?;
        dict.set_item("forces", vec3_list(&g.forces))?;
        Ok(dict.into())
    }

    /// Analytic molecular total excited-state TDA Hessian and its ground and
    /// excitation components. No nuclear finite-difference step is used.
    #[pyo3(signature=(numbers,positions,unit="angstrom",state=0,n_states=5,spin="singlet",memory_budget_bytes=536870912,*,manifold_average=false))]
    #[allow(clippy::too_many_arguments)]
    fn tda_hessian(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        state: usize,
        n_states: usize,
        spin: &str,
        memory_budget_bytes: usize,
        manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let spin = match spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            _ => return Err(PyValueError::new_err("spin must be singlet or triplet")),
        };

        let (result, roots) = py.allow_threads(|| {
            let tda = TdaOptions { n_states, spin };
            if manifold_average {
                crate::solve_tda_manifold_hessian_analytic_with_budget(
                    &system, &self.params, &self.electronic, state, tda, memory_budget_bytes,
                ).map(|result| (result.mean, Some(result.roots)))
            } else {
                crate::solve_tda_hessian_analytic_with_budget(
                    &system, &self.params, &self.electronic, state, tda, memory_budget_bytes,
                ).map(|result| (result, None))
            }
        }).map_err(to_py_err)?;
        let out = excited_hessian_dict(
            py,
            "TD-GFN1",
            state,
            self.electronic.electronic_temperature,
            result.ground_energy,
            result.excitation_energy,
            &result.ground_hessian,
            &result.excitation_hessian,
            &result.hessian,
        )?;
        if let Some(roots) = roots {
            out.bind(py).set_item("manifold_first_root", roots.start)?;
            out.bind(py).set_item("manifold_size", roots.len())?;
            out.bind(py).set_item("manifold_status", tda_manifold_status(roots.len()))?;
        }
        Ok(out)
    }

    /// Molecular full-TD RPA excitation spectrum on the calculator's reference.
    #[pyo3(signature=(numbers,positions,unit="angstrom",n_states=5,spin="singlet"))]
    fn rpa(
        &self, py: Python<'_>, numbers: Vec<u8>, positions: Vec<Vec<f64>>,
        unit: &str, n_states: usize, spin: &str,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(numbers, positions, unit, self.electronic.charge.unwrap_or(0.))?;
        let spin = parse_excited_spin(spin)?;
        let (electronic, result) = py.allow_threads(|| {
            let electronic = run_electronic(&system, &self.params, self.electronic.clone())?;
            let result = crate::solve_rpa(&system, &self.params, &electronic, TdaOptions { n_states, spin })?;
            Ok::<_, crate::Gfn1Error>((electronic, result))
        }).map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("electronic_temperature_kelvin", electronic.electronic_temperature)?;
        dict.set_item("ground_energy_hartree", electronic.total_free)?;
        dict.set_item("excitation_energies_hartree", result.states.iter().map(|s| s.excitation_energy).collect::<Vec<_>>())?;
        dict.set_item("excitation_energies_ev", result.states.iter().map(|s| s.excitation_energy * HARTREE_TO_EV).collect::<Vec<_>>())?;
        dict.set_item("oscillator_strengths", result.states.iter().map(|s| s.oscillator_strength).collect::<Vec<_>>())?;
        dict.set_item("transition_dipoles", result.states.iter().map(|s| s.transition_dipole.to_array()).collect::<Vec<_>>())?;
        Ok(dict.into())
    }

    /// Analytic RPA excited-state gradient, with optional manifold averaging.
    #[pyo3(signature=(numbers,positions,unit="angstrom",state=0,spin="singlet",*,manifold_average=false))]
    fn rpa_gradient(
        &self, py: Python<'_>, numbers: Vec<u8>, positions: Vec<Vec<f64>>,
        unit: &str, state: usize, spin: &str, manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(numbers, positions, unit, self.electronic.charge.unwrap_or(0.))?;
        let options = TdaOptions { n_states: state.saturating_add(1), spin: parse_excited_spin(spin)? };
        let result = py.allow_threads(|| crate::td::hessian::solve_molecular_manifold_gradient(
            &system, &self.params, &self.electronic, state, options, crate::td::ResponseMethod::Rpa,
            manifold_average, None,
        )).map_err(to_py_err)?;
        let dict = PyDict::new(py);
        if manifold_average {
            set_manifold_metadata(&dict, &result.roots)?;
        }
        let result = result.mean;
        dict.set_item("method", "analytic")?;
        dict.set_item("total_energy_hartree", result.total_energy)?;
        dict.set_item("excitation_energy_hartree", result.excitation_energy)?;
        dict.set_item("gradient", vec3_list(&result.gradient))?;
        dict.set_item("forces", vec3_list(&result.forces))?;
        Ok(dict.into())
    }

    /// Analytic RPA excited-state Hessian, with optional manifold averaging.
    #[pyo3(signature=(numbers,positions,unit="angstrom",state=0,spin="singlet",memory_budget_bytes=536870912,*,manifold_average=false))]
    fn rpa_hessian(
        &self, py: Python<'_>, numbers: Vec<u8>, positions: Vec<Vec<f64>>,
        unit: &str, state: usize, spin: &str, memory_budget_bytes: usize, manifold_average: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(numbers, positions, unit, self.electronic.charge.unwrap_or(0.))?;
        let options = TdaOptions { n_states: state.saturating_add(1), spin: parse_excited_spin(spin)? };
        let result = py.allow_threads(|| crate::td::hessian::solve_response_hessian(
            &system, &self.params, &self.electronic, state, options, memory_budget_bytes,
            crate::td::ResponseMethod::Rpa, manifold_average,
        )).map_err(to_py_err)?;
        let h = &result.mean;
        let dict = excited_hessian_dict(
            py, "RPA-GFN1", state, self.electronic.electronic_temperature,
            h.ground_energy, h.excitation_energy, &h.ground_hessian, &h.excitation_hessian, &h.hessian,
        )?;
        if manifold_average {
            set_manifold_metadata(dict.bind(py), &result.roots)?;
        }
        Ok(dict)
    }

    /// Finite-difference parameter derivatives. `targets` are strings like
    /// `glob:ks`, `elem:1:GAM`, `pair:1:6`. Returns a list of dicts with `target`,
    /// `value`, and `energy_derivative` (Hartree per unit parameter).
    #[pyo3(signature = (numbers, positions, targets, unit = "angstrom", step = 1.0e-4))]
    fn parameter_derivatives(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        targets: Vec<String>,
        unit: &str,
        step: f64,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let parsed = targets
            .iter()
            .map(|t| ParameterTarget::parse(t))
            .collect::<crate::error::Result<Vec<_>>>()
            .map_err(to_py_err)?;
        let options = ParamDerivativeOptions {
            step,
            electronic: self.electronic.clone(),
            include_forces: false,
            include_stress: false,
        };
        let derivs = parameter_finite_difference(&system, &self.params, &parsed, &options)
            .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(derivs.len());
        for d in derivs {
            let dict = PyDict::new(py);
            dict.set_item("target", d.target.label())?;
            dict.set_item("value", d.value)?;
            dict.set_item("energy_derivative", d.energy_derivative)?;
            out.push(dict.into());
        }
        Ok(out)
    }

    /// Current values of the addressed parameter targets (the starting point for
    /// the PyTorch interop).
    fn parameter_values(&self, targets: Vec<String>) -> PyResult<Vec<f64>> {
        let parsed = parse_targets(&targets)?;
        parsed
            .iter()
            .map(|t| self.params.parameter_value(t).map_err(to_py_err))
            .collect()
    }

    /// Evaluate the total free energy and its parameter gradient `dE/dp` at an
    /// explicit set of parameter `values` (one per target). This is the building
    /// block for the PyTorch `autograd.Function` interop (`gfn1_rs.torch_interop`):
    /// the parameter file is not mutated and PyTorch is not a dependency.
    #[pyo3(signature = (numbers, positions, targets, values, unit = "angstrom", step = 1.0e-4))]
    #[allow(clippy::too_many_arguments)]
    fn parameter_energy_and_gradient(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        targets: Vec<String>,
        values: Vec<f64>,
        unit: &str,
        step: f64,
    ) -> PyResult<(f64, Vec<f64>)> {
        if targets.len() != values.len() {
            return Err(PyValueError::new_err(format!(
                "targets ({}) and values ({}) must have equal length",
                targets.len(),
                values.len()
            )));
        }
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let parsed = parse_targets(&targets)?;
        // Apply the requested parameter values to a working copy.
        let mut params = self.params.clone();
        for (target, &value) in parsed.iter().zip(values.iter()) {
            params = params.with_parameter(target, value).map_err(to_py_err)?;
        }
        let energy = run_electronic(&system, &params, self.electronic.clone())
            .map_err(to_py_err)?
            .total_free;
        let options = ParamDerivativeOptions {
            step,
            electronic: self.electronic.clone(),
            include_forces: false,
            include_stress: false,
        };
        let derivs =
            parameter_finite_difference(&system, &params, &parsed, &options).map_err(to_py_err)?;
        let grad = derivs.iter().map(|d| d.energy_derivative).collect();
        Ok((energy, grad))
    }

    /// Finite-difference dipole-moment parameter derivatives `dmu/dp` (atomic
    /// units). Returns a list of dicts with `target` and `dipole_derivative`
    /// (`[dmu_x, dmu_y, dmu_z]`).
    #[pyo3(signature = (numbers, positions, targets, unit = "angstrom", step = 1.0e-4))]
    fn dipole_parameter_derivatives(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        targets: Vec<String>,
        unit: &str,
        step: f64,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let parsed = parse_targets(&targets)?;
        let derivs =
            parameter_dipole_derivatives(&system, &self.params, &parsed, &self.electronic, step)
                .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(derivs.len());
        for (target, dmu) in derivs {
            let dict = PyDict::new(py);
            dict.set_item("target", target.label())?;
            dict.set_item("dipole_derivative", dmu.to_vec())?;
            out.push(dict.into());
        }
        Ok(out)
    }

    /// Analytic nuclear Hessian `∂²E/∂R_a∂R_b` (Hartree / bohr²). Returns the
    /// `3N x 3N` matrix as a nested list (row-major, atom-major Cartesian ordering
    /// `[a0x, a0y, a0z, a1x, ...]`). Fully analytic (no finite differences); the
    /// same Hessian that drives `ir_spectrum` / `raman_spectrum` internally.
    /// Non-periodic — for a periodic cell use [`Self::hessian_periodic`].
    #[pyo3(signature = (numbers, positions, unit = "angstrom"))]
    fn hessian(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let h = crate::hessian::analytic_hessian(&system, &self.params, self.hessian_options())
            .map_err(to_py_err)?
            .hessian;
        let n = h.rows();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push((0..n).map(|j| h[(i, j)]).collect());
        }
        Ok(rows)
    }

    /// **Periodic** analytic nuclear Hessian `∂²E/∂R_a∂R_b` (Hartree / bohr²) at fixed cell. `cell`
    /// is 3 lattice vectors (rows a, b, c) in the chosen unit; `pbc` is the per-axis periodicity;
    /// `kgrid` is the Monkhorst-Pack mesh (`None` or `[1,1,1]` → the Γ-point path
    /// [`crate::pbc::pbc_gamma_hessian`], otherwise the k-point path
    /// [`crate::pbc::pbc_kpoint_hessian`] whose coupled complex CPXTB is solved with preconditioned
    /// CG). Returns the `3N x 3N` matrix as a nested list (same ordering as [`Self::hessian`]).
    #[pyo3(signature = (numbers, positions, cell, pbc = (true, true, true), kgrid = None, unit = "angstrom", gamma_centered = false))]
    #[allow(clippy::too_many_arguments)]
    fn hessian_periodic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        unit: &str,
        gamma_centered: bool,
    ) -> PyResult<Vec<Vec<f64>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let kmesh = match kgrid {
            None | Some((1, 1, 1)) => KMesh::gamma(),
            Some((a, b, c)) => KMesh::monkhorst_pack([a, b, c]),
        };
        let mut pbc_options = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        pbc_options.kmesh = kmesh;
        let options = self.electronic.clone();
        let is_gamma = matches!(kgrid, None | Some((1, 1, 1)));
        pbc_options.kmesh.gamma_centered = gamma_centered;
        let h = if is_gamma {
            pbc_gamma_hessian(&system, &self.params, &options, &pbc_options)
        } else {
            pbc_kpoint_hessian(&system, &self.params, &options, &pbc_options)
        }
        .map_err(to_py_err)?
        .hessian;
        let n = h.rows();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push((0..n).map(|j| h[(i, j)]).collect());
        }
        Ok(rows)
    }

    /// Harmonic vibrational analysis from the analytic Hessian. Returns a dict:
    /// `wavenumbers` (cm⁻¹; imaginary modes reported as negative) and `modes`
    /// (mass-weighted normal-mode displacements, each a length-`3N` list).
    /// Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom"))]
    fn vibrational_frequencies(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers.clone(),
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let h = crate::hessian::analytic_hessian(&system, &self.params, self.hessian_options())
            .map_err(to_py_err)?
            .hessian;
        let modes = crate::vibrational::vibrational_analysis(&h, &numbers).map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("wavenumbers", modes.wavenumbers)?;
        dict.set_item("modes", modes.modes)?;
        Ok(dict.into())
    }

    /// Commensurate finite-q modes using an analytic supercell CP-SCC Hessian.
    #[pyo3(signature=(numbers,positions,cell,repetitions,qpoints,unit="angstrom",pbc=(true,true,true),masses_amu=None,memory_budget_bytes=536870912,max_atoms=128,zeta_hartree=None,parameter_provenance=None))]
    fn phonons_commensurate(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        repetitions: (usize, usize, usize),
        qpoints: Vec<(f64, f64, f64)>,
        unit: &str,
        pbc: (bool, bool, bool),
        masses_amu: Option<Vec<f64>>,
        memory_budget_bytes: usize,
        max_atoms: usize,
        zeta_hartree: Option<Vec<(u8, usize, f64)>>,
        parameter_provenance: Option<String>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let masses = masses_amu.unwrap_or_else(|| {
            numbers
                .iter()
                .map(|&z| crate::data_tables::relative_atomic_mass(z))
                .collect()
        });
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let limits = crate::pbc::phonon::PhononSupercellOptions {
            repetitions: [repetitions.0, repetitions.1, repetitions.2],
            max_atoms,
            memory_budget_bytes,
        };
        let pbc_options = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        let reference = if let Some(rows) = zeta_hartree {
            let count = rows.len();
            let soc = crate::soc::SocParameters {
                zeta: rows.into_iter().map(|(z, l, v)| ((z, l), v)).collect(),
                provenance: parameter_provenance
                    .ok_or_else(|| PyValueError::new_err("SOC parameter provenance is required"))?,
            };
            if soc.zeta.len() != count {
                return Err(PyValueError::new_err("duplicate SOC element/shell"));
            }
            crate::pbc::phonon::AnalyticSupercellPhonons::build_soc(
                &system,
                &self.params,
                &self.electronic,
                &pbc_options,
                &masses,
                limits,
                &soc,
            )
        } else {
            if parameter_provenance.is_some() {
                return Err(PyValueError::new_err(
                    "parameter_provenance requires zeta_hartree",
                ));
            }
            crate::pbc::phonon::AnalyticSupercellPhonons::build(
                &system,
                &self.params,
                &self.electronic,
                &pbc_options,
                &masses,
                limits,
            )
        }
        .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(qpoints.len());
        for q in qpoints {
            let result = reference.at_q([q.0, q.1, q.2]).map_err(to_py_err)?;
            let dict = PyDict::new(py);
            dict.set_item("backend", result.backend)?;
            dict.set_item("q_fractional", result.q_fractional)?;
            dict.set_item("wavenumbers_cm1", result.wavenumbers_cm1)?;
            dict.set_item("raw_hermiticity_error", result.raw_hermiticity_error)?;
            dict.set_item(
                "raw_hessian_asymmetry_hartree_per_bohr2",
                reference.raw_hessian_asymmetry,
            )?;
            dict.set_item(
                "max_spinor_response_residual",
                reference.max_spinor_response_residual,
            )?;
            dict.set_item("method_citations", result.method_citations)?;
            dict.set_item("repetitions", repetitions)?;
            dict.set_item(
                "electronic_temperature_K",
                reference.electronic_temperature_kelvin,
            )?;
            out.push(dict.into());
        }
        Ok(out)
    }

    /// Mass/Eckart-projected modes with explicit isotope masses (amu).
    #[pyo3(signature=(numbers,positions,unit="angstrom",masses_amu=None))]
    fn molecular_vibrations(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        masses_amu: Option<Vec<f64>>,
    ) -> PyResult<Py<PyDict>> {
        let masses = masses_amu.unwrap_or_else(|| {
            numbers
                .iter()
                .map(|&z| crate::data_tables::relative_atomic_mass(z))
                .collect()
        });
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let h = crate::analytic_hessian(&system, &self.params, self.hessian_options())
            .map_err(to_py_err)?
            .hessian;
        let positions: Vec<_> = system.atoms.iter().map(|a| a.position).collect();
        let modes =
            crate::molecular_vibrational_analysis(&h, &positions, &masses, Default::default())
                .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("wavenumbers_cm1", modes.vibrations.wavenumbers)?;
        dict.set_item("modes_cartesian_amu", modes.vibrations.modes)?;
        dict.set_item("masses_amu", masses)?;
        dict.set_item("rigid_rank", modes.rigid_directions.len())?;
        dict.set_item("raw_hessian_asymmetry", modes.raw_hessian_asymmetry)?;
        Ok(dict.into())
    }

    /// PES-only analytic VPT2; requires a separately optimized stationary minimum.
    #[pyo3(signature=(numbers,positions,unit="angstrom",masses_amu=None,gradient_tolerance=1e-5,resonance_threshold_cm1=10.0))]
    fn vpt2_pes(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        masses_amu: Option<Vec<f64>>,
        gradient_tolerance: f64,
        resonance_threshold_cm1: f64,
    ) -> PyResult<Py<PyDict>> {
        let masses = masses_amu.unwrap_or_else(|| {
            numbers
                .iter()
                .map(|&z| crate::data_tables::relative_atomic_mass(z))
                .collect()
        });
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let result = crate::molecular_vpt2_pes(
            &system,
            &self.params,
            self.electronic.clone(),
            &masses,
            crate::Vpt2Options {
                resonance_threshold_hartree: resonance_threshold_cm1
                    / crate::constants::HARTREE_TO_WAVENUMBER,
                ..Default::default()
            },
            gradient_tolerance,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("scope", result.spectrum.scope)?;
        dict.set_item("method_citations", result.spectrum.method_citations)?;
        dict.set_item("harmonic_cm1", result.normal_modes.vibrations.wavenumbers)?;
        dict.set_item("fundamentals_cm1", result.spectrum.fundamentals_cm1)?;
        dict.set_item("zpe_hartree", result.spectrum.ground.energy_hartree())?;
        dict.set_item(
            "residual_gradient_hartree_per_bohr",
            result.residual_gradient_hartree_per_bohr,
        )?;
        dict.set_item("masses_amu", masses)?;
        Ok(dict.into())
    }

    /// Several analytic FC3 contractions sharing one SCF and charge-space LU.
    #[pyo3(signature=(numbers,positions,directions,unit="angstrom",memory_budget_bytes=536870912))]
    fn third_derivative_contractions(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        directions: Vec<Vec<f64>>,
        unit: &str,
        memory_budget_bytes: usize,
    ) -> PyResult<Vec<f64>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let reference = crate::third_derivative::CubicReference::build(
            &system,
            &self.params,
            &self.electronic,
            memory_budget_bytes,
        )
        .map_err(to_py_err)?;
        directions
            .iter()
            .map(|v| reference.directional(v).map_err(to_py_err))
            .collect()
    }

    /// Physical max-1 complex-spinor SCC with explicit zeta L.S constants.
    #[pyo3(signature=(numbers,positions,zeta_hartree,parameter_provenance,unit="angstrom",compute_forces=false,compute_response=false))]
    fn soc_calculate(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        zeta_hartree: Vec<(u8, usize, f64)>,
        parameter_provenance: String,
        unit: &str,
        compute_forces: bool,
        compute_response: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let count = zeta_hartree.len();
        let parameters = crate::soc::SocParameters {
            zeta: zeta_hartree
                .into_iter()
                .map(|(z, l, x)| ((z, l), x))
                .collect(),
            provenance: parameter_provenance,
        };
        if parameters.zeta.len() != count {
            return Err(PyValueError::new_err("duplicate SOC element/shell"));
        }
        let (result, gradient) = if compute_forces {
            let (r, g) = crate::soc::soc_analytic_gradient(
                &system,
                &self.params,
                &self.electronic,
                &parameters,
            )
            .map_err(to_py_err)?;
            (r, Some(g))
        } else {
            (
                crate::soc::run_soc_scc(&system, &self.params, &self.electronic, &parameters)
                    .map_err(to_py_err)?,
                None,
            )
        };
        let dict = PyDict::new(py);
        if compute_response {
            let response = crate::soc::soc_static_polarizability(&system, &self.params, &result)
                .map_err(to_py_err)?;
            dict.set_item("polarizability_au", response.tensor)?;
            dict.set_item(
                "response_normalized_residual",
                response.max_normalized_residual,
            )?;
        }
        if let Some(g) = gradient {
            dict.set_item("gradient_hartree_per_bohr", vec3_list(&g))?;
        }
        dict.set_item("energy_hartree", result.total_free)?;
        dict.set_item("soc_energy_hartree", result.soc_energy)?;
        dict.set_item("calculation_id", result.calculation_id)?;
        dict.set_item("charges", result.atomic_charges)?;
        dict.set_item("occupations", result.occupations)?;
        dict.set_item("orbital_energies_hartree", result.eigen.values)?;
        dict.set_item("occupation_limit", 1)?;
        dict.set_item("method_citations", result.method_citations)?;
        dict.set_item("parameter_provenance", result.parameter_provenance)?;
        dict.set_item("max_charge_residual", result.max_charge_residual)?;
        Ok(dict.into())
    }

    /// Physical-spinor periodic SOC energy/forces/stress and nuclear CP-SCC.
    #[pyo3(signature=(numbers,positions,cell,zeta_hartree,parameter_provenance,unit="angstrom",pbc=(true,true,true),kgrid=None,compute_forces=false,compute_stress=false,compute_hessian=false))]
    fn soc_calculate_periodic(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        zeta_hartree: Vec<(u8, usize, f64)>,
        parameter_provenance: String,
        unit: &str,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        compute_forces: bool,
        compute_stress: bool,
        compute_hessian: bool,
    ) -> PyResult<Py<PyDict>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let count = zeta_hartree.len();
        let parameters = crate::soc::SocParameters {
            zeta: zeta_hartree
                .into_iter()
                .map(|(z, l, x)| ((z, l), x))
                .collect(),
            provenance: parameter_provenance,
        };
        if parameters.zeta.len() != count {
            return Err(PyValueError::new_err("duplicate SOC element/shell"));
        }
        let mut pbc = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        pbc.kmesh = match kgrid {
            None | Some((1, 1, 1)) => KMesh::gamma(),
            Some((a, b, c)) => KMesh::monkhorst_pack([a, b, c]),
        };
        let (result, hessian) = if compute_hessian {
            let h = crate::pbc::pbc_soc_hessian(
                &system,
                &self.params,
                &self.electronic,
                &pbc,
                &parameters,
            )
            .map_err(to_py_err)?;
            (
                h.soc,
                Some((h.hessian, h.raw_max_asymmetry, h.max_response_residual)),
            )
        } else {
            (
                crate::pbc::run_pbc_soc_scc(
                    &system,
                    &self.params,
                    &self.electronic,
                    &pbc,
                    &parameters,
                )
                .map_err(to_py_err)?,
                None,
            )
        };
        let dict = PyDict::new(py);
        if compute_forces {
            let gradient = result
                .analytic_gradient(&system, &self.params, &self.electronic, &pbc)
                .map_err(to_py_err)?;
            dict.set_item("gradient_hartree_per_bohr", vec3_list(&gradient))?;
            dict.set_item(
                "forces_hartree_per_bohr",
                vec3_list(&gradient.iter().map(|v| -*v).collect::<Vec<_>>()),
            )?;
        }
        if compute_stress {
            let stress = result
                .analytic_stress(&system, &self.params, &self.electronic, &pbc)
                .map_err(to_py_err)?;
            dict.set_item(
                "stress_hartree_per_bohr3",
                (0..3)
                    .map(|a| (0..3).map(|b| stress[(a, b)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
        }
        if let Some((h, raw, residual)) = hessian {
            dict.set_item(
                "hessian_hartree_per_bohr2",
                (0..h.rows())
                    .map(|a| (0..h.cols()).map(|b| h[(a, b)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            dict.set_item("raw_hessian_asymmetry_hartree_per_bohr2", raw)?;
            dict.set_item("max_response_residual", residual)?;
        }
        dict.set_item("energy_hartree", result.total_free)?;
        dict.set_item("soc_energy_hartree", result.soc_energy)?;
        dict.set_item("calculation_id", result.calculation_id)?;
        dict.set_item("charges", result.atomic_charges)?;
        dict.set_item("occupation_limit", 1)?;
        dict.set_item("occupations_k", result.occupations_k)?;
        dict.set_item(
            "orbital_energies_hartree_k",
            result
                .eigen_k
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "kpoints_fractional",
            result
                .kpoints
                .iter()
                .map(|k| k.fractional)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "kweights",
            result.kpoints.iter().map(|k| k.weight).collect::<Vec<_>>(),
        )?;
        dict.set_item("electronic_temperature_K", result.electronic_temperature)?;
        dict.set_item("parameter_provenance", result.parameter_provenance)?;
        dict.set_item("method_citations", result.method_citations)?;
        dict.set_item("max_charge_residual", result.max_charge_residual)?;
        Ok(dict.into())
    }

    /// Finite-difference Hessian parameter derivatives `dH/dp` (Hartree/bohr^2).
    /// Returns a list of dicts with `target` and `hessian_derivative` (a `3N x 3N`
    /// nested list).
    #[pyo3(signature = (numbers, positions, targets, unit = "angstrom", step = 1.0e-4))]
    fn hessian_parameter_derivatives(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        targets: Vec<String>,
        unit: &str,
        step: f64,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let parsed = parse_targets(&targets)?;
        let derivs = parameter_hessian_derivatives(
            &system,
            &self.params,
            &parsed,
            &self.hessian_options(),
            step,
        )
        .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(derivs.len());
        for (target, dh) in derivs {
            let dict = PyDict::new(py);
            dict.set_item("target", target.label())?;
            let rows: Vec<Vec<f64>> = (0..dh.rows())
                .map(|i| (0..dh.cols()).map(|j| dh[(i, j)]).collect())
                .collect();
            dict.set_item("hessian_derivative", rows)?;
            out.push(dict.into());
        }
        Ok(out)
    }

    /// Semi-numerical nuclear **third derivative** (cubic force constants) along a direction.
    /// `direction` is a flat `3N` vector (e.g. a normal mode); returns the `3N x 3N` matrix
    /// `K[a][b] = Σ_c v_c · ∂³E/∂R_a∂R_b∂R_c` — the directional derivative of the analytic
    /// Hessian, by central finite difference (**two** analytic-Hessian evaluations). Non-periodic.
    #[pyo3(signature = (numbers, positions, direction, unit = "angstrom", step = 1.0e-3))]
    fn third_derivative_along(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        direction: Vec<f64>,
        unit: &str,
        step: f64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let k = crate::third_derivative::third_derivative_seminumerical_vector(
            &system,
            &self.params,
            self.hessian_options(),
            &direction,
            step,
        )
        .map_err(to_py_err)?;
        let n = k.rows();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push((0..n).map(|j| k[(i, j)]).collect());
        }
        Ok(rows)
    }

    /// Strict **closed-form** nuclear third derivative (cubic force constants)
    /// `T_abc = ∂³E/∂R_a∂R_b∂R_c`. Returns a list of `3N` dense slabs, each a `3N x 3N`
    /// matrix `slab[c][a][b] = T_abc` (in Hartree / bohr³). Fully analytic — no finite
    /// differences anywhere. Non-periodic. For best accuracy set a tight SCF on the
    /// calculator (`energy_tolerance`/`charge_tolerance`); the directional
    /// [`third_derivative_along`] is the cheaper semi-numerical alternative.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", *, disk_storage = false, output_path = None, memory_budget_bytes = None, cell = None, pbc = (true, true, true), kmesh = None))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        disk_storage: bool,
        output_path: Option<std::path::PathBuf>,
        memory_budget_bytes: Option<usize>,
        cell: Option<Vec<Vec<f64>>>,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
    ) -> PyResult<Py<PyAny>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = match cell {
            Some(cell) => build_periodic_system(numbers, positions, cell, pbc, unit, charge)?,
            None => build_system(numbers, positions, unit, charge)?,
        };
        let periodic = system.lattice.as_ref().map(|_| {
            pbc_kmesh_options(kmesh, self.electronic.pbc_ao_cutoff, None, None)
        });
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let storage = crate::DerivativeStorageOptions { disk_storage, output_path, memory_budget_bytes };
        let result = py.allow_threads(|| crate::third_derivative_with_storage(
            &system, &self.params, &options, cutoff, periodic.as_ref(), &storage,
        )).map_err(to_py_err)?;
        match result {
            crate::DerivativeOutput::File(file) => packed_derivative_dict(py, &file),
            crate::DerivativeOutput::Memory(tensor) =>
                Ok(PyList::new(py, symmetric_third_to_nested(&tensor))?.into_any().unbind()),
        }
    }

    /// Closed-form **Vector mode** of the cubic force constants: the directional third derivative
    /// `K[a][b] = sum_c v_c T_abc` (the derivative of the Hessian along `direction`, e.g. a normal
    /// mode) as a single `3N x 3N` matrix. Returns only the `3N x 3N` contraction (not the full `3N^3`
    /// tensor) — the route to use when you need a directional cubic constant. `direction` is a flat
    /// `3N` vector. Fully analytic (no finite differences). Non-periodic.
    #[pyo3(signature = (numbers, positions, direction, unit = "angstrom"))]
    fn third_derivative_vector(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        direction: Vec<f64>,
        unit: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        if self.electronic.electronic_temperature > 0. {
            return Ok(matrix_to_nested(
                &crate::third_derivative_finite_t_vector(
                    &system,
                    &self.params,
                    &options,
                    cutoff,
                    &direction,
                )
                .map_err(to_py_err)?,
            ));
        }
        let k = crate::third_derivative::third_derivative_analytic_vector(
            &system,
            &self.params,
            options,
            cutoff,
            &direction,
        )
        .map_err(to_py_err)?;
        let n = k.rows();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push((0..n).map(|j| k[(i, j)]).collect());
        }
        Ok(rows)
    }

    /// Closed-form **Block mode** of the cubic force constants: the sub-tensor restricted to the
    /// Cartesian DOFs of the chosen `atoms` (atom indices). Returns `(dofs, slabs)` where `dofs` are
    /// the global DOF indices (`3*atom + axis`) and `slabs[ci][ai][bi] = T[dofs[ai]][dofs[bi]][dofs[ci]]`
    /// — an `O(|block|^3)` tensor for local anharmonicity over a chosen subregion. Non-periodic.
    #[pyo3(signature = (numbers, positions, atoms, unit = "angstrom"))]
    fn third_derivative_block(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        atoms: Vec<usize>,
        unit: &str,
    ) -> PyResult<(Vec<usize>, Vec<Vec<Vec<f64>>>)> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        if self.electronic.electronic_temperature > 0. {
            let mut seen = std::collections::HashSet::new();
            if atoms
                .iter()
                .any(|&a| a >= system.atoms.len() || !seen.insert(a))
            {
                return Err(PyValueError::new_err(
                    "block atoms must be distinct valid atom indices",
                ));
            }
            let dofs: Vec<usize> = atoms
                .iter()
                .flat_map(|&a| (0..3).map(move |k| 3 * a + k))
                .collect();
            let tensor = crate::third_derivative_finite_t_block(
                &system,
                &self.params,
                &options,
                cutoff,
                &dofs,
            )
            .map_err(to_py_err)?;
            return Ok((dofs, symmetric_third_to_nested(&tensor)));
        }
        let (dofs, slabs) = crate::third_derivative::third_derivative_analytic_block(
            &system,
            &self.params,
            options,
            cutoff,
            &atoms,
        )
        .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(slabs.len());
        for slab in &slabs {
            let m = slab.rows();
            let mut rows = Vec::with_capacity(m);
            for i in 0..m {
                rows.push((0..m).map(|j| slab[(i, j)]).collect());
            }
            out.push(rows);
        }
        Ok((dofs, out))
    }

    // -----------------------------------------------------------------------
    // v0.5.0: semi-numerical + finite-temperature cubic force constants (FC3),
    // the quartic ladder (FC4), and the periodic third-derivative / Grueneisen
    // stack. Every value below is in ATOMIC UNITS (Hartree / bohr^n); the ASE
    // layer (`gfn1_rs.ase.GFN1RSCalculator`) is the only unit boundary.
    // -----------------------------------------------------------------------

    /// **Semi-numerical Dense mode** of the cubic force constants: the full tensor
    /// `T_abc = ∂³E/∂R_a∂R_b∂R_c` (Hartree / bohr³) built by central finite difference of the
    /// *analytic* Hessian along each Cartesian DOF (`2·3N` analytic-Hessian evaluations).
    ///
    /// Returned **expanded** as an `ndof³` nested list `out[c][a][b] = T_abc`; the tensor is
    /// fully symmetric, so the index order is immaterial. Memory is `8·(3N)³` bytes
    /// (30 DOF → 216 kB, 90 DOF → 5.8 MB) — the packed Rust store is ~6x smaller, so prefer
    /// [`Self::third_derivative_along`] (Vector) or [`Self::third_derivative_seminumerical_block`]
    /// for large systems.
    ///
    /// Like the analytic [`Self::third_derivative`], this route supports **Fermi smearing**
    /// (`electronic_temperature > 0` with fractional occupations) and then returns the
    /// free-energy cubic force constants. Non-periodic; `step` is the displacement in **bohr**.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", step = 1.0e-3))]
    fn third_derivative_seminumerical(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        step: f64,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let store = crate::third_derivative::third_derivative_seminumerical_dense(
            &system,
            &self.params,
            self.hessian_options(),
            step,
        )
        .map_err(to_py_err)?;
        Ok(symmetric_third_to_nested(&store))
    }

    /// **Semi-numerical Block mode** of the cubic force constants over the Cartesian DOFs of the
    /// chosen `atoms` (atom indices). Returns `(dofs, slabs)` with the global DOF indices
    /// (`3*atom + axis`) and `slabs[ci][ai][bi] = T[dofs[ai]][dofs[bi]][dofs[ci]]`
    /// (Hartree / bohr³) — **bit-for-bit** the sub-block of
    /// [`Self::third_derivative_seminumerical`], but only `|dofs|` central Hessian pairs are
    /// taken, so both cost and memory scale with the subset rather than with `N`.
    /// Supports Fermi smearing. Non-periodic; `step` is in **bohr**.
    #[pyo3(signature = (numbers, positions, atoms, unit = "angstrom", step = 1.0e-3))]
    fn third_derivative_seminumerical_block(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        atoms: Vec<usize>,
        unit: &str,
        step: f64,
    ) -> PyResult<(Vec<usize>, Vec<Vec<Vec<f64>>>)> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let (dofs, slabs) = crate::third_derivative::third_derivative_seminumerical_block(
            &system,
            &self.params,
            self.hessian_options(),
            &atoms,
            step,
        )
        .map_err(to_py_err)?;
        Ok((dofs, slabs.iter().map(matrix_to_nested).collect()))
    }

    /// **Finite-temperature analytic directional third derivative** — the single contracted
    /// scalar `e³[v] = Σ_abc T_abc v_a v_b v_c` (Hartree / bohr³ for a dimensionless `direction`),
    /// assembled by the product rule over the directional Hessian's composition.
    ///
    /// Every ingredient is occupation-agnostic, so ONE code path serves `T_elec = 0` and
    /// **Fermi-smeared** systems alike; at `T_elec = 0` it is equality-gated against the
    /// adjoint-assembled [`Self::third_derivative_vector`] contracted `vvv`. No finite
    /// differences anywhere. `direction` is a flat `3N` vector. Non-periodic.
    ///
    /// Exactly degenerate *fractionally occupied* blocks are **supported** — the second-order
    /// charge-space solver takes the frame-free resolvent (Daleckii–Krein) form there, gated
    /// on symmetry-exact NiO at 3000 K. The quartic sibling still rejects them.
    #[pyo3(signature = (numbers, positions, direction, unit = "angstrom"))]
    fn third_derivative_finite_t_directional(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        direction: Vec<f64>,
        unit: &str,
    ) -> PyResult<f64> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        crate::third_derivative::finite_t::directional_third_finite_t(
            &system,
            &self.params,
            &options,
            cutoff,
            &direction,
        )
        .map_err(to_py_err)
    }

    /// **Finite-temperature analytic Dense mode**: the full `T_abc` (Hartree / bohr³) recovered
    /// from shared-reference directional evaluations of
    /// [`Self::third_derivative_finite_t_directional`] by the cubic polarization identity.
    ///
    /// Returned **expanded** as an `ndof³` nested list `out[c][a][b] = T_abc` (fully symmetric,
    /// so the index order is immaterial); memory is `8·(3N)³` bytes. Cost scales as
    /// `~C(3N+2, 3)` directional evaluations against ONE shared SCF/CPXTB reference, so this is
    /// the expensive mode — prefer the directional route, or
    /// [`Self::third_derivative_finite_t_block`] for a subregion. Supports Fermi smearing.
    /// Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom"))]
    fn third_derivative_finite_t(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let store = crate::third_derivative::finite_t::third_derivative_finite_t_dense(
            &system,
            &self.params,
            &options,
            cutoff,
        )
        .map_err(to_py_err)?;
        Ok(symmetric_third_to_nested(&store))
    }

    /// **Finite-temperature analytic Block mode**: the `|dofs|³` sub-tensor of
    /// [`Self::third_derivative_finite_t`] over the given **global DOF indices** `dofs`
    /// (`3*atom + axis`, unlike the atom-indexed seminumerical block). Returns `(dofs, slabs)`
    /// with `slabs[ci][ai][bi] = T[dofs[ai]][dofs[bi]][dofs[ci]]` (Hartree / bohr³) — only the
    /// polarization directions the requested triples need are evaluated. Supports Fermi
    /// smearing. Non-periodic.
    #[pyo3(signature = (numbers, positions, dofs, unit = "angstrom"))]
    fn third_derivative_finite_t_block(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        dofs: Vec<usize>,
        unit: &str,
    ) -> PyResult<(Vec<usize>, Vec<Vec<Vec<f64>>>)> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let store = crate::third_derivative::finite_t::third_derivative_finite_t_block(
            &system,
            &self.params,
            &options,
            cutoff,
            &dofs,
        )
        .map_err(to_py_err)?;
        Ok((dofs, symmetric_third_to_nested(&store)))
    }

    /// **The analytic directional fourth derivative** `e⁗[v] = Σ_abcd Q_abcd v_a v_b v_c v_d`
    /// (Hartree / bohr⁴ for a dimensionless `direction`) — one SCF, one CPXTB solve, one
    /// charge-space first-order solve and one second-order solve along `v`, then the five gated
    /// stages summed. No finite differences.
    ///
    /// Guards (both raised as `ValueError`): every active term must support analytic order 4
    /// (e.g. `multipole=True` blocks it). Positive electronic temperature selects the
    /// analytic free-energy quartic, including exactly degenerate fractional occupations.
    /// `direction` is a flat `3N` vector. Non-periodic.
    #[pyo3(signature = (numbers, positions, direction, unit = "angstrom"))]
    fn fourth_derivative_directional(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        direction: Vec<f64>,
        unit: &str,
    ) -> PyResult<f64> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        if self.electronic.electronic_temperature > 0. {
            return crate::directional_fourth_finite_t(
                &system,
                &self.params,
                &options,
                cutoff,
                &direction,
            )
            .map_err(to_py_err);
        }
        crate::fourth_derivative::directional_fourth_derivative(
            &system,
            &self.params,
            &options,
            cutoff,
            &direction,
        )
        .map_err(to_py_err)
    }

    /// **Seminumerical directional fourth derivative** (Hartree / bohr⁴): the central finite
    /// difference along `v`, with everything reconverged at `R ± step·v`, of the analytic
    /// directional THIRD derivative. This is the verification reference of
    /// [`Self::fourth_derivative_directional`] and the production fallback where the analytic
    /// quartic is unavailable.
    ///
    /// The finite-differenced reference is the occupation-agnostic
    /// [`Self::third_derivative_finite_t_directional`], so this route ALSO serves
    /// **Fermi-smeared** systems. `step` is the displacement in **bohr** (`1e-3` is a good
    /// default: the `h²` truncation error meets the SCF/CPXTB noise floor there for a tight SCF).
    /// Non-periodic.
    #[pyo3(signature = (numbers, positions, direction, unit = "angstrom", step = 1.0e-3))]
    fn fourth_derivative_directional_seminumerical(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        direction: Vec<f64>,
        unit: &str,
        step: f64,
    ) -> PyResult<f64> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        crate::fourth_derivative::directional_fourth_seminumerical(
            &system,
            &self.params,
            &options,
            cutoff,
            &direction,
            step,
        )
        .map_err(to_py_err)
    }

    /// **Analytic Dense mode** of the quartic force constants `Q_abcd = ∂⁴E/∂R_a∂R_b∂R_c∂R_d`
    /// (Hartree / bohr⁴), every element recovered from the directional quartic by the
    /// polarization identity against ONE shared SCF/CPXTB reference.
    ///
    /// Returned **expanded** as an `ndof⁴` nested list `out[a][b][c][d] = Q_abcd`; the tensor is
    /// fully symmetric, so the index order is immaterial. **Small systems only**: the expansion
    /// stores `8·(3N)⁴` bytes (9 DOF → 52 kB, 30 DOF → 6.5 MB) and the assembly costs
    /// `~(3N)⁴/24` directional evaluations. The binding therefore enforces the crate cap
    /// `MAX_FOURTH_DERIVATIVE_NDOF` (= 30 DOF / 10 atoms, exported by this module and shared
    /// with the analytic D3 quartic) and raises `ValueError` above it — use
    /// [`Self::fourth_derivative_block`] or the directional mode instead.
    ///
    /// Same guards as [`Self::fourth_derivative_directional`] (analytic order 4 for every active
    /// term; finite-temperature free-energy response when temperature is positive). Non-periodic.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", *, disk_storage = false, output_path = None, memory_budget_bytes = None))]
    fn fourth_derivative(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        disk_storage: bool,
        output_path: Option<std::path::PathBuf>,
        memory_budget_bytes: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let system = build_system(numbers, positions, unit, self.electronic.charge.unwrap_or(0.0))?;
        if !disk_storage {
            check_fourth_dense_size(3 * system.atoms.len(), "fourth_derivative")?;
        }
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let storage = crate::DerivativeStorageOptions { disk_storage, output_path, memory_budget_bytes };
        let result = py.allow_threads(|| crate::fourth_derivative_with_storage(
            &system, &self.params, &options, cutoff, &storage,
        )).map_err(to_py_err)?;
        match result {
            crate::DerivativeOutput::File(file) => packed_derivative_dict(py, &file),
            crate::DerivativeOutput::Memory(tensor) =>
                Ok(PyList::new(py, symmetric_fourth_to_nested(&tensor))?.into_any().unbind()),
        }
    }

    /// **Analytic Block mode** of the quartic force constants: the `|dofs|⁴` sub-tensor over the
    /// given **global DOF indices** `dofs` (`3*atom + axis`). Returns `(dofs, q_block)` with
    /// `q_block[ai][bi][ci][di] = Q[dofs[ai]][dofs[bi]][dofs[ci]][dofs[di]]` (Hartree / bohr⁴)
    /// — the production interface when only a few modes/atoms matter, since the
    /// directional-evaluation count drops to `O(|dofs|⁴/24)`.
    ///
    /// The same `MAX_FOURTH_DERIVATIVE_NDOF` cap is applied to `|dofs|` (not to `3N`), so a
    /// small block of a large molecule is fine. Same guards as
    /// [`Self::fourth_derivative_directional`]. Non-periodic.
    #[pyo3(signature = (numbers, positions, dofs, unit = "angstrom"))]
    fn fourth_derivative_block(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        dofs: Vec<usize>,
        unit: &str,
    ) -> PyResult<(Vec<usize>, Vec<Vec<Vec<Vec<f64>>>>)> {
        let system = build_system(
            numbers,
            positions,
            unit,
            self.electronic.charge.unwrap_or(0.0),
        )?;
        check_fourth_dense_size(dofs.len(), "fourth_derivative_block")?;
        let options = self.hessian_options();
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        if self.electronic.electronic_temperature > 0. {
            return Ok((
                dofs.clone(),
                symmetric_fourth_to_nested(
                    &crate::fourth_derivative_finite_t_block(
                        &system,
                        &self.params,
                        &options,
                        cutoff,
                        &dofs,
                    )
                    .map_err(to_py_err)?,
                ),
            ));
        }
        let store = crate::fourth_derivative::fourth_derivative_analytic_block(
            &system,
            &self.params,
            &options,
            cutoff,
            &dofs,
        )
        .map_err(to_py_err)?;
        Ok((dofs, symmetric_fourth_to_nested(&store)))
    }

    /// **Periodic (Γ-point) semi-numerical third derivative, Dense mode**: `3N` slabs
    /// `out[c][a][b] = ∂(H_ab)/∂R_c ≈ ∂³E/∂R_a∂R_b∂R_c` (Hartree / bohr³), each a central finite
    /// difference of the analytic Γ-point PBC Hessian along the atomic Cartesian DOF `c` with
    /// the **lattice held fixed**.
    ///
    /// Cost: `2·3N` analytic periodic Hessians (each re-runs the periodic SCC). Every slab is
    /// `(a,b)`-symmetric, but the tensor is deliberately **not** re-symmetrised across `c`, so
    /// invariance checks (e.g. the acoustic sum rule) stay genuine tests.
    ///
    /// `cell` is 3 lattice vectors (rows a, b, c) in the chosen unit and `step` is in **bohr**.
    /// `ao_cutoff` / `ewald_real_cutoff` / `ewald_sr_cutoff` (bohr) override the library defaults
    /// (30 / 40 / 10) — lowering them is the standard speed lever. Integer and Fermi-smeared
    /// occupations are supported. Tighten SCC convergence when differencing a thermal
    /// Hessian so reconvergence noise remains below the requested derivative accuracy.
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), unit = "angstrom", step = 1.0e-3,
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        unit: &str,
        step: f64,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        let slabs = crate::pbc::pbc_third_derivative_seminumerical_dense(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
            step,
        )
        .map_err(to_py_err)?;
        Ok(slabs.iter().map(matrix_to_nested).collect())
    }

    /// **Periodic (Γ-point) semi-numerical third derivative, Vector mode**: the directional
    /// contraction `K[a][b] = Σ_c v_c ∂(H_ab)/∂R_c` as a single `3N x 3N` matrix
    /// (Hartree / bohr³ for a dimensionless `direction`).
    ///
    /// This is an **exact** contraction of the same per-DOF central differences
    /// [`Self::third_derivative_periodic`] builds — bit-for-bit equal to contracting the dense
    /// slabs, not merely equal to finite-difference truncation order. DOFs with `v_c == 0` are
    /// skipped, so the cost is `2·nnz(v)` analytic periodic Hessians and the peak memory is one
    /// matrix instead of `3N`. Arguments as in [`Self::third_derivative_periodic`].
    #[pyo3(signature = (
        numbers, positions, cell, direction, pbc = (true, true, true), unit = "angstrom",
        step = 1.0e-3, ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic_vector(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        direction: Vec<f64>,
        pbc: (bool, bool, bool),
        unit: &str,
        step: f64,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        let k = crate::pbc::pbc_third_derivative_seminumerical_vector(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
            step,
            &direction,
        )
        .map_err(to_py_err)?;
        Ok(matrix_to_nested(&k))
    }

    /// **Analytic Γ-point periodic third derivative, Dense mode**: the fully symmetric cubic
    /// force constants `T_abc = ∂³A/∂R_a∂R_b∂R_c` as an `(3N, 3N, 3N)` nested list
    /// (Hartree / bohr³), `out[c][a][b] = T_abc`.
    ///
    /// **No finite differences anywhere** — unlike [`Self::third_derivative_periodic`], which
    /// differences the analytic periodic Hessian. The assembly differentiates the periodic CPXTB
    /// response chain analytically through the Ewald-split Klopman–Ohno γ. It differentiates
    /// the fixed-electron-number free energy at the calculator's electronic temperature,
    /// including occupation and chemical-potential response and exactly degenerate fractional
    /// bands. All-component Hessian-difference validation is recorded in `docs/pbc.md`.
    /// Positions are in `unit`; the tensor is in **atomic units**.
    ///
    /// Γ-point only; use [`Self::third_derivative_periodic_kpoint_analytic`] for a k-mesh.
    /// Model options without analytic third derivatives (multipole, long-range exchange,
    /// DFT+U, spin polarization, external field, experimental D4) raise; the seminumerical
    /// [`Self::third_derivative_periodic`] supports the Hessian's option coverage.
    ///
    /// Cost is `~C(3N+2, 3)` directional evaluations (56 for a 2-atom cell, 816 for 8 atoms),
    /// evaluated in parallel against one shared reference; it grows as `N³`, so prefer
    /// [`Self::third_derivative_periodic_analytic_vector`] when one direction suffices.
    /// `ao_cutoff` / `ewald_real_cutoff` / `ewald_sr_cutoff` (bohr) override the library
    /// defaults (30 / 40 / 10).
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), unit = "angstrom",
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic_analytic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        unit: &str,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        let store = crate::pbc::pbc_gamma_third_analytic_dense(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
        )
        .map_err(to_py_err)?;
        Ok(symmetric_third_to_nested(&store))
    }

    /// **Analytic Γ-point periodic third derivative, Vector mode**: the single contracted scalar
    /// `e³[v] = Σ_abc T_abc v_a v_b v_c` along one direction (Hartree / bohr³ for a
    /// dimensionless `direction`) — the cubic anharmonicity along e.g. a normal mode.
    ///
    /// The cheapest analytic output mode by far: one periodic SCC and one directional assembly,
    /// where [`Self::third_derivative_periodic_analytic`] needs `~C(3N+2, 3)` of them and
    /// [`Self::third_derivative_periodic_vector`] needs `2·nnz(v)` periodic Hessians. No finite
    /// differences. `direction` is a flat `3N` vector; guards and cutoffs as in
    /// [`Self::third_derivative_periodic_analytic`].
    #[pyo3(signature = (
        numbers, positions, cell, direction, pbc = (true, true, true), unit = "angstrom",
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic_analytic_vector(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        direction: Vec<f64>,
        pbc: (bool, bool, bool),
        unit: &str,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<f64> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        crate::pbc::pbc_gamma_third_analytic_vector(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
            &direction,
        )
        .map_err(to_py_err)
    }

    /// **Analytic Brillouin-zone-sampled periodic third derivative, Dense mode**: the fully
    /// symmetric cubic force constants `T_abc = ∂³A/∂R_a∂R_b∂R_c` as an `(3N, 3N, 3N)` nested
    /// list (Hartree / bohr³), `out[c][a][b] = T_abc`, over the Monkhorst–Pack mesh `kgrid`.
    ///
    /// **No finite differences anywhere**, and — unlike
    /// [`Self::third_derivative_periodic_analytic`] — **no Γ-only restriction**: that is the
    /// point of this entry point. The assembly is the same five-group closed form, fed a
    /// first-order response from the complex k-point CPXTB and a second-order response from the
    /// complex resolvent (Daleckii–Krein) form; the frozen and response paths consume the
    /// Brillouin-zone inverse transform of those, which is real. On `kgrid = (1,1,1)` it
    /// reproduces the Γ path to `~1e-17` relative (`docs/pbc.md` §2d).
    ///
    /// Canonical Fermi-smearing and exactly degenerate fractionally occupied bands
    /// are supported, including the global chemical-potential response. This is
    /// the third derivative of the requested-temperature free energy at fixed Ne.
    /// Model options without analytic third derivatives (multipole, long-range
    /// exchange, DFT+U, spin polarization, external field, experimental D4) are rejected.
    ///
    /// Cost is `~C(3N+2, 3)` directional evaluations, each `n_k` times an already substantial
    /// per-direction cost; prefer [`Self::third_derivative_periodic_kpoint_analytic_vector`]
    /// whenever one direction suffices. `ao_cutoff` / `ewald_real_cutoff` / `ewald_sr_cutoff`
    /// (bohr) override the library defaults (30 / 40 / 10).
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), kgrid = None, unit = "angstrom",
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic_kpoint_analytic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        unit: &str,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_kmesh_options(kgrid, ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        let store = crate::pbc::pbc_kpoint_third_analytic_dense(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
        )
        .map_err(to_py_err)?;
        Ok(symmetric_third_to_nested(&store))
    }

    /// **Analytic Brillouin-zone-sampled periodic third derivative, Vector mode**: the single
    /// contracted scalar `e³[v] = Σ_abc T_abc v_a v_b v_c` along one direction (Hartree / bohr³
    /// for a dimensionless `direction`) — the cubic anharmonicity along e.g. a normal mode, with
    /// the k-mesh convergence that a Γ-only cell cannot give you.
    ///
    /// By far the cheapest way to get a k-sampled cubic force constant: one periodic SCC, one
    /// per-DOF CPXTB sweep and one directional assembly, where the seminumerical route
    /// (`gfn1_rs::pbc_kpoint_third_derivative_seminumerical_vector`) needs `2·nnz(v)` full
    /// k-point Hessians and [`Self::third_derivative_periodic_kpoint_analytic`] needs
    /// `~C(3N+2, 3)` assemblies. No
    /// finite differences. `direction` is a flat `3N` vector; guards, mesh and cutoffs as in
    /// [`Self::third_derivative_periodic_kpoint_analytic`].
    #[pyo3(signature = (
        numbers, positions, cell, direction, pbc = (true, true, true), kgrid = None,
        unit = "angstrom", ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn third_derivative_periodic_kpoint_analytic_vector(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        direction: Vec<f64>,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        unit: &str,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<f64> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_kmesh_options(kgrid, ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        crate::pbc::pbc_kpoint_third_analytic_vector(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
            &direction,
        )
        .map_err(to_py_err)
    }

    /// **Strain-mixed third derivative** `∂(H_ab)/∂(ln V)` as a `3N x 3N` matrix
    /// (Hartree / bohr², since `ln V` is dimensionless), by a central difference of the analytic
    /// Γ-point PBC Hessian under **isotropic frozen-ion** volumetric strain: the two displaced
    /// cells are `V(1 ± delta)`, reached by scaling all three lattice vectors by
    /// `(1 ± delta)^(1/3)` with the fractional coordinates frozen. The denominator is the exact
    /// log-volume separation `ln((1+delta)/(1-delta))`, so the estimator is `O(delta²)`-accurate.
    ///
    /// Contracting this with a mass-weighted normal mode gives that mode's Grüneisen parameter
    /// directly; [`Self::gruneisen`] instead re-diagonalises at both volumes, which additionally
    /// resolves mode crossings. Cost: exactly **2** analytic periodic Hessians. `delta` must be
    /// in `(0, 1)`; cutoffs as in [`Self::third_derivative_periodic`].
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), unit = "angstrom", delta = 5.0e-3,
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn strain_hessian_derivative(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        unit: &str,
        delta: f64,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let pbc_options = pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff);
        let m = crate::pbc::pbc_strain_hessian_derivative(
            &system,
            &self.params,
            &self.electronic,
            &pbc_options,
            delta,
        )
        .map_err(to_py_err)?;
        Ok(matrix_to_nested(&m))
    }

    /// Mode and thermodynamic **Grüneisen parameters** of a periodic system at the Γ point, from
    /// three analytic PBC Hessians (`V0`, `V0(1±delta)`) under isotropic frozen-ion volumetric
    /// strain, with the strained modes matched onto the reference ordering by maximum overlap and
    /// degenerate subspaces averaged.
    ///
    /// With `second_order = True` the curvature `gamma2_i = d²ln ω_i / d(ln V)²` is fitted on its
    /// own `ln V` nodes at `V(1 ± delta_second)`; `stencil` is `"three_point"` (default) or
    /// `"five_point"` (adds `V(1±2·delta_second)`: two more Hessians, `O(delta⁴)` truncation,
    /// needs `delta_second < 0.5`).
    ///
    /// **`delta_second` is deliberately larger than `delta`** (default `2e-2` vs `5e-3`): a second
    /// difference amplifies phonon noise by `delta⁻²` where the first-order `gamma` only suffers
    /// `delta⁻¹`, so one shared step cannot serve both orders. Passing `delta_second = delta`
    /// restores the historical behaviour where the curvature cost no extra Hessian — and with it
    /// the noise that made the three- and five-point stencils disagree by more than the value.
    ///
    /// Returns a dict — `mode_gamma`, `mode_gamma2`, `mode_gamma_refit`, `mode_q`,
    /// `thermodynamic_gamma`, `thermodynamic_gamma2`, `thermodynamic_gamma2_full` and
    /// `match_overlaps` are **dimensionless**; `frequencies_cm1` (+ `_expanded` / `_compressed`,
    /// permuted onto the reference ordering, imaginary modes negative) are in **cm⁻¹**; `volume`
    /// is in **bohr³**. The thermodynamic tables are lists of `[T_kelvin, value]`. Second-order
    /// fields are `NaN` / empty unless `second_order = True`, and `second_order_stencil` is then
    /// the stencil name (else `None`). Acoustic modes (the lowest `acoustic_modes`, default 3)
    /// and unusable modes carry `NaN`.
    ///
    /// Integer and Fermi-smeared occupations are supported at the calculator's electronic
    /// temperature. Tighten SCC convergence for thermal Hessian differences. Cutoffs as in
    /// [`Self::third_derivative_periodic`].
    #[pyo3(signature = (
        numbers, positions, cell, pbc = (true, true, true), unit = "angstrom", delta = 5.0e-3,
        temperatures = vec![300.0], acoustic_modes = 3, degeneracy_tolerance_cm1 = 1.0,
        second_order = false, stencil = "three_point", delta_second = None,
        ao_cutoff = None, ewald_real_cutoff = None, ewald_sr_cutoff = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gruneisen(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        unit: &str,
        delta: f64,
        temperatures: Vec<f64>,
        acoustic_modes: usize,
        degeneracy_tolerance_cm1: f64,
        second_order: bool,
        stencil: &str,
        delta_second: Option<f64>,
        ao_cutoff: Option<f64>,
        ewald_real_cutoff: Option<f64>,
        ewald_sr_cutoff: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let second_order_stencil = match stencil
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .as_str()
        {
            "three_point" | "three" | "3" | "" => SecondOrderStencil::ThreePoint,
            "five_point" | "five" | "5" => SecondOrderStencil::FivePoint,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown Grueneisen second-order stencil `{other}` \
                     (use three_point or five_point)"
                )))
            }
        };
        let options = GruneisenOptions {
            delta,
            temperatures,
            electronic: self.electronic.clone(),
            pbc: pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff),
            // The binding hardwires a Gamma-only mesh (`pbc_gamma_options`), so the
            // k-point Hessian route would have nothing to sample. Exposing it needs a
            // `kgrid` argument here first.
            kpoint: false,
            acoustic_modes,
            degeneracy_tolerance_cm1,
            second_order,
            second_order_stencil,
            // `None` here means "the library default", not "reuse `delta`":
            // `gamma2` is a second difference and needs a wider step than
            // `gamma` (pass `delta_second = delta` for the old shared-step
            // behaviour, which is free but noise-dominated).
            delta_second: Some(
                delta_second.unwrap_or(crate::pbc::gruneisen::DEFAULT_GRUNEISEN_DELTA_SECOND),
            ),
        };
        let g = pbc_gruneisen(&system, &self.params, &options).map_err(to_py_err)?;
        let pairs = |table: &[(f64, f64)]| -> Vec<Vec<f64>> {
            table.iter().map(|&(t, v)| vec![t, v]).collect()
        };
        let dict = PyDict::new(py);
        dict.set_item("volume", g.volume)?;
        dict.set_item("delta", g.delta)?;
        dict.set_item("delta_second", g.delta_second)?;
        dict.set_item("frequencies_cm1", g.frequencies_cm1.clone())?;
        dict.set_item(
            "frequencies_cm1_expanded",
            g.frequencies_cm1_expanded.clone(),
        )?;
        dict.set_item(
            "frequencies_cm1_compressed",
            g.frequencies_cm1_compressed.clone(),
        )?;
        dict.set_item("mode_gamma", g.mode_gamma.clone())?;
        dict.set_item("mode_gamma2", g.mode_gamma2.clone())?;
        dict.set_item("mode_gamma_refit", g.mode_gamma_refit.clone())?;
        dict.set_item("mode_q", g.mode_q())?;
        dict.set_item("thermodynamic_gamma", pairs(&g.thermodynamic_gamma))?;
        dict.set_item("thermodynamic_gamma2", pairs(&g.thermodynamic_gamma2))?;
        dict.set_item(
            "thermodynamic_gamma2_full",
            pairs(&g.thermodynamic_gamma2_full),
        )?;
        dict.set_item(
            "second_order_stencil",
            g.second_order_stencil.map(|s| match s {
                SecondOrderStencil::ThreePoint => "three_point",
                SecondOrderStencil::FivePoint => "five_point",
            }),
        )?;
        dict.set_item("match_overlaps", g.match_overlaps.clone())?;
        dict.set_item("min_optical_overlap", g.min_optical_overlap())?;
        dict.set_item("acoustic_modes", g.acoustic_modes)?;
        dict.set_item(
            "degenerate_groups",
            g.degenerate_groups
                .iter()
                .map(|&(start, len)| vec![start, len])
                .collect::<Vec<_>>(),
        )?;
        Ok(dict.into())
    }
    /// Full commensurate phonon mesh from one analytic supercell Hessian.
    #[pyo3(signature=(numbers,positions,cell,repetitions,unit="angstrom",pbc=(true,true,true),masses_amu=None,memory_budget_bytes=536870912,max_atoms=128))]
    fn phonon_mesh(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        repetitions: (usize, usize, usize),
        unit: &str,
        pbc: (bool, bool, bool),
        masses_amu: Option<Vec<f64>>,
        memory_budget_bytes: usize,
        max_atoms: usize,
    ) -> PyResult<Py<PyAny>> {
        let masses = masses_amu.unwrap_or_else(|| {
            numbers
                .iter()
                .map(|&z| crate::data_tables::relative_atomic_mass(z))
                .collect()
        });
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let mut periodic = self.crystal_pbc_options(Some((1, 1, 1)), true)?;
        periodic.kmesh = KMesh::gamma();
        let reference = crate::pbc::phonon::AnalyticSupercellPhonons::build(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &masses,
            crate::pbc::phonon::PhononSupercellOptions {
                repetitions: [repetitions.0, repetitions.1, repetitions.2],
                max_atoms,
                memory_budget_bytes,
            },
        )
        .map_err(to_py_err)?;
        let result = crate::pbc::spectra::sample_phonon_mesh(&reference).map_err(to_py_err)?;
        let mut value = result.as_json();
        value["electronic_sampling"] =
            serde_json::json!("Gamma of supercell; primitive folding follows repetitions");
        self.crystal_json(py, value)
    }

    #[pyo3(signature=(wavenumbers_cm1,weights,grid_cm1,sigma_cm1=10.0))]
    fn phonon_dos(
        &self,
        py: Python<'_>,
        wavenumbers_cm1: Vec<Vec<f64>>,
        weights: Vec<f64>,
        grid_cm1: Vec<f64>,
        sigma_cm1: f64,
    ) -> PyResult<Py<PyAny>> {
        let value =
            crate::pbc::spectra::phonon_dos(&wavenumbers_cm1, &weights, &grid_cm1, sigma_cm1)
                .map_err(to_py_err)?;
        json_value_to_py(py, value.as_json())
    }

    #[pyo3(signature=(wavenumbers_cm1,weights,temperatures_kelvin,zero_tolerance_cm1=1e-3))]
    fn harmonic_thermodynamics(
        &self,
        py: Python<'_>,
        wavenumbers_cm1: Vec<Vec<f64>>,
        weights: Vec<f64>,
        temperatures_kelvin: Vec<f64>,
        zero_tolerance_cm1: f64,
    ) -> PyResult<Py<PyAny>> {
        let values = crate::pbc::spectra::harmonic_thermodynamics(
            &wavenumbers_cm1,
            &weights,
            &temperatures_kelvin,
            zero_tolerance_cm1,
        )
        .map_err(to_py_err)?;
        json_value_to_py(
            py,
            serde_json::Value::Array(values.iter().map(|v| v.as_json()).collect()),
        )
    }

    /// Model bands on a requested path, using a separately converged k mesh.
    #[pyo3(signature=(numbers,positions,cell,kpoints,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=false))]
    fn band_structure(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        kpoints: Vec<(f64, f64, f64)>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        let points: Vec<_> = kpoints.into_iter().map(|k| [k.0, k.1, k.2]).collect();
        let result = crate::pbc::spectra::pbc_band_structure(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &points,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    #[pyo3(signature=(energies,weights,grid_ev,sigma_ev=0.1,energy_unit="hartree"))]
    fn electronic_dos(
        &self,
        py: Python<'_>,
        energies: Vec<Vec<f64>>,
        weights: Vec<f64>,
        grid_ev: Vec<f64>,
        sigma_ev: f64,
        energy_unit: &str,
    ) -> PyResult<Py<PyAny>> {
        let result = match energy_unit {
            "hartree" | "au" => crate::pbc::spectra::electronic_dos_from_energies(
                &energies, &weights, &grid_ev, sigma_ev,
            ),
            "ev" | "eV" => crate::pbc::spectra::electronic_dos_from_energies_ev(
                &energies, &weights, &grid_ev, sigma_ev,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "electronic DOS energy_unit must be hartree or ev",
                ))
            }
        }
        .map_err(to_py_err)?;
        json_value_to_py(py, result.as_json())
    }

    #[pyo3(signature=(volumes_bohr3,static_energies_hartree,phonons_by_volume_cm1,q_weights,temperatures_kelvin,zero_tolerance_cm1=1e-3))]
    fn quasiharmonic_zero_pressure(
        &self,
        py: Python<'_>,
        volumes_bohr3: Vec<f64>,
        static_energies_hartree: Vec<f64>,
        phonons_by_volume_cm1: Vec<Vec<Vec<f64>>>,
        q_weights: Vec<f64>,
        temperatures_kelvin: Vec<f64>,
        zero_tolerance_cm1: f64,
    ) -> PyResult<Py<PyAny>> {
        let result = crate::pbc::spectra::quasiharmonic_zero_pressure(
            &volumes_bohr3,
            &static_energies_hartree,
            &phonons_by_volume_cm1,
            &q_weights,
            &temperatures_kelvin,
            zero_tolerance_cm1,
        )
        .map_err(to_py_err)?;
        json_value_to_py(
            py,
            serde_json::Value::Array(result.iter().map(|v| v.as_json()).collect()),
        )
    }

    /// Analytic-force/stress fixed-cell or variable-cell periodic relaxation.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=false,
        relax_cell=false,max_iterations=300,force_tolerance=1e-7,stress_tolerance=1e-7,target_pressure=0.0,
        max_atom_step=0.2,max_strain_step=0.025,history=12))]
    fn relax_periodic(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
        relax_cell: bool,
        max_iterations: usize,
        force_tolerance: f64,
        stress_tolerance: f64,
        target_pressure: f64,
        max_atom_step: f64,
        max_strain_step: f64,
        history: usize,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        let settings = crate::pbc::mechanical::PeriodicRelaxOptions {
            relax_cell,
            max_iterations,
            force_tolerance,
            stress_tolerance,
            target_pressure,
            max_atom_step,
            max_strain_step,
            history,
        };
        let result = crate::pbc::mechanical::pbc_relax(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &settings,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    #[pyo3(signature=(numbers,positions,cell,volume_ratios,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=false,
        relax_ions=false,max_iterations=300,force_tolerance=1e-7))]
    fn equation_of_state(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        volume_ratios: Vec<f64>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
        relax_ions: bool,
        max_iterations: usize,
        force_tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        let settings = crate::pbc::mechanical::PeriodicRelaxOptions {
            max_iterations,
            force_tolerance,
            ..Default::default()
        };
        let result = crate::pbc::mechanical::pbc_equation_of_state(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &volume_ratios,
            relax_ions,
            &settings,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    #[pyo3(signature=(volumes_bohr3,energies_hartree))]
    fn fit_equation_of_state(
        &self,
        py: Python<'_>,
        volumes_bohr3: Vec<f64>,
        energies_hartree: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        let result = crate::pbc::mechanical::fit_birch_murnaghan(&volumes_bohr3, &energies_hartree)
            .map_err(to_py_err)?;
        json_value_to_py(py, serde_json::json!(result))
    }

    /// Analytic joint nuclear/affine CP response; no displaced-geometry backend.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=false,
        relax_reference=false,max_iterations=300,force_tolerance=1e-6,stability_tolerance=1e-10))]
    fn elastic_constants(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
        relax_reference: bool,
        max_iterations: usize,
        force_tolerance: f64,
        stability_tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        let settings = crate::pbc::elastic_response::AnalyticElasticOptions {
            relax_reference,
            force_tolerance,
            stability_tolerance,
            relax: crate::pbc::mechanical::PeriodicRelaxOptions {
                max_iterations,
                force_tolerance,
                ..Default::default()
            },
        };
        let result = crate::pbc::elastic_response::pbc_elastic_response(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &settings,
        )
        .map_err(to_py_err)?;
        let mut value = result.as_json();
        value["reference_stationary"] =
            serde_json::json!(result.reference_max_gradient <= force_tolerance);
        value["relaxed_response_available"] = serde_json::json!(result.relaxed_stiffness.is_some());
        value["response_settings"] = serde_json::json!({"relax_reference":relax_reference,"max_iterations":max_iterations,
            "force_tolerance_hartree_per_bohr":force_tolerance,"stability_tolerance_hartree_per_bohr2":stability_tolerance});
        self.crystal_json(py, value)
    }

    /// Berry bulk polarization of the same scalar/ML Bloch Hamiltonian.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=true))]
    fn berry_polarization(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let mut periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        periodic.kmesh.fold_time_reversal = false;
        let berry = crate::pbc::polarization::BerryPolarizationOptions::from_kmesh(periodic.kmesh);
        let result = crate::pbc::polarization::pbc_berry_polarization(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &berry,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py,serde_json::json!({"polarization_au":result.polarization,"polarization_raw_au":result.polarization_raw,
            "polarization_c_per_m2":result.polarization.map(|x|x*crate::pbc::polarization::POLARIZATION_AU_TO_C_PER_M2),
            "quantum_au":result.quantum,"total_phase_raw":result.total_phase_raw,"mesh":result.mesh,
            "min_band_gap_hartree":result.min_band_gap,"volume_bohr3":result.volume,"electronic_temperature_kelvin":self.electronic.electronic_temperature}))
    }

    /// Analytic Born charges and proper clamped-ion piezoelectric response.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=true))]
    fn born_effective_charges(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let mut periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        periodic.kmesh.fold_time_reversal = false;
        let berry = crate::pbc::polarization::BerryPolarizationOptions::from_kmesh(periodic.kmesh);
        let result = crate::pbc::polarization::response::pbc_polarization_derivatives(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &berry,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    /// Clamped-ion electronic dielectric from variational Berry electric enthalpy.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=true))]
    fn dielectric_tensor(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
    ) -> PyResult<Py<PyAny>> {
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let mut periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        periodic.kmesh.fold_time_reversal = false;
        let berry = crate::pbc::polarization::BerryPolarizationOptions::from_kmesh(periodic.kmesh);
        let result = crate::pbc::polarization::response::pbc_electronic_dielectric(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &berry,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    /// Same-reference analytic elastic, Born, piezoelectric, dielectric and IR data.
    #[pyo3(signature=(numbers,positions,cell,unit="angstrom",pbc=(true,true,true),kmesh=None,gamma_centered=true,masses_amu=None,
        relax_reference=false,max_iterations=300,force_tolerance=1e-6,stability_tolerance=1e-10,
        zero_curvature_tolerance=1e-10,zero_stress_tolerance=1e-6))]
    fn electromechanical_response(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        unit: &str,
        pbc: (bool, bool, bool),
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
        masses_amu: Option<Vec<f64>>,
        relax_reference: bool,
        max_iterations: usize,
        force_tolerance: f64,
        stability_tolerance: f64,
        zero_curvature_tolerance: f64,
        zero_stress_tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        let masses = masses_amu.unwrap_or_else(|| {
            numbers
                .iter()
                .map(|&z| crate::data_tables::relative_atomic_mass(z))
                .collect()
        });
        let system = build_periodic_system(
            numbers,
            positions,
            cell,
            pbc,
            unit,
            self.electronic.charge.unwrap_or(0.),
        )?;
        let mut periodic = self.crystal_pbc_options(kmesh, gamma_centered)?;
        periodic.kmesh.fold_time_reversal = false;
        let settings = crate::pbc::electromechanical::ElectromechanicalOptions {
            berry: crate::pbc::polarization::BerryPolarizationOptions::from_kmesh(periodic.kmesh),
            elastic: crate::pbc::elastic_response::AnalyticElasticOptions {
                relax_reference,
                force_tolerance,
                stability_tolerance,
                relax: crate::pbc::mechanical::PeriodicRelaxOptions {
                    max_iterations,
                    force_tolerance,
                    ..Default::default()
                },
            },
            zero_curvature_tolerance,
            zero_stress_tolerance,
        };
        let result = crate::pbc::electromechanical::pbc_electromechanical_response(
            &system,
            &self.params,
            &self.electronic,
            &periodic,
            &masses,
            &settings,
        )
        .map_err(to_py_err)?;
        self.crystal_json(py, result.as_json())
    }

    /// Polar-mode oscillator strengths and stable harmonic ionic dielectric increment.
    #[pyo3(signature=(hessian,born_effective_charges,masses_amu,volume_bohr3,zero_tolerance=1e-10))]
    fn polar_mode_properties(
        &self,
        py: Python<'_>,
        hessian: Vec<Vec<f64>>,
        born_effective_charges: Vec<Vec<Vec<f64>>>,
        masses_amu: Vec<f64>,
        volume_bohr3: f64,
        zero_tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        let h = crystal_response_matrix(hessian, 3 * masses_amu.len())?;
        let born = born_effective_charges
            .into_iter()
            .map(|z| crystal_response_matrix(z, 3))
            .collect::<PyResult<Vec<_>>>()?;
        let result = crate::pbc::electromechanical::polar_mode_properties(
            &h,
            &born,
            &masses_amu,
            volume_bohr3,
            zero_tolerance,
        )
        .map_err(to_py_err)?;
        json_value_to_py(py, result.as_json())
    }

    /// Directional Gamma LO-TO limit with explicit Born and electronic dielectric tensors.
    #[pyo3(signature=(hessian,born_effective_charges,dielectric,volume_bohr3,masses_amu,direction))]
    fn lo_to_modes(
        &self,
        py: Python<'_>,
        hessian: Vec<Vec<f64>>,
        born_effective_charges: Vec<Vec<Vec<f64>>>,
        dielectric: Vec<Vec<f64>>,
        volume_bohr3: f64,
        masses_amu: Vec<f64>,
        direction: (f64, f64, f64),
    ) -> PyResult<Py<PyAny>> {
        let h = crystal_response_matrix(hessian, 3 * masses_amu.len())?;
        let born = born_effective_charges
            .into_iter()
            .map(|z| crystal_response_matrix(z, 3))
            .collect::<PyResult<Vec<_>>>()?;
        let eps = crystal_response_matrix(dielectric, 3)?;
        let result = crate::pbc::electromechanical::lo_to_modes(
            &h,
            &born,
            &eps,
            volume_bohr3,
            &masses_amu,
            [direction.0, direction.1, direction.2],
        )
        .map_err(to_py_err)?;
        json_value_to_py(py, result.as_json())
    }

    #[pyo3(signature=(stiffness,density_kg_m3))]
    fn elastic_properties(
        &self,
        py: Python<'_>,
        stiffness: Vec<Vec<f64>>,
        density_kg_m3: f64,
    ) -> PyResult<Py<PyAny>> {
        let matrix = crystal_stiffness(stiffness)?;
        let result = crate::pbc::mechanical::elastic_properties(&matrix, density_kg_m3)
            .map_err(to_py_err)?;
        json_value_to_py(py, serde_json::json!(result))
    }

    #[pyo3(signature=(stiffness,density_kg_m3,direction,transverse))]
    fn directional_elastic_properties(
        &self,
        py: Python<'_>,
        stiffness: Vec<Vec<f64>>,
        density_kg_m3: f64,
        direction: (f64, f64, f64),
        transverse: (f64, f64, f64),
    ) -> PyResult<Py<PyAny>> {
        let matrix = crystal_stiffness(stiffness)?;
        let result = crate::pbc::mechanical::directional_elastic_response(
            &matrix,
            density_kg_m3,
            [direction.0, direction.1, direction.2],
            [transverse.0, transverse.1, transverse.2],
        )
        .map_err(to_py_err)?;
        json_value_to_py(py, serde_json::json!(result))
    }
}

impl PyGfn1NativeCalculator {
    fn crystal_pbc_options(
        &self,
        kmesh: Option<(usize, usize, usize)>,
        gamma_centered: bool,
    ) -> PyResult<PbcOptions> {
        let mut result = PbcOptions::for_electronic(&self.electronic).map_err(to_py_err)?;
        if let Some((a, b, c)) = kmesh {
            if [a, b, c].contains(&0) {
                return Err(PyValueError::new_err("kmesh dimensions must be positive"));
            }
            result.kmesh = KMesh {
                size: [a, b, c],
                gamma_centered,
                fold_time_reversal: true,
            };
        } else {
            result.kmesh.gamma_centered = gamma_centered;
        }
        Ok(result)
    }

    fn crystal_json(&self, py: Python<'_>, mut value: serde_json::Value) -> PyResult<Py<PyAny>> {
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "ml_model_identity".into(),
                self.ml_identity
                    .as_ref()
                    .map(|identity| identity.as_json())
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        json_value_to_py(py, value)
    }

    fn hessian_options(&self) -> AnalyticHessianOptions {
        AnalyticHessianOptions {
            electronic_options: self.electronic.clone(),
            ..AnalyticHessianOptions::default()
        }
    }

    fn gradient_options(&self) -> AnalyticGradientOptions {
        let mut options = AnalyticGradientOptions::default();
        options.electronic = self.electronic.clone();
        options.electronic.d3_reference_path = self.d3_reference_path.clone();
        options
    }
}

#[pyclass(name = "CalculationResult")]
#[derive(Clone, Debug)]
pub struct PyCalculationResult {
    #[pyo3(get)]
    pub calculation_id: Option<String>,
    solvation_options: Option<crate::SolvationOptions>,
    #[pyo3(get)]
    pub ml_model_sha256: Option<String>,
    #[pyo3(get)]
    pub method_citations: Vec<String>,
    /// Total energy: the **finite-temperature (Mermin) free energy** `E − T·S_elec` — the quantity
    /// the forces/stress differentiate (= the internal energy at `T_elec = 0`). The bare internal
    /// energy is available as `energy_terms_hartree()["total_internal"]`.
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub energy_ev: f64,
    #[pyo3(get)]
    pub gradient_hartree_per_bohr: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub forces_ev_per_angstrom: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub charges: Vec<f64>,
    /// Mulliken dipole moment (atomic units, e*a0) as (x, y, z).
    #[pyo3(get)]
    pub dipole: (f64, f64, f64),
    /// Periodic stress tensor (3x3, atomic units Hartree/bohr^3); None for
    /// non-periodic results or when stress was not requested.
    #[pyo3(get)]
    pub stress: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
    terms: Vec<(String, f64)>,
}

#[pymethods]
impl PyCalculationResult {
    fn solvation_metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(s) = &self.solvation_options else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item(
            "model",
            match s.model {
                crate::SolvationModel::Alpb => "alpb",
                crate::SolvationModel::Gbsa => "gbsa",
            },
        )?;
        dict.set_item("solvent", &s.solvent)?;
        dict.set_item(
            "state",
            match s.state {
                crate::SolutionState::Gsolv => "gsolv",
                crate::SolutionState::Bar1Mol => "bar1mol",
                crate::SolutionState::Reference => "reference",
            },
        )?;
        dict.set_item("temperature_K", s.temperature)?;
        dict.set_item("grid_points", s.grid_points)?;
        dict.set_item(
            "kernel",
            match s.model {
                crate::SolvationModel::Alpb => "P16",
                crate::SolvationModel::Gbsa => "Still",
            },
        )?;
        dict.set_item("reference_commit", crate::solvation::UPSTREAM_COMMIT)?;
        Ok(Some(dict.into()))
    }
    fn energy_terms_hartree(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.terms {
            dict.set_item(key, *value)?;
        }
        Ok(dict.into())
    }

    fn energy_terms_ev(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.terms {
            dict.set_item(key, *value * HARTREE_TO_EV)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "CalculationResult(energy_hartree={:.12}, natoms={}, converged={}, iterations={})",
            self.energy_hartree,
            self.charges.len(),
            self.converged,
            self.iterations
        )
    }
}

impl PyCalculationResult {
    fn from_electronic(result: crate::ElectronicResult) -> Self {
        let terms = energy_terms(&result);
        Self {
            calculation_id: result.calculation_id.clone(),
            solvation_options: result.solvation.as_ref().map(|s| s.options().clone()),
            ml_model_sha256: result
                .ml_parameters
                .as_ref()
                .map(|m| m.model_sha256.clone()),
            method_citations: result
                .method_citations
                .iter()
                .map(|s| s.to_string())
                .collect(),
            energy_hartree: result.total_free,
            energy_ev: result.total_free * HARTREE_TO_EV,
            gradient_hartree_per_bohr: None,
            forces_ev_per_angstrom: None,
            charges: result.atomic_charges,
            dipole: (result.dipole.x, result.dipole.y, result.dipole.z),
            stress: None,
            iterations: result.iterations,
            converged: result.converged,
            terms,
        }
    }

    fn from_gradient(result: crate::AnalyticGradientResult) -> Self {
        let electronic = result.electronic_result.clone();
        let terms = energy_terms(&electronic);
        let dipole = electronic.dipole;
        Self {
            calculation_id: electronic.calculation_id.clone(),
            solvation_options: electronic.solvation.as_ref().map(|s| s.options().clone()),
            ml_model_sha256: electronic
                .ml_parameters
                .as_ref()
                .map(|m| m.model_sha256.clone()),
            method_citations: electronic
                .method_citations
                .iter()
                .map(|s| s.to_string())
                .collect(),
            energy_hartree: electronic.total_free,
            energy_ev: electronic.total_free * HARTREE_TO_EV,
            gradient_hartree_per_bohr: Some(vec3_list(&result.gradient)),
            forces_ev_per_angstrom: Some(
                result
                    .forces
                    .iter()
                    .map(|v| {
                        vec![
                            v.x * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                            v.y * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                            v.z * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                        ]
                    })
                    .collect(),
            ),
            charges: electronic.atomic_charges,
            dipole: (dipole.x, dipole.y, dipole.z),
            stress: None,
            iterations: electronic.iterations,
            converged: electronic.converged,
            terms,
        }
    }
}

#[pyclass(name = "OptimizationResult")]
#[derive(Clone, Debug)]
pub struct PyOptimizationResult {
    #[pyo3(get)]
    pub energy_hartree: f64,
    #[pyo3(get)]
    pub energy_ev: f64,
    #[pyo3(get)]
    pub positions_angstrom: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub gradient_hartree_per_bohr: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub forces_ev_per_angstrom: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub max_gradient: f64,
    /// Atomic numbers, in order (for writing XYZ).
    #[pyo3(get)]
    pub numbers: Vec<u8>,
    /// L-BFGS trajectory geometries (frames x atoms x 3, Angstrom); frame 0 is the input.
    #[pyo3(get)]
    pub trajectory_positions_angstrom: Vec<Vec<Vec<f64>>>,
    /// Per-step total energies along the trajectory (Hartree).
    #[pyo3(get)]
    pub trajectory_energies_hartree: Vec<f64>,
}

impl PyOptimizationResult {
    fn from_result(result: GeometryOptimizationResult) -> Self {
        Self {
            energy_hartree: result.energy,
            energy_ev: result.energy * HARTREE_TO_EV,
            positions_angstrom: result
                .system
                .atoms
                .iter()
                .map(|atom| {
                    vec![
                        atom.position.x * BOHR_TO_ANGSTROM,
                        atom.position.y * BOHR_TO_ANGSTROM,
                        atom.position.z * BOHR_TO_ANGSTROM,
                    ]
                })
                .collect(),
            numbers: result.system.atoms.iter().map(|a| a.z).collect(),
            trajectory_positions_angstrom: result
                .trajectory
                .iter()
                .map(|step| {
                    step.positions
                        .iter()
                        .map(|p| {
                            vec![
                                p.x * BOHR_TO_ANGSTROM,
                                p.y * BOHR_TO_ANGSTROM,
                                p.z * BOHR_TO_ANGSTROM,
                            ]
                        })
                        .collect()
                })
                .collect(),
            trajectory_energies_hartree: result.trajectory.iter().map(|s| s.energy).collect(),
            gradient_hartree_per_bohr: vec3_list(&result.gradient),
            forces_ev_per_angstrom: result
                .forces
                .iter()
                .map(|v| {
                    vec![
                        v.x * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                        v.y * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                        v.z * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                    ]
                })
                .collect(),
            iterations: result.iterations,
            converged: result.converged,
            max_gradient: result.max_gradient,
        }
    }
}

/// One XYZ frame from atomic numbers + Angstrom positions.
fn xyz_frame(numbers: &[u8], positions: &[Vec<f64>], comment: &str) -> String {
    let mut out = format!("{}\n{}\n", numbers.len(), comment);
    for (z, p) in numbers.iter().zip(positions) {
        let sym = crate::z_to_symbol(*z).unwrap_or("X");
        out.push_str(&format!(
            "{sym:2} {:18.10} {:18.10} {:18.10}\n",
            p[0], p[1], p[2]
        ));
    }
    out
}

#[pymethods]
impl PyOptimizationResult {
    /// Final optimized geometry as an XYZ string (Angstrom).
    #[pyo3(signature = (comment=None))]
    fn to_xyz(&self, comment: Option<String>) -> String {
        let c = comment.unwrap_or_else(|| "gfn1-rs optimized geometry".to_string());
        xyz_frame(&self.numbers, &self.positions_angstrom, &c)
    }

    /// Full L-BFGS trajectory as a multi-frame XYZ string (one frame per step, Angstrom),
    /// with the per-step iteration + energy (Hartree) in each comment line. Write it to a
    /// `.xyz` file to view the optimization path in any trajectory viewer.
    fn trajectory_xyz(&self) -> String {
        let mut out = String::new();
        for (i, (frame, e)) in self
            .trajectory_positions_angstrom
            .iter()
            .zip(&self.trajectory_energies_hartree)
            .enumerate()
        {
            out.push_str(&xyz_frame(
                &self.numbers,
                frame,
                &format!("iter {i} energy {e:.10} Ha"),
            ));
        }
        out
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::python_resources::register(m)?;
    m.add_function(pyo3::wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(training_main, m)?)?;
    m.add_class::<PyGfn1NativeCalculator>()?;
    m.add_class::<PyCalculationResult>()?;
    m.add_class::<PyOptimizationResult>()?;
    m.add("HARTREE_TO_EV", HARTREE_TO_EV)?;
    m.add("BOHR_TO_ANGSTROM", BOHR_TO_ANGSTROM)?;
    m.add("ANGSTROM_TO_BOHR", ANGSTROM_TO_BOHR)?;
    m.add(
        "FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
        FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    )?;
    m.add(
        "HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2",
        HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2,
    )?;
    // The degree-of-freedom cap the expanded n^4 quartic bindings enforce (shared with the
    // analytic D3 fourth derivative). Exported so callers can size a block instead of failing.
    m.add(
        "MAX_FOURTH_DERIVATIVE_NDOF",
        crate::MAX_FOURTH_DERIVATIVE_NDOF,
    )?;
    m.add_function(pyo3::wrap_pyfunction!(roundtrip_param_file, m)?)?;
    Ok(())
}

#[pyfunction]
fn cli_main(py: Python<'_>, arguments: Vec<String>, training: bool) -> i32 {
    py.allow_threads(|| crate::cli::main(arguments, training))
}

#[pyfunction]
fn training_main(py: Python<'_>, arguments: Vec<String>) -> PyResult<()> {
    py.allow_threads(|| crate::cli::train_model(arguments))
        .map_err(to_py_err)
}

/// Convert a dense [`crate::linalg::Matrix`] to a row-major nested `Vec<Vec<f64>>` for
/// returning raw AO-integral tensors to Python.
fn matrix_to_nested(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Expand the packed [`crate::third_derivative::SymmetricThird`] into the **`n³` nested list**
/// Python receives, `out[c][a][b] = T_abc`.
///
/// The packed store keeps only the `n(n+1)(n+2)/6` canonical entries; expanding costs ~6x the
/// memory but hands Python a plain rectangular nesting that `numpy.asarray` turns into an
/// `(n, n, n)` array with no unpacking step. The tensor is fully symmetric, so the `[c][a][b]`
/// layout (chosen to match the pre-existing dense-slab bindings) is the same data as `[a][b][c]`.
fn packed_derivative_dict(py: Python<'_>, file: &crate::PackedDerivativeMetadata) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    out.set_item("path", &file.path)?;
    out.set_item("order", file.order)?;
    out.set_item("n_dof", file.n_dof)?;
    out.set_item("packed_entries", file.packed_entries)?;
    out.set_item("units", file.units())?;
    out.set_item("offset_bytes", 24)?;
    Ok(out.into_any().unbind())
}

fn symmetric_third_to_nested(t: &crate::third_derivative::SymmetricThird) -> Vec<Vec<Vec<f64>>> {
    let n = t.n();
    (0..n)
        .map(|c| {
            (0..n)
                .map(|a| (0..n).map(|b| t.get(a, b, c)).collect())
                .collect()
        })
        .collect()
}

/// Expand the packed [`crate::fourth_derivative::SymmetricFourth`] into the **`n⁴` nested list**
/// Python receives, `out[a][b][c][d] = Q_abcd` (fully symmetric, so the index order is
/// immaterial). Guarded by [`check_fourth_dense_size`] at every call site.
fn symmetric_fourth_to_nested(
    q: &crate::fourth_derivative::SymmetricFourth,
) -> Vec<Vec<Vec<Vec<f64>>>> {
    let n = q.n();
    (0..n)
        .map(|a| {
            (0..n)
                .map(|b| {
                    (0..n)
                        .map(|c| (0..n).map(|d| q.get(a, b, c, d)).collect())
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Reject an expanded `n⁴` quartic above the crate cap [`crate::MAX_FOURTH_DERIVATIVE_NDOF`].
///
/// The cap is shared with the analytic D3 quartic (where it bounds the `ndof⁵` Jet4 working set);
/// here it additionally bounds the `8·n⁴`-byte Python expansion and the `~n⁴/24` directional
/// evaluations behind it. It is applied to `|dofs|` for the block mode, so a small block of a
/// large molecule stays legal.
fn check_fourth_dense_size(n: usize, who: &str) -> PyResult<()> {
    if n > crate::MAX_FOURTH_DERIVATIVE_NDOF {
        let mb = (n as f64).powi(4) * 8.0 / (1024.0 * 1024.0);
        return Err(PyValueError::new_err(format!(
            "{who}: the expanded n^4 quartic is capped at {} degrees of freedom ({} atoms); got \
             {n}, which would materialize {mb:.0} MB of nested lists and cost ~{} directional \
             evaluations. Use fourth_derivative_block over a DOF subset, or the directional mode",
            crate::MAX_FOURTH_DERIVATIVE_NDOF,
            crate::MAX_FOURTH_DERIVATIVE_NDOF / 3,
            n.pow(4) / 24
        )));
    }
    Ok(())
}

/// [`PbcOptions`] on an explicit Monkhorst-Pack mesh, cutoffs as in [`pbc_gamma_options`].
/// `kgrid = None` or `(1,1,1)` collapses to the Γ-only mesh, which every k-point entry point
/// accepts and reproduces the Γ path on.
fn pbc_kmesh_options(
    kgrid: Option<(usize, usize, usize)>,
    ao_cutoff: Option<f64>,
    ewald_real_cutoff: Option<f64>,
    ewald_sr_cutoff: Option<f64>,
) -> PbcOptions {
    let kmesh = match kgrid {
        None | Some((1, 1, 1)) => KMesh::gamma(),
        Some((a, b, c)) => KMesh::monkhorst_pack([a.max(1), b.max(1), c.max(1)]),
    };
    PbcOptions {
        kmesh,
        ..pbc_gamma_options(ao_cutoff, ewald_real_cutoff, ewald_sr_cutoff)
    }
}

/// Γ-point [`PbcOptions`] for the periodic third-derivative / Grüneisen entry points, with the
/// three real-space cutoffs (bohr) optionally overridden. The library defaults (`ao_cutoff` 30,
/// Ewald `real_cutoff` 40, `sr_cutoff` 10) are conservative; lowering them is the standard speed
/// lever for these `O(N)`-Hessian sweeps.
fn pbc_gamma_options(
    ao_cutoff: Option<f64>,
    ewald_real_cutoff: Option<f64>,
    ewald_sr_cutoff: Option<f64>,
) -> PbcOptions {
    let mut options = PbcOptions {
        kmesh: KMesh::gamma(),
        ..PbcOptions::default()
    };
    if let Some(value) = ao_cutoff {
        options.ao_cutoff = value;
    }
    if let Some(value) = ewald_real_cutoff {
        options.ewald.real_cutoff = value;
    }
    if let Some(value) = ewald_sr_cutoff {
        options.ewald.sr_cutoff = value;
    }
    options
}

fn build_system(
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    unit: &str,
    charge: f64,
) -> PyResult<PeriodicSystem> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err(format!(
            "numbers has length {}, positions has length {}",
            numbers.len(),
            positions.len()
        )));
    }
    let scale = match unit {
        "angstrom" | "Angstrom" | "A" => ANGSTROM_TO_BOHR,
        "bohr" | "Bohr" | "au" => 1.0,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported coordinate unit `{other}`"
            )))
        }
    };
    let mut atoms = Vec::with_capacity(numbers.len());
    for (idx, (z, xyz)) in numbers.into_iter().zip(positions.into_iter()).enumerate() {
        if xyz.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions[{idx}] has length {}, expected 3",
                xyz.len()
            )));
        }
        atoms.push(Atom {
            z,
            position: Vec3::new(xyz[0] * scale, xyz[1] * scale, xyz[2] * scale),
        });
    }
    Ok(PeriodicSystem::new(atoms, None).with_charge(charge))
}

fn unit_scale(unit: &str) -> PyResult<f64> {
    match unit {
        "angstrom" | "Angstrom" | "A" => Ok(ANGSTROM_TO_BOHR),
        "bohr" | "Bohr" | "au" => Ok(1.0),
        other => Err(PyValueError::new_err(format!(
            "unsupported coordinate unit `{other}`"
        ))),
    }
}

fn build_periodic_system(
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: (bool, bool, bool),
    unit: &str,
    charge: f64,
) -> PyResult<PeriodicSystem> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err(format!(
            "numbers has length {}, positions has length {}",
            numbers.len(),
            positions.len()
        )));
    }
    let scale = unit_scale(unit)?;
    let mut atoms = Vec::with_capacity(numbers.len());
    for (idx, (z, xyz)) in numbers.into_iter().zip(positions.into_iter()).enumerate() {
        if xyz.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "positions[{idx}] has length {}, expected 3",
                xyz.len()
            )));
        }
        atoms.push(Atom {
            z,
            position: Vec3::new(xyz[0] * scale, xyz[1] * scale, xyz[2] * scale),
        });
    }
    if cell.len() != 3 || cell.iter().any(|row| row.len() != 3) {
        return Err(PyValueError::new_err(
            "cell must be a 3x3 list of lattice vectors (rows a, b, c)".to_string(),
        ));
    }
    let vec = |row: &Vec<f64>| Vec3::new(row[0] * scale, row[1] * scale, row[2] * scale);
    let lattice = Lattice::new(
        Mat3::from_columns(vec(&cell[0]), vec(&cell[1]), vec(&cell[2])),
        [pbc.0, pbc.1, pbc.2],
    )
    .map_err(to_py_err)?;
    Ok(PeriodicSystem::new(atoms, Some(lattice)).with_charge(charge))
}

fn vec3_list(values: &[Vec3]) -> Vec<Vec<f64>> {
    values.iter().map(|v| vec![v.x, v.y, v.z]).collect()
}

fn fragment_result_dict(
    py: Python<'_>,
    result: &crate::fragment::FragmentResult,
    model: &str,
    derivatives: bool,
    temperature: f64,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model", model)?;
    dict.set_item("electronic_temperature_kelvin", temperature)?;
    dict.set_item("total_free_hartree", result.total_free)?;
    dict.set_item("shell_charges", &result.shell_charges)?;
    dict.set_item(
        "gradient",
        if derivatives {
            Some(vec3_list(&result.gradient))
        } else {
            None
        },
    )?;
    dict.set_item(
        "forces",
        if derivatives {
            Some(vec3_list(&result.forces))
        } else {
            None
        },
    )?;
    dict.set_item(
        "stress",
        if derivatives {
            result.stress.as_ref().map(matrix_to_nested)
        } else {
            None
        },
    )?;
    dict.set_item("embedding_iterations", result.embedding_iterations)?;
    dict.set_item("local_scc_iterations", result.local_scc_iterations)?;
    dict.set_item("cluster_count", result.cluster_count)?;
    dict.set_item("largest_cluster_ao", result.largest_cluster_ao)?;
    Ok(dict.into())
}

fn parse_targets(targets: &[String]) -> PyResult<Vec<ParameterTarget>> {
    targets
        .iter()
        .map(|t| ParameterTarget::parse(t))
        .collect::<crate::error::Result<Vec<_>>>()
        .map_err(to_py_err)
}

fn origin_vec(origin: Option<(f64, f64, f64)>) -> Vec3 {
    match origin {
        Some((x, y, z)) => Vec3::new(x, y, z),
        None => Vec3::zero(),
    }
}

/// Round-trip a `param_gfn1-xtb.txt` file: parse `in_path` and write the
/// canonical, value-exact serialization to `out_path`.
#[pyfunction]
fn roundtrip_param_file(in_path: String, out_path: String) -> PyResult<()> {
    let params = Gfn1Parameters::from_file(&in_path).map_err(to_py_err)?;
    params.write_param_file(&out_path).map_err(to_py_err)?;
    Ok(())
}

fn energy_terms(result: &crate::ElectronicResult) -> Vec<(String, f64)> {
    result
        .energy_terms()
        .named_values()
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect()
}

fn to_py_err(err: Gfn1Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Shared conversion for the typed crystal-property JSON views.
pub(crate) fn json_value_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("json")?
        .call_method1("loads", (value.to_string(),))?
        .unbind())
}

fn crystal_response_matrix(rows: Vec<Vec<f64>>, n: usize) -> PyResult<crate::linalg::Matrix> {
    if n == 0 || rows.len() != n || rows.iter().any(|r| r.len() != n) {
        return Err(PyValueError::new_err(format!(
            "expected a {n} by {n} response tensor"
        )));
    }
    crate::resources::check_dense_workspace(n, n, 1, "response input").map_err(to_py_err)?;
    crate::linalg::Matrix::from_vec(n, n, rows.into_iter().flatten().collect()).map_err(to_py_err)
}

fn crystal_stiffness(rows: Vec<Vec<f64>>) -> PyResult<crate::linalg::Matrix> {
    if rows.len() != 6 || rows.iter().any(|r| r.len() != 6) {
        return Err(PyValueError::new_err(
            "stiffness must be a 6x6 engineering-Voigt matrix in Hartree/bohr^3",
        ));
    }
    let mut result = crate::linalg::Matrix::zeros(6, 6);
    for i in 0..6 {
        for j in 0..6 {
            result[(i, j)] = rows[i][j];
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn parse_excited_spin(spin: &str) -> PyResult<TdaSpin> {
    match spin.to_ascii_lowercase().as_str() {
        "singlet" | "s" => Ok(TdaSpin::Singlet),
        "triplet" | "t" => Ok(TdaSpin::Triplet),
        _ => Err(PyValueError::new_err("spin must be singlet or triplet")),
    }
}

fn set_manifold_metadata(dict: &Bound<'_, PyDict>, roots: &std::ops::Range<usize>) -> PyResult<()> {
    dict.set_item("manifold_first_root", roots.start)?;
    dict.set_item("manifold_size", roots.len())?;
    dict.set_item("manifold_status", tda_manifold_status(roots.len()))
}

fn stda_occupation_model(thermal: bool) -> &'static str {
    if thermal {
        "fractional canonical ensemble interband CI"
    } else {
        "fixed closed-shell CI partition; canonical thermal VTB updates"
    }
}

fn tda_manifold_status(size: usize) -> &'static str {
    if size == 1 { "isolated_root" } else { "degenerate_mean" }
}

fn excited_hessian_dict(
    py: Python<'_>,
    model: &str,
    state: usize,
    temperature: f64,
    ground_energy: f64,
    excitation_energy: f64,
    ground: &crate::linalg::Matrix,
    excitation: &crate::linalg::Matrix,
    total: &crate::linalg::Matrix,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    result.set_item("implementation", "native-rust")?;
    result.set_item("method", "analytic")?;
    result.set_item("model", model)?;
    result.set_item("state", state)?;
    result.set_item("electronic_temperature_kelvin", temperature)?;
    result.set_item("ground_reference", "GFN1 free energy")?;
    result.set_item("ground_energy_hartree", ground_energy)?;
    result.set_item("excitation_energy_hartree", excitation_energy)?;
    result.set_item("total_energy_hartree", ground_energy + excitation_energy)?;
    for (name, matrix) in [
        ("hessian", total),
        ("ground_hessian", ground),
        ("excitation_hessian", excitation),
    ] {
        let value = PyList::new(py, matrix_to_nested(matrix))?;
        result.set_item(name, &value)?;
        result.set_item(format!("{name}_hartree_per_bohr2"), value)?;
    }
    Ok(result.into())
}
