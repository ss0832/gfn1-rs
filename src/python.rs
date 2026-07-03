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
    pbc_electronic_result, pbc_gamma_hessian, pbc_gradient_from_scc, pbc_kpoint_hessian,
    pbc_stress_from_scc, run_pbc_scc_with_guess, KMesh, PbcOptions, PbcSccGuess,
};
use crate::properties::{dipole_derivatives, ir_spectrum, raman_spectrum, static_polarizability};
use crate::system::{Atom, PeriodicSystem};
use crate::td::{
    solve_tda, solve_tda_gradient_method, solve_tda_kpoint, solve_tda_kpoint_gradient,
    solve_tda_kpoint_gradient_analytic, tda_optical_rotation, tda_rotatory_strengths,
    TdaGradientMethod, TdaOptions, TdaSpin,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

fn parse_accelerator(name: Option<&str>) -> PyResult<SccAccelerator> {
    match name.map(|s| s.to_ascii_lowercase()) {
        None => Ok(SccAccelerator::Broyden),
        Some(s) => match s.as_str() {
            "broyden" => Ok(SccAccelerator::Broyden),
            "linear" => Ok(SccAccelerator::Linear),
            "cdiis" => Ok(SccAccelerator::Cdiis),
            "newton" => Ok(SccAccelerator::Newton),
            other => Err(PyValueError::new_err(format!(
                "unknown scc_accelerator `{other}` (use broyden/linear/cdiis/newton)"
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
}

#[pymethods]
impl PyGfn1NativeCalculator {
    #[new]
    #[pyo3(signature = (
        param_path,
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
        plus_u_all_d = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        param_path: String,
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
    ) -> PyResult<Self> {
        let params = Gfn1Parameters::from_file(&param_path).map_err(to_py_err)?;
        let mut electronic = ElectronicOptions::default();
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
        })
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
        unit = "angstrom", compute_gradient = false, compute_stress = false
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
    ) -> PyResult<PyCalculationResult> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let kmesh = match kgrid {
            None | Some((1, 1, 1)) => KMesh::gamma(),
            Some((a, b, c)) => KMesh::monkhorst_pack([a, b, c]),
        };
        let pbc_options = PbcOptions {
            kmesh,
            ..PbcOptions::default()
        };
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

    /// Faraday / MCD tensor `d alpha_ij / d B_k` (atomic units), indexed `[k][i][j]`, by
    /// finite difference of the magnetic-field polarizability about `B = 0`. Returns a
    /// `3x3x3` nested list. NOTE: this is **identically zero in the GFN1 monopole model**
    /// (`dq/dB = 0` by time reversal); it is the correct general `d alpha / d B` raw
    /// tensor and would be nonzero for a dipole-coupled (length-gauge) model — see
    /// `lao_dipole`. `e_step`/`b_step` are the field steps; `m1_basis_path` selects M1.
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
    /// Hellmann-Feynman analytic gradient (one SCC + cheap integral derivatives);
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
        let g = match method.trim().to_ascii_lowercase().replace('-', "_").as_str() {
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
    /// `transition_dipoles` (atomic units). Non-periodic only.
    #[pyo3(signature = (numbers, positions, unit = "angstrom", n_states = 5, spin = "singlet"))]
    fn tda(
        &self,
        py: Python<'_>,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        unit: &str,
        n_states: usize,
        spin: &str,
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
        let result = solve_tda(
            &system,
            &self.params,
            &electronic,
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

    /// TD-GFN1 excited-state gradient. `method` selects the algorithm:
    /// `"semi_numerical"` (default) = analytic ground gradient + finite difference
    /// of the frozen-amplitude excitation energy (recommended; exact for a tracked
    /// state, non-periodic); `"fd"` = full finite difference with root tracking
    /// (robust across state crossings; the only option for periodic Gamma-point
    /// cells); `"analytic"` = fully analytic direct-CPHF gradient (exact, matches
    /// finite difference to FD precision; non-periodic, one 3N CPHF solve).
    /// Periodic systems automatically fall back to `"fd"`. Returns a dict with
    /// `total_energy_hartree`, `excitation_energy_hartree`, `gradient` and
    /// `forces` (Hartree/bohr).
    #[pyo3(signature = (
        numbers, positions, unit = "angstrom", state = 0, n_states = 5,
        spin = "singlet", step = 1.0e-3, method = "semi_numerical",
        cell = None, pbc = (true, true, true)
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
        let mut grad_method = TdaGradientMethod::parse(method).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown TDA gradient method `{method}` (use semi_numerical, fd, or analytic)"
            ))
        })?;
        // Periodic (Gamma-point) systems support the finite-difference and the fully
        // analytic paths; the semi-numerical hybrid is non-periodic, so fall back to
        // finite difference for it.
        if system.lattice.is_some() && grad_method == TdaGradientMethod::SemiNumerical {
            grad_method = TdaGradientMethod::FiniteDifference;
        }
        let g = solve_tda_gradient_method(
            &system,
            &self.params,
            &self.electronic,
            state,
            TdaOptions {
                n_states: n_states.max(state + 1),
                spin,
            },
            step,
            grad_method,
        )
        .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        dict.set_item("total_energy_hartree", g.total_energy)?;
        dict.set_item("excitation_energy_hartree", g.excitation_energy)?;
        dict.set_item("gradient", vec3_list(&g.gradient))?;
        dict.set_item("forces", vec3_list(&g.forces))?;
        Ok(dict.into())
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
    #[pyo3(signature = (numbers, positions, cell, pbc = (true, true, true), kgrid = None, unit = "angstrom"))]
    #[allow(clippy::too_many_arguments)]
    fn hessian_periodic(
        &self,
        numbers: Vec<u8>,
        positions: Vec<Vec<f64>>,
        cell: Vec<Vec<f64>>,
        pbc: (bool, bool, bool),
        kgrid: Option<(usize, usize, usize)>,
        unit: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let charge = self.electronic.charge.unwrap_or(0.0);
        let system = build_periodic_system(numbers, positions, cell, pbc, unit, charge)?;
        let kmesh = match kgrid {
            None | Some((1, 1, 1)) => KMesh::gamma(),
            Some((a, b, c)) => KMesh::monkhorst_pack([a, b, c]),
        };
        let pbc_options = PbcOptions {
            kmesh,
            ..PbcOptions::default()
        };
        let options = self.electronic.clone();
        let is_gamma = matches!(kgrid, None | Some((1, 1, 1)));
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
    #[pyo3(signature = (numbers, positions, unit = "angstrom"))]
    fn third_derivative(
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
        let slabs = crate::third_derivative::third_derivative_analytic(
            &system,
            &self.params,
            options,
            cutoff,
        )
        .map_err(to_py_err)?;
        let mut out = Vec::with_capacity(slabs.len());
        for slab in &slabs {
            let n = slab.rows();
            let mut rows = Vec::with_capacity(n);
            for i in 0..n {
                rows.push((0..n).map(|j| slab[(i, j)]).collect());
            }
            out.push(rows);
        }
        Ok(out)
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
}

impl PyGfn1NativeCalculator {
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
    m.add_function(pyo3::wrap_pyfunction!(roundtrip_param_file, m)?)?;
    Ok(())
}

/// Convert a dense [`crate::linalg::Matrix`] to a row-major nested `Vec<Vec<f64>>` for
/// returning raw AO-integral tensors to Python.
fn matrix_to_nested(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
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
