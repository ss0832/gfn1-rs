// SPDX-License-Identifier: GPL-3.0-or-later
//! Crystal spectra and thermodynamics from the existing analytic periodic model.
//!
//! Phonons reuse one analytic supercell Hessian; evaluating a commensurate mesh,
//! DOS and harmonic thermodynamics introduces no nuclear finite differences.
//! Electronic bands resample the converged learned/scalar Bloch Hamiltonian,
//! rather than rebuilding a bare GFN1 Hamiltonian after an ML calculation.
use super::complex::hermitian_generalized_eigen;
use super::phonon::AnalyticSupercellPhonons;
use super::scf::fock_at_k;
use super::{run_pbc_scc, PbcOptions, PbcSccResult};
use crate::constants::{HARTREE_TO_EV, HARTREE_TO_WAVENUMBER, KB_HARTREE_PER_K};
use crate::linalg::{DenseLu, Matrix};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};
use rayon::prelude::*;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct PhononMeshResult {
    pub q_fractional: Vec<[f64; 3]>,
    /// Normalized Brillouin-zone weights. Modes are counted per primitive cell.
    pub weights: Vec<f64>,
    pub wavenumbers_cm1: Vec<Vec<f64>>,
    pub eigenvalues_hartree_per_bohr2_amu: Vec<Vec<f64>>,
    pub raw_hermiticity_errors: Vec<f64>,
    pub electronic_temperature_kelvin: f64,
    pub reference_energy_hartree_per_primitive_cell: f64,
    pub ml_model_sha256: Option<String>,
    pub supercell_repetitions: [usize; 3],
    pub method_citations: Vec<&'static str>,
}
impl PhononMeshResult {
    pub fn as_json(&self) -> Value {
        json!({"backend":"analytic commensurate supercell CP-SCC",
            "q_fractional":self.q_fractional,"weights":self.weights,
            "wavenumbers_cm1":self.wavenumbers_cm1,
            "eigenvalues_hartree_per_bohr2_amu":self.eigenvalues_hartree_per_bohr2_amu,
            "raw_hermiticity_errors":self.raw_hermiticity_errors,
            "electronic_temperature_kelvin":self.electronic_temperature_kelvin,
            "reference_energy_hartree_per_primitive_cell":self.reference_energy_hartree_per_primitive_cell,
            "ml_model_sha256":self.ml_model_sha256,
            "supercell_repetitions":self.supercell_repetitions,
            "normalization":"per primitive cell; q weights sum to one",
            "method_citations":self.method_citations})
    }
}

/// Evaluate all q points commensurate with the already built supercell.
/// The electronic sampling remains the sampling used to build `reference`.
pub fn sample_phonon_mesh(reference: &AnalyticSupercellPhonons) -> Result<PhononMeshResult> {
    let reps = reference.repetitions;
    let count = reps
        .iter()
        .try_fold(1usize, |n, &r| n.checked_mul(r))
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("phonon mesh dimensions overflow or contain zero"))?;
    crate::resources::check_dense_workspace(count, 3, 2, "phonon q mesh")?;
    let mut q = Vec::with_capacity(count);
    for a in 0..reps[0] {
        for b in 0..reps[1] {
            for c in 0..reps[2] {
                q.push([
                    a as f64 / reps[0] as f64,
                    b as f64 / reps[1] as f64,
                    c as f64 / reps[2] as f64,
                ]);
            }
        }
    }
    sample_phonon_qpoints(reference, &q, &vec![1.0 / count as f64; count])
}

/// Evaluate an explicitly weighted set of commensurate q points.
pub fn sample_phonon_qpoints(
    reference: &AnalyticSupercellPhonons,
    q_fractional: &[[f64; 3]],
    weights: &[f64],
) -> Result<PhononMeshResult> {
    validate_weights(q_fractional.len(), weights)?;
    let copies = reference
        .repetitions
        .iter()
        .try_fold(1usize, |n, &r| n.checked_mul(r))
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("invalid phonon reference repetition count"))?;
    let modes = reference.cartesian_hessian.rows() / copies;
    if modes == 0
        || reference.cartesian_hessian.rows() % copies != 0
        || reference.cartesian_hessian.rows() != reference.cartesian_hessian.cols()
    {
        return Err(invalid(
            "phonon reference Hessian and repetitions are inconsistent",
        ));
    }
    crate::resources::check_dense_workspace(
        q_fractional.len(),
        modes.max(3),
        4,
        "phonon mesh output",
    )?;
    // A single dynamical matrix/eigenvector workspace is alive at once. Do not
    // multiply the memory demand by parallelizing q diagonalizations here.
    let mut frequencies = Vec::with_capacity(q_fractional.len());
    let mut eigenvalues = Vec::with_capacity(q_fractional.len());
    let mut errors = Vec::with_capacity(q_fractional.len());
    let mut citations = Vec::new();
    for &q in q_fractional {
        let value = reference.at_q(q)?;
        frequencies.push(value.wavenumbers_cm1);
        eigenvalues.push(value.eigenvalues_hartree_per_bohr2_amu);
        errors.push(value.raw_hermiticity_error);
        if citations.is_empty() {
            citations = value.method_citations;
        }
    }
    Ok(PhononMeshResult {
        q_fractional: q_fractional.to_vec(),
        weights: weights.to_vec(),
        wavenumbers_cm1: frequencies,
        eigenvalues_hartree_per_bohr2_amu: eigenvalues,
        raw_hermiticity_errors: errors,
        electronic_temperature_kelvin: reference.electronic_temperature_kelvin,
        reference_energy_hartree_per_primitive_cell: reference
            .reference_energy_hartree_per_primitive_cell,
        ml_model_sha256: reference.ml_model_sha256.clone(),
        supercell_repetitions: reference.repetitions,
        method_citations: citations,
    })
}

#[derive(Clone, Debug)]
pub struct GaussianDosResult {
    pub grid: Vec<f64>,
    pub density: Vec<f64>,
    pub sigma: f64,
    pub expected_states: f64,
    /// Trapezoidal integral on the requested finite grid, without renormalizing tails.
    pub integrated_states: f64,
    pub grid_unit: &'static str,
    pub density_unit: &'static str,
}
impl GaussianDosResult {
    pub fn as_json(&self) -> Value {
        json!({"grid":self.grid,"density":self.density,"sigma":self.sigma,
            "expected_states":self.expected_states,"integrated_states":self.integrated_states,
            "grid_unit":self.grid_unit,"density_unit":self.density_unit,
            "broadening":"normalized Gaussian; finite-grid tails are not renormalized"})
    }
}

/// Signed frequencies retain imaginary branches in a DOS. Thermodynamics below
/// separately rejects genuine imaginary modes instead of treating them as stable.
pub fn phonon_dos(
    wavenumbers_cm1: &[Vec<f64>],
    weights: &[f64],
    grid_cm1: &[f64],
    sigma_cm1: f64,
) -> Result<GaussianDosResult> {
    gaussian_dos(
        wavenumbers_cm1,
        weights,
        grid_cm1,
        sigma_cm1,
        1.0,
        "cm^-1",
        "states/(cm^-1 primitive_cell)",
    )
}

fn gaussian_dos(
    values: &[Vec<f64>],
    weights: &[f64],
    grid: &[f64],
    sigma: f64,
    multiplicity: f64,
    grid_unit: &'static str,
    density_unit: &'static str,
) -> Result<GaussianDosResult> {
    let bands = validate_values(values, weights)?;
    if grid.len() < 2
        || grid.iter().any(|x| !x.is_finite())
        || grid
            .windows(2)
            .any(|v| !((v[1] - v[0]).is_finite()) || v[1] <= v[0])
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(invalid(
            "DOS requires an increasing finite grid and a positive finite Gaussian width",
        ));
    }
    crate::resources::check_dense_workspace(grid.len(), 1, 2, "Gaussian DOS output")?;
    let prefactor = (multiplicity / (2.0 * std::f64::consts::PI).sqrt()) / sigma;
    let density: Vec<f64> = grid
        .par_iter()
        .map(|&x| {
            values
                .iter()
                .zip(weights)
                .map(|(row, &weight)| {
                    row.iter()
                        .map(|&level| {
                            let difference = x - level;
                            let y = if difference.is_finite() {
                                difference / sigma
                            } else {
                                x / sigma - level / sigma
                            };
                            (-0.5 * y * y).exp()
                        })
                        .sum::<f64>()
                        * weight
                })
                .sum::<f64>()
                * prefactor
        })
        .collect();
    if density.iter().any(|v| !v.is_finite()) {
        return Err(invalid("DOS exceeds floating-point range"));
    }
    let integral: f64 = grid
        .windows(2)
        .zip(density.windows(2))
        .map(|(x, y)| (x[1] - x[0]) * (y[0] + y[1]) * 0.5)
        .sum();
    if !integral.is_finite() {
        return Err(invalid("DOS grid integral exceeds floating-point range"));
    }
    Ok(GaussianDosResult {
        grid: grid.to_vec(),
        density,
        sigma,
        expected_states: bands as f64 * multiplicity * weights.iter().sum::<f64>(),
        integrated_states: integral,
        grid_unit,
        density_unit,
    })
}

#[derive(Clone, Debug)]
pub struct HarmonicThermodynamics {
    pub temperature_kelvin: f64,
    pub zero_point_energy_hartree: f64,
    pub free_energy_hartree: f64,
    pub internal_energy_hartree: f64,
    pub entropy_hartree_per_kelvin: f64,
    pub heat_capacity_hartree_per_kelvin: f64,
    pub included_modes_per_primitive_cell: f64,
    pub excluded_zero_modes_per_primitive_cell: f64,
    pub zero_tolerance_cm1: f64,
}
impl HarmonicThermodynamics {
    pub fn as_json(&self) -> Value {
        json!({"temperature_kelvin":self.temperature_kelvin,
            "temperature_kind":"lattice",
            "zero_point_energy_hartree":self.zero_point_energy_hartree,
            "free_energy_hartree":self.free_energy_hartree,"internal_energy_hartree":self.internal_energy_hartree,
            "entropy_hartree_per_kelvin":self.entropy_hartree_per_kelvin,
            "heat_capacity_hartree_per_kelvin":self.heat_capacity_hartree_per_kelvin,
            "included_modes_per_primitive_cell":self.included_modes_per_primitive_cell,
            "excluded_zero_modes_per_primitive_cell":self.excluded_zero_modes_per_primitive_cell,
            "zero_tolerance_cm1":self.zero_tolerance_cm1,
            "normalization":"per primitive cell; vibrational contribution only",
            "electronic_reference":"held fixed; no electronic entropy or heat capacity from changing the electronic temperature"})
    }
}

/// Harmonic oscillator partition functions with explicit BZ weights.
/// Modes with |frequency| <= zero_tolerance are counted and omitted; every
/// more-negative mode is an error. No ASR projection or instability clipping.
pub fn harmonic_thermodynamics(
    wavenumbers_cm1: &[Vec<f64>],
    weights: &[f64],
    temperatures_kelvin: &[f64],
    zero_tolerance_cm1: f64,
) -> Result<Vec<HarmonicThermodynamics>> {
    validate_values(wavenumbers_cm1, weights)?;
    crate::resources::check_dense_workspace(
        wavenumbers_cm1.len(),
        wavenumbers_cm1[0].len(),
        2,
        "harmonic mode table",
    )?;
    crate::resources::check_dense_workspace(temperatures_kelvin.len(), 12, 2, "harmonic output")?;
    if !zero_tolerance_cm1.is_finite()
        || zero_tolerance_cm1 < 0.0
        || temperatures_kelvin.is_empty()
        || temperatures_kelvin
            .iter()
            .any(|t| !t.is_finite() || *t < 0.0)
    {
        return Err(invalid(
            "harmonic thermodynamics needs nonnegative finite temperatures and zero tolerance",
        ));
    }
    let mut positive = Vec::new();
    let mut excluded = 0.0;
    for (iq, (row, &weight)) in wavenumbers_cm1.iter().zip(weights).enumerate() {
        for (mode, &frequency) in row.iter().enumerate() {
            if frequency < -zero_tolerance_cm1 {
                return Err(invalid(format!("harmonic thermodynamics is undefined for imaginary mode {mode} at q index {iq}: {frequency:e} cm^-1")));
            }
            if frequency.abs() <= zero_tolerance_cm1 {
                excluded += weight;
            } else {
                let energy = frequency / HARTREE_TO_WAVENUMBER;
                if energy <= 0.0 {
                    return Err(invalid("positive phonon energy underflows floating point"));
                }
                positive.push((energy, weight));
            }
        }
    }
    let zpe = positive.iter().map(|(e, w)| 0.5 * e * w).sum::<f64>();
    let count = positive.iter().map(|(_, w)| w).sum::<f64>();
    let mut result = Vec::with_capacity(temperatures_kelvin.len());
    for &temperature in temperatures_kelvin {
        let mut free = zpe;
        let mut internal = zpe;
        let mut entropy = 0.0;
        let mut capacity = 0.0;
        if temperature > 0.0 {
            let kt = KB_HARTREE_PER_K * temperature;
            for &(energy, weight) in &positive {
                let x = energy / kt;
                if x > 700.0 {
                    continue;
                }
                if x <= 0.0 {
                    return Err(invalid(
                        "phonon/temperature ratio underflows floating point",
                    ));
                }
                let exp_minus = (-x).exp();
                let one_minus = -(-x).exp_m1();
                let log_one_minus = if x < std::f64::consts::LN_2 {
                    one_minus.ln()
                } else {
                    (-exp_minus).ln_1p()
                };
                let bose = x.exp_m1();
                free += weight * kt * log_one_minus;
                internal += weight * energy / bose;
                entropy += weight * KB_HARTREE_PER_K * (x / bose - log_one_minus);
                capacity += weight * KB_HARTREE_PER_K * (x / one_minus).powi(2) * exp_minus;
            }
        }
        if [free, internal, entropy, capacity, zpe]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err(invalid(
                "harmonic thermodynamics exceeds floating-point range",
            ));
        }
        result.push(HarmonicThermodynamics {
            temperature_kelvin: temperature,
            zero_point_energy_hartree: zpe,
            free_energy_hartree: free,
            internal_energy_hartree: internal,
            entropy_hartree_per_kelvin: entropy,
            heat_capacity_hartree_per_kelvin: capacity,
            included_modes_per_primitive_cell: count,
            excluded_zero_modes_per_primitive_cell: excluded,
            zero_tolerance_cm1,
        });
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct SampledBandGap {
    /// Negative values represent band overlap, not a positive insulating gap.
    pub indirect_gap_hartree: f64,
    pub minimum_direct_gap_hartree: f64,
    pub valence_max_hartree: f64,
    pub conduction_min_hartree: f64,
    pub valence_k_index: usize,
    pub conduction_k_index: usize,
    pub occupied_bands: usize,
    pub thermally_integer_occupations: bool,
}
impl SampledBandGap {
    pub fn as_json(&self) -> Value {
        json!({"indirect_gap_hartree":self.indirect_gap_hartree,"indirect_gap_ev":self.indirect_gap_hartree*HARTREE_TO_EV,
            "minimum_direct_gap_hartree":self.minimum_direct_gap_hartree,
            "minimum_direct_gap_ev":self.minimum_direct_gap_hartree*HARTREE_TO_EV,
            "valence_max_hartree":self.valence_max_hartree,"conduction_min_hartree":self.conduction_min_hartree,
            "valence_k_index":self.valence_k_index,"conduction_k_index":self.conduction_k_index,
            "occupied_bands":self.occupied_bands,"thermally_integer_occupations":self.thermally_integer_occupations,
            "scope":"sampled model orbital gap; not a quasiparticle or optical excitation gap"})
    }
}

#[derive(Clone, Debug)]
pub struct BandStructureResult {
    pub k_fractional: Vec<[f64; 3]>,
    pub path_distance_inverse_bohr: Vec<f64>,
    /// One value per physical scalar band; real-embedding duplicates are removed.
    pub eigenvalues_hartree: Vec<Vec<f64>>,
    pub fermi_level_hartree: f64,
    pub electronic_temperature_kelvin: f64,
    pub electrons_per_cell: f64,
    pub sampled_gap: Option<SampledBandGap>,
    pub ml_model_sha256: Option<String>,
    pub ml_correction_energy_hartree: f64,
    pub method_citations: Vec<&'static str>,
}
impl BandStructureResult {
    pub fn as_json(&self) -> Value {
        let ev: Vec<Vec<f64>> = self
            .eigenvalues_hartree
            .iter()
            .map(|row| row.iter().map(|e| e * HARTREE_TO_EV).collect())
            .collect();
        json!({"k_fractional":self.k_fractional,"path_distance_inverse_bohr":self.path_distance_inverse_bohr,
            "energies_hartree":self.eigenvalues_hartree,"energies_ev":ev,
            "fermi_level_hartree":self.fermi_level_hartree,"fermi_level_ev":self.fermi_level_hartree*HARTREE_TO_EV,
            "electronic_temperature_kelvin":self.electronic_temperature_kelvin,"electrons_per_cell":self.electrons_per_cell,
            "spin_degeneracy":2,"sampled_gap":self.sampled_gap.as_ref().map(SampledBandGap::as_json),
            "ml_model_sha256":self.ml_model_sha256,"ml_correction_energy_hartree":self.ml_correction_energy_hartree,
            "method_citations":self.method_citations,
            "scope":"fixed converged SCC Hamiltonian sampled on requested k points; no self-consistent refilling on the path"})
    }
}

/// Self-consistent reference mesh followed by arbitrary-k band resampling.
pub fn pbc_band_structure(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    k_fractional: &[[f64; 3]],
) -> Result<BandStructureResult> {
    validate_band_options(options)?;
    validate_band_path(system, k_fractional)?;
    let scf = run_pbc_scc(system, params, options, pbc)?;
    band_structure_from_scc(system, &scf, options, k_fractional)
}

/// Reuse the same converged SCC state, including ML-adjusted Bloch blocks and
/// learned shell levels. A correction head contributes to total energy only.
pub fn band_structure_from_scc(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    k_fractional: &[[f64; 3]],
) -> Result<BandStructureResult> {
    validate_band_options(options)?;
    validate_band_path(system, k_fractional)?;
    let lattice = system
        .lattice
        .as_ref()
        .ok_or_else(|| invalid("bands require a periodic cell"))?;
    if !scf.converged
        || !scf.fermi_level.is_finite()
        || !scf.nelec.is_finite()
        || !scf.electronic_temperature.is_finite()
        || scf.electronic_temperature < 0.0
        || !scf.atomic_moments.is_empty()
        || k_fractional.is_empty()
        || k_fractional.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(invalid("bands need a finite converged scalar SCC state, a nonempty finite k path, and no multipole Fock"));
    }
    if options.ml_model.is_some() != scf.ml_parameters.is_some() {
        return Err(invalid(
            "band options and reference disagree on the active ML model",
        ));
    }
    if let (Some(model), Some(view)) = (&options.ml_model, &scf.ml_parameters) {
        if model.canonical_sha256()? != view.model_sha256 {
            return Err(invalid(
                "band reference was calculated with different ML weights",
            ));
        }
    }
    let n = scf.basis.len();
    crate::resources::check_dense_workspace(n, n, 64, "band eigensolver workspace")?;
    crate::resources::check_dense_workspace(k_fractional.len(), n, 2, "band output")?;
    let mut potential = vec![0.0; n];
    for (shell, &value) in scf.basis.shells.iter().zip(&scf.shell_scc_potential) {
        for ao in shell.first_ao..shell.first_ao + shell.nao {
            potential[ao] = value;
        }
    }
    let mut eigenvalues = Vec::with_capacity(k_fractional.len());
    for &k in k_fractional {
        let (h, s) = scf.bloch.h_s_at_k(k);
        let fock = fock_at_k(&h, &s, &potential);
        let eig = hermitian_generalized_eigen(&fock, &s, options.eigen_tolerance)?;
        eigenvalues.push((0..n).map(|b| eig.values[2 * b]).collect::<Vec<_>>());
    }
    let reciprocal = lattice.reciprocal_vectors_2pi();
    let mut distances = vec![0.0; k_fractional.len()];
    for i in 1..k_fractional.len() {
        let difference = reciprocal[0] * (k_fractional[i][0] - k_fractional[i - 1][0])
            + reciprocal[1] * (k_fractional[i][1] - k_fractional[i - 1][1])
            + reciprocal[2] * (k_fractional[i][2] - k_fractional[i - 1][2]);
        distances[i] = distances[i - 1] + difference.norm();
    }
    let gap = sampled_band_gap(
        &eigenvalues,
        scf.nelec,
        scf.fermi_level,
        scf.electronic_temperature,
    );
    Ok(BandStructureResult {
        k_fractional: k_fractional.to_vec(),
        path_distance_inverse_bohr: distances,
        eigenvalues_hartree: eigenvalues,
        fermi_level_hartree: scf.fermi_level,
        electronic_temperature_kelvin: scf.electronic_temperature,
        electrons_per_cell: scf.nelec,
        sampled_gap: gap,
        ml_model_sha256: scf.ml_parameters.as_ref().map(|m| m.model_sha256.clone()),
        ml_correction_energy_hartree: scf.ml_correction_energy,
        method_citations: scf.method_citations.clone(),
    })
}

/// DOS of scalar physical bands, including their factor-two spin degeneracy.
/// Use a BZ mesh and its weights, not an unweighted high-symmetry path, for a
/// bulk DOS. The energies and grid are absolute model energies in eV.
pub fn electronic_dos(
    bands: &BandStructureResult,
    weights: &[f64],
    grid_ev: &[f64],
    sigma_ev: f64,
) -> Result<GaussianDosResult> {
    electronic_dos_from_energies(&bands.eigenvalues_hartree, weights, grid_ev, sigma_ev)
}

/// Array-level postprocessor for already sampled physical scalar bands.
pub fn electronic_dos_from_energies(
    energies_hartree: &[Vec<f64>],
    weights: &[f64],
    grid_ev: &[f64],
    sigma_ev: f64,
) -> Result<GaussianDosResult> {
    let count = validate_values(energies_hartree, weights)?;
    crate::resources::check_dense_workspace(
        energies_hartree.len(),
        count,
        2,
        "electronic DOS energies",
    )?;
    let values: Vec<Vec<f64>> = energies_hartree
        .iter()
        .map(|row| row.iter().map(|e| e * HARTREE_TO_EV).collect())
        .collect();
    gaussian_dos(
        &values,
        weights,
        grid_ev,
        sigma_ev,
        2.0,
        "eV",
        "states/(eV primitive_cell)",
    )
}

/// Already converted energy rows. Used at an external unit boundary such as
/// ASE so its own Hartree/eV convention is not converted through a second set.
pub fn electronic_dos_from_energies_ev(
    energies_ev: &[Vec<f64>],
    weights: &[f64],
    grid_ev: &[f64],
    sigma_ev: f64,
) -> Result<GaussianDosResult> {
    gaussian_dos(
        energies_ev,
        weights,
        grid_ev,
        sigma_ev,
        2.0,
        "eV",
        "states/(eV primitive_cell)",
    )
}

#[derive(Clone, Debug)]
pub struct QuasiharmonicPoint {
    pub temperature_kelvin: f64,
    pub equilibrium_volume_bohr3: f64,
    pub minimum_free_energy_hartree: f64,
    pub bulk_modulus_hartree_per_bohr3: f64,
    pub bulk_modulus_gpa: f64,
    pub thermal_expansion_per_kelvin: f64,
    pub heat_capacity_cv_hartree_per_kelvin: f64,
    pub heat_capacity_cp_hartree_per_kelvin: f64,
    pub entropy_hartree_per_kelvin: f64,
    pub free_energy_fit_rms_hartree: f64,
}
impl QuasiharmonicPoint {
    pub fn as_json(&self) -> Value {
        json!({"temperature_kelvin":self.temperature_kelvin,
            "temperature_kind":"lattice",
            "equilibrium_volume_bohr3":self.equilibrium_volume_bohr3,
            "minimum_free_energy_hartree":self.minimum_free_energy_hartree,
            "bulk_modulus_hartree_per_bohr3":self.bulk_modulus_hartree_per_bohr3,
            "bulk_modulus_gpa":self.bulk_modulus_gpa,
            "thermal_expansion_per_kelvin":self.thermal_expansion_per_kelvin,
            "heat_capacity_cv_hartree_per_kelvin":self.heat_capacity_cv_hartree_per_kelvin,
            "heat_capacity_cp_hartree_per_kelvin":self.heat_capacity_cp_hartree_per_kelvin,
            "entropy_hartree_per_kelvin":self.entropy_hartree_per_kelvin,
            "free_energy_fit_rms_hartree":self.free_energy_fit_rms_hartree,
            "pressure_hartree_per_bohr3":0.0,
            "electronic_reference":"caller-supplied static free energies at a fixed electronic temperature",
            "heat_capacity_scope":"vibrational plus volume relaxation; excludes electronic heat capacity from varying electronic temperature",
            "method":"QHA: third-order Eulerian finite-strain F(V,T) fit and entropy volume derivative",
            "normalization":"one consistently normalized primitive cell at every volume"})
    }
}

/// Zero-external-pressure quasi-harmonic equation of state. Input phonon rows
/// are `[volume][q][mode]`, with the same cell contents and q weights at each
/// volume. Every static energy/phonon calculation must use the same active
/// Hamiltonian, electronic temperature, k sampling and relaxation convention.
///
/// The temperature derivatives of each oscillator are analytic. The volume
/// dependence is an explicitly fitted quasi-harmonic approximation, not a
/// claim of analytic strain derivatives of the full phonon spectrum.
pub fn quasiharmonic_zero_pressure(
    volumes_bohr3: &[f64],
    static_energies_hartree: &[f64],
    phonons_by_volume_cm1: &[Vec<Vec<f64>>],
    q_weights: &[f64],
    temperatures_kelvin: &[f64],
    zero_tolerance_cm1: f64,
) -> Result<Vec<QuasiharmonicPoint>> {
    use super::mechanical::fit_birch_murnaghan;
    if volumes_bohr3.len() != static_energies_hartree.len()
        || volumes_bohr3.len() != phonons_by_volume_cm1.len()
        || volumes_bohr3.len() < 5
    {
        return Err(invalid(
            "QHA requires at least five matched volume, static-energy and phonon samples",
        ));
    }
    crate::resources::check_dense_workspace(
        volumes_bohr3.len(),
        temperatures_kelvin.len(),
        16,
        "QHA thermodynamic table",
    )?;
    let mut thermo = Vec::with_capacity(volumes_bohr3.len());
    let mut shape = None;
    for phonons in phonons_by_volume_cm1 {
        let bands = validate_values(phonons, q_weights)?;
        if shape.is_some_and(|n| n != bands) {
            return Err(invalid("QHA phonon branch counts differ between volumes"));
        }
        shape = Some(bands);
        thermo.push(harmonic_thermodynamics(
            phonons,
            q_weights,
            temperatures_kelvin,
            zero_tolerance_cm1,
        )?);
    }
    let mut result = Vec::with_capacity(temperatures_kelvin.len());
    for (it, &temperature) in temperatures_kelvin.iter().enumerate() {
        let free: Vec<f64> = static_energies_hartree
            .iter()
            .zip(&thermo)
            .map(|(&e, t)| e + t[it].free_energy_hartree)
            .collect();
        let fit = fit_birch_murnaghan(volumes_bohr3, &free)?;
        let v = fit.equilibrium_volume_bohr3;
        let entropy: Vec<f64> = thermo
            .iter()
            .map(|t| t[it].entropy_hartree_per_kelvin)
            .collect();
        let cv: Vec<f64> = thermo
            .iter()
            .map(|t| t[it].heat_capacity_hartree_per_kelvin)
            .collect();
        let (entropy_value, entropy_derivative) = finite_strain_property_at(
            volumes_bohr3,
            &entropy,
            v,
            fit.reference_volume_bohr3,
            fit.finite_strain_center,
            fit.finite_strain_scale,
        )?;
        let (capacity, _) = finite_strain_property_at(
            volumes_bohr3,
            &cv,
            v,
            fit.reference_volume_bohr3,
            fit.finite_strain_center,
            fit.finite_strain_scale,
        )?;
        if capacity < -1e-14 || entropy_value < -1e-14 {
            return Err(invalid(
                "QHA entropy/Cv interpolation became negative; refine the volume grid",
            ));
        }
        let alpha = entropy_derivative / fit.bulk_modulus_hartree_per_bohr3;
        let cp = capacity + temperature * v * fit.bulk_modulus_hartree_per_bohr3 * alpha * alpha;
        if !alpha.is_finite() || !cp.is_finite() {
            return Err(invalid("QHA response exceeds floating-point range"));
        }
        result.push(QuasiharmonicPoint {
            temperature_kelvin: temperature,
            equilibrium_volume_bohr3: v,
            minimum_free_energy_hartree: fit.minimum_energy_hartree,
            bulk_modulus_hartree_per_bohr3: fit.bulk_modulus_hartree_per_bohr3,
            bulk_modulus_gpa: fit.bulk_modulus_gpa,
            thermal_expansion_per_kelvin: alpha,
            heat_capacity_cv_hartree_per_kelvin: capacity,
            heat_capacity_cp_hartree_per_kelvin: cp,
            entropy_hartree_per_kelvin: entropy_value,
            free_energy_fit_rms_hartree: fit.rms_energy_error_hartree,
        });
    }
    Ok(result)
}

// Same scaled finite-strain basis as the free-energy fit. Thus its entropy
// interpolation is the analytic temperature derivative of that linear fit.
fn finite_strain_property_at(
    volumes: &[f64],
    values: &[f64],
    volume: f64,
    reference: f64,
    center: f64,
    scale: f64,
) -> Result<(f64, f64)> {
    let x_at = |v: f64| (0.5 * ((reference / v).powf(2.0 / 3.0) - 1.0) - center) / scale;
    let mut a = Matrix::zeros(4, 4);
    let mut rhs = vec![0.0; 4];
    for (&v, &value) in volumes.iter().zip(values) {
        let x = x_at(v);
        let powers = [1.0, x, x * x, x * x * x];
        for i in 0..4 {
            rhs[i] += powers[i] * value;
            for j in 0..4 {
                a[(i, j)] += powers[i] * powers[j];
            }
        }
    }
    let c = DenseLu::factor(&a)?.solve_vec(&rhs)?;
    let x = x_at(volume);
    let value = c[0] + x * (c[1] + x * (c[2] + x * c[3]));
    let df_dv = -(reference / volume).powf(2.0 / 3.0) / (3.0 * volume);
    let derivative = (c[1] + x * (2.0 * c[2] + x * 3.0 * c[3])) * df_dv / scale;
    Ok((value, derivative))
}

fn sampled_band_gap(
    energies: &[Vec<f64>],
    electrons: f64,
    fermi: f64,
    temperature: f64,
) -> Option<SampledBandGap> {
    let occupied = (0.5 * electrons).round() as usize;
    if (electrons - 2.0 * occupied as f64).abs() > 1e-8
        || occupied == 0
        || occupied >= energies[0].len()
    {
        return None;
    }
    let (vki, valence) = energies
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e[occupied - 1]))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    let (cki, conduction) = energies
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e[occupied]))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    let direct = energies
        .iter()
        .map(|e| e[occupied] - e[occupied - 1])
        .fold(f64::INFINITY, f64::min);
    let kt = KB_HARTREE_PER_K * temperature;
    let integer = if kt > 0.0 {
        energies.iter().all(|row| {
            row.iter().enumerate().all(|(b, &e)| {
                let x = (e - fermi) / kt;
                let f = if x > 40.0 {
                    0.0
                } else if x < -40.0 {
                    1.0
                } else {
                    1.0 / (1.0 + x.exp())
                };
                (f - if b < occupied { 1.0 } else { 0.0 }).abs() < 1e-8
            })
        })
    } else {
        conduction > valence
    };
    Some(SampledBandGap {
        indirect_gap_hartree: conduction - valence,
        minimum_direct_gap_hartree: direct,
        valence_max_hartree: valence,
        conduction_min_hartree: conduction,
        valence_k_index: vki,
        conduction_k_index: cki,
        occupied_bands: occupied,
        thermally_integer_occupations: integer,
    })
}

fn validate_band_options(options: &ElectronicOptions) -> Result<()> {
    if options.multipole
        || options.spin_polarization
        || options.plus_u
        || options.lr_exchange
        || options.onsite_exchange
        || options.solvation.is_some()
        || options.external_field.is_active()
        || options.spin_multiplicity.is_some_and(|m| m != 1)
    {
        return Err(invalid("bands currently require scalar, zero-field SCC without multipole/exchange/spin/U/solvent terms; ML is supported"));
    }
    Ok(())
}
fn validate_band_path(system: &PeriodicSystem, k_fractional: &[[f64; 3]]) -> Result<()> {
    let lattice = system
        .lattice
        .as_ref()
        .ok_or_else(|| invalid("bands require a periodic cell"))?;
    if !lattice.periodic.iter().any(|&p| p)
        || k_fractional.is_empty()
        || k_fractional.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "bands require a nonempty finite k path and at least one periodic lattice direction",
        ));
    }
    if k_fractional
        .iter()
        .any(|k| (0..3).any(|d| !lattice.periodic[d] && k[d].abs() > 1e-12))
    {
        return Err(invalid(
            "band k components along a non-periodic direction must be zero",
        ));
    }
    Ok(())
}
fn validate_weights(count: usize, weights: &[f64]) -> Result<()> {
    if count == 0
        || weights.len() != count
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        return Err(invalid("spectral weights must be finite, nonnegative, match the nonempty sample set and sum to one"));
    }
    Ok(())
}
fn validate_values(values: &[Vec<f64>], weights: &[f64]) -> Result<usize> {
    validate_weights(values.len(), weights)?;
    let bands = values[0].len();
    if bands == 0
        || values
            .iter()
            .any(|row| row.len() != bands || row.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid(
            "spectral rows must have the same positive band count and finite values",
        ));
    }
    Ok(bands)
}
fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
