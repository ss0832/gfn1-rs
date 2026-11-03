// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic bulk electromechanical response and polar-mode postprocessing.
use super::complex::CMatrix;
use super::hessian::{shell_polynomial_log_derivative, KpointSkeleton};
use super::{PbcOptions, PbcSccResult};
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::data_tables::atomic_radius_bohr;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::contracted_pair_with_derivatives;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};

pub const STRAIN_PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// Action of one engineering-strain generator on a Cartesian vector.
pub(crate) fn strain_action(j: usize, v: Vec3) -> Vec3 {
    let (a, b) = STRAIN_PAIRS[j];
    let r = v.to_array();
    let mut out = [0.; 3];
    if a == b {
        out[a] = r[a];
    } else {
        out[a] = 0.5 * r[b];
        out[b] = 0.5 * r[a];
    }
    Vec3::new(out[0], out[1], out[2])
}

pub(crate) fn require_scalar(system: &PeriodicSystem, options: &ElectronicOptions) -> Result<()> {
    if system.lattice.is_none_or(|l| l.periodic != [true; 3])
        || system.atoms.is_empty()
        || options.multipole
        || options.spin_polarization
        || options.plus_u
        || options.spin_multiplicity.is_some_and(|m| m != 1)
        || options.lr_exchange
        || options.solvation.is_some()
        || options.external_field.is_active()
    {
        return Err(Gfn1Error::InvalidInput("bulk electromechanical response requires a nonempty 3D scalar GFN1/ML cell without external fields or composite Hamiltonians".into()));
    }
    Ok(())
}

/// Fixed-density homogeneous-strain Fock/overlap skeleton, including CN,
/// periodic Ewald and all learned structural level heads. No displaced cell.
pub fn strain_skeleton_at_k(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    fractional: [f64; 3],
) -> Result<KpointSkeleton> {
    Ok(
        strain_skeletons_at_points(system, params, scf, options, pbc, &[fractional], None)?
            .pop()
            .expect("one strain-skeleton point"),
    )
}

fn strain_skeletons_at_points(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    fractionals: &[[f64; 3]],
    reference_hs: Option<&[(CMatrix, CMatrix)]>,
) -> Result<Vec<KpointSkeleton>> {
    require_scalar(system, options)?;
    let lattice = system.lattice.unwrap();
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let nk = fractionals.len();
    if nk == 0 {
        return Ok(Vec::new());
    }
    if reference_hs.is_some_and(|blocks| blocks.len() != nk) {
        return Err(Gfn1Error::InvalidInput(
            "strain-skeleton overlap/k-point count mismatch".into(),
        ));
    }
    let bytes = (|| {
        // Twelve complex output matrices per k, plus one temporary H/S pair.
        let matrices =
            nk.checked_mul(24)?
                .checked_add(if reference_hs.is_some() { 0 } else { 4 })?;
        let dense = n.checked_mul(n)?.checked_mul(8)?.checked_mul(matrices)?;
        let vectors = basis
            .shells
            .len()
            .checked_add(nat)?
            .checked_mul(6 * 8)?
            .checked_mul(nk.checked_add(4)?)?;
        dense.checked_add(vectors)
    })()
    .ok_or_else(|| Gfn1Error::InvalidInput("strain-skeleton workspace overflow".into()))?;
    crate::resources::check_workspace(bytes, "all-k analytic strain skeletons")?;
    let mut coordination = vec![vec![0.; nat]; 6];
    for pair in coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: options.hamiltonian.coordination_cutoff,
            ..Default::default()
        },
    )?
    .pairs
    {
        let r = pair.r_ij.to_array();
        let distance = pair.r_ij.norm();
        for (j, &(a, b)) in STRAIN_PAIRS.iter().enumerate() {
            let value = pair.dcn_dr * r[a] * r[b] / distance;
            coordination[j][pair.i] += value;
            if pair.i != pair.j {
                coordination[j][pair.j] += value;
            }
        }
    }
    let learned = if let Some(view) = &scf.ml_parameters {
        view.affine_level_jacobian()?
    } else {
        vec![vec![[0.; 3]; nat]; 6]
    };
    let ns = basis.shells.len();
    let mut shell_potential = vec![vec![0.; ns]; 6];
    // Polarization of the analytic quadratic charge form gives its exact
    // bilinear coefficient. The unit charge slots are algebraic, not a finite
    // nuclear/cell displacement or a finite-step approximation to a derivative.
    for s in 0..ns {
        let atom = basis.shells[s].atom_index;
        let mut plus = scf.shell_charges.clone();
        let mut minus = plus.clone();
        let mut ap = scf.atomic_charges.clone();
        let mut am = ap.clone();
        plus[s] += 1.;
        minus[s] -= 1.;
        ap[atom] += 1.;
        am[atom] -= 1.;
        let a = super::stress::electrostatic_stress_for_charges(
            system,
            basis,
            &scf.shell_model,
            &ap,
            &plus,
            pbc,
            &lattice,
        );
        let b = super::stress::electrostatic_stress_for_charges(
            system,
            basis,
            &scf.shell_model,
            &am,
            &minus,
            pbc,
            &lattice,
        );
        for (j, &(row, col)) in STRAIN_PAIRS.iter().enumerate() {
            shell_potential[j][s] = 0.5 * lattice.volume() * (a[(row, col)] - b[(row, col)]);
        }
    }
    let mut outputs: Vec<_> = (0..nk)
        .map(|_| KpointSkeleton {
            overlap: vec![CMatrix::zeros(n); 6],
            fock: vec![CMatrix::zeros(n); 6],
            shell_potential: shell_potential.clone(),
            coordination: coordination.clone(),
        })
        .collect();
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let exponents = crate::basis::atom_min_exponents(basis, nat);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    for off in images {
        let translation = lattice.translation(off);
        let phases: Vec<_> = fractionals
            .iter()
            .map(|&fractional| super::kpoints::bloch_phase(fractional, off))
            .collect();
        for (mu, a) in basis.aos.iter().enumerate() {
            for (nu, b) in basis.aos.iter().enumerate() {
                let onsite = off.is_origin() && a.atom_index == b.atom_index;
                let ra = system.atoms[a.atom_index].position;
                let rb = system.atoms[b.atom_index].position + translation;
                let rvec = ra - rb;
                let r2 = rvec.norm2();
                if !onsite
                    && (r2 <= 1e-24
                        || r2 > cutoff2
                        || crate::basis::overlap_screened(
                            &exponents,
                            a.atom_index,
                            b.atom_index,
                            r2,
                        ))
                {
                    continue;
                }
                let i = a.shell_index;
                let t = b.shell_index;
                let si = &basis.shells[i];
                let st = &basis.shells[t];
                let (moments, da, _) = contracted_pair_with_derivatives(a, b, ra, rb);
                let value = moments[0];
                let hs = if onsite {
                    1.
                } else {
                    hscale(si, st, params)?
                        * shell_polynomial(
                            si,
                            st,
                            (r2.sqrt() / (atomic_radius_bohr(a.z)? + atomic_radius_bohr(b.z)?))
                                .sqrt(),
                        )
                };
                let hij = 0.5 * (scf.bloch.self_energies[i] + scf.bloch.self_energies[t]) * hs;
                let logp = if onsite {
                    Vec3::zero()
                } else {
                    shell_polynomial_log_derivative(si, st, rvec, r2)
                };
                for j in 0..6 {
                    let dr = strain_action(j, rvec);
                    let ds = da[0].dot(dr);
                    let mut dh = 0.5
                        * hs
                        * (learned[j][a.atom_index][si.angular.as_index()]
                            + learned[j][b.atom_index][st.angular.as_index()]);
                    if options.hamiltonian.enable_cn_hamiltonian {
                        dh += 0.5
                            * hs
                            * (scf.bloch.dsedcn[i] * coordination[j][a.atom_index]
                                + scf.bloch.dsedcn[t] * coordination[j][b.atom_index]);
                    }
                    let df = hij * ds + hij * value * logp.dot(dr) + dh * value
                        - 0.5 * (scf.shell_scc_potential[i] + scf.shell_scc_potential[t]) * ds;
                    // Each k receives the same image/AO/strain accumulation
                    // order as the single-point assembly. Geometry is streamed
                    // once, with no retained real-space derivative cache.
                    for (out, &(c, s)) in outputs.iter_mut().zip(&phases) {
                        out.overlap[j].accumulate(mu, nu, ds * c, ds * s);
                        out.fock[j].accumulate(mu, nu, df * c, df * s);
                    }
                }
            }
        }
    }
    for (k, out) in outputs.iter_mut().enumerate() {
        let owned;
        let sk = if let Some(blocks) = reference_hs {
            &blocks[k].1
        } else {
            owned = scf.bloch.h_s_at_k(fractionals[k]);
            &owned.1
        };
        for j in 0..6 {
            for mu in 0..n {
                for nu in 0..n {
                    let scale = -0.5
                        * (shell_potential[j][basis.aos[mu].shell_index]
                            + shell_potential[j][basis.aos[nu].shell_index]);
                    out.fock[j].accumulate(
                        mu,
                        nu,
                        scale * sk.re[(mu, nu)],
                        scale * sk.im[(mu, nu)],
                    );
                }
            }
        }
    }
    Ok(outputs)
}

pub fn strain_skeletons(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<Vec<KpointSkeleton>> {
    let fractionals: Vec<_> = scf.kpoints.iter().map(|k| k.fractional).collect();
    strain_skeletons_at_points(
        system,
        params,
        scf,
        options,
        pbc,
        &fractionals,
        Some(&scf.hs_k),
    )
}

fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
}
fn polar_inputs(born: &[Matrix], masses: &[f64], volume: f64) -> Result<()> {
    if born.is_empty()
        || born.len() != masses.len()
        || !volume.is_finite()
        || volume <= 0.
        || masses.iter().any(|x| !x.is_finite() || *x <= 0.)
        || born
            .iter()
            .any(|z| z.rows() != 3 || z.cols() != 3 || z.as_slice().iter().any(|x| !x.is_finite()))
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid Born tensors, masses or cell volume".into(),
        ));
    }
    Ok(())
}

/// Macroscopic q->0 nonanalytic dynamical correction in Hartree/bohr²/amu.
/// `direction` is Cartesian; `dielectric` is the electronic relative tensor.
/// No dielectric or Born parameter is silently guessed.
pub fn nonanalytic_gamma_correction(
    born: &[Matrix],
    dielectric: &Matrix,
    volume: f64,
    masses: &[f64],
    direction: [f64; 3],
) -> Result<Matrix> {
    polar_inputs(born, masses, volume)?;
    crate::resources::check_dense_workspace(
        3 * born.len(),
        3 * born.len(),
        6,
        "nonanalytic dynamical response",
    )?;
    if dielectric.rows() != 3
        || dielectric.cols() != 3
        || dielectric.as_slice().iter().any(|x| !x.is_finite())
        || direction.iter().any(|x| !x.is_finite())
    {
        return Err(Gfn1Error::InvalidInput(
            "LO-TO requires finite Cartesian direction and electronic dielectric3x3".into(),
        ));
    }
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 1e-30 {
        return Err(Gfn1Error::InvalidInput(
            "LO-TO direction must be nonzero".into(),
        ));
    }
    let q = direction.map(|x| x / norm);
    let asym = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| (dielectric[(i, j)] - dielectric[(j, i)]).abs())
        .fold(0., f64::max);
    if asym > 1e-10
        || symmetric_eigen(dielectric)?
            .values
            .iter()
            .any(|x| *x <= 1e-12)
    {
        return Err(Gfn1Error::InvalidInput(
            "electronic dielectric must be symmetric positive definite".into(),
        ));
    }
    let denominator = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| q[i] * dielectric[(i, j)] * q[j])
                .sum::<f64>()
        })
        .sum::<f64>();
    let charges: Vec<f64> = (0..3 * born.len())
        .map(|i| (0..3).map(|a| q[a] * born[i / 3][(a, i % 3)]).sum::<f64>() / masses[i / 3].sqrt())
        .collect();
    let mut out = Matrix::zeros(charges.len(), charges.len());
    for i in 0..charges.len() {
        for j in 0..charges.len() {
            out[(i, j)] =
                4. * std::f64::consts::PI * charges[i] * charges[j] / (volume * denominator);
        }
    }
    Ok(out)
}

pub struct PolarModeResult {
    pub wavenumbers_cm1: Vec<f64>,
    pub mode_effective_charges: Matrix,
    pub oscillator_strengths: Vec<Matrix>,
    pub ionic_dielectric: Option<Matrix>,
    pub ionic_dielectric_error: Option<String>,
    pub eigenvalues_hartree_per_bohr2_amu: Vec<f64>,
    pub eigenvectors: Matrix,
}
impl PolarModeResult {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({"wavenumbers_cm1":self.wavenumbers_cm1,
            "mode_effective_charges_e_per_sqrt_amu":rows(&self.mode_effective_charges),
            "oscillator_strengths_e2_per_amu":self.oscillator_strengths.iter().map(rows).collect::<Vec<_>>(),
            "ionic_dielectric":self.ionic_dielectric.as_ref().map(rows),"ionic_dielectric_error":self.ionic_dielectric_error,
            "eigenvalues_hartree_per_bohr2_amu":self.eigenvalues_hartree_per_bohr2_amu,
            "mass_weighted_eigenvectors":rows(&self.eigenvectors),"ir_scope":"mode oscillator strengths; no empirical line broadening"})
    }
}

fn mass_weight(hessian: &Matrix, masses: &[f64]) -> Result<Matrix> {
    let d = 3 * masses.len();
    crate::resources::check_dense_workspace(d, d, 6, "polar modes")?;
    if hessian.rows() != d
        || hessian.cols() != d
        || hessian.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(Gfn1Error::InvalidInput(
            "polar modes require a finite 3N-square Cartesian Hessian".into(),
        ));
    }
    let mut out = hessian.clone();
    let mut asym = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            asym = asym.max((hessian[(i, j)] - hessian[(j, i)]).abs());
            out[(i, j)] /= (masses[i / 3] * masses[j / 3]).sqrt();
        }
    }
    if asym > 1e-7 {
        return Err(Gfn1Error::InvalidInput(
            "polar-mode Hessian is not symmetric".into(),
        ));
    }
    Ok(out)
}

/// Zone-centre mode charges/IR oscillator tensors and the stable ionic
/// dielectric increment. Imaginary polar modes remain explicit; no clipping.
pub fn polar_mode_properties(
    hessian: &Matrix,
    born: &[Matrix],
    masses: &[f64],
    volume: f64,
    zero_tolerance: f64,
) -> Result<PolarModeResult> {
    polar_inputs(born, masses, volume)?;
    if !zero_tolerance.is_finite() || zero_tolerance <= 0. {
        return Err(Gfn1Error::InvalidInput(
            "positive curvature tolerance required".into(),
        ));
    }
    let eigen = symmetric_eigen(&mass_weight(hessian, masses)?)?;
    let d = eigen.values.len();
    let mut charge = Matrix::zeros(3, d);
    let mut strengths = Vec::with_capacity(d);
    let mut ionic = Matrix::zeros(3, 3);
    let mut error = None;
    for mode in 0..d {
        for a in 0..3 {
            charge[(a, mode)] = (0..d)
                .map(|i| born[i / 3][(a, i % 3)] * eigen.vectors[(i, mode)] / masses[i / 3].sqrt())
                .sum();
        }
        let mut strength = Matrix::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                strength[(a, b)] = charge[(a, mode)] * charge[(b, mode)];
            }
        }
        if eigen.values[mode] < -zero_tolerance {
            error = Some(
                "imaginary internal modes: a stable static ionic dielectric is undefined".into(),
            );
        } else if eigen.values[mode].abs() <= zero_tolerance {
            if (0..3).map(|a| charge[(a, mode)].abs()).fold(0., f64::max) > 1e-6 {
                error=Some("a zero-frequency mode carries nonzero charge; static dielectric is singular (check acoustic sum rules)".into());
            }
        } else {
            for a in 0..3 {
                for b in 0..3 {
                    ionic[(a, b)] += 4. * std::f64::consts::PI * strength[(a, b)]
                        / (volume * eigen.values[mode]);
                }
            }
        }
        strengths.push(strength);
    }
    let wavenumbers = eigen
        .values
        .iter()
        .map(|&x| {
            x.signum()
                * (x.abs() / crate::constants::AMU_TO_ELECTRON_MASS).sqrt()
                * crate::constants::HARTREE_TO_WAVENUMBER
        })
        .collect();
    Ok(PolarModeResult {
        wavenumbers_cm1: wavenumbers,
        mode_effective_charges: charge,
        oscillator_strengths: strengths,
        ionic_dielectric: if error.is_none() { Some(ionic) } else { None },
        ionic_dielectric_error: error,
        eigenvalues_hartree_per_bohr2_amu: eigen.values,
        eigenvectors: eigen.vectors,
    })
}

pub struct LoToModes {
    pub direction_cartesian: [f64; 3],
    pub nonanalytic_correction: Matrix,
    pub dynamical_matrix: Matrix,
    pub wavenumbers_cm1: Vec<f64>,
    pub eigenvalues: Vec<f64>,
}
impl LoToModes {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({"direction_cartesian":self.direction_cartesian,
        "nonanalytic_correction_hartree_per_bohr2_amu":rows(&self.nonanalytic_correction),
        "dynamical_matrix_hartree_per_bohr2_amu":rows(&self.dynamical_matrix),
        "wavenumbers_cm1":self.wavenumbers_cm1,"eigenvalues_hartree_per_bohr2_amu":self.eigenvalues,
        "scope":"directional Gamma limit with explicit electronic dielectric and Born tensors"})
    }
}
pub fn lo_to_modes(
    hessian: &Matrix,
    born: &[Matrix],
    dielectric: &Matrix,
    volume: f64,
    masses: &[f64],
    direction: [f64; 3],
) -> Result<LoToModes> {
    let correction = nonanalytic_gamma_correction(born, dielectric, volume, masses, direction)?;
    let mut dynamical = mass_weight(hessian, masses)?;
    for i in 0..dynamical.rows() {
        for j in 0..dynamical.cols() {
            dynamical[(i, j)] += correction[(i, j)];
        }
    }
    let eig = symmetric_eigen(&dynamical)?;
    let frequencies = eig
        .values
        .iter()
        .map(|&x| {
            x.signum()
                * (x.abs() / crate::constants::AMU_TO_ELECTRON_MASS).sqrt()
                * crate::constants::HARTREE_TO_WAVENUMBER
        })
        .collect();
    Ok(LoToModes {
        direction_cartesian: direction,
        nonanalytic_correction: correction,
        dynamical_matrix: dynamical,
        wavenumbers_cm1: frequencies,
        eigenvalues: eig.values,
    })
}

#[derive(Clone, Debug)]
pub struct ElectromechanicalOptions {
    pub elastic: super::elastic_response::AnalyticElasticOptions,
    pub berry: super::polarization::BerryPolarizationOptions,
    pub zero_curvature_tolerance: f64,
    pub zero_stress_tolerance: f64,
}
impl Default for ElectromechanicalOptions {
    fn default() -> Self {
        Self {
            elastic: Default::default(),
            berry: Default::default(),
            zero_curvature_tolerance: 1e-10,
            zero_stress_tolerance: 1e-6,
        }
    }
}
pub struct ElectromechanicalResponse {
    pub elastic: super::elastic_response::AnalyticElasticResponse,
    pub berry: super::polarization::response::BerryPolarizationDerivatives,
    pub electronic: super::polarization::response::ElectronicDielectric,
    pub polar_modes: PolarModeResult,
    pub relaxed_piezoelectric: Option<Matrix>,
    pub piezoelectric_strain: Option<Matrix>,
    pub static_dielectric_fixed_strain: Option<Matrix>,
    pub static_dielectric_free_stress: Option<Matrix>,
    pub relaxed_error: Option<String>,
}
impl ElectromechanicalResponse {
    pub fn as_json(&self) -> serde_json::Value {
        let piezo_si = |m: &Matrix| {
            rows(m)
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .map(|x| x * super::polarization::POLARIZATION_AU_TO_C_PER_M2)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let d_si = |m: &Matrix| {
            rows(m)
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .map(|x| {
                            x * 1.602_176_634e-19 * 5.291_772_109_03e-11 / 4.359_744_722_207_1e-18
                                * 1e12
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        serde_json::json!({"elastic":self.elastic.as_json(),"berry":self.berry.as_json(),
            "electronic":self.electronic.as_json(),"polar_modes":self.polar_modes.as_json(),
            "relaxed_piezoelectric_au":self.relaxed_piezoelectric.as_ref().map(rows),
            "relaxed_piezoelectric_c_per_m2":self.relaxed_piezoelectric.as_ref().map(piezo_si),
            "piezoelectric_strain_au":self.piezoelectric_strain.as_ref().map(rows),
            "piezoelectric_strain_pc_per_n":self.piezoelectric_strain.as_ref().map(d_si),
            "static_dielectric_fixed_strain":self.static_dielectric_fixed_strain.as_ref().map(rows),
            "static_dielectric_free_stress":self.static_dielectric_free_stress.as_ref().map(rows),
            "relaxed_response_error":self.relaxed_error,
            "boundary_conditions":"proper piezoelectric e at fixed E; ionic dielectric at fixed strain; free-stress dielectric only for stable zero-stress references",
            "production_derivatives":"analytic including structural ML; no displaced cell/nucleus or field differences"})
    }
}

/// Complete, same-reference scalar/ML bulk response. All electronic quantities
/// reuse the elastic calculation's single converged SCC state and CP settings.
pub fn pbc_electromechanical_response(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    masses: &[f64],
    settings: &ElectromechanicalOptions,
) -> Result<ElectromechanicalResponse> {
    require_scalar(system, options)?;
    super::polarization::response::validate_mesh(pbc, &settings.berry)?;
    if masses.len() != system.atoms.len() || masses.iter().any(|x| !x.is_finite() || *x <= 0.) {
        return Err(Gfn1Error::InvalidInput(
            "electromechanical masses must be positive and match atoms".into(),
        ));
    }
    if !settings.zero_stress_tolerance.is_finite() || settings.zero_stress_tolerance <= 0. {
        return Err(Gfn1Error::InvalidInput(
            "positive zero-stress tolerance required".into(),
        ));
    }
    let elastic = super::elastic_response::pbc_elastic_response(
        system,
        params,
        options,
        pbc,
        &settings.elastic,
    )?;
    let reference = &elastic.reference_system;
    let mut electronic_options = options.clone();
    electronic_options.energy_tolerance = electronic_options.energy_tolerance.min(1e-12);
    electronic_options.charge_tolerance = electronic_options.charge_tolerance.min(1e-10);
    let berry = super::polarization::response::pbc_polarization_derivatives_from_scc(
        reference,
        params,
        &electronic_options,
        pbc,
        &settings.berry,
        &elastic.scf,
    )?;
    let electronic = super::polarization::response::pbc_electronic_dielectric_from_scc(
        reference,
        &electronic_options,
        pbc,
        &settings.berry,
        &elastic.scf,
    )?;
    let volume = reference.lattice.unwrap().volume();
    let polar_modes = polar_mode_properties(
        &elastic.nuclear_hessian,
        &berry.born_effective_charges,
        masses,
        volume,
        settings.zero_curvature_tolerance,
    )?;
    let mut relaxed_error = elastic.relaxed_response_error.clone();
    let mut relaxed_piezoelectric = None;
    let mut piezoelectric_strain = None;
    let mut static_dielectric_fixed_strain = None;
    let mut static_dielectric_free_stress = None;
    if let Some(du) = &elastic.internal_displacement_strain {
        let mut e = berry.clamped_piezoelectric.clone();
        for i in 0..3 {
            for j in 0..6 {
                e[(i, j)] += (0..3 * masses.len())
                    .map(|y| berry.born_effective_charges[y / 3][(i, y % 3)] * du[(y, j)])
                    .sum::<f64>()
                    / volume;
            }
        }
        if let Some(ionic) = &polar_modes.ionic_dielectric {
            let mut eps = electronic.relative_dielectric.clone();
            for i in 0..3 {
                for j in 0..3 {
                    eps[(i, j)] += ionic[(i, j)];
                }
            }
            static_dielectric_fixed_strain = Some(eps);
        } else {
            relaxed_error = polar_modes.ionic_dielectric_error.clone();
        }
        if let Some(c) = &elastic.relaxed_stiffness {
            let eig = symmetric_eigen(c)?;
            let stress = elastic
                .reference_stress
                .as_slice()
                .iter()
                .map(|x| x.abs())
                .fold(0., f64::max);
            if eig.values[0] > settings.elastic.stability_tolerance
                && stress < settings.zero_stress_tolerance
            {
                let compliance =
                    crate::linalg::DenseLu::factor(c)?.solve_matrix(&Matrix::identity(6))?;
                let d = e.matmul(&compliance)?;
                if let Some(eps) = &static_dielectric_fixed_strain {
                    let increment = d.matmul(&e.transpose())?;
                    let mut free = eps.clone();
                    for i in 0..3 {
                        for j in 0..3 {
                            free[(i, j)] += 4. * std::f64::consts::PI * increment[(i, j)];
                        }
                    }
                    static_dielectric_free_stress = Some(free);
                }
                piezoelectric_strain = Some(d);
            } else {
                relaxed_error=Some(format!("free-stress/d response requires stable zero-stress stiffness; min curvature {}, stress {stress:e}",eig.values[0]));
            }
        }
        relaxed_piezoelectric = Some(e);
    }
    Ok(ElectromechanicalResponse {
        elastic,
        berry,
        electronic,
        polar_modes,
        relaxed_piezoelectric,
        piezoelectric_strain,
        static_dielectric_fixed_strain,
        static_dielectric_free_stress,
        relaxed_error,
    })
}
