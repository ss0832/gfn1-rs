// SPDX-License-Identifier: GPL-3.0-or-later
//! Molecular electrostatic-response properties and vibrational spectroscopy.
//!
//! All routines here are **non-periodic** (they reuse the molecular CPXTB response
//! and the external-field machinery). The quantities provided are:
//!
//! - the Mulliken (monopole) dipole moment `mu = sum_A q_A (R_A - origin)`;
//! - **analytic** Cartesian dipole derivatives `dmu/dR` from the coupled-perturbed
//!   (CPXTB) nuclear charge response — the raw tensor behind the IR intensities;
//! - the **analytic** static dipole polarizability `alpha = dmu/dE` from the CPXTB
//!   field response (see [`crate::cphf::solve_field_response`]);
//! - harmonic IR intensities and Raman activities, exposing the raw derivative
//!   tensors (`dmu/dR`, `dalpha/dR`, and the per-mode `dmu/dQ`, `dalpha/dQ`).
//!
//! The polarizability derivative `dalpha/dR` (for Raman) is the finite-field
//! derivative of the *analytic* dipole gradient — only the field derivative is
//! numerical; the nuclear part is analytic.

use crate::cphf::{solve_field_response, CpxtbOptions};
use crate::electronic::{run_electronic, ElectronicOptions, ElectronicResult};
use crate::error::{Gfn1Error, Result};
use crate::field::{mulliken_dipole, ExternalFieldOptions};
use crate::hessian::{analytic_hessian, AnalyticHessianOptions};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;
use crate::vibrational::vibrational_analysis;

use crate::basis::BasisSet;
use crate::linalg::Matrix;

/// Standard conversion from an atomic-unit IR intensity `(dmu/dQ)^2` (dipole in
/// `e*a0`, mass-weighted normal coordinate `Q` in `a0*sqrt(u)`) to the integrated
/// molar absorption coefficient in km/mol.
pub const IR_INTENSITY_AU_TO_KM_PER_MOL: f64 = 974.8801118;

fn atomic_numbers(system: &PeriodicSystem) -> Vec<u8> {
    system.atoms.iter().map(|a| a.z).collect()
}

fn require_nonpbc(system: &PeriodicSystem, what: &str) -> Result<()> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(format!(
            "{what} is implemented for non-periodic systems only"
        )));
    }
    Ok(())
}

fn axis_unit(axis: usize, value: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(value, 0.0, 0.0),
        1 => Vec3::new(0.0, value, 0.0),
        _ => Vec3::new(0.0, 0.0, value),
    }
}

/// First-order Mulliken atomic-charge response from a density response matrix:
/// `dq_A = - sum_{mu in A} sum_nu dP_(mu nu) S_(nu mu)`.
fn atomic_charge_response_from_density(
    system: &PeriodicSystem,
    basis: &BasisSet,
    overlap: &Matrix,
    density_response: &Matrix,
) -> Vec<f64> {
    let n = basis.len();
    let mut dq = vec![0.0_f64; system.atoms.len()];
    for mu in 0..n {
        let mut population = 0.0;
        for nu in 0..n {
            population += density_response[(mu, nu)] * overlap[(nu, mu)];
        }
        dq[basis.aos[mu].atom_index] -= population;
    }
    dq
}

// ---------------------------------------------------------------------------
// Dipole and analytic dipole derivatives (IR)
// ---------------------------------------------------------------------------

/// Analytic Cartesian dipole derivatives.
#[derive(Clone, Debug)]
pub struct DipoleDerivatives {
    /// Mulliken dipole moment (atomic units, `e*a0`).
    pub dipole: Vec3,
    /// `d mu_alpha / d R_coord`, indexed `[coord][alpha]` with `coord = 3*atom + axis`.
    pub ddipole_dr: Vec<[f64; 3]>,
}

/// Analytic dipole derivatives `dmu/dR` from the CPXTB nuclear charge response.
///
/// `d mu_alpha / d R_(B,beta) = delta_(alpha,beta) q_B
///   + sum_A (dq_A/dR_(B,beta)) (R_A - origin)_alpha`,
/// where `dq_A/dR` is the relaxed Mulliken charge response that also builds the
/// analytic Hessian.
pub fn dipole_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    origin: Vec3,
) -> Result<DipoleDerivatives> {
    require_nonpbc(system, "dipole derivatives")?;
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let basis = &electronic.basis;

    let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
        system,
        params,
        electronic,
        crate::cphf::AoDerivativeOptions::default(),
        CpxtbOptions::default(),
    )?;

    let positions = system
        .atoms
        .iter()
        .map(|a| (a.position - origin).to_array())
        .collect::<Vec<_>>();

    let mut ddipole_dr = vec![[0.0_f64; 3]; ndim];
    for coord in 0..ndim {
        let shell_response = &cphf.shell_charge_responses[coord];
        let mut dq_atom = vec![0.0_f64; nat];
        for (ish, shell) in basis.shells.iter().enumerate() {
            dq_atom[shell.atom_index] += shell_response[ish];
        }
        let b = coord / 3;
        let beta = coord % 3;
        ddipole_dr[coord][beta] += electronic.atomic_charges[b];
        for a in 0..nat {
            let dq = dq_atom[a];
            if dq != 0.0 {
                for alpha in 0..3 {
                    ddipole_dr[coord][alpha] += dq * positions[a][alpha];
                }
            }
        }
    }

    Ok(DipoleDerivatives {
        dipole: mulliken_dipole(system, &electronic.atomic_charges, origin),
        ddipole_dr,
    })
}

// ---------------------------------------------------------------------------
// Static polarizability (analytic) and its derivatives (Raman)
// ---------------------------------------------------------------------------

/// Static dipole polarizability and its rotational invariants.
#[derive(Clone, Copy, Debug)]
pub struct Polarizability {
    /// Symmetric polarizability tensor `alpha_(alpha,beta)` (atomic units, `a0^3`).
    pub tensor: [[f64; 3]; 3],
    /// Isotropic mean polarizability `(a_xx + a_yy + a_zz)/3`.
    pub isotropic: f64,
    /// Anisotropy `gamma` (atomic units).
    pub anisotropy: f64,
}

fn tensor_invariants(tensor: &[[f64; 3]; 3]) -> (f64, f64) {
    let isotropic = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let aniso2 = 0.5
        * ((tensor[0][0] - tensor[1][1]).powi(2)
            + (tensor[1][1] - tensor[2][2]).powi(2)
            + (tensor[2][2] - tensor[0][0]).powi(2))
        + 3.0 * (tensor[0][1].powi(2) + tensor[1][2].powi(2) + tensor[0][2].powi(2));
    (isotropic, aniso2.max(0.0).sqrt())
}

fn symmetrize(tensor: &mut [[f64; 3]; 3]) {
    for alpha in 0..3 {
        for beta in (alpha + 1)..3 {
            let avg = 0.5 * (tensor[alpha][beta] + tensor[beta][alpha]);
            tensor[alpha][beta] = avg;
            tensor[beta][alpha] = avg;
        }
    }
}

fn polarizability_from_tensor(mut tensor: [[f64; 3]; 3]) -> Polarizability {
    symmetrize(&mut tensor);
    let (isotropic, anisotropy) = tensor_invariants(&tensor);
    Polarizability {
        tensor,
        isotropic,
        anisotropy,
    }
}

/// Analytic static dipole polarizability `alpha = dmu/dE` from the CPXTB field
/// response. Requires gapped (integer) occupations; use
/// [`static_polarizability_finite_field`] for fractional-occupation systems.
pub fn static_polarizability(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Polarizability> {
    require_nonpbc(system, "polarizability")?;
    let response = solve_field_response(system, params, electronic, CpxtbOptions::default())?;
    let basis = &electronic.basis;
    let overlap = &electronic.integrals.overlap;
    let mut tensor = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let dq = atomic_charge_response_from_density(
            system,
            basis,
            overlap,
            &response.density_responses[beta],
        );
        for (a, &dqa) in dq.iter().enumerate() {
            let r = system.atoms[a].position.to_array();
            for alpha in 0..3 {
                tensor[alpha][beta] += dqa * r[alpha];
            }
        }
    }
    Ok(polarizability_from_tensor(tensor))
}

fn dipole_at_field(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    base: &ElectronicOptions,
    field: Vec3,
) -> Result<Vec3> {
    let mut options = base.clone();
    options.external_field = ExternalFieldOptions {
        electric_field: Some(field),
        magnetic_field: None,
        origin: base.external_field.origin,
    };
    Ok(run_electronic(system, params, options)?.dipole)
}

/// Static polarizability by symmetric finite field (fallback route, e.g. for
/// fractional occupations where the analytic CPXTB response does not apply).
pub fn static_polarizability_finite_field(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    field_step: f64,
) -> Result<Polarizability> {
    require_nonpbc(system, "polarizability")?;
    if !(field_step.is_finite() && field_step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "polarizability field step must be positive".to_string(),
        ));
    }
    let mut tensor = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let plus = dipole_at_field(
            system,
            params,
            electronic_options,
            axis_unit(beta, field_step),
        )?;
        let minus = dipole_at_field(
            system,
            params,
            electronic_options,
            axis_unit(beta, -field_step),
        )?;
        let d = (plus - minus) * (1.0 / (2.0 * field_step));
        let arr = d.to_array();
        for alpha in 0..3 {
            tensor[alpha][beta] = arr[alpha];
        }
    }
    Ok(polarizability_from_tensor(tensor))
}

/// Raw polarizability derivatives `dalpha/dR` (the Raman derivative tensors).
#[derive(Clone, Debug)]
pub struct PolarizabilityDerivatives {
    /// Static polarizability at the reference geometry (analytic).
    pub polarizability: Polarizability,
    /// `d alpha_(alpha,beta) / d R_coord`, indexed `[coord][alpha][beta]`.
    pub dpolarizability_dr: Vec<[[f64; 3]; 3]>,
}

fn dipole_gradient_at_field(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    base: &ElectronicOptions,
    origin: Vec3,
    field: Vec3,
) -> Result<DipoleDerivatives> {
    let mut options = base.clone();
    options.external_field = ExternalFieldOptions {
        electric_field: Some(field),
        magnetic_field: None,
        origin,
    };
    let electronic = run_electronic(system, params, options)?;
    dipole_derivatives(system, params, &electronic, origin)
}

/// Polarizability and its Cartesian derivatives `dalpha/dR`.
///
/// `alpha` is analytic; `dalpha_(alpha,beta)/dR_c = d/dE_beta (dmu_alpha/dR_c)` is
/// the finite-field derivative of the analytic dipole gradient (six CPXTB solves).
pub fn polarizability_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    origin: Vec3,
    field_step: f64,
) -> Result<PolarizabilityDerivatives> {
    require_nonpbc(system, "polarizability derivatives")?;
    if !(field_step.is_finite() && field_step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "polarizability field step must be positive".to_string(),
        ));
    }
    let ndim = 3 * system.atoms.len();

    let electronic = run_electronic(system, params, electronic_options.clone())?;
    let polarizability = static_polarizability(system, params, &electronic)?;

    let mut dpolarizability_dr = vec![[[0.0_f64; 3]; 3]; ndim];
    for beta in 0..3 {
        let plus = dipole_gradient_at_field(
            system,
            params,
            electronic_options,
            origin,
            axis_unit(beta, field_step),
        )?;
        let minus = dipole_gradient_at_field(
            system,
            params,
            electronic_options,
            origin,
            axis_unit(beta, -field_step),
        )?;
        for c in 0..ndim {
            for alpha in 0..3 {
                dpolarizability_dr[c][alpha][beta] =
                    (plus.ddipole_dr[c][alpha] - minus.ddipole_dr[c][alpha]) / (2.0 * field_step);
            }
        }
    }
    for c in 0..ndim {
        symmetrize(&mut dpolarizability_dr[c]);
    }

    Ok(PolarizabilityDerivatives {
        polarizability,
        dpolarizability_dr,
    })
}

// ---------------------------------------------------------------------------
// IR and Raman spectra
// ---------------------------------------------------------------------------

/// One vibrational mode with its IR intensity.
#[derive(Clone, Debug)]
pub struct IrMode {
    pub wavenumber: f64,
    /// `d mu / d Q_k` projected on the normal mode (atomic units) — raw derivative.
    pub dipole_gradient: [f64; 3],
    /// `|d mu / d Q_k|^2` in atomic units (`e^2 / u`).
    pub intensity_au: f64,
    /// Integrated molar absorption coefficient (km/mol).
    pub intensity_km_per_mol: f64,
}

/// Harmonic IR spectrum together with the raw dipole-derivative tensor.
#[derive(Clone, Debug)]
pub struct IrSpectrum {
    pub modes: Vec<IrMode>,
    /// Raw analytic Cartesian dipole derivatives used to build the intensities.
    pub dipole_derivatives: DipoleDerivatives,
}

/// Compute the harmonic IR spectrum (frequencies + analytic intensities).
pub fn ir_spectrum(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    hessian_options: AnalyticHessianOptions,
    origin: Vec3,
) -> Result<IrSpectrum> {
    require_nonpbc(system, "IR spectrum")?;
    let hess = analytic_hessian(system, params, hessian_options)?;
    let electronic = hess.electronic_result.clone().ok_or_else(|| {
        Gfn1Error::InvalidInput("IR spectrum requires the electronic Hessian result".to_string())
    })?;
    let modes = vibrational_analysis(&hess.hessian, &atomic_numbers(system))?;
    let ddip = dipole_derivatives(system, params, &electronic, origin)?;

    let ndim = ddip.ddipole_dr.len();
    let mut ir_modes = Vec::with_capacity(modes.wavenumbers.len());
    for (k, mode) in modes.modes.iter().enumerate() {
        let mut dmu = [0.0_f64; 3];
        for coord in 0..ndim {
            let l = mode[coord];
            for alpha in 0..3 {
                dmu[alpha] += ddip.ddipole_dr[coord][alpha] * l;
            }
        }
        let intensity_au = dmu[0] * dmu[0] + dmu[1] * dmu[1] + dmu[2] * dmu[2];
        ir_modes.push(IrMode {
            wavenumber: modes.wavenumbers[k],
            dipole_gradient: dmu,
            intensity_au,
            intensity_km_per_mol: intensity_au * IR_INTENSITY_AU_TO_KM_PER_MOL,
        });
    }
    Ok(IrSpectrum {
        modes: ir_modes,
        dipole_derivatives: ddip,
    })
}

/// One vibrational mode with its Raman activity.
#[derive(Clone, Debug)]
pub struct RamanMode {
    pub wavenumber: f64,
    /// `d alpha / d Q_k` projected on the normal mode (atomic units) — raw tensor.
    pub dpolarizability_dq: [[f64; 3]; 3],
    /// Isotropic invariant `a' = Tr(dalpha/dQ_k)/3` (atomic units).
    pub mean_polarizability_derivative: f64,
    /// Anisotropy invariant `gamma'^2` (atomic units).
    pub anisotropy_squared: f64,
    /// Raman scattering activity `45 a'^2 + 7 gamma'^2` (atomic units, `a0^4 / u`).
    pub activity: f64,
}

/// Harmonic Raman spectrum together with the raw polarizability-derivative tensor.
#[derive(Clone, Debug)]
pub struct RamanSpectrum {
    pub modes: Vec<RamanMode>,
    /// Raw `dalpha/dR` derivatives used to build the activities.
    pub polarizability_derivatives: PolarizabilityDerivatives,
}

/// Compute the harmonic Raman spectrum.
pub fn raman_spectrum(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    hessian_options: AnalyticHessianOptions,
    origin: Vec3,
    field_step: f64,
) -> Result<RamanSpectrum> {
    require_nonpbc(system, "Raman spectrum")?;
    let ndim = 3 * system.atoms.len();

    let hess = analytic_hessian(system, params, hessian_options.clone())?;
    let modes = vibrational_analysis(&hess.hessian, &atomic_numbers(system))?;
    let pd = polarizability_derivatives(
        system,
        params,
        &hessian_options.electronic_options,
        origin,
        field_step,
    )?;

    let mut raman_modes = Vec::with_capacity(modes.wavenumbers.len());
    for (k, mode) in modes.modes.iter().enumerate() {
        let mut da = [[0.0_f64; 3]; 3];
        for c in 0..ndim {
            let l = mode[c];
            for alpha in 0..3 {
                for beta in 0..3 {
                    da[alpha][beta] += pd.dpolarizability_dr[c][alpha][beta] * l;
                }
            }
        }
        let (mean, gamma) = tensor_invariants(&da);
        let gamma2 = gamma * gamma;
        raman_modes.push(RamanMode {
            wavenumber: modes.wavenumbers[k],
            dpolarizability_dq: da,
            mean_polarizability_derivative: mean,
            anisotropy_squared: gamma2,
            activity: 45.0 * mean * mean + 7.0 * gamma2,
        });
    }
    Ok(RamanSpectrum {
        modes: raman_modes,
        polarizability_derivatives: pd,
    })
}
