// SPDX-License-Identifier: GPL-3.0-or-later
//! Non-PBC Cartesian Hessian assembly.

use crate::basis::BasisSet;
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::{harmonic_average, ShellChargeModel, GFN1_COULOMB_EXPONENT};
use crate::cphf::{
    solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions,
    GammaCartesianCpxtbResult,
};
use crate::data_tables::{atomic_radius_bohr, covalent_radius_d3_bohr};
use crate::dispersion::{dispersion_energy_gradient_hessian, DispersionHessianResult};
use crate::electronic::{run_electronic, ElectronicOptions, ElectronicResult};
use crate::error::{Gfn1Error, Result};
use crate::halogen::{halogen_energy_gradient_hessian, HalogenHessianResult};
use crate::hamiltonian::hscale;
use crate::integrals::{
    contracted_pair_with_derivatives, contracted_pair_with_second_derivatives,
    contracted_pair_with_third_derivatives,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::repulsion::{repulsion_energy_gradient_hessian, RepulsionHessianResult};
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct AnalyticHessianOptions {
    pub include_repulsion: bool,
    pub include_fixed_scc: bool,
    pub include_fixed_pulay: bool,
    pub include_fixed_cn_h0: bool,
    pub include_electronic: bool,
    pub include_dispersion: bool,
    pub include_halogen: bool,
    pub electronic_options: ElectronicOptions,
}

impl Default for AnalyticHessianOptions {
    fn default() -> Self {
        Self {
            include_repulsion: true,
            include_fixed_scc: true,
            include_fixed_pulay: true,
            include_fixed_cn_h0: true,
            include_electronic: true,
            include_dispersion: true,
            include_halogen: true,
            electronic_options: ElectronicOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalyticHessianResult {
    pub hessian: Matrix,
    pub repulsion: Option<RepulsionHessianResult>,
    pub fixed_scc: Option<FixedSccHessianResult>,
    pub fixed_pulay: Option<FixedDensityPulayHessianResult>,
    pub fixed_cn_h0: Option<FixedDensityCnH0HessianResult>,
    pub dispersion: Option<DispersionHessianResult>,
    pub halogen: Option<HalogenHessianResult>,
    pub cpxtb_response: Option<GammaCartesianCpxtbResult>,
    pub electronic_result: Option<ElectronicResult>,
}

#[derive(Clone, Debug)]
pub struct FixedSccHessianResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
}

#[derive(Clone, Debug)]
pub struct FixedDensityPulayHessianResult {
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
}

#[derive(Clone, Debug)]
pub struct FixedDensityCnH0HessianResult {
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
}

pub fn analytic_repulsion_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<RepulsionHessianResult> {
    ensure_non_pbc(system)?;
    repulsion_energy_gradient_hessian(system, params)
}

pub fn analytic_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: AnalyticHessianOptions,
) -> Result<AnalyticHessianResult> {
    ensure_non_pbc(system)?;
    let _profile = crate::profile::scope("hessian.nonpbc.total");
    let electronic = if options.include_fixed_scc
        || options.include_fixed_pulay
        || options.include_fixed_cn_h0
        || options.include_electronic
    {
        Some(run_electronic(
            system,
            params,
            options.electronic_options.clone(),
        )?)
    } else {
        None
    };
    analytic_hessian_from_result(system, params, electronic, options)
}

pub fn analytic_hessian_from_result(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: Option<ElectronicResult>,
    options: AnalyticHessianOptions,
) -> Result<AnalyticHessianResult> {
    let _profile = crate::profile::scope("hessian.assemble.total");
    ensure_non_pbc(system)?;
    let ndof = 3 * system.atoms.len();
    let mut hessian = Matrix::zeros(ndof, ndof);
    let repulsion = if options.include_repulsion {
        let _profile = crate::profile::scope("hessian.repulsion");
        let result = repulsion_energy_gradient_hessian(system, params)?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    let fixed_scc = if options.include_fixed_scc {
        let _profile = crate::profile::scope("hessian.fixed_scc");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput("fixed SCC Hessian requires an ElectronicResult".to_string())
        })?;
        let result = fixed_shell_charge_scc_hessian(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            params,
        )?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    let fixed_pulay = if options.include_fixed_pulay {
        let _profile = crate::profile::scope("hessian.fixed_pulay");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "fixed-density Pulay Hessian requires an ElectronicResult".to_string(),
            )
        })?;
        let result = fixed_density_pulay_hessian(system, params, electronic)?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    let fixed_cn_h0 = if options.include_fixed_cn_h0
        && options.electronic_options.hamiltonian.enable_cn_hamiltonian
    {
        let _profile = crate::profile::scope("hessian.fixed_cn_h0");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "fixed-density CN-H0 Hessian requires an ElectronicResult".to_string(),
            )
        })?;
        let result = fixed_density_cn_h0_hessian(
            system,
            params,
            electronic,
            options.electronic_options.hamiltonian.coordination_cutoff,
        )?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    if options.include_fixed_pulay && options.include_fixed_scc {
        let _profile = crate::profile::scope("hessian.fixed_scalar_overlap");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "fixed metric SCC Hessian requires an ElectronicResult".to_string(),
            )
        })?;
        let scalar_overlap = fixed_density_scalar_overlap_hessian(system, params, electronic)?;
        add_matrix(&mut hessian, &scalar_overlap)?;
    }
    if options.include_fixed_pulay
        && options.include_fixed_cn_h0
        && options.electronic_options.hamiltonian.enable_cn_hamiltonian
    {
        let _profile = crate::profile::scope("hessian.fixed_cn_pulay_cross");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "fixed-density CN-Pulay cross Hessian requires an ElectronicResult".to_string(),
            )
        })?;
        let cross = fixed_density_cn_h0_pulay_cross_hessian(
            system,
            params,
            electronic,
            options.electronic_options.hamiltonian.coordination_cutoff,
        )?;
        add_matrix(&mut hessian, &cross)?;
    }
    let halogen = if options.include_halogen {
        let _profile = crate::profile::scope("hessian.halogen");
        let result = halogen_energy_gradient_hessian(system)?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    let dispersion = if options.include_dispersion && options.electronic_options.enable_dispersion {
        let _profile = crate::profile::scope("hessian.dispersion");
        let result = dispersion_energy_gradient_hessian(
            system,
            params,
            options.electronic_options.d3_reference_path.as_deref(),
        )?;
        add_matrix(&mut hessian, &result.hessian)?;
        Some(result)
    } else {
        None
    };
    let cpxtb_response = if options.include_electronic {
        let _profile = crate::profile::scope("hessian.cpxtb_response");
        let electronic = electronic.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "CPXTB Hessian response requires an ElectronicResult".to_string(),
            )
        })?;
        let response = solve_nonpbc_cpxtb_hessian_response(
            system,
            params,
            electronic,
            AoDerivativeOptions {
                coordination_cutoff: options.electronic_options.hamiltonian.coordination_cutoff,
                include_cn_h0: options.electronic_options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )?;
        if !response.converged {
            return Err(Gfn1Error::InvalidInput(format!(
                "CPXTB Hessian response did not converge; max residual {:.3e}",
                response.max_residual_norm
            )));
        }
        add_matrix(&mut hessian, &response.hessian_response)?;
        Some(response)
    } else {
        None
    };
    // The assembled Hessian is symmetric in exact arithmetic; average it with its
    // transpose to remove the small asymmetry left by finite-precision accumulation
    // across the per-term contributions.
    symmetrize_in_place(&mut hessian);
    Ok(AnalyticHessianResult {
        hessian,
        repulsion,
        fixed_scc,
        fixed_pulay,
        fixed_cn_h0,
        dispersion,
        halogen,
        cpxtb_response,
        electronic_result: electronic,
    })
}

pub fn fixed_shell_charge_scc_hessian(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<FixedSccHessianResult> {
    ensure_non_pbc(system)?;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let nat = system.atoms.len();
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);

    for i in 0..basis.shells.len() {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let ri = system.atoms[ai].position;
            let rj = system.atoms[aj].position;
            let rvec = ri - rj;
            let r = rvec.norm();
            if r <= 1.0e-12 {
                continue;
            }
            let gamma = harmonic_average(shell_model.hardness[i], shell_model.hardness[j]);
            let derivatives = effective_kernel_derivatives(r, gamma);
            let scale = shell_charges[i] * shell_charges[j];
            energy += scale * derivatives.value;
            let gi = rvec * (scale * derivatives.gradient_prefactor);
            gradient[ai] += gi;
            gradient[aj] -= gi;
            add_central_hessian_block(
                &mut hessian,
                ai,
                aj,
                rvec,
                scale * derivatives.gradient_prefactor,
                scale * derivatives.gradient_prefactor_derivative,
            );
        }
    }

    Ok(FixedSccHessianResult {
        energy,
        gradient,
        hessian,
    })
}

/// Charge-path derivative of the SCC2 Hessian: `∂_q[½ Σ_{i≠j} q_i q_j γ_RR_ij]·q^(c)` =
/// `Σ_{i<j} (q^(c)_i q_j + q_i q^(c)_j) K_RR_ij` (the bilinear form of the SCC2 nuclear Hessian with one
/// charge factor replaced by its response). Strict-analytic density-path of the `s2` frozen block — no FD.
pub fn fixed_shell_charge_scc_hessian_charge_path(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    shell_charge_response: &[f64],
    params: &Gfn1Parameters,
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let nat = system.atoms.len();
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
    for i in 0..basis.shells.len() {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let rvec = system.atoms[ai].position - system.atoms[aj].position;
            let r = rvec.norm();
            if r <= 1.0e-12 {
                continue;
            }
            let gamma = harmonic_average(shell_model.hardness[i], shell_model.hardness[j]);
            let d = effective_kernel_derivatives(r, gamma);
            let scale = shell_charge_response[i] * shell_charges[j]
                + shell_charges[i] * shell_charge_response[j];
            add_central_hessian_block(
                &mut hessian,
                ai,
                aj,
                rvec,
                scale * d.gradient_prefactor,
                scale * d.gradient_prefactor_derivative,
            );
        }
    }
    Ok(hessian)
}

/// Analytic third Cartesian derivative of the **frozen-shell-charge** GFN1 second-order
/// electrostatics `E₂ = ½ Σ_{i≠j} q_i q_j γ(R_ij)` (shell charges held fixed), returned as
/// `ndof` slabs. This is the `L_abc` frozen block of the SCC2 electrostatics for the 2n+1
/// driver; with the charges frozen it carries no electronic response, so it FD-validates in
/// isolation against [`fixed_shell_charge_scc_hessian`] (slab `c = ∂H/∂R_c`).
pub fn fixed_shell_charge_scc_third_derivative(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    for i in 0..basis.shells.len() {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let rvec = system.atoms[ai].position - system.atoms[aj].position;
            let r = rvec.norm();
            if r <= 1.0e-12 {
                continue;
            }
            let gamma = harmonic_average(shell_model.hardness[i], shell_model.hardness[j]);
            let d = effective_kernel_derivatives(r, gamma);
            let scale = shell_charges[i] * shell_charges[j];
            // rvec = R_i − R_j is already the true relative vector (σ_i = +1).
            crate::third_derivative::add_radial_third_block(
                &mut tensor,
                ai,
                aj,
                rvec,
                d.gradient_prefactor_derivative,
                d.radial_third_derivative,
                scale,
            );
        }
    }
    Ok(tensor)
}

pub fn fixed_density_pulay_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<FixedDensityPulayHessianResult> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_second(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let p = electronic.density[(mu, nu)];
            let w = electronic.energy_weighted_density[(mu, nu)];
            if p.abs().max(w.abs()) <= 1.0e-18 {
                continue;
            }
            let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
            let overlap_coeff = p * (2.0 * h0.value - scalar_shift) - 2.0 * w;
            gradient[atom_mu] += pair.d_bra[0] * overlap_coeff + h0.d_bra * (2.0 * p * overlap);
            gradient[atom_nu] += pair.d_ket[0] * overlap_coeff + h0.d_ket * (2.0 * p * overlap);

            for row_center in [Center::Bra, Center::Ket] {
                for col_center in [Center::Bra, Center::Ket] {
                    let row_atom = match row_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    let col_atom = match col_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    for row_axis in 0..3 {
                        for col_axis in 0..3 {
                            let d2s = second(
                                &pair.h_bra_bra[0],
                                &pair.h_bra_ket[0],
                                &pair.h_ket_ket[0],
                                row_center,
                                col_center,
                                row_axis,
                                col_axis,
                            );
                            let ds_row =
                                first(&pair.d_bra[0], &pair.d_ket[0], row_center, row_axis);
                            let ds_col =
                                first(&pair.d_bra[0], &pair.d_ket[0], col_center, col_axis);
                            let dh_row = first_vec(h0.d_bra, h0.d_ket, row_center, row_axis);
                            let dh_col = first_vec(h0.d_bra, h0.d_ket, col_center, col_axis);
                            let d2h = second(
                                &h0.h_bra_bra,
                                &h0.h_bra_ket,
                                &h0.h_ket_ket,
                                row_center,
                                col_center,
                                row_axis,
                                col_axis,
                            );
                            let value = overlap_coeff * d2s
                                + 2.0 * p * (dh_col * ds_row + ds_col * dh_row + overlap * d2h);
                            hessian[(3 * row_atom + row_axis, 3 * col_atom + col_axis)] += value;
                        }
                    }
                }
            }
        }
    }
    Ok(FixedDensityPulayHessianResult { gradient, hessian })
}

/// DIAGNOSTIC (pulay 3rd-deriv cross localization): the fixed-density Pulay Hessian split into its two
/// geometric channels — `part_c_sab` = the `C·S_ab` (overlap-coeff × overlap-2nd-deriv) channel, and
/// `part_h0` = the `2p·(h0_c·S_r + h0_r·S_c + S·h0_rc)` (h0-derivative) channel. Same loop/weights as
/// [`fixed_density_pulay_hessian`]; used to FD-attribute the reconverged density-path residual `miss[c]`.
#[cfg(test)]
pub(crate) fn fixed_density_pulay_hessian_parts(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<(Matrix, Matrix)> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let mut part_c_sab = Matrix::zeros(3 * nat, 3 * nat);
    let mut part_h0 = Matrix::zeros(3 * nat, 3 * nat);
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);
    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_second(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let p = electronic.density[(mu, nu)];
            let w = electronic.energy_weighted_density[(mu, nu)];
            if p.abs().max(w.abs()) <= 1.0e-18 {
                continue;
            }
            let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
            let overlap_coeff = p * (2.0 * h0.value - scalar_shift) - 2.0 * w;
            for row_center in [Center::Bra, Center::Ket] {
                for col_center in [Center::Bra, Center::Ket] {
                    let row_atom = match row_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    let col_atom = match col_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    for row_axis in 0..3 {
                        for col_axis in 0..3 {
                            let d2s = second(
                                &pair.h_bra_bra[0], &pair.h_bra_ket[0], &pair.h_ket_ket[0],
                                row_center, col_center, row_axis, col_axis,
                            );
                            let ds_row = first(&pair.d_bra[0], &pair.d_ket[0], row_center, row_axis);
                            let ds_col = first(&pair.d_bra[0], &pair.d_ket[0], col_center, col_axis);
                            let dh_row = first_vec(h0.d_bra, h0.d_ket, row_center, row_axis);
                            let dh_col = first_vec(h0.d_bra, h0.d_ket, col_center, col_axis);
                            let d2h = second(
                                &h0.h_bra_bra, &h0.h_bra_ket, &h0.h_ket_ket,
                                row_center, col_center, row_axis, col_axis,
                            );
                            let (ri, ci) = (3 * row_atom + row_axis, 3 * col_atom + col_axis);
                            part_c_sab[(ri, ci)] += overlap_coeff * d2s;
                            part_h0[(ri, ci)] +=
                                2.0 * p * (dh_col * ds_row + ds_col * dh_row + overlap * d2h);
                        }
                    }
                }
            }
        }
    }
    Ok((part_c_sab, part_h0))
}

/// DIAGNOSTIC (pulay 3rd-deriv, plan step 1): the C:S_ab channel of the fixed-density Pulay Hessian
/// split into its THREE overlap-coefficient sub-blocks, each contracted with the 2nd overlap derivative:
///   `.0` = `[p·2h0]·d²S_ab`, `.1` = `[−p·(v_μ+v_ν)]·d²S_ab`, `.2` = `[−2w]·d²S_ab`.
/// Their sum is exactly `part_c_sab` from [`fixed_density_pulay_hessian_parts`]. Used to attribute the
/// reconverged density-path residual `miss_csab` to the P·2h0 / −P·V / −2W sub-channels (the −2W block is
/// the EWD/W channel; its analytic linearized density-path is `−2·W^(c)·d²S_ab`).
#[cfg(test)]
pub(crate) fn fixed_density_pulay_hessian_csab_subparts(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<(Matrix, Matrix, Matrix)> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let mut sub_p2h0 = Matrix::zeros(3 * nat, 3 * nat);
    let mut sub_npv = Matrix::zeros(3 * nat, 3 * nat);
    let mut sub_n2w = Matrix::zeros(3 * nat, 3 * nat);
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);
    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let h0 =
                h0_prefactor_second(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let p = electronic.density[(mu, nu)];
            let w = electronic.energy_weighted_density[(mu, nu)];
            if p.abs().max(w.abs()) <= 1.0e-18 {
                continue;
            }
            let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
            let c_p2h0 = p * (2.0 * h0.value);
            let c_npv = -p * scalar_shift;
            let c_n2w = -2.0 * w;
            for row_center in [Center::Bra, Center::Ket] {
                for col_center in [Center::Bra, Center::Ket] {
                    let row_atom = match row_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    let col_atom = match col_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    for row_axis in 0..3 {
                        for col_axis in 0..3 {
                            let d2s = second(
                                &pair.h_bra_bra[0], &pair.h_bra_ket[0], &pair.h_ket_ket[0],
                                row_center, col_center, row_axis, col_axis,
                            );
                            let (ri, ci) = (3 * row_atom + row_axis, 3 * col_atom + col_axis);
                            sub_p2h0[(ri, ci)] += c_p2h0 * d2s;
                            sub_npv[(ri, ci)] += c_npv * d2s;
                            sub_n2w[(ri, ci)] += c_n2w * d2s;
                        }
                    }
                }
            }
        }
    }
    Ok((sub_p2h0, sub_npv, sub_n2w))
}

/// The Pulay Hessian's COORDINATION-NUMBER response along one geometric coordinate `R_c`, i.e. the term the
/// analytic Pulay density-path omits: `h0` in [`fixed_density_pulay_hessian`] reads a CN cached in
/// `electronic`, so the reconverged geometric path differentiates it while the base-CN linearized
/// density-path holds it fixed. `cn_grad[atom] = ∂CN_atom/∂R_c`. The CN-derivative of `h0` along `cn_grad`
/// is `s_cn·(hscale·shell_poly)` field-for-field, with `s_cn = −½·(kcn_i·cn_grad[i] + kcn_j·cn_grad[j])`,
/// because `h0 = ½(self_i+self_j)·hscale·poly` and `∂self_i/∂CN_i = −kcn_i`. It carries BOTH pulay geometric
/// channels — the `C:S_ab` (overlap-coeff × d²S) block `2P·h0^cn·d²S_ab` AND the h0-derivative block
/// `2P·(h0^cn_col·dS_row + dS_col·h0^cn_row + S·h0^cn_rc)`. FROZEN-density, first-order (∂CN/∂R) — no 2nd-order
/// response. `−P·V` and `−2W` have no CN-dependence, so only the `p·2h0` part contributes.
pub(crate) fn fixed_density_pulay_cn_h0_response(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cn_grad: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let mut out = Matrix::zeros(3 * nat, 3 * nat);
    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            let w = electronic.energy_weighted_density[(mu, nu)];
            if p.abs().max(w.abs()) <= 1.0e-18 {
                continue;
            }
            let kcn_i = basis.shells[shell_mu_index].kcn_raw.unwrap_or(0.0);
            let kcn_j = basis.shells[shell_nu_index].kcn_raw.unwrap_or(0.0);
            // s_cn = ∂base/∂R_c / hscale = −½(kcn_i·∂CN_i/∂R_c + kcn_j·∂CN_j/∂R_c).
            let s_cn = -0.5 * (kcn_i * cn_grad[atom_mu] + kcn_j * cn_grad[atom_nu]);
            if s_cn == 0.0 {
                continue;
            }
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            // CN-derivative of h0 along cn_grad = s_cn · (hscale·poly) field-for-field.
            let hs = h0_scale_second(system, params, shell_mu_index, shell_nu_index, basis)?;
            let h0cn = H0Second {
                value: s_cn * hs.value,
                d_bra: hs.d_bra * s_cn,
                d_ket: hs.d_ket * s_cn,
                h_bra_bra: scale_3x3(hs.h_bra_bra, s_cn),
                h_bra_ket: scale_3x3(hs.h_bra_ket, s_cn),
                h_ket_ket: scale_3x3(hs.h_ket_ket, s_cn),
            };
            let overlap_coeff_cn = 2.0 * p * h0cn.value;
            for row_center in [Center::Bra, Center::Ket] {
                for col_center in [Center::Bra, Center::Ket] {
                    let row_atom = match row_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    let col_atom = match col_center {
                        Center::Bra => atom_mu,
                        Center::Ket => atom_nu,
                    };
                    for row_axis in 0..3 {
                        for col_axis in 0..3 {
                            let d2s = second(
                                &pair.h_bra_bra[0], &pair.h_bra_ket[0], &pair.h_ket_ket[0],
                                row_center, col_center, row_axis, col_axis,
                            );
                            let ds_row = first(&pair.d_bra[0], &pair.d_ket[0], row_center, row_axis);
                            let ds_col = first(&pair.d_bra[0], &pair.d_ket[0], col_center, col_axis);
                            let dh_row = first_vec(h0cn.d_bra, h0cn.d_ket, row_center, row_axis);
                            let dh_col = first_vec(h0cn.d_bra, h0cn.d_ket, col_center, col_axis);
                            let d2h = second(
                                &h0cn.h_bra_bra, &h0cn.h_bra_ket, &h0cn.h_ket_ket,
                                row_center, col_center, row_axis, col_axis,
                            );
                            let value = overlap_coeff_cn * d2s
                                + 2.0 * p * (dh_col * ds_row + ds_col * dh_row + overlap * d2h);
                            let (ri, ci) = (3 * row_atom + row_axis, 3 * col_atom + col_axis);
                            out[(ri, ci)] += value;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn fixed_density_cn_h0_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
) -> Result<FixedDensityCnH0HessianResult> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut d_edcn = vec![0.0; nat];
    let mut d_edcn_dr = Matrix::zeros(nat, ndof);

    for (ish, shell) in basis.shells.iter().enumerate() {
        let dsedcn = -shell.kcn_raw.unwrap_or(0.0);
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            d_edcn[shell.atom_index] += dsedcn * electronic.density[(iao, iao)];
        }
        let _ = ish;
    }

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let shell_mu = &basis.shells[shell_mu_index];
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let shell_nu = &basis.shells[shell_nu_index];
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let scale = h0_scale_second(
                system,
                params,
                shell_mu_index,
                shell_nu_index,
                &electronic.basis,
            )?;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let value = p * scale.value * overlap;
            let dval_bra = (pair.d_bra[0] * scale.value + scale.d_bra * overlap) * p;
            let dval_ket = (pair.d_ket[0] * scale.value + scale.d_ket * overlap) * p;
            let dsedcn_mu = -shell_mu.kcn_raw.unwrap_or(0.0);
            let dsedcn_nu = -shell_nu.kcn_raw.unwrap_or(0.0);

            d_edcn[atom_mu] += dsedcn_mu * value;
            d_edcn[atom_nu] += dsedcn_nu * value;
            add_vec3_to_row_dof(&mut d_edcn_dr, atom_mu, atom_mu, dval_bra * dsedcn_mu);
            add_vec3_to_row_dof(&mut d_edcn_dr, atom_mu, atom_nu, dval_ket * dsedcn_mu);
            add_vec3_to_row_dof(&mut d_edcn_dr, atom_nu, atom_mu, dval_bra * dsedcn_nu);
            add_vec3_to_row_dof(&mut d_edcn_dr, atom_nu, atom_nu, dval_ket * dsedcn_nu);
        }
    }

    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut hessian = Matrix::zeros(ndof, ndof);
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let rvec = pair.r_ij;
        let r = rvec.norm();
        if r <= 1.0e-12 {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let derivatives = coordination_value_derivatives(CoordinationOptions::default().kcn, r, rc);
        let c = d_edcn[pair.i] + d_edcn[pair.j];
        let pref = c * derivatives.first / r;
        let dpref = c * (derivatives.second / r - derivatives.first / (r * r));
        gradient[pair.i] += rvec * pref;
        gradient[pair.j] -= rvec * pref;
        add_central_hessian_block(&mut hessian, pair.i, pair.j, rvec, pref, dpref);

        let cn_grad_scale = derivatives.first / r;
        for col in 0..ndof {
            let dc = d_edcn_dr[(pair.i, col)] + d_edcn_dr[(pair.j, col)];
            if dc.abs() <= 1.0e-18 {
                continue;
            }
            let scaled = rvec * (cn_grad_scale * dc);
            for axis in 0..3 {
                hessian[(3 * pair.i + axis, col)] += scaled.to_array()[axis];
                hessian[(3 * pair.j + axis, col)] -= scaled.to_array()[axis];
            }
        }
    }
    Ok(FixedDensityCnH0HessianResult { gradient, hessian })
}

pub fn fixed_density_cn_h0_pulay_cross_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let cn_derivatives = coordination_number_first_derivatives(system, coordination_cutoff)?;
    let mut hessian = Matrix::zeros(ndof, ndof);

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let shell_mu = &basis.shells[shell_mu_index];
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let shell_nu = &basis.shells[shell_nu_index];
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let (moments, d_bra, d_ket) =
                contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = moments[0];
            let scale = h0_scale_second(
                system,
                params,
                shell_mu_index,
                shell_nu_index,
                &electronic.basis,
            )?;
            let dsedcn_mu = -shell_mu.kcn_raw.unwrap_or(0.0);
            let dsedcn_nu = -shell_nu.kcn_raw.unwrap_or(0.0);
            if dsedcn_mu.abs().max(dsedcn_nu.abs()) <= 1.0e-30 {
                continue;
            }
            let active_cols = (0..ndof)
                .filter_map(|col| {
                    let dbase = 0.5
                        * (dsedcn_mu * cn_derivatives[(atom_mu, col)]
                            + dsedcn_nu * cn_derivatives[(atom_nu, col)]);
                    (dbase.abs() > 1.0e-30).then_some((col, dbase))
                })
                .collect::<Vec<_>>();
            for (col, dbase) in active_cols {
                let dh = dbase * scale.value;
                let dh_bra = scale.d_bra * dbase;
                let dh_ket = scale.d_ket * dbase;
                let row_bra = (d_bra[0] * dh + dh_bra * overlap) * (2.0 * p);
                let row_ket = (d_ket[0] * dh + dh_ket * overlap) * (2.0 * p);
                add_vec3_to_column_dof(&mut hessian, atom_mu, col, row_bra);
                add_vec3_to_column_dof(&mut hessian, atom_nu, col, row_ket);
            }
        }
    }
    Ok(hessian)
}

pub fn fixed_density_scalar_overlap_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let shell_scalar_derivatives =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let mut hessian = Matrix::zeros(ndof, ndof);

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let active_cols = (0..ndof)
                .filter_map(|col_coord| {
                    let dscalar_col = shell_scalar_derivatives[(shell_mu_index, col_coord)]
                        + shell_scalar_derivatives[(shell_nu_index, col_coord)];
                    (dscalar_col.abs() > 1.0e-30).then_some((col_coord, dscalar_col))
                })
                .collect::<Vec<_>>();
            if active_cols.is_empty() {
                continue;
            }
            let (_, d_bra, d_ket) =
                contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            for row_center in [Center::Bra, Center::Ket] {
                let row_atom = match row_center {
                    Center::Bra => atom_mu,
                    Center::Ket => atom_nu,
                };
                for row_axis in 0..3 {
                    let ds_row = first(&d_bra[0], &d_ket[0], row_center, row_axis);
                    if ds_row.abs() <= 1.0e-30 {
                        continue;
                    }
                    let row_coord = 3 * row_atom + row_axis;
                    for &(col_coord, dscalar_col) in &active_cols {
                        hessian[(row_coord, col_coord)] -= p * ds_row * dscalar_col;
                    }
                }
            }
        }
    }
    Ok(hessian)
}

#[derive(Clone, Copy, Debug)]
struct ScalarDerivatives {
    first: f64,
    second: f64,
    /// Third radial derivative of the smooth CN counting function, for the analytic third
    /// nuclear derivative (CN-H0 and D3 chain rules).
    third: f64,
}

fn coordination_value_derivatives(kcn: f64, r: f64, rc: f64) -> ScalarDerivatives {
    let raw_arg = -kcn * (rc / r - 1.0);
    if !(-80.0..=80.0).contains(&raw_arg) {
        return ScalarDerivatives {
            first: 0.0,
            second: 0.0,
            third: 0.0,
        };
    }
    let expterm = raw_arg.exp();
    let denom = 1.0 + expterm;
    let arg1 = kcn * rc / (r * r);
    let arg2 = -2.0 * kcn * rc / (r * r * r);
    let arg3 = 6.0 * kcn * rc / (r * r * r * r);
    let first = -expterm * arg1 / (denom * denom);
    let second = -expterm * (arg1 * arg1 + arg2) / (denom * denom)
        + 2.0 * expterm * expterm * arg1 * arg1 / (denom * denom * denom);
    // Third derivative via the sigmoid `σ = 1/denom`: with `cn = σ(arg(r))`,
    // `cn''' = σ₃ arg'³ + 3 σ₂ arg' arg'' + σ₁ arg'''`, where
    // `σ₁ = −σ(1−σ)`, `σ₂ = σ(1−σ)(1−2σ)`, `σ₃ = −σ(1−σ)(1−6σ+6σ²)`. (σ₂ form matches `second`.)
    let sig = 1.0 / denom;
    let a = sig * (1.0 - sig); // σ(1−σ)
    let sig1 = -a;
    let sig2 = a * (1.0 - 2.0 * sig);
    let sig3 = -a * (1.0 - 6.0 * sig + 6.0 * sig * sig);
    let third = sig3 * arg1 * arg1 * arg1 + 3.0 * sig2 * arg1 * arg2 + sig1 * arg3;
    ScalarDerivatives {
        first,
        second,
        third,
    }
}

fn add_vec3_to_row_dof(matrix: &mut Matrix, row: usize, atom: usize, value: Vec3) {
    matrix[(row, 3 * atom)] += value.x;
    matrix[(row, 3 * atom + 1)] += value.y;
    matrix[(row, 3 * atom + 2)] += value.z;
}

fn add_vec3_to_column_dof(matrix: &mut Matrix, atom: usize, col: usize, value: Vec3) {
    matrix[(3 * atom, col)] += value.x;
    matrix[(3 * atom + 1, col)] += value.y;
    matrix[(3 * atom + 2, col)] += value.z;
}

fn coordination_number_first_derivatives(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
) -> Result<Matrix> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let mut out = Matrix::zeros(nat, ndof);
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let r = pair.r_ij.norm();
        if r <= 1.0e-12 {
            continue;
        }
        let deriv = pair.r_ij * (pair.dcn_dr / r);
        add_vec3_to_row_dof(&mut out, pair.i, pair.i, deriv);
        add_vec3_to_row_dof(&mut out, pair.j, pair.i, deriv);
        add_vec3_to_row_dof(&mut out, pair.i, pair.j, -deriv);
        add_vec3_to_row_dof(&mut out, pair.j, pair.j, -deriv);
    }
    Ok(out)
}

/// Plain coordination-number SECOND derivatives `∂²CN[atom]/∂R_b∂R_c`, returned as one `ndof×ndof` matrix
/// per atom. Mirrors `coordination_number_first_derivatives` (the `deriv = u·f'` per pair, scattered to
/// both atoms' CN) one order higher: the radial Hessian `f''·u⊗u + (f'/r)(I−u⊗u)` (= `add_central_hessian_block`
/// with `pref=f'/r`, `dpref=f''/r−f'/r²`) added to BOTH `cn[i]` and `cn[j]` (each counts the pair).
fn coordination_number_second_derivatives(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
) -> Result<Vec<Matrix>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let kcn = CoordinationOptions::default().kcn;
    let mut out = vec![Matrix::zeros(ndof, ndof); nat];
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let rvec = pair.r_ij;
        let r = rvec.norm();
        if r <= 1.0e-12 {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let d = coordination_value_derivatives(kcn, r, rc);
        let pref = d.first / r;
        let dpref = d.second / r - d.first / (r * r);
        add_central_hessian_block(&mut out[pair.i], pair.i, pair.j, rvec, pref, dpref);
        add_central_hessian_block(&mut out[pair.j], pair.i, pair.j, rvec, pref, dpref);
    }
    Ok(out)
}

fn ao_scalar_potentials(basis: &BasisSet, shell_potentials: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; basis.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            out[iao] = shell_potentials.get(ish).copied().unwrap_or(0.0);
        }
    }
    out
}

/// Geometric first derivatives of the per-shell SCC scalar potential at FIXED charges,
/// `out[(s, dof)] = ∂V_s/∂R_dof|_q = Σ_t (∂γ_{st}/∂R_dof) q_t`. (`V_s = Σ_t γ_{st} q_t`.) Pub so the
/// strict-analytic third derivative can form the TOTAL `dV/dR_c = ∂V/∂R|_q + E_qq·q^(c)`.
pub fn shell_scalar_potential_first_derivatives(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<Matrix> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let nsh = basis.shells.len();
    if shell_charges.len() != nsh {
        return Err(Gfn1Error::InvalidInput(
            "shell charge dimension mismatch for scalar-potential derivatives".to_string(),
        ));
    }
    let model = ShellChargeModel::build(system, basis, params)?;
    let mut out = Matrix::zeros(nsh, ndof);
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let rvec = system.atoms[ai].position - system.atoms[aj].position;
            let r = rvec.norm();
            if r <= 1.0e-12 {
                continue;
            }
            let gamma = harmonic_average(model.hardness[i], model.hardness[j]);
            let pref = effective_kernel_derivatives(r, gamma).gradient_prefactor;
            let dk = rvec * pref;
            for axis in 0..3 {
                let value = dk.to_array()[axis];
                out[(i, 3 * ai + axis)] += value * shell_charges[j];
                out[(j, 3 * ai + axis)] += value * shell_charges[i];
                out[(i, 3 * aj + axis)] -= value * shell_charges[j];
                out[(j, 3 * aj + axis)] -= value * shell_charges[i];
            }
        }
    }
    Ok(out)
}

/// Second nuclear derivatives of the per-shell SCC scalar potential `V_s = Σ_t γ_{st}(R) q_t`, at FIXED
/// charges: `out[s][(b,c)] = ∂²V_s/∂R_b∂R_c = Σ_t (∂²γ_{st}/∂R_b∂R_c) q_t`. Mirrors
/// `shell_scalar_potential_first_derivatives` but contracts the central kernel HESSIAN block
/// `K_ab = p·δ_ab + r·p'·u_a·u_b` (same form as `add_central_hessian_block`) with the charge of the OTHER
/// shell. Used by the scalar-overlap third derivative (the `∂_c dscalar_b` factor) and the Group-A
/// `scc_kernel` response-gradient derivative (`D_c scc_kernel[a] = Σ_s Δq_s·(d2vdr_q[s][(a,c)]+dvdr_qc[(s,a)])`).
pub fn shell_scalar_potential_second_derivatives(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<Vec<Matrix>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let nsh = basis.shells.len();
    if shell_charges.len() != nsh {
        return Err(Gfn1Error::InvalidInput(
            "shell charge dimension mismatch for scalar-potential second derivatives".to_string(),
        ));
    }
    let model = ShellChargeModel::build(system, basis, params)?;
    let mut out = vec![Matrix::zeros(ndof, ndof); nsh];
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let rvec = system.atoms[ai].position - system.atoms[aj].position;
            let r = rvec.norm();
            if r <= 1.0e-12 {
                continue;
            }
            let gamma = harmonic_average(model.hardness[i], model.hardness[j]);
            let d = effective_kernel_derivatives(r, gamma);
            let u = (rvec / r).to_array();
            for a in 0..3 {
                for b in 0..3 {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    let k = d.gradient_prefactor * delta
                        + r * d.gradient_prefactor_derivative * u[a] * u[b];
                    // ∂²γ_{ij} block (same structure as add_central_hessian_block), weighted by the OTHER
                    // shell's charge for each potential.
                    let wi = k * shell_charges[j]; // contributes to ∂²V_i
                    let wj = k * shell_charges[i]; // contributes to ∂²V_j
                    for (sh, w) in [(i, wi), (j, wj)] {
                        out[sh][(3 * ai + a, 3 * ai + b)] += w;
                        out[sh][(3 * aj + a, 3 * aj + b)] += w;
                        out[sh][(3 * ai + a, 3 * aj + b)] -= w;
                        out[sh][(3 * aj + a, 3 * ai + b)] -= w;
                    }
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
enum Center {
    Bra,
    Ket,
}

#[derive(Clone, Debug)]
struct H0Second {
    value: f64,
    d_bra: Vec3,
    d_ket: Vec3,
    h_bra_bra: [[f64; 3]; 3],
    h_bra_ket: [[f64; 3]; 3],
    h_ket_ket: [[f64; 3]; 3],
}

fn h0_prefactor_second(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    shell_mu: usize,
    shell_nu: usize,
) -> Result<H0Second> {
    let si = &electronic.basis.shells[shell_mu];
    let sj = &electronic.basis.shells[shell_nu];
    let self_i = shell_self_energy(si, electronic.coordination_numbers[si.atom_index]);
    let self_j = shell_self_energy(sj, electronic.coordination_numbers[sj.atom_index]);
    let base = 0.5 * (self_i + self_j) * hscale(si, sj, params)?;
    let poly = shell_poly_second(
        system,
        si.atom_index,
        sj.atom_index,
        si.z,
        sj.z,
        si.poly_raw,
        sj.poly_raw,
    )?;
    Ok(H0Second {
        value: base * poly.value,
        d_bra: poly.d_bra * base,
        d_ket: poly.d_ket * base,
        h_bra_bra: scale_3x3(poly.h_bra_bra, base),
        h_bra_ket: scale_3x3(poly.h_bra_ket, base),
        h_ket_ket: scale_3x3(poly.h_ket_ket, base),
    })
}

fn h0_scale_second(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    shell_mu: usize,
    shell_nu: usize,
    basis: &BasisSet,
) -> Result<H0Second> {
    let si = &basis.shells[shell_mu];
    let sj = &basis.shells[shell_nu];
    let base = hscale(si, sj, params)?;
    let poly = shell_poly_second(
        system,
        si.atom_index,
        sj.atom_index,
        si.z,
        sj.z,
        si.poly_raw,
        sj.poly_raw,
    )?;
    Ok(H0Second {
        value: base * poly.value,
        d_bra: poly.d_bra * base,
        d_ket: poly.d_ket * base,
        h_bra_bra: scale_3x3(poly.h_bra_bra, base),
        h_bra_ket: scale_3x3(poly.h_bra_ket, base),
        h_ket_ket: scale_3x3(poly.h_ket_ket, base),
    })
}

fn shell_self_energy(shell: &crate::basis::BasisShell, cn: f64) -> f64 {
    shell.hdiag_ha - shell.kcn_raw.unwrap_or(0.0) * cn
}

/// Bare-H0 FIRST nuclear derivative `∂H0/∂R_b` (n×n) at FIXED coordination number, where
/// `H0_μν = self_avg·scale·overlap` (`self_avg = ½(self_i+self_j)`, `scale = hscale·shell_poly`). This is
/// the H0-only contribution to `AoDerivativeMatrices[b].h0_deriv` with the CN-self-energy held fixed and the
/// SCC-scalar coupling excluded (the `D_c(CᵀF_bC)` block ladder: H0 block first). Matches
/// `h0_bare_second_derivative_matrix`'s FD.
pub fn h0_bare_first_derivative_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    b: usize,
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let cn = &electronic.coordination_numbers;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            if atom_b != atom_mu && atom_b != atom_nu {
                continue;
            }
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let self_avg = 0.5
                * (shell_self_energy(&basis.shells[shell_mu], cn[atom_mu])
                    + shell_self_energy(&basis.shells[shell_nu], cn[atom_nu]));
            let scale = h0_scale_second(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let cb = if atom_b == atom_mu {
                Center::Bra
            } else {
                Center::Ket
            };
            let scale_b = first_vec(scale.d_bra, scale.d_ket, cb, axis_b);
            let ov_b = first(&pair.d_bra[0], &pair.d_ket[0], cb, axis_b);
            out[(mu, nu)] = self_avg * (scale_b * pair.moments[0] + scale.value * ov_b);
        }
    }
    Ok(out)
}

/// The SCC-scalar contribution to `AoDerivativeMatrices[b].h0_deriv`, in isolation:
/// `−scalar_shift·ds_b − dscalar_b·overlap` (`scalar_shift = ½(V_μ+V_ν)`, `dscalar_b = ½(∂V/∂R_b|_q)`).
/// Lets the SCC block of `F_bc` be FD-gated without the entangled H0/CN prefactor.
pub fn h0_scc_scalar_first_derivative_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    b: usize,
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let v = &electronic.shell_scc_potential;
    let dvdr_q =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            if atom_mu == atom_nu {
                // SAME-ATOM (incl. on-site diagonal): add_scalar contributes `−dscalar_b·S_μν` here too;
                // add_center's `−scalar_shift·ds_b` vanishes (same-atom overlap is geometry-rigid).
                let ov = electronic.integrals.overlap[(mu, nu)];
                if ov.abs() > 1.0e-30 {
                    let dscalar_b = 0.5 * (dvdr_q[(shell_mu, b)] + dvdr_q[(shell_nu, b)]);
                    out[(mu, nu)] = -dscalar_b * ov;
                }
                continue;
            }
            let rnu = system.atoms[atom_nu].position;
            let (_, d_bra, d_ket) =
                contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let cb = if atom_b == atom_mu {
                Some(Center::Bra)
            } else if atom_b == atom_nu {
                Some(Center::Ket)
            } else {
                None
            };
            let ds_b = cb.map_or(0.0, |x| first(&d_bra[0], &d_ket[0], x, axis_b));
            let scalar_shift = 0.5 * (v[shell_mu] + v[shell_nu]);
            let dscalar_b = 0.5 * (dvdr_q[(shell_mu, b)] + dvdr_q[(shell_nu, b)]);
            out[(mu, nu)] = -scalar_shift * ds_b - dscalar_b * pair.moments[0];
        }
    }
    Ok(out)
}

/// SCC-scalar block of `F_bc = ∂²(AoDerivativeMatrices[b].h0_deriv)/∂R_c`: the second derivative of the
/// SCC-scalar contribution to `h0_deriv[b]`, which is `−scalar_shift·ds_b − dscalar_b·overlap`
/// (`scalar_shift = ½(V_μ+V_ν)`, `dscalar_b = ½(∂V/∂R_b|_q)_{μ,ν}`). Applying `D_c`:
///   `−(D_c scalar_shift)·ds_b − scalar_shift·overlap_bc − (D_c dscalar_b)·overlap − dscalar_b·overlap_c`,
/// with `D_c scalar_shift = ½(v_c_μ+v_c_ν)` where `v_c` is the **TOTAL** `dV/dR_c = ∂V/∂R_c|_q + E_qq·q^(c)`
/// (passed in), and `D_c dscalar_b = ½(∂²V/∂R_b∂R_c|_q)_{μ,ν} + ½(∂V/∂R_b)|_{q^(c)}` (geometric kernel 2nd
/// derivative + charge-path). The SCC block of the `D_c(CᵀF_bC)` ladder (Step 2).
#[allow(clippy::too_many_arguments)]
pub fn h0_scc_scalar_second_derivative_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v_c: &[f64],
    q_c: &[f64],
    b: usize,
    c: usize,
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let (atom_c, axis_c) = (c / 3, c % 3);
    let v = &electronic.shell_scc_potential;
    let dvdr_q =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let dvdr_qc = shell_scalar_potential_first_derivatives(system, basis, q_c, params)?;
    let d2vdr_q = shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            if atom_mu == atom_nu {
                // SAME-ATOM (incl. on-site diagonal): add_scalar applies `−dscalar·overlap` here too.
                // The overlap S_μν is geometry-rigid (all overlap derivatives vanish), so the second
                // derivative is `−dc_dscalar_b · S_μν` with S_μν the ACTUAL SCF overlap (not the broken
                // zero-separation moments[0]).
                let ov = electronic.integrals.overlap[(mu, nu)];
                if ov.abs() > 1.0e-30 {
                    let dc_dscalar_b = 0.5
                        * (d2vdr_q[shell_mu][(b, c)] + d2vdr_q[shell_nu][(b, c)])
                        + 0.5 * (dvdr_qc[(shell_mu, b)] + dvdr_qc[(shell_nu, b)]);
                    out[(mu, nu)] = -dc_dscalar_b * ov;
                }
                continue;
            }
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let ov = pair.moments[0];
            // overlap derivatives w.r.t. b and c (only the pair's centers contribute geometrically)
            let cb = if atom_b == atom_mu {
                Some(Center::Bra)
            } else if atom_b == atom_nu {
                Some(Center::Ket)
            } else {
                None
            };
            let cc = if atom_c == atom_mu {
                Some(Center::Bra)
            } else if atom_c == atom_nu {
                Some(Center::Ket)
            } else {
                None
            };
            let ds_b = cb.map_or(0.0, |x| first(&pair.d_bra[0], &pair.d_ket[0], x, axis_b));
            let ds_c = cc.map_or(0.0, |x| first(&pair.d_bra[0], &pair.d_ket[0], x, axis_c));
            let ov_bc = match (cb, cc) {
                (Some(x), Some(y)) => second(
                    &pair.h_bra_bra[0],
                    &pair.h_bra_ket[0],
                    &pair.h_ket_ket[0],
                    x,
                    y,
                    axis_b,
                    axis_c,
                ),
                _ => 0.0,
            };
            let scalar_shift = 0.5 * (v[shell_mu] + v[shell_nu]);
            let dc_scalar_shift = 0.5 * (v_c[shell_mu] + v_c[shell_nu]);
            let dscalar_b = 0.5 * (dvdr_q[(shell_mu, b)] + dvdr_q[(shell_nu, b)]);
            let dc_dscalar_b = 0.5 * (d2vdr_q[shell_mu][(b, c)] + d2vdr_q[shell_nu][(b, c)])
                + 0.5 * (dvdr_qc[(shell_mu, b)] + dvdr_qc[(shell_nu, b)]);
            out[(mu, nu)] = -dc_scalar_shift * ds_b
                - scalar_shift * ov_bc
                - dc_dscalar_b * ov
                - dscalar_b * ds_c;
        }
    }
    Ok(out)
}

/// CN-H0 block of `F_bc`: the coordination-number coupling that `h0_bare_second` (fixed CN) omits, i.e.
/// `D_c` of the CN-dependent part of `h0_deriv[b] = h0_bare_first + add_cn_h0`:
///   Part A = `∂(h0_bare_first)/∂CN · CN^(c)` = `−½(kcn_μ·CN_c_μ + kcn_ν·CN_c_ν)·(scale_b·ov + scale·ov_b)`,
///   Part B = `D_c(add_cn_h0[b])`, `add_cn_h0[b] = ov·g_b`, `g_b = −½(kcn_μ·scale·CN_b_μ + kcn_ν·scale·CN_b_ν)`,
///            so Part B = `ov_c·g_b + ov·(−½)(kcn_μ(scale_c·CN_b_μ + scale·CN_bc_μ) + kcn_ν(...))`.
/// `CN_b/CN_c/CN_bc` are coordination-number 1st/2nd derivatives (`cn_h0_cn_jets`, many-body); `scale` is the
/// H0 geometric scale (`h0_scale_second`). The last (heaviest) block of `F_bc` for the `D_c(CᵀF_bC)` ladder.
pub fn h0_cn_block_second_derivative_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    b: usize,
    c: usize,
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let n = basis.len();
    let ndof = 3 * system.atoms.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let (atom_c, axis_c) = (c / 3, c % 3);
    // PLAIN coordination-number derivatives (NOT cn_h0_cn_jets, which are dsedcn·p·(scale·overlap) jets).
    let cn1 = coordination_number_first_derivatives(system, coordination_cutoff)?; // [(atom, dof)]
    let cn2 = coordination_number_second_derivatives(system, coordination_cutoff)?; // per-atom ndof×ndof
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let kcn_mu = basis.shells[shell_mu].kcn_raw.unwrap_or(0.0);
            let kcn_nu = basis.shells[shell_nu].kcn_raw.unwrap_or(0.0);
            if kcn_mu == 0.0 && kcn_nu == 0.0 {
                continue;
            }
            if atom_mu == atom_nu {
                // ON-SITE CN-H0: add_cn_h0 (cphf.rs) builds `dh0 = S_μν · (−½k_μ·CN[A] − ½k_ν·CN[A])`
                // with coeff = −½k (NO scale) and S_μν the ACTUAL SCF overlap. Same-atom overlap is
                // geometry-rigid (∂_c S_μν = 0 for ALL c), so ∂_b∂_c reduces to S_μν · coeff · CN_bc.
                // (Must read the real S from electronic.integrals.overlap; the second-derivative pair
                // helper returns a broken moments[0] at zero separation for the on-site diagonal.)
                let s_mn = electronic.integrals.overlap[(mu, nu)];
                let cn_bc = cn2[atom_mu][(b, c)];
                out[(mu, nu)] = -0.5 * (kcn_mu + kcn_nu) * s_mn * cn_bc;
                continue;
            }
            let scale = h0_scale_second(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let ov = pair.moments[0];
            let cb = if atom_b == atom_mu {
                Some(Center::Bra)
            } else if atom_b == atom_nu {
                Some(Center::Ket)
            } else {
                None
            };
            let cc = if atom_c == atom_mu {
                Some(Center::Bra)
            } else if atom_c == atom_nu {
                Some(Center::Ket)
            } else {
                None
            };
            let scale_b = cb.map_or(0.0, |x| first_vec(scale.d_bra, scale.d_ket, x, axis_b));
            let scale_c = cc.map_or(0.0, |x| first_vec(scale.d_bra, scale.d_ket, x, axis_c));
            let ov_b = cb.map_or(0.0, |x| first(&pair.d_bra[0], &pair.d_ket[0], x, axis_b));
            let ov_c = cc.map_or(0.0, |x| first(&pair.d_bra[0], &pair.d_ket[0], x, axis_c));
            // CN derivatives (many-body)
            let cn_b_mu = cn1[(atom_mu, b)];
            let cn_b_nu = cn1[(atom_nu, b)];
            let cn_c_mu = cn1[(atom_mu, c)];
            let cn_c_nu = cn1[(atom_nu, c)];
            let cn_bc_mu = cn2[atom_mu][(b, c)];
            let cn_bc_nu = cn2[atom_nu][(b, c)];
            let _ = ndof;
            // Part A
            let bare_b = scale_b * ov + scale.value * ov_b;
            let part_a = -0.5 * (kcn_mu * cn_c_mu + kcn_nu * cn_c_nu) * bare_b;
            // Part B
            let g_b = -0.5 * (kcn_mu * scale.value * cn_b_mu + kcn_nu * scale.value * cn_b_nu);
            let dg_b = -0.5
                * (kcn_mu * (scale_c * cn_b_mu + scale.value * cn_bc_mu)
                    + kcn_nu * (scale_c * cn_b_nu + scale.value * cn_bc_nu));
            let part_b = ov_c * g_b + ov * dg_b;
            out[(mu, nu)] = part_a + part_b;
        }
    }
    Ok(out)
}

/// Bare-H0 SECOND nuclear derivative `∂²H0/∂R_b∂R_c` (n×n) at FIXED CN: `self_avg·(scale_bc·overlap +
/// scale_b·overlap_c + scale_c·overlap_b + scale·overlap_bc)`. The H0-only block of `F_bc` for the
/// `D_c(CᵀF_bC)` ladder.
pub fn h0_bare_second_derivative_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    b: usize,
    c: usize,
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let (atom_c, axis_c) = (c / 3, c % 3);
    let cn = &electronic.coordination_numbers;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            if (atom_b != atom_mu && atom_b != atom_nu) || (atom_c != atom_mu && atom_c != atom_nu)
            {
                continue;
            }
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let self_avg = 0.5
                * (shell_self_energy(&basis.shells[shell_mu], cn[atom_mu])
                    + shell_self_energy(&basis.shells[shell_nu], cn[atom_nu]));
            let scale = h0_scale_second(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let cb = if atom_b == atom_mu {
                Center::Bra
            } else {
                Center::Ket
            };
            let cc = if atom_c == atom_mu {
                Center::Bra
            } else {
                Center::Ket
            };
            let scale_b = first_vec(scale.d_bra, scale.d_ket, cb, axis_b);
            let scale_c = first_vec(scale.d_bra, scale.d_ket, cc, axis_c);
            let scale_bc = second(
                &scale.h_bra_bra,
                &scale.h_bra_ket,
                &scale.h_ket_ket,
                cb,
                cc,
                axis_b,
                axis_c,
            );
            let ov = pair.moments[0];
            let ov_b = first(&pair.d_bra[0], &pair.d_ket[0], cb, axis_b);
            let ov_c = first(&pair.d_bra[0], &pair.d_ket[0], cc, axis_c);
            let ov_bc = second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                cb,
                cc,
                axis_b,
                axis_c,
            );
            out[(mu, nu)] =
                self_avg * (scale_bc * ov + scale_b * ov_c + scale_c * ov_b + scale.value * ov_bc);
        }
    }
    Ok(out)
}

/// H0 geometric scale `hscale·shell_poly` to **third** order (the `h0_scale_second` analog, used by
/// the CN-H0 third derivative's `d_edcn` build).
fn h0_scale_third(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    shell_mu: usize,
    shell_nu: usize,
    basis: &BasisSet,
) -> Result<H0Third> {
    let si = &basis.shells[shell_mu];
    let sj = &basis.shells[shell_nu];
    let base = hscale(si, sj, params)?;
    let poly = shell_poly_third(
        system,
        si.atom_index,
        sj.atom_index,
        si.z,
        sj.z,
        si.poly_raw,
        sj.poly_raw,
    )?;
    Ok(H0Third {
        value: base * poly.value,
        d_bra: poly.d_bra * base,
        d_ket: poly.d_ket * base,
        h_bra_bra: scale_3x3(poly.h_bra_bra, base),
        h_bra_ket: scale_3x3(poly.h_bra_ket, base),
        h_ket_ket: scale_3x3(poly.h_ket_ket, base),
        t_bra_bra_bra: scale_ten3(poly.t_bra_bra_bra, base),
        t_bra_bra_ket: scale_ten3(poly.t_bra_bra_ket, base),
        t_bra_ket_ket: scale_ten3(poly.t_bra_ket_ket, base),
        t_ket_ket_ket: scale_ten3(poly.t_ket_ket_ket, base),
    })
}

/// `∂E/∂CN_A` and its first/second/third nuclear derivatives, as a dense forward-AD "jet" over the
/// `ndof` Cartesian coordinates (value + gradient `[ndof]` + Hessian `[ndof²]` + third `[ndof³]`).
/// `d_edcn_A = Σ_shells∈A dsedcn_sh · Σ_{μ∈sh,ν} P_{μν}·scale_{μν}·S_{μν}` at **frozen density**;
/// the on-site diagonal block is `R`-independent (overlap ≡ 1), so it adds to the value only. The
/// off-site geometric derivatives come from the `scale·overlap` Leibniz (`h0_scale_third` ×
/// `contracted_pair_with_third_derivatives`), the same slot machinery as the Pulay third
/// derivative. The CN-H0 frozen third derivative contracts these (via the product Leibniz) with the
/// coordination-number jet.
pub(crate) struct DedcnJet {
    pub(crate) value: f64,
    pub(crate) grad: Vec<f64>,
    pub(crate) hess: Vec<f64>,
    pub(crate) third: Vec<f64>,
}

pub(crate) fn cn_h0_dedcn_jets(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<DedcnJet>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut jets: Vec<DedcnJet> = (0..nat)
        .map(|_| DedcnJet {
            value: 0.0,
            grad: vec![0.0; ndof],
            hess: vec![0.0; ndof * ndof],
            third: vec![0.0; ndof * ndof * ndof],
        })
        .collect();

    // On-site diagonal block (R-independent): value only.
    for (ish, shell) in basis.shells.iter().enumerate() {
        let dsedcn = -shell.kcn_raw.unwrap_or(0.0);
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            jets[shell.atom_index].value += dsedcn * electronic.density[(iao, iao)];
        }
        let _ = ish;
    }

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let shell_mu = &basis.shells[shell_mu_index];
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let shell_nu = &basis.shells[shell_nu_index];
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let scale = h0_scale_third(system, params, shell_mu_index, shell_nu_index, basis)?;
            let dsedcn_mu = -shell_mu.kcn_raw.unwrap_or(0.0);
            let dsedcn_nu = -shell_nu.kcn_raw.unwrap_or(0.0);

            // value: Σ dsedcn · p · (scale·overlap).
            let val0 = p * scale.value * overlap;
            jets[atom_mu].value += dsedcn_mu * val0;
            jets[atom_nu].value += dsedcn_nu * val0;

            let atom_of = |c: Center| match c {
                Center::Bra => atom_mu,
                Center::Ket => atom_nu,
            };
            let s1 = |c: Center, ax: usize| first(&pair.d_bra[0], &pair.d_ket[0], c, ax);
            let s2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(
                    &pair.h_bra_bra[0],
                    &pair.h_bra_ket[0],
                    &pair.h_ket_ket[0],
                    ca,
                    cb,
                    axa,
                    axb,
                )
            };
            let h1 = |c: Center, ax: usize| first_vec(scale.d_bra, scale.d_ket, c, ax);
            let h2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(
                    &scale.h_bra_bra,
                    &scale.h_bra_ket,
                    &scale.h_ket_ket,
                    ca,
                    cb,
                    axa,
                    axb,
                )
            };
            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let dof = |s: usize| 3 * atom_of(slot_center(s)) + s % 3;

            // (scale·overlap) derivatives via Leibniz; scatter to both atoms' jets.
            for i in 0..6 {
                let (ci, axi) = (slot_center(i), i % 3);
                let v1 = scale.value * s1(ci, axi) + h1(ci, axi) * overlap;
                jets[atom_mu].grad[dof(i)] += dsedcn_mu * p * v1;
                jets[atom_nu].grad[dof(i)] += dsedcn_nu * p * v1;
                for j in 0..6 {
                    let (cj, axj) = (slot_center(j), j % 3);
                    let v2 = scale.value * s2(ci, cj, axi, axj)
                        + h1(ci, axi) * s1(cj, axj)
                        + h1(cj, axj) * s1(ci, axi)
                        + h2(ci, cj, axi, axj) * overlap;
                    let hidx = dof(i) * ndof + dof(j);
                    jets[atom_mu].hess[hidx] += dsedcn_mu * p * v2;
                    jets[atom_nu].hess[hidx] += dsedcn_nu * p * v2;
                    for k in 0..6 {
                        let (ck, axk) = (slot_center(k), k % 3);
                        let s_abc = third_select(
                            &pair.t_bra_bra_bra[0],
                            &pair.t_bra_bra_ket[0],
                            &pair.t_bra_ket_ket[0],
                            &pair.t_ket_ket_ket[0],
                            [ci, cj, ck],
                            [axi, axj, axk],
                        );
                        let h_abc = third_select(
                            &scale.t_bra_bra_bra,
                            &scale.t_bra_bra_ket,
                            &scale.t_bra_ket_ket,
                            &scale.t_ket_ket_ket,
                            [ci, cj, ck],
                            [axi, axj, axk],
                        );
                        let v3 = scale.value * s_abc
                            + h_abc * overlap
                            + h2(ci, cj, axi, axj) * s1(ck, axk)
                            + h2(ci, ck, axi, axk) * s1(cj, axj)
                            + h2(cj, ck, axj, axk) * s1(ci, axi)
                            + h1(ci, axi) * s2(cj, ck, axj, axk)
                            + h1(cj, axj) * s2(ci, ck, axi, axk)
                            + h1(ck, axk) * s2(ci, cj, axi, axj);
                        let tidx = (dof(i) * ndof + dof(j)) * ndof + dof(k);
                        jets[atom_mu].third[tidx] += dsedcn_mu * p * v3;
                        jets[atom_nu].third[tidx] += dsedcn_nu * p * v3;
                    }
                }
            }
        }
    }
    Ok(jets)
}

/// Coordination-number jet: `CN_A` and its 1st/2nd/3rd nuclear derivatives (value + `[ndof]` +
/// `[ndof²]` + `[ndof³]`), built from the smooth counting function's central radial blocks scattered
/// over each `(i,j)` pair's `i`/`j` slots (`σ_i = +1`, `σ_j = −1`). The CN-H0 third derivative
/// contracts this with [`cn_h0_dedcn_jets`] via the product Leibniz.
pub(crate) fn cn_h0_cn_jets(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
) -> Result<Vec<DedcnJet>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let kcn = CoordinationOptions::default().kcn;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let mut jets: Vec<DedcnJet> = (0..nat)
        .map(|_| DedcnJet {
            value: 0.0,
            grad: vec![0.0; ndof],
            hess: vec![0.0; ndof * ndof],
            third: vec![0.0; ndof * ndof * ndof],
        })
        .collect();
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let rvec = pair.r_ij;
        let r = rvec.norm();
        if r <= 1.0e-12 {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let d = coordination_value_derivatives(kcn, r, rc);
        let counting = 1.0 / (1.0 + (-kcn * (rc / r - 1.0)).exp());
        let u = (rvec / r).to_array();
        let g = d.second / r - d.first / (r * r);
        let coeff_uuu = d.third - 3.0 * g;
        // 6 slots: 0..3 = atom i axes (σ=+1), 3..6 = atom j axes (σ=−1).
        let dof = |s: usize| {
            if s < 3 {
                3 * pair.i + s
            } else {
                3 * pair.j + (s - 3)
            }
        };
        let sigma = |s: usize| if s < 3 { 1.0_f64 } else { -1.0 };
        for jet_atom in [pair.i, pair.j] {
            let jet = &mut jets[jet_atom];
            jet.value += counting;
            for s in 0..6 {
                let a = s % 3;
                jet.grad[dof(s)] += sigma(s) * d.first * u[a];
                for t in 0..6 {
                    let b = t % 3;
                    let dab = if a == b { 1.0 } else { 0.0 };
                    let hrel = d.second * u[a] * u[b] + (d.first / r) * (dab - u[a] * u[b]);
                    jet.hess[dof(s) * ndof + dof(t)] += sigma(s) * sigma(t) * hrel;
                    for q in 0..6 {
                        let c = q % 3;
                        let kron = if a == b { u[c] } else { 0.0 }
                            + if a == c { u[b] } else { 0.0 }
                            + if b == c { u[a] } else { 0.0 };
                        let trel = coeff_uuu * u[a] * u[b] * u[c] + g * kron;
                        let idx = (dof(s) * ndof + dof(t)) * ndof + dof(q);
                        jet.third[idx] += sigma(s) * sigma(t) * sigma(q) * trel;
                    }
                }
            }
        }
    }
    Ok(jets)
}

/// Grad-only coordination-number derivatives: `grad[atom][dof] = ∂CN_atom/∂R_dof`. The lean (no hess/third,
/// so `O(nat·ndof)` not `O(nat·ndof³)`) slice of [`cn_h0_cn_jets`]'s gradient, for the Pulay CN-response
/// 3rd-derivative term [`fixed_density_pulay_cn_h0_response`], which needs only `∂CN/∂R`.
pub(crate) fn cn_gradient_matrix(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
) -> Result<Vec<Vec<f64>>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let kcn = CoordinationOptions::default().kcn;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let mut grad = vec![vec![0.0; ndof]; nat];
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let rvec = pair.r_ij;
        let r = rvec.norm();
        if r <= 1.0e-12 {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let d = coordination_value_derivatives(kcn, r, rc);
        let u = (rvec / r).to_array();
        let dof = |s: usize| {
            if s < 3 {
                3 * pair.i + s
            } else {
                3 * pair.j + (s - 3)
            }
        };
        let sigma = |s: usize| if s < 3 { 1.0_f64 } else { -1.0 };
        for jet_atom in [pair.i, pair.j] {
            for s in 0..6 {
                grad[jet_atom][dof(s)] += sigma(s) * d.first * u[s % 3];
            }
        }
    }
    Ok(grad)
}

/// **CN-H0 frozen third derivative** (slab `c` = `∂H_cn-h0/∂R_c`). The frozen-density CN-coupling
/// energy is the product `E = Σ_A CN_A(R)·d_edcn_A(R)`; this assembles its analytic third derivative
/// from the two scalar jets via the 8-term product Leibniz, summed over atoms. FD-validates against
/// the central FD of the analytic CN-H0 Hessian (`fixed_density_cn_h0_hessian` +
/// `fixed_density_cn_h0_pulay_cross_hessian`). Frozen ⇒ no electronic response; the last frozen
/// `L_abc` block of the non-PBC analytic third derivative.
pub fn fixed_density_cn_h0_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let de = cn_h0_dedcn_jets(system, params, electronic)?;
    let cn = cn_h0_cn_jets(system, coordination_cutoff)?;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    // The CN-H0 Hessian BLOCK is `H_bc = CN_bc·de + CN_b·de_c + de_b·CN_c` (it deliberately omits
    // `CN·de_bc`, which the band/Pulay block carries with the converged self-energy held fixed). So
    // slab `a` is `∂H_bc/∂R_a` of exactly that 3-term block — not the symmetric full product.
    for a in 0..ndof {
        for b in 0..ndof {
            for c in 0..ndof {
                let mut t = 0.0;
                for atom in 0..nat {
                    let (n, d) = (&cn[atom], &de[atom]);
                    t += d.value * n.third[(a * ndof + b) * ndof + c]
                        + n.hess[b * ndof + c] * d.grad[a]
                        + n.hess[a * ndof + b] * d.grad[c]
                        + n.grad[b] * d.hess[a * ndof + c]
                        + d.hess[a * ndof + b] * n.grad[c]
                        + d.grad[b] * n.hess[a * ndof + c];
                }
                tensor[a][(b, c)] = t;
            }
        }
    }
    Ok(tensor)
}

/// Geometric third derivative (fixed density) of [`fixed_density_scalar_overlap_hessian`], the
/// SCC-scalar-potential × overlap-derivative coupling block `H_ab = −Σ_pairs p · ds_a · dscalar_b`
/// (a = overlap-derivative direction, b = scalar-potential-derivative DOF). Differentiating once more at
/// fixed density `p`:
///   `T_abc = −Σ_pairs p · ( ∂_c ds_a · dscalar_b + ds_a · ∂_c dscalar_b )`,
/// where `∂_c ds_a` is the overlap second derivative (geometric; `c ∈ {bra,ket} centers only) and
/// `∂_c dscalar_b` is the scalar-potential second derivative (`shell_scalar_potential_second_derivatives`).
/// Returned ordered slabs `tensor[c][(a,b)]`.
pub fn fixed_density_scalar_overlap_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let dscalar1 =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let dscalar2 = shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let hbb = &pair.h_bra_bra[0];
            let hbk = &pair.h_bra_ket[0];
            let hkk = &pair.h_ket_ket[0];
            let d_bra = pair.d_bra[0];
            let d_ket = pair.d_ket[0];
            for row_center in [Center::Bra, Center::Ket] {
                let row_atom = match row_center {
                    Center::Bra => atom_mu,
                    Center::Ket => atom_nu,
                };
                for row_axis in 0..3 {
                    let ds_row = first(&d_bra, &d_ket, row_center, row_axis);
                    let a = 3 * row_atom + row_axis;
                    for b in 0..ndof {
                        let dscalar_b = dscalar1[(shell_mu, b)] + dscalar1[(shell_nu, b)];
                        for c in 0..ndof {
                            let dscalar_bc =
                                dscalar2[shell_mu][(b, c)] + dscalar2[shell_nu][(b, c)];
                            let mut t = ds_row * dscalar_bc;
                            let atom_c = c / 3;
                            let c_axis = c % 3;
                            let c_center = if atom_c == atom_mu {
                                Some(Center::Bra)
                            } else if atom_c == atom_nu {
                                Some(Center::Ket)
                            } else {
                                None
                            };
                            if let Some(cc) = c_center {
                                let dds = second(hbb, hbk, hkk, row_center, cc, row_axis, c_axis);
                                t += dds * dscalar_b;
                            }
                            tensor[c][(a, b)] -= p * t;
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

fn shell_poly_second(
    system: &PeriodicSystem,
    ai: usize,
    aj: usize,
    zi: u8,
    zj: u8,
    pi: Option<f64>,
    pj: Option<f64>,
) -> Result<H0Second> {
    let ipoly = pi.unwrap_or(0.0);
    let jpoly = pj.unwrap_or(0.0);
    if ipoly == 0.0 && jpoly == 0.0 {
        return Ok(H0Second::constant(1.0));
    }
    let dr = system.atoms[aj].position - system.atoms[ai].position;
    let r = dr.norm();
    if r <= 1.0e-12 {
        return Ok(H0Second::constant(1.0));
    }
    let rad_sum = atomic_radius_bohr(zi)? + atomic_radius_bohr(zj)?;
    let scaled = (r / rad_sum).sqrt();
    let fi = 1.0 + ipoly * scaled;
    let fj = 1.0 + jpoly * scaled;
    let value = fi * fj;
    let dscaled_dr = 0.5 / (rad_sum * scaled.max(1.0e-16));
    let d2scaled_dr2 = -0.25 / (rad_sum * rad_sum * scaled.max(1.0e-16).powi(3));
    let linear = ipoly * fj + jpoly * fi;
    let dpoly_dr = linear * dscaled_dr;
    let d2poly_dr2 = 2.0 * ipoly * jpoly * dscaled_dr * dscaled_dr + linear * d2scaled_dr2;
    let u = dr / r;
    let u_arr = u.to_array();
    let mut radial = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let uu = u_arr[a] * u_arr[b];
            let delta = if a == b { 1.0 } else { 0.0 };
            radial[a][b] = d2poly_dr2 * uu + (dpoly_dr / r) * (delta - uu);
        }
    }
    Ok(H0Second {
        value,
        d_bra: u * (-dpoly_dr),
        d_ket: u * dpoly_dr,
        h_bra_bra: radial,
        h_bra_ket: scale_3x3(radial, -1.0),
        h_ket_ket: radial,
    })
}

impl H0Second {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_bra: Vec3::zero(),
            d_ket: Vec3::zero(),
            h_bra_bra: [[0.0; 3]; 3],
            h_bra_ket: [[0.0; 3]; 3],
            h_ket_ket: [[0.0; 3]; 3],
        }
    }
}

fn first(bra: &Vec3, ket: &Vec3, center: Center, axis: usize) -> f64 {
    first_vec(*bra, *ket, center, axis)
}

fn first_vec(bra: Vec3, ket: Vec3, center: Center, axis: usize) -> f64 {
    let value = match center {
        Center::Bra => bra,
        Center::Ket => ket,
    };
    value.to_array()[axis]
}

fn second(
    h_bra_bra: &[[f64; 3]; 3],
    h_bra_ket: &[[f64; 3]; 3],
    h_ket_ket: &[[f64; 3]; 3],
    row_center: Center,
    col_center: Center,
    row_axis: usize,
    col_axis: usize,
) -> f64 {
    match (row_center, col_center) {
        (Center::Bra, Center::Bra) => h_bra_bra[row_axis][col_axis],
        (Center::Bra, Center::Ket) => h_bra_ket[row_axis][col_axis],
        (Center::Ket, Center::Bra) => h_bra_ket[col_axis][row_axis],
        (Center::Ket, Center::Ket) => h_ket_ket[row_axis][col_axis],
    }
}

fn scale_3x3(mut matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

#[derive(Clone, Copy, Debug)]
struct KernelDerivatives {
    value: f64,
    gradient_prefactor: f64,
    gradient_prefactor_derivative: f64,
    /// Third radial derivative `f'''` of the kernel `f(r)`, for the analytic third nuclear
    /// derivative. With `p = f'/r = gradient_prefactor`, `L = log_derivative`, `p' = p·L`,
    /// `p'' = p(L² + L')`, the chain gives `f''' = 2 p' + r p''`.
    radial_third_derivative: f64,
}

fn effective_kernel_derivatives(r: f64, gamma: f64) -> KernelDerivatives {
    let g = GFN1_COULOMB_EXPONENT;
    let sum = r.powf(g) + gamma.powf(-g);
    let value = sum.powf(-1.0 / g);
    let prefactor = -r.powf(g - 2.0) * sum.powf(-1.0 / g - 1.0);
    let log_derivative = (g - 2.0) / r + (-1.0 / g - 1.0) * g * r.powf(g - 1.0) / sum;
    // L' = d(log_derivative)/dr, with the second-term coefficient `a = (-1/g - 1)g = -(1+g)`.
    let a = -(1.0 + g);
    let log_derivative_prime = -(g - 2.0) / (r * r) + a * (g - 1.0) * r.powf(g - 2.0) / sum
        - a * g * r.powf(2.0 * g - 2.0) / (sum * sum);
    let p = prefactor;
    let p_prime = p * log_derivative;
    let p_double = p * (log_derivative * log_derivative + log_derivative_prime);
    let radial_third_derivative = 2.0 * p_prime + r * p_double;
    KernelDerivatives {
        value,
        gradient_prefactor: prefactor,
        gradient_prefactor_derivative: p_prime,
        radial_third_derivative,
    }
}

fn add_central_hessian_block(
    hessian: &mut Matrix,
    i: usize,
    j: usize,
    rvec: Vec3,
    gradient_prefactor: f64,
    gradient_prefactor_derivative: f64,
) {
    let r = rvec.norm();
    if r <= 1.0e-12 {
        return;
    }
    let u = (rvec / r).to_array();
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            let value =
                gradient_prefactor * delta + r * gradient_prefactor_derivative * u[a] * u[b];
            let ia = 3 * i + a;
            let ib = 3 * i + b;
            let ja = 3 * j + a;
            let jb = 3 * j + b;
            hessian[(ia, ib)] += value;
            hessian[(ja, jb)] += value;
            hessian[(ia, jb)] -= value;
            hessian[(ja, ib)] -= value;
        }
    }
}

fn ensure_non_pbc(system: &PeriodicSystem) -> Result<()> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "analytic Hessian currently supports non-PBC only".to_string(),
        ));
    }
    Ok(())
}

fn add_matrix(lhs: &mut Matrix, rhs: &Matrix) -> Result<()> {
    if lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() {
        return Err(Gfn1Error::InvalidInput(format!(
            "cannot add Hessian blocks with shapes {}x{} and {}x{}",
            lhs.rows(),
            lhs.cols(),
            rhs.rows(),
            rhs.cols()
        )));
    }
    for (l, r) in lhs.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
        *l += *r;
    }
    Ok(())
}

/// Replace a square matrix with `½(H + Hᵀ)`, enforcing exact symmetry on the
/// final Hessian.
fn symmetrize_in_place(hessian: &mut Matrix) {
    let n = hessian.rows();
    debug_assert_eq!(n, hessian.cols(), "Hessian must be square to symmetrize");
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = avg;
            hessian[(j, i)] = avg;
        }
    }
}

type Ten3 = [[[f64; 3]; 3]; 3];

/// `H0` (or bare `poly`) value plus its bra/ket Cartesian derivative tensors up to **third**
/// order. The ket indices of each `t_*` come last (`t_bra_bra_ket[a][b][c] = ∂_{A_a}∂_{A_b}∂_{B_c}`).
#[derive(Clone, Debug)]
struct H0Third {
    value: f64,
    d_bra: Vec3,
    d_ket: Vec3,
    h_bra_bra: [[f64; 3]; 3],
    h_bra_ket: [[f64; 3]; 3],
    h_ket_ket: [[f64; 3]; 3],
    t_bra_bra_bra: Ten3,
    t_bra_bra_ket: Ten3,
    t_bra_ket_ket: Ten3,
    t_ket_ket_ket: Ten3,
}

impl H0Third {
    fn constant(value: f64) -> Self {
        Self {
            value,
            d_bra: Vec3::zero(),
            d_ket: Vec3::zero(),
            h_bra_bra: [[0.0; 3]; 3],
            h_bra_ket: [[0.0; 3]; 3],
            h_ket_ket: [[0.0; 3]; 3],
            t_bra_bra_bra: [[[0.0; 3]; 3]; 3],
            t_bra_bra_ket: [[[0.0; 3]; 3]; 3],
            t_bra_ket_ket: [[[0.0; 3]; 3]; 3],
            t_ket_ket_ket: [[[0.0; 3]; 3]; 3],
        }
    }

    fn scale(&self, s: f64) -> Self {
        let mut out = self.clone();
        out.value *= s;
        out.d_bra = out.d_bra * s;
        out.d_ket = out.d_ket * s;
        for a in 0..3 {
            for b in 0..3 {
                out.h_bra_bra[a][b] *= s;
                out.h_bra_ket[a][b] *= s;
                out.h_ket_ket[a][b] *= s;
                for c in 0..3 {
                    out.t_bra_bra_bra[a][b][c] *= s;
                    out.t_bra_bra_ket[a][b][c] *= s;
                    out.t_bra_ket_ket[a][b][c] *= s;
                    out.t_ket_ket_ket[a][b][c] *= s;
                }
            }
        }
        out
    }
}

/// Third-order bra/ket derivative tensors of the GFN1 distance polynomial `poly(r) = fi·fj`,
/// `fi = 1 + p_i√(r/R_AB)`. With the relative vector `dr = R_ket − R_bra` (so `σ_bra = −1`,
/// `σ_ket = +1` w.r.t. `∂/∂dr`), the radial third derivative `T_rel` is signed per center.
fn shell_poly_third(
    system: &PeriodicSystem,
    ai: usize,
    aj: usize,
    zi: u8,
    zj: u8,
    pi: Option<f64>,
    pj: Option<f64>,
) -> Result<H0Third> {
    let ipoly = pi.unwrap_or(0.0);
    let jpoly = pj.unwrap_or(0.0);
    if ipoly == 0.0 && jpoly == 0.0 {
        return Ok(H0Third::constant(1.0));
    }
    let dr = system.atoms[aj].position - system.atoms[ai].position;
    let r = dr.norm();
    if r <= 1.0e-12 {
        return Ok(H0Third::constant(1.0));
    }
    let rad_sum = atomic_radius_bohr(zi)? + atomic_radius_bohr(zj)?;
    let s = (r / rad_sum).sqrt();
    let s_safe = s.max(1.0e-16);
    let fi = 1.0 + ipoly * s;
    let fj = 1.0 + jpoly * s;
    let value = fi * fj;
    let ds = 0.5 / (rad_sum * s_safe);
    let d2s = -0.25 / (rad_sum * rad_sum * s_safe.powi(3));
    let d3s = 3.0 / (8.0 * rad_sum.powi(3) * s_safe.powi(5));
    let linear = ipoly * fj + jpoly * fi;
    let phi_p = linear * ds;
    let phi_pp = 2.0 * ipoly * jpoly * ds * ds + linear * d2s;
    let phi_ppp = 6.0 * ipoly * jpoly * ds * d2s + linear * d3s;

    let u = (dr / r).to_array();
    let g = phi_pp / r - phi_p / (r * r);
    let coeff_uuu3 = phi_ppp - 3.0 * g;
    let mut t_rel2 = [[0.0_f64; 3]; 3];
    let mut t_rel3 = [[[0.0_f64; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let dab = if a == b { 1.0 } else { 0.0 };
            t_rel2[a][b] = phi_pp * u[a] * u[b] + (phi_p / r) * (dab - u[a] * u[b]);
            for c in 0..3 {
                let kron = if a == b { u[c] } else { 0.0 }
                    + if a == c { u[b] } else { 0.0 }
                    + if b == c { u[a] } else { 0.0 };
                t_rel3[a][b][c] = coeff_uuu3 * u[a] * u[b] * u[c] + g * kron;
            }
        }
    }
    // σ_bra = −1, σ_ket = +1 ⇒ even #bra → +T_rel, odd → −T_rel.
    let neg2 = scale_3x3(t_rel2, -1.0);
    let neg3 = scale_ten3(t_rel3, -1.0);
    Ok(H0Third {
        value,
        d_bra: Vec3::new(-phi_p * u[0], -phi_p * u[1], -phi_p * u[2]),
        d_ket: Vec3::new(phi_p * u[0], phi_p * u[1], phi_p * u[2]),
        h_bra_bra: t_rel2,
        h_bra_ket: neg2,
        h_ket_ket: t_rel2,
        t_bra_bra_bra: neg3,
        t_bra_bra_ket: t_rel3,
        t_bra_ket_ket: neg3,
        t_ket_ket_ket: t_rel3,
    })
}

fn scale_ten3(mut t: Ten3, s: f64) -> Ten3 {
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                t[a][b][c] *= s;
            }
        }
    }
    t
}

/// `base · poly_third`, the third-order analogue of [`h0_prefactor_second`]. `base`
/// (`½(self_i+self_j)·hscale`, with the **frozen** CN-dependent self-energies) is a geometry
/// constant in this fixed-density block; only the polynomial varies with geometry.
fn h0_prefactor_third(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    shell_mu: usize,
    shell_nu: usize,
) -> Result<H0Third> {
    let si = &electronic.basis.shells[shell_mu];
    let sj = &electronic.basis.shells[shell_nu];
    let self_i = shell_self_energy(si, electronic.coordination_numbers[si.atom_index]);
    let self_j = shell_self_energy(sj, electronic.coordination_numbers[sj.atom_index]);
    let base = 0.5 * (self_i + self_j) * hscale(si, sj, params)?;
    let poly = shell_poly_third(
        system,
        si.atom_index,
        sj.atom_index,
        si.z,
        sj.z,
        si.poly_raw,
        sj.poly_raw,
    )?;
    Ok(poly.scale(base))
}

/// The distinct ordered permutations of the (possibly repeated) triple `(i,j,k)` — used to
/// scatter a symmetry-unique third-derivative value to every tensor position it occupies.
fn distinct_perms(i: usize, j: usize, k: usize) -> Vec<(usize, usize, usize)> {
    let all = [
        (i, j, k),
        (i, k, j),
        (j, i, k),
        (j, k, i),
        (k, i, j),
        (k, j, i),
    ];
    let mut out: Vec<(usize, usize, usize)> = Vec::with_capacity(6);
    for p in all {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Select the third overlap derivative from the four center-pattern tensors (ket indices
/// last), permuting axes so the bras come first. Mixed partials commute, so only the
/// per-center axis assignment matters.
#[allow(clippy::too_many_arguments)]
fn third_select(
    t_bbb: &Ten3,
    t_bbk: &Ten3,
    t_bkk: &Ten3,
    t_kkk: &Ten3,
    centers: [Center; 3],
    axes: [usize; 3],
) -> f64 {
    let mut bra = [0usize; 3];
    let mut ket = [0usize; 3];
    let mut nb = 0;
    let mut nk = 0;
    for k in 0..3 {
        match centers[k] {
            Center::Bra => {
                bra[nb] = axes[k];
                nb += 1;
            }
            Center::Ket => {
                ket[nk] = axes[k];
                nk += 1;
            }
        }
    }
    match nb {
        3 => t_bbb[bra[0]][bra[1]][bra[2]],
        2 => t_bbk[bra[0]][bra[1]][ket[0]],
        1 => t_bkk[bra[0]][ket[0]][ket[1]],
        _ => t_kkk[ket[0]][ket[1]][ket[2]],
    }
}

/// Analytic third Cartesian derivative of the **fixed-density** band/H0 + SCC-overlap + Pulay
/// energy (frozen `P`, `W`, shell potential, CN), returned as `ndof` slabs. This is the
/// `L_abc` frozen block for the band/overlap/H0 sector of the 2n+1 driver: with the density
/// frozen it carries no response, so it FD-validates in isolation against
/// [`fixed_density_pulay_hessian`].
pub fn fixed_density_pulay_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            let w = electronic.energy_weighted_density[(mu, nu)];
            if p.abs().max(w.abs()) <= 1.0e-18 {
                continue;
            }
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_third(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
            let overlap_coeff = p * (2.0 * h0.value - scalar_shift) - 2.0 * w;
            let two_p = 2.0 * p;

            let atom_of = |c: Center| match c {
                Center::Bra => atom_mu,
                Center::Ket => atom_nu,
            };
            // Overlap derivative accessors (moment 0).
            let s1 = |c: Center, ax: usize| first(&pair.d_bra[0], &pair.d_ket[0], c, ax);
            let s2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(
                    &pair.h_bra_bra[0],
                    &pair.h_bra_ket[0],
                    &pair.h_ket_ket[0],
                    ca,
                    cb,
                    axa,
                    axb,
                )
            };
            // H0 (poly·base) derivative accessors.
            let h1 = |c: Center, ax: usize| first_vec(h0.d_bra, h0.d_ket, c, ax);
            let h2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(
                    &h0.h_bra_bra,
                    &h0.h_bra_ket,
                    &h0.h_ket_ket,
                    ca,
                    cb,
                    axa,
                    axb,
                )
            };

            // The third derivative is fully symmetric in its three nuclear indices, so the
            // Leibniz value depends only on the *unordered* triple of (center,axis) "slots"
            // (s ∈ 0..6: bra/ket × axis). Compute each unique unordered triple once and
            // scatter it to its distinct ordered permutations — 56 evaluations instead of
            // 6³ = 216 per AO pair.
            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let value_of = |i: usize, j: usize, k: usize| -> f64 {
                let (ca, axa) = (slot_center(i), i % 3);
                let (cb, axb) = (slot_center(j), j % 3);
                let (cc, axc) = (slot_center(k), k % 3);
                let s_abc = third_select(
                    &pair.t_bra_bra_bra[0],
                    &pair.t_bra_bra_ket[0],
                    &pair.t_bra_ket_ket[0],
                    &pair.t_ket_ket_ket[0],
                    [ca, cb, cc],
                    [axa, axb, axc],
                );
                let h_abc = third_select(
                    &h0.t_bra_bra_bra,
                    &h0.t_bra_bra_ket,
                    &h0.t_bra_ket_ket,
                    &h0.t_ket_ket_ket,
                    [ca, cb, cc],
                    [axa, axb, axc],
                );
                let (s_a, s_b, s_c) = (s1(ca, axa), s1(cb, axb), s1(cc, axc));
                let s_ab = s2(ca, cb, axa, axb);
                let s_ac = s2(ca, cc, axa, axc);
                let s_bc = s2(cb, cc, axb, axc);
                let (h_a, h_b, h_c) = (h1(ca, axa), h1(cb, axb), h1(cc, axc));
                let h_ab = h2(ca, cb, axa, axb);
                let h_ac = h2(ca, cc, axa, axc);
                let h_bc = h2(cb, cc, axb, axc);
                overlap_coeff * s_abc
                    + two_p
                        * (overlap * h_abc
                            + (h_ab * s_c + h_ac * s_b + h_bc * s_a)
                            + (h_a * s_bc + h_b * s_ac + h_c * s_ab))
            };
            let dof = |s: usize| 3 * atom_of(slot_center(s)) + s % 3;
            for i in 0..6 {
                for j in i..6 {
                    for k in j..6 {
                        let v = value_of(i, j, k);
                        if v == 0.0 {
                            continue;
                        }
                        for &(p, q, r) in &distinct_perms(i, j, k) {
                            tensor[dof(r)][(dof(p), dof(q))] += v;
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

/// PULAY density-path GEOMETRY×RESPONSE cross term — the missing 3rd-derivative piece, ORDERED form.
///
/// The reconverged density-path residual `miss[c]` = the geometry slab-`c` derivative of the pulay
/// Hessian's density-linear kernel, with the coefficient carrying the first-order density response. The
/// slab index `c` is the FD displacement direction — it is NOT symmetric with the Hessian row/col `(a,b)`.
/// So this is the ORDERED third derivative (c fixed as the displacement slot), NOT the fully-symmetric
/// 3-index contraction (which overshoots ~150× because it averages over all 3 orderings).
///
/// Per AO pair, with response coefficient `C^(c) = P^(c)(2h0−V) − P·V^(c) − 2W^(c)` and `two_p^(c)=2P^(c)`:
///   - C:S_ab channel: `∂_c[ C^(c)·S_ab ] = C^(c)·S_{ab,c} + (∂_c^geom C^(c))·S_ab`,
///        with `∂_c^geom C^(c) = P^(c)·2·(∂_c h0_poly) − P^(c)·(∂_c V)`  (γ/poly geometry, fixed charges),
///   - h0 channel: `∂_c[ 2P^(c)·(h0_a·S_b + h0_b·S_a + S·h0_ab) ]`  (all geometry derivatives ordered).
/// The Hessian pair `(a,b)` are the row/col AO-pair center slots; `c` is the third, distinct slot. Matches
/// the AO-pair convention / sign of [`fixed_density_pulay_hessian`]. First-order-response only.
///
/// `response_density`=P^(c), `response_ew_density`=W^(c), `response_scalar_potential`=V^(c). Returned
/// ordered slabs `tensor[c][(a,b)]`. `only_channel`: Some(0)=C:S_ab only, Some(1)=h0 only, None=both.
#[cfg(test)]
pub(crate) fn pulay_density_path_geom_cross_ordered(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    response_density: &Matrix,
    response_ew_density: &Matrix,
    response_scalar_potential: &[f64],
    only_channel: Option<usize>,
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    let ao_v = ao_scalar_potentials(basis, &electronic.shell_scc_potential);
    let ao_vc = ao_scalar_potentials(basis, response_scalar_potential);
    // Geometric derivative of the shell scalar potential at FIXED charge: vgeo_shell[dof][shell] = ∂V_shell/∂R_dof.
    let vgeo_shell = crate::cphf::shell_scalar_potential_derivatives(
        system,
        basis,
        params,
        &electronic.shell_charges,
    )?;

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let shell_nu_index = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            let pc = response_density[(mu, nu)];
            let wc = response_ew_density[(mu, nu)];
            if p.abs().max(pc.abs()).max(wc.abs()) <= 1.0e-18 {
                continue;
            }
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_third(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let scalar_shift = ao_v[mu] + ao_v[nu];
            let scalar_shift_c = ao_vc[mu] + ao_vc[nu];
            // Response coefficient (density-linear part of overlap_coeff): C^(c) = P^(c)(2h0−V) − P·V^(c) − 2W^(c).
            let overlap_coeff = pc * (2.0 * h0.value - scalar_shift) - p * scalar_shift_c - 2.0 * wc;
            let two_p = 2.0 * pc;

            let atom_of = |c: Center| match c {
                Center::Bra => atom_mu,
                Center::Ket => atom_nu,
            };
            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let dof = |s: usize| 3 * atom_of(slot_center(s)) + s % 3;
            let s1 = |c: Center, ax: usize| first(&pair.d_bra[0], &pair.d_ket[0], c, ax);
            let s2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(&pair.h_bra_bra[0], &pair.h_bra_ket[0], &pair.h_ket_ket[0], ca, cb, axa, axb)
            };
            let s3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &pair.t_bra_bra_bra[0], &pair.t_bra_bra_ket[0], &pair.t_bra_ket_ket[0],
                    &pair.t_ket_ket_ket[0], [ca, cb, cc], [axa, axb, axc],
                )
            };
            let h1 = |c: Center, ax: usize| first_vec(h0.d_bra, h0.d_ket, c, ax);
            let h2 = |ca: Center, cb: Center, axa: usize, axb: usize| {
                second(&h0.h_bra_bra, &h0.h_bra_ket, &h0.h_ket_ket, ca, cb, axa, axb)
            };
            let h3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &h0.t_bra_bra_bra, &h0.t_bra_bra_ket, &h0.t_bra_ket_ket,
                    &h0.t_ket_ket_ket, [ca, cb, cc], [axa, axb, axc],
                )
            };
            // ORDERED: c is the slab (displacement) index; (a,b) are the Hessian row/col. Loop all 3 slots
            // independently (6×6×6), each mapped to its own dof; scatter to tensor[c][(a,b)] with c=slab.
            for ia in 0..6 {
                let (ca, axa) = (slot_center(ia), ia % 3);
                for ib in 0..6 {
                    let (cb, axb) = (slot_center(ib), ib % 3);
                    for ic in 0..6 {
                        let (cc, axc) = (slot_center(ic), ic % 3);
                        let mut v = 0.0;
                        // --- C:S_ab channel: ∂_c[C^(c)·S_ab] = C^(c)·S_{ab,c} + (∂_c C^(c))·S_ab.
                        // ∂_c^geom C^(c) hits the h0-poly (2h0) and the V/V^(c) γ (via scalar_shift):
                        //   here approximated by the h0-poly geometric part only for the C-coefficient's 2h0;
                        //   the V-γ geometry piece is handled through the reconverged V^(c) already. Start with
                        //   the dominant `C^(c)·S_abc` + `P^(c)·2·h0_c·S_ab` terms.
                        // C:S_ab channel missing term (CORRECTED): the FD isolates the density-DIFFERENCE
                        // (O(h)) times the base kernel; ∂_c of that leaves the geometry derivative hitting the
                        // COEFFICIENT `(2h0−V)` (NOT S_ab — that product is O(h²) and vanishes). So the term is
                        //   P^(c) · [∂_c(2h0 − V)] · S_ab = P^(c)·(2·h0_c − V_geo_c)·S_ab.
                        // Here h0_c = ∂_c h0.value = h1(cc,axc); V_geo_c = ∂_c V = the AO-pair scalar-potential
                        // geometry derivative in slot c. Only the c-slot carries a geometry derivative.
                        let vgeo_c = vgeo_shell[dof(ic)][shell_mu_index] + vgeo_shell[dof(ic)][shell_nu_index];
                        if only_channel != Some(1) && only_channel != Some(2) && only_channel != Some(3) {
                            let s_ab = s2(ca, cb, axa, axb);
                            let h0c = h1(cc, axc);
                            v += pc * (2.0 * h0c - vgeo_c) * s_ab;
                        }
                        // channel 2: ONLY the 2·h0_c part.  channel 3: ONLY the −V_geo_c part.
                        if only_channel == Some(2) {
                            v += pc * 2.0 * h1(cc, axc) * s2(ca, cb, axa, axb);
                        }
                        if only_channel == Some(3) {
                            v += pc * (-vgeo_c) * s2(ca, cb, axa, axb);
                        }
                        // --- h0 channel: ∂_c[ 2P^(c)·(h0_a·S_b + h0_b·S_a + S·h0_ab) ] (all ordered).
                        if only_channel != Some(0) {
                            let (s_a, s_b, s_c) = (s1(ca, axa), s1(cb, axb), s1(cc, axc));
                            let s_ab = s2(ca, cb, axa, axb);
                            let s_ac = s2(ca, cc, axa, axc);
                            let s_bc = s2(cb, cc, axb, axc);
                            let (h_a, h_b) = (h1(ca, axa), h1(cb, axb));
                            let h_ab = h2(ca, cb, axa, axb);
                            let h_ac = h2(ca, cc, axa, axc);
                            let h_bc = h2(cb, cc, axb, axc);
                            let h_abc = h3(ca, cb, cc, axa, axb, axc);
                            // ∂_c of (h0_a·S_b + h0_b·S_a + S·h0_ab):
                            v += two_p
                                * (h_ac * s_b + h_a * s_bc
                                    + h_bc * s_a + h_b * s_ac
                                    + s_c * h_ab + overlap * h_abc);
                        }
                        if v != 0.0 {
                            tensor[dof(ic)][(dof(ia), dof(ib))] += v;
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

#[cfg(test)]
mod third_derivative_tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSet};

    // The frozen-shell-charge SCC2 third derivative (slab c = ∂H/∂R_c) must match the
    // central FD of its own analytic Hessian. The charges are held fixed, so this isolated
    // FD is valid (no electronic response in the frozen block).
    #[test]
    fn fixed_scc_third_derivative_matches_hessian_finite_difference() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let nsh = basis.shells.len();
        // Arbitrary but fixed shell charges (the block is bilinear in them).
        let q: Vec<f64> = (0..nsh)
            .map(|i| 0.13 * ((i % 3) as f64 - 1.0) + 0.05)
            .collect();

        let third = fixed_shell_charge_scc_third_derivative(&system, &basis, &q, &params).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = fixed_shell_charge_scc_hessian(&plus, &basis, &q, &params)
                .unwrap()
                .hessian;
            let hm = fixed_shell_charge_scc_hessian(&minus, &basis, &q, &params)
                .unwrap()
                .hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "frozen SCC third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // The frozen-density Pulay/overlap+H0 third derivative (slab c = ∂H/∂R_c) must match the
    // central FD of its own analytic Hessian. P, W, the shell potential and CN are all held
    // fixed (the `electronic` result is reused at displaced geometries), so this isolated FD
    // is valid — it carries no electronic response. This block is the first consumer of the
    // B1 third-derivative AO integrals.
    #[test]
    fn fixed_pulay_third_derivative_matches_hessian_finite_difference() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third = fixed_density_pulay_third_derivative(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = fixed_density_pulay_hessian(&plus, &params, &electronic)
                .unwrap()
                .hessian;
            let hm = fixed_density_pulay_hessian(&minus, &params, &electronic)
                .unwrap()
                .hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "frozen Pulay third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // The `d_edcn` jet (∂E/∂CN_A and its 1st/2nd/3rd nuclear derivatives at frozen density) must
    // satisfy its own derivative ladder: grad = FD(value), hess = FD(grad), third = FD(hess). This
    // validates the `scale·overlap` Leibniz (the error-prone core of the CN-H0 third derivative) in
    // isolation, reusing the same `electronic` (frozen density) at displaced geometries.
    #[test]
    fn cn_h0_dedcn_jet_derivative_ladder_matches_fd() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let jets = cn_h0_dedcn_jets(&system, &params, &electronic).unwrap();
        let nat = system.atoms.len();
        let ndof = 3 * nat;
        let step = 1.0e-5;
        let (mut max_g, mut max_h, mut max_t) = (0.0_f64, 0.0_f64, 0.0_f64);
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, d, step);
            displace(&mut minus, d, -step);
            let jp = cn_h0_dedcn_jets(&plus, &params, &electronic).unwrap();
            let jm = cn_h0_dedcn_jets(&minus, &params, &electronic).unwrap();
            for atom in 0..nat {
                let fd_g = (jp[atom].value - jm[atom].value) / (2.0 * step);
                max_g = max_g.max((jets[atom].grad[d] - fd_g).abs());
                for a in 0..ndof {
                    let fd_h = (jp[atom].grad[a] - jm[atom].grad[a]) / (2.0 * step);
                    max_h = max_h.max((jets[atom].hess[a * ndof + d] - fd_h).abs());
                    for b in 0..ndof {
                        let fd_t = (jp[atom].hess[a * ndof + b] - jm[atom].hess[a * ndof + b])
                            / (2.0 * step);
                        max_t =
                            max_t.max((jets[atom].third[(a * ndof + b) * ndof + d] - fd_t).abs());
                    }
                }
            }
        }
        assert!(max_g < 1.0e-6, "d_edcn grad vs FD: {max_g:.3e}");
        assert!(max_h < 1.0e-5, "d_edcn hess vs FD: {max_h:.3e}");
        assert!(max_t < 1.0e-4, "d_edcn third vs FD: {max_t:.3e}");
    }

    // The SCC-scalar × overlap-derivative coupling block's third derivative matches the central FD of
    // `fixed_density_scalar_overlap_hessian` at FIXED density (reference `electronic` reused at displaced
    // geometries). This is the block `analytic_hessian` adds but `third_derivative_frozen_complete` lacked.
    #[test]
    fn fixed_density_scalar_overlap_third_derivative_matches_hessian_fd() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third =
            fixed_density_scalar_overlap_third_derivative(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let hess = |sys: &PeriodicSystem| -> Matrix {
            fixed_density_scalar_overlap_hessian(sys, &params, &electronic).unwrap()
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = hess(&plus);
            let hm = hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "scalar-overlap third-derivative FD max delta {max_delta:.3e}"
        );
    }

    // Response-side block ladder, Step 1: the bare-H0 SECOND nuclear derivative (at fixed CN) matches the
    // central FD of the bare-H0 first derivative. Establishes the AO pair mapping + signs + the
    // scale·overlap 2nd-derivative machinery before adding SCC-scalar and CN-H0 blocks to `F_bc`.
    #[test]
    fn h0_bare_second_derivative_matches_fd() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, options).unwrap();
        let ndof = 3 * system.atoms.len();
        let n = electronic.basis.len();
        let step = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for b in 0..ndof {
            for c in 0..ndof {
                let analytic =
                    h0_bare_second_derivative_matrix(&system, &params, &electronic, b, c).unwrap();
                let (atom, ax) = (c / 3, c % 3);
                let mut sp = system.clone();
                let mut sm = system.clone();
                displace(&mut sp, 3 * atom + ax, step);
                displace(&mut sm, 3 * atom + ax, -step);
                // bare-H0 first deriv at displaced geometry, fixed reference CN (pass reference electronic).
                let fp = h0_bare_first_derivative_matrix(&sp, &params, &electronic, b).unwrap();
                let fm = h0_bare_first_derivative_matrix(&sm, &params, &electronic, b).unwrap();
                for mu in 0..n {
                    for nu in 0..n {
                        let fd = (fp[(mu, nu)] - fm[(mu, nu)]) / (2.0 * step);
                        max_delta = max_delta.max((analytic[(mu, nu)] - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "bare-H0 second derivative FD max delta {max_delta:.3e}"
        );
    }

    // The CN-H0 frozen third derivative (slab a = ∂H_bc/∂R_a) matches the central FD of the analytic
    // CN-H0 Hessian block (`fixed_density_cn_h0_hessian` + `fixed_density_cn_h0_pulay_cross_hessian`),
    // with the frozen `electronic` reused at displaced geometries. The last frozen `L_abc` block.
    #[test]
    fn fixed_density_cn_h0_third_derivative_matches_hessian_fd() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, options).unwrap();
        let third =
            fixed_density_cn_h0_third_derivative(&system, &params, &electronic, cutoff).unwrap();
        let ndof = 3 * system.atoms.len();
        let step = 1.0e-4;
        let cn_h0_hess = |sys: &PeriodicSystem| -> Matrix {
            let mut h = fixed_density_cn_h0_hessian(sys, &params, &electronic, cutoff)
                .unwrap()
                .hessian;
            let cross =
                fixed_density_cn_h0_pulay_cross_hessian(sys, &params, &electronic, cutoff).unwrap();
            for r in 0..ndof {
                for c in 0..ndof {
                    h[(r, c)] += cross[(r, c)];
                }
            }
            h
        };
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = cn_h0_hess(&plus);
            let hm = cn_h0_hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "CN-H0 third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    // The CN counting-function third radial derivative (sigmoid chain rule) must match the
    // central FD of its analytic second derivative. Prerequisite for the CN-H0 and D3
    // many-body third-derivative chain rules. Self-contained (no params).
    #[test]
    fn coordination_third_derivative_matches_second_finite_difference() {
        let kcn = CoordinationOptions::default().kcn;
        let h = 1.0e-6;
        for &rc in &[3.0_f64, 4.5] {
            for &r in &[1.2_f64, 2.5, 4.0, 6.0] {
                let d = coordination_value_derivatives(kcn, r, rc);
                let fd = (coordination_value_derivatives(kcn, r + h, rc).second
                    - coordination_value_derivatives(kcn, r - h, rc).second)
                    / (2.0 * h);
                assert!(
                    (d.third - fd).abs() < 1.0e-5 * (1.0 + d.third.abs()),
                    "CN''' at r={r}, rc={rc}: analytic {} vs FD {fd}",
                    d.third
                );
            }
        }
    }

    // The CN pair counting function's Cartesian third-derivative tensor (CN radials fed into
    // the shared central rank-3 block) must match the FD of the analytic CN pair Hessian.
    // This is the per-pair kernel the CN-H0 and D3 many-body assemblies build on.
    #[test]
    fn cn_pair_third_block_matches_hessian_finite_difference() {
        let kcn = CoordinationOptions::default().kcn;
        let rc = 3.2_f64;
        let pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.6, 0.7, 0.2)];
        let ndof = 6;
        let cn_hess = |p: &[Vec3; 2]| -> Matrix {
            let rel = p[0] - p[1];
            let r = rel.norm();
            let u = (rel / r).to_array();
            let d = coordination_value_derivatives(kcn, r, rc);
            let signs = [1.0_f64, -1.0];
            let mut hm = Matrix::zeros(ndof, ndof);
            for xi in 0..2 {
                for yi in 0..2 {
                    for a in 0..3 {
                        for b in 0..3 {
                            let dab = if a == b { 1.0 } else { 0.0 };
                            let hrel = d.second * u[a] * u[b] + (d.first / r) * (dab - u[a] * u[b]);
                            hm[(3 * xi + a, 3 * yi + b)] += signs[xi] * signs[yi] * hrel;
                        }
                    }
                }
            }
            hm
        };
        let rel = pos[0] - pos[1];
        let r = rel.norm();
        let d = coordination_value_derivatives(kcn, r, rc);
        let g = d.second / r - d.first / (r * r);
        let mut third = vec![Matrix::zeros(ndof, ndof); ndof];
        crate::third_derivative::add_radial_third_block(&mut third, 0, 1, rel, g, d.third, 1.0);
        let h = 1.0e-6;
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = pos;
            let mut minus = pos;
            let atom = slab / 3;
            match slab % 3 {
                0 => {
                    plus[atom].x += h;
                    minus[atom].x -= h;
                }
                1 => {
                    plus[atom].y += h;
                    minus[atom].y -= h;
                }
                _ => {
                    plus[atom].z += h;
                    minus[atom].z -= h;
                }
            }
            let hp = cn_hess(&plus);
            let hmn = cn_hess(&minus);
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hmn[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-5,
            "CN pair third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    fn displace(system: &mut PeriodicSystem, dof: usize, step: f64) {
        let atom = dof / 3;
        match dof % 3 {
            0 => system.atoms[atom].position.x += step,
            1 => system.atoms[atom].position.y += step,
            _ => system.atoms[atom].position.z += step,
        }
    }
}
