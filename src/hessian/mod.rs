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
    contracted_pair_with_derivatives, contracted_pair_with_fourth_derivatives,
    contracted_pair_with_second_derivatives, contracted_pair_with_third_derivatives,
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

/// Reject `ElectronicOptions` features whose second-derivative terms are NOT
/// implemented in the analytic Hessian assembly. Before v0.5.0 these were
/// silently dropped, so `analytic_hessian` returned a Hessian belonging to a
/// *different* energy expression than `run_electronic` — with no warning.
/// Delegates to the [`crate::terms`] registry (the single source of truth for
/// per-term derivative coverage).
pub(crate) fn ensure_hessian_supported_options(
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
) -> Result<()> {
    crate::terms::require_order(
        &options.electronic_options,
        params,
        2,
        "the analytic Hessian",
    )
}

pub fn analytic_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: AnalyticHessianOptions,
) -> Result<AnalyticHessianResult> {
    ensure_non_pbc(system)?;
    ensure_hessian_supported_options(params, &options)?;
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
    ensure_hessian_supported_options(params, &options)?;
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
        let result = halogen_energy_gradient_hessian(system, params)?;
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

/// Charge-path derivative of the SCC2 **third** derivative — the one-order-up twin of
/// [`fixed_shell_charge_scc_hessian_charge_path`]: `∂_q[T_abc(q)]·q^(c)` with
/// `T_abc(q) = Σ_{i<j} q_i q_j ∂³γ_ij`, i.e. the same radial third-derivative ladder as
/// [`fixed_shell_charge_scc_third_derivative`] with the quadratic weight `q_i q_j` replaced by the
/// bilinear `q^path_i q_j + q_i q^path_j`.
///
/// Consumed by the quartic assembly: the Hessian-level density path evaluates the `s2` block as
/// `fixed_shell_charge_scc_hessian_charge_path(q, q^(v))`, whose *geometric* `λ`-derivative is
/// exactly this third-level bilinear path. Because the block is quadratic in `q`, the same object
/// equals `[T(q + q^path) − T(q) − T(q^path)]` — asserted in `s2_third_charge_path_matches_polarization`.
pub(crate) fn fixed_shell_charge_scc_third_charge_path(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    shell_charge_response: &[f64],
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
            let scale = shell_charge_response[i] * shell_charges[j]
                + shell_charges[i] * shell_charge_response[j];
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

/// Analytic fourth Cartesian derivative `Q_abcd = ∂⁴E₂/∂R_a∂R_b∂R_c∂R_d` of the
/// **frozen-shell-charge** GFN1 second-order electrostatics
/// `E₂ = ½ Σ_{s≠t} q_s γ_st(R_st) q_t = Σ_{s<t} q_s q_t γ_st(R_st)` (shell charges held fixed),
/// in packed [`crate::fourth_derivative::SymmetricFourth`] storage. This is the frozen SCC2
/// block of the quartic assembly, one order above
/// [`fixed_shell_charge_scc_third_derivative`]; with the charges frozen it carries no
/// electronic response, so `Q.get(a, b, c, d)` equals `∂(third-derivative slab)/∂R_d`, which
/// is the FD gate.
///
/// Only *inter-atomic* shell pairs contribute: the on-site `s`/`t`-on-the-same-atom kernel is
/// the constant `harmonic_average(η_s, η_t)`, and the on-site third-order `⅓ Σ_A Γ_A q_A³`
/// (and any higher on-site charge order) is built from geometry-independent Hubbard
/// parameters only. At fixed `q` all of those are geometry-*independent* constants, so they
/// contribute nothing at any nuclear derivative order ≥ 1 — the fourth derivative, like the
/// third, is the pure inter-atomic Klopman–Ohno radial ladder.
pub fn fixed_shell_charge_scc_fourth_derivative(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<crate::fourth_derivative::SymmetricFourth> {
    ensure_non_pbc(system)?;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let nat = system.atoms.len();
    let mut store = crate::fourth_derivative::SymmetricFourth::zeros(3 * nat);
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
            let (c2, c3, c4) = effective_kernel_hat_derivatives(r, gamma);
            // Unordered shell pairs (j < i) with weight q_i q_j — the same convention the
            // third derivative uses, i.e. the ½ of E₂ is already absorbed by the pair count.
            // rvec = R_i − R_j is already the true relative vector (σ_i = +1).
            crate::fourth_derivative::add_radial_fourth_block_sym(
                &mut store,
                ai,
                aj,
                rvec,
                c2,
                c3,
                c4,
                shell_charges[i] * shell_charges[j],
            );
        }
    }
    Ok(store)
}

/// Peak scratch the chunked AO-pair reductions below may hold. Each chunk owns
/// one `ndof × ndof` partial, so this is what bounds the chunk count once the
/// system is large enough for that to matter.
const CHUNK_SCRATCH_BUDGET_BYTES: usize = 128 << 20;

/// The most chunks any sweep splits into. Comfortably above the core count of
/// current machines, so the tail of a chunk never idles a worker for long.
const MAX_PAIR_SWEEP_CHUNKS: usize = 64;

/// Contiguous outer-loop chunks carrying EQUAL triangular work, for the
/// `for mu in 0..n { for nu in 0..mu { … } }` AO-pair sweeps below.
///
/// Row `mu` costs `~mu`, so equal-WIDTH chunks would leave the first worker
/// nearly idle while the last one does most of the sweep. Splitting at
/// `n·√(k/K)` equalises `Σ mu` per chunk instead.
///
/// The chunk count is a function of `(n, ndof)` ALONE — never of
/// `rayon::current_num_threads()` — so the partial sums are reduced in a fixed
/// order that does not depend on the machine or on `RAYON_NUM_THREADS`. That
/// keeps the assembled block bit-reproducible, the same property the parallel
/// gradient's reduce-in-pair-order gives. (Rayon's own `reduce` splits by work
/// stealing, so its association — and hence the last bits — would vary run to
/// run.) The result still differs from the serial sweep by the floating-point
/// reassociation of one sum, which is orders below every FD gate's tolerance.
fn triangular_chunks(n: usize, ndof: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let per_chunk_bytes = (ndof * ndof * std::mem::size_of::<f64>()).max(1);
    let affordable = (CHUNK_SCRATCH_BUDGET_BYTES / per_chunk_bytes).max(1);
    let k = MAX_PAIR_SWEEP_CHUNKS.min(affordable).min(n);
    let mut bounds = Vec::with_capacity(k);
    let mut lo = 0usize;
    for step in 1..=k {
        let hi = if step == k {
            n
        } else {
            let frac = (step as f64 / k as f64).sqrt();
            (((n as f64) * frac).round() as usize).clamp(lo, n)
        };
        if hi > lo {
            bounds.push((lo, hi));
        }
        lo = hi;
    }
    bounds
}

/// Sum `src` into `dst` element-wise; both are the same shape.
fn accumulate_into(dst: &mut Matrix, src: &Matrix) {
    for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
        *d += *s;
    }
}

pub fn fixed_density_pulay_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<FixedDensityPulayHessianResult> {
    use rayon::prelude::*;
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);

    let partials = triangular_chunks(basis.len(), 3 * nat)
        .par_iter()
        .map(|&(first_mu, end_mu)| -> Result<(Vec<Vec3>, Matrix)> {
            let mut gradient = vec![Vec3::zero(); nat];
            let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
            for mu in first_mu..end_mu {
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
                    let pair = contracted_pair_with_second_derivatives(
                        &basis.aos[mu],
                        &basis.aos[nu],
                        rmu,
                        rnu,
                    );
                    let overlap = pair.moments[0];
                    let h0 = h0_prefactor_second(
                        system,
                        params,
                        electronic,
                        shell_mu_index,
                        shell_nu_index,
                    )?;
                    let p = electronic.density[(mu, nu)];
                    let w = electronic.energy_weighted_density[(mu, nu)];
                    if p.abs().max(w.abs()) <= 1.0e-18 {
                        continue;
                    }
                    let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
                    let overlap_coeff = p * (2.0 * h0.value - scalar_shift) - 2.0 * w;
                    gradient[atom_mu] +=
                        pair.d_bra[0] * overlap_coeff + h0.d_bra * (2.0 * p * overlap);
                    gradient[atom_nu] +=
                        pair.d_ket[0] * overlap_coeff + h0.d_ket * (2.0 * p * overlap);

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
                                    let ds_row = first(
                                        &pair.d_bra[0],
                                        &pair.d_ket[0],
                                        row_center,
                                        row_axis,
                                    );
                                    let ds_col = first(
                                        &pair.d_bra[0],
                                        &pair.d_ket[0],
                                        col_center,
                                        col_axis,
                                    );
                                    let dh_row =
                                        first_vec(h0.d_bra, h0.d_ket, row_center, row_axis);
                                    let dh_col =
                                        first_vec(h0.d_bra, h0.d_ket, col_center, col_axis);
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
                                        + 2.0
                                            * p
                                            * (dh_col * ds_row + ds_col * dh_row + overlap * d2h);
                                    hessian
                                        [(3 * row_atom + row_axis, 3 * col_atom + col_axis)] +=
                                        value;
                                }
                            }
                        }
                    }
                }
            }
            Ok((gradient, hessian))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut gradient = vec![Vec3::zero(); nat];
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
    for (chunk_gradient, chunk_hessian) in &partials {
        for (dst, src) in gradient.iter_mut().zip(chunk_gradient) {
            *dst += *src;
        }
        accumulate_into(&mut hessian, chunk_hessian);
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
    use rayon::prelude::*;
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

    let pair_partials = triangular_chunks(basis.len(), 3 * nat)
        .par_iter()
        .map(|&(first_mu, end_mu)| -> Result<(Vec<f64>, Matrix)> {
            let mut d_edcn = vec![0.0; nat];
            let mut d_edcn_dr = Matrix::zeros(nat, ndof);
            for mu in first_mu..end_mu {
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
                    let pair = contracted_pair_with_second_derivatives(
                        &basis.aos[mu],
                        &basis.aos[nu],
                        rmu,
                        rnu,
                    );
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
            Ok((d_edcn, d_edcn_dr))
        })
        .collect::<Result<Vec<_>>>()?;
    for (chunk_edcn, chunk_edcn_dr) in &pair_partials {
        for (dst, src) in d_edcn.iter_mut().zip(chunk_edcn) {
            *dst += *src;
        }
        accumulate_into(&mut d_edcn_dr, chunk_edcn_dr);
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
    use rayon::prelude::*;
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let cn_derivatives = coordination_number_first_derivatives(system, coordination_cutoff)?;

    let partials = triangular_chunks(basis.len(), 3 * nat)
        .par_iter()
        .map(|&(first_mu, end_mu)| -> Result<Matrix> {
            let mut hessian = Matrix::zeros(ndof, ndof);
            for mu in first_mu..end_mu {
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
        })
        .collect::<Result<Vec<_>>>()?;

    let mut hessian = Matrix::zeros(ndof, ndof);
    for chunk in &partials {
        accumulate_into(&mut hessian, chunk);
    }
    Ok(hessian)
}

pub fn fixed_density_scalar_overlap_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Matrix> {
    use rayon::prelude::*;
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let shell_scalar_derivatives =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;

    let partials: Vec<Matrix> = triangular_chunks(basis.len(), 3 * nat)
        .par_iter()
        .map(|&(first_mu, end_mu)| {
            let mut hessian = Matrix::zeros(ndof, ndof);
            for mu in first_mu..end_mu {
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
                            let dscalar_col = shell_scalar_derivatives
                                [(shell_mu_index, col_coord)]
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
            hessian
        })
        .collect();

    let mut hessian = Matrix::zeros(ndof, ndof);
    for chunk in &partials {
        accumulate_into(&mut hessian, chunk);
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
    /// Fourth radial derivative of the smooth CN counting function, for the analytic **fourth**
    /// nuclear derivative (the frozen CN-H0 quartic block). Same sigmoid ladder one order up.
    fourth: f64,
}

fn coordination_value_derivatives(kcn: f64, r: f64, rc: f64) -> ScalarDerivatives {
    let raw_arg = -kcn * (rc / r - 1.0);
    if !(-80.0..=80.0).contains(&raw_arg) {
        return ScalarDerivatives {
            first: 0.0,
            second: 0.0,
            third: 0.0,
            fourth: 0.0,
        };
    }
    let expterm = raw_arg.exp();
    let denom = 1.0 + expterm;
    let arg1 = kcn * rc / (r * r);
    let arg2 = -2.0 * kcn * rc / (r * r * r);
    let arg3 = 6.0 * kcn * rc / (r * r * r * r);
    let arg4 = -24.0 * kcn * rc / (r * r * r * r * r);
    let first = -expterm * arg1 / (denom * denom);
    let second = -expterm * (arg1 * arg1 + arg2) / (denom * denom)
        + 2.0 * expterm * expterm * arg1 * arg1 / (denom * denom * denom);
    // Third/fourth derivatives via the sigmoid `σ = 1/denom`: with `cn = σ(arg(r))`,
    // `cn''' = σ₃ arg'³ + 3 σ₂ arg' arg'' + σ₁ arg'''` and (Faà di Bruno one order up)
    // `cn'''' = σ₄ arg'⁴ + 6 σ₃ arg'² arg'' + σ₂ (3 arg''² + 4 arg' arg''') + σ₁ arg''''`.
    //
    // The σ-ladder is written as rational functions of `e = expterm` rather than of `σ`:
    //
    //   σ₁ = −e/D²,  σ₂ = e(e−1)/D³,  σ₃ = −e(e²−4e+1)/D⁴,  σ₄ = e(e³−11e²+11e−1)/D⁵,  D = 1+e
    //
    // which is the same ladder as the equivalent `σ(1−σ)·polynomial(σ)` form but WITHOUT the
    // catastrophic `1 − σ` cancellation. That cancellation is not academic: at saturation
    // (`σ → 1`, i.e. `r` well inside `rc`) `1 − σ` keeps only ~6 significant digits for
    // `e ~ 1e−11`, and `third` inherited that error — visible as ~7 % noise when
    // finite-differencing `third` to check `fourth`, and as an exact-zero `third` once
    // `1 + e` rounds to `1`. `first`/`second` were already written in the `e` form and are
    // left untouched.
    let d2 = denom * denom;
    let d3 = d2 * denom;
    let d4 = d3 * denom;
    let d5 = d4 * denom;
    let e2 = expterm * expterm;
    let sig1 = -expterm / d2;
    let sig2 = expterm * (expterm - 1.0) / d3;
    let sig3 = -expterm * (e2 - 4.0 * expterm + 1.0) / d4;
    let sig4 = expterm * (e2 * expterm - 11.0 * e2 + 11.0 * expterm - 1.0) / d5;
    let third = sig3 * arg1 * arg1 * arg1 + 3.0 * sig2 * arg1 * arg2 + sig1 * arg3;
    let fourth = sig4 * arg1 * arg1 * arg1 * arg1
        + 6.0 * sig3 * arg1 * arg1 * arg2
        + sig2 * (3.0 * arg2 * arg2 + 4.0 * arg1 * arg3)
        + sig1 * arg4;
    ScalarDerivatives {
        first,
        second,
        third,
        fourth,
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
///
/// Pub(crate) so the directional quartic assembly can build the SECOND directional CN response
/// `CN^vv_A = Σ_cd v_c v_d ∂²CN_A/∂R_c∂R_d` — the `λ`-motion of the `CN^v` vector that the FC3
/// composition's Pulay CN-response term reads (see
/// [`crate::fourth_derivative::directional::directional_fourth_cn_response_stage`]). Cheaper than
/// [`cn_h0_cn_jets`], which also builds the `O(nat·ndof³)` third derivatives.
pub(crate) fn coordination_number_second_derivatives(
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

/// Third nuclear derivatives of the per-shell SCC scalar potential `V_s = Σ_t γ_{st}(R) q_t`, at
/// FIXED charges: `out[s][(b·ndof + c)·ndof + d] = ∂³V_s/∂R_b∂R_c∂R_d = Σ_t (∂³γ_{st}) q_t`. One
/// order above [`shell_scalar_potential_second_derivatives`]: the same per-pair central block,
/// now the rank-3 radial tensor `T_abc = (f''' − 3g)·u_a u_b u_c + g·(δ_ab u_c + δ_ac u_b + δ_bc u_a)`
/// with `g = f''/r − f'/r² = p'` (identical to the kernel coefficients
/// [`crate::third_derivative::add_radial_third_block`] consumes), scattered over the `2³` atom
/// assignments with sign `(−1)^{#(indices on the second atom)}` and weighted by the OTHER shell's
/// charge. Consumed by [`fixed_density_scalar_overlap_fourth_derivative`] as the `∂_d ∂_c dscalar_b`
/// factor. Flat `ndof³` storage per shell (a `Vec<Matrix>` per shell would be `ndof` matrices).
pub fn shell_scalar_potential_third_derivatives(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<Vec<Vec<f64>>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let nsh = basis.shells.len();
    if shell_charges.len() != nsh {
        return Err(Gfn1Error::InvalidInput(
            "shell charge dimension mismatch for scalar-potential third derivatives".to_string(),
        ));
    }
    let model = ShellChargeModel::build(system, basis, params)?;
    let mut out = vec![vec![0.0_f64; ndof * ndof * ndof]; nsh];
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
            let g = d.gradient_prefactor_derivative;
            let coeff_uuu = d.radial_third_derivative - 3.0 * g;
            let u = (rvec / r).to_array();
            let atoms = [ai, aj];
            let signs = [1.0_f64, -1.0];
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let kron = if a == b { u[c] } else { 0.0 }
                            + if a == c { u[b] } else { 0.0 }
                            + if b == c { u[a] } else { 0.0 };
                        let t_rel = coeff_uuu * u[a] * u[b] * u[c] + g * kron;
                        if t_rel == 0.0 {
                            continue;
                        }
                        for (xi, &ax) in atoms.iter().enumerate() {
                            for (yi, &ay) in atoms.iter().enumerate() {
                                for (zi, &az) in atoms.iter().enumerate() {
                                    let value = signs[xi] * signs[yi] * signs[zi] * t_rel;
                                    let idx = ((3 * ax + a) * ndof + (3 * ay + b)) * ndof
                                        + (3 * az + c);
                                    out[i][idx] += value * shell_charges[j];
                                    out[j][idx] += value * shell_charges[i];
                                }
                            }
                        }
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
///
/// **No coordination-number dependence** — deliberately checked. Unlike the Pulay block (whose `h0`
/// prefactor reads the cached `electronic.coordination_numbers`, and therefore needs
/// [`fixed_density_pulay_third_cn_response`] once the electronic reference is allowed to move), this
/// block is built only from `P`, the overlap derivatives, and the shell scalar-potential derivatives.
/// The latter come from `shell_scalar_potential_{first,second}_derivatives`, which take only
/// `(system, basis, shell_charges, params)` and build the γ/hardness `ShellChargeModel`: no
/// self-energy, no `kcn`, no CN anywhere. So its `λ`-derivative is fully covered by the geometric
/// fourth block plus the `(P, q)` density paths, with no CN-response companion term.
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
    /// Fourth radial derivative `f''''` of the kernel `f(r)`, for the analytic fourth nuclear
    /// derivative. Continuing the same ladder one order with `p''' = p(L³ + 3 L L' + L'')`,
    /// the chain gives `f'''' = 3 p'' + r p'''`. For the GFN1 exponent `g = 2` this reduces to
    /// the Klopman–Ohno closed form
    /// `γ'''' = 105 r⁴ s^(−9/2) − 90 r² s^(−7/2) + 9 s^(−5/2)`, `s = r² + 1/γ_h²`.
    radial_fourth_derivative: f64,
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
    // L'' = d²(log_derivative)/dr², differentiating `L' ` term by term (same `a` coefficient).
    let log_derivative_double = 2.0 * (g - 2.0) / (r * r * r)
        + a * (g - 1.0) * (g - 2.0) * r.powf(g - 3.0) / sum
        - 3.0 * a * g * (g - 1.0) * r.powf(2.0 * g - 3.0) / (sum * sum)
        + 2.0 * a * g * g * r.powf(3.0 * g - 3.0) / (sum * sum * sum);
    let p = prefactor;
    let p_prime = p * log_derivative;
    let p_double = p * (log_derivative * log_derivative + log_derivative_prime);
    let p_triple = p
        * (log_derivative * log_derivative * log_derivative
            + 3.0 * log_derivative * log_derivative_prime
            + log_derivative_double);
    let radial_third_derivative = 2.0 * p_prime + r * p_double;
    let radial_fourth_derivative = 3.0 * p_double + r * p_triple;
    KernelDerivatives {
        value,
        gradient_prefactor: prefactor,
        gradient_prefactor_derivative: p_prime,
        radial_third_derivative,
        radial_fourth_derivative,
    }
}

/// Radial "hat" ladder of the effective Klopman–Ohno kernel `γ(r) = (r^g + γ_h^(−g))^(−1/g)`,
/// returned as `(c2, c3, c4) = (D̂²γ, D̂³γ, D̂⁴γ)` with `D̂ = (1/r) d/dr` -- exactly the Cartesian
/// tensor coefficients consumed by [`crate::fourth_derivative::add_radial_fourth_block_sym`]:
///
/// ```text
///   c2 = γ''/r² − γ'/r³
///   c3 = γ'''/r³ − 3γ''/r⁴ + 3γ'/r⁵
///   c4 = γ''''/r⁴ − 6γ'''/r⁵ + 15γ''/r⁶ − 15γ'/r⁷
/// ```
///
/// The plain radial derivatives come from [`effective_kernel_derivatives`] (`γ' = r·p`,
/// `γ'' = p + r·p'`), so both orders share one ladder.
fn effective_kernel_hat_derivatives(r: f64, gamma: f64) -> (f64, f64, f64) {
    let d = effective_kernel_derivatives(r, gamma);
    let f1 = r * d.gradient_prefactor;
    let f2 = d.gradient_prefactor + r * d.gradient_prefactor_derivative;
    let f3 = d.radial_third_derivative;
    let f4 = d.radial_fourth_derivative;
    let c2 = f2 / r.powi(2) - f1 / r.powi(3);
    let c3 = f3 / r.powi(3) - 3.0 * f2 / r.powi(4) + 3.0 * f1 / r.powi(5);
    let c4 = f4 / r.powi(4) - 6.0 * f3 / r.powi(5) + 15.0 * f2 / r.powi(6) - 15.0 * f1 / r.powi(7);
    (c2, c3, c4)
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

/// The Pulay THIRD derivative's COORDINATION-NUMBER response along a per-atom CN response vector —
/// the one-order-up twin of [`fixed_density_pulay_cn_h0_response`], and the term
/// [`fixed_density_pulay_third_derivative`] omits when the electronic reference is allowed to move.
///
/// [`h0_prefactor_third`] builds `h0 = ½(self_i+self_j)·hscale·poly` from the **cached**
/// `electronic.coordination_numbers`, so the block is CN-frozen: a reconverged (FD) reference also
/// differentiates that cached CN, while the analytic frozen fourth block
/// [`fixed_density_pulay_fourth_derivative`] does not. With `cn_response[A] = Σ_c v_c·∂CN_A/∂R_c`
/// (purely geometric, from [`cn_gradient_matrix`]), the CN-derivative of `h0` along that vector is
/// `s_cn·(hscale·shell_poly)` field-for-field — exactly the Hessian-level construction, one
/// derivative order up (`h0_scale_third` instead of `h0_scale_second`) — with
/// `s_cn = −½·(kcn_i·cn_response[i] + kcn_j·cn_response[j])`, because `∂self_i/∂CN_i = −kcn_i`.
///
/// Every CN-carrying field of the Pulay third-derivative Leibniz is then re-evaluated with
/// `h0 → h0^cn`: the coefficient channel becomes `2P·h0^cn·S_abc` (the `−P·V` and `−2W` parts of
/// `overlap_coeff` have no CN dependence) and the h0 channel keeps its full 7-term shape with the
/// CN-differentiated prefactor. Returned as dense slabs `tensor[c][(a,b)]` in the same fully
/// symmetric layout as [`fixed_density_pulay_third_derivative`], so it contracts with `vvv`.
///
/// This term belongs to the FOURTH derivative only — the FC3 assembly already carries its own
/// (one-order-down) CN-response term, [`fixed_density_pulay_cn_h0_response`].
///
/// `coordination_cutoff` is accepted for call-site symmetry with the other CN-coupled blocks; the
/// CN response itself is supplied pre-contracted, so no coordination sum is rebuilt here.
pub fn fixed_density_pulay_third_cn_response(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    _coordination_cutoff: f64,
    cn_response: &[f64],
) -> Result<Vec<Matrix>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if cn_response.len() != nat {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "fixed_density_pulay_third_cn_response: cn_response length {} != natoms {}",
            cn_response.len(),
            nat
        )));
    }
    let basis = &electronic.basis;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];

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
            let kcn_i = basis.shells[shell_mu_index].kcn_raw.unwrap_or(0.0);
            let kcn_j = basis.shells[shell_nu_index].kcn_raw.unwrap_or(0.0);
            // s_cn = ∂base/∂λ|_CN / hscale = −½(kcn_i·CN^v_i + kcn_j·CN^v_j).
            let s_cn = -0.5 * (kcn_i * cn_response[atom_mu] + kcn_j * cn_response[atom_nu]);
            if s_cn == 0.0 {
                continue;
            }
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            // CN-derivative of the h0 prefactor along cn_response: s_cn·(hscale·poly), field-for-field.
            let h0 = h0_scale_third(system, params, shell_mu_index, shell_nu_index, basis)?
                .scale(s_cn);
            // Only the `2·P·h0` part of `overlap_coeff = P(2h0 − V) − 2W` carries CN.
            let overlap_coeff = 2.0 * p * h0.value;
            let two_p = 2.0 * p;

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
                        for &(a, b, c) in &distinct_perms(i, j, k) {
                            tensor[dof(c)][(dof(a), dof(b))] += v;
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
            // (Kept for documentation; this probe currently exercises only the two_p channel.)
            let _overlap_coeff =
                pc * (2.0 * h0.value - scalar_shift) - p * scalar_shift_c - 2.0 * wc;
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
            // (Overlap third selector kept for probe symmetry with h3; currently unused.)
            let _s3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
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

// ===========================================================================
// Phase 4 — frozen (geometry-only) FOURTH-derivative ladders and blocks.
//
// Every public block below returns `Vec<Vec<Matrix>>` with
//
//     out[c][d][(a, b)] = ∂_d ( third[c][(a, b)] )
//
// where `third` is the block's own third-derivative function. That is the ONLY
// contract these blocks honour, and it is deliberately the *unsymmetrised* one:
// the third-order slabs carry the asymmetries of their own decomposition (most
// visibly `fixed_density_cn_h0_third_derivative`, which omits the `CN·de_bc`
// term the band/Pulay block already carries), so their ∂_d derivative inherits
// exactly those asymmetries. Phase 6 symmetrises when it assembles `Q_abcd`.
//
// Consequence: the acoustic sum rule does NOT hold for these blocks in
// isolation. Frozen `P`, `W`, `V` and CN are *not* translated with the nuclei,
// so a rigid translation changes the block's value; the third-order blocks have
// the same property. ASR is a property of the fully assembled, response-complete
// quartic tensor, not of any frozen ingredient.
// ===========================================================================

/// Rank-4 Cartesian tensor, bra indices first and ket indices last — the same convention as
/// [`crate::integrals::ContractedPairFourthDerivatives`]'s `q_*` fields
/// (`q_bbkk[a][b][c][d] = ∂_{A_a}∂_{A_b}∂_{B_c}∂_{B_d}`).
type Ten4 = [[[[f64; 3]; 3]; 3]; 3];

fn scale_ten4(mut t: Ten4, s: f64) -> Ten4 {
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    t[a][b][c][d] *= s;
                }
            }
        }
    }
    t
}

/// `H0` (or bare `poly`) value plus its bra/ket Cartesian derivative tensors up to **fourth**
/// order: [`H0Third`] with the five fourth-order centre patterns appended, mirroring how
/// [`crate::integrals::ContractedPairFourthDerivatives`] extends its third-order struct.
#[derive(Clone, Debug)]
struct H0Fourth {
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
    q_bbbb: Ten4,
    q_bbbk: Ten4,
    q_bbkk: Ten4,
    q_bkkk: Ten4,
    q_kkkk: Ten4,
}

impl H0Fourth {
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
            q_bbbb: [[[[0.0; 3]; 3]; 3]; 3],
            q_bbbk: [[[[0.0; 3]; 3]; 3]; 3],
            q_bbkk: [[[[0.0; 3]; 3]; 3]; 3],
            q_bkkk: [[[[0.0; 3]; 3]; 3]; 3],
            q_kkkk: [[[[0.0; 3]; 3]; 3]; 3],
        }
    }

    fn scale(&self, s: f64) -> Self {
        let mut out = self.clone();
        out.value *= s;
        out.d_bra = out.d_bra * s;
        out.d_ket = out.d_ket * s;
        out.h_bra_bra = scale_3x3(out.h_bra_bra, s);
        out.h_bra_ket = scale_3x3(out.h_bra_ket, s);
        out.h_ket_ket = scale_3x3(out.h_ket_ket, s);
        out.t_bra_bra_bra = scale_ten3(out.t_bra_bra_bra, s);
        out.t_bra_bra_ket = scale_ten3(out.t_bra_bra_ket, s);
        out.t_bra_ket_ket = scale_ten3(out.t_bra_ket_ket, s);
        out.t_ket_ket_ket = scale_ten3(out.t_ket_ket_ket, s);
        out.q_bbbb = scale_ten4(out.q_bbbb, s);
        out.q_bbbk = scale_ten4(out.q_bbbk, s);
        out.q_bbkk = scale_ten4(out.q_bbkk, s);
        out.q_bkkk = scale_ten4(out.q_bkkk, s);
        out.q_kkkk = scale_ten4(out.q_kkkk, s);
        out
    }

    /// The embedded lower-order data as an [`H0Third`], so the fourth-order blocks can hand the
    /// same object to the third-order Leibniz helpers (and so a consistency test can compare
    /// against [`shell_poly_third`] directly).
    #[cfg(test)]
    fn third(&self) -> H0Third {
        H0Third {
            value: self.value,
            d_bra: self.d_bra,
            d_ket: self.d_ket,
            h_bra_bra: self.h_bra_bra,
            h_bra_ket: self.h_bra_ket,
            h_ket_ket: self.h_ket_ket,
            t_bra_bra_bra: self.t_bra_bra_bra,
            t_bra_bra_ket: self.t_bra_bra_ket,
            t_bra_ket_ket: self.t_bra_ket_ket,
            t_ket_ket_ket: self.t_ket_ket_ket,
        }
    }
}

/// Fourth-order bra/ket derivative tensors of the GFN1 distance polynomial `poly(r) = fi·fj`,
/// `fi = 1 + p_i√(r/R_AB)` — [`shell_poly_third`] extended by one rung of the same two ladders.
///
/// *Radial* rung: with `s = √(r/R)` the polynomial is `P(s) = 1 + (p_i+p_j)s + p_i p_j s²`, so
/// `P''' ≡ 0` and Faà di Bruno truncates to
/// `φ'''' = P''(3 s''² + 4 s' s''') + P' s''''`, with `s'''' = −15/(16 R⁴ s⁷)` continuing the
/// `s'`/`s''`/`s'''` chain already in [`shell_poly_third`].
///
/// *Angular* rung: the relative-vector rank-4 block in "hat" form (`D̂ = (1/r) d/dr`),
///
/// ```text
///   T_abcd = A4 u_a u_b u_c u_d
///          + A3 (δ_ab u_c u_d + δ_ac u_b u_d + δ_ad u_b u_c
///                + δ_bc u_a u_d + δ_bd u_a u_c + δ_cd u_a u_b)
///          + A2 (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc)
/// ```
///
/// with `A2 = g/r`, `A3 = (φ''' − 3g)/r` (`g = φ''/r − φ'/r²`, the third-order coefficients
/// reused verbatim) and `A4 = φ'''' − 6φ'''/r + 15φ''/r² − 15φ'/r³`. Same signature as
/// [`crate::fourth_derivative::add_radial_fourth_block_sym`]'s `(c2, c3, c4)` ladder, written
/// against the unit vector instead of the raw relative vector.
///
/// Signs: `dr = R_ket − R_bra` gives `σ_bra = −1`, `σ_ket = +1`, so a pattern with `m` bra
/// indices carries `(−1)^m` — even for `q_bbbb`/`q_bbkk`/`q_kkkk`, odd for `q_bbbk`/`q_bkkk`.
fn shell_poly_fourth(
    system: &PeriodicSystem,
    ai: usize,
    aj: usize,
    zi: u8,
    zj: u8,
    pi: Option<f64>,
    pj: Option<f64>,
) -> Result<H0Fourth> {
    let ipoly = pi.unwrap_or(0.0);
    let jpoly = pj.unwrap_or(0.0);
    if ipoly == 0.0 && jpoly == 0.0 {
        return Ok(H0Fourth::constant(1.0));
    }
    let dr = system.atoms[aj].position - system.atoms[ai].position;
    let r = dr.norm();
    if r <= 1.0e-12 {
        return Ok(H0Fourth::constant(1.0));
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
    let d4s = -15.0 / (16.0 * rad_sum.powi(4) * s_safe.powi(7));
    let linear = ipoly * fj + jpoly * fi; // = P'(s)
    let quad = 2.0 * ipoly * jpoly; // = P''(s); P'''(s) ≡ 0
    let phi_p = linear * ds;
    let phi_pp = quad * ds * ds + linear * d2s;
    let phi_ppp = 3.0 * quad * ds * d2s + linear * d3s;
    let phi_pppp = quad * (3.0 * d2s * d2s + 4.0 * ds * d3s) + linear * d4s;

    let u = (dr / r).to_array();
    let g = phi_pp / r - phi_p / (r * r);
    let coeff_uuu3 = phi_ppp - 3.0 * g;
    let a2 = g / r;
    let a3 = coeff_uuu3 / r;
    let a4 = phi_pppp - 6.0 * phi_ppp / r + 15.0 * phi_pp / (r * r) - 15.0 * phi_p / (r * r * r);
    let delta = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
    let mut t_rel2 = [[0.0_f64; 3]; 3];
    let mut t_rel3 = [[[0.0_f64; 3]; 3]; 3];
    let mut t_rel4 = [[[[0.0_f64; 3]; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let dab = delta(a, b);
            t_rel2[a][b] = phi_pp * u[a] * u[b] + (phi_p / r) * (dab - u[a] * u[b]);
            for c in 0..3 {
                let (dac, dbc) = (delta(a, c), delta(b, c));
                let kron = dab * u[c] + dac * u[b] + dbc * u[a];
                t_rel3[a][b][c] = coeff_uuu3 * u[a] * u[b] * u[c] + g * kron;
                for d in 0..3 {
                    let (dad, dbd, dcd) = (delta(a, d), delta(b, d), delta(c, d));
                    t_rel4[a][b][c][d] = a4 * u[a] * u[b] * u[c] * u[d]
                        + a3 * (dab * u[c] * u[d]
                            + dac * u[b] * u[d]
                            + dad * u[b] * u[c]
                            + dbc * u[a] * u[d]
                            + dbd * u[a] * u[c]
                            + dcd * u[a] * u[b])
                        + a2 * (dab * dcd + dac * dbd + dad * dbc);
                }
            }
        }
    }
    let neg2 = scale_3x3(t_rel2, -1.0);
    let neg3 = scale_ten3(t_rel3, -1.0);
    let neg4 = scale_ten4(t_rel4, -1.0);
    Ok(H0Fourth {
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
        q_bbbb: t_rel4,
        q_bbbk: neg4,
        q_bbkk: t_rel4,
        q_bkkk: neg4,
        q_kkkk: t_rel4,
    })
}

/// H0 geometric scale `hscale·shell_poly` to **fourth** order — the [`h0_scale_third`] analogue,
/// used by the CN-H0 fourth derivative's `d_edcn` jet build.
fn h0_scale_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    shell_mu: usize,
    shell_nu: usize,
    basis: &BasisSet,
) -> Result<H0Fourth> {
    let si = &basis.shells[shell_mu];
    let sj = &basis.shells[shell_nu];
    let base = hscale(si, sj, params)?;
    let poly = shell_poly_fourth(
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

/// `base · poly_fourth`, the fourth-order analogue of [`h0_prefactor_third`]. `base`
/// (`½(self_i+self_j)·hscale`, with the **frozen** CN-dependent self-energies) is a geometry
/// constant in this fixed-density block; only the polynomial varies with geometry.
fn h0_prefactor_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    shell_mu: usize,
    shell_nu: usize,
) -> Result<H0Fourth> {
    let si = &electronic.basis.shells[shell_mu];
    let sj = &electronic.basis.shells[shell_nu];
    let self_i = shell_self_energy(si, electronic.coordination_numbers[si.atom_index]);
    let self_j = shell_self_energy(sj, electronic.coordination_numbers[sj.atom_index]);
    let base = 0.5 * (self_i + self_j) * hscale(si, sj, params)?;
    let poly = shell_poly_fourth(
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

/// The distinct ordered permutations of the (possibly repeated) quadruple `(i,j,k,l)` — the
/// [`distinct_perms`] analogue one index up, used to scatter a symmetry-unique fourth-derivative
/// value to every tensor position it occupies.
fn distinct_perms4(i: usize, j: usize, k: usize, l: usize) -> Vec<(usize, usize, usize, usize)> {
    let base = [i, j, k, l];
    let mut out: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(24);
    for a in 0..4 {
        for b in 0..4 {
            if b == a {
                continue;
            }
            for c in 0..4 {
                if c == a || c == b {
                    continue;
                }
                let d = 6 - a - b - c; // 0+1+2+3 = 6
                let p = (base[a], base[b], base[c], base[d]);
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Select the fourth overlap derivative from the five center-pattern tensors (ket indices last),
/// permuting axes so the bras come first — [`third_select`] with one extra index. Mixed partials
/// commute, so only the per-center axis assignment matters.
#[allow(clippy::too_many_arguments)]
fn fourth_select(
    q_bbbb: &Ten4,
    q_bbbk: &Ten4,
    q_bbkk: &Ten4,
    q_bkkk: &Ten4,
    q_kkkk: &Ten4,
    centers: [Center; 4],
    axes: [usize; 4],
) -> f64 {
    let mut bra = [0usize; 4];
    let mut ket = [0usize; 4];
    let mut nb = 0;
    let mut nk = 0;
    for k in 0..4 {
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
        4 => q_bbbb[bra[0]][bra[1]][bra[2]][bra[3]],
        3 => q_bbbk[bra[0]][bra[1]][bra[2]][ket[0]],
        2 => q_bbkk[bra[0]][bra[1]][ket[0]][ket[1]],
        1 => q_bkkk[bra[0]][ket[0]][ket[1]][ket[2]],
        _ => q_kkkk[ket[0]][ket[1]][ket[2]][ket[3]],
    }
}

/// Analytic **fourth** Cartesian derivative of the fixed-density band/H0 + SCC-overlap + Pulay
/// energy (frozen `P`, `W`, shell potential, CN) — [`fixed_density_pulay_third_derivative`] one
/// order up, returned as `out[c][d][(a,b)] = ∂_d ( third[c][(a,b)] )`.
///
/// Per AO pair the third-order Leibniz value already telescopes to the true third derivative of
/// the pair energy `E = 2P·h0·S − P·V·S − 2W·S` (the `P·2h0·S_abc` terms cancel between
/// `overlap_coeff·S_abc` and the `2P·h0·S_abc` inside the `h0` channel), so the fourth order is
/// the same expression with one more index:
///
/// ```text
///   overlap_coeff·S_abcd
///     + 2P·[ ∂⁴(h0·S) − h0·S_abcd ]
///   = −P·V·S_abcd − 2W·S_abcd + 2P·∂⁴(h0·S)
/// ```
///
/// with the 15 Leibniz partners (`4 + 6 + 4 + 1`) written out explicitly. Because `P`, `W`, `V`
/// and the CN-dependent `self_avg` are geometry constants here, the result is the genuine fourth
/// derivative of a scalar and is fully index-symmetric — each unordered slot quadruple is
/// evaluated once and scattered over [`distinct_perms4`].
///
/// Integral patterns come from [`crate::integrals::contracted_pair_with_fourth_derivatives`] via
/// [`fourth_select`]; the EHT prefactor from [`h0_prefactor_fourth`]. Frozen ⇒ no electronic
/// response, so it FD-validates in isolation against its own third derivative. The acoustic sum
/// rule does **not** apply (see the module-section note above).
pub fn fixed_density_pulay_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<Vec<Matrix>>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut tensor = vec![vec![Matrix::zeros(ndof, ndof); ndof]; ndof];
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
                contracted_pair_with_fourth_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_fourth(system, params, electronic, shell_mu_index, shell_nu_index)?;
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
            let s3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &pair.t_bra_bra_bra[0],
                    &pair.t_bra_bra_ket[0],
                    &pair.t_bra_ket_ket[0],
                    &pair.t_ket_ket_ket[0],
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let s4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &pair.q_bbbb,
                    &pair.q_bbbk,
                    &pair.q_bbkk,
                    &pair.q_bkkk,
                    &pair.q_kkkk,
                    centers,
                    axes,
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
            let h3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &h0.t_bra_bra_bra,
                    &h0.t_bra_bra_ket,
                    &h0.t_bra_ket_ket,
                    &h0.t_ket_ket_ket,
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let h4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &h0.q_bbbb,
                    &h0.q_bbbk,
                    &h0.q_bbkk,
                    &h0.q_bkkk,
                    &h0.q_kkkk,
                    centers,
                    axes,
                )
            };

            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let value_of = |i: usize, j: usize, k: usize, l: usize| -> f64 {
                let (ca, axa) = (slot_center(i), i % 3);
                let (cb, axb) = (slot_center(j), j % 3);
                let (cc, axc) = (slot_center(k), k % 3);
                let (cd, axd) = (slot_center(l), l % 3);
                let s_abcd = s4([ca, cb, cc, cd], [axa, axb, axc, axd]);
                let h_abcd = h4([ca, cb, cc, cd], [axa, axb, axc, axd]);
                let (s_a, s_b, s_c, s_d) =
                    (s1(ca, axa), s1(cb, axb), s1(cc, axc), s1(cd, axd));
                let (h_a, h_b, h_c, h_d) =
                    (h1(ca, axa), h1(cb, axb), h1(cc, axc), h1(cd, axd));
                let s_ab = s2(ca, cb, axa, axb);
                let s_ac = s2(ca, cc, axa, axc);
                let s_ad = s2(ca, cd, axa, axd);
                let s_bc = s2(cb, cc, axb, axc);
                let s_bd = s2(cb, cd, axb, axd);
                let s_cd = s2(cc, cd, axc, axd);
                let h_ab = h2(ca, cb, axa, axb);
                let h_ac = h2(ca, cc, axa, axc);
                let h_ad = h2(ca, cd, axa, axd);
                let h_bc = h2(cb, cc, axb, axc);
                let h_bd = h2(cb, cd, axb, axd);
                let h_cd = h2(cc, cd, axc, axd);
                let s_abc = s3(ca, cb, cc, axa, axb, axc);
                let s_abd = s3(ca, cb, cd, axa, axb, axd);
                let s_acd = s3(ca, cc, cd, axa, axc, axd);
                let s_bcd = s3(cb, cc, cd, axb, axc, axd);
                let h_abc = h3(ca, cb, cc, axa, axb, axc);
                let h_abd = h3(ca, cb, cd, axa, axb, axd);
                let h_acd = h3(ca, cc, cd, axa, axc, axd);
                let h_bcd = h3(cb, cc, cd, axb, axc, axd);
                overlap_coeff * s_abcd
                    + two_p
                        * (overlap * h_abcd
                            + (h_abc * s_d + h_abd * s_c + h_acd * s_b + h_bcd * s_a)
                            + (h_ab * s_cd
                                + h_ac * s_bd
                                + h_ad * s_bc
                                + h_bc * s_ad
                                + h_bd * s_ac
                                + h_cd * s_ab)
                            + (h_a * s_bcd + h_b * s_acd + h_c * s_abd + h_d * s_abc))
            };
            let dof = |s: usize| 3 * atom_of(slot_center(s)) + s % 3;
            for i in 0..6 {
                for j in i..6 {
                    for k in j..6 {
                        for l in k..6 {
                            let v = value_of(i, j, k, l);
                            if v == 0.0 {
                                continue;
                            }
                            for &(p1, p2, p3, p4) in &distinct_perms4(i, j, k, l) {
                                tensor[dof(p3)][dof(p4)][(dof(p1), dof(p2))] += v;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

/// A scalar's dense forward-AD jet over the `ndof` Cartesian coordinates carried to **fourth**
/// order: [`DedcnJet`] with the `[ndof⁴]` flat `fourth` array appended, exactly the way `third`
/// extends `hess`. Costs `O(ndof⁴)` per atom, so it is built only by the fourth-derivative path.
pub(crate) struct DedcnJet4 {
    pub(crate) value: f64,
    pub(crate) grad: Vec<f64>,
    pub(crate) hess: Vec<f64>,
    pub(crate) third: Vec<f64>,
    pub(crate) fourth: Vec<f64>,
}

impl DedcnJet4 {
    fn zeros(ndof: usize) -> Self {
        Self {
            value: 0.0,
            grad: vec![0.0; ndof],
            hess: vec![0.0; ndof * ndof],
            third: vec![0.0; ndof * ndof * ndof],
            fourth: vec![0.0; ndof * ndof * ndof * ndof],
        }
    }

    /// The embedded lower orders as a plain [`DedcnJet`], so a consistency test can compare
    /// against the third-order builders field-for-field.
    #[cfg(test)]
    pub(crate) fn to_third(&self) -> DedcnJet {
        DedcnJet {
            value: self.value,
            grad: self.grad.clone(),
            hess: self.hess.clone(),
            third: self.third.clone(),
        }
    }
}

/// [`cn_h0_dedcn_jets`] carried one order up: `∂E/∂CN_A` with its 1st–4th nuclear derivatives at
/// frozen density. Same `scale·overlap` Leibniz, now the 16-term fourth-order product rule fed by
/// [`h0_scale_fourth`] and [`crate::integrals::contracted_pair_with_fourth_derivatives`].
pub(crate) fn cn_h0_dedcn_jets_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<DedcnJet4>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let mut jets: Vec<DedcnJet4> = (0..nat).map(|_| DedcnJet4::zeros(ndof)).collect();

    // On-site diagonal block (R-independent): value only.
    for shell in basis.shells.iter() {
        let dsedcn = -shell.kcn_raw.unwrap_or(0.0);
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            jets[shell.atom_index].value += dsedcn * electronic.density[(iao, iao)];
        }
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
                contracted_pair_with_fourth_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let scale = h0_scale_fourth(system, params, shell_mu_index, shell_nu_index, basis)?;
            let dsedcn_mu = -shell_mu.kcn_raw.unwrap_or(0.0);
            let dsedcn_nu = -shell_nu.kcn_raw.unwrap_or(0.0);

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
            let s3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &pair.t_bra_bra_bra[0],
                    &pair.t_bra_bra_ket[0],
                    &pair.t_bra_ket_ket[0],
                    &pair.t_ket_ket_ket[0],
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let s4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &pair.q_bbbb,
                    &pair.q_bbbk,
                    &pair.q_bbkk,
                    &pair.q_bkkk,
                    &pair.q_kkkk,
                    centers,
                    axes,
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
            let h3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &scale.t_bra_bra_bra,
                    &scale.t_bra_bra_ket,
                    &scale.t_bra_ket_ket,
                    &scale.t_ket_ket_ket,
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let h4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &scale.q_bbbb,
                    &scale.q_bbbk,
                    &scale.q_bbkk,
                    &scale.q_bkkk,
                    &scale.q_kkkk,
                    centers,
                    axes,
                )
            };
            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let dof = |s: usize| 3 * atom_of(slot_center(s)) + s % 3;

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
                        let v3 = scale.value * s3(ci, cj, ck, axi, axj, axk)
                            + h3(ci, cj, ck, axi, axj, axk) * overlap
                            + h2(ci, cj, axi, axj) * s1(ck, axk)
                            + h2(ci, ck, axi, axk) * s1(cj, axj)
                            + h2(cj, ck, axj, axk) * s1(ci, axi)
                            + h1(ci, axi) * s2(cj, ck, axj, axk)
                            + h1(cj, axj) * s2(ci, ck, axi, axk)
                            + h1(ck, axk) * s2(ci, cj, axi, axj);
                        let tidx = (dof(i) * ndof + dof(j)) * ndof + dof(k);
                        jets[atom_mu].third[tidx] += dsedcn_mu * p * v3;
                        jets[atom_nu].third[tidx] += dsedcn_nu * p * v3;
                        for l in 0..6 {
                            let (cl, axl) = (slot_center(l), l % 3);
                            let v4 = scale.value * s4([ci, cj, ck, cl], [axi, axj, axk, axl])
                                + h4([ci, cj, ck, cl], [axi, axj, axk, axl]) * overlap
                                + h3(ci, cj, ck, axi, axj, axk) * s1(cl, axl)
                                + h3(ci, cj, cl, axi, axj, axl) * s1(ck, axk)
                                + h3(ci, ck, cl, axi, axk, axl) * s1(cj, axj)
                                + h3(cj, ck, cl, axj, axk, axl) * s1(ci, axi)
                                + h2(ci, cj, axi, axj) * s2(ck, cl, axk, axl)
                                + h2(ci, ck, axi, axk) * s2(cj, cl, axj, axl)
                                + h2(ci, cl, axi, axl) * s2(cj, ck, axj, axk)
                                + h2(cj, ck, axj, axk) * s2(ci, cl, axi, axl)
                                + h2(cj, cl, axj, axl) * s2(ci, ck, axi, axk)
                                + h2(ck, cl, axk, axl) * s2(ci, cj, axi, axj)
                                + h1(ci, axi) * s3(cj, ck, cl, axj, axk, axl)
                                + h1(cj, axj) * s3(ci, ck, cl, axi, axk, axl)
                                + h1(ck, axk) * s3(ci, cj, cl, axi, axj, axl)
                                + h1(cl, axl) * s3(ci, cj, ck, axi, axj, axk);
                            let qidx = ((dof(i) * ndof + dof(j)) * ndof + dof(k)) * ndof + dof(l);
                            jets[atom_mu].fourth[qidx] += dsedcn_mu * p * v4;
                            jets[atom_nu].fourth[qidx] += dsedcn_nu * p * v4;
                        }
                    }
                }
            }
        }
    }
    Ok(jets)
}

/// [`cn_h0_cn_jets`] carried one order up: `CN_A` with its 1st–4th nuclear derivatives, from the
/// smooth counting function's central radial blocks scattered over each `(i,j)` pair's slots
/// (`σ_i = +1`, `σ_j = −1`). The rank-4 block uses the same hat-coefficient form documented on
/// [`shell_poly_fourth`], fed by [`ScalarDerivatives::fourth`].
pub(crate) fn cn_h0_cn_jets_fourth(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
) -> Result<Vec<DedcnJet4>> {
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
    let mut jets: Vec<DedcnJet4> = (0..nat).map(|_| DedcnJet4::zeros(ndof)).collect();
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
        let a2 = g / r;
        let a3 = coeff_uuu / r;
        let a4 = d.fourth - 6.0 * d.third / r + 15.0 * d.second / (r * r)
            - 15.0 * d.first / (r * r * r);
        let delta = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
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
                    let dab = delta(a, b);
                    let hrel = d.second * u[a] * u[b] + (d.first / r) * (dab - u[a] * u[b]);
                    jet.hess[dof(s) * ndof + dof(t)] += sigma(s) * sigma(t) * hrel;
                    for q in 0..6 {
                        let c = q % 3;
                        let (dac, dbc) = (delta(a, c), delta(b, c));
                        let kron = dab * u[c] + dac * u[b] + dbc * u[a];
                        let trel = coeff_uuu * u[a] * u[b] * u[c] + g * kron;
                        let idx = (dof(s) * ndof + dof(t)) * ndof + dof(q);
                        jet.third[idx] += sigma(s) * sigma(t) * sigma(q) * trel;
                        for w in 0..6 {
                            let e = w % 3;
                            let (dae, dbe, dce) = (delta(a, e), delta(b, e), delta(c, e));
                            let qrel = a4 * u[a] * u[b] * u[c] * u[e]
                                + a3 * (dab * u[c] * u[e]
                                    + dac * u[b] * u[e]
                                    + dae * u[b] * u[c]
                                    + dbc * u[a] * u[e]
                                    + dbe * u[a] * u[c]
                                    + dce * u[a] * u[b])
                                + a2 * (dab * dce + dac * dbe + dae * dbc);
                            let qidx = idx * ndof + dof(w);
                            jet.fourth[qidx] +=
                                sigma(s) * sigma(t) * sigma(q) * sigma(w) * qrel;
                        }
                    }
                }
            }
        }
    }
    Ok(jets)
}

/// **CN-H0 frozen fourth derivative**, `out[a][d][(b,c)] = ∂_d ( third[a][(b,c)] )` with `third` =
/// [`fixed_density_cn_h0_third_derivative`].
///
/// That third-order slab is the `∂_a` derivative of the CN-H0 Hessian **block**
/// `H_bc = CN_bc·de + CN_b·de_c + de_b·CN_c` — which deliberately omits `CN·de_bc` (the band/Pulay
/// block carries it with the converged self-energy held fixed). This function follows that
/// decomposition exactly one order up: `∂_d` of each of the six third-order terms, giving the
/// twelve products below. It is therefore **not** the physically complete quartic CN-H0 term and
/// **not** index-symmetric — by construction, since its only contract is being the exact `∂_d` of
/// the existing third-order object.
///
/// ```text
///   ∂_d(CN_abc·de ) = CN_abcd·de  + CN_abc·de_d
///   ∂_d(CN_bc ·de_a) = CN_bcd·de_a + CN_bc ·de_ad
///   ∂_d(CN_ab ·de_c) = CN_abd·de_c + CN_ab ·de_cd
///   ∂_d(CN_b  ·de_ac) = CN_bd·de_ac + CN_b ·de_acd
///   ∂_d(de_ab ·CN_c) = de_abd·CN_c + de_ab ·CN_cd
///   ∂_d(de_b  ·CN_ac) = de_bd·CN_ac + de_b ·CN_acd
/// ```
pub fn fixed_density_cn_h0_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
) -> Result<Vec<Vec<Matrix>>> {
    ensure_non_pbc(system)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let de = cn_h0_dedcn_jets_fourth(system, params, electronic)?;
    let cn = cn_h0_cn_jets_fourth(system, coordination_cutoff)?;
    let mut tensor = vec![vec![Matrix::zeros(ndof, ndof); ndof]; ndof];
    let i3 = |x: usize, y: usize, z: usize| (x * ndof + y) * ndof + z;
    let i4 = |x: usize, y: usize, z: usize, w: usize| ((x * ndof + y) * ndof + z) * ndof + w;
    for a in 0..ndof {
        for d in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    let mut t = 0.0;
                    for atom in 0..nat {
                        let (n, e) = (&cn[atom], &de[atom]);
                        t += e.value * n.fourth[i4(a, b, c, d)]
                            + n.third[i3(a, b, c)] * e.grad[d]
                            + n.third[i3(b, c, d)] * e.grad[a]
                            + n.hess[b * ndof + c] * e.hess[a * ndof + d]
                            + n.third[i3(a, b, d)] * e.grad[c]
                            + n.hess[a * ndof + b] * e.hess[c * ndof + d]
                            + n.hess[b * ndof + d] * e.hess[a * ndof + c]
                            + n.grad[b] * e.third[i3(a, c, d)]
                            + e.third[i3(a, b, d)] * n.grad[c]
                            + e.hess[a * ndof + b] * n.hess[c * ndof + d]
                            + e.hess[b * ndof + d] * n.hess[a * ndof + c]
                            + e.grad[b] * n.third[i3(a, c, d)];
                    }
                    tensor[a][d][(b, c)] = t;
                }
            }
        }
    }
    Ok(tensor)
}

/// Geometric **fourth** derivative (fixed density) of the SCC-scalar-potential × overlap-derivative
/// coupling block, `out[c][d][(a,b)] = ∂_d ( third[c][(a,b)] )` with `third` =
/// [`fixed_density_scalar_overlap_third_derivative`].
///
/// The third-order slab is `T_abc = −Σ_pairs p·( ds_a·dscalar_bc + dds_ac·dscalar_b )`; applying
/// `∂_d` at fixed density gives the four terms
///
/// ```text
///   Q_abcd = −Σ_pairs p · ( dds_ad·dscalar_bc + ds_a ·dscalar_bcd
///                         + ddds_acd·dscalar_b + dds_ac·dscalar_bd )
/// ```
///
/// where `a` (and now also `c`, `d` inside the overlap factors) must sit on the pair's bra/ket
/// centres for the overlap derivative to be non-zero, while `b`, `c`, `d` range over all DOFs in
/// the scalar-potential factors. `dscalar_bcd` is the new
/// [`shell_scalar_potential_third_derivatives`] ladder rung; `ddds_acd` is the third overlap
/// derivative already available from [`crate::integrals::contracted_pair_with_third_derivatives`].
/// Like the third-order block this is an ORDERED object (the `(c,d)` slots are the two
/// displacement directions, `(a,b)` the Hessian row/col) and is not symmetrised here.
pub fn fixed_density_scalar_overlap_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Vec<Vec<Matrix>>> {
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
    let dscalar3 =
        shell_scalar_potential_third_derivatives(system, basis, &electronic.shell_charges, params)?;
    let mut tensor = vec![vec![Matrix::zeros(ndof, ndof); ndof]; ndof];
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
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let hbb = &pair.h_bra_bra[0];
            let hbk = &pair.h_bra_ket[0];
            let hkk = &pair.h_ket_ket[0];
            let d_bra = pair.d_bra[0];
            let d_ket = pair.d_ket[0];
            let center_of = |dofx: usize| {
                let atom = dofx / 3;
                if atom == atom_mu {
                    Some(Center::Bra)
                } else if atom == atom_nu {
                    Some(Center::Ket)
                } else {
                    None
                }
            };
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
                            let c_center = center_of(c);
                            let dds_ac = c_center.map(|cc| {
                                second(hbb, hbk, hkk, row_center, cc, row_axis, c % 3)
                            });
                            for d in 0..ndof {
                                let dscalar_bd =
                                    dscalar2[shell_mu][(b, d)] + dscalar2[shell_nu][(b, d)];
                                let idx3 = (b * ndof + c) * ndof + d;
                                let dscalar_bcd = dscalar3[shell_mu][idx3] + dscalar3[shell_nu][idx3];
                                let d_center = center_of(d);
                                let mut t = ds_row * dscalar_bcd;
                                if let Some(dc) = d_center {
                                    t += second(hbb, hbk, hkk, row_center, dc, row_axis, d % 3)
                                        * dscalar_bc;
                                }
                                if let Some(ac) = dds_ac {
                                    t += ac * dscalar_bd;
                                }
                                if let (Some(cc), Some(dc)) = (c_center, d_center) {
                                    let ddds = third_select(
                                        &pair.t_bra_bra_bra[0],
                                        &pair.t_bra_bra_ket[0],
                                        &pair.t_bra_ket_ket[0],
                                        &pair.t_ket_ket_ket[0],
                                        [row_center, cc, dc],
                                        [row_axis, c % 3, d % 3],
                                    );
                                    t += ddds * dscalar_b;
                                }
                                tensor[c][d][(a, b)] -= p * t;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

// =============================================================================================
// Directional THIRD-derivative AO-matrix builders (quartic response stage)
//
// Each builder below is the one-order-up, DIRECTION-CONTRACTED twin of an existing
// AO-matrix-valued SECOND-derivative skeleton object: for a fixed direction `v` (length
// `3·nat`) they return the n×n matrix
//
//     `M3_μν = Σ_bcd v_b v_c v_d ∂³X_μν/∂R_b∂R_c∂R_d`,
//
// for `X ∈ { bare H0 at frozen CN, the CN-coupling completion, the −½(vS) SCC-scalar block,
// the overlap S }`. All three derivative legs are contracted with the SAME `v` (there is no
// per-triple API), which collapses the `2³` bra/ket centre assignments of a two-centre pair
// integral into the four distinct patterns with multiplicities `1, 3, 3, 1` — see
// [`directional_third`]. The `v`-components of atoms that are not one of the pair's two
// centres contract to zero, which is exactly the `atom_b ∈ {atom_μ, atom_ν}` screen that the
// per-DOF builders apply explicitly.
//
// The electronic reference (`P`, `q`, the shell potential, the cached coordination numbers)
// is FROZEN in all of these, exactly as it is in the second-derivative twins: each builder is
// the *total geometric* derivative of its second-order twin at a fixed electronic reference,
// which is what the FD gates in `tests.rs` check.
// =============================================================================================

/// Length check shared by the directional third-derivative builders.
fn ensure_direction_len(system: &PeriodicSystem, v: &[f64]) -> Result<()> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "direction vector length {} does not match the {ndof} nuclear degrees of freedom",
            v.len()
        )));
    }
    Ok(())
}

/// The two Cartesian sub-vectors of `v` that a two-centre AO pair can see: `(v_bra, v_ket)`.
#[inline]
fn direction_on_pair(v: &[f64], atom_bra: usize, atom_ket: usize) -> ([f64; 3], [f64; 3]) {
    let pick = |a: usize| [v[3 * a], v[3 * a + 1], v[3 * a + 2]];
    (pick(atom_bra), pick(atom_ket))
}

/// `Σ_b v_b ∂f/∂R_b` for a two-centre quantity — the `2¹` centre assignments of
/// [`first_vec`], weighted by the direction components of the owning atoms.
#[inline]
fn directional_first(d_bra: Vec3, d_ket: Vec3, va: &[f64; 3], vk: &[f64; 3]) -> f64 {
    let (b, k) = (d_bra.to_array(), d_ket.to_array());
    let mut out = 0.0;
    for a in 0..3 {
        out += va[a] * b[a] + vk[a] * k[a];
    }
    out
}

/// `Σ_bc v_b v_c ∂²f/∂R_b∂R_c` — the `2²` centre assignments of [`second`]. The mixed
/// `(bra,ket)`/`(ket,bra)` pair contributes `2·Σ_ab va_a vk_b h_bra_ket[a][b]` (the second
/// assignment is the index relabelling of the first).
#[inline]
fn directional_second(
    h_bra_bra: &[[f64; 3]; 3],
    h_bra_ket: &[[f64; 3]; 3],
    h_ket_ket: &[[f64; 3]; 3],
    va: &[f64; 3],
    vk: &[f64; 3],
) -> f64 {
    let mut out = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            out += va[a] * va[b] * h_bra_bra[a][b]
                + 2.0 * va[a] * vk[b] * h_bra_ket[a][b]
                + vk[a] * vk[b] * h_ket_ket[a][b];
        }
    }
    out
}

/// `Σ_bcd v_b v_c v_d ∂³f/∂R_b∂R_c∂R_d` — the `2³` centre assignments of [`third_select`]
/// (ket indices last in `t_bra_bra_ket`/`t_bra_ket_ket`) collapsed to the four distinct
/// bra/ket patterns with multiplicities `1, 3, 3, 1` (the number of ways to pick which of the
/// three identical legs sit on the ket centre).
#[inline]
fn directional_third(
    t_bbb: &Ten3,
    t_bbk: &Ten3,
    t_bkk: &Ten3,
    t_kkk: &Ten3,
    va: &[f64; 3],
    vk: &[f64; 3],
) -> f64 {
    let mut out = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                out += va[a] * va[b] * va[c] * t_bbb[a][b][c]
                    + 3.0 * va[a] * va[b] * vk[c] * t_bbk[a][b][c]
                    + 3.0 * va[a] * vk[b] * vk[c] * t_bkk[a][b][c]
                    + vk[a] * vk[b] * vk[c] * t_kkk[a][b][c];
            }
        }
    }
    out
}

/// Directional SECOND nuclear derivative of the bare (frozen-CN) H0 AO matrix:
/// `M2_μν = Σ_bc v_b v_c ∂²H0_μν/∂R_b∂R_c` — the ONE-PASS twin of
/// [`h0_bare_second_derivative_matrix`].
///
/// The per-pair second-derivative data (the `h0_scale_second` ladder and the
/// `contracted_pair_with_second_derivatives` overlap ladder) does not depend on
/// `(b, c)` at all — only the contraction weights do. Contracting the two legs
/// against `v` INSIDE the pair sweep therefore replaces `ndof²` matrix builds
/// (each re-evaluating the same pair integrals) with a single one, which is what
/// makes the directional skeleton seconds of
/// [`crate::fourth_derivative::assemble::directional_second_order_legs`]
/// affordable.
///
/// With `H0_μν = self_avg·scale·S_μν` at the cached (frozen) coordination
/// numbers — `self_avg`'s motion is the CN block,
/// [`directional_h0_cn_block_second_matrix`] — the second-order product Leibniz
/// of the two geometric factors is
///
/// ```text
///   M2 = self_avg·( scale₂·S + 2·scale₁·S₁ + scale·S₂ )
/// ```
///
/// with the subscript the directional derivative order. The per-DOF version's
/// `(atom_b, atom_c) ⊆ {atom_μ, atom_ν}` screening is automatic here: a leg can
/// only land on one of the pair's two centres, which is exactly what
/// [`direction_on_pair`] picks out.
pub(crate) fn directional_h0_bare_second_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
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
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            if va == [0.0; 3] && vk == [0.0; 3] {
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
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let g1 = directional_first(scale.d_bra, scale.d_ket, &va, &vk);
            let g2 = directional_second(
                &scale.h_bra_bra,
                &scale.h_bra_ket,
                &scale.h_ket_ket,
                &va,
                &vk,
            );
            out[(mu, nu)] = self_avg * (g2 * s0 + 2.0 * g1 * s1 + scale.value * s2);
        }
    }
    Ok(out)
}

/// Per-atom directional coordination-number derivatives `(CN¹, CN²)` along `v`,
/// contracted from the SAME [`coordination_number_first_derivatives`] /
/// [`coordination_number_second_derivatives`] tables the per-DOF CN block reads
/// — so the one-pass CN block agrees with the double loop to rounding, not just
/// to the jets' tolerance.
fn directional_cn_derivatives_second(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let cn1 = coordination_number_first_derivatives(system, coordination_cutoff)?;
    let cn2 = coordination_number_second_derivatives(system, coordination_cutoff)?;
    let mut d1 = vec![0.0; nat];
    let mut d2 = vec![0.0; nat];
    for atom in 0..nat {
        let (mut a1, mut a2) = (0.0, 0.0);
        for (b, &vb) in v.iter().enumerate().take(ndof) {
            if vb == 0.0 {
                continue;
            }
            a1 += vb * cn1[(atom, b)];
            for (c, &vc) in v.iter().enumerate().take(ndof) {
                a2 += vb * vc * cn2[atom][(b, c)];
            }
        }
        d1[atom] = a1;
        d2[atom] = a2;
    }
    Ok((d1, d2))
}

/// Directional SECOND derivative of the CN-coupling completion — the ONE-PASS
/// twin of [`h0_cn_block_second_derivative_matrix`].
///
/// That per-DOF builder is the most expensive leg of the `ndof²` double loop: it
/// screens nothing (every AO pair is visited for every `(b, c)`) and it rebuilds
/// the many-body coordination-number first AND second derivative tables on every
/// call. Here both are built once.
///
/// Write the CN-dependent piece of the H0 prefactor as
/// `c(R) = −½(kcn_μ·CN_μ + kcn_ν·CN_ν)` and the pure geometric factor as
/// `P = scale·S`. The per-DOF matrix is the Leibniz remainder
/// `∂_b∂_c(c·P) − c·∂_b∂_c P` = `c_c·P_b + c_b·P_c + c_bc·P`, so directionally
///
/// ```text
///   M2 = c₂·P + 2·c₁·P₁ ,   P₁ = scale₁·S + scale·S₁ .
/// ```
///
/// The on-site block keeps its rigid-overlap form `M2 = c₂·S_μν` with `S` the
/// actual (frozen) SCF overlap — same-atom overlap is geometry-rigid, so every
/// `P` derivative vanishes there.
pub(crate) fn directional_h0_cn_block_second_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
    let (cn1, cn2) = directional_cn_derivatives_second(system, coordination_cutoff, v)?;
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            let kcn_mu = basis.shells[shell_mu].kcn_raw.unwrap_or(0.0);
            let kcn_nu = basis.shells[shell_nu].kcn_raw.unwrap_or(0.0);
            if kcn_mu == 0.0 && kcn_nu == 0.0 {
                continue;
            }
            let c1 = -0.5 * (kcn_mu * cn1[atom_mu] + kcn_nu * cn1[atom_nu]);
            let c2 = -0.5 * (kcn_mu * cn2[atom_mu] + kcn_nu * cn2[atom_nu]);
            if atom_mu == atom_nu {
                out[(mu, nu)] = c2 * electronic.integrals.overlap[(mu, nu)];
                continue;
            }
            let rnu = system.atoms[atom_nu].position;
            let scale = h0_scale_second(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let g1 = directional_first(scale.d_bra, scale.d_ket, &va, &vk);
            let p0 = scale.value * s0;
            let p1 = g1 * s0 + scale.value * s1;
            out[(mu, nu)] = c2 * p0 + 2.0 * c1 * p1;
        }
    }
    Ok(out)
}

/// Directional SECOND derivative of the SCC-scalar block — the ONE-PASS twin of
/// [`h0_scc_scalar_second_derivative_matrix`], with the response-carrying legs
/// supplied by the caller in already-directional form:
///
/// * `v_c[s]` — the directional derivative of the shell potential that the
///   per-DOF builder receives per `(b, c)` as its `v_c` column,
/// * `q_c[s]` — the directional first-order shell-charge response.
///
/// The per-DOF builder rebuilds the whole `∂V/∂R` and `∂²V/∂R²` shell-potential
/// ladders on every one of the `ndof²` calls; here they are built once and
/// contracted against `v` up front.
///
/// With `Φ = ½(V_μ+V_ν)` frozen at the electronic reference, `A = ½(v_c_μ+v_c_ν)`,
/// `S₀₁₂` the overlap ladder, and the geometric γ-chain shifts
/// `G = ½(∂V/∂R|_q·v)`, `G' = ½(∂²V/∂R²|_q:vv)`, the four-term output is
///
/// ```text
///   M2 = −A·S₁ − Φ·S₂ − B·S₀ − G·S₁ ,   B = G' + ½(∂V/∂R|_{q_c}·v) ,
/// ```
///
/// exactly the object [`directional_h0_scc_scalar_third_matrix`] documents as
/// the order it differentiates. The same-atom (incl. on-site diagonal) block
/// keeps the rigid-overlap form `M2 = −B·S_μν` with `S` the actual SCF overlap.
pub(crate) fn directional_h0_scc_scalar_second_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
    v_c: &[f64],
    q_c: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
    let nsh = basis.shells.len();
    for (name, leg) in [("v_c", v_c), ("q_c", q_c)] {
        if leg.len() != nsh {
            return Err(Gfn1Error::InvalidInput(format!(
                "directional SCC-scalar leg `{name}` has length {} but the basis has {nsh} shells",
                leg.len()
            )));
        }
    }
    let v_shell = &electronic.shell_scc_potential;
    let dvdr_q =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let d2vdr_q = shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let dvdr_qc = shell_scalar_potential_first_derivatives(system, basis, q_c, params)?;
    // Directional contractions of the γ-chain ladders, per shell.
    let mut g1 = vec![0.0; nsh]; // ∂V/∂R|_q · v
    let mut g2 = vec![0.0; nsh]; // ∂²V/∂R²|_q : vv
    let mut h1 = vec![0.0; nsh]; // ∂V/∂R|_{q_c} · v
    for s in 0..nsh {
        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let mut b1 = 0.0;
        for (b, &vb) in v.iter().enumerate() {
            if vb == 0.0 {
                continue;
            }
            a1 += vb * dvdr_q[(s, b)];
            b1 += vb * dvdr_qc[(s, b)];
            for (c, &vc) in v.iter().enumerate() {
                a2 += vb * vc * d2vdr_q[s][(b, c)];
            }
        }
        g1[s] = a1;
        g2[s] = a2;
        h1[s] = b1;
    }
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            let avg = |x: &[f64]| 0.5 * (x[shell_mu] + x[shell_nu]);
            // `B` = the directional `D_c dscalar_b`: geometric second + charge path.
            let b_shift = avg(&g2) + avg(&h1);
            if atom_mu == atom_nu {
                // SAME-ATOM: the overlap is geometry-rigid, so `−B·S_μν` is all
                // that survives (S from the actual SCF overlap, not the broken
                // zero-separation pair moment).
                let ov = electronic.integrals.overlap[(mu, nu)];
                if ov.abs() > 1.0e-30 {
                    out[(mu, nu)] = -b_shift * ov;
                }
                continue;
            }
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let phi = avg(v_shell);
            let a_leg = avg(v_c);
            let g_shift = avg(&g1);
            out[(mu, nu)] = -a_leg * s1 - phi * s2 - b_shift * s0 - g_shift * s1;
        }
    }
    Ok(out)
}

/// Directional SECOND nuclear derivative of the AO overlap matrix,
/// `M2_μν = Σ_bc v_b v_c ∂²S_μν/∂R_b∂R_c` — the ONE-PASS twin of
/// [`crate::response::cpxtb::overlap_second_derivative_matrix`], built straight
/// from the per-pair second-order centre patterns. Same-centre pairs are kept
/// (not screened) exactly as in the per-DOF version, whose four centre
/// assignments then cancel by translational invariance.
pub(crate) fn directional_overlap_second_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    v: &[f64],
) -> Result<Matrix> {
    ensure_direction_len(system, v)?;
    let n = basis.len();
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let bra_atom = basis.aos[mu].atom_index;
        let ra = system.atoms[bra_atom].position;
        for nu in 0..n {
            let ket_atom = basis.aos[nu].atom_index;
            let (va, vk) = direction_on_pair(v, bra_atom, ket_atom);
            if va == [0.0; 3] && vk == [0.0; 3] {
                continue;
            }
            let rk = system.atoms[ket_atom].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rk);
            out[(mu, nu)] = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
        }
    }
    Ok(out)
}

/// Directional THIRD nuclear derivative of the bare (frozen-CN) H0 AO matrix:
/// `M3_μν = Σ_bcd v_b v_c v_d ∂³H0_μν/∂R_b∂R_c∂R_d`, i.e.
/// [`h0_bare_second_derivative_matrix`] one order up with all legs contracted against `v`.
///
/// `H0_μν = self_avg·scale·S_μν` with `self_avg = ½(self_μ+self_ν)` read from the **cached**
/// `electronic.coordination_numbers` (a geometry constant here — its motion is the CN block,
/// [`directional_h0_cn_block_third_matrix`]). So this is `self_avg` times the third-order
/// product Leibniz of the two geometric factors,
///
/// ```text
///   M3 = self_avg·( scale₃·S + 3·scale₂·S₁ + 3·scale₁·S₂ + scale·S₃ )
/// ```
///
/// with the subscript the directional derivative order: the geometric scale ladder from
/// [`h0_scale_third`] and the overlap ladder from `contracted_pair_with_third_derivatives`,
/// each contracted over the pair's two centres.
pub(crate) fn directional_h0_bare_third_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
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
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            if va == [0.0; 3] && vk == [0.0; 3] {
                continue;
            }
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let self_avg = 0.5
                * (shell_self_energy(&basis.shells[shell_mu], cn[atom_mu])
                    + shell_self_energy(&basis.shells[shell_nu], cn[atom_nu]));
            let scale = h0_scale_third(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let s3 = directional_third(
                &pair.t_bra_bra_bra[0],
                &pair.t_bra_bra_ket[0],
                &pair.t_bra_ket_ket[0],
                &pair.t_ket_ket_ket[0],
                &va,
                &vk,
            );
            let g1 = directional_first(scale.d_bra, scale.d_ket, &va, &vk);
            let g2 = directional_second(
                &scale.h_bra_bra,
                &scale.h_bra_ket,
                &scale.h_ket_ket,
                &va,
                &vk,
            );
            let g3 = directional_third(
                &scale.t_bra_bra_bra,
                &scale.t_bra_bra_ket,
                &scale.t_bra_ket_ket,
                &scale.t_ket_ket_ket,
                &va,
                &vk,
            );
            out[(mu, nu)] =
                self_avg * (g3 * s0 + 3.0 * g2 * s1 + 3.0 * g1 * s2 + scale.value * s3);
        }
    }
    Ok(out)
}

/// Per-atom directional coordination-number derivatives `(CN¹, CN², CN³)` along `v`, i.e.
/// `CNⁿ_A = Σ v…v ∂ⁿCN_A/∂R…∂R`, contracted from the [`cn_h0_cn_jets`] tensors. The jets'
/// `grad`/`hess` are bit-equivalent to [`coordination_number_first_derivatives`] /
/// [`coordination_number_second_derivatives`] (same per-pair radial ladder, same `σ` sign
/// law), so the directional first/second orders agree with what the per-DOF CN-block second
/// derivative reads.
fn directional_cn_derivatives(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let jets = cn_h0_cn_jets(system, coordination_cutoff)?;
    let mut cn1 = vec![0.0; nat];
    let mut cn2 = vec![0.0; nat];
    let mut cn3 = vec![0.0; nat];
    for atom in 0..nat {
        let jet = &jets[atom];
        let (mut d1, mut d2, mut d3) = (0.0, 0.0, 0.0);
        for b in 0..ndof {
            let vb = v[b];
            if vb == 0.0 {
                continue;
            }
            d1 += vb * jet.grad[b];
            for c in 0..ndof {
                let vc = v[c];
                if vc == 0.0 {
                    continue;
                }
                d2 += vb * vc * jet.hess[b * ndof + c];
                let base = (b * ndof + c) * ndof;
                let mut inner = 0.0;
                for (d, &vd) in v.iter().enumerate() {
                    inner += vd * jet.third[base + d];
                }
                d3 += vb * vc * inner;
            }
        }
        cn1[atom] = d1;
        cn2[atom] = d2;
        cn3[atom] = d3;
    }
    Ok((cn1, cn2, cn3))
}

/// Directional THIRD derivative of the CN-coupling completion,
/// [`h0_cn_block_second_derivative_matrix`] one order up with all legs contracted against `v`.
///
/// Write the CN-dependent piece of the H0 prefactor as `c(R) = −½(kcn_μ·CN_μ + kcn_ν·CN_ν)`
/// (so `self_avg = const + c`) and the pure geometric factor as `P = scale·S`. The per-DOF
/// second matrix is exactly the Leibniz remainder `∂_b∂_c(c·P) − c·∂_b∂_c P`, i.e. its
/// documented Part A (`c_c·P_b`) plus Part B (`c_bc·P + c_b·P_c`); directionally that is
///
/// ```text
///   M2 = c₂·P + 2·c₁·P₁ .
/// ```
///
/// Differentiating that 2-part structure once more along `v` — the reference `self_avg`/`c`
/// VALUE never enters, only its derivatives, and the frozen twin
/// [`directional_h0_bare_third_matrix`] keeps the cached `self_avg` fixed — gives
///
/// ```text
///   M3 = c₃·P + 3·c₂·P₁ + 2·c₁·P₂ ,
/// ```
///
/// with `c₁ ₂ ₃` from the directional CN jets ([`directional_cn_derivatives`]) and
/// `P₁ = scale₁·S + scale·S₁`, `P₂ = scale₂·S + 2·scale₁·S₁ + scale·S₂` from the third-order
/// scale ([`h0_scale_third`]) and overlap ladders. The on-site block keeps its rigid-overlap
/// form: `M3 = c₃·S_μν` with `S` the actual (frozen) SCF overlap.
pub(crate) fn directional_h0_cn_block_third_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
    let (cn1, cn2, cn3) = directional_cn_derivatives(system, coordination_cutoff, v)?;
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
            // c(R) = −½(kcn_μ·CN_μ + kcn_ν·CN_ν): the CN-dependent piece of the prefactor.
            let c1 = -0.5 * (kcn_mu * cn1[atom_mu] + kcn_nu * cn1[atom_nu]);
            let c2 = -0.5 * (kcn_mu * cn2[atom_mu] + kcn_nu * cn2[atom_nu]);
            let c3 = -0.5 * (kcn_mu * cn3[atom_mu] + kcn_nu * cn3[atom_nu]);
            if atom_mu == atom_nu {
                // ON-SITE: the overlap is geometry-rigid, so every `P` derivative vanishes and
                // only `c₃·S` survives (`S` = the actual SCF overlap, as in the second version).
                out[(mu, nu)] = c3 * electronic.integrals.overlap[(mu, nu)];
                continue;
            }
            let scale = h0_scale_third(system, params, shell_mu, shell_nu, basis)?;
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let g1 = directional_first(scale.d_bra, scale.d_ket, &va, &vk);
            let g2 = directional_second(
                &scale.h_bra_bra,
                &scale.h_bra_ket,
                &scale.h_ket_ket,
                &va,
                &vk,
            );
            // P = scale·S and its first/second directional derivatives (Leibniz).
            let p0 = scale.value * s0;
            let p1 = g1 * s0 + scale.value * s1;
            let p2 = g2 * s0 + 2.0 * g1 * s1 + scale.value * s2;
            out[(mu, nu)] = c3 * p0 + 3.0 * c2 * p1 + 2.0 * c1 * p2;
        }
    }
    Ok(out)
}

/// Directional THIRD derivative of the SCC-scalar block,
/// [`h0_scc_scalar_second_derivative_matrix`] one order up, with the response-carrying legs
/// supplied by the caller in already-directional form:
///
/// * `v_c[s]` — the first directional derivative of the shell potential (the **TOTAL**
///   `Σ_c v_c dV_s/dR_c = ∂V_s/∂R|_q·v + (E_qq·q^v)_s`),
/// * `v_cc[s]` — its second directional derivative,
/// * `q_c[s]`, `q_cc[s]` — the first/second directional shell-charge responses.
///
/// With `Φ = ½(V_μ+V_ν)` (**frozen** at the electronic reference, as in the second version),
/// `A = ½(v_c_μ+v_c_ν)`, `A' = ½(v_cc_μ+v_cc_ν)`, `S₀₁₂₃` the overlap ladder, and the
/// geometric γ-chain shifts
/// `G = ½(∂V/∂R|_q·v)`, `G' = ½(∂²V/∂R²|_q:vv)`, `G'' = ½(∂³V/∂R³|_q⋮vvv)` from
/// [`shell_scalar_potential_first_derivatives`] /
/// [`shell_scalar_potential_second_derivatives`] /
/// [`shell_scalar_potential_third_derivatives`], the second version's four-term output is
/// `M2 = −A·S₁ − Φ·S₂ − B·S₀ − G·S₁` with `B = G' + ½(∂V/∂R|_{q_c}·v)` its `D_c dscalar_b`.
/// One more directional derivative (Leibniz; `Φ` frozen, `A → A'`, `q_c → q_cc`) gives the
/// seven-term
///
/// ```text
///   M3 = −A'·S₁ − A·S₂ − Φ·S₃ − B'·S₀ − B·S₁ − G'·S₁ − G·S₂ ,
///   B' = G'' + ½(∂²V/∂R²|_{q_c}:vv) + ½(∂V/∂R|_{q_cc}·v) .
/// ```
///
/// The same-atom (incl. on-site diagonal) block keeps the rigid-overlap form `M3 = −B'·S_μν`
/// with `S` the actual SCF overlap. `M3` is affine in the four supplied legs, which is the
/// cheap linearity gate the tests exercise alongside the pure-geometric FD gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_h0_scc_scalar_third_matrix(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
    v_c: &[f64],
    v_cc: &[f64],
    q_c: &[f64],
    q_cc: &[f64],
) -> Result<Matrix> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let n = basis.len();
    let nsh = basis.shells.len();
    let ndof = 3 * system.atoms.len();
    for (name, leg) in [
        ("v_c", v_c),
        ("v_cc", v_cc),
        ("q_c", q_c),
        ("q_cc", q_cc),
    ] {
        if leg.len() != nsh {
            return Err(Gfn1Error::InvalidInput(format!(
                "directional SCC-scalar leg `{name}` has length {} but the basis has {nsh} shells",
                leg.len()
            )));
        }
    }
    let v_shell = &electronic.shell_scc_potential;
    // Geometric γ-chain ladders at the FROZEN reference charges, and the charge-path legs.
    let dvdr_q =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let d2vdr_q = shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let d3vdr_q =
        shell_scalar_potential_third_derivatives(system, basis, &electronic.shell_charges, params)?;
    let dvdr_qc = shell_scalar_potential_first_derivatives(system, basis, q_c, params)?;
    let d2vdr_qc = shell_scalar_potential_second_derivatives(system, basis, q_c, params)?;
    let dvdr_qcc = shell_scalar_potential_first_derivatives(system, basis, q_cc, params)?;
    // Directional contractions, per shell.
    let contract1 = |m: &Matrix, s: usize| -> f64 {
        let mut acc = 0.0;
        for (b, &vb) in v.iter().enumerate() {
            acc += vb * m[(s, b)];
        }
        acc
    };
    let contract2 = |blocks: &[Matrix], s: usize| -> f64 {
        let mut acc = 0.0;
        for (b, &vb) in v.iter().enumerate() {
            if vb == 0.0 {
                continue;
            }
            for (c, &vc) in v.iter().enumerate() {
                acc += vb * vc * blocks[s][(b, c)];
            }
        }
        acc
    };
    let contract3 = |flat: &[Vec<f64>], s: usize| -> f64 {
        let mut acc = 0.0;
        for (b, &vb) in v.iter().enumerate() {
            if vb == 0.0 {
                continue;
            }
            for (c, &vc) in v.iter().enumerate() {
                if vc == 0.0 {
                    continue;
                }
                let base = (b * ndof + c) * ndof;
                let mut inner = 0.0;
                for (d, &vd) in v.iter().enumerate() {
                    inner += vd * flat[s][base + d];
                }
                acc += vb * vc * inner;
            }
        }
        acc
    };
    let mut g1 = vec![0.0; nsh]; // ½-free: per-shell ∂V/∂R|_q·v
    let mut g2 = vec![0.0; nsh]; // ∂²V/∂R²|_q : vv
    let mut g3 = vec![0.0; nsh]; // ∂³V/∂R³|_q ⋮ vvv
    let mut h1 = vec![0.0; nsh]; // ∂V/∂R|_{q_c}·v
    let mut h2 = vec![0.0; nsh]; // ∂²V/∂R²|_{q_c} : vv
    let mut k1 = vec![0.0; nsh]; // ∂V/∂R|_{q_cc}·v
    for s in 0..nsh {
        g1[s] = contract1(&dvdr_q, s);
        g2[s] = contract2(&d2vdr_q, s);
        g3[s] = contract3(&d3vdr_q, s);
        h1[s] = contract1(&dvdr_qc, s);
        h2[s] = contract2(&d2vdr_qc, s);
        k1[s] = contract1(&dvdr_qcc, s);
    }
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..n {
            let atom_nu = basis.aos[nu].atom_index;
            let shell_nu = basis.aos[nu].shell_index;
            let avg = |x: &[f64]| 0.5 * (x[shell_mu] + x[shell_nu]);
            // `B'` = D(dc_dscalar_b): geometric third + the two charge-path legs.
            let b_prime = avg(&g3) + avg(&h2) + avg(&k1);
            if atom_mu == atom_nu {
                // SAME-ATOM: the overlap is geometry-rigid, so `−B'·S_μν` is all that survives
                // (S from the actual SCF overlap, not the zero-separation pair moment).
                let ov = electronic.integrals.overlap[(mu, nu)];
                if ov.abs() > 1.0e-30 {
                    out[(mu, nu)] = -b_prime * ov;
                }
                continue;
            }
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            let rnu = system.atoms[atom_nu].position;
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let s0 = pair.moments[0];
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let s3 = directional_third(
                &pair.t_bra_bra_bra[0],
                &pair.t_bra_bra_ket[0],
                &pair.t_bra_ket_ket[0],
                &pair.t_ket_ket_ket[0],
                &va,
                &vk,
            );
            let phi = avg(v_shell);
            let a_leg = avg(v_c);
            let a_leg_prime = avg(v_cc);
            let g_shift = avg(&g1);
            let g_shift_prime = avg(&g2);
            let b_shift = g_shift_prime + avg(&h1);
            out[(mu, nu)] = -a_leg_prime * s1
                - a_leg * s2
                - phi * s3
                - b_prime * s0
                - b_shift * s1
                - g_shift_prime * s1
                - g_shift * s2;
        }
    }
    Ok(out)
}

/// Directional THIRD nuclear derivative of the AO overlap matrix,
/// `M3_μν = Σ_bcd v_b v_c v_d ∂³S_μν/∂R_b∂R_c∂R_d` — the one-order-up twin of
/// [`crate::response::cpxtb::overlap_second_derivative_matrix`], built straight from the
/// per-pair third-order centre patterns. Same-centre pairs are kept (not screened) exactly as
/// in the second version: their four patterns cancel by translational invariance.
pub(crate) fn directional_overlap_third_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    v: &[f64],
) -> Result<Matrix> {
    ensure_direction_len(system, v)?;
    let n = basis.len();
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let bra_atom = basis.aos[mu].atom_index;
        let ra = system.atoms[bra_atom].position;
        for nu in 0..n {
            let ket_atom = basis.aos[nu].atom_index;
            let (va, vk) = direction_on_pair(v, bra_atom, ket_atom);
            if va == [0.0; 3] && vk == [0.0; 3] {
                continue;
            }
            let rk = system.atoms[ket_atom].position;
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rk);
            out[(mu, nu)] = directional_third(
                &pair.t_bra_bra_bra[0],
                &pair.t_bra_bra_ket[0],
                &pair.t_bra_ket_ket[0],
                &pair.t_ket_ket_ket[0],
                &va,
                &vk,
            );
        }
    }
    Ok(out)
}

// =============================================================================================
// Directional FOURTH-derivative builders (frozen-density quartic stage)
//
// Each builder below returns the SCALAR `Σ_abcd v_a v_b v_c v_d Q_abcd` of a frozen-density
// fourth-derivative block directly, in one AO-pair sweep, instead of materialising the nested
// `out[c][d][(a,b)]` store (`ndof⁴` doubles) and contracting it afterwards. The per-pair
// fourth-derivative data — the overlap ladder, the H0 prefactor ladder, the CN jets — does not
// depend on the four derivative indices at all; only the contraction weights do. So contracting
// all four legs against `v` inside the sweep is exact, and it removes both the `O(ndof⁴)` store
// and the `O(ndof⁴)` scatter/contract.
//
// This matters twice over: it is the dominant cost of the frozen-density stage, and it is what
// keeps the quartic's working set `O(n²)` so the directional mode can run above the
// `MAX_FOURTH_DERIVATIVE_NDOF` system size the full-tensor builders imposed.
//
// Every builder is gated element-wise/scalar against the nested-block + `contract_nested_vvvv`
// path it replaces by `directional_fourth_tests`.
// =============================================================================================

/// `Σ_bcde v_b v_c v_d v_e ∂⁴f/∂R_b∂R_c∂R_d∂R_e` — the `2⁴` centre assignments of
/// [`fourth_select`] collapsed to the five distinct bra/ket patterns with multiplicities
/// `1, 4, 6, 4, 1` (the number of ways to pick which of the four identical legs sit on the ket
/// centre), i.e. [`directional_third`] one order up.
#[inline]
fn directional_fourth(
    q_bbbb: &Ten4,
    q_bbbk: &Ten4,
    q_bbkk: &Ten4,
    q_bkkk: &Ten4,
    q_kkkk: &Ten4,
    va: &[f64; 3],
    vk: &[f64; 3],
) -> f64 {
    let mut out = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    out += va[a] * va[b] * va[c] * va[d] * q_bbbb[a][b][c][d]
                        + 4.0 * va[a] * va[b] * va[c] * vk[d] * q_bbbk[a][b][c][d]
                        + 6.0 * va[a] * va[b] * vk[c] * vk[d] * q_bbkk[a][b][c][d]
                        + 4.0 * va[a] * vk[b] * vk[c] * vk[d] * q_bkkk[a][b][c][d]
                        + vk[a] * vk[b] * vk[c] * vk[d] * q_kkkk[a][b][c][d];
                }
            }
        }
    }
    out
}

/// The number of distinct ordered permutations of a SORTED quadruple, i.e.
/// `distinct_perms4(i,j,k,l).len()` computed in closed form as `4!/∏(run length)!`.
///
/// The directional builders need only the COUNT, never the permutations themselves: every
/// permutation of a quadruple carries the same four direction weights, so it contributes the same
/// product. Pinned against [`distinct_perms4`] over every sorted slot quadruple by
/// `multiplicity4_counts_distinct_perms4`.
#[inline]
fn multiplicity4(i: usize, j: usize, k: usize, l: usize) -> f64 {
    let mut runs = [0usize; 4];
    let mut nruns = 0usize;
    let mut prev = usize::MAX;
    for &x in &[i, j, k, l] {
        if x != prev {
            nruns += 1;
            prev = x;
        }
        runs[nruns - 1] += 1;
    }
    let factorial = |m: usize| match m {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        _ => 24.0,
    };
    24.0 / (factorial(runs[0]) * factorial(runs[1]) * factorial(runs[2]) * factorial(runs[3]))
}

/// **Directional frozen-density Pulay fourth derivative** — the one-pass twin of
/// [`fixed_density_pulay_fourth_derivative`] followed by a `vvvv` contraction.
///
/// The nested builder evaluates one symmetry-unique value per unordered slot quadruple and
/// scatters it over [`distinct_perms4`]. Every permutation lands on a tensor position whose four
/// DOF weights are the SAME four numbers, so the whole scatter-then-contract collapses to
/// `value × multiplicity × w_i w_j w_k w_l` — [`multiplicity4`] replaces both the `Vec`
/// allocation and the quadratic duplicate scan the nested path pays per quadruple.
pub(crate) fn directional_fixed_density_pulay_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
) -> Result<f64> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let basis = &electronic.basis;
    let ao_scalar_potential = ao_scalar_potentials(basis, &electronic.shell_scc_potential);
    let mut acc = 0.0;

    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu_index = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            if va == [0.0; 3] && vk == [0.0; 3] {
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
                contracted_pair_with_fourth_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let overlap = pair.moments[0];
            let h0 =
                h0_prefactor_fourth(system, params, electronic, shell_mu_index, shell_nu_index)?;
            let scalar_shift = ao_scalar_potential[mu] + ao_scalar_potential[nu];
            let overlap_coeff = p * (2.0 * h0.value - scalar_shift) - 2.0 * w;
            let two_p = 2.0 * p;

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
            let s3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &pair.t_bra_bra_bra[0],
                    &pair.t_bra_bra_ket[0],
                    &pair.t_bra_ket_ket[0],
                    &pair.t_ket_ket_ket[0],
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let s4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &pair.q_bbbb,
                    &pair.q_bbbk,
                    &pair.q_bbkk,
                    &pair.q_bkkk,
                    &pair.q_kkkk,
                    centers,
                    axes,
                )
            };
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
            let h3 = |ca: Center, cb: Center, cc: Center, axa: usize, axb: usize, axc: usize| {
                third_select(
                    &h0.t_bra_bra_bra,
                    &h0.t_bra_bra_ket,
                    &h0.t_bra_ket_ket,
                    &h0.t_ket_ket_ket,
                    [ca, cb, cc],
                    [axa, axb, axc],
                )
            };
            let h4 = |centers: [Center; 4], axes: [usize; 4]| {
                fourth_select(
                    &h0.q_bbbb,
                    &h0.q_bbbk,
                    &h0.q_bbkk,
                    &h0.q_bkkk,
                    &h0.q_kkkk,
                    centers,
                    axes,
                )
            };

            let slot_center = |s: usize| if s < 3 { Center::Bra } else { Center::Ket };
            let value_of = |i: usize, j: usize, k: usize, l: usize| -> f64 {
                let (ca, axa) = (slot_center(i), i % 3);
                let (cb, axb) = (slot_center(j), j % 3);
                let (cc, axc) = (slot_center(k), k % 3);
                let (cd, axd) = (slot_center(l), l % 3);
                let s_abcd = s4([ca, cb, cc, cd], [axa, axb, axc, axd]);
                let h_abcd = h4([ca, cb, cc, cd], [axa, axb, axc, axd]);
                let (s_a, s_b, s_c, s_d) = (s1(ca, axa), s1(cb, axb), s1(cc, axc), s1(cd, axd));
                let (h_a, h_b, h_c, h_d) = (h1(ca, axa), h1(cb, axb), h1(cc, axc), h1(cd, axd));
                let s_ab = s2(ca, cb, axa, axb);
                let s_ac = s2(ca, cc, axa, axc);
                let s_ad = s2(ca, cd, axa, axd);
                let s_bc = s2(cb, cc, axb, axc);
                let s_bd = s2(cb, cd, axb, axd);
                let s_cd = s2(cc, cd, axc, axd);
                let h_ab = h2(ca, cb, axa, axb);
                let h_ac = h2(ca, cc, axa, axc);
                let h_ad = h2(ca, cd, axa, axd);
                let h_bc = h2(cb, cc, axb, axc);
                let h_bd = h2(cb, cd, axb, axd);
                let h_cd = h2(cc, cd, axc, axd);
                let s_abc = s3(ca, cb, cc, axa, axb, axc);
                let s_abd = s3(ca, cb, cd, axa, axb, axd);
                let s_acd = s3(ca, cc, cd, axa, axc, axd);
                let s_bcd = s3(cb, cc, cd, axb, axc, axd);
                let h_abc = h3(ca, cb, cc, axa, axb, axc);
                let h_abd = h3(ca, cb, cd, axa, axb, axd);
                let h_acd = h3(ca, cc, cd, axa, axc, axd);
                let h_bcd = h3(cb, cc, cd, axb, axc, axd);
                overlap_coeff * s_abcd
                    + two_p
                        * (overlap * h_abcd
                            + (h_abc * s_d + h_abd * s_c + h_acd * s_b + h_bcd * s_a)
                            + (h_ab * s_cd
                                + h_ac * s_bd
                                + h_ad * s_bc
                                + h_bc * s_ad
                                + h_bd * s_ac
                                + h_cd * s_ab)
                            + (h_a * s_bcd + h_b * s_acd + h_c * s_abd + h_d * s_abc))
            };
            let weight = |s: usize| v[3 * atom_of(slot_center(s)) + s % 3];
            let w: [f64; 6] = std::array::from_fn(weight);
            for i in 0..6 {
                for j in i..6 {
                    for k in j..6 {
                        for l in k..6 {
                            let weights = w[i] * w[j] * w[k] * w[l];
                            if weights == 0.0 {
                                continue;
                            }
                            let value = value_of(i, j, k, l);
                            if value == 0.0 {
                                continue;
                            }
                            acc += value * multiplicity4(i, j, k, l) * weights;
                        }
                    }
                }
            }
        }
    }
    Ok(acc)
}

/// A scalar's directional Taylor coefficients along `v`: the value plus the four directional
/// derivatives `Σ v…v ∂^k/∂R^k`. The direction-contracted replacement for [`DedcnJet4`], whose
/// `ndof⁴` flat array is what forces the frozen CN-H0 quartic into an `O(ndof⁴)` working set.
#[derive(Clone, Copy, Default)]
pub(crate) struct DirectionalJet4 {
    pub(crate) value: f64,
    pub(crate) d1: f64,
    pub(crate) d2: f64,
    pub(crate) d3: f64,
    pub(crate) d4: f64,
}

/// [`cn_h0_dedcn_jets_fourth`] with every derivative leg contracted against `v` — `∂E/∂CN_A` and
/// its four directional nuclear derivatives, `O(1)` storage per atom.
///
/// Per AO pair the jet entries are the order-`k` product Leibniz of the two geometric factors
/// `scale` and `S`, so contracting all legs against the same `v` turns the explicit
/// `1 / (1,1) / (1,2,1) / (1,3,3,1) / (1,4,6,4,1)` term lists into the binomial combinations of
/// the two directional ladders below.
pub(crate) fn directional_cn_h0_dedcn_jets_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
) -> Result<Vec<DirectionalJet4>> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    let mut jets = vec![DirectionalJet4::default(); nat];

    // On-site diagonal block (R-independent): value only.
    for shell in basis.shells.iter() {
        let dsedcn = -shell.kcn_raw.unwrap_or(0.0);
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            jets[shell.atom_index].value += dsedcn * electronic.density[(iao, iao)];
        }
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
            let dsedcn_mu = -shell_mu.kcn_raw.unwrap_or(0.0);
            let dsedcn_nu = -shell_nu.kcn_raw.unwrap_or(0.0);
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            let pair =
                contracted_pair_with_fourth_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let scale = h0_scale_fourth(system, params, shell_mu_index, shell_nu_index, basis)?;

            let s0 = pair.moments[0];
            let val0 = p * scale.value * s0;
            jets[atom_mu].value += dsedcn_mu * val0;
            jets[atom_nu].value += dsedcn_nu * val0;
            if va == [0.0; 3] && vk == [0.0; 3] {
                continue;
            }
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let s3 = directional_third(
                &pair.t_bra_bra_bra[0],
                &pair.t_bra_bra_ket[0],
                &pair.t_bra_ket_ket[0],
                &pair.t_ket_ket_ket[0],
                &va,
                &vk,
            );
            let s4 = directional_fourth(
                &pair.q_bbbb,
                &pair.q_bbbk,
                &pair.q_bbkk,
                &pair.q_bkkk,
                &pair.q_kkkk,
                &va,
                &vk,
            );
            let g0 = scale.value;
            let g1 = directional_first(scale.d_bra, scale.d_ket, &va, &vk);
            let g2 = directional_second(
                &scale.h_bra_bra,
                &scale.h_bra_ket,
                &scale.h_ket_ket,
                &va,
                &vk,
            );
            let g3 = directional_third(
                &scale.t_bra_bra_bra,
                &scale.t_bra_bra_ket,
                &scale.t_bra_ket_ket,
                &scale.t_ket_ket_ket,
                &va,
                &vk,
            );
            let g4 = directional_fourth(
                &scale.q_bbbb,
                &scale.q_bbbk,
                &scale.q_bbkk,
                &scale.q_bkkk,
                &scale.q_kkkk,
                &va,
                &vk,
            );
            let v1 = g0 * s1 + g1 * s0;
            let v2 = g0 * s2 + 2.0 * g1 * s1 + g2 * s0;
            let v3 = g0 * s3 + 3.0 * g1 * s2 + 3.0 * g2 * s1 + g3 * s0;
            let v4 = g0 * s4 + 4.0 * g1 * s3 + 6.0 * g2 * s2 + 4.0 * g3 * s1 + g4 * s0;
            for (atom, dsedcn) in [(atom_mu, dsedcn_mu), (atom_nu, dsedcn_nu)] {
                let jet = &mut jets[atom];
                jet.d1 += dsedcn * p * v1;
                jet.d2 += dsedcn * p * v2;
                jet.d3 += dsedcn * p * v3;
                jet.d4 += dsedcn * p * v4;
            }
        }
    }
    Ok(jets)
}

/// [`cn_h0_cn_jets_fourth`] with every derivative leg contracted against `v` — `CN_A` and its four
/// directional nuclear derivatives, `O(1)` storage AND `O(1)` work per counting-function pair.
///
/// Each pair's rank-`k` block is an isotropic tensor in the unit separation `u` and the Kronecker
/// delta, and the six slots carry `σ = +1` on atom `i`, `σ = −1` on atom `j`. Contracting all legs
/// against `v` therefore reduces the whole block to the two scalars `u·t` and `t·t` of the
/// **relative** direction `t = v_i − v_j`, with the Kronecker terms' multiplicities (`3` at third
/// order; `6` and `3` at fourth) counting the distinct index pairings.
pub(crate) fn directional_cn_h0_cn_jets_fourth(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Vec<DirectionalJet4>> {
    ensure_direction_len(system, v)?;
    let nat = system.atoms.len();
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
    let mut jets = vec![DirectionalJet4::default(); nat];
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
        let a2 = g / r;
        let a3 = coeff_uuu / r;
        let a4 =
            d.fourth - 6.0 * d.third / r + 15.0 * d.second / (r * r) - 15.0 * d.first / (r * r * r);
        // The RELATIVE direction the σ-weighted slot sum collapses to.
        let t = [
            v[3 * pair.i] - v[3 * pair.j],
            v[3 * pair.i + 1] - v[3 * pair.j + 1],
            v[3 * pair.i + 2] - v[3 * pair.j + 2],
        ];
        let ut = u[0] * t[0] + u[1] * t[1] + u[2] * t[2];
        let tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
        let d1 = d.first * ut;
        let d2 = d.second * ut * ut + (d.first / r) * (tt - ut * ut);
        let d3 = coeff_uuu * ut * ut * ut + 3.0 * g * tt * ut;
        let d4 = a4 * ut * ut * ut * ut + 6.0 * a3 * tt * ut * ut + 3.0 * a2 * tt * tt;
        for jet_atom in [pair.i, pair.j] {
            let jet = &mut jets[jet_atom];
            jet.value += counting;
            jet.d1 += d1;
            jet.d2 += d2;
            jet.d3 += d3;
            jet.d4 += d4;
        }
    }
    Ok(jets)
}

/// **Directional frozen CN-H0 fourth derivative** — the one-pass twin of
/// [`fixed_density_cn_h0_fourth_derivative`] followed by a `vvvv` contraction.
///
/// The nested builder's twelve `CN`×`∂E/∂CN` products are contracted index by index against the
/// same `v`, so each collapses to a product of the two directional jets' orders. Grouping by
/// order gives `1 × (n₄·e₀)`, `4 × (n₃·e₁)`, `5 × (n₂·e₂)` and `2 × (n₁·e₃)` — the multiplicities
/// of the twelve terms, NOT a binomial expansion, because the source object is deliberately the
/// exact `∂_d` of the third-order block rather than the symmetric quartic.
pub(crate) fn directional_fixed_density_cn_h0_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let de = directional_cn_h0_dedcn_jets_fourth(system, params, electronic, v)?;
    let cn = directional_cn_h0_cn_jets_fourth(system, coordination_cutoff, v)?;
    let mut acc = 0.0;
    for atom in 0..system.atoms.len() {
        let (n, e) = (&cn[atom], &de[atom]);
        acc += e.value * n.d4
            + 4.0 * n.d3 * e.d1
            + 5.0 * n.d2 * e.d2
            + 2.0 * n.d1 * e.d3;
    }
    Ok(acc)
}

/// **Directional frozen scalar-overlap fourth derivative** — the one-pass twin of
/// [`fixed_density_scalar_overlap_fourth_derivative`] followed by a `vvvv` contraction.
///
/// The nested builder's four terms are
/// `dds_ad·dscalar_bc + ds_a·dscalar_bcd + ddds_acd·dscalar_b + dds_ac·dscalar_bd`, where the
/// overlap-derivative indices are restricted to the pair's two centres and the shell-potential
/// indices range over all DOF. Contracting every index against `v` splits each term into
/// (directional overlap derivative) × (directional shell-potential derivative), and the first and
/// fourth terms become the SAME object `S₂·V₂` — hence the factor 2:
///
/// ```text
///   −Σ_pairs p · ( S₁·V₃ + 2·S₂·V₂ + S₃·V₁ )
/// ```
///
/// with `S_k` the pair's directional overlap ladder and `V_k = Σ v…v ∂^k(V_μ + V_ν)`.
pub(crate) fn directional_fixed_density_scalar_overlap_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    v: &[f64],
) -> Result<f64> {
    ensure_non_pbc(system)?;
    ensure_direction_len(system, v)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &electronic.basis;
    let nsh = basis.shells.len();
    let dscalar1 =
        shell_scalar_potential_first_derivatives(system, basis, &electronic.shell_charges, params)?;
    let dscalar2 = shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let dscalar3 =
        shell_scalar_potential_third_derivatives(system, basis, &electronic.shell_charges, params)?;
    // The three directional shell-potential ladders, built once.
    let mut v1 = vec![0.0_f64; nsh];
    let mut v2 = vec![0.0_f64; nsh];
    let mut v3 = vec![0.0_f64; nsh];
    for s in 0..nsh {
        let (mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0);
        for (b, &vb) in v.iter().enumerate() {
            if vb == 0.0 {
                continue;
            }
            a1 += vb * dscalar1[(s, b)];
            for (c, &vc) in v.iter().enumerate() {
                if vc == 0.0 {
                    continue;
                }
                a2 += vb * vc * dscalar2[s][(b, c)];
                let base = (b * ndof + c) * ndof;
                let mut inner = 0.0;
                for (d, &vd) in v.iter().enumerate() {
                    inner += vd * dscalar3[s][base + d];
                }
                a3 += vb * vc * inner;
            }
        }
        v1[s] = a1;
        v2[s] = a2;
        v3[s] = a3;
    }

    let mut acc = 0.0;
    for mu in 0..basis.len() {
        let atom_mu = basis.aos[mu].atom_index;
        let shell_mu = basis.aos[mu].shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..mu {
            let atom_nu = basis.aos[nu].atom_index;
            if atom_mu == atom_nu {
                continue;
            }
            let (va, vk) = direction_on_pair(v, atom_mu, atom_nu);
            if va == [0.0; 3] && vk == [0.0; 3] {
                continue;
            }
            let shell_nu = basis.aos[nu].shell_index;
            let rnu = system.atoms[atom_nu].position;
            let p = electronic.density[(mu, nu)];
            if p.abs() <= 1.0e-18 {
                continue;
            }
            let pair =
                contracted_pair_with_third_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
            let s1 = directional_first(pair.d_bra[0], pair.d_ket[0], &va, &vk);
            let s2 = directional_second(
                &pair.h_bra_bra[0],
                &pair.h_bra_ket[0],
                &pair.h_ket_ket[0],
                &va,
                &vk,
            );
            let s3 = directional_third(
                &pair.t_bra_bra_bra[0],
                &pair.t_bra_bra_ket[0],
                &pair.t_bra_ket_ket[0],
                &pair.t_ket_ket_ket[0],
                &va,
                &vk,
            );
            let (p1, p2, p3) = (
                v1[shell_mu] + v1[shell_nu],
                v2[shell_mu] + v2[shell_nu],
                v3[shell_mu] + v3[shell_nu],
            );
            acc -= p * (s1 * p3 + 2.0 * s2 * p2 + s3 * p1);
        }
    }
    Ok(acc)
}

/// **The one-pass directional FOURTH builders vs the nested `out[c][d][(a,b)]` store.**
///
/// Each builder must reproduce `contract_nested_vvvv(nested_builder(...), v)` — the code it
/// replaces — as a SCALAR to rounding. Both routes evaluate the same per-pair data; only the order
/// in which the four `v` weights are applied differs, so anything above ~1e-12 relative is a
/// derivation error rather than float noise.
#[cfg(test)]
mod directional_fourth_tests {
    use super::*;
    use crate::electronic::ElectronicOptions;

    /// `Σ_abcd v_a v_b v_c v_d blocks[c][d][(a,b)]` — a local copy of the contraction the
    /// frozen-density stage applies to the nested store (which lives in
    /// `crate::fourth_derivative::directional` and is private there).
    fn contract_nested_vvvv(blocks: &[Vec<Matrix>], v: &[f64]) -> f64 {
        let ndof = v.len();
        let mut acc = 0.0;
        for (c, row) in blocks.iter().enumerate() {
            if v[c] == 0.0 {
                continue;
            }
            for (d, slab) in row.iter().enumerate() {
                let vcd = v[c] * v[d];
                if vcd == 0.0 {
                    continue;
                }
                for a in 0..ndof {
                    for b in 0..ndof {
                        acc += vcd * v[a] * v[b] * slab[(a, b)];
                    }
                }
            }
        }
        acc
    }

    fn fixture(xyz: &str) -> (PeriodicSystem, Gfn1Parameters, ElectronicResult, f64, Vec<f64>) {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-12,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = crate::electronic::run_electronic(&system, &params, options).unwrap();
        let ndof = 3 * system.atoms.len();
        // Generic skew direction: no zero components, no accidental symmetry.
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        (system, params, electronic, cutoff, v)
    }

    fn assert_close(label: &str, got: f64, want: f64) {
        let delta = (got - want).abs();
        let scale = want.abs().max(1.0);
        eprintln!("{label}: one-pass {got:.17e} vs nested·vvvv {want:.17e} (delta {delta:.3e})");
        assert!(
            want.abs() > 1.0e-10,
            "{label}: the reference value is numerically zero — the gate is vacuous"
        );
        assert!(
            delta <= 1.0e-12 * scale,
            "{label}: one-pass directional fourth differs from the nested store contraction: \
             got {got:.17e} want {want:.17e} delta {delta:.3e}"
        );
    }

    fn run_gate(xyz: &str, label: &str) {
        let (system, params, electronic, cutoff, v) = fixture(xyz);
        assert_close(
            &format!("{label} / pulay"),
            directional_fixed_density_pulay_fourth(&system, &params, &electronic, &v).unwrap(),
            contract_nested_vvvv(
                &fixed_density_pulay_fourth_derivative(&system, &params, &electronic).unwrap(),
                &v,
            ),
        );
        assert_close(
            &format!("{label} / cn_h0"),
            directional_fixed_density_cn_h0_fourth(&system, &params, &electronic, cutoff, &v)
                .unwrap(),
            contract_nested_vvvv(
                &fixed_density_cn_h0_fourth_derivative(&system, &params, &electronic, cutoff)
                    .unwrap(),
                &v,
            ),
        );
        assert_close(
            &format!("{label} / scalar_overlap"),
            directional_fixed_density_scalar_overlap_fourth(&system, &params, &electronic, &v)
                .unwrap(),
            contract_nested_vvvv(
                &fixed_density_scalar_overlap_fourth_derivative(&system, &params, &electronic)
                    .unwrap(),
                &v,
            ),
        );
    }

    /// [`multiplicity4`] must count exactly what [`distinct_perms4`] enumerates, for every sorted
    /// slot quadruple of a two-centre pair (the `4!`, `4!/2!`, `4!/2!2!`, `4!/3!` and `4!/4!`
    /// patterns all occur). Pure combinatorics — this is what licenses the directional builders
    /// dropping the explicit permutation list.
    #[test]
    fn multiplicity4_counts_distinct_perms4() {
        for i in 0..6 {
            for j in i..6 {
                for k in j..6 {
                    for l in k..6 {
                        let want = distinct_perms4(i, j, k, l).len() as f64;
                        let got = multiplicity4(i, j, k, l);
                        assert!(
                            (got - want).abs() < 1.0e-12,
                            "multiplicity4({i},{j},{k},{l}) = {got} but distinct_perms4 has {want}"
                        );
                    }
                }
            }
        }
    }

    /// Non-equilibrium water: every channel active, both centres CN-coupled.
    #[test]
    fn directional_fourth_builders_match_nested_store_water() {
        run_gate(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            "water",
        );
    }

    /// The CH3Br fragment of the stage-1 gate geometry: a heavy centre with `d` shells, a very
    /// different CN environment, and enough DOF that the nested store's index bookkeeping is not
    /// trivially satisfied.
    #[test]
    fn directional_fourth_builders_match_nested_store_ch3br() {
        run_gate(
            "5\nCH3Br\nC 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\n\
             H 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\n\
             H -0.515000 -0.892000 -0.330000\n",
            "ch3br",
        );
    }
}

/// **The one-pass directional SECOND builders vs the `O(ndof²)` double loop.**
///
/// The four `..._second_directional` builders replace the per-`(c,d)` block
/// builds of
/// [`crate::fourth_derivative::assemble::directional_second_order_legs`]. They
/// must reproduce that double loop ELEMENT-WISE to rounding — the only
/// difference is the summation order, so anything above ~1e-13 relative is a
/// derivation error, not float noise.
#[cfg(test)]
mod directional_second_tests {
    use super::*;
    use crate::electronic::ElectronicOptions;

    /// A converged reference, a skew direction, and a synthetic per-DOF
    /// shell-charge response. The SCC-scalar block is AFFINE in that response
    /// leg and the potential-derivative ladders are exactly linear in the
    /// charges, so an arbitrary non-degenerate stand-in gates the charge path
    /// just as well as the real `q^(d)` — and keeps the gate free of a
    /// charge-space solve.
    struct Fixture {
        system: PeriodicSystem,
        params: Gfn1Parameters,
        electronic: ElectronicResult,
        v: Vec<f64>,
        qresp: Vec<Vec<f64>>,
        cutoff: f64,
    }

    fn fixture(xyz: &str, temperature: f64) -> Fixture {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let mut electronic_options = ElectronicOptions::default();
        electronic_options.enable_dispersion = false;
        electronic_options.electronic_temperature = temperature;
        electronic_options.energy_tolerance = 1.0e-12;
        electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = electronic_options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, electronic_options).unwrap();
        let ndof = 3 * system.atoms.len();
        let nsh = electronic.basis.shells.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let qresp: Vec<Vec<f64>> = (0..ndof)
            .map(|d| {
                (0..nsh)
                    .map(|s| {
                        0.017 * (((d * 5 + s * 3) % 7) as f64) - 0.041 * (((d + s) % 3) as f64)
                    })
                    .collect()
            })
            .collect();
        Fixture {
            system,
            params,
            electronic,
            v,
            qresp,
            cutoff,
        }
    }

    fn max_abs(m: &Matrix) -> f64 {
        m.as_slice().iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    /// The pre-optimization reference: exactly the `O(ndof²)` loop
    /// `directional_second_order_legs` used to run, kept split per block so a
    /// mismatch localises itself.
    fn double_loop(f: &Fixture) -> [Matrix; 4] {
        let basis = &f.electronic.basis;
        let n = basis.len();
        let nshell = basis.shells.len();
        let ndof = 3 * f.system.atoms.len();
        let dvdr_q = shell_scalar_potential_first_derivatives(
            &f.system,
            basis,
            &f.electronic.shell_charges,
            &f.params,
        )
        .unwrap();
        let mut out = [
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
            Matrix::zeros(n, n),
        ];
        for c in 0..ndof {
            if f.v[c] == 0.0 {
                continue;
            }
            for d in 0..ndof {
                let w = f.v[c] * f.v[d];
                if w == 0.0 {
                    continue;
                }
                let v_geo_d: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, d)]).collect();
                let blocks = [
                    h0_bare_second_derivative_matrix(&f.system, &f.params, &f.electronic, c, d)
                        .unwrap(),
                    h0_cn_block_second_derivative_matrix(
                        &f.system,
                        &f.params,
                        &f.electronic,
                        f.cutoff,
                        c,
                        d,
                    )
                    .unwrap(),
                    h0_scc_scalar_second_derivative_matrix(
                        &f.system,
                        &f.params,
                        &f.electronic,
                        &v_geo_d,
                        &f.qresp[d],
                        c,
                        d,
                    )
                    .unwrap(),
                    crate::response::cpxtb::overlap_second_derivative_matrix(&f.system, basis, c, d)
                        .unwrap(),
                ];
                for (acc, block) in out.iter_mut().zip(&blocks) {
                    let dst = acc.as_mut_slice();
                    let src = block.as_slice();
                    for k in 0..n * n {
                        dst[k] += w * src[k];
                    }
                }
            }
        }
        out
    }

    fn one_pass(f: &Fixture) -> [Matrix; 4] {
        let basis = &f.electronic.basis;
        let nshell = basis.shells.len();
        let ndof = 3 * f.system.atoms.len();
        let dvdr_q = shell_scalar_potential_first_derivatives(
            &f.system,
            basis,
            &f.electronic.shell_charges,
            &f.params,
        )
        .unwrap();
        // The two already-directional legs the one-pass SCC builder expects.
        let v_c: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|b| f.v[b] * dvdr_q[(s, b)]).sum())
            .collect();
        let q_c: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|d| f.v[d] * f.qresp[d][s]).sum())
            .collect();
        [
            directional_h0_bare_second_matrix(&f.system, &f.params, &f.electronic, &f.v).unwrap(),
            directional_h0_cn_block_second_matrix(
                &f.system,
                &f.params,
                &f.electronic,
                f.cutoff,
                &f.v,
            )
            .unwrap(),
            directional_h0_scc_scalar_second_matrix(
                &f.system,
                &f.params,
                &f.electronic,
                &f.v,
                &v_c,
                &q_c,
            )
            .unwrap(),
            directional_overlap_second_matrix(&f.system, basis, &f.v).unwrap(),
        ]
    }

    fn run_gate(xyz: &str, temperature: f64, label: &str) {
        let f = fixture(xyz, temperature);
        let reference = double_loop(&f);
        let one_pass = one_pass(&f);
        let names = ["h0 bare", "h0 CN block", "SCC scalar", "overlap"];
        for ((name, want), got) in names.iter().zip(&reference).zip(&one_pass) {
            let scale = max_abs(want).max(1.0);
            let delta = want.max_abs_diff(got);
            eprintln!("{label} / {name}: max |one-pass − double loop| {delta:.3e} (scale {scale:.3e})");
            assert!(
                delta <= 1.0e-12 * scale,
                "{label} / {name}: one-pass directional second differs from the ndof² double \
                 loop by {delta:.6e} (scale {scale:.6e})"
            );
            // A gate that only ever compared zeros would pass vacuously.
            assert!(
                max_abs(want) > 1.0e-8,
                "{label} / {name}: the reference block is numerically zero — the gate is vacuous"
            );
        }
    }

    /// Non-equilibrium water: every channel active, no symmetry cancellations,
    /// both centres CN-coupled.
    #[test]
    fn directional_second_builders_match_double_loop_water() {
        run_gate(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            "water",
        );
    }

    /// A distorted Ni-C-O fragment: exercises the `d`-shell AO blocks and a
    /// transition-metal CN coupling, which water cannot reach.
    #[test]
    fn directional_second_builders_match_double_loop_metal_carbonyl() {
        run_gate(
            "3\ndistorted NiCO\nNi 0.02 -0.03 0.01\nC 1.66 0.21 -0.09\nO 2.81 0.14 0.12\n",
            3000.0,
            "NiCO",
        );
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod third_derivative_tests;
