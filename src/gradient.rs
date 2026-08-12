// SPDX-License-Identifier: GPL-3.0-or-later

use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::{harmonic_average, ShellChargeModel};
use crate::data_tables::atomic_radius_bohr;
use crate::dispersion::{d4_dispersion_energy_gradient, dispersion_energy_gradient};
use crate::electronic::{run_electronic, ElectronicOptions, ElectronicResult};
use crate::error::{Gfn1Error, Result};
use crate::halogen::halogen_energy_gradient;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::{contracted_pair_with_derivatives, IntegralMatrices};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::repulsion::repulsion_energy_gradient;
use crate::system::PeriodicSystem;
use rayon::prelude::*;

const DIST_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct AnalyticGradientOptions {
    pub electronic: ElectronicOptions,
    pub include_repulsion: bool,
    pub include_dispersion: bool,
    pub include_hamiltonian: bool,
    pub include_scc: bool,
    pub include_halogen: bool,
}

impl Default for AnalyticGradientOptions {
    fn default() -> Self {
        Self {
            electronic: ElectronicOptions::default(),
            include_repulsion: true,
            include_dispersion: true,
            include_hamiltonian: true,
            include_scc: true,
            include_halogen: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalyticGradientResult {
    pub electronic_result: ElectronicResult,
    pub total_energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub electronic_gradient: Vec<Vec3>,
    pub repulsion_gradient: Vec<Vec3>,
    pub dispersion_gradient: Vec<Vec3>,
    pub halogen_gradient: Vec<Vec3>,
    pub max_gradient: f64,
}

pub fn analytic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: AnalyticGradientOptions,
) -> Result<AnalyticGradientResult> {
    if system.lattice.is_some()
        || options.electronic.boundary != crate::model::BoundaryCondition::NonPeriodic
    {
        return pbc_analytic_gradient_result(system, params, &options);
    }
    let _profile = crate::profile::scope("gradient.nonpbc.total");
    let electronic = run_electronic(system, params, options.electronic.clone())?;
    analytic_gradient_from_result(system, params, electronic, &options)
}

/// Periodic analytic gradient projected into the molecular-shaped result type.
/// Routes to the Gamma-point / k-point PBC gradient and splits out the periodic
/// repulsion, D3 dispersion, and halogen-bond correction.
fn pbc_analytic_gradient_result(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticGradientOptions,
) -> Result<AnalyticGradientResult> {
    if options.electronic.experimental_d4 {
        return Err(Gfn1Error::InvalidInput(
            "experimental D4 dispersion is implemented for non-PBC gradients only".to_string(),
        ));
    }
    let pbc = crate::pbc::PbcOptions::for_boundary(options.electronic.boundary);
    let result = crate::pbc::pbc_analytic_gradient(system, params, &options.electronic, &pbc)?;
    let electronic_result =
        crate::pbc::pbc_electronic_result(result.scf.clone(), system, pbc.ao_cutoff)?;
    let nat = system.atoms.len();
    let repulsion_gradient = repulsion_energy_gradient(system, params)?.gradient;
    let dispersion_gradient = if options.electronic.enable_dispersion {
        dispersion_energy_gradient(
            system,
            params,
            options.electronic.d3_reference_path.as_deref(),
        )?
        .gradient
    } else {
        vec![Vec3::zero(); nat]
    };
    let halogen_gradient = halogen_energy_gradient(system, params)?.gradient;
    let electronic_gradient: Vec<Vec3> = result
        .gradient
        .iter()
        .zip(&repulsion_gradient)
        .zip(&dispersion_gradient)
        .zip(&halogen_gradient)
        .map(|(((total, rep), disp), hal)| *total - *rep - *disp - *hal)
        .collect();
    Ok(AnalyticGradientResult {
        electronic_result,
        total_energy: result.total_energy,
        gradient: result.gradient,
        forces: result.forces,
        electronic_gradient,
        repulsion_gradient,
        dispersion_gradient,
        halogen_gradient,
        max_gradient: result.max_gradient,
    })
}

pub fn analytic_gradient_from_result(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: ElectronicResult,
    options: &AnalyticGradientOptions,
) -> Result<AnalyticGradientResult> {
    let _profile = crate::profile::scope("gradient.assemble.total");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "analytic gradient currently supports non-PBC only".to_string(),
        ));
    }
    let nat = system.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];
    let mut electronic_gradient = vec![Vec3::zero(); nat];
    let mut repulsion_gradient = vec![Vec3::zero(); nat];
    let mut dispersion_gradient = vec![Vec3::zero(); nat];
    let mut halogen_gradient = vec![Vec3::zero(); nat];

    if options.include_hamiltonian || options.include_scc {
        let _profile = crate::profile::scope("gradient.electronic_terms");
        electronic_gradient = electronic_gradient_terms(system, params, &electronic, options)?;
        for atom in 0..nat {
            gradient[atom] += electronic_gradient[atom];
        }
    }

    // DFT+U/+U+V consistent-force correction `F_corr` for the LINEAR-RESPONSE path:
    // the explicit geometry dependence of the recomputed Hubbard parameters,
    // `F_corr = Σ_I (∂E/∂U_I)(dU_I/dR) + Σ_pairs (∂E/∂V_IJ)(dV_IJ/dR)`. Omitted in
    // fixed-`U` mode (U/V are geometry-independent constants there → already exact)
    // and when `+U` is off. This is what makes the force consistent with the
    // per-geometry-recomputed `U(R)`; the frozen-U overlap-Pulay term lives in
    // `electronic_gradient_terms`.
    if options.electronic.plus_u
        && options.electronic.hubbard_u_linear_response
        && options.include_scc
    {
        let _profile = crate::profile::scope("gradient.plus_u_consistency");
        let fc = plus_u_consistency_gradient_terms(system, params, &electronic, options)?;
        for atom in 0..nat {
            electronic_gradient[atom] += fc[atom];
            gradient[atom] += fc[atom];
        }
    }

    if options.electronic.multipole && (options.include_hamiltonian || options.include_scc) {
        let _profile = crate::profile::scope("gradient.multipole");
        let mp = multipole_gradient_terms(system, params, &electronic, options)?;
        for atom in 0..nat {
            electronic_gradient[atom] += mp[atom];
            gradient[atom] += mp[atom];
        }
    }

    if options.electronic.lr_exchange && (options.include_hamiltonian || options.include_scc) {
        let _profile = crate::profile::scope("gradient.exchange");
        let ex = exchange_gradient_terms(system, params, &electronic, options)?;
        for atom in 0..nat {
            electronic_gradient[atom] += ex[atom];
            gradient[atom] += ex[atom];
        }
    }

    if options.include_repulsion {
        let _profile = crate::profile::scope("gradient.repulsion");
        let rep = repulsion_energy_gradient(system, params)?;
        repulsion_gradient = rep.gradient;
        for atom in 0..nat {
            gradient[atom] += repulsion_gradient[atom];
        }
    }

    if options.include_dispersion && options.electronic.enable_dispersion {
        let _profile = crate::profile::scope("gradient.dispersion");
        let dispersion = if options.electronic.experimental_d4 {
            d4_dispersion_energy_gradient(
                system,
                params,
                &electronic.atomic_charges,
                options.electronic.d4_dispersion_options(),
            )?
        } else {
            dispersion_energy_gradient(
                system,
                params,
                options.electronic.d3_reference_path.as_deref(),
            )?
        };
        dispersion_gradient = dispersion.gradient;
        for atom in 0..nat {
            gradient[atom] += dispersion_gradient[atom];
        }
    }

    if options.include_halogen {
        let _profile = crate::profile::scope("gradient.halogen");
        let halogen = halogen_energy_gradient(system, params)?;
        halogen_gradient = halogen.gradient;
        for atom in 0..nat {
            gradient[atom] += halogen_gradient[atom];
        }
    }

    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    let max_gradient = gradient
        .iter()
        .map(|g| g.x.abs().max(g.y.abs()).max(g.z.abs()))
        .fold(0.0, f64::max);
    Ok(AnalyticGradientResult {
        total_energy: electronic.total_free,
        electronic_result: electronic,
        gradient,
        forces,
        electronic_gradient,
        repulsion_gradient,
        dispersion_gradient,
        halogen_gradient,
        max_gradient,
    })
}

fn electronic_gradient_terms(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
    options: &AnalyticGradientOptions,
) -> Result<Vec<Vec3>> {
    let nat = system.atoms.len();
    let basis = &result.basis;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut d_edcn = vec![0.0; nat];

    let mut self_energy = vec![0.0; basis.shells.len()];
    let mut dsedcn = vec![0.0; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        dsedcn[ish] = if options.electronic.hamiltonian.enable_cn_hamiltonian {
            -shell.kcn_raw.unwrap_or(0.0)
        } else {
            0.0
        };
        self_energy[ish] =
            shell.hdiag_ha + dsedcn[ish] * result.coordination_numbers[shell.atom_index];
    }

    let mut ao_potential = vec![0.0; basis.len()];
    if options.include_scc {
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                ao_potential[iao] = result.shell_scc_potential[ish];
            }
        }
    }

    // Spin-polarized GFN1 ("spGFN1"): the spin potential `V^s_l = Σ_{l'} W_{ll'} m_{l'}` enters the
    // per-spin Fock with opposite sign for α/β, so its overlap-derivative (Pulay) contribution to
    // the gradient is `−P_spin·(V^s_i + V^s_j)·dS/dR`, exactly analogous to the charge potential's
    // `−P_total·(v_i + v_j)·dS/dR`. `result.density` / `energy_weighted_density` already hold the
    // α+β totals (so the H0 band and charge-potential terms below are unchanged); here we add the
    // spin channel. No extra `dE_spin/dR` is needed: at the SCC minimum the energy is variational,
    // and `E_spin`'s only explicit position dependence is through the overlap in the Mulliken
    // magnetization, which is precisely this Pulay term.
    let spin_density: Option<crate::linalg::Matrix> = match (&result.spin, options.include_scc) {
        (Some(s), true) => Some({
            let mut sp = s.density_alpha.clone();
            let a = sp.as_mut_slice();
            let b = s.density_beta.as_slice();
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x -= *y;
            }
            sp
        }),
        _ => None,
    };
    let spin_ao_potential: Vec<f64> = match (&result.spin, options.include_scc) {
        (Some(s), true) => {
            let mut v = vec![0.0; basis.len()];
            for (ish, shell) in basis.shells.iter().enumerate() {
                for iao in shell.first_ao..shell.first_ao + shell.nao {
                    v[iao] = s.shell_spin_potential[ish];
                }
            }
            v
        }
        _ => Vec::new(),
    };
    let spin_on = spin_density.is_some();

    // DFT+U/+U+V overlap-Pulay weight W_{+U} = Σ_σ ½(P^σ Ṽ^σ + Ṽ^σ P^σ) = ∂E_{+U+V}/∂S.
    // The only explicit geometry dependence of the +U+V energy is through the overlap
    // (the dual population n = ½(PS+SP)); the density response is already carried by the
    // energy-weighted density (the +U Fock potential G shifts the converged orbitals). Only
    // off-site elements survive the dS/dR contraction (same-atom dS = 0).
    let plus_u_weight: Option<crate::linalg::Matrix> = match (&result.spin, options.include_scc) {
        (Some(s), true) if !s.plus_u_subspace.is_empty() => {
            let ovl = &result.integrals.overlap;
            let qa = crate::plus_u::plus_u_v_overlap_weight(&s.density_alpha, ovl, &s.plus_u_subspace, &s.plus_u_pairs);
            let qb = crate::plus_u::plus_u_v_overlap_weight(&s.density_beta, ovl, &s.plus_u_subspace, &s.plus_u_pairs);
            let nn = qa.rows();
            let mut w = qa;
            for i in 0..nn {
                for j in 0..nn {
                    w[(i, j)] += qb[(i, j)];
                }
            }
            Some(w)
        }
        _ => None,
    };
    let plus_u_on = plus_u_weight.is_some();

    if options.include_hamiltonian || options.include_scc {
        let mut atom_shell_ranges = vec![(0, 0); nat];
        for (sh_idx, sh) in basis.shells.iter().enumerate() {
            let a = sh.atom_index;
            if atom_shell_ranges[a].1 == 0 {
                atom_shell_ranges[a].0 = sh_idx;
            }
            atom_shell_ranges[a].1 += 1;
        }

        let cutoff = options.electronic.hamiltonian.integral_cutoff;
        let pairs = crate::pairlist::unique_short_range_pairs(system, cutoff)?;

        // Per-shell atomic radii precomputed once (element-only), so the per-pair work below is
        // `?`-free and can run in parallel. Each pair's contribution is accumulated locally and
        // reduced serially in pair order — deterministic, and equal to the serial result up to
        // the floating-point reassociation of the per-atom sum (well within FD-gate tolerance).
        let shell_radius: Vec<f64> = basis
            .shells
            .iter()
            .map(|s| atomic_radius_bohr(s.z))
            .collect::<Result<Vec<_>>>()?;
        let cn_on = options.electronic.hamiltonian.enable_cn_hamiltonian;
        let scc_on = options.include_scc;

        let contributions: Vec<(usize, usize, Vec3, Vec3, f64, f64)> = pairs
            .par_iter()
            .map(|pair| -> Result<(usize, usize, Vec3, Vec3, f64, f64)> {
                let atom_nu = pair.i; // i < j in unique_short_range_pairs
                let atom_mu = pair.j;
                let rmu = system.atoms[atom_mu].position;
                let rnu = system.atoms[atom_nu].position;
                let rvec = rmu - rnu;
                let r2 = rvec.norm2();
                let mut g_mu = Vec3::zero();
                let mut g_nu = Vec3::zero();
                let mut dc_mu = 0.0_f64;
                let mut dc_nu = 0.0_f64;
                if r2 > DIST_EPS {
                    let (first_sh_mu, n_sh_mu) = atom_shell_ranges[atom_mu];
                    let (first_sh_nu, n_sh_nu) = atom_shell_ranges[atom_nu];
                    for shell_mu_index in first_sh_mu..first_sh_mu + n_sh_mu {
                        let shell_mu = &basis.shells[shell_mu_index];
                        let rad_mu = shell_radius[shell_mu_index];
                        for shell_nu_index in first_sh_nu..first_sh_nu + n_sh_nu {
                            let shell_nu = &basis.shells[shell_nu_index];
                            let rad_nu = shell_radius[shell_nu_index];

                            let hs = hscale(shell_mu, shell_nu, params)?
                                * shell_polynomial(
                                    shell_mu,
                                    shell_nu,
                                    (r2.sqrt() / (rad_mu + rad_nu)).sqrt(),
                                );
                            let hij = 0.5
                                * (self_energy[shell_mu_index] + self_energy[shell_nu_index])
                                * hs;
                            let dlog_poly = shell_polynomial_log_derivative_precomputed(
                                shell_mu,
                                shell_nu,
                                rvec,
                                r2,
                                rad_mu + rad_nu,
                            );

                            for mu in shell_mu.first_ao..shell_mu.first_ao + shell_mu.nao {
                                for nu in shell_nu.first_ao..shell_nu.first_ao + shell_nu.nao {
                                    let (moments, d_bra, d_ket) = contracted_pair_with_derivatives(
                                        &basis.aos[mu],
                                        &basis.aos[nu],
                                        rmu,
                                        rnu,
                                    );
                                    let overlap = moments[0];

                                    let p = result.density[(mu, nu)];
                                    let w = result.energy_weighted_density[(mu, nu)];
                                    let scalar_shift = if scc_on {
                                        ao_potential[mu] + ao_potential[nu]
                                    } else {
                                        0.0
                                    };
                                    let mut overlap_coeff = p * (2.0 * hij - scalar_shift) - 2.0 * w;
                                    // spGFN1 spin Pulay term. The per-spin effective shell potential
                                    // is `u^σ = v^c ∓ V^s` (α: −V^s, β: +V^s; see `crate::spin`),
                                    // so the overlap-derivative contribution summed over spin is
                                    // `−½ Σ_σ P^σ(u^σ_μ+u^σ_ν)dS = −½ P_total v^c dS
                                    //  + ½ P_spin(V^s_μ+V^s_ν)dS` (P_spin = P^α − P^β). The charge
                                    // half is already in `scalar_shift`; here we add the spin half,
                                    // which (per the `coeff = 2·X` convention of this loop) is
                                    // `+P_spin·(V^s_μ + V^s_ν)`.
                                    if spin_on {
                                        let p_spin = spin_density.as_ref().unwrap()[(mu, nu)];
                                        overlap_coeff +=
                                            p_spin * (spin_ao_potential[mu] + spin_ao_potential[nu]);
                                    }
                                    // DFT+U/+U+V explicit overlap-Pulay term +Tr(W_{+U} dS).
                                    // The loop's `coeff = 2·X` convention (unordered atom pair
                                    // visited once, symmetric (μν)+(νμ)) gives `+2·W_{+U,μν}`.
                                    if plus_u_on {
                                        overlap_coeff +=
                                            2.0 * plus_u_weight.as_ref().unwrap()[(mu, nu)];
                                    }
                                    g_mu += d_bra[0] * overlap_coeff;
                                    g_nu += d_ket[0] * overlap_coeff;

                                    let hp = p * hij;
                                    let poly_grad = dlog_poly * (2.0 * hp * overlap);
                                    g_mu += poly_grad;
                                    g_nu -= poly_grad;

                                    if cn_on {
                                        dc_mu += dsedcn[shell_mu_index] * hs * p * overlap;
                                        dc_nu += dsedcn[shell_nu_index] * hs * p * overlap;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok((atom_mu, atom_nu, g_mu, g_nu, dc_mu, dc_nu))
            })
            .collect::<Result<Vec<_>>>()?;

        for (atom_mu, atom_nu, g_mu, g_nu, dc_mu, dc_nu) in contributions {
            gradient[atom_mu] += g_mu;
            gradient[atom_nu] += g_nu;
            d_edcn[atom_mu] += dc_mu;
            d_edcn[atom_nu] += dc_nu;
        }

        if options.electronic.hamiltonian.enable_cn_hamiltonian {
            for (ish, shell) in basis.shells.iter().enumerate() {
                for iao in shell.first_ao..shell.first_ao + shell.nao {
                    d_edcn[shell.atom_index] += dsedcn[ish] * result.density[(iao, iao)];
                }
            }
            let cn = coordination_with_derivatives(
                system,
                CoordinationOptions {
                    cutoff: options.electronic.hamiltonian.coordination_cutoff,
                    ..CoordinationOptions::default()
                },
            )?;
            for pair in cn.pairs {
                if pair.i == pair.j {
                    continue;
                }
                let r = pair.r_ij.norm();
                if r <= DIST_EPS {
                    continue;
                }
                let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
                let gi = pair.r_ij * pref;
                gradient[pair.i] += gi;
                gradient[pair.j] -= gi;
            }
        }
    }

    if options.include_scc {
        let shell_model = ShellChargeModel::build(system, basis, params)?;
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
                if r <= DIST_EPS {
                    continue;
                }
                let gamma = harmonic_average(shell_model.hardness[i], shell_model.hardness[j]);
                let dkernel = effective_kernel_derivative_vector(rvec, gamma);
                let scale = result.shell_charges[i] * result.shell_charges[j];
                gradient[ai] += dkernel * scale;
                gradient[aj] -= dkernel * scale;
            }
        }
    }

    // Explicit external electric-field term: with the field site potential
    // v_ext_i = -E·R_i folded into `shell_scc_potential` (so the overlap-derivative
    // "scalar_shift" already carries -P·v_ext·S'), the remaining explicit piece of
    // dE_field/dR at fixed charges is sum_i q_i dv_ext_i/dR_A = -q_A E.
    if options.include_scc {
        if let Some(field) = options.electronic.external_field.electric_field {
            for (atom, &q) in result.atomic_charges.iter().enumerate() {
                gradient[atom] -= field * q;
            }
        }
    }

    Ok(gradient)
}

/// DFT+U/+U+V **consistent-force correction** for the linear-response path: the
/// contribution from the explicit geometry dependence of the recomputed Hubbard
/// parameters `U(R)`, `V(R)`,
///
/// ```text
/// F_corr = Σ_I (∂E/∂U_I)(dU_I/dR) + Σ_{pairs} (∂E/∂V_IJ)(dV_IJ/dR) .
/// ```
///
/// The Hellmann–Feynman partials `∂E/∂U_I = ½ Σ_σ Tr[n^σ_I(1−n^σ_I)]`,
/// `∂E/∂V_IJ = −Σ_σ Tr[n^σ_AB n^σ_BA]` are evaluated **analytically** at the
/// converged α/β densities ([`crate::plus_u::plus_u_param_derivatives`], Stage 1,
/// FD-verified). Because `U`/`V` are explicit parameters of the energy (not
/// variational quantities), this is the exact missing term — at the +U-SCC
/// minimum the implicit `dP/dU` drops out (the frozen-U force already carries the
/// density response through the energy-weighted density).
///
/// The geometry derivatives `dU_I/dR`, `dV_IJ/dR` are **analytic**
/// ([`crate::plus_u_dudr::analytic_dudr`]): the SCC-CPHF geometry response of the
/// linear-response `χ0`/`χ`. It solves the coupled-perturbed **geometry** response of
/// the base state — `dC^σ/dR`, `dε^σ/dR`, `df^σ/dR` — for the **spin-unrestricted,
/// finite-temperature** two-channel base MO bases (the orbital-geometry response
/// accounts for ~94% of `dχ0/dR`), differentiates the bare (`χ0^x`) and screened
/// (`χ^x`, via the `(I−M)u^x = b^x + M^x u` fixed point) response matrices, and
/// assembles `K^x = −X0 χ0^x X0 + X χ^x X` with the same Tikhonov-regularized inverses
/// [`crate::plus_u::extract_uv_from_response`] uses, so `U^x_I = K^x_II`,
/// `V^x_IJ = −K^x_IJ` match the FD-differenced extraction. Every stage (`P^x`, `χ0^x`,
/// `χ^x`, `K^x`) is FD-gated at `electronic_temperature = 300 K` on ScH; the assembled
/// `dU/dR` matches the FD oracle
/// ([`crate::spin::linear_response_uv_for_system`], differenced) to ~1e-9. Returns the
/// per-atom gradient contribution (force = −gradient).
fn plus_u_consistency_gradient_terms(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
    options: &AnalyticGradientOptions,
) -> Result<Vec<Vec3>> {
    let nat = system.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];
    // Need the converged spin densities and the reference subspace/pairs (with the U applied).
    let Some(spin) = &result.spin else {
        return Ok(gradient);
    };
    let subspace = &spin.plus_u_subspace;
    if subspace.is_empty() {
        return Ok(gradient);
    }
    let pairs = &spin.plus_u_pairs;
    let ovl = &result.integrals.overlap;

    // ∂E/∂U_I (summed over the two spin channels) and ∂E/∂V_IJ = −Σ_σ Σ(n_IJ)².
    let (du_a, dv_a) = crate::plus_u::plus_u_param_derivatives(&spin.density_alpha, ovl, subspace, pairs);
    let (du_b, dv_b) = crate::plus_u::plus_u_param_derivatives(&spin.density_beta, ovl, subspace, pairs);
    let de_du: Vec<f64> = du_a.iter().zip(du_b.iter()).map(|(a, b)| a + b).collect();
    // dv_* are +Σ(n_IJ)² per channel; ∂E/∂V = −(dv_a + dv_b).
    let de_dv: Vec<f64> = dv_a.iter().zip(dv_b.iter()).map(|(a, b)| -(a + b)).collect();

    // dU_I/dR, dV_IJ/dR **analytically** via the SCC-CPHF geometry response of the linear-
    // response χ0/χ (see [`crate::plus_u_dudr`]): per Cartesian DOF, `dU/dR` is aligned to
    // `subspace` (by atom_index) and `dV/dR` to `pairs` (by endpoint atom_index). The consistency
    // force contracts these with the analytic `∂E/∂U`, `∂E/∂V` (`de_du`, `de_dv`). Gated against
    // the FD oracle (`plus_u_dudr::tests::analytic_dudr_matches_fd`, ~1e-9 on ScH 300 K).
    let Some((du_dr, dv_dr)) =
        crate::plus_u_dudr::analytic_dudr(system, params, &options.electronic, subspace, pairs)?
    else {
        return Ok(gradient);
    };
    for atom in 0..nat {
        for axis in 0..3 {
            let dof = 3 * atom + axis;
            let mut g = 0.0;
            for i in 0..de_du.len() {
                g += de_du[i] * du_dr[dof][i];
            }
            for k in 0..de_dv.len() {
                g += de_dv[k] * dv_dr[dof][k];
            }
            match axis {
                0 => gradient[atom].x += g,
                1 => gradient[atom].y += g,
                _ => gradient[atom].z += g,
            }
        }
    }
    Ok(gradient)
}

/// Analytic gradient of the experimental mDFTB2 multipole correction (non-PBC), evaluated at
/// the converged density. Variational, so only the explicit position derivatives are needed:
/// (i) the off-site kernel `df^(mn)/dR` contracted with the fixed atomic moment pairs, and
/// (ii) the overlap-Pulay term `Σ_{κν} W_{κν} dS_{κν}/dR` with `W = ∂E_mp/∂S`. The on-site
/// `d̄`/`Q̄` and the on-site kernels translate rigidly, contributing no derivative. Added on
/// top of the GFN1 electronic gradient (whose energy-weighted-density Pulay already carries
/// the multipole shift through the converged eigenvalues).
fn multipole_gradient_terms(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
    options: &AnalyticGradientOptions,
) -> Result<Vec<Vec3>> {
    let nat = system.atoms.len();
    let basis = &result.basis;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let integrals = IntegralMatrices::build(system, basis)?;
    // Atomic Klopman-Ohno hardness η_A (the s-shell hardness) and positions.
    let hardness: Vec<f64> = (0..nat)
        .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
        .collect();
    let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
    // mDFTB monopole Δq_A = (population − reference) = −(GFN1 atomic charge).
    let mut q = vec![0.0_f64; nat];
    for (sh, shell) in basis.shells.iter().enumerate() {
        q[shell.atom_index] -= result.shell_charges[sh];
    }

    // CAMM-on-mDFTB2 (v0.4.2): the off-site anisotropy is the GFN2-style AES on cumulative
    // moments — a different kernel (erf-cloud) and moment source than the mDFTB routines below.
    // Its analytic gradient is fully self-contained (kernel force + cumulative-moment integral
    // derivatives); the implicit density response rides the base energy-weighted-density Pulay.
    if options.electronic.multipole_model == crate::electronic::MultipoleModel::CammOnMdftb2 {
        if options.electronic.camm_damp_charge.is_some() {
            return Err(crate::error::Gfn1Error::InvalidInput(
                "analytic forces for charge-dependent CAMM κ (camm_damp_charge) are not yet \
                 implemented (the κ(q) self-consistency adds ∂κ/∂q terms); single-point energies \
                 are supported"
                    .to_string(),
            ));
        }
        // Per-atom κ: element override if present, else the global camm_damp.
        let camm_kappa: Vec<f64> = system
            .atoms
            .iter()
            .map(|atom| {
                options
                    .electronic
                    .camm_damp_elem
                    .iter()
                    .find(|&&(z, _)| z == atom.z)
                    .map(|&(_, k)| k)
                    .unwrap_or(options.electronic.camm_damp)
            })
            .collect();
        // Per-atom on-site penalty scale: element override if present, else global s_onsite.
        let camm_onsite: Vec<f64> = system
            .atoms
            .iter()
            .map(|atom| {
                options
                    .electronic
                    .camm_onsite_scale_elem
                    .iter()
                    .find(|&&(z, _)| z == atom.z)
                    .map(|&(_, s)| s)
                    .unwrap_or(options.electronic.camm_onsite_scale)
            })
            .collect();
        return Ok(crate::multipole::camm_aes_gradient(
            basis,
            nat,
            &hardness,
            &atom_pos,
            &integrals,
            &result.density,
            &q,
            &camm_kappa,
            options.electronic.camm_aes_scale,
            &camm_onsite,
        ));
    }

    // Stage 5: optional richer secondary-basis on-site moment integrals — built identically to
    // the SCC so the analytic gradient stays consistent with the (enriched) energy.
    let secondary_aos: Option<Vec<crate::basis::AOBasisFunction>> = options
        .electronic
        .multipole_secondary_basis
        .as_ref()
        .map(|sec| crate::magnetic::build_secondary_aos(basis, system, sec));
    let secondary_moment_ints: Option<IntegralMatrices> = secondary_aos.as_ref().map(|sec_aos| {
        crate::multipole::secondary_moment_integrals(&integrals, basis, nat, &atom_pos, sec_aos)
    });
    let mp_ints: &IntegralMatrices = secondary_moment_ints.as_ref().unwrap_or(&integrals);

    // Geometry-fixed on-site octupole AO integrals, built once for both octupole gradient terms.
    let octu_cache = if options.electronic.multipole_octupole {
        Some(crate::multipole::OnsiteOctupoleCache::build(
            basis, nat, &atom_pos,
        ))
    } else {
        None
    };

    // (i) off-site kernel forces + (ii) overlap-Pulay weight `W = ∂E_mp/∂S`. The arbitrary-rank
    // generic path (`multipole_order ≥ 4`) supersedes the dipole/quad/octupole-specific routines
    // with the unified rank loop; otherwise the legacy (speed-optimized) paths run unchanged.
    // Per-rank multipole×charge cross terms also force the generic path (mirrors `run_electronic`),
    // so the analytic gradient's overlap-Pulay weight stays consistent with the cross-term energy.
    let multipole_charge_cross = !options.electronic.multipole_charge_order.is_empty();
    let generic_rank: Option<usize> =
        if options.electronic.multipole_order >= 4 || multipole_charge_cross {
            Some(options.electronic.multipole_order)
        } else {
            None
        };
    let (mut grad, mut w) = if let Some(l) = generic_rank {
        // Geometry-fixed on-site rank-`l` AO moment integrals, built once for the moments + weight
        // (over the secondary AOs when supplied, so the gradient matches the secondary-enriched
        // generic energy — consistent with the SCC `generic_moment_cache`).
        let moment_cache = crate::multipole::OnsiteMomentCache::build_with_aos(
            basis,
            nat,
            &atom_pos,
            l,
            secondary_aos.as_deref(),
        );
        let moments = crate::multipole::build_generic_moments(
            basis,
            nat,
            &atom_pos,
            mp_ints,
            &result.density,
            &q,
            l,
            Some(&moment_cache),
        );
        let g = crate::multipole::multipole_kernel_forces_generic(
            nat, &hardness, &atom_pos, &moments, l,
        );
        // Combined generic multipole + per-rank charge-cross overlap-Pulay weight `W = ∂E/∂S` (one
        // shared shift assembly; the cross block is skipped internally when the order vec is empty).
        let gam3: Vec<f64> = if multipole_charge_cross {
            (0..nat)
                .map(|a| shell_model.hubbard_derivs[shell_model.atom_offsets[a]])
                .collect()
        } else {
            Vec::new()
        };
        let ww = crate::multipole::multipole_overlap_weight_generic_with_cross(
            basis,
            nat,
            &hardness,
            &gam3,
            &atom_pos,
            &result.density,
            &moments,
            &q,
            &options.electronic.multipole_charge_order,
            l,
            Some(&moment_cache),
        );
        (g, ww)
    } else {
        // (i) Off-site kernel forces (fixed moments).
        let mut grad = crate::multipole::multipole_kernel_forces(
            basis,
            nat,
            &hardness,
            &atom_pos,
            mp_ints,
            &result.density,
            &q,
        );
        if options.electronic.multipole_octupole {
            let og = crate::multipole::octupole_kernel_forces(
                basis,
                nat,
                &hardness,
                &atom_pos,
                mp_ints,
                &result.density,
                &q,
                octu_cache.as_ref(),
            );
            for a in 0..nat {
                grad[a] += og[a];
            }
        }
        // (ii) Overlap-Pulay term: W = ∂E_mp/∂S contracted with dS/dR. Only inter-atomic AO
        // pairs have a nonzero overlap derivative (same-atom blocks translate rigidly). Each
        // unordered atom pair is visited once, so the symmetric (κν)+(νκ) sum gives a factor 2.
        let mut w = crate::multipole::multipole_overlap_weight(
            basis,
            nat,
            &hardness,
            &atom_pos,
            mp_ints,
            &result.density,
            &q,
        );
        if options.electronic.multipole_octupole {
            let ow = crate::multipole::octupole_overlap_weight(
                basis,
                nat,
                &hardness,
                &atom_pos,
                mp_ints,
                &result.density,
                &q,
                octu_cache.as_ref(),
            );
            let nn = basis.len();
            for i in 0..nn {
                for j in 0..nn {
                    w[(i, j)] += ow[(i, j)];
                }
            }
        }
        (grad, w)
    };
    if options.electronic.field_multipole {
        // Stage 3: the field–dipole energy's explicit dS/dR term (W = ∂E_field^dip/∂S). The
        // uniform field has no off-site kernel, so there is no field-dipole kernel-force term.
        if let Some(field) = options.electronic.external_field.electric_field {
            let fw = crate::multipole::field_dipole_overlap_weight(
                basis,
                nat,
                mp_ints,
                &result.density,
                field,
            );
            let nn = basis.len();
            for i in 0..nn {
                for j in 0..nn {
                    w[(i, j)] += fw[(i, j)];
                }
            }
        }
    }
    if options.electronic.multipole_third_order && generic_rank.is_none() {
        // Third-order on-site multipole energy's explicit dS/dR term (W = ∂E³/∂S). On-site only
        // (no off-site kernel-force), exactly like the second-order overlap-Pulay weight.
        // (Superseded by the generic path, which does not include the third-order cross terms.)
        let gam3: Vec<f64> = (0..nat)
            .map(|a| shell_model.hubbard_derivs[shell_model.atom_offsets[a]])
            .collect();
        let tw = crate::multipole::third_order_overlap_weight(
            basis,
            nat,
            &hardness,
            &gam3,
            mp_ints,
            &result.density,
            &q,
        );
        let nn = basis.len();
        for i in 0..nn {
            for j in 0..nn {
                w[(i, j)] += tw[(i, j)];
            }
        }
    }
    let cutoff = options.electronic.hamiltonian.integral_cutoff;
    let pairs = crate::pairlist::unique_short_range_pairs(system, cutoff)?;
    let mut atom_shell_ranges = vec![(0usize, 0usize); nat];
    for (sh_idx, sh) in basis.shells.iter().enumerate() {
        let a = sh.atom_index;
        if atom_shell_ranges[a].1 == 0 {
            atom_shell_ranges[a].0 = sh_idx;
        }
        atom_shell_ranges[a].1 += 1;
    }
    for pair in pairs {
        let atom_nu = pair.i;
        let atom_mu = pair.j;
        let rmu = system.atoms[atom_mu].position;
        let rnu = system.atoms[atom_nu].position;
        if (rmu - rnu).norm2() <= DIST_EPS {
            continue;
        }
        let (first_sh_mu, n_sh_mu) = atom_shell_ranges[atom_mu];
        let (first_sh_nu, n_sh_nu) = atom_shell_ranges[atom_nu];
        for shell_mu_index in first_sh_mu..first_sh_mu + n_sh_mu {
            let shell_mu = &basis.shells[shell_mu_index];
            for shell_nu_index in first_sh_nu..first_sh_nu + n_sh_nu {
                let shell_nu = &basis.shells[shell_nu_index];
                for mu in shell_mu.first_ao..shell_mu.first_ao + shell_mu.nao {
                    for nu in shell_nu.first_ao..shell_nu.first_ao + shell_nu.nao {
                        let (_m, d_bra, d_ket) = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let weight = 2.0 * w[(mu, nu)];
                        grad[atom_mu] += d_bra[0] * weight;
                        grad[atom_nu] += d_ket[0] * weight;
                    }
                }
            }
        }
    }
    Ok(grad)
}

/// Analytic gradient of the experimental long-range Fock exchange (MFX), non-PBC. The total energy
/// is variational at the SCF solution, so only the **explicit** nuclear derivatives of
/// `E_x = ½Tr[ΔP K[ΔP]]` appear here: (i) the off-site kernel force through `Γ(R)` and (ii) the
/// overlap-Pulay term `∂E_x/∂S · dS/dR`. The implicit density response is carried by the base
/// band-structure energy-weighted-density Pulay term (the exchange Fock is in the converged Fock
/// that builds the energy-weighted density). `ΔP = P − P0` with the geometry-independent
/// neutral-atom reference `P0`.
fn exchange_gradient_terms(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &ElectronicResult,
    options: &AnalyticGradientOptions,
) -> Result<Vec<Vec3>> {
    use crate::coulomb::{
        exchange_sigma_pair, local_size_factor_from_cn, lr_gamma_exchange_deriv,
        lr_gamma_exchange_omega_deriv, omega_hardness_pairwise, omega_local_geometry,
        omega_pair_local_geometry, OmegaScheme,
    };
    let nat = system.atoms.len();
    let basis = &result.basis;
    let shell_model = ShellChargeModel::build(system, basis, params)?;
    let integrals = IntegralMatrices::build(system, basis)?;
    let s = &integrals.overlap;
    let hardness: Vec<f64> = (0..nat)
        .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
        .collect();
    let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
    let n = basis.len();
    // ΔP = P − P0 (neutral-atom reference; geometry-independent).
    let p0 = crate::exchange::neutral_atom_reference_density(basis);
    let mut dp = crate::linalg::Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            dp[(i, j)] = result.density[(i, j)] - p0[(i, j)];
        }
    }
    // LocalGeometry (dynamic ω): per-atom size factor `s_A = (1+CN_A)^(−1/3)` from the GFN1
    // coordination number (same cutoff as the H⁰ CN); `None` ⇒ geometry-independent HardnessPairwise.
    let dynamic = options.electronic.dynamic_omega;
    let cn_deriv = if dynamic {
        Some(coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: options.electronic.hamiltonian.coordination_cutoff,
                ..CoordinationOptions::default()
            },
        )?)
    } else {
        None
    };
    let s_factor: Vec<f64> = match &cn_deriv {
        Some(c) => {
            c.cn.iter()
                .map(|&v| local_size_factor_from_cn(v).0)
                .collect()
        }
        None => vec![1.0; nat],
    };
    let omega_atom = |a: usize| -> f64 {
        if dynamic {
            omega_local_geometry(hardness[a], s_factor[a])
        } else {
            hardness[a]
        }
    };
    let omega_ab = |a: usize, b: usize| -> f64 {
        if dynamic {
            omega_pair_local_geometry(hardness[a], s_factor[a], hardness[b], s_factor[b])
        } else {
            omega_hardness_pairwise(hardness[a], hardness[b])
        }
    };
    let gamma = if dynamic {
        crate::exchange::lr_exchange_gamma_matrix_local(basis, nat, &atom_pos, &hardness, &s_factor)
    } else {
        crate::exchange::lr_exchange_gamma_matrix(
            basis,
            nat,
            &atom_pos,
            &hardness,
            OmegaScheme::HardnessPairwise,
        )
    };
    let mut grad = vec![Vec3::zero(); nat];

    // (i) Off-site kernel force from dΓ/dR. Aggregate the AO-pair Γ-weight `∂E_x/∂Γ` to atom pairs;
    // every AO pair on atoms (A,B) shares the same `γ^lr_{AB}(R_AB)`, so the force on A is
    // `Σ_{B≠A} (G_AB + G_BA) · γ^lr'(R_AB) · (R_A−R_B)/R_AB`.
    let wg = crate::exchange::mfx_gamma_weight(&dp, s);
    let atom_of: Vec<usize> = basis.aos.iter().map(|ao| ao.atom_index).collect();
    let mut gab = vec![0.0_f64; nat * nat];
    for mu in 0..n {
        for nu in 0..n {
            gab[atom_of[mu] * nat + atom_of[nu]] += wg[(mu, nu)];
        }
    }
    for a in 0..nat {
        for b in 0..nat {
            if a == b {
                continue;
            }
            let x = atom_pos[a] - atom_pos[b];
            let r = x.norm();
            if r <= DIST_EPS {
                continue;
            }
            let sigma = exchange_sigma_pair(hardness[a], hardness[b]);
            let omega = omega_ab(a, b);
            let gp = lr_gamma_exchange_deriv(r, sigma, omega);
            let w = gab[a * nat + b] + gab[b * nat + a];
            grad[a] += x * (w * gp / r);
        }
    }

    // (ii) Overlap-Pulay term: Σ_{μν} (∂E_x/∂S)_{μν} dS_{μν}/dR (inter-atomic AO pairs only; same
    // assembly as the multipole overlap weight — each unordered pair once, factor 2 for (μν)+(νμ)).
    let ws = crate::exchange::mfx_overlap_weight(&dp, s, &gamma);
    let cutoff = options.electronic.hamiltonian.integral_cutoff;
    let pairs = crate::pairlist::unique_short_range_pairs(system, cutoff)?;
    let mut atom_shell_ranges = vec![(0usize, 0usize); nat];
    for (sh_idx, sh) in basis.shells.iter().enumerate() {
        let a = sh.atom_index;
        if atom_shell_ranges[a].1 == 0 {
            atom_shell_ranges[a].0 = sh_idx;
        }
        atom_shell_ranges[a].1 += 1;
    }
    for pair in pairs {
        let atom_nu = pair.i;
        let atom_mu = pair.j;
        let rmu = system.atoms[atom_mu].position;
        let rnu = system.atoms[atom_nu].position;
        if (rmu - rnu).norm2() <= DIST_EPS {
            continue;
        }
        let (first_sh_mu, n_sh_mu) = atom_shell_ranges[atom_mu];
        let (first_sh_nu, n_sh_nu) = atom_shell_ranges[atom_nu];
        for shell_mu_index in first_sh_mu..first_sh_mu + n_sh_mu {
            let shell_mu = &basis.shells[shell_mu_index];
            for shell_nu_index in first_sh_nu..first_sh_nu + n_sh_nu {
                let shell_nu = &basis.shells[shell_nu_index];
                for mu in shell_mu.first_ao..shell_mu.first_ao + shell_mu.nao {
                    for nu in shell_nu.first_ao..shell_nu.first_ao + shell_nu.nao {
                        let (_m, d_bra, d_ket) = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let weight = 2.0 * ws[(mu, nu)];
                        grad[atom_mu] += d_bra[0] * weight;
                        grad[atom_nu] += d_ket[0] * weight;
                    }
                }
            }
        }
    }

    // (iii) ω-reorganisation force (LocalGeometry only): the dynamic `ω_AB` moves with the geometry
    // through the coordination number, so `∂E_x/∂R_C = Σ_A (∂E_x/∂CN_A)(∂CN_A/∂R_C)` with
    // `∂E_x/∂CN_A = [Σ_{B≠A} (gab_AB+gab_BA)·∂γ^lr/∂ω(R_AB;ω_AB)·∂ω_AB/∂ω_A]·(∂ω_A/∂s_A)·(∂s_A/∂CN_A)`,
    // `ω_A = η_A/s_A ⇒ ∂ω_A/∂s_A = −η_A/s_A²`, and the CN-derivative assembly is the same as the H⁰ CN
    // force. `∂ω_AB/∂ω_A = 2 ω_B²/(ω_A+ω_B)²` (harmonic mean).
    if let Some(cn) = cn_deriv {
        let mut g_omega = vec![0.0_f64; nat]; // ∂E_x/∂ω_A
        for a in 0..nat {
            let wa = omega_atom(a);
            // Onsite (A=A) reorganisation: the MFX kernel's diagonal `γ^lr_AA(R=0; ω_AA)` depends on
            // `ω_AA = ω_A` (no R-force — even kernel — but `∂/∂ω ≠ 0`). `∂E_x/∂γ_AA = gab[A,A]`
            // (single AO-pair group), `∂ω_AA/∂ω_A = 1`.
            let sigma_aa = exchange_sigma_pair(hardness[a], hardness[a]);
            g_omega[a] += gab[a * nat + a] * lr_gamma_exchange_omega_deriv(0.0, sigma_aa, wa);
            for b in 0..nat {
                if a == b {
                    continue;
                }
                let x = atom_pos[a] - atom_pos[b];
                let r = x.norm();
                if r <= DIST_EPS {
                    continue;
                }
                let wb = omega_atom(b);
                let denom = wa + wb;
                if denom <= 0.0 {
                    continue;
                }
                let sigma = exchange_sigma_pair(hardness[a], hardness[b]);
                let dg = lr_gamma_exchange_omega_deriv(r, sigma, omega_ab(a, b));
                let de_dwab = (gab[a * nat + b] + gab[b * nat + a]) * dg;
                let dwab_dwa = 2.0 * wb * wb / (denom * denom);
                g_omega[a] += de_dwab * dwab_dwa;
            }
        }
        // On-site exchange (OFX) reorganisation: its one-center kernel is screened at `ω_AA = ω_A`,
        // so `∂E_OFX/∂ω_A` (refined ERIs + Mulliken `γ_AA`) folds straight into `∂E/∂ω_A`.
        if options.electronic.onsite_exchange {
            let omega_per_atom: Vec<f64> = (0..nat).map(|a| omega_atom(a)).collect();
            let de_ofx = crate::exchange::onsite_exchange_omega_energy_derivs(
                basis,
                nat,
                s,
                &dp,
                &hardness,
                &omega_per_atom,
            );
            for a in 0..nat {
                g_omega[a] += de_ofx[a];
            }
        }
        let mut d_edcn = vec![0.0_f64; nat]; // ∂E_x/∂CN_A
        for a in 0..nat {
            let (s_a, ds_a) = local_size_factor_from_cn(cn.cn[a]);
            let dwa_dsa = -hardness[a] / (s_a * s_a);
            d_edcn[a] = g_omega[a] * dwa_dsa * ds_a;
        }
        for pair in &cn.pairs {
            if pair.i == pair.j {
                continue;
            }
            let r = pair.r_ij.norm();
            if r <= DIST_EPS {
                continue;
            }
            let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
            let gi = pair.r_ij * pref;
            grad[pair.i] += gi;
            grad[pair.j] -= gi;
        }
    }
    Ok(grad)
}

fn shell_polynomial_log_derivative_precomputed(
    si: &crate::basis::BasisShell,
    sj: &crate::basis::BasisShell,
    rvec: Vec3,
    r2: f64,
    rad_sum: f64,
) -> Vec3 {
    let rr = (r2.sqrt() / rad_sum).sqrt();
    let pi = si.poly_raw.unwrap_or(0.0);
    let pj = sj.poly_raw.unwrap_or(0.0);
    let fi = 1.0 + pi * rr;
    let fj = 1.0 + pj * rr;
    let poly = fi * fj;
    if poly.abs() <= 1.0e-18 {
        return Vec3::zero();
    }
    let dpoly = (fi * pj + fj * pi) * 0.5 * rr / r2;
    rvec * (dpoly / poly)
}

pub(crate) fn effective_kernel_derivative_vector(rvec: Vec3, gamma: f64) -> Vec3 {
    let r = rvec.norm();
    if r <= DIST_EPS {
        return Vec3::zero();
    }
    let r2 = r * r;
    let denom = r2 + 1.0 / (gamma * gamma);
    let pref = -1.0 / (denom * denom.sqrt());
    rvec * pref
}

#[cfg(test)]
mod multipole_grad_tests {
    use super::*;

    fn load_params() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    fn displaced(system: &PeriodicSystem, atom: usize, axis: usize, delta: f64) -> PeriodicSystem {
        let mut s = system.clone();
        match axis {
            0 => s.atoms[atom].position.x += delta,
            1 => s.atoms[atom].position.y += delta,
            _ => s.atoms[atom].position.z += delta,
        }
        s
    }

    /// Primary correctness gate for Part B5: the full analytic gradient with the mDFTB2
    /// multipole correction on must match a central finite-difference of the total
    /// (GFN1 + multipole) energy. A slightly distorted, polarized water is used so the
    /// atomic dipole/quadrupole moments — and every multipole gradient term — are nonzero.
    #[test]
    fn multipole_analytic_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.78 0.55 -0.05\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "multipole analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// MFX gradient gate: with long-range exchange on, the full analytic gradient (exchange kernel
    /// force + overlap-Pulay weight, on top of the base band-structure Pulay term that carries the
    /// density response) must match a central finite-difference of the total (GFN1 + exchange)
    /// energy. Polarized water so the exchange and its gradient are nonzero.
    #[test]
    fn mfx_exchange_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.05 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.lr_exchange = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "MFX analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Dynamic-ω (LocalGeometry) gradient gate: with `dynamic_omega` on, the analytic exchange force
    /// must include the `∂ω/∂R` reorganisation term — the screening `ω_A = η_A·(1+CN_A)^(1/3)` moves
    /// with the coordination number — checked against a central finite difference of the total energy.
    /// Bonded water (CN > 0) makes the dynamic term nonzero.
    #[test]
    fn mfx_dynamic_omega_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.05 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.lr_exchange = true;
        opt.electronic.dynamic_omega = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "dynamic-ω analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// OFX + dynamic-ω gradient gate: with both `onsite_exchange` and `dynamic_omega` on, the
    /// analytic force must *also* include the on-site exchange's ω-reorganisation
    /// (`∂E_OFX/∂ω_A·∂ω_A/∂R` — the one-center ERIs and Mulliken `γ_AA` are screened at
    /// `ω_A = η_A(1+CN_A)^(1/3)`, which moves with the geometry). Checked vs a central finite
    /// difference of the OFX total energy.
    #[test]
    fn mfx_ofx_dynamic_omega_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.05 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.lr_exchange = true;
        opt.electronic.onsite_exchange = true;
        opt.electronic.dynamic_omega = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "OFX+dynamic-ω analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// OFX gradient gate. The on-site Fock-exchange correction is built from *one-center* integrals:
    /// the refined ERIs `(μκ|νλ)^lr` are translation-invariant (all four AOs on one atom), and the
    /// subtracted on-site Mulliken term uses `γ_AA^lr` (R=0, an element constant ⇒ `∂γ/∂R=0`) and
    /// intra-atomic `S_{μσ}` (rigid-translation invariant ⇒ `∂S/∂R=0`). So OFX adds **no explicit
    /// gradient term** — its entire effect on the forces is through the OFX-relaxed SCC density (which
    /// already feeds `result.density` and the energy-weighted density). This gate proves that claim:
    /// with both `lr_exchange` and `onsite_exchange` on, the *unchanged* base+MFX analytic gradient,
    /// evaluated at the OFX-converged result, still matches a finite difference of the OFX total
    /// energy. (If OFX did contribute an explicit force, this would fail.) Polar water keeps it fast.
    #[test]
    fn ofx_exchange_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.05 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.lr_exchange = true;
        opt.electronic.onsite_exchange = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "OFX analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// OFX is a genuine, *active* correction: turning `onsite_exchange` on (atop `lr_exchange`) must
    /// converge and shift the total energy relative to MFX-only — the exact one-center exchange and
    /// its Mulliken approximation differ. Confirms the SCC wiring is live (not a silent no-op).
    #[test]
    fn ofx_changes_energy_and_converges() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut mfx = crate::electronic::ElectronicOptions::default();
        mfx.lr_exchange = true;
        let mut ofx = mfx.clone();
        ofx.onsite_exchange = true;
        let r_mfx = run_electronic(&system, &params, mfx).unwrap();
        let r_ofx = run_electronic(&system, &params, ofx).unwrap();
        assert!(r_mfx.converged, "MFX-only SCC did not converge");
        assert!(r_ofx.converged, "MFX+OFX SCC did not converge");
        assert!(
            (r_mfx.total_free - r_ofx.total_free).abs() > 1.0e-6,
            "OFX should change the energy vs MFX-only: {} vs {}",
            r_mfx.total_free,
            r_ofx.total_free
        );
    }

    /// **M4.5 gate (real functional)** — the Trust-Region Augmented Hessian SCF driver converges the
    /// live exchange-augmented SCC (MFX+OFX) to the **same energy** as the commutator-DIIS driver on
    /// water. Proves TRAH is a working second-order SCF for the real functional (not just the model),
    /// reaching the same stationary point through orbital rotations instead of density mixing.
    #[test]
    fn trah_exchange_scf_matches_diis() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut diis = crate::electronic::ElectronicOptions::default();
        diis.lr_exchange = true;
        diis.onsite_exchange = true;
        let mut trah = diis.clone();
        trah.scf_trah = true;
        let r_diis = run_electronic(&system, &params, diis).unwrap();
        let r_trah = run_electronic(&system, &params, trah).unwrap();
        assert!(r_trah.converged, "TRAH exchange SCF did not converge");
        assert!(
            (r_diis.total_free - r_trah.total_free).abs() < 1.0e-5,
            "TRAH vs DIIS total energy: {} vs {}",
            r_diis.total_free,
            r_trah.total_free
        );
    }

    /// v0.2.0 **arbitrary-rank** end-to-end gate: with `multipole_order = 4` the unified generic
    /// SCC path (self-consistently mixing atomic moments of ranks 1..4) and its analytic gradient
    /// must match a central finite-difference of the total energy. HCl (Cl carries d) is used so
    /// the rank-3/4 moments are genuinely active, with a single off-site pair to keep the rank-8
    /// `f^(4,4)` contraction cost (3⁸ elements) tractable for the test suite. Verifies the generic
    /// SCC self-consistency + energy + gradient end-to-end (the per-term orchestration — and that
    /// generic == legacy for ranks ≤3 — is gated separately in `multipole`).
    #[test]
    fn multipole_order_4_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            PeriodicSystem::from_xyz_str("2\nHCl\nCl 0.0 0.0 0.0\nH 0.30 0.20 1.25\n", 0.0, false)
                .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_order = 4;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "order-4 multipole analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// With the correction off, the analytic gradient must be byte-for-byte the GFN1
    /// gradient (the multipole path is fully gated by the flag).
    #[test]
    fn multipole_off_gradient_unchanged() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.78 0.55 -0.05\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let g_off = analytic_gradient(&system, &params, AnalyticGradientOptions::default())
            .unwrap()
            .gradient;
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = false;
        let g_explicit = analytic_gradient(&system, &params, opt).unwrap().gradient;
        for (a, b) in g_off.iter().zip(&g_explicit) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
        }
    }

    const POLAR_WATER: &str = "3\nwater\nO 0.02 0.01 0.10\nH 0.78 0.55 -0.05\nH -0.74 0.58 0.03\n";

    /// Stage 1 gate: with the experimental 4th-order on-site charge term on, the analytic
    /// gradient (which carries *no* explicit 4th-order term — it rides the SCC-converged
    /// charges, since ∂E⁽⁴⁾/∂R|_q = 0) must still match a central finite-difference of the
    /// total energy. This simultaneously verifies the energy, the self-consistent
    /// potential, and the "no new gradient code" claim.
    #[test]
    fn fourth_order_charge_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.charge_order = 4;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "4th-order charge analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// 4th-order charge off ⇒ byte-for-byte the stock GFN1 gradient (fully flag-gated).
    #[test]
    fn fourth_order_charge_off_unchanged() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let g_off = analytic_gradient(&system, &params, AnalyticGradientOptions::default())
            .unwrap()
            .gradient;
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.charge_order = 3;
        let g_explicit = analytic_gradient(&system, &params, opt).unwrap().gradient;
        for (a, b) in g_off.iter().zip(&g_explicit) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
        }
    }

    /// Stage 3 gate: with the experimental field–dipole coupling on (multipole + an external
    /// electric field), the full analytic gradient must match a central finite-difference of
    /// the total energy. Exercises the field–dipole Fock (carried through the SCC density into
    /// the energy-weighted Pulay term) and the field–dipole overlap-Pulay weight added to
    /// `multipole_gradient_terms`.
    #[test]
    fn field_multipole_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.field_multipole = true;
        opt.electronic.external_field =
            crate::field::ExternalFieldOptions::electric(Vec3::new(0.012, -0.008, 0.005));
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "field-multipole analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Stage 3 field-coupling gate: the reported dipole (now including the atomic part Σ d_A)
    /// must equal `−∂E/∂E_field` by central finite difference of the field. This confirms the
    /// field–dipole energy term `−E·Σ d_A` is the exact conjugate of the physically complete
    /// dipole (Hellmann–Feynman at the variational SCC minimum).
    #[test]
    fn field_multipole_dipole_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let base = Vec3::new(0.010, -0.006, 0.004);
        let mut opt = ElectronicOptions::default();
        opt.multipole = true;
        opt.field_multipole = true;
        opt.external_field = crate::field::ExternalFieldOptions::electric(base);
        let dipole = run_electronic(&system, &params, opt.clone())
            .unwrap()
            .dipole;
        let energy_at = |field: Vec3| -> f64 {
            let mut o = opt.clone();
            o.external_field = crate::field::ExternalFieldOptions::electric(field);
            run_electronic(&system, &params, o).unwrap().total_free
        };
        let h = 1.0e-4;
        for axis in 0..3 {
            let mut fp = base;
            let mut fm = base;
            match axis {
                0 => {
                    fp.x += h;
                    fm.x -= h;
                }
                1 => {
                    fp.y += h;
                    fm.y -= h;
                }
                _ => {
                    fp.z += h;
                    fm.z -= h;
                }
            }
            let dedf = (energy_at(fp) - energy_at(fm)) / (2.0 * h);
            let mu = match axis {
                0 => dipole.x,
                1 => dipole.y,
                _ => dipole.z,
            };
            assert!(
                (mu - (-dedf)).abs() < 1.0e-5,
                "axis {axis}: dipole {mu:.6} vs -dE/dE_field {:.6}",
                -dedf
            );
        }
    }

    /// Sanity: the field–dipole coupling actually changes the energy (it is active, not a
    /// no-op). With the same field, turning `field_multipole` on shifts the total energy.
    #[test]
    fn field_multipole_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let field = Vec3::new(0.02, 0.0, 0.0);
        let mut base = ElectronicOptions::default();
        base.multipole = true;
        base.external_field = crate::field::ExternalFieldOptions::electric(field);
        let e_off = run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let mut on = base.clone();
        on.field_multipole = true;
        let e_on = run_electronic(&system, &params, on).unwrap().total_free;
        assert!(
            (e_on - e_off).abs() > 1.0e-7,
            "field-dipole coupling did not change the energy (Δ = {:.3e})",
            e_on - e_off
        );
    }

    /// Stage 4 gate: with the experimental third-order on-site multipole cross terms on
    /// (multipole + `multipole_third_order`), the full analytic gradient must match a central
    /// finite-difference of the total energy — the energy↔Fock↔gradient internal consistency of
    /// the term (it rides the SCC-converged moments + charge via the energy-weighted Pulay, plus
    /// the explicit third-order overlap-Pulay weight).
    #[test]
    fn third_order_multipole_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_third_order = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "third-order multipole analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Third-order multipole off ⇒ byte-for-byte the plain mDFTB2 gradient (fully flag-gated).
    #[test]
    fn third_order_multipole_off_unchanged() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = AnalyticGradientOptions::default();
        base.electronic.multipole = true;
        let g_base = analytic_gradient(&system, &params, base.clone())
            .unwrap()
            .gradient;
        let mut explicit = base.clone();
        explicit.electronic.multipole_third_order = false;
        let g_explicit = analytic_gradient(&system, &params, explicit)
            .unwrap()
            .gradient;
        for (a, b) in g_base.iter().zip(&g_explicit) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
        }
    }

    /// Sanity: the third-order multipole term actually changes the energy (active, not a no-op).
    #[test]
    fn third_order_multipole_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = ElectronicOptions::default();
        base.multipole = true;
        let e_off = run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let mut on = base.clone();
        on.multipole_third_order = true;
        let e_on = run_electronic(&system, &params, on).unwrap().total_free;
        assert!(
            (e_on - e_off).abs() > 1.0e-9,
            "third-order multipole term did not change the energy (Δ = {:.3e})",
            e_on - e_off
        );
    }

    /// v0.2.1 gate: with the experimental PER-RANK multipole×charge cross terms on, the full
    /// analytic gradient (the cross-term overlap-Pulay weight + the SCC-converged density carrying
    /// the cross Fock) must match a central finite difference of the total energy. Exercises every
    /// active rank (1..=4) at several charge orders simultaneously, validating the energy, the
    /// self-consistent cross Fock, and the cross overlap-Pulay weight together.
    #[test]
    fn multipole_charge_cross_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_order = 4;
        opt.electronic.multipole_charge_order = vec![5, 4, 4, 4];
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "per-rank multipole×charge cross-term analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// v0.2.1: an empty `multipole_charge_order` ⇒ byte-for-byte the generic order-4 multipole
    /// gradient (the cross terms are fully gated by the non-empty vector).
    #[test]
    fn multipole_charge_cross_off_unchanged() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = AnalyticGradientOptions::default();
        base.electronic.multipole = true;
        base.electronic.multipole_order = 4;
        let g_base = analytic_gradient(&system, &params, base.clone())
            .unwrap()
            .gradient;
        let mut explicit = base.clone();
        explicit.electronic.multipole_charge_order = Vec::new();
        let g_explicit = analytic_gradient(&system, &params, explicit)
            .unwrap()
            .gradient;
        for (a, b) in g_base.iter().zip(&g_explicit) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
        }
    }

    /// v0.2.1 sanity: the cross terms actually change the energy (active, not a no-op).
    #[test]
    fn multipole_charge_cross_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = ElectronicOptions::default();
        base.multipole = true;
        base.multipole_order = 4;
        let e_off = run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let mut on = base.clone();
        on.multipole_charge_order = vec![5, 4, 4, 4];
        let e_on = run_electronic(&system, &params, on).unwrap().total_free;
        assert!(
            (e_on - e_off).abs() > 1.0e-9,
            "multipole×charge cross terms did not change the energy (Δ = {:.3e})",
            e_on - e_off
        );
    }

    /// v0.2.1: a charge order above the rank-`l` termination bound `2l+3` is a hard error (never
    /// silently truncated) — e.g. order 6 on the dipole (rank 1, bound 5).
    #[test]
    fn multipole_charge_order_rejects_too_high_order() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = ElectronicOptions::default();
        opt.multipole = true;
        opt.multipole_order = 2;
        opt.multipole_charge_order = vec![6, 4]; // dipole order 6 > 2·1+3 = 5
        assert!(
            run_electronic(&system, &params, opt).is_err(),
            "dipole charge order 6 (> bound 5) must be rejected, not silently truncated"
        );
    }

    /// v0.2.1 robust-SCC: the **multipole + long-range exchange** combination now runs through the
    /// robust density-matrix driver (ADIIS→C-DIIS→TRAH) rather than the old charge-vector linear-mixing
    /// loop. This gate checks the combined SCC both converges and stays energy↔gradient consistent:
    /// the full analytic gradient (multipole + MFX) must match a central finite difference of the
    /// total energy (both evaluated through the same robust driver).
    #[test]
    fn multipole_mfx_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.lr_exchange = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "multipole + MFX (robust density-matrix SCC) analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Stage 5 gate: with no secondary basis the result is byte-for-byte the primary-basis
    /// multipole gradient (the secondary-moment path is fully gated by the `Option`).
    #[test]
    fn secondary_moments_off_equals_primary() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = AnalyticGradientOptions::default();
        base.electronic.multipole = true;
        let g_base = analytic_gradient(&system, &params, base.clone())
            .unwrap()
            .gradient;
        let mut explicit = base.clone();
        explicit.electronic.multipole_secondary_basis = None;
        let g_explicit = analytic_gradient(&system, &params, explicit)
            .unwrap()
            .gradient;
        for (a, b) in g_base.iter().zip(&g_explicit) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
        }
    }

    /// Stage 5 gate: the analytic gradient with the richer secondary-basis on-site moments must
    /// match a central finite difference of the (secondary-enriched) total energy — the energy /
    /// Fock / gradient all consume the same secondary moment integrals.
    #[test]
    fn secondary_moment_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let Some(Ok(sec)) = crate::secondary_bases::builtin_secondary("cc-pVDZ") else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_secondary_basis = Some(sec);
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "secondary-moment analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// CAMM-on-mDFTB2 gate: the analytic CAMM gradient (kernel force + cumulative-moment integral
    /// derivatives) must match a central finite difference of the total CAMM SCC energy, with both
    /// calibration levers (`camm_damp` κ and `camm_aes_scale` s_AES) set to non-trivial values.
    #[test]
    fn camm_aes_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_model = crate::electronic::MultipoleModel::CammOnMdftb2;
        opt.electronic.camm_damp = 1.2;
        opt.electronic.camm_aes_scale = 0.7;
        opt.electronic.camm_onsite_scale = 0.6; // global s_onsite (fallback for elements w/o override)
        // element-specific κ (O vs H) exercises the √(κ_A·κ_B) per-atom path in the gradient.
        opt.electronic.camm_damp_elem = vec![(8, 1.6), (1, 0.7)];
        // element-specific s_onsite (O override; H falls back to the global 0.6) exercises the
        // per-atom on-site-penalty path in the analytic gradient.
        opt.electronic.camm_onsite_scale_elem = vec![(8, 0.35)];
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "CAMM analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Sanity: the secondary basis actually changes the on-site moments (and hence the energy)
    /// relative to the primary minimal basis — the enrichment is active, not a no-op.
    #[test]
    fn secondary_moments_change_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let Some(Ok(sec)) = crate::secondary_bases::builtin_secondary("cc-pVDZ") else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = ElectronicOptions::default();
        base.multipole = true;
        let e_primary = run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let mut sec_opt = base.clone();
        sec_opt.multipole_secondary_basis = Some(sec);
        let e_sec = run_electronic(&system, &params, sec_opt)
            .unwrap()
            .total_free;
        assert!(
            (e_sec - e_primary).abs() > 1.0e-9,
            "secondary basis did not change the energy (Δ = {:.3e})",
            e_sec - e_primary
        );
    }

    /// v0.2.1: the **generic arbitrary-rank** path (`multipole_order ≥ 4`) must also consume the
    /// secondary basis (previously it silently used the primary AOs for its moment cache). With the
    /// `OnsiteMomentCache::build_with_aos` fix, order-4 + cc-pVDZ now differs from order-4 primary.
    #[test]
    fn secondary_basis_active_in_generic_path() {
        let Some(params) = load_params() else {
            return;
        };
        let Some(Ok(sec)) = crate::secondary_bases::builtin_secondary("cc-pVDZ") else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut base = ElectronicOptions::default();
        base.multipole = true;
        base.multipole_order = 4;
        let e_primary = run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let mut sec_opt = base.clone();
        sec_opt.multipole_secondary_basis = Some(sec);
        let e_sec = run_electronic(&system, &params, sec_opt)
            .unwrap()
            .total_free;
        assert!(
            (e_sec - e_primary).abs() > 1.0e-9,
            "secondary basis ignored by the generic (order-4) multipole path (Δ = {:.3e})",
            e_sec - e_primary
        );
    }

    /// v0.2.1: the order-4 generic gradient with the secondary basis on must match a central
    /// finite difference of the (secondary-enriched) energy — the SCC moment cache and the gradient
    /// moment cache now both run over the secondary AOs, so energy and forces stay consistent.
    #[test]
    fn multipole_order_4_secondary_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let Some(Ok(sec)) = crate::secondary_bases::builtin_secondary("cc-pVDZ") else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_order = 4;
        opt.electronic.multipole_secondary_basis = Some(sec);
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "order-4 secondary-basis analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Sanity: the 4th-order term actually perturbs the energy (it is active, not a no-op).
    #[test]
    fn fourth_order_charge_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let e_off = run_electronic(
            &system,
            &params,
            AnalyticGradientOptions::default().electronic,
        )
        .unwrap()
        .total_free;
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.charge_order = 4;
        let e_on = run_electronic(&system, &params, opt.electronic)
            .unwrap()
            .total_free;
        // A small higher-order correction for near-neutral water (~3e-8 Ha); larger for
        // strongly charged sites. The gradient FD gate is the quantitative check.
        assert!(
            (e_on - e_off).abs() > 1.0e-9,
            "4th-order charge term had no effect on the energy: {e_off} vs {e_on}"
        );
    }

    /// Generalization gate: the on-site charge expansion to an arbitrary order
    /// (`charge_order = 6`) must still give an analytic gradient matching FD — the
    /// Linear Breathing-Radius closed form `X_n = (γ/(n−1))(2Γ/γ)^(n−2)` is variational
    /// at every order (the terms ride the SCC charges).
    #[test]
    fn charge_order_six_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(POLAR_WATER, 0.0, false).unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.charge_order = 6;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "charge_order=6 analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Stage 2f gate: with the octupole multipole on, the full analytic gradient (mDFTB2
    /// + octupole kernel forces + octupole overlap-Pulay) must match a central FD of the
    /// total energy. Distorted H2S (S has d → nonzero traceless octupole and asymmetric
    /// moments, so every octupole gradient term is exercised).
    #[test]
    fn octupole_analytic_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.05 0.02 0.10\nH 0.10 0.95 0.93\nH -0.02 -0.97 0.90\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = AnalyticGradientOptions::default();
        opt.electronic.multipole = true;
        opt.electronic.multipole_octupole = true;
        let ana = analytic_gradient(&system, &params, opt.clone())
            .unwrap()
            .gradient;
        let energy = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, opt.electronic.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 5.0e-5,
            "octupole analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }
}
