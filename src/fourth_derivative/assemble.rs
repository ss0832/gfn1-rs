// SPDX-License-Identifier: GPL-3.0-or-later
//! Final assembly of the directional analytic fourth derivative
//! `e⁗[v] = Σ_abcd Q_abcd v_a v_b v_c v_d`.
//!
//! [`directional_fourth_derivative`] sums the five FD-gated stages of
//! [`super::directional`] / [`super::response_stage`] — each stage is the exact
//! `λ`-derivative of one ingredient of the (validated) analytic third
//! derivative along `R + λv`, so the sum is the `λ`-derivative of the FULL
//! third derivative contracted `vvv`:
//!
//! 1. geometric (repulsion + halogen + frozen SCC2 + D3) — `L_λλλλ`;
//! 2. frozen-density Hamiltonian blocks (Pulay/CN-H0/scalar-overlap fourths +
//!    the density path of their thirds + the Pulay-third CN response);
//! 3. the FC3 density-path term's derivative (needs the SECOND-order screened
//!    bundle `X^vv` from the charge-space solver);
//! 4. the FC3 Pulay CN-response term's derivative (needs `CN^vv`);
//! 5. the FC3 response term's derivative (the 2n+1 quartic response stage).
//!
//! By the 2n+1 rule the whole assembly needs only the first- and second-order
//! responses ALONG `v` — one charge-space first-order solve and one
//! second-order solve — never the full `O(ndof)`/`O(ndof²)` response sets.
//! [`directional_fourth_seminumerical`] (central FD of the analytic third
//! derivative) is the verification reference and production fallback.

use crate::error::{Gfn1Error, Result};
use crate::hessian::AnalyticHessianOptions;
use crate::linalg::Matrix;
use crate::params::Gfn1Parameters;
use crate::response::charge_space::{
    ChargeSpaceContext, FirstOrderField, SecondOrderBundle, SecondOrderField,
};
use crate::response::cpxtb::{
    solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions,
};
use crate::system::PeriodicSystem;

use super::directional::{
    directional_fourth_cn_response_stage, directional_fourth_frozen_density_with,
    directional_fourth_geometric_with, directional_fourth_hessian_path_stage_with,
};
use super::response_stage::directional_response_fourth;
use super::SymmetricFourth;

/// The screened directional first-order legs: the charge-space field along `v`
/// plus the TOTAL potential derivative `V^v = (∂V/∂R)·v + K q^v`.
pub(crate) fn directional_first_order_legs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    cphf: &crate::response::cpxtb::GammaCartesianCpxtbResult,
    ctx: &ChargeSpaceContext,
    v: &[f64],
) -> Result<(FirstOrderField, Vec<f64>)> {
    let field = directional_first_order_field(electronic, cphf, ctx, v)?;
    let ndof = 3 * system.atoms.len();
    let nshell = electronic.basis.shells.len();
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    let v_pot: Vec<f64> = (0..nshell)
        .map(|s| {
            let geo: f64 = (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum();
            geo + field.bundle.screened_potential[s]
        })
        .collect();
    Ok((field, v_pot))
}

pub(crate) fn directional_first_order_field(
    electronic: &crate::electronic::ElectronicResult,
    cphf: &crate::response::cpxtb::GammaCartesianCpxtbResult,
    ctx: &ChargeSpaceContext,
    v: &[f64],
) -> Result<FirstOrderField> {
    let n = electronic.basis.len();
    let mut f_skel = Matrix::zeros(n, n);
    let mut s_dir = Matrix::zeros(n, n);
    for (c, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        let h0 = cphf.derivative_matrices[c].h0_deriv.as_slice();
        let ov = cphf.derivative_matrices[c].overlap_deriv.as_slice();
        let fs = f_skel.as_mut_slice();
        let ss = s_dir.as_mut_slice();
        for k in 0..n * n {
            fs[k] += vc * h0[k];
            ss[k] += vc * ov[k];
        }
    }
    ctx.first_order_field(f_skel, s_dir)
}

/// The screened directional second-order legs `(P^vv, W^vv, q^vv)` plus the
/// total second potential derivative
/// `V^vv = (∂²V/∂R²)·vv + 2(∂_vγ)q^v + E'''·(q^v)² + K q^vv`.
///
/// The skeleton second derivatives `F^vv`/`S^vv` come from the ONE-PASS
/// directional builders of [`crate::hessian`]: the per-pair second-derivative
/// data is `(c,d)`-independent, so contracting both legs against `v` inside a
/// single AO-pair sweep replaces what used to be `O(ndof²)` per-`(c,d)` block
/// builds (each re-evaluating the same pair integrals, and rebuilding the
/// coordination-number and shell-potential ladders every time). Memory stays
/// `O(n²)`.
pub(crate) fn directional_second_order_legs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    ctx: &ChargeSpaceContext,
    field: &FirstOrderField,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    v: &[f64],
) -> Result<(SecondOrderBundle, Vec<f64>)> {
    let (field, potential) = directional_second_order_field_legs(
        system,
        params,
        electronic,
        ctx,
        field,
        coordination_cutoff,
        include_cn_h0,
        v,
    )?;
    Ok((field.bundle, potential))
}

/// Directional second response retaining the analytic MO-motion fields.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_second_order_field_legs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    ctx: &ChargeSpaceContext,
    field: &FirstOrderField,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    v: &[f64],
) -> Result<(SecondOrderField, Vec<f64>)> {
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    directional_second_order_field(
        system,
        params,
        electronic,
        ctx,
        field,
        coordination_cutoff,
        include_cn_h0,
        v,
        &dvdr_q,
        true,
    )
}

/// The TD derivative uses the screened field without the separate FC4
/// potential leg. The unperturbed geometric potential derivative is shared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_second_order_field(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    ctx: &ChargeSpaceContext,
    field: &FirstOrderField,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    v: &[f64],
    dvdr_q: &Matrix,
    need_potential: bool,
) -> Result<(SecondOrderField, Vec<f64>)> {
    let basis = &electronic.basis;
    let n = basis.len();
    let nshell = basis.shells.len();
    let ndof = 3 * system.atoms.len();
    let zeros = vec![0.0_f64; nshell];
    // The already-directional shell-potential leg the one-pass SCC-scalar
    // builder takes: `Σ_d v_d ∂V_s/∂R_d|_q` (the double loop handed it the d-th
    // column of `dvdr_q` once per `(c,d)` pair).
    let v_geo: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
        .collect();
    let cn_block = if include_cn_h0 {
        Some(crate::hessian::directional_h0_cn_block_second_matrix(
            system,
            params,
            electronic,
            coordination_cutoff,
            v,
        )?)
    } else {
        None
    };
    let scc = crate::hessian::directional_h0_scc_scalar_second_matrix(
        system, params, electronic, v, &v_geo, &zeros,
    )?;
    let mut f_vv =
        crate::hessian::directional_h0_bare_second_matrix(system, params, electronic, v)?;
    {
        let fs = f_vv.as_mut_slice();
        for k in 0..n * n {
            fs[k] += cn_block.as_ref().map_or(0.0, |m| m.as_slice()[k]) + scc.as_slice()[k];
        }
    }
    let s_vv = crate::hessian::directional_overlap_second_matrix(system, basis, v)?;
    // (∂γ/∂R) q^v — both the mirrored second-order source and the V^vv cross term.
    let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &field.bundle.shell_charges,
        params,
    )?;
    let dgamma_v_qv: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
        .collect();
    let second = ctx.second_order_field(field, field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)?;
    if !need_potential {
        return Ok((second, Vec::new()));
    }

    let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let kernel = crate::response::cpxtb::response_shell_scc_kernel(system, params, electronic)?;
    // Onsite anharmonic chain: per shell `E'''_A · (q^v_A)²` (the ∂²V/∂q² term).
    let chain =
        ctx.kernel_chain_potential(&field.bundle.shell_charges, &field.bundle.shell_charges);
    let v_pot_vv: Vec<f64> = (0..nshell)
        .map(|s| {
            let geo2: f64 = (0..ndof)
                .map(|c| {
                    (0..ndof)
                        .map(|d| v[c] * v[d] * d2vdr_q[s][(c, d)])
                        .sum::<f64>()
                })
                .sum();
            let cross: f64 = 2.0 * (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum::<f64>();
            let kq: f64 = (0..nshell)
                .map(|t| kernel[(s, t)] * second.bundle.shell_charges[t])
                .sum();
            geo2 + cross + chain[s] + kq
        })
        .collect();
    Ok((second, v_pot_vv))
}

/// Direction-length guard shared by the public directional entry points.
fn check_direction_len(system: &PeriodicSystem, v: &[f64], what: &str) -> Result<()> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "{what}: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Gfn1Error::InvalidInput(format!(
            "{what}: direction must be finite"
        )));
    }
    Ok(())
}

/// The **`v`-independent reference state** of the analytic quartic: one SCF, one
/// CPXTB Hessian-response solve and one charge-space context.
///
/// It depends on the geometry only, so the mixed-index driver
/// ([`fourth_derivative_analytic_dense`]) builds it ONCE and evaluates every
/// polarization direction against it via
/// [`directional_fourth_with_reference`] — hundreds of SCF + CPXTB solves
/// collapse to one.
///
/// The order-4 registry guard lives here. Positive-temperature references use
/// the canonical spectral free-energy response, including fractional degeneracy.
pub struct QuarticReference {
    identity: [u8; 32],
    backend: QuarticBackend,
}

enum QuarticBackend {
    Variational(Box<crate::third_derivative::CubicReference>),
    Integer(Box<IntegerQuarticReference>),
    Thermal(Box<crate::third_derivative::finite_t::FiniteTThirdReference>),
}

struct IntegerQuarticReference {
    electronic: crate::electronic::ElectronicResult,
    cphf: crate::response::cpxtb::GammaCartesianCpxtbResult,
    ctx: ChargeSpaceContext,
    /// The UNDOCTORED frozen-density Pulay third derivative. Stages 2 and 3 both subtract this
    /// block (it is the `V`-shift baseline of the Pulay bilinear charge path), and it depends on
    /// the geometry and the converged reference only — so a build that used to happen twice per
    /// direction happens once per geometry instead. `O(ndof³)` doubles, the same order the stages
    /// already allocate for their own doctored blocks.
    pulay_third_reference: Vec<Matrix>,
}

fn use_variational_quartic(
    system: &PeriodicSystem,
    options: &AnalyticHessianOptions,
    cutoff: f64,
) -> bool {
    // Retain the validated staged gas-phase route outside the dense-size
    // regime. Solvent derivatives share the variational backend at any size.
    (options.electronic_options.solvation.is_some()
        || 3 * system.atoms.len() <= crate::MAX_FOURTH_DERIVATIVE_NDOF)
        && crate::variational_derivatives::supports_cartesian(options, cutoff)
}

impl QuarticReference {
    /// Converge the SCF, solve the CPXTB Hessian response and build the
    /// charge-space context for `system`, checking order 4 and selecting the
    /// requested-temperature analytic backend without changing temperature.
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &AnalyticHessianOptions,
        coordination_cutoff: f64,
    ) -> Result<Self> {
        Self::build_with_budget(
            system,
            params,
            options,
            coordination_cutoff,
            512 * 1024 * 1024,
        )
    }

    pub fn build_with_budget(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &AnalyticHessianOptions,
        coordination_cutoff: f64,
        memory_budget_bytes: usize,
    ) -> Result<Self> {
        let memory_budget_bytes = crate::resources::effective_memory_budget(memory_budget_bytes);
        if system.lattice.is_some() {
            return Err(Gfn1Error::InvalidInput(
                "molecular FC4 reference does not support PBC".into(),
            ));
        }
        crate::terms::require_order(
            &options.electronic_options,
            params,
            4,
            "the analytic fourth derivative",
        )?;
        if use_variational_quartic(system, options, coordination_cutoff) {
            let reference = crate::third_derivative::CubicReference::build(
                system,
                params,
                &options.electronic_options,
                memory_budget_bytes,
            )?;
            return Ok(Self {
                identity: reference_identity(system, params, options, coordination_cutoff)?,
                backend: QuarticBackend::Variational(Box::new(reference)),
            });
        }
        if options.electronic_options.electronic_temperature > 0.
            || options.electronic_options.solvation.is_some()
        {
            let thermal =
                crate::third_derivative::finite_t::FiniteTThirdReference::build_with_budget(
                    system,
                    params,
                    options,
                    coordination_cutoff,
                    memory_budget_bytes,
                )?;
            return Ok(Self {
                identity: reference_identity(system, params, options, coordination_cutoff)?,
                backend: QuarticBackend::Thermal(Box::new(thermal)),
            });
        }
        let _profile = crate::profile::scope("fc4.reference.total");
        let estimate =
            crate::third_derivative::finite_t::FiniteTThirdReference::estimated_workspace_bytes(
                system, params, options,
            )?;
        if estimate > memory_budget_bytes {
            return Err(Gfn1Error::InvalidInput(format!("FC4 reference/workspace estimate {estimate} bytes exceeds budget {memory_budget_bytes}")));
        }
        let electronic = {
            let _p = crate::profile::scope("fc4.reference.scf");
            crate::electronic::run_electronic(system, params, options.electronic_options.clone())?
        };
        if !electronic.converged {
            return Err(Gfn1Error::InvalidInput(
                "FC4 requires a converged SCF reference".into(),
            ));
        }
        if electronic
            .occupations
            .iter()
            .any(|&f| f > 1.0e-8 && (f - 2.0).abs() > 1.0e-8)
        {
            return Err(Gfn1Error::InvalidInput(
                "zero-temperature analytic fourth derivative requires integer occupations; \
                 choose a positive electronic temperature for a fractional ensemble"
                    .to_string(),
            ));
        }
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff,
            include_cn_h0: options.electronic_options.hamiltonian.enable_cn_hamiltonian,
        };
        let cphf = {
            let _p = crate::profile::scope("fc4.reference.cpxtb");
            solve_nonpbc_cpxtb_hessian_response(
                system,
                params,
                &electronic,
                ao_opts,
                CpxtbOptions::default(),
            )?
        };
        let ctx = {
            let _p = crate::profile::scope("fc4.reference.charge_space");
            ChargeSpaceContext::build(system, params, &electronic)?
        };
        let pulay_third_reference = {
            let _p = crate::profile::scope("fc4.reference.pulay_third");
            crate::hessian::fixed_density_pulay_third_derivative(system, params, &electronic)?
        };
        Ok(Self {
            identity: reference_identity(system, params, options, coordination_cutoff)?,
            backend: QuarticBackend::Integer(Box::new(IntegerQuarticReference {
                electronic,
                cphf,
                ctx,
                pulay_third_reference,
            })),
        })
    }

    /// The converged reference SCF state (shared by all directions).
    pub fn electronic(&self) -> &crate::electronic::ElectronicResult {
        match &self.backend {
            QuarticBackend::Variational(reference) => reference.electronic(),
            QuarticBackend::Integer(reference) => &reference.electronic,
            QuarticBackend::Thermal(reference) => reference.electronic(),
        }
    }
}

/// **The analytic directional fourth derivative** `e⁗[v] = Q·vvvv` — one SCF,
/// one CPXTB solve, one charge-space first-order solve and one second-order
/// solve along `v`, then the five gated stages summed.
///
/// The term registry must support analytic order 4 for every active term.
/// Positive temperature selects canonical analytic free-energy response;
/// no nuclear finite differences or temperature substitution are used.
pub fn directional_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    check_direction_len(system, v, "directional_fourth_derivative")?;
    let reference = QuarticReference::build(system, params, options, coordination_cutoff)?;
    directional_fourth_with_reference(system, params, options, coordination_cutoff, &reference, v)
}

/// The per-direction half of [`directional_fourth_derivative`]: the first- and
/// second-order legs along `v` plus the five gated stages, evaluated against a
/// PRE-BUILT [`QuarticReference`].
///
/// Splitting the reference out is what makes the mixed-index polarization build
/// affordable: hundreds of directions share one SCF/CPXTB/charge-space setup.
pub fn directional_fourth_with_reference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    reference: &QuarticReference,
    v: &[f64],
) -> Result<f64> {
    if reference.identity != reference_identity(system, params, options, coordination_cutoff)? {
        return Err(Gfn1Error::InvalidInput(
            "FC4 reference identity mismatch: geometry, parameters, options or D3 data changed"
                .into(),
        ));
    }
    directional_fourth_reference_unchecked(
        system,
        params,
        options,
        coordination_cutoff,
        reference,
        v,
    )
}

fn reference_identity(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    cutoff: f64,
) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(system.charge.to_bits().to_le_bytes());
    for atom in &system.atoms {
        hash.update([atom.z]);
        for x in [atom.position.x, atom.position.y, atom.position.z] {
            hash.update(x.to_bits().to_le_bytes());
        }
    }
    let mut global: Vec<_> = params.globpar.iter().collect();
    global.sort_by_key(|(key, _)| *key);
    for (key, value) in global {
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key.as_bytes());
        hash.update(value.to_bits().to_le_bytes());
    }
    let mut elements: Vec<_> = params.elements.iter().collect();
    elements.sort_by_key(|(z, _)| **z);
    for (&z, e) in elements {
        hash.update([z]);
        for x in [e.gam, e.gam3_raw, e.repa, e.repb] {
            hash.update(x.to_bits().to_le_bytes());
        }
        hash.update(format!("{:?}", e.shells).as_bytes());
        hash.update(e.ao_raw.as_bytes());
        let mut raw: Vec<_> = e.raw.iter().collect();
        raw.sort_by_key(|(k, _)| *k);
        for (key, values) in raw {
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key.as_bytes());
            for x in values {
                hash.update(x.to_bits().to_le_bytes());
            }
        }
    }
    let mut pairs: Vec<_> = params.pairpar.iter().collect();
    pairs.sort_by_key(|(key, _)| **key);
    for (&(a, b), x) in pairs {
        hash.update([a, b]);
        hash.update(x.to_bits().to_le_bytes());
    }
    let mut physical_options = options.clone();
    physical_options.electronic_options.logger = None;
    hash.update(format!("{physical_options:?}").as_bytes());
    hash.update(cutoff.to_bits().to_le_bytes());
    if let Some(path) = &options.electronic_options.d3_reference_path {
        hash.update(std::fs::read(path)?);
    }
    Ok(hash.finalize().into())
}

pub(crate) fn directional_fourth_reference_unchecked(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    reference: &QuarticReference,
    v: &[f64],
) -> Result<f64> {
    check_direction_len(system, v, "directional_fourth_with_reference")?;
    let reference = match &reference.backend {
        QuarticBackend::Variational(reference) => return reference.directional_fourth(v),
        QuarticBackend::Thermal(thermal) => {
            return crate::third_derivative::finite_t::directional_fourth_with_reference(
                system,
                params,
                options,
                coordination_cutoff,
                thermal,
                v,
            )
        }
        QuarticBackend::Integer(integer) => integer,
    };
    let electronic = &reference.electronic;
    let cphf = &reference.cphf;
    let ctx = &reference.ctx;
    let include_cn_h0 = options.electronic_options.hamiltonian.enable_cn_hamiltonian;
    let ao_opts = AoDerivativeOptions {
        coordination_cutoff,
        include_cn_h0,
    };
    let _profile = crate::profile::scope("fc4.direction.total");
    let (field, v_pot_v) = {
        let _p = crate::profile::scope("fc4.legs.first_order");
        directional_first_order_legs(system, params, electronic, cphf, ctx, v)?
    };
    let (second, v_pot_vv) = {
        let _p = crate::profile::scope("fc4.legs.second_order");
        directional_second_order_legs(
            system,
            params,
            electronic,
            ctx,
            &field,
            coordination_cutoff,
            include_cn_h0,
            v,
        )?
    };

    let s1 = {
        let _p = crate::profile::scope("fc4.stage1.geometric");
        directional_fourth_geometric_with(system, params, electronic, options, v)?
    };
    // Stage 1 evaluates the frozen SCC2 fourth at FIXED reference charges (its
    // gate freezes the electronic reference), but the FC3 composition's SCC2
    // third block is evaluated at q(R): its `λ`-derivative therefore carries
    // the bilinear charge path `∂(scc2_third·vvv)/∂q · q^v`. The block is
    // exactly quadratic in the shell charges, so the polarization-identity-
    // pinned `fixed_shell_charge_scc_third_charge_path` IS that derivative.
    let s1_charge_path = if options.include_fixed_scc {
        let _p = crate::profile::scope("fc4.stage1.scc_charge_path");
        let path = crate::hessian::fixed_shell_charge_scc_third_charge_path(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            &field.bundle.shell_charges,
            params,
        )?;
        super::directional::contract_slabs_vvv(&path, v)
    } else {
        0.0
    };
    // Stages 2 and 3 both consume the reference's cached undoctored Pulay third block.
    let pulay_third_reference = Some(reference.pulay_third_reference.as_slice());
    let s2 = {
        let _p = crate::profile::scope("fc4.stage2.frozen_density");
        directional_fourth_frozen_density_with(
            system,
            params,
            electronic,
            coordination_cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            &field.bundle.shell_charges,
            &v_pot_v,
            v,
            pulay_third_reference,
        )?
    };
    let _p3 = crate::profile::scope("fc4.stage3.hessian_path");
    let s3 = directional_fourth_hessian_path_stage_with(
        system,
        params,
        electronic,
        coordination_cutoff,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        &v_pot_v,
        &second.density,
        &second.energy_weighted,
        &second.shell_charges,
        &v_pot_vv,
        v,
        pulay_third_reference,
    )?;
    drop(_p3);
    let s4 = if include_cn_h0 {
        let _p = crate::profile::scope("fc4.stage4.cn_response");
        directional_fourth_cn_response_stage(
            system,
            params,
            electronic,
            coordination_cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            v,
        )?
    } else {
        0.0
    };
    let s5 = {
        let _p = crate::profile::scope("fc4.stage5.response");
        directional_response_fourth(
            system,
            params,
            electronic,
            cphf,
            ao_opts,
            coordination_cutoff,
            v,
        )?
    };
    Ok(s1 + s1_charge_path + s2 + s3 + s4 + s5)
}

/// A polarization direction as the sorted `(dof, weight)` list of a non-empty
/// sub-multiset of an index quadruple — the deduplication key of the
/// mixed-index build. At most four entries, weights in `1..=4`.
// Sorted multiset, padded by a sentinel: no heap allocation for a direction key.
#[cfg(test)]
type DirectionKey = [usize; 4];

/// The 15 non-empty-subset terms of the polarization identity for one index
/// quadruple: `(sign, direction)` with
///
/// ```text
///   Q(x₁,x₂,x₃,x₄) = (1/24) Σ_{∅ ≠ S ⊆ {1,2,3,4}} (−1)^(4−|S|) e⁗[Σ_{i∈S} x_i]
/// ```
///
/// and `x₁..x₄ = e_a, e_b, e_c, e_d`. Repeated indices are allowed — the
/// identity holds for arbitrary (possibly equal) vectors; a repeat simply gives
/// that DOF weight 2, 3 or 4 in some of the 15 directions.
#[cfg(test)]
fn polarization_terms(quad: [usize; 4]) -> [(f64, DirectionKey); 15] {
    let mut terms = [(0.0, [usize::MAX; 4]); 15];
    for mask in 1u32..16 {
        let mut key = [usize::MAX; 4];
        let mut size = 0usize;
        for (slot, &dof) in quad.iter().enumerate() {
            if mask & (1 << slot) == 0 {
                continue;
            }
            key[size] = dof;
            size += 1;
        }
        key.sort_unstable();
        let sign = if (4 - size) % 2 == 0 { 1.0 } else { -1.0 };
        terms[mask as usize - 1] = (sign, key);
    }
    terms
}

/// Limits for the packed-output/batch assembly, excluding the chemistry driver's
/// reference/workspace. `workspace_reserve_bytes` lets callers reserve those
/// separately; the limit is not an OS peak-RSS guarantee. No full O(d^4) plan or
/// global direction cache is retained. Parallel evaluation is capped by the
/// worker limit, the active pool and the workspace budget.
#[derive(Clone, Copy, Debug)]
pub struct QuarticAssemblyOptions {
    pub memory_budget_bytes: usize,
    pub batch_quadruples: usize,
    pub max_parallel_directions: usize,
    pub workspace_reserve_bytes: usize,
}

impl Default for QuarticAssemblyOptions {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 512 * 1024 * 1024,
            batch_quadruples: 2048,
            max_parallel_directions: 4,
            workspace_reserve_bytes: 0,
        }
    }
}

impl QuarticAssemblyOptions {
    /// Conservative bounded-container estimate; checked before SCF or allocation.
    pub fn estimated_bytes(&self, m: usize, ndof: usize) -> Result<usize> {
        crate::derivative_assembly::estimated_bytes(
            4,
            m,
            ndof,
            self.max_parallel_directions,
            self.batch_quadruples,
            true,
        )?
        .checked_add(self.workspace_reserve_bytes)
        .ok_or_else(|| Gfn1Error::InvalidInput("FC4 assembly size overflow".into()))
    }

    pub fn preflight(&self, m: usize, ndof: usize) -> Result<()> {
        let bytes = self.estimated_bytes(m, ndof)?;
        let budget = crate::resources::effective_memory_budget(self.memory_budget_bytes);
        if bytes > budget {
            return Err(Gfn1Error::InvalidInput(format!(
                "FC4 assembly needs an estimated {bytes} bytes; budget is {}; use a smaller block or direct contractions",
                budget)));
        }
        Ok(())
    }
}

#[cfg(test)]
fn assemble_with_evaluator(
    dofs: &[usize],
    ndof: usize,
    limits: &QuarticAssemblyOptions,
    evaluate: impl Fn(&[f64]) -> Result<f64> + Sync,
) -> Result<SymmetricFourth> {
    crate::derivative_assembly::CartesianAssembly::new(
        4,
        dofs,
        ndof,
        limits.memory_budget_bytes,
        0,
        limits.workspace_reserve_bytes,
        limits.max_parallel_directions,
        limits.batch_quadruples,
    )?
    .quartic(evaluate)
}

/// Share one reference and cache only the lower-order direction values.
pub(crate) fn assemble_polarized_quartic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
    limits: &QuarticAssemblyOptions,
) -> Result<SymmetricFourth> {
    let ndof = 3 * system.atoms.len();
    // Reject impossible packed output before consulting parameter tables.
    limits.preflight(dofs.len(), ndof)?;
    if dofs.is_empty() {
        return SymmetricFourth::try_zeros(0, limits.memory_budget_bytes);
    }
    let (plan, reference) = prepare_quartic(
        system,
        params,
        options,
        coordination_cutoff,
        dofs,
        limits,
        true,
    )?;
    plan.quartic(|v| {
        directional_fourth_reference_unchecked(
            system,
            params,
            options,
            coordination_cutoff,
            &reference,
            v,
        )
    })
}

pub(crate) fn prepare_quartic<'a>(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &'a [usize],
    limits: &QuarticAssemblyOptions,
    packed_output: bool,
) -> Result<(
    crate::derivative_assembly::CartesianAssembly<'a>,
    QuarticReference,
)> {
    let ndof = 3 * system.atoms.len();
    crate::terms::require_order(
        &options.electronic_options,
        params,
        4,
        "the analytic fourth derivative",
    )?;
    let variational = use_variational_quartic(system, options, coordination_cutoff);
    let workspace = if variational {
        crate::third_derivative::CubicReference::estimate_workspace(
            system,
            params,
            &options.electronic_options,
        )?
    } else {
        crate::third_derivative::finite_t::FiniteTThirdReference::estimated_workspace_bytes(
            system, params, options,
        )?
    };
    let plan = crate::derivative_assembly::CartesianAssembly::new_with_output(
        4,
        dofs,
        ndof,
        limits.memory_budget_bytes,
        workspace,
        limits.workspace_reserve_bytes,
        limits.max_parallel_directions,
        limits.batch_quadruples,
        packed_output,
    )?;
    let reference = QuarticReference::build_with_budget(
        system,
        params,
        options,
        coordination_cutoff,
        limits.memory_budget_bytes,
    )?;
    Ok((plan, reference))
}

/// **The full mixed-index analytic quartic force constants** `Q_abcd`, packed
/// into a [`SymmetricFourth`].
///
/// Every element is recovered from the (integration-gated) DIRECTIONAL quartic
/// `e⁗[v] = Q·vvvv` by the polarization identity, so the mixed-index tensor
/// inherits the directional assembly's correctness by construction — no
/// separate mixed-index derivation, no finite differences in the nuclear
/// coordinates. One SCF / CPXTB / charge-space reference is shared by every
/// direction. Lower-order directional values are cached in packed arrays.
///
/// Cost remains O(ndof^4), with one new directional evaluation for each
/// general packed quadruple. Homogeneity reuses directions with uniformly
/// repeated indices. Every direction has at most four non-zero components. Use
/// [`fourth_derivative_analytic_block`] when only a few DOFs matter.
///
/// Guards: identical to [`directional_fourth_derivative`] — they live in
/// [`QuarticReference::build`].
pub fn fourth_derivative_analytic_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<SymmetricFourth> {
    let dofs: Vec<usize> = (0..3 * system.atoms.len()).collect();
    assemble_polarized_quartic(
        system,
        params,
        options,
        coordination_cutoff,
        &dofs,
        &QuarticAssemblyOptions::default(),
    )
}

/// The `|dofs|⁴` **sub-block** of the analytic quartic, packed as a
/// [`SymmetricFourth`] over the SELECTED degrees of freedom: entry
/// `(ai, bi, ci, di)` is `Q[dofs[ai], dofs[bi], dofs[ci], dofs[di]]`.
///
/// Same machinery as [`fourth_derivative_analytic_dense`], restricted to the
/// quadruples drawn from `dofs` — the natural production interface when only a
/// few modes/atoms matter (e.g. the quartic couplings of a reaction
/// coordinate), since the directional-evaluation count drops to
/// `O(|dofs|⁴/24)`.
pub fn fourth_derivative_analytic_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<SymmetricFourth> {
    let ndof = 3 * system.atoms.len();
    for &dof in dofs {
        if dof >= ndof {
            return Err(Gfn1Error::InvalidInput(format!(
                "fourth_derivative_analytic_block: dof {dof} out of range (3*natoms = {ndof})"
            )));
        }
    }
    fourth_derivative_analytic_block_with_limits(
        system,
        params,
        options,
        coordination_cutoff,
        dofs,
        &QuarticAssemblyOptions::default(),
    )
}

/// Budgeted block build. Preflight takes place before the expensive SCF reference.
pub fn fourth_derivative_analytic_block_with_limits(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
    limits: &QuarticAssemblyOptions,
) -> Result<SymmetricFourth> {
    let ndof = 3 * system.atoms.len();
    if dofs.iter().any(|&d| d >= ndof) {
        return Err(Gfn1Error::InvalidInput(
            "FC4 selected DOF out of range".into(),
        ));
    }
    assemble_polarized_quartic(system, params, options, coordination_cutoff, dofs, limits)
}

/// Direct semi-diagonal contractions `T4[u_i,u_i,u_j,u_j]` without a Cartesian
/// tensor. Values of q(v) are FOURTH DERIVATIVES, not Taylor coefficients.
/// The independent-polynomial tests check the 1/12 factor and repeated indices.
pub fn semidiagonal_from_directional(
    directions: &[Vec<f64>],
    mut evaluate: impl FnMut(&[f64]) -> Result<f64>,
) -> Result<Matrix> {
    let n = directions.len();
    let ndof = directions.first().map_or(0, Vec::len);
    if directions
        .iter()
        .any(|v| v.len() != ndof || v.iter().any(|x| !x.is_finite()))
    {
        return Err(Gfn1Error::InvalidInput(
            "quartic contraction directions must have equal length and be finite".into(),
        ));
    }
    if n.checked_mul(n)
        .and_then(|x| x.checked_mul(8))
        .is_none_or(|bytes| bytes > 512 * 1024 * 1024)
    {
        return Err(Gfn1Error::InvalidInput(
            "quartic mode-pair output exceeds the 512 MiB budget".into(),
        ));
    }
    let mut out = Matrix::zeros(n, n);
    let mut diagonal = Vec::with_capacity(n);
    for (i, v) in directions.iter().enumerate() {
        let q = evaluate(v)?;
        if !q.is_finite() {
            return Err(Gfn1Error::InvalidInput(
                "non-finite quartic contraction".into(),
            ));
        }
        diagonal.push(q);
        out[(i, i)] = q;
    }
    let mut sum = vec![0.0; ndof];
    let mut difference = vec![0.0; ndof];
    for j in 0..n {
        for i in 0..j {
            for k in 0..ndof {
                sum[k] = directions[i][k] + directions[j][k];
                difference[k] = directions[i][k] - directions[j][k];
            }
            let value =
                (evaluate(&sum)? + evaluate(&difference)? - 2.0 * diagonal[i] - 2.0 * diagonal[j])
                    / 12.0;
            if !value.is_finite() {
                return Err(Gfn1Error::InvalidInput(
                    "non-finite quartic contraction".into(),
                ));
            }
            out[(i, j)] = value;
            out[(j, i)] = value;
        }
    }
    Ok(out)
}

/// Analytic mode-pair FC4, sharing a single screened reference across all modes.
pub fn fourth_derivative_analytic_semidiagonal(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    directions: &[Vec<f64>],
) -> Result<Matrix> {
    for v in directions {
        check_direction_len(system, v, "FC4 semidiagonal")?;
    }
    if directions.is_empty() {
        return Ok(Matrix::zeros(0, 0));
    }
    let reference = QuarticReference::build(system, params, options, coordination_cutoff)?;
    semidiagonal_from_directional(directions, |v| {
        directional_fourth_reference_unchecked(
            system,
            params,
            options,
            coordination_cutoff,
            &reference,
            v,
        )
    })
}

/// **Seminumerical directional fourth derivative** — the central finite
/// difference along `v` of the analytic directional THIRD derivative
/// `e³[v] = Σ_abc T_abc v_a v_b v_c`, evaluated at `R ± h·v` with everything
/// reconverged: the verification reference of the analytic assembly and the
/// production fallback where the analytic quartic is not available.
///
/// The FD'd reference is the occupation-agnostic
/// [`crate::third_derivative::finite_t::directional_third_finite_t`], so this
/// route ALSO serves **Fermi-smeared** systems. The analytic quartic
/// ([`directional_fourth_derivative`]) supports those systems without this FD.
/// At `T = 0` that reference is equality-gated against the adjoint-assembled
/// [`crate::third_derivative::third_derivative_analytic_vector`] contracted
/// `vvv`, so the integer-occupation numbers are unchanged by the routing.
///
/// `step` is the FD displacement in bohr (`1e-3` is a good default: the `h²`
/// truncation error meets the SCF/CPXTB noise floor there for tight SCF
/// tolerances).
pub fn directional_fourth_seminumerical(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
    step: f64,
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "directional_fourth_seminumerical: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "directional_fourth_seminumerical: step must be positive and finite, got {step}"
        )));
    }
    let third_vvv = |sign: f64| -> Result<f64> {
        let mut displaced = system.clone();
        for (atom_idx, atom) in displaced.atoms.iter_mut().enumerate() {
            atom.position.x += sign * step * v[3 * atom_idx];
            atom.position.y += sign * step * v[3 * atom_idx + 1];
            atom.position.z += sign * step * v[3 * atom_idx + 2];
        }
        crate::third_derivative::finite_t::directional_third_finite_t(
            &displaced,
            params,
            options,
            coordination_cutoff,
            v,
        )
    };
    Ok((third_vvv(1.0)? - third_vvv(-1.0)?) / (2.0 * step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::ElectronicOptions;

    #[test]
    fn streamed_batches_and_direct_pairs_match_independent_polynomial() {
        // q(x,y) = 3 x^4 + 12 x^3 y + 30 x^2 y^2 - 8 x y^3 + 7 y^4
        // Independent coefficients: T0000=3, T0001=3, T0011=5, T0111=-2, T1111=7.
        let q = |v: &[f64]| -> Result<f64> {
            let (x, y) = (v[0], v[1]);
            Ok(
                3.0 * x.powi(4) + 12.0 * x.powi(3) * y + 30.0 * x * x * y * y - 8.0 * x * y.powi(3)
                    + 7.0 * y.powi(4),
            )
        };
        for batch in [1, 3, 128] {
            let limits = QuarticAssemblyOptions {
                batch_quadruples: batch,
                max_parallel_directions: 2,
                ..Default::default()
            };
            let t = assemble_with_evaluator(&[0, 1], 2, &limits, q).unwrap();
            assert!((t.get(0, 0, 0, 0) - 3.0).abs() < 1e-12);
            assert!((t.get(0, 0, 0, 1) - 3.0).abs() < 1e-12);
            assert!((t.get(0, 0, 1, 1) - 5.0).abs() < 1e-12);
            assert!((t.get(0, 1, 1, 1) + 2.0).abs() < 1e-12);
            assert!((t.get(1, 1, 1, 1) - 7.0).abs() < 1e-12);
        }
        let pair =
            semidiagonal_from_directional(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 0.0]], q)
                .unwrap();
        assert_eq!(pair[(0, 1)], 5.0);
        assert_eq!(pair[(0, 2)], 12.0);
        assert_eq!(pair[(2, 2)], 48.0);
    }

    #[test]
    fn size_overflow_and_budget_are_rejected_before_reference() {
        assert_eq!(super::super::checked_packed_len(300).unwrap(), 344_291_325);
        assert!(super::super::checked_packed_len(usize::MAX).is_err());
        assert!(QuarticAssemblyOptions::default()
            .preflight(300, 300)
            .is_err());
        assert!(SymmetricFourth::try_zeros(300, 1024).is_err());
        let sys = PeriodicSystem::from_xyz_str("2\nh2\nH 0 0 0\nH 0.74 0 0\n", 0.0, false).unwrap();
        let limits = QuarticAssemblyOptions {
            memory_budget_bytes: 1,
            ..Default::default()
        };
        // Invalid parameters would fail during SCF; the budget error must arrive first.
        let err = fourth_derivative_analytic_block_with_limits(
            &sys,
            &Gfn1Parameters::default(),
            &AnalyticHessianOptions::default(),
            20.0,
            &[0, 1],
            &limits,
        )
        .unwrap_err();
        assert!(err.to_string().contains("budget"));
    }

    /// The shared gate fixture: non-equilibrium water (all response channels
    /// active, no symmetry cancellations) + tight SCF + a fixed skew direction.
    fn gate_fixture(enable_dispersion: bool) -> (PeriodicSystem, AnalyticHessianOptions, Vec<f64>) {
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut electronic_options = ElectronicOptions::default();
        electronic_options.enable_dispersion = enable_dispersion;
        electronic_options.energy_tolerance = 1.0e-12;
        electronic_options.charge_tolerance = 1.0e-10;
        let options = AnalyticHessianOptions {
            electronic_options,
            ..AnalyticHessianOptions::default()
        };
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        (system, options, v)
    }

    /// **The directional-quartic integration gate.** The five-stage analytic
    /// assembly must equal the central FD along `v` of the FULL analytic third
    /// derivative contracted `vvv`, with everything reconverged at the
    /// displaced geometries. Two FD steps assert the `h²` truncation scaling —
    /// a residual that does not shrink ~4× on halving the step is a missing or
    /// double-counted analytic term, not FD noise.
    fn run_integration_gate(enable_dispersion: bool) {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let (system, options, v) = gate_fixture(enable_dispersion);
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;

        let analytic =
            directional_fourth_derivative(&system, &params, &options, cutoff, &v).unwrap();
        let fd_at = |h: f64| -> f64 {
            directional_fourth_seminumerical(&system, &params, &options, cutoff, &v, h).unwrap()
        };
        let h1 = 1.0e-3;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "directional quartic total (disp {enable_dispersion}): analytic {analytic:.10e} \
             fd(h) {fd1:.10e} fd(h/2) {fd2:.10e} delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} \
             ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-6 * (1.0 + fd1.abs()),
            "directional fourth vs FD(analytic third): analytic {analytic:.10e} fd {fd1:.10e} \
             delta {delta1:.3e}"
        );
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h² (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing or double-counted analytic term"
        );
    }

    /// **The polarization combinatorics gate** — pure algebra, no quantum
    /// chemistry, microseconds to run. For a synthetic FULLY SYMMETRIC quartic
    /// form the sign / subset-weight bookkeeping of [`polarization_terms`] must
    /// reproduce every mixed element exactly, including the repeated-index
    /// patterns (`aaaa`, `aaab`, `aabb`, `aabc`) where a DOF picks up weight 2,
    /// 3 or 4. A wrong sign, a missing subset or a wrong `1/24` would show up
    /// here rather than as a mystery residual in the FD gates.
    #[test]
    fn polarization_identity_recovers_a_synthetic_symmetric_quartic() {
        let n = 5usize;
        // Symmetric by construction: the generator only sees the sorted indices.
        let q = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            let mut t = [a, b, c, d];
            t.sort_unstable();
            let (a, b, c, d) = (t[0] as f64, t[1] as f64, t[2] as f64, t[3] as f64);
            1.0 + a + 2.0 * b - 0.5 * c + a * b * c * d - 0.25 * (a * a + d * d) + (a + b) * (c + d)
        };
        let quartic = |v: &[f64]| -> f64 {
            let mut acc = 0.0;
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            acc += q(a, b, c, d) * v[a] * v[b] * v[c] * v[d];
                        }
                    }
                }
            }
            acc
        };
        for d in 0..n {
            for c in 0..=d {
                for b in 0..=c {
                    for a in 0..=b {
                        let mut acc = 0.0;
                        for (sign, key) in polarization_terms([a, b, c, d]) {
                            let mut v = vec![0.0_f64; n];
                            for dof in key.into_iter().take_while(|&d| d != usize::MAX) {
                                v[dof] += 1.0;
                            }
                            acc += sign * quartic(&v);
                        }
                        let got = acc / 24.0;
                        let want = q(a, b, c, d);
                        assert!(
                            (got - want).abs() < 1.0e-9 * (1.0 + want.abs()),
                            "polarization identity failed at ({a},{b},{c},{d}): \
                             got {got:.12e}, want {want:.12e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn directional_fourth_total_matches_third_fd_along_v() {
        run_integration_gate(false);
    }

    #[test]
    fn directional_fourth_total_matches_third_fd_dispersion_on() {
        run_integration_gate(true);
    }
}
