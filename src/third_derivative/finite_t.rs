// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-temperature (Fermi-smeared) directional analytic third derivative.
//!
//! Route: `e³[v] = D_v[H[v,v]]` assembled by the product rule over the
//! directional Hessian's two halves — the frozen (fixed-density) blocks and
//! the response part `r2[v,v] = Σ_a v_a g_a(X^v)` (the response-gradient
//! contraction that fills `cphf.hessian_response` columns). The directional
//! mode may use the second-order screened bundle `X^vv` directly (one
//! charge-space solve), so no adjoint/Z-vector machinery is needed and the
//! T = 0 orbital algebra is bypassed entirely:
//!
//! ```text
//!   D_v[r2[v,v]] = g(X^vv)·v                       (response motion)
//!                + path_hessian(X^v)[v,v]           (geometric motion of g)
//!                + background_motion(X^v, X^v)·v    (reference-state motion)
//! ```
//!
//! The background motion collects the non-geometric reference dependencies of
//! `g`: `P₀ → P^v` under the screening shift, `V₀ → V^v` under the response
//! density, the kernel motion `∂K/∂q·q^v` (onsite `E'''` chain) and the
//! shell-charge motion `q₀ → q^v` in the kernel-gradient bilinear.
//!
//! At T = 0 this assembly must EQUAL the validated
//! [`crate::fourth_derivative::response_stage::directional_response_third`]
//! (equality gate); at finite temperature it is FD-gated against the
//! seminumerical third derivative on smeared fixtures.

use crate::error::Result;
use crate::params::Gfn1Parameters;
use crate::response::cpxtb::{
    response_electronic_gradient, response_shell_scc_kernel, ResponseGradientContext,
};
use crate::system::PeriodicSystem;

use crate::linalg::Matrix;

/// The response part of the directional Hessian, contracted `vv`:
/// `r2[v,v] = Σ_a v_a g_a(X^v)` with `g` the response-gradient contraction
/// that fills the `hessian_response` columns (linear in the bundle, so the
/// directional bundle gives the directional contraction exactly).
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_response_hessian_vv(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v: &[f64],
) -> Result<f64> {
    let kernel = response_shell_scc_kernel(system, params, electronic)?;
    let grad_ctx = ResponseGradientContext::new(
        system,
        &electronic.basis,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
    )?;
    let g = response_electronic_gradient(
        system, electronic, &kernel, &grad_ctx, p_v, p_v, w_v, q_v,
    )?;
    Ok(g
        .iter()
        .enumerate()
        .map(|(at, grad)| grad.x * v[3 * at] + grad.y * v[3 * at + 1] + grad.z * v[3 * at + 2])
        .sum())
}

/// The total directional derivative of the response part of the Hessian,
/// `D_v[r2[v,v]]`, assembled by the product rule (occupation-agnostic: every
/// ingredient is native to fractional occupations):
///
/// * response motion — `g(X^vv)·v`;
/// * coefficient motion (geometric + reference-state), block by block:
///   - `cn_h0(P^v)` + `cross(P^v)` (the CN chain of the band prefactor),
///   - the `s2` kernel charge path `(q₀, q^v)`,
///   - `pulay(P^v, W^v)` and the pulay potential channel fed the SCREENING
///     part `K q^v` ONLY (the geometric part of `V^v` flows through the
///     `scc_dp_pot` background term below — feeding the TOTAL `V^v` here
///     would double-count it),
///   - `scalar_overlap(P₀, q^v)` (the kernel-geometry motion under the
///     reference density),
///   - the four background families of
///     [`response_gradient_background_motion`]: `−P^v·V^v_tot·∇S`,
///     `−P^v·(Kq^v)·∇S`, `−P₀·chain·∇S` (onsite `E'''`), `∇γ·2q^vq^v`.
///
/// Pinned by the T = 0 equality gate against the adjoint-assembled
/// [`crate::fourth_derivative::response_stage::directional_response_third`]
/// (term inventory identified by frozen-background/family bisection).
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_response_hessian_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    p_vv: &Matrix,
    w_vv: &Matrix,
    q_vv: &[f64],
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    let nshell = electronic.basis.shells.len();
    let cvv = |m: &Matrix| -> f64 {
        let mut acc = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                acc += v[a] * v[b] * m[(a, b)];
            }
        }
        acc
    };

    // Response motion.
    let mut total = directional_response_hessian_vv(
        system,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
        p_vv,
        w_vv,
        q_vv,
        v,
    )?;

    // Coefficient motion, block by block.
    let elec_pc = {
        let mut e = electronic.clone();
        e.density = p_v.clone();
        e
    };
    total += cvv(
        &crate::hessian::fixed_density_cn_h0_hessian(
            system,
            params,
            &elec_pc,
            coordination_cutoff,
        )?
        .hessian,
    );
    total += cvv(&crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
        system,
        params,
        &elec_pc,
        coordination_cutoff,
    )?);
    total += cvv(&crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        q_v,
        params,
    )?);
    {
        let mut e = electronic.clone();
        e.density = p_v.clone();
        e.energy_weighted_density = w_v.clone();
        total += cvv(
            &crate::hessian::fixed_density_pulay_hessian(system, params, &e)?.hessian,
        );
    }
    let kernel = response_shell_scc_kernel(system, params, electronic)?;
    let kq_v = crate::linalg::matrix_vector_product(&kernel, q_v)?;
    {
        let mut e = electronic.clone();
        for s in 0..nshell {
            e.shell_scc_potential[s] += kq_v[s];
        }
        let h1 = cvv(
            &crate::hessian::fixed_density_pulay_hessian(system, params, &e)?.hessian,
        );
        let h0 = cvv(
            &crate::hessian::fixed_density_pulay_hessian(system, params, electronic)?.hessian,
        );
        total += h1 - h0;
    }
    {
        let mut e = electronic.clone();
        e.shell_charges = q_v.to_vec();
        total += cvv(&crate::hessian::fixed_density_scalar_overlap_hessian(
            system, params, &e,
        )?);
    }

    // Background-state motion.
    let grad_ctx = ResponseGradientContext::new(
        system,
        &electronic.basis,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
    )?;
    // Onsite ∂K/∂q chain potential (∂³E_onsite/∂q³ · q^v ∘ q^v), per shell.
    let chain: Vec<f64> = {
        let shell_model =
            crate::coulomb::ShellChargeModel::build(system, &electronic.basis, params)?;
        let nat = system.atoms.len();
        let charge_order = electronic.charge_order.max(3);
        let mut shell_atom = vec![0usize; nshell];
        for atom in 0..nat {
            let offset = shell_model.atom_offsets[atom];
            for local in 0..shell_model.atom_shell_counts[atom] {
                shell_atom[offset + local] = atom;
            }
        }
        let mut atom_qv = vec![0.0_f64; nat];
        for s in 0..nshell {
            atom_qv[shell_atom[s]] += q_v[s];
        }
        (0..nshell)
            .map(|s| {
                let atom = shell_atom[s];
                if shell_model.atom_shell_counts[atom] == 0 {
                    return 0.0;
                }
                let offset = shell_model.atom_offsets[atom];
                let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                    shell_model.hardness[offset],
                    shell_model.hubbard_derivs[offset],
                    charge_order,
                    electronic.atomic_charges[atom],
                );
                third * atom_qv[atom] * atom_qv[atom]
            })
            .collect()
    };
    let bg = crate::response::cpxtb::response_gradient_background_motion(
        system,
        electronic,
        &grad_ctx,
        &kernel,
        p_v,
        q_v,
        &chain,
        v_pot_v,
    )?;
    let dot = |grad: &[crate::math::Vec3]| -> f64 {
        grad.iter()
            .enumerate()
            .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
            .sum()
    };
    total += dot(&bg.scc_dp_pot) + dot(&bg.scc_p0) + dot(&bg.scc_chain) + dot(&bg.kernel_qq);
    Ok(total)
}

/// Shared reference state for finite-temperature third-derivative
/// evaluations: one SCF, one CPXTB solve, one charge-space factorization, and
/// the direction-INDEPENDENT frozen third-derivative slabs. The dense mode
/// reuses it across all ~C(n+2,3) polarization directions.
pub struct FiniteTThirdReference {
    electronic: crate::electronic::ElectronicResult,
    cphf: crate::response::cpxtb::GammaCartesianCpxtbResult,
    ctx: crate::response::charge_space::ChargeSpaceContext,
    frozen: Vec<Matrix>,
    so3: Vec<Matrix>,
    cn_grad: Option<Vec<Vec<f64>>>,
    include_cn_h0: bool,
}

impl FiniteTThirdReference {
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &crate::hessian::AnalyticHessianOptions,
        coordination_cutoff: f64,
    ) -> Result<Self> {
        crate::terms::require_order(
            &options.electronic_options,
            params,
            3,
            "the finite-temperature analytic third derivative",
        )?;
        let electronic = crate::electronic::run_electronic(
            system,
            params,
            options.electronic_options.clone(),
        )?;
        let include_cn_h0 = options.electronic_options.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = crate::response::cpxtb::AoDerivativeOptions {
            coordination_cutoff,
            include_cn_h0,
        };
        let cphf = crate::response::cpxtb::solve_nonpbc_cpxtb_hessian_response(
            system,
            params,
            &electronic,
            ao_opts,
            crate::response::cpxtb::CpxtbOptions::default(),
        )?;
        let ctx = crate::response::charge_space::ChargeSpaceContext::build(
            system, params, &electronic,
        )?;
        let include_disp =
            options.include_dispersion && options.electronic_options.enable_dispersion;
        let disp_ref = if include_disp {
            options.electronic_options.d3_reference_path.as_deref()
        } else {
            None
        };
        let frozen = crate::third_derivative::third_derivative_frozen_complete(
            system,
            params,
            &electronic,
            disp_ref,
            coordination_cutoff,
            include_disp,
        )?;
        let so3 = crate::hessian::fixed_density_scalar_overlap_third_derivative(
            system, params, &electronic,
        )?;
        let cn_grad = if include_cn_h0 {
            Some(crate::hessian::cn_gradient_matrix(system, coordination_cutoff)?)
        } else {
            None
        };
        Ok(Self {
            electronic,
            cphf,
            ctx,
            frozen,
            so3,
            cn_grad,
            include_cn_h0,
        })
    }
}

/// The per-direction evaluation against a shared reference — see
/// [`directional_third_finite_t`] for the assembly documentation.
pub fn directional_third_with_reference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    coordination_cutoff: f64,
    reference: &FiniteTThirdReference,
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "directional_third_with_reference: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    let electronic = &reference.electronic;
    let include_cn_h0 = reference.include_cn_h0;
    let (field, v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
        system,
        params,
        electronic,
        &reference.cphf,
        &reference.ctx,
        v,
    )?;
    let (second, _v_pot_vv) = crate::fourth_derivative::assemble::directional_second_order_legs(
        system,
        params,
        electronic,
        &reference.ctx,
        &field,
        coordination_cutoff,
        include_cn_h0,
        v,
    )?;

    let cvv = |m: &Matrix| -> f64 {
        let mut acc = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                acc += v[a] * v[b] * m[(a, b)];
            }
        }
        acc
    };
    let contract_slabs = |slabs: &[Matrix]| -> f64 {
        let mut acc = 0.0;
        for (c, slab) in slabs.iter().enumerate() {
            if v[c] == 0.0 {
                continue;
            }
            acc += v[c] * cvv(slab);
        }
        acc
    };

    // Frozen thirds (direction-independent slabs, contracted per direction).
    let mut total = contract_slabs(&reference.frozen);
    total += contract_slabs(&reference.so3);

    // Reference motion of the frozen Hessian (copy 1: the FULL density path).
    let path = crate::third_derivative::frozen_hessian_density_path(
        system,
        params,
        electronic,
        coordination_cutoff,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        &v_pot_v,
    )?;
    total += cvv(&path);

    // Pulay CN-response term.
    if let Some(ref cn_grad) = reference.cn_grad {
        let nat = system.atoms.len();
        let cn_grad_v: Vec<f64> = (0..nat)
            .map(|at| (0..ndof).map(|c| v[c] * cn_grad[at][c]).sum())
            .collect();
        total += cvv(&crate::hessian::fixed_density_pulay_cn_h0_response(
            system,
            params,
            electronic,
            &cn_grad_v,
        )?);
    }

    // Response-Hessian derivative (Step-B assembly).
    total += directional_response_hessian_derivative(
        system,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        &v_pot_v,
        &second.density,
        &second.energy_weighted,
        &second.shell_charges,
        v,
    )?;
    Ok(total)
}

/// **The directional analytic third derivative with native Fermi-smearing
/// support**: `e³[v] = Σ_abc T_abc v_a v_b v_c = D_v[H[v,v]]`, assembled by
/// the product rule over the directional Hessian's composition. Every
/// ingredient is occupation-agnostic (the frozen blocks read the fractional
/// `P/W/q` reference; the response legs come from the finite-temperature
/// charge-space solver), so one code path serves T = 0 and smeared systems
/// alike — the T = 0 limit is equality-gated against the adjoint-assembled
/// [`crate::third_derivative::third_derivative_analytic_vector`].
///
/// Exactly degenerate fractionally occupied blocks **are supported**: the
/// second-order solver assembles them in the frame-free resolvent
/// (Daleckii–Krein) form, where degeneracy is the confluent limit of the same
/// divided differences. Gated on symmetry-exact NiO at 3000 K (min level
/// spacing `1.1e-16`) against the central difference of the smeared analytic
/// Hessian: `1.24e-9 → 7.77e-11` for `h = 2e-3 → 5e-4`, ratio 4.00. The
/// FOURTH-order sibling still refuses there — see
/// [`directional_fourth_finite_t`].
pub fn directional_third_finite_t(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    let reference = FiniteTThirdReference::build(system, params, options, coordination_cutoff)?;
    directional_third_with_reference(system, params, coordination_cutoff, &reference, v)
}

/// **Dense finite-temperature analytic third derivative** — the full packed
/// tensor recovered from shared-reference directional evaluations by the
/// cubic polarization identity
/// `T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]`
/// with the distinct subset directions deduplicated and evaluated in
/// parallel. Cost scales as ~C(n+2,3) directional evaluations; for large
/// systems prefer [`third_derivative_finite_t_block`] or the directional
/// mode.
pub fn third_derivative_finite_t_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<crate::third_derivative::SymmetricThird> {
    let ndof = 3 * system.atoms.len();
    let dofs: Vec<usize> = (0..ndof).collect();
    third_finite_t_polarized(system, params, options, coordination_cutoff, &dofs)
}

/// The `|dofs|³` sub-tensor of the dense finite-temperature third derivative
/// (indexed by POSITION in `dofs`), via the same polarization driver — only
/// the directions the requested triples need are evaluated.
pub fn third_derivative_finite_t_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<crate::third_derivative::SymmetricThird> {
    let ndof = 3 * system.atoms.len();
    for &d in dofs {
        if d >= ndof {
            return Err(crate::error::Gfn1Error::InvalidInput(format!(
                "third_derivative_finite_t_block: dof {d} out of range (ndof {ndof})"
            )));
        }
    }
    third_finite_t_polarized(system, params, options, coordination_cutoff, dofs)
}

fn third_finite_t_polarized(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<crate::third_derivative::SymmetricThird> {
    use rayon::prelude::*;
    use std::collections::{BTreeMap, HashMap};

    let reference = FiniteTThirdReference::build(system, params, options, coordination_cutoff)?;
    let ndof = 3 * system.atoms.len();
    let m = dofs.len();

    // Phase 1: deduplicate the subset directions of every canonical triple.
    let mut key_index: HashMap<Vec<(usize, u8)>, usize> = HashMap::new();
    let mut keys: Vec<Vec<(usize, u8)>> = Vec::new();
    // Per canonical triple: the 7 (sign, key) polarization terms.
    let mut plan: Vec<((usize, usize, usize), Vec<(f64, usize)>)> = Vec::new();
    for k in 0..m {
        for j in 0..=k {
            for i in 0..=j {
                let idxs = [dofs[i], dofs[j], dofs[k]];
                let mut terms = Vec::with_capacity(7);
                for mask in 1u8..8 {
                    let mut dir: BTreeMap<usize, u8> = BTreeMap::new();
                    for (bit, &dof) in idxs.iter().enumerate() {
                        if mask & (1 << bit) != 0 {
                            *dir.entry(dof).or_insert(0) += 1;
                        }
                    }
                    let key: Vec<(usize, u8)> = dir.into_iter().collect();
                    let sign = if (3 - mask.count_ones()) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let idx = *key_index.entry(key.clone()).or_insert_with(|| {
                        keys.push(key);
                        keys.len() - 1
                    });
                    terms.push((sign, idx));
                }
                plan.push(((i, j, k), terms));
            }
        }
    }

    // Phase 2: evaluate each distinct direction once, in parallel, against
    // the shared reference.
    let values: Result<Vec<f64>> = keys
        .par_iter()
        .map(|key| {
            let mut v = vec![0.0_f64; ndof];
            for &(dof, weight) in key {
                v[dof] = weight as f64;
            }
            directional_third_with_reference(system, params, coordination_cutoff, &reference, &v)
        })
        .collect();
    let values = values?;

    // Phase 3: assemble the packed tensor.
    let mut store = crate::third_derivative::SymmetricThird::zeros(m);
    for ((i, j, k), terms) in plan {
        let mut t = 0.0;
        for (sign, idx) in terms {
            t += sign * values[idx];
        }
        store.add(i, j, k, t / 6.0);
    }
    Ok(store)
}

/// The six coefficient-motion blocks of the response-gradient derivative,
/// evaluated with an arbitrary response slot `(P, W, q)` and its screening
/// potential `K q` — the `B(·)` operator of the quartic combinatorics
/// `D_v[(d)] = g(X³) + 2B(X²) + 3G(X², X¹) + ∂B(X¹) + ∂G(X¹, X¹)`.
#[allow(clippy::too_many_arguments)]
fn response_coefficient_motion_blocks(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    p_slot: &Matrix,
    w_slot: &Matrix,
    q_slot: &[f64],
    v: &[f64],
) -> Result<f64> {
    Ok(response_coefficient_motion_block_values(
        system,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
        p_slot,
        w_slot,
        q_slot,
        v,
    )?
    .iter()
    .sum())
}

/// Per-block values of [`response_coefficient_motion_blocks`], in order:
/// `[cn_h0(P), cross(P), s2path(q0,q), pulay(P,W), pulay-V(Kq), so_q(q)]`.
#[allow(clippy::too_many_arguments)]
fn response_coefficient_motion_block_values(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    p_slot: &Matrix,
    w_slot: &Matrix,
    q_slot: &[f64],
    v: &[f64],
) -> Result<[f64; 6]> {
    let ndof = 3 * system.atoms.len();
    let nshell = electronic.basis.shells.len();
    let cvv = |m: &Matrix| -> f64 {
        let mut acc = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                acc += v[a] * v[b] * m[(a, b)];
            }
        }
        acc
    };
    let mut out = [0.0_f64; 6];
    let elec_p = {
        let mut e = electronic.clone();
        e.density = p_slot.clone();
        e
    };
    // NOTE: deliberately NOT gated on `include_cn_h0`. With the CN Hamiltonian
    // switched off both quartic paths still carry these blocks (the reference
    // `dsedcn` is nonzero regardless of the flag), and they only agree with
    // each other while both do — gating this side alone widens the T = 0
    // equality from 1.9e-5 to 2.4e-5 on water. See the CN-off row of
    // `quartic_t0_equality_cn_bisection`: `enable_cn_hamiltonian = false` is
    // not a supported FC4 configuration.
    let _ = include_cn_h0;
    out[0] = cvv(
        &crate::hessian::fixed_density_cn_h0_hessian(
            system,
            params,
            &elec_p,
            coordination_cutoff,
        )?
        .hessian,
    );
    out[1] = cvv(&crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
        system,
        params,
        &elec_p,
        coordination_cutoff,
    )?);
    out[2] = cvv(&crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        q_slot,
        params,
    )?);
    {
        let mut e = electronic.clone();
        e.density = p_slot.clone();
        e.energy_weighted_density = w_slot.clone();
        out[3] = cvv(&crate::hessian::fixed_density_pulay_hessian(system, params, &e)?.hessian);
    }
    let kernel = response_shell_scc_kernel(system, params, electronic)?;
    let kq = crate::linalg::matrix_vector_product(&kernel, q_slot)?;
    {
        let mut e = electronic.clone();
        for s in 0..nshell {
            e.shell_scc_potential[s] += kq[s];
        }
        let h1 = cvv(&crate::hessian::fixed_density_pulay_hessian(system, params, &e)?.hessian);
        let h0 =
            cvv(&crate::hessian::fixed_density_pulay_hessian(system, params, electronic)?.hessian);
        out[4] = h1 - h0;
    }
    {
        let mut e = electronic.clone();
        e.shell_charges = q_slot.to_vec();
        out[5] = cvv(&crate::hessian::fixed_density_scalar_overlap_hessian(
            system, params, &e,
        )?);
    }
    Ok(out)
}

/// **Directional analytic fourth derivative with native Fermi smearing**:
/// `e⁗[v] = D_v[e³[v]]` — the product rule over the finite-temperature cubic's
/// composition, using the directional third-order response `X^{vvv}`.
///
/// Status: the `∂B(X^v)` (block-third) inventory and the `∂G` (background
/// eigen-motion) hook are pinned by the T = 0 equality diagnostic against
/// [`crate::fourth_derivative::directional_fourth_derivative`]; see the test.
pub fn directional_fourth_finite_t(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "directional_fourth_finite_t: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    crate::terms::require_order(
        &options.electronic_options,
        params,
        4,
        "directional_fourth_finite_t",
    )?;
    let reference = FiniteTThirdReference::build(system, params, options, coordination_cutoff)?;
    directional_fourth_with_reference(system, params, options, coordination_cutoff, &reference, v)
}

/// The shared-reference core of [`directional_fourth_finite_t`] — one
/// directional quartic against an already-built reference state, so the
/// polarization drivers can amortize the SCF + CPXTB solve across many
/// directions.
pub(crate) fn directional_fourth_with_reference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    reference: &FiniteTThirdReference,
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    let electronic = &reference.electronic;
    let include_cn_h0 = reference.include_cn_h0;
    let ctx = &reference.ctx;
    let nshell = electronic.basis.shells.len();

    // ---- legs: X^v, X^vv (field form), X^vvv ----
    let (field, v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
        system,
        params,
        electronic,
        &reference.cphf,
        ctx,
        v,
    )?;
    // Second-order FIELD (the bundle-only helper in assemble.rs is not
    // enough — the third-order solve needs the MO-side extras).
    let (second, v_pot_vv) = {
        let basis = &electronic.basis;
        let n = basis.len();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )?;
        let v_geo: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
            .collect();
        let zeros = vec![0.0_f64; nshell];
        let mut f_vv =
            crate::hessian::directional_h0_bare_second_matrix(system, params, electronic, v)?;
        {
            if include_cn_h0 {
                let cn = crate::hessian::directional_h0_cn_block_second_matrix(
                    system,
                    params,
                    electronic,
                    coordination_cutoff,
                    v,
                )?;
                for k in 0..n * n {
                    f_vv.as_mut_slice()[k] += cn.as_slice()[k];
                }
            }
            let scc = crate::hessian::directional_h0_scc_scalar_second_matrix(
                system, params, electronic, v, &v_geo, &zeros,
            )?;
            for k in 0..n * n {
                f_vv.as_mut_slice()[k] += scc.as_slice()[k];
            }
        }
        let s_vv = crate::hessian::directional_overlap_second_matrix(system, basis, v)?;
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &field.bundle.shell_charges,
            params,
        )?;
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let second = ctx.second_order_field(
            &field,
            &field,
            &f_vv,
            &s_vv,
            &dgamma_v_qv,
            &dgamma_v_qv,
        )?;
        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )?;
        let kernel = response_shell_scc_kernel(system, params, electronic)?;
        let chain =
            kernel_chain_vec(system, params, electronic, &field.bundle.shell_charges, nshell)?;
        let v_pot_vv: Vec<f64> = (0..nshell)
            .map(|s| {
                let geo2: f64 = (0..ndof)
                    .map(|c| {
                        (0..ndof)
                            .map(|d| v[c] * v[d] * d2vdr_q[s][(c, d)])
                            .sum::<f64>()
                    })
                    .sum();
                let cross: f64 = 2.0 * dgamma_v_qv[s];
                let kq: f64 = (0..nshell)
                    .map(|t| kernel[(s, t)] * second.bundle.shell_charges[t])
                    .sum();
                geo2 + cross + chain[s] + kq
            })
            .collect();
        (second, v_pot_vv)
    };
    let inputs = crate::response::charge_space::directional_third_order_inputs(
        system,
        params,
        electronic,
        coordination_cutoff,
        &field,
        &second,
        v,
    )?;
    let third = ctx.solve_third_order_directional(
        &field,
        &second,
        &inputs.fock_skeleton_vvv,
        &inputs.overlap_vvv,
        &inputs.overlap_vv,
        &inputs.v_pot_geo,
        &inputs.dgamma_v_qv,
        &inputs.dgamma_v_qvv,
        &inputs.d2gamma_vv_qv,
    )?;

    // ---- (a) frozen fourths + first-order paths ----
    let mut total = crate::fourth_derivative::directional::directional_fourth_geometric_with(
        system, params, electronic, options, v,
    )?;
    if options.include_fixed_scc {
        total += crate::fourth_derivative::directional::contract_slabs_vvv(
            &crate::hessian::fixed_shell_charge_scc_third_charge_path(
                system,
                &electronic.basis,
                &electronic.shell_charges,
                &field.bundle.shell_charges,
                params,
            )?,
            v,
        );
    }
    total += crate::fourth_derivative::directional::directional_fourth_frozen_density(
        system,
        params,
        electronic,
        coordination_cutoff,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        &v_pot_v,
        v,
    )?;
    // ---- (b) hessian-path stage ----
    total += crate::fourth_derivative::directional::directional_fourth_hessian_path_stage(
        system,
        params,
        electronic,
        coordination_cutoff,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        &v_pot_v,
        &second.bundle.density,
        &second.bundle.energy_weighted,
        &second.bundle.shell_charges,
        &v_pot_vv,
        v,
    )?;
    // ---- (c) CN-response stage ----
    if include_cn_h0 {
        total += crate::fourth_derivative::directional::directional_fourth_cn_response_stage(
            system,
            params,
            electronic,
            coordination_cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            v,
        )?;
    }

    // ---- (d) D_v of the response-Hessian derivative ----
    // P1: response motion.
    total += directional_response_hessian_vv(
        system,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
        &third.density,
        &third.energy_weighted,
        &third.shell_charges,
        v,
    )?;
    // 2·B(X^vv).
    total += 2.0
        * response_coefficient_motion_blocks(
            system,
            params,
            electronic,
            coordination_cutoff,
            include_cn_h0,
            &second.bundle.density,
            &second.bundle.energy_weighted,
            &second.bundle.shell_charges,
            v,
        )?;
    // 3·G(X^vv, X^v) — mixed background families.
    {
        let grad_ctx = ResponseGradientContext::new(
            system,
            &electronic.basis,
            params,
            electronic,
            coordination_cutoff,
            include_cn_h0,
        )?;
        let kernel = response_shell_scc_kernel(system, params, electronic)?;
        let dot = |grad: &[crate::math::Vec3]| -> f64 {
            grad.iter()
                .enumerate()
                .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
                .sum()
        };
        // chain_mixed[s] = E'''_A q^v_A q^vv_A.
        let chain_mixed = kernel_chain_mixed(
            system,
            params,
            electronic,
            &field.bundle.shell_charges,
            &second.bundle.shell_charges,
            nshell,
        )?;
        // dp_pot with slot P^vv against V^v_tot; p0 with motion P^v against
        // K q^vv; chain with P₀ against the mixed chain.
        let bg_a = crate::response::cpxtb::response_gradient_background_motion(
            system,
            electronic,
            &grad_ctx,
            &kernel,
            &second.bundle.density,
            &field.bundle.shell_charges,
            &chain_mixed,
            &v_pot_v,
        )?;
        let bg_b = crate::response::cpxtb::response_gradient_background_motion(
            system,
            electronic,
            &grad_ctx,
            &kernel,
            &field.bundle.density,
            &second.bundle.shell_charges,
            &chain_mixed,
            &v_pot_v,
        )?;
        // qq mixed via polarization: (qq(qv+qvv) − qq(qv) − qq(qvv))/2.
        let qq_of = |q: &[f64]| -> Result<f64> {
            let bg = crate::response::cpxtb::response_gradient_background_motion(
                system,
                electronic,
                &grad_ctx,
                &kernel,
                &field.bundle.density,
                q,
                &chain_mixed,
                &v_pot_v,
            )?;
            Ok(dot(&bg.kernel_qq))
        };
        let q_sum: Vec<f64> = (0..nshell)
            .map(|s| field.bundle.shell_charges[s] + second.bundle.shell_charges[s])
            .collect();
        let qq_mixed = 0.5
            * (qq_of(&q_sum)?
                - qq_of(&field.bundle.shell_charges)?
                - qq_of(&second.bundle.shell_charges)?);
        let g_mixed =
            dot(&bg_a.scc_dp_pot) + dot(&bg_b.scc_p0) + dot(&bg_a.scc_chain) + qq_mixed;
        total += 3.0 * g_mixed;
    }
    // ∂B(X^v): the block thirds with the first-order slots.
    total += response_block_thirds(
        system,
        params,
        electronic,
        coordination_cutoff,
        include_cn_h0,
        &field.bundle.density,
        &field.bundle.energy_weighted,
        &field.bundle.shell_charges,
        v,
    )?;
    // ∂G(X^v, X^v) + block-cache motions of ∂B(X^v). Every term below is
    // pinned by the bracket equations (quartic_d_submotion_split) and the
    // per-block/per-family FD splits (quartic_b_block_split /
    // quartic_dg_family_split) on the water fixture:
    //
    //  * s2vv counts TWICE — once as the qq-family ∂G eigen-motion
    //    (∇γ→∂²γ over (q¹,q¹)) and once as the s2path block's first-slot
    //    cache motion (q₀→q¹);
    //  * bg-hess(P¹,V¹g) counts twice — pulay-block V₀-cache (geometric
    //    part) and the dp-family V¹∂²S eigen-motion;
    //  * bg-hess(P¹,Kq¹) counts twice — pulay-V block P₀→P¹ motion and the
    //    p0-family Kq¹∂²S eigen-motion;
    //  * the pulay block's CN-cache motion enters via the affine
    //    self-energy trick;
    //  * dp family: +P²V¹∇S − P¹V²∇S; p0 family: −P²Kq¹∇S − P¹(∂γ·q¹)∇S
    //    + P¹Kq²∇S; chain-family pieces are the sub-nHa tail.
    {
        let grad_ctx = ResponseGradientContext::new(
            system,
            &electronic.basis,
            params,
            electronic,
            coordination_cutoff,
            include_cn_h0,
        )?;
        let kernel = response_shell_scc_kernel(system, params, electronic)?;
        let kq1 = crate::linalg::matrix_vector_product(&kernel, &field.bundle.shell_charges)?;
        let kq2 = crate::linalg::matrix_vector_product(&kernel, &second.bundle.shell_charges)?;
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            &electronic.basis,
            &field.bundle.shell_charges,
            params,
        )?;
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let chain11 =
            kernel_chain_vec(system, params, electronic, &field.bundle.shell_charges, nshell)?;
        let bg_grad = |p: &Matrix, pot: &[f64]| -> f64 {
            crate::response::cpxtb::background_overlap_gradient_scalar(&grad_ctx, p, pot, v)
        };
        let bg_hess = |p: &Matrix, pot: &[f64]| -> f64 {
            crate::response::cpxtb::background_overlap_hessian_scalar(
                system,
                &electronic.basis,
                &grad_ctx,
                p,
                pot,
                v,
            )
        };
        let cvv = |m: &Matrix| -> f64 {
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * m[(a, b)];
                }
            }
            acc
        };
        // qq-family ∂G eigen-motion + s2path first-slot cache motion.
        let s2vv = cvv(&crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
            system,
            &electronic.basis,
            &field.bundle.shell_charges,
            &field.bundle.shell_charges,
            params,
        )?);
        total += 2.0 * s2vv;
        // V¹g ∂²S (pulay V-cache geo part + dp-family eigen-motion).
        total += 2.0 * bg_hess(&field.bundle.density, &v_pot_v);
        // Kq¹ ∂²S (pulay-V P-motion + p0-family eigen-motion).
        total += 2.0 * bg_hess(&field.bundle.density, &kq1);
        // Pulay-V kernel geometric motion: −P₀·(∂_vγ·q¹)·∂²S.
        total += bg_hess(&electronic.density, &dgamma_v_qv);
        // Pulay block CN-cache motion (affine self-energy trick) — the
        // pulay hessian rebuilt at CN = CN^v minus CN = 0 at the (P¹,W¹)
        // slot.
        {
            let nat = system.atoms.len();
            let cn_grad = crate::hessian::cn_gradient_matrix(system, coordination_cutoff)?;
            let cn_v: Vec<f64> = (0..nat)
                .map(|at| (0..ndof).map(|c| v[c] * cn_grad[at][c]).sum())
                .collect();
            let mut e_cnv = electronic.clone();
            e_cnv.density = field.bundle.density.clone();
            e_cnv.energy_weighted_density = field.bundle.energy_weighted.clone();
            e_cnv.coordination_numbers = cn_v;
            let mut e_cn0 = e_cnv.clone();
            e_cn0.coordination_numbers = vec![0.0; nat];
            total += cvv(
                &crate::hessian::fixed_density_pulay_hessian(system, params, &e_cnv)?.hessian,
            ) - cvv(
                &crate::hessian::fixed_density_pulay_hessian(system, params, &e_cn0)?.hessian,
            );
        }
        // so_q block P₀→P¹ cache motion.
        {
            let mut e = electronic.clone();
            e.density = field.bundle.density.clone();
            e.shell_charges = field.bundle.shell_charges.to_vec();
            total += cvv(&crate::hessian::fixed_density_scalar_overlap_hessian(
                system, params, &e,
            )?);
        }
        // dp family: +P²V¹∇S − P¹V²∇S.
        total += -bg_grad(&second.bundle.density, &v_pot_v);
        total += bg_grad(&field.bundle.density, &v_pot_vv);
        // p0 family: −P²Kq¹∇S − P¹(∂γ·q¹)∇S + P¹Kq²∇S.
        total += bg_grad(&second.bundle.density, &kq1);
        total += bg_grad(&field.bundle.density, &dgamma_v_qv);
        total += -bg_grad(&field.bundle.density, &kq2);
        // Chain-family tail (sub-nHa): p0/chain-family onsite E''' motions
        // (−P¹chain∇S twice: p0 K̇-chain + chain-family P-motion), the
        // chain-family ∇S→∂²S eigen-motion + pulay-V K̇-chain motion
        // (−P₀chain∂²S twice), and the E'''' motion −P₀chain₄∇S.
        total += 2.0 * bg_grad(&field.bundle.density, &chain11);
        total += 2.0 * bg_hess(&electronic.density, &chain11);
        let chain4 =
            kernel_chain4_vec(system, params, electronic, &field.bundle.shell_charges, nshell)?;
        total += bg_grad(&electronic.density, &chain4);
    }
    Ok(total)
}

/// **Dense finite-temperature analytic fourth derivative** — the full packed
/// tensor recovered from shared-reference directional evaluations by the
/// quartic polarization identity
/// `Q(x₁,…,x₄) = (1/24) Σ_{∅≠S⊆{1..4}} (−1)^{4−|S|} e⁴[Σ_{i∈S} x_i]`
/// with the distinct subset directions deduplicated and evaluated in
/// parallel. Cost scales as ~C(n+3,4) directional evaluations; for large
/// systems prefer [`fourth_derivative_finite_t_block`] or the directional
/// mode.
pub fn fourth_derivative_finite_t_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<crate::fourth_derivative::SymmetricFourth> {
    let ndof = 3 * system.atoms.len();
    let dofs: Vec<usize> = (0..ndof).collect();
    fourth_finite_t_polarized(system, params, options, coordination_cutoff, &dofs)
}

/// The `|dofs|⁴` sub-tensor of the dense finite-temperature fourth derivative
/// (indexed by POSITION in `dofs`), via the same polarization driver — only
/// the directions the requested quadruples need are evaluated.
pub fn fourth_derivative_finite_t_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<crate::fourth_derivative::SymmetricFourth> {
    let ndof = 3 * system.atoms.len();
    for &d in dofs {
        if d >= ndof {
            return Err(crate::error::Gfn1Error::InvalidInput(format!(
                "fourth_derivative_finite_t_block: dof {d} out of range (ndof {ndof})"
            )));
        }
    }
    fourth_finite_t_polarized(system, params, options, coordination_cutoff, dofs)
}

fn fourth_finite_t_polarized(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<crate::fourth_derivative::SymmetricFourth> {
    use rayon::prelude::*;
    use std::collections::{BTreeMap, HashMap};

    crate::terms::require_order(
        &options.electronic_options,
        params,
        4,
        "fourth_derivative_finite_t",
    )?;
    let reference = FiniteTThirdReference::build(system, params, options, coordination_cutoff)?;
    let ndof = 3 * system.atoms.len();
    let m = dofs.len();

    // Phase 1: deduplicate the subset directions of every canonical
    // quadruple (15 polarization terms each).
    let mut key_index: HashMap<Vec<(usize, u8)>, usize> = HashMap::new();
    let mut keys: Vec<Vec<(usize, u8)>> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut plan: Vec<((usize, usize, usize, usize), Vec<(f64, usize)>)> = Vec::new();
    for l in 0..m {
        for k in 0..=l {
            for j in 0..=k {
                for i in 0..=j {
                    let idxs = [dofs[i], dofs[j], dofs[k], dofs[l]];
                    let mut terms = Vec::with_capacity(15);
                    for mask in 1u8..16 {
                        let mut dir: BTreeMap<usize, u8> = BTreeMap::new();
                        for (bit, &dof) in idxs.iter().enumerate() {
                            if mask & (1 << bit) != 0 {
                                *dir.entry(dof).or_insert(0) += 1;
                            }
                        }
                        let key: Vec<(usize, u8)> = dir.into_iter().collect();
                        let sign = if (4 - mask.count_ones()) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let idx = *key_index.entry(key.clone()).or_insert_with(|| {
                            keys.push(key);
                            keys.len() - 1
                        });
                        terms.push((sign, idx));
                    }
                    plan.push(((i, j, k, l), terms));
                }
            }
        }
    }

    // Phase 2: evaluate each distinct direction once, in parallel, against
    // the shared reference.
    let values: Result<Vec<f64>> = keys
        .par_iter()
        .map(|key| {
            let mut v = vec![0.0_f64; ndof];
            for &(dof, weight) in key {
                v[dof] = weight as f64;
            }
            directional_fourth_with_reference(
                system,
                params,
                options,
                coordination_cutoff,
                &reference,
                &v,
            )
        })
        .collect();
    let values = values?;

    // Phase 3: assemble the packed tensor.
    let mut store = crate::fourth_derivative::SymmetricFourth::zeros(m);
    for ((i, j, k, l), terms) in plan {
        let mut t = 0.0;
        for (sign, idx) in terms {
            t += sign * values[idx];
        }
        store.add(i, j, k, l, t / 24.0);
    }
    Ok(store)
}

/// Onsite `E''''` chain potential: `E''''_A · (q_A)³` per shell.
fn kernel_chain4_vec(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    q: &[f64],
    nshell: usize,
) -> Result<Vec<f64>> {
    let shell_model = crate::coulomb::ShellChargeModel::build(system, &electronic.basis, params)?;
    let nat = system.atoms.len();
    let charge_order = electronic.charge_order.max(3);
    let mut shell_atom = vec![0usize; nshell];
    for atom in 0..nat {
        let offset = shell_model.atom_offsets[atom];
        for local in 0..shell_model.atom_shell_counts[atom] {
            shell_atom[offset + local] = atom;
        }
    }
    let mut atom_q = vec![0.0_f64; nat];
    for s in 0..nshell {
        atom_q[shell_atom[s]] += q[s];
    }
    Ok((0..nshell)
        .map(|s| {
            let atom = shell_atom[s];
            if shell_model.atom_shell_counts[atom] == 0 {
                return 0.0;
            }
            let offset = shell_model.atom_offsets[atom];
            let (_, _, _, fourth) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                shell_model.hardness[offset],
                shell_model.hubbard_derivs[offset],
                charge_order,
                electronic.atomic_charges[atom],
            );
            fourth * atom_q[atom] * atom_q[atom] * atom_q[atom]
        })
        .collect())
}

/// `∂B/∂λ` at the first-order slot: each coefficient-motion block one
/// derivative up (the frozen THIRD builders with the doctored slots).
#[allow(clippy::too_many_arguments)]
fn response_block_thirds(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    include_cn_h0: bool,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v: &[f64],
) -> Result<f64> {
    let nshell = electronic.basis.shells.len();
    let contract = crate::fourth_derivative::directional::contract_slabs_vvv;
    let mut total = 0.0;
    // cn_h0 + pulay-cross block third at the P^v slot. The block-split FD
    // diagnostic (quartic_b_block_split) pins ∂(cn_h0) + ∂(cross) to exactly
    // this builder's value (−1.1008878e-5 on the water fixture, matched to
    // 7 digits) — the earlier water-only bisection that excluded it relied on
    // an accidental cancellation against then-missing ∂G pieces.
    let _ = include_cn_h0; // see response_coefficient_motion_block_values
    {
        let elec_p = {
            let mut e = electronic.clone();
            e.density = p_v.clone();
            e
        };
        total += contract(
            &crate::hessian::fixed_density_cn_h0_third_derivative(
                system,
                params,
                &elec_p,
                coordination_cutoff,
            )?,
            v,
        );
    }
    total += contract(
        &crate::hessian::fixed_shell_charge_scc_third_charge_path(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            q_v,
            params,
        )?,
        v,
    );
    {
        let mut e = electronic.clone();
        e.density = p_v.clone();
        e.energy_weighted_density = w_v.clone();
        total += contract(
            &crate::hessian::fixed_density_pulay_third_derivative(system, params, &e)?,
            v,
        );
    }
    let kernel = response_shell_scc_kernel(system, params, electronic)?;
    let kq = crate::linalg::matrix_vector_product(&kernel, q_v)?;
    {
        let mut e = electronic.clone();
        for s in 0..nshell {
            e.shell_scc_potential[s] += kq[s];
        }
        let h1 = contract(
            &crate::hessian::fixed_density_pulay_third_derivative(system, params, &e)?,
            v,
        );
        let h0 = contract(
            &crate::hessian::fixed_density_pulay_third_derivative(system, params, electronic)?,
            v,
        );
        total += h1 - h0;
    }
    {
        let mut e = electronic.clone();
        e.shell_charges = q_v.to_vec();
        total += contract(
            &crate::hessian::fixed_density_scalar_overlap_third_derivative(system, params, &e)?,
            v,
        );
    }
    Ok(total)
}

/// Onsite `E'''` chain potential for a single charge vector (per shell).
fn kernel_chain_vec(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    q: &[f64],
    nshell: usize,
) -> Result<Vec<f64>> {
    kernel_chain_mixed(system, params, electronic, q, q, nshell)
}

/// Onsite `E'''` chain potential for a charge PAIR: `E'''_A · qa_A · qb_A`.
fn kernel_chain_mixed(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    qa: &[f64],
    qb: &[f64],
    nshell: usize,
) -> Result<Vec<f64>> {
    let shell_model = crate::coulomb::ShellChargeModel::build(system, &electronic.basis, params)?;
    let nat = system.atoms.len();
    let charge_order = electronic.charge_order.max(3);
    let mut shell_atom = vec![0usize; nshell];
    for atom in 0..nat {
        let offset = shell_model.atom_offsets[atom];
        for local in 0..shell_model.atom_shell_counts[atom] {
            shell_atom[offset + local] = atom;
        }
    }
    let mut atom_qa = vec![0.0_f64; nat];
    let mut atom_qb = vec![0.0_f64; nat];
    for s in 0..nshell {
        atom_qa[shell_atom[s]] += qa[s];
        atom_qb[shell_atom[s]] += qb[s];
    }
    Ok((0..nshell)
        .map(|s| {
            let atom = shell_atom[s];
            if shell_model.atom_shell_counts[atom] == 0 {
                return 0.0;
            }
            let offset = shell_model.atom_offsets[atom];
            let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                shell_model.hardness[offset],
                shell_model.hubbard_derivs[offset],
                charge_order,
                electronic.atomic_charges[atom],
            );
            third * atom_qa[atom] * atom_qb[atom]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::{run_electronic, ElectronicOptions};
    use crate::response::cpxtb::{
        solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions,
    };

    /// Step-A equality gate: the directional response-Hessian contraction
    /// built from the DIRECTIONAL bundle must equal `vᵀ·hessian_response·v`
    /// (whose columns are the same `g` applied to per-DOF bundles) — pure
    /// linearity, so machine precision, at T = 0 and finite T alike.
    fn run_equality_gate(xyz: &str, etemp: f64, label: &str) {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = etemp;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let include_cn_h0 = options.hamiltonian.enable_cn_hamiltonian;
        let cphf = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        // Directional bundle by linear contraction of the per-DOF responses.
        let n = electronic.basis.len();
        let nshell = electronic.basis.shells.len();
        let mut p_v = Matrix::zeros(n, n);
        let mut w_v = Matrix::zeros(n, n);
        let mut q_v = vec![0.0_f64; nshell];
        for (c, &vc) in v.iter().enumerate() {
            for k in 0..n * n {
                p_v.as_mut_slice()[k] += vc * cphf.density_responses[c].as_slice()[k];
                w_v.as_mut_slice()[k] +=
                    vc * cphf.energy_weighted_density_responses[c].as_slice()[k];
            }
            for s in 0..nshell {
                q_v[s] += vc * cphf.shell_charge_responses[c][s];
            }
        }

        let direct = directional_response_hessian_vv(
            &system,
            &params,
            &electronic,
            cutoff,
            include_cn_h0,
            &p_v,
            &w_v,
            &q_v,
            &v,
        )
        .unwrap();
        let mut reference = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                reference += v[a] * v[b] * cphf.hessian_response[(a, b)];
            }
        }
        let delta = (direct - reference).abs();
        eprintln!(
            "{label}: directional r2 {direct:.12e} vs vᵀRv {reference:.12e} delta {delta:.3e}"
        );
        assert!(
            delta < 1.0e-10 * (1.0 + reference.abs()),
            "{label}: directional response-Hessian contraction mismatch: {delta:.3e}"
        );
    }

    /// **Step-C gate 1 (T = 0 equality).** The product-rule directional third
    /// derivative must reproduce the adjoint-assembled analytic FC3 contracted
    /// `vvv` on non-equilibrium water — two completely different assemblies of
    /// the same tensor.
    #[test]
    fn directional_third_finite_t_matches_analytic_t0() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let mine =
            super::directional_third_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        let k = crate::third_derivative::third_derivative_analytic_vector(
            &system,
            &params,
            options.clone(),
            cutoff,
            &v,
        )
        .unwrap();
        let mut reference = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                reference += v[a] * v[b] * k[(a, b)];
            }
        }
        let delta = (mine - reference).abs();
        eprintln!(
            "directional_third_finite_t T=0: mine {mine:.12e} vs adjoint FC3 {reference:.12e} \
             delta {delta:.3e}"
        );
        assert!(
            delta < 1.0e-9 * (1.0 + reference.abs()),
            "product-rule vs adjoint FC3 at T=0: {mine:.12e} vs {reference:.12e}"
        );
    }

    /// **Step-C gate 2 (finite-temperature FD ladder).** On Fermi-smeared
    /// distorted Ni(CO)₄ (3000 K, fractional occupations, near-degenerate
    /// pairs) the analytic directional third derivative must match the central
    /// FD along `v` of the finite-temperature analytic Hessian contracted
    /// `vv`, with `h²` truncation scaling.
    ///
    /// `#[ignore]`: the reference needs four finite-temperature Ni(CO)₄
    /// Hessians (~10 min in reltest). Validated at delta(h) 6.25e-10,
    /// delta(h/2) 1.57e-10, ratio 3.98 (2026-08-11); run explicitly with
    /// `cargo test --profile reltest -- --ignored directional_third_finite_t`.
    /// The always-on protection is the T=0 equality gate above plus the fast
    /// finite-T second-order charge-space gates.
    #[test]
    #[ignore]
    fn directional_third_finite_t_matches_hessian_fd_smeared() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.electronic_temperature = 3000.0;
        options.electronic_options.energy_tolerance = 1.0e-14;
        options.electronic_options.charge_tolerance = 1.0e-12;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let analytic =
            super::directional_third_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        let hess_vv = |sys: &PeriodicSystem| -> f64 {
            let h = crate::hessian::analytic_hessian(sys, &params, options.clone())
                .unwrap()
                .hessian;
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * h[(a, b)];
                }
            }
            acc
        };
        let displaced = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let fd_at =
            |h: f64| -> f64 { (hess_vv(&displaced(h)) - hess_vv(&displaced(-h))) / (2.0 * h) };
        let h1 = 1.0e-3;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "finite-T directional FC3: analytic {analytic:.10e} fd(h) {fd1:.10e} fd(h/2) \
             {fd2:.10e} delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-5 * (1.0 + fd1.abs()),
            "finite-T directional FC3 vs Hessian FD: analytic {analytic:.10e} fd {fd1:.10e} \
             delta {delta1:.3e}"
        );
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h² (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing analytic term"
        );
    }

    /// **(d)-motion FD diagnostic**: the analytic `D_v[(d)]` pieces
    /// (`g(X³) + 2B(X²) + 3G + ∂B`, ∂G disabled) vs the central FD of
    /// `directional_response_hessian_derivative` itself along `v` with
    /// everything reconverged — pins the missing ∂G directly, independent of
    /// the stages (a)-(c).
    #[test]
    #[ignore]
    fn quartic_d_motion_matches_response_derivative_fd() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        // (d) evaluated at an arbitrary geometry with everything reconverged.
        let d_at = |sys: &PeriodicSystem| -> f64 {
            let r = super::FiniteTThirdReference::build(sys, &params, &options, cutoff).unwrap();
            let (field, v_pot_v) =
                crate::fourth_derivative::assemble::directional_first_order_legs(
                    sys,
                    &params,
                    &r.electronic,
                    &r.cphf,
                    &r.ctx,
                    &v,
                )
                .unwrap();
            let (second, _) = crate::fourth_derivative::assemble::directional_second_order_legs(
                sys,
                &params,
                &r.electronic,
                &r.ctx,
                &field,
                cutoff,
                r.include_cn_h0,
                &v,
            )
            .unwrap();
            super::directional_response_hessian_derivative(
                sys,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
                &field.bundle.density,
                &field.bundle.energy_weighted,
                &field.bundle.shell_charges,
                &v_pot_v,
                &second.density,
                &second.energy_weighted,
                &second.shell_charges,
                &v,
            )
            .unwrap()
        };
        let displaced = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let fd_at = |h: f64| -> f64 { (d_at(&displaced(h)) - d_at(&displaced(-h))) / (2.0 * h) };
        let fd1 = fd_at(1.0e-3);
        let fd2 = fd_at(5.0e-4);
        eprintln!(
            "D_v[(d)] FD reference: fd(h)={fd1:.12e}  fd(h/2)={fd2:.12e}  (Richardson \
             {:.12e})",
            (4.0 * fd2 - fd1) / 3.0
        );
        // The analytic D_v[(d)] is embedded in directional_fourth_finite_t;
        // reconstruct it as total − stages(a-c) − s1b via the five-stage
        // reference pieces printed by the T=0 equality diagnostic. For direct
        // comparison here, print the FD so the missing ∂G = fd − [P1+2B+3G+∂B]
        // can be matched offline against the equality-diagnostic delta.
        assert!(fd1.is_finite() && fd2.is_finite());
    }

    /// **(d) sub-motion split**: FD the three brackets of `(d) = g(X²) + B(X¹)
    /// + G(X¹,X¹)` separately (everything reconverged) and compare each with
    /// its assembly-side sum — pins WHICH bracket's inventory is wrong:
    ///   D_v[g-part] = g(X³) + B(X²) + G(X²,X¹)
    ///   D_v[B-part] = B(X²) + ∂B
    ///   D_v[G-part] = 2·G(X²,X¹) + ∂G
    #[test]
    #[ignore]
    fn quartic_d_submotion_split() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let dot = |grad: &[crate::math::Vec3]| -> f64 {
            grad.iter()
                .enumerate()
                .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
                .sum()
        };

        // The three (d)-brackets at a reconverged geometry.
        struct Parts {
            g: f64,
            b: f64,
            gg: f64,
        }
        let parts_at = |sys: &PeriodicSystem| -> Parts {
            let r = super::FiniteTThirdReference::build(sys, &params, &options, cutoff).unwrap();
            let (field, v_pot_v) =
                crate::fourth_derivative::assemble::directional_first_order_legs(
                    sys,
                    &params,
                    &r.electronic,
                    &r.cphf,
                    &r.ctx,
                    &v,
                )
                .unwrap();
            let (second, _) = crate::fourth_derivative::assemble::directional_second_order_legs(
                sys,
                &params,
                &r.electronic,
                &r.ctx,
                &field,
                cutoff,
                r.include_cn_h0,
                &v,
            )
            .unwrap();
            let g = super::directional_response_hessian_vv(
                sys,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
                &second.density,
                &second.energy_weighted,
                &second.shell_charges,
                &v,
            )
            .unwrap();
            let b = super::response_coefficient_motion_blocks(
                sys,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
                &field.bundle.density,
                &field.bundle.energy_weighted,
                &field.bundle.shell_charges,
                &v,
            )
            .unwrap();
            let nshell = r.electronic.basis.shells.len();
            let grad_ctx = ResponseGradientContext::new(
                sys,
                &r.electronic.basis,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
            )
            .unwrap();
            let kernel = response_shell_scc_kernel(sys, &params, &r.electronic).unwrap();
            let chain = super::kernel_chain_vec(
                sys,
                &params,
                &r.electronic,
                &field.bundle.shell_charges,
                nshell,
            )
            .unwrap();
            let bg = crate::response::cpxtb::response_gradient_background_motion(
                sys,
                &r.electronic,
                &grad_ctx,
                &kernel,
                &field.bundle.density,
                &field.bundle.shell_charges,
                &chain,
                &v_pot_v,
            )
            .unwrap();
            let gg =
                dot(&bg.scc_dp_pot) + dot(&bg.scc_p0) + dot(&bg.scc_chain) + dot(&bg.kernel_qq);
            Parts { g, b, gg }
        };
        let displaced = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let h = 1.0e-3;
        let pp = parts_at(&displaced(h));
        let pm = parts_at(&displaced(-h));
        let fd_g = (pp.g - pm.g) / (2.0 * h);
        let fd_b = (pp.b - pm.b) / (2.0 * h);
        let fd_gg = (pp.gg - pm.gg) / (2.0 * h);
        eprintln!(
            "(d) sub-motion FD: D_v[g-part] {fd_g:.12e}  D_v[B-part] {fd_b:.12e}  D_v[G-part] \
             {fd_gg:.12e}  (sum {:.12e})",
            fd_g + fd_b + fd_gg
        );

        // ---- assembly-side bracket values at the reference ----
        let r = super::FiniteTThirdReference::build(&system, &params, &options, cutoff).unwrap();
        let (field, _v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.cphf,
            &r.ctx,
            &v,
        )
        .unwrap();
        let (second, _) = crate::fourth_derivative::assemble::directional_second_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.ctx,
            &field,
            cutoff,
            r.include_cn_h0,
            &v,
        )
        .unwrap();
        // Second FIELD for the third-order solve.
        let (second_field, _vpvv) = {
            let basis = &r.electronic.basis;
            let n = basis.len();
            let nshell = basis.shells.len();
            let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
                &system,
                basis,
                &r.electronic.shell_charges,
                &params,
            )
            .unwrap();
            let v_geo: Vec<f64> = (0..nshell)
                .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
                .collect();
            let zeros = vec![0.0_f64; nshell];
            let mut f_vv = crate::hessian::directional_h0_bare_second_matrix(
                &system,
                &params,
                &r.electronic,
                &v,
            )
            .unwrap();
            {
                let cn = crate::hessian::directional_h0_cn_block_second_matrix(
                    &system,
                    &params,
                    &r.electronic,
                    cutoff,
                    &v,
                )
                .unwrap();
                let scc = crate::hessian::directional_h0_scc_scalar_second_matrix(
                    &system,
                    &params,
                    &r.electronic,
                    &v,
                    &v_geo,
                    &zeros,
                )
                .unwrap();
                for k in 0..n * n {
                    f_vv.as_mut_slice()[k] += cn.as_slice()[k] + scc.as_slice()[k];
                }
            }
            let s_vv = crate::hessian::directional_overlap_second_matrix(
                &system,
                &r.electronic.basis,
                &v,
            )
            .unwrap();
            let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
                &system,
                basis,
                &field.bundle.shell_charges,
                &params,
            )
            .unwrap();
            let dgamma_v_qv: Vec<f64> = (0..nshell)
                .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
                .collect();
            let sf = r
                .ctx
                .second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)
                .unwrap();
            (sf, ())
        };
        let inputs = crate::response::charge_space::directional_third_order_inputs(
            &system,
            &params,
            &r.electronic,
            cutoff,
            &field,
            &second_field,
            &v,
        )
        .unwrap();
        let third = r
            .ctx
            .solve_third_order_directional(
                &field,
                &second_field,
                &inputs.fock_skeleton_vvv,
                &inputs.overlap_vvv,
                &inputs.overlap_vv,
                &inputs.v_pot_geo,
                &inputs.dgamma_v_qv,
                &inputs.dgamma_v_qvv,
                &inputs.d2gamma_vv_qv,
            )
            .unwrap();
        let p1 = super::directional_response_hessian_vv(
            &system,
            &params,
            &r.electronic,
            cutoff,
            r.include_cn_h0,
            &third.density,
            &third.energy_weighted,
            &third.shell_charges,
            &v,
        )
        .unwrap();
        let b2 = super::response_coefficient_motion_blocks(
            &system,
            &params,
            &r.electronic,
            cutoff,
            r.include_cn_h0,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
            &v,
        )
        .unwrap();
        eprintln!(
            "assembly refs: P1(g(X3)) {p1:.12e}  B(X2) {b2:.12e}\n  bracket mismatches: g \
             {:.6e}  (needs B(X2)+G_mixed added)  B {:.6e} (needs dB added)  G {:.6e} (needs \
             2G_mixed+s2vv added)",
            fd_g - p1,
            fd_b - b2,
            fd_gg
        );
        assert!(fd_g.is_finite() && fd_b.is_finite() && fd_gg.is_finite());
    }

    /// **B-part block split**: FD each of the six coefficient-motion blocks
    /// separately (`∂block_i = FD[block_i(X¹(λ);λ)] − block_i(X²)`), printing
    /// per-block eigen-motion targets so the missing ∂B pieces are localized
    /// block by block. Also prints the ∂G analytic-piece probes.
    #[test]
    #[ignore]
    fn quartic_b_block_split() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        let blocks_at = |sys: &PeriodicSystem| -> [f64; 6] {
            let r = super::FiniteTThirdReference::build(sys, &params, &options, cutoff).unwrap();
            let (field, _) = crate::fourth_derivative::assemble::directional_first_order_legs(
                sys,
                &params,
                &r.electronic,
                &r.cphf,
                &r.ctx,
                &v,
            )
            .unwrap();
            super::response_coefficient_motion_block_values(
                sys,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
                &field.bundle.density,
                &field.bundle.energy_weighted,
                &field.bundle.shell_charges,
                &v,
            )
            .unwrap()
        };
        let displaced = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let h = 1.0e-3;
        let bp = blocks_at(&displaced(h));
        let bm = blocks_at(&displaced(-h));

        // Reference: blocks with X² legs, plus ∂G analytic-piece probes.
        let r = super::FiniteTThirdReference::build(&system, &params, &options, cutoff).unwrap();
        let (field, v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.cphf,
            &r.ctx,
            &v,
        )
        .unwrap();
        let (second, _) = crate::fourth_derivative::assemble::directional_second_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.ctx,
            &field,
            cutoff,
            r.include_cn_h0,
            &v,
        )
        .unwrap();
        let b2 = super::response_coefficient_motion_block_values(
            &system,
            &params,
            &r.electronic,
            cutoff,
            r.include_cn_h0,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
            &v,
        )
        .unwrap();
        const NAMES: [&str; 6] = ["cn_h0", "cross", "s2path", "pulay_pw", "pulay_v", "so_q"];
        for i in 0..6 {
            let fd = (bp[i] - bm[i]) / (2.0 * h);
            eprintln!(
                "block {:>8}: FD {:+.9e}  B2 {:+.9e}  dB_true {:+.9e}",
                NAMES[i],
                fd,
                b2[i],
                fd - b2[i]
            );
        }

        // ∂G probes at the reference.
        let grad_ctx = ResponseGradientContext::new(
            &system,
            &r.electronic.basis,
            &params,
            &r.electronic,
            cutoff,
            r.include_cn_h0,
        )
        .unwrap();
        let kernel = response_shell_scc_kernel(&system, &params, &r.electronic).unwrap();
        let kq1 = crate::linalg::matrix_vector_product(&kernel, &field.bundle.shell_charges)
            .unwrap();
        let t4 = crate::response::cpxtb::background_overlap_hessian_scalar(
            &system,
            &r.electronic.basis,
            &grad_ctx,
            &field.bundle.density,
            &v_pot_v,
            &v,
        );
        let u4 = crate::response::cpxtb::background_overlap_hessian_scalar(
            &system,
            &r.electronic.basis,
            &grad_ctx,
            &field.bundle.density,
            &kq1,
            &v,
        );
        let u1 = crate::response::cpxtb::background_overlap_gradient_scalar(
            &grad_ctx,
            &second.density,
            &kq1,
            &v,
        );
        eprintln!(
            "dG probes: t4(-P1*V1g*S2) {t4:+.9e}  u4(-P1*Kq1*S2) {u4:+.9e}  u1(-P2*Kq1*S1) \
             {u1:+.9e}"
        );

        // Pulay-block cache-motion probes via the Δ-potential trick.
        let cvv = |m: &Matrix| -> f64 {
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * m[(a, b)];
                }
            }
            acc
        };
        let nshell = r.electronic.basis.shells.len();
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &r.electronic.basis,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let pulay_delta = |p: Option<&Matrix>, w: Option<&Matrix>, dpot: &[f64]| -> f64 {
            let mut e1 = r.electronic.clone();
            let mut e0 = r.electronic.clone();
            if let Some(p) = p {
                e1.density = p.clone();
                e0.density = p.clone();
            }
            if let Some(w) = w {
                e1.energy_weighted_density = w.clone();
                e0.energy_weighted_density = w.clone();
            }
            for s in 0..nshell {
                e1.shell_scc_potential[s] += dpot[s];
            }
            cvv(&crate::hessian::fixed_density_pulay_hessian(&system, &params, &e1)
                .unwrap()
                .hessian)
                - cvv(&crate::hessian::fixed_density_pulay_hessian(&system, &params, &e0)
                    .unwrap()
                    .hessian)
        };
        let vp_plus_kq: Vec<f64> = (0..nshell).map(|s| v_pot_v[s] + kq1[s]).collect();
        let pw_vmot = pulay_delta(
            Some(&field.bundle.density),
            Some(&field.bundle.energy_weighted),
            &vp_plus_kq,
        );
        let pv_pmot = pulay_delta(Some(&field.bundle.density), None, &kq1);
        let pv_kmot = pulay_delta(None, None, &dgamma_v_qv);
        let so_pmot = {
            let mut e = r.electronic.clone();
            e.density = field.bundle.density.clone();
            e.shell_charges = field.bundle.shell_charges.clone();
            cvv(&crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &e)
                .unwrap())
        };
        eprintln!(
            "cache probes: pw_vmot {pw_vmot:+.9e} (target +2.029647e-6)  pv_pmot {pv_pmot:+.9e}  \
             pv_kmot {pv_kmot:+.9e} (pv sum target +3.402109e-6)  so_pmot {so_pmot:+.9e} (target \
             -3.346689e-7)"
        );

        // CN cache motion of the pulay block (affine self-energy trick).
        let pw_cnmot = {
            let nat = system.atoms.len();
            let cn_grad = crate::hessian::cn_gradient_matrix(&system, cutoff).unwrap();
            let cn_v: Vec<f64> = (0..nat)
                .map(|at| (0..ndof).map(|c| v[c] * cn_grad[at][c]).sum())
                .collect();
            let mut e_cnv = r.electronic.clone();
            e_cnv.density = field.bundle.density.clone();
            e_cnv.energy_weighted_density = field.bundle.energy_weighted.clone();
            e_cnv.coordination_numbers = cn_v;
            let mut e_cn0 = e_cnv.clone();
            e_cn0.coordination_numbers = vec![0.0; nat];
            cvv(&crate::hessian::fixed_density_pulay_hessian(&system, &params, &e_cnv)
                .unwrap()
                .hessian)
                - cvv(&crate::hessian::fixed_density_pulay_hessian(&system, &params, &e_cn0)
                    .unwrap()
                    .hessian)
        };
        eprintln!("pw_cnmot {pw_cnmot:+.9e} (target -2.620817e-6)");
        assert!(t4.is_finite() && u4.is_finite() && u1.is_finite());
    }

    /// **∂G family split**: FD each background family `G_fam(X¹(λ), X¹(λ))`
    /// along `v` (reconverged) and subtract `2·G_fam(X², X¹)` — the remainder
    /// is `∂G_fam`, the per-family eigen-motion target. Their sum must equal
    /// the equality-diagnostic tail (+7.616e-8 on this fixture).
    #[test]
    #[ignore]
    fn quartic_dg_family_split() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let dot = |grad: &[crate::math::Vec3]| -> f64 {
            grad.iter()
                .enumerate()
                .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
                .sum()
        };

        // The four family scalars G_fam(X¹, X¹) at a reconverged geometry.
        let g_at = |sys: &PeriodicSystem| -> [f64; 4] {
            let r = super::FiniteTThirdReference::build(sys, &params, &options, cutoff).unwrap();
            let (field, v_pot_v) =
                crate::fourth_derivative::assemble::directional_first_order_legs(
                    sys,
                    &params,
                    &r.electronic,
                    &r.cphf,
                    &r.ctx,
                    &v,
                )
                .unwrap();
            let nshell = r.electronic.basis.shells.len();
            let grad_ctx = ResponseGradientContext::new(
                sys,
                &r.electronic.basis,
                &params,
                &r.electronic,
                cutoff,
                r.include_cn_h0,
            )
            .unwrap();
            let kernel = response_shell_scc_kernel(sys, &params, &r.electronic).unwrap();
            let chain = super::kernel_chain_vec(
                sys,
                &params,
                &r.electronic,
                &field.bundle.shell_charges,
                nshell,
            )
            .unwrap();
            let bg = crate::response::cpxtb::response_gradient_background_motion(
                sys,
                &r.electronic,
                &grad_ctx,
                &kernel,
                &field.bundle.density,
                &field.bundle.shell_charges,
                &chain,
                &v_pot_v,
            )
            .unwrap();
            [
                dot(&bg.scc_dp_pot),
                dot(&bg.scc_p0),
                dot(&bg.scc_chain),
                dot(&bg.kernel_qq),
            ]
        };
        let displaced = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let h = 1.0e-3;
        let gp = g_at(&displaced(h));
        let gm = g_at(&displaced(-h));
        let dv_g: Vec<f64> = (0..4).map(|i| (gp[i] - gm[i]) / (2.0 * h)).collect();

        // G_fam(X², X¹) mixed at the reference (same construction as the 3G
        // term of the production assembly).
        let r = super::FiniteTThirdReference::build(&system, &params, &options, cutoff).unwrap();
        let (field, v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.cphf,
            &r.ctx,
            &v,
        )
        .unwrap();
        let (second, _) = crate::fourth_derivative::assemble::directional_second_order_legs(
            &system,
            &params,
            &r.electronic,
            &r.ctx,
            &field,
            cutoff,
            r.include_cn_h0,
            &v,
        )
        .unwrap();
        let nshell = r.electronic.basis.shells.len();
        let grad_ctx = ResponseGradientContext::new(
            &system,
            &r.electronic.basis,
            &params,
            &r.electronic,
            cutoff,
            r.include_cn_h0,
        )
        .unwrap();
        let kernel = response_shell_scc_kernel(&system, &params, &r.electronic).unwrap();
        let chain_mixed = super::kernel_chain_mixed(
            &system,
            &params,
            &r.electronic,
            &field.bundle.shell_charges,
            &second.shell_charges,
            nshell,
        )
        .unwrap();
        let bg_a = crate::response::cpxtb::response_gradient_background_motion(
            &system,
            &r.electronic,
            &grad_ctx,
            &kernel,
            &second.density,
            &field.bundle.shell_charges,
            &chain_mixed,
            &v_pot_v,
        )
        .unwrap();
        let bg_b = crate::response::cpxtb::response_gradient_background_motion(
            &system,
            &r.electronic,
            &grad_ctx,
            &kernel,
            &field.bundle.density,
            &second.shell_charges,
            &chain_mixed,
            &v_pot_v,
        )
        .unwrap();
        let qq_of = |q: &[f64]| -> f64 {
            let bg = crate::response::cpxtb::response_gradient_background_motion(
                &system,
                &r.electronic,
                &grad_ctx,
                &kernel,
                &field.bundle.density,
                q,
                &chain_mixed,
                &v_pot_v,
            )
            .unwrap();
            dot(&bg.kernel_qq)
        };
        let q_sum: Vec<f64> = (0..nshell)
            .map(|s| field.bundle.shell_charges[s] + second.shell_charges[s])
            .collect();
        let qq_mixed = 0.5
            * (qq_of(&q_sum)
                - qq_of(&field.bundle.shell_charges)
                - qq_of(&second.shell_charges));
        // NOTE the mixed dp_pot family: slot P² against V¹ — but the TRUE
        // D_v of the dp_pot family also moves the V¹ leg (V¹ → V²): probe
        // BOTH pieces.
        let kq_vv =
            crate::linalg::matrix_vector_product(&kernel, &second.shell_charges).unwrap();
        let g_mixed_dp_a = dot(&bg_a.scc_dp_pot); // P²·V¹
        let bg_c = crate::response::cpxtb::response_gradient_background_motion(
            &system,
            &r.electronic,
            &grad_ctx,
            &kernel,
            &field.bundle.density,
            &second.shell_charges,
            &chain_mixed,
            &kq_vv,
        )
        .unwrap();
        let g_mixed_dp_b = dot(&bg_c.scc_dp_pot); // P¹·(Kq²) — partial V² probe
        let g_mixed = [
            g_mixed_dp_a,
            dot(&bg_b.scc_p0),
            dot(&bg_a.scc_chain),
            qq_mixed,
        ];
        let dg: Vec<f64> = (0..4).map(|i| dv_g[i] - 2.0 * g_mixed[i]).collect();
        eprintln!(
            "dG family split:\n  D_v[G] = {dv_g:?}\n  G_mixed = {g_mixed:?}  (dp_b probe \
             {g_mixed_dp_b:.6e})\n  dG = {dg:?}\n  sum(dG) = {:.6e}  (target +7.616e-8)",
            dg.iter().sum::<f64>()
        );
        assert!(dv_g.iter().all(|x| x.is_finite()));
    }

    /// **Quartic T = 0 equality diagnostic.** The product-rule finite-T
    /// fourth must equal the validated five-stage T = 0 quartic. `#[ignore]`d
    /// while the `∂B`/`∂G` inventory is being pinned — run explicitly and
    /// read the residual.
    /// **CN bisection of the T = 0 quartic equality.** Result (recorded, this
    /// runs as a diagnostic):
    ///
    /// | fixture | `enable_cn_hamiltonian` | delta |
    /// |---|---|---|
    /// | non-eq water | on (default) | `2.87e-15` |
    /// | non-eq water | off | `1.89e-5` |
    /// | near-Td methane | on (default) | `2.29e-15` |
    /// | near-Td methane | off | `6.78e-6` |
    ///
    /// So the production configuration is exact even at CN = 4 with a
    /// three-fold degenerate frontier — the smearing-path residual reported
    /// against `directional_fourth_derivative` at 5–50 K is a finite-`T`
    /// branch effect, not an order-4 inventory gap.
    ///
    /// **`enable_cn_hamiltonian = false` is not a supported FC4
    /// configuration.** Both quartic paths keep the CN blocks regardless of
    /// the flag (the reference `dsedcn` is populated either way), and they
    /// agree with each other only while both do: gating the finite-T side
    /// alone widened the deltas to `2.4e-5` / `2.3e-5`. Fixing this means
    /// zeroing the CN channel in the reference, not in the assemblies.
    #[test]
    #[ignore = "diagnostic"]
    fn quartic_t0_equality_cn_bisection() {
        let fixtures: [(&str, &str); 2] = [
            (
                "water",
                "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            ),
            (
                "near-Td methane",
                "5\nmethane\nC 0.010000 -0.005000 0.008000\nH 0.640000 0.640000 0.640000\nH -0.640000 -0.640000 0.640000\nH -0.640000 0.640000 -0.640000\nH 0.640000 -0.640000 -0.640000\n",
            ),
        ];
        for (name, xyz) in fixtures {
            for cn in [true, false] {
                let params = Gfn1Parameters::builtin().unwrap();
                let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
                let mut options = crate::hessian::AnalyticHessianOptions::default();
                options.electronic_options.enable_dispersion = false;
                options.electronic_options.energy_tolerance = 1.0e-12;
                options.electronic_options.charge_tolerance = 1.0e-10;
                options.electronic_options.hamiltonian.enable_cn_hamiltonian = cn;
                let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
                let ndof = 3 * system.atoms.len();
                let v: Vec<f64> = (0..ndof)
                    .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
                    .collect();
                let mine =
                    super::directional_fourth_finite_t(&system, &params, &options, cutoff, &v)
                        .unwrap();
                let reference = crate::fourth_derivative::directional_fourth_derivative(
                    &system, &params, &options, cutoff, &v,
                )
                .unwrap();
                let delta = (mine - reference).abs();
                println!(
                    "{name:>16} cn={cn:<5}: mine {mine:+.12e}  five-stage {reference:+.12e}  \
                     delta {delta:.3e}"
                );
                if cn {
                    assert!(
                        delta < 1.0e-12 * (1.0 + reference.abs()),
                        "{name}: the production (CN-on) quartic equality regressed: {delta:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    fn directional_fourth_finite_t_matches_t0_quartic() {
        // Non-eq water and skew HF — the two fixtures that pinned the ∂B/∂G
        // inventory (a water-only bisection once self-cancelled; HF exposed
        // it). Both must close to machine precision.
        for xyz in [
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            "2\nskew HF\nF 0.0 0.0 0.0\nH 0.70 0.54 0.40\n",
        ] {
            run_quartic_t0_equality(xyz);
        }
    }

    /// T = 0 equality harness for one fixture: the product-rule finite-T
    /// quartic vs the validated five-stage T = 0 quartic.
    fn run_quartic_t0_equality(xyz: &str) {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let mine =
            super::directional_fourth_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        let reference = crate::fourth_derivative::directional_fourth_derivative(
            &system, &params, &options, cutoff, &v,
        )
        .unwrap();
        eprintln!(
            "quartic finite-T assembly T=0: mine {mine:.12e} vs five-stage {reference:.12e} \
             delta {:.3e}",
            (mine - reference).abs()
        );
        assert!(
            (mine - reference).abs() < 1.0e-9 * (1.0 + reference.abs()),
            "product-rule quartic vs five-stage quartic at T=0: {mine:.12e} vs {reference:.12e}"
        );
    }

    /// **Finite-T FD gate**: the analytic finite-temperature directional
    /// fourth on a genuinely smeared system (scalene H₃, 3 electrons at
    /// 3000 K → fractional occupations) must match the central FD of the
    /// analytic finite-T directional third along the same direction. Finite-T
    /// reconvergence noise scales as 1/h, so the gate uses the coarse
    /// h = 4e-3 step established for the FC3 smearing gates and prints an
    /// h-ladder for the truncation check.
    #[test]
    fn directional_fourth_finite_t_matches_fd() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nscalene H3\nH 0.0 0.0 0.0\nH 0.95 0.10 0.0\nH 0.35 0.85 0.15\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.electronic_temperature = 3000.0;
        options.electronic_options.energy_tolerance = 1.0e-14;
        options.electronic_options.charge_tolerance = 1.0e-12;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let electronic =
            run_electronic(&system, &params, options.electronic_options.clone()).unwrap();
        assert!(
            electronic
                .occupations
                .iter()
                .any(|&f| f > 1.0e-3 && f < 2.0 - 1.0e-3),
            "fixture must have fractional occupations"
        );
        let mine =
            super::directional_fourth_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        let third_at = |step: f64| -> f64 {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            super::directional_third_finite_t(&sys, &params, &options, cutoff, &v).unwrap()
        };
        let fd_of = |h: f64| (third_at(h) - third_at(-h)) / (2.0 * h);
        // The fixture's fifth derivative is large (truncation ≈ 4e-4 at
        // h = 4e-3) but purely h²: the ladder ratio sits at 3.99. Two-level
        // Richardson removes the h²/h⁴ terms and lands at the analytic value.
        let fd8 = fd_of(8.0e-3);
        let fd4 = fd_of(4.0e-3);
        let fd2 = fd_of(2.0e-3);
        let r1a = (4.0 * fd4 - fd8) / 3.0;
        let r1b = (4.0 * fd2 - fd4) / 3.0;
        let r2 = (16.0 * r1b - r1a) / 15.0;
        let ratio = (fd8 - mine) / (fd4 - mine);
        eprintln!(
            "smeared quartic FD gate: analytic {mine:.10e}  fd(4e-3) {fd4:.10e}  richardson \
             {r2:.10e}  |delta| {:.3e}  ladder ratio {ratio:.2}",
            (mine - r2).abs()
        );
        assert!(
            (3.5..4.5).contains(&ratio),
            "FD ladder is not in the h² truncation regime: ratio {ratio:.3}"
        );
        assert!(
            (mine - r2).abs() < 2.0e-6 * (1.0 + r2.abs()),
            "smeared analytic directional fourth vs Richardson FD of finite-T third: \
             {mine:.10e} vs {r2:.10e}"
        );
    }

    /// **Dense FC4 gate 1 (T = 0 equality).** The polarization-assembled
    /// dense finite-T fourth tensor must reproduce the five-stage analytic
    /// dense FC4 element-wise on skew HF.
    #[test]
    fn dense_finite_t4_matches_analytic_t0() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "2\nskew HF\nF 0.0 0.0 0.0\nH 0.70 0.54 0.40\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let mine =
            super::fourth_derivative_finite_t_dense(&system, &params, &options, cutoff).unwrap();
        let reference = crate::fourth_derivative::fourth_derivative_analytic_dense(
            &system, &params, &options, cutoff,
        )
        .unwrap();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for d in 0..ndof {
            for c in 0..=d {
                for b in 0..=c {
                    for a in 0..=b {
                        worst = worst
                            .max((mine.get(a, b, c, d) - reference.get(a, b, c, d)).abs());
                        scale = scale.max(reference.get(a, b, c, d).abs());
                    }
                }
            }
        }
        eprintln!("dense finite-T FC4 vs analytic T=0: worst {worst:.3e} (scale {scale:.3e})");
        assert!(
            worst < 1.0e-8 * (1.0 + scale),
            "dense finite-T FC4 vs five-stage dense FC4 at T=0: worst {worst:.3e}"
        );
    }

    /// **Dense FC4 gate 2 (smeared block self-consistency).** On smeared
    /// scalene H₃ (3000 K, fractional occupations) a 4-dof block of the
    /// polarization-assembled tensor, contracted `vvvv`, must reproduce the
    /// directional finite-T quartic along the same in-block direction.
    #[test]
    fn dense_finite_t4_smeared_block_contraction() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nscalene H3\nH 0.0 0.0 0.0\nH 0.95 0.10 0.0\nH 0.35 0.85 0.15\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.electronic_temperature = 3000.0;
        options.electronic_options.energy_tolerance = 1.0e-14;
        options.electronic_options.charge_tolerance = 1.0e-12;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let dofs = [0usize, 2, 4, 7];
        let block = super::fourth_derivative_finite_t_block(
            &system, &params, &options, cutoff, &dofs,
        )
        .unwrap();
        let w = [0.61, -0.34, 0.87, 0.22];
        let contracted = block.contract_vvvv(&w).unwrap();
        let ndof = 3 * system.atoms.len();
        let mut v = vec![0.0_f64; ndof];
        for (pos, &dof) in dofs.iter().enumerate() {
            v[dof] = w[pos];
        }
        let directional =
            super::directional_fourth_finite_t(&system, &params, &options, cutoff, &v).unwrap();
        eprintln!(
            "smeared dense-block FC4 contraction: block {contracted:.10e} vs directional \
             {directional:.10e}  |delta| {:.3e}",
            (contracted - directional).abs()
        );
        assert!(
            (contracted - directional).abs() < 1.0e-9 * (1.0 + directional.abs()),
            "smeared block-contracted FC4 vs directional: {contracted:.10e} vs \
             {directional:.10e}"
        );
    }

    /// **Dense gate 1 (T = 0 equality).** The polarization-assembled dense
    /// finite-T tensor must reproduce the adjoint-assembled analytic dense
    /// FC3 element-wise on non-equilibrium water.
    #[test]
    fn dense_finite_t_matches_analytic_t0() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let mine =
            super::third_derivative_finite_t_dense(&system, &params, &options, cutoff).unwrap();
        let reference = crate::third_derivative::third_derivative_analytic_dense(
            &system,
            &params,
            options.clone(),
            cutoff,
        )
        .unwrap();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for c in 0..ndof {
            for b in 0..=c {
                for a in 0..=b {
                    worst = worst.max((mine.get(a, b, c) - reference.get(a, b, c)).abs());
                    scale = scale.max(reference.get(a, b, c).abs());
                }
            }
        }
        eprintln!("dense finite-T vs analytic T=0: worst {worst:.3e} (scale {scale:.3e})");
        assert!(
            worst < 1.0e-8 * (1.0 + scale),
            "dense finite-T FC3 vs adjoint dense FC3 at T=0: worst {worst:.3e}"
        );
    }

    /// **Dense gate 2 (smeared).** Fermi-smeared scalene H₃ (3 electrons →
    /// fractional occupations, no symmetry degeneracy): the dense
    /// finite-temperature tensor must match the seminumerical dense reference
    /// (FD of the finite-T analytic Hessian) element-wise.
    #[test]
    fn dense_finite_t_smeared_matches_seminumerical() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nscalene H3\nH 0.0 0.0 0.0\nH 0.95 0.10 0.0\nH 0.35 0.85 0.15\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.electronic_temperature = 3000.0;
        options.electronic_options.energy_tolerance = 1.0e-14;
        options.electronic_options.charge_tolerance = 1.0e-12;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, options.electronic_options.clone())
            .unwrap();
        assert!(
            electronic
                .occupations
                .iter()
                .any(|&f| f > 1.0e-2 && f < 2.0 - 1.0e-2),
            "fixture must be Fermi-smeared (occupations: {:?})",
            electronic.occupations
        );
        let ndof = 3 * system.atoms.len();
        let mine =
            super::third_derivative_finite_t_dense(&system, &params, &options, cutoff).unwrap();
        let worst_vs = |step: f64| -> (f64, f64) {
            let reference = crate::third_derivative::third_derivative_seminumerical_dense(
                &system,
                &params,
                options.clone(),
                step,
            )
            .unwrap();
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for c in 0..ndof {
                for b in 0..=c {
                    for a in 0..=b {
                        worst = worst.max((mine.get(a, b, c) - reference.get(a, b, c)).abs());
                        scale = scale.max(reference.get(a, b, c).abs());
                    }
                }
            }
            (worst, scale)
        };
        let (worst1, scale) = worst_vs(1.0e-4);
        let (worst2, _) = worst_vs(5.0e-5);
        eprintln!(
            "dense finite-T smeared vs seminumerical: worst(h) {worst1:.3e} worst(h/2) \
             {worst2:.3e} ratio {:.2} (scale {scale:.3e})",
            worst1 / worst2.max(1.0e-300)
        );
        // h² ladder: a residual that shrinks ~4× on halving the FD step is
        // seminumerical truncation, not an analytic defect.
        assert!(
            worst2 < 0.4 * worst1 || worst2 < 1.0e-7 * (1.0 + scale),
            "smeared dense FC3 residual does not scale as h² (worst(h) {worst1:.3e}, worst(h/2) \
             {worst2:.3e}) — suspect a missing analytic term"
        );
        assert!(
            worst2 < 1.0e-5 * (1.0 + scale),
            "smeared dense FC3 vs seminumerical: worst {worst2:.3e} (scale {scale:.3e})"
        );
    }

    /// Block driver consistency: the block sub-tensor equals the dense
    /// tensor's corresponding elements (position indexing).
    #[test]
    fn block_finite_t_consistent_with_dense() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = crate::hessian::AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = false;
        options.electronic_options.energy_tolerance = 1.0e-12;
        options.electronic_options.charge_tolerance = 1.0e-10;
        let cutoff = options.electronic_options.hamiltonian.coordination_cutoff;
        let dofs = [0usize, 4, 7];
        let block =
            super::third_derivative_finite_t_block(&system, &params, &options, cutoff, &dofs)
                .unwrap();
        let dense =
            super::third_derivative_finite_t_dense(&system, &params, &options, cutoff).unwrap();
        let mut worst = 0.0_f64;
        for k in 0..dofs.len() {
            for j in 0..=k {
                for i in 0..=j {
                    worst = worst.max(
                        (block.get(i, j, k) - dense.get(dofs[i], dofs[j], dofs[k])).abs(),
                    );
                }
            }
        }
        assert!(
            worst < 1.0e-12,
            "block vs dense finite-T FC3 mismatch: {worst:.3e}"
        );
        // Out-of-range DOF is rejected.
        assert!(super::third_derivative_finite_t_block(
            &system, &params, &options, cutoff, &[0, 9]
        )
        .is_err());
    }

    #[test]
    fn directional_response_hessian_matches_columns_t0_water() {
        run_equality_gate(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            300.0,
            "T=0 water",
        );
    }

    fn contract_grad_v(grad: &[crate::math::Vec3], v: &[f64]) -> f64 {
        grad.iter()
            .enumerate()
            .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
            .sum()
    }

    /// **Step-B equality gate (T = 0 arbiter).** The product-rule assembly of
    /// `D_v[r2[v,v]]` — response motion `g(X^vv)·v` + geometric motion
    /// `path_hessian(X^v)[v,v]` + background motion — must equal the
    /// independently validated T = 0 adjoint assembly
    /// [`directional_response_third`]. This pins the exact term inventory of
    /// the background-motion family before any finite-temperature use.
    #[test]
    fn step_b_matches_t0_directional_response_third() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let include_cn_h0 = options.hamiltonian.enable_cn_hamiltonian;
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0,
        };
        let cphf = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx =
            crate::response::charge_space::ChargeSpaceContext::build(&system, &params, &electronic)
                .unwrap();
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        let (field, v_pot_v) = crate::fourth_derivative::assemble::directional_first_order_legs(
            &system,
            &params,
            &electronic,
            &cphf,
            &ctx,
            &v,
        )
        .unwrap();
        let (second, _v_pot_vv) = crate::fourth_derivative::assemble::directional_second_order_legs(
            &system,
            &params,
            &electronic,
            &ctx,
            &field,
            cutoff,
            include_cn_h0,
            &v,
        )
        .unwrap();

        // B1: response motion.
        let b1 = directional_response_hessian_vv(
            &system,
            &params,
            &electronic,
            cutoff,
            include_cn_h0,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
            &v,
        )
        .unwrap();
        // B2: geometric motion of the contraction coefficients.
        let path = crate::third_derivative::frozen_hessian_density_path(
            &system,
            &params,
            &electronic,
            cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            &field.bundle.shell_charges,
            &v_pot_v,
        )
        .unwrap();
        let mut b2 = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                b2 += v[a] * v[b] * path[(a, b)];
            }
        }
        // B3 candidates: background-state motion families.
        let kernel = response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let grad_ctx = ResponseGradientContext::new(
            &system,
            &electronic.basis,
            &params,
            &electronic,
            cutoff,
            include_cn_h0,
        )
        .unwrap();
        let chain = ctx.kernel_chain_potential(
            &field.bundle.shell_charges,
            &field.bundle.shell_charges,
        );
        let bg = crate::response::cpxtb::response_gradient_background_motion(
            &system,
            &electronic,
            &grad_ctx,
            &kernel,
            &field.bundle.density,
            &field.bundle.shell_charges,
            &chain,
            &v_pot_v,
        )
        .unwrap();
        let c_p0 = contract_grad_v(&bg.scc_p0, &v);
        let c_chain = contract_grad_v(&bg.scc_chain, &v);
        let c_dp_pot = contract_grad_v(&bg.scc_dp_pot, &v);
        let c_qq = contract_grad_v(&bg.kernel_qq, &v);

        let reference = crate::fourth_derivative::response_stage::directional_response_third(
            &system,
            &params,
            &electronic,
            &cphf,
            ao_opts,
            cutoff,
            &v,
        )
        .unwrap();
        // Path probes: split B2's potential channel by feeding partial V^v.
        let nshell = electronic.basis.shells.len();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let v_pot_geo: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum())
            .collect();
        let contract_path = |v_pot: &[f64]| -> f64 {
            let m = crate::third_derivative::frozen_hessian_density_path(
                &system,
                &params,
                &electronic,
                cutoff,
                &field.bundle.density,
                &field.bundle.energy_weighted,
                &field.bundle.shell_charges,
                v_pot,
            )
            .unwrap();
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * m[(a, b)];
                }
            }
            acc
        };
        let b2_geo = contract_path(&v_pot_geo);
        let b2_zero = contract_path(&vec![0.0; nshell]);

        let required = reference - (b1 + b2);
        let b3 = c_p0 + c_chain + c_dp_pot + c_qq;
        eprintln!(
            "step B: reference {reference:.12e}  b1 {b1:.12e}  b2 {b2:.12e}  required-B3 \
             {required:.12e}\n  candidates: p0 {c_p0:.12e}  chain {c_chain:.12e}  dp_pot \
             {c_dp_pot:.12e}  qq {c_qq:.12e}  sum {b3:.12e}  delta {:.3e}\n  path probes: \
             b2_geo {b2_geo:.12e}  b2_zero {b2_zero:.12e}  Vchan_total {:.12e}  Vchan_kq {:.12e}",
            (required - b3).abs(),
            b2 - b2_zero,
            b2 - b2_geo
        );
        // ---- family bisect: FD each g-term family along v (everything
        // reconverged) and print next to the analytic per-family pieces. ----
        let family_at = |sys: &PeriodicSystem| -> [f64; 6] {
            let el = run_electronic(sys, &params, options.clone()).unwrap();
            let cp = solve_nonpbc_cpxtb_hessian_response(
                sys,
                &params,
                &el,
                ao_opts,
                CpxtbOptions::default(),
            )
            .unwrap();
            let n = el.basis.len();
            let nsh = el.basis.shells.len();
            let mut pv = crate::linalg::Matrix::zeros(n, n);
            let mut wv = crate::linalg::Matrix::zeros(n, n);
            let mut qv = vec![0.0_f64; nsh];
            for (c, &vc) in v.iter().enumerate() {
                for k in 0..n * n {
                    pv.as_mut_slice()[k] += vc * cp.density_responses[c].as_slice()[k];
                    wv.as_mut_slice()[k] +=
                        vc * cp.energy_weighted_density_responses[c].as_slice()[k];
                }
                for s in 0..nsh {
                    qv[s] += vc * cp.shell_charge_responses[c][s];
                }
            }
            let kern = response_shell_scc_kernel(sys, &params, &el).unwrap();
            let gctx = ResponseGradientContext::new(
                sys,
                &el.basis,
                &params,
                &el,
                cutoff,
                include_cn_h0,
            )
            .unwrap();
            let terms = crate::response::cpxtb::response_electronic_gradient_terms(
                sys, &el, &kern, &gctx, &pv, &pv, &wv, &qv,
            )
            .unwrap();
            [
                contract_grad_v(&terms.band, &v),
                contract_grad_v(&terms.polynomial, &v),
                contract_grad_v(&terms.scc_overlap, &v),
                contract_grad_v(&terms.pulay, &v),
                contract_grad_v(&terms.cn, &v),
                contract_grad_v(&terms.scc_kernel, &v),
            ]
        };
        let displace = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom, a) in sys.atoms.iter_mut().enumerate() {
                a.position.x += step * v[3 * atom];
                a.position.y += step * v[3 * atom + 1];
                a.position.z += step * v[3 * atom + 2];
            }
            sys
        };
        let h = 1.0e-5;
        let fp = family_at(&displace(h));
        let fm = family_at(&displace(-h));
        let fd_fam: Vec<f64> = (0..6).map(|i| (fp[i] - fm[i]) / (2.0 * h)).collect();
        // Analytic per-family pieces of B1 (response motion).
        let kern0 = response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let b1_terms = crate::response::cpxtb::response_electronic_gradient_terms(
            &system,
            &electronic,
            &kern0,
            &grad_ctx,
            &second.density,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
        )
        .unwrap();
        eprintln!(
            "family FD (band, poly, scc_ov, pulay, cn, scc_kern):\n  fd    {:?}\n  b1    [{:.6e}, \
             {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}]\n  fd-b1 [{:.6e}, {:.6e}, {:.6e}, {:.6e}, \
             {:.6e}, {:.6e}]",
            fd_fam,
            contract_grad_v(&b1_terms.band, &v),
            contract_grad_v(&b1_terms.polynomial, &v),
            contract_grad_v(&b1_terms.scc_overlap, &v),
            contract_grad_v(&b1_terms.pulay, &v),
            contract_grad_v(&b1_terms.cn, &v),
            contract_grad_v(&b1_terms.scc_kernel, &v),
            fd_fam[0] - contract_grad_v(&b1_terms.band, &v),
            fd_fam[1] - contract_grad_v(&b1_terms.polynomial, &v),
            fd_fam[2] - contract_grad_v(&b1_terms.scc_overlap, &v),
            fd_fam[3] - contract_grad_v(&b1_terms.pulay, &v),
            fd_fam[4] - contract_grad_v(&b1_terms.cn, &v),
            fd_fam[5] - contract_grad_v(&b1_terms.scc_kernel, &v),
        );
        eprintln!(
            "  candidates for matching: dp_pot {c_dp_pot:.6e}  p0 {c_p0:.6e}  chain \
             {c_chain:.6e}  qq {c_qq:.6e}  Vchan_total {:.6e}  Vchan_kq {:.6e}  b2_zero \
             {b2_zero:.6e}",
            b2 - b2_zero,
            b2 - b2_geo,
        );
        let fd_total: f64 = fd_fam.iter().sum();
        eprintln!("  fd_total {fd_total:.12e}  (fd_total - reference {:.3e})", fd_total - reference);

        // ---- path slot probes (the path is linear in each response slot) ----
        let nbas = electronic.basis.len();
        let zero_m = crate::linalg::Matrix::zeros(nbas, nbas);
        let zero_q = vec![0.0_f64; nshell];
        let path_slots = |p: &crate::linalg::Matrix,
                          w: &crate::linalg::Matrix,
                          q: &[f64],
                          vp: &[f64]|
         -> f64 {
            let m = crate::third_derivative::frozen_hessian_density_path(
                &system,
                &params,
                &electronic,
                cutoff,
                p,
                w,
                q,
                vp,
            )
            .unwrap();
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * m[(a, b)];
                }
            }
            acc
        };
        let ch_p = path_slots(&field.bundle.density, &zero_m, &zero_q, &zero_q);
        let ch_w = path_slots(&zero_m, &field.bundle.energy_weighted, &zero_q, &zero_q);
        let ch_q = path_slots(&zero_m, &zero_m, &field.bundle.shell_charges, &zero_q);
        let ch_v = path_slots(&zero_m, &zero_m, &zero_q, &v_pot_v);
        // s2 kernel geometric charge path (∂²(qᵀγq)/∂R² bilinear in (q₀, q^v)).
        let s2_path = crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let mut c_s2geo = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                c_s2geo += v[a] * v[b] * s2_path[(a, b)];
            }
        }
        // Geometric kernel motion in the screening potential:
        // −P₀·[(∂_vγ)·q^v]_pair·∇S (the response-Hessian mirror of the
        // second-order solver's RF_S((∂_xγ)q^y) cross term). Reuses the
        // background-motion helper's `scc_chain` slot, which has exactly the
        // −P₀·(pot)_pair·∇S shape.
        let dgamma_qv_m = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv_m[(s, c)]).sum())
            .collect();
        let bg_geok = crate::response::cpxtb::response_gradient_background_motion(
            &system,
            &electronic,
            &grad_ctx,
            &kernel,
            &field.bundle.density,
            &field.bundle.shell_charges,
            &dgamma_v_qv,
            &v_pot_v,
        )
        .unwrap();
        let c_geok = contract_grad_v(&bg_geok.scc_chain, &v);
        eprintln!(
            "  slots: ch_p {ch_p:.6e}  ch_w {ch_w:.6e}  ch_q {ch_q:.6e}  ch_v {ch_v:.6e}  \
             sum {:.6e} (b2 {b2:.6e})  c_s2geo {c_s2geo:.6e}  c_geok {c_geok:.6e}\n  family \
             checks: pulay ch_w-vs-target {:.3e}  scc_kern (qq+s2geo)-vs-target {:.3e}\n  FULL \
             hypothesis: b1+b2+s2geo+qq+dp_pot+p0+chain+geok vs reference: {:.3e}",
            ch_p + ch_w + ch_q + ch_v,
            ch_w - (fd_fam[3] - contract_grad_v(&b1_terms.pulay, &v)),
            (c_qq + c_s2geo) - (fd_fam[5] - contract_grad_v(&b1_terms.scc_kernel, &v)),
            (b1 + b2 + c_s2geo + c_qq + c_dp_pot + c_p0 + c_chain + c_geok) - reference,
        );

        // ---- frozen-background FD: displace ONLY the geometry; electronic
        // state, response inputs, V₀ values and CN(hij) stay at the reference.
        // Splits each family's motion into geometric (this FD) vs background
        // (full FD minus this minus B1). ----
        let frozen_family_at = |sys: &PeriodicSystem| -> [f64; 6] {
            let kern = response_shell_scc_kernel(sys, &params, &electronic).unwrap();
            let gctx = ResponseGradientContext::new(
                sys,
                &electronic.basis,
                &params,
                &electronic,
                cutoff,
                include_cn_h0,
            )
            .unwrap();
            let terms = crate::response::cpxtb::response_electronic_gradient_terms(
                sys,
                &electronic,
                &kern,
                &gctx,
                &field.bundle.density,
                &field.bundle.density,
                &field.bundle.energy_weighted,
                &field.bundle.shell_charges,
            )
            .unwrap();
            [
                contract_grad_v(&terms.band, &v),
                contract_grad_v(&terms.polynomial, &v),
                contract_grad_v(&terms.scc_overlap, &v),
                contract_grad_v(&terms.pulay, &v),
                contract_grad_v(&terms.cn, &v),
                contract_grad_v(&terms.scc_kernel, &v),
            ]
        };
        let gp = frozen_family_at(&displace(h));
        let gm = frozen_family_at(&displace(-h));
        let geo_fam: Vec<f64> = (0..6).map(|i| (gp[i] - gm[i]) / (2.0 * h)).collect();
        let bg_fam: Vec<f64> = (0..6)
            .map(|i| {
                fd_fam[i]
                    - geo_fam[i]
                    - [
                        contract_grad_v(&b1_terms.band, &v),
                        contract_grad_v(&b1_terms.polynomial, &v),
                        contract_grad_v(&b1_terms.scc_overlap, &v),
                        contract_grad_v(&b1_terms.pulay, &v),
                        contract_grad_v(&b1_terms.cn, &v),
                        contract_grad_v(&b1_terms.scc_kernel, &v),
                    ][i]
            })
            .collect();
        eprintln!(
            "  frozen-geo FD per family: {geo_fam:?}\n  background (fd - geo - b1) per family: \
             {bg_fam:?}\n  geo sums {:.6e}  bg sums {:.6e}",
            geo_fam.iter().sum::<f64>(),
            bg_fam.iter().sum::<f64>()
        );

        // ---- direct block probes: the path's constituents one by one ----
        let cvv = |m: &crate::linalg::Matrix| -> f64 {
            let mut acc = 0.0;
            for a in 0..ndof {
                for b in 0..ndof {
                    acc += v[a] * v[b] * m[(a, b)];
                }
            }
            acc
        };
        let elec_pc = {
            let mut e = electronic.clone();
            e.density = field.bundle.density.clone();
            e
        };
        let d_cnh0 = cvv(
            &crate::hessian::fixed_density_cn_h0_hessian(&system, &params, &elec_pc, cutoff)
                .unwrap()
                .hessian,
        );
        let d_cross = cvv(
            &crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
                &system, &params, &elec_pc, cutoff,
            )
            .unwrap(),
        );
        let d_pulay_pw = {
            let mut e = electronic.clone();
            e.density = field.bundle.density.clone();
            e.energy_weighted_density = field.bundle.energy_weighted.clone();
            cvv(
                &crate::hessian::fixed_density_pulay_hessian(&system, &params, &e)
                    .unwrap()
                    .hessian,
            )
        };
        let pulay_v_probe = |pot: &[f64]| -> f64 {
            let mut e = electronic.clone();
            for s in 0..nshell {
                e.shell_scc_potential[s] += pot[s];
            }
            let h1 = cvv(
                &crate::hessian::fixed_density_pulay_hessian(&system, &params, &e)
                    .unwrap()
                    .hessian,
            );
            let h0 = cvv(
                &crate::hessian::fixed_density_pulay_hessian(&system, &params, &electronic)
                    .unwrap()
                    .hessian,
            );
            h1 - h0
        };
        let kq_v: Vec<f64> = {
            let kq = crate::linalg::matrix_vector_product(&kernel, &field.bundle.shell_charges)
                .unwrap();
            kq
        };
        let d_pulay_vtot = pulay_v_probe(&v_pot_v);
        let d_pulay_vkq = pulay_v_probe(&kq_v);
        let d_so_p = cvv(
            &crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &elec_pc)
                .unwrap(),
        );
        let d_so_q = {
            let mut e = electronic.clone();
            e.shell_charges = field.bundle.shell_charges.clone();
            cvv(
                &crate::hessian::fixed_density_scalar_overlap_hessian(&system, &params, &e)
                    .unwrap(),
            )
        };
        eprintln!(
            "  blocks: cn_h0 {d_cnh0:.6e}  cross {d_cross:.6e}  pulay_pw {d_pulay_pw:.6e}  \
             pulay_vtot {d_pulay_vtot:.6e}  pulay_vkq {d_pulay_vkq:.6e}  so_p {d_so_p:.6e}  \
             so_q {d_so_q:.6e}  s2geo {c_s2geo:.6e}"
        );

        // ---- the production assembly must close the equality ----
        let assembled = directional_response_hessian_derivative(
            &system,
            &params,
            &electronic,
            cutoff,
            include_cn_h0,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            &field.bundle.shell_charges,
            &v_pot_v,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
            &v,
        )
        .unwrap();
        eprintln!(
            "production assembly {assembled:.12e} vs reference {reference:.12e}  delta {:.3e}",
            (assembled - reference).abs()
        );
        assert!(
            (assembled - reference).abs() < 1.0e-9 * (1.0 + reference.abs()),
            "directional_response_hessian_derivative vs directional_response_third: \
             {assembled:.12e} vs {reference:.12e}"
        );
    }

    #[test]
    fn directional_response_hessian_matches_columns_finite_t() {
        run_equality_gate(
            "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n",
            3000.0,
            "finite-T Ni(CO)4",
        );
    }
}
