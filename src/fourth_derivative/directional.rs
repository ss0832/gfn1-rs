// SPDX-License-Identifier: GPL-3.0-or-later
//! Directional quartic force constants: `e⁗[v] = Σ_abcd Q_abcd v_a v_b v_c v_d`
//! along a fixed Cartesian direction `v` — the memory-lean flagship mode of the
//! analytic fourth derivative (by the 2n+1 rule it needs only the first- and
//! second-order responses along `v`).
//!
//! Assembly strategy: the analytic directional fourth derivative is the exact
//! `λ`-derivative of the (validated) directional third derivative along
//! `R + λv`, computed ingredient by ingredient: frozen third → frozen fourth
//! (Phase 4 blocks), first-order bundles → second-order bundles (Phase 5
//! charge-space solves), reference-state objects → their first-order responses.
//! Every sub-step is FD-gated against the ingredient one order down.
//!
//! This file builds the ladder bottom-up, one FD-gated stage per ingredient of
//! the third derivative it differentiates:
//!   1. [`directional_fourth_geometric_with`] — pure geometry (repulsion,
//!      halogen, frozen SCC2, D3): `L_λλλλ`, no electronic response at all;
//!   2. [`directional_fourth_frozen_density`] — the frozen-density Hamiltonian
//!      blocks: Phase-4d fourth blocks + the density path of the third blocks;
//!   3. [`directional_fourth_hessian_path_stage`] — the FC3 composition's
//!      density-path term `path(X^v)·vv`, whose `λ`-derivative first needs the
//!      SECOND-order bundle `X^vv` from the charge-space solver;
//!   4. [`directional_fourth_cn_response_stage`] — the FC3 composition's Pulay
//!      CN-response term `cn_resp(CN^v)·vv`, whose `λ`-derivative needs the
//!      second directional coordination response `CN^vv`.

use crate::error::Result;
use crate::hessian::AnalyticHessianOptions;
use crate::linalg::Matrix;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

/// Contract dense third-derivative slabs `slab[c][(a,b)]` with `v` three times.
pub(crate) fn contract_slabs_vvv(slabs: &[Matrix], v: &[f64]) -> f64 {
    let ndof = v.len();
    let mut acc = 0.0;
    for (c, slab) in slabs.iter().enumerate() {
        let vc = v[c];
        if vc == 0.0 {
            continue;
        }
        for a in 0..ndof {
            for b in 0..ndof {
                acc += vc * v[a] * v[b] * slab[(a, b)];
            }
        }
    }
    acc
}

/// The FROZEN (pure-geometry, density/charges/potential held at the reference)
/// part of the directional fourth derivative:
/// repulsion + halogen + frozen SCC2 + D3 (two-body + ATM when active),
/// i.e. the `L_λλλλ` blocks that carry no electronic-response input at all.
///
/// The frozen-density Hamiltonian blocks (Pulay / CN-H0 / scalar-overlap) are
/// intentionally NOT here — they enter the assembly together with their
/// density-path companions, mirroring the third-derivative composition.
///
/// **No system-size cap.** The D3 and halogen legs go through their DIRECTIONAL
/// entry points ([`crate::dispersion::dispersion_fourth_directional`],
/// [`crate::halogen::halogen_fourth_directional`]), which carry the univariate
/// Taylor of `E(R + t·v)` in a `Jet1` instead of a full-space `Jet4`. That is
/// `O(1)` storage per jet rather than `O(ndof⁴)`, so the
/// [`crate::MAX_FOURTH_DERIVATIVE_NDOF`] guard the full-tensor builders raise
/// (30 DOF / 10 atoms) does not apply to the directional quartic at all.
pub fn directional_fourth_geometric_with(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    options: &AnalyticHessianOptions,
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "directional_fourth_geometric_with: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    let mut acc = 0.0;
    if options.include_repulsion {
        let _p = crate::profile::scope("fc4.stage1.repulsion");
        let rep = crate::repulsion::repulsion_fourth_derivative(system, params)?;
        acc += rep.contract_vvvv(v)?;
    }
    if options.include_halogen {
        let _p = crate::profile::scope("fc4.stage1.halogen");
        acc += crate::halogen::halogen_fourth_directional(system, params, v)?;
    }
    if options.include_fixed_scc {
        let _p = crate::profile::scope("fc4.stage1.fixed_scc");
        let scc = crate::hessian::fixed_shell_charge_scc_fourth_derivative(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            params,
        )?;
        acc += scc.contract_vvvv(v)?;
    }
    let include_disp =
        options.include_dispersion && options.electronic_options.enable_dispersion;
    if include_disp {
        let _p = crate::profile::scope("fc4.stage1.dispersion");
        acc += crate::dispersion::dispersion_fourth_directional(
            system,
            params,
            options.electronic_options.d3_reference_path.as_deref(),
            v,
        )?;
    }
    Ok(acc)
}

/// The frozen-density Hamiltonian stage of the directional fourth derivative:
/// the Phase-4d fourth blocks (Pulay + CN-H0 + scalar-overlap) contracted
/// `vvvv`, PLUS the density path of the corresponding THIRD blocks along the
/// screened first-order bundle `(P^v, W^v, q^v, V^v_total)` contracted `vvv` —
/// together the exact `λ`-derivative of the frozen-density third-derivative
/// stage when the electronic reference moves with the geometry.
///
/// The density paths exploit the blocks' multilinearity via doctored
/// references, mirroring `frozen_hessian_density_path` one order up:
/// Pulay = linear(P, W) + bilinear(P·V); CN-H0 = linear(P);
/// scalar-overlap = bilinear(P, q).
#[allow(clippy::too_many_arguments)]
pub fn directional_fourth_frozen_density(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    v: &[f64],
) -> Result<f64> {
    directional_fourth_frozen_density_with(
        system,
        params,
        electronic,
        coordination_cutoff,
        p_v,
        w_v,
        q_v,
        v_pot_v,
        v,
        None,
    )
}

/// [`directional_fourth_frozen_density`] with the DIRECTION-INDEPENDENT undoctored Pulay third
/// derivative supplied from [`super::assemble::QuarticReference`] instead of rebuilt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_fourth_frozen_density_with(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    v: &[f64],
    pulay_third_reference: Option<&[Matrix]>,
) -> Result<f64> {
    let mut acc = 0.0;

    // Frozen fourth blocks · vvvv, built ONE-PASS: the per-AO-pair fourth-derivative data is
    // index-independent, so all four legs are contracted inside the pair sweep instead of
    // materialising the `out[c][d][(a,b)]` store (`ndof⁴` doubles) and contracting it afterwards.
    // Gated scalar-wise against exactly that nested path by `directional_fourth_tests`.
    {
        let _p = crate::profile::scope("fc4.stage2.pulay_fourth");
        acc += crate::hessian::directional_fixed_density_pulay_fourth(
            system, params, electronic, v,
        )?;
    }
    {
        let _p = crate::profile::scope("fc4.stage2.cn_h0_fourth");
        acc += crate::hessian::directional_fixed_density_cn_h0_fourth(
            system,
            params,
            electronic,
            coordination_cutoff,
            v,
        )?;
    }
    {
        let _p = crate::profile::scope("fc4.stage2.scalar_overlap_fourth");
        acc += crate::hessian::directional_fixed_density_scalar_overlap_fourth(
            system, params, electronic, v,
        )?;
    }

    // Density path of the third blocks · vvv, including the CN-cache response
    // (the reconverged reference moves `electronic.coordination_numbers`).
    let _p = crate::profile::scope("fc4.stage2.third_density_path");
    let cn_response = cn_first_response(system, coordination_cutoff, v)?;
    third_density_path_vvv(
        system,
        params,
        electronic,
        coordination_cutoff,
        p_v,
        w_v,
        q_v,
        v_pot_v,
        Some(&cn_response),
        v,
        pulay_third_reference,
        &mut acc,
    )?;
    Ok(acc)
}

/// The density path of the frozen-density **third** blocks along a bundle
/// `(P, W, q, V)`, contracted `vvv` and accumulated into `acc`:
/// `Σ_c v_c v_a v_b ∂(third_block[c][(a,b)])/∂(density fields) · bundle`.
///
/// Two callers, two meanings — they differ ONLY in `cn_response`:
///   * [`directional_fourth_frozen_density`] (stage 2) passes `Some(CN^v)`: it is taking the total
///     `λ`-derivative of the third blocks with the electronic reference RECONVERGED, so the cached
///     coordination number the Pulay third block reads moves too
///     ([`crate::hessian::fixed_density_pulay_third_cn_response`] = `∂(pulay_third)/∂CN`).
///   * [`directional_fourth_hessian_path_stage`] (stage 3) passes `None`: there this helper supplies
///     the purely GEOMETRIC motion `∂_R(Hessian blocks)·v` of the Hessian-level density path at
///     FIXED (doctored) references — and the Hessian block's cached-CN motion is a *separate*
///     product-rule term one order down ([`crate::hessian::fixed_density_pulay_cn_h0_response`]
///     evaluated with the doctored density), so including `∂(pulay_third)/∂CN` here would
///     double-differentiate.
///
/// Blocks and their multilinear structure (mirrors `frozen_hessian_density_path` one order up):
/// Pulay = linear(P, W) + bilinear(P·V); CN-H0 (+ its Pulay cross) = linear(P);
/// scalar-overlap = bilinear(P, q). `acc` is accumulated in place so the stage-2 arithmetic order
/// is preserved bit-for-bit across the refactor.
#[allow(clippy::too_many_arguments)]
fn third_density_path_vvv(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    cn_response: Option<&[f64]>,
    v: &[f64],
    pulay_third_reference: Option<&[Matrix]>,
    acc: &mut f64,
) -> Result<()> {
    let nshell = electronic.shell_charges.len();
    // Pulay third: linear (P, W) + bilinear (P·V).
    {
        let mut e1 = electronic.clone();
        e1.density = p_v.clone();
        e1.energy_weighted_density = w_v.clone();
        let t1 = crate::hessian::fixed_density_pulay_third_derivative(system, params, &e1)?;
        *acc += contract_slabs_vvv(&t1, v);
        let mut e2 = electronic.clone();
        for s in 0..nshell {
            e2.shell_scc_potential[s] += v_pot_v[s];
        }
        let t2 = crate::hessian::fixed_density_pulay_third_derivative(system, params, &e2)?;
        // `t0` is the UNDOCTORED block: it depends on the geometry and the converged reference
        // only, so the caller can hand in the copy `QuarticReference` built once for every
        // direction instead of paying for it twice per direction (stages 2 and 3 both land here).
        let owned_t0;
        let t0: &[Matrix] = match pulay_third_reference {
            Some(cached) => cached,
            None => {
                owned_t0 =
                    crate::hessian::fixed_density_pulay_third_derivative(system, params, electronic)?;
                &owned_t0
            }
        };
        *acc += contract_slabs_vvv(&t2, v) - contract_slabs_vvv(t0, v);
        // CN response: the Pulay third block reads a CACHED coordination number for its H0
        // self-energy prefactor, so a reconverged reference differentiates it while the frozen
        // fourth block does not. Add ∂(pulay_third)/∂CN · CN^v with CN^v_A = Σ_c v_c ∂CN_A/∂R_c.
        if let Some(cn_response) = cn_response {
            let t_cn = crate::hessian::fixed_density_pulay_third_cn_response(
                system,
                params,
                electronic,
                coordination_cutoff,
                cn_response,
            )?;
            *acc += contract_slabs_vvv(&t_cn, v);
        }
    }
    // CN-H0 third: linear in P. (Covers BOTH the CN-H0 Hessian and its Pulay cross block — the
    // third-derivative routine is the geometric derivative of their sum.)
    {
        let mut e1 = electronic.clone();
        e1.density = p_v.clone();
        let t = crate::hessian::fixed_density_cn_h0_third_derivative(
            system,
            params,
            &e1,
            coordination_cutoff,
        )?;
        *acc += contract_slabs_vvv(&t, v);
    }
    // Scalar-overlap third: bilinear (P, q).
    {
        let mut e1 = electronic.clone();
        e1.density = p_v.clone();
        let t1 = crate::hessian::fixed_density_scalar_overlap_third_derivative(
            system, params, &e1,
        )?;
        *acc += contract_slabs_vvv(&t1, v);
        let mut e2 = electronic.clone();
        e2.shell_charges = q_v.to_vec();
        let t2 = crate::hessian::fixed_density_scalar_overlap_third_derivative(
            system, params, &e2,
        )?;
        *acc += contract_slabs_vvv(&t2, v);
    }
    Ok(())
}

/// Contract a Hessian-shaped matrix with `v` twice: `Σ_ab v_a v_b m[(a,b)]`.
fn contract_vv(m: &Matrix, v: &[f64]) -> f64 {
    let ndof = v.len();
    let mut acc = 0.0;
    for a in 0..ndof {
        let va = v[a];
        if va == 0.0 {
            continue;
        }
        for b in 0..ndof {
            acc += va * v[b] * m[(a, b)];
        }
    }
    acc
}

/// **Stage 3** of the directional quartic assembly: the total `λ`-derivative of the FC3
/// composition's density-path term,
/// `d/dλ [ frozen_hessian_density_path(R+λv, el(λ), X^v(λ)) · vv ]`,
/// where `X^v = (P^v, W^v, q^v, V^v_total)` is the screened directional FIRST-order bundle and
/// `X^vv = (P^vv, W^vv, q^vv, V^vv_total)` the SECOND-order one (both supplied by the caller — the
/// charge-space solver builds them, see the gate test).
///
/// The path matrix `H(R, E, X)` depends on the geometry `R`, the electronic reference
/// `E = (P_ref, W_ref, q_ref, V_ref, CN)` and — linearly — on the bundle `X`. So the product rule
/// has exactly three groups (`d/dλ = ∂_X H·X^vv + ∂_R H·v + ∂_E H·E^v`), with `E^v = X^v` plus the
/// coordination-number response `CN^v`:
///
/// **(a) bundle motion** — `H` is linear in every bundle slot, so this is the SAME
/// `frozen_hessian_density_path` fed the second-order bundle: `path(X^vv)·vv`.
///
/// **(b) geometric motion at frozen references** — each Hessian block's `∂_R` is its third-level
/// sibling evaluated with the same doctored references ([`third_density_path_vvv`] with
/// `cn_response = None`), plus the `s2` block's third-level bilinear charge path
/// ([`crate::hessian::fixed_shell_charge_scc_third_charge_path`]). The `s2` block appears ONLY here
/// and in (c): stage 1 carries frozen-charge SCC2 and stage 2 carries no `s2` at all, because the
/// SCC2 charge path first enters the composition through this Hessian-path term.
///
/// **(c) doctored-reference motion** — the blocks also read UNDOCTORED reference slots that move
/// with `λ`. Because each such block is bilinear (one doctored slot × one reference slot) and the
/// path already contains BOTH orderings, the two cross terms coincide and the multiplicity is 2:
///   * `2 × [pulay_hess(P^v, W^v, V+V^v) − pulay_hess(P^v, W^v, V)]` — the `(P^v, V^v)` bilinear,
///     from `V_ref → V^v` in `pulay(P^v,W^v,V_ref)` and from `P_ref → P^v` in the V-shifted term;
///   * `2 × scalar_overlap_hess(P^v, q^v)` — from `q_ref → q^v` and `P_ref → P^v`;
///   * `1 × s2_charge_path(q^v, q^v)` — the `q_ref → q^v` motion of the bilinear `s2` charge path
///     (calling the helper with both arguments `q^v` already yields the `2 q^v_i q^v_j` weight);
///   * `1 × pulay_cn_h0_response(P^v, W^v; CN^v)` — the Pulay Hessian's `h0` prefactor reads the
///     CACHED coordination number, which moves with the reconverged reference. There is exactly
///     ONE such term: the V-shifted Pulay difference is `−P·V`-only, which carries no CN (and `−2W`
///     none either), so its CN response cancels identically; the CN-H0/cross/scalar-overlap/s2
///     blocks recompute or never touch CN.
///
/// **(d)** repulsion / halogen / D3 have no density dependence — their path is identically zero.
#[allow(clippy::too_many_arguments)]
pub fn directional_fourth_hessian_path_stage(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    p_vv: &Matrix,
    w_vv: &Matrix,
    q_vv: &[f64],
    v_pot_vv: &[f64],
    v: &[f64],
) -> Result<f64> {
    directional_fourth_hessian_path_stage_with(
        system,
        params,
        electronic,
        coordination_cutoff,
        p_v,
        w_v,
        q_v,
        v_pot_v,
        p_vv,
        w_vv,
        q_vv,
        v_pot_vv,
        v,
        None,
    )
}

/// [`directional_fourth_hessian_path_stage`] with the DIRECTION-INDEPENDENT undoctored Pulay
/// third derivative supplied from [`super::assemble::QuarticReference`] instead of rebuilt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_fourth_hessian_path_stage_with(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    q_v: &[f64],
    v_pot_v: &[f64],
    p_vv: &Matrix,
    w_vv: &Matrix,
    q_vv: &[f64],
    v_pot_vv: &[f64],
    v: &[f64],
    pulay_third_reference: Option<&[Matrix]>,
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "directional_fourth_hessian_path_stage: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    let nshell = electronic.shell_charges.len();
    let mut acc = 0.0;

    // ---- (a) bundle motion: the same path fed the SECOND-order bundle ----
    {
        let _p = crate::profile::scope("fc4.stage3.bundle_path");
        let path_second = crate::third_derivative::frozen_hessian_density_path(
            system,
            params,
            electronic,
            coordination_cutoff,
            p_vv,
            w_vv,
            q_vv,
            v_pot_vv,
        )?;
        acc += contract_vv(&path_second, v);
    }

    // ---- (b) geometric motion of the path's blocks at FIXED doctored references ----
    {
        let _p = crate::profile::scope("fc4.stage3.geometric_path");
        third_density_path_vvv(
            system,
            params,
            electronic,
            coordination_cutoff,
            p_v,
            w_v,
            q_v,
            v_pot_v,
            None,
            v,
            pulay_third_reference,
            &mut acc,
        )?;
    }
    // s2: the Hessian path's bilinear charge path, one geometric order up.
    {
        let _p = crate::profile::scope("fc4.stage3.scc_charge_path");
        let t = crate::hessian::fixed_shell_charge_scc_third_charge_path(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            q_v,
            params,
        )?;
        acc += contract_slabs_vvv(&t, v);
    }

    // ---- (c) motion of the UNDOCTORED reference slots the path's blocks read ----
    // Pulay `(P^v, V^v)` bilinear, multiplicity 2 (V_ref-motion of pulay(P^v,W^v,V) and
    // P_ref-motion of the V-shifted term are the same object).
    {
        let _p = crate::profile::scope("fc4.stage3.pulay_reference_motion");
        let mut e1 = electronic.clone();
        e1.density = p_v.clone();
        e1.energy_weighted_density = w_v.clone();
        let h1 = crate::hessian::fixed_density_pulay_hessian(system, params, &e1)?.hessian;
        let mut e2 = e1.clone();
        for s in 0..nshell {
            e2.shell_scc_potential[s] += v_pot_v[s];
        }
        let h2 = crate::hessian::fixed_density_pulay_hessian(system, params, &e2)?.hessian;
        acc += 2.0 * (contract_vv(&h2, v) - contract_vv(&h1, v));
        // CN-cache response of the DOCTORED Pulay Hessian (only the `2P·h0` channel carries CN).
        let cn_response = cn_first_response(system, coordination_cutoff, v)?;
        let h_cn = crate::hessian::fixed_density_pulay_cn_h0_response(
            system,
            params,
            &e1,
            &cn_response,
        )?;
        acc += contract_vv(&h_cn, v);
    }
    // Scalar-overlap `(P^v, q^v)` bilinear, multiplicity 2.
    {
        let _p = crate::profile::scope("fc4.stage3.scalar_overlap_reference_motion");
        let mut e = electronic.clone();
        e.density = p_v.clone();
        e.shell_charges = q_v.to_vec();
        let h = crate::hessian::fixed_density_scalar_overlap_hessian(system, params, &e)?;
        acc += 2.0 * contract_vv(&h, v);
    }
    // s2 charge path `(q_ref → q^v, q^v)`.
    {
        let _p = crate::profile::scope("fc4.stage3.scc_hessian_charge_path");
        let h = crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
            system,
            &electronic.basis,
            q_v,
            q_v,
            params,
        )?;
        acc += contract_vv(&h, v);
    }
    Ok(acc)
}

/// The directional FIRST coordination response `CN^v_A = Σ_c v_c ∂CN_A/∂R_c`.
fn cn_first_response(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Vec<f64>> {
    let cn_grad = crate::hessian::cn_gradient_matrix(system, coordination_cutoff)?;
    Ok(cn_grad
        .iter()
        .map(|row| row.iter().zip(v).map(|(g, vc)| g * vc).sum())
        .collect())
}

/// The directional SECOND coordination response `CN^vv_A = Σ_cd v_c v_d ∂²CN_A/∂R_c∂R_d`.
fn cn_second_response(
    system: &PeriodicSystem,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Vec<f64>> {
    let cn_hess =
        crate::hessian::coordination_number_second_derivatives(system, coordination_cutoff)?;
    Ok(cn_hess
        .iter()
        .map(|m| {
            let mut acc = 0.0;
            for (c, &vc) in v.iter().enumerate() {
                if vc == 0.0 {
                    continue;
                }
                for (d, &vd) in v.iter().enumerate() {
                    acc += vc * vd * m[(c, d)];
                }
            }
            acc
        })
        .collect())
}

/// **Stage 4** of the directional quartic assembly: the total `λ`-derivative of the FC3
/// composition's Pulay **coordination-number response** term,
/// `d/dλ [ fixed_density_pulay_cn_h0_response(R+λv, el(λ), CN^v(λ)) · vv ]`.
///
/// FC3 (`third_derivative_closed_form_total`) adds, per DOF `c`, the block
/// `fixed_density_pulay_cn_h0_response(system, params, electronic, ∂CN/∂R_c)` — the term that
/// exists because the Pulay Hessian's `h0` prefactor reads a CN **cached** in `electronic`.
/// Directionally that contributes `cn_resp(CN^v; P_ref, R)·vv` with `CN^v_A = Σ_c v_c ∂CN_A/∂R_c`.
///
/// The block `C(R, P, CN^v)` is LINEAR in the density slot and LINEAR in the `CN^v` slot (its only
/// CN entry is `s_cn = −½(kcn_i·CN^v_i + kcn_j·CN^v_j)` scaling the *geometric* factor
/// `hscale·shell_poly`, which carries no CN of its own — the shell self-energy is linear in CN, so
/// there is NO `∂²/∂CN²` term). The product rule therefore has exactly THREE groups:
///
/// **(1) bundle motion** — `∂_P C · P^v`: the same routine fed the screened directional first-order
/// bundle as a doctored reference (`density → P^v`, `energy_weighted_density → W^v`; the formula
/// reads only `P`, but the pair screen looks at both, so both are doctored), same `CN^v`.
///
/// **(2) CN-grad motion** — `∂_{CN^v} C · CN^vv`: the same routine at the REFERENCE electronic, fed
/// the second directional coordination response `CN^vv_A = Σ_cd v_c v_d ∂²CN_A/∂R_c∂R_d`.
///
/// **(3) geometric motion** — `∂_R C · v`: [`crate::hessian::fixed_density_pulay_third_cn_response`]
/// with the SAME `CN^v`, contracted `vvv`. This is a REUSE, not a new routine, and it is exact by
/// commutation of partials: that function was written as `∂_CN(pulay_third)·CN^v`, i.e.
/// `2P·∂_a∂_b∂_c(h0^cn·S)`, while what is wanted here is `∂_R(∂_CN pulay_hess · CN^v)`, i.e.
/// `∂_c[2P·∂_a∂_b(h0^cn·S)]`. Both `R`- and `CN`-derivatives act on the same frozen-density
/// `Σ_{μν} 2P_{μν}·h0_{μν}(R,CN)·S_{μν}(R)` sum, and mixed partials commute — field for field the
/// two routines are the second- and third-order Leibniz expansions of the SAME product
/// `h0^cn·S`, with `h0_scale_third` the exact one-order-up twin of `h0_scale_second`. The
/// `pulay_third_cn_response_is_geometric_derivative_of_hessian_cn_response` test FD-pins the
/// identity at frozen density.
pub fn directional_fourth_cn_response_stage(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &crate::electronic::ElectronicResult,
    coordination_cutoff: f64,
    p_v: &Matrix,
    w_v: &Matrix,
    v: &[f64],
) -> Result<f64> {
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "directional_fourth_cn_response_stage: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    let cn_v = cn_first_response(system, coordination_cutoff, v)?;
    let mut acc = 0.0;

    // ---- (1) bundle motion: the block is linear in P, so feed it the first-order bundle ----
    {
        let mut e1 = electronic.clone();
        e1.density = p_v.clone();
        e1.energy_weighted_density = w_v.clone();
        let h = crate::hessian::fixed_density_pulay_cn_h0_response(system, params, &e1, &cn_v)?;
        acc += contract_vv(&h, v);
    }

    // ---- (2) CN-grad motion: the block is linear in CN^v, so feed it CN^vv ----
    {
        let _p = crate::profile::scope("fc4.stage4.cn_second_response");
        let cn_vv = cn_second_response(system, coordination_cutoff, v)?;
        let h = crate::hessian::fixed_density_pulay_cn_h0_response(
            system, params, electronic, &cn_vv,
        )?;
        acc += contract_vv(&h, v);
    }

    // ---- (3) geometric motion at frozen density and frozen CN^v ----
    {
        let _p = crate::profile::scope("fc4.stage4.pulay_third_cn_response");
        let t = crate::hessian::fixed_density_pulay_third_cn_response(
            system,
            params,
            electronic,
            coordination_cutoff,
            &cn_v,
        )?;
        acc += contract_slabs_vvv(&t, v);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::{run_electronic, ElectronicOptions};
    use crate::response::charge_space::{ChargeSpaceContext, FirstOrderField, SecondOrderBundle};
    use crate::response::cpxtb::{
        solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions,
    };

    /// The non-equilibrium water gate geometry + tight-SCF options shared by the
    /// response-stage tests.
    fn gate_system_options() -> (PeriodicSystem, ElectronicOptions, Vec<f64>) {
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
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        (system, options, v)
    }

    /// Reconverge the SCF at `system` and build the screened directional FIRST-order bundle along
    /// `v` together with the TOTAL potential derivative `V^v = ∂V/∂R·v + K q^v`.
    fn directional_first_order_at(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        v: &[f64],
    ) -> (
        crate::electronic::ElectronicResult,
        ChargeSpaceContext,
        FirstOrderField,
        Vec<f64>,
    ) {
        let electronic = run_electronic(system, params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            system,
            params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(system, params, &electronic).unwrap();
        let n = electronic.basis.len();
        let ndof = 3 * system.atoms.len();
        let mut f_skel = Matrix::zeros(n, n);
        let mut s_dir = Matrix::zeros(n, n);
        for (c, &vc) in v.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    f_skel[(i, j)] += vc * cpxtb.derivative_matrices[c].h0_deriv[(i, j)];
                    s_dir[(i, j)] += vc * cpxtb.derivative_matrices[c].overlap_deriv[(i, j)];
                }
            }
        }
        let field = ctx.first_order_field(f_skel, s_dir).unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            params,
        )
        .unwrap();
        let nshell = electronic.basis.shells.len();
        let v_pot: Vec<f64> = (0..nshell)
            .map(|s| {
                let geo: f64 = (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum();
                geo + field.bundle.screened_potential[s]
            })
            .collect();
        (electronic, ctx, field, v_pot)
    }

    /// The screened directional SECOND-order bundle `(P^vv, W^vv, q^vv)` plus the total second
    /// potential derivative `V^vv = (∂²V/∂R²)·vv + 2(∂γ·v)q^v + E'''(q^v_A)² + K q^vv`.
    ///
    /// The skeleton second derivatives are built DIRECTIONALLY by summing the `(c, d)` blocks with
    /// weights `v_c v_d` — `O(ndof²)` block builds, fine for the 9-DOF gate molecule.
    fn directional_second_order_at(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &crate::electronic::ElectronicResult,
        ctx: &ChargeSpaceContext,
        field: &FirstOrderField,
        cutoff: f64,
        v: &[f64],
    ) -> (SecondOrderBundle, Vec<f64>) {
        let basis = &electronic.basis;
        let n = basis.len();
        let nshell = basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )
        .unwrap();
        let zeros = vec![0.0_f64; nshell];
        let mut f_vv = Matrix::zeros(n, n);
        let mut s_vv = Matrix::zeros(n, n);
        for c in 0..ndof {
            if v[c] == 0.0 {
                continue;
            }
            for d in 0..ndof {
                let w = v[c] * v[d];
                if w == 0.0 {
                    continue;
                }
                let bare = crate::hessian::h0_bare_second_derivative_matrix(
                    system, params, electronic, c, d,
                )
                .unwrap();
                let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                    system, params, electronic, cutoff, c, d,
                )
                .unwrap();
                let v_geo_d: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, d)]).collect();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    system, params, electronic, &v_geo_d, &zeros, c, d,
                )
                .unwrap();
                let sov = crate::response::cpxtb::overlap_second_derivative_matrix(
                    system, basis, c, d,
                )
                .unwrap();
                for k in 0..n * n {
                    f_vv.as_mut_slice()[k] +=
                        w * (bare.as_slice()[k] + cn_block.as_slice()[k] + scc.as_slice()[k]);
                    s_vv.as_mut_slice()[k] += w * sov.as_slice()[k];
                }
            }
        }
        // (∂γ/∂R) q^v — both the mirrored second-order source and the V^vv cross term.
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &field.bundle.shell_charges,
            params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let second = ctx
            .solve_second_order(field, field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)
            .unwrap();

        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )
        .unwrap();
        let kernel =
            crate::response::cpxtb::response_shell_scc_kernel(system, params, electronic).unwrap();
        // Onsite anharmonic chain: per shell `E'''_A · (q^v_A)²` (the ∂²V/∂q² term).
        let chain = ctx.kernel_chain_potential(
            &field.bundle.shell_charges,
            &field.bundle.shell_charges,
        );
        let v_pot_vv: Vec<f64> = (0..nshell)
            .map(|s| {
                let geo2: f64 = (0..ndof)
                    .map(|c| {
                        (0..ndof)
                            .map(|d| v[c] * v[d] * d2vdr_q[s][(c, d)])
                            .sum::<f64>()
                    })
                    .sum();
                let cross: f64 =
                    2.0 * (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum::<f64>();
                let kq: f64 = (0..nshell)
                    .map(|t| kernel[(s, t)] * second.shell_charges[t])
                    .sum();
                geo2 + cross + chain[s] + kq
            })
            .collect();
        (second, v_pot_vv)
    }

    /// The `s2` block is QUADRATIC in the shell charges, so its third-derivative bilinear charge
    /// path must reproduce the polarization identity `T(q + u) − T(q) − T(u)` exactly. Guards the
    /// weight convention (`q^path_i q_j + q_i q^path_j`, no stray ½ or 2) that stage 3 relies on.
    #[test]
    fn s2_third_charge_path_matches_polarization() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, _v) = gate_system_options();
        let electronic = run_electronic(&system, &params, options).unwrap();
        let basis = &electronic.basis;
        let q = &electronic.shell_charges;
        let u: Vec<f64> = (0..q.len()).map(|s| 0.31 - 0.17 * (s as f64)).collect();
        let path = crate::hessian::fixed_shell_charge_scc_third_charge_path(
            &system, basis, q, &u, &params,
        )
        .unwrap();
        let third = |c: &[f64]| {
            crate::hessian::fixed_shell_charge_scc_third_derivative(&system, basis, c, &params)
                .unwrap()
        };
        let sum: Vec<f64> = q.iter().zip(&u).map(|(a, b)| a + b).collect();
        let (t_sum, t_q, t_u) = (third(&sum), third(q), third(&u));
        let mut worst = 0.0_f64;
        for c in 0..path.len() {
            for k in 0..path[c].as_slice().len() {
                let identity =
                    t_sum[c].as_slice()[k] - t_q[c].as_slice()[k] - t_u[c].as_slice()[k];
                worst = worst.max((path[c].as_slice()[k] - identity).abs());
            }
        }
        assert!(
            worst < 1.0e-12,
            "s2 third charge path vs polarization identity: {worst:.3e}"
        );
    }

    /// **Stage 3 gate.** The analytic total `λ`-derivative of the FC3 composition's Hessian-level
    /// density path must match the central FD of `frozen_hessian_density_path(X^v)·vv` along `v`
    /// with EVERYTHING rebuilt at the displaced geometries (reconverged SCF, fresh
    /// `ChargeSpaceContext`, fresh directional first-order field). Two FD steps assert the h²
    /// truncation scaling, separating FD noise from a missing analytic term.
    #[test]
    fn directional_hessian_path_stage_matches_fd_along_v() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = gate_system_options();
        let cutoff = options.hamiltonian.coordination_cutoff;

        let (electronic, ctx, field, v_pot_v) =
            directional_first_order_at(&system, &params, &options, &v);
        let (second, v_pot_vv) =
            directional_second_order_at(&system, &params, &electronic, &ctx, &field, cutoff, &v);

        let analytic = directional_fourth_hessian_path_stage(
            &system,
            &params,
            &electronic,
            cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            &field.bundle.shell_charges,
            &v_pot_v,
            &second.density,
            &second.energy_weighted,
            &second.shell_charges,
            &v_pot_vv,
            &v,
        )
        .unwrap();

        let path_vv = |sys: &PeriodicSystem| -> f64 {
            let (el, _ctx, fld, vp) = directional_first_order_at(sys, &params, &options, &v);
            let m = crate::third_derivative::frozen_hessian_density_path(
                sys,
                &params,
                &el,
                cutoff,
                &fld.bundle.density,
                &fld.bundle.energy_weighted,
                &fld.bundle.shell_charges,
                &vp,
            )
            .unwrap();
            contract_vv(&m, &v)
        };
        let displace = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
                atom.position.x += step * v[3 * atom_idx];
                atom.position.y += step * v[3 * atom_idx + 1];
                atom.position.z += step * v[3 * atom_idx + 2];
            }
            sys
        };
        let fd_at = |h: f64| -> f64 { (path_vv(&displace(h)) - path_vv(&displace(-h))) / (2.0 * h) };
        // Step choice: this stage's analytic value is so close to the FD that at h = 1e-4 the
        // residual (~2e-14 abs, ~2e-10 rel) is already BELOW the central-difference roundoff floor
        // (which grows as 1/h), so halving h there makes the residual grow — noise, not a missing
        // term. h = 1e-3 puts the h² truncation an order above the floor, making the scaling
        // assertion discriminating again.
        let h1 = 1.0e-3;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "hessian-path stage: analytic {analytic:.10e} fd(h) {fd1:.10e} fd(h/2) {fd2:.10e} \
             delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-6 * (1.0 + fd1.abs()),
            "hessian-path stage vs FD: analytic {analytic:.10e} fd {fd1:.10e} delta {delta1:.3e}"
        );
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h^2 (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing analytic term"
        );
    }

    /// **Stage-4 reuse pin.** [`crate::hessian::fixed_density_pulay_third_cn_response`] was written
    /// as `∂_CN(pulay_third)·CN^v`; stage 4 needs `∂_R(∂_CN pulay_hess·CN^v)`. Mixed partials of the
    /// same frozen-density sum `Σ 2P·h0(R,CN)·S(R)` commute, so the two are the SAME object — this
    /// probe FD-pins the identity (frozen electronic reference, frozen `CN^v` vector, so ONLY the
    /// geometric slot moves) and licenses the reuse instead of a new `..._response_third` routine.
    #[test]
    fn pulay_third_cn_response_is_geometric_derivative_of_hessian_cn_response() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = gate_system_options();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let electronic = run_electronic(&system, &params, options).unwrap();
        // The CN response vector is held FIXED at its reference value: this probe isolates the
        // geometric slot, exactly the role term (3) of the stage plays.
        let cn_v = cn_first_response(&system, cutoff, &v).unwrap();

        let t = crate::hessian::fixed_density_pulay_third_cn_response(
            &system, &params, &electronic, cutoff, &cn_v,
        )
        .unwrap();
        let analytic = contract_slabs_vvv(&t, &v);

        let hess_vv = |sys: &PeriodicSystem| -> f64 {
            let m = crate::hessian::fixed_density_pulay_cn_h0_response(
                sys, &params, &electronic, &cn_v,
            )
            .unwrap();
            contract_vv(&m, &v)
        };
        let displace = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
                atom.position.x += step * v[3 * atom_idx];
                atom.position.y += step * v[3 * atom_idx + 1];
                atom.position.z += step * v[3 * atom_idx + 2];
            }
            sys
        };
        let h = 1.0e-4;
        let fd = (hess_vv(&displace(h)) - hess_vv(&displace(-h))) / (2.0 * h);
        let delta = (analytic - fd).abs();
        eprintln!(
            "pulay third-CN vs FD(hessian CN-response): analytic {analytic:.10e} fd {fd:.10e} \
             delta {delta:.3e}"
        );
        assert!(
            delta < 1.0e-6 * (1.0 + fd.abs()),
            "pulay_third_cn_response is not the geometric derivative of \
             fixed_density_pulay_cn_h0_response: analytic {analytic:.10e} fd {fd:.10e} \
             delta {delta:.3e}"
        );
    }

    /// **Stage 4 gate.** The analytic total `λ`-derivative of the FC3 composition's Pulay
    /// CN-response term must match the central FD of
    /// `fixed_density_pulay_cn_h0_response(sys(λ), el(λ), CN^v(λ))·vv` along `v` with EVERYTHING
    /// rebuilt at the displaced geometries: the SCF reconverged (so the analytic bundle term must
    /// use the SCREENED directional density response `P^v`, not a skeleton one) and `CN^v` rebuilt
    /// from the displaced geometry's CN gradients (so the analytic side needs `CN^vv`). Two FD
    /// steps assert the h² truncation scaling, separating FD noise from a missing analytic term.
    #[test]
    fn directional_cn_response_stage_matches_fd_along_v() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = gate_system_options();
        let cutoff = options.hamiltonian.coordination_cutoff;

        let (electronic, _ctx, field, _v_pot_v) =
            directional_first_order_at(&system, &params, &options, &v);

        let analytic = directional_fourth_cn_response_stage(
            &system,
            &params,
            &electronic,
            cutoff,
            &field.bundle.density,
            &field.bundle.energy_weighted,
            &v,
        )
        .unwrap();

        let cn_resp_vv = |sys: &PeriodicSystem| -> f64 {
            let el = run_electronic(sys, &params, options.clone()).unwrap();
            let cn_v = cn_first_response(sys, cutoff, &v).unwrap();
            let m =
                crate::hessian::fixed_density_pulay_cn_h0_response(sys, &params, &el, &cn_v)
                    .unwrap();
            contract_vv(&m, &v)
        };
        let displace = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
                atom.position.x += step * v[3 * atom_idx];
                atom.position.y += step * v[3 * atom_idx + 1];
                atom.position.z += step * v[3 * atom_idx + 2];
            }
            sys
        };
        let fd_at =
            |h: f64| -> f64 { (cn_resp_vv(&displace(h)) - cn_resp_vv(&displace(-h))) / (2.0 * h) };
        // Same step choice as the stage-3 gate: h = 1e-3 keeps the h² truncation an order above
        // the central-difference roundoff floor, so the halving assertion stays discriminating.
        let h1 = 1.0e-3;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "cn-response stage: analytic {analytic:.10e} fd(h) {fd1:.10e} fd(h/2) {fd2:.10e} \
             delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-6 * (1.0 + fd1.abs()),
            "cn-response stage vs FD: analytic {analytic:.10e} fd {fd1:.10e} delta {delta1:.3e}"
        );
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h^2 (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing analytic term"
        );
    }

    /// The frozen-density Hamiltonian stage (Phase-4d fourth blocks + third
    /// density paths along the screened directional bundle) must match the
    /// central FD along `v` of the frozen-density THIRD blocks with the
    /// electronic reference RECONVERGED at the displaced geometries. Two FD
    /// steps assert h² truncation scaling, separating FD noise from analytic
    /// error.
    #[test]
    fn directional_frozen_density_fourth_matches_third_fd_along_v() {
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
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        // Directional screened first-order bundle.
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        let n = electronic.basis.len();
        let mut f_skel = Matrix::zeros(n, n);
        let mut s_dir = Matrix::zeros(n, n);
        for (c, &vc) in v.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    f_skel[(i, j)] += vc * cpxtb.derivative_matrices[c].h0_deriv[(i, j)];
                    s_dir[(i, j)] += vc * cpxtb.derivative_matrices[c].overlap_deriv[(i, j)];
                }
            }
        }
        let bundle = ctx.solve_first_order(&f_skel, &s_dir).unwrap();
        // Total potential derivative: geometric + screening.
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nshell = electronic.basis.shells.len();
        let v_pot_v: Vec<f64> = (0..nshell)
            .map(|s| {
                let geo: f64 = (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum();
                geo + bundle.screened_potential[s]
            })
            .collect();

        let analytic = directional_fourth_frozen_density(
            &system,
            &params,
            &electronic,
            cutoff,
            &bundle.density,
            &bundle.energy_weighted,
            &bundle.shell_charges,
            &v_pot_v,
            &v,
        )
        .unwrap();

        // FD reference with reconverged electronic state.
        let third_vvv = |sys: &PeriodicSystem| -> f64 {
            let el = run_electronic(sys, &params, options.clone()).unwrap();
            let mut acc = 0.0;
            let p3 =
                crate::hessian::fixed_density_pulay_third_derivative(sys, &params, &el).unwrap();
            acc += contract_slabs_vvv(&p3, &v);
            let c3 = crate::hessian::fixed_density_cn_h0_third_derivative(
                sys, &params, &el, cutoff,
            )
            .unwrap();
            acc += contract_slabs_vvv(&c3, &v);
            let s3 = crate::hessian::fixed_density_scalar_overlap_third_derivative(
                sys, &params, &el,
            )
            .unwrap();
            acc += contract_slabs_vvv(&s3, &v);
            acc
        };
        let displace = |step: f64| -> PeriodicSystem {
            let mut sys = system.clone();
            for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
                atom.position.x += step * v[3 * atom_idx];
                atom.position.y += step * v[3 * atom_idx + 1];
                atom.position.z += step * v[3 * atom_idx + 2];
            }
            sys
        };
        let fd_at = |h: f64| -> f64 {
            (third_vvv(&displace(h)) - third_vvv(&displace(-h))) / (2.0 * h)
        };
        let h1 = 1.0e-4;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "frozen-density directional fourth: analytic {analytic:.10e} fd(h) {fd1:.10e} \
             delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-5 * (1.0 + fd1.abs()),
            "frozen-density stage vs FD: delta {delta1:.3e}"
        );
        // h² truncation: halving the step must shrink the residual ~4× —
        // proving the mismatch is FD noise, not a missing analytic term.
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h^2 (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing analytic term"
        );
    }

    /// The pure-geometry frozen fourth (repulsion + halogen + SCC2 + D3) along
    /// `v` must match the central FD along `v` of the same terms' directional
    /// THIRD derivative — validating the contraction plumbing end to end.
    #[test]
    fn directional_geometric_fourth_matches_third_fd_along_v() {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "8\nCH3Br...OH2\nC 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\nH 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\nH -0.515000 -0.892000 -0.330000\nO 0.000000 0.100000 4.900000\nH 0.760000 0.100000 5.470000\nH -0.760000 0.100000 5.470000\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = AnalyticHessianOptions::default();
        options.electronic_options.enable_dispersion = true;
        options.electronic_options.energy_tolerance = 1.0e-11;
        options.electronic_options.charge_tolerance = 1.0e-9;
        let electronic = run_electronic(&system, &params, options.electronic_options.clone())
            .unwrap();
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.11 + 0.07 * ((k * 13 % 7) as f64) - 0.15 * ((k % 3) as f64))
            .collect();

        let analytic =
            directional_fourth_geometric_with(&system, &params, &electronic, &options, &v)
                .unwrap();

        // FD reference: the same terms' directional third derivative at R ± hv.
        let third_vvv = |sys: &PeriodicSystem, el: &crate::electronic::ElectronicResult| -> f64 {
            let mut acc = 0.0;
            let geo = crate::third_derivative::third_derivative_geometric(sys, &params).unwrap();
            acc += geo.contract_vvv(&v);
            let scc = crate::hessian::fixed_shell_charge_scc_third_derivative(
                sys,
                &el.basis,
                &el.shell_charges,
                &params,
            )
            .unwrap();
            acc += contract_slabs_vvv(&scc, &v);
            let disp = crate::third_derivative::third_derivative_dispersion(
                sys,
                &params,
                options.electronic_options.d3_reference_path.as_deref(),
            )
            .unwrap();
            acc += disp.contract_vvv(&v);
            acc
        };
        let h = 1.0e-4;
        let displace = |sign: f64| -> (PeriodicSystem, crate::electronic::ElectronicResult) {
            let mut sys = system.clone();
            for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
                atom.position.x += sign * h * v[3 * atom_idx];
                atom.position.y += sign * h * v[3 * atom_idx + 1];
                atom.position.z += sign * h * v[3 * atom_idx + 2];
            }
            // IMPORTANT: freeze the electronic reference — the frozen SCC2
            // block holds the charges fixed, so the FD must too.
            (sys, electronic.clone())
        };
        let (sys_p, el_p) = displace(1.0);
        let (sys_m, el_m) = displace(-1.0);
        let fd = (third_vvv(&sys_p, &el_p) - third_vvv(&sys_m, &el_m)) / (2.0 * h);
        let delta = (analytic - fd).abs();
        assert!(
            delta < 1.0e-5 * (1.0 + fd.abs()),
            "directional geometric fourth vs third FD: analytic {analytic:.10e} fd {fd:.10e} delta {delta:.3e}"
        );
    }
}
