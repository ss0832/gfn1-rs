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
use crate::response::charge_space::{ChargeSpaceContext, FirstOrderField, SecondOrderBundle};
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
    let n = electronic.basis.len();
    let ndof = 3 * system.atoms.len();
    let nshell = electronic.basis.shells.len();
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
    let field = ctx.first_order_field(f_skel, s_dir)?;
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
    let basis = &electronic.basis;
    let n = basis.len();
    let nshell = basis.shells.len();
    let ndof = 3 * system.atoms.len();
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
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
    let second = ctx.solve_second_order(field, field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)?;

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
                .map(|t| kernel[(s, t)] * second.shell_charges[t])
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
/// Both guards of the analytic quartic live here — the order-4 term registry
/// check and the integer-occupation check — so every path that goes through a
/// reference inherits them.
pub struct QuarticReference {
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

impl QuarticReference {
    /// Converge the SCF, solve the CPXTB Hessian response and build the
    /// charge-space context for `system`, after checking the analytic-order-4
    /// and integer-occupation guards.
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &AnalyticHessianOptions,
        coordination_cutoff: f64,
    ) -> Result<Self> {
        crate::terms::require_order(
            &options.electronic_options,
            params,
            4,
            "the analytic fourth derivative",
        )?;
        let _profile = crate::profile::scope("fc4.reference.total");
        let electronic = {
            let _p = crate::profile::scope("fc4.reference.scf");
            crate::electronic::run_electronic(system, params, options.electronic_options.clone())?
        };
        if electronic
            .occupations
            .iter()
            .any(|&f| f > 1.0e-8 && (f - 2.0).abs() > 1.0e-8)
        {
            return Err(Gfn1Error::InvalidInput(
                "analytic fourth derivative with fractional (Fermi-smeared) occupations is not \
                 yet supported; use directional_fourth_seminumerical until the finite-temperature \
                 analytic path lands"
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
            electronic,
            cphf,
            ctx,
            pulay_third_reference,
        })
    }

    /// The converged reference SCF state (shared by all directions).
    pub fn electronic(&self) -> &crate::electronic::ElectronicResult {
        &self.electronic
    }
}

/// **The analytic directional fourth derivative** `e⁗[v] = Q·vvvv` — one SCF,
/// one CPXTB solve, one charge-space first-order solve and one second-order
/// solve along `v`, then the five gated stages summed.
///
/// Guards: the term registry must support analytic order 4 for every active
/// term, and the converged occupations must be integers (the finite-temperature
/// quartic response is not implemented yet — use
/// [`directional_fourth_seminumerical`] for smeared systems).
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
    check_direction_len(system, v, "directional_fourth_with_reference")?;
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
type DirectionKey = Vec<(usize, u8)>;

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
fn polarization_terms(quad: [usize; 4]) -> Vec<(f64, DirectionKey)> {
    let mut terms = Vec::with_capacity(15);
    for mask in 1u32..16 {
        let mut key: DirectionKey = Vec::with_capacity(4);
        let mut size = 0u32;
        for (slot, &dof) in quad.iter().enumerate() {
            if mask & (1 << slot) == 0 {
                continue;
            }
            size += 1;
            match key.iter_mut().find(|(d, _)| *d == dof) {
                Some(entry) => entry.1 += 1,
                None => key.push((dof, 1)),
            }
        }
        key.sort_unstable();
        let sign = if (4 - size) % 2 == 0 { 1.0 } else { -1.0 };
        terms.push((sign, key));
    }
    terms
}

/// Assemble the analytic quartic over the canonical quadruples drawn from
/// `dofs`, indexing the packed store by POSITION in `dofs`.
///
/// Two phases so the expensive half parallelises cleanly:
///
/// 1. deduplicate the polarization directions of every quadruple — the 15
///    subsets are massively shared, so a build over `m` DOFs needs only the
///    multisets of size `1..=4` drawn from them, `Σ_{k=1}^{4} C(m+k−1, k)`
///    directions (714 for a water molecule, versus 495·15 = 7425 without the
///    deduplication);
/// 2. evaluate each distinct direction ONCE against the shared reference (in
///    parallel — the evaluations are independent and read-only), then contract
///    the cached values with the identity's signs.
///
/// The result is order-independent: every element is a fixed linear combination
/// of cached direction values.
fn assemble_polarized_quartic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticHessianOptions,
    coordination_cutoff: f64,
    dofs: &[usize],
) -> Result<SymmetricFourth> {
    use rayon::prelude::*;

    let m = dofs.len();
    let mut slot_of: std::collections::HashMap<DirectionKey, usize> =
        std::collections::HashMap::new();
    let mut directions: Vec<DirectionKey> = Vec::new();
    // `plan[q]` holds the 15 `(sign, direction slot)` pairs of the q-th
    // canonical quadruple, in the same order the assembly loop visits them.
    let mut plan: Vec<Vec<(f64, usize)>> = Vec::new();
    for di in 0..m {
        for ci in 0..=di {
            for bi in 0..=ci {
                for ai in 0..=bi {
                    let terms = polarization_terms([dofs[ai], dofs[bi], dofs[ci], dofs[di]]);
                    let mut entry = Vec::with_capacity(terms.len());
                    for (sign, key) in terms {
                        let slot = *slot_of.entry(key.clone()).or_insert_with(|| {
                            directions.push(key);
                            directions.len() - 1
                        });
                        entry.push((sign, slot));
                    }
                    plan.push(entry);
                }
            }
        }
    }

    let reference = QuarticReference::build(system, params, options, coordination_cutoff)?;
    let ndof = 3 * system.atoms.len();
    let values: Vec<f64> = directions
        .par_iter()
        .map(|key| {
            let mut v = vec![0.0_f64; ndof];
            for &(dof, weight) in key {
                v[dof] = f64::from(weight);
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
        .collect::<Result<Vec<f64>>>()?;

    let mut store = SymmetricFourth::zeros(m);
    let mut quad = 0usize;
    for di in 0..m {
        for ci in 0..=di {
            for bi in 0..=ci {
                for ai in 0..=bi {
                    let acc: f64 = plan[quad]
                        .iter()
                        .map(|&(sign, slot)| sign * values[slot])
                        .sum();
                    quad += 1;
                    // One write per unordered quadruple: `add` accumulates, and
                    // the canonical loop visits each packed slot exactly once.
                    store.add(ai, bi, ci, di, acc / 24.0);
                }
            }
        }
    }
    Ok(store)
}

/// **The full mixed-index analytic quartic force constants** `Q_abcd`, packed
/// into a [`SymmetricFourth`].
///
/// Every element is recovered from the (integration-gated) DIRECTIONAL quartic
/// `e⁗[v] = Q·vvvv` by the polarization identity, so the mixed-index tensor
/// inherits the directional assembly's correctness by construction — no
/// separate mixed-index derivation, no finite differences in the nuclear
/// coordinates. One SCF / CPXTB / charge-space reference is shared by every
/// direction, and identical directions across quadruples are evaluated once.
///
/// Cost: `Σ_{k=1}^{4} C(ndof+k−1, k)` directional evaluations — asymptotically
/// `ndof⁴/24`, i.e. one per distinct tensor element — each cheap because its
/// direction has at most four non-zero components and the per-direction
/// skeleton loops skip zero components. Use
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
    assemble_polarized_quartic(system, params, options, coordination_cutoff, &dofs)
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
    assemble_polarized_quartic(system, params, options, coordination_cutoff, dofs)
}

/// **Seminumerical directional fourth derivative** — the central finite
/// difference along `v` of the analytic directional THIRD derivative
/// `e³[v] = Σ_abc T_abc v_a v_b v_c`, evaluated at `R ± h·v` with everything
/// reconverged: the verification reference of the analytic assembly and the
/// production fallback where the analytic quartic is not available.
///
/// The FD'd reference is the occupation-agnostic
/// [`crate::third_derivative::finite_t::directional_third_finite_t`], so this
/// route ALSO serves **Fermi-smeared** systems, for which the analytic quartic
/// ([`directional_fourth_derivative`]) still errs on fractional occupations.
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
        let cutoff = options
            .electronic_options
            .hamiltonian
            .coordination_cutoff;

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
            1.0 + a + 2.0 * b - 0.5 * c + a * b * c * d - 0.25 * (a * a + d * d)
                + (a + b) * (c + d)
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
                            for (dof, weight) in key {
                                v[dof] = f64::from(weight);
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
