// SPDX-License-Identifier: GPL-3.0-or-later
//! **Directional response stage** of the quartic assembly: the single scalar
//!
//! ```text
//!   r3[v] = Σ_abc v_a v_b v_c · D_c(cphf.hessian_response_ab)
//! ```
//!
//! i.e. the `vvv` contraction of the validated dense slabs produced by
//! [`crate::third_derivative::closed_form_response_hessian_derivative`], but computed WITHOUT ever
//! materializing a Cartesian-indexed object: every free nuclear index is pre-contracted with the
//! fixed direction `v` before any algebra happens. That is what the directional quartic stage needs
//! — a form compact enough to hand-differentiate one more order in `λ` along `R + λv`.
//!
//! # Why the pre-contraction is exact
//!
//! The reference builds `resp_c[(a,b)]` from three index roles:
//!   * `c` — the differentiation slab (`D_c`), entering through the per-`c` first-order bundle
//!     (`C^(c)`, `Λ^c`, `S_c`, `P^(c)`, `q^(c)`, `V^(c)`) **and** through the geometric `c` leg of the
//!     skeleton second derivatives;
//!   * `b` — the perturbation of the Hessian column, entering through `S_b`, `F_b`, `x_b` and the
//!     static/orbital bundles built from them;
//!   * `a` — the Hessian *row*, entering only as the gradient DOF of the bundle-gradient `G_a` and as
//!     the Z-vector index `y_a`.
//!
//! Every term of the reference is **linear in the `a` role** and **bilinear in the `(b, c)` roles**
//! (linear in each separately). This was audited term by term while porting:
//!
//! | reference term | `b` factor | `c` factor | verdict |
//! |---|---|---|---|
//! | `f_bc`, `s_bc` | geometric `b` leg | geometric `c` leg + `V^(c)`/`q^(c)` | bilinear (the `V^(c)`/`q^(c)` pieces are *additive* against the geometric `c` leg, never multiplied by it — `D_c` is a derivation) |
//! | `d_s_tilde`, `d_f_tilde` | `S_b`, `F_b` | `C^(c)`, `S_bc`, `F_bc` | bilinear |
//! | orbital bundle `B x_b` | `x_b` (exactly linear, see `OrbitalResponseBundle`) | — | linear in `b` |
//! | `D_c(B x_b)` | `x_b` | `C^(c)`, `Λ^c`, `S_c`, `q^(c)` | bilinear |
//! | static bundle | `S̃_b`, `F̃_b` | — | linear in `b` |
//! | `D_c(static_b)` | `S̃_b`, `F̃_b` | `C^(c)`, `S_bc`, `F_bc`, `P^(c)`, `S_c` | bilinear |
//! | `bundle_grad` Group A | `dp`, `dq` | `h0_bare_second(·,c)+cn(·,c)`, `∂²V/∂R∂R_c`, `∂V[q^(c)]/∂R` | bilinear |
//! | `bundle_grad` pulay loop | `dp`, `dw`, `dq` | `∂²S/∂R∂R_c` (via `atom_c`/`axis_c`), `V^(c)`, `P^(c)`, `q^(c)` | bilinear |
//! | `bundle_grad` Group B | — | — | linear in the (already bilinear) derivative bundle |
//! | `d_rhs`, `d_axb` | `S̃_b`, `F̃_b`, `x_b` | `Λ^c`, `C^(c)`, `S_c`, `q^(c)` | bilinear |
//!
//! **No term is quadratic in a single role.** In particular no term multiplies two per-`c` objects
//! (that would make `D_c` a second derivative, which it is not) and no term multiplies two per-`b`
//! objects. Concretely, the two places that *look* dangerous are safe:
//!   * `dq_s = pop(dp_s, S) + pop(P, S_b)` — a **sum** of two `b`-linear pieces, not a product, so
//!     `dq_s^v = pop(dp_s^v, S) + pop(P, S_v)`;
//!   * `d_dq_s` carries `pop(dp_s, S_c)` and `pop(P^(c), S_b)` — a `b`-object times a `c`-object, so
//!     both collapse to `pop(·, S_v)` under `Σ_b v_b Σ_c v_c`.
//!
//! Because `b` and `c` are **always distinct loop variables contracted against separate `v` factors**
//! in the reference, `Σ_bc v_b v_c f(b, c) = f(v, v)` for the bilinear `f` above, and the directional
//! substitution is sound. Had any term been bilinear in per-`c` and per-`b` objects with the SAME
//! index (`c = b`), naive contraction would have been wrong — that case does not occur here.
//!
//! # What this buys
//!
//! * `ndof` Z-vector solves collapse to **one**: `y_a = A⁻¹L_a` is linear in `a`, so
//!   `y_v = Σ_a v_a y_a = A⁻¹(Σ_a v_a L_a) = A⁻¹L_v`.
//! * The `ndof²` (`c`, `b`) double loop collapses to a single pass over one directional bundle.
//! * The skeleton second derivatives are built **once** as `Σ_xy v_x v_y block(x, y)` (`O(ndof²)`
//!   block builds — the same count the reference pays *per slab*).
//! * The `Λ`-covariant degenerate-block algebra (`block_members` / `lam` / `pair_of`) is preserved
//!   verbatim, with `Λ^c → Λ^v = Σ_c v_c Λ^c` (legitimate: `Λ` is linear in the derivative direction,
//!   and the block structure itself is a property of the reference state, not of `c`).
//!
//! Gated bit-for-bit (to summation-order roundoff) against the reference by
//! `directional_response_third_matches_closed_form_contraction`.
//!
//! # The quartic stage
//!
//! [`directional_response_fourth`] takes the whole thing ONE more order: the exact total
//! `λ`-derivative of `r3` along `R + λv` with the electronic reference reconverged, i.e.
//! `r4[v] = Σ_abcd v_a v_b v_c v_d D_dD_c(cphf.hessian_response_ab)`. Written in the operator form
//! `r3 = G₂[X] + G₁[X′]` (see that function's docs), the second `D` collapses to
//! `r4 = G₃[X] + 2G₂[X′] + G₁[X″]`, so the whole quartic stage reuses the third's two bundle
//! gradients unchanged and adds exactly one new operator (`G₃`, the response gradient's SECOND
//! geometric derivative) plus the second-order bundle ladder. FD-gated against `r3` by
//! `directional_response_fourth_matches_third_fd_along_v` with an `h²` scaling assertion.

use crate::electronic::ElectronicResult;
use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

/// `Σ_c v_c src(c)` for an `n×n` matrix family indexed by nuclear DOF — the elementary
/// direction-contraction used for every first-order (per-`c`) object below.
fn accum_dir<'a, F: Fn(usize) -> &'a Matrix>(n: usize, v: &[f64], src: F) -> Matrix {
    let mut out = Matrix::zeros(n, n);
    for (c, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        let m = src(c);
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] += vc * m[(i, j)];
            }
        }
    }
    out
}

/// The six direction-contracted **skeleton second-derivative** AO matrices that group (2) of
/// [`directional_response_fourth`] consumes.
struct SkeletonSecond {
    /// `Σ_bc v_b v_c ∂²(H0_bare + CN-completion)/∂R_b∂R_c` at the frozen reference.
    m_vv: Matrix,
    /// The reconverged reference's cached-CN motion of `h0_bare²`. The bare builder is AFFINE in
    /// the cached coordination numbers, so the doctored-`CN^v` build minus the `CN = 0` build is
    /// exactly `∂(h0_bare²)/∂CN · CN^v`.
    m_vv_cache_motion: Matrix,
    /// `Σ_bc v_b v_c ∂²S/∂R_b∂R_c`.
    s_vv: Matrix,
    /// The SCC-scalar block fed the TOTAL (screened) potential and charge legs.
    f_scc_vv: Matrix,
    /// The SCC-scalar block fed the frozen-charge (skeleton) geometric leg only — the convention
    /// the charge-space solver's second-order source expects.
    f_scc_vv_skeleton: Matrix,
    /// The reconverged reference's cached `(V, q)` motion of the SCC-scalar block (the builder is
    /// linear-homogeneous in both cached fields, so re-running it on their `λ`-derivatives IS the
    /// motion).
    f_scc_vv_cache_motion: Matrix,
}

/// **The one-pass directional build of [`SkeletonSecond`].**
///
/// Every one of the six matrices is `Σ_bc v_b v_c (per-`(b,c)` block)`, and the per-AO-pair
/// second-derivative data the blocks are made of does not depend on `(b, c)` at all — only the
/// contraction weights do. Contracting both legs against `v` INSIDE the pair sweep therefore
/// replaces the `ndof²` block builds (`7` matrices each, every one of them re-evaluating the same
/// pair integrals and rebuilding the whole coordination-number and shell-potential ladders) with
/// six single sweeps. This is the dominant cost of the quartic response stage: on the 8-atom
/// CH3Br···OH2 gate fixture the double loop was 609 s of the 634 s directional evaluation.
///
/// The legs are supplied already direction-contracted, which is legitimate because each builder is
/// LINEAR in them: `geo_v = ∂V/∂R|_q·v`, `pot_v = geo_v + K q^v` (the total screened potential
/// derivative) and `q_v` the directional first-order shell-charge response; `cn_v` is the
/// directional first coordination response.
///
/// Gated element-wise against [`skeleton_second_double_loop`] — the code it replaces — by
/// `skeleton_second_one_pass_matches_double_loop`.
#[allow(clippy::too_many_arguments)]
fn skeleton_second_one_pass(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cutoff: f64,
    v: &[f64],
    geo_v: &[f64],
    pot_v: &[f64],
    q_v: &[f64],
    cn_v: &[f64],
) -> Result<SkeletonSecond> {
    let basis = &electronic.basis;
    let nat = system.atoms.len();
    let zero_shell = vec![0.0_f64; basis.shells.len()];
    // Doctored references: each cached field the builders read enters them (homogeneously)
    // linearly, so re-running the builder on the field's λ-derivative yields exactly
    // `∂(builder)/∂field · field^v` (for CN, affinely — hence the two-call difference).
    let mut el_cn_v = electronic.clone();
    el_cn_v.coordination_numbers = cn_v.to_vec();
    let mut el_cn_0 = electronic.clone();
    el_cn_0.coordination_numbers = vec![0.0; nat];
    let mut el_field_v = electronic.clone();
    el_field_v.shell_scc_potential = pot_v.to_vec();
    el_field_v.shell_charges = q_v.to_vec();

    let mut m_vv =
        crate::hessian::directional_h0_bare_second_matrix(system, params, electronic, v)?;
    axpy(
        &mut m_vv,
        &crate::hessian::directional_h0_cn_block_second_matrix(
            system, params, electronic, cutoff, v,
        )?,
        1.0,
    );
    let m_vv_cache_motion = {
        let mut m = crate::hessian::directional_h0_bare_second_matrix(system, params, &el_cn_v, v)?;
        let at_zero =
            crate::hessian::directional_h0_bare_second_matrix(system, params, &el_cn_0, v)?;
        axpy(&mut m, &at_zero, -1.0);
        m
    };
    Ok(SkeletonSecond {
        m_vv,
        m_vv_cache_motion,
        s_vv: crate::hessian::directional_overlap_second_matrix(system, basis, v)?,
        f_scc_vv: crate::hessian::directional_h0_scc_scalar_second_matrix(
            system, params, electronic, v, pot_v, q_v,
        )?,
        f_scc_vv_skeleton: crate::hessian::directional_h0_scc_scalar_second_matrix(
            system,
            params,
            electronic,
            v,
            geo_v,
            &zero_shell,
        )?,
        f_scc_vv_cache_motion: crate::hessian::directional_h0_scc_scalar_second_matrix(
            system,
            params,
            &el_field_v,
            v,
            &zero_shell,
            &zero_shell,
        )?,
    })
}

/// The `ndof²` per-`(b,c)` double loop [`skeleton_second_one_pass`] replaced, kept verbatim as the
/// element-wise gate reference.
#[cfg(test)]
fn skeleton_second_double_loop(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &crate::cphf::GammaCartesianCpxtbResult,
    cutoff: f64,
    v: &[f64],
    cn_v: &[f64],
) -> Result<SkeletonSecond> {
    let basis = &electronic.basis;
    let n = basis.len();
    let nshell = basis.shells.len();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let zero_shell = vec![0.0_f64; nshell];
    let shell_kernel = crate::cphf::response_shell_scc_kernel(system, params, electronic)?;
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let kvec = |u: &[f64]| -> Vec<f64> {
        (0..nshell)
            .map(|s| (0..nshell).map(|t| shell_kernel[(s, t)] * u[t]).sum::<f64>())
            .collect()
    };
    let q_v: Vec<f64> = (0..nshell)
        .map(|s| {
            (0..ndof)
                .map(|c| v[c] * cphf.shell_charge_responses[c][s])
                .sum()
        })
        .collect();
    let pot_v: Vec<f64> = {
        let kq = kvec(&q_v);
        (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum::<f64>() + kq[s])
            .collect()
    };
    let mut el_cn_v = electronic.clone();
    el_cn_v.coordination_numbers = cn_v.to_vec();
    let mut el_cn_0 = electronic.clone();
    el_cn_0.coordination_numbers = vec![0.0; nat];
    let mut el_field_v = electronic.clone();
    el_field_v.shell_scc_potential = pot_v;
    el_field_v.shell_charges = q_v;

    let mut out = SkeletonSecond {
        m_vv: Matrix::zeros(n, n),
        m_vv_cache_motion: Matrix::zeros(n, n),
        s_vv: Matrix::zeros(n, n),
        f_scc_vv: Matrix::zeros(n, n),
        f_scc_vv_skeleton: Matrix::zeros(n, n),
        f_scc_vv_cache_motion: Matrix::zeros(n, n),
    };
    for c in 0..ndof {
        if v[c] == 0.0 {
            continue;
        }
        let q_c = &cphf.shell_charge_responses[c];
        let pot_c: Vec<f64> = {
            let kq = kvec(q_c);
            (0..nshell).map(|s| dvdr_q[(s, c)] + kq[s]).collect()
        };
        let geo_c: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, c)]).collect();
        for b in 0..ndof {
            let w = v[b] * v[c];
            if w == 0.0 {
                continue;
            }
            let h0 =
                crate::hessian::h0_bare_second_derivative_matrix(system, params, electronic, b, c)?;
            let cn = crate::hessian::h0_cn_block_second_derivative_matrix(
                system, params, electronic, cutoff, b, c,
            )?;
            let h0_cnv =
                crate::hessian::h0_bare_second_derivative_matrix(system, params, &el_cn_v, b, c)?;
            let h0_cn0 =
                crate::hessian::h0_bare_second_derivative_matrix(system, params, &el_cn_0, b, c)?;
            let sbc = crate::cphf::overlap_second_derivative_matrix(system, basis, b, c)?;
            let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                system, params, electronic, &pot_c, q_c, b, c,
            )?;
            let scc_skel = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                system,
                params,
                electronic,
                &geo_c,
                &zero_shell,
                b,
                c,
            )?;
            let scc_motion = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                system,
                params,
                &el_field_v,
                &zero_shell,
                &zero_shell,
                b,
                c,
            )?;
            for i in 0..n {
                for j in 0..n {
                    out.m_vv[(i, j)] += w * (h0[(i, j)] + cn[(i, j)]);
                    out.m_vv_cache_motion[(i, j)] += w * (h0_cnv[(i, j)] - h0_cn0[(i, j)]);
                    out.s_vv[(i, j)] += w * sbc[(i, j)];
                    out.f_scc_vv[(i, j)] += w * scc[(i, j)];
                    out.f_scc_vv_skeleton[(i, j)] += w * scc_skel[(i, j)];
                    out.f_scc_vv_cache_motion[(i, j)] += w * scc_motion[(i, j)];
                }
            }
        }
    }
    Ok(out)
}

/// `r3[v] = Σ_abc v_a v_b v_c · D_c(cphf.hessian_response_ab)` — the direction-contracted
/// specialization of [`crate::third_derivative::closed_form_response_hessian_derivative`].
///
/// `cphf` must be the result of `solve_nonpbc_cpxtb_hessian_response` for the SAME
/// `(system, params, electronic, ao_opts)` that produced `electronic` — exactly as for the reference.
/// The per-`c` bundles inside `cphf` are read only through their `v`-contractions (see the module
/// docs for why that is exact).
#[allow(clippy::too_many_arguments)]
pub fn directional_response_third(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &crate::cphf::GammaCartesianCpxtbResult,
    ao_opts: crate::cphf::AoDerivativeOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    use crate::linalg::Matrix as M;
    let basis = &electronic.basis;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let n = basis.len();
    let nshell = basis.shells.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "directional_response_third: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let cutoff = coordination_cutoff;
    let mos = cphf.mos.clone();
    let occ = &electronic.occupations;
    let eps = &cphf.orbital_energies;
    let s_mat = &electronic.integrals.overlap;
    let p_mat = &electronic.density;
    let v_ref = &electronic.shell_scc_potential;
    let space = crate::cphf::CpxtbSpace::from_occupations(occ)?;
    let npair = space.pairs.len();
    let c_analytic = crate::cphf::mo_coefficient_derivatives(system, params, electronic, cphf)?;
    let cand = crate::cphf::relaxed_fock_derivative_candidates(system, params, electronic, cphf)?;

    // ===== Degenerate ε-block structure (verbatim from the reference) =====
    // For orbitals inside a degenerate block the per-orbital ε^{(c)}_p is gauge-dependent; the
    // gauge-INVARIANT object is the in-block matrix Λ^c_pq = F̃^c_pq − ε S̃^c_pq. The block partition
    // is a property of the REFERENCE state (ε, occupations), so it is direction-independent and
    // carries over unchanged; only Λ itself is contracted (Λ^v = Σ_c v_c Λ^c).
    let occ_flag: Vec<bool> = occ.iter().map(|&o| o > 1.0e-8).collect();
    let block_members: Vec<Vec<usize>> = {
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for p in 0..n {
            let start_new = match blocks.last() {
                Some(block) => {
                    let q = *block.last().unwrap();
                    (eps[p] - eps[q]).abs() >= 1.0e-6 || occ_flag[p] != occ_flag[q]
                }
                None => true,
            };
            if start_new {
                blocks.push(vec![p]);
            } else {
                blocks.last_mut().unwrap().push(p);
            }
        }
        let mut per_orbital = vec![Vec::new(); n];
        for block in &blocks {
            for &p in block {
                per_orbital[p] = block.clone();
            }
        }
        per_orbital
    };
    let pair_of: Vec<usize> = {
        let mut map = vec![usize::MAX; n * n];
        for (idx, &(i, a)) in space.pairs.iter().enumerate() {
            map[i * n + a] = idx;
        }
        map
    };

    let shell_kernel = crate::cphf::response_shell_scc_kernel(system, params, electronic)?;
    // ∂K/∂q chain data (verbatim from the reference): K = γ + 2Γ_A q_A is charge-dependent, so
    // D_c(K·u)|_u = (∂γ/∂R_c)·u + K·(D_c u) + 2Γ_A q_A^{(c)} (Σ_{t∈A} u_t). Only the LAST piece
    // carries `c` through q^{(c)}; it contracts to q^v.
    let shell_model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let charge_order = electronic.charge_order.max(3);
    let shell_atom: Vec<usize> = {
        let mut map = vec![0usize; nshell];
        for atom in 0..nat {
            let offset = shell_model.atom_offsets[atom];
            for local in 0..shell_model.atom_shell_counts[atom] {
                map[offset + local] = atom;
            }
        }
        map
    };
    let kernel_q_atom: Vec<f64> = (0..nat)
        .map(|atom| {
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
            third
        })
        .collect();

    let ref_ctx = crate::cphf::ResponseGradientContext::new(
        system,
        basis,
        params,
        electronic,
        cutoff,
        ao_opts.include_cn_h0,
    )?;
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let scale_occ: Vec<f64> = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occ[i] - occ[a]))
        .collect();
    let q_trans = crate::cphf::transition_shell_charges(basis, &mos, occ, s_mat)?;
    let sc = s_mat.matmul(&mos)?; // S·C (reference)

    // ===== shared reference-state closures (identical to the reference) =====
    let motrans = |m: &M, u: &M| -> M { u.transpose().matmul(&m.matmul(u).unwrap()).unwrap() };
    let population = |dens: &M, ov: &M| -> Vec<f64> {
        let mut out = vec![0.0_f64; nshell];
        for nu in 0..n {
            let mut a = 0.0;
            for k in 0..n {
                a += dens[(nu, k)] * ov[(k, nu)];
            }
            out[basis.aos[nu].shell_index] -= a;
        }
        out
    };
    let kvec = |u: &[f64]| -> Vec<f64> {
        (0..nshell)
            .map(|s| {
                (0..nshell)
                    .map(|t| shell_kernel[(s, t)] * u[t])
                    .sum::<f64>()
            })
            .collect()
    };

    // ===== direction-contracted FIRST-order objects (all linear in `c`) =====
    // F_v = Σ_c v_c h0_deriv[c] ; S_v = Σ_c v_c overlap_deriv[c]
    let f_v = accum_dir(n, v, |c| &cphf.derivative_matrices[c].h0_deriv);
    let s_v = accum_dir(n, v, |c| &cphf.derivative_matrices[c].overlap_deriv);
    // C^v = Σ_c v_c C^(c) (the MO-coefficient directional derivative)
    let cc_v = accum_dir(n, v, |c| &c_analytic[c]);
    // Λ^v = Σ_c v_c Λ^c, built from the same three candidate blocks the reference uses.
    let lam_v = {
        let h0_mo_v = accum_dir(n, v, |c| &cand[c].0);
        let resp_mo_v = accum_dir(n, v, |c| &cand[c].1);
        let s_tilde_v = accum_dir(n, v, |c| &cand[c].2);
        move |p: usize, q: usize| -> f64 {
            h0_mo_v[(p, q)] + resp_mo_v[(p, q)] - 0.5 * (eps[p] + eps[q]) * s_tilde_v[(p, q)]
        }
    };
    // P^v: only P^(c) and q^(c) are read by the reference slab body.
    let p_v = accum_dir(n, v, |c| &cphf.density_responses[c]);
    let q_v: Vec<f64> = {
        let mut out = vec![0.0_f64; nshell];
        for c in 0..ndof {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            let q_c = &cphf.shell_charge_responses[c];
            for s in 0..nshell {
                out[s] += vc * q_c[s];
            }
        }
        out
    };
    // V^v = Σ_c v_c (∂V/∂R_c|_q + K q^(c)) = (∂V/∂R·v)|_q + K q^v.
    let pot_v: Vec<f64> = {
        let kq = kvec(&q_v);
        (0..nshell)
            .map(|s| {
                let geo: f64 = (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum();
                geo + kq[s]
            })
            .collect()
    };
    // Atomic charge response along v, and the directional onsite ∂K/∂q chain action.
    let qat_v: Vec<f64> = {
        let mut out = vec![0.0_f64; nat];
        for s in 0..nshell {
            out[shell_atom[s]] += q_v[s];
        }
        out
    };
    let dk_chain_v = |u: &[f64]| -> Vec<f64> {
        let mut atom_sum = vec![0.0_f64; nat];
        for s in 0..nshell {
            atom_sum[shell_atom[s]] += u[s];
        }
        (0..nshell)
            .map(|s| {
                let atom = shell_atom[s];
                kernel_q_atom[atom] * qat_v[atom] * atom_sum[atom]
            })
            .collect()
    };
    // `Σ_c v_c (∂V[u]/∂R_c)` — the directional geometric derivative of the scalar potential of an
    // arbitrary (already b-contracted) shell-charge vector `u`.
    let dvdr_dir = |u: &[f64]| -> Result<Vec<f64>> {
        let d =
            crate::hessian::shell_scalar_potential_first_derivatives(system, basis, u, params)?;
        Ok((0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * d[(s, c)]).sum::<f64>())
            .collect())
    };
    // dSC^v = S_v·C + S·C^v (transition-charge derivative)
    let dsc_v = {
        let a = s_v.matmul(&mos)?;
        let b = s_mat.matmul(&cc_v)?;
        let mut m = a;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += b[(i, j)];
            }
        }
        m
    };
    // Directional CP amplitudes x^v = Σ_b v_b x_b.
    let x_v: Vec<f64> = {
        let mut out = vec![0.0_f64; npair];
        for b in 0..ndof {
            let vb = v[b];
            if vb == 0.0 {
                continue;
            }
            let x_b = &cphf.solutions[b].amplitudes;
            for p in 0..npair {
                out[p] += vb * x_b[p];
            }
        }
        out
    };

    // ===== direction-contracted SKELETON SECOND derivatives (bilinear; O(ndof²) block builds) =====
    // M_vv = Σ_xy v_x v_y [h0_bare_second(x,y) + cn_block(x,y)]. Used in BOTH roles the reference
    // needs it for — the `(a, c)` pair inside `bundle_grad` Group A and the `(b, c)` pair inside
    // `f_bc` — because both contract the two legs with the same `v`.
    let mut m_vv = M::zeros(n, n);
    for x in 0..ndof {
        if v[x] == 0.0 {
            continue;
        }
        for y in 0..ndof {
            let w = v[x] * v[y];
            if w == 0.0 {
                continue;
            }
            let h0 =
                crate::hessian::h0_bare_second_derivative_matrix(system, params, electronic, x, y)?;
            let cn = crate::hessian::h0_cn_block_second_derivative_matrix(
                system, params, electronic, cutoff, x, y,
            )?;
            for i in 0..n {
                for j in 0..n {
                    m_vv[(i, j)] += w * (h0[(i, j)] + cn[(i, j)]);
                }
            }
        }
    }
    // S_vv = Σ_bc v_b v_c ∂²S/∂R_b∂R_c ; F_scc_vv = Σ_bc v_b v_c ∂²F_scc/∂R_b∂R_c.
    // The SCC block's `c` leg is BOTH geometric and response-carrying (`V^(c)`, `q^(c)`), and the
    // two enter additively, so it must be summed with the PER-c bundle — not with V^v/q^v against a
    // second Σ_c (which would double-count the direction).
    let mut s_vv = M::zeros(n, n);
    let mut f_scc_vv = M::zeros(n, n);
    for c in 0..ndof {
        if v[c] == 0.0 {
            continue;
        }
        let q_c = &cphf.shell_charge_responses[c];
        let pot_c: Vec<f64> = {
            let kq = kvec(q_c);
            (0..nshell).map(|s| dvdr_q[(s, c)] + kq[s]).collect()
        };
        for b in 0..ndof {
            let w = v[b] * v[c];
            if w == 0.0 {
                continue;
            }
            let sbc = crate::cphf::overlap_second_derivative_matrix(system, basis, b, c)?;
            let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                system, params, electronic, &pot_c, q_c, b, c,
            )?;
            for i in 0..n {
                for j in 0..n {
                    s_vv[(i, j)] += w * sbc[(i, j)];
                    f_scc_vv[(i, j)] += w * scc[(i, j)];
                }
            }
        }
    }
    let f_vv = {
        let mut m = m_vv.clone();
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += f_scc_vv[(i, j)];
            }
        }
        m
    };
    // Directional scalar-potential kernel of Group A:
    //   Σ_ac v_a v_c [∂²V/∂R_a∂R_c|_q + ∂V[q^(c)]/∂R_a] = (∂²V·vv)|_q + (∂V[q^v]/∂R)·v.
    let dv_kern_vv: Vec<f64> = {
        let dvdr_qv =
            crate::hessian::shell_scalar_potential_first_derivatives(system, basis, &q_v, params)?;
        (0..nshell)
            .map(|s| {
                let mut acc = 0.0;
                for a in 0..ndof {
                    let va = v[a];
                    if va == 0.0 {
                        continue;
                    }
                    let mut inner = 0.0;
                    for c in 0..ndof {
                        inner += v[c] * d2vdr_q[s][(a, c)];
                    }
                    acc += va * (inner + dvdr_qv[(s, a)]);
                }
                acc
            })
            .collect()
    };

    // ===== the single Z-vector solve: y^v = A⁻¹ L^v, L^v = Σ_a v_a L_a =====
    // The adjoint operator A is `a`-independent, so the ndof solves of the reference collapse to one.
    let l_vectors = crate::cphf::density_gradient_adjoint_vectors(
        system, params, electronic, ao_opts, &mos, eps,
    )?;
    let l_v: Vec<f64> = {
        let mut out = vec![0.0_f64; npair];
        for a in 0..ndof {
            let va = v[a];
            if va == 0.0 {
                continue;
            }
            for p in 0..npair {
                out[p] += va * l_vectors[a][p];
            }
        }
        out
    };
    let setup = crate::cphf::build_cpxtb_setup(system, params, electronic, ao_opts, Some(&mos))?;
    let y_v = setup.solve_adjoint(&l_v, 1.0e-11, 4000)?.amplitudes;

    // ===== direction-contracted closures that used to be per-`c` =====
    let d_motrans = |m: &M, m_c: &M| -> M {
        let t1 = cc_v.transpose().matmul(&m.matmul(&mos).unwrap()).unwrap();
        let t2 = motrans(m_c, &mos);
        let t3 = mos.transpose().matmul(&m.matmul(&cc_v).unwrap()).unwrap();
        let mut r = t1;
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] += t2[(i, j)] + t3[(i, j)];
            }
        }
        r
    };
    let triple = |coeff: &M, dcoeff: &M| -> M {
        let a1 = cc_v
            .matmul(&coeff.matmul(&mos.transpose()).unwrap())
            .unwrap();
        let a2 = mos
            .matmul(&dcoeff.matmul(&mos.transpose()).unwrap())
            .unwrap();
        let a3 = mos
            .matmul(&coeff.matmul(&cc_v.transpose()).unwrap())
            .unwrap();
        let mut m = a1;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += a2[(i, j)] + a3[(i, j)];
            }
        }
        m
    };

    // Directional bundle-gradient: `Σ_ac v_a v_c { (D_c G_a)[bundle] + G_a[D_c bundle] }`. The
    // bundle arguments are already `b`-contracted, the derivative-bundle arguments already
    // `(b, c)`-contracted; the remaining `a` and `c` legs are folded in here.
    let bundle_grad_dir = |dp: &M,
                           dw: &M,
                           dq: &[f64],
                           d_dp: &M,
                           d_dw: &M,
                           d_dq: &[f64]|
     -> Result<f64> {
        let mut acc = 0.0;
        let sp_resp = kvec(dq);
        let dk_dq_v = dvdr_dir(dq)?;
        let chain_dq = dk_chain_v(dq);
        // Group A: H0+CN reuse + scc_kernel reuse (both legs pre-contracted).
        {
            let mut band = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    band += dp[(mu, nu)] * m_vv[(mu, nu)];
                }
            }
            let mut kern = 0.0;
            for s in 0..nshell {
                kern += dq[s] * dv_kern_vv[s];
            }
            acc += band + kern;
        }
        // Group A: pulay + scc_overlap (ao-pair loop). The `c` leg enters only through the
        // overlap second derivative (via atom_c/axis_c) and through V^(c)/P^(c)/q^(c) in `dcf`;
        // both are contracted below.
        for mu in 0..n {
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
                if (rmu - rnu).norm2() <= 1.0e-18 {
                    continue;
                }
                let pair = crate::integrals::contracted_pair_with_second_derivatives(
                    &basis.aos[mu],
                    &basis.aos[nu],
                    rmu,
                    rnu,
                );
                let dw_val = dw[(mu, nu)];
                let scalar_shift = v_ref[shell_mu] + v_ref[shell_nu];
                let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                let dp_val = dp[(mu, nu)];
                let p0_val = p_mat[(mu, nu)];
                let f_scc = -(dp_val * scalar_shift + p0_val * scalar_response);
                let dcf = -(dp_val * (pot_v[shell_mu] + pot_v[shell_nu])
                    + p_v[(mu, nu)] * scalar_response
                    + p0_val
                        * (dk_dq_v[shell_mu]
                            + dk_dq_v[shell_nu]
                            + chain_dq[shell_mu]
                            + chain_dq[shell_nu]));
                let dbra0 = pair.d_bra[0].to_array();
                let dket0 = pair.d_ket[0].to_array();
                for alpha in 0..3 {
                    // db_v/dk_v = Σ_c v_c (the reference's db/dk). Only DOFs on the two pair
                    // centres contribute, so the sum over `c` reduces to the two 3-vectors below.
                    let mut db = 0.0;
                    let mut dk = 0.0;
                    for beta in 0..3 {
                        let v_mu = v[3 * atom_mu + beta];
                        let v_nu = v[3 * atom_nu + beta];
                        db += v_mu * pair.h_bra_bra[0][alpha][beta]
                            + v_nu * pair.h_bra_ket[0][alpha][beta];
                        dk += v_mu * pair.h_bra_ket[0][beta][alpha]
                            + v_nu * pair.h_ket_ket[0][alpha][beta];
                    }
                    acc += v[3 * atom_mu + alpha]
                        * (db * (-2.0 * dw_val) + db * f_scc + dbra0[alpha] * dcf);
                    acc += v[3 * atom_nu + alpha]
                        * (dk * (-2.0 * dw_val) + dk * f_scc + dket0[alpha] * dcf);
                }
            }
        }
        // Group B: G_a[D_c bundle] — linear in the bundle, so one call on the vv-bundle suffices.
        let gb = crate::cphf::response_electronic_gradient(
            system,
            electronic,
            &shell_kernel,
            &ref_ctx,
            d_dp,
            d_dp,
            d_dw,
            d_dq,
        )?;
        for at in 0..nat {
            acc += v[3 * at] * gb[at].x + v[3 * at + 1] * gb[at].y + v[3 * at + 2] * gb[at].z;
        }
        Ok(acc)
    };

    // ===== the (collapsed) `b` body =====
    let s_tilde_b = motrans(&s_v, &mos);
    let f_tilde_b = motrans(&f_v, &mos);
    let d_s_tilde = d_motrans(&s_v, &s_vv);
    let d_f_tilde = d_motrans(&f_v, &f_vv);
    let zero = M::zeros(n, n);

    // ----- ORBITAL bundle B x^v and its directional derivative -----
    let ob = crate::cphf::orbital_response_bundle_from_amplitudes(
        basis,
        s_mat,
        p_mat,
        &mos,
        occ,
        eps,
        &space,
        &shell_kernel,
        &x_v,
    )?;
    let (dp_o, dw_o, dq_o) = (
        ob.density.clone(),
        ob.weighted.clone(),
        ob.shell_charges.clone(),
    );
    let mut coeff_po = M::zeros(n, n);
    let mut coeff_w1o = M::zeros(n, n);
    for (pi, &(i, a)) in space.pairs.iter().enumerate() {
        let w = (occ[i] - occ[a]) * x_v[pi];
        coeff_po[(a, i)] += w;
        coeff_po[(i, a)] += w;
        let w1 = w * eps[i];
        coeff_w1o[(a, i)] += w1;
        coeff_w1o[(i, a)] += w1;
    }
    let sp_o = kvec(&dq_o);
    let rf_o = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp_o)?;
    let rf_mo_o = motrans(&rf_o, &mos);
    let mut coeff_w2o = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            coeff_w2o[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo_o[(i, j)];
        }
    }
    let d_dp_o = triple(&coeff_po, &zero);
    let d_dq_o = {
        let a = population(&d_dp_o, s_mat);
        let b2 = population(&dp_o, &s_v);
        (0..nshell).map(|s| a[s] + b2[s]).collect::<Vec<f64>>()
    };
    let d_sp_o: Vec<f64> = {
        let dk_dqo = dvdr_dir(&dq_o)?;
        let kdq = kvec(&d_dq_o);
        let chain = dk_chain_v(&dq_o);
        (0..nshell)
            .map(|s| dk_dqo[s] + kdq[s] + chain[s])
            .collect()
    };
    let d_rf_o = {
        let t1 = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d_sp_o)?;
        let mut m = t1;
        for mu in 0..n {
            let smu = sp_o[basis.aos[mu].shell_index];
            for nu in 0..n {
                let snu = sp_o[basis.aos[nu].shell_index];
                m[(mu, nu)] += -0.5 * (smu + snu) * s_v[(mu, nu)];
            }
        }
        m
    };
    let d_rf_mo_o = d_motrans(&rf_o, &d_rf_o);
    let mut dcoeff_w1o = M::zeros(n, n);
    let mut dcoeff_w2o = M::zeros(n, n);
    for &(i, a) in space.pairs.iter() {
        // Λ-covariant ε^{(c)}·x contraction (degenerate-block safe), now with Λ^v.
        let mut e_x = 0.0;
        for &i2 in &block_members[i] {
            let p2 = pair_of[i2 * n + a];
            if p2 != usize::MAX {
                e_x += lam_v(i, i2) * x_v[p2];
            }
        }
        let dw1 = (occ[i] - occ[a]) * e_x;
        dcoeff_w1o[(a, i)] += dw1;
        dcoeff_w1o[(i, a)] += dw1;
    }
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            dcoeff_w2o[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo_o[(i, j)];
        }
    }
    let d_dw_o = {
        let a = triple(&coeff_w1o, &dcoeff_w1o);
        let b2 = triple(&coeff_w2o, &dcoeff_w2o);
        let mut m = a;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += b2[(i, j)];
            }
        }
        m
    };
    let orb = bundle_grad_dir(&dp_o, &dw_o, &dq_o, &d_dp_o, &d_dw_o, &d_dq_o)?;

    // ----- STATIC bundle and its directional derivative -----
    let mut bmat = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            bmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * s_tilde_b[(i, j)];
        }
    }
    let dp_s = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &bmat)?;
    let dq_s =
        crate::cphf::response_shell_charges_from_density(basis, s_mat, p_mat, &dp_s, &s_v)?;
    let sp_s = kvec(&dq_s);
    let rf_s = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp_s)?;
    let rf_mo_s = motrans(&rf_s, &mos);
    let mut cwa = M::zeros(n, n);
    let mut cwb = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            cwa[(i, j)] = 0.5
                * (occ[i] + occ[j])
                * (f_tilde_b[(i, j)] - (eps[i] + eps[j]) * s_tilde_b[(i, j)]);
            cwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo_s[(i, j)];
        }
    }
    let dw_s = {
        let a = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwa)?;
        let b2 = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwb)?;
        let mut m = a;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += b2[(i, j)];
            }
        }
        m
    };
    let mut dbmat = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            dbmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * d_s_tilde[(i, j)];
        }
    }
    let d_dp_s = triple(&bmat, &dbmat);
    let d_dq_s = {
        // D(pop(dp_s, S) + pop(P, S_b)) with both legs contracted: `pop(dp_s, S_c)` and
        // `pop(P^(c), S_b)` are b×c products, so both collapse onto S_v.
        let a = population(&d_dp_s, s_mat);
        let b2 = population(&dp_s, &s_v);
        let d = population(&p_v, &s_v);
        let e = population(p_mat, &s_vv);
        (0..nshell)
            .map(|s| a[s] + b2[s] + d[s] + e[s])
            .collect::<Vec<f64>>()
    };
    let d_sp_s: Vec<f64> = {
        let dk_dqs = dvdr_dir(&dq_s)?;
        let kdq = kvec(&d_dq_s);
        let chain = dk_chain_v(&dq_s);
        (0..nshell)
            .map(|s| dk_dqs[s] + kdq[s] + chain[s])
            .collect()
    };
    let d_rf_s = {
        let t1 = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d_sp_s)?;
        let mut m = t1;
        for mu in 0..n {
            let smu = sp_s[basis.aos[mu].shell_index];
            for nu in 0..n {
                let snu = sp_s[basis.aos[nu].shell_index];
                m[(mu, nu)] += -0.5 * (smu + snu) * s_v[(mu, nu)];
            }
        }
        m
    };
    let d_rf_mo_s = d_motrans(&rf_s, &d_rf_s);
    let mut dcwa = M::zeros(n, n);
    let mut dcwb = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            // Λ-covariant (ε^{(c)}_i + ε^{(c)}_j)·S̃_b contraction with Λ^v and S̃_v.
            let mut e_s = 0.0;
            for &k in &block_members[i] {
                e_s += lam_v(i, k) * s_tilde_b[(k, j)];
            }
            for &k in &block_members[j] {
                e_s += s_tilde_b[(i, k)] * lam_v(k, j);
            }
            dcwa[(i, j)] = 0.5
                * (occ[i] + occ[j])
                * (d_f_tilde[(i, j)] - e_s - (eps[i] + eps[j]) * d_s_tilde[(i, j)]);
            dcwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo_s[(i, j)];
        }
    }
    let d_dw_s = {
        let a = triple(&cwa, &dcwa);
        let b2 = triple(&cwb, &dcwb);
        let mut m = a;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += b2[(i, j)];
            }
        }
        m
    };
    let stat = bundle_grad_dir(&dp_s, &dw_s, &dq_s, &d_dp_s, &d_dw_s, &d_dq_s)?;

    // ----- D_c rhs_b (non-metric + metric-SCC), directionally contracted -----
    let d_rf_mo_b_oo = &d_rf_mo_s; // metric RF_b derivative = static RF_b derivative
    let mut d_rhs = vec![0.0_f64; npair];
    for (idx, &(i, a)) in space.pairs.iter().enumerate() {
        let mut e_s = 0.0;
        for &i2 in &block_members[i] {
            e_s += lam_v(i, i2) * s_tilde_b[(i2, a)];
        }
        let drhs0 = -d_f_tilde[(i, a)] + e_s + eps[i] * d_s_tilde[(i, a)];
        let dmetric = -d_rf_mo_b_oo[(i, a)];
        d_rhs[idx] = drhs0 + dmetric;
    }
    // ----- (D_c A) x_b, directionally contracted -----
    let mut d_axb = vec![0.0_f64; npair];
    {
        let dqt: Vec<Vec<f64>> = space
            .pairs
            .iter()
            .map(|&(i, a)| {
                let mut q = vec![0.0_f64; nshell];
                for (sh, shell) in basis.shells.iter().enumerate() {
                    let end = shell.first_ao + shell.nao;
                    for mu in shell.first_ao..end {
                        q[sh] -= cc_v[(mu, a)] * sc[(mu, i)]
                            + mos[(mu, a)] * dsc_v[(mu, i)]
                            + cc_v[(mu, i)] * sc[(mu, a)]
                            + mos[(mu, i)] * dsc_v[(mu, a)];
                    }
                }
                q
            })
            .collect();
        let mut g = vec![0.0_f64; nshell];
        let mut dg = vec![0.0_f64; nshell];
        for p in 0..npair {
            for s in 0..nshell {
                g[s] += q_trans[p][s] * scale_occ[p] * x_v[p];
                dg[s] += dqt[p][s] * scale_occ[p] * x_v[p];
            }
        }
        let pot = kvec(&g);
        let dk_g = dvdr_dir(&g)?;
        let k_dg = kvec(&dg);
        let chain_g = dk_chain_v(&g);
        let dpot: Vec<f64> = (0..nshell)
            .map(|s| dk_g[s] + k_dg[s] + chain_g[s])
            .collect();
        for (p, &(i, a)) in space.pairs.iter().enumerate() {
            // Λ-covariant gap derivative [x Λ^v_vv − Λ^v_oo x]_(i,a).
            let mut val = 0.0;
            for &a2 in &block_members[a] {
                let p2 = pair_of[i * n + a2];
                if p2 != usize::MAX {
                    val += lam_v(a, a2) * x_v[p2];
                }
            }
            for &i2 in &block_members[i] {
                let p2 = pair_of[i2 * n + a];
                if p2 != usize::MAX {
                    val -= lam_v(i, i2) * x_v[p2];
                }
            }
            for s in 0..nshell {
                val += dqt[p][s] * pot[s] + q_trans[p][s] * dpot[s];
            }
            d_axb[p] = val;
        }
    }
    // ===== assemble: three scalars =====
    let mut zterm = 0.0;
    for p in 0..npair {
        zterm += y_v[p] * (d_rhs[p] - d_axb[p]);
    }
    Ok(stat + orb + zterm)
}

// =====================================================================================
//   QUARTIC STAGE — the total `λ`-derivative of `directional_response_third`
// =====================================================================================

/// `dst += w · src` for equally-shaped dense matrices.
fn axpy(dst: &mut Matrix, src: &Matrix, w: f64) {
    for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
        *d += w * *s;
    }
}

/// `Cᵀ m C`.
fn mo_rep(mos: &Matrix, m: &Matrix) -> Result<Matrix> {
    mos.transpose().matmul(&m.matmul(mos)?)
}

/// `D_λ(Cᵀ m C)` with `Ċ = cd`, `ṁ = m1`.
fn d_mo_rep(mos: &Matrix, cd: &Matrix, m0: &Matrix, m1: &Matrix) -> Result<Matrix> {
    let mut out = cd.transpose().matmul(&m0.matmul(mos)?)?;
    let t2 = mos.transpose().matmul(&m1.matmul(mos)?)?;
    let t3 = mos.transpose().matmul(&m0.matmul(cd)?)?;
    axpy(&mut out, &t2, 1.0);
    axpy(&mut out, &t3, 1.0);
    Ok(out)
}

/// `D²_λ(Cᵀ m C)` with `Ċ = cd`, `C̈ = cdd`, `ṁ = m1`, `m̈ = m2` — the six Leibniz terms
/// `C̈ᵀm C + Cᵀm C̈ + 2ĊᵀmĊ + 2Ċᵀm₁C + 2Cᵀm₁Ċ + Cᵀm₂C`.
fn d2_mo_rep(
    mos: &Matrix,
    cd: &Matrix,
    cdd: &Matrix,
    m0: &Matrix,
    m1: &Matrix,
    m2: &Matrix,
) -> Result<Matrix> {
    let mut out = cdd.transpose().matmul(&m0.matmul(mos)?)?;
    axpy(&mut out, &mos.transpose().matmul(&m0.matmul(cdd)?)?, 1.0);
    axpy(&mut out, &cd.transpose().matmul(&m0.matmul(cd)?)?, 2.0);
    axpy(&mut out, &cd.transpose().matmul(&m1.matmul(mos)?)?, 2.0);
    axpy(&mut out, &mos.transpose().matmul(&m1.matmul(cd)?)?, 2.0);
    axpy(&mut out, &mos.transpose().matmul(&m2.matmul(mos)?)?, 1.0);
    Ok(out)
}

/// `D_λ(C c Cᵀ)` with `Ċ = cd`, `ċ = c1` (the reference's `triple`).
fn d_ao_rep(mos: &Matrix, cd: &Matrix, c0: &Matrix, c1: &Matrix) -> Result<Matrix> {
    let mut out = cd.matmul(&c0.matmul(&mos.transpose())?)?;
    axpy(&mut out, &mos.matmul(&c1.matmul(&mos.transpose())?)?, 1.0);
    axpy(&mut out, &mos.matmul(&c0.matmul(&cd.transpose())?)?, 1.0);
    Ok(out)
}

/// `D²_λ(C c Cᵀ)` — the six-term sibling of [`d_ao_rep`].
fn d2_ao_rep(
    mos: &Matrix,
    cd: &Matrix,
    cdd: &Matrix,
    c0: &Matrix,
    c1: &Matrix,
    c2: &Matrix,
) -> Result<Matrix> {
    let mut out = cdd.matmul(&c0.matmul(&mos.transpose())?)?;
    axpy(&mut out, &mos.matmul(&c0.matmul(&cdd.transpose())?)?, 1.0);
    axpy(&mut out, &cd.matmul(&c0.matmul(&cd.transpose())?)?, 2.0);
    axpy(&mut out, &cd.matmul(&c1.matmul(&mos.transpose())?)?, 2.0);
    axpy(&mut out, &mos.matmul(&c1.matmul(&cd.transpose())?)?, 2.0);
    axpy(&mut out, &mos.matmul(&c2.matmul(&mos.transpose())?)?, 1.0);
    Ok(out)
}

/// A response bundle together with its first and second directional `λ`-derivatives.
struct BundleLadder {
    p0: Matrix,
    w0: Matrix,
    q0: Vec<f64>,
    p1: Matrix,
    w1: Matrix,
    q1: Vec<f64>,
    p2: Matrix,
    w2: Matrix,
    q2: Vec<f64>,
}

/// `r4[v] = Σ_abcd v_a v_b v_c v_d · D_d D_c(cphf.hessian_response_ab)` — the exact total
/// `λ`-derivative along `R + λv` of [`directional_response_third`], with the electronic reference
/// RECONVERGED (every cached field of `electronic` moves with the geometry).
///
/// # The differentiation map
///
/// Write the reference third derivative in operator form (see the module docs for the audit that
/// licenses the direction pre-contraction):
///
/// ```text
///   r3 = G₂[X] + G₁[X′] ,
///     G₁[·] = Σ_a v_a G_a[·]                     (response_electronic_gradient, "Group B")
///     G₂[·] = Σ_ac v_a v_c (D_c G_a)[·]          ("Group A" of `bundle_grad_dir`)
///     X     = X_static + B x^v                   (the b-contracted response bundle)
///     X′    = D_v X                              (its total directional derivative)
/// ```
///
/// Because `D_d D_c (G_a[X_b]) = (D_dD_cG_a)[X_b] + (D_cG_a)[D_dX_b] + (D_dG_a)[D_cX_b]
/// + G_a[D_dD_cX_b]`, the `vvvv` contraction collapses to the three-term master formula
///
/// ```text
///   r4 = G₃[X] + 2·G₂[X′] + G₁[X″] ,   G₃[·] = Σ_acd v_a v_c v_d (D_dD_cG_a)[·] .
/// ```
///
/// Each numbered group of the derivation below maps onto one block of this function:
///
/// **(1) first-order → second-order objects.** Every directional first-order object of the third
/// is promoted to its second-order counterpart: `S_v→S_vv→S_vvv`, `F_v→F_vv→F_vvv`, `P^v→P^vv`,
/// `q^v→q^vv`, `V^v→V^vv`, `C^v = C·U → C^vv = C(U·U + U̇)`, `Λ^v→Λ̇^v`. The screened `P^vv`/`q^vv`
/// come from the charge-space second-order solver
/// ([`crate::response::charge_space::ChargeSpaceContext::second_order_field`]); the MO-frame
/// objects (`U`, `U̇`, `ḣ̃`, `ṡ̃`, `ε^v`, `ε^vv`) are rebuilt HERE from the cphf MOs so that every
/// frame quantity is bit-consistent with what `directional_response_third` differentiates
/// (`U = CᵀS·C^v` inverts `C^v = C U` exactly).
///
/// **(2) skeleton seconds → skeleton thirds.** `F_vv = M_vv + F^scc_vv` becomes
/// `F_vvv = M_vvv + F^scc_vvv` with `M_vvv` from [`crate::hessian::directional_h0_bare_third_matrix`]
/// + [`crate::hessian::directional_h0_cn_block_third_matrix`] and `F^scc_vvv` from
/// [`crate::hessian::directional_h0_scc_scalar_third_matrix`] fed the promoted legs
/// `(V^v, V^vv, q^v, q^vv)`; `S_vv → S_vvv` from
/// [`crate::hessian::directional_overlap_third_matrix`].
///
/// Those four builders are, by construction and by their FD gates, the `λ`-derivatives of their
/// second-order twins at a **frozen** `electronic`. The reconverged reference moves three cached
/// fields the second-order twins read — `coordination_numbers` (inside `h0_bare_second`'s
/// `self_avg`), `shell_scc_potential` and `shell_charges` (inside `h0_scc_scalar_second`) — and
/// each enters those routines HOMOGENEOUSLY LINEARLY (affinely, for CN). Their motion is therefore
/// recovered exactly by re-running the same second-order builders on a **doctored** electronic
/// reference holding the fields' `λ`-derivatives (`CN→CN^v`, `V→V^v`, `q→q^v`) — the two
/// `..._cache_motion` accumulators below. Omitting them leaves an `h`-independent FD residual.
///
/// **(3) `∂K/∂q` chain.** `K = γ(R) + E″(q_A)` gives `K̇ = ∂γ/∂R·v + E‴·q^v_A` (the third's
/// `dvdr_dir + dk_chain_v`) and, one order up,
/// `K̈ = ∂²γ/∂R²:vv + E⁗·(q^v_A)² + E‴·q^vv_A` — the `E⁗` leg is why
/// [`crate::coulomb::onsite_charge_anharmonic_derivatives`] now also returns the fourth charge
/// derivative (identically zero for stock GFN1's `⅓Γq³`, non-zero from `charge_order ≥ 4`).
///
/// **(4) `dx^v`.** The third already assembles `A dx^v = d_rhs^v − d_axb^v`; one extra
/// `solve_adjoint` (A is self-adjoint) turns it into the amplitude derivative itself, which the
/// master formula needs inside `X′` and `X″`.
///
/// **(5) the Z-vector term.** `d²x^v` is never formed: `A d²x^v = D²_v rhs − (D²_vA)x^v
/// − 2(D_vA)dx^v`, so `G₁[B d²x^v] = y^v·[D²rhs − (D²A)x^v − 2(D_vA)dx^v]` with the SAME `y^v` the
/// third solves for. `(D_vA)·w` and `(D²_vA)·w` are the third's `d_axb` block re-read as functions
/// of an arbitrary amplitude vector.
///
/// **(6)/(7)** `d_rhs`/`d_axb` and the `bundle_grad` groups are promoted factor by factor with the
/// Leibniz rule; the `G₃` pair loop replaces the second-derivative overlap legs with the
/// third-order centre patterns of [`crate::integrals::contracted_pair_with_third_derivatives`].
///
/// Gated by `directional_response_fourth_matches_third_fd_along_v` (central FD of
/// `directional_response_third` with everything rebuilt at the displaced geometries, plus the `h²`
/// truncation-scaling assertion).
///
/// Integer occupations only (the charge-space second-order solve has no finite-temperature branch
/// yet); non-PBC only.
#[allow(clippy::too_many_arguments)]
pub fn directional_response_fourth(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &crate::cphf::GammaCartesianCpxtbResult,
    ao_opts: crate::cphf::AoDerivativeOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<f64> {
    use crate::linalg::Matrix as M;
    let basis = &electronic.basis;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let n = basis.len();
    let nshell = basis.shells.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "directional_response_fourth: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let cutoff = coordination_cutoff;
    let mos = cphf.mos.clone();
    let occ = &electronic.occupations;
    let eps = &cphf.orbital_energies;
    let s_mat = &electronic.integrals.overlap;
    let p_mat = &electronic.density;
    let v_ref = &electronic.shell_scc_potential;
    let space = crate::cphf::CpxtbSpace::from_occupations(occ)?;
    let npair = space.pairs.len();
    let c_analytic = {
        let _p = crate::profile::scope("fc4.stage5.mo_coefficient_derivatives");
        crate::cphf::mo_coefficient_derivatives(system, params, electronic, cphf)?
    };
    let cand = {
        let _p = crate::profile::scope("fc4.stage5.relaxed_fock_candidates");
        crate::cphf::relaxed_fock_derivative_candidates(system, params, electronic, cphf)?
    };
    let zero = M::zeros(n, n);

    // ---- degenerate ε-block structure (identical to the third) ----
    let occ_flag: Vec<bool> = occ.iter().map(|&o| o > 1.0e-8).collect();
    let block_members: Vec<Vec<usize>> = {
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for p in 0..n {
            let start_new = match blocks.last() {
                Some(block) => {
                    let q = *block.last().unwrap();
                    (eps[p] - eps[q]).abs() >= 1.0e-6 || occ_flag[p] != occ_flag[q]
                }
                None => true,
            };
            if start_new {
                blocks.push(vec![p]);
            } else {
                blocks.last_mut().unwrap().push(p);
            }
        }
        let mut per_orbital = vec![Vec::new(); n];
        for block in &blocks {
            for &p in block {
                per_orbital[p] = block.clone();
            }
        }
        per_orbital
    };
    let pair_of: Vec<usize> = {
        let mut map = vec![usize::MAX; n * n];
        for (idx, &(i, a)) in space.pairs.iter().enumerate() {
            map[i * n + a] = idx;
        }
        map
    };

    let shell_kernel = crate::cphf::response_shell_scc_kernel(system, params, electronic)?;
    let shell_model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let charge_order = electronic.charge_order.max(3);
    let shell_atom: Vec<usize> = {
        let mut map = vec![0usize; nshell];
        for atom in 0..nat {
            let offset = shell_model.atom_offsets[atom];
            for local in 0..shell_model.atom_shell_counts[atom] {
                map[offset + local] = atom;
            }
        }
        map
    };
    // Group (3): the onsite charge kernels E‴ (chain) and E⁗ (its λ-derivative leg).
    let (kernel_q_atom, kernel_q4_atom): (Vec<f64>, Vec<f64>) = {
        let mut third = vec![0.0_f64; nat];
        let mut fourth = vec![0.0_f64; nat];
        for atom in 0..nat {
            if shell_model.atom_shell_counts[atom] == 0 {
                continue;
            }
            let offset = shell_model.atom_offsets[atom];
            let (_, _, e3, e4) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                shell_model.hardness[offset],
                shell_model.hubbard_derivs[offset],
                charge_order,
                electronic.atomic_charges[atom],
            );
            third[atom] = e3;
            fourth[atom] = e4;
        }
        (third, fourth)
    };

    let ref_ctx = {
        let _p = crate::profile::scope("fc4.stage5.response_gradient_context");
        crate::cphf::ResponseGradientContext::new(
            system,
            basis,
            params,
            electronic,
            cutoff,
            ao_opts.include_cn_h0,
        )?
    };
    let _pot = crate::profile::scope("fc4.stage5.potential_ladders");
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    drop(_pot);
    let scale_occ: Vec<f64> = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occ[i] - occ[a]))
        .collect();
    let q_trans = crate::cphf::transition_shell_charges(basis, &mos, occ, s_mat)?;
    let sc = s_mat.matmul(&mos)?;

    let population = |dens: &M, ov: &M| -> Vec<f64> {
        let mut out = vec![0.0_f64; nshell];
        for nu in 0..n {
            let mut a = 0.0;
            for k in 0..n {
                a += dens[(nu, k)] * ov[(k, nu)];
            }
            out[basis.aos[nu].shell_index] -= a;
        }
        out
    };
    let kvec = |u: &[f64]| -> Vec<f64> {
        (0..nshell)
            .map(|s| {
                (0..nshell)
                    .map(|t| shell_kernel[(s, t)] * u[t])
                    .sum::<f64>()
            })
            .collect()
    };
    let dvdr_dir = |u: &[f64]| -> Result<Vec<f64>> {
        let d =
            crate::hessian::shell_scalar_potential_first_derivatives(system, basis, u, params)?;
        Ok((0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * d[(s, c)]).sum::<f64>())
            .collect())
    };
    let d2vdr_dir = |u: &[f64]| -> Result<Vec<f64>> {
        let d =
            crate::hessian::shell_scalar_potential_second_derivatives(system, basis, u, params)?;
        Ok((0..nshell)
            .map(|s| {
                let mut acc = 0.0;
                for b in 0..ndof {
                    if v[b] == 0.0 {
                        continue;
                    }
                    for c in 0..ndof {
                        acc += v[b] * v[c] * d[s][(b, c)];
                    }
                }
                acc
            })
            .collect())
    };

    // ===== group (1): direction-contracted FIRST-order objects (verbatim from the third) =====
    let f_v = accum_dir(n, v, |c| &cphf.derivative_matrices[c].h0_deriv);
    let s_v = accum_dir(n, v, |c| &cphf.derivative_matrices[c].overlap_deriv);
    let cc_v = accum_dir(n, v, |c| &c_analytic[c]);
    let s_tilde_v = mo_rep(&mos, &s_v)?;
    let lam_v = {
        let h0_mo_v = accum_dir(n, v, |c| &cand[c].0);
        let resp_mo_v = accum_dir(n, v, |c| &cand[c].1);
        let mut m = M::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                m[(p, q)] = h0_mo_v[(p, q)] + resp_mo_v[(p, q)]
                    - 0.5 * (eps[p] + eps[q]) * s_tilde_v[(p, q)];
            }
        }
        m
    };
    let eps_v: Vec<f64> = (0..n).map(|p| lam_v[(p, p)]).collect();
    let p_v = accum_dir(n, v, |c| &cphf.density_responses[c]);
    let q_v: Vec<f64> = {
        let mut out = vec![0.0_f64; nshell];
        for c in 0..ndof {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for s in 0..nshell {
                out[s] += vc * cphf.shell_charge_responses[c][s];
            }
        }
        out
    };
    let pot_v: Vec<f64> = {
        let kq = kvec(&q_v);
        (0..nshell)
            .map(|s| {
                let geo: f64 = (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum();
                geo + kq[s]
            })
            .collect()
    };
    let qat_v: Vec<f64> = {
        let mut out = vec![0.0_f64; nat];
        for s in 0..nshell {
            out[shell_atom[s]] += q_v[s];
        }
        out
    };
    let x_v: Vec<f64> = {
        let mut out = vec![0.0_f64; npair];
        for b in 0..ndof {
            let vb = v[b];
            if vb == 0.0 {
                continue;
            }
            for p in 0..npair {
                out[p] += vb * cphf.solutions[b].amplitudes[p];
            }
        }
        out
    };
    // The directional FIRST coordination response — needed for the cached-CN motion of `h0_bare²`.
    let cn_v: Vec<f64> = {
        let cn_grad = crate::hessian::cn_gradient_matrix(system, cutoff)?;
        cn_grad
            .iter()
            .map(|row| row.iter().zip(v).map(|(g, vc)| g * vc).sum())
            .collect()
    };

    // ===== group (2): skeleton SECOND derivatives + their cached-reference motion =====
    // ONE PASS per matrix: the per-AO-pair second-derivative data is `(b, c)`-independent, so both
    // legs are contracted against `v` inside the pair sweep instead of summing `ndof²` per-`(b,c)`
    // block builds. Gated element-wise against that double loop (kept as
    // `skeleton_second_double_loop`) by `skeleton_second_one_pass_matches_double_loop`.
    let geo_v: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum())
        .collect();
    let SkeletonSecond {
        m_vv,
        m_vv_cache_motion,
        s_vv,
        f_scc_vv,
        f_scc_vv_skeleton,
        f_scc_vv_cache_motion,
    } = {
        let _p = crate::profile::scope("fc4.stage5.skeleton_second");
        skeleton_second_one_pass(
            system, params, electronic, cutoff, v, &geo_v, &pot_v, &q_v, &cn_v,
        )?
    };
    let f_vv = {
        let mut m = m_vv.clone();
        axpy(&mut m, &f_scc_vv, 1.0);
        m
    };
    // Skeleton (frozen-charge) second derivative — the charge-space solver's own convention.
    let f_vv_skeleton = {
        let mut m = m_vv.clone();
        axpy(&mut m, &f_scc_vv_skeleton, 1.0);
        m
    };
    let dv_kern_vv: Vec<f64> = {
        let dvdr_qv =
            crate::hessian::shell_scalar_potential_first_derivatives(system, basis, &q_v, params)?;
        (0..nshell)
            .map(|s| {
                let mut acc = 0.0;
                for a in 0..ndof {
                    let va = v[a];
                    if va == 0.0 {
                        continue;
                    }
                    let mut inner = 0.0;
                    for c in 0..ndof {
                        inner += v[c] * d2vdr_q[s][(a, c)];
                    }
                    acc += va * (inner + dvdr_qv[(s, a)]);
                }
                acc
            })
            .collect()
    };

    // ===== group (1) continued: the SCREENED second-order bundle `(P^vv, q^vv)` =====
    let _pcs = crate::profile::scope("fc4.stage5.charge_space_second_order");
    let cs = crate::response::charge_space::ChargeSpaceContext::build(system, params, electronic)?;
    let cs_field = cs.first_order_field(f_v.clone(), s_v.clone())?;
    let dgamma_v_qv: Vec<f64> = {
        let d = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &cs_field.bundle.shell_charges,
            params,
        )?;
        (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * d[(s, c)]).sum::<f64>())
            .collect()
    };
    let cs_second = cs.second_order_field(
        &cs_field,
        &cs_field,
        &f_vv_skeleton,
        &s_vv,
        &dgamma_v_qv,
        &dgamma_v_qv,
    )?;
    let p_vv = cs_second.bundle.density.clone();
    let q_vv = cs_second.bundle.shell_charges.clone();
    let qat_vv: Vec<f64> = {
        let mut out = vec![0.0_f64; nat];
        for s in 0..nshell {
            out[shell_atom[s]] += q_vv[s];
        }
        out
    };
    drop(_pcs);

    // ===== group (3): the ∂K/∂q chain and its λ-derivative =====
    let chain1 = |u: &[f64]| -> Vec<f64> {
        let mut atom_sum = vec![0.0_f64; nat];
        for s in 0..nshell {
            atom_sum[shell_atom[s]] += u[s];
        }
        (0..nshell)
            .map(|s| {
                let a = shell_atom[s];
                kernel_q_atom[a] * qat_v[a] * atom_sum[a]
            })
            .collect()
    };
    let chain2 = |u: &[f64]| -> Vec<f64> {
        let mut atom_sum = vec![0.0_f64; nat];
        for s in 0..nshell {
            atom_sum[shell_atom[s]] += u[s];
        }
        (0..nshell)
            .map(|s| {
                let a = shell_atom[s];
                (kernel_q4_atom[a] * qat_v[a] * qat_v[a] + kernel_q_atom[a] * qat_vv[a])
                    * atom_sum[a]
            })
            .collect()
    };
    // `K̇ u` and `K̈ u`.
    let kdot = |u: &[f64]| -> Result<Vec<f64>> {
        let g = dvdr_dir(u)?;
        let c = chain1(u);
        Ok((0..nshell).map(|s| g[s] + c[s]).collect())
    };
    let kddot = |u: &[f64]| -> Result<Vec<f64>> {
        let g = d2vdr_dir(u)?;
        let c = chain2(u);
        Ok((0..nshell).map(|s| g[s] + c[s]).collect())
    };
    // `V^vv = D²_λ(shell_scc_potential)`.
    let pot_vv: Vec<f64> = {
        let geo2 = d2vdr_dir(&electronic.shell_charges)?;
        let cross = dvdr_dir(&q_v)?;
        let chain = chain1(&q_v);
        let kq = kvec(&q_vv);
        (0..nshell)
            .map(|s| geo2[s] + 2.0 * cross[s] + chain[s] + kq[s])
            .collect()
    };

    // ===== group (2) continued: skeleton THIRD derivatives =====
    let _pskel3 = crate::profile::scope("fc4.stage5.skeleton_third");
    let s_vvv = crate::hessian::directional_overlap_third_matrix(system, basis, v)?;
    let m_vvv = {
        let mut m = crate::hessian::directional_h0_bare_third_matrix(system, params, electronic, v)?;
        let cn3 = crate::hessian::directional_h0_cn_block_third_matrix(
            system, params, electronic, cutoff, v,
        )?;
        axpy(&mut m, &cn3, 1.0);
        // The reconverged reference's cached-CN motion of `h0_bare²` (worth 4.8e-7 in the water
        // gate — an h-INDEPENDENT residual if dropped).
        axpy(&mut m, &m_vv_cache_motion, 1.0);
        m
    };
    let f_vvv = {
        let mut m = m_vvv.clone();
        let scc3 = crate::hessian::directional_h0_scc_scalar_third_matrix(
            system, params, electronic, v, &pot_v, &pot_vv, &q_v, &q_vv,
        )?;
        axpy(&mut m, &scc3, 1.0);
        // The reconverged reference's cached V/q motion of `h0_scc_scalar²` (worth 1.7e-6 in the
        // water gate — likewise an h-INDEPENDENT residual if dropped).
        axpy(&mut m, &f_scc_vv_cache_motion, 1.0);
        m
    };
    // The G₃ scalar-potential kernel: `Σ_acd v_a v_c v_d [∂³V|_q + 2 ∂²V[q^v] + ∂V[q^vv]]`.
    let dv_kern_vvv: Vec<f64> = {
        let d3 = crate::hessian::shell_scalar_potential_third_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )?;
        let d2qv =
            crate::hessian::shell_scalar_potential_second_derivatives(system, basis, &q_v, params)?;
        let d1qvv = crate::hessian::shell_scalar_potential_first_derivatives(
            system, basis, &q_vv, params,
        )?;
        (0..nshell)
            .map(|s| {
                let mut acc = 0.0;
                for a in 0..ndof {
                    let va = v[a];
                    if va == 0.0 {
                        continue;
                    }
                    let mut inner = d1qvv[(s, a)];
                    for c in 0..ndof {
                        let vc = v[c];
                        if vc == 0.0 {
                            continue;
                        }
                        let base = (a * ndof + c) * ndof;
                        let mut t = 0.0;
                        for (d, &vd) in v.iter().enumerate() {
                            t += vd * d3[s][base + d];
                        }
                        inner += vc * (t + 2.0 * d2qv[s][(a, c)]);
                    }
                    acc += va * inner;
                }
                acc
            })
            .collect()
    };

    drop(_pskel3);

    // ===== group (1) continued: the MO frame ladder (U, U̇, C^v, C^vv, ḣ̃, ṡ̃, ε^vv, Λ̇) =====
    // `C^v = C U` ⇒ `U = CᵀS C^v` exactly (CᵀSC = I) — this keeps `U` bit-consistent with the
    // cphf-derived `C^v` the third differentiates, whatever gauge the CP amplitudes fixed.
    let _pmo = crate::profile::scope("fc4.stage5.mo_frame_and_assembly");
    let u_v = mos.transpose().matmul(&s_mat.matmul(&cc_v)?)?;
    let d_s_tilde = d_mo_rep(&mos, &cc_v, &s_v, &s_vv)?;
    // Total (SCC-relaxed) Fock derivative in the AO basis and its λ-derivative.
    let rf_v = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &kvec(&q_v))?;
    let h_tot_v = {
        let mut m = f_v.clone();
        axpy(&mut m, &rf_v, 1.0);
        m
    };
    let h_tot_vv = {
        let mut m = f_vv.clone();
        let du: Vec<f64> = {
            let a = kdot(&q_v)?;
            let b = kvec(&q_vv);
            (0..nshell).map(|s| a[s] + b[s]).collect()
        };
        axpy(
            &mut m,
            &crate::cphf::scalar_response_fock_matrix(basis, s_mat, &du)?,
            1.0,
        );
        axpy(
            &mut m,
            &crate::cphf::scalar_response_fock_matrix(basis, &s_v, &kvec(&q_v))?,
            1.0,
        );
        m
    };
    let h_dot = d_mo_rep(&mos, &cc_v, &h_tot_v, &h_tot_vv)?;
    // `Λ̇^v_pq = ḣ̃_pq − ½(ε^v_p+ε^v_q)s̃_pq − ½(ε_p+ε_q)ṡ̃_pq` — the second-order Λ that replaces
    // `ε̇` inside every degenerate-block-safe contraction of the third. Its diagonal is exactly the
    // second-order orbital-energy response `ε^vv_p = ḣ̃_pp − ε^v_p s̃_pp − ε_p ṡ̃_pp`.
    let lam2 = {
        let mut m = M::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                m[(p, q)] = h_dot[(p, q)]
                    - 0.5 * (eps_v[p] + eps_v[q]) * s_tilde_v[(p, q)]
                    - 0.5 * (eps[p] + eps[q]) * d_s_tilde[(p, q)];
            }
        }
        m
    };
    // `U̇` in the same gauge as `U` (−½ṡ̃ on the diagonal and inside (near-)degenerate blocks).
    let u_vv = {
        let mut m = M::zeros(n, n);
        for p in 0..n {
            m[(p, p)] = -0.5 * d_s_tilde[(p, p)];
            for q in 0..n {
                if p == q {
                    continue;
                }
                let gap = eps[q] - eps[p];
                if gap.abs() < 1.0e-6 {
                    m[(p, q)] = -0.5 * d_s_tilde[(p, q)];
                } else {
                    let num = h_dot[(p, q)]
                        - eps_v[q] * s_tilde_v[(p, q)]
                        - eps[q] * d_s_tilde[(p, q)];
                    m[(p, q)] = num / gap - u_v[(p, q)] * (eps_v[q] - eps_v[p]) / gap;
                }
            }
        }
        m
    };
    // `C^vv = D_λ(C U) = C^v U + C U̇`.
    let cc_vv = {
        let mut m = cc_v.matmul(&u_v)?;
        axpy(&mut m, &mos.matmul(&u_vv)?, 1.0);
        m
    };
    let d2_s_tilde = d2_mo_rep(&mos, &cc_v, &cc_vv, &s_v, &s_vv, &s_vvv)?;
    let s_tilde_b = s_tilde_v.clone();
    let f_tilde_b = mo_rep(&mos, &f_v)?;
    let d_f_tilde = d_mo_rep(&mos, &cc_v, &f_v, &f_vv)?;
    let d2_f_tilde = d2_mo_rep(&mos, &cc_v, &cc_vv, &f_v, &f_vv, &f_vvv)?;
    // `S·C` and its two λ-derivatives (transition-charge frame transport).
    let dsc_v = {
        let mut m = s_v.matmul(&mos)?;
        axpy(&mut m, &s_mat.matmul(&cc_v)?, 1.0);
        m
    };
    let d2sc_v = {
        let mut m = s_vv.matmul(&mos)?;
        axpy(&mut m, &s_v.matmul(&cc_v)?, 2.0);
        axpy(&mut m, &s_mat.matmul(&cc_vv)?, 1.0);
        m
    };

    // ===== groups (6)+(7): the three bundle-gradient operators =====
    // `G₁[X] = Σ_a v_a G_a[X]`.
    let group_1 = |dp: &M, dw: &M, dq: &[f64]| -> Result<f64> {
        let g = crate::cphf::response_electronic_gradient(
            system,
            electronic,
            &shell_kernel,
            &ref_ctx,
            dp,
            dp,
            dw,
            dq,
        )?;
        let mut acc = 0.0;
        for at in 0..nat {
            acc += v[3 * at] * g[at].x + v[3 * at + 1] * g[at].y + v[3 * at + 2] * g[at].z;
        }
        Ok(acc)
    };
    // `G₂[X] = Σ_ac v_a v_c (D_c G_a)[X]` — the third's Group A, verbatim.
    let group_2 = |dp: &M, dw: &M, dq: &[f64]| -> Result<f64> {
        let mut acc = 0.0;
        let sp_resp = kvec(dq);
        let sp_dot = kdot(dq)?;
        for mu in 0..n {
            for nu in 0..n {
                acc += dp[(mu, nu)] * m_vv[(mu, nu)];
            }
        }
        for s in 0..nshell {
            acc += dq[s] * dv_kern_vv[s];
        }
        for mu in 0..n {
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
                if (rmu - rnu).norm2() <= 1.0e-18 {
                    continue;
                }
                let pair = crate::integrals::contracted_pair_with_second_derivatives(
                    &basis.aos[mu],
                    &basis.aos[nu],
                    rmu,
                    rnu,
                );
                let dw_val = dw[(mu, nu)];
                let dp_val = dp[(mu, nu)];
                let p0_val = p_mat[(mu, nu)];
                let scalar_shift = v_ref[shell_mu] + v_ref[shell_nu];
                let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                let f_scc = -(dp_val * scalar_shift + p0_val * scalar_response);
                let coef = -2.0 * dw_val + f_scc;
                let dcf = -(dp_val * (pot_v[shell_mu] + pot_v[shell_nu])
                    + p_v[(mu, nu)] * scalar_response
                    + p0_val * (sp_dot[shell_mu] + sp_dot[shell_nu]));
                let dbra0 = pair.d_bra[0].to_array();
                let dket0 = pair.d_ket[0].to_array();
                for alpha in 0..3 {
                    let mut db = 0.0;
                    let mut dk = 0.0;
                    for beta in 0..3 {
                        let v_mu = v[3 * atom_mu + beta];
                        let v_nu = v[3 * atom_nu + beta];
                        db += v_mu * pair.h_bra_bra[0][alpha][beta]
                            + v_nu * pair.h_bra_ket[0][alpha][beta];
                        dk += v_mu * pair.h_bra_ket[0][beta][alpha]
                            + v_nu * pair.h_ket_ket[0][alpha][beta];
                    }
                    acc += v[3 * atom_mu + alpha] * (db * coef + dbra0[alpha] * dcf);
                    acc += v[3 * atom_nu + alpha] * (dk * coef + dket0[alpha] * dcf);
                }
            }
        }
        Ok(acc)
    };
    // `G₃[X] = Σ_acd v_a v_c v_d (D_dD_cG_a)[X]` — Group A one geometric order up: the H0/overlap
    // legs become the directional THIRD matrices / third-order pair patterns, and the pair
    // coefficient `coef = −2dw + f_scc` picks up its SECOND λ-derivative `ddcf`.
    let group_3 = |dp: &M, dw: &M, dq: &[f64]| -> Result<f64> {
        let mut acc = 0.0;
        let sp_resp = kvec(dq);
        let sp_dot = kdot(dq)?;
        let sp_ddot = kddot(dq)?;
        for mu in 0..n {
            for nu in 0..n {
                acc += dp[(mu, nu)] * m_vvv[(mu, nu)];
            }
        }
        for s in 0..nshell {
            acc += dq[s] * dv_kern_vvv[s];
        }
        for mu in 0..n {
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
                if (rmu - rnu).norm2() <= 1.0e-18 {
                    continue;
                }
                let pair = crate::integrals::contracted_pair_with_third_derivatives(
                    &basis.aos[mu],
                    &basis.aos[nu],
                    rmu,
                    rnu,
                );
                let dw_val = dw[(mu, nu)];
                let dp_val = dp[(mu, nu)];
                let p0_val = p_mat[(mu, nu)];
                let scalar_shift = v_ref[shell_mu] + v_ref[shell_nu];
                let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                let response_dot = sp_dot[shell_mu] + sp_dot[shell_nu];
                let f_scc = -(dp_val * scalar_shift + p0_val * scalar_response);
                let coef = -2.0 * dw_val + f_scc;
                let dcf = -(dp_val * (pot_v[shell_mu] + pot_v[shell_nu])
                    + p_v[(mu, nu)] * scalar_response
                    + p0_val * response_dot);
                let ddcf = -(dp_val * (pot_vv[shell_mu] + pot_vv[shell_nu])
                    + p_vv[(mu, nu)] * scalar_response
                    + 2.0 * p_v[(mu, nu)] * response_dot
                    + p0_val * (sp_ddot[shell_mu] + sp_ddot[shell_nu]));
                let dbra0 = pair.d_bra[0].to_array();
                let dket0 = pair.d_ket[0].to_array();
                let vmu = [
                    v[3 * atom_mu],
                    v[3 * atom_mu + 1],
                    v[3 * atom_mu + 2],
                ];
                let vnu = [
                    v[3 * atom_nu],
                    v[3 * atom_nu + 1],
                    v[3 * atom_nu + 2],
                ];
                for alpha in 0..3 {
                    let mut s2b = 0.0;
                    let mut s2k = 0.0;
                    for beta in 0..3 {
                        s2b += vmu[beta] * pair.h_bra_bra[0][alpha][beta]
                            + vnu[beta] * pair.h_bra_ket[0][alpha][beta];
                        s2k += vmu[beta] * pair.h_bra_ket[0][beta][alpha]
                            + vnu[beta] * pair.h_ket_ket[0][alpha][beta];
                    }
                    let mut s3b = 0.0;
                    let mut s3k = 0.0;
                    for beta in 0..3 {
                        for gamma in 0..3 {
                            let wbb = vmu[beta] * vmu[gamma];
                            let wbk = vmu[beta] * vnu[gamma];
                            let wkk = vnu[beta] * vnu[gamma];
                            s3b += wbb * pair.t_bra_bra_bra[0][alpha][beta][gamma]
                                + 2.0 * wbk * pair.t_bra_bra_ket[0][alpha][beta][gamma]
                                + wkk * pair.t_bra_ket_ket[0][alpha][beta][gamma];
                            s3k += wbb * pair.t_bra_bra_ket[0][beta][gamma][alpha]
                                + 2.0 * wbk * pair.t_bra_ket_ket[0][beta][gamma][alpha]
                                + wkk * pair.t_ket_ket_ket[0][alpha][beta][gamma];
                        }
                    }
                    acc += vmu[alpha] * (coef * s3b + 2.0 * dcf * s2b + ddcf * dbra0[alpha]);
                    acc += vnu[alpha] * (coef * s3k + 2.0 * dcf * s2k + ddcf * dket0[alpha]);
                }
            }
        }
        Ok(acc)
    };

    // ===== the STATIC bundle ladder (X_s, D_vX_s, D²_vX_s) =====
    let mut bmat = M::zeros(n, n);
    let mut dbmat = M::zeros(n, n);
    let mut d2bmat = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            let w = -0.5 * (occ[i] + occ[j]);
            bmat[(i, j)] = w * s_tilde_b[(i, j)];
            dbmat[(i, j)] = w * d_s_tilde[(i, j)];
            d2bmat[(i, j)] = w * d2_s_tilde[(i, j)];
        }
    }
    let dp_s = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &bmat)?;
    let d_dp_s = d_ao_rep(&mos, &cc_v, &bmat, &dbmat)?;
    let d2_dp_s = d2_ao_rep(&mos, &cc_v, &cc_vv, &bmat, &dbmat, &d2bmat)?;
    let dq_s = crate::cphf::response_shell_charges_from_density(basis, s_mat, p_mat, &dp_s, &s_v)?;
    let d_dq_s: Vec<f64> = {
        let a = population(&d_dp_s, s_mat);
        let b2 = population(&dp_s, &s_v);
        let d = population(&p_v, &s_v);
        let e = population(p_mat, &s_vv);
        (0..nshell).map(|s| a[s] + b2[s] + d[s] + e[s]).collect()
    };
    let d2_dq_s: Vec<f64> = {
        let a = population(&d2_dp_s, s_mat);
        let b2 = population(&d_dp_s, &s_v);
        let c2 = population(&dp_s, &s_vv);
        let d = population(&p_vv, &s_v);
        let e = population(&p_v, &s_vv);
        let f = population(p_mat, &s_vvv);
        (0..nshell)
            .map(|s| a[s] + 2.0 * b2[s] + c2[s] + d[s] + 2.0 * e[s] + f[s])
            .collect()
    };
    let sp_s = kvec(&dq_s);
    let d_sp_s: Vec<f64> = {
        let a = kdot(&dq_s)?;
        let b2 = kvec(&d_dq_s);
        (0..nshell).map(|s| a[s] + b2[s]).collect()
    };
    let d2_sp_s: Vec<f64> = {
        let a = kddot(&dq_s)?;
        let b2 = kdot(&d_dq_s)?;
        let c2 = kvec(&d2_dq_s);
        (0..nshell)
            .map(|s| a[s] + 2.0 * b2[s] + c2[s])
            .collect()
    };
    let rf_s = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp_s)?;
    let d_rf_s = {
        let mut m = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d_sp_s)?;
        axpy(
            &mut m,
            &crate::cphf::scalar_response_fock_matrix(basis, &s_v, &sp_s)?,
            1.0,
        );
        m
    };
    let d2_rf_s = {
        let mut m = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d2_sp_s)?;
        axpy(
            &mut m,
            &crate::cphf::scalar_response_fock_matrix(basis, &s_v, &d_sp_s)?,
            2.0,
        );
        axpy(
            &mut m,
            &crate::cphf::scalar_response_fock_matrix(basis, &s_vv, &sp_s)?,
            1.0,
        );
        m
    };
    let rf_mo_s = mo_rep(&mos, &rf_s)?;
    let d_rf_mo_s = d_mo_rep(&mos, &cc_v, &rf_s, &d_rf_s)?;
    let d2_rf_mo_s = d2_mo_rep(&mos, &cc_v, &cc_vv, &rf_s, &d_rf_s, &d2_rf_s)?;
    // The Λ-covariant `ε`-contractions of the metric weights (degenerate-block safe).
    let lam_s = |l: &M, m: &M, i: usize, j: usize| -> f64 {
        let mut acc = 0.0;
        for &k in &block_members[i] {
            acc += l[(i, k)] * m[(k, j)];
        }
        for &k in &block_members[j] {
            acc += m[(i, k)] * l[(k, j)];
        }
        acc
    };
    let mut cwa = M::zeros(n, n);
    let mut cwb = M::zeros(n, n);
    let mut dcwa = M::zeros(n, n);
    let mut dcwb = M::zeros(n, n);
    let mut d2cwa = M::zeros(n, n);
    let mut d2cwb = M::zeros(n, n);
    for i in 0..n {
        if occ[i] <= 1e-8 {
            continue;
        }
        for j in 0..n {
            if occ[j] <= 1e-8 {
                continue;
            }
            let w = 0.5 * (occ[i] + occ[j]);
            let de = eps[i] + eps[j];
            cwa[(i, j)] = w * (f_tilde_b[(i, j)] - de * s_tilde_b[(i, j)]);
            cwb[(i, j)] = w * rf_mo_s[(i, j)];
            dcwa[(i, j)] = w
                * (d_f_tilde[(i, j)]
                    - lam_s(&lam_v, &s_tilde_b, i, j)
                    - de * d_s_tilde[(i, j)]);
            dcwb[(i, j)] = w * d_rf_mo_s[(i, j)];
            d2cwa[(i, j)] = w
                * (d2_f_tilde[(i, j)]
                    - lam_s(&lam2, &s_tilde_b, i, j)
                    - 2.0 * lam_s(&lam_v, &d_s_tilde, i, j)
                    - de * d2_s_tilde[(i, j)]);
            d2cwb[(i, j)] = w * d2_rf_mo_s[(i, j)];
        }
    }
    let dw_s = {
        let mut m = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwa)?;
        axpy(
            &mut m,
            &crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwb)?,
            1.0,
        );
        m
    };
    let d_dw_s = {
        let mut m = d_ao_rep(&mos, &cc_v, &cwa, &dcwa)?;
        axpy(&mut m, &d_ao_rep(&mos, &cc_v, &cwb, &dcwb)?, 1.0);
        m
    };
    let d2_dw_s = {
        let mut m = d2_ao_rep(&mos, &cc_v, &cc_vv, &cwa, &dcwa, &d2cwa)?;
        axpy(
            &mut m,
            &d2_ao_rep(&mos, &cc_v, &cc_vv, &cwb, &dcwb, &d2cwb)?,
            1.0,
        );
        m
    };

    // ===== the ORBITAL bundle ladder `B u`, `(D_vB)u`, `(D²_vB)u` for a fixed amplitude `u` =====
    let orbital_ladder = |u: &[f64]| -> Result<BundleLadder> {
        let ob = crate::cphf::orbital_response_bundle_from_amplitudes(
            basis,
            s_mat,
            p_mat,
            &mos,
            occ,
            eps,
            &space,
            &shell_kernel,
            u,
        )?;
        let (p0, w0, q0) = (
            ob.density.clone(),
            ob.weighted.clone(),
            ob.shell_charges.clone(),
        );
        let mut coeff_p = M::zeros(n, n);
        let mut coeff_w1 = M::zeros(n, n);
        let mut dcoeff_w1 = M::zeros(n, n);
        let mut d2coeff_w1 = M::zeros(n, n);
        for (pi, &(i, a)) in space.pairs.iter().enumerate() {
            let w = (occ[i] - occ[a]) * u[pi];
            coeff_p[(a, i)] += w;
            coeff_p[(i, a)] += w;
            let w1 = w * eps[i];
            coeff_w1[(a, i)] += w1;
            coeff_w1[(i, a)] += w1;
            // Λ-covariant `ε^{(v)}·u` and `ε^{(vv)}·u` contractions.
            let mut e1 = 0.0;
            let mut e2 = 0.0;
            for &i2 in &block_members[i] {
                let p2 = pair_of[i2 * n + a];
                if p2 != usize::MAX {
                    e1 += lam_v[(i, i2)] * u[p2];
                    e2 += lam2[(i, i2)] * u[p2];
                }
            }
            let dw1 = (occ[i] - occ[a]) * e1;
            dcoeff_w1[(a, i)] += dw1;
            dcoeff_w1[(i, a)] += dw1;
            let d2w1 = (occ[i] - occ[a]) * e2;
            d2coeff_w1[(a, i)] += d2w1;
            d2coeff_w1[(i, a)] += d2w1;
        }
        let p1 = d_ao_rep(&mos, &cc_v, &coeff_p, &zero)?;
        let p2 = d2_ao_rep(&mos, &cc_v, &cc_vv, &coeff_p, &zero, &zero)?;
        let q1: Vec<f64> = {
            let a = population(&p1, s_mat);
            let b2 = population(&p0, &s_v);
            (0..nshell).map(|s| a[s] + b2[s]).collect()
        };
        let q2: Vec<f64> = {
            let a = population(&p2, s_mat);
            let b2 = population(&p1, &s_v);
            let c2 = population(&p0, &s_vv);
            (0..nshell)
                .map(|s| a[s] + 2.0 * b2[s] + c2[s])
                .collect()
        };
        let sp0 = kvec(&q0);
        let sp1: Vec<f64> = {
            let a = kdot(&q0)?;
            let b2 = kvec(&q1);
            (0..nshell).map(|s| a[s] + b2[s]).collect()
        };
        let sp2: Vec<f64> = {
            let a = kddot(&q0)?;
            let b2 = kdot(&q1)?;
            let c2 = kvec(&q2);
            (0..nshell)
                .map(|s| a[s] + 2.0 * b2[s] + c2[s])
                .collect()
        };
        let rf0 = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp0)?;
        let rf1 = {
            let mut m = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp1)?;
            axpy(
                &mut m,
                &crate::cphf::scalar_response_fock_matrix(basis, &s_v, &sp0)?,
                1.0,
            );
            m
        };
        let rf2 = {
            let mut m = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp2)?;
            axpy(
                &mut m,
                &crate::cphf::scalar_response_fock_matrix(basis, &s_v, &sp1)?,
                2.0,
            );
            axpy(
                &mut m,
                &crate::cphf::scalar_response_fock_matrix(basis, &s_vv, &sp0)?,
                1.0,
            );
            m
        };
        let rf_mo0 = mo_rep(&mos, &rf0)?;
        let rf_mo1 = d_mo_rep(&mos, &cc_v, &rf0, &rf1)?;
        let rf_mo2 = d2_mo_rep(&mos, &cc_v, &cc_vv, &rf0, &rf1, &rf2)?;
        let mut coeff_w2 = M::zeros(n, n);
        let mut dcoeff_w2 = M::zeros(n, n);
        let mut d2coeff_w2 = M::zeros(n, n);
        for i in 0..n {
            if occ[i] <= 1e-8 {
                continue;
            }
            for j in 0..n {
                if occ[j] <= 1e-8 {
                    continue;
                }
                let w = 0.5 * (occ[i] + occ[j]);
                coeff_w2[(i, j)] = w * rf_mo0[(i, j)];
                dcoeff_w2[(i, j)] = w * rf_mo1[(i, j)];
                d2coeff_w2[(i, j)] = w * rf_mo2[(i, j)];
            }
        }
        let w1 = {
            let mut m = d_ao_rep(&mos, &cc_v, &coeff_w1, &dcoeff_w1)?;
            axpy(&mut m, &d_ao_rep(&mos, &cc_v, &coeff_w2, &dcoeff_w2)?, 1.0);
            m
        };
        let w2 = {
            let mut m = d2_ao_rep(&mos, &cc_v, &cc_vv, &coeff_w1, &dcoeff_w1, &d2coeff_w1)?;
            axpy(
                &mut m,
                &d2_ao_rep(&mos, &cc_v, &cc_vv, &coeff_w2, &dcoeff_w2, &d2coeff_w2)?,
                1.0,
            );
            m
        };
        Ok(BundleLadder {
            p0,
            w0,
            q0,
            p1,
            w1,
            q1,
            p2,
            w2,
            q2,
        })
    };

    // ===== the amplitude sector: `x^v`, `y^v`, `dx^v` (group 4) =====
    let _padj = crate::profile::scope("fc4.stage5.adjoint_sector");
    let l_vectors = crate::cphf::density_gradient_adjoint_vectors(
        system, params, electronic, ao_opts, &mos, eps,
    )?;
    let l_v: Vec<f64> = {
        let mut out = vec![0.0_f64; npair];
        for a in 0..ndof {
            let va = v[a];
            if va == 0.0 {
                continue;
            }
            for p in 0..npair {
                out[p] += va * l_vectors[a][p];
            }
        }
        out
    };
    let setup = crate::cphf::build_cpxtb_setup(system, params, electronic, ao_opts, Some(&mos))?;
    let y_v = setup.solve_adjoint(&l_v, 1.0e-11, 4000)?.amplitudes;
    drop(_padj);

    // Transition-charge derivatives (group 6): `q^trans`, `D_v q^trans`, `D²_v q^trans`.
    let dqt: Vec<Vec<f64>> = space
        .pairs
        .iter()
        .map(|&(i, a)| {
            let mut q = vec![0.0_f64; nshell];
            for (sh, shell) in basis.shells.iter().enumerate() {
                let end = shell.first_ao + shell.nao;
                for mu in shell.first_ao..end {
                    q[sh] -= cc_v[(mu, a)] * sc[(mu, i)]
                        + mos[(mu, a)] * dsc_v[(mu, i)]
                        + cc_v[(mu, i)] * sc[(mu, a)]
                        + mos[(mu, i)] * dsc_v[(mu, a)];
                }
            }
            q
        })
        .collect();
    let d2qt: Vec<Vec<f64>> = space
        .pairs
        .iter()
        .map(|&(i, a)| {
            let mut q = vec![0.0_f64; nshell];
            for (sh, shell) in basis.shells.iter().enumerate() {
                let end = shell.first_ao + shell.nao;
                for mu in shell.first_ao..end {
                    q[sh] -= cc_vv[(mu, a)] * sc[(mu, i)]
                        + 2.0 * cc_v[(mu, a)] * dsc_v[(mu, i)]
                        + mos[(mu, a)] * d2sc_v[(mu, i)]
                        + cc_vv[(mu, i)] * sc[(mu, a)]
                        + 2.0 * cc_v[(mu, i)] * dsc_v[(mu, a)]
                        + mos[(mu, i)] * d2sc_v[(mu, a)];
                }
            }
            q
        })
        .collect();

    // `(D_vA)·w` — the third's `d_axb` block read as a function of an arbitrary amplitude vector.
    let axb_first = |w: &[f64]| -> Result<Vec<f64>> {
        let mut g = vec![0.0_f64; nshell];
        let mut dg = vec![0.0_f64; nshell];
        for p in 0..npair {
            for s in 0..nshell {
                g[s] += q_trans[p][s] * scale_occ[p] * w[p];
                dg[s] += dqt[p][s] * scale_occ[p] * w[p];
            }
        }
        let pot = kvec(&g);
        let dpot: Vec<f64> = {
            let a = kdot(&g)?;
            let b2 = kvec(&dg);
            (0..nshell).map(|s| a[s] + b2[s]).collect()
        };
        let mut out = vec![0.0_f64; npair];
        for (p, &(i, a)) in space.pairs.iter().enumerate() {
            let mut val = 0.0;
            for &a2 in &block_members[a] {
                let p2 = pair_of[i * n + a2];
                if p2 != usize::MAX {
                    val += lam_v[(a, a2)] * w[p2];
                }
            }
            for &i2 in &block_members[i] {
                let p2 = pair_of[i2 * n + a];
                if p2 != usize::MAX {
                    val -= lam_v[(i, i2)] * w[p2];
                }
            }
            for s in 0..nshell {
                val += dqt[p][s] * pot[s] + q_trans[p][s] * dpot[s];
            }
            out[p] = val;
        }
        Ok(out)
    };
    // `(D²_vA)·w` — one more Leibniz order on the same block.
    let axb_second = |w: &[f64]| -> Result<Vec<f64>> {
        let mut g = vec![0.0_f64; nshell];
        let mut dg = vec![0.0_f64; nshell];
        let mut d2g = vec![0.0_f64; nshell];
        for p in 0..npair {
            for s in 0..nshell {
                g[s] += q_trans[p][s] * scale_occ[p] * w[p];
                dg[s] += dqt[p][s] * scale_occ[p] * w[p];
                d2g[s] += d2qt[p][s] * scale_occ[p] * w[p];
            }
        }
        let pot = kvec(&g);
        let dpot: Vec<f64> = {
            let a = kdot(&g)?;
            let b2 = kvec(&dg);
            (0..nshell).map(|s| a[s] + b2[s]).collect()
        };
        let d2pot: Vec<f64> = {
            let a = kddot(&g)?;
            let b2 = kdot(&dg)?;
            let c2 = kvec(&d2g);
            (0..nshell)
                .map(|s| a[s] + 2.0 * b2[s] + c2[s])
                .collect()
        };
        let mut out = vec![0.0_f64; npair];
        for (p, &(i, a)) in space.pairs.iter().enumerate() {
            let mut val = 0.0;
            for &a2 in &block_members[a] {
                let p2 = pair_of[i * n + a2];
                if p2 != usize::MAX {
                    val += lam2[(a, a2)] * w[p2];
                }
            }
            for &i2 in &block_members[i] {
                let p2 = pair_of[i2 * n + a];
                if p2 != usize::MAX {
                    val -= lam2[(i, i2)] * w[p2];
                }
            }
            for s in 0..nshell {
                val += d2qt[p][s] * pot[s]
                    + 2.0 * dqt[p][s] * dpot[s]
                    + q_trans[p][s] * d2pot[s];
            }
            out[p] = val;
        }
        Ok(out)
    };

    // `D_v rhs^v` and `D²_v rhs^v` (group 6).
    let mut d_rhs = vec![0.0_f64; npair];
    let mut d2_rhs = vec![0.0_f64; npair];
    for (idx, &(i, a)) in space.pairs.iter().enumerate() {
        let mut e1 = 0.0;
        let mut e2 = 0.0;
        let mut e1d = 0.0;
        for &i2 in &block_members[i] {
            e1 += lam_v[(i, i2)] * s_tilde_b[(i2, a)];
            e2 += lam2[(i, i2)] * s_tilde_b[(i2, a)];
            e1d += lam_v[(i, i2)] * d_s_tilde[(i2, a)];
        }
        d_rhs[idx] = -d_f_tilde[(i, a)] + e1 + eps[i] * d_s_tilde[(i, a)] - d_rf_mo_s[(i, a)];
        d2_rhs[idx] = -d2_f_tilde[(i, a)]
            + e2
            + 2.0 * e1d
            + eps[i] * d2_s_tilde[(i, a)]
            - d2_rf_mo_s[(i, a)];
    }
    let d_axb = axb_first(&x_v)?;
    let d2_axb = axb_second(&x_v)?;
    // `A dx^v = d_rhs − d_axb` (group 4): one extra self-adjoint solve.
    let dx_rhs: Vec<f64> = (0..npair).map(|p| d_rhs[p] - d_axb[p]).collect();
    let dx_v = setup.solve_adjoint(&dx_rhs, 1.0e-11, 4000)?.amplitudes;
    let axb_dx = axb_first(&dx_v)?;

    // ===== assemble the master formula `r4 = G₃[X] + 2G₂[X′] + G₁[X″]` =====
    let orb_x = orbital_ladder(&x_v)?;
    let orb_dx = orbital_ladder(&dx_v)?;

    // X = X_static + B x^v
    let mut x_p = dp_s.clone();
    axpy(&mut x_p, &orb_x.p0, 1.0);
    let mut x_w = dw_s.clone();
    axpy(&mut x_w, &orb_x.w0, 1.0);
    let x_q: Vec<f64> = (0..nshell).map(|s| dq_s[s] + orb_x.q0[s]).collect();

    // X′ = D_vX_static + (D_vB)x^v + B dx^v
    let mut x1_p = d_dp_s.clone();
    axpy(&mut x1_p, &orb_x.p1, 1.0);
    axpy(&mut x1_p, &orb_dx.p0, 1.0);
    let mut x1_w = d_dw_s.clone();
    axpy(&mut x1_w, &orb_x.w1, 1.0);
    axpy(&mut x1_w, &orb_dx.w0, 1.0);
    let x1_q: Vec<f64> = (0..nshell)
        .map(|s| d_dq_s[s] + orb_x.q1[s] + orb_dx.q0[s])
        .collect();

    // X″ = D²_vX_static + (D²_vB)x^v + 2(D_vB)dx^v  [+ B d²x^v, folded into the Z-vector term]
    let mut x2_p = d2_dp_s.clone();
    axpy(&mut x2_p, &orb_x.p2, 1.0);
    axpy(&mut x2_p, &orb_dx.p1, 2.0);
    let mut x2_w = d2_dw_s.clone();
    axpy(&mut x2_w, &orb_x.w2, 1.0);
    axpy(&mut x2_w, &orb_dx.w1, 2.0);
    let x2_q: Vec<f64> = (0..nshell)
        .map(|s| d2_dq_s[s] + orb_x.q2[s] + 2.0 * orb_dx.q1[s])
        .collect();

    let term_g3 = group_3(&x_p, &x_w, &x_q)?;
    let term_g2 = group_2(&x1_p, &x1_w, &x1_q)?;
    let term_g1 = group_1(&x2_p, &x2_w, &x2_q)?;
    // `G₁[B d²x^v] = y^v·[D²rhs − (D²A)x^v − 2(D_vA)dx^v]` (group 5) — no `d²x^v`, no `ẏ^v`.
    let mut term_z = 0.0;
    for p in 0..npair {
        term_z += y_v[p] * (d2_rhs[p] - d2_axb[p] - 2.0 * axb_dx[p]);
    }
    // Bisection aid (`GFN1_R4_DEBUG=1`): `G₂[X] + G₁[X′]` re-assembles `r3` from the SAME `X`/`X′`
    // this function differentiates, so a mismatch against `directional_response_third` localizes a
    // fault in the zeroth/first-order ladder rather than in `G₃`/`X″`/the Z-vector term.
    if std::env::var_os("GFN1_R4_DEBUG").is_some() {
        let r3_g2 = group_2(&x_p, &x_w, &x_q)?;
        let r3_g1 = group_1(&x1_p, &x1_w, &x1_q)?;
        eprintln!(
            "[r4] r3-reconstruction {:.17e} (G2 {r3_g2:.6e} + G1 {r3_g1:.6e})",
            r3_g2 + r3_g1
        );
        eprintln!(
            "[r4] G3 {term_g3:.6e}  2*G2[X'] {:.6e}  G1[X''] {term_g1:.6e}  z {term_z:.6e}",
            2.0 * term_g2
        );
    }
    Ok(term_g3 + 2.0 * term_g2 + term_g1 + term_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cphf::{solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions};
    use crate::electronic::{run_electronic, ElectronicOptions};

    /// **Directional-specialization gate.** `directional_response_third` must reproduce the `vvv`
    /// contraction of the validated dense slabs from
    /// [`crate::third_derivative::closed_form_response_hessian_derivative`] to within summation-order
    /// roundoff (gate: 1e-12 relative — the algebra is identical, only the contraction order and the
    /// Z-vector solve differ, so the residual is pure f64 reassociation noise).
    ///
    /// Two molecules: non-equilibrium water (generic, no symmetry) and C3v ammonia (exactly
    /// degenerate `e` levels — exercises the Λ-covariant degenerate-block algebra, whose per-`c`
    /// `block_members`/`lam`/`pair_of` machinery is the most delicate part of the port).
    #[test]
    fn directional_response_third_matches_closed_form_contraction() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let cases: [(&str, &str); 2] = [
            (
                "non-eq water",
                "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            ),
            (
                "ammonia C3v",
                "4\nnh3\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\n\
                 H 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n",
            ),
        ];
        for (label, xyz) in cases {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let options = ElectronicOptions {
                enable_dispersion: false,
                energy_tolerance: 1.0e-12,
                charge_tolerance: 1.0e-10,
                ..ElectronicOptions::default()
            };
            let cutoff = options.hamiltonian.coordination_cutoff;
            let electronic = run_electronic(&system, &params, options.clone()).unwrap();
            let ao_opts = AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            };
            let cphf = solve_nonpbc_cpxtb_hessian_response(
                &system,
                &params,
                &electronic,
                ao_opts,
                CpxtbOptions::default(),
            )
            .unwrap();
            let ndof = 3 * system.atoms.len();
            // Generic non-uniform direction (no accidental symmetry, no zero components).
            let v: Vec<f64> = (0..ndof)
                .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
                .collect();

            let slabs = crate::third_derivative::closed_form_response_hessian_derivative(
                &system,
                &params,
                &electronic,
                &cphf,
                ao_opts,
                cutoff,
            )
            .unwrap();
            let mut want = 0.0_f64;
            for c in 0..ndof {
                for a in 0..ndof {
                    for b in 0..ndof {
                        want += v[a] * v[b] * v[c] * slabs[c][(a, b)];
                    }
                }
            }
            let got = directional_response_third(
                &system,
                &params,
                &electronic,
                &cphf,
                ao_opts,
                cutoff,
                &v,
            )
            .unwrap();
            let rel = (got - want).abs() / want.abs().max(1.0e-30);
            eprintln!(
                "{label}: directional r3[v] = {got:.17e}  vs vvv-contraction {want:.17e} \
                 (abs {:.3e}, rel {rel:.3e})",
                (got - want).abs()
            );
            assert!(
                rel <= 1.0e-12,
                "{label}: directional response third deviates from the closed-form vvv \
                 contraction: got={got:.17e} want={want:.17e} rel={rel:.3e}"
            );
        }
    }

    /// The non-equilibrium water gate geometry + tight-SCF options shared by the quartic
    /// response-stage gate (mirrors the stage-3/4 fixtures in `fourth_derivative::directional`).
    fn gate_fixture() -> (PeriodicSystem, ElectronicOptions, Vec<f64>) {
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-12,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        };
        let ndof = 3 * system.atoms.len();
        // Generic non-uniform direction (no accidental symmetry, no zero components).
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        (system, options, v)
    }

    /// Reconverge the SCF and re-solve CPXTB at `sys`.
    fn rebuild_at(
        sys: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
    ) -> (
        crate::electronic::ElectronicResult,
        crate::cphf::GammaCartesianCpxtbResult,
        AoDerivativeOptions,
    ) {
        let electronic = run_electronic(sys, params, options.clone()).unwrap();
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: options.hamiltonian.coordination_cutoff,
            include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
        };
        let cphf = solve_nonpbc_cpxtb_hessian_response(
            sys,
            params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        (electronic, cphf, ao_opts)
    }

    fn displace_along(system: &PeriodicSystem, v: &[f64], step: f64) -> PeriodicSystem {
        let mut sys = system.clone();
        for (atom_idx, atom) in sys.atoms.iter_mut().enumerate() {
            atom.position.x += step * v[3 * atom_idx];
            atom.position.y += step * v[3 * atom_idx + 1];
            atom.position.z += step * v[3 * atom_idx + 2];
        }
        sys
    }

    /// **The one-pass replacement gate for group (2) of the quartic response stage.** All six
    /// skeleton second-derivative matrices built in a single AO-pair sweep each must reproduce the
    /// `ndof²` per-`(b,c)` double loop they replace ELEMENT BY ELEMENT to `1e-12 · scale`
    /// (summation-order roundoff only — the two routes evaluate the same pair data, and only the
    /// order in which the `v_b v_c` weights are applied differs).
    ///
    /// Run on two fixtures so no channel is silently zero: non-equilibrium water (both centres
    /// CN-coupled, every response channel live) and the CH3Br fragment of the stage-1 gate
    /// geometry (a heavy centre with `d` shells and a very different CN environment). The
    /// reference side is the `ndof²` double loop, so the fixtures stay small on purpose — the gate
    /// is about per-element agreement, and the full CH3Br···OH2 complex it is drawn from costs
    /// `~6×` more for no additional channel.
    fn run_skeleton_second_gate(xyz: &str, label: &str) {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let options = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-12,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        let (electronic, cphf, _ao_opts) = rebuild_at(&system, &params, &options);
        let basis = &electronic.basis;
        let nshell = basis.shells.len();

        let cn_v: Vec<f64> = {
            let cn_grad = crate::hessian::cn_gradient_matrix(&system, cutoff).unwrap();
            cn_grad
                .iter()
                .map(|row| row.iter().zip(&v).map(|(g, vc)| g * vc).sum())
                .collect()
        };
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let shell_kernel =
            crate::cphf::response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let q_v: Vec<f64> = (0..nshell)
            .map(|s| {
                (0..ndof)
                    .map(|c| v[c] * cphf.shell_charge_responses[c][s])
                    .sum()
            })
            .collect();
        let geo_v: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dvdr_q[(s, c)]).sum())
            .collect();
        let pot_v: Vec<f64> = (0..nshell)
            .map(|s| {
                geo_v[s]
                    + (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * q_v[t])
                        .sum::<f64>()
            })
            .collect();

        let want = skeleton_second_double_loop(
            &system,
            &params,
            &electronic,
            &cphf,
            cutoff,
            &v,
            &cn_v,
        )
        .unwrap();
        let got = skeleton_second_one_pass(
            &system,
            &params,
            &electronic,
            cutoff,
            &v,
            &geo_v,
            &pot_v,
            &q_v,
            &cn_v,
        )
        .unwrap();

        let blocks: [(&str, &Matrix, &Matrix); 6] = [
            ("m_vv", &want.m_vv, &got.m_vv),
            (
                "m_vv_cache_motion",
                &want.m_vv_cache_motion,
                &got.m_vv_cache_motion,
            ),
            ("s_vv", &want.s_vv, &got.s_vv),
            ("f_scc_vv", &want.f_scc_vv, &got.f_scc_vv),
            (
                "f_scc_vv_skeleton",
                &want.f_scc_vv_skeleton,
                &got.f_scc_vv_skeleton,
            ),
            (
                "f_scc_vv_cache_motion",
                &want.f_scc_vv_cache_motion,
                &got.f_scc_vv_cache_motion,
            ),
        ];
        for (name, reference, one_pass) in blocks {
            let scale = reference
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, x| m.max(x.abs()));
            let delta = reference
                .as_slice()
                .iter()
                .zip(one_pass.as_slice())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            eprintln!(
                "{label} / {name}: max |one-pass − double loop| {delta:.3e} (scale {scale:.3e})"
            );
            assert!(
                scale > 1.0e-10,
                "{label} / {name}: the reference block is numerically zero — the gate is vacuous"
            );
            assert!(
                delta <= 1.0e-12 * scale.max(1.0),
                "{label} / {name}: one-pass skeleton second differs from the ndof² double loop \
                 by {delta:.6e} (scale {scale:.6e})"
            );
        }
    }

    #[test]
    fn skeleton_second_one_pass_matches_double_loop_water() {
        run_skeleton_second_gate(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            "water",
        );
    }

    #[test]
    fn skeleton_second_one_pass_matches_double_loop_ch3br() {
        run_skeleton_second_gate(
            "5\nCH3Br\nC 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\n\
             H 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\n\
             H -0.515000 -0.892000 -0.330000\n",
            "ch3br",
        );
    }

    /// **Quartic response-stage gate.** [`directional_response_fourth`] must equal the central
    /// finite difference along `v` of [`directional_response_third`] with EVERYTHING rebuilt at the
    /// displaced geometries: SCF reconverged (so every cached reference field — density, shell
    /// charges, SCC potential, coordination numbers — moves), a fresh CPXTB solve (so the CP
    /// amplitudes, MO coefficients and orbital energies move), same fixed direction `v`.
    ///
    /// Two FD steps assert the `h²` truncation scaling: a residual that does NOT shrink ~4× when
    /// the step is halved is a missing analytic term, not FD noise. The target is a SCALAR
    /// (a `vvvv` contraction), so no MO-phase alignment is needed between the displaced solves.
    #[test]
    fn directional_response_fourth_matches_third_fd_along_v() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let (system, options, v) = gate_fixture();
        let cutoff = options.hamiltonian.coordination_cutoff;

        let (electronic, cphf, ao_opts) = rebuild_at(&system, &params, &options);
        let analytic = directional_response_fourth(
            &system,
            &params,
            &electronic,
            &cphf,
            ao_opts,
            cutoff,
            &v,
        )
        .unwrap();

        let third_at = |sys: &PeriodicSystem| -> f64 {
            let (el, cp, ao) = rebuild_at(sys, &params, &options);
            directional_response_third(sys, &params, &el, &cp, ao, cutoff, &v).unwrap()
        };
        let fd_at = |h: f64| -> f64 {
            (third_at(&displace_along(&system, &v, h)) - third_at(&displace_along(&system, &v, -h)))
                / (2.0 * h)
        };
        let h1 = 1.0e-3;
        let fd1 = fd_at(h1);
        let delta1 = (analytic - fd1).abs();
        let fd2 = fd_at(0.5 * h1);
        let delta2 = (analytic - fd2).abs();
        eprintln!(
            "response quartic stage: analytic {analytic:.10e} fd(h) {fd1:.10e} fd(h/2) {fd2:.10e} \
             delta(h) {delta1:.3e} delta(h/2) {delta2:.3e} ratio {:.2}",
            delta1 / delta2.max(1.0e-300)
        );
        assert!(
            delta1 < 1.0e-6 * (1.0 + fd1.abs()),
            "directional response fourth vs FD(third): analytic {analytic:.10e} fd {fd1:.10e} \
             delta {delta1:.3e}"
        );
        assert!(
            delta2 < 0.4 * delta1,
            "residual does not scale as h^2 (delta(h) {delta1:.3e}, delta(h/2) {delta2:.3e}) — \
             suspect a missing analytic term"
        );
    }
}
