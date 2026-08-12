// SPDX-License-Identifier: GPL-3.0-or-later
// Staged builders for the Brillouin-zone-sampled analytic third derivative, plus the assembly
// that consumes them. The public entry points at the bottom of the file are live; a handful of
// intermediate fields (`KSecondOrder::p_k` / `w_k`, `DkTablesK::len`) exist so the gates can
// inspect the Bloch representation the assembly itself only ever sees through its real-space
// images, and would otherwise read as dead code.
#![allow(dead_code)]
//! **k-point extension of the analytic Gamma-point periodic third derivative** (task #14).
//!
//! The Gamma-point analytic FC3 (`pbc::gamma_third` for the frozen half,
//! `pbc::gamma_response` for the assembly) is
//!
//! ```text
//!   d³E/dλ³ = frozen.total().third + density_path + g(X²) + B6 + 2·bg4
//! ```
//!
//! This module supplies the pieces of that structure that change when `pbc.kmesh` is a real
//! Brillouin-zone mesh instead of `Gamma`. The complex second-order response `X²` is **not** here.
//!
//! # What actually depends on `k`
//!
//! Surprisingly little. Walking the frozen builders in `pbc::gamma_third`:
//!
//! * every classical block (repulsion / halogen / D3) sees only geometry;
//! * the CN, band/Pulay and SCC2 blocks read the electronic state through exactly two objects —
//!   the **real-space density images** `P(T)` / `W(T)` and the **real shell charges** `q`;
//! * `gamma_realspace_densities` is already a general inverse Bloch transform
//!   `M(T) = Σ_k w_k Re[M(k) e^{-ik·T}]` over `scf.kpoints`, not a Gamma special case;
//! * `shell_potential_second_directional` reads only `scf.shell_charges` and the lattice.
//!
//! The one Gamma-shaped input the frozen bundle takes is `GammaSkeletonDerivatives`, and it uses
//! *one field* of it: `shell_potential`, the frozen-charge `∂V_s/∂R`. That field is built by
//! `pbc::hessian::shell_potential_derivatives`, which is a pure charge/lattice object — the
//! k-point skeleton builds it with the very same call. So the frozen half needs **no** k-point
//! mathematics at all, only a wrapper that sources `V₁` from a place that does not require a real
//! Gamma skeleton; [`kpoint_shell_potential_first_directional`] is that place and
//! [`kpoint_frozen_third_directional`] is that wrapper. `kpoint_frozen_third_matches_bundle_second_fd`
//! is the gate that turns this argument into a measurement.
//!
//! # Inventory
//!
//! | item | supplies | gate |
//! |---|---|---|
//! | [`kpoint_shell_potential_first_directional`] | `V₁` at a k-point SCC | production `∂V/∂R` + frozen-charge FD |
//! | [`kpoint_frozen_third_directional`] | the whole frozen bundle over a k-mesh | its own `second`'s central difference |
//! | [`kpoint_first_order_directional`] | `X¹ = (P¹(T), W¹(T), q¹)` | `q¹` vs reconverged-SCC FD |
//! | [`kpoint_directional_second_matrices`] | `(F^vv(k), S^vv(k))` | Gamma-mesh equality + per-k skeleton FD |
//!
//! # Why the small radial/geometry helpers are duplicated here
//!
//! `pbc::gamma_third` keeps `radial_chain`, `prefactor_radial3`, `cn_count_value_derivatives3`,
//! `atom_direction`, `pair_direction`, `mat3_contract` and `ao_tables` private, and that module is
//! owned by another workstream. [`kpoint_directional_second_matrices`] is a term-by-term Bloch
//! transcription of `gamma_directional_second_matrices` and needs all seven. They are reproduced
//! verbatim below rather than reached for, and the Gamma-mesh equality gate
//! (`kpoint_second_matrices_match_gamma_builder`, element-wise against the Gamma function itself)
//! is what keeps the two copies from drifting: any edit to one of these helpers on either side
//! breaks that gate immediately.

use std::collections::HashMap;

use crate::basis::{BasisSet, BasisShell};
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::data_tables::{atomic_radius_bohr, covalent_radius_d3_bohr};
use crate::electronic::ElectronicOptions;
use crate::error::Result;
use crate::hamiltonian::hscale;
use crate::integrals::{contracted_pair, contracted_pair_with_second_derivatives};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::pbc::complex::CMatrix;
use crate::pbc::gamma_third::{
    gamma_realspace_densities, gamma_response_path_directional, pbc_band_pulay_third_directional,
    pbc_cn_third_directional, pbc_dispersion_third_directional, pbc_halogen_third_directional,
    pbc_repulsion_third_directional, pbc_scc2_bilinear_second_directional,
    pbc_scc2_ewald_third_directional, pbc_scc2_realspace_third_directional,
    shell_potential_second_directional, DirectionalDerivs, FrozenDensityImages, GammaFrozenThird,
};
use crate::pbc::hessian::{
    build_response_band_pairs, kpoint_cpxtb_density_responses, periodic_response_kernel,
    response_gradient, shell_potential_derivatives, DensityLookup, ResponseBandPair,
};
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::PbcSccResult;
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use crate::third_derivative::SymmetricThird;

/// Mirrors the image-sum cutoff convention of the block being differentiated.
const DIST_EPS: f64 = 1.0e-12;

// -------------------------------------------------------------------------------------------
// Shell scalar potential at a k-point SCC
// -------------------------------------------------------------------------------------------

/// `V₁_s = Σ_col v_col ∂V_s/∂R_col` at **frozen shell charges**, for a k-point SCC result.
///
/// This is the k-point twin of `pbc::gamma_third::shell_potential_first_directional`, which reads
/// the same numbers off a `GammaSkeletonDerivatives`. It is a wrapper and not a reimplementation:
/// `pbc::hessian::shell_potential_derivatives` is exactly what *both*
/// `gamma_skeleton_derivatives` and `kpoint_skeleton_derivatives` populate their
/// `shell_potential` field with, and it depends on the electronic state only through the real
/// `scf.shell_charges` / `scf.atomic_charges`. The scalar SCC potential is a property of the
/// converged charge density, not of the Bloch phase, so there is **nothing** k-resolved to add —
/// which is precisely what `kpoint_shell_potential_first_matches_production_and_fd` measures, in
/// both directions (equality with the production k-point skeleton, and a frozen-charge central
/// difference of the potential itself).
///
/// `params` and `options` are taken for signature parity with the rest of this module's entry
/// points (and so a future charge-model change that does need them is a body edit, not a
/// call-site churn); the shell potential itself needs neither.
pub(crate) fn kpoint_shell_potential_first_directional(
    system: &PeriodicSystem,
    _params: &Gfn1Parameters,
    scf: &PbcSccResult,
    _options: &ElectronicOptions,
    pbc: &PbcOptions,
    v: &[f64],
) -> Result<Vec<f64>> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic third derivative requires a lattice");
    let dv = shell_potential_derivatives(system, &lattice, scf, pbc)?;
    let nsh = scf.basis.shells.len();
    let mut out = vec![0.0; nsh];
    for (col, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        for (s, o) in out.iter_mut().enumerate() {
            *o += vc * dv[col][s];
        }
    }
    Ok(out)
}

// -------------------------------------------------------------------------------------------
// Frozen bundle over a k-mesh
// -------------------------------------------------------------------------------------------

/// Every frozen (response-free) component of the periodic third derivative for one direction `v`,
/// evaluated against a **k-point** SCC result.
///
/// Component for component the same bundle `pbc_gamma_frozen_third_directional` assembles; the
/// two differences are that `V₁` comes from [`kpoint_shell_potential_first_directional`] (no real
/// Gamma skeleton is built or needed) and that the density images fed to the CN and band/Pulay
/// blocks are the Brillouin-zone inverse transform `Σ_k w_k Re[M(k) e^{-ik·T}]` rather than the
/// single Gamma matrix. `gamma_realspace_densities` already performs that sum for a general
/// `scf.kpoints`, so it is reused verbatim.
///
/// The claim that no *other* block needs a k-point form is a claim about the frozen half of the
/// theory, and `kpoint_frozen_third_matches_bundle_second_fd` is the measurement: a central
/// difference of this bundle's own `second` along `λ`, at frozen SCC, must reproduce `third` with
/// an `h`-ladder ratio of 4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kpoint_frozen_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    enable_dispersion: bool,
    dispersion_reference: Option<&str>,
    v: &[f64],
) -> Result<GammaFrozenThird> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic third derivative requires a lattice");
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let coordination_cutoff = options.hamiltonian.coordination_cutoff;
    let dens = gamma_realspace_densities(scf, &lattice, pbc.ao_cutoff);
    let v1 = kpoint_shell_potential_first_directional(system, params, scf, options, pbc, v)?;
    let v2 = shell_potential_second_directional(system, &lattice, scf, pbc, v);
    Ok(GammaFrozenThird {
        repulsion: pbc_repulsion_third_directional(system, params, v)?,
        halogen: pbc_halogen_third_directional(system, params, v)?,
        dispersion: if enable_dispersion {
            pbc_dispersion_third_directional(system, params, dispersion_reference, v)?
        } else {
            DirectionalDerivs::default()
        },
        coordination: if enable_cn {
            pbc_cn_third_directional(
                system,
                params,
                scf,
                pbc,
                coordination_cutoff,
                &dens.p,
                v,
            )?
        } else {
            DirectionalDerivs::default()
        },
        band_pulay: pbc_band_pulay_third_directional(
            system, params, scf, pbc, &dens, &v1, &v2, v,
        )?,
        scc2_realspace: pbc_scc2_realspace_third_directional(system, &lattice, scf, pbc, v),
        scc2_ewald: pbc_scc2_ewald_third_directional(system, &lattice, scf, pbc, v),
    })
}

// -------------------------------------------------------------------------------------------
// Directional first-order response over the mesh
// -------------------------------------------------------------------------------------------

/// The directional first-order response `X¹` of a k-point SCC: the real-space density images the
/// frozen builders consume, the shell-charge response, and the per-k complex matrices the
/// second-order response will need.
#[derive(Clone, Debug)]
pub(crate) struct KFirstOrder {
    /// `P¹(T) = Σ_k w_k Re[P¹(k) e^{-ik·T}]` over the AO-cutoff image set — the same real-space
    /// convention (and the same key type) as `FrozenDensityImages::p`, so it drops straight into
    /// the frozen builders' density slot.
    pub p_images: HashMap<[i32; 3], Matrix>,
    /// `W¹(T)`, the energy-weighted twin.
    pub w_images: HashMap<[i32; 3], Matrix>,
    /// `q¹_s`, the real shell-charge response. Gated against a reconverged-SCC central difference.
    pub q: Vec<f64>,
    /// `P¹(k)` per k-point, in `scf.kpoints` order. Kept because the complex second-order
    /// response is built in the Bloch representation, not the real-space one.
    pub p_k: Vec<CMatrix>,
    /// `W¹(k)` per k-point.
    pub w_k: Vec<CMatrix>,
}

/// One directional first-order CPXTB response over the whole Brillouin-zone mesh: `(P¹, W¹, q¹)`
/// for the displacement direction `v`.
///
/// Built as the `v`-contraction of the per-DOF response
/// [`crate::pbc::hessian::kpoint_cpxtb_density_responses`]. That is exact, not an approximation:
/// the coupled-perturbed solve is linear in its right-hand side, the right-hand side is linear in
/// the skeleton pair `(F^y(k), S^y(k))`, and the metric, charge and energy-weighted assemblies are
/// each linear in the solution — so `Σ_y v_y X¹_y` **is** the response to `Σ_y v_y (F^y, S^y)`.
///
/// # Cost
///
/// This is the correctness-first form and it pays the per-DOF price: `3N` coupled solves where a
/// direction-contracted skeleton would need one, the same relationship the Gamma path has between
/// `gamma_cpxtb_response_directional` (one PCG solve on a pre-contracted `(F¹, S¹)`) and the
/// per-DOF sweep. The k-point analogue of that shortcut is a complex PCG over the mesh driven by
/// `Σ_y v_y` of `kpoint_skeleton_derivatives`; it is a strict refactor of
/// `kpoint_cpxtb_density_responses`'s inner block and changes no result, so it belongs behind this
/// function's gate rather than in front of it.
pub(crate) fn kpoint_first_order_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    v: &[f64],
) -> Result<KFirstOrder> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic third derivative requires a lattice");
    let (dp, dw, dq) = kpoint_cpxtb_density_responses(system, params, scf, options, pbc, true)?;
    Ok(kpoint_first_order_contract(
        scf,
        &lattice,
        pbc.ao_cutoff,
        &dp,
        &dw,
        &dq,
        v,
    ))
}

/// The `v`-contraction half of [`kpoint_first_order_directional`], split out so a driver that
/// evaluates many directions against one geometry pays for the `3N` coupled solves **once**.
///
/// The split is exact for the reason spelled out on [`kpoint_first_order_directional`]: every
/// stage between the skeleton pair and `(P¹, W¹, q¹)` is linear, so contracting the per-DOF
/// responses and responding to the contracted skeleton are the same object. `KpointThirdReference`
/// caches `(dp, dw, dq)` and calls this per direction.
fn kpoint_first_order_contract(
    scf: &PbcSccResult,
    lattice: &Lattice,
    ao_cutoff: f64,
    dp: &[Vec<CMatrix>],
    dw: &[Vec<CMatrix>],
    dq: &[Vec<f64>],
    v: &[f64],
) -> KFirstOrder {
    let nk = scf.kpoints.len();
    let n = scf.basis.len();
    let nsh = scf.basis.shells.len();

    let mut p_k = vec![CMatrix::zeros(n); nk];
    let mut w_k = vec![CMatrix::zeros(n); nk];
    let mut q = vec![0.0; nsh];
    for (y, &vy) in v.iter().enumerate() {
        if vy == 0.0 {
            continue;
        }
        for ik in 0..nk {
            caxpy(&mut p_k[ik], vy, &dp[y][ik]);
            caxpy(&mut w_k[ik], vy, &dw[y][ik]);
        }
        for (s, qs) in q.iter_mut().enumerate() {
            *qs += vy * dq[y][s];
        }
    }

    let p_images = realspace_images(scf, lattice, ao_cutoff, &p_k);
    let w_images = realspace_images(scf, lattice, ao_cutoff, &w_k);
    KFirstOrder {
        p_images,
        w_images,
        q,
        p_k,
        w_k,
    }
}

/// `dst += a · src` for complex matrices.
fn caxpy(dst: &mut CMatrix, a: f64, src: &CMatrix) {
    let n = dst.n;
    for i in 0..n {
        for j in 0..n {
            dst.re[(i, j)] += a * src.re[(i, j)];
            dst.im[(i, j)] += a * src.im[(i, j)];
        }
    }
}

/// Inverse Bloch transform `M(T) = Σ_k w_k Re[M(k) e^{-ik·T}]` of an arbitrary per-k complex
/// matrix set, over the AO-cutoff image list.
///
/// Byte-for-byte the transform `gamma_realspace_densities` applies to `scf.density_k` /
/// `scf.ew_density_k`; that function hard-wires those two fields, so the response images need
/// their own entry point into the same sum. `kpoint_realspace_images_match_frozen_transform`
/// pins the two against each other by feeding this the SCF's own density.
fn realspace_images(
    scf: &PbcSccResult,
    lattice: &Lattice,
    ao_cutoff: f64,
    per_k: &[CMatrix],
) -> HashMap<[i32; 3], Matrix> {
    let n = per_k[0].n;
    let offsets = lattice.image_offsets(ao_cutoff);
    let mut map = HashMap::with_capacity(offsets.len());
    for off in &offsets {
        let mut m = Matrix::zeros(n, n);
        for (ik, kp) in scf.kpoints.iter().enumerate() {
            let (c, s) = bloch_phase(kp.fractional, *off);
            let wk = kp.weight;
            let mk = &per_k[ik];
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += wk * (mk.re[(i, j)] * c + mk.im[(i, j)] * s);
                }
            }
        }
        map.insert(off.n, m);
    }
    map
}

// -------------------------------------------------------------------------------------------
// Second-order skeleton matrices at a k-point
// -------------------------------------------------------------------------------------------

/// **Directional SECOND derivatives of the k-point skeleton matrices**, `(F^vv(k), S^vv(k))`:
///
/// ```text
///   S^vv_μν(k) = Σ_bc v_b v_c ∂²S(k)_μν/∂R_b∂R_c
///   F^vv_μν(k) = Σ_bc v_b v_c ∂²F(k)_μν/∂R_b∂R_c   at FROZEN shell charges
/// ```
///
/// the one-order-up twin of `pbc::hessian::kpoint_skeleton_derivatives`'s `fock` / `overlap`
/// contracted once with `v`, and the complex Bloch transcription of
/// `pbc::gamma_third::gamma_directional_second_matrices`. The three-way split of the Fock piece is
/// carried over unchanged, because it is a statement about the GFN1 Hamiltonian and not about the
/// sampling — with `hs = hscale·poly(r)`, `h = ½(se_i+se_j)·hs` and `S₀₁₂` the directional overlap
/// ladder:
///
/// ```text
///   (i)   bare H0   :  h₂·S + 2h₁·S₁ + h·S₂
///   (ii)  CN-se     :  c₂·(hs·S) + 2c₁·(hs₁·S + hs·S₁),  c_n = ½(dsedcn_i·CN^n_i + dsedcn_j·CN^n_j)
///   (iii) SCC scalar:  −½(V₂_i+V₂_j)·S(k) − (V₁_i+V₁_j)·S₁(k) − ½(V₀_i+V₀_j)·S₂(k)
/// ```
///
/// Every geometric factor is live (the full product rule, cross terms carrying their factor of
/// two) for exactly the reason the Gamma docs give: re-evaluating the production skeleton at a
/// displaced geometry against a frozen `scf` freezes three genuinely geometry-dependent reference
/// values (`shell_scc_potential`, `bloch.self_energies` through `CN`, and the `S(k)` multiplying
/// the potential derivative), so its `λ`-derivative lands cross terms short. The FD gate below
/// refreshes those three before differencing.
///
/// # Image convention: phases replace the real symmetrisation
///
/// The Gamma version sweeps one canonical image pair per unordered pair
/// (`canonical_positive_offset`, `a < b` at the origin) and scatters each contribution to both
/// `(μ,ν)` and `(ν,μ)`, which reproduces the production skeleton's fully ordered sweep at half the
/// integral cost because `S(Γ)` is real symmetric.
///
/// At a general `k` the same halving still works, but the two scatters carry **conjugate** phases.
/// The production sweep gives element `(μ_a, ν_b)` the weight `e^{ik·T}` of the image `T` applied
/// to the ket atom `b`; the mirror term of the canonical pair `(a, b, T)` is `(b, a, −T)`, which
/// lands on `(ν, μ)` with `e^{−ik·T}`. Its integral value is *identical* — the overlap ladder,
/// the radial prefactors and the CN coefficients are all invariant under simultaneously swapping
/// bra/ket and negating the translation — so the halved sweep accumulates one real scalar per AO
/// pair against `(cos, +sin)` and `(cos, −sin)`. Hermiticity is therefore structural rather than
/// imposed, and at `k = 0` every `sin` vanishes and the whole thing collapses onto the Gamma code
/// element for element. That collapse is a gate
/// (`kpoint_second_matrices_match_gamma_builder`, measured at `0.0e0`), not a comment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kpoint_directional_second_matrices(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    v1_pot: &[f64],
    v2_pot: &[f64],
    v: &[f64],
    kpoint_frac: [f64; 3],
) -> Result<(CMatrix, CMatrix)> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic third derivative requires a lattice");
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let self_energy = &scf.bloch.self_energies;
    let dsedcn = &scf.bloch.dsedcn;
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let (cn1, cn2) = if enable_cn {
        cn_directional_derivatives(system, options.hamiltonian.coordination_cutoff, v)?
    } else {
        (vec![0.0; nat], vec![0.0; nat])
    };

    let mut fock = CMatrix::zeros(n);
    let mut s1_mat = CMatrix::zeros(n);
    let mut s2_mat = CMatrix::zeros(n);

    let (atom_aos, atom_min_exp) = ao_tables(basis, nat);
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let (cph, sph) = bloch_phase(kpoint_frac, *off);
        let translation = lattice.translation(*off);
        for a in 0..nat {
            let ra = system.atoms[a].position;
            let va = atom_direction(v, a);
            for b in 0..nat {
                if is_origin && a >= b {
                    continue;
                }
                let rb = system.atoms[b].position + translation;
                let rvec = ra - rb;
                let r2 = rvec.norm2();
                if r2 <= DIST_EPS || r2 > cutoff2 {
                    continue;
                }
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let r = r2.sqrt();
                let vb = atom_direction(v, b);
                let dv = va - vb;
                let dv2 = dv.norm2();
                let sdir = rvec.dot(dv) / r;
                for &mu in &atom_aos[a] {
                    let si_idx = basis.aos[mu].shell_index;
                    let si = &basis.shells[si_idx];
                    for &nu in &atom_aos[b] {
                        let sj_idx = basis.aos[nu].shell_index;
                        let sj = &basis.shells[sj_idx];
                        let pair = contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        let overlap = pair.moments[0];
                        let s1 = pair.d_bra[0].dot(va) + pair.d_ket[0].dot(vb);
                        let s2 = mat3_contract(&pair.h_bra_bra[0], va, va)
                            + 2.0 * mat3_contract(&pair.h_bra_ket[0], va, vb)
                            + mat3_contract(&pair.h_ket_ket[0], vb, vb);

                        // (i) bare H0 at frozen self-energies.
                        let scale = hscale(si, sj, params)?;
                        let coeff = 0.5 * (self_energy[si_idx] + self_energy[sj_idx]) * scale;
                        let (hval, hp, hpp, _) = prefactor_radial3(coeff, si, sj, r)?;
                        let (h1, h2, _) = radial_chain(hp, hpp, 0.0, r, sdir, dv2);
                        let mut f2 = h2 * overlap + 2.0 * h1 * s1 + hval * s2;

                        // (ii) CN motion of the self-energies (linear in CN for GFN1).
                        if enable_cn {
                            let (hs, hsp, hspp, _) = prefactor_radial3(scale, si, sj, r)?;
                            let (hs1, _, _) = radial_chain(hsp, hspp, 0.0, r, sdir, dv2);
                            let c1 = 0.5 * (dsedcn[si_idx] * cn1[a] + dsedcn[sj_idx] * cn1[b]);
                            let c2 = 0.5 * (dsedcn[si_idx] * cn2[a] + dsedcn[sj_idx] * cn2[b]);
                            let p0 = hs * overlap;
                            let p1 = hs1 * overlap + hs * s1;
                            f2 += c2 * p0 + 2.0 * c1 * p1;
                        }

                        // The canonical pair carries `e^{+ik·T}` into `(μ,ν)`; its mirror image
                        // pair carries the conjugate into `(ν,μ)` with the same real value.
                        scatter_conjugate_pair(&mut fock, mu, nu, f2, cph, sph);
                        scatter_conjugate_pair(&mut s1_mat, mu, nu, s1, cph, sph);
                        scatter_conjugate_pair(&mut s2_mat, mu, nu, s2, cph, sph);
                    }
                }
            }
        }
    }

    // On-site (a == b, T = 0) CN block: the overlap is geometry-rigid there, so only `c₂·S₀`
    // survives. Phase 1 and both AO orders already visited, so no conjugate scatter.
    if enable_cn {
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for &mu in &atom_aos[a] {
                let si_idx = basis.aos[mu].shell_index;
                for &nu in &atom_aos[a] {
                    let sj_idx = basis.aos[nu].shell_index;
                    let s0 = contracted_pair(&basis.aos[mu], &basis.aos[nu], ra, ra).0;
                    if s0 == 0.0 {
                        continue;
                    }
                    fock.re[(mu, nu)] +=
                        0.5 * (dsedcn[si_idx] + dsedcn[sj_idx]) * s0 * cn2[a];
                }
            }
        }
    }

    // (iii) SCC scalar potential, over the full folded `S(k)` (the on-site block included,
    // exactly as the production k-point skeleton's last pass does).
    let (_, s_k) = scf.bloch.h_s_at_k(kpoint_frac);
    for mu in 0..n {
        let sh_mu = basis.aos[mu].shell_index;
        for nu in 0..n {
            let sh_nu = basis.aos[nu].shell_index;
            let a2 = -0.5 * (v2_pot[sh_mu] + v2_pot[sh_nu]);
            let a1 = -(v1_pot[sh_mu] + v1_pot[sh_nu]);
            let a0 = -0.5
                * (scf.shell_scc_potential[sh_mu] + scf.shell_scc_potential[sh_nu]);
            fock.re[(mu, nu)] += a2 * s_k.re[(mu, nu)]
                + a1 * s1_mat.re[(mu, nu)]
                + a0 * s2_mat.re[(mu, nu)];
            fock.im[(mu, nu)] += a2 * s_k.im[(mu, nu)]
                + a1 * s1_mat.im[(mu, nu)]
                + a0 * s2_mat.im[(mu, nu)];
        }
    }

    Ok((fock, s2_mat))
}

/// Accumulate one real AO-pair value into `(μ,ν)` with phase `e^{ik·T}` and into `(ν,μ)` with its
/// conjugate — the halved-sweep bookkeeping described on
/// [`kpoint_directional_second_matrices`].
#[inline]
fn scatter_conjugate_pair(m: &mut CMatrix, mu: usize, nu: usize, value: f64, cph: f64, sph: f64) {
    m.re[(mu, nu)] += value * cph;
    m.im[(mu, nu)] += value * sph;
    m.re[(nu, mu)] += value * cph;
    m.im[(nu, mu)] -= value * sph;
}

// -------------------------------------------------------------------------------------------
// Local copies of `pbc::gamma_third`'s private directional helpers (see the module docs)
// -------------------------------------------------------------------------------------------

/// The direction's relative displacement across a pair, `v_a − v_b`, as a 3-vector.
#[inline]
fn pair_direction(v: &[f64], a: usize, b: usize) -> Vec3 {
    Vec3::new(
        v[3 * a] - v[3 * b],
        v[3 * a + 1] - v[3 * b + 1],
        v[3 * a + 2] - v[3 * b + 2],
    )
}

/// The direction restricted to one atom, as a 3-vector.
#[inline]
fn atom_direction(v: &[f64], a: usize) -> Vec3 {
    Vec3::new(v[3 * a], v[3 * a + 1], v[3 * a + 2])
}

/// Directional derivatives of a scalar radial function `f(r)` along the pair displacement, with
/// `s = (rvec/r)·d` and `d2 = |d|²`.
#[inline]
fn radial_chain(f1: f64, f2: f64, f3: f64, r: f64, s: f64, d2: f64) -> (f64, f64, f64) {
    let t = (d2 - s * s) / r;
    (
        f1 * s,
        f2 * s * s + f1 * t,
        f3 * s * s * s + 3.0 * f2 * s * t - 3.0 * f1 * s * t / r,
    )
}

#[inline]
fn mat3_contract(m: &[[f64; 3]; 3], a: Vec3, b: Vec3) -> f64 {
    let (aa, bb) = (a.to_array(), b.to_array());
    let mut acc = 0.0;
    for (i, &ai) in aa.iter().enumerate() {
        for (j, &bj) in bb.iter().enumerate() {
            acc += m[i][j] * ai * bj;
        }
    }
    acc
}

/// GFN1 CN counting function and its first three radial derivatives, written in the exponential.
#[inline]
fn cn_count_value_derivatives3(kcn: f64, r: f64, rc: f64) -> (f64, f64, f64) {
    let raw = -kcn * (rc / r - 1.0);
    if !(-80.0..=80.0).contains(&raw) {
        return (0.0, 0.0, 0.0);
    }
    let e = raw.exp();
    let denom = 1.0 + e;
    let arg1 = kcn * rc / (r * r);
    let arg2 = -2.0 * kcn * rc / (r * r * r);
    let arg3 = 6.0 * kcn * rc / (r * r * r * r);
    let d2 = denom * denom;
    let d3 = d2 * denom;
    let d4 = d3 * denom;
    let e2 = e * e;
    let sig1 = -e / d2;
    let sig2 = e * (e - 1.0) / d3;
    let sig3 = -e * (e2 - 4.0 * e + 1.0) / d4;
    let first = sig1 * arg1;
    let second = sig2 * arg1 * arg1 + sig1 * arg2;
    let third = sig3 * arg1 * arg1 * arg1 + 3.0 * sig2 * arg1 * arg2 + sig1 * arg3;
    (first, second, third)
}

/// The `H0` geometric prefactor `f(r) = coeff·(1 + p_i·rr)(1 + p_j·rr)`, `rr = √(r/rad)`, and its
/// first three radial derivatives, `coeff` held constant.
fn prefactor_radial3(
    coeff: f64,
    si: &BasisShell,
    sj: &BasisShell,
    r: f64,
) -> Result<(f64, f64, f64, f64)> {
    let rad = atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?;
    let pi = si.poly_raw.unwrap_or(0.0);
    let pj = sj.poly_raw.unwrap_or(0.0);
    let a1 = pi + pj;
    let a2 = pi * pj;
    let rr = (r / rad).sqrt();
    let rr3 = rr * rr * rr;
    let rr5 = rr3 * rr * rr;
    Ok((
        coeff * (1.0 + a1 * rr + a2 * rr * rr),
        coeff * (a1 / (2.0 * rad * rr) + a2 / rad),
        coeff * (-a1 / (4.0 * rad * rad * rr3)),
        coeff * (3.0 * a1 / (8.0 * rad * rad * rad * rr5)),
    ))
}

/// Directional first and second derivatives of every coordination number, `(CN¹_k, CN²_k)`.
fn cn_directional_derivatives(
    system: &PeriodicSystem,
    cutoff: f64,
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
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
            cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let mut cn1 = vec![0.0; nat];
    let mut cn2 = vec![0.0; nat];
    for pair in &cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let r = pair.r_ij.norm();
        if r <= DIST_EPS {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let (f1, f2, f3) = cn_count_value_derivatives3(kcn, r, rc);
        let dv = pair_direction(v, pair.i, pair.j);
        let s = pair.r_ij.dot(dv) / r;
        let (h1, h2, _) = radial_chain(f1, f2, f3, r, s, dv.norm2());
        cn1[pair.i] += h1;
        cn1[pair.j] += h1;
        cn2[pair.i] += h2;
        cn2[pair.j] += h2;
    }
    Ok((cn1, cn2))
}

/// Per-atom AO index lists and the smallest primitive exponent per atom.
fn ao_tables(basis: &BasisSet, nat: usize) -> (Vec<Vec<usize>>, Vec<f64>) {
    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if p.exponent < *e {
                *e = p.exponent;
            }
        }
    }
    (atom_aos, atom_min_exp)
}

// ---------------------------------------------------------------------------------------------
// Complex linear-algebra helpers
// ---------------------------------------------------------------------------------------------
//
// `pbc::hessian` keeps `cmatmul` / `cmo_element` / `accumulate_complex_shell_charges` private and
// that module belongs to another workstream, so — exactly as for the radial helpers above — the
// handful of complex primitives this file needs are reproduced here. They are pinned by
// `kpoint_second_order_matches_gamma_charge_space`, which reduces the whole complex path to the
// Gamma-point real one element for element.

/// `A · B` for two `n × n` complex matrices.
fn cmul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let n = a.n;
    let rr = a.re.matmul(&b.re).expect("re*re");
    let ii = a.im.matmul(&b.im).expect("im*im");
    let ri = a.re.matmul(&b.im).expect("re*im");
    let ir = a.im.matmul(&b.re).expect("im*re");
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = rr[(i, j)] - ii[(i, j)];
            out.im[(i, j)] = ri[(i, j)] + ir[(i, j)];
        }
    }
    out
}

/// Conjugate transpose `Aᴴ`.
fn cadjoint(a: &CMatrix) -> CMatrix {
    let n = a.n;
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = a.re[(j, i)];
            out.im[(i, j)] = -a.im[(j, i)];
        }
    }
    out
}

/// MO representation `Cᴴ M C` of an AO-basis operator — the complex twin of the molecular
/// `ChargeSpaceContext::mo_transform`.
fn mo_transform_k(coeff: &CMatrix, m: &CMatrix) -> CMatrix {
    cmul(&cadjoint(coeff), &cmul(m, coeff))
}

/// Back-transform `C M Cᴴ` of an MO-representation coefficient matrix into the AO basis — the
/// complex twin of `mo_coefficient_matrix_to_ao`.
fn mo_to_ao_k(coeff: &CMatrix, m: &CMatrix) -> CMatrix {
    cmul(coeff, &cmul(m, &cadjoint(coeff)))
}

/// Scalar response Fock `RF(v)_μν = −½(v_μ + v_ν) M_μν` for an arbitrary complex metric `M`
/// (`S(k)` for the value channel, `S¹(k)` for the overlap-derivative channel). Same `−½(v_i+v_j)S`
/// convention as `response::cpxtb::scalar_response_fock_matrix`, verified identical between the
/// molecular and periodic sides.
fn cscalar_response_fock(basis: &BasisSet, metric: &CMatrix, shell_potential: &[f64]) -> CMatrix {
    let n = metric.n;
    let mut out = CMatrix::zeros(n);
    for mu in 0..n {
        let v_mu = shell_potential[basis.aos[mu].shell_index];
        for nu in 0..n {
            let v_nu = shell_potential[basis.aos[nu].shell_index];
            let scale = -0.5 * (v_mu + v_nu);
            out.re[(mu, nu)] = scale * metric.re[(mu, nu)];
            out.im[(mu, nu)] = scale * metric.im[(mu, nu)];
        }
    }
    out
}

/// `dq_s −= w · Re Tr_s[A B]` — the shell-resolved Mulliken channel of
/// `response_shell_charges_from_density`, one `(density, metric)` pair at a time. Only the real
/// part survives the Brillouin-zone sum, which is why the charge response is a real `nsh` vector
/// even though every per-k factor is complex.
fn accumulate_shell_trace(dq: &mut [f64], basis: &BasisSet, weight: f64, a: &CMatrix, b: &CMatrix) {
    let n = a.n;
    for mu in 0..n {
        let mut acc = 0.0;
        for kappa in 0..n {
            acc += a.re[(mu, kappa)] * b.re[(kappa, mu)] - a.im[(mu, kappa)] * b.im[(kappa, mu)];
        }
        dq[basis.aos[mu].shell_index] -= weight * acc;
    }
}

// ---------------------------------------------------------------------------------------------
// Complex Daleckii-Krein response tables
// ---------------------------------------------------------------------------------------------

/// Orbital-energy separation below which a divided difference switches to its confluent
/// (derivative) limit. Same value as the molecular `response::charge_space::DK_CONFLUENT_EPS`.
pub(crate) const DK_CONFLUENT_EPS: f64 = 1.0e-9;

/// Reference-spectrum divided differences of the grand-canonical Fermi function at **one
/// k-point** — the per-k twin of the molecular `DkTables`.
///
/// # Why nothing here is complex
///
/// The resolvent form `P = (2πi)^{-1} ∮ f(z)(zS − H)^{-1} dz` has a Hermitian pencil at every
/// `k`, so the band energies `ε_p(k)` are real and `f` is a real function of them. Every weight
/// this table holds is therefore a **real** divided difference; the complexification of the
/// response lives entirely in the matrices `A`, `S`, `𝒫` that these weights multiply. Getting
/// this wrong in the other direction (complexifying the weights) is the single most likely
/// transcription error, so it is stated here rather than left implicit.
pub(crate) struct DkTablesK {
    eps: Vec<f64>,
    f: Vec<f64>,
    fp: Vec<f64>,
    fpp: Vec<f64>,
    f1: Matrix,
    fp1: Matrix,
}

impl DkTablesK {
    /// Build the tables from one k-point's band energies and occupations.
    ///
    /// `kt = 0` is not the molecular code's path (the molecular context builds DK tables only on
    /// its finite-temperature branch, so it can divide by `kt` unguarded) but it is the *normal*
    /// case here: the k-point analytic FC3 targets gapped insulators. The `kt → 0` limit of a
    /// gapped occupation function is `f' = f'' = 0` — `f` is locally constant at every band
    /// energy — and taking it explicitly keeps every divided difference finite. The chemical
    /// potential then drops out of the response entirely (`w_sum = Σ(−f') = 0`), which is the
    /// correct statement that a gapped insulator's particle number is already exact.
    pub(crate) fn build(energies: &[f64], occupations: &[f64], kt: f64) -> Self {
        let n = energies.len();
        let eps = energies.to_vec();
        let f = occupations.to_vec();
        let (fp, fpp): (Vec<f64>, Vec<f64>) = if kt > 0.0 {
            let fp: Vec<f64> = f
                .iter()
                .map(|&fq| -(fq * (1.0 - 0.5 * fq)).max(0.0) / kt)
                .collect();
            let fpp: Vec<f64> = f
                .iter()
                .zip(&fp)
                .map(|(&fq, &fpq)| -fpq * (1.0 - fq) / kt)
                .collect();
            (fp, fpp)
        } else {
            (vec![0.0; n], vec![0.0; n])
        };
        let mut f1 = Matrix::zeros(n, n);
        let mut fp1 = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                let d = eps[p] - eps[q];
                if d.abs() > DK_CONFLUENT_EPS {
                    f1[(p, q)] = (f[p] - f[q]) / d;
                    fp1[(p, q)] = (fp[p] - fp[q]) / d;
                } else {
                    f1[(p, q)] = fp[p];
                    fp1[(p, q)] = fpp[p];
                }
            }
        }
        Self {
            eps,
            f,
            fp,
            fpp,
            f1,
            fp1,
        }
    }

    /// Second divided difference `f^{[2]}(ε_p, ε_r, ε_q)` with both confluent branches: `p ≈ q`
    /// (pinched) and the fully confluent `p ≈ r ≈ q`. Transcribed exactly from the molecular
    /// `DkTables::f2` — this is where degeneracy is handled, and it is handled by taking a limit
    /// rather than by a case split on the eigenbasis.
    #[inline]
    pub(crate) fn f2(&self, p: usize, r: usize, q: usize) -> f64 {
        let dpq = self.eps[p] - self.eps[q];
        if dpq.abs() > DK_CONFLUENT_EPS {
            return (self.f1[(p, r)] - self.f1[(r, q)]) / dpq;
        }
        let drp = self.eps[r] - self.eps[p];
        if drp.abs() > DK_CONFLUENT_EPS {
            return (self.f1[(p, r)] - self.fp[p]) / drp;
        }
        0.5 * self.fpp[p]
    }

    /// `−f'_p`, the Fermi weight that carries the chemical-potential response. Zero for a gapped
    /// reference.
    #[inline]
    pub(crate) fn mu_weight(&self, p: usize) -> f64 {
        -self.fp[p]
    }

    pub(crate) fn len(&self) -> usize {
        self.eps.len()
    }
}

/// First-order MO-representation response of `z^L f(z)` at one k-point (`L = 0` density, `L = 1`
/// energy-weighted density) — the complex twin of the molecular `dk_first_order_mo`.
///
/// Used for the particle-number condition that fixes `μ^{xy}`, for the static susceptibility
/// `χ⁰`, and as the first-order consistency check against the CPXTB solve.
pub(crate) fn kpoint_dk_first_order_mo(
    t: &DkTablesK,
    level: usize,
    a_x: &CMatrix,
    s_x: &CMatrix,
    mu_x: f64,
) -> CMatrix {
    let n = t.len();
    let e = &t.eps;
    let pow = |x: f64, k: usize| -> f64 { (0..k).fold(1.0, |acc, _| acc * x) };
    let w0 = |k: usize, q: usize| -> f64 { pow(e[q], k) * t.f[q] };
    let w1 = |k: usize, p: usize, q: usize| -> f64 {
        let mut acc = t.f1[(p, q)];
        for j in 0..k {
            acc = e[p] * acc + w0(j, q);
        }
        acc
    };
    let mut out = CMatrix::zeros(n);
    for p in 0..n {
        for q in 0..n {
            let wa = w1(level, p, q);
            let ws = w1(level + 1, p, q);
            out.re[(p, q)] = wa * a_x.re[(p, q)] - ws * s_x.re[(p, q)];
            out.im[(p, q)] = wa * a_x.im[(p, q)] - ws * s_x.im[(p, q)];
        }
        out.re[(p, p)] += mu_x * (-pow(e[p], level) * t.fp[p]);
    }
    out
}

/// **Second-order MO-representation response of `z^L f(z)` at one k-point** — the complex twin of
/// the molecular `ChargeSpaceContext::dk_second_order_mo`, and the one genuinely new object task
/// #14 needs.
///
/// From the second derivative of the resolvent,
///
/// ```text
///   d²G = G Bˣ G Bʸ G + G Bʸ G Bˣ G − G Bˣʸ G ,  B = z dS − dH
/// ```
///
/// contour integration turns each `z^k` into a divided difference of `z^{L+k} f`, built by the
/// Leibniz recursion `w_{k+1}^{[m]}(p, …, q) = ε_p w_k^{[m]}(p, …, q) + w_k^{[m−1]}(…, q)`. The
/// chemical-potential response enters as `∂_μ` chains of the same object
/// (`∂_μ z^L f(z−μ) = −z^L f′`). Returns the `μ^{xy}`-independent part; the caller adds the
/// diagonal `μ^{xy}` term after the Brillouin-zone-wide particle-number condition has fixed it.
///
/// # What the complexification changes, and what it does not
///
/// Only the matrices. `A` (the MO representation of the total Fock derivative), `S` (the overlap
/// derivative) and the result `𝒫` become Hermitian complex; every weight `f^{[1]}, f^{[2]}, g,
/// k` stays real because it is a divided difference of a real function of the real band energies
/// (see [`DkTablesK`]). Each `A_pr B_rq` that was a product of reals is now a complex product —
/// that is the entire edit, and `kpoint_second_order_matches_gamma_charge_space` measures it.
///
/// # Hermiticity is structural
///
/// Divided differences are symmetric in their nodes, so `w^{[1]}(p,q) = w^{[1]}(q,p)` and
/// `w^{[2]}(p,r,q) = w^{[2]}(q,r,p)`; the bilinear terms are written in both orderings, which is
/// exactly the pairing that maps to its own conjugate transpose. `𝒫^{xy}(k)` is therefore
/// Hermitian to round-off without any symmetrisation, and
/// `kpoint_second_order_response_is_hermitian` measures that rather than asserting it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kpoint_dk_second_order_mo(
    t: &DkTablesK,
    level: usize,
    a_x: &CMatrix,
    s_x: &CMatrix,
    a_y: &CMatrix,
    s_y: &CMatrix,
    a_xy: &CMatrix,
    s_xy: &CMatrix,
    mu_x: f64,
    mu_y: f64,
) -> CMatrix {
    let n = t.len();
    let e = &t.eps;
    // Level-lifted divided differences (Leibniz with the linear factor z).
    // w^{[0]}_k(q) = ε_q^k f_q ; w^{[1]}_k(p, q) ; w^{[2]}_k(p, r, q).
    let pow = |x: f64, k: usize| -> f64 { (0..k).fold(1.0, |acc, _| acc * x) };
    let w0 = |k: usize, q: usize| -> f64 { pow(e[q], k) * t.f[q] };
    let w1 = |k: usize, p: usize, q: usize| -> f64 {
        let mut acc = t.f1[(p, q)];
        for j in 0..k {
            acc = e[p] * acc + w0(j, q);
        }
        acc
    };
    let w2 = |k: usize, p: usize, r: usize, q: usize| -> f64 {
        let mut acc = t.f2(p, r, q);
        for j in 0..k {
            acc = e[p] * acc + w1(j, r, q);
        }
        acc
    };
    // ∂_μ chains: u = −z^k f′.
    let u0 = |k: usize, q: usize| -> f64 { -pow(e[q], k) * t.fp[q] };
    let u1 = |k: usize, p: usize, q: usize| -> f64 {
        let mut acc = -t.fp1[(p, q)];
        for j in 0..k {
            acc = e[p] * acc + u0(j, q);
        }
        acc
    };
    // `(re, im) += wt · (a · b)` for complex scalars held as (re, im) pairs — the one place the
    // real formula's plain `a * b` becomes a complex product.
    #[inline]
    fn cacc(wt: f64, ar: f64, ai: f64, br: f64, bi: f64, re: &mut f64, im: &mut f64) {
        *re += wt * (ar * br - ai * bi);
        *im += wt * (ar * bi + ai * br);
    }

    let mut out = CMatrix::zeros(n);
    for p in 0..n {
        for q in 0..n {
            // Second-order skeleton term: w^{[1]}_L A^{xy} − w^{[1]}_{L+1} S^{xy}.
            let wa = w1(level, p, q);
            let ws = w1(level + 1, p, q);
            let mut re = wa * a_xy.re[(p, q)] - ws * s_xy.re[(p, q)];
            let mut im = wa * a_xy.im[(p, q)] - ws * s_xy.im[(p, q)];
            // Bilinear resolvent term, both orderings.
            for r in 0..n {
                let f2 = w2(level, p, r, q);
                let g2 = w2(level + 1, p, r, q);
                let k2 = w2(level + 2, p, r, q);
                let (axpr, axpi) = (a_x.re[(p, r)], a_x.im[(p, r)]);
                let (axrr, axri) = (a_x.re[(r, q)], a_x.im[(r, q)]);
                let (aypr, aypi) = (a_y.re[(p, r)], a_y.im[(p, r)]);
                let (ayrr, ayri) = (a_y.re[(r, q)], a_y.im[(r, q)]);
                let (sxpr, sxpi) = (s_x.re[(p, r)], s_x.im[(p, r)]);
                let (sxrr, sxri) = (s_x.re[(r, q)], s_x.im[(r, q)]);
                let (sypr, sypi) = (s_y.re[(p, r)], s_y.im[(p, r)]);
                let (syrr, syri) = (s_y.re[(r, q)], s_y.im[(r, q)]);
                cacc(f2, axpr, axpi, ayrr, ayri, &mut re, &mut im);
                cacc(f2, aypr, aypi, axrr, axri, &mut re, &mut im);
                cacc(-g2, sxpr, sxpi, ayrr, ayri, &mut re, &mut im);
                cacc(-g2, axpr, axpi, syrr, syri, &mut re, &mut im);
                cacc(-g2, sypr, sypi, axrr, axri, &mut re, &mut im);
                cacc(-g2, aypr, aypi, sxrr, sxri, &mut re, &mut im);
                cacc(k2, sxpr, sxpi, syrr, syri, &mut re, &mut im);
                cacc(k2, sypr, sypi, sxrr, sxri, &mut re, &mut im);
            }
            // μ cross chains.
            let ua = u1(level, p, q);
            let us = u1(level + 1, p, q);
            re += mu_y * (ua * a_x.re[(p, q)] - us * s_x.re[(p, q)]);
            im += mu_y * (ua * a_x.im[(p, q)] - us * s_x.im[(p, q)]);
            re += mu_x * (ua * a_y.re[(p, q)] - us * s_y.re[(p, q)]);
            im += mu_x * (ua * a_y.im[(p, q)] - us * s_y.im[(p, q)]);
            out.re[(p, q)] = re;
            out.im[(p, q)] = im;
        }
    }
    // μ^x μ^y curvature (diagonal): ∂²_μ z^L f = +z^L f''.
    for p in 0..n {
        out.re[(p, p)] += mu_x * mu_y * pow(e[p], level) * t.fpp[p];
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Directional second-order response over the mesh
// ---------------------------------------------------------------------------------------------

/// The directional second-order response `X² = (P²(T), W²(T), q²)` of a k-point SCC, in the same
/// real-space image convention as [`KFirstOrder`] so it drops straight into the frozen builders'
/// density slots and into `response_gradient`.
#[derive(Clone, Debug)]
pub(crate) struct KSecondOrder {
    /// `P^{vv}(T) = Σ_k w_k Re[P^{vv}(k) e^{-ik·T}]`.
    pub p_images: HashMap<[i32; 3], Matrix>,
    /// `W^{vv}(T)`, the energy-weighted twin.
    pub w_images: HashMap<[i32; 3], Matrix>,
    /// `q^{vv}_s`, the real shell-charge second response.
    pub q: Vec<f64>,
    /// `P^{vv}(k)` per k-point, in `scf.kpoints` order (kept for the Hermiticity gate and for a
    /// future assembly that wants the Bloch representation).
    pub p_k: Vec<CMatrix>,
    /// `W^{vv}(k)` per k-point.
    pub w_k: Vec<CMatrix>,
}

/// Per-k-point reference data the second-order driver reuses across its two passes.
struct KRefData {
    mos: crate::pbc::hessian::KpointComplexMos,
    tables: DkTablesK,
    weight: f64,
    /// `S¹(k)` — the `v`-contracted overlap derivative (AO).
    s1: CMatrix,
    /// `A^v(k) = C(k)ᴴ (F¹_skel(k) + RF_{S(k)}(K q¹)) C(k)`, the MO representation of the TOTAL
    /// first-order Fock derivative (screening included), i.e. the exact analogue of the molecular
    /// `FirstOrderField::h_tilde`.
    a_x: CMatrix,
    /// `S^v(k) = C(k)ᴴ S¹(k) C(k)`, the analogue of `FirstOrderField::s_tilde`.
    s_x: CMatrix,
    /// `S^{vv}(k) = C(k)ᴴ S^vv_AO(k) C(k)` — the FIXED-basis second-order metric (no frame
    /// transport), the analogue of `s_dot_fixed`.
    s_xy: CMatrix,
    /// `S^vv(k)` in the AO basis, for the `−Tr_s(P₀ S^{vv})` channel of `q̃^{vv}`.
    s_vv: CMatrix,
    /// The `q^{vv}`-independent part of the AO second-order Fock derivative.
    df_ext: CMatrix,
}

/// **The all-k directional second-order response** — the last piece of the analytic k-point FC3.
///
/// This is the Brillouin-zone version of `ChargeSpaceContext::second_order_field` specialised to
/// `x = y = v`, built on [`kpoint_dk_second_order_mo`]. Structurally it follows the molecular
/// driver exactly: an **ext pass** that evaluates the response with the `q^{vv}`-independent
/// Fock derivative, a **dielectric solve** for the shell-charge second response, and a **final
/// pass** with the full `dF` including the `K q^{vv}` screening add.
///
/// # The two things that are not a transcription
///
/// **The chemical potential is one scalar for the whole zone, not one per k-point.** `μ^v` and
/// `μ^{vv}` are properties of a single global Fermi level, so the molecular particle-number
/// condition `d²Tr[SP] = 0` generalises to the *weighted sum over the mesh*
/// `d² Σ_k w_k Tr[S(k)P(k)] = 0`, giving one equation for one unknown. Solving it per k-point
/// instead would conserve each k-point's occupancy separately, which is not a physical
/// constraint. For a gapped reference every weight `−f'` vanishes and `μ` drops out entirely.
///
/// **The dielectric coupling is real and `nsh × nsh`.** The per-k responses are complex, but the
/// only thing that couples them is the shell-charge vector, which is real: `q^{vv}` solves the
/// same `(I − χ⁰K) q^{vv} = q̃^{vv}` the Gamma path solves, with `χ⁰` accumulated over the mesh.
/// So the whole Brillouin zone is coupled through one small real linear system.
pub(crate) fn kpoint_second_order_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    first: &KFirstOrder,
    v: &[f64],
) -> Result<KSecondOrder> {
    let skeletons = kpoint_mesh_skeletons(system, params, scf, options, pbc)?;
    kpoint_second_order_directional_cached(
        system, params, scf, options, pbc, &skeletons, first, v,
    )
}

/// The production k-point skeleton derivative set at **every** k-point of the mesh.
///
/// Direction-independent (it is the full per-DOF `(∂F(k)/∂R_y, ∂S(k)/∂R_y)` table), so
/// [`KpointThirdReference`] builds it once and every direction reuses it. That matters: it is the
/// single most expensive per-k object the second-order response needs, and rebuilding it per
/// direction would dominate the dense driver.
fn kpoint_mesh_skeletons(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<Vec<crate::pbc::hessian::KpointSkeleton>> {
    scf.kpoints
        .iter()
        .map(|kp| {
            crate::pbc::hessian::kpoint_skeleton_derivatives(
                system,
                params,
                scf,
                options,
                pbc,
                kp.fractional,
            )
        })
        .collect()
}

/// [`kpoint_second_order_directional`] against a precomputed per-k skeleton table.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kpoint_second_order_directional_cached(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    skeletons: &[crate::pbc::hessian::KpointSkeleton],
    first: &KFirstOrder,
    v: &[f64],
) -> Result<KSecondOrder> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic third derivative requires a lattice");
    let basis = &scf.basis;
    let n = basis.len();
    let nsh = basis.shells.len();
    let nat = system.atoms.len();
    let nk = scf.kpoints.len();
    if v.len() != 3 * nat {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "kpoint_second_order_directional: direction length {} != 3*natoms {}",
            v.len(),
            3 * nat
        )));
    }
    let kt = scf.electronic_temperature.max(0.0) * crate::constants::KB_HARTREE_PER_K;
    let kernel = crate::pbc::hessian::periodic_response_kernel(scf);

    // ---- first-order inputs shared by every k-point ----
    let q1 = &first.q;
    let vscr1 = crate::linalg::matrix_vector_product(&kernel, q1)?;

    // ---- the frozen-charge potential ladder and the second-order skeleton pair ----
    let v1_pot = kpoint_shell_potential_first_directional(system, params, scf, options, pbc, v)?;
    let v2_pot = shell_potential_second_directional(system, &lattice, scf, pbc, v);

    // ---- geometric kernel motion at the response charges: (∂γ/∂R·v)·q¹ ----
    // The k-point twin of the Gamma assembly's `dgamma_v_q1`: the shell potential is a property
    // of the converged charge density and the lattice, so the SAME builder evaluated on an SCC
    // whose charges have been replaced by `q¹` is the `q¹` channel of `∂V/∂R`.
    let dgamma_v_q1 = {
        let mut doctored = scf.clone();
        doctored.shell_charges = q1.clone();
        let mut atom_q = vec![0.0_f64; nat];
        for (ish, shell) in basis.shells.iter().enumerate() {
            atom_q[shell.atom_index] += q1[ish];
        }
        doctored.atomic_charges = atom_q;
        kpoint_shell_potential_first_directional(system, params, &doctored, options, pbc, v)?
    };

    // ---- ∂K/∂q chain: the third-order on-site charge kernel differentiated along q¹ ----
    let chain = kernel_chain_potential(scf, options.charge_order, q1, q1);

    // dv_ext = (∂_y γ)q^x + (∂_x γ)q^y + chain, with x = y = v.
    let dv_ext: Vec<f64> = (0..nsh)
        .map(|s| 2.0 * dgamma_v_q1[s] + chain[s])
        .collect();

    // ---- per-k reference data ----
    let mut refs: Vec<KRefData> = Vec::with_capacity(nk);
    for ik in 0..nk {
        let mos = crate::pbc::hessian::kpoint_complex_mos(scf, ik)?;
        let kfrac = scf.kpoints[ik].fractional;
        let sk = &skeletons[ik];
        let mut f1 = CMatrix::zeros(n);
        let mut s1 = CMatrix::zeros(n);
        for (y, &vy) in v.iter().enumerate() {
            if vy == 0.0 {
                continue;
            }
            caxpy(&mut f1, vy, &sk.fock[y]);
            caxpy(&mut s1, vy, &sk.overlap[y]);
        }
        // Total first-order Fock derivative: skeleton + the screening the SCC charge response
        // induces. This is `FirstOrderField::h_tilde`'s AO preimage.
        let rf1 = cscalar_response_fock(basis, &mos.overlap, &vscr1);
        let mut df1 = f1;
        for i in 0..n {
            for j in 0..n {
                df1.re[(i, j)] += rf1.re[(i, j)];
                df1.im[(i, j)] += rf1.im[(i, j)];
            }
        }
        let a_x = mo_transform_k(&mos.coeff, &df1);
        let s_x = mo_transform_k(&mos.coeff, &s1);

        // Second-order skeleton pair at frozen charges.
        let (f_vv, s_vv) = kpoint_directional_second_matrices(
            system, params, scf, options, pbc, &v1_pot, &v2_pot, v, kfrac,
        )?;
        // dF_ext = F^{vv}_skel + RF_S(dv_ext) + 2·RF_{S¹}(K q¹).
        let rf_ext = cscalar_response_fock(basis, &mos.overlap, &dv_ext);
        let rf_s1 = cscalar_response_fock(basis, &s1, &vscr1);
        let mut df_ext = f_vv;
        for i in 0..n {
            for j in 0..n {
                df_ext.re[(i, j)] += rf_ext.re[(i, j)] + 2.0 * rf_s1.re[(i, j)];
                df_ext.im[(i, j)] += rf_ext.im[(i, j)] + 2.0 * rf_s1.im[(i, j)];
            }
        }
        let s_xy = mo_transform_k(&mos.coeff, &s_vv);
        let tables = DkTablesK::build(&mos.energies, &mos.occupations, kt);
        refs.push(KRefData {
            mos,
            tables,
            weight: scf.kpoints[ik].weight,
            s1,
            a_x,
            s_x,
            s_xy,
            s_vv,
            df_ext,
        });
    }

    // ---- Brillouin-zone-wide chemical-potential response μ^v ----
    // One scalar for the whole mesh: μ is the single global Fermi level's response, weighted by
    // the mesh-summed Fermi weight Σ_k w_k Σ_p (−f'_kp). Zero for a gapped reference.
    let mut w_sum = 0.0;
    let mut mu_num = 0.0;
    for r in &refs {
        let nb = r.tables.len();
        for p in 0..nb {
            let w = r.tables.mu_weight(p);
            let eps_first = r.a_x.re[(p, p)] - r.tables.eps[p] * r.s_x.re[(p, p)];
            w_sum += r.weight * w;
            mu_num += r.weight * w * eps_first;
        }
    }
    let mu_x = if w_sum > 0.0 { mu_num / w_sum } else { 0.0 };

    // ---- the response at a given AO second-order Fock derivative ----
    // Mirrors the molecular `dk_response` closure, with the particle-number condition summed
    // over the mesh instead of evaluated on a single spectrum.
    let dk_response = |a_xy_ao: &[CMatrix]| -> (Vec<CMatrix>, Vec<CMatrix>) {
        let mut p_mo: Vec<CMatrix> = Vec::with_capacity(nk);
        let mut w_mo: Vec<CMatrix> = Vec::with_capacity(nk);
        let mut target = 0.0;
        let mut tr = 0.0;
        for (ik, r) in refs.iter().enumerate() {
            let a_xy = mo_transform_k(&r.mos.coeff, &a_xy_ao[ik]);
            let pk = kpoint_dk_second_order_mo(
                &r.tables, 0, &r.a_x, &r.s_x, &r.a_x, &r.s_x, &a_xy, &r.s_xy, mu_x, mu_x,
            );
            let wk = kpoint_dk_second_order_mo(
                &r.tables, 1, &r.a_x, &r.s_x, &r.a_x, &r.s_x, &a_xy, &r.s_xy, mu_x, mu_x,
            );
            // Particle number: d² Σ_k w_k Tr[S(k)P(k)] = 0 fixes the single μ^{xy}.
            let p1 = kpoint_dk_first_order_mo(&r.tables, 0, &r.a_x, &r.s_x, mu_x);
            let nb = r.tables.len();
            let mut tgt = 0.0;
            for p in 0..nb {
                tgt -= r.tables.f[p] * r.s_xy.re[(p, p)];
                for q in 0..nb {
                    // 2·Re Tr[S^v P^{1v}] (both cross orderings coincide for x = y = v).
                    tgt -= 2.0
                        * (r.s_x.re[(p, q)] * p1.re[(q, p)] - r.s_x.im[(p, q)] * p1.im[(q, p)]);
                }
                tr += r.weight * pk.re[(p, p)];
            }
            target += r.weight * tgt;
            p_mo.push(pk);
            w_mo.push(wk);
        }
        let mu_xy = if w_sum > 0.0 {
            (target - tr) / w_sum
        } else {
            0.0
        };
        for (ik, r) in refs.iter().enumerate() {
            let nb = r.tables.len();
            for p in 0..nb {
                let w = r.tables.mu_weight(p);
                p_mo[ik].re[(p, p)] += mu_xy * w;
                w_mo[ik].re[(p, p)] += mu_xy * r.tables.eps[p] * w;
            }
        }
        (p_mo, w_mo)
    };

    // ---- ext pass ----
    let df_ext_all: Vec<CMatrix> = refs.iter().map(|r| r.df_ext.clone()).collect();
    let (p_mo_ext, _) = dk_response(&df_ext_all);
    let p_ao_ext: Vec<CMatrix> = refs
        .iter()
        .zip(&p_mo_ext)
        .map(|(r, m)| mo_to_ao_k(&r.mos.coeff, m))
        .collect();

    // ---- dielectric solve for q^{vv} ----
    // q̃^{vv} = −Σ_k w_k [ Tr_s(P^{vv}_ext S) + Tr_s(P₀ S^{vv}) + 2 Tr_s(P¹ S¹) ], then
    // (I − χ⁰K) q^{vv} = q̃^{vv}.
    let mut q_tilde = vec![0.0_f64; nsh];
    for (ik, r) in refs.iter().enumerate() {
        accumulate_shell_trace(&mut q_tilde, basis, r.weight, &p_ao_ext[ik], &r.mos.overlap);
        accumulate_shell_trace(&mut q_tilde, basis, r.weight, &scf.density_k[ik], &r.s_vv);
        accumulate_shell_trace(&mut q_tilde, basis, 2.0 * r.weight, &first.p_k[ik], &r.s1);
    }
    let chi0 = kpoint_static_susceptibility(scf, &refs, w_sum)?;
    let dielectric = crate::pbc::hessian::PeriodicChargeDielectric::build(&chi0, &kernel)?;
    let q_xy = dielectric.solve(&q_tilde)?;

    // ---- final pass with the full dv (including K q^{vv}) ----
    let kq_xy = crate::linalg::matrix_vector_product(&kernel, &q_xy)?;
    let df_all: Vec<CMatrix> = refs
        .iter()
        .map(|r| {
            let rf = cscalar_response_fock(basis, &r.mos.overlap, &kq_xy);
            let mut m = r.df_ext.clone();
            for i in 0..n {
                for j in 0..n {
                    m.re[(i, j)] += rf.re[(i, j)];
                    m.im[(i, j)] += rf.im[(i, j)];
                }
            }
            m
        })
        .collect();
    let (p_mo, w_mo) = dk_response(&df_all);
    let p_k: Vec<CMatrix> = refs
        .iter()
        .zip(&p_mo)
        .map(|(r, m)| mo_to_ao_k(&r.mos.coeff, m))
        .collect();
    let w_k: Vec<CMatrix> = refs
        .iter()
        .zip(&w_mo)
        .map(|(r, m)| mo_to_ao_k(&r.mos.coeff, m))
        .collect();

    let p_images = realspace_images(scf, &lattice, pbc.ao_cutoff, &p_k);
    let w_images = realspace_images(scf, &lattice, pbc.ao_cutoff, &w_k);
    Ok(KSecondOrder {
        p_images,
        w_images,
        q: q_xy,
        p_k,
        w_k,
    })
}

/// `∂K/∂q` chain potential: the third-order on-site charge kernel differentiated along one charge
/// response and contracted with another. Reproduces
/// `response::charge_space::ChargeSpaceContext::kernel_chain_potential` from the periodic SCC's
/// own shell model, since that method is not reachable from here.
fn kernel_chain_potential(
    scf: &PbcSccResult,
    charge_order: usize,
    u: &[f64],
    q_y: &[f64],
) -> Vec<f64> {
    let model = &scf.shell_model;
    let nat = scf.atomic_charges.len();
    let nsh = scf.basis.shells.len();
    let mut shell_atom = vec![0_usize; nsh];
    for (ish, shell) in scf.basis.shells.iter().enumerate() {
        shell_atom[ish] = shell.atom_index;
    }
    let mut kernel_q_atom = vec![0.0_f64; nat];
    for (atom, kq) in kernel_q_atom.iter_mut().enumerate() {
        if model.atom_shell_counts[atom] == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
            model.hardness[offset],
            model.hubbard_derivs[offset],
            charge_order,
            scf.atomic_charges[atom],
        );
        *kq = third;
    }
    let mut atom_u = vec![0.0_f64; nat];
    let mut atom_qy = vec![0.0_f64; nat];
    for s in 0..nsh {
        atom_u[shell_atom[s]] += u[s];
        atom_qy[shell_atom[s]] += q_y[s];
    }
    (0..nsh)
        .map(|s| {
            let a = shell_atom[s];
            kernel_q_atom[a] * atom_qy[a] * atom_u[a]
        })
        .collect()
}

/// **Static charge susceptibility `χ⁰` over the mesh**: the shell charges induced by a unit
/// potential on each shell, at frozen geometry and without screening.
///
/// Same definition as the molecular `ChargeSpaceContext::build`'s `χ⁰` loop, evaluated with the
/// first-order resolvent form so the susceptibility and the second-order response come from one
/// algebra. A unit shell potential is a pure Fock perturbation (`S^{(1)} = 0`), so the
/// particle-number condition reduces to `Σ_k w_k Tr[P^{(1)}(k)] = 0`, which is what fixes `μ`
/// here; for a gapped reference every Fermi weight vanishes and the term is absent.
fn kpoint_static_susceptibility(
    scf: &PbcSccResult,
    refs: &[KRefData],
    w_sum: f64,
) -> Result<Matrix> {
    let basis = &scf.basis;
    let nsh = basis.shells.len();
    let n = basis.len();
    let zero = CMatrix::zeros(n);
    let mut chi0 = Matrix::zeros(nsh, nsh);
    for t in 0..nsh {
        let mut unit = vec![0.0_f64; nsh];
        unit[t] = 1.0;
        let mut mo: Vec<CMatrix> = Vec::with_capacity(refs.len());
        let mut tr = 0.0;
        for r in refs {
            let rf = cscalar_response_fock(basis, &r.mos.overlap, &unit);
            let a_u = mo_transform_k(&r.mos.coeff, &rf);
            let m = kpoint_dk_first_order_mo(&r.tables, 0, &a_u, &zero, 0.0);
            for p in 0..r.tables.len() {
                tr += r.weight * m.re[(p, p)];
            }
            mo.push(m);
        }
        let mu = if w_sum > 0.0 { -tr / w_sum } else { 0.0 };
        let mut dq = vec![0.0_f64; nsh];
        for (ik, r) in refs.iter().enumerate() {
            if mu != 0.0 {
                for p in 0..r.tables.len() {
                    mo[ik].re[(p, p)] += mu * r.tables.mu_weight(p);
                }
            }
            let dp = mo_to_ao_k(&r.mos.coeff, &mo[ik]);
            accumulate_shell_trace(&mut dq, basis, r.weight, &dp, &r.mos.overlap);
        }
        for s in 0..nsh {
            chi0[(s, t)] = dq[s];
        }
    }
    Ok(chi0)
}

// ---------------------------------------------------------------------------------------------
// Shared reference, assembly, and the public analytic entry points
// ---------------------------------------------------------------------------------------------

/// Occupations closer than this to 0 or 2 count as integer. Mirrors
/// `pbc::gamma_response`'s own epsilon.
const FRACTIONAL_OCC_EPS: f64 = 1.0e-10;

/// **Shared, direction-INDEPENDENT reference state** for the analytic Brillouin-zone-sampled
/// third derivative: one k-mesh SCC, one full per-DOF CPXTB response sweep, the real-space ground
/// densities, the response band-pair table, the screening kernel, the coordination derivatives
/// and the frozen-charge `∂V_s/∂R` table.
///
/// The k-point analogue of [`crate::pbc::GammaThirdReference`], with one extra and rather large
/// item hoisted: the per-DOF first-order responses `(∂P/∂R_y, ∂W/∂R_y, ∂q/∂R_y)` over the whole
/// mesh. Those are `3N` coupled-perturbed solves and they do **not** depend on the direction, so
/// the dense / block drivers pay for them once and every subsequent direction is a contraction
/// (see [`kpoint_first_order_contract`]). What genuinely stays per-direction is listed on
/// [`pbc_kpoint_third_with_reference`].
pub struct KpointThirdReference {
    scf: PbcSccResult,
    dens0: FrozenDensityImages,
    band_pairs: Vec<ResponseBandPair>,
    kernel: Matrix,
    cn: Option<crate::coordination::CoordinationDerivatives>,
    lattice: Lattice,
    /// `∂P(k)/∂R_y` for every DOF `y` and k-point.
    dp: Vec<Vec<CMatrix>>,
    /// `∂W(k)/∂R_y`.
    dw: Vec<Vec<CMatrix>>,
    /// `∂q_s/∂R_y`.
    dq: Vec<Vec<f64>>,
    /// The production per-DOF skeleton derivatives at every k-point of the mesh.
    skeletons: Vec<crate::pbc::hessian::KpointSkeleton>,
    /// `∂V_s/∂R_col` at the converged (frozen) shell charges — the `V₁` builder, hoisted.
    dv0: Vec<Vec<f64>>,
    nat: usize,
    ndof: usize,
}

impl KpointThirdReference {
    /// Build the shared reference, rejecting every option set the analytic k-point assembly does
    /// not cover.
    ///
    /// Guards, in cheap-to-expensive order: the term registry at analytic order 3 (which rejects
    /// multipole / long-range exchange / DFT+U / spin polarization / external fields /
    /// experimental D4, all capped at order 1), a lattice, SCC convergence, and — only knowable
    /// after the SCC — integer occupations at **every** k-point of the mesh.
    ///
    /// There is deliberately **no k-mesh restriction**: that is the whole point of this path.
    /// A Gamma-only mesh is accepted and reproduces
    /// [`crate::pbc::pbc_gamma_third_analytic_vector`] (gated in
    /// `tests/pbc_kpoint_third_analytic.rs`).
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> Result<Self> {
        crate::terms::require_order(
            options,
            params,
            3,
            "the analytic k-point periodic third derivative",
        )?;
        let lattice = *system.lattice.as_ref().ok_or_else(|| {
            crate::error::Gfn1Error::InvalidInput("periodic third needs a lattice".into())
        })?;
        let scf = crate::pbc::scf::run_pbc_scc(system, params, options, pbc)?;
        if !scf.converged {
            return Err(crate::error::Gfn1Error::InvalidInput(
                "periodic SCC did not converge for the analytic k-point third derivative".into(),
            ));
        }
        for ik in 0..scf.kpoints.len() {
            let mos = crate::pbc::hessian::kpoint_complex_mos(&scf, ik)?;
            if mos
                .occupations
                .iter()
                .any(|&f| f > FRACTIONAL_OCC_EPS && f < 2.0 - FRACTIONAL_OCC_EPS)
            {
                return Err(crate::error::Gfn1Error::InvalidInput(
                    "the analytic k-point periodic third derivative requires integer (gapped) \
                     occupations, but the periodic SCC converged to a Fermi-smeared filling; set \
                     ElectronicOptions::electronic_temperature = 0, or use \
                     pbc_kpoint_third_derivative_seminumerical_* which supports fractional \
                     occupations"
                        .into(),
                ));
            }
        }
        let dens0 = gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff);
        let band_pairs = build_response_band_pairs(system, params, &scf, &dens0.p, pbc)?;
        let kernel = periodic_response_kernel(&scf);
        let cn = if options.hamiltonian.enable_cn_hamiltonian {
            Some(coordination_with_derivatives(
                system,
                CoordinationOptions {
                    cutoff: options.hamiltonian.coordination_cutoff,
                    ..CoordinationOptions::default()
                },
            )?)
        } else {
            None
        };
        let (dp, dw, dq) =
            kpoint_cpxtb_density_responses(system, params, &scf, options, pbc, true)?;
        let skeletons = kpoint_mesh_skeletons(system, params, &scf, options, pbc)?;
        let dv0 = shell_potential_derivatives(system, &lattice, &scf, pbc)?;
        let nat = system.atoms.len();
        Ok(Self {
            scf,
            dens0,
            band_pairs,
            kernel,
            cn,
            lattice,
            dp,
            dw,
            dq,
            skeletons,
            dv0,
            nat,
            ndof: 3 * nat,
        })
    }

    /// The converged k-mesh SCC the reference was built on.
    pub fn scc(&self) -> &PbcSccResult {
        &self.scf
    }
}

/// **Analytic Brillouin-zone-sampled directional third derivative** `e³[v]` against a shared
/// [`KpointThirdReference`], assembled in exactly the shape the Gamma path uses
/// ([`crate::pbc::pbc_gamma_third_with_reference`]):
///
/// ```text
///   e³[v] = frozen third                      (kpoint_frozen_third_directional, all blocks)
///         + density path                      (∂frozen²/∂X₀ · X¹)
///         + g(X²)·v                           (response gradient, second-order slots)
///         + B6(X¹)[v,v] + 2·bg4(X¹, X¹)·v     (gamma_response_path_directional)
/// ```
///
/// # Why the Gamma consumers work unchanged
///
/// Every consumer above reads the electronic state through exactly two objects: **real-space
/// density images** `P(T)`, `W(T)` and **real shell charges** `q`. Both are Brillouin-zone
/// invariants — `M(T) = Σ_k w_k Re[M(k) e^{-ik·T}]` is a real matrix per image, and Mulliken
/// charges are real by construction. So `X¹` and `X²` are handed over in the same real-space
/// image convention the Gamma path uses ([`KFirstOrder`] / [`KSecondOrder`]), and the frozen
/// builders, `response_gradient` and `gamma_response_path_directional` cannot tell the difference.
/// The k-dependence is confined to the two response solves that produce those images.
///
/// # Cost note — what stays per-direction
///
/// The frozen bundle, the `V₁/V₂` potential ladders, the `X¹` contraction, the whole second-order
/// response (which rebuilds `nk` complex skeletons and `nk` second-order skeleton pairs), the
/// response gradient and both paths. What is *not* per-direction — the `3N` coupled solves, the
/// SCC, the band pairs, the kernel — lives in the reference. Unlike the Gamma path there is no
/// second `O(N²)` skeleton build per direction: the k-point `dγ_v·q¹` needs only
/// [`kpoint_shell_potential_first_directional`], which is a pure charge/lattice object.
pub fn pbc_kpoint_third_with_reference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    reference: &KpointThirdReference,
    v: &[f64],
) -> Result<f64> {
    let KpointThirdReference {
        scf,
        dens0,
        band_pairs,
        kernel,
        cn,
        lattice,
        dp,
        dw,
        dq,
        skeletons,
        dv0,
        nat,
        ndof,
    } = reference;
    let (nat, ndof) = (*nat, *ndof);
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "pbc_kpoint_third_with_reference: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let coordination_cutoff = options.hamiltonian.coordination_cutoff;
    let nsh = scf.basis.shells.len();

    // ---- frozen third (all blocks, incl. the Ewald SCC2 sums) ----
    let frozen = kpoint_frozen_third_directional(
        system,
        params,
        scf,
        options,
        pbc,
        options.enable_dispersion,
        options.d3_reference_path.as_deref(),
        v,
    )?;

    // ---- the frozen-charge potential ladder ----
    let mut v1_pot = vec![0.0_f64; nsh];
    for (col, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        for (s, o) in v1_pot.iter_mut().enumerate() {
            *o += vc * dv0[col][s];
        }
    }
    let v2_pot = shell_potential_second_directional(system, lattice, scf, pbc, v);

    // ---- X¹: the cached per-DOF response contracted with `v` ----
    let x1 = kpoint_first_order_contract(scf, lattice, pbc.ao_cutoff, dp, dw, dq, v);
    let dens1 = FrozenDensityImages {
        p: x1.p_images.clone(),
        w: x1.w_images.clone(),
    };

    // ---- X²: the complex resolvent second-order response over the mesh ----
    let x2 = kpoint_second_order_directional_cached(
        system, params, scf, options, pbc, skeletons, &x1, v,
    )?;

    // ---- g(X²)·v ----
    let g2_grad = response_gradient(
        system,
        params,
        scf,
        band_pairs,
        DensityLookup::Images(&x2.p_images),
        DensityLookup::Images(&x2.w_images),
        &x2.q,
        kernel,
        pbc,
        cn.as_ref(),
    )?;
    let g2: f64 = g2_grad
        .iter()
        .enumerate()
        .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
        .sum();

    // ---- B6 + bg4 ----
    let path = gamma_response_path_directional(
        system, params, scf, options, pbc, dens0, &dens1, &x1.q, &v1_pot, v,
    )?;

    // ---- density path: ∂frozen²/∂X₀ · X¹ ----
    // Identical inventory to the Gamma assembly (see `pbc_gamma_third_with_reference`): the
    // Hessian-shaped frozen blocks with `X¹` in the frozen slots — band/Pulay WITH the
    // frozen-charge potential legs, the CN block in its two-sided Hessian convention, the SCC2
    // charge-path bilinear, and the `V(q₀)`-cache motion via the Δ-potential trick.
    let density_path = {
        let mut acc = pbc_band_pulay_third_directional(
            system, params, scf, pbc, &dens1, &v1_pot, &v2_pot, v,
        )?
        .second;
        if enable_cn {
            acc += pbc_cn_third_directional(
                system,
                params,
                scf,
                pbc,
                coordination_cutoff,
                &dens1.p,
                v,
            )?
            .second;
        }
        acc += 2.0
            * pbc_scc2_bilinear_second_directional(
                system,
                lattice,
                scf,
                pbc,
                &scf.shell_charges,
                &x1.q,
                v,
            );
        // V(q₀)-cache motion: value shift K·q¹ plus its geometric legs at q¹.
        let kq1 = crate::linalg::matrix_vector_product(kernel, &x1.q)?;
        let mut doctored = scf.clone();
        doctored.shell_charges = x1.q.clone();
        let mut atom_q = vec![0.0_f64; nat];
        for (ish, shell) in scf.basis.shells.iter().enumerate() {
            atom_q[shell.atom_index] += x1.q[ish];
        }
        doctored.atomic_charges = atom_q;
        let dgamma_v_q1 =
            kpoint_shell_potential_first_directional(system, params, &doctored, options, pbc, v)?;
        let dgamma2_v_q1 = shell_potential_second_directional(system, lattice, &doctored, pbc, v);
        let mut scf_shift = scf.clone();
        for (s, dv) in scf_shift.shell_scc_potential.iter_mut().zip(&kq1) {
            *s += dv;
        }
        let v1_shift: Vec<f64> = v1_pot.iter().zip(&dgamma_v_q1).map(|(a, b)| a + b).collect();
        let v2_shift: Vec<f64> = v2_pot
            .iter()
            .zip(&dgamma2_v_q1)
            .map(|(a, b)| a + b)
            .collect();
        let shifted = pbc_band_pulay_third_directional(
            system, params, &scf_shift, pbc, dens0, &v1_shift, &v2_shift, v,
        )?
        .second;
        // The unshifted reference is the frozen bundle's own band/Pulay `second`: same `scf`,
        // same ground densities, same `(v1_pot, v2_pot)`, so recomputing it would be a
        // bit-for-bit duplicate sweep.
        acc += shifted - frozen.band_pulay.second;
        acc
    };

    if std::env::var("GFN1_K3_DEBUG").is_ok() {
        eprintln!(
            "k3 components: frozen {:+.10e}  dpath {density_path:+.10e}  g2 {g2:+.10e}  b6 \
             {:+.10e}  bg4 {:+.10e}\n  b6 blocks {:?}\n  bg4 families {:?}",
            frozen.total().third,
            path.b6,
            path.bg4,
            path.b6_blocks,
            path.bg4_families
        );
    }
    Ok(frozen.total().third + density_path + g2 + path.b6 + 2.0 * path.bg4)
}

/// **Analytic k-point periodic third derivative, Vector mode**: the single contracted scalar
/// `e³[v] = Σ_abc T_abc v_a v_b v_c` along one direction.
///
/// The cheapest output mode — one shared-reference build plus one directional evaluation. Unlike
/// the Gamma entry points this accepts **any** Monkhorst-Pack mesh; see
/// [`KpointThirdReference::build`] for the option coverage (integer occupations, analytic order 3
/// terms) and [`pbc_kpoint_third_with_reference`] for the assembly.
pub fn pbc_kpoint_third_analytic_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    v: &[f64],
) -> Result<f64> {
    let reference = KpointThirdReference::build(system, params, options, pbc)?;
    pbc_kpoint_third_with_reference(system, params, options, pbc, &reference, v)
}

/// **Dense mode**: the full packed `T_abc` recovered from directional evaluations by the cubic
/// polarization identity
/// `T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]`, mirroring the Gamma
/// driver: the subset directions of every canonical triple are deduplicated, evaluated once each
/// in parallel against ONE shared [`KpointThirdReference`], and recombined.
///
/// Cost is `~C(3N+2, 3)` directional evaluations and grows as `N³`, with each evaluation itself
/// `nk` times more expensive than its Gamma counterpart — prefer
/// [`pbc_kpoint_third_analytic_block`] or the vector mode for anything but small cells.
pub fn pbc_kpoint_third_analytic_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<SymmetricThird> {
    let dofs: Vec<usize> = (0..3 * system.atoms.len()).collect();
    kpoint_third_polarized(system, params, options, pbc, &dofs)
}

/// **Block mode**: the `|dofs|³` sub-tensor of the dense analytic k-point third derivative,
/// indexed by POSITION in `dofs`, via the same polarization driver — only the directions the
/// requested triples actually need are evaluated.
pub fn pbc_kpoint_third_analytic_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    dofs: &[usize],
) -> Result<SymmetricThird> {
    let ndof = 3 * system.atoms.len();
    for &d in dofs {
        if d >= ndof {
            return Err(crate::error::Gfn1Error::InvalidInput(format!(
                "pbc_kpoint_third_analytic_block: dof {d} out of range (ndof {ndof})"
            )));
        }
    }
    kpoint_third_polarized(system, params, options, pbc, dofs)
}

/// The shared polarization driver behind the dense and block modes.
fn kpoint_third_polarized(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    dofs: &[usize],
) -> Result<SymmetricThird> {
    use rayon::prelude::*;
    use std::collections::BTreeMap;

    let reference = KpointThirdReference::build(system, params, options, pbc)?;
    let ndof = 3 * system.atoms.len();
    let m = dofs.len();

    // Phase 1: deduplicate the subset directions of every canonical triple.
    let mut key_index: HashMap<Vec<(usize, u8)>, usize> = HashMap::new();
    let mut keys: Vec<Vec<(usize, u8)>> = Vec::new();
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

    // Phase 2: evaluate each distinct direction once, in parallel, against the shared reference.
    let values: Result<Vec<f64>> = keys
        .par_iter()
        .map(|key| {
            let mut v = vec![0.0_f64; ndof];
            for &(dof, weight) in key {
                v[dof] = weight as f64;
            }
            pbc_kpoint_third_with_reference(system, params, options, pbc, &reference, &v)
        })
        .collect();
    let values = values?;

    // Phase 3: assemble the packed tensor.
    let mut store = SymmetricThird::zeros(m);
    for ((i, j, k), terms) in plan {
        let mut t = 0.0;
        for (sign, idx) in terms {
            t += sign * values[idx];
        }
        store.add(i, j, k, t / 6.0);
    }
    Ok(store)
}

// ---------------------------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::hessian::kpoint_skeleton_derivatives;
    use crate::pbc::scf::run_pbc_scc;
    use crate::pbc::{EwaldOptions, KMesh};

    /// The same skewed primitive diamond cell the Gamma frozen-block gates use: an asymmetric
    /// cell strain (no residual point group, so no block can pass by accidental cancellation)
    /// plus an internal displacement that lifts diamond's triply degenerate `t2` frontier
    /// orbitals — mandatory here, because the k-point CPXTB is documented (`docs/pbc.md` §2b) to
    /// be wrong on an exactly degenerate frontier manifold.
    const DIAMOND_SKEW: &str = "2\n\
Lattice=\"0.06 1.83 1.75 1.75 0.04 1.81 1.82 1.76 0.03\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.930000 0.880000 0.905000\n";

    /// The heteronuclear partner fixture, the same skewed zincblende BN cell the Gamma gates use.
    ///
    /// Needed here for one specific reason: `DIAMOND_SKEW` has two *identical* atoms, one of them
    /// at the origin, so the cell retains inversion symmetry about the bond midpoint mapping the
    /// two sublattices onto each other. That forces every shell-charge response to cancel, and it
    /// is measured — the Gamma-equivalence gate reports `max|q^vv| = 1.6e-14` there. Diamond
    /// therefore gates the density and energy-weighted channels only; BN is what makes the
    /// charge channel (and with it the dielectric solve) non-vacuous.
    const BN_SKEW: &str = "2\n\
Lattice=\"0.06 1.86 1.78 1.78 0.04 1.84 1.85 1.79 0.03\" pbc=\"T T T\"\n\
B 0.000000 0.000000 0.000000\n\
N 0.940000 0.890000 0.920000\n";

    fn params() -> Gfn1Parameters {
        Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
    }

    /// Tight SCC with the electronic temperature pinned to exactly zero: these gates must not
    /// drift onto the periodic finite-temperature branch.
    fn electronic() -> ElectronicOptions {
        ElectronicOptions {
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            max_scc: 500,
            electronic_temperature: 0.0,
            ..ElectronicOptions::default()
        }
    }

    /// Lean real-space cutoffs, purely for test speed, over a `2 x 2 x 2` Monkhorst-Pack mesh
    /// (four k-points after time-reversal folding, all with `|k| > 0` — a Gamma-only mesh would
    /// make every gate here vacuous).
    fn kpbc_opts() -> PbcOptions {
        PbcOptions {
            kmesh: KMesh::monkhorst_pack([2, 2, 2]),
            ao_cutoff: 9.0,
            ewald: EwaldOptions {
                real_cutoff: 14.0,
                sr_cutoff: 8.0,
                ..EwaldOptions::default()
            },
        }
    }

    fn gamma_pbc_opts() -> PbcOptions {
        PbcOptions {
            kmesh: KMesh::gamma(),
            ..kpbc_opts()
        }
    }

    fn system_of(xyz: &str) -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture parse")
    }

    /// Deterministic sign-mixed direction in `[-1, 1)^ndof` (a near-uniform direction would be
    /// near a rigid translation, where every block vanishes and the gates go vacuous).
    fn direction(ndof: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..ndof)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                2.0 * ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect()
    }

    fn displaced(system: &PeriodicSystem, v: &[f64], lambda: f64) -> PeriodicSystem {
        let mut out = system.clone();
        for (a, atom) in out.atoms.iter_mut().enumerate() {
            atom.position =
                atom.position + Vec3::new(v[3 * a], v[3 * a + 1], v[3 * a + 2]) * lambda;
        }
        out
    }

    fn central(f: &dyn Fn(f64) -> f64, h: f64) -> f64 {
        (f(h) - f(-h)) / (2.0 * h)
    }

    /// The standard ladder gate: the central difference of a block's directional **second**
    /// derivative must reproduce the analytic **third**, with the error shrinking by ~4 when the
    /// step halves. The magnitude test uses the Richardson extrapolant so it is not a disguised
    /// statement about `h`; the ratio is the real discriminator (flat ⇒ a missing term,
    /// falling ⇒ the round-off floor).
    fn assert_ladder(name: &str, f: &dyn Fn(f64) -> f64, analytic: f64, h: f64, rel_tol: f64, abs_tol: f64) {
        let fd_h = central(f, h);
        let fd_h2 = central(f, 0.5 * h);
        let e1 = (fd_h - analytic).abs();
        let e2 = (fd_h2 - analytic).abs();
        let ratio = if e2 > 0.0 { e1 / e2 } else { f64::INFINITY };
        let richardson = (4.0 * fd_h2 - fd_h) / 3.0;
        let e_rich = (richardson - analytic).abs();
        println!(
            "{name:<52} analytic={analytic:+.10e} err(h)={e1:.3e} err(h/2)={e2:.3e} \
             ratio={ratio:.2} richardson_err={e_rich:.3e}"
        );
        let bound = rel_tol * analytic.abs() + abs_tol;
        assert!(
            e_rich <= bound,
            "{name}: |richardson - analytic| = {e_rich:.3e} exceeds {bound:.3e} \
             (analytic {analytic:.10e}, richardson {richardson:.10e})"
        );
        assert!(
            e2 <= 1.0e-11 * analytic.abs().max(1.0e-6) || ratio >= 2.5,
            "{name}: h-ladder ratio {ratio:.2} is below 2.5 (errors {e1:.3e} -> {e2:.3e}); \
             a flat ratio means a missing term in the analytic third, a falling one means the \
             difference has hit the round-off floor"
        );
    }

    /// Element-wise ladder for a complex matrix-valued first derivative. Reports the element with
    /// the largest Richardson residual and reads the ratio off that same element.
    fn assert_cmatrix_ladder(
        name: &str,
        at: &dyn Fn(f64) -> CMatrix,
        analytic: &CMatrix,
        h: f64,
        rel_tol: f64,
    ) {
        let n = analytic.n;
        let (p1, m1) = (at(h), at(-h));
        let (p2, m2) = (at(0.5 * h), at(-0.5 * h));
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(analytic.re[(i, j)].abs()).max(analytic.im[(i, j)].abs());
            }
        }
        assert!(scale > 1.0e-10, "{name}: the matrix is identically zero, the gate is vacuous");
        let mut worst = (0.0_f64, 0.0_f64, 0.0_f64, 0usize, 0usize, 'r');
        for i in 0..n {
            for j in 0..n {
                for part in ['r', 'i'] {
                    let pick = |m: &CMatrix| {
                        if part == 'r' {
                            m.re[(i, j)]
                        } else {
                            m.im[(i, j)]
                        }
                    };
                    let a = pick(analytic);
                    let fd1 = (pick(&p1) - pick(&m1)) / (2.0 * h);
                    let fd2 = (pick(&p2) - pick(&m2)) / h;
                    let rich = (4.0 * fd2 - fd1) / 3.0;
                    let e_rich = (rich - a).abs();
                    if e_rich > worst.2 {
                        worst = ((fd1 - a).abs(), (fd2 - a).abs(), e_rich, i, j, part);
                    }
                }
            }
        }
        let ratio = if worst.1 > 0.0 { worst.0 / worst.1 } else { f64::INFINITY };
        println!(
            "{name:<52} max|analytic|={scale:.4e} worst=({},{},{}) err(h)={:.3e} \
             err(h/2)={:.3e} ratio={ratio:.2} richardson_err={:.3e}",
            worst.3, worst.4, worst.5, worst.0, worst.1, worst.2
        );
        assert!(
            worst.2 <= rel_tol * scale,
            "{name}: worst element ({}, {}, {}) Richardson residual {:.3e} exceeds {:.3e}",
            worst.3,
            worst.4,
            worst.5,
            worst.2,
            rel_tol * scale
        );
        assert!(
            worst.1 <= 1.0e-11 * scale || ratio >= 2.5,
            "{name}: h-ladder ratio {ratio:.2} at element ({}, {}, {}) is below 2.5 (errors \
             {:.3e} -> {:.3e}); a flat ratio means a missing term in the analytic second",
            worst.3,
            worst.4,
            worst.5,
            worst.0,
            worst.1
        );
    }

    /// One converged k-point fixture plus the frozen fields every block reads.
    struct KFrozen {
        system: PeriodicSystem,
        params: Gfn1Parameters,
        scf: PbcSccResult,
        opts: ElectronicOptions,
        pbc: PbcOptions,
        v: Vec<f64>,
    }

    impl KFrozen {
        fn build(xyz: &str, pbc: PbcOptions, seed: u64) -> Self {
            let system = system_of(xyz);
            let params = params();
            let opts = electronic();
            let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("periodic SCC");
            assert!(scf.converged, "fixture SCC did not converge");
            let v = direction(3 * system.atoms.len(), seed);
            Self {
                system,
                params,
                scf,
                opts,
                pbc,
                v,
            }
        }

        fn lattice(&self) -> Lattice {
            self.system.lattice.as_ref().copied().unwrap()
        }

        fn at(&self, lambda: f64) -> PeriodicSystem {
            displaced(&self.system, &self.v, lambda)
        }

        /// The frozen SCC with every geometry-dependent **reference value** refreshed at the
        /// displaced geometry, charges and densities held fixed — the k-point copy of
        /// `gamma_third`'s `Frozen::refreshed_at`, and for the same reason: three fields of
        /// `PbcSccResult` (`bloch.self_energies` through `CN`, the Bloch `S(k)` builder, and
        /// `shell_scc_potential`) are cached values of geometry-dependent quantities, so
        /// differencing the production skeleton against an *unrefreshed* `scf` would gate the
        /// freezing convention rather than the analytic second derivative.
        fn refreshed_at(&self, lambda: f64) -> (PeriodicSystem, PbcSccResult) {
            let sys = self.at(lambda);
            let mut scf = self.scf.clone();
            scf.bloch = crate::pbc::bloch::BlochBuilder::build(
                &sys,
                &scf.basis,
                &self.params,
                self.pbc.ao_cutoff,
                self.opts.hamiltonian.coordination_cutoff,
                self.opts.hamiltonian.enable_cn_hamiltonian,
            )
            .expect("Bloch rebuild");
            scf.coordination_numbers = scf.bloch.coordination_numbers.clone();
            let gamma = crate::pbc::ewald::periodic_gamma_matrix(
                &sys,
                &scf.basis,
                &scf.shell_model,
                &self.pbc.ewald,
            )
            .expect("periodic gamma");
            scf.shell_scc_potential = crate::coulomb::coulomb_energy_potential_from_matrix(
                &scf.basis,
                &scf.shell_model,
                &scf.shell_charges,
                &gamma,
            )
            .expect("frozen-charge potential")
            .shell_potential;
            scf.gamma = gamma;
            (sys, scf)
        }

        /// `(F¹(k), S¹(k))` — the production k-point skeleton at the refreshed displaced
        /// geometry, contracted with `v`. This is the object
        /// [`kpoint_directional_second_matrices`] differentiates.
        fn skeleton_directional_at(&self, lambda: f64, kfrac: [f64; 3]) -> (CMatrix, CMatrix) {
            let (sys, scf) = self.refreshed_at(lambda);
            let sk =
                kpoint_skeleton_derivatives(&sys, &self.params, &scf, &self.opts, &self.pbc, kfrac)
                    .expect("k-point skeleton");
            let n = scf.basis.len();
            let mut f1 = CMatrix::zeros(n);
            let mut s1 = CMatrix::zeros(n);
            for (y, &vy) in self.v.iter().enumerate() {
                if vy == 0.0 {
                    continue;
                }
                caxpy(&mut f1, vy, &sk.fock[y]);
                caxpy(&mut s1, vy, &sk.overlap[y]);
            }
            (f1, s1)
        }
    }

    // -- Gate 2: the shell potential is k-independent -------------------------------------------

    /// `V₁` from [`kpoint_shell_potential_first_directional`] must (a) equal the production
    /// k-point gradient path's `∂V/∂R` contracted with `v` — i.e. the `shell_potential` field
    /// `kpoint_skeleton_derivatives` itself carries, at *every* k-point of the mesh, which is the
    /// k-independence claim — and (b) be the central difference of the frozen-charge shell
    /// potential `V(q₀; R)` rebuilt at displaced geometries.
    #[test]
    #[ignore = "periodic k-point gate: one k-mesh SCC plus 4 Ewald gamma rebuilds"]
    fn kpoint_shell_potential_first_matches_production_and_fd() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 17);
        let v1 = kpoint_shell_potential_first_directional(
            &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.v,
        )
        .expect("k-point V1");
        assert!(f.scf.kpoints.len() > 1, "the mesh collapsed to a single k-point");
        // Both halves below compare `V₁` against something; if `V₁` were identically zero they
        // would both pass on nothing. `DIAMOND_SKEW`'s two carbons carry zero *atomic* charge by
        // symmetry, but their s/p **shell** charges do not cancel, so the scalar potential and
        // its gradient are live — measured, not assumed.
        let v1_scale = v1.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        println!("kpoint_V1: max|V1| {v1_scale:.4e}");
        assert!(
            v1_scale > 1.0e-6,
            "V1 is identically zero, so both halves of this gate are vacuous"
        );

        // (a) equality with the production skeleton, at every k of the mesh.
        let mut worst_prod = 0.0_f64;
        for kp in &f.scf.kpoints {
            let sk = kpoint_skeleton_derivatives(
                &f.system,
                &f.params,
                &f.scf,
                &f.opts,
                &f.pbc,
                kp.fractional,
            )
            .expect("k-point skeleton");
            for s in 0..v1.len() {
                let mut acc = 0.0;
                for (y, &vy) in f.v.iter().enumerate() {
                    acc += vy * sk.shell_potential[y][s];
                }
                worst_prod = worst_prod.max((acc - v1[s]).abs());
            }
        }
        println!("kpoint_V1/production: worst |analytic - skeleton| {worst_prod:.3e}");
        assert!(
            worst_prod < 1.0e-14,
            "V1 differs from the production k-point skeleton path: {worst_prod:.3e}"
        );

        // (b) central difference of the frozen-charge shell potential itself.
        let pot_at = |lam: f64| -> Vec<f64> {
            let sys = f.at(lam);
            let gamma = crate::pbc::ewald::periodic_gamma_matrix(
                &sys,
                &f.scf.basis,
                &f.scf.shell_model,
                &f.pbc.ewald,
            )
            .expect("periodic gamma");
            crate::coulomb::coulomb_energy_potential_from_matrix(
                &f.scf.basis,
                &f.scf.shell_model,
                &f.scf.shell_charges,
                &gamma,
            )
            .expect("frozen-charge potential")
            .shell_potential
        };
        let h = 1.0e-4;
        let (pp, pm) = (pot_at(h), pot_at(-h));
        let mut worst_fd = 0.0_f64;
        for s in 0..v1.len() {
            worst_fd = worst_fd.max(((pp[s] - pm[s]) / (2.0 * h) - v1[s]).abs());
        }
        println!("kpoint_V1/fd: worst |analytic - FD| {worst_fd:.3e}");
        assert!(
            worst_fd < 1.0e-9,
            "V1 vs frozen-charge central difference: {worst_fd:.3e}"
        );
    }

    // -- Gate 1: the frozen half needs no k-point mathematics -----------------------------------

    /// **The load-bearing gate of this module.** The whole frozen bundle evaluated against a
    /// `2 x 2 x 2` k-mesh SCC: a central difference of its own directional `second` along `λ`
    /// (SCC frozen, geometry only) must reproduce its `third`. If the k-point density images
    /// broke any frozen block, the ratio would go flat here.
    #[test]
    #[ignore = "periodic k-point gate: full frozen bundle, 5 sweeps over a 2x2x2 mesh"]
    fn kpoint_frozen_third_matches_bundle_second_fd() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 23);
        assert!(f.scf.kpoints.len() > 1, "the mesh collapsed to a single k-point");
        let bundle_at = |lam: f64| {
            kpoint_frozen_third_directional(
                &f.at(lam),
                &f.params,
                &f.scf,
                &f.opts,
                &f.pbc,
                true,
                None,
                &f.v,
            )
            .expect("k-point frozen bundle")
        };
        let analytic = bundle_at(0.0).total();
        let second_at = |lam: f64| bundle_at(lam).total().second;
        assert_ladder(
            "kpoint_frozen_bundle/diamond-skew",
            &second_at,
            analytic.third,
            2.0e-3,
            1.0e-6,
            1.0e-9,
        );
    }

    /// Component-resolved companion to the bundle gate: if one block breaks under k-sampling, the
    /// bundle ladder can still pass by cancellation, so each frozen component closes its own
    /// ladder too.
    #[test]
    #[ignore = "periodic k-point gate: 5 frozen-bundle sweeps over a 2x2x2 mesh"]
    fn kpoint_frozen_components_match_their_own_second_fd() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 29);
        let bundle_at = |lam: f64| {
            kpoint_frozen_third_directional(
                &f.at(lam),
                &f.params,
                &f.scf,
                &f.opts,
                &f.pbc,
                true,
                None,
                &f.v,
            )
            .expect("k-point frozen bundle")
        };
        let at0 = bundle_at(0.0);
        let picks: [(&str, fn(&GammaFrozenThird) -> DirectionalDerivs); 4] = [
            ("coordination", |b| b.coordination),
            ("band_pulay", |b| b.band_pulay),
            ("scc2_realspace", |b| b.scc2_realspace),
            ("scc2_ewald", |b| b.scc2_ewald),
        ];
        // A component whose analytic third is ~zero has nothing to gate, but skipping it
        // SILENTLY would let the per-component coverage evaporate without saying so — the same
        // failure mode `kpoint_first_order_charges_match_reconverged_fd` had. Record and report.
        let mut ran: Vec<&str> = Vec::new();
        let mut skipped: Vec<(&str, f64)> = Vec::new();
        for (name, pick) in picks {
            let analytic = pick(&at0);
            if analytic.third.abs() < 1.0e-12 {
                skipped.push((name, analytic.third));
                continue;
            }
            ran.push(name);
            // A smaller step than the bundle gate uses. These are individually much smaller
            // numbers than their sum, and the CN counting function in particular has a large
            // fourth derivative on this cell, so `h = 2e-3` sits outside the `O(h²)` window for
            // it — the ratio overshoots 4 (the `h²` coefficient nearly cancels between the two
            // steps) and the Richardson extrapolant, which assumes that window, stops being an
            // improvement. `h = 5e-4` puts every component back inside it.
            assert_ladder(
                &format!("kpoint_frozen/{name}"),
                &|lam| pick(&bundle_at(lam)).second,
                analytic.third,
                5.0e-4,
                1.0e-6,
                1.0e-9,
            );
        }
        println!("kpoint_frozen/components: gated {ran:?}  skipped-as-zero {skipped:?}");
        assert!(
            ran.contains(&"band_pulay") && ran.contains(&"coordination"),
            "the two electronic frozen blocks must both be gated, but only {ran:?} ran"
        );
    }

    // -- Gate 3: the directional first-order response -------------------------------------------

    /// Physical gate: `q¹` from the directional k-point response must match the central FD of the
    /// **reconverged** k-mesh SCC shell charges along `v`. The k-point twin of
    /// `gamma_third`'s `gamma_first_order_charges_match_reconverged_fd`.
    ///
    /// # Why both fixtures, and why the non-vacuity assertion
    ///
    /// The homonuclear `DIAMOND_SKEW` cell is where this module's charge channel goes silent —
    /// but *only at Gamma*. `kpoint_second_order_matches_gamma_charge_space`, which drives a
    /// Gamma-only mesh, measures `max|q¹| = 9.05e-16` and `max|q^vv| = 1.60e-14` there: on that
    /// mesh diamond gates the density and energy-weighted channels only.
    ///
    /// Over the `2 x 2 x 2` mesh this gate uses, diamond's charge response is **not** small —
    /// measured `max|q¹| = 1.15e-2`, comparable to BN's `2.35e-2`. So this particular gate was
    /// never the vacuous one; the Gamma-mesh legs are. It still runs both fixtures (a
    /// heteronuclear partner exercises the Ewald/Klopman-Ohno charge path differently), and it
    /// **asserts** a non-zero `q¹` and a non-zero `P¹` rather than trusting either — which is
    /// what turns "the charge channel was tested" from an assumption into a measurement, and
    /// what would catch the day a fixture or mesh change sends the response to zero.
    #[test]
    #[ignore = "periodic k-point gate: 4 reconverged k-mesh SCCs plus 2 full CPXTB sweeps"]
    fn kpoint_first_order_charges_match_reconverged_fd() {
        let mut scales = Vec::new();
        for (label, xyz) in [("diamond-skew", DIAMOND_SKEW), ("BN-skew", BN_SKEW)] {
            let system = system_of(xyz);
            let params = params();
            let pbc = kpbc_opts();
            let mut opts = electronic();
            opts.energy_tolerance = 1.0e-13;
            opts.charge_tolerance = 1.0e-12;
            let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("tight k-mesh SCC");
            assert!(scf.converged && scf.kpoints.len() > 1);
            let v = direction(3 * system.atoms.len(), 31);
            let x1 = kpoint_first_order_directional(&system, &params, &scf, &opts, &pbc, &v)
                .expect("k-point directional response");

            let charges_at = |lam: f64| -> Vec<f64> {
                run_pbc_scc(&displaced(&system, &v, lam), &params, &opts, &pbc)
                    .expect("displaced k-mesh SCC")
                    .shell_charges
            };
            let h = 1.0e-4;
            let (cp, cm) = (charges_at(h), charges_at(-h));
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for s in 0..x1.q.len() {
                worst = worst.max(((cp[s] - cm[s]) / (2.0 * h) - x1.q[s]).abs());
                scale = scale.max(x1.q[s].abs());
            }
            println!(
                "kpoint_q1/{label}: max|q1| {scale:.4e}  worst |analytic - FD| {worst:.3e}"
            );
            assert!(
                worst < 1.0e-7,
                "[{label}] directional k-point shell-charge response vs reconverged FD: \
                 {worst:.3e}"
            );
            scales.push((label, scale));

            // The real-space images must be the inverse Bloch transform of the per-k responses,
            // and (unlike the per-k blocks) real: `P¹(T)` is what the frozen builders consume.
            let lattice = system.lattice.as_ref().copied().unwrap();
            let ref_images = realspace_images(&scf, &lattice, pbc.ao_cutoff, &x1.p_k);
            let mut worst_img = 0.0_f64;
            let mut p_scale = 0.0_f64;
            for (key, m) in &x1.p_images {
                let r = &ref_images[key];
                for i in 0..m.rows() {
                    for j in 0..m.cols() {
                        worst_img = worst_img.max((m[(i, j)] - r[(i, j)]).abs());
                        p_scale = p_scale.max(m[(i, j)].abs());
                    }
                }
            }
            assert!(p_scale > 1.0e-6, "[{label}] P1 images are zero, the gate is vacuous");
            assert!(
                worst_img == 0.0,
                "[{label}] P1 images drifted from the transform: {worst_img:.3e}"
            );
        }
        assert!(
            scales.iter().any(|&(_, s)| s > 1.0e-4),
            "no fixture produced a non-zero q1, so the charge channel went untested: {scales:?}"
        );
    }

    /// [`realspace_images`] must be the same sum `gamma_realspace_densities` applies: fed the
    /// SCF's own per-k density it has to reproduce `FrozenDensityImages::p` bit for bit.
    #[test]
    #[ignore = "periodic k-point gate: one k-mesh SCC"]
    fn kpoint_realspace_images_match_frozen_transform() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 37);
        let lattice = f.lattice();
        let dens = gamma_realspace_densities(&f.scf, &lattice, f.pbc.ao_cutoff);
        let mine = realspace_images(&f.scf, &lattice, f.pbc.ao_cutoff, &f.scf.density_k);
        assert_eq!(dens.p.len(), mine.len());
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for (key, m) in &dens.p {
            let r = &mine[key];
            for i in 0..m.rows() {
                for j in 0..m.cols() {
                    worst = worst.max((m[(i, j)] - r[(i, j)]).abs());
                    scale = scale.max(m[(i, j)].abs());
                }
            }
        }
        println!("kpoint_images: max|P(T)| {scale:.4e} worst drift {worst:.3e}");
        assert!(scale > 1.0e-6, "the density images are zero, the gate is vacuous");
        assert!(worst == 0.0, "realspace_images drifted from gamma_realspace_densities");
    }

    // -- Gate 4: the complex second-order skeleton matrices -------------------------------------

    /// Evaluated at `k = 0` the complex builder must reproduce
    /// `gamma_directional_second_matrices` element for element, with an identically zero
    /// imaginary part. This pins the phase bookkeeping of the halved image sweep *and* keeps this
    /// module's local copies of the private radial helpers from drifting.
    ///
    /// Run over **both** a Gamma-only SCF and the `2 x 2 x 2` mesh SCF used by the
    /// finite-difference gate. The second is what makes the equality load-bearing there: it says
    /// the `k = 0` leg of `kpoint_second_matrices_match_skeleton_fd` differentiates a matrix that
    /// is bit-for-bit the Gamma module's, whose own FD ladder closes at `1e-13` — so any residual
    /// that leg reports belongs to the finite-difference reference, not to this builder.
    #[test]
    #[ignore = "periodic k-point gate: 2 SCCs plus 4 second-order skeleton sweeps"]
    fn kpoint_second_matrices_match_gamma_builder() {
        use crate::pbc::gamma_third::gamma_directional_second_matrices;
        for (label, opts) in [("gamma-mesh", gamma_pbc_opts()), ("2x2x2-mesh", kpbc_opts())] {
            let f = KFrozen::build(DIAMOND_SKEW, opts, if label == "gamma-mesh" { 41 } else { 43 });
            let v1 = kpoint_shell_potential_first_directional(
                &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.v,
            )
            .expect("V1");
            let v2 =
                shell_potential_second_directional(&f.system, &f.lattice(), &f.scf, &f.pbc, &f.v);
            let (gf, gs) = gamma_directional_second_matrices(
                &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &v1, &v2, &f.v,
            )
            .expect("gamma second matrices");
            let (kf, ks) = kpoint_directional_second_matrices(
                &f.system,
                &f.params,
                &f.scf,
                &f.opts,
                &f.pbc,
                &v1,
                &v2,
                &f.v,
                [0.0, 0.0, 0.0],
            )
            .expect("k-point second matrices");

            for (name, g, k) in [("F^vv", &gf, &kf), ("S^vv", &gs, &ks)] {
                let n = g.rows();
                let mut worst_re = 0.0_f64;
                let mut worst_im = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        worst_re = worst_re.max((g[(i, j)] - k.re[(i, j)]).abs());
                        worst_im = worst_im.max(k.im[(i, j)].abs());
                        scale = scale.max(g[(i, j)].abs());
                    }
                }
                println!(
                    "kpoint_second/{label}/{name}: max|gamma|={scale:.4e} \
                     worst|Δre|={worst_re:.3e} worst|im|={worst_im:.3e}"
                );
                assert!(scale > 1.0e-8, "{name}: the Gamma matrix is zero, the gate is vacuous");
                assert!(
                    worst_re <= 1.0e-12 * scale.max(1.0) && worst_im <= 1.0e-12 * scale.max(1.0),
                    "{label}/{name}: k = 0 does not reproduce the Gamma builder \
                     (Δre {worst_re:.3e}, im {worst_im:.3e})"
                );
            }
        }
    }

    /// **General `k`.** `(F^vv(k), S^vv(k))` must be the element-wise central difference of the
    /// production k-point skeleton's `(F¹(k), S¹(k))` contracted with `v`, at frozen charges and
    /// frozen density, with the geometry-dependent reference values refreshed.
    ///
    /// Run at `k = 0` **and** at a live `k`, both against the *k-point* skeleton. The `k = 0` leg
    /// is not redundant with `kpoint_second_matrices_match_gamma_builder`: that one is an
    /// equality against another analytic builder, while this one is the physics check, so having
    /// both means a residual can be attributed to the phase bookkeeping (fails only at live `k`)
    /// or to the shared algebra / the finite-difference reference (fails at both).
    #[test]
    #[ignore = "periodic k-point gate: 8 refreshed complex skeleton evaluations"]
    fn kpoint_second_matrices_match_skeleton_fd() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 43);
        // A k-point that is genuinely away from Gamma, so the phases are live.
        let live = f
            .scf
            .kpoints
            .iter()
            .map(|kp| kp.fractional)
            .find(|k| k.iter().any(|x| x.abs() > 1.0e-8))
            .expect("the mesh has no non-Gamma k-point");
        let v1 = kpoint_shell_potential_first_directional(
            &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.v,
        )
        .expect("V1");
        let v2 =
            shell_potential_second_directional(&f.system, &f.lattice(), &f.scf, &f.pbc, &f.v);
        for (label, kfrac) in [("k=0", [0.0, 0.0, 0.0]), ("k=live", live)] {
            println!("kpoint_second/fd {label}: k = {kfrac:?}");
            let (fock2, overlap2) = kpoint_directional_second_matrices(
                &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &v1, &v2, &f.v, kfrac,
            )
            .expect("k-point second matrices");
            // `h = 5e-4`, not the `2e-3` the Gamma twin uses. The **reference** side of this gate
            // — not the analytic one — has a hard pair-inclusion boundary: both the Bloch builder
            // and the skeleton sweep drop an AO pair the moment `r` crosses `ao_cutoff`. For this
            // direction a pair sits within `2e-3` Bohr of that boundary, so the wide step
            // straddles it and the difference quotient picks up `jump / 2h`. Measured at
            // `h = 2e-3`: `err(h) = 3.2e-7` against `err(h/2) = 8.9e-9` (ratio 36) — the tell-tale
            // shape of a discontinuity inside the wide window only, which then poisons the
            // Richardson extrapolant. The analytic side has no step and cannot be the culprit:
            // at `k = 0` it is bit-identical to `gamma_directional_second_matrices`
            // (`kpoint_second_matrices_match_gamma_builder`, `0.0e0` on this very fixture), which
            // closes its own FD ladder at `1e-13`.
            let h = 5.0e-4;
            assert_cmatrix_ladder(
                &format!("S^vv({label})/diamond-skew"),
                &|lam| f.skeleton_directional_at(lam, kfrac).1,
                &overlap2,
                h,
                1.0e-6,
            );
            assert_cmatrix_ladder(
                &format!("F^vv({label})/diamond-skew"),
                &|lam| f.skeleton_directional_at(lam, kfrac).0,
                &fock2,
                h,
                1.0e-6,
            );
        }
    }

    /// Both second-order skeleton matrices must be Hermitian at a general `k`, as `S(k)` and
    /// `F(k)` themselves are — a structural consequence of the conjugate-pair scatter that a sign
    /// slip in the phase would destroy without necessarily breaking the FD ladder.
    #[test]
    #[ignore = "periodic k-point gate: one k-mesh SCC plus one second-order skeleton sweep"]
    fn kpoint_second_matrices_are_hermitian() {
        let f = KFrozen::build(DIAMOND_SKEW, kpbc_opts(), 47);
        let kfrac = f
            .scf
            .kpoints
            .iter()
            .map(|kp| kp.fractional)
            .find(|k| k.iter().any(|x| x.abs() > 1.0e-8))
            .expect("the mesh has no non-Gamma k-point");
        let v1 = kpoint_shell_potential_first_directional(
            &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.v,
        )
        .expect("V1");
        let v2 =
            shell_potential_second_directional(&f.system, &f.lattice(), &f.scf, &f.pbc, &f.v);
        let (fock2, overlap2) = kpoint_directional_second_matrices(
            &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &v1, &v2, &f.v, kfrac,
        )
        .expect("k-point second matrices");
        for (name, m) in [("F^vv", &fock2), ("S^vv", &overlap2)] {
            let n = m.n;
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            let mut im_scale = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    worst = worst
                        .max((m.re[(i, j)] - m.re[(j, i)]).abs())
                        .max((m.im[(i, j)] + m.im[(j, i)]).abs());
                    scale = scale.max(m.re[(i, j)].abs());
                    im_scale = im_scale.max(m.im[(i, j)].abs());
                }
            }
            println!(
                "kpoint_second_hermitian/{name}: max|re|={scale:.4e} max|im|={im_scale:.4e} \
                 worst asymmetry {worst:.3e}"
            );
            assert!(
                im_scale > 1.0e-8,
                "{name}: the imaginary part is zero, so the phase bookkeeping is untested"
            );
            assert!(
                worst <= 1.0e-12 * scale.max(1.0),
                "{name} is not Hermitian at k = {kfrac:?}: worst {worst:.3e}"
            );
        }
    }

    // -- Gate 5: the complex second-order response -----------------------------------------------

    /// The divided-difference table is pure arithmetic, so its confluent branches can be pinned
    /// without an SCC — and they are the single most error-prone part of the transcription
    /// (`f^{[2]}` has to switch to a derivative limit twice, once for `p ≈ q` and once for the
    /// fully confluent `p ≈ r ≈ q`). Checked against a smooth reference: for a spectrum built
    /// from an analytic `f`, every divided difference must agree with the exact Taylor limit as
    /// the nodes collapse.
    #[test]
    fn dk_tables_confluent_branches_match_their_limits() {
        // A three-level spectrum with one exactly degenerate pair and one well-separated level,
        // plus a finite temperature so f', f'' are genuinely nonzero.
        let kt = 0.02;
        let mu = 0.0;
        let occ = |e: f64| 2.0 / (1.0 + ((e - mu) / kt).exp());
        let eps = vec![-0.05, -0.05, 0.30];
        let f: Vec<f64> = eps.iter().map(|&e| occ(e)).collect();
        let t = DkTablesK::build(&eps, &f, kt);

        // Analytic derivatives of the Fermi function.
        let fp = |e: f64| {
            let fe = occ(e);
            -(fe * (1.0 - 0.5 * fe)) / kt
        };
        let fpp = |e: f64| -fp(e) * (1.0 - occ(e)) / kt;

        // Confluent first divided difference on the degenerate pair is f'.
        let e1 = (t.f1[(0, 1)] - fp(eps[0])).abs();
        // Fully confluent second divided difference is f''/2.
        let e2 = (t.f2(0, 1, 0) - 0.5 * fpp(eps[0])).abs();
        // Pinched branch (p = q degenerate, r apart) against its own numerical limit:
        // f^{[2]}(a, r, a) = (f^{[1]}(a, r) - f'(a)) / (r - a).
        let pinched = (t.f1[(0, 2)] - fp(eps[0])) / (eps[2] - eps[0]);
        let e3 = (t.f2(0, 2, 1) - pinched).abs();
        // Non-confluent branch, on a spectrum with three DISTINCT levels, against the classical
        // symmetric form of the second divided difference
        //   f[a,b,c] = f(a)/((a−b)(a−c)) + f(b)/((b−a)(b−c)) + f(c)/((c−a)(c−b))
        // which shares no code path with the implementation's recursion. (This check previously
        // compared `t.f2(2,0,2)` with itself and `plain` with itself, i.e. it was the tautology
        // `|x − x| = 0` and asserted nothing.)
        let eps_d = vec![-0.20, 0.05, 0.30];
        let f_d: Vec<f64> = eps_d.iter().map(|&e| occ(e)).collect();
        let t_d = DkTablesK::build(&eps_d, &f_d, kt);
        let (a, b, c) = (eps_d[0], eps_d[1], eps_d[2]);
        let classical = f_d[0] / ((a - b) * (a - c))
            + f_d[1] / ((b - a) * (b - c))
            + f_d[2] / ((c - a) * (c - b));
        let e4 = (t_d.f2(0, 1, 2) - classical).abs();
        println!(
            "dk_tables_confluent: |f1-f'|={e1:.3e} |f2-f''/2|={e2:.3e} |pinched|={e3:.3e} \
             |plain-classical|={e4:.3e} (classical {classical:+.6e})"
        );
        assert!(e1 < 1.0e-14, "confluent f^[1] branch: {e1:.3e}");
        assert!(e2 < 1.0e-14, "fully confluent f^[2] branch: {e2:.3e}");
        assert!(e3 < 1.0e-14, "pinched f^[2] branch: {e3:.3e}");
        assert!(classical.abs() > 1.0, "the non-confluent reference is ~zero, the check is vacuous");
        assert!(
            e4 < 1.0e-12 * classical.abs(),
            "non-confluent f^[2] branch vs the classical symmetric form: {e4:.3e}"
        );
        // The symmetry of a divided difference in its nodes is what makes the response
        // Hermitian, so it is asserted rather than assumed.
        for (p, r, q) in [(0, 1, 2), (0, 2, 1), (2, 0, 1), (1, 2, 0)] {
            let a = t.f2(p, r, q);
            let b = t.f2(q, r, p);
            assert!(
                (a - b).abs() < 1.0e-13 * a.abs().max(1.0),
                "f^[2] is not symmetric under p <-> q at ({p},{r},{q}): {a:.6e} vs {b:.6e}"
            );
        }
        // The T = 0 limit must be finite (the molecular table divides by kt unguarded because it
        // is only ever built on the finite-temperature branch; this one is not).
        let t0 = DkTablesK::build(&eps, &[2.0, 2.0, 0.0], 0.0);
        assert!(t0.f2(0, 1, 2).is_finite() && t0.f2(0, 1, 0) == 0.0);
    }

    /// **The equivalence gate: a Gamma-only mesh must reproduce the production Gamma path.**
    ///
    /// Driven on `KMesh::gamma()`, [`kpoint_second_order_directional`] and
    /// `ChargeSpaceContext::second_order_field` are two *independent* algebras applied to the
    /// same physics: the complex resolvent (Daleckii-Krein) form over one k-point versus the real
    /// coefficient/frame-rotation form that is in production for the analytic Gamma FC3. They
    /// share no code — different divided differences, different degeneracy handling, different
    /// intermediate objects — so element-wise agreement on `P^{vv}`, `W^{vv}` and `q^{vv}` is a
    /// statement about the derivation, not about a shared implementation.
    ///
    /// The first-order legs are compared first and separately. That is deliberate: if the
    /// second-order comparison ever fails, this report says immediately whether the discrepancy
    /// entered through the inputs (the CPXTB `X¹` versus the charge-space `X¹`) or through the
    /// second-order algebra itself.
    #[test]
    #[ignore = "periodic k-point gate: 2 Gamma SCCs plus both second-order response paths"]
    fn kpoint_second_order_matches_gamma_charge_space() {
        // Diamond gates the density/energy-weighted channels; BN is what makes the charge
        // channel non-vacuous (see [`BN_SKEW`]).
        let q_scales = [
            gamma_equivalence_on("diamond-skew", DIAMOND_SKEW, 29),
            gamma_equivalence_on("BN-skew", BN_SKEW, 5),
        ];
        assert!(
            q_scales.iter().any(|&s| s > 1.0e-6),
            "no fixture produced a non-zero q^vv, so the charge channel and the dielectric solve \
             went untested: {q_scales:?}"
        );
    }

    /// One fixture's worth of [`kpoint_second_order_matches_gamma_charge_space`]. Returns the
    /// magnitude of the Gamma reference `q^vv` so the caller can insist that at least one fixture
    /// exercised the charge channel.
    fn gamma_equivalence_on(label: &str, xyz: &str, seed: u64) -> f64 {
        use crate::pbc::gamma_response::gamma_charge_space_context;
        use crate::pbc::gamma_third::{
            gamma_directional_second_matrices, shell_potential_first_directional,
        };
        use crate::pbc::hessian::{gamma_mos, gamma_skeleton_derivatives};

        let system = system_of(xyz);
        let params = params();
        let pbc = gamma_pbc_opts();
        let mut opts = electronic();
        opts.energy_tolerance = 1.0e-13;
        opts.charge_tolerance = 1.0e-12;
        let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("Gamma-mesh SCC");
        assert!(scf.converged && scf.kpoints.len() == 1);
        let lattice = system.lattice.as_ref().copied().unwrap();
        let n = scf.basis.len();
        let v = direction(3 * system.atoms.len(), seed);

        // ---- the k-point path (complex, resolvent form) ----
        let x1 = kpoint_first_order_directional(&system, &params, &scf, &opts, &pbc, &v)
            .expect("k first order");
        let x2 = kpoint_second_order_directional(&system, &params, &scf, &opts, &pbc, &x1, &v)
            .expect("k second order");

        // ---- the Gamma path (real, coefficient/frame form) — the production assembly's own
        // sequence, reproduced here because `GammaThirdReference`'s fields are private ----
        let mos = gamma_mos(&scf, scf.nelec).expect("Gamma MOs");
        let ctx = gamma_charge_space_context(&system, &params, &scf, &mos, opts.charge_order)
            .expect("Gamma charge-space context");
        let sk = gamma_skeleton_derivatives(&system, &params, &scf, &opts, &pbc).expect("skeleton");
        let mut f1 = Matrix::zeros(n, n);
        let mut s1 = Matrix::zeros(n, n);
        for (y, &vy) in v.iter().enumerate() {
            if vy == 0.0 {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    f1[(i, j)] += vy * sk.fock[y][(i, j)];
                    s1[(i, j)] += vy * sk.overlap[y][(i, j)];
                }
            }
        }
        let field = ctx.first_order_field(f1, s1).expect("Gamma first order");
        let q1 = field.bundle.shell_charges.clone();
        let dgamma_v_q1 = {
            let mut doctored = scf.clone();
            doctored.shell_charges = q1.clone();
            let mut atom_q = vec![0.0_f64; system.atoms.len()];
            for (ish, shell) in scf.basis.shells.iter().enumerate() {
                atom_q[shell.atom_index] += q1[ish];
            }
            doctored.atomic_charges = atom_q;
            let sk1 = gamma_skeleton_derivatives(&system, &params, &doctored, &opts, &pbc)
                .expect("doctored skeleton");
            shell_potential_first_directional(&sk1, &v)
        };
        let v1_pot = shell_potential_first_directional(&sk, &v);
        let v2_pot = shell_potential_second_directional(&system, &lattice, &scf, &pbc, &v);
        let (f_vv, s_vv) = gamma_directional_second_matrices(
            &system, &params, &scf, &opts, &pbc, &v1_pot, &v2_pot, &v,
        )
        .expect("Gamma second matrices");
        let second = ctx
            .second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_q1, &dgamma_v_q1)
            .expect("Gamma second order");

        // ---- first-order legs, reported separately (the fault-isolation half) ----
        let mut w_q1 = 0.0_f64;
        for s in 0..q1.len() {
            w_q1 = w_q1.max((x1.q[s] - q1[s]).abs());
        }
        let mut w_p1 = 0.0_f64;
        let mut w_p1_im = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                w_p1 = w_p1.max((x1.p_k[0].re[(i, j)] - field.bundle.density[(i, j)]).abs());
                w_p1_im = w_p1_im.max(x1.p_k[0].im[(i, j)].abs());
            }
        }
        let mut q1_scale = 0.0_f64;
        for &q in &q1 {
            q1_scale = q1_scale.max(q.abs());
        }
        println!(
            "gamma-equiv[{label}]/first-order: max|q1| {q1_scale:.4e}  |dq1| {w_q1:.3e}  \
             |dP1| {w_p1:.3e}  max|Im P1| {w_p1_im:.3e}"
        );

        // ---- second-order comparison, element by element ----
        let cmp = |name: &str, mine: &CMatrix, theirs: &Matrix| -> f64 {
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            let mut im = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    worst = worst.max((mine.re[(i, j)] - theirs[(i, j)]).abs());
                    scale = scale.max(theirs[(i, j)].abs());
                    im = im.max(mine.im[(i, j)].abs());
                }
            }
            println!(
                "gamma-equiv[{label}]/{name:<4}: max|ref| {scale:.4e}  worst |k - gamma| \
                 {worst:.3e}  max|Im| {im:.3e}"
            );
            assert!(scale > 1.0e-8, "{name}: the reference is zero, the gate is vacuous");
            assert!(im < 1.0e-12, "{name}: a Gamma-only mesh left a nonzero imaginary part: {im:.3e}");
            worst
        };
        let w_p = cmp("P^vv", &x2.p_k[0], &second.bundle.density);
        let w_w = cmp("W^vv", &x2.w_k[0], &second.bundle.energy_weighted);
        let mut w_q = 0.0_f64;
        let mut q_scale = 0.0_f64;
        for s in 0..x2.q.len() {
            w_q = w_q.max((x2.q[s] - second.bundle.shell_charges[s]).abs());
            q_scale = q_scale.max(second.bundle.shell_charges[s].abs());
        }
        println!("gamma-equiv[{label}]/q^vv : max|ref| {q_scale:.4e}  worst |k - gamma| {w_q:.3e}");

        // The real-space image at the origin must be the k = 0 block itself (a Gamma-only mesh
        // carries a single unit phase), which is what makes the images a valid drop-in for the
        // frozen builders.
        let mut w_img = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                w_img = w_img.max((x2.p_images[&[0, 0, 0]][(i, j)] - x2.p_k[0].re[(i, j)]).abs());
            }
        }
        assert!(w_img == 0.0, "the origin image is not the k = 0 block: {w_img:.3e}");

        let tol = 1.0e-9;
        assert!(w_q1 < tol && w_p1 < tol, "[{label}] the FIRST-order inputs already disagree (q {w_q1:.3e}, P {w_p1:.3e}) — the second-order algebra is not implicated");
        assert!(w_p <= tol, "[{label}] P^vv: complex resolvent vs Gamma coefficient path: {w_p:.3e}");
        assert!(w_w <= tol, "[{label}] W^vv: complex resolvent vs Gamma coefficient path: {w_w:.3e}");
        assert!(w_q <= tol, "[{label}] q^vv: complex resolvent vs Gamma coefficient path: {w_q:.3e}");
        q_scale
    }

    /// **The physical gate**: over a genuine `2 x 2 x 2` mesh, `q^{vv}` must be the second
    /// central difference of the reconverged k-point SCC's shell charges.
    ///
    /// Nothing about this reference is shared with the analytic path — it is a fully independent
    /// self-consistent solve at three geometries — so it tests the dielectric coupling, the
    /// Brillouin-zone weighting and the resolvent algebra at once. The step is the usual
    /// compromise: a second difference divides by `h²`, so the reconvergence noise floor grows as
    /// `1/h²` and `h` cannot be shrunk indefinitely; `2e-3` against a `1e-12` charge tolerance
    /// sits above that floor while keeping the `h²` truncation error small.
    #[test]
    #[ignore = "periodic k-point gate: 3 reconverged k-mesh SCCs plus the full second-order path"]
    fn kpoint_second_order_charges_match_reconverged_fd() {
        // Heteronuclear by necessity: `DIAMOND_SKEW`'s inversion symmetry sends every shell-charge
        // response to zero, which would make this gate vacuous (see [`BN_SKEW`]).
        let system = system_of(BN_SKEW);
        let params = params();
        let pbc = kpbc_opts();
        let mut opts = electronic();
        opts.energy_tolerance = 1.0e-13;
        opts.charge_tolerance = 1.0e-12;
        let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("tight k-mesh SCC");
        assert!(scf.converged && scf.kpoints.len() > 1, "the mesh must be a real one");
        let v = direction(3 * system.atoms.len(), 31);

        let x1 = kpoint_first_order_directional(&system, &params, &scf, &opts, &pbc, &v)
            .expect("k first order");
        let x2 = kpoint_second_order_directional(&system, &params, &scf, &opts, &pbc, &x1, &v)
            .expect("k second order");

        let charges_at = |lam: f64| -> Vec<f64> {
            run_pbc_scc(&displaced(&system, &v, lam), &params, &opts, &pbc)
                .expect("displaced k-mesh SCC")
                .shell_charges
        };
        let h = 2.0e-3;
        let (qp, qm, q0) = (charges_at(h), charges_at(-h), scf.shell_charges.clone());
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for s in 0..x2.q.len() {
            let fd = (qp[s] - 2.0 * q0[s] + qm[s]) / (h * h);
            worst = worst.max((fd - x2.q[s]).abs());
            scale = scale.max(x2.q[s].abs());
        }
        println!("kpoint_q2/BN-skew: max|q^vv| {scale:.4e}  worst |analytic - FD| {worst:.3e}");
        assert!(scale > 1.0e-4, "q^vv is ~zero, the gate is vacuous");
        assert!(
            worst < 1.0e-7,
            "directional k-point second charge response vs reconverged second difference: \
             {worst:.3e}"
        );
    }

    /// `P^{vv}(k)` and `W^{vv}(k)` must be Hermitian at every k-point of a real mesh.
    ///
    /// This is not a symmetrisation check — nothing in [`kpoint_dk_second_order_mo`] symmetrises.
    /// Hermiticity follows from the node symmetry of the divided differences combined with
    /// writing every bilinear term in both orderings, so a broken transcription of either shows
    /// up here immediately. The imaginary part is asserted to be genuinely nonzero first, so the
    /// gate cannot pass by the matrices being accidentally real.
    #[test]
    #[ignore = "periodic k-point gate: one k-mesh SCC plus the full second-order path"]
    fn kpoint_second_order_response_is_hermitian() {
        let system = system_of(DIAMOND_SKEW);
        let params = params();
        let pbc = kpbc_opts();
        let opts = electronic();
        let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("k-mesh SCC");
        assert!(scf.converged && scf.kpoints.len() > 1);
        let v = direction(3 * system.atoms.len(), 37);
        let x1 = kpoint_first_order_directional(&system, &params, &scf, &opts, &pbc, &v)
            .expect("k first order");
        let x2 = kpoint_second_order_directional(&system, &params, &scf, &opts, &pbc, &x1, &v)
            .expect("k second order");

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut im_scale = 0.0_f64;
        for (name, set) in [("P^vv", &x2.p_k), ("W^vv", &x2.w_k)] {
            for (ik, m) in set.iter().enumerate() {
                let n = m.n;
                let mut w = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        w = w
                            .max((m.re[(i, j)] - m.re[(j, i)]).abs())
                            .max((m.im[(i, j)] + m.im[(j, i)]).abs());
                        scale = scale.max(m.re[(i, j)].abs());
                        im_scale = im_scale.max(m.im[(i, j)].abs());
                    }
                }
                println!("kpoint_second_hermitian/{name} k={ik}: worst asymmetry {w:.3e}");
                worst = worst.max(w);
            }
        }
        println!(
            "kpoint_second_hermitian: max|Re| {scale:.4e} max|Im| {im_scale:.4e} worst {worst:.3e}"
        );
        assert!(
            im_scale > 1.0e-8,
            "the responses are real, so the complex path is untested by this gate"
        );
        assert!(
            worst <= 1.0e-12 * scale.max(1.0),
            "the second-order response is not Hermitian: {worst:.3e}"
        );
    }
}
