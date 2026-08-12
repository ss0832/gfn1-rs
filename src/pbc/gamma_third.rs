// SPDX-License-Identifier: GPL-3.0-or-later
// Staged builders: every item here is `pub(crate)` and is consumed by the Gamma-point third
// derivative assembly (frozen half + response half), which lands separately. Until that entry
// point exists nothing in the library calls them, so the whole module would otherwise be one
// long dead-code warning. Same treatment, and for the same reason, as
// `integrals_third_derivatives`. The `#[cfg(test)]` gates below exercise every builder.
#![allow(dead_code)]
//! **Frozen (response-free) component builders for the analytic Gamma-point periodic third
//! derivative** (Phase 8.3).
//!
//! Everything here is `pub(crate)`: this module deliberately exposes **no public API**. It
//! supplies the frozen half of the periodic cubic force constants; the response half and the
//! public entry point are assembled elsewhere.
//!
//! # Contract: directional, not dense
//!
//! Every builder returns a [`DirectionalDerivs`] for one direction `v` in R^{3N} (lattice held
//! fixed, Gamma point, real matrices):
//!
//! ```text
//!   second = vᵀ H_block v
//!   third  = d/dλ [ vᵀ H_block(R + λv) v ] |_{λ=0}   ==   Σ_abc T_abc v_a v_b v_c
//! ```
//!
//! `second` is emitted alongside `third` by the *same* loop, at negligible extra cost, for two
//! reasons: it is what the finite-difference gate needs (a central difference of `second` over
//! `λ` must reproduce `third`), and it lets a caller cross-check the block against the analytic
//! Hessian without a second traversal.
//!
//! ## Why a directional scalar rather than a `T_abc` slab
//!
//! Several periodic frozen blocks are **deliberately asymmetric**. `band_pulay_fixed_hessian`
//! in [`crate::pbc::hessian`] is `∂(gradient_row)/∂R_col`, not `∂²Φ` of any scalar: the
//! three-centre Pulay piece `−P ∂S_row ∂V_col` is accumulated only in the (pair-row, any-col)
//! orientation, and `pbc_gamma_hessian` symmetrises the *total* (frozen + response) at the end.
//! A per-block `T_abc` would therefore have to carry a convention for that asymmetry, whereas
//! `vᵀ H v` is invariant under it (`vᵀ A v = vᵀ Aᵀ v`) and is exactly the quantity a central
//! finite difference of the analytic Hessian measures. Polarisation identities recover the dense
//! tensor from the directional scalar when the caller wants one.
//!
//! # Component inventory
//!
//! | builder | mirrors (in `pbc::hessian`) |
//! |---|---|
//! | [`pbc_repulsion_third_directional`]  | `repulsion_energy_gradient_hessian` |
//! | [`pbc_halogen_third_directional`]    | `halogen_energy_gradient_hessian` |
//! | [`pbc_dispersion_third_directional`] | `dispersion_energy_gradient_hessian` |
//! | [`pbc_cn_third_directional`]         | `cn_fixed_hessian` |
//! | [`pbc_band_pulay_third_directional`] | `band_pulay_fixed_hessian` |
//! | [`pbc_scc2_realspace_third_directional`] | `electrostatic_fixed_hessian` (real-space QCore block only) |
//! | [`shell_potential_second_directional`] | `shell_potential_derivatives`, one order up |
//!
//! ## Response-support components
//!
//! The blocks above are response-free. The remaining builders here are the **inputs** the
//! caller's response half needs, and they are gated the same way (every one against a central
//! difference of the production object it extends):
//!
//! | builder | supplies |
//! |---|---|
//! | [`gamma_first_order_directional`] | `X¹ = (P¹, W¹, q¹)` from one directional CPXTB solve |
//! | [`gamma_directional_second_matrices`] | `(F^vv, S^vv)`, the second-order response field's skeleton |
//! | [`pbc_scc2_bilinear_second_directional`] | the two-charge-vector SCC2 charge path |
//! | [`gamma_response_path_directional`] | `B6` (density path) and `bg4` (background motion) |
//!
//! ## The classical blocks are already periodic
//!
//! `repulsion_third_derivative`, `halogen_third_derivative` and `dispersion_third_derivative`
//! take a [`PeriodicSystem`] and enumerate lattice images internally — via
//! `for_each_unique_short_range_pair` / `unique_short_range_pairs` / `halogen_triples`, the same
//! image-aware helpers their own Hessian twins use. They therefore need **no image-sum
//! extension**: the three builders above are thin directional contractions, and the gates in
//! this module prove the periodicity by finite-differencing the *production* periodic Hessian
//! that `pbc_gamma_hessian` itself adds.
//!
//! ## Ewald-entangled SCC2: deliberately not implemented
//!
//! [`pbc_scc2_realspace_third_directional`] covers **only** the real-space QCore block of
//! `electrostatic_fixed_hessian` (the generalised `R^-3` Ewald *real* remainder plus the
//! short-range Klopman-Ohno residual) — the molecular mirror plus an image sum. The `1/R` Ewald
//! `erfc` sum, the reciprocal structure-factor sum and the QCore `R^-3` reciprocal sum all need
//! the **third** derivative of the Ewald-split gamma kernel, which is owned elsewhere; see
//! [`pbc_scc2_ewald_third_directional_todo`] for the exact list.
//!
//! Note that the *second* derivative of the periodic gamma kernel **is** implemented here, in
//! [`shell_potential_second_directional`] — it is required by the band/Pulay third (the
//! `−P S₁ V₂` term) and involves no new kernel mathematics beyond what
//! `electrostatic_fixed_hessian` already contains.

use std::collections::HashMap;

use crate::basis::{BasisSet, BasisShell};
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::harmonic_average;
use crate::data_tables::{atomic_radius_bohr, covalent_radius_d3_bohr};
use crate::electronic::ElectronicOptions;
use crate::error::Result;
use crate::hamiltonian::hscale;
use crate::integrals::{contracted_pair_with_second_derivatives, contracted_pair_with_third_derivatives};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::{erfc, Vec3};
use crate::params::Gfn1Parameters;
use crate::pbc::ewald::{exp1, resolve_alpha, QCORE_R3_COEFF};
use crate::pbc::hessian::GammaSkeletonDerivatives;
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::PbcSccResult;
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use std::f64::consts::PI;

/// Mirrors `pbc::hessian`'s private constants so the image-sum cutoff conventions match the
/// blocks being differentiated exactly.
const SQRT_PI: f64 = 1.772_453_850_905_516;
const TAU: f64 = 5.5;
const DIST_EPS: f64 = 1.0e-12;

/// One direction's second and third derivative of a single frozen block.
///
/// `second = vᵀ H v` and `third = d/dλ (vᵀ H(R+λv) v)`. Both are produced by the same traversal;
/// `second` is the finite-difference gate's observable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DirectionalDerivs {
    pub second: f64,
    pub third: f64,
}

impl DirectionalDerivs {
    #[inline]
    fn accumulate(&mut self, second: f64, third: f64) {
        self.second += second;
        self.third += third;
    }
}

impl std::ops::Add for DirectionalDerivs {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            second: self.second + rhs.second,
            third: self.third + rhs.third,
        }
    }
}

impl std::iter::Sum for DirectionalDerivs {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

/// Real-space frozen densities `P(T)` / `W(T)` keyed by integer image offset, the input the
/// band/Pulay and CN blocks read. Built by [`gamma_realspace_densities`].
#[derive(Clone, Debug)]
pub(crate) struct FrozenDensityImages {
    pub p: HashMap<[i32; 3], Matrix>,
    pub w: HashMap<[i32; 3], Matrix>,
}

// ---------------------------------------------------------------------------------------------
// Shared directional machinery
// ---------------------------------------------------------------------------------------------

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

/// Directional derivatives of a scalar radial function `f(r)` along the pair displacement.
///
/// With `r(λ) = |rvec + λ d|`, `s = (rvec/r)·d` and `d2 = |d|²`, the chain rule through
/// `ṙ = s` and `ṡ = (d2 − s²)/r` gives
///
/// ```text
///   h1 = f' s
///   h2 = f'' s² + f' (d2 − s²)/r
///   h3 = f''' s³ + 3 f'' s (d2 − s²)/r − 3 f' s (d2 − s²)/r²
/// ```
#[inline]
fn radial_chain(f1: f64, f2: f64, f3: f64, r: f64, s: f64, d2: f64) -> (f64, f64, f64) {
    let t = (d2 - s * s) / r;
    (
        f1 * s,
        f2 * s * s + f1 * t,
        f3 * s * s * s + 3.0 * f2 * s * t - 3.0 * f1 * s * t / r,
    )
}

/// `vᵀ H v` for one radial pair block, the object `add_radial_hessian` accumulates.
///
/// `pref = f'/r` and `dpref = d/dr(f'/r) = f''/r − f'/r²`, exactly the two scalars the periodic
/// Hessian passes. The pattern `H = [[A,−A],[−A,A]]` contracts to `d`-only dependence.
#[inline]
fn radial_pair_second_vv(pref: f64, dpref: f64, r: f64, s: f64, d2: f64) -> f64 {
    pref * d2 + dpref * r * s * s
}

/// `d/dλ` of [`radial_pair_second_vv`] with `f` itself held fixed.
///
/// `pref2 = d²/dr²(f'/r) = f'''/r − 2f''/r² + 2f'/r³`. Derivation: differentiating
/// `pref·d2 + dpref·r·s²` through `ṙ = s`, `ṡ = (d2 − s²)/r` collapses to
/// `3 dpref s d2 + (pref2 r − dpref) s³`.
#[inline]
fn radial_pair_third_vv(dpref: f64, pref2: f64, r: f64, s: f64, d2: f64) -> f64 {
    3.0 * dpref * s * d2 + (pref2 * r - dpref) * s * s * s
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

#[inline]
fn ten3_contract(t: &[[[f64; 3]; 3]; 3], a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (aa, bb, cc) = (a.to_array(), b.to_array(), c.to_array());
    let mut acc = 0.0;
    for (i, &ai) in aa.iter().enumerate() {
        for (j, &bj) in bb.iter().enumerate() {
            for (k, &ck) in cc.iter().enumerate() {
                acc += t[i][j][k] * ai * bj * ck;
            }
        }
    }
    acc
}

/// Contract `ndof` dense slabs `slab[c][(a,b)]` with `v` three times.
fn contract_slabs_vvv(slabs: &[Matrix], v: &[f64]) -> f64 {
    let n = v.len();
    let mut total = 0.0;
    for (c, slab) in slabs.iter().enumerate().take(n) {
        if v[c] == 0.0 {
            continue;
        }
        let mut inner = 0.0;
        for (a, &va) in v.iter().enumerate() {
            if va == 0.0 {
                continue;
            }
            for (b, &vb) in v.iter().enumerate() {
                inner += slab[(a, b)] * va * vb;
            }
        }
        total += v[c] * inner;
    }
    total
}

/// Contract a row-major dense `ndof³` tensor `t[(a·n+b)·n+c]` with `v` three times.
fn contract_dense_vvv(t: &[f64], n: usize, v: &[f64]) -> f64 {
    let mut total = 0.0;
    for (a, &va) in v.iter().enumerate().take(n) {
        if va == 0.0 {
            continue;
        }
        for (b, &vb) in v.iter().enumerate().take(n) {
            if vb == 0.0 {
                continue;
            }
            let base = (a * n + b) * n;
            let mut inner = 0.0;
            for (c, &vc) in v.iter().enumerate().take(n) {
                inner += t[base + c] * vc;
            }
            total += va * vb * inner;
        }
    }
    total
}

/// `vᵀ M v` for a dense `ndof × ndof` matrix.
fn contract_matrix_vv(m: &Matrix, v: &[f64]) -> f64 {
    let mut total = 0.0;
    for (a, &va) in v.iter().enumerate() {
        if va == 0.0 {
            continue;
        }
        let mut inner = 0.0;
        for (b, &vb) in v.iter().enumerate() {
            inner += m[(a, b)] * vb;
        }
        total += va * inner;
    }
    total
}

// ---------------------------------------------------------------------------------------------
// Radial ladders: one order beyond what `pbc::hessian` / `pbc::ewald` expose
// ---------------------------------------------------------------------------------------------

/// Klopman-Ohno kernel `γ(r) = (r² + η⁻²)^{-1/2}` and its first three radial derivatives.
///
/// The value/first/second agree with `pbc::ewald::ko_value_derivatives`; the third,
/// `γ''' = 9r D^{-5/2} − 15 r³ D^{-7/2}` with `D = r² + η⁻²`, is new here.
#[inline]
fn ko_value_derivatives3(r: f64, eta: f64) -> (f64, f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let d = r * r + inv_eta2;
    let d15 = d.powf(1.5);
    let d25 = d.powf(2.5);
    let d35 = d.powf(3.5);
    (
        1.0 / d.sqrt(),
        -r / d15,
        3.0 * r * r / d25 - 1.0 / d15,
        9.0 * r / d25 - 15.0 * r * r * r / d35,
    )
}

/// Short-range QCore Klopman-Ohno residual `γ_KO − 1/r + ½η⁻²/r³`, to third radial order.
/// Value/first/second reproduce `pbc::ewald::qcore_short_value_derivatives`.
#[inline]
fn qcore_short_value_derivatives3(r: f64, eta: f64) -> (f64, f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let (ko, dko, d2ko, d3ko) = ko_value_derivatives3(r, eta);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r3 * r3;
    (
        ko - 1.0 / r + 0.5 * inv_eta2 / r3,
        dko + 1.0 / r2 - 1.5 * inv_eta2 / r4,
        d2ko - 2.0 / r3 + 6.0 * inv_eta2 / r5,
        d3ko + 6.0 / r4 - 30.0 * inv_eta2 / r6,
    )
}

/// Real-space remainder of the generalised `R^-3` Ewald, `η⁻² q(r)/r³` with
/// `q = erfc(αr) + (2αr/√π)e^{-α²r²}`, to third radial order.
///
/// Value/first/second reproduce `pbc::ewald::qcore_r3_real_value_derivatives`. The new pieces
/// are `q''' = (8α³/√π) e^{-α²r²}(−2α⁴r⁴ + 5α²r² − 1)` and the Leibniz expansion
/// `(q r^{-3})''' = q'''/r³ − 9q''/r⁴ + 36q'/r⁵ − 60q/r⁶`.
#[inline]
fn qcore_r3_real_value_derivatives3(r: f64, eta: f64, alpha: f64) -> (f64, f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let a2 = alpha * alpha;
    let a3 = a2 * alpha;
    let ar = alpha * r;
    let ar2 = ar * ar;
    let exp_ar2 = (-ar2).exp();
    let q = erfc(ar) + (2.0 * ar / SQRT_PI) * exp_ar2;
    let dq = -(4.0 * a3 * r * r / SQRT_PI) * exp_ar2;
    let d2q = (8.0 * a3 * r / SQRT_PI) * exp_ar2 * (ar2 - 1.0);
    let d3q = (8.0 * a3 / SQRT_PI) * exp_ar2 * (-2.0 * ar2 * ar2 + 5.0 * ar2 - 1.0);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r3 * r3;
    (
        inv_eta2 * q / r3,
        inv_eta2 * (dq / r3 - 3.0 * q / r4),
        inv_eta2 * (d2q / r3 - 6.0 * dq / r4 + 12.0 * q / r5),
        inv_eta2 * (d3q / r3 - 9.0 * d2q / r4 + 36.0 * dq / r5 - 60.0 * q / r6),
    )
}

/// The `1/R` Ewald real-space kernel `g(d) = erfc(αd)/d` and its first two radial derivatives,
/// matching `pbc::hessian`'s private `ewald_real_radial_derivs`. Only two orders are needed:
/// this feeds the *second* derivative of the shell potential, not the SCC2 third.
#[inline]
fn ewald_real_radial_derivs(d: f64, alpha: f64) -> (f64, f64) {
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;
    let e = (-alpha * alpha * d * d).exp();
    let erfc_ad = erfc(alpha * d);
    let gp = -erfc_ad / (d * d) - two_alpha_sqrtpi * e / d;
    let gpp = (two_alpha_sqrtpi * e) / (d * d)
        + 2.0 * erfc_ad / (d * d * d)
        + two_alpha_sqrtpi * (2.0 * alpha * alpha * e + e / (d * d));
    (gp, gpp)
}

/// The same `g(d) = erfc(αd)/d` carried to the **third** radial order, for the SCC2 Ewald
/// third derivative. With `c = 2α/√π`, `F = e^{−α²d²}`, `E = erfc(αd)` (so `E' = −cF`,
/// `F' = −2α²dF`):
///
/// ```text
/// g⁰ =  E/d
/// g¹ = −E/d² − cF/d
/// g² =  2E/d³ + 2cF/d² + 2cα²F
/// g³ = −6E/d⁴ − 6cF/d³ − 4cα²F/d − 4cα⁴dF
/// ```
///
/// The first two orders reduce exactly to [`ewald_real_radial_derivs`].
#[inline]
fn ewald_real_value_derivatives3(d: f64, alpha: f64) -> (f64, f64, f64, f64) {
    let c = 2.0 * alpha / SQRT_PI;
    let a2 = alpha * alpha;
    let f = (-a2 * d * d).exp();
    let e = erfc(alpha * d);
    let d2 = d * d;
    let d3 = d2 * d;
    let d4 = d3 * d;
    let g0 = e / d;
    let g1 = -e / d2 - c * f / d;
    let g2 = 2.0 * e / d3 + 2.0 * c * f / d2 + 2.0 * c * a2 * f;
    let g3 = -6.0 * e / d4 - 6.0 * c * f / d3 - 4.0 * c * a2 * f / d - 4.0 * c * a2 * a2 * d * f;
    (g0, g1, g2, g3)
}

/// Smooth coordination-number counting function `f(r) = σ(−k(r_c/r − 1))` to third radial order.
///
/// Value/first/second agree with `pbc::hessian`'s `cn_count_value_derivatives`. The third uses
/// the sigmoid ladder written in the exponential `e` rather than in `σ`
/// (`σ₁ = −e/D²`, `σ₂ = e(e−1)/D³`, `σ₃ = −e(e²−4e+1)/D⁴`, `D = 1+e`), which avoids the
/// catastrophic `1 − σ` cancellation at CN saturation — the same form, and for the same reason,
/// as the non-PBC `coordination_value_derivatives`.
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

/// The `H0` geometric prefactor `f(r) = coeff · (1 + p_i·rr)(1 + p_j·rr)`, `rr = √(r/rad)`, and
/// its first three radial derivatives. `coeff` is treated as a constant, exactly as
/// `pbc::hessian::prefactor_radial` does (the CN dependence is a separate block).
///
/// Written directly in `r` — with `a₁ = p_i + p_j`, `a₂ = p_i p_j` and `rr = (r/rad)^{1/2}`:
///
/// ```text
///   f   = coeff (1 + a₁ rr + a₂ rr²)
///   f'  = coeff ( a₁/(2 rad rr) + a₂/rad )
///   f'' = coeff ( −a₁ / (4 rad² rr³) )
///   f'''= coeff ( +3 a₁ / (8 rad³ rr⁵) )
/// ```
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

// ---------------------------------------------------------------------------------------------
// 1-3. Classical periodic blocks (repulsion / halogen / D3)
// ---------------------------------------------------------------------------------------------

/// **Periodic repulsion third derivative**, directional.
///
/// `repulsion_third_derivative` already runs the lattice-image sum
/// (`for_each_unique_short_range_pair` streams atom/image pairs, skipping `i == j` self-images
/// exactly as `repulsion_energy_gradient_hessian` does), so this is a contraction, not an
/// extension. Gated against a central difference of the periodic repulsion Hessian that
/// `pbc_gamma_hessian` adds verbatim.
pub(crate) fn pbc_repulsion_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    v: &[f64],
) -> Result<DirectionalDerivs> {
    let hess = crate::repulsion::repulsion_energy_gradient_hessian(system, params)?;
    let third = crate::repulsion::repulsion_third_derivative(system, params)?;
    Ok(DirectionalDerivs {
        second: contract_matrix_vv(&hess.hessian, v),
        third: contract_slabs_vvv(&third, v),
    })
}

/// **Periodic halogen-bond third derivative**, directional.
///
/// `halogen_third_derivative` builds its `Jet3` over image-resolved triples
/// (`halogen_triples` carries `acceptor_translation` / `neighbor_translation`, fed through
/// `jet_image_sub`), matching `halogen_energy_gradient_hessian`'s image convention. Only active
/// for systems containing Br/I with an N/O/P/S acceptor in range.
pub(crate) fn pbc_halogen_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    v: &[f64],
) -> Result<DirectionalDerivs> {
    let hess = crate::halogen::halogen_energy_gradient_hessian(system, params)?;
    let third = crate::halogen::halogen_third_derivative(system, params)?;
    Ok(DirectionalDerivs {
        second: contract_matrix_vv(&hess.hessian, v),
        third: contract_slabs_vvv(&third, v),
    })
}

/// **Periodic D3(BJ) dispersion third derivative**, directional — two-body BJ plus the ATM
/// three-body term when `s9 != 0`, with the full `C6(CN(R))` chain carried by forward AD.
///
/// `dispersion_third_derivative` shares `unique_short_range_pairs` / `d3_coordination_jets` /
/// `d3_atm_accumulate_jet` with `dispersion_energy_gradient_hessian`, so the image pairs and
/// triples are already the periodic ones. Memory is the dense `Jet3` (`O(nat²·ndof³)`), which is
/// why this is a small-cell path.
pub(crate) fn pbc_dispersion_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    reference_path: Option<&str>,
    v: &[f64],
) -> Result<DirectionalDerivs> {
    let hess = crate::dispersion::dispersion_energy_gradient_hessian(system, params, reference_path)?;
    let third = crate::dispersion::dispersion_third_derivative(system, params, reference_path)?;
    Ok(DirectionalDerivs {
        second: contract_matrix_vv(&hess.hessian, v),
        third: contract_dense_vvv(&third.third, third.ndof, v),
    })
}

// ---------------------------------------------------------------------------------------------
// Shell scalar potential: first and second directional derivatives at frozen charges
// ---------------------------------------------------------------------------------------------

/// `V₁_s = Σ_col v_col ∂V_s/∂R_col`, read straight off the production skeleton.
pub(crate) fn shell_potential_first_directional(
    skeleton: &GammaSkeletonDerivatives,
    v: &[f64],
) -> Vec<f64> {
    let nsh = skeleton.shell_potential[0].len();
    let mut out = vec![0.0; nsh];
    for (col, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        for (s, o) in out.iter_mut().enumerate() {
            *o += vc * skeleton.shell_potential[col][s];
        }
    }
    out
}

/// `V₂_s = Σ_{col,col'} v_col v_col' ∂²V_s/∂R_col ∂R_col'` at **frozen shell charges** — the
/// one-order-up twin of `pbc::hessian::shell_potential_derivatives`.
///
/// `V_s = Σ_t γ_st q_t` with the periodic QCore gamma, so this sums the same four pieces in the
/// same order and with the same cutoffs:
///
/// 1. `1/R` Ewald real-space `erfc` (radial, per atom pair/image);
/// 2. `1/R` Ewald reciprocal (`cos` phases — `d²/dλ² cos θ = −τ² cos θ` with `τ = G·(v_A − v_B)`);
/// 3. QCore `R^-3` reciprocal (same `cos` structure, per shell pair);
/// 4. QCore `R^-3` real remainder + short-range Klopman-Ohno residual (radial, per shell pair).
///
/// This is second order only: it needs no kernel derivative beyond what
/// `electrostatic_fixed_hessian` already carries, and is independent of the reserved Ewald
/// *third*-derivative work. Gated by a central difference of the production
/// `gamma_skeleton_derivatives(...).shell_potential` evaluated at displaced geometries against
/// the *same* frozen SCC result.
pub(crate) fn shell_potential_second_directional(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    v: &[f64],
) -> Vec<f64> {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let q_atom = &scf.atomic_charges;
    let nat = system.atoms.len();
    let nsh = basis.shells.len();
    let mut out = vec![0.0; nsh];

    let alpha = resolve_alpha(system, &pbc.ewald);

    // (1) + (2): the `1/R` Ewald atomic potential, second directional derivative per atom.
    let mut phi2 = vec![0.0; nat];
    let ew_real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let r_offsets = lattice.image_offsets(ew_real_cut);
    let r_trans: Vec<Vec3> = r_offsets.iter().map(|o| lattice.translation(*o)).collect();
    for a in 0..nat {
        for b in 0..nat {
            let dv = pair_direction(v, a, b);
            let dv2 = dv.norm2();
            for t in &r_trans {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > ew_real_cut {
                    continue;
                }
                let (gp, gpp) = ewald_real_radial_derivs(d, alpha);
                let s = vec.dot(dv) / d;
                // Second directional derivative of g(d) along the pair displacement.
                phi2[a] += q_atom[b] * (gpp * s * s + gp * (dv2 - s * s) / d);
            }
        }
    }
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / lattice.volume();
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = (-g2 * inv_4a2).exp() / g2;
        for a in 0..nat {
            for b in 0..nat {
                let phase = g.dot(system.atoms[a].position - system.atoms[b].position);
                let tau = g.dot(pair_direction(v, a, b));
                phi2[a] += -four_pi_v * w_g * tau * tau * phase.cos() * q_atom[b];
            }
        }
    }
    for (i, shell) in basis.shells.iter().enumerate() {
        out[i] += phi2[shell.atom_index];
    }

    // (3) QCore `R^-3` reciprocal, per shell pair.
    let pref0 = QCORE_R3_COEFF * 2.0 * PI / lattice.volume();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff = pref0 * exp1(g.norm2() * inv_4a2);
        for i in 0..nsh {
            let ai = basis.shells[i].atom_index;
            for j in 0..nsh {
                if q[j] == 0.0 {
                    continue;
                }
                let aj = basis.shells[j].atom_index;
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let tau = g.dot(pair_direction(v, ai, aj));
                out[i] +=
                    -coeff * q[j] / (eta * eta) * tau * tau * (phases[i] - phases[j]).cos();
            }
        }
    }

    // (4) QCore `R^-3` real remainder + short-range residual, per shell pair.
    let r3_cut = TAU / alpha;
    let sr_cut = pbc.ewald.sr_cutoff;
    let real_cut = r3_cut.max(sr_cut);
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        let ra = system.atoms[ai].position;
        for j in 0..nsh {
            if q[j] == 0.0 {
                continue;
            }
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let dv = pair_direction(v, ai, aj);
            let dv2 = dv.norm2();
            for (off, t) in offsets.iter().zip(&translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let vec = ra - system.atoms[aj].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let mut d1 = 0.0;
                let mut d2 = 0.0;
                if d <= r3_cut {
                    let (_, a1, a2, _) = qcore_r3_real_value_derivatives3(d, eta, alpha);
                    d1 += QCORE_R3_COEFF * a1;
                    d2 += QCORE_R3_COEFF * a2;
                }
                if d <= sr_cut {
                    let (_, b1, b2, _) = qcore_short_value_derivatives3(d, eta);
                    d1 += b1;
                    d2 += b2;
                }
                let s = vec.dot(dv) / d;
                out[i] += q[j] * (d2 * s * s + d1 * (dv2 - s * s) / d);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------------------------
// 4. Periodic CN block (cross term + counting-function radial block)
// ---------------------------------------------------------------------------------------------

/// The frozen-density CN potential `dE_band/dCN_k` and its first two directional derivatives.
///
/// Mirrors `pbc::hessian::band_cn_potential` (value) and
/// `band_cn_potential_position_derivative` (first), and adds the second. Only off-site image
/// pairs carry geometry: the on-site `dsedcn·P` diagonal is position independent, so it enters
/// the value and drops out of both derivatives.
///
/// Per image pair the geometric factor is `Hs(r)·S(R)` with `Hs = hscale·poly` (the `H0`
/// prefactor *without* the `½(se_i + se_j)` self-energy factor), so
/// `E₁ = Hs₁ S + Hs S₁` and `E₂ = Hs₂ S + 2 Hs₁ S₁ + Hs S₂`.
fn band_cn_potential_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    density: &HashMap<[i32; 3], Matrix>,
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let dsedcn = &scf.bloch.dsedcn;
    let lattice = system.lattice.as_ref().copied().unwrap();
    let mut value = vec![0.0; nat];
    let mut e1 = vec![0.0; nat];
    let mut e2 = vec![0.0; nat];

    let p_origin = &density[&[0, 0, 0]];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            value[shell.atom_index] += dsedcn[ish] * p_origin[(iao, iao)];
        }
    }

    let (atom_aos, atom_min_exp) = ao_tables(basis, nat);
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p_off = &density[&off.n];
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
                        let (hs, hs1r, hs2r, _) =
                            prefactor_radial3(hscale(si, sj, params)?, si, sj, r)?;
                        let (hs1, hs2, _) = radial_chain(hs1r, hs2r, 0.0, r, sdir, dv2);
                        let s1 = pair.d_bra[0].dot(va) + pair.d_ket[0].dot(vb);
                        let s2 = mat3_contract(&pair.h_bra_bra[0], va, va)
                            + 2.0 * mat3_contract(&pair.h_bra_ket[0], va, vb)
                            + mat3_contract(&pair.h_ket_ket[0], vb, vb);
                        let p = p_off[(mu, nu)];
                        let f0 = hs * overlap;
                        let f1 = hs1 * overlap + hs * s1;
                        let f2 = hs2 * overlap + 2.0 * hs1 * s1 + hs * s2;
                        value[a] += dsedcn[si_idx] * p * f0;
                        value[b] += dsedcn[sj_idx] * p * f0;
                        e1[a] += dsedcn[si_idx] * p * f1;
                        e1[b] += dsedcn[sj_idx] * p * f1;
                        e2[a] += dsedcn[si_idx] * p * f2;
                        e2[b] += dsedcn[sj_idx] * p * f2;
                    }
                }
            }
        }
    }
    Ok((value, e1, e2))
}

/// **Periodic fixed-density coordination-number block**, directional second and third.
///
/// One order above `pbc::hessian::cn_fixed_hessian`, which has two pieces:
///
/// * the cross term `M[row][col] = Σ_k (∂CN_k/∂R_row)(∂(dE/dCN_k)/∂R_col)`, accumulated as
///   `M + Mᵀ`, so `vᵀ(M + Mᵀ)v = 2 Σ_k CN₁_k E₁_k` and its derivative is
///   `2 Σ_k (CN₂_k E₁_k + CN₁_k E₂_k)`;
/// * the counting-function radial block weighted by `c = dE/dCN_i + dE/dCN_j`. `c` is itself
///   geometry dependent (it is the frozen-*density* CN potential, not a constant), so the
///   product rule contributes `ċ = E₁_i + E₁_j` times the block's own `vᵀH v`.
///
/// With GFN1's linear CN self-energy (`d²se/dCN² = 0`) there is no further chain term.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pbc_cn_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    coordination_cutoff: f64,
    density: &HashMap<[i32; 3], Matrix>,
    v: &[f64],
) -> Result<DirectionalDerivs> {
    let nat = system.atoms.len();
    let (d_edcn, e1, e2) = band_cn_potential_directional(system, params, scf, pbc, density, v)?;

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

    // Directional first/second derivatives of every CN_k. Each unique pair feeds both members
    // with the same value, matching `coordination_derivatives`' scatter pattern.
    let mut cn1 = vec![0.0; nat];
    let mut cn2 = vec![0.0; nat];
    let mut out = DirectionalDerivs::default();
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
        let dv2 = dv.norm2();
        let s = pair.r_ij.dot(dv) / r;
        let (h1, h2, _) = radial_chain(f1, f2, f3, r, s, dv2);
        cn1[pair.i] += h1;
        cn1[pair.j] += h1;
        cn2[pair.i] += h2;
        cn2[pair.j] += h2;

        // Counting-function radial block, weighted by the geometry-dependent `c`.
        let c = d_edcn[pair.i] + d_edcn[pair.j];
        let cdot = e1[pair.i] + e1[pair.j];
        let pref = f1 / r;
        let dpref = f2 / r - f1 / (r * r);
        let pref2 = f3 / r - 2.0 * f2 / (r * r) + 2.0 * f1 / (r * r * r);
        let unit_second = radial_pair_second_vv(pref, dpref, r, s, dv2);
        let unit_third = radial_pair_third_vv(dpref, pref2, r, s, dv2);
        out.accumulate(c * unit_second, cdot * unit_second + c * unit_third);
    }

    for k in 0..nat {
        out.accumulate(
            2.0 * cn1[k] * e1[k],
            2.0 * (cn2[k] * e1[k] + cn1[k] * e2[k]),
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// 5a. Gamma Bloch H0/S image-pair third contracted with the frozen P/W/V
// ---------------------------------------------------------------------------------------------

/// **Periodic frozen band + Pulay + scalar-overlap third derivative**, directional — the
/// one-order-up twin of `pbc::hessian::band_pulay_fixed_hessian`.
///
/// Per canonical image pair `(a, b+T)` and AO pair `(μ ∈ a, ν ∈ b)` that block contributes
///
/// ```text
///   H[row,col] = C · ∂²S + 2P ( ∂H·∂S|cross + S ∂²H )      (row,col ∈ pair centres)
///   H[row,col] −= P · ∂S_row · ∂σ_col                      (row ∈ pair centres, col ∈ all)
///   C = P(2·H0scale − σ) − 2W ,  σ = V_μ + V_ν
/// ```
///
/// Contracting with `v` twice and differentiating once along `λ` gives the per-pair third
///
/// ```text
///   t = C·S₃ + 2P·S·H₃ + P(6H₁ − V₁)·S₂ + P(6H₂ − V₂)·S₁
/// ```
///
/// **`σ` is frozen inside `C`, and live only in the three-centre channel.** `C` reads the
/// converged `scf.shell_scc_potential` *value*, so `Ċ = 2P·H₁` — there is no `−P·V₁` in it. Only
/// the three-centre term `−P S₁ σ₁` carries geometry-dependent potential, contributing
/// `−P(S₂ V₁ + S₁ V₂)`. That asymmetry is not an accident: it is the same convention the
/// molecular `L_abc` block uses, where the frozen third holds `V` fixed and the full `dV/dR`
/// instead flows through the separate density-path channel (`frozen_hessian_density_path`), so
/// that the caller can feed the *total* potential motion, geometric plus charge response, in one
/// place. Carrying `V₁` in `C` as well double-counts it; the finite-difference gate detects that
/// immediately as a **flat** h-ladder (ratio 1.00 instead of 4.00, ~0.35 % of the block).
///
/// where `S₁..S₃` are directional overlap derivatives from
/// [`contracted_pair_with_third_derivatives`] (bra displaced by `v_a`, ket by `v_b`, the lattice
/// translation held fixed), `H₁..H₃` come from the radial `H0` prefactor ladder, and `V₁`/`V₂`
/// are the shell scalar potential's first/second directional derivatives at frozen charges.
///
/// The `−P S₁ V₂` term is the reason [`shell_potential_second_directional`] exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pbc_band_pulay_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    dens: &FrozenDensityImages,
    v1: &[f64],
    v2: &[f64],
    v: &[f64],
) -> Result<DirectionalDerivs> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let self_energy = &scf.bloch.self_energies;

    let mut vao = vec![0.0; basis.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let (atom_aos, atom_min_exp) = ao_tables(basis, nat);
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut out = DirectionalDerivs::default();

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p0 = &dens.p[&off.n];
        let w0 = &dens.w[&off.n];
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
                        // Zero-overlap AO pairs are NOT skipped: a p_x-p_y pair across a bond
                        // axis has S == 0 but nonzero derivatives, and dropping it breaks
                        // low-symmetry cells (same reasoning as the Hessian block).
                        let pair = contracted_pair_with_third_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        let overlap = pair.moments[0];
                        let coeff = 0.5
                            * (self_energy[si_idx] + self_energy[sj_idx])
                            * hscale(si, sj, params)?;
                        let (hval, hp, hpp, hppp) = prefactor_radial3(coeff, si, sj, r)?;
                        let (h1, h2, h3) = radial_chain(hp, hpp, hppp, r, sdir, dv2);

                        let s1 = pair.d_bra[0].dot(va) + pair.d_ket[0].dot(vb);
                        let s2 = mat3_contract(&pair.h_bra_bra[0], va, va)
                            + 2.0 * mat3_contract(&pair.h_bra_ket[0], va, vb)
                            + mat3_contract(&pair.h_ket_ket[0], vb, vb);
                        let s3 = ten3_contract(&pair.t_bra_bra_bra[0], va, va, va)
                            + 3.0 * ten3_contract(&pair.t_bra_bra_ket[0], va, va, vb)
                            + 3.0 * ten3_contract(&pair.t_bra_ket_ket[0], va, vb, vb)
                            + ten3_contract(&pair.t_ket_ket_ket[0], vb, vb, vb);

                        let p = p0[(mu, nu)];
                        let w = w0[(mu, nu)];
                        let sigma = vao[mu] + vao[nu];
                        let sigma1 = v1[si_idx] + v1[sj_idx];
                        let sigma2 = v2[si_idx] + v2[sj_idx];
                        let c = p * (2.0 * hval - sigma) - 2.0 * w;

                        let second =
                            c * s2 + 2.0 * p * (2.0 * h1 * s1 + overlap * h2) - p * s1 * sigma1;
                        // `sigma` (the potential VALUE) is frozen, so it does not differentiate
                        // out of `c`; `sigma1`/`sigma2` enter only through the three-centre
                        // channel. See this function's docs.
                        let third = c * s3
                            + 2.0 * p * overlap * h3
                            + p * (6.0 * h1 - sigma1) * s2
                            + p * (6.0 * h2 - sigma2) * s1;
                        out.accumulate(second, third);
                    }
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// 5b. SCC2 (second-order electrostatics), real-space Klopman-Ohno part only
// ---------------------------------------------------------------------------------------------

/// **Periodic SCC2 third derivative, real-space QCore block only**, directional.
///
/// Covers exactly the first loop of `pbc::hessian::electrostatic_fixed_hessian`: the frozen
/// shell-charge energy `½ Σ_ij q_i q_j Σ_T rem(|R_i − R_j − T|)` whose radial remainder is the
/// generalised `R^-3` Ewald *real*-space part plus the short-range Klopman-Ohno residual. Charges
/// are frozen (the charge response is the CPXTB half), and the on-site / QCore `k = 0` terms are
/// position independent, so they contribute nothing at any derivative order.
///
/// **Not included** — see [`pbc_scc2_ewald_third_directional_todo`].
pub(crate) fn pbc_scc2_realspace_third_directional(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    v: &[f64],
) -> DirectionalDerivs {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let alpha = resolve_alpha(system, &pbc.ewald);
    let r3_cut = TAU / alpha;
    let sr_cut = pbc.ewald.sr_cutoff;
    let real_cut = r3_cut.max(sr_cut);
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    let nsh = basis.shells.len();
    let mut out = DirectionalDerivs::default();

    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        let ra = system.atoms[ai].position;
        for j in 0..nsh {
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let scale = 0.5 * q[i] * q[j];
            if scale == 0.0 {
                continue;
            }
            let dv = pair_direction(v, ai, aj);
            let dv2 = dv.norm2();
            for (off, t) in offsets.iter().zip(&translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let vec = ra - system.atoms[aj].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let mut d1 = 0.0;
                let mut d2 = 0.0;
                let mut d3 = 0.0;
                if d <= r3_cut {
                    let (_, a1, a2, a3) = qcore_r3_real_value_derivatives3(d, eta, alpha);
                    d1 += QCORE_R3_COEFF * a1;
                    d2 += QCORE_R3_COEFF * a2;
                    d3 += QCORE_R3_COEFF * a3;
                }
                if d <= sr_cut {
                    let (_, b1, b2, b3) = qcore_short_value_derivatives3(d, eta);
                    d1 += b1;
                    d2 += b2;
                    d3 += b3;
                }
                let pref = scale * d1 / d;
                let dpref = scale * (d2 / d - d1 / (d * d));
                let pref2 = scale * (d3 / d - 2.0 * d2 / (d * d) + 2.0 * d1 / (d * d * d));
                let s = vec.dot(dv) / d;
                out.accumulate(
                    radial_pair_second_vv(pref, dpref, d, s, dv2),
                    radial_pair_third_vv(dpref, pref2, d, s, dv2),
                );
            }
        }
    }
    out
}

/// **The Ewald-entangled half of the SCC2 frozen third derivative** — the pieces of
/// `pbc::hessian::electrostatic_fixed_hessian` whose kernel carries the `α`-split:
///
/// 1. **`1/R` Ewald real-space `erfc` sum** — `radial_pair_third_vv` fed by
///    [`ewald_real_value_derivatives3`], per atom pair/image, weight `½ q_A q_B`;
/// 2. **`1/R` Ewald reciprocal structure-factor sum** — `(4π/V) Σ_G w_G cos(G·R_AB)`:
///    `d²/dλ² cos θ = −τ² cos θ`, `d³/dλ³ cos θ = +τ³ sin θ` with `τ = G·(v_A − v_B)`;
/// 3. **QCore `R⁻³` reciprocal sum** — the same phase structure with the
///    `exp1(G²/4α²)/η²` weight, per shell pair;
/// 4. **QCore `k = 0` / self terms** — position independent, contribute zero.
///
/// The *second* orders of the identical kernel set are the ones inside
/// [`shell_potential_second_directional`], which is gated bit-for-bit against the production
/// `gamma_skeleton_derivatives`; the FD gate here closes the value → second → third chain of
/// this energy contraction on the same cutoffs.
pub(crate) fn pbc_scc2_ewald_third_directional(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    v: &[f64],
) -> DirectionalDerivs {
    let (_, second, third) = scc2_ewald_pieces(system, lattice, scf, pbc, v);
    DirectionalDerivs { second, third }
}

/// One sweep over the three geometry-dependent Ewald SCC2 sums, returning the frozen-charge
/// energy piece and its directional second/third — the value is carried so the gate can FD the
/// whole chain from one function.
fn scc2_ewald_pieces(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    v: &[f64],
) -> (f64, f64, f64) {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let q_atom = &scf.atomic_charges;
    let nat = system.atoms.len();
    let nsh = basis.shells.len();
    let alpha = resolve_alpha(system, &pbc.ewald);
    let mut value = 0.0;
    let mut second = 0.0;
    let mut third = 0.0;

    // (1) `1/R` Ewald real-space erfc sum: E = ½ Σ_{A,B,T} q_A q_B erfc(αd)/d.
    let ew_real_cut = TAU / alpha;
    let r_offsets = lattice.image_offsets(ew_real_cut);
    let r_trans: Vec<Vec3> = r_offsets.iter().map(|o| lattice.translation(*o)).collect();
    for a in 0..nat {
        for b in 0..nat {
            let scale = 0.5 * q_atom[a] * q_atom[b];
            if scale == 0.0 {
                continue;
            }
            let dv = pair_direction(v, a, b);
            let dv2 = dv.norm2();
            for t in &r_trans {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > ew_real_cut {
                    continue;
                }
                let (g0, g1, g2, g3) = ewald_real_value_derivatives3(d, alpha);
                value += scale * g0;
                if dv2 == 0.0 {
                    continue;
                }
                let s = vec.dot(dv) / d;
                let pref = scale * g1 / d;
                let dpref = scale * (g2 / d - g1 / (d * d));
                let pref2 = scale * (g3 / d - 2.0 * g2 / (d * d) + 2.0 * g1 / (d * d * d));
                second += radial_pair_second_vv(pref, dpref, d, s, dv2);
                third += radial_pair_third_vv(dpref, pref2, d, s, dv2);
            }
        }
    }

    // (2) `1/R` Ewald reciprocal sum: E = ½ (4π/V) Σ_G w_G Σ_{A,B} q_A q_B cos(G·R_AB).
    let g_cut = 2.0 * alpha * TAU;
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / lattice.volume();
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = four_pi_v * (-g2 * inv_4a2).exp() / g2;
        for a in 0..nat {
            for b in 0..nat {
                let scale = 0.5 * w_g * q_atom[a] * q_atom[b];
                if scale == 0.0 {
                    continue;
                }
                let theta = g.dot(system.atoms[a].position - system.atoms[b].position);
                let tau = g.dot(pair_direction(v, a, b));
                value += scale * theta.cos();
                second += -scale * tau * tau * theta.cos();
                third += scale * tau * tau * tau * theta.sin();
            }
        }
    }

    // (3) QCore `R⁻³` reciprocal sum, per shell pair, weight `exp1(G²/4α²)/η²`.
    let pref0 = QCORE_R3_COEFF * 2.0 * PI / lattice.volume();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff = pref0 * exp1(g.norm2() * inv_4a2);
        for i in 0..nsh {
            if q[i] == 0.0 {
                continue;
            }
            let ai = basis.shells[i].atom_index;
            for j in 0..nsh {
                if q[j] == 0.0 {
                    continue;
                }
                let aj = basis.shells[j].atom_index;
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let scale = 0.5 * coeff * q[i] * q[j] / (eta * eta);
                let theta = phases[i] - phases[j];
                let tau = g.dot(pair_direction(v, ai, aj));
                value += scale * theta.cos();
                second += -scale * tau * tau * theta.cos();
                third += scale * tau * tau * tau * theta.sin();
            }
        }
    }

    (value, second, third)
}

// ---------------------------------------------------------------------------------------------
// First-order response leg (Gamma, integer occupations)
// ---------------------------------------------------------------------------------------------

/// **Directional first-order response** `X¹ = (P¹, W¹, q¹)` at the Gamma point: contract the
/// per-DOF skeleton derivatives with `v` and run ONE integer-occupation CPXTB solve
/// ([`crate::pbc::hessian::gamma_cpxtb_response_directional`]) instead of `3N`.
///
/// Gates: bit-tight linearity against the per-DOF
/// [`crate::pbc::hessian::gamma_cpxtb_density_responses`] contraction, and a physical FD gate of
/// `q¹` against the reconverged SCC shell charges at displaced geometries.
pub(crate) fn gamma_first_order_directional(
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    mos: &crate::pbc::hessian::GammaMos,
    v: &[f64],
) -> Result<(Matrix, Matrix, Vec<f64>)> {
    let n = scf.basis.len();
    let mut f1 = Matrix::zeros(n, n);
    let mut s1 = Matrix::zeros(n, n);
    for (y, &vy) in v.iter().enumerate() {
        if vy == 0.0 {
            continue;
        }
        let fy = &skeleton.fock[y];
        let sy = &skeleton.overlap[y];
        for i in 0..n {
            for j in 0..n {
                f1[(i, j)] += vy * fy[(i, j)];
                s1[(i, j)] += vy * sy[(i, j)];
            }
        }
    }
    crate::pbc::hessian::gamma_cpxtb_response_directional(scf, mos, &f1, &s1)
}

// ---------------------------------------------------------------------------------------------
// Second-order skeleton matrices (F^vv, S^vv)
// ---------------------------------------------------------------------------------------------

/// Directional first and second derivatives of every coordination number, `(CN¹_k, CN²_k)`.
///
/// The scatter pattern is `coordination_derivatives`': each unique pair feeds **both** members
/// with the same scalar, because `dCN_i/dλ` and `dCN_j/dλ` share the radial chain factor
/// (`r̂_ij·(v_i − v_j)` flips sign twice between the two members). Deliberately a separate sweep
/// from the one inside [`pbc_cn_third_directional`] — that one needs `f₃` and the counting-block
/// contraction in the same pass, and fusing the two would make both harder to read. The gate
/// `cn_directional_derivatives_match_fd` finite-differences this helper on its own.
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

/// **Directional SECOND derivatives of the Gamma skeleton matrices**, `(F^vv, S^vv)` — the
/// one-order-up twin of [`crate::pbc::hessian::gamma_skeleton_derivatives`]'s `fock` / `overlap`
/// contracted once with `v`:
///
/// ```text
///   S^vv_μν = Σ_bc v_b v_c ∂²S(Γ)_μν/∂R_b∂R_c
///   F^vv_μν = Σ_bc v_b v_c ∂²F(Γ)_μν/∂R_b∂R_c   at FROZEN shell charges
/// ```
///
/// This is the periodic mirror of the molecular trio
/// `directional_h0_bare_second_matrix` + `directional_h0_cn_block_second_matrix` +
/// `directional_h0_scc_scalar_second_matrix` (and `directional_overlap_second_matrix`) that
/// [`crate::response::charge_space::ChargeSpaceContext::second_order_field`] consumes, so the
/// three pieces are split the same way. With `hs = hscale·poly(r)`, `h = ½(se_i+se_j)·hs`,
/// `Φ = ½(V_μ+V_ν)` and `S₀₁₂` the directional overlap ladder:
///
/// ```text
///   (i)   bare H0  :  h₂·S + 2h₁·S₁ + h·S₂
///   (ii)  CN-se    :  c₂·(hs·S) + 2c₁·(hs₁·S + hs·S₁) ,  c_n = ½(dsedcn_i·CN^n_i + dsedcn_j·CN^n_j)
///   (iii) SCC scalar: −½(V₂_i+V₂_j)·S(Γ) − (V₁_i+V₁_j)·S₁ − ½(V₀_i+V₀_j)·S₂
/// ```
///
/// `V₀` is `scf.shell_scc_potential` (the converged value), `V₁ = v1` is the production skeleton's
/// `shell_potential` contracted with `v` and `V₂ = v2` is
/// [`shell_potential_second_directional`]. Piece (ii) carries only the CN **motion**: the CN
/// value at the reference geometry is already inside `se`, which is why `c₀` never appears —
/// exactly the `c₂·P + 2c₁·P₁` Leibniz remainder the molecular CN block uses.
///
/// # Every geometric factor is live — unlike a re-evaluation of the production skeleton
///
/// All three pieces are the **full** product rule, so the cross terms carry their factor of two.
/// That is *not* what re-evaluating `gamma_skeleton_derivatives` at a displaced geometry against
/// a frozen `scf` produces: that path freezes three reference *values* that are genuinely
/// geometry dependent at fixed charges — `scf.shell_scc_potential` (`V₀`),
/// `scf.bloch.self_energies` (`se`, through `CN`) and `scf.bloch.h_s_gamma_real()` (the `S(Γ)`
/// multiplying the potential derivative). Its `λ`-derivative therefore lands one cross term short
/// in (ii) and (iii) and misses `−(V₁)·S₁` outright. The finite-difference gate
/// `gamma_directional_second_matrices_match_skeleton_fd` refreshes those three fields at the
/// displaced geometry (a rebuilt `BlochBuilder` plus the frozen-charge `periodic_gamma_matrix`
/// potential) before differencing, which is what makes it a gate on this function rather than on
/// the production skeleton's freezing convention.
///
/// # Image convention
///
/// One canonical image pair per unordered pair (`canonical_positive_offset`, `a < b` at the
/// origin) — the [`pbc_band_pulay_third_directional`] sweep — with every AO-pair contribution
/// scattered to **both** `(μ,ν)` and `(ν,μ)`. `S(Γ)` and its derivatives are real symmetric at
/// Gamma and the mirror image pair `(b, a, −T)` carries the transposed value of the same
/// integral, so this reproduces the production skeleton's fully ordered sweep at half the
/// integral cost.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gamma_directional_second_matrices(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    v1: &[f64],
    v2: &[f64],
    v: &[f64],
) -> Result<(Matrix, Matrix)> {
    let lattice = system.lattice.as_ref().copied().unwrap();
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

    let mut fock = Matrix::zeros(n, n);
    let mut s1_mat = Matrix::zeros(n, n);
    let mut s2_mat = Matrix::zeros(n, n);

    let (atom_aos, atom_min_exp) = ao_tables(basis, nat);
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
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

                        fock[(mu, nu)] += f2;
                        fock[(nu, mu)] += f2;
                        s1_mat[(mu, nu)] += s1;
                        s1_mat[(nu, mu)] += s1;
                        s2_mat[(mu, nu)] += s2;
                        s2_mat[(nu, mu)] += s2;
                    }
                }
            }
        }
    }

    // On-site (a == b, T = 0) CN block: the overlap is geometry-rigid there, so only `c₂·S₀`
    // survives — the same term `gamma_skeleton_derivatives` carries one order down.
    if enable_cn {
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for &mu in &atom_aos[a] {
                let si_idx = basis.aos[mu].shell_index;
                for &nu in &atom_aos[a] {
                    let sj_idx = basis.aos[nu].shell_index;
                    let s0 = crate::integrals::contracted_pair(
                        &basis.aos[mu],
                        &basis.aos[nu],
                        ra,
                        ra,
                    )
                    .0;
                    if s0 == 0.0 {
                        continue;
                    }
                    fock[(mu, nu)] +=
                        0.5 * (dsedcn[si_idx] + dsedcn[sj_idx]) * s0 * cn2[a];
                }
            }
        }
    }

    // (iii) SCC scalar potential, over the full folded Gamma overlap (the on-site block
    // included, exactly as the production skeleton's last pass does).
    let (_, s_gamma) = scf.bloch.h_s_gamma_real();
    for mu in 0..n {
        let sh_mu = basis.aos[mu].shell_index;
        for nu in 0..n {
            let sh_nu = basis.aos[nu].shell_index;
            fock[(mu, nu)] += -0.5 * (v2[sh_mu] + v2[sh_nu]) * s_gamma[(mu, nu)]
                - (v1[sh_mu] + v1[sh_nu]) * s1_mat[(mu, nu)]
                - 0.5
                    * (scf.shell_scc_potential[sh_mu] + scf.shell_scc_potential[sh_nu])
                    * s2_mat[(mu, nu)];
        }
    }

    Ok((fock, s2_mat))
}

// ---------------------------------------------------------------------------------------------
// SCC2 charge-path bilinear (two charge vectors)
// ---------------------------------------------------------------------------------------------

/// **Bilinear SCC2 charge-path derivatives**, directional: the frozen-geometry-kernel sums of
/// [`pbc_scc2_realspace_third_directional`] and [`pbc_scc2_ewald_third_directional`] with the
/// quadratic weight `½ q_i q_j` replaced by the symmetrised bilinear `¼(qa_i qb_j + qb_i qa_j)`.
///
/// Returns `(first, second)`: the first and second directional derivatives of
///
/// ```text
///   Ẽ(qa, qb) = ¼ Σ_ij γ_ij(R) (qa_i qb_j + qb_i qa_j)  ==  ½ Σ_ij γ_ij qa_i qb_j
/// ```
///
/// The normalisation is fixed by `Ẽ(q, q) = E_SCC2(q)`, which is what
/// `scc2_bilinear_reduces_to_quadratic` pins. A caller who wants the *unsymmetrised* contraction
/// `Σ_ij ∂ⁿγ_ij qa_i qb_j` — which is what both the charge-path Hessian block and the
/// `∇γ` background family need — therefore uses **twice** these values.
///
/// All four geometry-dependent sums are covered, with the cutoffs and image conventions of the
/// quadratic twins: real-space QCore `R⁻³` remainder + short-range Klopman-Ohno residual (shell
/// pairs), `1/R` Ewald real `erfc` and reciprocal structure factor (atom pairs, on the
/// model-aggregated charges), and the QCore `R⁻³` reciprocal sum (shell pairs). The QCore `k = 0`
/// and self terms are position independent and contribute at no derivative order.
fn scc2_bilinear_pieces(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    qa: &[f64],
    qb: &[f64],
    v: &[f64],
) -> (f64, f64) {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let nat = system.atoms.len();
    let nsh = basis.shells.len();
    let qa_atom = model.atomic_charges(basis, qa);
    let qb_atom = model.atomic_charges(basis, qb);
    let alpha = resolve_alpha(system, &pbc.ewald);
    let mut first = 0.0;
    let mut second = 0.0;

    // (1) Real-space QCore `R⁻³` remainder + short-range Klopman-Ohno residual, per shell pair.
    let r3_cut = TAU / alpha;
    let sr_cut = pbc.ewald.sr_cutoff;
    let real_cut = r3_cut.max(sr_cut);
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        let ra = system.atoms[ai].position;
        for j in 0..nsh {
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let scale = 0.25 * (qa[i] * qb[j] + qb[i] * qa[j]);
            if scale == 0.0 {
                continue;
            }
            let dv = pair_direction(v, ai, aj);
            let dv2 = dv.norm2();
            for (off, t) in offsets.iter().zip(&translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let vec = ra - system.atoms[aj].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let mut d1 = 0.0;
                let mut d2 = 0.0;
                if d <= r3_cut {
                    let (_, a1, a2, _) = qcore_r3_real_value_derivatives3(d, eta, alpha);
                    d1 += QCORE_R3_COEFF * a1;
                    d2 += QCORE_R3_COEFF * a2;
                }
                if d <= sr_cut {
                    let (_, b1, b2, _) = qcore_short_value_derivatives3(d, eta);
                    d1 += b1;
                    d2 += b2;
                }
                let s = vec.dot(dv) / d;
                first += scale * d1 * s;
                second += radial_pair_second_vv(
                    scale * d1 / d,
                    scale * (d2 / d - d1 / (d * d)),
                    d,
                    s,
                    dv2,
                );
            }
        }
    }

    // (2) `1/R` Ewald real-space erfc sum, per atom pair.
    let ew_real_cut = TAU / alpha;
    let r_offsets = lattice.image_offsets(ew_real_cut);
    let r_trans: Vec<Vec3> = r_offsets.iter().map(|o| lattice.translation(*o)).collect();
    for a in 0..nat {
        for b in 0..nat {
            let scale = 0.25 * (qa_atom[a] * qb_atom[b] + qb_atom[a] * qa_atom[b]);
            if scale == 0.0 {
                continue;
            }
            let dv = pair_direction(v, a, b);
            let dv2 = dv.norm2();
            for t in &r_trans {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > ew_real_cut {
                    continue;
                }
                let (_, g1, g2, _) = ewald_real_value_derivatives3(d, alpha);
                let s = vec.dot(dv) / d;
                first += scale * g1 * s;
                second += radial_pair_second_vv(
                    scale * g1 / d,
                    scale * (g2 / d - g1 / (d * d)),
                    d,
                    s,
                    dv2,
                );
            }
        }
    }

    // (3) `1/R` Ewald reciprocal structure-factor sum, per atom pair.
    let g_cut = 2.0 * alpha * TAU;
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / lattice.volume();
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = four_pi_v * (-g2 * inv_4a2).exp() / g2;
        for a in 0..nat {
            let scale0 = 0.25 * w_g;
            for b in 0..nat {
                let scale = scale0 * (qa_atom[a] * qb_atom[b] + qb_atom[a] * qa_atom[b]);
                if scale == 0.0 {
                    continue;
                }
                let theta = g.dot(system.atoms[a].position - system.atoms[b].position);
                let tau = g.dot(pair_direction(v, a, b));
                first += -scale * tau * theta.sin();
                second += -scale * tau * tau * theta.cos();
            }
        }
    }

    // (4) QCore `R⁻³` reciprocal sum, per shell pair.
    let pref0 = QCORE_R3_COEFF * 2.0 * PI / lattice.volume();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff = pref0 * exp1(g.norm2() * inv_4a2);
        for i in 0..nsh {
            let ai = basis.shells[i].atom_index;
            for j in 0..nsh {
                let aj = basis.shells[j].atom_index;
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let scale =
                    0.25 * coeff * (qa[i] * qb[j] + qb[i] * qa[j]) / (eta * eta);
                if scale == 0.0 {
                    continue;
                }
                let theta = phases[i] - phases[j];
                let tau = g.dot(pair_direction(v, ai, aj));
                first += -scale * tau * theta.sin();
                second += -scale * tau * tau * theta.cos();
            }
        }
    }

    (first, second)
}

/// Second directional derivative of the SCC2 charge-path bilinear — see [`scc2_bilinear_pieces`]
/// for the normalisation (`(q, q)` reproduces the frozen quadratic exactly).
pub(crate) fn pbc_scc2_bilinear_second_directional(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    qa: &[f64],
    qb: &[f64],
    v: &[f64],
) -> f64 {
    scc2_bilinear_pieces(system, lattice, scf, pbc, qa, qb, v).1
}

/// First directional derivative of the same bilinear — the `∇γ` object the `kernel_qq`
/// background family contracts.
pub(crate) fn pbc_scc2_bilinear_first_directional(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    qa: &[f64],
    qb: &[f64],
    v: &[f64],
) -> f64 {
    scc2_bilinear_pieces(system, lattice, scf, pbc, qa, qb, v).0
}

// ---------------------------------------------------------------------------------------------
// Response density path (B6) and background motion (bg4)
// ---------------------------------------------------------------------------------------------

/// `Σ_pairs −P_slot_{μν}·(pot_{sh μ} + pot_{sh ν})·(S₁, S₂)` — the shape every `−P·V·∂S`
/// background family shares, at **gradient** (`S₁`) and **Hessian** (`S₂`) level in one sweep.
///
/// The periodic mirror of `cpxtb::background_overlap_gradient_scalar` /
/// `background_overlap_hessian_scalar`, over the canonical image pairs that
/// `build_response_band_pairs` enumerates (so a `pot` slot lands on exactly the pairs the
/// production response gradient scales).
fn gamma_background_overlap_scalars(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    density: &HashMap<[i32; 3], Matrix>,
    pot: &[f64],
    v: &[f64],
) -> (f64, f64) {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let (atom_aos, atom_min_exp) = ao_tables(basis, nat);
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut grad_level = 0.0;
    let mut hess_level = 0.0;

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p_off = &density[&off.n];
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
                let vb = atom_direction(v, b);
                for &mu in &atom_aos[a] {
                    let sh_mu = basis.aos[mu].shell_index;
                    for &nu in &atom_aos[b] {
                        let sh_nu = basis.aos[nu].shell_index;
                        let p = p_off[(mu, nu)];
                        if p == 0.0 {
                            continue;
                        }
                        let pot_pair = pot[sh_mu] + pot[sh_nu];
                        if pot_pair == 0.0 {
                            continue;
                        }
                        let pair = contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        let s1 = pair.d_bra[0].dot(va) + pair.d_ket[0].dot(vb);
                        let s2 = mat3_contract(&pair.h_bra_bra[0], va, va)
                            + 2.0 * mat3_contract(&pair.h_bra_ket[0], va, vb)
                            + mat3_contract(&pair.h_ket_ket[0], vb, vb);
                        let t = -(p * pot_pair);
                        grad_level += t * s1;
                        hess_level += t * s2;
                    }
                }
            }
        }
    }
    (grad_level, hess_level)
}

/// On-site `E'''` chain potential for a charge pair, `E'''_A · qa_A · qb_A` per shell — the
/// periodic mirror of the molecular `kernel_chain_mixed`. This is `∂K/∂q` contracted with the
/// response charges: only the on-site anharmonic block of
/// [`crate::pbc::hessian::periodic_response_kernel`] depends on `q` at all (the Ewald `γ` half is
/// charge independent), so the chain is diagonal in the atom index.
fn gamma_onsite_chain_potential(scf: &PbcSccResult, qa: &[f64], qb: &[f64]) -> Vec<f64> {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let atom_qa = model.atomic_charges(basis, qa);
    let atom_qb = model.atomic_charges(basis, qb);
    basis
        .shells
        .iter()
        .map(|shell| {
            let atom = shell.atom_index;
            if model.atom_shell_counts[atom] == 0 {
                return 0.0;
            }
            let offset = model.atom_offsets[atom];
            let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                model.hardness[offset],
                model.hubbard_derivs[offset],
                model.charge_order,
                scf.atomic_charges[atom],
            );
            third * atom_qa[atom] * atom_qb[atom]
        })
        .collect()
}

/// Replicate one Gamma-point response matrix pair to every image offset, the `dP(T) = dP(Γ)`
/// identity `pbc_gamma_hessian` exploits with `DensityLookup::Uniform`. The image-resolved form
/// is what the frozen builders in this module consume.
pub(crate) fn uniform_density_images(
    lattice: &Lattice,
    ao_cutoff: f64,
    p: &Matrix,
    w: &Matrix,
) -> FrozenDensityImages {
    let offsets = lattice.image_offsets(ao_cutoff);
    let mut pm = HashMap::with_capacity(offsets.len());
    let mut wm = HashMap::with_capacity(offsets.len());
    for off in &offsets {
        pm.insert(off.n, p.clone());
        wm.insert(off.n, w.clone());
    }
    FrozenDensityImages { p: pm, w: wm }
}

/// The two halves of the Gamma response derivative that this module supplies to the caller's
/// third-derivative assembly.
///
/// * `b6` — the **density path**: `D_v` of the periodic response-gradient contraction at a
///   frozen response slot `X¹ = (P¹, W¹, q¹)`, i.e. the pure geometric motion of
///   [`crate::pbc::hessian::response_gradient`];
/// * `bg4` — the **background motion**: the same contraction differentiated through its frozen
///   *reference-state* coefficients instead (gradient level).
///
/// `b6_blocks` and `bg4_families` carry the per-term breakdown in the documented order, for the
/// same reason the molecular assembly keeps `quartic_dg_family_split` around: a discrepancy has
/// to be attributable to one term.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GammaResponsePath {
    /// `Σ b6_blocks`.
    pub b6: f64,
    /// `Σ bg4_families`.
    pub bg4: f64,
    /// `[cn, band_pulay, scc2_charge_path, screening_pulay]`.
    pub b6_blocks: [f64; 4],
    /// `[scc_dp_pot, scc_p0, scc_chain, kernel_qq]`.
    pub bg4_families: [f64; 4],
}

/// **Gamma response density path (`B6`) and background motion (`bg4`)** for one direction.
///
/// # `B6` is exactly `D_v[ g(X¹; R)·v ]`, and that fixes the block list
///
/// `g` here is [`crate::pbc::hessian::response_gradient`] — the object whose `ndof` columns are
/// the entire CPXTB half of `pbc_gamma_hessian`. Holding the response slot `X¹ = (P¹, W¹, q¹)`,
/// the kernel `K`, the reference charges/potential/self-energies frozen and moving **only** the
/// geometry, its contraction with `v` is
///
/// ```text
///   g·v = Σ_pairs [ 2P¹·(hS)₁ − (σ₀ P¹ + 2W¹ + P₀ κ)·S₁ ]      (band / Pulay / screening)
///       + D_v[ Σ_ij γ_ij q⁰_i q¹_j ]                            (electrostatic charge path)
///       + Σ_k E_k(P¹; R)·CN¹_k                                  (CN response)
/// ```
///
/// with `κ = (K q¹)_μ + (K q¹)_ν` and `h = ½(se_i+se_j)·hscale·poly`. One more `D_v` gives the
/// four blocks this function returns:
///
/// ```text
///   ① cn               = Σ_k ( E¹_k·CN¹_k + E_k·CN²_k )
///   ② band_pulay       = pbc_band_pulay_third_directional(X¹, v1 = 0, v2 = 0).second
///   ③ scc2_charge_path = 2·pbc_scc2_bilinear_second_directional(q⁰, q¹)
///   ④ screening_pulay  = −Σ_pairs P₀·κ·S₂
/// ```
///
/// ## Two deliberate departures from the molecular six-block list
///
/// The molecular mirror is `response_coefficient_motion_block_values`, whose six entries are
/// `[cn_h0(P), cross(P), s2path(q₀,q), pulay(P,W), pulay-V(Kq), so_q(q)]`. Two of them collapse
/// here, and both collapses are *forced* by the shape of the periodic response gradient — the
/// finite-difference gate `response_path_b6_matches_response_gradient_fd` is what settled it:
///
/// * **no `∂V/∂R` anywhere in `B6`.** The periodic response column contains no shell-potential
///   derivative at all: the only potential-like object in it is the *frozen* screening vector
///   `K q¹`. So block ② must be evaluated with `v1 = v2 = 0` — feeding it the geometric `V₁`
///   (which is what the frozen third's band/Pulay block does) adds a spurious `−P¹ S₁ V₁`. The
///   geometric motion of `σ₀` is not missing, it simply belongs to the *background* half, where
///   it appears as `bg4.scc_dp_pot`. For the same reason there is no separate `so_q(q¹)` block:
///   the `q¹`-induced potential enters only through `κ`, and its geometric motion is
///   `bg4.scc_p0`. Feeding `so_q` in as well double-counts against ④.
/// * **the CN cross term carries a factor of one, not two.** `pbc_cn_third_directional`'s
///   `second` is `Σ_k E_k·CN²_k + 2 Σ_k CN¹_k·E¹_k`, because the fixed-density CN *Hessian* block
///   is `M + Mᵀ`. The response gradient's CN part is one-sided (`Σ_k E_k·∂CN_k/∂R` with `E` built
///   from `P¹`), so its derivative has a single cross term. Block ① is therefore written out
///   directly rather than reusing the Hessian-shaped builder.
///
/// # `bg4`: the reference-state motion, gradient level
///
/// The four families are the periodic mirror of `cpxtb::ResponseGradientBackgroundMotion`, each
/// one factor of `g` differentiated at frozen geometry-and-response:
///
/// ```text
///   scc_dp_pot = −Σ P¹·(V₁)_pair·S₁          (motion of the reference potential σ₀)
///   scc_p0     = −Σ P¹·(K q¹)_pair·S₁        (motion of the reference density P₀ in ④)
///   scc_chain  = −Σ P₀·(E'''q¹q¹)_pair·S₁    (motion of the kernel itself, ∂K/∂q·q¹)
///   kernel_qq  = 2·pbc_scc2_bilinear_first_directional(q¹, q¹)   (∇γ·q¹q¹)
/// ```
///
/// These are **not** gated against a finite difference here — each is one term of a product-rule
/// split whose sum is only meaningful inside the caller's assembly, where the total is what a
/// physical FD sees. `response_path_bg4_scaling_is_multilinear` pins their structural degrees
/// (linear in `P¹`, linear/quadratic in `q¹`) so a mis-slotted argument cannot pass unnoticed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gamma_response_path_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    dens0: &FrozenDensityImages,
    dens1: &FrozenDensityImages,
    q1: &[f64],
    v1_pot: &[f64],
    v: &[f64],
) -> Result<GammaResponsePath> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let nat = system.atoms.len();
    let nsh = scf.basis.shells.len();
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let kernel = crate::pbc::hessian::periodic_response_kernel(scf);
    let kq1 = crate::linalg::matrix_vector_product(&kernel, q1)?;

    // ① CN response block (one-sided cross term; see the docs).
    let cn_block = if enable_cn {
        let (e0, e1, _) =
            band_cn_potential_directional(system, params, scf, pbc, &dens1.p, v)?;
        let (cn1, cn2) =
            cn_directional_derivatives(system, options.hamiltonian.coordination_cutoff, v)?;
        (0..nat).map(|k| e1[k] * cn1[k] + e0[k] * cn2[k]).sum()
    } else {
        0.0
    };

    // ② band + Pulay + scalar-overlap at the response slot, with NO potential motion.
    let no_potential = vec![0.0; nsh];
    let band_block = pbc_band_pulay_third_directional(
        system,
        params,
        scf,
        pbc,
        dens1,
        &no_potential,
        &no_potential,
        v,
    )?
    .second;

    // ③ electrostatic charge path `Σ_ij ∂²γ_ij q⁰_i q¹_j` (twice the symmetrised bilinear).
    let scc2_block = 2.0
        * pbc_scc2_bilinear_second_directional(
            system,
            &lattice,
            scf,
            pbc,
            &scf.shell_charges,
            q1,
            v,
        );

    // ④ screening Pulay `−Σ P₀·(K q¹)·S₂`.
    let (_, screen_block) =
        gamma_background_overlap_scalars(system, scf, pbc, &dens0.p, &kq1, v);

    // bg4: reference-state motion at gradient level.
    let (dp_pot, _) = gamma_background_overlap_scalars(system, scf, pbc, &dens1.p, v1_pot, v);
    let (p0_family, _) = gamma_background_overlap_scalars(system, scf, pbc, &dens1.p, &kq1, v);
    let chain = gamma_onsite_chain_potential(scf, q1, q1);
    let (chain_family, _) =
        gamma_background_overlap_scalars(system, scf, pbc, &dens0.p, &chain, v);
    let kernel_qq =
        2.0 * pbc_scc2_bilinear_first_directional(system, &lattice, scf, pbc, q1, q1, v);

    let b6_blocks = [cn_block, band_block, scc2_block, screen_block];
    let bg4_families = [dp_pot, p0_family, chain_family, kernel_qq];
    Ok(GammaResponsePath {
        b6: b6_blocks.iter().sum(),
        bg4: bg4_families.iter().sum(),
        b6_blocks,
        bg4_families,
    })
}

// ---------------------------------------------------------------------------------------------
// Shared helpers and the frozen bundle
// ---------------------------------------------------------------------------------------------

/// Per-atom AO index lists and the minimum primitive exponent per atom, the two lookup tables
/// every image-pair loop in `pbc::hessian` builds before its sweep.
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

/// Real-space frozen densities `P(T)` / `W(T)` over the AO-cutoff image set, the inverse Bloch
/// transform `M(T) = Σ_k w_k Re[M(k) e^{-ik·T}]`. Same convention as the Hessian's private
/// `realspace_images`; at a Gamma-only mesh every image equals `M(Gamma)`.
pub(crate) fn gamma_realspace_densities(
    scf: &PbcSccResult,
    lattice: &Lattice,
    ao_cutoff: f64,
) -> FrozenDensityImages {
    let build = |per_k: &[crate::pbc::complex::CMatrix]| {
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
    };
    FrozenDensityImages {
        p: build(&scf.density_k),
        w: build(&scf.ew_density_k),
    }
}

/// Per-component breakdown of the frozen directional third, so a caller can attribute a
/// discrepancy to one block rather than to the bundle.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GammaFrozenThird {
    pub repulsion: DirectionalDerivs,
    pub halogen: DirectionalDerivs,
    pub dispersion: DirectionalDerivs,
    pub coordination: DirectionalDerivs,
    pub band_pulay: DirectionalDerivs,
    pub scc2_realspace: DirectionalDerivs,
    pub scc2_ewald: DirectionalDerivs,
}

impl GammaFrozenThird {
    /// Sum over every frozen component. The Gamma-point **response** derivative is the one
    /// contribution added by the caller.
    pub(crate) fn total(&self) -> DirectionalDerivs {
        self.repulsion
            + self.halogen
            + self.dispersion
            + self.coordination
            + self.band_pulay
            + self.scc2_realspace
            + self.scc2_ewald
    }
}

/// Assemble every frozen component this module implements for one direction `v`.
///
/// `skeleton` must be `gamma_skeleton_derivatives` evaluated at **this** geometry against the
/// frozen `scf`; that is where the production `∂V/∂R` comes from.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pbc_gamma_frozen_third_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    pbc: &PbcOptions,
    coordination_cutoff: f64,
    enable_cn: bool,
    enable_dispersion: bool,
    dispersion_reference: Option<&str>,
    v: &[f64],
) -> Result<GammaFrozenThird> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let dens = gamma_realspace_densities(scf, &lattice, pbc.ao_cutoff);
    let v1 = shell_potential_first_directional(skeleton, v);
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

// ---------------------------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::ElectronicOptions;
    use crate::pbc::hessian::gamma_skeleton_derivatives;
    use crate::pbc::scf::run_pbc_scc;
    use crate::pbc::{EwaldOptions, KMesh};

    /// Primitive 2-atom fcc diamond carrying **both** an asymmetric cell strain and an internal
    /// displacement. The strain destroys the cubic point group (so no frozen block can pass by
    /// accidental cancellation), and the internal displacement lifts diamond's triply degenerate
    /// `t2` frontier orbitals — see the module docs of `tests/pbc_third_derivative.rs` for why an
    /// undistorted diamond cell is a bad periodic fixture.
    const DIAMOND_SKEW: &str = "2\n\
Lattice=\"0.06 1.83 1.75 1.75 0.04 1.81 1.82 1.76 0.03\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.930000 0.880000 0.905000\n";

    /// Distorted zincblende BN, skewed the same way. A **heteronuclear** partner for the diamond
    /// cell: the charge transfer onto N gives the SCC2 and scalar-potential channels real weight,
    /// which they almost entirely lack in homonuclear diamond.
    ///
    /// (A distorted primitive rocksalt LiH was tried first and rejected: at the fcc primitive
    /// cell's ~2.9 A vectors the diffuse H `s` shell overlaps its own periodic images hard enough
    /// that the Gamma overlap matrix loses positive definiteness — `eigenvalue 0 = -1.8e-1` — so
    /// the fixture never reaches an SCC at all.)
    const BN_SKEW: &str = "2\n\
Lattice=\"0.06 1.86 1.78 1.78 0.04 1.84 1.85 1.79 0.03\" pbc=\"T T T\"\n\
B 0.000000 0.000000 0.000000\n\
N 0.940000 0.890000 0.920000\n";

    /// A periodic C-Br...O chain: the only fixture here with a live halogen-bond term (GFN1 gives
    /// F and Cl a zero `CXB`, so the donor must be Br or I, and the acceptor must be N/O/P/S).
    /// Br sits 1.95 A from its C neighbour and 2.65 A from the O acceptor.
    const BR_CHAIN: &str = "4\n\
Lattice=\"9.0 0.0 0.0 0.0 9.5 0.0 0.0 0.0 8.0\" pbc=\"T T T\"\n\
C  0.000000 0.000000 0.000000\n\
Br 0.100000 0.050000 1.950000\n\
O  0.200000 0.150000 4.600000\n\
H  0.950000 0.200000 5.100000\n";

    /// The two SCC-bearing fixtures every electronic-block gate loops over.
    const SCF_FIXTURES: [(&str, &str); 2] = [("diamond-skew", DIAMOND_SKEW), ("BN-skew", BN_SKEW)];

    /// The classical blocks need no SCC, so they also run on the halogen-bonded chain.
    const CLASSICAL_FIXTURES: [(&str, &str); 3] = [
        ("diamond-skew", DIAMOND_SKEW),
        ("BN-skew", BN_SKEW),
        ("CBr-O-chain", BR_CHAIN),
    ];

    fn params() -> Gfn1Parameters {
        Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
    }

    /// Tight SCC with the electronic temperature pinned to exactly zero: these are frozen-block
    /// gates and must not drift onto the periodic finite-temperature branch.
    fn electronic() -> ElectronicOptions {
        ElectronicOptions {
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            max_scc: 500,
            electronic_temperature: 0.0,
            ..ElectronicOptions::default()
        }
    }

    /// Lean real-space cutoffs, purely for test speed. Every gate here is a self-consistency
    /// statement (analytic third == central difference of the same block's directional second),
    /// so the cutoff choice cannot affect whether it passes — only how long it takes.
    fn pbc_opts() -> PbcOptions {
        PbcOptions {
            kmesh: KMesh::gamma(),
            ao_cutoff: 9.0,
            ewald: EwaldOptions {
                real_cutoff: 14.0,
                sr_cutoff: 8.0,
                ..EwaldOptions::default()
            },
        }
    }

    fn system_of(xyz: &str) -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(xyz, 0.0, false).expect("fixture parse")
    }

    /// Deterministic direction in `[-1, 1)^ndof`. Deliberately sign-mixed: a direction that
    /// happened to be near-uniform would be near a rigid translation, where every frozen block
    /// vanishes identically and the gates would be vacuous.
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

    /// Rigid translation of every atom by the same 3-vector — the acoustic-sum-rule direction.
    fn rigid_translation(ndof: usize) -> Vec<f64> {
        let step = [0.37, -0.61, 0.22];
        (0..ndof).map(|i| step[i % 3]).collect()
    }

    fn displaced(system: &PeriodicSystem, v: &[f64], lambda: f64) -> PeriodicSystem {
        let mut out = system.clone();
        for (a, atom) in out.atoms.iter_mut().enumerate() {
            atom.position = atom.position
                + Vec3::new(v[3 * a], v[3 * a + 1], v[3 * a + 2]) * lambda;
        }
        out
    }

    fn central(f: &dyn Fn(f64) -> f64, h: f64) -> f64 {
        (f(h) - f(-h)) / (2.0 * h)
    }

    /// The standard gate: a central difference of the block's own directional **second**
    /// derivative must reproduce the analytic **third**, and halving the step must shrink the
    /// error by ~4.
    ///
    /// The **ratio is the real discriminator**, not the tolerance. A central difference is
    /// `O(h²)`, so a correct analytic third gives ~4; a *flat* ratio means the analytic third is
    /// missing a term (the FD is converging to something else), and a `1/h` blow-up means the
    /// difference has dropped onto the round-off floor. The magnitude test is therefore
    /// **relative**: the residual FD truncation error scales with the size of the block, and an
    /// absolute bound would just be a disguised statement about which fixture is largest.
    /// The magnitude test uses the **Richardson extrapolation** `(4·fd(h/2) − fd(h))/3` rather
    /// than `fd(h/2)` itself. Both steps are in the `O(h²)` regime (that is what the ratio
    /// asserts), so the extrapolant is `O(h⁴)` and the assertion stops being a disguised
    /// statement about the step size. Without it a block whose third derivative is small but
    /// whose *fourth* is not — the periodic D3 term on these cells, third `2e-5`, raw truncation
    /// `2.5e-8` — would fail a tight bound while converging at a textbook ratio of 4.00.
    fn assert_ladder(
        name: &str,
        f: &dyn Fn(f64) -> f64,
        analytic: f64,
        h: f64,
        rel_tol: f64,
        abs_tol: f64,
    ) {
        let fd_h = central(f, h);
        let fd_h2 = central(f, 0.5 * h);
        let e1 = (fd_h - analytic).abs();
        let e2 = (fd_h2 - analytic).abs();
        let ratio = if e2 > 0.0 { e1 / e2 } else { f64::INFINITY };
        let richardson = (4.0 * fd_h2 - fd_h) / 3.0;
        let e_rich = (richardson - analytic).abs();
        println!(
            "{name:<44} analytic={analytic:+.10e} err(h)={e1:.3e} err(h/2)={e2:.3e} \
             ratio={ratio:.2} richardson_err={e_rich:.3e}"
        );
        let bound = rel_tol * analytic.abs() + abs_tol;
        assert!(
            e_rich <= bound,
            "{name}: |richardson - analytic| = {e_rich:.3e} exceeds {bound:.3e} \
             (analytic {analytic:.10e}, richardson {richardson:.10e})"
        );
        // Lower bound only. The failure modes worth catching both *depress* the ratio: a missing
        // term in the analytic third pins it at ~1 (the difference converges to a limit that is
        // not the analytic value), and a round-off-dominated difference drives it below 1 (the
        // central-difference noise floor grows like `eps/h`). A ratio well *above* 4 is benign —
        // the leading `O(h²)` error happened to cancel at the smaller step, which is common for a
        // near-zero block (diamond's D3 third gives ~90, BN's ~800). The Richardson bound above
        // is what constrains those.
        assert!(
            e2 <= 1.0e-11 * analytic.abs().max(1.0e-6) || ratio >= 2.5,
            "{name}: h-ladder ratio {ratio:.2} is below 2.5 (errors {e1:.3e} -> {e2:.3e}); \
             a flat ratio means a missing term in the analytic third, a falling one means the \
             difference has hit the round-off floor"
        );
    }

    /// The element-wise twin of [`assert_ladder`] for a matrix-valued first derivative: the
    /// central difference of `at(λ)` must reproduce `analytic` in **every** element. The reported
    /// element is the one with the largest Richardson residual, and the ratio is read off that
    /// same element (a matrix whose worst element converges at 4 has no worse offender).
    fn assert_matrix_ladder(
        name: &str,
        at: &dyn Fn(f64) -> Matrix,
        analytic: &Matrix,
        h: f64,
        rel_tol: f64,
    ) {
        let n = analytic.rows();
        let (p1, m1) = (at(h), at(-h));
        let (p2, m2) = (at(0.5 * h), at(-0.5 * h));
        let mut scale = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(analytic[(i, j)].abs());
            }
        }
        assert!(scale > 1.0e-10, "{name}: the matrix is identically zero, the gate is vacuous");
        let mut worst = (0.0_f64, 0.0_f64, 0.0_f64, 0usize, 0usize);
        for i in 0..n {
            for j in 0..n {
                let a = analytic[(i, j)];
                let fd1 = (p1[(i, j)] - m1[(i, j)]) / (2.0 * h);
                let fd2 = (p2[(i, j)] - m2[(i, j)]) / h;
                let rich = (4.0 * fd2 - fd1) / 3.0;
                let e_rich = (rich - a).abs();
                if e_rich > worst.2 {
                    worst = ((fd1 - a).abs(), (fd2 - a).abs(), e_rich, i, j);
                }
            }
        }
        let ratio = if worst.1 > 0.0 {
            worst.0 / worst.1
        } else {
            f64::INFINITY
        };
        println!(
            "{name:<44} max|analytic|={scale:.4e} worst=({},{}) err(h)={:.3e} \
             err(h/2)={:.3e} ratio={ratio:.2} richardson_err={:.3e}",
            worst.3, worst.4, worst.0, worst.1, worst.2
        );
        assert!(
            worst.2 <= rel_tol * scale,
            "{name}: worst element ({}, {}) Richardson residual {:.3e} exceeds {:.3e}",
            worst.3,
            worst.4,
            worst.2,
            rel_tol * scale
        );
        assert!(
            worst.1 <= 1.0e-11 * scale || ratio >= 2.5,
            "{name}: h-ladder ratio {ratio:.2} at element ({}, {}) is below 2.5 (errors \
             {:.3e} -> {:.3e}); a flat ratio means a missing term in the analytic second",
            worst.3,
            worst.4,
            worst.0,
            worst.1
        );
    }

    // -- Radial ladders -----------------------------------------------------------------------

    /// Every kernel whose third radial derivative is written by hand in this module is checked
    /// against a central difference of the second derivative it extends. Fast (no SCF), so this
    /// runs in the default suite rather than behind `#[ignore]`.
    #[test]
    fn radial_kernel_third_derivatives_match_fd() {
        let eta = 0.45;
        let alpha = 0.30;
        for &r in &[0.8_f64, 1.7, 3.1, 5.4] {
            let h = 1.0e-5 * r;
            let checks: [(&str, Box<dyn Fn(f64) -> (f64, f64, f64, f64)>); 3] = [
                ("ko", Box::new(move |x| ko_value_derivatives3(x, eta))),
                (
                    "qcore_short",
                    Box::new(move |x| qcore_short_value_derivatives3(x, eta)),
                ),
                (
                    "qcore_r3_real",
                    Box::new(move |x| qcore_r3_real_value_derivatives3(x, eta, alpha)),
                ),
            ];
            for (name, f) in &checks {
                let fd = (f(r + h).2 - f(r - h).2) / (2.0 * h);
                let an = f(r).3;
                let scale = an.abs().max(1.0);
                assert!(
                    (fd - an).abs() <= 1.0e-5 * scale,
                    "{name} f''' at r={r}: analytic {an:.10e} vs fd {fd:.10e}"
                );
            }
            // CN counting function.
            let cn = |x: f64| cn_count_value_derivatives3(16.0, x, 3.2);
            let fd = (cn(r + h).1 - cn(r - h).1) / (2.0 * h);
            let an = cn(r).2;
            assert!(
                (fd - an).abs() <= 1.0e-5 * an.abs().max(1.0),
                "cn_count f''' at r={r}: analytic {an:.10e} vs fd {fd:.10e}"
            );
        }
    }

    /// The `*_value_derivatives3` ladders in this module re-derive kernels that `pbc::ewald`
    /// already provides to second order. Their first three components must reproduce the
    /// production functions **bit for bit** — otherwise the third derivative would be the exact
    /// derivative of the wrong second derivative, and the self-consistency gates above would
    /// happily pass while the block disagreed with `electrostatic_fixed_hessian`.
    #[test]
    fn radial_kernel_ladders_agree_with_production_second_order() {
        use crate::pbc::ewald::{qcore_r3_real_value_derivatives, qcore_short_value_derivatives};
        for &eta in &[0.30_f64, 0.45, 0.72] {
            for &alpha in &[0.20_f64, 0.35] {
                for &r in &[0.8_f64, 1.7, 3.1, 5.4, 9.0] {
                    let (v, d1, d2, _) = qcore_short_value_derivatives3(r, eta);
                    assert_eq!(
                        (v, d1, d2),
                        qcore_short_value_derivatives(r, eta),
                        "qcore_short ladder drifted at r={r}, eta={eta}"
                    );
                    let (v, d1, d2, _) = qcore_r3_real_value_derivatives3(r, eta, alpha);
                    assert_eq!(
                        (v, d1, d2),
                        qcore_r3_real_value_derivatives(r, eta, alpha),
                        "qcore_r3_real ladder drifted at r={r}, eta={eta}, alpha={alpha}"
                    );
                }
            }
        }
    }

    /// The `H0` prefactor ladder, checked against a central difference on a real shell pair.
    #[test]
    fn prefactor_radial3_third_matches_fd() {
        let params = params();
        let system = system_of(DIAMOND_SKEW);
        let basis = BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
            .expect("basis");
        let si = &basis.shells[0];
        let sj = &basis.shells[basis.shells.len() - 1];
        for &r in &[1.6_f64, 2.9, 4.3] {
            let h = 1.0e-5 * r;
            let f = |x: f64| prefactor_radial3(0.7, si, sj, x).unwrap();
            let fd = (f(r + h).2 - f(r - h).2) / (2.0 * h);
            let an = f(r).3;
            assert!(
                (fd - an).abs() <= 1.0e-5 * an.abs().max(1.0),
                "prefactor_radial3 f''' at r={r}: analytic {an:.10e} vs fd {fd:.10e}"
            );
        }
    }

    /// The two directional collapses every radial block relies on:
    /// [`radial_chain`] (`f(r(λ))` to third order) and [`radial_pair_third_vv`]
    /// (`d/dλ` of the pair block's `vᵀHv`). Checked against finite differences of a concrete
    /// kernel on a concrete pair geometry.
    #[test]
    fn radial_directional_collapses_match_fd() {
        let eta = 0.5;
        let r0 = Vec3::new(1.3, -0.7, 2.1);
        let dv = Vec3::new(0.31, 0.72, -0.45);
        let d2 = dv.norm2();
        let at = |lam: f64| {
            let vec = r0 + dv * lam;
            let r = vec.norm();
            (vec, r, vec.dot(dv) / r, ko_value_derivatives3(r, eta))
        };
        let (_, r, s, (_, f1, f2, f3)) = at(0.0);

        // radial_chain: h1/h2/h3 vs FD of the value/h1/h2.
        let (h1, h2, h3) = radial_chain(f1, f2, f3, r, s, d2);
        let value_at = |lam: f64| ko_value_derivatives3(at(lam).1, eta).0;
        let h1_at = |lam: f64| {
            let (_, r, s, (_, a1, a2, a3)) = at(lam);
            radial_chain(a1, a2, a3, r, s, d2).0
        };
        let h2_at = |lam: f64| {
            let (_, r, s, (_, a1, a2, a3)) = at(lam);
            radial_chain(a1, a2, a3, r, s, d2).1
        };
        assert_ladder("radial_chain h1", &value_at, h1, 1.0e-3, 1.0e-6, 1.0e-12);
        assert_ladder("radial_chain h2", &h1_at, h2, 1.0e-3, 1.0e-6, 1.0e-12);
        assert_ladder("radial_chain h3", &h2_at, h3, 1.0e-3, 1.0e-6, 1.0e-12);

        // radial_pair_third_vv vs FD of radial_pair_second_vv.
        let second_at = |lam: f64| {
            let (_, r, s, (_, a1, a2, _)) = at(lam);
            radial_pair_second_vv(a1 / r, a2 / r - a1 / (r * r), r, s, d2)
        };
        let analytic = radial_pair_third_vv(
            f2 / r - f1 / (r * r),
            f3 / r - 2.0 * f2 / (r * r) + 2.0 * f1 / (r * r * r),
            r,
            s,
            d2,
        );
        assert_ladder("radial_pair_third_vv", &second_at, analytic, 1.0e-3, 1.0e-6, 1.0e-12);
    }

    // -- Classical periodic blocks ------------------------------------------------------------

    /// Shared driver for the three classical blocks: the analytic directional third must be the
    /// central difference of `vᵀ H_periodic v`, where `H_periodic` is the **production** periodic
    /// Hessian that `pbc_gamma_hessian` adds verbatim. Passing this is what proves the image sums
    /// inside `*_third_derivative` match their Hessian twins'.
    fn classical_gate(
        label: &str,
        fixtures: &[(&str, &str)],
        h: f64,
        rel_tol: f64,
        abs_tol: f64,
        build: impl Fn(&PeriodicSystem, &Gfn1Parameters, &[f64]) -> DirectionalDerivs,
    ) {
        let params = params();
        for (name, xyz) in fixtures {
            let system = system_of(xyz);
            let v = direction(3 * system.atoms.len(), 7);
            let analytic = build(&system, &params, &v);
            assert!(
                analytic.second.abs() > 1.0e-12,
                "{label}/{name}: the block is identically zero, the gate would be vacuous"
            );
            let second_at = |lam: f64| build(&displaced(&system, &v, lam), &params, &v).second;
            assert_ladder(
                &format!("{label}/{name}"),
                &second_at,
                analytic.third,
                h,
                rel_tol,
                abs_tol,
            );
        }
    }

    #[test]
    #[ignore = "periodic third-derivative gate: seconds of image sums per fixture"]
    fn repulsion_third_matches_periodic_hessian_fd() {
        classical_gate(
            "repulsion",
            &CLASSICAL_FIXTURES,
            2.0e-3,
            1.0e-6,
            1.0e-9,
            |s, p, v| pbc_repulsion_third_directional(s, p, v).unwrap(),
        );
    }

    #[test]
    #[ignore = "periodic third-derivative gate: seconds of image sums per fixture"]
    fn halogen_third_matches_periodic_hessian_fd() {
        // Only the Br chain has a live halogen term; the other fixtures would be vacuous.
        classical_gate(
            "halogen",
            &[("CBr-O-chain", BR_CHAIN)],
            2.0e-3,
            1.0e-6,
            1.0e-9,
            |s, p, v| pbc_halogen_third_directional(s, p, v).unwrap(),
        );
    }

    #[test]
    #[ignore = "periodic third-derivative gate: dense Jet3 over image pairs and ATM triples"]
    fn dispersion_third_matches_periodic_hessian_fd() {
        // `abs_tol` is looser here than for the other classical blocks on purpose: the D3 third
        // derivative is genuinely near zero on these cells (`-1.5e-8` on BN), so a relative
        // bound is meaningless and the useful statement is that the block stays negligible.
        classical_gate(
            "dispersion",
            &CLASSICAL_FIXTURES,
            2.0e-3,
            1.0e-6,
            1.0e-8,
            |s, p, v| pbc_dispersion_third_directional(s, p, None, v).unwrap(),
        );
    }

    // -- Frozen electronic blocks -------------------------------------------------------------

    /// One converged fixture plus the frozen fields every electronic block reads. Displacing the
    /// geometry while keeping `scf` fixed is exactly the "frozen block" definition: `P`, `W`, the
    /// shell charges and the self-energies stay at their reference values, and only the geometry
    /// (hence the integrals and `∂V/∂R`) moves.
    struct Frozen {
        system: PeriodicSystem,
        params: Gfn1Parameters,
        scf: PbcSccResult,
        opts: ElectronicOptions,
        pbc: PbcOptions,
        dens: FrozenDensityImages,
        v: Vec<f64>,
    }

    impl Frozen {
        fn build(xyz: &str, seed: u64) -> Self {
            let system = system_of(xyz);
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("periodic SCC");
            assert!(scf.converged, "fixture SCC did not converge");
            let lattice = system.lattice.as_ref().copied().unwrap();
            let dens = gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff);
            let v = direction(3 * system.atoms.len(), seed);
            Self {
                system,
                params,
                scf,
                opts,
                pbc,
                dens,
                v,
            }
        }

        fn lattice(&self) -> Lattice {
            self.system.lattice.as_ref().copied().unwrap()
        }

        fn at(&self, lambda: f64) -> PeriodicSystem {
            displaced(&self.system, &self.v, lambda)
        }

        /// Production `∂V_shell/∂R` at a displaced geometry, contracted with `v`, charges frozen.
        /// `gamma_skeleton_derivatives` reads its geometry from `system` and everything else from
        /// `scf`, so handing it a displaced system with the reference SCC is precisely the
        /// frozen-charge derivative the band/Pulay block consumes.
        fn v1_at(&self, lambda: f64) -> Vec<f64> {
            let sys = self.at(lambda);
            let sk = gamma_skeleton_derivatives(&sys, &self.params, &self.scf, &self.opts, &self.pbc)
                .expect("skeleton");
            shell_potential_first_directional(&sk, &self.v)
        }

        fn v2_at(&self, lambda: f64) -> Vec<f64> {
            shell_potential_second_directional(
                &self.at(lambda),
                &self.lattice(),
                &self.scf,
                &self.pbc,
                &self.v,
            )
        }

        fn band_pulay_at(&self, lambda: f64) -> DirectionalDerivs {
            let sys = self.at(lambda);
            let v1 = self.v1_at(lambda);
            let v2 = self.v2_at(lambda);
            pbc_band_pulay_third_directional(
                &sys, &self.params, &self.scf, &self.pbc, &self.dens, &v1, &v2, &self.v,
            )
            .expect("band/Pulay directional")
        }

        fn cn_at(&self, lambda: f64) -> DirectionalDerivs {
            pbc_cn_third_directional(
                &self.at(lambda),
                &self.params,
                &self.scf,
                &self.pbc,
                self.opts.hamiltonian.coordination_cutoff,
                &self.dens.p,
                &self.v,
            )
            .expect("CN directional")
        }

        /// The frozen SCC with every geometry-dependent **reference value** refreshed at the
        /// displaced geometry, at frozen charges and frozen densities.
        ///
        /// Three fields of `PbcSccResult` are cached values of geometry-dependent quantities:
        /// `bloch.self_energies` (through `CN`), `bloch`'s `S(Γ)`, and `shell_scc_potential`
        /// (`= Γ(R)·q` plus the geometry-independent on-site anharmonic shift). Re-running
        /// `gamma_skeleton_derivatives` against an *unrefreshed* `scf` at a displaced geometry
        /// therefore yields a hybrid — reference values times displaced derivatives — whose
        /// `λ`-derivative is not the frozen-charge second derivative of anything. Refreshing them
        /// here is what makes the finite difference a gate on
        /// [`gamma_directional_second_matrices`] rather than on that freezing convention.
        /// `frozen_scf_refresh_is_identity_at_zero` pins the reconstruction at `λ = 0`.
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

        /// `(F¹, S¹)` — the production skeleton at the refreshed displaced geometry, contracted
        /// with `v`. This is the object [`gamma_directional_second_matrices`] differentiates.
        fn skeleton_directional_at(&self, lambda: f64) -> (Matrix, Matrix) {
            let (sys, scf) = self.refreshed_at(lambda);
            let sk = gamma_skeleton_derivatives(&sys, &self.params, &scf, &self.opts, &self.pbc)
                .expect("skeleton");
            let n = scf.basis.len();
            let mut f1 = Matrix::zeros(n, n);
            let mut s1 = Matrix::zeros(n, n);
            for (y, &vy) in self.v.iter().enumerate() {
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
            (f1, s1)
        }

        fn scc2_at(&self, lambda: f64) -> DirectionalDerivs {
            pbc_scc2_realspace_third_directional(
                &self.at(lambda),
                &self.lattice(),
                &self.scf,
                &self.pbc,
                &self.v,
            )
        }

        /// The Ewald-entangled SCC2 sums: frozen-charge energy piece, second and third, at a
        /// displaced geometry.
        fn scc2_ewald_at(&self, lambda: f64) -> (f64, f64, f64) {
            scc2_ewald_pieces(&self.at(lambda), &self.lattice(), &self.scf, &self.pbc, &self.v)
        }
    }

    /// `V₂` (this module) must be the central difference of the **production** `V₁`
    /// (`gamma_skeleton_derivatives(...).shell_potential`) at frozen charges. Gates the periodic
    /// gamma kernel's second derivative — real `erfc`, reciprocal phases, QCore `R^-3` real and
    /// reciprocal, and the short-range Klopman-Ohno residual — in one shot.
    #[test]
    #[ignore = "periodic third-derivative gate: 4 skeleton evaluations per fixture"]
    fn shell_potential_second_matches_production_first_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 11);
            let analytic = shell_potential_second_directional(
                &f.system,
                &f.lattice(),
                &f.scf,
                &f.pbc,
                &f.v,
            );
            let h = 2.0e-3;
            let (vp, vm) = (f.v1_at(h), f.v1_at(-h));
            let (vp2, vm2) = (f.v1_at(0.5 * h), f.v1_at(-0.5 * h));
            let mut worst = (0.0_f64, 0.0_f64, 0usize);
            for s in 0..analytic.len() {
                let e1 = ((vp[s] - vm[s]) / (2.0 * h) - analytic[s]).abs();
                let e2 = ((vp2[s] - vm2[s]) / h - analytic[s]).abs();
                if e2 > worst.1 {
                    worst = (e1, e2, s);
                }
            }
            let ratio = worst.0 / worst.1;
            let scale = analytic.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            println!(
                "shell_potential V2/{name:<12} max|V2|={scale:.4e} worst shell={} \
                 err(h)={:.3e} err(h/2)={:.3e} ratio={ratio:.2}",
                worst.2, worst.0, worst.1
            );
            assert!(scale > 1.0e-10, "{name}: V2 is identically zero, gate is vacuous");
            assert!(
                worst.1 <= 1.0e-4 * scale,
                "{name}: worst shell V2 error {:.3e} exceeds 1e-4 x {scale:.3e}",
                worst.1
            );
            assert!(
                (2.5..=6.5).contains(&ratio),
                "{name}: V2 h-ladder ratio {ratio:.2} is not ~4"
            );
        }
    }

    #[test]
    #[ignore = "periodic third-derivative gate: 5 image-pair sweeps per fixture"]
    fn cn_third_matches_block_second_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 13);
            let analytic = f.cn_at(0.0);
            assert!(
                analytic.second.abs() > 1.0e-12,
                "{name}: CN block is identically zero, gate is vacuous"
            );
            let second_at = |lam: f64| f.cn_at(lam).second;
            assert_ladder(
                &format!("cn/{name}"),
                &second_at,
                analytic.third,
                2.0e-3,
                1.0e-6,
                1.0e-11,
            );
        }
    }

    #[test]
    #[ignore = "periodic third-derivative gate: 5 third-derivative image sweeps per fixture"]
    fn band_pulay_third_matches_block_second_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 17);
            let analytic = f.band_pulay_at(0.0);
            assert!(
                analytic.second.abs() > 1.0e-12,
                "{name}: band/Pulay block is identically zero, gate is vacuous"
            );
            let second_at = |lam: f64| f.band_pulay_at(lam).second;
            assert_ladder(
                &format!("band_pulay/{name}"),
                &second_at,
                analytic.third,
                2.0e-3,
                1.0e-6,
                1.0e-9,
            );
        }
    }

    #[test]
    #[ignore = "periodic third-derivative gate: 5 real-space Ewald sweeps per fixture"]
    fn scc2_realspace_third_matches_block_second_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 19);
            let analytic = f.scc2_at(0.0);
            assert!(
                analytic.second.abs() > 1.0e-14,
                "{name}: SCC2 real-space block is identically zero, gate is vacuous"
            );
            let second_at = |lam: f64| f.scc2_at(lam).second;
            assert_ladder(
                &format!("scc2_real/{name}"),
                &second_at,
                analytic.third,
                2.0e-3,
                1.0e-6,
                1.0e-12,
            );
        }
    }

    /// The Ewald-entangled SCC2 sums, gated over the whole value → second → third chain: the
    /// analytic `second` must be the second central difference of the frozen energy piece, and
    /// the analytic `third` the first central difference of the analytic `second`. This is the
    /// only frozen block whose kernel carries the `α`-split, so both links are checked — a
    /// consistent-but-wrong pair (second and third both derived from a mistranscribed kernel)
    /// cannot pass the value link, because the value sums are transcribed from the production
    /// energy code path that `shell_potential_second_directional` is gated against bit-for-bit.
    #[test]
    #[ignore = "periodic third-derivative gate: 9 Ewald sweeps per fixture"]
    fn scc2_ewald_third_matches_value_and_second_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 23);
            let (_, second, third) = f.scc2_ewald_at(0.0);
            assert!(
                second.abs() > 1.0e-14,
                "{name}: SCC2 Ewald block is identically zero, gate is vacuous"
            );
            let h = 2.0e-3;
            // value -> second: second central difference (f(+h) - 2 f(0) + f(-h)) / h².
            let value_at = |lam: f64| f.scc2_ewald_at(lam).0;
            let second_fd = |hh: f64| {
                (value_at(hh) - 2.0 * value_at(0.0) + value_at(-hh)) / (hh * hh)
            };
            let (s1, s2) = (second_fd(h), second_fd(0.5 * h));
            let s_rich = (4.0 * s2 - s1) / 3.0;
            println!(
                "scc2_ewald_value/{name:<30} analytic={second:+.10e} \
                 richardson_err={:.3e} ratio={:.2}",
                (s_rich - second).abs(),
                ((s1 - second) / (s2 - second)).abs()
            );
            assert!(
                (s_rich - second).abs() <= 2.0e-3 * second.abs() + 1.0e-10,
                "{name}: Ewald SCC2 second vs value FD: {second:.10e} vs {s_rich:.10e}"
            );
            // second -> third: the standard ladder.
            let second_at = |lam: f64| f.scc2_ewald_at(lam).1;
            assert_ladder(
                &format!("scc2_ewald/{name}"),
                &second_at,
                third,
                2.0e-3,
                1.0e-6,
                1.0e-12,
            );
        }
    }

    /// The directional first-order response must be the `v`-contraction of the per-DOF
    /// responses (both paths are linear in the skeleton operators and share every reference
    /// ingredient, so this pins the transcription of the directional solve, not physics).
    #[test]
    #[ignore = "periodic response gate: one full per-DOF CPXTB sweep per fixture"]
    fn gamma_first_order_directional_matches_dof_contraction() {
        use crate::pbc::hessian::{gamma_cpxtb_density_responses, gamma_mos};
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 29);
            let mos = gamma_mos(&f.scf, f.scf.nelec).expect("gamma mos");
            let sk =
                gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &f.opts, &f.pbc)
                    .expect("skeleton");
            let (pd, wd) = gamma_cpxtb_density_responses(&f.scf, &sk, &mos).expect("per-DOF");
            let (p1, w1, _q1) =
                gamma_first_order_directional(&f.scf, &sk, &mos, &f.v).expect("directional");
            let n = f.scf.basis.len();
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let (mut ps, mut ws) = (0.0, 0.0);
                    for (y, &vy) in f.v.iter().enumerate() {
                        ps += vy * pd[y][(i, j)];
                        ws += vy * wd[y][(i, j)];
                    }
                    worst = worst.max((ps - p1[(i, j)]).abs()).max((ws - w1[(i, j)]).abs());
                }
            }
            println!("gamma_first_order/{name}: worst |directional - contraction| {worst:.3e}");
            assert!(
                worst < 1.0e-9,
                "{name}: directional first-order response drifts from the per-DOF path: \
                 {worst:.3e}"
            );
        }
    }

    /// Physical gate: `q¹` from the directional CPXTB solve must match the central FD of the
    /// **reconverged** SCC shell charges along `v`. This is the same ground truth the periodic
    /// finite-T shell-charge gate uses; integer occupations here.
    #[test]
    #[ignore = "periodic response gate: 4 reconverged SCCs per fixture"]
    fn gamma_first_order_charges_match_reconverged_fd() {
        use crate::pbc::hessian::gamma_mos;
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 31);
            let mut opts = f.opts.clone();
            opts.energy_tolerance = 1.0e-13;
            opts.charge_tolerance = 1.0e-12;
            let scf = run_pbc_scc(&f.system, &f.params, &opts, &f.pbc).expect("tight SCC");
            let mos = gamma_mos(&scf, scf.nelec).expect("gamma mos");
            let sk = gamma_skeleton_derivatives(&f.system, &f.params, &scf, &opts, &f.pbc)
                .expect("skeleton");
            let (_p1, _w1, q1) =
                gamma_first_order_directional(&scf, &sk, &mos, &f.v).expect("directional");
            let charges_at = |lam: f64| -> Vec<f64> {
                let sys = displaced(&f.system, &f.v, lam);
                run_pbc_scc(&sys, &f.params, &opts, &f.pbc)
                    .expect("displaced SCC")
                    .shell_charges
            };
            let h = 1.0e-4;
            let (cp, cm) = (charges_at(h), charges_at(-h));
            let mut worst = 0.0_f64;
            for s in 0..q1.len() {
                worst = worst.max(((cp[s] - cm[s]) / (2.0 * h) - q1[s]).abs());
            }
            println!("gamma_q1/{name}: worst |analytic - FD| {worst:.3e}");
            assert!(
                worst < 1.0e-7,
                "{name}: directional shell-charge response vs reconverged FD: {worst:.3e}"
            );
        }
    }

    /// The whole frozen bundle at once — now including the Ewald-entangled SCC2 sums — so a
    /// sign error that two components cancel between cannot hide. Deliberately **not** the
    /// complete periodic third derivative: the Gamma response is added by the caller.
    #[test]
    #[ignore = "periodic third-derivative gate: full frozen bundle, 5 sweeps per fixture"]
    fn frozen_bundle_third_matches_bundle_second_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 23);
            let bundle_at = |lam: f64| {
                let sys = f.at(lam);
                let sk =
                    gamma_skeleton_derivatives(&sys, &f.params, &f.scf, &f.opts, &f.pbc).unwrap();
                pbc_gamma_frozen_third_directional(
                    &sys,
                    &f.params,
                    &f.scf,
                    &sk,
                    &f.pbc,
                    f.opts.hamiltonian.coordination_cutoff,
                    f.opts.hamiltonian.enable_cn_hamiltonian,
                    true,
                    None,
                    &f.v,
                )
                .unwrap()
                .total()
            };
            let analytic = bundle_at(0.0);
            let second_at = |lam: f64| bundle_at(lam).second;
            assert_ladder(
                &format!("frozen_bundle/{name}"),
                &second_at,
                analytic.third,
                2.0e-3,
                1.0e-6,
                1.0e-9,
            );
        }
    }

    // -- Response-support components ----------------------------------------------------------

    /// `CN¹`/`CN²` from [`cn_directional_derivatives`] against a central difference of the
    /// coordination numbers themselves. No SCC needed, so this runs in the default suite.
    #[test]
    fn cn_directional_derivatives_match_fd() {
        for (name, xyz) in CLASSICAL_FIXTURES {
            let system = system_of(xyz);
            let cutoff = ElectronicOptions::default().hamiltonian.coordination_cutoff;
            let v = direction(3 * system.atoms.len(), 47);
            let (cn1, cn2) = cn_directional_derivatives(&system, cutoff, &v).unwrap();
            let cn_at = |lam: f64| -> Vec<f64> {
                coordination_with_derivatives(
                    &displaced(&system, &v, lam),
                    CoordinationOptions {
                        cutoff,
                        ..CoordinationOptions::default()
                    },
                )
                .unwrap()
                .cn
            };
            let cn1_at = |lam: f64| -> Vec<f64> {
                cn_directional_derivatives(&displaced(&system, &v, lam), cutoff, &v)
                    .unwrap()
                    .0
            };
            let h = 1.0e-3;
            let scale = cn1.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(scale > 1.0e-6, "{name}: CN¹ is zero, the gate would be vacuous");
            for (k, (&a1, &a2)) in cn1.iter().zip(&cn2).enumerate() {
                let f1 = |lam: f64| cn_at(lam)[k];
                let f2 = |lam: f64| cn1_at(lam)[k];
                assert_ladder(&format!("cn1/{name}/atom{k}"), &f1, a1, h, 1.0e-6, 1.0e-9);
                assert_ladder(&format!("cn2/{name}/atom{k}"), &f2, a2, h, 1.0e-6, 1.0e-9);
            }
        }
    }

    /// The refreshed frozen SCC must be the reference one at `λ = 0`: if the reconstruction of
    /// `self_energies` / `S(Γ)` / `shell_scc_potential` drifted from what the SCC itself built,
    /// the second-derivative gate below would be differencing a different function.
    #[test]
    #[ignore = "periodic response gate: one SCC plus one Bloch/gamma rebuild per fixture"]
    fn frozen_scf_refresh_is_identity_at_zero() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 53);
            let (_, scf) = f.refreshed_at(0.0);
            let mut worst_pot = 0.0_f64;
            for s in 0..scf.shell_scc_potential.len() {
                worst_pot = worst_pot
                    .max((scf.shell_scc_potential[s] - f.scf.shell_scc_potential[s]).abs());
            }
            let mut worst_se = 0.0_f64;
            for s in 0..scf.bloch.self_energies.len() {
                worst_se =
                    worst_se.max((scf.bloch.self_energies[s] - f.scf.bloch.self_energies[s]).abs());
            }
            let (_, s_new) = scf.bloch.h_s_gamma_real();
            let (_, s_ref) = f.scf.bloch.h_s_gamma_real();
            let mut worst_s = 0.0_f64;
            for i in 0..s_new.rows() {
                for j in 0..s_new.cols() {
                    worst_s = worst_s.max((s_new[(i, j)] - s_ref[(i, j)]).abs());
                }
            }
            println!(
                "refresh/{name}: worst |ΔV| {worst_pot:.3e} |Δse| {worst_se:.3e} |ΔS| {worst_s:.3e}"
            );
            assert!(
                worst_pot < 1.0e-12 && worst_se < 1.0e-14 && worst_s < 1.0e-14,
                "{name}: refreshed frozen SCC differs from the reference at λ = 0"
            );
        }
    }

    /// **Gate T1.** `(F^vv, S^vv)` must be the element-wise central difference of the production
    /// skeleton's `(F¹, S¹)` contracted with `v`, at frozen charges and frozen density.
    #[test]
    #[ignore = "periodic response gate: 4 refreshed skeleton evaluations per fixture"]
    fn gamma_directional_second_matrices_match_skeleton_fd() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 59);
            let sk = gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &f.opts, &f.pbc)
                .expect("skeleton");
            let v1 = shell_potential_first_directional(&sk, &f.v);
            let v2 = f.v2_at(0.0);
            let (fock2, overlap2) = gamma_directional_second_matrices(
                &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &v1, &v2, &f.v,
            )
            .expect("second-order skeleton matrices");
            let h = 2.0e-3;
            assert_matrix_ladder(
                &format!("S^vv/{name}"),
                &|lam| f.skeleton_directional_at(lam).1,
                &overlap2,
                h,
                1.0e-6,
            );
            assert_matrix_ladder(
                &format!("F^vv/{name}"),
                &|lam| f.skeleton_directional_at(lam).0,
                &fock2,
                h,
                1.0e-6,
            );
        }
    }

    /// **Gate T2.** The charge-path bilinear must reduce to the frozen quadratic SCC2 second
    /// derivative at `(q, q)`, be symmetric under swapping its two charge vectors, be linear in
    /// each of them, and close its own value → first → second ladder.
    #[test]
    #[ignore = "periodic response gate: several Ewald sweeps per fixture"]
    fn scc2_bilinear_reduces_to_quadratic_and_is_bilinear() {
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 61);
            let lattice = f.lattice();
            let q = f.scf.shell_charges.clone();
            let second = |qa: &[f64], qb: &[f64]| {
                pbc_scc2_bilinear_second_directional(
                    &f.system, &lattice, &f.scf, &f.pbc, qa, qb, &f.v,
                )
            };
            let first = |qa: &[f64], qb: &[f64]| {
                pbc_scc2_bilinear_first_directional(
                    &f.system, &lattice, &f.scf, &f.pbc, qa, qb, &f.v,
                )
            };

            // (q, q) == the quadratic frozen block.
            let quad = f.scc2_at(0.0).second + f.scc2_ewald_at(0.0).1;
            let bil = second(&q, &q);
            let rel = (bil - quad).abs() / quad.abs().max(1.0e-30);
            println!("scc2_bilinear/{name}: quadratic {quad:+.12e} bilinear {bil:+.12e} rel {rel:.3e}");
            // Not bit-identical, and cannot be: the two reciprocal sums fold their charge weight
            // into the `G`-dependent prefactor in a different multiply ORDER
            // (`0.5·w_G·q_i·q_j` versus `0.25·w_G·(qa_i qb_j + qb_i qa_j)`), so each term differs
            // by an ulp while the sums themselves cancel heavily. Measured: 2.2e-15 on the
            // homonuclear cell, 4.0e-14 on charge-transferring BN. The failure this gate exists
            // to catch — a wrong bilinear normalisation — is a factor of 2 or 4, not an ulp.
            assert!(
                rel <= 1.0e-12,
                "{name}: bilinear(q, q) = {bil:.12e} differs from the quadratic {quad:.12e}"
            );

            // A second, independent charge vector.
            let qb: Vec<f64> = q
                .iter()
                .enumerate()
                .map(|(i, x)| 0.37 * x + 0.11 * ((i % 3) as f64 - 1.0))
                .collect();
            let qc: Vec<f64> = q.iter().map(|x| 1.0 - 0.8 * x).collect();
            for (label, ab, ba) in [
                ("second", second(&q, &qb), second(&qb, &q)),
                ("first", first(&q, &qb), first(&qb, &q)),
            ] {
                let rel = (ab - ba).abs() / ab.abs().max(1.0e-30);
                assert!(
                    rel <= 1.0e-14,
                    "{name}/{label}: bilinear is not symmetric ({ab:.12e} vs {ba:.12e})"
                );
            }
            let mixed: Vec<f64> = q
                .iter()
                .zip(&qc)
                .map(|(a, b)| 2.0 * a + 3.0 * b)
                .collect();
            for (label, lhs, rhs) in [
                (
                    "second",
                    second(&mixed, &qb),
                    2.0 * second(&q, &qb) + 3.0 * second(&qc, &qb),
                ),
                (
                    "first",
                    first(&mixed, &qb),
                    2.0 * first(&q, &qb) + 3.0 * first(&qc, &qb),
                ),
            ] {
                let rel = (lhs - rhs).abs() / lhs.abs().max(1.0e-30);
                assert!(
                    rel <= 1.0e-13,
                    "{name}/{label}: bilinear is not linear in its first slot \
                     ({lhs:.12e} vs {rhs:.12e})"
                );
            }

            // first -> second ladder on a genuinely mixed pair.
            let first_at = |lam: f64| {
                pbc_scc2_bilinear_first_directional(
                    &f.at(lam),
                    &lattice,
                    &f.scf,
                    &f.pbc,
                    &q,
                    &qb,
                    &f.v,
                )
            };
            assert_ladder(
                &format!("scc2_bilinear_ladder/{name}"),
                &first_at,
                second(&q, &qb),
                2.0e-3,
                1.0e-6,
                1.0e-12,
            );
        }
    }

    /// **Gate T3** — the load-bearing one. `B6` must be the central difference of the
    /// **production** periodic response gradient
    /// ([`crate::pbc::hessian::response_gradient`]) contracted with `v`, evaluated at displaced
    /// geometries with the response slot `X¹`, the kernel and the whole reference SCC frozen.
    /// The band pairs and the coordination derivatives are rebuilt at each displaced geometry
    /// because they are the geometry-carrying inputs of that column.
    #[test]
    #[ignore = "periodic response gate: 4 response-gradient sweeps plus one CPXTB solve per fixture"]
    fn response_path_b6_matches_response_gradient_fd() {
        use crate::pbc::hessian::{
            build_response_band_pairs, gamma_mos, periodic_response_kernel, response_gradient,
            DensityLookup,
        };
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 67);
            let mos = gamma_mos(&f.scf, f.scf.nelec).expect("gamma mos");
            let sk = gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &f.opts, &f.pbc)
                .expect("skeleton");
            let (p1, w1, q1) =
                gamma_first_order_directional(&f.scf, &sk, &mos, &f.v).expect("first order");
            let dens1 = uniform_density_images(&f.lattice(), f.pbc.ao_cutoff, &p1, &w1);
            let v1 = shell_potential_first_directional(&sk, &f.v);
            let path = gamma_response_path_directional(
                &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.dens, &dens1, &q1, &v1, &f.v,
            )
            .expect("response path");
            let kernel = periodic_response_kernel(&f.scf);
            let cutoff = f.opts.hamiltonian.coordination_cutoff;
            let enable_cn = f.opts.hamiltonian.enable_cn_hamiltonian;
            let g_at = |lam: f64| -> f64 {
                let sys = f.at(lam);
                let band_pairs =
                    build_response_band_pairs(&sys, &f.params, &f.scf, &f.dens.p, &f.pbc)
                        .expect("band pairs");
                let cn = if enable_cn {
                    Some(
                        coordination_with_derivatives(
                            &sys,
                            CoordinationOptions {
                                cutoff,
                                ..CoordinationOptions::default()
                            },
                        )
                        .expect("cn"),
                    )
                } else {
                    None
                };
                let grad = response_gradient(
                    &sys,
                    &f.params,
                    &f.scf,
                    &band_pairs,
                    DensityLookup::Uniform(&p1),
                    DensityLookup::Uniform(&w1),
                    &q1,
                    &kernel,
                    &f.pbc,
                    cn.as_ref(),
                )
                .expect("response gradient");
                grad.iter()
                    .enumerate()
                    .map(|(a, g)| g.dot(atom_direction(&f.v, a)))
                    .sum()
            };
            println!(
                "b6/{name}: cn={:+.6e} band_pulay={:+.6e} scc2={:+.6e} screening={:+.6e}",
                path.b6_blocks[0], path.b6_blocks[1], path.b6_blocks[2], path.b6_blocks[3]
            );
            assert!(
                path.b6_blocks.iter().all(|x| x.is_finite()),
                "{name}: non-finite B6 block"
            );
            assert_ladder(&format!("b6/{name}"), &g_at, path.b6, 2.0e-3, 1.0e-6, 1.0e-9);
        }
    }

    /// Structural gate on `bg4`: each background family is a product of a density slot and one or
    /// two charge slots, so scaling those slots must scale the family by the corresponding power.
    /// This cannot see a wrong *sign*, which is deliberate — the four families only add up to a
    /// finite-differenceable quantity inside the caller's assembly, so their physical gate lives
    /// there. What it does catch is a mis-slotted argument (`P₀` where `P¹` belongs, `q¹` fed to
    /// the wrong leg of the chain), which is the realistic failure mode.
    ///
    /// Homonuclear diamond exercises only `scc_dp_pot`: with no charge transfer its `q¹` is at
    /// the `1e-17` level, so the three `q¹`-driven families are numerically zero there (the
    /// scaling law still holds bit-exactly — every leg is scaled by an exact power of two — but
    /// the check would be vacuous). Coverage is therefore accumulated **across** fixtures and
    /// asserted at the end, which is what forces the heteronuclear cell to be in the list.
    #[test]
    #[ignore = "periodic response gate: 3 response-path evaluations per fixture"]
    fn response_path_bg4_scaling_is_multilinear() {
        use crate::pbc::hessian::gamma_mos;
        let mut alive = [false; 4];
        for (name, xyz) in SCF_FIXTURES {
            let f = Frozen::build(xyz, 71);
            let mos = gamma_mos(&f.scf, f.scf.nelec).expect("gamma mos");
            let sk = gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &f.opts, &f.pbc)
                .expect("skeleton");
            let (p1, w1, q1) =
                gamma_first_order_directional(&f.scf, &sk, &mos, &f.v).expect("first order");
            let v1 = shell_potential_first_directional(&sk, &f.v);
            let scaled = |m: &Matrix, s: f64| {
                let mut out = m.clone();
                for i in 0..out.rows() {
                    for j in 0..out.cols() {
                        out[(i, j)] *= s;
                    }
                }
                out
            };
            let run = |p: &Matrix, w: &Matrix, q: &[f64]| {
                let dens = uniform_density_images(&f.lattice(), f.pbc.ao_cutoff, p, w);
                gamma_response_path_directional(
                    &f.system, &f.params, &f.scf, &f.opts, &f.pbc, &f.dens, &dens, q, &v1, &f.v,
                )
                .expect("response path")
                .bg4_families
            };
            let base = run(&p1, &w1, &q1);
            for (k, b) in base.iter().enumerate() {
                if b.abs() > 1.0e-10 {
                    alive[k] = true;
                }
            }
            let check = |label: &str, got: [f64; 4], want: [f64; 4]| {
                for k in 0..4 {
                    if base[k] == 0.0 {
                        assert!(got[k] == 0.0, "{name}/{label}: bg4 family {k} grew from zero");
                        continue;
                    }
                    let rel = (got[k] - want[k] * base[k]).abs() / (want[k] * base[k]).abs();
                    assert!(
                        rel <= 1.0e-12,
                        "{name}/{label}: bg4 family {k} scales as {:.6}, expected {:.1}",
                        got[k] / base[k],
                        want[k]
                    );
                }
            };
            // q¹ -> 2 q¹: [dp_pot, scc_p0, scc_chain, kernel_qq] scale by [1, 2, 4, 4].
            let q_scaled: Vec<f64> = q1.iter().map(|x| 2.0 * x).collect();
            check("q1x2", run(&p1, &w1, &q_scaled), [1.0, 2.0, 4.0, 4.0]);
            // P¹ -> 3 P¹: [3, 3, 1, 1].
            check(
                "P1x3",
                run(&scaled(&p1, 3.0), &scaled(&w1, 3.0), &q1),
                [3.0, 3.0, 1.0, 1.0],
            );
            println!("bg4/{name}: families {base:?}");
        }
        assert!(
            alive.iter().all(|&x| x),
            "no fixture in SCF_FIXTURES exercised every bg4 family: {alive:?}"
        );
    }

    /// Acoustic sum rule. Under a rigid translation every pair displacement `v_a − v_b` vanishes,
    /// so each frozen block's directional second **and** third must be zero. For the electronic
    /// Cache-channel decomposition of the band/Pulay block's TRUE `X₀` motion (the ~1e-7
    /// density-path tail): FD the block's `.second` at reconverged `scf(λ)` with each cache
    /// selectively pinned to its reference value — the difference of neighbouring variants is
    /// that channel's true motion, to be matched against the assembly attribution
    /// (`bp(X¹; v1, v2)` + ΔV-cache).
    struct ProbeRef {
        system: PeriodicSystem,
        params: Gfn1Parameters,
        scf: crate::pbc::scf::PbcSccResult,
        v: Vec<f64>,
    }

    #[test]
    #[ignore = "diagnostic"]
    fn band_pulay_cache_channel_probe() {
        use crate::pbc::scf::run_pbc_scc;
        for (name, xyz) in SCF_FIXTURES {
            // Same direction formula and options as `pbc::gamma_response`'s
            // gates (seed 41), so the channel values line up with the total
            // gate's measured +1.06e-7 / +1.4e-8 band/Pulay tail.
            let system = system_of(xyz);
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let ndof = 3 * system.atoms.len();
            let seed = 41u64;
            let dir: Vec<f64> = (0..ndof)
                .map(|k| {
                    let x = ((k as u64 + 1) * (seed + 7)) % 13;
                    0.31 - 0.05 * (x as f64) + 0.01 * ((k % 3) as f64)
                })
                .collect();
            let scf0 = crate::pbc::scf::run_pbc_scc(&system, &params, &opts, &pbc).unwrap();
            let lattice0 = *system.lattice.as_ref().unwrap();
            let f = ProbeRef {
                system: system.clone(),
                params: params.clone(),
                scf: scf0,
                v: dir,
            };
            let se0 = f.scf.bloch.self_energies.clone();
            let v0_pot = f.scf.shell_scc_potential.clone();
            let dens_ref = gamma_realspace_densities(&f.scf, &lattice0, pbc.ao_cutoff);
            let v1_ref = {
                let sk = gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &opts, &pbc)
                    .unwrap();
                shell_potential_first_directional(&sk, &f.v)
            };
            let v2_ref =
                shell_potential_second_directional(&f.system, &lattice0, &f.scf, &pbc, &f.v);
            // Variant evaluator: which caches move with λ.
            let bp_at = |lam: f64,
                         move_se: bool,
                         move_vpot: bool,
                         move_dens: bool,
                         move_legs: bool|
             -> f64 {
                let sys = displaced(&f.system, &f.v, lam);
                let lattice = *sys.lattice.as_ref().unwrap();
                let mut scf = run_pbc_scc(&sys, &f.params, &opts, &pbc).unwrap();
                if !move_se {
                    scf.bloch.self_energies = se0.clone();
                }
                if !move_vpot {
                    scf.shell_scc_potential = v0_pot.clone();
                }
                let dens = if move_dens {
                    gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff)
                } else {
                    FrozenDensityImages {
                        p: dens_ref.p.clone(),
                        w: dens_ref.w.clone(),
                    }
                };
                let (v1, v2) = if move_legs {
                    let sk =
                        gamma_skeleton_derivatives(&sys, &f.params, &scf, &opts, &pbc).unwrap();
                    (
                        shell_potential_first_directional(&sk, &f.v),
                        shell_potential_second_directional(&sys, &lattice, &scf, &pbc, &f.v),
                    )
                } else {
                    (v1_ref.clone(), v2_ref.clone())
                };
                pbc_band_pulay_third_directional(
                    &sys, &f.params, &scf, &pbc, &dens, &v1, &v2, &f.v,
                )
                .unwrap()
                .second
            };
            let h = 1.0e-3;
            let fd = |se: bool, vp: bool, de: bool, le: bool| -> f64 {
                (bp_at(h, se, vp, de, le) - bp_at(-h, se, vp, de, le)) / (2.0 * h)
            };
            let full = fd(true, true, true, true);
            let base = fd(false, false, false, false); // geometry only = .third
            let ch_se = fd(true, false, false, false) - base;
            let ch_vp = fd(false, true, false, false) - base;
            let ch_de = fd(false, false, true, false) - base;
            let ch_le = fd(false, false, false, true) - base;
            println!(
                "bp_cache/{name}: full {full:+.10e}  geo {base:+.10e}\n  se {ch_se:+.10e}  \
                 vpot {ch_vp:+.10e}  dens {ch_de:+.10e}  legs {ch_le:+.10e}  sum(geo+ch) \
                 {:+.10e}",
                base + ch_se + ch_vp + ch_de + ch_le
            );
            assert!(full.is_finite());
        }
    }

    /// blocks this holds term by term (every factor carries a pair difference); for the classical
    /// blocks it is a genuine numerical check on the production tensors.
    #[test]
    #[ignore = "periodic third-derivative gate: one bundle evaluation per fixture"]
    fn frozen_third_vanishes_for_rigid_translation() {
        for (name, xyz) in SCF_FIXTURES {
            let mut f = Frozen::build(xyz, 29);
            f.v = rigid_translation(3 * f.system.atoms.len());
            let sk =
                gamma_skeleton_derivatives(&f.system, &f.params, &f.scf, &f.opts, &f.pbc).unwrap();
            let parts = pbc_gamma_frozen_third_directional(
                &f.system,
                &f.params,
                &f.scf,
                &sk,
                &f.pbc,
                f.opts.hamiltonian.coordination_cutoff,
                f.opts.hamiltonian.enable_cn_hamiltonian,
                true,
                None,
                &f.v,
            )
            .unwrap();
            for (label, d) in [
                ("repulsion", parts.repulsion),
                ("halogen", parts.halogen),
                ("dispersion", parts.dispersion),
                ("coordination", parts.coordination),
                ("band_pulay", parts.band_pulay),
                ("scc2_realspace", parts.scc2_realspace),
            ] {
                println!(
                    "ASR {name}/{label:<15} second={:+.3e} third={:+.3e}",
                    d.second, d.third
                );
                assert!(
                    d.second.abs() <= 1.0e-9 && d.third.abs() <= 1.0e-9,
                    "{name}/{label}: rigid translation gives second={:.3e}, third={:.3e}",
                    d.second,
                    d.third
                );
            }
        }
    }
}
