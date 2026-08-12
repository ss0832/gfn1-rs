// SPDX-License-Identifier: GPL-3.0-or-later
#![allow(dead_code)]
//! Fourth Cartesian derivatives of the contracted AO overlap, on top of the full
//! third-derivative moment set. This is the integral half of the frozen (geometry-only)
//! **fourth nuclear derivative**; it extends [`super::third_derivatives`] by exactly one
//! order, reusing the same 1D moment recurrence ([`super::moment_1d_table`]) and the same
//! elementary Gaussian center-derivative operators
//!
//! ```text
//!   (∂/∂A_x) M(i,j) = 2α M(i+1,j) − i M(i−1,j)        (bra)
//!   (∂/∂B_x) M(i,j) = 2β M(i,j+1) − j M(i,j−1)        (ket)
//! ```
//!
//! Because mixed partials commute, the fourth order needs **no new math**: it is one more
//! application of the very same per-axis operators to the third-order ladder, contracted by
//! the same multiplicity-counting [`product`] helper. The five distinct center patterns of a
//! fourth derivative are stored as `q_bbbb`/`q_bbbk`/`q_bbkk`/`q_bkkk`/`q_kkkk`, each a
//! fully-index-symmetric-within-center `3×3×3×3` Cartesian tensor with the **bra indices
//! first and the ket indices last**, matching the third-order convention
//! (`q_bbkk[a][b][c][d] = ∂_{A_a}∂_{A_b}∂_{B_c}∂_{B_d}`).
//!
//! # Why only the overlap at fourth order
//!
//! GFN1's frozen fourth-order terms differentiate the overlap-derived quantities only (the
//! dipole/quadrupole moment blocks enter the energy at most through lower orders in the
//! frozen expansion), so carrying all `NMOMENT = 10` moments at fourth order would be pure
//! waste: five `3×3×3×3` tensors × 10 moments × 8 bytes = 32.4 kB per contracted AO pair,
//! versus 3.2 kB for the overlap alone. The fourth-order fields therefore correspond to
//! moment index `[0]` (the overlap) and to that index only, while everything up to third
//! order keeps the full 10-moment set.
//!
//! The API is deliberately shaped so a 10-moment variant can be added later without
//! disturbing callers: the per-moment work lives in [`primitive_overlap_fourth_derivatives`]
//! guarded by a single `k == OVERLAP` test, so turning `q_*: Ten4` into
//! `q_*: [Ten4; NMOMENT]` is a mechanical change (drop the guard, index by `k`) that leaves
//! the axis ladder, [`axis_fourth_derivatives`] and [`product`] untouched.

use crate::basis::{AOBasisFunction, CartesianPower};
use crate::math::Vec3;

use super::third_derivatives::ContractedPairThirdDerivatives;
use super::{moment_1d_get, moment_1d_table};

/// Number of AO moment components carried up to third order (overlap + 3 dipole +
/// 6 quadrupole), matching [`super::third_derivatives`].
pub(crate) const NMOMENT: usize = 10;

/// Index of the overlap inside the moment set; the only moment carried at fourth order.
pub(crate) const OVERLAP: usize = 0;

type Mat3 = [[f64; 3]; 3];
type Ten3 = [[[f64; 3]; 3]; 3];
/// Rank-4 Cartesian tensor, bra indices first and ket indices last.
pub(crate) type Ten4 = [[[[f64; 3]; 3]; 3]; 3];

/// Everything [`ContractedPairThirdDerivatives`] carries (replicated field-by-field, the way
/// the third-order struct replicates the second-order one) plus the five fourth-order centre
/// patterns of the **overlap**.
#[derive(Clone, Debug)]
pub struct ContractedPairFourthDerivatives {
    pub moments: [f64; NMOMENT],
    pub d_bra: [Vec3; NMOMENT],
    pub d_ket: [Vec3; NMOMENT],
    pub h_bra_bra: [Mat3; NMOMENT],
    pub h_bra_ket: [Mat3; NMOMENT],
    pub h_ket_ket: [Mat3; NMOMENT],
    /// `∂_{A_a}∂_{A_b}∂_{A_c}`.
    pub t_bra_bra_bra: [Ten3; NMOMENT],
    /// `∂_{A_a}∂_{A_b}∂_{B_c}` (ket index last).
    pub t_bra_bra_ket: [Ten3; NMOMENT],
    /// `∂_{A_a}∂_{B_b}∂_{B_c}` (ket indices last).
    pub t_bra_ket_ket: [Ten3; NMOMENT],
    /// `∂_{B_a}∂_{B_b}∂_{B_c}`.
    pub t_ket_ket_ket: [Ten3; NMOMENT],
    /// `∂_{A_a}∂_{A_b}∂_{A_c}∂_{A_d}` of the overlap (`moments[0]`).
    pub q_bbbb: Ten4,
    /// `∂_{A_a}∂_{A_b}∂_{A_c}∂_{B_d}` of the overlap (ket index last).
    pub q_bbbk: Ten4,
    /// `∂_{A_a}∂_{A_b}∂_{B_c}∂_{B_d}` of the overlap (ket indices last).
    pub q_bbkk: Ten4,
    /// `∂_{A_a}∂_{B_b}∂_{B_c}∂_{B_d}` of the overlap (ket indices last).
    pub q_bkkk: Ten4,
    /// `∂_{B_a}∂_{B_b}∂_{B_c}∂_{B_d}` of the overlap.
    pub q_kkkk: Ten4,
}

impl ContractedPairFourthDerivatives {
    fn zero() -> Self {
        Self {
            moments: [0.0; NMOMENT],
            d_bra: [Vec3::zero(); NMOMENT],
            d_ket: [Vec3::zero(); NMOMENT],
            h_bra_bra: [[[0.0; 3]; 3]; NMOMENT],
            h_bra_ket: [[[0.0; 3]; 3]; NMOMENT],
            h_ket_ket: [[[0.0; 3]; 3]; NMOMENT],
            t_bra_bra_bra: [[[[0.0; 3]; 3]; 3]; NMOMENT],
            t_bra_bra_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
            t_bra_ket_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
            t_ket_ket_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
            q_bbbb: [[[[0.0; 3]; 3]; 3]; 3],
            q_bbbk: [[[[0.0; 3]; 3]; 3]; 3],
            q_bbkk: [[[[0.0; 3]; 3]; 3]; 3],
            q_bkkk: [[[[0.0; 3]; 3]; 3]; 3],
            q_kkkk: [[[[0.0; 3]; 3]; 3]; 3],
        }
    }

    /// The embedded lower-order data, repackaged as the third-order struct so callers (and
    /// the consistency test) can hand it to code written against
    /// [`ContractedPairThirdDerivatives`].
    pub(crate) fn third_derivatives(&self) -> ContractedPairThirdDerivatives {
        ContractedPairThirdDerivatives {
            moments: self.moments,
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

pub(crate) fn contracted_pair_with_fourth_derivatives(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> ContractedPairFourthDerivatives {
    let mut out = ContractedPairFourthDerivatives::zero();
    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            let r2 = (ca - cb).norm2();
            let kab_3d = (-alpha * beta / p * r2).exp();
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient;
                    let prim = primitive_moments_fourth_derivatives_up_to_quadrupole(
                        ca_term.power,
                        cb_term.power,
                        alpha,
                        beta,
                        ca,
                        cb,
                    );
                    let s = cfac * kab_3d;
                    for k in 0..NMOMENT {
                        out.moments[k] += s * prim.moments[k];
                        out.d_bra[k] += prim.d_bra[k] * s;
                        out.d_ket[k] += prim.d_ket[k] * s;
                        for i in 0..3 {
                            for j in 0..3 {
                                out.h_bra_bra[k][i][j] += s * prim.h_bra_bra[k][i][j];
                                out.h_bra_ket[k][i][j] += s * prim.h_bra_ket[k][i][j];
                                out.h_ket_ket[k][i][j] += s * prim.h_ket_ket[k][i][j];
                                for l in 0..3 {
                                    out.t_bra_bra_bra[k][i][j][l] +=
                                        s * prim.t_bra_bra_bra[k][i][j][l];
                                    out.t_bra_bra_ket[k][i][j][l] +=
                                        s * prim.t_bra_bra_ket[k][i][j][l];
                                    out.t_bra_ket_ket[k][i][j][l] +=
                                        s * prim.t_bra_ket_ket[k][i][j][l];
                                    out.t_ket_ket_ket[k][i][j][l] +=
                                        s * prim.t_ket_ket_ket[k][i][j][l];
                                }
                            }
                        }
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            for l in 0..3 {
                                for n in 0..3 {
                                    out.q_bbbb[i][j][l][n] += s * prim.q_bbbb[i][j][l][n];
                                    out.q_bbbk[i][j][l][n] += s * prim.q_bbbk[i][j][l][n];
                                    out.q_bbkk[i][j][l][n] += s * prim.q_bbkk[i][j][l][n];
                                    out.q_bkkk[i][j][l][n] += s * prim.q_bkkk[i][j][l][n];
                                    out.q_kkkk[i][j][l][n] += s * prim.q_kkkk[i][j][l][n];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub(crate) struct PrimitiveMomentFourthDerivatives {
    pub moments: [f64; NMOMENT],
    pub d_bra: [Vec3; NMOMENT],
    pub d_ket: [Vec3; NMOMENT],
    pub h_bra_bra: [Mat3; NMOMENT],
    pub h_bra_ket: [Mat3; NMOMENT],
    pub h_ket_ket: [Mat3; NMOMENT],
    pub t_bra_bra_bra: [Ten3; NMOMENT],
    pub t_bra_bra_ket: [Ten3; NMOMENT],
    pub t_bra_ket_ket: [Ten3; NMOMENT],
    pub t_ket_ket_ket: [Ten3; NMOMENT],
    pub q_bbbb: Ten4,
    pub q_bbbk: Ten4,
    pub q_bbkk: Ten4,
    pub q_bkkk: Ten4,
    pub q_kkkk: Ten4,
}

/// Per-axis 1D moment value and its bra/ket derivative ladder up to **fourth** order. Built
/// by composing the elementary operators [`op_bra`]/[`op_ket`]; mixed entries use the
/// commuted canonical form (`q_bbkk = ∂_ket ∂_ket ∂_bra ∂_bra`, etc.).
#[derive(Clone, Copy, Debug, Default)]
struct AxisFourthDerivatives {
    value: f64,
    d_bra: f64,
    d_ket: f64,
    h_bra_bra: f64,
    h_bra_ket: f64,
    h_ket_ket: f64,
    t_bra_bra_bra: f64,
    t_bra_bra_ket: f64,
    t_bra_ket_ket: f64,
    t_ket_ket_ket: f64,
    q_bbbb: f64,
    q_bbbk: f64,
    q_bbkk: f64,
    q_bkkk: f64,
    q_kkkk: f64,
}

#[derive(Clone, Copy, Debug)]
enum Center {
    Bra,
    Ket,
}

#[inline]
fn op_bra(g: &dyn Fn(isize, isize) -> f64, alpha: f64, i: isize, j: isize) -> f64 {
    2.0 * alpha * g(i + 1, j) - (i as f64) * g(i - 1, j)
}

#[inline]
fn op_ket(g: &dyn Fn(isize, isize) -> f64, beta: f64, i: isize, j: isize) -> f64 {
    2.0 * beta * g(i, j + 1) - (j as f64) * g(i, j - 1)
}

fn axis_fourth_derivatives(
    table: &[f64],
    max_j: usize,
    i: usize,
    j: usize,
    alpha: f64,
    beta: f64,
) -> AxisFourthDerivatives {
    let m = |ii: isize, jj: isize| -> f64 {
        if ii < 0 || jj < 0 {
            0.0
        } else {
            moment_1d_get(table, max_j, ii as usize, jj as usize)
        }
    };
    // First/second/third-order ladders as closures so the next operator can compose them.
    // Identical to `third_derivatives::axis_third_derivatives`, with one more layer on top.
    let d_b = |ii: isize, jj: isize| op_bra(&m, alpha, ii, jj);
    let d_k = |ii: isize, jj: isize| op_ket(&m, beta, ii, jj);
    let d_bb = |ii: isize, jj: isize| op_bra(&d_b, alpha, ii, jj);
    let d_bk = |ii: isize, jj: isize| op_ket(&d_b, beta, ii, jj);
    let d_kk = |ii: isize, jj: isize| op_ket(&d_k, beta, ii, jj);
    let t_bbb = |ii: isize, jj: isize| op_bra(&d_bb, alpha, ii, jj);
    let t_bbk = |ii: isize, jj: isize| op_ket(&d_bb, beta, ii, jj);
    let t_bkk = |ii: isize, jj: isize| op_bra(&d_kk, alpha, ii, jj);
    let t_kkk = |ii: isize, jj: isize| op_ket(&d_kk, beta, ii, jj);
    let i = i as isize;
    let j = j as isize;
    AxisFourthDerivatives {
        value: m(i, j),
        d_bra: d_b(i, j),
        d_ket: d_k(i, j),
        h_bra_bra: d_bb(i, j),
        h_bra_ket: d_bk(i, j),
        h_ket_ket: d_kk(i, j),
        t_bra_bra_bra: t_bbb(i, j),
        t_bra_bra_ket: t_bbk(i, j),
        t_bra_ket_ket: t_bkk(i, j),
        t_ket_ket_ket: t_kkk(i, j),
        q_bbbb: op_bra(&t_bbb, alpha, i, j),
        q_bbbk: op_ket(&t_bbb, beta, i, j),
        q_bbkk: op_ket(&t_bbk, beta, i, j),
        q_bkkk: op_bra(&t_kkk, alpha, i, j),
        q_kkkk: op_ket(&t_kkk, beta, i, j),
    }
}

/// The 1D factor for `bra` bra-derivatives and `ket` ket-derivatives on one Cartesian axis
/// (their order is irrelevant — mixed partials commute, so only the counts matter).
#[inline]
fn axis_factor(a: &AxisFourthDerivatives, bra: usize, ket: usize) -> f64 {
    match (bra, ket) {
        (0, 0) => a.value,
        (1, 0) => a.d_bra,
        (0, 1) => a.d_ket,
        (2, 0) => a.h_bra_bra,
        (1, 1) => a.h_bra_ket,
        (0, 2) => a.h_ket_ket,
        (3, 0) => a.t_bra_bra_bra,
        (2, 1) => a.t_bra_bra_ket,
        (1, 2) => a.t_bra_ket_ket,
        (0, 3) => a.t_ket_ket_ket,
        (4, 0) => a.q_bbbb,
        (3, 1) => a.q_bbbk,
        (2, 2) => a.q_bbkk,
        (1, 3) => a.q_bkkk,
        (0, 4) => a.q_kkkk,
        _ => unreachable!("fourth derivative needs at most 4 selections per axis"),
    }
}

/// The 3D moment derivative selected by up to four `(center, axis)` selections: a product of
/// the three per-axis factors, each at the multiplicity implied by the selections that land
/// on that axis.
fn product(axes: &[AxisFourthDerivatives; 3], sels: &[(Center, usize)]) -> f64 {
    let mut value = 1.0;
    for (ax, axis) in axes.iter().enumerate() {
        let mut bra = 0;
        let mut ket = 0;
        for &(c, a) in sels {
            if a == ax {
                match c {
                    Center::Bra => bra += 1,
                    Center::Ket => ket += 1,
                }
            }
        }
        value *= axis_factor(axis, bra, ket);
    }
    value
}

/// Fill the five fourth-order centre patterns from an axis ladder. Split out so that a
/// future 10-moment variant only has to call it per moment index instead of once for the
/// overlap (see the module docs).
fn primitive_overlap_fourth_derivatives(
    axes: &[AxisFourthDerivatives; 3],
    out: &mut PrimitiveMomentFourthDerivatives,
) {
    for a0 in 0..3 {
        for b0 in 0..3 {
            for c0 in 0..3 {
                for d0 in 0..3 {
                    out.q_bbbb[a0][b0][c0][d0] = product(
                        axes,
                        &[
                            (Center::Bra, a0),
                            (Center::Bra, b0),
                            (Center::Bra, c0),
                            (Center::Bra, d0),
                        ],
                    );
                    out.q_bbbk[a0][b0][c0][d0] = product(
                        axes,
                        &[
                            (Center::Bra, a0),
                            (Center::Bra, b0),
                            (Center::Bra, c0),
                            (Center::Ket, d0),
                        ],
                    );
                    out.q_bbkk[a0][b0][c0][d0] = product(
                        axes,
                        &[
                            (Center::Bra, a0),
                            (Center::Bra, b0),
                            (Center::Ket, c0),
                            (Center::Ket, d0),
                        ],
                    );
                    out.q_bkkk[a0][b0][c0][d0] = product(
                        axes,
                        &[
                            (Center::Bra, a0),
                            (Center::Ket, b0),
                            (Center::Ket, c0),
                            (Center::Ket, d0),
                        ],
                    );
                    out.q_kkkk[a0][b0][c0][d0] = product(
                        axes,
                        &[
                            (Center::Ket, a0),
                            (Center::Ket, b0),
                            (Center::Ket, c0),
                            (Center::Ket, d0),
                        ],
                    );
                }
            }
        }
    }
}

fn primitive_moments_fourth_derivatives_up_to_quadrupole(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
) -> PrimitiveMomentFourthDerivatives {
    // bra power raised by ≤4 (fourth derivative); ket power raised by the moment operator
    // (≤2, quadrupole) plus the fourth derivative (≤4) ⇒ +6.
    let max_ix = pa.x + 4;
    let max_iy = pa.y + 4;
    let max_iz = pa.z + 4;
    let max_jx = pb.x + 6;
    let max_jy = pb.y + 6;
    let max_jz = pb.z + 6;
    let mx = moment_1d_table(max_ix, max_jx, alpha, beta, ca.x, cb.x);
    let my = moment_1d_table(max_iy, max_jy, alpha, beta, ca.y, cb.y);
    let mz = moment_1d_table(max_iz, max_jz, alpha, beta, ca.z, cb.z);
    // Moment operator: the (r - C) factors raise the ket Cartesian power per axis.
    let extras: [[usize; 3]; NMOMENT] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
    ];

    let mut out = PrimitiveMomentFourthDerivatives {
        moments: [0.0; NMOMENT],
        d_bra: [Vec3::zero(); NMOMENT],
        d_ket: [Vec3::zero(); NMOMENT],
        h_bra_bra: [[[0.0; 3]; 3]; NMOMENT],
        h_bra_ket: [[[0.0; 3]; 3]; NMOMENT],
        h_ket_ket: [[[0.0; 3]; 3]; NMOMENT],
        t_bra_bra_bra: [[[[0.0; 3]; 3]; 3]; NMOMENT],
        t_bra_bra_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
        t_bra_ket_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
        t_ket_ket_ket: [[[[0.0; 3]; 3]; 3]; NMOMENT],
        q_bbbb: [[[[0.0; 3]; 3]; 3]; 3],
        q_bbbk: [[[[0.0; 3]; 3]; 3]; 3],
        q_bbkk: [[[[0.0; 3]; 3]; 3]; 3],
        q_bkkk: [[[[0.0; 3]; 3]; 3]; 3],
        q_kkkk: [[[[0.0; 3]; 3]; 3]; 3],
    };

    for (k, extra) in extras.iter().enumerate() {
        let axes = [
            axis_fourth_derivatives(&mx, max_jx, pa.x, pb.x + extra[0], alpha, beta),
            axis_fourth_derivatives(&my, max_jy, pa.y, pb.y + extra[1], alpha, beta),
            axis_fourth_derivatives(&mz, max_jz, pa.z, pb.z + extra[2], alpha, beta),
        ];
        out.moments[k] = product(&axes, &[]);
        for c in 0..3 {
            set(&mut out.d_bra[k], c, product(&axes, &[(Center::Bra, c)]));
            set(&mut out.d_ket[k], c, product(&axes, &[(Center::Ket, c)]));
        }
        for a0 in 0..3 {
            for b0 in 0..3 {
                out.h_bra_bra[k][a0][b0] = product(&axes, &[(Center::Bra, a0), (Center::Bra, b0)]);
                out.h_bra_ket[k][a0][b0] = product(&axes, &[(Center::Bra, a0), (Center::Ket, b0)]);
                out.h_ket_ket[k][a0][b0] = product(&axes, &[(Center::Ket, a0), (Center::Ket, b0)]);
                for c0 in 0..3 {
                    out.t_bra_bra_bra[k][a0][b0][c0] = product(
                        &axes,
                        &[(Center::Bra, a0), (Center::Bra, b0), (Center::Bra, c0)],
                    );
                    out.t_bra_bra_ket[k][a0][b0][c0] = product(
                        &axes,
                        &[(Center::Bra, a0), (Center::Bra, b0), (Center::Ket, c0)],
                    );
                    out.t_bra_ket_ket[k][a0][b0][c0] = product(
                        &axes,
                        &[(Center::Bra, a0), (Center::Ket, b0), (Center::Ket, c0)],
                    );
                    out.t_ket_ket_ket[k][a0][b0][c0] = product(
                        &axes,
                        &[(Center::Ket, a0), (Center::Ket, b0), (Center::Ket, c0)],
                    );
                }
            }
        }
        // Fourth order: overlap only (see the module docs for the size rationale).
        if k == OVERLAP {
            primitive_overlap_fourth_derivatives(&axes, &mut out);
        }
    }
    out
}

#[inline]
fn set(v: &mut Vec3, axis: usize, value: f64) {
    match axis {
        0 => v.x = value,
        1 => v.y = value,
        2 => v.z = value,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::third_derivatives::contracted_pair_with_third_derivatives;
    use super::*;
    use crate::basis::CartesianComponent;
    use crate::params::AngularMomentum;

    fn shifted(mut v: Vec3, axis: usize, delta: f64) -> Vec3 {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            2 => v.z += delta,
            _ => panic!("axis"),
        }
        v
    }

    fn test_ao(
        angular: AngularMomentum,
        components: Vec<CartesianComponent>,
        primitives: Vec<(f64, f64)>,
    ) -> AOBasisFunction {
        AOBasisFunction {
            atom_index: 0,
            z: 1,
            shell_index: 0,
            shell_param_index: 0,
            shell_label: "test".to_string(),
            angular,
            cart_label: "test",
            components,
            hdiag_ev: 0.0,
            hdiag_ha: 0.0,
            slater: 1.0,
            principal_n: 2,
            nprim: primitives.len(),
            reference_occ: 0.0,
            is_valence: true,
            poly_raw: None,
            kcn_raw: None,
            lpar_raw: None,
            primitives: primitives
                .into_iter()
                .map(|(exponent, coefficient)| crate::sto::PrimitiveGaussian {
                    exponent,
                    coefficient,
                })
                .collect(),
        }
    }

    fn s_ao() -> AOBasisFunction {
        test_ao(
            AngularMomentum::S,
            vec![CartesianComponent::new(CartesianPower::new(0, 0, 0), 1.0)],
            vec![(0.9, 0.6), (2.3, 0.35)],
        )
    }

    fn p_ao() -> AOBasisFunction {
        test_ao(
            AngularMomentum::P,
            vec![CartesianComponent::new(CartesianPower::new(1, 0, 0), 1.0)],
            vec![(0.8, 0.7), (1.6, -0.2)],
        )
    }

    /// A spherical `d_{z²}`-like contraction: three Cartesian components, so the component
    /// loop is genuinely exercised.
    fn d_ao() -> AOBasisFunction {
        test_ao(
            AngularMomentum::D,
            vec![
                CartesianComponent::new(CartesianPower::new(0, 0, 2), 1.0),
                CartesianComponent::new(CartesianPower::new(2, 0, 0), -0.5),
                CartesianComponent::new(CartesianPower::new(0, 2, 0), -0.5),
            ],
            vec![(1.1, 0.5), (0.45, 0.25)],
        )
    }

    fn pairs() -> Vec<(&'static str, AOBasisFunction, AOBasisFunction)> {
        vec![
            ("s|s", s_ao(), s_ao()),
            ("s|p", s_ao(), p_ao()),
            ("p|d", p_ao(), d_ao()),
            ("d|d", d_ao(), d_ao()),
        ]
    }

    const CA: Vec3 = Vec3::new(-0.21, 0.13, 0.31);
    const CB: Vec3 = Vec3::new(0.44, -0.27, 0.19);

    /// Ladder test: every fourth-order pattern is the central finite difference of the
    /// matching third-order pattern with respect to the appropriate centre. The displaced
    /// evaluations use the *third*-derivative module, so this also links the new fourth
    /// order back to the already-validated third-order code (which the consistency test
    /// below proves is bit-identical to the fourth module's embedded third order).
    #[test]
    fn fourth_derivatives_match_third_derivative_fd() {
        let h = 1.0e-5;
        for (label, a, b) in pairs() {
            let ana = contracted_pair_with_fourth_derivatives(&a, &b, CA, CB);
            // Displaced third-order data, indexed by the displaced Cartesian axis.
            let bra_p: Vec<_> = (0..3)
                .map(|d| contracted_pair_with_third_derivatives(&a, &b, shifted(CA, d, h), CB))
                .collect();
            let bra_m: Vec<_> = (0..3)
                .map(|d| contracted_pair_with_third_derivatives(&a, &b, shifted(CA, d, -h), CB))
                .collect();
            let ket_p: Vec<_> = (0..3)
                .map(|d| contracted_pair_with_third_derivatives(&a, &b, CA, shifted(CB, d, h)))
                .collect();
            let ket_m: Vec<_> = (0..3)
                .map(|d| contracted_pair_with_third_derivatives(&a, &b, CA, shifted(CB, d, -h)))
                .collect();
            let fd_bra = |d: usize, pick: &dyn Fn(&ContractedPairThirdDerivatives) -> f64| {
                (pick(&bra_p[d]) - pick(&bra_m[d])) / (2.0 * h)
            };
            let fd_ket = |d: usize, pick: &dyn Fn(&ContractedPairThirdDerivatives) -> f64| {
                (pick(&ket_p[d]) - pick(&ket_m[d])) / (2.0 * h)
            };
            let mut worst = [0.0_f64; 6];
            for a0 in 0..3 {
                for b0 in 0..3 {
                    for c0 in 0..3 {
                        for d0 in 0..3 {
                            // ∂_{A_d} t_bbb[a][b][c] → q_bbbb (fully bra-symmetric).
                            let fd = fd_bra(d0, &|t| t.t_bra_bra_bra[OVERLAP][a0][b0][c0]);
                            check(
                                "q_bbbb",
                                label,
                                ana.q_bbbb[a0][b0][c0][d0],
                                fd,
                                &mut worst[0],
                            );
                            // ∂_{B_d} t_bbb[a][b][c] → q_bbbk (ket index last).
                            let fd = fd_ket(d0, &|t| t.t_bra_bra_bra[OVERLAP][a0][b0][c0]);
                            check(
                                "q_bbbk",
                                label,
                                ana.q_bbbk[a0][b0][c0][d0],
                                fd,
                                &mut worst[1],
                            );
                            // ∂_{B_d} t_bbk[a][b][c] → q_bbkk[a][b][c][d].
                            let fd = fd_ket(d0, &|t| t.t_bra_bra_ket[OVERLAP][a0][b0][c0]);
                            check(
                                "q_bbkk",
                                label,
                                ana.q_bbkk[a0][b0][c0][d0],
                                fd,
                                &mut worst[2],
                            );
                            // ∂_{B_d} t_bkk[a][b][c] → q_bkkk[a][b][c][d].
                            let fd = fd_ket(d0, &|t| t.t_bra_ket_ket[OVERLAP][a0][b0][c0]);
                            check(
                                "q_bkkk",
                                label,
                                ana.q_bkkk[a0][b0][c0][d0],
                                fd,
                                &mut worst[3],
                            );
                            // ∂_{B_d} t_kkk[a][b][c] → q_kkkk (fully ket-symmetric).
                            let fd = fd_ket(d0, &|t| t.t_ket_ket_ket[OVERLAP][a0][b0][c0]);
                            check(
                                "q_kkkk",
                                label,
                                ana.q_kkkk[a0][b0][c0][d0],
                                fd,
                                &mut worst[4],
                            );
                            // Mixed-pattern cross-check from the other side: differentiating
                            // t_bkk[a][b][c] = ∂_{A_a}∂_{B_b}∂_{B_c} w.r.t. A_d gives
                            // ∂_{A_a}∂_{A_d}∂_{B_b}∂_{B_c} = q_bbkk[a][d][b][c].
                            let fd = fd_bra(d0, &|t| t.t_bra_ket_ket[OVERLAP][a0][b0][c0]);
                            check(
                                "q_bbkk(cross)",
                                label,
                                ana.q_bbkk[a0][d0][b0][c0],
                                fd,
                                &mut worst[5],
                            );
                        }
                    }
                }
            }
            println!(
                "{label}: max |analytic-fd| bbbb={:.3e} bbbk={:.3e} bbkk={:.3e} bkkk={:.3e} kkkk={:.3e} bbkk-cross={:.3e}",
                worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
            );
        }
    }

    fn check(name: &str, label: &str, analytic: f64, fd: f64, worst: &mut f64) {
        let delta = (analytic - fd).abs();
        if delta > *worst {
            *worst = delta;
        }
        assert!(
            delta < 1.0e-8,
            "{name} [{label}]: analytic={analytic} fd={fd} delta={delta}"
        );
    }

    /// Translational invariance: the overlap depends only on `A - B`, so moving both centres
    /// together annihilates it. Applying `(∂_{A_δ} + ∂_{B_δ})` to each third-order pattern
    /// therefore gives zero, which in this module's index convention (bra first, ket last)
    /// reads
    ///
    /// ```text
    ///   ∂_A t_bbb[a][b][c] = q_bbbb[a][b][c][δ] ; ∂_B t_bbb[a][b][c] = q_bbbk[a][b][c][δ]
    ///   ∂_A t_bbk[a][b][c] = q_bbbk[a][b][δ][c] ; ∂_B t_bbk[a][b][c] = q_bbkk[a][b][c][δ]
    ///   ∂_A t_bkk[a][b][c] = q_bbkk[a][δ][b][c] ; ∂_B t_bkk[a][b][c] = q_bkkk[a][b][c][δ]
    ///   ∂_A t_kkk[a][b][c] = q_bkkk[δ][a][b][c] ; ∂_B t_kkk[a][b][c] = q_kkkk[a][b][c][δ]
    /// ```
    ///
    /// i.e. the new (bra) index is inserted **after the existing bra indices** and the new
    /// (ket) index **after the existing ket indices**. Each of the four sums must vanish.
    #[test]
    fn fourth_derivatives_are_translationally_invariant() {
        for (label, a, b) in pairs() {
            let q = contracted_pair_with_fourth_derivatives(&a, &b, CA, CB);
            let mut scale = 0.0_f64;
            for a0 in 0..3 {
                for b0 in 0..3 {
                    for c0 in 0..3 {
                        for d0 in 0..3 {
                            for v in [
                                q.q_bbbb[a0][b0][c0][d0],
                                q.q_bbbk[a0][b0][c0][d0],
                                q.q_bbkk[a0][b0][c0][d0],
                                q.q_bkkk[a0][b0][c0][d0],
                                q.q_kkkk[a0][b0][c0][d0],
                            ] {
                                scale = scale.max(v.abs());
                            }
                        }
                    }
                }
            }
            let tol = 1.0e-11 * (1.0 + scale);
            for a0 in 0..3 {
                for b0 in 0..3 {
                    for c0 in 0..3 {
                        for d0 in 0..3 {
                            let sums = [
                                (
                                    "t_bbb",
                                    q.q_bbbb[a0][b0][c0][d0] + q.q_bbbk[a0][b0][c0][d0],
                                ),
                                (
                                    "t_bbk",
                                    q.q_bbbk[a0][b0][d0][c0] + q.q_bbkk[a0][b0][c0][d0],
                                ),
                                (
                                    "t_bkk",
                                    q.q_bbkk[a0][d0][b0][c0] + q.q_bkkk[a0][b0][c0][d0],
                                ),
                                (
                                    "t_kkk",
                                    q.q_bkkk[d0][a0][b0][c0] + q.q_kkkk[a0][b0][c0][d0],
                                ),
                            ];
                            for (which, sum) in sums {
                                assert!(
                                    sum.abs() < tol,
                                    "{which} [{label}] {a0}{b0}{c0}{d0}: (∂_A+∂_B) = {sum}, tol {tol}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The embedded lower-order data must be *bit-identical* to what the third-derivative
    /// module produces for the same pair: the fourth-order module only appends one more
    /// application of the same operators, it must not perturb anything below it.
    #[test]
    fn embedded_third_order_matches_third_derivative_module() {
        for (label, a, b) in pairs() {
            let got = contracted_pair_with_fourth_derivatives(&a, &b, CA, CB).third_derivatives();
            let want = contracted_pair_with_third_derivatives(&a, &b, CA, CB);
            for k in 0..NMOMENT {
                assert_eq!(got.moments[k], want.moments[k], "moments {label} k={k}");
                for (name, g, w) in [
                    ("d_bra", got.d_bra[k], want.d_bra[k]),
                    ("d_ket", got.d_ket[k], want.d_ket[k]),
                ] {
                    assert_eq!((g.x, g.y, g.z), (w.x, w.y, w.z), "{name} {label} k={k}");
                }
                for i in 0..3 {
                    for j in 0..3 {
                        for (name, g, w) in [
                            ("h_bra_bra", got.h_bra_bra[k][i][j], want.h_bra_bra[k][i][j]),
                            ("h_bra_ket", got.h_bra_ket[k][i][j], want.h_bra_ket[k][i][j]),
                            ("h_ket_ket", got.h_ket_ket[k][i][j], want.h_ket_ket[k][i][j]),
                        ] {
                            assert_eq!(g, w, "{name} {label} k={k} {i}{j}");
                        }
                        for l in 0..3 {
                            for (name, g, w) in [
                                (
                                    "t_bra_bra_bra",
                                    got.t_bra_bra_bra[k][i][j][l],
                                    want.t_bra_bra_bra[k][i][j][l],
                                ),
                                (
                                    "t_bra_bra_ket",
                                    got.t_bra_bra_ket[k][i][j][l],
                                    want.t_bra_bra_ket[k][i][j][l],
                                ),
                                (
                                    "t_bra_ket_ket",
                                    got.t_bra_ket_ket[k][i][j][l],
                                    want.t_bra_ket_ket[k][i][j][l],
                                ),
                                (
                                    "t_ket_ket_ket",
                                    got.t_ket_ket_ket[k][i][j][l],
                                    want.t_ket_ket_ket[k][i][j][l],
                                ),
                            ] {
                                assert_eq!(g, w, "{name} {label} k={k} {i}{j}{l}");
                            }
                        }
                    }
                }
            }
        }
    }
}
