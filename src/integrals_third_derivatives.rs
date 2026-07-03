// SPDX-License-Identifier: GPL-3.0-or-later
#![allow(dead_code)]
//! Third Cartesian derivatives of the contracted AO moment integrals (overlap and the
//! dipole/quadrupole moments), w.r.t. the bra (`A`) and ket (`B`) centres. This is the
//! integral half of the non-PBC analytic **third nuclear derivative** (2n+1 rule); it
//! extends [`super::second_derivatives`] by exactly one order, reusing the same 1D moment
//! recurrence ([`super::moment_1d_table`]) and the elementary Gaussian center-derivative
//! operators
//!
//! ```text
//!   (∂/∂A_x) M(i,j) = 2α M(i+1,j) − i M(i−1,j)        (bra)
//!   (∂/∂B_x) M(i,j) = 2β M(i,j+1) − j M(i,j−1)        (ket)
//! ```
//!
//! Because mixed partials commute, every higher derivative is obtained by composing these
//! two elementary operators; we therefore build the per-axis derivative ladder once and
//! contract it with a single multiplicity-counting [`product`] helper (no hand-expanded
//! recurrence beyond the elementary step, which keeps the third order correct by
//! construction). The four distinct center patterns of a third derivative are stored as
//! `t_bbb`/`t_bbk`/`t_bkk`/`t_kkk`, each a fully-symmetric `3×3×3` Cartesian tensor with the
//! ket indices last (`t_bbk[a][b][c] = ∂_{A_a}∂_{A_b}∂_{B_c}`).

use crate::basis::{AOBasisFunction, CartesianPower};
use crate::math::Vec3;

use super::{moment_1d_get, moment_1d_table};

/// Number of AO moment components carried (overlap + 3 dipole + 6 quadrupole), matching
/// [`super::second_derivatives::ContractedPairSecondDerivatives`].
pub(crate) const NMOMENT: usize = 10;

type Mat3 = [[f64; 3]; 3];
type Ten3 = [[[f64; 3]; 3]; 3];

#[derive(Clone, Debug)]
pub(crate) struct ContractedPairThirdDerivatives {
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
}

impl ContractedPairThirdDerivatives {
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
        }
    }
}

pub(crate) fn contracted_pair_with_third_derivatives(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> ContractedPairThirdDerivatives {
    let mut out = ContractedPairThirdDerivatives::zero();
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
                    let prim = primitive_moments_third_derivatives_up_to_quadrupole(
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
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub(crate) struct PrimitiveMomentThirdDerivatives {
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
}

/// Per-axis 1D moment value and its bra/ket derivative ladder up to **third** order. Built
/// by composing the elementary operators [`op_bra`]/[`op_ket`]; mixed entries use the
/// commuted canonical form (`t_bra_bra_ket = ∂_ket ∂_bra ∂_bra`, etc.).
#[derive(Clone, Copy, Debug, Default)]
struct AxisThirdDerivatives {
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

fn axis_third_derivatives(
    table: &[f64],
    max_j: usize,
    i: usize,
    j: usize,
    alpha: f64,
    beta: f64,
) -> AxisThirdDerivatives {
    let m = |ii: isize, jj: isize| -> f64 {
        if ii < 0 || jj < 0 {
            0.0
        } else {
            moment_1d_get(table, max_j, ii as usize, jj as usize)
        }
    };
    // First-order ladders as closures so the higher operators can compose them.
    let d_b = |ii: isize, jj: isize| op_bra(&m, alpha, ii, jj);
    let d_k = |ii: isize, jj: isize| op_ket(&m, beta, ii, jj);
    let d_bb = |ii: isize, jj: isize| op_bra(&d_b, alpha, ii, jj);
    let d_bk = |ii: isize, jj: isize| op_ket(&d_b, beta, ii, jj);
    let d_kk = |ii: isize, jj: isize| op_ket(&d_k, beta, ii, jj);
    let i = i as isize;
    let j = j as isize;
    AxisThirdDerivatives {
        value: m(i, j),
        d_bra: d_b(i, j),
        d_ket: d_k(i, j),
        h_bra_bra: d_bb(i, j),
        h_bra_ket: d_bk(i, j),
        h_ket_ket: d_kk(i, j),
        t_bra_bra_bra: op_bra(&d_bb, alpha, i, j),
        t_bra_bra_ket: op_ket(&d_bb, beta, i, j),
        t_bra_ket_ket: op_bra(&d_kk, alpha, i, j),
        t_ket_ket_ket: op_ket(&d_kk, beta, i, j),
    }
}

/// The 1D factor for `bra` bra-derivatives and `ket` ket-derivatives on one Cartesian axis
/// (their order is irrelevant — mixed partials commute, so only the counts matter).
#[inline]
fn axis_factor(a: &AxisThirdDerivatives, bra: usize, ket: usize) -> f64 {
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
        _ => unreachable!("third derivative needs at most 3 selections per axis"),
    }
}

/// The 3D moment derivative selected by up to three `(center, axis)` selections: a product
/// of the three per-axis factors, each at the multiplicity implied by the selections that
/// land on that axis.
fn product(axes: &[AxisThirdDerivatives; 3], sels: &[(Center, usize)]) -> f64 {
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

fn primitive_moments_third_derivatives_up_to_quadrupole(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
) -> PrimitiveMomentThirdDerivatives {
    // bra power raised by ≤3 (third derivative); ket power raised by the moment operator
    // (≤2, quadrupole) plus the third derivative (≤3) ⇒ +5.
    let max_ix = pa.x + 3;
    let max_iy = pa.y + 3;
    let max_iz = pa.z + 3;
    let max_jx = pb.x + 5;
    let max_jy = pb.y + 5;
    let max_jz = pb.z + 5;
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

    let mut out = PrimitiveMomentThirdDerivatives {
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
    };

    for (k, extra) in extras.iter().enumerate() {
        let axes = [
            axis_third_derivatives(&mx, max_jx, pa.x, pb.x + extra[0], alpha, beta),
            axis_third_derivatives(&my, max_jy, pa.y, pb.y + extra[1], alpha, beta),
            axis_third_derivatives(&mz, max_jz, pa.z, pb.z + extra[2], alpha, beta),
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
    use super::*;
    use crate::integrals::primitive_moments_derivatives_up_to_quadrupole;

    fn shifted(mut v: Vec3, axis: usize, delta: f64) -> Vec3 {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            2 => v.z += delta,
            _ => panic!("axis"),
        }
        v
    }

    fn component(v: Vec3, axis: usize) -> f64 {
        match axis {
            0 => v.x,
            1 => v.y,
            2 => v.z,
            _ => panic!("axis"),
        }
    }

    fn kab(alpha: f64, beta: f64, ca: Vec3, cb: Vec3) -> f64 {
        let p = alpha + beta;
        (-alpha * beta / p * (ca - cb).norm2()).exp()
    }

    // Link 1 (external): this module's analytic *second* derivatives, ×kab, match the
    // central FD of the EXISTING primitive *first* derivatives ×kab. This validates the
    // axis ladder + `product` machinery against independently written code.
    #[test]
    fn second_derivatives_match_first_derivative_fd() {
        let pa = CartesianPower::new(1, 0, 0);
        let pb = CartesianPower::new(0, 1, 0);
        let (alpha, beta) = (0.8, 1.1);
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let h = 1.0e-5;
        let k0 = kab(alpha, beta, ca, cb);
        let ana = primitive_moments_third_derivatives_up_to_quadrupole(pa, pb, alpha, beta, ca, cb);
        for m in 0..NMOMENT {
            for l in 0..3 {
                for r in 0..3 {
                    // ∂_{A_r} of d_bra[l] → h_bra_bra[l][r]
                    let plus = {
                        let cap = shifted(ca, r, h);
                        let (_, db, _) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, cap, cb,
                        );
                        component(db[m], l) * kab(alpha, beta, cap, cb)
                    };
                    let minus = {
                        let cam = shifted(ca, r, -h);
                        let (_, db, _) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, cam, cb,
                        );
                        component(db[m], l) * kab(alpha, beta, cam, cb)
                    };
                    let fd = (plus - minus) / (2.0 * h);
                    assert!(
                        (ana.h_bra_bra[m][l][r] * k0 - fd).abs() < 1.0e-7,
                        "h_bb m={m} l={l} r={r}: {} vs {fd}",
                        ana.h_bra_bra[m][l][r] * k0
                    );
                    // ∂_{B_r} of d_bra[l] → h_bra_ket[l][r]
                    let plus = {
                        let cbp = shifted(cb, r, h);
                        let (_, db, _) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, ca, cbp,
                        );
                        component(db[m], l) * kab(alpha, beta, ca, cbp)
                    };
                    let minus = {
                        let cbm = shifted(cb, r, -h);
                        let (_, db, _) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, ca, cbm,
                        );
                        component(db[m], l) * kab(alpha, beta, ca, cbm)
                    };
                    let fd = (plus - minus) / (2.0 * h);
                    assert!(
                        (ana.h_bra_ket[m][l][r] * k0 - fd).abs() < 1.0e-7,
                        "h_bk m={m} l={l} r={r}: {} vs {fd}",
                        ana.h_bra_ket[m][l][r] * k0
                    );
                    // ∂_{B_r} of d_ket[l] → h_ket_ket[l][r]
                    let plus = {
                        let cbp = shifted(cb, r, h);
                        let (_, _, dk) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, ca, cbp,
                        );
                        component(dk[m], l) * kab(alpha, beta, ca, cbp)
                    };
                    let minus = {
                        let cbm = shifted(cb, r, -h);
                        let (_, _, dk) = primitive_moments_derivatives_up_to_quadrupole(
                            pa, pb, alpha, beta, ca, cbm,
                        );
                        component(dk[m], l) * kab(alpha, beta, ca, cbm)
                    };
                    let fd = (plus - minus) / (2.0 * h);
                    assert!(
                        (ana.h_ket_ket[m][l][r] * k0 - fd).abs() < 1.0e-7,
                        "h_kk m={m} l={l} r={r}: {} vs {fd}",
                        ana.h_ket_ket[m][l][r] * k0
                    );
                }
            }
        }
    }

    // Link 2: this module's analytic *third* derivatives, ×kab, match the central FD of its
    // own analytic *second* derivatives ×kab. Combined with Link 1, the third derivatives
    // are validated end-to-end (the third = ∂ of an independently-checked second).
    #[test]
    fn third_derivatives_match_second_derivative_fd() {
        let pa = CartesianPower::new(1, 0, 0);
        let pb = CartesianPower::new(0, 1, 0);
        let (alpha, beta) = (0.8, 1.1);
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let h = 1.0e-5;
        let k0 = kab(alpha, beta, ca, cb);
        let ana = primitive_moments_third_derivatives_up_to_quadrupole(pa, pb, alpha, beta, ca, cb);
        let second = |ca: Vec3, cb: Vec3| {
            let p =
                primitive_moments_third_derivatives_up_to_quadrupole(pa, pb, alpha, beta, ca, cb);
            let kk = kab(alpha, beta, ca, cb);
            (p, kk)
        };
        for m in 0..NMOMENT {
            for a0 in 0..3 {
                for b0 in 0..3 {
                    for c0 in 0..3 {
                        // t_bbb = ∂_{A_c} h_bra_bra[a][b]
                        let fd = {
                            let (pp, kp) = second(shifted(ca, c0, h), cb);
                            let (pm, km) = second(shifted(ca, c0, -h), cb);
                            (pp.h_bra_bra[m][a0][b0] * kp - pm.h_bra_bra[m][a0][b0] * km)
                                / (2.0 * h)
                        };
                        assert!(
                            (ana.t_bra_bra_bra[m][a0][b0][c0] * k0 - fd).abs() < 1.0e-6,
                            "t_bbb m={m} {a0}{b0}{c0}: {} vs {fd}",
                            ana.t_bra_bra_bra[m][a0][b0][c0] * k0
                        );
                        // t_bbk = ∂_{B_c} h_bra_bra[a][b]
                        let fd = {
                            let (pp, kp) = second(ca, shifted(cb, c0, h));
                            let (pm, km) = second(ca, shifted(cb, c0, -h));
                            (pp.h_bra_bra[m][a0][b0] * kp - pm.h_bra_bra[m][a0][b0] * km)
                                / (2.0 * h)
                        };
                        assert!(
                            (ana.t_bra_bra_ket[m][a0][b0][c0] * k0 - fd).abs() < 1.0e-6,
                            "t_bbk m={m} {a0}{b0}{c0}: {} vs {fd}",
                            ana.t_bra_bra_ket[m][a0][b0][c0] * k0
                        );
                        // t_bkk = ∂_{A_a} h_ket_ket[b][c]
                        let fd = {
                            let (pp, kp) = second(shifted(ca, a0, h), cb);
                            let (pm, km) = second(shifted(ca, a0, -h), cb);
                            (pp.h_ket_ket[m][b0][c0] * kp - pm.h_ket_ket[m][b0][c0] * km)
                                / (2.0 * h)
                        };
                        assert!(
                            (ana.t_bra_ket_ket[m][a0][b0][c0] * k0 - fd).abs() < 1.0e-6,
                            "t_bkk m={m} {a0}{b0}{c0}: {} vs {fd}",
                            ana.t_bra_ket_ket[m][a0][b0][c0] * k0
                        );
                        // t_kkk = ∂_{B_a} h_ket_ket[b][c]
                        let fd = {
                            let (pp, kp) = second(ca, shifted(cb, a0, h));
                            let (pm, km) = second(ca, shifted(cb, a0, -h));
                            (pp.h_ket_ket[m][b0][c0] * kp - pm.h_ket_ket[m][b0][c0] * km)
                                / (2.0 * h)
                        };
                        assert!(
                            (ana.t_ket_ket_ket[m][a0][b0][c0] * k0 - fd).abs() < 1.0e-6,
                            "t_kkk m={m} {a0}{b0}{c0}: {} vs {fd}",
                            ana.t_ket_ket_ket[m][a0][b0][c0] * k0
                        );
                    }
                }
            }
        }
    }
}
