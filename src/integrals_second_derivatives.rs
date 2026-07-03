// SPDX-License-Identifier: GPL-3.0-or-later
#![allow(dead_code)]

use crate::basis::{AOBasisFunction, CartesianPower};
use crate::math::Vec3;

use super::{moment_1d_get, moment_1d_table};

#[derive(Clone, Debug)]
pub(crate) struct ContractedPairSecondDerivatives {
    pub moments: [f64; 10],
    pub d_bra: [Vec3; 10],
    pub d_ket: [Vec3; 10],
    pub h_bra_bra: [[[f64; 3]; 3]; 10],
    pub h_bra_ket: [[[f64; 3]; 3]; 10],
    pub h_ket_ket: [[[f64; 3]; 3]; 10],
}

pub(crate) fn contracted_pair_with_second_derivatives(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> ContractedPairSecondDerivatives {
    let mut out = ContractedPairSecondDerivatives {
        moments: [0.0; 10],
        d_bra: [Vec3::zero(); 10],
        d_ket: [Vec3::zero(); 10],
        h_bra_bra: [[[0.0; 3]; 3]; 10],
        h_bra_ket: [[[0.0; 3]; 3]; 10],
        h_ket_ket: [[[0.0; 3]; 3]; 10],
    };

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
                    let primitive = primitive_moments_second_derivatives_up_to_quadrupole(
                        ca_term.power,
                        cb_term.power,
                        pa.exponent,
                        pb.exponent,
                        ca,
                        cb,
                    );
                    let cfac_kab = cfac * kab_3d;
                    for k in 0..10 {
                        out.moments[k] += cfac_kab * primitive.moments[k];
                        out.d_bra[k] += primitive.d_bra[k] * cfac_kab;
                        out.d_ket[k] += primitive.d_ket[k] * cfac_kab;
                        for a_axis in 0..3 {
                            for b_axis in 0..3 {
                                out.h_bra_bra[k][a_axis][b_axis] +=
                                    cfac_kab * primitive.h_bra_bra[k][a_axis][b_axis];
                                out.h_bra_ket[k][a_axis][b_axis] +=
                                    cfac_kab * primitive.h_bra_ket[k][a_axis][b_axis];
                                out.h_ket_ket[k][a_axis][b_axis] +=
                                    cfac_kab * primitive.h_ket_ket[k][a_axis][b_axis];
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
struct PrimitiveMomentSecondDerivatives {
    moments: [f64; 10],
    d_bra: [Vec3; 10],
    d_ket: [Vec3; 10],
    h_bra_bra: [[[f64; 3]; 3]; 10],
    h_bra_ket: [[[f64; 3]; 3]; 10],
    h_ket_ket: [[[f64; 3]; 3]; 10],
}

#[derive(Clone, Copy, Debug)]
struct AxisDerivatives {
    value: f64,
    d_bra: f64,
    d_ket: f64,
    h_bra_bra: f64,
    h_bra_ket: f64,
    h_ket_ket: f64,
}

fn primitive_moments_second_derivatives_up_to_quadrupole(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
) -> PrimitiveMomentSecondDerivatives {
    let max_ix = pa.x + 2;
    let max_iy = pa.y + 2;
    let max_iz = pa.z + 2;
    let max_jx = pb.x + 4;
    let max_jy = pb.y + 4;
    let max_jz = pb.z + 4;
    let mx = moment_1d_table(max_ix, max_jx, alpha, beta, ca.x, cb.x);
    let my = moment_1d_table(max_iy, max_jy, alpha, beta, ca.y, cb.y);
    let mz = moment_1d_table(max_iz, max_jz, alpha, beta, ca.z, cb.z);
    let extras = [
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

    let mut out = PrimitiveMomentSecondDerivatives {
        moments: [0.0; 10],
        d_bra: [Vec3::zero(); 10],
        d_ket: [Vec3::zero(); 10],
        h_bra_bra: [[[0.0; 3]; 3]; 10],
        h_bra_ket: [[[0.0; 3]; 3]; 10],
        h_ket_ket: [[[0.0; 3]; 3]; 10],
    };

    for (k, extra) in extras.iter().enumerate() {
        let powers = [
            (pa.x, pb.x + extra[0]),
            (pa.y, pb.y + extra[1]),
            (pa.z, pb.z + extra[2]),
        ];
        let axes = [
            axis_derivatives(&mx, max_jx, powers[0].0, powers[0].1, alpha, beta),
            axis_derivatives(&my, max_jy, powers[1].0, powers[1].1, alpha, beta),
            axis_derivatives(&mz, max_jz, powers[2].0, powers[2].1, alpha, beta),
        ];
        out.moments[k] = axes[0].value * axes[1].value * axes[2].value;
        out.d_bra[k] = Vec3::new(
            axes[0].d_bra * axes[1].value * axes[2].value,
            axes[0].value * axes[1].d_bra * axes[2].value,
            axes[0].value * axes[1].value * axes[2].d_bra,
        );
        out.d_ket[k] = Vec3::new(
            axes[0].d_ket * axes[1].value * axes[2].value,
            axes[0].value * axes[1].d_ket * axes[2].value,
            axes[0].value * axes[1].value * axes[2].d_ket,
        );
        for a_axis in 0..3 {
            for b_axis in 0..3 {
                out.h_bra_bra[k][a_axis][b_axis] = second_product(
                    &axes,
                    a_axis,
                    b_axis,
                    CenterDerivative::Bra,
                    CenterDerivative::Bra,
                );
                out.h_bra_ket[k][a_axis][b_axis] = second_product(
                    &axes,
                    a_axis,
                    b_axis,
                    CenterDerivative::Bra,
                    CenterDerivative::Ket,
                );
                out.h_ket_ket[k][a_axis][b_axis] = second_product(
                    &axes,
                    a_axis,
                    b_axis,
                    CenterDerivative::Ket,
                    CenterDerivative::Ket,
                );
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum CenterDerivative {
    Bra,
    Ket,
}

fn second_product(
    axes: &[AxisDerivatives; 3],
    lhs_axis: usize,
    rhs_axis: usize,
    lhs_center: CenterDerivative,
    rhs_center: CenterDerivative,
) -> f64 {
    let mut value = 1.0;
    for axis in 0..3 {
        let derivative = if axis == lhs_axis && axis == rhs_axis {
            match (lhs_center, rhs_center) {
                (CenterDerivative::Bra, CenterDerivative::Bra) => axes[axis].h_bra_bra,
                (CenterDerivative::Bra, CenterDerivative::Ket)
                | (CenterDerivative::Ket, CenterDerivative::Bra) => axes[axis].h_bra_ket,
                (CenterDerivative::Ket, CenterDerivative::Ket) => axes[axis].h_ket_ket,
            }
        } else if axis == lhs_axis {
            first_axis_derivative(axes[axis], lhs_center)
        } else if axis == rhs_axis {
            first_axis_derivative(axes[axis], rhs_center)
        } else {
            axes[axis].value
        };
        value *= derivative;
    }
    value
}

fn first_axis_derivative(axis: AxisDerivatives, center: CenterDerivative) -> f64 {
    match center {
        CenterDerivative::Bra => axis.d_bra,
        CenterDerivative::Ket => axis.d_ket,
    }
}

fn axis_derivatives(
    table: &[f64],
    max_j: usize,
    i: usize,
    j: usize,
    alpha: f64,
    beta: f64,
) -> AxisDerivatives {
    let get = |ii: isize, jj: isize| -> f64 {
        if ii < 0 || jj < 0 {
            0.0
        } else {
            moment_1d_get(table, max_j, ii as usize, jj as usize)
        }
    };
    let i = i as isize;
    let j = j as isize;
    let value = get(i, j);
    let d_bra = 2.0 * alpha * get(i + 1, j) - (i as f64) * get(i - 1, j);
    let d_ket = 2.0 * beta * get(i, j + 1) - (j as f64) * get(i, j - 1);
    let h_bra_bra = 4.0 * alpha * alpha * get(i + 2, j)
        - 2.0 * alpha * (2.0 * i as f64 + 1.0) * get(i, j)
        + (i * (i - 1)) as f64 * get(i - 2, j);
    let h_ket_ket = 4.0 * beta * beta * get(i, j + 2)
        - 2.0 * beta * (2.0 * j as f64 + 1.0) * get(i, j)
        + (j * (j - 1)) as f64 * get(i, j - 2);
    let h_bra_ket = 4.0 * alpha * beta * get(i + 1, j + 1)
        - 2.0 * alpha * j as f64 * get(i + 1, j - 1)
        - 2.0 * beta * i as f64 * get(i - 1, j + 1)
        + (i * j) as f64 * get(i - 1, j - 1);
    AxisDerivatives {
        value,
        d_bra,
        d_ket,
        h_bra_bra,
        h_bra_ket,
        h_ket_ket,
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
            _ => panic!("axis out of range"),
        }
        v
    }

    #[test]
    fn primitive_second_derivatives_match_first_derivative_finite_difference() {
        let pa = CartesianPower::new(1, 0, 0);
        let pb = CartesianPower::new(0, 1, 0);
        let alpha = 0.8;
        let beta = 1.1;
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let h = 1.0e-5;
        let p = alpha + beta;
        let kab_3d = (-alpha * beta / p * (ca - cb).norm2()).exp();
        let analytic =
            primitive_moments_second_derivatives_up_to_quadrupole(pa, pb, alpha, beta, ca, cb);
        for moment in 0..10 {
            for lhs_axis in 0..3 {
                for rhs_axis in 0..3 {
                    let ca_plus = shifted(ca, rhs_axis, h);
                    let kab_plus_bra = (-alpha * beta / p * (ca_plus - cb).norm2()).exp();
                    let (_, bra_plus, _) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca_plus, cb,
                    );
                    let bra_plus_scaled = bra_plus[moment] * kab_plus_bra;

                    let ca_minus = shifted(ca, rhs_axis, -h);
                    let kab_minus_bra = (-alpha * beta / p * (ca_minus - cb).norm2()).exp();
                    let (_, bra_minus, _) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca_minus, cb,
                    );
                    let bra_minus_scaled = bra_minus[moment] * kab_minus_bra;

                    let fd_bb = (component(bra_plus_scaled, lhs_axis)
                        - component(bra_minus_scaled, lhs_axis))
                        / (2.0 * h);
                    let analytic_bb = analytic.h_bra_bra[moment][lhs_axis][rhs_axis] * kab_3d;
                    assert!(
                        (analytic_bb - fd_bb).abs() < 1.0e-7,
                        "bra-bra moment={moment} lhs={lhs_axis} rhs={rhs_axis}: analytic={} fd={}",
                        analytic_bb,
                        fd_bb
                    );

                    let cb_plus = shifted(cb, rhs_axis, h);
                    let kab_plus_ket = (-alpha * beta / p * (ca - cb_plus).norm2()).exp();
                    let (_, bra_plus, _) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca, cb_plus,
                    );
                    let bra_plus_scaled = bra_plus[moment] * kab_plus_ket;

                    let cb_minus = shifted(cb, rhs_axis, -h);
                    let kab_minus_ket = (-alpha * beta / p * (ca - cb_minus).norm2()).exp();
                    let (_, bra_minus, _) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca, cb_minus,
                    );
                    let bra_minus_scaled = bra_minus[moment] * kab_minus_ket;

                    let fd_bk = (component(bra_plus_scaled, lhs_axis)
                        - component(bra_minus_scaled, lhs_axis))
                        / (2.0 * h);
                    let analytic_bk = analytic.h_bra_ket[moment][lhs_axis][rhs_axis] * kab_3d;
                    assert!(
                        (analytic_bk - fd_bk).abs() < 1.0e-7,
                        "bra-ket moment={moment} lhs={lhs_axis} rhs={rhs_axis}: analytic={} fd={}",
                        analytic_bk,
                        fd_bk
                    );

                    let cb_plus = shifted(cb, rhs_axis, h);
                    let kab_plus_ket = (-alpha * beta / p * (ca - cb_plus).norm2()).exp();
                    let (_, _, ket_plus) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca, cb_plus,
                    );
                    let ket_plus_scaled = ket_plus[moment] * kab_plus_ket;

                    let cb_minus = shifted(cb, rhs_axis, -h);
                    let kab_minus_ket = (-alpha * beta / p * (ca - cb_minus).norm2()).exp();
                    let (_, _, ket_minus) = primitive_moments_derivatives_up_to_quadrupole(
                        pa, pb, alpha, beta, ca, cb_minus,
                    );
                    let ket_minus_scaled = ket_minus[moment] * kab_minus_ket;

                    let fd_kk = (component(ket_plus_scaled, lhs_axis)
                        - component(ket_minus_scaled, lhs_axis))
                        / (2.0 * h);
                    let analytic_kk = analytic.h_ket_ket[moment][lhs_axis][rhs_axis] * kab_3d;
                    assert!(
                        (analytic_kk - fd_kk).abs() < 1.0e-7,
                        "ket-ket moment={moment} lhs={lhs_axis} rhs={rhs_axis}: analytic={} fd={}",
                        analytic_kk,
                        fd_kk
                    );
                }
            }
        }
    }

    fn component(v: Vec3, axis: usize) -> f64 {
        match axis {
            0 => v.x,
            1 => v.y,
            2 => v.z,
            _ => panic!("axis out of range"),
        }
    }
}
