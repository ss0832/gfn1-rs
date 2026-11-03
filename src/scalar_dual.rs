// SPDX-License-Identifier: GPL-3.0-or-later
//! Scalar analytic forward differentiation through second order; no displacement.
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Dual {
    pub value: f64,
    pub derivative: f64,
    pub second: f64,
}
impl Dual {
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.,
            second: 0.,
        }
    }
    pub fn seed(value: f64, derivative: f64) -> Self {
        Self {
            value,
            derivative,
            second: 0.,
        }
    }
    pub fn seed_second(value: f64, derivative: f64, second: f64) -> Self {
        Self {
            value,
            derivative,
            second,
        }
    }
    pub fn powf(self, p: f64) -> Self {
        if p == 0. {
            return Self::constant(1.);
        }
        if p == 1. {
            return self;
        }
        Self {
            value: self.value.powf(p),
            derivative: if self.derivative == 0. {
                0.
            } else {
                p * self.value.powf(p - 1.) * self.derivative
            },
            second: (if self.second == 0. {
                0.
            } else {
                p * self.value.powf(p - 1.) * self.second
            }) + (if self.derivative == 0. {
                0.
            } else {
                p * (p - 1.) * self.value.powf(p - 2.) * self.derivative * self.derivative
            }),
        }
    }
    pub fn powi(self, p: usize) -> Self {
        if p == 0 {
            Self::constant(1.)
        } else if p == 1 {
            self
        } else {
            Self {
                value: self.value.powi(p as i32),
                derivative: p as f64 * self.value.powi(p as i32 - 1) * self.derivative,
                second: p as f64 * self.value.powi(p as i32 - 1) * self.second
                    + (p * (p - 1)) as f64
                        * self.value.powi(p as i32 - 2)
                        * self.derivative
                        * self.derivative,
            }
        }
    }
    pub fn sqrt(self) -> Self {
        self.powf(0.5)
    }
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            derivative: value * self.derivative,
            second: value * (self.second + self.derivative * self.derivative),
        }
    }
}
impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}
impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::seed_second(
            self.value + rhs.value,
            self.derivative + rhs.derivative,
            self.second + rhs.second,
        )
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::seed_second(
            self.value - rhs.value,
            self.derivative - rhs.derivative,
            self.second - rhs.second,
        )
    }
}
impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::seed_second(
            self.value * rhs.value,
            self.derivative * rhs.value + self.value * rhs.derivative,
            self.second * rhs.value
                + 2. * self.derivative * rhs.derivative
                + self.value * rhs.second,
        )
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        let derivative =
            (self.derivative * rhs.value - self.value * rhs.derivative) / (rhs.value * rhs.value);
        Self::seed_second(
            value,
            derivative,
            (self.second - value * rhs.second - 2. * derivative * rhs.derivative) / rhs.value,
        )
    }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self::seed_second(-self.value, -self.derivative, -self.second)
    }
}
macro_rules! scalar_ops {
    ($trait:ident,$method:ident,$op:tt)=>{
        impl $trait<f64> for Dual {type Output=Self;fn $method(self,rhs:f64)->Self {self $op Self::constant(rhs)}}
        impl $trait<Dual> for f64 {type Output=Dual;fn $method(self,rhs:Dual)->Dual {Dual::constant(self) $op rhs}}
    };
}
scalar_ops!(Add,add,+);
scalar_ops!(Sub,sub,-);
scalar_ops!(Mul,mul,*);
scalar_ops!(Div,div,/);
impl std::ops::AddAssign for Dual {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::SubAssign for Dual {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
