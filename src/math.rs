// SPDX-License-Identifier: GPL-3.0-or-later
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    #[inline]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
    #[inline]
    pub fn outer_trace_with(self, rhs: Self) -> f64 {
        self.dot(rhs)
    }
    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub col: [Vec3; 3],
}

impl Mat3 {
    #[inline]
    pub const fn from_columns(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { col: [a, b, c] }
    }
    #[inline]
    pub fn mul_vec(self, frac: Vec3) -> Vec3 {
        self.col[0] * frac.x + self.col[1] * frac.y + self.col[2] * frac.z
    }
    #[inline]
    pub fn determinant(self) -> f64 {
        self.col[0].dot(self.col[1].cross(self.col[2]))
    }
    #[inline]
    pub fn volume(self) -> f64 {
        self.determinant().abs()
    }
    #[inline]
    pub fn column(self, i: usize) -> Vec3 {
        self.col[i]
    }
}

#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

#[inline]
pub fn reciprocal_vectors_2pi(cell: Mat3) -> [Vec3; 3] {
    let a = cell.col[0];
    let b = cell.col[1];
    let c = cell.col[2];
    let v = a.dot(b.cross(c));
    let scale = 2.0 * PI / v;
    [b.cross(c) * scale, c.cross(a) * scale, a.cross(b) * scale]
}
