// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared forward-mode AD jets up to fourth order.
//!
//! `Jet2`/`Jet3` mirror the private dual-number types that historically lived
//! in `dispersion.rs` (and halogen's local variants) so those modules can port
//! onto one implementation; `Jet4` extends the same design to fourth
//! derivatives for the quartic force-constant work.
//!
//! Layout: `gradient[i]`, `hessian[i*n + j]`, `third[(i*n + j)*n + k]`,
//! `fourth[((i*n + j)*n + k)*n + l]` — flat row-major over `n` seed variables.
//! All arrays are dense: a `Jet4` costs `O(n⁴)` memory, so keep the seed count
//! per jet small (per-pair/per-triple seeds, or a capped full-space `n`).
//!
//! `compose(value, φ1..φk)` applies a scalar function through Faà di Bruno's
//! formula given the outer derivatives φ; `powf`/`exp`/`sqrt` are thin wrappers.
//!
//! [`Jet1`] is the DIRECTIONAL specialisation: `f(R + t·v)` carried in the single
//! scalar `t`, so it stores five doubles instead of `O(n⁴)` and its products cost
//! `O(1)` instead of `O(n⁴)`. A *directional* fourth derivative `e⁗[v]` needs
//! nothing more than that univariate Taylor coefficient, which is what lets the
//! directional quartic run above the full-space `Jet4` system-size cap.

use std::cell::RefCell;

/// A scalar carried to fourth order along ONE variable `t`: `f(t₀)` together with
/// `f′, f″, f‴, f⁗` at `t₀`.
///
/// The seed convention is the *directional* one: a nuclear coordinate `R_a` is seeded with
/// `d1 = v_a` (see [`direction_component`]), so `d4` of the assembled energy IS
/// `Σ_abcd v_a v_b v_c v_d ∂⁴E/∂R_a∂R_b∂R_c∂R_d`. Every operation below is the univariate
/// specialisation of the corresponding [`Jet4`] operation, term for term, so an expression
/// written once against the shared jet op-sets differentiates identically at either width.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Jet1 {
    pub value: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
}

thread_local! {
    /// The direction the [`Jet1`] geometry seeds differentiate along.
    ///
    /// The generic jet pipelines (`dispersion`'s `DispJet`, `halogen`'s `HalJet`) seed geometry
    /// through per-DOF hooks — `seed_gradient(dof, …)` / `variable(…, dof)` — whose signatures
    /// carry no direction. Rather than fork those pipelines, the directional entry points install
    /// `v` here for the duration of one call via [`DirectionScope`]; the seeds then read
    /// `v[dof]`. It is a THREAD-LOCAL, so the polarization driver's rayon fan-out over directions
    /// is safe by construction: each worker installs and reads its own direction, and the jet
    /// pipelines themselves are sequential within one evaluation. Reading a component with no
    /// direction installed panics rather than silently returning zero.
    static JET1_DIRECTION: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Installs a direction for [`Jet1`] geometry seeds and restores the previous one on drop, so
/// nesting (a directional term evaluated inside another) cannot leak a stale direction.
#[must_use = "the direction is uninstalled when the scope is dropped"]
pub(crate) struct DirectionScope {
    previous: Vec<f64>,
}

impl DirectionScope {
    pub(crate) fn install(v: &[f64]) -> Self {
        let previous = JET1_DIRECTION.with(|slot| slot.replace(v.to_vec()));
        Self { previous }
    }
}

impl Drop for DirectionScope {
    fn drop(&mut self) {
        let previous = std::mem::take(&mut self.previous);
        JET1_DIRECTION.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// `v[dof]` of the direction installed by the innermost live [`DirectionScope`].
pub(crate) fn direction_component(dof: usize) -> f64 {
    JET1_DIRECTION.with(|slot| {
        let v = slot.borrow();
        assert!(
            dof < v.len(),
            "Jet1 geometry seed for DOF {dof} with no direction installed (len {}) — a \
             directional jet pipeline must run inside a DirectionScope",
            v.len()
        );
        v[dof]
    })
}

impl Jet1 {
    #[inline]
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
        }
    }

    /// A nuclear coordinate seeded directionally: `R_dof + t·v_dof`.
    #[inline]
    pub(crate) fn variable(value: f64, dof: usize) -> Self {
        Self {
            value,
            d1: direction_component(dof),
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
        }
    }

    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value + rhs.value,
            d1: self.d1 + rhs.d1,
            d2: self.d2 + rhs.d2,
            d3: self.d3 + rhs.d3,
            d4: self.d4 + rhs.d4,
        }
    }

    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self {
            value: self.value - rhs.value,
            d1: self.d1 - rhs.d1,
            d2: self.d2 - rhs.d2,
            d3: self.d3 - rhs.d3,
            d4: self.d4 - rhs.d4,
        }
    }

    #[inline]
    pub fn add_scalar(&self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            ..*self
        }
    }

    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            value: self.value * s,
            d1: self.d1 * s,
            d2: self.d2 * s,
            d3: self.d3 * s,
            d4: self.d4 * s,
        }
    }

    /// `self += s · other`.
    #[inline]
    pub fn add_scaled(&mut self, other: &Self, s: f64) {
        self.value += other.value * s;
        self.d1 += other.d1 * s;
        self.d2 += other.d2 * s;
        self.d3 += other.d3 * s;
        self.d4 += other.d4 * s;
    }

    /// The univariate Leibniz rule: `(fg)^(k) = Σ C(k,m) f^(m) g^(k−m)`.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = (self, rhs);
        Self {
            value: a.value * b.value,
            d1: a.d1 * b.value + a.value * b.d1,
            d2: a.d2 * b.value + 2.0 * a.d1 * b.d1 + a.value * b.d2,
            d3: a.d3 * b.value + 3.0 * a.d2 * b.d1 + 3.0 * a.d1 * b.d2 + a.value * b.d3,
            d4: a.d4 * b.value
                + 4.0 * a.d3 * b.d1
                + 6.0 * a.d2 * b.d2
                + 4.0 * a.d1 * b.d3
                + a.value * b.d4,
        }
    }

    #[inline]
    pub fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    /// Faà di Bruno for one variable: the chain rule for `g(f(t))` given `g`'s derivatives
    /// `φ1..φ4` at `f(t₀)`.
    #[inline]
    pub fn compose(&self, value: f64, p1: f64, p2: f64, p3: f64, p4: f64) -> Self {
        let (u1, u2, u3, u4) = (self.d1, self.d2, self.d3, self.d4);
        Self {
            value,
            d1: p1 * u1,
            d2: p1 * u2 + p2 * u1 * u1,
            d3: p1 * u3 + 3.0 * p2 * u1 * u2 + p3 * u1 * u1 * u1,
            d4: p1 * u4
                + p2 * (4.0 * u1 * u3 + 3.0 * u2 * u2)
                + p3 * (6.0 * u1 * u1 * u2)
                + p4 * (u1 * u1 * u1 * u1),
        }
    }

    #[inline]
    pub fn powf(&self, p: f64) -> Self {
        let x = self.value;
        let v = x.powf(p);
        let p1 = p * x.powf(p - 1.0);
        let p2 = p * (p - 1.0) * x.powf(p - 2.0);
        let p3 = p * (p - 1.0) * (p - 2.0) * x.powf(p - 3.0);
        let p4 = p * (p - 1.0) * (p - 2.0) * (p - 3.0) * x.powf(p - 4.0);
        self.compose(v, p1, p2, p3, p4)
    }

    #[inline]
    pub fn exp(&self) -> Self {
        let e = self.value.exp();
        self.compose(e, e, e, e, e)
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

#[derive(Clone, Debug)]
pub struct Jet2 {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
}

impl Jet2 {
    pub fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; n],
            hessian: vec![0.0; n * n],
        }
    }

    pub fn variable(value: f64, n: usize, dof: usize) -> Self {
        let mut out = Self::constant(value, n);
        out.gradient[dof] = 1.0;
        out
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.gradient.len()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value + rhs.value, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] + rhs.gradient[i];
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] + rhs.hessian[i];
        }
        out
    }

    pub fn add_scalar(&self, rhs: f64) -> Self {
        let mut out = self.clone();
        out.value += rhs;
        out
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    pub fn scale(&self, s: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value * s, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] * s;
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] * s;
        }
        out
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let n = self.n();
        let (a, b) = (self, rhs);
        let mut out = Self::constant(a.value * b.value, n);
        for i in 0..n {
            out.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] = a.hessian[i * n + j] * b.value
                    + a.value * b.hessian[i * n + j]
                    + a.gradient[i] * b.gradient[j]
                    + a.gradient[j] * b.gradient[i];
            }
        }
        out
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    /// Faà di Bruno composition given the outer scalar derivatives `φ1, φ2`.
    pub fn compose(&self, value: f64, phi1: f64, phi2: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(value, n);
        for i in 0..n {
            out.gradient[i] = phi1 * self.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] =
                    phi1 * self.hessian[i * n + j] + phi2 * self.gradient[i] * self.gradient[j];
            }
        }
        out
    }

    pub fn powf(&self, p: f64) -> Self {
        let v = self.value;
        self.compose(
            v.powf(p),
            p * v.powf(p - 1.0),
            p * (p - 1.0) * v.powf(p - 2.0),
        )
    }

    pub fn exp(&self) -> Self {
        let e = self.value.exp();
        self.compose(e, e, e)
    }

    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

#[derive(Clone, Debug)]
pub struct Jet3 {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
    pub third: Vec<f64>,
}

impl Jet3 {
    pub fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; n],
            hessian: vec![0.0; n * n],
            third: vec![0.0; n * n * n],
        }
    }

    pub fn variable(value: f64, n: usize, dof: usize) -> Self {
        let mut out = Self::constant(value, n);
        out.gradient[dof] = 1.0;
        out
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.gradient.len()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value + rhs.value, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] + rhs.gradient[i];
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] + rhs.hessian[i];
        }
        for i in 0..n * n * n {
            out.third[i] = self.third[i] + rhs.third[i];
        }
        out
    }

    pub fn add_scalar(&self, rhs: f64) -> Self {
        let mut out = self.clone();
        out.value += rhs;
        out
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    pub fn scale(&self, s: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value * s, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] * s;
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] * s;
        }
        for i in 0..n * n * n {
            out.third[i] = self.third[i] * s;
        }
        out
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let n = self.n();
        let (a, b) = (self, rhs);
        let mut out = Self::constant(a.value * b.value, n);
        for i in 0..n {
            out.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] = a.hessian[i * n + j] * b.value
                    + a.value * b.hessian[i * n + j]
                    + a.gradient[i] * b.gradient[j]
                    + a.gradient[j] * b.gradient[i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = (i * n + j) * n + k;
                    out.third[idx] = a.third[idx] * b.value
                        + a.value * b.third[idx]
                        + a.hessian[i * n + j] * b.gradient[k]
                        + a.hessian[i * n + k] * b.gradient[j]
                        + a.hessian[j * n + k] * b.gradient[i]
                        + a.gradient[i] * b.hessian[j * n + k]
                        + a.gradient[j] * b.hessian[i * n + k]
                        + a.gradient[k] * b.hessian[i * n + j];
                }
            }
        }
        out
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    /// Faà di Bruno composition given the outer scalar derivatives `φ1..φ3`.
    pub fn compose(&self, value: f64, phi1: f64, phi2: f64, phi3: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(value, n);
        for i in 0..n {
            out.gradient[i] = phi1 * self.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] =
                    phi1 * self.hessian[i * n + j] + phi2 * self.gradient[i] * self.gradient[j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = (i * n + j) * n + k;
                    out.third[idx] = phi3 * self.gradient[i] * self.gradient[j] * self.gradient[k]
                        + phi2
                            * (self.hessian[i * n + j] * self.gradient[k]
                                + self.hessian[i * n + k] * self.gradient[j]
                                + self.hessian[j * n + k] * self.gradient[i])
                        + phi1 * self.third[idx];
                }
            }
        }
        out
    }

    pub fn powf(&self, p: f64) -> Self {
        let v = self.value;
        self.compose(
            v.powf(p),
            p * v.powf(p - 1.0),
            p * (p - 1.0) * v.powf(p - 2.0),
            p * (p - 1.0) * (p - 2.0) * v.powf(p - 3.0),
        )
    }

    pub fn exp(&self) -> Self {
        let e = self.value.exp();
        self.compose(e, e, e, e)
    }

    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

#[derive(Clone, Debug)]
pub struct Jet4 {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
    pub third: Vec<f64>,
    pub fourth: Vec<f64>,
}

impl Jet4 {
    pub fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; n],
            hessian: vec![0.0; n * n],
            third: vec![0.0; n * n * n],
            fourth: vec![0.0; n * n * n * n],
        }
    }

    pub fn variable(value: f64, n: usize, dof: usize) -> Self {
        let mut out = Self::constant(value, n);
        out.gradient[dof] = 1.0;
        out
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.gradient.len()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value + rhs.value, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] + rhs.gradient[i];
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] + rhs.hessian[i];
        }
        for i in 0..n * n * n {
            out.third[i] = self.third[i] + rhs.third[i];
        }
        for i in 0..n * n * n * n {
            out.fourth[i] = self.fourth[i] + rhs.fourth[i];
        }
        out
    }

    pub fn add_scalar(&self, rhs: f64) -> Self {
        let mut out = self.clone();
        out.value += rhs;
        out
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    pub fn scale(&self, s: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(self.value * s, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] * s;
        }
        for i in 0..n * n {
            out.hessian[i] = self.hessian[i] * s;
        }
        for i in 0..n * n * n {
            out.third[i] = self.third[i] * s;
        }
        for i in 0..n * n * n * n {
            out.fourth[i] = self.fourth[i] * s;
        }
        out
    }

    /// Fourth-order Leibniz product. Index partitions of `{i,j,k,l}`:
    /// `4+0`, `3+1` (4 pairings), `2+2` (3 pairings, both operand orders),
    /// `1+3` (4 pairings), `0+4`.
    pub fn mul(&self, rhs: &Self) -> Self {
        let n = self.n();
        let (a, b) = (self, rhs);
        let mut out = Self::constant(a.value * b.value, n);
        for i in 0..n {
            out.gradient[i] = a.gradient[i] * b.value + a.value * b.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] = a.hessian[i * n + j] * b.value
                    + a.value * b.hessian[i * n + j]
                    + a.gradient[i] * b.gradient[j]
                    + a.gradient[j] * b.gradient[i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = (i * n + j) * n + k;
                    out.third[idx] = a.third[idx] * b.value
                        + a.value * b.third[idx]
                        + a.hessian[i * n + j] * b.gradient[k]
                        + a.hessian[i * n + k] * b.gradient[j]
                        + a.hessian[j * n + k] * b.gradient[i]
                        + a.gradient[i] * b.hessian[j * n + k]
                        + a.gradient[j] * b.hessian[i * n + k]
                        + a.gradient[k] * b.hessian[i * n + j];
                }
            }
        }
        let h = |m: &Self, p: usize, q: usize| m.hessian[p * n + q];
        let t = |m: &Self, p: usize, q: usize, r: usize| m.third[(p * n + q) * n + r];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let idx = ((i * n + j) * n + k) * n + l;
                        out.fourth[idx] = a.fourth[idx] * b.value
                            + a.value * b.fourth[idx]
                            // 3+1 partitions
                            + t(a, i, j, k) * b.gradient[l]
                            + t(a, i, j, l) * b.gradient[k]
                            + t(a, i, k, l) * b.gradient[j]
                            + t(a, j, k, l) * b.gradient[i]
                            + t(b, i, j, k) * a.gradient[l]
                            + t(b, i, j, l) * a.gradient[k]
                            + t(b, i, k, l) * a.gradient[j]
                            + t(b, j, k, l) * a.gradient[i]
                            // 2+2 partitions (both operand orders)
                            + h(a, i, j) * h(b, k, l)
                            + h(a, i, k) * h(b, j, l)
                            + h(a, i, l) * h(b, j, k)
                            + h(b, i, j) * h(a, k, l)
                            + h(b, i, k) * h(a, j, l)
                            + h(b, i, l) * h(a, j, k);
                    }
                }
            }
        }
        out
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    /// Faà di Bruno composition given the outer scalar derivatives `φ1..φ4`.
    /// Partitions of `{i,j,k,l}`: one 4-block (φ1·u⁗), `3+1` (4 terms) and
    /// `2+2` (3 terms) at φ2, `2+1+1` (6 terms) at φ3, and `1+1+1+1` at φ4.
    pub fn compose(&self, value: f64, phi1: f64, phi2: f64, phi3: f64, phi4: f64) -> Self {
        let n = self.n();
        let mut out = Self::constant(value, n);
        for i in 0..n {
            out.gradient[i] = phi1 * self.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] =
                    phi1 * self.hessian[i * n + j] + phi2 * self.gradient[i] * self.gradient[j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = (i * n + j) * n + k;
                    out.third[idx] = phi3 * self.gradient[i] * self.gradient[j] * self.gradient[k]
                        + phi2
                            * (self.hessian[i * n + j] * self.gradient[k]
                                + self.hessian[i * n + k] * self.gradient[j]
                                + self.hessian[j * n + k] * self.gradient[i])
                        + phi1 * self.third[idx];
                }
            }
        }
        let g = &self.gradient;
        let h = |p: usize, q: usize| self.hessian[p * n + q];
        let t = |p: usize, q: usize, r: usize| self.third[(p * n + q) * n + r];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let idx = ((i * n + j) * n + k) * n + l;
                        out.fourth[idx] = phi1 * self.fourth[idx]
                            + phi2
                                * (t(i, j, k) * g[l]
                                    + t(i, j, l) * g[k]
                                    + t(i, k, l) * g[j]
                                    + t(j, k, l) * g[i]
                                    + h(i, j) * h(k, l)
                                    + h(i, k) * h(j, l)
                                    + h(i, l) * h(j, k))
                            + phi3
                                * (h(i, j) * g[k] * g[l]
                                    + h(i, k) * g[j] * g[l]
                                    + h(i, l) * g[j] * g[k]
                                    + h(j, k) * g[i] * g[l]
                                    + h(j, l) * g[i] * g[k]
                                    + h(k, l) * g[i] * g[j])
                            + phi4 * g[i] * g[j] * g[k] * g[l];
                    }
                }
            }
        }
        out
    }

    pub fn powf(&self, p: f64) -> Self {
        let v = self.value;
        self.compose(
            v.powf(p),
            p * v.powf(p - 1.0),
            p * (p - 1.0) * v.powf(p - 2.0),
            p * (p - 1.0) * (p - 2.0) * v.powf(p - 3.0),
            p * (p - 1.0) * (p - 2.0) * (p - 3.0) * v.powf(p - 4.0),
        )
    }

    pub fn exp(&self) -> Self {
        let e = self.value.exp();
        self.compose(e, e, e, e, e)
    }

    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative smooth 2-variable test function built from the shared
    /// op set: f(u, v) = exp(u) · (u + 2v + 3)^(−2) + u·v.
    fn f_jet4(u: f64, v: f64) -> Jet4 {
        let ju = Jet4::variable(u, 2, 0);
        let jv = Jet4::variable(v, 2, 1);
        let base = ju.add(&jv.scale(2.0)).add_scalar(3.0);
        ju.exp().mul(&base.powf(-2.0)).add(&ju.mul(&jv))
    }

    fn f_jet3(u: f64, v: f64) -> Jet3 {
        let ju = Jet3::variable(u, 2, 0);
        let jv = Jet3::variable(v, 2, 1);
        let base = ju.add(&jv.scale(2.0)).add_scalar(3.0);
        ju.exp().mul(&base.powf(-2.0)).add(&ju.mul(&jv))
    }

    #[test]
    fn jet4_lower_orders_match_jet3() {
        let (u, v) = (0.31, -0.12);
        let j4 = f_jet4(u, v);
        let j3 = f_jet3(u, v);
        assert!((j4.value - j3.value).abs() < 1.0e-14);
        for i in 0..2 {
            assert!((j4.gradient[i] - j3.gradient[i]).abs() < 1.0e-14);
        }
        for i in 0..4 {
            assert!((j4.hessian[i] - j3.hessian[i]).abs() < 1.0e-14);
        }
        for i in 0..8 {
            assert!((j4.third[i] - j3.third[i]).abs() < 1.0e-14);
        }
    }

    /// The fourth derivative must equal the central FD of the Jet3 third
    /// derivative along each seed variable.
    #[test]
    fn jet4_fourth_matches_fd_of_jet3_third() {
        let (u, v) = (0.31, -0.12);
        let h = 1.0e-5;
        let j4 = f_jet4(u, v);
        let n = 2;
        for l in 0..n {
            let (up, vp) = if l == 0 { (u + h, v) } else { (u, v + h) };
            let (um, vm) = if l == 0 { (u - h, v) } else { (u, v - h) };
            let tp = f_jet3(up, vp).third;
            let tm = f_jet3(um, vm).third;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let idx3 = (i * n + j) * n + k;
                        let fd = (tp[idx3] - tm[idx3]) / (2.0 * h);
                        let idx4 = ((i * n + j) * n + k) * n + l;
                        assert!(
                            (j4.fourth[idx4] - fd).abs() < 1.0e-6,
                            "fourth[{i}{j}{k}{l}] = {} vs FD {}",
                            j4.fourth[idx4],
                            fd
                        );
                    }
                }
            }
        }
    }

    /// The packed fourth array must be symmetric under all index permutations.
    #[test]
    fn jet4_fourth_is_permutation_symmetric() {
        let j4 = f_jet4(0.31, -0.12);
        let n = 2;
        let idx = |i: usize, j: usize, k: usize, l: usize| ((i * n + j) * n + k) * n + l;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let v = j4.fourth[idx(i, j, k, l)];
                        for &(a, b, c, d) in &[
                            (j, i, k, l),
                            (i, k, j, l),
                            (i, j, l, k),
                            (l, k, j, i),
                            (k, l, i, j),
                        ] {
                            assert!((v - j4.fourth[idx(a, b, c, d)]).abs() < 1.0e-12);
                        }
                    }
                }
            }
        }
    }

    /// div = mul with powf(-1): validate the quotient's fourth derivative
    /// against FD of Jet3 as an independent path through the op set.
    #[test]
    fn jet4_div_matches_fd() {
        let g4 = |u: f64, v: f64| -> Jet4 {
            let ju = Jet4::variable(u, 2, 0);
            let jv = Jet4::variable(v, 2, 1);
            ju.exp().div(&ju.mul(&jv).add_scalar(4.0))
        };
        let g3 = |u: f64, v: f64| -> Jet3 {
            let ju = Jet3::variable(u, 2, 0);
            let jv = Jet3::variable(v, 2, 1);
            ju.exp().div(&ju.mul(&jv).add_scalar(4.0))
        };
        let (u, v) = (0.42, 0.17);
        let h = 1.0e-5;
        let j4 = g4(u, v);
        let n = 2;
        for l in 0..n {
            let (up, vp) = if l == 0 { (u + h, v) } else { (u, v + h) };
            let (um, vm) = if l == 0 { (u - h, v) } else { (u, v - h) };
            let tp = g3(up, vp).third;
            let tm = g3(um, vm).third;
            for idx3 in 0..n * n * n {
                let fd = (tp[idx3] - tm[idx3]) / (2.0 * h);
                let idx4 = idx3 * n + l;
                assert!(
                    (j4.fourth[idx4] - fd).abs() < 1.0e-6,
                    "div fourth[{idx4}] = {} vs FD {}",
                    j4.fourth[idx4],
                    fd
                );
            }
        }
    }
}
