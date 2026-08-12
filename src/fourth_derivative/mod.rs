// SPDX-License-Identifier: GPL-3.0-or-later
//! Quartic force constants: storage and assembly of the analytic fourth
//! nuclear derivative `Q_abcd = ∂⁴E/∂R_a∂R_b∂R_c∂R_d`.
//!
//! Three layers:
//!
//! * [`SymmetricFourth`] — packed storage for fully-symmetric rank-4 tensors,
//!   mirroring [`crate::third_derivative::SymmetricThird`] one order up, with
//!   the contracted / block accessors that are the production interface for
//!   large systems;
//! * [`directional_fourth_derivative`] — the five-stage 2n+1 DIRECTIONAL
//!   quartic `e⁗[v] = Q·vvvv`, needing only the first- and second-order
//!   responses along `v` (see [`assemble`]);
//! * [`fourth_derivative_analytic_dense`] / [`fourth_derivative_analytic_block`]
//!   — the full MIXED-INDEX tensor, recovered element by element from the
//!   directional quartic by the polarization identity for symmetric quartic
//!   forms, all directions sharing one [`QuarticReference`] (SCF + CPXTB +
//!   charge-space context).

pub mod assemble;
pub mod directional;
pub mod response_stage;

pub use assemble::{
    directional_fourth_derivative, directional_fourth_seminumerical,
    directional_fourth_with_reference, fourth_derivative_analytic_block,
    fourth_derivative_analytic_dense, QuarticReference,
};

use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::third_derivative::SymmetricThird;

/// Packed storage for a fully index-symmetric fourth-derivative tensor over
/// `n` degrees of freedom: `n(n+1)(n+2)(n+3)/24` entries, one per unordered
/// index quadruple. Canonical index for sorted `a ≤ b ≤ c ≤ d`:
/// `C(d+3,4) + C(c+2,3) + C(b+1,2) + a`.
///
/// Memory: dense equivalents are `O(n⁴)`; the packed store is ~24× smaller but
/// still grows quartically (300 DOF ≈ 2.8 GB dense, ≈ 120 MB packed), so the
/// contracted/block accessors are the production interface for large systems.
#[derive(Clone, Debug)]
pub struct SymmetricFourth {
    n: usize,
    data: Vec<f64>,
}

#[inline]
fn c2(x: usize) -> usize {
    x * (x + 1) / 2
}

#[inline]
fn c3(x: usize) -> usize {
    x * (x + 1) * (x + 2) / 6
}

#[inline]
fn c4(x: usize) -> usize {
    x * (x + 1) * (x + 2) * (x + 3) / 24
}

impl SymmetricFourth {
    pub fn zeros(n: usize) -> Self {
        // Packed length C(n+3, 4) = n(n+1)(n+2)(n+3)/24 ≡ c4(n).
        Self {
            n,
            data: vec![0.0; c4(n)],
        }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Canonical packed index of the unordered quadruple `{a, b, c, d}`.
    #[inline]
    pub fn index(&self, a: usize, b: usize, c: usize, d: usize) -> usize {
        debug_assert!(a < self.n && b < self.n && c < self.n && d < self.n);
        // Sort the four indices ascending (sorting network, branch-light).
        let (mut w, mut x, mut y, mut z) = (a, b, c, d);
        if w > x {
            core::mem::swap(&mut w, &mut x);
        }
        if y > z {
            core::mem::swap(&mut y, &mut z);
        }
        if w > y {
            core::mem::swap(&mut w, &mut y);
        }
        if x > z {
            core::mem::swap(&mut x, &mut z);
        }
        if x > y {
            core::mem::swap(&mut x, &mut y);
        }
        c4(z) + c3(y) + c2(x) + w
    }

    #[inline]
    pub fn add(&mut self, a: usize, b: usize, c: usize, d: usize, value: f64) {
        let idx = self.index(a, b, c, d);
        self.data[idx] += value;
    }

    #[inline]
    pub fn get(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.data[self.index(a, b, c, d)]
    }

    pub fn scale(&mut self, s: f64) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn add_from(&mut self, other: &Self) -> Result<()> {
        if self.n != other.n {
            return Err(Gfn1Error::InvalidInput(format!(
                "SymmetricFourth size mismatch: {} vs {}",
                self.n, other.n
            )));
        }
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            *dst += *src;
        }
        Ok(())
    }

    /// Contract the last index with `v`: `T_abc = Σ_d Q_abcd v_d`, returning a
    /// packed [`SymmetricThird`] (exact, since `Q` is fully symmetric).
    pub fn contract_last(&self, v: &[f64]) -> Result<SymmetricThird> {
        if v.len() != self.n {
            return Err(Gfn1Error::InvalidInput(format!(
                "contract_last: direction length {} != n {}",
                v.len(),
                self.n
            )));
        }
        let mut out = SymmetricThird::zeros(self.n);
        for c in 0..self.n {
            for b in 0..=c {
                for a in 0..=b {
                    let mut acc = 0.0;
                    for (d, &vd) in v.iter().enumerate() {
                        acc += self.get(a, b, c, d) * vd;
                    }
                    out.add(a, b, c, acc);
                }
            }
        }
        Ok(out)
    }

    /// Contract the last two indices: `M_ab = Σ_cd Q_abcd v_c w_d`.
    pub fn contract_last2(&self, v: &[f64], w: &[f64]) -> Result<Matrix> {
        if v.len() != self.n || w.len() != self.n {
            return Err(Gfn1Error::InvalidInput(format!(
                "contract_last2: direction lengths {}/{} != n {}",
                v.len(),
                w.len(),
                self.n
            )));
        }
        let mut out = Matrix::zeros(self.n, self.n);
        for a in 0..self.n {
            for b in a..self.n {
                let mut acc = 0.0;
                for (c, &vc) in v.iter().enumerate() {
                    if vc == 0.0 {
                        continue;
                    }
                    for (d, &wd) in w.iter().enumerate() {
                        acc += self.get(a, b, c, d) * vc * wd;
                    }
                }
                out[(a, b)] = acc;
                out[(b, a)] = acc;
            }
        }
        Ok(out)
    }

    /// Full contraction `Σ_abcd Q_abcd v_a v_b v_c v_d`.
    pub fn contract_vvvv(&self, v: &[f64]) -> Result<f64> {
        if v.len() != self.n {
            return Err(Gfn1Error::InvalidInput(format!(
                "contract_vvvv: direction length {} != n {}",
                v.len(),
                self.n
            )));
        }
        let m = self.contract_last2(v, v)?;
        let mut acc = 0.0;
        for a in 0..self.n {
            for b in 0..self.n {
                acc += m[(a, b)] * v[a] * v[b];
            }
        }
        Ok(acc)
    }

    /// The `|dofs|⁴` sub-tensor restricted to `dofs`, as packed slabs
    /// `slabs[ci][di ≥ ci]`-style dense matrices: returns `(dofs, slabs)` with
    /// `slabs[ci * |dofs| + di][(ai, bi)] = Q[dofs[ai], dofs[bi], dofs[ci], dofs[di]]`.
    pub fn block(&self, dofs: &[usize]) -> Result<Vec<Matrix>> {
        for &d in dofs {
            if d >= self.n {
                return Err(Gfn1Error::InvalidInput(format!(
                    "block: dof {d} out of range (n = {})",
                    self.n
                )));
            }
        }
        let m = dofs.len();
        let mut out = Vec::with_capacity(m * m);
        for &c in dofs {
            for &d in dofs {
                let mut slab = Matrix::zeros(m, m);
                for (ai, &a) in dofs.iter().enumerate() {
                    for (bi, &b) in dofs.iter().enumerate() {
                        slab[(ai, bi)] = self.get(a, b, c, d);
                    }
                }
                out.push(slab);
            }
        }
        Ok(out)
    }
}

/// Add the rank-4 central block of a radial pair function `f(r)` to a packed
/// [`SymmetricFourth`] store -- the fourth-order analogue of
/// [`crate::third_derivative::add_radial_third_block_sym`].
///
/// `rel = R_i − R_j` is the true relative vector (**not** normalised) and `c2`, `c3`, `c4`
/// are the "hat" derivatives `D̂^k f` with `D̂ = (1/r) d/dr`,
///
/// ```text
///   c2 = f''/r² − f'/r³
///   c3 = f'''/r³ − 3f''/r⁴ + 3f'/r⁵
///   c4 = f''''/r⁴ − 6f'''/r⁵ + 15f''/r⁶ − 15f'/r⁷
/// ```
///
/// so that, writing `u = rel`, the relative-vector fourth derivative
/// `∂⁴f/∂u_a∂u_b∂u_c∂u_d` is
///
/// ```text
///   T_abcd = c4 u_a u_b u_c u_d
///          + c3 (δ_ab u_c u_d + δ_ac u_b u_d + δ_ad u_b u_c
///                + δ_bc u_a u_d + δ_bd u_a u_c + δ_cd u_a u_b)
///          + c2 (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc).
/// ```
///
/// DOF `(atom_i, α)` carries `+∂/∂u_α` and `(atom_j, α)` carries `−∂/∂u_α`, so the block
/// for an assignment placing `m` of the four indices on `atom_j` picks up sign `(−1)^m`;
/// all `2⁴` assignments are visited. `scale` multiplies the whole block (e.g. `q_i q_j`
/// for shell-charge electrostatics; `1` for repulsion). As in the third-order routine each
/// unordered DOF quadruple is written **once** (its sorted representative), so
/// `store.get(a, b, c, d)` equals the dense `∂⁴E/∂R_a∂R_b∂R_c∂R_d`.
pub(crate) fn add_radial_fourth_block_sym(
    store: &mut SymmetricFourth,
    atom_i: usize,
    atom_j: usize,
    rel: Vec3,
    c2: f64,
    c3: f64,
    c4: f64,
    scale: f64,
) {
    let r = rel.norm();
    if r <= 1.0e-12 || scale == 0.0 {
        return;
    }
    let u = rel.to_array();
    let atoms = [atom_i, atom_j];
    let signs = [1.0_f64, -1.0_f64];
    let delta = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let (dab, dac, dad) = (delta(a, b), delta(a, c), delta(a, d));
                    let (dbc, dbd, dcd) = (delta(b, c), delta(b, d), delta(c, d));
                    let t_rel = c4 * u[a] * u[b] * u[c] * u[d]
                        + c3 * (dab * u[c] * u[d]
                            + dac * u[b] * u[d]
                            + dad * u[b] * u[c]
                            + dbc * u[a] * u[d]
                            + dbd * u[a] * u[c]
                            + dcd * u[a] * u[b])
                        + c2 * (dab * dcd + dac * dbd + dad * dbc);
                    let value = scale * t_rel;
                    if value == 0.0 {
                        continue;
                    }
                    for (xi, &ax) in atoms.iter().enumerate() {
                        for (yi, &ay) in atoms.iter().enumerate() {
                            for (zi, &az) in atoms.iter().enumerate() {
                                for (wi, &aw) in atoms.iter().enumerate() {
                                    let (i1, i2, i3, i4) =
                                        (3 * ax + a, 3 * ay + b, 3 * az + c, 3 * aw + d);
                                    // One write per unordered quadruple (sorted
                                    // representative), exploiting the full index symmetry.
                                    if i1 <= i2 && i2 <= i3 && i3 <= i4 {
                                        let sign = signs[xi] * signs[yi] * signs[zi] * signs[wi];
                                        store.add(i1, i2, i3, i4, sign * value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_len_matches_binomial() {
        for n in 0..8 {
            let s = SymmetricFourth::zeros(n);
            let expect = if n == 0 {
                0
            } else {
                n * (n + 1) * (n + 2) * (n + 3) / 24
            };
            assert_eq!(s.len(), expect, "n = {n}");
        }
    }

    #[test]
    fn canonical_indices_are_dense_and_unique() {
        let n = 6;
        let s = SymmetricFourth::zeros(n);
        let mut seen = vec![false; s.len()];
        for d in 0..n {
            for c in 0..=d {
                for b in 0..=c {
                    for a in 0..=b {
                        let idx = s.index(a, b, c, d);
                        assert!(!seen[idx], "duplicate index for ({a},{b},{c},{d})");
                        seen[idx] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&x| x), "index map is not surjective");
    }

    #[test]
    fn index_is_permutation_invariant_and_addition_symmetric() {
        let n = 5;
        let mut s = SymmetricFourth::zeros(n);
        s.add(3, 1, 4, 2, 2.5);
        for &(a, b, c, d) in &[
            (1usize, 2usize, 3usize, 4usize),
            (4, 3, 2, 1),
            (2, 4, 1, 3),
            (3, 1, 4, 2),
        ] {
            assert_eq!(s.get(a, b, c, d), 2.5);
        }
        s.add(2, 2, 2, 2, -1.0);
        assert_eq!(s.get(2, 2, 2, 2), -1.0);
        assert_eq!(s.get(1, 2, 3, 4), 2.5);
    }

    /// Brute-force reference: fill a dense symmetric tensor and compare every
    /// contraction path.
    #[test]
    fn contractions_match_dense_reference() {
        let n = 4;
        let mut s = SymmetricFourth::zeros(n);
        // Symmetric-by-construction values from a symmetric generator.
        let gen = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            let (a, b, c, d) = (a as f64, b as f64, c as f64, d as f64);
            (a + b + c + d) + 0.1 * (a * b * c * d) + 0.01 * (a * a + b * b + c * c + d * d)
        };
        for d in 0..n {
            for c in 0..=d {
                for b in 0..=c {
                    for a in 0..=b {
                        let idx = s.index(a, b, c, d);
                        s.data[idx] = gen(a, b, c, d);
                    }
                }
            }
        }
        let v: Vec<f64> = (0..n).map(|k| 0.3 + 0.2 * k as f64).collect();
        let w: Vec<f64> = (0..n).map(|k| 1.0 - 0.15 * k as f64).collect();

        let t3 = s.contract_last(&v).unwrap();
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let mut want = 0.0;
                    for d in 0..n {
                        want += s.get(a, b, c, d) * v[d];
                    }
                    assert!((t3.get(a, b, c) - want).abs() < 1.0e-12);
                }
            }
        }

        let m = s.contract_last2(&v, &w).unwrap();
        for a in 0..n {
            for b in 0..n {
                let mut want = 0.0;
                for c in 0..n {
                    for d in 0..n {
                        want += s.get(a, b, c, d) * v[c] * w[d];
                    }
                }
                assert!((m[(a, b)] - want).abs() < 1.0e-12);
            }
        }

        let full = s.contract_vvvv(&v).unwrap();
        let mut want = 0.0;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        want += s.get(a, b, c, d) * v[a] * v[b] * v[c] * v[d];
                    }
                }
            }
        }
        assert!((full - want).abs() < 1.0e-10);

        let dofs = [1usize, 3];
        let blocks = s.block(&dofs).unwrap();
        for (ci, &c) in dofs.iter().enumerate() {
            for (di, &d) in dofs.iter().enumerate() {
                let slab = &blocks[ci * dofs.len() + di];
                for (ai, &a) in dofs.iter().enumerate() {
                    for (bi, &b) in dofs.iter().enumerate() {
                        assert_eq!(slab[(ai, bi)], s.get(a, b, c, d));
                    }
                }
            }
        }
    }
}
