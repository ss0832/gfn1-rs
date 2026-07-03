// SPDX-License-Identifier: GPL-3.0-or-later
//! k-point sampling for periodic GFN1-xTB.
//!
//! The Bloch phase factor for an atomic-orbital image at integer lattice offset
//! `n = [n1, n2, n3]` and a k-point with fractional coordinates `f` is
//! `exp(i k . T) = exp(i 2pi (f1 n1 + f2 n2 + f3 n3))`, because the reciprocal
//! and direct bases satisfy `b_i . a_j = 2pi delta_ij`. The phase therefore
//! needs only the fractional k and the integer image offset, never the
//! Cartesian reciprocal vectors.

use crate::lattice::ImageOffset;
use crate::model::KPoint;
use std::f64::consts::PI;

/// Single Gamma point (`k = 0`), weight 1.
pub fn gamma_only() -> Vec<KPoint> {
    vec![KPoint::gamma()]
}

/// Monkhorst-Pack mesh with uniform weights. Non-periodic axes are collapsed to
/// a single `k = 0`. When `gamma_centered` is true the grid includes the Gamma
/// point; otherwise the standard MP offset `(2r - n + 1)/(2n)` is used.
///
/// Weights are uniform `1/Nk` over the full (unreduced) mesh, which is exact;
/// time-reversal folding is an optional optimization layered on top.
pub fn monkhorst_pack(mesh: [usize; 3], periodic: [bool; 3], gamma_centered: bool) -> Vec<KPoint> {
    let m = [
        if periodic[0] { mesh[0].max(1) } else { 1 },
        if periodic[1] { mesh[1].max(1) } else { 1 },
        if periodic[2] { mesh[2].max(1) } else { 1 },
    ];
    let total = m[0] * m[1] * m[2];
    let weight = 1.0 / total as f64;
    let mut points = Vec::with_capacity(total);
    for a in 0..m[0] {
        for b in 0..m[1] {
            for c in 0..m[2] {
                points.push(KPoint {
                    fractional: [
                        mp_coord(a, m[0], gamma_centered),
                        mp_coord(b, m[1], gamma_centered),
                        mp_coord(c, m[2], gamma_centered),
                    ],
                    weight,
                });
            }
        }
    }
    points
}

/// Fold a full mesh by time-reversal symmetry (`k` and `-k` are equivalent for a
/// real Hamiltonian), accumulating weights of merged points. Reduces the number
/// of diagonalizations by roughly a factor of two without changing any result.
pub fn fold_time_reversal(points: &[KPoint]) -> Vec<KPoint> {
    let mut reduced: Vec<KPoint> = Vec::new();
    'outer: for kp in points {
        for existing in &mut reduced {
            if is_negative(existing.fractional, kp.fractional)
                || is_equal(existing.fractional, kp.fractional)
            {
                existing.weight += kp.weight;
                continue 'outer;
            }
        }
        reduced.push(*kp);
    }
    reduced
}

fn mp_coord(r: usize, n: usize, gamma_centered: bool) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let raw = if gamma_centered {
        r as f64 / n as f64
    } else {
        (2 * r as i64 - n as i64 + 1) as f64 / (2.0 * n as f64)
    };
    // Wrap into (-1/2, 1/2].
    let wrapped = raw - (raw + 0.5).floor();
    if wrapped <= -0.5 {
        wrapped + 1.0
    } else {
        wrapped
    }
}

fn is_equal(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| frac_close(a[i], b[i]))
}

fn is_negative(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| frac_close(a[i], -b[i]))
}

fn frac_close(a: f64, b: f64) -> bool {
    let d = a - b;
    let wrapped = d - d.round();
    wrapped.abs() < 1.0e-9
}

/// Bloch phase `(cos, sin)` of `exp(i 2pi f . n)` for a fractional k-point `f`
/// and integer image offset `n`.
#[inline]
pub fn bloch_phase(fractional: [f64; 3], offset: ImageOffset) -> (f64, f64) {
    let theta = 2.0
        * PI
        * (fractional[0] * offset.n[0] as f64
            + fractional[1] * offset.n[1] as f64
            + fractional[2] * offset.n[2] as f64);
    (theta.cos(), theta.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_mesh_is_single_point() {
        let pts = monkhorst_pack([1, 1, 1], [true, true, true], true);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].fractional, [0.0, 0.0, 0.0]);
        assert!((pts[0].weight - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn non_periodic_axis_collapses() {
        let pts = monkhorst_pack([4, 4, 4], [true, true, false], false);
        assert_eq!(pts.len(), 16);
        assert!(pts.iter().all(|p| p.fractional[2] == 0.0));
        let wsum: f64 = pts.iter().map(|p| p.weight).sum();
        assert!((wsum - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn gamma_phase_is_unity() {
        let (c, s) = bloch_phase([0.0, 0.0, 0.0], ImageOffset { n: [3, -2, 1] });
        assert!((c - 1.0).abs() < 1.0e-12);
        assert!(s.abs() < 1.0e-12);
    }

    #[test]
    fn time_reversal_folding_conserves_weight() {
        let full = monkhorst_pack([4, 4, 4], [true, true, true], false);
        let folded = fold_time_reversal(&full);
        assert!(folded.len() < full.len());
        let wsum: f64 = folded.iter().map(|p| p.weight).sum();
        assert!((wsum - 1.0).abs() < 1.0e-12);
    }
}
