// SPDX-License-Identifier: GPL-3.0-or-later
//! Complex Hermitian linear algebra for k-point GFN1-xTB.
//!
//! The k-resolved generalized eigenproblem `H(k) C = S(k) C eps` with Hermitian
//! `H(k)` and Hermitian positive-definite `S(k)` is solved through a real
//! `2n x 2n` embedding so that the crate's existing (tested) real Lowdin
//! eigensolver can be reused, instead of pulling in a complex eigensolver.
//!
//! For a Hermitian matrix `H = Hr + i Hi` (`Hr` symmetric, `Hi` antisymmetric)
//! the embedding
//!
//! ```text
//!   M = [[ Hr, -Hi ],
//!        [ Hi,  Hr ]]
//! ```
//!
//! is real symmetric and has every eigenvalue of `H`, each appearing twice.
//! A complex eigenvector `c = u + i v` is encoded by the real pair `[u; v]`
//! and its phase-rotated partner `[-v; u]`. Density matrices are recovered by
//! block extraction, which is invariant to how the doubly-degenerate real
//! eigenvectors mix and therefore robust.

use crate::error::Result;
use crate::linalg::{
    column_weighted_gram, lowdin_orthogonalizer, lowdin_solve_generalized,
    lowdin_solve_with_orthogonalizer, LowdinOrthogonalizer, Matrix,
};

/// Dense complex matrix stored as separate real and imaginary parts.
#[derive(Clone, Debug)]
pub struct CMatrix {
    pub n: usize,
    pub re: Matrix,
    pub im: Matrix,
}

impl CMatrix {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            re: Matrix::zeros(n, n),
            im: Matrix::zeros(n, n),
        }
    }

    #[inline]
    pub fn accumulate(&mut self, i: usize, j: usize, re: f64, im: f64) {
        self.re[(i, j)] += re;
        self.im[(i, j)] += im;
    }

    /// Project onto the Hermitian part `H <- (H + H^H)/2`. Cleans up the tiny
    /// asymmetries left by finite-precision Bloch accumulation.
    pub fn hermitianize(&mut self) {
        let n = self.n;
        for i in 0..n {
            for j in 0..=i {
                let re = 0.5 * (self.re[(i, j)] + self.re[(j, i)]);
                let im = 0.5 * (self.im[(i, j)] - self.im[(j, i)]);
                self.re[(i, j)] = re;
                self.re[(j, i)] = re;
                self.im[(i, j)] = im;
                self.im[(j, i)] = -im;
            }
            self.im[(i, i)] = 0.0;
        }
    }

    /// Real part of `tr(self * rhs)` for two `n x n` complex matrices.
    pub fn real_trace_product(&self, rhs: &CMatrix) -> f64 {
        let n = self.n;
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                // Re[ sum_j A_ij B_ji ] = sum_ij (Ar_ij Br_ji - Ai_ij Bi_ji)
                acc += self.re[(i, j)] * rhs.re[(j, i)] - self.im[(i, j)] * rhs.im[(j, i)];
            }
        }
        acc
    }
}

/// Real-embedding solution of the k-point eigenproblem.
///
/// `values` holds the `2n` ascending eigenvalues (each physical band twice) and
/// `vectors` the `2n x 2n` real eigenvectors (columns), orthonormal with respect
/// to the embedded overlap. Single-electron occupations of length `2n` recover
/// max-2 band occupations after block extraction.
#[derive(Clone, Debug)]
pub struct KEigen {
    pub values: Vec<f64>,
    pub vectors: Matrix,
    pub n: usize,
}

/// Solve `H(k) C = S(k) C eps` for Hermitian `H` and Hermitian PD `S`.
pub fn hermitian_generalized_eigen(h: &CMatrix, s: &CMatrix, tol: f64) -> Result<KEigen> {
    let n = h.n;
    let m = real_embedding(&h.re, &h.im);
    let nmat = real_embedding(&s.re, &s.im);
    let eig = lowdin_solve_generalized(&m, &nmat, tol)?;
    Ok(KEigen {
        values: eig.values,
        vectors: eig.vectors,
        n,
    })
}

/// Löwdin (symmetric) orthogonaliser `X = S(k)^{-1/2}` for the real `2n x 2n`
/// embedding of the Hermitian PD overlap `S(k)`.
///
/// `S(k)` is geometry-fixed and does not change across SCC iterations, so this
/// factorisation can be built once per k-point and reused in every iteration's
/// eigensolve via [`hermitian_generalized_eigen_with_orthogonalizer`]. This is a
/// pure caching helper: the orthogonaliser it returns is bit-for-bit the one
/// [`hermitian_generalized_eigen`] would have rebuilt internally.
pub fn embedded_overlap_orthogonalizer(s: &CMatrix, tol: f64) -> Result<LowdinOrthogonalizer> {
    let nmat = real_embedding(&s.re, &s.im);
    lowdin_orthogonalizer(&nmat, tol)
}

/// Solve `H(k) C = S(k) C eps` reusing a precomputed orthogonaliser
/// `X = S(k)^{-1/2}` (see [`embedded_overlap_orthogonalizer`]).
///
/// Numerically identical to [`hermitian_generalized_eigen`]; it only skips the
/// per-call rebuild of the `S(k)`-only Löwdin factorisation.
pub fn hermitian_generalized_eigen_with_orthogonalizer(
    h: &CMatrix,
    orth: &LowdinOrthogonalizer,
    n: usize,
    tol: f64,
) -> Result<KEigen> {
    let m = real_embedding(&h.re, &h.im);
    let eig = lowdin_solve_with_orthogonalizer(&m, orth, tol)?;
    Ok(KEigen {
        values: eig.values,
        vectors: eig.vectors,
        n,
    })
}

fn real_embedding(re: &Matrix, im: &Matrix) -> Matrix {
    let n = re.rows();
    let mut out = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let r = re[(i, j)];
            let m = im[(i, j)];
            out[(i, j)] = r; // Hr (top-left)
            out[(n + i, n + j)] = r; // Hr (bottom-right)
            out[(i, n + j)] = -m; // -Hi (top-right)
            out[(n + i, j)] = m; // Hi (bottom-left)
        }
    }
    out
}

/// Build the physical complex density `P(k) = sum_band f_band c c^H` from
/// single-electron occupations `g` (length `2n`, max 1 each) and extract the
/// `n x n` Re/Im blocks.
///
/// Each physical band `n` is represented by a degenerate real pair `(a, a')`
/// with `g_a = g_a' = g_n`, so the real-embedding density blocks satisfy
/// `Re P = 2 A` and `Im P = 2 C` with `f_band = 2 g_n` (the factor two is the
/// two electrons of a doubly-occupied spatial orbital). The band energy is the
/// plain `sum_a g_a eps_a`, since the doubling is already in the `2n` sum.
pub fn weighted_density(eig: &KEigen, occupations: &[f64]) -> Result<CMatrix> {
    let p_real = column_weighted_gram(&eig.vectors, occupations)?;
    let n = eig.n;
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = 2.0 * p_real[(i, j)]; // 2 * top-left block A = Re P
            out.im[(i, j)] = 2.0 * p_real[(n + i, j)]; // 2 * bottom-left block C = Im P
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small 2x2 Hermitian generalized problem solved against a hand value.
    #[test]
    fn hermitian_eigen_recovers_real_spectrum() {
        // H = [[2, i],[-i, 2]] has eigenvalues 1 and 3 (S = I).
        let mut h = CMatrix::zeros(2);
        h.re[(0, 0)] = 2.0;
        h.re[(1, 1)] = 2.0;
        h.im[(0, 1)] = 1.0;
        h.im[(1, 0)] = -1.0;
        let mut s = CMatrix::zeros(2);
        s.re[(0, 0)] = 1.0;
        s.re[(1, 1)] = 1.0;
        let eig = hermitian_generalized_eigen(&h, &s, 1.0e-12).unwrap();
        // 2n = 4 eigenvalues: {1,1,3,3}
        assert_eq!(eig.values.len(), 4);
        assert!((eig.values[0] - 1.0).abs() < 1.0e-10);
        assert!((eig.values[1] - 1.0).abs() < 1.0e-10);
        assert!((eig.values[2] - 3.0).abs() < 1.0e-10);
        assert!((eig.values[3] - 3.0).abs() < 1.0e-10);

        // Fully occupy the lowest band (g=1 on the lowest degenerate pair).
        let occ = vec![1.0, 1.0, 0.0, 0.0];
        let p = weighted_density(&eig, &occ).unwrap();
        // Trace of P should be 2 (one doubly-occupied band).
        let trace = p.re[(0, 0)] + p.re[(1, 1)];
        assert!((trace - 2.0).abs() < 1.0e-10);
        // P must be Hermitian: imaginary diagonal zero.
        assert!(p.im[(0, 0)].abs() < 1.0e-12);
        assert!(p.im[(1, 1)].abs() < 1.0e-12);
    }

    // The cached-orthogonaliser eigensolve (built once from S(k)) must be BIT-FOR-BIT
    // identical to the per-call `hermitian_generalized_eigen` that rebuilds S(k)^{-1/2}
    // internally — for the SAME Hermitian H and the SAME Hermitian PD S. This is the
    // no-result-change guarantee for the SCC overlap-factorisation caching: only the
    // Fock iterate changes between SCC iterations, S(k) (and hence its orthogonaliser)
    // does not, so reusing the cached factorisation cannot move any number.
    #[test]
    fn cached_orthogonalizer_matches_per_call_eigen_bit_for_bit() {
        // A non-trivial 3x3 Hermitian generalized problem with off-diagonal complex S.
        let n = 3;
        let mut h = CMatrix::zeros(n);
        // Hermitian H = Hr (symmetric) + i Hi (antisymmetric).
        let hr = [[1.5, 0.4, -0.2], [0.4, 2.1, 0.3], [-0.2, 0.3, 0.9]];
        let hi = [[0.0, 0.25, -0.1], [-0.25, 0.0, 0.15], [0.1, -0.15, 0.0]];
        for i in 0..n {
            for j in 0..n {
                h.re[(i, j)] = hr[i][j];
                h.im[(i, j)] = hi[i][j];
            }
        }
        // Hermitian PD S: identity + small Hermitian perturbation (kept diagonally dominant).
        let mut s = CMatrix::zeros(n);
        let sr = [[1.0, 0.12, 0.05], [0.12, 1.0, -0.08], [0.05, -0.08, 1.0]];
        let si = [[0.0, 0.06, -0.03], [-0.06, 0.0, 0.04], [0.03, -0.04, 0.0]];
        for i in 0..n {
            for j in 0..n {
                s.re[(i, j)] = sr[i][j];
                s.im[(i, j)] = si[i][j];
            }
        }
        let tol = 1.0e-12;

        let reference = hermitian_generalized_eigen(&h, &s, tol).unwrap();

        // Build the orthogonaliser ONCE from S, then solve with a (different) reused H.
        let orth = embedded_overlap_orthogonalizer(&s, tol).unwrap();
        let cached =
            hermitian_generalized_eigen_with_orthogonalizer(&h, &orth, n, tol).unwrap();

        assert_eq!(reference.values.len(), cached.values.len());
        // Eigenvalues must be exactly equal (same code paths post-orthogonalisation).
        for (a, b) in reference.values.iter().zip(cached.values.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "eigenvalue mismatch: {a} (per-call) vs {b} (cached)"
            );
        }
        // Eigenvectors must be exactly equal too.
        assert_eq!(reference.vectors.rows(), cached.vectors.rows());
        assert_eq!(reference.vectors.cols(), cached.vectors.cols());
        for i in 0..reference.vectors.rows() {
            for j in 0..reference.vectors.cols() {
                assert_eq!(
                    reference.vectors[(i, j)].to_bits(),
                    cached.vectors[(i, j)].to_bits(),
                    "eigenvector mismatch at ({i},{j})"
                );
            }
        }
    }
}
