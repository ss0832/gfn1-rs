// SPDX-License-Identifier: GPL-3.0-or-later
use crate::error::{Gfn1Error, Result};
use faer::{Mat as FaerMat, Side};
use std::ops::{Index, IndexMut};
use std::sync::OnceLock;

/// Switch faer's dense linear algebra (eigensolves, matmuls, factorizations) to the
/// multi-threaded path once, lazily. The O(N³) SCF eigensolve and the Löwdin congruence
/// matmuls dominate large-system cost; without this faer runs single-threaded. `Par::rayon(0)`
/// uses the shared rayon global pool (so it composes with the crate's existing `into_par_iter`
/// loops rather than oversubscribing). Numerically identical — only the scheduling changes.
/// `GFN1_FAER_THREADS=1` (or `0`) forces the sequential path for benchmarking/reproducibility.
#[inline]
pub fn ensure_parallelism() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let seq = std::env::var("GFN1_FAER_THREADS")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "0" || v == "1"
            })
            .unwrap_or(false);
        let par = if seq {
            faer::Par::Seq
        } else {
            faer::Par::rayon(0)
        };
        faer::set_global_parallelism(par);
    });
}

#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct EigenDecomposition {
    pub values: Vec<f64>,
    /// Eigenvectors are stored column-wise in a dense row-major matrix.
    pub vectors: Matrix,
}

#[derive(Clone, Debug)]
pub struct LowdinOrthogonalizer {
    /// Symmetric orthogonalizer X = S^{-1/2}.
    pub x: Matrix,
    /// Cached transpose of X. For Lowdin X this is numerically symmetric, but
    /// storing it once avoids repeated allocation in SCC loops.
    pub xt: Matrix,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut out = Self::zeros(n, n);
        for i in 0..n {
            out[(i, i)] = 1.0;
        }
        out
    }
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Gfn1Error::InvalidInput(format!(
                "matrix data length {} does not match {rows}x{cols}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        matmul_dense(self, rhs)
    }
    pub fn max_abs_diff(&self, rhs: &Self) -> f64 {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
    pub fn column(&self, col: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, col)]).collect()
    }
    pub fn symmetrize_from_lower(&mut self) {
        for i in 0..self.rows {
            for j in 0..i {
                self[(j, i)] = self[(i, j)];
            }
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

pub fn symmetric_eigen_jacobi(
    input: &Matrix,
    tol: f64,
    max_sweeps: usize,
) -> Result<EigenDecomposition> {
    let _ = (tol, max_sweeps);
    if input.rows != input.cols {
        return Err(Gfn1Error::InvalidInput(
            "symmetric eigensolver requires a square matrix".to_string(),
        ));
    }
    let n = input.rows;
    if n == 0 {
        return Ok(EigenDecomposition {
            values: Vec::new(),
            vectors: Matrix::zeros(0, 0),
        });
    }
    if n == 1 {
        return Ok(EigenDecomposition {
            values: vec![input[(0, 0)]],
            vectors: Matrix::identity(1),
        });
    }

    ensure_parallelism();
    let faer_input = matrix_to_faer(input);
    let eig = faer_input.self_adjoint_eigen(Side::Lower).map_err(|err| {
        Gfn1Error::InvalidInput(format!("faer symmetric eigensolver failed: {err:?}"))
    })?;
    let mut values = Vec::with_capacity(n);
    for value in eig.S().column_vector().iter() {
        values.push(*value);
    }
    let u = eig.U();
    let mut vectors = Matrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            vectors[(i, j)] = u[(i, j)];
        }
    }
    Ok(EigenDecomposition { values, vectors })
}

pub fn lowdin_orthogonalizer(s: &Matrix, tol: f64) -> Result<LowdinOrthogonalizer> {
    if s.rows != s.cols {
        return Err(Gfn1Error::InvalidInput(
            "Lowdin orthogonalizer requires a square overlap matrix".to_string(),
        ));
    }
    let n = s.rows;
    let seig = symmetric_eigen_jacobi(s, tol, 100 * n.max(1) * n.max(1))?;
    let mut scaled_vectors = Matrix::zeros(n, n);
    for a in 0..n {
        let lambda = seig.values[a];
        if lambda <= tol {
            return Err(Gfn1Error::InvalidInput(format!(
                "overlap matrix is not positive definite; eigenvalue {a} = {lambda:.3e}"
            )));
        }
        let fac = 1.0 / lambda.sqrt();
        for i in 0..n {
            scaled_vectors[(i, a)] = seig.vectors[(i, a)] * fac;
        }
    }
    let x = matmul_dense(&scaled_vectors, &seig.vectors.transpose())?;
    let xt = x.transpose();
    Ok(LowdinOrthogonalizer { x, xt })
}

pub fn lowdin_solve_with_orthogonalizer(
    h: &Matrix,
    orth: &LowdinOrthogonalizer,
    tol: f64,
) -> Result<EigenDecomposition> {
    if h.rows != h.cols || h.rows != orth.x.rows || h.cols != orth.x.cols {
        return Err(Gfn1Error::InvalidInput(
            "generalized eigensolver/orthogonalizer shape mismatch".to_string(),
        ));
    }
    let n = h.rows;
    let h_orth = lowdin_congruence_transform(h, orth)?;
    let eig = symmetric_eigen_jacobi(&h_orth, tol, 100 * n.max(1) * n.max(1))?;
    let coeff = matmul_dense(&orth.x, &eig.vectors)?;
    Ok(EigenDecomposition {
        values: eig.values,
        vectors: coeff,
    })
}

pub fn column_weighted_gram(c: &Matrix, weights: &[f64]) -> Result<Matrix> {
    if c.cols != weights.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "column-weighted Gram shape mismatch: matrix has {} columns but {} weights were provided",
            c.cols,
            weights.len()
        )));
    }
    if c.rows == 0 {
        return Ok(Matrix::zeros(0, 0));
    }
    let mut scaled = Matrix::zeros(c.rows, c.cols);
    for col in 0..c.cols {
        let weight = weights[col];
        if weight == 0.0 {
            continue;
        }
        for row in 0..c.rows {
            scaled[(row, col)] = c[(row, col)] * weight;
        }
    }
    let mut out = matmul_dense(&scaled, &c.transpose())?;
    for i in 0..out.rows {
        for j in 0..i {
            let avg = 0.5 * (out[(i, j)] + out[(j, i)]);
            out[(i, j)] = avg;
            out[(j, i)] = avg;
        }
    }
    Ok(out)
}

pub fn matrix_vector_product(a: &Matrix, x: &[f64]) -> Result<Vec<f64>> {
    if a.cols != x.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "matrix-vector shape mismatch: {}x{} times {}",
            a.rows,
            a.cols,
            x.len()
        )));
    }
    if a.rows == 0 {
        return Ok(Vec::new());
    }
    let faer_a = matrix_to_faer(a);
    let faer_x = FaerMat::from_fn(x.len(), 1, |i, _| x[i]);
    let product = &faer_a * &faer_x;
    let mut out = Vec::with_capacity(a.rows);
    for i in 0..a.rows {
        out.push(product[(i, 0)]);
    }
    Ok(out)
}

pub fn row_gram(rows: &[Vec<f64>]) -> Result<Matrix> {
    let m = rows.len();
    if m == 0 {
        return Ok(Matrix::zeros(0, 0));
    }
    let n = rows[0].len();
    let mut data = Vec::with_capacity(m * n);
    for row in rows {
        if row.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "row Gram input has inconsistent row lengths".to_string(),
            ));
        }
        data.extend_from_slice(row);
    }
    let r = Matrix::from_vec(m, n, data)?;
    r.matmul(&r.transpose())
}

fn lowdin_congruence_transform(h: &Matrix, orth: &LowdinOrthogonalizer) -> Result<Matrix> {
    let n = h.rows;
    if h.cols != n || orth.x.rows != n || orth.x.cols != n || orth.xt.rows != n || orth.xt.cols != n
    {
        return Err(Gfn1Error::InvalidInput(
            "Lowdin transform shape mismatch".to_string(),
        ));
    }
    let hx = matmul_dense(h, &orth.x)?;
    matmul_dense(&orth.xt, &hx)
}

fn matmul_dense(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.cols != b.rows {
        return Err(Gfn1Error::InvalidInput(format!(
            "matrix multiply shape mismatch: {}x{} times {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    ensure_parallelism();
    let faer_a = matrix_to_faer(a);
    let faer_b = matrix_to_faer(b);
    let product = &faer_a * &faer_b;
    Ok(matrix_from_faer(&product))
}

fn matrix_to_faer(input: &Matrix) -> FaerMat<f64> {
    FaerMat::from_fn(input.rows, input.cols, |i, j| input[(i, j)])
}

fn matrix_from_faer(input: &FaerMat<f64>) -> Matrix {
    let rows = input.nrows();
    let cols = input.ncols();
    let mut out = Matrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            out[(i, j)] = input[(i, j)];
        }
    }
    out
}

pub fn lowdin_solve_generalized(h: &Matrix, s: &Matrix, tol: f64) -> Result<EigenDecomposition> {
    if h.rows != h.cols || s.rows != s.cols || h.rows != s.rows {
        return Err(Gfn1Error::InvalidInput(
            "generalized eigensolver requires same-size square H and S".to_string(),
        ));
    }
    let orth = lowdin_orthogonalizer(s, tol)?;
    lowdin_solve_with_orthogonalizer(h, &orth, tol)
}
