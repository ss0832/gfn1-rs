// SPDX-License-Identifier: GPL-3.0-or-later
use crate::error::{Gfn1Error, Result};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Accum, MatMut as FaerMatMut, MatRef as FaerMatRef};
use std::ops::{Index, IndexMut};
/// Compatibility hook. Faer's default global sentinel resolves parallelism in
/// the current Rayon pool on each call. Setting `Par::rayon(0)` globally would
/// instead freeze the size of whichever pool first reached this function.
/// Per-calculation scopes therefore leave the process-global setting alone.
#[inline]
pub fn ensure_parallelism() {}

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

/// Deprecated alias for [`symmetric_eigen`]. The `tol` and `max_sweeps`
/// arguments were never used: this has delegated to a direct (faer) symmetric
/// eigendecomposition rather than a Jacobi sweep loop since well before v0.5.0.
#[deprecated(
    since = "0.5.0",
    note = "use symmetric_eigen; tol/max_sweeps were ignored"
)]
pub fn symmetric_eigen_jacobi(
    input: &Matrix,
    tol: f64,
    max_sweeps: usize,
) -> Result<EigenDecomposition> {
    let _ = (tol, max_sweeps);
    symmetric_eigen(input)
}

/// Eigenvalues (ascending) and eigenvectors of a real symmetric matrix.
pub fn symmetric_eigen(input: &Matrix) -> Result<EigenDecomposition> {
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
    // Zero-copy strided view over the row-major buffer — no O(N²) conversion.
    let faer_input = FaerMatRef::from_row_major_slice(input.as_slice(), n, n);
    crate::resources::check_dense_workspace(n, n, 8, "symmetric eigensolver")?;
    let par = crate::resources::faer_parallelism();
    let mut values_native = faer::diag::Diag::<f64>::zeros(n);
    let mut vectors_native = faer::Mat::<f64>::zeros(n, n);
    let mut scratch = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    faer::linalg::evd::self_adjoint_evd(
        faer_input,
        values_native.as_mut(),
        Some(vectors_native.as_mut()),
        par,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|err| {
        Gfn1Error::InvalidInput(format!("faer symmetric eigensolver failed: {err:?}"))
    })?;
    let mut values = Vec::with_capacity(n);
    for value in values_native.column_vector().iter() {
        values.push(*value);
    }
    let u = vectors_native.as_ref();
    let mut vectors = Matrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            vectors[(i, j)] = u[(i, j)];
        }
    }
    Ok(EigenDecomposition { values, vectors })
}

/// Eigenpairs of D A D for positive diagonal D and positive-definite A.
/// Factor A = L L^T and take the SVD of L^T D to preserve the smallest
/// singular values at the scale of D instead of its square.
pub(crate) fn scaled_positive_eigen(mut a: Matrix, d: &[f64]) -> Result<EigenDecomposition> {
    use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
    use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};

    let n = d.len();
    crate::resources::check_dense_workspace(n, n, 8, "scaled response eigensolver")?;
    let par = crate::resources::faer_parallelism();
    cholesky_in_place(
        FaerMatMut::from_row_major_slice_mut(a.as_mut_slice(), n, n),
        Default::default(),
        par,
        MemStack::new(&mut MemBuffer::new(cholesky_in_place_scratch::<f64>(
            n,
            par,
            Default::default(),
        ))),
        Default::default(),
    )
    .map_err(|err| {
        Gfn1Error::InvalidInput(format!(
            "unstable RPA reference: scaled response is not positive definite ({err:?})"
        ))
    })?;
    for j in 0..n {
        for i in 0..=j {
            a[(i, j)] = a[(j, i)] * d[j];
        }
    }
    for i in 0..n {
        for j in 0..i {
            a[(i, j)] = 0.;
        }
    }
    let mut singular = faer::diag::Diag::<f64>::zeros(n);
    let mut native_vectors = faer::Mat::<f64>::zeros(n, n);
    let mut scratch = MemBuffer::new(svd_scratch::<f64>(
        n,
        n,
        ComputeSvdVectors::No,
        ComputeSvdVectors::Thin,
        par,
        Default::default(),
    ));
    svd(
        FaerMatRef::from_row_major_slice(a.as_slice(), n, n),
        singular.as_mut(),
        None,
        Some(native_vectors.as_mut()),
        par,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|err| Gfn1Error::LinearAlgebra(format!("scaled response SVD failed: {err:?}")))?;
    let sigma = singular.column_vector();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_unstable_by(|&i, &j| sigma[i].total_cmp(&sigma[j]));
    let mut vectors = Matrix::zeros(n, n);
    for (col, &old) in order.iter().enumerate() {
        for row in 0..n {
            vectors[(row, col)] = native_vectors[(row, old)];
        }
    }
    let values = order
        .iter()
        .map(|&index| sigma[index] * sigma[index])
        .collect();
    Ok(EigenDecomposition { values, vectors })
}

pub fn lowdin_orthogonalizer(s: &Matrix, tol: f64) -> Result<LowdinOrthogonalizer> {
    if s.rows != s.cols {
        return Err(Gfn1Error::InvalidInput(
            "Lowdin orthogonalizer requires a square overlap matrix".to_string(),
        ));
    }
    let n = s.rows;
    let seig = symmetric_eigen(s)?;
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
    // `tol` is kept for signature compatibility with `lowdin_solve_generalized`
    // (where it does gate the overlap positive-definiteness check); the
    // orthogonalizer is already built here, and the symmetric eigensolver takes
    // no tolerance.
    let _ = tol;
    if h.rows != h.cols || h.rows != orth.x.rows || h.cols != orth.x.cols {
        return Err(Gfn1Error::InvalidInput(
            "generalized eigensolver/orthogonalizer shape mismatch".to_string(),
        ));
    }
    let h_orth = lowdin_congruence_transform(h, orth)?;
    let eig = symmetric_eigen(&h_orth)?;
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
    // Direct row-major dot products: BLAS-2 is memory-bound, and the row-major
    // layout makes each row contiguous — no faer matrix rebuild per call (this
    // runs every SCC iteration on the nsh×nsh Coulomb matrix).
    Ok(a.data
        .chunks_exact(a.cols)
        .map(|row| row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
        .collect())
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

/// `Aᵀ · B` for two row-major matrices sharing their **row** count, without ever
/// materializing `Aᵀ`.
///
/// A faer `MatRef::transpose()` only swaps the strides, so the transposed operand
/// costs nothing; the explicit [`Matrix::transpose`] would pay an `O(rows·cols)`
/// copy (and, for the tall-skinny shapes this exists for — `npair × nsh` transition
/// charge blocks with `npair ≫ nsh` — a second full-size allocation). Numerically
/// this is the same GEMM as `a.transpose().matmul(b)`.
pub fn matmul_transpose_a(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.rows() != b.rows() {
        return Err(Gfn1Error::InvalidInput(format!(
            "transpose-multiply shape mismatch: {}x{} transposed times {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        )));
    }
    ensure_parallelism();
    let (rows, cols) = (a.cols(), b.cols());
    crate::resources::check_dense_workspace(
        rows.max(a.rows()),
        cols.max(a.cols()),
        4,
        "transpose GEMM",
    )?;
    let mut out = Matrix::zeros(rows, cols);
    if rows == 0 || cols == 0 || a.rows() == 0 {
        return Ok(out);
    }
    let lhs = FaerMatRef::from_row_major_slice(a.as_slice(), a.rows(), a.cols()).transpose();
    let rhs = FaerMatRef::from_row_major_slice(b.as_slice(), b.rows(), b.cols());
    let dst = FaerMatMut::from_row_major_slice_mut(out.as_mut_slice(), rows, cols);
    faer::linalg::matmul::matmul(
        dst,
        Accum::Replace,
        lhs,
        rhs,
        1.0,
        crate::resources::faer_parallelism(),
    );
    Ok(out)
}

fn matmul_dense(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.cols != b.rows {
        return Err(Gfn1Error::InvalidInput(format!(
            "matrix multiply shape mismatch: {}x{} times {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    ensure_parallelism();
    // Zero-copy GEMM: strided views over the row-major inputs, writing straight
    // into the row-major output. This removes the three O(N²) layout copies the
    // old Mat conversions paid on every multiply (~11 per SCC iteration).
    let (rows, cols) = (a.rows, b.cols);
    crate::resources::check_dense_workspace(rows.max(a.cols), cols.max(b.rows), 4, "GEMM")?;
    let mut out = Matrix::zeros(rows, cols);
    let lhs = FaerMatRef::from_row_major_slice(a.as_slice(), a.rows, a.cols);
    let rhs = FaerMatRef::from_row_major_slice(b.as_slice(), b.rows, b.cols);
    let dst = FaerMatMut::from_row_major_slice_mut(out.as_mut_slice(), rows, cols);
    faer::linalg::matmul::matmul(
        dst,
        Accum::Replace,
        lhs,
        rhs,
        1.0,
        crate::resources::faer_parallelism(),
    );
    Ok(out)
}

/// Dense LU factorization (partial pivoting) of a square matrix, reusable for
/// many right-hand sides — the workhorse behind the charge-space dielectric
/// solve, where one nsh×nsh factorization serves all first- and second-order
/// response right-hand sides.
pub struct DenseLu {
    n: usize,
    lu: faer::Mat<f64>,
    forward: Vec<usize>,
    inverse: Vec<usize>,
}

impl DenseLu {
    pub fn factor(a: &Matrix) -> Result<Self> {
        if a.rows != a.cols {
            return Err(Gfn1Error::InvalidInput(
                "DenseLu requires a square matrix".to_string(),
            ));
        }
        crate::resources::check_dense_workspace(a.rows, a.cols, 4, "dense LU")?;
        ensure_parallelism();
        let view = FaerMatRef::from_row_major_slice(a.as_slice(), a.rows, a.cols);
        let n = a.rows;
        let par = crate::resources::faer_parallelism();
        let mut lu = view.to_owned();
        let mut forward = vec![0usize; n];
        let mut inverse = vec![0usize; n];
        let mut scratch = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
                n,
                n,
                par,
                Default::default(),
            ),
        );
        let _ = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut forward,
            &mut inverse,
            par,
            MemStack::new(&mut scratch),
            Default::default(),
        );
        Ok(Self {
            n: a.rows,
            lu,
            forward,
            inverse,
        })
    }

    pub fn solve_vec(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.n {
            return Err(Gfn1Error::InvalidInput(format!(
                "DenseLu rhs length {} != n {}",
                rhs.len(),
                self.n
            )));
        }
        let solution = self.solve_matrix(&Matrix::from_vec(self.n, 1, rhs.to_vec())?)?;
        Ok((0..self.n).map(|i| solution[(i, 0)]).collect())
    }

    pub fn solve_matrix(&self, rhs: &Matrix) -> Result<Matrix> {
        if rhs.rows() != self.n {
            return Err(Gfn1Error::InvalidInput(
                "DenseLu RHS matrix dimension mismatch".into(),
            ));
        }
        crate::resources::check_dense_workspace(self.n, rhs.cols(), 4, "dense LU RHS")?;
        let par = crate::resources::faer_parallelism();
        let mut solution =
            FaerMatRef::from_row_major_slice(rhs.as_slice(), rhs.rows(), rhs.cols()).to_owned();
        let mut scratch = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, f64>(
                self.n,
                rhs.cols(),
                par,
            ),
        );
        let permutation = faer::perm::PermRef::new_checked(&self.forward, &self.inverse, self.n);
        faer::linalg::lu::partial_pivoting::solve::solve_in_place(
            self.lu.as_ref(),
            self.lu.as_ref(),
            permutation,
            solution.as_mut(),
            par,
            MemStack::new(&mut scratch),
        );
        let mut result = Matrix::zeros(self.n, rhs.cols());
        for i in 0..self.n {
            for j in 0..rhs.cols() {
                result[(i, j)] = solution[(i, j)];
            }
        }
        Ok(result)
    }
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
