// SPDX-License-Identifier: GPL-3.0-or-later
//! Generalized-metric first derivative of a Fermi matrix function.
use crate::error::{Gfn1Error, Result};
use crate::linalg::{column_weighted_gram, lowdin_solve_generalized, EigenDecomposition, Matrix};

pub(super) fn trace(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows())
        .map(|i| (0..a.cols()).map(|j| a[(i, j)] * b[(j, i)]).sum::<f64>())
        .sum()
}
pub(super) fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
pub(crate) fn fermi(energy: f64, mu: f64, kt: f64) -> f64 {
    let x = (energy - mu) / kt;
    if x > 0. {
        let t = (-x).exp();
        2. * t / (1. + t)
    } else {
        2. / (1. + x.exp())
    }
}
pub(super) fn entropy(f: f64, kt: f64) -> f64 {
    let t = 0.5 * f;
    let x = if t > 0. { t * t.ln() } else { 0. };
    let y = if t < 1. { (1. - t) * (1. - t).ln() } else { 0. };
    2. * kt * (x + y)
}
pub(crate) fn canonical_mu(energies: &[f64], electrons: f64, kt: f64) -> Result<f64> {
    if electrons < -1e-9 || electrons > 2. * energies.len() as f64 + 1e-9 {
        return Err(Gfn1Error::InvalidInput(
            "fragment electron count exceeds its orbital capacity".into(),
        ));
    }
    if kt == 0. {
        let n = (electrons / 2.).round() as usize;
        if (electrons - 2. * n as f64).abs() > 1e-8 {
            return Err(Gfn1Error::InvalidInput("zero-temperature fragment response requires an even integer electron count; use Fermi smearing for open-shell fragments".into()));
        }
        if n == 0 {
            return Ok(energies[0] - 1.);
        }
        if n == energies.len() {
            return Ok(energies[n - 1] + 1.);
        }
        if energies[n] - energies[n - 1] < 1e-8 {
            return Err(Gfn1Error::InvalidInput(
                "fragment occupied/virtual gap is singular at zero temperature; use Fermi smearing"
                    .into(),
            ));
        }
        return Ok(0.5 * (energies[n - 1] + energies[n]));
    }
    let mut low = energies[0] - 100. * kt - 1.;
    let mut high = energies[energies.len() - 1] + 100. * kt + 1.;
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if mid == low || mid == high {
            break;
        }
        if energies.iter().map(|&e| fermi(e, mid, kt)).sum::<f64>() < electrons {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub(crate) struct Spectrum {
    pub eig: EigenDecomposition,
    pub occupations: Vec<f64>,
    pub mu: f64,
    pub kt: f64,
    pub density: Matrix,
    pub entropy_density: Matrix,
}
impl Spectrum {
    pub fn new(
        h: &Matrix,
        s: &Matrix,
        tolerance: f64,
        kt: f64,
        electrons: Option<f64>,
        mu: f64,
    ) -> Result<Self> {
        let eig = lowdin_solve_generalized(h, s, tolerance)?;
        let mu = if let Some(n) = electrons {
            canonical_mu(&eig.values, n, kt)?
        } else {
            mu
        };
        Self::from_eigen(eig, kt, mu)
    }
    pub fn from_eigen(eig: EigenDecomposition, kt: f64, mu: f64) -> Result<Self> {
        let occupations: Vec<_> = eig
            .values
            .iter()
            .map(|&e| {
                if kt > 0. {
                    fermi(e, mu, kt)
                } else if e < mu {
                    2.
                } else {
                    0.
                }
            })
            .collect();
        let density = column_weighted_gram(&eig.vectors, &occupations)?;
        let weights: Vec<_> = occupations.iter().map(|&f| entropy(f, kt)).collect();
        let entropy_density = column_weighted_gram(&eig.vectors, &weights)?;
        Ok(Self {
            eig,
            occupations,
            mu,
            kt,
            density,
            entropy_density,
        })
    }
    pub fn response(
        &self,
        dh: &Matrix,
        ds: &Matrix,
        dmu: f64,
        entropy_channel: bool,
    ) -> Result<Matrix> {
        let c = &self.eig.vectors;
        let n = c.cols();
        let h = c.transpose().matmul(dh)?.matmul(c)?;
        let s = c.transpose().matmul(ds)?.matmul(c)?;
        let scalar = |energy: f64, f: f64| {
            let slope = if self.kt > 0. {
                -f * (1. - 0.5 * f) / self.kt
            } else {
                0.
            };
            if entropy_channel {
                (entropy(f, self.kt), -(energy - self.mu) * slope)
            } else {
                (f, slope)
            }
        };
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let ei = self.eig.values[i];
                let ej = self.eig.values[j];
                let (fi, di) = scalar(ei, self.occupations[i]);
                let (fj, _) = scalar(ej, self.occupations[j]);
                let gap = ei - ej;
                let (dd, energy_dd) = if i == j || gap.abs() < 1e-7 * self.kt.max(1e-8) {
                    let middle = 0.5 * (ei + ej);
                    let f = if self.kt > 0. {
                        fermi(middle, self.mu, self.kt)
                    } else {
                        self.occupations[i]
                    };
                    let (value, slope) = scalar(middle, f);
                    (slope, value + middle * slope)
                } else {
                    ((fi - fj) / gap, (ei * fi - ej * fj) / gap)
                };
                d[(i, j)] = dd * h[(i, j)] - energy_dd * s[(i, j)];
                if i == j {
                    d[(i, j)] -= di * dmu;
                }
            }
        }
        c.matmul(&d)?.matmul(&c.transpose())
    }
    pub fn canonical_response(
        &self,
        dh: &Matrix,
        ds: &Matrix,
        s: &Matrix,
    ) -> Result<(Matrix, Matrix)> {
        let mut dp = self.response(dh, ds, 0., false)?;
        let zero = Matrix::zeros(s.rows(), s.cols());
        let p_mu = self.response(&zero, &zero, 1., false)?;
        let denom = trace(&p_mu, s);
        let dn = trace(&dp, s) + trace(&self.density, ds);
        let dmu = if denom > 1e-18 { -dn / denom } else { 0. };
        add(&mut dp, &p_mu, dmu);
        let de = self.response(dh, ds, dmu, true)?;
        Ok((dp, de))
    }

    /// Canonical analytic second density response, including the full overlap
    /// metric and both chemical-potential derivatives. Confluent matrix-function
    /// divided differences support exactly degenerate thermal blocks.
    pub(crate) fn canonical_second_density(
        &self,
        dh: &Matrix,
        ds: &Matrix,
        d2h: &Matrix,
        d2s: &Matrix,
    ) -> Result<Matrix> {
        use crate::response::charge_space::{ChargeSpaceContext, DkTables};
        let c = &self.eig.vectors;
        let transform = |matrix: &Matrix| c.transpose().matmul(matrix)?.matmul(c);
        let h1 = transform(dh)?;
        let s1 = transform(ds)?;
        let h2 = transform(d2h)?;
        let s2 = transform(d2s)?;
        let tables = DkTables::new(&self.eig.values, &self.occupations, self.kt);
        let first0 = ChargeSpaceContext::dk_first_order_mo(&tables, 0, &h1, &s1, 0.);
        let n = c.cols();
        let denom = -tables.fp.iter().sum::<f64>();
        let count1: f64 = (0..n)
            .map(|i| first0[(i, i)] + self.occupations[i] * s1[(i, i)])
            .sum();
        let mu1 = if denom > 1e-18 { -count1 / denom } else { 0. };
        let first = ChargeSpaceContext::dk_first_order_mo(&tables, 0, &h1, &s1, mu1);
        let mut second = ChargeSpaceContext::dk_second_order_mo(
            &tables, 0, &h1, &s1, &h1, &s1, &h2, &s2, mu1, mu1,
        );
        let count2: f64 = (0..n)
            .map(|i| second[(i, i)] + self.occupations[i] * s2[(i, i)])
            .sum::<f64>()
            + 2. * trace(&first, &s1);
        let mu2 = if denom > 1e-18 { -count2 / denom } else { 0. };
        for i in 0..n {
            second[(i, i)] -= tables.fp[i] * mu2;
        }
        c.matmul(&second)?.matmul(&c.transpose())
    }
}
