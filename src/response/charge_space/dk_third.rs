// SPDX-License-Identifier: GPL-3.0-or-later
//! Canonical third resolvent derivative in a fixed reference MO basis.
use super::*;

fn choose(n: usize, k: usize) -> usize {
    if n < k {
        return 0;
    }
    (0..k).fold(1usize, |value, i| value * (n - i) / (i + 1))
}
fn packed(nodes: &[usize]) -> usize {
    let mut sorted = [0; 4];
    sorted[..nodes.len()].copy_from_slice(nodes);
    sorted[..nodes.len()].sort_unstable();
    sorted[..nodes.len()]
        .iter()
        .enumerate()
        .map(|(i, &n)| choose(n + i, i + 1))
        .sum()
}
pub(super) struct ThermalWeightCache {
    values: Vec<Vec<[f64; 5]>>,
}
impl ThermalWeightCache {
    fn build(context: &ChargeSpaceContext, t: &DkTables) -> Self {
        let n = t.eps.len();
        let mut values: Vec<Vec<[f64; 5]>> = Vec::new();
        for degree in 1..=4 {
            let mut table = Vec::with_capacity(choose(n + degree - 1, degree));
            let row = |nodes: &[usize]| {
                let mut result = [0.; 5];
                result[0] = context.thermal_dd(t, nodes);
                for level in 1..5 {
                    result[level] = t.eps[nodes[0]] * result[level - 1];
                    if degree > 1 {
                        result[level] += values[degree - 2][packed(&nodes[1..])][level - 1];
                    }
                }
                result
            };
            for a in 0..n {
                if degree == 1 {
                    table.push(row(&[a]));
                    continue;
                }
                for b in 0..=a {
                    if degree == 2 {
                        table.push(row(&[b, a]));
                        continue;
                    }
                    for c in 0..=b {
                        if degree == 3 {
                            table.push(row(&[c, b, a]));
                            continue;
                        }
                        for d in 0..=c {
                            table.push(row(&[d, c, b, a]));
                        }
                    }
                }
            }
            values.push(table);
        }
        Self { values }
    }
}

fn trace(a: &Matrix) -> f64 {
    (0..a.rows()).map(|i| a[(i, i)]).sum()
}
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows())
        .flat_map(|i| (0..a.cols()).map(move |j| a[(i, j)] * b[(j, i)]))
        .sum()
}
fn add_scaled(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}

impl ChargeSpaceContext {
    // Hermite divided differences. Nearby nodes use the Fermi ODE Taylor
    // series, avoiding subtraction of almost equal occupation numbers.
    fn thermal_dd(&self, t: &DkTables, nodes: &[usize]) -> f64 {
        if nodes.len() == 1 {
            return t.f[nodes[0]];
        }
        let mut order = nodes.to_vec();
        order.sort_unstable_by(|&a, &b| t.eps[a].total_cmp(&t.eps[b]));
        let a = t.eps[order[0]];
        let b = t.eps[*order.last().unwrap()];
        let m = order.len() - 1;
        if b - a <= 0.05 * self.kt {
            let anchor = order[order.len() / 2];
            let mut c = [0.; 17];
            c[0] = t.f[anchor];
            for k in 0..16 {
                c[k + 1] = (-c[k] + 0.5 * (0..=k).map(|j| c[j] * c[k - j]).sum::<f64>())
                    / (self.kt * (k + 1) as f64);
            }
            // Complete homogeneous polynomials in the shifted nodes.
            let mut h = [0.; 17];
            h[0] = 1.;
            for &i in &order {
                let x = t.eps[i] - t.eps[anchor];
                for k in 1..=16 - m {
                    h[k] += x * h[k - 1];
                }
            }
            return (m..=16).map(|k| c[k] * h[k - m]).sum();
        }
        (self.thermal_dd(t, &order[1..]) - self.thermal_dd(t, &order[..order.len() - 1])) / (b - a)
    }
    fn thermal_weight(&self, t: &DkTables, level: usize, nodes: &[usize]) -> f64 {
        // At most about 11 MiB; larger spectra compute weights on demand.
        let cache = self.dk_third_cache.get_or_init(|| {
            if t.eps.len() <= 48 {
                Some(ThermalWeightCache::build(self, t))
            } else {
                None
            }
        });
        if let Some(cache) = cache {
            return cache.values[nodes.len() - 1][packed(nodes)][level];
        }
        if level == 0 {
            return self.thermal_dd(t, nodes);
        }
        let mut value = t.eps[nodes[0]] * self.thermal_weight(t, level - 1, nodes);
        if nodes.len() > 1 {
            value += self.thermal_weight(t, level - 1, &nodes[1..]);
        }
        value
    }
    // Return derivatives of P or of the shifted energy-weighted density.
    // A_k already contains all chemical-potential derivative terms.
    #[allow(clippy::too_many_arguments)]
    fn third_resolvent(
        &self,
        t: &DkTables,
        level: usize,
        a1: &Matrix,
        s1: &Matrix,
        a2: &Matrix,
        s2: &Matrix,
        a3: &Matrix,
        s3: &Matrix,
    ) -> Matrix {
        let n = t.eps.len();
        let mut out = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                let mut value = self.thermal_weight(t, level, &[p, q]) * a3[(p, q)]
                    - self.thermal_weight(t, level + 1, &[p, q]) * s3[(p, q)];
                for r in 0..n {
                    let w: [f64; 3] =
                        std::array::from_fn(|k| self.thermal_weight(t, level + k, &[p, r, q]));
                    value += 3.
                        * (w[0] * (a2[(p, r)] * a1[(r, q)] + a1[(p, r)] * a2[(r, q)])
                            - w[1]
                                * (s2[(p, r)] * a1[(r, q)]
                                    + a2[(p, r)] * s1[(r, q)]
                                    + s1[(p, r)] * a2[(r, q)]
                                    + a1[(p, r)] * s2[(r, q)])
                            + w[2] * (s2[(p, r)] * s1[(r, q)] + s1[(p, r)] * s2[(r, q)]));
                    for s in 0..n {
                        let w: [f64; 4] = std::array::from_fn(|k| {
                            self.thermal_weight(t, level + k, &[p, r, s, q])
                        });
                        let a = a1[(p, r)];
                        let b = a1[(r, s)];
                        let c = a1[(s, q)];
                        let x = s1[(p, r)];
                        let y = s1[(r, s)];
                        let z = s1[(s, q)];
                        value += 6.
                            * (w[0] * a * b * c - w[1] * (x * b * c + a * y * c + a * b * z)
                                + w[2] * (x * y * c + x * b * z + a * y * z)
                                - w[3] * x * y * z);
                    }
                }
                out[(p, q)] = value;
            }
        }
        out
    }

    /// A canonical, frame-free third response. The factored dielectric is
    /// shared with the first and second responses; no displaced SCF is used.
    pub(super) fn solve_third_order_spectral(
        &self,
        first: &FirstOrderField,
        second: &SecondOrderField,
        f3_ext: &Matrix,
        s2_ao: &Matrix,
        s3_ao: &Matrix,
    ) -> Result<ThirdOrderBundle> {
        let t = self.dk_tables();
        let n = t.eps.len();
        let s1 = &first.s_tilde;
        let s2 = &second.s_dot_fixed;
        let h1 = &first.h_tilde;
        let h2 = &second.h_dot_fixed;
        let s3 = self.mo_transform(s3_ao)?;
        let p0 = {
            let mut p = Matrix::zeros(n, n);
            for i in 0..n {
                p[(i, i)] = t.f[i];
            }
            p
        };
        let fps: f64 = t.fp.iter().sum();
        let mu = |number: f64| if fps.abs() > 1e-30 { number / fps } else { 0. };
        let mut p1 = Self::dk_first_order_mo(t, 0, h1, s1, 0.);
        let mu1 = mu(trace(&p1) + trace_product(&p0, s1));
        for i in 0..n {
            p1[(i, i)] -= mu1 * t.fp[i];
        }
        let mut p2 = Self::dk_second_order_mo(t, 0, h1, s1, h1, s1, h2, s2, mu1, mu1);
        let mu2 = mu(trace(&p2) + 2. * trace_product(&p1, s1) + trace_product(&p0, s2));
        for i in 0..n {
            p2[(i, i)] -= mu2 * t.fp[i];
        }
        let mut a1 = h1.clone();
        let mut a2 = h2.clone();
        for i in 0..n {
            a1[(i, i)] -= mu1;
            a2[(i, i)] -= mu2;
        }
        add_scaled(&mut a2, s1, -2. * mu1);
        let solve = |h3: &Matrix| -> (Matrix, f64, Matrix) {
            let mut a3 = h3.clone();
            add_scaled(&mut a3, s1, -3. * mu2);
            add_scaled(&mut a3, s2, -3. * mu1);
            let mut p3 = self.third_resolvent(t, 0, &a1, s1, &a2, s2, &a3, &s3);
            let mu3 = mu(trace(&p3)
                + 3. * trace_product(&p2, s1)
                + 3. * trace_product(&p1, s2)
                + trace_product(&p0, &s3));
            for i in 0..n {
                p3[(i, i)] -= mu3 * t.fp[i];
                a3[(i, i)] -= mu3;
            }
            (p3, mu3, a3)
        };
        let h3_ext = self.mo_transform(f3_ext)?;
        let (p3_ext, _, _) = solve(&h3_ext);
        let density_ext = mo_coefficient_matrix_to_ao(&self.mos, &p3_ext)?;
        let zero = Matrix::zeros(self.basis.len(), self.basis.len());
        let mut rhs = response_shell_charges_from_density(
            &self.basis,
            &self.overlap,
            &self.density0,
            &density_ext,
            s3_ao,
        )?;
        let cross21 = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &second.bundle.density,
            &zero,
            &first.overlap_deriv,
        )?;
        let cross12 = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &first.bundle.density,
            &zero,
            s2_ao,
        )?;
        for i in 0..rhs.len() {
            rhs[i] += 3. * (cross21[i] + cross12[i]);
        }
        let charges = self.dielectric.solve_vec(&rhs)?;
        let potential = matrix_vector_product(&self.kernel, &charges)?;
        let correction = scalar_response_fock_matrix(&self.basis, &self.overlap, &potential)?;
        let mut h3 = h3_ext;
        add_scaled(&mut h3, &self.mo_transform(&correction)?, 1.);
        let (p3, mu3, a3) = solve(&h3);
        let mut w3 = self.third_resolvent(t, 1, &a1, s1, &a2, s2, &a3, &s3);
        // Undo the moving chemical-potential shift for physical W = sum eps f CC^T.
        add_scaled(&mut w3, &p2, 3. * mu1);
        add_scaled(&mut w3, &p1, 3. * mu2);
        add_scaled(&mut w3, &p0, mu3);
        let density = mo_coefficient_matrix_to_ao(&self.mos, &p3)?;
        let energy_weighted = mo_coefficient_matrix_to_ao(&self.mos, &w3)?;
        // Diagonal of the occupation matrix in the reference symmetric
        // orthonormal frame. Individual MO occupation labels inside a
        // degenerate block are gauge dependent; the matrix response is physical.
        let mut l1 = s1.clone();
        for x in l1.as_mut_slice() {
            *x *= 0.5;
        }
        let mut l2 = s2.clone();
        for x in l2.as_mut_slice() {
            *x *= 0.5;
        }
        add_scaled(&mut l2, &s1.matmul(s1)?, -0.25);
        let mut l3 = s3.clone();
        for x in l3.as_mut_slice() {
            *x *= 0.5;
        }
        add_scaled(&mut l3, &s1.matmul(s2)?, -0.375);
        add_scaled(&mut l3, &s2.matmul(s1)?, -0.375);
        add_scaled(&mut l3, &s1.matmul(s1)?.matmul(s1)?, 0.375);
        let mut occupation = p3;
        for (l, p, scale) in [(&l1, &p2, 3.), (&l2, &p1, 3.), (&l3, &p0, 1.)] {
            add_scaled(&mut occupation, &l.matmul(p)?, scale);
            add_scaled(&mut occupation, &p.matmul(l)?, scale);
        }
        add_scaled(&mut occupation, &l1.matmul(&p1)?.matmul(&l1)?, 6.);
        add_scaled(&mut occupation, &l2.matmul(&p0)?.matmul(&l1)?, 3.);
        add_scaled(&mut occupation, &l1.matmul(&p0)?.matmul(&l2)?, 3.);
        Ok(ThirdOrderBundle {
            density,
            energy_weighted,
            shell_charges: charges,
            occupation_response: (0..n).map(|i| occupation[(i, i)]).collect(),
        })
    }
}
