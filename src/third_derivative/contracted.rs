// SPDX-License-Identifier: GPL-3.0-or-later
//! Memory-bounded canonical free-energy FC3 via the variational 2n+1 rule.
//! For occupied columns U(t)=C_occ+t C_occ', use the exactly constrained density
//! P(t)=2 U(t)[U(t)^T S(t) U(t)]^-1 U(t)^T. The trial orbitals have the exact
//! first-order response, so stationarity makes the energy error O(t^4).
//! Evaluating its third Taylor coefficient therefore needs neither C'' nor any
//! Cartesian FC3 slabs. Geometry and density use the same GFN1 energy expression.
//! At positive temperature the trial orbitals are metric-orthonormalized and
//! the full occupation-matrix first response supplies the analytic entropy
//! trace. This retains exactly degenerate fractional blocks without choosing
//! an arbitrary orbital frame or displacing nuclear coordinates.
use crate::coulomb::{harmonic_average, ShellChargeModel};
use crate::error::{Gfn1Error, Result};
use crate::jets::Jet1;
use crate::linalg::{lowdin_solve_generalized, Matrix};
use crate::response::charge_space::{ChargeSpaceContext, DkTables};
use crate::solvation::SolvationDirection;
use crate::{BasisSet, ElectronicOptions, ElectronicResult, Gfn1Parameters, PeriodicSystem};

const FACTORIAL: [f64; 5] = [1., 1., 2., 6., 24.];
type Series = [Matrix; 5];
fn zeros(r: usize, c: usize) -> Series {
    std::array::from_fn(|_| Matrix::zeros(r, c))
}
fn multiply(a: &Series, b: &Series, order: usize) -> Result<Series> {
    let mut out = zeros(a[0].rows(), b[0].cols());
    for k in 0..=order {
        for i in 0..=k {
            if a[i].as_slice().iter().all(|&x| x == 0.0)
                || b[k - i].as_slice().iter().all(|&x| x == 0.0)
            {
                continue;
            }
            let term = a[i].matmul(&b[k - i])?;
            for (x, y) in out[k].as_mut_slice().iter_mut().zip(term.as_slice()) {
                *x += y;
            }
        }
    }
    Ok(out)
}
fn transpose(a: &Series) -> Series {
    std::array::from_fn(|k| a[k].transpose())
}
fn coefficient(j: Jet1) -> [f64; 5] {
    [j.value, j.d1, j.d2 / 2.0, j.d3 / 6.0, j.d4 / 24.0]
}
fn jet(c: [f64; 5]) -> Jet1 {
    Jet1 {
        value: c[0],
        d1: c[1],
        d2: 2.0 * c[2],
        d3: 6.0 * c[3],
        d4: 24.0 * c[4],
    }
}

// Return (scale, scale^order * log^[order]). Scaling the occupation response
// before multiplication avoids inf*0 for exponentially small Fermi tails.
fn log_first_scaled(a: f64, b: f64) -> (f64, f64) {
    let mean = 0.5 * (a + b);
    let delta = (a - b) / (2. * mean);
    let value = if delta.abs() < 0.25 {
        let mut power = 1.;
        let mut sum = 0.;
        for k in 0..24 {
            sum += power / (2 * k + 1) as f64;
            power *= delta * delta;
        }
        sum
    } else {
        ((a / mean).ln() - (b / mean).ln()) / ((a - b) / mean)
    };
    (mean, value)
}

fn log_second_scaled(a: f64, b: f64, c: f64) -> (f64, f64) {
    let mean = (a + b + c) / 3.;
    let mut nodes = [a / mean, b / mean, c / mean];
    let deviations = nodes.map(|x| x - 1.);
    if deviations.iter().all(|x| x.abs() < 0.25) {
        let [x, y, z] = deviations;
        let e1 = x + y + z;
        let e2 = x * y + x * z + y * z;
        let e3 = x * y * z;
        let mut h = [0., 0., 1.];
        let mut sum = 0.;
        for k in 0..40 {
            let value = if k == 0 {
                1.
            } else {
                e1 * h[2] - e2 * h[1] + e3 * h[0]
            };
            if k != 0 {
                h = [h[1], h[2], value];
            }
            sum += if k % 2 == 0 { -value } else { value } / (k + 2) as f64;
        }
        return (mean, sum);
    }
    nodes.sort_by(f64::total_cmp);
    let [x, y, z] = nodes;
    let (my, ly) = log_first_scaled(y, z);
    let (mx, lx) = log_first_scaled(x, y);
    (mean, (ly / my - lx / mx) / (z - x))
}

fn log_third_scaled(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let mean = 0.25 * (a + b + c + d);
    let mut nodes = [a / mean, b / mean, c / mean, d / mean];
    let deviations = nodes.map(|x| x - 1.);
    if deviations.iter().all(|x| x.abs() < 0.25) {
        let [a, b, c, d] = deviations;
        let e = [
            a + b + c + d,
            a * b + a * c + a * d + b * c + b * d + c * d,
            a * b * c + a * b * d + a * c * d + b * c * d,
            a * b * c * d,
        ];
        let mut h = [1., 0., 0., 0.];
        let mut result = 1. / 3.;
        for k in 1..40 {
            let next = e[0] * h[0] - e[1] * h[1] + e[2] * h[2] - e[3] * h[3];
            h = [next, h[0], h[1], h[2]];
            result += if k % 2 == 0 { next } else { -next } / (k + 3) as f64;
        }
        return (mean, result);
    }
    nodes.sort_by(f64::total_cmp);
    let [a, b, c, d] = nodes;
    let (sy, y) = log_second_scaled(b, c, d);
    let (sx, x) = log_second_scaled(a, b, c);
    (mean, (y / sy.powi(2) - x / sx.powi(2)) / (d - a))
}

fn entropy_trial(
    occupations: &[f64],
    response: &Matrix,
    second: Option<&Matrix>,
    kt: f64,
    value: f64,
) -> Jet1 {
    let n = occupations.len();
    let mut result = Jet1::constant(value);
    for i in 0..n {
        let fi = occupations[i];
        let bii = response[(i, i)];
        if bii != 0. {
            result.d1 += kt * (fi.ln() - (2. - fi).ln()) * bii;
        }
        if let Some(second) = second {
            let cii = 2. * second[(i, i)];
            if cii != 0. {
                result.d2 += kt * (fi.ln() - (2. - fi).ln()) * cii;
            }
        }
        for j in 0..n {
            let bij = response[(i, j)];

            let cij = second.map_or(0., |c| 2. * c[(i, j)]);
            if bij == 0. && cij == 0. {
                continue;
            }
            let fj = occupations[j];
            let (scale, weight) = log_first_scaled(fi, fj);

            result.d2 += kt * weight * bij * (response[(j, i)] / scale);
            if cij != 0. {
                result.d3 += 3. * kt * weight * cij * (bij / scale);
                result.d4 += 3. * kt * weight * cij * (cij / scale);
            }
            let (scale, weight) = log_first_scaled(2. - fi, 2. - fj);

            result.d2 += kt * weight * bij * (response[(j, i)] / scale);
            if cij != 0. {
                result.d3 += 3. * kt * weight * cij * (bij / scale);
                result.d4 += 3. * kt * weight * cij * (cij / scale);
            }
            if bij == 0. {
                continue;
            }
            for k in 0..n {
                let bjk = response[(j, k)];

                if bjk == 0. {
                    continue;
                }
                let bki = response[(k, i)];
                let cki = second.map_or(0., |c| 2. * c[(k, i)]);
                let fk = occupations[k];
                if bki != 0. || cki != 0. {
                    let (scale, weight) = log_second_scaled(fi, fj, fk);
                    let real = kt * weight * bij * (bjk / scale);
                    result.d3 += 2. * real * (bki / scale);
                    result.d4 += 12. * real * (cki / scale);
                    let (scale, weight) = log_second_scaled(2. - fi, 2. - fj, 2. - fk);
                    let hole = kt * weight * bij * (bjk / scale);
                    result.d3 -= 2. * hole * (bki / scale);
                    result.d4 -= 12. * hole * (cki / scale);
                }
                if second.is_some() {
                    for l in 0..n {
                        let bkl = response[(k, l)];
                        let bli = response[(l, i)];
                        if bkl == 0. || bli == 0. {
                            continue;
                        }
                        let fl = occupations[l];
                        let (scale, weight) = log_third_scaled(fi, fj, fk, fl);
                        result.d4 +=
                            6. * kt * weight * bij * (bjk / scale) * (bkl / scale) * (bli / scale);
                        let (scale, weight) = log_third_scaled(2. - fi, 2. - fj, 2. - fk, 2. - fl);
                        result.d4 +=
                            6. * kt * weight * bij * (bjk / scale) * (bkl / scale) * (bli / scale);
                    }
                }
            }
        }
    }
    result
}

fn thermal_occupation_matrices(
    f: &[f64],
    table: &DkTables,
    first: &Matrix,
    second: Option<&Matrix>,
    blocks: &[usize],
) -> (Matrix, Matrix) {
    let n = f.len();
    let slopes = &table.fp;
    let total_slope = slopes.iter().sum::<f64>();
    let mut f1 = Matrix::zeros(n, n);
    let mut f2 = Matrix::zeros(n, n);
    if total_slope.abs() <= 1e-30 {
        return (f1, f2);
    }
    let mu1 = slopes
        .iter()
        .enumerate()
        .map(|(i, &slope)| slope * first[(i, i)])
        .sum::<f64>()
        / total_slope;
    let same_block = |i: usize, j: usize| blocks[i] == blocks[j];
    let perturbation = |i: usize, j: usize| first[(i, j)] - if i == j { mu1 } else { 0. };
    for i in 0..n {
        if !(0. < f[i] && f[i] < 2.) {
            continue;
        }
        for j in 0..=i {
            if !(0. < f[j] && f[j] < 2. && same_block(i, j)) {
                continue;
            }
            let slope = table.f1(i, j);
            let change = slope * perturbation(i, j);
            f1[(i, j)] = change;
            f1[(j, i)] = change;
            if let Some(second) = second {
                let square = (blocks[i]..n)
                    .take_while(|&k| same_block(i, k))
                    .map(|k| table.f2(i, k, j) * perturbation(i, k) * perturbation(k, j))
                    .sum::<f64>();
                let change = slope * second[(i, j)] + 2. * square;
                f2[(i, j)] = change;
                f2[(j, i)] = change;
            }
        }
    }
    if second.is_some() {
        let mu2 = (0..n).map(|i| f2[(i, i)]).sum::<f64>() / total_slope;
        for (i, &slope) in slopes.iter().enumerate() {
            f2[(i, i)] = 0.5 * (f2[(i, i)] - slope * mu2);
            for j in 0..i {
                f2[(i, j)] *= 0.5;
                f2[(j, i)] *= 0.5;
            }
        }
    }
    (f1, f2)
}

pub struct CubicReference {
    system: PeriodicSystem,
    params: Gfn1Parameters,
    options: ElectronicOptions,
    electronic: ElectronicResult,
    context: ChargeSpaceContext,
    model: ShellChargeModel,
    occupied: Matrix,
    mos: Matrix,
    orbital_energies: Vec<f64>,
    thermal_weights: Option<DkTables>,
    /// Conservative container estimate, not a measured process-RSS bound.
    pub estimated_workspace_bytes: usize,
}
impl CubicReference {
    pub fn mixed(&self, u: &[f64], v: &[f64], w: &[f64]) -> Result<f64> {
        let n = 3 * self.system.atoms.len();
        if u.len() != n || v.len() != n || w.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "FC3 mixed direction dimension mismatch".into(),
            ));
        }
        let mut direction = vec![0.0; n];
        let mut sum = 0.0;
        for (a, b, sign) in [
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ] {
            for i in 0..n {
                direction[i] = w[i] + a * u[i] + b * v[i];
            }
            sum += sign * self.directional(&direction)?;
        }
        Ok(sum / 24.0)
    }
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        memory_budget_bytes: usize,
    ) -> Result<Self> {
        let mut stationary = options.clone();
        stationary.energy_tolerance = stationary.energy_tolerance.min(1e-12);
        stationary.charge_tolerance = stationary.charge_tolerance.min(1e-11);
        Self::build_from_result(system, params, &stationary, memory_budget_bytes, None)
    }

    pub(crate) fn estimate_workspace(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
    ) -> Result<usize> {
        let basis = BasisSet::build(
            system,
            params,
            crate::BasisOptions {
                nprim: options.nprim,
            },
        )?;
        let n = basis.len();
        let ns = basis.shells.len();
        n.checked_mul(n)
            .and_then(|x| {
                x.checked_mul(if options.electronic_temperature > 0. {
                    112 * 8
                } else {
                    80 * 8
                })
            })
            .and_then(|x| {
                ns.checked_mul(ns)
                    .and_then(|s| s.checked_mul(16 * 8))
                    .and_then(|s| x.checked_add(s))
            })
            .ok_or_else(|| {
                Gfn1Error::InvalidInput("variational derivative workspace size overflow".into())
            })
    }

    pub(crate) fn build_from_result(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        memory_budget_bytes: usize,
        electronic: Option<ElectronicResult>,
    ) -> Result<Self> {
        let memory_budget_bytes = crate::resources::effective_memory_budget(memory_budget_bytes);
        crate::terms::require_order(options, params, 3, "contracted variational FC3")?;
        if system.lattice.is_some() {
            return Err(Gfn1Error::InvalidInput(
                "contracted FC3 is molecular".into(),
            ));
        }
        let estimate = Self::estimate_workspace(system, params, options)?;
        if estimate > memory_budget_bytes {
            return Err(Gfn1Error::InvalidInput(format!(
                "FC3 workspace estimate {estimate} bytes exceeds budget {memory_budget_bytes}"
            )));
        }
        let electronic = match electronic {
            Some(electronic) => electronic,
            None => crate::run_electronic(system, params, options.clone())?,
        };
        if electronic.solvation.is_some() != options.solvation.is_some() {
            return Err(Gfn1Error::InvalidInput(
                "variational response solvent reference does not match electronic options".into(),
            ));
        }
        if !electronic.converged {
            return Err(Gfn1Error::InvalidInput(
                "contracted FC3 requires a converged SCF reference".into(),
            ));
        }
        if options.electronic_temperature <= 0.
            && electronic
                .occupations
                .iter()
                .any(|&f| f > 1e-8 && (f - 2.0).abs() > 1e-8)
        {
            return Err(Gfn1Error::InvalidInput("zero-temperature contracted FC3 requires integer occupations; choose positive electronic temperature for a fractional ensemble".into()));
        }
        let eigen =
            lowdin_solve_generalized(&electronic.fock, &electronic.integrals.overlap, 1e-12)?;
        let indices: Vec<usize> = electronic
            .occupations
            .iter()
            .enumerate()
            .filter(|(_, f)| **f > 1.0)
            .map(|(i, _)| i)
            .collect();
        let n = electronic.basis.len();
        let mut occupied = Matrix::zeros(n, indices.len());
        for i in 0..n {
            for (j, &k) in indices.iter().enumerate() {
                occupied[(i, j)] = eigen.vectors[(i, k)];
            }
        }
        let context = ChargeSpaceContext::build(system, params, &electronic)?;
        let mut model = ShellChargeModel::build(system, &electronic.basis, params)?;
        model.charge_order = options.charge_order.max(3);
        let thermal_weights = context.is_finite_temperature().then(|| {
            DkTables::new(
                &eigen.values,
                &electronic.occupations,
                electronic.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
            )
        });
        Ok(Self {
            system: system.clone(),
            params: params.clone(),
            options: options.clone(),
            electronic,
            context,
            model,
            occupied,
            mos: eigen.vectors,
            orbital_energies: eigen.values,
            thermal_weights,
            estimated_workspace_bytes: estimate,
        })
    }

    /// Directional third energy derivative, not a Taylor coefficient.
    pub fn directional(&self, v: &[f64]) -> Result<f64> {
        Ok(self.directional_derivatives(v)?[3])
    }

    pub fn electronic(&self) -> &ElectronicResult {
        &self.electronic
    }

    pub(crate) fn nuclear_response(
        &self,
        v: &[f64],
    ) -> Result<(crate::response::charge_space::FirstOrderBundle, Matrix)> {
        if v.len() != 3 * self.system.atoms.len() || v.iter().any(|x| !x.is_finite()) {
            return Err(Gfn1Error::InvalidInput(
                "invalid molecular response direction".into(),
            ));
        }

        let (s, h, gamma, _) = self.geometry(v, 1)?;
        let solvent = self.solvent_direction(v)?;
        let potential = self.explicit_potential(&gamma, solvent.as_ref());
        let field = self
            .context
            .first_order_field(self.skeleton(&s, &h, &potential, 1), s[1].clone())?;
        Ok((field.bundle, s[1].clone()))
    }

    pub(crate) fn dipole_response(
        &self,
        field: crate::math::Vec3,
    ) -> Result<crate::response::charge_space::FirstOrderBundle> {
        let e = &self.electronic;
        let basis = &e.basis;
        let n = basis.len();
        let potential: Vec<f64> = basis
            .shells
            .iter()
            .map(|s| {
                -field.dot(
                    self.system.atoms[s.atom_index].position - self.options.external_field.origin,
                )
            })
            .collect();
        let f = crate::electronic::fock_from_shell_potential(
            basis,
            &e.integrals.overlap,
            &Matrix::zeros(n, n),
            &potential,
        );
        Ok(self
            .context
            .first_order_field(f, Matrix::zeros(n, n))?
            .bundle)
    }

    fn solvent_direction(&self, v: &[f64]) -> Result<Option<SolvationDirection>> {
        self.electronic
            .solvation
            .as_ref()
            .map(|solvent| solvent.directional_geometry(v))
            .transpose()
    }

    fn explicit_potential(
        &self,
        gamma: &[Jet1],
        solvent: Option<&SolvationDirection>,
    ) -> Vec<Jet1> {
        let ns = self.electronic.basis.shells.len();
        let atomic = solvent.map(|solvent| solvent.potential(&self.electronic.atomic_charges));
        gamma
            .chunks_exact(ns)
            .enumerate()
            .map(|(i, row)| {
                let mut potential = atomic.as_ref().map_or(Jet1::default(), |atom| {
                    atom[self.electronic.basis.shells[i].atom_index]
                });
                for (g, &q) in row.iter().zip(&self.electronic.shell_charges) {
                    potential.add_scaled(g, q);
                }
                potential.value = self.electronic.shell_scc_potential[i];
                potential
            })
            .collect()
    }

    /// Nuclear Fock derivative at frozen charges, including the CM5 affine leg.
    fn skeleton(&self, s: &Series, h: &Series, potential: &[Jet1], order: usize) -> Matrix {
        let basis = &self.electronic.basis;
        let mut skeleton = h[order].clone();
        for (x, value) in skeleton.as_mut_slice().iter_mut().enumerate() {
            *value *= FACTORIAL[order];
            let a = basis.aos[x / basis.len()].shell_index;
            let b = basis.aos[x % basis.len()].shell_index;
            let coefficients = coefficient(potential[a].add(&potential[b]));
            for k in 0..=order {
                *value -= 0.5 * FACTORIAL[order] * coefficients[k] * s[order - k].as_slice()[x];
            }
        }
        skeleton
    }

    fn second_response(
        &self,
        s: &Series,
        h: &Series,
        gamma: &[Jet1],
        solvent: Option<&SolvationDirection>,
        potential: &[Jet1],
        first: &crate::response::charge_space::FirstOrderField,
    ) -> Result<crate::response::charge_space::SecondOrderField> {
        let basis = &self.electronic.basis;
        let dq = &first.bundle.shell_charges;
        let mut kernel_response = vec![0.; dq.len()];
        for (i, row) in gamma.chunks_exact(dq.len()).enumerate() {
            for (j, g) in row.iter().enumerate() {
                let mut derivative = g.d1;
                if let Some(solvent) = solvent {
                    let a = basis.shells[i].atom_index;
                    let b = basis.shells[j].atom_index;
                    derivative += solvent.matrix[a * self.system.atoms.len() + b].d1;
                }
                kernel_response[i] += derivative * dq[j];
            }
        }
        let mut s2 = s[2].clone();
        for value in s2.as_mut_slice() {
            *value *= 2.;
        }
        self.context.second_order_field(
            first,
            first,
            &self.skeleton(s, h, potential, 2),
            &s2,
            &kernel_response,
            &kernel_response,
        )
    }

    fn thermal_trial_density(
        &self,
        s: &Series,
        response: &crate::response::charge_space::FirstOrderField,

        second: Option<&crate::response::charge_space::SecondOrderField>,
        order: usize,
    ) -> Result<(Series, Jet1)> {
        let mos = &self.mos;
        let n = mos.cols();
        let kt = self.electronic.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
        let thermal = self.context.is_finite_temperature();
        let broad_frame = thermal && self.electronic.solvation.is_none();
        let tolerance = if broad_frame {
            (0.5 * kt).clamp(1e-8, 1e-3)
        } else if thermal {
            1e-10
        } else {
            1e-6
        };
        let mut start = 0;
        let blocks: Vec<_> = (0..n)
            .map(|i| {
                if self.orbital_energies[i] - self.orbital_energies[start] >= tolerance {
                    start = i;
                }
                start
            })
            .collect();
        let (rotation, fock1, second_frame) = if broad_frame {
            crate::response::charge_space::directional_block_frame(
                response, second, &self.orbital_energies, &blocks,
            )?
        } else {
            let (f1, frame2) = crate::response::charge_space::directional_fock_frame(
                response,
                second,
                &self.orbital_energies,
                tolerance,
            )?;
            let frame2 = frame2
                .map(|(f2, u2)| -> Result<_> {
                    let mut total = response.u_rotation.matmul(&response.u_rotation)?;
                    for (value, &add) in total.as_mut_slice().iter_mut().zip(u2.as_slice()) {
                        *value += add;
                    }
                    Ok((total, f2))
                })
                .transpose()?;
            (response.u_rotation.clone(), f1, frame2)
        };
        let f = &self.electronic.occupations;
        let mut u = zeros(n, n);
        u[0] = mos.clone();
        u[1] = mos.matmul(&rotation)?;
        if let Some((u2, _)) = &second_frame {
            u[2] = mos.matmul(u2)?;
            for value in u[2].as_mut_slice() {
                *value *= 0.5;
            }
        }
        let metric = multiply(&multiply(&transpose(&u), s, order)?, &u, order)?;
        let mut x = metric;
        x[0] = Matrix::zeros(n, n);

        let x2 = multiply(&x, &x, order)?;
        let x3 = multiply(&x2, &x, order)?;
        let x4 = if order == 4 {
            Some(multiply(&x3, &x, order)?)
        } else {
            None
        };
        let mut normalization = zeros(n, n);
        normalization[0] = Matrix::identity(n);
        for k in 1..=order {
            for p in 0..n * n {
                normalization[k].as_mut_slice()[p] = -0.5 * x[k].as_slice()[p]
                    + 0.375 * x2[k].as_slice()[p]
                    - 0.3125 * x3[k].as_slice()[p]
                    + x4.as_ref().map_or(0., |x4| 0.2734375 * x4[k].as_slice()[p]);
            }
        }
        let trial_mos = multiply(&u, &normalization, order)?;
        let mut occupation = zeros(n, n);
        for (i, &value) in f.iter().enumerate() {
            occupation[0][(i, i)] = if thermal {
                value
            } else if value > 1. {
                2.
            } else {
                0.
            };
        }
        let entropy = if thermal {
            // The occupation jet acts within the transported energy blocks.
            let (f1, f2) = thermal_occupation_matrices(
                f,
                self.thermal_weights
                    .as_ref()
                    .expect("thermal occupation weights"),
                &fock1,
                second_frame.as_ref().map(|(_, f2)| f2),
                &blocks,
            );
            occupation[1] = f1;
            occupation[2] = f2;
            entropy_trial(
                f,
                &occupation[1],
                second.map(|_| &occupation[2]),
                kt,
                self.electronic.electronic_entropy_term,
            )
        } else {
            Jet1::constant(self.electronic.electronic_entropy_term)
        };
        let density = multiply(
            &multiply(&trial_mos, &occupation, order)?,
            &transpose(&trial_mos),
            order,
        )?;
        Ok((density, entropy))
    }

    /// Stationary directional [E, E', E'', E''']; masses are not applied.
    /// The exactly normalized trial density is variationally exact to order 3.

    pub fn directional_derivatives(&self, v: &[f64]) -> Result<[f64; 4]> {
        let [e, d1, d2, d3, _] = self.directional_energy(v, 3)?;
        Ok([e, d1, d2, d3])
    }

    pub(crate) fn directional_fourth(&self, v: &[f64]) -> Result<f64> {
        Ok(self.directional_energy(v, 4)?[4])
    }

    pub(crate) fn directional_second(&self, v: &[f64]) -> Result<f64> {
        Ok(self.directional_energy(v, 2)?[2])
    }

    fn directional_energy(&self, v: &[f64], order: usize) -> Result<[f64; 5]> {
        let nat = self.system.atoms.len();
        let ndof = 3 * nat;
        if v.len() != ndof || v.iter().any(|x| !x.is_finite()) {
            return Err(Gfn1Error::InvalidInput(
                "invalid contracted FC3 direction".into(),
            ));
        }
        if v.iter().all(|&x| x == 0.0) {
            return Ok([self.electronic.total_free, 0., 0., 0., 0.]);
        }
        let _profile = crate::profile::scope("fc3.contracted.direction");
        let basis = &self.electronic.basis;
        let n = basis.len();
        let ns = basis.shells.len();

        let (s, h, gamma, structural_energy) = self.geometry(v, order)?;
        let solvent = self.solvent_direction(v)?;
        let potential = self.explicit_potential(&gamma, solvent.as_ref());
        let response = self
            .context
            .first_order_field(self.skeleton(&s, &h, &potential, 1), s[1].clone())?;
        let second = if order == 4 {
            Some(self.second_response(&s, &h, &gamma, solvent.as_ref(), &potential, &response)?)
        } else {
            None
        };
        let (density, entropy) = if self.context.is_finite_temperature()
            || (second.is_some() && solvent.is_some())
        {
            self.thermal_trial_density(&s, &response, second.as_ref(), order)?
        } else {
            // In the symmetric occupied gauge: C'=P' S C/2+C(C^T S' C)/2.
            let s_occ = s[0].matmul(&self.occupied)?;
            let term1 = response.bundle.density.matmul(&s_occ)?;
            let metric = self
                .occupied
                .transpose()
                .matmul(&s[1].matmul(&self.occupied)?)?;
            let term2 = self.occupied.matmul(&metric)?;
            let mut u = zeros(n, self.occupied.cols());
            u[0] = self.occupied.clone();
            for i in 0..n {
                for j in 0..u[1].cols() {
                    u[1][(i, j)] = 0.5 * (term1[(i, j)] + term2[(i, j)]);
                }
            }
            if let Some(second) = &second {
                // The exact second derivative of the occupied projector
                // determines the virtual component of C''. The occupied
                // component is supplied by normalization, avoiding orbital
                // denominators inside the occupied and virtual subspaces.
                let c1_overlap = u[1].transpose().matmul(&s_occ)?;
                let quadratic = u[1].matmul(&c1_overlap)?;
                let mut motion = second.bundle.density.matmul(&s_occ)?;
                for (value, &term) in motion.as_mut_slice().iter_mut().zip(quadratic.as_slice()) {
                    *value = 0.5 * *value - 2. * term;
                }
                let projection = self.occupied.matmul(&s_occ.transpose().matmul(&motion)?)?;
                for (value, (&motion, &projection)) in u[2]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(motion.as_slice().iter().zip(projection.as_slice()))
                {
                    *value = 0.5 * (motion - projection);
                }
            }
            let ut = transpose(&u);
            let overlap = multiply(&multiply(&ut, &s, order)?, &u, order)?;
            let no = self.occupied.cols();
            let mut inverse = zeros(no, no);
            inverse[0] = Matrix::identity(no);
            for k in 1..=order {
                for i in 1..=k {
                    let term = overlap[i].matmul(&inverse[k - i])?;
                    for (x, y) in inverse[k].as_mut_slice().iter_mut().zip(term.as_slice()) {
                        *x -= y;
                    }
                }
            }
            let mut density = multiply(&multiply(&u, &inverse, order)?, &ut, order)?;
            for m in &mut density {
                for x in m.as_mut_slice() {
                    *x *= 2.0;
                }
            }
            (
                density,
                Jet1::constant(self.electronic.electronic_entropy_term),
            )
        };
        let mut energy = entropy;
        let mut band = [0.0; 5];
        for k in 0..=order {
            for i in 0..=k {
                band[k] += density[i]
                    .as_slice()
                    .iter()
                    .zip(h[k - i].as_slice())
                    .map(|(p, h)| p * h)
                    .sum::<f64>();
            }
        }
        energy = energy.add(&jet(band));
        let ps = multiply(&density, &s, order)?;
        let mut charges = vec![Jet1::constant(0.0); ns];
        for (sindex, shell) in basis.shells.iter().enumerate() {
            let mut q = [shell.reference_occ, 0., 0., 0., 0.];
            for k in 0..=order {
                for mu in shell.first_ao..shell.first_ao + shell.nao {
                    q[k] -= ps[k][(mu, mu)];
                }
            }
            charges[sindex] = jet(q);
        }
        for i in 0..ns {
            for j in 0..ns {
                energy = energy.add(
                    &charges[i]
                        .mul(&charges[j])
                        .mul(&gamma[i * ns + j])
                        .scale(0.5),
                );
            }
        }
        let model = &self.model;
        let mut atomic_charges = Vec::with_capacity(nat);
        for a in 0..nat {
            let mut qa = Jet1::constant(0.0);
            for (s, shell) in basis.shells.iter().enumerate() {
                if shell.atom_index == a {
                    qa = qa.add(&charges[s]);
                }
            }
            atomic_charges.push(qa);
            let offset = model.atom_offsets[a];
            let gam3 = model.hubbard_derivs[offset];
            energy = energy.add(&qa.mul(&qa).mul(&qa).scale(gam3 / 3.0));
            let hard = model.hardness[offset];
            if model.charge_order > 3 && hard.abs() > 1e-8 {
                let mut power = qa.mul(&qa).mul(&qa);
                for order in 4..=model.charge_order {
                    power = power.mul(&qa);
                    let xn =
                        hard / ((order - 1) as f64) * (2.0 * gam3 / hard).powi(order as i32 - 2);
                    energy = energy.add(&power.scale(xn / order as f64));
                }
            }
        }
        if let Some(solvent) = solvent {
            energy = energy.add(&solvent.energy(&atomic_charges));
        }
        energy = energy.add(&crate::repulsion::repulsion_directional_jet(
            &self.system,
            &self.params,
            v,
        )?);
        energy = energy.add(&crate::halogen::halogen_directional_jet(
            &self.system,
            &self.params,
            v,
        )?);
        if self.options.enable_dispersion {
            energy = energy.add(&crate::dispersion::dispersion_directional_jet(
                &self.system,
                &self.params,
                self.options.d3_reference_path.as_deref(),
                v,
            )?);
        }
        if let Some(correction) = structural_energy {
            energy = energy.add(&correction);
        }
        let derivatives = [energy.value, energy.d1, energy.d2, energy.d3, energy.d4];
        if derivatives[..=order].iter().any(|x| !x.is_finite()) {
            return Err(Gfn1Error::InvalidInput(
                "non-finite variational energy derivative".into(),
            ));
        }
        Ok(derivatives)
    }

    fn geometry(
        &self,
        v: &[f64],
        order: usize,
    ) -> Result<(Series, Series, Vec<Jet1>, Option<Jet1>)> {
        let e = &self.electronic;
        let basis = &e.basis;
        let n = basis.len();
        let ns = basis.shells.len();
        let nat = self.system.atoms.len();
        let mut s = zeros(n, n);
        s[0] = e.integrals.overlap.clone();
        for i in 0..n {
            let ai = basis.aos[i].atom_index;
            let ri = self.system.atoms[ai].position;
            for j in 0..i {
                let aj = basis.aos[j].atom_index;
                if ai == aj {
                    continue;
                }
                let rj = self.system.atoms[aj].position;
                if (ri - rj).norm() > self.options.hamiltonian.integral_cutoff {
                    continue;
                }
                let velocity = crate::math::Vec3::new(
                    v[3 * ai] - v[3 * aj],
                    v[3 * ai + 1] - v[3 * aj + 1],
                    v[3 * ai + 2] - v[3 * aj + 2],
                );
                if velocity.norm2() == 0. {
                    continue;
                }
                let overlap = coefficient(crate::integrals::directional_overlap(
                    &basis.aos[i],
                    &basis.aos[j],
                    ri - rj,
                    velocity,
                ));
                for k in 1..=order {
                    s[k][(i, j)] = overlap[k];
                    s[k][(j, i)] = overlap[k];
                }
            }
        }
        let mut distance = vec![Jet1::constant(0.0); nat * nat];
        for a in 0..nat {
            for b in 0..a {
                let dr = (self.system.atoms[a].position - self.system.atoms[b].position).to_array();
                let mut rr = Jet1::constant(0.0);
                for k in 0..3 {
                    let x = Jet1 {
                        value: dr[k],
                        d1: v[3 * a + k] - v[3 * b + k],
                        ..Default::default()
                    };
                    rr = rr.add(&x.mul(&x));
                }
                let r = rr.sqrt();
                distance[a * nat + b] = r;
                distance[b * nat + a] = r;
            }
        }
        let mut cn = vec![Jet1::constant(0.0); nat];
        if self.options.hamiltonian.enable_cn_hamiltonian {
            let radii: Vec<f64> = self
                .system
                .atoms
                .iter()
                .map(|a| crate::data_tables::covalent_radius_d3_bohr(a.z))
                .collect::<Result<_>>()?;
            for a in 0..nat {
                for b in 0..a {
                    let r = distance[a * nat + b];
                    if r.value > self.options.hamiltonian.coordination_cutoff {
                        continue;
                    }
                    let mut arg = Jet1::constant(radii[a] + radii[b])
                        .div(&r)
                        .add_scalar(-1.0)
                        .scale(-16.0);
                    if !(-80.0..=80.0).contains(&arg.value) {
                        arg = Jet1::constant(arg.value.clamp(-80.0, 80.0));
                    }
                    let val = arg.exp().add_scalar(1.0).powf(-1.0);
                    cn[a] = cn[a].add(&val);
                    cn[b] = cn[b].add(&val);
                }
            }
        }
        let structural = self
            .options
            .ml_model
            .as_ref()
            .map(|model| model.directional_parameters(&self.system, v))
            .transpose()?;
        let self_energy: Vec<Jet1> = basis
            .shells
            .iter()
            .map(|sh| {
                let mut value = cn[sh.atom_index]
                    .scale(-sh.kcn_raw.unwrap_or(0.0))
                    .add_scalar(sh.hdiag_ha);
                if let Some(parameters) = &structural {
                    // The basis already stores theta(R0). Add only structural
                    // derivatives, keeping its reference value exactly intact.
                    let mut delta = parameters.levels[sh.atom_index][sh.angular.as_index()];
                    delta.value = 0.0;
                    value = value.add(&delta);
                }
                value
            })
            .collect();
        let mut h = zeros(n, n);
        for i in 0..n {
            let a = &basis.shells[basis.aos[i].shell_index];
            for j in 0..=i {
                let b = &basis.shells[basis.aos[j].shell_index];
                let mut prefactor = self_energy[basis.aos[i].shell_index]
                    .add(&self_energy[basis.aos[j].shell_index])
                    .scale(0.5);
                if a.atom_index != b.atom_index {
                    let radius = crate::data_tables::atomic_radius_bohr(a.z)?
                        + crate::data_tables::atomic_radius_bohr(b.z)?;
                    let rr = distance[a.atom_index * nat + b.atom_index]
                        .scale(1.0 / radius)
                        .sqrt();
                    let polynomial = rr
                        .scale(a.poly_raw.unwrap_or(0.0))
                        .add_scalar(1.0)
                        .mul(&rr.scale(b.poly_raw.unwrap_or(0.0)).add_scalar(1.0));
                    prefactor = prefactor.mul(&polynomial).scale(crate::hamiltonian::hscale(
                        a,
                        b,
                        &self.params,
                    )?);
                }
                let sj = jet(std::array::from_fn(|k| s[k][(i, j)]));
                let hj = coefficient(prefactor.mul(&sj));
                for k in 0..=order {
                    h[k][(i, j)] = hj[k];
                    h[k][(j, i)] = hj[k];
                }
            }
        }
        let model = &self.model;
        let mut gamma = vec![Jet1::constant(0.0); ns * ns];
        for i in 0..ns {
            for j in 0..=i {
                let a = basis.shells[i].atom_index;
                let b = basis.shells[j].atom_index;
                let hardness = harmonic_average(model.hardness[i], model.hardness[j]);
                let g = if a == b {
                    Jet1::constant(hardness)
                } else {
                    distance[a * nat + b]
                        .mul(&distance[a * nat + b])
                        .add_scalar(1.0 / (hardness * hardness))
                        .powf(-0.5)
                };
                gamma[i * ns + j] = g;
                gamma[j * ns + i] = g;
            }
        }
        Ok((
            s,
            h,
            gamma,
            structural.map(|parameters| parameters.correction),
        ))
    }
}
