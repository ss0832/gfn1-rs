// SPDX-License-Identifier: GPL-3.0-or-later
use super::geometry::{Direction, Geometry};
use super::spectral::{add, canonical_mu, trace, Spectrum};
use super::CovalentCut;
use crate::electronic::{BroydenMixer, ElectronicOptions};
use crate::error::{Gfn1Error, Result};
use crate::linalg::{
    lowdin_orthogonalizer, matrix_vector_product, symmetric_eigen, DenseLu, EigenDecomposition,
    Matrix,
};
use crate::math::Vec3;
use crate::system::PeriodicSystem;

pub(super) fn subset(m: &Matrix, rows: &[usize], cols: &[usize]) -> Matrix {
    let mut result = Matrix::zeros(rows.len(), cols.len());
    for (i, &a) in rows.iter().enumerate() {
        for (j, &b) in cols.iter().enumerate() {
            result[(i, j)] = m[(a, b)];
        }
    }
    result
}
pub(super) struct Hop {
    indices: [usize; 4],
    bond: Vec3,
    bda: usize,
    baa: usize,
    ghost: bool,
}
pub(super) struct Cluster {
    pub aos: Vec<usize>,
    pub shells: Vec<usize>,
    pub ao_shell: Vec<usize>,
    pub shell_atom: Vec<usize>,
    pub references: Vec<f64>,
    pub scales: Vec<f64>,
    pub s: Matrix,
    pub h: Matrix,
    pub gamma: Matrix,
    pub core: Vec<f64>,
    pub electrons: f64,
    pub gam3: Vec<f64>,
    d: Matrix,
    hops: Vec<Hop>,
    hop_chart: Option<(Matrix, Vec<f64>)>,
}

impl Cluster {
    pub fn new(
        geometry: &Geometry,
        system: &PeriodicSystem,
        atoms: &[usize],
        core_atoms: &[usize],
        charge: f64,
        cuts: &[CovalentCut],
        penalty: f64,
    ) -> Result<Self> {
        let nat = system.atoms.len();
        let mut scales = vec![0.; nat];
        for &a in atoms {
            scales[a] = 1.;
        }
        let real = scales.clone();
        for cut in cuts {
            let bda = real[cut.bda] > 0.;
            let baa = real[cut.baa] > 0.;
            if bda && !baa {
                let reference: f64 = geometry
                    .basis
                    .shells
                    .iter()
                    .filter(|s| s.atom_index == cut.bda)
                    .map(|s| s.reference_occ)
                    .sum();
                scales[cut.bda] = (reference - 1.) / reference;
            } else if baa && !bda {
                let reference: f64 = geometry
                    .basis
                    .shells
                    .iter()
                    .filter(|s| s.atom_index == cut.bda)
                    .map(|s| s.reference_occ)
                    .sum();
                scales[cut.bda] = 1. / reference;
            }
        }
        let shells: Vec<_> = geometry
            .basis
            .shells
            .iter()
            .enumerate()
            .filter(|(_, s)| scales[s.atom_index] > 0.)
            .map(|(i, _)| i)
            .collect();
        let aos: Vec<_> = geometry
            .basis
            .aos
            .iter()
            .enumerate()
            .filter(|(_, a)| scales[a.atom_index] > 0.)
            .map(|(i, _)| i)
            .collect();
        let ao_shell: Vec<_> = aos
            .iter()
            .map(|&i| {
                shells
                    .iter()
                    .position(|&s| s == geometry.basis.aos[i].shell_index)
                    .expect("selected shell")
            })
            .collect();
        let shell_atom: Vec<_> = shells
            .iter()
            .map(|&i| geometry.basis.shells[i].atom_index)
            .collect();
        let references: Vec<_> = shells
            .iter()
            .map(|&i| {
                geometry.basis.shells[i].reference_occ * scales[geometry.basis.shells[i].atom_index]
            })
            .collect();
        let electrons = references.iter().sum::<f64>() - charge;
        let core: Vec<_> = aos
            .iter()
            .map(|&i| {
                if core_atoms.contains(&geometry.basis.aos[i].atom_index) {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        let gam3: Vec<_> = shells
            .iter()
            .map(|&i| geometry.model.hubbard_derivs[i])
            .collect();
        let s = subset(&geometry.s, &aos, &aos);
        let h = subset(&geometry.h, &aos, &aos);
        let gamma = subset(&geometry.gamma, &shells, &shells);
        let mut hops = Vec::new();
        for cut in cuts {
            let bda = real[cut.bda] > 0.;
            let baa = real[cut.baa] > 0.;
            if bda == baa {
                continue;
            }
            let mut indices = [usize::MAX; 4];
            for (local, &global) in aos.iter().enumerate() {
                let ao = &geometry.basis.aos[global];
                if ao.atom_index != cut.bda || !ao.is_valence {
                    continue;
                }
                let index = match ao.cart_label {
                    "s" => Some(0),
                    "px" => Some(1),
                    "py" => Some(2),
                    "pz" => Some(3),
                    _ => None,
                };
                if let Some(i) = index {
                    indices[i] = local;
                }
            }
            if indices.contains(&usize::MAX) {
                return Err(Gfn1Error::InvalidInput(
                    "HOP bond-detached atom requires a complete valence s/p shell".into(),
                ));
            }
            let translation = system.lattice.map_or(Vec3::zero(), |l| {
                l.translation(crate::lattice::ImageOffset { n: cut.image })
            });
            let bond =
                system.atoms[cut.baa].position + translation - system.atoms[cut.bda].position;
            if bond.norm() < 1e-8 {
                return Err(Gfn1Error::InvalidInput(
                    "HOP cut bond has zero length".into(),
                ));
            }
            hops.push(Hop {
                indices,
                bond,
                bda: cut.bda,
                baa: cut.baa,
                ghost: !bda,
            });
        }
        let mut d = Matrix::zeros(aos.len(), aos.len());
        for hop in &hops {
            add_hop(&mut d, hop, penalty, None, 3 * nat);
        }
        let hop_chart = if hops.is_empty() {
            None
        } else {
            // Diagonalize the geometry-only constraint separately. Adding a
            // 10^6-Hartree projector to the AO Fock buries SCC updates below its
            // ulp and stalls tight charge convergence. In this chart all varying
            // physical Fock elements retain their ordinary Hartree scale.
            let x = lowdin_orthogonalizer(&s, 1e-12)?.x;
            let constraint = x
                .transpose()
                .matmul(&s.matmul(&d)?.matmul(&s)?)?
                .matmul(&x)?;
            let eigen = symmetric_eigen(&constraint)?;
            let rank: usize = hops.iter().map(|h| if h.ghost { 3 } else { 1 }).sum();
            let null = aos.len().checked_sub(rank).ok_or_else(|| {
                Gfn1Error::InvalidInput(
                    "HOP constraints exceed the fragment orbital dimension".into(),
                )
            })?;
            let mut barriers = eigen.values;
            if barriers[null] <= penalty * 1e-8 {
                return Err(Gfn1Error::InvalidInput(
                    "linearly dependent HOP boundary constraints".into(),
                ));
            }
            barriers[..null].fill(0.);
            Some((x.matmul(&eigen.vectors)?, barriers))
        };
        Ok(Self {
            aos,
            shells,
            ao_shell,
            shell_atom,
            references,
            scales,
            s,
            h,
            gamma,
            core,
            electrons,
            gam3,
            d,
            hops,
            hop_chart,
        })
    }
    fn spectrum(&self, potential: &[f64], options: &ElectronicOptions) -> Result<Spectrum> {
        let kt = options.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
        let Some((transform, barriers)) = &self.hop_chart else {
            return Spectrum::new(
                &self.fock(potential)?,
                &self.s,
                options.eigen_tolerance,
                kt,
                Some(self.electrons),
                0.,
            );
        };
        let mut physical = self.h.clone();
        add(&mut physical, &self.shift(potential, &self.s), 1.);
        let mut h = transform.transpose().matmul(&physical)?.matmul(transform)?;
        for (i, &barrier) in barriers.iter().enumerate() {
            h[(i, i)] += barrier;
        }
        let mut eig = relative_jacobi(&h)?;
        eig.vectors = transform.matmul(&eig.vectors)?;
        let mu = canonical_mu(&eig.values, self.electrons, kt)?;
        if eig.values.iter().filter(|&&e| e < mu).count()
            > barriers.iter().filter(|&&b| b == 0.).count()
        {
            return Err(Gfn1Error::InvalidInput(
                "fragment electron count fills forbidden HOP orbitals".into(),
            ));
        }
        Spectrum::from_eigen(eig, kt, mu)
    }
    pub fn shell_q(&self, p: &Matrix, ds: Option<&Matrix>, reference: bool) -> Vec<f64> {
        let mut q = if reference {
            self.references.clone()
        } else {
            vec![0.; self.shells.len()]
        };
        let s = ds.unwrap_or(&self.s);
        for i in 0..self.aos.len() {
            for j in 0..self.aos.len() {
                q[self.ao_shell[i]] -= p[(i, j)] * s[(j, i)];
            }
        }
        q
    }
    pub fn atomic_q(&self, q: &[f64]) -> Vec<f64> {
        let mut atomic = vec![0.; self.scales.len()];
        for (i, &a) in self.shell_atom.iter().enumerate() {
            atomic[a] += q[i];
        }
        atomic
    }
    pub fn potential(&self, q: &[f64], external: &[f64]) -> Result<Vec<f64>> {
        let atomic = self.atomic_q(q);
        let mut v = matrix_vector_product(&self.gamma, q)?;
        for i in 0..v.len() {
            v[i] += self.gam3[i] * atomic[self.shell_atom[i]].powi(2) + external[self.shells[i]];
        }
        Ok(v)
    }
    pub fn shift(&self, potential: &[f64], s: &Matrix) -> Matrix {
        let n = self.aos.len();
        let mut out = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] =
                    -0.5 * (potential[self.ao_shell[i]] + potential[self.ao_shell[j]]) * s[(i, j)];
            }
        }
        out
    }
    pub fn fock(&self, potential: &[f64]) -> Result<Matrix> {
        let mut fock = self.h.clone();
        add(&mut fock, &self.shift(potential, &self.s), 1.);
        add(&mut fock, &self.s.matmul(&self.d)?.matmul(&self.s)?, 1.);
        Ok(fock)
    }
    pub fn kernel(&self, q: &[f64]) -> Matrix {
        let atomic = self.atomic_q(q);
        let mut k = self.gamma.clone();
        for i in 0..q.len() {
            for j in 0..q.len() {
                if self.shell_atom[i] == self.shell_atom[j] {
                    k[(i, j)] += 2. * self.gam3[i] * atomic[self.shell_atom[i]];
                }
            }
        }
        k
    }
    pub fn global_q(&self, q: &[f64], nsh: usize) -> Vec<f64> {
        let mut result = vec![0.; nsh];
        for (i, &s) in self.shells.iter().enumerate() {
            result[s] += q[i];
        }
        result
    }
    pub fn fixed_fock_derivative(
        &self,
        direction: &Direction,
        potential: &[f64],
        dv: &[f64],
        y: usize,
        ncart: usize,
        penalty: f64,
    ) -> Result<(Matrix, Matrix)> {
        let s = subset(&direction.s, &self.aos, &self.aos);
        let mut h = subset(&direction.h, &self.aos, &self.aos);
        add(&mut h, &self.shift(potential, &s), 1.);
        add(&mut h, &self.shift(dv, &self.s), 1.);
        add(&mut h, &s.matmul(&self.d)?.matmul(&self.s)?, 1.);
        add(&mut h, &self.s.matmul(&self.d)?.matmul(&s)?, 1.);
        let mut dd = Matrix::zeros(self.aos.len(), self.aos.len());
        for hop in &self.hops {
            add_hop(&mut dd, hop, penalty, Some(y), ncart);
        }
        add(&mut h, &self.s.matmul(&dd)?.matmul(&self.s)?, 1.);
        Ok((h, s))
    }
    pub fn physical_energy(&self, q: &[f64], spec: &Spectrum) -> Result<f64> {
        let atomic = self.atomic_q(q);
        let gq = matrix_vector_product(&self.gamma, q)?;
        let e3 = (0..q.len())
            .map(|i| self.gam3[i] * q[i] * atomic[self.shell_atom[i]].powi(2) / 3.)
            .sum::<f64>();
        Ok(trace(&spec.density, &self.h)
            + 0.5 * q.iter().zip(&gq).map(|(q, v)| q * v).sum::<f64>()
            + e3
            + trace(&spec.entropy_density, &self.s))
    }
    pub fn physical_derivative(
        &self,
        q: &[f64],
        dq: &[f64],
        spec: &Spectrum,
        dp: &Matrix,
        ds_entropy: &Matrix,
        direction: &Direction,
    ) -> Result<f64> {
        let dh = subset(&direction.h, &self.aos, &self.aos);
        let ds = subset(&direction.s, &self.aos, &self.aos);
        let dg = subset(&direction.gamma, &self.shells, &self.shells);
        let potential = self.potential(q, &vec![0.; direction.gamma.rows()])?;
        Ok(trace(dp, &self.h)
            + trace(&spec.density, &dh)
            + dq.iter().zip(&potential).map(|(a, b)| a * b).sum::<f64>()
            + 0.5
                * q.iter()
                    .zip(matrix_vector_product(&dg, q)?)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            + trace(ds_entropy, &self.s)
            + trace(&spec.entropy_density, &ds))
    }
}

fn add_hop(d: &mut Matrix, hop: &Hop, penalty: f64, direction: Option<usize>, ncart: usize) {
    let length = hop.bond.norm();
    let b = hop.bond / length;
    let factor = 3.0_f64.sqrt() / 2.;
    let h = [0.5, factor * b.x, factor * b.y, factor * b.z];
    let db = if let Some(y) = direction {
        let mut motion = [0.; 3];
        if y < ncart {
            motion[y % 3] =
                (usize::from(hop.baa == y / 3) as f64) - (usize::from(hop.bda == y / 3) as f64);
        } else {
            motion[(y - ncart) / 3] = hop.bond.to_array()[(y - ncart) % 3];
        }
        let v = Vec3::new(motion[0], motion[1], motion[2]);
        let v = (v - b * b.dot(v)) / length;
        [0., factor * v.x, factor * v.y, factor * v.z]
    } else {
        [0.; 4]
    };
    for i in 0..4 {
        for j in 0..4 {
            let sign = if hop.ghost { -1. } else { 1. };
            let value = if direction.is_some() {
                sign * (db[i] * h[j] + h[i] * db[j])
            } else {
                sign * h[i] * h[j] + if hop.ghost && i == j { 1. } else { 0. }
            };
            d[(hop.indices[i], hop.indices[j])] += penalty * value;
        }
    }
}

pub(super) struct LocalState {
    pub q: Vec<f64>,
    pub potential: Vec<f64>,
    pub spec: Spectrum,
    pub iterations: usize,
}
pub(super) fn solve(
    cluster: &Cluster,
    external: &[f64],
    options: &ElectronicOptions,
) -> Result<LocalState> {
    let mut q = vec![0.; cluster.shells.len()];
    let mut mixer = BroydenMixer::new(
        q.len(),
        options.scc_broyden_size.max(2),
        options.mixing.clamp(0.01, 1.),
    );
    let mut rms = f64::INFINITY;
    for iterations in 1..=options.max_scc {
        let potential = cluster.potential(&q, external)?;
        let spec = cluster.spectrum(&potential, options)?;
        let new_q = cluster.shell_q(&spec.density, None, true);
        let residual: Vec<_> = new_q.iter().zip(&q).map(|(a, b)| a - b).collect();
        rms = (residual.iter().map(|x| x * x).sum::<f64>() / q.len() as f64).sqrt();
        if rms < options.charge_tolerance.min(1e-10) {
            return Ok(LocalState {
                q: new_q,
                potential,
                spec,
                iterations,
            });
        }
        q = mixer
            .next(&q, &residual)
            .filter(|q| q.iter().all(|x| x.is_finite() && x.abs() < 20.))
            .unwrap_or_else(|| q.iter().zip(&residual).map(|(a, b)| a + 0.2 * b).collect());
    }
    Err(Gfn1Error::SccNotConverged {
        iterations: options.max_scc,
        rms,
    })
}

/// Cyclic Jacobi rotations preserve the relative accuracy of the small
/// physical diagonal block next to the large, diagonal HOP barriers.
fn relative_jacobi(input: &Matrix) -> Result<EigenDecomposition> {
    let n = input.rows();
    let mut a = input.clone();
    let mut vectors = Matrix::identity(n);
    let mut converged = false;
    for _ in 0..80 {
        let mut maximum = 0.0_f64;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[(p, q)];
                maximum = maximum.max(apq.abs());
                if apq.abs() < 1e-14 {
                    continue;
                }
                let tau = (a[(q, q)] - a[(p, p)]) / (2. * apq);
                let t = if tau >= 0. {
                    1. / (tau + tau.hypot(1.))
                } else {
                    -1. / (-tau + tau.hypot(1.))
                };
                let c = 1. / (1. + t * t).sqrt();
                let s = t * c;
                a[(p, p)] -= t * apq;
                a[(q, q)] += t * apq;
                a[(p, q)] = 0.;
                a[(q, p)] = 0.;
                for k in 0..n {
                    if k != p && k != q {
                        let x = a[(k, p)];
                        let y = a[(k, q)];
                        a[(k, p)] = c * x - s * y;
                        a[(p, k)] = a[(k, p)];
                        a[(k, q)] = s * x + c * y;
                        a[(q, k)] = a[(k, q)];
                    }
                    let x = vectors[(k, p)];
                    let y = vectors[(k, q)];
                    vectors[(k, p)] = c * x - s * y;
                    vectors[(k, q)] = s * x + c * y;
                }
            }
        }
        if maximum < 1e-13 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Gfn1Error::LinearAlgebra(
            "HOP chart Jacobi eigensolver did not converge".into(),
        ));
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].total_cmp(&a[(j, j)]));
    let values = order.iter().map(|&i| a[(i, i)]).collect();
    let mut sorted = Matrix::zeros(n, n);
    for (j, &index) in order.iter().enumerate() {
        for i in 0..n {
            sorted[(i, j)] = vectors[(i, index)];
        }
    }
    Ok(EigenDecomposition {
        values,
        vectors: sorted,
    })
}

pub(super) struct Screened<'a> {
    cluster: &'a Cluster,
    state: &'a LocalState,
    kernel: Matrix,
    lu: DenseLu,
}
impl<'a> Screened<'a> {
    pub fn new(cluster: &'a Cluster, state: &'a LocalState) -> Result<Self> {
        let kernel = cluster.kernel(&state.q);
        let n = state.q.len();
        let mut matrix = Matrix::identity(n);
        let zero = Matrix::zeros(cluster.aos.len(), cluster.aos.len());
        for j in 0..n {
            let potential: Vec<_> = (0..n).map(|i| kernel[(i, j)]).collect();
            let (dp, _) = state.spec.canonical_response(
                &cluster.shift(&potential, &cluster.s),
                &zero,
                &cluster.s,
            )?;
            let dq = cluster.shell_q(&dp, None, false);
            for i in 0..n {
                matrix[(i, j)] -= dq[i];
            }
        }
        Ok(Self {
            cluster,
            state,
            kernel,
            lu: DenseLu::factor(&matrix)?,
        })
    }
    pub fn response(&self, dh: &Matrix, ds: &Matrix) -> Result<(Vec<f64>, Matrix, Matrix)> {
        let (dp, _) = self
            .state
            .spec
            .canonical_response(dh, ds, &self.cluster.s)?;
        let mut rhs = self.cluster.shell_q(&dp, None, false);
        let metric = self
            .cluster
            .shell_q(&self.state.spec.density, Some(ds), false);
        for (x, y) in rhs.iter_mut().zip(metric) {
            *x += y;
        }
        let dq = self.lu.solve_vec(&rhs)?;
        let v = matrix_vector_product(&self.kernel, &dq)?;
        let mut full = dh.clone();
        add(&mut full, &self.cluster.shift(&v, &self.cluster.s), 1.);
        let (dp, de) = self
            .state
            .spec
            .canonical_response(&full, ds, &self.cluster.s)?;
        Ok((dq, dp, de))
    }
}
