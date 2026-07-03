// SPDX-License-Identifier: GPL-3.0-or-later
//! Classical GFN1 halogen-bond correction.
//!
//! This follows the non-periodic branch of tblite's `classical/halogen.f90`.

use crate::data_tables::atomic_radius_bohr;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pairlist::center_short_range_neighbors;
use crate::system::PeriodicSystem;

const DAMPING: f64 = 0.44;
const RAD_SCALE: f64 = 1.3;
const ALP: f64 = 6.0;
const LJ: f64 = 12.0;
const LJ2: f64 = 0.5 * LJ;
const CUTOFF: f64 = 20.0;
const DIST_EPS: f64 = 1.0e-18;

#[derive(Clone, Debug)]
struct HalogenTriple {
    donor: usize,
    acceptor: usize,
    neighbor: usize,
    acceptor_translation: Vec3,
    neighbor_translation: Vec3,
}

#[derive(Clone, Debug)]
pub struct HalogenResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    /// Derivative with respect to homogeneous strain divided by cell volume.
    /// Present only for periodic systems.
    pub stress: Option<Matrix>,
}

#[derive(Clone, Debug)]
pub struct HalogenHessianResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
    /// Derivative with respect to homogeneous strain divided by cell volume.
    /// Present only for periodic systems.
    pub stress: Option<Matrix>,
}

pub fn halogen_energy(system: &PeriodicSystem) -> Result<f64> {
    Ok(halogen_energy_gradient(system)?.energy)
}

pub fn halogen_energy_gradient(system: &PeriodicSystem) -> Result<HalogenResult> {
    let triples = halogen_triples(system)?;
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); system.atoms.len()];

    for triple in triples {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = RAD_SCALE * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);

        let dxj = image_vector(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
        );
        let dxk = image_vector(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
        );
        let dkj = image_vector(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
        );

        let d2jx = dxj.norm2();
        let d2kx = dxk.norm2();
        let d2jk = dkj.norm2();
        if d2jx <= DIST_EPS || d2kx <= DIST_EPS {
            continue;
        }
        let rjx = d2jx.sqrt() + DIST_EPS;
        let rkx = d2kx.sqrt() + DIST_EPS;
        let xy = (d2kx * d2jx).sqrt();
        if xy <= DIST_EPS {
            continue;
        }
        let term = (d2kx + d2jx - d2jk) / xy;
        let angle_base = 0.5 - 0.25 * term;
        let aterm = angle_base.powf(ALP);

        let t13 = r0jx / rjx;
        let t14_lj = t13.powf(LJ);
        let term_energy = aterm * cc * (t14_lj - DAMPING * t13.powf(LJ2)) / (1.0 + t14_lj);
        energy += term_energy;

        let t14 = (r0jx / rjx).powf(LJ2);
        let numerator = t14 * t14 - DAMPING * t14;
        let denominator = 1.0 + t14 * t14;
        let term_lj = numerator / denominator;

        let mut dtermlj = 2.0 * LJ2 * numerator * t14 * t14 / (rjx * denominator * denominator);
        dtermlj += LJ2 * t14 * (DAMPING - 2.0 * t14) / (rjx * denominator);
        dtermlj *= aterm * cc / rjx;
        gradient[acceptor] += dxj * dtermlj;
        gradient[donor] -= dxj * dtermlj;

        let prefactor = -0.25 * ALP * angle_base.powf(ALP - 1.0) * cc * term_lj;
        let dcosterm_jx = (2.0 / rkx - term / rjx) * prefactor / rjx;
        gradient[acceptor] += dxj * dcosterm_jx;
        gradient[donor] -= dxj * dcosterm_jx;

        let dcosterm_kx = (2.0 / rjx - term / rkx) * prefactor / rkx;
        gradient[neighbor] += dxk * dcosterm_kx;
        gradient[donor] -= dxk * dcosterm_kx;

        let dcosterm_jk = 2.0 * prefactor / xy;
        gradient[acceptor] -= dkj * dcosterm_jk;
        gradient[neighbor] += dkj * dcosterm_jk;
    }

    let stress = halogen_stress(system)?;
    Ok(HalogenResult {
        energy,
        gradient,
        stress,
    })
}

pub fn halogen_energy_gradient_hessian(system: &PeriodicSystem) -> Result<HalogenHessianResult> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut total = Jet2::constant(0.0, ndof);

    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = RAD_SCALE * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet_vec_image_sub(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet_vec_image_sub(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet_vec_image_sub(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term_energy) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, ndof) {
            total = total.add(&term_energy);
        }
    }

    let mut gradient = vec![Vec3::zero(); nat];
    for (dof, &value) in total.gradient.iter().enumerate() {
        let atom = dof / 3;
        match dof % 3 {
            0 => gradient[atom].x = value,
            1 => gradient[atom].y = value,
            _ => gradient[atom].z = value,
        }
    }
    let hessian = Matrix::from_vec(ndof, ndof, total.hessian)?;
    let stress = halogen_stress(system)?;
    Ok(HalogenHessianResult {
        energy: total.value,
        gradient,
        hessian,
        stress,
    })
}

pub fn halogen_stress(system: &PeriodicSystem) -> Result<Option<Matrix>> {
    let Some(lattice) = system.lattice.as_ref() else {
        return Ok(None);
    };
    let ndof = 9;
    let mut total = Jet2::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = RAD_SCALE * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = strain_vector_jets(
            image_vector(
                system,
                acceptor,
                triple.acceptor_translation,
                donor,
                Vec3::zero(),
            ),
            ndof,
        );
        let dxk = strain_vector_jets(
            image_vector(
                system,
                neighbor,
                triple.neighbor_translation,
                donor,
                Vec3::zero(),
            ),
            ndof,
        );
        let dkj = strain_vector_jets(
            image_vector(
                system,
                acceptor,
                triple.acceptor_translation,
                neighbor,
                triple.neighbor_translation,
            ),
            ndof,
        );
        if let Some(term_energy) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, ndof) {
            total = total.add(&term_energy);
        }
    }
    let inv_volume = 1.0 / lattice.volume();
    let mut stress = Matrix::zeros(3, 3);
    for a in 0..3 {
        for b in 0..3 {
            stress[(a, b)] = total.gradient[3 * a + b] * inv_volume;
        }
    }
    Ok(Some(stress))
}

/// Analytic third Cartesian derivative `T[a][b][c] = ∂³E_halogen/∂R_a∂R_b∂R_c`, returned as
/// `ndof` slabs (one `ndof×ndof` matrix per third index). The halogen correction is a smooth
/// classical 3-body function with no electronic response, so it is obtained by third-order
/// forward AD ([`Jet3`]) of the same per-triple energy used for the gradient/Hessian, and
/// FD-validates in isolation against [`halogen_energy_gradient_hessian`].
pub fn halogen_third_derivative(system: &PeriodicSystem) -> Result<Vec<Matrix>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut total = Jet3::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = RAD_SCALE * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet3_image_sub(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet3_image_sub(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet3_image_sub(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term) = halogen_term_energy_jet3(&dxj, &dxk, &dkj, r0jx, cc, ndof) {
            total = total.add(&term);
        }
    }
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    for a in 0..ndof {
        for b in 0..ndof {
            for c in 0..ndof {
                tensor[c][(a, b)] = total.third[(a * ndof + b) * ndof + c];
            }
        }
    }
    Ok(tensor)
}

#[derive(Clone, Debug)]
struct Jet2 {
    value: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
}

impl Jet2 {
    fn constant(value: f64, ndof: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; ndof],
            hessian: vec![0.0; ndof * ndof],
        }
    }

    fn variable(value: f64, ndof: usize, dof: usize) -> Self {
        let mut out = Self::constant(value, ndof);
        out.gradient[dof] = 1.0;
        out
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::constant(self.value + rhs.value, self.gradient.len());
        for i in 0..self.gradient.len() {
            out.gradient[i] = self.gradient[i] + rhs.gradient[i];
        }
        for i in 0..self.hessian.len() {
            out.hessian[i] = self.hessian[i] + rhs.hessian[i];
        }
        out
    }

    fn add_scalar(&self, rhs: f64) -> Self {
        let mut out = self.clone();
        out.value += rhs;
        out
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    fn scale(&self, scale: f64) -> Self {
        let mut out = Self::constant(self.value * scale, self.gradient.len());
        for i in 0..self.gradient.len() {
            out.gradient[i] = self.gradient[i] * scale;
        }
        for i in 0..self.hessian.len() {
            out.hessian[i] = self.hessian[i] * scale;
        }
        out
    }

    fn mul(&self, rhs: &Self) -> Self {
        let n = self.gradient.len();
        let mut out = Self::constant(self.value * rhs.value, n);
        for i in 0..n {
            out.gradient[i] = self.gradient[i] * rhs.value + rhs.gradient[i] * self.value;
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] = self.hessian[i * n + j] * rhs.value
                    + rhs.hessian[i * n + j] * self.value
                    + self.gradient[i] * rhs.gradient[j]
                    + rhs.gradient[i] * self.gradient[j];
            }
        }
        out
    }

    fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    fn powf(&self, power: f64) -> Self {
        let n = self.gradient.len();
        let value = self.value.powf(power);
        let first = power * self.value.powf(power - 1.0);
        let second = power * (power - 1.0) * self.value.powf(power - 2.0);
        let mut out = Self::constant(value, n);
        for i in 0..n {
            out.gradient[i] = first * self.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] =
                    first * self.hessian[i * n + j] + second * self.gradient[i] * self.gradient[j];
            }
        }
        out
    }
}

/// Third-order forward-AD dual number: value, gradient, Hessian (`n×n`), and third
/// derivative (`n×n×n`, index `(i·n+j)·n+k`). Mirrors [`Jet2`] one order higher.
#[derive(Clone, Debug)]
struct Jet3 {
    value: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
    third: Vec<f64>,
    n: usize,
}

impl Jet3 {
    fn constant(value: f64, n: usize) -> Self {
        Self {
            value,
            gradient: vec![0.0; n],
            hessian: vec![0.0; n * n],
            third: vec![0.0; n * n * n],
            n,
        }
    }

    fn variable(value: f64, n: usize, dof: usize) -> Self {
        let mut out = Self::constant(value, n);
        out.gradient[dof] = 1.0;
        out
    }

    fn add(&self, rhs: &Self) -> Self {
        let n = self.n;
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

    fn add_scalar(&self, rhs: f64) -> Self {
        let mut out = self.clone();
        out.value += rhs;
        out
    }

    fn scale(&self, s: f64) -> Self {
        let n = self.n;
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

    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-1.0))
    }

    fn mul(&self, rhs: &Self) -> Self {
        let n = self.n;
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

    fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.powf(-1.0))
    }

    fn powf(&self, power: f64) -> Self {
        let n = self.n;
        let v = self.value;
        let phi1 = power * v.powf(power - 1.0);
        let phi2 = power * (power - 1.0) * v.powf(power - 2.0);
        let phi3 = power * (power - 1.0) * (power - 2.0) * v.powf(power - 3.0);
        let mut out = Self::constant(v.powf(power), n);
        for i in 0..n {
            out.gradient[i] = phi1 * self.gradient[i];
        }
        for i in 0..n {
            for j in 0..n {
                out.hessian[i * n + j] =
                    phi2 * self.gradient[i] * self.gradient[j] + phi1 * self.hessian[i * n + j];
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
}

fn jet3_image_sub(
    system: &PeriodicSystem,
    lhs: usize,
    lhs_translation: Vec3,
    rhs: usize,
    rhs_translation: Vec3,
    n: usize,
) -> [Jet3; 3] {
    let lp = system.atoms[lhs].position + lhs_translation;
    let rp = system.atoms[rhs].position + rhs_translation;
    [
        Jet3::variable(lp.x, n, 3 * lhs).sub(&Jet3::variable(rp.x, n, 3 * rhs)),
        Jet3::variable(lp.y, n, 3 * lhs + 1).sub(&Jet3::variable(rp.y, n, 3 * rhs + 1)),
        Jet3::variable(lp.z, n, 3 * lhs + 2).sub(&Jet3::variable(rp.z, n, 3 * rhs + 2)),
    ]
}

fn jet3_dot(lhs: &[Jet3; 3], rhs: &[Jet3; 3]) -> Jet3 {
    lhs[0]
        .mul(&rhs[0])
        .add(&lhs[1].mul(&rhs[1]))
        .add(&lhs[2].mul(&rhs[2]))
}

fn halogen_term_energy_jet3(
    dxj: &[Jet3; 3],
    dxk: &[Jet3; 3],
    dkj: &[Jet3; 3],
    r0jx: f64,
    cc: f64,
    n: usize,
) -> Option<Jet3> {
    let d2jx = jet3_dot(dxj, dxj);
    let d2kx = jet3_dot(dxk, dxk);
    let d2jk = jet3_dot(dkj, dkj);
    if d2jx.value <= DIST_EPS || d2kx.value <= DIST_EPS {
        return None;
    }
    let rjx = d2jx.powf(0.5).add_scalar(DIST_EPS);
    let xy = d2kx.mul(&d2jx).powf(0.5);
    if xy.value <= DIST_EPS {
        return None;
    }
    let term = d2kx.add(&d2jx).sub(&d2jk).div(&xy);
    let angle_base = Jet3::constant(0.5, n).sub(&term.scale(0.25));
    let aterm = angle_base.powf(ALP);
    let t13 = Jet3::constant(r0jx, n).div(&rjx);
    let t14_lj = t13.powf(LJ);
    Some(
        aterm
            .scale(cc)
            .mul(&t14_lj.sub(&t13.powf(LJ2).scale(DAMPING)))
            .div(&Jet3::constant(1.0, n).add(&t14_lj)),
    )
}

fn halogen_term_energy_jet(
    dxj: &[Jet2; 3],
    dxk: &[Jet2; 3],
    dkj: &[Jet2; 3],
    r0jx: f64,
    cc: f64,
    ndof: usize,
) -> Option<Jet2> {
    let d2jx = jet_dot(dxj, dxj);
    let d2kx = jet_dot(dxk, dxk);
    let d2jk = jet_dot(dkj, dkj);
    if d2jx.value <= DIST_EPS || d2kx.value <= DIST_EPS {
        return None;
    }
    let rjx = d2jx.powf(0.5).add_scalar(DIST_EPS);
    let xy = d2kx.mul(&d2jx).powf(0.5);
    if xy.value <= DIST_EPS {
        return None;
    }
    let term = d2kx.add(&d2jx).sub(&d2jk).div(&xy);
    let angle_base = Jet2::constant(0.5, ndof).sub(&term.scale(0.25));
    let aterm = angle_base.powf(ALP);
    let t13 = Jet2::constant(r0jx, ndof).div(&rjx);
    let t14_lj = t13.powf(LJ);
    Some(
        aterm
            .scale(cc)
            .mul(&t14_lj.sub(&t13.powf(LJ2).scale(DAMPING)))
            .div(&Jet2::constant(1.0, ndof).add(&t14_lj)),
    )
}

fn jet_vec_image_sub(
    system: &PeriodicSystem,
    lhs: usize,
    lhs_translation: Vec3,
    rhs: usize,
    rhs_translation: Vec3,
    ndof: usize,
) -> [Jet2; 3] {
    [
        Jet2::variable(
            system.atoms[lhs].position.x + lhs_translation.x,
            ndof,
            3 * lhs,
        )
        .sub(&Jet2::variable(
            system.atoms[rhs].position.x + rhs_translation.x,
            ndof,
            3 * rhs,
        )),
        Jet2::variable(
            system.atoms[lhs].position.y + lhs_translation.y,
            ndof,
            3 * lhs + 1,
        )
        .sub(&Jet2::variable(
            system.atoms[rhs].position.y + rhs_translation.y,
            ndof,
            3 * rhs + 1,
        )),
        Jet2::variable(
            system.atoms[lhs].position.z + lhs_translation.z,
            ndof,
            3 * lhs + 2,
        )
        .sub(&Jet2::variable(
            system.atoms[rhs].position.z + rhs_translation.z,
            ndof,
            3 * rhs + 2,
        )),
    ]
}

fn strain_vector_jets(vector: Vec3, ndof: usize) -> [Jet2; 3] {
    let components = vector.to_array();
    let mut out = [
        Jet2::constant(vector.x, ndof),
        Jet2::constant(vector.y, ndof),
        Jet2::constant(vector.z, ndof),
    ];
    for row in 0..3 {
        for col in 0..3 {
            out[row].gradient[3 * row + col] = components[col];
        }
    }
    out
}

fn jet_dot(lhs: &[Jet2; 3], rhs: &[Jet2; 3]) -> Jet2 {
    lhs[0]
        .mul(&rhs[0])
        .add(&lhs[1].mul(&rhs[1]))
        .add(&lhs[2].mul(&rhs[2]))
}

fn halogen_triples(system: &PeriodicSystem) -> Result<Vec<HalogenTriple>> {
    let mut triples = Vec::new();
    for (donor, atom) in system.atoms.iter().enumerate() {
        if !is_halogen(atom.z) {
            continue;
        }
        let neighbors = center_short_range_neighbors(system, donor, CUTOFF)?;
        let Some(neighbor) = nearest_neighbor(donor, &neighbors) else {
            continue;
        };
        for acceptor_pair in neighbors {
            let acceptor = acceptor_pair.j;
            if !is_acceptor(system.atoms[acceptor].z) {
                continue;
            }
            triples.push(HalogenTriple {
                donor,
                acceptor,
                neighbor: neighbor.j,
                acceptor_translation: acceptor_pair.translation,
                neighbor_translation: neighbor.translation,
            });
        }
    }
    Ok(triples)
}

fn nearest_neighbor(
    center: usize,
    neighbors: &[crate::pairlist::ShortRangePair],
) -> Option<crate::pairlist::ShortRangePair> {
    neighbors
        .iter()
        .copied()
        .filter(|pair| pair.j != center)
        .min_by(|a, b| a.r2.partial_cmp(&b.r2).unwrap_or(std::cmp::Ordering::Equal))
}

fn image_vector(
    system: &PeriodicSystem,
    lhs: usize,
    lhs_translation: Vec3,
    rhs: usize,
    rhs_translation: Vec3,
) -> Vec3 {
    system.atoms[lhs].position + lhs_translation - system.atoms[rhs].position - rhs_translation
}

fn is_halogen(z: u8) -> bool {
    matches!(z, 17 | 35 | 53 | 85)
}

fn is_acceptor(z: u8) -> bool {
    matches!(z, 7 | 8 | 15 | 16)
}

fn bond_strength(z: u8) -> f64 {
    match z {
        35 => 0.381_742 * 0.1,
        53 => 0.321_944 * 0.1,
        85 => 0.220_000 * 0.1,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::{halogen_energy, halogen_energy_gradient, halogen_energy_gradient_hessian};
    use crate::lattice::Lattice;
    use crate::math::{Mat3, Vec3};
    use crate::system::PeriodicSystem;

    #[test]
    fn fluorine_and_chlorine_have_no_gfn1_halogen_correction() {
        let system =
            PeriodicSystem::from_xyz_str("2\nHF\nH 0.0 0.0 0.0\nF 0.917 0.0 0.0\n", 0.0, false)
                .unwrap();
        assert_eq!(halogen_energy(&system).unwrap(), 0.0);

        let system = PeriodicSystem::from_xyz_str(
            "4\nCCl...O\nC 0.0 0.0 0.0\nCl 1.8 0.0 0.0\nO 4.3 0.2 0.0\nH 4.7 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(halogen_energy(&system).unwrap(), 0.0);
    }

    #[test]
    fn bromine_halogen_correction_is_active() {
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        assert!(halogen_energy(&system).unwrap() < -1.0e-6);
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = halogen_energy_gradient(&system).unwrap();
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd =
                    (halogen_energy(&plus).unwrap() - halogen_energy(&minus).unwrap()) / (2.0 * h);
                let an = match component {
                    0 => result.gradient[atom].x,
                    1 => result.gradient[atom].y,
                    _ => result.gradient[atom].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-8,
                    "atom {atom} component {component}: analytic {an} FD {fd}"
                );
            }
        }
    }

    #[test]
    fn analytic_hessian_matches_gradient_finite_difference() {
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = halogen_energy_gradient_hessian(&system).unwrap();
        let grad = halogen_energy_gradient(&system).unwrap();
        for atom in 0..system.atoms.len() {
            assert!((result.gradient[atom].x - grad.gradient[atom].x).abs() < 1.0e-10);
            assert!((result.gradient[atom].y - grad.gradient[atom].y).abs() < 1.0e-10);
            assert!((result.gradient[atom].z - grad.gradient[atom].z).abs() < 1.0e-10);
        }
        let h = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, col / 3, col % 3, h);
            shift(&mut minus, col / 3, col % 3, -h);
            let gp = halogen_energy_gradient(&plus).unwrap().gradient;
            let gm = halogen_energy_gradient(&minus).unwrap().gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * h);
                max_delta = max_delta.max((result.hessian[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-7,
            "halogen Hessian finite-difference max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn halogen_third_derivative_matches_hessian_finite_difference() {
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let third = super::halogen_third_derivative(&system).unwrap();
        let h = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, slab / 3, slab % 3, h);
            shift(&mut minus, slab / 3, slab % 3, -h);
            let hp = halogen_energy_gradient_hessian(&plus).unwrap().hessian;
            let hm = halogen_energy_gradient_hessian(&minus).unwrap().hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * h);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "halogen third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn periodic_bromine_halogen_image_correction_is_active() {
        let system = periodic_halogen_system();
        let energy = halogen_energy(&system).unwrap();
        assert!(energy.is_finite());
        assert!(
            energy.abs() > 1.0e-9,
            "periodic halogen correction should include image acceptors"
        );
    }

    #[test]
    fn periodic_gradient_matches_finite_difference() {
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient(&system).unwrap();
        assert!(result.stress.is_some());
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd =
                    (halogen_energy(&plus).unwrap() - halogen_energy(&minus).unwrap()) / (2.0 * h);
                let an = match component {
                    0 => result.gradient[atom].x,
                    1 => result.gradient[atom].y,
                    _ => result.gradient[atom].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-8,
                    "periodic atom {atom} component {component}: analytic {an} FD {fd}"
                );
            }
        }
    }

    #[test]
    fn periodic_hessian_matches_gradient_finite_difference() {
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient_hessian(&system).unwrap();
        assert!(result.stress.is_some());
        let grad = halogen_energy_gradient(&system).unwrap();
        for atom in 0..system.atoms.len() {
            assert!((result.gradient[atom].x - grad.gradient[atom].x).abs() < 1.0e-10);
            assert!((result.gradient[atom].y - grad.gradient[atom].y).abs() < 1.0e-10);
            assert!((result.gradient[atom].z - grad.gradient[atom].z).abs() < 1.0e-10);
        }
        let h = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, col / 3, col % 3, h);
            shift(&mut minus, col / 3, col % 3, -h);
            let gp = halogen_energy_gradient(&plus).unwrap().gradient;
            let gm = halogen_energy_gradient(&minus).unwrap().gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * h);
                max_delta = max_delta.max((result.hessian[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-7,
            "periodic halogen Hessian finite-difference max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn periodic_stress_matches_strain_finite_difference() {
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient(&system).unwrap();
        let stress = result.stress.as_ref().unwrap();
        let volume = system.lattice.as_ref().unwrap().volume();
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let plus = strained_system(&system, row, col, h);
                let minus = strained_system(&system, row, col, -h);
                let fd = (halogen_energy(&plus).unwrap() - halogen_energy(&minus).unwrap())
                    / (2.0 * h * volume);
                max_delta = max_delta.max((stress[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-8,
            "periodic halogen stress finite-difference max delta {max_delta:.3e}"
        );
    }

    fn shift(system: &mut PeriodicSystem, atom: usize, component: usize, delta: f64) {
        match component {
            0 => system.atoms[atom].position.x += delta,
            1 => system.atoms[atom].position.y += delta,
            _ => system.atoms[atom].position.z += delta,
        }
    }

    fn component(values: &[crate::math::Vec3], dof: usize) -> f64 {
        let atom = dof / 3;
        match dof % 3 {
            0 => values[atom].x,
            1 => values[atom].y,
            _ => values[atom].z,
        }
    }

    fn periodic_halogen_system() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "4\nLattice=\"8 0 0 0 8 0 0 0 8\" pbc=\"T T T\"\n\
             C 5.000000 0.000000 0.000000\n\
             Br 6.900000 0.000000 0.000000\n\
             O 1.400000 0.200000 0.000000\n\
             H 1.800000 0.800000 0.000000\n",
            0.0,
            false,
        )
        .unwrap()
    }

    fn strained_system(
        system: &PeriodicSystem,
        row: usize,
        col: usize,
        delta: f64,
    ) -> PeriodicSystem {
        let mut out = system.clone();
        for atom in &mut out.atoms {
            atom.position = strain_vec(atom.position, row, col, delta);
        }
        if let Some(lattice) = system.lattice {
            let cell = Mat3::from_columns(
                strain_vec(lattice.cell.col[0], row, col, delta),
                strain_vec(lattice.cell.col[1], row, col, delta),
                strain_vec(lattice.cell.col[2], row, col, delta),
            );
            out.lattice = Some(Lattice::new(cell, lattice.periodic).unwrap());
        }
        out
    }

    fn strain_vec(vector: Vec3, row: usize, col: usize, delta: f64) -> Vec3 {
        let mut out = vector;
        let components = vector.to_array();
        match row {
            0 => out.x += delta * components[col],
            1 => out.y += delta * components[col],
            _ => out.z += delta * components[col],
        }
        out
    }
}
