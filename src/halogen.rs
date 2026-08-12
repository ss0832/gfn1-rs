// SPDX-License-Identifier: GPL-3.0-or-later
//! Classical GFN1 halogen-bond correction.
//!
//! This follows the non-periodic branch of tblite's `classical/halogen.f90`.
//!
//! The model values come from the GFN1 parameter file: the damping fraction is
//! the global `xbdamp`, the radius scaling is the global `xbrad`, and the
//! per-element bond strength is the element `CXB` entry (times the 0.1 GFN1
//! internal scaling). Before v0.5.0 these were hardcoded copies of the builtin
//! parametrization, so edited parameter files were silently ignored and
//! parameter derivatives w.r.t. them were zero; fixed in v0.5.0. The remaining
//! constants below ([`ALP`], [`LJ`], [`CUTOFF`], [`DIST_EPS`]) are model
//! structure, not parameter-file entries.

use crate::data_tables::atomic_radius_bohr;
use crate::dispersion::MAX_FOURTH_DERIVATIVE_NDOF;
use crate::error::{Gfn1Error, Result};
use crate::jets::{Jet1, Jet2, Jet3, Jet4};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pairlist::center_short_range_neighbors;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

const ALP: f64 = 6.0;
const LJ: f64 = 12.0;
const LJ2: f64 = 0.5 * LJ;
/// Halogen-bond pair cutoff in Bohr; 20 bohr follows tblite's `classical/halogen.f90`.
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

pub fn halogen_energy(system: &PeriodicSystem, params: &Gfn1Parameters) -> Result<f64> {
    Ok(halogen_energy_gradient(system, params)?.energy)
}

pub fn halogen_energy_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<HalogenResult> {
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let triples = halogen_triples(system)?;
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); system.atoms.len()];

    for triple in triples {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);

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
        let term_energy = aterm * cc * (t14_lj - damping * t13.powf(LJ2)) / (1.0 + t14_lj);
        energy += term_energy;

        let t14 = (r0jx / rjx).powf(LJ2);
        let numerator = t14 * t14 - damping * t14;
        let denominator = 1.0 + t14 * t14;
        let term_lj = numerator / denominator;

        let mut dtermlj = 2.0 * LJ2 * numerator * t14 * t14 / (rjx * denominator * denominator);
        dtermlj += LJ2 * t14 * (damping - 2.0 * t14) / (rjx * denominator);
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

    let stress = halogen_stress(system, params)?;
    Ok(HalogenResult {
        energy,
        gradient,
        stress,
    })
}

pub fn halogen_energy_gradient_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<HalogenHessianResult> {
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut total = Jet2::constant(0.0, ndof);

    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet_image_sub::<Jet2>(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet_image_sub::<Jet2>(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet_image_sub::<Jet2>(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term_energy) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, damping, ndof)
        {
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
    let stress = halogen_stress(system, params)?;
    Ok(HalogenHessianResult {
        energy: total.value,
        gradient,
        hessian,
        stress,
    })
}

pub fn halogen_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<Option<Matrix>> {
    let Some(lattice) = system.lattice.as_ref() else {
        return Ok(None);
    };
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let ndof = 9;
    let mut total = Jet2::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
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
        if let Some(term_energy) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, damping, ndof)
        {
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
pub fn halogen_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<Vec<Matrix>> {
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut total = Jet3::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet_image_sub::<Jet3>(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet_image_sub::<Jet3>(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet_image_sub::<Jet3>(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, damping, ndof) {
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

/// Result of the analytic halogen-bond **fourth** derivative.
#[derive(Clone, Debug)]
pub struct HalogenFourthResult {
    pub energy: f64,
    /// Dense `ndof⁴` fourth derivative `∂⁴E_halogen/∂R⁴`, row-major
    /// `((a·ndof+b)·ndof+c)·ndof+d`.
    pub fourth: Vec<f64>,
    pub ndof: usize,
}

/// Analytic fourth Cartesian derivative `∂⁴E_halogen/∂R⁴` (quartic force constants), via the
/// [`Jet4`] promotion of the per-triple energy that already feeds the gradient, Hessian and
/// [`halogen_third_derivative`]. Like the third derivative this is a purely geometric,
/// response-free block of a smooth classical 3-body function, so it FD-isolates against
/// [`halogen_third_derivative`] and satisfies the acoustic sum rule exactly.
///
/// Seeding matches the lower orders: full `3·nat` coordinate space rather than per-triple
/// 9-DOF jets, which keeps the tensor assembly a plain accumulation. A full-space [`Jet4`]
/// stores `ndof⁴` doubles, so the same [`MAX_FOURTH_DERIVATIVE_NDOF`] cap the dispersion path
/// uses guards this one.
pub fn halogen_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<HalogenFourthResult> {
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if ndof > MAX_FOURTH_DERIVATIVE_NDOF {
        let per_jet_mb = (ndof as f64).powi(4) * 8.0 / (1024.0 * 1024.0);
        return Err(Gfn1Error::InvalidInput(format!(
            "analytic halogen-bond fourth derivative is capped at {MAX_FOURTH_DERIVATIVE_NDOF} \
             degrees of freedom ({} atoms); got {ndof} ({nat} atoms). A full-space Jet4 stores \
             ndof^4 doubles ({per_jet_mb:.0} MB each at this size) and the per-triple assembly \
             keeps several of them alive at once. Use a smaller system or raise \
             `MAX_FOURTH_DERIVATIVE_NDOF` deliberately",
            MAX_FOURTH_DERIVATIVE_NDOF / 3
        )));
    }
    let mut total = Jet4::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet_image_sub::<Jet4>(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet_image_sub::<Jet4>(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet_image_sub::<Jet4>(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, damping, ndof) {
            total = total.add(&term);
        }
    }
    Ok(HalogenFourthResult {
        energy: total.value,
        fourth: total.fourth,
        ndof,
    })
}

/// **Directional** analytic halogen-bond fourth derivative
/// `e⁗[v] = Σ_abcd v_a v_b v_c v_d ∂⁴E_halogen/∂R_a∂R_b∂R_c∂R_d` — the same per-triple energy
/// [`halogen_fourth_derivative`] differentiates, instantiated on the univariate [`Jet1`].
///
/// A directional fourth derivative is the 4th Taylor coefficient of `E(R + t·v)`, so one
/// differentiation variable suffices: each jet costs five doubles instead of `ndof⁴`, and this
/// route therefore carries **no** [`MAX_FOURTH_DERIVATIVE_NDOF`] cap. Gated against
/// `contract_vvvv` of the full tensor on systems small enough for both by
/// `halogen_fourth_directional_matches_full_tensor`.
pub fn halogen_fourth_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    v: &[f64],
) -> Result<f64> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "halogen_fourth_directional: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let damping = params.required_global("xbdamp")?;
    let rad_scale = params.required_global("xbrad")?;
    let _direction = crate::jets::DirectionScope::install(v);
    let mut total = <Jet1 as HalJet>::constant(0.0, ndof);
    for triple in halogen_triples(system)? {
        let donor = triple.donor;
        let acceptor = triple.acceptor;
        let neighbor = triple.neighbor;
        let xzp = system.atoms[donor].z;
        let jzp = system.atoms[acceptor].z;
        let cc = bond_strength(params, xzp);
        if cc == 0.0 {
            continue;
        }
        let r0jx = rad_scale * (atomic_radius_bohr(xzp)? + atomic_radius_bohr(jzp)?);
        let dxj = jet_image_sub::<Jet1>(
            system,
            acceptor,
            triple.acceptor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dxk = jet_image_sub::<Jet1>(
            system,
            neighbor,
            triple.neighbor_translation,
            donor,
            Vec3::zero(),
            ndof,
        );
        let dkj = jet_image_sub::<Jet1>(
            system,
            acceptor,
            triple.acceptor_translation,
            neighbor,
            triple.neighbor_translation,
            ndof,
        );
        if let Some(term) = halogen_term_energy_jet(&dxj, &dxk, &dkj, r0jx, cc, damping, ndof) {
            total = HalJet::add(&total, &term);
        }
    }
    Ok(total.d4)
}

// --- Order-generic forward-AD plumbing -----------------------------------------------------
//
// The per-triple halogen energy (angular factor x LJ-like radial factor) is written **once**
// against this op set and instantiated at second ([`Jet2`]: Hessian and strain derivatives),
// third ([`Jet3`]) and fourth ([`Jet4`]) order, so every order differentiates the *same*
// expression through the *same* operation sequence. The jets themselves are the shared
// [`crate::jets`] implementations; halogen carried private copies before v0.5.0.

/// The shared-jet operations the halogen term energy needs.
trait HalJet: Clone {
    fn constant(value: f64, n: usize) -> Self;
    fn variable(value: f64, n: usize, dof: usize) -> Self;
    fn value(&self) -> f64;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn add_scalar(&self, rhs: f64) -> Self;
    fn scale(&self, s: f64) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
    fn powf(&self, p: f64) -> Self;
}

macro_rules! impl_hal_jet {
    ($ty:ty) => {
        impl HalJet for $ty {
            #[inline]
            fn constant(value: f64, n: usize) -> Self {
                <$ty>::constant(value, n)
            }
            #[inline]
            fn variable(value: f64, n: usize, dof: usize) -> Self {
                <$ty>::variable(value, n, dof)
            }
            #[inline]
            fn value(&self) -> f64 {
                self.value
            }
            #[inline]
            fn add(&self, rhs: &Self) -> Self {
                <$ty>::add(self, rhs)
            }
            #[inline]
            fn sub(&self, rhs: &Self) -> Self {
                <$ty>::sub(self, rhs)
            }
            #[inline]
            fn add_scalar(&self, rhs: f64) -> Self {
                <$ty>::add_scalar(self, rhs)
            }
            #[inline]
            fn scale(&self, s: f64) -> Self {
                <$ty>::scale(self, s)
            }
            #[inline]
            fn mul(&self, rhs: &Self) -> Self {
                <$ty>::mul(self, rhs)
            }
            #[inline]
            fn div(&self, rhs: &Self) -> Self {
                <$ty>::div(self, rhs)
            }
            #[inline]
            fn powf(&self, p: f64) -> Self {
                <$ty>::powf(self, p)
            }
        }
    };
}

impl_hal_jet!(Jet2);
impl_hal_jet!(Jet3);
impl_hal_jet!(Jet4);

/// The DIRECTIONAL instantiation of the halogen op set: [`Jet1`] carries the univariate Taylor of
/// `E(R + t·v)`, so [`halogen_term_energy_jet`] — written once, order-generic — yields
/// `e⁗[v]` at `O(1)` storage per jet instead of `O(ndof⁴)`.
///
/// `variable(value, _n, dof)` seeds `dR_dof/dt = v_dof` from the direction installed by
/// [`crate::jets::DirectionScope`], which is the ONLY place geometry enters the per-triple
/// expression.
impl HalJet for Jet1 {
    #[inline]
    fn constant(value: f64, _n: usize) -> Self {
        Jet1::constant(value)
    }
    #[inline]
    fn variable(value: f64, _n: usize, dof: usize) -> Self {
        Jet1::variable(value, dof)
    }
    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        Jet1::add(self, rhs)
    }
    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        Jet1::sub(self, rhs)
    }
    #[inline]
    fn add_scalar(&self, rhs: f64) -> Self {
        Jet1::add_scalar(self, rhs)
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        Jet1::scale(self, s)
    }
    #[inline]
    fn mul(&self, rhs: &Self) -> Self {
        Jet1::mul(self, rhs)
    }
    #[inline]
    fn div(&self, rhs: &Self) -> Self {
        Jet1::div(self, rhs)
    }
    #[inline]
    fn powf(&self, p: f64) -> Self {
        Jet1::powf(self, p)
    }
}

/// Seeded jets of the (image-shifted) separation `r_lhs - r_rhs`, one per Cartesian component,
/// over the full `n = 3*nat` coordinate space.
fn jet_image_sub<J: HalJet>(
    system: &PeriodicSystem,
    lhs: usize,
    lhs_translation: Vec3,
    rhs: usize,
    rhs_translation: Vec3,
    n: usize,
) -> [J; 3] {
    let lp = system.atoms[lhs].position + lhs_translation;
    let rp = system.atoms[rhs].position + rhs_translation;
    [
        J::variable(lp.x, n, 3 * lhs).sub(&J::variable(rp.x, n, 3 * rhs)),
        J::variable(lp.y, n, 3 * lhs + 1).sub(&J::variable(rp.y, n, 3 * rhs + 1)),
        J::variable(lp.z, n, 3 * lhs + 2).sub(&J::variable(rp.z, n, 3 * rhs + 2)),
    ]
}

fn jet_dot<J: HalJet>(lhs: &[J; 3], rhs: &[J; 3]) -> J {
    lhs[0]
        .mul(&rhs[0])
        .add(&lhs[1].mul(&rhs[1]))
        .add(&lhs[2].mul(&rhs[2]))
}

/// Per-triple halogen-bond energy carried as a jet of whatever order `J` provides. Returns
/// `None` for a degenerate triple (coincident sites), mirroring the closed-form path's guards.
fn halogen_term_energy_jet<J: HalJet>(
    dxj: &[J; 3],
    dxk: &[J; 3],
    dkj: &[J; 3],
    r0jx: f64,
    cc: f64,
    damping: f64,
    n: usize,
) -> Option<J> {
    let d2jx = jet_dot(dxj, dxj);
    let d2kx = jet_dot(dxk, dxk);
    let d2jk = jet_dot(dkj, dkj);
    if d2jx.value() <= DIST_EPS || d2kx.value() <= DIST_EPS {
        return None;
    }
    let rjx = d2jx.powf(0.5).add_scalar(DIST_EPS);
    let xy = d2kx.mul(&d2jx).powf(0.5);
    if xy.value() <= DIST_EPS {
        return None;
    }
    let term = d2kx.add(&d2jx).sub(&d2jk).div(&xy);
    let angle_base = J::constant(0.5, n).sub(&term.scale(0.25));
    let aterm = angle_base.powf(ALP);
    let t13 = J::constant(r0jx, n).div(&rjx);
    let t14_lj = t13.powf(LJ);
    Some(
        aterm
            .scale(cc)
            .mul(&t14_lj.sub(&t13.powf(LJ2).scale(damping)))
            .div(&J::constant(1.0, n).add(&t14_lj)),
    )
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

/// Per-element halogen-bond strength: the parameter file's `CXB` entry times the
/// 0.1 GFN1 internal scaling. Elements without a `CXB` entry (including Cl and F
/// in the official parametrization) get 0.0, which skips the term.
fn bond_strength(params: &Gfn1Parameters, z: u8) -> f64 {
    params
        .elements
        .get(&z)
        .and_then(|e| e.raw.get("CXB"))
        .and_then(|v| v.first())
        .copied()
        .unwrap_or(0.0)
        * 0.1
}

#[cfg(test)]
mod tests {
    use super::{halogen_energy, halogen_energy_gradient, halogen_energy_gradient_hessian};
    use crate::lattice::Lattice;
    use crate::math::{Mat3, Vec3};
    use crate::params::{Gfn1Parameters, ParameterTarget};
    use crate::system::PeriodicSystem;

    fn builtin_params() -> Gfn1Parameters {
        Gfn1Parameters::builtin().unwrap()
    }

    #[test]
    fn fluorine_and_chlorine_have_no_gfn1_halogen_correction() {
        let params = builtin_params();
        let system =
            PeriodicSystem::from_xyz_str("2\nHF\nH 0.0 0.0 0.0\nF 0.917 0.0 0.0\n", 0.0, false)
                .unwrap();
        assert_eq!(halogen_energy(&system, &params).unwrap(), 0.0);

        let system = PeriodicSystem::from_xyz_str(
            "4\nCCl...O\nC 0.0 0.0 0.0\nCl 1.8 0.0 0.0\nO 4.3 0.2 0.0\nH 4.7 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(halogen_energy(&system, &params).unwrap(), 0.0);
    }

    #[test]
    fn bromine_halogen_correction_is_active() {
        let params = builtin_params();
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        assert!(halogen_energy(&system, &params).unwrap() < -1.0e-6);
    }

    fn ch3br_water_system() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "8\nCH3Br...OH2 halogen bond\n\
             C 0.0 0.0 0.0\n\
             Br 0.0 0.0 1.95\n\
             H 1.03 0.0 -0.33\n\
             H -0.515 0.892 -0.33\n\
             H -0.515 -0.892 -0.33\n\
             O 0.0 0.1 4.9\n\
             H 0.76 0.1 5.47\n\
             H -0.76 0.1 5.47\n",
            0.0,
            false,
        )
        .unwrap()
    }

    #[test]
    fn halogen_energy_responds_to_xbdamp() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let base = halogen_energy(&system, &params).unwrap();
        assert!(
            base != 0.0,
            "reference CH3Br...OH2 halogen energy must be nonzero"
        );

        for (target, value) in [
            ("glob:xbdamp", 0.5),
            ("glob:xbrad", 1.5),
            ("elem:35:CXB", 0.5),
        ] {
            let modified = params
                .with_parameter(&ParameterTarget::parse(target).unwrap(), value)
                .unwrap();
            let energy = halogen_energy(&system, &modified).unwrap();
            assert!(
                (energy - base).abs() > 1.0e-10,
                "halogen energy must respond to {target}: base {base} modified {energy}"
            );
        }
    }

    #[test]
    fn halogen_param_derivative_nonzero() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let target = ParameterTarget::parse("glob:xbdamp").unwrap();
        let base = params.parameter_value(&target).unwrap();
        let h = 1.0e-4;
        let plus = halogen_energy(
            &system,
            &params.with_parameter(&target, base + h).unwrap(),
        )
        .unwrap();
        let minus = halogen_energy(
            &system,
            &params.with_parameter(&target, base - h).unwrap(),
        )
        .unwrap();
        let derivative = (plus - minus) / (2.0 * h);
        assert!(
            derivative.abs() > 1.0e-10,
            "dE_halogen/d(xbdamp) must be nonzero, got {derivative}"
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let params = builtin_params();
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = halogen_energy_gradient(&system, &params).unwrap();
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd = (halogen_energy(&plus, &params).unwrap()
                    - halogen_energy(&minus, &params).unwrap())
                    / (2.0 * h);
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
        let params = builtin_params();
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = halogen_energy_gradient_hessian(&system, &params).unwrap();
        let grad = halogen_energy_gradient(&system, &params).unwrap();
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
            let gp = halogen_energy_gradient(&plus, &params).unwrap().gradient;
            let gm = halogen_energy_gradient(&minus, &params).unwrap().gradient;
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
        let params = builtin_params();
        let system = PeriodicSystem::from_xyz_str(
            "4\nCBr...O\nC 0.0 0.0 0.0\nBr 1.9 0.0 0.0\nO 4.6 0.2 0.0\nH 5.0 0.8 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let third = super::halogen_third_derivative(&system, &params).unwrap();
        let h = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, slab / 3, slab % 3, h);
            shift(&mut minus, slab / 3, slab % 3, -h);
            let hp = halogen_energy_gradient_hessian(&plus, &params)
                .unwrap()
                .hessian;
            let hm = halogen_energy_gradient_hessian(&minus, &params)
                .unwrap()
                .hessian;
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

    /// Σ_A Q_{Aα,bcd} over atoms — the fourth-order acoustic sum rule residual. The halogen
    /// correction depends only on interatomic separations, so a rigid translation cannot change
    /// any derivative: this must vanish to numerical precision.
    fn fourth_acoustic_residual(fourth: &[f64], ndof: usize) -> f64 {
        let nat = ndof / 3;
        let mut max = 0.0_f64;
        for alpha in 0..3 {
            for b in 0..ndof {
                for c in 0..ndof {
                    for d in 0..ndof {
                        let sum: f64 = (0..nat)
                            .map(|atom| {
                                fourth[(((3 * atom + alpha) * ndof + b) * ndof + c) * ndof + d]
                            })
                            .sum();
                        max = max.max(sum.abs());
                    }
                }
            }
        }
        max
    }

    /// Largest deviation of the flat `ndof⁴` tensor from full permutation symmetry.
    fn fourth_permutation_residual(fourth: &[f64], n: usize) -> f64 {
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
        let mut max = 0.0_f64;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        let v = fourth[idx(a, b, c, d)];
                        for &(w, x, y, z) in &[
                            (b, a, c, d),
                            (a, c, b, d),
                            (a, b, d, c),
                            (d, c, b, a),
                            (c, d, a, b),
                        ] {
                            max = max.max((v - fourth[idx(w, x, y, z)]).abs());
                        }
                    }
                }
            }
        }
        max
    }

    // FD-fourth gate: the Jet4 promotion must reproduce a central finite difference of the
    // analytic third derivative on a genuine halogen-bonded complex (CH3Br...OH2, 24 DOF).
    #[test]
    fn halogen_fourth_derivative_matches_third_finite_difference() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let analytic = super::halogen_fourth_derivative(&system, &params).unwrap();
        let ndof = analytic.ndof;
        assert_eq!(ndof, 3 * system.atoms.len());
        // The Jet4 value must reproduce the closed-form energy (same expression).
        let energy = halogen_energy(&system, &params).unwrap();
        assert!(
            energy.abs() > 1.0e-9,
            "reference CH3Br...OH2 halogen energy must be nonzero, got {energy}"
        );
        assert!(
            (analytic.energy - energy).abs() < 1.0e-12,
            "Jet4 energy {} vs closed-form energy {energy}",
            analytic.energy
        );
        let h = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for d in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, d / 3, d % 3, h);
            shift(&mut minus, d / 3, d % 3, -h);
            let tp = super::halogen_third_derivative(&plus, &params).unwrap();
            let tm = super::halogen_third_derivative(&minus, &params).unwrap();
            for a in 0..ndof {
                for b in 0..ndof {
                    for c in 0..ndof {
                        let fd = (tp[c][(a, b)] - tm[c][(a, b)]) / (2.0 * h);
                        let an = analytic.fourth[((a * ndof + b) * ndof + c) * ndof + d];
                        max_delta = max_delta.max((an - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "halogen fourth-derivative FD-vs-third max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn halogen_fourth_derivative_acoustic_sum_rule() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let analytic = super::halogen_fourth_derivative(&system, &params).unwrap();
        let max = fourth_acoustic_residual(&analytic.fourth, analytic.ndof);
        assert!(
            max < 1.0e-9,
            "halogen fourth-derivative acoustic sum rule violated: max {max:.3e}"
        );
    }

    /// **The directional 1-D-jet gate.** [`super::halogen_fourth_directional`] must reproduce the
    /// `vvvv` contraction of the full `ndof⁴` tensor to machine precision: both differentiate the
    /// same per-triple expression through the same operation sequence, only the jet width differs
    /// (one variable `t` along `v` versus the full `3·nat` coordinate space). A mismatch here
    /// means the `Jet1` Leibniz/Faà-di-Bruno rules or the directional seeding are wrong.
    #[test]
    fn halogen_fourth_directional_matches_full_tensor() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let ndof = 3 * system.atoms.len();
        // Generic skew direction: no zero components, no accidental symmetry.
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.11 + 0.07 * ((k * 13 % 7) as f64) - 0.15 * ((k % 3) as f64))
            .collect();
        let full = super::halogen_fourth_derivative(&system, &params).unwrap();
        let mut want = 0.0;
        for a in 0..ndof {
            for b in 0..ndof {
                for c in 0..ndof {
                    let base = ((a * ndof + b) * ndof + c) * ndof;
                    for d in 0..ndof {
                        want += v[a] * v[b] * v[c] * v[d] * full.fourth[base + d];
                    }
                }
            }
        }
        let got = super::halogen_fourth_directional(&system, &params, &v).unwrap();
        let delta = (got - want).abs();
        eprintln!(
            "halogen directional fourth: 1-D jet {got:.17e} vs full-tensor vvvv {want:.17e} \
             (delta {delta:.3e})"
        );
        assert!(
            want.abs() > 1.0e-9,
            "the full-tensor reference is numerically zero — the gate is vacuous"
        );
        assert!(
            delta <= 1.0e-12 * want.abs(),
            "halogen directional fourth deviates from the full-tensor contraction: \
             got {got:.17e} want {want:.17e} delta {delta:.3e}"
        );
    }

    #[test]
    fn halogen_fourth_derivative_is_permutation_symmetric() {
        let params = builtin_params();
        let system = ch3br_water_system();
        let analytic = super::halogen_fourth_derivative(&system, &params).unwrap();
        let max = fourth_permutation_residual(&analytic.fourth, analytic.ndof);
        assert!(
            max < 1.0e-12,
            "halogen fourth derivative is not permutation symmetric: {max:.3e}"
        );
    }

    // Memory guard: a full-space Jet4 costs ndof^4 doubles, so oversized systems must be
    // rejected up front with an actionable message rather than exhausting memory.
    #[test]
    fn halogen_fourth_derivative_rejects_oversized_systems() {
        let params = builtin_params();
        let nat = crate::dispersion::MAX_FOURTH_DERIVATIVE_NDOF / 3 + 2;
        let mut xyz = format!("{nat}\nchain\n");
        for i in 0..nat {
            xyz.push_str(&format!("H {:.6} 0.000000 0.000000\n", 1.2 * i as f64));
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
        let err = super::halogen_fourth_derivative(&system, &params).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("fourth derivative") && message.contains("degrees of freedom"),
            "unexpected guard message: {message}"
        );
    }

    #[test]
    fn periodic_bromine_halogen_image_correction_is_active() {
        let params = builtin_params();
        let system = periodic_halogen_system();
        let energy = halogen_energy(&system, &params).unwrap();
        assert!(energy.is_finite());
        assert!(
            energy.abs() > 1.0e-9,
            "periodic halogen correction should include image acceptors"
        );
    }

    #[test]
    fn periodic_gradient_matches_finite_difference() {
        let params = builtin_params();
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient(&system, &params).unwrap();
        assert!(result.stress.is_some());
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd = (halogen_energy(&plus, &params).unwrap()
                    - halogen_energy(&minus, &params).unwrap())
                    / (2.0 * h);
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
        let params = builtin_params();
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient_hessian(&system, &params).unwrap();
        assert!(result.stress.is_some());
        let grad = halogen_energy_gradient(&system, &params).unwrap();
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
            let gp = halogen_energy_gradient(&plus, &params).unwrap().gradient;
            let gm = halogen_energy_gradient(&minus, &params).unwrap().gradient;
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
        let params = builtin_params();
        let system = periodic_halogen_system();
        let result = halogen_energy_gradient(&system, &params).unwrap();
        let stress = result.stress.as_ref().unwrap();
        let volume = system.lattice.as_ref().unwrap().volume();
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let plus = strained_system(&system, row, col, h);
                let minus = strained_system(&system, row, col, -h);
                let fd = (halogen_energy(&plus, &params).unwrap()
                    - halogen_energy(&minus, &params).unwrap())
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
