// SPDX-License-Identifier: GPL-3.0-or-later
use crate::basis::{AOBasisFunction, BasisSet, CartesianPower};
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::PeriodicSystem;
use std::f64::consts::PI;

/// Drop a primitive Gaussian pair from a contracted integral when its
/// Gaussian-product contribution bound (the same per-primitive form as
/// [`contracted_pair_bound`]) is below this. The bound over-estimates the true
/// contribution by many orders of magnitude, so this only skips primitive pairs
/// that are negligible far below SCC/gradient accuracy. The most-diffuse primitive
/// of a far-but-atom-screen-surviving pair (bound ~1e-11) is always kept; the
/// tighter primitives of such pairs (exponentially smaller) are skipped, which is
/// where the speedup comes from. Near (bonded) pairs keep every primitive.
const PRIMITIVE_SCREEN_EPS: f64 = 1.0e-13;

/// Sum of absolute spherical-component coefficients of an AO (the per-AO factor of
/// the Gaussian-product screening bound).
#[inline]
fn component_abs_sum(a: &AOBasisFunction) -> f64 {
    a.components.iter().map(|c| c.coefficient.abs()).sum()
}

/// Conservative upper bound on one primitive pair's contribution to a contracted
/// moment integral of total polynomial order `poly_order` at separation `r`
/// (`r2 = r*r`). Mirrors the per-primitive term of [`contracted_pair_bound`].
#[inline]
fn primitive_pair_bound(
    pref: f64,
    coeff_scale: f64,
    alpha: f64,
    beta: f64,
    r: f64,
    r2: f64,
    poly_order: i32,
) -> f64 {
    let p = alpha + beta;
    let kab = (-alpha * beta / p * r2).exp();
    let spread = (1.0 / p).sqrt();
    let gpref = (PI / p) * (PI / p).sqrt();
    let poly = (1.0 + r + spread).powi(poly_order);
    pref.abs() * coeff_scale * kab * gpref * poly
}

#[path = "integrals_second_derivatives.rs"]
mod second_derivatives;
#[allow(unused_imports)]
pub(crate) use second_derivatives::{
    contracted_pair_with_second_derivatives, ContractedPairSecondDerivatives,
};

#[path = "integrals_third_derivatives.rs"]
mod third_derivatives;
#[allow(unused_imports)]
pub(crate) use third_derivatives::{
    contracted_pair_with_third_derivatives, ContractedPairThirdDerivatives,
};

#[derive(Clone, Copy, Debug)]
pub struct IntegralOptions {
    /// AO image-sum cutoff in Bohr. Non-positive means the home image only.
    pub cutoff: f64,
    /// Drop AO image pairs whose Gaussian-product upper bound is below this threshold.
    /// Set to zero for an unscreened, distance-cutoff-only build.
    pub screening_threshold: f64,
}

impl Default for IntegralOptions {
    fn default() -> Self {
        Self {
            cutoff: 30.0,
            screening_threshold: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntegralMatrices {
    pub overlap: Matrix,
    pub dipole_x: Matrix,
    pub dipole_y: Matrix,
    pub dipole_z: Matrix,
    pub quad_xx: Matrix,
    pub quad_xy: Matrix,
    pub quad_yy: Matrix,
    pub quad_xz: Matrix,
    pub quad_yz: Matrix,
    pub quad_zz: Matrix,
}

impl IntegralMatrices {
    /// Build non-periodic/minimum-image AO moments. For periodic gamma-point sums,
    /// prefer `build_with_cutoff` so image contributions are included explicitly.
    pub fn build(system: &PeriodicSystem, basis: &BasisSet) -> Result<Self> {
        Self::build_with_cutoff(system, basis, 0.0)
    }

    /// Build AO overlap and origin-independent moment integrals with a simple
    /// distance cutoff and no Gaussian screening.
    pub fn build_with_cutoff(
        system: &PeriodicSystem,
        basis: &BasisSet,
        cutoff: f64,
    ) -> Result<Self> {
        Self::build_with_options(
            system,
            basis,
            IntegralOptions {
                cutoff,
                screening_threshold: 0.0,
            },
        )
    }

    /// Build AO overlap and origin-independent moment integrals.
    ///
    /// The dipole/quadrupole operator origin is the ket AO centre:
    /// D_{mu,nu}=<mu|r-R_nu|nu>. In periodic cells the ket basis function is
    /// explicitly translated over lattice images up to `cutoff`.
    ///
    /// `screening_threshold` applies a conservative contracted-Gaussian product
    /// bound before expensive primitive moment evaluation. This does not change
    /// any method parameter; it is a numerical cutoff for cost control.
    pub fn build_with_options(
        system: &PeriodicSystem,
        basis: &BasisSet,
        options: IntegralOptions,
    ) -> Result<Self> {
        let cutoff = options.cutoff;
        let screening_threshold = options.screening_threshold.max(0.0);
        let n = basis.len();
        let mut out = Self {
            overlap: Matrix::zeros(n, n),
            dipole_x: Matrix::zeros(n, n),
            dipole_y: Matrix::zeros(n, n),
            dipole_z: Matrix::zeros(n, n),
            quad_xx: Matrix::zeros(n, n),
            quad_xy: Matrix::zeros(n, n),
            quad_yy: Matrix::zeros(n, n),
            quad_xz: Matrix::zeros(n, n),
            quad_yz: Matrix::zeros(n, n),
            quad_zz: Matrix::zeros(n, n),
        };

        let nat = system.atoms.len();
        let mut atom_ao_ranges = vec![(0, 0); nat];
        for (ao_idx, ao) in basis.aos.iter().enumerate() {
            let a = ao.atom_index;
            if atom_ao_ranges[a].1 == 0 {
                atom_ao_ranges[a].0 = ao_idx;
            }
            atom_ao_ranges[a].1 += 1;
        }
        let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();

        if cutoff > 0.0 {
            // 1. Self-interactions in the home cell (translation = 0)
            for a in 0..nat {
                let ri = positions[a];
                let (first_a, n_a) = atom_ao_ranges[a];
                for i in first_a..first_a + n_a {
                    for j in first_a..first_a + n_a {
                        let pair = contracted_pair(&basis.aos[i], &basis.aos[j], ri, ri);
                        out.overlap[(i, j)] = pair.0;
                        out.dipole_x[(i, j)] = pair.1;
                        out.dipole_y[(i, j)] = pair.2;
                        out.dipole_z[(i, j)] = pair.3;
                        out.quad_xx[(i, j)] = pair.4;
                        out.quad_xy[(i, j)] = pair.5;
                        out.quad_yy[(i, j)] = pair.6;
                        out.quad_xz[(i, j)] = pair.7;
                        out.quad_yz[(i, j)] = pair.8;
                        out.quad_zz[(i, j)] = pair.9;
                    }
                }
            }

            // 2. Directed short-range pairs (different atoms or periodic self-images)
            let atom_pairs = crate::pairlist::directed_short_range_pairs(system, cutoff)?;
            for pair in atom_pairs {
                let a = pair.i;
                let b = pair.j;
                let translation = pair.translation;
                let ri = positions[a];
                let rj = positions[b] + translation;

                let (first_a, n_a) = atom_ao_ranges[a];
                let (first_b, n_b) = atom_ao_ranges[b];

                for i in first_a..first_a + n_a {
                    for j in first_b..first_b + n_b {
                        if screening_threshold > 0.0
                            && contracted_pair_bound(&basis.aos[i], &basis.aos[j], ri, rj, 2)
                                < screening_threshold
                        {
                            continue;
                        }
                        let p_int = contracted_pair(&basis.aos[i], &basis.aos[j], ri, rj);
                        out.overlap[(i, j)] += p_int.0;
                        out.dipole_x[(i, j)] += p_int.1;
                        out.dipole_y[(i, j)] += p_int.2;
                        out.dipole_z[(i, j)] += p_int.3;
                        out.quad_xx[(i, j)] += p_int.4;
                        out.quad_xy[(i, j)] += p_int.5;
                        out.quad_yy[(i, j)] += p_int.6;
                        out.quad_xz[(i, j)] += p_int.7;
                        out.quad_yz[(i, j)] += p_int.8;
                        out.quad_zz[(i, j)] += p_int.9;
                    }
                }
            }
        } else {
            // Fallback for cutoff <= 0.0 (no cutoff)
            for i in 0..n {
                let ri = positions[basis.aos[i].atom_index];
                for j in 0..n {
                    let rj = positions[basis.aos[j].atom_index];
                    let pair = contracted_pair(&basis.aos[i], &basis.aos[j], ri, rj);
                    out.overlap[(i, j)] = pair.0;
                    out.dipole_x[(i, j)] = pair.1;
                    out.dipole_y[(i, j)] = pair.2;
                    out.dipole_z[(i, j)] = pair.3;
                    out.quad_xx[(i, j)] = pair.4;
                    out.quad_xy[(i, j)] = pair.5;
                    out.quad_yy[(i, j)] = pair.6;
                    out.quad_xz[(i, j)] = pair.7;
                    out.quad_yz[(i, j)] = pair.8;
                    out.quad_zz[(i, j)] = pair.9;
                }
            }
        }
        Ok(out)
    }
}

/// Conservative magnitude bound for all contracted overlap/moment integrals up
/// to `max_extra_order` on the ket centre. It is intentionally loose: false
/// positives cost time, false negatives would change the result.
pub(crate) fn contracted_pair_bound(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    max_extra_order: usize,
) -> f64 {
    let r = (cb - ca).norm();
    let component_abs_a = a
        .components
        .iter()
        .map(|c| c.coefficient.abs())
        .sum::<f64>();
    let component_abs_b = b
        .components
        .iter()
        .map(|c| c.coefficient.abs())
        .sum::<f64>();
    let component_degree = a.components.first().map(|c| c.power.total()).unwrap_or(0)
        + b.components.first().map(|c| c.power.total()).unwrap_or(0);
    let coefficient_scale = component_abs_a * component_abs_b;
    if coefficient_scale == 0.0 {
        return 0.0;
    }
    let mut bound = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            let p = pa.exponent + pb.exponent;
            if p <= 0.0 {
                continue;
            }
            let mu = pa.exponent * pb.exponent / p;
            let gaussian = (-mu * r * r).exp() * (PI / p).powf(1.5);
            let spread = (1.0 / p).sqrt();
            let polynomial = (1.0 + r + spread).powi((component_degree + max_extra_order) as i32);
            bound +=
                (pa.coefficient * pb.coefficient).abs() * coefficient_scale * gaussian * polynomial;
        }
    }
    bound
}

pub(crate) fn contracted_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let mut s = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut dz = 0.0;
    let mut qxx = 0.0;
    let mut qxy = 0.0;
    let mut qyy = 0.0;
    let mut qxz = 0.0;
    let mut qyz = 0.0;
    let mut qzz = 0.0;

    // Geometry and screening scale are constant across primitive pairs.
    let r2 = (ca - cb).norm2();
    let r = r2.sqrt();
    let coeff_scale = component_abs_sum(a) * component_abs_sum(b);
    // Overlap..quadrupole moments reach polynomial order 2 on top of the AO degree.
    let poly_order = (a.components.first().map(|c| c.power.total()).unwrap_or(0)
        + b.components.first().map(|c| c.power.total()).unwrap_or(0)
        + 2) as i32;

    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            // Primitive screening: skip negligible primitive pairs (the tighter
            // primitives of far-but-surviving atom pairs).
            if primitive_pair_bound(pref, coeff_scale, alpha, beta, r, r2, poly_order)
                < PRIMITIVE_SCREEN_EPS
            {
                continue;
            }
            let kab_3d = (-alpha * beta / p * r2).exp();

            let mut g_moments = [0.0; 7];
            g_moments[0] = (PI / p).sqrt();
            for m in 1..=6 {
                g_moments[m] = g_moments[m - 1] * (m as f64 - 0.5) / p;
            }

            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient * kab_3d;
                    let ap = ca_term.power;
                    let bp = cb_term.power;
                    let m = primitive_moments_up_to_quadrupole_no_kab(
                        ap, bp, alpha, beta, ca, cb, p, &g_moments,
                    );
                    s += cfac * m[0];
                    dx += cfac * m[1];
                    dy += cfac * m[2];
                    dz += cfac * m[3];
                    qxx += cfac * m[4];
                    qxy += cfac * m[5];
                    qyy += cfac * m[6];
                    qxz += cfac * m[7];
                    qyz += cfac * m[8];
                    qzz += cfac * m[9];
                }
            }
        }
    }
    (s, dx, dy, dz, qxx, qxy, qyy, qxz, qyz, qzz)
}

/// 1D kinetic-energy contribution `<(x-a)^la | -1/2 d^2/dx^2 | (x-b)^lb>` (no
/// `K_ab` factor), built from 1D overlap moments at shifted ket angular momenta:
/// `-1/2 [ lb(lb-1) S(la,lb-2) - 2 beta (2 lb + 1) S(la,lb) + 4 beta^2 S(la,lb+2) ]`
/// (the second derivative of the ket Gaussian).
fn kinetic_1d_no_kab(
    la: usize,
    lb: usize,
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    p: f64,
    g_moments: &[f64],
) -> f64 {
    let s0 = moment_1d_no_kab(la, lb, alpha, beta, a, b, p, g_moments);
    let sp2 = moment_1d_no_kab(la, lb + 2, alpha, beta, a, b, p, g_moments);
    let sm2 = if lb >= 2 {
        moment_1d_no_kab(la, lb - 2, alpha, beta, a, b, p, g_moments)
    } else {
        0.0
    };
    -0.5 * ((lb * (lb.wrapping_sub(1))) as f64 * sm2 - 2.0 * beta * (2 * lb + 1) as f64 * s0
        + 4.0 * beta * beta * sp2)
}

/// Primitive 3D kinetic-energy integral `<a|-1/2 nabla^2|b>` (no `K_ab` factor):
/// `Tx Sy Sz + Sx Ty Sz + Sx Sy Tz`, with `Sx`/`Tx` the 1D overlap/kinetic moments.
fn primitive_kinetic_no_kab(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
    p: f64,
    g_moments: &[f64],
) -> f64 {
    let sx = moment_1d_no_kab(pa.x, pb.x, alpha, beta, ca.x, cb.x, p, g_moments);
    let sy = moment_1d_no_kab(pa.y, pb.y, alpha, beta, ca.y, cb.y, p, g_moments);
    let sz = moment_1d_no_kab(pa.z, pb.z, alpha, beta, ca.z, cb.z, p, g_moments);
    let tx = kinetic_1d_no_kab(pa.x, pb.x, alpha, beta, ca.x, cb.x, p, g_moments);
    let ty = kinetic_1d_no_kab(pa.y, pb.y, alpha, beta, ca.y, cb.y, p, g_moments);
    let tz = kinetic_1d_no_kab(pa.z, pb.z, alpha, beta, ca.z, cb.z, p, g_moments);
    tx * sy * sz + sx * ty * sz + sx * sy * tz
}

/// Kinetic-energy integral `<phi_a|-1/2 nabla^2|phi_b>` between two contracted AOs
/// centred at `ca`/`cb`. Zero-field (real); `<phi_a|p^2|phi_b> = 2 * this`. The
/// building block for the GFN1-xTB-M kinetic-energy correction (the `<phi|p^2|phi>`
/// term); the field-dependent `<omega|pi^2|omega>` adds the paramagnetic /
/// diamagnetic London terms on top of the same overlap machinery.
pub(crate) fn contracted_kinetic(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> f64 {
    let r2 = (ca - cb).norm2();
    let r = r2.sqrt();
    let coeff_scale = component_abs_sum(a) * component_abs_sum(b);
    // Overlap reaches the ket degree + 2 (the second-derivative term S(la,lb+2)).
    let poly_order = (a.components.first().map(|c| c.power.total()).unwrap_or(0)
        + b.components.first().map(|c| c.power.total()).unwrap_or(0)
        + 2) as i32;
    let mut t = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            if primitive_pair_bound(pref, coeff_scale, alpha, beta, r, r2, poly_order)
                < PRIMITIVE_SCREEN_EPS
            {
                continue;
            }
            let kab_3d = (-alpha * beta / p * r2).exp();
            let mut g_moments = [0.0; 7];
            g_moments[0] = (PI / p).sqrt();
            for m in 1..=6 {
                g_moments[m] = g_moments[m - 1] * (m as f64 - 0.5) / p;
            }
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient * kab_3d;
                    t += cfac
                        * primitive_kinetic_no_kab(
                            ca_term.power,
                            cb_term.power,
                            alpha,
                            beta,
                            ca,
                            cb,
                            p,
                            &g_moments,
                        );
                }
            }
        }
    }
    t
}

/// AO kinetic-energy matrix `T_{mu,nu} = <mu|-1/2 nabla^2|nu>` for a non-periodic
/// system. Used by the GFN1-xTB-M kinetic-energy correction.
pub fn kinetic_energy_matrix(system: &PeriodicSystem, basis: &BasisSet) -> Matrix {
    let n = basis.len();
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let mut t = Matrix::zeros(n, n);
    for i in 0..n {
        let ri = positions[basis.aos[i].atom_index];
        for j in 0..n {
            let rj = positions[basis.aos[j].atom_index];
            t[(i, j)] = contracted_kinetic(&basis.aos[i], &basis.aos[j], ri, rj);
        }
    }
    t
}

pub(crate) fn contracted_pair_with_derivatives(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> ([f64; 10], [Vec3; 10], [Vec3; 10]) {
    let mut moments = [0.0_f64; 10];
    let mut da = [Vec3::zero(); 10];
    let mut db = [Vec3::zero(); 10];

    // Geometry and screening scale are constant across primitive pairs.
    let r2 = (ca - cb).norm2();
    let r = r2.sqrt();
    let coeff_scale = component_abs_sum(a) * component_abs_sum(b);
    // Quadrupole moments (order 2) plus a first derivative (+1) over the AO degree.
    let poly_order = (a.components.first().map(|c| c.power.total()).unwrap_or(0)
        + b.components.first().map(|c| c.power.total()).unwrap_or(0)
        + 3) as i32;

    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            // Primitive screening: skip negligible primitive pairs (the tighter
            // primitives of far-but-surviving atom pairs).
            if primitive_pair_bound(pref, coeff_scale, alpha, beta, r, r2, poly_order)
                < PRIMITIVE_SCREEN_EPS
            {
                continue;
            }
            let kab_3d = (-alpha * beta / p * r2).exp();

            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient;
                    let (m, d_bra, d_ket) = primitive_moments_derivatives_up_to_quadrupole(
                        ca_term.power,
                        cb_term.power,
                        pa.exponent,
                        pb.exponent,
                        ca,
                        cb,
                    );
                    let cfac_kab = cfac * kab_3d;
                    for k in 0..10 {
                        moments[k] += cfac_kab * m[k];
                        da[k] += d_bra[k] * cfac_kab;
                        db[k] += d_ket[k] * cfac_kab;
                    }
                }
            }
        }
    }
    (moments, da, db)
}

fn primitive_moments_up_to_quadrupole_no_kab(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
    p: f64,
    g_moments: &[f64],
) -> [f64; 10] {
    let mx = [
        moment_1d_no_kab(pa.x, pb.x, alpha, beta, ca.x, cb.x, p, g_moments),
        moment_1d_no_kab(pa.x, pb.x + 1, alpha, beta, ca.x, cb.x, p, g_moments),
        moment_1d_no_kab(pa.x, pb.x + 2, alpha, beta, ca.x, cb.x, p, g_moments),
    ];
    let my = [
        moment_1d_no_kab(pa.y, pb.y, alpha, beta, ca.y, cb.y, p, g_moments),
        moment_1d_no_kab(pa.y, pb.y + 1, alpha, beta, ca.y, cb.y, p, g_moments),
        moment_1d_no_kab(pa.y, pb.y + 2, alpha, beta, ca.y, cb.y, p, g_moments),
    ];
    let mz = [
        moment_1d_no_kab(pa.z, pb.z, alpha, beta, ca.z, cb.z, p, g_moments),
        moment_1d_no_kab(pa.z, pb.z + 1, alpha, beta, ca.z, cb.z, p, g_moments),
        moment_1d_no_kab(pa.z, pb.z + 2, alpha, beta, ca.z, cb.z, p, g_moments),
    ];
    [
        mx[0] * my[0] * mz[0],
        mx[1] * my[0] * mz[0],
        mx[0] * my[1] * mz[0],
        mx[0] * my[0] * mz[1],
        mx[2] * my[0] * mz[0],
        mx[1] * my[1] * mz[0],
        mx[0] * my[2] * mz[0],
        mx[1] * my[0] * mz[1],
        mx[0] * my[1] * mz[1],
        mx[0] * my[0] * mz[2],
    ]
}

/// Ten unique Cartesian **octupole** moments of a primitive pair (no `K_ab` factor),
/// ket-centred: `<(r-A)^pa | (r-B)_i (r-B)_j (r-B)_k | (r-B)^pb>` for the symmetric
/// rank-3 index set `[xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz]`. Built from the
/// 1D moments raised to 3rd order on the ket (`mx[k] = <a|(x-B)^(pb.x+k)|b>`), exactly
/// as [`primitive_moments_up_to_quadrupole_no_kab`] does for orders 0..2.
fn primitive_octupole_no_kab(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
    p: f64,
    g_moments: &[f64],
) -> [f64; 10] {
    let mx = [
        moment_1d_no_kab(pa.x, pb.x, alpha, beta, ca.x, cb.x, p, g_moments),
        moment_1d_no_kab(pa.x, pb.x + 1, alpha, beta, ca.x, cb.x, p, g_moments),
        moment_1d_no_kab(pa.x, pb.x + 2, alpha, beta, ca.x, cb.x, p, g_moments),
        moment_1d_no_kab(pa.x, pb.x + 3, alpha, beta, ca.x, cb.x, p, g_moments),
    ];
    let my = [
        moment_1d_no_kab(pa.y, pb.y, alpha, beta, ca.y, cb.y, p, g_moments),
        moment_1d_no_kab(pa.y, pb.y + 1, alpha, beta, ca.y, cb.y, p, g_moments),
        moment_1d_no_kab(pa.y, pb.y + 2, alpha, beta, ca.y, cb.y, p, g_moments),
        moment_1d_no_kab(pa.y, pb.y + 3, alpha, beta, ca.y, cb.y, p, g_moments),
    ];
    let mz = [
        moment_1d_no_kab(pa.z, pb.z, alpha, beta, ca.z, cb.z, p, g_moments),
        moment_1d_no_kab(pa.z, pb.z + 1, alpha, beta, ca.z, cb.z, p, g_moments),
        moment_1d_no_kab(pa.z, pb.z + 2, alpha, beta, ca.z, cb.z, p, g_moments),
        moment_1d_no_kab(pa.z, pb.z + 3, alpha, beta, ca.z, cb.z, p, g_moments),
    ];
    [
        mx[3] * my[0] * mz[0], // xxx
        mx[2] * my[1] * mz[0], // xxy
        mx[2] * my[0] * mz[1], // xxz
        mx[1] * my[2] * mz[0], // xyy
        mx[1] * my[1] * mz[1], // xyz
        mx[1] * my[0] * mz[2], // xzz
        mx[0] * my[3] * mz[0], // yyy
        mx[0] * my[2] * mz[1], // yyz
        mx[0] * my[1] * mz[2], // yzz
        mx[0] * my[0] * mz[3], // zzz
    ]
}

/// Contracted Cartesian octupole integrals `<a | (r-R_b)_i (r-R_b)_j (r-R_b)_k | b>`
/// (operator origin = ket centre `R_b`), returned in the symmetric rank-3 order
/// `[xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz]`. Separate from
/// [`contracted_pair`] so the overlap/dipole/quadrupole path is untouched; evaluated
/// only when the experimental octupole multipole correction is enabled.
// Consumed by the octupole on-site moments (mDFTB2 octupole, Stage 2c).
#[allow(dead_code)]
pub(crate) fn contracted_octupole_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
) -> [f64; 10] {
    let mut o = [0.0_f64; 10];
    let r2 = (ca - cb).norm2();
    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            let kab_3d = (-alpha * beta / p * r2).exp();
            let mut g_moments = [0.0; 7];
            g_moments[0] = (PI / p).sqrt();
            for m in 1..=6 {
                g_moments[m] = g_moments[m - 1] * (m as f64 - 0.5) / p;
            }
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient * kab_3d;
                    let m = primitive_octupole_no_kab(
                        ca_term.power,
                        cb_term.power,
                        alpha,
                        beta,
                        ca,
                        cb,
                        p,
                        &g_moments,
                    );
                    for k in 0..10 {
                        o[k] += cfac * m[k];
                    }
                }
            }
        }
    }
    o
}

/// Canonical ordering of the `(L+1)(L+2)/2` unique symmetric Cartesian components of a rank-`L`
/// tensor: `lx` descending, then `ly` descending (`lz = L−lx−ly`). For `L=1` this is `[x,y,z]`,
/// for `L=3` `[xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz]` — i.e. it **matches the hard-coded
/// octupole order**, so `contracted_moment_rank(..,3)` reproduces [`contracted_octupole_pair`].
#[allow(dead_code)] // used by the arbitrary-rank multipole path (multipole_order ≥ 4)
pub(crate) fn cartesian_rank_components(l: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::with_capacity((l + 1) * (l + 2) / 2);
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            v.push((lx, ly, l - lx - ly));
        }
    }
    v
}

/// Unique symmetric Cartesian rank-`l` moments of a primitive pair (no `K_ab`), ket-centred,
/// `<(r−A)^pa | Π (r−B)_c | (r−B)^pb>`, built from the 1D moments raised to order `l` on the ket
/// (`mx[k] = <a|(x−B)^(pb.x+k)|b>`), generalizing [`primitive_octupole_no_kab`] to arbitrary `l`.
#[allow(clippy::too_many_arguments)]
fn primitive_cartesian_moment_rank(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
    p: f64,
    g_moments: &[f64],
    l: usize,
) -> Vec<f64> {
    let m1d = |pa1: usize, pb1: usize, ca1: f64, cb1: f64| {
        moment_1d_no_kab(pa1, pb1, alpha, beta, ca1, cb1, p, g_moments)
    };
    let mx: Vec<f64> = (0..=l).map(|k| m1d(pa.x, pb.x + k, ca.x, cb.x)).collect();
    let my: Vec<f64> = (0..=l).map(|k| m1d(pa.y, pb.y + k, ca.y, cb.y)).collect();
    let mz: Vec<f64> = (0..=l).map(|k| m1d(pa.z, pb.z + k, ca.z, cb.z)).collect();
    cartesian_rank_components(l)
        .into_iter()
        .map(|(lx, ly, lz)| mx[lx] * my[ly] * mz[lz])
        .collect()
}

/// Contracted Cartesian rank-`l` moment integrals `<a | Π_c (r−R_b)_c | b>` (operator origin =
/// ket centre `R_b`), in the [`cartesian_rank_components`] order. The general-rank analogue of
/// [`contracted_octupole_pair`] (which it reproduces at `l=3`); used by the arbitrary-rank
/// multipole moments (`multipole_order ≥ 4`). The legacy rank ≤3 paths are kept byte-identical.
#[allow(dead_code)] // wired in by the arbitrary-rank multipole moments (multipole_order ≥ 4)
pub(crate) fn contracted_moment_rank(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    l: usize,
) -> Vec<f64> {
    let ncomp = (l + 1) * (l + 2) / 2;
    let mut o = vec![0.0_f64; ncomp];
    let r2 = (ca - cb).norm2();
    // Highest 1D Gaussian-moment order needed: (max bra power) + (max ket power) + l.
    let la = a
        .components
        .iter()
        .map(|c| c.power.total())
        .max()
        .unwrap_or(0);
    let lb = b
        .components
        .iter()
        .map(|c| c.power.total())
        .max()
        .unwrap_or(0);
    let gmax = la + lb + l;
    for pa in &a.primitives {
        for pb in &b.primitives {
            let pref = pa.coefficient * pb.coefficient;
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let p = alpha + beta;
            let kab_3d = (-alpha * beta / p * r2).exp();
            let mut g_moments = vec![0.0_f64; gmax + 1];
            g_moments[0] = (PI / p).sqrt();
            for m in 1..=gmax {
                g_moments[m] = g_moments[m - 1] * (m as f64 - 0.5) / p;
            }
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let cfac = pref * ca_term.coefficient * cb_term.coefficient * kab_3d;
                    let m = primitive_cartesian_moment_rank(
                        ca_term.power,
                        cb_term.power,
                        alpha,
                        beta,
                        ca,
                        cb,
                        p,
                        &g_moments,
                        l,
                    );
                    for (ok, mk) in o.iter_mut().zip(m.iter()) {
                        *ok += cfac * mk;
                    }
                }
            }
        }
    }
    o
}

fn primitive_moments_derivatives_up_to_quadrupole(
    pa: CartesianPower,
    pb: CartesianPower,
    alpha: f64,
    beta: f64,
    ca: Vec3,
    cb: Vec3,
) -> ([f64; 10], [Vec3; 10], [Vec3; 10]) {
    let max_ix = pa.x + 1;
    let max_iy = pa.y + 1;
    let max_iz = pa.z + 1;
    let max_jx = pb.x + 3;
    let max_jy = pb.y + 3;
    let max_jz = pb.z + 3;
    let mx = moment_1d_table(max_ix, max_jx, alpha, beta, ca.x, cb.x);
    let my = moment_1d_table(max_iy, max_jy, alpha, beta, ca.y, cb.y);
    let mz = moment_1d_table(max_iz, max_jz, alpha, beta, ca.z, cb.z);
    let extras = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 2],
    ];

    let total = |ix: usize, iy: usize, iz: usize, jx: usize, jy: usize, jz: usize| -> f64 {
        moment_1d_get(&mx, max_jx, ix, jx)
            * moment_1d_get(&my, max_jy, iy, jy)
            * moment_1d_get(&mz, max_jz, iz, jz)
    };

    let mut moments = [0.0_f64; 10];
    let mut da = [Vec3::zero(); 10];
    let mut db = [Vec3::zero(); 10];
    for (k, extra) in extras.iter().enumerate() {
        let jx = pb.x + extra[0];
        let jy = pb.y + extra[1];
        let jz = pb.z + extra[2];
        moments[k] = total(pa.x, pa.y, pa.z, jx, jy, jz);

        let dax = 2.0 * alpha * total(pa.x + 1, pa.y, pa.z, jx, jy, jz)
            - if pa.x > 0 {
                (pa.x as f64) * total(pa.x - 1, pa.y, pa.z, jx, jy, jz)
            } else {
                0.0
            };
        let day = 2.0 * alpha * total(pa.x, pa.y + 1, pa.z, jx, jy, jz)
            - if pa.y > 0 {
                (pa.y as f64) * total(pa.x, pa.y - 1, pa.z, jx, jy, jz)
            } else {
                0.0
            };
        let daz = 2.0 * alpha * total(pa.x, pa.y, pa.z + 1, jx, jy, jz)
            - if pa.z > 0 {
                (pa.z as f64) * total(pa.x, pa.y, pa.z - 1, jx, jy, jz)
            } else {
                0.0
            };
        da[k] = Vec3::new(dax, day, daz);

        let dbx = 2.0 * beta * total(pa.x, pa.y, pa.z, jx + 1, jy, jz)
            - if jx > 0 {
                (jx as f64) * total(pa.x, pa.y, pa.z, jx - 1, jy, jz)
            } else {
                0.0
            };
        let dby = 2.0 * beta * total(pa.x, pa.y, pa.z, jx, jy + 1, jz)
            - if jy > 0 {
                (jy as f64) * total(pa.x, pa.y, pa.z, jx, jy - 1, jz)
            } else {
                0.0
            };
        let dbz = 2.0 * beta * total(pa.x, pa.y, pa.z, jx, jy, jz + 1)
            - if jz > 0 {
                (jz as f64) * total(pa.x, pa.y, pa.z, jx, jy, jz - 1)
            } else {
                0.0
            };
        db[k] = Vec3::new(dbx, dby, dbz);
    }
    (moments, da, db)
}

fn moment_1d_table(max_i: usize, max_j: usize, alpha: f64, beta: f64, a: f64, b: f64) -> Vec<f64> {
    let p = alpha + beta;
    let max_n = max_i + max_j;
    let max_m = max_n / 2;
    let mut g_moments = Vec::with_capacity(max_m + 1);
    g_moments.push((PI / p).sqrt());
    for m in 1..=max_m {
        let val = g_moments[m - 1] * (m as f64 - 0.5) / p;
        g_moments.push(val);
    }

    let mut out = vec![0.0_f64; (max_i + 1) * (max_j + 1)];
    for i in 0..=max_i {
        for j in 0..=max_j {
            out[i * (max_j + 1) + j] = moment_1d_no_kab(i, j, alpha, beta, a, b, p, &g_moments);
        }
    }
    out
}

#[inline]
fn moment_1d_get(table: &[f64], max_j: usize, i: usize, j: usize) -> f64 {
    table[i * (max_j + 1) + j]
}

fn moment_1d_no_kab(
    ia: usize,
    jb_total: usize,
    alpha: f64,
    beta: f64,
    a: f64,
    b: f64,
    p: f64,
    g_moments: &[f64],
) -> f64 {
    let product_center = (alpha * a + beta * b) / p;
    let pa = product_center - a;
    let pb = product_center - b;
    let mut sum = 0.0;
    for u in 0..=ia {
        let ca = binom(ia, u) * pa.powi((ia - u) as i32);
        for v in 0..=jb_total {
            let cb = binom(jb_total, v) * pb.powi((jb_total - v) as i32);
            let uv = u + v;
            if uv % 2 == 0 {
                sum += ca * cb * g_moments[uv / 2];
            }
        }
    }
    sum
}

const BINOM_TABLE: [[f64; 16]; 16] = [
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 4.0, 6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 7.0, 21.0, 35.0, 35.0, 21.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 9.0, 36.0, 84.0, 126.0, 126.0, 84.0, 36.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        1.0, 10.0, 45.0, 120.0, 210.0, 252.0, 210.0, 120.0, 45.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0,
    ],
    [
        1.0, 11.0, 55.0, 165.0, 330.0, 462.0, 462.0, 330.0, 165.0, 55.0, 11.0, 1.0, 0.0, 0.0, 0.0,
        0.0,
    ],
    [
        1.0, 12.0, 66.0, 220.0, 495.0, 792.0, 924.0, 792.0, 495.0, 220.0, 66.0, 12.0, 1.0, 0.0,
        0.0, 0.0,
    ],
    [
        1.0, 13.0, 78.0, 286.0, 715.0, 1287.0, 1716.0, 1716.0, 1287.0, 715.0, 286.0, 78.0, 13.0,
        1.0, 0.0, 0.0,
    ],
    [
        1.0, 14.0, 91.0, 364.0, 1001.0, 2002.0, 3003.0, 3432.0, 3003.0, 2002.0, 1001.0, 364.0,
        91.0, 14.0, 1.0, 0.0,
    ],
    [
        1.0, 15.0, 105.0, 455.0, 1365.0, 3003.0, 5005.0, 6435.0, 6435.0, 5005.0, 3003.0, 1365.0,
        455.0, 105.0, 15.0, 1.0,
    ],
];

#[inline]
fn binom(n: usize, k: usize) -> f64 {
    if n < 16 && k < 16 {
        BINOM_TABLE[n][k]
    } else {
        let k = k.min(n - k);
        let mut out = 1.0;
        for i in 0..k {
            out *= (n - i) as f64 / (i + 1) as f64;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::CartesianComponent;
    use crate::params::AngularMomentum;

    fn shifted(mut v: Vec3, axis: usize, delta: f64) -> Vec3 {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            2 => v.z += delta,
            _ => panic!("axis out of range"),
        }
        v
    }

    fn component(v: Vec3, axis: usize) -> f64 {
        match axis {
            0 => v.x,
            1 => v.y,
            2 => v.z,
            _ => panic!("axis out of range"),
        }
    }

    #[test]
    fn kinetic_integral_matches_analytic_and_is_hermitian() {
        let alpha = 0.8;
        let beta = 1.1;
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let p = alpha + beta;
        let mut g = [0.0; 7];
        g[0] = (PI / p).sqrt();
        for m in 1..=6 {
            g[m] = g[m - 1] * (m as f64 - 0.5) / p;
        }
        let r2 = (ca - cb).norm2();
        let kab = (-alpha * beta / p * r2).exp();
        // s-s kinetic vs the closed form T = (ab/p)(3 - 2 ab R^2/p) S_ss.
        let s0 = CartesianPower::new(0, 0, 0);
        let t_ss = primitive_kinetic_no_kab(s0, s0, alpha, beta, ca, cb, p, &g) * kab;
        let s_ss = (PI / p).powf(1.5) * kab;
        let analytic = (alpha * beta / p) * (3.0 - 2.0 * alpha * beta * r2 / p) * s_ss;
        assert!(
            (t_ss - analytic).abs() < 1.0e-12,
            "kinetic s-s {t_ss} vs analytic {analytic}"
        );
        // Hermiticity for a p-s pair: <a|T|b> == <b|T|a> (swap centres/exponents/powers).
        let px = CartesianPower::new(1, 0, 0);
        let tab = primitive_kinetic_no_kab(px, s0, alpha, beta, ca, cb, p, &g) * kab;
        let tba = primitive_kinetic_no_kab(s0, px, beta, alpha, cb, ca, p, &g) * kab;
        assert!(
            (tab - tba).abs() < 1.0e-12,
            "kinetic Hermiticity p-s {tab} vs {tba}"
        );
    }

    /// Stage 2a gate: the new Cartesian octupole primitive integrals must match a direct
    /// numerical quadrature of `<(r-A)^pa | (r-B)_i (r-B)_j (r-B)_k | (r-B)^pb>` (the
    /// integral is separable, so each axis is a 1D trapezoidal sum of the actual
    /// polynomial-times-Gaussian integrand — independent of the moment recurrence).
    #[test]
    fn primitive_octupole_matches_numerical_quadrature() {
        let pa = CartesianPower::new(1, 0, 0); // p_x bra
        let pb = CartesianPower::new(0, 1, 0); // p_y ket
        let alpha = 0.9;
        let beta = 1.3;
        let ca = Vec3::new(-0.2, 0.15, 0.3);
        let cb = Vec3::new(0.35, -0.25, 0.1);
        let p = alpha + beta;
        let mut g = [0.0; 7];
        g[0] = (PI / p).sqrt();
        for m in 1..=6 {
            g[m] = g[m - 1] * (m as f64 - 0.5) / p;
        }
        let octu = primitive_octupole_no_kab(pa, pb, alpha, beta, ca, cb, p, &g);
        // 1D moment (no K_ab) ∫ (x-A)^la (x-B)^lb e^{-p (x-Px)^2} dx by fine trapezoid.
        let num_1d = |la: usize, lb: usize, aa: f64, bb: f64| -> f64 {
            let px = (alpha * aa + beta * bb) / p;
            let width = 13.0 / p.sqrt();
            let n = 120_000usize;
            let (lo, hi) = (px - width, px + width);
            let dx = (hi - lo) / n as f64;
            let mut s = 0.0;
            for i in 0..=n {
                let x = lo + i as f64 * dx;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                let f = (x - aa).powi(la as i32)
                    * (x - bb).powi(lb as i32)
                    * (-p * (x - px) * (x - px)).exp();
                s += w * f;
            }
            s * dx
        };
        // (ox, oy, oz) raised-ket powers for each octupole component.
        let comps = [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ];
        for (k, &(ox, oy, oz)) in comps.iter().enumerate() {
            let num = num_1d(pa.x, pb.x + ox, ca.x, cb.x)
                * num_1d(pa.y, pb.y + oy, ca.y, cb.y)
                * num_1d(pa.z, pb.z + oz, ca.z, cb.z);
            assert!(
                (octu[k] - num).abs() < 1.0e-6 * (1.0 + num.abs()),
                "octupole comp {k}: analytic {} vs numerical {}",
                octu[k],
                num
            );
        }
    }

    /// v0.2.0 arbitrary-rank gate: the general `contracted_moment_rank(..,3)` must reproduce the
    /// hard-coded `contracted_octupole_pair` byte-for-byte (same 1D moments, same canonical order),
    /// and rank-4 (hexadecapole, 15 unique components) must be finite — proving the rank-L integral
    /// generalizes the octupole.
    #[test]
    fn contracted_moment_rank_matches_octupole() {
        // angular metadata is unused by the moment integrals; only components/primitives matter.
        let a = test_ao(
            AngularMomentum::S,
            vec![
                CartesianComponent::new(CartesianPower::new(2, 0, 0), 1.0),
                CartesianComponent::new(CartesianPower::new(1, 1, 0), 0.7),
            ],
            vec![(0.8, 0.5), (2.1, 0.3)],
        );
        let b = test_ao(
            AngularMomentum::S,
            vec![
                CartesianComponent::new(CartesianPower::new(0, 1, 0), 1.0),
                CartesianComponent::new(CartesianPower::new(0, 0, 1), -0.4),
            ],
            vec![(1.3, 0.6), (0.5, 0.2)],
        );
        let ca = Vec3::new(0.1, -0.2, 0.05);
        let cb = Vec3::new(-0.15, 0.25, 0.3);
        let octu = contracted_octupole_pair(&a, &b, ca, cb);
        let rank3 = contracted_moment_rank(&a, &b, ca, cb, 3);
        assert_eq!(rank3.len(), 10);
        for k in 0..10 {
            assert!(
                (octu[k] - rank3[k]).abs() < 1.0e-13,
                "rank-3 comp {k}: octupole {} vs general {}",
                octu[k],
                rank3[k]
            );
        }
        // rank-1 ordering is [x,y,z]; rank-4 (hexadecapole) is finite with 15 unique components.
        assert_eq!(
            cartesian_rank_components(1),
            vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        );
        let rank4 = contracted_moment_rank(&a, &b, ca, cb, 4);
        assert_eq!(rank4.len(), 15);
        assert!(rank4.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn primitive_first_derivatives_match_finite_difference() {
        let powers = [
            (CartesianPower::new(0, 0, 0), CartesianPower::new(0, 0, 0)),
            (CartesianPower::new(1, 0, 0), CartesianPower::new(0, 0, 0)),
            (CartesianPower::new(0, 1, 0), CartesianPower::new(0, 0, 1)),
            (CartesianPower::new(1, 0, 0), CartesianPower::new(0, 1, 0)),
        ];
        let alpha = 0.8;
        let beta = 1.1;
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let h = 1.0e-5;
        let p = alpha + beta;
        let mut g_moments = [0.0; 7];
        g_moments[0] = (PI / p).sqrt();
        for m in 1..=6 {
            g_moments[m] = g_moments[m - 1] * (m as f64 - 0.5) / p;
        }
        for (pa, pb) in powers {
            let (_, d_bra, d_ket) =
                primitive_moments_derivatives_up_to_quadrupole(pa, pb, alpha, beta, ca, cb);
            for moment in 0..10 {
                for axis in 0..3 {
                    let ca_plus = shifted(ca, axis, h);
                    let kab_plus = (-alpha * beta / p * (ca_plus - cb).norm2()).exp();
                    let bra_plus = primitive_moments_up_to_quadrupole_no_kab(
                        pa, pb, alpha, beta, ca_plus, cb, p, &g_moments,
                    );
                    let bra_plus_scaled = bra_plus[moment] * kab_plus;

                    let ca_minus = shifted(ca, axis, -h);
                    let kab_minus = (-alpha * beta / p * (ca_minus - cb).norm2()).exp();
                    let bra_minus = primitive_moments_up_to_quadrupole_no_kab(
                        pa, pb, alpha, beta, ca_minus, cb, p, &g_moments,
                    );
                    let bra_minus_scaled = bra_minus[moment] * kab_minus;

                    let fd_bra = (bra_plus_scaled - bra_minus_scaled) / (2.0 * h);
                    let kab_3d = (-alpha * beta / p * (ca - cb).norm2()).exp();
                    let analytic_bra = component(d_bra[moment], axis) * kab_3d;
                    assert!(
                        (analytic_bra - fd_bra).abs() < 1.0e-8,
                        "bra pa={pa:?} pb={pb:?} moment={moment} axis={axis}: analytic={} fd={}",
                        analytic_bra,
                        fd_bra,
                    );

                    let cb_plus = shifted(cb, axis, h);
                    let kab_plus = (-alpha * beta / p * (ca - cb_plus).norm2()).exp();
                    let ket_plus = primitive_moments_up_to_quadrupole_no_kab(
                        pa, pb, alpha, beta, ca, cb_plus, p, &g_moments,
                    );
                    let ket_plus_scaled = ket_plus[moment] * kab_plus;

                    let cb_minus = shifted(cb, axis, -h);
                    let kab_minus = (-alpha * beta / p * (ca - cb_minus).norm2()).exp();
                    let ket_minus = primitive_moments_up_to_quadrupole_no_kab(
                        pa, pb, alpha, beta, ca, cb_minus, p, &g_moments,
                    );
                    let ket_minus_scaled = ket_minus[moment] * kab_minus;

                    let fd_ket = (ket_plus_scaled - ket_minus_scaled) / (2.0 * h);
                    let analytic_ket = component(d_ket[moment], axis) * kab_3d;
                    assert!(
                        (analytic_ket - fd_ket).abs() < 1.0e-8,
                        "ket pa={pa:?} pb={pb:?} moment={moment} axis={axis}: analytic={} fd={}",
                        analytic_ket,
                        fd_ket,
                    );
                }
            }
        }
    }

    #[test]
    fn contracted_overlap_derivatives_match_finite_difference() {
        let a = test_ao(
            AngularMomentum::P,
            vec![CartesianComponent::new(CartesianPower::new(1, 0, 0), 1.0)],
            vec![(0.8, 0.7), (1.6, -0.2)],
        );
        let b = test_ao(
            AngularMomentum::P,
            vec![CartesianComponent::new(CartesianPower::new(0, 1, 0), 1.0)],
            vec![(1.1, 0.5), (0.6, 0.3)],
        );
        let ca = Vec3::new(-0.2, 0.1, 0.3);
        let cb = Vec3::new(0.4, -0.3, 0.2);
        let h = 1.0e-5;
        let (_, d_bra, d_ket) = contracted_pair_with_derivatives(&a, &b, ca, cb);
        for axis in 0..3 {
            let bra_plus = contracted_pair(&a, &b, shifted(ca, axis, h), cb).0;
            let bra_minus = contracted_pair(&a, &b, shifted(ca, axis, -h), cb).0;
            let fd_bra = (bra_plus - bra_minus) / (2.0 * h);
            assert!(
                (component(d_bra[0], axis) - fd_bra).abs() < 1.0e-8,
                "bra axis={axis}: analytic={} fd={fd_bra}",
                component(d_bra[0], axis),
            );

            let ket_plus = contracted_pair(&a, &b, ca, shifted(cb, axis, h)).0;
            let ket_minus = contracted_pair(&a, &b, ca, shifted(cb, axis, -h)).0;
            let fd_ket = (ket_plus - ket_minus) / (2.0 * h);
            assert!(
                (component(d_ket[0], axis) - fd_ket).abs() < 1.0e-8,
                "ket axis={axis}: analytic={} fd={fd_ket}",
                component(d_ket[0], axis),
            );
        }
    }

    fn test_ao(
        angular: AngularMomentum,
        components: Vec<CartesianComponent>,
        primitives: Vec<(f64, f64)>,
    ) -> AOBasisFunction {
        AOBasisFunction {
            atom_index: 0,
            z: 1,
            shell_index: 0,
            shell_param_index: 0,
            shell_label: "test".to_string(),
            angular,
            cart_label: "test",
            components,
            hdiag_ev: 0.0,
            hdiag_ha: 0.0,
            slater: 1.0,
            principal_n: 2,
            nprim: primitives.len(),
            reference_occ: 0.0,
            is_valence: true,
            poly_raw: None,
            kcn_raw: None,
            lpar_raw: None,
            primitives: primitives
                .into_iter()
                .map(|(exponent, coefficient)| crate::sto::PrimitiveGaussian {
                    exponent,
                    coefficient,
                })
                .collect(),
        }
    }

    #[test]
    fn gfn1_basis_overlap_derivatives_match_matrix_finite_difference() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = crate::params::Gfn1Parameters::from_file(param_path).unwrap();
        let xyz = "24
caffeine
N 0.000000 0.000000 0.000000
C 1.250000 0.000000 0.000000
N 2.000000 1.100000 0.000000
C 1.250000 2.200000 0.000000
C 0.000000 2.200000 0.000000
C -0.700000 1.100000 0.000000
N 1.750000 3.350000 0.000000
C 0.750000 4.250000 0.000000
N -0.350000 3.350000 0.000000
O 1.900000 -1.050000 0.000000
O -1.950000 1.100000 0.000000
C -0.800000 -1.200000 0.250000
H -1.830000 -0.880000 0.250000
H -0.550000 -1.780000 1.140000
H -0.550000 -1.820000 -0.620000
C 3.450000 1.100000 0.250000
H 3.800000 2.130000 0.250000
H 3.780000 0.580000 1.150000
H 3.850000 0.540000 -0.600000
C 3.100000 3.900000 0.250000
H 3.060000 4.990000 0.250000
H 3.640000 3.580000 1.140000
H 3.700000 3.520000 -0.580000
H 0.780000 5.330000 0.000000
";
        let system = crate::system::PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let h = 1.0e-5;
        let atom = 0usize;
        let axis = 1usize;
        let mut plus = system.clone();
        let mut minus = system.clone();
        shifted_system(&mut plus, atom, axis, h);
        shifted_system(&mut minus, atom, axis, -h);
        let sp = IntegralMatrices::build(&plus, &basis).unwrap().overlap;
        let sm = IntegralMatrices::build(&minus, &basis).unwrap().overlap;
        let mut max_diff = 0.0_f64;
        let mut max_pair = (0usize, 0usize, 0.0, 0.0);
        for mu in 0..basis.len() {
            let atom_mu = basis.aos[mu].atom_index;
            let rmu = system.atoms[atom_mu].position;
            for nu in 0..mu {
                let atom_nu = basis.aos[nu].atom_index;
                if atom_mu != atom && atom_nu != atom {
                    continue;
                }
                let rnu = system.atoms[atom_nu].position;
                let (_, d_bra, d_ket) =
                    contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
                let analytic = match (atom_mu == atom, atom_nu == atom) {
                    (true, true) => component(d_bra[0] + d_ket[0], axis),
                    (true, false) => component(d_bra[0], axis),
                    (false, true) => component(d_ket[0], axis),
                    (false, false) => unreachable!(),
                };
                let fd = (sp[(mu, nu)] - sm[(mu, nu)]) / (2.0 * h);
                let diff = (analytic - fd).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_pair = (mu, nu, analytic, fd);
                }
            }
        }
        println!(
            "max overlap derivative diff {max_diff:.3e} pair {:?}",
            max_pair
        );
        assert!(max_diff < 1.0e-7);
    }

    #[test]
    fn gfn1_h0_derivatives_match_matrix_finite_difference_without_cn() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = crate::params::Gfn1Parameters::from_file(param_path).unwrap();
        let xyz = "24
caffeine
N 0.000000 0.000000 0.000000
C 1.250000 0.000000 0.000000
N 2.000000 1.100000 0.000000
C 1.250000 2.200000 0.000000
C 0.000000 2.200000 0.000000
C -0.700000 1.100000 0.000000
N 1.750000 3.350000 0.000000
C 0.750000 4.250000 0.000000
N -0.350000 3.350000 0.000000
O 1.900000 -1.050000 0.000000
O -1.950000 1.100000 0.000000
C -0.800000 -1.200000 0.250000
H -1.830000 -0.880000 0.250000
H -0.550000 -1.780000 1.140000
H -0.550000 -1.820000 -0.620000
C 3.450000 1.100000 0.250000
H 3.800000 2.130000 0.250000
H 3.780000 0.580000 1.150000
H 3.850000 0.540000 -0.600000
C 3.100000 3.900000 0.250000
H 3.060000 4.990000 0.250000
H 3.640000 3.580000 1.140000
H 3.700000 3.520000 -0.580000
H 0.780000 5.330000 0.000000
";
        let system = crate::system::PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let mut hopt = crate::hamiltonian::HamiltonianOptions::default();
        hopt.enable_cn_hamiltonian = false;
        let core = crate::hamiltonian::build_h0(&system, &basis, &params, &hopt).unwrap();
        let h = 1.0e-5;
        let atom = 0usize;
        let axis = 1usize;
        let mut plus = system.clone();
        let mut minus = system.clone();
        shifted_system(&mut plus, atom, axis, h);
        shifted_system(&mut minus, atom, axis, -h);
        let hp = crate::hamiltonian::build_h0(&plus, &basis, &params, &hopt)
            .unwrap()
            .h0;
        let hm = crate::hamiltonian::build_h0(&minus, &basis, &params, &hopt)
            .unwrap()
            .h0;
        let mut max_diff = 0.0_f64;
        let mut max_pair = (0usize, 0usize, 0.0, 0.0);
        for mu in 0..basis.len() {
            let atom_mu = basis.aos[mu].atom_index;
            let shell_mu_index = basis.aos[mu].shell_index;
            let shell_mu = &basis.shells[shell_mu_index];
            let rmu = system.atoms[atom_mu].position;
            for nu in 0..mu {
                let atom_nu = basis.aos[nu].atom_index;
                if atom_mu == atom_nu || (atom_mu != atom && atom_nu != atom) {
                    continue;
                }
                let shell_nu_index = basis.aos[nu].shell_index;
                let shell_nu = &basis.shells[shell_nu_index];
                let rnu = system.atoms[atom_nu].position;
                let rvec = rmu - rnu;
                let r2 = rvec.norm2();
                let (moments, d_bra, d_ket) =
                    contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
                let overlap = moments[0];
                let hs = crate::hamiltonian::hscale(shell_mu, shell_nu, &params).unwrap()
                    * crate::hamiltonian::shell_polynomial(
                        shell_mu,
                        shell_nu,
                        (r2.sqrt()
                            / (crate::data_tables::atomic_radius_bohr(shell_mu.z).unwrap()
                                + crate::data_tables::atomic_radius_bohr(shell_nu.z).unwrap()))
                        .sqrt(),
                    );
                let hij = 0.5
                    * (core.self_energies[shell_mu_index] + core.self_energies[shell_nu_index])
                    * hs;
                let d_overlap = if atom_mu == atom {
                    component(d_bra[0], axis)
                } else {
                    component(d_ket[0], axis)
                };
                let dlog = h0_poly_log_derivative(shell_mu, shell_nu, rvec, r2);
                let dlog_component = if atom_mu == atom {
                    component(dlog, axis)
                } else {
                    -component(dlog, axis)
                };
                let analytic = hij * d_overlap + overlap * hij * dlog_component;
                let fd = (hp[(mu, nu)] - hm[(mu, nu)]) / (2.0 * h);
                let diff = (analytic - fd).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_pair = (mu, nu, analytic, fd);
                }
            }
        }
        println!("max h0 derivative diff {max_diff:.3e} pair {:?}", max_pair);
        assert!(max_diff < 1.0e-7);
    }

    fn h0_poly_log_derivative(
        si: &crate::basis::BasisShell,
        sj: &crate::basis::BasisShell,
        rvec: Vec3,
        r2: f64,
    ) -> Vec3 {
        let rad_sum = crate::data_tables::atomic_radius_bohr(si.z).unwrap()
            + crate::data_tables::atomic_radius_bohr(sj.z).unwrap();
        let rr = (r2.sqrt() / rad_sum).sqrt();
        let pi = si.poly_raw.unwrap_or(0.0);
        let pj = sj.poly_raw.unwrap_or(0.0);
        let fi = 1.0 + pi * rr;
        let fj = 1.0 + pj * rr;
        let poly = fi * fj;
        let dpoly = (fi * pj + fj * pi) * 0.5 * rr / r2;
        rvec * (dpoly / poly)
    }

    fn shifted_system(
        system: &mut crate::system::PeriodicSystem,
        atom: usize,
        axis: usize,
        h: f64,
    ) {
        match axis {
            0 => system.atoms[atom].position.x += h,
            1 => system.atoms[atom].position.y += h,
            2 => system.atoms[atom].position.z += h,
            _ => panic!("axis out of range"),
        }
    }
}
