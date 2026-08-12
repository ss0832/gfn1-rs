// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic second-order isotropic electrostatics for GFN1-xTB.
//!
//! GFN1 uses the Klopman-Ohno (KO) shell-shell interaction
//! `gamma(R) = 1 / sqrt(R^2 + eta^-2)`, with `eta` the harmonic mean of the two
//! shell hardnesses. Because `gamma -> 1/R` at long range, a bare real-space
//! lattice sum is only conditionally convergent. Following the tblite / CP2K
//! periodic scheme (Buccheri et al. 2025, JCTC Eq. 11-24) the interaction is
//! split by a binomial expansion as
//!
//! ```text
//!   gamma(R) = [1/R - 1/(2 eta^2 R^3)]        (long range, Ewald summed)
//!            + [gamma(R) - 1/R + 1/(2 eta^2 R^3)]  (short range)
//! ```
//!
//! The `1/R` part is evaluated with a standard Ewald summation; the `R^-3`
//! binomial term uses the QCore generalized Ewald expression, including the
//! special `k = 0` contribution from Eq. 24. The remaining short-range residual
//! decays as `R^-5` and is summed directly in real space. The intra-atomic
//! on-site term is the KO `R -> 0` limit `eta`, added directly.
//!
//! The result is assembled into a shell-resolved interaction matrix `Gamma`,
//! identical in role to the molecular `effective_coulomb_matrix`, so the existing
//! `coulomb_energy_potential_from_matrix` (second order, third order, potential)
//! is reused unchanged. `Gamma` is k-independent: the shell charges live in the
//! reference cell.
//!
use crate::basis::BasisSet;
use crate::coulomb::{harmonic_average, ShellChargeModel};
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::erfc;
use crate::pbc::EwaldOptions;
use crate::system::PeriodicSystem;
use std::f64::consts::PI;

const SQRT_PI: f64 = 1.772_453_850_905_516;
pub(crate) const QCORE_R3_COEFF: f64 = -0.5;
/// Convergence factor: `erfc(TAU)` and `exp(-TAU^2)` are ~1e-14.
pub(crate) const TAU: f64 = 5.5;
const DIST_EPS: f64 = 1.0e-12;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const DIGAMMA_3_2: f64 = 0.036_489_973_978_576_52;

/// Resolve the Ewald splitting parameter `alpha` (Gaussian width). When the
/// caller leaves it unset, a volume-balanced value keeps the real- and
/// reciprocal-space work cell-size independent.
pub fn resolve_alpha(system: &PeriodicSystem, opts: &EwaldOptions) -> f64 {
    if let Some(a) = opts.k_split {
        return a;
    }
    match &system.lattice {
        Some(lattice) => {
            let v = lattice.volume().max(1.0e-6);
            SQRT_PI / v.cbrt()
        }
        None => 1.0,
    }
}

/// Smooth truncation polynomial `f_smooth(r)` (SI Eq. 5): 1 below `rc - dr`,
/// 0 above `rc`, quintic smootherstep in between.
#[inline]
pub fn f_smooth(r: f64, rc: f64, dr: f64) -> f64 {
    if dr <= 0.0 {
        return if r <= rc { 1.0 } else { 0.0 };
    }
    if r <= rc - dr {
        1.0
    } else if r >= rc {
        0.0
    } else {
        let t = (r - (rc - dr)) / dr;
        1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    }
}

/// Derivative `d f_smooth / d r`.
#[inline]
pub fn f_smooth_derivative(r: f64, rc: f64, dr: f64) -> f64 {
    if dr <= 0.0 || r <= rc - dr || r >= rc {
        return 0.0;
    }
    let t = (r - (rc - dr)) / dr;
    -(30.0 / dr) * t * t * (1.0 - t) * (1.0 - t)
}

/// Klopman-Ohno value and radial derivative `d gamma / d r` at distance `r`.
#[inline]
pub fn ko_value_derivative(r: f64, eta: f64) -> (f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let denom = (r * r + inv_eta2).sqrt();
    let value = 1.0 / denom;
    let dvalue = -r / (denom * denom * denom);
    (value, dvalue)
}

#[inline]
pub(crate) fn ko_value_derivatives(r: f64, eta: f64) -> (f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let denom2 = r * r + inv_eta2;
    let denom = denom2.sqrt();
    let value = 1.0 / denom;
    let dvalue = -r / denom2.powf(1.5);
    let d2value = 3.0 * r * r / denom2.powf(2.5) - 1.0 / denom2.powf(1.5);
    (value, dvalue, d2value)
}

#[inline]
pub(crate) fn qcore_short_value_derivatives(r: f64, eta: f64) -> (f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let (ko, dko, d2ko) = ko_value_derivatives(r, eta);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let value = ko - 1.0 / r + 0.5 * inv_eta2 / r3;
    let dvalue = dko + 1.0 / r2 - 1.5 * inv_eta2 / r4;
    let d2value = d2ko - 2.0 / r3 + 6.0 * inv_eta2 / r5;
    (value, dvalue, d2value)
}

#[inline]
pub(crate) fn qcore_r3_real_value_derivatives(r: f64, eta: f64, alpha: f64) -> (f64, f64, f64) {
    let inv_eta2 = 1.0 / (eta * eta);
    let ar = alpha * r;
    let exp_ar2 = (-(ar * ar)).exp();
    let q = erfc(ar) + (2.0 * ar / SQRT_PI) * exp_ar2;
    let dq = -(4.0 * alpha * alpha * alpha * r * r / SQRT_PI) * exp_ar2;
    let d2q = (8.0 * alpha * alpha * alpha * r / SQRT_PI) * exp_ar2 * (ar * ar - 1.0);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let value = inv_eta2 * q / r3;
    let dvalue = inv_eta2 * (dq / r3 - 3.0 * q / r4);
    let d2value = inv_eta2 * (d2q / r3 - 6.0 * dq / r4 + 12.0 * q / r5);
    (value, dvalue, d2value)
}

#[inline]
pub(crate) fn exp1(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    if x <= 1.0 {
        let mut ans = -EULER_GAMMA - x.ln();
        let mut fact = 1.0;
        for k in 1..=200 {
            fact *= -x / k as f64;
            let term = -fact / k as f64;
            ans += term;
            if term.abs() <= 1.0e-16 * ans.abs().max(1.0) {
                break;
            }
        }
        ans
    } else {
        let fpmin = 1.0e-300;
        let mut b = x + 1.0;
        let mut c = 1.0 / fpmin;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200 {
            let a = -(i as f64) * (i as f64);
            b += 2.0;
            let mut denom_d = a * d + b;
            if denom_d.abs() < fpmin {
                denom_d = fpmin;
            }
            d = 1.0 / denom_d;
            c = b + a / c;
            if c.abs() < fpmin {
                c = fpmin;
            }
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() <= 1.0e-14 {
                break;
            }
        }
        h * (-x).exp()
    }
}

#[inline]
pub(crate) fn qcore_k_parameter(alpha: f64) -> f64 {
    alpha / SQRT_PI
}

#[inline]
pub(crate) fn qcore_r3_k0_log(alpha: f64) -> f64 {
    let k = qcore_k_parameter(alpha);
    k.ln() + 0.5 * (PI.ln() - DIGAMMA_3_2)
}

/// Standard Ewald lattice potential matrix between atom centres for a `1/R`
/// interaction (real-space `erfc`, reciprocal structure factor, self and
/// neutralising-background corrections). Element `(A,B)` is the periodic
/// Madelung potential felt by a unit charge on `A` from a unit charge on `B`
/// and all its images.
pub fn ewald_atom_matrix(system: &PeriodicSystem, alpha: f64) -> Result<Matrix> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("ewald_atom_matrix requires a periodic system");
    let nat = system.atoms.len();
    let volume = lattice.volume();
    let real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let images = lattice.image_offsets(real_cut);
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / volume;
    let self_term = -2.0 * alpha / SQRT_PI;
    let background = -PI / (alpha * alpha * volume);

    let translations: Vec<_> = images.iter().map(|o| lattice.translation(*o)).collect();

    let mut phi = Matrix::zeros(nat, nat);
    for a in 0..nat {
        for b in 0..=a {
            let rab = system.atoms[a].position - system.atoms[b].position;
            let mut real = 0.0;
            for t in &translations {
                let d = (rab - *t).norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                real += erfc(alpha * d) / d;
            }
            let mut rec = 0.0;
            for (_, g) in &recip {
                let g2 = g.norm2();
                rec += (-g2 * inv_4a2).exp() / g2 * g.dot(rab).cos();
            }
            let mut value = real + rec * four_pi_v + background;
            if a == b {
                value += self_term;
            }
            phi[(a, b)] = value;
            phi[(b, a)] = value;
        }
    }
    Ok(phi)
}

/// Build the periodic shell-resolved second-order interaction matrix `Gamma`.
///
/// `Gamma[i][j] = phi_AB (Ewald 1/R) + [on-site eta if same atom]
///              - 1/2 eta^-2 phi_AB (generalized Ewald R^-3)
///              + sum_T [KO - 1/R + 1/(2 eta^2 R^3)]`
///
/// where the short-range remainder sum skips the on-site `T = 0` term for shells
/// on the same atom (replaced by the on-site `eta`).
pub fn periodic_gamma_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    model: &ShellChargeModel,
    opts: &EwaldOptions,
) -> Result<Matrix> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_gamma_matrix requires a periodic system");
    let alpha = resolve_alpha(system, opts);
    let phi = ewald_atom_matrix(system, alpha)?;

    let nsh = basis.shells.len();
    let sr_cut = opts.sr_cutoff;
    let sr_images = lattice.image_offsets(sr_cut);
    let sr_translations: Vec<_> = sr_images.iter().map(|o| lattice.translation(*o)).collect();
    let r3_cut = TAU / alpha;
    let r3_images = lattice.image_offsets(r3_cut);
    let r3_translations: Vec<_> = r3_images.iter().map(|o| lattice.translation(*o)).collect();
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let r3_rec_pref = 2.0 * PI / lattice.volume();
    let r3_k0_pref = 4.0 * PI * qcore_r3_k0_log(alpha) / lattice.volume();
    let r3_self_pref = -(4.0 * PI / 3.0) * qcore_k_parameter(alpha).powi(3);

    let mut gamma = Matrix::zeros(nsh, nsh);
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for j in 0..=i {
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let inv_eta2 = 1.0 / (eta * eta);
            let mut value = phi[(ai, aj)];
            if ai == aj {
                value += eta;
            }
            let rab = system.atoms[ai].position - system.atoms[aj].position;
            for (off, t) in r3_images.iter().zip(&r3_translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let d = (rab - *t).norm();
                if d <= DIST_EPS || d > r3_cut {
                    continue;
                }
                value += QCORE_R3_COEFF * qcore_r3_real_value_derivatives(d, eta, alpha).0;
            }
            for (_, g) in &recip {
                let g2 = g.norm2();
                let x = g2 * inv_4a2;
                value += QCORE_R3_COEFF * r3_rec_pref * inv_eta2 * exp1(x) * g.dot(rab).cos();
            }
            value += QCORE_R3_COEFF * r3_k0_pref * inv_eta2;
            if ai == aj {
                value += QCORE_R3_COEFF * r3_self_pref * inv_eta2;
            }
            for (off, t) in sr_images.iter().zip(&sr_translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let d = (rab - *t).norm();
                if d <= DIST_EPS || d > sr_cut {
                    continue;
                }
                value += qcore_short_value_derivatives(d, eta).0;
            }
            gamma[(i, j)] = value;
            gamma[(j, i)] = value;
        }
    }
    Ok(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSet};
    use crate::params::Gfn1Parameters;

    fn load_params() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    fn nacl_like() -> PeriodicSystem {
        // Rocksalt-ish 2-atom cell, just for the electrostatics machinery.
        PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap()
    }

    // The total electrostatic energy must not depend on the Ewald splitting
    // parameter alpha -- the defining correctness property of an Ewald sum.
    #[test]
    fn ewald_energy_is_alpha_independent() {
        let Some(params) = load_params() else {
            return;
        };
        let system = nacl_like();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let model = ShellChargeModel::build(&system, &basis, &params).unwrap();

        let charges: Vec<f64> = (0..basis.shells.len())
            .map(|i| if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();

        let energy_for = |alpha: f64| {
            let opts = EwaldOptions {
                k_split: Some(alpha),
                ..EwaldOptions::default()
            };
            let gamma = periodic_gamma_matrix(&system, &basis, &model, &opts).unwrap();
            let mut e = 0.0;
            for i in 0..basis.shells.len() {
                for j in 0..basis.shells.len() {
                    e += 0.5 * charges[i] * gamma[(i, j)] * charges[j];
                }
            }
            e
        };

        let e1 = energy_for(0.20);
        let e2 = energy_for(0.35);
        let e3 = energy_for(0.50);
        assert!(
            (e1 - e2).abs() < 1.0e-8 && (e1 - e3).abs() < 1.0e-8,
            "alpha dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // A neutral heteronuclear molecule in a large cell reproduces the molecular
    // second-order energy. Individual Gamma elements still carry the (physical)
    // periodic Madelung shift, but for a charge-neutral vector that uniform shift
    // cancels and only the fast-decaying dipole-image interaction remains.
    #[test]
    fn large_cell_matches_molecular_second_order_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let mol =
            PeriodicSystem::from_xyz_str("2\nLiH\nLi 0 0 0\nH 0 0 1.6\n", 0.0, false).unwrap();
        let cell = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"40 0 0 0 40 0 0 0 40\" pbc=\"T T T\"\nLi 0 0 0\nH 0 0 1.6\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();
        let model = ShellChargeModel::build(&mol, &basis, &params).unwrap();

        let nsh = basis.shells.len();
        let mut charges: Vec<f64> = (0..nsh)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.15 })
            .collect();
        let mean: f64 = charges.iter().sum::<f64>() / nsh as f64;
        for q in &mut charges {
            *q -= mean;
        }

        let mol_gamma = crate::coulomb::effective_coulomb_matrix(&mol, &basis, &model);
        let pbc_gamma =
            periodic_gamma_matrix(&cell, &basis, &model, &EwaldOptions::default()).unwrap();

        let energy = |g: &Matrix| -> f64 {
            let mut e = 0.0;
            for i in 0..nsh {
                for j in 0..nsh {
                    e += 0.5 * charges[i] * g[(i, j)] * charges[j];
                }
            }
            e
        };
        let e_mol = energy(&mol_gamma);
        let e_pbc = energy(&pbc_gamma);
        assert!(
            (e_mol - e_pbc).abs() < 1.0e-3,
            "E2 molecular {e_mol:.8} vs periodic {e_pbc:.8}"
        );
    }

    // Alpha-independence must also hold for a NON-neutral charge vector: the
    // neutralising-background term -pi/(alpha^2 V) per pair contributes
    // -pi/(2 alpha^2 V) (sum q)^2 to the energy, which is exactly the charged-cell
    // correction that cancels the alpha-dependence left by omitting the G=0 term.
    // If the background were dropped (treating only neutral cells), this would fail.
    #[test]
    fn ewald_energy_is_alpha_independent_charged() {
        let Some(params) = load_params() else {
            return;
        };
        let system = nacl_like();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let model = ShellChargeModel::build(&system, &basis, &params).unwrap();

        // Deliberately net-charged: charges do not sum to zero.
        let charges: Vec<f64> = (0..basis.shells.len())
            .map(|i| if i % 2 == 0 { 0.4 } else { 0.1 })
            .collect();
        let net: f64 = charges.iter().sum();
        assert!(
            net.abs() > 0.1,
            "test vector must be non-neutral (net {net})"
        );

        let energy_for = |alpha: f64| {
            let opts = EwaldOptions {
                k_split: Some(alpha),
                ..EwaldOptions::default()
            };
            let gamma = periodic_gamma_matrix(&system, &basis, &model, &opts).unwrap();
            let mut e = 0.0;
            for i in 0..basis.shells.len() {
                for j in 0..basis.shells.len() {
                    e += 0.5 * charges[i] * gamma[(i, j)] * charges[j];
                }
            }
            e
        };

        let e1 = energy_for(0.20);
        let e2 = energy_for(0.35);
        let e3 = energy_for(0.50);
        assert!(
            (e1 - e2).abs() < 1.0e-8 && (e1 - e3).abs() < 1.0e-8,
            "charged alpha dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }
}
