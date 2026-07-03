// SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::model::Cutoffs;
use crate::pairlist::for_each_unique_short_range_pair;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct RepulsionResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct RepulsionHessianResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
}

pub fn repulsion_energy(system: &PeriodicSystem, params: &Gfn1Parameters) -> Result<f64> {
    let mut energy = 0.0;
    let cutoff = Cutoffs::default().repulsion;
    for_each_unique_short_range_pair(system, cutoff, |pair| {
        energy += repulsion_pair_energy(
            system.atoms[pair.i].z,
            system.atoms[pair.j].z,
            pair.r,
            params,
        )?;
        Ok(())
    })?;
    Ok(energy)
}

pub fn repulsion_energy_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<RepulsionResult> {
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); system.atoms.len()];
    let cutoff = Cutoffs::default().repulsion;
    for_each_unique_short_range_pair(system, cutoff, |pair| {
        let za = system.atoms[pair.i].z;
        let zb = system.atoms[pair.j].z;
        let (value, prefactor) = repulsion_pair_energy_gradient_prefactor(za, zb, pair.r, params)?;
        energy += value;
        if pair.i != pair.j {
            let dgi = pair.dr * prefactor;
            gradient[pair.i] += dgi;
            gradient[pair.j] -= dgi;
        }
        Ok(())
    })?;
    Ok(RepulsionResult { energy, gradient })
}

pub fn repulsion_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<Option<Matrix>> {
    let Some(lattice) = system.lattice.as_ref() else {
        return Ok(None);
    };
    let mut stress = Matrix::zeros(3, 3);
    let cutoff = Cutoffs::default().repulsion;
    for_each_unique_short_range_pair(system, cutoff, |pair| {
        let za = system.atoms[pair.i].z;
        let zb = system.atoms[pair.j].z;
        let (_, prefactor) = repulsion_pair_energy_gradient_prefactor(za, zb, pair.r, params)?;
        let dr = pair.dr.to_array();
        for row in 0..3 {
            for col in 0..3 {
                stress[(row, col)] -= prefactor * dr[row] * dr[col];
            }
        }
        Ok(())
    })?;
    let inv_volume = 1.0 / lattice.volume();
    for value in stress.as_mut_slice() {
        *value *= inv_volume;
    }
    Ok(Some(stress))
}

pub fn repulsion_energy_gradient_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<RepulsionHessianResult> {
    let nat = system.atoms.len();
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
    let cutoff = Cutoffs::default().repulsion;
    for_each_unique_short_range_pair(system, cutoff, |pair| {
        let za = system.atoms[pair.i].z;
        let zb = system.atoms[pair.j].z;
        let pair_result = repulsion_pair_energy_derivatives(za, zb, pair.r, params)?;
        energy += pair_result.energy;
        if pair.i != pair.j {
            let dgi = pair.dr * pair_result.gradient_prefactor;
            gradient[pair.i] += dgi;
            gradient[pair.j] -= dgi;
            add_radial_hessian_block(
                &mut hessian,
                pair.i,
                pair.j,
                pair.dr,
                pair_result.gradient_prefactor,
                pair_result.gradient_prefactor_derivative,
            );
        }
        Ok(())
    })?;
    Ok(RepulsionHessianResult {
        energy,
        gradient,
        hessian,
    })
}

pub fn repulsion_pair_energy(za: u8, zb: u8, r: f64, params: &Gfn1Parameters) -> Result<f64> {
    Ok(repulsion_pair_energy_gradient_prefactor(za, zb, r, params)?.0)
}

#[derive(Clone, Copy, Debug)]
struct PairRepulsionDerivatives {
    energy: f64,
    gradient_prefactor: f64,
    gradient_prefactor_derivative: f64,
    /// Third radial derivative `f'''` of the pair energy `f(r)`, needed for the analytic
    /// third nuclear derivative (the central rank-3 block).
    radial_third_derivative: f64,
}

fn repulsion_pair_energy_gradient_prefactor(
    za: u8,
    zb: u8,
    r: f64,
    params: &Gfn1Parameters,
) -> Result<(f64, f64)> {
    let pa = params.element(za)?;
    let pb = params.element(zb)?;
    let alpha = (pa.repa * pb.repa).sqrt();
    let zeff = pa.repb * pb.repb;
    let kexp = if za <= 2 && zb <= 2 {
        params.global("kexplight", params.global("kexp", 1.5))
    } else {
        params.global("kexp", 1.5)
    };
    let rexp = 1.0;
    let rk = r.powf(kexp);
    let energy = zeff * (-alpha * rk).exp() / r.powf(rexp);
    let prefactor = (alpha * rk * kexp + rexp) * energy / (r * r);
    Ok((energy, prefactor))
}

fn repulsion_pair_energy_derivatives(
    za: u8,
    zb: u8,
    r: f64,
    params: &Gfn1Parameters,
) -> Result<PairRepulsionDerivatives> {
    let pa = params.element(za)?;
    let pb = params.element(zb)?;
    let alpha = (pa.repa * pb.repa).sqrt();
    let zeff = pa.repb * pb.repb;
    let kexp = if za <= 2 && zb <= 2 {
        params.global("kexplight", params.global("kexp", 1.5))
    } else {
        params.global("kexp", 1.5)
    };
    let rexp = 1.0;
    let rk = r.powf(kexp);
    let energy = zeff * (-alpha * rk).exp() / r.powf(rexp);
    let b = alpha * kexp * rk + rexp;
    let prefactor = b * energy / (r * r);
    let db_dr = alpha * kexp * kexp * r.powf(kexp - 1.0);
    let dlog_energy_dr = -alpha * kexp * r.powf(kexp - 1.0) - rexp / r;
    let dprefactor_dr = prefactor * (db_dr / b + dlog_energy_dr - 2.0 / r);
    // Second derivative of the prefactor `p = f'/r` from the radial ladder of
    // `f = zeff·exp(-α r^k)/r`. With `L1 = f'/f = -α k r^{k-1} - 1/r` and its derivatives,
    // `f' = f L1`, `f'' = f(L1²+L1')`, `f''' = f(L1³+3 L1 L1'+L1'')`, and
    // `p'' = -f'''/r + 2 f''/r² − 2 f'/r³`. (p, p' from this ladder equal the closed forms
    // above; only p'' is new.)
    let l1 = -alpha * kexp * r.powf(kexp - 1.0) - rexp / r;
    let l1p = -alpha * kexp * (kexp - 1.0) * r.powf(kexp - 2.0) + rexp / (r * r);
    let l1pp =
        -alpha * kexp * (kexp - 1.0) * (kexp - 2.0) * r.powf(kexp - 3.0) - 2.0 * rexp / (r * r * r);
    let fppp = energy * (l1 * l1 * l1 + 3.0 * l1 * l1p + l1pp);
    Ok(PairRepulsionDerivatives {
        energy,
        gradient_prefactor: prefactor,
        gradient_prefactor_derivative: dprefactor_dr,
        radial_third_derivative: fppp,
    })
}

/// Analytic third Cartesian derivative `T[a][b][c] = ∂³E_rep/∂R_a∂R_b∂R_c` of the
/// (non-PBC) repulsion energy, returned as `ndof` slabs (one `ndof×ndof` matrix per third
/// index `c`). Repulsion is a classical pair sum with no electronic response, so this is
/// purely geometric — the 2n+1 driver consumes it as the `L_abc` frozen block for the
/// repulsion term. Slab `c` equals `∂(Hessian)/∂R_c`, which is exactly the FD gate.
pub fn repulsion_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<Vec<Matrix>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut tensor = vec![Matrix::zeros(ndof, ndof); ndof];
    let cutoff = Cutoffs::default().repulsion;
    for_each_unique_short_range_pair(system, cutoff, |pair| {
        if pair.i == pair.j {
            return Ok(());
        }
        let za = system.atoms[pair.i].z;
        let zb = system.atoms[pair.j].z;
        let d = repulsion_pair_energy_derivatives(za, zb, pair.r, params)?;
        // `pair.dr = R_j − R_i`, so the true relative vector `R_i − R_j` is its negation,
        // and `g = f''/r − f'/r² = −(gradient_prefactor_derivative)`.
        crate::third_derivative::add_radial_third_block(
            &mut tensor,
            pair.i,
            pair.j,
            pair.dr * (-1.0),
            -d.gradient_prefactor_derivative,
            d.radial_third_derivative,
            1.0,
        );
        Ok(())
    })?;
    Ok(tensor)
}

fn add_radial_hessian_block(
    hessian: &mut Matrix,
    i: usize,
    j: usize,
    dr: Vec3,
    gradient_prefactor: f64,
    gradient_prefactor_derivative: f64,
) {
    let r = dr.norm();
    if r <= 1.0e-12 {
        return;
    }
    let unit = dr / r;
    let u = unit.to_array();
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            let value =
                -gradient_prefactor * delta - r * gradient_prefactor_derivative * u[a] * u[b];
            let ia = 3 * i + a;
            let ib = 3 * i + b;
            let ja = 3 * j + a;
            let jb = 3 * j + b;
            hessian[(ia, ib)] += value;
            hessian[(ja, jb)] += value;
            hessian[(ia, jb)] -= value;
            hessian[(ja, ib)] -= value;
        }
    }
}

#[cfg(test)]
mod hessian_tests {
    use super::{repulsion_energy_gradient, repulsion_energy_gradient_hessian};
    use crate::math::Vec3;
    use crate::params::Gfn1Parameters;
    use crate::system::PeriodicSystem;

    #[test]
    fn repulsion_hessian_matches_gradient_finite_difference() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let analytic = repulsion_energy_gradient_hessian(&system, &params).unwrap();
        let step = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let gp = repulsion_energy_gradient(&plus, &params).unwrap().gradient;
            let gm = repulsion_energy_gradient(&minus, &params).unwrap().gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
                max_delta = max_delta.max((analytic.hessian[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-7,
            "repulsion Hessian finite-difference max delta {max_delta:.3e}"
        );
    }

    // The analytic repulsion third derivative (slab c = ∂H/∂R_c) must match the central
    // finite difference of the analytic repulsion Hessian — the chosen FD-vs-Hessian gate.
    #[test]
    fn repulsion_third_derivative_matches_hessian_finite_difference() {
        let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
            return;
        };
        let params = Gfn1Parameters::from_file(param_path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let third = super::repulsion_third_derivative(&system, &params).unwrap();
        let step = 1.0e-4;
        let ndof = 3 * system.atoms.len();
        let mut max_delta = 0.0_f64;
        for slab in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, slab, step);
            displace(&mut minus, slab, -step);
            let hp = repulsion_energy_gradient_hessian(&plus, &params)
                .unwrap()
                .hessian;
            let hm = repulsion_energy_gradient_hessian(&minus, &params)
                .unwrap()
                .hessian;
            for row in 0..ndof {
                for col in 0..ndof {
                    let fd = (hp[(row, col)] - hm[(row, col)]) / (2.0 * step);
                    max_delta = max_delta.max((third[slab][(row, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "repulsion third-derivative FD-vs-Hessian max delta {max_delta:.3e}"
        );
    }

    fn displace(system: &mut PeriodicSystem, dof: usize, step: f64) {
        let atom = dof / 3;
        match dof % 3 {
            0 => system.atoms[atom].position.x += step,
            1 => system.atoms[atom].position.y += step,
            _ => system.atoms[atom].position.z += step,
        }
    }

    fn component(values: &[Vec3], dof: usize) -> f64 {
        let atom = dof / 3;
        match dof % 3 {
            0 => values[atom].x,
            1 => values[atom].y,
            _ => values[atom].z,
        }
    }
}
