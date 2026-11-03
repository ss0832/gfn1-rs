// SPDX-License-Identifier: GPL-3.0-or-later
//! Harmonic vibrational (normal-mode) analysis from a Cartesian Hessian.
//!
//! Given the mass-weighted Hessian `H_ij / sqrt(m_i m_j)` (with `H` the Cartesian
//! second derivative of the energy in Hartree/Bohr^2 and atomic masses in amu),
//! the eigenvalues are `omega^2` in atomic units and the wavenumbers follow as
//! `nu~ [cm^-1] = sign(lambda) * sqrt(|lambda|) * 5140.4843`. The eigenvectors are
//! mass-weighted; the Cartesian normal-mode displacements divide out `sqrt(m)`.
//!
//! For an isolated molecule the three translational modes are zero by translational
//! invariance; at a stationary point the three rotational modes are zero too, so a
//! non-linear equilibrium structure has six near-zero modes. Negative wavenumbers
//! flag imaginary modes (transition states / unstable geometries).

use crate::data_tables::relative_atomic_mass;
use crate::error::{Gfn1Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;

/// Wavenumber (cm^-1) per `sqrt(Hartree / (Bohr^2 * amu))`.
pub const WAVENUMBER_PER_SQRT_AU: f64 = 5140.4843;

/// Result of a harmonic vibrational analysis.
#[derive(Clone, Debug)]
pub struct VibrationalModes {
    /// Harmonic wavenumbers (cm^-1), ascending. Negative entries are imaginary
    /// frequencies (the corresponding mass-weighted eigenvalue is negative).
    pub wavenumbers: Vec<f64>,
    /// Mass-weighted Hessian eigenvalues (Hartree / (Bohr^2 * amu)), ascending.
    pub eigenvalues: Vec<f64>,
    /// Cartesian normal-mode displacement vectors, one per mode (length `3*nat`).
    pub modes: Vec<Vec<f64>>,
}

/// Harmonic frequencies and normal modes from a Cartesian Hessian (Hartree/Bohr^2)
/// and the per-atom atomic numbers (for the masses). The Hessian must be ordered
/// `(3*atom + axis)`.
pub fn vibrational_analysis(hessian: &Matrix, atomic_numbers: &[u8]) -> Result<VibrationalModes> {
    let nat = atomic_numbers.len();
    let ndof = 3 * nat;
    validate_hessian(hessian, ndof)?;
    if atomic_numbers.iter().any(|&z| z == 0 || z > 86) {
        return Err(Gfn1Error::InvalidInput(
            "vibrational analysis requires supported atomic numbers (1..86)".into(),
        ));
    }
    if ndof == 0 {
        return Ok(VibrationalModes {
            wavenumbers: vec![],
            eigenvalues: vec![],
            modes: vec![],
        });
    }

    // Per-DOF mass (amu); DOF i belongs to atom i/3.
    let mass: Vec<f64> = (0..ndof)
        .map(|i| relative_atomic_mass(atomic_numbers[i / 3]).max(1.0e-12))
        .collect();

    // Mass-weighted, symmetrised Hessian.
    let mut mw = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            mw[(i, j)] = hessian[(i, j)] / (mass[i] * mass[j]).sqrt();
        }
    }
    for i in 0..ndof {
        for j in 0..i {
            let avg = 0.5 * (mw[(i, j)] + mw[(j, i)]);
            mw[(i, j)] = avg;
            mw[(j, i)] = avg;
        }
    }

    let eig = symmetric_eigen(&mw)?;

    let mut wavenumbers = Vec::with_capacity(ndof);
    let mut modes = Vec::with_capacity(ndof);
    for k in 0..ndof {
        let lambda = eig.values[k];
        let w = lambda.abs().sqrt() * WAVENUMBER_PER_SQRT_AU;
        wavenumbers.push(if lambda < 0.0 { -w } else { w });
        // Cartesian displacement = mass-weighted eigenvector / sqrt(m).
        let mut mode = vec![0.0; ndof];
        for i in 0..ndof {
            mode[i] = eig.vectors[(i, k)] / mass[i].sqrt();
        }
        modes.push(mode);
    }

    Ok(VibrationalModes {
        wavenumbers,
        eigenvalues: eig.values,
        modes,
    })
}

fn validate_hessian(hessian: &Matrix, ndof: usize) -> Result<()> {
    if hessian.rows() != ndof
        || hessian.cols() != ndof
        || hessian.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(Gfn1Error::InvalidInput(format!(
            "expected a finite {ndof}x{ndof} Cartesian Hessian"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct MolecularVibrationOptions {
    /// Relative principal-inertia threshold for linear geometry; not a cutoff
    /// on the Hessian's lowest eigenvalues.
    pub linearity_tolerance: f64,
    /// Raw, pre-symmetrization max |H_ij-H_ji| in hartree/bohr^2.
    pub hessian_asymmetry_tolerance: f64,
}

impl Default for MolecularVibrationOptions {
    fn default() -> Self {
        Self {
            linearity_tolerance: 1e-10,
            hessian_asymmetry_tolerance: 1e-7,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MolecularVibrations {
    /// Only the vibrational subspace, sorted by eigenvalue. Cartesian modes
    /// obey u_i^T M_amu u_j = delta_ij.
    pub vibrations: VibrationalModes,
    pub masses_amu: Vec<f64>,
    pub center_of_mass_bohr: Vec3,
    /// Orthonormal rigid directions in mass-weighted Cartesian coordinates.
    pub rigid_directions: Vec<Vec<f64>>,
    pub principal_moments_amu_bohr2: [f64; 3],
    pub raw_hessian_asymmetry: f64,
}

impl MolecularVibrations {
    /// Directions for Q with masses in electron masses (hbar=m_e=e=a0=1),
    /// as required by the PES-only VPT2 ladder-operator convention.
    pub fn atomic_unit_directions(&self) -> Vec<Vec<f64>> {
        let factor = crate::constants::AMU_TO_ELECTRON_MASS.sqrt();
        self.vibrations
            .modes
            .iter()
            .map(|v| v.iter().map(|x| x / factor).collect())
            .collect()
    }

    pub fn angular_frequencies_au(&self) -> Result<Vec<f64>> {
        self.vibrations
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x <= 0.0 || !x.is_finite() {
                    Err(Gfn1Error::InvalidInput(format!(
                        "VPT2 requires a stable minimum with positive vibrational eigenvalues; mode {} has signed wavenumber {:.6} cm^-1",
                        i + 1, self.vibrations.wavenumbers[i]
                    )))
                } else {
                    Ok((x / crate::constants::AMU_TO_ELECTRON_MASS).sqrt())
                }
            })
            .collect()
    }
}

fn orthogonalize(v: &mut [f64], basis: &[Vec<f64>]) -> f64 {
    // Twice-reorthogonalized MGS avoids losing orthogonality for near-linear
    // geometries and canonical directions almost contained in the rigid space.
    for _ in 0..2 {
        for u in basis {
            let dot: f64 = u.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(u) {
                *x -= dot * y;
            }
        }
    }
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Explicit mass/Eckart rigid-space removal for an isolated molecule. Geometry
/// determines whether 3N-5 or 3N-6 modes remain; low Hessian eigenvalues are never
/// simply discarded. The raw Hessian is audited before any symmetrization.
/// Projection is meaningful away from stationarity too, but it does not imply
/// H(Omega R)=0 there. The VPT2 driver separately checks the residual gradient.
pub fn molecular_vibrational_analysis(
    hessian: &Matrix,
    positions_bohr: &[Vec3],
    masses_amu: &[f64],
    options: MolecularVibrationOptions,
) -> Result<MolecularVibrations> {
    let nat = masses_amu.len();
    let ndof = nat
        .checked_mul(3)
        .ok_or_else(|| Gfn1Error::InvalidInput("vibrational size overflow".into()))?;
    validate_hessian(hessian, ndof)?;
    if nat == 0
        || positions_bohr.len() != nat
        || masses_amu.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || positions_bohr
            .iter()
            .any(|r| [r.x, r.y, r.z].iter().any(|x| !x.is_finite()))
        || !(options.linearity_tolerance > 0.0 && options.linearity_tolerance < 1.0)
        || !options.hessian_asymmetry_tolerance.is_finite()
        || options.hessian_asymmetry_tolerance < 0.0
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid molecular positions, isotope masses, or projection tolerances".into(),
        ));
    }
    let total_mass: f64 = masses_amu.iter().sum();
    let mut center = Vec3::zero();
    for (&r, &m) in positions_bohr.iter().zip(masses_amu) {
        center = center + r * (m / total_mass);
    }
    let centered: Vec<Vec3> = positions_bohr.iter().map(|&r| r - center).collect();
    let mut inertia = Matrix::zeros(3, 3);
    for (&r, &m) in centered.iter().zip(masses_amu) {
        let x = [r.x, r.y, r.z];
        for i in 0..3 {
            for j in 0..3 {
                inertia[(i, j)] += m * (if i == j { r.norm2() } else { 0.0 }) - m * x[i] * x[j];
            }
        }
    }
    let axes = symmetric_eigen(&inertia)?;
    let mut rigid = Vec::new();
    for axis in 0..3 {
        let mut v = vec![0.0; ndof];
        for (a, &m) in masses_amu.iter().enumerate() {
            v[3 * a + axis] = (m / total_mass).sqrt();
        }
        rigid.push(v);
    }
    let largest = axes.values[2].max(0.0);
    for k in 0..3 {
        if axes.values[k] <= options.linearity_tolerance * largest {
            continue;
        }
        let axis = Vec3::new(
            axes.vectors[(0, k)],
            axes.vectors[(1, k)],
            axes.vectors[(2, k)],
        );
        let mut v = vec![0.0; ndof];
        for a in 0..nat {
            let r = axis.cross(centered[a]) * masses_amu[a].sqrt();
            v[3 * a..3 * a + 3].copy_from_slice(&[r.x, r.y, r.z]);
        }
        let norm = orthogonalize(&mut v, &rigid);
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
            rigid.push(v);
        }
    }
    let nvib = ndof - rigid.len();
    let mut complement = Vec::with_capacity(nvib);
    for j in 0..ndof {
        let mut v = vec![0.0; ndof];
        v[j] = 1.0;
        for _ in 0..2 {
            orthogonalize(&mut v, &rigid);
            orthogonalize(&mut v, &complement);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
            complement.push(v);
        }
        if complement.len() == nvib {
            break;
        }
    }
    if complement.len() != nvib {
        return Err(Gfn1Error::InvalidInput(
            "failed to construct Eckart complement".into(),
        ));
    }
    let mut asymmetry = 0.0_f64;
    let mut weighted = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            asymmetry = asymmetry.max((hessian[(i, j)] - hessian[(j, i)]).abs());
            weighted[(i, j)] = 0.5 * (hessian[(i, j)] + hessian[(j, i)])
                / (masses_amu[i / 3] * masses_amu[j / 3]).sqrt();
        }
    }
    if asymmetry > options.hessian_asymmetry_tolerance {
        return Err(Gfn1Error::InvalidInput(format!(
            "raw Hessian asymmetry {asymmetry:e} exceeds tolerance before projection"
        )));
    }
    let u = Matrix::from_vec(
        ndof,
        nvib,
        (0..ndof)
            .flat_map(|i| complement.iter().map(move |v| v[i]))
            .collect(),
    )?;
    let reduced = u.transpose().matmul(&weighted.matmul(&u)?)?;
    let (eigenvalues, modes) = if nvib == 0 {
        (vec![], vec![])
    } else {
        let eig = symmetric_eigen(&reduced)?;
        let full = u.matmul(&eig.vectors)?;
        let modes = (0..nvib)
            .map(|k| {
                (0..ndof)
                    .map(|i| full[(i, k)] / masses_amu[i / 3].sqrt())
                    .collect()
            })
            .collect();
        (eig.values, modes)
    };
    let wavenumbers = eigenvalues
        .iter()
        .map(|&x| {
            x.signum()
                * (x.abs() / crate::constants::AMU_TO_ELECTRON_MASS).sqrt()
                * crate::constants::HARTREE_TO_WAVENUMBER
        })
        .collect();
    Ok(MolecularVibrations {
        vibrations: VibrationalModes {
            eigenvalues,
            wavenumbers,
            modes,
        },
        masses_amu: masses_amu.to_vec(),
        center_of_mass_bohr: center,
        rigid_directions: rigid,
        principal_moments_amu_bohr2: [axes.values[0], axes.values[1], axes.values[2]],
        raw_hessian_asymmetry: asymmetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A diagonal Hessian of a single harmonic DOF gives the expected wavenumber.
    #[test]
    fn single_oscillator_wavenumber() {
        // One hydrogen atom (mass m amu), Hessian k = 1 Hartree/Bohr^2 on x; y,z zero.
        let mut h = Matrix::zeros(3, 3);
        h[(0, 0)] = 1.0;
        let v = vibrational_analysis(&h, &[1]).unwrap();
        // Highest mode: sqrt(k/m) * factor, with m the tabulated H mass (1.008 amu).
        let m = relative_atomic_mass(1);
        let expected = (1.0_f64 / m).sqrt() * WAVENUMBER_PER_SQRT_AU;
        let top = *v.wavenumbers.last().unwrap();
        assert!(
            (top - expected).abs() < 1.0e-6,
            "single oscillator wavenumber {top}, expected {expected}"
        );
        // The two zero-force directions are ~0.
        assert!(v.wavenumbers[0].abs() < 1.0e-6 && v.wavenumbers[1].abs() < 1.0e-6);
    }

    // Translational invariance: a Hessian satisfying the acoustic sum rule has three
    // exactly-zero modes. Build a tiny 2-atom Hessian from a single bond constant.
    #[test]
    fn translational_modes_are_zero() {
        // Two atoms on x, harmonic bond along x with constant k: H is the 6x6 block
        // [[k,-k],[-k,k]] on the x-x entries (sum rule satisfied), zero elsewhere.
        let nat = 2;
        let ndof = 3 * nat;
        let mut h = Matrix::zeros(ndof, ndof);
        let k = 0.5;
        h[(0, 0)] = k;
        h[(3, 3)] = k;
        h[(0, 3)] = -k;
        h[(3, 0)] = -k;
        let v = vibrational_analysis(&h, &[1, 1]).unwrap();
        // Five zero modes (the only restoring direction is the x stretch).
        let zeros = v.wavenumbers.iter().filter(|w| w.abs() < 1.0e-6).count();
        assert_eq!(zeros, 5, "expected 5 zero modes, got {zeros}");
        // The stretch: reduced mass 1/2 amu, omega^2 = k*(1/m1+1/m2)=k*2 => sqrt(2k/ (mass-weighted)).
        let top = *v.wavenumbers.last().unwrap();
        assert!(top > 0.0, "stretch wavenumber should be positive: {top}");
    }
}
