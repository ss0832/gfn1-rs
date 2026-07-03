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
use crate::error::Result;
use crate::linalg::{symmetric_eigen_jacobi, Matrix};

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

    let eig = symmetric_eigen_jacobi(&mw, 1.0e-12, 100 * ndof.max(1) * ndof.max(1))?;

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
