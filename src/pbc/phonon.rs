// SPDX-License-Identifier: GPL-3.0-or-later
//! Commensurate finite-q phonons from the analytic supercell CP-SCC Hessian.
//! The reciprocal-space projection is an exact zone-folded response for the
//! selected supercell sampling, not a finite displacement of the nuclei.
//! Primitive-cell k<->k+q Sternheimer, arbitrary-q interpolation and macroscopic
//! LO-TO corrections are separate backends, not claimed by this implementation.
use super::complex::{native_hermitian_eigen, CMatrix};
use super::{pbc_gamma_hessian, PbcOptions};
use crate::error::{Gfn1Error, Result};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::Mat3;
use crate::{Atom, BasisSet, ElectronicOptions, Gfn1Parameters, PeriodicSystem};

#[derive(Clone, Copy, Debug)]
pub struct PhononSupercellOptions {
    pub repetitions: [usize; 3],
    pub max_atoms: usize,
    pub memory_budget_bytes: usize,
}
impl Default for PhononSupercellOptions {
    fn default() -> Self {
        Self {
            repetitions: [1, 1, 1],
            max_atoms: 128,
            memory_budget_bytes: 512 * 1024 * 1024,
        }
    }
}
pub struct AnalyticSupercellPhonons {
    pub supercell: PeriodicSystem,
    pub cartesian_hessian: Matrix,
    pub repetitions: [usize; 3],
    pub electronic_temperature_kelvin: f64,
    /// Electronic free energy of the same converged supercell reference, divided
    /// by the number of primitive copies. Reuse it for consistent QHA sampling.
    pub reference_energy_hartree_per_primitive_cell: f64,
    /// Identity of the learned Hamiltonian used by the reference, when active.
    pub ml_model_sha256: Option<String>,
    pub raw_hessian_asymmetry: f64,
    pub max_spinor_response_residual: Option<f64>,
    method_citations: Vec<&'static str>,
    primitive_atoms: usize,
    translations: Vec<[usize; 3]>,
    masses: Vec<f64>,
}
pub struct PhononResult {
    pub backend: &'static str,
    pub q_fractional: [f64; 3],
    pub dynamical_matrix: CMatrix,
    pub eigenvalues_hartree_per_bohr2_amu: Vec<f64>,
    pub wavenumbers_cm1: Vec<f64>,
    pub mass_weighted_eigenvectors: CMatrix,
    pub raw_hermiticity_error: f64,
    pub method_citations: Vec<&'static str>,
}

impl AnalyticSupercellPhonons {
    pub fn build(
        primitive: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &ElectronicOptions,
        pbc: &PbcOptions,
        masses_amu: &[f64],
        limits: PhononSupercellOptions,
    ) -> Result<Self> {
        Self::build_impl(primitive, params, electronic, pbc, masses_amu, limits, None)
    }
    /// Physical native-spinor analytic supercell response with explicit zeta.
    pub fn build_soc(
        primitive: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &ElectronicOptions,
        pbc: &PbcOptions,
        masses_amu: &[f64],
        limits: PhononSupercellOptions,
        soc: &crate::soc::SocParameters,
    ) -> Result<Self> {
        Self::build_impl(
            primitive,
            params,
            electronic,
            pbc,
            masses_amu,
            limits,
            Some(soc),
        )
    }
    fn build_impl(
        primitive: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &ElectronicOptions,
        pbc: &PbcOptions,
        masses_amu: &[f64],
        limits: PhononSupercellOptions,
        soc: Option<&crate::soc::SocParameters>,
    ) -> Result<Self> {
        let lattice = primitive.lattice.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput("phonons require a periodic primitive cell".into())
        })?;
        let reps = limits.repetitions;
        if primitive.atoms.is_empty()
            || masses_amu.len() != primitive.atoms.len()
            || masses_amu.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || reps.contains(&0)
            || (0..3).any(|i| !lattice.periodic[i] && reps[i] != 1)
            || !pbc.kmesh.is_gamma_only()
            || electronic.solvation.is_some()
            || (soc.is_some() && electronic.ml_model.is_some())
            || electronic.spin_polarization
            || electronic.plus_u
            || electronic.multipole
            || electronic.lr_exchange
            || electronic.experimental_d4
            || electronic.external_field.is_active()
        {
            return Err(Gfn1Error::InvalidInput(
                "unsupported phonon supercell, masses, sampling or active energy terms".into(),
            ));
        }
        let copies = reps
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Gfn1Error::InvalidInput("supercell size overflow".into()))?;
        let nat = primitive
            .atoms
            .len()
            .checked_mul(copies)
            .ok_or_else(|| Gfn1Error::InvalidInput("supercell atom count overflow".into()))?;
        if nat > limits.max_atoms {
            return Err(Gfn1Error::InvalidInput(format!(
                "phonon supercell has {nat} atoms; limit {}",
                limits.max_atoms
            )));
        }
        let mut atoms = Vec::with_capacity(nat);
        let mut translations = Vec::with_capacity(copies);
        for x in 0..reps[0] {
            for y in 0..reps[1] {
                for z in 0..reps[2] {
                    let translation = lattice.cell.col[0] * x as f64
                        + lattice.cell.col[1] * y as f64
                        + lattice.cell.col[2] * z as f64;
                    translations.push([x, y, z]);
                    atoms.extend(primitive.atoms.iter().map(|a| Atom {
                        z: a.z,
                        position: a.position + translation,
                    }));
                }
            }
        }
        let cell = Mat3::from_columns(
            lattice.cell.col[0] * reps[0] as f64,
            lattice.cell.col[1] * reps[1] as f64,
            lattice.cell.col[2] * reps[2] as f64,
        );
        let supercell = PeriodicSystem {
            atoms,
            lattice: Some(Lattice::new(cell, lattice.periodic)?),
            charge: primitive.charge * copies as f64,
        };
        let basis = BasisSet::build(
            &supercell,
            params,
            crate::BasisOptions {
                nprim: electronic.nprim,
            },
        )?;
        let d = nat
            .checked_mul(3)
            .ok_or_else(|| Gfn1Error::InvalidInput("phonon DOF overflow".into()))?;
        let bytes = basis
            .len()
            .checked_mul(basis.len())
            .and_then(|n| n.checked_mul(d))
            .and_then(|n| n.checked_mul(if soc.is_some() { 48 * 8 } else { 16 * 8 }))
            .and_then(|n| {
                d.checked_mul(d)
                    .and_then(|d| d.checked_mul(8 * 8))
                    .and_then(|d| n.checked_add(d))
            })
            .ok_or_else(|| Gfn1Error::InvalidInput("phonon workspace overflow".into()))?;
        let memory_budget = crate::resources::effective_memory_budget(limits.memory_budget_bytes);
        if bytes > memory_budget {
            return Err(Gfn1Error::InvalidInput(format!(
                "analytic phonon workspace estimate {bytes} bytes exceeds budget {}",
                memory_budget
            )));
        }
        drop(basis);
        let mut options = electronic.clone();
        options.charge = Some(electronic.charge.unwrap_or(primitive.charge) * copies as f64);
        let (result, max_spinor_response_residual) = if let Some(soc) = soc {
            let result = super::pbc_soc_hessian(&supercell, params, &options, pbc, soc)?;
            (
                super::PbcHessianResult {
                    scf: result.soc.scalar_proxy,
                    hessian: result.hessian,
                    raw_max_asymmetry: result.raw_max_asymmetry,
                },
                Some(result.max_response_residual),
            )
        } else {
            (pbc_gamma_hessian(&supercell, params, &options, pbc)?, None)
        };
        if !result.scf.converged {
            return Err(Gfn1Error::InvalidInput(
                "phonon supercell SCF did not converge".into(),
            ));
        }
        let mut method_citations = result.scf.method_citations.clone();
        method_citations.push("Baroni2001DFPT");
        Ok(Self {
            supercell,
            cartesian_hessian: result.hessian,
            repetitions: reps,
            electronic_temperature_kelvin: result.scf.electronic_temperature,
            reference_energy_hartree_per_primitive_cell: result.scf.total_free / copies as f64,
            ml_model_sha256: result
                .scf
                .ml_parameters
                .as_ref()
                .map(|view| view.model_sha256.clone()),
            raw_hessian_asymmetry: result.raw_max_asymmetry,
            max_spinor_response_residual,
            method_citations,
            primitive_atoms: primitive.atoms.len(),
            translations,
            masses: masses_amu.to_vec(),
        })
    }

    /// Cell-periodic phase convention exp(i 2pi q.T). The translation average
    /// reduces numerical SCF asymmetry but is not used to certify a sum rule.
    pub fn at_q(&self, q: [f64; 3]) -> Result<PhononResult> {
        if q.iter().any(|x| !x.is_finite())
            || (0..3).any(|i| {
                let x = q[i] * self.repetitions[i] as f64;
                (x - x.round()).abs() > 1e-10
            })
        {
            return Err(Gfn1Error::InvalidInput(
                "q is not commensurate with the analytic supercell".into(),
            ));
        }
        let d = 3 * self.primitive_atoms;
        let copies = self.translations.len();
        let mut dynamical = CMatrix::zeros(d);
        for (a, ta) in self.translations.iter().enumerate() {
            for (b, tb) in self.translations.iter().enumerate() {
                let phase = 2.0
                    * std::f64::consts::PI
                    * (0..3)
                        .map(|i| q[i] * (tb[i] as f64 - ta[i] as f64))
                        .sum::<f64>();
                let (im, re) = phase.sin_cos();
                for i in 0..d {
                    for j in 0..d {
                        let value = self.cartesian_hessian[(a * d + i, b * d + j)]
                            / (copies as f64 * (self.masses[i / 3] * self.masses[j / 3]).sqrt());
                        dynamical.re[(i, j)] += re * value;
                        dynamical.im[(i, j)] += im * value;
                    }
                }
            }
        }
        let mut hermiticity = 0.0_f64;
        for i in 0..d {
            for j in 0..d {
                hermiticity = hermiticity
                    .max((dynamical.re[(i, j)] - dynamical.re[(j, i)]).abs())
                    .max((dynamical.im[(i, j)] + dynamical.im[(j, i)]).abs());
            }
        }
        let eigen = native_hermitian_eigen(&dynamical)?;
        let frequencies = eigen
            .values
            .iter()
            .map(|&x| {
                x.signum()
                    * (x.abs() / crate::constants::AMU_TO_ELECTRON_MASS).sqrt()
                    * crate::constants::HARTREE_TO_WAVENUMBER
            })
            .collect();
        Ok(PhononResult {
            backend: "analytic commensurate supercell CP-SCC",
            q_fractional: q,
            dynamical_matrix: dynamical,
            eigenvalues_hartree_per_bohr2_amu: eigen.values,
            wavenumbers_cm1: frequencies,
            mass_weighted_eigenvectors: eigen.vectors,
            raw_hermiticity_error: hermiticity,
            method_citations: self.method_citations.clone(),
        })
    }
}
