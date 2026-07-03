// SPDX-License-Identifier: GPL-3.0-or-later

use crate::basis::{BasisSet, BasisShell};
use crate::coordination::{
    coordination_numbers, coordination_with_derivatives, CoordinationOptions,
};
use crate::data_tables::{atomic_radius_bohr, pauling_en};
use crate::error::Result;
use crate::integrals::{IntegralMatrices, IntegralOptions};
use crate::linalg::Matrix;
use crate::model::Cutoffs;
use crate::params::{AngularMomentum, Gfn1Parameters};
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct HamiltonianOptions {
    pub integral_cutoff: f64,
    pub integral_screening: f64,
    pub coordination_cutoff: f64,
    pub enable_cn_hamiltonian: bool,
}

impl Default for HamiltonianOptions {
    fn default() -> Self {
        let cutoffs = Cutoffs::default();
        Self {
            integral_cutoff: cutoffs.integral,
            integral_screening: 0.0,
            coordination_cutoff: cutoffs.coordination,
            enable_cn_hamiltonian: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HamiltonianCore {
    pub integrals: IntegralMatrices,
    pub h0: Matrix,
    pub self_energies: Vec<f64>,
    pub dsedcn: Vec<f64>,
    pub coordination_numbers: Vec<f64>,
}

pub fn build_h0(
    system: &PeriodicSystem,
    basis: &BasisSet,
    params: &Gfn1Parameters,
    options: &HamiltonianOptions,
) -> Result<HamiltonianCore> {
    let cn = if options.enable_cn_hamiltonian {
        coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: options.coordination_cutoff,
                ..CoordinationOptions::default()
            },
        )?
        .cn
    } else {
        vec![0.0; system.atoms.len()]
    };
    build_h0_with_coordination(system, basis, params, options, &cn)
}

pub fn build_h0_with_coordination(
    system: &PeriodicSystem,
    basis: &BasisSet,
    params: &Gfn1Parameters,
    options: &HamiltonianOptions,
    cn: &[f64],
) -> Result<HamiltonianCore> {
    let integrals = IntegralMatrices::build_with_options(
        system,
        basis,
        IntegralOptions {
            cutoff: options.integral_cutoff,
            screening_threshold: options.integral_screening,
        },
    )?;
    let (h0, self_energies, dsedcn) =
        build_h0_from_overlap(system, basis, params, &integrals.overlap, cn)?;
    Ok(HamiltonianCore {
        integrals,
        h0,
        self_energies,
        dsedcn,
        coordination_numbers: cn.to_vec(),
    })
}

pub fn build_h0_from_overlap(
    system: &PeriodicSystem,
    basis: &BasisSet,
    params: &Gfn1Parameters,
    overlap: &Matrix,
    cn: &[f64],
) -> Result<(Matrix, Vec<f64>, Vec<f64>)> {
    let nsh = basis.shells.len();
    let mut self_energies = vec![0.0; nsh];
    let mut dsedcn = vec![0.0; nsh];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let kcn = shell.kcn_raw.unwrap_or(0.0);
        dsedcn[ish] = -kcn;
        self_energies[ish] = shell.hdiag_ha - kcn * cn[shell.atom_index];
    }

    let n = basis.len();
    let mut h0 = Matrix::zeros(n, n);
    for iao in 0..n {
        let ishell = basis.aos[iao].shell_index;
        let si = &basis.shells[ishell];
        for jao in 0..=iao {
            let jshell = basis.aos[jao].shell_index;
            let sj = &basis.shells[jshell];
            let hij = if si.atom_index == sj.atom_index {
                0.5 * (self_energies[ishell] + self_energies[jshell])
            } else {
                let ri = system.atoms[si.atom_index].position;
                let rj = system.atoms[sj.atom_index].position;
                let r2 = (ri - rj).norm2();
                let rad_sum = atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?;
                let rr = (r2.sqrt() / rad_sum).sqrt();
                0.5 * (self_energies[ishell] + self_energies[jshell])
                    * hscale(si, sj, params)?
                    * shell_polynomial(si, sj, rr)
            };
            let value = overlap[(iao, jao)] * hij;
            h0[(iao, jao)] = value;
            h0[(jao, iao)] = value;
        }
    }

    Ok((h0, self_energies, dsedcn))
}

pub fn shell_polynomial(si: &BasisShell, sj: &BasisShell, rr: f64) -> f64 {
    (1.0 + si.poly_raw.unwrap_or(0.0) * rr) * (1.0 + sj.poly_raw.unwrap_or(0.0) * rr)
}

pub fn hscale(si: &BasisShell, sj: &BasisShell, params: &Gfn1Parameters) -> Result<f64> {
    let kdiff = params.global("kdiff", 2.85);
    if si.is_valence && sj.is_valence {
        let den = (pauling_en(si.z)? - pauling_en(sj.z)?).powi(2);
        let enscale = params.global("enscale", -0.7) * 0.01;
        Ok(params.pair_scaling(si.z, sj.z)
            * kshell_pair(si.angular, sj.angular, params)
            * (1.0 + enscale * den))
    } else if si.is_valence {
        Ok(0.5 * (kshell_pair(si.angular, si.angular, params) + kdiff))
    } else if sj.is_valence {
        Ok(0.5 * (kshell_pair(sj.angular, sj.angular, params) + kdiff))
    } else {
        Ok(kdiff)
    }
}

pub fn kshell_pair(li: AngularMomentum, lj: AngularMomentum, params: &Gfn1Parameters) -> f64 {
    if (li == AngularMomentum::S && lj == AngularMomentum::P)
        || (li == AngularMomentum::P && lj == AngularMomentum::S)
    {
        params.global("ksp", 2.08)
    } else {
        0.5 * (params.k_shell(li) + params.k_shell(lj))
    }
}

pub fn default_coordination_numbers(system: &PeriodicSystem) -> Result<Vec<f64>> {
    coordination_numbers(system)
}
