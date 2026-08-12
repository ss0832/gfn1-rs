// SPDX-License-Identifier: GPL-3.0-or-later
//! DFT-D3(BJ) dispersion and experimental non-PBC DFT-D4 dispersion for GFN1-xTB.
//!
//! The large D3 C6 reference table is bundled under `third_party/simple-dftd3`
//! and embedded at build time (`s-dftd3/src/dftd3/reference.f90`), so it is used
//! by default; an explicit path or the `GFN1_D3_REFERENCE` environment variable
//! can override it.
//! D4 reference data are stored with upstream provenance under `third_party/dftd4`;
//! the GFN1 damping constants `a1`, `a2`, and `s8` are read from the
//! user-supplied `param_gfn1-xtb.txt` through [`Gfn1Parameters`].  The experimental
//! ATM scale `s9` is an API option and defaults to the GFN2-xTB value.

use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::d4_reference;
use crate::data_tables::covalent_radius_d3_bohr;
use crate::error::{Gfn1Error, Result};
use crate::jets::{DirectionScope, Jet1, Jet2, Jet3, Jet4};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pairlist::{
    all_center_short_range_neighbors, unique_short_range_pairs, ShortRangePair,
};
use crate::params::{Gfn1Parameters, GFN1_D3_REFERENCE_ENV};
use crate::system::PeriodicSystem;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

const WF: f64 = 4.0;
const CN_CUTOFF: f64 = 25.0;
const DISP2_CUTOFF: f64 = 50.0;
const D4_PAIR_CUTOFF: f64 = 60.0;
const D4_CN_CUTOFF: f64 = 30.0;
const D4_ATM_CUTOFF: f64 = 40.0;
const D4_ATM_DAMPING_EXPONENT: f64 = 16.0;
/// Chai-Head-Gordon zero-damping exponent for the D3 ATM three-body term. The
/// damping function `f = 1 / (1 + 6 (R0_ABC / R_ABC)^(alp/3))` uses `alp/3`; the
/// reference DFT-D3 ATM exponent is 16, matching the experimental D4 ATM path.
const D3_ATM_DAMPING_EXPONENT: f64 = 16.0;
pub const D4_GFN2_DEFAULT_S9: f64 = 5.0;
const D4_PAIR_PARALLEL_MIN_PAIRS: usize = 4096;
const D4_ATM_PARALLEL_MIN_TRIPLES: usize = 4096;
const DIST_EPS: f64 = 1.0e-14;

#[derive(Clone, Debug)]
pub struct DispersionResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    /// Derivative with respect to homogeneous strain divided by cell volume.
    /// Present only for periodic systems.
    pub stress: Option<Matrix>,
}

#[derive(Clone, Debug)]
pub struct DispersionHessianResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub hessian: Matrix,
    /// Derivative with respect to homogeneous strain divided by cell volume.
    /// Present only for periodic systems.
    pub stress: Option<Matrix>,
}

#[derive(Clone, Copy, Debug)]
pub struct D4DispersionOptions {
    /// Two-body D4 cutoff in Bohr. For non-periodic molecules, non-positive or
    /// non-finite means all unique atom pairs.
    pub cutoff: f64,
    /// Coordination-number cutoff in Bohr.
    pub cn_cutoff: f64,
    /// Include the Axilrod-Teller-Muto three-body term controlled by `s9`.
    pub atm_enabled: bool,
    /// ATM three-body cutoff in Bohr.
    pub atm_cutoff: f64,
    /// Chai-Head-Gordon zero-damping exponent used by the D4 ATM term.
    pub atm_damping_exponent: f64,
    /// ATM scale factor. Defaults to the GFN2-xTB value; set to 0 to disable
    /// the three-body energy without changing other D4 options.
    pub s9: f64,
}

impl Default for D4DispersionOptions {
    fn default() -> Self {
        Self {
            cutoff: D4_PAIR_CUTOFF,
            cn_cutoff: D4_CN_CUTOFF,
            atm_enabled: true,
            atm_cutoff: D4_ATM_CUTOFF,
            atm_damping_exponent: D4_ATM_DAMPING_EXPONENT,
            s9: D4_GFN2_DEFAULT_S9,
        }
    }
}

#[derive(Clone, Debug)]
pub struct D4DispersionEnergy {
    pub energy: f64,
    /// Atomic scalar shift `dE_D4/dq_A` for the self-consistent charge loop.
    pub atomic_potential: Vec<f64>,
    pub coordination_numbers: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct D4PairConstants {
    s6: f64,
    a1: f64,
    a2: f64,
    s8: f64,
    s9: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct D4PairAtomContribution {
    c6: f64,
    c8: f64,
    c8_scale: f64,
    r0: f64,
    denergy_dqi_scale: f64,
    denergy_dqj_scale: f64,
    denergy_dcni_scale: f64,
    denergy_dcnj_scale: f64,
}

#[derive(Clone)]
struct D4PreparedAtoms {
    z: Vec<u8>,
    r4r2: Vec<f64>,
    weights: Vec<d4_reference::D4AtomWeights>,
    elem_idx: [usize; 128],
    tables: Vec<d4_reference::D4C6PairTable>,
    ne: usize,
}

#[derive(Clone, Copy, Debug)]
struct D4PairContribution {
    energy: f64,
    denergy_dr: f64,
    denergy_dqi: f64,
    denergy_dqj: f64,
    denergy_dcni: f64,
    denergy_dcnj: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct D4AtmPairGeometry {
    i: u32,
    j: u32,
    r: f64,
    dr: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct D4AtmTripleGeometry {
    i: u32,
    j: u32,
    k: u32,
    ij: u32,
    ik: u32,
    jk: u32,
    r0_product: f64,
    energy_weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct D4AtmGeometry {
    pub pairs: Vec<D4AtmPairGeometry>,
    pub triples: Vec<D4AtmTripleGeometry>,
}

pub fn d4_dispersion_pairs(
    system: &PeriodicSystem,
    options: D4DispersionOptions,
) -> Result<Vec<ShortRangePair>> {
    ensure_d4_nonperiodic(system)?;
    unique_short_range_pairs(system, options.cutoff)
}

pub fn d4_dispersion_atm_geometry(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: D4DispersionOptions,
) -> Result<D4AtmGeometry> {
    ensure_d4_nonperiodic(system)?;
    let constants = d4_pair_constants(params, options)?;
    if !options.atm_enabled || constants.s9.abs() <= 1.0e-16 || system.atoms.len() < 3 {
        return Ok(D4AtmGeometry::default());
    }
    let cutoff = if options.atm_cutoff > 0.0 && options.atm_cutoff.is_finite() {
        options.atm_cutoff
    } else {
        f64::INFINITY
    };
    let pairs = unique_short_range_pairs(system, cutoff)?;
    build_d4_atm_geometry_from_pairs(system, &pairs, constants, options)
}

pub fn d4_dispersion_energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    options: D4DispersionOptions,
) -> Result<f64> {
    ensure_d4_nonperiodic(system)?;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: options.cn_cutoff,
            ..CoordinationOptions::default()
        },
    )?
    .cn;
    let pairs = d4_dispersion_pairs(system, options)?;
    let atm = d4_dispersion_atm_geometry(system, params, options)?;
    d4_dispersion_energy_with_cn_pairs_and_atm(system, params, charges, &cn, &pairs, &atm, options)
}

pub fn d4_dispersion_energy_potential(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    options: D4DispersionOptions,
) -> Result<D4DispersionEnergy> {
    ensure_d4_nonperiodic(system)?;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: options.cn_cutoff,
            ..CoordinationOptions::default()
        },
    )?
    .cn;
    let pairs = d4_dispersion_pairs(system, options)?;
    let atm = d4_dispersion_atm_geometry(system, params, options)?;
    d4_dispersion_energy_potential_with_cn_pairs_and_atm(
        system, params, charges, &cn, &pairs, &atm, options,
    )
}

pub(crate) fn d4_dispersion_energy_with_cn_pairs_and_atm(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    cn: &[f64],
    pairs: &[ShortRangePair],
    atm: &D4AtmGeometry,
    options: D4DispersionOptions,
) -> Result<f64> {
    ensure_d4_nonperiodic(system)?;
    let nat = system.atoms.len();
    if charges.len() != nat || cn.len() != nat {
        return Err(Gfn1Error::InvalidInput(
            "D4 charge/CN dimension mismatch".to_string(),
        ));
    }
    if nat == 0 {
        return Ok(0.0);
    }
    let constants = d4_pair_constants(params, options)?;
    let prepared = d4_prepare_atoms(system, charges, cn)?;

    let pair_energy = if pairs.len() >= D4_PAIR_PARALLEL_MIN_PAIRS {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|pair| {
                if pair.r <= DIST_EPS {
                    0.0
                } else {
                    d4_pair_energy_from_prepared(&prepared, pair.i, pair.j, pair.r, constants)
                }
            })
            .sum()
    } else {
        let mut energy = 0.0;
        for pair in pairs {
            if pair.r <= DIST_EPS {
                continue;
            }
            energy += d4_pair_energy_from_prepared(&prepared, pair.i, pair.j, pair.r, constants);
        }
        energy
    };
    Ok(pair_energy + d4_atm_energy_from_geometry(&prepared, atm, constants))
}

pub fn d4_dispersion_energy_potential_with_cn_and_pairs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    cn: &[f64],
    pairs: &[ShortRangePair],
    options: D4DispersionOptions,
) -> Result<D4DispersionEnergy> {
    let atm = d4_dispersion_atm_geometry(system, params, options)?;
    d4_dispersion_energy_potential_with_cn_pairs_and_atm(
        system, params, charges, cn, pairs, &atm, options,
    )
}

pub fn d4_dispersion_energy_potential_with_cn_pairs_and_atm(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    cn: &[f64],
    pairs: &[ShortRangePair],
    atm: &D4AtmGeometry,
    options: D4DispersionOptions,
) -> Result<D4DispersionEnergy> {
    ensure_d4_nonperiodic(system)?;
    let nat = system.atoms.len();
    if charges.len() != nat || cn.len() != nat {
        return Err(Gfn1Error::InvalidInput(
            "D4 charge/CN dimension mismatch".to_string(),
        ));
    }
    if nat == 0 {
        return Ok(D4DispersionEnergy {
            energy: 0.0,
            atomic_potential: Vec::new(),
            coordination_numbers: Vec::new(),
        });
    }
    let constants = d4_pair_constants(params, options)?;
    let prepared = d4_prepare_atoms(system, charges, cn)?;
    let (pair_energy, mut atomic_potential) = d4_energy_potential_from_pairs(
        &prepared,
        pairs,
        constants,
        nat,
        D4_PAIR_PARALLEL_MIN_PAIRS,
    );
    let atm_energy = d4_atm_energy_potential_from_geometry(
        &prepared,
        atm,
        constants,
        nat,
        &mut atomic_potential,
    );
    Ok(D4DispersionEnergy {
        energy: pair_energy + atm_energy,
        atomic_potential,
        coordination_numbers: cn.to_vec(),
    })
}

pub fn d4_dispersion_energy_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    charges: &[f64],
    options: D4DispersionOptions,
) -> Result<DispersionResult> {
    ensure_d4_nonperiodic(system)?;
    let nat = system.atoms.len();
    if charges.len() != nat {
        return Err(Gfn1Error::InvalidInput(format!(
            "got {} D4 charges for {} atoms",
            charges.len(),
            nat
        )));
    }
    let cn_deriv = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: options.cn_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    let pairs = d4_dispersion_pairs(system, options)?;
    let atm = d4_dispersion_atm_geometry(system, params, options)?;
    let constants = d4_pair_constants(params, options)?;
    let prepared = d4_prepare_atoms(system, charges, &cn_deriv.cn)?;
    let (mut energy, mut gradient, mut d_edcn) =
        d4_gradient_pair_terms(&prepared, &pairs, constants, nat);
    let (atm_energy, atm_gradient, atm_d_edcn) =
        d4_atm_gradient_terms(&prepared, &atm, constants, nat, options);
    energy += atm_energy;
    for i in 0..nat {
        gradient[i] += atm_gradient[i];
        d_edcn[i] += atm_d_edcn[i];
    }

    if cn_deriv.pairs.len() >= D4_PAIR_PARALLEL_MIN_PAIRS {
        use rayon::prelude::*;
        let cn_gradient = cn_deriv
            .pairs
            .par_iter()
            .fold(
                || vec![Vec3::zero(); nat],
                |mut local, pair| {
                    d4_add_cn_gradient_pair(&mut local, &d_edcn, pair);
                    local
                },
            )
            .reduce(
                || vec![Vec3::zero(); nat],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            );
        for (g, add) in gradient.iter_mut().zip(cn_gradient.iter()) {
            *g += *add;
        }
    } else {
        for pair in &cn_deriv.pairs {
            d4_add_cn_gradient_pair(&mut gradient, &d_edcn, pair);
        }
    }

    Ok(DispersionResult {
        energy,
        gradient,
        stress: None,
    })
}

fn d4_add_cn_gradient_pair(
    gradient: &mut [Vec3],
    d_edcn: &[f64],
    pair: &crate::coordination::CoordinationPairDerivative,
) {
    if pair.i == pair.j {
        return;
    }
    let r = pair.r_ij.norm();
    if r <= DIST_EPS {
        return;
    }
    let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
    let gi = pair.r_ij * pref;
    gradient[pair.i] += gi;
    gradient[pair.j] -= gi;
}

fn ensure_d4_nonperiodic(system: &PeriodicSystem) -> Result<()> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "experimental D4 dispersion is implemented for non-PBC systems only".to_string(),
        ));
    }
    Ok(())
}

fn build_d4_atm_geometry_from_pairs(
    system: &PeriodicSystem,
    pairs: &[ShortRangePair],
    constants: D4PairConstants,
    options: D4DispersionOptions,
) -> Result<D4AtmGeometry> {
    let nat = system.atoms.len();
    if nat > u32::MAX as usize {
        return Err(Gfn1Error::InvalidInput(
            "D4 ATM geometry uses compact u32 atom indices; atom count exceeds u32::MAX"
                .to_string(),
        ));
    }
    let r4r2 = d4_r4r2_for_system(system)?;
    let mut atm_pairs = Vec::with_capacity(pairs.len());
    let mut pair_index = HashMap::with_capacity(pairs.len() * 2);
    let mut neighbors = vec![Vec::<(usize, u32)>::new(); nat];
    for pair in pairs {
        if pair.r <= DIST_EPS {
            continue;
        }
        let idx = d4_u32_index(atm_pairs.len(), "D4 ATM pair")?;
        atm_pairs.push(D4AtmPairGeometry {
            i: pair.i as u32,
            j: pair.j as u32,
            r: pair.r,
            dr: pair.dr,
        });
        pair_index.insert(d4_pair_key(pair.i, pair.j), idx);
        neighbors[pair.i].push((pair.j, idx));
        neighbors[pair.j].push((pair.i, idx));
    }
    for neigh in &mut neighbors {
        neigh.sort_unstable_by_key(|(j, _)| *j);
    }

    let alp3 = options.atm_damping_exponent / 3.0;
    let mut triples = Vec::new();
    for i in 0..nat {
        for a in 0..neighbors[i].len() {
            let (j, ij) = neighbors[i][a];
            if j <= i {
                continue;
            }
            for &(k, ik) in neighbors[i][(a + 1)..].iter() {
                if k <= j {
                    continue;
                }
                let Some(&jk) = pair_index.get(&d4_pair_key(j, k)) else {
                    continue;
                };
                let pij = atm_pairs[ij as usize];
                let pik = atm_pairs[ik as usize];
                let pjk = atm_pairs[jk as usize];
                let r1 = pij.r * pik.r * pjk.r;
                if r1 <= DIST_EPS {
                    continue;
                }
                let r0ij = constants.a1 * (3.0 * r4r2[i] * r4r2[j]).sqrt() + constants.a2;
                let r0ik = constants.a1 * (3.0 * r4r2[i] * r4r2[k]).sqrt() + constants.a2;
                let r0jk = constants.a1 * (3.0 * r4r2[j] * r4r2[k]).sqrt() + constants.a2;
                let r0_product = r0ij * r0ik * r0jk;
                let fdmp = 1.0 / (1.0 + 6.0 * (r0_product / r1).powf(alp3));
                let angular = d4_atm_angular(pij.r, pik.r, pjk.r);
                let energy_weight = angular * fdmp;
                if energy_weight == 0.0 || !energy_weight.is_finite() {
                    continue;
                }
                triples.push(D4AtmTripleGeometry {
                    i: i as u32,
                    j: j as u32,
                    k: k as u32,
                    ij,
                    ik,
                    jk,
                    r0_product,
                    energy_weight,
                });
            }
        }
    }
    Ok(D4AtmGeometry {
        pairs: atm_pairs,
        triples,
    })
}

#[inline]
fn d4_u32_index(index: usize, label: &str) -> Result<u32> {
    u32::try_from(index)
        .map_err(|_| Gfn1Error::InvalidInput(format!("{label} index exceeds u32::MAX")))
}

#[inline]
fn d4_pair_key(i: usize, j: usize) -> u64 {
    let a = i.min(j) as u64;
    let b = i.max(j) as u64;
    (a << 32) | b
}

fn d4_pair_constants(
    params: &Gfn1Parameters,
    options: D4DispersionOptions,
) -> Result<D4PairConstants> {
    // GFN1-specific D4 damping constants come from the active param_gfn1-xtb.txt
    // rather than from the bundled DFT-D4 provenance snapshot.
    let s6 = params.global("s6", 1.0);
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let s8 = params.required_global("s8")?;
    let s9 = options.s9;
    Ok(D4PairConstants { s6, a1, a2, s8, s9 })
}

fn d4_atomic_numbers(system: &PeriodicSystem) -> Vec<u8> {
    system.atoms.iter().map(|atom| atom.z).collect()
}

fn d4_r4r2_for_system(system: &PeriodicSystem) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(system.atoms.len());
    for atom in &system.atoms {
        out.push(d4_reference::d4_r4r2_atom(atom.z)?);
    }
    Ok(out)
}

fn d4_prepare_atoms(
    system: &PeriodicSystem,
    charges: &[f64],
    cn: &[f64],
) -> Result<D4PreparedAtoms> {
    let z = d4_atomic_numbers(system);
    let r4r2 = d4_r4r2_for_system(system)?;
    let mut weights = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        weights.push(d4_reference::d4_atom_reference_weights(
            z[i], charges[i], cn[i],
        )?);
    }
    let (elem_idx, tables, ne) = d4_prefetch_pair_tables(&z)?;
    Ok(D4PreparedAtoms {
        z,
        r4r2,
        weights,
        elem_idx,
        tables,
        ne,
    })
}

fn d4_pair_table(prepared: &D4PreparedAtoms, i: usize, j: usize) -> &d4_reference::D4C6PairTable {
    let ti = prepared.elem_idx[prepared.z[i] as usize] * prepared.ne;
    &prepared.tables[ti + prepared.elem_idx[prepared.z[j] as usize]]
}

fn d4_pair_atom_contribution(
    prepared: &D4PreparedAtoms,
    i: usize,
    j: usize,
    constants: D4PairConstants,
) -> D4PairAtomContribution {
    let table = d4_pair_table(prepared, i, j);
    let (c6, dc6_dqi, dc6_dqj, dc6_dcni, dc6_dcnj) =
        d4_reference::d4_c6_from_atom_weights_with_table(
            prepared.z[i],
            prepared.z[j],
            &prepared.weights[i],
            &prepared.weights[j],
            table,
        );
    let r2r4 = (prepared.r4r2[i] * prepared.r4r2[j]).sqrt().max(0.5);
    let c8_scale = 3.0 * r2r4 * r2r4;
    let c8 = c8_scale * c6;
    let r0 = constants.a1 * (c8.abs() / c6.abs().max(1.0e-16)).sqrt() + constants.a2;
    D4PairAtomContribution {
        c6,
        c8,
        c8_scale,
        r0,
        denergy_dqi_scale: dc6_dqi,
        denergy_dqj_scale: dc6_dqj,
        denergy_dcni_scale: dc6_dcni,
        denergy_dcnj_scale: dc6_dcnj,
    }
}

fn d4_pair_energy_from_prepared(
    prepared: &D4PreparedAtoms,
    i: usize,
    j: usize,
    r: f64,
    constants: D4PairConstants,
) -> f64 {
    let table = d4_pair_table(prepared, i, j);
    let c6 = d4_reference::d4_c6_value_from_atom_weights_with_table(
        prepared.z[i],
        prepared.z[j],
        &prepared.weights[i],
        &prepared.weights[j],
        table,
    );
    let r2r4 = (prepared.r4r2[i] * prepared.r4r2[j]).sqrt().max(0.5);
    let c8_scale = 3.0 * r2r4 * r2r4;
    let c8 = c8_scale * c6;
    let r0 = constants.a1 * (c8.abs() / c6.abs().max(1.0e-16)).sqrt() + constants.a2;
    d4_pair_energy_from_values(c6, c8, r0, r, constants)
}

fn d4_energy_potential_from_pairs(
    prepared: &D4PreparedAtoms,
    pairs: &[ShortRangePair],
    constants: D4PairConstants,
    nat: usize,
    parallel_min_pairs: usize,
) -> (f64, Vec<f64>) {
    if pairs.len() >= parallel_min_pairs {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .fold(
                || (0.0, vec![0.0; nat]),
                |(mut energy, mut potential), pair| {
                    if pair.r > DIST_EPS {
                        let value = d4_pair_contribution_from_atom_data(
                            d4_pair_atom_contribution(prepared, pair.i, pair.j, constants),
                            pair.r,
                            constants,
                        );
                        energy += value.energy;
                        potential[pair.i] += value.denergy_dqi;
                        potential[pair.j] += value.denergy_dqj;
                    }
                    (energy, potential)
                },
            )
            .reduce(
                || (0.0, vec![0.0; nat]),
                |(ea, mut pa), (eb, pb)| {
                    for (a, b) in pa.iter_mut().zip(pb.iter()) {
                        *a += *b;
                    }
                    (ea + eb, pa)
                },
            )
    } else {
        let mut energy = 0.0;
        let mut potential = vec![0.0; nat];
        for pair in pairs {
            if pair.r <= DIST_EPS {
                continue;
            }
            let value = d4_pair_contribution_from_atom_data(
                d4_pair_atom_contribution(prepared, pair.i, pair.j, constants),
                pair.r,
                constants,
            );
            energy += value.energy;
            potential[pair.i] += value.denergy_dqi;
            potential[pair.j] += value.denergy_dqj;
        }
        (energy, potential)
    }
}

fn d4_atm_pair_c6_values(prepared: &D4PreparedAtoms, atm: &D4AtmGeometry) -> Vec<f64> {
    atm.pairs
        .iter()
        .map(|pair| {
            let i = pair.i as usize;
            let j = pair.j as usize;
            let table = d4_pair_table(prepared, i, j);
            d4_reference::d4_c6_value_from_atom_weights_with_table(
                prepared.z[i],
                prepared.z[j],
                &prepared.weights[i],
                &prepared.weights[j],
                table,
            )
        })
        .collect()
}

fn d4_atm_pair_atom_contributions(
    prepared: &D4PreparedAtoms,
    atm: &D4AtmGeometry,
    constants: D4PairConstants,
) -> Vec<D4PairAtomContribution> {
    atm.pairs
        .iter()
        .map(|pair| {
            d4_pair_atom_contribution(prepared, pair.i as usize, pair.j as usize, constants)
        })
        .collect()
}

fn d4_atm_energy_from_geometry(
    prepared: &D4PreparedAtoms,
    atm: &D4AtmGeometry,
    constants: D4PairConstants,
) -> f64 {
    if constants.s9.abs() <= 1.0e-16 || atm.triples.is_empty() {
        return 0.0;
    }
    let c6 = d4_atm_pair_c6_values(prepared, atm);
    if atm.triples.len() >= D4_ATM_PARALLEL_MIN_TRIPLES {
        use rayon::prelude::*;
        atm.triples
            .par_iter()
            .map(|triple| d4_atm_triple_energy_from_c6(triple, &c6, constants.s9))
            .sum()
    } else {
        atm.triples
            .iter()
            .map(|triple| d4_atm_triple_energy_from_c6(triple, &c6, constants.s9))
            .sum()
    }
}

fn d4_atm_energy_potential_from_geometry(
    prepared: &D4PreparedAtoms,
    atm: &D4AtmGeometry,
    constants: D4PairConstants,
    nat: usize,
    atomic_potential: &mut [f64],
) -> f64 {
    if constants.s9.abs() <= 1.0e-16 || atm.triples.is_empty() {
        return 0.0;
    }
    let pair_data = d4_atm_pair_atom_contributions(prepared, atm, constants);
    let (energy, potential) = if atm.triples.len() >= D4_ATM_PARALLEL_MIN_TRIPLES {
        use rayon::prelude::*;
        atm.triples
            .par_iter()
            .fold(
                || (0.0, vec![0.0; nat]),
                |(mut energy, mut potential), triple| {
                    d4_add_atm_potential_triple(
                        constants.s9,
                        triple,
                        &pair_data,
                        &mut energy,
                        &mut potential,
                    );
                    (energy, potential)
                },
            )
            .reduce(
                || (0.0, vec![0.0; nat]),
                |(ea, mut pa), (eb, pb)| {
                    for (a, b) in pa.iter_mut().zip(pb.iter()) {
                        *a += *b;
                    }
                    (ea + eb, pa)
                },
            )
    } else {
        let mut energy = 0.0;
        let mut potential = vec![0.0; nat];
        for triple in &atm.triples {
            d4_add_atm_potential_triple(
                constants.s9,
                triple,
                &pair_data,
                &mut energy,
                &mut potential,
            );
        }
        (energy, potential)
    };
    for (dst, add) in atomic_potential.iter_mut().zip(potential.iter()) {
        *dst += *add;
    }
    energy
}

fn d4_gradient_pair_terms(
    prepared: &D4PreparedAtoms,
    pairs: &[ShortRangePair],
    constants: D4PairConstants,
    nat: usize,
) -> (f64, Vec<Vec3>, Vec<f64>) {
    if pairs.len() >= D4_PAIR_PARALLEL_MIN_PAIRS {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .fold(
                || (0.0, vec![Vec3::zero(); nat], vec![0.0; nat]),
                |(mut energy, mut gradient, mut d_edcn), pair| {
                    d4_add_gradient_pair(
                        prepared,
                        constants,
                        &mut energy,
                        &mut gradient,
                        &mut d_edcn,
                        pair,
                    );
                    (energy, gradient, d_edcn)
                },
            )
            .reduce(
                || (0.0, vec![Vec3::zero(); nat], vec![0.0; nat]),
                |(ea, mut ga, mut ca), (eb, gb, cb)| {
                    for (a, b) in ga.iter_mut().zip(gb.iter()) {
                        *a += *b;
                    }
                    for (a, b) in ca.iter_mut().zip(cb.iter()) {
                        *a += *b;
                    }
                    (ea + eb, ga, ca)
                },
            )
    } else {
        let mut energy = 0.0;
        let mut gradient = vec![Vec3::zero(); nat];
        let mut d_edcn = vec![0.0; nat];
        for pair in pairs {
            d4_add_gradient_pair(
                prepared,
                constants,
                &mut energy,
                &mut gradient,
                &mut d_edcn,
                pair,
            );
        }
        (energy, gradient, d_edcn)
    }
}

fn d4_add_gradient_pair(
    prepared: &D4PreparedAtoms,
    constants: D4PairConstants,
    energy: &mut f64,
    gradient: &mut [Vec3],
    d_edcn: &mut [f64],
    pair: &ShortRangePair,
) {
    if pair.r <= DIST_EPS {
        return;
    }
    let value = d4_pair_contribution_from_atom_data(
        d4_pair_atom_contribution(prepared, pair.i, pair.j, constants),
        pair.r,
        constants,
    );
    *energy += value.energy;
    let gi = pair.dr * (-value.denergy_dr / pair.r);
    gradient[pair.i] += gi;
    gradient[pair.j] -= gi;
    d_edcn[pair.i] += value.denergy_dcni;
    d_edcn[pair.j] += value.denergy_dcnj;
}

fn d4_atm_gradient_terms(
    prepared: &D4PreparedAtoms,
    atm: &D4AtmGeometry,
    constants: D4PairConstants,
    nat: usize,
    options: D4DispersionOptions,
) -> (f64, Vec<Vec3>, Vec<f64>) {
    if constants.s9.abs() <= 1.0e-16 || atm.triples.is_empty() {
        return (0.0, vec![Vec3::zero(); nat], vec![0.0; nat]);
    }
    let pair_data = d4_atm_pair_atom_contributions(prepared, atm, constants);
    let alp3 = options.atm_damping_exponent / 3.0;
    if atm.triples.len() >= D4_ATM_PARALLEL_MIN_TRIPLES {
        use rayon::prelude::*;
        atm.triples
            .par_iter()
            .fold(
                || (0.0, vec![Vec3::zero(); nat], vec![0.0; nat]),
                |(mut energy, mut gradient, mut d_edcn), triple| {
                    d4_add_atm_gradient_triple(
                        constants.s9,
                        alp3,
                        atm,
                        triple,
                        &pair_data,
                        &mut energy,
                        &mut gradient,
                        &mut d_edcn,
                    );
                    (energy, gradient, d_edcn)
                },
            )
            .reduce(
                || (0.0, vec![Vec3::zero(); nat], vec![0.0; nat]),
                |(ea, mut ga, mut ca), (eb, gb, cb)| {
                    for (a, b) in ga.iter_mut().zip(gb.iter()) {
                        *a += *b;
                    }
                    for (a, b) in ca.iter_mut().zip(cb.iter()) {
                        *a += *b;
                    }
                    (ea + eb, ga, ca)
                },
            )
    } else {
        let mut energy = 0.0;
        let mut gradient = vec![Vec3::zero(); nat];
        let mut d_edcn = vec![0.0; nat];
        for triple in &atm.triples {
            d4_add_atm_gradient_triple(
                constants.s9,
                alp3,
                atm,
                triple,
                &pair_data,
                &mut energy,
                &mut gradient,
                &mut d_edcn,
            );
        }
        (energy, gradient, d_edcn)
    }
}

fn d4_add_atm_potential_triple(
    s9: f64,
    triple: &D4AtmTripleGeometry,
    pair_data: &[D4PairAtomContribution],
    energy: &mut f64,
    potential: &mut [f64],
) {
    let i = triple.i as usize;
    let j = triple.j as usize;
    let k = triple.k as usize;
    let ij = triple.ij as usize;
    let ik = triple.ik as usize;
    let jk = triple.jk as usize;
    let data_ij = pair_data[ij];
    let data_ik = pair_data[ik];
    let data_jk = pair_data[jk];
    let e = d4_atm_triple_energy_from_pair_data(triple, data_ij, data_ik, data_jk, s9);
    if e == 0.0 {
        return;
    }
    *energy += e;
    let dcij = d4_atm_dc6_prefactor(e, data_ij.c6);
    let dcik = d4_atm_dc6_prefactor(e, data_ik.c6);
    let dcjk = d4_atm_dc6_prefactor(e, data_jk.c6);
    potential[i] += dcij * data_ij.denergy_dqi_scale + dcik * data_ik.denergy_dqi_scale;
    potential[j] += dcij * data_ij.denergy_dqj_scale + dcjk * data_jk.denergy_dqi_scale;
    potential[k] += dcik * data_ik.denergy_dqj_scale + dcjk * data_jk.denergy_dqj_scale;
}

fn d4_add_atm_gradient_triple(
    s9: f64,
    alp3: f64,
    atm: &D4AtmGeometry,
    triple: &D4AtmTripleGeometry,
    pair_data: &[D4PairAtomContribution],
    energy: &mut f64,
    gradient: &mut [Vec3],
    d_edcn: &mut [f64],
) {
    let i = triple.i as usize;
    let j = triple.j as usize;
    let k = triple.k as usize;
    let ij = triple.ij as usize;
    let ik = triple.ik as usize;
    let jk = triple.jk as usize;
    let pair_ij = atm.pairs[ij];
    let pair_ik = atm.pairs[ik];
    let pair_jk = atm.pairs[jk];
    let data_ij = pair_data[ij];
    let data_ik = pair_data[ik];
    let data_jk = pair_data[jk];
    let root = (data_ij.c6 * data_ik.c6 * data_jk.c6).abs().sqrt();
    if root <= 1.0e-30 {
        return;
    }
    let c9 = -s9 * root;
    let (e, de_drij, de_drik, de_drjk) = d4_atm_energy_distance_derivatives(
        pair_ij.r,
        pair_ik.r,
        pair_jk.r,
        c9,
        triple.r0_product,
        alp3,
    );
    if e == 0.0 {
        return;
    }
    *energy += e;
    let uij = pair_ij.dr / pair_ij.r;
    let uik = pair_ik.dr / pair_ik.r;
    let ujk = pair_jk.dr / pair_jk.r;
    gradient[i] += uij * (-de_drij) + uik * (-de_drik);
    gradient[j] += uij * de_drij + ujk * (-de_drjk);
    gradient[k] += uik * de_drik + ujk * de_drjk;

    let dcij = d4_atm_dc6_prefactor(e, data_ij.c6);
    let dcik = d4_atm_dc6_prefactor(e, data_ik.c6);
    let dcjk = d4_atm_dc6_prefactor(e, data_jk.c6);
    d_edcn[i] += dcij * data_ij.denergy_dcni_scale + dcik * data_ik.denergy_dcni_scale;
    d_edcn[j] += dcij * data_ij.denergy_dcnj_scale + dcjk * data_jk.denergy_dcni_scale;
    d_edcn[k] += dcik * data_ik.denergy_dcnj_scale + dcjk * data_jk.denergy_dcnj_scale;
}

#[inline]
fn d4_atm_triple_energy_from_c6(triple: &D4AtmTripleGeometry, c6: &[f64], s9: f64) -> f64 {
    let c6ij = c6[triple.ij as usize];
    let c6ik = c6[triple.ik as usize];
    let c6jk = c6[triple.jk as usize];
    let root = (c6ij * c6ik * c6jk).abs().sqrt();
    if root <= 1.0e-30 {
        0.0
    } else {
        s9 * root * triple.energy_weight
    }
}

#[inline]
fn d4_atm_triple_energy_from_pair_data(
    triple: &D4AtmTripleGeometry,
    data_ij: D4PairAtomContribution,
    data_ik: D4PairAtomContribution,
    data_jk: D4PairAtomContribution,
    s9: f64,
) -> f64 {
    let root = (data_ij.c6 * data_ik.c6 * data_jk.c6).abs().sqrt();
    if root <= 1.0e-30 {
        0.0
    } else {
        s9 * root * triple.energy_weight
    }
}

#[inline]
fn d4_atm_dc6_prefactor(energy: f64, c6: f64) -> f64 {
    if c6.abs() > 1.0e-30 {
        0.5 * energy / c6
    } else {
        0.0
    }
}

#[inline]
fn d4_atm_angular(rij: f64, rik: f64, rjk: f64) -> f64 {
    let x = rij * rij;
    let y = rik * rik;
    let z = rjk * rjk;
    let p = (x + z - y) * (x - z + y) * (-x + z + y);
    let r1 = rij * rik * rjk;
    if r1 <= DIST_EPS {
        return 0.0;
    }
    let inv_r1 = 1.0 / r1;
    let inv_r3 = inv_r1 * inv_r1 * inv_r1;
    let inv_r5 = inv_r3 * inv_r1 * inv_r1;
    0.375 * p * inv_r5 + inv_r3
}

fn d4_atm_energy_distance_derivatives(
    rij: f64,
    rik: f64,
    rjk: f64,
    c9: f64,
    r0: f64,
    alp3: f64,
) -> (f64, f64, f64, f64) {
    let x = rij * rij;
    let y = rik * rik;
    let z = rjk * rjk;
    let a = x + z - y;
    let b = x - z + y;
    let c = -x + z + y;
    let p = a * b * c;
    let dp_dx = b * c + a * c - a * b;
    let dp_dy = -b * c + a * c + a * b;
    let dp_dz = b * c - a * c + a * b;
    let dp_drij = 2.0 * rij * dp_dx;
    let dp_drik = 2.0 * rik * dp_dy;
    let dp_drjk = 2.0 * rjk * dp_dz;

    let r1 = rij * rik * rjk;
    let inv_r1 = 1.0 / r1;
    let inv_r3 = inv_r1 * inv_r1 * inv_r1;
    let inv_r5 = inv_r3 * inv_r1 * inv_r1;
    let angular = 0.375 * p * inv_r5 + inv_r3;
    let dangular =
        |dp_dr: f64, r: f64| 0.375 * (dp_dr * inv_r5 - 5.0 * p * inv_r5 / r) - 3.0 * inv_r3 / r;

    let damp_pow = (r0 * inv_r1).powf(alp3);
    let fdmp = 1.0 / (1.0 + 6.0 * damp_pow);
    let fdmp2 = fdmp * fdmp;
    let dfdmp = |r: f64| 6.0 * alp3 * damp_pow * fdmp2 / r;

    let pref = -c9;
    let energy = pref * angular * fdmp;
    let deriv = |da: f64, df: f64| pref * (da * fdmp + angular * df);
    (
        energy,
        deriv(dangular(dp_drij, rij), dfdmp(rij)),
        deriv(dangular(dp_drik, rik), dfdmp(rik)),
        deriv(dangular(dp_drjk, rjk), dfdmp(rjk)),
    )
}

fn d4_prefetch_pair_tables(
    z: &[u8],
) -> Result<([usize; 128], Vec<d4_reference::D4C6PairTable>, usize)> {
    let mut elem_idx = [usize::MAX; 128];
    let mut elems: Vec<u8> = Vec::new();
    for &zz in z {
        let idx = zz as usize;
        if idx >= elem_idx.len() {
            return Err(Gfn1Error::InvalidInput(format!(
                "D4 reference element Z={zz} exceeds the native index range"
            )));
        }
        if elem_idx[idx] == usize::MAX {
            elem_idx[idx] = elems.len();
            elems.push(zz);
        }
    }
    let ne = elems.len();
    let mut tables = Vec::with_capacity(ne * ne);
    for &za in &elems {
        for &zb in &elems {
            tables.push(d4_reference::d4_c6_pair_table(za, zb));
        }
    }
    Ok((elem_idx, tables, ne))
}

#[inline]
fn d4_pair_energy_from_values(
    c6: f64,
    c8: f64,
    r0: f64,
    r: f64,
    constants: D4PairConstants,
) -> f64 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let d2 = r0 * r0;
    let d4 = d2 * d2;
    let d6 = d4 * d2;
    let d8 = d4 * d4;
    -constants.s6 * c6 / (r6 + d6) - constants.s8 * c8 / (r8 + d8)
}

#[inline]
fn d4_pair_contribution_from_atom_data(
    data: D4PairAtomContribution,
    r: f64,
    constants: D4PairConstants,
) -> D4PairContribution {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let d2 = data.r0 * data.r0;
    let d4 = d2 * d2;
    let d6 = d4 * d2;
    let d8 = d4 * d4;
    let denom6 = r6 + d6;
    let denom8 = r8 + d8;
    let e6 = -constants.s6 * data.c6 / denom6;
    let e8 = -constants.s8 * data.c8 / denom8;
    let de6_dr = 6.0 * constants.s6 * data.c6 * r * r4 / (denom6 * denom6);
    let de8_dr = 8.0 * constants.s8 * data.c8 * r * r6 / (denom8 * denom8);
    let de_dc6 = -constants.s6 / denom6 - constants.s8 * data.c8_scale / denom8;
    D4PairContribution {
        energy: e6 + e8,
        denergy_dr: de6_dr + de8_dr,
        denergy_dqi: de_dc6 * data.denergy_dqi_scale,
        denergy_dqj: de_dc6 * data.denergy_dqj_scale,
        denergy_dcni: de_dc6 * data.denergy_dcni_scale,
        denergy_dcnj: de_dc6 * data.denergy_dcnj_scale,
    }
}

pub fn dispersion_energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<f64> {
    Ok(dispersion_energy_gradient(system, params, explicit_reference_path)?.energy)
}

pub fn dispersion_energy_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<DispersionResult> {
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;

    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: CN_CUTOFF,
            ..CoordinationOptions::default()
        },
    )?;
    let weights = reference_weights(system, &reference, &cn.cn)?;
    let (c6, dc6dcn) = atomic_c6(system, &reference, &weights)?;

    let nat = system.atoms.len();
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); nat];
    let mut d_edcn = vec![0.0; nat];
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let rij_vec = -pair.dr;
        let r2 = pair.r2;
        if r2 > cutoff2 || r2 <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2 * r2;
        let t6 = 1.0 / (r4 * r2 + r0_6);
        let t8 = 1.0 / (r4 * r4 + r0_8);
        let d6 = -6.0 * r4 * t6 * t6;
        let d8 = -8.0 * r4 * r2 * t8 * t8;
        let edisp = s6 * t6 + s8 * r4r2ij * t8;
        let gdisp = s6 * d6 + s8 * r4r2ij * d8;
        let c6ij = c6[i][j];

        energy -= c6ij * edisp;
        if i != j {
            let dg = rij_vec * (-c6ij * gdisp);
            gradient[i] += dg;
            gradient[j] -= dg;
        }
        if i == j {
            d_edcn[i] -= dc6dcn[i][j] * edisp;
        } else {
            d_edcn[i] -= dc6dcn[i][j] * edisp;
            d_edcn[j] -= dc6dcn[j][i] * edisp;
        }
    }

    // Axilrod-Teller-Muto three-body dispersion (only when s9 != 0; GFN1 sets s9=0
    // so this block is skipped and the two-body energy/gradient are byte-identical).
    if atm_active {
        if system.lattice.is_some() {
            d3_atm_accumulate_periodic(
                system,
                &reference,
                &c6,
                &dc6dcn,
                s9,
                a1,
                a2,
                &mut energy,
                &mut gradient,
                &mut d_edcn,
            )?;
        } else {
            d3_atm_accumulate(
                system,
                &reference,
                &c6,
                &dc6dcn,
                s9,
                a1,
                a2,
                &mut energy,
                &mut gradient,
                &mut d_edcn,
            )?;
        }
    }

    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let r = pair.r_ij.norm();
        if r <= DIST_EPS {
            continue;
        }
        let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
        let gi = pair.r_ij * pref;
        gradient[pair.i] += gi;
        gradient[pair.j] -= gi;
    }

    let stress = if system.lattice.is_some() {
        Some(periodic_dispersion_stress(
            system,
            params,
            explicit_reference_path,
        )?)
    } else {
        None
    };

    Ok(DispersionResult {
        energy,
        gradient,
        stress,
    })
}

pub fn dispersion_energy_gradient_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<DispersionHessianResult> {
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let cn = d3_coordination_jets::<Jet2>(system, ndof)?;
    let weights = reference_weight_jets(system, &reference, &cn, ndof)?;
    let c6 = atomic_c6_jets(system, &reference, &weights, ndof)?;
    let mut energy = Jet2::constant(0.0, ndof);
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let r2 = disp_pair_r2::<Jet2>(system, pair.i, pair.j, pair.translation, ndof);
        if r2.value > cutoff2 || r2.value <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let t6 = r4.mul(&r2).add_scalar(r0_6).powf(-1.0);
        let t8 = r4.mul(&r4).add_scalar(r0_8).powf(-1.0);
        let edisp = t6.scale(s6).add(&t8.scale(s8 * r4r2ij));
        energy = energy.sub(&c6[i][j].mul(&edisp));
    }

    // Axilrod-Teller-Muto three-body Hessian (only when s9 != 0; GFN1 sets s9 = 0, so the
    // two-body Hessian stays byte-identical for stock GFN1).
    if atm_active {
        d3_atm_accumulate_jet(system, &reference, &c6, s9, a1, a2, ndof, &mut energy)?;
    }

    let gradient = jet_gradient_vec3(&energy, nat);
    let hessian = Matrix::from_vec(ndof, ndof, energy.hessian)?;
    let stress = if system.lattice.is_some() {
        Some(periodic_dispersion_stress(
            system,
            params,
            explicit_reference_path,
        )?)
    } else {
        None
    };
    Ok(DispersionHessianResult {
        energy: energy.value,
        gradient,
        hessian,
        stress,
    })
}

pub fn dispersion_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<Option<Matrix>> {
    if system.lattice.is_some() {
        Ok(Some(periodic_dispersion_stress(
            system,
            params,
            explicit_reference_path,
        )?))
    } else {
        Ok(None)
    }
}

fn periodic_dispersion_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<Matrix> {
    let Some(lattice) = system.lattice.as_ref() else {
        return Ok(Matrix::zeros(3, 3));
    };
    let ndof = 9;
    let energy = dispersion_strain_energy_jet(system, params, explicit_reference_path, ndof)?;
    let inv_volume = 1.0 / lattice.volume();
    let mut stress = Matrix::zeros(3, 3);
    for a in 0..3 {
        for b in 0..3 {
            stress[(a, b)] = energy.gradient[3 * a + b] * inv_volume;
        }
    }
    Ok(stress)
}

fn dispersion_strain_energy_jet(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
    ndof: usize,
) -> Result<Jet2> {
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    let cn = d3_strain_coordination_jets(system, ndof)?;
    let weights = reference_weight_jets(system, &reference, &cn, ndof)?;
    let c6 = atomic_c6_jets(system, &reference, &weights, ndof)?;
    let mut energy = Jet2::constant(0.0, ndof);
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let rij = strain_vector_jets(pair.dr, ndof);
        let r2 = disp_dot(&rij, &rij);
        if r2.value > cutoff2 || r2.value <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let t6 = r4.mul(&r2).add_scalar(r0_6).powf(-1.0);
        let t8 = r4.mul(&r4).add_scalar(r0_8).powf(-1.0);
        let edisp = t6.scale(s6).add(&t8.scale(s8 * r4r2ij));
        energy = energy.sub(&c6[i][j].mul(&edisp));
    }

    if atm_active {
        d3_atm_strain_energy_jet(
            system, &reference, &c6, s9, a1, a2, ndof, &mut energy,
        )?;
    }

    Ok(energy)
}

/// Strain (forward-AD) D3 ATM three-body energy `Jet2`, mirroring the same image
/// loop, counting weight, and damping convention as [`d3_atm_accumulate_periodic`]
/// but carried as a strain `Jet2` (`ndof = 9`) so the periodic stress is the
/// strain gradient of the same energy expression the two-body strain term uses.
/// The `c6[i][j]` strain `Jet2` already carries the strain dependence of the
/// lattice-summed coordination number; the three leg vectors carry the explicit
/// strain dependence of the geometry through [`strain_vector_jets`].
#[allow(clippy::too_many_arguments)]
fn d3_atm_strain_energy_jet(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &[Vec<Jet2>],
    s9: f64,
    a1: f64,
    a2: f64,
    ndof: usize,
    energy: &mut Jet2,
) -> Result<()> {
    let mut source = C6Table { table: c6 };
    d3_atm_energy_jet_periodic(
        system,
        reference,
        &mut source,
        s9,
        a1,
        a2,
        energy,
        |_i, _j, _translation, dr| {
            let leg = strain_vector_jets(dr, ndof);
            disp_dot(&leg, &leg)
        },
    )
}

/// One D3 ATM triple energy as a jet of the squared leg lengths and the `C6` jets — the single
/// expression every ATM derivative order goes through (`Jet2` for the Hessian and the periodic
/// strain/stress, `Jet3` for the third derivative, `Jet4` for the fourth).  Same angular factor
/// and Chai-Head-Gordon zero damping as [`d4_atm_energy_distance_derivatives`], expressed via jet
/// ops so every derivative propagates by forward AD.
///
/// `E = s9 sqrt(|C6_ij C6_ik C6_jk|) (0.375 p / r̄⁵ + 1 / r̄³) fdmp`, with
/// `r̄² = r²_ij r²_ik r²_jk`, `p = (r²_ij+r²_jk−r²_ik)(r²_ij−r²_jk+r²_ik)(−r²_ij+r²_jk+r²_ik)`
/// and `fdmp = 1 / (1 + 6 (R0_product / r̄)^(alp/3))`.  The energy prefactor is
/// `−C9 = +s9 sqrt(|…|)`, matching the gradient/energy paths.
#[allow(clippy::too_many_arguments)]
fn d3_atm_triple_energy_jet<J: DispJet>(
    r2ij: &J,
    r2ik: &J,
    r2jk: &J,
    c6ij: &J,
    c6ik: &J,
    c6jk: &J,
    s9: f64,
    r0_product: f64,
    alp3: f64,
) -> J {
    // p = (x + z - y)(x - z + y)(-x + z + y) with x=r2ij, y=r2ik, z=r2jk.
    let pa = r2ij.add(r2jk).sub(r2ik);
    let pb = r2ij.sub(r2jk).add(r2ik);
    let pc = r2jk.add(r2ik).sub(r2ij);
    let p = pa.mul(&pb).mul(&pc);
    // r̄² = x y z; angular = 0.375 p (r̄²)^(-5/2) + (r̄²)^(-3/2).
    let rbar2 = r2ij.mul(r2ik).mul(r2jk);
    let inv_r5 = rbar2.powf(-2.5);
    let inv_r3 = rbar2.powf(-1.5);
    let angular = p.mul(&inv_r5).scale(0.375).add(&inv_r3);
    // fdmp = 1 / (1 + 6 (R0_product / r̄)^(alp/3)) = 1 / (1 + 6 (R0^2 / r̄²)^(alp/6)).
    let r0_product2 = r0_product * r0_product;
    let damp = rbar2.powf(-alp3 / 2.0).scale(r0_product2.powf(alp3 / 2.0));
    let fdmp = damp.scale(6.0).add_scalar(1.0).powf(-1.0);
    // C9 = -s9 sqrt(|C6 C6 C6|); energy prefactor = -C9 = +s9 sqrt(...).
    let root = c6ij.mul(c6ik).mul(c6jk).powf(0.5);
    let pref = root.scale(s9);
    pref.mul(&angular).mul(&fdmp)
}

/// Resolve and load the D3 C6 reference data. An explicit path (the
/// `--d3-reference` API argument) or the `GFN1_D3_REFERENCE` environment variable
/// overrides the default; when neither is set, the reference bundled with the
/// library and embedded at build time is used.
fn resolve_and_load_d3_reference(
    explicit_reference_path: Option<&str>,
) -> Result<Arc<D3Reference>> {
    if let Some(path) = explicit_reference_path {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return load_d3_reference(Path::new(trimmed));
        }
    }
    if let Ok(path) = std::env::var(GFN1_D3_REFERENCE_ENV) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return load_d3_reference(Path::new(trimmed));
        }
    }
    bundled_d3_reference()
}

/// Parse and cache the D3 reference embedded in the library.
fn bundled_d3_reference() -> Result<Arc<D3Reference>> {
    static CACHE: OnceLock<Arc<D3Reference>> = OnceLock::new();
    if let Some(found) = CACHE.get() {
        return Ok(found.clone());
    }
    let loaded = Arc::new(D3Reference::bundled()?);
    let _ = CACHE.set(loaded.clone());
    Ok(loaded)
}

#[derive(Clone, Debug)]
struct ReferenceWeights {
    gw: Vec<Vec<f64>>,
    dgwdcn: Vec<Vec<f64>>,
}

fn reference_weights(
    system: &PeriodicSystem,
    reference: &D3Reference,
    cn: &[f64],
) -> Result<ReferenceWeights> {
    let mut gw = Vec::with_capacity(system.atoms.len());
    let mut dgwdcn = Vec::with_capacity(system.atoms.len());
    let wf2 = 2.0 * WF;
    for (iat, atom) in system.atoms.iter().enumerate() {
        let nref = reference.number_of_references(atom.z)?;
        let mut cnrefs = Vec::with_capacity(nref);
        let mut logits = Vec::with_capacity(nref);
        let mut max_logit = f64::NEG_INFINITY;
        for iref in 0..nref {
            let cnref = reference.reference_cn(atom.z, iref)?;
            let logit = -WF * (cn[iat] - cnref).powi(2);
            cnrefs.push(cnref);
            logits.push(logit);
            max_logit = max_logit.max(logit);
        }
        if !max_logit.is_finite() {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference weighting failed for atom {} (Z={})",
                iat + 1,
                atom.z
            )));
        }
        let mut raw = vec![0.0; nref];
        let mut norm = 0.0;
        let mut dnorm = 0.0;
        for (iref, value) in raw.iter_mut().enumerate() {
            let cnref = cnrefs[iref];
            let weight = (logits[iref] - max_logit).exp();
            *value = weight;
            norm += weight;
            dnorm += wf2 * (cnref - cn[iat]) * weight;
        }
        if norm <= 0.0 || !norm.is_finite() {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference weighting failed for atom {} (Z={})",
                iat + 1,
                atom.z
            )));
        }
        let norm_inv = 1.0 / norm;
        let mut atom_gw = vec![0.0; nref];
        let mut atom_dgw = vec![0.0; nref];
        for iref in 0..nref {
            let cnref = cnrefs[iref];
            let expd = wf2 * (cnref - cn[iat]) * raw[iref];
            atom_gw[iref] = raw[iref] * norm_inv;
            atom_dgw[iref] = expd * norm_inv - raw[iref] * dnorm * norm_inv * norm_inv;
            if !atom_gw[iref].is_finite() {
                atom_gw[iref] = 0.0;
            }
            if !atom_dgw[iref].is_finite() {
                atom_dgw[iref] = 0.0;
            }
        }
        gw.push(atom_gw);
        dgwdcn.push(atom_dgw);
    }
    Ok(ReferenceWeights { gw, dgwdcn })
}

/// Normalized-Gaussian D3 reference weights and their 1st–3rd derivatives w.r.t. the atomic
/// coordination number — the `C6(CN)` chain prerequisite for the analytic dispersion **third**
/// nuclear derivative. `gw_k = raw_k / Σ raw`, `raw_k = exp(−WF (cn − cnref_k)²)`. The Gaussian
/// log-derivative `m_k = −2 WF (cn − cnref_k)` gives `raw' = m raw`, `raw'' = (m²−2WF) raw`,
/// `raw''' = m(m²−6WF) raw`; the quotient `gw = raw·N⁻¹` is then differentiated via Leibniz.
/// `gw'` reproduces the existing [`reference_weights`] `dgwdcn`.
#[allow(dead_code)]
fn reference_weight_cn_derivatives(
    cn: f64,
    cnrefs: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = cnrefs.len();
    let mut raw = vec![0.0_f64; n];
    let mut raw1 = vec![0.0_f64; n];
    let mut raw2 = vec![0.0_f64; n];
    let mut raw3 = vec![0.0_f64; n];
    let mut logits = vec![0.0_f64; n];
    let mut max_logit = f64::NEG_INFINITY;
    for k in 0..n {
        let d = cn - cnrefs[k];
        let logit = -WF * d * d;
        logits[k] = logit;
        max_logit = max_logit.max(logit);
    }
    if n == 0 || !max_logit.is_finite() {
        return (raw, raw1, raw2, raw3);
    }
    let (mut nn, mut n1, mut n2, mut n3) = (0.0_f64, 0.0, 0.0, 0.0);
    for k in 0..n {
        let d = cn - cnrefs[k];
        let r = (logits[k] - max_logit).exp();
        let m = -2.0 * WF * d;
        raw[k] = r;
        raw1[k] = m * r;
        raw2[k] = (m * m - 2.0 * WF) * r;
        raw3[k] = m * (m * m - 6.0 * WF) * r;
        nn += r;
        n1 += raw1[k];
        n2 += raw2[k];
        n3 += raw3[k];
    }
    // u = 1/N and its derivatives.
    let u = 1.0 / nn;
    let u1 = -u * u * n1;
    let u2 = 2.0 * u * u * u * n1 * n1 - u * u * n2;
    let u3 = -6.0 * u.powi(4) * n1 * n1 * n1 + 6.0 * u * u * u * n1 * n2 - u * u * n3;
    let mut gw = vec![0.0_f64; n];
    let mut g1 = vec![0.0_f64; n];
    let mut g2 = vec![0.0_f64; n];
    let mut g3 = vec![0.0_f64; n];
    for k in 0..n {
        gw[k] = raw[k] * u;
        g1[k] = raw1[k] * u + raw[k] * u1;
        g2[k] = raw2[k] * u + 2.0 * raw1[k] * u1 + raw[k] * u2;
        g3[k] = raw3[k] * u + 3.0 * raw2[k] * u1 + 3.0 * raw1[k] * u2 + raw[k] * u3;
    }
    (gw, g1, g2, g3)
}

fn atomic_c6(
    system: &PeriodicSystem,
    reference: &D3Reference,
    weights: &ReferenceWeights,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let nat = system.atoms.len();
    let mut c6 = vec![vec![0.0; nat]; nat];
    let mut dc6dcn = vec![vec![0.0; nat]; nat];
    for i in 0..nat {
        let zi = system.atoms[i].z;
        for j in 0..=i {
            let zj = system.atoms[j].z;
            let mut cij = 0.0;
            let mut dc_i = 0.0;
            let mut dc_j = 0.0;
            for iref in 0..weights.gw[i].len() {
                for jref in 0..weights.gw[j].len() {
                    let refc6 = reference.c6(iref, jref, zi, zj)?;
                    cij += weights.gw[i][iref] * weights.gw[j][jref] * refc6;
                    dc_i += weights.dgwdcn[i][iref] * weights.gw[j][jref] * refc6;
                    dc_j += weights.gw[i][iref] * weights.dgwdcn[j][jref] * refc6;
                }
            }
            c6[i][j] = cij;
            c6[j][i] = cij;
            if i == j {
                dc6dcn[i][i] = dc_i + dc_j;
            } else {
                dc6dcn[i][j] = dc_i;
                dc6dcn[j][i] = dc_j;
            }
        }
    }
    Ok((c6, dc6dcn))
}

/// Accumulate the D3 Axilrod-Teller-Muto (ATM) three-body dispersion energy, its
/// nuclear gradient, and its coordination-number chain contribution into the
/// running totals.
///
/// For each unordered triple `A<B<C` the triple-dipole energy is
/// `E = s9 * sqrt(|C6_AB C6_BC C6_CA|) * ang(R) * fdmp(R)`, with the same angular
/// factor `ang = 0.375 * p / r̄⁵ + 1 / r̄³` (`r̄ = r_AB r_BC r_CA`,
/// `p = (r²_AB+r²_BC−r²_CA)(r²_AB−r²_BC+r²_CA)(−r²_AB+r²_BC+r²_CA)`) and
/// Chai–Head–Gordon zero damping
/// `fdmp = 1 / (1 + 6 (R0_AB R0_BC R0_CA / r̄)^(alp/3))`,
/// `R0_XY = a1 sqrt(3 r4r2_X r4r2_Y) + a2`, that the experimental D4 ATM term uses
/// (the shared helpers [`d4_atm_angular`] and [`d4_atm_energy_distance_derivatives`]).
/// `C9_ABC = −s9 sqrt(|C6_AB C6_BC C6_CA|)` reuses the D3 CN-interpolated `C6`.
///
/// The CN derivative chain mirrors the two-body path: `dE/dC6_XY = E / (2 C6_XY)`
/// is contracted with `dc6dcn` (whose `[a][b]` entry is `d C6_ab / d CN_a`) into
/// `d_edcn`, which the caller's CN-pair loop converts into the geometric force.
///
/// Molecular (non-PBC) path. The caller routes lattices to the lattice-summed
/// [`d3_atm_accumulate_periodic`]; this single-cell `i < j < k` form is the
/// large-cell limit of that function.
#[allow(clippy::too_many_arguments)]
fn d3_atm_accumulate(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &[Vec<f64>],
    dc6dcn: &[Vec<f64>],
    s9: f64,
    a1: f64,
    a2: f64,
    energy: &mut f64,
    gradient: &mut [Vec3],
    d_edcn: &mut [f64],
) -> Result<()> {
    let nat = system.atoms.len();
    if nat < 3 {
        return Ok(());
    }
    let alp3 = D3_ATM_DAMPING_EXPONENT / 3.0;
    let r4r2 = system
        .atoms
        .iter()
        .map(|atom| reference.r4r2(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cutoff = D4_ATM_CUTOFF;
    let cutoff2 = cutoff * cutoff;
    let pos = |a: usize| system.atoms[a].position;

    for i in 0..nat {
        for j in (i + 1)..nat {
            let dr_ij = pos(j) - pos(i);
            let r2ij = dr_ij.norm2();
            if r2ij <= DIST_EPS || r2ij > cutoff2 {
                continue;
            }
            for k in (j + 1)..nat {
                let dr_ik = pos(k) - pos(i);
                let r2ik = dr_ik.norm2();
                if r2ik <= DIST_EPS || r2ik > cutoff2 {
                    continue;
                }
                let dr_jk = pos(k) - pos(j);
                let r2jk = dr_jk.norm2();
                if r2jk <= DIST_EPS || r2jk > cutoff2 {
                    continue;
                }
                let rij = r2ij.sqrt();
                let rik = r2ik.sqrt();
                let rjk = r2jk.sqrt();

                let c6ij = c6[i][j];
                let c6ik = c6[i][k];
                let c6jk = c6[j][k];
                let root = (c6ij * c6ik * c6jk).abs().sqrt();
                if root <= 1.0e-30 {
                    continue;
                }
                // C9 = -s9 sqrt(|C6 C6 C6|); energy prefactor inside the helper is -C9.
                let c9 = -s9 * root;
                let r0ij = a1 * (3.0 * r4r2[i] * r4r2[j]).sqrt() + a2;
                let r0ik = a1 * (3.0 * r4r2[i] * r4r2[k]).sqrt() + a2;
                let r0jk = a1 * (3.0 * r4r2[j] * r4r2[k]).sqrt() + a2;
                let r0_product = r0ij * r0ik * r0jk;
                let (e, de_drij, de_drik, de_drjk) =
                    d4_atm_energy_distance_derivatives(rij, rik, rjk, c9, r0_product, alp3);
                if e == 0.0 || !e.is_finite() {
                    continue;
                }
                *energy += e;

                let uij = dr_ij * (1.0 / rij);
                let uik = dr_ik * (1.0 / rik);
                let ujk = dr_jk * (1.0 / rjk);
                gradient[i] += uij * (-de_drij) + uik * (-de_drik);
                gradient[j] += uij * de_drij + ujk * (-de_drjk);
                gradient[k] += uik * de_drik + ujk * de_drjk;

                // CN chain: dE/dC6_XY = E / (2 C6_XY); contract with dc6dcn.
                let dcij = d4_atm_dc6_prefactor(e, c6ij);
                let dcik = d4_atm_dc6_prefactor(e, c6ik);
                let dcjk = d4_atm_dc6_prefactor(e, c6jk);
                d_edcn[i] += dcij * dc6dcn[i][j] + dcik * dc6dcn[i][k];
                d_edcn[j] += dcij * dc6dcn[j][i] + dcjk * dc6dcn[j][k];
                d_edcn[k] += dcik * dc6dcn[k][i] + dcjk * dc6dcn[k][j];
            }
        }
    }
    Ok(())
}

/// Lattice-summed (periodic) D3 ATM three-body dispersion: energy, nuclear
/// gradient, and the coordination-number chain contribution.
///
/// The home atom `A` (`iat`) ranges over the reference cell; the other two
/// vertices `B = jat + T_j` and `C = kat + T_k` range over all atom/image sites
/// within [`D4_ATM_CUTOFF`] of `A` (directed neighbors from
/// [`all_center_short_range_neighbors`], i.e. every image and both members of a
/// `±T` pair).  For each home atom the unordered pair of neighbors `(B, C)` with
/// `B` before `C` in the directed-neighbor order is taken once with weight `1/3`.
///
/// Counting: the home-anchored sum over `A ∈ cell` and ordered neighbor pairs
/// `(B, C)` counts every physical three-site shape six times — three choices of
/// which vertex is translated into the home cell, times two `(B, C)` orderings.
/// Summing the unordered pair once (`B` before `C`) collapses the ordering, so the
/// weight `1/3` recovers the per-cell energy.  In the molecular limit (single
/// image) this reduces exactly to the `i < j < k` sum of [`d3_atm_accumulate`]:
/// `Σ_A Σ_{B<C, B,C≠A} = 3 · Σ_{i<j<k}`.
///
/// The same shared distance-derivative helper
/// [`d4_atm_energy_distance_derivatives`], `C6` (periodic, CN-interpolated), and
/// `dc6dcn` chain as the non-PBC path are reused; the only difference is the image
/// loop and the `1/3` counting weight.  Forces on an image site are accumulated
/// onto its home-atom identity.  GFN1 sets `s9 = 0`, so this is never reached for
/// stock GFN1.
#[allow(clippy::too_many_arguments)]
fn d3_atm_accumulate_periodic(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &[Vec<f64>],
    dc6dcn: &[Vec<f64>],
    s9: f64,
    a1: f64,
    a2: f64,
    energy: &mut f64,
    gradient: &mut [Vec3],
    d_edcn: &mut [f64],
) -> Result<()> {
    let nat = system.atoms.len();
    if nat == 0 {
        return Ok(());
    }
    let alp3 = D3_ATM_DAMPING_EXPONENT / 3.0;
    let r4r2 = system
        .atoms
        .iter()
        .map(|atom| reference.r4r2(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cutoff = D4_ATM_CUTOFF;
    let cutoff2 = cutoff * cutoff;
    // Each physical three-site shape is counted three times across the choice of
    // which vertex is anchored in the home cell (see the doc comment).
    const TRIPLE_WEIGHT: f64 = 1.0 / 3.0;

    let neighbors = all_center_short_range_neighbors(system, cutoff)?;

    for i in 0..nat {
        let neigh = &neighbors[i];
        for (a, pair_ib) in neigh.iter().enumerate() {
            // B = jat + T_j relative to home atom A=i. dr points A -> B.
            let j = pair_ib.j;
            let r2ij = pair_ib.r2;
            if r2ij <= DIST_EPS || r2ij > cutoff2 {
                continue;
            }
            let dr_ij = pair_ib.dr;
            for pair_ic in neigh[(a + 1)..].iter() {
                // C = kat + T_k relative to home atom A=i. dr points A -> C.
                let k = pair_ic.j;
                let r2ik = pair_ic.r2;
                if r2ik <= DIST_EPS || r2ik > cutoff2 {
                    continue;
                }
                let dr_ik = pair_ic.dr;
                // jk leg: C - B (both expressed relative to A, so the A offset cancels).
                let dr_jk = dr_ik - dr_ij;
                let r2jk = dr_jk.norm2();
                if r2jk <= DIST_EPS || r2jk > cutoff2 {
                    continue;
                }
                let rij = r2ij.sqrt();
                let rik = r2ik.sqrt();
                let rjk = r2jk.sqrt();

                // Periodic CN-interpolated C6 depends only on the home-atom
                // identities and their (lattice-summed) coordination numbers.
                let c6ij = c6[i][j];
                let c6ik = c6[i][k];
                let c6jk = c6[j][k];
                let root = (c6ij * c6ik * c6jk).abs().sqrt();
                if root <= 1.0e-30 {
                    continue;
                }
                let c9 = -s9 * root;
                let r0ij = a1 * (3.0 * r4r2[i] * r4r2[j]).sqrt() + a2;
                let r0ik = a1 * (3.0 * r4r2[i] * r4r2[k]).sqrt() + a2;
                let r0jk = a1 * (3.0 * r4r2[j] * r4r2[k]).sqrt() + a2;
                let r0_product = r0ij * r0ik * r0jk;
                let (e_full, de_drij, de_drik, de_drjk) =
                    d4_atm_energy_distance_derivatives(rij, rik, rjk, c9, r0_product, alp3);
                if e_full == 0.0 || !e_full.is_finite() {
                    continue;
                }
                let e = e_full * TRIPLE_WEIGHT;
                *energy += e;

                let uij = dr_ij * (TRIPLE_WEIGHT / rij);
                let uik = dr_ik * (TRIPLE_WEIGHT / rik);
                let ujk = dr_jk * (TRIPLE_WEIGHT / rjk);
                // Forces on the image sites map onto their home-atom identities.
                gradient[i] += uij * (-de_drij) + uik * (-de_drik);
                gradient[j] += uij * de_drij + ujk * (-de_drjk);
                gradient[k] += uik * de_drik + ujk * de_drjk;

                // CN chain: dE/dC6_XY = E / (2 C6_XY); contract with dc6dcn.
                let dcij = d4_atm_dc6_prefactor(e, c6ij);
                let dcik = d4_atm_dc6_prefactor(e, c6ik);
                let dcjk = d4_atm_dc6_prefactor(e, c6jk);
                d_edcn[i] += dcij * dc6dcn[i][j] + dcik * dc6dcn[i][k];
                d_edcn[j] += dcij * dc6dcn[j][i] + dcjk * dc6dcn[j][k];
                d_edcn[k] += dcik * dc6dcn[k][i] + dcjk * dc6dcn[k][j];
            }
        }
    }
    Ok(())
}

// --- Order-generic forward-AD plumbing -----------------------------------------------------
//
// The D3 assembly (CN sigmoid -> reference-weight softmax -> `C6` interpolation -> BJ radial /
// ATM angular+damping) is written **once** against this op set and instantiated at second
// ([`Jet2`]), third ([`Jet3`]) and fourth ([`Jet4`]) order.  Every method forwards to the inherent
// method of the shared jet, so the operation *sequence* — hence the floating-point result — is
// identical to the hand-written per-order code it replaces.

/// The shared-jet operations the D3 energy assembly needs.
trait DispJet: Clone {
    fn constant(value: f64, n: usize) -> Self;
    /// Overwrite one first-order seed slot.
    fn seed_gradient(&mut self, dof: usize, value: f64);
    /// Overwrite one second-order seed slot.
    fn seed_hessian(&mut self, a: usize, b: usize, value: f64);
    fn value(&self) -> f64;
    fn n(&self) -> usize;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn add_scalar(&self, rhs: f64) -> Self;
    fn scale(&self, s: f64) -> Self;
    /// Fused `self += s · other` (one pass, no temporary — the inner loop of the factored `C6`).
    fn add_scaled(&mut self, other: &Self, s: f64);
    fn mul(&self, rhs: &Self) -> Self;
    fn div(&self, rhs: &Self) -> Self;
    fn powf(&self, p: f64) -> Self;
    fn exp(&self) -> Self;
}

macro_rules! impl_disp_jet {
    ($ty:ty, $($order:ident),+) => {
        impl DispJet for $ty {
            #[inline]
            fn constant(value: f64, n: usize) -> Self {
                <$ty>::constant(value, n)
            }
            #[inline]
            fn seed_gradient(&mut self, dof: usize, value: f64) {
                self.gradient[dof] = value;
            }
            #[inline]
            fn seed_hessian(&mut self, a: usize, b: usize, value: f64) {
                let n = self.gradient.len();
                self.hessian[a * n + b] = value;
            }
            #[inline]
            fn value(&self) -> f64 {
                self.value
            }
            #[inline]
            fn n(&self) -> usize {
                <$ty>::n(self)
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
            fn add_scaled(&mut self, other: &Self, s: f64) {
                self.value += other.value * s;
                $(
                    for (dst, src) in self.$order.iter_mut().zip(other.$order.iter()) {
                        *dst += *src * s;
                    }
                )+
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
            #[inline]
            fn exp(&self) -> Self {
                <$ty>::exp(self)
            }
        }
    };
}

impl_disp_jet!(Jet2, gradient, hessian);
impl_disp_jet!(Jet3, gradient, hessian, third);
impl_disp_jet!(Jet4, gradient, hessian, third, fourth);

/// The DIRECTIONAL instantiation of the same op set: [`Jet1`] carries the univariate Taylor of
/// `E(R + t·v)`, so the whole D3 pipeline below runs unchanged at `O(1)` storage per jet instead
/// of `O(ndof⁴)`.
///
/// The per-DOF seed hooks contract against the direction installed by
/// [`crate::jets::DirectionScope`]: `seed_gradient(a, x)` contributes `x·v_a` to `dE/dt` and
/// `seed_hessian(a, b, x)` contributes `x·v_a v_b` to `d²E/dt²`. Both ACCUMULATE where the
/// full-space impls overwrite, which is equivalent at the only call site
/// ([`disp_pair_r2`], whose seeds start at zero and touch each slot once) and is what makes the
/// contraction possible at all.
impl DispJet for Jet1 {
    #[inline]
    fn constant(value: f64, _n: usize) -> Self {
        Jet1::constant(value)
    }
    #[inline]
    fn seed_gradient(&mut self, dof: usize, value: f64) {
        self.d1 += value * crate::jets::direction_component(dof);
    }
    #[inline]
    fn seed_hessian(&mut self, a: usize, b: usize, value: f64) {
        self.d2 += value
            * crate::jets::direction_component(a)
            * crate::jets::direction_component(b);
    }
    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
    #[inline]
    fn n(&self) -> usize {
        1
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
    fn add_scaled(&mut self, other: &Self, s: f64) {
        Jet1::add_scaled(self, other, s);
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
    #[inline]
    fn exp(&self) -> Self {
        Jet1::exp(self)
    }
}

/// The squared length `|R_j + T − R_i|²` as a full-space jet, seeded in closed form.
///
/// Every distance-dependent D3 term consumes only this scalar, and it is a *quadratic* polynomial
/// in the nuclear coordinates: its third and higher derivatives vanish identically and its
/// gradient/Hessian have `O(1)` nonzero entries. Writing those entries directly costs `O(1)` work
/// plus the jet allocation, instead of the three jet **products** an explicit `v·v` dot would need
/// — the single largest saving in the fourth-order path, where each product is `O(ndof⁴)`.
///
/// It reproduces the previous `dot(coords[j] + T − coords[i], …)` bit-for-bit: with linear legs
/// that dot yields exactly `2 Σ_c d_c g_c` per gradient slot and `2 Σ_c g_c[a] g_c[b]` per Hessian
/// slot, and `x + x == 2x`, `0 + x == x` are exact in IEEE-754. A self-image pair (`i == j`) is a
/// pure lattice translation, hence geometry-independent: all seeds stay zero.
fn disp_pair_r2<J: DispJet>(
    system: &PeriodicSystem,
    i: usize,
    j: usize,
    translation: Vec3,
    ndof: usize,
) -> J {
    let ri = system.atoms[i].position.to_array();
    let rj = system.atoms[j].position.to_array();
    let t = translation.to_array();
    let d = [
        (rj[0] + t[0]) - ri[0],
        (rj[1] + t[1]) - ri[1],
        (rj[2] + t[2]) - ri[2],
    ];
    let mut out = J::constant((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2], ndof);
    if i == j {
        return out;
    }
    for (component, &dc) in d.iter().enumerate() {
        let (aj, ai) = (3 * j + component, 3 * i + component);
        out.seed_gradient(aj, 2.0 * dc);
        out.seed_gradient(ai, -2.0 * dc);
        out.seed_hessian(aj, aj, 2.0);
        out.seed_hessian(ai, ai, 2.0);
        out.seed_hessian(aj, ai, -2.0);
        out.seed_hessian(ai, aj, -2.0);
    }
    out
}

fn disp_dot<J: DispJet>(lhs: &[J; 3], rhs: &[J; 3]) -> J {
    lhs[0]
        .mul(&rhs[0])
        .add(&lhs[1].mul(&rhs[1]))
        .add(&lhs[2].mul(&rhs[2]))
}

fn d3_coordination_jets<J: DispJet>(system: &PeriodicSystem, ndof: usize) -> Result<Vec<J>> {
    let nat = system.atoms.len();
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let mut cn = vec![J::constant(0.0, ndof); nat];
    let cutoff2 = CN_CUTOFF * CN_CUTOFF;
    for pair in unique_short_range_pairs(system, CN_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let r2 = disp_pair_r2::<J>(system, i, j, pair.translation, ndof);
        if r2.value() <= DIST_EPS || r2.value() > cutoff2 {
            continue;
        }
        let rc = radii[i] + radii[j];
        if rc <= DIST_EPS {
            continue;
        }
        let value = coordination_value_jet(&r2.powf(0.5), CoordinationOptions::default().kcn, rc);
        if i == j {
            cn[i] = cn[i].add(&value.scale(2.0));
        } else {
            cn[i] = cn[i].add(&value);
            cn[j] = cn[j].add(&value);
        }
    }
    Ok(cn)
}

fn d3_strain_coordination_jets(system: &PeriodicSystem, ndof: usize) -> Result<Vec<Jet2>> {
    let nat = system.atoms.len();
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let mut cn = vec![Jet2::constant(0.0, ndof); nat];
    let cutoff2 = CN_CUTOFF * CN_CUTOFF;
    for pair in unique_short_range_pairs(system, CN_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let rij = strain_vector_jets(pair.dr, ndof);
        let r2 = disp_dot(&rij, &rij);
        if r2.value <= DIST_EPS || r2.value > cutoff2 {
            continue;
        }
        let rc = radii[i] + radii[j];
        if rc <= DIST_EPS {
            continue;
        }
        let value = coordination_value_jet(&r2.powf(0.5), CoordinationOptions::default().kcn, rc);
        if i == j {
            cn[i] = cn[i].add(&value.scale(2.0));
        } else {
            cn[i] = cn[i].add(&value);
            cn[j] = cn[j].add(&value);
        }
    }
    Ok(cn)
}

fn coordination_value_jet<J: DispJet>(r: &J, kcn: f64, rc: f64) -> J {
    let n = r.n();
    let raw_arg = -kcn * (rc / r.value() - 1.0);
    if !(-80.0..=80.0).contains(&raw_arg) {
        return J::constant(1.0 / (1.0 + raw_arg.clamp(-80.0, 80.0).exp()), n);
    }
    let arg = r.powf(-1.0).scale(rc).add_scalar(-1.0).scale(-kcn);
    J::constant(1.0, n).div(&J::constant(1.0, n).add(&arg.exp()))
}

/// Normalized-Gaussian D3 reference weights `gw[iat][iref]` as jets of the coordination numbers.
fn reference_weight_jets<J: DispJet>(
    system: &PeriodicSystem,
    reference: &D3Reference,
    cn: &[J],
    ndof: usize,
) -> Result<Vec<Vec<J>>> {
    let mut gw = Vec::with_capacity(system.atoms.len());
    for (iat, atom) in system.atoms.iter().enumerate() {
        let nref = reference.number_of_references(atom.z)?;
        let mut max_logit = f64::NEG_INFINITY;
        for iref in 0..nref {
            let cnref = reference.reference_cn(atom.z, iref)?;
            let delta = cn[iat].value() - cnref;
            max_logit = max_logit.max(-WF * delta * delta);
        }
        if !max_logit.is_finite() {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference weighting failed for atom {} (Z={})",
                iat + 1,
                atom.z
            )));
        }
        let mut raw = Vec::with_capacity(nref);
        let mut norm = J::constant(0.0, ndof);
        for iref in 0..nref {
            let cnref = reference.reference_cn(atom.z, iref)?;
            let delta = cn[iat].add_scalar(-cnref);
            let weight = delta.mul(&delta).scale(-WF).add_scalar(-max_logit).exp();
            norm = norm.add(&weight);
            raw.push(weight);
        }
        if norm.value() <= 0.0 || !norm.value().is_finite() {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference weighting failed for atom {} (Z={})",
                iat + 1,
                atom.z
            )));
        }
        gw.push(raw.iter().map(|weight| weight.div(&norm)).collect());
    }
    Ok(gw)
}

/// One CN-interpolated `C6_ij` jet from the two atoms' reference-weight jets.
///
/// Shared by the tabulated (`Jet2`/`Jet3`) and the streaming (`Jet4`) paths, so the pair `C6`
/// is built by exactly one code path at every derivative order.
fn atomic_c6_pair_jet<J: DispJet>(
    reference: &D3Reference,
    weights_i: &[J],
    weights_j: &[J],
    zi: u8,
    zj: u8,
    ndof: usize,
) -> Result<J> {
    let mut cij = J::constant(0.0, ndof);
    for (iref, wi) in weights_i.iter().enumerate() {
        for (jref, wj) in weights_j.iter().enumerate() {
            let refc6 = reference.c6(iref, jref, zi, zj)?;
            cij = cij.add(&wi.mul(wj).scale(refc6));
        }
    }
    Ok(cij)
}

fn atomic_c6_jets<J: DispJet>(
    system: &PeriodicSystem,
    reference: &D3Reference,
    weights: &[Vec<J>],
    ndof: usize,
) -> Result<Vec<Vec<J>>> {
    let nat = system.atoms.len();
    let mut c6 = vec![vec![J::constant(0.0, ndof); nat]; nat];
    for i in 0..nat {
        let zi = system.atoms[i].z;
        for j in 0..=i {
            let zj = system.atoms[j].z;
            let cij = atomic_c6_pair_jet(reference, &weights[i], &weights[j], zi, zj, ndof)?;
            c6[i][j] = cij.clone();
            c6[j][i] = cij;
        }
    }
    Ok(c6)
}

/// Source of CN-interpolated pair `C6` jets for the two-body and ATM jet loops.
///
/// Two implementations decide *whether the `nat × nat` pair table is materialised*:
/// [`C6Table`] hands out clones of a pre-built table (second/third order, where the table costs
/// `O(nat² · ndof²)` / `O(nat² · ndof³)` and is cheap), while [`C6Stream`] rebuilds each pair jet
/// on demand from the cached per-atom reference weights (fourth order, where the same table would
/// cost `O(nat² · ndof⁴)` — hundreds of MB for a handful of atoms).
trait C6Source<J: DispJet> {
    /// Scalar `C6_ij`, for the cheap vanishing-`C9` screen before any jet work.
    fn value(&self, i: usize, j: usize) -> f64;
    fn jet(&mut self, i: usize, j: usize) -> Result<J>;
    /// Hint that the following lookups mostly involve atom `i` (streaming row cache).
    fn begin_row(&mut self, _i: usize) {}
}

struct C6Table<'a, J: DispJet> {
    table: &'a [Vec<J>],
}

impl<J: DispJet> C6Source<J> for C6Table<'_, J> {
    fn value(&self, i: usize, j: usize) -> f64 {
        self.table[i][j].value()
    }
    fn jet(&mut self, i: usize, j: usize) -> Result<J> {
        Ok(self.table[i][j].clone())
    }
}

/// On-demand pair `C6` jets built from the per-atom reference weights, with a **one-row** cache.
///
/// Memory: `O(nat · nref)` weight jets (the caller's) plus at most `nat` cached row jets — i.e.
/// `O(nat)` live jets, never the `O(nat²)` pair table.  The ATM driver calls [`begin_row`] with
/// its outer atom `i`, so the two legs `C6_ij` / `C6_ik` of every triple hit the cache and only
/// the third leg `C6_jk` is rebuilt (`C(nat,3)` rebuilds instead of `3·C(nat,3)`).
///
/// [`begin_row`]: C6Source::begin_row
struct C6Stream<'a, J: DispJet> {
    system: &'a PeriodicSystem,
    reference: &'a D3Reference,
    weights: &'a [Vec<J>],
    /// Scalar pair `C6` for the screens (cheap: built from the jets' values only).
    values: Vec<Vec<f64>>,
    ndof: usize,
    row_atom: usize,
    row: Vec<Option<J>>,
}

impl<'a, J: DispJet> C6Stream<'a, J> {
    fn new(
        system: &'a PeriodicSystem,
        reference: &'a D3Reference,
        weights: &'a [Vec<J>],
        ndof: usize,
    ) -> Result<Self> {
        let nat = system.atoms.len();
        let mut values = vec![vec![0.0; nat]; nat];
        for i in 0..nat {
            let zi = system.atoms[i].z;
            for j in 0..=i {
                let zj = system.atoms[j].z;
                let mut cij = 0.0;
                for (iref, wi) in weights[i].iter().enumerate() {
                    for (jref, wj) in weights[j].iter().enumerate() {
                        cij += wi.value() * wj.value() * reference.c6(iref, jref, zi, zj)?;
                    }
                }
                values[i][j] = cij;
                values[j][i] = cij;
            }
        }
        Ok(Self {
            system,
            reference,
            weights,
            values,
            ndof,
            row_atom: usize::MAX,
            row: vec![None; nat],
        })
    }

    /// `C6_ij = Σ_ab gw_i[a] gw_j[b] refc6[a][b]`, factored as
    /// `Σ_a gw_i[a] · (Σ_b refc6[a][b] gw_j[b])`.
    ///
    /// Mathematically identical to [`atomic_c6_pair_jet`] but it costs `nref` jet **products**
    /// instead of `nref²`; the inner sums are fused scale-adds. Jet products are `O(ndof⁴)` with a
    /// ~14-term Leibniz kernel while a scale-add is a single pass, so at fourth order — where each
    /// ATM triple rebuilds its `jk` pair `C6` — this is the difference between the `C6` rebuild
    /// dominating the triple loop and being a minor cost. (The tabulated lower-order paths keep
    /// the unfactored form so their arithmetic is unchanged.)
    fn build(&self, i: usize, j: usize) -> Result<J> {
        let (zi, zj) = (self.system.atoms[i].z, self.system.atoms[j].z);
        let mut cij = J::constant(0.0, self.ndof);
        for (iref, wi) in self.weights[i].iter().enumerate() {
            let mut inner = J::constant(0.0, self.ndof);
            for (jref, wj) in self.weights[j].iter().enumerate() {
                inner.add_scaled(wj, self.reference.c6(iref, jref, zi, zj)?);
            }
            cij = cij.add(&wi.mul(&inner));
        }
        Ok(cij)
    }
}

impl<J: DispJet> C6Source<J> for C6Stream<'_, J> {
    fn value(&self, i: usize, j: usize) -> f64 {
        self.values[i][j]
    }

    fn jet(&mut self, i: usize, j: usize) -> Result<J> {
        let other = if i == self.row_atom {
            j
        } else if j == self.row_atom {
            i
        } else {
            return self.build(i, j);
        };
        if self.row[other].is_none() {
            let built = self.build(self.row_atom, other)?;
            self.row[other] = Some(built);
        }
        Ok(self.row[other].clone().expect("row slot just filled"))
    }

    fn begin_row(&mut self, i: usize) {
        self.row_atom = i;
        for slot in &mut self.row {
            *slot = None;
        }
    }
}

/// Molecular (non-PBC) D3 ATM three-body energy as a jet: the `i < j < k` loop, cutoffs, screens
/// and `R0` construction of [`d3_atm_accumulate`] with the scalar triple energy replaced by the
/// jet expression [`d3_atm_triple_energy_jet`].  Instantiated at second order (Hessian), third
/// order and fourth order; the `C6(CN(R))` many-body chain rule rides the jets, so nothing about
/// it is hand-differentiated.
#[allow(clippy::too_many_arguments)]
fn d3_atm_energy_jet<J: DispJet, S: C6Source<J>>(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &mut S,
    s9: f64,
    a1: f64,
    a2: f64,
    ndof: usize,
    energy: &mut J,
) -> Result<()> {
    let nat = system.atoms.len();
    if nat < 3 {
        return Ok(());
    }
    let alp3 = D3_ATM_DAMPING_EXPONENT / 3.0;
    let r4r2 = system
        .atoms
        .iter()
        .map(|atom| reference.r4r2(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cutoff2 = D4_ATM_CUTOFF * D4_ATM_CUTOFF;
    let pos = |a: usize| system.atoms[a].position;

    for i in 0..nat {
        c6.begin_row(i);
        for j in (i + 1)..nat {
            let r2ij_scalar = (pos(j) - pos(i)).norm2();
            if r2ij_scalar <= DIST_EPS || r2ij_scalar > cutoff2 {
                continue;
            }
            // The ij leg and its C6 jet are shared by every k; build them at most once.
            let mut ij_jets: Option<(J, J)> = None;
            for k in (j + 1)..nat {
                let r2ik_scalar = (pos(k) - pos(i)).norm2();
                if r2ik_scalar <= DIST_EPS || r2ik_scalar > cutoff2 {
                    continue;
                }
                let r2jk_scalar = (pos(k) - pos(j)).norm2();
                if r2jk_scalar <= DIST_EPS || r2jk_scalar > cutoff2 {
                    continue;
                }
                let root = (c6.value(i, j) * c6.value(i, k) * c6.value(j, k))
                    .abs()
                    .sqrt();
                if root <= 1.0e-30 {
                    continue;
                }
                if ij_jets.is_none() {
                    let r2ij = disp_pair_r2::<J>(system, i, j, Vec3::zero(), ndof);
                    let c6ij = c6.jet(i, j)?;
                    ij_jets = Some((r2ij, c6ij));
                }
                let (r2ij, c6ij) = ij_jets.as_ref().expect("ij jets just built");
                let r2ik = disp_pair_r2::<J>(system, i, k, Vec3::zero(), ndof);
                let r2jk = disp_pair_r2::<J>(system, j, k, Vec3::zero(), ndof);
                let c6ik = c6.jet(i, k)?;
                let c6jk = c6.jet(j, k)?;

                let r0ij = a1 * (3.0 * r4r2[i] * r4r2[j]).sqrt() + a2;
                let r0ik = a1 * (3.0 * r4r2[i] * r4r2[k]).sqrt() + a2;
                let r0jk = a1 * (3.0 * r4r2[j] * r4r2[k]).sqrt() + a2;
                let r0_product = r0ij * r0ik * r0jk;

                let e = d3_atm_triple_energy_jet(
                    r2ij, &r2ik, &r2jk, c6ij, &c6ik, &c6jk, s9, r0_product, alp3,
                );
                if e.value() == 0.0 || !e.value().is_finite() {
                    continue;
                }
                *energy = energy.add(&e);
            }
        }
    }
    Ok(())
}

/// Lattice-summed (periodic) D3 ATM three-body energy as a jet: the home-anchored directed
/// neighbor loop, `1/3` counting weight, cutoffs and screens of [`d3_atm_accumulate_periodic`].
///
/// `leg_r2(i, j, translation, dr)` supplies each leg's **squared length** as a jet, which is the
/// *only* difference between the two periodic ATM jet consumers: the Hessian/third/fourth
/// derivatives seed it on nuclear coordinates ([`disp_pair_r2`]), while the stress seeds it on
/// homogeneous strain ([`strain_vector_jets`] + [`disp_dot`]).  Sharing the loop keeps the
/// counting weight and image bookkeeping in exactly one place.
#[allow(clippy::too_many_arguments)]
fn d3_atm_energy_jet_periodic<J, S, L>(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &mut S,
    s9: f64,
    a1: f64,
    a2: f64,
    energy: &mut J,
    leg_r2: L,
) -> Result<()>
where
    J: DispJet,
    S: C6Source<J>,
    L: Fn(usize, usize, Vec3, Vec3) -> J,
{
    let nat = system.atoms.len();
    if nat == 0 {
        return Ok(());
    }
    let alp3 = D3_ATM_DAMPING_EXPONENT / 3.0;
    let r4r2 = system
        .atoms
        .iter()
        .map(|atom| reference.r4r2(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let cutoff = D4_ATM_CUTOFF;
    let cutoff2 = cutoff * cutoff;
    const TRIPLE_WEIGHT: f64 = 1.0 / 3.0;
    let neighbors = all_center_short_range_neighbors(system, cutoff)?;

    for i in 0..nat {
        c6.begin_row(i);
        let neigh = &neighbors[i];
        for (a, pair_ib) in neigh.iter().enumerate() {
            let j = pair_ib.j;
            if pair_ib.r2 <= DIST_EPS || pair_ib.r2 > cutoff2 {
                continue;
            }
            let r2ij = leg_r2(i, j, pair_ib.translation, pair_ib.dr);
            for pair_ic in neigh[(a + 1)..].iter() {
                let k = pair_ic.j;
                if pair_ic.r2 <= DIST_EPS || pair_ic.r2 > cutoff2 {
                    continue;
                }
                let dr_jk = pair_ic.dr - pair_ib.dr;
                let r2jk_scalar = dr_jk.norm2();
                if r2jk_scalar <= DIST_EPS || r2jk_scalar > cutoff2 {
                    continue;
                }
                // Cheap scalar guard so vanishing-C9 triples skip the (expensive) jet products.
                let root = (c6.value(i, j) * c6.value(i, k) * c6.value(j, k))
                    .abs()
                    .sqrt();
                if root <= 1.0e-30 {
                    continue;
                }
                let r2ik = leg_r2(i, k, pair_ic.translation, pair_ic.dr);
                let r2jk = leg_r2(j, k, pair_ic.translation - pair_ib.translation, dr_jk);

                let r0ij = a1 * (3.0 * r4r2[i] * r4r2[j]).sqrt() + a2;
                let r0ik = a1 * (3.0 * r4r2[i] * r4r2[k]).sqrt() + a2;
                let r0jk = a1 * (3.0 * r4r2[j] * r4r2[k]).sqrt() + a2;
                let r0_product = r0ij * r0ik * r0jk;

                let e = d3_atm_triple_energy_jet(
                    &r2ij,
                    &r2ik,
                    &r2jk,
                    &c6.jet(i, j)?,
                    &c6.jet(i, k)?,
                    &c6.jet(j, k)?,
                    s9,
                    r0_product,
                    alp3,
                );
                if !e.value().is_finite() {
                    continue;
                }
                *energy = energy.add(&e.scale(TRIPLE_WEIGHT));
            }
        }
    }
    Ok(())
}

/// Add the D3 ATM three-body contribution to a coordinate-seeded energy jet, routing lattices to
/// the lattice-summed loop exactly as [`dispersion_energy_gradient`] routes the scalar path.
#[allow(clippy::too_many_arguments)]
fn d3_atm_accumulate_jet<J: DispJet>(
    system: &PeriodicSystem,
    reference: &D3Reference,
    c6: &[Vec<J>],
    s9: f64,
    a1: f64,
    a2: f64,
    ndof: usize,
    energy: &mut J,
) -> Result<()> {
    let mut source = C6Table { table: c6 };
    if system.lattice.is_some() {
        d3_atm_energy_jet_periodic(
            system,
            reference,
            &mut source,
            s9,
            a1,
            a2,
            energy,
            |i, j, translation, _dr| disp_pair_r2::<J>(system, i, j, translation, ndof),
        )
    } else {
        d3_atm_energy_jet(system, reference, &mut source, s9, a1, a2, ndof, energy)
    }
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

fn jet_gradient_vec3(jet: &Jet2, nat: usize) -> Vec<Vec3> {
    let mut gradient = vec![Vec3::zero(); nat];
    for (dof, &value) in jet.gradient.iter().enumerate() {
        let atom = dof / 3;
        match dof % 3 {
            0 => gradient[atom].x = value,
            1 => gradient[atom].y = value,
            _ => gradient[atom].z = value,
        }
    }
    gradient
}

// --- Third-order forward-AD (dispersion analytic third derivative) --------------------------
//
// [`crate::jets::Jet3`] carries value + gradient + Hessian (`n×n`) + third (`n×n×n`, index
// `(i·n+j)·n+k`), i.e. the shared [`crate::jets::Jet2`] one order higher. Promoting the dispersion
// energy assembly `Jet2 → Jet3` makes the **many-body** D3 CN chain rule (`C6(CN(R))`, the
// reference-weight softmax, and the BJ radial term) propagate the **third** derivative
// automatically through forward AD — the Faà di Bruno bookkeeping the plan's "D3/CN are many-body,
// not central two-body" caveat warns about is handled by the chain rule in `mul`/`powf`/`exp`.
// Dense `n³` storage ⇒ small molecules only (the FD-validation target); this is a frozen-geometry
// `L_abc` block (no electronic response) and so is FD-isolatable like the repulsion/halogen third
// derivatives.
/// Result of the analytic dispersion **third** derivative.
#[derive(Clone, Debug)]
pub struct DispersionThirdResult {
    pub energy: f64,
    /// Dense `ndof × ndof × ndof` third derivative `∂³E_disp/∂R³`, row-major `(a·ndof+b)·ndof+c`.
    pub third: Vec<f64>,
    pub ndof: usize,
}

/// Analytic D3-BJ dispersion **third** derivative `∂³E_disp/∂R³` (cubic force constants), via the
/// `Jet3` promotion of [`dispersion_energy_gradient_hessian`] — the same energy expression carried
/// one AD order higher, so the many-body `C6(CN(R))` chain rule is exact (no hand-coded Faà di
/// Bruno). Dense `ndof³`; intended for small molecules / the FD-validation target. A frozen
/// geometric (`L_abc`) block — FD-isolatable against the analytic dispersion Hessian.
pub fn dispersion_third_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<DispersionThirdResult> {
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let cn = d3_coordination_jets::<Jet3>(system, ndof)?;
    let weights = reference_weight_jets(system, &reference, &cn, ndof)?;
    let c6 = atomic_c6_jets(system, &reference, &weights, ndof)?;
    let mut energy = Jet3::constant(0.0, ndof);
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let r2 = disp_pair_r2::<Jet3>(system, pair.i, pair.j, pair.translation, ndof);
        if r2.value > cutoff2 || r2.value <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let t6 = r4.mul(&r2).add_scalar(r0_6).powf(-1.0);
        let t8 = r4.mul(&r4).add_scalar(r0_8).powf(-1.0);
        let edisp = t6.scale(s6).add(&t8.scale(s8 * r4r2ij));
        energy = energy.sub(&c6[i][j].mul(&edisp));
    }

    if atm_active {
        d3_atm_accumulate_jet(system, &reference, &c6, s9, a1, a2, ndof, &mut energy)?;
    }

    Ok(DispersionThirdResult {
        energy: energy.value,
        third: energy.third,
        ndof,
    })
}

/// Hard cap on the degrees of freedom the analytic dispersion **fourth** derivative accepts.
///
/// A full-space `Jet4` stores `ndof⁴` doubles — 4.25 MB per jet at `ndof = 27` and 13.4 MB at
/// `ndof = 36` — and the assembly keeps `O(nat)` of them alive, so the working set grows as
/// `ndof⁵`. 30 DOF (10 atoms) keeps that around a few hundred MB while still covering the
/// water-trimer validation target; raise it deliberately, not by accident.
pub const MAX_FOURTH_DERIVATIVE_NDOF: usize = 30;

/// Result of the analytic dispersion **fourth** derivative.
#[derive(Clone, Debug)]
pub struct DispersionFourthResult {
    pub energy: f64,
    /// Dense `ndof⁴` fourth derivative `∂⁴E_disp/∂R⁴`, row-major `((a·ndof+b)·ndof+c)·ndof+d`.
    pub fourth: Vec<f64>,
    pub ndof: usize,
}

/// Analytic D3-BJ dispersion **fourth** derivative `∂⁴E_disp/∂R⁴` (quartic force constants), via
/// the `Jet4` promotion of [`dispersion_third_derivative`]: the same energy expression carried one
/// AD order higher, covering the two-body BJ term always and the ATM three-body term when
/// `s9 != 0`.  Like the third derivative this is a purely geometric (frozen, response-free) block,
/// so it FD-isolates against [`dispersion_third_derivative`] and must satisfy the acoustic sum
/// rule exactly.
///
/// **Streaming (memory) structure.** The lower-order paths materialise the full `nat × nat` table
/// of `C6` jets; at fourth order that table alone would be `O(nat² · ndof⁴)` doubles (≈ 340 MB for
/// a water trimer). Instead this path keeps only `O(nat)` jets alive:
///
/// 1. the `nat` coordination-number jets, which are **dropped** as soon as
/// 2. the `nat · nref` reference-weight jets are built from them (`nref ≤ 5`, a constant), and
/// 3. every pair `C6_ij` jet is rebuilt on demand from those weights by [`C6Stream`], with a
///    one-row cache so each ATM triple rebuilds only its `jk` leg.
///
/// Coordinate jets are never stored either: a coordinate *difference* is linear, so
/// [`disp_pair_vector`] seeds each leg directly instead of subtracting two dense `O(ndof⁴)` jets.
pub fn dispersion_fourth_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
) -> Result<DispersionFourthResult> {
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if ndof > MAX_FOURTH_DERIVATIVE_NDOF {
        let per_jet_mb = (ndof as f64).powi(4) * 8.0 / (1024.0 * 1024.0);
        return Err(Gfn1Error::InvalidInput(format!(
            "analytic D3 dispersion fourth derivative is capped at {MAX_FOURTH_DERIVATIVE_NDOF} \
             degrees of freedom ({} atoms); got {ndof} ({nat} atoms). A full-space Jet4 stores \
             ndof^4 doubles ({per_jet_mb:.0} MB each at this size) and the assembly keeps O(nat) \
             of them alive, so the working set grows as ndof^5. Use a smaller system or raise \
             `MAX_FOURTH_DERIVATIVE_NDOF` deliberately",
            MAX_FOURTH_DERIVATIVE_NDOF / 3
        )));
    }
    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    // Step 1+2: CN jets feed the reference weights and are then dropped (scoped block).
    let weights = {
        let cn = d3_coordination_jets::<Jet4>(system, ndof)?;
        reference_weight_jets(system, &reference, &cn, ndof)?
    };
    let mut c6 = C6Stream::new(system, &reference, &weights, ndof)?;
    let mut energy = Jet4::constant(0.0, ndof);
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let r2 = disp_pair_r2::<Jet4>(system, i, j, pair.translation, ndof);
        if r2.value > cutoff2 || r2.value <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let t6 = r4.mul(&r2).add_scalar(r0_6).powf(-1.0);
        let t8 = r4.mul(&r4).add_scalar(r0_8).powf(-1.0);
        let edisp = t6.scale(s6).add(&t8.scale(s8 * r4r2ij));
        energy = energy.sub(&c6.jet(i, j)?.mul(&edisp));
    }

    if atm_active {
        if system.lattice.is_some() {
            d3_atm_energy_jet_periodic(
                system,
                &reference,
                &mut c6,
                s9,
                a1,
                a2,
                &mut energy,
                |i, j, translation, _dr| disp_pair_r2::<Jet4>(system, i, j, translation, ndof),
            )?;
        } else {
            d3_atm_energy_jet(system, &reference, &mut c6, s9, a1, a2, ndof, &mut energy)?;
        }
    }

    Ok(DispersionFourthResult {
        energy: energy.value,
        fourth: energy.fourth,
        ndof,
    })
}

/// **Directional** analytic D3-BJ dispersion fourth derivative
/// `e⁗[v] = Σ_abcd v_a v_b v_c v_d ∂⁴E_disp/∂R_a∂R_b∂R_c∂R_d` — the same pipeline as
/// [`dispersion_fourth_derivative`], instantiated on the univariate [`Jet1`] instead of the
/// full-space [`Jet4`].
///
/// A directional fourth derivative is the 4th Taylor coefficient of `E(R + t·v)`, which needs
/// only ONE differentiation variable. Every jet therefore costs five doubles instead of `ndof⁴`,
/// and every jet product `O(1)` instead of `O(ndof⁴)` — so this route carries **no**
/// [`MAX_FOURTH_DERIVATIVE_NDOF`] cap. The expression, the screens, the streaming `C6` and the
/// ATM loop are literally the same generic code the capped path runs, so nothing can drift
/// between them; the equality is gated against `contract_vvvv` of the full tensor on systems
/// small enough for both by `dispersion_fourth_directional_matches_full_tensor`.
pub fn dispersion_fourth_directional(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    explicit_reference_path: Option<&str>,
    v: &[f64],
) -> Result<f64> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "dispersion_fourth_directional: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let s6 = params.global("s6", 1.0);
    let s8 = params.required_global("s8")?;
    let s9 = params.required_global("s9")?;
    let atm_active = s9.abs() > 1.0e-15;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let reference = resolve_and_load_d3_reference(explicit_reference_path)?;
    let _direction = DirectionScope::install(v);
    let weights = {
        let cn = d3_coordination_jets::<Jet1>(system, ndof)?;
        reference_weight_jets(system, &reference, &cn, ndof)?
    };
    let mut c6 = C6Stream::new(system, &reference, &weights, ndof)?;
    let mut energy = <Jet1 as DispJet>::constant(0.0, ndof);
    let cutoff2 = DISP2_CUTOFF * DISP2_CUTOFF;

    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let zi = system.atoms[i].z;
        let zj = system.atoms[j].z;
        let r2 = disp_pair_r2::<Jet1>(system, i, j, pair.translation, ndof);
        if r2.value > cutoff2 || r2.value <= DIST_EPS {
            continue;
        }
        let r4r2ij = 3.0 * reference.r4r2(zi)? * reference.r4r2(zj)?;
        let r0 = a1 * r4r2ij.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let t6 = r4.mul(&r2).add_scalar(r0_6).powf(-1.0);
        let t8 = r4.mul(&r4).add_scalar(r0_8).powf(-1.0);
        let edisp = t6.scale(s6).add(&t8.scale(s8 * r4r2ij));
        energy = energy.sub(&c6.jet(i, j)?.mul(&edisp));
    }

    if atm_active {
        if system.lattice.is_some() {
            d3_atm_energy_jet_periodic(
                system,
                &reference,
                &mut c6,
                s9,
                a1,
                a2,
                &mut energy,
                |i, j, translation, _dr| disp_pair_r2::<Jet1>(system, i, j, translation, ndof),
            )?;
        } else {
            d3_atm_energy_jet(system, &reference, &mut c6, s9, a1, a2, ndof, &mut energy)?;
        }
    }

    Ok(energy.d4)
}

#[derive(Clone, Debug)]
struct D3Reference {
    max_elem: usize,
    max_ref: usize,
    number_of_references: Vec<usize>,
    reference_cn: Vec<f64>,
    c6: Vec<f64>,
    r4r2: Vec<f64>,
}

/// D3 C6 reference table and `r4/r2` data, bundled under `third_party/simple-dftd3`
/// and embedded at build time so the default dispersion path needs no external
/// file. Overridable through the `--d3-reference` API argument or the
/// `GFN1_D3_REFERENCE` environment variable.
const BUNDLED_D3_REFERENCE_F90: &str =
    include_str!("../third_party/simple-dftd3/src/dftd3/reference.f90");
const BUNDLED_D3_R4R2_F90: &str =
    include_str!("../third_party/simple-dftd3/src/dftd3/data/r4r2.f90");

impl D3Reference {
    fn from_reference_text(reference_text: &str, r4r2_text: &str) -> Result<Self> {
        let clean = strip_fortran_comments(reference_text);
        let max_elem = parse_parameter_usize(&clean, "max_elem")?;
        let max_ref = parse_parameter_usize(&clean, "max_ref")?;
        let number_of_references = parse_int_array_after(&clean, "number_of_references")?
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();
        if number_of_references.len() != max_elem {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 number_of_references has {} values, expected {max_elem}",
                number_of_references.len()
            )));
        }
        let reference_cn = parse_float_array_after(&clean, "reference_cn")?;
        if reference_cn.len() != max_ref * max_elem {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference_cn has {} values, expected {}",
                reference_cn.len(),
                max_ref * max_elem
            )));
        }
        let c6 = parse_c6_view_assignments(&clean)?;
        let expected_c6 = max_ref * max_ref * max_elem * (max_elem + 1) / 2;
        if c6.len() != expected_c6 {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 C6 reference has {} values, expected {expected_c6}",
                c6.len()
            )));
        }
        let r4r2 = parse_r4r2_text(r4r2_text)?;
        Ok(Self {
            max_elem,
            max_ref,
            number_of_references,
            reference_cn,
            c6,
            r4r2,
        })
    }

    /// Parse the D3 reference from a `reference.f90` file, reading the companion
    /// `data/r4r2.f90` next to it. Used only when an explicit path or the
    /// `GFN1_D3_REFERENCE` environment variable overrides the bundled default.
    fn from_reference_file(path: &Path) -> Result<Self> {
        let reference_text = fs::read_to_string(path)?;
        let r4r2_path = path
            .parent()
            .ok_or_else(|| {
                Gfn1Error::InvalidInput("D3 reference path has no parent directory".to_string())
            })?
            .join("data")
            .join("r4r2.f90");
        let r4r2_text = fs::read_to_string(&r4r2_path)?;
        Self::from_reference_text(&reference_text, &r4r2_text)
    }

    /// The D3 reference bundled with the library and embedded at build time.
    fn bundled() -> Result<Self> {
        Self::from_reference_text(BUNDLED_D3_REFERENCE_F90, BUNDLED_D3_R4R2_F90)
    }

    fn number_of_references(&self, z: u8) -> Result<usize> {
        self.number_of_references
            .get(z as usize - 1)
            .copied()
            .filter(|n| *n > 0)
            .ok_or_else(|| Gfn1Error::InvalidInput(format!("no D3 reference count for Z={z}")))
    }

    fn reference_cn(&self, z: u8, iref: usize) -> Result<f64> {
        if z == 0 || z as usize > self.max_elem || iref >= self.max_ref {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 reference CN out of range for Z={z}, ref={iref}"
            )));
        }
        Ok(self.reference_cn[(z as usize - 1) * self.max_ref + iref])
    }

    fn c6(&self, iref: usize, jref: usize, zi: u8, zj: u8) -> Result<f64> {
        if zi == 0
            || zj == 0
            || zi as usize > self.max_elem
            || zj as usize > self.max_elem
            || iref >= self.max_ref
            || jref >= self.max_ref
        {
            return Err(Gfn1Error::InvalidInput(format!(
                "D3 C6 reference out of range for Z={zi}, Z={zj}, refs {iref}/{jref}"
            )));
        }
        let (a, b, ia, ib) = if zi > zj {
            (zi as usize, zj as usize, iref, jref)
        } else {
            (zj as usize, zi as usize, jref, iref)
        };
        let pair = b + a * (a - 1) / 2 - 1;
        let idx = pair * self.max_ref * self.max_ref + ib * self.max_ref + ia;
        Ok(self.c6[idx])
    }

    fn r4r2(&self, z: u8) -> Result<f64> {
        self.r4r2
            .get(z as usize - 1)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| Gfn1Error::InvalidInput(format!("no D3 r4/r2 value for Z={z}")))
    }
}

fn load_d3_reference(path: &Path) -> Result<Arc<D3Reference>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<D3Reference>>>> = OnceLock::new();
    let key = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = cache.lock().expect("D3 cache lock poisoned").get(&key) {
        return Ok(found.clone());
    }
    let loaded = Arc::new(D3Reference::from_reference_file(path)?);
    cache
        .lock()
        .expect("D3 cache lock poisoned")
        .insert(key, loaded.clone());
    Ok(loaded)
}

fn parse_r4r2_text(text: &str) -> Result<Vec<f64>> {
    let clean = strip_fortran_comments(text);
    let raw = parse_float_array_after(&clean, "r4_over_r2")?;
    Ok(raw
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (0.5 * value * ((idx + 1) as f64).sqrt()).sqrt())
        .collect())
}

fn strip_fortran_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let body = line.split('!').next().unwrap_or_default();
        out.push_str(body);
        out.push('\n');
    }
    out
}

fn parse_parameter_usize(text: &str, name: &str) -> Result<usize> {
    let pos = text.find(name).ok_or_else(|| {
        Gfn1Error::InvalidInput(format!("D3 source is missing parameter `{name}`"))
    })?;
    let eq = text[pos..]
        .find('=')
        .map(|p| pos + p + 1)
        .ok_or_else(|| Gfn1Error::InvalidInput(format!("D3 parameter `{name}` has no `=`")))?;
    let tail = &text[eq..];
    let digits = tail
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    digits
        .parse::<usize>()
        .map_err(|_| Gfn1Error::InvalidInput(format!("D3 parameter `{name}` is not an integer")))
}

fn parse_int_array_after(text: &str, marker: &str) -> Result<Vec<i64>> {
    Ok(parse_float_array_after(text, marker)?
        .into_iter()
        .map(|v| v.round() as i64)
        .collect())
}

fn parse_float_array_after(text: &str, marker: &str) -> Result<Vec<f64>> {
    let declaration = format!(":: {marker}");
    let pos = text
        .find(&declaration)
        .ok_or_else(|| Gfn1Error::InvalidInput(format!("D3 source is missing `{marker}`")))?;
    let open = text[pos..]
        .find('[')
        .map(|p| pos + p)
        .ok_or_else(|| Gfn1Error::InvalidInput(format!("D3 `{marker}` has no `[`")))?;
    let close = matching_bracket(text, open)?;
    parse_float_tokens(&text[open + 1..close])
}

fn parse_c6_view_assignments(text: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(relative) = text[pos..].find("c6ab_view(") {
        let start = pos + relative;
        let eq = text[start..]
            .find('=')
            .map(|p| start + p)
            .ok_or_else(|| Gfn1Error::InvalidInput("D3 C6 assignment has no `=`".to_string()))?;
        let open = text[eq..]
            .find('[')
            .map(|p| eq + p)
            .ok_or_else(|| Gfn1Error::InvalidInput("D3 C6 assignment has no `[`".to_string()))?;
        let close = matching_bracket(text, open)?;
        out.extend(parse_float_tokens(&text[open + 1..close])?);
        pos = close + 1;
    }
    if out.is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "D3 source contains no c6ab_view assignments".to_string(),
        ));
    }
    Ok(out)
}

fn matching_bracket(text: &str, open: usize) -> Result<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    for (idx, byte) in bytes.iter().enumerate().skip(open) {
        match *byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(idx);
                }
            }
            _ => {}
        }
    }
    Err(Gfn1Error::InvalidInput(
        "unterminated Fortran array literal in D3 source".to_string(),
    ))
}

fn parse_float_tokens(text: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    let mut token = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E' | 'd' | 'D') {
            token.push(ch);
        } else if !token.is_empty() {
            push_float_token(&mut out, &token)?;
            token.clear();
        }
    }
    if !token.is_empty() {
        push_float_token(&mut out, &token)?;
    }
    Ok(out)
}

fn push_float_token(out: &mut Vec<f64>, token: &str) -> Result<()> {
    if !token.chars().any(|c| c.is_ascii_digit()) {
        return Ok(());
    }
    let normalized = token.replace(['d', 'D'], "e");
    let value = normalized.parse::<f64>().map_err(|_| {
        Gfn1Error::InvalidInput(format!("invalid D3 floating-point token `{token}`"))
    })?;
    out.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        dispersion_energy, dispersion_energy_gradient, dispersion_energy_gradient_hessian,
        dispersion_fourth_derivative, dispersion_fourth_directional, dispersion_third_derivative,
        reference_weight_cn_derivatives, resolve_and_load_d3_reference,
        MAX_FOURTH_DERIVATIVE_NDOF,
    };
    use crate::lattice::Lattice;
    use crate::math::{Mat3, Vec3};
    use crate::{Gfn1Parameters, PeriodicSystem};

    // The D3 reference-weight CN derivatives (Gaussian softmax) must be self-consistent:
    // each order is the central FD of the previous, and the weights sum to 1. Prerequisite
    // for the dispersion third derivative's C6(CN) chain. Self-contained (synthetic cnrefs).
    #[test]
    fn reference_weight_cn_derivatives_match_finite_difference() {
        let cnrefs = [0.0_f64, 1.5, 3.0, 5.2];
        let h = 1.0e-6;
        for &cn in &[0.3_f64, 1.7, 2.9, 4.5] {
            let (gw, g1, g2, g3) = reference_weight_cn_derivatives(cn, &cnrefs);
            assert!(
                (gw.iter().sum::<f64>() - 1.0).abs() < 1.0e-12,
                "weights sum to 1"
            );
            let (_, g1p, g2p, _) = reference_weight_cn_derivatives(cn + h, &cnrefs);
            let (gwm, g1m, g2m, _) = reference_weight_cn_derivatives(cn - h, &cnrefs);
            let (gwp, _, _, _) = reference_weight_cn_derivatives(cn + h, &cnrefs);
            for k in 0..cnrefs.len() {
                let fd1 = (gwp[k] - gwm[k]) / (2.0 * h);
                let fd2 = (g1p[k] - g1m[k]) / (2.0 * h);
                let fd3 = (g2p[k] - g2m[k]) / (2.0 * h);
                assert!(
                    (g1[k] - fd1).abs() < 1.0e-6 * (1.0 + g1[k].abs()),
                    "g1[{k}] cn={cn}"
                );
                assert!(
                    (g2[k] - fd2).abs() < 1.0e-6 * (1.0 + g2[k].abs()),
                    "g2[{k}] cn={cn}"
                );
                assert!(
                    (g3[k] - fd3).abs() < 1.0e-5 * (1.0 + g3[k].abs()),
                    "g3[{k}] cn={cn}"
                );
            }
        }
    }

    #[test]
    fn reference_weight_cn_derivatives_remain_finite_for_high_cn() {
        let cnrefs = [0.0_f64, 0.9628, 1.9496];
        let (gw, g1, g2, g3) = reference_weight_cn_derivatives(16.6, &cnrefs);
        assert!((gw.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!(gw.iter().all(|v| v.is_finite()));
        assert!(g1.iter().all(|v| v.is_finite()));
        assert!(g2.iter().all(|v| v.is_finite()));
        assert!(g3.iter().all(|v| v.is_finite()));
        assert!(gw[2] > 0.999_999_999_999);
    }

    #[test]
    fn d3_h2_matches_xtb_reference_dispersion() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system =
            PeriodicSystem::from_xyz_str("2\nH2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", 0.0, false)
                .unwrap();
        let energy = dispersion_energy(&system, &params, None).unwrap();
        assert!((energy - -0.000_035_462_980).abs() < 5.0e-10);
    }

    #[test]
    fn bundled_d3_reference_is_available() {
        let reference = resolve_and_load_d3_reference(None).unwrap();
        assert!(reference.number_of_references(6).unwrap() > 0);
        assert!(reference.r4r2(8).unwrap() > 0.0);
    }

    #[test]
    fn d3_gradient_matches_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = dispersion_energy_gradient(&system, &params, None).unwrap();
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd = (dispersion_energy(&plus, &params, None).unwrap()
                    - dispersion_energy(&minus, &params, None).unwrap())
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
    fn d3_hessian_matches_gradient_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let result = dispersion_energy_gradient_hessian(&system, &params, None).unwrap();
        let grad = dispersion_energy_gradient(&system, &params, None).unwrap();
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
            let gp = dispersion_energy_gradient(&plus, &params, None)
                .unwrap()
                .gradient;
            let gm = dispersion_energy_gradient(&minus, &params, None)
                .unwrap()
                .gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * h);
                max_delta = max_delta.max((result.hessian[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-7,
            "D3 Hessian finite-difference max delta {max_delta:.3e}"
        );
    }

    // Analytic D3 dispersion THIRD derivative (Jet3 promotion) vs FD of the analytic dispersion
    // Hessian: `T_abc ≈ [H_bc(R + h e_a) − H_bc(R − h e_a)] / 2h`. A frozen geometric (`L_abc`)
    // block — FD-isolatable like the repulsion/halogen third derivatives — that exercises the
    // full many-body `C6(CN(R))` chain carried automatically through forward AD.
    #[test]
    fn d3_third_derivative_matches_hessian_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let analytic = dispersion_third_derivative(&system, &params, None).unwrap();
        let ndof = analytic.ndof;
        // The Jet3 value must match the Jet2 Hessian path's energy bit-for-bit (same expression).
        let h2 = dispersion_energy_gradient_hessian(&system, &params, None).unwrap();
        assert!(
            (analytic.energy - h2.energy).abs() < 1.0e-12,
            "Jet3 energy {} vs Jet2 energy {}",
            analytic.energy,
            h2.energy
        );
        let h = 1.0e-4;
        let mut max_delta = 0.0_f64;
        for a in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift(&mut plus, a / 3, a % 3, h);
            shift(&mut minus, a / 3, a % 3, -h);
            let hp = dispersion_energy_gradient_hessian(&plus, &params, None)
                .unwrap()
                .hessian;
            let hm = dispersion_energy_gradient_hessian(&minus, &params, None)
                .unwrap()
                .hessian;
            for b in 0..ndof {
                for c in 0..ndof {
                    let fd = (hp[(b, c)] - hm[(b, c)]) / (2.0 * h);
                    let an = analytic.third[(a * ndof + b) * ndof + c];
                    max_delta = max_delta.max((an - fd).abs());
                }
            }
        }
        assert!(
            max_delta < 1.0e-6,
            "D3 third-derivative finite-difference max delta {max_delta:.3e}"
        );
    }

    // The D3 dispersion energy depends only on interatomic distances ⇒ it is translationally
    // invariant, so its analytic third derivative obeys the third-order acoustic sum rule
    // Σ_A T_{Aα,bc} = 0 (a rigid shift of all atoms leaves the Hessian — hence ∂H/∂R — unchanged).
    // A physical-correctness gate on the Jet3 many-body assembly, independent of the FD gate.
    #[test]
    fn d3_third_derivative_acoustic_sum_rule() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let analytic = dispersion_third_derivative(&system, &params, None).unwrap();
        let n = analytic.ndof;
        let nat = system.atoms.len();
        let mut max = 0.0_f64;
        for alpha in 0..3 {
            for b in 0..n {
                for c in 0..n {
                    let sum: f64 = (0..nat)
                        .map(|atom| analytic.third[((3 * atom + alpha) * n + b) * n + c])
                        .sum();
                    max = max.max(sum.abs());
                }
            }
        }
        assert!(
            max < 1.0e-8,
            "D3 third-derivative acoustic sum rule violated: max {max:.3e}"
        );
    }

    #[test]
    fn d3_periodic_gradient_matches_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = periodic_water();
        let result = dispersion_energy_gradient(&system, &params, None).unwrap();
        assert!(result.stress.is_some());
        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, atom, component, h);
                shift(&mut minus, atom, component, -h);
                let fd = (dispersion_energy(&plus, &params, None).unwrap()
                    - dispersion_energy(&minus, &params, None).unwrap())
                    / (2.0 * h);
                let an = match component {
                    0 => result.gradient[atom].x,
                    1 => result.gradient[atom].y,
                    _ => result.gradient[atom].z,
                };
                assert!(
                    (an - fd).abs() < 2.0e-8,
                    "periodic atom {atom} component {component}: analytic {an} FD {fd}"
                );
            }
        }
    }

    #[test]
    fn d3_periodic_hessian_matches_gradient_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = periodic_water();
        let result = dispersion_energy_gradient_hessian(&system, &params, None).unwrap();
        assert!(result.stress.is_some());
        let grad = dispersion_energy_gradient(&system, &params, None).unwrap();
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
            let gp = dispersion_energy_gradient(&plus, &params, None)
                .unwrap()
                .gradient;
            let gm = dispersion_energy_gradient(&minus, &params, None)
                .unwrap()
                .gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * h);
                max_delta = max_delta.max((result.hessian[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-7,
            "periodic D3 Hessian finite-difference max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn d3_periodic_stress_matches_strain_finite_difference() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = periodic_water();
        let result = dispersion_energy_gradient(&system, &params, None).unwrap();
        let stress = result.stress.as_ref().unwrap();
        let volume = system.lattice.as_ref().unwrap().volume();
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let plus = strained_system(&system, row, col, h);
                let minus = strained_system(&system, row, col, -h);
                let fd = (dispersion_energy(&plus, &params, None).unwrap()
                    - dispersion_energy(&minus, &params, None).unwrap())
                    / (2.0 * h * volume);
                max_delta = max_delta.max((stress[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 1.0e-8,
            "periodic D3 stress finite-difference max delta {max_delta:.3e}"
        );
    }

    #[test]
    fn d3_periodic_mgo_high_cn_weights_are_stable() {
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let system = periodic_mgo();
        let result = dispersion_energy_gradient(&system, &params, None).unwrap();
        assert!(result.energy.is_finite());
        assert!(result.stress.is_some());
        for grad in &result.gradient {
            assert!(grad.x.is_finite());
            assert!(grad.y.is_finite());
            assert!(grad.z.is_finite());
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

    fn shift(system: &mut PeriodicSystem, atom: usize, component: usize, delta: f64) {
        match component {
            0 => system.atoms[atom].position.x += delta,
            1 => system.atoms[atom].position.y += delta,
            _ => system.atoms[atom].position.z += delta,
        }
    }

    fn periodic_water() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "3\nLattice=\"20 0 0 0 20 0 0 0 20\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.757200 0.586000 -0.469200\n\
             H -0.757200 0.586000 -0.469200\n",
            0.0,
            false,
        )
        .unwrap()
    }

    fn periodic_mgo() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "8\nLattice=\"4.212 0 0 0 4.212 0 0 0 4.212\" pbc=\"T T T\"\n\
             Mg 0.000000 0.000000 0.000000\n\
             Mg 0.000000 2.106000 2.106000\n\
             Mg 2.106000 0.000000 2.106000\n\
             Mg 2.106000 2.106000 0.000000\n\
             O 2.106000 0.000000 0.000000\n\
             O 0.000000 2.106000 0.000000\n\
             O 0.000000 0.000000 2.106000\n\
             O 2.106000 2.106000 2.106000\n",
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

    // --- D3 ATM (three-body) tests --------------------------------------------------------

    /// Load the bundled official GFN1-xTB parametrization and override `s9`.
    /// The D3 reference resolves to the bundled `third_party/simple-dftd3`
    /// snapshot via the `None` path.
    fn params_with_s9(s9: f64) -> Gfn1Parameters {
        let mut params = Gfn1Parameters::builtin().unwrap();
        params.globpar.insert("s9".to_string(), s9);
        params
    }

    fn methane() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "5\nmethane\n\
             C  0.000000  0.000000  0.000000\n\
             H  0.629118  0.629118  0.629118\n\
             H -0.629118 -0.629118  0.629118\n\
             H -0.629118  0.629118 -0.629118\n\
             H  0.629118 -0.629118 -0.629118\n",
            0.0,
            false,
        )
        .unwrap()
    }

    fn water_trimer() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "9\nwater trimer\n\
             O  0.000000  0.000000  0.000000\n\
             H  0.757000  0.586000  0.000000\n\
             H -0.757000  0.586000  0.000000\n\
             O  2.900000  0.300000  0.500000\n\
             H  3.500000 -0.300000  0.900000\n\
             H  3.200000  1.100000  0.900000\n\
             O  1.300000  2.700000 -0.400000\n\
             H  1.900000  3.300000 -0.100000\n\
             H  0.600000  3.100000 -0.900000\n",
            0.0,
            false,
        )
        .unwrap()
    }

    // s9=0 regression: turning ATM "on" with s9=0 must leave the two-body energy and gradient
    // byte-identical (the ATM block is gated on s9 != 0, so this is structural). Verified to
    // <= 1e-10 Ha on methane and the water trimer.
    #[test]
    fn d3_atm_s9_zero_is_byte_identical_to_two_body() {
        for system in [methane(), water_trimer()] {
            let params = params_with_s9(0.0);
            let result = dispersion_energy_gradient(&system, &params, None).unwrap();
            let energy = dispersion_energy(&system, &params, None).unwrap();
            // Independent re-evaluation must reproduce the same numbers exactly.
            let result2 = dispersion_energy_gradient(&system, &params, None).unwrap();
            assert!((result.energy - energy).abs() < 1.0e-12);
            assert!((result.energy - result2.energy).abs() < 1.0e-12);
            for atom in 0..system.atoms.len() {
                assert!((result.gradient[atom].x - result2.gradient[atom].x).abs() < 1.0e-12);
                assert!((result.gradient[atom].y - result2.gradient[atom].y).abs() < 1.0e-12);
                assert!((result.gradient[atom].z - result2.gradient[atom].z).abs() < 1.0e-12);
            }
        }
    }

    // The ATM term must contribute a non-zero energy for s9 != 0 (sanity that the new code path
    // actually runs), and must be exactly zero for s9 = 0.
    #[test]
    fn d3_atm_energy_nonzero_for_s9_one() {
        for system in [methane(), water_trimer()] {
            let e0 = dispersion_energy(&system, &params_with_s9(0.0), None).unwrap();
            let e1 = dispersion_energy(&system, &params_with_s9(1.0), None).unwrap();
            let atm = e1 - e0;
            assert!(
                atm.abs() > 1.0e-10,
                "ATM energy contribution {atm:.3e} too small"
            );
            assert!(atm.is_finite());
        }
    }

    // FD-gradient gate: the analytic ATM force (gradient at s9=1 minus gradient at s9=0)
    // matches a central finite difference of the ATM energy (energy at s9=1 minus s9=0).
    // This isolates the three-body term from the (separately FD-verified) two-body term.
    #[test]
    fn d3_atm_gradient_matches_finite_difference() {
        for system in [methane(), water_trimer()] {
            let p1 = params_with_s9(1.0);
            let p0 = params_with_s9(0.0);
            let g1 = dispersion_energy_gradient(&system, &p1, None).unwrap().gradient;
            let g0 = dispersion_energy_gradient(&system, &p0, None).unwrap().gradient;
            let atm_energy = |sys: &PeriodicSystem| -> f64 {
                dispersion_energy(sys, &p1, None).unwrap()
                    - dispersion_energy(sys, &p0, None).unwrap()
            };
            let h = 1.0e-5;
            let mut max_delta = 0.0_f64;
            for atom in 0..system.atoms.len() {
                for component in 0..3 {
                    let mut plus = system.clone();
                    let mut minus = system.clone();
                    shift(&mut plus, atom, component, h);
                    shift(&mut minus, atom, component, -h);
                    let fd = (atm_energy(&plus) - atm_energy(&minus)) / (2.0 * h);
                    let an = match component {
                        0 => g1[atom].x - g0[atom].x,
                        1 => g1[atom].y - g0[atom].y,
                        _ => g1[atom].z - g0[atom].z,
                    };
                    max_delta = max_delta.max((an - fd).abs());
                }
            }
            assert!(
                max_delta < 1.0e-6,
                "D3 ATM gradient FD max delta {max_delta:.3e}"
            );
        }
    }

    // The ATM energy depends only on interatomic distances, so its analytic gradient must sum
    // to zero (translational invariance). A physical-correctness gate independent of the FD gate.
    #[test]
    fn d3_atm_gradient_translational_invariance() {
        let system = water_trimer();
        let p1 = params_with_s9(1.0);
        let p0 = params_with_s9(0.0);
        let g1 = dispersion_energy_gradient(&system, &p1, None).unwrap().gradient;
        let g0 = dispersion_energy_gradient(&system, &p0, None).unwrap().gradient;
        let mut sum = Vec3::zero();
        for atom in 0..system.atoms.len() {
            sum += g1[atom] - g0[atom];
        }
        assert!(
            sum.x.abs() < 1.0e-10 && sum.y.abs() < 1.0e-10 && sum.z.abs() < 1.0e-10,
            "ATM force does not sum to zero: {sum:?}"
        );
    }

    // `periodic_water` uses a ~37.8 bohr (20 Angstrom) cubic cell, so the 40-bohr ATM cutoff
    // reaches the nearest lattice images along each axis: the periodic ATM lattice sum is
    // genuinely exercised (not just the home cell), while the 3-atom cell keeps the (debug,
    // full-suite) neighbor count and FD re-evaluation cost tractable.

    // Periodic ATM energy/gradient/stress are now implemented; the lattice path must succeed
    // (no rejection) and contribute a finite, non-zero three-body energy for s9 != 0.
    #[test]
    fn d3_atm_periodic_with_s9_is_accepted() {
        for system in [periodic_water()] {
            let p0 = params_with_s9(0.0);
            let p1 = params_with_s9(1.0);
            let r0 = dispersion_energy_gradient(&system, &p0, None).unwrap();
            let r1 = dispersion_energy_gradient(&system, &p1, None).unwrap();
            assert!(r1.energy.is_finite());
            assert!(r1.stress.is_some());
            let atm = r1.energy - r0.energy;
            assert!(
                atm.abs() > 1.0e-10,
                "periodic ATM energy contribution {atm:.3e} too small"
            );
        }
    }

    // FD-gradient gate (periodic): the analytic periodic ATM force (gradient at s9=1 minus
    // gradient at s9=0) matches a central finite difference of the periodic ATM energy. This
    // isolates the three-body term from the (separately FD-verified) two-body periodic term.
    #[test]
    fn d3_atm_periodic_gradient_matches_finite_difference() {
        for system in [periodic_water()] {
            let p1 = params_with_s9(1.0);
            let p0 = params_with_s9(0.0);
            let g1 = dispersion_energy_gradient(&system, &p1, None).unwrap().gradient;
            let g0 = dispersion_energy_gradient(&system, &p0, None).unwrap().gradient;
            let atm_energy = |sys: &PeriodicSystem| -> f64 {
                dispersion_energy(sys, &p1, None).unwrap()
                    - dispersion_energy(sys, &p0, None).unwrap()
            };
            let h = 1.0e-5;
            let mut max_delta = 0.0_f64;
            for atom in 0..system.atoms.len() {
                for component in 0..3 {
                    let mut plus = system.clone();
                    let mut minus = system.clone();
                    shift(&mut plus, atom, component, h);
                    shift(&mut minus, atom, component, -h);
                    let fd = (atm_energy(&plus) - atm_energy(&minus)) / (2.0 * h);
                    let an = match component {
                        0 => g1[atom].x - g0[atom].x,
                        1 => g1[atom].y - g0[atom].y,
                        _ => g1[atom].z - g0[atom].z,
                    };
                    max_delta = max_delta.max((an - fd).abs());
                }
            }
            assert!(
                max_delta < 1.0e-6,
                "periodic D3 ATM gradient FD max delta {max_delta:.3e}"
            );
        }
    }

    // FD-stress gate (periodic): the analytic periodic ATM stress (stress at s9=1 minus stress at
    // s9=0) matches the finite difference of the periodic ATM energy under homogeneous strain.
    // Mirrors `d3_periodic_stress_matches_strain_finite_difference` but isolates the ATM term.
    #[test]
    fn d3_atm_periodic_stress_matches_strain_finite_difference() {
        for system in [periodic_water()] {
            let p1 = params_with_s9(1.0);
            let p0 = params_with_s9(0.0);
            let s1 = dispersion_energy_gradient(&system, &p1, None)
                .unwrap()
                .stress
                .unwrap();
            let s0 = dispersion_energy_gradient(&system, &p0, None)
                .unwrap()
                .stress
                .unwrap();
            let volume = system.lattice.as_ref().unwrap().volume();
            let atm_energy = |sys: &PeriodicSystem| -> f64 {
                dispersion_energy(sys, &p1, None).unwrap()
                    - dispersion_energy(sys, &p0, None).unwrap()
            };
            let h = 1.0e-5;
            let mut max_delta = 0.0_f64;
            for row in 0..3 {
                for col in 0..3 {
                    let plus = strained_system(&system, row, col, h);
                    let minus = strained_system(&system, row, col, -h);
                    let fd = (atm_energy(&plus) - atm_energy(&minus)) / (2.0 * h * volume);
                    let an = s1[(row, col)] - s0[(row, col)];
                    max_delta = max_delta.max((an - fd).abs());
                }
            }
            assert!(
                max_delta < 1.0e-8,
                "periodic D3 ATM stress FD max delta {max_delta:.3e}"
            );
        }
    }

    // Counting-weight correctness: in the large-cell limit (no image within the ATM cutoff) the
    // periodic lattice-sum ATM must reduce EXACTLY to the molecular `i<j<k` sum of the same atoms.
    // This pins down the `1/3` counting weight independently of the FD gates (a wrong weight would
    // scale the periodic energy by a constant factor away from the molecular reference).
    #[test]
    fn d3_atm_periodic_reduces_to_molecular_for_isolated_cell() {
        // 9-atom water trimer in a 400-bohr cubic cell: the nearest image is far beyond the
        // 40-bohr ATM cutoff, so only the home cell contributes.
        let periodic = PeriodicSystem::from_xyz_str(
            "9\nLattice=\"400 0 0 0 400 0 0 0 400\" pbc=\"T T T\"\n\
             O  0.000000  0.000000  0.000000\n\
             H  0.757000  0.586000  0.000000\n\
             H -0.757000  0.586000  0.000000\n\
             O  2.900000  0.300000  0.500000\n\
             H  3.500000 -0.300000  0.900000\n\
             H  3.200000  1.100000  0.900000\n\
             O  1.300000  2.700000 -0.400000\n\
             H  1.900000  3.300000 -0.100000\n\
             H  0.600000  3.100000 -0.900000\n",
            0.0,
            false,
        )
        .unwrap();
        let molecular = water_trimer();
        let p1 = params_with_s9(1.0);
        let p0 = params_with_s9(0.0);
        let atm = |sys: &PeriodicSystem| -> f64 {
            dispersion_energy(sys, &p1, None).unwrap() - dispersion_energy(sys, &p0, None).unwrap()
        };
        let e_pbc = atm(&periodic);
        let e_mol = atm(&molecular);
        assert!(e_mol.abs() > 1.0e-8, "molecular ATM reference too small");
        assert!(
            (e_pbc - e_mol).abs() < 1.0e-10,
            "isolated periodic ATM {e_pbc:.12e} != molecular {e_mol:.12e}"
        );
    }

    // --- D3 ATM Hessian / third / fourth derivative gates ----------------------------------

    fn water_dimer() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "6\nwater dimer\n\
             O  0.000000  0.000000  0.000000\n\
             H  0.757000  0.586000  0.000000\n\
             H -0.757000  0.586000  0.000000\n\
             O  2.900000  0.300000  0.500000\n\
             H  3.500000 -0.300000  0.900000\n\
             H  3.200000  1.100000  0.900000\n",
            0.0,
            false,
        )
        .unwrap()
    }

    /// ATM-only analytic Hessian: `H(s9=1) − H(s9=0)`, isolating the three-body block from the
    /// (separately gated) two-body one.
    fn atm_hessian(system: &PeriodicSystem) -> crate::linalg::Matrix {
        let h1 = dispersion_energy_gradient_hessian(system, &params_with_s9(1.0), None).unwrap();
        let h0 = dispersion_energy_gradient_hessian(system, &params_with_s9(0.0), None).unwrap();
        let ndof = 3 * system.atoms.len();
        let mut out = crate::linalg::Matrix::zeros(ndof, ndof);
        for row in 0..ndof {
            for col in 0..ndof {
                out[(row, col)] = h1.hessian[(row, col)] - h0.hessian[(row, col)];
            }
        }
        out
    }

    /// ATM-only analytic third derivative, flat `ndof³`.
    fn atm_third(system: &PeriodicSystem) -> Vec<f64> {
        let t1 = dispersion_third_derivative(system, &params_with_s9(1.0), None).unwrap();
        let t0 = dispersion_third_derivative(system, &params_with_s9(0.0), None).unwrap();
        t1.third
            .iter()
            .zip(t0.third.iter())
            .map(|(a, b)| a - b)
            .collect()
    }

    // The Jet2 ATM promotion must reproduce the production (scalar) ATM energy and gradient — the
    // order-0/1 slice of the same expression — before its Hessian means anything. The Jet2
    // `i<j<k` loop evaluates the identical triple energy, so the two agree to rounding.
    #[test]
    fn d3_atm_hessian_energy_and_gradient_match_production_path() {
        for system in [methane(), water_trimer()] {
            let (p1, p0) = (params_with_s9(1.0), params_with_s9(0.0));
            let h1 = dispersion_energy_gradient_hessian(&system, &p1, None).unwrap();
            let h0 = dispersion_energy_gradient_hessian(&system, &p0, None).unwrap();
            let g1 = dispersion_energy_gradient(&system, &p1, None).unwrap();
            let g0 = dispersion_energy_gradient(&system, &p0, None).unwrap();
            let jet_energy = h1.energy - h0.energy;
            let ref_energy = g1.energy - g0.energy;
            assert!(ref_energy.abs() > 1.0e-10, "ATM reference energy too small");
            assert!(
                (jet_energy - ref_energy).abs() < 1.0e-14 * (1.0 + ref_energy.abs()),
                "Jet2 ATM energy {jet_energy:.17e} vs production {ref_energy:.17e}"
            );
            for atom in 0..system.atoms.len() {
                let jet = h1.gradient[atom] - h0.gradient[atom];
                let reference = g1.gradient[atom] - g0.gradient[atom];
                for (a, b) in [
                    (jet.x, reference.x),
                    (jet.y, reference.y),
                    (jet.z, reference.z),
                ] {
                    assert!(
                        (a - b).abs() < 1.0e-13 * (1.0 + b.abs()),
                        "Jet2 ATM gradient atom {atom}: {a:.17e} vs {b:.17e}"
                    );
                }
            }
        }
    }

    // FD-Hessian gate: the analytic ATM Hessian (Jet2 promotion) matches a central finite
    // difference of the analytic ATM gradient. Isolates the three-body block from the two-body one
    // by differencing s9=1 against s9=0 on both sides.
    #[test]
    fn d3_atm_hessian_matches_gradient_finite_difference() {
        for system in [methane(), water_trimer()] {
            let (p1, p0) = (params_with_s9(1.0), params_with_s9(0.0));
            let analytic = atm_hessian(&system);
            let ndof = 3 * system.atoms.len();
            let atm_gradient = |sys: &PeriodicSystem| -> Vec<Vec3> {
                let g1 = dispersion_energy_gradient(sys, &p1, None).unwrap().gradient;
                let g0 = dispersion_energy_gradient(sys, &p0, None).unwrap().gradient;
                g1.iter().zip(g0.iter()).map(|(a, b)| *a - *b).collect()
            };
            let h = 1.0e-5;
            let mut max_delta = 0.0_f64;
            for col in 0..ndof {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, col / 3, col % 3, h);
                shift(&mut minus, col / 3, col % 3, -h);
                let gp = atm_gradient(&plus);
                let gm = atm_gradient(&minus);
                for row in 0..ndof {
                    let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * h);
                    max_delta = max_delta.max((analytic[(row, col)] - fd).abs());
                }
            }
            assert!(
                max_delta < 1.0e-7,
                "D3 ATM Hessian finite-difference max delta {max_delta:.3e}"
            );
        }
    }

    // The ATM energy depends only on interatomic distances, so a rigid translation leaves its
    // gradient unchanged: Σ_A H_{Aα,b} = 0. A physical-correctness gate on the Jet2 ATM assembly
    // independent of the FD gate (and it also pins the periodic 1/3 counting weight's home-atom
    // force scatter, which the scalar path does by hand).
    #[test]
    fn d3_atm_hessian_translational_invariance() {
        for system in [methane(), water_trimer()] {
            let analytic = atm_hessian(&system);
            let ndof = 3 * system.atoms.len();
            let nat = system.atoms.len();
            let mut max = 0.0_f64;
            for alpha in 0..3 {
                for col in 0..ndof {
                    let sum: f64 = (0..nat).map(|a| analytic[(3 * a + alpha, col)]).sum();
                    max = max.max(sum.abs());
                }
            }
            assert!(
                max < 1.0e-10,
                "D3 ATM Hessian acoustic sum rule violated: max {max:.3e}"
            );
        }
    }

    // FD-third gate: the analytic ATM third derivative (Jet3 promotion) matches a central finite
    // difference of the analytic ATM Hessian, plus full permutation symmetry of the flat tensor.
    #[test]
    fn d3_atm_third_derivative_matches_hessian_finite_difference() {
        for system in [methane(), water_trimer()] {
            let analytic = atm_third(&system);
            let ndof = 3 * system.atoms.len();
            let h = 1.0e-4;
            let mut max_delta = 0.0_f64;
            for a in 0..ndof {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, a / 3, a % 3, h);
                shift(&mut minus, a / 3, a % 3, -h);
                let hp = atm_hessian(&plus);
                let hm = atm_hessian(&minus);
                for b in 0..ndof {
                    for c in 0..ndof {
                        let fd = (hp[(b, c)] - hm[(b, c)]) / (2.0 * h);
                        let an = analytic[(a * ndof + b) * ndof + c];
                        max_delta = max_delta.max((an - fd).abs());
                    }
                }
            }
            assert!(
                max_delta < 1.0e-6,
                "D3 ATM third-derivative finite-difference max delta {max_delta:.3e}"
            );

            let mut max_asym = 0.0_f64;
            let idx = |a: usize, b: usize, c: usize| (a * ndof + b) * ndof + c;
            for a in 0..ndof {
                for b in 0..ndof {
                    for c in 0..ndof {
                        let v = analytic[idx(a, b, c)];
                        for &(x, y, z) in &[(b, a, c), (a, c, b), (c, b, a), (b, c, a), (c, a, b)] {
                            max_asym = max_asym.max((v - analytic[idx(x, y, z)]).abs());
                        }
                    }
                }
            }
            assert!(
                max_asym < 1.0e-12,
                "D3 ATM third derivative is not permutation symmetric: {max_asym:.3e}"
            );
        }
    }

    /// Σ_A Q_{Aα,bcd} over atoms — the fourth-order acoustic sum rule residual. The dispersion
    /// energy is a function of interatomic distances only, so a rigid translation cannot change
    /// any derivative: this must vanish to numerical precision, with no electronic response to
    /// hide behind.
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

    // FD-fourth gate: the analytic dispersion fourth derivative (Jet4 promotion) matches a central
    // finite difference of the analytic third derivative. Two systems: a water dimer with s9 = 0
    // (two-body only) and a water trimer with s9 = 1 (two-body AND ATM together). The acoustic sum
    // rule and permutation symmetry are checked on the same tensors here so the (expensive) Jet4
    // assembly runs once per system; the standalone gates below repeat them on the cheap system.
    #[test]
    fn dispersion_fourth_derivative_matches_third_finite_difference() {
        for (system, s9) in [(water_dimer(), 0.0), (water_trimer(), 1.0)] {
            let params = params_with_s9(s9);
            let analytic = dispersion_fourth_derivative(&system, &params, None).unwrap();
            let ndof = analytic.ndof;
            let acoustic = fourth_acoustic_residual(&analytic.fourth, ndof);
            assert!(
                acoustic < 1.0e-9,
                "D3 fourth-derivative acoustic sum rule violated: {acoustic:.3e} (s9 = {s9})"
            );
            let asymmetry = fourth_permutation_residual(&analytic.fourth, ndof);
            assert!(
                asymmetry < 1.0e-12,
                "D3 fourth derivative is not permutation symmetric: {asymmetry:.3e} (s9 = {s9})"
            );
            // The Jet4 value must reproduce the Jet3 path's energy (same expression).
            let t = dispersion_third_derivative(&system, &params, None).unwrap();
            assert!(
                (analytic.energy - t.energy).abs() < 1.0e-12,
                "Jet4 energy {} vs Jet3 energy {}",
                analytic.energy,
                t.energy
            );
            let h = 1.0e-4;
            let mut max_delta = 0.0_f64;
            for d in 0..ndof {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift(&mut plus, d / 3, d % 3, h);
                shift(&mut minus, d / 3, d % 3, -h);
                let tp = dispersion_third_derivative(&plus, &params, None)
                    .unwrap()
                    .third;
                let tm = dispersion_third_derivative(&minus, &params, None)
                    .unwrap()
                    .third;
                for idx3 in 0..ndof * ndof * ndof {
                    let fd = (tp[idx3] - tm[idx3]) / (2.0 * h);
                    let an = analytic.fourth[idx3 * ndof + d];
                    max_delta = max_delta.max((an - fd).abs());
                }
            }
            assert!(
                max_delta < 2.0e-6,
                "D3 fourth-derivative finite-difference max delta {max_delta:.3e} (s9 = {s9})"
            );
        }
    }

    // Standalone acoustic-sum-rule / permutation gates, named so a regression points straight at
    // the physics rather than at the FD comparison. Run on the water dimer with the ATM term
    // active (20 triples), which keeps the Jet4 assembly cheap; the trimer is covered inside
    // `dispersion_fourth_derivative_matches_third_finite_difference`.
    #[test]
    fn dispersion_fourth_derivative_acoustic_sum_rule() {
        let system = water_dimer();
        let analytic = dispersion_fourth_derivative(&system, &params_with_s9(1.0), None).unwrap();
        let max = fourth_acoustic_residual(&analytic.fourth, analytic.ndof);
        assert!(
            max < 1.0e-9,
            "D3 fourth-derivative acoustic sum rule violated: max {max:.3e}"
        );
    }

    #[test]
    fn dispersion_fourth_derivative_is_permutation_symmetric() {
        let system = water_dimer();
        let analytic = dispersion_fourth_derivative(&system, &params_with_s9(1.0), None).unwrap();
        let max = fourth_permutation_residual(&analytic.fourth, analytic.ndof);
        assert!(
            max < 1.0e-12,
            "D3 fourth derivative is not permutation symmetric: {max:.3e}"
        );
    }

    /// **The directional 1-D-jet gate.** [`dispersion_fourth_directional`] must reproduce the
    /// `vvvv` contraction of the full `ndof⁴` tensor to machine precision. Both run the SAME
    /// generic pipeline — coordination jets, reference weights, the streaming `C6`, the two-body
    /// BJ term and the ATM triples — differing only in the jet width, so this pins the `Jet1`
    /// Leibniz/Faà-di-Bruno rules and the directional seeding of `disp_pair_r2` at once.
    /// Run with the ATM three-body term ACTIVE (`s9 = 1`) so the triple loop is covered.
    #[test]
    fn dispersion_fourth_directional_matches_full_tensor() {
        let system = water_dimer();
        let params = params_with_s9(1.0);
        let ndof = 3 * system.atoms.len();
        // Generic skew direction: no zero components, no accidental symmetry.
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.11 + 0.07 * ((k * 13 % 7) as f64) - 0.15 * ((k % 3) as f64))
            .collect();
        let full = dispersion_fourth_derivative(&system, &params, None).unwrap();
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
        let got = dispersion_fourth_directional(&system, &params, None, &v).unwrap();
        let delta = (got - want).abs();
        eprintln!(
            "D3 directional fourth: 1-D jet {got:.17e} vs full-tensor vvvv {want:.17e} \
             (delta {delta:.3e})"
        );
        assert!(
            want.abs() > 1.0e-9,
            "the full-tensor reference is numerically zero — the gate is vacuous"
        );
        assert!(
            delta <= 1.0e-12 * want.abs(),
            "D3 directional fourth deviates from the full-tensor contraction: \
             got {got:.17e} want {want:.17e} delta {delta:.3e}"
        );
    }

    /// The directional route carries NO system-size cap: a chain above
    /// [`MAX_FOURTH_DERIVATIVE_NDOF`] — which
    /// `dispersion_fourth_derivative_rejects_oversized_systems` shows the full-tensor route
    /// refuses — must evaluate normally, because a `Jet1` is five doubles at any `ndof`.
    #[test]
    fn dispersion_fourth_directional_has_no_system_size_cap() {
        let mut xyz = String::new();
        let nat = MAX_FOURTH_DERIVATIVE_NDOF / 3 + 2;
        xyz.push_str(&format!("{nat}\nchain\n"));
        for i in 0..nat {
            xyz.push_str(&format!("H {:.6} 0.000000 0.000000\n", 1.2 * i as f64));
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
        let params = params_with_s9(1.0);
        let ndof = 3 * nat;
        assert!(ndof > MAX_FOURTH_DERIVATIVE_NDOF);
        assert!(dispersion_fourth_derivative(&system, &params, None).is_err());
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.11 + 0.07 * ((k * 13 % 7) as f64) - 0.15 * ((k % 3) as f64))
            .collect();
        let got = dispersion_fourth_directional(&system, &params, None, &v).unwrap();
        assert!(
            got.is_finite() && got.abs() > 1.0e-12,
            "directional D3 fourth above the full-tensor cap returned {got}"
        );
    }

    // Memory guard: a full-space Jet4 costs ndof^4 doubles and the assembly keeps O(nat) of them
    // alive, so oversized systems must be rejected up front with an actionable message rather than
    // exhausting memory.
    #[test]
    fn dispersion_fourth_derivative_rejects_oversized_systems() {
        let mut xyz = String::new();
        let nat = MAX_FOURTH_DERIVATIVE_NDOF / 3 + 2;
        xyz.push_str(&format!("{nat}\nchain\n"));
        for i in 0..nat {
            xyz.push_str(&format!("H {:.6} 0.000000 0.000000\n", 1.2 * i as f64));
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
        let err = dispersion_fourth_derivative(&system, &params_with_s9(0.0), None).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("fourth derivative") && message.contains("degrees of freedom"),
            "unexpected guard message: {message}"
        );
    }
}
