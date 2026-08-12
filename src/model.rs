// SPDX-License-Identifier: GPL-3.0-or-later

use crate::lattice::ImageOffset;
use crate::math::Vec3;
use crate::system::PeriodicSystem;

/// Real-space cutoffs (in Bohr) shared by the terms that consume them.
///
/// Only the cutoffs that are actually wired into a term live here. Before
/// v0.5.0 this struct also carried `coulomb`, `dispersion` and `halogen`
/// fields that no code ever read — the Coulomb, dispersion and halogen terms
/// carry their own cutoffs (e.g. [`crate::halogen`]'s tblite-faithful 20 bohr),
/// so the unused entries only advertised limits that were never enforced.
#[derive(Clone, Copy, Debug)]
pub struct Cutoffs {
    pub repulsion: f64,
    /// Coordination-number cutoff. [`crate::coordination::CoordinationOptions`]'s
    /// default must match this so the standalone CN helper and the CN the
    /// Hamiltonian builds agree.
    pub coordination: f64,
    pub integral: f64,
}

impl Default for Cutoffs {
    fn default() -> Self {
        Self {
            repulsion: 25.0,
            coordination: 30.0,
            integral: 30.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryCondition {
    NonPeriodic,
    GammaPointPbc,
    KPointPbc,
}

#[derive(Clone, Copy, Debug)]
pub struct KPoint {
    pub fractional: [f64; 3],
    pub weight: f64,
}

impl KPoint {
    pub const fn gamma() -> Self {
        Self {
            fractional: [0.0, 0.0, 0.0],
            weight: 1.0,
        }
    }
}

pub fn image_translations(system: &PeriodicSystem, cutoff: f64) -> Vec<(ImageOffset, Vec3)> {
    match &system.lattice {
        Some(lattice) => lattice
            .image_offsets(cutoff)
            .into_iter()
            .map(|offset| (offset, lattice.translation(offset)))
            .collect(),
        None => vec![(ImageOffset::origin(), Vec3::zero())],
    }
}
