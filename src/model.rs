// SPDX-License-Identifier: GPL-3.0-or-later

use crate::lattice::ImageOffset;
use crate::math::Vec3;
use crate::system::PeriodicSystem;

#[derive(Clone, Copy, Debug)]
pub struct Cutoffs {
    pub repulsion: f64,
    pub coordination: f64,
    pub coulomb: f64,
    pub integral: f64,
    pub dispersion: f64,
    pub halogen: f64,
}

impl Default for Cutoffs {
    fn default() -> Self {
        Self {
            repulsion: 25.0,
            coordination: 30.0,
            coulomb: 30.0,
            integral: 30.0,
            dispersion: 60.0,
            halogen: 30.0,
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
