// SPDX-License-Identifier: GPL-3.0-or-later
//! External electromagnetic field perturbations for GFN1-xTB.
//!
//! The uniform **electric field** `E` is coupled to the GFN1 Mulliken monopoles,
//! consistent with the point-charge electrostatics of the method. The
//! electrostatic potential of a uniform field at position `r` is
//! `phi(r) = -E·(r - origin)`, so a shell carrying net charge `q_i` on atom `A`
//! feels an external site potential
//!
//! ```text
//! v_ext_i = -E · (R_A - origin)
//! ```
//!
//! which is added to the self-consistent shell potential exactly like the SCC
//! potential. The corresponding energy is `E_field = sum_i q_i v_ext_i = -E·mu`
//! with the Mulliken dipole `mu = sum_A q_A (R_A - origin)`. This couples the
//! field self-consistently (the density polarizes) and yields analytic forces and
//! the static dipole/polarizability response.
//!
//! The uniform **magnetic field** is scaffolding only; see [`crate::magnetic`].

use crate::basis::BasisSet;
use crate::math::Vec3;
use crate::system::PeriodicSystem;

/// Uniform external field perturbations applied on top of the GFN1 baseline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExternalFieldOptions {
    /// Uniform external electric field `E` in atomic units (Hartree / (e * a0)).
    pub electric_field: Option<Vec3>,
    /// Uniform external magnetic field `B` in atomic units. Foothold only — the
    /// SCC rejects an enabled magnetic field until a physical term is wired up
    /// (see [`crate::magnetic`]).
    pub magnetic_field: Option<Vec3>,
    /// Gauge / reference origin for the position operator `r` used by the fields.
    pub origin: Vec3,
}

impl ExternalFieldOptions {
    /// A pure electric-field perturbation referenced to the coordinate origin.
    pub fn electric(field: Vec3) -> Self {
        Self {
            electric_field: Some(field),
            magnetic_field: None,
            origin: Vec3::zero(),
        }
    }

    /// Whether any field perturbation is enabled.
    pub fn is_active(&self) -> bool {
        self.electric_field.is_some() || self.magnetic_field.is_some()
    }
}

/// Per-shell external electric potential `v_ext_i = -E·(R_atom(i) - origin)`.
///
/// Returns `None` when no electric field is set (so the caller can skip the
/// extra work entirely).
pub fn electric_shell_potential(
    options: &ExternalFieldOptions,
    system: &PeriodicSystem,
    basis: &BasisSet,
) -> Option<Vec<f64>> {
    let field = options.electric_field?;
    let mut potential = vec![0.0; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let r = system.atoms[shell.atom_index].position - options.origin;
        potential[ish] = -field.dot(r);
    }
    Some(potential)
}

/// External-field interaction energy `E_field = sum_i q_i v_ext_i`.
pub fn electric_field_energy(shell_potential: &[f64], shell_charges: &[f64]) -> f64 {
    shell_potential
        .iter()
        .zip(shell_charges.iter())
        .map(|(v, q)| v * q)
        .sum()
}

/// Mulliken (monopole) dipole moment `mu = sum_A q_A (R_A - origin)` in atomic
/// units (e * a0).
pub fn mulliken_dipole(system: &PeriodicSystem, atomic_charges: &[f64], origin: Vec3) -> Vec3 {
    let mut mu = Vec3::zero();
    for (atom, &q) in system.atoms.iter().zip(atomic_charges.iter()) {
        mu += (atom.position - origin) * q;
    }
    mu
}
