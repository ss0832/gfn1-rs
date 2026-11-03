// SPDX-License-Identifier: GPL-3.0-or-later
//! Track the fixed real-space cutoff branch required by nuclear derivatives.
//! A finite displacement that changes a nonzero cutoff term is not a local
//! derivative reference at the original geometry; reduce the displacement and
//! record the rejected branches instead of loosening the derivative tolerance.
use crate::pbc::PbcOptions;
use crate::{ElectronicOptions, Gfn1Error, PeriodicSystem, Result};
type PairKey = (usize, usize, [i32; 3]);
pub struct NuclearNeighborhood {
    cutoffs: Vec<(&'static str, f64)>,
    keys: Vec<Vec<PairKey>>,
    species: Vec<u8>,
    cell_key: Vec<u64>,
    periodic: [bool; 3],
}
impl NuclearNeighborhood {
    pub fn periodic_scalar(
        system: &PeriodicSystem,
        options: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> Result<Self> {
        if system.lattice.is_none() {
            return Err(Gfn1Error::InvalidInput(
                "periodic nuclear neighborhood requires a lattice".into(),
            ));
        }
        let shared = crate::model::Cutoffs::default();
        let mut cutoffs = vec![
            ("ao", pbc.ao_cutoff),
            ("repulsion", shared.repulsion),
            ("ewald_real", pbc.ewald.real_cutoff),
            ("qcore_short", pbc.ewald.sr_cutoff),
            (
                "qcore_r3_real",
                5.5 / crate::pbc::ewald::resolve_alpha(system, &pbc.ewald),
            ),
        ];
        if options.hamiltonian.enable_cn_hamiltonian {
            cutoffs.push(("hamiltonian_cn", options.hamiltonian.coordination_cutoff));
        }
        if options.enable_dispersion {
            cutoffs.extend([
                ("d3_cn", crate::dispersion::CN_CUTOFF),
                ("d3_two_body", crate::dispersion::DISP2_CUTOFF),
            ]);
        }
        if let Some(model) = &options.ml_model {
            cutoffs.push(("ml_descriptor", model.cutoff_bohr));
        }
        cutoffs.retain(|(_, r)| *r > 0.);
        if cutoffs.iter().any(|(_, r)| !r.is_finite()) {
            return Err(Gfn1Error::InvalidInput("non-finite nuclear cutoff".into()));
        }
        let keys = Self::capture(system, &cutoffs)?;
        let species = system.atoms.iter().map(|a| a.z).collect();
        let cell_key = system
            .lattice
            .unwrap()
            .cell
            .col
            .iter()
            .flat_map(|v| v.to_array().map(f64::to_bits))
            .collect();
        Ok(Self {
            cutoffs,
            keys,
            species,
            cell_key,
            periodic: system.lattice.unwrap().periodic,
        })
    }
    fn capture(
        system: &PeriodicSystem,
        cutoffs: &[(&'static str, f64)],
    ) -> Result<Vec<Vec<PairKey>>> {
        let largest = cutoffs.iter().map(|(_, r)| *r).fold(0., f64::max);
        let pairs = crate::pairlist::unique_short_range_pairs(system, largest)?;
        let mut keys = Vec::with_capacity(cutoffs.len());
        for (_, cutoff) in cutoffs {
            let mut set: Vec<_> = pairs
                .iter()
                .filter(|p| p.r2 <= cutoff * cutoff)
                .map(|p| (p.i, p.j, p.offset.n))
                .collect();
            set.sort_unstable();
            keys.push(set);
        }
        Ok(keys)
    }
    pub fn changed_terms(&self, system: &PeriodicSystem) -> Result<Vec<&'static str>> {
        let lattice = system.lattice.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput(
                "nuclear cutoff comparison requires the reference lattice".into(),
            )
        })?;
        let key: Vec<_> = lattice
            .cell
            .col
            .iter()
            .flat_map(|v| v.to_array().map(f64::to_bits))
            .collect();
        if key != self.cell_key
            || lattice.periodic != self.periodic
            || system.atoms.iter().map(|a| a.z).collect::<Vec<_>>() != self.species
        {
            return Err(Gfn1Error::InvalidInput(
                "nuclear cutoff comparison requires unchanged cell and species".into(),
            ));
        }
        let current = Self::capture(system, &self.cutoffs)?;
        Ok(self
            .cutoffs
            .iter()
            .zip(&self.keys)
            .zip(current)
            .filter(|((_, a), b)| **a != *b)
            .map(|(((label, _), _), _)| *label)
            .collect())
    }
    pub fn cutoffs(&self) -> Vec<(&'static str, f64)> {
        self.cutoffs.clone()
    }
}
