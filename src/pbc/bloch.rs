// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic zeroth-order Hamiltonian and overlap via Bloch sums.
//!
//! For each lattice image `T` the real-space blocks are
//!
//! ```text
//!   S_{mu nu}(T) = <mu 0 | nu T>
//!   H0_{mu nu}(T) = 0.5 (h_i + h_j) * K^Huckel_ij * Pi_ij(|R_AB + T|) * S_{mu nu}(T)
//! ```
//!
//! where `h_i = H_i (1 + k_CN CN_A)` are the CN-dependent self-energies
//! (periodic CN), `K^Huckel` is the shell-pair / EN scaling (`hscale`), and
//! `Pi_ij` is the distance polynomial (`shell_polynomial`). This matches the
//! non-periodic `build_h0_from_overlap`, but the polynomial and overlap are
//! evaluated per image because they depend on `|R_AB + T|` (SI Eq. 1-3, 30).
//!
//! The k-resolved matrices are the Bloch sums
//! `H(k) = sum_T H0(T) exp(i k . T)` and `S(k) = sum_T S(T) exp(i k . T)`.
//! Expensive integral evaluation happens once; each k-point only does a cheap
//! phase-weighted accumulation over the stored image blocks.

use crate::basis::BasisSet;
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::data_tables::atomic_radius_bohr;
use crate::error::Result;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::contracted_pair;
use crate::lattice::ImageOffset;
use crate::linalg::Matrix;
use crate::model::image_translations;
use crate::params::Gfn1Parameters;
use crate::pbc::complex::CMatrix;
use crate::pbc::kpoints::bloch_phase;
use crate::system::PeriodicSystem;
use rayon::prelude::*;

/// One stored AO-pair contribution at a fixed lattice image.
#[derive(Clone, Copy, Debug)]
pub struct ImagePair {
    pub mu: usize,
    pub nu: usize,
    pub offset: ImageOffset,
    /// Overlap `S_{mu nu}(T)`.
    pub s: f64,
    /// Zeroth-order Hamiltonian `H0_{mu nu}(T)`.
    pub h0: f64,
}

/// Precomputed Bloch building blocks for a periodic system.
#[derive(Clone, Debug)]
pub struct BlochBuilder {
    pub n: usize,
    pub pairs: Vec<ImagePair>,
    pub self_energies: Vec<f64>,
    pub dsedcn: Vec<f64>,
    pub coordination_numbers: Vec<f64>,
}

impl BlochBuilder {
    pub fn build(
        system: &PeriodicSystem,
        basis: &BasisSet,
        params: &Gfn1Parameters,
        ao_cutoff: f64,
        coordination_cutoff: f64,
        enable_cn: bool,
    ) -> Result<Self> {
        let nat = system.atoms.len();
        let cn = if enable_cn {
            coordination_with_derivatives(
                system,
                CoordinationOptions {
                    cutoff: coordination_cutoff,
                    ..CoordinationOptions::default()
                },
            )?
            .cn
        } else {
            vec![0.0; nat]
        };

        let nsh = basis.shells.len();
        let mut self_energies = vec![0.0; nsh];
        let mut dsedcn = vec![0.0; nsh];
        for (ish, shell) in basis.shells.iter().enumerate() {
            let kcn = if enable_cn {
                shell.kcn_raw.unwrap_or(0.0)
            } else {
                0.0
            };
            dsedcn[ish] = -kcn;
            self_energies[ish] = shell.hdiag_ha - kcn * cn[shell.atom_index];
        }

        // Group AO indices by atom for block construction.
        let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
        for (iao, ao) in basis.aos.iter().enumerate() {
            atom_aos[ao.atom_index].push(iao);
        }

        // Per-atom smallest (most diffuse) primitive exponent, for overlap
        // screening: the overlap of two contracted Gaussians decays no slower than
        // exp(-e_a e_b/(e_a+e_b) r^2), so atom pairs beyond that range contribute a
        // strictly negligible S/H0 and can be skipped before the integral.
        let mut atom_min_exp = vec![f64::INFINITY; nat];
        for ao in &basis.aos {
            for p in &ao.primitives {
                if p.exponent < atom_min_exp[ao.atom_index] {
                    atom_min_exp[ao.atom_index] = p.exponent;
                }
            }
        }

        let images = image_translations(system, ao_cutoff);
        let cutoff2 = if ao_cutoff > 0.0 {
            ao_cutoff * ao_cutoff
        } else {
            f64::INFINITY
        };

        // Each lattice image is independent, so build its AO-pair contributions in
        // parallel and concatenate in image order. Collecting per-image Vecs in
        // order (then flattening) keeps `pairs` bit-identical to the serial build,
        // so the Bloch sums accumulate in exactly the same order.
        let per_image: Vec<Vec<ImagePair>> = images
            .par_iter()
            .map(|(offset, translation)| -> Result<Vec<ImagePair>> {
                let is_origin = offset.is_origin();
                let mut local = Vec::new();
                for a in 0..nat {
                    let ra = system.atoms[a].position;
                    for b in 0..nat {
                        let rb = system.atoms[b].position + *translation;
                        let dr = ra - rb;
                        let r2 = dr.norm2();
                        let same_site = is_origin && a == b;
                        if !same_site {
                            if r2 > cutoff2 {
                                continue;
                            }
                            // Overlap screening (exp(-40) ~ 4e-18): r2 > 40 (ea+eb)/(ea eb).
                            let ea = atom_min_exp[a];
                            let eb = atom_min_exp[b];
                            if r2 * ea * eb > 40.0 * (ea + eb) {
                                continue;
                            }
                        }
                        let r = r2.sqrt();
                        // Off-site polynomial / scaling prefactor (per image distance).
                        let rad_sum = atomic_radius_bohr(system.atoms[a].z)?
                            + atomic_radius_bohr(system.atoms[b].z)?;
                        let rr = if rad_sum > 0.0 {
                            (r / rad_sum).sqrt()
                        } else {
                            0.0
                        };
                        for &mu in &atom_aos[a] {
                            let shell_mu_index = basis.aos[mu].shell_index;
                            let shell_mu = &basis.shells[shell_mu_index];
                            for &nu in &atom_aos[b] {
                                let shell_nu_index = basis.aos[nu].shell_index;
                                let shell_nu = &basis.shells[shell_nu_index];
                                let overlap =
                                    contracted_pair(&basis.aos[mu], &basis.aos[nu], ra, rb).0;
                                if overlap == 0.0 {
                                    continue;
                                }
                                let hij = if same_site {
                                    0.5 * (self_energies[shell_mu_index]
                                        + self_energies[shell_nu_index])
                                } else {
                                    0.5 * (self_energies[shell_mu_index]
                                        + self_energies[shell_nu_index])
                                        * hscale(shell_mu, shell_nu, params)?
                                        * shell_polynomial(shell_mu, shell_nu, rr)
                                };
                                local.push(ImagePair {
                                    mu,
                                    nu,
                                    offset: *offset,
                                    s: overlap,
                                    h0: overlap * hij,
                                });
                            }
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Result<Vec<_>>>()?;
        let pairs: Vec<ImagePair> = per_image.into_iter().flatten().collect();

        Ok(Self {
            n: basis.len(),
            pairs,
            self_energies,
            dsedcn,
            coordination_numbers: cn,
        })
    }

    /// Bloch sums `H(k)` and `S(k)` for a fractional k-point.
    pub fn h_s_at_k(&self, fractional: [f64; 3]) -> (CMatrix, CMatrix) {
        let mut h = CMatrix::zeros(self.n);
        let mut s = CMatrix::zeros(self.n);
        for p in &self.pairs {
            let (c, sn) = bloch_phase(fractional, p.offset);
            h.accumulate(p.mu, p.nu, p.h0 * c, p.h0 * sn);
            s.accumulate(p.mu, p.nu, p.s * c, p.s * sn);
        }
        h.hermitianize();
        s.hermitianize();
        (h, s)
    }

    /// Real folded `H` and `S` at the Gamma point (`k = 0`), where every phase is
    /// unity and the matrices are real symmetric.
    pub fn h_s_gamma_real(&self) -> (Matrix, Matrix) {
        let (h, s) = self.h_s_at_k([0.0, 0.0, 0.0]);
        (h.re, s.re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSet};

    fn load_params() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    // A large cubic cell holding a single H2 molecule must reproduce the
    // molecular Gamma-point H/S of the isolated molecule (images negligible).
    #[test]
    fn gamma_large_cell_matches_molecular() {
        let Some(params) = load_params() else {
            return;
        };
        let mol = PeriodicSystem::from_xyz_str("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0.0, false).unwrap();
        let cell = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"30 0 0 0 30 0 0 0 30\" pbc=\"T T T\"\nH 0 0 0\nH 0 0 0.74\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();

        let builder = BlochBuilder::build(&cell, &basis, &params, 30.0, 30.0, true).unwrap();
        let (h_pbc, s_pbc) = builder.h_s_gamma_real();

        let core = crate::hamiltonian::build_h0(
            &mol,
            &basis,
            &params,
            &crate::hamiltonian::HamiltonianOptions::default(),
        )
        .unwrap();

        let mut max_h = 0.0_f64;
        let mut max_s = 0.0_f64;
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                max_h = max_h.max((h_pbc[(i, j)] - core.h0[(i, j)]).abs());
                max_s = max_s.max((s_pbc[(i, j)] - core.integrals.overlap[(i, j)]).abs());
            }
        }
        assert!(max_h < 1.0e-8, "max H diff {max_h:.3e}");
        assert!(max_s < 1.0e-8, "max S diff {max_s:.3e}");
    }
}
