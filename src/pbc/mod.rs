// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic boundary conditions for GFN1-xTB: Gamma-point and k-point paths.
//!
//! Layout (all new code is isolated here to avoid colliding with the non-PBC
//! modules):
//!
//! - [`complex`]: complex Hermitian generalized eigensolver via a real embedding.
//! - [`kpoints`]: Monkhorst-Pack meshes and the Bloch phase factor.
//! - `bloch`: per-image `H0(T)`/`S(T)` and their Bloch sums `H(k)`/`S(k)`.
//! - `ewald`: generalized Ewald for the Klopman-Ohno second-order electrostatics.
//! - `scf`: k-point self-consistent charge driver and energy assembly.
//! - `gradient`: analytic Cartesian gradients (band + Pulay + electrostatics).
//! - `hessian`: Gamma-point Cartesian Hessian assembly, including periodic
//!   repulsion, D3(BJ), and halogen-bond classical blocks.
//!
//! Reference: Buccheri, Li, Deustua, Moosavi, Bygrave, Manby,
//! "Periodic GFN1-xTB Tight-Binding: A Generalised Ewald Partitioning Scheme for
//! the Klopman-Ohno Function", J. Chem. Theory Comput. 2025 (and its SI), which
//! provides the periodic H0 terms (SI Eq. 1-4), the Ewald partitioning of the
//! KO gamma-potential (SI Eq. 6-10), and the gradient derivation (SI Eq. 23-48).

pub mod bloch;
pub mod complex;
pub mod ewald;
pub mod ewald_multipole;
pub mod gradient;
pub mod hessian;
pub mod kpoints;
pub mod scf;
pub mod stress;

use crate::electronic::{ElectronicOptions, ElectronicResult};
use crate::error::Result;
use crate::model::BoundaryCondition;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

pub use gradient::{pbc_analytic_gradient, pbc_gradient_from_scc, PbcGradientResult};
pub use hessian::{pbc_gamma_hessian, pbc_kpoint_hessian, PbcHessianResult};
pub use scf::{
    pbc_electronic_result, run_pbc_scc, run_pbc_scc_with_guess, PbcSccGuess, PbcSccResult,
};
pub use stress::{pbc_stress, pbc_stress_from_scc, PbcStressResult};

/// Run a periodic GFN1-xTB single point and project it into the molecular-shaped
/// [`ElectronicResult`]. The k-mesh follows the boundary condition: Gamma-only
/// for [`BoundaryCondition::GammaPointPbc`] (and for a bare lattice), a default
/// Monkhorst-Pack mesh for [`BoundaryCondition::KPointPbc`].
pub fn run_electronic_pbc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> Result<ElectronicResult> {
    let pbc = PbcOptions::for_boundary(options.boundary);
    let scf = run_pbc_scc(system, params, options, &pbc)?;
    pbc_electronic_result(scf, system, pbc.ao_cutoff)
}

/// Default real-space cutoff (Bohr) for translation-vector sums of the
/// short-range AO and classical terms. Matches the tblite periodic default.
pub const DEFAULT_REALSPACE_CUTOFF: f64 = 40.0;

/// k-mesh / sampling selection for a periodic calculation.
#[derive(Clone, Copy, Debug)]
pub struct KMesh {
    pub size: [usize; 3],
    pub gamma_centered: bool,
    pub fold_time_reversal: bool,
}

impl KMesh {
    pub const fn gamma() -> Self {
        Self {
            size: [1, 1, 1],
            gamma_centered: true,
            fold_time_reversal: false,
        }
    }

    pub const fn monkhorst_pack(size: [usize; 3]) -> Self {
        Self {
            size,
            gamma_centered: false,
            fold_time_reversal: true,
        }
    }

    pub fn is_gamma_only(&self) -> bool {
        self.size == [1, 1, 1]
    }
}

impl Default for KMesh {
    fn default() -> Self {
        Self::gamma()
    }
}

/// Numerical controls for the generalized Ewald summation of the KO potential.
#[derive(Clone, Copy, Debug)]
pub struct EwaldOptions {
    /// Ewald Gaussian splitting parameter `alpha`. In the Buccheri et al.
    /// notation `alpha = sqrt(pi) K`. When `None`, a volume-based default is
    /// chosen so real and reciprocal sums are balanced.
    pub k_split: Option<f64>,
    /// Real-space cutoff (Bohr) for the screened lattice sum.
    pub real_cutoff: f64,
    /// Reciprocal-space cutoff (Bohr^-1) for the structure-factor sum.
    pub recip_cutoff: f64,
    /// Real-space cutoff (Bohr) for the rapidly decaying QCore KO residual.
    pub sr_cutoff: f64,
    /// Legacy smooth-truncation width retained for API compatibility; QCore
    /// residual sums do not use it.
    pub sr_width: f64,
}

impl Default for EwaldOptions {
    fn default() -> Self {
        Self {
            k_split: None,
            real_cutoff: DEFAULT_REALSPACE_CUTOFF,
            recip_cutoff: 0.0, // resolved from k_split at build time
            sr_cutoff: 10.0,
            sr_width: 1.0,
        }
    }
}

/// Options controlling a periodic GFN1-xTB calculation.
#[derive(Clone, Copy, Debug)]
pub struct PbcOptions {
    pub kmesh: KMesh,
    pub ewald: EwaldOptions,
    /// AO image-sum cutoff (Bohr) for the H0/overlap Bloch sums.
    pub ao_cutoff: f64,
}

impl Default for PbcOptions {
    fn default() -> Self {
        Self {
            kmesh: KMesh::default(),
            ewald: EwaldOptions::default(),
            ao_cutoff: 30.0,
        }
    }
}

impl PbcOptions {
    /// Pick the PBC options implied by a [`BoundaryCondition`] when the caller
    /// did not specify an explicit mesh.
    pub fn for_boundary(boundary: BoundaryCondition) -> Self {
        match boundary {
            BoundaryCondition::GammaPointPbc | BoundaryCondition::NonPeriodic => Self::default(),
            BoundaryCondition::KPointPbc => Self {
                kmesh: KMesh::monkhorst_pack([4, 4, 4]),
                ..Self::default()
            },
        }
    }
}
