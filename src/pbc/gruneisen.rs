// SPDX-License-Identifier: GPL-3.0-or-later
//! Mode and thermodynamic Grueneisen parameters of one periodic cell (`q = 0`),
//! over the Gamma-point or the k-point analytic Hessian.
//!
//! The mode Grueneisen parameter measures how a phonon stiffens under
//! compression,
//!
//! ```text
//!     gamma_i = - d ln omega_i / d ln V
//! ```
//!
//! and is the microscopic origin of thermal expansion: the Grueneisen relation
//! `alpha_V = gamma_th * C_V * kappa_T / V` ties the mode-averaged
//! `gamma_th(T)` to the volumetric thermal expansion coefficient.
//!
//! # What this module computes
//!
//! Three analytic PBC Hessians are evaluated — at the reference volume `V0` and
//! at `V0 (1 +/- delta)` — each mass-weighted and diagonalised. They are
//! Gamma-point Hessians by default, or k-point Hessians on
//! [`GruneisenOptions::pbc`]'s mesh with [`GruneisenOptions::kpoint`].
//! The two strained mode sets are then matched back onto the reference set by
//! maximum eigenvector overlap, and
//!
//! ```text
//!     gamma_i = - (ln omega_i(V+) - ln omega_i(V-)) / (ln V+ - ln V-)
//!             = - (ln lambda_i(V+) - ln lambda_i(V-)) / (2 * ln((1+delta)/(1-delta)))
//! ```
//!
//! with `lambda` the mass-weighted Hessian eigenvalue (`omega = sqrt(lambda)`).
//! The exact log-volume separation is used rather than the leading-order
//! `2 delta`, so the estimator is a genuine `O(delta^2)` central difference.
//!
//! # Conventions and limitations
//!
//! * **Frozen-ion (fixed fractional coordinates).** The strained cells scale the
//!   lattice vectors isotropically and carry the atoms along with their
//!   fractional coordinates frozen; no internal-coordinate relaxation is done.
//!   This is the standard convention for mode Grueneisen parameters and is exact
//!   for structures whose internal coordinates are fixed by symmetry (diamond,
//!   silicon, fcc/bcc metals — every Wyckoff position with no free parameter).
//!   *Future work (not implemented in v0.5.0):* the **relaxed-ion** variant,
//!   which re-optimises the internal coordinates at each strained volume and
//!   adds the `-H^-1 * dF/d(ln V)` internal-strain coupling. It matters for
//!   structures with free internal parameters (wurtzite `u`, molecular crystals)
//!   and requires coupling the geometry optimiser into this loop.
//! * **Phonons at `q = 0` only.** The three acoustic branches are the
//!   `omega -> 0` modes of the cell and carry no meaningful `gamma`; they are
//!   excluded from both the reported optical set and the thermodynamic average
//!   (see [`GruneisenOptions::acoustic_modes`]). A physically converged
//!   `gamma_th(T)` at low `T` needs a Brillouin-zone sum over acoustic
//!   *phonon* branches, which one cell's Hessian cannot provide;
//!   `gamma_th(300 K)` for a hard covalent solid such as diamond is nevertheless
//!   dominated by the optical branches.
//!
//!   [`GruneisenOptions::kpoint`] does **not** lift that restriction — it routes
//!   the three (or five) Hessians through [`pbc_kpoint_hessian`] so the
//!   *electronic* Brillouin-zone sum behind each one is converged on
//!   [`GruneisenOptions::pbc`]'s mesh. The nuclear displacement pattern is still
//!   one cell's worth of Cartesian DOFs, i.e. the `q = 0` dynamical matrix. A
//!   phonon-dispersion Grueneisen average needs a supercell (or a `q`-dependent
//!   response) and is separate future work.
//! * **Gated at `T = 0` only.** Everything quoted below is a gapped insulator
//!   with integer occupations. The periodic finite-temperature response is a
//!   direct dielectric solve as of v0.5.0 and no longer silently unconverged, but
//!   a smeared Grueneisen run still differences a *reconverged* Hessian at five
//!   volumes: tighten `charge_tolerance` / `energy_tolerance` first, and expect
//!   band reordering across the strain stencil to show up as mode-matching noise
//!   rather than as an error.
//!
//! # Thermodynamic average
//!
//! ```text
//!     gamma_th(T) = sum_i gamma_i c_i(T) / sum_i c_i(T)
//!     c_i(T)      = k_B x^2 e^x / (e^x - 1)^2 ,   x = hbar omega_i / (k_B T)
//! ```
//!
//! i.e. each mode is weighted by its Einstein heat capacity. The `k_B`
//! prefactor cancels in the ratio but is kept for dimensional clarity.
//!
//! # Second order: the curvature of `ln omega` in `ln V`
//!
//! With [`GruneisenOptions::second_order`] the same matched mode sets also yield
//! the **second-order mode Grueneisen parameter**
//!
//! ```text
//!     gamma2_i = d^2 ln omega_i / d(ln V)^2
//! ```
//!
//! **Sign convention (read this before comparing with anything).** `gamma2` is
//! the *plain* second log-derivative of the frequency; unlike `gamma` it does
//! **not** carry a leading minus sign. The two therefore relate as
//!
//! ```text
//!     gamma2_i = - d gamma_i / d ln V = q_i * gamma_i ,
//!     q_i      = - d ln gamma_i / d ln V = gamma2_i / gamma_i
//! ```
//!
//! with `q_i` the dimensionless volume dependence used by the Mie-Grueneisen
//! thermal equation-of-state literature (where `gamma(V) = gamma_0 (V/V_0)^q` is
//! the usual one-parameter model). [`GruneisenResult::mode_q`] returns `q_i`
//! directly. `gamma2 < 0` (equivalently `q < 0`) means `d gamma / d ln V > 0`:
//! `gamma` grows with volume, i.e. *falls* under compression. GFN1 diamond lands
//! there but barely — `gamma2 = -0.0383`, `q = -0.0423` against `gamma = 0.905`,
//! so the model's `gamma` is very nearly volume-independent (see the crate's
//! periodic docs for the numbers and the `delta` ladder behind them).
//!
//! The estimator is a polynomial fit of `ln lambda_i` (equivalently
//! `2 ln omega_i`) in `ln V` through the matched mode sets, with the
//! Fornberg finite-difference weights for the *actual*, non-uniformly spaced
//! nodes `ln(1 +/- delta)` (and `ln(1 +/- 2 delta)` for the five-point stencil).
//! Every strained mode set — including the outer pair — is matched onto the
//! **central** volume's modes, so a single reference ordering carries through the
//! whole stencil. Because the fit also produces the first derivative, the module
//! reports [`GruneisenResult::mode_gamma_refit`], which must reproduce
//! [`GruneisenResult::mode_gamma`] (an internal consistency check: two different
//! estimators of the same slope, differing only by the `O(delta^2)` node
//! asymmetry).
//!
//! # The two orders do not share a step
//!
//! `gamma2` is a *second* difference of `ln lambda(ln V)`, so a residual noise
//! `eps` in the phonon frequencies reaches it as `eps / delta^2`, where the
//! first-order `gamma` only suffers `eps / delta`. The optimal step is therefore
//! genuinely different for the two orders, and [`GruneisenOptions::delta_second`]
//! (default `2e-2`, against `delta = 5e-3`) carries the second-order node set.
//! Setting it equal to `delta` puts the second-order nodes back on the
//! first-order ones — free, because those Hessians exist anyway, but 16x more
//! exposed to noise.
//!
//! Two things had to be right for `gamma2` to be trustworthy at all:
//!
//! 1. **The real-space cutoffs travel with the strain.** They are radii in Bohr
//!    and every lattice sum runs over the integer images inside them, so a cell
//!    that breathes under a *fixed* radius crosses image shells at discrete
//!    volumes and `ln lambda(ln V)` acquires steps of `~5e-7`. Invisible in
//!    `gamma`; fatal once divided by `delta^2`. Scaling every cutoff by the
//!    node's linear factor `(V/V0)^(1/3)` freezes the integer image set across
//!    the whole stencil. Before that fix, the three- and five-point stencils
//!    returned `-0.0372` and `+0.0674` for the same diamond `gamma2` at the lean
//!    test cutoffs; after it, `-0.03719` and `-0.03746`.
//! 2. **The step is the measured optimum**, not an inherited one — see
//!    [`DEFAULT_GRUNEISEN_DELTA_SECOND`] for the ladder.
//!
//! With the second-order nodes at their own step, [`SecondOrderStencil::ThreePoint`]
//! costs two extra Hessians and [`SecondOrderStencil::FivePoint`] four; both are
//! free when `delta_second == delta` (two, for the five-point outer pair).
//!
//! # Second-order thermodynamic average
//!
//! Two conventions are reported, and they are *not* the same thing:
//!
//! ```text
//!     gamma2_th(T)      = sum_i gamma2_i c_i(T) / sum_i c_i(T)          (mode average)
//!     gamma2_th_full(T) = - d gamma_th(T, V) / d ln V
//!                       = gamma2_th(T) - sum_i w_i D_i (gamma_i - gamma_th(T))
//! ```
//!
//! with `w_i = c_i / sum_j c_j` and `D_i = d ln c_i / d ln V`. The first is the
//! heat-capacity-weighted mode average of `gamma2_i` — the direct analogue of
//! `gamma_th`. The second is the honest volume derivative of `gamma_th(T, V)`
//! itself: the *weights* also move with volume, because `c_i` depends on `V`
//! only through `omega_i(V)`, so
//!
//! ```text
//!     D_i = d ln c_i / d ln V = (d ln c / dx)(x_i) * dx_i/d ln V
//!         = - gamma_i * x_i * (d ln c / dx)(x_i) ,   x_i = hbar omega_i / (k_B T)
//!     (d ln c / dx)(x) = 2/x - coth(x/2)
//! ```
//!
//! The correction vanishes identically when every mode shares one `gamma`
//! (`gamma_i - gamma_th = 0`), which is exactly the case for diamond's degenerate
//! optical triplet — so on that system the two numbers agree to machine
//! precision, and the correction term is gated by a synthetic-model unit test
//! against a numerical `d gamma_th / d ln V` instead.
//!
//! Both averages inherit the Gamma-only caveat above: they are mode averages over
//! the optical branches of one cell, not Brillouin-zone integrals.

use crate::constants::KB_HARTREE_PER_K;
use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::linalg::Matrix;
use crate::params::Gfn1Parameters;
use crate::pbc::hessian::{pbc_gamma_hessian, pbc_kpoint_hessian};
use crate::pbc::third_derivative::scale_lattice_isotropic;
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use crate::vibrational::{vibrational_analysis, WAVENUMBER_PER_SQRT_AU};

/// Atomic mass unit expressed in electron masses, `m_u / m_e = 1822.888486209`
/// (CODATA 2018).
///
/// This is the one unit ratio the Grueneisen thermodynamics needs that
/// [`crate::constants`] does not already carry. Mass-weighted Hessian
/// eigenvalues come out in `Hartree / (Bohr^2 * amu)`; in atomic units
/// (`hbar = 1`) the mode energy in Hartree — the unit
/// [`KB_HARTREE_PER_K`] is expressed in — is `sqrt(lambda / (m_u/m_e))`.
/// It is *not* a second copy of any existing constant, and `gamma_i` itself does
/// not depend on it at all (it is a ratio of logarithms); only the temperature
/// weighting does.
const AMU_IN_ELECTRON_MASSES: f64 = 1822.888_486_209;

/// Node set for the `ln V` polynomial fit behind the second-order parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecondOrderStencil {
    /// `V(1 - d2), V0, V(1 + d2)` with `d2 = `[`GruneisenOptions::delta_second`]:
    /// two extra analytic Hessians, or none when `d2 == delta` (the first-order
    /// estimator already evaluates those two volumes). Truncation `O(d2^2)`,
    /// noise amplification `~ 4 / d2^2`.
    ThreePoint,
    /// Adds `V(1 - 2 d2)` and `V(1 + 2 d2)`: four extra analytic Hessians (two
    /// when `d2 == delta`) for `O(d2^4)` truncation. Requires `d2 < 0.5`.
    ///
    /// Its value is the stable one at large steps — on diamond it sits at
    /// `-0.03717` from `d2 = 4e-2` to `1e-1` while the three-point estimate walks
    /// away as `O(d2^2)` — so running both stencils is the honest convergence
    /// check on `gamma2`.
    FivePoint,
}

/// Controls for [`pbc_gruneisen`].
#[derive(Clone, Debug)]
pub struct GruneisenOptions {
    /// Relative **volumetric** strain of the two displaced cells, `V(1 +/- delta)`.
    /// The lattice vectors are scaled by `(1 +/- delta)^(1/3)`. Default `5e-3`.
    pub delta: f64,
    /// Temperatures (K) at which the heat-capacity-weighted `gamma_th(T)` is
    /// reported. Default `[300.0]`.
    pub temperatures: Vec<f64>,
    /// Electronic options for each of the three periodic Hessians. Keep
    /// `electronic_temperature = 0`.
    pub electronic: ElectronicOptions,
    /// Periodic options (Ewald, AO cutoff, and — with [`Self::kpoint`] — the
    /// k-mesh). Ignored for the mesh unless `kpoint` is set: the default Hessian
    /// path is Gamma-only regardless of `pbc.kmesh`.
    pub pbc: PbcOptions,
    /// Route the strained Hessians through the **k-point** analytic Hessian
    /// ([`pbc_kpoint_hessian`]) instead of the Gamma-point one, sampling the
    /// Brillouin zone on [`Self::pbc`]'s `kmesh`. Default `false`.
    ///
    /// This converges the *electronic* Brillouin-zone sum behind each Hessian;
    /// it does **not** turn the calculation into a phonon-dispersion Grueneisen
    /// average, which would need dynamical matrices at finite `q` (i.e. a
    /// supercell or a q-dependent response). The acoustic-branch caveat in the
    /// module docs therefore stands unchanged. Cost scales with the number of
    /// k-points; a `1 x 1 x 1` mesh reproduces the Gamma path.
    pub kpoint: bool,
    /// Number of lowest-frequency modes treated as acoustic and excluded from
    /// the optical set and the thermodynamic average. Default `3`.
    pub acoustic_modes: usize,
    /// Modes closer than this (cm^-1, at the reference volume) are treated as a
    /// degenerate subspace: their `gamma` is averaged over the subspace, which
    /// is invariant under the arbitrary rotation the eigensolver picks inside a
    /// degenerate block. Default `1.0`.
    pub degeneracy_tolerance_cm1: f64,
    /// Also compute the second-order parameters `gamma2_i = d^2 ln omega_i / d(ln V)^2`
    /// and their thermodynamic averages. Default `false`, in which case the
    /// second-order fields of [`GruneisenResult`] are `NaN` / empty and the run
    /// is bit-for-bit what it was before the option existed.
    pub second_order: bool,
    /// Which `ln V` nodes the second-order fit uses. Ignored unless
    /// [`Self::second_order`]. Default [`SecondOrderStencil::ThreePoint`];
    /// [`SecondOrderStencil::FivePoint`] costs two more Hessians.
    pub second_order_stencil: SecondOrderStencil,
    /// Volumetric strain for the **second-order** node set, independent of
    /// [`Self::delta`]. Ignored unless [`Self::second_order`]. Default
    /// `Some(`[`DEFAULT_GRUNEISEN_DELTA_SECOND`]`)` = `2e-2`; `None` reuses
    /// [`Self::delta`].
    ///
    /// `gamma2` is a *second* difference of `ln lambda(ln V)`, so phonon noise
    /// `eps` enters it as `eps / delta^2` where the first-order `gamma` only
    /// suffers `eps / delta`. The two orders therefore have genuinely different
    /// optimal steps, and one shared `delta` cannot serve both: at
    /// `delta = 5e-3` the three- and five-point stencils return `-0.0372` and
    /// `+0.0674` for the same diamond `gamma2` — opposite signs, a gap larger
    /// than the value — while the first-order `gamma` agrees to three digits.
    /// See the crate's periodic docs for the measured ladder.
    ///
    /// Splitting the step costs two extra Hessians (four with
    /// [`SecondOrderStencil::FivePoint`]), because the second-order nodes no
    /// longer coincide with the first-order ones. Setting this to
    /// `Some(options.delta)` — or to `None` — restores the historical
    /// zero-extra-cost behaviour together with its noise.
    pub delta_second: Option<f64>,
}

/// Default volumetric strain of the second-order node set, `2e-2` — the measured
/// minimum of the three- vs five-point disagreement on diamond.
///
/// The ladder (diamond, lean cutoffs, relative `|gamma2_3pt - gamma2_5pt|`):
///
/// ```text
///   delta_second   5e-3     1e-2     2e-2     3e-2     4e-2     6e-2     1e-1
///   rel gap      7.2e-3   6.6e-3   1.9e-3   2.7e-3   4.3e-3   9.2e-3   2.6e-2
/// ```
///
/// Left of the minimum the residual `eps / delta^2` noise amplification wins;
/// right of it the `O(delta^2)` truncation of the three-point stencil does (the
/// five-point value is flat at `-0.03717` from `4e-2` to `1e-1`). Reproduce with
/// `tests/gruneisen_second_order_delta.rs`.
pub const DEFAULT_GRUNEISEN_DELTA_SECOND: f64 = 2.0e-2;

impl Default for GruneisenOptions {
    fn default() -> Self {
        Self {
            delta: 5.0e-3,
            temperatures: vec![300.0],
            electronic: ElectronicOptions::default(),
            pbc: PbcOptions::default(),
            kpoint: false,
            acoustic_modes: 3,
            degeneracy_tolerance_cm1: 1.0,
            second_order: false,
            second_order_stencil: SecondOrderStencil::ThreePoint,
            delta_second: Some(DEFAULT_GRUNEISEN_DELTA_SECOND),
        }
    }
}

/// Output of [`pbc_gruneisen`].
#[derive(Clone, Debug)]
pub struct GruneisenResult {
    /// Cell volume (Bohr^3) at the reference geometry.
    pub volume: f64,
    /// The volumetric strain actually used by the first-order estimator.
    pub delta: f64,
    /// The volumetric strain actually used by the second-order node set, or
    /// `None` when [`GruneisenOptions::second_order`] was off. Equal to
    /// [`Self::delta`] only when the caller asked for that explicitly; by
    /// default it is the wider [`DEFAULT_GRUNEISEN_DELTA_SECOND`].
    pub delta_second: Option<f64>,
    /// Harmonic wavenumbers (cm^-1) at the reference volume, ascending.
    /// Imaginary modes are reported negative, as in [`vibrational_analysis`].
    pub frequencies_cm1: Vec<f64>,
    /// Wavenumbers at `V(1 + delta)`, permuted onto the reference mode ordering.
    pub frequencies_cm1_expanded: Vec<f64>,
    /// Wavenumbers at `V(1 - delta)`, permuted onto the reference mode ordering.
    pub frequencies_cm1_compressed: Vec<f64>,
    /// Mode Grueneisen parameters, one per reference mode. The first
    /// [`GruneisenOptions::acoustic_modes`] entries are `NaN` (excluded acoustic
    /// branches); degenerate subspaces share their subspace-averaged value.
    pub mode_gamma: Vec<f64>,
    /// Second-order mode Grueneisen parameters
    /// `gamma2_i = d^2 ln omega_i / d(ln V)^2 = - d gamma_i / d ln V`, one per
    /// reference mode, or all-`NaN` when [`GruneisenOptions::second_order`] is
    /// off. Acoustic and unusable modes are `NaN`; degenerate subspaces share
    /// their subspace-averaged value. See the module docs for the sign
    /// convention and the relation to the literature `q = gamma2 / gamma`.
    pub mode_gamma2: Vec<f64>,
    /// First-order `gamma_i` re-derived from the *same* multi-volume polynomial
    /// fit that produces [`Self::mode_gamma2`], i.e. `-(1/2) f'(0)` of the fitted
    /// `ln lambda_i(ln V)`. All-`NaN` unless [`GruneisenOptions::second_order`].
    ///
    /// This is the internal consistency check on the whole second-order path: it
    /// must reproduce [`Self::mode_gamma`], which is built by an independent
    /// two-point central difference. The two differ only by the `O(delta^2)`
    /// asymmetry of the nodes `ln(1 + delta)` and `-ln(1 - delta)` — for the
    /// three-point stencil the gap is `-(1/2) gamma2 ln(1 - delta^2)` to leading
    /// order, i.e. `(q/2) delta^2` relative to `gamma` itself (`~5e-7` on
    /// diamond at the default `delta`).
    pub mode_gamma_refit: Vec<f64>,
    /// `(T, gamma_th(T))` for every requested temperature.
    pub thermodynamic_gamma: Vec<(f64, f64)>,
    /// `(T, gamma2_th(T))` — the heat-capacity-weighted **mode average** of
    /// `gamma2_i`, `sum_i gamma2_i c_i / sum_i c_i`. Empty unless
    /// [`GruneisenOptions::second_order`].
    pub thermodynamic_gamma2: Vec<(f64, f64)>,
    /// `(T, -d gamma_th(T, V) / d ln V)` — the *full* volume derivative of the
    /// thermodynamic average, which adds the `d c_i / d ln V` reweighting term to
    /// [`Self::thermodynamic_gamma2`] (see the module docs). Empty unless
    /// [`GruneisenOptions::second_order`]. It coincides with the mode average
    /// whenever all modes share one `gamma`.
    pub thermodynamic_gamma2_full: Vec<(f64, f64)>,
    /// The stencil actually used for the second-order fit, or `None` when
    /// [`GruneisenOptions::second_order`] was off.
    pub second_order_stencil: Option<SecondOrderStencil>,
    /// Per-reference-mode matching quality, worst over every strained volume
    /// actually evaluated (two, or four with the five-point second-order
    /// stencil): the norm of the projection of the reference eigenvector onto the **span**
    /// of the strained eigenvectors assigned to its degenerate subspace,
    /// `sqrt(sum_j <u0_i, u_j>^2)`.
    ///
    /// The projection is taken onto the subspace rather than onto the single
    /// assigned partner because inside a degenerate block the eigensolver picks
    /// an arbitrary basis, so a *per-mode* overlap is meaningless there (it can
    /// be anything down to `1/sqrt(len)`) while the subspace projection is
    /// invariant. Values near 1 mean a clean assignment; a value well below 1
    /// flags a genuine mode crossing or a subspace that split under strain.
    pub match_overlaps: Vec<f64>,
    /// The number of leading modes excluded as acoustic.
    pub acoustic_modes: usize,
    /// Index ranges (start, len) of the degenerate subspaces used for averaging.
    pub degenerate_groups: Vec<(usize, usize)>,
}

impl GruneisenResult {
    /// The smallest matching overlap over the optical (non-acoustic) modes — the
    /// single number to gate mode-assignment quality on.
    pub fn min_optical_overlap(&self) -> f64 {
        self.match_overlaps
            .iter()
            .skip(self.acoustic_modes)
            .fold(f64::INFINITY, |m, &v| m.min(v))
    }

    /// `gamma_th` at a requested temperature, if it was computed.
    pub fn gamma_at(&self, temperature: f64) -> Option<f64> {
        lookup(&self.thermodynamic_gamma, temperature)
    }

    /// The mode-average `gamma2_th` at a requested temperature, if it was
    /// computed (needs [`GruneisenOptions::second_order`]).
    pub fn gamma2_at(&self, temperature: f64) -> Option<f64> {
        lookup(&self.thermodynamic_gamma2, temperature)
    }

    /// The full `-d gamma_th / d ln V` at a requested temperature, if it was
    /// computed (needs [`GruneisenOptions::second_order`]).
    pub fn gamma2_full_at(&self, temperature: f64) -> Option<f64> {
        lookup(&self.thermodynamic_gamma2_full, temperature)
    }

    /// The literature volume exponent `q_i = - d ln gamma_i / d ln V = gamma2_i / gamma_i`
    /// per mode (`NaN` where either input is `NaN`, or where `gamma_i` is too
    /// close to zero for the ratio to mean anything).
    pub fn mode_q(&self) -> Vec<f64> {
        self.mode_gamma
            .iter()
            .zip(self.mode_gamma2.iter())
            .map(|(&g, &g2)| if g.abs() > 1.0e-8 { g2 / g } else { f64::NAN })
            .collect()
    }
}

fn lookup(table: &[(f64, f64)], temperature: f64) -> Option<f64> {
    table
        .iter()
        .find(|(t, _)| (t - temperature).abs() < 1.0e-9)
        .map(|(_, g)| *g)
}

/// A mass-weighted normal-mode set: eigenvalues (Hartree/(Bohr^2 amu), ascending)
/// and the corresponding orthonormal mass-weighted eigenvectors as columns.
struct ModeSet {
    values: Vec<f64>,
    vectors: Matrix,
}

fn mode_set(hessian: &Matrix, atomic_numbers: &[u8]) -> Result<ModeSet> {
    let vib = vibrational_analysis(hessian, atomic_numbers)?;
    let ndof = 3 * atomic_numbers.len();
    // `vibrational_analysis` returns Cartesian displacements (eigenvector / sqrt(m));
    // undo the division to recover the orthonormal mass-weighted eigenvectors.
    let mass: Vec<f64> = (0..ndof)
        .map(|i| crate::data_tables::relative_atomic_mass(atomic_numbers[i / 3]).max(1.0e-12))
        .collect();
    let mut vectors = Matrix::zeros(ndof, ndof);
    for k in 0..ndof {
        for i in 0..ndof {
            vectors[(i, k)] = vib.modes[k][i] * mass[i].sqrt();
        }
    }
    Ok(ModeSet {
        values: vib.eigenvalues,
        vectors,
    })
}

/// Greedy maximum-overlap assignment of `other`'s modes onto `reference`'s.
///
/// The full `|<u_ref_i, u_other_j>|` matrix is built, all `n^2` pairs are sorted
/// by decreasing magnitude, and pairs are consumed greedily, skipping any whose
/// reference or partner index is already taken. This is the standard cheap
/// stand-in for the optimal (Hungarian) assignment; because the overlap matrix
/// of two nearly identical orthonormal bases is nearly a permutation matrix, the
/// greedy and optimal assignments coincide except inside degenerate blocks,
/// where the assignment is arbitrary anyway and the subspace average below makes
/// the result assignment-independent.
///
/// Returns `(perm, abs_overlap)` with `perm[i]` the `other`-index assigned to
/// reference mode `i` and `abs_overlap[(i, j)] = |<u_ref_i, u_other_j>|`.
fn match_modes(reference: &ModeSet, other: &ModeSet, ndof: usize) -> (Vec<usize>, Matrix) {
    let mut abs_overlap = Matrix::zeros(ndof, ndof);
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(ndof * ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let mut dot = 0.0;
            for r in 0..ndof {
                dot += reference.vectors[(r, i)] * other.vectors[(r, j)];
            }
            abs_overlap[(i, j)] = dot.abs();
            pairs.push((dot.abs(), i, j));
        }
    }
    // Deterministic order: magnitude descending, then index order.
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut perm = vec![usize::MAX; ndof];
    let mut taken = vec![false; ndof];
    let mut assigned = 0;
    for &(_, i, j) in &pairs {
        if assigned == ndof {
            break;
        }
        if perm[i] != usize::MAX || taken[j] {
            continue;
        }
        perm[i] = j;
        taken[j] = true;
        assigned += 1;
    }
    (perm, abs_overlap)
}

/// Assignment quality that is invariant under the arbitrary basis choice inside
/// a degenerate block: for each reference mode, the norm of its projection onto
/// the span of the strained modes assigned to its whole degenerate subspace.
fn subspace_projection_quality(
    abs_overlap: &Matrix,
    perm: &[usize],
    groups: &[(usize, usize)],
) -> Vec<f64> {
    let ndof = perm.len();
    let mut quality = vec![0.0; ndof];
    for &(start, len) in groups {
        for i in start..start + len {
            let mut s2 = 0.0;
            for k in start..start + len {
                let o = abs_overlap[(i, perm[k])];
                s2 += o * o;
            }
            quality[i] = s2.sqrt().min(1.0);
        }
    }
    quality
}

/// Group consecutive reference modes whose wavenumbers differ by less than
/// `tol_cm1` into degenerate subspaces. Returns `(start, len)` ranges covering
/// `0..ndof`.
fn degenerate_groups(frequencies_cm1: &[f64], tol_cm1: f64) -> Vec<(usize, usize)> {
    let n = frequencies_cm1.len();
    let mut groups = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || (frequencies_cm1[i] - frequencies_cm1[i - 1]).abs() > tol_cm1 {
            groups.push((start, i - start));
            start = i;
        }
    }
    groups
}

/// Finite-difference weights on **arbitrary (non-uniform) nodes**, by Fornberg's
/// recursion (B. Fornberg, *Math. Comp.* **51** (1988) 699).
///
/// `weights[k][n]` is the coefficient of `f(nodes[n])` in the approximation of
/// `f^(k)(z)`, for `k = 0..=max_order`. The weights are those of the interpolating
/// polynomial through *all* the nodes, so `nodes.len() - 1` is the polynomial
/// degree and the truncation order follows from the node spacing.
///
/// This is needed here rather than a hard-coded stencil because the natural
/// volume nodes are `ln(1 +/- delta)` (and `ln(1 +/- 2 delta)`), which are **not**
/// equally spaced in `ln V`: `ln(1 + delta) + ln(1 - delta) = ln(1 - delta^2)`,
/// i.e. the two arms differ by `~delta^2`. That is fatal for a second difference,
/// not cosmetic: feeding those nodes to the textbook `(f+ - 2 f0 + f-) / h^2`
/// leaks `f'(a - b)/h^2 ~ -f'`, the **first** derivative at full size. On diamond
/// that spurious term is `+1.81` against a true `f'' = 2 gamma2 = -0.077`, so the
/// naive stencil would report `gamma2 ~ +0.87` (`q ~ 0.96`) — a plausible-looking
/// number that is pure artefact, and one a `delta` vs `delta/2` study would never
/// catch, because it is `delta`-independent.
///
/// (The alternative fix is to place the volumes symmetrically in `ln V`,
/// `V0 exp(+/- s)`. The nodes are instead inherited from the first-order
/// estimator, which is defined on `V(1 +/- delta)`, so that the second-order path
/// costs no extra Hessian and both orders describe the same three cells.)
fn fd_weights(nodes: &[f64], z: f64, max_order: usize) -> Vec<Vec<f64>> {
    let n = nodes.len();
    // c[node][order]
    let mut c = vec![vec![0.0; max_order + 1]; n];
    if n == 0 {
        return vec![Vec::new(); max_order + 1];
    }
    c[0][0] = 1.0;
    let mut c1 = 1.0;
    let mut c4 = nodes[0] - z;
    for i in 1..n {
        let mn = i.min(max_order);
        let mut c2 = 1.0;
        let c5 = c4;
        c4 = nodes[i] - z;
        for j in 0..i {
            let c3 = nodes[i] - nodes[j];
            c2 *= c3;
            if j == i - 1 {
                for k in (1..=mn).rev() {
                    c[i][k] = c1 * (k as f64 * c[i - 1][k - 1] - c5 * c[i - 1][k]) / c2;
                }
                c[i][0] = -c1 * c5 * c[i - 1][0] / c2;
            }
            for k in (1..=mn).rev() {
                c[j][k] = (c4 * c[j][k] - k as f64 * c[j][k - 1]) / c3;
            }
            c[j][0] *= c4 / c3;
        }
        c1 = c2;
    }
    (0..=max_order)
        .map(|k| (0..n).map(|i| c[i][k]).collect())
        .collect()
}

/// Einstein single-mode heat capacity `c(T) = k_B x^2 e^x / (e^x - 1)^2` with
/// `x = hbar omega / (k_B T)`, written as `k_B (x / (2 sinh(x/2)))^2` for
/// numerical stability. Zero for non-positive `omega` or `T`.
fn einstein_heat_capacity(omega_hartree: f64, temperature: f64) -> f64 {
    if !(omega_hartree > 0.0) || !(temperature > 0.0) {
        return 0.0;
    }
    let x = omega_hartree / (KB_HARTREE_PER_K * temperature);
    if x > 600.0 {
        return 0.0; // frozen out; e^x would overflow
    }
    if x < 1.0e-8 {
        return KB_HARTREE_PER_K; // classical (Dulong-Petit) limit
    }
    let s = (0.5 * x).sinh();
    KB_HARTREE_PER_K * (x * x) / (4.0 * s * s)
}

/// Logarithmic derivative `d ln c / dx = 2/x - coth(x/2)` of the Einstein heat
/// capacity with respect to `x = hbar omega / (k_B T)`.
///
/// This is the only extra ingredient the *full* `d gamma_th / d ln V` needs
/// beyond the mode parameters, because `c_i` depends on the volume solely through
/// `omega_i(V)`. Both terms of the closed form diverge as `2/x` when `x -> 0` and
/// cancel to `-x/6`, so the series `-x/6 + x^3/360` (from
/// `ln(c/k_B) = -x^2/12 + x^4/1440 - ...`) is used below `x = 1e-3`.
fn einstein_heat_capacity_log_derivative(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    if x < 1.0e-3 {
        return -x / 6.0 + x * x * x / 360.0;
    }
    if x > 600.0 {
        return 2.0 / x - 1.0; // 2/(e^x - 1) has underflowed
    }
    2.0 / x - 1.0 - 2.0 / (x.exp() - 1.0)
}

/// Heat-capacity-weighted first- and second-order thermodynamic averages at one
/// temperature, over the modes for which every input is finite.
///
/// Returns `(gamma_th, gamma2_th_mode_average, gamma2_th_full)` where
///
/// ```text
///     gamma_th   = sum w_i gamma_i ,                w_i = c_i / sum_j c_j
///     gamma2_th  = sum w_i gamma2_i
///     gamma2_full = gamma2_th - sum_i w_i D_i (gamma_i - gamma_th)
///     D_i        = d ln c_i / d ln V = - gamma_i x_i (d ln c / dx)(x_i)
/// ```
///
/// `gamma2_full` is `-d gamma_th(T, V) / d ln V`: differentiating the weighted
/// mean gives the `sum w gamma'` term (that is `-gamma2_th`) plus the reweighting
/// term `sum w (c'/c)(gamma - gamma_th)`, and the leading minus sign of the
/// `gamma` convention flips both.
///
/// `omega_hartree` are the reference-volume mode energies `hbar omega_i` in
/// Hartree; a non-positive entry, or a non-finite `gamma`/`gamma2`, drops the
/// mode from all three averages.
fn thermodynamic_second_order(
    omega_hartree: &[f64],
    gamma: &[f64],
    gamma2: &[f64],
    temperature: f64,
) -> (f64, f64, f64) {
    let mut den = 0.0;
    let mut num_g = 0.0;
    let mut num_g2 = 0.0;
    for i in 0..omega_hartree.len() {
        if !gamma[i].is_finite() || !gamma2[i].is_finite() {
            continue;
        }
        let c = einstein_heat_capacity(omega_hartree[i], temperature);
        den += c;
        num_g += gamma[i] * c;
        num_g2 += gamma2[i] * c;
    }
    if !(den > 0.0) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let gamma_th = num_g / den;
    let gamma2_th = num_g2 / den;
    let mut correction = 0.0;
    for i in 0..omega_hartree.len() {
        if !gamma[i].is_finite() || !gamma2[i].is_finite() {
            continue;
        }
        let c = einstein_heat_capacity(omega_hartree[i], temperature);
        if c <= 0.0 {
            continue;
        }
        let x = omega_hartree[i] / (KB_HARTREE_PER_K * temperature);
        let dlnc_dlnv = -gamma[i] * x * einstein_heat_capacity_log_derivative(x);
        correction += (c / den) * dlnc_dlnv * (gamma[i] - gamma_th);
    }
    (gamma_th, gamma2_th, gamma2_th - correction)
}

/// Mode energy `hbar omega` in Hartree from a mass-weighted Hessian eigenvalue
/// in `Hartree / (Bohr^2 amu)`.
fn mode_energy_hartree(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    (lambda / AMU_IN_ELECTRON_MASSES).sqrt()
}

/// The matched `ln lambda` of one strained volume, indexed by **reference** mode:
/// `out[i] = ln lambda(perm[i])`, or `NaN` where the strained eigenvalue is not
/// positive (an imaginary mode carries no `ln omega`).
fn matched_ln_lambda(values: &[f64], perm: &[usize], ndof: usize) -> Vec<f64> {
    (0..ndof)
        .map(|i| {
            let l = values[perm[i]];
            if l > 0.0 {
                l.ln()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Mode and thermodynamic Grueneisen parameters of a periodic system at `q = 0`,
/// from three analytic PBC Hessians (`V0`, `V0(1+delta)`,
/// `V0(1-delta)`) under isotropic frozen-ion volumetric strain — Gamma-point
/// Hessians by default, k-point Hessians with [`GruneisenOptions::kpoint`] —
/// plus, with
/// [`GruneisenOptions::second_order`], the curvature
/// `gamma2_i = d^2 ln omega_i / d(ln V)^2` from the same three volumes (or five,
/// with [`SecondOrderStencil::FivePoint`]).
///
/// See the module documentation for the conventions, the sign convention of
/// `gamma2` and its relation to the literature `q`, the acoustic-mode
/// exclusion, the degenerate-subspace averaging and the relaxed-ion variant left
/// as future work.
pub fn pbc_gruneisen(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &GruneisenOptions,
) -> Result<GruneisenResult> {
    let lattice = system.lattice.ok_or_else(|| {
        Gfn1Error::InvalidInput("pbc_gruneisen: the system has no lattice (not periodic)".into())
    })?;
    let delta = options.delta;
    if !(delta.is_finite() && delta > 0.0 && delta < 1.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_gruneisen: delta must be finite and in (0, 1) (got {delta})"
        )));
    }
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    if options.acoustic_modes > ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_gruneisen: acoustic_modes {} exceeds the {ndof} degrees of freedom",
            options.acoustic_modes
        )));
    }
    let stencil = if options.second_order {
        Some(options.second_order_stencil)
    } else {
        None
    };
    // The second-order node set has its own step (see `delta_second`): `gamma2`
    // is a second difference, so it needs a wider stencil than `gamma`.
    let delta_second = if stencil.is_some() {
        options.delta_second.unwrap_or(delta)
    } else {
        delta
    };
    if !(delta_second.is_finite() && delta_second > 0.0 && delta_second < 1.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_gruneisen: delta_second must be finite and in (0, 1) (got {delta_second})"
        )));
    }
    if stencil == Some(SecondOrderStencil::FivePoint) && delta_second >= 0.5 {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_gruneisen: the five-point stencil needs 2 * delta_second < 1 \
             (got delta_second = {delta_second})"
        )));
    }
    let atomic_numbers: Vec<u8> = system.atoms.iter().map(|a| a.z).collect();

    let third = 1.0 / 3.0;
    let expanded = scale_lattice_isotropic(system, (1.0 + delta).powf(third))?;
    let compressed = scale_lattice_isotropic(system, (1.0 - delta).powf(third))?;

    // One evaluator for every volume in the stencil: Gamma-only by default, the
    // k-point Hessian on `options.pbc.kmesh` when `options.kpoint` is set. The
    // mesh is held fixed across the strain (it is defined in fractional
    // reciprocal coordinates, so it scales with the reciprocal lattice).
    //
    // **The real-space cutoffs travel with the strain** (`scale` is the linear
    // factor `(V/V0)^(1/3)` of this node). They are radii in Bohr, and the image
    // lists behind every lattice sum are `|T| < cutoff` over integer offsets, so
    // holding them fixed while the cell breathes makes images cross the boundary
    // at discrete volumes: `ln lambda(ln V)` acquires steps instead of being
    // smooth. Scaling them by the same factor keeps the *integer* image set
    // identical at every node, which is what a volume derivative needs. Measured
    // on diamond at the lean cutoffs of the physical-consistency fixture, this is
    // the difference between a `gamma2` that is noise (the three- and five-point
    // stencils disagreed by 1.5x the value itself at delta = 5e-3) and one that
    // agrees with the production-cutoff answer.
    let hessian_at = |strained: &PeriodicSystem, scale: f64| -> Result<Matrix> {
        let pbc = PbcOptions {
            ao_cutoff: options.pbc.ao_cutoff * scale,
            ewald: crate::pbc::EwaldOptions {
                real_cutoff: options.pbc.ewald.real_cutoff * scale,
                sr_cutoff: options.pbc.ewald.sr_cutoff * scale,
                ..options.pbc.ewald
            },
            ..options.pbc
        };
        let result = if options.kpoint {
            pbc_kpoint_hessian(strained, params, &options.electronic, &pbc)?
        } else {
            pbc_gamma_hessian(strained, params, &options.electronic, &pbc)?
        };
        Ok(result.hessian)
    };

    let h0 = hessian_at(system, 1.0)?;
    let hp = hessian_at(&expanded, (1.0 + delta).powf(third))?;
    let hm = hessian_at(&compressed, (1.0 - delta).powf(third))?;

    let ref_modes = mode_set(&h0, &atomic_numbers)?;
    let plus_modes = mode_set(&hp, &atomic_numbers)?;
    let minus_modes = mode_set(&hm, &atomic_numbers)?;

    let (perm_p, abs_ov_p) = match_modes(&ref_modes, &plus_modes, ndof);
    let (perm_m, abs_ov_m) = match_modes(&ref_modes, &minus_modes, ndof);

    let signed_cm1 = |lambda: f64| {
        let w = lambda.abs().sqrt() * WAVENUMBER_PER_SQRT_AU;
        if lambda < 0.0 {
            -w
        } else {
            w
        }
    };
    let frequencies_cm1: Vec<f64> = ref_modes.values.iter().map(|&l| signed_cm1(l)).collect();
    let frequencies_cm1_expanded: Vec<f64> = (0..ndof)
        .map(|i| signed_cm1(plus_modes.values[perm_p[i]]))
        .collect();
    let frequencies_cm1_compressed: Vec<f64> = (0..ndof)
        .map(|i| signed_cm1(minus_modes.values[perm_m[i]]))
        .collect();

    // gamma = -d ln omega / d ln V = -(1/2) d ln lambda / d ln V.
    let dln_v = ((1.0 + delta) / (1.0 - delta)).ln();
    let groups = degenerate_groups(&frequencies_cm1, options.degeneracy_tolerance_cm1);
    let quality_p = subspace_projection_quality(&abs_ov_p, &perm_p, &groups);
    let quality_m = subspace_projection_quality(&abs_ov_m, &perm_m, &groups);
    let mut match_overlaps: Vec<f64> = (0..ndof).map(|i| quality_p[i].min(quality_m[i])).collect();
    let mut mode_gamma = vec![f64::NAN; ndof];
    for &(start, len) in &groups {
        if start + len <= options.acoustic_modes {
            continue; // wholly acoustic group: no meaningful gamma
        }
        // Subspace trace of ln lambda: invariant under the arbitrary rotation the
        // eigensolver picks inside a degenerate block, so this is the
        // assignment-independent way to average gamma over the subspace.
        let mut sum_plus = 0.0;
        let mut sum_minus = 0.0;
        let mut usable = 0usize;
        for i in start..start + len {
            let lp = plus_modes.values[perm_p[i]];
            let lm = minus_modes.values[perm_m[i]];
            if lp <= 0.0 || lm <= 0.0 {
                continue;
            }
            sum_plus += lp.ln();
            sum_minus += lm.ln();
            usable += 1;
        }
        if usable == 0 {
            continue;
        }
        let gamma = -0.5 * (sum_plus - sum_minus) / (usable as f64 * dln_v);
        for g in mode_gamma.iter_mut().take(start + len).skip(start) {
            *g = gamma;
        }
    }
    // Acoustic branches carry no meaningful gamma at Gamma; mark them explicitly.
    for g in mode_gamma.iter_mut().take(options.acoustic_modes) {
        *g = f64::NAN;
    }

    // Second order: fit ln lambda_i against ln V through every matched volume.
    let mut mode_gamma2 = vec![f64::NAN; ndof];
    let mut mode_gamma_refit = vec![f64::NAN; ndof];
    if let Some(stencil) = stencil {
        // Every strained node — including the outer pair of the five-point
        // stencil — is matched onto the *central* volume's modes, so one
        // reference ordering runs through the whole fit.
        //
        // The nodes sit at `V(1 + m * delta_second)`. When `delta_second` equals
        // `delta` the inner pair *is* the first-order pair, and the fit reuses
        // those two Hessians for free; otherwise it costs two (four for the
        // five-point stencil) extra ones.
        let reuses_first_order_nodes =
            (delta_second - delta).abs() <= f64::EPSILON * delta.max(1.0);
        let multipliers: &[i32] = match stencil {
            SecondOrderStencil::ThreePoint => &[-1, 1],
            SecondOrderStencil::FivePoint => &[-2, -1, 1, 2],
        };
        let identity: Vec<usize> = (0..ndof).collect();
        let mut nodes: Vec<(f64, Vec<f64>)> = Vec::with_capacity(5);
        nodes.push((1.0, matched_ln_lambda(&ref_modes.values, &identity, ndof)));
        for &m in multipliers {
            let ratio = 1.0 + f64::from(m) * delta_second;
            if reuses_first_order_nodes && m == -1 {
                nodes.push((ratio, matched_ln_lambda(&minus_modes.values, &perm_m, ndof)));
                continue;
            }
            if reuses_first_order_nodes && m == 1 {
                nodes.push((ratio, matched_ln_lambda(&plus_modes.values, &perm_p, ndof)));
                continue;
            }
            let scale = ratio.powf(third);
            let strained = scale_lattice_isotropic(system, scale)?;
            let h = hessian_at(&strained, scale)?;
            let modes = mode_set(&h, &atomic_numbers)?;
            let (perm, abs_ov) = match_modes(&ref_modes, &modes, ndof);
            let quality = subspace_projection_quality(&abs_ov, &perm, &groups);
            for (q_min, q) in match_overlaps.iter_mut().zip(quality.iter()) {
                *q_min = q_min.min(*q);
            }
            nodes.push((ratio, matched_ln_lambda(&modes.values, &perm, ndof)));
        }
        // Ascending in `ln V`; the Fornberg recursion is happier on ordered nodes.
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let node_u: Vec<f64> = nodes.iter().map(|(ratio, _)| ratio.ln()).collect();
        let node_ln_lambda: Vec<Vec<f64>> =
            nodes.into_iter().map(|(_, ln_lambda)| ln_lambda).collect();

        let weights = fd_weights(&node_u, 0.0, 2);
        for &(start, len) in &groups {
            if start + len <= options.acoustic_modes {
                continue; // wholly acoustic group
            }
            // Same subspace trace as the first-order path: sum ln lambda over the
            // degenerate block at every node, so the fit is taken on a quantity
            // that does not depend on the eigensolver's basis inside the block.
            let mut sums = vec![0.0; node_u.len()];
            let mut usable = 0usize;
            for i in start..start + len {
                if node_ln_lambda.iter().any(|node| !node[i].is_finite()) {
                    continue;
                }
                for (s, node) in sums.iter_mut().zip(node_ln_lambda.iter()) {
                    *s += node[i];
                }
                usable += 1;
            }
            if usable == 0 {
                continue;
            }
            let inv = 1.0 / usable as f64;
            let mut d1 = 0.0;
            let mut d2 = 0.0;
            for (k, s) in sums.iter().enumerate() {
                d1 += weights[1][k] * s * inv;
                d2 += weights[2][k] * s * inv;
            }
            // lambda = omega^2, so ln omega = (1/2) ln lambda.
            for i in start..start + len {
                mode_gamma_refit[i] = -0.5 * d1;
                mode_gamma2[i] = 0.5 * d2;
            }
        }
        for i in 0..options.acoustic_modes {
            mode_gamma2[i] = f64::NAN;
            mode_gamma_refit[i] = f64::NAN;
        }
    }

    let mut thermodynamic_gamma = Vec::with_capacity(options.temperatures.len());
    for &t in &options.temperatures {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in options.acoustic_modes..ndof {
            let gamma = mode_gamma[i];
            if !gamma.is_finite() {
                continue;
            }
            let c = einstein_heat_capacity(mode_energy_hartree(ref_modes.values[i]), t);
            num += gamma * c;
            den += c;
        }
        thermodynamic_gamma.push((t, if den > 0.0 { num / den } else { f64::NAN }));
    }

    let mut thermodynamic_gamma2 = Vec::new();
    let mut thermodynamic_gamma2_full = Vec::new();
    if stencil.is_some() {
        let omega: Vec<f64> = ref_modes
            .values
            .iter()
            .map(|&l| mode_energy_hartree(l))
            .collect();
        let optical = options.acoustic_modes;
        for &t in &options.temperatures {
            let (_, g2, g2_full) = thermodynamic_second_order(
                &omega[optical..],
                &mode_gamma[optical..],
                &mode_gamma2[optical..],
                t,
            );
            thermodynamic_gamma2.push((t, g2));
            thermodynamic_gamma2_full.push((t, g2_full));
        }
    }

    Ok(GruneisenResult {
        volume: lattice.volume(),
        delta,
        delta_second: stencil.map(|_| delta_second),
        frequencies_cm1,
        frequencies_cm1_expanded,
        frequencies_cm1_compressed,
        mode_gamma,
        mode_gamma2,
        mode_gamma_refit,
        thermodynamic_gamma,
        thermodynamic_gamma2,
        thermodynamic_gamma2_full,
        second_order_stencil: stencil,
        match_overlaps,
        acoustic_modes: options.acoustic_modes,
        degenerate_groups: groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Einstein heat capacity must reach the classical limit k_B at high T and
    // freeze out at low T, with no overflow at either extreme.
    #[test]
    fn einstein_heat_capacity_limits() {
        // 1332 cm^-1 (diamond optical) expressed in Hartree.
        let omega = mode_energy_hartree((1332.0_f64 / WAVENUMBER_PER_SQRT_AU).powi(2));
        assert!(
            (omega - 6.069e-3).abs() < 1.0e-5,
            "diamond optical mode energy {omega:.6e} Hartree"
        );
        let hot = einstein_heat_capacity(omega, 1.0e7);
        assert!(
            (hot / KB_HARTREE_PER_K - 1.0).abs() < 1.0e-6,
            "classical limit {}",
            hot / KB_HARTREE_PER_K
        );
        let cold = einstein_heat_capacity(omega, 1.0);
        assert!(
            cold >= 0.0 && cold < 1.0e-100 * KB_HARTREE_PER_K,
            "frozen-out capacity {cold:.3e}"
        );
        assert_eq!(einstein_heat_capacity(omega, 0.0), 0.0);
        assert_eq!(einstein_heat_capacity(-1.0, 300.0), 0.0);
    }

    // A uniform gamma over all optical modes must average to itself regardless of
    // the temperature weighting; and the degenerate grouping must find the blocks.
    #[test]
    fn degenerate_grouping_splits_on_tolerance() {
        let freqs = [0.0, 0.0, 0.0, 1300.0, 1300.0, 1300.0, 1500.0];
        let groups = degenerate_groups(&freqs, 1.0);
        assert_eq!(groups, vec![(0, 3), (3, 3), (6, 1)]);
        let groups = degenerate_groups(&freqs, 1.0e4);
        assert_eq!(groups, vec![(0, 7)]);
    }

    // Greedy maximum-overlap matching must recover an exact permutation when the
    // two mode sets differ only by a reordering.
    #[test]
    fn mode_matching_recovers_a_permutation() {
        let n = 4;
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 1.0;
        }
        let order = [2usize, 0, 3, 1];
        let mut b = Matrix::zeros(n, n);
        for (j, &src) in order.iter().enumerate() {
            b[(src, j)] = -1.0; // opposite sign: matching must use |overlap|
        }
        let ref_set = ModeSet {
            values: vec![1.0, 2.0, 3.0, 4.0],
            vectors: a,
        };
        let other = ModeSet {
            values: vec![3.0, 1.0, 4.0, 2.0],
            vectors: b,
        };
        let (perm, abs_overlap) = match_modes(&ref_set, &other, n);
        for i in 0..n {
            assert_eq!(order[perm[i]], i, "mode {i} matched to {}", perm[i]);
            assert!((abs_overlap[(i, perm[i])] - 1.0).abs() < 1.0e-12);
        }
        // Every mode is its own (non-degenerate) group: the subspace projection
        // quality collapses to the per-mode overlap, which is 1 for a permutation.
        let groups: Vec<(usize, usize)> = (0..n).map(|i| (i, 1)).collect();
        let quality = subspace_projection_quality(&abs_overlap, &perm, &groups);
        for q in quality {
            assert!((q - 1.0).abs() < 1.0e-12, "projection quality {q}");
        }
    }

    // Fornberg weights must reproduce the textbook uniform-grid stencils and be
    // exact on a polynomial of the interpolating degree at non-uniform nodes.
    #[test]
    fn finite_difference_weights_match_known_stencils() {
        let h = 0.25_f64;
        let w = fd_weights(&[-h, 0.0, h], 0.0, 2);
        let want1 = [-0.5 / h, 0.0, 0.5 / h];
        let want2 = [1.0 / (h * h), -2.0 / (h * h), 1.0 / (h * h)];
        for k in 0..3 {
            assert!(
                (w[1][k] - want1[k]).abs() < 1.0e-12,
                "3-point d1[{k}] {}",
                w[1][k]
            );
            assert!(
                (w[2][k] - want2[k]).abs() < 1.0e-12,
                "3-point d2[{k}] {}",
                w[2][k]
            );
        }
        let w = fd_weights(&[-2.0 * h, -h, 0.0, h, 2.0 * h], 0.0, 2);
        let want1 = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];
        let want2 = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0];
        for k in 0..5 {
            assert!(
                (w[1][k] - want1[k] / h).abs() < 1.0e-12,
                "5-point d1[{k}] {}",
                w[1][k]
            );
            assert!(
                (w[2][k] - want2[k] / (h * h)).abs() < 1.0e-12,
                "5-point d2[{k}] {}",
                w[2][k]
            );
        }
        // Non-uniform nodes (the actual case here: ln(1+delta) != -ln(1-delta)):
        // three points reproduce any quadratic exactly.
        let nodes = [-0.3, 0.0, 0.7];
        let f = |x: f64| 2.0 + 3.0 * x - 5.0 * x * x;
        let w = fd_weights(&nodes, 0.0, 2);
        let value: f64 = (0..3).map(|k| w[0][k] * f(nodes[k])).sum();
        let d1: f64 = (0..3).map(|k| w[1][k] * f(nodes[k])).sum();
        let d2: f64 = (0..3).map(|k| w[2][k] * f(nodes[k])).sum();
        assert!((value - 2.0).abs() < 1.0e-12, "interpolated value {value}");
        assert!((d1 - 3.0).abs() < 1.0e-12, "first derivative {d1}");
        assert!((d2 + 10.0).abs() < 1.0e-12, "second derivative {d2}");

        // The real volume nodes, and the size of the trap they set. On diamond
        // (f = ln lambda, f' = -2 gamma = -1.8108, f'' = 2 gamma2 = -0.0767) the
        // exact weights recover f''; the textbook symmetric second difference
        // instead returns ~ -f' = +1.73, i.e. gamma2 = +0.867 -- a
        // plausible-looking number and a pure artefact of the node asymmetry.
        let delta = 5.0e-3_f64;
        let nodes = [(1.0 - delta).ln(), 0.0, (1.0 + delta).ln()];
        let (slope, curvature) = (-2.0 * 0.905418, 2.0 * -0.038336);
        let f = |x: f64| slope * x + 0.5 * curvature * x * x;
        let w = fd_weights(&nodes, 0.0, 2);
        let exact: f64 = (0..3).map(|k| w[2][k] * f(nodes[k])).sum();
        assert!(
            (exact - curvature).abs() < 1.0e-9,
            "exact curvature on the volume nodes: {exact} vs {curvature}"
        );
        let h = 0.5 * (nodes[2] - nodes[0]);
        let naive = (f(nodes[2]) - 2.0 * f(nodes[1]) + f(nodes[0])) / (h * h);
        assert!(
            (naive - 1.73414).abs() < 1.0e-4,
            "the naive symmetric stencil is expected to leak -f' = {} here, got {naive}",
            -slope
        );
    }

    // `d ln c / dx` must match a numerical derivative of the heat capacity across
    // the whole range, including the small-x series branch where the closed form
    // is two diverging terms cancelling.
    #[test]
    fn heat_capacity_log_derivative_matches_finite_difference() {
        // With T = 1 K the argument of `einstein_heat_capacity` is x * k_B.
        let c_of_x = |x: f64| einstein_heat_capacity(x * KB_HARTREE_PER_K, 1.0);
        for &x in &[1.0e-4_f64, 1.0e-2, 0.5, 1.0, 6.3, 20.0, 50.0] {
            let h = 1.0e-6 * x.max(1.0e-2);
            let numeric = (c_of_x(x + h).ln() - c_of_x(x - h).ln()) / (2.0 * h);
            let analytic = einstein_heat_capacity_log_derivative(x);
            assert!(
                (analytic - numeric).abs() < 1.0e-6 * (1.0 + analytic.abs()),
                "x={x}: analytic {analytic:.9e} vs numeric {numeric:.9e}"
            );
        }
        // Limits: classical modes are volume-insensitive, stiff ones lose weight.
        assert!(einstein_heat_capacity_log_derivative(1.0e-9).abs() < 1.0e-9);
        assert!((einstein_heat_capacity_log_derivative(1.0e4) + 1.0).abs() < 1.0e-3);
        assert_eq!(einstein_heat_capacity_log_derivative(-1.0), 0.0);
    }

    // The *full* second-order thermodynamic average must be the true volume
    // derivative of `gamma_th(T, V)`. Gate it on a synthetic model where the
    // volume dependence is known in closed form:
    //
    //     ln omega_i(v) = ln omega_i0 - gamma_i0 v + (1/2) g2_i v^2,   v = ln(V/V0)
    //
    // so gamma_i(v) = gamma_i0 - g2_i v and gamma2_i = g2_i exactly, and
    // gamma_th(T, v) can be differentiated numerically. This is the gate on the
    // `d c_i / d ln V` reweighting term, which cancels identically on diamond
    // (one gamma for the whole degenerate optical triplet).
    #[test]
    fn full_thermodynamic_second_order_matches_numerical_volume_derivative() {
        let omega0 = [2.0e-3, 6.0e-3, 1.0e-2]; // Hartree; x(300 K) ~ 2.1, 6.3, 10.5
        let gamma0 = [1.5, 0.5, 1.0];
        let g2 = [-0.4, 0.8, 1.2];
        let t = 300.0;
        let model_gamma_th = |v: f64| {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..omega0.len() {
                let omega = omega0[i] * (-gamma0[i] * v + 0.5 * g2[i] * v * v).exp();
                let c = einstein_heat_capacity(omega, t);
                num += (gamma0[i] - g2[i] * v) * c;
                den += c;
            }
            num / den
        };
        let h = 1.0e-4;
        let numeric = -(model_gamma_th(h) - model_gamma_th(-h)) / (2.0 * h);
        let (gamma_th, gamma2_mode, gamma2_full) =
            thermodynamic_second_order(&omega0, &gamma0, &g2, t);
        assert!(
            (gamma_th - model_gamma_th(0.0)).abs() < 1.0e-12,
            "gamma_th {gamma_th} vs model {}",
            model_gamma_th(0.0)
        );
        assert!(
            (gamma2_full - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
            "full -d gamma_th/d ln V: {gamma2_full:.9} vs numerical {numeric:.9}"
        );
        // The reweighting term must be big enough here that dropping it fails.
        assert!(
            (gamma2_full - gamma2_mode).abs() > 1.0e-3,
            "reweighting term {:.3e} is too small to gate anything",
            gamma2_full - gamma2_mode
        );

        // ... and it must vanish identically when every mode shares one gamma,
        // which is the diamond case.
        let uniform = [0.9; 3];
        let (_, mode_avg, full) = thermodynamic_second_order(&omega0, &uniform, &g2, t);
        assert!(
            (mode_avg - full).abs() < 1.0e-14,
            "uniform-gamma correction should vanish, got {:.3e}",
            mode_avg - full
        );
    }

    // Inside a degenerate block the individual overlaps are arbitrary, but the
    // projection onto the assigned subspace must still be 1.
    #[test]
    fn subspace_projection_is_rotation_invariant() {
        // 2x2 degenerate block: the strained basis is the reference basis rotated
        // by 45 degrees, so every per-mode overlap is 1/sqrt(2).
        let n = 2;
        let mut abs_overlap = Matrix::zeros(n, n);
        let c = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..n {
            for j in 0..n {
                abs_overlap[(i, j)] = c;
            }
        }
        let perm = vec![0usize, 1];
        let quality = subspace_projection_quality(&abs_overlap, &perm, &[(0, 2)]);
        for q in &quality {
            assert!((q - 1.0).abs() < 1.0e-12, "rotated block quality {q}");
        }
        // Treated as two separate non-degenerate modes the same data looks bad.
        let quality = subspace_projection_quality(&abs_overlap, &perm, &[(0, 1), (1, 1)]);
        for q in &quality {
            assert!((q - c).abs() < 1.0e-12, "per-mode quality {q}");
        }
    }
}
