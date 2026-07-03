// SPDX-License-Identifier: GPL-3.0-or-later

use crate::basis::{BasisOptions, BasisSet, AUTO_NPRIM};
use crate::coulomb::{
    coulomb_energy_potential_from_matrix, effective_coulomb_matrix, ShellChargeModel,
};
use crate::cphf::{transition_shell_charges, CpxtbSpace};
use crate::dispersion::{
    d4_dispersion_atm_geometry, d4_dispersion_energy_potential_with_cn_pairs_and_atm,
    d4_dispersion_energy_with_cn_pairs_and_atm, d4_dispersion_pairs, dispersion_energy,
    D4DispersionOptions,
};
use crate::error::{Gfn1Error, Result};
use crate::field::{
    electric_field_energy, electric_shell_potential, mulliken_dipole, ExternalFieldOptions,
};
use crate::halogen::halogen_energy;
use crate::hamiltonian::{build_h0, HamiltonianCore, HamiltonianOptions};
use crate::integrals::IntegralMatrices;
use crate::linalg::{
    column_weighted_gram, lowdin_orthogonalizer, lowdin_solve_with_orthogonalizer, Matrix,
};
use crate::math::Vec3;
use crate::model::BoundaryCondition;
use crate::params::Gfn1Parameters;
use crate::repulsion::repulsion_energy;
use crate::secondary_basis::SecondaryBasis;
use crate::system::PeriodicSystem;

const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_808_578_545_117e-6;
const ELECTRON_COUNT_TOLERANCE: f64 = 1.0e-8;

/// SCC charge-convergence accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SccAccelerator {
    /// Damped linear (fixed-point) mixing.
    Linear,
    /// Broyden quasi-Newton charge mixing (the historical default).
    Broyden,
    /// Pulay DIIS on the charge residual (the SCC realization of CDIIS).
    Cdiis,
    /// Second-order (Newton) step using the transition-charge susceptibility.
    Newton,
}

impl Default for SccAccelerator {
    fn default() -> Self {
        Self::Broyden
    }
}

/// Which off-site anisotropic-electrostatics model the multipole correction uses (requires
/// [`ElectronicOptions::multipole`]). Default [`MultipoleModel::Mdftb2`] keeps the existing
/// behaviour byte-for-byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultipoleModel {
    /// The current mDFTB2 multipole expansion (Ohno-kernel off-site tensors, on-site AO
    /// fluctuation moments). Unchanged.
    Mdftb2,
    /// **Experimental CAMM-on-mDFTB2** (v0.4.2): mDFTB2 supplies only the SCF/Fock/gradient
    /// skeleton and the on-site response penalty; the *off-site* anisotropy is replaced by a
    /// GFN2-style CAMM/AES term (`q–μ, q–Θ, μ–μ` only) using full **cumulative** atomic multipole
    /// moments and the parameter-free erf-cloud damped kernel. The mDFTB2 off-site Ohno multipole
    /// is disabled in this mode (no double counting). See [`crate::multipole::camm_aes_energy_fock`].
    CammOnMdftb2,
}

impl Default for MultipoleModel {
    fn default() -> Self {
        Self::Mdftb2
    }
}

/// Named **CAMM-on-mDFTB2 parameter presets**, returning `(global_κ, element_κ_overrides, s_AES,
/// global_s_onsite, element_s_onsite_overrides)` for a preset name, or `None` if unknown. Both
/// override lists are `(Z, value)` pairs; `global_κ` / `global_s_onsite` apply to every other
/// element (incl. TM, intentionally left untuned at κ=1.0). Only `"sigma-hole"` uses per-element
/// `s_onsite`; the older presets return an empty override list (single global `s_onsite`).
///
/// Provenance: the κ + s_onsite values come from the full-data multi-regime optimization in
/// `scripts/optimize_kappa.py` / `scripts/optimize_sigma_hole.py` (coordinate descent over ALL
/// per-element κ + the on-site penalty scale `s_onsite`; `s_AES` fixed at 1.0; no element pinning).
/// CAMM is regime-dependent — for transition-metal reaction *energies* none helps; use plain GFN1:
/// - `"polar"`: H-bonds / salt bridges / dispersion — fit on S66 + A24 + SSI energies + NCI geometry.
/// - `"halogen"`: σ-hole / halogen bonds — fit on HAL59 (energy + geometry) with S66/A24 guards and
///   MOR41 geometry; `s_onsite ≪ 1` tempers the on-site over-penalty that `κ` alone cannot fix.
/// - `"sigma-hole"` (v0.4.4): the unified σ-hole preset — per-element κ AND **per-element s_onsite**
///   (halogen wants s_onsite≈0, tetrel/Si wants ≈1, a split a single global scalar cannot express),
///   fit across HAL59/A24/S66/SSI energies + MOR41/NBPRC gradients (`optimize_sigma_hole.py`, obj
///   6.54). σ-hole NCI oriented; geometry-neutral on TM covalent frameworks (see the module note).
pub fn camm_preset(name: &str) -> Option<(f64, Vec<(u8, f64)>, f64, f64, Vec<(u8, f64)>)> {
    // (Z, κ): H=1 B=5 C=6 N=7 O=8 F=9 Si=14 P=15 S=16 Cl=17 Br=35 I=53.
    // From the full-data, GFN1-normalized, all-element fit (scripts/optimize_kappa.py).
    match name {
        // H-bonds / salt bridges / dispersion: NCI interaction energies (S66+A24+SSI) plus NCI and
        // SSI geometry gradients (the all-grad fit; richer gradient info than the energy-only fit,
        // tmQM50 mean 0.127 vs 0.133 Å). NOTE: tuned for NCI energetics — it does NOT beat plain
        // GFN1 on molecular geometry (tmQM50 RMSD-to-g-xTB 0.127 vs GFN1 0.073 Å).
        "polar" => Some((
            1.0,
            vec![(1, 2.625), (5, 0.175), (6, 2.425), (7, 7.375), (8, 7.375), (9, 7.375)],
            1.0,
            0.85,
            vec![],
        )),
        // σ-hole / halogen bonds (HAL59 energy + geometry). s_onsite≈0 tempers the on-site
        // over-penalty that κ cannot fix. NOTE: specialized for halogen bonds — NOT for TM geometry.
        "halogen" => Some((
            1.0,
            vec![(1, 1.675), (5, 0.125), (6, 2.425), (7, 1.65), (8, 2.425), (9, 3.4), (15, 7.75),
                 (16, 7.375), (17, 0.225), (35, 1.225), (53, 7.75)],
            1.0,
            0.02,
            vec![],
        )),
        // Legacy halogen fit (pre-normalization). Kept because it gives the best tested TM-complex
        // geometry of the halogen-type presets (gutxok Ni–donor MAD 0.059) while still improving
        // HAL59 over GFN1; a milder s_onsite than the rigorous "halogen".
        "halogen-v1" => Some((
            1.0,
            vec![(1, 1.75), (5, 0.1), (7, 3.25), (8, 3.0), (9, 6.0), (15, 0.1), (17, 0.5),
                 (35, 2.25), (53, 4.75)],
            1.0,
            0.05,
            vec![],
        )),
        // All-gradient halogen fit (HAL59 energy + all-grad; optimized_kappa_allgrad.json). Richer
        // gradient information than "halogen"; the mildest CAMM perturbation of TM geometry tested
        // (tmQM50 0.085 vs GFN1 0.073 Å). Same s_onsite≈0 σ-hole tempering.
        "halogen-allgrad" => Some((
            1.0,
            vec![(1, 1.675), (5, 0.125), (6, 2.625), (7, 2.45), (8, 2.65), (9, 3.4), (15, 7.75),
                 (16, 7.375), (17, 0.2), (35, 1.2), (53, 7.75)],
            1.0,
            0.02,
            vec![],
        )),
        // v0.4.4 unified σ-hole preset (scripts/optimize_sigma_hole.py, obj 6.54): per-element κ +
        // per-element s_onsite. The per-element s_onsite resolves the halogen(≈0)/tetrel(≈1) split
        // no single global scalar can express. Global s_onsite=0.05 is the off-list fallback (e.g.
        // TM); global κ=1.0. σ-hole/halogen + tetrel oriented; geometry-neutral on TM frameworks.
        "sigma-hole" => Some((
            1.0,
            vec![(1, 1.9), (5, 4.9), (6, 1.2), (7, 2.1), (8, 2.575), (9, 7.375), (14, 1.775),
                 (15, 0.225), (16, 7.75), (17, 2.475), (35, 5.625), (53, 4.25)],
            1.0,
            0.05,
            vec![(1, 0.0), (5, 0.0), (6, 0.2), (7, 0.15), (8, 1.22), (9, 0.02), (14, 1.23),
                 (15, 0.24), (16, 0.52), (17, 0.11), (35, 0.02), (53, 0.0)],
        )),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct ElectronicOptions {
    pub charge: Option<f64>,
    pub spin_multiplicity: Option<usize>,
    pub max_scc: usize,
    pub energy_tolerance: f64,
    pub charge_tolerance: f64,
    pub mixing: f64,
    pub scc_broyden: bool,
    pub scc_broyden_size: usize,
    /// SCC convergence accelerator. When left at [`SccAccelerator::Broyden`] the
    /// legacy `scc_broyden` flag still selects Broyden vs. linear mixing, so the
    /// default behavior is unchanged; setting `Cdiis`/`Newton` overrides it.
    pub scc_accelerator: SccAccelerator,
    /// Virtual-orbital level shift (Hartree) applied during the SCC to damp
    /// oscillations in small-gap systems. 0 disables it.
    pub level_shift: f64,
    pub nprim: usize,
    pub eigen_tolerance: f64,
    pub electronic_temperature: f64,
    pub boundary: BoundaryCondition,
    pub enable_dispersion: bool,
    pub d3_reference_path: Option<String>,
    /// Experimental non-PBC self-consistent D4 dispersion. When true,
    /// `enable_dispersion` selects D4 instead of the stock D3(BJ) term. The D4
    /// charge potential enters the SCC as an atomic scalar shift; analytic
    /// gradients include the fixed-charge D4 geometry derivative.
    pub experimental_d4: bool,
    pub d4_cutoff: f64,
    pub d4_cn_cutoff: f64,
    pub d4_atm: bool,
    pub d4_atm_cutoff: f64,
    pub d4_s9: Option<f64>,
    pub hamiltonian: HamiltonianOptions,
    /// Uniform external electric/magnetic field perturbation (off by default).
    pub external_field: ExternalFieldOptions,
    /// Experimental parameter-free **mDFTB2 multipole** electrostatics correction
    /// (atomic dipole + quadrupole density-fluctuation interactions, self-consistent),
    /// non-periodic only. Off by default — when enabled, the multipole field enters the
    /// SCC so the density relaxes, and its energy adds to the total. See
    /// [`crate::multipole`].
    pub multipole: bool,
    /// Experimental **octupole** extension of the mDFTB2 multipole electrostatics
    /// (adds the atomic rank-3 octupole to the self-consistent multipole field). Requires
    /// `multipole`; off by default. Only meaningful for atoms with d functions (the
    /// traceless octupole vanishes for an s/p-only minimal basis). See [`crate::multipole`].
    pub multipole_octupole: bool,
    /// Experimental **first-order field–dipole coupling** (Stage 3). With an external electric
    /// field on, couples the mDFTB2 atomic dipole moments to the field, adding
    /// `E_field = −E·Σ_A d_A` on top of the monopole `−E·Σ_A q_A R_A`, self-consistently (the
    /// `d_A` respond to the field). Requires `multipole` + an electric field; off by default
    /// (≡ monopole-only field coupling). Also folds `Σ_A d_A` into the reported dipole. See
    /// [`crate::multipole::field_dipole_fock`].
    pub field_multipole: bool,
    /// Experimental **third-order on-site multipole** electrostatics cross terms. Adds the
    /// leading angular generalizations of the monopole `(1/3)ΓΔq³`, namely the on-site
    /// charge·dipole² and charge·quad² terms `E³ = Σ_A [α_A Δq_A(d_A·d_A) + β_A Δq_A(Q_A:Q_A)]`
    /// with parameter-free `α,β` from the hardness charge-derivative. Requires `multipole`; off
    /// by default. See [`crate::multipole::third_order_fock_from_moments`].
    pub multipole_third_order: bool,
    /// Experimental **per-rank multipole×charge cross terms** (arbitrary rank × arbitrary charge
    /// order). `multipole_charge_order[l−1]` is the highest charge order (≥3) the rank-`l` atomic
    /// multipole self-energy couples to, via the breathing-radius Taylor expansion of
    /// `½ g_l(η_A(q))(m_l·m_l)`. Empty (default) ⇒ no cross terms (just the base 2nd-order multipole).
    /// E.g. `[6, 4, 2, 2]` = dipole→6th, quadrupole→4th, octupole/hexadecapole→2nd only. Requires
    /// `multipole`; uses (and forces on) the generic arbitrary-rank path so the cross terms cover all
    /// ranks `1..=L`. Parameter-free; self-consistent energy + analytic gradient. The legacy
    /// [`Self::multipole_third_order`] (rank 1/2, order 3) is subsumed (`≈ [3, 3]`). See
    /// [`crate::multipole::multipole_charge_cross_fields`].
    pub multipole_charge_order: Vec<usize>,
    /// Experimental **richer secondary-basis on-site moments**. When `Some`, the mDFTB2 atomic
    /// dipole/quadrupole moment *integrals* are evaluated over the node-correct secondary
    /// (GFN1-xTB-cc-pVnZ) basis instead of the minimal primary basis (the Mulliken population is
    /// still the primary one) — better-resolved moments for every moment-based term. Requires
    /// `multipole`; `None` (default) ≡ primary-basis moments. See
    /// [`crate::multipole::secondary_moment_integrals`].
    pub multipole_secondary_basis: Option<SecondaryBasis>,
    /// **Highest atomic multipole rank** of the experimental mDFTB2 electrostatics (requires
    /// `multipole`). How to choose the rank:
    ///
    /// | want | set |
    /// |---|---|
    /// | rank 1–2 (dipole + quadrupole) | `multipole = true`, `multipole_order = 0` (default) |
    /// | rank 1–3 (+ octupole) | also `multipole_octupole = true` (still `multipole_order = 0`) |
    /// | rank 1–`n`, `n ≥ 4` (+ hexadecapole, …) | `multipole_order = n` |
    ///
    /// Ranks ≤ 3 run the speed-optimized, byte-compatible **legacy** paths (the booleans above pick
    /// them; `multipole_order < 4` is ignored there). `multipole_order ≥ 4` switches the (non-periodic)
    /// correction to the unified parameter-free **arbitrary-rank** path
    /// ([`crate::multipole::multipole_fock_generic`]), which self-consistently mixes the atomic
    /// moments of ranks `1..=n`, superseding the dipole/quad/octupole blocks. Experimental; the cost
    /// grows with rank (only atoms with the corresponding angular momentum carry a nonzero traceless
    /// moment). Independent of [`Self::charge_order`] (which expands the *isotropic monopole*).
    pub multipole_order: usize,
    /// **Off-site anisotropic-electrostatics model** (requires `multipole`). Default
    /// [`MultipoleModel::Mdftb2`] ≡ the current mDFTB2 off-site Ohno multipole expansion
    /// (byte-compatible). [`MultipoleModel::CammOnMdftb2`] replaces the off-site term by a
    /// GFN2-style CAMM/AES (`q–μ, q–Θ, μ–μ`) on full cumulative atomic multipole moments while
    /// keeping the mDFTB2 on-site penalty; the mDFTB2 off-site Ohno multipole is then disabled
    /// (no double counting). See [`crate::multipole::camm_aes_energy_fock`].
    pub multipole_model: MultipoleModel,
    /// **CAMM range factor `κ`** (only used by [`MultipoleModel::CammOnMdftb2`]; **primary**
    /// calibration lever). Scales the erf-cloud width `σ_AB = κ·σ_AB^HP`, tuning the short-range
    /// damping of the AES multipole tensors **range-selectively** — larger `κ` strengthens the
    /// contact screening (tempers CAMM's over-attraction) while leaving the long-range `1/Rⁿ`
    /// multipole tail unchanged (the `R≫σ` limit is `σ`-independent). Default `1.0` (the
    /// parameter-free hardness width). Must be `> 0`.
    pub camm_damp: f64,
    /// **CAMM AES amplitude `s_AES`** (only used by [`MultipoleModel::CammOnMdftb2`];
    /// *secondary/diagnostic* lever). Multiplies the whole AES energy + Fock + forces uniformly.
    /// Note this scales all distances/orders together, so unlike [`Self::camm_damp`] it cannot fix
    /// short-range over-attraction without also weakening the (correct) long-range tail. Default
    /// `1.0`. Must be `≥ 0`.
    pub camm_aes_scale: f64,
    /// **CAMM on-site penalty scale `s_onsite`** (only used by [`MultipoleModel::CammOnMdftb2`]).
    /// Multiplies the on-site (`a==a`) mDFTB self-energy penalty `½ g_l(η_A)(m_l·m_l)` fed the
    /// *cumulative* CAMM moments. This is a **distinct lever from κ**: κ only damps the off-site
    /// AES, but the per-atom on-site penalty (large because cumulative moments include the full
    /// cloud) is what dominates e.g. the spurious halogen-bond over-binding — a residue κ cannot
    /// touch. `s_onsite < 1` tempers that over-penalization; `s_onsite = 1.0` (default) is
    /// byte-identical to the original un-scaled penalty. Must be `≥ 0`.
    pub camm_onsite_scale: f64,
    /// **Element-specific on-site penalty scale `s_onsite`** overrides (`MultipoleModel::CammOnMdftb2`):
    /// a list of `(Z, s_Z)` pairs. Each atom of element `Z` scales its on-site (`a==a`) mDFTB
    /// self-energy penalty by `s_Z` instead of the global [`Self::camm_onsite_scale`]. Empty
    /// (default) ⇒ all atoms use the global scale. Motivated by the finding that the *optimal*
    /// on-site temper is opposite across σ-hole types — halogens want `s_onsite ≈ 0` (the on-site
    /// penalty over-binds), tetrels (Si) want `s_onsite ≈ 1` (the penalty is corrective) — which no
    /// single global value can satisfy. Independent of [`Self::camm_damp_elem`] (κ damps the
    /// off-site AES; this tempers the on-site penalty). Each value must be `≥ 0`.
    pub camm_onsite_scale_elem: Vec<(u8, f64)>,
    /// **Element-specific CAMM range factor** overrides (`MultipoleModel::CammOnMdftb2`): a list of
    /// `(Z, κ_Z)` pairs. Each atom of element `Z` uses `κ_Z` instead of the global
    /// [`Self::camm_damp`]; a pair `A–B` then screens with `σ_AB = √(κ_A·κ_B)·σ_AB^HP`. Empty
    /// (default) ⇒ all atoms use the global `camm_damp`. Motivated by the finding that the optimal
    /// κ differs by interaction type (H-bond elements want a *larger* κ, charged/ionic groups a
    /// *smaller* κ), which no single global κ can satisfy.
    pub camm_damp_elem: Vec<(u8, f64)>,
    /// **Charge-dependent CAMM range factor** `(κ₀, γ)` (`MultipoleModel::CammOnMdftb2`). When
    /// `Some`, each atom's κ is `κ_A = κ₀ / (1 + γ·Δq_A²)` from its (self-consistent) Mulliken
    /// charge — so neutral atoms (H-bond) keep the large `κ₀` (strong screening) while ionic atoms
    /// (large `|Δq|`) get a *smaller* κ (less screening, more attraction). This is the variable
    /// that actually distinguishes neutral H-bonds from ionic salt bridges of the *same element*
    /// (which `camm_damp_elem` cannot). κ is recomputed each SCC iteration from the mixed charges
    /// (frozen within the Fock; energy-only — forces are a follow-up). Overrides `camm_damp_elem`.
    pub camm_damp_charge: Option<(f64, f64)>,
    /// Experimental parameter-free **long-range Fock exchange** (MFX, LC-DFTB style). When on, the
    /// Mulliken-approximated long-range exact-exchange kernel `K[ΔP]` (built from the hardness-derived
    /// `γ^lr` and the HardnessPairwise ω) is added self-consistently to the SCC Fock on the density
    /// fluctuation `ΔP = P − P0` (neutral-atom reference), and its energy `½Tr[ΔP·K[ΔP]]` to the total.
    /// Off by default (≡ stock GFN1); non-periodic. See [`crate::exchange`].
    pub lr_exchange: bool,
    /// Experimental **on-site Fock-exchange (OFX) correction** layered on top of MFX. When on (and
    /// `lr_exchange` is on), the same-atom exchange is upgraded from the Mulliken approximation to the
    /// *exact* one-center long-range two-electron integrals via the difference kernel `K_OFX =
    /// K_onsite,refined^lr − K_onsite,Mulliken^lr` (no double count — the Mulliken half MFX already
    /// applies is subtracted). Real STO-nG one-center ERIs are built once per element (geometry-
    /// independent; cached, see [`crate::exchange::OnsiteExchangeCache`]) and contracted with `ΔP` each
    /// SCC iteration. Off by default; requires `lr_exchange`; non-periodic. See [`crate::exchange`].
    pub onsite_exchange: bool,
    /// Use the experimental **Trust-Region Augmented Hessian** second-order SCF for the
    /// exchange-augmented SCC (instead of commutator DIIS). TRAH minimises the energy directly over
    /// orbital rotations with a matrix-free Newton/trust-region step — robust where DIIS on the
    /// off-diagonal exchange Fock stalls. Only affects the exchange-only path (`lr_exchange` on,
    /// multipole off); closed-shell/gapped, integer occupations; non-periodic. Off by default
    /// (≡ the DIIS driver). See [`crate::trah`].
    pub scf_trah: bool,
    /// **Highest order of the isotropic on-site charge (monopole `Δq`) expansion** — the radial
    /// counterpart of [`Self::multipole_order`] (which handles the *angular* multipoles), and
    /// independent of it (no `multipole` flag needed). `3` (default) ≡ stock GFN1 (2nd-order
    /// Klopman–Ohno + 3rd-order DFTB3). `n ≥ 4` adds the experimental parameter-free **Linear
    /// Breathing-Radius** terms `E_k = Σ_A (1/k) X_k Δq_A^k` for `4 ≤ k ≤ n`, with the deterministic
    /// coefficients `X_k = (γ_A/(k−1))(2Γ_A/γ_A)^(k−2)` from the existing hardness `γ_A` and Hubbard
    /// derivative `Γ_A` (no fitting). Enters the SCC self-consistently. See [`crate::coulomb`].
    pub charge_order: usize,
    /// Experimental **dynamic (geometry-adaptive) range separation** for the long-range Fock
    /// exchange (the `LocalGeometry` ω scheme). When `true` (and `lr_exchange` is on), each atom's
    /// screening is `ω_A = η_A / s_A` with the parameter-free size factor `s_A = (1+CN_A)^(−1/3)` from
    /// the GFN1-Hamiltonian coordination number ([`crate::coulomb::local_size_factor_from_cn`]): a
    /// more-coordinated atom screens at shorter range. `false` (default) keeps the geometry-independent
    /// `HardnessPairwise` ω (`ω_A = η_A`), to which this reduces atom-by-atom at `CN = 0`. The analytic
    /// gradient adds the `∂ω/∂R` reorganisation force (the CN moves with the nuclei). Non-periodic.
    pub dynamic_omega: bool,
    /// Experimental **spin-polarized GFN1-xTB ("spGFN1")**. When `true`, an open-shell system
    /// (`spin_multiplicity` ≥ 2, or an odd electron count) is solved with a spin-**unrestricted**
    /// SCC that adds the atomic-spin-constant (`W`) spin-polarization energy
    /// `E_spin = ½ Σ_A Σ_{l,l'} W_{A,ll'} m_{A,l} m_{A,l'}` (shell magnetization
    /// `m_{A,l} = n^α − n^β`) and its self-consistent spin potential `V^σ = ±Σ_{l'} W m_{l'}`.
    /// A closed-shell singlet has zero spin density, so the term vanishes and the result is
    /// byte-identical to plain GFN1 (the closed-shell case is delegated to the restricted path).
    /// The `W` constants come from tblite (LGPL-3.0; see `third_party/tblite/`). Non-periodic
    /// only; off by default. v1 is the bare GFN1 electronic model — it errors if combined with
    /// the experimental multipole / exchange / external-field / D4 paths. See [`crate::spin`].
    pub spin_polarization: bool,
    /// Master switch for the experimental DFT+U / +U+V extended-Hubbard correction on
    /// the correlated (`d`) shell — the orbital-resolved self-interaction penalty that
    /// bare GFN1 lacks, targeted at transition-metal spin-state energetics and
    /// geometries. Off by default; non-periodic. See [`crate::plus_u`].
    pub plus_u: bool,
    /// Per-element on-site Hubbard `U` (Hartree) for the correlated shell, applied when
    /// `plus_u` is on and `hubbard_u_linear_response` is off (fixed-`U` mode).
    pub hubbard_u: Vec<(u8, f64)>,
    /// Include the inter-site Hubbard `+V` term (requires `plus_u`). The `+V` coupling
    /// restores the metal–ligand hybridisation that bare `+U` over-localises — the
    /// high-spin-overstabilisation fix for covalent complexes.
    pub plus_u_v: bool,
    /// Inter-site Hubbard `V` (Hartree) per unordered element pair `(z_a, z_b)`, applied
    /// to correlated atom pairs within `hubbard_v_cutoff`. Used when `plus_u_v` is on.
    pub hubbard_v: Vec<(u8, u8, f64)>,
    /// Distance cutoff (bohr) for the inter-site `+V` neighbour pairs.
    pub hubbard_v_cutoff: f64,
    /// Compute the on-site `U` self-consistently by the linear-response method
    /// (Cococcioni–de Gironcoli: `U = χ0⁻¹ − χ⁻¹` from the bare one-shot and the
    /// self-consistent occupation responses `dn/dα` to a localised potential `α` on the
    /// correlated subspace) instead of the fixed `hubbard_u` values. Requires `plus_u`.
    /// Off by default. NOTE: in the screened semiempirical SCC the bare/screened
    /// response separation is approximate (the GFN1 `γ` is already a screened kernel).
    pub hubbard_u_linear_response: bool,
    /// Apply the linear-response `+U` to **all** atoms carrying a `d` shell (also the
    /// nearly-empty main-group `d` polarisation shells), not only the transition
    /// metals (`reference_occ > 0`). Off by default. Only affects the auto-selected
    /// (`hubbard_u_linear_response`) subspace; the FLL penalty on an empty `d` shell
    /// is tiny, and such shells are more prone to the response ill-conditioning the
    /// extraction regularises.
    pub plus_u_all_d: bool,
    /// Finite-difference step (bohr) for the linear-response `+U` **consistent-force**
    /// term `dU/dR`, `dV/dR` (central difference of the per-geometry recomputed
    /// Hubbard parameters). Only used by the analytic gradient when `plus_u` and
    /// `hubbard_u_linear_response` are both on; ignored otherwise. Default `2.0e-3`.
    pub plus_u_force_fd_step: f64,
    /// Optional **warm-start** shell charges seeding the SCC mixing vector. Used by the
    /// multipole **rank-continuation** ladder ([`run_electronic_rank_ladder`]): converge a low
    /// multipole rank, then seed the next (higher) rank's SCC with the converged charges so the
    /// monopole/low-multipole channels start near their solution and only the new high-rank
    /// moment has to relax — far more robust than a cold high-rank SCC. `None` (default) =
    /// cold start from zero charges. Length must equal the shell count or it is ignored.
    pub scc_initial_shell_charges: Option<Vec<f64>>,
}

impl Default for ElectronicOptions {
    fn default() -> Self {
        Self {
            charge: None,
            spin_multiplicity: None,
            max_scc: 250,
            energy_tolerance: 1.0e-6,
            charge_tolerance: 2.0e-5,
            mixing: 0.4,
            scc_broyden: true,
            scc_broyden_size: 250,
            scc_accelerator: SccAccelerator::default(),
            level_shift: 0.0,
            nprim: AUTO_NPRIM,
            eigen_tolerance: 1.0e-12,
            electronic_temperature: 300.0,
            boundary: BoundaryCondition::NonPeriodic,
            enable_dispersion: true,
            d3_reference_path: std::env::var(crate::params::GFN1_D3_REFERENCE_ENV).ok(),
            experimental_d4: false,
            d4_cutoff: D4DispersionOptions::default().cutoff,
            d4_cn_cutoff: D4DispersionOptions::default().cn_cutoff,
            d4_atm: D4DispersionOptions::default().atm_enabled,
            d4_atm_cutoff: D4DispersionOptions::default().atm_cutoff,
            d4_s9: None,
            hamiltonian: HamiltonianOptions::default(),
            external_field: ExternalFieldOptions::default(),
            multipole: false,
            multipole_octupole: false,
            field_multipole: false,
            multipole_third_order: false,
            multipole_charge_order: Vec::new(),
            multipole_secondary_basis: None,
            multipole_order: 0,
            multipole_model: MultipoleModel::default(),
            camm_damp: 1.0,
            camm_aes_scale: 1.0,
            camm_onsite_scale: 1.0,
            camm_onsite_scale_elem: Vec::new(),
            camm_damp_elem: Vec::new(),
            camm_damp_charge: None,
            lr_exchange: false,
            onsite_exchange: false,
            scf_trah: false,
            charge_order: 3,
            dynamic_omega: false,
            spin_polarization: false,
            plus_u: false,
            hubbard_u: Vec::new(),
            plus_u_v: false,
            hubbard_v: Vec::new(),
            hubbard_v_cutoff: 10.0,
            hubbard_u_linear_response: false,
            plus_u_all_d: false,
            plus_u_force_fd_step: 2.0e-3,
            scc_initial_shell_charges: None,
        }
    }
}

impl ElectronicOptions {
    pub fn d4_dispersion_options(&self) -> D4DispersionOptions {
        let defaults = D4DispersionOptions::default();
        D4DispersionOptions {
            cutoff: if self.d4_cutoff > 0.0 {
                self.d4_cutoff
            } else {
                defaults.cutoff
            },
            cn_cutoff: if self.d4_cn_cutoff > 0.0 {
                self.d4_cn_cutoff
            } else {
                defaults.cn_cutoff
            },
            atm_enabled: self.d4_atm,
            atm_cutoff: if self.d4_atm_cutoff > 0.0 {
                self.d4_atm_cutoff
            } else {
                defaults.atm_cutoff
            },
            atm_damping_exponent: defaults.atm_damping_exponent,
            s9: self.d4_s9.unwrap_or(if self.experimental_d4 {
                defaults.s9
            } else {
                0.0
            }),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SpinChannels {
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
}

pub(crate) fn validate_electron_count(nelec: f64, norbitals: usize) -> Result<()> {
    if !nelec.is_finite() {
        return Err(Gfn1Error::InvalidInput(
            "electron count is not finite".to_string(),
        ));
    }
    if nelec < -ELECTRON_COUNT_TOLERANCE {
        return Err(Gfn1Error::InvalidInput(format!(
            "electron count is negative: {nelec}"
        )));
    }
    let capacity = 2.0 * norbitals as f64;
    if nelec > capacity + ELECTRON_COUNT_TOLERANCE {
        return Err(Gfn1Error::InvalidInput(format!(
            "electron count {nelec} exceeds basis capacity {}",
            2 * norbitals
        )));
    }
    Ok(())
}

/// Validate the per-rank multipole×charge cross-term orders ([`ElectronicOptions::multipole_charge_order`])
/// against the physical termination bound `order ≤ 2l+3` for each rank `l` (the rank-`l` on-site
/// self-energy `½ g_l(η(q))(m_l·m_l)` is a degree-`(2l+1)` polynomial in `Δq`, so charge orders above
/// `2l+3` contribute *exactly zero*). An out-of-range order is a hard error rather than a silent
/// truncation, so the user is never misled into thinking a too-high order is active. `max_rank` =
/// the multipole expansion order (`multipole_order`); a cross term for a rank beyond it also errors.
pub(crate) fn validate_multipole_charge_order(orders: &[usize], max_rank: usize) -> Result<()> {
    for (i, &order) in orders.iter().enumerate() {
        let l = i + 1;
        if order < 3 {
            continue; // 0/1/2 ⇒ no cross term for this rank (the base 2nd-order multipole only)
        }
        if l > max_rank {
            return Err(Gfn1Error::InvalidInput(format!(
                "multipole_charge_order requests charge order {order} for multipole rank {l} \
                 (2^{l}-pole), but multipole_order = {max_rank}; raise multipole_order to ≥ {l} \
                 or drop the trailing entries"
            )));
        }
        let bound = 2 * l + 3;
        if order > bound {
            return Err(Gfn1Error::InvalidInput(format!(
                "multipole_charge_order rank {l} (2^{l}-pole): charge order {order} exceeds the \
                 termination bound {bound} (= 2·{l}+3). The rank-{l} on-site self-energy is a \
                 degree-{deg} polynomial in Δq, so charge orders above {bound} contribute exactly \
                 zero — lower the order to ≤ {bound} (do not rely on silent truncation)",
                deg = 2 * l + 1
            )));
        }
    }
    Ok(())
}

pub(crate) fn resolve_spin_channels(
    nelec: f64,
    spin_multiplicity: Option<usize>,
    norbitals: usize,
) -> Result<Option<SpinChannels>> {
    let Some(multiplicity) = spin_multiplicity else {
        return Ok(None);
    };
    if multiplicity == 0 {
        return Err(Gfn1Error::InvalidInput(
            "spin multiplicity must be at least 1".to_string(),
        ));
    }
    let rounded = nelec.round();
    if (nelec - rounded).abs() > ELECTRON_COUNT_TOLERANCE {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin multiplicity {multiplicity} requires an integer electron count; got {nelec}"
        )));
    }
    if rounded < -ELECTRON_COUNT_TOLERANCE {
        return Err(Gfn1Error::InvalidInput(format!(
            "electron count is negative: {nelec}"
        )));
    }
    let ne = rounded as i64;
    let unpaired = (multiplicity - 1) as i64;
    if unpaired > ne {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin multiplicity {multiplicity} requires {unpaired} unpaired electrons, but the system has {ne} electrons"
        )));
    }
    if (ne - unpaired) % 2 != 0 {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin multiplicity {multiplicity} is incompatible with {ne} electrons"
        )));
    }
    let alpha = (ne + unpaired) / 2;
    let beta = (ne - unpaired) / 2;
    if alpha as usize > norbitals {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin multiplicity {multiplicity} needs {alpha} alpha electrons, but the basis has only {norbitals} spatial orbitals"
        )));
    }
    if beta as usize > norbitals {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin multiplicity {multiplicity} needs {beta} beta electrons, but the basis has only {norbitals} spatial orbitals"
        )));
    }
    Ok(Some(SpinChannels {
        alpha: alpha as f64,
        beta: beta as f64,
    }))
}

#[derive(Clone, Debug)]
pub struct ElectronicResult {
    pub basis: BasisSet,
    pub integrals: IntegralMatrices,
    pub h0: Matrix,
    pub fock: Matrix,
    pub density: Matrix,
    pub energy_weighted_density: Matrix,
    pub orbital_energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub electronic_temperature: f64,
    pub fermi_level: f64,
    pub shell_charges: Vec<f64>,
    pub atomic_charges: Vec<f64>,
    pub shell_scc_potential: Vec<f64>,
    pub coordination_numbers: Vec<f64>,
    pub electronic_energy: f64,
    pub repulsion_energy: f64,
    pub isotropic_scc_energy: f64,
    pub third_order_energy: f64,
    pub dispersion_energy: f64,
    pub halogen_energy: f64,
    /// External electric-field interaction energy `sum_i q_i v_ext_i` (0 when no
    /// field is applied).
    pub external_field_energy: f64,
    pub electronic_entropy_term: f64,
    pub total_internal: f64,
    pub total_free: f64,
    /// Mulliken (monopole) dipole moment in atomic units (e * a0), referenced to
    /// the external-field origin (the coordinate origin when no field is set).
    pub dipole: Vec3,
    pub nelec: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Spin-resolved data, populated **only** by the spin-polarized GFN1
    /// ("spGFN1") path ([`crate::spin::run_spin_polarized`]) for an open-shell
    /// system. `None` for every restricted (closed-shell or non-spin-polarized)
    /// calculation, so all existing consumers are unaffected. See
    /// [`crate::spin::SpinResolved`].
    pub spin: Option<crate::spin::SpinResolved>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyTerms {
    pub total_free: f64,
    pub total_internal: f64,
    pub repulsion: f64,
    pub electronic: f64,
    pub isotropic_scc: f64,
    pub third_order: f64,
    pub dispersion: f64,
    pub halogen: f64,
    pub external_field: f64,
    pub electronic_entropy: f64,
}

impl EnergyTerms {
    pub fn named_values(&self) -> [(&'static str, f64); 10] {
        [
            ("total_free", self.total_free),
            ("total_internal", self.total_internal),
            ("repulsion", self.repulsion),
            ("electronic", self.electronic),
            ("isotropic_scc", self.isotropic_scc),
            ("third_order", self.third_order),
            ("dispersion", self.dispersion),
            ("halogen", self.halogen),
            ("external_field", self.external_field),
            ("electronic_entropy", self.electronic_entropy),
        ]
    }
}

impl ElectronicResult {
    pub fn energy_terms(&self) -> EnergyTerms {
        EnergyTerms {
            total_free: self.total_free,
            total_internal: self.total_internal,
            repulsion: self.repulsion_energy,
            electronic: self.electronic_energy,
            isotropic_scc: self.isotropic_scc_energy,
            third_order: self.third_order_energy,
            dispersion: self.dispersion_energy,
            halogen: self.halogen_energy,
            external_field: self.external_field_energy,
            electronic_entropy: self.electronic_entropy_term,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Gfn1Calculator {
    pub params: Gfn1Parameters,
    pub options: ElectronicOptions,
}

impl Gfn1Calculator {
    pub fn new(params: Gfn1Parameters) -> Self {
        Self {
            params,
            options: ElectronicOptions::default(),
        }
    }

    pub fn with_options(params: Gfn1Parameters, options: ElectronicOptions) -> Self {
        Self { params, options }
    }

    pub fn calculate(&self, system: &PeriodicSystem) -> Result<ElectronicResult> {
        run_electronic(system, &self.params, self.options.clone())
    }
}

/// **Multipole rank-continuation ("rank ladder") SCC.** Converges the multipole SCC one rank
/// at a time — `base_rank`, then `base_rank+1`, … up to `target_rank` — warm-starting each
/// stage's shell charges from the previous (lower-rank) converged result. A cold high-rank
/// (16-pole+) multipole SCC can struggle because the monopole↔high-multipole coupling
/// oscillates; staging lets the well-conditioned low ranks converge first so only the newly
/// added rank has to relax. Ranks ≤ 3 use the legacy dipole+quad(+octupole) path; ranks ≥ 4
/// use the generic arbitrary-rank path (`multipole_order = rank`). Returns the final
/// (`target_rank`) converged result; it is the same SCF solution a direct high-rank run would
/// reach, just more robustly. Non-periodic; the periodic SCF (A2) will reuse this strategy.
pub fn run_electronic_rank_ladder(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    base_rank: usize,
    target_rank: usize,
) -> Result<ElectronicResult> {
    if base_rank > target_rank {
        return Err(Gfn1Error::InvalidInput(format!(
            "rank ladder base_rank {base_rank} exceeds target_rank {target_rank}"
        )));
    }
    let mut prev_charges: Option<Vec<f64>> = None;
    let mut result: Option<ElectronicResult> = None;
    for rank in base_rank..=target_rank {
        let mut opt = options.clone();
        opt.multipole = true;
        if rank <= 3 {
            opt.multipole_order = 0; // legacy dipole+quad
            opt.multipole_octupole = rank >= 3;
        } else {
            opt.multipole_order = rank; // generic ranks 1..=rank (16-pole at 4, 32-pole at 5, …)
            opt.multipole_octupole = false;
        }
        opt.scc_initial_shell_charges = prev_charges.take();
        let r = run_electronic(system, params, opt)?;
        prev_charges = Some(r.shell_charges.clone());
        result = Some(r);
    }
    result.ok_or_else(|| Gfn1Error::InvalidInput("rank ladder produced no stage".to_string()))
}

pub fn run_electronic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: ElectronicOptions,
) -> Result<ElectronicResult> {
    if options.plus_u
        && (system.lattice.is_some() || options.boundary != BoundaryCondition::NonPeriodic)
    {
        return Err(Gfn1Error::InvalidInput(
            "DFT+U/+U+V is implemented for non-periodic systems only".to_string(),
        ));
    }
    if options.experimental_d4
        && (system.lattice.is_some() || options.boundary != BoundaryCondition::NonPeriodic)
    {
        return Err(Gfn1Error::InvalidInput(
            "experimental D4 dispersion is implemented for non-PBC systems only".to_string(),
        ));
    }
    if system.lattice.is_some() || options.boundary != BoundaryCondition::NonPeriodic {
        // Periodic systems are routed to the Gamma-point / k-point PBC path.
        return crate::pbc::run_electronic_pbc(system, params, &options);
    }
    if options.spin_polarization || options.plus_u {
        // Spin-polarized GFN1 ("spGFN1") and/or DFT+U/+U+V both run through the spin module: it
        // decides open-shell (spin-unrestricted SCC + W spin term + any +U) vs closed-shell (with
        // +U the closed-shell case stays on the unrestricted path at zero magnetization so the +U
        // term still applies; without +U it delegates back to the restricted path below for a
        // byte-identical result). `plus_u` thus implies the spin machinery.
        return crate::spin::run_spin_polarized(system, params, options);
    }
    let _profile = crate::profile::scope("electronic.nonpbc.total");
    let basis = {
        let _profile = crate::profile::scope("electronic.basis");
        BasisSet::build(
            system,
            params,
            BasisOptions {
                nprim: options.nprim,
            },
        )?
    };
    let core = {
        let _profile = crate::profile::scope("electronic.h0_integrals");
        build_h0(system, &basis, params, &options.hamiltonian)?
    };
    run_scc_with_core(system, params, basis, core, options)
}

fn run_scc_with_core(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: BasisSet,
    core: HamiltonianCore,
    options: ElectronicOptions,
) -> Result<ElectronicResult> {
    let _profile = crate::profile::scope("electronic.scf.total");
    if options.experimental_d4 && options.lr_exchange {
        return Err(Gfn1Error::InvalidInput(
            "experimental D4 dispersion is not yet wired into the long-range exchange SCF driver"
                .to_string(),
        ));
    }
    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    if basis.is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "cannot run GFN1 SCC for an empty basis".to_string(),
        ));
    }
    validate_electron_count(nelec, basis.len())?;
    let spin_channels = resolve_spin_channels(nelec, options.spin_multiplicity, basis.len())?;

    let shell_model = {
        let _profile = crate::profile::scope("electronic.shell_model");
        let mut m = ShellChargeModel::build(system, &basis, params)?;
        m.charge_order = options.charge_order.max(3);
        m
    };
    let amat = {
        let _profile = crate::profile::scope("electronic.coulomb_matrix");
        effective_coulomb_matrix(system, &basis, &shell_model)
    };
    let orth = {
        let _profile = crate::profile::scope("electronic.lowdin_orthogonalizer");
        lowdin_orthogonalizer(&core.integrals.overlap, options.eigen_tolerance)?
    };
    let repulsion = {
        let _profile = crate::profile::scope("electronic.repulsion_energy");
        repulsion_energy(system, params)?
    };
    let d4_active = options.enable_dispersion && options.experimental_d4;
    let d4_options = options.d4_dispersion_options();
    let d4_coordination = if d4_active {
        let _profile = crate::profile::scope("electronic.d4_coordination");
        crate::coordination::coordination_with_derivatives(
            system,
            crate::coordination::CoordinationOptions {
                cutoff: d4_options.cn_cutoff,
                ..crate::coordination::CoordinationOptions::default()
            },
        )?
        .cn
    } else {
        Vec::new()
    };
    let d4_pairs = if d4_active {
        let _profile = crate::profile::scope("electronic.d4_pairs");
        d4_dispersion_pairs(system, d4_options)?
    } else {
        Vec::new()
    };
    let d4_atm = if d4_active {
        let _profile = crate::profile::scope("electronic.d4_atm_geometry");
        d4_dispersion_atm_geometry(system, params, d4_options)?
    } else {
        crate::dispersion::D4AtmGeometry::default()
    };
    let mut dispersion = if options.enable_dispersion && !options.experimental_d4 {
        let _profile = crate::profile::scope("electronic.dispersion_energy");
        dispersion_energy(system, params, options.d3_reference_path.as_deref())?
    } else {
        0.0
    };
    let halogen = {
        let _profile = crate::profile::scope("electronic.halogen_energy");
        halogen_energy(system)?
    };

    if options.external_field.magnetic_field.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "external magnetic field is a foothold only and not yet wired into the SCC; \
             see the `magnetic` module"
                .to_string(),
        ));
    }
    // Geometry-fixed external electric-field site potential v_ext_i = -E·R_i.
    let external_shell_potential =
        electric_shell_potential(&options.external_field, system, &basis);

    // mDFTB2 multipole context (geometry-only): the atomic Klopman-Ohno hardness (the
    // s-shell hardness of each atom) and the atomic positions. The multipole field is
    // rebuilt from the density every iteration so it relaxes self-consistently.
    let nat = system.atoms.len();
    let multipole_ctx: Option<(Vec<f64>, Vec<crate::math::Vec3>)> = if options.multipole {
        if system.lattice.is_some() {
            return Err(Gfn1Error::InvalidInput(
                "mDFTB2 multipole correction is implemented for non-periodic systems only"
                    .to_string(),
            ));
        }
        let hardness: Vec<f64> = (0..nat)
            .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
            .collect();
        let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|at| at.position).collect();
        Some((hardness, pos))
    } else {
        None
    };
    let mut mp_energy = 0.0_f64;

    // Experimental long-range Fock exchange (MFX) context (geometry-fixed): the AO×AO long-range
    // kernel `Γ` and the neutral-atom reference density `P0`. The exchange Fock `K[P_prev − P0]` is
    // added to the SCC Fock each iteration (density-matrix self-consistency) and its energy
    // `½Tr[ΔP·K[ΔP]]` to the total. Non-periodic. See [`crate::exchange`].
    let exchange_ctx: Option<(Matrix, Matrix, Option<crate::exchange::OnsiteExchangeCache>)> =
        if options.lr_exchange {
            if system.lattice.is_some() {
                return Err(Gfn1Error::InvalidInput(
                    "long-range Fock exchange (MFX) is implemented for non-periodic systems only"
                        .to_string(),
                ));
            }
            let scheme = crate::coulomb::OmegaScheme::HardnessPairwise;
            let hardness: Vec<f64> = (0..nat)
                .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
                .collect();
            let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|at| at.position).collect();
            // LocalGeometry (dynamic ω): per-atom size factor s_A = (1+CN_A)^(−1/3) from the GFN1
            // coordination number; `None` ⇒ geometry-independent HardnessPairwise (s ≡ 1).
            let s_factor: Option<Vec<f64>> = if options.dynamic_omega {
                let cn = crate::coordination::coordination_with_derivatives(
                    system,
                    crate::coordination::CoordinationOptions {
                        cutoff: options.hamiltonian.coordination_cutoff,
                        ..crate::coordination::CoordinationOptions::default()
                    },
                )?
                .cn;
                Some(
                    cn.iter()
                        .map(|&c| crate::coulomb::local_size_factor_from_cn(c).0)
                        .collect(),
                )
            } else {
                None
            };
            let gamma = match &s_factor {
                Some(s) => {
                    crate::exchange::lr_exchange_gamma_matrix_local(&basis, nat, &pos, &hardness, s)
                }
                None => {
                    crate::exchange::lr_exchange_gamma_matrix(&basis, nat, &pos, &hardness, scheme)
                }
            };
            let p0 = crate::exchange::neutral_atom_reference_density(&basis);
            // On-site Fock exchange (OFX): build the per-element one-center ERI cache at the same
            // on-site screening ω_AA the MFX γ^lr uses (= ω_pair(scheme, η_A, η_A), or η_A/s_A under
            // LocalGeometry). Geometry-fixed.
            let onsite_cache = if options.onsite_exchange {
                let omega_aa: Vec<f64> = match &s_factor {
                    Some(s) => hardness
                        .iter()
                        .zip(s.iter())
                        .map(|(&eta, &sa)| crate::coulomb::omega_local_geometry(eta, sa))
                        .collect(),
                    None => hardness
                        .iter()
                        .map(|&eta| crate::coulomb::omega_pair(scheme, eta, eta))
                        .collect(),
                };
                Some(crate::exchange::OnsiteExchangeCache::build(
                    &basis, nat, &omega_aa,
                ))
            } else {
                None
            };
            Some((gamma, p0, onsite_cache))
        } else {
            None
        };
    let mut exchange_energy = 0.0_f64;

    // Stage 5: richer secondary-basis on-site moments. When a secondary basis is supplied (and
    // the multipole correction is on), the on-site dipole/quad moment integrals are rebuilt over
    // the node-correct secondary AOs (overlap stays primary). Every multipole moment read below
    // then uses `mp_ints`, so the energy / Fock / gradient stay mutually consistent.
    // Build the secondary (node-correct) AOs once when a secondary basis is supplied + the
    // multipole correction is on. Reused for BOTH the legacy rank-1/2 secondary moment integrals
    // (`mp_ints`) and — new in v0.2.1 — the generic arbitrary-rank on-site moment cache, so the
    // generic path (order ≥ 4 / cross terms) consumes the secondary basis at every rank too.
    let secondary_aos: Option<Vec<crate::basis::AOBasisFunction>> =
        match (&multipole_ctx, &options.multipole_secondary_basis) {
            (Some(_), Some(sec)) => Some(crate::magnetic::build_secondary_aos(&basis, system, sec)),
            _ => None,
        };
    let secondary_moment_ints: Option<IntegralMatrices> = match (&multipole_ctx, &secondary_aos) {
        (Some((_, pos)), Some(sec_aos)) => Some(crate::multipole::secondary_moment_integrals(
            &core.integrals,
            &basis,
            nat,
            pos,
            sec_aos,
        )),
        _ => None,
    };
    let mp_ints: &IntegralMatrices = secondary_moment_ints.as_ref().unwrap_or(&core.integrals);

    // Third-order on-site multipole cross terms: requires the multipole correction. The
    // per-atom Hubbard derivative Γ_A (gam3) drives the parameter-free α/β coefficients.
    let multipole_third_order = multipole_ctx.is_some() && options.multipole_third_order;
    // Per-rank multipole×charge cross terms (generalised arbitrary rank × arbitrary charge order):
    // active when an order vector is supplied and the multipole correction is on. It forces the
    // generic arbitrary-rank path (below) and reuses the same per-atom Γ_A (gam3). Validate the
    // requested orders against the rank-`l` termination bound `2l+3` (error, never silent truncate).
    let multipole_charge_cross =
        multipole_ctx.is_some() && !options.multipole_charge_order.is_empty();
    if multipole_charge_cross {
        if options.multipole_order < 1 {
            return Err(Gfn1Error::InvalidInput(
                "multipole_charge_order requires multipole_order ≥ 1 (set it to the maximum \
                 multipole rank the cross terms should cover)"
                    .to_string(),
            ));
        }
        validate_multipole_charge_order(&options.multipole_charge_order, options.multipole_order)?;
    }
    // CAMM-on-mDFTB2 (v0.4.2): the off-site anisotropy is replaced by a GFN2-style AES term on
    // cumulative atomic multipole moments. v1 supports only the base dipole+quad correction in the
    // standard charge-vector SCC; reject the combinations that route through the generic /
    // density-matrix / on-site-cross paths it does not yet cover.
    // `camm_params = (per-atom κ, s_AES, per-atom s_onsite)`.
    let camm_params: Option<(Vec<f64>, f64, Vec<f64>)> =
        if multipole_ctx.is_some() && options.multipole_model == MultipoleModel::CammOnMdftb2 {
            if !(options.camm_damp > 0.0) {
                return Err(Gfn1Error::InvalidInput(
                    "camm_damp (CAMM range factor κ) must be > 0".to_string(),
                ));
            }
            if options.camm_aes_scale < 0.0 {
                return Err(Gfn1Error::InvalidInput(
                    "camm_aes_scale (s_AES) must be ≥ 0".to_string(),
                ));
            }
            if options.camm_onsite_scale < 0.0 {
                return Err(Gfn1Error::InvalidInput(
                    "camm_onsite_scale (s_onsite) must be ≥ 0".to_string(),
                ));
            }
            if options.camm_onsite_scale_elem.iter().any(|&(_, s)| s < 0.0) {
                return Err(Gfn1Error::InvalidInput(
                    "camm_onsite_scale_elem s_onsite values must be ≥ 0".to_string(),
                ));
            }
            if options.camm_damp_elem.iter().any(|&(_, k)| !(k > 0.0)) {
                return Err(Gfn1Error::InvalidInput(
                    "camm_damp_elem κ values must be > 0".to_string(),
                ));
            }
            if options.multipole_octupole
                || options.multipole_order >= 4
                || multipole_third_order
                || multipole_charge_cross
                || options.field_multipole
                || options.multipole_secondary_basis.is_some()
                || options.lr_exchange
                || options.scf_trah
            {
                return Err(Gfn1Error::InvalidInput(
                    "multipole_model = camm_on_mdftb2 (v0.4.2) supports only the base dipole+quad \
                     correction in the standard SCC: disable multipole_octupole, multipole_order \
                     (≥4), multipole_third_order, multipole_charge_order, field_multipole, \
                     multipole_secondary_basis, lr_exchange, and scf_trah"
                        .to_string(),
                ));
            }
            // Per-atom κ: element override if present, else the global camm_damp.
            let kappa: Vec<f64> = system
                .atoms
                .iter()
                .map(|atom| {
                    options
                        .camm_damp_elem
                        .iter()
                        .find(|&&(z, _)| z == atom.z)
                        .map(|&(_, k)| k)
                        .unwrap_or(options.camm_damp)
                })
                .collect();
            // Per-atom on-site penalty scale: element override if present, else global s_onsite.
            let onsite: Vec<f64> = system
                .atoms
                .iter()
                .map(|atom| {
                    options
                        .camm_onsite_scale_elem
                        .iter()
                        .find(|&&(z, _)| z == atom.z)
                        .map(|&(_, s)| s)
                        .unwrap_or(options.camm_onsite_scale)
                })
                .collect();
            Some((kappa, options.camm_aes_scale, onsite))
        } else {
            None
        };
    let gam3_atom: Vec<f64> = if multipole_third_order || multipole_charge_cross {
        (0..nat)
            .map(|a| shell_model.hubbard_derivs[shell_model.atom_offsets[a]])
            .collect()
    } else {
        Vec::new()
    };

    // Stage 3 field–dipole coupling: active only with the multipole correction on, the
    // `field_multipole` flag set, and an external electric field present. The coupling Fock
    // `∂(−E·Σ d_A)/∂P` is constant per geometry (the external field does not depend on the
    // SCC moments), so build it once and add it to the multipole Fock every iteration.
    let field_dipole: Option<crate::math::Vec3> = if options.field_multipole {
        multipole_ctx
            .as_ref()
            .and(options.external_field.electric_field)
    } else {
        None
    };
    let field_dipole_fock: Option<Matrix> =
        field_dipole.map(|f| crate::multipole::field_dipole_fock(&basis, nat, mp_ints, f));

    // SCC mixing vector. With the mDFTB2 correction on it is the joint tblite-style vector
    // [shell charges (nsh) | atomic dipole+quadrupole moments (MOMENT_STRIDE*nat)], so the
    // Broyden mixer captures the monopole/multipole coupling and the multipole self-
    // consistency converges robustly. Off, it is just the nsh shell charges (unchanged path).
    let nsh = basis.shells.len();
    // Arbitrary-rank generic multipole path: when `multipole_order ≥ 4` and the correction is on,
    // the unified generic path (ranks 1..=L) supersedes the legacy dipole/quad/octupole blocks.
    // `None` for all legacy configurations → the legacy branches below run byte-identically.
    // The per-rank multipole×charge cross terms (`multipole_charge_cross`) also require the generic
    // path (with `multipole_order ≥ 1`, validated above), so force it on even for ranks 2/3.
    let generic_rank: Option<usize> =
        if multipole_ctx.is_some() && (options.multipole_order >= 4 || multipole_charge_cross) {
            Some(options.multipole_order)
        } else {
            None
        };
    let moment_len = match (&multipole_ctx, generic_rank) {
        (Some(_), Some(l)) => crate::multipole::generic_moment_stride(l) * nat,
        (Some(_), None) => crate::multipole::MOMENT_STRIDE * nat,
        (None, _) => 0,
    };
    // Experimental octupole block, appended after the dipole+quad moments in the joint
    // Broyden vector (requires the multipole correction; off by default; folded into the generic
    // path when that is active).
    let octupole = multipole_ctx.is_some() && options.multipole_octupole && generic_rank.is_none();
    let octu_len = if octupole {
        crate::multipole::OCTU_STRIDE * nat
    } else {
        0
    };
    // The on-site octupole AO integrals are geometry-fixed; build them once and reuse across all
    // SCC iterations (instead of recomputing the costly rank-3 integrals every iteration).
    let octu_cache: Option<crate::multipole::OnsiteOctupoleCache> = if octupole {
        let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|at| at.position).collect();
        Some(crate::multipole::OnsiteOctupoleCache::build(
            &basis, nat, &pos,
        ))
    } else {
        None
    };
    // Arbitrary-rank generic path: the on-site rank-`l` AO moment integrals are geometry-fixed, so
    // build them once here and reuse across all SCC iterations (the heavy per-iteration integral
    // recompute would otherwise dominate the high-rank SCC).
    let generic_moment_cache: Option<crate::multipole::OnsiteMomentCache> = generic_rank.map(|l| {
        let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|at| at.position).collect();
        crate::multipole::OnsiteMomentCache::build_with_aos(
            &basis,
            nat,
            &pos,
            l,
            secondary_aos.as_deref(),
        )
    });
    let mix_len = nsh + moment_len + octu_len;
    let mut v_mixed = vec![0.0; mix_len];
    // Warm-start: seed the shell charges from a previous (lower-rank) converged SCC so the
    // monopole/low-multipole channels begin near their solution (multipole rank continuation).
    if let Some(q0) = &options.scc_initial_shell_charges {
        if q0.len() == nsh {
            v_mixed[0..nsh].copy_from_slice(q0);
        }
    }
    let mut last_result: Option<SccStep> = None;
    let mut converged = false;
    let mut final_rms = f64::INFINITY;
    let mut last_scc_energy: Option<f64> = None;
    let mut iterations = 0usize;
    // The long-range exchange Fock depends on the off-diagonal density, a response the GFN1-tuned
    // charge mixer does not model. Broyden's quasi-Newton charge step then overshoots
    // *unpredictably* (it can diverge at one mixing yet converge at another), so when exchange is
    // on we fall back to plain **linear** charge mixing with a conservative step — monotone and
    // robust (the commutator DIIS on the exchange density still accelerates that channel). A
    // unified commutator DIIS on the full Fock would restore quasi-Newton speed; documented future
    // upgrade.
    let base_mixing = if options.lr_exchange {
        options.mixing.clamp(0.01, 1.0).min(EXCHANGE_MAX_MIXING)
    } else {
        options.mixing.clamp(0.01, 1.0)
    };
    // Resolve the accelerator: the legacy `scc_broyden = false` selects linear
    // mixing only when the accelerator is left at its default.
    let base_accelerator = if options.lr_exchange {
        SccAccelerator::Linear
    } else {
        match options.scc_accelerator {
            SccAccelerator::Broyden if !options.scc_broyden => SccAccelerator::Linear,
            other => other,
        }
    };
    // Per-attempt SCC controls (the robust-fallback ladder below overwrites these on a stall).
    let mut mixing = base_mixing;
    let mut accelerator = base_accelerator;
    let mut cur_level_shift = options.level_shift;
    let mut cur_etemp = options.electronic_temperature;
    let mut broyden = BroydenMixer::new(mix_len, options.scc_broyden_size.max(2), mixing);
    let mut cdiis = CdiisMixer::new(options.scc_broyden_size.max(2).min(20));
    let mut prev_density: Option<Matrix> = None;
    // Density-matrix self-consistency for the long-range exchange: the Fock depends on the full
    // density (off-diagonal), which the charge mixers cannot stabilize, so the density used to build
    // the exchange Fock is extrapolated by commutator DIIS (engaged only when `lr_exchange` is on).
    let mut exchange_diis = CommutatorDiis::new(options.scc_broyden_size.max(2).min(20));
    let mut exchange_density: Option<Matrix> = None;

    // Robust density-matrix SCF for **any exchange** SCC (with or without the multipole correction):
    // full-Fock commutator ADIIS→C-DIIS→TRAH driven from a single trial density. The off-diagonal
    // exchange Fock destabilises the projected charge mixer, so this supersedes it (charges, exchange
    // **and** the multipole moments are all rebuilt from one P each iteration). The exchange-off path
    // (plain GFN1 / multipole-only) still uses the charge-vector loop below.
    if let Some((gamma, p0, onsite_cache)) = &exchange_ctx {
        // The multipole correction (when on) rides into the robust driver via this context: its Fock
        // is rebuilt from the trial density each iteration, so the density-derived atomic moments
        // relax with the density (no separate moment mixing). `None` ⇒ pure exchange SCC (unchanged).
        let mp_scf = multipole_ctx.as_ref().map(|(hardness, pos)| MultipoleScf {
            basis: &basis,
            nat,
            hardness,
            pos,
            mp_ints,
            shell_model: &shell_model,
            generic_rank,
            generic_moment_cache: generic_moment_cache.as_ref(),
            octupole,
            octu_cache: octu_cache.as_ref(),
            field_dipole,
            field_dipole_fock: field_dipole_fock.as_ref(),
            third_order: multipole_third_order,
            gam3: &gam3_atom,
            charge_order: &options.multipole_charge_order,
        });
        let mp_ref = mp_scf.as_ref();
        // Driver selection for the exchange-augmented SCC:
        //  • `scf_trah` → the second-order TRAH driver directly;
        //  • otherwise **AutoTRAH**: try the (fast) commutator-DIIS driver first, and if it fails to
        //    converge, automatically fall back to TRAH (which minimises the energy directly over
        //    orbital rotations and is robust where DIIS on the off-diagonal exchange Fock stalls).
        let oc = onsite_cache.as_ref();
        let run_trah = |initial_mo: Option<&Matrix>| {
            run_trah_exchange_scf(
                &basis,
                nat,
                &core.integrals.overlap,
                &core.h0,
                &orth,
                &shell_model,
                &amat,
                nelec,
                spin_channels,
                options.electronic_temperature,
                options.eigen_tolerance,
                external_shell_potential.as_deref(),
                gamma,
                p0,
                oc,
                mp_ref,
                options.max_scc,
                options.energy_tolerance,
                options.charge_tolerance,
                initial_mo,
            )
        };
        let (step, ex_e, iters, conv, exch_rms) = if options.scf_trah {
            run_trah(None)? // explicit request: second-order Newton from the core-H guess
        } else {
            // First-order ADIIS→C-DIIS→ADIIS-fallback driver — robust and cheap for the bulk of the
            // descent, and it converges gapped molecules outright.
            let diis = run_exchange_scf(
                &basis,
                nat,
                &core.integrals.overlap,
                &core.h0,
                &orth,
                &shell_model,
                &amat,
                nelec,
                spin_channels,
                options.electronic_temperature,
                options.eigen_tolerance,
                external_shell_potential.as_deref(),
                gamma,
                p0,
                onsite_cache.as_ref(),
                mp_ref,
                options.max_scc,
                options.energy_tolerance,
                options.charge_tolerance,
            )?;
            if diis.3 {
                diis
            } else {
                // Not tight yet (e.g. ADIIS grinding the linear tail of a near-degenerate metal):
                // hand the near-converged density to the second-order TRAH for a quadratically
                // convergent polish (continuation — no size cap, since it needs only a few steps).
                let mo = diis.0.mo_coeff.clone();
                let trah = run_trah(Some(&mo))?;
                // TRAH is a monotone trust-region energy minimisation from the DIIS density, so its
                // result is never worse; report it with the combined iteration budget.
                (trah.0, trah.1, diis.2 + trah.2, trah.3, trah.4)
            }
        };
        // Multipole correction energy at the converged density (the robust driver tracks only the
        // exchange energy in `ex_e`; the multipole energy is recomputed once here from the result).
        if let Some(mp) = mp_ref {
            mp_energy = mp.fock_energy(&step.density, &step.shell_charges)?.1;
        }
        exchange_energy = ex_e;
        iterations = iters;
        converged = conv;
        final_rms = exch_rms; // honest residual for the "did not converge" message (was a stale ∞)
        last_result = Some(step);
    } else {
        // **Robust-fallback ladder** (pure convergence accelerator — never changes the energy of a
        // system that already converges). The first attempt runs the user's exact scheme (Broyden /
        // the resolved accelerator, the requested mixing, level shift and electronic temperature, and
        // the requested iteration budget), so every system that converges today exits on attempt 0
        // byte-for-byte. Only when that primary scheme *stalls* do the later rungs engage.
        //
        // Failure mode (diagnosed on the CAMM `halogen-allgrad` small-gap TM complexes): the HOMO–LUMO
        // gap collapses toward zero, the cold-Fermi occupations flip discontinuously between near-
        // degenerate frontier orbitals, and Broyden's quasi-Newton charge step overshoots — the charges
        // slosh and never settle. The later rungs cure it without touching the energy of converging
        // systems:
        //   • rungs 1–2: **monotone linear charge mixing + a virtual level shift** (which lifts the
        //     virtuals so the occupied manifold stops fluctuating). These keep the user's electronic
        //     temperature, so the energy they converge to is the *true* SCC fixed point (no smearing
        //     artifact). They need many iterations (the linear tail of a near-degenerate metal), hence
        //     the enlarged per-rung budget.
        //   • rung 3 (last resort): **Broyden + a raised electronic temperature** — Fermi-smears the
        //     near-degenerate frontier so the response is well-conditioned and Broyden converges fast.
        //     This is the only rung that perturbs the energy (via fractional occupations / entropy), and
        //     it runs only for systems that have *no* converged result at all on the energy-preserving
        //     rungs — for them any converged number strictly beats a hard SCF failure.
        // (accelerator, mixing, level_shift, etemp, max_iter). Rung 0 is overwritten below with the
        // user's exact settings; the literals on its row are placeholders.
        let robust_budget = options.max_scc.max(800);
        let ladder: [(SccAccelerator, f64, f64, f64, usize); 4] = [
            (base_accelerator, base_mixing, options.level_shift, options.electronic_temperature, options.max_scc),
            (SccAccelerator::Linear, 0.20, options.level_shift.max(0.20), options.electronic_temperature, robust_budget),
            (SccAccelerator::Linear, 0.10, options.level_shift.max(0.10), options.electronic_temperature, robust_budget),
            (SccAccelerator::Broyden, base_mixing, 0.0, options.electronic_temperature.max(3000.0), options.max_scc.max(250)),
        ];
        'attempts: for (attempt, &(acc, mix, lshift, etemp, rung_max_scc)) in ladder.iter().enumerate() {
            // Configure this attempt. Rung 0 == the user's exact scheme (byte-identical default path).
            accelerator = acc;
            mixing = mix;
            cur_level_shift = lshift;
            cur_etemp = etemp;
            if attempt > 0 {
                // Reset all mixer state and the trial vector for a clean restart from the initial guess.
                v_mixed = vec![0.0; mix_len];
                if let Some(q0) = &options.scc_initial_shell_charges {
                    if q0.len() == nsh {
                        v_mixed[0..nsh].copy_from_slice(q0);
                    }
                }
                broyden = BroydenMixer::new(mix_len, options.scc_broyden_size.max(2), mixing);
                cdiis = CdiisMixer::new(options.scc_broyden_size.max(2).min(20));
                prev_density = None;
                last_scc_energy = None;
                last_result = None;
                final_rms = f64::INFINITY;
                if std::env::var("GFN1_SCC_DEBUG").is_ok() {
                    eprintln!(
                        "[SCC] --- fallback rung {attempt}: accel={acc:?} mixing={mix} \
                         level_shift={lshift} etemp={etemp} max_scc={rung_max_scc} ---"
                    );
                }
            }
        for iter in 1..=rung_max_scc {
            let _profile = crate::profile::scope("electronic.scf.iteration");
            iterations = iter;
            let q_shell = &v_mixed[0..nsh];
            let d4_shell_potential = if d4_active {
                let _profile = crate::profile::scope("electronic.scf.d4_energy_potential");
                let q_atom = shell_model.atomic_charges(&basis, q_shell);
                let d4 = d4_dispersion_energy_potential_with_cn_pairs_and_atm(
                    system,
                    params,
                    &q_atom,
                    &d4_coordination,
                    &d4_pairs,
                    &d4_atm,
                    d4_options,
                )?;
                Some(atomic_potential_to_shell_potential(
                    &shell_model,
                    &d4.atomic_potential,
                ))
            } else {
                None
            };
            // Build the mDFTB2 multipole Fock from the *mixed* moments (tblite-style multipole
            // SCF): the moments are mixed jointly with the charges below, so the field relaxes
            // self-consistently rather than chasing the unmixed output density.
            let mp_fock: Option<Matrix> = if let Some((hardness, pos)) = &multipole_ctx {
                if let Some(l) = generic_rank {
                    // Arbitrary-rank generic multipole Fock from the mixed moments (ranks 1..=L) + the
                    // monopole Δq from the mixed shell charges. Supersedes the dipole/quad/octupole blocks.
                    let mut moments = crate::multipole::unpack_generic_moments(
                        &v_mixed[nsh..nsh + moment_len],
                        nat,
                        l,
                    );
                    let gfn1_atomic = shell_model.atomic_charges(&basis, q_shell);
                    let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
                    for (a, m) in moments.iter_mut().enumerate() {
                        m[0] = vec![qm[a]];
                    }
                    // Combined generic multipole + per-rank charge-cross Fock from the *mixed* moments +
                    // mixed monopole Δq (one shared shift assembly; the cross terms relax self-consistently).
                    // The cross block is skipped internally when `multipole_charge_order` is empty.
                    let mut fock = crate::multipole::multipole_fock_generic_with_cross(
                        &basis,
                        nat,
                        hardness,
                        &gam3_atom,
                        pos,
                        mp_ints,
                        &moments,
                        &qm,
                        &options.multipole_charge_order,
                        l,
                        generic_moment_cache.as_ref(),
                    )
                    .fock;
                    if let Some(fdf) = &field_dipole_fock {
                        for i in 0..basis.len() {
                            for j in 0..basis.len() {
                                fock[(i, j)] += fdf[(i, j)];
                            }
                        }
                    }
                    Some(fock)
                } else {
                    let moments =
                        crate::multipole::unpack_moments(&v_mixed[nsh..nsh + moment_len], nat);
                    let gfn1_atomic = shell_model.atomic_charges(&basis, q_shell);
                    let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
                    let mut fock = if let Some((kappa, scale, onsite)) = &camm_params {
                        // CAMM-on-mDFTB2: off-site GFN2-AES on the (mixed) cumulative moments +
                        // mDFTB on-site penalty; the mDFTB off-site Ohno multipole is not built.
                        // Charge-dependent κ (if set): κ_A = κ₀/(1+γ Δq_A²) from the mixed charges.
                        let dyn_kappa: Vec<f64>;
                        let eff_kappa: &[f64] = if let Some((k0, g)) = options.camm_damp_charge {
                            dyn_kappa = qm.iter().map(|&q| (k0 / (1.0 + g * q * q)).max(0.05)).collect();
                            &dyn_kappa
                        } else {
                            kappa
                        };
                        crate::multipole::camm_aes_energy_fock(
                            &basis, nat, hardness, pos, mp_ints, &moments, &qm, eff_kappa, *scale,
                            onsite,
                        )
                        .fock
                    } else {
                        crate::multipole::multipole_fock_from_moments(
                            &basis, nat, hardness, pos, mp_ints, &moments, &qm,
                        )
                        .fock
                    };
                    if octupole {
                        // Add the octupole Fock built from the mixed octupole + dipole/quad moments.
                        let octu = crate::multipole::unpack_octu(&v_mixed[nsh + moment_len..], nat);
                        let of = crate::multipole::octupole_fock_from_moments(
                            &basis,
                            nat,
                            hardness,
                            pos,
                            mp_ints,
                            &qm,
                            &moments.dipole,
                            &moments.quad,
                            &octu,
                            octu_cache.as_ref(),
                        )
                        .fock;
                        for i in 0..basis.len() {
                            for j in 0..basis.len() {
                                fock[(i, j)] += of[(i, j)];
                            }
                        }
                    }
                    if let Some(fdf) = &field_dipole_fock {
                        // Stage 3: add the (constant) external-field–dipole coupling shift.
                        for i in 0..basis.len() {
                            for j in 0..basis.len() {
                                fock[(i, j)] += fdf[(i, j)];
                            }
                        }
                    }
                    if multipole_third_order {
                        // Third-order on-site charge·dipole² / charge·quad² shift, built from the mixed
                        // moments + the mixed monopole Δq (so it relaxes self-consistently).
                        let m3 = crate::multipole::third_order_fock_from_moments(
                            &basis, nat, hardness, &gam3_atom, mp_ints, &moments, &qm,
                        )
                        .fock;
                        for i in 0..basis.len() {
                            for j in 0..basis.len() {
                                fock[(i, j)] += m3[(i, j)];
                            }
                        }
                    }
                    Some(fock)
                }
            } else {
                None
            };
            // Long-range Fock exchange (MFX): add `K[ΔP_prev]` (ΔP from the previous iteration's
            // density; none at iteration 1 → no exchange shift yet) to the SCC Fock, combined with the
            // multipole Fock. Density-matrix self-consistency: the exchange relaxes as the density does.
            let extra_fock: Option<Matrix> = if let Some((gamma, p0, onsite_cache)) = &exchange_ctx
            {
                // ΔP from the commutator-DIIS-extrapolated density (none at iteration 1 → no shift yet).
                let kx = exchange_density.as_ref().map(|p| {
                    let nn = basis.len();
                    let mut dp = Matrix::zeros(nn, nn);
                    for i in 0..nn {
                        for j in 0..nn {
                            dp[(i, j)] = p[(i, j)] - p0[(i, j)];
                        }
                    }
                    let mut k = crate::exchange::mfx_kernel(&dp, &core.integrals.overlap, gamma);
                    // OFX correction (exact one-center exchange − its Mulliken approximation).
                    if let Some(cache) = onsite_cache {
                        let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
                            &basis,
                            nat,
                            &core.integrals.overlap,
                            gamma,
                            &dp,
                            cache,
                        );
                        for i in 0..nn {
                            for j in 0..nn {
                                k[(i, j)] += kofx[(i, j)];
                            }
                        }
                    }
                    k
                });
                match (mp_fock, kx) {
                    (Some(mut f), Some(k)) => {
                        let nn = basis.len();
                        for i in 0..nn {
                            for j in 0..nn {
                                f[(i, j)] += k[(i, j)];
                            }
                        }
                        Some(f)
                    }
                    (Some(f), None) => Some(f),
                    (None, Some(k)) => Some(k),
                    (None, None) => None,
                }
            } else {
                mp_fock
            };
            let step = scc_step(
                &basis,
                &core.integrals.overlap,
                &core.h0,
                &orth,
                &shell_model,
                &amat,
                q_shell,
                nelec,
                spin_channels,
                cur_etemp,
                options.eigen_tolerance,
                external_shell_potential.as_deref(),
                d4_shell_potential.as_deref(),
                cur_level_shift,
                prev_density.as_ref(),
                extra_fock.as_ref(),
            )?;
            let d4_iteration_energy = if d4_active {
                let _profile = crate::profile::scope("electronic.scf.d4_energy");
                let q_atom = shell_model.atomic_charges(&basis, &step.shell_charges);
                d4_dispersion_energy_with_cn_pairs_and_atm(
                    system,
                    params,
                    &q_atom,
                    &d4_coordination,
                    &d4_pairs,
                    &d4_atm,
                    d4_options,
                )?
            } else {
                0.0
            };
            // Output vector [shell charges | atomic moments] from the new density, and the
            // multipole energy at the output density (eqs 16/21).
            let mut v_out = vec![0.0; mix_len];
            v_out[0..nsh].copy_from_slice(&step.shell_charges);
            if let Some((hardness, pos)) = &multipole_ctx {
                if let Some(l) = generic_rank {
                    // Arbitrary-rank output moments (ranks 1..=L, rank-0 = mixed monopole) + energy.
                    let gfn1_atomic = shell_model.atomic_charges(&basis, &step.shell_charges);
                    let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
                    let out_moments = crate::multipole::build_generic_moments(
                        &basis,
                        nat,
                        pos,
                        mp_ints,
                        &step.density,
                        &qm,
                        l,
                        generic_moment_cache.as_ref(),
                    );
                    // Energy-only (no Fock build): the standard ½ M·V multipole energy at the output density.
                    mp_energy = crate::multipole::multipole_energy_generic(
                        nat,
                        hardness,
                        pos,
                        &out_moments,
                        l,
                    );
                    if let Some(f) = field_dipole {
                        // Stage 3: external field's coupling to the atomic dipoles, −E·Σ d_A (rank-1).
                        let mut dsum = crate::math::Vec3::zero();
                        for m in &out_moments {
                            dsum += crate::math::Vec3::new(m[1][0], m[1][1], m[1][2]);
                        }
                        mp_energy += -f.dot(dsum);
                    }
                    if multipole_charge_cross {
                        // Per-rank multipole×charge cross-term energy at the output density (same Δq = qm).
                        mp_energy += crate::multipole::multipole_charge_cross_energy(
                            nat,
                            hardness,
                            &gam3_atom,
                            &out_moments,
                            &qm,
                            &options.multipole_charge_order,
                            l,
                        );
                    }
                    crate::multipole::pack_generic_moments(
                        &out_moments,
                        l,
                        &mut v_out[nsh..nsh + moment_len],
                    );
                } else {
                    let gfn1_atomic = shell_model.atomic_charges(&basis, &step.shell_charges);
                    let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
                    let out_moments = if camm_params.is_some() {
                        crate::multipole::camm_atomic_moments(
                            &basis,
                            nat,
                            mp_ints,
                            &step.density,
                            pos,
                        )
                    } else {
                        crate::multipole::atomic_moments(&basis, nat, mp_ints, &step.density)
                    };
                    mp_energy = if let Some((kappa, scale, onsite)) = &camm_params {
                        // CAMM-on-mDFTB2 off-site AES + on-site penalty energy at the output density.
                        let dyn_kappa: Vec<f64>;
                        let eff_kappa: &[f64] = if let Some((k0, g)) = options.camm_damp_charge {
                            dyn_kappa = qm.iter().map(|&q| (k0 / (1.0 + g * q * q)).max(0.05)).collect();
                            &dyn_kappa
                        } else {
                            kappa
                        };
                        crate::multipole::camm_aes_energy_fock(
                            &basis, nat, hardness, pos, mp_ints, &out_moments, &qm, eff_kappa, *scale,
                            onsite,
                        )
                        .energy
                    } else {
                        crate::multipole::multipole_energy_from_moments(
                            nat,
                            hardness,
                            pos,
                            &out_moments,
                            &qm,
                        )
                    };
                    if let Some(f) = field_dipole {
                        // Stage 3: external field's coupling to the atomic dipoles, −E·Σ d_A.
                        mp_energy += crate::multipole::field_dipole_energy(f, &out_moments);
                    }
                    if multipole_third_order {
                        // Third-order on-site multipole energy at the output density.
                        mp_energy += crate::multipole::third_order_energy_from_moments(
                            nat,
                            hardness,
                            &gam3_atom,
                            &out_moments,
                            &qm,
                        );
                    }
                    crate::multipole::pack_moments(&out_moments, &mut v_out[nsh..nsh + moment_len]);
                    if octupole {
                        let out_octu = crate::multipole::atomic_octupole_moments(
                            &basis,
                            nat,
                            pos,
                            mp_ints,
                            &step.density,
                            octu_cache.as_ref(),
                        );
                        mp_energy += crate::multipole::octupole_fock_from_moments(
                            &basis,
                            nat,
                            hardness,
                            pos,
                            mp_ints,
                            &qm,
                            &out_moments.dipole,
                            &out_moments.quad,
                            &out_octu,
                            octu_cache.as_ref(),
                        )
                        .energy;
                        crate::multipole::pack_octu(&out_octu, &mut v_out[nsh + moment_len..]);
                    }
                }
            }
            // Long-range Fock exchange energy at the output density, `½Tr[ΔP·K[ΔP]]` (added to the
            // total electronic functional, like the multipole correction — no double-counting since the
            // SCC energy is the explicit DFTB functional, not a band sum).
            if let Some((gamma, p0, onsite_cache)) = &exchange_ctx {
                exchange_energy = crate::exchange::mfx_energy_fock(
                    &step.density,
                    p0,
                    &core.integrals.overlap,
                    gamma,
                )
                .energy;
                // OFX correction energy `½Tr[ΔP·K_OFX]` at the output density (same ΔP = P − P0).
                if let Some(cache) = onsite_cache {
                    let nn = basis.len();
                    let mut dp = Matrix::zeros(nn, nn);
                    for i in 0..nn {
                        for j in 0..nn {
                            dp[(i, j)] = step.density[(i, j)] - p0[(i, j)];
                        }
                    }
                    let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
                        &basis,
                        nat,
                        &core.integrals.overlap,
                        gamma,
                        &dp,
                        cache,
                    );
                    let e_ofx = 0.5
                        * dp.as_slice()
                            .iter()
                            .zip(kofx.as_slice().iter())
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                    exchange_energy += e_ofx;
                }
                // Commutator-DIIS-extrapolate the density for the next iteration's exchange Fock. The
                // commutator uses the full Fock (which already contains the exchange shift), so its
                // error is the true SCF gradient — the exchange density-matrix self-consistency is
                // accelerated even though the charge mixer cannot see the off-diagonal oscillations.
                exchange_density =
                    Some(exchange_diis.next(&step.density, &step.fock, &core.integrals.overlap));
            }
            final_rms = charge_rms(&step.shell_charges, &v_mixed[0..nsh]);
            let scc_energy = scc_electronic_free_energy(
                &basis,
                &shell_model,
                &amat,
                &core.h0,
                &step,
                external_shell_potential.as_deref(),
            )? + d4_iteration_energy
                + mp_energy
                + exchange_energy;
            let energy_error = last_scc_energy
                .map(|last| (scc_energy - last).abs())
                .unwrap_or(f64::INFINITY);
            if std::env::var("GFN1_SCC_DEBUG").is_ok() {
                // Temporary diagnostic: per-iteration energy/charge change + HOMO-LUMO gap.
                let occ = &step.occupations;
                let eps = &step.orbital_energies;
                let mut homo = f64::NEG_INFINITY;
                let mut lumo = f64::INFINITY;
                for (e, o) in eps.iter().zip(occ.iter()) {
                    if *o > 0.5 {
                        if *e > homo { homo = *e; }
                    } else if *e < lumo {
                        lumo = *e;
                    }
                }
                let gap = lumo - homo;
                let qmax = v_out[0..nsh]
                    .iter()
                    .zip(v_mixed[0..nsh].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "[SCC] it={:3} dE={:.3e} rms={:.3e} dqmax={:.3e} gap={:.4} E={:.8}",
                    iter, energy_error, final_rms, qmax, gap, scc_energy
                );
            }
            if energy_error < options.energy_tolerance && final_rms < options.charge_tolerance {
                converged = true;
                last_result = Some(step);
                break;
            }
            let residual = v_out
                .iter()
                .zip(v_mixed.iter())
                .map(|(qnew, qold)| qnew - qold)
                .collect::<Vec<_>>();
            let damped = || damped_charge_step(&v_mixed, &residual, mixing);
            let q_next = if multipole_ctx.is_some() {
                // Joint Broyden over [charges | moments]; the charge-trust filter checks only the
                // charge block. (CDIIS/Newton are monopole-specific and not used here.)
                match accelerator {
                    SccAccelerator::Linear => damped(),
                    _ => broyden
                        .next(&v_mixed, &residual)
                        .filter(|candidate| trusted_charge_vector(&candidate[0..nsh]))
                        .unwrap_or_else(damped),
                }
            } else {
                match accelerator {
                    SccAccelerator::Linear => damped(),
                    SccAccelerator::Broyden => broyden
                        .next(&v_mixed, &residual)
                        .filter(|candidate| trusted_charge_vector(candidate))
                        .unwrap_or_else(damped),
                    SccAccelerator::Cdiis => cdiis
                        .next(&v_mixed, &step.shell_charges, &residual)
                        .filter(|candidate| trusted_charge_vector(candidate))
                        .unwrap_or_else(damped),
                    SccAccelerator::Newton => newton_charge_step(
                        &core.integrals.overlap,
                        &amat,
                        &shell_model,
                        &basis,
                        &step,
                        &v_mixed,
                        &residual,
                    )
                    .filter(|candidate| trusted_charge_vector(candidate))
                    .unwrap_or_else(|| {
                        broyden
                            .next(&v_mixed, &residual)
                            .filter(|candidate| trusted_charge_vector(candidate))
                            .unwrap_or_else(damped)
                    }),
                }
            };
            prev_density = if cur_level_shift != 0.0 {
                Some(step.density.clone())
            } else {
                None
            };
            v_mixed = q_next;
            last_scc_energy = Some(scc_energy);
            last_result = Some(step);
            if iter == rung_max_scc {
                break;
            }
        }
            // Converged on this rung → done. Otherwise fall through to the next, more robust rung.
            if converged {
                break 'attempts;
            }
        }
    } // end charge-vector SCC loop (exchange-only uses run_exchange_scf above)

    if !converged {
        return Err(Gfn1Error::SccNotConverged {
            iterations,
            rms: final_rms,
        });
    }

    let step = last_result.expect("SCC loop produced no step");

    let electronic_energy = electronic_energy(&core.h0, &step.density);
    let scc =
        coulomb_energy_potential_from_matrix(&basis, &shell_model, &step.shell_charges, &amat)?;
    let final_d4 = if d4_active {
        let _profile = crate::profile::scope("electronic.final.d4_energy_potential");
        let q_atom = shell_model.atomic_charges(&basis, &step.shell_charges);
        Some(d4_dispersion_energy_potential_with_cn_pairs_and_atm(
            system,
            params,
            &q_atom,
            &d4_coordination,
            &d4_pairs,
            &d4_atm,
            d4_options,
        )?)
    } else {
        None
    };
    if let Some(d4) = &final_d4 {
        dispersion = d4.energy;
    }
    let mut final_shell_scc_potential = scc.shell_potential.clone();
    if let Some(external) = external_shell_potential.as_deref() {
        for (v, ext) in final_shell_scc_potential.iter_mut().zip(external.iter()) {
            *v += *ext;
        }
    }
    if let Some(d4) = &final_d4 {
        let d4_shell = atomic_potential_to_shell_potential(&shell_model, &d4.atomic_potential);
        for (v, add) in final_shell_scc_potential.iter_mut().zip(d4_shell.iter()) {
            *v += *add;
        }
    }
    let external_field_energy = external_shell_potential
        .as_ref()
        .map(|v| electric_field_energy(v, &step.shell_charges))
        .unwrap_or(0.0);
    let mut dipole = mulliken_dipole(system, &scc.atomic_charges, options.external_field.origin);
    if multipole_ctx.is_some() && options.field_multipole {
        // Stage 3: the physically complete dipole μ = Σ_A q_A R_A + Σ_A d_A (so that the
        // reported dipole equals −∂E/∂E_field, the field-coupling FD gate).
        let fmoments = crate::multipole::atomic_moments(&basis, nat, mp_ints, &step.density);
        dipole += crate::multipole::total_atomic_dipole(&fmoments);
    }
    let total_internal = electronic_energy
        + scc.second_order
        + scc.third_order
        + scc.higher_order // 4th+ on-site charge orders (0 unless options.charge_order > 3)
        + repulsion
        + dispersion
        + halogen
        + external_field_energy
        + mp_energy // mDFTB2 multipole correction (0 unless options.multipole)
        + exchange_energy; // long-range Fock exchange (0 unless options.lr_exchange)
    let total_free = total_internal + step.entropy_term;

    Ok(ElectronicResult {
        basis,
        integrals: core.integrals,
        h0: core.h0,
        fock: step.fock,
        density: step.density,
        energy_weighted_density: step.energy_weighted_density,
        orbital_energies: step.orbital_energies,
        occupations: step.occupations,
        // The electronic temperature actually used (== requested, unless the robust-fallback ladder
        // had to raise it on its last rung to converge an otherwise-divergent small-gap system).
        electronic_temperature: cur_etemp,
        fermi_level: step.fermi_level,
        shell_charges: step.shell_charges,
        atomic_charges: scc.atomic_charges,
        shell_scc_potential: final_shell_scc_potential,
        coordination_numbers: core.coordination_numbers,
        electronic_energy,
        repulsion_energy: repulsion,
        isotropic_scc_energy: scc.second_order,
        third_order_energy: scc.third_order,
        dispersion_energy: dispersion,
        halogen_energy: halogen,
        external_field_energy,
        electronic_entropy_term: step.entropy_term,
        total_internal,
        total_free,
        dipole,
        nelec,
        iterations,
        converged,
        spin: None,
    })
}

#[derive(Clone, Debug)]
struct SccStep {
    fock: Matrix,
    density: Matrix,
    energy_weighted_density: Matrix,
    mo_coeff: Matrix,
    orbital_energies: Vec<f64>,
    occupations: Vec<f64>,
    fermi_level: f64,
    entropy_term: f64,
    shell_charges: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn scc_step(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    shell_charges: &[f64],
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    electronic_temperature: f64,
    eigen_tolerance: f64,
    external_shell_potential: Option<&[f64]>,
    additional_shell_potential: Option<&[f64]>,
    level_shift: f64,
    prev_density: Option<&Matrix>,
    multipole_fock: Option<&Matrix>,
) -> Result<SccStep> {
    let scc = {
        let _profile = crate::profile::scope("electronic.scf.scc_potential");
        coulomb_energy_potential_from_matrix(basis, shell_model, shell_charges, amat)?
    };
    // Fold the geometry-fixed external-field site potential into the effective
    // shell potential so the density polarizes self-consistently and downstream
    // gradient/CPXTB consumers see the full effective potential.
    let mut shell_scc_potential = scc.shell_potential;
    if let Some(external) = external_shell_potential {
        for (v, ext) in shell_scc_potential.iter_mut().zip(external.iter()) {
            *v += *ext;
        }
    }
    if let Some(additional) = additional_shell_potential {
        for (v, add) in shell_scc_potential.iter_mut().zip(additional.iter()) {
            *v += *add;
        }
    }
    let mut fock = {
        let _profile = crate::profile::scope("electronic.scf.fock");
        fock_from_shell_potential(basis, overlap, h0, &shell_scc_potential)
    };
    // Self-consistent mDFTB2 multipole Fock shift (built from the previous iteration's
    // density; the field relaxes with the density across the SCC).
    if let Some(mp) = multipole_fock {
        let f = fock.as_mut_slice();
        let m = mp.as_slice();
        for (fi, mi) in f.iter_mut().zip(m.iter()) {
            *fi += *mi;
        }
    }
    // Optional virtual level shift F += b (S - 1/2 S P S), which raises the virtual
    // orbital energies by b (leaving the occupied block unchanged at self
    // consistency) and damps SCC oscillations in small-gap systems. The projector
    // uses the previous iteration's density.
    if level_shift != 0.0 {
        if let Some(prev) = prev_density {
            let _profile = crate::profile::scope("electronic.scf.level_shift");
            let sp = overlap.matmul(prev)?;
            let sps = sp.matmul(overlap)?;
            let f = fock.as_mut_slice();
            let s = overlap.as_slice();
            let m = sps.as_slice();
            for idx in 0..f.len() {
                f[idx] += level_shift * (s[idx] - 0.5 * m[idx]);
            }
        }
    }
    let eig = {
        let _profile = crate::profile::scope("electronic.scf.eigensolve");
        lowdin_solve_with_orthogonalizer(&fock, orth, eigen_tolerance)?
    };
    let kt = (electronic_temperature.max(0.0)) * BOLTZMANN_HARTREE_PER_K;
    let occupation = occupations(&eig.values, nelec, spin_channels, kt)?;
    let density = {
        let _profile = crate::profile::scope("electronic.scf.density");
        column_weighted_gram(&eig.vectors, &occupation.occupations)?
    };
    let weighted = eig
        .values
        .iter()
        .zip(occupation.occupations.iter())
        .map(|(eps, occ)| eps * occ)
        .collect::<Vec<_>>();
    let energy_weighted_density = {
        let _profile = crate::profile::scope("electronic.scf.weighted_density");
        column_weighted_gram(&eig.vectors, &weighted)?
    };
    let shell_charges = {
        let _profile = crate::profile::scope("electronic.scf.mulliken");
        mulliken_shell_charges(basis, overlap, &density)
    };
    Ok(SccStep {
        fock,
        density,
        energy_weighted_density,
        mo_coeff: eig.vectors,
        orbital_energies: eig.values,
        occupations: occupation.occupations,
        fermi_level: occupation.fermi_level,
        entropy_term: occupation.entropy_term,
        shell_charges,
    })
}

fn scc_electronic_free_energy(
    basis: &BasisSet,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    h0: &Matrix,
    step: &SccStep,
    external_shell_potential: Option<&[f64]>,
) -> Result<f64> {
    let scc = coulomb_energy_potential_from_matrix(basis, shell_model, &step.shell_charges, amat)?;
    let field = external_shell_potential
        .map(|v| electric_field_energy(v, &step.shell_charges))
        .unwrap_or(0.0);
    Ok(electronic_energy(h0, &step.density)
        + scc.second_order
        + scc.third_order
        + scc.higher_order
        + field
        + step.entropy_term)
}

fn atomic_potential_to_shell_potential(
    shell_model: &ShellChargeModel,
    atomic_potential: &[f64],
) -> Vec<f64> {
    let mut shell_potential = vec![0.0; shell_model.hardness.len()];
    for atom in 0..shell_model.atom_shell_counts.len() {
        let value = atomic_potential.get(atom).copied().unwrap_or(0.0);
        let offset = shell_model.atom_offsets[atom];
        for local in 0..shell_model.atom_shell_counts[atom] {
            shell_potential[offset + local] = value;
        }
    }
    shell_potential
}

/// Robust **density-matrix SCF** for the long-range Fock exchange (the exchange-only path; no
/// multipole). The SCF variable is the density `P`: each iteration rebuilds the *full* Fock
/// `F[P] = H0 + F_SCC(charges(P)) + K_MFX[P − P0]` from a single trial density, extrapolates the
/// whole Fock with full-Fock commutator DIIS ([`FockDiis`], error `R = FPS − SPF`), and
/// rediagonalises. This replaces the projected charge-vector mixing — which the off-diagonal
/// exchange Fock destabilises erratically (it can diverge at one mixing yet converge at another) —
/// so the exchange SCF converges robustly out-of-the-box. Returns the converged [`SccStep`], the
/// exchange energy `½Tr[ΔP K[ΔP]]`, the iteration count, and the converged flag. Non-periodic.
#[allow(clippy::too_many_arguments)]
/// Moment-channel damping factor for the multipole correction inside the robust density-matrix SCF.
/// The atomic multipole moments are **linear** in the density, so building the multipole Fock from a
/// linearly damped density `P̃ ← (1−β)P̃ + βP` is exactly a linear mix of the moments — restoring the
/// stabilisation the legacy joint charge+moment Broyden mixer provided (the multipole Fock on metal
/// centres is more nonlinear/sensitive than the exchange Fock, so without this the combined `F[P]`
/// sloshes). `β=1` reproduces the undamped (slaved-to-P) behaviour; a smaller value damps the
/// multipole channel without touching the exchange/charge channels.
const MULTIPOLE_DENSITY_MIX: f64 = 0.5;

/// Add `src` into `dst` element-wise (`dst += src`). Both are the same `n×n` shape.
fn add_assign_matrix(dst: &mut Matrix, src: &Matrix) {
    for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
        *d += *s;
    }
}

/// Geometry-fixed context for the experimental **mDFTB2 multipole correction**, bundled so the robust
/// **density-matrix** SCF drivers ([`run_exchange_scf`] / [`run_trah_exchange_scf`]) can add the
/// multipole Fock + energy built **from the trial density** each iteration. Because the atomic moments
/// are density-derived, they relax automatically with the density — no separate moment mixing (unlike
/// the legacy joint charge+moment Broyden loop). This is what lets the robust ADIIS→C-DIIS→TRAH driver
/// host the multipole(+exchange) SCC. All multipole sub-features (generic arbitrary-rank + per-rank
/// charge cross, or the legacy rank-2 + octupole + third-order + field-dipole) are covered.
struct MultipoleScf<'a> {
    basis: &'a BasisSet,
    nat: usize,
    hardness: &'a [f64],
    pos: &'a [crate::math::Vec3],
    mp_ints: &'a IntegralMatrices,
    shell_model: &'a ShellChargeModel,
    generic_rank: Option<usize>,
    generic_moment_cache: Option<&'a crate::multipole::OnsiteMomentCache>,
    octupole: bool,
    octu_cache: Option<&'a crate::multipole::OnsiteOctupoleCache>,
    field_dipole: Option<crate::math::Vec3>,
    field_dipole_fock: Option<&'a Matrix>,
    third_order: bool,
    gam3: &'a [f64],
    charge_order: &'a [usize],
}

impl MultipoleScf<'_> {
    /// The multipole-correction Fock and energy at the given trial density (with its Mulliken shell
    /// charges). Mirrors the multipole assembly of the legacy charge-vector loop, but sources the
    /// atomic moments from `density` directly (density-matrix SCF).
    fn fock_energy(&self, density: &Matrix, shell_charges: &[f64]) -> Result<(Matrix, f64)> {
        let gfn1_atomic = self.shell_model.atomic_charges(self.basis, shell_charges);
        let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
        let (mut fock, energy) = if let Some(l) = self.generic_rank {
            // Arbitrary-rank generic path (+ per-rank multipole×charge cross terms).
            let moments = crate::multipole::build_generic_moments(
                self.basis,
                self.nat,
                self.pos,
                self.mp_ints,
                density,
                &qm,
                l,
                self.generic_moment_cache,
            );
            let ef = crate::multipole::multipole_fock_generic_with_cross(
                self.basis,
                self.nat,
                self.hardness,
                self.gam3,
                self.pos,
                self.mp_ints,
                &moments,
                &qm,
                self.charge_order,
                l,
                self.generic_moment_cache,
            );
            let mut e = ef.energy;
            if !self.charge_order.is_empty() {
                e += crate::multipole::multipole_charge_cross_energy(
                    self.nat,
                    self.hardness,
                    self.gam3,
                    &moments,
                    &qm,
                    self.charge_order,
                    l,
                );
            }
            if let Some(field) = self.field_dipole {
                let mut dsum = crate::math::Vec3::zero();
                for m in &moments {
                    dsum += crate::math::Vec3::new(m[1][0], m[1][1], m[1][2]);
                }
                e += -field.dot(dsum);
            }
            (ef.fock, e)
        } else {
            // Legacy rank-2 dipole/quadrupole path (+ optional octupole / third-order / field-dipole).
            let moments =
                crate::multipole::atomic_moments(self.basis, self.nat, self.mp_ints, density);
            let ef = crate::multipole::multipole_fock_from_moments(
                self.basis,
                self.nat,
                self.hardness,
                self.pos,
                self.mp_ints,
                &moments,
                &qm,
            );
            let mut f = ef.fock;
            let mut e = ef.energy;
            if self.octupole {
                let octu = crate::multipole::atomic_octupole_moments(
                    self.basis,
                    self.nat,
                    self.pos,
                    self.mp_ints,
                    density,
                    self.octu_cache,
                );
                let oef = crate::multipole::octupole_fock_from_moments(
                    self.basis,
                    self.nat,
                    self.hardness,
                    self.pos,
                    self.mp_ints,
                    &qm,
                    &moments.dipole,
                    &moments.quad,
                    &octu,
                    self.octu_cache,
                );
                add_assign_matrix(&mut f, &oef.fock);
                e += oef.energy;
            }
            if self.third_order {
                let m3 = crate::multipole::third_order_fock_from_moments(
                    self.basis,
                    self.nat,
                    self.hardness,
                    self.gam3,
                    self.mp_ints,
                    &moments,
                    &qm,
                );
                add_assign_matrix(&mut f, &m3.fock);
                e += m3.energy;
            }
            if let Some(field) = self.field_dipole {
                e += crate::multipole::field_dipole_energy(field, &moments);
            }
            (f, e)
        };
        if let Some(fdf) = self.field_dipole_fock {
            add_assign_matrix(&mut fock, fdf);
        }
        Ok((fock, energy))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_exchange_scf(
    basis: &BasisSet,
    nat: usize,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    electronic_temperature: f64,
    eigen_tolerance: f64,
    external_shell_potential: Option<&[f64]>,
    gamma: &Matrix,
    p0: &Matrix,
    onsite_cache: Option<&crate::exchange::OnsiteExchangeCache>,
    mp: Option<&MultipoleScf>,
    max_scc: usize,
    energy_tolerance: f64,
    charge_tolerance: f64,
) -> Result<(SccStep, f64, usize, bool, f64)> {
    let n = basis.len();
    let kt = (electronic_temperature.max(0.0)) * BOLTZMANN_HARTREE_PER_K;
    // Start from the neutral-atom reference (ΔP = 0 ⇒ no exchange ⇒ F ≈ H0, the usual first guess).
    let mut density = p0.clone();
    // Damped density for the multipole channel (moment mixing); `None` until the first iteration.
    let mut mp_density: Option<Matrix> = None;
    let mut diis = FockDiis::new(20); // fast C-DIIS for the quadratic tail
    let mut adiis = AdiisMixer::new(20); // globally-convergent ADIIS far from the solution
    let mut last_energy: Option<f64> = None;
    let mut last_step: Option<SccStep> = None;
    // Previous-iteration MOs / occupations (the basis that defines the current density), for the
    // orbital-rotation (κ) estimate, and the previous commutator norm for stagnation detection.
    let mut prev_mo: Option<Matrix> = None;
    let mut prev_occ: Option<Vec<f64>> = None;
    let mut last_comm_rms: Option<f64> = None;
    // Solver state machine (one-way): κ-shifted **ADIIS** (globally convergent) → **C-DIIS** (fast
    // tail) once the residual is small; the shift is re-armed if C-DIIS then stalls (a GDM-lite
    // second-order regulariser). Following Q-Chem / Helmich-Paris (2021) robust-SCF practice.
    let mut use_adiis = true;
    let mut below_count = 0usize; // consecutive iters with comm_rms < the ADIIS→C-DIIS threshold
    let mut diis_stall = 0usize; // consecutive non-decreasing C-DIIS iters
    let mut cdiis_failed = false; // C-DIIS limit-cycled ⇒ stay on the monotone ADIIS for the tail
    let mut fallback_iters = 0usize; // iterations spent on the monotone ADIIS tail after that switch
    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_comm_rms = f64::INFINITY; // real last commutator RMS, for honest non-convergence reporting
    for iter in 1..=max_scc {
        iterations = iter;
        // Charges + SCC potential from the current trial density.
        let shell_charges = mulliken_shell_charges(basis, overlap, &density);
        let scc = coulomb_energy_potential_from_matrix(basis, shell_model, &shell_charges, amat)?;
        let mut shell_pot = scc.shell_potential;
        if let Some(ext) = external_shell_potential {
            for (v, e) in shell_pot.iter_mut().zip(ext.iter()) {
                *v += *e;
            }
        }
        let mut fock = fock_from_shell_potential(basis, overlap, h0, &shell_pot);
        // Exchange Fock K[ΔP] from the same trial density.
        let mut dp = Matrix::zeros(n, n);
        {
            let d = dp.as_mut_slice();
            for (dk, (pk, p0k)) in d
                .iter_mut()
                .zip(density.as_slice().iter().zip(p0.as_slice().iter()))
            {
                *dk = pk - p0k;
            }
        }
        let mut kx = {
            let _p = crate::profile::scope("exch.mfx_kernel");
            crate::exchange::mfx_kernel(&dp, overlap, gamma)
        };
        // OFX: upgrade the same-atom exchange to exact one-center ERIs (cached per element).
        if let Some(cache) = onsite_cache {
            let _p = crate::profile::scope("exch.ofx_kernel");
            let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
                basis, nat, overlap, gamma, &dp, cache,
            );
            let k = kx.as_mut_slice();
            for (ki, oi) in k.iter_mut().zip(kofx.as_slice().iter()) {
                *ki += *oi;
            }
        }
        {
            let f = fock.as_mut_slice();
            for (fi, ki) in f.iter_mut().zip(kx.as_slice().iter()) {
                *fi += *ki;
            }
        }
        // mDFTB2 multipole correction Fock + energy. The atomic moments are density-derived, but the
        // multipole channel is **damped** (built from a linearly mixed density `mp_density`, ≡ damped
        // moments) so it doesn't slosh the density-matrix SCF the way the raw moments do on metal
        // centres — the analogue of the legacy joint charge+moment Broyden mixing. At convergence
        // `mp_density == density`, so the result is unchanged; only the path is stabilised.
        let mp_energy = if let Some(mp) = mp {
            let mpd = match &mp_density {
                Some(prev) => {
                    let mut d = prev.clone();
                    for (di, ci) in d.as_mut_slice().iter_mut().zip(density.as_slice().iter()) {
                        *di = (1.0 - MULTIPOLE_DENSITY_MIX) * *di + MULTIPOLE_DENSITY_MIX * *ci;
                    }
                    d
                }
                None => density.clone(),
            };
            let mp_charges = mulliken_shell_charges(basis, overlap, &mpd);
            let (mf, me) = mp.fock_energy(&mpd, &mp_charges)?;
            add_assign_matrix(&mut fock, &mf);
            mp_density = Some(mpd);
            me
        } else {
            0.0
        };
        // Commutator SCF error R = F P S − S P F (zero at convergence). F, P, S are symmetric, so
        // S P F = (F P S)ᵀ — one triple matrix product instead of two.
        let fps = {
            let _p = crate::profile::scope("exch.commutator");
            fock.matmul(&density)?.matmul(overlap)?
        };
        let spf = fps.transpose();
        let err: Vec<f64> = fps
            .as_slice()
            .iter()
            .zip(spf.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        let comm_rms = (err.iter().map(|x| x * x).sum::<f64>() / err.len().max(1) as f64).sqrt();
        final_comm_rms = comm_rms;
        // `kx` is the full exchange Fock (MFX + OFX); both are self-adjoint ⇒ E = ½Tr[ΔP·K].
        let exchange_energy = 0.5
            * dp.as_slice()
                .iter()
                .zip(kx.as_slice().iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        // --- Solver state machine (one-way ADIIS → C-DIIS, history discarded at the switch) ---
        const ADIIS_TO_CDIIS: f64 = 1.0e-3; // switch to C-DIIS once the residual is this small …
        const SWITCH_PERSIST: usize = 2; //   … for this many consecutive iterations
        const DIIS_STALL_MAX: usize = 8; // C-DIIS stalled this long ⇒ re-arm the trust-region shift
        if use_adiis {
            // Only graduate to C-DIIS the first time; once C-DIIS has proven to limit-cycle on this
            // system (`cdiis_failed`) we stay on the monotone ADIIS for the rest of the run.
            if !cdiis_failed && comm_rms < ADIIS_TO_CDIIS {
                below_count += 1;
                if below_count >= SWITCH_PERSIST {
                    use_adiis = false; // hand off to the fast C-DIIS tail
                    adiis.clear();
                    diis.clear(); // start C-DIIS clean (no extrapolation across the ADIIS mapping)
                }
            } else {
                below_count = 0;
            }
        } else {
            if last_comm_rms.is_some_and(|p| comm_rms > 0.9 * p) {
                diis_stall += 1;
            } else {
                diis_stall = 0;
            }
            if diis_stall >= DIIS_STALL_MAX {
                // C-DIIS is limit-cycling on a near-degenerate system: fall back — once, one-way — to
                // the monotone ADIIS, which cannot diverge (convex combination on the simplex). This
                // is the "DIIS stalls ⇒ GDM/Newton-class settler" stage of the robust-SCF pipeline.
                use_adiis = true;
                cdiis_failed = true;
                adiis.clear();
                diis.clear();
                below_count = 0;
                diis_stall = 0;
            }
        }
        // Feed only the ACTIVE mixer — and the **physical** Fock / commutator, never the shifted Fock
        // (the level shift is a solver device for the orbital update, not a physical operator).
        let fock_phys = fock.clone(); // physical F[P_k], kept for the κ estimate
        let mut f_star = if use_adiis {
            adiis.next(fock, density.clone())
        } else {
            diis.next(fock, err)
        };
        // **Trust-region level shift** while ADIIS is active (or when C-DIIS has stalled), triggered by
        // the predicted orbital rotation, not the bare gap: `κ⁰_ai = −F_ai/Δ_ai` from the **physical**
        // Fock `F[P_k]` in the MOs that define `P_k` (`F_ai≠0` there; ≡0 in `F`'s own eigenbasis, which
        // must NOT be used). `Δ_ai = F^MO_aa − F^MO_ii` (the canonical-ish diagonal). Pick the smallest
        // `λ` keeping `‖κ(λ)‖₂ = √Σ(F_ai/(Δ_ai+λ))² ≤ h`, with the bisection lower bound chosen so every
        // denominator is positive (handles negative gaps / level crossings). The virtual-only shift
        // `F += λ(S − ½SPS)` leaves the occupied block — hence the fixed point — unchanged.
        let h_trust = if diis_stall > 0 { 0.1_f64 } else { 0.4_f64 };
        const GAP_FLOOR: f64 = 0.05;
        let want_shift = use_adiis || diis_stall >= DIIS_STALL_MAX;
        let mut lshift = 0.0_f64;
        if iter > 3 && want_shift {
            if let (Some(c), Some(occ_p)) = (&prev_mo, &prev_occ) {
                let _p = crate::profile::scope("exch.kappa_fmo");
                let fmo = c.transpose().matmul(&fock_phys)?.matmul(c)?;
                let mut pairs: Vec<(f64, f64)> = Vec::new(); // (F_ai, Δ_ai)
                let mut min_delta = f64::INFINITY;
                for i in 0..occ_p.len() {
                    if occ_p[i] <= 0.5 {
                        continue;
                    }
                    for a in 0..occ_p.len() {
                        if occ_p[a] <= 0.5 {
                            let delta = fmo[(a, a)] - fmo[(i, i)];
                            min_delta = min_delta.min(delta);
                            pairs.push((fmo[(a, i)], delta));
                        }
                    }
                }
                if !pairs.is_empty() {
                    // λ_lo keeps all (Δ_ai + λ) > 0 even with negative/near-degenerate gaps.
                    let lam_lo = (-min_delta + GAP_FLOOR).max(0.0);
                    let knorm = |lam: f64| -> f64 {
                        pairs
                            .iter()
                            .map(|(f, d)| (f / (d + lam)).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    };
                    if knorm(lam_lo) > h_trust {
                        let mut lo = lam_lo;
                        let mut hi = lam_lo + 1.0;
                        while knorm(hi) > h_trust && hi < lam_lo + 200.0 {
                            hi *= 2.0;
                        }
                        for _ in 0..40 {
                            let mid = 0.5 * (lo + hi);
                            if knorm(mid) > h_trust {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        lshift = hi;
                    } else {
                        lshift = lam_lo; // still apply the floor to keep denominators positive
                    }
                }
            }
        }
        if lshift > 1.0e-8 {
            // Virtual-only shift `F += λ(S − ½ S P S)` ⇒ ε_a → ε_a+λ, occupied unchanged (P has occ 2),
            // so Δ_ai → Δ_ai+λ exactly as the κ(λ) estimate assumes, and the fixed point is preserved.
            let sps = overlap.matmul(&density)?.matmul(overlap)?;
            let f = f_star.as_mut_slice();
            for ((fk, sk), spsk) in f
                .iter_mut()
                .zip(overlap.as_slice().iter())
                .zip(sps.as_slice().iter())
            {
                *fk += lshift * (sk - 0.5 * spsk);
            }
        }
        let eig = {
            let _p = crate::profile::scope("exch.diagonalize");
            lowdin_solve_with_orthogonalizer(&f_star, orth, eigen_tolerance)?
        };
        let occ = occupations(&eig.values, nelec, spin_channels, kt)?;
        let new_density = column_weighted_gram(&eig.vectors, &occ.occupations)?;
        prev_mo = Some(eig.vectors.clone());
        prev_occ = Some(occ.occupations.clone());
        last_comm_rms = Some(comm_rms);
        if std::env::var("GFN1_EXCH_DEBUG").is_ok() {
            // HOMO–LUMO gap (occupation crosses ½) + the exchange-energy growth + ‖ΔP‖, to diagnose
            // the small-gap → exchange-feedback instability on metallic systems.
            let mut homo = f64::NEG_INFINITY;
            let mut lumo = f64::INFINITY;
            for (e, o) in eig.values.iter().zip(occ.occupations.iter()) {
                if *o > 0.5 {
                    homo = homo.max(*e);
                } else {
                    lumo = lumo.min(*e);
                }
            }
            let dp_norm = dp.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
            let solver = if use_adiis { "ADIIS" } else { "CDIIS" };
            eprintln!(
                "[exch] iter={iter} {solver} gap={:.3}eV lshift={lshift:.3} ex_E={exchange_energy:.5} |dP|={dp_norm:.4} comm_rms={comm_rms:.3e}",
                (lumo - homo) * 27.211_386
            );
        }
        let field = external_shell_potential
            .map(|v| electric_field_energy(v, &shell_charges))
            .unwrap_or(0.0);
        let energy = electronic_energy(h0, &density)
            + scc.second_order
            + scc.third_order
            + scc.higher_order
            + field
            + exchange_energy
            + mp_energy
            + occ.entropy_term;
        let de = last_energy
            .map(|l| (energy - l).abs())
            .unwrap_or(f64::INFINITY);
        // Build this iteration's step (the new density's quantities).
        let weighted: Vec<f64> = eig
            .values
            .iter()
            .zip(occ.occupations.iter())
            .map(|(e, o)| e * o)
            .collect();
        let ewd = column_weighted_gram(&eig.vectors, &weighted)?;
        let new_charges = mulliken_shell_charges(basis, overlap, &new_density);
        let step = SccStep {
            fock: f_star,
            density: new_density.clone(),
            energy_weighted_density: ewd,
            mo_coeff: eig.vectors,
            orbital_energies: eig.values,
            occupations: occ.occupations,
            fermi_level: occ.fermi_level,
            entropy_term: occ.entropy_term,
            shell_charges: new_charges,
        };
        if comm_rms < charge_tolerance && de < energy_tolerance {
            converged = true;
            last_step = Some(step);
            break;
        }
        last_energy = Some(energy);
        last_step = Some(step);
        density = new_density;
        // Early hand-off to the second-order TRAH: once C-DIIS has limit-cycled and the monotone
        // ADIIS fallback has stabilised the density (a small residual that it now only grinds down
        // linearly), stop and let the caller's quadratically-convergent TRAH polish finish the tail
        // — far cheaper than running the whole SCC budget here. (Gapped systems never reach this:
        // C-DIIS converges them before `cdiis_failed`.)
        if cdiis_failed {
            fallback_iters += 1;
            const TRAH_HANDOFF: usize = 10;
            if fallback_iters >= TRAH_HANDOFF && comm_rms < 1.0e-2 {
                break; // returns converged=false ⇒ the caller runs the TRAH continuation from here
            }
        }
    }
    let step = last_step.expect("exchange SCF produced no step");
    // Exchange energy consistent with the returned density.
    let mut dp_final = Matrix::zeros(n, n);
    {
        let d = dp_final.as_mut_slice();
        for (dk, (pk, p0k)) in d
            .iter_mut()
            .zip(step.density.as_slice().iter().zip(p0.as_slice().iter()))
        {
            *dk = pk - p0k;
        }
    }
    let mut kx_final = crate::exchange::mfx_kernel(&dp_final, overlap, gamma);
    if let Some(cache) = onsite_cache {
        let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
            basis, nat, overlap, gamma, &dp_final, cache,
        );
        let k = kx_final.as_mut_slice();
        for (ki, oi) in k.iter_mut().zip(kofx.as_slice().iter()) {
            *ki += *oi;
        }
    }
    let exchange_energy = 0.5
        * dp_final
            .as_slice()
            .iter()
            .zip(kx_final.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    Ok((step, exchange_energy, iterations, converged, final_comm_rms))
}

/// **Nested SCF driver** for the exchange-augmented SCC — the robust default, especially for
/// small-gap / metallic systems. The off-diagonal exchange Fock makes quasi-Newton charge mixing
/// (Broyden/DIIS) erratic, while plain damped density-matrix iteration *plateaus* on a stiff metal
/// (a near-unit SCF eigenmode the charge-vector Broyden handles but linear mixing cannot). This driver
/// decouples the two difficulties:
///
/// - **inner loop** — converge the ordinary charge SCC with the exchange Fock `K` held **fixed**
///   (injected via `scc_step`'s `multipole_fock` slot), using the metal-capable charge-vector
///   **Broyden** mixer. With `K` constant there is no exchange-induced instability, so Broyden
///   converges as it does for plain GFN1;
/// - **outer loop** — rebuild `K = K_MFX[ΔP] (+ K_OFX[ΔP])` from the converged inner density and damp
///   it in. The exchange is a perturbation, so a handful of outer cycles converge it; the inner
///   charges are warm-started from the previous cycle, so later inner solves take only a few steps.
///
/// Returns the canonical [`SccStep`] at the converged density (its Fock / energy-weighted density
/// include the exchange) and `E_x = ½Tr[ΔP·K]`. Non-periodic.
///
/// **FLAWED / UNUSED.** The decoupled outer loop treats the exchange Fock as a *fixed linear*
/// potential, so the inner density over-responds (it misses the ½ self-consistency factor of the
/// quadratic exchange energy) and the outer fixed-point is expansive — it diverges even for gapped
/// molecules. Kept only as a record; the coupled [`run_exchange_scf`] is the correct driver. Do not
/// wire this in.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn run_nested_exchange_scf(
    basis: &BasisSet,
    nat: usize,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    electronic_temperature: f64,
    eigen_tolerance: f64,
    external_shell_potential: Option<&[f64]>,
    gamma: &Matrix,
    p0: &Matrix,
    onsite_cache: Option<&crate::exchange::OnsiteExchangeCache>,
    max_scc: usize,
    energy_tolerance: f64,
    charge_tolerance: f64,
) -> Result<(SccStep, f64, usize, bool, f64)> {
    let n = basis.len();
    let nsh = basis.shells.len();
    // Exchange Fock K[ΔP] = MFX (+ OFX) from a density fluctuation.
    let exch_fock = |p: &Matrix| -> Matrix {
        let mut dp = Matrix::zeros(n, n);
        for (d, (pk, p0k)) in dp
            .as_mut_slice()
            .iter_mut()
            .zip(p.as_slice().iter().zip(p0.as_slice().iter()))
        {
            *d = pk - p0k;
        }
        let mut k = crate::exchange::mfx_kernel(&dp, overlap, gamma);
        if let Some(cache) = onsite_cache {
            let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
                basis, nat, overlap, gamma, &dp, cache,
            );
            for (ki, oi) in k.as_mut_slice().iter_mut().zip(kofx.as_slice().iter()) {
                *ki += *oi;
            }
        }
        k
    };

    const MAX_OUTER: usize = 40;
    const OUTER_DAMP: f64 = 0.6;
    let inner_max = max_scc.max(60);
    let mut k_exch = Matrix::zeros(n, n);
    let mut q_start = vec![0.0_f64; nsh]; // warm-started charge vector across outer cycles
    let mut total_iters = 0usize;
    let mut last_step: Option<SccStep> = None;
    let mut last_ex_energy: Option<f64> = None;
    let mut converged = false;

    for outer in 0..MAX_OUTER {
        // Inner: charge SCC with the exchange Fock fixed, charge-vector Broyden.
        let mut broyden = BroydenMixer::new(nsh, 20, 0.4);
        let mut q = q_start.clone();
        let mut inner_step: Option<SccStep> = None;
        for _inner in 0..inner_max {
            total_iters += 1;
            let step = scc_step(
                basis,
                overlap,
                h0,
                orth,
                shell_model,
                amat,
                &q,
                nelec,
                spin_channels,
                electronic_temperature,
                eigen_tolerance,
                external_shell_potential,
                None,
                0.0,
                None,
                Some(&k_exch),
            )?;
            let q_out = step.shell_charges.clone();
            let rms = charge_rms(&q_out, &q);
            inner_step = Some(step);
            if rms < charge_tolerance {
                break;
            }
            let dq: Vec<f64> = q_out.iter().zip(q.iter()).map(|(o, i)| o - i).collect();
            q = broyden.next(&q, &dq).unwrap_or(q_out);
        }
        let step = inner_step.expect("nested exchange inner SCC produced no step");
        q_start = step.shell_charges.clone();

        // Outer: rebuild the exchange Fock from the converged inner density; converge on its energy.
        let mut dp = Matrix::zeros(n, n);
        for (d, (pk, p0k)) in dp
            .as_mut_slice()
            .iter_mut()
            .zip(step.density.as_slice().iter().zip(p0.as_slice().iter()))
        {
            *d = pk - p0k;
        }
        let k_new = exch_fock(&step.density);
        let ex_energy = 0.5
            * dp.as_slice()
                .iter()
                .zip(k_new.as_slice().iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let de = last_ex_energy
            .map(|l| (ex_energy - l).abs())
            .unwrap_or(f64::INFINITY);
        last_ex_energy = Some(ex_energy);
        last_step = Some(step);
        if std::env::var("GFN1_EXCH_DEBUG").is_ok() {
            eprintln!(
                "[nest] outer={outer} total_iters={total_iters} ex_energy={ex_energy:.8} de={de:.3e}"
            );
        }
        if outer > 0 && de < energy_tolerance {
            converged = true;
            break;
        }
        // Damp the exchange Fock update for outer-loop stability.
        for (ke, kn) in k_exch
            .as_mut_slice()
            .iter_mut()
            .zip(k_new.as_slice().iter())
        {
            *ke += OUTER_DAMP * (*kn - *ke);
        }
    }

    let step = last_step.expect("nested exchange SCF produced no step");
    let k_final = exch_fock(&step.density);
    let mut dp_final = Matrix::zeros(n, n);
    for (d, (pk, p0k)) in dp_final
        .as_mut_slice()
        .iter_mut()
        .zip(step.density.as_slice().iter().zip(p0.as_slice().iter()))
    {
        *d = pk - p0k;
    }
    let exchange_energy = 0.5
        * dp_final
            .as_slice()
            .iter()
            .zip(k_final.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    Ok((step, exchange_energy, total_iters, converged, f64::INFINITY))
}

/// **Trust-Region Augmented Hessian** SCF driver for the exchange-augmented SCC (MFX/OFX) — the
/// robust second-order fallback to [`run_exchange_scf`]'s commutator-DIIS. Instead of mixing, it
/// minimises the electronic energy directly over orbital rotations `C→C·exp(κ)` with a matrix-free
/// Newton/trust-region step (see [`crate::trah`]). It builds two closures over the real functional:
/// `fock_energy(P)→(F,E)` (one-electron + isotropic SCC + MFX/OFX exchange) and the **linear** Fock
/// response `fock_response(δP)→δF` (second-order charge kernel `−½S∘(A·δq)` + MFX + OFX kernels). The
/// gradient uses the *exact* `F`, so TRAH's fixed point (`g=0`) is the true SCC solution regardless of
/// the (approximate, second-order) response that only steers the step. Returns the same canonical
/// [`SccStep`] as the DIIS driver (one final `scc_step` at the converged density). Closed-shell /
/// gapped, integer occupations (the Fermi entropy term vanishes at the gap); non-periodic.
#[allow(clippy::too_many_arguments)]
fn run_trah_exchange_scf(
    basis: &BasisSet,
    nat: usize,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    electronic_temperature: f64,
    eigen_tolerance: f64,
    external_shell_potential: Option<&[f64]>,
    gamma: &Matrix,
    p0: &Matrix,
    onsite_cache: Option<&crate::exchange::OnsiteExchangeCache>,
    mp: Option<&MultipoleScf>,
    max_scc: usize,
    energy_tolerance: f64,
    charge_tolerance: f64,
    initial_mo: Option<&Matrix>,
) -> Result<(SccStep, f64, usize, bool, f64)> {
    let n = basis.len();
    let nsh = basis.shells.len();
    // Starting MOs: a *continuation* from a near-converged density (the ADIIS/C-DIIS result) when
    // provided, otherwise the core-Hamiltonian guess. From a near-converged density the second-order
    // Newton step needs only a handful of macro-iterations, so the polish is affordable at any size.
    // Integer closed-shell aufbau occupations either way (the Fermi entropy term vanishes at the gap).
    let start_mo = match initial_mo {
        Some(c) => c.clone(),
        None => lowdin_solve_with_orthogonalizer(h0, orth, eigen_tolerance)?.vectors,
    };
    let nmo = start_mo.cols();
    let mut occ = vec![0.0_f64; nmo];
    let mut remaining = nelec;
    for o in occ.iter_mut() {
        let fill = remaining.clamp(0.0, 2.0);
        *o = fill;
        remaining -= fill;
        if remaining <= 1.0e-12 {
            break;
        }
    }

    // Exchange Fock K[ΔP] = MFX (+ OFX) at a density (also the response, since both are linear in P).
    let exchange_kernel = |dp: &Matrix| -> Matrix {
        let mut k = crate::exchange::mfx_kernel(dp, overlap, gamma);
        if let Some(cache) = onsite_cache {
            let kofx = crate::exchange::onsite_fock_exchange_kernel_cached(
                basis, nat, overlap, gamma, dp, cache,
            );
            for (ki, oi) in k.as_mut_slice().iter_mut().zip(kofx.as_slice().iter()) {
                *ki += *oi;
            }
        }
        k
    };
    let delta_p = |p: &Matrix| -> Matrix {
        let mut dp = Matrix::zeros(n, n);
        for (d, (pk, p0k)) in dp
            .as_mut_slice()
            .iter_mut()
            .zip(p.as_slice().iter().zip(p0.as_slice().iter()))
        {
            *d = pk - p0k;
        }
        dp
    };

    let ext = external_shell_potential;
    let fock_energy = |p: &Matrix| -> (Matrix, f64) {
        let qsh = mulliken_shell_charges(basis, overlap, p);
        let scc = coulomb_energy_potential_from_matrix(basis, shell_model, &qsh, amat)
            .expect("TRAH SCC potential build (internal invariant)");
        let mut shell_pot = scc.shell_potential;
        if let Some(e) = ext {
            for (v, ee) in shell_pot.iter_mut().zip(e.iter()) {
                *v += *ee;
            }
        }
        let mut f = fock_from_shell_potential(basis, overlap, h0, &shell_pot);
        let dp = delta_p(p);
        let k = exchange_kernel(&dp);
        for (fi, ki) in f.as_mut_slice().iter_mut().zip(k.as_slice().iter()) {
            *fi += *ki;
        }
        let ex_e = 0.5
            * dp.as_slice()
                .iter()
                .zip(k.as_slice().iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        // mDFTB2 multipole correction in F[P] and the energy. NOTE: it is added to the gradient
        // `F[P]` here but intentionally NOT to `fock_response` below, so TRAH's second-order step
        // sees the exchange+charge curvature only — the multipole block is treated as a fixed
        // first-order potential. TRAH is the *continuation* from a near-converged ADIIS/C-DIIS
        // density (where the multipole is already nearly self-consistent), so this preserves the
        // correct fixed point; only the (rarely-needed) polish's convergence rate is affected.
        let mp_e = if let Some(mp) = mp {
            let (mf, me) = mp
                .fock_energy(p, &qsh)
                .expect("TRAH multipole Fock build (internal invariant)");
            add_assign_matrix(&mut f, &mf);
            me
        } else {
            0.0
        };
        let field = ext.map(|v| electric_field_energy(v, &qsh)).unwrap_or(0.0);
        let e = electronic_energy(h0, p)
            + scc.second_order
            + scc.third_order
            + scc.higher_order
            + field
            + ex_e
            + mp_e;
        (f, e)
    };

    let h0_zero = Matrix::zeros(n, n);
    let fock_response = |dp: &Matrix| -> Matrix {
        // Second-order charge response: δq_sh = −Σ_{μ∈sh,ν} S_{μν} δP_{μν}; δV = A·δq; δF = −½S∘(δVi+δVj).
        let mut dq = vec![0.0_f64; nsh];
        for (ish, sh) in basis.shells.iter().enumerate() {
            let mut pop = 0.0;
            for iao in sh.first_ao..sh.first_ao + sh.nao {
                for nu in 0..n {
                    pop += overlap[(iao, nu)] * dp[(iao, nu)];
                }
            }
            dq[ish] = -pop;
        }
        let mut dv = vec![0.0_f64; nsh];
        for (ish, dvi) in dv.iter_mut().enumerate() {
            let mut s = 0.0;
            for (jsh, &dqj) in dq.iter().enumerate() {
                s += amat[(ish, jsh)] * dqj;
            }
            *dvi = s;
        }
        let mut df = fock_from_shell_potential(basis, overlap, &h0_zero, &dv);
        let k = exchange_kernel(dp);
        for (fi, ki) in df.as_mut_slice().iter_mut().zip(k.as_slice().iter()) {
            *fi += *ki;
        }
        df
    };

    let opt = crate::trah::TrahOptions {
        // A continuation from a near-converged density needs only a handful of Newton steps; the
        // from-scratch path keeps the full budget. `cg_tol = 1e-4` is an inexact-Newton inner solve
        // (a rough direction suffices and is retightened by the outer loop) — with the Jacobi
        // preconditioner this keeps each macro-iteration to O(10) O(N³) Hessian-vector products.
        max_iter: if initial_mo.is_some() {
            max_scc.min(50)
        } else {
            max_scc
        },
        grad_tol: charge_tolerance,
        cg_tol: 1.0e-4,
        ..crate::trah::TrahOptions::default()
    };
    let res = crate::trah::run_trah_scf(&start_mo, &occ, fock_energy, fock_response, &opt)?;

    // Canonical SccStep at the converged density (one diagonalisation with the converged charges +
    // exchange Fock; Fermi occupations match the integer TRAH ones at the gap).
    let shell_charges = mulliken_shell_charges(basis, overlap, &res.density);
    let k_final = exchange_kernel(&delta_p(&res.density));
    let step = scc_step(
        basis,
        overlap,
        h0,
        orth,
        shell_model,
        amat,
        &shell_charges,
        nelec,
        spin_channels,
        electronic_temperature,
        eigen_tolerance,
        external_shell_potential,
        None,
        0.0,
        None,
        Some(&k_final),
    )?;
    let dp_final = delta_p(&step.density);
    let k2 = exchange_kernel(&dp_final);
    let exchange_energy = 0.5
        * dp_final
            .as_slice()
            .iter()
            .zip(k2.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let _ = energy_tolerance;
    Ok((
        step,
        exchange_energy,
        res.iterations,
        res.converged,
        res.gradient_norm,
    ))
}

pub fn fock_from_shell_potential(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    shell_potential: &[f64],
) -> Matrix {
    let n = basis.len();
    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = shell_potential[ish];
        }
    }

    let mut fock = h0.clone();
    for i in 0..n {
        for j in 0..=i {
            let value = -overlap[(i, j)] * 0.5 * (vao[i] + vao[j]);
            fock[(i, j)] += value;
            if i != j {
                fock[(j, i)] += value;
            }
        }
    }
    fock
}

pub fn mulliken_shell_charges(basis: &BasisSet, overlap: &Matrix, density: &Matrix) -> Vec<f64> {
    let n = basis.len();
    let overlap_slice = overlap.as_slice();
    let density_slice = density.as_slice();
    let mut qsh = vec![0.0; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut population = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            let offset = iao * n;
            let overlap_row = &overlap_slice[offset..offset + n];
            let density_row = &density_slice[offset..offset + n];
            population += overlap_row
                .iter()
                .zip(density_row.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
        qsh[ish] = shell.reference_occ - population;
    }
    qsh
}

pub fn electronic_energy(h0: &Matrix, density: &Matrix) -> f64 {
    h0.as_slice()
        .iter()
        .zip(density.as_slice().iter())
        .map(|(a, b)| a * b)
        .sum()
}

fn charge_rms(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let ss = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>();
    (ss / a.len() as f64).sqrt()
}

#[derive(Clone, Debug)]
pub(crate) struct BroydenMixer {
    memory: usize,
    damp: f64,
    iter: usize,
    df: Vec<Vec<f64>>,
    u: Vec<Vec<f64>>,
    omega: Vec<f64>,
    qlast: Vec<f64>,
    dqlast: Vec<f64>,
}

impl BroydenMixer {
    pub(crate) fn new(ndim: usize, memory: usize, damp: f64) -> Self {
        Self {
            memory,
            damp,
            iter: 0,
            df: vec![vec![0.0; ndim]; memory],
            u: vec![vec![0.0; ndim]; memory],
            omega: vec![0.0; memory],
            qlast: vec![0.0; ndim],
            dqlast: vec![0.0; ndim],
        }
    }

    pub(crate) fn next(&mut self, q: &[f64], dq: &[f64]) -> Option<Vec<f64>> {
        if q.len() != self.qlast.len() || dq.len() != self.dqlast.len() {
            return None;
        }
        self.iter += 1;
        if self.iter == 1 {
            self.dqlast.copy_from_slice(dq);
            self.qlast.copy_from_slice(q);
            return Some(damped_charge_step(q, dq, self.damp));
        }

        let itn = self.iter - 1;
        let it1 = (itn - 1) % self.memory;
        let omega0 = 0.01;
        let minw = 1.0;
        let maxw = 100000.0;
        let wfac = 0.01;

        let dq_norm = dot(dq, dq).sqrt();
        self.omega[it1] = if dq_norm > wfac / maxw {
            wfac / dq_norm
        } else {
            maxw
        };
        if self.omega[it1] < minw {
            self.omega[it1] = minw;
        }

        let mut df_col = dq
            .iter()
            .zip(self.dqlast.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let inv = 1.0 / dot(&df_col, &df_col).sqrt().max(f64::EPSILON);
        for value in &mut df_col {
            *value *= inv;
        }
        self.df[it1] = df_col;

        self.u[it1] = self.df[it1]
            .iter()
            .zip(q.iter().zip(self.qlast.iter()))
            .map(|(df, (q_now, q_last))| self.damp * df + inv * (q_now - q_last))
            .collect();

        let start = if itn > self.memory {
            itn - self.memory + 1
        } else {
            1
        };
        let active = (start..=itn)
            .map(|j| (j - 1) % self.memory)
            .collect::<Vec<_>>();
        let m = active.len();
        if m == 0 {
            return None;
        }

        let mut beta = vec![vec![0.0; m]; m];
        let mut c = vec![0.0; m];
        for (row, &i) in active.iter().enumerate() {
            c[row] = self.omega[i] * dot(&self.df[i], dq);
            for (col, &j) in active.iter().enumerate() {
                beta[row][col] = self.omega[i] * self.omega[j] * dot(&self.df[i], &self.df[j]);
            }
            beta[row][row] += omega0 * omega0;
        }

        let coeff = solve_linear(beta, c)?;
        self.dqlast.copy_from_slice(dq);
        self.qlast.copy_from_slice(q);

        let mut out = damped_charge_step(q, dq, self.damp);
        for (row, &i) in active.iter().enumerate() {
            let scale = self.omega[i] * coeff[row];
            for (value, u) in out.iter_mut().zip(self.u[i].iter()) {
                *value -= scale * u;
            }
        }
        Some(out)
    }
}

pub(crate) fn damped_charge_step(q: &[f64], residual: &[f64], mixing: f64) -> Vec<f64> {
    q.iter()
        .zip(residual.iter())
        .map(|(q, r)| q + mixing * r)
        .collect()
}

/// Pulay DIIS on the SCC charge residual (the SCC realization of CDIIS): the new
/// input charges are the residual-minimizing combination of the history outputs.
#[derive(Clone, Debug)]
pub(crate) struct CdiisMixer {
    max_hist: usize,
    q_out: Vec<Vec<f64>>,
    err: Vec<Vec<f64>>,
}

impl CdiisMixer {
    pub(crate) fn new(max_hist: usize) -> Self {
        Self {
            max_hist: max_hist.max(2),
            q_out: Vec::new(),
            err: Vec::new(),
        }
    }

    pub(crate) fn next(
        &mut self,
        q_in: &[f64],
        q_out: &[f64],
        residual: &[f64],
    ) -> Option<Vec<f64>> {
        self.q_out.push(q_out.to_vec());
        self.err.push(residual.to_vec());
        while self.q_out.len() > self.max_hist {
            self.q_out.remove(0);
            self.err.remove(0);
        }
        let m = self.err.len();
        if m < 2 {
            return Some(damped_charge_step(q_in, residual, 0.4));
        }
        // Augmented Pulay system: [B -1; -1^T 0] [c; lambda] = [0; -1].
        let mut a = vec![vec![0.0; m + 1]; m + 1];
        let mut b = vec![0.0; m + 1];
        for i in 0..m {
            for j in 0..m {
                a[i][j] = dot(&self.err[i], &self.err[j]);
            }
            a[i][m] = -1.0;
            a[m][i] = -1.0;
        }
        b[m] = -1.0;
        let coeff = solve_linear(a, b)?;
        let n = q_out.len();
        let mut out = vec![0.0; n];
        for (i, qout) in self.q_out.iter().enumerate() {
            let c = coeff[i];
            for s in 0..n {
                out[s] += c * qout[s];
            }
        }
        Some(out)
    }
}

/// Conservative cap on the (linear) charge-mixing step when long-range exchange is on (see the SCF
/// loop): paired with the forced linear mixing it converges the closed-shell test set (H₂, water,
/// polar water, the dissociation dimer, the gradient-test geometry) out-of-the-box.
const EXCHANGE_MAX_MIXING: f64 = 0.1;

/// **Commutator (Pulay) DIIS on the density matrix** — SCF stabilization for the long-range Fock
/// exchange. The exchange Fock depends on the full (off-diagonal) density matrix, which the
/// charge-vector mixers ([`BroydenMixer`]/[`CdiisMixer`]) cannot see, so a density-matrix-dependent
/// Fock can oscillate. This extrapolates the density used to build the exchange Fock from the AO
/// **commutator error** `e = F P S − S P F` (which vanishes exactly at the SCF fixed point):
/// `P* = Σ_i c_i P_i` with the Pulay coefficients `c` minimizing `‖Σ_i c_i e_i‖` subject to
/// `Σ_i c_i = 1`. Engaged automatically when `lr_exchange` is on; standard CDIIS (Pulay 1980,
/// 1982). Until two errors are available it returns the input density (plain fixed-point).
pub(crate) struct CommutatorDiis {
    max_hist: usize,
    densities: Vec<Matrix>,
    errors: Vec<Vec<f64>>,
}

impl CommutatorDiis {
    pub(crate) fn new(max_hist: usize) -> Self {
        Self {
            max_hist: max_hist.max(2),
            densities: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Push the latest `(density, fock)` and return the DIIS-extrapolated density to build the next
    /// exchange Fock from. `overlap` is the AO overlap `S`.
    pub(crate) fn next(&mut self, density: &Matrix, fock: &Matrix, overlap: &Matrix) -> Matrix {
        // AO commutator error e = F P S − S P F (the SCF gradient; 0 at convergence).
        let fps = fock
            .matmul(density)
            .and_then(|fp| fp.matmul(overlap))
            .expect("FPS conformable");
        let spf = overlap
            .matmul(density)
            .and_then(|sp| sp.matmul(fock))
            .expect("SPF conformable");
        let e: Vec<f64> = fps
            .as_slice()
            .iter()
            .zip(spf.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        self.densities.push(density.clone());
        self.errors.push(e);
        while self.densities.len() > self.max_hist {
            self.densities.remove(0);
            self.errors.remove(0);
        }
        let m = self.errors.len();
        if m < 2 {
            return density.clone();
        }
        // Augmented Pulay system [B -1; -1^T 0][c; λ] = [0; -1], B_ij = <e_i, e_j>.
        let mut a = vec![vec![0.0; m + 1]; m + 1];
        let mut b = vec![0.0; m + 1];
        for i in 0..m {
            for j in 0..m {
                a[i][j] = dot(&self.errors[i], &self.errors[j]);
            }
            a[i][m] = -1.0;
            a[m][i] = -1.0;
        }
        b[m] = -1.0;
        let Some(coeff) = solve_linear(a, b) else {
            return density.clone();
        };
        let (rows, cols) = (density.rows(), density.cols());
        let mut out = Matrix::zeros(rows, cols);
        {
            let o = out.as_mut_slice();
            for (i, p) in self.densities.iter().enumerate() {
                let c = coeff[i];
                for (ok, pk) in o.iter_mut().zip(p.as_slice().iter()) {
                    *ok += c * pk;
                }
            }
        }
        out
    }
}

/// **Full-Fock commutator (Pulay) DIIS** — extrapolates the *entire* Fock matrix `F* = Σ_i c_i F_i`
/// from the AO commutator errors `R_i = F_i P_i S − S P_i F_i` (the SCF gradient; zero at the fixed
/// point). This is the robust driver for the density-matrix SCF used when long-range exchange is on:
/// charges, multipoles and the exchange all come from one trial density and are extrapolated
/// together with a single set of coefficients (Gaussian/Q-Chem-style CDIIS), instead of mixing the
/// projected charge vector. Until two errors are available it returns the input Fock.
pub(crate) struct FockDiis {
    max_hist: usize,
    focks: Vec<Matrix>,
    errors: Vec<Vec<f64>>,
}

impl FockDiis {
    pub(crate) fn new(max_hist: usize) -> Self {
        Self {
            max_hist: max_hist.max(2),
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Discard the extrapolation history (e.g. when switching solver / level-shift state, so the next
    /// step does not extrapolate across an inconsistent mapping).
    pub(crate) fn clear(&mut self) {
        self.focks.clear();
        self.errors.clear();
    }

    /// Push `(fock, error)` (error = the AO commutator `FPS − SPF`, flattened) and return the
    /// DIIS-extrapolated Fock.
    pub(crate) fn next(&mut self, fock: Matrix, error: Vec<f64>) -> Matrix {
        self.focks.push(fock.clone());
        self.errors.push(error);
        while self.focks.len() > self.max_hist {
            self.focks.remove(0);
            self.errors.remove(0);
        }
        let m = self.errors.len();
        if m < 2 {
            return fock;
        }
        let mut a = vec![vec![0.0; m + 1]; m + 1];
        let mut b = vec![0.0; m + 1];
        for i in 0..m {
            for j in 0..m {
                a[i][j] = dot(&self.errors[i], &self.errors[j]);
            }
            a[i][m] = -1.0;
            a[m][i] = -1.0;
        }
        b[m] = -1.0;
        let Some(coeff) = solve_linear(a, b) else {
            return fock;
        };
        let (rows, cols) = (fock.rows(), fock.cols());
        let mut out = Matrix::zeros(rows, cols);
        {
            let o = out.as_mut_slice();
            for (i, f) in self.focks.iter().enumerate() {
                let c = coeff[i];
                for (ok, fk) in o.iter_mut().zip(f.as_slice().iter()) {
                    *ok += c * fk;
                }
            }
        }
        out
    }
}

/// Euclidean projection of `v` onto the probability simplex `{c ≥ 0, Σ c = 1}`
/// (Duchi, Shalev-Shwartz, Singer & Chandra, ICML 2008). Used by [`AdiisMixer`].
fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut css = 0.0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

/// **ADIIS** (Augmented Roothaan–Hall / "augmented DIIS"; X. Hu & W. Yang, *J. Chem. Phys.* **132**,
/// 054109 (2010)) density-matrix mixer. Unlike commutator (Pulay/C-)DIIS — whose *unconstrained*
/// extrapolation can blow up on stiff, small-gap systems (the exact failure seen for metallic
/// complexes with the long-range exchange Fock) — ADIIS minimises a model of the energy
/// `E(c) = E_n + 2 Σ_i c_i ⟨D_i−D_n|F_n⟩ + Σ_ij c_i c_j ⟨D_i−D_n|F_j−F_n⟩` over the **probability
/// simplex** `c ≥ 0, Σ c = 1`, so the extrapolated Fock `Σ c_i F_i` is always a convex combination of
/// the history and the iteration cannot diverge. It is globally convergent far from the solution;
/// switch to C-DIIS near it for the fast quadratic tail. `⟨A|B⟩ = Tr(AB)` = the symmetric matrices'
/// elementwise dot.
pub(crate) struct AdiisMixer {
    max_hist: usize,
    focks: Vec<Matrix>,
    densities: Vec<Matrix>,
}

impl AdiisMixer {
    pub(crate) fn new(max_hist: usize) -> Self {
        Self {
            max_hist: max_hist.max(2),
            focks: Vec::new(),
            densities: Vec::new(),
        }
    }

    /// Discard the history (on the one-way ADIIS → C-DIIS switch, so DIIS starts clean).
    pub(crate) fn clear(&mut self) {
        self.focks.clear();
        self.densities.clear();
    }

    /// Push `(fock, density)` (the trial Fock built from `density`; both AO, symmetric) and return
    /// the ADIIS-extrapolated Fock `Σ c_i F_i`.
    pub(crate) fn next(&mut self, fock: Matrix, density: Matrix) -> Matrix {
        self.focks.push(fock.clone());
        self.densities.push(density);
        while self.focks.len() > self.max_hist {
            self.focks.remove(0);
            self.densities.remove(0);
        }
        let m = self.focks.len();
        if m < 2 {
            return fock;
        }
        let n_ref = m - 1; // reference = most recent entry
        let fn_ref = self.focks[n_ref].as_slice().to_vec();
        let dn_ref = self.densities[n_ref].as_slice().to_vec();
        // b_i = 2⟨D_i − D_n | F_n⟩ ; M_ij = ⟨D_i − D_n | F_j − F_n⟩
        let mut bvec = vec![0.0; m];
        let mut mmat = vec![vec![0.0; m]; m];
        for i in 0..m {
            let di = self.densities[i].as_slice();
            let ddi: Vec<f64> = di.iter().zip(dn_ref.iter()).map(|(a, b)| a - b).collect();
            bvec[i] = 2.0
                * ddi
                    .iter()
                    .zip(fn_ref.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            for j in 0..m {
                let fj = self.focks[j].as_slice();
                mmat[i][j] = ddi
                    .iter()
                    .zip(fj.iter().zip(fn_ref.iter()))
                    .map(|(d, (f, fr))| d * (f - fr))
                    .sum::<f64>();
            }
        }
        // Minimise E(c) = bᵀc + cᵀMc on the simplex by projected gradient descent (the precise `c`
        // is not critical — any convex combination is non-divergent — so a fixed schedule suffices).
        let mut c = vec![1.0 / m as f64; m];
        let mscale = (0..m).map(|i| mmat[i][i].abs()).fold(1.0e-12, f64::max);
        let lr = 0.25 / mscale;
        for _ in 0..500 {
            let grad: Vec<f64> = (0..m)
                .map(|i| {
                    bvec[i]
                        + (0..m)
                            .map(|j| (mmat[i][j] + mmat[j][i]) * c[j])
                            .sum::<f64>()
                })
                .collect();
            for (ci, gi) in c.iter_mut().zip(grad.iter()) {
                *ci -= lr * gi;
            }
            c = project_simplex(&c);
        }
        let (rows, cols) = (fock.rows(), fock.cols());
        let mut out = Matrix::zeros(rows, cols);
        {
            let o = out.as_mut_slice();
            for (i, f) in self.focks.iter().enumerate() {
                let ci = c[i];
                if ci == 0.0 {
                    continue;
                }
                for (ok, fk) in o.iter_mut().zip(f.as_slice().iter()) {
                    *ok += ci * fk;
                }
            }
        }
        out
    }
}

/// Full SCC response kernel `dv/dq = A + 3rd-order`, mirroring
/// [`crate::cphf::response_shell_scc_kernel`] but from raw SCC data: the
/// second-order Coulomb matrix `A` plus the on-atom third-order augmentation
/// `2 q_atom gamma3` on each atom's shell block.
pub(crate) fn scc_response_kernel(
    amat: &Matrix,
    shell_model: &ShellChargeModel,
    basis: &BasisSet,
    shell_charges: &[f64],
) -> Matrix {
    let mut kernel = amat.clone();
    let atomic = shell_model.atomic_charges(basis, shell_charges);
    for (atom, &qat) in atomic.iter().enumerate() {
        let count = shell_model.atom_shell_counts[atom];
        if count == 0 {
            continue;
        }
        let offset = shell_model.atom_offsets[atom];
        let add = 2.0 * qat * shell_model.hubbard_derivs[offset];
        for li in 0..count {
            for lj in 0..count {
                kernel[(offset + li, offset + lj)] += add;
            }
        }
    }
    kernel
}

/// Non-interacting Mulliken charge susceptibility times the SCC response kernel,
/// `(chi K)_st`, the Jacobian block `dq_out/dq_in` of the SCC fixed-point map.
/// `chi_st = sum_ia 0.5 (occ_i - occ_a)/(eps_i - eps_a) Q_ia,s Q_ia,t` and `K` is
/// the full `dv/dq` kernel ([`scc_response_kernel`]).
pub(crate) fn susceptibility_times_a(
    overlap: &Matrix,
    kernel: &Matrix,
    basis: &BasisSet,
    mo_coeff: &Matrix,
    orbital_energies: &[f64],
    occupations: &[f64],
) -> Result<Matrix> {
    let nsh = basis.shells.len();
    let space = CpxtbSpace::from_occupations(occupations)?;
    let q = transition_shell_charges(basis, mo_coeff, occupations, overlap)?;
    let mut chi = Matrix::zeros(nsh, nsh);
    for (row, &(i, a)) in space.pairs.iter().enumerate() {
        let denom = orbital_energies[i] - orbital_energies[a];
        if denom.abs() < 1.0e-9 {
            return Err(Gfn1Error::InvalidInput(
                "Newton susceptibility hit a near-degenerate occ-virt gap".to_string(),
            ));
        }
        let scale = 0.5 * (occupations[i] - occupations[a]) / denom;
        let qrow = &q[row];
        for s in 0..nsh {
            let qs = qrow[s];
            if qs == 0.0 {
                continue;
            }
            for t in 0..nsh {
                chi[(s, t)] += scale * qs * qrow[t];
            }
        }
    }
    chi.matmul(kernel)
}

/// Second-order (Newton) SCC step: solve `(I - chi K) dq = residual` and return
/// `q_in + dq`. Returns `None` (so the caller can fall back) on a singular or
/// non-finite solve.
fn newton_charge_step(
    overlap: &Matrix,
    amat: &Matrix,
    shell_model: &ShellChargeModel,
    basis: &BasisSet,
    step: &SccStep,
    q_in: &[f64],
    residual: &[f64],
) -> Option<Vec<f64>> {
    let nsh = basis.shells.len();
    let kernel = scc_response_kernel(amat, shell_model, basis, q_in);
    let chia = susceptibility_times_a(
        overlap,
        &kernel,
        basis,
        &step.mo_coeff,
        &step.orbital_energies,
        &step.occupations,
    )
    .ok()?;
    // M = I - chi A.
    let mut m = vec![vec![0.0; nsh]; nsh];
    for s in 0..nsh {
        for t in 0..nsh {
            m[s][t] = -chia[(s, t)];
        }
        m[s][s] += 1.0;
    }
    let dq = solve_linear(m, residual.to_vec())?;
    if dq.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(q_in.iter().zip(dq.iter()).map(|(q, d)| q + d).collect())
}

pub(crate) fn trusted_charge_vector(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() < 10.0)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in col + 1..n {
            let value = a[row][col].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs < 1.0e-18 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

#[derive(Clone, Debug)]
struct OccupationResult {
    occupations: Vec<f64>,
    fermi_level: f64,
    entropy_term: f64,
}

fn occupations(
    orbital_energies: &[f64],
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    kt: f64,
) -> Result<OccupationResult> {
    if let Some(channels) = spin_channels {
        return Ok(spin_constrained_occupations(orbital_energies, channels, kt));
    }
    if kt <= 0.0 {
        return Ok(aufbau_occupations(orbital_energies, nelec));
    }
    Ok(fermi_occupations(orbital_energies, nelec, kt))
}

fn aufbau_occupations(orbital_energies: &[f64], nelec: f64) -> OccupationResult {
    let mut occupations = vec![0.0; orbital_energies.len()];
    let mut remaining = nelec.max(0.0);
    for occ in &mut occupations {
        let fill = remaining.min(2.0);
        *occ = fill;
        remaining -= fill;
        if remaining <= 0.0 {
            break;
        }
    }
    let homo = occupations
        .iter()
        .rposition(|occ| *occ > 1.0e-12)
        .unwrap_or(0);
    OccupationResult {
        occupations,
        fermi_level: orbital_energies[homo],
        entropy_term: 0.0,
    }
}

fn fermi_occupations(orbital_energies: &[f64], nelec: f64, kt: f64) -> OccupationResult {
    if nelec <= 0.0 {
        return OccupationResult {
            occupations: vec![0.0; orbital_energies.len()],
            fermi_level: orbital_energies.first().copied().unwrap_or(0.0),
            entropy_term: 0.0,
        };
    }
    let capacity = 2.0 * orbital_energies.len() as f64;
    if nelec >= capacity {
        return OccupationResult {
            occupations: vec![2.0; orbital_energies.len()],
            fermi_level: orbital_energies.last().copied().unwrap_or(0.0),
            entropy_term: 0.0,
        };
    }

    let min_e = orbital_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_e = orbital_energies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut lo = min_e - 100.0 * kt - 10.0;
    let mut hi = max_e + 100.0 * kt + 10.0;
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let sum = orbital_energies
            .iter()
            .map(|eps| fermi_occ(*eps, mu, kt))
            .sum::<f64>();
        if sum < nelec {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let occupations = orbital_energies
        .iter()
        .map(|eps| fermi_occ(*eps, mu, kt))
        .collect::<Vec<_>>();
    let entropy_term = occupations
        .iter()
        .map(|occ| {
            let n = (0.5 * occ).clamp(1.0e-16, 1.0 - 1.0e-16);
            2.0 * kt * (n * n.ln() + (1.0 - n) * (1.0 - n).ln())
        })
        .sum::<f64>();
    OccupationResult {
        occupations,
        fermi_level: mu,
        entropy_term,
    }
}

fn fermi_occ(eps: f64, mu: f64, kt: f64) -> f64 {
    let x = ((eps - mu) / kt).clamp(-80.0, 80.0);
    2.0 / (1.0 + x.exp())
}

#[derive(Clone, Debug)]
struct SpinChannelOccupation {
    occupations: Vec<f64>,
    fermi_level: f64,
    entropy_term: f64,
}

fn spin_constrained_occupations(
    orbital_energies: &[f64],
    channels: SpinChannels,
    kt: f64,
) -> OccupationResult {
    let alpha = if kt <= 0.0 {
        aufbau_spin_channel(orbital_energies, channels.alpha)
    } else {
        fermi_spin_channel(orbital_energies, channels.alpha, kt)
    };
    let beta = if kt <= 0.0 {
        aufbau_spin_channel(orbital_energies, channels.beta)
    } else {
        fermi_spin_channel(orbital_energies, channels.beta, kt)
    };
    let occupations = alpha
        .occupations
        .iter()
        .zip(beta.occupations.iter())
        .map(|(a, b)| a + b)
        .collect();
    OccupationResult {
        occupations,
        fermi_level: 0.5 * (alpha.fermi_level + beta.fermi_level),
        entropy_term: alpha.entropy_term + beta.entropy_term,
    }
}

fn aufbau_spin_channel(orbital_energies: &[f64], electrons: f64) -> SpinChannelOccupation {
    let mut occupations = vec![0.0; orbital_energies.len()];
    let mut remaining = electrons.max(0.0);
    for occ in &mut occupations {
        let fill = remaining.min(1.0);
        *occ = fill;
        remaining -= fill;
        if remaining <= 0.0 {
            break;
        }
    }
    let homo = occupations
        .iter()
        .rposition(|occ| *occ > 1.0e-12)
        .unwrap_or(0);
    SpinChannelOccupation {
        occupations,
        fermi_level: orbital_energies.get(homo).copied().unwrap_or(0.0),
        entropy_term: 0.0,
    }
}

fn fermi_spin_channel(orbital_energies: &[f64], electrons: f64, kt: f64) -> SpinChannelOccupation {
    if electrons <= 0.0 {
        return SpinChannelOccupation {
            occupations: vec![0.0; orbital_energies.len()],
            fermi_level: orbital_energies.first().copied().unwrap_or(0.0),
            entropy_term: 0.0,
        };
    }
    let capacity = orbital_energies.len() as f64;
    if electrons >= capacity {
        return SpinChannelOccupation {
            occupations: vec![1.0; orbital_energies.len()],
            fermi_level: orbital_energies.last().copied().unwrap_or(0.0),
            entropy_term: 0.0,
        };
    }

    let min_e = orbital_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_e = orbital_energies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut lo = min_e - 100.0 * kt - 10.0;
    let mut hi = max_e + 100.0 * kt + 10.0;
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let sum = orbital_energies
            .iter()
            .map(|eps| fermi_occ_spin(*eps, mu, kt))
            .sum::<f64>();
        if sum < electrons {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let occupations = orbital_energies
        .iter()
        .map(|eps| fermi_occ_spin(*eps, mu, kt))
        .collect::<Vec<_>>();
    let entropy_term = occupations
        .iter()
        .map(|occ| spin_entropy_term(*occ, kt))
        .sum::<f64>();
    SpinChannelOccupation {
        occupations,
        fermi_level: mu,
        entropy_term,
    }
}

fn fermi_occ_spin(eps: f64, mu: f64, kt: f64) -> f64 {
    let x = ((eps - mu) / kt).clamp(-80.0, 80.0);
    1.0 / (1.0 + x.exp())
}

fn spin_entropy_term(occupation: f64, kt: f64) -> f64 {
    let f = occupation.clamp(1.0e-16, 1.0 - 1.0e-16);
    kt * (f * f.ln() + (1.0 - f) * (1.0 - f).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CommutatorDiis` correctness: at an SCF fixed point the AO commutator `F P S − S P F`
    /// vanishes, so the DIIS error is ~0 and the extrapolation returns the (already-converged)
    /// density unchanged. With `S = I` and `P` a spectral projector of a symmetric `F`, `[F,P]=0`.
    #[test]
    fn commutator_diis_zero_error_at_fixed_point() {
        // Symmetric 3×3 F and S = I.
        let f = {
            let mut m = Matrix::zeros(3, 3);
            let v = [1.3, -0.4, 0.7, -0.4, 2.1, 0.5, 0.7, 0.5, -0.9];
            for i in 0..3 {
                for j in 0..3 {
                    m[(i, j)] = v[i * 3 + j];
                }
            }
            m
        };
        let mut s = Matrix::zeros(3, 3);
        for i in 0..3 {
            s[(i, i)] = 1.0;
        }
        // P = projector onto the lowest eigenvector of F (so F and P commute).
        let eig = crate::linalg::symmetric_eigen_jacobi(&f, 1.0e-13, 100).unwrap();
        let mut p = Matrix::zeros(3, 3);
        let c0 = eig.vectors.column(0);
        for i in 0..3 {
            for j in 0..3 {
                p[(i, j)] = 2.0 * c0[i] * c0[j]; // doubly occupied
            }
        }
        let mut diis = CommutatorDiis::new(5);
        // First push: m<2 → returns the input density; the commutator error must be ~0.
        let out1 = diis.next(&p, &f, &s);
        assert!(
            out1.max_abs_diff(&p) < 1.0e-12,
            "DIIS altered a converged density"
        );
        // Second push (same fixed point): the extrapolated density stays the fixed point.
        let out2 = diis.next(&p, &f, &s);
        assert!(
            out2.max_abs_diff(&p) < 1.0e-8,
            "DIIS drifted from the fixed point: {}",
            out2.max_abs_diff(&p)
        );
    }

    #[test]
    fn spin_multiplicity_triplet_sets_two_singly_occupied_orbitals() {
        let spin = resolve_spin_channels(4.0, Some(3), 4).unwrap();
        let occ = occupations(&[-1.0, -0.5, 0.2, 0.3], 4.0, spin, 0.0).unwrap();
        assert_eq!(occ.occupations, vec![2.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn spin_multiplicity_rejects_wrong_electron_parity() {
        let err = resolve_spin_channels(4.0, Some(2), 4).unwrap_err();
        assert!(format!("{err}").contains("incompatible"));
    }

    #[test]
    fn electron_count_cannot_exceed_basis_capacity() {
        let err = validate_electron_count(7.0, 3).unwrap_err();
        assert!(format!("{err}").contains("exceeds basis capacity"));
    }

    // The Newton susceptibility-times-A must equal the finite difference of the
    // SCC fixed-point map dq_out/dq_in (this nails the prefactor of chi).
    #[test]
    fn newton_jacobian_matches_finite_difference() {
        use crate::basis::BasisOptions;
        use crate::hamiltonian::build_h0;
        use crate::linalg::lowdin_orthogonalizer;
        use crate::params::Gfn1Parameters;
        use crate::system::PeriodicSystem;

        let Ok(path) = std::env::var(crate::params::GFN1_PARAM_ENV) else {
            return;
        };
        let params = Gfn1Parameters::from_file(path).unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let options = ElectronicOptions::default();
        let basis = BasisSet::build(
            &system,
            &params,
            BasisOptions {
                nprim: options.nprim,
            },
        )
        .unwrap();
        let core = build_h0(&system, &basis, &params, &options.hamiltonian).unwrap();
        let shell_model = ShellChargeModel::build(&system, &basis, &params).unwrap();
        let amat = effective_coulomb_matrix(&system, &basis, &shell_model);
        let orth = lowdin_orthogonalizer(&core.integrals.overlap, options.eigen_tolerance).unwrap();
        let nelec = basis.total_reference_electrons;
        let nsh = basis.shells.len();
        let overlap = &core.integrals.overlap;

        // Evaluate one fixed-point map step q_out(q_in) at T = 0 (clean integer occ).
        let q_out = |q_in: &[f64]| -> Vec<f64> {
            scc_step(
                &basis,
                overlap,
                &core.h0,
                &orth,
                &shell_model,
                &amat,
                q_in,
                nelec,
                None,
                0.0,
                1.0e-12,
                None,
                None,
                0.0,
                None,
                None,
            )
            .unwrap()
            .shell_charges
        };

        // A non-trivial input: one fixed-point sweep from zero.
        let q_in = q_out(&vec![0.0; nsh]);
        let step = scc_step(
            &basis,
            overlap,
            &core.h0,
            &orth,
            &shell_model,
            &amat,
            &q_in,
            nelec,
            None,
            0.0,
            1.0e-12,
            None,
            None,
            0.0,
            None,
            None,
        )
        .unwrap();
        let kernel = scc_response_kernel(&amat, &shell_model, &basis, &q_in);
        let chia = susceptibility_times_a(
            overlap,
            &kernel,
            &basis,
            &step.mo_coeff,
            &step.orbital_energies,
            &step.occupations,
        )
        .unwrap();

        let delta = 1.0e-5;
        let mut max_err = 0.0_f64;
        for t in 0..nsh {
            let mut qp = q_in.clone();
            let mut qm = q_in.clone();
            qp[t] += delta;
            qm[t] -= delta;
            let op = q_out(&qp);
            let om = q_out(&qm);
            for s in 0..nsh {
                let fd = (op[s] - om[s]) / (2.0 * delta);
                max_err = max_err.max((fd - chia[(s, t)]).abs());
            }
        }
        assert!(
            max_err < 1.0e-5,
            "Newton Jacobian chi*K vs finite difference max error {max_err:.3e}"
        );
    }
}
