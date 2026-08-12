// SPDX-License-Identifier: GPL-3.0-or-later
//! **Berry-phase bulk polarization** for periodic GFN1-xTB (the modern theory of
//! polarization): the King-Smith–Vanderbilt (KSV) discretised k-string Berry phase
//! and the Resta single-point form for Γ-only sampling.
//!
//! # Why a Berry phase
//!
//! In a periodic solid the dipole `∫ r ρ(r)` of the reference cell is not a bulk
//! observable: it depends on where the cell boundary is drawn. The bulk
//! polarization is instead a **Berry phase** of the occupied Bloch manifold, which
//! is well defined *modulo a polarization quantum* `e R / V` (`R` a lattice
//! vector). Everything in this module returns both the raw phase and the
//! branch-reduced value, plus the quantum, so a caller can never confuse the two.
//!
//! # What is computed
//!
//! For each lattice direction `d` (reciprocal partner `b_d`, `b_d · a_e = 2π δ_de`):
//!
//! ```text
//!   phi_d = Im ln  prod_j  det M^(j),
//!   M^(j)_mn = <psi_{m, k_{j+1}} | e^{i dk . r} | psi_{n, k_j}>,   dk = b_d / N_d
//! ```
//!
//! over a string of `N_d` k-points `k_j = k_0 + j b_d / N_d`, averaged over the
//! strings of the perpendicular mesh. `N_d = 1` is the **Resta single-point** form
//! `phi_d = Im ln det <psi_m | e^{i b_d . r} | psi_n>`.
//!
//! In the LCAO cell gauge used throughout `src/pbc` — Bloch AOs
//! `phi_mu(k) = sum_T e^{i k . T} chi_mu(r - tau_mu - T)`, so `H(k+G) = H(k)` and
//! `C(k+G) = C(k)` **exactly** — the wrap-around link `k_{N-1} -> k_0 + b_d` needs
//! no periodic-gauge correction: it is the same expression with the coefficients of
//! `k_0`. The AO ingredient is a Bloch sum of **boosted AO overlaps**
//!
//! ```text
//!   M^AO(k_j)_{mu nu} = sum_T e^{i k_j . T} <chi_mu(tau_mu)| e^{i dk . r} |chi_nu(tau_nu + T)>
//! ```
//!
//! and `M^(j) = C(k_{j+1})^H M^AO(k_j) C(k_j)` restricted to the occupied bands.
//! At `dk = 0` this collapses to `C^H S(k) C = 1`, i.e. `phi = 0`.
//!
//! # Reused integral machinery
//!
//! The boosted overlap `<chi_mu| e^{i q . r} |chi_nu>` is **not** new integral code.
//! It is the same complex Gaussian product theorem the London/GIAO overlap in
//! [`crate::magnetic`] already implements: completing the square in
//! `exp(-zeta |r - P|^2 + i q . r)` gives the complex centre `Pbar = P + (i/2 zeta) q`
//! and prefactor `exp(i P.q - q.q/(4 zeta))`, which is the LAO kernel at
//! `chi = -q`. This module calls [`crate::magnetic::boosted_overlap_pair`], the thin
//! public wrapper over that shared kernel.
//!
//! # Sign and quantum conventions
//!
//! ```text
//!   Phi_d = 2 pi sum_A z_A s_{A,d}  -  n_spin * phi_d        (total phase)
//!   P     = (e / (2 pi V)) sum_d a_d Phi_d
//! ```
//!
//! `z_A` are the GFN1 valence (core) charges [`crate::basis::BasisSet::reference_electrons`],
//! `s_{A,d}` the *unwrapped* fractional coordinate of atom `A`, and `n_spin = 2` the
//! closed-shell spin degeneracy. The signs are fixed by the molecular limit: for a
//! localized closed-shell density in a large box, `phi_d -> b_d . sum_n <r>_n`, so
//! `-n_spin phi_d` reproduces the electronic dipole `-2 sum_n <r>_n` exactly.
//!
//! Because `phi_d` is only recovered modulo `2 pi` and enters multiplied by
//! `n_spin`, the **spin-restricted** polarization quantum is
//! `n_spin e a_d / V = 2 e a_d / V`: a restricted calculation moves electrons in
//! pairs. `Phi_d` is therefore reduced into `(-2 pi, 2 pi]`, one quantum wide.
//! A centrosymmetric crystal has `P = 0` modulo **half** that quantum (`e a_d / V`),
//! i.e. `Phi_d = 0 mod 2 pi`.
//!
//! # References
//!
//! - R. D. King-Smith and D. Vanderbilt, "Theory of polarization of crystalline
//!   solids", *Phys. Rev. B* **47**, 1651 (1993).
//! - R. Resta, "Quantum-Mechanical Position Operator in Extended Systems",
//!   *Phys. Rev. Lett.* **80**, 1800 (1998) (the single-point form).
//! - R. Resta and D. Vanderbilt, "Theory of Polarization: A Modern Approach", in
//!   *Physics of Ferroelectrics*, Topics Appl. Physics **105**, 31 (2007).

use crate::basis::BasisSet;
use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::lattice::ImageOffset;
use crate::linalg::Matrix;
use crate::magnetic::boosted_overlap_pair;
use crate::math::Vec3;
use crate::model::image_translations;
use crate::params::Gfn1Parameters;
use crate::pbc::complex::{hermitian_generalized_eigen, CMatrix, KEigen};
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::fock_at_k;
use crate::pbc::{run_pbc_scc, KMesh, PbcOptions, PbcSccResult};
use crate::system::PeriodicSystem;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Polarization in atomic units (`e / bohr^2`) converted to SI (`C / m^2`):
/// `e / a0^2` with the SI-exact elementary charge `e = 1.602176634e-19 C` and the
/// CODATA-2018 Bohr radius `a0 = 5.29177210903e-11 m`. Evaluates to `57.2147…`.
/// Written as the ratio rather than a transcribed decimal so it cannot drift.
pub const POLARIZATION_AU_TO_C_PER_M2: f64 =
    1.602_176_634e-19 / (5.291_772_109_03e-11 * 5.291_772_109_03e-11);

/// Which discretisation produced a [`BerryPolarizationResult`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BerryPolarizationMethod {
    /// King-Smith–Vanderbilt: a product of link determinants along a k-string,
    /// averaged over the perpendicular mesh.
    KingSmithVanderbilt,
    /// Resta single-point form: one determinant per direction at Γ, with the boost
    /// set to the full reciprocal vector `b_d`.
    Resta,
}

/// Method selection for [`pbc_berry_polarization`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BerryMethodSelector {
    /// Resta when the Berry mesh is Γ-only on every periodic axis, KSV otherwise.
    Auto,
    /// Force the Resta single-point form (ignores the Berry mesh).
    Resta,
    /// Force the KSV string form (a `[1,1,1]` mesh then means one k-point per
    /// string, which must reproduce Resta).
    KingSmithVanderbilt,
}

/// Options for [`pbc_berry_polarization`].
#[derive(Clone, Debug)]
pub struct BerryPolarizationOptions {
    /// Berry-phase k-mesh. `mesh[d]` is the number of k-points on each string
    /// *along* direction `d`; the strings themselves are enumerated by the other
    /// two entries (`mesh[e1] * mesh[e2]` parallel strings). Non-periodic axes are
    /// collapsed to a single point. Defaults to `[1, 1, 1]` (Γ-only).
    pub mesh: [usize; 3],
    /// Resta vs. KSV. Defaults to [`BerryMethodSelector::Auto`].
    pub method: BerryMethodSelector,
    /// Lattice directions to evaluate. Non-periodic axes are skipped regardless.
    pub directions: [bool; 3],
    /// Largest tolerated deviation of any band occupation from an integer. Beyond
    /// this the calculation is rejected: polarization is undefined for a metal.
    pub occupation_tolerance: f64,
    /// Smallest tolerated HOMO/LUMO band gap (Hartree) at any Berry k-point.
    pub min_band_gap: f64,
    /// AO image cutoff (Bohr) for the boosted-overlap lattice sums. `None` reuses
    /// [`PbcOptions::ao_cutoff`].
    pub ao_cutoff: Option<f64>,
}

impl Default for BerryPolarizationOptions {
    fn default() -> Self {
        Self {
            mesh: [1, 1, 1],
            method: BerryMethodSelector::Auto,
            directions: [true; 3],
            occupation_tolerance: 1.0e-6,
            min_band_gap: 1.0e-6,
            ao_cutoff: None,
        }
    }
}

impl BerryPolarizationOptions {
    /// Take the Berry mesh from an existing [`KMesh`] (the usual choice: sample the
    /// Berry phase on the same grid the SCC used).
    pub fn from_kmesh(kmesh: KMesh) -> Self {
        Self {
            mesh: kmesh.size,
            ..Self::default()
        }
    }
}

/// Berry-phase polarization of a periodic cell.
///
/// All phases are in radians; polarizations in atomic units (`e / bohr^2`, use
/// [`POLARIZATION_AU_TO_C_PER_M2`] for SI); dipoles in `e bohr`.
#[derive(Clone, Debug)]
pub struct BerryPolarizationResult {
    /// Discretisation actually used.
    pub method: BerryPolarizationMethod,
    /// Berry k-mesh actually used (non-periodic axes collapsed to 1).
    pub mesh: [usize; 3],
    /// Which lattice directions carry a result; the others are left at zero.
    pub evaluated: [bool; 3],
    /// Per-spin-channel electronic Berry phase `phi_d`, in `(-pi, pi]`.
    pub electronic_phase: [f64; 3],
    /// Ionic phase `2 pi sum_A z_A s_{A,d}` (unwrapped: the raw fractional
    /// coordinates enter, not the wrapped ones).
    pub ionic_phase: [f64; 3],
    /// Raw total phase `Phi_d = ionic_phase - n_spin * electronic_phase`, unwrapped.
    pub total_phase_raw: [f64; 3],
    /// `total_phase_raw` reduced into `(-2 pi, 2 pi]`, i.e. one polarization
    /// quantum wide for the spin-restricted formalism.
    pub total_phase_reduced: [f64; 3],
    /// Polarization from [`Self::total_phase_reduced`] (`e / bohr^2`).
    pub polarization: [f64; 3],
    /// Polarization from [`Self::total_phase_raw`], i.e. the branch as computed.
    pub polarization_raw: [f64; 3],
    /// `P * V` from the reduced phase (`e bohr`) — directly comparable with a
    /// molecular dipole in the large-box limit.
    pub dipole: [f64; 3],
    /// Polarization quantum vectors `n_spin e a_d / V` (`e / bohr^2`), one per
    /// lattice direction. `P` is only defined modulo integer combinations of these.
    pub quantum: [[f64; 3]; 3],
    /// Cell volume (bohr^3).
    pub volume: f64,
    /// Spin degeneracy folded into the electronic phase (2 for the restricted path).
    pub spin_degeneracy: f64,
    /// Number of occupied bands per k-point.
    pub occupied_bands: usize,
    /// Per-string electronic phases (radians) for each direction — the spread over
    /// strings is the honest convergence diagnostic for the perpendicular mesh.
    pub string_phases: [Vec<f64>; 3],
    /// Smallest band gap (Hartree) seen over all Berry k-points.
    pub min_band_gap: f64,
}

/// Berry-phase bulk polarization of a periodic GFN1-xTB cell.
///
/// Runs the periodic SCC (`run_pbc_scc` with `pbc`), then evaluates the
/// King-Smith–Vanderbilt k-string Berry phase — or the Resta single-point form for
/// Γ-only sampling — for every requested periodic lattice direction. See the
/// [module documentation](self) for the conventions, and
/// [`BerryPolarizationResult`] for what comes back.
///
/// # Errors
///
/// - non-periodic system, or an unconverged SCC;
/// - `options.multipole` (the on-site multipole Fock is not reconstructible from
///   [`PbcSccResult`], so the Berry states would not be the SCC states);
/// - a charged cell (its polarization is origin dependent);
/// - fractional occupations / a closed band gap (metallic polarization is
///   ill-defined).
pub fn pbc_berry_polarization(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
) -> Result<BerryPolarizationResult> {
    let _profile = crate::profile::scope("pbc.polarization.berry");
    let lattice = system.lattice.as_ref().ok_or_else(|| {
        Gfn1Error::InvalidInput(
            "pbc_berry_polarization requires a periodic system with a lattice".to_string(),
        )
    })?;
    if options.multipole {
        return Err(Gfn1Error::InvalidInput(
            "pbc_berry_polarization does not support the on-site multipole SCC \
             (the multipole Fock is not carried by PbcSccResult, so the Berry states \
             would not be the SCC states); rerun with ElectronicOptions::multipole = false"
                .to_string(),
        ));
    }
    let charge = options.charge.unwrap_or(system.charge);
    if charge.abs() > 1.0e-8 {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_berry_polarization requires a neutral cell (net charge {charge:.3}); \
             the polarization of a charged periodic cell depends on the coordinate origin"
        )));
    }

    let scf = run_pbc_scc(system, params, options, pbc)?;
    if !scf.converged {
        return Err(Gfn1Error::InvalidInput(
            "pbc_berry_polarization requires a converged periodic SCC".to_string(),
        ));
    }

    let basis = &scf.basis;
    let n = basis.len();
    let nocc_f = scf.nelec / 2.0;
    let nocc = nocc_f.round() as usize;
    if (nocc_f - nocc as f64).abs() > 1.0e-8 || nocc == 0 || nocc >= n {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_berry_polarization requires an integer closed-shell band filling \
             (got {nocc_f} occupied bands out of {n}); open-shell and fractional \
             fillings have no well-defined Berry phase"
        )));
    }

    // AO-resolved converged effective potential (SCC + external field), exactly the
    // one `scc_step` used, so `fock_at_k` reproduces the SCC Fock at any k.
    let mut vao = vec![0.0_f64; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }

    let periodic = lattice.periodic;
    let mut mesh = [
        if periodic[0] { berry.mesh[0].max(1) } else { 1 },
        if periodic[1] { berry.mesh[1].max(1) } else { 1 },
        if periodic[2] { berry.mesh[2].max(1) } else { 1 },
    ];
    let method = match berry.method {
        BerryMethodSelector::Resta => BerryPolarizationMethod::Resta,
        BerryMethodSelector::KingSmithVanderbilt => BerryPolarizationMethod::KingSmithVanderbilt,
        BerryMethodSelector::Auto => {
            if mesh == [1, 1, 1] {
                BerryPolarizationMethod::Resta
            } else {
                BerryPolarizationMethod::KingSmithVanderbilt
            }
        }
    };
    if method == BerryPolarizationMethod::Resta {
        // The single-point form is by construction Gamma-only: report what was
        // actually sampled, not what was asked for.
        mesh = [1, 1, 1];
    }
    let ao_cutoff = berry.ao_cutoff.unwrap_or(pbc.ao_cutoff);
    let recip = lattice.reciprocal_vectors_2pi();
    let volume = lattice.volume();
    let spin_degeneracy = 2.0_f64;

    let mut electronic_phase = [0.0_f64; 3];
    let mut ionic_phase = [0.0_f64; 3];
    let mut evaluated = [false; 3];
    let mut string_phases: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut min_gap = f64::INFINITY;

    for d in 0..3 {
        if !periodic[d] || !berry.directions[d] {
            continue;
        }
        // Ionic phase from the GFN1 valence (core) charges at the *unwrapped*
        // fractional coordinates: wrapping would only shift Phi by 2 pi z_A, i.e.
        // whole quanta, but the unwrapped form makes the lattice-translation
        // bookkeeping exact and inspectable.
        let mut phi_ion = 0.0_f64;
        for (a, atom) in system.atoms.iter().enumerate() {
            let s = lattice.frac_of(atom.position);
            phi_ion += basis.reference_electrons[a] * s.to_array()[d];
        }
        ionic_phase[d] = 2.0 * PI * phi_ion;

        let strings = match method {
            BerryPolarizationMethod::Resta => 1,
            BerryPolarizationMethod::KingSmithVanderbilt => {
                let (e1, e2) = perpendicular_axes(d);
                mesh[e1] * mesh[e2]
            }
        };
        let n_string = match method {
            BerryPolarizationMethod::Resta => 1,
            BerryPolarizationMethod::KingSmithVanderbilt => mesh[d],
        };

        // The boost is the same for every link of every string along `d`, so the
        // (expensive) boosted AO image blocks are built exactly once per direction.
        let dk = recip[d] / n_string as f64;
        let blocks = boosted_image_blocks(system, basis, ao_cutoff, dk);

        let raw: Vec<(f64, f64)> = (0..strings)
            .into_par_iter()
            .map(|s| -> Result<(f64, f64)> {
                let base = string_base(d, s, mesh);
                string_berry_phase(
                    &scf,
                    &vao,
                    &blocks,
                    n,
                    nocc,
                    d,
                    n_string,
                    base,
                    options.eigen_tolerance,
                    berry.min_band_gap,
                    berry.occupation_tolerance,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Unwrap every string onto the branch of the first one before averaging:
        // for an insulator the strings agree, and a 2 pi mismatch is a branch
        // artefact of `atan2`, not physics.
        let reference = raw[0].0;
        let mut mean = 0.0_f64;
        let mut phases = Vec::with_capacity(raw.len());
        for (phi, gap) in &raw {
            let aligned = phi + 2.0 * PI * ((reference - phi) / (2.0 * PI)).round();
            phases.push(aligned);
            mean += aligned;
            min_gap = min_gap.min(*gap);
        }
        mean /= raw.len() as f64;
        electronic_phase[d] = reduce_branch(mean, 2.0 * PI);
        string_phases[d] = phases;
        evaluated[d] = true;
    }

    let mut total_phase_raw = [0.0_f64; 3];
    let mut total_phase_reduced = [0.0_f64; 3];
    let mut polarization = [0.0_f64; 3];
    let mut polarization_raw = [0.0_f64; 3];
    let mut dipole = [0.0_f64; 3];
    let mut quantum = [[0.0_f64; 3]; 3];
    let inv = 1.0 / (2.0 * PI * volume);
    for d in 0..3 {
        let a_d = lattice.cell.col[d];
        for c in 0..3 {
            quantum[d][c] = spin_degeneracy * a_d.to_array()[c] / volume;
        }
        if !evaluated[d] {
            continue;
        }
        let raw = ionic_phase[d] - spin_degeneracy * electronic_phase[d];
        let reduced = reduce_branch(raw, spin_degeneracy * 2.0 * PI);
        total_phase_raw[d] = raw;
        total_phase_reduced[d] = reduced;
        for c in 0..3 {
            polarization[c] += inv * a_d.to_array()[c] * reduced;
            polarization_raw[c] += inv * a_d.to_array()[c] * raw;
        }
    }
    for c in 0..3 {
        dipole[c] = polarization[c] * volume;
    }

    Ok(BerryPolarizationResult {
        method,
        mesh,
        evaluated,
        electronic_phase,
        ionic_phase,
        total_phase_raw,
        total_phase_reduced,
        polarization,
        polarization_raw,
        dipole,
        quantum,
        volume,
        spin_degeneracy,
        occupied_bands: nocc,
        string_phases,
        min_band_gap: if min_gap.is_finite() { min_gap } else { 0.0 },
    })
}

/// The two lattice axes perpendicular (in the index sense) to `d`.
fn perpendicular_axes(d: usize) -> (usize, usize) {
    match d {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    }
}

/// Γ-centred fractional origin of string `s` along direction `d`.
fn string_base(d: usize, s: usize, mesh: [usize; 3]) -> [f64; 3] {
    let (e1, e2) = perpendicular_axes(d);
    let mut base = [0.0_f64; 3];
    base[e1] = (s / mesh[e2]) as f64 / mesh[e1] as f64;
    base[e2] = (s % mesh[e2]) as f64 / mesh[e2] as f64;
    base
}

/// One boosted real-space AO block `<chi_mu(tau_mu)| e^{i q . r} |chi_nu(tau_nu + T)>`.
#[derive(Clone, Copy, Debug)]
struct BoostPair {
    mu: usize,
    nu: usize,
    offset: ImageOffset,
    re: f64,
    im: f64,
}

/// Boosted AO overlap blocks for every lattice image inside `cutoff`.
///
/// Mirrors [`crate::pbc::bloch::BlochBuilder::build`]: the same image enumeration,
/// the same `exp(-40)` Gaussian distance screen (a boost has unit modulus and the
/// complex GPT adds `exp(-q.q/4 zeta) <= 1`, so the field-free bound is still
/// conservative), and the same per-image ordering so the Bloch sums accumulate
/// deterministically. The integral itself is [`boosted_overlap_pair`].
fn boosted_image_blocks(
    system: &PeriodicSystem,
    basis: &BasisSet,
    cutoff: f64,
    q: Vec3,
) -> Vec<BoostPair> {
    let _profile = crate::profile::scope("pbc.polarization.boost_blocks");
    let nat = system.atoms.len();
    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            if p.exponent < atom_min_exp[ao.atom_index] {
                atom_min_exp[ao.atom_index] = p.exponent;
            }
        }
    }
    let images = image_translations(system, cutoff);
    let cutoff2 = if cutoff > 0.0 {
        cutoff * cutoff
    } else {
        f64::INFINITY
    };
    let per_image: Vec<Vec<BoostPair>> = images
        .par_iter()
        .map(|(offset, translation)| {
            let is_origin = offset.is_origin();
            let mut local = Vec::new();
            for a in 0..nat {
                let ra = system.atoms[a].position;
                for b in 0..nat {
                    let rb = system.atoms[b].position + *translation;
                    let r2 = (ra - rb).norm2();
                    if !(is_origin && a == b) {
                        if r2 > cutoff2 {
                            continue;
                        }
                        let ea = atom_min_exp[a];
                        let eb = atom_min_exp[b];
                        if r2 * ea * eb > 40.0 * (ea + eb) {
                            continue;
                        }
                    }
                    for &mu in &atom_aos[a] {
                        for &nu in &atom_aos[b] {
                            let (re, im) =
                                boosted_overlap_pair(&basis.aos[mu], &basis.aos[nu], ra, rb, q);
                            if re == 0.0 && im == 0.0 {
                                continue;
                            }
                            local.push(BoostPair {
                                mu,
                                nu,
                                offset: *offset,
                                re,
                                im,
                            });
                        }
                    }
                }
            }
            local
        })
        .collect();
    per_image.into_iter().flatten().collect()
}

/// Bloch sum `sum_T e^{i k . T} B(T)` of the boosted blocks. Unlike `S(k)` this is
/// **not** Hermitian for `q != 0` (its Hermitian conjugate is the `-q` sum), so it
/// is deliberately left unsymmetrised.
fn boost_at_k(blocks: &[BoostPair], n: usize, fractional: [f64; 3]) -> CMatrix {
    let mut out = CMatrix::zeros(n);
    for p in blocks {
        let (c, s) = bloch_phase(fractional, p.offset);
        out.accumulate(p.mu, p.nu, p.re * c - p.im * s, p.re * s + p.im * c);
    }
    out
}

/// Occupied complex Bloch coefficients `C(k)` as an `n x nocc` (re, im) pair.
/// Physical band `b` is real-embedding column `2b` (see [`crate::pbc::complex`]).
/// No gauge fixing is needed: each `C(k_j)` appears once as bra and once as ket
/// around the closed string, so any per-band phase (and any `U(nocc)` mixing of
/// degenerate bands) cancels exactly from the product of link determinants.
fn occupied_coefficients(eig: &KEigen, n: usize, nocc: usize) -> (Matrix, Matrix) {
    let mut re = Matrix::zeros(n, nocc);
    let mut im = Matrix::zeros(n, nocc);
    for b in 0..nocc {
        for mu in 0..n {
            re[(mu, b)] = eig.vectors[(mu, 2 * b)];
            im[(mu, b)] = eig.vectors[(n + mu, 2 * b)];
        }
    }
    (re, im)
}

/// Berry phase of one k-string: returns `(unwrapped phase, smallest band gap)`.
///
/// The link phases are summed rather than taken from `Im ln` of the accumulated
/// product; the two differ only by multiples of `2 pi` (whole polarization
/// quanta), and the summed form is the numerically stable one. The caller reduces
/// the branch.
#[allow(clippy::too_many_arguments)]
fn string_berry_phase(
    scf: &PbcSccResult,
    vao: &[f64],
    blocks: &[BoostPair],
    n: usize,
    nocc: usize,
    d: usize,
    n_string: usize,
    base: [f64; 3],
    eigen_tol: f64,
    min_gap: f64,
    occupation_tolerance: f64,
) -> Result<(f64, f64)> {
    let kt = scf.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
    let mut coeffs: Vec<(Matrix, Matrix)> = Vec::with_capacity(n_string);
    let mut fractionals: Vec<[f64; 3]> = Vec::with_capacity(n_string);
    let mut gap_seen = f64::INFINITY;
    for j in 0..n_string {
        let mut frac = base;
        frac[d] += j as f64 / n_string as f64;
        let (h0k, sk) = scf.bloch.h_s_at_k(frac);
        let fock = fock_at_k(&h0k, &sk, vao);
        let eig = hermitian_generalized_eigen(&fock, &sk, eigen_tol)?;
        let gap = eig.values[2 * nocc] - eig.values[2 * nocc - 1];
        if gap < min_gap {
            return Err(Gfn1Error::InvalidInput(format!(
                "pbc_berry_polarization requires integer band occupations: the band gap at \
                 k = ({:.4}, {:.4}, {:.4}) is {gap:.3e} Hartree, below the {min_gap:.3e} \
                 threshold. A metallic (fractionally occupied) manifold has no well-defined \
                 Berry phase",
                frac[0], frac[1], frac[2]
            )));
        }
        if kt > 0.0 {
            for (band, target) in (0..n).map(|b| (b, if b < nocc { 1.0 } else { 0.0 })) {
                let x = (eig.values[2 * band] - scf.fermi_level) / kt;
                let f = if x > 40.0 {
                    0.0
                } else if x < -40.0 {
                    1.0
                } else {
                    1.0 / (1.0 + x.exp())
                };
                if (f - target).abs() > occupation_tolerance {
                    return Err(Gfn1Error::InvalidInput(format!(
                        "pbc_berry_polarization requires integer band occupations: band {band} at \
                         k = ({:.4}, {:.4}, {:.4}) has Fermi occupation {f:.6} (deviation \
                         {:.3e} > {occupation_tolerance:.3e}). Rerun at \
                         ElectronicOptions::electronic_temperature = 0.0, or on a gapped insulator",
                        frac[0],
                        frac[1],
                        frac[2],
                        (f - target).abs()
                    )));
                }
            }
        }
        gap_seen = gap_seen.min(gap);
        coeffs.push(occupied_coefficients(&eig, n, nocc));
        fractionals.push(frac);
    }

    let mut phase = 0.0_f64;
    let mut link_re = vec![0.0_f64; nocc * nocc];
    let mut link_im = vec![0.0_f64; nocc * nocc];
    for j in 0..n_string {
        // The wrap-around link k_{N-1} -> k_0 + b_d reuses C(k_0) verbatim: in the
        // cell gauge the Bloch AO basis is exactly periodic in k, so C(k+G) = C(k).
        let next = (j + 1) % n_string;
        let a = boost_at_k(blocks, n, fractionals[j]);
        let (kre, kim) = &coeffs[j];
        let (bre, bim) = &coeffs[next];
        // W = A C(k_j)  (n x nocc)
        let mut w_re = Matrix::zeros(n, nocc);
        let mut w_im = Matrix::zeros(n, nocc);
        for mu in 0..n {
            for nu in 0..n {
                let ar = a.re[(mu, nu)];
                let ai = a.im[(mu, nu)];
                if ar == 0.0 && ai == 0.0 {
                    continue;
                }
                for b in 0..nocc {
                    let cr = kre[(nu, b)];
                    let ci = kim[(nu, b)];
                    w_re[(mu, b)] += ar * cr - ai * ci;
                    w_im[(mu, b)] += ar * ci + ai * cr;
                }
            }
        }
        // M = C(k_{j+1})^H W  (nocc x nocc)
        for m in 0..nocc {
            for b in 0..nocc {
                let mut sr = 0.0_f64;
                let mut si = 0.0_f64;
                for mu in 0..n {
                    let cr = bre[(mu, m)];
                    let ci = bim[(mu, m)];
                    sr += cr * w_re[(mu, b)] + ci * w_im[(mu, b)];
                    si += cr * w_im[(mu, b)] - ci * w_re[(mu, b)];
                }
                link_re[m * nocc + b] = sr;
                link_im[m * nocc + b] = si;
            }
        }
        let (log_abs, arg) = complex_det_log(&mut link_re, &mut link_im, nocc);
        if !log_abs.is_finite() {
            return Err(Gfn1Error::InvalidInput(
                "pbc_berry_polarization: a Berry link determinant vanished (the occupied \
                 manifolds at neighbouring k-points are orthogonal); refine the Berry mesh"
                    .to_string(),
            ));
        }
        phase += arg;
    }
    Ok((phase, gap_seen))
}

/// `(ln|det|, arg det)` of a row-major complex `m x m` matrix by LU with partial
/// pivoting. The inputs are consumed (overwritten with the factorisation).
fn complex_det_log(re: &mut [f64], im: &mut [f64], m: usize) -> (f64, f64) {
    let mut log_abs = 0.0_f64;
    let mut phase = 0.0_f64;
    for col in 0..m {
        let mut piv = col;
        let mut best = -1.0_f64;
        for row in col..m {
            let v = re[row * m + col] * re[row * m + col] + im[row * m + col] * im[row * m + col];
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best <= 0.0 {
            return (f64::NEG_INFINITY, 0.0);
        }
        if piv != col {
            for c in 0..m {
                re.swap(piv * m + c, col * m + c);
                im.swap(piv * m + c, col * m + c);
            }
            phase += PI;
        }
        let ar = re[col * m + col];
        let ai = im[col * m + col];
        let den = ar * ar + ai * ai;
        log_abs += 0.5 * den.ln();
        phase += ai.atan2(ar);
        for row in (col + 1)..m {
            let br = re[row * m + col];
            let bi = im[row * m + col];
            let fr = (br * ar + bi * ai) / den;
            let fi = (bi * ar - br * ai) / den;
            re[row * m + col] = 0.0;
            im[row * m + col] = 0.0;
            for c in (col + 1)..m {
                let ur = re[col * m + c];
                let ui = im[col * m + c];
                re[row * m + c] -= fr * ur - fi * ui;
                im[row * m + c] -= fr * ui + fi * ur;
            }
        }
    }
    (log_abs, phase)
}

/// Reduce `x` into `(-period/2, period/2]`.
fn reduce_branch(x: f64, period: f64) -> f64 {
    let mut r = x - period * (x / period + 0.5).floor();
    if r <= -0.5 * period {
        r += period;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarization_si_conversion_is_the_expected_magnitude() {
        // e / a0^2 ~ 57.21 C/m^2 — the scale that makes a ferroelectric's tenths
        // of a C/m^2 land at ~1e-3 atomic units.
        assert!(
            (POLARIZATION_AU_TO_C_PER_M2 - 57.2147).abs() < 1.0e-3,
            "{POLARIZATION_AU_TO_C_PER_M2}"
        );
    }

    #[test]
    fn branch_reduction_lands_in_the_half_open_interval() {
        let p = 4.0 * PI;
        for k in -4..=4 {
            for &x in &[0.0_f64, 1.0, -1.0, 6.0, -6.0] {
                let r = reduce_branch(x + p * k as f64, p);
                assert!(r > -0.5 * p - 1.0e-9 && r <= 0.5 * p + 1.0e-9, "{r}");
                // Reduction only moves by whole periods.
                let shift = (x - r) / p;
                assert!((shift - shift.round()).abs() < 1.0e-9, "{shift}");
            }
        }
    }

    #[test]
    fn complex_determinant_matches_a_hand_value() {
        // [[1+i, 2],[3, 4-i]] has det = (1+i)(4-i) - 6 = (4 - i + 4i + 1) - 6 = -1 + 3i.
        let mut re = vec![1.0, 2.0, 3.0, 4.0];
        let mut im = vec![1.0, 0.0, 0.0, -1.0];
        let (log_abs, arg) = complex_det_log(&mut re, &mut im, 2);
        let mag = log_abs.exp();
        let expected = (-1.0_f64, 3.0_f64);
        let exp_mag = (expected.0 * expected.0 + expected.1 * expected.1).sqrt();
        assert!((mag - exp_mag).abs() < 1.0e-12, "|det| {mag}");
        let exp_arg = expected.1.atan2(expected.0);
        let d = reduce_branch(arg - exp_arg, 2.0 * PI);
        assert!(d.abs() < 1.0e-12, "arg {arg} vs {exp_arg}");
    }

    #[test]
    fn string_bases_tile_the_perpendicular_mesh() {
        let mesh = [1, 2, 3];
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for s in 0..(mesh[1] * mesh[2]) {
            let b = string_base(0, s, mesh);
            let i = (b[1] * mesh[1] as f64).round() as usize;
            let j = (b[2] * mesh[2] as f64).round() as usize;
            assert!(!seen.contains(&(i, j)), "duplicate string {i} {j}");
            seen.push((i, j));
        }
        assert_eq!(seen.len(), mesh[1] * mesh[2]);
    }
}
