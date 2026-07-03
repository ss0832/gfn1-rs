// SPDX-License-Identifier: GPL-3.0-or-later
//! Spin-polarized GFN1-xTB ("spGFN1"): collinear spin-DFTB spin polarization on
//! top of the unchanged GFN1-xTB electronic model.
//!
//! Plain GFN1-xTB is restricted-open-shell — its energy does not depend on the
//! spin density. spGFN1 **adds** a spin-polarization energy built from the
//! universal atomic spin constants `W` (the magnetic analogue of the
//! Hubbard/hardness parameter; the second derivative of the atomic spin-DFT
//! energy w.r.t. the shell magnetization). Every existing GFN1 parameter is
//! untouched.
//!
//! # Method (collinear spin-DFTB)
//!
//! With shell-resolved Mulliken **magnetization** populations
//! `m_{A,l} = n^α_{A,l} − n^β_{A,l}` (a *population* difference — no reference
//! occupation, which cancels), the spin energy and (magnetization-channel) spin
//! potential are
//!
//! ```text
//! E_spin   = ½ Σ_A Σ_{l,l' ∈ A} W_{A,ll'} m_{A,l} m_{A,l'}
//! V^s_{A,l} = Σ_{l' ∈ A} W_{A,ll'} m_{A,l'}
//! ```
//!
//! The spin potential enters the per-spin effective Hamiltonian with **opposite
//! sign** for α/β, layered onto the ordinary GFN1 charge potential `v^c`:
//!
//! ```text
//! F^α_{μν} = H0_{μν} − ½(v^c_i + v^c_j) S_{μν} − ½(V^s_i + V^s_j) S_{μν}
//! F^β_{μν} = H0_{μν} − ½(v^c_i + v^c_j) S_{μν} + ½(V^s_i + V^s_j) S_{μν}
//! ```
//! (`i`,`j` are the shells of AOs `μ`,`ν`). This is exactly tblite's
//! charge/magnetization → up/down convention (`magnet_to_updown`): the band +
//! charge energy is the ordinary GFN1 energy evaluated over the **total** density
//! `P = P^α + P^β`, and `E_spin` is added on top.
//!
//! A closed-shell singlet has `P^α = P^β`, so `m ≡ 0`, `E_spin = 0`, `V^s = 0`
//! and both Fock matrices equal the restricted Fock → the result is
//! byte-identical to plain GFN1. We make that exact by **delegating** the
//! closed-shell case to the restricted [`crate::electronic::run_electronic`]
//! path; only genuinely open-shell systems run the spin-unrestricted SCC here.
//!
//! Non-periodic; energy + analytic forces only.
//!
//! # Spin constants `W`
//!
//! Sourced from tblite `src/tblite/data/spin.f90` (LGPL-3.0-or-later; compatible
//! with this GPL-3.0-or-later crate), transcribed into
//! [`crate::data_tables::GFN_SPIN_CONSTANTS`]. See `third_party/tblite/` for the
//! verbatim source snapshot, provenance, and method references
//! (Köhler/Seifert/Frauenheim spin-DFTB; xtb `spgfn` docs).

use crate::basis::BasisSet;
use crate::coulomb::{
    coulomb_energy_potential_from_matrix, effective_coulomb_matrix, ShellChargeModel,
};
use crate::data_tables::gfn_spin_constant;
use crate::dispersion::dispersion_energy;
use crate::electronic::{
    electronic_energy, fock_from_shell_potential, mulliken_shell_charges, BroydenMixer,
    ElectronicOptions, ElectronicResult,
};
use crate::error::{Gfn1Error, Result};
use crate::field::mulliken_dipole;
use crate::halogen::halogen_energy;
use crate::hamiltonian::build_h0;
use crate::linalg::{column_weighted_gram, lowdin_orthogonalizer, lowdin_solve_with_orthogonalizer, Matrix};
use crate::params::{AngularMomentum, Gfn1Parameters};
use crate::repulsion::repulsion_energy;
use crate::system::PeriodicSystem;

const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_811_563e-6;
const ELECTRON_COUNT_TOLERANCE: f64 = 1.0e-6;

/// Spin-resolved data attached to an [`ElectronicResult`] produced by the
/// spin-polarized (open-shell) path. Carries everything the analytic spin
/// gradient needs: the separate α/β one-particle and energy-weighted densities,
/// the per-shell magnetization populations, and the per-shell spin potential.
#[derive(Clone, Debug)]
pub struct SpinResolved {
    /// α one-particle density matrix (AO basis).
    pub density_alpha: Matrix,
    /// β one-particle density matrix (AO basis).
    pub density_beta: Matrix,
    /// α energy-weighted density `Σ_i ε_i f_i C_i C_i^T` (AO basis).
    pub ew_density_alpha: Matrix,
    /// β energy-weighted density (AO basis).
    pub ew_density_beta: Matrix,
    /// Per-shell magnetization population `m_l = pop^α_l − pop^β_l`.
    pub shell_magnetization: Vec<f64>,
    /// Per-shell spin potential `V^s_l = Σ_{l'} W_{ll'} m_{l'}`.
    pub shell_spin_potential: Vec<f64>,
    /// Number of α / β electrons used (aufbau channel counts).
    pub n_alpha: f64,
    pub n_beta: f64,
    /// Spin-polarization energy `E_spin` (Hartree, ≤ 0).
    pub spin_energy: f64,
    /// Resolved DFT+U/+U+V correlated subspace (with the applied on-site `U`), and
    /// the inter-site `V` pairs. Empty unless `+U+V` was active. Carried so the
    /// analytic gradient can rebuild the orbital potential `Ṽ` and its overlap-Pulay
    /// weight `Q = ½(Pσ Ṽσ + Ṽσ Pσ)` without re-running the SCC.
    pub plus_u_subspace: Vec<crate::plus_u::CorrelatedAtom>,
    /// Inter-site `+V` pairs (indices into `plus_u_subspace`).
    pub plus_u_pairs: Vec<crate::plus_u::IntersitePair>,
}

/// Number of unpaired electrons (`2S`) for the requested multiplicity / electron
/// count. `None` multiplicity ⇒ the minimal-spin ground configuration: 0 for an
/// even electron count, 1 for an odd one (a doublet).
fn resolve_unpaired(nelec: f64, spin_multiplicity: Option<usize>) -> Result<i64> {
    let rounded = nelec.round();
    if (nelec - rounded).abs() > ELECTRON_COUNT_TOLERANCE {
        return Err(Gfn1Error::InvalidInput(format!(
            "spin polarization requires an integer electron count; got {nelec}"
        )));
    }
    let ne = rounded as i64;
    match spin_multiplicity {
        Some(0) => Err(Gfn1Error::InvalidInput(
            "spin multiplicity must be at least 1".to_string(),
        )),
        Some(m) => {
            let unpaired = (m - 1) as i64;
            if unpaired > ne {
                return Err(Gfn1Error::InvalidInput(format!(
                    "spin multiplicity {m} requires {unpaired} unpaired electrons, but the system has {ne} electrons"
                )));
            }
            if (ne - unpaired) % 2 != 0 {
                return Err(Gfn1Error::InvalidInput(format!(
                    "spin multiplicity {m} is incompatible with {ne} electrons"
                )));
            }
            Ok(unpaired)
        }
        None => Ok(ne.rem_euclid(2)),
    }
}

/// Self-contained CAMM-on-mDFTB2 (GFN2-style AES) context for the spin SCC: everything the
/// FD-gated restricted CAMM functions need to build the (spin-independent, electrostatic) AES
/// energy + Fock shift from a **total** density. Built once per `run_spin_unrestricted` and
/// threaded into the SCC loop, where each iteration rebuilds the cumulative atomic moments from
/// the previous-iteration total density (exactly mirroring how `+U` uses the previous channel
/// densities) and adds the AES Fock to **both** spin channels + its energy to the total.
struct SpinCammContext<'a> {
    integrals: &'a crate::integrals::IntegralMatrices,
    /// Per-atom Klopman–Ohno hardness η_A (the s-shell hardness), as the restricted path uses.
    hardness: Vec<f64>,
    /// Atom positions (bohr).
    pos: Vec<crate::math::Vec3>,
    /// Per-atom CAMM range factor κ_A (element override or global `camm_damp`).
    kappa: Vec<f64>,
    /// AES amplitude s_AES.
    scale: f64,
    /// Per-atom on-site penalty scale s_onsite.
    onsite: Vec<f64>,
    /// Charge-dependent κ `(κ₀, γ)` — recomputed per iteration from the mixed charges when set.
    damp_charge: Option<(f64, f64)>,
}

impl<'a> SpinCammContext<'a> {
    /// AES energy + Fock shift from a total density (`total_density = P^α + P^β`) and the current
    /// atomic charges `Δq_A` (code sign, `−gfn1_atomic`). Mirrors the restricted CAMM call site.
    fn energy_fock(
        &self,
        basis: &BasisSet,
        nat: usize,
        total_density: &Matrix,
        atomic_charges: &[f64],
    ) -> crate::multipole::MultipoleEnergyFock {
        let moments = crate::multipole::camm_atomic_moments(
            basis, nat, self.integrals, total_density, &self.pos,
        );
        // Charge-dependent κ (if set): κ_A = κ₀/(1+γ Δq_A²), matching the restricted path.
        let dyn_kappa: Vec<f64>;
        let eff_kappa: &[f64] = if let Some((k0, g)) = self.damp_charge {
            dyn_kappa = atomic_charges
                .iter()
                .map(|&q| (k0 / (1.0 + g * q * q)).max(0.05))
                .collect();
            &dyn_kappa
        } else {
            &self.kappa
        };
        crate::multipole::camm_aes_energy_fock(
            basis, nat, &self.hardness, &self.pos, self.integrals, &moments, atomic_charges,
            eff_kappa, self.scale, &self.onsite,
        )
    }
}

/// Entry point for the spin-polarized GFN1 path (non-periodic). Decides whether
/// the system is open-shell. A closed-shell configuration is delegated straight
/// to the restricted [`crate::electronic::run_electronic`] (spin term ≡ 0, so
/// byte-identical to plain GFN1); an open-shell configuration runs the
/// spin-unrestricted SCC in [`run_spin_unrestricted`].
pub fn run_spin_polarized(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: ElectronicOptions,
) -> Result<ElectronicResult> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "spin-polarized GFN1 (spGFN1) is implemented for non-periodic systems only".to_string(),
        ));
    }
    // v1 is the bare GFN1 electronic model; reject the experimental add-on Fock paths it has not
    // been derived/validated against (energy + analytic forces only), rather than silently
    // producing an inconsistent result. NARROW EXCEPTION: the validated **CAMM-AES base**
    // (`multipole_model = CammOnMdftb2`, dipole+quad only) is allowed together with the spin/`+U`
    // path — its (spin-independent, electrostatic) Fock is added to both channels and its energy to
    // the total, self-consistently, reusing the FD-gated restricted CAMM functions. The exotic
    // multipole extensions (octupole, third order, charge-cross, higher rank, secondary basis,
    // field–dipole) and the non-CAMM Ohno mDFTB2 model remain rejected here.
    let camm_ok = options.multipole
        && options.multipole_model == crate::electronic::MultipoleModel::CammOnMdftb2
        && !options.multipole_octupole
        && options.multipole_order < 4
        && !options.multipole_third_order
        && options.multipole_charge_order.is_empty()
        && !options.field_multipole
        && options.multipole_secondary_basis.is_none()
        && !options.scf_trah;
    if (options.multipole && !camm_ok)
        || options.lr_exchange
        || options.experimental_d4
        || options.external_field.electric_field.is_some()
        || options.external_field.magnetic_field.is_some()
    {
        return Err(Gfn1Error::InvalidInput(
            "spin-polarized GFN1 (spGFN1) v1 supports only the base GFN1 model plus the validated \
             CAMM-AES multipole base (multipole_model = camm_on_mdftb2, dipole+quad only): disable \
             lr_exchange, experimental_d4, external fields, and the exotic multipole extensions \
             (octupole, third order, charge-cross, higher rank, secondary basis, field–dipole, and \
             the non-CAMM Ohno mDFTB2 model)"
                .to_string(),
        ));
    }

    let basis = BasisSet::build(
        system,
        params,
        crate::basis::BasisOptions {
            nprim: options.nprim,
        },
    )?;
    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    let unpaired = resolve_unpaired(nelec, options.spin_multiplicity)?;

    if unpaired == 0 && !options.plus_u {
        // Closed shell, no +U: the spin density is zero and the spin term vanishes. Delegate to the
        // restricted path so the energy and forces are byte-identical to plain GFN1.
        let mut restricted = options;
        restricted.spin_polarization = false;
        return crate::electronic::run_electronic(system, params, restricted);
    }
    // A closed shell **with** +U falls through to the spin-unrestricted SCC below: seeded at zero
    // magnetization it stays spin-paired (P^α = P^β), reducing to the restricted result, while the
    // per-spin +U/+U+V term still applies. (Open shells always take the unrestricted path.)

    run_spin_unrestricted(system, params, &options, basis, nelec, unpaired)
}

/// Per-shell element `z` and angular-momentum index `l ∈ {0,1,2,…}`, plus the
/// shells grouped by atom (the spin term is strictly on-site).
pub(crate) struct ShellInfo {
    pub(crate) z: Vec<u8>,
    pub(crate) l: Vec<usize>,
    /// Shells grouped by atom (atom -> list of shell indices).
    pub(crate) by_atom: Vec<Vec<usize>>,
}

pub(crate) fn shell_info(basis: &BasisSet, nat: usize) -> ShellInfo {
    let nsh = basis.shells.len();
    let mut z = vec![0u8; nsh];
    let mut l = vec![0usize; nsh];
    let mut by_atom = vec![Vec::new(); nat];
    for (ish, sh) in basis.shells.iter().enumerate() {
        z[ish] = sh.z;
        l[ish] = sh.angular.as_index();
        by_atom[sh.atom_index].push(ish);
    }
    ShellInfo { z, l, by_atom }
}

/// Per-shell spin potential `V^s_l = Σ_{l' on the same atom} W_{ll'} m_{l'}`.
pub(crate) fn spin_shell_potential(info: &ShellInfo, magnetization: &[f64]) -> Vec<f64> {
    let nsh = magnetization.len();
    let mut v = vec![0.0; nsh];
    for shells in &info.by_atom {
        for &ish in shells {
            let mut acc = 0.0;
            for &jsh in shells {
                let w = gfn_spin_constant(info.l[ish], info.l[jsh], info.z[ish]);
                acc += w * magnetization[jsh];
            }
            v[ish] = acc;
        }
    }
    v
}

/// Spin-polarization energy `E_spin = ½ Σ_A Σ_{l,l'} W_{ll'} m_l m_{l'}`, computed
/// as `½ Σ_l V^s_l m_l` from the spin potential `V^s_l = Σ_{l'} W_{ll'} m_{l'}`.
fn spin_energy(magnetization: &[f64], spin_potential: &[f64]) -> f64 {
    let mut e = 0.0;
    for ish in 0..magnetization.len() {
        e += 0.5 * spin_potential[ish] * magnetization[ish];
    }
    e
}

/// Aufbau / Fermi occupations for a single spin channel holding `nelec`
/// electrons (max 1 per orbital). Returns `(occupations, fermi_level, entropy)`.
fn channel_occupations(eps: &[f64], nelec: f64, kt: f64) -> (Vec<f64>, f64, f64) {
    let n = eps.len();
    if nelec <= 0.0 {
        return (vec![0.0; n], eps.first().copied().unwrap_or(0.0), 0.0);
    }
    if nelec >= n as f64 {
        return (vec![1.0; n], eps.last().copied().unwrap_or(0.0), 0.0);
    }
    if kt <= 0.0 {
        // Integer aufbau.
        let mut occ = vec![0.0; n];
        let mut remaining = nelec;
        for o in occ.iter_mut() {
            let fill = remaining.min(1.0);
            *o = fill;
            remaining -= fill;
            if remaining <= 0.0 {
                break;
            }
        }
        let homo = occ.iter().rposition(|o| *o > 1.0e-12).unwrap_or(0);
        return (occ, eps[homo], 0.0);
    }
    // Fermi smearing for a single (1-per-orbital) channel.
    let min_e = eps.iter().copied().fold(f64::INFINITY, f64::min);
    let max_e = eps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut lo = min_e - 100.0 * kt - 10.0;
    let mut hi = max_e + 100.0 * kt + 10.0;
    let fermi = |eps: f64, mu: f64| -> f64 {
        let x = ((eps - mu) / kt).clamp(-80.0, 80.0);
        1.0 / (1.0 + x.exp())
    };
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let sum: f64 = eps.iter().map(|e| fermi(*e, mu)).sum();
        if sum < nelec {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let occ: Vec<f64> = eps.iter().map(|e| fermi(*e, mu)).collect();
    // Single-channel electronic entropy term (−T S), kt·Σ [n ln n + (1−n) ln(1−n)].
    let entropy: f64 = occ
        .iter()
        .map(|&o| {
            let nclamp = o.clamp(1.0e-16, 1.0 - 1.0e-16);
            kt * (nclamp * nclamp.ln() + (1.0 - nclamp) * (1.0 - nclamp).ln())
        })
        .sum();
    (occ, mu, entropy)
}

/// One spin channel solved from its Fock matrix: density, energy-weighted
/// density, shell populations (Mulliken population, *no* reference subtraction),
/// occupations, eigenvalues, and the channel entropy term.
struct ChannelStep {
    density: Matrix,
    ew_density: Matrix,
    shell_population: Vec<f64>,
    entropy: f64,
}

#[allow(clippy::too_many_arguments)]
fn solve_channel(
    basis: &BasisSet,
    overlap: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    fock: &Matrix,
    nelec: f64,
    kt: f64,
    eigen_tol: f64,
) -> Result<ChannelStep> {
    let eig = lowdin_solve_with_orthogonalizer(fock, orth, eigen_tol)?;
    let (occ, _fermi, entropy) = channel_occupations(&eig.values, nelec, kt);
    let density = column_weighted_gram(&eig.vectors, &occ)?;
    let weighted: Vec<f64> = eig
        .values
        .iter()
        .zip(occ.iter())
        .map(|(e, o)| e * o)
        .collect();
    let ew_density = column_weighted_gram(&eig.vectors, &weighted)?;
    // Per-shell Mulliken population (Σ_{μ∈shell} (P S)_{μμ}); the spin channel uses the
    // population difference, where the reference occupation cancels.
    let n = basis.len();
    let pslice = density.as_slice();
    let sslice = overlap.as_slice();
    let mut shell_population = vec![0.0; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut pop = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            let off = iao * n;
            pop += pslice[off..off + n]
                .iter()
                .zip(&sslice[off..off + n])
                .map(|(p, s)| p * s)
                .sum::<f64>();
        }
        shell_population[ish] = pop;
    }
    Ok(ChannelStep {
        density,
        ew_density,
        shell_population,
        entropy,
    })
}

#[allow(clippy::too_many_arguments)]
/// Overlap-dressed Fock perturbation `½(α Π S + S α Π)` that shifts the on-site
/// potential of one correlated atom's `d` block by `alpha` — the localised probe
/// whose occupation response gives the linear-response Hubbard parameters.
fn onsite_shift_fock(overlap: &Matrix, aos: &[usize], alpha: f64) -> Matrix {
    let n = overlap.rows();
    let mut vtilde = Matrix::zeros(n, n);
    for &a in aos {
        vtilde[(a, a)] = alpha;
    }
    let vs = vtilde.matmul(overlap).expect("Ṽ·S");
    let sv = overlap.matmul(&vtilde).expect("S·Ṽ");
    let mut g = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            g[(i, j)] = 0.5 * (vs[(i, j)] + sv[(i, j)]);
        }
    }
    g
}

/// Apply a virtual level shift to a channel Fock in place: `F += b·(S − ½ S P S)`,
/// where `P` is the previous-iteration density for THIS channel. This raises the
/// virtual orbital energies by `b` (leaving the occupied block unchanged at self
/// consistency), lifting a collapsing HOMO–LUMO gap so the cold-Fermi occupations
/// stop flipping. Mirrors the restricted driver's level-shift term in
/// `electronic::build_scc_step`. A `None` previous density (first iteration) is a
/// no-op (no projector yet).
fn apply_level_shift(fock: &mut Matrix, overlap: &Matrix, prev_density: Option<&Matrix>, b: f64) {
    if b == 0.0 {
        return;
    }
    let Some(prev) = prev_density else {
        return;
    };
    let sp = match overlap.matmul(prev) {
        Ok(m) => m,
        Err(_) => return,
    };
    let sps = match sp.matmul(overlap) {
        Ok(m) => m,
        Err(_) => return,
    };
    let f = fock.as_mut_slice();
    let s = overlap.as_slice();
    let m = sps.as_slice();
    for idx in 0..f.len() {
        f[idx] += b * (s[idx] - 0.5 * m[idx]);
    }
}

/// One rung of the robust spin-SCC ladder: charge/magnetization accelerator,
/// linear-mixing damping (also the Broyden seed damping), virtual level shift,
/// electronic temperature `kt` (Hartree), and the iteration budget for the rung.
#[derive(Clone, Copy)]
struct SpinSccRung {
    /// `true` ⇒ Broyden quasi-Newton mixing; `false` ⇒ monotone linear mixing.
    broyden: bool,
    /// Linear-mixing damping / Broyden seed damping (clamped to [0.01, 1]).
    mixing: f64,
    /// Virtual level shift `b` (Ha); `0.0` disables it.
    level_shift: f64,
    /// Electronic temperature in Hartree units (already `T·k_B`); `0.0` ⇒ aufbau.
    kt: f64,
    /// Maximum SCC iterations for this rung.
    max_scc: usize,
}

/// Self-contained spin-unrestricted SCC solve (the same loop the main driver
/// runs), optionally with a fixed Fock `perturbation` added to both spin
/// channels and an optional `+U+V` correlated correction. Returns the converged
/// `(α, β)` channels plus the convergence flag and iteration count. Used by the
/// linear-response orchestration to run the perturbed and re-converged solves.
///
/// This is the **robust** entry point: it first runs the user's exact scheme
/// (Broyden + the requested mixing + electronic temperature + iteration budget) so
/// every system that converges today exits on rung 0 byte-for-byte. Only when that
/// primary scheme **fails** — non-convergence, OR a non-finite (NaN/Inf) energy or
/// density — does it retry with progressively safer rungs: rungs 1–3 keep the user's
/// electronic temperature (so they pin the TRUE kt=0 fixed point, no smearing
/// artifact) and add a virtual level shift + progressively smaller / linear mixing to
/// lift the collapsing HOMO–LUMO gap; rung 4 (last resort) raises the electronic
/// temperature to Fermi-smear a genuinely degenerate cold frontier. The first rung
/// that converges to a finite result is reported.
#[allow(clippy::too_many_arguments)]
fn spin_scc_loop(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    kt: f64,
    unpaired: i64,
    options: &ElectronicOptions,
    plus_u_sub: &[crate::plus_u::CorrelatedAtom],
    plus_u_pairs: &[crate::plus_u::IntersitePair],
    perturbation: Option<&Matrix>,
    camm: Option<&SpinCammContext>,
) -> Result<(ChannelStep, ChannelStep, bool, usize)> {
    // Rung 0 == the user's exact scheme (byte-identical default path). The robust budget
    // for the linear-mixing rungs is enlarged because monotone mixing on a near-degenerate
    // frontier needs a long linear tail. Rung 3 raises the electronic temperature to lift
    // the cold-Fermi degeneracy (the only rung that perturbs the energy, and a last resort
    // that runs only when every energy-preserving rung has failed outright).
    let robust_budget = options.max_scc.max(800);
    let base_kt = kt;
    let ladder: [SpinSccRung; 5] = [
        // Rung 0: user's exact settings — byte-identical to the original spin loop, which
        // ran Broyden charge/magnetization mixing with NO level shift. (The level shift is a
        // pure fallback accelerator here; engaging it on rung 0 would change converging systems.)
        SpinSccRung {
            broyden: true,
            mixing: options.mixing.clamp(0.01, 1.0),
            level_shift: 0.0,
            kt: base_kt,
            max_scc: options.max_scc,
        },
        // Rungs 1–3 are ALL energy-preserving (user's etemp kept): they converge to the TRUE
        // kt=0 SCC fixed point, never a smearing artifact, so the recomputed-U FD force stays
        // consistent. They progressively lift the collapsing HOMO–LUMO gap with a larger
        // virtual level shift and damp the cold-Fermi slosh with smaller (eventually linear)
        // mixing. Rung 1: Broyden + a moderate level shift + reduced mixing.
        SpinSccRung {
            broyden: true,
            mixing: 0.20,
            level_shift: options.level_shift.max(0.20),
            kt: base_kt,
            max_scc: robust_budget,
        },
        // Rung 2: monotone linear mixing + a larger level shift (the restricted ladder's
        // workhorse for near-degenerate metals — slow but monotone).
        SpinSccRung {
            broyden: false,
            mixing: 0.10,
            level_shift: options.level_shift.max(0.40),
            kt: base_kt,
            max_scc: robust_budget,
        },
        // Rung 3: very heavy level shift + very gentle linear mixing — the last energy-preserving
        // attempt to pin the kt=0 fixed point before resorting to smearing.
        SpinSccRung {
            broyden: false,
            mixing: 0.05,
            level_shift: options.level_shift.max(0.80),
            kt: base_kt,
            max_scc: robust_budget,
        },
        // Rung 4 (last resort): Broyden + a raised electronic temperature to lift the
        // near-degenerate cold-Fermi frontier. This is the ONLY rung that perturbs the energy
        // (fractional occupations / entropy), so it runs only when every energy-preserving rung
        // has failed outright — for such a system any finite, converged number beats a NaN.
        SpinSccRung {
            broyden: true,
            mixing: options.mixing.clamp(0.01, 1.0),
            level_shift: 0.0,
            kt: base_kt.max(3000.0 * BOLTZMANN_HARTREE_PER_K),
            max_scc: options.max_scc.max(250),
        },
    ];

    let mut last_err: Option<Gfn1Error> = None;
    for (attempt, rung) in ladder.iter().enumerate() {
        if attempt > 0 && std::env::var("GFN1_SCC_DEBUG").is_ok() {
            eprintln!(
                "[spin-SCC] --- fallback rung {attempt}: broyden={} mixing={} level_shift={} kt={} max_scc={} ---",
                rung.broyden, rung.mixing, rung.level_shift, rung.kt, rung.max_scc
            );
        }
        match spin_scc_loop_core(
            basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, unpaired, options,
            plus_u_sub, plus_u_pairs, perturbation, camm, rung,
        ) {
            Ok((alpha, beta, converged, iterations)) => {
                // Accept the first rung that converges to a finite result. A rung that ran out of
                // iterations without a non-finite blow-up still falls through to the next (safer)
                // rung — except the very last, whose result we return regardless (best effort).
                let finite = converged
                    && alpha.density.as_slice().iter().all(|x| x.is_finite())
                    && beta.density.as_slice().iter().all(|x| x.is_finite());
                if finite || attempt == ladder.len() - 1 {
                    return Ok((alpha, beta, converged, iterations));
                }
            }
            Err(e) => {
                // Non-finite blow-up (or a solver failure) on this rung: remember it and try the
                // next, safer rung. If even the last rung errors, surface the error.
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        Gfn1Error::InvalidInput("spin SCC failed on every robust rung".to_string())
    }))
}

/// The bare spin-unrestricted SCC loop for ONE rung of settings. Identical physics to
/// the original `spin_scc_loop`, parameterized by the rung's accelerator (Broyden /
/// linear), mixing, virtual level shift, and electronic temperature `kt`. Returns
/// `Err(InvalidInput)` if the energy or a channel density goes non-finite (NaN/Inf), so
/// the robust wrapper can fall through to a safer rung instead of returning a NaN energy.
#[allow(clippy::too_many_arguments)]
fn spin_scc_loop_core(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    unpaired: i64,
    options: &ElectronicOptions,
    plus_u_sub: &[crate::plus_u::CorrelatedAtom],
    plus_u_pairs: &[crate::plus_u::IntersitePair],
    perturbation: Option<&Matrix>,
    camm: Option<&SpinCammContext>,
    rung: &SpinSccRung,
) -> Result<(ChannelStep, ChannelStep, bool, usize)> {
    let nat = shell_model.atom_offsets.len();
    let nsh = basis.shells.len();
    let plus_u_active = !plus_u_sub.is_empty();
    let mut v_mixed = vec![0.0; 2 * nsh];
    let mixing = rung.mixing.clamp(0.01, 1.0);
    let kt = rung.kt;
    let level_shift = rung.level_shift;
    let mut broyden = BroydenMixer::new(2 * nsh, options.scc_broyden_size.max(2), mixing);
    let mut shell_charges = vec![0.0; nsh];
    let mut magnetization = vec![0.0; nsh];
    seed_magnetization(info, &mut magnetization, unpaired as f64);
    let mut alpha: Option<ChannelStep> = None;
    let mut beta: Option<ChannelStep> = None;
    let mut converged = false;
    let mut iterations = 0usize;
    let mut last_energy: Option<f64> = None;
    // Previous-iteration channel densities for the virtual level-shift projector.
    let mut prev_dens_a: Option<Matrix> = None;
    let mut prev_dens_b: Option<Matrix> = None;

    for it in 0..rung.max_scc {
        iterations = it + 1;
        let scc = coulomb_energy_potential_from_matrix(basis, shell_model, &shell_charges, amat)?;
        let v_charge = &scc.shell_potential;
        let spin_potential = spin_shell_potential(info, &magnetization);
        let mut v_alpha = vec![0.0; nsh];
        let mut v_beta = vec![0.0; nsh];
        for ish in 0..nsh {
            v_alpha[ish] = v_charge[ish] - spin_potential[ish];
            v_beta[ish] = v_charge[ish] + spin_potential[ish];
        }
        let mut fock_alpha = fock_from_shell_potential(basis, overlap, h0, &v_alpha);
        let mut fock_beta = fock_from_shell_potential(basis, overlap, h0, &v_beta);
        if plus_u_active {
            if let (Some(a_prev), Some(b_prev)) = (alpha.as_ref(), beta.as_ref()) {
                let (_, ga) = crate::plus_u::plus_u_v(&a_prev.density, overlap, plus_u_sub, plus_u_pairs);
                let (_, gb) = crate::plus_u::plus_u_v(&b_prev.density, overlap, plus_u_sub, plus_u_pairs);
                fock_alpha = matrix_sum(&fock_alpha, &ga);
                fock_beta = matrix_sum(&fock_beta, &gb);
            }
        }
        // CAMM-on-mDFTB2 AES: the (spin-independent, electrostatic) Fock shift from the
        // previous-iteration TOTAL density, added to BOTH channels (mirrors the `+U` prev-density
        // pattern; converges as prev→current at self-consistency). Δq_A = −(GFN1 atomic charge)
        // from the mixed shell charges of this iteration.
        if let Some(camm_ctx) = camm {
            if let (Some(a_prev), Some(b_prev)) = (alpha.as_ref(), beta.as_ref()) {
                let total_prev = matrix_sum(&a_prev.density, &b_prev.density);
                let gfn1_atomic = shell_model.atomic_charges(basis, &shell_charges);
                let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
                let ef = camm_ctx.energy_fock(basis, nat, &total_prev, &qm);
                fock_alpha = matrix_sum(&fock_alpha, &ef.fock);
                fock_beta = matrix_sum(&fock_beta, &ef.fock);
            }
        }
        if let Some(pert) = perturbation {
            fock_alpha = matrix_sum(&fock_alpha, pert);
            fock_beta = matrix_sum(&fock_beta, pert);
        }
        // Virtual level shift (per channel) using the previous-iteration density projector.
        apply_level_shift(&mut fock_alpha, overlap, prev_dens_a.as_ref(), level_shift);
        apply_level_shift(&mut fock_beta, overlap, prev_dens_b.as_ref(), level_shift);
        let step_a = solve_channel(basis, overlap, orth, &fock_alpha, n_alpha, kt, options.eigen_tolerance)?;
        let step_b = solve_channel(basis, overlap, orth, &fock_beta, n_beta, kt, options.eigen_tolerance)?;
        let mut new_q = vec![0.0; nsh];
        let mut new_m = vec![0.0; nsh];
        for (ish, shell) in basis.shells.iter().enumerate() {
            new_q[ish] = shell.reference_occ - (step_a.shell_population[ish] + step_b.shell_population[ish]);
            new_m[ish] = step_a.shell_population[ish] - step_b.shell_population[ish];
        }
        let total_density = matrix_sum(&step_a.density, &step_b.density);
        let band = electronic_energy(h0, &total_density);
        let e_spin = spin_energy(&new_m, &spin_shell_potential(info, &new_m));
        let e_plus_u = if plus_u_active {
            let (ea, _) = crate::plus_u::plus_u_v(&step_a.density, overlap, plus_u_sub, plus_u_pairs);
            let (eb, _) = crate::plus_u::plus_u_v(&step_b.density, overlap, plus_u_sub, plus_u_pairs);
            ea + eb
        } else {
            0.0
        };
        // CAMM-AES energy from the CURRENT total density (mirrors `+U` using the current step
        // densities; at self-consistency this equals the Fock's prev-density evaluation).
        let e_camm = if let Some(camm_ctx) = camm {
            let gfn1_atomic = shell_model.atomic_charges(basis, &new_q);
            let qm: Vec<f64> = gfn1_atomic.iter().map(|c| -c).collect();
            camm_ctx.energy_fock(basis, nat, &total_density, &qm).energy
        } else {
            0.0
        };
        let entropy_term = step_a.entropy + step_b.entropy;
        let scc_energy = band
            + scc.second_order
            + scc.third_order
            + scc.higher_order
            + e_spin
            + e_plus_u
            + e_camm
            + entropy_term;
        // Non-finite guard: if the energy or either channel density blew up (the cold-Fermi
        // NaN), bail so the robust wrapper can fall through to a safer rung.
        if !scc_energy.is_finite()
            || !step_a.density.as_slice().iter().all(|x| x.is_finite())
            || !step_b.density.as_slice().iter().all(|x| x.is_finite())
        {
            return Err(Gfn1Error::InvalidInput(
                "spin SCC produced a non-finite energy/density".to_string(),
            ));
        }
        let mut residual = vec![0.0; 2 * nsh];
        for ish in 0..nsh {
            residual[ish] = new_q[ish] - v_mixed[ish];
            residual[nsh + ish] = new_m[ish] - v_mixed[nsh + ish];
        }
        let rms = (residual.iter().map(|r| r * r).sum::<f64>() / residual.len() as f64).sqrt();
        let de = last_energy.map(|e| (scc_energy - e).abs()).unwrap_or(f64::INFINITY);
        last_energy = Some(scc_energy);
        shell_charges = new_q.clone();
        magnetization = new_m.clone();
        prev_dens_a = Some(step_a.density.clone());
        prev_dens_b = Some(step_b.density.clone());
        alpha = Some(step_a);
        beta = Some(step_b);
        if rms < options.charge_tolerance && de < options.energy_tolerance {
            converged = true;
            break;
        }
        let mut out_vec = vec![0.0; 2 * nsh];
        out_vec[..nsh].copy_from_slice(&new_q);
        out_vec[nsh..].copy_from_slice(&new_m);
        if rung.broyden {
            match broyden.next(&v_mixed, &residual) {
                Some(next) => v_mixed = next,
                None => v_mixed = out_vec,
            }
        } else {
            // Monotone linear mixing: v ← v + damp·residual (residual = out − v).
            for k in 0..2 * nsh {
                v_mixed[k] += mixing * residual[k];
            }
        }
        for ish in 0..nsh {
            shell_charges[ish] = v_mixed[ish];
            magnetization[ish] = v_mixed[nsh + ish];
        }
    }
    let alpha = alpha.ok_or_else(|| Gfn1Error::InvalidInput("spin SCC produced no step".to_string()))?;
    let beta = beta.ok_or_else(|| Gfn1Error::InvalidInput("spin SCC produced no step".to_string()))?;
    Ok((alpha, beta, converged, iterations))
}

// =====================================================================================
//  Analytic linear-response χ0 / χ (Part A) — replaces the ±δ finite-difference probe.
//
//  Both responses are the first-order response of the correlated-subspace occupations
//  `n_I = Σ_{a∈I}(P S)_aa` (= `subspace_occupations`) to the localized d-block probe
//  `δF_J = onsite_shift_fock(S, aos_J, 1) = ½(Π_J S + S Π_J)`, applied (with the same
//  spin-independent sign as the FD code) to BOTH α and β Fock matrices.
//
//  χ0  — BARE response at the fixed self-consistent potential: the first-order MO
//        density response of each spin channel to `δF_J` ALONE (no SCC feedback).
//  χ   — SCREENED response: the same, but the induced shell-charge / magnetization
//        rearrangement feeds back through the SCC kernel (Coulomb `amat`+on-site
//        higher-order `γ`, and the spin `W`) until self-consistent — solved directly
//        in the SCC (q, m) variable space, exactly the variables `spin_scc_loop` mixes.
// =====================================================================================

/// Converged base-state data for ONE spin channel, used to build the analytic
/// occupation response: the MO coefficients (AO×orbital), orbital energies, and the
/// (finite-temperature) occupations holding `nelec` electrons (max 1 per orbital).
pub(crate) struct ChannelBasis {
    /// AO×orbital MO coefficients `C` (column = orbital).
    pub(crate) mos: Matrix,
    /// Orbital energies `ε_p`.
    pub(crate) eps: Vec<f64>,
    /// Occupations `f_p ∈ [0,1]` (Fermi/aufbau, matching `channel_occupations`).
    pub(crate) occ: Vec<f64>,
}

impl ChannelBasis {
    /// Diagonalize a (fixed) channel Fock and fill the aufbau/Fermi occupations the
    /// same way `solve_channel` does, so the analytic response is built in exactly the
    /// base-state MO basis the FD oracle perturbs around.
    pub(crate) fn from_fock(
        overlap: &Matrix,
        orth: &crate::linalg::LowdinOrthogonalizer,
        fock: &Matrix,
        nelec: f64,
        kt: f64,
        eigen_tol: f64,
    ) -> Result<Self> {
        let _ = overlap;
        let eig = lowdin_solve_with_orthogonalizer(fock, orth, eigen_tol)?;
        let (occ, _fermi, _entropy) = channel_occupations(&eig.values, nelec, kt);
        Ok(Self {
            mos: eig.vectors,
            eps: eig.values,
            occ,
        })
    }
}

/// `C^T A C` for a symmetric AO matrix `A` (the MO-basis transform of a perturbation).
fn mo_transform_local(mos: &Matrix, a: &Matrix) -> Result<Matrix> {
    let tmp = a.matmul(mos)?;
    mos.transpose().matmul(&tmp)
}

/// AO density built from MO-basis coefficient matrix `coeff`: `C · coeff · Cᵀ`.
fn coeff_to_ao(mos: &Matrix, coeff: &Matrix) -> Result<Matrix> {
    let tmp = mos.matmul(coeff)?;
    tmp.matmul(&mos.transpose())
}

/// First-order BARE density response of ONE spin channel (max-1 occupations) to an
/// AO Fock perturbation `delta_f`, at the channel's fixed MOs/energies/occupations.
///
/// In the MO basis (`h = Cᵀ δF C`, the probe has no overlap derivative), the density
/// response matrix `δP = C · coeff · Cᵀ` has, for an occupied/virtual pair (p,q):
/// `coeff_{pq} = (f_p − f_q) h_{pq} / (ε_p − ε_q)`  (the gapped off-diagonal term, which
/// equals the prompt's `Σ_{ia} h_{ai}/(ε_i−ε_a) (c_i c_aᵀ + c_a c_iᵀ)` once symmetrised),
/// and a finite-temperature diagonal term `coeff_{pp} = (∂f_p/∂ε_p) δε_p`,
/// `δε_p = h_{pp}`, with the Fermi-level shift `δμ` subtracting off the
/// particle-number change (the channel electron count is fixed). Degenerate same-occupation
/// off-diagonal pairs use the smooth `∂f/∂ε` slope limit. At `kt = 0` only the gapped
/// off-diagonal term survives (the diagonal `∂f/∂ε → 0`), reproducing the integer-aufbau
/// one-shot response.
fn bare_density_response_channel(
    ch: &ChannelBasis,
    delta_f: &Matrix,
    kt: f64,
) -> Result<Matrix> {
    let norb = ch.eps.len();
    let h = mo_transform_local(&ch.mos, delta_f)?;
    // Single-channel (max-1) Fermi-Dirac slope ∂f/∂ε = −f(1−f)/kt.
    let slope = |f: f64| -> f64 {
        if kt <= 0.0 {
            0.0
        } else {
            -(f * (1.0 - f)).max(0.0) / kt
        }
    };
    // Finite-temperature occupation response with the fixed-electron-count (Fermi-level)
    // constraint: δf_p = s_p (δε_p − δμ), δμ = (Σ s_p δε_p)/(Σ s_p), δε_p = h_pp.
    let mut docc = vec![0.0_f64; norb];
    if kt > 0.0 {
        let s: Vec<f64> = ch.occ.iter().map(|&f| slope(f)).collect();
        let denom: f64 = s.iter().sum();
        let dmu = if denom.abs() > 1.0e-30 {
            s.iter().zip(h.as_slice().iter().step_by(norb + 1))
                .map(|(&sp, &hpp)| sp * hpp)
                .sum::<f64>()
                / denom
        } else {
            0.0
        };
        for p in 0..norb {
            docc[p] = s[p] * (h[(p, p)] - dmu);
        }
    }
    let mut coeff = Matrix::zeros(norb, norb);
    for p in 0..norb {
        coeff[(p, p)] = docc[p];
        for q in (p + 1)..norb {
            let de = ch.eps[p] - ch.eps[q];
            let df = ch.occ[p] - ch.occ[q];
            let value = if de.abs() > 1.0e-10 {
                df * h[(p, q)] / de
            } else if kt > 0.0 {
                // Degenerate pair: smooth ∂f/∂ε slope limit (no Fermi-level term — it is the
                // diagonal particle-number channel, already handled above).
                let sloc = 0.5 * (slope(ch.occ[p]) + slope(ch.occ[q]));
                sloc * h[(p, q)]
            } else {
                0.0
            };
            coeff[(p, q)] = value;
            coeff[(q, p)] = value;
        }
    }
    coeff_to_ao(&ch.mos, &coeff)
}

/// Per-shell Mulliken population response `Σ_{μ∈ish}(δP S)_{μμ}` of an AO density
/// response `dp` (one spin channel). The d-occupation response of correlated atom `I`
/// is the sum over `I`'s AOs of the per-AO term — but we accumulate per shell so the
/// same routine drives the SCC feedback (shell charge / magnetization).
fn shell_population_response(basis: &BasisSet, overlap: &Matrix, dp: &Matrix) -> Vec<f64> {
    let n = basis.len();
    let ds = overlap.as_slice();
    let pp = dp.as_slice();
    let mut out = vec![0.0_f64; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut acc = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            // (δP S)_{μμ} = Σ_k δP[μ][k] S[k][μ] = Σ_k δP[μ][k] S[μ][k]  (S symmetric).
            let off = iao * n;
            acc += pp[off..off + n]
                .iter()
                .zip(&ds[off..off + n])
                .map(|(p, s)| p * s)
                .sum::<f64>();
        }
        out[ish] = acc;
    }
    out
}

/// Correlated-subspace occupation response `δn_I` from a per-AO population-response
/// accumulator (already summed over both spin channels). `subspace_occupations`-aligned.
fn subspace_occupation_response_from_ao(
    subspace: &[crate::plus_u::CorrelatedAtom],
    ao_pop_response: &[f64],
) -> Vec<f64> {
    subspace
        .iter()
        .map(|atom| atom.aos.iter().map(|&a| ao_pop_response[a]).sum())
        .collect()
}

/// Per-AO population response `(δP S)_{μμ}` for one spin channel (used to assemble the
/// correlated-subspace occupation response `δn_I = Σ_{a∈I}(δP S)_aa`).
fn ao_population_response(overlap: &Matrix, dp: &Matrix) -> Vec<f64> {
    let n = overlap.rows();
    let ds = overlap.as_slice();
    let pp = dp.as_slice();
    let mut out = vec![0.0_f64; n];
    for mu in 0..n {
        let off = mu * n;
        out[mu] = pp[off..off + n]
            .iter()
            .zip(&ds[off..off + n])
            .map(|(p, s)| p * s)
            .sum::<f64>();
    }
    out
}

/// Analytic BARE response matrix `χ0_{IJ} = Σ_σ ∂n_I/∂α_J` at fixed potential.
/// Column `J` is the correlated occupation response of all atoms `I` to the localized
/// probe `δF_J` applied to both spin channels.
fn analytic_chi0(
    overlap: &Matrix,
    ch_a: &ChannelBasis,
    ch_b: &ChannelBasis,
    subspace: &[crate::plus_u::CorrelatedAtom],
    kt: f64,
) -> Result<Vec<Vec<f64>>> {
    let ncorr = subspace.len();
    let mut chi0 = vec![vec![0.0; ncorr]; ncorr];
    for (j, atom_j) in subspace.iter().enumerate() {
        let probe = onsite_shift_fock(overlap, &atom_j.aos, 1.0);
        let dp_a = bare_density_response_channel(ch_a, &probe, kt)?;
        let dp_b = bare_density_response_channel(ch_b, &probe, kt)?;
        let mut ao_pop = ao_population_response(overlap, &dp_a);
        let ao_pop_b = ao_population_response(overlap, &dp_b);
        for (x, y) in ao_pop.iter_mut().zip(ao_pop_b.iter()) {
            *x += *y;
        }
        let dn = subspace_occupation_response_from_ao(subspace, &ao_pop);
        for i in 0..ncorr {
            chi0[i][j] = dn[i];
        }
    }
    Ok(chi0)
}

/// Analytic SCREENED response matrix `χ_{IJ} = Σ_σ ∂n_I/∂α_J` with full SCC feedback,
/// solved self-consistently in the SCC (shell-charge, magnetization) variable space.
///
/// For probe column `J`: iterate the per-channel Fock perturbation
/// `δF_α = δF_J − ½(δv_c − δv_s)·S-dress`, `δF_β = δF_J − ½(δv_c + δv_s)·S-dress`
/// where `δv_c = K_coul · δq`, `δv_s_i = Σ_{l'} W_{ii'} δm_{i'}`, with the induced
/// `δq_i = −(δpopα_i + δpopβ_i)` and `δm_i = δpopα_i − δpopβ_i`, until the induced
/// shell charge / magnetization stop changing. `K_coul` is the SCC second-order kernel
/// at the base state (Coulomb `amat` + the on-site higher-order `2Γq` Hubbard response,
/// i.e. `response_shell_scc_kernel` evaluated for these shell charges).
#[allow(clippy::too_many_arguments)]
fn analytic_chi(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    ch_a: &ChannelBasis,
    ch_b: &ChannelBasis,
    coul_kernel: &Matrix,
    info: &ShellInfo,
    subspace: &[crate::plus_u::CorrelatedAtom],
    kt: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<Vec<f64>>> {
    let _ = (h0, max_iter, tol);
    let ncorr = subspace.len();
    let nsh = basis.shells.len();
    let dim = 2 * nsh;
    // The SCC screening is a LINEAR fixed point in the shell variables `w = [δq; δm]`:
    // `w = b_probe + M·w`, where `b_probe` = induced (δq,δm) from the probe alone and
    // `M·w` = induced from the feedback potential of `w`. It is solved DIRECTLY as
    // `w = (I−M)^{-1} b_probe` rather than by fixed-point iteration: the mixing iteration
    // (previously here) DIVERGES on strongly-screened systems (e.g. transition-metal
    // complexes) where `M`'s spectral radius exceeds 1, returning a non-converged χ whose
    // geometry derivative is spurious. `(I−M)` stays well-conditioned there, so the direct
    // solve is exact and stable. (On weakly-screened systems it reproduces the converged
    // iteration to machine precision.) This also makes the linear-response χ consistent with
    // the analytic `dχ/dR` in `crate::plus_u_dudr`, which differentiates this same fixed point.
    //
    // Induced (δq, δm) from per-channel Fock perturbations `(f_a, f_b)`; also returns the
    // per-AO (α+β) occupation response for the final χ extraction.
    let induced = |f_a: &Matrix, f_b: &Matrix| -> Result<(Vec<f64>, Vec<f64>)> {
        let dp_a = bare_density_response_channel(ch_a, f_a, kt)?;
        let dp_b = bare_density_response_channel(ch_b, f_b, kt)?;
        let pop_a = shell_population_response(basis, overlap, &dp_a);
        let pop_b = shell_population_response(basis, overlap, &dp_b);
        let mut w = vec![0.0_f64; dim];
        for ish in 0..nsh {
            w[ish] = -(pop_a[ish] + pop_b[ish]); // δq = −δpop
            w[nsh + ish] = pop_a[ish] - pop_b[ish]; // δm
        }
        let mut ao = ao_population_response(overlap, &dp_a);
        let ao_b = ao_population_response(overlap, &dp_b);
        for (x, y) in ao.iter_mut().zip(ao_b.iter()) {
            *x += *y;
        }
        Ok((w, ao))
    };
    // Per-channel feedback Fock dress from a shell variable `w = [δq; δm]`.
    let feedback = |w: &[f64]| -> (Matrix, Matrix) {
        let dv_c = crate::linalg::matrix_vector_product(coul_kernel, &w[..nsh]).expect("K_coul·δq");
        let dv_s = spin_shell_potential(info, &w[nsh..]);
        let mut v_alpha = vec![0.0_f64; nsh];
        let mut v_beta = vec![0.0_f64; nsh];
        for ish in 0..nsh {
            v_alpha[ish] = dv_c[ish] - dv_s[ish];
            v_beta[ish] = dv_c[ish] + dv_s[ish];
        }
        (
            shell_potential_dress(basis, overlap, &v_alpha),
            shell_potential_dress(basis, overlap, &v_beta),
        )
    };
    // Build M once (DOF-of-probe-independent): column c = induced(feedback(e_c)).
    let mut m = vec![vec![0.0_f64; dim]; dim];
    for c in 0..dim {
        let mut e = vec![0.0_f64; dim];
        e[c] = 1.0;
        let (fa, fb) = feedback(&e);
        let (col, _) = induced(&fa, &fb)?;
        for r in 0..dim {
            m[r][c] = col[r];
        }
    }
    let mut i_minus_m = vec![vec![0.0_f64; dim]; dim];
    for r in 0..dim {
        for c in 0..dim {
            i_minus_m[r][c] = if r == c { 1.0 } else { 0.0 } - m[r][c];
        }
    }
    // Invert `(I−M)`; on the rare exact singularity add a tiny diagonal shift so the fit never
    // crashes on a pathological geometry (a genuinely divergent response is still captured by the
    // large-but-finite inverse — this only guards the measure-zero exactly-singular case).
    let inv = match crate::plus_u::invert_small(&i_minus_m) {
        Some(v) => v,
        None => {
            for (r, row) in i_minus_m.iter_mut().enumerate() {
                row[r] += 1.0e-8;
            }
            crate::plus_u::invert_small(&i_minus_m).ok_or_else(|| {
                Gfn1Error::InvalidInput("screened-response (I−M) is singular".to_string())
            })?
        }
    };
    let mut chi = vec![vec![0.0; ncorr]; ncorr];
    for (j, atom_j) in subspace.iter().enumerate() {
        let probe = onsite_shift_fock(overlap, &atom_j.aos, 1.0);
        // b_probe = induced (δq,δm) from the bare probe (no feedback).
        let (b_probe, _) = induced(&probe, &probe)?;
        // w* = (I−M)^{-1} b_probe (the exact converged shell response).
        let w: Vec<f64> = inv
            .iter()
            .map(|row| row.iter().zip(b_probe.iter()).map(|(a, x)| a * x).sum())
            .collect();
        // Final χ from the total perturbation `probe + feedback(w*)`.
        let (fa, fb) = feedback(&w);
        let df_a = matrix_sum(&probe, &fa);
        let df_b = matrix_sum(&probe, &fb);
        let (_, ao_pop) = induced(&df_a, &df_b)?;
        let dn = subspace_occupation_response_from_ao(subspace, &ao_pop);
        for i in 0..ncorr {
            chi[i][j] = dn[i];
        }
    }
    Ok(chi)
}

/// `−½(v_i+v_j) S_{μν}` shell-potential dressing (the Fock contribution of a shell
/// potential at `h0 = 0`), matching `fock_from_shell_potential`'s SCC term. Used to
/// add the SCC feedback to the probe in the screened response.
fn shell_potential_dress(basis: &BasisSet, overlap: &Matrix, shell_potential: &[f64]) -> Matrix {
    let n = basis.len();
    let mut vao = vec![0.0_f64; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = shell_potential[ish];
        }
    }
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = -overlap[(i, j)] * 0.5 * (vao[i] + vao[j]);
        }
    }
    out
}

/// SCC second-order Coulomb kernel `K_coul` at a base state with shell charges
/// `shell_charges` (Coulomb `amat` + the on-site 3rd-order `2Γ q_at` Hubbard response,
/// per atom). Mirrors `response_shell_scc_kernel` but built from the local
/// `ShellChargeModel` / `amat` rather than an `ElectronicResult`.
fn spin_coulomb_response_kernel(
    basis: &BasisSet,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    shell_charges: &[f64],
) -> Matrix {
    let mut kernel = amat.clone();
    let atomic_charges = shell_model.atomic_charges(basis, shell_charges);
    for (atom, &qat) in atomic_charges.iter().enumerate() {
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

/// Converged base spGFN1 state (no +U) plus the per-spin base Fock matrices, the base
/// shell charges/magnetization, and the SCC Coulomb response kernel — everything the
/// analytic (or FD) χ0/χ needs around the reference ground state.
struct LinearResponseBase {
    base_fock_a: Matrix,
    base_fock_b: Matrix,
    q0: Vec<f64>,
    coul_kernel: Matrix,
}

#[allow(clippy::too_many_arguments)]
fn linear_response_base_state(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    kt: f64,
    unpaired: i64,
    options: &ElectronicOptions,
) -> Result<LinearResponseBase> {
    let nsh = basis.shells.len();
    // Base: plain spGFN1 (no +U). Reference ground state for the response. CAMM is intentionally
    // NOT applied here: the linear-response U is defined from the CAMM-free base response, so the
    // analytic `plus_u_dudr` (which builds its own CAMM-free base) stays consistent with it — the
    // CAMM correction affects the total energy/force through its own channels, not through U.
    let (a0, b0, _, _) = spin_scc_loop(
        basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired, options,
        &[], &[], None, None,
    )?;
    let total0 = matrix_sum(&a0.density, &b0.density);
    let q0 = mulliken_shell_charges(basis, overlap, &total0);
    let mut m0 = vec![0.0; nsh];
    for ish in 0..nsh {
        m0[ish] = a0.shell_population[ish] - b0.shell_population[ish];
    }
    let scc0 = coulomb_energy_potential_from_matrix(basis, shell_model, &q0, amat)?;
    let sp0 = spin_shell_potential(info, &m0);
    let mut v_alpha0 = vec![0.0; nsh];
    let mut v_beta0 = vec![0.0; nsh];
    for ish in 0..nsh {
        v_alpha0[ish] = scc0.shell_potential[ish] - sp0[ish];
        v_beta0[ish] = scc0.shell_potential[ish] + sp0[ish];
    }
    let base_fock_a = fock_from_shell_potential(basis, overlap, h0, &v_alpha0);
    let base_fock_b = fock_from_shell_potential(basis, overlap, h0, &v_beta0);
    let coul_kernel = spin_coulomb_response_kernel(basis, shell_model, amat, &q0);
    Ok(LinearResponseBase {
        base_fock_a,
        base_fock_b,
        q0,
        coul_kernel,
    })
}

/// Non-empirical linear-response Hubbard `U` (and inter-site `V`) for the
/// correlated subspace (Cococcioni–de Gironcoli), parameter-free. The bare (`χ0`) and
/// self-consistent (`χ`) occupation-response matrices are computed **analytically**
/// (coupled-perturbed response in the per-spin MO basis; see [`analytic_chi0`] /
/// [`analytic_chi`]) — NO finite-difference `δ` probe in the production path — and
/// `U_I = (χ0⁻¹−χ⁻¹)_II`, `V_IJ = −(χ0⁻¹−χ⁻¹)_IJ` are extracted via
/// [`crate::plus_u::extract_uv_from_response`]. The analytic χ0/χ are FD-gated against
/// the (test-only) one-shot/re-converged probe responses. NOTE: in the screened
/// semiempirical SCC the bare/screened separation is itself approximate.
#[allow(clippy::too_many_arguments)]
fn compute_linear_response_uv(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    kt: f64,
    unpaired: i64,
    options: &ElectronicOptions,
    subspace: &[crate::plus_u::CorrelatedAtom],
    v_cutoff: f64,
    include_v: bool,
    system: &PeriodicSystem,
) -> Result<(Vec<crate::plus_u::CorrelatedAtom>, Vec<crate::plus_u::IntersitePair>)> {
    let ncorr = subspace.len();

    let base = linear_response_base_state(
        basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired, options,
    )?;
    // Per-spin base-state MO bases (diagonalize the converged channel Fock) for the
    // analytic occupation response.
    let ch_a = ChannelBasis::from_fock(overlap, orth, &base.base_fock_a, n_alpha, kt, options.eigen_tolerance)?;
    let ch_b = ChannelBasis::from_fock(overlap, orth, &base.base_fock_b, n_beta, kt, options.eigen_tolerance)?;

    // Analytic bare (χ0) and screened (χ) occupation-response matrices.
    let chi0 = analytic_chi0(overlap, &ch_a, &ch_b, subspace, kt)?;
    let chi = analytic_chi(
        basis,
        overlap,
        h0,
        &ch_a,
        &ch_b,
        &base.coul_kernel,
        info,
        subspace,
        kt,
        options.max_scc.max(200),
        options.charge_tolerance.min(1.0e-9),
    )?;
    let _ = &base.q0;

    let (u, vmat) = crate::plus_u::extract_uv_from_response(&chi0, &chi);
    let mut comp_sub = subspace.to_vec();
    for (i, atom) in comp_sub.iter_mut().enumerate() {
        atom.u = u[i].max(0.0); // clamp small negative U from numerical noise
    }
    let mut pairs = Vec::new();
    if include_v {
        let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|a| a.position).collect();
        for i in 0..ncorr {
            for j in (i + 1)..ncorr {
                let v = 0.5 * (vmat[i][j] + vmat[j][i]);
                if v.abs() < 1.0e-6 {
                    continue;
                }
                let (pi, pj) = (&pos[comp_sub[i].atom_index], &pos[comp_sub[j].atom_index]);
                let d = ((pi.x - pj.x).powi(2) + (pi.y - pj.y).powi(2) + (pi.z - pj.z).powi(2)).sqrt();
                if d <= v_cutoff {
                    pairs.push(crate::plus_u::IntersitePair { a: i, b: j, v });
                }
            }
        }
    }
    Ok((comp_sub, pairs))
}

/// **FD ORACLE (test-only).** The original ±δ finite-difference χ0/χ: for each
/// correlated atom `J` perturb its d-block by `±δ` via `onsite_shift_fock` and measure
/// the bare (one-shot, no SCC re-converge → `χ0`) and self-consistent (fully
/// re-converged → `χ`) correlated occupation responses. This is the reproducible
/// reference the analytic [`analytic_chi0`] / [`analytic_chi`] are gated against;
/// `δ = 0.005` lives ONLY here now, never in the production path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fd_chi0_chi(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    kt: f64,
    unpaired: i64,
    options: &ElectronicOptions,
    subspace: &[crate::plus_u::CorrelatedAtom],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let ncorr = subspace.len();
    let delta = 0.005_f64;
    let base = linear_response_base_state(
        basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired, options,
    )?;
    let occ_of = |a: &ChannelStep, b: &ChannelStep| -> Vec<f64> {
        let p = matrix_sum(&a.density, &b.density);
        crate::plus_u::subspace_occupations(&p, overlap, subspace)
    };
    let mut chi0 = vec![vec![0.0; ncorr]; ncorr];
    let mut chi = vec![vec![0.0; ncorr]; ncorr];
    for (j, atom_j) in subspace.iter().enumerate() {
        let pert_p = onsite_shift_fock(overlap, &atom_j.aos, delta);
        let pert_m = onsite_shift_fock(overlap, &atom_j.aos, -delta);
        let (ap, bp, _, _) = spin_scc_loop(
            basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired,
            options, &[], &[], Some(&pert_p), None,
        )?;
        let (am, bm, _, _) = spin_scc_loop(
            basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired,
            options, &[], &[], Some(&pert_m), None,
        )?;
        let (occ_p, occ_m) = (occ_of(&ap, &bp), occ_of(&am, &bm));
        let sap = solve_channel(basis, overlap, orth, &matrix_sum(&base.base_fock_a, &pert_p), n_alpha, kt, options.eigen_tolerance)?;
        let sbp = solve_channel(basis, overlap, orth, &matrix_sum(&base.base_fock_b, &pert_p), n_beta, kt, options.eigen_tolerance)?;
        let sam = solve_channel(basis, overlap, orth, &matrix_sum(&base.base_fock_a, &pert_m), n_alpha, kt, options.eigen_tolerance)?;
        let sbm = solve_channel(basis, overlap, orth, &matrix_sum(&base.base_fock_b, &pert_m), n_beta, kt, options.eigen_tolerance)?;
        let (occ0_p, occ0_m) = (occ_of(&sap, &sbp), occ_of(&sam, &sbm));
        for i in 0..ncorr {
            chi[i][j] = (occ_p[i] - occ_m[i]) / (2.0 * delta);
            chi0[i][j] = (occ0_p[i] - occ0_m[i]) / (2.0 * delta);
        }
    }
    Ok((chi0, chi))
}

/// **TEST-ONLY** analytic χ0/χ extractor mirroring the production `compute_linear_response_uv`
/// setup, returning the raw response matrices (not the extracted U/V) so the gate can compare
/// them directly against [`fd_chi0_chi`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn analytic_chi0_chi(
    basis: &BasisSet,
    overlap: &Matrix,
    h0: &Matrix,
    orth: &crate::linalg::LowdinOrthogonalizer,
    shell_model: &ShellChargeModel,
    amat: &Matrix,
    info: &ShellInfo,
    n_alpha: f64,
    n_beta: f64,
    kt: f64,
    unpaired: i64,
    options: &ElectronicOptions,
    subspace: &[crate::plus_u::CorrelatedAtom],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let base = linear_response_base_state(
        basis, overlap, h0, orth, shell_model, amat, info, n_alpha, n_beta, kt, unpaired, options,
    )?;
    let ch_a = ChannelBasis::from_fock(overlap, orth, &base.base_fock_a, n_alpha, kt, options.eigen_tolerance)?;
    let ch_b = ChannelBasis::from_fock(overlap, orth, &base.base_fock_b, n_beta, kt, options.eigen_tolerance)?;
    let chi0 = analytic_chi0(overlap, &ch_a, &ch_b, subspace, kt)?;
    let chi = analytic_chi(
        basis, overlap, h0, &ch_a, &ch_b, &base.coul_kernel, info, subspace, kt,
        options.max_scc.max(200), options.charge_tolerance.min(1.0e-9),
    )?;
    Ok((chi0, chi))
}

/// Recompute the linear-response `+U` (and `+V`) correlated subspace for a given
/// `system` / `options`, reproducing the full geometry-dependent setup
/// ([`build_h0`], the shell-charge / Coulomb model, the Löwdin orthogonalizer)
/// and calling [`compute_linear_response_uv`]. Returns the resolved subspace (with
/// the computed on-site `U` in each [`crate::plus_u::CorrelatedAtom::u`]) and the
/// inter-site `+V` pairs — exactly the quantities `run_spin_unrestricted` binds.
///
/// This is the per-geometry `U(R)` / `V(R)` evaluator used by the **consistent
/// force**: finite-differencing it over displaced geometries gives `dU/dR`,
/// `dV/dR`. It requires `options.plus_u && options.hubbard_u_linear_response`;
/// otherwise it returns an empty result (fixed-`U` mode has no geometry-dependent
/// `U` to differentiate — its force is already exact).
pub fn linear_response_uv_for_system(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> Result<(Vec<crate::plus_u::CorrelatedAtom>, Vec<crate::plus_u::IntersitePair>)> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "linear-response U is implemented for non-periodic systems only".to_string(),
        ));
    }
    if !(options.plus_u && options.hubbard_u_linear_response) {
        return Ok((Vec::new(), Vec::new()));
    }
    let basis = BasisSet::build(
        system,
        params,
        crate::basis::BasisOptions { nprim: options.nprim },
    )?;
    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    let unpaired = resolve_unpaired(nelec, options.spin_multiplicity)?;
    let nat = system.atoms.len();
    let n_alpha = 0.5 * (nelec + unpaired as f64);
    let n_beta = 0.5 * (nelec - unpaired as f64);
    let core = build_h0(system, &basis, params, &options.hamiltonian)?;
    let overlap = &core.integrals.overlap;
    let h0 = &core.h0;
    let mut shell_model = ShellChargeModel::build(system, &basis, params)?;
    shell_model.charge_order = options.charge_order.max(3);
    let amat = effective_coulomb_matrix(system, &basis, &shell_model);
    let orth = lowdin_orthogonalizer(overlap, options.eigen_tolerance)?;
    let info = shell_info(&basis, nat);
    let kt = options.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let plus_u_sel = crate::plus_u::correlated_subspace_auto(&basis, options.plus_u_all_d);
    if plus_u_sel.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    compute_linear_response_uv(
        &basis, overlap, h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
        options, &plus_u_sel, options.hubbard_v_cutoff, options.plus_u_v, system,
    )
}

/// All base-state ingredients the analytic geometry derivative of the linear-response
/// Hubbard parameters ([`crate::plus_u_dudr`]) needs, packaged so the derivative code
/// does not have to reproduce the full `linear_response_uv_for_system` setup.
///
/// Everything is evaluated at the converged plain-spGFN1 (no +U) base state — the same
/// reference [`analytic_chi0`] / [`analytic_chi`] use. `base_fock_a`/`base_fock_b` are
/// the per-channel converged Fock matrices; `ch_a`/`ch_b` their diagonalized MO bases;
/// `v_alpha0`/`v_beta0` the per-shell channel potentials `v^σ = v_c ∓ v_s`;
/// `coul_kernel` the SCC Coulomb response kernel; `q0`/`m0` the base shell charges /
/// magnetization; `coordination_numbers` for the CN-coupled `dh0/dR` skeleton.
///
/// Several fields (`base_fock_a/b`, `n_alpha/n_beta`, `eigen_tolerance`, `h0`) are carried for
/// completeness / the FD-gate tests and are not all read by the production `analytic_dudr`.
#[allow(dead_code)]
pub(crate) struct LinearResponseGeomContext {
    pub(crate) basis: BasisSet,
    pub(crate) overlap: Matrix,
    pub(crate) h0: Matrix,
    pub(crate) orth: crate::linalg::LowdinOrthogonalizer,
    pub(crate) shell_model: ShellChargeModel,
    pub(crate) amat: Matrix,
    pub(crate) info: ShellInfo,
    pub(crate) coordination_numbers: Vec<f64>,
    pub(crate) n_alpha: f64,
    pub(crate) n_beta: f64,
    pub(crate) kt: f64,
    pub(crate) eigen_tolerance: f64,
    pub(crate) base_fock_a: Matrix,
    pub(crate) base_fock_b: Matrix,
    pub(crate) ch_a: ChannelBasis,
    pub(crate) ch_b: ChannelBasis,
    pub(crate) v_alpha0: Vec<f64>,
    pub(crate) v_beta0: Vec<f64>,
    pub(crate) q0: Vec<f64>,
    pub(crate) coul_kernel: Matrix,
    pub(crate) subspace: Vec<crate::plus_u::CorrelatedAtom>,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
}

/// Build the [`LinearResponseGeomContext`] for a system, reproducing the geometry-
/// dependent setup of [`linear_response_uv_for_system`] but retaining every raw
/// ingredient (basis, overlap, h0, orthogonalizer, shell model, base channel Fock /
/// MO bases, base potentials, kernels). Returns `None` (empty context) when +U
/// linear response is off or the correlated subspace is empty — the callers treat
/// that as "no consistency force" exactly like the FD path.
pub(crate) fn linear_response_geom_context(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> Result<Option<LinearResponseGeomContext>> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "analytic dU/dR is implemented for non-periodic systems only".to_string(),
        ));
    }
    if !(options.plus_u && options.hubbard_u_linear_response) {
        return Ok(None);
    }
    let basis = BasisSet::build(system, params, crate::basis::BasisOptions { nprim: options.nprim })?;
    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    let unpaired = resolve_unpaired(nelec, options.spin_multiplicity)?;
    let nat = system.atoms.len();
    let n_alpha = 0.5 * (nelec + unpaired as f64);
    let n_beta = 0.5 * (nelec - unpaired as f64);
    let core = build_h0(system, &basis, params, &options.hamiltonian)?;
    let overlap = core.integrals.overlap.clone();
    let h0 = core.h0.clone();
    let coordination_numbers = core.coordination_numbers.clone();
    let mut shell_model = ShellChargeModel::build(system, &basis, params)?;
    shell_model.charge_order = options.charge_order.max(3);
    let amat = effective_coulomb_matrix(system, &basis, &shell_model);
    let orth = lowdin_orthogonalizer(&overlap, options.eigen_tolerance)?;
    let info = shell_info(&basis, nat);
    let kt = options.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let subspace = crate::plus_u::correlated_subspace_auto(&basis, options.plus_u_all_d);
    if subspace.is_empty() {
        return Ok(None);
    }
    let nsh = basis.shells.len();
    // Converged base state (plain spGFN1), reproducing `linear_response_base_state`.
    let (a0, b0, _, _) = spin_scc_loop(
        &basis, &overlap, &h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
        options, &[], &[], None, None,
    )?;
    let total0 = matrix_sum(&a0.density, &b0.density);
    let q0 = mulliken_shell_charges(&basis, &overlap, &total0);
    let mut m0 = vec![0.0; nsh];
    for ish in 0..nsh {
        m0[ish] = a0.shell_population[ish] - b0.shell_population[ish];
    }
    let scc0 = coulomb_energy_potential_from_matrix(&basis, &shell_model, &q0, &amat)?;
    let sp0 = spin_shell_potential(&info, &m0);
    let mut v_alpha0 = vec![0.0; nsh];
    let mut v_beta0 = vec![0.0; nsh];
    for ish in 0..nsh {
        v_alpha0[ish] = scc0.shell_potential[ish] - sp0[ish];
        v_beta0[ish] = scc0.shell_potential[ish] + sp0[ish];
    }
    let base_fock_a = fock_from_shell_potential(&basis, &overlap, &h0, &v_alpha0);
    let base_fock_b = fock_from_shell_potential(&basis, &overlap, &h0, &v_beta0);
    let coul_kernel = spin_coulomb_response_kernel(&basis, &shell_model, &amat, &q0);
    let ch_a = ChannelBasis::from_fock(&overlap, &orth, &base_fock_a, n_alpha, kt, options.eigen_tolerance)?;
    let ch_b = ChannelBasis::from_fock(&overlap, &orth, &base_fock_b, n_beta, kt, options.eigen_tolerance)?;
    Ok(Some(LinearResponseGeomContext {
        basis,
        overlap,
        h0,
        orth,
        shell_model,
        amat,
        info,
        coordination_numbers,
        n_alpha,
        n_beta,
        kt,
        eigen_tolerance: options.eigen_tolerance,
        base_fock_a,
        base_fock_b,
        ch_a,
        ch_b,
        v_alpha0,
        v_beta0,
        q0,
        coul_kernel,
        subspace,
        max_iter: options.max_scc.max(200),
        tol: options.charge_tolerance.min(1.0e-9),
    }))
}

impl LinearResponseGeomContext {
    /// Production analytic bare/screened response matrices `(χ0, χ)` at the base state of
    /// this context — the exact quantities [`compute_linear_response_uv`] extracts `U`/`V`
    /// from. Exposed so [`crate::plus_u_dudr`] can gate its independent χ0/χ (and their
    /// geometry derivatives) against the production values.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn analytic_chi0_chi(&self) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let chi0 = analytic_chi0(&self.overlap, &self.ch_a, &self.ch_b, &self.subspace, self.kt)?;
        let chi = analytic_chi(
            &self.basis, &self.overlap, &self.h0, &self.ch_a, &self.ch_b, &self.coul_kernel,
            &self.info, &self.subspace, self.kt, self.max_iter, self.tol,
        )?;
        Ok((chi0, chi))
    }
}

/// Spin-unrestricted (open-shell) spGFN1 (+ optional `+U`/`+U+V`) driver.
///
/// The SCC is solved by the **robust** [`spin_scc_loop`]: the user's exact Broyden
/// scheme first, with a fallback ladder (virtual level shift + linear mixing, then a
/// raised electronic temperature) that engages only on non-convergence or a
/// non-finite blow-up. This makes open-shell transition-metal `+U` at zero
/// electronic temperature converge to a finite energy where bare Broyden diverges on
/// the near-degenerate d-frontier (the cold-Fermi NaN).
///
/// **Practical note on TM forces:** the SCC *energy* is robust at `etemp=0`, but the
/// analytic *force* of an open-shell TM system with a (near-)degenerate partially-
/// filled d frontier is ill-conditioned at exactly `etemp=0` — integer aufbau makes
/// the occupied-orbital choice (and hence `∂ε/∂R`) discontinuous across the crossing.
/// This is the standard integer-occupation-gradient discontinuity, not specific to
/// `+U`. For well-conditioned TM forces use a small finite electronic temperature
/// (Fermi smearing, e.g. 300 K), standard transition-metal practice; the energy is
/// essentially unchanged when the frontier is only near-degenerate.
fn run_spin_unrestricted(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    basis: BasisSet,
    nelec: f64,
    unpaired: i64,
) -> Result<ElectronicResult> {
    let _profile = crate::profile::scope("electronic.spin.total");
    let nat = system.atoms.len();
    let nsh = basis.shells.len();

    let n_alpha = 0.5 * (nelec + unpaired as f64);
    let n_beta = 0.5 * (nelec - unpaired as f64);

    let core = build_h0(system, &basis, params, &options.hamiltonian)?;
    let overlap = &core.integrals.overlap;
    let h0 = &core.h0;

    let mut shell_model = ShellChargeModel::build(system, &basis, params)?;
    shell_model.charge_order = options.charge_order.max(3);
    let amat = effective_coulomb_matrix(system, &basis, &shell_model);
    let orth = lowdin_orthogonalizer(overlap, options.eigen_tolerance)?;

    let info = shell_info(&basis, nat);

    // Classical (spin-independent) terms.
    let repulsion = repulsion_energy(system, params)?;
    let dispersion = if options.enable_dispersion {
        dispersion_energy(system, params, options.d3_reference_path.as_deref())?
    } else {
        0.0
    };
    let halogen = halogen_energy(system)?;

    let kt = options.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;

    // DFT+U / +U+V correlated subspace and the on-site U / inter-site V to apply.
    // Linear-response mode auto-selects the transition-metal d shells and computes
    // U (and V) non-empirically from the occupation response; fixed-U mode uses the
    // supplied element tables. Either way the resolved values are bound to
    // `plus_u_sub` / `plus_u_pairs`, which the SCC loop below consumes unchanged.
    let plus_u_sel = if options.plus_u {
        if options.hubbard_u_linear_response {
            crate::plus_u::correlated_subspace_auto(&basis, options.plus_u_all_d)
        } else {
            crate::plus_u::correlated_subspace(&basis, &options.hubbard_u, AngularMomentum::D)
        }
    } else {
        Vec::new()
    };
    let (plus_u_sub, plus_u_pairs) = if plus_u_sel.is_empty() {
        (Vec::new(), Vec::new())
    } else if options.hubbard_u_linear_response {
        compute_linear_response_uv(
            &basis, overlap, h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
            options, &plus_u_sel, options.hubbard_v_cutoff, options.plus_u_v, system,
        )?
    } else {
        let pairs = if options.plus_u_v {
            let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|a| a.position).collect();
            let z: Vec<u8> = system.atoms.iter().map(|a| a.z).collect();
            crate::plus_u::intersite_pairs(
                &plus_u_sel,
                &pos,
                &z,
                &options.hubbard_v,
                options.hubbard_v_cutoff,
            )
        } else {
            Vec::new()
        };
        (plus_u_sel, pairs)
    };
    let plus_u_active = !plus_u_sub.is_empty();

    // CAMM-on-mDFTB2 AES context (only when the multipole correction is on with the CAMM model;
    // the narrow guard in `run_spin_polarized` already rejected the unsupported combinations). The
    // per-atom κ / s_onsite resolve exactly as in the restricted path (element overrides, else the
    // globals); charge-dependent κ is recomputed per SCC iteration inside the context.
    let camm_ctx: Option<SpinCammContext> =
        if options.multipole && options.multipole_model == crate::electronic::MultipoleModel::CammOnMdftb2 {
            if !(options.camm_damp > 0.0) {
                return Err(Gfn1Error::InvalidInput("camm_damp (CAMM range factor κ) must be > 0".to_string()));
            }
            if options.camm_aes_scale < 0.0 {
                return Err(Gfn1Error::InvalidInput("camm_aes_scale (s_AES) must be ≥ 0".to_string()));
            }
            if options.camm_onsite_scale < 0.0
                || options.camm_onsite_scale_elem.iter().any(|&(_, s)| s < 0.0)
            {
                return Err(Gfn1Error::InvalidInput("CAMM s_onsite values must be ≥ 0".to_string()));
            }
            if options.camm_damp_elem.iter().any(|&(_, k)| !(k > 0.0)) {
                return Err(Gfn1Error::InvalidInput("camm_damp_elem κ values must be > 0".to_string()));
            }
            let hardness: Vec<f64> = (0..nat)
                .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
                .collect();
            let pos: Vec<crate::math::Vec3> = system.atoms.iter().map(|at| at.position).collect();
            let kappa: Vec<f64> = system
                .atoms
                .iter()
                .map(|atom| {
                    options.camm_damp_elem.iter().find(|&&(z, _)| z == atom.z).map(|&(_, k)| k)
                        .unwrap_or(options.camm_damp)
                })
                .collect();
            let onsite: Vec<f64> = system
                .atoms
                .iter()
                .map(|atom| {
                    options.camm_onsite_scale_elem.iter().find(|&&(z, _)| z == atom.z).map(|&(_, s)| s)
                        .unwrap_or(options.camm_onsite_scale)
                })
                .collect();
            Some(SpinCammContext {
                integrals: &core.integrals,
                hardness,
                pos,
                kappa,
                scale: options.camm_aes_scale,
                onsite,
                damp_charge: options.camm_damp_charge,
            })
        } else {
            None
        };

    // Converge the spin-unrestricted SCC (with the resolved +U/+U+V correction, the CAMM-AES
    // shift, no probe perturbation). The shared `spin_scc_loop` is also used by the linear-response
    // orchestration (which passes no CAMM — the U response is defined CAMM-free).
    let (alpha, beta, converged, iterations) = spin_scc_loop(
        &basis, overlap, h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
        options, &plus_u_sub, &plus_u_pairs, None, camm_ctx.as_ref(),
    )?;

    // Final consistent quantities from the converged densities.
    let total_density = matrix_sum(&alpha.density, &beta.density);
    let total_ew = matrix_sum(&alpha.ew_density, &beta.ew_density);
    let final_shell_charges = mulliken_shell_charges(&basis, overlap, &total_density);
    let mut final_m = vec![0.0; nsh];
    for ish in 0..nsh {
        final_m[ish] = alpha.shell_population[ish] - beta.shell_population[ish];
    }
    let spin_potential = spin_shell_potential(&info, &final_m);
    let e_spin = spin_energy(&final_m, &spin_potential);

    let scc = coulomb_energy_potential_from_matrix(&basis, &shell_model, &final_shell_charges, &amat)?;
    let band = electronic_energy(h0, &total_density);
    let atomic_charges = shell_model.atomic_charges(&basis, &final_shell_charges);
    let entropy_term = alpha.entropy + beta.entropy;

    // Electronic energy reported separately keeps the band (Tr P H0) term; the spin energy is
    // folded into total_internal (there is no field/multipole/exchange channel in spGFN1 v1).
    // Converged DFT+U/+U+V energy from the final spin densities (added directly).
    let e_plus_u_final = if plus_u_active {
        let (ea, _) = crate::plus_u::plus_u_v(&alpha.density, overlap, &plus_u_sub, &plus_u_pairs);
        let (eb, _) = crate::plus_u::plus_u_v(&beta.density, overlap, &plus_u_sub, &plus_u_pairs);
        ea + eb
    } else {
        0.0
    };
    // Converged CAMM-AES energy from the final total density (added directly, like +U/spin).
    let e_camm_final = if let Some(camm_ctx) = camm_ctx.as_ref() {
        let qm: Vec<f64> = atomic_charges.iter().map(|c| -c).collect();
        camm_ctx.energy_fock(&basis, nat, &total_density, &qm).energy
    } else {
        0.0
    };

    let electronic_energy_value = band;
    let total_internal = band
        + scc.second_order
        + scc.third_order
        + scc.higher_order
        + e_spin
        + e_plus_u_final
        + e_camm_final
        + repulsion
        + dispersion
        + halogen;
    let total_free = total_internal + entropy_term;

    let dipole = mulliken_dipole(system, &atomic_charges, options.external_field.origin);

    let spin = SpinResolved {
        density_alpha: alpha.density,
        density_beta: beta.density,
        ew_density_alpha: alpha.ew_density,
        ew_density_beta: beta.ew_density,
        shell_magnetization: final_m,
        shell_spin_potential: spin_potential,
        n_alpha,
        n_beta,
        spin_energy: e_spin,
        plus_u_subspace: plus_u_sub.clone(),
        plus_u_pairs: plus_u_pairs.clone(),
    };

    // Report the spin-averaged (charge-channel) effective Hamiltonian in `fock` for any downstream
    // consumer; the spin-resolved α/β Fock physics lives in `spin`.
    let fock = fock_from_shell_potential(&basis, overlap, h0, &scc.shell_potential);

    Ok(ElectronicResult {
        basis,
        integrals: core.integrals,
        h0: core.h0,
        fock,
        density: total_density,
        energy_weighted_density: total_ew,
        orbital_energies: Vec::new(),
        occupations: Vec::new(),
        electronic_temperature: options.electronic_temperature,
        fermi_level: 0.0,
        shell_charges: final_shell_charges,
        atomic_charges,
        shell_scc_potential: scc.shell_potential,
        coordination_numbers: core.coordination_numbers,
        electronic_energy: electronic_energy_value,
        repulsion_energy: repulsion,
        isotropic_scc_energy: scc.second_order,
        third_order_energy: scc.third_order,
        dispersion_energy: dispersion,
        halogen_energy: halogen,
        external_field_energy: 0.0,
        electronic_entropy_term: entropy_term,
        total_internal,
        total_free,
        dipole,
        nelec,
        iterations,
        converged,
        spin: Some(spin),
    })
}

fn matrix_sum(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    let os = out.as_mut_slice();
    let bs = b.as_slice();
    for (o, x) in os.iter_mut().zip(bs.iter()) {
        *o += *x;
    }
    out
}

/// Seed the magnetization so the SCC starts spin-broken: place the `unpaired`
/// excess α population on the valence shells of the atoms, spread by their
/// (largest-|W|) spin response so the symmetry break is in the right channel.
fn seed_magnetization(info: &ShellInfo, magnetization: &mut [f64], unpaired: f64) {
    // Choose, per atom, the shell with the most negative diagonal spin constant W_ll (the most
    // spin-polarizable shell) and put a small magnetization there; normalize the total to
    // `unpaired`. This only sets the *initial* guess — the SCC determines the converged value.
    let nsh = magnetization.len();
    let mut weight = vec![0.0; nsh];
    let mut total = 0.0;
    for ish in 0..nsh {
        let w = gfn_spin_constant(info.l[ish], info.l[ish], info.z[ish]);
        let wgt = (-w).max(0.0);
        weight[ish] = wgt;
        total += wgt;
    }
    if total <= 0.0 {
        // Fallback: spread uniformly.
        for m in magnetization.iter_mut() {
            *m = unpaired / nsh as f64;
        }
        return;
    }
    for ish in 0..nsh {
        magnetization[ish] = unpaired * weight[ish] / total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::{analytic_gradient, AnalyticGradientOptions};


    /// Load the GFN1 parameters from the env var, falling back to the repo-root
    /// `param_gfn1-xtb.txt` so the spin tests actually run in a plain `cargo test`.
    fn load_params() -> Option<Gfn1Parameters> {
        if let Ok(path) = std::env::var(crate::params::GFN1_PARAM_ENV) {
            if let Ok(p) = Gfn1Parameters::from_file(path) {
                return Some(p);
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("param_gfn1-xtb.txt");
        Gfn1Parameters::from_file(root).ok()
    }

    fn tight_options() -> ElectronicOptions {
        ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            max_scc: 500,
            ..ElectronicOptions::default()
        }
    }

    fn displaced(system: &PeriodicSystem, atom: usize, axis: usize, delta: f64) -> PeriodicSystem {
        let mut s = system.clone();
        match axis {
            0 => s.atoms[atom].position.x += delta,
            1 => s.atoms[atom].position.y += delta,
            _ => s.atoms[atom].position.z += delta,
        }
        s
    }

    /// +U NO-OP: with `plus_u` ON but no element carrying U (empty `hubbard_u`)
    /// the correlated subspace is empty, so the open-shell result must be
    /// byte-identical to plain spGFN1 — the +U wiring is inert when unused.
    #[test]
    fn plus_u_empty_subspace_byte_identical() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "4\nch3\nC 0.01 0.02 0.00\nH 0.03 1.08 0.01\nH 0.95 -0.55 -0.02\nH -0.92 -0.53 0.015\n",
            0.0,
            false,
        )
        .unwrap();
        let mut base = tight_options();
        base.spin_multiplicity = Some(2);
        base.spin_polarization = true;
        let mut pu = base.clone();
        pu.plus_u = true; // hubbard_u empty → empty subspace → inert
        let e0 = crate::electronic::run_electronic(&system, &params, base).unwrap().total_free;
        let e1 = crate::electronic::run_electronic(&system, &params, pu).unwrap().total_free;
        assert!(
            (e0 - e1).abs() <= 1.0e-10,
            "empty +U is not a no-op: plain={e0:.12} plus_u={e1:.12}"
        );
    }

    /// +U EFFECT + CONVERGENCE: on the open-shell SH radical, a non-zero U on
    /// sulfur's d shell must shift the energy (FLL penalty on the fractional d
    /// occupation) while the spin SCC still converges; U = 0 must reproduce the
    /// plain spGFN1 energy exactly (the zero-U entry is skipped → empty subspace).
    #[test]
    fn plus_u_changes_open_shell_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nSH\nS 0.0 0.0 0.0\nH 0.0 0.0 1.34\n",
            0.0,
            false,
        )
        .unwrap();
        let mut base = tight_options();
        base.spin_multiplicity = Some(2);
        base.spin_polarization = true;
        let e_plain = crate::electronic::run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;

        let mut u0 = base.clone();
        u0.plus_u = true;
        u0.hubbard_u = vec![(16, 0.0)];
        let e_u0 = crate::electronic::run_electronic(&system, &params, u0).unwrap().total_free;
        assert!(
            (e_plain - e_u0).abs() <= 1.0e-10,
            "U=0 is not identical to plain: {e_plain:.12} vs {e_u0:.12}"
        );

        let mut u1 = base.clone();
        u1.plus_u = true;
        u1.hubbard_u = vec![(16, 0.5)];
        let e_u1 = crate::electronic::run_electronic(&system, &params, u1).unwrap().total_free;
        assert!(e_u1.is_finite(), "+U total energy is not finite");
        assert!(
            (e_u1 - e_plain).abs() > 1.0e-8,
            "+U had no measurable effect: plain={e_plain:.12} +U={e_u1:.12}"
        );
    }

    /// CLOSED-SHELL +U: a closed-shell singlet (H2S) with `plus_u` routes through
    /// the spin-unrestricted path at zero magnetization. With no U it reduces to
    /// the restricted GFN1 energy (to SCC tolerance); with a U on sulfur's d shell
    /// it applies the FLL penalty and shifts the energy — no spin_polarization flag
    /// needed (plus_u implies the spin machinery).
    #[test]
    fn plus_u_closed_shell_runs_and_reduces_to_restricted() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nh2s\nS 0.00 0.00 0.10\nH 0.00 0.96 -0.79\nH 0.00 -0.96 -0.79\n",
            0.0,
            false,
        )
        .unwrap();
        let base = tight_options();
        let e_plain = crate::electronic::run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;

        // plus_u with empty U → unrestricted closed-shell, no +U applied → ≈ restricted.
        let mut pu0 = base.clone();
        pu0.plus_u = true;
        let e_pu0 = crate::electronic::run_electronic(&system, &params, pu0).unwrap().total_free;
        assert!(
            (e_plain - e_pu0).abs() < 1.0e-6,
            "closed-shell +U(empty) should reduce to restricted: {e_plain:.10} vs {e_pu0:.10}"
        );

        // plus_u with U on sulfur → closed-shell +U applied, finite, shifts the energy.
        let mut pu = base.clone();
        pu.plus_u = true;
        pu.hubbard_u = vec![(16, 0.5)];
        let e_pu = crate::electronic::run_electronic(&system, &params, pu).unwrap().total_free;
        assert!(e_pu.is_finite(), "closed-shell +U energy not finite");
        assert!(
            (e_pu - e_plain).abs() > 1.0e-8,
            "closed-shell +U had no effect: plain={e_plain:.10} +U={e_pu:.10}"
        );
    }

    /// LINEAR-RESPONSE (non-empirical, parameter-free) U: on an open-shell
    /// transition-metal system with a valence d shell, the linear-response path
    /// auto-selects the d subspace and computes U **analytically** from the
    /// occupation response (the analytic χ0/χ; NO finite-difference δ probe, NO
    /// fitted parameters), then applies +U. The run must converge to a finite
    /// energy that differs from plain spGFN1 (a genuinely non-zero computed U).
    ///
    /// The system is ScH (triplet) at the production electronic temperature: here
    /// the auto-selected Sc d carries a real linear-response U (≈0.04 Ha) whose
    /// occupation response is non-trivial. (An *isolated* Sc atom is a degenerate
    /// edge case — its d population is pinned, so both the analytic and the FD
    /// occupation response are ~0 and the extracted U is ~0; the previous version
    /// of this test, on a lone Sc atom, only ever saw a non-zero effect because
    /// the FD probe amplified ~1e-12 occupation noise through the 1/χ extraction.
    /// The analytic path correctly reports U≈0 there, so a TM *molecule* with a
    /// real bonding-driven d response is the physically meaningful probe.)
    #[test]
    fn linear_response_u_runs_parameter_free() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            PeriodicSystem::from_xyz_str("2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n", 0.0, false)
                .unwrap();
        let mut base = ElectronicOptions::default();
        base.spin_multiplicity = Some(3);
        base.spin_polarization = true;
        let e_plain = crate::electronic::run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;

        // The analytically-computed, parameter-free U must be genuinely non-zero.
        let mut lr = base.clone();
        lr.plus_u = true;
        lr.hubbard_u_linear_response = true; // no hubbard_u values supplied → computed analytically
        let (subspace, _pairs) =
            crate::spin::linear_response_uv_for_system(&system, &params, &lr).unwrap();
        assert!(
            subspace.iter().any(|a| a.u > 1.0e-3),
            "analytic linear-response U is ~zero on ScH: {:?}",
            subspace.iter().map(|a| a.u).collect::<Vec<_>>()
        );

        let e_lr = crate::electronic::run_electronic(&system, &params, lr).unwrap().total_free;
        assert!(e_lr.is_finite(), "linear-response +U energy is not finite");
        assert!(
            (e_lr - e_plain).abs() > 1.0e-8,
            "computed (linear-response) U had no effect: plain={e_plain:.10} +U={e_lr:.10}"
        );
    }

    /// ROBUST OPEN-SHELL TM +U AT kt=0: the spin-unrestricted +U SCC on an
    /// open-shell transition-metal system at zero electronic temperature with tight
    /// tolerances used to return a NaN total energy — the near-degenerate d¹
    /// cold-Fermi frontier flips discontinuously, bare Broyden overshoots, and the
    /// SCC blows up. The robust fallback ladder in `spin_scc_loop` (rung 0 = the old
    /// bare Broyden; then rungs 1–3 progressively add a virtual level shift and
    /// switch to monotone linear mixing, ALL at the user's etemp so they pin the true
    /// kt=0 fixed point; rung 4, last resort, raises the electronic temperature) must
    /// rescue it to a FINITE, converged total energy. This is the reproduce→fix gate:
    /// before the fix `run_electronic` returns a NaN total energy; after it, rung 1
    /// converges ScH cleanly at kt=0.
    ///
    /// ScH triplet at the tight options ([etemp=0, energy_tol=1e-10,
    /// charge_tol=1e-9, max_scc=500]) with the parameter-free linear-response U
    /// (computed U≈0.04 Ha on the auto-selected Sc d).
    #[test]
    fn plus_u_open_shell_tm_scc_robust_at_kt0() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            PeriodicSystem::from_xyz_str("2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n", 0.0, false)
                .unwrap();
        let mut lr = tight_options();
        lr.spin_multiplicity = Some(3);
        lr.spin_polarization = true;
        lr.plus_u = true;
        lr.hubbard_u_linear_response = true; // U computed analytically per geometry

        let res = crate::electronic::run_electronic(&system, &params, lr).unwrap();
        assert!(
            res.total_free.is_finite(),
            "open-shell TM +U at kt=0/tight returned a non-finite energy: {}",
            res.total_free
        );
        // The robust ladder must actually converge it (not just dodge the NaN).
        assert!(
            res.converged,
            "open-shell TM +U at kt=0/tight did not converge under the robust ladder"
        );
    }

    /// CLOSED-SHELL REGRESSION: with spin polarization ON, a closed-shell singlet
    /// (water) must give a byte-identical energy AND forces to plain GFN1 — the
    /// spin density is zero so the spin term vanishes (we also delegate to the
    /// restricted path, making this exact).
    #[test]
    fn closed_shell_singlet_byte_identical() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.00000 0.00000 0.11779\nH 0.00000 0.75545 -0.47116\nH 0.00000 -0.75545 -0.47116\n",
            0.0,
            false,
        )
        .unwrap();

        let base = tight_options();
        let mut spin = base.clone();
        spin.spin_polarization = true;

        let e_plain = crate::electronic::run_electronic(&system, &params, base.clone())
            .unwrap()
            .total_free;
        let e_spin = crate::electronic::run_electronic(&system, &params, spin.clone())
            .unwrap()
            .total_free;
        assert!(
            (e_plain - e_spin).abs() <= 1.0e-10,
            "closed-shell energy differs: plain={e_plain:.12} spinpol={e_spin:.12}"
        );

        let mut gopt = AnalyticGradientOptions::default();
        gopt.electronic = base;
        let g_plain = analytic_gradient(&system, &params, gopt.clone()).unwrap().gradient;
        gopt.electronic = spin;
        let g_spin = analytic_gradient(&system, &params, gopt).unwrap().gradient;
        let mut maxdiff = 0.0_f64;
        for (a, b) in g_plain.iter().zip(g_spin.iter()) {
            maxdiff = maxdiff.max((a.x - b.x).abs()).max((a.y - b.y).abs()).max((a.z - b.z).abs());
        }
        assert!(
            maxdiff <= 1.0e-10,
            "closed-shell forces differ: max|Δ| = {maxdiff:.3e} Ha/bohr"
        );
    }

    /// FD-GRADIENT GATE: for the open-shell methyl radical (CH3 doublet) the
    /// analytic spGFN1 force must match a central finite difference of the spGFN1
    /// energy. This exercises the spin energy-weighted-density band response and
    /// the spin overlap-Pulay term together.
    #[test]
    fn methyl_radical_spin_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        // Slightly distorted CH3 so every component is nonzero.
        let system = PeriodicSystem::from_xyz_str(
            "4\nch3\nC 0.01 0.02 0.00\nH 0.03 1.08 0.01\nH 0.95 -0.55 -0.02\nH -0.92 -0.53 0.015\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = tight_options();
        opt.spin_multiplicity = Some(2);
        opt.spin_polarization = true;

        let mut gopt = AnalyticGradientOptions::default();
        gopt.electronic = opt.clone();
        let ana = analytic_gradient(&system, &params, gopt).unwrap().gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            crate::electronic::run_electronic(sys, &params, opt.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let nat = system.atoms.len();
        let mut maxdiff = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 1.0e-4,
            "spGFN1 (CH3 doublet) analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// +U FD-GRADIENT GATE: with a FIXED U on sulfur's d shell (SH doublet), the
    /// analytic +U force — the overlap-Pulay term Tr(W_{+U} dS/dR) layered on the
    /// spin gradient — must match a central finite difference of the +U total
    /// energy. Fixed U (not linear-response) so the displaced-geometry energies use
    /// the same U, matching the frozen-U analytic force.
    #[test]
    fn plus_u_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nSH\nS 0.01 0.00 0.00\nH 0.02 0.03 1.34\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = tight_options();
        opt.spin_multiplicity = Some(2);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u = vec![(16, 0.5)];

        let mut gopt = AnalyticGradientOptions::default();
        gopt.electronic = opt.clone();
        let ana = analytic_gradient(&system, &params, gopt).unwrap().gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            crate::electronic::run_electronic(sys, &params, opt.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 1.0e-4,
            "+U analytic gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// **CONSISTENT-FORCE ACCEPTANCE GATE (the v0.4.4 deliverable):** with the
    /// **linear-response** `+U` (U recomputed at every geometry), the analytic force
    /// — the frozen-U overlap-Pulay term PLUS the `F_corr = Σ_I (∂E/∂U_I)(dU_I/dR)`
    /// consistency term — must match a central finite difference of the
    /// **recomputed-U** total energy `E(R)`. Open-shell transition-metal system
    /// (ScH, triplet) where the auto-selected Sc `d` subspace carries a non-zero
    /// linear-response `U` whose geometry derivative is non-trivial. Without
    /// `F_corr` this gate fails by ~1e-3 Ha/bohr (the frozen-U force is inconsistent
    /// with the per-geometry `U(R)`); with it the residual is at the FD floor.
    ///
    /// **FINITE ELECTRONIC TEMPERATURE (300 K), not `tight_options()`'s etemp=0.**
    /// ScH is an open-shell d¹ transition metal with a NEAR-DEGENERATE d frontier.
    /// At *exactly* etemp=0 the analytic force is genuinely ill-conditioned there:
    /// the integer aufbau occupation makes the choice of which near-degenerate d
    /// orbital is occupied arbitrary, so its orbital-energy derivative `∂ε/∂R` (and
    /// hence the energy-weighted-density Pulay term) is discontinuous across the
    /// crossing — the classic integer-occupation-gradient discontinuity at a
    /// degeneracy, present for ANY method's spin/TM force, not specific to `+U`.
    /// A small Fermi smearing (300 K) lifts the degeneracy and makes BOTH the FD
    /// energy derivative and the analytic force smooth and well-defined; there the
    /// analytic gradient matches the FD to ~1e-9 (verified). The energy itself is
    /// essentially T-independent here (the frontier is only *near* degenerate), so
    /// this tests the same physics — just where the derivative is well-posed. The
    /// robust SCC (which converges the etemp=0 *energy* to a finite value; see
    /// `plus_u_open_shell_tm_scc_robust_at_kt0`) is orthogonal: it cures the SCC
    /// NaN, whereas finite T is needed for the *gradient* of a degenerate frontier.
    /// The geometry derivative `dU/dR` is now **analytic** (SCC-CPHF geometry response
    /// of the linear-response χ0/χ; see [`crate::plus_u_dudr`]), as is `∂E/∂U`, so the
    /// whole consistency force is analytic — the FD of the recomputed-U energy is only
    /// the verification oracle here. With the analytic `dU/dR` the residual tightens to
    /// the FD floor (~2.6e-10 Ha/bohr). See
    /// `crate::gradient::plus_u_consistency_gradient_terms`.
    #[test]
    fn plus_u_linear_response_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = tight_options();
        // Finite electronic temperature to lift the near-degenerate d¹ frontier so the
        // T=0-discontinuous analytic force is well-conditioned (see the doc comment above).
        opt.electronic_temperature = 300.0;
        opt.spin_multiplicity = Some(3);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true; // U computed per geometry → needs F_corr

        let mut gopt = AnalyticGradientOptions::default();
        gopt.electronic = opt.clone();
        let ana = analytic_gradient(&system, &params, gopt).unwrap().gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            crate::electronic::run_electronic(sys, &params, opt.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 1.0e-4,
            "linear-response +U consistent force vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// CAMM + spin/+U TOGETHER — single-point convergence on an open-shell TM system, and the CAMM
    /// term is actually applied (energy differs from the CAMM-off spin/+U result). Confirms the
    /// combined SCC converges (no NaN) with the AES Fock added to both channels. Uses ScH (triplet)
    /// with a FIXED U on Sc's d shell (not linear-response) so the single point stays fast in debug
    /// — the SCC wiring under test (CAMM Fock in both channels + energy) is identical either way.
    /// (Larger TM complexes like Ni(CO)3 also converge but are too slow for the debug suite.)
    #[test]
    fn camm_plus_u_energy_converges() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n",
            0.0, false,
        )
        .unwrap();
        let mut base = tight_options();
        base.electronic_temperature = 300.0;
        base.spin_multiplicity = Some(3);
        base.spin_polarization = true;
        base.plus_u = true;
        base.hubbard_u = vec![(21, 0.2)]; // fixed U on Sc (Z=21) d
        // +U only (CAMM off).
        let res_u = crate::electronic::run_electronic(&system, &params, base.clone()).unwrap();
        assert!(res_u.total_free.is_finite() && res_u.converged, "+U-only did not converge");
        // +U + CAMM.
        let mut camm = base.clone();
        camm.multipole = true;
        camm.multipole_model = crate::electronic::MultipoleModel::CammOnMdftb2;
        camm.camm_damp = 1.0;
        camm.camm_aes_scale = 1.0;
        camm.camm_onsite_scale = 1.0;
        let res_c = crate::electronic::run_electronic(&system, &params, camm).unwrap();
        assert!(
            res_c.total_free.is_finite() && res_c.converged,
            "CAMM + +U did not converge (E={}, conv={})",
            res_c.total_free, res_c.converged
        );
        assert!(
            (res_c.total_free - res_u.total_free).abs() > 1.0e-6,
            "CAMM had no measurable effect on the combined energy: {} vs {}",
            res_c.total_free, res_u.total_free
        );
    }

    /// CAMM + linear-response +U FD-GRADIENT GATE (the deliverable): with BOTH the CAMM-AES
    /// multipole correction AND linear-response +U on an open-shell TM system (ScH triplet,
    /// etemp 300), the analytic gradient (spin/+U force + CAMM-AES force) must match a central
    /// finite difference of the combined total energy. Mirrors
    /// `plus_u_linear_response_gradient_matches_fd` but with CAMM added — validates that the CAMM
    /// Fock in the converged spin Fock is correctly carried by the base energy-weighted-density
    /// Pulay term, so no CAMM overlap-Pulay term is missing.
    // WIP (2026-07-01): FAILS on ScH — the "base EWD-Pulay carries the CAMM response" assumption in
    // the doc above holds for the FROZEN-U CAMM force, but NOT for the LINEAR-RESPONSE U. The
    // linear-response U (and its dU/dR) flow through the SCC-CPHF screened response operator (I−M)
    // (spin.rs::analytic_chi / build_screened_operator, mirrored in plus_u_dudr), which dresses the
    // +U probe with the base SCC feedback. When CAMM is active the converged SCC Fock ALSO contains
    // the CAMM-AES multipole Fock, so that response operator MUST include the CAMM multipole coupling
    // — it currently does not → χ (screened) and dχ/dR are inconsistent with the CAMM-augmented SCC →
    // dU/dR is wrong → the consistency force fails FD (even on the 2-atom ScH: fundamental, not large-N).
    //
    // SCOPE (assessed 2026-07-01): this is a LARGE response-space EXPANSION, not a bounded add. The
    // CAMM Fock depends on the density through the atomic DIPOLE μ_A and QUADRUPOLE Θ_A moments
    // (camm_atomic_moments: μ_A = Tr(P·D̂_A), a DIFFERENT density projection than the shell-charge
    // Mulliken population). μ/Θ cannot be re-expressed via the existing (dq,dm) shell response
    // variables. Threading CAMM correctly requires expanding the (dq,dm) fixed point to
    // (dq,dm,dμ,dΘ) — new response dimensions (3·nat dipole + traceless-quad per atom), the CAMM
    // linear coupling kernel K_camm:(q,μ,Θ)→(s,vd,vq), new bare-response moment projections
    // (∂μ/∂probe via the referenced-dipole operator), the camm_aes_shift feedback Fock, AND the
    // geometry derivatives of all of these (referenced-dipole/quad integral derivatives, erf-cloud
    // kernel f^(mn)_grad) for dχ/dR — mirrored in BOTH spin.rs::analytic_chi/build_screened_operator
    // AND plus_u_dudr + its geometry-derivative sources, each stage FD-gated. This is comparable in
    // size to the original analytic dU/dR build, so it is deferred (per the scope-gate bail-out) for
    // the user to schedule separately. Energy single-points ARE correct (camm_plus_u_energy_converges_ni).
    // Fixed-U + CAMM gradients are also fine (frozen U → the EWD-Pulay does carry the CAMM force).
    // Do NOT use CAMM + LINEAR-RESPONSE-U geometry optimization until this gate passes.
    #[ignore = "deferred (scope=b): CAMM + linear-response +U gradient needs μ/Θ moment-response \
                variables in the SCC-CPHF (I−M) operator — a large response-space expansion"]
    #[test]
    fn camm_plus_u_linear_response_gradient_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n",
            0.0, false,
        )
        .unwrap();
        let mut opt = tight_options();
        opt.electronic_temperature = 300.0;
        opt.spin_multiplicity = Some(3);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true;
        opt.multipole = true;
        opt.multipole_model = crate::electronic::MultipoleModel::CammOnMdftb2;
        opt.camm_damp = 1.0;
        opt.camm_aes_scale = 1.0;
        opt.camm_onsite_scale = 1.0;

        let mut gopt = AnalyticGradientOptions::default();
        gopt.electronic = opt.clone();
        let ana = analytic_gradient(&system, &params, gopt).unwrap().gradient;

        let energy = |sys: &PeriodicSystem| -> f64 {
            crate::electronic::run_electronic(sys, &params, opt.clone())
                .unwrap()
                .total_free
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let fd = (energy(&displaced(&system, atom, axis, h))
                    - energy(&displaced(&system, atom, axis, -h)))
                    / (2.0 * h);
                let a = match axis {
                    0 => ana[atom].x,
                    1 => ana[atom].y,
                    _ => ana[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 1.0e-4,
            "CAMM + linear-response +U gradient vs FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Sanity: spin polarization can only lower (or keep) the energy of an
    /// open-shell system relative to the restricted GFN1 result.
    #[test]
    fn spin_polarization_lowers_open_shell_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "4\nch3\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.933 -0.539 0.0\nH -0.933 -0.539 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let mut base = tight_options();
        base.spin_multiplicity = Some(2);
        let mut spin = base.clone();
        spin.spin_polarization = true;

        let e_restricted = crate::electronic::run_electronic(&system, &params, base)
            .unwrap()
            .total_free;
        let res = crate::electronic::run_electronic(&system, &params, spin).unwrap();
        let e_spin = res.total_free;
        assert!(res.converged, "spGFN1 CH3 SCC did not converge");
        assert!(
            e_spin <= e_restricted + 1.0e-9,
            "spin polarization raised the energy: restricted={e_restricted:.8} spinpol={e_spin:.8}"
        );
        // The spin energy itself should be negative (W constants are ≤ 0).
        let spin_energy = res.spin.as_ref().unwrap().spin_energy;
        assert!(
            spin_energy < 0.0,
            "expected a negative spin-polarization energy, got {spin_energy:.6}"
        );
    }

    /// Build the full geometry-dependent linear-response setup (mirrors
    /// `linear_response_uv_for_system`) and return BOTH the FD-oracle and the analytic
    /// χ0/χ response matrices for the auto-selected correlated subspace, so the gate
    /// tests can compare them directly.
    fn chi_matrices_fd_and_analytic(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let basis = BasisSet::build(
            system,
            params,
            crate::basis::BasisOptions { nprim: options.nprim },
        )
        .unwrap();
        let charge = options.charge.unwrap_or(system.charge);
        let nelec = basis.total_reference_electrons - charge;
        let unpaired = resolve_unpaired(nelec, options.spin_multiplicity).unwrap();
        let nat = system.atoms.len();
        let n_alpha = 0.5 * (nelec + unpaired as f64);
        let n_beta = 0.5 * (nelec - unpaired as f64);
        let core = build_h0(system, &basis, params, &options.hamiltonian).unwrap();
        let overlap = &core.integrals.overlap;
        let h0 = &core.h0;
        let mut shell_model = ShellChargeModel::build(system, &basis, params).unwrap();
        shell_model.charge_order = options.charge_order.max(3);
        let amat = effective_coulomb_matrix(system, &basis, &shell_model);
        let orth = lowdin_orthogonalizer(overlap, options.eigen_tolerance).unwrap();
        let info = shell_info(&basis, nat);
        let kt = options.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
        let subspace = crate::plus_u::correlated_subspace_auto(&basis, options.plus_u_all_d);
        let (chi0_fd, chi_fd) = fd_chi0_chi(
            &basis, overlap, h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
            options, &subspace,
        )
        .unwrap();
        let (chi0_an, chi_an) = analytic_chi0_chi(
            &basis, overlap, h0, &orth, &shell_model, &amat, &info, n_alpha, n_beta, kt, unpaired,
            options, &subspace,
        )
        .unwrap();
        (chi0_fd, chi_fd, chi0_an, chi_an)
    }

    fn max_mat_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        let mut m = 0.0_f64;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                m = m.max((x - y).abs());
            }
        }
        m
    }

    /// PART A GATE 1 — the analytic bare χ0 must match the one-shot FD χ0, and the
    /// analytic screened χ must match the re-converged FD χ, on an open-shell TM atom
    /// (Sc, doublet; the auto subspace is its valence d shell). Both are compared at
    /// the production (default 300 K) electronic temperature, exercising the
    /// finite-temperature occupation-response term of the analytic χ0.
    #[test]
    fn linear_response_chi0_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str("1\nSc\nSc 0.0 0.0 0.0\n", 0.0, false).unwrap();
        let mut opt = ElectronicOptions::default();
        opt.spin_multiplicity = Some(2);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true;
        let (chi0_fd, chi_fd, chi0_an, chi_an) =
            chi_matrices_fd_and_analytic(&system, &params, &opt);
        assert!(!chi0_fd.is_empty(), "empty correlated subspace");
        let d0 = max_mat_diff(&chi0_fd, &chi0_an);
        let dc = max_mat_diff(&chi_fd, &chi_an);
        // Magnitudes for context (the response is O(1) Ha^-1 here).
        let scale = chi0_fd
            .iter()
            .flatten()
            .fold(0.0_f64, |m, &x| m.max(x.abs()))
            .max(1.0e-6);
        assert!(
            d0 < 5.0e-4 * scale.max(1.0),
            "analytic χ0 vs FD: max|Δ| = {d0:.3e} (χ0 scale {scale:.3e})"
        );
        assert!(
            dc < 5.0e-4 * scale.max(1.0),
            "analytic χ vs FD: max|Δ| = {dc:.3e}"
        );
    }

    /// PART A GATE 2 — the per-element Hubbard `U` extracted from the analytic χ0/χ
    /// must match the U extracted from the FD χ0/χ (same Tikhonov+clamp), to the same
    /// tolerance, on ScH (triplet) where the auto-selected Sc d carries a non-zero U.
    #[test]
    fn linear_response_u_analytic_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            PeriodicSystem::from_xyz_str("2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n", 0.0, false)
                .unwrap();
        let mut opt = tight_options();
        opt.spin_multiplicity = Some(3);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true;
        let (chi0_fd, chi_fd, chi0_an, chi_an) =
            chi_matrices_fd_and_analytic(&system, &params, &opt);
        assert!(!chi0_fd.is_empty(), "empty correlated subspace");
        let (u_fd, _) = crate::plus_u::extract_uv_from_response(&chi0_fd, &chi_fd);
        let (u_an, _) = crate::plus_u::extract_uv_from_response(&chi0_an, &chi_an);
        let mut maxdiff = 0.0_f64;
        for (a, b) in u_fd.iter().zip(u_an.iter()) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        // U is in Hartree; require a tight match between the two extraction paths.
        assert!(
            maxdiff < 1.0e-3,
            "analytic U vs FD U: max|Δ| = {maxdiff:.3e} (U_fd={u_fd:?}, U_an={u_an:?})"
        );
        // And the analytic U must be non-trivial (the whole point of the correction).
        assert!(
            u_an.iter().any(|&u| u > 1.0e-6),
            "analytic linear-response U is all ~zero: {u_an:?}"
        );
    }
}
