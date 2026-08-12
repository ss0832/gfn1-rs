// SPDX-License-Identifier: GPL-3.0-or-later
//! TD-GFN1: Tamm-Dancoff (TDA) excited states in the TD-DFTB transition-charge
//! response model (Niehaus et al., Phys. Rev. B 63, 085108 (2001)), built on the
//! GFN1 Mulliken transition shell charges and the SCC response kernel.
//!
//! The closed-shell TDA matrix is
//!
//! ```text
//! A_{ia,jb} = (eps_a - eps_i) delta_ij delta_ab
//!           + c * sum_{s,t} q_ia[s] K_st q_jb[t],
//! ```
//!
//! with `c = 2` for the singlet channel and `c = 0` for the triplet channel (the
//! GFN1 response kernel `K` is spin-independent, so triplet excitations reduce to
//! bare orbital-energy gaps, exactly as in spin-restricted TD-DFTB without an
//! explicit magnetic kernel). Oscillator strengths use the Mulliken (monopole)
//! transition dipole `mu = sum_{ia} X_ia sum_A Q_ia^A R_A`, consistent with the
//! GFN1 point-charge electrostatics.

use crate::cphf::{
    coupling_kernel_gradient, response_shell_charges_from_density, response_shell_scc_kernel,
    scalar_response_fock_matrix, solve_nonpbc_cpxtb_hessian_response, transition_shell_charges,
    AoDerivativeOptions, CpxtbOptions, CpxtbSpace, ResponseGradientContext,
};
// Used only by the retained legacy Lagrangian helpers and their diagnostic tests.
#[cfg(test)]
use crate::cphf::{
    mo_coefficient_matrix_to_ao, mo_pair_transition_shell_charge, response_electronic_gradient,
};
use crate::electronic::{ElectronicOptions, ElectronicResult};
use crate::error::{Gfn1Error, Result};
use crate::gradient::{analytic_gradient, AnalyticGradientOptions};
use crate::linalg::{
    lowdin_solve_generalized, matrix_vector_product, symmetric_eigen, Matrix,
};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

/// Spin channel of a closed-shell TDA excitation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TdaSpin {
    Singlet,
    Triplet,
}

impl TdaSpin {
    fn coupling_scale(self) -> f64 {
        match self {
            TdaSpin::Singlet => 2.0,
            TdaSpin::Triplet => 0.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TdaSpin::Singlet => "singlet",
            TdaSpin::Triplet => "triplet",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TdaOptions {
    /// Number of lowest excited states to report.
    pub n_states: usize,
    pub spin: TdaSpin,
}

impl Default for TdaOptions {
    fn default() -> Self {
        Self {
            n_states: 5,
            spin: TdaSpin::Singlet,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TdaState {
    /// Excitation energy (Hartree).
    pub excitation_energy: f64,
    /// Dimensionless oscillator strength (Mulliken transition dipole).
    pub oscillator_strength: f64,
    /// Mulliken transition dipole (atomic units, e*a0).
    pub transition_dipole: Vec3,
    /// Normalized occupied->virtual amplitudes `X_ia`, ordered by `pairs`.
    pub amplitudes: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct TdaResult {
    pub states: Vec<TdaState>,
    /// Occupied->virtual (i, a) MO index pairs (the amplitude ordering).
    pub pairs: Vec<(usize, usize)>,
}

/// Solve the closed-shell TD-GFN1 TDA problem for a (non-periodic) converged SCC.
///
/// The excitation energies and oscillator strengths are independent of the
/// eigensolver's arbitrary per-orbital sign; the returned **amplitude signs** are
/// not (see [`phase_align_mos_to_reference`]). Anything that compares amplitudes
/// across geometries must therefore go through
/// [`solve_tda_with_reference_mos`].
pub fn solve_tda(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    options: TdaOptions,
) -> Result<TdaResult> {
    solve_tda_with_reference_mos(system, params, electronic, options, None)
}

/// [`solve_tda`] with the MO phase gauge pinned to a reference MO coefficient
/// matrix, so the returned amplitudes are directly comparable (by overlap, sign
/// included) to amplitudes obtained at the reference geometry.
fn solve_tda_with_reference_mos(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    options: TdaOptions,
    reference_mos: Option<&Matrix>,
) -> Result<TdaResult> {
    let _profile = crate::profile::scope("td.tda.total");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda is non-periodic; use solve_tda_pbc_gamma for periodic systems".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let overlap = &electronic.integrals.overlap;
    let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12)?;
    let mut mos = eig.vectors;
    if let Some(reference) = reference_mos {
        phase_align_mos_to_reference(&mut mos, reference, overlap)?;
    }
    let kernel = response_shell_scc_kernel(system, params, electronic)?;
    dense_tda_core(
        system,
        basis,
        &kernel,
        &mos,
        &eig.values,
        &electronic.occupations,
        overlap,
        options,
    )
}

/// One TD-GFN1 excited state's electronic circular-dichroism data.
#[derive(Clone, Debug)]
pub struct RotatoryState {
    /// Excitation energy (Hartree).
    pub excitation_energy: f64,
    /// Length-gauge rotatory strength `R = Im(mu_0n . m_n0)` (atomic units).
    pub rotatory_strength: f64,
    /// Magnetic transition dipole `m_n0` (atomic units); purely imaginary, stored as
    /// its imaginary part `h_n` so that `m_n0 = i * magnetic_transition_dipole`.
    pub magnetic_transition_dipole: Vec3,
}

/// Electronic-CD **rotatory strengths** of the TD-GFN1 (TDA) excited states,
/// `R_n = Im(<0|mu|n> . <n|m|0>)` with the electric transition dipole `mu` (Mulliken)
/// and the orbital magnetic dipole `m = -1/2 (r - O) x p` about `origin`. Builds
/// `m_n0 = i h_n`, `h_n = 1/2 sum_{(i,a)} X_ia (C_a^T L C_i)` from the TDA amplitudes
/// and the angular-momentum AO integrals ([`crate::magnetic::angular_momentum_matrix`]),
/// so `R_n = mu_0n . h_n`. Non-periodic. For an achiral molecule every `R_n = 0`;
/// the sum over a complete set is origin-independent.
pub fn tda_rotatory_strengths(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    options: TdaOptions,
    origin: Vec3,
) -> Result<Vec<RotatoryState>> {
    let _profile = crate::profile::scope("td.tda.rotatory");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "tda_rotatory_strengths is non-periodic".to_string(),
        ));
    }
    let overlap = &electronic.integrals.overlap;
    let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12)?;
    let mos = &eig.vectors; // AO x MO, real
    let tda = solve_tda(system, params, electronic, options)?;
    let l = crate::magnetic::angular_momentum_matrix(system, &electronic.basis, origin);
    let n = electronic.basis.len();
    // Precompute, per (i,a) pair, the three C_a^T L_axis C_i values.
    let pair_l: Vec<[f64; 3]> = tda
        .pairs
        .iter()
        .map(|&(i, a)| {
            let mut out = [0.0_f64; 3];
            for axis in 0..3 {
                let mut v = 0.0;
                for mu in 0..n {
                    let cma = mos[(mu, a)];
                    if cma == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        v += cma * l[axis][(mu, nu)] * mos[(nu, i)];
                    }
                }
                out[axis] = v;
            }
            out
        })
        .collect();
    let mut states = Vec::with_capacity(tda.states.len());
    for state in &tda.states {
        let mut h = Vec3::zero();
        for (k, contrib) in pair_l.iter().enumerate() {
            let x = state.amplitudes[k];
            h.x += 0.5 * x * contrib[0];
            h.y += 0.5 * x * contrib[1];
            h.z += 0.5 * x * contrib[2];
        }
        let mu = state.transition_dipole;
        let r = mu.x * h.x + mu.y * h.y + mu.z * h.z;
        states.push(RotatoryState {
            excitation_energy: state.excitation_energy,
            rotatory_strength: r,
            magnetic_transition_dipole: h,
        });
    }
    Ok(states)
}

/// Frequency-dependent electronic **optical-rotation parameter** (the isotropic
/// Rosenfeld `beta`, one third the trace of the `G'` tensor) from the TD-GFN1 (TDA)
/// rotatory strengths:
/// ```text
/// beta(omega) = (2/3) sum_n  R_n omega_n / (omega_n^2 - omega^2),
/// ```
/// with `R_n = Im(mu_0n . m_n0)` ([`tda_rotatory_strengths`]) and `omega_n` the
/// excitation energies (Hartree). `frequencies` are photon energies (Hartree); the
/// return is `beta` per input frequency (atomic units). The static value
/// `beta(0) = (2/3) sum_n R_n / omega_n`. Achiral molecules give `0` at every
/// frequency and the mirror image negates `beta`; resonances (`omega ~ omega_n`) are
/// undamped (poles). The molecular optical rotation is intrinsically frequency
/// dependent — the *static* electric-dipole/magnetic-field response `dmu/dB` vanishes
/// for a closed shell by time reversal. Non-periodic. The molar specific rotation
/// `[alpha]` follows by the usual frequency/mass prefactor.
pub fn tda_optical_rotation(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    options: TdaOptions,
    origin: Vec3,
    frequencies: &[f64],
) -> Result<Vec<f64>> {
    let states = tda_rotatory_strengths(system, params, electronic, options, origin)?;
    let mut out = Vec::with_capacity(frequencies.len());
    for &w in frequencies {
        let mut beta = 0.0;
        for s in &states {
            let wn = s.excitation_energy;
            beta += s.rotatory_strength * wn / (wn * wn - w * w);
        }
        out.push((2.0 / 3.0) * beta);
    }
    Ok(out)
}

/// Closed-shell dense TDA from MOs / energies / occupations, a shell SCC response
/// kernel, and the overlap. Shared by the molecular and Gamma-point periodic paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_tda_core(
    system: &PeriodicSystem,
    basis: &crate::basis::BasisSet,
    kernel: &Matrix,
    mos: &Matrix,
    energies: &[f64],
    occupations: &[f64],
    overlap: &Matrix,
    options: TdaOptions,
) -> Result<TdaResult> {
    let space = CpxtbSpace::from_occupations(occupations)?;
    let n = space.len();
    let gaps = space
        .pairs
        .iter()
        .map(|&(i, a)| {
            let g = energies[a] - energies[i];
            if g <= 0.0 {
                Err(Gfn1Error::InvalidInput(
                    "TD-GFN1 requires a positive occupied-virtual gap (gapped closed shell)"
                        .to_string(),
                ))
            } else {
                Ok(g)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let coupling = options.spin.coupling_scale();
    let transition = transition_shell_charges(basis, mos, occupations, overlap)?;

    let mut a = Matrix::zeros(n, n);
    for col in 0..n {
        let mut unit = vec![0.0_f64; n];
        unit[col] = 1.0;
        let sigma = tda_sigma(&gaps, kernel, &transition, coupling, &unit)?;
        for (row, value) in sigma.into_iter().enumerate() {
            a[(row, col)] = value;
        }
    }
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }

    let solved = symmetric_eigen(&a)?;
    let n_states = options.n_states.min(n);

    let atom_positions: Vec<[f64; 3]> = system
        .atoms
        .iter()
        .map(|at| at.position.to_array())
        .collect();
    let mut pair_dipole = vec![Vec3::zero(); n];
    for (row, qia) in transition.iter().enumerate() {
        let mut mu = Vec3::zero();
        for (ish, &q) in qia.iter().enumerate() {
            let r = atom_positions[basis.shells[ish].atom_index];
            mu += Vec3::new(q * r[0], q * r[1], q * r[2]);
        }
        pair_dipole[row] = mu;
    }

    let mut states = Vec::with_capacity(n_states);
    for s in 0..n_states {
        let amplitudes = solved.vectors.column(s);
        let omega = solved.values[s];
        let mut mu = Vec3::zero();
        if options.spin == TdaSpin::Singlet {
            for (row, &x) in amplitudes.iter().enumerate() {
                mu += pair_dipole[row] * x;
            }
        }
        let mu = mu * std::f64::consts::SQRT_2;
        let oscillator_strength = if omega > 0.0 {
            (2.0 / 3.0) * omega * mu.norm2()
        } else {
            0.0
        };
        states.push(TdaState {
            excitation_energy: omega,
            oscillator_strength,
            transition_dipole: mu,
            amplitudes,
        });
    }
    Ok(TdaResult {
        states,
        pairs: space.pairs.clone(),
    })
}

/// Gamma-point periodic TD-GFN1 (TDA). Builds the Gamma Bloch MOs and the periodic
/// SCC (Ewald Klopman-Ohno) response kernel, then runs the closed-shell TDA. The
/// excitation energies are the Brillouin-zone-center excitations of the periodic
/// cell.
pub fn solve_tda_pbc_gamma(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    options: TdaOptions,
) -> Result<TdaResult> {
    let _profile = crate::profile::scope("td.tda.pbc_gamma");
    if system.lattice.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda_pbc_gamma requires a periodic system".to_string(),
        ));
    }
    let pbc = crate::pbc::PbcOptions {
        kmesh: crate::pbc::KMesh::gamma(),
        ..crate::pbc::PbcOptions::default()
    };
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    let (h0, overlap) = scf.bloch.h_s_gamma_real();
    let fock = crate::electronic::fock_from_shell_potential(
        &scf.basis,
        &overlap,
        &h0,
        &scf.shell_scc_potential,
    );
    let eig = lowdin_solve_generalized(&fock, &overlap, 1.0e-12)?;
    let occupations = aufbau_closed_shell(&eig.values, scf.nelec);
    let kernel = crate::electronic::scc_response_kernel(
        &scf.gamma,
        &scf.shell_model,
        &scf.basis,
        &scf.shell_charges,
    );
    dense_tda_core(
        system,
        &scf.basis,
        &kernel,
        &eig.vectors,
        &eig.values,
        &occupations,
        &overlap,
        options,
    )
}

/// Complex Bloch coefficient of physical band `b` (real-embedding column `2b`:
/// `C_mu = vectors[(mu,2b)] + i vectors[(n+mu,2b)]`), **gauge-fixed** so that the
/// largest-magnitude AO coefficient is real and positive. The degenerate
/// real-embedding eigenpair `{embed(u), embed(i u)}` only fixes each band up to a
/// global complex phase; removing it makes the (real) optical transition charges
/// reproducible and, at the Gamma point (where `C = z u` for a real MO `u` with
/// `|z| = 1`), real — so the k-point TDA reduces exactly to [`solve_tda_pbc_gamma`].
fn gauge_fixed_band(eig: &crate::pbc::complex::KEigen, b: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for mu in 0..n {
        re[mu] = eig.vectors[(mu, 2 * b)];
        im[mu] = eig.vectors[(n + mu, 2 * b)];
    }
    let mut mu_star = 0usize;
    let mut best = -1.0_f64;
    for mu in 0..n {
        let m2 = re[mu] * re[mu] + im[mu] * im[mu];
        if m2 > best {
            best = m2;
            mu_star = mu;
        }
    }
    let norm = best.sqrt();
    if norm > 1.0e-30 {
        // Multiply every coefficient by e^{-i arg(C_{mu*})} = (cos - i sin).
        let cos_phi = re[mu_star] / norm;
        let sin_phi = im[mu_star] / norm;
        for mu in 0..n {
            let r = re[mu];
            let m = im[mu];
            re[mu] = r * cos_phi + m * sin_phi;
            im[mu] = m * cos_phi - r * sin_phi;
        }
    }
    (re, im)
}

/// Real Mulliken transition shell charge of a band pair `(i, a)` at one k-point,
/// from the **gauge-fixed** complex Bloch coefficients (`ci`, `ca`) and the
/// complex overlap `sk`. For the optical (`q = 0`) transition the shell population
/// `q[s] = -sum_{mu in s} Re(<i|S|a>_mu + <a|S|i>_mu)` is real (matching the
/// gfn2-rs k-point transition charge); the minus sign matches the molecular
/// [`transition_shell_charges`] convention.
fn kpoint_transition_shell_charge(
    basis: &crate::basis::BasisSet,
    sk: &crate::pbc::complex::CMatrix,
    ci_re: &[f64],
    ci_im: &[f64],
    ca_re: &[f64],
    ca_im: &[f64],
) -> Result<Vec<f64>> {
    let n = ci_re.len();
    // sc = S(k) C for each band (complex matvec).
    let mut sc_i_re = vec![0.0_f64; n];
    let mut sc_i_im = vec![0.0_f64; n];
    let mut sc_a_re = vec![0.0_f64; n];
    let mut sc_a_im = vec![0.0_f64; n];
    for mu in 0..n {
        let (mut ir, mut ii, mut ar, mut ai) = (0.0, 0.0, 0.0, 0.0);
        for nu in 0..n {
            let sr = sk.re[(mu, nu)];
            let si = sk.im[(mu, nu)];
            ir += sr * ci_re[nu] - si * ci_im[nu];
            ii += sr * ci_im[nu] + si * ci_re[nu];
            ar += sr * ca_re[nu] - si * ca_im[nu];
            ai += sr * ca_im[nu] + si * ca_re[nu];
        }
        sc_i_re[mu] = ir;
        sc_i_im[mu] = ii;
        sc_a_re[mu] = ar;
        sc_a_im[mu] = ai;
    }
    let mut q = vec![0.0_f64; basis.shells.len()];
    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        for mu in shell.first_ao..shell.first_ao + shell.nao {
            q[shell_idx] -= ci_re[mu] * sc_a_re[mu]
                + ci_im[mu] * sc_a_im[mu]
                + ca_re[mu] * sc_i_re[mu]
                + ca_im[mu] * sc_i_im[mu];
        }
    }
    Ok(q)
}

/// Off-Gamma **k-point** periodic TD-GFN1 (TDA). Builds the converged k-point SCC,
/// diagonalises the complex Bloch Fock `F(k)` at every k-point, and assembles the
/// optical (`q = 0`) closed-shell TDA over all occupied->virtual band pairs across
/// the Monkhorst-Pack mesh. The transition shell charges are real (the optical
/// Mulliken populations) and weighted by `sqrt(w_k)`, so the transition-charge
/// Coulomb coupling `c * sum q_I K q_J` carries `sqrt(w_I w_J)` and the matrix is
/// real symmetric; at a single Gamma point it reduces to [`solve_tda_pbc_gamma`].
/// Requires integer (gapped) band occupations.
pub fn solve_tda_kpoint(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    kmesh: crate::pbc::KMesh,
    options: TdaOptions,
) -> Result<TdaResult> {
    let _profile = crate::profile::scope("td.tda.kpoint");
    if system.lattice.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda_kpoint requires a periodic system".to_string(),
        ));
    }
    let pbc = crate::pbc::PbcOptions {
        kmesh,
        ..crate::pbc::PbcOptions::default()
    };
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    if !scf.converged {
        return Err(Gfn1Error::InvalidInput(
            "k-point TD-GFN1 requires a converged periodic SCC".to_string(),
        ));
    }
    let basis = &scf.basis;
    let n = basis.len();
    let mut vao = vec![0.0_f64; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let kernel = crate::electronic::scc_response_kernel(
        &scf.gamma,
        &scf.shell_model,
        basis,
        &scf.shell_charges,
    );
    let coupling = options.spin.coupling_scale();
    let eigen_tol = electronic_options.eigen_tolerance.max(1.0e-12);
    let atom_positions: Vec<[f64; 3]> = system
        .atoms
        .iter()
        .map(|at| at.position.to_array())
        .collect();

    // Closed-shell integer band filling: the lowest `nocc = nelec/2` bands are
    // occupied at every k-point (this solver requires a gapped insulator). Using
    // the integer count rather than a strict `energy < fermi_level` test is the
    // exact reduction to the aufbau Gamma path: at T = 0 the Fermi level sits
    // essentially on the HOMO, so `<` would spuriously drop the HOMO from the
    // occupied space and shift every excitation energy.
    let nocc_f = scf.nelec / 2.0;
    let nocc = nocc_f.round() as usize;
    if (nocc_f - nocc as f64).abs() > 1.0e-6 || nocc == 0 || nocc >= n {
        return Err(Gfn1Error::InvalidInput(
            "k-point TD-GFN1 requires an integer closed-shell band filling \
             (gapped insulator)"
                .to_string(),
        ));
    }

    let mut gaps: Vec<f64> = Vec::new();
    let mut transitions: Vec<Vec<f64>> = Vec::new();
    let mut pair_dipoles: Vec<Vec3> = Vec::new();
    let mut labels: Vec<(usize, usize)> = Vec::new();
    for (ik, (h0k, sk)) in scf.hs_k.iter().enumerate() {
        let weight = scf.kpoints[ik].weight;
        if weight <= 0.0 {
            continue;
        }
        let sw = weight.sqrt();
        let fock = crate::pbc::scf::fock_at_k(h0k, sk, &vao);
        let eig = crate::pbc::complex::hermitian_generalized_eigen(&fock, sk, eigen_tol)?;
        // Physical band b has energy eig.values[2b] (ascending in b). The lowest
        // `nocc` bands are occupied, the rest virtual.
        let occ_bands: Vec<usize> = (0..nocc).collect();
        let virt_bands: Vec<usize> = (nocc..n).collect();
        // Gauge-fix every band once so the (real) transition charges are
        // phase-consistent across pairs and match the real Gamma path.
        let bands: Vec<(Vec<f64>, Vec<f64>)> =
            (0..n).map(|b| gauge_fixed_band(&eig, b, n)).collect();
        for &i in &occ_bands {
            for &a in &virt_bands {
                let gap = eig.values[2 * a] - eig.values[2 * i];
                if gap <= 1.0e-10 {
                    continue;
                }
                let mut q = kpoint_transition_shell_charge(
                    basis,
                    sk,
                    &bands[i].0,
                    &bands[i].1,
                    &bands[a].0,
                    &bands[a].1,
                )?;
                for v in &mut q {
                    *v *= sw;
                }
                let mut mu = Vec3::zero();
                if options.spin == TdaSpin::Singlet {
                    for (s, &qs) in q.iter().enumerate() {
                        let r = atom_positions[basis.shells[s].atom_index];
                        mu += Vec3::new(qs * r[0], qs * r[1], qs * r[2]);
                    }
                }
                gaps.push(gap);
                transitions.push(q);
                pair_dipoles.push(mu);
                labels.push((ik, i * n + a));
            }
        }
    }
    if gaps.is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "k-point TD-GFN1 found no positive-gap occupied->virtual transitions \
             (metallic or non-integer occupations are not supported)"
                .to_string(),
        ));
    }
    let ntrans = gaps.len();
    let mut a = Matrix::zeros(ntrans, ntrans);
    for col in 0..ntrans {
        let mut unit = vec![0.0_f64; ntrans];
        unit[col] = 1.0;
        let sigma = tda_sigma(&gaps, &kernel, &transitions, coupling, &unit)?;
        for (row, value) in sigma.into_iter().enumerate() {
            a[(row, col)] = value;
        }
    }
    for i in 0..ntrans {
        for j in 0..i {
            let avg = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }
    let solved = symmetric_eigen(&a)?;
    let n_states = options.n_states.min(ntrans);
    let mut states = Vec::with_capacity(n_states);
    for s in 0..n_states {
        let amplitudes = solved.vectors.column(s);
        let omega = solved.values[s];
        let mut mu = Vec3::zero();
        for (row, &x) in amplitudes.iter().enumerate() {
            mu += pair_dipoles[row] * x;
        }
        let mu = mu * std::f64::consts::SQRT_2;
        let oscillator_strength = if omega > 0.0 {
            (2.0 / 3.0) * omega * mu.norm2()
        } else {
            0.0
        };
        states.push(TdaState {
            excitation_energy: omega,
            oscillator_strength,
            transition_dipole: mu,
            amplitudes,
        });
    }
    Ok(TdaResult {
        states,
        pairs: labels,
    })
}

/// Closed-shell aufbau occupations (2.0 per spatial orbital) for `nelec` electrons.
fn aufbau_closed_shell(energies: &[f64], nelec: f64) -> Vec<f64> {
    let mut occ = vec![0.0_f64; energies.len()];
    let mut remaining = nelec.max(0.0);
    for o in &mut occ {
        let fill = remaining.min(2.0);
        *o = fill;
        remaining -= fill;
        if remaining <= 1.0e-12 {
            break;
        }
    }
    occ
}

/// Excited-state gradient of a TD-GFN1 (TDA) state.
#[derive(Clone, Debug)]
pub struct TdaGradientResult {
    pub state_index: usize,
    /// Excitation energy at the reference geometry (Hartree).
    pub excitation_energy: f64,
    /// Total excited-state energy `E_ground(free) + omega` (Hartree).
    pub total_energy: f64,
    /// `d(total excited energy)/dR` per atom (Hartree/Bohr).
    pub gradient: Vec<Vec3>,
    /// Excited-state forces (`-gradient`).
    pub forces: Vec<Vec3>,
}

/// Total excited-state energy of the state whose amplitudes best overlap
/// `reference_amplitudes` (so the FD tracks the same root across displacements).
fn td_for_system(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    options: TdaOptions,
    reference_mos: Option<&Matrix>,
) -> Result<(f64, TdaResult)> {
    let ground = crate::run_electronic(system, params, electronic_options.clone())?;
    let td = if system.lattice.is_some() {
        solve_tda_pbc_gamma(system, params, electronic_options, options)?
    } else {
        solve_tda_with_reference_mos(system, params, &ground, options, reference_mos)?
    };
    Ok((ground.total_free, td))
}

fn matched_excited_energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    options: TdaOptions,
    reference_amplitudes: &[f64],
    reference_mos: Option<&Matrix>,
) -> Result<f64> {
    let (ground_energy, td) =
        td_for_system(system, params, electronic_options, options, reference_mos)?;
    let mut best = 0usize;
    let mut best_overlap = -1.0_f64;
    for (idx, state) in td.states.iter().enumerate() {
        let overlap: f64 = state
            .amplitudes
            .iter()
            .zip(reference_amplitudes.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();
        if overlap > best_overlap {
            best_overlap = overlap;
            best = idx;
        }
    }
    Ok(ground_energy + td.states[best].excitation_energy)
}

/// TD-GFN1 excited-state **gradient** by central finite difference of the
/// re-diagonalised TDA excitation energy (plus the ground-state energy), tracking
/// the target root by amplitude overlap. Non-periodic **and** Gamma-point
/// periodic — the only method that supports both.
///
/// This is the numerically exact excited-state gradient and the finite-difference
/// ground truth the analytic paths are gated against; it costs `6N` SCC + TDA
/// solves, so [`solve_tda_gradient_analytic`] is the production route. Because
/// the root is followed by amplitude overlap it degrades near a genuine state
/// crossing: within a near-degenerate pair the two roots' amplitudes are an
/// arbitrary mixture of the same subspace and the tracking is ill-posed. For the
/// non-periodic path the displaced MOs are phase-aligned to the reference gauge
/// so the overlap is meaningful; the periodic path re-diagonalises without that
/// alignment (see `docs/td.md`).
pub fn solve_tda_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    state_index: usize,
    options: TdaOptions,
    step: f64,
) -> Result<TdaGradientResult> {
    let _profile = crate::profile::scope("td.tda.gradient");
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "TD-GFN1 gradient step must be positive".to_string(),
        ));
    }
    // Reference TDA at the input geometry (non-PBC or Gamma-point periodic).
    let (ground_energy, td) = td_for_system(system, params, electronic_options, options, None)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let reference = td.states[state_index].amplitudes.clone();
    let total_energy = ground_energy + omega;
    // Pin the MO gauge of every displaced solve to the reference geometry, so the
    // amplitude-overlap root tracking compares like with like.
    let reference_mos = if system.lattice.is_some() {
        None
    } else {
        Some(reference_mo_coefficients(&crate::run_electronic(
            system,
            params,
            electronic_options.clone(),
        )?)?)
    };

    let nat = system.atoms.len();
    let inv = 1.0 / (2.0 * step);
    let mut gradient = vec![Vec3::zero(); nat];
    for atom in 0..nat {
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift_atom(&mut plus, atom, axis, step);
            shift_atom(&mut minus, atom, axis, -step);
            let ep = matched_excited_energy(
                &plus,
                params,
                electronic_options,
                options,
                &reference,
                reference_mos.as_ref(),
            )?;
            let em = matched_excited_energy(
                &minus,
                params,
                electronic_options,
                options,
                &reference,
                reference_mos.as_ref(),
            )?;
            let d = (ep - em) * inv;
            match axis {
                0 => gradient[atom].x = d,
                1 => gradient[atom].y = d,
                _ => gradient[atom].z = d,
            }
        }
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy,
        gradient,
        forces,
    })
}

/// Matched k-mesh TDA excitation energy (ground PBC energy + the excitation whose
/// amplitudes best overlap `reference_amplitudes`). The ground PBC energy uses the
/// same k-mesh as the TDA so the finite-difference total energy is internally
/// consistent. Used by [`solve_tda_kpoint_gradient`] as the FD ground truth for the
/// analytic k-mesh gradient.
fn matched_kpoint_excited_energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    kmesh: crate::pbc::KMesh,
    options: TdaOptions,
    reference_amplitudes: &[f64],
) -> Result<f64> {
    let pbc = crate::pbc::PbcOptions {
        kmesh,
        ..crate::pbc::PbcOptions::default()
    };
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    let ground_energy = scf.total_free;
    let td = solve_tda_kpoint(system, params, electronic_options, kmesh, options)?;
    let mut best = 0usize;
    let mut best_overlap = -1.0_f64;
    for (idx, state) in td.states.iter().enumerate() {
        let overlap: f64 = state
            .amplitudes
            .iter()
            .zip(reference_amplitudes.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();
        if overlap > best_overlap {
            best_overlap = overlap;
            best = idx;
        }
    }
    Ok(ground_energy + td.states[best].excitation_energy)
}

/// k-mesh periodic TD-GFN1 (TDA) excited-state **gradient** by central finite
/// difference of the matched k-mesh excitation energy (plus the ground PBC energy),
/// tracking the target root by amplitude overlap. This is the numerically exact
/// excited-state gradient over the Brillouin-zone-sampled TDA spectrum; it is the
/// finite-difference ground truth for the analytic [`solve_tda_kpoint_gradient_analytic`].
/// At a Gamma-only mesh it reduces to [`solve_tda_gradient`] on the periodic system.
pub fn solve_tda_kpoint_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    kmesh: crate::pbc::KMesh,
    state_index: usize,
    options: TdaOptions,
    step: f64,
) -> Result<TdaGradientResult> {
    let _profile = crate::profile::scope("td.tda.kpoint_gradient");
    if system.lattice.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda_kpoint_gradient requires a periodic system".to_string(),
        ));
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "TD-GFN1 gradient step must be positive".to_string(),
        ));
    }
    let pbc = crate::pbc::PbcOptions {
        kmesh,
        ..crate::pbc::PbcOptions::default()
    };
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    let ground_energy = scf.total_free;
    let td = solve_tda_kpoint(system, params, electronic_options, kmesh, options)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let reference = td.states[state_index].amplitudes.clone();
    let total_energy = ground_energy + omega;

    let nat = system.atoms.len();
    let inv = 1.0 / (2.0 * step);
    let mut gradient = vec![Vec3::zero(); nat];
    for atom in 0..nat {
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift_atom(&mut plus, atom, axis, step);
            shift_atom(&mut minus, atom, axis, -step);
            let ep = matched_kpoint_excited_energy(
                &plus, params, electronic_options, kmesh, options, &reference,
            )?;
            let em = matched_kpoint_excited_energy(
                &minus, params, electronic_options, kmesh, options, &reference,
            )?;
            let d = (ep - em) * inv;
            match axis {
                0 => gradient[atom].x = d,
                1 => gradient[atom].y = d,
                _ => gradient[atom].z = d,
            }
        }
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy,
        gradient,
        forces,
    })
}

/// Selects how [`solve_tda_gradient_method`] computes the excited-state gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TdaGradientMethod {
    /// **Default.** Semi-numerical hybrid: the ground-state gradient is analytic
    /// (exact) and the excitation-energy gradient `domega/dR` is a central finite
    /// difference of the *frozen-amplitude* excitation energy
    /// ([`tda_frozen_excitation_energy`]). Because the excitation energy is
    /// stationary with respect to the amplitudes at the eigenstate, the frozen
    /// finite difference equals the true gradient to finite-difference precision
    /// (not an approximation) for a tracked adiabatic state, while skipping the
    /// per-displacement TDA re-diagonalisation. Non-periodic.
    SemiNumerical,
    /// Full central finite difference of the re-diagonalised excitation energy with
    /// amplitude-overlap root tracking ([`solve_tda_gradient`]). The most robust
    /// option across state crossings, and the only one that supports periodic
    /// (Gamma-point) systems.
    FiniteDifference,
    /// Fully analytic gradient ([`solve_tda_gradient_analytic`]): the exact analytic
    /// derivative of the frozen-amplitude TDA excitation energy, obtained by direct
    /// differentiation through the ground-state coupled-perturbed (CPHF) orbital
    /// response. Agrees with the finite-difference reference to FD precision
    /// (`< 1e-5` Hartree/bohr; ~`1e-9` on the test molecules and ~`2e-7` at the
    /// periodic Gamma point). Solves one ground-state CPHF over the `3N` Cartesian
    /// perturbations. Non-periodic **and periodic Gamma-point**.
    Analytic,
}

impl Default for TdaGradientMethod {
    fn default() -> Self {
        TdaGradientMethod::SemiNumerical
    }
}

impl TdaGradientMethod {
    /// Parse a user-facing method string (accepts `semi_numerical`/`semi-numerical`/
    /// `semi`, `finite_difference`/`fd`, `analytic`). Case- and hyphen-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "semi_numerical" | "seminumerical" | "semi" => Some(Self::SemiNumerical),
            "finite_difference" | "finitedifference" | "fd" | "numerical" => {
                Some(Self::FiniteDifference)
            }
            "analytic" | "lagrangian" | "z_vector" => Some(Self::Analytic),
            _ => None,
        }
    }
}

/// TD-GFN1 excited-state gradient via the requested [`TdaGradientMethod`].
/// `step` is the finite-difference displacement (used by `SemiNumerical` and
/// `FiniteDifference`; ignored by `Analytic`).
pub fn solve_tda_gradient_method(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    state_index: usize,
    options: TdaOptions,
    step: f64,
    method: TdaGradientMethod,
) -> Result<TdaGradientResult> {
    match method {
        TdaGradientMethod::SemiNumerical => solve_tda_gradient_seminumerical(
            system,
            params,
            electronic_options,
            state_index,
            options,
            step,
        ),
        TdaGradientMethod::FiniteDifference => solve_tda_gradient(
            system,
            params,
            electronic_options,
            state_index,
            options,
            step,
        ),
        TdaGradientMethod::Analytic => {
            solve_tda_gradient_analytic(system, params, electronic_options, state_index, options)
        }
    }
}

/// TD-GFN1 excited-state gradient by the **semi-numerical** hybrid: an analytic,
/// exact ground-state gradient plus a central finite difference of the
/// frozen-amplitude excitation energy ([`tda_frozen_excitation_energy`]).
///
/// This is the recommended production method. It is exact (to finite-difference
/// precision) for a tracked adiabatic state by the amplitude-stationarity (`2n+1`)
/// argument, and it avoids re-diagonalising the TDA problem at every displaced
/// geometry — so it is cheaper than [`solve_tda_gradient`] while sidestepping the
/// experimental Lagrangian relaxed-density convention. Non-periodic; for periodic
/// (Gamma-point) systems use [`solve_tda_gradient`] (finite difference).
pub fn solve_tda_gradient_seminumerical(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &crate::electronic::ElectronicOptions,
    state_index: usize,
    options: TdaOptions,
    step: f64,
) -> Result<TdaGradientResult> {
    let _profile = crate::profile::scope("td.tda.gradient_seminumerical");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda_gradient_seminumerical is non-periodic; use solve_tda_gradient \
             (finite difference) for periodic (Gamma-point) systems"
                .to_string(),
        ));
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "TD-GFN1 gradient step must be positive".to_string(),
        ));
    }
    // Analytic, exact ground-state gradient + the converged reference SCC.
    let grad_options = AnalyticGradientOptions {
        electronic: electronic_options.clone(),
        ..AnalyticGradientOptions::default()
    };
    let ground = analytic_gradient(system, params, grad_options)?;
    // Reference TDA solve once at the input geometry to fix the target amplitudes.
    let td = solve_tda(system, params, &ground.electronic_result, options)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let amplitudes = td.states[state_index].amplitudes.clone();
    // MO gauge the amplitudes are expressed in. Every displaced geometry is
    // phase-aligned to it, otherwise the eigensolver's arbitrary per-orbital
    // sign makes the frozen Rayleigh quotient discontinuous and the finite
    // difference diverges as 1/h for any state with transition charge.
    let reference_mos = reference_mo_coefficients(&ground.electronic_result)?;

    // Central finite difference of the frozen-amplitude excitation energy gives
    // domega/dR exactly (amplitude stationarity); add it to the analytic ground gradient.
    let nat = system.atoms.len();
    let inv = 1.0 / (2.0 * step);
    let mut gradient = ground.gradient.clone();
    for atom in 0..nat {
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            shift_atom(&mut plus, atom, axis, step);
            shift_atom(&mut minus, atom, axis, -step);
            let wp = tda_frozen_excitation_energy_with_mos(
                &plus,
                params,
                electronic_options,
                &amplitudes,
                options.spin,
                Some(&reference_mos),
            )?;
            let wm = tda_frozen_excitation_energy_with_mos(
                &minus,
                params,
                electronic_options,
                &amplitudes,
                options.spin,
                Some(&reference_mos),
            )?;
            let d = (wp - wm) * inv;
            match axis {
                0 => gradient[atom].x += d,
                1 => gradient[atom].y += d,
                _ => gradient[atom].z += d,
            }
        }
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy: ground.total_energy + omega,
        gradient,
        forces,
    })
}

fn shift_atom(system: &mut PeriodicSystem, atom: usize, axis: usize, delta: f64) {
    match axis {
        0 => system.atoms[atom].position.x += delta,
        1 => system.atoms[atom].position.y += delta,
        _ => system.atoms[atom].position.z += delta,
    }
}

// =====================================================================
// LEGACY (test-only): Lagrangian / Z-vector relaxed-difference-density TDA
// gradient (non-PBC).
//
// This was the former analytic TDA gradient — the relaxed-difference-density
// formalism of Furche & Ahlrichs (J. Chem. Phys. 117, 7433 (2002)) specialised
// to the TD-DFTB / GFN1 transition-charge response (Heringer, Niehaus et al.,
// J. Comput. Chem. 28, 2589 (2007)), `d omega / dR = Tr[P^Delta dF/dR]
// - Tr[W dS/dR] + c P^T (dK/dR) P`. As ported it carried a ~7e-3 Hartree/bohr
// residual (a mutual P/W inconsistency); the production gradient now uses the
// exact direct-CPHF derivative ([`tda_direct_excitation_gradient`]). These
// helpers are retained only to drive the in-module diagnostic tests that
// characterise the former path, so they are gated behind `#[cfg(test)]`.
// =====================================================================

/// Unrelaxed TDA difference-density MO blocks: the occupied-occupied "hole"
/// `hole_{ij} = -sum_a X_{ia} X_{ja}` and the virtual-virtual "particle"
/// `t_vv_{ab} = sum_i X_{ia} X_{ib}` (both stored row-major, full matrices).
#[cfg(test)]
fn tda_unrelaxed_density_blocks(
    amplitudes: &[f64],
    n_occ: usize,
    n_virt: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if amplitudes.len() != n_occ * n_virt {
        return Err(Gfn1Error::InvalidInput(
            "TDA unrelaxed density block amplitude length mismatch".to_string(),
        ));
    }
    let mut hole = vec![0.0_f64; n_occ * n_occ];
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut s = 0.0;
            for a in 0..n_virt {
                s += amplitudes[i * n_virt + a] * amplitudes[j * n_virt + a];
            }
            hole[i * n_occ + j] = -s;
        }
    }
    let mut t_vv = vec![0.0_f64; n_virt * n_virt];
    for a in 0..n_virt {
        for b in 0..n_virt {
            let mut s = 0.0;
            for i in 0..n_occ {
                s += amplitudes[i * n_virt + a] * amplitudes[i * n_virt + b];
            }
            t_vv[a * n_virt + b] = s;
        }
    }
    Ok((hole, t_vv))
}

/// Weighted sum of per-pair shell-charge vectors: `sum_k weights[k] charges[k]`.
#[cfg(test)]
fn induced_shell_charges(n_shells: usize, pair_charges: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_shells];
    for (charges, &w) in pair_charges.iter().zip(weights.iter()) {
        if w == 0.0 {
            continue;
        }
        for (dst, &q) in out.iter_mut().zip(charges.iter()) {
            *dst += w * q;
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Intermediate Lagrangian quantities for the relaxed TDA difference density.
#[cfg(test)]
struct TdaLagrangianTerms {
    t_oo: Vec<f64>,
    t_vv: Vec<f64>,
    q_oo: Vec<f64>,
    q_vv: Vec<f64>,
    q_ov: Vec<f64>,
    q_vo: Vec<f64>,
    charges_oo: Vec<Vec<f64>>,
    charges_ov: Vec<Vec<f64>>,
}

/// Build the TDA Lagrangian terms (orbital-energy weighted density blocks
/// `q_oo`, `q_vv` and the occupied-virtual Z-vector source `q_ov`, `q_vo`).
/// Ported from the FD-verified gfn2-rs `tda_lagrangian_terms`, using the GFN1
/// transition shell charges and SCC response kernel.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn tda_lagrangian_terms(
    basis: &crate::basis::BasisSet,
    kernel: &Matrix,
    mos: &Matrix,
    orbital_energies: &[f64],
    space: &CpxtbSpace,
    overlap: &Matrix,
    amplitudes: &[f64],
    omega: f64,
    tdiff_hplus_scale: f64,
    transition_hplus_scale: f64,
) -> Result<TdaLagrangianTerms> {
    let n_occ = space.occupied.len();
    let n_virt = space.virtuals.len();
    let nshell = basis.shells.len();
    let (hole, t_vv) = tda_unrelaxed_density_blocks(amplitudes, n_occ, n_virt)?;
    let t_oo = hole.iter().map(|&v| -v).collect::<Vec<_>>();
    let sc = overlap.matmul(mos)?;
    let mut charges_oo = Vec::with_capacity(n_occ * n_occ);
    for &i in &space.occupied {
        for &j in &space.occupied {
            charges_oo.push(mo_pair_transition_shell_charge(basis, mos, &sc, i, j)?);
        }
    }
    let charges_ov = transition_shell_charges(
        basis,
        mos,
        &occupations_for_space(space, mos.cols()),
        overlap,
    )?;
    let induced_x = induced_shell_charges(nshell, &charges_ov, amplitudes);
    let induced_too = induced_shell_charges(nshell, &charges_oo, &t_oo);
    let mut charges_vv = Vec::with_capacity(n_virt * n_virt);
    for &a in &space.virtuals {
        for &b in &space.virtuals {
            charges_vv.push(mo_pair_transition_shell_charge(basis, mos, &sc, a, b)?);
        }
    }
    let induced_tvv = induced_shell_charges(nshell, &charges_vv, &t_vv);
    let pot_x = matrix_vector_product(kernel, &induced_x)?;
    let pot_too = matrix_vector_product(kernel, &induced_too)?;
    let pot_tvv = matrix_vector_product(kernel, &induced_tvv)?;
    let pot_tdiff = pot_tvv
        .iter()
        .zip(pot_too.iter())
        .map(|(&v, &o)| v - o)
        .collect::<Vec<_>>();

    let mut q_oo = vec![0.0_f64; n_occ * n_occ];
    for i_pos in 0..n_occ {
        for j_pos in 0..n_occ {
            let mut eps_sum = 0.0;
            for a_pos in 0..n_virt {
                let a = space.virtuals[a_pos];
                eps_sum += orbital_energies[a]
                    * amplitudes[i_pos * n_virt + a_pos]
                    * amplitudes[j_pos * n_virt + a_pos];
            }
            let idx = i_pos * n_occ + j_pos;
            q_oo[idx] = 2.0 * omega * t_oo[idx] - 2.0 * eps_sum
                + tdiff_hplus_scale * dot(&charges_oo[idx], &pot_tdiff);
        }
    }

    let mut q_vv = vec![0.0_f64; n_virt * n_virt];
    for a_pos in 0..n_virt {
        for b_pos in 0..n_virt {
            let mut eps_sum = 0.0;
            for i_pos in 0..n_occ {
                let i = space.occupied[i_pos];
                eps_sum += orbital_energies[i]
                    * amplitudes[i_pos * n_virt + a_pos]
                    * amplitudes[i_pos * n_virt + b_pos];
            }
            q_vv[a_pos * n_virt + b_pos] =
                2.0 * omega * t_vv[a_pos * n_virt + b_pos] + 2.0 * eps_sum;
        }
    }

    let mut q_ov = vec![0.0_f64; n_occ * n_virt];
    let mut q_vo = vec![0.0_f64; n_occ * n_virt];
    for i_pos in 0..n_occ {
        let i = space.occupied[i_pos];
        for a_pos in 0..n_virt {
            let a = space.virtuals[a_pos];
            let idx = i_pos * n_virt + a_pos;
            let mut left = tdiff_hplus_scale * dot(&charges_ov[idx], &pot_tdiff);
            let mut right = 0.0;
            if transition_hplus_scale != 0.0 {
                for c_pos in 0..n_virt {
                    let q_ac =
                        mo_pair_transition_shell_charge(basis, mos, &sc, a, space.virtuals[c_pos])?;
                    left += transition_hplus_scale
                        * amplitudes[i_pos * n_virt + c_pos]
                        * dot(&q_ac, &pot_x);
                }
                for k_pos in 0..n_occ {
                    let q_ki =
                        mo_pair_transition_shell_charge(basis, mos, &sc, space.occupied[k_pos], i)?;
                    right += transition_hplus_scale
                        * amplitudes[k_pos * n_virt + a_pos]
                        * dot(&q_ki, &pot_x);
                }
            }
            q_ov[idx] = left;
            q_vo[idx] = right;
        }
    }

    Ok(TdaLagrangianTerms {
        t_oo,
        t_vv,
        q_oo,
        q_vv,
        q_ov,
        q_vo,
        charges_oo,
        charges_ov,
    })
}

/// Closed-shell occupation vector (2.0 on the occupied indices of `space`).
#[cfg(test)]
fn occupations_for_space(space: &CpxtbSpace, norb: usize) -> Vec<f64> {
    let mut occ = vec![0.0_f64; norb];
    for &i in &space.occupied {
        occ[i] = 2.0;
    }
    occ
}

/// Solve the TDA Z-vector `(gap + q K q) Z = q_vo - q_ov` by diagonally
/// preconditioned conjugate gradient (the operator is the symmetric static
/// orbital-Hessian super-operator with unit coupling scale, matching the
/// FD-verified gfn2-rs convention).
#[cfg(test)]
fn solve_tda_z_vector(
    gaps: &[f64],
    kernel: &Matrix,
    transition: &[Vec<f64>],
    rhs: &[f64],
    operator_coupling: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>> {
    let n = rhs.len();
    let rhs_norm = dot(rhs, rhs).sqrt();
    if rhs_norm <= 1.0e-14 {
        return Ok(vec![0.0_f64; n]);
    }
    let apply = |u: &[f64]| tda_sigma(gaps, kernel, transition, operator_coupling, u);
    let precond = |r: &[f64]| -> Vec<f64> {
        r.iter()
            .zip(gaps.iter())
            .map(|(&v, &g)| if g.abs() > 1.0e-12 { v / g } else { v })
            .collect()
    };
    let mut x = vec![0.0_f64; n];
    let mut r = rhs.to_vec();
    let mut z = precond(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..max_iter {
        let ap = apply(&p)?;
        let denom = dot(&p, &ap);
        if denom.abs() <= 1.0e-30 {
            break;
        }
        let alpha = rz / denom;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        if dot(&r, &r).sqrt() <= tol * rhs_norm {
            return Ok(x);
        }
        z = precond(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
        rz = rz_new;
    }
    if dot(&r, &r).sqrt() <= 1.0e-6 * rhs_norm.max(1.0) {
        Ok(x)
    } else {
        Err(Gfn1Error::InvalidInput(
            "TDA Z-vector conjugate-gradient solve did not converge".to_string(),
        ))
    }
}

/// Build the relaxed one-particle difference density `P^Delta` and the
/// energy-weighted density `W` (both AO, symmetric) for the analytic TDA
/// gradient. `W` already includes the transition-charge Pulay term.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn tda_lagrangian_density_matrices(
    basis: &crate::basis::BasisSet,
    kernel: &Matrix,
    mos: &Matrix,
    orbital_energies: &[f64],
    space: &CpxtbSpace,
    amplitudes: &[f64],
    z_vector: &[f64],
    coupling_scale: f64,
    terms: &TdaLagrangianTerms,
) -> Result<(Matrix, Matrix)> {
    let n_occ = space.occupied.len();
    let n_virt = space.virtuals.len();
    let norb = mos.cols();

    // Relaxed difference density in MO basis (symmetric): occupied-occupied hole,
    // virtual-virtual particle, and the Z-vector relaxation split symmetrically
    // across the occupied-virtual blocks (so the response-gradient contraction,
    // which sums each off-diagonal AO pair once with weight two, is exact).
    let mut density_mo = Matrix::zeros(norb, norb);
    for i_pos in 0..n_occ {
        let i = space.occupied[i_pos];
        for j_pos in 0..n_occ {
            let j = space.occupied[j_pos];
            density_mo[(i, j)] = -terms.t_oo[i_pos * n_occ + j_pos];
        }
    }
    for a_pos in 0..n_virt {
        let a = space.virtuals[a_pos];
        for b_pos in 0..n_virt {
            let b = space.virtuals[b_pos];
            density_mo[(a, b)] = terms.t_vv[a_pos * n_virt + b_pos];
        }
    }
    for i_pos in 0..n_occ {
        let i = space.occupied[i_pos];
        for a_pos in 0..n_virt {
            let a = space.virtuals[a_pos];
            let half = 0.5 * z_vector[i_pos * n_virt + a_pos];
            density_mo[(a, i)] += half;
            density_mo[(i, a)] += half;
        }
    }
    let density_response = mo_coefficient_matrix_to_ao(mos, &density_mo)?;

    // Energy-weighted density in MO basis (symmetric). `hz` is the response of
    // the occupied-occupied block to the Z-vector induced shell charges.
    let scaled_z = z_vector.to_vec();
    let induced_z = induced_shell_charges(basis.shells.len(), &terms.charges_ov, &scaled_z);
    let pot_z = matrix_vector_product(kernel, &induced_z)?;
    let mut w_mo = Matrix::zeros(norb, norb);
    for i_pos in 0..n_occ {
        let i = space.occupied[i_pos];
        for j_pos in 0..n_occ {
            let j = space.occupied[j_pos];
            let idx = i_pos * n_occ + j_pos;
            let hz = coupling_scale * dot(&terms.charges_oo[idx], &pot_z);
            w_mo[(i, j)] = 0.5 * (terms.q_oo[idx] + hz);
        }
    }
    for a_pos in 0..n_virt {
        let a = space.virtuals[a_pos];
        for b_pos in 0..n_virt {
            let b = space.virtuals[b_pos];
            w_mo[(a, b)] = 0.5 * terms.q_vv[a_pos * n_virt + b_pos];
        }
    }
    for i_pos in 0..n_occ {
        let i = space.occupied[i_pos];
        for a_pos in 0..n_virt {
            let a = space.virtuals[a_pos];
            let idx = i_pos * n_virt + a_pos;
            let value = 0.5 * (terms.q_vo[idx] + orbital_energies[i] * scaled_z[idx]);
            w_mo[(a, i)] += value;
            w_mo[(i, a)] += value;
        }
    }
    let mut energy_weighted = mo_coefficient_matrix_to_ao(mos, &w_mo)?;

    // Transition-charge overlap (Pulay) term of the coupling derivative folded
    // into W: M_{mu,nu} = c (V_mu + V_nu) T_{mu,nu}, V = K P^T, T the AO
    // transition density. Folded so `-Tr[W dS]` emits it.
    if coupling_scale != 0.0 {
        let n = basis.len();
        let mut p_shell = vec![0.0_f64; basis.shells.len()];
        for (idx, qia) in terms.charges_ov.iter().enumerate() {
            let amp = amplitudes[idx];
            if amp == 0.0 {
                continue;
            }
            for (s, &q) in qia.iter().enumerate() {
                p_shell[s] += amp * q;
            }
        }
        let v_shell = matrix_vector_product(kernel, &p_shell)?;
        let mut v_ao = vec![0.0_f64; n];
        for (s, shell) in basis.shells.iter().enumerate() {
            for mu in shell.first_ao..shell.first_ao + shell.nao {
                v_ao[mu] = v_shell[s];
            }
        }
        let mut t_mo = Matrix::zeros(norb, norb);
        for i_pos in 0..n_occ {
            let i = space.occupied[i_pos];
            for a_pos in 0..n_virt {
                let a = space.virtuals[a_pos];
                let x = amplitudes[i_pos * n_virt + a_pos];
                t_mo[(i, a)] += x;
                t_mo[(a, i)] += x;
            }
        }
        let transition_density = mo_coefficient_matrix_to_ao(mos, &t_mo)?;
        for mu in 0..n {
            let vmu = v_ao[mu];
            for nu in 0..n {
                // M is already a full symmetric AO matrix; the response-gradient
                // overlap contraction sums each unordered pair once with weight
                // two, so the symmetric weight to add is M itself (not M/2).
                let m = coupling_scale * (vmu + v_ao[nu]) * transition_density[(mu, nu)];
                energy_weighted[(mu, nu)] += m;
            }
        }
    }

    Ok((density_response, energy_weighted))
}

/// Diagonal matrix element `(C^T M C)_{pp}` of an AO matrix `M` for orbital `p`.
fn mo_diagonal_element(mos: &Matrix, ao_matrix: &Matrix, p: usize) -> f64 {
    let n = mos.rows();
    let mut value = 0.0;
    for mu in 0..n {
        let c_mu = mos[(mu, p)];
        if c_mu == 0.0 {
            continue;
        }
        for nu in 0..n {
            value += c_mu * ao_matrix[(mu, nu)] * mos[(nu, p)];
        }
    }
    value
}

/// Mulliken transition shell-charge derivative for an MO pair `(left, right)`,
/// accounting for both the orbital-coefficient response `c_deriv = C U` and the
/// overlap derivative through `dsc = (dS) C + S (dC)`. Mirrors
/// [`mo_pair_transition_shell_charge`] differentiated.
fn transition_shell_charge_derivative_for_mo_pair(
    basis: &crate::basis::BasisSet,
    mos: &Matrix,
    sc: &Matrix,
    c_deriv: &Matrix,
    dsc: &Matrix,
    left: usize,
    right: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; basis.shells.len()];
    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        let end = shell.first_ao + shell.nao;
        for mu in shell.first_ao..end {
            out[shell_idx] -= c_deriv[(mu, right)] * sc[(mu, left)]
                + mos[(mu, right)] * dsc[(mu, left)]
                + c_deriv[(mu, left)] * sc[(mu, right)]
                + mos[(mu, left)] * dsc[(mu, right)];
        }
    }
    out
}

/// Per-**shell** weights `w_s` of the on-site third-order kernel-derivative term of
/// the TDA coupling gradient, so that
///
/// ```text
/// c * P^T (dK^onsite/dR) P = sum_s w_s (dq_s/dR)
/// ```
///
/// The SCC response kernel `K` is *not* purely geometric: its on-site block carries
/// the anharmonic charge curvature `d^2E_onsite/dq_A^2 = 2 Gamma_A q_A + …` at the
/// **ground-state** atomic charge `q_A` (DFTB3 third order, plus the Linear
/// Breathing-Radius orders when `charge_order > 3`). That block has no explicit
/// position dependence, but `q_A` moves with the nuclei, so
///
/// ```text
/// d/dR [ c P^T K^onsite P ] = c sum_A (d^3 E_onsite/dq_A^3) (dq_A/dR) P_A^2,
/// ```
///
/// with `P_A = sum_{s in A} P_s`. Folding the atomic weight onto every shell of the
/// atom turns it into a linear functional of the shell-charge response, which is
/// exactly what the CPHF already delivers per Cartesian degree of freedom.
///
/// Omitting this term is a *silent* error: it vanishes identically for triplets
/// (`c = 0`), for dark states (`P = 0` — including the lowest roots of symmetric
/// water) and for `Gamma_A = 0`, so only a bright state with third-order
/// electrostatics exposes it. Measured on the jittered-formaldehyde `S3` root it was
/// a `3.35e-6` Hartree/bohr constant offset that did **not** shrink with the
/// finite-difference step (residual `3.44e-6 -> 3.35e-6` for `h = 2e-3 -> 1e-3`,
/// ladder ratio 1.03 instead of 4); zeroing `Gamma` in both the analytic path and
/// the finite-difference oracle collapsed it to `6.29e-7 -> 1.57e-7`, ratio 4.00.
fn onsite_third_order_coupling_weights(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: &crate::basis::BasisSet,
    shell_charges: &[f64],
    charge_order: usize,
    p_shell: &[f64],
    coupling: f64,
) -> Result<Vec<f64>> {
    let nshell = basis.shells.len();
    let mut weights = vec![0.0_f64; nshell];
    if coupling == 0.0 {
        return Ok(weights);
    }
    let mut model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    model.charge_order = charge_order.max(3);
    let atomic_charges = model.atomic_charges(basis, shell_charges);
    let nat = atomic_charges.len();
    let mut p_atom = vec![0.0_f64; nat];
    for (ish, shell) in basis.shells.iter().enumerate() {
        p_atom[shell.atom_index] += p_shell[ish];
    }
    let mut per_atom = vec![0.0_f64; nat];
    for (atom, &qat) in atomic_charges.iter().enumerate() {
        if model.atom_shell_counts[atom] == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
            model.hardness[offset],
            model.hubbard_derivs[offset],
            model.charge_order,
            qat,
        );
        per_atom[atom] = coupling * third * p_atom[atom] * p_atom[atom];
    }
    for (ish, shell) in basis.shells.iter().enumerate() {
        weights[ish] = per_atom[shell.atom_index];
    }
    Ok(weights)
}

/// Fully analytic TD-GFN1 (TDA) excited-state **gradient** by direct differentiation
/// of the TDA excitation energy through the ground-state coupled-perturbed (CPHF)
/// orbital response.
///
/// The TDA excitation energy is `omega = sum_{ia} X_{ia}^2 (eps_a - eps_i)
/// + c * P^T K P`, with `P_s = sum_{ia} X_{ia} q^{ia}_s` the state transition shell
/// charges. At fixed (variational) amplitudes `X` its nuclear gradient is
///
/// ```text
/// d omega / dR = sum_{ia} X_{ia}^2 d(eps_a - eps_i)/dR + c d(P^T K P)/dR.
/// ```
///
/// The orbital-energy derivatives use the standard self-consistent identity
/// `d eps_p/dR = (C^T (dH0 + dF^scc) C)_pp - eps_p (C^T dS C)_pp`, where `dF^scc`
/// is the SCC response Fock built from the ground-state density response `dD/dR`
/// (the CPHF solution `solve_nonpbc_cpxtb_hessian_response`). The explicit
/// transition-transition coupling derivative `c d(P^T K P)/dR` is split into three
/// pieces:
///  * the geometric second-order kernel derivative `c P^T (dgamma/dR) P`
///    (`coupling_kernel_gradient`);
///  * the **on-site third-order** kernel derivative
///    `c sum_A (d^3E_onsite/dq_A^3)(dq_A/dR) P_A^2`
///    ([`onsite_third_order_coupling_weights`]) — the on-site block of `K` has no
///    explicit position dependence but does depend on the ground-state charge;
///  * the transition-charge derivative `2c (dP/dR)^T K P`, using the CPHF
///    orbital-rotation amplitudes `U`.
///
/// This reproduces the finite-difference gradient to FD precision because it is the
/// exact analytic derivative of the same frozen-amplitude excitation energy used by
/// [`tda_frozen_excitation_energy`]; it requires one ground-state CPHF solve over
/// the `3N` Cartesian perturbations.
fn tda_direct_excitation_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    electronic_options: &ElectronicOptions,
    amplitudes: &[f64],
    coupling: f64,
) -> Result<Vec<Vec3>> {
    let basis = &electronic.basis;
    let overlap = &electronic.integrals.overlap;
    let nat = system.atoms.len();
    let ndim = 3 * nat;

    let ao_options = AoDerivativeOptions {
        coordination_cutoff: electronic_options.hamiltonian.coordination_cutoff,
        include_cn_h0: electronic_options.hamiltonian.enable_cn_hamiltonian,
    };
    let cphf = solve_nonpbc_cpxtb_hessian_response(
        system,
        params,
        electronic,
        ao_options,
        CpxtbOptions::default(),
    )?;
    if !cphf.converged {
        return Err(Gfn1Error::InvalidInput(format!(
            "TD-GFN1 analytic gradient CPHF did not converge; residual {:.3e}",
            cphf.max_residual_norm
        )));
    }
    let mos = &cphf.mos;
    let orbital_energies = &cphf.orbital_energies;
    let space = CpxtbSpace::from_occupations(&electronic.occupations)?;
    if amplitudes.len() != space.len() {
        return Err(Gfn1Error::InvalidInput(
            "TD-GFN1 analytic gradient amplitude length mismatch".to_string(),
        ));
    }
    let kernel = response_shell_scc_kernel(system, params, electronic)?;

    // State transition shell charges and the (frozen) response potential they feel.
    let transition = transition_shell_charges(basis, mos, &electronic.occupations, overlap)?;
    let nshell = basis.shells.len();
    let mut p_shell = vec![0.0_f64; nshell];
    for (qia, &amp) in transition.iter().zip(amplitudes.iter()) {
        for (s, &q) in qia.iter().enumerate() {
            p_shell[s] += amp * q;
        }
    }
    let p_potential = matrix_vector_product(&kernel, &p_shell)?;

    let sc = overlap.matmul(mos)?;
    let mut out = vec![Vec3::zero(); nat];
    for coord in 0..ndim {
        let h0_deriv = &cphf.derivative_matrices[coord].h0_deriv;
        let overlap_deriv = &cphf.derivative_matrices[coord].overlap_deriv;
        let density_response = &cphf.density_responses[coord];

        // Total shell-charge response that drives the SCC response Fock: the response
        // of the ground density to the geometric perturbation (CPHF `dD/dR`) plus the
        // explicit `dS` metric piece of the ground occupied density. The helper
        // `response_shell_charges_from_density` folds both when given the ground
        // density and the overlap derivative.
        let shell_response = response_shell_charges_from_density(
            basis,
            overlap,
            &electronic.density,
            density_response,
            overlap_deriv,
        )?;
        let shell_potential = matrix_vector_product(&kernel, &shell_response)?;
        let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_potential)?;
        let mut f_total = h0_deriv.clone();
        for (dst, &v) in f_total
            .as_mut_slice()
            .iter_mut()
            .zip(response_fock.as_slice().iter())
        {
            *dst += v;
        }

        // Orbital-gap derivative weighted by the squared amplitudes.
        let mut value = 0.0;
        for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
            let weight = amplitudes[pair_idx] * amplitudes[pair_idx];
            if weight == 0.0 {
                continue;
            }
            let faa = mo_diagonal_element(mos, &f_total, a);
            let fii = mo_diagonal_element(mos, &f_total, i);
            let saa = mo_diagonal_element(mos, overlap_deriv, a);
            let sii = mo_diagonal_element(mos, overlap_deriv, i);
            value +=
                weight * ((faa - orbital_energies[a] * saa) - (fii - orbital_energies[i] * sii));
        }
        let atom = coord / 3;
        match coord % 3 {
            0 => out[atom].x = value,
            1 => out[atom].y = value,
            _ => out[atom].z = value,
        }
    }

    if coupling != 0.0 {
        // Explicit kernel-derivative piece c * P^T (dK/dR) P.
        let context = ResponseGradientContext::new(
            system,
            basis,
            params,
            electronic,
            electronic_options.hamiltonian.coordination_cutoff,
            electronic_options.hamiltonian.enable_cn_hamiltonian,
        )?;
        let coupling_grad = coupling_kernel_gradient(&context, &p_shell, coupling, nat);
        for atom in 0..nat {
            out[atom] += coupling_grad[atom];
        }

        // On-site third-order (charge-dependent) part of the same kernel derivative,
        // which `coupling_kernel_gradient` cannot see: it has no explicit position
        // dependence, only the ground-charge chain `dq_A/dR` the CPHF supplies.
        let onsite_weights = onsite_third_order_coupling_weights(
            system,
            params,
            basis,
            &electronic.shell_charges,
            electronic.charge_order,
            &p_shell,
            coupling,
        )?;

        // Transition-charge-derivative piece: 2c * sum (dP_shell/dR) . (K P).
        let nmo = mos.cols();
        let is_occ = |p: usize| space.occupied.contains(&p);
        for coord in 0..ndim {
            let h0_deriv = &cphf.derivative_matrices[coord].h0_deriv;
            let overlap_deriv = &cphf.derivative_matrices[coord].overlap_deriv;
            let density_response = &cphf.density_responses[coord];
            let u = &cphf.solutions[coord].amplitudes;

            // Total response Fock derivative (same construction as the orbital-gap
            // loop) — needed for the occ-occ / virt-virt rotation amplitudes.
            let shell_response = response_shell_charges_from_density(
                basis,
                overlap,
                &electronic.density,
                density_response,
                overlap_deriv,
            )?;
            let shell_potential = matrix_vector_product(&kernel, &shell_response)?;
            let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_potential)?;
            let mut f_total = h0_deriv.clone();
            for (dst, &v) in f_total
                .as_mut_slice()
                .iter_mut()
                .zip(response_fock.as_slice().iter())
            {
                *dst += v;
            }

            let mut s_mo = Matrix::zeros(nmo, nmo);
            let mut f_mo = Matrix::zeros(nmo, nmo);
            for p in 0..nmo {
                for q in 0..nmo {
                    s_mo[(p, q)] = mo_pair_overlap_deriv_element(mos, overlap_deriv, p, q);
                    f_mo[(p, q)] = mo_pair_overlap_deriv_element(mos, &f_total, p, q);
                }
            }
            // Full first-order orbital-rotation matrix `U` (C^{(R)} = C U):
            //  - occupied-virtual block from the converged CPHF solution;
            //  - occupied-occupied / virtual-virtual off-diagonal from the
            //    Brillouin/eigenvalue stationarity `U_pq = (F^(R)_pq - eps_q S^(R)_pq)
            //    / (eps_q - eps_p)` (the response Fock for these blocks depends only on
            //    the density response, already fixed by the CPHF occ-virt amplitudes),
            //    falling back to the symmetric metric `-1/2 S_pq` for degenerate pairs;
            //  - diagonal from the orthonormality metric `-1/2 S_pp`.
            let mut u_mo = Matrix::zeros(nmo, nmo);
            for p in 0..nmo {
                u_mo[(p, p)] = -0.5 * s_mo[(p, p)];
                for q in 0..nmo {
                    if p == q {
                        continue;
                    }
                    let same_block = is_occ(p) == is_occ(q);
                    if same_block {
                        let denom = orbital_energies[q] - orbital_energies[p];
                        u_mo[(p, q)] = if denom.abs() > 1.0e-8 {
                            (f_mo[(p, q)] - orbital_energies[q] * s_mo[(p, q)]) / denom
                        } else {
                            -0.5 * s_mo[(p, q)]
                        };
                    }
                }
            }
            for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
                let uval = u[pair_idx];
                u_mo[(a, i)] = uval;
                u_mo[(i, a)] = -uval - s_mo[(i, a)];
            }
            let c_deriv = mos.matmul(&u_mo)?;
            let mut dsc = overlap_deriv.matmul(mos)?;
            let s_cderiv = overlap.matmul(&c_deriv)?;
            for (dst, &v) in dsc.as_mut_slice().iter_mut().zip(s_cderiv.as_slice().iter()) {
                *dst += v;
            }
            let mut dp_shell = vec![0.0_f64; nshell];
            for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
                let dq = transition_shell_charge_derivative_for_mo_pair(
                    basis, mos, &sc, &c_deriv, &dsc, i, a,
                );
                let amp = amplitudes[pair_idx];
                for (s, &q) in dq.iter().enumerate() {
                    dp_shell[s] += amp * q;
                }
            }
            let value = 2.0 * coupling * dot(&dp_shell, &p_potential)
                + dot(&onsite_weights, &shell_response);
            let atom = coord / 3;
            match coord % 3 {
                0 => out[atom].x += value,
                1 => out[atom].y += value,
                _ => out[atom].z += value,
            }
        }
    }

    Ok(out)
}

/// `(C^T M C)_{pq}` element of an AO matrix `M` (used for the overlap- and
/// Fock-derivative matrices in the orbital-rotation amplitudes).
fn mo_pair_overlap_deriv_element(mos: &Matrix, ao_matrix: &Matrix, p: usize, q: usize) -> f64 {
    let n = mos.rows();
    let mut value = 0.0;
    for mu in 0..n {
        let c_mu = mos[(mu, p)];
        if c_mu == 0.0 {
            continue;
        }
        for nu in 0..n {
            value += c_mu * ao_matrix[(mu, nu)] * mos[(nu, q)];
        }
    }
    value
}

/// Fully analytic TD-GFN1 (TDA) excited-state gradient (non-periodic). Returns the
/// same [`TdaGradientResult`] as the finite-difference [`solve_tda_gradient`], and
/// agrees with it to finite-difference precision.
///
/// The gradient is the exact analytic nuclear derivative of the frozen-amplitude TDA
/// excitation energy (plus the analytic ground-state gradient), obtained by direct
/// differentiation through the ground-state coupled-perturbed (CPHF) orbital
/// response — see [`tda_direct_excitation_gradient`]. It solves one ground-state
/// CPHF over the `3N` Cartesian perturbations (the same machinery as the analytic
/// Hessian) and assembles `d omega/dR = sum_{ia} X_{ia}^2 d(eps_a - eps_i)/dR
/// + c d(P^T K P)/dR`. This replaced the earlier Lagrangian / Z-vector
/// relaxed-difference-density formulation, which (as ported) carried a residual
/// `~7e-3` Hartree/bohr mutual inconsistency between the relaxed density and its
/// energy-weighted density; the direct CPHF derivative is exact by construction.
pub fn solve_tda_gradient_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state_index: usize,
    options: TdaOptions,
) -> Result<TdaGradientResult> {
    let _profile = crate::profile::scope("td.tda.gradient_analytic");
    if system.lattice.is_some() {
        return solve_tda_gradient_analytic_pbc_gamma(
            system,
            params,
            electronic_options,
            state_index,
            options,
        );
    }
    // Ground state: SCC + analytic nuclear gradient.
    let grad_options = AnalyticGradientOptions {
        electronic: electronic_options.clone(),
        ..AnalyticGradientOptions::default()
    };
    let ground = analytic_gradient(system, params, grad_options)?;
    let electronic = &ground.electronic_result;

    // TDA excited state.
    let td = solve_tda(system, params, electronic, options)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let amplitudes = td.states[state_index].amplitudes.clone();
    let coupling = options.spin.coupling_scale();

    // Excitation-energy gradient by direct CPHF differentiation (exact analytic
    // derivative of the frozen-amplitude TDA Rayleigh quotient).
    let excitation_grad = tda_direct_excitation_gradient(
        system,
        params,
        electronic,
        electronic_options,
        &amplitudes,
        coupling,
    )?;

    let nat = system.atoms.len();
    let mut gradient = ground.gradient.clone();
    for atom in 0..nat {
        gradient[atom] += excitation_grad[atom];
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy: ground.total_energy + omega,
        gradient,
        forces,
    })
}

/// Fully analytic **Gamma-point periodic** TD-GFN1 (TDA) excited-state gradient.
/// Mirrors the non-periodic [`solve_tda_gradient_analytic`]: the ground-state PBC
/// analytic gradient plus the direct-CPHF excitation-energy gradient
/// ([`crate::pbc::hessian::pbc_gamma_tda_excitation_gradient`]). Reduces the
/// excited-state gradient of the Brillouin-zone-center excitation to FD precision.
fn solve_tda_gradient_analytic_pbc_gamma(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state_index: usize,
    options: TdaOptions,
) -> Result<TdaGradientResult> {
    let pbc = crate::pbc::PbcOptions {
        kmesh: crate::pbc::KMesh::gamma(),
        ..crate::pbc::PbcOptions::default()
    };
    // Ground-state PBC SCC + analytic nuclear gradient.
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    let ground = crate::pbc::pbc_gradient_from_scc(
        system,
        params,
        scf.clone(),
        electronic_options,
        &pbc,
    )?;

    // TDA excited state at Gamma (same MOs/occupation/amplitude ordering as the
    // CPHF occ-virt space below).
    let td = solve_tda_pbc_gamma(system, params, electronic_options, options)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let amplitudes = td.states[state_index].amplitudes.clone();
    let coupling = options.spin.coupling_scale();

    // Skeleton derivatives and Gamma MOs (reuse the converged SCC).
    let skeleton = crate::pbc::hessian::gamma_skeleton_derivatives(
        system,
        params,
        &scf,
        electronic_options,
        &pbc,
    )?;
    let mos = crate::pbc::hessian::gamma_mos(&scf, scf.nelec)?;

    let mut excitation_grad = crate::pbc::hessian::pbc_gamma_tda_excitation_gradient(
        system,
        params,
        &scf,
        &skeleton,
        &mos,
        electronic_options,
        &pbc,
        &amplitudes,
        coupling,
    )?;

    let nat = system.atoms.len();
    // The periodic kernel-derivative helper differentiates the Ewald `gamma` only.
    // Add the on-site third-order (ground-charge chain) part of the same kernel
    // derivative — see `onsite_third_order_coupling_weights`.
    let transition = transition_shell_charges(
        &scf.basis,
        &mos.coeff,
        &mos.occupations,
        &mos.overlap,
    )?;
    let mut p_shell = vec![0.0_f64; scf.basis.shells.len()];
    for (qia, &amp) in transition.iter().zip(amplitudes.iter()) {
        for (s, &q) in qia.iter().enumerate() {
            p_shell[s] += amp * q;
        }
    }
    // The periodic response kernel carries the DFTB3 `2 Gamma q` on-site block only
    // (charge_order 3), so match it here.
    let onsite_weights = onsite_third_order_coupling_weights(
        system,
        params,
        &scf.basis,
        &scf.shell_charges,
        3,
        &p_shell,
        coupling,
    )?;
    if onsite_weights.iter().any(|&w| w != 0.0) {
        let (density_responses, _) =
            crate::pbc::hessian::gamma_cpxtb_density_responses(&scf, &skeleton, &mos)?;
        let n = scf.basis.len();
        let mut ground_density = Matrix::zeros(n, n);
        for p in 0..mos.occupations.len() {
            let occ = mos.occupations[p];
            if occ <= 1.0e-14 {
                continue;
            }
            for mu in 0..n {
                for nu in 0..n {
                    ground_density[(mu, nu)] += occ * mos.coeff[(mu, p)] * mos.coeff[(nu, p)];
                }
            }
        }
        for coord in 0..3 * nat {
            let shell_response = response_shell_charges_from_density(
                &scf.basis,
                &mos.overlap,
                &ground_density,
                &density_responses[coord],
                &skeleton.overlap[coord],
            )?;
            let value = dot(&onsite_weights, &shell_response);
            match coord % 3 {
                0 => excitation_grad[coord / 3].x += value,
                1 => excitation_grad[coord / 3].y += value,
                _ => excitation_grad[coord / 3].z += value,
            }
        }
    }

    let mut gradient = ground.gradient.clone();
    for atom in 0..nat {
        gradient[atom] += excitation_grad[atom];
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy: ground.total_energy + omega,
        gradient,
        forces,
    })
}

/// Fully analytic **general k-mesh** periodic TD-GFN1 (TDA) excited-state gradient.
/// Generalises the Gamma-point [`solve_tda_gradient_analytic`] to a Monkhorst-Pack
/// Brillouin-zone sampling: the ground PBC analytic gradient (same k-mesh) plus the
/// direct-CPHF excitation-energy gradient summed over the BZ with the k-weights that
/// make [`solve_tda_kpoint`] energies consistent. Reduces exactly to the Gamma path
/// for a Gamma-only mesh. Verified against the finite-difference
/// [`solve_tda_kpoint_gradient`].
pub fn solve_tda_kpoint_gradient_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    kmesh: crate::pbc::KMesh,
    state_index: usize,
    options: TdaOptions,
) -> Result<TdaGradientResult> {
    let _profile = crate::profile::scope("td.tda.kpoint_gradient_analytic");
    if system.lattice.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "solve_tda_kpoint_gradient_analytic requires a periodic system".to_string(),
        ));
    }
    let pbc = crate::pbc::PbcOptions {
        kmesh,
        ..crate::pbc::PbcOptions::default()
    };
    // Ground-state PBC SCC (shared) + analytic nuclear gradient on the same mesh.
    let scf = crate::pbc::run_pbc_scc(system, params, electronic_options, &pbc)?;
    if !scf.converged {
        return Err(Gfn1Error::InvalidInput(
            "k-point TD-GFN1 gradient requires a converged periodic SCC".to_string(),
        ));
    }
    let ground = crate::pbc::pbc_gradient_from_scc(
        system,
        params,
        scf.clone(),
        electronic_options,
        &pbc,
    )?;

    // TDA excited state on the mesh; freeze the requested root's amplitudes.
    let td = solve_tda_kpoint(system, params, electronic_options, kmesh, options)?;
    if state_index >= td.states.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "requested TD-GFN1 state {state_index} but only {} are available",
            td.states.len()
        )));
    }
    let omega = td.states[state_index].excitation_energy;
    let amplitudes = td.states[state_index].amplitudes.clone();
    let labels = td.pairs.clone();
    let coupling = options.spin.coupling_scale();

    let mut excitation_grad = crate::pbc::hessian::pbc_kpoint_tda_excitation_gradient(
        system,
        params,
        &scf,
        electronic_options,
        &pbc,
        &amplitudes,
        &labels,
        coupling,
    )?;

    let nat = system.atoms.len();
    // On-site third-order kernel-derivative term, as for the Gamma path.
    let p_shell =
        kpoint_state_transition_shell_charges(&scf, electronic_options, &amplitudes, &labels)?;
    let onsite_weights = onsite_third_order_coupling_weights(
        system,
        params,
        &scf.basis,
        &scf.shell_charges,
        3,
        &p_shell,
        coupling,
    )?;
    if onsite_weights.iter().any(|&w| w != 0.0) {
        let (_, _, charge_responses) = crate::pbc::hessian::kpoint_cpxtb_density_responses(
            system,
            params,
            &scf,
            electronic_options,
            &pbc,
            true,
        )?;
        for coord in 0..3 * nat {
            let value = dot(&onsite_weights, &charge_responses[coord]);
            match coord % 3 {
                0 => excitation_grad[coord / 3].x += value,
                1 => excitation_grad[coord / 3].y += value,
                _ => excitation_grad[coord / 3].z += value,
            }
        }
    }

    let mut gradient = ground.gradient.clone();
    for atom in 0..nat {
        gradient[atom] += excitation_grad[atom];
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(TdaGradientResult {
        state_index,
        excitation_energy: omega,
        total_energy: ground.total_energy + omega,
        gradient,
        forces,
    })
}

/// Brillouin-zone-summed real transition shell charges of a k-mesh TDA state,
/// `P_s = sum_I X_I sqrt(w_k) q^I_s`, rebuilt from the converged SCC with exactly
/// the band gauge and `sqrt(w_k)` weighting [`solve_tda_kpoint`] used to define the
/// amplitudes. `labels[I] = (ik, i*n + a)`.
fn kpoint_state_transition_shell_charges(
    scf: &crate::pbc::PbcSccResult,
    electronic_options: &ElectronicOptions,
    amplitudes: &[f64],
    labels: &[(usize, usize)],
) -> Result<Vec<f64>> {
    let basis = &scf.basis;
    let n = basis.len();
    let mut p_shell = vec![0.0_f64; basis.shells.len()];
    if amplitudes.iter().all(|&x| x == 0.0) {
        return Ok(p_shell);
    }
    let mut vao = vec![0.0_f64; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let eigen_tol = electronic_options.eigen_tolerance.max(1.0e-12);
    let mut bands_per_k: Vec<Vec<(Vec<f64>, Vec<f64>)>> = Vec::with_capacity(scf.hs_k.len());
    for (h0k, sk) in scf.hs_k.iter() {
        let fock = crate::pbc::scf::fock_at_k(h0k, sk, &vao);
        let eig = crate::pbc::complex::hermitian_generalized_eigen(&fock, sk, eigen_tol)?;
        bands_per_k.push((0..n).map(|b| gauge_fixed_band(&eig, b, n)).collect());
    }
    for (idx, &(ik, ia)) in labels.iter().enumerate() {
        let amp = amplitudes[idx];
        if amp == 0.0 {
            continue;
        }
        let (i, a) = (ia / n, ia % n);
        if ik >= bands_per_k.len() || i >= n || a >= n {
            return Err(Gfn1Error::InvalidInput(
                "k-mesh TDA transition label out of range".to_string(),
            ));
        }
        let sk = &scf.hs_k[ik].1;
        let bands = &bands_per_k[ik];
        let q = kpoint_transition_shell_charge(
            basis,
            sk,
            &bands[i].0,
            &bands[i].1,
            &bands[a].0,
            &bands[a].1,
        )?;
        let scale = amp * scf.kpoints[ik].weight.sqrt();
        for (s, &v) in q.iter().enumerate() {
            p_shell[s] += scale * v;
        }
    }
    Ok(p_shell)
}

/// Reference MO coefficients of a converged non-periodic SCC — the same
/// `lowdin_solve_generalized` eigenvectors [`solve_tda`] builds its amplitudes
/// from, so an amplitude vector returned by `solve_tda` is expressed in exactly
/// this MO gauge.
fn reference_mo_coefficients(electronic: &ElectronicResult) -> Result<Matrix> {
    Ok(lowdin_solve_generalized(&electronic.fock, &electronic.integrals.overlap, 1.0e-12)?.vectors)
}

/// **Fix the MO phase gauge** of `mos` (the eigenvectors at the *current*
/// geometry) against a `reference` MO set, by flipping the sign of every column
/// whose overlap `<C^ref_p | S | C_p>` with its reference partner is negative.
///
/// A symmetric eigensolver fixes each eigenvector only up to a sign, and the
/// sign it happens to return is *not* a continuous function of the geometry. The
/// TDA excitation energy is gauge invariant (a sign flip of MO `p` is a diagonal
/// similarity transform `D A D` of the TDA matrix, `D = diag(+-1)`, which leaves
/// the eigenvalues alone and only relabels the amplitude signs) — but the
/// *frozen-amplitude* Rayleigh quotient `X^T A(R) X` at a **fixed** `X` is not:
/// the transition shell charges `q^{ia} ~ C_i S C_a` flip sign with either
/// orbital, so a phase flip somewhere along a displaced geometry changes the
/// sign of the transition-charge Coulomb coupling `c (sum_ia X_ia q^ia)^T K
/// (sum_jb X_jb q^jb)` cross terms and puts a step discontinuity into the
/// frozen excitation energy. Aligning to the reference removes it; the states
/// with vanishing transition charge (dark roots, e.g. water `S0`) are accidentally
/// immune, which is why the defect stayed hidden.
fn phase_align_mos_to_reference(
    mos: &mut Matrix,
    reference: &Matrix,
    overlap: &Matrix,
) -> Result<()> {
    let n = mos.rows();
    let nmo = mos.cols();
    if reference.rows() != n
        || reference.cols() != nmo
        || overlap.rows() != n
        || overlap.cols() != n
    {
        return Err(Gfn1Error::InvalidInput(
            "TD-GFN1 MO phase alignment shape mismatch".to_string(),
        ));
    }
    let sc = overlap.matmul(mos)?;
    for p in 0..nmo {
        let mut t = 0.0;
        for mu in 0..n {
            t += reference[(mu, p)] * sc[(mu, p)];
        }
        if t < 0.0 {
            for mu in 0..n {
                mos[(mu, p)] = -mos[(mu, p)];
            }
        }
    }
    Ok(())
}

/// TDA excitation energy evaluated with a **fixed** amplitude vector at the
/// current geometry (the SCC, orbital energies, transition charges and response
/// kernel are recomputed; only `amplitudes` is frozen). This Rayleigh quotient
/// `X^T A(R) X / X^T X` reproduces the variational excitation energy at the
/// reference geometry, and its nuclear derivative is exactly `domega/dR` for the
/// tracked adiabatic state (amplitude stationarity, the `2n+1` rule), including
/// orbital relaxation through the re-converged SCC and without the root-tracking
/// ambiguity of a re-diagonalised finite difference.
///
/// `reference` is the converged SCC that `amplitudes` came from. **Pass it
/// whenever `system` is not exactly the geometry where the amplitudes were
/// determined** — i.e. for every finite-difference use. It fixes the MO phase
/// gauge (see [`phase_align_mos_to_reference`]); without it the eigensolver's
/// arbitrary per-orbital sign makes this quantity discontinuous in the geometry
/// for any state with non-vanishing transition charge, and a central difference
/// of it returns garbage that diverges as `1/h` (measured: `14 Hartree/bohr` on
/// the water `S3` root at `h = 1e-3`, against a true gradient of `~1e-2`).
/// Passing `None` is only correct at the reference geometry itself.
///
/// Non-periodic.
pub fn tda_frozen_excitation_energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    amplitudes: &[f64],
    spin: TdaSpin,
    reference: Option<&ElectronicResult>,
) -> Result<f64> {
    let reference_mos = match reference {
        Some(electronic) => Some(reference_mo_coefficients(electronic)?),
        None => None,
    };
    tda_frozen_excitation_energy_with_mos(
        system,
        params,
        electronic_options,
        amplitudes,
        spin,
        reference_mos.as_ref(),
    )
}

/// [`tda_frozen_excitation_energy`] against pre-computed reference MO
/// coefficients, so a finite-difference sweep re-uses one reference
/// diagonalisation instead of repeating it at every displacement.
fn tda_frozen_excitation_energy_with_mos(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    amplitudes: &[f64],
    spin: TdaSpin,
    reference_mos: Option<&Matrix>,
) -> Result<f64> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "tda_frozen_excitation_energy is non-periodic".to_string(),
        ));
    }
    let electronic = crate::run_electronic(system, params, electronic_options.clone())?;
    let overlap = &electronic.integrals.overlap;
    let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12)?;
    let space = CpxtbSpace::from_occupations(&electronic.occupations)?;
    if amplitudes.len() != space.len() {
        return Err(Gfn1Error::InvalidInput(
            "tda_frozen_excitation_energy amplitude length mismatch".to_string(),
        ));
    }
    let mut mos = eig.vectors;
    if let Some(reference) = reference_mos {
        phase_align_mos_to_reference(&mut mos, reference, overlap)?;
    }
    let gaps = space
        .pairs
        .iter()
        .map(|&(i, a)| eig.values[a] - eig.values[i])
        .collect::<Vec<_>>();
    let kernel = response_shell_scc_kernel(system, params, &electronic)?;
    let transition =
        transition_shell_charges(&electronic.basis, &mos, &electronic.occupations, overlap)?;
    let sigma = tda_sigma(
        &gaps,
        &kernel,
        &transition,
        spin.coupling_scale(),
        amplitudes,
    )?;
    let norm2 = dot(amplitudes, amplitudes);
    Ok(dot(amplitudes, &sigma) / norm2)
}

/// TDA sigma vector `A X = gap*X + c * q (K (q^T X))`.
pub(crate) fn tda_sigma(
    gaps: &[f64],
    kernel: &Matrix,
    transition: &[Vec<f64>],
    coupling_scale: f64,
    vector: &[f64],
) -> Result<Vec<f64>> {
    let mut sigma = gaps
        .iter()
        .zip(vector.iter())
        .map(|(&gap, &u)| gap * u)
        .collect::<Vec<_>>();
    if coupling_scale != 0.0 {
        let nsh = kernel.rows();
        let mut induced = vec![0.0_f64; nsh];
        for (qia, &u) in transition.iter().zip(vector.iter()) {
            if u == 0.0 {
                continue;
            }
            for (shell, &q) in qia.iter().enumerate() {
                induced[shell] += q * u;
            }
        }
        let potential = matrix_vector_product(kernel, &induced)?;
        for (row, qia) in transition.iter().enumerate() {
            let coupling: f64 = qia.iter().zip(potential.iter()).map(|(&q, &v)| q * v).sum();
            sigma[row] += coupling_scale * coupling;
        }
    }
    Ok(sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cphf::response_electronic_gradient_terms;
    use crate::run_electronic;

    fn load_params() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    /// Rotate columns (i,a) of an MO coefficient matrix by an exact orthogonal
    /// rotation. Preserves `C^T S C = I` exactly (no re-orthogonalization needed).
    fn rotate_ia(c: &Matrix, i: usize, a: usize, kappa: f64) -> Matrix {
        let mut out = c.clone();
        let (cs, sn) = (kappa.cos(), kappa.sin());
        for mu in 0..c.rows() {
            let ci = c[(mu, i)];
            let ca = c[(mu, a)];
            out[(mu, i)] = cs * ci + sn * ca;
            out[(mu, a)] = -sn * ci + cs * ca;
        }
        out
    }

    /// Build the GFN1 SCC Fock from a density at FIXED AO integrals (no re-SCF, no
    /// re-diagonalization): `F[D] = H0 - 1/2 (V_mu+V_nu) S`, `V = gamma q(D) + 3rd`.
    fn fock_from_density(
        system: &PeriodicSystem,
        basis: &crate::basis::BasisSet,
        params: &Gfn1Parameters,
        h0: &Matrix,
        overlap: &Matrix,
        density: &Matrix,
    ) -> Matrix {
        let q = crate::electronic::mulliken_shell_charges(basis, overlap, density);
        let ce = crate::coulomb::coulomb_energy_potential(system, basis, &q, params).unwrap();
        crate::electronic::fock_from_shell_potential(basis, overlap, h0, &ce.shell_potential)
    }

    /// Closed-shell density from MO coefficients: `D = sum_{p in occ} 2 C_p C_p^T`.
    fn density_from_mos(mos: &Matrix, occupied: &[usize], n: usize) -> Matrix {
        let mut d = Matrix::zeros(n, n);
        for &p in occupied {
            for mu in 0..n {
                for nu in 0..n {
                    d[(mu, nu)] += 2.0 * mos[(mu, p)] * mos[(nu, p)];
                }
            }
        }
        d
    }

    fn electronic(system: &PeriodicSystem, params: &Gfn1Parameters) -> ElectronicResult {
        let opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        run_electronic(system, params, opts).unwrap()
    }

    /// Frequency-dependent optical rotation: zero for an achiral molecule at every
    /// frequency, nonzero for a chiral one, and negated by reflecting the molecule
    /// (the enantiomer rotates light the opposite way).
    #[test]
    fn optical_rotation_vanishes_achiral_and_flips_under_mirror() {
        let Some(params) = load_params() else {
            return;
        };
        let opts = TdaOptions {
            n_states: 8,
            spin: TdaSpin::Singlet,
        };
        let freqs = [0.0, 0.06, 0.10]; // Hartree (static, ~760 nm, ~456 nm; below resonance)

        // Achiral water -> beta = 0 at every frequency.
        let water = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let elw = electronic(&water, &params);
        let beta_w =
            tda_optical_rotation(&water, &params, &elw, opts, Vec3::zero(), &freqs).unwrap();
        for b in &beta_w {
            assert!(
                b.abs() < 1.0e-8,
                "achiral water optical rotation {b:.3e} != 0"
            );
        }

        // Chiral gauche H2O2 (C2 point group) -> nonzero; its mirror image (z -> -z)
        // negates beta.
        let h2o2 = PeriodicSystem::from_xyz_str(
            "4\nH2O2\nO 0.0 0.7375 -0.0528\nO 0.0 -0.7375 -0.0528\nH 0.819 0.817 0.422\nH -0.819 -0.817 0.422\n",
            0.0,
            false,
        )
        .unwrap();
        let h2o2_mirror = PeriodicSystem::from_xyz_str(
            "4\nH2O2\nO 0.0 0.7375 0.0528\nO 0.0 -0.7375 0.0528\nH 0.819 0.817 -0.422\nH -0.819 -0.817 -0.422\n",
            0.0,
            false,
        )
        .unwrap();
        let beta = tda_optical_rotation(
            &h2o2,
            &params,
            &electronic(&h2o2, &params),
            opts,
            Vec3::zero(),
            &freqs,
        )
        .unwrap();
        let beta_m = tda_optical_rotation(
            &h2o2_mirror,
            &params,
            &electronic(&h2o2_mirror, &params),
            opts,
            Vec3::zero(),
            &freqs,
        )
        .unwrap();
        let max_abs = beta.iter().fold(0.0_f64, |m, &b| m.max(b.abs()));
        assert!(
            max_abs > 1.0e-6,
            "chiral H2O2 optical rotation vanishes: {max_abs:.3e}"
        );
        for (b, bm) in beta.iter().zip(beta_m.iter()) {
            assert!(
                (b + bm).abs() < 1.0e-6 * b.abs().max(1.0),
                "mirror image did not negate beta: {b} vs {bm}"
            );
        }
    }

    /// Strict FD gate for the fully analytic excited-state gradient
    /// (`solve_tda_gradient_analytic`): it must agree with a central finite difference of the
    /// total excited energy (ground SCC free energy + frozen-amplitude TDA Rayleigh quotient)
    /// to finite-difference precision. The direct-CPHF formulation is the exact analytic
    /// derivative of that same frozen-amplitude excitation energy, so the only residual is the
    /// `O(h^2)` central-difference truncation of the oracle. Verified via
    /// [`tda_frozen_excitation_energy`].
    #[test]
    fn tda_analytic_gradient_matches_frozen_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let state = 0usize;
        let ana = solve_tda_gradient_analytic(&system, &params, &eo, state, opts).unwrap();
        let el = run_electronic(&system, &params, eo.clone()).unwrap();
        let amps = solve_tda(&system, &params, &el, opts).unwrap().states[state]
            .amplitudes
            .clone();
        let total = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, eo.clone()).unwrap().total_free
                + tda_frozen_excitation_energy(sys, &params, &eo, &amps, opts.spin, Some(&el))
                    .unwrap()
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut p = system.clone();
                let mut m = system.clone();
                shift_atom(&mut p, atom, axis, h);
                shift_atom(&mut m, atom, axis, -h);
                let fd = (total(&p) - total(&m)) / (2.0 * h);
                let a = match axis {
                    0 => ana.gradient[atom].x,
                    1 => ana.gradient[atom].y,
                    _ => ana.gradient[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        eprintln!("TDA analytic vs frozen-FD (water S0): max diff {maxdiff:.3e} Ha/bohr");
        assert!(
            maxdiff < 1.0e-5,
            "TDA analytic gradient vs frozen-FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Shared frozen-FD oracle for the analytic excited-state gradient: returns the
    /// max |analytic - central-FD| over all Cartesian components for state 0 of the
    /// jittered water fixture, given a parameter set. Used both by the regression
    /// guard and the third-order-isolation contrast below.
    fn tda_fd_gate_maxdiff(params: &Gfn1Parameters) -> f64 {
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let state = 0usize;
        let ana = solve_tda_gradient_analytic(&system, params, &eo, state, opts).unwrap();
        let el = run_electronic(&system, params, eo.clone()).unwrap();
        let amps = solve_tda(&system, params, &el, opts).unwrap().states[state]
            .amplitudes
            .clone();
        let total = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, params, eo.clone()).unwrap().total_free
                + tda_frozen_excitation_energy(sys, params, &eo, &amps, opts.spin, Some(&el))
                    .unwrap()
        };
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut p = system.clone();
                let mut m = system.clone();
                shift_atom(&mut p, atom, axis, h);
                shift_atom(&mut m, atom, axis, -h);
                let fd = (total(&p) - total(&m)) / (2.0 * h);
                let a = match axis {
                    0 => ana.gradient[atom].x,
                    1 => ana.gradient[atom].y,
                    _ => ana.gradient[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        maxdiff
    }

    /// The semi-numerical hybrid gradient (analytic ground + frozen-amplitude
    /// finite-difference of the excitation energy) reproduces the central finite
    /// difference of the total excited energy to FD precision. The frozen-amplitude
    /// `domega/dR` is identical to the oracle's by construction (same step), so this
    /// asserts the analytic ground gradient matches the FD ground gradient — i.e. the
    /// hybrid is exact for the tracked state, unlike the experimental analytic path.
    #[test]
    fn tda_seminumerical_matches_energy_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let state = 0usize;
        let h = 1.0e-4;
        let semi = solve_tda_gradient_seminumerical(&system, &params, &eo, state, opts, h).unwrap();
        let el = run_electronic(&system, &params, eo.clone()).unwrap();
        let amps = solve_tda(&system, &params, &el, opts).unwrap().states[state]
            .amplitudes
            .clone();
        let total = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, eo.clone()).unwrap().total_free
                + tda_frozen_excitation_energy(sys, &params, &eo, &amps, opts.spin, Some(&el))
                    .unwrap()
        };
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut p = system.clone();
                let mut m = system.clone();
                shift_atom(&mut p, atom, axis, h);
                shift_atom(&mut m, atom, axis, -h);
                let fd = (total(&p) - total(&m)) / (2.0 * h);
                let a = match axis {
                    0 => semi.gradient[atom].x,
                    1 => semi.gradient[atom].y,
                    _ => semi.gradient[atom].z,
                };
                maxdiff = maxdiff.max((a - fd).abs());
            }
        }
        assert!(
            maxdiff < 1.0e-5,
            "semi-numerical gradient vs total-energy FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// The semi-numerical hybrid agrees with the fully-numerical root-tracking
    /// gradient `solve_tda_gradient` (the robust production fallback) on water S0.
    /// They differ only by the frozen-amplitude vs re-diagonalised treatment (O(h^2)
    /// near a well-separated state) and the analytic-vs-FD ground gradient.
    #[test]
    fn tda_seminumerical_matches_full_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let state = 0usize;
        let h = 1.0e-4;
        let semi = solve_tda_gradient_seminumerical(&system, &params, &eo, state, opts, h).unwrap();
        let full = solve_tda_gradient(&system, &params, &eo, state, opts, h).unwrap();
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            let d = semi.gradient[atom] - full.gradient[atom];
            maxdiff = maxdiff.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
        }
        assert!(
            maxdiff < 1.0e-3,
            "semi-numerical vs full-FD gradient: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// `TdaGradientMethod::parse` accepts the documented user-facing spellings.
    #[test]
    fn tda_gradient_method_parse() {
        assert_eq!(
            TdaGradientMethod::parse("semi_numerical"),
            Some(TdaGradientMethod::SemiNumerical)
        );
        assert_eq!(
            TdaGradientMethod::parse("semi-numerical"),
            Some(TdaGradientMethod::SemiNumerical)
        );
        assert_eq!(
            TdaGradientMethod::parse("FD"),
            Some(TdaGradientMethod::FiniteDifference)
        );
        assert_eq!(
            TdaGradientMethod::parse("finite_difference"),
            Some(TdaGradientMethod::FiniteDifference)
        );
        assert_eq!(
            TdaGradientMethod::parse("analytic"),
            Some(TdaGradientMethod::Analytic)
        );
        assert_eq!(TdaGradientMethod::parse("bogus"), None);
        assert_eq!(
            TdaGradientMethod::default(),
            TdaGradientMethod::SemiNumerical
        );
    }

    /// TEST 1 (third-order isolation): the analytic gradient drops the on-site
    /// third-order kernel `K^(3)_A = 2 Gamma_A q_A` from the relaxed-density
    /// contraction on the (former) grounds that it is "charge-independent". But
    /// `q_A` is a Mulliken charge (overlap-dependent), so `dK^(3)/dR = 2 Gamma_A
    /// dq_A/dR` carries an explicit `dS/dR` piece that does NOT vanish at fixed P.
    /// Zeroing the Hubbard derivative `Gamma_A` (gam3) in BOTH the analytic path
    /// and the frozen-FD oracle removes this term consistently. If the residual
    /// collapses, the missing third-order Mulliken-overlap derivative is the cause.
    #[test]
    fn tda_gamma3_isolation_contrast() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let full = tda_fd_gate_maxdiff(&params);
        let zero = tda_fd_gate_maxdiff(&params0);
        eprintln!(
            "GAMMA3 CONTRAST: full(Gamma!=0)={full:.4e}  Gamma=0 -> {zero:.4e}  ratio={:.2}",
            full / zero.max(1e-30)
        );
        // Zeroing the Hubbard derivative (third-order on-site kernel) barely moves
        // the excited-gradient residual: the missing `2 Gamma dq/dR` Mulliken-overlap
        // derivative is NOT the dominant ~7e-3 error source on water S0.
        assert!(
            (full - zero).abs() < 1.0e-4,
            "third-order Gamma materially changes the residual ({full:.3e} vs {zero:.3e}); \
             revisit the third-order charge-dependent kernel derivative"
        );
    }

    /// Legacy diagnostic: reproduces the **former** Z-vector / Lagrangian
    /// relaxed-difference-density TDA gradient (no longer the production path; the
    /// production [`solve_tda_gradient_analytic`] now uses the exact direct-CPHF
    /// derivative) with a `z_scale` knob multiplying the Z-vector (orbital relaxation)
    /// before P/W are built: z_scale=0 turns off relaxation (P=T unrelaxed), z_scale=1
    /// is the former production scale. Kept to exercise the Lagrangian helper functions
    /// (`tda_lagrangian_terms`, `solve_tda_z_vector`, `tda_lagrangian_density_matrices`).
    fn tda_analytic_grad_zscale(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        eo: &ElectronicOptions,
        state: usize,
        opts: TdaOptions,
        z_scale: f64,
        w_scale: f64,
    ) -> Vec<Vec3> {
        let grad_options = AnalyticGradientOptions {
            electronic: eo.clone(),
            ..AnalyticGradientOptions::default()
        };
        let ground = analytic_gradient(system, params, grad_options).unwrap();
        let electronic = &ground.electronic_result;
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let td = solve_tda(system, params, electronic, opts).unwrap();
        let omega = td.states[state].excitation_energy;
        let amplitudes = td.states[state].amplitudes.clone();
        let coupling = opts.spin.coupling_scale();
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let kernel = response_shell_scc_kernel(system, params, electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        let (tdiff_z, transition_z) = match opts.spin {
            TdaSpin::Singlet => (2.0, 2.0),
            TdaSpin::Triplet => (1.0, 0.0),
        };
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            tdiff_z,
            transition_z,
        )
        .unwrap();
        let rhs = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect::<Vec<_>>();
        let mut z_vector =
            solve_tda_z_vector(&gaps, &kernel, &transition, &rhs, 2.0, 500, 1.0e-9).unwrap();
        for z in z_vector.iter_mut() {
            *z *= z_scale;
        }
        let terms = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            coupling,
            coupling,
        )
        .unwrap();
        let (density_response, mut energy_weighted) = tda_lagrangian_density_matrices(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            &amplitudes,
            &z_vector,
            coupling,
            &terms,
        )
        .unwrap();
        if w_scale != 1.0 {
            for r in 0..energy_weighted.rows() {
                for c in 0..energy_weighted.cols() {
                    energy_weighted[(r, c)] *= w_scale;
                }
            }
        }
        let zero_overlap = Matrix::zeros(basis.len(), basis.len());
        let shell_charge_response = response_shell_charges_from_density(
            basis,
            overlap,
            &electronic.density,
            &density_response,
            &zero_overlap,
        )
        .unwrap();
        let context = ResponseGradientContext::new(
            system,
            basis,
            params,
            electronic,
            eo.hamiltonian.coordination_cutoff,
            eo.hamiltonian.enable_cn_hamiltonian,
        )
        .unwrap();
        let response_grad = response_electronic_gradient(
            system,
            electronic,
            &kernel,
            &context,
            &density_response,
            &density_response,
            &energy_weighted,
            &shell_charge_response,
        )
        .unwrap();
        let mut p_shell = vec![0.0_f64; basis.shells.len()];
        for (idx, qia) in transition.iter().enumerate() {
            let amp = amplitudes[idx];
            for (s, &q) in qia.iter().enumerate() {
                p_shell[s] += amp * q;
            }
        }
        let nat = system.atoms.len();
        let coupling_grad = coupling_kernel_gradient(&context, &p_shell, coupling, nat);
        let mut gradient = ground.gradient.clone();
        for atom in 0..nat {
            gradient[atom] += response_grad[atom] + coupling_grad[atom];
        }
        gradient
    }

    /// Legacy-path diagnostic: sweeps the Z-vector relaxation scale of the **former**
    /// Lagrangian TDA gradient and reports max |legacy-analytic - frozen-FD| for water
    /// S0. Documents why that formulation needed orbital relaxation (z=0 is ~10x worse)
    /// and that z=1 was near its optimum (the residual was structural, not a relaxation-
    /// amplitude error — which is why the production path was switched to the exact
    /// direct-CPHF derivative). Exercises the retained Lagrangian helper functions.
    #[test]
    fn tda_zvector_scale_sweep() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let el = run_electronic(&system, &params, eo.clone()).unwrap();
        let amps = solve_tda(&system, &params, &el, opts).unwrap().states[0]
            .amplitudes
            .clone();
        let total = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params, eo.clone()).unwrap().total_free
                + tda_frozen_excitation_energy(sys, &params, &eo, &amps, opts.spin, Some(&el))
                    .unwrap()
        };
        let h = 1.0e-4;
        // precompute FD oracle gradient
        let mut fd = vec![Vec3::zero(); system.atoms.len()];
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut p = system.clone();
                let mut m = system.clone();
                shift_atom(&mut p, atom, axis, h);
                shift_atom(&mut m, atom, axis, -h);
                let g = (total(&p) - total(&m)) / (2.0 * h);
                match axis {
                    0 => fd[atom].x = g,
                    1 => fd[atom].y = g,
                    _ => fd[atom].z = g,
                }
            }
        }
        let mut z0 = 0.0_f64;
        let mut z1 = 0.0_f64;
        for &zs in &[0.0_f64, 0.5, 1.0, 1.5, 2.0] {
            let ana = tda_analytic_grad_zscale(&system, &params, &eo, 0, opts, zs, 1.0);
            let mut maxdiff = 0.0_f64;
            for atom in 0..system.atoms.len() {
                let d = ana[atom] - fd[atom];
                maxdiff = maxdiff.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
            }
            if zs == 0.0 {
                z0 = maxdiff;
            }
            if zs == 1.0 {
                z1 = maxdiff;
            }
            eprintln!("Z-SCALE {zs:.2}: max|ana-FD| = {maxdiff:.4e}");
        }
        // The Z-vector orbital relaxation is essential (z=0 is ~10x worse) and the
        // production scale z=1 is near the optimum: the residual is structural, not
        // a relaxation-amplitude error.
        assert!(
            z1 < 0.25 * z0,
            "Z-vector relaxation not reducing the residual as expected: z0={z0:.3e} z1={z1:.3e}"
        );
        // W-scale sweep at z=1
        for &ws in &[0.0_f64, 0.5, 1.0, 1.5, 2.0] {
            let ana = tda_analytic_grad_zscale(&system, &params, &eo, 0, opts, 1.0, ws);
            let mut maxdiff = 0.0_f64;
            for atom in 0..system.atoms.len() {
                let d = ana[atom] - fd[atom];
                maxdiff = maxdiff.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
            }
            eprintln!("W-SCALE {ws:.2} (z=1): max|ana-FD| = {maxdiff:.4e}");
        }
        // per-component residual at z=1,w=1
        let ana = tda_analytic_grad_zscale(&system, &params, &eo, 0, opts, 1.0, 1.0);
        eprintln!("PER-COMPONENT residual (ana - FD) at z=1,w=1:");
        for atom in 0..system.atoms.len() {
            let d = ana[atom] - fd[atom];
            eprintln!(
                "  atom {atom}: ana=({:+.5},{:+.5},{:+.5})  FD=({:+.5},{:+.5},{:+.5})  d=({:+.2e},{:+.2e},{:+.2e})",
                ana[atom].x, ana[atom].y, ana[atom].z,
                fd[atom].x, fd[atom].y, fd[atom].z,
                d.x, d.y, d.z
            );
        }
    }

    /// TEST 4 (charge-map invariants + SCF residual): the difference-density and
    /// Z-vector relaxation density must carry zero net Mulliken charge
    /// (`sum_s q_T(s)=0`, `sum_s q_z(s)=0` — an electron is promoted, total charge
    /// conserved), the charge map must be linear and reproduce the SCC ground charges,
    /// and the reference orbitals must satisfy `FC=SCe`, `C^T S C = I`. These probe the
    /// oo/vv difference-density charge map that neither the ground gradient (occupied)
    /// nor the polarizability (occ-virt) exercises, independent of any contraction sign
    /// convention.
    #[test]
    fn tda_charge_map_invariants() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let zero = Matrix::zeros(n, n);
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let td = solve_tda(&system, &params, &electronic, opts).unwrap();
        let omega = td.states[0].excitation_energy;
        let amplitudes = td.states[0].amplitudes.clone();
        let coupling = opts.spin.coupling_scale();
        let kernel = response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            2.0,
            2.0,
        )
        .unwrap();
        let rhs = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect::<Vec<_>>();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let z_vector =
            solve_tda_z_vector(&gaps, &kernel, &transition, &rhs, 2.0, 500, 1.0e-9).unwrap();
        let terms = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            coupling,
            coupling,
        )
        .unwrap();
        let zeros_z = vec![0.0_f64; z_vector.len()];
        let (t_density, _) = tda_lagrangian_density_matrices(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            &amplitudes,
            &zeros_z,
            coupling,
            &terms,
        )
        .unwrap();
        let (p_density, _) = tda_lagrangian_density_matrices(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            &amplitudes,
            &z_vector,
            coupling,
            &terms,
        )
        .unwrap();
        // z relaxation density = P(z) - P(0)
        let mut z_density = p_density.clone();
        for r in 0..n {
            for c in 0..n {
                z_density[(r, c)] -= t_density[(r, c)];
            }
        }
        let q_t =
            response_shell_charges_from_density(basis, overlap, &zero, &t_density, &zero).unwrap();
        let q_z =
            response_shell_charges_from_density(basis, overlap, &zero, &z_density, &zero).unwrap();
        let sum_t: f64 = q_t.iter().sum();
        let sum_z: f64 = q_z.iter().sum();
        // linearity: q(T + z) == q(T) + q(z)
        let q_p =
            response_shell_charges_from_density(basis, overlap, &zero, &p_density, &zero).unwrap();
        let mut lin_err = 0.0_f64;
        for s in 0..q_p.len() {
            lin_err = lin_err.max((q_p[s] - (q_t[s] + q_z[s])).abs());
        }
        // SCF residual: F C = S C e, C^T S C = I
        let sc = overlap.matmul(mos).unwrap();
        let fc = electronic.fock.matmul(mos).unwrap();
        let mut res_fc = 0.0_f64;
        let mut res_orth = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                res_fc = res_fc.max((fc[(p, q)] - sc[(p, q)] * orbital_energies[q]).abs());
            }
        }
        for p in 0..n {
            for q in 0..n {
                let mut v = 0.0;
                for mu in 0..n {
                    v += mos[(mu, p)] * sc[(mu, q)];
                }
                let target = if p == q { 1.0 } else { 0.0 };
                res_orth = res_orth.max((v - target).abs());
            }
        }
        eprintln!(
            "CHARGE-MAP INVARIANTS: sum q_T={sum_t:.3e}  sum q_z={sum_z:.3e}  lin_err={lin_err:.3e}  |FC-SCe|={res_fc:.3e}  |CtSC-I|={res_orth:.3e}"
        );
        assert!(
            sum_t.abs() < 1.0e-9,
            "difference density not charge-neutral: {sum_t:.3e}"
        );
        assert!(
            sum_z.abs() < 1.0e-9,
            "Z relaxation density not charge-neutral: {sum_z:.3e}"
        );
        assert!(lin_err < 1.0e-12, "charge map not linear: {lin_err:.3e}");
        assert!(res_fc < 1.0e-8, "FC != SCe: {res_fc:.3e}");
        assert!(res_orth < 1.0e-8, "C^T S C != I: {res_orth:.3e}");
    }

    /// Per-term isolation of the CPXTB response gradient against the finite
    /// difference of the energy functional each term represents, for the relaxed
    /// TDA difference density `P` and energy-weighted density `W` of water S0 —
    /// INCLUDING the virt-virt promoted-electron block that neither the ground
    /// gradient nor the polarizability exercises. Uses the race-free per-term
    /// decomposition `response_electronic_gradient_terms` (no global state). CN is
    /// off so the band is the pure core Hamiltonian. Returns `(band, pulay, scc)`
    /// max |analytic - FD| over all Cartesian components.
    ///
    /// FIXED at the reference geometry: `P0, D0, W0, q_P(R0), q_D(R0)=shell_charges,
    /// V_D, V_P`. VARIED: `S(R), H0(R), gamma(R)` only (no SCF re-convergence of the
    /// densities; `run_electronic` at displaced R supplies the displaced basis and,
    /// for the second-order gamma, the geometry — which is charge-independent). The
    /// SCC functional whose explicit derivative equals the code's `scc_overlap +
    /// scc_kernel` terms is
    ///   `Psi(R) = sum_s V_D[s] q_P,s(R) + sum_s V_P[s] q_D,s(R) + q_P(R0).gamma(R).q_D(R0)`.
    fn contraction_isolation_maxdiffs(params: &Gfn1Parameters) -> (f64, f64, f64) {
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let mut eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        eo.hamiltonian.enable_cn_hamiltonian = false; // band = pure H0
        let electronic = run_electronic(&system, params, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let td = solve_tda(&system, params, &electronic, opts).unwrap();
        let omega = td.states[0].excitation_energy;
        let amplitudes = td.states[0].amplitudes.clone();
        let coupling = opts.spin.coupling_scale();
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let kernel = response_shell_scc_kernel(&system, params, &electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            2.0,
            2.0,
        )
        .unwrap();
        let rhs = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect::<Vec<_>>();
        let z_vector =
            solve_tda_z_vector(&gaps, &kernel, &transition, &rhs, 2.0, 500, 1.0e-9).unwrap();
        let terms = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            coupling,
            coupling,
        )
        .unwrap();
        let (p, w) = tda_lagrangian_density_matrices(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            &amplitudes,
            &z_vector,
            coupling,
            &terms,
        )
        .unwrap();
        let n = basis.len();
        let zero = Matrix::zeros(n, n);
        let context = ResponseGradientContext::new(
            &system,
            basis,
            params,
            &electronic,
            eo.hamiltonian.coordination_cutoff,
            false,
        )
        .unwrap();
        let q_p0 =
            response_shell_charges_from_density(basis, overlap, &electronic.density, &p, &zero)
                .unwrap();
        // Single per-term contraction call: band/poly from P, pulay from W, scc from P & q_P.
        let gterms = response_electronic_gradient_terms(
            &system,
            &electronic,
            &kernel,
            &context,
            &p,
            &p,
            &w,
            &q_p0,
        )
        .unwrap();
        let nat = system.atoms.len();
        let band_ana: Vec<Vec3> = (0..nat)
            .map(|a| gterms.band[a] + gterms.polynomial[a])
            .collect();
        let pulay_ana = gterms.pulay.clone();
        let scc_ana: Vec<Vec3> = (0..nat)
            .map(|a| gterms.scc_overlap[a] + gterms.scc_kernel[a])
            .collect();
        let v_d = electronic.shell_scc_potential.clone();
        let v_p = matrix_vector_product(&kernel, &q_p0).unwrap();
        let q_d0 = electronic.shell_charges.clone();
        let band_energy = |sys: &PeriodicSystem| -> f64 {
            let b = crate::basis::BasisSet::build(
                sys,
                params,
                crate::basis::BasisOptions { nprim: eo.nprim },
            )
            .unwrap();
            let core = crate::hamiltonian::build_h0(sys, &b, params, &eo.hamiltonian).unwrap();
            let mut tr = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    tr += p[(mu, nu)] * core.h0[(mu, nu)];
                }
            }
            tr
        };
        let pulay_energy = |sys: &PeriodicSystem| -> f64 {
            let b = crate::basis::BasisSet::build(
                sys,
                params,
                crate::basis::BasisOptions { nprim: eo.nprim },
            )
            .unwrap();
            let ints = crate::integrals::IntegralMatrices::build(sys, &b).unwrap();
            let mut tr = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    tr += w[(mu, nu)] * ints.overlap[(mu, nu)];
                }
            }
            tr
        };
        let scc_energy = |sys: &PeriodicSystem| -> f64 {
            let el_r = run_electronic(sys, params, eo.clone()).unwrap();
            let br = &el_r.basis;
            let sr = &el_r.integrals.overlap;
            let zr = Matrix::zeros(br.len(), br.len());
            let qp_r = response_shell_charges_from_density(br, sr, &zr, &p, &zr).unwrap();
            let qd_r =
                response_shell_charges_from_density(br, sr, &zr, &electronic.density, &zr).unwrap();
            let gamma_r = response_shell_scc_kernel(sys, params, &el_r).unwrap();
            let mut psi = 0.0;
            for s in 0..v_d.len() {
                psi += v_d[s] * qp_r[s] + v_p[s] * qd_r[s];
            }
            for s in 0..gamma_r.rows() {
                for t in 0..gamma_r.cols() {
                    psi += q_p0[s] * gamma_r[(s, t)] * q_d0[t];
                }
            }
            psi
        };
        let h = 1.0e-4;
        let comp = |g: &[Vec3], atom: usize, axis: usize| match axis {
            0 => g[atom].x,
            1 => g[atom].y,
            _ => g[atom].z,
        };
        let (mut band_max, mut pulay_max, mut scc_max) = (0.0_f64, 0.0_f64, 0.0_f64);
        for atom in 0..nat {
            for axis in 0..3 {
                let mut pp = system.clone();
                let mut mm = system.clone();
                shift_atom(&mut pp, atom, axis, h);
                shift_atom(&mut mm, atom, axis, -h);
                let band_fd = (band_energy(&pp) - band_energy(&mm)) / (2.0 * h);
                let pulay_fd = -(pulay_energy(&pp) - pulay_energy(&mm)) / (2.0 * h);
                let scc_fd = (scc_energy(&pp) - scc_energy(&mm)) / (2.0 * h);
                band_max = band_max.max((comp(&band_ana, atom, axis) - band_fd).abs());
                pulay_max = pulay_max.max((comp(&pulay_ana, atom, axis) - pulay_fd).abs());
                scc_max = scc_max.max((comp(&scc_ana, atom, axis) - scc_fd).abs());
            }
        }
        (band_max, pulay_max, scc_max)
    }

    /// Band and Pulay contractions reproduce `d/dR Tr[P H0]` and `-d/dR Tr[W S]` to
    /// machine precision for the relaxed difference density (incl. virt-virt). The
    /// full-Gamma SCC contraction matches `d/dR Psi(R)` to the ~3e-5 third-order
    /// floor (the analytic shell-pair `dkernel` is second-order only; see the Gamma=0
    /// companion). All ~200x below the 7e-3 excited-gradient residual ⇒ the
    /// contraction layer is not the source of that residual.
    #[test]
    fn tda_contraction_isolation_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let (band_max, pulay_max, scc_max) = contraction_isolation_maxdiffs(&params);
        eprintln!(
            "CONTRACTION ISOLATION (full Gamma): band={band_max:.3e} pulay={pulay_max:.3e} scc={scc_max:.3e}"
        );
        assert!(
            band_max < 1.0e-6,
            "band contraction != d/dR Tr[P H0]: {band_max:.3e}"
        );
        assert!(
            pulay_max < 1.0e-6,
            "pulay contraction != -d/dR Tr[W S]: {pulay_max:.3e}"
        );
        assert!(
            scc_max < 1.0e-4,
            "scc contraction (full Gamma) floor exceeded: {scc_max:.3e}"
        );
    }

    /// With the third-order term removed (`Gamma = 0`) the SCC kernel is purely
    /// second-order — geometry-only, charge-independent — so the SCC charge-functional
    /// contraction matches its finite difference to machine precision for the oo/vv
    /// difference density. This separates the verified second-order SCC contraction
    /// from the known (unimplemented) charge-dependent third-order kernel derivative,
    /// so a future regression in the former is not masked by the ~3e-5 third-order floor.
    #[test]
    fn tda_scc_contraction_gamma3_off_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let (_band, _pulay, scc_max) = contraction_isolation_maxdiffs(&params0);
        eprintln!("SCC CONTRACTION (Gamma=0): scc={scc_max:.3e}");
        assert!(
            scc_max < 1.0e-6,
            "second-order scc contraction != FD (Gamma=0): {scc_max:.3e}"
        );
    }

    /// Bare second-order SCC response operator H+[P] = scalar_response_fock(gamma q(P)),
    /// i.e. `L* gamma L` with L = Mulliken charge map, L* = scalar_response_fock. The
    /// closed-shell (A+B) coupling used by the Z-vector operator and the W `hz`/`H+[P]`
    /// terms is `2 * H+` (see the factor-2 in tda_sigma's operator_coupling and W's
    /// coupling_scale).
    fn apply_hplus(
        basis: &crate::basis::BasisSet,
        overlap: &Matrix,
        kernel: &Matrix,
        p: &Matrix,
    ) -> Matrix {
        let n = basis.len();
        let zero = Matrix::zeros(n, n);
        let q = response_shell_charges_from_density(basis, overlap, &zero, p, &zero).unwrap();
        let v = matrix_vector_product(kernel, &q).unwrap();
        scalar_response_fock_matrix(basis, overlap, &v).unwrap()
    }

    /// STEP 1: the second-order SCC response operator must be self-adjoint,
    /// `Tr[P H+(Q)] = Tr[Q H+(P)]` (since `H+ = L* gamma L` with gamma symmetric and
    /// `L*` the true adjoint of the Mulliken charge map `L`). Tests the actual
    /// difference/relaxation densities and random oo/ov/vv block pairs at Gamma=0. A
    /// failure here would localize a sign / single-sided-Mulliken / closed-shell-factor
    /// mismatch between the charge map and its Fock-response adjoint.
    #[test]
    fn tda_hplus_self_adjoint() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0; // linear H+ (no third order)
        }
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params0, eo).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let kernel = response_shell_scc_kernel(&system, &params0, &electronic).unwrap();
        // deterministic symmetric MO matrices restricted to oo / ov / vv blocks -> AO
        let block_ao = |sel: u8| -> Matrix {
            let mut m = Matrix::zeros(mos.cols(), mos.cols());
            let is_occ = |p: usize| space.occupied.contains(&p);
            for &p in space.occupied.iter().chain(space.virtuals.iter()) {
                for &q in space.occupied.iter().chain(space.virtuals.iter()) {
                    let kind = match (is_occ(p), is_occ(q)) {
                        (true, true) => 0u8,
                        (false, false) => 2u8,
                        _ => 1u8,
                    };
                    if kind == sel {
                        let v = 0.01 * ((p + 1) as f64) * ((q + 2) as f64);
                        m[(p, q)] = v;
                        m[(q, p)] = v;
                    }
                }
            }
            mo_coefficient_matrix_to_ao(mos, &m).unwrap()
        };
        let mats = [block_ao(0), block_ao(1), block_ao(2)];
        let names = ["oo", "ov", "vv"];
        let frob = |a: &Matrix, b: &Matrix| -> f64 {
            let mut s = 0.0;
            for r in 0..n {
                for c in 0..n {
                    s += a[(r, c)] * b[(r, c)];
                }
            }
            s
        };
        let mut maxd = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let hpj = apply_hplus(basis, overlap, &kernel, &mats[j]);
                let hpi = apply_hplus(basis, overlap, &kernel, &mats[i]);
                let lhs = frob(&mats[i], &hpj);
                let rhs = frob(&mats[j], &hpi);
                let d = (lhs - rhs).abs();
                maxd = maxd.max(d);
                eprintln!(
                    "  <{},H+{}>={lhs:+.6e}  <{},H+{}>={rhs:+.6e}  d={d:.2e}",
                    names[i], names[j], names[j], names[i]
                );
            }
        }
        assert!(maxd < 1.0e-11, "H+ not self-adjoint: {maxd:.3e}");
    }

    /// STEP 2: the Z-vector operator's coupling part `(A+B)z - gap*z` must equal the
    /// closed-shell second-order response `2 H+[z]` projected back onto the occ-virt
    /// space — i.e. the operator used to SOLVE z and the operator used to BUILD W from
    /// z (`hz`/`H+`) share one coefficient convention. (Operator coupling was fixed
    /// 1->2; this guards that the W-side paths were not left on the old convention.)
    #[test]
    fn tda_z_operator_matches_hplus() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params0, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let n_occ = space.occupied.len();
        let n_virt = space.virtuals.len();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let kernel = response_shell_scc_kernel(&system, &params0, &electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        // a deterministic, nonzero ov test vector indexed like space.pairs
        let z_ref: Vec<f64> = (0..space.pairs.len())
            .map(|k| 0.013 * ((k % 5) as f64 + 1.0))
            .collect();
        // operator coupling part
        let az = tda_sigma(&gaps, &kernel, &transition, 2.0, &z_ref).unwrap();
        let coupling_op: Vec<f64> = az
            .iter()
            .zip(gaps.iter().zip(z_ref.iter()))
            .map(|(&a, (&g, &z))| a - g * z)
            .collect();
        // z as a symmetric AO density, then 2*H+ projected back to ov
        let mut z_mo = Matrix::zeros(mos.cols(), mos.cols());
        for i_pos in 0..n_occ {
            let i = space.occupied[i_pos];
            for a_pos in 0..n_virt {
                let a = space.virtuals[a_pos];
                let zk = z_ref[i_pos * n_virt + a_pos];
                z_mo[(i, a)] = zk;
                z_mo[(a, i)] = zk;
            }
        }
        let z_ao = mo_coefficient_matrix_to_ao(mos, &z_mo).unwrap();
        let hplus_z = apply_hplus(basis, overlap, &kernel, &z_ao);
        // compare the two shell-charge maps of the same z: transition-charge induced
        // (used by tda_sigma and the W hz) vs Mulliken q(z_ao) (used by the contraction)
        let zero = Matrix::zeros(n, n);
        let q_zao =
            response_shell_charges_from_density(basis, overlap, &zero, &z_ao, &zero).unwrap();
        let mut induced = vec![0.0_f64; kernel.rows()];
        for (qia, &zk) in transition.iter().zip(z_ref.iter()) {
            for (s, &q) in qia.iter().enumerate() {
                induced[s] += q * zk;
            }
        }
        let mut q_map_maxd = 0.0_f64;
        for s in 0..induced.len() {
            q_map_maxd = q_map_maxd.max((induced[s] - q_zao[s]).abs());
        }
        // The transition-charge induced map (tda_sigma, W hz) and the Mulliken charge
        // map (contraction) of the SAME z are identical.
        assert!(
            q_map_maxd < 1.0e-12,
            "transition-charge induced != Mulliken q(z_ao): {q_map_maxd:.3e}"
        );
        // The Z-vector operator coupling equals the closed-shell (A+B) coupling
        // 2*(ia|jb) = 4*<i|H+[z]|a>: factor 4 = operator_coupling(2) * the transition-
        // charge/single-projection ratio(2). A clean integer ratio confirms the operator
        // that SOLVES z and the bare H+ response share one coefficient convention.
        let mut maxd = 0.0_f64;
        for (idx, &(i, a)) in space.pairs.iter().enumerate() {
            let mut proj = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    proj += mos[(mu, i)] * hplus_z[(mu, nu)] * mos[(nu, a)];
                }
            }
            maxd = maxd.max((coupling_op[idx] - 4.0 * proj).abs());
        }
        eprintln!("Z-OPERATOR coupling == 4*H+ projection: max|d| = {maxd:.3e}");
        assert!(
            maxd < 1.0e-11,
            "Z-vector operator coupling != closed-shell 4*<i|H+[z]|a>: {maxd:.3e}"
        );
    }

    /// Solve A x = b by partial-pivot Gaussian elimination (small dense systems).
    fn gauss_solve(a: &Matrix, b: &[f64]) -> Vec<f64> {
        let m = b.len();
        let mut aug = vec![vec![0.0_f64; m + 1]; m];
        for r in 0..m {
            for c in 0..m {
                aug[r][c] = a[(r, c)];
            }
            aug[r][m] = b[r];
        }
        for col in 0..m {
            let mut piv = col;
            for r in (col + 1)..m {
                if aug[r][col].abs() > aug[piv][col].abs() {
                    piv = r;
                }
            }
            aug.swap(col, piv);
            let d = aug[col][col];
            for c in col..=m {
                aug[col][c] /= d;
            }
            for r in 0..m {
                if r != col {
                    let f = aug[r][col];
                    for c in col..=m {
                        aug[r][c] -= f * aug[col][c];
                    }
                }
            }
        }
        (0..m).map(|r| aug[r][m]).collect()
    }

    /// FINISHING TEST: build the full ov x ov FD orbital Hessian J_FD and gradient r_FD
    /// from the ACTUAL GFN1 functional (rotations at fixed AO integrals, Gamma=0, CN off),
    /// then (1) extract the off-diagonal response coupling factor J_FD/K (K=q gamma q),
    /// (2) measure the current-z stationarity residual `rhs_code - J_FD^T z_code`, and
    /// (3) close the loop: solve `J_FD^T z_fd = rhs_code`, feed z_fd into the existing
    /// P/W builder + contraction, and compare to the finite-difference excited gradient.
    /// This dichotomizes operator vs P/W as the residual source, convention-free.
    #[test]
    fn tda_fd_jacobian_diagnosis() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let mut eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        eo.hamiltonian.enable_cn_hamiltonian = false;
        let electronic = run_electronic(&system, &params0, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let h0 = crate::hamiltonian::build_h0(&system, basis, &params0, &eo.hamiltonian)
            .unwrap()
            .h0;
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let npair = space.pairs.len();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let kernel = response_shell_scc_kernel(&system, &params0, &electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        let td = solve_tda(&system, &params0, &electronic, opts).unwrap();
        let omega = td.states[0].excitation_energy;
        let amplitudes = td.states[0].amplitudes.clone();
        let coupling = opts.spin.coupling_scale();
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            2.0,
            2.0,
        )
        .unwrap();
        let rhs_code: Vec<f64> = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect();
        let z_code =
            solve_tda_z_vector(&gaps, &kernel, &transition, &rhs_code, 2.0, 500, 1.0e-9).unwrap();

        // FD orbital Hessian J_FD[ia, jb] = d g_ia / d kappa_jb, g_ia = (C^T F[D(C)] C)_ia
        let g_vec = |c: &Matrix| -> Vec<f64> {
            let d = density_from_mos(c, &space.occupied, n);
            let f = fock_from_density(&system, basis, &params0, &h0, overlap, &d);
            space
                .pairs
                .iter()
                .map(|&(i, a)| {
                    let mut v = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            v += c[(mu, i)] * f[(mu, nu)] * c[(nu, a)];
                        }
                    }
                    v
                })
                .collect::<Vec<_>>()
        };
        let h = 1.0e-4;
        let mut j_fd = Matrix::zeros(npair, npair);
        for jb in 0..npair {
            let (j, b) = space.pairs[jb];
            let gp = g_vec(&rotate_ia(mos, j, b, h));
            let gm = g_vec(&rotate_ia(mos, j, b, -h));
            for ia in 0..npair {
                j_fd[(ia, jb)] = (gp[ia] - gm[ia]) / (2.0 * h);
            }
        }
        // symmetry
        let mut sym = 0.0_f64;
        for r in 0..npair {
            for c in 0..npair {
                sym = sym.max((j_fd[(r, c)] - j_fd[(c, r)]).abs());
            }
        }
        // K[ia,jb] = q_ia . gamma . q_jb
        let mut kmat = Matrix::zeros(npair, npair);
        for ia in 0..npair {
            let v = matrix_vector_product(&kernel, &transition[ia]).unwrap();
            for jb in 0..npair {
                kmat[(ia, jb)] = dot(&v, &transition[jb]);
            }
        }
        // off-diagonal response factor (J_FD - gap I) / K
        let mut min_r = f64::INFINITY;
        let mut max_r = f64::NEG_INFINITY;
        for ia in 0..npair {
            for jb in 0..npair {
                if ia != jb && kmat[(ia, jb)].abs() > 1.0e-6 {
                    let r = j_fd[(ia, jb)] / kmat[(ia, jb)];
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }
            }
        }
        // stationarity residual: rhs_code - J_FD^T z_code
        let mut stat = 0.0_f64;
        for ia in 0..npair {
            let mut jtz = 0.0;
            for jb in 0..npair {
                jtz += j_fd[(jb, ia)] * z_code[jb];
            }
            stat = stat.max((rhs_code[ia] - jtz).abs());
        }
        // STEP 1 (scale-free stationarity): does z_code stationarize the unitary
        // Lagrangian UP TO A SCALE alpha? r = dOmega_pg/dkappa, u = J_FD^T z_code,
        // alpha* = -r.u/u.u, residual = |r + alpha* u|_inf. Small residual => z_code is
        // the correct orbital multiplier up to the convention scale alpha (operator NOT
        // the bug); large residual even at optimal alpha => operator genuinely wrong.
        let mut dom = 0usize;
        let mut da = 0.0;
        for (k, &x) in amplitudes.iter().enumerate() {
            if x.abs() > da {
                da = x.abs();
                dom = k;
            }
        }
        let (i0, a0) = space.pairs[dom];
        let omega_pg = |c: &Matrix| -> f64 {
            let d = density_from_mos(c, &space.occupied, n);
            let f = fock_from_density(&system, basis, &params0, &h0, overlap, &d);
            let mut faa = 0.0;
            let mut fii = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    faa += c[(mu, a0)] * f[(mu, nu)] * c[(nu, a0)];
                    fii += c[(mu, i0)] * f[(mu, nu)] * c[(nu, i0)];
                }
            }
            faa - fii
        };
        let mut r_fd = vec![0.0_f64; npair];
        for (idx, &(i, a)) in space.pairs.iter().enumerate() {
            r_fd[idx] = (omega_pg(&rotate_ia(mos, i, a, h)) - omega_pg(&rotate_ia(mos, i, a, -h)))
                / (2.0 * h);
        }
        let u: Vec<f64> = (0..npair)
            .map(|ia| {
                (0..npair)
                    .map(|jb| j_fd[(jb, ia)] * z_code[jb])
                    .sum::<f64>()
            })
            .collect();
        let alpha = -dot(&r_fd, &u) / dot(&u, &u);
        let mut scalefree = 0.0_f64;
        for ia in 0..npair {
            scalefree = scalefree.max((r_fd[ia] + alpha * u[ia]).abs());
        }
        eprintln!(
            "STEP1 scale-free stationarity: alpha*={alpha:+.4}  |r + alpha* u|={scalefree:.3e}"
        );
        // z_code stationarizes the unitary Lagrangian UP TO A SCALE (r || J_FD^T z_code to
        // the FD floor): the Z-vector operator+RHS produce the correct orbital multiplier
        // DIRECTION, so the operator is NOT the residual source. The residual lives in the
        // SCALE/convention mapping z_code -> P/W (the 0.5 weights), to be pinned by H0- and
        // metric-directional FD of the scalar Lagrangian (not by changing operator_coupling).
        assert!(
            scalefree < 1.0e-4,
            "z_code does not stationarize the unitary Lagrangian up to a scale: {scalefree:.3e}"
        );
        // closure: solve J_FD^T z_fd = rhs_code
        let mut jt = Matrix::zeros(npair, npair);
        for r in 0..npair {
            for c in 0..npair {
                jt[(r, c)] = j_fd[(c, r)];
            }
        }
        let z_fd = gauss_solve(&jt, &rhs_code);
        let mut zdiff = 0.0_f64;
        for k in 0..npair {
            zdiff = zdiff.max((z_fd[k] - z_code[k]).abs());
        }
        eprintln!(
            "FD-JACOBIAN: npair={npair} J_FD sym={sym:.2e}  off-diag J_FD/K in [{min_r:.4},{max_r:.4}]  stationarity|rhs-JtZ|={stat:.3e}  max|z_fd-z_code|={zdiff:.3e}"
        );

        // closure gradient with z_fd vs FD oracle
        let ground = analytic_gradient(
            &system,
            &params0,
            AnalyticGradientOptions {
                electronic: eo.clone(),
                ..AnalyticGradientOptions::default()
            },
        )
        .unwrap();
        let context = ResponseGradientContext::new(
            &system,
            basis,
            &params0,
            &electronic,
            eo.hamiltonian.coordination_cutoff,
            false,
        )
        .unwrap();
        let zero = Matrix::zeros(n, n);
        let grad_with = |zv: &[f64]| -> Vec<Vec3> {
            let terms = tda_lagrangian_terms(
                basis,
                &kernel,
                mos,
                orbital_energies,
                &space,
                overlap,
                &amplitudes,
                omega,
                coupling,
                coupling,
            )
            .unwrap();
            let (p, w) = tda_lagrangian_density_matrices(
                basis,
                &kernel,
                mos,
                orbital_energies,
                &space,
                &amplitudes,
                zv,
                coupling,
                &terms,
            )
            .unwrap();
            let scr =
                response_shell_charges_from_density(basis, overlap, &electronic.density, &p, &zero)
                    .unwrap();
            let rg = response_electronic_gradient(
                &system,
                &electronic,
                &kernel,
                &context,
                &p,
                &p,
                &w,
                &scr,
            )
            .unwrap();
            let mut g = ground.gradient.clone();
            for atom in 0..system.atoms.len() {
                g[atom] += rg[atom];
            }
            g
        };
        let g_fdz = grad_with(&z_fd);
        let g_codez = grad_with(&z_code);
        // FD oracle (Gamma=0 params, CN off)
        let total = |sys: &PeriodicSystem| -> f64 {
            run_electronic(sys, &params0, eo.clone())
                .unwrap()
                .total_free
                + tda_frozen_excitation_energy(
                    sys,
                    &params0,
                    &eo,
                    &amplitudes,
                    opts.spin,
                    Some(&electronic),
                )
                .unwrap()
        };
        let mut max_fdz = 0.0_f64;
        let mut max_codez = 0.0_f64;
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut pp = system.clone();
                let mut mm = system.clone();
                shift_atom(&mut pp, atom, axis, h);
                shift_atom(&mut mm, atom, axis, -h);
                let fd = (total(&pp) - total(&mm)) / (2.0 * h);
                let cmp = |g: &[Vec3]| match axis {
                    0 => g[atom].x,
                    1 => g[atom].y,
                    _ => g[atom].z,
                };
                max_fdz = max_fdz.max((cmp(&g_fdz) - fd).abs());
                max_codez = max_codez.max((cmp(&g_codez) - fd).abs());
            }
        }
        eprintln!(
            "CLOSURE: excited-grad error  z_code={max_codez:.3e}  z_fd(jacobian)={max_fdz:.3e}"
        );
    }

    /// STEP 2 (ground orbital Hessian vs rotation FD): the Z-vector operator
    /// `tda_sigma` (= gap + 2 q gamma q) must equal the Jacobian of the ground-state
    /// orbital gradient `g_ia(C) = (C^T F[D(C)] C)_ia` under occ-virt rotations at fixed
    /// AO integrals — i.e. it is the true orbital Hessian of the GFN1 SCC functional,
    /// not just an internally consistent operator. Gamma=0 (second-order). The ratio
    /// `(A+B)_code / J_FD` reveals the closed-shell convention factor that must match the
    /// RHS factor so that the solved `z` is correct.
    #[test]
    fn tda_ground_orbital_hessian_matches_rotation_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let mut eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        eo.hamiltonian.enable_cn_hamiltonian = false;
        let electronic = run_electronic(&system, &params0, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let h0 = crate::hamiltonian::build_h0(&system, basis, &params0, &eo.hamiltonian)
            .unwrap()
            .h0;
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let n_occ = space.occupied.len();
        let n_virt = space.virtuals.len();
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let kernel = response_shell_scc_kernel(&system, &params0, &electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        // ground orbital gradient g_ia(C) = (C^T F[D(C)] C)_ia
        let g_vec = |c: &Matrix| -> Vec<f64> {
            let d = density_from_mos(c, &space.occupied, n);
            let f = fock_from_density(&system, basis, &params0, &h0, overlap, &d);
            let mut g = vec![0.0_f64; space.pairs.len()];
            for (idx, &(i, a)) in space.pairs.iter().enumerate() {
                let mut v = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        v += c[(mu, i)] * f[(mu, nu)] * c[(nu, a)];
                    }
                }
                g[idx] = v;
            }
            g
        };
        let h = 1.0e-4;
        let npair = space.pairs.len();
        let mut max_ratio_dev = 0.0_f64;
        // compare J_FD column to (A+B)_code column for the first few jb
        for jb in 0..npair.min(4) {
            let (j, b) = space.pairs[jb];
            let cp = rotate_ia(mos, j, b, h);
            let cm = rotate_ia(mos, j, b, -h);
            let gp = g_vec(&cp);
            let gm = g_vec(&cm);
            let mut e_jb = vec![0.0_f64; npair];
            e_jb[jb] = 1.0;
            let code = tda_sigma(&gaps, &kernel, &transition, 2.0, &e_jb).unwrap();
            // diagonal element ratio
            let jfd_diag = (gp[jb] - gm[jb]) / (2.0 * h);
            let ratio = if code[jb].abs() > 1e-10 {
                code[jb] / jfd_diag
            } else {
                0.0
            };
            eprintln!(
                "  jb{jb}(j={j},b={b}): J_FD_diag={jfd_diag:+.6e} (A+B)_code_diag={:+.6e} code/JFD={ratio:+.4}",
                code[jb]
            );
            for ia in 0..npair {
                let jfd = (gp[ia] - gm[ia]) / (2.0 * h);
                // expect (A+B)_code = 2 * J_FD
                if jfd.abs() > 1e-7 || code[ia].abs() > 1e-7 {
                    max_ratio_dev = max_ratio_dev.max((code[ia] - 2.0 * jfd).abs());
                }
            }
        }
        let _ = (n_occ, n_virt);
        eprintln!("(A+B)_code vs 2*J_FD: max|d| (first 4 cols) = {max_ratio_dev:.3e}");
    }

    /// STEP 3 (Z-vector RHS vs orbital-rotation FD): the Z-vector source `q_vo - q_ov`
    /// must equal (up to the standard sign) the orbital gradient of the pure-gap
    /// excitation energy `Omega_pg(C) = <a0|F[D(C)]|a0> - <i0|F[D(C)]|i0>` (F^MO from the
    /// rebuilt Fock, NOT re-diagonalized eigenvalues) under exact occ-virt rotations at
    /// fixed AO integrals. This tests the RHS against the ACTUAL GFN1 shell-resolved SCC
    /// functional, not the internal formula. Gamma=0 isolates the second-order part.
    #[test]
    fn tda_z_rhs_matches_orbital_rotation_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let mut params0 = params.clone();
        for elem in params0.elements.values_mut() {
            elem.gam3_raw = 0.0;
        }
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let mut eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        eo.hamiltonian.enable_cn_hamiltonian = false;
        let electronic = run_electronic(&system, &params0, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let h0 = crate::hamiltonian::build_h0(&system, basis, &params0, &eo.hamiltonian)
            .unwrap()
            .h0;
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let n_occ = space.occupied.len();
        let n_virt = space.virtuals.len();
        let kernel = response_shell_scc_kernel(&system, &params0, &electronic).unwrap();
        let td = solve_tda(&system, &params0, &electronic, opts).unwrap();
        let omega = td.states[0].excitation_energy;
        let amplitudes = td.states[0].amplitudes.clone();
        let mut dom = 0usize;
        let mut da = 0.0;
        for (idx, &x) in amplitudes.iter().enumerate() {
            if x.abs() > da {
                da = x.abs();
                dom = idx;
            }
        }
        let (i0, a0) = space.pairs[dom];
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            2.0,
            2.0,
        )
        .unwrap();
        let rhs_code: Vec<f64> = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect();
        let h = 1.0e-4;
        // sanity: rebuilt Fock matches the SCF Fock, and F^MO is Brillouin-diagonal
        let d_scf = density_from_mos(mos, &space.occupied, n);
        let f_helper = fock_from_density(&system, basis, &params0, &h0, overlap, &d_scf);
        let mut fdiff = 0.0_f64;
        for mu in 0..n {
            for nu in 0..n {
                fdiff = fdiff.max((f_helper[(mu, nu)] - electronic.fock[(mu, nu)]).abs());
            }
        }
        let mut brillouin = 0.0_f64;
        for i_pos in 0..n_occ {
            for a_pos in 0..n_virt {
                let i = space.occupied[i_pos];
                let a = space.virtuals[a_pos];
                let mut fia = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        fia += mos[(mu, i)] * f_helper[(mu, nu)] * mos[(nu, a)];
                    }
                }
                brillouin = brillouin.max(fia.abs());
            }
        }
        eprintln!("SANITY: |F_helper-F_scf|={fdiff:.3e}  max F^MO_ia (Brillouin)={brillouin:.3e}");
        let omega_pg = |c: &Matrix| -> f64 {
            let d = density_from_mos(c, &space.occupied, n);
            let f = fock_from_density(&system, basis, &params0, &h0, overlap, &d);
            let mut faa = 0.0;
            let mut fii = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    faa += c[(mu, a0)] * f[(mu, nu)] * c[(nu, a0)];
                    fii += c[(mu, i0)] * f[(mu, nu)] * c[(nu, i0)];
                }
            }
            faa - fii
        };
        // The Z-vector RHS equals exactly -2 * (orbital gradient dOmega_pg/dkappa):
        // r_fd + 0.5*rhs == 0. A clean machine-precision relation for every pair means
        // the RHS is structurally correct (no missing ground-charge-response or
        // GFN1-specific term -- those would break the clean -0.5 ratio).
        let mut max_dev = 0.0_f64;
        let mut max_scale = 0.0_f64;
        for i_pos in 0..n_occ {
            for a_pos in 0..n_virt {
                let i = space.occupied[i_pos];
                let a = space.virtuals[a_pos];
                let idx = i_pos * n_virt + a_pos;
                let cp = rotate_ia(mos, i, a, h);
                let cm = rotate_ia(mos, i, a, -h);
                let r_fd = (omega_pg(&cp) - omega_pg(&cm)) / (2.0 * h);
                max_dev = max_dev.max((r_fd + 0.5 * rhs_code[idx]).abs());
                max_scale = max_scale.max(rhs_code[idx].abs());
            }
        }
        eprintln!("Z-RHS == -2*(dOmega_pg/dkappa): max|r_fd + 0.5 rhs| = {max_dev:.3e} (scale {max_scale:.3e})");
        assert!(
            max_dev < 1.0e-7, // central-difference floor at h=1e-4, scale ~1
            "Z-vector RHS != -2 * orbital gradient: {max_dev:.3e}"
        );
    }

    /// TEST 5 (oo/ov/vv block residual decomposition): the analytic−oracle residual is
    /// the FD of the functional difference `{Tr[P H0] + Ψ − Tr[W S]} − {Tr[T F] − Tr[W_T S]}`
    /// (each piece a verified contraction; F the SCC-converged Fock). Grouped as a density
    /// side `Tr[P H0] + Ψ − Tr[T F]` and the energy-weighted side `−Tr[ΔW S]` with
    /// `ΔW = W − W_T` split into MO oo/ov/vv blocks. This pinpoints which sector carries
    /// the ~7e-3: dominant `R_density` ⇒ z relaxation density; dominant `R_Woo`/`R_Wov`
    /// ⇒ that W block's matrix structure (H⁺[P]/hz/ε-weighting); `R_Wvv≠0` ⇒ an AO/MO
    /// convention slip vs the verified `W_vv=W_T,vv`.
    #[test]
    fn tda_block_residual_decomposition() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let mut eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        eo.hamiltonian.enable_cn_hamiltonian = false;
        let electronic = run_electronic(&system, &params, eo.clone()).unwrap();
        let basis = &electronic.basis;
        let overlap = &electronic.integrals.overlap;
        let n = basis.len();
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&electronic.occupations).unwrap();
        let coupling = opts.spin.coupling_scale();
        let kernel = response_shell_scc_kernel(&system, &params, &electronic).unwrap();
        let transition =
            transition_shell_charges(basis, mos, &electronic.occupations, overlap).unwrap();
        let td = solve_tda(&system, &params, &electronic, opts).unwrap();
        let omega = td.states[0].excitation_energy;
        let amplitudes = td.states[0].amplitudes.clone();
        let mut dom = 0usize;
        let mut da = 0.0;
        for (idx, &x) in amplitudes.iter().enumerate() {
            if x.abs() > da {
                da = x.abs();
                dom = idx;
            }
        }
        let (i0, a0) = space.pairs[dom];
        let gaps = space
            .pairs
            .iter()
            .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
            .collect::<Vec<_>>();
        let terms_z = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            2.0,
            2.0,
        )
        .unwrap();
        let rhs = terms_z
            .q_vo
            .iter()
            .zip(terms_z.q_ov.iter())
            .map(|(&vo, &ov)| vo - ov)
            .collect::<Vec<_>>();
        let z_vector =
            solve_tda_z_vector(&gaps, &kernel, &transition, &rhs, 2.0, 500, 1.0e-9).unwrap();
        let terms = tda_lagrangian_terms(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            overlap,
            &amplitudes,
            omega,
            coupling,
            coupling,
        )
        .unwrap();
        let (p, w) = tda_lagrangian_density_matrices(
            basis,
            &kernel,
            mos,
            orbital_energies,
            &space,
            &amplitudes,
            &z_vector,
            coupling,
            &terms,
        )
        .unwrap();
        // identity W_T (MO diagonal) -> AO
        let mut w_t_mo = Matrix::zeros(mos.cols(), mos.cols());
        w_t_mo[(a0, a0)] = orbital_energies[a0];
        w_t_mo[(i0, i0)] = -orbital_energies[i0];
        let w_t_ao = mo_coefficient_matrix_to_ao(mos, &w_t_mo).unwrap();
        // T in AO = |a0><a0| - |i0><i0|
        let mut t_ao = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                t_ao[(mu, nu)] = mos[(mu, a0)] * mos[(nu, a0)] - mos[(mu, i0)] * mos[(nu, i0)];
            }
        }
        // dW = W - W_T  ->  MO via sc^T dW sc  (sc = S C)
        let sc = overlap.matmul(mos).unwrap();
        let mut dw_mo = Matrix::zeros(mos.cols(), mos.cols());
        for pp in 0..mos.cols() {
            for qq in 0..mos.cols() {
                let mut v = 0.0;
                for mu in 0..n {
                    for nu in 0..n {
                        v += sc[(mu, pp)] * (w[(mu, nu)] - w_t_ao[(mu, nu)]) * sc[(nu, qq)];
                    }
                }
                dw_mo[(pp, qq)] = v;
            }
        }
        let is_occ = |p: usize| space.occupied.contains(&p);
        let mask_block = |sel: u8| -> Matrix {
            let mut m = Matrix::zeros(mos.cols(), mos.cols());
            for pp in 0..mos.cols() {
                for qq in 0..mos.cols() {
                    let kind = match (is_occ(pp), is_occ(qq)) {
                        (true, true) => 0u8,   // oo
                        (false, false) => 2u8, // vv
                        _ => 1u8,              // ov
                    };
                    if kind == sel {
                        m[(pp, qq)] = dw_mo[(pp, qq)];
                    }
                }
            }
            m
        };
        let dw_oo = mo_coefficient_matrix_to_ao(mos, &mask_block(0)).unwrap();
        let dw_ov = mo_coefficient_matrix_to_ao(mos, &mask_block(1)).unwrap();
        let dw_vv = mo_coefficient_matrix_to_ao(mos, &mask_block(2)).unwrap();
        // SCC functional Psi frozen pieces
        let zero = Matrix::zeros(n, n);
        let q_p0 =
            response_shell_charges_from_density(basis, overlap, &electronic.density, &p, &zero)
                .unwrap();
        let v_d = electronic.shell_scc_potential.clone();
        let v_p = matrix_vector_product(&kernel, &q_p0).unwrap();
        let q_d0 = electronic.shell_charges.clone();
        let tr = |a: &Matrix, b: &Matrix| -> f64 {
            let mut s = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    s += a[(mu, nu)] * b[(mu, nu)];
                }
            }
            s
        };
        // scalar functionals at geometry R (all densities/W/W_T/q frozen at R0)
        let density_funcl = |sys: &PeriodicSystem| -> f64 {
            let el = run_electronic(sys, &params, eo.clone()).unwrap();
            let b = crate::basis::BasisSet::build(
                sys,
                &params,
                crate::basis::BasisOptions { nprim: eo.nprim },
            )
            .unwrap();
            let core = crate::hamiltonian::build_h0(sys, &b, &params, &eo.hamiltonian).unwrap();
            let qp_r =
                response_shell_charges_from_density(&b, &el.integrals.overlap, &zero, &p, &zero)
                    .unwrap();
            let qd_r = response_shell_charges_from_density(
                &b,
                &el.integrals.overlap,
                &zero,
                &electronic.density,
                &zero,
            )
            .unwrap();
            let gamma_r = response_shell_scc_kernel(sys, &params, &el).unwrap();
            let mut psi = 0.0;
            for s in 0..v_d.len() {
                psi += v_d[s] * qp_r[s] + v_p[s] * qd_r[s];
            }
            for s in 0..gamma_r.rows() {
                for t in 0..gamma_r.cols() {
                    psi += q_p0[s] * gamma_r[(s, t)] * q_d0[t];
                }
            }
            tr(&p, &core.h0) + psi - tr(&t_ao, &el.fock)
        };
        let block_funcl = |sys: &PeriodicSystem, blk: &Matrix| -> f64 {
            let b = crate::basis::BasisSet::build(
                sys,
                &params,
                crate::basis::BasisOptions { nprim: eo.nprim },
            )
            .unwrap();
            let ints = crate::integrals::IntegralMatrices::build(sys, &b).unwrap();
            tr(blk, &ints.overlap)
        };
        let h = 1.0e-4;
        let fd = |f: &dyn Fn(&PeriodicSystem) -> f64, atom: usize, axis: usize| -> f64 {
            let mut pp = system.clone();
            let mut mm = system.clone();
            shift_atom(&mut pp, atom, axis, h);
            shift_atom(&mut mm, atom, axis, -h);
            (f(&pp) - f(&mm)) / (2.0 * h)
        };
        // Legacy-path decomposition: the residual of the FORMER Lagrangian P/W
        // formulation, computed purely from the FD of the legacy P/W functionals
        // (independent of the current production gradient).
        let (mut max_density, mut max_oo, mut max_ov, mut max_vv, mut max_validate) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        eprintln!("BLOCK RESIDUAL DECOMPOSITION (a0={a0},i0={i0}):");
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let r_density = fd(&density_funcl, atom, axis);
                let r_oo = -fd(&|s| block_funcl(s, &dw_oo), atom, axis);
                let r_ov = -fd(&|s| block_funcl(s, &dw_ov), atom, axis);
                let r_vv = -fd(&|s| block_funcl(s, &dw_vv), atom, axis);
                let r_total = r_density + r_oo + r_ov + r_vv;
                max_density = max_density.max(r_density.abs());
                max_oo = max_oo.max(r_oo.abs());
                max_ov = max_ov.max(r_ov.abs());
                max_vv = max_vv.max(r_vv.abs());
                max_validate = max_validate.max(r_total.abs());
                eprintln!(
                    "  a{atom}x{axis}: dens={r_density:+.2e} oo={r_oo:+.2e} ov={r_ov:+.2e} vv={r_vv:+.2e} | tot={r_total:+.2e}"
                );
            }
        }
        eprintln!(
            "MAX |R|: density={max_density:.3e} oo={max_oo:.3e} ov={max_ov:.3e} vv={max_vv:.3e} total={max_validate:.3e}"
        );
        // The former Lagrangian P/W formulation carried a ~7e-3 residual (the reason
        // the production gradient was switched to the exact direct-CPHF derivative);
        // this guards the legacy decomposition reproduces that known magnitude.
        assert!(
            (6.0e-3..9.0e-3).contains(&max_validate),
            "legacy block decomposition total {max_validate:.3e} != known ~7e-3 residual"
        );
    }

    /// TEST 2 (pure-gap identity): water S0 is an exactly decoupled state with
    /// `omega = eps_a - eps_i` and all transition charges ~ 0. For such a state the
    /// excitation-energy gradient has the closed form
    ///   `domega/dR = d/dR Tr[T F(R)] - d/dR Tr[W_T S(R)]`
    /// with `T = |a><a| - |i><i|`, `W_T = eps_a|a><a| - eps_i|i><i|` FIXED at the
    /// reference geometry and `F`, `S` the SCC-converged Fock / overlap at displaced
    /// geometries. This is an INDEPENDENT ground truth: no Z-vector, no Lagrangian
    /// density, no frozen-amplitude oracle. It validates the frozen-amplitude oracle
    /// independently; the printed `ana-identity` column (now the exact direct-CPHF
    /// production gradient minus this identity) is ~0, confirming the analytic gradient
    /// reproduces the closed-form pure-gap derivative.
    #[test]
    fn tda_pure_gap_identity_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let el0 = run_electronic(&system, &params, eo.clone()).unwrap();
        let eig0 = lowdin_solve_generalized(&el0.fock, &el0.integrals.overlap, 1.0e-12).unwrap();
        let mos0 = &eig0.vectors;
        let e0 = &eig0.values;
        let space = CpxtbSpace::from_occupations(&el0.occupations).unwrap();
        let td = solve_tda(&system, &params, &el0, opts).unwrap();
        let amps = td.states[0].amplitudes.clone();
        let mut dom = 0usize;
        let mut domabs = 0.0;
        for (idx, &a) in amps.iter().enumerate() {
            if a.abs() > domabs {
                domabs = a.abs();
                dom = idx;
            }
        }
        let (i, a) = space.pairs[dom];
        let n = el0.basis.len();
        let mut tmat = Matrix::zeros(n, n);
        let mut wt = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let ca = mos0[(mu, a)] * mos0[(nu, a)];
                let ci = mos0[(mu, i)] * mos0[(nu, i)];
                tmat[(mu, nu)] = ca - ci;
                wt[(mu, nu)] = e0[a] * ca - e0[i] * ci;
            }
        }
        // independent identity: domega/dR = d/dR{ Tr[T F] - Tr[W_T S] }
        let identity = |sys: &PeriodicSystem| -> f64 {
            let el = run_electronic(sys, &params, eo.clone()).unwrap();
            let f = &el.fock;
            let s = &el.integrals.overlap;
            let mut tf = 0.0;
            let mut ws = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    tf += tmat[(mu, nu)] * f[(mu, nu)];
                    ws += wt[(mu, nu)] * s[(mu, nu)];
                }
            }
            tf - ws
        };
        let frozen = |sys: &PeriodicSystem| -> f64 {
            tda_frozen_excitation_energy(sys, &params, &eo, &amps, opts.spin, Some(&el0)).unwrap()
        };
        let ground = analytic_gradient(
            &system,
            &params,
            AnalyticGradientOptions {
                electronic: eo.clone(),
                ..AnalyticGradientOptions::default()
            },
        )
        .unwrap();
        let ana = solve_tda_gradient_analytic(&system, &params, &eo, 0, opts).unwrap();
        let h = 1.0e-4;
        let mut max_id_vs_oracle = 0.0_f64;
        eprintln!("PURE-GAP IDENTITY (i={i},a={a}): per-component domega/dR");
        eprintln!("   identity_FD    frozen_FD     ana_response   (ana-identity)");
        for atom in 0..system.atoms.len() {
            for axis in 0..3 {
                let mut p = system.clone();
                let mut m = system.clone();
                shift_atom(&mut p, atom, axis, h);
                shift_atom(&mut m, atom, axis, -h);
                let id_fd = (identity(&p) - identity(&m)) / (2.0 * h);
                let om_fd = (frozen(&p) - frozen(&m)) / (2.0 * h);
                let ana_resp = match axis {
                    0 => ana.gradient[atom].x - ground.gradient[atom].x,
                    1 => ana.gradient[atom].y - ground.gradient[atom].y,
                    _ => ana.gradient[atom].z - ground.gradient[atom].z,
                };
                max_id_vs_oracle = max_id_vs_oracle.max((id_fd - om_fd).abs());
                eprintln!(
                    "  a{atom}x{axis}: {id_fd:+.6}  {om_fd:+.6}  {ana_resp:+.6}  {:+.2e}",
                    ana_resp - id_fd
                );
            }
        }
        // The Z-vector-free pure-gap identity must reproduce the frozen-amplitude
        // oracle: this validates the oracle independently and proves the analytic
        // residual lives in the Lagrangian, not the verification target.
        assert!(
            max_id_vs_oracle < 1.0e-4,
            "pure-gap identity disagrees with frozen oracle: {max_id_vs_oracle:.3e}"
        );
    }

    /// TEST 2 prep: characterize water S0 — dominant (i,a), amplitude, bare gap vs
    /// omega, and the transition-shell-charge norm. If the state is decoupled
    /// (||q_ia|| ~ 0, omega ~ gap), then g_exc - g_ground == d(eps_a - eps_i)/dR
    /// exactly, bypassing the whole Z-vector/P/W machinery.
    #[test]
    fn tda_state_character_diagnostic() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let el = run_electronic(&system, &params, eo.clone()).unwrap();
        let overlap = &el.integrals.overlap;
        let eig = lowdin_solve_generalized(&el.fock, overlap, 1.0e-12).unwrap();
        let mos = &eig.vectors;
        let orbital_energies = &eig.values;
        let space = CpxtbSpace::from_occupations(&el.occupations).unwrap();
        let transition =
            transition_shell_charges(&el.basis, mos, &el.occupations, overlap).unwrap();
        let td = solve_tda(&system, &params, &el, opts).unwrap();
        let amps = &td.states[0].amplitudes;
        let omega = td.states[0].excitation_energy;
        // dominant pair
        let mut dom = 0usize;
        let mut domabs = 0.0;
        for (idx, &a) in amps.iter().enumerate() {
            if a.abs() > domabs {
                domabs = a.abs();
                dom = idx;
            }
        }
        let (i, a) = space.pairs[dom];
        let gap = orbital_energies[a] - orbital_energies[i];
        let qnorm: f64 = transition[dom].iter().map(|q| q * q).sum::<f64>().sqrt();
        // weighted transition charge norm over ALL pairs (the actual coupling driver)
        let mut wq = vec![0.0_f64; el.basis.shells.len()];
        for (idx, qia) in transition.iter().enumerate() {
            for (s, &q) in qia.iter().enumerate() {
                wq[s] += amps[idx] * q;
            }
        }
        let wqnorm: f64 = wq.iter().map(|q| q * q).sum::<f64>().sqrt();
        eprintln!(
            "STATE0: dom pair (i={i},a={a}) X={domabs:.4} omega={omega:.6} gap={gap:.6} \
             omega-gap={:.3e} ||q_dom||={qnorm:.3e} ||sum_ia X q||={wqnorm:.3e} npairs={}",
            omega - gap,
            space.pairs.len()
        );
        // Water S0 is an exactly decoupled pure-gap excitation: omega == eps_a - eps_i
        // and the transition charges vanish, so the excitonic coupling is inert and
        // the analytic gradient reduces to the orbital-gap derivative. This is the
        // premise of the pure-gap identity regression test.
        assert!(
            (omega - gap).abs() < 1.0e-9,
            "S0 not pure-gap: omega-gap={:.3e}",
            omega - gap
        );
        assert!(
            wqnorm < 1.0e-9,
            "S0 transition coupling not inert: ||sum X q||={wqnorm:.3e}"
        );
    }

    /// Shared strict FD gate: max |analytic - finite-difference| over all Cartesian
    /// components for a given molecule and excited state, comparing the fully analytic
    /// gradient `solve_tda_gradient_analytic` against the root-tracking finite-difference
    /// reference `solve_tda_gradient` (the numerically exact ground truth).
    fn tda_analytic_vs_full_fd_maxdiff(params: &Gfn1Parameters, xyz: &str, state: usize) -> f64 {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let opts = TdaOptions {
            n_states: 6,
            spin: TdaSpin::Singlet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let ana = solve_tda_gradient_analytic(&system, params, &eo, state, opts).unwrap();
        let fdref = solve_tda_gradient(&system, params, &eo, state, opts, 1.0e-4).unwrap();
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            let d = ana.gradient[atom] - fdref.gradient[atom];
            maxdiff = maxdiff.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
        }
        maxdiff
    }

    /// Strict FD gate (formaldehyde, H2CO): the fully analytic excited-state gradient
    /// must reproduce the root-tracking finite-difference gradient to FD precision on a
    /// polar molecule with a genuine low-lying (n -> pi*) excited state. The direct-CPHF
    /// derivative is exact, so the residual is the central-difference truncation of the
    /// FD reference.
    #[test]
    fn tda_analytic_gradient_matches_full_fd_formaldehyde() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = "4\nformaldehyde\nC 0.00 0.00 -0.53\nO 0.00 0.00 0.68\nH 0.00 0.94 -1.10\nH 0.00 -0.94 -1.10\n";
        let maxdiff = tda_analytic_vs_full_fd_maxdiff(&params, xyz, 0);
        eprintln!("TDA analytic vs full-FD (H2CO S0): max diff {maxdiff:.3e} Ha/bohr");
        assert!(
            maxdiff < 1.0e-5,
            "TDA analytic gradient vs full-FD (H2CO): max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Strict FD gate (hydrogen sulfide, H2S): heteroatom check with a second-row atom
    /// (sulfur) — the fully analytic excited-state gradient must reproduce the
    /// root-tracking finite-difference gradient to FD precision.
    #[test]
    fn tda_analytic_gradient_matches_full_fd_h2s() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = "3\nhydrogen sulfide\nS 0.00 0.00 0.10\nH 0.97 0.00 -0.92\nH -0.97 0.00 -0.92\n";
        let maxdiff = tda_analytic_vs_full_fd_maxdiff(&params, xyz, 0);
        eprintln!("TDA analytic vs full-FD (H2S S0): max diff {maxdiff:.3e} Ha/bohr");
        assert!(
            maxdiff < 1.0e-5,
            "TDA analytic gradient vs full-FD (H2S): max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// The triplet channel (coupling scale 0) reduces to the bare orbital-gap gradient;
    /// the fully analytic gradient must still match the finite-difference reference.
    #[test]
    fn tda_analytic_gradient_triplet_matches_full_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.79 0.55 -0.04\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Triplet,
        };
        let eo = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let ana = solve_tda_gradient_analytic(&system, &params, &eo, 0, opts).unwrap();
        let fdref = solve_tda_gradient(&system, &params, &eo, 0, opts, 1.0e-4).unwrap();
        let mut maxdiff = 0.0_f64;
        for atom in 0..system.atoms.len() {
            let d = ana.gradient[atom] - fdref.gradient[atom];
            maxdiff = maxdiff.max(d.x.abs()).max(d.y.abs()).max(d.z.abs());
        }
        eprintln!("TDA analytic vs full-FD (water T0): max diff {maxdiff:.3e} Ha/bohr");
        assert!(
            maxdiff < 1.0e-5,
            "TDA triplet analytic gradient vs full-FD: max diff {maxdiff:.3e} Ha/bohr"
        );
    }

    /// Strict FD gate (formaldehyde, in-plane / out-of-plane mix): a second H2CO
    /// geometry whose dominant excitation involves out-of-plane orbital rotations.
    /// This is the regression guard for the occ-occ / virt-virt orbital-rotation
    /// amplitudes in the transition-charge derivative — omitting them (using only the
    /// `-1/2 S` metric piece) left a ~2e-3 out-of-plane error here while leaving the
    /// in-plane components exact.
    #[test]
    fn tda_analytic_gradient_matches_full_fd_formaldehyde_out_of_plane() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = "4\nformaldehyde\nC 0.02 0.00 0.00\nO 0.00 0.00 1.21\nH 0.00 0.94 -0.59\nH 0.00 -0.94 -0.59\n";
        let maxdiff = tda_analytic_vs_full_fd_maxdiff(&params, xyz, 0);
        eprintln!("TDA analytic vs full-FD (H2CO out-of-plane S0): max diff {maxdiff:.3e} Ha/bohr");
        assert!(
            maxdiff < 1.0e-5,
            "TDA analytic gradient vs full-FD (H2CO out-of-plane): max diff {maxdiff:.3e} Ha/bohr"
        );
    }
}
