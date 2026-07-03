// SPDX-License-Identifier: GPL-3.0-or-later
//! k-point self-consistent-charge driver for periodic GFN1-xTB.
//!
//! The zeroth-order `H0(k)` and overlap `S(k)` depend only on geometry (through
//! the periodic coordination number), so they are built once before the SCC
//! loop. Each iteration only adds the shell SCC potential shift
//! `F(k)_{mu nu} = H0(k)_{mu nu} - 1/2 (V_mu + V_nu) S(k)_{mu nu}`, diagonalises
//! the Hermitian generalized problem at every k, fills a single global Fermi
//! level over the whole Brillouin zone, and rebuilds the Mulliken shell charges.
//!
//! Energies are assembled exactly as in the molecular path: the band/H0 piece
//! `sum_k w_k Re tr[P(k) H0(k)]`, the second/third-order electrostatics from the
//! periodic Ewald `Gamma` via the shared `coulomb_energy_potential_from_matrix`,
//! the periodic repulsion, and the electronic entropy.

use crate::basis::{BasisOptions, BasisSet};
use crate::coulomb::{coulomb_energy_potential_from_matrix, ShellChargeModel};
use crate::dispersion::dispersion_energy;
use crate::electronic::{
    resolve_spin_channels, validate_electron_count, BroydenMixer, ElectronicOptions,
    ElectronicResult, SpinChannels,
};
use crate::error::{Gfn1Error, Result};
use crate::field::{electric_field_energy, electric_shell_potential, mulliken_dipole};
use crate::halogen::halogen_energy;
use crate::integrals::IntegralMatrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::model::KPoint;
use crate::params::Gfn1Parameters;
use crate::pbc::bloch::BlochBuilder;
use crate::linalg::LowdinOrthogonalizer;
use crate::pbc::complex::{
    embedded_overlap_orthogonalizer, hermitian_generalized_eigen_with_orthogonalizer,
    weighted_density, CMatrix,
};
use crate::pbc::ewald::periodic_gamma_matrix;
use crate::pbc::kpoints::{fold_time_reversal, gamma_only, monkhorst_pack};
use crate::pbc::PbcOptions;
use crate::repulsion::repulsion_energy;
use crate::system::PeriodicSystem;

const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_808_578_545_117e-6;

/// Rich result of a periodic SCC calculation, carrying everything the gradient
/// module needs (per-k densities, Bloch blocks, Ewald matrix).
#[derive(Clone, Debug)]
pub struct PbcSccResult {
    pub basis: BasisSet,
    pub bloch: BlochBuilder,
    pub gamma: Matrix,
    pub shell_model: ShellChargeModel,
    pub kpoints: Vec<KPoint>,
    /// `H0(k)` and `S(k)` per k-point (geometry-only, SCC-invariant).
    pub hs_k: Vec<(CMatrix, CMatrix)>,
    /// Physical complex density `P(k)` per k-point.
    pub density_k: Vec<CMatrix>,
    /// Energy-weighted complex density `W(k)` per k-point.
    pub ew_density_k: Vec<CMatrix>,
    pub shell_charges: Vec<f64>,
    pub atomic_charges: Vec<f64>,
    pub shell_scc_potential: Vec<f64>,
    pub coordination_numbers: Vec<f64>,
    pub fermi_level: f64,
    pub electronic_temperature: f64,
    pub electronic_energy: f64,
    pub isotropic_scc_energy: f64,
    pub third_order_energy: f64,
    pub repulsion_energy: f64,
    pub dispersion_energy: f64,
    pub halogen_energy: f64,
    /// External electric-field interaction energy `sum_i q_i v_ext_i`.
    pub external_field_energy: f64,
    pub electronic_entropy_term: f64,
    pub total_internal: f64,
    pub total_free: f64,
    /// Reference-cell Mulliken (monopole) dipole `sum_A q_A (R_A - origin)`.
    pub dipole: Vec3,
    /// Converged on-site atomic dipoles `d_A` (the rank-1 slice of [`Self::atomic_moments`]; empty/
    /// zero unless `options.multipole`). Kept as a convenience for the dipole-only consumers.
    pub atomic_dipoles: Vec<Vec3>,
    /// Converged **arbitrary-rank** on-site atomic moments `M[A][l]` (`l = 0..=L`, full `3^l`
    /// layout; rank 0 = `qm = −(atomic charge)`). Empty unless `options.multipole`. These are the
    /// SCF moment variables the analytic gradient/stress (A4/A5) consume — fed directly to
    /// [`crate::pbc::ewald_multipole::periodic_multipole_forces_generic`].
    pub atomic_moments: Vec<Vec<Vec<f64>>>,
    /// Periodic fields conjugate to `atomic_moments` at the converged SCC state.
    pub atomic_multipole_fields: Vec<Vec<Vec<f64>>>,
    /// Multipole correction energy `E_mp = ½ Σ_A Σ_l M·V` (0 unless `options.multipole`).
    pub multipole_energy: f64,
    pub nelec: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PbcSccGuess {
    pub shell_charges: Vec<f64>,
    pub atomic_moments: Vec<Vec<Vec<f64>>>,
}

impl From<&PbcSccResult> for PbcSccGuess {
    fn from(result: &PbcSccResult) -> Self {
        Self {
            shell_charges: result.shell_charges.clone(),
            atomic_moments: result.atomic_moments.clone(),
        }
    }
}

/// Build the k-point set implied by the mesh selection.
pub fn build_kpoints(system: &PeriodicSystem, mesh: crate::pbc::KMesh) -> Vec<KPoint> {
    let periodic = system
        .lattice
        .as_ref()
        .map(|l| l.periodic)
        .unwrap_or([false, false, false]);
    if mesh.is_gamma_only() {
        return gamma_only();
    }
    let pts = monkhorst_pack(mesh.size, periodic, mesh.gamma_centered);
    if mesh.fold_time_reversal {
        fold_time_reversal(&pts)
    } else {
        pts
    }
}

/// Run the periodic SCC for a `Gamma`-point or k-point mesh.
pub fn run_pbc_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcSccResult> {
    run_pbc_scc_with_guess(system, params, options, pbc, None)
}

/// Run the periodic SCC with an optional previous-step charge/moment initial guess.
pub fn run_pbc_scc_with_guess(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    guess: Option<&PbcSccGuess>,
) -> Result<PbcSccResult> {
    let _profile = crate::profile::scope("pbc.scf.total");
    if system.lattice.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "run_pbc_scc requires a periodic system with a lattice".to_string(),
        ));
    }
    let basis = BasisSet::build(
        system,
        params,
        BasisOptions {
            nprim: options.nprim,
        },
    )?;
    if basis.is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "cannot run periodic GFN1 SCC for an empty basis".to_string(),
        ));
    }
    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    let n = basis.len();
    validate_electron_count(nelec, n)?;
    let spin_channels = resolve_spin_channels(nelec, options.spin_multiplicity, n)?;

    let bloch = {
        let _p = crate::profile::scope("pbc.scf.bloch_build");
        BlochBuilder::build(
            system,
            &basis,
            params,
            pbc.ao_cutoff,
            options.hamiltonian.coordination_cutoff,
            options.hamiltonian.enable_cn_hamiltonian,
        )?
    };
    let shell_model = {
        // Apply the on-site charge-expansion order (mirrors the molecular path,
        // `electronic.rs`). The higher-order on-site term is purely per-atom — no
        // lattice sum — so it needs no change to the periodic Ewald `Gamma`; only the
        // `coulomb_energy_potential_from_matrix` call below picks it up.
        let mut m = ShellChargeModel::build(system, &basis, params)?;
        m.charge_order = options.charge_order.max(3);
        m
    };
    let gamma = {
        let _p = crate::profile::scope("pbc.scf.gamma_matrix");
        periodic_gamma_matrix(system, &basis, &shell_model, &pbc.ewald)?
    };

    let kpoints = build_kpoints(system, pbc.kmesh);
    let hs_k: Vec<(CMatrix, CMatrix)> = {
        let _p = crate::profile::scope("pbc.scf.hs_k");
        kpoints
            .iter()
            .map(|kp| bloch.h_s_at_k(kp.fractional))
            .collect()
    };

    // Löwdin orthogonaliser `X = S(k)^{-1/2}` per k-point. `S(k)` is geometry-fixed
    // (built once, above) and does NOT change during the SCC — only the Fock
    // iterate does — so this symmetric-orthogonalisation factorisation is computed
    // ONCE here and reused in every SCC iteration's generalised eigensolve, instead
    // of being rebuilt per iteration inside `hermitian_generalized_eigen`. Pure
    // speedup: the cached orthogonaliser is bit-for-bit the one the per-iteration
    // path produced (same embedded `S(k)`, same `eigen_tolerance`).
    let orth_k: Vec<LowdinOrthogonalizer> = {
        let _p = crate::profile::scope("pbc.scf.overlap_orthogonalizer");
        hs_k
            .iter()
            .map(|(_, sk)| embedded_overlap_orthogonalizer(sk, options.eigen_tolerance))
            .collect::<Result<Vec<_>>>()?
    };

    let repulsion = repulsion_energy(system, params)?;
    let dispersion = if options.enable_dispersion {
        let _p = crate::profile::scope("pbc.scf.dispersion_energy");
        dispersion_energy(system, params, options.d3_reference_path.as_deref())?
    } else {
        0.0
    };
    let halogen = {
        let _p = crate::profile::scope("pbc.scf.halogen_energy");
        halogen_energy(system)?
    };

    if options.external_field.magnetic_field.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "external magnetic field is a foothold only and not yet wired into the periodic SCC; \
             see the `magnetic` module"
                .to_string(),
        ));
    }
    // Uniform electric field as a site potential v_ext_i = -E·(R_i - origin),
    // referenced to the cell origin. This is the dipole-coupling / finite-field
    // form of the field for the reference cell; for true bulk polarization a
    // Berry-phase treatment would be required (see docs).
    let external_shell_potential =
        electric_shell_potential(&options.external_field, system, &basis);

    let kt = options.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let mixing = options.mixing.clamp(0.01, 1.0);
    let nsh = basis.shells.len();
    let nat = system.atoms.len();

    // **Arbitrary-rank** mDFTB2 multipole context (geometry-only). The atomic moments (ranks
    // `1..=L`) are mixed JOINTLY with the shell charges in one Broyden vector (tblite/GFN2 style),
    // and their periodic field (the QCore Ewald `periodic_multipole_fields_generic`: every
    // rank-pair real+reciprocal+rank-diagonal self) is rebuilt from the *mixed* state each
    // iteration so it relaxes self-consistently. `L` follows the molecular convention: explicit
    // `multipole_order ≥ 1` sets the rank; the bare `multipole` flag defaults to **dipole+
    // quadrupole** (rank 2), matching the molecular legacy path. The on-site moment AO integrals
    // and the per-atom Klopman–Ohno hardness come from the reference cell.
    let multipole_on = options.multipole;
    let mp_rank = if options.multipole_order >= 1 {
        options.multipole_order
    } else {
        2
    };
    let moment_len = crate::multipole::generic_moment_stride(mp_rank) * nat;
    let mp_alpha = crate::pbc::ewald::resolve_alpha(system, &pbc.ewald);
    let mp_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
    let mp_hardness: Vec<f64> = (0..nat)
        .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
        .collect();
    let mp_integrals: Option<IntegralMatrices> = if multipole_on {
        Some(IntegralMatrices::build_with_cutoff(
            system,
            &basis,
            pbc.ao_cutoff,
        )?)
    } else {
        None
    };
    let mp_field_kernel = if multipole_on {
        crate::pbc::ewald_multipole::PeriodicMultipoleFieldKernel::try_build(
            system,
            mp_alpha,
            &mp_hardness,
            mp_rank,
        )
    } else {
        None
    };
    let mp_moment_cache = if multipole_on {
        Some(crate::multipole::OnsiteMomentCache::build_with_aos(
            &basis, nat, &mp_pos, mp_rank, None,
        ))
    } else {
        None
    };

    // Joint SCF vector: `[shell charges (nsh) | packed moments ranks 1..=L]` when multipole is on.
    let mix_len = if multipole_on { nsh + moment_len } else { nsh };
    let mut broyden = BroydenMixer::new(mix_len, options.scc_broyden_size.max(2), mixing);
    let mut v_mixed = vec![0.0; mix_len];
    if let Some(guess) = guess {
        if guess.shell_charges.len() == nsh && trusted_charge_vector(&guess.shell_charges) {
            v_mixed[0..nsh].copy_from_slice(&guess.shell_charges);
        }
        if multipole_on
            && guess.atomic_moments.len() == nat
            && guess.atomic_moments.iter().all(|m| {
                m.len() > mp_rank && (1..=mp_rank).all(|l| m[l].len() == 3usize.pow(l as u32))
            })
        {
            crate::multipole::pack_generic_moments(
                &guess.atomic_moments,
                mp_rank,
                &mut v_mixed[nsh..nsh + moment_len],
            );
        }
    }
    let mut converged = false;
    let mut iterations = 0;
    let mut last_energy: Option<f64> = None;
    let mut final_state: Option<SccState> = None;
    let mut final_mp_energy = 0.0_f64;
    let mut final_dipoles: Vec<Vec3> = vec![Vec3::zero(); nat];
    let mut final_moments: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut final_mp_fields: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut final_rms = f64::INFINITY;

    for iter in 1..=options.max_scc {
        iterations = iter;
        let _p_iter = crate::profile::scope("pbc.scf.iteration");
        let q = v_mixed[0..nsh].to_vec();
        let scc = {
            let _p = crate::profile::scope("pbc.scf.scc_potential");
            coulomb_energy_potential_from_matrix(&basis, &shell_model, &q, &gamma)?
        };
        let mut effective_potential =
            combine_shell_potential(&scc.shell_potential, external_shell_potential.as_deref());

        // Arbitrary-rank multipole Fock + charge potential from the MIXED moments + mixed monopole
        // `qm = −(GFN1 atomic charge)`. The periodic field `V[A][l] = ∂E_mp/∂M[A][l]` drives the
        // **two routes**: the on-site moment operator (ranks ≥ 1) is the AO-block Fock added to
        // every `H(k)`; the charge potential `V[A][0] = ∂E_mp/∂qm` enters the shell potential as
        // `∂E_mp/∂q_shell = −V[A][0]` (since `qm = −q_atom`, `q_atom = Σ_shell q_shell`).
        let mp_fock: Option<Matrix> = if multipole_on {
            let mut moments = crate::multipole::unpack_generic_moments(
                &v_mixed[nsh..nsh + moment_len],
                nat,
                mp_rank,
            );
            let qm: Vec<f64> = shell_model
                .atomic_charges(&basis, &q)
                .iter()
                .map(|c| -c)
                .collect();
            for (a, m) in moments.iter_mut().enumerate() {
                m[0] = vec![qm[a]];
            }
            let mut v_field = match mp_field_kernel.as_ref() {
                Some(kernel) => kernel.apply(&moments),
                None => crate::pbc::ewald_multipole::periodic_multipole_fields_generic(
                    system,
                    mp_alpha,
                    &moments,
                    &mp_hardness,
                    mp_rank,
                ),
            };
            // Charge route → shell potential.
            for (ish, shell) in basis.shells.iter().enumerate() {
                effective_potential[ish] -= v_field[shell.atom_index][0][0];
            }
            // Moment route → on-site AO Fock (zero the charge field so it stays in the S(k) route).
            for v in v_field.iter_mut() {
                v[0][0] = 0.0;
            }
            Some(crate::multipole::multipole_fock_from_fields(
                &basis,
                nat,
                &mp_pos,
                mp_integrals.as_ref().unwrap(),
                &v_field,
                mp_rank,
                mp_moment_cache.as_ref(),
            ))
        } else {
            None
        };

        let state = {
            let _p = crate::profile::scope("pbc.scf.step");
            scc_step(
                &basis,
                &hs_k,
                &orth_k,
                &kpoints,
                &effective_potential,
                nelec,
                spin_channels,
                kt,
                options.eigen_tolerance,
                mp_fock.as_ref(),
            )?
        };
        let new_q = state.shell_charges.clone();
        final_rms = charge_rms(&new_q, &q);

        // Output vector `[shell charges | packed moments]` from the reference-cell (T=0) density,
        // and the multipole energy `E_mp = ½ Σ_A Σ_l M·V` at the output density (periodic field).
        let mut v_out = vec![0.0; mix_len];
        v_out[0..nsh].copy_from_slice(&new_q);
        let mut mp_energy = 0.0_f64;
        if multipole_on {
            let mut p_ref = Matrix::zeros(n, n);
            for (ik, kp) in kpoints.iter().enumerate() {
                let w = kp.weight;
                for i in 0..n {
                    for j in 0..n {
                        p_ref[(i, j)] += w * state.density_k[ik].re[(i, j)];
                    }
                }
            }
            let qm_out: Vec<f64> = shell_model
                .atomic_charges(&basis, &new_q)
                .iter()
                .map(|c| -c)
                .collect();
            let out_moments = crate::multipole::build_generic_moments(
                &basis,
                nat,
                &mp_pos,
                mp_integrals.as_ref().unwrap(),
                &p_ref,
                &qm_out,
                mp_rank,
                mp_moment_cache.as_ref(),
            );
            final_dipoles = (0..nat)
                .map(|a| {
                    Vec3::new(
                        out_moments[a][1][0],
                        out_moments[a][1][1],
                        out_moments[a][1][2],
                    )
                })
                .collect();
            final_moments = out_moments.clone();
            crate::multipole::pack_generic_moments(
                &out_moments,
                mp_rank,
                &mut v_out[nsh..nsh + moment_len],
            );
            let v_out_field = match mp_field_kernel.as_ref() {
                Some(kernel) => kernel.apply(&out_moments),
                None => crate::pbc::ewald_multipole::periodic_multipole_fields_generic(
                    system,
                    mp_alpha,
                    &out_moments,
                    &mp_hardness,
                    mp_rank,
                ),
            };
            final_mp_fields = v_out_field.clone();
            let mut e = 0.0;
            for a in 0..nat {
                for l in 0..=mp_rank {
                    e += out_moments[a][l]
                        .iter()
                        .zip(v_out_field[a][l].iter())
                        .map(|(m, vv)| m * vv)
                        .sum::<f64>();
                }
            }
            mp_energy = 0.5 * e;
        }

        let field_energy = external_shell_potential
            .as_ref()
            .map(|v| electric_field_energy(v, &new_q))
            .unwrap_or(0.0);
        let scc_energy = state.band_h0_energy
            + scc.second_order
            + scc.third_order
            + scc.higher_order // 4th+ on-site charge orders (0 unless charge_order > 3)
            + field_energy
            + mp_energy
            + state.entropy;
        let energy_error = last_energy
            .map(|e| (scc_energy - e).abs())
            .unwrap_or(f64::INFINITY);

        if energy_error < options.energy_tolerance && final_rms < options.charge_tolerance {
            converged = true;
            final_state = Some(state);
            final_mp_energy = mp_energy;
            break;
        }

        let residual: Vec<f64> = v_out.iter().zip(&v_mixed).map(|(nw, ol)| nw - ol).collect();
        v_mixed = if options.scc_broyden {
            broyden
                .next(&v_mixed, &residual)
                .filter(|c| trusted_charge_vector(&c[0..nsh]))
                .unwrap_or_else(|| damped_charge_step(&v_mixed, &residual, mixing))
        } else {
            damped_charge_step(&v_mixed, &residual, mixing)
        };
        last_energy = Some(scc_energy);
        final_state = Some(state);
        final_mp_energy = mp_energy;
    }

    let state = final_state.expect("SCC produced no state");
    if !converged {
        return Err(Gfn1Error::SccNotConverged {
            iterations,
            rms: final_rms,
        });
    }

    // Final energies from the converged charges.
    let scc =
        coulomb_energy_potential_from_matrix(&basis, &shell_model, &state.shell_charges, &gamma)?;
    let atomic_charges = shell_model.atomic_charges(&basis, &state.shell_charges);
    let external_field_energy = external_shell_potential
        .as_ref()
        .map(|v| electric_field_energy(v, &state.shell_charges))
        .unwrap_or(0.0);
    let dipole = mulliken_dipole(system, &atomic_charges, options.external_field.origin);
    // Store the full effective potential (SCC + external field) so the gradient
    // path's overlap-derivative coupling carries the field automatically.
    let shell_scc_potential =
        combine_shell_potential(&scc.shell_potential, external_shell_potential.as_deref());
    let total_internal = state.band_h0_energy
        + scc.second_order
        + scc.third_order
        + scc.higher_order // 4th+ on-site charge orders (0 unless charge_order > 3)
        + repulsion
        + dispersion
        + halogen
        + external_field_energy
        + final_mp_energy; // dipole-rank mDFTB2 multipole correction (0 unless options.multipole)
    let total_free = total_internal + state.entropy;
    let coordination_numbers = bloch.coordination_numbers.clone();

    Ok(PbcSccResult {
        basis,
        bloch,
        gamma,
        shell_model,
        kpoints,
        hs_k,
        density_k: state.density_k,
        ew_density_k: state.ew_density_k,
        shell_charges: state.shell_charges,
        atomic_charges,
        shell_scc_potential,
        coordination_numbers,
        fermi_level: state.fermi_level,
        electronic_temperature: options.electronic_temperature,
        electronic_energy: state.band_h0_energy,
        isotropic_scc_energy: scc.second_order,
        third_order_energy: scc.third_order,
        repulsion_energy: repulsion,
        dispersion_energy: dispersion,
        halogen_energy: halogen,
        external_field_energy,
        electronic_entropy_term: state.entropy,
        total_internal,
        total_free,
        dipole,
        atomic_dipoles: final_dipoles,
        atomic_moments: final_moments,
        atomic_multipole_fields: final_mp_fields,
        multipole_energy: final_mp_energy,
        nelec,
        iterations,
        converged,
    })
}

/// Add an optional external site potential to the SCC shell potential.
fn combine_shell_potential(scc_potential: &[f64], external: Option<&[f64]>) -> Vec<f64> {
    let mut out = scc_potential.to_vec();
    if let Some(ext) = external {
        for (v, e) in out.iter_mut().zip(ext.iter()) {
            *v += *e;
        }
    }
    out
}

/// Project a periodic SCC result into the molecular-shaped [`ElectronicResult`]
/// so the existing CLI / Python / API can consume periodic calculations.
///
/// Scalar energies and Mulliken charges are the proper k-averaged values. The
/// matrix fields (`h0`, `fock`, `density`) are the reference-cell / Gamma-point
/// slice: for a Gamma-only run they are exact; for a k-mesh they are the `T = 0`
/// real-space density and the folded Gamma `H0`/`S`. The k-resolved bands live in
/// [`PbcSccResult`]; `orbital_energies`/`occupations` are therefore left empty
/// here (band structure is not a single list across a mesh).
pub fn pbc_electronic_result(
    scf: PbcSccResult,
    system: &PeriodicSystem,
    ao_cutoff: f64,
) -> Result<ElectronicResult> {
    let basis = scf.basis.clone();
    let integrals = IntegralMatrices::build_with_cutoff(system, &basis, ao_cutoff)?;
    let (h0, s_gamma) = scf.bloch.h_s_gamma_real();
    let n = basis.len();

    // Reference-cell (T = 0) real-space density and energy-weighted density.
    let mut density = Matrix::zeros(n, n);
    let mut energy_weighted_density = Matrix::zeros(n, n);
    for (ik, kp) in scf.kpoints.iter().enumerate() {
        let w = kp.weight;
        for i in 0..n {
            for j in 0..n {
                density[(i, j)] += w * scf.density_k[ik].re[(i, j)];
                energy_weighted_density[(i, j)] += w * scf.ew_density_k[ik].re[(i, j)];
            }
        }
    }

    // Gamma-point real Fock from the converged SCC potential.
    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let mut fock = h0.clone();
    for i in 0..n {
        for j in 0..n {
            fock[(i, j)] -= 0.5 * (vao[i] + vao[j]) * s_gamma[(i, j)];
        }
    }

    Ok(ElectronicResult {
        basis,
        integrals,
        h0,
        fock,
        density,
        energy_weighted_density,
        orbital_energies: Vec::new(),
        occupations: Vec::new(),
        electronic_temperature: scf.electronic_temperature,
        fermi_level: scf.fermi_level,
        shell_charges: scf.shell_charges,
        atomic_charges: scf.atomic_charges,
        shell_scc_potential: scf.shell_scc_potential,
        coordination_numbers: scf.coordination_numbers,
        electronic_energy: scf.electronic_energy,
        repulsion_energy: scf.repulsion_energy,
        isotropic_scc_energy: scf.isotropic_scc_energy,
        third_order_energy: scf.third_order_energy,
        dispersion_energy: scf.dispersion_energy,
        halogen_energy: scf.halogen_energy,
        external_field_energy: scf.external_field_energy,
        electronic_entropy_term: scf.electronic_entropy_term,
        total_internal: scf.total_internal,
        total_free: scf.total_free,
        dipole: scf.dipole,
        nelec: scf.nelec,
        iterations: scf.iterations,
        converged: scf.converged,
        spin: None,
    })
}

#[derive(Clone, Debug)]
struct SccState {
    shell_charges: Vec<f64>,
    density_k: Vec<CMatrix>,
    ew_density_k: Vec<CMatrix>,
    band_h0_energy: f64,
    entropy: f64,
    fermi_level: f64,
}

#[allow(clippy::too_many_arguments)]
fn scc_step(
    basis: &BasisSet,
    hs_k: &[(CMatrix, CMatrix)],
    orth_k: &[LowdinOrthogonalizer],
    kpoints: &[KPoint],
    shell_potential: &[f64],
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    kt: f64,
    eigen_tol: f64,
    moment_fock: Option<&Matrix>,
) -> Result<SccState> {
    let n = basis.len();
    // AO-resolved potential.
    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = shell_potential[ish];
        }
    }

    // Diagonalise every k-point and collect eigenvalues. The optional on-site multipole moment
    // Fock is a *real* intra-atomic (T=0) block, so it carries a trivial Bloch phase and is added
    // identically to the real part of every `H0(k)` before the SCC shell-potential shift. It does
    // not touch `hs_k[*].0`, so the band energy `tr[P H0]` stays the bare-H0 trace.
    let mut eig_per_k = Vec::with_capacity(kpoints.len());
    for ((h0k, sk), orth) in hs_k.iter().zip(orth_k.iter()) {
        let fock = if let Some(mf) = moment_fock {
            let mut h = h0k.clone();
            for i in 0..n {
                for j in 0..n {
                    h.re[(i, j)] += mf[(i, j)];
                }
            }
            fock_at_k(&h, sk, &vao)
        } else {
            fock_at_k(h0k, sk, &vao)
        };
        // Reuse the cached `S(k)^{-1/2}` orthogonaliser instead of rebuilding it.
        let eig = hermitian_generalized_eigen_with_orthogonalizer(&fock, orth, n, eigen_tol)?;
        eig_per_k.push(eig);
    }

    // Single global Fermi level over the Brillouin zone.
    let weights: Vec<f64> = kpoints.iter().map(|kp| kp.weight).collect();
    let occ = global_occupations(&eig_per_k, &weights, nelec, spin_channels, kt)?;

    // Per-k densities and band energy.
    let mut density_k = Vec::with_capacity(kpoints.len());
    let mut ew_density_k = Vec::with_capacity(kpoints.len());
    let mut band_h0_energy = 0.0;
    for (ik, eig) in eig_per_k.iter().enumerate() {
        let g = &occ.occupations[ik];
        let ew: Vec<f64> = g.iter().zip(&eig.values).map(|(gi, ei)| gi * ei).collect();
        let p = weighted_density(eig, g)?;
        let w = weighted_density(eig, &ew)?;
        band_h0_energy += weights[ik] * p.real_trace_product(&hs_k[ik].0);
        density_k.push(p);
        ew_density_k.push(w);
    }

    // Mulliken shell charges from the k-averaged populations.
    let shell_charges = mulliken_shell_charges(basis, hs_k, &density_k, &weights);

    Ok(SccState {
        shell_charges,
        density_k,
        ew_density_k,
        band_h0_energy,
        entropy: occ.entropy,
        fermi_level: occ.fermi_level,
    })
}

pub(crate) fn fock_at_k(h0k: &CMatrix, sk: &CMatrix, vao: &[f64]) -> CMatrix {
    let n = h0k.n;
    let mut f = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let shift = 0.5 * (vao[i] + vao[j]);
            f.re[(i, j)] = h0k.re[(i, j)] - shift * sk.re[(i, j)];
            f.im[(i, j)] = h0k.im[(i, j)] - shift * sk.im[(i, j)];
        }
    }
    f
}

fn mulliken_shell_charges(
    basis: &BasisSet,
    hs_k: &[(CMatrix, CMatrix)],
    density_k: &[CMatrix],
    weights: &[f64],
) -> Vec<f64> {
    let n = basis.len();
    // AO populations: pop_mu = sum_k w_k Re[(P(k) S(k))_mu mu].
    let mut pop = vec![0.0; n];
    for (ik, p) in density_k.iter().enumerate() {
        let s = &hs_k[ik].1;
        let w = weights[ik];
        for mu in 0..n {
            let mut acc = 0.0;
            for nu in 0..n {
                acc += p.re[(mu, nu)] * s.re[(nu, mu)] - p.im[(mu, nu)] * s.im[(nu, mu)];
            }
            pop[mu] += w * acc;
        }
    }
    let mut qsh = vec![0.0; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut population = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            population += pop[iao];
        }
        qsh[ish] = shell.reference_occ - population;
    }
    qsh
}

struct OccupationResult {
    occupations: Vec<Vec<f64>>,
    fermi_level: f64,
    entropy: f64,
}

fn global_occupations(
    eig_per_k: &[crate::pbc::complex::KEigen],
    weights: &[f64],
    nelec: f64,
    spin_channels: Option<SpinChannels>,
    kt: f64,
) -> Result<OccupationResult> {
    if let Some(channels) = spin_channels {
        return global_spin_occupations(eig_per_k, weights, channels, kt);
    }
    if kt <= 0.0 {
        return Ok(aufbau_occupations(eig_per_k, weights, nelec));
    }
    // Bisection on the Fermi level.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for eig in eig_per_k {
        for &e in &eig.values {
            lo = lo.min(e);
            hi = hi.max(e);
        }
    }
    lo -= 100.0 * kt + 10.0;
    hi += 100.0 * kt + 10.0;
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let mut count = 0.0;
        for (ik, eig) in eig_per_k.iter().enumerate() {
            let mut s = 0.0;
            for &e in &eig.values {
                s += fermi1(e, mu, kt);
            }
            count += weights[ik] * s;
        }
        if count < nelec {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let mut occupations = Vec::with_capacity(eig_per_k.len());
    let mut entropy = 0.0;
    for (ik, eig) in eig_per_k.iter().enumerate() {
        let mut g = Vec::with_capacity(eig.values.len());
        for &e in &eig.values {
            let f = fermi1(e, mu, kt);
            g.push(f);
            let fc = f.clamp(1.0e-16, 1.0 - 1.0e-16);
            entropy += weights[ik] * kt * (fc * fc.ln() + (1.0 - fc) * (1.0 - fc).ln());
        }
        occupations.push(g);
    }
    Ok(OccupationResult {
        occupations,
        fermi_level: mu,
        entropy,
    })
}

fn aufbau_occupations(
    eig_per_k: &[crate::pbc::complex::KEigen],
    weights: &[f64],
    nelec: f64,
) -> OccupationResult {
    // Flatten (weight, eps, k-index, state-index), sort by energy, fill.
    let mut flat: Vec<(f64, usize, usize)> = Vec::new();
    for (ik, eig) in eig_per_k.iter().enumerate() {
        for (ia, &e) in eig.values.iter().enumerate() {
            flat.push((e, ik, ia));
        }
    }
    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut occupations: Vec<Vec<f64>> = eig_per_k
        .iter()
        .map(|e| vec![0.0; e.values.len()])
        .collect();
    let mut remaining = nelec.max(0.0);
    let mut fermi = flat.first().map(|f| f.0).unwrap_or(0.0);
    for (e, ik, ia) in flat {
        if remaining <= 0.0 {
            break;
        }
        let capacity = weights[ik]; // single-electron state weighted by k-weight
        let fill = remaining.min(capacity);
        occupations[ik][ia] = fill / weights[ik];
        remaining -= fill;
        fermi = e;
    }
    OccupationResult {
        occupations,
        fermi_level: fermi,
        entropy: 0.0,
    }
}

#[inline]
fn fermi1(eps: f64, mu: f64, kt: f64) -> f64 {
    let x = ((eps - mu) / kt).clamp(-80.0, 80.0);
    1.0 / (1.0 + x.exp())
}

#[derive(Clone, Debug)]
struct PhysicalOccupation {
    occupations: Vec<Vec<f64>>,
    fermi_level: f64,
    entropy: f64,
}

fn global_spin_occupations(
    eig_per_k: &[crate::pbc::complex::KEigen],
    weights: &[f64],
    channels: SpinChannels,
    kt: f64,
) -> Result<OccupationResult> {
    let bands = physical_band_energies(eig_per_k)?;
    let alpha = if kt <= 0.0 {
        physical_aufbau_occupations(&bands, weights, channels.alpha)
    } else {
        physical_fermi_occupations(&bands, weights, channels.alpha, kt)
    };
    let beta = if kt <= 0.0 {
        physical_aufbau_occupations(&bands, weights, channels.beta)
    } else {
        physical_fermi_occupations(&bands, weights, channels.beta, kt)
    };

    let mut occupations = Vec::with_capacity(eig_per_k.len());
    for (ik, eig) in eig_per_k.iter().enumerate() {
        let nbands = bands[ik].len();
        let mut g = vec![0.0; eig.values.len()];
        for band in 0..nbands {
            let physical_occ = alpha.occupations[ik][band] + beta.occupations[ik][band];
            let embedded_occ = 0.5 * physical_occ;
            g[2 * band] = embedded_occ;
            g[2 * band + 1] = embedded_occ;
        }
        occupations.push(g);
    }

    Ok(OccupationResult {
        occupations,
        fermi_level: 0.5 * (alpha.fermi_level + beta.fermi_level),
        entropy: alpha.entropy + beta.entropy,
    })
}

fn physical_band_energies(eig_per_k: &[crate::pbc::complex::KEigen]) -> Result<Vec<Vec<f64>>> {
    eig_per_k
        .iter()
        .map(|eig| {
            if eig.values.len() != 2 * eig.n {
                return Err(Gfn1Error::InvalidInput(format!(
                    "real-embedding eigenvalue count {} does not match 2n={}",
                    eig.values.len(),
                    2 * eig.n
                )));
            }
            Ok(eig
                .values
                .chunks_exact(2)
                .map(|pair| 0.5 * (pair[0] + pair[1]))
                .collect::<Vec<_>>())
        })
        .collect()
}

fn physical_aufbau_occupations(
    bands: &[Vec<f64>],
    weights: &[f64],
    electrons: f64,
) -> PhysicalOccupation {
    let mut flat = Vec::new();
    for (ik, values) in bands.iter().enumerate() {
        for (band, &e) in values.iter().enumerate() {
            flat.push((e, ik, band));
        }
    }
    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut occupations: Vec<Vec<f64>> =
        bands.iter().map(|values| vec![0.0; values.len()]).collect();
    let mut remaining = electrons.max(0.0);
    let mut fermi = flat.first().map(|f| f.0).unwrap_or(0.0);
    for (e, ik, band) in flat {
        if remaining <= 0.0 {
            break;
        }
        let capacity = weights[ik];
        let fill = remaining.min(capacity);
        occupations[ik][band] = fill / weights[ik];
        remaining -= fill;
        fermi = e;
    }
    PhysicalOccupation {
        occupations,
        fermi_level: fermi,
        entropy: 0.0,
    }
}

fn physical_fermi_occupations(
    bands: &[Vec<f64>],
    weights: &[f64],
    electrons: f64,
    kt: f64,
) -> PhysicalOccupation {
    let empty = || PhysicalOccupation {
        occupations: bands.iter().map(|values| vec![0.0; values.len()]).collect(),
        fermi_level: bands
            .iter()
            .flat_map(|values| values.iter())
            .copied()
            .next()
            .unwrap_or(0.0),
        entropy: 0.0,
    };
    if electrons <= 0.0 {
        return empty();
    }
    let capacity: f64 = bands
        .iter()
        .enumerate()
        .map(|(ik, values)| weights[ik] * values.len() as f64)
        .sum();
    if electrons >= capacity {
        return PhysicalOccupation {
            occupations: bands.iter().map(|values| vec![1.0; values.len()]).collect(),
            fermi_level: bands
                .iter()
                .flat_map(|values| values.iter())
                .copied()
                .last()
                .unwrap_or(0.0),
            entropy: 0.0,
        };
    }

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for values in bands {
        for &e in values {
            lo = lo.min(e);
            hi = hi.max(e);
        }
    }
    lo -= 100.0 * kt + 10.0;
    hi += 100.0 * kt + 10.0;
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        let mut count = 0.0;
        for (ik, values) in bands.iter().enumerate() {
            let mut s = 0.0;
            for &e in values {
                s += fermi1(e, mu, kt);
            }
            count += weights[ik] * s;
        }
        if count < electrons {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    let mut occupations = Vec::with_capacity(bands.len());
    let mut entropy = 0.0;
    for (ik, values) in bands.iter().enumerate() {
        let mut occ_k = Vec::with_capacity(values.len());
        for &e in values {
            let f = fermi1(e, mu, kt);
            occ_k.push(f);
            let fc = f.clamp(1.0e-16, 1.0 - 1.0e-16);
            entropy += weights[ik] * kt * (fc * fc.ln() + (1.0 - fc) * (1.0 - fc).ln());
        }
        occupations.push(occ_k);
    }
    PhysicalOccupation {
        occupations,
        fermi_level: mu,
        entropy,
    }
}

fn charge_rms(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let ss: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    (ss / a.len() as f64).sqrt()
}

fn damped_charge_step(q: &[f64], residual: &[f64], mixing: f64) -> Vec<f64> {
    q.iter()
        .zip(residual)
        .map(|(q, r)| q + mixing * r)
        .collect()
}

fn trusted_charge_vector(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() < 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::{EwaldOptions, KMesh, PbcOptions};

    fn load_params() -> Option<Gfn1Parameters> {
        let path = std::env::var("GFN1_XTB_PARAM").ok()?;
        Gfn1Parameters::from_file(path).ok()
    }

    const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";

    fn water_in_cell(l: f64) -> PeriodicSystem {
        let comment = format!("Lattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"");
        let xyz = format!(
            "3\n{comment}\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n"
        );
        PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap()
    }

    fn nondispersive_options() -> ElectronicOptions {
        ElectronicOptions {
            enable_dispersion: false,
            ..ElectronicOptions::default()
        }
    }

    #[test]
    fn spin_constrained_occupations_preserve_real_embedding_pairs() {
        let eig = crate::pbc::complex::KEigen {
            values: vec![-1.0, -1.0, -0.5, -0.5, 0.2, 0.2],
            vectors: Matrix::zeros(6, 6),
            n: 3,
        };
        let occ = global_occupations(
            &[eig],
            &[1.0],
            3.0,
            Some(SpinChannels {
                alpha: 2.0,
                beta: 1.0,
            }),
            0.0,
        )
        .unwrap();
        assert_eq!(occ.occupations[0], vec![1.0, 1.0, 0.5, 0.5, 0.0, 0.0]);
    }

    // Gamma-point PBC for an isolated molecule in a large cell reproduces the
    // molecular total (electronic + SCC + repulsion + entropy; dispersion off).
    #[test]
    fn gamma_large_cell_matches_molecular_total() {
        let Some(params) = load_params() else {
            return;
        };
        let mol = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let mol_result =
            crate::electronic::run_electronic(&mol, &params, nondispersive_options()).unwrap();

        let cell = water_in_cell(40.0);
        let pbc = run_pbc_scc(
            &cell,
            &params,
            &nondispersive_options(),
            &PbcOptions::default(),
        )
        .unwrap();

        assert!(pbc.converged, "periodic SCC did not converge");
        let diff = (mol_result.total_free - pbc.total_free).abs();
        assert!(
            diff < 1.0e-4,
            "molecular {:.8} vs Gamma-PBC {:.8} (diff {diff:.2e})",
            mol_result.total_free,
            pbc.total_free
        );
    }

    // The on-site higher-order charge expansion (charge_order > 3) is purely local,
    // so a Gamma-point large cell must still reproduce the molecular total when both
    // use the same charge_order — and the higher order must actually move the energy
    // (otherwise the A0 wiring is a silent no-op).
    #[test]
    fn gamma_large_cell_matches_molecular_total_higher_charge_order() {
        let Some(params) = load_params() else {
            return;
        };
        let opts5 = ElectronicOptions {
            charge_order: 5,
            ..nondispersive_options()
        };
        let opts3 = nondispersive_options();
        let mol = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let mol5 = crate::electronic::run_electronic(&mol, &params, opts5.clone()).unwrap();
        let mol3 = crate::electronic::run_electronic(&mol, &params, opts3.clone()).unwrap();

        let cell = water_in_cell(40.0);
        let pbc5 = run_pbc_scc(&cell, &params, &opts5, &PbcOptions::default()).unwrap();
        let pbc3 = run_pbc_scc(&cell, &params, &opts3, &PbcOptions::default()).unwrap();
        assert!(
            pbc5.converged,
            "periodic SCC (charge_order=5) did not converge"
        );
        assert!(
            (mol5.total_free - pbc5.total_free).abs() < 1.0e-4,
            "molecular {:.8} vs Gamma-PBC {:.8} at charge_order=5",
            mol5.total_free,
            pbc5.total_free
        );

        // The on-site higher-order term is purely local, so the energy *shift* from
        // raising the order (4th+5th) must be (a) nonzero — the wiring is active — and
        // (b) identical in the molecular and periodic paths. Water's small Mulliken
        // charges make the absolute shift tiny (~1e-8 Ha), so it is the molecular/PBC
        // *agreement* of the shift, not its magnitude, that proves correctness.
        let pbc_shift = pbc5.total_free - pbc3.total_free;
        let mol_shift = mol5.total_free - mol3.total_free;
        assert!(
            pbc_shift.abs() > 1.0e-9,
            "charge_order=5 did not change the periodic energy (shift {pbc_shift:.2e})"
        );
        assert!(
            (pbc_shift - mol_shift).abs() < 1.0e-9,
            "periodic higher-order shift {pbc_shift:.3e} != molecular shift {mol_shift:.3e}"
        );
    }

    // For a large cell the molecular bands are flat, so a denser k-mesh must give
    // essentially the same total energy as the Gamma point.
    #[test]
    fn gamma_matches_dense_k_for_flat_bands() {
        let Some(params) = load_params() else {
            return;
        };
        let cell = water_in_cell(40.0);
        let opts = nondispersive_options();
        let gamma = run_pbc_scc(&cell, &params, &opts, &PbcOptions::default()).unwrap();
        let dense = run_pbc_scc(
            &cell,
            &params,
            &opts,
            &PbcOptions {
                kmesh: KMesh::monkhorst_pack([2, 2, 2]),
                ..PbcOptions::default()
            },
        )
        .unwrap();
        let diff = (gamma.total_free - dense.total_free).abs();
        assert!(
            diff < 1.0e-5,
            "Gamma {:.8} vs 2x2x2 {:.8} (diff {diff:.2e})",
            gamma.total_free,
            dense.total_free
        );
    }

    // A real periodic solid (cubic diamond) runs and converges.
    #[test]
    fn diamond_scc_converges() {
        let Some(params) = load_params() else {
            return;
        };
        // Conventional 8-atom diamond cell, a = 3.567 Angstrom.
        let cell = PeriodicSystem::from_xyz_str(
            "8\nLattice=\"3.567 0 0 0 3.567 0 0 0 3.567\" pbc=\"T T T\"\n\
             C 0.000000 0.000000 0.000000\n\
             C 0.891750 0.891750 0.891750\n\
             C 0.000000 1.783500 1.783500\n\
             C 0.891750 2.675250 2.675250\n\
             C 1.783500 0.000000 1.783500\n\
             C 2.675250 0.891750 2.675250\n\
             C 1.783500 1.783500 0.000000\n\
             C 2.675250 2.675250 0.891750\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = nondispersive_options();
        let result = run_pbc_scc(
            &cell,
            &params,
            &opts,
            &PbcOptions {
                kmesh: KMesh::monkhorst_pack([2, 2, 2]),
                ewald: EwaldOptions::default(),
                ..PbcOptions::default()
            },
        )
        .unwrap();
        assert!(result.converged, "diamond SCC did not converge");
        assert!(result.total_free.is_finite());
    }

    // Moment-from-density bridge (A2 prerequisite): the molecular `build_generic_moments` runs
    // unchanged on a periodic SCC's reference-cell density (on-site moments are local), and a
    // polar cell yields nonzero on-site atomic dipoles — the SCF input the joint mixer consumes.
    #[test]
    fn periodic_density_yields_atomic_moments() {
        let Some(params) = load_params() else {
            return;
        };
        let cell = water_in_cell(40.0);
        let electronic =
            crate::pbc::run_electronic_pbc(&cell, &params, &nondispersive_options()).unwrap();
        let nat = cell.atoms.len();
        let pos: Vec<Vec3> = cell.atoms.iter().map(|a| a.position).collect();
        let qm: Vec<f64> = electronic.atomic_charges.iter().map(|c| -c).collect();
        let moments = crate::multipole::build_generic_moments(
            &electronic.basis,
            nat,
            &pos,
            &electronic.integrals,
            &electronic.density,
            &qm,
            2, // dipole + quadrupole
            None,
        );
        assert_eq!(moments.len(), nat);
        // Total on-site dipole magnitude (rank-1 block) must be nonzero for polar water.
        let total_dipole_sq: f64 = moments
            .iter()
            .map(|m| m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2])
            .sum();
        assert!(
            total_dipole_sq > 1.0e-8,
            "periodic density produced no on-site dipoles ({total_dipole_sq:.3e})"
        );
    }

    // A2 SCF Fock kernel: the periodic dipole Fock `F = ∂E/∂P` of the periodic dipole self-energy
    // `E = ½ Σ_A d_A·V_A` (V from the periodic Ewald field, dipoles built from the reference-cell
    // density). Variational consistency — `⟨F, δP⟩` matches the central FD of `E` w.r.t. a
    // symmetric density perturbation (dipoles rebuilt, periodic field recomputed each
    // displacement) — closing the moment→field→Fock loop the multipole SCC injects into every
    // `H(k)`. This is the new physics (periodic field) feeding the validated on-site shift
    // machinery; the chain must be mutually consistent for the SCC to stay variational.
    #[test]
    fn periodic_dipole_fock_matches_energy_derivative() {
        let Some(params) = load_params() else {
            return;
        };
        let cell = water_in_cell(40.0);
        let electronic =
            crate::pbc::run_electronic_pbc(&cell, &params, &nondispersive_options()).unwrap();
        let nat = cell.atoms.len();
        let pos: Vec<Vec3> = cell.atoms.iter().map(|a| a.position).collect();
        let basis = &electronic.basis;
        let integrals = &electronic.integrals;
        let n = basis.len();
        let alpha = 0.35_f64;
        let hardnesses = vec![1.0_f64, 0.8, 0.8]; // arbitrary positive per-atom η

        // Rank-1 atomic dipoles from a density block (the SCF moment channel).
        let dipoles_from = |p: &Matrix| -> Vec<Vec3> {
            let qm = vec![0.0_f64; nat];
            let m = crate::multipole::build_generic_moments(
                basis, nat, &pos, integrals, p, &qm, 1, None,
            );
            (0..nat)
                .map(|a| Vec3::new(m[a][1][0], m[a][1][1], m[a][1][2]))
                .collect()
        };
        let energy = |p: &Matrix| -> f64 {
            let d = dipoles_from(p);
            crate::pbc::ewald_multipole::periodic_dipole_dipole_energy_ko_pairwise(
                &cell,
                alpha,
                &d,
                &hardnesses,
            )
        };

        let p0 = electronic.density.clone();
        let d0 = dipoles_from(&p0);
        let field = crate::pbc::ewald_multipole::periodic_dipole_field_ko_pairwise(
            &cell,
            alpha,
            &d0,
            &hardnesses,
        );
        let fock = crate::multipole::periodic_dipole_fock(basis, nat, integrals, &field);

        // Symmetric density perturbation δP (deterministic, mean-removed sign pattern).
        let mut dp = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                dp[(i, j)] = ((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.5;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let s = 0.5 * (dp[(i, j)] + dp[(j, i)]);
                dp[(i, j)] = s;
                dp[(j, i)] = s;
            }
        }

        let analytic: f64 = (0..n)
            .map(|i| (0..n).map(|j| fock[(i, j)] * dp[(i, j)]).sum::<f64>())
            .sum();

        let h = 1.0e-5;
        let mut pp = p0.clone();
        let mut pm = p0.clone();
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] = p0[(i, j)] + h * dp[(i, j)];
                pm[(i, j)] = p0[(i, j)] - h * dp[(i, j)];
            }
        }
        let fd = (energy(&pp) - energy(&pm)) / (2.0 * h);
        assert!(
            (analytic - fd).abs() < 1.0e-6 * (1.0 + fd.abs()),
            "periodic dipole Fock ⟨F,δP⟩ {analytic:.8e} vs FD {fd:.8e}"
        );
    }

    // A2/A3 end-to-end: the dipole-rank periodic multipole SCC (joint charge+dipole Broyden mixing,
    // QCore Ewald field rebuilt from the mixed state each iteration) converges, and its converged
    // total energy is **α-independent** — the defining correctness property of the Ewald multipole,
    // here exercised through the whole self-consistent loop (field + two Fock routes + energy). It
    // must also shift the energy versus the monopole-only SCC (the correction is non-trivial).
    #[test]
    fn periodic_multipole_scc_converges_and_is_alpha_independent() {
        let Some(params) = load_params() else {
            return;
        };
        let cell = water_in_cell(10.0);
        let mono_opts = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let mp_opts = ElectronicOptions {
            multipole: true,
            ..mono_opts.clone()
        };
        let pbc_a = PbcOptions {
            ewald: EwaldOptions {
                k_split: Some(0.30),
                ..EwaldOptions::default()
            },
            ..PbcOptions::default()
        };
        let pbc_b = PbcOptions {
            ewald: EwaldOptions {
                k_split: Some(0.50),
                ..EwaldOptions::default()
            },
            ..PbcOptions::default()
        };

        let ra = run_pbc_scc(&cell, &params, &mp_opts, &pbc_a).unwrap();
        let rb = run_pbc_scc(&cell, &params, &mp_opts, &pbc_b).unwrap();
        assert!(
            ra.converged && rb.converged,
            "multipole SCC did not converge"
        );
        assert!(
            (ra.total_free - rb.total_free).abs() < 1.0e-6 * (1.0 + ra.total_free.abs()),
            "multipole SCC total not α-independent: {:.10} (α=0.30) vs {:.10} (α=0.50)",
            ra.total_free,
            rb.total_free
        );

        let mono = run_pbc_scc(&cell, &params, &mono_opts, &pbc_a).unwrap();
        assert!(
            (ra.total_free - mono.total_free).abs() > 1.0e-7,
            "dipole-rank multipole did not change the periodic energy (shift {:.2e})",
            (ra.total_free - mono.total_free).abs()
        );

        // The converged dipoles + multipole energy are stored (for the A4 gradient / reporting);
        // the monopole-only run carries neither.
        let nat = cell.atoms.len();
        assert_eq!(ra.atomic_dipoles.len(), nat);
        let dmag: f64 = ra.atomic_dipoles.iter().map(|d| d.norm2()).sum();
        assert!(
            dmag > 1.0e-8,
            "polar cell should carry nonzero on-site dipoles"
        );
        assert!(
            ra.multipole_energy.abs() > 1.0e-9,
            "multipole energy should be stored nonzero"
        );
        // Full arbitrary-rank moments stored (default rank 2: ranks 0,1,2); rank-1 == dipoles.
        assert_eq!(ra.atomic_moments.len(), nat);
        for a in 0..nat {
            assert!(
                ra.atomic_moments[a].len() >= 3,
                "expected ranks 0..=2 stored"
            );
            assert!((ra.atomic_moments[a][1][0] - ra.atomic_dipoles[a].x).abs() < 1.0e-12);
            assert!((ra.atomic_moments[a][1][1] - ra.atomic_dipoles[a].y).abs() < 1.0e-12);
            assert!((ra.atomic_moments[a][1][2] - ra.atomic_dipoles[a].z).abs() < 1.0e-12);
        }
        assert!(
            mono.atomic_dipoles.iter().all(|d| d.norm2() < 1.0e-30),
            "monopole-only run must carry no dipoles"
        );
        assert_eq!(mono.multipole_energy, 0.0);
    }

    // QUADRUPOLE in the SCF: the arbitrary-rank generic path runs `multipole_order = 2`
    // (dipole+quadrupole) to convergence, the total is α-independent, and it differs from the
    // dipole-only (`multipole_order = 1`) SCC — i.e. the on-site quadrupole genuinely enters the
    // self-consistent periodic energy (water's O carries a quadrupole).
    #[test]
    fn periodic_multipole_scc_quadrupole_converges_alpha_independent_and_differs_from_dipole() {
        let Some(params) = load_params() else {
            return;
        };
        let cell = water_in_cell(10.0);
        let base = ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            multipole: true,
            ..ElectronicOptions::default()
        };
        let dip = ElectronicOptions {
            multipole_order: 1,
            ..base.clone()
        };
        let quad = ElectronicOptions {
            multipole_order: 2,
            ..base
        };
        let pbc_a = PbcOptions {
            ewald: EwaldOptions {
                k_split: Some(0.30),
                ..EwaldOptions::default()
            },
            ..PbcOptions::default()
        };
        let pbc_b = PbcOptions {
            ewald: EwaldOptions {
                k_split: Some(0.50),
                ..EwaldOptions::default()
            },
            ..PbcOptions::default()
        };

        let rd = run_pbc_scc(&cell, &params, &dip, &pbc_a).unwrap();
        let rq_a = run_pbc_scc(&cell, &params, &quad, &pbc_a).unwrap();
        let rq_b = run_pbc_scc(&cell, &params, &quad, &pbc_b).unwrap();
        assert!(
            rd.converged && rq_a.converged && rq_b.converged,
            "dipole/quadrupole SCC did not converge"
        );
        assert!(
            (rq_a.total_free - rq_b.total_free).abs() < 1.0e-6 * (1.0 + rq_a.total_free.abs()),
            "quadrupole SCC not α-independent: {:.10} vs {:.10}",
            rq_a.total_free,
            rq_b.total_free
        );
        assert!(
            (rq_a.total_free - rd.total_free).abs() > 1.0e-8,
            "quadrupole did not change the periodic energy vs dipole-only (shift {:.2e})",
            (rq_a.total_free - rd.total_free).abs()
        );
    }
}
