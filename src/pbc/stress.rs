// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic periodic stress tensor.
//!
//! Convention: `sigma_ab = (1/V) dE_free / d eps_ab` for a homogeneous strain
//! `F = I + eps` at fixed fractional coordinates. The electronic part mirrors
//! the variational PBC gradient, replacing Cartesian center derivatives by
//! strain derivatives of the same AO image blocks. The Ewald reciprocal phase is
//! invariant at fixed fractional coordinates; its stress comes from the volume
//! factor and reciprocal-vector metric dependence.

use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::harmonic_average;
use crate::data_tables::atomic_radius_bohr;
use crate::dispersion::dispersion_stress;
use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::halogen::halogen_stress;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::contracted_pair_with_derivatives;
use crate::lattice::{ImageOffset, Lattice};
use crate::linalg::Matrix;
use crate::math::{erfc, Vec3};
use crate::pairlist::canonical_positive_offset;
use crate::params::Gfn1Parameters;
use crate::pbc::ewald::{
    exp1, qcore_r3_k0_log, qcore_r3_real_value_derivatives, qcore_short_value_derivatives,
    resolve_alpha, QCORE_R3_COEFF,
};
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::{run_pbc_scc, PbcSccResult};
use crate::pbc::PbcOptions;
use crate::repulsion::repulsion_stress;
use crate::system::PeriodicSystem;
use rayon::prelude::*;

const SQRT_PI: f64 = 1.772_453_850_905_516;
const TAU: f64 = 5.5;
const DIST_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct PbcStressResult {
    pub scf: PbcSccResult,
    pub total_energy: f64,
    pub stress: Matrix,
    pub electronic_stress: Matrix,
    pub electrostatic_stress: Matrix,
    pub repulsion_stress: Matrix,
    pub dispersion_stress: Matrix,
    pub halogen_stress: Matrix,
    /// Periodic multipole correction stress (0 unless `options.multipole`).
    pub multipole_stress: Matrix,
}

pub fn pbc_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcStressResult> {
    let lattice =
        system.lattice.as_ref().copied().ok_or_else(|| {
            Gfn1Error::InvalidInput("stress requires a periodic lattice".to_string())
        })?;
    let scf = run_pbc_scc(system, params, options, pbc)?;
    pbc_stress_from_scc(system, params, scf, options, pbc, &lattice)
}

pub fn pbc_stress_from_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    lattice: &Lattice,
) -> Result<PbcStressResult> {
    let _profile = crate::profile::scope("pbc.stress.total");
    let electronic_stress = {
        let _p = crate::profile::scope("pbc.stress.band_cn");
        band_and_cn_stress(system, params, &scf, options, pbc.ao_cutoff, lattice)?
    };
    let electrostatic_stress = {
        let _p = crate::profile::scope("pbc.stress.electrostatic");
        electrostatic_stress(system, &scf, pbc, lattice)
    };
    let repulsion_stress = {
        let _p = crate::profile::scope("pbc.stress.repulsion");
        repulsion_stress(system, params)?.unwrap_or_else(|| Matrix::zeros(3, 3))
    };
    let dispersion_stress = if options.enable_dispersion {
        let _p = crate::profile::scope("pbc.stress.dispersion");
        dispersion_stress(system, params, options.d3_reference_path.as_deref())?
            .unwrap_or_else(|| Matrix::zeros(3, 3))
    } else {
        Matrix::zeros(3, 3)
    };
    let halogen_stress = {
        let _p = crate::profile::scope("pbc.stress.halogen");
        halogen_stress(system)?.unwrap_or_else(|| Matrix::zeros(3, 3))
    };
    let multipole_stress = if options.multipole {
        let _p = crate::profile::scope("pbc.stress.multipole");
        multipole_stress_terms(system, params, &scf, pbc, lattice)?
    } else {
        Matrix::zeros(3, 3)
    };

    let mut stress = Matrix::zeros(3, 3);
    add_matrix_in_place(&mut stress, &electronic_stress);
    add_matrix_in_place(&mut stress, &electrostatic_stress);
    add_matrix_in_place(&mut stress, &repulsion_stress);
    add_matrix_in_place(&mut stress, &dispersion_stress);
    add_matrix_in_place(&mut stress, &halogen_stress);
    add_matrix_in_place(&mut stress, &multipole_stress);

    Ok(PbcStressResult {
        total_energy: scf.total_free,
        scf,
        stress,
        electronic_stress,
        electrostatic_stress,
        repulsion_stress,
        dispersion_stress,
        halogen_stress,
        multipole_stress,
    })
}

fn offset_has_active_pair(
    system: &PeriodicSystem,
    lattice: &Lattice,
    atom_min_exp: &[f64],
    ao_cutoff: f64,
    off: ImageOffset,
) -> bool {
    let nat = system.atoms.len();
    let cutoff2 = ao_cutoff * ao_cutoff;
    let translation = lattice.translation(off);
    for a in 0..nat {
        let ra = system.atoms[a].position;
        for b in 0..nat {
            let rb = system.atoms[b].position + translation;
            let r2 = (ra - rb).norm2();
            if r2 > DIST_EPS
                && r2 <= cutoff2
                && !crate::basis::overlap_screened(atom_min_exp, a, b, r2)
            {
                return true;
            }
        }
    }
    false
}

enum RealspaceDensity<'a> {
    Borrowed { p: &'a Matrix, w: &'a Matrix },
    Owned { p: Matrix, w: Matrix },
}

impl RealspaceDensity<'_> {
    fn p(&self) -> &Matrix {
        match self {
            Self::Borrowed { p, .. } => p,
            Self::Owned { p, .. } => p,
        }
    }

    fn w(&self) -> &Matrix {
        match self {
            Self::Borrowed { w, .. } => w,
            Self::Owned { w, .. } => w,
        }
    }
}

fn realspace_density_image(scf: &PbcSccResult, off: ImageOffset) -> RealspaceDensity<'_> {
    if scf.kpoints.len() == 1 && scf.kpoints[0].fractional == [0.0, 0.0, 0.0] {
        return RealspaceDensity::Borrowed {
            p: &scf.density_k[0].re,
            w: &scf.ew_density_k[0].re,
        };
    }

    let n = scf.basis.len();
    let mut p = Matrix::zeros(n, n);
    let mut w = Matrix::zeros(n, n);
    for (ik, kp) in scf.kpoints.iter().enumerate() {
        let (c, s) = bloch_phase(kp.fractional, off);
        let wk = kp.weight;
        let pk = &scf.density_k[ik];
        let wk_mat = &scf.ew_density_k[ik];
        for i in 0..n {
            for j in 0..n {
                p[(i, j)] += wk * (pk.re[(i, j)] * c + pk.im[(i, j)] * s);
                w[(i, j)] += wk * (wk_mat.re[(i, j)] * c + wk_mat.im[(i, j)] * s);
            }
        }
    }
    RealspaceDensity::Owned { p, w }
}

#[allow(clippy::too_many_arguments)]
fn band_and_cn_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    ao_cutoff: f64,
    lattice: &Lattice,
) -> Result<Matrix> {
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let n = basis.len();
    let self_energy = &scf.bloch.self_energies;
    let dsedcn = &scf.bloch.dsedcn;
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let inv_volume = 1.0 / lattice.volume();
    let mut deriv = Matrix::zeros(3, 3);

    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }

    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }

    let mut d_edcn = vec![0.0; nat];
    let atom_min_exp = crate::basis::atom_min_exponents(basis, nat);
    let images: Vec<ImageOffset> = lattice
        .image_offsets(ao_cutoff)
        .into_iter()
        .filter(|off| {
            off.is_origin()
                || (canonical_positive_offset(*off)
                    && offset_has_active_pair(system, lattice, &atom_min_exp, ao_cutoff, *off))
        })
        .collect();
    let cutoff2 = ao_cutoff * ao_cutoff;

    let per_image = images
        .par_iter()
        .map(|off| -> Result<(Matrix, Vec<f64>)> {
            let mut local_deriv = Matrix::zeros(3, 3);
            let mut local_dedcn = vec![0.0; nat];
            let density = realspace_density_image(scf, *off);
            let p_img = density.p();
            let w_img = density.w();
            let is_origin = off.is_origin();
            let translation = lattice.translation(*off);

            for a in 0..nat {
                let ra = system.atoms[a].position;
                for b in 0..nat {
                    if is_origin && a >= b {
                        continue;
                    }
                    let rb = system.atoms[b].position + translation;
                    let rvec = ra - rb;
                    let r2 = rvec.norm2();
                    if r2 <= DIST_EPS || r2 > cutoff2 {
                        continue;
                    }
                    if crate::basis::overlap_screened(&atom_min_exp, a, b, r2) {
                        continue;
                    }
                    let rad_sum = atomic_radius_bohr(system.atoms[a].z)?
                        + atomic_radius_bohr(system.atoms[b].z)?;
                    let rr = (r2.sqrt() / rad_sum).sqrt();
                    for &mu in &atom_aos[a] {
                        let si_idx = basis.aos[mu].shell_index;
                        let si = &basis.shells[si_idx];
                        for &nu in &atom_aos[b] {
                            let sj_idx = basis.aos[nu].shell_index;
                            let sj = &basis.shells[sj_idx];
                            let (moments, d_bra, d_ket) = contracted_pair_with_derivatives(
                                &basis.aos[mu],
                                &basis.aos[nu],
                                ra,
                                rb,
                            );
                            let overlap = moments[0];
                            let hs = hscale(si, sj, params)? * shell_polynomial(si, sj, rr);
                            let hij = 0.5 * (self_energy[si_idx] + self_energy[sj_idx]) * hs;
                            let p = p_img[(mu, nu)];
                            let w = w_img[(mu, nu)];
                            let scalar_shift = vao[mu] + vao[nu];
                            let overlap_coeff = p * (2.0 * hij - scalar_shift) - 2.0 * w;
                            let dlog_poly = shell_polynomial_log_derivative(si, sj, rvec, r2);
                            let poly_coeff = 2.0 * p * hij * overlap;
                            for row in 0..3 {
                                let d_bra_row = component(d_bra[0], row);
                                let d_ket_row = component(d_ket[0], row);
                                let dlog_row = component(dlog_poly, row);
                                for col in 0..3 {
                                    let d_overlap = d_bra_row * component(ra, col)
                                        + d_ket_row * component(rb, col);
                                    let d_log = dlog_row * component(rvec, col);
                                    local_deriv[(row, col)] +=
                                        overlap_coeff * d_overlap + poly_coeff * d_log;
                                }
                            }

                            if enable_cn {
                                local_dedcn[a] += dsedcn[si_idx] * hs * p * overlap;
                                local_dedcn[b] += dsedcn[sj_idx] * hs * p * overlap;
                            }
                        }
                    }
                }
            }

            if enable_cn && is_origin {
                for (ish, shell) in basis.shells.iter().enumerate() {
                    for iao in shell.first_ao..shell.first_ao + shell.nao {
                        local_dedcn[shell.atom_index] += dsedcn[ish] * p_img[(iao, iao)];
                    }
                }
            }

            Ok((local_deriv, local_dedcn))
        })
        .collect::<Result<Vec<_>>>()?;

    for (local_deriv, local_dedcn) in per_image {
        add_matrix_in_place(&mut deriv, &local_deriv);
        for a in 0..nat {
            d_edcn[a] += local_dedcn[a];
        }
    }

    if enable_cn {
        let cn = coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: options.hamiltonian.coordination_cutoff,
                ..CoordinationOptions::default()
            },
        )?;
        for pair in cn.pairs {
            let r = pair.r_ij.norm();
            if r <= DIST_EPS {
                continue;
            }
            let coeff = if pair.i == pair.j {
                d_edcn[pair.i]
            } else {
                d_edcn[pair.i] + d_edcn[pair.j]
            };
            add_outer(&mut deriv, pair.r_ij, coeff * pair.dcn_dr / r);
        }
    }

    scale_matrix(&mut deriv, inv_volume);
    Ok(deriv)
}

/// A5: stress of the **arbitrary-rank periodic multipole** correction `σ_mp = (1/V) dE_mp/dε`,
/// the two explicit strain routes mirroring the A4 gradient:
///
/// (i) **Kernel strain** (semi-numerical, at fixed converged moments): the periodic multipole
/// field is purely geometric (atomic positions + lattice + reciprocal `G`), so straining the
/// system and recomputing it captures the real + reciprocal + self strain in a single
/// **α-independent** central difference — no basis/integral rebuild. `E_kernel(ε) = ½ Σ M·V(ε)`.
///
/// (ii) **Overlap-Pulay strain** (analytic virial): the moments depend on the reference-cell
/// overlap, so `E_mp` has an explicit `∂/∂S`. With `W = ∂E_mp/∂S` (the same weight as the gradient,
/// full fields), the term is `2·W·dS/dε` contracted over reference-cell (`T=0`) off-site pairs, in
/// virial form `dS/dε_{ab} = ∂S/∂r_a · r_b`.
///
/// The implicit density response is carried by the base band stress (the moment Fock is in the
/// converged Fock that builds the energy-weighted density). Variable-cell-ready.
fn multipole_stress_terms(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    lattice: &Lattice,
) -> Result<Matrix> {
    let nat = system.atoms.len();
    let basis = &scf.basis;
    let n = basis.len();
    let mut stress = Matrix::zeros(3, 3);
    if scf.atomic_moments.is_empty() {
        return Ok(stress);
    }
    let max_rank = scf.atomic_moments[0].len() - 1;
    let inv_volume = 1.0 / lattice.volume();
    let shell_model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let hardness: Vec<f64> = (0..nat)
        .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
        .collect();
    let moments = &scf.atomic_moments;

    // (i) Kernel strain (virial) at fixed moments — central FD of the geometric multipole field.
    let e_kernel = |strained: &PeriodicSystem| -> f64 {
        let alpha = resolve_alpha(strained, &pbc.ewald);
        let v = crate::pbc::ewald_multipole::periodic_multipole_fields_generic(
            strained, alpha, moments, &hardness, max_rank,
        );
        let mut e = 0.0;
        for a in 0..nat {
            for l in 0..=max_rank {
                e += 0.5
                    * moments[a][l]
                        .iter()
                        .zip(v[a][l].iter())
                        .map(|(m, vv)| m * vv)
                        .sum::<f64>();
            }
        }
        e
    };
    let h = 1.0e-5;
    for row in 0..3 {
        for col in 0..3 {
            let sp = mp_strained_system(system, row, col, h);
            let sm = mp_strained_system(system, row, col, -h);
            stress[(row, col)] += (e_kernel(&sp) - e_kernel(&sm)) / (2.0 * h) * inv_volume;
        }
    }

    // (ii) Overlap-Pulay strain (analytic): 2·W·dS/dε over reference-cell off-site pairs.
    let mut p_ref = Matrix::zeros(n, n);
    for (ik, kp) in scf.kpoints.iter().enumerate() {
        let wk = kp.weight;
        for i in 0..n {
            for j in 0..n {
                p_ref[(i, j)] += wk * scf.density_k[ik].re[(i, j)];
            }
        }
    }
    let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
    let alpha = resolve_alpha(system, &pbc.ewald);
    let v_field = crate::pbc::ewald_multipole::periodic_multipole_fields_generic(
        system, alpha, moments, &hardness, max_rank,
    );
    let cache =
        crate::multipole::OnsiteMomentCache::build_with_aos(basis, nat, &atom_pos, max_rank, None);
    let w_mp = crate::multipole::multipole_weight_from_fields(
        basis,
        nat,
        &atom_pos,
        &p_ref,
        &v_field,
        max_rank,
        Some(&cache),
    );
    let pairs = crate::pairlist::unique_short_range_pairs(system, pbc.ao_cutoff)?;
    let mut atom_shell_ranges = vec![(0usize, 0usize); nat];
    for (sh_idx, sh) in basis.shells.iter().enumerate() {
        let a = sh.atom_index;
        if atom_shell_ranges[a].1 == 0 {
            atom_shell_ranges[a].0 = sh_idx;
        }
        atom_shell_ranges[a].1 += 1;
    }
    for pair in pairs {
        if !pair.offset.is_origin() {
            continue;
        }
        let atom_nu = pair.i;
        let atom_mu = pair.j;
        let rmu = system.atoms[atom_mu].position;
        let rnu = system.atoms[atom_nu].position;
        if (rmu - rnu).norm2() <= DIST_EPS {
            continue;
        }
        let (first_sh_mu, n_sh_mu) = atom_shell_ranges[atom_mu];
        let (first_sh_nu, n_sh_nu) = atom_shell_ranges[atom_nu];
        for shell_mu_index in first_sh_mu..first_sh_mu + n_sh_mu {
            let shell_mu = &basis.shells[shell_mu_index];
            for shell_nu_index in first_sh_nu..first_sh_nu + n_sh_nu {
                let shell_nu = &basis.shells[shell_nu_index];
                for mu in shell_mu.first_ao..shell_mu.first_ao + shell_mu.nao {
                    for nu in shell_nu.first_ao..shell_nu.first_ao + shell_nu.nao {
                        let (_m, d_bra, d_ket) = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let coeff = 2.0 * w_mp[(mu, nu)];
                        for row in 0..3 {
                            let d_bra_row = component(d_bra[0], row);
                            let d_ket_row = component(d_ket[0], row);
                            for col in 0..3 {
                                let d_overlap = d_bra_row * component(rmu, col)
                                    + d_ket_row * component(rnu, col);
                                stress[(row, col)] += coeff * d_overlap * inv_volume;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(stress)
}

/// Homogeneous strain `F = I + δ·(e_row ⊗ e_col)` applied to atom positions and lattice vectors at
/// fixed fractional coordinates (for the kernel-strain finite difference).
fn mp_strained_system(
    system: &PeriodicSystem,
    row: usize,
    col: usize,
    delta: f64,
) -> PeriodicSystem {
    let strain = |v: Vec3| -> Vec3 {
        let c = match col {
            0 => v.x,
            1 => v.y,
            _ => v.z,
        };
        let mut out = v;
        match row {
            0 => out.x += delta * c,
            1 => out.y += delta * c,
            _ => out.z += delta * c,
        }
        out
    };
    let mut out = system.clone();
    for atom in &mut out.atoms {
        atom.position = strain(atom.position);
    }
    if let Some(lattice) = out.lattice {
        let cell = [
            strain(lattice.cell.col[0]),
            strain(lattice.cell.col[1]),
            strain(lattice.cell.col[2]),
        ];
        out.lattice =
            Some(Lattice::new(crate::math::Mat3 { col: cell }, lattice.periodic).unwrap());
    }
    out
}

fn electrostatic_stress(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    lattice: &Lattice,
) -> Matrix {
    let alpha = resolve_alpha(system, &pbc.ewald);
    let mut deriv = ewald_stress_derivative(system, lattice, alpha, &scf.atomic_charges);

    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    qcore_r3_reciprocal_stress_derivative(system, lattice, alpha, basis, model, q, &mut deriv);
    qcore_r3_k0_stress_derivative(lattice, alpha, model, q, &mut deriv);

    let r3_cut = TAU / alpha;
    let sr_cut = pbc.ewald.sr_cutoff;
    let real_cut = r3_cut.max(sr_cut);
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    let nsh = basis.shells.len();
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        let ra = system.atoms[ai].position;
        for j in 0..nsh {
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let qiqj = q[i] * q[j];
            if qiqj == 0.0 {
                continue;
            }
            for (off, t) in offsets.iter().zip(&translations) {
                if ai == aj && off.is_origin() {
                    continue;
                }
                let vec = ra - system.atoms[aj].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let mut radial = 0.0;
                if d <= r3_cut {
                    radial += QCORE_R3_COEFF * qcore_r3_real_value_derivatives(d, eta, alpha).1;
                }
                if d <= sr_cut {
                    radial += qcore_short_value_derivatives(d, eta).1;
                }
                add_outer(&mut deriv, vec, 0.5 * qiqj * radial / d);
            }
        }
    }
    scale_matrix(&mut deriv, 1.0 / lattice.volume());
    deriv
}

fn qcore_r3_reciprocal_stress_derivative(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    basis: &crate::basis::BasisSet,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    deriv: &mut Matrix,
) {
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let pref0 = QCORE_R3_COEFF * std::f64::consts::PI / lattice.volume();
    let nsh = basis.shells.len();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let x = g.norm2() * inv_4a2;
        let e1 = exp1(x);
        let de1_dx = -(-x).exp() / x;
        let mut structure = 0.0;
        for i in 0..nsh {
            if q[i] == 0.0 {
                continue;
            }
            for j in 0..nsh {
                let qiqj = q[i] * q[j];
                if qiqj == 0.0 {
                    continue;
                }
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                structure += qiqj * (phases[i] - phases[j]).cos() / (eta * eta);
            }
        }
        let garr = g.to_array();
        for row in 0..3 {
            for col in 0..3 {
                let delta = if row == col { 1.0 } else { 0.0 };
                let dx = -garr[row] * garr[col] / (2.0 * alpha * alpha);
                deriv[(row, col)] += pref0 * structure * (de1_dx * dx - delta * e1);
            }
        }
    }
}

fn qcore_r3_k0_stress_derivative(
    lattice: &Lattice,
    alpha: f64,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    deriv: &mut Matrix,
) {
    let mut structure = 0.0;
    for i in 0..q.len() {
        for j in 0..q.len() {
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            structure += q[i] * q[j] / (eta * eta);
        }
    }
    let energy = QCORE_R3_COEFF
        * (2.0 * std::f64::consts::PI / lattice.volume())
        * structure
        * qcore_r3_k0_log(alpha);
    for axis in 0..3 {
        deriv[(axis, axis)] -= energy;
    }

    // The Eq. 24 self term depends on the fixed Ewald broadening parameter only;
    // it has no strain derivative under the same fixed-alpha convention as the
    // existing 1/R stress.
}

fn ewald_stress_derivative(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    q_atom: &[f64],
) -> Matrix {
    let nat = system.atoms.len();
    let volume = lattice.volume();
    let real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let rec_prefactor = 2.0 * std::f64::consts::PI / volume;
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;
    let mut deriv = Matrix::zeros(3, 3);

    for a in 0..nat {
        for b in 0..nat {
            let qaqb = q_atom[a] * q_atom[b];
            if qaqb == 0.0 {
                continue;
            }
            for t in &translations {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let dgdr = -erfc(alpha * d) / (d * d)
                    - two_alpha_sqrtpi * (-alpha * alpha * d * d).exp() / d;
                add_outer(&mut deriv, vec, 0.5 * qaqb * dgdr / d);
            }
        }
    }

    for (_, g) in &recip {
        let g2 = g.norm2();
        let expg = (-g2 * inv_4a2).exp();
        let w_g = expg / g2;
        let dw_dg2 = expg * (-inv_4a2 / g2 - 1.0 / (g2 * g2));
        let mut sc = 0.0;
        let mut ss = 0.0;
        for b in 0..nat {
            let ph = g.dot(system.atoms[b].position);
            sc += q_atom[b] * ph.cos();
            ss += q_atom[b] * ph.sin();
        }
        let sf2 = sc * sc + ss * ss;
        let garr = g.to_array();
        for row in 0..3 {
            for col in 0..3 {
                let delta = if row == col { 1.0 } else { 0.0 };
                let dg2 = -2.0 * garr[row] * garr[col];
                deriv[(row, col)] += rec_prefactor * sf2 * (dw_dg2 * dg2 - delta * w_g);
            }
        }
    }

    let qtot: f64 = q_atom.iter().sum();
    if qtot != 0.0 {
        let background_deriv = 0.5 * qtot * qtot * std::f64::consts::PI / (alpha * alpha * volume);
        for axis in 0..3 {
            deriv[(axis, axis)] += background_deriv;
        }
    }

    deriv
}

fn shell_polynomial_log_derivative(
    si: &crate::basis::BasisShell,
    sj: &crate::basis::BasisShell,
    rvec: Vec3,
    r2: f64,
) -> Vec3 {
    let rad_sum = match (atomic_radius_bohr(si.z), atomic_radius_bohr(sj.z)) {
        (Ok(a), Ok(b)) => a + b,
        _ => return Vec3::zero(),
    };
    let rr = (r2.sqrt() / rad_sum).sqrt();
    let pi = si.poly_raw.unwrap_or(0.0);
    let pj = sj.poly_raw.unwrap_or(0.0);
    let fi = 1.0 + pi * rr;
    let fj = 1.0 + pj * rr;
    let poly = fi * fj;
    if poly.abs() <= 1.0e-18 {
        return Vec3::zero();
    }
    let dpoly = (fi * pj + fj * pi) * 0.5 * rr / r2;
    rvec * (dpoly / poly)
}

fn add_outer(matrix: &mut Matrix, vector: Vec3, scale: f64) {
    let v = vector.to_array();
    for row in 0..3 {
        for col in 0..3 {
            matrix[(row, col)] += scale * v[row] * v[col];
        }
    }
}

fn component(vector: Vec3, axis: usize) -> f64 {
    match axis {
        0 => vector.x,
        1 => vector.y,
        2 => vector.z,
        _ => unreachable!("axis must be 0..3"),
    }
}

fn add_matrix_in_place(lhs: &mut Matrix, rhs: &Matrix) {
    for (l, r) in lhs.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
        *l += *r;
    }
}

fn scale_matrix(matrix: &mut Matrix, scale: f64) {
    for value in matrix.as_mut_slice() {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::KMesh;

    fn load_params() -> Option<Gfn1Parameters> {
        let path = std::env::var("GFN1_XTB_PARAM").ok()?;
        Gfn1Parameters::from_file(path).ok()
    }

    #[test]
    fn stress_tensor_is_finite_for_gamma_pbc_water() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-10;
        options.charge_tolerance = 1.0e-8;
        let result = pbc_stress(&system, &params, &options, &PbcOptions::default()).unwrap();
        for value in result.stress.as_slice() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn stress_tensor_is_finite_for_kpoint_pbc_mgo() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"0 2.106 2.106 2.106 0 2.106 2.106 2.106 0\" pbc=\"T T T\"\n\
             Mg 0.000000 0.000000 0.000000\n\
             O  2.106000 0.000000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-9;
        options.charge_tolerance = 1.0e-7;
        let pbc = PbcOptions {
            kmesh: KMesh::monkhorst_pack([2, 1, 1]),
            ..PbcOptions::default()
        };
        let result = pbc_stress(&system, &params, &options, &pbc).unwrap();
        assert!(result.scf.converged, "k-point MgO SCC did not converge");
        for value in result.stress.as_slice() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn stress_matches_strain_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-10;
        options.charge_tolerance = 1.0e-8;
        let pbc = PbcOptions::default();
        let result = pbc_stress(&system, &params, &options, &pbc).unwrap();
        let volume = system.lattice.as_ref().unwrap().volume();
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let plus = strained_system(&system, row, col, h);
                let minus = strained_system(&system, row, col, -h);
                let ep = pbc_stress(&plus, &params, &options, &pbc)
                    .unwrap()
                    .total_energy;
                let em = pbc_stress(&minus, &params, &options, &pbc)
                    .unwrap()
                    .total_energy;
                let fd = (ep - em) / (2.0 * h * volume);
                max_delta = max_delta.max((result.stress[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 5.0e-6,
            "PBC stress vs strain finite-difference max delta {max_delta:.3e}"
        );
    }

    // A5: the periodic multipole stress (kernel-strain + overlap-Pulay-strain) folded into the full
    // SCC stress matches the finite difference of the periodic free energy on a polar cell, at rank
    // 2 (dipole+quadrupole). Closes Part A (periodic multipole: energy + gradient + stress).
    #[test]
    fn multipole_stress_matches_strain_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.multipole = true; // dipole + quadrupole
        options.energy_tolerance = 1.0e-10;
        options.charge_tolerance = 1.0e-8;
        let pbc = PbcOptions::default();
        let result = pbc_stress(&system, &params, &options, &pbc).unwrap();
        let volume = system.lattice.as_ref().unwrap().volume();
        let h = 1.0e-5;
        let mut max_delta = 0.0_f64;
        for row in 0..3 {
            for col in 0..3 {
                let plus = strained_system(&system, row, col, h);
                let minus = strained_system(&system, row, col, -h);
                let ep = run_pbc_scc(&plus, &params, &options, &pbc)
                    .unwrap()
                    .total_free;
                let em = run_pbc_scc(&minus, &params, &options, &pbc)
                    .unwrap()
                    .total_free;
                let fd = (ep - em) / (2.0 * h * volume);
                max_delta = max_delta.max((result.stress[(row, col)] - fd).abs());
            }
        }
        assert!(
            max_delta < 5.0e-5,
            "multipole stress vs strain finite-difference max delta {max_delta:.3e}"
        );
    }

    fn strained_system(
        system: &PeriodicSystem,
        row: usize,
        col: usize,
        delta: f64,
    ) -> PeriodicSystem {
        let mut out = system.clone();
        for atom in &mut out.atoms {
            atom.position = strain_vec(atom.position, row, col, delta);
        }
        if let Some(lattice) = out.lattice {
            let cell = [
                strain_vec(lattice.cell.col[0], row, col, delta),
                strain_vec(lattice.cell.col[1], row, col, delta),
                strain_vec(lattice.cell.col[2], row, col, delta),
            ];
            out.lattice =
                Some(Lattice::new(crate::math::Mat3 { col: cell }, lattice.periodic).unwrap());
        }
        out
    }

    fn strain_vec(vector: Vec3, row: usize, col: usize, delta: f64) -> Vec3 {
        let mut out = vector;
        let component = match col {
            0 => vector.x,
            1 => vector.y,
            _ => vector.z,
        };
        match row {
            0 => out.x += delta * component,
            1 => out.y += delta * component,
            _ => out.z += delta * component,
        }
        out
    }
}
