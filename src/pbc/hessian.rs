// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic Hessian for periodic GFN1-xTB (Gamma-point first, k-point to follow).
//!
//! The Hessian is the total derivative of the analytic gradient,
//! `H_{xy} = d g_x / d R_y`, split into a fixed-density (skeleton) second
//! derivative and a coupled-perturbed (CPXTB) density-response term. This module
//! is built and validated bottom-up and *independently*: every piece is checked
//! against the finite difference of the already-trusted analytic gradient (and,
//! for the skeleton matrices, against finite differences of the Gamma-point
//! `S`/`Fock`), rather than mirroring any other Hessian implementation.
//!
//! This first stage establishes and validates the Gamma-point skeleton
//! derivative matrices `dS(Gamma)/dR` and `dFock0(Gamma)/dR` (the fixed-density
//! one-electron Fock derivative), which feed both the CPXTB right-hand side and
//! the Hessian assembly.

use crate::basis::BasisSet;
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::harmonic_average;
use crate::data_tables::atomic_radius_bohr;
use crate::dispersion::dispersion_energy_gradient_hessian;
use crate::electronic::ElectronicOptions;
use crate::error::Result;
use crate::halogen::halogen_energy_gradient_hessian;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::{
    contracted_pair, contracted_pair_with_derivatives, contracted_pair_with_second_derivatives,
};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::{erfc, Vec3};
use crate::model::KPoint;
use crate::params::Gfn1Parameters;
use crate::pbc::complex::CMatrix;
use crate::pbc::ewald::{
    exp1, qcore_r3_real_value_derivatives, qcore_short_value_derivatives, resolve_alpha,
    QCORE_R3_COEFF,
};
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::{run_pbc_scc, PbcSccResult};
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use rayon::prelude::*;
use std::f64::consts::PI;

const SQRT_PI: f64 = 1.772_453_850_905_516;
const TAU: f64 = 5.5;
const DIST_EPS: f64 = 1.0e-12;
const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_808_578_545_117e-6;
/// Occupation window (exclusive of 0 and the doubly-occupied 2.0) that flags a
/// genuinely fractional / finite-temperature band, triggering the finite-T CPXTB
/// response instead of the integer occ-virt path.
const FRACTIONAL_OCC_EPS: f64 = 1.0e-10;

/// Per-Cartesian-DOF skeleton derivative matrices at the Gamma point.
#[derive(Clone, Debug)]
pub struct GammaSkeletonDerivatives {
    /// `dS(Gamma)/dR_y` for each DOF `y` (real, `n x n`).
    pub overlap: Vec<Matrix>,
    /// `dFock0(Gamma)/dR_y` (fixed-density one-electron Fock skeleton) per DOF.
    pub fock: Vec<Matrix>,
    /// `dV_shell/dR_y` (SCC scalar potential, fixed charges) per DOF, `[shell]`.
    pub shell_potential: Vec<Vec<f64>>,
    /// `dCN_A/dR_y` per DOF, `[atom]`.
    pub coordination: Vec<Vec<f64>>,
}

/// Build the Gamma-point skeleton derivative matrices for a converged periodic
/// SCC result. "Skeleton" means the explicit position derivative at fixed MO
/// coefficients / fixed shell charges: it does not contain the CPXTB response.
pub fn gamma_skeleton_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<GammaSkeletonDerivatives> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic Hessian requires a lattice");
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;

    // CN derivatives dCN_A/dR_y (periodic, [dof][atom]).
    let coordination = coordination_derivatives(system, options.hamiltonian.coordination_cutoff)?;

    // Scalar SCC potential derivatives dV_shell/dR_y (fixed charges).
    let shell_potential = shell_potential_derivatives(system, &lattice, scf, pbc)?;

    // Self-energies and their CN coupling.
    let self_energy = &scf.bloch.self_energies;
    let dsedcn = &scf.bloch.dsedcn;

    // AO-resolved fixed SCC potential.
    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }

    let mut overlap = vec![Matrix::zeros(n, n); ndof];
    let mut fock = vec![Matrix::zeros(n, n); ndof];

    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }

    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if p.exponent < *e {
                *e = p.exponent;
            }
        }
    }

    // Off-site image pairs: every ordered (a, b, T) except the on-site (a==b, T=0)
    // block. Each contributes to dS(Gamma)/dR and dFock0(Gamma)/dR at the bra and
    // ket atom centres.
    for off in &images {
        let is_origin = off.is_origin();
        let translation = lattice.translation(*off);
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for b in 0..nat {
                if is_origin && a == b {
                    continue;
                }
                let rb = system.atoms[b].position + translation;
                let rvec = ra - rb;
                let r2 = rvec.norm2();
                if r2 <= DIST_EPS || r2 > cutoff2 {
                    continue;
                }
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let rad_sum =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
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
                        let s = moments[0];
                        let ds_bra = d_bra[0];
                        let ds_ket = d_ket[0];
                        let hs = hscale(si, sj, params)? * shell_polynomial(si, sj, rr);
                        let hij = 0.5 * (self_energy[si_idx] + self_energy[sj_idx]) * hs;
                        // d(poly)/dR via log-derivative (acts at the distance).
                        let dlog_poly = shell_polynomial_log_derivative(si, sj, rvec, r2);
                        // h0 = hij * S ; dh0 = hij*dS + S*dhij ; dhij (poly part) =
                        // hij * dlog_poly w.r.t. bra (+) / ket (-).
                        for axis in 0..3 {
                            let ds_bra_a = component(ds_bra, axis);
                            let ds_ket_a = component(ds_ket, axis);
                            let dpoly_a = component(dlog_poly, axis);
                            // dS(Gamma)
                            overlap[3 * a + axis][(mu, nu)] += ds_bra_a;
                            overlap[3 * b + axis][(mu, nu)] += ds_ket_a;
                            // dH0(Gamma): overlap-derivative * hij + S * d(hij_poly)
                            let dh0_bra = hij * ds_bra_a + hij * dpoly_a * s;
                            let dh0_ket = hij * ds_ket_a - hij * dpoly_a * s;
                            fock[3 * a + axis][(mu, nu)] += dh0_bra;
                            fock[3 * b + axis][(mu, nu)] += dh0_ket;
                            // Pulay potential at fixed charges: -1/2 (V_mu+V_nu) dS.
                            let pulay_bra = -0.5 * (vao[mu] + vao[nu]) * ds_bra_a;
                            let pulay_ket = -0.5 * (vao[mu] + vao[nu]) * ds_ket_a;
                            fock[3 * a + axis][(mu, nu)] += pulay_bra;
                            fock[3 * b + axis][(mu, nu)] += pulay_ket;
                        }
                        // CN contribution to dH0: dh_i/dCN * dCN/dR, applied to S.
                        if enable_cn {
                            let dhij_dcn_i = 0.5 * dsedcn[si_idx] * hs * s;
                            let dhij_dcn_j = 0.5 * dsedcn[sj_idx] * hs * s;
                            for y in 0..ndof {
                                let dcn_i = coordination[y][si.atom_index];
                                let dcn_j = coordination[y][sj.atom_index];
                                fock[y][(mu, nu)] += dhij_dcn_i * dcn_i + dhij_dcn_j * dcn_j;
                            }
                        }
                    }
                }
            }
        }
    }

    // On-site (a==b, T=0) self-energy CN derivative: H0_onsite = 1/2 (se_i+se_j)
    // S0, whose CN dependence the off-site loop skips. S0 is the on-site overlap
    // (identity on the diagonal, ~0 between different same-atom shells).
    if enable_cn {
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for &mu in &atom_aos[a] {
                let si_idx = basis.aos[mu].shell_index;
                for &nu in &atom_aos[a] {
                    let sj_idx = basis.aos[nu].shell_index;
                    let s0 = contracted_pair(&basis.aos[mu], &basis.aos[nu], ra, ra).0;
                    if s0 == 0.0 {
                        continue;
                    }
                    let coeff = 0.5 * (dsedcn[si_idx] + dsedcn[sj_idx]) * s0;
                    for y in 0..ndof {
                        fock[y][(mu, nu)] += coeff * coordination[y][a];
                    }
                }
            }
        }
    }

    // Scalar SCC potential derivative: -1/2 (dV_mu+dV_nu) S(Gamma), added to all
    // AO pairs (including the on-site overlap, which equals delta but is captured
    // by the folded Gamma overlap).
    let (_, s_gamma) = scf.bloch.h_s_gamma_real();
    for y in 0..ndof {
        for mu in 0..n {
            let dv_mu = shell_potential[y][basis.aos[mu].shell_index];
            for nu in 0..n {
                let dv_nu = shell_potential[y][basis.aos[nu].shell_index];
                fock[y][(mu, nu)] += -0.5 * (dv_mu + dv_nu) * s_gamma[(mu, nu)];
            }
        }
    }

    Ok(GammaSkeletonDerivatives {
        overlap,
        fock,
        shell_potential,
        coordination,
    })
}

/// Per-Cartesian-DOF complex skeleton derivative matrices at a general k-point.
#[derive(Clone, Debug)]
pub struct KpointSkeleton {
    /// `dS(k)/dR_y` per DOF (complex Bloch sum).
    pub overlap: Vec<CMatrix>,
    /// `dFock0(k)/dR_y` (fixed-density one-electron Fock skeleton) per DOF.
    pub fock: Vec<CMatrix>,
    /// `dV_shell/dR_y` (real, shared with the Gamma path) per DOF, `[shell]`.
    pub shell_potential: Vec<Vec<f64>>,
    /// `dCN_A/dR_y` per DOF, `[atom]`.
    pub coordination: Vec<Vec<f64>>,
}

/// Build the complex skeleton derivative matrices `dS(k)/dR` and `dFock0(k)/dR`
/// at a general fractional k-point. Identical in structure to
/// [`gamma_skeleton_derivatives`] but each image `T` carries the Bloch phase
/// `e^{i k.T}`, so the matrices are complex Hermitian-derivative blocks. Like the
/// Gamma path, all AO image pairs are iterated directly (NOT via the overlap-
/// filtered Bloch builder), so zero-overlap pairs with nonzero derivatives are
/// retained.
pub fn kpoint_skeleton_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    fractional: [f64; 3],
) -> Result<KpointSkeleton> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic Hessian requires a lattice");
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;

    let coordination = coordination_derivatives(system, options.hamiltonian.coordination_cutoff)?;
    let shell_potential = shell_potential_derivatives(system, &lattice, scf, pbc)?;
    let self_energy = &scf.bloch.self_energies;
    let dsedcn = &scf.bloch.dsedcn;

    let mut vao = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }

    let mut overlap = vec![CMatrix::zeros(n); ndof];
    let mut fock = vec![CMatrix::zeros(n); ndof];

    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }

    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if p.exponent < *e {
                *e = p.exponent;
            }
        }
    }

    for off in &images {
        let is_origin = off.is_origin();
        let (cph, sph) = bloch_phase(fractional, *off);
        let translation = lattice.translation(*off);
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for b in 0..nat {
                if is_origin && a == b {
                    continue;
                }
                let rb = system.atoms[b].position + translation;
                let rvec = ra - rb;
                let r2 = rvec.norm2();
                if r2 <= DIST_EPS || r2 > cutoff2 {
                    continue;
                }
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let rad_sum =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
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
                        let s = moments[0];
                        let ds_bra = d_bra[0];
                        let ds_ket = d_ket[0];
                        let hs = hscale(si, sj, params)? * shell_polynomial(si, sj, rr);
                        let hij = 0.5 * (self_energy[si_idx] + self_energy[sj_idx]) * hs;
                        let dlog_poly = shell_polynomial_log_derivative(si, sj, rvec, r2);
                        for axis in 0..3 {
                            let ds_bra_a = component(ds_bra, axis);
                            let ds_ket_a = component(ds_ket, axis);
                            let dpoly_a = component(dlog_poly, axis);
                            let ra_dof = 3 * a + axis;
                            let rb_dof = 3 * b + axis;
                            overlap[ra_dof].re[(mu, nu)] += ds_bra_a * cph;
                            overlap[ra_dof].im[(mu, nu)] += ds_bra_a * sph;
                            overlap[rb_dof].re[(mu, nu)] += ds_ket_a * cph;
                            overlap[rb_dof].im[(mu, nu)] += ds_ket_a * sph;
                            let dh0_bra = hij * ds_bra_a + hij * dpoly_a * s
                                - 0.5 * (vao[mu] + vao[nu]) * ds_bra_a;
                            let dh0_ket = hij * ds_ket_a
                                - hij * dpoly_a * s
                                - 0.5 * (vao[mu] + vao[nu]) * ds_ket_a;
                            fock[ra_dof].re[(mu, nu)] += dh0_bra * cph;
                            fock[ra_dof].im[(mu, nu)] += dh0_bra * sph;
                            fock[rb_dof].re[(mu, nu)] += dh0_ket * cph;
                            fock[rb_dof].im[(mu, nu)] += dh0_ket * sph;
                        }
                        if enable_cn {
                            let dhij_dcn_i = 0.5 * dsedcn[si_idx] * hs * s;
                            let dhij_dcn_j = 0.5 * dsedcn[sj_idx] * hs * s;
                            for y in 0..ndof {
                                let add = dhij_dcn_i * coordination[y][si.atom_index]
                                    + dhij_dcn_j * coordination[y][sj.atom_index];
                                fock[y].re[(mu, nu)] += add * cph;
                                fock[y].im[(mu, nu)] += add * sph;
                            }
                        }
                    }
                }
            }
        }
    }

    // On-site (T=0, phase 1) self-energy CN derivative.
    if enable_cn {
        for a in 0..nat {
            let ra = system.atoms[a].position;
            for &mu in &atom_aos[a] {
                let si_idx = basis.aos[mu].shell_index;
                for &nu in &atom_aos[a] {
                    let sj_idx = basis.aos[nu].shell_index;
                    let s0 = contracted_pair(&basis.aos[mu], &basis.aos[nu], ra, ra).0;
                    if s0 == 0.0 {
                        continue;
                    }
                    let coeff = 0.5 * (dsedcn[si_idx] + dsedcn[sj_idx]) * s0;
                    for y in 0..ndof {
                        fock[y].re[(mu, nu)] += coeff * coordination[y][a];
                    }
                }
            }
        }
    }

    // Scalar SCC potential derivative: -1/2 (dV_mu+dV_nu) S(k).
    let (_, s_k) = scf.bloch.h_s_at_k(fractional);
    for y in 0..ndof {
        for mu in 0..n {
            let dv_mu = shell_potential[y][basis.aos[mu].shell_index];
            for nu in 0..n {
                let dv_nu = shell_potential[y][basis.aos[nu].shell_index];
                let scale = -0.5 * (dv_mu + dv_nu);
                fock[y].re[(mu, nu)] += scale * s_k.re[(mu, nu)];
                fock[y].im[(mu, nu)] += scale * s_k.im[(mu, nu)];
            }
        }
    }

    Ok(KpointSkeleton {
        overlap,
        fock,
        shell_potential,
        coordination,
    })
}

/// Periodic CN derivatives `dCN_A/dR_y`, indexed `[dof][atom]`.
fn coordination_derivatives(system: &PeriodicSystem, cutoff: f64) -> Result<Vec<Vec<f64>>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut out = vec![vec![0.0_f64; nat]; ndof];
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    for pair in cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let r = pair.r_ij.norm();
        if r <= DIST_EPS {
            continue;
        }
        let deriv = pair.r_ij * (pair.dcn_dr / r);
        for axis in 0..3 {
            let value = component(deriv, axis);
            out[3 * pair.i + axis][pair.i] += value;
            out[3 * pair.i + axis][pair.j] += value;
            out[3 * pair.j + axis][pair.i] -= value;
            out[3 * pair.j + axis][pair.j] -= value;
        }
    }
    Ok(out)
}

/// Periodic SCC scalar potential derivatives `dV_shell/dR_y` at fixed shell
/// charges, `[dof][shell]`. `V_i = sum_j Gamma_ij q_j`, with the periodic
/// QCore `Gamma` (`1/R` Ewald + generalized `R^-3` Ewald + KO residual).
fn shell_potential_derivatives(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
) -> Result<Vec<Vec<f64>>> {
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let q_atom = &scf.atomic_charges;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let nsh = basis.shells.len();
    let mut out = vec![vec![0.0_f64; nsh]; ndof];

    // Ewald 1/R part: dV_i^ew/dR_c depends only on the atoms. Build the atomic
    // potential gradient PHI_DERIV[A][c] = d(sum_B Q_B phi_{A,B})/dR_c.
    let alpha = resolve_alpha(system, &pbc.ewald);
    let phi_deriv = ewald_atom_potential_derivatives(system, lattice, alpha, q_atom);
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for c in 0..nat {
            for axis in 0..3 {
                out[3 * c + axis][i] += phi_deriv[ai][3 * c + axis];
            }
        }
    }
    qcore_r3_shell_potential_reciprocal_derivatives(
        system, lattice, alpha, basis, model, q, &mut out,
    );

    // QCore real-space R^-3 Ewald term plus short-range residual.
    let r3_cut = TAU / alpha;
    let sr_cut = pbc.ewald.sr_cutoff;
    let real_cut = r3_cut.max(sr_cut);
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        let ra = system.atoms[ai].position;
        for j in 0..nsh {
            let aj = basis.shells[j].atom_index;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
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
                let grad = vec * (radial / d); // d gamma / d R_ai
                for axis in 0..3 {
                    let g = component(grad, axis);
                    out[3 * ai + axis][i] += g * q[j];
                    out[3 * aj + axis][i] -= g * q[j];
                }
            }
        }
    }

    Ok(out)
}

fn qcore_r3_shell_potential_reciprocal_derivatives(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    basis: &BasisSet,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    out: &mut [Vec<f64>],
) {
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let pref0 = QCORE_R3_COEFF * 2.0 * PI / lattice.volume();
    let nsh = basis.shells.len();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff = pref0 * exp1(g.norm2() * inv_4a2);
        let garr = g.to_array();
        for i in 0..nsh {
            let ai = basis.shells[i].atom_index;
            for j in 0..nsh {
                if q[j] == 0.0 {
                    continue;
                }
                let aj = basis.shells[j].atom_index;
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let scale = coeff * q[j] / (eta * eta) * (phases[i] - phases[j]).sin();
                for axis in 0..3 {
                    let value = -scale * garr[axis];
                    out[3 * ai + axis][i] += value;
                    out[3 * aj + axis][i] -= value;
                }
            }
        }
    }
}

/// Ewald atomic potential derivatives `PHI_DERIV[A][3c+axis] = d/dR_c (sum_B Q_B
/// phi_{A,B})`, for the `1/R` Ewald potential. Real `erfc` + reciprocal parts.
fn ewald_atom_potential_derivatives(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    q_atom: &[f64],
) -> Vec<Vec<f64>> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let volume = lattice.volume();
    let real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / volume;
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;

    let mut out = vec![vec![0.0_f64; ndof]; nat];

    // Real space: d phi_{A,B}/dR_A = g'(d) (R_A - R_B - T)/d, summed over B,T.
    for a in 0..nat {
        for b in 0..nat {
            for t in &translations {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                let dgdr = -erfc(alpha * d) / (d * d)
                    - two_alpha_sqrtpi * (-alpha * alpha * d * d).exp() / d;
                let grad = vec * (dgdr / d); // d phi_{A,B}/dR_A
                for axis in 0..3 {
                    let g = component(grad, axis) * q_atom[b];
                    // V_A = sum_B Q_B phi_{A,B}: dV_A/dR_A += Q_B dphi/dR_A
                    out[a][3 * a + axis] += g;
                    // dV_A/dR_B += Q_B dphi/dR_B = -Q_B dphi/dR_A
                    out[a][3 * b + axis] -= g;
                }
            }
        }
    }

    // Reciprocal space: phi_{A,B} = (4pi/V) sum_G w_G cos(G.(R_A-R_B)).
    // d/dR_A = -(4pi/V) sum_G w_G G sin(G.(R_A-R_B)).
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = (-g2 * inv_4a2).exp() / g2;
        for a in 0..nat {
            for b in 0..nat {
                let phase = g.dot(system.atoms[a].position - system.atoms[b].position);
                let s = phase.sin();
                let coeff = -four_pi_v * w_g * s * q_atom[b];
                for axis in 0..3 {
                    let gax = component(*g, axis);
                    out[a][3 * a + axis] += coeff * gax;
                    out[a][3 * b + axis] -= coeff * gax;
                }
            }
        }
    }

    out
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

#[inline]
fn component(v: Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Gamma-point molecular orbitals (real) for the CPXTB.
#[derive(Clone, Debug)]
pub struct GammaMos {
    pub coeff: Matrix,
    pub energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub overlap: Matrix,
}

/// Solve the Gamma-point real generalized eigenproblem of the converged Fock to
/// obtain MOs, orbital energies, and integer occupations (gapped systems).
pub fn gamma_mos(scf: &PbcSccResult, nelec: f64) -> Result<GammaMos> {
    let (h0, s) = scf.bloch.h_s_gamma_real();
    let n = scf.basis.len();
    let mut vao = vec![0.0; n];
    for (ish, shell) in scf.basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let mut fock = h0;
    for i in 0..n {
        for j in 0..n {
            fock[(i, j)] -= 0.5 * (vao[i] + vao[j]) * s[(i, j)];
        }
    }
    let eig = crate::linalg::lowdin_solve_generalized(&fock, &s, 1.0e-12)?;
    // Fermi-Dirac occupations at the SCC electronic temperature. For kt = 0 (or a
    // gapped system, where exp(-gap/kt) underflows) this reduces to the integer
    // Aufbau filling, so gapped insulators take the existing integer CPXTB path.
    let kt = scf.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let occupations = fermi_fill_occupations(&eig.values, nelec, kt);
    Ok(GammaMos {
        coeff: eig.vectors,
        energies: eig.values,
        occupations,
        overlap: s,
    })
}

/// Doubly-occupied Fermi-Dirac fill (occupations in `[0, 2]`) for a single
/// (Gamma) k-point. Bisects the Fermi level to reproduce `nelec`. For `kt <= 0`
/// it returns the integer Aufbau filling.
fn fermi_fill_occupations(energies: &[f64], nelec: f64, kt: f64) -> Vec<f64> {
    let n = energies.len();
    if kt <= 0.0 || n == 0 {
        let mut occ = vec![0.0; n];
        let mut remaining = nelec.max(0.0);
        for o in &mut occ {
            let fill = remaining.min(2.0);
            *o = fill;
            remaining -= fill;
            if remaining <= 0.0 {
                break;
            }
        }
        return occ;
    }
    let min_e = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_e = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut lo = min_e - 100.0 * kt - 10.0;
    let mut hi = max_e + 100.0 * kt + 10.0;
    let count = |mu: f64| energies.iter().map(|&e| fermi_occ2(e, mu, kt)).sum::<f64>();
    for _ in 0..200 {
        let mu = 0.5 * (lo + hi);
        if count(mu) < nelec {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    let mu = 0.5 * (lo + hi);
    energies.iter().map(|&e| fermi_occ2(e, mu, kt)).collect()
}

#[inline]
fn fermi_occ2(eps: f64, mu: f64, kt: f64) -> f64 {
    let x = ((eps - mu) / kt).clamp(-80.0, 80.0);
    2.0 / (1.0 + x.exp())
}

/// Complex MOs at a general k-point: the `2n x 2n` real-embedding eigensolution
/// of `Fock(k) = H0(k) - 1/2 (V_mu+V_nu) S(k)`, with integer single-electron
/// occupations (1 on the lowest `nelec` embedded states; gapped insulators).
pub struct KpointMos {
    pub eig: crate::pbc::complex::KEigen,
    /// Length `2n`, 1.0 on the lowest `nelec` embedded states.
    pub occupations: Vec<f64>,
    /// `S(k)` (for Mulliken transition charges).
    pub overlap: CMatrix,
    /// `Fock(k)` (for energy-weighted-density responses).
    pub fock: CMatrix,
}

pub fn kpoint_mos(scf: &PbcSccResult, ik: usize) -> Result<KpointMos> {
    let n = scf.basis.len();
    let (h0, s) = &scf.hs_k[ik];
    let mut vao = vec![0.0; n];
    for (ish, shell) in scf.basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let mut fock = h0.clone();
    for i in 0..n {
        for j in 0..n {
            let scale = 0.5 * (vao[i] + vao[j]);
            fock.re[(i, j)] -= scale * s.re[(i, j)];
            fock.im[(i, j)] -= scale * s.im[(i, j)];
        }
    }
    let eig = crate::pbc::complex::hermitian_generalized_eigen(&fock, s, 1.0e-12)?;
    let nfill = scf.nelec.round() as usize;
    let mut occupations = vec![0.0; 2 * n];
    for occ in occupations.iter_mut().take(nfill) {
        *occ = 1.0;
    }
    Ok(KpointMos {
        eig,
        occupations,
        overlap: s.clone(),
        fock,
    })
}

/// Physical complex MOs at a k-point: `n` bands extracted from the `2n`
/// real-embedding eigensolution. Each physical band is a degenerate embedded
/// pair `(2p, 2p+1)`; the 2-D real eigenspace is a 1-D complex ray, so *any*
/// representative `phi = [u; v]` yields a valid `S(k)`-normalised MO
/// `c = u + i v` (unique up to a global phase that drops out of the density).
/// Taking the lower index `2p` of each pair gives the band coefficients,
/// energies, and integer (doubly-occupied) band occupations.
pub struct KpointComplexMos {
    /// `n x n` complex MO coefficients `C(k)`; column `p` is band `p`.
    pub coeff: CMatrix,
    /// Physical band energies (length `n`).
    pub energies: Vec<f64>,
    /// Physical occupations (length `n`): `2.0` for a doubly-occupied band.
    pub occupations: Vec<f64>,
    /// `S(k)` overlap.
    pub overlap: CMatrix,
    /// `Fock(k)`.
    pub fock: CMatrix,
}

pub fn kpoint_complex_mos(scf: &PbcSccResult, ik: usize) -> Result<KpointComplexMos> {
    let km = kpoint_mos(scf, ik)?;
    let n = scf.basis.len();
    let nelec = scf.nelec.round() as usize;
    // Finite-temperature band occupations use the single global Fermi level the
    // SCC already solved over the whole Brillouin zone: occ = 2 f(eps, mu, kt).
    // For kt = 0 (or a gapped band, where exp underflows) this is the integer fill.
    let kt = scf.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let mut coeff = CMatrix::zeros(n);
    let mut energies = vec![0.0; n];
    let mut occupations = vec![0.0; n];
    for p in 0..n {
        let col = 2 * p; // representative of the degenerate embedded pair
        for mu in 0..n {
            coeff.re[(mu, p)] = km.eig.vectors[(mu, col)]; // u
            coeff.im[(mu, p)] = km.eig.vectors[(n + mu, col)]; // v
        }
        energies[p] = km.eig.values[col];
        occupations[p] = if kt > 0.0 {
            fermi_occ2(energies[p], scf.fermi_level, kt)
        } else if col < nelec {
            2.0
        } else {
            0.0
        };
    }
    Ok(KpointComplexMos {
        coeff,
        energies,
        occupations,
        overlap: km.overlap,
        fock: km.fock,
    })
}

/// Complex matrix product `A * B` (both `n x n`).
fn cmatmul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let n = a.n;
    let rr = a.re.matmul(&b.re).expect("re*re");
    let ii = a.im.matmul(&b.im).expect("im*im");
    let ri = a.re.matmul(&b.im).expect("re*im");
    let ir = a.im.matmul(&b.re).expect("im*re");
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = rr[(i, j)] - ii[(i, j)];
            out.im[(i, j)] = ri[(i, j)] + ir[(i, j)];
        }
    }
    out
}

/// `<i|Op|a> = sum_mu conj(C_mu,i) (Op C)_mu,a` given the precomputed `opc = Op*C`.
/// Returns the complex value as `(re, im)`.
#[inline]
fn cmo_element(coeff: &CMatrix, opc: &CMatrix, i: usize, a: usize) -> (f64, f64) {
    let n = coeff.n;
    let mut re = 0.0;
    let mut im = 0.0;
    for mu in 0..n {
        let cr = coeff.re[(mu, i)];
        let ci = coeff.im[(mu, i)];
        let or = opc.re[(mu, a)];
        let oi = opc.im[(mu, a)];
        // (cr - i ci)(or + i oi)
        re += cr * or + ci * oi;
        im += cr * oi - ci * or;
    }
    (re, im)
}

/// Per-k-point data cached across DOFs for the complex CPXTB.
struct KCpxtbData {
    mos: KpointComplexMos,
    pairs: Vec<(usize, usize)>,
    gaps: Vec<f64>,
    weight: f64,
    /// Complex Mulliken transition charge `Q_ia,s` per pair, per shell.
    q_re: Vec<Vec<f64>>,
    q_im: Vec<Vec<f64>>,
}

/// Build the per-k CPXTB data: complex MOs, occ-virt pairs, gaps, and the complex
/// transition charges `Q_ia,s = sum_{mu in s} [ C_mu,a conj(SC_mu,i) +
/// conj(C_mu,i) SC_mu,a ]`. `Q` is the negative of the Gamma `transition_charges`
/// in the real limit; the same `Q` drives both the induced charge and the Fock
/// coupling.
fn build_kcpxtb_data(scf: &PbcSccResult, ik: usize) -> Result<KCpxtbData> {
    let mos = kpoint_complex_mos(scf, ik)?;
    let pairs = occ_virt_pairs(&mos.occupations);
    let gaps: Vec<f64> = pairs
        .iter()
        .map(|&(i, a)| mos.energies[a] - mos.energies[i])
        .collect();
    let sc = cmatmul(&mos.overlap, &mos.coeff);
    let n = scf.basis.len();
    let nsh = scf.basis.shells.len();
    let mut q_re = vec![vec![0.0; nsh]; pairs.len()];
    let mut q_im = vec![vec![0.0; nsh]; pairs.len()];
    for (row, &(i, a)) in pairs.iter().enumerate() {
        for mu in 0..n {
            let s = scf.basis.aos[mu].shell_index;
            let (car, cai) = (mos.coeff.re[(mu, a)], mos.coeff.im[(mu, a)]);
            let (cir, cii) = (mos.coeff.re[(mu, i)], mos.coeff.im[(mu, i)]);
            let (sir, sii) = (sc.re[(mu, i)], sc.im[(mu, i)]);
            let (sar, sai) = (sc.re[(mu, a)], sc.im[(mu, a)]);
            // C_a conj(SC_i) = (car + i cai)(sir - i sii)
            let t1r = car * sir + cai * sii;
            let t1i = cai * sir - car * sii;
            // conj(C_i) SC_a = (cir - i cii)(sar + i sai)
            let t2r = cir * sar + cii * sai;
            let t2i = cir * sai - cii * sar;
            q_re[row][s] += t1r + t2r;
            q_im[row][s] += t1i + t2i;
        }
    }
    Ok(KCpxtbData {
        mos,
        pairs,
        gaps,
        weight: scf.kpoints[ik].weight,
        q_re,
        q_im,
    })
}

/// Occupied-occupied complex metric density response from the overlap derivative
/// `S^1(k)`: `dP_metric_munu = sum_ij w_ij C_mu,i conj(C_nu,j)`,
/// `w_ij = -1/2 (focc_i+focc_j) <i|S^1|j>`. Hermitian.
fn complex_metric_density(mos: &KpointComplexMos, sc1: &CMatrix) -> CMatrix {
    let n = mos.coeff.n;
    let occ = &mos.occupations;
    let mut out = CMatrix::zeros(n);
    for i in 0..occ.len() {
        if occ[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occ.len() {
            if occ[j] <= 1.0e-8 {
                continue;
            }
            let (s1r, s1i) = cmo_element(&mos.coeff, sc1, i, j);
            let scale = -0.5 * (occ[i] + occ[j]);
            let (wr, wi) = (scale * s1r, scale * s1i);
            if wr.abs() <= 1.0e-30 && wi.abs() <= 1.0e-30 {
                continue;
            }
            for mu in 0..n {
                let (air, aii) = (mos.coeff.re[(mu, i)], mos.coeff.im[(mu, i)]);
                for nu in 0..n {
                    let (bjr, bji) = (mos.coeff.re[(nu, j)], mos.coeff.im[(nu, j)]);
                    // C_mu,i conj(C_nu,j) = (air+i aii)(bjr - i bji)
                    let gr = air * bjr + aii * bji;
                    let gi = aii * bjr - air * bji;
                    out.re[(mu, nu)] += wr * gr - wi * gi;
                    out.im[(mu, nu)] += wr * gi + wi * gr;
                }
            }
        }
    }
    out
}

/// Accumulate the real shell-charge response at one k-point into `dq` (weighted):
/// `dq_s -= w_k Re[ (dP S(k) + P0(k) S^1(k))_mu mu ]`, summed over `mu in s`.
/// `dP` is a complex density response and `s1 = S^1(k)` the overlap derivative.
fn accumulate_complex_shell_charges(
    dq: &mut [f64],
    scf: &PbcSccResult,
    ik: usize,
    mos: &KpointComplexMos,
    dp: &CMatrix,
    s1: &CMatrix,
    weight: f64,
) {
    let n = scf.basis.len();
    let dps = cmatmul(dp, &mos.overlap);
    let p0s1 = cmatmul(&scf.density_k[ik], s1);
    for mu in 0..n {
        let s = scf.basis.aos[mu].shell_index;
        dq[s] -= weight * (dps.re[(mu, mu)] + p0s1.re[(mu, mu)]);
    }
}

/// Solve the k-point CPXTB for every Cartesian DOF and return the complex AO
/// density responses `dP(k)/dR_y`, indexed `[dof][k]`. With `couple = false` the
/// SCC charge response is switched off (frozen-potential orbital relaxation only),
/// which isolates and validates the orbital-relaxation and metric machinery.
///
/// The coupling (when enabled) is solved by a fixed-point iteration
/// `gap u = b_eff + 1/2 Q (K dq_ov(u))`, where `dq_ov(u) = -sum_k w_k sum_ia
/// (focc_i-focc_a) Re[u_ia Q_ia]` is the occupied-virtual charge response and the
/// constant metric/overlap-derivative charge is folded into `b_eff`. This avoids
/// the non-symmetry of the complex coupled operator under the naive real inner
/// product (the Gamma PCG works only because its `Q` is real).
/// Apply the coupled complex k-point CPXTB operator `M u = diag(gap) u - 0.5 C u`.
/// `C` is the SCC response that couples every k-point through the shell kernel; the
/// resulting `M` is symmetric positive definite (the gaps are positive and the SCC
/// kernel is positive semidefinite), so the system can be solved with CG.
fn kpoint_cpxtb_matvec(
    kdata: &[KCpxtbData],
    kernel: &Matrix,
    nsh: usize,
    u_re: &[Vec<f64>],
    u_im: &[Vec<f64>],
    out_re: &mut [Vec<f64>],
    out_im: &mut [Vec<f64>],
) -> Result<()> {
    // Induced shell charge dq(u) = - sum_k w_k sum_row 2 Re[u Q^*].
    let mut dq = vec![0.0; nsh];
    for (ik, kd) in kdata.iter().enumerate() {
        for row in 0..kd.pairs.len() {
            let (ur, ui) = (u_re[ik][row], u_im[ik][row]);
            for s in 0..nsh {
                dq[s] -= kd.weight * 2.0 * (ur * kd.q_re[row][s] + ui * kd.q_im[row][s]);
            }
        }
    }
    let pot = crate::linalg::matrix_vector_product(kernel, &dq)?;
    for (ik, kd) in kdata.iter().enumerate() {
        for row in 0..kd.pairs.len() {
            let mut addr = 0.0;
            let mut addi = 0.0;
            for s in 0..nsh {
                addr += kd.q_re[row][s] * pot[s];
                addi += kd.q_im[row][s] * pot[s];
            }
            out_re[ik][row] = kd.gaps[row] * u_re[ik][row] - 0.5 * addr;
            out_im[ik][row] = kd.gaps[row] * u_im[ik][row] - 0.5 * addi;
        }
    }
    Ok(())
}

fn kcpxtb_zeros(kdata: &[KCpxtbData]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let v: Vec<Vec<f64>> = kdata.iter().map(|kd| vec![0.0; kd.pairs.len()]).collect();
    (v.clone(), v)
}

/// k-point-weighted inner product `<a, b>_w = sum_k w_k (a_re.b_re + a_im.b_im)`.
/// The coupled CPXTB operator is self-adjoint only in this weighted metric (the SCC
/// coupling carries the source k-point weight), so PCG must use it to be a valid
/// conjugate-gradient solve (a plain inner product makes the operator
/// non-symmetric and CG can stall).
fn kcpxtb_dot(
    kdata: &[KCpxtbData],
    ar: &[Vec<f64>],
    ai: &[Vec<f64>],
    br: &[Vec<f64>],
    bi: &[Vec<f64>],
) -> f64 {
    let mut s = 0.0;
    for (ik, kd) in kdata.iter().enumerate() {
        let w = kd.weight;
        for row in 0..kd.pairs.len() {
            s += w * (ar[ik][row] * br[ik][row] + ai[ik][row] * bi[ik][row]);
        }
    }
    s
}

/// Solve the coupled complex k-point CPXTB `M u = rhs` with preconditioned
/// conjugate gradient (Jacobi/gap preconditioner). Replaces the earlier naive
/// fixed-point iteration, which diverged when the SCC coupling exceeded the gaps.
fn solve_kpoint_cpxtb_pcg(
    kdata: &[KCpxtbData],
    kernel: &Matrix,
    nsh: usize,
    rhs_re: &[Vec<f64>],
    rhs_im: &[Vec<f64>],
    u_re: &mut [Vec<f64>],
    u_im: &mut [Vec<f64>],
) -> Result<()> {
    const PRECOND_FLOOR: f64 = 1.0e-4;
    let nk = kdata.len();
    let total: usize = kdata.iter().map(|kd| kd.pairs.len()).sum();
    if total == 0 {
        return Ok(());
    }

    // r = rhs - M u0
    let (mut mu_re, mut mu_im) = kcpxtb_zeros(kdata);
    kpoint_cpxtb_matvec(kdata, kernel, nsh, u_re, u_im, &mut mu_re, &mut mu_im)?;
    let (mut r_re, mut r_im) = kcpxtb_zeros(kdata);
    for ik in 0..nk {
        for row in 0..kdata[ik].pairs.len() {
            r_re[ik][row] = rhs_re[ik][row] - mu_re[ik][row];
            r_im[ik][row] = rhs_im[ik][row] - mu_im[ik][row];
        }
    }
    // z = M_precond^{-1} r
    let (mut z_re, mut z_im) = kcpxtb_zeros(kdata);
    let apply_precond =
        |r_re: &[Vec<f64>], r_im: &[Vec<f64>], z_re: &mut [Vec<f64>], z_im: &mut [Vec<f64>]| {
            for (ik, kd) in kdata.iter().enumerate() {
                for row in 0..kd.pairs.len() {
                    let inv = 1.0 / kd.gaps[row].max(PRECOND_FLOOR);
                    z_re[ik][row] = r_re[ik][row] * inv;
                    z_im[ik][row] = r_im[ik][row] * inv;
                }
            }
        };
    apply_precond(&r_re, &r_im, &mut z_re, &mut z_im);
    let (mut p_re, mut p_im) = (z_re.clone(), z_im.clone());
    let mut rz = kcpxtb_dot(kdata, &r_re, &r_im, &z_re, &z_im);

    let rhs_norm = kcpxtb_dot(kdata, rhs_re, rhs_im, rhs_re, rhs_im)
        .sqrt()
        .max(1.0);
    let tol = 1.0e-10 * rhs_norm;
    let max_iter = (4 * total).clamp(50, 2000);
    let (mut ap_re, mut ap_im) = kcpxtb_zeros(kdata);
    for _ in 0..max_iter {
        let rnorm = kcpxtb_dot(kdata, &r_re, &r_im, &r_re, &r_im).sqrt();
        if !rnorm.is_finite() {
            break;
        }
        if rnorm <= tol {
            break;
        }
        kpoint_cpxtb_matvec(kdata, kernel, nsh, &p_re, &p_im, &mut ap_re, &mut ap_im)?;
        let pap = kcpxtb_dot(kdata, &p_re, &p_im, &ap_re, &ap_im);
        if !(pap.is_finite() && pap.abs() > 1.0e-30) {
            break;
        }
        let alpha = rz / pap;
        for ik in 0..nk {
            for row in 0..kdata[ik].pairs.len() {
                u_re[ik][row] += alpha * p_re[ik][row];
                u_im[ik][row] += alpha * p_im[ik][row];
                r_re[ik][row] -= alpha * ap_re[ik][row];
                r_im[ik][row] -= alpha * ap_im[ik][row];
            }
        }
        apply_precond(&r_re, &r_im, &mut z_re, &mut z_im);
        let rz_new = kcpxtb_dot(kdata, &r_re, &r_im, &z_re, &z_im);
        if !(rz.is_finite() && rz.abs() > 1.0e-300) {
            break;
        }
        let beta = rz_new / rz;
        for ik in 0..nk {
            for row in 0..kdata[ik].pairs.len() {
                p_re[ik][row] = z_re[ik][row] + beta * p_re[ik][row];
                p_im[ik][row] = z_im[ik][row] + beta * p_im[ik][row];
            }
        }
        rz = rz_new;
    }
    Ok(())
}

pub fn kpoint_cpxtb_density_responses(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    couple: bool,
) -> Result<(Vec<Vec<CMatrix>>, Vec<Vec<CMatrix>>, Vec<Vec<f64>>)> {
    let nk = scf.kpoints.len();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let n = scf.basis.len();
    let nsh = scf.basis.shells.len();

    let kdata: Vec<KCpxtbData> = (0..nk)
        .map(|ik| build_kcpxtb_data(scf, ik))
        .collect::<Result<_>>()?;
    let skeletons: Vec<KpointSkeleton> = (0..nk)
        .map(|ik| {
            kpoint_skeleton_derivatives(
                system,
                params,
                scf,
                options,
                pbc,
                scf.kpoints[ik].fractional,
            )
        })
        .collect::<Result<_>>()?;
    let kernel = periodic_response_kernel(scf);

    // Finite-temperature (Fermi-smearing) path: when any band is genuinely
    // fractionally occupied the integer occ-virt CPXTB cannot carry the occupation
    // response df/dR, which couples ALL k-points through the single global Fermi
    // level. The finite-T branch (below) replaces the per-k occ-virt solve with the
    // full-band complex response and a Brillouin-zone-wide chemical-potential
    // constraint. Gapped systems keep the integer path.
    let kt = scf.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let finite_temperature = kt > 0.0
        && kdata.iter().any(|kd| {
            kd.mos
                .occupations
                .iter()
                .any(|&f| f > FRACTIONAL_OCC_EPS && f < 2.0 - FRACTIONAL_OCC_EPS)
        });

    let mut out: Vec<Vec<CMatrix>> = Vec::with_capacity(ndof);
    let mut out_w: Vec<Vec<CMatrix>> = Vec::with_capacity(ndof);
    let mut out_dq: Vec<Vec<f64>> = Vec::with_capacity(ndof);
    for y in 0..ndof {
        if finite_temperature {
            let (dp_k, dw_k, dq_total) = kpoint_finite_temperature_response_dof(
                scf, &kdata, &skeletons, &kernel, y, kt, couple,
            )?;
            out.push(dp_k);
            out_w.push(dw_k);
            out_dq.push(dq_total);
            continue;
        }
        // RHS b^k_ia and metric density per k.
        let mut rhs_re: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut rhs_im: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut metric: Vec<CMatrix> = Vec::with_capacity(nk);
        let mut fc_k: Vec<CMatrix> = Vec::with_capacity(nk);
        let mut sc1_k: Vec<CMatrix> = Vec::with_capacity(nk);
        for ik in 0..nk {
            let kd = &kdata[ik];
            let fc = cmatmul(&skeletons[ik].fock[y], &kd.mos.coeff);
            let sc1 = cmatmul(&skeletons[ik].overlap[y], &kd.mos.coeff);
            let mut br = vec![0.0; kd.pairs.len()];
            let mut bi = vec![0.0; kd.pairs.len()];
            for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                let (f1r, f1i) = cmo_element(&kd.mos.coeff, &fc, i, a);
                let (s1r, s1i) = cmo_element(&kd.mos.coeff, &sc1, i, a);
                let eps_i = kd.mos.energies[i];
                br[row] = -(f1r - eps_i * s1r);
                bi[row] = -(f1i - eps_i * s1i);
            }
            metric.push(complex_metric_density(&kd.mos, &sc1));
            rhs_re.push(br);
            rhs_im.push(bi);
            fc_k.push(fc);
            sc1_k.push(sc1);
        }

        if couple {
            // Constant metric + overlap-derivative charge -> potential -> RHS.
            let mut dq = vec![0.0; nsh];
            for ik in 0..nk {
                let kd = &kdata[ik];
                accumulate_complex_shell_charges(
                    &mut dq,
                    scf,
                    ik,
                    &kd.mos,
                    &metric[ik],
                    &skeletons[ik].overlap[y],
                    kd.weight,
                );
            }
            let pot = crate::linalg::matrix_vector_product(&kernel, &dq)?;
            for ik in 0..nk {
                let kd = &kdata[ik];
                for row in 0..kd.pairs.len() {
                    let mut addr = 0.0;
                    let mut addi = 0.0;
                    for s in 0..nsh {
                        addr += kd.q_re[row][s] * pot[s];
                        addi += kd.q_im[row][s] * pot[s];
                    }
                    rhs_re[ik][row] += 0.5 * addr;
                    rhs_im[ik][row] += 0.5 * addi;
                }
            }
        }

        // Amplitudes u^k_ia (complex). Initialise uncoupled u = b_eff/gap.
        let mut u_re: Vec<Vec<f64>> = (0..nk)
            .map(|ik| {
                (0..kdata[ik].pairs.len())
                    .map(|row| rhs_re[ik][row] / kdata[ik].gaps[row])
                    .collect()
            })
            .collect();
        let mut u_im: Vec<Vec<f64>> = (0..nk)
            .map(|ik| {
                (0..kdata[ik].pairs.len())
                    .map(|row| rhs_im[ik][row] / kdata[ik].gaps[row])
                    .collect()
            })
            .collect();

        if couple {
            solve_kpoint_cpxtb_pcg(&kdata, &kernel, nsh, &rhs_re, &rhs_im, &mut u_re, &mut u_im)?;
        }

        // Assemble dP(k) = occ-virt(u) + metric.
        let mut dp_k: Vec<CMatrix> = Vec::with_capacity(nk);
        for ik in 0..nk {
            let kd = &kdata[ik];
            let mut dp = metric[ik].clone();
            let c = &kd.mos.coeff;
            for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                let (ur, ui) = (u_re[ik][row], u_im[ik][row]);
                let focc = kd.mos.occupations[i] - kd.mos.occupations[a];
                if focc == 0.0 {
                    continue;
                }
                // The skeleton RHS convention gives u_ia = (U_ai)^*, so the density
                // response is dP = focc [ u^* C_a C_i^H + u C_i C_a^H ] (the Gamma
                // code's real u hides this conjugation; complex k needs it).
                for mu in 0..n {
                    let (car, cai) = (c.re[(mu, a)], c.im[(mu, a)]);
                    let (eir, eii) = (c.re[(mu, i)], c.im[(mu, i)]);
                    // conj(u) * C_mu,a
                    let p1r = ur * car + ui * cai;
                    let p1i = ur * cai - ui * car;
                    // u * C_mu,i
                    let p2r = ur * eir - ui * eii;
                    let p2i = ur * eii + ui * eir;
                    for nu in 0..n {
                        let (dr, di) = (c.re[(nu, i)], c.im[(nu, i)]); // C_nu,i (conj below)
                        let (fr, fi) = (c.re[(nu, a)], c.im[(nu, a)]); // C_nu,a (conj below)
                                                                       // term1 = (conj(u) C_mu,a) conj(C_nu,i)
                        let t1r = p1r * dr + p1i * di;
                        let t1i = p1i * dr - p1r * di;
                        // term2 = (u C_mu,i) conj(C_nu,a)
                        let t2r = p2r * fr + p2i * fi;
                        let t2i = p2i * fr - p2r * fi;
                        dp.re[(mu, nu)] += focc * (t1r + t2r);
                        dp.im[(mu, nu)] += focc * (t1i + t2i);
                    }
                }
            }
            dp_k.push(dp);
        }

        // Total real shell-charge response dq(y) = sum_k w_k Mulliken[dP(k),P0(k)].
        // Needed both for the response columns and (via the kernel) the SCC
        // response Fock that drives the energy-weighted occ-occ response.
        let mut dq_total = vec![0.0; nsh];
        for ik in 0..nk {
            accumulate_complex_shell_charges(
                &mut dq_total,
                scf,
                ik,
                &kdata[ik].mos,
                &dp_k[ik],
                &skeletons[ik].overlap[y],
                kdata[ik].weight,
            );
        }
        let vresp = if couple {
            crate::linalg::matrix_vector_product(&kernel, &dq_total)?
        } else {
            vec![0.0; nsh]
        };

        // Assemble the energy-weighted density response dW(k).
        let mut dw_k: Vec<CMatrix> = Vec::with_capacity(nk);
        for ik in 0..nk {
            dw_k.push(complex_weighted_density_response(
                &kdata[ik], &fc_k[ik], &sc1_k[ik], &u_re[ik], &u_im[ik], &vresp, &scf.basis,
            ));
        }

        out.push(dp_k);
        out_w.push(dw_k);
        out_dq.push(dq_total);
    }
    Ok((out, out_w, out_dq))
}

/// Complex energy-weighted density response `dW(k)/dR` for one k-point, mirroring
/// the Gamma [`weighted_density_response`]: an occupied-virtual part weighted by
/// `eps_i`, plus the occupied-occupied block
/// `0.5(focc_i+focc_j)(<i|F^1|j> + <i|RF|j> - (eps_i+eps_j)<i|S^1|j>)`, where the
/// response Fock `RF = -1/2 (v_mu+v_nu) S(k)` carries the SCC charge response
/// (`vresp`). Conjugations follow the `dP` convention (`u_ia = (U_ai)^*`).
#[allow(clippy::too_many_arguments)]
fn complex_weighted_density_response(
    kd: &KCpxtbData,
    fc: &CMatrix,
    sc1: &CMatrix,
    u_re: &[f64],
    u_im: &[f64],
    vresp: &[f64],
    basis: &BasisSet,
) -> CMatrix {
    let n = kd.mos.coeff.n;
    let c = &kd.mos.coeff;
    let occ = &kd.mos.occupations;
    let energies = &kd.mos.energies;
    let mut out = CMatrix::zeros(n);

    // occ-virt part, weighted by eps_i (mirror of the dP occ-virt assembly).
    for (row, &(i, a)) in kd.pairs.iter().enumerate() {
        let focc = occ[i] - occ[a];
        if focc == 0.0 {
            continue;
        }
        let w = focc * energies[i];
        let (ur, ui) = (u_re[row], u_im[row]);
        for mu in 0..n {
            let (car, cai) = (c.re[(mu, a)], c.im[(mu, a)]);
            let (eir, eii) = (c.re[(mu, i)], c.im[(mu, i)]);
            let p1r = ur * car + ui * cai; // conj(u) C_a
            let p1i = ur * cai - ui * car;
            let p2r = ur * eir - ui * eii; // u C_i
            let p2i = ur * eii + ui * eir;
            for nu in 0..n {
                let (dr, di) = (c.re[(nu, i)], c.im[(nu, i)]);
                let (fr, fi) = (c.re[(nu, a)], c.im[(nu, a)]);
                let t1r = p1r * dr + p1i * di;
                let t1i = p1i * dr - p1r * di;
                let t2r = p2r * fr + p2i * fi;
                let t2i = p2i * fr - p2r * fi;
                out.re[(mu, nu)] += w * (t1r + t2r);
                out.im[(mu, nu)] += w * (t1i + t2i);
            }
        }
    }

    // Response Fock RF = -1/2 (v_mu + v_nu) S(k), and RF*C for the MO elements.
    let mut rf = CMatrix::zeros(n);
    for mu in 0..n {
        let vmu = vresp[basis.aos[mu].shell_index];
        for nu in 0..n {
            let vnu = vresp[basis.aos[nu].shell_index];
            let scale = -0.5 * (vmu + vnu);
            rf.re[(mu, nu)] = scale * kd.mos.overlap.re[(mu, nu)];
            rf.im[(mu, nu)] = scale * kd.mos.overlap.im[(mu, nu)];
        }
    }
    let rfc = cmatmul(&rf, c);

    // occ-occ block: 0.5(focc_i+focc_j)(<i|F^1|j> + <i|RF|j> - (eps_i+eps_j)<i|S^1|j>).
    for i in 0..occ.len() {
        if occ[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occ.len() {
            if occ[j] <= 1.0e-8 {
                continue;
            }
            let (f1r, f1i) = cmo_element(c, fc, i, j);
            let (rfr, rfi) = cmo_element(c, &rfc, i, j);
            let (s1r, s1i) = cmo_element(c, sc1, i, j);
            let scale = 0.5 * (occ[i] + occ[j]);
            let esum = energies[i] + energies[j];
            let wr = scale * (f1r + rfr - esum * s1r);
            let wi = scale * (f1i + rfi - esum * s1i);
            if wr.abs() <= 1.0e-30 && wi.abs() <= 1.0e-30 {
                continue;
            }
            for mu in 0..n {
                let (air, aii) = (c.re[(mu, i)], c.im[(mu, i)]);
                for nu in 0..n {
                    let (bjr, bji) = (c.re[(nu, j)], c.im[(nu, j)]);
                    let gr = air * bjr + aii * bji; // C_mu,i conj(C_nu,j)
                    let gi = aii * bjr - air * bji;
                    out.re[(mu, nu)] += wr * gr - wi * gi;
                    out.im[(mu, nu)] += wr * gi + wi * gr;
                }
            }
        }
    }
    out
}

/// Conjugate transpose `A^H` of a complex matrix.
fn cconj_transpose(a: &CMatrix) -> CMatrix {
    let n = a.n;
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = a.re[(j, i)];
            out.im[(i, j)] = -a.im[(j, i)];
        }
    }
    out
}

/// Full MO-basis transform `M_ij = sum_mu conj(C_mu,i) (opc)_mu,j` for `opc = Op*C`
/// (the all-band-pair generalization of [`cmo_element`]), i.e. `C^H (Op C)`.
fn cmo_full(coeff: &CMatrix, opc: &CMatrix) -> CMatrix {
    cmatmul(&cconj_transpose(coeff), opc)
}

/// AO density-like matrix from an MO-basis coefficient matrix: `C * M * C^H`.
fn density_from_mo_coeff(coeff: &CMatrix, mo_matrix: &CMatrix) -> CMatrix {
    let cm = cmatmul(coeff, mo_matrix);
    cmatmul(&cm, &cconj_transpose(coeff))
}

/// SCC response Fock at a k-point: `RF(k)_munu = -1/2 (v_mu + v_nu) S(k)_munu`.
fn build_response_fock_k(overlap: &CMatrix, vresp: &[f64], basis: &BasisSet) -> CMatrix {
    let n = overlap.n;
    let mut rf = CMatrix::zeros(n);
    for mu in 0..n {
        let vmu = vresp[basis.aos[mu].shell_index];
        for nu in 0..n {
            let vnu = vresp[basis.aos[nu].shell_index];
            let scale = -0.5 * (vmu + vnu);
            rf.re[(mu, nu)] = scale * overlap.re[(mu, nu)];
            rf.im[(mu, nu)] = scale * overlap.im[(mu, nu)];
        }
    }
    rf
}

/// In-place complex matrix addition `dst += src`.
fn cadd_in_place(dst: &mut CMatrix, src: &CMatrix) {
    for (d, s) in dst.re.as_mut_slice().iter_mut().zip(src.re.as_slice()) {
        *d += *s;
    }
    for (d, s) in dst.im.as_mut_slice().iter_mut().zip(src.im.as_slice()) {
        *d += *s;
    }
}

/// Complex finite-temperature response coefficient matrix in the MO basis: the
/// k-point analogue of the molecular `finite_temperature_response_coefficients_from_mo`.
/// `h_mo = C^H (F^1 + RF) C` and `s_mo = C^H S^1 C` are Hermitian, so the returned
/// coefficient matrix is Hermitian and `C * coeff * C^H` is a Hermitian density.
/// `occupation_response[i] = df_i/dR` carries the (global-mu) Fermi-occupation
/// response; the off-diagonal small-gap limit uses the same kt-slope as the Gamma
/// path.
fn complex_ft_response_coefficients(
    occupations: &[f64],
    energies: &[f64],
    occupation_response: &[f64],
    h_mo: &CMatrix,
    s_mo: &CMatrix,
    kt: f64,
    energy_weighted: bool,
) -> CMatrix {
    let norb = occupations.len();
    let mut coeff = CMatrix::zeros(norb);
    for i in 0..norb {
        let f_i = occupations[i];
        let e_i = energies[i];
        let df_i = occupation_response[i];
        // Diagonal: the MO-basis diagonals of the Hermitian h_mo/s_mo are real.
        coeff.re[(i, i)] = if energy_weighted {
            let h_ii = h_mo.re[(i, i)] - e_i * s_mo.re[(i, i)];
            f_i * h_ii + e_i * df_i - f_i * e_i * s_mo.re[(i, i)]
        } else {
            df_i - f_i * s_mo.re[(i, i)]
        };
        for j in i + 1..norb {
            let f_j = occupations[j];
            let e_j = energies[j];
            let (hr, hi) = (h_mo.re[(i, j)], h_mo.im[(i, j)]);
            let (sr, si) = (s_mo.re[(i, j)], s_mo.im[(i, j)]);
            let gap = e_i - e_j;
            // Density coefficient `a*H - b*S` with (a, b) the occupation/energy
            // divided differences (kt-slope limit for a vanishing gap).
            let (a, b) = if gap.abs() > 1.0e-10 {
                if energy_weighted {
                    let w_i = f_i * e_i;
                    let w_j = f_j * e_j;
                    ((w_i - w_j) / gap, (w_i * e_i - w_j * e_j) / gap)
                } else {
                    ((f_i - f_j) / gap, (f_i * e_i - f_j * e_j) / gap)
                }
            } else {
                let eps = 0.5 * (e_i + e_j);
                let f = 0.5 * (f_i + f_j);
                let slope_f = -0.5 * (f_i * (1.0 - 0.5 * f_i) + f_j * (1.0 - 0.5 * f_j)) / kt;
                if energy_weighted {
                    (f + eps * slope_f, 2.0 * eps * f + eps * eps * slope_f)
                } else {
                    (slope_f, f + eps * slope_f)
                }
            };
            let vr = a * hr - b * sr;
            let vi = a * hi - b * si;
            coeff.re[(i, j)] = vr;
            coeff.im[(i, j)] = vi;
            coeff.re[(j, i)] = vr;
            coeff.im[(j, i)] = -vi; // Hermitian partner
        }
    }
    coeff
}

/// Finite-temperature complex k-point CPXTB response for one Cartesian DOF, with a
/// single Brillouin-zone-wide chemical-potential constraint. Returns the per-k
/// density response `dP(k)`, energy-weighted density response `dW(k)`, and the
/// shared real shell-charge response `dq`. The SCC charge response is iterated to
/// self-consistency (mirrors the Gamma finite-T helper); the occupation response
/// `df_ik/dR` is fixed by the global Fermi-level constraint
/// `sum_k w_k sum_i df_ik = 0`.
#[allow(clippy::type_complexity)]
fn kpoint_finite_temperature_response_dof(
    scf: &PbcSccResult,
    kdata: &[KCpxtbData],
    skeletons: &[KpointSkeleton],
    kernel: &Matrix,
    y: usize,
    kt: f64,
    couple: bool,
) -> Result<(Vec<CMatrix>, Vec<CMatrix>, Vec<f64>)> {
    let basis = &scf.basis;
    let n = basis.len();
    let nsh = basis.shells.len();
    let nk = kdata.len();

    // Per-k skeleton MO products and the (vresp-independent) base MO-basis blocks.
    let fc_k: Vec<CMatrix> = (0..nk)
        .map(|ik| cmatmul(&skeletons[ik].fock[y], &kdata[ik].mos.coeff))
        .collect();
    let sc1_k: Vec<CMatrix> = (0..nk)
        .map(|ik| cmatmul(&skeletons[ik].overlap[y], &kdata[ik].mos.coeff))
        .collect();
    let smo_all: Vec<CMatrix> = (0..nk)
        .map(|ik| cmo_full(&kdata[ik].mos.coeff, &sc1_k[ik]))
        .collect();
    let hmo_base: Vec<CMatrix> = (0..nk)
        .map(|ik| cmo_full(&kdata[ik].mos.coeff, &fc_k[ik]))
        .collect();

    // One response pass for a given SCC response potential `vresp`: total
    // H^1_mo(k), the global mu, the occupation response df(k), and dP(k).
    let pass = |vresp: &[f64]| -> (Vec<CMatrix>, Vec<Vec<f64>>, Vec<CMatrix>) {
        let mut hmo_all = Vec::with_capacity(nk);
        let mut deps_all = Vec::with_capacity(nk);
        for ik in 0..nk {
            let mut hmo = hmo_base[ik].clone();
            if couple {
                let rf = build_response_fock_k(&kdata[ik].mos.overlap, vresp, basis);
                let rfc = cmatmul(&rf, &kdata[ik].mos.coeff);
                let hmo_rf = cmo_full(&kdata[ik].mos.coeff, &rfc);
                cadd_in_place(&mut hmo, &hmo_rf);
            }
            let eps = &kdata[ik].mos.energies;
            let deps: Vec<f64> = (0..n)
                .map(|i| hmo.re[(i, i)] - eps[i] * smo_all[ik].re[(i, i)])
                .collect();
            deps_all.push(deps);
            hmo_all.push(hmo);
        }
        // Global chemical-potential response enforcing sum_k w_k sum_i df_ik = 0.
        let mut num = 0.0;
        let mut den = 0.0;
        for ik in 0..nk {
            let wk = kdata[ik].weight;
            let occ = &kdata[ik].mos.occupations;
            for i in 0..n {
                let w_ik = (occ[i] * (1.0 - 0.5 * occ[i])).max(0.0) / kt;
                num += wk * w_ik * deps_all[ik][i];
                den += wk * w_ik;
            }
        }
        let dmu = if den > 1.0e-30 { num / den } else { 0.0 };
        let mut dp_k = Vec::with_capacity(nk);
        let mut df_all = Vec::with_capacity(nk);
        for ik in 0..nk {
            let occ = &kdata[ik].mos.occupations;
            let energies = &kdata[ik].mos.energies;
            let df: Vec<f64> = (0..n)
                .map(|i| {
                    let w_ik = (occ[i] * (1.0 - 0.5 * occ[i])).max(0.0) / kt;
                    -w_ik * (deps_all[ik][i] - dmu)
                })
                .collect();
            let coeff = complex_ft_response_coefficients(
                occ,
                energies,
                &df,
                &hmo_all[ik],
                &smo_all[ik],
                kt,
                false,
            );
            dp_k.push(density_from_mo_coeff(&kdata[ik].mos.coeff, &coeff));
            df_all.push(df);
        }
        (dp_k, df_all, hmo_all)
    };

    // Self-consistent SCC response loop (mirrors the Gamma finite-T helper).
    let mut vresp = vec![0.0; nsh];
    let mut shell_response = vec![0.0; nsh];
    let response_mixing = 0.35;
    for _ in 0..50 {
        let (dp_k, _df, _hmo) = pass(&vresp);
        let mut next_shell = vec![0.0; nsh];
        for ik in 0..nk {
            accumulate_complex_shell_charges(
                &mut next_shell,
                scf,
                ik,
                &kdata[ik].mos,
                &dp_k[ik],
                &skeletons[ik].overlap[y],
                kdata[ik].weight,
            );
        }
        if !couple {
            break;
        }
        let shell_delta = shell_response
            .iter()
            .zip(next_shell.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let mixed = if shell_response.iter().any(|v| v.abs() > 0.0) {
            shell_response
                .iter()
                .zip(next_shell.iter())
                .map(|(&old, &new)| old + response_mixing * (new - old))
                .collect::<Vec<_>>()
        } else {
            next_shell.clone()
        };
        vresp = crate::linalg::matrix_vector_product(kernel, &mixed)?;
        shell_response = mixed;
        if shell_delta < 1.0e-12 {
            break;
        }
    }

    // Final pass with the converged response potential -> dP(k), dq, then dW(k).
    let (dp_k, df_all, hmo_all) = pass(&vresp);
    let mut dq_total = vec![0.0; nsh];
    for ik in 0..nk {
        accumulate_complex_shell_charges(
            &mut dq_total,
            scf,
            ik,
            &kdata[ik].mos,
            &dp_k[ik],
            &skeletons[ik].overlap[y],
            kdata[ik].weight,
        );
    }
    let mut dw_k = Vec::with_capacity(nk);
    for ik in 0..nk {
        let occ = &kdata[ik].mos.occupations;
        let energies = &kdata[ik].mos.energies;
        let coeff_w = complex_ft_response_coefficients(
            occ,
            energies,
            &df_all[ik],
            &hmo_all[ik],
            &smo_all[ik],
            kt,
            true,
        );
        dw_k.push(density_from_mo_coeff(&kdata[ik].mos.coeff, &coeff_w));
    }
    Ok((dp_k, dw_k, dq_total))
}

/// Periodic SCC response kernel `K_ij = dV_i/dq_j = Gamma_ij + on-site third
/// order`, used as the CPXTB coupling matrix.
pub fn periodic_response_kernel(scf: &PbcSccResult) -> Matrix {
    let model = &scf.shell_model;
    let mut kernel = scf.gamma.clone();
    for (atom, &qat) in scf.atomic_charges.iter().enumerate() {
        let count = model.atom_shell_counts[atom];
        if count == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let add = 2.0 * qat * model.hubbard_derivs[offset];
        for li in 0..count {
            for lj in 0..count {
                kernel[(offset + li, offset + lj)] += add;
            }
        }
    }
    kernel
}

/// Occupied-virtual pair list for the CPXTB (gapped systems: occ in {0,2}).
fn occ_virt_pairs(occupations: &[f64]) -> Vec<(usize, usize)> {
    let occ: Vec<usize> = (0..occupations.len())
        .filter(|&i| occupations[i] > 1.0e-8)
        .collect();
    let virt: Vec<usize> = (0..occupations.len())
        .filter(|&a| occupations[a] <= 1.0e-8)
        .collect();
    let mut pairs = Vec::with_capacity(occ.len() * virt.len());
    for &i in &occ {
        for &a in &virt {
            pairs.push((i, a));
        }
    }
    pairs
}

/// Shell-resolved Mulliken transition charges `q_ia` for each occ-virt pair.
fn transition_charges(mos: &GammaMos, pairs: &[(usize, usize)], basis: &BasisSet) -> Vec<Vec<f64>> {
    let n = basis.len();
    let sc = mos.overlap.matmul(&mos.coeff).expect("S C product");
    let mut out = Vec::with_capacity(pairs.len());
    for &(i, a) in pairs {
        let mut q = vec![0.0; basis.shells.len()];
        for (sidx, shell) in basis.shells.iter().enumerate() {
            for mu in shell.first_ao..shell.first_ao + shell.nao {
                q[sidx] -= mos.coeff[(mu, a)] * sc[(mu, i)] + mos.coeff[(mu, i)] * sc[(mu, a)];
            }
        }
        out.push(q);
        let _ = n;
    }
    out
}

#[inline]
fn mo_element(mos: &Matrix, op: &Matrix, left: usize, right: usize) -> f64 {
    let n = mos.rows();
    let mut acc = 0.0;
    for mu in 0..n {
        let cl = mos[(mu, left)];
        if cl == 0.0 {
            continue;
        }
        for nu in 0..n {
            acc += cl * op[(mu, nu)] * mos[(nu, right)];
        }
    }
    acc
}

/// Solve the Gamma-point CPXTB for every Cartesian DOF and return the AO density
/// responses `dP/dR_y` and energy-weighted density responses `dW/dR_y`.
pub fn gamma_cpxtb_density_responses(
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    mos: &GammaMos,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    let basis = &scf.basis;
    let n = basis.len();
    let ndof = skeleton.overlap.len();
    let pairs = occ_virt_pairs(&mos.occupations);
    let gaps: Vec<f64> = pairs
        .iter()
        .map(|&(i, a)| mos.energies[a] - mos.energies[i])
        .collect();
    let transition = transition_charges(mos, &pairs, basis);
    let kernel = periodic_response_kernel(scf);

    // Finite-temperature (Fermi-smearing) path: when any band is genuinely
    // fractionally occupied the integer occ-virt CPXTB is replaced by the
    // occupation-response formulation (mirroring the molecular `crate::cphf`
    // finite-T branch). Gapped systems keep the integer path below.
    let kt = scf.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let finite_temperature = kt > 0.0
        && mos
            .occupations
            .iter()
            .any(|&f| f > FRACTIONAL_OCC_EPS && f < 2.0 - FRACTIONAL_OCC_EPS);
    // Reference-cell real-space ground density P(Gamma) = sum_k w_k Re P(k) (for a
    // Gamma-only mesh this is just P(Gamma)); the finite-T shell-charge response
    // contracts it with dS.
    let ground_density = if finite_temperature {
        let mut g = Matrix::zeros(n, n);
        for (ik, kp) in scf.kpoints.iter().enumerate() {
            let w = kp.weight;
            for i in 0..n {
                for j in 0..n {
                    g[(i, j)] += w * scf.density_k[ik].re[(i, j)];
                }
            }
        }
        g
    } else {
        Matrix::zeros(0, 0)
    };

    // CPXTB operator A u = gap*u + q^T K q u (SCC coupling).
    let matvec = |u: &[f64]| -> Vec<f64> {
        let nsh = basis.shells.len();
        let mut induced = vec![0.0; nsh];
        for (qia, &ui) in transition.iter().zip(u) {
            for s in 0..nsh {
                induced[s] += qia[s] * ui;
            }
        }
        let pot = crate::linalg::matrix_vector_product(&kernel, &induced).expect("kernel mv");
        let mut out = vec![0.0; u.len()];
        for (row, qia) in transition.iter().enumerate() {
            let coupling: f64 = qia.iter().zip(&pot).map(|(&q, &v)| q * v).sum();
            out[row] = gaps[row] * u[row] + coupling;
        }
        out
    };

    let mut density_responses = Vec::with_capacity(ndof);
    let mut weighted_responses = Vec::with_capacity(ndof);
    for y in 0..ndof {
        if finite_temperature {
            let (density, weighted) = gamma_finite_temperature_response(
                scf,
                mos,
                &skeleton.fock[y],
                &skeleton.overlap[y],
                &kernel,
                &ground_density,
                kt,
            )?;
            density_responses.push(density);
            weighted_responses.push(weighted);
            continue;
        }
        // RHS b_ia = -(F^y_ia - eps_i S^y_ia) + metric-SCC term.
        let mut rhs = vec![0.0; pairs.len()];
        for (row, &(i, a)) in pairs.iter().enumerate() {
            let f1 = mo_element(&mos.coeff, &skeleton.fock[y], i, a);
            let s1 = mo_element(&mos.coeff, &skeleton.overlap[y], i, a);
            rhs[row] = -(f1 - mos.energies[i] * s1);
        }
        // Metric density response from S^y changes the shell charges -> potential
        // -> additional RHS.
        let metric_density = metric_density_response(mos, &skeleton.overlap[y]);
        let metric_shell = density_shell_charges(basis, mos, &metric_density, &skeleton.overlap[y]);
        let metric_pot = crate::linalg::matrix_vector_product(&kernel, &metric_shell)?;
        // Fock response to the known (skeleton) charge change is dF_ai = 1/2 q_ia
        // K dq; with no (f_j-f_b) factor the 1/2 stays explicit (unlike the
        // occ-virt coupling, where 1/2 * (f_j-f_b)=2 cancels). Moves to the RHS
        // with a minus sign.
        for (row, qia) in transition.iter().enumerate() {
            let add: f64 = qia.iter().zip(&metric_pot).map(|(&q, &v)| q * v).sum();
            rhs[row] -= 0.5 * add;
        }

        let u = solve_pcg(&matvec, &rhs, &gaps, 1.0e-9, 200);

        // AO density response: occ-virt part + metric part.
        let mut density = Matrix::zeros(n, n);
        for (row, &(i, a)) in pairs.iter().enumerate() {
            let weight = (mos.occupations[i] - mos.occupations[a]) * u[row];
            if weight == 0.0 {
                continue;
            }
            for mu in 0..n {
                for nu in 0..n {
                    density[(mu, nu)] += weight
                        * (mos.coeff[(mu, a)] * mos.coeff[(nu, i)]
                            + mos.coeff[(mu, i)] * mos.coeff[(nu, a)]);
                }
            }
        }
        add_in_place(&mut density, &metric_density);

        // Energy-weighted density response (for Pulay Hessian later).
        let weighted = weighted_density_response(
            mos,
            &skeleton.fock[y],
            &skeleton.overlap[y],
            &u,
            &pairs,
            &kernel,
            &transition,
            basis,
        );

        density_responses.push(density);
        weighted_responses.push(weighted);
    }
    Ok((density_responses, weighted_responses))
}

/// Analytic Gamma-point TDA excitation-energy gradient `d omega/dR` for a frozen
/// amplitude vector `amplitudes` (the periodic analog of the molecular
/// `tda_direct_excitation_gradient`). Returns the per-atom Cartesian gradient.
///
/// `d omega/dR = sum_ia X_ia^2 [(F~_aa - eps_a S~_aa) - (F~_ii - eps_i S~_ii)]`
/// (orbital-gap term, `F~` the full SCC-relaxed Fock derivative built from the
/// periodic CPHF charge response on top of the skeleton `dFock0/dR`), plus the
/// transition-transition coupling derivative `c d(P^T gamma P)/dR` split into the
/// kernel-derivative piece (`transition_kernel_gamma_gradient`) and the
/// transition-charge-derivative piece (the full first-order orbital-rotation matrix
/// `U`: occ-virt from the CPHF solution, occ-occ/virt-virt from Brillouin
/// stationarity, diagonal from the `-1/2 S` metric). Integer (gapped) occupations.
/// Per-k gauge-fixed data for the complex k-mesh TDA gradient.
struct KTdaData {
    mos: GaugeFixedKMos,
    /// `S(k) C(k)` (complex) for transition charges.
    sc: CMatrix,
    /// occ-virt pairs `(i, a)` in the order used for the per-k CPHF amplitudes.
    pairs: Vec<(usize, usize)>,
    gaps: Vec<f64>,
    /// Real Mulliken transition charge per pair per shell (unweighted by sqrt(w)).
    q: Vec<Vec<f64>>,
    weight: f64,
}

/// Analytic **general k-mesh** TDA excitation-energy gradient `d omega/dR` for the
/// frozen amplitude vector `amplitudes` with transition labels `labels[I] = (ik,
/// i*n + a)` (as returned by [`crate::td::solve_tda_kpoint`]). Returns the per-atom
/// Cartesian gradient.
///
/// Structure (generalising the Gamma path to complex Bloch k-points and the BZ sum):
///  - **Gap term** `sum_I X_I^2 d(eps_a(k)-eps_i(k))/dR`, gauge-invariant, with the
///    complex MO diagonal of the SCC-relaxed Fock derivative `F~(k)` and `dS(k)/dR`.
///  - **Kernel-derivative term** `c d(P^T gamma P)/dR` on the BZ-summed real
///    transition shell charges `P_s = sum_I X_I sqrt(w_k) q^I_s` (the same Ewald
///    helper as the Gamma path; gauge-invariant in `P`).
///  - **Transition-charge-derivative term** `2c (dP/dR).(gamma P)`, where each
///    `dq^I/dR` is differentiated through the complex per-k orbital-rotation matrix
///    `U(k)` (occ-virt from the per-k CPHF, occ-occ/virt-virt from Brillouin
///    stationarity, diagonal from the `-1/2 S` Hermitian metric, i.e. the natural
///    CPHF gauge `Im U_pp = 0`). The discrete max-AO phase-fixing derivative is
///    never taken; the gauge-fixed transition charges enter only through the
///    gauge-covariant `<i|.|a>` products, whose derivative the CPHF orbital response
///    supplies, and the result matches the finite-difference gradient.
///
/// The Brillouin-zone weights `w_k` enter exactly as in `solve_tda_kpoint`: the
/// transition charges carry `sqrt(w_k)`, so `P` and the per-pair products carry the
/// physical `w_k` normalization. Integer (gapped) band occupations.
#[allow(clippy::too_many_arguments)]
pub fn pbc_kpoint_tda_excitation_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    amplitudes: &[f64],
    labels: &[(usize, usize)],
    coupling: f64,
) -> Result<Vec<Vec3>> {
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let nshell = basis.shells.len();
    let nk = scf.kpoints.len();
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic k-mesh TDA gradient requires a lattice");

    if amplitudes.len() != labels.len() {
        return Err(crate::error::Gfn1Error::InvalidInput(
            "k-mesh TDA gradient amplitude/label length mismatch".to_string(),
        ));
    }

    let nocc = (scf.nelec / 2.0).round() as usize;
    if nocc == 0 || nocc >= n {
        return Err(crate::error::Gfn1Error::InvalidInput(
            "k-mesh TDA gradient requires an integer closed-shell band filling".to_string(),
        ));
    }
    let kernel = periodic_response_kernel(scf);

    // Per-k gauge-fixed MOs, pairs, transition charges (matching solve_tda_kpoint).
    let mut kdata: Vec<KTdaData> = Vec::with_capacity(nk);
    for ik in 0..nk {
        let mos = gauge_fixed_kmos(scf, ik)?;
        let sc = cmatmul(&mos.overlap, &mos.coeff);
        let mut pairs = Vec::new();
        let mut gaps = Vec::new();
        let mut q = Vec::new();
        for i in 0..nocc {
            for a in nocc..n {
                pairs.push((i, a));
                gaps.push(mos.energies[a] - mos.energies[i]);
                q.push(kmos_transition_charge(basis, &mos.coeff, &sc, i, a));
            }
        }
        kdata.push(KTdaData {
            mos,
            sc,
            pairs,
            gaps,
            q,
            weight: scf.kpoints[ik].weight,
        });
    }

    // Map the global transition index I -> (ik, pair-row) using the labels, and
    // accumulate the BZ-summed real transition shell charge P_s = sum_I X_I sqrt(w) q^I.
    // labels[I] = (ik, i*n + a).
    let mut amp_by_k: Vec<Vec<f64>> = kdata.iter().map(|kd| vec![0.0; kd.pairs.len()]).collect();
    let mut p_shell = vec![0.0_f64; nshell];
    for (idx, &(ik, ia)) in labels.iter().enumerate() {
        let i = ia / n;
        let a = ia % n;
        // Find the pair row in kdata[ik].
        let row = kdata[ik]
            .pairs
            .iter()
            .position(|&(pi, pa)| pi == i && pa == a)
            .ok_or_else(|| {
                crate::error::Gfn1Error::InvalidInput(
                    "k-mesh TDA gradient: transition label not found in occ-virt space".to_string(),
                )
            })?;
        let x = amplitudes[idx];
        amp_by_k[ik][row] = x;
        let sw = kdata[ik].weight.sqrt();
        for s in 0..nshell {
            p_shell[s] += x * sw * kdata[ik].q[row][s];
        }
    }
    let p_potential = crate::linalg::matrix_vector_product(&kernel, &p_shell)?;

    // ---- Solve the complex per-k CPHF for every DOF, keeping u(k), dP(k), dq(y). ----
    // RHS and amplitudes follow the existing kpoint CPXTB convention (skeleton RHS
    // b_ia = -(F^1_ia - eps_i S^1_ia), self-consistently coupled through the SCC
    // kernel; metric occ-occ density from S^1).
    let skeletons: Vec<KpointSkeleton> = (0..nk)
        .map(|ik| {
            kpoint_skeleton_derivatives(system, params, scf, options, pbc, scf.kpoints[ik].fractional)
        })
        .collect::<Result<_>>()?;

    // Complex transition charges Q_ia (gauge-fixed) for the CPHF coupling, following
    // the build_kcpxtb_data convention so the coupled solve reuses the same operator.
    let couple: Vec<KCoupleRef> = kdata
        .iter()
        .map(|kd| {
            let mut q_re = vec![vec![0.0; nshell]; kd.pairs.len()];
            let mut q_im = vec![vec![0.0; nshell]; kd.pairs.len()];
            for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                for mu in 0..n {
                    let s = basis.aos[mu].shell_index;
                    let (car, cai) = (kd.mos.coeff.re[(mu, a)], kd.mos.coeff.im[(mu, a)]);
                    let (cir, cii) = (kd.mos.coeff.re[(mu, i)], kd.mos.coeff.im[(mu, i)]);
                    let (sir, sii) = (kd.sc.re[(mu, i)], kd.sc.im[(mu, i)]);
                    let (sar, sai) = (kd.sc.re[(mu, a)], kd.sc.im[(mu, a)]);
                    let t1r = car * sir + cai * sii;
                    let t1i = cai * sir - car * sii;
                    let t2r = cir * sar + cii * sai;
                    let t2i = cir * sai - cii * sar;
                    q_re[row][s] += t1r + t2r;
                    q_im[row][s] += t1i + t2i;
                }
            }
            KCoupleRef { q_re, q_im }
        })
        .collect();

    // Solve all DOFs (independent) in parallel: returns per-DOF (u(k), dq_total).
    let columns: Vec<(Vec<Vec<(f64, f64)>>, Vec<f64>)> = (0..ndof)
        .into_par_iter()
        .map(|y| -> Result<(Vec<Vec<(f64, f64)>>, Vec<f64>)> {
            // Per-k skeleton-Fock/overlap MO products and RHS.
            let mut rhs_re: Vec<Vec<f64>> = Vec::with_capacity(nk);
            let mut rhs_im: Vec<Vec<f64>> = Vec::with_capacity(nk);
            let mut metric: Vec<CMatrix> = Vec::with_capacity(nk);
            for ik in 0..nk {
                let kd = &kdata[ik];
                let fc = cmatmul(&skeletons[ik].fock[y], &kd.mos.coeff);
                let sc1 = cmatmul(&skeletons[ik].overlap[y], &kd.mos.coeff);
                let mut br = vec![0.0; kd.pairs.len()];
                let mut bi = vec![0.0; kd.pairs.len()];
                for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                    let (f1r, f1i) = cmo_pair(&kd.mos.coeff, &fc, i, a);
                    let (s1r, s1i) = cmo_pair(&kd.mos.coeff, &sc1, i, a);
                    let eps_i = kd.mos.energies[i];
                    br[row] = -(f1r - eps_i * s1r);
                    bi[row] = -(f1i - eps_i * s1i);
                }
                metric.push(kmos_complex_metric_density(&kd.mos, &sc1, nocc));
                rhs_re.push(br);
                rhs_im.push(bi);
            }
            // Constant metric + dS charge -> potential -> RHS (couple).
            let mut dq = vec![0.0; nshell];
            for ik in 0..nk {
                kmos_accumulate_shell_charges(
                    &mut dq,
                    scf,
                    ik,
                    &kdata[ik].mos,
                    &metric[ik],
                    &skeletons[ik].overlap[y],
                    kdata[ik].weight,
                    nocc,
                );
            }
            let pot = crate::linalg::matrix_vector_product(&kernel, &dq)?;
            for ik in 0..nk {
                for row in 0..kdata[ik].pairs.len() {
                    let mut addr = 0.0;
                    let mut addi = 0.0;
                    for s in 0..nshell {
                        addr += couple[ik].q_re[row][s] * pot[s];
                        addi += couple[ik].q_im[row][s] * pot[s];
                    }
                    rhs_re[ik][row] += 0.5 * addr;
                    rhs_im[ik][row] += 0.5 * addi;
                }
            }

            // Initial u = b/gap, then coupled PCG.
            let mut u_re: Vec<Vec<f64>> = (0..nk)
                .map(|ik| {
                    (0..kdata[ik].pairs.len())
                        .map(|row| rhs_re[ik][row] / kdata[ik].gaps[row])
                        .collect()
                })
                .collect();
            let mut u_im: Vec<Vec<f64>> = (0..nk)
                .map(|ik| {
                    (0..kdata[ik].pairs.len())
                        .map(|row| rhs_im[ik][row] / kdata[ik].gaps[row])
                        .collect()
                })
                .collect();
            solve_ktda_cpxtb_pcg(
                &kdata, &couple, &kernel, nshell, &rhs_re, &rhs_im, &mut u_re, &mut u_im,
            )?;

            // dP(k) = metric + occ-virt(u); accumulate dq_total(y) = sum_k w_k
            // Mulliken[dP(k), P0(k)] on the fly and drop the dense dP(k) (only its
            // shell-charge response is needed downstream — avoids holding N_k full
            // n x n complex matrices per DOF).
            let mut u_pairs: Vec<Vec<(f64, f64)>> = Vec::with_capacity(nk);
            let mut dq_total = vec![0.0; nshell];
            for ik in 0..nk {
                let kd = &kdata[ik];
                let mut dp = metric[ik].clone();
                let c = &kd.mos.coeff;
                let mut up = Vec::with_capacity(kd.pairs.len());
                for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                    let (ur, ui) = (u_re[ik][row], u_im[ik][row]);
                    up.push((ur, ui));
                    // focc = occ_i - occ_a = 2 (closed shell).
                    let focc = 2.0;
                    for mu in 0..n {
                        let (car, cai) = (c.re[(mu, a)], c.im[(mu, a)]);
                        let (eir, eii) = (c.re[(mu, i)], c.im[(mu, i)]);
                        let p1r = ur * car + ui * cai; // conj(u) C_a
                        let p1i = ur * cai - ui * car;
                        let p2r = ur * eir - ui * eii; // u C_i
                        let p2i = ur * eii + ui * eir;
                        for nu in 0..n {
                            let (dr, di) = (c.re[(nu, i)], c.im[(nu, i)]);
                            let (fr, fi) = (c.re[(nu, a)], c.im[(nu, a)]);
                            let t1r = p1r * dr + p1i * di;
                            let t1i = p1i * dr - p1r * di;
                            let t2r = p2r * fr + p2i * fi;
                            let t2i = p2i * fr - p2r * fi;
                            dp.re[(mu, nu)] += focc * (t1r + t2r);
                            dp.im[(mu, nu)] += focc * (t1i + t2i);
                        }
                    }
                }
                kmos_accumulate_shell_charges(
                    &mut dq_total,
                    scf,
                    ik,
                    &kd.mos,
                    &dp,
                    &skeletons[ik].overlap[y],
                    kd.weight,
                    nocc,
                );
                u_pairs.push(up);
            }
            Ok((u_pairs, dq_total))
        })
        .collect::<Result<Vec<_>>>()?;

    // ---- (2) Kernel-derivative term c d(P^T gamma P)/dR (computed once). ----
    let mut out = vec![Vec3::zero(); nat];
    if coupling != 0.0 {
        let kern_grad =
            crate::pbc::gradient::transition_kernel_gamma_gradient(system, scf, pbc, &lattice, &p_shell);
        for atom in 0..nat {
            out[atom] += kern_grad[atom] * (2.0 * coupling);
        }
    }

    // ---- (1) gap term + (3) transition-charge term, per DOF (BZ-summed). ----
    let per_dof: Vec<Vec3> = (0..ndof)
        .into_par_iter()
        .map(|y| -> Result<Vec3> {
            let (u_pairs, dq_total) = &columns[y];
            // SCC response potential from the total charge response (drives F~(k)).
            let vresp = crate::linalg::matrix_vector_product(&kernel, dq_total)?;
            let mut value = 0.0;

            for ik in 0..nk {
                let kd = &kdata[ik];
                let s1 = &skeletons[ik].overlap[y];
                // F~(k) = skeleton fock(k) + responseFock(k); responseFock = -1/2 (v+v) S(k).
                let mut f_total = skeletons[ik].fock[y].clone();
                for mu in 0..n {
                    let vmu = vresp[basis.aos[mu].shell_index];
                    for nu in 0..n {
                        let vnu = vresp[basis.aos[nu].shell_index];
                        let scale = -0.5 * (vmu + vnu);
                        f_total.re[(mu, nu)] += scale * kd.mos.overlap.re[(mu, nu)];
                        f_total.im[(mu, nu)] += scale * kd.mos.overlap.im[(mu, nu)];
                    }
                }
                let fc = cmatmul(&f_total, &kd.mos.coeff);
                let sc1 = cmatmul(s1, &kd.mos.coeff);

                // (1) gap term: sum_ia X^2 [ (F~_aa - eps_a S^1_aa) - (F~_ii - eps_i S^1_ii) ].
                for (row, &(i, a)) in kd.pairs.iter().enumerate() {
                    let x = amp_by_k[ik][row];
                    if x == 0.0 {
                        continue;
                    }
                    let (faa, _) = cmo_pair(&kd.mos.coeff, &fc, a, a);
                    let (fii, _) = cmo_pair(&kd.mos.coeff, &fc, i, i);
                    let (saa, _) = cmo_pair(&kd.mos.coeff, &sc1, a, a);
                    let (sii, _) = cmo_pair(&kd.mos.coeff, &sc1, i, i);
                    value += x * x
                        * ((faa - kd.mos.energies[a] * saa) - (fii - kd.mos.energies[i] * sii));
                }

                // (3) transition-charge term contribution from this k-point.
                if coupling != 0.0 {
                    // Build complex orbital-rotation matrix U(k): occ-virt from CPHF u,
                    // occ-occ/virt-virt off-diag from Brillouin, diagonal -1/2 S^1 (real).
                    let dp_shell_k = ktda_transition_charge_derivative_shell(
                        kd,
                        &fc,
                        &sc1,
                        s1,
                        &u_pairs[ik],
                        &amp_by_k[ik],
                        nocc,
                        n,
                        basis,
                    );
                    let sw = kd.weight.sqrt();
                    let contrib: f64 = dp_shell_k
                        .iter()
                        .zip(p_potential.iter())
                        .map(|(&dq, &v)| dq * v)
                        .sum::<f64>();
                    value += 2.0 * coupling * sw * contrib;
                }
            }
            Ok(match y % 3 {
                0 => Vec3::new(value, 0.0, 0.0),
                1 => Vec3::new(0.0, value, 0.0),
                _ => Vec3::new(0.0, 0.0, value),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for (y, v) in per_dof.into_iter().enumerate() {
        out[y / 3] += v;
    }

    Ok(out)
}

/// Occupied-occupied complex metric density response for gauge-fixed MOs (occ bands
/// `0..nocc`, focc = 2): `dP_metric = sum_ij -1/2 (2+2) <i|S^1|j> C_i conj(C_j)^T`.
fn kmos_complex_metric_density(mos: &GaugeFixedKMos, sc1: &CMatrix, nocc: usize) -> CMatrix {
    let n = mos.coeff.n;
    let mut out = CMatrix::zeros(n);
    for i in 0..nocc {
        for j in 0..nocc {
            let (s1r, s1i) = cmo_element(&mos.coeff, sc1, i, j);
            let scale = -0.5 * (2.0 + 2.0);
            let (wr, wi) = (scale * s1r, scale * s1i);
            if wr.abs() <= 1.0e-30 && wi.abs() <= 1.0e-30 {
                continue;
            }
            for mu in 0..n {
                let (air, aii) = (mos.coeff.re[(mu, i)], mos.coeff.im[(mu, i)]);
                for nu in 0..n {
                    let (bjr, bji) = (mos.coeff.re[(nu, j)], mos.coeff.im[(nu, j)]);
                    let gr = air * bjr + aii * bji;
                    let gi = aii * bjr - air * bji;
                    out.re[(mu, nu)] += wr * gr - wi * gi;
                    out.im[(mu, nu)] += wr * gi + wi * gr;
                }
            }
        }
    }
    out
}

/// Accumulate the real shell-charge response at one k-point (gauge-fixed MOs).
#[allow(clippy::too_many_arguments)]
fn kmos_accumulate_shell_charges(
    dq: &mut [f64],
    scf: &PbcSccResult,
    ik: usize,
    mos: &GaugeFixedKMos,
    dp: &CMatrix,
    s1: &CMatrix,
    weight: f64,
    _nocc: usize,
) {
    let n = scf.basis.len();
    let dps = cmatmul(dp, &mos.overlap);
    let p0s1 = cmatmul(&scf.density_k[ik], s1);
    for mu in 0..n {
        let s = scf.basis.aos[mu].shell_index;
        dq[s] -= weight * (dps.re[(mu, mu)] + p0s1.re[(mu, mu)]);
    }
}

/// Coupled complex k-mesh TDA-CPXTB matvec `M u = diag(gap) u - 1/2 C u` for the
/// gauge-fixed transition charges `couple` (mirrors `kpoint_cpxtb_matvec`).
fn ktda_cpxtb_matvec(
    kdata: &[KTdaData],
    couple: &[(Vec<Vec<f64>>, Vec<Vec<f64>>)],
    kernel: &Matrix,
    nsh: usize,
    u_re: &[Vec<f64>],
    u_im: &[Vec<f64>],
    out_re: &mut [Vec<f64>],
    out_im: &mut [Vec<f64>],
) -> Result<()> {
    let mut dq = vec![0.0; nsh];
    for (ik, kd) in kdata.iter().enumerate() {
        let (q_re, q_im) = &couple[ik];
        for row in 0..kd.pairs.len() {
            let (ur, ui) = (u_re[ik][row], u_im[ik][row]);
            for s in 0..nsh {
                dq[s] -= kd.weight * 2.0 * (ur * q_re[row][s] + ui * q_im[row][s]);
            }
        }
    }
    let pot = crate::linalg::matrix_vector_product(kernel, &dq)?;
    for (ik, kd) in kdata.iter().enumerate() {
        let (q_re, q_im) = &couple[ik];
        for row in 0..kd.pairs.len() {
            let mut addr = 0.0;
            let mut addi = 0.0;
            for s in 0..nsh {
                addr += q_re[row][s] * pot[s];
                addi += q_im[row][s] * pot[s];
            }
            out_re[ik][row] = kd.gaps[row] * u_re[ik][row] - 0.5 * addr;
            out_im[ik][row] = kd.gaps[row] * u_im[ik][row] - 0.5 * addi;
        }
    }
    Ok(())
}

/// k-weighted inner product for the coupled complex k-mesh CPXTB PCG.
fn ktda_dot(
    kdata: &[KTdaData],
    ar: &[Vec<f64>],
    ai: &[Vec<f64>],
    br: &[Vec<f64>],
    bi: &[Vec<f64>],
) -> f64 {
    let mut s = 0.0;
    for (ik, kd) in kdata.iter().enumerate() {
        let w = kd.weight;
        for row in 0..kd.pairs.len() {
            s += w * (ar[ik][row] * br[ik][row] + ai[ik][row] * bi[ik][row]);
        }
    }
    s
}

/// Solve the coupled complex k-mesh TDA-CPXTB `M u = rhs` with preconditioned CG.
#[allow(clippy::too_many_arguments)]
fn solve_ktda_cpxtb_pcg(
    kdata: &[KTdaData],
    couple_in: &[KCoupleRef],
    kernel: &Matrix,
    nsh: usize,
    rhs_re: &[Vec<f64>],
    rhs_im: &[Vec<f64>],
    u_re: &mut [Vec<f64>],
    u_im: &mut [Vec<f64>],
) -> Result<()> {
    const PRECOND_FLOOR: f64 = 1.0e-4;
    let nk = kdata.len();
    let total: usize = kdata.iter().map(|kd| kd.pairs.len()).sum();
    if total == 0 {
        return Ok(());
    }
    let couple: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = couple_in
        .iter()
        .map(|c| (c.q_re.clone(), c.q_im.clone()))
        .collect();
    let zeros = || -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let v: Vec<Vec<f64>> = kdata.iter().map(|kd| vec![0.0; kd.pairs.len()]).collect();
        (v.clone(), v)
    };
    let (mut mu_re, mut mu_im) = zeros();
    ktda_cpxtb_matvec(kdata, &couple, kernel, nsh, u_re, u_im, &mut mu_re, &mut mu_im)?;
    let (mut r_re, mut r_im) = zeros();
    for ik in 0..nk {
        for row in 0..kdata[ik].pairs.len() {
            r_re[ik][row] = rhs_re[ik][row] - mu_re[ik][row];
            r_im[ik][row] = rhs_im[ik][row] - mu_im[ik][row];
        }
    }
    let (mut z_re, mut z_im) = zeros();
    let apply_precond =
        |r_re: &[Vec<f64>], r_im: &[Vec<f64>], z_re: &mut [Vec<f64>], z_im: &mut [Vec<f64>]| {
            for (ik, kd) in kdata.iter().enumerate() {
                for row in 0..kd.pairs.len() {
                    let inv = 1.0 / kd.gaps[row].max(PRECOND_FLOOR);
                    z_re[ik][row] = r_re[ik][row] * inv;
                    z_im[ik][row] = r_im[ik][row] * inv;
                }
            }
        };
    apply_precond(&r_re, &r_im, &mut z_re, &mut z_im);
    let (mut p_re, mut p_im) = (z_re.clone(), z_im.clone());
    let mut rz = ktda_dot(kdata, &r_re, &r_im, &z_re, &z_im);
    let rhs_norm = ktda_dot(kdata, rhs_re, rhs_im, rhs_re, rhs_im).sqrt().max(1.0);
    let tol = 1.0e-11 * rhs_norm;
    let max_iter = (4 * total).clamp(50, 4000);
    let (mut ap_re, mut ap_im) = zeros();
    for _ in 0..max_iter {
        let rnorm = ktda_dot(kdata, &r_re, &r_im, &r_re, &r_im).sqrt();
        if !rnorm.is_finite() || rnorm <= tol {
            break;
        }
        ktda_cpxtb_matvec(kdata, &couple, kernel, nsh, &p_re, &p_im, &mut ap_re, &mut ap_im)?;
        let pap = ktda_dot(kdata, &p_re, &p_im, &ap_re, &ap_im);
        if !(pap.is_finite() && pap.abs() > 1.0e-30) {
            break;
        }
        let alpha = rz / pap;
        for ik in 0..nk {
            for row in 0..kdata[ik].pairs.len() {
                u_re[ik][row] += alpha * p_re[ik][row];
                u_im[ik][row] += alpha * p_im[ik][row];
                r_re[ik][row] -= alpha * ap_re[ik][row];
                r_im[ik][row] -= alpha * ap_im[ik][row];
            }
        }
        apply_precond(&r_re, &r_im, &mut z_re, &mut z_im);
        let rz_new = ktda_dot(kdata, &r_re, &r_im, &z_re, &z_im);
        if !(rz.is_finite() && rz.abs() > 1.0e-300) {
            break;
        }
        let beta = rz_new / rz;
        for ik in 0..nk {
            for row in 0..kdata[ik].pairs.len() {
                p_re[ik][row] = z_re[ik][row] + beta * p_re[ik][row];
                p_im[ik][row] = z_im[ik][row] + beta * p_im[ik][row];
            }
        }
        rz = rz_new;
    }
    Ok(())
}

/// Reference to the gauge-fixed complex transition charges of one k-point.
struct KCoupleRef {
    q_re: Vec<Vec<f64>>,
    q_im: Vec<Vec<f64>>,
}

/// BZ-unweighted transition-charge derivative `sum_ia X_ia dq^{ia}_s/dR` at one
/// k-point, from the complex orbital-rotation matrix `U(k)` (occ-virt from CPHF,
/// occ-occ/virt-virt from Brillouin, diagonal `-1/2 S^1`). `fc = F~ C`, `sc1 = S^1 C`.
#[allow(clippy::too_many_arguments)]
fn ktda_transition_charge_derivative_shell(
    kd: &KTdaData,
    fc: &CMatrix,
    sc1: &CMatrix,
    s1: &CMatrix,
    u_pairs: &[(f64, f64)],
    amp: &[f64],
    nocc: usize,
    n: usize,
    basis: &BasisSet,
) -> Vec<f64> {
    let nmo = n;
    let c = &kd.mos.coeff;
    // Full complex U(k).
    let mut u = CMatrix::zeros(nmo);
    // S^1 and F~ in the MO basis.
    let mut smo = CMatrix::zeros(nmo);
    let mut fmo = CMatrix::zeros(nmo);
    for p in 0..nmo {
        for q in 0..nmo {
            let (sr, si) = cmo_element(c, sc1, p, q);
            let (fr, fi) = cmo_element(c, fc, p, q);
            smo.re[(p, q)] = sr;
            smo.im[(p, q)] = si;
            fmo.re[(p, q)] = fr;
            fmo.im[(p, q)] = fi;
        }
    }
    let is_occ = |p: usize| p < nocc;
    for p in 0..nmo {
        // Diagonal: -1/2 S^1_pp (Hermitian metric, natural gauge Im U_pp via S^1 is
        // imaginary part of <p|S^1|p>; keep the full complex -1/2 S^1_pp).
        u.re[(p, p)] = -0.5 * smo.re[(p, p)];
        u.im[(p, p)] = -0.5 * smo.im[(p, p)];
        for q in 0..nmo {
            if p == q {
                continue;
            }
            if is_occ(p) == is_occ(q) {
                let denom = kd.mos.energies[q] - kd.mos.energies[p];
                if denom.abs() > 1.0e-8 {
                    // U_pq = (F~_pq - eps_q S^1_pq)/(eps_q - eps_p).
                    u.re[(p, q)] = (fmo.re[(p, q)] - kd.mos.energies[q] * smo.re[(p, q)]) / denom;
                    u.im[(p, q)] = (fmo.im[(p, q)] - kd.mos.energies[q] * smo.im[(p, q)]) / denom;
                } else {
                    u.re[(p, q)] = -0.5 * smo.re[(p, q)];
                    u.im[(p, q)] = -0.5 * smo.im[(p, q)];
                }
            }
        }
    }
    // occ-virt blocks from the CPHF amplitudes. The skeleton RHS convention gives
    // u_ia = (U_ai)^*; the constraint U_ia + (U_ai)^* = -S^1_ia fixes U_ia.
    for (row, &(i, a)) in kd.pairs.iter().enumerate() {
        let (ur, ui) = u_pairs[row];
        // U_ai = conj(u_ia) = (ur, -ui).
        u.re[(a, i)] = ur;
        u.im[(a, i)] = -ui;
        // U_ia = -conj(U_ai) - S^1_ia = -(ur, ui)... wait: U_ia = -(U_ai)^* - S^1_ia.
        // (U_ai)^* = (ur, ui); so U_ia = -(ur + i ui) - S^1_ia.
        u.re[(i, a)] = -ur - smo.re[(i, a)];
        u.im[(i, a)] = -ui - smo.im[(i, a)];
    }

    // dC = C U (complex). dSC = (dS) C + S (dC).
    let c_deriv = cmatmul(c, &u);
    let s_cderiv = cmatmul(&kd.mos.overlap, &c_deriv);
    let dsc = {
        // dsc = s1*C + S*dC
        let s1c = cmatmul(s1, c);
        let mut m = CMatrix::zeros(nmo);
        for mu in 0..nmo {
            for nu in 0..nmo {
                m.re[(mu, nu)] = s1c.re[(mu, nu)] + s_cderiv.re[(mu, nu)];
                m.im[(mu, nu)] = s1c.im[(mu, nu)] + s_cderiv.im[(mu, nu)];
            }
        }
        m
    };

    // dq^{ia}_s = -2 Re sum_{mu in s} [ d(conj(C_i)) (SC_a) + conj(C_i) d(SC_a) ]
    //   with d(SC_a) over both centres = dsc_a, and the symmetric a<->i term.
    // Mirroring kmos_transition_charge differentiated:
    //   q_ia,s = -[ conj(C_i)(SC_a) + conj(C_a)(SC_i) ]_real summed.
    // dq = -[ conj(dC_i)(SC_a) + conj(C_i)(dSC_a) + conj(dC_a)(SC_i) + conj(C_a)(dSC_i) ]_real.
    let mut out = vec![0.0_f64; basis.shells.len()];
    let sc = &kd.sc;
    for (row, &(i, a)) in kd.pairs.iter().enumerate() {
        let x = amp[row];
        if x == 0.0 {
            continue;
        }
        for (sidx, shell) in basis.shells.iter().enumerate() {
            let mut acc = 0.0;
            for mu in shell.first_ao..shell.first_ao + shell.nao {
                // conj(dC_i,mu)(SC_a,mu): Re
                acc += c_deriv.re[(mu, i)] * sc.re[(mu, a)] + c_deriv.im[(mu, i)] * sc.im[(mu, a)];
                // conj(C_i,mu)(dSC_a,mu): Re
                acc += c.re[(mu, i)] * dsc.re[(mu, a)] + c.im[(mu, i)] * dsc.im[(mu, a)];
                // conj(dC_a,mu)(SC_i,mu): Re
                acc += c_deriv.re[(mu, a)] * sc.re[(mu, i)] + c_deriv.im[(mu, a)] * sc.im[(mu, i)];
                // conj(C_a,mu)(dSC_i,mu): Re
                acc += c.re[(mu, a)] * dsc.re[(mu, i)] + c.im[(mu, a)] * dsc.im[(mu, i)];
            }
            out[sidx] -= x * acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn pbc_gamma_tda_excitation_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    mos: &GammaMos,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    amplitudes: &[f64],
    coupling: f64,
) -> Result<Vec<Vec3>> {
    let basis = &scf.basis;
    let n = basis.len();
    let nat = system.atoms.len();
    let ndof = skeleton.overlap.len();
    let nshell = basis.shells.len();
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic TDA gradient requires a lattice");

    let pairs = occ_virt_pairs(&mos.occupations);
    if amplitudes.len() != pairs.len() {
        return Err(crate::error::Gfn1Error::InvalidInput(
            "PBC TDA gradient amplitude length mismatch".to_string(),
        ));
    }
    // Reject fractional (finite-T) occupations: the integer occ-virt CPXTB path is
    // required (matching the gapped TDA assembly).
    if mos
        .occupations
        .iter()
        .any(|&f| f > FRACTIONAL_OCC_EPS && f < 2.0 - FRACTIONAL_OCC_EPS)
    {
        return Err(crate::error::Gfn1Error::InvalidInput(
            "PBC TDA analytic gradient requires integer (gapped) occupations".to_string(),
        ));
    }

    let gaps: Vec<f64> = pairs
        .iter()
        .map(|&(i, a)| mos.energies[a] - mos.energies[i])
        .collect();
    let transition = transition_charges(mos, &pairs, basis);
    let kernel = periodic_response_kernel(scf);

    // State transition shell charges P_s = sum_ia X_ia q^ia_s and the response
    // potential they feel (gamma P).
    let mut p_shell = vec![0.0_f64; nshell];
    for (qia, &amp) in transition.iter().zip(amplitudes.iter()) {
        for (s, &q) in qia.iter().enumerate() {
            p_shell[s] += amp * q;
        }
    }
    let p_potential = crate::linalg::matrix_vector_product(&kernel, &p_shell)?;

    // CPXTB operator A u = gap*u + q^T K q u (same as gamma_cpxtb_density_responses).
    let matvec = |u: &[f64]| -> Vec<f64> {
        let mut induced = vec![0.0; nshell];
        for (qia, &ui) in transition.iter().zip(u) {
            for s in 0..nshell {
                induced[s] += qia[s] * ui;
            }
        }
        let pot = crate::linalg::matrix_vector_product(&kernel, &induced).expect("kernel mv");
        let mut out = vec![0.0; u.len()];
        for (row, qia) in transition.iter().enumerate() {
            let coupling: f64 = qia.iter().zip(&pot).map(|(&q, &v)| q * v).sum();
            out[row] = gaps[row] * u[row] + coupling;
        }
        out
    };

    // Solve the CPHF for every DOF, keeping both the AO density response and the
    // occ-virt amplitude vector u (needed for the transition-charge derivative).
    let mut density_responses: Vec<Matrix> = Vec::with_capacity(ndof);
    let mut u_responses: Vec<Vec<f64>> = Vec::with_capacity(ndof);
    for y in 0..ndof {
        let mut rhs = vec![0.0; pairs.len()];
        for (row, &(i, a)) in pairs.iter().enumerate() {
            let f1 = mo_element(&mos.coeff, &skeleton.fock[y], i, a);
            let s1 = mo_element(&mos.coeff, &skeleton.overlap[y], i, a);
            rhs[row] = -(f1 - mos.energies[i] * s1);
        }
        let metric_density = metric_density_response(mos, &skeleton.overlap[y]);
        let metric_shell = density_shell_charges(basis, mos, &metric_density, &skeleton.overlap[y]);
        let metric_pot = crate::linalg::matrix_vector_product(&kernel, &metric_shell)?;
        for (row, qia) in transition.iter().enumerate() {
            let add: f64 = qia.iter().zip(&metric_pot).map(|(&q, &v)| q * v).sum();
            rhs[row] -= 0.5 * add;
        }
        let u = solve_pcg(&matvec, &rhs, &gaps, 1.0e-10, 400);

        let mut density = Matrix::zeros(n, n);
        for (row, &(i, a)) in pairs.iter().enumerate() {
            let weight = (mos.occupations[i] - mos.occupations[a]) * u[row];
            if weight == 0.0 {
                continue;
            }
            for mu in 0..n {
                for nu in 0..n {
                    density[(mu, nu)] += weight
                        * (mos.coeff[(mu, a)] * mos.coeff[(nu, i)]
                            + mos.coeff[(mu, i)] * mos.coeff[(nu, a)]);
                }
            }
        }
        add_in_place(&mut density, &metric_density);
        density_responses.push(density);
        u_responses.push(u);
    }

    // Ground reference density P0 (Gamma): used to fold the dS metric piece of the
    // shell-charge response that drives the SCC response Fock.
    let mut ground_density = Matrix::zeros(n, n);
    for i in 0..mos.occupations.len() {
        let occ = mos.occupations[i];
        if occ <= 1.0e-14 {
            continue;
        }
        for mu in 0..n {
            for nu in 0..n {
                ground_density[(mu, nu)] += occ * mos.coeff[(mu, i)] * mos.coeff[(nu, i)];
            }
        }
    }

    // (2) Kernel-derivative piece c * d(P^T gamma P)/dR (DOF-coupled, computed once).
    // The Ewald helper returns d(1/2 P^T gamma P)/dR (the energy-gradient convention
    // shared with the ground-state electrostatic gradient), so the bilinear form
    // carries an explicit factor of 2 (matching the non-PBC `coupling_kernel_gradient`).
    let mut out = vec![Vec3::zero(); nat];
    if coupling != 0.0 {
        let kern_grad = crate::pbc::gradient::transition_kernel_gamma_gradient(
            system, scf, pbc, &lattice, &p_shell,
        );
        for atom in 0..nat {
            out[atom] += kern_grad[atom] * (2.0 * coupling);
        }
    }

    // Per-DOF terms (1) orbital-gap and (3) transition-charge derivative. The DOFs
    // are independent, so the columns are computed in parallel; each needs the full
    // SCC-relaxed Fock derivative F~ = dFock0/dR + responseFock(charge response),
    // built once per DOF and shared between the two terms. The C^T M C MO matrices
    // are formed by two matmuls (O(n^3)) rather than element-wise (O(n^4)).
    let nmo = mos.coeff.cols();
    let occ_set: std::collections::HashSet<usize> = (0..nmo)
        .filter(|&p| mos.occupations[p] > 1.0e-8)
        .collect();
    let sc = mos.overlap.matmul(&mos.coeff)?;
    let ct = mos.coeff.transpose();
    let per_dof: Vec<Vec3> = (0..ndof)
        .into_par_iter()
        .map(|y| -> Result<Vec3> {
            let overlap_deriv = &skeleton.overlap[y];
            let shell_response = crate::cphf::response_shell_charges_from_density(
                basis,
                &mos.overlap,
                &ground_density,
                &density_responses[y],
                overlap_deriv,
            )?;
            let shell_potential = crate::linalg::matrix_vector_product(&kernel, &shell_response)?;
            let response_fock =
                crate::cphf::scalar_response_fock_matrix(basis, &mos.overlap, &shell_potential)?;
            let mut f_total = skeleton.fock[y].clone();
            add_in_place(&mut f_total, &response_fock);

            // MO-basis derivative matrices: s_mo = C^T (dS) C, f_mo = C^T F~ C.
            let s_mo = ct.matmul(&overlap_deriv.matmul(&mos.coeff)?)?;
            let f_mo = ct.matmul(&f_total.matmul(&mos.coeff)?)?;

            // (1) Orbital-gap term (uses only the diagonal of s_mo / f_mo).
            let mut value = 0.0;
            for (pair_idx, &(i, a)) in pairs.iter().enumerate() {
                let weight = amplitudes[pair_idx] * amplitudes[pair_idx];
                if weight == 0.0 {
                    continue;
                }
                value += weight
                    * ((f_mo[(a, a)] - mos.energies[a] * s_mo[(a, a)])
                        - (f_mo[(i, i)] - mos.energies[i] * s_mo[(i, i)]));
            }

            // (3) Transition-charge derivative (needs the full orbital-rotation U).
            if coupling != 0.0 {
                let mut u_mo = Matrix::zeros(nmo, nmo);
                for p in 0..nmo {
                    u_mo[(p, p)] = -0.5 * s_mo[(p, p)];
                    for q in 0..nmo {
                        if p == q {
                            continue;
                        }
                        let same_block = occ_set.contains(&p) == occ_set.contains(&q);
                        if same_block {
                            let denom = mos.energies[q] - mos.energies[p];
                            u_mo[(p, q)] = if denom.abs() > 1.0e-8 {
                                (f_mo[(p, q)] - mos.energies[q] * s_mo[(p, q)]) / denom
                            } else {
                                -0.5 * s_mo[(p, q)]
                            };
                        }
                    }
                }
                for (pair_idx, &(i, a)) in pairs.iter().enumerate() {
                    let uval = u_responses[y][pair_idx];
                    u_mo[(a, i)] = uval;
                    u_mo[(i, a)] = -uval - s_mo[(i, a)];
                }
                let c_deriv = mos.coeff.matmul(&u_mo)?;
                let mut dsc = overlap_deriv.matmul(&mos.coeff)?;
                let s_cderiv = mos.overlap.matmul(&c_deriv)?;
                add_in_place(&mut dsc, &s_cderiv);

                let mut dp_shell = vec![0.0_f64; nshell];
                for (pair_idx, &(i, a)) in pairs.iter().enumerate() {
                    let amp = amplitudes[pair_idx];
                    if amp == 0.0 {
                        continue;
                    }
                    let dq = transition_shell_charge_derivative_mo_pair(
                        basis, &mos.coeff, &sc, &c_deriv, &dsc, i, a,
                    );
                    for (s, &q) in dq.iter().enumerate() {
                        dp_shell[s] += amp * q;
                    }
                }
                value += 2.0
                    * coupling
                    * dp_shell
                        .iter()
                        .zip(p_potential.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>();
            }

            Ok(match y % 3 {
                0 => Vec3::new(value, 0.0, 0.0),
                1 => Vec3::new(0.0, value, 0.0),
                _ => Vec3::new(0.0, 0.0, value),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for (y, v) in per_dof.into_iter().enumerate() {
        out[y / 3] += v;
    }

    let _ = (params, options);
    Ok(out)
}

/// Mulliken transition shell-charge derivative for an MO pair `(left, right)`,
/// from the orbital-coefficient response `c_deriv = C U` and `dsc = (dS)C + S(dC)`
/// (periodic Gamma analog of the molecular helper of the same role).
fn transition_shell_charge_derivative_mo_pair(
    basis: &BasisSet,
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

/// Gauge-fixed complex MOs at a k-point: the physical band coefficients
/// `C(k) = u + i v` with each band's global phase fixed so the largest-magnitude AO
/// coefficient is real and positive (the same convention as the energy-path
/// `td::gauge_fixed_band`). This makes the real Mulliken transition charges
/// `q_ia = -2 Re<i|S|a>` reproducible and phase-consistent with the amplitudes from
/// [`crate::td::solve_tda_kpoint`].
struct GaugeFixedKMos {
    /// `n x n` complex MO coefficients, column `b` = band `b`.
    coeff: CMatrix,
    energies: Vec<f64>,
    overlap: CMatrix,
}

fn gauge_fixed_kmos(scf: &PbcSccResult, ik: usize) -> Result<GaugeFixedKMos> {
    let km = kpoint_mos(scf, ik)?;
    let n = scf.basis.len();
    let mut coeff = CMatrix::zeros(n);
    let mut energies = vec![0.0; n];
    for b in 0..n {
        let col = 2 * b;
        // Raw representative of the degenerate embedded pair.
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for mu in 0..n {
            re[mu] = km.eig.vectors[(mu, col)];
            im[mu] = km.eig.vectors[(n + mu, col)];
        }
        // Phase-fix on the largest-magnitude AO coefficient.
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
            let cos_phi = re[mu_star] / norm;
            let sin_phi = im[mu_star] / norm;
            for mu in 0..n {
                let r = re[mu];
                let m = im[mu];
                coeff.re[(mu, b)] = r * cos_phi + m * sin_phi;
                coeff.im[(mu, b)] = m * cos_phi - r * sin_phi;
            }
        }
        energies[b] = km.eig.values[col];
    }
    Ok(GaugeFixedKMos {
        coeff,
        energies,
        overlap: km.overlap,
    })
}

/// `<p|Op|q> = sum_mu conj(C_mu,p) (Op C)_mu,q` for gauge-fixed complex MOs, given
/// the precomputed `opc = Op*C`. Returns `(re, im)`.
#[inline]
fn cmo_pair(coeff: &CMatrix, opc: &CMatrix, p: usize, q: usize) -> (f64, f64) {
    cmo_element(coeff, opc, p, q)
}

/// Real Mulliken transition shell charge `q_ia,s = -2 Re sum_{mu in s} conj(C_i,mu)
/// (S C_a)_mu` for gauge-fixed complex MOs (`sc = S C` precomputed). Matches the
/// `solve_tda_kpoint` sign/`Re` convention (before the sqrt(w_k) weighting).
fn kmos_transition_charge(
    basis: &BasisSet,
    coeff: &CMatrix,
    sc: &CMatrix,
    i: usize,
    a: usize,
) -> Vec<f64> {
    let mut q = vec![0.0; basis.shells.len()];
    for (sidx, shell) in basis.shells.iter().enumerate() {
        for mu in shell.first_ao..shell.first_ao + shell.nao {
            // conj(C_i,mu) (S C_a)_mu  +  conj(C_a,mu) (S C_i)_mu  = 2 Re(conj(C_i)(SC_a))
            let cir = coeff.re[(mu, i)];
            let cii = coeff.im[(mu, i)];
            let car = coeff.re[(mu, a)];
            let cai = coeff.im[(mu, a)];
            let sar = sc.re[(mu, a)];
            let sai = sc.im[(mu, a)];
            let sir = sc.re[(mu, i)];
            let sii = sc.im[(mu, i)];
            // Re(conj(C_i)(SC_a)) = cir*sar + cii*sai
            // Re(conj(C_a)(SC_i)) = car*sir + cai*sii
            q[sidx] -= (cir * sar + cii * sai) + (car * sir + cai * sii);
        }
    }
    q
}

/// Finite-temperature Gamma-point CPXTB density and energy-weighted density
/// response for one Cartesian DOF. The integer occ-virt CPXTB cannot represent the
/// occupation response `df_i/dR` of fractionally occupied bands, so this mirrors
/// the molecular finite-T branch in [`crate::cphf`]: the full-band response
/// coefficient matrix (occupation-diagonal + (f_i-f_j)/(eps_i-eps_j) off-diagonal
/// with the kt-slope small-gap limit) with a single global Fermi level, coupled
/// self-consistently through the periodic SCC response kernel. The resulting `dP`,
/// `dW` already carry the occupation response, so the downstream shell-charge
/// response (`density_shell_charges`) and Hessian assembly are unchanged.
fn gamma_finite_temperature_response(
    scf: &PbcSccResult,
    mos: &GammaMos,
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    kernel: &Matrix,
    ground_density: &Matrix,
    kt: f64,
) -> Result<(Matrix, Matrix)> {
    let basis = &scf.basis;
    let n = basis.len();
    let nsh = basis.shells.len();
    // Self-consistent response Fock: the geometry derivative changes the Mulliken
    // charges, whose induced potential feeds back into the density response.
    let mut response_fock = Matrix::zeros(n, n);
    let mut shell_response = vec![0.0_f64; nsh];
    let response_mixing = 0.35_f64;
    for _ in 0..50 {
        let (next_density, _) = crate::cphf::finite_temperature_density_response(
            &mos.coeff,
            &mos.occupations,
            &mos.energies,
            fock_deriv,
            overlap_deriv,
            &response_fock,
            kt,
        )?;
        let next_shell = crate::cphf::response_shell_charges_from_density(
            basis,
            &mos.overlap,
            ground_density,
            &next_density,
            overlap_deriv,
        )?;
        let shell_delta = shell_response
            .iter()
            .zip(next_shell.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let mixed_shell = if shell_response.iter().any(|v| v.abs() > 0.0) {
            shell_response
                .iter()
                .zip(next_shell.iter())
                .map(|(&old, &new)| old + response_mixing * (new - old))
                .collect::<Vec<_>>()
        } else {
            next_shell.clone()
        };
        let shell_potential = crate::linalg::matrix_vector_product(kernel, &mixed_shell)?;
        response_fock =
            crate::cphf::scalar_response_fock_matrix(basis, &mos.overlap, &shell_potential)?;
        shell_response = mixed_shell;
        if shell_delta < 1.0e-12 {
            break;
        }
    }
    let (final_density, final_occupation) = crate::cphf::finite_temperature_density_response(
        &mos.coeff,
        &mos.occupations,
        &mos.energies,
        fock_deriv,
        overlap_deriv,
        &response_fock,
        kt,
    )?;
    let weighted = crate::cphf::finite_temperature_energy_weighted_response(
        &mos.coeff,
        &mos.occupations,
        &final_occupation,
        &mos.energies,
        fock_deriv,
        overlap_deriv,
        &response_fock,
        kt,
    )?;
    Ok((final_density, weighted))
}

/// Occupied-occupied metric density response from the overlap derivative:
/// `-1/2 (f_i+f_j) <i|S^y|j> C_i C_j^T`.
fn metric_density_response(mos: &GammaMos, overlap_deriv: &Matrix) -> Matrix {
    let n = mos.coeff.rows();
    let mut out = Matrix::zeros(n, n);
    let occ = &mos.occupations;
    for i in 0..occ.len() {
        if occ[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occ.len() {
            if occ[j] <= 1.0e-8 {
                continue;
            }
            let s1 = mo_element(&mos.coeff, overlap_deriv, i, j);
            let weight = -0.5 * (occ[i] + occ[j]) * s1;
            if weight.abs() <= 1.0e-30 {
                continue;
            }
            for mu in 0..n {
                for nu in 0..n {
                    out[(mu, nu)] += weight * mos.coeff[(mu, i)] * mos.coeff[(nu, j)];
                }
            }
        }
    }
    out
}

/// Shell charge response from an AO density response and the overlap derivative
/// (Mulliken): `dq_shell = -sum [ (dP S)_mu mu + (P dS)_mu mu ]`.
fn density_shell_charges(
    basis: &BasisSet,
    mos: &GammaMos,
    density_response: &Matrix,
    overlap_deriv: &Matrix,
) -> Vec<f64> {
    let n = basis.len();
    let mut out = vec![0.0; basis.shells.len()];
    for nu in 0..n {
        let mut pop = 0.0;
        for kappa in 0..n {
            pop += density_response[(nu, kappa)] * mos.overlap[(kappa, nu)];
        }
        out[basis.aos[nu].shell_index] -= pop;
    }
    // ground-state density contracted with dS
    let occ = &mos.occupations;
    for nu in 0..n {
        let mut pop = 0.0;
        for kappa in 0..n {
            let mut p0 = 0.0;
            for i in 0..occ.len() {
                if occ[i] > 1.0e-14 {
                    p0 += occ[i] * mos.coeff[(nu, i)] * mos.coeff[(kappa, i)];
                }
            }
            pop += p0 * overlap_deriv[(kappa, nu)];
        }
        out[basis.aos[nu].shell_index] -= pop;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn weighted_density_response(
    mos: &GammaMos,
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    u: &[f64],
    pairs: &[(usize, usize)],
    kernel: &Matrix,
    _transition: &[Vec<f64>],
    basis: &BasisSet,
) -> Matrix {
    let n = mos.coeff.rows();
    let occ = &mos.occupations;
    let mut out = Matrix::zeros(n, n);
    // occ-virt part: weight = (f_i-f_a) eps_i u
    for (row, &(i, a)) in pairs.iter().enumerate() {
        let weight = (occ[i] - occ[a]) * mos.energies[i] * u[row];
        if weight == 0.0 {
            continue;
        }
        for mu in 0..n {
            for nu in 0..n {
                out[(mu, nu)] += weight
                    * (mos.coeff[(mu, a)] * mos.coeff[(nu, i)]
                        + mos.coeff[(mu, i)] * mos.coeff[(nu, a)]);
            }
        }
    }
    // occ-occ metric energy-weighted part: 0.5(f_i+f_j)(F^y_ij - (eps_i+eps_j) S^y_ij)
    // plus the SCC response potential contribution.
    let metric_density = metric_density_response(mos, overlap_deriv);
    let metric_shell = density_shell_charges(basis, mos, &metric_density, overlap_deriv);
    // occ-virt density response shell charges
    let mut ov_density = Matrix::zeros(n, n);
    for (row, &(i, a)) in pairs.iter().enumerate() {
        let weight = (occ[i] - occ[a]) * u[row];
        if weight == 0.0 {
            continue;
        }
        for mu in 0..n {
            for nu in 0..n {
                ov_density[(mu, nu)] += weight
                    * (mos.coeff[(mu, a)] * mos.coeff[(nu, i)]
                        + mos.coeff[(mu, i)] * mos.coeff[(nu, a)]);
            }
        }
    }
    let ov_shell = density_shell_charges(basis, mos, &ov_density, &Matrix::zeros(n, n));
    let mut total_shell = vec![0.0; basis.shells.len()];
    for s in 0..total_shell.len() {
        total_shell[s] = metric_shell[s] + ov_shell[s];
    }
    let response_pot =
        crate::linalg::matrix_vector_product(kernel, &total_shell).expect("kernel mv");
    let mut response_fock = Matrix::zeros(n, n);
    for mu in 0..n {
        let vmu = response_pot[basis.aos[mu].shell_index];
        for nu in 0..n {
            let vnu = response_pot[basis.aos[nu].shell_index];
            response_fock[(mu, nu)] = -0.5 * (vmu + vnu) * mos.overlap[(mu, nu)];
        }
    }
    for i in 0..occ.len() {
        if occ[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occ.len() {
            if occ[j] <= 1.0e-8 {
                continue;
            }
            let f1 = mo_element(&mos.coeff, fock_deriv, i, j)
                + mo_element(&mos.coeff, &response_fock, i, j);
            let s1 = mo_element(&mos.coeff, overlap_deriv, i, j);
            let weight = 0.5 * (occ[i] + occ[j]) * (f1 - (mos.energies[i] + mos.energies[j]) * s1);
            if weight.abs() <= 1.0e-30 {
                continue;
            }
            for mu in 0..n {
                for nu in 0..n {
                    out[(mu, nu)] += weight * mos.coeff[(mu, i)] * mos.coeff[(nu, j)];
                }
            }
        }
    }
    out
}

fn add_in_place(dst: &mut Matrix, src: &Matrix) {
    for i in 0..dst.rows() {
        for j in 0..dst.cols() {
            dst[(i, j)] += src[(i, j)];
        }
    }
}

fn solve_pcg<F>(matvec: &F, rhs: &[f64], precond: &[f64], tol: f64, max_iter: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = rhs.len();
    if n == 0 {
        return Vec::new();
    }
    let inv: Vec<f64> = precond
        .iter()
        .map(|&d| if d.abs() > 1.0e-6 { 1.0 / d } else { 1.0e6 })
        .collect();
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    let target = tol * rhs_norm;
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&inv).map(|(&ri, &mi)| ri * mi).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(&a, &b)| a * b).sum();
    for _ in 0..max_iter {
        let ap = matvec(&p);
        let denom: f64 = p.iter().zip(&ap).map(|(&a, &b)| a * b).sum();
        if denom.abs() < 1.0e-30 {
            break;
        }
        let alpha = rz / denom;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rnorm <= target {
            break;
        }
        for k in 0..n {
            z[k] = r[k] * inv[k];
        }
        let rz_next: f64 = r.iter().zip(&z).map(|(&a, &b)| a * b).sum();
        let beta = rz_next / rz;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
        rz = rz_next;
    }
    x
}

/// Convenience: run the periodic SCC and build the skeleton derivatives.
pub fn gamma_skeleton_from_scratch(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<(PbcSccResult, GammaSkeletonDerivatives)> {
    let scf = run_pbc_scc(system, params, options, pbc)?;
    let skeleton = gamma_skeleton_derivatives(system, params, &scf, options, pbc)?;
    Ok((scf, skeleton))
}

/// Result of a Gamma-point analytic Hessian calculation.
#[derive(Clone, Debug)]
pub struct PbcHessianResult {
    pub scf: PbcSccResult,
    pub hessian: Matrix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Center {
    Bra,
    Ket,
}

#[inline]
fn first_deriv(d_bra: Vec3, d_ket: Vec3, center: Center, axis: usize) -> f64 {
    match center {
        Center::Bra => component(d_bra, axis),
        Center::Ket => component(d_ket, axis),
    }
}

#[inline]
fn second_deriv(
    h_bb: &[[f64; 3]; 3],
    h_bk: &[[f64; 3]; 3],
    h_kk: &[[f64; 3]; 3],
    row: Center,
    col: Center,
    row_axis: usize,
    col_axis: usize,
) -> f64 {
    match (row, col) {
        (Center::Bra, Center::Bra) => h_bb[row_axis][col_axis],
        (Center::Bra, Center::Ket) => h_bk[row_axis][col_axis],
        (Center::Ket, Center::Bra) => h_bk[col_axis][row_axis],
        (Center::Ket, Center::Ket) => h_kk[row_axis][col_axis],
    }
}

/// Radial value, first and second derivatives of the `H0` prefactor
/// `f(r) = coeff * poly(rr)`, `rr = sqrt(r / rad_sum)`, as functions of the
/// interatomic distance `r`. `coeff = 0.5 (se_i+se_j) * hscale` is treated as a
/// constant (the CN dependence is a separate Hessian term).
fn prefactor_radial(
    coeff: f64,
    si: &crate::basis::BasisShell,
    sj: &crate::basis::BasisShell,
    r: f64,
) -> Result<(f64, f64, f64)> {
    let rad = atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?;
    let pi = si.poly_raw.unwrap_or(0.0);
    let pj = sj.poly_raw.unwrap_or(0.0);
    let rr = (r / rad).sqrt();
    let poly = (1.0 + pi * rr) * (1.0 + pj * rr);
    let dpoly_drr = pi * (1.0 + pj * rr) + pj * (1.0 + pi * rr);
    let d2poly_drr2 = 2.0 * pi * pj;
    let drr_dr = 0.5 / (rad * rr).max(1.0e-300);
    let d2rr_dr2 = -0.25 / (rad * rad * rr.powi(3)).max(1.0e-300);
    let dpoly_dr = dpoly_drr * drr_dr;
    let d2poly_dr2 = d2poly_drr2 * drr_dr * drr_dr + dpoly_drr * d2rr_dr2;
    Ok((coeff * poly, coeff * dpoly_dr, coeff * d2poly_dr2))
}

/// Build the radial Hessian blocks `h_bra_bra`, `h_bra_ket`, `h_ket_ket` of a
/// scalar radial function from its value derivatives `(fp, fpp)` and the
/// bra->ket vector `rvec` (`= R_bra - R_ket`).
fn radial_second_blocks(
    fp: f64,
    fpp: f64,
    rvec: Vec3,
    r: f64,
) -> ([[f64; 3]; 3], [[f64; 3]; 3], [[f64; 3]; 3]) {
    let n = (rvec / r).to_array();
    let mut bb = [[0.0; 3]; 3];
    let mut bk = [[0.0; 3]; 3];
    let mut kk = [[0.0; 3]; 3];
    for ax in 0..3 {
        for bx in 0..3 {
            let delta = if ax == bx { 1.0 } else { 0.0 };
            // d2 f / dR_bra dR_bra
            let val = fpp * n[ax] * n[bx] + (fp / r) * (delta - n[ax] * n[bx]);
            bb[ax][bx] = val;
            kk[ax][bx] = val;
            bk[ax][bx] = -val;
        }
    }
    (bb, bk, kk)
}

/// Radial-pair second-derivative Hessian block for a pairwise function whose
/// gradient on atom `a` is `dr * prefactor` (so `prefactor = f'(r)/r`), with
/// `prefactor_derivative = d/dr(f'(r)/r)`. Mirrors the repulsion Hessian block.
fn add_radial_hessian(
    hessian: &mut Matrix,
    a: usize,
    b: usize,
    dr: Vec3,
    prefactor: f64,
    prefactor_derivative: f64,
) {
    let r = dr.norm();
    if r <= DIST_EPS {
        return;
    }
    let u = (dr / r).to_array();
    for ax in 0..3 {
        for bx in 0..3 {
            let delta = if ax == bx { 1.0 } else { 0.0 };
            // d2E/dRa_ax dRa_bx = prefactor*delta + prefactor_derivative*r*u_a u_b
            let value = prefactor * delta + prefactor_derivative * r * u[ax] * u[bx];
            hessian[(3 * a + ax, 3 * a + bx)] += value;
            hessian[(3 * b + ax, 3 * b + bx)] += value;
            hessian[(3 * a + ax, 3 * b + bx)] -= value;
            hessian[(3 * b + ax, 3 * a + bx)] -= value;
        }
    }
}

/// Fixed-charge electrostatic Hessian: `1/2 sum_ij q_i q_j d2 Gamma_ij/dx dy`,
/// with QCore `Gamma = Ewald(1/R) - 1/2 eta^-2 Ewald(R^-3) + SR residual`.
/// The on-site and QCore k=0/self terms are position independent. Charges are
/// held fixed (the charge response is the CPXTB part).
fn electrostatic_fixed_hessian(
    system: &PeriodicSystem,
    lattice: &Lattice,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
) -> Matrix {
    let nat = system.atoms.len();
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let q_atom = &scf.atomic_charges;

    let alpha = resolve_alpha(system, &pbc.ewald);

    // QCore real-space R^-3 Ewald term plus short-range residual.
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
            let scale = 0.5 * q[i] * q[j];
            if scale == 0.0 {
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
                let mut drem = 0.0;
                let mut d2rem = 0.0;
                if d <= r3_cut {
                    let (_, d1, d2) = qcore_r3_real_value_derivatives(d, eta, alpha);
                    drem += QCORE_R3_COEFF * d1;
                    d2rem += QCORE_R3_COEFF * d2;
                }
                if d <= sr_cut {
                    let (_, d1, d2) = qcore_short_value_derivatives(d, eta);
                    drem += d1;
                    d2rem += d2;
                }
                // prefactor = E'/r = scale*rem'/d ; prefactor' = d/dr(prefactor).
                let prefactor = scale * drem / d;
                let prefactor_deriv = scale * (d2rem / d - drem / (d * d));
                add_radial_hessian(&mut hessian, ai, aj, vec, prefactor, prefactor_deriv);
            }
        }
    }

    // Ewald 1/R: 1/2 sum_AB Q_A Q_B d2 phi_AB/dx dy. Real erfc (radial) +
    // reciprocal (structure-factor) parts.
    let real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let r_offsets = lattice.image_offsets(real_cut);
    let r_trans: Vec<Vec3> = r_offsets.iter().map(|o| lattice.translation(*o)).collect();
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / lattice.volume();
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;

    // Real-space erfc Hessian.
    for a in 0..nat {
        for b in 0..nat {
            let scale = 0.5 * q_atom[a] * q_atom[b];
            if scale == 0.0 {
                continue;
            }
            for t in &r_trans {
                let vec = system.atoms[a].position - system.atoms[b].position - *t;
                let d = vec.norm();
                if d <= DIST_EPS || d > real_cut {
                    continue;
                }
                // g(d)=erfc(ad)/d ; g'(d), g''(d).
                let (gp, gpp) = ewald_real_radial_derivs(d, alpha, two_alpha_sqrtpi);
                let prefactor = scale * gp / d;
                let prefactor_deriv = scale * (gpp / d - gp / (d * d));
                add_radial_hessian(&mut hessian, a, b, vec, prefactor, prefactor_deriv);
            }
        }
    }

    // Reciprocal-space Hessian: phi_AB = (4pi/V) sum_G w_G cos(G.R_AB);
    // d2/dRc dRd = -(4pi/V) sum_G w_G G_c G_d cos(G.R_AB) wrt the same atom,
    // +.. between atoms. Build via structure factors.
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = (-g2 * inv_4a2).exp() / g2;
        let garr = g.to_array();
        // S(G) = sum_A Q_A cos(G.R_A), Sc; Ss = sum_A Q_A sin(G.R_A).
        // 1/2 sum_AB Q_A Q_B cos(G.(R_A-R_B)) = 1/2 (Sc^2 + Ss^2).
        // d2/dR... gives -G_c G_d * [that], distributed over atom pairs.
        for a in 0..nat {
            let pha = g.dot(system.atoms[a].position);
            for b in 0..nat {
                let phb = g.dot(system.atoms[b].position);
                let cosab = (pha - phb).cos();
                let scale = 0.5 * q_atom[a] * q_atom[b] * four_pi_v * w_g * cosab;
                for cx in 0..3 {
                    for dx in 0..3 {
                        let val = scale * garr[cx] * garr[dx];
                        // d2/dRa dRa = -G G cos ; d2/dRa dRb = +G G cos.
                        hessian[(3 * a + cx, 3 * a + dx)] -= val;
                        hessian[(3 * b + cx, 3 * b + dx)] -= val;
                        hessian[(3 * a + cx, 3 * b + dx)] += val;
                        hessian[(3 * b + cx, 3 * a + dx)] += val;
                    }
                }
            }
        }
    }
    qcore_r3_reciprocal_fixed_hessian(system, lattice, alpha, basis, model, q, &mut hessian);

    hessian
}

fn qcore_r3_reciprocal_fixed_hessian(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    basis: &BasisSet,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    hessian: &mut Matrix,
) {
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let pref0 = QCORE_R3_COEFF * PI / lattice.volume();
    let nsh = basis.shells.len();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff_g = pref0 * exp1(g.norm2() * inv_4a2);
        let garr = g.to_array();
        for i in 0..nsh {
            let ai = basis.shells[i].atom_index;
            for j in 0..nsh {
                let qiqj = q[i] * q[j];
                if qiqj == 0.0 {
                    continue;
                }
                let aj = basis.shells[j].atom_index;
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let scale = coeff_g * qiqj * (phases[i] - phases[j]).cos() / (eta * eta);
                for cx in 0..3 {
                    for dx in 0..3 {
                        let val = scale * garr[cx] * garr[dx];
                        hessian[(3 * ai + cx, 3 * ai + dx)] -= val;
                        hessian[(3 * aj + cx, 3 * aj + dx)] -= val;
                        hessian[(3 * ai + cx, 3 * aj + dx)] += val;
                        hessian[(3 * aj + cx, 3 * ai + dx)] += val;
                    }
                }
            }
        }
    }
}

/// First and second radial derivatives of the Ewald real-space term
/// `g(d) = erfc(a d)/d`.
fn ewald_real_radial_derivs(d: f64, alpha: f64, two_alpha_sqrtpi: f64) -> (f64, f64) {
    let e = (-alpha * alpha * d * d).exp();
    let erfc_ad = erfc(alpha * d);
    // g'(d) = -erfc(ad)/d^2 - (2a/sqrtpi) e / d
    let gp = -erfc_ad / (d * d) - two_alpha_sqrtpi * e / d;
    // g''(d): differentiate gp.
    // d/dd[-erfc/d^2] = (2a/sqrtpi) e / d^2 + 2 erfc / d^3
    // d/dd[-(2a/sqrtpi) e / d] = (2a/sqrtpi)[ 2 a^2 d e / d + e / d^2 ]
    //                          = (2a/sqrtpi)[ 2 a^2 e + e / d^2 ]
    let gpp = (two_alpha_sqrtpi * e) / (d * d)
        + 2.0 * erfc_ad / (d * d * d)
        + two_alpha_sqrtpi * (2.0 * alpha * alpha * e + e / (d * d));
    (gp, gpp)
}

/// Inverse Bloch transform of a per-k complex matrix set to real-space images:
/// `M(T)_{mu nu} = sum_k w_k Re[M(k)_{mu nu} e^{-i k.T}]`, keyed by the integer
/// image offset over the AO-cutoff image set. Same convention as the gradient's
/// `realspace_density_images`; at a Gamma-only mesh every image equals `M(Gamma)`.
/// Used for both the ground densities (P,W) and the CPXTB responses (dP,dW).
fn realspace_images(
    per_k: &[CMatrix],
    kpoints: &[KPoint],
    lattice: &Lattice,
    ao_cutoff: f64,
) -> std::collections::HashMap<[i32; 3], Matrix> {
    let n = per_k[0].n;
    let offsets = lattice.image_offsets(ao_cutoff);
    let mut map = std::collections::HashMap::with_capacity(offsets.len());
    for off in &offsets {
        let mut m = Matrix::zeros(n, n);
        for (ik, kp) in kpoints.iter().enumerate() {
            let (c, s) = bloch_phase(kp.fractional, *off);
            let wk = kp.weight;
            let mk = &per_k[ik];
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += wk * (mk.re[(i, j)] * c + mk.im[(i, j)] * s);
                }
            }
        }
        map.insert(off.n, m);
    }
    map
}

/// Real-space ground/response density images bundled for the image-loop Hessian
/// helpers: `p` is P(T) (or dP(T)) and `w` is W(T) (or dW(T)).
struct RealspaceDensity {
    p: std::collections::HashMap<[i32; 3], Matrix>,
    w: std::collections::HashMap<[i32; 3], Matrix>,
}

/// Fixed-density band + Pulay second derivative (no CN second-derivative term),
/// summed over symmetry-unique AO image pairs. Mirrors the molecular
/// fixed-density Pulay Hessian, generalised to lattice images. The density
/// `dens.p`/`dens.w` is the real-space P(T)/W(T) per image (equal to the
/// Gamma density at every image for a Gamma-only mesh).
fn band_pulay_fixed_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    pbc: &PbcOptions,
    dens: &RealspaceDensity,
) -> Result<Matrix> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut hessian = Matrix::zeros(ndof, ndof);
    let self_energy = &scf.bloch.self_energies;
    let mut vao = vec![0.0; basis.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = scf.shell_scc_potential[ish];
        }
    }
    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if p.exponent < *e {
                *e = p.exponent;
            }
        }
    }
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let centers = [Center::Bra, Center::Ket];

    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p0 = &dens.p[&off.n];
        let w0 = &dens.w[&off.n];
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
                // Overlap screening: the AO overlap (and its derivatives) are
                // below exp(-40) ~ 4e-18 beyond this range; skip before the integral.
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let r = r2.sqrt();
                for &mu in &atom_aos[a] {
                    let si_idx = basis.aos[mu].shell_index;
                    let si = &basis.shells[si_idx];
                    for &nu in &atom_aos[b] {
                        let sj_idx = basis.aos[nu].shell_index;
                        let sj = &basis.shells[sj_idx];
                        let pair = contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        // Do NOT skip overlap == 0 pairs: for p-p (and p-d) pairs
                        // a cross component (e.g. p_x-p_y across a bond axis) can
                        // have exactly zero overlap yet a nonzero overlap derivative
                        // and nonzero density, so the H0 scalar second derivative,
                        // the V-Pulay and energy-weighted terms all contribute.
                        // Skipping drops them and breaks low-symmetry Hessians.
                        let overlap = pair.moments[0];
                        let coeff = 0.5
                            * (self_energy[si_idx] + self_energy[sj_idx])
                            * hscale(si, sj, params)?;
                        let (hval, hp, hpp) = prefactor_radial(coeff, si, sj, r)?;
                        let n = (rvec / r).to_array();
                        let h0_d_bra = Vec3::new(hp * n[0], hp * n[1], hp * n[2]);
                        let h0_d_ket = h0_d_bra * -1.0;
                        let (h0_bb, h0_bk, h0_kk) = radial_second_blocks(hp, hpp, rvec, r);
                        let p = p0[(mu, nu)];
                        let w = w0[(mu, nu)];
                        let scalar_shift = vao[mu] + vao[nu];
                        let overlap_coeff = p * (2.0 * hval - scalar_shift) - 2.0 * w;
                        let ds_b = pair.d_bra[0];
                        let ds_k = pair.d_ket[0];
                        for &rc in &centers {
                            let row_atom = if rc == Center::Bra { a } else { b };
                            for ra_ax in 0..3 {
                                let row_coord = 3 * row_atom + ra_ax;
                                let ds_row = first_deriv(ds_b, ds_k, rc, ra_ax);
                                let dh_row = first_deriv(h0_d_bra, h0_d_ket, rc, ra_ax);
                                // Two-centre terms (H0 second derivative and the
                                // V-Pulay/W * d2S piece): the perturbed atom is one
                                // of the pair centres a, b.
                                for &cc in &centers {
                                    let col_atom = if cc == Center::Bra { a } else { b };
                                    for ca_ax in 0..3 {
                                        let d2s = second_deriv(
                                            &pair.h_bra_bra[0],
                                            &pair.h_bra_ket[0],
                                            &pair.h_ket_ket[0],
                                            rc,
                                            cc,
                                            ra_ax,
                                            ca_ax,
                                        );
                                        let ds_col = first_deriv(ds_b, ds_k, cc, ca_ax);
                                        let dh_col = first_deriv(h0_d_bra, h0_d_ket, cc, ca_ax);
                                        let d2h = second_deriv(
                                            &h0_bb, &h0_bk, &h0_kk, rc, cc, ra_ax, ca_ax,
                                        );
                                        let value = overlap_coeff * d2s
                                            + 2.0
                                                * p
                                                * (dh_col * ds_row
                                                    + ds_col * dh_row
                                                    + overlap * d2h);
                                        hessian[(row_coord, 3 * col_atom + ca_ax)] += value;
                                    }
                                }
                                // Three-centre V-Pulay: -p * dS_row * d(V_mu+V_nu)/dR_C
                                // for EVERY atom C. The SCC potential V_mu depends on
                                // the position of every atom (via gamma_{mu,k} q_k), so
                                // the perturbed atom need not be one of the pair centres.
                                for col_coord in 0..ndof {
                                    let dscalar_col = skeleton.shell_potential[col_coord][si_idx]
                                        + skeleton.shell_potential[col_coord][sj_idx];
                                    hessian[(row_coord, col_coord)] -= p * ds_row * dscalar_col;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(hessian)
}

/// First and second radial derivatives of the GFN1 coordination-number counting
/// function `f(r) = 1/(1 + exp(-kcn (rc/r - 1)))` with respect to the distance.
fn cn_count_value_derivatives(kcn: f64, r: f64, rc: f64) -> (f64, f64) {
    let raw = -kcn * (rc / r - 1.0);
    if !(-80.0..=80.0).contains(&raw) {
        return (0.0, 0.0);
    }
    let expt = raw.exp();
    let denom = 1.0 + expt;
    let arg1 = kcn * rc / (r * r); // d(raw)/dr (positive sign convention factored below)
    let arg2 = -2.0 * kcn * rc / (r * r * r); // d2(raw')/dr-ish helper
    let first = -expt * arg1 / (denom * denom);
    let second = -expt * (arg1 * arg1 + arg2) / (denom * denom)
        + 2.0 * expt * expt * arg1 * arg1 / (denom * denom * denom);
    (first, second)
}

/// Per-atom band-energy coordination-number potential `dE_band/dCN_k` for a given
/// Gamma real-space density matrix. Mirrors the CN contribution of the band
/// gradient: on-site diagonal self-energies plus off-site `H0` prefactors.
fn band_cn_potential(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    density: &std::collections::HashMap<[i32; 3], Matrix>,
) -> Result<Vec<f64>> {
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let dsedcn = &scf.bloch.dsedcn;
    let lattice = system.lattice.as_ref().copied().unwrap();
    let mut d_edcn = vec![0.0; nat];

    // On-site diagonal: H0_mu mu = se_mu, so dE/dCN gets dsedcn * P(0)_mu mu.
    let p_origin = &density[&[0, 0, 0]];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            d_edcn[shell.atom_index] += dsedcn[ish] * p_origin[(iao, iao)];
        }
    }

    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for prim in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if prim.exponent < *e {
                *e = prim.exponent;
            }
        }
    }
    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p_off = &density[&off.n];
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
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let rad_sum =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
                let rr = (r2.sqrt() / rad_sum).sqrt();
                for &mu in &atom_aos[a] {
                    let si_idx = basis.aos[mu].shell_index;
                    let si = &basis.shells[si_idx];
                    for &nu in &atom_aos[b] {
                        let sj_idx = basis.aos[nu].shell_index;
                        let sj = &basis.shells[sj_idx];
                        let (moments, _, _) = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        let overlap = moments[0];
                        let hs = hscale(si, sj, params)? * shell_polynomial(si, sj, rr);
                        let val = hs * p_off[(mu, nu)] * overlap;
                        d_edcn[a] += dsedcn[si_idx] * val;
                        d_edcn[b] += dsedcn[sj_idx] * val;
                    }
                }
            }
        }
    }
    Ok(d_edcn)
}

/// Position derivative of the CN potential at fixed density:
/// `de_dcn_dr[k][col] = d(dE_band/dCN_k)/dR_col`. Only off-site pairs contribute
/// (the on-site overlaps are unity and geometry independent).
fn band_cn_potential_position_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    density: &std::collections::HashMap<[i32; 3], Matrix>,
) -> Result<Matrix> {
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let dsedcn = &scf.bloch.dsedcn;
    let lattice = system.lattice.as_ref().copied().unwrap();
    let mut out = Matrix::zeros(nat, ndof);

    let mut atom_aos: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (iao, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(iao);
    }
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for prim in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if prim.exponent < *e {
                *e = prim.exponent;
            }
        }
    }
    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p_off = &density[&off.n];
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
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let rad_sum =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
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
                        let dlog = shell_polynomial_log_derivative(si, sj, rvec, r2);
                        let p = p_off[(mu, nu)];
                        // d(hs*overlap)/dR_a (bra) and /dR_b (ket).
                        let dval_bra = d_bra[0] * hs + dlog * (hs * overlap);
                        let dval_ket = d_ket[0] * hs - dlog * (hs * overlap);
                        let cb = dsedcn[si_idx] * p; // weight for d_edcn[a]
                        let ck = dsedcn[sj_idx] * p; // weight for d_edcn[b]
                        let ba = (dval_bra * cb).to_array();
                        let bk = (dval_ket * cb).to_array();
                        let ka = (dval_bra * ck).to_array();
                        let kk = (dval_ket * ck).to_array();
                        for ax in 0..3 {
                            out[(a, 3 * a + ax)] += ba[ax];
                            out[(a, 3 * b + ax)] += bk[ax];
                            out[(b, 3 * a + ax)] += ka[ax];
                            out[(b, 3 * b + ax)] += kk[ax];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Fixed-density coordination-number second-derivative Hessian. With the GFN1
/// linear CN self-energy (`se = hdiag - kcn*CN`, so `d2 se/dCN2 = 0`) the only
/// contributions are the cross term `dE/dCN_k * (dCN_k/dR_col)(d(hs S)/dR_row)`
/// (and its transpose) plus the counting-function second derivative weighted by
/// `dE/dCN_k`.
fn cn_fixed_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    skeleton: &GammaSkeletonDerivatives,
    pbc: &PbcOptions,
    coordination_cutoff: f64,
    dens: &RealspaceDensity,
) -> Result<Matrix> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut hessian = Matrix::zeros(ndof, ndof);
    let d_edcn = band_cn_potential(system, params, scf, pbc, &dens.p)?;
    let de_dcn_dr = band_cn_potential_position_derivative(system, params, scf, pbc, &dens.p)?;

    // Cross term: M[row][col] = sum_k (dCN_k/dR_row) * (d(dE/dCN_k)/dR_col),
    // added together with its transpose (the two are symmetric counterparts of
    // the H0 self-energy / overlap mixed second derivative).
    for row in 0..ndof {
        for col in 0..ndof {
            let mut m = 0.0;
            for k in 0..nat {
                m += skeleton.coordination[row][k] * de_dcn_dr[(k, col)];
            }
            hessian[(row, col)] += m;
            hessian[(col, row)] += m;
        }
    }

    // Counting-function second derivative weighted by dE/dCN.
    let radii = system
        .atoms
        .iter()
        .map(|atom| crate::data_tables::covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let kcn = CoordinationOptions::default().kcn;
    let cn = coordination_with_derivatives(
        system,
        CoordinationOptions {
            cutoff: coordination_cutoff,
            ..CoordinationOptions::default()
        },
    )?;
    for pair in &cn.pairs {
        if pair.i == pair.j {
            continue;
        }
        let r = pair.r_ij.norm();
        if r <= DIST_EPS {
            continue;
        }
        let rc = radii[pair.i] + radii[pair.j];
        let (first, second) = cn_count_value_derivatives(kcn, r, rc);
        let c = d_edcn[pair.i] + d_edcn[pair.j];
        let pref = c * first / r;
        let dpref = c * (second / r - first / (r * r));
        add_radial_hessian(&mut hessian, pair.i, pair.j, pair.r_ij, pref, dpref);
    }
    Ok(hessian)
}

/// Ewald `1/R` "cross" gradient `sum_AB (dQ_a Q_b + Q_a dQ_b) grad phi_AB`, used
/// for the charge-response part of the electrostatic gradient.
fn ewald_cross_gradient(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    q_atom: &[f64],
    dq_atom: &[f64],
    gradient: &mut [Vec3],
) {
    let nat = system.atoms.len();
    let volume = lattice.volume();
    let real_cut = TAU / alpha;
    let g_cut = 2.0 * alpha * TAU;
    let offsets = lattice.image_offsets(real_cut);
    let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
    let recip = lattice.reciprocal_vectors_within(g_cut, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * PI / volume;
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;

    for a in 0..nat {
        for b in 0..nat {
            let prod = dq_atom[a] * q_atom[b] + q_atom[a] * dq_atom[b];
            if prod == 0.0 {
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
                gradient[a] += vec * (prod * dgdr / d);
            }
        }
    }

    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = (-g2 * inv_4a2).exp() / g2;
        let mut sc = 0.0;
        let mut ss = 0.0;
        let mut dsc = 0.0;
        let mut dss = 0.0;
        for b in 0..nat {
            let ph = g.dot(system.atoms[b].position);
            sc += q_atom[b] * ph.cos();
            ss += q_atom[b] * ph.sin();
            dsc += dq_atom[b] * ph.cos();
            dss += dq_atom[b] * ph.sin();
        }
        for c in 0..nat {
            let ph = g.dot(system.atoms[c].position);
            let factor_q = ph.sin() * sc - ph.cos() * ss;
            let factor_dq = ph.sin() * dsc - ph.cos() * dss;
            let coeff = four_pi_v * w_g * (dq_atom[c] * factor_q + q_atom[c] * factor_dq);
            gradient[c] -= *g * coeff;
        }
    }
}

fn qcore_r3_reciprocal_cross_gradient(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    basis: &BasisSet,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    dq: &[f64],
    gradient: &mut [Vec3],
) {
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let pref0 = -2.0 * QCORE_R3_COEFF * PI / lattice.volume();
    let nsh = basis.shells.len();
    let mut phases = vec![0.0; nsh];
    for (_, g) in &recip {
        for (ish, shell) in basis.shells.iter().enumerate() {
            phases[ish] = g.dot(system.atoms[shell.atom_index].position);
        }
        let coeff = pref0 * exp1(g.norm2() * inv_4a2);
        for i in 0..nsh {
            let atom = basis.shells[i].atom_index;
            let mut factor = 0.0;
            for j in 0..nsh {
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                let phase = (phases[i] - phases[j]).sin() / (eta * eta);
                factor += (dq[i] * q[j] + q[i] * dq[j]) * phase;
            }
            gradient[atom] += *g * (coeff * factor);
        }
    }
}

/// Geometry-only data for one band/Pulay response-gradient AO image pair. The
/// `contracted_pair_with_derivatives` integrals and the prefactors depend only on
/// the geometry and the ground density (not on the perturbing DOF), so they are
/// built once by [`build_response_band_pairs`] and reused across all `ndof`
/// response columns instead of being recomputed per DOF.
struct ResponseBandPair {
    a: usize,
    b: usize,
    mu: usize,
    nu: usize,
    off: [i32; 3],
    d_bra0: Vec3,
    d_ket0: Vec3,
    dlog_poly: Vec3,
    /// `2 hij - (vao_mu + vao_nu)`.
    two_hij_minus_shift: f64,
    /// Ground real-space density `P(T)_{mu nu}`.
    p0_munu: f64,
    /// `2 hij * overlap`.
    two_hij_overlap: f64,
    /// `hscale * shell_poly * overlap` (band CN response off-site prefactor).
    hs_overlap: f64,
    /// `dsedcn` for the two shells (band CN response weights for atoms a, b).
    dsedcn_si: f64,
    dsedcn_sj: f64,
}

/// Density-response lookup for [`response_gradient`]: either a single matrix used
/// for every image (the Gamma case `dP(T) = dP(Gamma)`, which avoids cloning the
/// matrix to every offset) or a genuine per-image map (the k-point back-transform).
enum DensityLookup<'a> {
    Uniform(&'a Matrix),
    Images(&'a std::collections::HashMap<[i32; 3], Matrix>),
}

impl DensityLookup<'_> {
    #[inline]
    fn at(&self, off: &[i32; 3]) -> &Matrix {
        match self {
            DensityLookup::Uniform(m) => m,
            DensityLookup::Images(h) => &h[off],
        }
    }
}

/// Precompute the band/Pulay response-gradient pairs once (overlap-screened):
/// the integrals and ground-density prefactors are DOF-independent.
fn build_response_band_pairs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pground: &std::collections::HashMap<[i32; 3], Matrix>,
    pbc: &PbcOptions,
) -> Result<Vec<ResponseBandPair>> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let n = basis.len();
    let self_energy = &scf.bloch.self_energies;
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
    let mut atom_min_exp = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for p in &ao.primitives {
            let e = &mut atom_min_exp[ao.atom_index];
            if p.exponent < *e {
                *e = p.exponent;
            }
        }
    }
    let images = lattice.image_offsets(pbc.ao_cutoff);
    let cutoff2 = pbc.ao_cutoff * pbc.ao_cutoff;
    let mut out = Vec::new();
    for off in &images {
        let is_origin = off.is_origin();
        if !is_origin && !crate::pairlist::canonical_positive_offset(*off) {
            continue;
        }
        let translation = lattice.translation(*off);
        let p0 = &pground[&off.n];
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
                // Overlap screening (exp(-40)); safe even for the zero-overlap
                // nonzero-derivative short-range pairs (those are within range).
                let ea = atom_min_exp[a];
                let eb = atom_min_exp[b];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let rad_sum =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
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
                        out.push(ResponseBandPair {
                            a,
                            b,
                            mu,
                            nu,
                            off: off.n,
                            d_bra0: d_bra[0],
                            d_ket0: d_ket[0],
                            dlog_poly: shell_polynomial_log_derivative(si, sj, rvec, r2),
                            two_hij_minus_shift: 2.0 * hij - (vao[mu] + vao[nu]),
                            p0_munu: p0[(mu, nu)],
                            two_hij_overlap: 2.0 * hij * overlap,
                            hs_overlap: hs * overlap,
                            dsedcn_si: scf.bloch.dsedcn[si_idx],
                            dsedcn_sj: scf.bloch.dsedcn[sj_idx],
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// CPXTB response gradient for one Cartesian DOF: the change in the analytic
/// gradient due to the density response `(dp, dw, dq)`. The band/Pulay part is
/// driven by the precomputed [`ResponseBandPair`]s (geometry built once).
#[allow(clippy::too_many_arguments)]
fn response_gradient(
    system: &PeriodicSystem,
    _params: &Gfn1Parameters,
    scf: &PbcSccResult,
    band_pairs: &[ResponseBandPair],
    dp: DensityLookup,
    dw: DensityLookup,
    dq: &[f64],
    kernel: &Matrix,
    pbc: &PbcOptions,
    cn: Option<&crate::coordination::CoordinationDerivatives>,
) -> Result<Vec<Vec3>> {
    let lattice = system.lattice.as_ref().copied().unwrap();
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let n = basis.len();
    let mut grad = vec![Vec3::zero(); nat];

    // Response SCC potential (per DOF).
    let vresp_shell = crate::linalg::matrix_vector_product(kernel, dq)?;
    let mut vresp = vec![0.0; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vresp[iao] = vresp_shell[ish];
        }
    }

    // Band/Pulay response from the precomputed (geometry-only) pairs.
    for p in band_pairs {
        let dp_munu = dp.at(&p.off)[(p.mu, p.nu)];
        let dw_munu = dw.at(&p.off)[(p.mu, p.nu)];
        let scalar_resp = vresp[p.mu] + vresp[p.nu];
        let overlap_coeff =
            dp_munu * p.two_hij_minus_shift - p.p0_munu * scalar_resp - 2.0 * dw_munu;
        grad[p.a] += p.d_bra0 * overlap_coeff;
        grad[p.b] += p.d_ket0 * overlap_coeff;
        let poly_grad = p.dlog_poly * (dp_munu * p.two_hij_overlap);
        grad[p.a] += poly_grad;
        grad[p.b] -= poly_grad;
    }

    // Electrostatic charge-response gradient.
    let model = &scf.shell_model;
    let q = &scf.shell_charges;
    let alpha = resolve_alpha(system, &pbc.ewald);
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
            let prod = dq[i] * q[j] + q[i] * dq[j];
            if prod == 0.0 {
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
                grad[ai] += vec * (prod * radial / d);
            }
        }
    }
    let dq_atom = model.atomic_charges(basis, dq);
    ewald_cross_gradient(
        system,
        &lattice,
        alpha,
        &scf.atomic_charges,
        &dq_atom,
        &mut grad,
    );
    qcore_r3_reciprocal_cross_gradient(system, &lattice, alpha, basis, model, q, dq, &mut grad);

    // Coordination-number gradient response: the band CN potential dE/dCN_k
    // changes with the density, so the CN gradient picks up a response term
    // sum_k (dE_response/dCN_k) * dCN_k/dR distributed through the CN pairs.
    if let Some(cn) = cn {
        // Band CN response potential dE_response/dCN_k from the precomputed pairs
        // (off-site) plus the on-site diagonal, instead of recomputing the image/AO
        // integrals per DOF inside band_cn_potential.
        let mut de_resp = vec![0.0; nat];
        let dsedcn = &scf.bloch.dsedcn;
        let dp0 = dp.at(&[0, 0, 0]);
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                de_resp[shell.atom_index] += dsedcn[ish] * dp0[(iao, iao)];
            }
        }
        for p in band_pairs {
            let val = p.hs_overlap * dp.at(&p.off)[(p.mu, p.nu)];
            de_resp[p.a] += p.dsedcn_si * val;
            de_resp[p.b] += p.dsedcn_sj * val;
        }
        for pair in &cn.pairs {
            if pair.i == pair.j {
                continue;
            }
            let r = pair.r_ij.norm();
            if r <= DIST_EPS {
                continue;
            }
            let pref = (de_resp[pair.i] + de_resp[pair.j]) * pair.dcn_dr / r;
            grad[pair.i] += pair.r_ij * pref;
            grad[pair.j] -= pair.r_ij * pref;
        }
    }

    Ok(grad)
}

/// Full Gamma-point analytic Hessian: fixed-density band/Pulay (incl. the
/// three-centre V-Pulay term and, when enabled, the coordination-number second
/// derivative) + Ewald/KO electrostatics + repulsion + periodic D3/halogen
/// classical corrections + the full CPXTB density, charge, weighted-density and
/// CN responses.
pub fn pbc_gamma_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcHessianResult> {
    let _profile = crate::profile::scope("pbc.hessian.total");
    let scf = run_pbc_scc(system, params, options, pbc)?;
    let skeleton = {
        let _p = crate::profile::scope("pbc.hessian.skeleton");
        gamma_skeleton_derivatives(system, params, &scf, options, pbc)?
    };
    let mos = gamma_mos(&scf, scf.nelec)?;
    let (dp, dw) = {
        let _p = crate::profile::scope("pbc.hessian.cphf");
        gamma_cpxtb_density_responses(&scf, &skeleton, &mos)?
    };
    let kernel = periodic_response_kernel(&scf);
    let lattice = system.lattice.as_ref().copied().unwrap();
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let basis = &scf.basis;

    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let cn_cutoff = options.hamiltonian.coordination_cutoff;
    let cn_pairs = if enable_cn {
        Some(coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: cn_cutoff,
                ..CoordinationOptions::default()
            },
        )?)
    } else {
        None
    };

    // Real-space ground densities P(T)/W(T) (Gamma-only here: P(T)=P(Gamma)).
    let ground = RealspaceDensity {
        p: realspace_images(&scf.density_k, &scf.kpoints, &lattice, pbc.ao_cutoff),
        w: realspace_images(&scf.ew_density_k, &scf.kpoints, &lattice, pbc.ao_cutoff),
    };
    let mut hessian = {
        let _p = crate::profile::scope("pbc.hessian.band_pulay");
        band_pulay_fixed_hessian(system, params, &scf, &skeleton, pbc, &ground)?
    };
    {
        let _p = crate::profile::scope("pbc.hessian.electrostatic");
        let es = electrostatic_fixed_hessian(system, &lattice, &scf, pbc);
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += es[(i, j)];
            }
        }
    }
    let rep = crate::repulsion::repulsion_energy_gradient_hessian(system, params)?;
    for i in 0..ndof {
        for j in 0..ndof {
            hessian[(i, j)] += rep.hessian[(i, j)];
        }
    }
    if options.enable_dispersion {
        let _p = crate::profile::scope("pbc.hessian.dispersion");
        let d3 = dispersion_energy_gradient_hessian(
            system,
            params,
            options.d3_reference_path.as_deref(),
        )?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += d3.hessian[(i, j)];
            }
        }
    }
    {
        let _p = crate::profile::scope("pbc.hessian.halogen");
        let xb = halogen_energy_gradient_hessian(system)?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += xb.hessian[(i, j)];
            }
        }
    }
    if enable_cn {
        let _p = crate::profile::scope("pbc.hessian.cn_fixed");
        let cn_fixed = cn_fixed_hessian(system, params, &scf, &skeleton, pbc, cn_cutoff, &ground)?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += cn_fixed[(i, j)];
            }
        }
    }

    // CPXTB response columns. The band/Pulay geometry is precomputed once and
    // reused across all DOFs.
    let _p_resp = crate::profile::scope("pbc.hessian.response_columns");
    let band_pairs = build_response_band_pairs(system, params, &scf, &ground.p, pbc)?;
    for y in 0..ndof {
        let dq = density_shell_charges(basis, &mos, &dp[y], &skeleton.overlap[y]);
        // Gamma: dP(T)=dP(Gamma) for all images, so reuse the single matrix
        // (no per-image clone).
        let col = response_gradient(
            system,
            params,
            &scf,
            &band_pairs,
            DensityLookup::Uniform(&dp[y]),
            DensityLookup::Uniform(&dw[y]),
            &dq,
            &kernel,
            pbc,
            cn_pairs.as_ref(),
        )?;
        for atom in 0..nat {
            hessian[(3 * atom, y)] += col[atom].x;
            hessian[(3 * atom + 1, y)] += col[atom].y;
            hessian[(3 * atom + 2, y)] += col[atom].z;
        }
    }

    // Symmetrize.
    for i in 0..ndof {
        for j in 0..i {
            let avg = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = avg;
            hessian[(j, i)] = avg;
        }
    }

    Ok(PbcHessianResult { scf, hessian })
}

/// Full k-point analytic Hessian. Identical in structure to [`pbc_gamma_hessian`]
/// but (1) the fixed-density band/Pulay/CN second derivatives are summed over the
/// real-space image densities `P(T)/W(T)` (inverse Bloch transform of the per-k
/// SCC densities) and (2) the response columns use the complex k-point CPXTB
/// responses `dP(k)/dW(k)`, back-transformed to `dP(T)/dW(T)`, with the shared
/// real shell-charge response `dq`. The skeleton `dV/dR` and CN derivatives are
/// k-independent (real-space), so the Gamma skeleton is reused for the fixed part;
/// the CPXTB builds its own per-k complex skeletons internally. Reduces to
/// `pbc_gamma_hessian` for a Gamma-only mesh.
///
/// The periodic D3(BJ) dispersion and classical halogen-bond second derivatives
/// are classical (density- and k-independent) and already image-summed, so they
/// are added verbatim from the Gamma path.
pub fn pbc_kpoint_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcHessianResult> {
    let _profile = crate::profile::scope("pbc.kpoint_hessian.total");
    let scf = run_pbc_scc(system, params, options, pbc)?;
    let skeleton = gamma_skeleton_derivatives(system, params, &scf, options, pbc)?;
    let lattice = system.lattice.as_ref().copied().unwrap();
    let nat = system.atoms.len();
    let ndof = 3 * nat;

    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let cn_cutoff = options.hamiltonian.coordination_cutoff;
    let cn_pairs = if enable_cn {
        Some(coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: cn_cutoff,
                ..CoordinationOptions::default()
            },
        )?)
    } else {
        None
    };

    // Real-space ground densities P(T)/W(T) from the per-k SCC densities.
    let ground = RealspaceDensity {
        p: realspace_images(&scf.density_k, &scf.kpoints, &lattice, pbc.ao_cutoff),
        w: realspace_images(&scf.ew_density_k, &scf.kpoints, &lattice, pbc.ao_cutoff),
    };

    // Fixed-density Hessian.
    let mut hessian = {
        let _p = crate::profile::scope("pbc.kpoint_hessian.band_pulay");
        band_pulay_fixed_hessian(system, params, &scf, &skeleton, pbc, &ground)?
    };
    let es = electrostatic_fixed_hessian(system, &lattice, &scf, pbc);
    for i in 0..ndof {
        for j in 0..ndof {
            hessian[(i, j)] += es[(i, j)];
        }
    }
    let rep = crate::repulsion::repulsion_energy_gradient_hessian(system, params)?;
    for i in 0..ndof {
        for j in 0..ndof {
            hessian[(i, j)] += rep.hessian[(i, j)];
        }
    }
    // Periodic D3(BJ) dispersion and classical halogen-bond second derivatives.
    // Both are classical (density- and k-independent) and already image-summed, so
    // they are identical to the Gamma path.
    if options.enable_dispersion {
        let _p = crate::profile::scope("pbc.kpoint_hessian.dispersion");
        let d3 = dispersion_energy_gradient_hessian(
            system,
            params,
            options.d3_reference_path.as_deref(),
        )?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += d3.hessian[(i, j)];
            }
        }
    }
    {
        let _p = crate::profile::scope("pbc.kpoint_hessian.halogen");
        let xb = halogen_energy_gradient_hessian(system)?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += xb.hessian[(i, j)];
            }
        }
    }
    if enable_cn {
        let _p = crate::profile::scope("pbc.kpoint_hessian.cn_fixed");
        let cn_fixed = cn_fixed_hessian(system, params, &scf, &skeleton, pbc, cn_cutoff, &ground)?;
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += cn_fixed[(i, j)];
            }
        }
    }

    // Coupled k-point CPXTB: dP(k)/dW(k) per DOF and the shell-charge response dq.
    let kernel = periodic_response_kernel(&scf);
    let (dp_k, dw_k, dq) = {
        let _p = crate::profile::scope("pbc.kpoint_hessian.cphf");
        kpoint_cpxtb_density_responses(system, params, &scf, options, pbc, true)?
    };

    // Response columns: back-transform dP(k)/dW(k) to real space and contract.
    // The band/Pulay geometry is precomputed once and reused across all DOFs.
    {
        let _p = crate::profile::scope("pbc.kpoint_hessian.response_columns");
        let band_pairs = build_response_band_pairs(system, params, &scf, &ground.p, pbc)?;
        for y in 0..ndof {
            let dp_img = realspace_images(&dp_k[y], &scf.kpoints, &lattice, pbc.ao_cutoff);
            let dw_img = realspace_images(&dw_k[y], &scf.kpoints, &lattice, pbc.ao_cutoff);
            let col = response_gradient(
                system,
                params,
                &scf,
                &band_pairs,
                DensityLookup::Images(&dp_img),
                DensityLookup::Images(&dw_img),
                &dq[y],
                &kernel,
                pbc,
                cn_pairs.as_ref(),
            )?;
            for atom in 0..nat {
                hessian[(3 * atom, y)] += col[atom].x;
                hessian[(3 * atom + 1, y)] += col[atom].y;
                hessian[(3 * atom + 2, y)] += col[atom].z;
            }
        }
    }

    // Symmetrize.
    for i in 0..ndof {
        for j in 0..i {
            let avg = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = avg;
            hessian[(j, i)] = avg;
        }
    }

    Ok(PbcHessianResult { scf, hessian })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_params() -> Option<Gfn1Parameters> {
        let path = std::env::var("GFN1_XTB_PARAM").ok()?;
        Gfn1Parameters::from_file(path).ok()
    }

    fn tight() -> ElectronicOptions {
        ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            max_scc: 500,
            ..ElectronicOptions::default()
        }
    }

    const WATER: &str = "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
         O 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\n\
         H 0.000000 -0.757200 -0.469200\n";

    fn shift(system: &mut PeriodicSystem, dof: usize, h: f64) {
        let atom = dof / 3;
        match dof % 3 {
            0 => system.atoms[atom].position.x += h,
            1 => system.atoms[atom].position.y += h,
            _ => system.atoms[atom].position.z += h,
        }
    }

    // The Ewald transition-charge kernel gradient must equal d(1/2 P^T gamma P)/dR
    // (the energy-gradient convention; the TDA caller multiplies by 2 c).
    #[test]
    fn transition_kernel_gamma_gradient_matches_fd() {
        let Some(params) = load_params() else { return };
        let opts = tight();
        let pbc = PbcOptions::default();
        let cell = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"11 0 0 0 11 0 0 0 11\" pbc=\"T T T\"\n\
             O 0.0 0.0 0.08\nH 0.79 0.59 0.0\nH -0.74 0.57 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let scf = run_pbc_scc(&cell, &params, &opts, &pbc).unwrap();
        let nsh = scf.basis.shells.len();
        let p: Vec<f64> = (0..nsh).map(|s| 0.13 * ((s as f64) - 1.7).sin()).collect();
        let lattice = cell.lattice.as_ref().copied().unwrap();
        let analytic =
            crate::pbc::gradient::transition_kernel_gamma_gradient(&cell, &scf, &pbc, &lattice, &p);
        let energy = |sys: &PeriodicSystem| -> f64 {
            let model = crate::coulomb::ShellChargeModel::build(sys, &scf.basis, &params).unwrap();
            let g = crate::pbc::ewald::periodic_gamma_matrix(sys, &scf.basis, &model, &pbc.ewald)
                .unwrap();
            let mut e = 0.0;
            for i in 0..nsh {
                for j in 0..nsh {
                    e += 0.5 * p[i] * g[(i, j)] * p[j];
                }
            }
            e
        };
        let h = 1.0e-4;
        let mut max_diff = 0.0_f64;
        for dof in 0..3 * cell.atoms.len() {
            let mut plus = cell.clone();
            let mut minus = cell.clone();
            shift(&mut plus, dof, h);
            shift(&mut minus, dof, -h);
            let fd = (energy(&plus) - energy(&minus)) / (2.0 * h);
            let an = match dof % 3 {
                0 => analytic[dof / 3].x,
                1 => analytic[dof / 3].y,
                _ => analytic[dof / 3].z,
            };
            max_diff = max_diff.max((an - fd).abs());
        }
        assert!(
            max_diff < 1.0e-6,
            "transition kernel gamma gradient FD: {max_diff:.3e}"
        );
    }

    // The first-derivative integral routine (used by the gradient/response) and
    // the second-derivative routine (used by the fixed Hessian) must agree on the
    // overlap and its first derivatives, or the fixed and response halves of the
    // Hessian use inconsistent integrals.
    #[test]
    fn first_and_second_derivative_integrals_agree_on_overlap() {
        use crate::basis::{BasisOptions, BasisSet};
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions { nprim: 6 }).unwrap();
        let ra = system.atoms[0].position;
        let rb = system.atoms[1].position;
        let mut max_diff = 0.0_f64;
        for &mu in &[0usize, 1, 2, 3] {
            for &nu in &[4usize] {
                let (m1, db1, dk1) =
                    contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rb);
                let pair2 =
                    contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rb);
                max_diff = max_diff.max((m1[0] - pair2.moments[0]).abs());
                let (db1a, dk1a) = (db1[0].to_array(), dk1[0].to_array());
                let (db2a, dk2a) = (pair2.d_bra[0].to_array(), pair2.d_ket[0].to_array());
                for ax in 0..3 {
                    max_diff = max_diff.max((db1a[ax] - db2a[ax]).abs());
                    max_diff = max_diff.max((dk1a[ax] - dk2a[ax]).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-10,
            "first- vs second-derivative overlap integrals disagree by {max_diff:.3e}"
        );
    }

    // Frozen-charge Gamma Fock F = H0(Gamma) - 1/2 (V_mu+V_nu) S(Gamma) at an
    // arbitrary geometry, with the shell charges held fixed. Its position
    // derivative is exactly the skeleton dFock0/dR.
    fn frozen_fock_gamma(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        opts: &ElectronicOptions,
        pbc: &PbcOptions,
        q_frozen: &[f64],
    ) -> Matrix {
        use crate::basis::{BasisOptions, BasisSet};
        use crate::coulomb::{coulomb_energy_potential_from_matrix, ShellChargeModel};
        use crate::pbc::bloch::BlochBuilder;
        use crate::pbc::ewald::periodic_gamma_matrix;

        let basis = BasisSet::build(system, params, BasisOptions { nprim: opts.nprim }).unwrap();
        let bloch = BlochBuilder::build(
            system,
            &basis,
            params,
            pbc.ao_cutoff,
            opts.hamiltonian.coordination_cutoff,
            opts.hamiltonian.enable_cn_hamiltonian,
        )
        .unwrap();
        let (h0, s) = bloch.h_s_gamma_real();
        let model = ShellChargeModel::build(system, &basis, params).unwrap();
        let gamma = periodic_gamma_matrix(system, &basis, &model, &pbc.ewald).unwrap();
        let scc = coulomb_energy_potential_from_matrix(&basis, &model, q_frozen, &gamma).unwrap();
        let n = basis.len();
        let mut vao = vec![0.0; n];
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                vao[iao] = scc.shell_potential[ish];
            }
        }
        let mut fock = h0;
        for i in 0..n {
            for j in 0..n {
                fock[(i, j)] -= 0.5 * (vao[i] + vao[j]) * s[(i, j)];
            }
        }
        fock
    }

    // The skeleton dFock0(Gamma)/dR (band H0 derivative + Pulay + scalar SCC
    // potential derivative + CN) must match the finite difference of the
    // frozen-charge Gamma Fock.
    #[test]
    fn gamma_fock_skeleton_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let (scf, skeleton) = gamma_skeleton_from_scratch(&base, &params, &opts, &pbc).unwrap();
        let q = scf.shell_charges.clone();
        let n = scf.basis.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;

        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let fp = frozen_fock_gamma(&plus, &params, &opts, &pbc, &q);
            let fm = frozen_fock_gamma(&minus, &params, &opts, &pbc, &q);
            for i in 0..n {
                for j in 0..n {
                    let fd = (fp[(i, j)] - fm[(i, j)]) / (2.0 * h);
                    max_diff = max_diff.max((skeleton.fock[y][(i, j)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "dFock0(Gamma)/dR vs finite difference max diff {max_diff:.3e}"
        );
    }

    // Frozen-charge complex Fock at a general k-point, F(k) = H0(k) - 1/2 (V_mu +
    // V_nu) S(k), used to finite-difference the complex skeleton.
    fn frozen_fock_k(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        opts: &ElectronicOptions,
        pbc: &PbcOptions,
        q_frozen: &[f64],
        fractional: [f64; 3],
    ) -> CMatrix {
        use crate::basis::{BasisOptions, BasisSet};
        use crate::coulomb::{coulomb_energy_potential_from_matrix, ShellChargeModel};
        use crate::pbc::bloch::BlochBuilder;
        use crate::pbc::ewald::periodic_gamma_matrix;

        let basis = BasisSet::build(system, params, BasisOptions { nprim: opts.nprim }).unwrap();
        let bloch = BlochBuilder::build(
            system,
            &basis,
            params,
            pbc.ao_cutoff,
            opts.hamiltonian.coordination_cutoff,
            opts.hamiltonian.enable_cn_hamiltonian,
        )
        .unwrap();
        let (h0, s) = bloch.h_s_at_k(fractional);
        let model = ShellChargeModel::build(system, &basis, params).unwrap();
        let gamma = periodic_gamma_matrix(system, &basis, &model, &pbc.ewald).unwrap();
        let scc = coulomb_energy_potential_from_matrix(&basis, &model, q_frozen, &gamma).unwrap();
        let n = basis.len();
        let mut vao = vec![0.0; n];
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                vao[iao] = scc.shell_potential[ish];
            }
        }
        let mut fock = h0;
        for i in 0..n {
            for j in 0..n {
                let scale = 0.5 * (vao[i] + vao[j]);
                fock.re[(i, j)] -= scale * s.re[(i, j)];
                fock.im[(i, j)] -= scale * s.im[(i, j)];
            }
        }
        fock
    }

    // Frozen-charge complex density P(k): diagonalise the frozen-charge Fock(k)
    // against S(k) and fill the lowest `nelec` embedded states. Its geometry
    // derivative is the no-charge-response (couple = false) CPXTB density response.
    fn frozen_density_k(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        opts: &ElectronicOptions,
        pbc: &PbcOptions,
        q_frozen: &[f64],
        fractional: [f64; 3],
        nelec: usize,
    ) -> CMatrix {
        use crate::basis::{BasisOptions, BasisSet};
        use crate::coulomb::{coulomb_energy_potential_from_matrix, ShellChargeModel};
        use crate::pbc::bloch::BlochBuilder;
        use crate::pbc::ewald::periodic_gamma_matrix;

        let basis = BasisSet::build(system, params, BasisOptions { nprim: opts.nprim }).unwrap();
        let bloch = BlochBuilder::build(
            system,
            &basis,
            params,
            pbc.ao_cutoff,
            opts.hamiltonian.coordination_cutoff,
            opts.hamiltonian.enable_cn_hamiltonian,
        )
        .unwrap();
        let (h0, s) = bloch.h_s_at_k(fractional);
        let model = ShellChargeModel::build(system, &basis, params).unwrap();
        let gamma = periodic_gamma_matrix(system, &basis, &model, &pbc.ewald).unwrap();
        let scc = coulomb_energy_potential_from_matrix(&basis, &model, q_frozen, &gamma).unwrap();
        let n = basis.len();
        let mut vao = vec![0.0; n];
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                vao[iao] = scc.shell_potential[ish];
            }
        }
        let mut fock = h0;
        for i in 0..n {
            for j in 0..n {
                let scale = 0.5 * (vao[i] + vao[j]);
                fock.re[(i, j)] -= scale * s.re[(i, j)];
                fock.im[(i, j)] -= scale * s.im[(i, j)];
            }
        }
        let eig = crate::pbc::complex::hermitian_generalized_eigen(&fock, &s, 1.0e-12).unwrap();
        let mut occ = vec![0.0; 2 * n];
        for o in occ.iter_mut().take(nelec) {
            *o = 1.0;
        }
        crate::pbc::complex::weighted_density(&eig, &occ).unwrap()
    }

    // Uncoupled (frozen-potential) k-point CPXTB density response dP(k)/dR must
    // match the finite difference of the frozen-charge density P(k). This isolates
    // the complex orbital-relaxation RHS, gaps, occ-virt density, and the
    // occupied-occupied metric term from the SCC charge coupling.
    #[test]
    fn kpoint_cpxtb_density_response_frozen_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let q = scf.shell_charges.clone();
        let nelec = scf.nelec.round() as usize;
        let n = scf.basis.len();
        let nk = scf.kpoints.len();
        let ndof = 3 * base.atoms.len();
        let (dp, _dw, _dq) =
            kpoint_cpxtb_density_responses(&base, &params, &scf, &opts, &pbc, false).unwrap();
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            shift(&mut plus, y, h);
            let mut minus = base.clone();
            shift(&mut minus, y, -h);
            for ik in 0..nk {
                let frac = scf.kpoints[ik].fractional;
                let pp = frozen_density_k(&plus, &params, &opts, &pbc, &q, frac, nelec);
                let pm = frozen_density_k(&minus, &params, &opts, &pbc, &q, frac, nelec);
                for i in 0..n {
                    for j in 0..n {
                        let fr = (pp.re[(i, j)] - pm.re[(i, j)]) / (2.0 * h);
                        let fi = (pp.im[(i, j)] - pm.im[(i, j)]) / (2.0 * h);
                        maxdiff = maxdiff.max((dp[y][ik].re[(i, j)] - fr).abs());
                        maxdiff = maxdiff.max((dp[y][ik].im[(i, j)] - fi).abs());
                    }
                }
            }
        }
        assert!(
            maxdiff < 1.0e-6,
            "frozen k-point CPXTB density response vs FD max diff {maxdiff:.3e}"
        );
    }

    // The fully-coupled k-point CPXTB density response dP(k)/dR must match the
    // finite difference of the self-consistent SCC density P(k). This adds the
    // complex transition charges, the real SCC coupling kernel (which couples all
    // k-points through the shared charge), the metric/overlap-derivative charge in
    // the RHS, and the fixed-point solve on top of the frozen-potential pieces.
    #[test]
    fn kpoint_cpxtb_density_response_coupled_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let n = scf.basis.len();
        let nk = scf.kpoints.len();
        let ndof = 3 * base.atoms.len();
        let (dp, _dw, _dq) =
            kpoint_cpxtb_density_responses(&base, &params, &scf, &opts, &pbc, true).unwrap();
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            shift(&mut plus, y, h);
            let mut minus = base.clone();
            shift(&mut minus, y, -h);
            let sp = run_pbc_scc(&plus, &params, &opts, &pbc).unwrap();
            let sm = run_pbc_scc(&minus, &params, &opts, &pbc).unwrap();
            for ik in 0..nk {
                for i in 0..n {
                    for j in 0..n {
                        let fr =
                            (sp.density_k[ik].re[(i, j)] - sm.density_k[ik].re[(i, j)]) / (2.0 * h);
                        let fi =
                            (sp.density_k[ik].im[(i, j)] - sm.density_k[ik].im[(i, j)]) / (2.0 * h);
                        maxdiff = maxdiff.max((dp[y][ik].re[(i, j)] - fr).abs());
                        maxdiff = maxdiff.max((dp[y][ik].im[(i, j)] - fi).abs());
                    }
                }
            }
        }
        assert!(
            maxdiff < 1.0e-6,
            "coupled k-point CPXTB density response vs FD max diff {maxdiff:.3e}"
        );
    }

    // The coupled k-point energy-weighted density response dW(k)/dR must match the
    // finite difference of the self-consistent W(k) (`ew_density_k`). This adds the
    // eps-weighted occ-virt part and the occ-occ block (skeleton Fock + SCC
    // response Fock - overlap-derivative energy weighting) on top of dP.
    #[test]
    fn kpoint_cpxtb_weighted_density_response_coupled_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let n = scf.basis.len();
        let nk = scf.kpoints.len();
        let ndof = 3 * base.atoms.len();
        let (_dp, dw, _dq) =
            kpoint_cpxtb_density_responses(&base, &params, &scf, &opts, &pbc, true).unwrap();
        let h = 1.0e-4;
        let mut maxdiff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            shift(&mut plus, y, h);
            let mut minus = base.clone();
            shift(&mut minus, y, -h);
            let sp = run_pbc_scc(&plus, &params, &opts, &pbc).unwrap();
            let sm = run_pbc_scc(&minus, &params, &opts, &pbc).unwrap();
            for ik in 0..nk {
                for i in 0..n {
                    for j in 0..n {
                        let fr = (sp.ew_density_k[ik].re[(i, j)] - sm.ew_density_k[ik].re[(i, j)])
                            / (2.0 * h);
                        let fi = (sp.ew_density_k[ik].im[(i, j)] - sm.ew_density_k[ik].im[(i, j)])
                            / (2.0 * h);
                        maxdiff = maxdiff.max((dw[y][ik].re[(i, j)] - fr).abs());
                        maxdiff = maxdiff.max((dw[y][ik].im[(i, j)] - fi).abs());
                    }
                }
            }
        }
        assert!(
            maxdiff < 1.0e-6,
            "coupled k-point weighted-density response vs FD max diff {maxdiff:.3e}"
        );
    }

    // The k-point Hessian must reduce EXACTLY to the Gamma-point Hessian for a
    // Gamma-only mesh: P(T)=P(Gamma) at every image and the complex CPXTB collapses
    // to the real one. This validates the whole real-space assembly (fixed band/
    // Pulay/CN + response columns) independently of any finite difference.
    #[test]
    fn kpoint_hessian_reduces_to_gamma_at_gamma_only() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default(); // Gamma-only
        let gamma = pbc_gamma_hessian(&base, &params, &opts, &pbc).unwrap();
        let kpt = pbc_kpoint_hessian(&base, &params, &opts, &pbc).unwrap();
        let ndof = 3 * base.atoms.len();
        let mut max_diff = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_diff = max_diff.max((gamma.hessian[(i, j)] - kpt.hessian[(i, j)]).abs());
            }
        }
        assert!(
            max_diff < 1.0e-8,
            "k-point Hessian vs Gamma Hessian (Gamma-only) max diff {max_diff:.3e}"
        );
    }

    // The full k-point Hessian must match the finite difference of the (verified)
    // k-point analytic gradient. End-to-end validation of the complex CPXTB
    // responses back-transformed to real space and the image-summed fixed Hessian.
    #[test]
    fn kpoint_hessian_matches_gradient_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let result = pbc_kpoint_hessian(&base, &params, &opts, &pbc).unwrap();
        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 1.0e-4;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, &params, &opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "k-point Hessian vs gradient FD max diff {max_diff:.3e}"
        );
    }

    // Finite-temperature (Fermi-smearing) k-point Hessian. A metallic bcc-Li cell
    // on a [2,2,2] mesh at a high electronic temperature has genuinely fractional
    // band occupations and a single global Fermi level, so the complex CPXTB must
    // include the occupation response df_ik/dR with the Brillouin-zone-wide
    // chemical-potential constraint sum_k w_k sum_i df_ik = 0. The analytic Hessian
    // (Mermin free energy A = E - TS) must match the central finite difference of
    // the finite-T-correct k-point analytic free-energy gradient.
    #[test]
    fn kpoint_finite_temperature_hessian_matches_gradient_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"3.51 0 0 0 3.51 0 0 0 3.51\" pbc=\"T T T\"\n\
             Li 0.150000 0.050000 0.000000\n\
             Li 1.755000 1.755000 1.755000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = ElectronicOptions {
            enable_dispersion: false,
            electronic_temperature: 30000.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            max_scc: 500,
            ..ElectronicOptions::default()
        };
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([2, 2, 2]),
            ..PbcOptions::default()
        };
        let result = pbc_kpoint_hessian(&base, &params, &opts, &pbc).unwrap();
        assert!(
            result.scf.electronic_entropy_term.abs() > 1.0e-4,
            "entropy term too small to stress finite-T: {}",
            result.scf.electronic_entropy_term
        );
        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 1.0e-4;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, &params, &opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-8,
            "k-point finite-T Hessian vs gradient FD max diff {max_diff:.3e}"
        );
    }

    // The complex k-point skeleton dS(k)/dR and dFock0(k)/dR must match the finite
    // differences of S(k) and the frozen-charge Fock(k). Uses a small cell so the
    // Bloch images (and hence the imaginary parts) are non-negligible.
    #[test]
    fn kpoint_skeleton_matches_finite_difference() {
        use crate::basis::{BasisOptions, BasisSet};
        use crate::pbc::bloch::BlochBuilder;
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let frac = [0.3_f64, 0.0, 0.0];
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let skeleton =
            kpoint_skeleton_derivatives(&base, &params, &scf, &opts, &pbc, frac).unwrap();
        let q = scf.shell_charges.clone();
        let n = scf.basis.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;

        // Confirm the imaginary parts are actually exercised.
        let mut im_norm = 0.0_f64;
        for y in 0..ndof {
            for i in 0..n {
                for j in 0..n {
                    im_norm = im_norm.max(skeleton.overlap[y].im[(i, j)].abs());
                }
            }
        }
        assert!(
            im_norm > 1.0e-4,
            "k-point skeleton imaginary part negligible ({im_norm:.3e}); test is trivial"
        );

        let s_at = |system: &PeriodicSystem| -> CMatrix {
            let basis =
                BasisSet::build(system, &params, BasisOptions { nprim: opts.nprim }).unwrap();
            let bloch = BlochBuilder::build(
                system,
                &basis,
                &params,
                pbc.ao_cutoff,
                opts.hamiltonian.coordination_cutoff,
                opts.hamiltonian.enable_cn_hamiltonian,
            )
            .unwrap();
            bloch.h_s_at_k(frac).1
        };

        let mut max_s = 0.0_f64;
        let mut max_f = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let sp = s_at(&plus);
            let sm = s_at(&minus);
            let fp = frozen_fock_k(&plus, &params, &opts, &pbc, &q, frac);
            let fm = frozen_fock_k(&minus, &params, &opts, &pbc, &q, frac);
            for i in 0..n {
                for j in 0..n {
                    let fd_s_re = (sp.re[(i, j)] - sm.re[(i, j)]) / (2.0 * h);
                    let fd_s_im = (sp.im[(i, j)] - sm.im[(i, j)]) / (2.0 * h);
                    max_s = max_s.max((skeleton.overlap[y].re[(i, j)] - fd_s_re).abs());
                    max_s = max_s.max((skeleton.overlap[y].im[(i, j)] - fd_s_im).abs());
                    let fd_f_re = (fp.re[(i, j)] - fm.re[(i, j)]) / (2.0 * h);
                    let fd_f_im = (fp.im[(i, j)] - fm.im[(i, j)]) / (2.0 * h);
                    max_f = max_f.max((skeleton.fock[y].re[(i, j)] - fd_f_re).abs());
                    max_f = max_f.max((skeleton.fock[y].im[(i, j)] - fd_f_im).abs());
                }
            }
        }
        assert!(
            max_s < 1.0e-6 && max_f < 1.0e-6,
            "k-point skeleton vs FD: dS {max_s:.3e} dFock {max_f:.3e}"
        );
    }

    // Recomputed complex MOs at each k-point must reconstruct the converged
    // SCC density P(k) (gapped insulator, integer filling).
    #[test]
    fn kpoint_mos_reconstructs_density() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let n = scf.basis.len();
        let mut max_diff = 0.0_f64;
        for ik in 0..scf.kpoints.len() {
            let mos = kpoint_mos(&scf, ik).unwrap();
            let p = crate::pbc::complex::weighted_density(&mos.eig, &mos.occupations).unwrap();
            for i in 0..n {
                for j in 0..n {
                    max_diff = max_diff.max((p.re[(i, j)] - scf.density_k[ik].re[(i, j)]).abs());
                    max_diff = max_diff.max((p.im[(i, j)] - scf.density_k[ik].im[(i, j)]).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-7,
            "k-point MO density reconstruction vs SCC P(k) max diff {max_diff:.3e}"
        );
    }

    // Physical complex MO extraction (one representative per degenerate embedded
    // pair) must also reconstruct P(k) = sum_p focc_p c_p c_p^H. This validates
    // the explicit n-band complex coefficients C(k) used by the k-point CPXTB
    // (distinct from the block-extraction path of weighted_density above).
    #[test]
    fn kpoint_complex_mos_reconstruct_density() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"4.2 0 0 0 12 0 0 0 12\" pbc=\"T T T\"\n\
             H 0.000000 0.300000 0.000000\n\
             H 0.950000 -0.200000 0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: crate::pbc::KMesh::monkhorst_pack([3, 1, 1]),
            ..PbcOptions::default()
        };
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let n = scf.basis.len();
        let mut max_diff = 0.0_f64;
        for ik in 0..scf.kpoints.len() {
            let mos = kpoint_complex_mos(&scf, ik).unwrap();
            // P(k) = sum_p focc_p c_p c_p^H
            for i in 0..n {
                for j in 0..n {
                    let mut pr = 0.0;
                    let mut pi = 0.0;
                    for p in 0..n {
                        let f = mos.occupations[p];
                        if f == 0.0 {
                            continue;
                        }
                        let (cir, cii) = (mos.coeff.re[(i, p)], mos.coeff.im[(i, p)]);
                        let (cjr, cji) = (mos.coeff.re[(j, p)], mos.coeff.im[(j, p)]);
                        // c_i c_j^* = (cir+i cii)(cjr - i cji)
                        pr += f * (cir * cjr + cii * cji);
                        pi += f * (cii * cjr - cir * cji);
                    }
                    max_diff = max_diff.max((pr - scf.density_k[ik].re[(i, j)]).abs());
                    max_diff = max_diff.max((pi - scf.density_k[ik].im[(i, j)]).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-7,
            "k-point complex MO density reconstruction max diff {max_diff:.3e}"
        );
    }

    // The CPXTB Gamma-point density response dP/dR must match the finite
    // difference of the converged Gamma-point SCC density. This validates the
    // whole CPXTB core (RHS, periodic SCC coupling kernel, PCG solve, and the
    // occupied-occupied metric terms) independently.
    #[test]
    fn gamma_cpxtb_density_response_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let skeleton = gamma_skeleton_derivatives(&base, &params, &scf, &opts, &pbc).unwrap();
        let mos = gamma_mos(&scf, scf.nelec).unwrap();
        let (dp, _dw) = gamma_cpxtb_density_responses(&scf, &skeleton, &mos).unwrap();

        let n = scf.basis.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;
        let gamma_density = |system: &PeriodicSystem| -> Matrix {
            run_pbc_scc(system, &params, &opts, &pbc).unwrap().density_k[0]
                .re
                .clone()
        };

        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let pp = gamma_density(&plus);
            let pm = gamma_density(&minus);
            for i in 0..n {
                for j in 0..n {
                    let fd = (pp[(i, j)] - pm[(i, j)]) / (2.0 * h);
                    max_diff = max_diff.max((dp[y][(i, j)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "CPXTB density response vs finite difference max diff {max_diff:.3e}"
        );
    }

    // The shell-charge response dq = d(shell_charges)/dR built from the CPXTB
    // density response must match the finite difference of the converged shell
    // charges (this drives the electrostatic charge-response gradient).
    #[test]
    fn gamma_shell_charge_response_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let skeleton = gamma_skeleton_derivatives(&base, &params, &scf, &opts, &pbc).unwrap();
        let mos = gamma_mos(&scf, scf.nelec).unwrap();
        let (dp, _dw) = gamma_cpxtb_density_responses(&scf, &skeleton, &mos).unwrap();
        let nsh = scf.basis.shells.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;
        let charges = |system: &PeriodicSystem| -> Vec<f64> {
            run_pbc_scc(system, &params, &opts, &pbc)
                .unwrap()
                .shell_charges
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let dq = density_shell_charges(&scf.basis, &mos, &dp[y], &skeleton.overlap[y]);
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let qp = charges(&plus);
            let qm = charges(&minus);
            for s in 0..nsh {
                let fd = (qp[s] - qm[s]) / (2.0 * h);
                max_diff = max_diff.max((dq[s] - fd).abs());
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "shell charge response vs finite difference max diff {max_diff:.3e}"
        );
    }

    // Fixed-charge electrostatic gradient: 1/2 sum_ij q_i q_j dGamma_ij/dR with
    // the shell charges held at their base values.
    fn frozen_es_gradient(
        system: &PeriodicSystem,
        scf: &PbcSccResult,
        pbc: &PbcOptions,
    ) -> Vec<Vec3> {
        let lattice = system.lattice.as_ref().copied().unwrap();
        let nat = system.atoms.len();
        let mut g = vec![Vec3::zero(); nat];
        let basis = &scf.basis;
        let model = &scf.shell_model;
        let q = &scf.shell_charges;
        let alpha = resolve_alpha(system, &pbc.ewald);
        let r3_cut = TAU / alpha;
        let sr_cut = pbc.ewald.sr_cutoff;
        let real_cut = r3_cut.max(sr_cut);
        let offsets = lattice.image_offsets(real_cut);
        let translations: Vec<Vec3> = offsets.iter().map(|o| lattice.translation(*o)).collect();
        for i in 0..basis.shells.len() {
            let ai = basis.shells[i].atom_index;
            let ra = system.atoms[ai].position;
            for j in 0..basis.shells.len() {
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
                    g[ai] += vec * (qiqj * radial / d);
                }
            }
        }
        crate::pbc::gradient::ewald_gradient(system, &lattice, alpha, &scf.atomic_charges, &mut g);
        crate::pbc::gradient::qcore_r3_reciprocal_gradient(
            system, &lattice, alpha, basis, model, q, &mut g,
        );
        g
    }

    // The fixed-charge QCore electrostatic Hessian must match the finite
    // difference of the frozen-charge electrostatic gradient.
    #[test]
    fn electrostatic_fixed_hessian_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let lattice = base.lattice.as_ref().copied().unwrap();
        let analytic = electrostatic_fixed_hessian(&base, &lattice, &scf, &pbc);

        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 1.0e-4;
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = frozen_es_gradient(&plus, &scf, &pbc);
            let gm = frozen_es_gradient(&minus, &scf, &pbc);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((analytic[(3 * atom + axis, y)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "electrostatic fixed Hessian vs finite difference max diff {max_diff:.3e}"
        );
    }

    fn no_cn_tight() -> ElectronicOptions {
        let mut o = tight();
        o.hamiltonian.enable_cn_hamiltonian = false;
        o
    }

    // The Ewald "cross" gradient must equal the charge derivative of the Ewald
    // gradient (the bilinear response used in the electrostatic Hessian response).
    #[test]
    fn ewald_cross_gradient_matches_charge_derivative() {
        let Some(params) = load_params() else {
            return;
        };
        let _ = &params;
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"6 0 0 0 6 0 0 0 6\" pbc=\"T T T\"\nNa 0.3 0.1 0.0\nCl 3.0 3.1 2.9\n",
            0.0,
            false,
        )
        .unwrap();
        let lattice = system.lattice.as_ref().copied().unwrap();
        let alpha = 0.30;
        let q = vec![0.4_f64, -0.4];
        let dq = vec![0.15_f64, -0.05];
        let mut cross = vec![Vec3::zero(); 2];
        ewald_cross_gradient(&system, &lattice, alpha, &q, &dq, &mut cross);

        let eps = 1.0e-6;
        let qp: Vec<f64> = q.iter().zip(&dq).map(|(a, b)| a + eps * b).collect();
        let qm: Vec<f64> = q.iter().zip(&dq).map(|(a, b)| a - eps * b).collect();
        let mut gp = vec![Vec3::zero(); 2];
        let mut gm = vec![Vec3::zero(); 2];
        crate::pbc::gradient::ewald_gradient(&system, &lattice, alpha, &qp, &mut gp);
        crate::pbc::gradient::ewald_gradient(&system, &lattice, alpha, &qm, &mut gm);
        let mut max_diff = 0.0_f64;
        for a in 0..2 {
            for axis in 0..3 {
                let fd = (component(gp[a], axis) - component(gm[a], axis)) / (2.0 * eps);
                max_diff = max_diff.max((component(cross[a], axis) - fd).abs());
            }
        }
        assert!(
            max_diff < 1.0e-7,
            "Ewald cross gradient vs charge-derivative FD max diff {max_diff:.3e}"
        );
    }

    // The periodic repulsion Hessian must match the finite difference of the
    // periodic repulsion gradient (image pairs active in a moderate cell).
    #[test]
    fn periodic_repulsion_hessian_matches_gradient_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let analytic = crate::repulsion::repulsion_energy_gradient_hessian(&base, &params).unwrap();
        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 1.0e-4;
        let mut max_diff = 0.0_f64;
        for col in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, col, h);
            shift(&mut minus, col, -h);
            let gp = crate::repulsion::repulsion_energy_gradient(&plus, &params)
                .unwrap()
                .gradient;
            let gm = crate::repulsion::repulsion_energy_gradient(&minus, &params)
                .unwrap()
                .gradient;
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((analytic.hessian[(3 * atom + axis, col)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "periodic repulsion Hessian vs FD max diff {max_diff:.3e}"
        );
    }

    // Validate the second-derivative overlap integrals against the finite
    // difference of the first-derivative integrals, for a real GFN1 basis pair
    // (independent of any Hessian assembly).
    #[test]
    fn second_derivative_integrals_match_first_derivative_fd() {
        use crate::basis::{BasisOptions, BasisSet};
        let Some(params) = load_params() else {
            return;
        };
        let mol = PeriodicSystem::from_xyz_str("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0.0, false).unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();
        let ra = mol.atoms[0].position;
        let rb = mol.atoms[1].position;
        let h = 1.0e-5;
        let mut max_diff = 0.0_f64;
        // a few O-H AO pairs (including p orbitals)
        for &mu in &[0usize, 1, 2, 3] {
            for &nu in &[basis.shells[0].nao + 0] {
                if mu >= basis.len() || nu >= basis.len() {
                    continue;
                }
                let pair =
                    contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rb);
                // FD of d_bra[0] (overlap derivative wrt bra centre) wrt bra axes.
                for row in 0..3 {
                    for col in 0..3 {
                        let mut rp = ra;
                        let mut rm = ra;
                        match col {
                            0 => {
                                rp.x += h;
                                rm.x -= h;
                            }
                            1 => {
                                rp.y += h;
                                rm.y -= h;
                            }
                            _ => {
                                rp.z += h;
                                rm.z -= h;
                            }
                        }
                        let dp = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rp,
                            rb,
                        );
                        let dm = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rm,
                            rb,
                        );
                        let fd = (component(dp.1[0], row) - component(dm.1[0], row)) / (2.0 * h);
                        max_diff = max_diff.max((pair.h_bra_bra[0][row][col] - fd).abs());
                    }
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "second-derivative overlap integrals vs FD max diff {max_diff:.3e}"
        );
    }

    // The H0 prefactor radial value/first/second derivatives (prefactor_radial)
    // and their assembly into Cartesian gradient/Hessian blocks
    // (radial_second_blocks) must match finite differences of the scalar
    // f(R) = coeff * poly(sqrt(|R|/rad)).
    #[test]
    fn h0_prefactor_radial_derivatives_match_fd() {
        use crate::basis::{BasisOptions, BasisSet};
        let Some(params) = load_params() else {
            return;
        };
        let mol =
            PeriodicSystem::from_xyz_str("2\nCN\nC 0 0 0\nN 0.4 0.6 1.1\n", 0.0, false).unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();
        // last C shell (2p) and last N shell (2p)
        let si_idx = (0..basis.shells.len())
            .filter(|&s| basis.shells[s].atom_index == 0)
            .next_back()
            .unwrap();
        let sj_idx = (0..basis.shells.len())
            .filter(|&s| basis.shells[s].atom_index == 1)
            .next_back()
            .unwrap();
        let si = &basis.shells[si_idx];
        let sj = &basis.shells[sj_idx];
        let coeff = 0.37;
        let rad = atomic_radius_bohr(si.z).unwrap() + atomic_radius_bohr(sj.z).unwrap();
        let pi = si.poly_raw.unwrap_or(0.0);
        let pj = sj.poly_raw.unwrap_or(0.0);
        let f = |ra: Vec3, rb: Vec3| -> f64 {
            let r = (ra - rb).norm();
            let rr = (r / rad).sqrt();
            coeff * (1.0 + pi * rr) * (1.0 + pj * rr)
        };
        let ra = mol.atoms[0].position;
        let rb = mol.atoms[1].position;
        let r = (ra - rb).norm();
        let (_hval, hp, hpp) = prefactor_radial(coeff, si, sj, r).unwrap();
        let n = ((ra - rb) / r).to_array();
        let grad = [hp * n[0], hp * n[1], hp * n[2]];
        let (bb, _bk, _kk) = radial_second_blocks(hp, hpp, ra - rb, r);
        let h = 1.0e-5;
        let mut max_g = 0.0_f64;
        let mut max_h = 0.0_f64;
        for ax in 0..3 {
            let mut rp = ra;
            let mut rm = ra;
            match ax {
                0 => {
                    rp.x += h;
                    rm.x -= h;
                }
                1 => {
                    rp.y += h;
                    rm.y -= h;
                }
                _ => {
                    rp.z += h;
                    rm.z -= h;
                }
            }
            let fd_g = (f(rp, rb) - f(rm, rb)) / (2.0 * h);
            max_g = max_g.max((grad[ax] - fd_g).abs());
            // Hessian row ax: FD of the analytic gradient component.
            let g_at = |rax: Vec3| -> [f64; 3] {
                let rn = (rax - rb).norm();
                let nn = ((rax - rb) / rn).to_array();
                let (_v, hp2, _hpp2) = prefactor_radial(coeff, si, sj, rn).unwrap();
                [hp2 * nn[0], hp2 * nn[1], hp2 * nn[2]]
            };
            let gp = g_at(rp);
            let gm = g_at(rm);
            for bx in 0..3 {
                let fd_h = (gp[bx] - gm[bx]) / (2.0 * h);
                max_h = max_h.max((bb[bx][ax] - fd_h).abs());
            }
        }
        assert!(
            max_g < 1.0e-6 && max_h < 1.0e-6,
            "H0 prefactor radial vs FD: grad {max_g:.3e} hess {max_h:.3e}"
        );
    }

    // Second-derivative overlap integrals must match finite differences for a
    // pair with p AND d orbitals (C-Cl), across all AO combinations and all three
    // centre blocks. This is the d-orbital / p-p generalisation of the s/p test.
    #[test]
    fn second_derivative_integrals_cl_pair_all_aos() {
        use crate::basis::{BasisOptions, BasisSet};
        let Some(params) = load_params() else {
            return;
        };
        let mol =
            PeriodicSystem::from_xyz_str("2\nCCl\nC 0 0 0\nCl 0.3 0.5 1.70\n", 0.0, false).unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();
        let ra = mol.atoms[0].position;
        let rb = mol.atoms[1].position;
        let h = 1.0e-5;
        let c_aos: Vec<usize> = (0..basis.len())
            .filter(|&i| basis.aos[i].atom_index == 0)
            .collect();
        let cl_aos: Vec<usize> = (0..basis.len())
            .filter(|&i| basis.aos[i].atom_index == 1)
            .collect();
        let mut max_bb = 0.0_f64;
        let mut max_bk = 0.0_f64;
        let mut max_kk = 0.0_f64;
        for &mu in &c_aos {
            for &nu in &cl_aos {
                let pair =
                    contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rb);
                for row in 0..3 {
                    for col in 0..3 {
                        let (mut rap, mut ram) = (ra, ra);
                        let (mut rbp, mut rbm) = (rb, rb);
                        match col {
                            0 => {
                                rap.x += h;
                                ram.x -= h;
                                rbp.x += h;
                                rbm.x -= h;
                            }
                            1 => {
                                rap.y += h;
                                ram.y -= h;
                                rbp.y += h;
                                rbm.y -= h;
                            }
                            _ => {
                                rap.z += h;
                                ram.z -= h;
                                rbp.z += h;
                                rbm.z -= h;
                            }
                        }
                        let bbp = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rap,
                            rb,
                        );
                        let bbm = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ram,
                            rb,
                        );
                        let kp = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rbp,
                        );
                        let km = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rbm,
                        );
                        let fd_bb =
                            (component(bbp.1[0], row) - component(bbm.1[0], row)) / (2.0 * h);
                        let fd_bk = (component(kp.1[0], row) - component(km.1[0], row)) / (2.0 * h);
                        let fd_kk = (component(kp.2[0], row) - component(km.2[0], row)) / (2.0 * h);
                        max_bb = max_bb.max((pair.h_bra_bra[0][row][col] - fd_bb).abs());
                        max_bk = max_bk.max((pair.h_bra_ket[0][row][col] - fd_bk).abs());
                        max_kk = max_kk.max((pair.h_ket_ket[0][row][col] - fd_kk).abs());
                    }
                }
            }
        }
        assert!(
            max_bb < 1.0e-6 && max_bk < 1.0e-6 && max_kk < 1.0e-6,
            "C-Cl second-derivative integrals vs FD: bb {max_bb:.3e} bk {max_bk:.3e} kk {max_kk:.3e}"
        );
    }

    // The bra-ket and ket-ket second-derivative overlap blocks (used by the fixed
    // Hessian for off-diagonal atom pairs) must match finite differences: vary the
    // ket centre to obtain d2S/dR_bra dR_ket and d2S/dR_ket dR_ket.
    #[test]
    fn second_derivative_overlap_cross_blocks_match_fd() {
        use crate::basis::{BasisOptions, BasisSet};
        let Some(params) = load_params() else {
            return;
        };
        let mol = PeriodicSystem::from_xyz_str("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0.0, false).unwrap();
        let basis = BasisSet::build(&mol, &params, BasisOptions::default()).unwrap();
        let ra = mol.atoms[0].position;
        let rb = mol.atoms[1].position;
        let h = 1.0e-5;
        let mut max_bk = 0.0_f64;
        let mut max_kk = 0.0_f64;
        for &mu in &[0usize, 1, 2, 3] {
            for &nu in &[basis.shells[0].nao] {
                let pair =
                    contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rb);
                for row in 0..3 {
                    for col in 0..3 {
                        let mut rp = rb;
                        let mut rm = rb;
                        match col {
                            0 => {
                                rp.x += h;
                                rm.x -= h;
                            }
                            1 => {
                                rp.y += h;
                                rm.y -= h;
                            }
                            _ => {
                                rp.z += h;
                                rm.z -= h;
                            }
                        }
                        // d(d_bra)/dR_ket = h_bra_ket ; d(d_ket)/dR_ket = h_ket_ket
                        let dp = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rp,
                        );
                        let dm = contracted_pair_with_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rm,
                        );
                        let fd_bk = (component(dp.1[0], row) - component(dm.1[0], row)) / (2.0 * h);
                        let fd_kk = (component(dp.2[0], row) - component(dm.2[0], row)) / (2.0 * h);
                        max_bk = max_bk.max((pair.h_bra_ket[0][row][col] - fd_bk).abs());
                        max_kk = max_kk.max((pair.h_ket_ket[0][row][col] - fd_kk).abs());
                    }
                }
            }
        }
        assert!(
            max_bk < 1.0e-6 && max_kk < 1.0e-6,
            "cross-block second-derivative overlap vs FD: bra-ket {max_bk:.3e}, ket-ket {max_kk:.3e}"
        );
    }

    // The CPXTB energy-weighted density response dW/dR must match the finite
    // difference of the converged Gamma-point energy-weighted density.
    #[test]
    fn gamma_cpxtb_weighted_density_response_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let skeleton = gamma_skeleton_derivatives(&base, &params, &scf, &opts, &pbc).unwrap();
        let mos = gamma_mos(&scf, scf.nelec).unwrap();
        let (_dp, dw) = gamma_cpxtb_density_responses(&scf, &skeleton, &mos).unwrap();

        let n = scf.basis.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;
        let weighted = |system: &PeriodicSystem| -> Matrix {
            run_pbc_scc(system, &params, &opts, &pbc)
                .unwrap()
                .ew_density_k[0]
                .re
                .clone()
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let wp = weighted(&plus);
            let wm = weighted(&minus);
            for i in 0..n {
                for j in 0..n {
                    let fd = (wp[(i, j)] - wm[(i, j)]) / (2.0 * h);
                    max_diff = max_diff.max((dw[y][(i, j)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "CPXTB weighted-density response vs finite difference max diff {max_diff:.3e}"
        );
    }

    // The full Gamma-point analytic Hessian (fixed-density band/Pulay + Ewald/KO
    // electrostatics + repulsion, plus the full CPXTB density/charge response) vs
    // the finite difference of the analytic gradient, on planar water.
    #[test]
    fn gamma_hessian_matches_gradient_finite_difference_no_cn() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"20 0 0 0 20 0 0 0 20\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = no_cn_tight();
        let pbc = PbcOptions::default();
        let result = pbc_gamma_hessian(&base, &params, &opts, &pbc).unwrap();

        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 5.0e-4;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, &params, &opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    let an = result.hessian[(3 * atom + axis, y)];
                    max_diff = max_diff.max((an - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "Gamma Hessian vs gradient finite difference max diff {max_diff:.3e}"
        );
    }

    // Finite-temperature (Fermi-smearing) Gamma Hessian. A metallic bcc-Li cell at
    // a high electronic temperature has genuinely fractional occupations, so the
    // CPXTB must include the occupation response. The analytic Hessian (Mermin free
    // energy A = E - TS) must match the central finite difference of the analytic
    // free-energy gradient, which is itself finite-T-correct (occupation-stationary
    // and separately FD-verified in pbc::gradient).
    #[test]
    fn gamma_finite_temperature_hessian_matches_gradient_fd() {
        let Some(params) = load_params() else {
            return;
        };
        // bcc lithium, 2-atom cell, one atom displaced off-site so the Hessian is
        // nonzero (same geometry as the finite-T gradient FD test).
        let xyz = "2\nLattice=\"3.51 0 0 0 3.51 0 0 0 3.51\" pbc=\"T T T\"\n\
             Li 0.150000 0.050000 0.000000\n\
             Li 1.755000 1.755000 1.755000\n";
        let opts = ElectronicOptions {
            enable_dispersion: false,
            electronic_temperature: 30000.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            max_scc: 500,
            ..ElectronicOptions::default()
        };
        // Confirm the entropy term is non-trivial, i.e. occupations are genuinely
        // fractional and the finite-T branch is actually being exercised.
        let base = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let scf = run_pbc_scc(&base, &params, &opts, &PbcOptions::default()).unwrap();
        assert!(
            scf.electronic_entropy_term.abs() > 1.0e-4,
            "entropy term too small to stress finite-T: {}",
            scf.electronic_entropy_term
        );
        let diff = gamma_hessian_fd_max_diff(&params, xyz, &opts, 1.0e-4);
        assert!(
            diff < 1.0e-8,
            "Gamma finite-T Hessian vs gradient FD max diff {diff:.3e}"
        );
    }

    // Max abs difference between the analytic Gamma Hessian and the finite
    // difference of the analytic gradient, for a molecule in a large cell.
    fn gamma_hessian_fd_max_diff(
        params: &Gfn1Parameters,
        xyz: &str,
        opts: &ElectronicOptions,
        h: f64,
    ) -> f64 {
        let base = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let pbc = PbcOptions::default();
        let result = pbc_gamma_hessian(&base, params, opts, &pbc).unwrap();
        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, params, opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
                }
            }
        }
        max_diff
    }

    fn cell_xyz(natoms: usize, body: &str) -> String {
        format!("{natoms}\nLattice=\"24 0 0 0 24 0 0 0 24\" pbc=\"T T T\"\n{body}")
    }

    // CN-enabled Gamma Hessian on pyramidal ammonia (a genuinely 3D test of the
    // coordination-number second derivative and response).
    #[test]
    fn gamma_hessian_cn_ammonia() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = cell_xyz(
            4,
            "N 0.000000 0.000000 0.116500\n\
             H 0.000000 0.939700 -0.271800\n\
             H 0.813700 -0.469900 -0.271800\n\
             H -0.813700 -0.469900 -0.271800\n",
        );
        let diff = gamma_hessian_fd_max_diff(&params, &xyz, &tight(), 5.0e-4);
        assert!(
            diff < 1.0e-6,
            "ammonia CN Hessian vs FD max diff {diff:.3e}"
        );
    }

    // CN-enabled Gamma Hessian for a NON-NEUTRAL cell (closed-shell NH4+ cation,
    // net charge +1). The neutralising background is constant in the fixed cell
    // (no force/Hessian contribution) and the CPXTB charge response is traceless
    // (sum dq = 0, so the uniform background cancels there too); this test confirms
    // the full CPXTB Hessian is correct when the cell carries a net charge.
    #[test]
    fn gamma_hessian_cn_charged_ammonium() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "5\nLattice=\"24 0 0 0 24 0 0 0 24\" pbc=\"T T T\"\n\
             N 0.000000 0.000000 0.000000\n\
             H 0.594600 0.594600 0.594600\n\
             H 0.594600 -0.594600 -0.594600\n\
             H -0.594600 0.594600 -0.594600\n\
             H -0.520000 -0.594600 0.660000\n",
            1.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let result = pbc_gamma_hessian(&base, &params, &opts, &pbc).unwrap();
        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 5.0e-4;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, &params, &opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    max_diff = max_diff.max((result.hessian[(3 * atom + axis, y)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "charged ammonium CN Hessian vs FD max diff {max_diff:.3e}"
        );
    }

    // CN-enabled Gamma Hessian on chloromethanol (C, O, Cl, H).
    #[test]
    fn gamma_hessian_cn_chloromethanol() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = cell_xyz(
            6,
            "C  0.000000  0.000000  0.000000\n\
             Cl 1.781000  0.000000  0.000000\n\
             O -0.690000  1.220000  0.000000\n\
             H -0.530000 -0.560000  0.900000\n\
             H -0.530000 -0.560000 -0.900000\n\
             H -1.630000  1.060000  0.000000\n",
        );
        let diff = gamma_hessian_fd_max_diff(&params, &xyz, &tight(), 5.0e-4);
        assert!(
            diff < 1.0e-6,
            "chloromethanol CN Hessian vs FD max diff {diff:.3e}"
        );
    }

    // CN-enabled Gamma Hessian on bromoethanol (Br, C, C, O, H).
    #[test]
    fn gamma_hessian_cn_bromoethanol() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = cell_xyz(
            9,
            "Br 0.000000  0.000000  0.000000\n\
             C  1.970000  0.000000  0.000000\n\
             C  2.530000  1.410000  0.000000\n\
             O  3.950000  1.350000  0.000000\n\
             H  2.320000 -0.530000  0.880000\n\
             H  2.320000 -0.530000 -0.880000\n\
             H  2.170000  1.950000  0.880000\n\
             H  2.170000  1.950000 -0.880000\n\
             H  4.270000  2.260000  0.000000\n",
        );
        let diff = gamma_hessian_fd_max_diff(&params, &xyz, &tight(), 5.0e-4);
        assert!(
            diff < 1.0e-6,
            "bromoethanol CN Hessian vs FD max diff {diff:.3e}"
        );
    }

    // CN-enabled Gamma Hessian on glycine (N, C, C, O, O, H).
    #[test]
    fn gamma_hessian_cn_glycine() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = cell_xyz(
            10,
            "N  1.520000 -0.270000  0.000000\n\
             C  0.280000  0.490000  0.000000\n\
             C -0.980000 -0.340000  0.000000\n\
             O -2.060000  0.200000  0.000000\n\
             O -0.870000 -1.560000  0.000000\n\
             H  0.270000  1.140000  0.880000\n\
             H  0.270000  1.140000 -0.880000\n\
             H  2.330000  0.330000  0.000000\n\
             H  1.560000 -0.870000  0.810000\n\
             H -1.730000 -2.000000  0.000000\n",
        );
        let diff = gamma_hessian_fd_max_diff(&params, &xyz, &tight(), 5.0e-4);
        assert!(
            diff < 1.0e-6,
            "glycine CN Hessian vs FD max diff {diff:.3e}"
        );
    }

    // CN-enabled Gamma Hessian on tetracarbonylnickel(0), Ni(CO)4 (transition
    // metal with d orbitals, tetrahedral).
    #[test]
    fn gamma_hessian_cn_nickel_tetracarbonyl() {
        let Some(params) = load_params() else {
            return;
        };
        let xyz = cell_xyz(
            9,
            "Ni  0.000000  0.000000  0.000000\n\
             C   1.062000  1.062000  1.062000\n\
             O   1.726000  1.726000  1.726000\n\
             C   1.062000 -1.062000 -1.062000\n\
             O   1.726000 -1.726000 -1.726000\n\
             C  -1.062000  1.062000 -1.062000\n\
             O  -1.726000  1.726000 -1.726000\n\
             C  -1.062000 -1.062000  1.062000\n\
             O  -1.726000 -1.726000  1.726000\n",
        );
        let diff = gamma_hessian_fd_max_diff(&params, &xyz, &tight(), 5.0e-4);
        assert!(
            diff < 1.0e-6,
            "Ni(CO)4 CN Hessian vs FD max diff {diff:.3e}"
        );
    }

    // The full Gamma-point analytic Hessian with the coordination-number
    // Hamiltonian enabled (the GFN1 default), vs the finite difference of the
    // analytic gradient. Exercises the CN second-derivative term and the CN
    // gradient response on water.
    #[test]
    fn gamma_hessian_matches_gradient_finite_difference_cn() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "3\nLattice=\"20 0 0 0 20 0 0 0 20\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let result = pbc_gamma_hessian(&base, &params, &opts, &pbc).unwrap();

        let nat = base.atoms.len();
        let ndof = 3 * nat;
        let h = 5.0e-4;
        let grad = |system: &PeriodicSystem| {
            crate::pbc::gradient::pbc_analytic_gradient(system, &params, &opts, &pbc)
                .unwrap()
                .gradient
        };
        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let gp = grad(&plus);
            let gm = grad(&minus);
            for atom in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                    let an = result.hessian[(3 * atom + axis, y)];
                    max_diff = max_diff.max((an - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "CN Gamma Hessian vs gradient finite difference max diff {max_diff:.3e}"
        );
    }

    // The CPXTB response gradient must equal the finite difference of the
    // ground-state gradient with respect to a density perturbation along
    // (dp, dw, dq) at FIXED geometry. This validates response_gradient as a pure
    // density derivative, independent of the fixed Hessian and of geometry FD.
    #[test]
    fn response_gradient_matches_density_derivative_of_ground_gradient() {
        let Some(params) = load_params() else {
            return;
        };
        // Low-symmetry chloromethanol (C, Cl, O, H): exercises p-p and p-d pair
        // responses, where a residual symmetry-forbidden assembly error shows.
        let base = PeriodicSystem::from_xyz_str(
            "6\nLattice=\"24 0 0 0 24 0 0 0 24\" pbc=\"T T T\"\n\
             C  0.000000  0.000000  0.000000\n\
             Cl 1.781000  0.000000  0.000000\n\
             O -0.690000  1.220000  0.000000\n\
             H -0.530000 -0.560000  0.900000\n\
             H -0.530000 -0.560000 -0.900000\n\
             H -1.630000  1.060000  0.000000\n",
            0.0,
            false,
        )
        .unwrap();
        let opts = no_cn_tight();
        let pbc = PbcOptions::default();
        let scf = run_pbc_scc(&base, &params, &opts, &pbc).unwrap();
        let skeleton = gamma_skeleton_derivatives(&base, &params, &scf, &opts, &pbc).unwrap();
        let mos = gamma_mos(&scf, scf.nelec).unwrap();
        let (dp, dw) = gamma_cpxtb_density_responses(&scf, &skeleton, &mos).unwrap();
        let kernel = periodic_response_kernel(&scf);
        let nat = base.atoms.len();
        let n = scf.basis.len();
        let nsh = scf.basis.shells.len();

        let perturbed_ground = |y: usize, eps: f64| -> Vec<Vec3> {
            let dq = density_shell_charges(&scf.basis, &mos, &dp[y], &skeleton.overlap[y]);
            let dv = crate::linalg::matrix_vector_product(&kernel, &dq).unwrap();
            let dq_atom = scf.shell_model.atomic_charges(&scf.basis, &dq);
            let mut s = scf.clone();
            for i in 0..n {
                for j in 0..n {
                    s.density_k[0].re[(i, j)] += eps * dp[y][(i, j)];
                    s.ew_density_k[0].re[(i, j)] += eps * dw[y][(i, j)];
                }
            }
            for sh in 0..nsh {
                s.shell_charges[sh] += eps * dq[sh];
                s.shell_scc_potential[sh] += eps * dv[sh];
            }
            for a in 0..nat {
                s.atomic_charges[a] += eps * dq_atom[a];
            }
            crate::pbc::gradient::pbc_gradient_from_scc(&base, &params, s, &opts, &pbc)
                .unwrap()
                .gradient
        };

        let eps = 1.0e-6;
        let mut max_diff = 0.0_f64;
        let lattice = base.lattice.as_ref().copied().unwrap();
        let pg = realspace_images(&scf.density_k, &scf.kpoints, &lattice, pbc.ao_cutoff);
        let band_pairs = build_response_band_pairs(&base, &params, &scf, &pg, &pbc).unwrap();
        for y in 0..3 * nat {
            let dq = density_shell_charges(&scf.basis, &mos, &dp[y], &skeleton.overlap[y]);
            let analytic = response_gradient(
                &base,
                &params,
                &scf,
                &band_pairs,
                DensityLookup::Uniform(&dp[y]),
                DensityLookup::Uniform(&dw[y]),
                &dq,
                &kernel,
                &pbc,
                None,
            )
            .unwrap();
            let gp = perturbed_ground(y, eps);
            let gm = perturbed_ground(y, -eps);
            for a in 0..nat {
                for axis in 0..3 {
                    let fd = (component(gp[a], axis) - component(gm[a], axis)) / (2.0 * eps);
                    max_diff = max_diff.max((component(analytic[a], axis) - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-5,
            "response gradient vs density-derivative FD max diff {max_diff:.3e}"
        );
    }

    // The skeleton dS(Gamma)/dR matrices must match the finite difference of the
    // folded Gamma overlap.
    #[test]
    fn gamma_overlap_derivative_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let opts = tight();
        let pbc = PbcOptions::default();
        let (scf, skeleton) = gamma_skeleton_from_scratch(&base, &params, &opts, &pbc).unwrap();
        let n = scf.basis.len();
        let ndof = 3 * base.atoms.len();
        let h = 1.0e-4;

        let overlap_gamma = |system: &PeriodicSystem| -> Matrix {
            let s = run_pbc_scc(system, &params, &opts, &pbc).unwrap();
            s.bloch.h_s_gamma_real().1
        };

        let mut max_diff = 0.0_f64;
        for y in 0..ndof {
            let mut plus = base.clone();
            let mut minus = base.clone();
            shift(&mut plus, y, h);
            shift(&mut minus, y, -h);
            let sp = overlap_gamma(&plus);
            let sm = overlap_gamma(&minus);
            for i in 0..n {
                for j in 0..n {
                    let fd = (sp[(i, j)] - sm[(i, j)]) / (2.0 * h);
                    max_diff = max_diff.max((skeleton.overlap[y][(i, j)] - fd).abs());
                }
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "dS(Gamma)/dR vs finite difference max diff {max_diff:.3e}"
        );
    }
}
