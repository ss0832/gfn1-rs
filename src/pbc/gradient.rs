// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic Cartesian gradients for periodic GFN1-xTB.
//!
//! The gradient mirrors the molecular `electronic_gradient_terms`, generalised to
//! lattice images (SI Eq. 23-48):
//!
//! - Band-structure + Pulay forces from the overlap derivatives, summed over
//!   symmetry-unique AO image pairs with the real-space density `P(T)` and
//!   energy-weighted density `W(T)`; plus the distance-polynomial `Pi` and the
//!   coordination-number chain rule.
//! - Second-order electrostatics: the Hellmann-Feynman force of the periodic
//!   `Gamma` matrix, split into the Ewald `1/R` force, the QCore generalized
//!   Ewald `R^-3` binomial term, and the rapidly decaying KO residual.
//! - Periodic repulsion (reused from the shared, already-periodic module).
//! - Periodic D3(BJ) dispersion and the classical halogen-bond correction,
//!   evaluated as finite real-space image sums under the shared pair-list
//!   cutoff convention.
//!
//! The third-order term and the SCC charge response are variational and enter
//! through the SCC potential shift carried in the band loop, exactly as in the
//! molecular path.

use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::harmonic_average;
use crate::data_tables::atomic_radius_bohr;
use crate::dispersion::dispersion_energy_gradient;
use crate::electronic::ElectronicOptions;
use crate::error::Result;
use crate::halogen::halogen_energy_gradient;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::contracted_pair_with_derivatives;
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::{erfc, Vec3};
use crate::pairlist::canonical_positive_offset;
use crate::params::Gfn1Parameters;
use crate::pbc::ewald::{
    exp1, qcore_r3_real_value_derivatives, qcore_short_value_derivatives, resolve_alpha,
    QCORE_R3_COEFF,
};
use crate::pbc::kpoints::bloch_phase;
use crate::pbc::scf::{run_pbc_scc, PbcSccResult};
use crate::pbc::PbcOptions;
use crate::repulsion::repulsion_energy_gradient;
use crate::system::PeriodicSystem;
use rayon::prelude::*;
use std::collections::HashMap;

const SQRT_PI: f64 = 1.772_453_850_905_516;
const TAU: f64 = 5.5;
const DIST_EPS: f64 = 1.0e-12;

/// Result of a periodic analytic-gradient calculation.
#[derive(Clone, Debug)]
pub struct PbcGradientResult {
    pub scf: PbcSccResult,
    pub total_energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub max_gradient: f64,
}

/// Run a periodic SCC and then compute the analytic gradient.
pub fn pbc_analytic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcGradientResult> {
    let scf = run_pbc_scc(system, params, options, pbc)?;
    pbc_gradient_from_scc(system, params, scf, options, pbc)
}

/// Compute the analytic gradient from an existing converged periodic SCC result.
pub fn pbc_gradient_from_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: PbcSccResult,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<PbcGradientResult> {
    let lattice = system
        .lattice
        .as_ref()
        .copied()
        .expect("periodic gradient requires a lattice");
    let nat = system.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];

    // Real-space density P(T) and energy-weighted density W(T) for every image
    // in the AO cutoff (Bloch back-transform of the per-k matrices).
    let density_images = {
        let _p = crate::profile::scope("pbc.gradient.density_images");
        realspace_density_images(system, &scf, &lattice, pbc.ao_cutoff)
    };

    {
        let _p = crate::profile::scope("pbc.gradient.band_cn");
        band_and_cn_gradient(
            system,
            params,
            &scf,
            options,
            pbc.ao_cutoff,
            &lattice,
            &density_images,
            &mut gradient,
        )?;
    }
    {
        let _p = crate::profile::scope("pbc.gradient.electrostatic");
        electrostatic_gradient(system, &scf, pbc, &lattice, &mut gradient)?;
    }

    let rep = repulsion_energy_gradient(system, params)?;
    for atom in 0..nat {
        gradient[atom] += rep.gradient[atom];
    }
    if options.enable_dispersion {
        let _p = crate::profile::scope("pbc.gradient.dispersion");
        let d3 = dispersion_energy_gradient(system, params, options.d3_reference_path.as_deref())?;
        for atom in 0..nat {
            gradient[atom] += d3.gradient[atom];
        }
    }
    {
        let _p = crate::profile::scope("pbc.gradient.halogen");
        let xb = halogen_energy_gradient(system)?;
        for atom in 0..nat {
            gradient[atom] += xb.gradient[atom];
        }
    }
    if options.multipole {
        let _p = crate::profile::scope("pbc.gradient.multipole");
        let mp = multipole_gradient_terms_pbc(system, params, &scf, pbc)?;
        for atom in 0..nat {
            gradient[atom] += mp[atom];
        }
    }

    // Explicit external electric-field term: the band/overlap coupling already
    // carries the field through `scf.shell_scc_potential`; the remaining explicit
    // piece of dE_field/dR at fixed charges is sum_i q_i dv_ext_i/dR_A = -q_A E.
    if let Some(field) = options.external_field.electric_field {
        for (atom, &q) in scf.atomic_charges.iter().enumerate() {
            gradient[atom] -= field * q;
        }
    }

    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    let max_gradient = gradient
        .iter()
        .map(|g| g.x.abs().max(g.y.abs()).max(g.z.abs()))
        .fold(0.0, f64::max);
    let total_energy = scf.total_free;
    Ok(PbcGradientResult {
        scf,
        total_energy,
        gradient,
        forces,
        max_gradient,
    })
}

/// Bloch back-transform: `P_{mu nu}(T) = sum_k w_k Re[P(k)_{mu nu} e^{-i k.T}]`,
/// and likewise for the energy-weighted density.
struct DensityImages {
    p: HashMap<[i32; 3], Matrix>,
    w: HashMap<[i32; 3], Matrix>,
}

/// True if any atom pair survives the overlap screen at lattice image `off`, i.e.
/// the image contributes at least one band/Pulay term. Mirrors the screening in
/// [`band_and_cn_gradient`] so that the two functions agree on which images matter.
fn offset_has_active_pair(
    system: &PeriodicSystem,
    lattice: &Lattice,
    atom_min_exp: &[f64],
    ao_cutoff: f64,
    off: crate::lattice::ImageOffset,
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

/// Build the real-space densities `P(T)/W(T)` only for the images the gradient
/// actually consumes: the origin plus the canonical-positive offsets that carry at
/// least one surviving atom pair. The dense `n x n` matrices for the (typically far
/// more numerous) inactive / mirror images are never allocated, which cuts the peak
/// memory of this step by ~one order of magnitude on a 3D supercell (e.g. diamond
/// 2x2x2: ~729 images -> ~60). `band_and_cn_gradient` skips any image not present.
fn realspace_density_images(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    lattice: &Lattice,
    ao_cutoff: f64,
) -> DensityImages {
    let n = scf.basis.len();
    let nat = system.atoms.len();
    let atom_min_exp = crate::basis::atom_min_exponents(&scf.basis, nat);
    let offsets = lattice.image_offsets(ao_cutoff);
    let mut p = HashMap::new();
    let mut w = HashMap::new();
    for off in &offsets {
        let is_origin = off.is_origin();
        // Only the origin and canonical-positive images with a surviving atom pair
        // are read by the band/Pulay loop; skip building anything else.
        if !is_origin
            && (!canonical_positive_offset(*off)
                || !offset_has_active_pair(system, lattice, &atom_min_exp, ao_cutoff, *off))
        {
            continue;
        }
        let mut pm = Matrix::zeros(n, n);
        let mut wm = Matrix::zeros(n, n);
        for (ik, kp) in scf.kpoints.iter().enumerate() {
            let (c, s) = bloch_phase(kp.fractional, *off);
            let wk = kp.weight;
            let pk = &scf.density_k[ik];
            let wk_mat = &scf.ew_density_k[ik];
            for i in 0..n {
                for j in 0..n {
                    pm[(i, j)] += wk * (pk.re[(i, j)] * c + pk.im[(i, j)] * s);
                    wm[(i, j)] += wk * (wk_mat.re[(i, j)] * c + wk_mat.im[(i, j)] * s);
                }
            }
        }
        p.insert(off.n, pm);
        w.insert(off.n, wm);
    }
    DensityImages { p, w }
}

#[allow(clippy::too_many_arguments)]
fn band_and_cn_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    options: &ElectronicOptions,
    ao_cutoff: f64,
    lattice: &Lattice,
    density: &DensityImages,
    gradient: &mut [Vec3],
) -> Result<()> {
    let basis = &scf.basis;
    let nat = system.atoms.len();
    let n = basis.len();
    let self_energy = &scf.bloch.self_energies;
    let dsedcn = &scf.bloch.dsedcn;
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;

    // AO-resolved SCC potential.
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
    let images = lattice.image_offsets(ao_cutoff);
    let cutoff2 = ao_cutoff * ao_cutoff;

    // Each lattice image's band/Pulay contribution is independent. Compute them in
    // parallel into per-image local accumulators, then reduce in image order. (The
    // reduction regroups the floating-point sums by image, so the result differs
    // from the serial gradient only at the ~1e-15 level, far below SCC accuracy.)
    let per_image: Vec<(Vec<Vec3>, Vec<f64>)> = images
        .par_iter()
        .map(|off| -> Result<(Vec<Vec3>, Vec<f64>)> {
            let mut g_local = vec![Vec3::zero(); nat];
            let mut dedcn_local = vec![0.0; nat];
            let is_origin = off.is_origin();
            if !is_origin && !canonical_positive_offset(*off) {
                return Ok((g_local, dedcn_local));
            }
            // Inactive images (no surviving atom pair) are not built; the inner
            // loop would skip all their pairs anyway, so skip the whole image.
            let (p_img, w_img) = match (density.p.get(&off.n), density.w.get(&off.n)) {
                (Some(p), Some(w)) => (p, w),
                _ => return Ok((g_local, dedcn_local)),
            };
            let translation = lattice.translation(*off);
            for a in 0..nat {
                let ra = system.atoms[a].position;
                for b in 0..nat {
                    // Symmetry-unique atom pairs: origin uses a < b; non-origin
                    // (canonical T) uses all ordered atom pairs. Same-atom origin is
                    // the on-site block (handled by the diagonal CN term).
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
                            // Do NOT skip overlap == 0 pairs: at symmetric geometries
                            // a close pair can have exactly zero overlap yet a nonzero
                            // overlap derivative, so the band force P*hij*dS/dR is
                            // nonzero whenever the density P is nonzero. Skipping it
                            // drops a real force contribution and breaks the Hessian
                            // finite difference at symmetric reference geometries.
                            let overlap = moments[0];
                            let hs = hscale(si, sj, params)? * shell_polynomial(si, sj, rr);
                            let hij = 0.5 * (self_energy[si_idx] + self_energy[sj_idx]) * hs;
                            let p = p_img[(mu, nu)];
                            let w = w_img[(mu, nu)];
                            let scalar_shift = vao[mu] + vao[nu];
                            let overlap_coeff = p * (2.0 * hij - scalar_shift) - 2.0 * w;
                            g_local[a] += d_bra[0] * overlap_coeff;
                            g_local[b] += d_ket[0] * overlap_coeff;

                            let dlog_poly = shell_polynomial_log_derivative(si, sj, rvec, r2);
                            let poly_grad = dlog_poly * (2.0 * p * hij * overlap);
                            g_local[a] += poly_grad;
                            g_local[b] -= poly_grad;

                            if enable_cn {
                                dedcn_local[a] += dsedcn[si_idx] * hs * p * overlap;
                                dedcn_local[b] += dsedcn[sj_idx] * hs * p * overlap;
                            }
                        }
                    }
                }
            }
            Ok((g_local, dedcn_local))
        })
        .collect::<Result<Vec<_>>>()?;
    for (g_local, dedcn_local) in per_image {
        for a in 0..nat {
            gradient[a] += g_local[a];
            d_edcn[a] += dedcn_local[a];
        }
    }

    if enable_cn {
        // On-site diagonal CN contribution from the reference-cell density.
        if let Some(p0) = density.p.get(&[0, 0, 0]) {
            for (ish, shell) in basis.shells.iter().enumerate() {
                for iao in shell.first_ao..shell.first_ao + shell.nao {
                    d_edcn[shell.atom_index] += dsedcn[ish] * p0[(iao, iao)];
                }
            }
        }
        // Distribute dE/dCN through the periodic coordination-number derivatives.
        let cn = coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: options.hamiltonian.coordination_cutoff,
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
            let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
            let gi = pair.r_ij * pref;
            gradient[pair.i] += gi;
            gradient[pair.j] -= gi;
        }
    }
    Ok(())
}

/// Analytic gradient of the **arbitrary-rank periodic multipole** correction, from a converged
/// SCC. The energy `E_mp = ½ Σ_A Σ_l M·V` is variational (moments converged), so only the explicit
/// nuclear derivatives appear: (i) the inter-atomic **kernel force** through the QCore-Ewald field
/// (images), `−`[`periodic_multipole_forces_generic`] (force → gradient); and (ii) the on-site
/// **overlap-Pulay** term `∂E_mp/∂S · dS/dR`. Because the SCC builds the moments from the
/// reference-cell overlap, the Pulay weight `W = ∂E_mp/∂S` is contracted against the reference-cell
/// (`T=0`) overlap derivative, exactly like the molecular path; the full fields (incl. the rank-0
/// charge route) enter `W` since the multipole charge potential is not folded into
/// `shell_scc_potential`. The implicit density response is carried by the base band Pulay (the
/// moment Fock is in the converged Fock that builds the energy-weighted density).
fn multipole_gradient_terms_pbc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
) -> Result<Vec<Vec3>> {
    let nat = system.atoms.len();
    let basis = &scf.basis;
    let n = basis.len();
    if scf.atomic_moments.is_empty() {
        return Ok(vec![Vec3::zero(); nat]);
    }
    let max_rank = scf.atomic_moments[0].len() - 1;
    let alpha = resolve_alpha(system, &pbc.ewald);
    let shell_model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let hardness: Vec<f64> = (0..nat)
        .map(|a| shell_model.hardness[shell_model.atom_offsets[a]])
        .collect();
    let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();

    // (i) inter-atomic kernel force at fixed moments → gradient contribution is −force.
    let force = {
        let _p = crate::profile::scope("pbc.gradient.multipole.kernel_force");
        crate::pbc::ewald_multipole::periodic_multipole_forces_generic(
            system,
            alpha,
            &scf.atomic_moments,
            &hardness,
            max_rank,
        )
    };
    let mut grad: Vec<Vec3> = force.iter().map(|f| -*f).collect();

    // (ii) overlap-Pulay weight W = ∂E_mp/∂S (reference-cell density, full periodic fields).
    let mut p_ref = Matrix::zeros(n, n);
    for (ik, kp) in scf.kpoints.iter().enumerate() {
        let w = kp.weight;
        for i in 0..n {
            for j in 0..n {
                p_ref[(i, j)] += w * scf.density_k[ik].re[(i, j)];
            }
        }
    }
    let v_field = {
        let _p = crate::profile::scope("pbc.gradient.multipole.field");
        if scf.atomic_multipole_fields.is_empty() {
            match crate::pbc::ewald_multipole::PeriodicMultipoleFieldKernel::try_build(
                system, alpha, &hardness, max_rank,
            )
            .as_ref()
            {
                Some(kernel) => kernel.apply(&scf.atomic_moments),
                None => crate::pbc::ewald_multipole::periodic_multipole_fields_generic(
                    system,
                    alpha,
                    &scf.atomic_moments,
                    &hardness,
                    max_rank,
                ),
            }
        } else {
            scf.atomic_multipole_fields.clone()
        }
    };
    let cache = {
        let _p = crate::profile::scope("pbc.gradient.multipole.onsite_cache");
        crate::multipole::OnsiteMomentCache::build_with_aos(basis, nat, &atom_pos, max_rank, None)
    };
    let w_mp = {
        let _p = crate::profile::scope("pbc.gradient.multipole.weight");
        crate::multipole::multipole_weight_from_fields(
            basis,
            nat,
            &atom_pos,
            &p_ref,
            &v_field,
            max_rank,
            Some(&cache),
        )
    };

    // (iii) contract W · dS/dR over reference-cell (T=0) off-site atom pairs: grad += d_bra·2W.
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
            continue; // moments use the reference-cell overlap ⇒ only T=0 dS/dR
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
                        let weight = 2.0 * w_mp[(mu, nu)];
                        grad[atom_mu] += d_bra[0] * weight;
                        grad[atom_nu] += d_ket[0] * weight;
                    }
                }
            }
        }
    }
    Ok(grad)
}

fn electrostatic_gradient(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    lattice: &Lattice,
    gradient: &mut [Vec3],
) -> Result<()> {
    let alpha = resolve_alpha(system, &pbc.ewald);
    let basis = &scf.basis;
    let model = &scf.shell_model;
    let q = &scf.shell_charges;

    // Ewald 1/R force using atomic charges.
    ewald_gradient(system, lattice, alpha, &scf.atomic_charges, gradient);
    qcore_r3_reciprocal_gradient(system, lattice, alpha, basis, model, q, gradient);

    // QCore real-space R^-3 Ewald term plus the residual
    // KO - 1/R + 1/(2 eta^2 R^3).
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
                gradient[ai] += vec * (qiqj * radial / d);
            }
        }
    }
    Ok(())
}

/// Gradient of the second-order Coulomb (Ewald `gamma`) bilinear form `P^T gamma P`
/// for an arbitrary shell-charge vector `p_shell` (e.g. a TDA transition density),
/// returned per atom: `out[A] = d/dR_A (P^T gamma P)`. This is the periodic analog
/// of the molecular `coupling_kernel_gradient` (the `c d(P^T K P)/dR` kernel-
/// derivative piece of the analytic excited-state gradient): only the *explicit*
/// position derivative of the second-order kernel `gamma` enters; the third-order
/// (Hubbard, ground-charge dependent) part has no explicit position derivative at
/// fixed charges and is carried by the CPHF density response elsewhere.
///
/// `gamma = Ewald(atomic monopole) + QCore(shell)`, so the monopole part contracts
/// the atom-summed transition charges `p_atom` and the QCore part the shell charges.
/// The routines reproduce `d/dR (1/2 q^T gamma q)` for a single vector, so the
/// caller multiplies by `2 * coupling` to obtain `coupling * d(P^T gamma P)/dR`.
pub(crate) fn transition_kernel_gamma_gradient(
    system: &PeriodicSystem,
    scf: &PbcSccResult,
    pbc: &PbcOptions,
    lattice: &Lattice,
    p_shell: &[f64],
) -> Vec<Vec3> {
    let nat = system.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];
    let alpha = resolve_alpha(system, &pbc.ewald);
    let basis = &scf.basis;
    let model = &scf.shell_model;

    // Atom-summed transition charges for the Ewald monopole (1/R) part.
    let mut p_atom = vec![0.0; nat];
    for (ish, shell) in basis.shells.iter().enumerate() {
        p_atom[shell.atom_index] += p_shell[ish];
    }
    ewald_gradient(system, lattice, alpha, &p_atom, &mut gradient);
    qcore_r3_reciprocal_gradient(system, lattice, alpha, basis, model, p_shell, &mut gradient);

    // QCore real-space R^-3 Ewald term plus the residual KO - 1/R + 1/(2 eta^2 R^3).
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
            let pipj = p_shell[i] * p_shell[j];
            if pipj == 0.0 {
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
                gradient[ai] += vec * (pipj * radial / d);
            }
        }
    }
    gradient
}

pub(crate) fn qcore_r3_reciprocal_gradient(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    basis: &crate::basis::BasisSet,
    model: &crate::coulomb::ShellChargeModel,
    q: &[f64],
    gradient: &mut [Vec3],
) {
    let recip = lattice.reciprocal_vectors_within(2.0 * alpha * TAU, false);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let pref0 = -2.0 * QCORE_R3_COEFF * std::f64::consts::PI / lattice.volume();
    let nsh = basis.shells.len();
    let phases: Vec<Vec<f64>> = recip
        .iter()
        .map(|(_, g)| {
            basis
                .shells
                .iter()
                .map(|shell| g.dot(system.atoms[shell.atom_index].position))
                .collect()
        })
        .collect();
    for (ig, (_, g)) in recip.iter().enumerate() {
        let e1 = exp1(g.norm2() * inv_4a2);
        let coeff = pref0 * e1;
        for i in 0..nsh {
            if q[i] == 0.0 {
                continue;
            }
            let mut factor = 0.0;
            let phi_i = phases[ig][i];
            for j in 0..nsh {
                if q[j] == 0.0 {
                    continue;
                }
                let eta = harmonic_average(model.hardness[i], model.hardness[j]);
                factor += q[j] * (phi_i - phases[ig][j]).sin() / (eta * eta);
            }
            let atom = basis.shells[i].atom_index;
            gradient[atom] += *g * (coeff * q[i] * factor);
        }
    }
}

pub(crate) fn ewald_gradient(
    system: &PeriodicSystem,
    lattice: &Lattice,
    alpha: f64,
    q_atom: &[f64],
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
    let four_pi_v = 4.0 * std::f64::consts::PI / volume;
    let two_alpha_sqrtpi = 2.0 * alpha / SQRT_PI;

    // Real-space erfc force.
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
                gradient[a] += vec * (qaqb * dgdr / d);
            }
        }
    }

    // Reciprocal-space force via per-G structure factors.
    for (_, g) in &recip {
        let g2 = g.norm2();
        let w_g = (-g2 * inv_4a2).exp() / g2;
        let mut sc = 0.0;
        let mut ss = 0.0;
        for b in 0..nat {
            let ph = g.dot(system.atoms[b].position);
            sc += q_atom[b] * ph.cos();
            ss += q_atom[b] * ph.sin();
        }
        for c in 0..nat {
            let ph = g.dot(system.atoms[c].position);
            let factor = ph.sin() * sc - ph.cos() * ss;
            gradient[c] -= *g * (four_pi_v * w_g * q_atom[c] * factor);
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::PbcOptions;

    fn load_params() -> Option<Gfn1Parameters> {
        let path = std::env::var("GFN1_XTB_PARAM").ok()?;
        Gfn1Parameters::from_file(path).ok()
    }

    /// Tightly-converged SCC so the analytic gradient and the finite-difference
    /// reference agree at the 1e-6 level rather than being limited by the default
    /// loose convergence.
    fn tight() -> ElectronicOptions {
        ElectronicOptions {
            enable_dispersion: false,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            max_scc: 500,
            ..ElectronicOptions::default()
        }
    }

    fn free_energy(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        opts: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> f64 {
        run_pbc_scc(system, params, opts, pbc).unwrap().total_free
    }

    /// Maximum |analytic - central-difference| gradient component over all atoms.
    fn max_gradient_fd_error(
        base: &PeriodicSystem,
        params: &Gfn1Parameters,
        opts: &ElectronicOptions,
        pbc: &PbcOptions,
        h: f64,
    ) -> f64 {
        let analytic = pbc_analytic_gradient(base, params, opts, pbc).unwrap();
        let mut max_diff = 0.0_f64;
        for atom in 0..base.atoms.len() {
            for comp in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus, atom, comp, h);
                shift(&mut minus, atom, comp, -h);
                let fd = (free_energy(&plus, params, opts, pbc)
                    - free_energy(&minus, params, opts, pbc))
                    / (2.0 * h);
                let an = component(analytic.gradient[atom], comp);
                max_diff = max_diff.max((an - fd).abs());
            }
        }
        max_diff
    }

    fn cell(comment_atoms: &str) -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(comment_atoms, 0.0, false).unwrap()
    }

    const WATER_CELL: &str = "3\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
         O 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\n\
         H 0.000000 -0.757200 -0.469200\n";
    const AMMONIA_CELL: &str = "4\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
         N 0.000000 0.000000 0.120000\n\
         H 0.000000 0.938000 -0.280000\n\
         H 0.812000 -0.469000 -0.280000\n\
         H -0.812000 -0.469000 -0.280000\n";

    // Analytic Gamma-point gradient agrees with the finite-difference gradient of
    // the periodic free energy to 1e-6 for water and ammonia in moderate cells
    // (real images active), with tightly-converged SCC.
    #[test]
    fn gamma_gradient_matches_finite_difference_1e6() {
        let Some(params) = load_params() else {
            return;
        };
        let opts = tight();
        let pbc = PbcOptions::default();
        let h = 1.0e-3;
        for xyz in [WATER_CELL, AMMONIA_CELL] {
            let base = cell(xyz);
            let max_diff = max_gradient_fd_error(&base, &params, &opts, &pbc, h);
            assert!(
                max_diff < 1.0e-6,
                "Gamma gradient vs finite difference max diff {max_diff:.3e}"
            );
        }
    }

    // A NON-NEUTRAL periodic cell (closed-shell NH4+ cation, net charge +1) must
    // also match the finite difference. The neutralising background -pi/(alpha^2 V)
    // is constant in the (fixed) cell, so it drops out of the force, and the atomic
    // charges (summing to +1, not 0) flow through `ewald_gradient` unchanged. This
    // confirms the charged-cell electrostatics carry through to the gradient.
    #[test]
    fn charged_cell_gamma_gradient_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let base = PeriodicSystem::from_xyz_str(
            "5\nLattice=\"9 0 0 0 9 0 0 0 9\" pbc=\"T T T\"\n\
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
        let max_diff = max_gradient_fd_error(&base, &params, &opts, &pbc, 1.0e-3);
        assert!(
            max_diff < 1.0e-6,
            "charged-cell gradient vs finite difference max diff {max_diff:.3e}"
        );
    }

    // The same agreement at the 1e-6 level for a non-Gamma k-mesh, which exercises
    // the complex density back-transform (the imaginary blocks contribute only
    // away from Gamma).
    #[test]
    fn kpoint_gradient_matches_finite_difference_1e6() {
        let Some(params) = load_params() else {
            return;
        };
        use crate::pbc::KMesh;
        // Water in a smaller cell so off-Gamma phases are non-trivial.
        let base = cell(
            "3\nLattice=\"6 0 0 0 6 0 0 0 6\" pbc=\"T T T\"\n\
             O 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\n\
             H 0.000000 -0.757200 -0.469200\n",
        );
        let opts = tight();
        let pbc = PbcOptions {
            kmesh: KMesh::monkhorst_pack([2, 2, 2]),
            ..PbcOptions::default()
        };
        let max_diff = max_gradient_fd_error(&base, &params, &opts, &pbc, 1.0e-3);
        assert!(
            max_diff < 1.0e-6,
            "k-point gradient vs finite difference max diff {max_diff:.3e}"
        );
    }

    // Finite-temperature stress test: a metallic system with genuinely fractional
    // occupations. The analytic force is the gradient of the Mermin FREE energy
    // A = E - TS, which is stationary w.r.t. the occupations, so it matches the
    // finite-difference of `total_free` without any occupation-derivative terms.
    // (Differentiating the internal energy instead would NOT match.)
    #[test]
    fn finite_temperature_metal_gradient_matches_free_energy_fd() {
        let Some(params) = load_params() else {
            return;
        };
        use crate::pbc::KMesh;
        // bcc lithium, conventional 2-atom cell (a = 3.51 A); one atom displaced
        // off site so the forces are nonzero.
        let base = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"3.51 0 0 0 3.51 0 0 0 3.51\" pbc=\"T T T\"\n\
             Li 0.150000 0.050000 0.000000\n\
             Li 1.755000 1.755000 1.755000\n",
            0.0,
            false,
        )
        .unwrap();
        // A deliberately high electronic temperature drives genuinely fractional
        // occupations (the gradient's correctness is temperature-independent, so a
        // large smearing is the cleanest stress of the finite-T machinery).
        let opts = ElectronicOptions {
            enable_dispersion: false,
            electronic_temperature: 30000.0,
            ..ElectronicOptions::default()
        };
        let pbc = PbcOptions {
            kmesh: KMesh::monkhorst_pack([2, 2, 2]),
            ..PbcOptions::default()
        };
        let analytic = pbc_analytic_gradient(&base, &params, &opts, &pbc).unwrap();
        // Confirm the entropy term is non-trivial, i.e. occupations are fractional
        // and the finite-temperature machinery is actually being exercised.
        assert!(
            analytic.scf.electronic_entropy_term.abs() > 1.0e-4,
            "entropy term too small to stress finite-T: {}",
            analytic.scf.electronic_entropy_term
        );

        let energy = |system: &PeriodicSystem| run_pbc_scc(system, &params, &opts, &pbc).unwrap();
        let h = 1.0e-4;
        let mut max_free = 0.0_f64;
        for atom in 0..base.atoms.len() {
            for comp in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus, atom, comp, h);
                shift(&mut minus, atom, comp, -h);
                let ep = energy(&plus);
                let em = energy(&minus);
                let fd_free = (ep.total_free - em.total_free) / (2.0 * h);
                let an = component(analytic.gradient[atom], comp);
                max_free = max_free.max((an - fd_free).abs());
            }
        }
        assert!(
            max_free < 2.0e-4,
            "finite-T free-energy gradient vs FD max diff {max_free:.3e}"
        );
    }

    // A4: the periodic multipole gradient (kernel force + overlap-Pulay weight) folded into the
    // full SCC analytic gradient agrees with the finite difference of the periodic free energy on a
    // polar cell. Default rank 2 (dipole+quadrupole), tightly-converged SCC.
    #[test]
    fn gamma_multipole_gradient_matches_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        let mut opts = tight();
        opts.multipole = true; // dipole + quadrupole (rank 2)
        let pbc = PbcOptions::default();
        let base = cell(WATER_CELL);
        let max_diff = max_gradient_fd_error(&base, &params, &opts, &pbc, 1.0e-3);
        assert!(
            max_diff < 1.0e-5,
            "multipole gradient vs finite difference max diff {max_diff:.3e}"
        );
    }

    fn shift(system: &mut PeriodicSystem, atom: usize, comp: usize, h: f64) {
        match comp {
            0 => system.atoms[atom].position.x += h,
            1 => system.atoms[atom].position.y += h,
            _ => system.atoms[atom].position.z += h,
        }
    }

    fn component(v: Vec3, comp: usize) -> f64 {
        match comp {
            0 => v.x,
            1 => v.y,
            _ => v.z,
        }
    }
}
