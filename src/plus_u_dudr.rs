// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic geometry derivative of the DFT+U linear-response Hubbard parameters,
//! `dU_I/dR` (and `dV_IJ/dR`) — the piece that makes the DFT+U *consistent force*
//! `F_corr = Σ_I (∂E/∂U_I)(dU_I/dR)` (see
//! [`crate::gradient::plus_u_consistency_gradient_terms`]) fully analytic.
//!
//! # Background
//!
//! The linear-response Hubbard matrix is `K = χ0⁻¹ − χ⁻¹`, with `U_I = K_II`,
//! `V_IJ = −K_IJ`. Both the bare (`χ0`) and screened (`χ`) occupation-response
//! matrices are built **analytically** in the converged base-state MO basis by
//! [`crate::spin::analytic_chi0`] / [`crate::spin::analytic_chi`]. Their geometry
//! derivatives are
//!
//! ```text
//! dK/dR   = −χ0⁻¹ (dχ0/dR) χ0⁻¹ + χ⁻¹ (dχ/dR) χ⁻¹
//! dU_I/dR = (dK/dR)_II ,   dV_IJ/dR = −(dK/dR)_IJ
//! ```
//!
//! so the task is the **mixed second derivatives** `dχ0/dR`, `dχ/dR`
//! (`∂²n/∂α∂R`). Because `χ0`/`χ` live in the base MO basis, this needs the
//! coupled-perturbed **geometry** response of the base state — `dC^σ/dR`,
//! `dε^σ/dR`, `df^σ/dR` — for the **spin-unrestricted, finite-temperature**
//! two-channel base Fock (`base_fock_a`, `base_fock_b`), plus the AO overlap
//! derivatives `dS/dR` (which enter through the S-dressed probe and the population
//! extraction). A numerical decomposition on ScH (300 K) shows the orbital-geometry
//! response accounts for ~94% of `dχ0/dR`; the direct `dS/dR` term is ~6%.
//!
//! # Structure (stage-gated)
//!
//! Each stage is finite-difference gated (at `electronic_temperature = 300 K` on
//! ScH triplet — the well-conditioned regime) against the FD oracle before the next
//! is built:
//!
//! 1. `dS/dR` — AO overlap first derivative (reuses
//!    [`crate::cphf::overlap_first_derivative_matrix`]).
//! 2. `d(base_fock^σ)/dR` — total SCC-relaxed per-channel base-Fock derivative:
//!    the CN-coupled `dh0/dR` skeleton + explicit `dv_c/dR` (reuses
//!    [`crate::cphf::cartesian_ao_derivative_matrices_raw`] /
//!    [`crate::cphf::shell_scalar_potential_derivatives`]) plus the coupled
//!    `(dq/dR, dm/dR)` charge/magnetization response.
//! 3. `(dq/dR, dm/dR)` — coupled two-channel finite-T SCC CPHF geometry response.
//! 4. `dχ0/dR` — bare response derivative in the base MO basis.
//! 5. `dχ/dR` — screened response derivative (differentiating the χ = χ0 + χ0 K χ
//!    fixed point).
//! 6. `dK/dR → dU_I/dR, dV_IJ/dR` — assembled with the same Tikhonov-regularized
//!    inverses [`crate::plus_u::extract_uv_from_response`] uses, so the analytic
//!    derivative matches the FD-differenced extraction.
//!
//! Until the analytic path is complete and gated, the production consistent force
//! uses the FD `dU/dR` in [`crate::gradient::plus_u_consistency_gradient_terms`];
//! that FD path is the `#[cfg(test)]` oracle every stage here is verified against.

use crate::cphf::{
    cartesian_ao_derivative_matrices_raw, coordination_number_derivatives,
    shell_scalar_potential_derivatives,
};
use crate::error::Result;
use crate::linalg::{matrix_vector_product, Matrix};
use crate::spin::{spin_shell_potential, ChannelBasis, LinearResponseGeomContext, ShellInfo};

// ---------------------------------------------------------------------------------------------
// Sec 2: non-orthogonal finite-temperature MO/density response primitives (per spin channel).
// ---------------------------------------------------------------------------------------------

/// `C^T A C` (MO-basis transform of a symmetric AO matrix).
fn mo_transform(mos: &Matrix, a: &Matrix) -> Result<Matrix> {
    let tmp = a.matmul(mos)?;
    mos.transpose().matmul(&tmp)
}

/// `C · coeff · C^T` (AO density from an MO-basis coefficient matrix).
fn ao_from_mo(mos: &Matrix, coeff: &Matrix) -> Result<Matrix> {
    let tmp = mos.matmul(coeff)?;
    tmp.matmul(&mos.transpose())
}

/// Divided difference `(f_j − f_i)/(ε_j − ε_i)` with the smooth `∂f/∂ε` limit as the gap
/// closes — the near-degenerate ScH d¹-frontier handling. `slope_i`, `slope_j` are the
/// single-channel Fermi slopes `g = ∂f/∂ε = −f(1−f)/kt` at the two orbitals.
fn divided_diff(fi: f64, fj: f64, ei: f64, ej: f64, slope_i: f64, slope_j: f64) -> f64 {
    let de = ej - ei;
    if de.abs() > 1.0e-9 {
        (fj - fi) / de
    } else {
        0.5 * (slope_i + slope_j)
    }
}

/// Per-channel finite-temperature Fermi data around the base state.
struct ChannelResponse<'a> {
    ch: &'a ChannelBasis,
    kt: f64,
    /// `g_i = ∂f_i/∂ε_i = −f_i(1−f_i)/kt` (0 at kt=0).
    slope: Vec<f64>,
    /// `Σ_i g_i` (the Fermi-level denominator).
    slope_sum: f64,
    /// `C^T S` (MO×AO), precomputed once. Used to extract shell populations of a density response
    /// `P = C·coeff·C^T` directly from its MO coefficient — `(P S)_aa = (C·coeff·(C^T S))_aa` — via
    /// one GEMM `coeff·(C^T S)` + an O(N²) diagonal gather, avoiding the full AO `C·coeff·C^T`.
    cts: Matrix,
}

impl<'a> ChannelResponse<'a> {
    fn new(ch: &'a ChannelBasis, kt: f64, overlap: &Matrix) -> Self {
        let slope: Vec<f64> = ch
            .occ
            .iter()
            .map(|&f| if kt > 0.0 { -(f * (1.0 - f)).max(0.0) / kt } else { 0.0 })
            .collect();
        let slope_sum: f64 = slope.iter().sum();
        let cts = ch.mos.transpose().matmul(overlap).expect("C^T S");
        Self { ch, kt, slope, slope_sum, cts }
    }

    fn norb(&self) -> usize {
        self.ch.eps.len()
    }

    /// First-order density response `P^x = C D^x C^T` to an AO perturbation with MO-basis
    /// blocks `f_mo = C^T F^x C` and `s_mo = C^T S^x C` (Sec 2 formulas; overlap-dressed).
    fn density_response(&self, f_mo: &Matrix, s_mo: &Matrix) -> Result<DensityResponse> {
        let n = self.norb();
        let eps = &self.ch.eps;
        let occ = &self.ch.occ;
        // ε^x_i = F^x_ii − ε_i S^x_ii.
        let eps_x: Vec<f64> = (0..n).map(|i| f_mo[(i, i)] - eps[i] * s_mo[(i, i)]).collect();
        // μ^x and f^x (fixed electron count).
        let (mu_x, f_x) = if self.kt > 0.0 && self.slope_sum.abs() > 1.0e-30 {
            let mu = self
                .slope
                .iter()
                .zip(eps_x.iter())
                .map(|(&g, &ex)| g * ex)
                .sum::<f64>()
                / self.slope_sum;
            let fx: Vec<f64> = (0..n).map(|i| self.slope[i] * (eps_x[i] - mu)).collect();
            (mu, fx)
        } else {
            (0.0, vec![0.0; n])
        };
        // T = κ − ½ S^x_MO ;  κ_ij = (F^x_ij − ε̄_ij S^x_ij)/(ε_j − ε_i), κ antisymmetric.
        let mut t = Matrix::zeros(n, n);
        for i in 0..n {
            t[(i, i)] = -0.5 * s_mo[(i, i)];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let de = eps[j] - eps[i];
                let kappa = if de.abs() > 1.0e-9 {
                    let ebar = 0.5 * (eps[i] + eps[j]);
                    (f_mo[(i, j)] - ebar * s_mo[(i, j)]) / de
                } else {
                    0.0 // degenerate: gauge-arbitrary rotation, no contribution to P^x here
                };
                t[(i, j)] = kappa - 0.5 * s_mo[(i, j)];
            }
        }
        // D^x_ii = f^x_i − f_i S^x_ii ;
        // D^x_ij = (f_j−f_i)(F^x_ij−ε̄_ij S^x_ij)/(ε_j−ε_i) − ½(f_i+f_j)S^x_ij.
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            d[(i, i)] = f_x[i] - occ[i] * s_mo[(i, i)];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let ebar = 0.5 * (eps[i] + eps[j]);
                let dd = divided_diff(occ[i], occ[j], eps[i], eps[j], self.slope[i], self.slope[j]);
                d[(i, j)] = dd * (f_mo[(i, j)] - ebar * s_mo[(i, j)])
                    - 0.5 * (occ[i] + occ[j]) * s_mo[(i, j)];
            }
        }
        Ok(DensityResponse { d, eps_x, f_x, mu_x, t })
    }

}

struct DensityResponse {
    /// MO-basis density-response coefficient `D^x` (`P^x = C·D^x·C^T`). Kept in MO space so the
    /// shell populations `Tr(P^x W_ish)` can be taken directly (`shell_pop_from_mo_coeff`) without
    /// the O(N³) AO back-transform; the full AO `P^x` is materialized (`ao_from_mo`) only where a
    /// caller genuinely needs it (the Stage-1 FD gate).
    d: Matrix,
    eps_x: Vec<f64>,
    f_x: Vec<f64>,
    #[allow(dead_code)]
    mu_x: f64,
    t: Matrix,
}

// ---------------------------------------------------------------------------------------------
// Sec 3: SCC-CPHF ground-state geometry response (two-channel, finite-T, self-consistent).
// ---------------------------------------------------------------------------------------------


/// Per-shell Mulliken population `pop_ish = Σ_{a∈ish}(P S')_aa` of a density response
/// `P = C·coeff·C^T` (given by its MO coefficient `coeff`), where `S'` is the AO overlap folded
/// into the precomputed `cts = C^T S'` (`S` for the implicit part, `S^x` for the overlap-Pulay
/// part). Computes `(P S')_aa = (C·coeff·cts)_aa` via ONE GEMM `M1 = coeff·cts` plus an O(N²)
/// diagonal gather — WITHOUT materializing the full AO density `C·coeff·C^T`. This replaces the
/// per-DOF `ao_from_mo` (two GEMMs + an N×N allocation) in the all-shell screened-feedback path.
fn shell_pop_from_mo_coeff(
    basis: &crate::basis::BasisSet,
    mos: &Matrix,
    coeff: &Matrix,
    cts: &Matrix,
) -> Result<Vec<f64>> {
    let n = mos.rows();
    let m1 = coeff.matmul(cts)?; // (coeff · C^T S')  — N×N
    let c = mos.as_slice();
    let m1s = m1.as_slice();
    // diag_a = Σ_p C_ap M1_pa.
    let mut diag = vec![0.0_f64; n];
    for a in 0..n {
        let mut acc = 0.0;
        for p in 0..n {
            acc += c[a * n + p] * m1s[p * n + a];
        }
        diag[a] = acc;
    }
    let mut out = vec![0.0_f64; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut acc = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            acc += diag[iao];
        }
        out[ish] = acc;
    }
    Ok(out)
}

/// The `−½ S (v_μ+v_ν)` shell-potential AO dressing (MINUS convention, matching
/// `fock_from_shell_potential`/`shell_potential_dress`). `overlap` may be `S` (feedback Fock)
/// or `S^x` (explicit-overlap derivative of the feedback).
fn shell_potential_dress(
    basis: &crate::basis::BasisSet,
    overlap: &Matrix,
    shell_potential: &[f64],
) -> Matrix {
    let n = basis.len();
    let mut vao = vec![0.0_f64; n];
    for (ish, shell) in basis.shells.iter().enumerate() {
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            vao[iao] = shell_potential[ish];
        }
    }
    let mut out = Matrix::zeros(n, n);
    let os = out.as_mut_slice();
    let ss = overlap.as_slice();
    for i in 0..n {
        for j in 0..n {
            os[i * n + j] = -ss[i * n + j] * 0.5 * (vao[i] + vao[j]);
        }
    }
    out
}

/// Per-channel per-shell feedback potential `v^σ = v_c(dq) ∓ v_s(dm)` from induced
/// `(dq, dm)` — mirrors `analytic_chi`'s feedback exactly (`v_alpha = dv_c − dv_s`,
/// `v_beta = dv_c + dv_s`).
fn channel_feedback_potentials(
    coul_kernel: &Matrix,
    info: &ShellInfo,
    dq: &[f64],
    dm: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let nsh = dq.len();
    let dv_c = matrix_vector_product(coul_kernel, dq)?;
    let dv_s = spin_shell_potential(info, dm);
    let mut va = vec![0.0; nsh];
    let mut vb = vec![0.0; nsh];
    for ish in 0..nsh {
        va[ish] = dv_c[ish] - dv_s[ish];
        vb[ish] = dv_c[ish] + dv_s[ish];
    }
    Ok((va, vb))
}

/// SCC-CPHF ground-state geometry response for ONE Cartesian DOF: solves the coupled
/// two-channel finite-T fixed point `(I − M)(dq, dm)^x = b^x`. Carries everything the
/// χ0/χ derivative chain needs per channel: the density response `P^{σ,x}`, the
/// orbital-energy response `ε^{σ,x}`, the occupation response `f^{σ,x}`, and the rotation
/// generator `T^σ_x = κ^σ_x − ½ S^σ,x_MO` (all around the converged SCC feedback).
struct SccCphfDof {
    /// Per-channel MO-basis density-response coefficients `D^{σ,x}` (`P^{σ,x} = C·D^{σ,x}·C^T`).
    /// Kept in MO space (no eager O(N³) AO materialization); the Stage-1 FD gate reconstructs the
    /// AO `P^{σ,x}` from these when it needs to compare against the FD density.
    #[cfg_attr(not(test), allow(dead_code))]
    dx_a: Matrix,
    #[cfg_attr(not(test), allow(dead_code))]
    dx_b: Matrix,
    eps_x_a: Vec<f64>,
    eps_x_b: Vec<f64>,
    f_x_a: Vec<f64>,
    f_x_b: Vec<f64>,
    t_a: Matrix,
    t_b: Matrix,
    /// Converged ground-state shell-charge geometry response `(dq)^x` (feeds `coul_kernel^x`).
    dq_x: Vec<f64>,
}

fn solve_scc_cphf_dof(
    ctx: &LinearResponseGeomContext,
    cr_a: &ChannelResponse,
    cr_b: &ChannelResponse,
    op: &ScreenedOperator,
    frozen_fock_a: &Matrix,
    frozen_fock_b: &Matrix,
    sx: &Matrix,
) -> Result<SccCphfDof> {
    let basis = &ctx.basis;
    let overlap = &ctx.overlap;
    let nsh = basis.shells.len();
    let dim = 2 * nsh;
    let sx_mo_a = mo_transform(&ctx.ch_a.mos, sx)?;
    let sx_mo_b = mo_transform(&ctx.ch_b.mos, sx)?;
    // C_σ^T S^x, for the overlap-Pulay shell populations Tr(P W^x_ish) — direct-from-MO, no AO S^x.
    let cts_x_a = ctx.ch_a.mos.transpose().matmul(sx)?;
    let cts_x_b = ctx.ch_b.mos.transpose().matmul(sx)?;
    // Explicit-overlap part Tr(P^σ_base W^x_ish): P^σ_base has MO coeff diag(occ).
    let occ_coeff = |ch: &ChannelBasis| -> Matrix {
        let n = ch.mos.rows();
        let mut c = Matrix::zeros(n, n);
        for i in 0..ch.occ.len() {
            c[(i, i)] = ch.occ[i];
        }
        c
    };
    let occ_a = occ_coeff(&ctx.ch_a);
    let occ_b = occ_coeff(&ctx.ch_b);
    let pop_ov_a = shell_pop_from_mo_coeff(basis, &ctx.ch_a.mos, &occ_a, &cts_x_a)?;
    let pop_ov_b = shell_pop_from_mo_coeff(basis, &ctx.ch_b.mos, &occ_b, &cts_x_b)?;

    // The SCC-CPHF ground-state fixed point `w = b_frozen + M w` is LINEAR in `w = [dq;dm]^x`
    // (the feedback response operator `M` is exactly the DOF-independent screened operator). So
    // solve it DIRECTLY: `w^x = (I−M)^{-1} b_frozen`, one matvec — no iteration. `b_frozen` is the
    // induced (dq,dm) from the frozen source alone (w=0), including its S^x overlap-derivative term.
    // Shell populations are taken directly from the MO density-response coefficient `D^σ,x` — the
    // full AO `P^σ,x` is never materialized.
    let induced_from_frozen = |fa: &Matrix, fb: &Matrix| -> Result<Vec<f64>> {
        let f_mo_a = mo_transform(&ctx.ch_a.mos, fa)?;
        let f_mo_b = mo_transform(&ctx.ch_b.mos, fb)?;
        let dr_a = cr_a.density_response(&f_mo_a, &sx_mo_a)?;
        let dr_b = cr_b.density_response(&f_mo_b, &sx_mo_b)?;
        let mut pop_a = shell_pop_from_mo_coeff(basis, &ctx.ch_a.mos, &dr_a.d, &cr_a.cts)?;
        let mut pop_b = shell_pop_from_mo_coeff(basis, &ctx.ch_b.mos, &dr_b.d, &cr_b.cts)?;
        for ish in 0..nsh {
            pop_a[ish] += pop_ov_a[ish];
            pop_b[ish] += pop_ov_b[ish];
        }
        let mut b = vec![0.0; dim];
        for ish in 0..nsh {
            b[ish] = -(pop_a[ish] + pop_b[ish]);
            b[nsh + ish] = pop_a[ish] - pop_b[ish];
        }
        Ok(b)
    };
    let b_frozen = induced_from_frozen(frozen_fock_a, frozen_fock_b)?;
    let w = matvec_vv(&op.inv_i_minus_m, &b_frozen);
    let (dq, dm) = (&w[..nsh], &w[nsh..]);
    // Reconstruct the converged response: full Fock = frozen + feedback(w), one density_response.
    let (va, vb) = channel_feedback_potentials(&ctx.coul_kernel, &ctx.info, dq, dm)?;
    let feedback_a = shell_potential_dress(basis, overlap, &va);
    let feedback_b = shell_potential_dress(basis, overlap, &vb);
    let full_fock_a = matrix_add(frozen_fock_a, &feedback_a);
    let full_fock_b = matrix_add(frozen_fock_b, &feedback_b);
    let f_mo_a = mo_transform(&ctx.ch_a.mos, &full_fock_a)?;
    let f_mo_b = mo_transform(&ctx.ch_b.mos, &full_fock_b)?;
    let dr_a = cr_a.density_response(&f_mo_a, &sx_mo_a)?;
    let dr_b = cr_b.density_response(&f_mo_b, &sx_mo_b)?;
    Ok(SccCphfDof {
        dx_a: dr_a.d,
        dx_b: dr_b.d,
        eps_x_a: dr_a.eps_x,
        eps_x_b: dr_b.eps_x,
        f_x_a: dr_a.f_x,
        f_x_b: dr_b.f_x,
        t_a: dr_a.t,
        t_b: dr_b.t,
        dq_x: dq.to_vec(),
    })
}

/// AO base density of a channel: `P = C f C^T`. Only the Stage-1 FD gate materializes it now.
#[cfg_attr(not(test), allow(dead_code))]
fn channel_density(ch: &ChannelBasis) -> Matrix {
    let n = ch.mos.rows();
    let mut coeff = Matrix::zeros(n, n);
    for i in 0..ch.occ.len() {
        coeff[(i, i)] = ch.occ[i];
    }
    ao_from_mo(&ch.mos, &coeff).expect("channel density C f C^T")
}

fn matrix_add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    let os = out.as_mut_slice();
    let bs = b.as_slice();
    for (o, x) in os.iter_mut().zip(bs.iter()) {
        *o += *x;
    }
    out
}

/// Build the frozen source `(F^σ_fr)^x` and `S^x` for every Cartesian DOF (MINUS convention):
/// `(F^σ_fr)^x = (h0)^x − ½ S^x(v^σ+v^σ) − ½ S(a^x_μ+a^x_ν)`.
///
/// The expensive integral-derivative routine (`cartesian_ao_derivative_matrices_raw`, which loops
/// over all AO pairs computing `contracted_pair_with_derivatives`) is **spin-independent** — only
/// the `−½ S^x(v^σ+v^σ)` shell-potential term differs between the α/β channels. So it is called
/// **ONCE** (with a zero SCC potential → the skeleton `(h0)^x − ½ S(a^x+a^x)` plus `S^x`), and each
/// channel's frozen Fock is completed by adding the cheap O(N²)-per-DOF `−½ S^x(v^σ+v^σ)` term
/// (`shell_potential_dress` on `S^x`). This halves the integral-derivative work.
fn frozen_sources(
    ctx: &LinearResponseGeomContext,
    system: &crate::system::PeriodicSystem,
    params: &crate::params::Gfn1Parameters,
) -> Result<Vec<FrozenSource>> {
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let nsh = ctx.basis.shells.len();
    let a_x = shell_scalar_potential_derivatives(system, &ctx.basis, params, &ctx.q0)?;
    let cn_derivs = coordination_number_derivatives(
        system,
        crate::coordination::CoordinationOptions::default().cutoff,
    )?;
    // ONE integral-derivative pass with zero SCC potential → skeleton = (h0)^x − ½ S(a^x+a^x),
    // and S^x. (`add_scalar_derivative_matrices` still folds in the `a^x` term via `shell_scalar_derivatives`.)
    let zero_v = vec![0.0_f64; nsh];
    let skel = cartesian_ao_derivative_matrices_raw(
        system, params, &ctx.basis, &ctx.coordination_numbers, &zero_v, &a_x, Some(&cn_derivs),
    )?;
    let mut out = Vec::with_capacity(ndim);
    for c in 0..ndim {
        let sx = &skel[c].overlap_deriv;
        // −½ S^x (v^σ_μ + v^σ_ν) per channel (cheap, O(N²)).
        let dress_a = shell_potential_dress(&ctx.basis, sx, &ctx.v_alpha0);
        let dress_b = shell_potential_dress(&ctx.basis, sx, &ctx.v_beta0);
        out.push(FrozenSource {
            frozen_fock_a: matrix_add(&skel[c].h0_deriv, &dress_a),
            frozen_fock_b: matrix_add(&skel[c].h0_deriv, &dress_b),
            sx: sx.clone(),
        });
    }
    Ok(out)
}

struct FrozenSource {
    frozen_fock_a: Matrix,
    frozen_fock_b: Matrix,
    sx: Matrix,
}

// ---------------------------------------------------------------------------------------------
// Sec 4: dχ0/dR — geometry derivative of the bare occupation-response matrix.
// ---------------------------------------------------------------------------------------------

/// On-site probe / projector `W_A = ½(E_A M + M E_A)` for the correlated AOs of atom `A`,
/// where `M` is `S` (probe/projector) or `S^x` (its geometry derivative `W^x_A`). Matches
/// `spin::onsite_shift_fock(M, aos, 1)`.
fn onsite_selector_dress(m: &Matrix, aos: &[usize]) -> Matrix {
    let n = m.rows();
    let mut out = Matrix::zeros(n, n);
    // W = ½(E M + M E); E is the diagonal selector on `aos`. (E M)_{ab} = [a∈aos] M_{ab},
    // (M E)_{ab} = M_{ab} [b∈aos].
    let mut is_sel = vec![false; n];
    for &a in aos {
        is_sel[a] = true;
    }
    let ms = m.as_slice();
    let os = out.as_mut_slice();
    for i in 0..n {
        for j in 0..n {
            let mut v = 0.0;
            if is_sel[i] {
                v += ms[i * n + j];
            }
            if is_sel[j] {
                v += ms[i * n + j];
            }
            os[i * n + j] = 0.5 * v;
        }
    }
    out
}

/// `Tr(A B)` for square matrices (row-major).
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.rows();
    let as_ = a.as_slice();
    let bs = b.as_slice();
    let mut acc = 0.0;
    for i in 0..n {
        for k in 0..n {
            acc += as_[i * n + k] * bs[k * n + i];
        }
    }
    acc
}

/// Per-channel base bare-response bundle for a probe `Y` (AO) plus everything the geometry
/// derivative needs: `L = C^T Y C`, the coefficient matrix `D` (bare response), and the
/// intermediate scalars (`μ`, occupation response `f0`). Used for both `χ0` (Y = W_J) and,
/// in Sec 5, the screened probe.
struct BareBundle {
    l: Matrix,
    d: Matrix,
    /// Fermi-shift `μ = (Σ g_i L_ii)/(Σ g_i)`.
    mu: f64,
    /// `P = C D C^T` (AO density response to `Y`).
    p: Matrix,
}

/// Bare-response MO coefficients `(L = C^T y C, D, μ)` for channel `cr`, WITHOUT the AO
/// back-transform. `D` is the Sec-3 bare response coefficient; `P = C·D·C^T` (only formed when a
/// caller needs the full AO density — see [`bare_bundle`]).
fn bare_coeff(cr: &ChannelResponse, y: &Matrix) -> Result<(Matrix, Matrix, f64)> {
    let n = cr.norb();
    let l = mo_transform(&cr.ch.mos, y)?;
    let eps = &cr.ch.eps;
    let occ = &cr.ch.occ;
    let mu = if cr.kt > 0.0 && cr.slope_sum.abs() > 1.0e-30 {
        cr.slope.iter().zip((0..n).map(|i| l[(i, i)])).map(|(&g, lii)| g * lii).sum::<f64>()
            / cr.slope_sum
    } else {
        0.0
    };
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = if cr.kt > 0.0 { cr.slope[i] * (l[(i, i)] - mu) } else { 0.0 };
        for j in 0..n {
            if i == j {
                continue;
            }
            let dd = divided_diff(occ[i], occ[j], eps[i], eps[j], cr.slope[i], cr.slope[j]);
            d[(i, j)] = dd * l[(i, j)];
        }
    }
    Ok((l, d, mu))
}

/// Build the base bare response of channel `cr` to AO perturbation `y`, INCLUDING the full AO
/// density `P = C·D·C^T`. Use [`bare_coeff`] when only shell populations / correlated traces are
/// needed (avoids the O(N³) back-transform).
fn bare_bundle(cr: &ChannelResponse, y: &Matrix) -> Result<BareBundle> {
    let (l, d, mu) = bare_coeff(cr, y)?;
    let p = ao_from_mo(&cr.ch.mos, &d)?;
    Ok(BareBundle { l, d, mu, p })
}

/// The MO-basis inner coefficient `inner = T D + D^x + D T^T` of the bare-response derivative,
/// such that `(P^Y)^x = C · inner · C^T`. Building this is O(N²)–O(N³) in MO matmuls but does NOT
/// back-transform to the AO basis; callers either materialize the AO matrix (`bare_bundle_deriv`,
/// for the all-shell screened feedback) or contract it directly against MO-basis projectors
/// (`bare_deriv_traces`, for the correlated χ0/χ traces — O(|corr|·N²), no AO materialization).
#[allow(clippy::too_many_arguments)]
fn bare_deriv_inner(
    cr: &ChannelResponse,
    base_l: &Matrix,
    base_d: &Matrix,
    base_mu: f64,
    y_x: &Matrix,
    t: &Matrix,
    eps_x: &[f64],
    f_x: &[f64],
) -> Result<Matrix> {
    let n = cr.norb();
    let eps = &cr.ch.eps;
    let occ = &cr.ch.occ;
    let g = &cr.slope;
    // L^x = T^T L + C^T y^x C + L T. Since L (and yx_mo) are symmetric and L·T = (T^T·L)^T,
    // this equals sym(T^T·L) + yx_mo — ONE matmul instead of two (T^T L and L T).
    let yx_mo = mo_transform(&cr.ch.mos, y_x)?;
    let tt_l = t.transpose().matmul(base_l)?;
    let mut lx = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            lx[(i, j)] = tt_l[(i, j)] + tt_l[(j, i)] + yx_mo[(i, j)];
        }
    }
    // g^x_i = −β(1−2f_i) f^x_i  (β = 1/kt).
    let beta = if cr.kt > 0.0 { 1.0 / cr.kt } else { 0.0 };
    let gx: Vec<f64> = (0..n).map(|i| -beta * (1.0 - 2.0 * occ[i]) * f_x[i]).collect();
    // μ^x = (Σ(g^x_i L_ii + g_i L^x_ii) − μ Σ g^x_i)/(Σ g_i).
    let mu_x = if cr.kt > 0.0 && cr.slope_sum.abs() > 1.0e-30 {
        let numer: f64 = (0..n).map(|i| gx[i] * base_l[(i, i)] + g[i] * lx[(i, i)]).sum();
        let dn_x: f64 = gx.iter().sum();
        (numer - base_mu * dn_x) / cr.slope_sum
    } else {
        0.0
    };
    // D^x.
    let mut dx = Matrix::zeros(n, n);
    for i in 0..n {
        dx[(i, i)] = if cr.kt > 0.0 {
            gx[i] * (base_l[(i, i)] - base_mu) + g[i] * (lx[(i, i)] - mu_x)
        } else {
            0.0
        };
        for j in 0..n {
            if i == j {
                continue;
            }
            let de = eps[j] - eps[i];
            let dd = divided_diff(occ[i], occ[j], eps[i], eps[j], g[i], g[j]);
            let ddx = if de.abs() > 1.0e-9 {
                ((f_x[j] - f_x[i]) - dd * (eps_x[j] - eps_x[i])) / de
            } else {
                0.5 * (gx[i] + gx[j])
            };
            dx[(i, j)] = ddx * base_l[(i, j)] + dd * lx[(i, j)];
        }
    }
    // inner = T D + D^x + D T^T. Since D is symmetric and D·T^T = (T·D)^T, the two flanking
    // products are sym(T·D) — ONE matmul instead of two.
    let t_d = t.matmul(base_d)?;
    let mut inner = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inner[(i, j)] = t_d[(i, j)] + t_d[(j, i)] + dx[(i, j)];
        }
    }
    Ok(inner)
}

/// Correlated-block traces `[Tr((P^Y)^x W_I)]` of the bare-response derivative, for each correlated
/// atom I, WITHOUT materializing the full AO `(P^Y)^x`. Uses `Tr(C inner C^T W_I) = Tr(inner · C^T
/// W_I C)` with the precomputed MO projectors `wmo_i[I] = C^T W_I C`. O(|corr|·N²) instead of the
/// O(N³) AO back-transform + O(ncorr·N²) traces.
#[allow(clippy::too_many_arguments)]
fn bare_deriv_traces(
    cr: &ChannelResponse,
    base: &BareBundle,
    y_x: &Matrix,
    t: &Matrix,
    eps_x: &[f64],
    f_x: &[f64],
    wmo_i: &[Matrix],
) -> Result<Vec<f64>> {
    let inner = bare_deriv_inner(cr, &base.l, &base.d, base.mu, y_x, t, eps_x, f_x)?;
    Ok(wmo_i.iter().map(|w| trace_product(&inner, w)).collect())
}

/// All geometry-DOF-**independent** response data, built ONCE and reused across the 3N-DOF
/// loop (levers 1/3/4): the on-site projectors `Q_I = W_I`; per correlated atom `J` the base
/// bare probe bundles (`ba_j/bb_j`, → χ0) and the base screened solution (`u_j` and the screened
/// total bundles `sba_j/sbb_j`, → χ); and the base `χ0`/`χ` themselves. The per-DOF functions
/// consume this and touch only the geometry-derivative pieces.
struct PreparedResponse {
    /// Base bare bundles to the probe `Q_J = W_J` (α, β), per J — the χ0 building blocks.
    ba_j: Vec<BareBundle>,
    bb_j: Vec<BareBundle>,
    /// Base screened SCC variable `u_J = (I−M)^{-1} b_J` and induced `b_J`, per J.
    u_j: Vec<Vec<f64>>,
    /// Base screened total bundles (α, β) `R^σ[Y^σ_J]`, per J — the χ building blocks.
    sba_j: Vec<BareBundle>,
    sbb_j: Vec<BareBundle>,
    /// Base χ0, χ (`ncorr×ncorr`).
    chi0: Vec<Vec<f64>>,
    chi: Vec<Vec<f64>>,
    /// Precomputed `(∂A/∂x)·dq(u_J)` for ALL DOFs, per J: `amat_deriv_uj[j][dof][shell]`. Computed
    /// once (`shell_scalar_potential_derivatives` returns all DOFs at once) instead of recomputing
    /// the full `ndim×nsh` table inside every DOF iteration (avoids the O(ndim²) redundancy).
    amat_deriv_uj: Vec<Vec<Vec<f64>>>,
    /// MO-basis correlated projectors `C^T W_I C` per channel, one per correlated atom I. These let
    /// the χ0/χ derivative traces `Tr((P^Y)^x W_I) = Tr(inner · C^T W_I C)` be evaluated by an
    /// O(|corr|·N²) MO contraction — WITHOUT materializing the full AO `(P^Y)^x` (avoids the
    /// per-DOF O(ncorr·N³) back-transform in the trace path).
    wmo_a: Vec<Matrix>,
    wmo_b: Vec<Matrix>,
    /// Feedback base bare coefficients `(L, D, μ)[feedback(u_J)]` per J per channel (= screened −
    /// bare, by linearity of `D` in the perturbation). Lets the per-DOF `M^x u_J` skip the two
    /// `bare_coeff` MO transforms (they are DOF-independent).
    fb_a: Vec<(Matrix, Matrix, f64)>,
    fb_b: Vec<(Matrix, Matrix, f64)>,
}

fn prepare_response(
    ctx: &LinearResponseGeomContext,
    block: &ScreenedBlock,
    coul: &CoulKernel,
    op: &ScreenedOperator,
    system: &crate::system::PeriodicSystem,
    params: &crate::params::Gfn1Parameters,
) -> Result<PreparedResponse> {
    let subspace = &ctx.subspace;
    let ncorr = subspace.len();
    let nsh = ctx.basis.shells.len();
    let s = &ctx.overlap;
    let q_i: Vec<Matrix> = subspace.iter().map(|a| onsite_selector_dress(s, &a.aos)).collect();
    // MO-basis correlated projectors C^T W_I C (per channel), for the O(|corr|·N²) trace path.
    let mut wmo_a = Vec::with_capacity(ncorr);
    let mut wmo_b = Vec::with_capacity(ncorr);
    for qi in &q_i {
        wmo_a.push(mo_transform(&ctx.ch_a.mos, qi)?);
        wmo_b.push(mo_transform(&ctx.ch_b.mos, qi)?);
    }
    let mut ba_j = Vec::with_capacity(ncorr);
    let mut bb_j = Vec::with_capacity(ncorr);
    let mut u_j = Vec::with_capacity(ncorr);
    let mut sba_j = Vec::with_capacity(ncorr);
    let mut sbb_j = Vec::with_capacity(ncorr);
    let mut fb_a = Vec::with_capacity(ncorr);
    let mut fb_b = Vec::with_capacity(ncorr);
    let mut amat_deriv_uj = Vec::with_capacity(ncorr);
    let mut chi0 = vec![vec![0.0; ncorr]; ncorr];
    let mut chi = vec![vec![0.0; ncorr]; ncorr];
    // `D` is linear in the perturbation `Y`, so the feedback base coefficients
    // (l,d,mu)[feedback(u_J)] = (l,d,mu)[Q_J + feedback] − (l,d,mu)[Q_J] = screened − bare.
    let sub = |sc: &Matrix, ba: &Matrix| -> Matrix {
        let mut o = sc.clone();
        let os = o.as_mut_slice();
        let bs = ba.as_slice();
        for k in 0..os.len() {
            os[k] -= bs[k];
        }
        o
    };
    for (j, _atom_j) in subspace.iter().enumerate() {
        let q_j = &q_i[j]; // probe == projector
        // Bare (χ0) bundles.
        let ba = bare_bundle(block.cr_a, q_j)?;
        let bb = bare_bundle(block.cr_b, q_j)?;
        for i in 0..ncorr {
            chi0[i][j] = trace_product(&ba.p, &q_i[i]) + trace_product(&bb.p, &q_i[i]);
        }
        // Screened (χ) solve: b_J = R[(Q_J,Q_J)]; u_J = (I−M)^{-1} b_J.
        let b_j = block.induced(q_j, q_j)?;
        let u = matvec_vv(&op.inv_i_minus_m, &b_j);
        // Total screened perturbation Y^σ_J = Q_J + Dress_S(feedback(u_J)_σ).
        let (fya, fyb) = feedback_ao(ctx, coul, &u)?;
        let y_a = matrix_add(q_j, &fya);
        let y_b = matrix_add(q_j, &fyb);
        let sba = bare_bundle(block.cr_a, &y_a)?;
        let sbb = bare_bundle(block.cr_b, &y_b)?;
        for i in 0..ncorr {
            chi[i][j] = trace_product(&sba.p, &q_i[i]) + trace_product(&sbb.p, &q_i[i]);
        }
        // Feedback base coeffs (l,d,mu) = screened − bare (avoids re-`bare_coeff` per DOF).
        fb_a.push((sub(&sba.l, &ba.l), sub(&sba.d, &ba.d), sba.mu - ba.mu));
        fb_b.push((sub(&sbb.l, &bb.l), sub(&sbb.d, &bb.d), sbb.mu - bb.mu));
        // Precompute `(∂A/∂x)·dq(u_J)` for all DOFs at once (dq = u[..nsh]).
        amat_deriv_uj.push(shell_scalar_potential_derivatives(system, &ctx.basis, params, &u[..nsh])?);
        ba_j.push(ba);
        bb_j.push(bb);
        u_j.push(u);
        sba_j.push(sba);
        sbb_j.push(sbb);
    }
    let _ = q_i;
    Ok(PreparedResponse { ba_j, bb_j, u_j, sba_j, sbb_j, chi0, chi, amat_deriv_uj, wmo_a, wmo_b, fb_a, fb_b })
}

/// χ0^x for ONE Cartesian DOF, reusing the DOF-independent [`PreparedResponse`]. Only the
/// probe/projector derivatives (`Q^x_J`, `Q^x_I` on `S^x`) and the bare-response derivatives
/// (from the ground-state `dof`) are computed here.
fn chi0_deriv_prepared(
    ctx: &LinearResponseGeomContext,
    cr_a: &ChannelResponse,
    cr_b: &ChannelResponse,
    prep: &PreparedResponse,
    sx: &Matrix,
    dof: &SccCphfDof,
) -> Result<Vec<Vec<f64>>> {
    let subspace = &ctx.subspace;
    let ncorr = subspace.len();
    let qx_i: Vec<Matrix> = subspace.iter().map(|a| onsite_selector_dress(sx, &a.aos)).collect();
    let mut chi0_x = vec![vec![0.0; ncorr]; ncorr];
    for j in 0..ncorr {
        let probe_x = &qx_i[j]; // probe == projector
        // Correlated-block traces Tr((P^Y)^x W_I) via the MO projectors — no full AO materialization.
        let tr_a = bare_deriv_traces(cr_a, &prep.ba_j[j], probe_x, &dof.t_a, &dof.eps_x_a, &dof.f_x_a, &prep.wmo_a)?;
        let tr_b = bare_deriv_traces(cr_b, &prep.bb_j[j], probe_x, &dof.t_b, &dof.eps_x_b, &dof.f_x_b, &prep.wmo_b)?;
        for i in 0..ncorr {
            chi0_x[i][j] = tr_a[i]
                + tr_b[i]
                + trace_product(&prep.ba_j[j].p, &qx_i[i])
                + trace_product(&prep.bb_j[j].p, &qx_i[i]);
        }
    }
    Ok(chi0_x)
}

/// χ0 and its geometry derivative for ONE Cartesian DOF (standalone; used by the FD-gate tests).
/// Production goes through [`prepare_response`] + [`chi0_deriv_prepared`] to avoid recomputing
/// the DOF-independent base bundles.
#[cfg_attr(not(test), allow(dead_code))]
fn chi0_and_deriv_dof(
    ctx: &LinearResponseGeomContext,
    cr_a: &ChannelResponse,
    cr_b: &ChannelResponse,
    sx: &Matrix,
    dof: &SccCphfDof,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let subspace = &ctx.subspace;
    let ncorr = subspace.len();
    let s = &ctx.overlap;
    let q_i: Vec<Matrix> = subspace.iter().map(|a| onsite_selector_dress(s, &a.aos)).collect();
    let mut wmo_a = Vec::with_capacity(ncorr);
    let mut wmo_b = Vec::with_capacity(ncorr);
    for qi in &q_i {
        wmo_a.push(mo_transform(&ctx.ch_a.mos, qi)?);
        wmo_b.push(mo_transform(&ctx.ch_b.mos, qi)?);
    }
    let mut ba_j = Vec::with_capacity(ncorr);
    let mut bb_j = Vec::with_capacity(ncorr);
    let mut chi0 = vec![vec![0.0; ncorr]; ncorr];
    for j in 0..ncorr {
        let ba = bare_bundle(cr_a, &q_i[j])?;
        let bb = bare_bundle(cr_b, &q_i[j])?;
        for i in 0..ncorr {
            chi0[i][j] = trace_product(&ba.p, &q_i[i]) + trace_product(&bb.p, &q_i[i]);
        }
        ba_j.push(ba);
        bb_j.push(bb);
    }
    let _ = &q_i;
    let prep = PreparedResponse {
        ba_j,
        bb_j,
        u_j: Vec::new(),
        sba_j: Vec::new(),
        sbb_j: Vec::new(),
        chi0: chi0.clone(),
        chi: Vec::new(),
        amat_deriv_uj: Vec::new(),
        wmo_a,
        wmo_b,
        fb_a: Vec::new(),
        fb_b: Vec::new(),
    };
    let chi0_x = chi0_deriv_prepared(ctx, cr_a, cr_b, &prep, sx, dof)?;
    Ok((chi0, chi0_x))
}

/// Base SCC feedback response operator `M` (2·nsh × 2·nsh) and its inverse factor
/// `(I − M)^{-1}`, built once at the base state (DOF-independent).
struct ScreenedOperator {
    /// `(I − M)^{-1}` as row-major `Vec<Vec<f64>>` (via `plus_u::invert_small`).
    inv_i_minus_m: Vec<Vec<f64>>,
}

fn build_screened_operator(
    ctx: &LinearResponseGeomContext,
    block: &ScreenedBlock,
    coul: &CoulKernel,
) -> Result<ScreenedOperator> {
    let nsh = ctx.basis.shells.len();
    let dim = 2 * nsh;
    // M column c = R[Dress_S(feedback(e_c))].
    let mut m = vec![vec![0.0; dim]; dim];
    for c in 0..dim {
        let mut e = vec![0.0; dim];
        e[c] = 1.0;
        let (ya, yb) = feedback_ao(ctx, coul, &e)?;
        let col = block.induced(&ya, &yb)?;
        for r in 0..dim {
            m[r][c] = col[r];
        }
    }
    // I − M.
    let mut i_minus_m = vec![vec![0.0; dim]; dim];
    for r in 0..dim {
        for c in 0..dim {
            i_minus_m[r][c] = if r == c { 1.0 } else { 0.0 } - m[r][c];
        }
    }
    let inv = crate::plus_u::invert_small(&i_minus_m)
        .ok_or_else(|| crate::error::Gfn1Error::InvalidInput("(I−M) singular".to_string()))?;
    Ok(ScreenedOperator { inv_i_minus_m: inv })
}

fn matvec_vv(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(m, v)| m * v).sum()).collect()
}

/// χ^x for ONE Cartesian DOF, reusing the DOF-independent [`PreparedResponse`] (base `u_J`,
/// screened bundles) and the base `(I−M)^{-1}` factor. Only the geometry-derivative sources
/// (`Q^x`, `S^x`, `coul_kernel^x`, `b^x`, `M^x u`) and the screened-response derivatives are
/// computed here.
#[allow(clippy::too_many_arguments)]
fn chi_deriv_prepared(
    ctx: &LinearResponseGeomContext,
    block: &ScreenedBlock,
    coul: &CoulKernel,
    op: &ScreenedOperator,
    prep: &PreparedResponse,
    sx: &Matrix,
    dof: &SccCphfDof,
    dof_index: usize,
) -> Result<Vec<Vec<f64>>> {
    let subspace = &ctx.subspace;
    let ncorr = subspace.len();
    let nsh = ctx.basis.shells.len();
    let dim = 2 * nsh;
    let qx_i: Vec<Matrix> = subspace.iter().map(|a| onsite_selector_dress(sx, &a.aos)).collect();
    let q0_atom_x = ctx.shell_model.atomic_charges(&ctx.basis, &dof.dq_x);
    // C_σ^T S^x, precomputed once per DOF (shared by every J's overlap-Pulay population term).
    let cts_x_a = block.cr_a.ch.mos.transpose().matmul(sx)?;
    let cts_x_b = block.cr_b.ch.mos.transpose().matmul(sx)?;

    let mut chi_x = vec![vec![0.0; ncorr]; ncorr];
    for j in 0..ncorr {
        let qx_j = &qx_i[j]; // probe == projector
        let u_j = &prep.u_j[j];
        // b^x_J = Ṙ[(Q_J,Q_J),(Q^x_J,Q^x_J)] — probe base coeffs (l,d,mu) are the prepared χ0
        // bundles `ba_j`/`bb_j` (probe == Q_J); only y^x = Q^x_J is per-DOF.
        let (ba, bb) = (&prep.ba_j[j], &prep.bb_j[j]);
        let bx_j = block.induced_deriv_from_base(
            &ba.l, &ba.d, ba.mu, qx_j,
            &bb.l, &bb.d, bb.mu, qx_j,
            dof, &cts_x_a, &cts_x_b,
        )?;
        // `(∂A/∂x)·dq(u_J)` for this DOF — precomputed once per J in `prepare_response`.
        let amat_x_dq_uj = &prep.amat_deriv_uj[j][dof_index];
        // M^x u_J (hold u_J fixed: w^x = 0). Feedback base coeffs are prepared (`fb_a`/`fb_b`);
        // only y^x = feedback^x(u_J) is per-DOF.
        let fb_pert_uj = feedback_pert(ctx, coul, u_j, &vec![0.0; dim], sx, amat_x_dq_uj, &q0_atom_x)?;
        let (fla, fda, fmua) = &prep.fb_a[j];
        let (flb, fdb, fmub) = &prep.fb_b[j];
        let mx_uj = block.induced_deriv_from_base(
            fla, fda, *fmua, &fb_pert_uj.yx_a,
            flb, fdb, *fmub, &fb_pert_uj.yx_b,
            dof, &cts_x_a, &cts_x_b,
        )?;
        // (I−M) u^x_J = b^x_J + M^x u_J.
        let mut rhs = vec![0.0; dim];
        for r in 0..dim {
            rhs[r] = bx_j[r] + mx_uj[r];
        }
        let ux_j = matvec_vv(&op.inv_i_minus_m, &rhs);
        // Y^{σ,x}_J = Q^x_J + Dress derivative of feedback(u_J).
        let fb_full = feedback_pert(ctx, coul, u_j, &ux_j, sx, &amat_x_dq_uj, &q0_atom_x)?;
        let mut yx_a = fb_full.yx_a;
        let mut yx_b = fb_full.yx_b;
        for k in 0..yx_a.as_slice().len() {
            yx_a.as_mut_slice()[k] += qx_j.as_slice()[k];
            yx_b.as_mut_slice()[k] += qx_j.as_slice()[k];
        }
        // (δP^σ_J)^x correlated-block traces via MO projectors — no full AO materialization.
        let tr_a = bare_deriv_traces(block.cr_a, &prep.sba_j[j], &yx_a, &dof.t_a, &dof.eps_x_a, &dof.f_x_a, &prep.wmo_a)?;
        let tr_b = bare_deriv_traces(block.cr_b, &prep.sbb_j[j], &yx_b, &dof.t_b, &dof.eps_x_b, &dof.f_x_b, &prep.wmo_b)?;
        for i in 0..ncorr {
            chi_x[i][j] = tr_a[i]
                + tr_b[i]
                + trace_product(&prep.sba_j[j].p, &qx_i[i])
                + trace_product(&prep.sbb_j[j].p, &qx_i[i]);
        }
    }
    Ok(chi_x)
}

/// χ and χ^x for ONE Cartesian DOF (standalone; used by the FD-gate tests). Production goes
/// through [`prepare_response`] + [`chi_deriv_prepared`].
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn chi_and_deriv_dof(
    ctx: &LinearResponseGeomContext,
    block: &ScreenedBlock,
    coul: &CoulKernel,
    op: &ScreenedOperator,
    sx: &Matrix,
    dof: &SccCphfDof,
    system: &crate::system::PeriodicSystem,
    params: &crate::params::Gfn1Parameters,
    dof_index: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let prep = prepare_response(ctx, block, coul, op, system, params)?;
    let chi_x = chi_deriv_prepared(ctx, block, coul, op, &prep, sx, dof, dof_index)?;
    Ok((prep.chi, chi_x))
}

// ---------------------------------------------------------------------------------------------
// Sec 5: dχ/dR — screened occupation-response matrix and its geometry derivative.
//
// Reformulated in the code's (dq, dm) shell-variable basis (dimension 2·nsh), where the SCC
// feedback response operator M is `w → induced (dq', dm')` and the screened probe response
// solves `(I − M) u_J = b_J`. This reproduces `spin::analytic_chi` and its derivative follows
// `(I − M) u^x_J = b^x_J + M^x u_J`.
// ---------------------------------------------------------------------------------------------

/// The `atom-block` induced-potential kernel action `coul_kernel · w_dq` (the SCC Coulomb
/// response) and, when `deriv` inputs are supplied, its geometry derivative for a FIXED `w`:
/// `coul_kernel^x · w_dq = (∂A/∂x)·w_dq + Σ_shells 2Γ_atom (q0_atom)^x w_dq(same-atom shells sum)`.
struct CoulKernel<'a> {
    ctx: &'a LinearResponseGeomContext,
}

impl<'a> CoulKernel<'a> {
    /// `coul_kernel · dq` (value).
    fn apply(&self, dq: &[f64]) -> Result<Vec<f64>> {
        matrix_vector_product(&self.ctx.coul_kernel, dq)
    }

    /// `coul_kernel^x · dq` for a FIXED `dq`, DOF `c`. `amat_x_dq` = `(∂A/∂x)·dq` for this DOF
    /// (from `shell_scalar_potential_derivatives` called with `dq`), `q0_atom_x` = base atomic
    /// charge geometry response for this DOF.
    fn apply_deriv(
        &self,
        dq: &[f64],
        amat_x_dq: &[f64],
        q0_atom_x: &[f64],
    ) -> Vec<f64> {
        let model = &self.ctx.shell_model;
        let mut out = amat_x_dq.to_vec();
        // On-site 2Γ q0_atom term derivative: block over each atom's shells.
        let nat = model.atom_offsets.len();
        for atom in 0..nat {
            let count = model.atom_shell_counts[atom];
            if count == 0 {
                continue;
            }
            let offset = model.atom_offsets[atom];
            let add = 2.0 * q0_atom_x[atom] * model.hubbard_derivs[offset];
            // (block of ones) · dq over this atom's shells.
            let mut sum_dq = 0.0;
            for lj in 0..count {
                sum_dq += dq[offset + lj];
            }
            for li in 0..count {
                out[offset + li] += add * sum_dq;
            }
        }
        out
    }
}

/// Per-channel AO perturbation geometry derivative `(Y^{σ,x})` — the feedback-derivative input to
/// the screened-response building block. (The value part `Y^σ` is DOF-independent and handled via
/// the prepared base coefficients, so only the derivative is carried here.)
struct AoPert {
    yx_a: Matrix,
    yx_b: Matrix,
}

/// Induced `(dq, dm)` (length `2·nsh`, `[dq; dm]`) from a per-channel AO perturbation, plus —
/// when the DOF response is supplied — its geometry derivative. This is the response operator
/// `R` (and `Ṙ`) projected to the SCC shell variables.
struct ScreenedBlock<'a> {
    ctx: &'a LinearResponseGeomContext,
    cr_a: &'a ChannelResponse<'a>,
    cr_b: &'a ChannelResponse<'a>,
}

impl<'a> ScreenedBlock<'a> {
    /// Value only: `w = R[(Y_a, Y_b)]` = induced (dq, dm). Uses the direct shell-population-from-MO
    /// path (`shell_pop_from_mo_coeff`) — NO full AO density is materialized.
    fn induced(&self, y_a: &Matrix, y_b: &Matrix) -> Result<Vec<f64>> {
        let basis = &self.ctx.basis;
        let nsh = basis.shells.len();
        let (_, da, _) = bare_coeff(self.cr_a, y_a)?;
        let (_, db, _) = bare_coeff(self.cr_b, y_b)?;
        let pop_a = shell_pop_from_mo_coeff(basis, &self.cr_a.ch.mos, &da, &self.cr_a.cts)?;
        let pop_b = shell_pop_from_mo_coeff(basis, &self.cr_b.ch.mos, &db, &self.cr_b.cts)?;
        let mut w = vec![0.0; 2 * nsh];
        for ish in 0..nsh {
            w[ish] = -(pop_a[ish] + pop_b[ish]); // dq
            w[nsh + ish] = pop_a[ish] - pop_b[ish]; // dm
        }
        Ok(w)
    }

    /// Derivative-only induced `w^x`, given the DOF-INDEPENDENT base bare coefficients
    /// `(l, d, mu)_σ` of the perturbation `Y^σ` (precomputed once — the response coefficient `D` is
    /// linear in `Y`, so the base `Y` value part is DOF-independent and its induced `w` need not be
    /// recomputed here) and the per-DOF derivative pieces `y_x`, `cts_x`. Skips the value-`w`
    /// shell-pops and the two `bare_coeff` MO transforms of `induced_and_deriv`.
    #[allow(clippy::too_many_arguments)]
    fn induced_deriv_from_base(
        &self,
        la: &Matrix, da: &Matrix, mua: f64, yx_a: &Matrix,
        lb: &Matrix, db: &Matrix, mub: f64, yx_b: &Matrix,
        dof: &SccCphfDof,
        cts_x_a: &Matrix,
        cts_x_b: &Matrix,
    ) -> Result<Vec<f64>> {
        let basis = &self.ctx.basis;
        let nsh = basis.shells.len();
        let inner_a = bare_deriv_inner(self.cr_a, la, da, mua, yx_a, &dof.t_a, &dof.eps_x_a, &dof.f_x_a)?;
        let inner_b = bare_deriv_inner(self.cr_b, lb, db, mub, yx_b, &dof.t_b, &dof.eps_x_b, &dof.f_x_b)?;
        let popx_a = shell_pop_from_mo_coeff(basis, &self.cr_a.ch.mos, &inner_a, &self.cr_a.cts)?;
        let popx_b = shell_pop_from_mo_coeff(basis, &self.cr_b.ch.mos, &inner_b, &self.cr_b.cts)?;
        let pov_a = shell_pop_from_mo_coeff(basis, &self.cr_a.ch.mos, da, cts_x_a)?;
        let pov_b = shell_pop_from_mo_coeff(basis, &self.cr_b.ch.mos, db, cts_x_b)?;
        let mut wx = vec![0.0; 2 * nsh];
        for ish in 0..nsh {
            let pxa = popx_a[ish] + pov_a[ish];
            let pxb = popx_b[ish] + pov_b[ish];
            wx[ish] = -(pxa + pxb);
            wx[nsh + ish] = pxa - pxb;
        }
        Ok(wx)
    }
}

/// Build the per-channel feedback AO perturbation `Dress_S(feedback(w)_σ)` from an SCC variable
/// `w = [dq; dm]` (value only): `v^σ = coul_kernel·dq ∓ v_s(dm)`, dressed `−½S(v^σ+v^σ)`.
fn feedback_ao(
    ctx: &LinearResponseGeomContext,
    coul: &CoulKernel,
    w: &[f64],
) -> Result<(Matrix, Matrix)> {
    let nsh = ctx.basis.shells.len();
    let dq = &w[..nsh];
    let dm = &w[nsh..];
    let dv_c = coul.apply(dq)?;
    let dv_s = spin_shell_potential(&ctx.info, dm);
    let mut va = vec![0.0; nsh];
    let mut vb = vec![0.0; nsh];
    for ish in 0..nsh {
        va[ish] = dv_c[ish] - dv_s[ish];
        vb[ish] = dv_c[ish] + dv_s[ish];
    }
    Ok((
        shell_potential_dress(&ctx.basis, &ctx.overlap, &va),
        shell_potential_dress(&ctx.basis, &ctx.overlap, &vb),
    ))
}

/// Build the full feedback perturbation `(Y^σ, Y^{σ,x})` from an SCC variable `w` and its
/// geometry derivative `w^x`. Geometry derivative of `Dress_S(feedback(w)_σ)`:
/// `Dress_{S^x}(feedback(w)) + Dress_S(feedback^x(w) + feedback(w^x))`, with
/// `feedback^x(w) = coul_kernel^x·dq` in the charge channel (spin part has no geometry deriv).
#[allow(clippy::too_many_arguments)]
fn feedback_pert(
    ctx: &LinearResponseGeomContext,
    coul: &CoulKernel,
    w: &[f64],
    wx: &[f64],
    sx: &Matrix,
    amat_x_dq: &[f64],
    q0_atom_x: &[f64],
) -> Result<AoPert> {
    let nsh = ctx.basis.shells.len();
    let (dq, dm) = (&w[..nsh], &w[nsh..]);
    let (dqx, dmx) = (&wx[..nsh], &wx[nsh..]);
    // Value potentials.
    let dv_c = coul.apply(dq)?;
    let dv_s = spin_shell_potential(&ctx.info, dm);
    // Derivative potentials: coul^x·dq (fixed w) + coul·dqx  ;  spin: v_s(dmx) only.
    let dv_c_x_fixed = coul.apply_deriv(dq, amat_x_dq, q0_atom_x);
    let dv_c_from_wx = coul.apply(dqx)?;
    let dv_s_x = spin_shell_potential(&ctx.info, dmx);
    let mut va = vec![0.0; nsh];
    let mut vb = vec![0.0; nsh];
    let mut vax = vec![0.0; nsh];
    let mut vbx = vec![0.0; nsh];
    for ish in 0..nsh {
        va[ish] = dv_c[ish] - dv_s[ish];
        vb[ish] = dv_c[ish] + dv_s[ish];
        let vcx = dv_c_x_fixed[ish] + dv_c_from_wx[ish];
        vax[ish] = vcx - dv_s_x[ish];
        vbx[ish] = vcx + dv_s_x[ish];
    }
    let basis = &ctx.basis;
    let s = &ctx.overlap;
    // Y^{σ,x} = Dress_{S^x}(v^σ) + Dress_S(v^{σ,x}). (The value part Y^σ = Dress_S(v^σ) is
    // DOF-independent — handled via the prepared feedback base coefficients — so it is not built.)
    let mut yx_a = shell_potential_dress(basis, sx, &va);
    let mut yx_b = shell_potential_dress(basis, sx, &vb);
    let da = shell_potential_dress(basis, s, &vax);
    let db = shell_potential_dress(basis, s, &vbx);
    for k in 0..yx_a.as_slice().len() {
        yx_a.as_mut_slice()[k] += da.as_slice()[k];
        yx_b.as_mut_slice()[k] += db.as_slice()[k];
    }
    Ok(AoPert { yx_a, yx_b })
}

/// Solve the SCC-CPHF ground-state geometry response for ALL Cartesian DOFs via the DIRECT
/// `(I−M)^{-1} b_frozen` solve (the `op` factor is DOF-independent, built once). Returns per-DOF
/// the per-channel density responses and the frozen sources (`S^x` reused downstream).
fn analytic_scc_cphf(
    ctx: &LinearResponseGeomContext,
    op: &ScreenedOperator,
    system: &crate::system::PeriodicSystem,
    params: &crate::params::Gfn1Parameters,
) -> Result<(Vec<SccCphfDof>, Vec<FrozenSource>)> {
    let sources = frozen_sources(ctx, system, params)?;
    let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
    let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
    let mut dofs = Vec::with_capacity(sources.len());
    for src in &sources {
        dofs.push(solve_scc_cphf_dof(
            ctx,
            &cr_a,
            &cr_b,
            op,
            &src.frozen_fock_a,
            &src.frozen_fock_b,
            &src.sx,
        )?);
    }
    Ok((dofs, sources))
}

// ---------------------------------------------------------------------------------------------
// Sec 6–8: K^x → dU/dR, dV/dR (the analytic replacement for the FD dU/dR in the force).
// ---------------------------------------------------------------------------------------------

/// Regularized inverse `(χ + REG·I)^{-1}` matching `plus_u::extract_uv_from_response` (so the
/// analytic `dK/dR` matches the FD-differenced extraction; the constant REG drops out of the
/// derivative). Returns `None` if singular.
fn reg_inverse(chi: &[Vec<f64>], reg: f64) -> Option<Vec<Vec<f64>>> {
    let mut m: Vec<Vec<f64>> = chi.to_vec();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += reg;
    }
    crate::plus_u::invert_small(&m)
}

/// `A · B` for row-major `Vec<Vec<f64>>` (small dense).
fn matmul_vv(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// Analytic `dU_I/dR` and `dV_IJ/dR` for every Cartesian DOF, aligned to `ref_subspace`
/// (per correlated atom, by `atom_index`) and `ref_pairs` (per selected pair, by endpoint
/// `atom_index`). Returns `Ok(None)` when +U linear response is off / the subspace is empty.
///
/// `K = X0 − X` with `X0 = (χ0+REG·I)^{-1}`, `X = (χ+REG·I)^{-1}` (REG = 1e-3, matching
/// `extract_uv_from_response`); `K^x = −X0 χ0^x X0 + X χ^x X`. `U^x_I = K^x_II`, `V^x_IJ = −K^x_IJ`.
/// U is zeroed where the reference `U_I` hits the physical clamp `[0, U_MAX]` (matching the FD
/// oracle, which differences the clamped extraction).
pub(crate) fn analytic_dudr(
    system: &crate::system::PeriodicSystem,
    params: &crate::params::Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    ref_subspace: &[crate::plus_u::CorrelatedAtom],
    ref_pairs: &[crate::plus_u::IntersitePair],
) -> Result<Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>> {
    const REG: f64 = 1.0e-3;
    const U_MAX: f64 = 1.0;
    let Some(ctx) = crate::spin::linear_response_geom_context(system, params, options)? else {
        return Ok(None);
    };
    let subspace = &ctx.subspace;
    let ncorr = subspace.len();
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
    let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
    let block = ScreenedBlock { ctx: &ctx, cr_a: &cr_a, cr_b: &cr_b };
    let coul = CoulKernel { ctx: &ctx };
    // The screened operator `(I−M)^{-1}` is built ONCE and reused for BOTH the ground-state
    // SCC-CPHF direct solves and the screened-response solves (levers 1/3).
    let op = build_screened_operator(&ctx, &block, &coul)?;
    let (dofs, sources) = analytic_scc_cphf(&ctx, &op, system, params)?;

    // DOF-independent base data (projectors, per-J base bundles, screened `u_J`, base χ0/χ,
    // the amat-derivative table): built ONCE and reused for every DOF (levers 1/3/4). The
    // (I−M)^{-1} factor in `op` is also built once and reused across all 3N screened solves.
    let prep = prepare_response(&ctx, &block, &coul, &op, system, params)?;
    // Regularized inverses of the base χ0/χ (DOF-independent).
    let (Some(x0), Some(x)) = (reg_inverse(&prep.chi0, REG), reg_inverse(&prep.chi, REG)) else {
        return Ok(None);
    };
    // Reference U_I (= K_II clamped) — used to zero the derivative at the clamp boundary.
    let mut k_base = vec![vec![0.0; ncorr]; ncorr];
    for i in 0..ncorr {
        for j in 0..ncorr {
            k_base[i][j] = x0[i][j] - x[i][j];
        }
    }
    let u_ref: Vec<f64> = (0..ncorr).map(|i| k_base[i][i].clamp(0.0, U_MAX)).collect();

    // Map the context subspace slot → reference-subspace slot (by atom_index).
    let ctx_atom: Vec<usize> = subspace.iter().map(|a| a.atom_index).collect();
    let ref_slot_of_ctx: Vec<Option<usize>> = ctx_atom
        .iter()
        .map(|&ai| ref_subspace.iter().position(|r| r.atom_index == ai))
        .collect();
    // Reference pair (i,j)-in-context-slots for each ref pair (by endpoint atom_index).
    let ref_pair_ctx: Vec<Option<(usize, usize)>> = ref_pairs
        .iter()
        .map(|p| {
            let ia = ref_subspace[p.a].atom_index;
            let ib = ref_subspace[p.b].atom_index;
            let ci = ctx_atom.iter().position(|&a| a == ia);
            let cj = ctx_atom.iter().position(|&a| a == ib);
            match (ci, cj) {
                (Some(ci), Some(cj)) => Some((ci, cj)),
                _ => None,
            }
        })
        .collect();

    let mut du = vec![vec![0.0; ref_subspace.len()]; ndim];
    let mut dv = vec![vec![0.0; ref_pairs.len()]; ndim];
    for c in 0..ndim {
        let chi0_x = chi0_deriv_prepared(&ctx, &cr_a, &cr_b, &prep, &sources[c].sx, &dofs[c])?;
        let chi_x = chi_deriv_prepared(&ctx, &block, &coul, &op, &prep, &sources[c].sx, &dofs[c], c)?;
        // K^x = −X0 χ0^x X0 + X χ^x X.
        let t0 = matmul_vv(&x0, &matmul_vv(&chi0_x, &x0));
        let t1 = matmul_vv(&x, &matmul_vv(&chi_x, &x));
        let mut k_x = vec![vec![0.0; ncorr]; ncorr];
        for i in 0..ncorr {
            for j in 0..ncorr {
                k_x[i][j] = -t0[i][j] + t1[i][j];
            }
        }
        // U^x_I = K^x_II (zeroed at the clamp); map to reference subspace slots.
        for ci in 0..ncorr {
            if let Some(rs) = ref_slot_of_ctx[ci] {
                let interior = u_ref[ci] > 1.0e-9 && u_ref[ci] < U_MAX - 1.0e-9;
                du[c][rs] = if interior { k_x[ci][ci] } else { 0.0 };
            }
        }
        // V^x_IJ = −K^x_IJ (symmetrized), map to reference pairs.
        for (pk, ctx_ij) in ref_pair_ctx.iter().enumerate() {
            if let Some((ci, cj)) = *ctx_ij {
                dv[c][pk] = -0.5 * (k_x[ci][cj] + k_x[cj][ci]);
            }
        }
    }
    Ok(Some((du, dv)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::electronic::ElectronicOptions;
    use crate::params::Gfn1Parameters;
    use crate::system::PeriodicSystem;

    /// FD ORACLE for `dU/dR`, `dV/dR`: central finite difference of the
    /// linear-response evaluator [`crate::spin::linear_response_uv_for_system`] over
    /// displaced geometries, matched back to the reference subspace / pair ordering by
    /// `atom_index` (exactly as the production FD path in
    /// `crate::gradient::plus_u_consistency_gradient_terms`). Returns, per Cartesian
    /// DOF, the vectors `(dU/dR, dV/dR)` aligned to the reference subspace / pairs.
    pub(super) fn fd_dudr_oracle(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        step: f64,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let (ref_sub, ref_pairs) =
            crate::spin::linear_response_uv_for_system(system, params, options)?;
        let ref_atom: Vec<usize> = ref_sub.iter().map(|a| a.atom_index).collect();
        let ref_pair_atoms: Vec<(usize, usize)> = ref_pairs
            .iter()
            .map(|p| (ref_sub[p.a].atom_index, ref_sub[p.b].atom_index))
            .collect();
        let uv_at = |sys: &PeriodicSystem| -> Result<(Vec<f64>, Vec<f64>)> {
            let (sub, prs) = crate::spin::linear_response_uv_for_system(sys, params, options)?;
            let mut u = vec![0.0; ref_atom.len()];
            for (slot, &ai) in ref_atom.iter().enumerate() {
                if let Some(found) = sub.iter().find(|a| a.atom_index == ai) {
                    u[slot] = found.u;
                }
            }
            let mut v = vec![0.0; ref_pair_atoms.len()];
            for (slot, &(ia, ib)) in ref_pair_atoms.iter().enumerate() {
                if let Some(found) = prs.iter().find(|p| {
                    let (pa, pb) = (sub[p.a].atom_index, sub[p.b].atom_index);
                    (pa == ia && pb == ib) || (pa == ib && pb == ia)
                }) {
                    v[slot] = found.v;
                }
            }
            Ok((u, v))
        };
        let nat = system.atoms.len();
        let mut du = vec![vec![0.0; ref_atom.len()]; 3 * nat];
        let mut dv = vec![vec![0.0; ref_pair_atoms.len()]; 3 * nat];
        for atom in 0..nat {
            for axis in 0..3 {
                let mut sp = system.clone();
                let mut sm = system.clone();
                let (dp, dm) = match axis {
                    0 => (&mut sp.atoms[atom].position.x, &mut sm.atoms[atom].position.x),
                    1 => (&mut sp.atoms[atom].position.y, &mut sm.atoms[atom].position.y),
                    _ => (&mut sp.atoms[atom].position.z, &mut sm.atoms[atom].position.z),
                };
                *dp += step;
                *dm -= step;
                let (up, vp) = uv_at(&sp)?;
                let (um, vm) = uv_at(&sm)?;
                let dof = 3 * atom + axis;
                for i in 0..ref_atom.len() {
                    du[dof][i] = (up[i] - um[i]) / (2.0 * step);
                }
                for k in 0..ref_pair_atoms.len() {
                    dv[dof][k] = (vp[k] - vm[k]) / (2.0 * step);
                }
            }
        }
        Ok((du, dv))
    }

    /// Sanity: the FD oracle runs on ScH (triplet, 300 K) and produces a non-trivial
    /// `dU/dR` for the single correlated Sc `d` subspace. This pins the oracle the
    /// analytic stages are gated against.
    #[test]
    #[allow(non_snake_case)]
    fn fd_dudr_oracle_scH_nontrivial() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            PeriodicSystem::from_xyz_str("2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n", 0.0, false)
                .unwrap();
        let mut opt = ElectronicOptions::default();
        opt.electronic_temperature = 300.0;
        opt.energy_tolerance = 1.0e-10;
        opt.charge_tolerance = 1.0e-9;
        opt.max_scc = 500;
        opt.spin_multiplicity = Some(3);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true;
        let (du, _dv) = fd_dudr_oracle(&system, &params, &opt, 1.0e-3).unwrap();
        assert!(!du.is_empty(), "no DOFs");
        assert!(!du[0].is_empty(), "empty correlated subspace on ScH");
        let maxabs = du
            .iter()
            .flatten()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(
            maxabs.is_finite() && maxabs > 1.0e-4,
            "FD dU/dR is trivial/non-finite on ScH: max|dU/dR| = {maxabs:.3e}"
        );
    }

    fn load_params() -> Option<Gfn1Parameters> {
        Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
    }

    #[allow(non_snake_case)]
    fn scH_options() -> ElectronicOptions {
        let mut opt = ElectronicOptions::default();
        opt.electronic_temperature = 300.0;
        opt.energy_tolerance = 1.0e-10;
        opt.charge_tolerance = 1.0e-9;
        opt.max_scc = 500;
        opt.spin_multiplicity = Some(3);
        opt.spin_polarization = true;
        opt.plus_u = true;
        opt.hubbard_u_linear_response = true;
        opt
    }

    #[allow(non_snake_case)]
    fn scH_system() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str("2\nScH\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.80\n", 0.0, false).unwrap()
    }

    /// Build the DOF-independent screened operator `(I−M)^{-1}` for a context (test helper for
    /// the ground-state SCC-CPHF direct solve).
    fn op_of(ctx: &LinearResponseGeomContext) -> ScreenedOperator {
        let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
        let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
        let block = ScreenedBlock { ctx, cr_a: &cr_a, cr_b: &cr_b };
        let coul = CoulKernel { ctx };
        build_screened_operator(ctx, &block, &coul).unwrap()
    }

    fn displaced(system: &PeriodicSystem, atom: usize, axis: usize, d: f64) -> PeriodicSystem {
        let mut s = system.clone();
        match axis {
            0 => s.atoms[atom].position.x += d,
            1 => s.atoms[atom].position.y += d,
            _ => s.atoms[atom].position.z += d,
        }
        s
    }

    fn max_mat_diff(a: &Matrix, b: &Matrix) -> f64 {
        a.as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    /// STAGE 1 GATE (Sec 3): the analytic SCC-CPHF ground-state density response `P^{σ,x}`
    /// must match a central FD of the base per-channel density `P^σ(R)` on ScH (triplet,
    /// 300 K), for every Cartesian DOF. This is the foundation the whole χ0/χ derivative
    /// chain rests on. Also gates the correlated-occupation response `n^{σ,x}_I`.
    #[test]
    fn scc_cphf_density_response_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = scH_system();
        let opt = scH_options();
        let ctx = crate::spin::linear_response_geom_context(&system, &params, &opt)
            .unwrap()
            .expect("non-empty +U context on ScH");
        let op = op_of(&ctx);
        let (dofs, _sources) = analytic_scc_cphf(&ctx, &op, &system, &params).unwrap();

        let base_density = |sys: &PeriodicSystem| -> (Matrix, Matrix) {
            let c = crate::spin::linear_response_geom_context(sys, &params, &opt)
                .unwrap()
                .unwrap();
            (channel_density(&c.ch_a), channel_density(&c.ch_b))
        };
        let h = 5.0e-4;
        let nat = system.atoms.len();
        let mut worst = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let (pa_p, pb_p) = base_density(&displaced(&system, atom, axis, h));
                let (pa_m, pb_m) = base_density(&displaced(&system, atom, axis, -h));
                let mut fd_a = pa_p.clone();
                let mut fd_b = pb_p.clone();
                for k in 0..fd_a.as_slice().len() {
                    fd_a.as_mut_slice()[k] = (pa_p.as_slice()[k] - pa_m.as_slice()[k]) / (2.0 * h);
                    fd_b.as_mut_slice()[k] = (pb_p.as_slice()[k] - pb_m.as_slice()[k]) / (2.0 * h);
                }
                let dof = 3 * atom + axis;
                // Reconstruct the AO P^{σ,x} from the stored MO coefficient for the FD comparison.
                let px_a = ao_from_mo(&ctx.ch_a.mos, &dofs[dof].dx_a).unwrap();
                let px_b = ao_from_mo(&ctx.ch_b.mos, &dofs[dof].dx_b).unwrap();
                worst = worst
                    .max(max_mat_diff(&px_a, &fd_a))
                    .max(max_mat_diff(&px_b, &fd_b));
            }
        }
        assert!(
            worst < 1.0e-5,
            "analytic P^x vs FD (ScH 300 K): max|Δ| = {worst:.3e}"
        );
    }

    fn max_vv_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()))
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    /// χ0 (my independent build) must match the production `analytic_chi0`, and χ0^x must
    /// match a central FD of my χ0 over geometry (ScH triplet, 300 K). STAGE 2 GATE (Sec 4).
    #[test]
    fn chi0_deriv_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = scH_system();
        let opt = scH_options();
        let ctx = crate::spin::linear_response_geom_context(&system, &params, &opt)
            .unwrap()
            .unwrap();
        let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
        let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
        let op = op_of(&ctx);
        let (dofs, sources) = analytic_scc_cphf(&ctx, &op, &system, &params).unwrap();

        // Consistency: my χ0 == production χ0 (use DOF 0's build; χ0 itself is DOF-independent).
        let (chi0_mine, chi0_x0) =
            chi0_and_deriv_dof(&ctx, &cr_a, &cr_b, &sources[0].sx, &dofs[0]).unwrap();
        let (chi0_prod, _chi_prod) = ctx.analytic_chi0_chi().unwrap();
        let cons = max_vv_diff(&chi0_mine, &chi0_prod);
        assert!(cons < 1.0e-8, "my χ0 vs production χ0: max|Δ| = {cons:.3e}");
        let _ = chi0_x0;

        // FD my χ0 over geometry, per DOF, vs analytic χ0^x.
        let chi0_of = |sys: &PeriodicSystem| -> Vec<Vec<f64>> {
            let c = crate::spin::linear_response_geom_context(sys, &params, &opt).unwrap().unwrap();
            let ca = ChannelResponse::new(&c.ch_a, c.kt, &c.overlap);
            let cb = ChannelResponse::new(&c.ch_b, c.kt, &c.overlap);
            let o = op_of(&c);
            let (d, s) = analytic_scc_cphf(&c, &o, sys, &params).unwrap();
            chi0_and_deriv_dof(&c, &ca, &cb, &s[0].sx, &d[0]).unwrap().0
        };
        let h = 5.0e-4;
        let nat = system.atoms.len();
        let mut worst = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let chi0_p = chi0_of(&displaced(&system, atom, axis, h));
                let chi0_m = chi0_of(&displaced(&system, atom, axis, -h));
                let mut fd = chi0_p.clone();
                for i in 0..fd.len() {
                    for jj in 0..fd[i].len() {
                        fd[i][jj] = (chi0_p[i][jj] - chi0_m[i][jj]) / (2.0 * h);
                    }
                }
                let dof = 3 * atom + axis;
                let (_c0, c0x) =
                    chi0_and_deriv_dof(&ctx, &cr_a, &cr_b, &sources[dof].sx, &dofs[dof]).unwrap();
                worst = worst.max(max_vv_diff(&c0x, &fd));
            }
        }
        assert!(worst < 1.0e-4, "analytic χ0^x vs FD (ScH 300 K): max|Δ| = {worst:.3e}");
    }

    /// Compute χ (screened) for a system via the Sec-5 building blocks (value only), for the FD
    /// reference of χ^x.
    fn chi_value_of(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        opt: &ElectronicOptions,
    ) -> Vec<Vec<f64>> {
        let ctx = crate::spin::linear_response_geom_context(system, params, opt).unwrap().unwrap();
        let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
        let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
        let block = ScreenedBlock { ctx: &ctx, cr_a: &cr_a, cr_b: &cr_b };
        let coul = CoulKernel { ctx: &ctx };
        let op = build_screened_operator(&ctx, &block, &coul).unwrap();
        let (dofs, sources) = analytic_scc_cphf(&ctx, &op, system, params).unwrap();
        chi_and_deriv_dof(&ctx, &block, &coul, &op, &sources[0].sx, &dofs[0], system, params, 0)
            .unwrap()
            .0
    }

    /// STAGE 3 GATE (Sec 5): my χ must match production `analytic_chi`, and χ^x must match a
    /// central FD of my χ over geometry (ScH triplet, 300 K).
    #[test]
    fn chi_deriv_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = scH_system();
        let opt = scH_options();
        let ctx = crate::spin::linear_response_geom_context(&system, &params, &opt).unwrap().unwrap();
        let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
        let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
        let block = ScreenedBlock { ctx: &ctx, cr_a: &cr_a, cr_b: &cr_b };
        let coul = CoulKernel { ctx: &ctx };
        let op = build_screened_operator(&ctx, &block, &coul).unwrap();
        let (dofs, sources) = analytic_scc_cphf(&ctx, &op, &system, &params).unwrap();

        // Consistency: my χ == production χ.
        let (chi_mine, _cx) =
            chi_and_deriv_dof(&ctx, &block, &coul, &op, &sources[0].sx, &dofs[0], &system, &params, 0)
                .unwrap();
        let (_chi0_prod, chi_prod) = ctx.analytic_chi0_chi().unwrap();
        let cons = max_vv_diff(&chi_mine, &chi_prod);
        assert!(cons < 1.0e-6, "my χ vs production χ: max|Δ| = {cons:.3e}");

        // FD my χ over geometry vs analytic χ^x.
        let h = 5.0e-4;
        let nat = system.atoms.len();
        let mut worst = 0.0_f64;
        for atom in 0..nat {
            for axis in 0..3 {
                let chi_p = chi_value_of(&displaced(&system, atom, axis, h), &params, &opt);
                let chi_m = chi_value_of(&displaced(&system, atom, axis, -h), &params, &opt);
                let mut fd = chi_p.clone();
                for i in 0..fd.len() {
                    for jj in 0..fd[i].len() {
                        fd[i][jj] = (chi_p[i][jj] - chi_m[i][jj]) / (2.0 * h);
                    }
                }
                let dof = 3 * atom + axis;
                let (_c, cx) = chi_and_deriv_dof(
                    &ctx, &block, &coul, &op, &sources[dof].sx, &dofs[dof], &system, &params, dof,
                )
                .unwrap();
                worst = worst.max(max_vv_diff(&cx, &fd));
            }
        }
        assert!(worst < 1.0e-4, "analytic χ^x vs FD (ScH 300 K): max|Δ| = {worst:.3e}");
    }

    /// STAGE 4 GATE (Sec 6): the assembled analytic `dU/dR` must match the FD oracle
    /// (`linear_response_uv_for_system` central-differenced) on ScH triplet, 300 K — the
    /// quantity that replaces the FD dU/dR in the consistency force.
    #[test]
    fn analytic_dudr_matches_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = scH_system();
        let opt = scH_options();
        let (ref_sub, ref_pairs) =
            crate::spin::linear_response_uv_for_system(&system, &params, &opt).unwrap();
        let (du_an, _dv_an) = analytic_dudr(&system, &params, &opt, &ref_sub, &ref_pairs)
            .unwrap()
            .expect("non-empty +U context");
        let (du_fd, _dv_fd) = fd_dudr_oracle(&system, &params, &opt, 1.0e-3).unwrap();
        assert_eq!(du_an.len(), du_fd.len(), "DOF count mismatch");
        let mut worst = 0.0_f64;
        for c in 0..du_an.len() {
            for i in 0..du_an[c].len() {
                worst = worst.max((du_an[c][i] - du_fd[c][i]).abs());
            }
        }
        assert!(
            worst < 1.0e-6,
            "analytic dU/dR vs FD (ScH 300 K): max|Δ| = {worst:.3e}"
        );
    }

    /// Ni(CO)3 (mor41 ED03) geometry, Angstrom — a strongly-screened transition-metal complex
    /// (|corr|=1 Ni d) where the screened linear-response fixed point is ill-conditioned. This is
    /// the regime that exposed the divergent-iteration bug in `spin::analytic_chi` (now solved
    /// directly); ScH alone could not catch it.
    fn ni_co3_system() -> PeriodicSystem {
        PeriodicSystem::from_xyz_str(
            "7\nNiCO3\nNi -0.7629039 -0.2803608 0.4889495\nC 0.9417181 -0.5614688 -0.0291272\nC -1.6112009 1.2380907 0.0119109\nC -1.6202975 -1.5176032 1.4825547\nO 2.0270968 -0.7402781 -0.3595063\nO -2.1515398 2.2049067 -0.2921948\nO -2.1666934 -2.3052194 2.1152351\n",
            0.0, false,
        )
        .unwrap()
    }

    fn ni_options() -> ElectronicOptions {
        let mut opt = scH_options();
        opt.spin_multiplicity = None; // resolve from electron count
        opt
    }

    /// SCREENED-χ CONSISTENCY GATE (Ni(CO)3, 300 K): the analytic screened response `χ`
    /// (`chi_deriv_prepared`'s value / `prepare_response`) must equal the production
    /// `spin::analytic_chi` on a strongly-screened TM complex. Before the direct-solve fix,
    /// production's mixing iteration DIVERGED here (delta ~20 after 500 iters), so the two
    /// disagreed by ~1% and their geometry derivatives disagreed by ~5 orders of magnitude —
    /// the root cause of the spurious dU/dR force on Ni. Guards against regression.
    #[test]
    fn linear_response_chi_converges_on_ni() {
        let Some(params) = load_params() else {
            return;
        };
        let system = ni_co3_system();
        let opt = ni_options();
        let ctx = crate::spin::linear_response_geom_context(&system, &params, &opt)
            .unwrap()
            .expect("non-empty Ni +U context");
        let cr_a = ChannelResponse::new(&ctx.ch_a, ctx.kt, &ctx.overlap);
        let cr_b = ChannelResponse::new(&ctx.ch_b, ctx.kt, &ctx.overlap);
        let block = ScreenedBlock { ctx: &ctx, cr_a: &cr_a, cr_b: &cr_b };
        let coul = CoulKernel { ctx: &ctx };
        let op = build_screened_operator(&ctx, &block, &coul).unwrap();
        let prep = prepare_response(&ctx, &block, &coul, &op, &system, &params).unwrap();
        let (_chi0_prod, chi_prod) = ctx.analytic_chi0_chi().unwrap();
        let cons = max_vv_diff(&prep.chi, &chi_prod);
        assert!(
            cons < 1.0e-9,
            "my screened χ vs production analytic_chi on Ni: max|Δ| = {cons:.3e} \
             (production analytic_chi may be diverging again)"
        );
    }

    /// SCREENED-χ DERIVATIVE + dU/dR GATE ON Ni(CO)3, 300 K. The whole class of bug that
    /// slipped past the ScH-only gates: on a strongly-screened TM system, the analytic
    /// `dU/dR` must match a central FD of the (now converged) linear-response U. Exercises the
    /// screened `dχ/dR` chain (`b^x`, `M^x u`, the `(I−M)u^x=b^x+M^x u` solve, `G^x·u`, and the
    /// screened-total-Y output) end-to-end where they actually bite. Only the Ni atom's 3 DOFs
    /// are FD-differenced (cheap; the physics is the Ni d response).
    #[test]
    fn analytic_dudr_matches_fd_ni() {
        let Some(params) = load_params() else {
            return;
        };
        let system = ni_co3_system();
        let opt = ni_options();
        let (ref_sub, ref_pairs) =
            crate::spin::linear_response_uv_for_system(&system, &params, &opt).unwrap();
        assert!(!ref_sub.is_empty(), "empty Ni subspace");
        let (du_an, _) = analytic_dudr(&system, &params, &opt, &ref_sub, &ref_pairs)
            .unwrap()
            .expect("non-empty Ni +U context");
        // Central FD of the Ni-atom (atom 0) linear-response U, at two steps to confirm the FD
        // itself is converged (the divergent-iteration bug made this FD unstable across steps).
        let u_at = |sys: &PeriodicSystem| -> f64 {
            crate::spin::linear_response_uv_for_system(sys, &params, &opt).unwrap().0[0].u
        };
        for &h in &[1.0e-3_f64, 2.0e-3] {
            let mut worst = 0.0_f64;
            for axis in 0..3 {
                let fd = (u_at(&displaced(&system, 0, axis, h)) - u_at(&displaced(&system, 0, axis, -h)))
                    / (2.0 * h);
                worst = worst.max((du_an[axis][0] - fd).abs());
            }
            assert!(
                worst < 5.0e-4,
                "analytic dU/dR vs FD on Ni (h={h}): max|Δ| = {worst:.3e} Ha/bohr"
            );
        }
    }

    /// TIMING HARNESS (ignored by default; run with `--ignored`): wall-time of the analytic
    /// `analytic_dudr` (debug). Benchmarks ScH (tiny, N≈10) AND a larger Sc-centered cluster
    /// (|corr|=1, larger N) so the O(N³)/DOF constant-factor improvements are measurable. Not a
    /// correctness gate — a manual before/after tool. Reports per-call and per-DOF ms.
    #[test]
    #[ignore]
    fn bench_analytic_dudr_sch() {
        let Some(params) = load_params() else {
            return;
        };
        let bench = |label: &str, system: &PeriodicSystem, opt: &ElectronicOptions, reps: usize| {
            let Ok(Some((ref_sub, ref_pairs))) = crate::spin::linear_response_uv_for_system(system, &params, opt)
                .map(Some)
                .or_else(|_| Ok::<_, ()>(None))
            else {
                eprintln!("BENCH {label}: base linear-response failed");
                return;
            };
            let _ = (&ref_sub, &ref_pairs);
            // Warm up + confirm it produces a result.
            if analytic_dudr(system, &params, opt, &ref_sub, &ref_pairs).ok().flatten().is_none() {
                eprintln!("BENCH {label}: analytic_dudr empty");
                return;
            }
            let ndof = 3 * system.atoms.len();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = analytic_dudr(system, &params, opt, &ref_sub, &ref_pairs).unwrap();
            }
            let per = t0.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "BENCH {label}: {:.2} ms/call, {:.3} ms/DOF (N_atoms={}, {} DOFs)",
                per * 1e3, per * 1e3 / ndof as f64, system.atoms.len(), ndof
            );
        };
        bench("ScH", &scH_system(), &scH_options(), 30);
        // Larger |corr|=1 clusters that push N up while keeping one correlated d — several
        // candidates; whichever converge under linear-response at 300 K get benched.
        let candidates: &[(&str, &str, Option<usize>)] = &[
            ("ScH2", "3\nScH2\nSc 0.0 0.0 0.0\nH 0.0 0.0 1.85\nH 0.0 1.85 0.0\n", Some(2)),
            ("ScCH4", "6\nScCH4\nSc 0.0 0.0 0.0\nC 0.0 0.0 2.1\nH 0.6 0.6 2.6\nH -0.6 -0.6 2.6\nH 0.6 -0.6 2.6\nH -0.6 0.6 2.6\n", Some(2)),
            ("ScC2H4", "7\nScC2H4\nSc 0.0 0.0 0.0\nC 0.7 0.0 2.0\nC -0.7 0.0 2.0\nH 1.3 0.9 2.2\nH 1.3 -0.9 2.2\nH -1.3 0.9 2.2\nH -1.3 -0.9 2.2\n", Some(2)),
        ];
        for (label, xyz, mult) in candidates {
            if let Ok(sys) = PeriodicSystem::from_xyz_str(xyz, 0.0, false) {
                let mut o = scH_options();
                o.spin_multiplicity = *mult;
                bench(label, &sys, &o, 5);
            }
        }
    }
}
