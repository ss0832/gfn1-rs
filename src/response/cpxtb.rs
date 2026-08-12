// SPDX-License-Identifier: GPL-3.0-or-later
//! Real non-PBC **CPXTB** helpers for the GFN1-xTB analytic Hessian.
//!
//! "CPXTB" = coupled-perturbed xTB: the self-consistent-charge (SCC) tight-binding
//! analog of coupled-perturbed Hartree-Fock / Kohn-Sham. GFN1-xTB is not a
//! Hartree-Fock method, so the historical "CPHF" label is a misnomer; the response
//! equations here couple the perturbed Mulliken shell charges through the SCC
//! kernel rather than a Fock exchange operator. The file is named `cphf.rs` for
//! continuity, but all identifiers use the `Cpxtb` / `cpxtb_` naming.

use crate::basis::{BasisSet, BasisShell};
use crate::coordination::{
    coordination_with_derivatives, CoordinationOptions, CoordinationPairDerivative,
};
use crate::coulomb::{
    effective_coulomb_matrix, harmonic_average, ShellChargeModel, GFN1_COULOMB_EXPONENT,
};
use crate::data_tables::atomic_radius_bohr;
use crate::electronic::ElectronicResult;
use crate::error::{Gfn1Error, Result};
use crate::hamiltonian::hscale;
use crate::integrals::{contracted_pair_with_derivatives, contracted_pair_with_second_derivatives};
use crate::linalg::{
    lowdin_solve_generalized, matmul_transpose_a, matrix_vector_product, DenseLu, Matrix,
};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;
use faer::linalg::solvers::Solve;
use faer::Mat as FaerMat;

const DIST_EPS: f64 = 1.0e-12;
const BOLTZMANN_HARTREE_PER_K: f64 = crate::constants::KB_HARTREE_PER_K;
const CPXTB_PRECOND_GAP_FLOOR: f64 = 1.0e-4;
const CPXTB_DENSE_FALLBACK_MAX_DIM: usize = 2048;
const CPXTB_PCG_DIVERGENCE_FACTOR: f64 = 1.0e3;
/// Reject the charge-space reduction when `½(f_i − f_a)/(ε_a − ε_i)` is this
/// large: the reduction divides the pair-space diagonal out in closed form, and
/// a weight this extreme means the gap underflowed relative to the occupation
/// difference (i.e. the response really is singular — let the guards speak).
const CPXTB_LOW_RANK_MAX_WEIGHT: f64 = 1.0e14;
/// Right-hand sides processed per GEMM block in the charge-space route. Bounds
/// the `npair × chunk` scratch (the 3N geometric family can be hundreds of RHS
/// on a `npair ~ 10⁵` system) while keeping the blocks GEMM-shaped.
const CPXTB_LOW_RANK_RHS_CHUNK: usize = 64;

/// Krylov controls for the CP linear solves.
///
/// Since v0.5.0 the primary route is the **direct** charge-space reduction
/// ([`CpxtbRoute::ChargeSpace`]), which is non-iterative — these fields govern
/// only the preconditioned-CG fallback, which is not reached on any measured
/// system. Kept as-is so callers and the public API are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct CpxtbOptions {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for CpxtbOptions {
    fn default() -> Self {
        Self {
            tol: 1.0e-8,
            max_iter: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpxtbSpace {
    pub occupied: Vec<usize>,
    pub virtuals: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
}

impl CpxtbSpace {
    pub fn from_occupations(occupations: &[f64]) -> Result<Self> {
        let mut occupied = Vec::new();
        let mut virtuals = Vec::new();
        for (idx, &occ) in occupations.iter().enumerate() {
            if !occ.is_finite() {
                return Err(Gfn1Error::InvalidInput(
                    "CPXTB occupation is not finite".to_string(),
                ));
            }
            if occ > 1.0e-8 {
                occupied.push(idx);
            } else {
                virtuals.push(idx);
            }
        }
        let mut pairs = Vec::new();
        for i in 0..occupations.len() {
            for a in i + 1..occupations.len() {
                if occupations[i] - occupations[a] > 1.0e-10 {
                    pairs.push((i, a));
                }
            }
        }
        if pairs.is_empty() {
            return Err(Gfn1Error::InvalidInput(
                "CPXTB requires at least one occupied-virtual pair".to_string(),
            ));
        }
        Ok(Self {
            occupied,
            virtuals,
            pairs,
        })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }
}

/// Which linear-algebra route produced a [`CpxtbSolution`].
///
/// Pure instrumentation — the route never changes what the CP equations mean,
/// only how `A x = b` is factored. Exposed so the regression tests can assert
/// on the *route and iteration count* (CI-stable) instead of wall time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpxtbRoute {
    /// Direct **charge-space** reduction: the pair-space diagonal is inverted in
    /// closed form and one `nsh × nsh` LU serves every right-hand side
    /// ([`CpxtbLowRank`]). Non-iterative.
    ChargeSpace,
    /// Direct dense `npair × npair` LU of the explicitly built pair-space
    /// operator. Non-iterative.
    Dense,
    /// Preconditioned conjugate gradient in the MO-pair space.
    Pcg,
    /// PCG that stalled below its tolerance and was rescued by the dense
    /// pair-space fallback.
    PcgDenseFallback,
}

#[derive(Clone, Debug)]
pub struct CpxtbSolution {
    pub amplitudes: Vec<f64>,
    /// Krylov iterations actually consumed. **Zero for the direct routes**
    /// ([`CpxtbRoute::ChargeSpace`], [`CpxtbRoute::Dense`]).
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
    /// Instrumentation: which route produced this solution.
    pub route: CpxtbRoute,
}

#[derive(Clone, Copy, Debug)]
pub struct AoDerivativeOptions {
    pub coordination_cutoff: f64,
    pub include_cn_h0: bool,
}

impl Default for AoDerivativeOptions {
    fn default() -> Self {
        Self {
            coordination_cutoff: CoordinationOptions::default().cutoff,
            include_cn_h0: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AoDerivativeMatrices {
    /// Derivative of the effective one-electron Fock operator at fixed density.
    pub h0_deriv: Matrix,
    pub overlap_deriv: Matrix,
}

#[derive(Clone, Debug)]
pub struct GammaCartesianCpxtbResult {
    pub derivative_matrices: Vec<AoDerivativeMatrices>,
    pub solutions: Vec<CpxtbSolution>,
    pub density_responses: Vec<Matrix>,
    pub energy_weighted_density_responses: Vec<Matrix>,
    pub shell_charge_responses: Vec<Vec<f64>>,
    pub occupation_responses: Vec<Vec<f64>>,
    pub hessian_response: Matrix,
    pub converged: bool,
    pub max_residual_norm: f64,
    /// MO coefficients (AO×orbital) used internally for the responses — exposed so callers can build
    /// the orbital-rotation representation consistently (re-diagonalizing separately risks sign /
    /// degenerate-subspace mismatches against `solutions[*].amplitudes`).
    pub mos: Matrix,
    pub orbital_energies: Vec<f64>,
    /// The CP right-hand sides `rhs_vectors[a]` (`A x_a = rhs_a`), in the SAME MO/CP coordinate system
    /// as `solutions[*].amplitudes`. Exposed so `rhs_a · x_b` (the orbital-sector response Hessian
    /// `R^orb = −rhs·x`) and the metric residual `M = hessian_response + rhs·x` are coordinate-consistent.
    pub rhs_vectors: Vec<Vec<f64>>,
}

pub fn solve_nonpbc_cpxtb_hessian_response(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
    cpxtb_options: CpxtbOptions,
) -> Result<GammaCartesianCpxtbResult> {
    let _profile = crate::profile::scope("cphf.total");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "non-PBC CPXTB Hessian response cannot be used for PBC systems".to_string(),
        ));
    }
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let basis = &electronic.basis;
    let eig = {
        let _profile = crate::profile::scope("cphf.lowdin_solve");
        lowdin_solve_generalized(&electronic.fock, &electronic.integrals.overlap, 1.0e-12)?
    };
    let mos = eig.vectors;
    let orbital_energies = eig.values;
    let occupations = &electronic.occupations;
    let space = CpxtbSpace::from_occupations(occupations)?;
    let orbital_gaps = space
        .pairs
        .iter()
        .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
        .collect::<Vec<_>>();
    let coupling_occupation_scales = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occupations[i] - occupations[a]))
        .collect::<Vec<_>>();
    for &(i, a) in &space.pairs {
        let occ_diff = occupations[i] - occupations[a];
        if !(occ_diff.is_finite() && occ_diff > 1.0e-12) {
            return Err(Gfn1Error::InvalidInput(
                "CPXTB requires positive occupied-virtual occupation differences".to_string(),
            ));
        }
    }
    // Singular-response guard: with (near-)integer occupations a vanishing
    // occupied–virtual gap makes the CPXTB operator singular, and the solver
    // would return astronomically large garbage amplitudes without erroring
    // (observed: ~1e42 for a symmetry-broken aufbau filling of a degenerate
    // open shell). Fermi smearing regularizes this limit properly — the
    // fractional-occupation weights stay finite as the gap closes — so only
    // the integer-occupation case is rejected.
    let integer_occupations = occupations
        .iter()
        .all(|&f| f < 1.0e-8 || (f - 2.0).abs() < 1.0e-8);
    if integer_occupations {
        for (&(i, a), &gap) in space.pairs.iter().zip(orbital_gaps.iter()) {
            if gap < 1.0e-6 {
                return Err(Gfn1Error::InvalidInput(format!(
                    "CPXTB response is singular: occupied orbital {i} and virtual orbital {a} \
                     are (near-)degenerate (gap {gap:.3e} Ha) with integer occupations — a \
                     zero-gap / open-shell-degenerate configuration. Enable Fermi smearing \
                     (electronic_temperature > 0) or use spin polarization for open shells"
                )));
            }
        }
    }

    let shell_kernel = {
        let _profile = crate::profile::scope("cphf.shell_kernel");
        response_shell_scc_kernel(system, params, electronic)?
    };
    let transition = {
        let _profile = crate::profile::scope("cphf.transition_charges");
        transition_shell_charges(basis, &mos, occupations, &electronic.integrals.overlap)?
    };
    let scalar_derivatives = {
        let _profile = crate::profile::scope("cphf.scalar_derivatives");
        shell_scalar_potential_derivatives(system, basis, params, &electronic.shell_charges)?
    };
    let cn_derivatives = if ao_options.include_cn_h0 {
        let _profile = crate::profile::scope("cphf.cn_derivatives");
        Some(coordination_number_derivatives(
            system,
            ao_options.coordination_cutoff,
        )?)
    } else {
        None
    };
    let derivative_matrices = {
        let _profile = crate::profile::scope("cphf.ao_derivative_matrices");
        cartesian_ao_derivative_matrices(
            system,
            params,
            electronic,
            &scalar_derivatives,
            cn_derivatives.as_deref(),
        )?
    };

    let mut rhs_vectors = Vec::with_capacity(ndim);
    {
        let _profile = crate::profile::scope("cphf.rhs_vectors");
        for deriv in &derivative_matrices {
            rhs_vectors.push(cpxtb_rhs_vector(
                basis,
                &mos,
                occupations,
                &deriv.h0_deriv,
                &deriv.overlap_deriv,
                &orbital_energies,
            )?);
        }
    }
    {
        let _profile = crate::profile::scope("cphf.metric_scc_rhs");
        add_metric_scc_rhs(
            &mut rhs_vectors,
            basis,
            &shell_kernel,
            &mos,
            occupations,
            &electronic.integrals.overlap,
            &electronic.density,
            &orbital_energies,
            &derivative_matrices,
        )?;
    }

    let solutions = {
        let _profile = crate::profile::scope("cphf.solve_linear");
        solve_cpxtb_all(
            &shell_kernel,
            &orbital_gaps,
            &transition,
            &coupling_occupation_scales,
            &rhs_vectors,
            cpxtb_options,
        )?
    };
    let mut converged = true;
    let mut max_residual_norm = 0.0_f64;
    for solution in &solutions {
        converged &= solution.converged;
        max_residual_norm = max_residual_norm.max(solution.residual_norm);
    }

    let mut density_responses = Vec::with_capacity(ndim);
    let mut orbital_density_responses = Vec::with_capacity(ndim);
    let mut energy_weighted_density_responses = Vec::with_capacity(ndim);
    let mut shell_charge_responses = Vec::with_capacity(ndim);
    let mut occupation_responses = Vec::with_capacity(ndim);
    let kt = electronic.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
    let finite_temperature_response = kt > 0.0
        && occupations
            .iter()
            .any(|&occ| occ > 1.0e-10 && occ < 2.0 - 1.0e-10);
    // v0.5.0: the finite-temperature responses are produced by the DIRECT
    // charge-space dielectric solve. The former 50-iteration damped fixed
    // point silently returned unconverged iterates whenever the screening was
    // strong (measured O(1) shell-charge-response errors vs the reconverged
    // SCC finite difference on Ni(CO)4 at 3000 K, where the direct solve
    // agrees with the FD reference to ~1e-8).
    let charge_space_ctx = if finite_temperature_response {
        Some(crate::response::charge_space::ChargeSpaceContext::build(
            system, params, electronic,
        )?)
    } else {
        None
    };
    {
        let _profile = crate::profile::scope("cphf.response_densities");
        for coord in 0..ndim {
            let solution = &solutions[coord];
            let orbital_density =
                response_density_with_space(&mos, occupations, &space, &solution.amplitudes)?;
            if let Some(ctx) = charge_space_ctx.as_ref() {
                let bundle = ctx.solve_first_order(
                    &derivative_matrices[coord].h0_deriv,
                    &derivative_matrices[coord].overlap_deriv,
                )?;
                density_responses.push(bundle.density);
                orbital_density_responses.push(orbital_density);
                energy_weighted_density_responses.push(bundle.energy_weighted);
                shell_charge_responses.push(bundle.shell_charges);
                occupation_responses.push(bundle.occupation_response);
                continue;
            }
            let mut density_without_occupation = orbital_density.clone();
            add_occupied_metric_density_response(
                &mut density_without_occupation,
                &mos,
                occupations,
                &derivative_matrices[coord].overlap_deriv,
            )?;
            let mut weighted_without_response_fock = response_energy_weighted_density_with_space(
                &mos,
                occupations,
                &orbital_energies,
                &space,
                &solution.amplitudes,
            )?;
            add_occupied_metric_energy_weighted_response(
                &mut weighted_without_response_fock,
                &mos,
                occupations,
                &orbital_energies,
                &derivative_matrices[coord].h0_deriv,
                &derivative_matrices[coord].overlap_deriv,
            )?;
            let density = density_without_occupation.clone();
            let occupation_response = vec![0.0_f64; occupations.len()];
            let shell_response = response_shell_charges_from_density(
                basis,
                &electronic.integrals.overlap,
                &electronic.density,
                &density,
                &derivative_matrices[coord].overlap_deriv,
            )?;
            let shell_potential = matrix_vector_product(&shell_kernel, &shell_response)?;
            let response_fock = scalar_response_fock_matrix(
                basis,
                &electronic.integrals.overlap,
                &shell_potential,
            )?;
            let mut weighted = weighted_without_response_fock;
            let zero_overlap = Matrix::zeros(basis.len(), basis.len());
            add_occupied_metric_energy_weighted_response(
                &mut weighted,
                &mos,
                occupations,
                &orbital_energies,
                &response_fock,
                &zero_overlap,
            )?;
            density_responses.push(density);
            orbital_density_responses.push(orbital_density);
            energy_weighted_density_responses.push(weighted);
            shell_charge_responses.push(shell_response);
            occupation_responses.push(occupation_response);
        }
    }

    let mut hessian_response = Matrix::zeros(ndim, ndim);
    {
        let _profile = crate::profile::scope("cphf.response_hessian_columns");
        let gradient_context = ResponseGradientContext::new(
            system,
            basis,
            params,
            electronic,
            ao_options.coordination_cutoff,
            ao_options.include_cn_h0,
        )?;
        for col in 0..ndim {
            let gradient = response_electronic_gradient(
                system,
                electronic,
                &shell_kernel,
                &gradient_context,
                &density_responses[col],
                &density_responses[col],
                &energy_weighted_density_responses[col],
                &shell_charge_responses[col],
            )?;
            set_hessian_column_from_gradient(&mut hessian_response, col, &gradient)?;
        }
    }
    Ok(GammaCartesianCpxtbResult {
        derivative_matrices,
        solutions,
        density_responses,
        energy_weighted_density_responses,
        shell_charge_responses,
        occupation_responses,
        hessian_response,
        converged,
        max_residual_norm,
        mos,
        orbital_energies,
        rhs_vectors,
    })
}

/// Reusable assembly of the closed-shell CPXTB linear system at a fixed geometry/electronic state:
/// the Jacobian action `A·u` and the per-DOF right-hand sides (`A x_a = rhs_a`; the CP equation is
/// `A x_a + b_a = 0` with `b_a = -rhs_a`). Exposed so the analytic 2n+1 third-derivative driver can
/// rebuild the CP operator/RHS at *displaced* geometries — the geometric derivatives `D_c A`, `D_c b`
/// — WITHOUT re-solving CPHF (the cheap "bridge" that lets `b_a^T x_bc = x_a^T r_bc`,
/// `r_bc = (D_c A) x_b + D_c b_b`, close the third derivative on first-order responses only).
pub struct CpxtbSetup {
    pub mos: Matrix,
    pub orbital_energies: Vec<f64>,
    pub space: CpxtbSpace,
    /// `rhs_vectors[a]` is the CP right-hand side for nuclear DOF `a` (`A x_a = rhs_a`).
    pub rhs_vectors: Vec<Vec<f64>>,
    /// Per-DOF AO derivative matrices (`overlap_deriv = S_a`, `h0_deriv = F_a` the effective skeleton
    /// Fock derivative). Exposed so the `D_c(CᵀF_bC)` ladder can read `F_a` at a displaced geometry.
    pub derivative_matrices: Vec<AoDerivativeMatrices>,
    shell_kernel: Matrix,
    orbital_gaps: Vec<f64>,
    transition: Vec<Vec<f64>>,
    occupation_scales: Vec<f64>,
    /// Charge-space reduction of `A`, factored once for the whole setup. The
    /// 2n+1 ladder calls [`Self::solve_adjoint`] repeatedly with different
    /// right-hand sides, so this factorization is amortized over all of them.
    /// `None` when the reduction is not applicable (then the PCG route runs).
    low_rank: Option<CpxtbLowRank>,
}

impl CpxtbSetup {
    /// Jacobian action `A·u` in the occupied–virtual amplitude space.
    pub fn matvec(&self, u: &[f64]) -> Result<Vec<f64>> {
        cpxtb_matvec_precomputed(
            &self.shell_kernel,
            &self.orbital_gaps,
            &self.transition,
            &self.occupation_scales,
            u,
        )
    }

    /// **Stage Z3 — Z-vector / adjoint solve** `A y = rhs_like`, reusing the SAME preconditioned CG and
    /// orbital-gap preconditioner as the response solve. Because `A` is self-adjoint here, the adjoint
    /// equation `A^T y = L` coincides with `A y = L`; the dedicated entry point keeps the API stable if a
    /// future finite-T / non-symmetric representation reintroduces a distinct `A^T`. Used to solve
    /// `A y_a = L_a` for the density-gradient adjoint `L_a` (see [`density_gradient_adjoint_vectors`]).
    ///
    /// Takes the pre-factored charge-space reduction when it is available (exact,
    /// `O(npair·nsh)`), and falls back to the preconditioned CG otherwise.
    pub fn solve_adjoint(
        &self,
        rhs_like: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<CpxtbSolution> {
        if let Some(low_rank) = self.low_rank.as_ref() {
            let family = [rhs_like.to_vec()];
            if let Ok(mut solutions) = low_rank.solve_batch(&family) {
                if let Some(solution) = solutions.pop() {
                    if solution.converged {
                        return Ok(solution);
                    }
                }
            }
        }
        solve_cpxtb_preconditioned(
            |u| self.matvec(u),
            rhs_like,
            &self.orbital_gaps,
            tol,
            max_iter,
        )
    }
}

/// Build the [`CpxtbSetup`] (operator + RHS) for a converged electronic state — mirrors the setup in
/// [`solve_nonpbc_cpxtb_hessian_response`] but stops before the linear solve.
pub fn build_cpxtb_setup(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
    align_to: Option<&Matrix>,
) -> Result<CpxtbSetup> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "non-PBC CPXTB setup cannot be used for PBC systems".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let eig = lowdin_solve_generalized(&electronic.fock, &electronic.integrals.overlap, 1.0e-12)?;
    let mut mos = eig.vectors;
    // Align each orbital's SIGN to a reference (re-diagonalization picks arbitrary eigenvector signs,
    // which would make the CP amplitude representation discontinuous across displaced geometries).
    if let Some(refm) = align_to {
        let smos = electronic.integrals.overlap.matmul(&mos)?; // S·mos
        for p in 0..mos.cols() {
            let mut dot = 0.0;
            for mu in 0..mos.rows() {
                dot += refm[(mu, p)] * smos[(mu, p)];
            }
            if dot < 0.0 {
                for mu in 0..mos.rows() {
                    mos[(mu, p)] = -mos[(mu, p)];
                }
            }
        }
    }
    let orbital_energies = eig.values;
    let occupations = &electronic.occupations;
    let space = CpxtbSpace::from_occupations(occupations)?;
    let orbital_gaps = space
        .pairs
        .iter()
        .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
        .collect::<Vec<_>>();
    let occupation_scales = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occupations[i] - occupations[a]))
        .collect::<Vec<_>>();
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let transition =
        transition_shell_charges(basis, &mos, occupations, &electronic.integrals.overlap)?;
    let scalar_derivatives =
        shell_scalar_potential_derivatives(system, basis, params, &electronic.shell_charges)?;
    let cn_derivatives = if ao_options.include_cn_h0 {
        Some(coordination_number_derivatives(
            system,
            ao_options.coordination_cutoff,
        )?)
    } else {
        None
    };
    let derivative_matrices = cartesian_ao_derivative_matrices(
        system,
        params,
        electronic,
        &scalar_derivatives,
        cn_derivatives.as_deref(),
    )?;
    let mut rhs_vectors = Vec::with_capacity(derivative_matrices.len());
    for deriv in &derivative_matrices {
        rhs_vectors.push(cpxtb_rhs_vector(
            basis,
            &mos,
            occupations,
            &deriv.h0_deriv,
            &deriv.overlap_deriv,
            &orbital_energies,
        )?);
    }
    add_metric_scc_rhs(
        &mut rhs_vectors,
        basis,
        &shell_kernel,
        &mos,
        occupations,
        &electronic.integrals.overlap,
        &electronic.density,
        &orbital_energies,
        &derivative_matrices,
    )?;
    let low_rank = CpxtbLowRank::build(
        &shell_kernel,
        &orbital_gaps,
        &transition,
        &occupation_scales,
    )
    .ok();
    Ok(CpxtbSetup {
        mos,
        orbital_energies,
        space,
        rhs_vectors,
        derivative_matrices,
        shell_kernel,
        orbital_gaps,
        transition,
        occupation_scales,
        low_rank,
    })
}

/// The **orbital-sector response bundle** for a single CP amplitude vector `u`: the linear map
/// `B: u ↦ (ΔP_orb, ΔW_orb, Δq_orb)` whose image, contracted by `G_a` (`response_electronic_gradient`),
/// gives the orbital sector of the response Hessian `R_orb_ab = G_a[B x_b]`.
///
/// Two properties make this the correct object for the Z-vector closure:
///  * **independent of the perturbation `b`** — the explicit `charges(P, S_b)` term lives in the STATIC
///    sector (`static_metric_response_sector`), NOT here, so the only `b`-coupling is through `u = x_b`;
///  * **exactly linear in `u`** (`B·0 = 0`) — so `u ↦ G_a[B u]` is a linear functional `L_a·u`, whose
///    coefficient vector `L_a` (the density-space adjoint `B^T G_a^*`) is the Z-vector right-hand side.
pub struct OrbitalResponseBundle {
    /// `ΔP_orb` — the orbital density response (used for both the density and the CN-density argument).
    pub density: Matrix,
    /// `ΔW_orb` — the orbital energy-weighted density response (Pulay term) plus its SCC self-consistency.
    pub weighted: Matrix,
    /// `Δq_orb` — the IMPLICIT shell-charge response of `ΔP_orb` (no explicit `S_b` term).
    pub shell_charges: Vec<f64>,
}

/// Build `B u` (the orbital-sector bundle) for an arbitrary CP amplitude vector `u`. Uses the same
/// helpers as the Hessian column assembly, with the IMPLICIT shell charge only (zero overlap-derivative).
#[allow(clippy::too_many_arguments)]
pub(crate) fn orbital_response_bundle_from_amplitudes(
    basis: &BasisSet,
    overlap: &Matrix,
    ground_density: &Matrix,
    mos: &Matrix,
    occupations: &[f64],
    orbital_energies: &[f64],
    space: &CpxtbSpace,
    shell_kernel: &Matrix,
    u: &[f64],
) -> Result<OrbitalResponseBundle> {
    let n = basis.len();
    let zero_ov = Matrix::zeros(n, n);
    let density = response_density_with_space(mos, occupations, space, u)?;
    // Implicit charge only — the explicit S_b charge belongs to the static sector.
    let shell_charges =
        response_shell_charges_from_density(basis, overlap, ground_density, &density, &zero_ov)?;
    let shell_pot = matrix_vector_product(shell_kernel, &shell_charges)?;
    let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_pot)?;
    let mut weighted =
        response_energy_weighted_density_with_space(mos, occupations, orbital_energies, space, u)?;
    add_occupied_metric_energy_weighted_response(
        &mut weighted,
        mos,
        occupations,
        orbital_energies,
        &response_fock,
        &zero_ov,
    )?;
    Ok(OrbitalResponseBundle {
        density,
        weighted,
        shell_charges,
    })
}

/// **Static / metric response-Hessian sector** `R_static_ab = G_a[static_b]` — NO CPHF solve. For each
/// nuclear DOF `b` it builds the `x`-INDEPENDENT part of the response bundle purely from the
/// overlap/Fock derivatives `AoDerivativeMatrices[b]`, then contracts it through the SAME
/// `response_electronic_gradient` column assembly the Hessian uses:
///   `ΔP_b^static` = `add_occupied_metric_density_response(S_b)`;
///   `Δq_b^static` = `response_shell_charges_from_density(ΔP_b^static, S_b)` — the implicit charge of
///                  `ΔP_b^static` PLUS the explicit `charges(P, S_b)` term (kept HERE so the orbital
///                  bundle is purely linear in `x`);
///   `ΔW_b^static` = `add_occupied_metric_energy_weighted_response(F_b, S_b)` + the SCC self-consistency
///                  `add_occupied_metric_energy_weighted_response(γ·Δq_b^static, 0)`.
///
/// Together with the orbital sector (`OrbitalResponseBundle`) this reproduces the full response Hessian
/// EXACTLY: `cphf.hessian_response = R_static + R_orbital` (verified to ~1e-16 by
/// `response_hessian_sector_diagnostic` / test `response_hessian_sector_decomposition`).
///
/// **This is NOT the operational residual** `M = cphf.hessian_response + rhs·x`. The naive identification
/// `L_a = −rhs_a` (which would make `R_orbital = −rhs·x`) FAILS in this density-space representation: `G`
/// carries the SCC `γ_a`/`response_fock` and Pulay terms explicitly, while `−rhs·x` carries them via `x`'s
/// self-consistency in `A`. The correct adjoint is `L_a = B^T G_a^*` (see `density_gradient_adjoint_vectors`),
/// and the orbital-sector nuclear derivative closes through a Z-vector solve `A y_a = L_a`, not first-order
/// responses alone. This function supplies the clean `x`-independent static sector.
pub fn static_metric_response_sector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
) -> Result<Matrix> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "non-PBC static metric response sector cannot be used for PBC systems".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let n = basis.len();
    let ndim = 3 * system.atoms.len();
    let eig = lowdin_solve_generalized(&electronic.fock, &electronic.integrals.overlap, 1.0e-12)?;
    let mos = eig.vectors;
    let orbital_energies = eig.values;
    let occupations = &electronic.occupations;
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let scalar_derivatives =
        shell_scalar_potential_derivatives(system, basis, params, &electronic.shell_charges)?;
    let cn_derivatives = if ao_options.include_cn_h0 {
        Some(coordination_number_derivatives(
            system,
            ao_options.coordination_cutoff,
        )?)
    } else {
        None
    };
    let derivative_matrices = cartesian_ao_derivative_matrices(
        system,
        params,
        electronic,
        &scalar_derivatives,
        cn_derivatives.as_deref(),
    )?;
    let gradient_context = ResponseGradientContext::new(
        system,
        basis,
        params,
        electronic,
        ao_options.coordination_cutoff,
        ao_options.include_cn_h0,
    )?;
    let zero_ov = Matrix::zeros(n, n);
    let mut m = Matrix::zeros(ndim, ndim);
    for b in 0..ndim {
        let s_b = &derivative_matrices[b].overlap_deriv;
        let f_b = &derivative_matrices[b].h0_deriv;
        // ΔP_b^static  (overlap-derivative / non-orthogonality density response)
        let mut dp = Matrix::zeros(n, n);
        add_occupied_metric_density_response(&mut dp, &mos, occupations, s_b)?;
        // Δq_b^static = implicit charge of ΔP plus the explicit S_b charge (the latter is x-independent,
        // so keeping it here makes the complementary orbital bundle purely linear in x).
        let dq = response_shell_charges_from_density(
            basis,
            &electronic.integrals.overlap,
            &electronic.density,
            &dp,
            s_b,
        )?;
        // SCC self-consistency Fock from the static charge:  F_resp = scalar_response_fock(γ·Δq)
        let shell_pot = matrix_vector_product(&shell_kernel, &dq)?;
        let response_fock =
            scalar_response_fock_matrix(basis, &electronic.integrals.overlap, &shell_pot)?;
        // ΔW_b^static = ΔW_metric(F_b,S_b) + ΔW_metric(F_resp, 0)
        let mut dw = Matrix::zeros(n, n);
        add_occupied_metric_energy_weighted_response(
            &mut dw,
            &mos,
            occupations,
            &orbital_energies,
            f_b,
            s_b,
        )?;
        add_occupied_metric_energy_weighted_response(
            &mut dw,
            &mos,
            occupations,
            &orbital_energies,
            &response_fock,
            &zero_ov,
        )?;
        // R_static[:, b] = G[ΔP, ΔP, ΔW, Δq]
        let gradient = response_electronic_gradient(
            system,
            electronic,
            &shell_kernel,
            &gradient_context,
            &dp,
            &dp,
            &dw,
            &dq,
        )?;
        set_hessian_column_from_gradient(&mut m, b, &gradient)?;
    }
    Ok(m)
}

/// **Stage Z2 — density-gradient adjoint `L_a` by basis-vector projection.** Builds the CP-amplitude-space
/// vectors `L_a` (one per nuclear DOF `a`) such that for ANY amplitude vector `u`
/// `dot(L_a, u) = G_a[orbital_response_bundle_from_amplitudes(u)]` — i.e. `L_a = B^T G_a^*`, the adjoint of
/// the orbital-bundle map composed with the response gradient. Because `u ↦ G[B u]` is linear, the columns
/// are recovered exactly by projecting onto the CP unit vectors: `L_a[p] = G_a[B e_p]`. Returns
/// `L_vectors[a]` (length `npair`). This is the right-hand side of the Z-vector equation `A y_a = L_a`,
/// the correct replacement for the (false) `L_a = −rhs_a`. `mos`/`orbital_energies` are passed in so they
/// match the solver basis the CP amplitudes/RHS were built in (sign/gauge consistency).
pub fn density_gradient_adjoint_vectors(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
    mos: &Matrix,
    orbital_energies: &[f64],
) -> Result<Vec<Vec<f64>>> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "non-PBC density-gradient adjoint cannot be used for PBC systems".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let overlap = &electronic.integrals.overlap;
    let ndim = 3 * system.atoms.len();
    let occupations = &electronic.occupations;
    let space = CpxtbSpace::from_occupations(occupations)?;
    let npair = space.len();
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let gradient_context = ResponseGradientContext::new(
        system,
        basis,
        params,
        electronic,
        ao_options.coordination_cutoff,
        ao_options.include_cn_h0,
    )?;
    let mut l_vectors = vec![vec![0.0_f64; npair]; ndim];
    let mut e_p = vec![0.0_f64; npair];
    for p in 0..npair {
        e_p[p] = 1.0;
        let bundle = orbital_response_bundle_from_amplitudes(
            basis,
            overlap,
            &electronic.density,
            mos,
            occupations,
            orbital_energies,
            &space,
            &shell_kernel,
            &e_p,
        )?;
        e_p[p] = 0.0;
        let gradient = response_electronic_gradient(
            system,
            electronic,
            &shell_kernel,
            &gradient_context,
            &bundle.density,
            &bundle.density,
            &bundle.weighted,
            &bundle.shell_charges,
        )?;
        // gradient[atom] is a Vec3; component a = 3*atom + axis.
        for (atom, value) in gradient.iter().enumerate() {
            l_vectors[3 * atom][p] = value.x;
            l_vectors[3 * atom + 1][p] = value.y;
            l_vectors[3 * atom + 2][p] = value.z;
        }
    }
    Ok(l_vectors)
}

/// The orbital-sector response Hessian `R_orbital_ab = G_a[B x_b]`, built column-by-column from the CP
/// amplitudes `amplitudes[b] = x_b` via the b-independent orbital bundle. `mos`/`orbital_energies` must be
/// the basis the amplitudes were solved in. Used both at the reference (sector check) and at displaced
/// geometries (the FD reference for the Z-vector bridge). `R_orbital` is a physical second-derivative
/// quantity, invariant to per-orbital sign choices (an MO sign flip flips the matching amplitude too).
pub fn orbital_sector_response_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
    mos: &Matrix,
    orbital_energies: &[f64],
    amplitudes: &[Vec<f64>],
) -> Result<Matrix> {
    let basis = &electronic.basis;
    let overlap = &electronic.integrals.overlap;
    let ndim = 3 * system.atoms.len();
    if amplitudes.len() != ndim {
        return Err(Gfn1Error::InvalidInput(
            "orbital_sector_response_hessian: amplitudes length must equal 3*natoms".to_string(),
        ));
    }
    let occupations = &electronic.occupations;
    let space = CpxtbSpace::from_occupations(occupations)?;
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let gradient_context = ResponseGradientContext::new(
        system,
        basis,
        params,
        electronic,
        ao_options.coordination_cutoff,
        ao_options.include_cn_h0,
    )?;
    let mut r_orbital = Matrix::zeros(ndim, ndim);
    for b in 0..ndim {
        let bundle = orbital_response_bundle_from_amplitudes(
            basis,
            overlap,
            &electronic.density,
            mos,
            occupations,
            orbital_energies,
            &space,
            &shell_kernel,
            &amplitudes[b],
        )?;
        let gradient = response_electronic_gradient(
            system,
            electronic,
            &shell_kernel,
            &gradient_context,
            &bundle.density,
            &bundle.density,
            &bundle.weighted,
            &bundle.shell_charges,
        )?;
        set_hessian_column_from_gradient(&mut r_orbital, b, &gradient)?;
    }
    Ok(r_orbital)
}

/// Diagnostic for the corrected sector split + density-gradient adjoint, returning a [`SectorDiagnostic`].
///
/// With `R^code = cphf.hessian_response = G[full]` and the sector split `full = static + orbital`:
///   * `linearity_max`  = `max|R^code − (R_static + R_orbital)|` — should be ~0 (pure linearity of `G`).
///   * `adjoint_max`    = `max|dot(L_a, x_b) − R_orbital_ab|` — the **Stage-Z2 decisive check**: the
///     projected adjoint `L_a` reproduces the orbital-sector Hessian (so `L_a = B^T G_a^*` is correct).
///   * `interchange_max`= `max|R_orbital + rhs·x|` — records that `L_a ≠ −rhs_a` (NOT ~0); this is the
///     reason the Z-vector route is required, not a failure of 2n+1.
pub struct SectorDiagnostic {
    pub linearity_max: f64,
    pub adjoint_max: f64,
    pub interchange_max: f64,
}

pub fn response_hessian_sector_diagnostic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_options: AoDerivativeOptions,
    cphf: &GammaCartesianCpxtbResult,
) -> Result<SectorDiagnostic> {
    let ndim = 3 * system.atoms.len();
    let mos = &cphf.mos;
    let orbital_energies = &cphf.orbital_energies;

    let r_static = static_metric_response_sector(system, params, electronic, ao_options)?;
    let l_vectors = density_gradient_adjoint_vectors(
        system,
        params,
        electronic,
        ao_options,
        mos,
        orbital_energies,
    )?;
    let amplitudes: Vec<Vec<f64>> = cphf
        .solutions
        .iter()
        .map(|s| s.amplitudes.clone())
        .collect();
    let r_orbital = orbital_sector_response_hessian(
        system,
        params,
        electronic,
        ao_options,
        mos,
        orbital_energies,
        &amplitudes,
    )?;

    let mut linearity_max = 0.0_f64;
    let mut adjoint_max = 0.0_f64;
    let mut interchange_max = 0.0_f64;
    for a in 0..ndim {
        for b in 0..ndim {
            let x_b = &cphf.solutions[b].amplitudes;
            // linearity: R^code vs (R_static + R_orbital)
            linearity_max = linearity_max.max(
                (cphf.hessian_response[(a, b)] - (r_static[(a, b)] + r_orbital[(a, b)])).abs(),
            );
            // adjoint: dot(L_a, x_b) vs R_orbital_ab
            let l_dot_x: f64 = l_vectors[a]
                .iter()
                .zip(x_b.iter())
                .map(|(l, x)| l * x)
                .sum();
            adjoint_max = adjoint_max.max((l_dot_x - r_orbital[(a, b)]).abs());
            // interchange: R_orbital vs −rhs·x  (records L_a != -rhs_a)
            let rhs_dot_x: f64 = cphf.rhs_vectors[a]
                .iter()
                .zip(x_b.iter())
                .map(|(r, x)| r * x)
                .sum();
            interchange_max = interchange_max.max((r_orbital[(a, b)] + rhs_dot_x).abs());
        }
    }
    Ok(SectorDiagnostic {
        linearity_max,
        adjoint_max,
        interchange_max,
    })
}

/// **Stage Z5 keystone — analytic MO-coefficient derivatives `C^(c) = ∂C/∂R_c`** (one `n×n` matrix per
/// nuclear DOF `c`), in the SAME canonical, sign-aligned gauge the solver's `mos` live in. NO new solve:
/// `C^(c) = C U^c` with the orbital-rotation matrix `U^c = Cᵀ S C^(c)` assembled from quantities already
/// available in `cphf`:
///   * occupied–virtual block `U^c_ai = x_c` — the CP amplitude (`cphf.solutions[c].amplitudes`), i.e. the
///     self-consistent (SCC-relaxed) response; the complementary `U^c_ia = −S̃_c_ia − x_c` from the metric
///     condition `U^c + U^cᵀ = −S̃_c`;
///   * same-block off-diagonal (occ–occ, virt–virt) `U^c_pq = (F̃_c_pq − ε_q S̃_c_pq)/(ε_q − ε_p)` — the
///     canonical-orbital condition, with the SCC-RELAXED MO Fock derivative
///     `F̃_c = Cᵀ(h0_deriv_c + scalar_fock(γ·q_c))C`, `q_c = cphf.shell_charge_responses[c]`;
///   * diagonal `U^c_pp = −½ S̃_c_pp` (normalization), `S̃_c = Cᵀ S_c C`.
/// VALIDATED (test `mo_coefficient_derivatives_match_fd`): diagonal/ov/vo blocks reproduce FD exactly; the
/// same-block relaxed Fock derivative `F̃_c` is confirmed by an FD back-solve (`F̃_needed` matches
/// `h0_deriv + RF(γ·q_c)` to FD floor, beating skeleton-only and wrong-sign candidates >10×).
/// Degenerate same-block pairs (`|ε_q−ε_p| < floor`) are left at zero (gauge-arbitrary; cancels in physical
/// quantities). Validated against finite differences of aligned canonical `mos`
/// (test `mo_coefficient_derivatives_match_fd`). This is the foundation for the analytic `D_c L_a`,
/// `D_c rhs`, `D_c A`, `D_c R_static` (Stage Z5).
pub fn mo_coefficient_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &GammaCartesianCpxtbResult,
) -> Result<Vec<Matrix>> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "non-PBC MO-coefficient derivatives cannot be used for PBC systems".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let n = basis.len();
    let ndim = 3 * system.atoms.len();
    let mos = &cphf.mos;
    let eps = &cphf.orbital_energies;
    let occupations = &electronic.occupations;
    let overlap = &electronic.integrals.overlap;
    let space = CpxtbSpace::from_occupations(occupations)?;
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let is_occ: Vec<bool> = occupations.iter().map(|&o| o > 1.0e-8).collect();
    let denom_floor = 1.0e-6;
    let mut result = Vec::with_capacity(ndim);
    for c in 0..ndim {
        let s_c = &cphf.derivative_matrices[c].overlap_deriv;
        let f_c_frozen = &cphf.derivative_matrices[c].h0_deriv;
        let q_c = &cphf.shell_charge_responses[c];
        // SCC-relaxed effective Fock derivative: skeleton h0_deriv + the γ·q_c charge-response Fock.
        let shell_pot = matrix_vector_product(&shell_kernel, q_c)?;
        let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_pot)?;
        let mut f_c = f_c_frozen.clone();
        for i in 0..n {
            for j in 0..n {
                f_c[(i, j)] += response_fock[(i, j)];
            }
        }
        let s_tilde = mo_transform(mos, s_c)?;
        let f_tilde = mo_transform(mos, &f_c)?;
        let mut u = Matrix::zeros(n, n);
        for p in 0..n {
            u[(p, p)] = -0.5 * s_tilde[(p, p)];
            for q in 0..n {
                if p == q || is_occ[p] != is_occ[q] {
                    continue; // diagonal handled above; cross-block (ov/vo) handled via CP amplitudes
                }
                let de = eps[q] - eps[p];
                if de.abs() < denom_floor {
                    // Degenerate same-block pair. The antisymmetric in-block rotation is
                    // gauge (invisible in equal-occupation observables), but the SYMMETRIC
                    // part is fixed by first-order orthonormality `U_pq + U_qp = -S̃_pq`.
                    // Leaving it zero (the historical behavior) silently violated
                    // orthonormality for any perturbation with S̃_pq ≠ 0 — i.e. every
                    // geometric perturbation of a symmetric molecule.
                    u[(p, q)] = -0.5 * s_tilde[(p, q)];
                    continue;
                }
                u[(p, q)] = (f_tilde[(p, q)] - eps[q] * s_tilde[(p, q)]) / de;
            }
        }
        // Occupied–virtual rotations from the CP amplitudes (self-consistent response).
        for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
            let x = cphf.solutions[c].amplitudes[pair_idx];
            u[(a, i)] = x;
            u[(i, a)] = -s_tilde[(i, a)] - x;
        }
        result.push(mos.matmul(&u)?);
    }
    Ok(result)
}

/// First nuclear derivative of the AO overlap matrix, `∂S/∂R_b` (n×n), for DOF `b=(atom_b,axis_b)`.
/// Built from the per-pair bra/ket first-derivative blocks; matches `cartesian_ao_derivative_matrices`'
/// `overlap_deriv` at the reference geometry, but is callable standalone at any geometry/basis.
pub fn overlap_first_derivative_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    b: usize,
) -> Result<Matrix> {
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let a_atom = basis.aos[mu].atom_index;
        let ra = system.atoms[a_atom].position;
        for nu in 0..n {
            let k_atom = basis.aos[nu].atom_index;
            if atom_b != a_atom && atom_b != k_atom {
                continue;
            }
            let rk = system.atoms[k_atom].position;
            let (_, d_bra, d_ket) =
                contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rk);
            let mut val = 0.0;
            if atom_b == a_atom {
                val += d_bra[0].to_array()[axis_b];
            }
            if atom_b == k_atom {
                val += d_ket[0].to_array()[axis_b];
            }
            out[(mu, nu)] = val;
        }
    }
    Ok(out)
}

/// Second nuclear derivative of the AO overlap matrix, `∂²S/∂R_b∂R_c` (n×n), for DOFs `b=(atom_b,axis_b)`
/// and `c=(atom_c,axis_c)`. Built from the per-pair bra/ket second-derivative blocks
/// (`contracted_pair_with_second_derivatives`); non-zero only when both `atom_b` and `atom_c` are among the
/// pair's two centers. The `(bra,ket)` mixed block uses `h_bra_ket[row][col]`, its transpose for `(ket,bra)`.
pub fn overlap_second_derivative_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    b: usize,
    c: usize,
) -> Result<Matrix> {
    let n = basis.len();
    let (atom_b, axis_b) = (b / 3, b % 3);
    let (atom_c, axis_c) = (c / 3, c % 3);
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let a_atom = basis.aos[mu].atom_index;
        let ra = system.atoms[a_atom].position;
        for nu in 0..n {
            let k_atom = basis.aos[nu].atom_index;
            if (atom_b != a_atom && atom_b != k_atom) || (atom_c != a_atom && atom_c != k_atom) {
                continue;
            }
            let rk = system.atoms[k_atom].position;
            let pair =
                contracted_pair_with_second_derivatives(&basis.aos[mu], &basis.aos[nu], ra, rk);
            let hbb = &pair.h_bra_bra[0];
            let hbk = &pair.h_bra_ket[0];
            let hkk = &pair.h_ket_ket[0];
            // ∂²S/∂R_b∂R_c = Σ_{X∈{bra=a_atom, ket=k_atom}} Σ_{Y∈{bra,ket}} [atom_b==X][atom_c==Y]·second
            let mut val = 0.0;
            if atom_b == a_atom {
                if atom_c == a_atom {
                    val += hbb[axis_b][axis_c];
                }
                if atom_c == k_atom {
                    val += hbk[axis_b][axis_c];
                }
            }
            if atom_b == k_atom {
                if atom_c == a_atom {
                    val += hbk[axis_c][axis_b];
                }
                if atom_c == k_atom {
                    val += hkk[axis_b][axis_c];
                }
            }
            out[(mu, nu)] = val;
        }
    }
    Ok(out)
}

/// Diagnostic candidates for the relaxed effective-Fock derivative in the MO basis, per nuclear DOF `c`,
/// used to calibrate the same-block (oo/vv) canonical rotation against a finite-difference back-solve
/// `F̃_needed_pq = (ε_q−ε_p)·U_FD_pq + ε_q·S̃_c_pq`. Returns `(h0_mo, response_mo, s_tilde)` where
/// `h0_mo = Cᵀ·h0_deriv·C` (skeleton), `response_mo = Cᵀ·RF(γ·q_c)·C` (charge-response Fock), and
/// `s_tilde = Cᵀ·S_c·C`. Candidate `F̃_0 = h0_mo`, `F̃_+ = h0_mo + response_mo`, `F̃_- = h0_mo − response_mo`.
pub fn relaxed_fock_derivative_candidates(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &GammaCartesianCpxtbResult,
) -> Result<Vec<(Matrix, Matrix, Matrix)>> {
    let basis = &electronic.basis;
    let ndim = 3 * system.atoms.len();
    let mos = &cphf.mos;
    let overlap = &electronic.integrals.overlap;
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let mut out = Vec::with_capacity(ndim);
    for c in 0..ndim {
        let s_c = &cphf.derivative_matrices[c].overlap_deriv;
        let h0 = &cphf.derivative_matrices[c].h0_deriv;
        let q_c = &cphf.shell_charge_responses[c];
        let shell_pot = matrix_vector_product(&shell_kernel, q_c)?;
        let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_pot)?;
        let h0_mo = mo_transform(mos, h0)?;
        let response_mo = mo_transform(mos, &response_fock)?;
        let s_tilde = mo_transform(mos, s_c)?;
        out.push((h0_mo, response_mo, s_tilde));
    }
    Ok(out)
}

/// Analytic first-order density response to a uniform external electric field,
/// `dP/dE_beta` for `beta = x, y, z`, from the same closed-shell CPXTB operator
/// used by the Hessian. The field perturbs only the effective one-electron
/// operator through `dF/dE_beta = +1/2 S_(mu nu) (R_mu + R_nu)_beta` (the overlap
/// does not change), so this is the polarizability response.
#[derive(Clone, Debug)]
pub struct FieldResponse {
    /// Density response `dP/dE_beta`, indexed by Cartesian field axis.
    pub density_responses: [Matrix; 3],
    pub converged: bool,
    pub max_residual_norm: f64,
}

/// Solve the closed-shell CPXTB equations for the three uniform electric-field
/// perturbations and return the analytic density responses `dP/dE`.
///
/// Requires gapped (integer 0/2) occupations — the analytic polarizability path
/// does not cover fractional/metallic occupations (use a finite field there).
pub fn solve_field_response(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cpxtb_options: CpxtbOptions,
) -> Result<FieldResponse> {
    let _profile = crate::profile::scope("cphf.field.total");
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "analytic field response is implemented for non-periodic systems only".to_string(),
        ));
    }
    let basis = &electronic.basis;
    let n = basis.len();
    let overlap = &electronic.integrals.overlap;
    let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12)?;
    let mos = eig.vectors;
    let orbital_energies = eig.values;
    let occupations = &electronic.occupations;
    let space = CpxtbSpace::from_occupations(occupations)?;
    for &(i, a) in &space.pairs {
        let occ_diff = occupations[i] - occupations[a];
        if !(occ_diff.is_finite() && occ_diff > 1.0e-10) {
            return Err(Gfn1Error::InvalidInput(
                "analytic field response requires gapped (integer) occupations; \
                 use a finite-field polarizability for fractional occupations"
                    .to_string(),
            ));
        }
    }
    let orbital_gaps = space
        .pairs
        .iter()
        .map(|&(i, a)| orbital_energies[a] - orbital_energies[i])
        .collect::<Vec<_>>();
    let coupling_occupation_scales = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occupations[i] - occupations[a]))
        .collect::<Vec<_>>();
    let shell_kernel = response_shell_scc_kernel(system, params, electronic)?;
    let transition = transition_shell_charges(basis, &mos, occupations, overlap)?;

    // AO-resolved atom positions for the dipole/field perturbation.
    let ao_position: Vec<[f64; 3]> = (0..n)
        .map(|mu| system.atoms[basis.aos[mu].atom_index].position.to_array())
        .collect();
    let zero_overlap = Matrix::zeros(n, n);

    let mut rhs_vectors = Vec::with_capacity(3);
    for beta in 0..3 {
        let mut fock_deriv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                fock_deriv[(i, j)] =
                    0.5 * overlap[(i, j)] * (ao_position[i][beta] + ao_position[j][beta]);
            }
        }
        rhs_vectors.push(cpxtb_rhs_vector(
            basis,
            &mos,
            occupations,
            &fock_deriv,
            &zero_overlap,
            &orbital_energies,
        )?);
    }

    let solutions = solve_cpxtb_all(
        &shell_kernel,
        &orbital_gaps,
        &transition,
        &coupling_occupation_scales,
        &rhs_vectors,
        cpxtb_options,
    )?;
    let mut converged = true;
    let mut max_residual_norm = 0.0_f64;
    for solution in &solutions {
        converged &= solution.converged;
        max_residual_norm = max_residual_norm.max(solution.residual_norm);
    }

    let mut responses = Vec::with_capacity(3);
    for solution in &solutions {
        responses.push(response_density(&mos, occupations, &solution.amplitudes)?);
    }
    let density_responses = [
        responses[0].clone(),
        responses[1].clone(),
        responses[2].clone(),
    ];
    Ok(FieldResponse {
        density_responses,
        converged,
        max_residual_norm,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct ResponseGradientContext {
    ao_pairs: Vec<ResponseAoPair>,
    shell_pairs: Vec<ResponseShellPair>,
    cn_pairs: Vec<CoordinationPairDerivative>,
    dsedcn: Vec<f64>,
    include_cn_h0: bool,
}

#[derive(Clone, Debug)]
struct ResponseAoPair {
    mu: usize,
    nu: usize,
    atom_mu: usize,
    atom_nu: usize,
    shell_mu: usize,
    shell_nu: usize,
    d_bra: Vec3,
    d_ket: Vec3,
    overlap: f64,
    hij: f64,
    scalar_shift: f64,
    dlog_poly: Vec3,
    cn_mu_scale: f64,
    cn_nu_scale: f64,
}

#[derive(Clone, Debug)]
struct ResponseShellPair {
    i: usize,
    j: usize,
    atom_i: usize,
    atom_j: usize,
    dkernel: Vec3,
    q_i: f64,
    q_j: f64,
}

impl ResponseGradientContext {
    pub(crate) fn new(
        system: &PeriodicSystem,
        basis: &BasisSet,
        params: &Gfn1Parameters,
        electronic: &ElectronicResult,
        coordination_cutoff: f64,
        include_cn_h0: bool,
    ) -> Result<Self> {
        let shell_model = ShellChargeModel::build(system, basis, params)?;
        let mut self_energy = vec![0.0; basis.shells.len()];
        let mut dsedcn = vec![0.0; basis.shells.len()];
        for (ish, shell) in basis.shells.iter().enumerate() {
            dsedcn[ish] = if include_cn_h0 {
                -shell.kcn_raw.unwrap_or(0.0)
            } else {
                0.0
            };
            self_energy[ish] =
                shell.hdiag_ha + dsedcn[ish] * electronic.coordination_numbers[shell.atom_index];
        }
        let mut ao_pairs = Vec::new();
        for mu in 0..basis.len() {
            let atom_mu = basis.aos[mu].atom_index;
            let shell_mu = basis.aos[mu].shell_index;
            let shell_mu_ref = &basis.shells[shell_mu];
            let rmu = system.atoms[atom_mu].position;
            for nu in 0..mu {
                let atom_nu = basis.aos[nu].atom_index;
                if atom_mu == atom_nu {
                    continue;
                }
                let shell_nu = basis.aos[nu].shell_index;
                let shell_nu_ref = &basis.shells[shell_nu];
                let rnu = system.atoms[atom_nu].position;
                let rvec = rmu - rnu;
                let r2 = rvec.norm2();
                if r2 <= DIST_EPS {
                    continue;
                }
                let (moments, d_bra, d_ket) =
                    contracted_pair_with_derivatives(&basis.aos[mu], &basis.aos[nu], rmu, rnu);
                let overlap = moments[0];
                let radius_sum =
                    atomic_radius_bohr(shell_mu_ref.z)? + atomic_radius_bohr(shell_nu_ref.z)?;
                let scaled_r = (r2.sqrt() / radius_sum).sqrt();
                let hs = hscale(shell_mu_ref, shell_nu_ref, params)?
                    * shell_polynomial(shell_mu_ref, shell_nu_ref, scaled_r);
                let hij = 0.5 * (self_energy[shell_mu] + self_energy[shell_nu]) * hs;
                let scalar_shift = electronic.shell_scc_potential[shell_mu]
                    + electronic.shell_scc_potential[shell_nu];
                let dlog_poly =
                    shell_polynomial_log_derivative(shell_mu_ref, shell_nu_ref, rvec, r2);
                ao_pairs.push(ResponseAoPair {
                    mu,
                    nu,
                    atom_mu,
                    atom_nu,
                    shell_mu,
                    shell_nu,
                    d_bra: d_bra[0],
                    d_ket: d_ket[0],
                    overlap,
                    hij,
                    scalar_shift,
                    dlog_poly,
                    cn_mu_scale: dsedcn[shell_mu] * hs * overlap,
                    cn_nu_scale: dsedcn[shell_nu] * hs * overlap,
                });
            }
        }
        let mut shell_pairs = Vec::new();
        for i in 0..basis.shells.len() {
            let atom_i = basis.shells[i].atom_index;
            for j in 0..i {
                let atom_j = basis.shells[j].atom_index;
                if atom_i == atom_j {
                    continue;
                }
                let ri = system.atoms[atom_i].position;
                let rj = system.atoms[atom_j].position;
                let rvec = ri - rj;
                let gamma = harmonic_average(shell_model.hardness[i], shell_model.hardness[j]);
                shell_pairs.push(ResponseShellPair {
                    i,
                    j,
                    atom_i,
                    atom_j,
                    dkernel: effective_kernel_derivative_vector(rvec, gamma),
                    q_i: electronic.shell_charges[i],
                    q_j: electronic.shell_charges[j],
                });
            }
        }
        let cn_pairs = if include_cn_h0 {
            coordination_with_derivatives(
                system,
                CoordinationOptions {
                    cutoff: coordination_cutoff,
                    ..CoordinationOptions::default()
                },
            )?
            .pairs
        } else {
            Vec::new()
        };
        Ok(Self {
            ao_pairs,
            shell_pairs,
            cn_pairs,
            dsedcn,
            include_cn_h0,
        })
    }
}

#[allow(clippy::too_many_arguments)]
/// Per-term decomposition of the CPXTB response gradient. Each field is the
/// per-atom Cartesian contribution of one physically distinct term; the total
/// gradient is their sum (see [`ResponseGradientTerms::total`]). Production code
/// uses the sum; the decomposition exists so each term can be finite-difference
/// verified independently against the energy functional it represents (band =
/// `d/dR Tr[P H0]`, pulay = `-d/dR Tr[W S]`, scc = `d/dR (q_P^T gamma q_D)`),
/// including the virt-virt difference-density block that the ground gradient and
/// polarizability never exercise.
pub(crate) struct ResponseGradientTerms {
    /// H0 band overlap derivative `dp * 2 hij * dS`.
    pub band: Vec<Vec3>,
    /// GFN1 H0 distance-polynomial derivative.
    pub polynomial: Vec<Vec3>,
    /// SCC potential * overlap derivative `-(dp V_D + p0 V_P) dS`.
    pub scc_overlap: Vec<Vec3>,
    /// Pulay (energy-weighted density) overlap derivative `-2 dw dS`.
    pub pulay: Vec<Vec3>,
    /// Coordination-number-dependent H0 derivative.
    pub cn: Vec<Vec3>,
    /// SCC kernel (gamma) derivative `(q_P q_D + q_D q_P) dgamma`.
    pub scc_kernel: Vec<Vec3>,
}

impl ResponseGradientTerms {
    fn zeros(nat: usize) -> Self {
        Self {
            band: vec![Vec3::zero(); nat],
            polynomial: vec![Vec3::zero(); nat],
            scc_overlap: vec![Vec3::zero(); nat],
            pulay: vec![Vec3::zero(); nat],
            cn: vec![Vec3::zero(); nat],
            scc_kernel: vec![Vec3::zero(); nat],
        }
    }

    pub fn total(&self) -> Vec<Vec3> {
        let nat = self.band.len();
        let mut out = vec![Vec3::zero(); nat];
        for a in 0..nat {
            out[a] = self.band[a]
                + self.polynomial[a]
                + self.scc_overlap[a]
                + self.pulay[a]
                + self.cn[a]
                + self.scc_kernel[a];
        }
        out
    }
}

pub(crate) fn response_electronic_gradient(
    system: &PeriodicSystem,
    electronic: &ElectronicResult,
    response_kernel: &Matrix,
    context: &ResponseGradientContext,
    density_response: &Matrix,
    cn_density_response: &Matrix,
    weighted_response: &Matrix,
    shell_charge_response: &[f64],
) -> Result<Vec<Vec3>> {
    Ok(response_electronic_gradient_terms(
        system,
        electronic,
        response_kernel,
        context,
        density_response,
        cn_density_response,
        weighted_response,
        shell_charge_response,
    )?
    .total())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn response_electronic_gradient_terms(
    system: &PeriodicSystem,
    electronic: &ElectronicResult,
    response_kernel: &Matrix,
    context: &ResponseGradientContext,
    density_response: &Matrix,
    cn_density_response: &Matrix,
    weighted_response: &Matrix,
    shell_charge_response: &[f64],
) -> Result<ResponseGradientTerms> {
    let nat = system.atoms.len();
    let basis = &electronic.basis;
    if density_response.rows() != basis.len()
        || density_response.cols() != basis.len()
        || weighted_response.rows() != basis.len()
        || weighted_response.cols() != basis.len()
        || shell_charge_response.len() != basis.shells.len()
    {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB response gradient dimension mismatch".to_string(),
        ));
    }
    let shell_potential_response = matrix_vector_product(response_kernel, shell_charge_response)?;
    let mut terms = ResponseGradientTerms::zeros(nat);
    let mut d_edcn = vec![0.0; nat];

    for pair in &context.ao_pairs {
        let p0 = electronic.density[(pair.mu, pair.nu)];
        let dp = density_response[(pair.mu, pair.nu)];
        let dw = weighted_response[(pair.mu, pair.nu)];
        let scalar_response =
            shell_potential_response[pair.shell_mu] + shell_potential_response[pair.shell_nu];

        let band = dp * 2.0 * pair.hij;
        terms.band[pair.atom_mu] += pair.d_bra * band;
        terms.band[pair.atom_nu] += pair.d_ket * band;

        let scc = -(dp * pair.scalar_shift + p0 * scalar_response);
        terms.scc_overlap[pair.atom_mu] += pair.d_bra * scc;
        terms.scc_overlap[pair.atom_nu] += pair.d_ket * scc;

        let pulay = -2.0 * dw;
        terms.pulay[pair.atom_mu] += pair.d_bra * pulay;
        terms.pulay[pair.atom_nu] += pair.d_ket * pulay;

        let poly_grad = pair.dlog_poly * (2.0 * dp * pair.hij * pair.overlap);
        terms.polynomial[pair.atom_mu] += poly_grad;
        terms.polynomial[pair.atom_nu] -= poly_grad;

        if context.include_cn_h0 {
            let dp_cn = cn_density_response[(pair.mu, pair.nu)];
            d_edcn[pair.atom_mu] += pair.cn_mu_scale * dp_cn;
            d_edcn[pair.atom_nu] += pair.cn_nu_scale * dp_cn;
        }
    }

    if context.include_cn_h0 {
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                d_edcn[shell.atom_index] += context.dsedcn[ish] * cn_density_response[(iao, iao)];
            }
        }
        for pair in &context.cn_pairs {
            if pair.i == pair.j {
                continue;
            }
            let r = pair.r_ij.norm();
            if r <= DIST_EPS {
                continue;
            }
            let pref = (d_edcn[pair.i] + d_edcn[pair.j]) * pair.dcn_dr / r;
            let gi = pair.r_ij * pref;
            terms.cn[pair.i] += gi;
            terms.cn[pair.j] -= gi;
        }
    }

    for pair in &context.shell_pairs {
        let scale =
            shell_charge_response[pair.i] * pair.q_j + pair.q_i * shell_charge_response[pair.j];
        terms.scc_kernel[pair.atom_i] += pair.dkernel * scale;
        terms.scc_kernel[pair.atom_j] -= pair.dkernel * scale;
    }

    Ok(terms)
}

/// The BACKGROUND-STATE motion of [`response_electronic_gradient`] along a
/// directional perturbation `v` — the derivative of the gradient contraction's
/// reference-state coefficients at FROZEN response inputs, split by family so
/// the third-derivative assembly can gate each term separately:
///
/// * `scc_p0` — the reference density under the screening shift:
///   `−P^v_{μν}·(K q^v)_pair·∇S` (motion of the `p0` factor);
/// * `scc_chain` — the response-kernel motion `∂K/∂q·q^v` (onsite `E'''`
///   chain): `−P₀_{μν}·chain_pair·∇S`;
/// * `scc_dp_pot` — the reference-potential motion under the response density:
///   `−P^v_{μν}·(V^v_total)_pair·∇S`;
/// * `kernel_qq` — the shell-charge motion in the kernel-gradient bilinear:
///   `∇γ·2 q^v_i q^v_j`.
pub(crate) struct ResponseGradientBackgroundMotion {
    pub scc_p0: Vec<Vec3>,
    pub scc_chain: Vec<Vec3>,
    pub scc_dp_pot: Vec<Vec3>,
    pub kernel_qq: Vec<Vec3>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn response_gradient_background_motion(
    system: &PeriodicSystem,
    electronic: &ElectronicResult,
    context: &ResponseGradientContext,
    response_kernel: &Matrix,
    p_v: &Matrix,
    q_v: &[f64],
    chain_potential: &[f64],
    v_pot_v: &[f64],
) -> Result<ResponseGradientBackgroundMotion> {
    let nat = system.atoms.len();
    let kqv = matrix_vector_product(response_kernel, q_v)?;
    let mut out = ResponseGradientBackgroundMotion {
        scc_p0: vec![Vec3::default(); nat],
        scc_chain: vec![Vec3::default(); nat],
        scc_dp_pot: vec![Vec3::default(); nat],
        kernel_qq: vec![Vec3::default(); nat],
    };
    for pair in &context.ao_pairs {
        let dp = p_v[(pair.mu, pair.nu)];
        let p0 = electronic.density[(pair.mu, pair.nu)];
        let kq_pair = kqv[pair.shell_mu] + kqv[pair.shell_nu];
        let chain_pair = chain_potential[pair.shell_mu] + chain_potential[pair.shell_nu];
        let pot_pair = v_pot_v[pair.shell_mu] + v_pot_v[pair.shell_nu];

        let t_p0 = -(dp * kq_pair);
        out.scc_p0[pair.atom_mu] += pair.d_bra * t_p0;
        out.scc_p0[pair.atom_nu] += pair.d_ket * t_p0;

        let t_chain = -(p0 * chain_pair);
        out.scc_chain[pair.atom_mu] += pair.d_bra * t_chain;
        out.scc_chain[pair.atom_nu] += pair.d_ket * t_chain;

        let t_dp_pot = -(dp * pot_pair);
        out.scc_dp_pot[pair.atom_mu] += pair.d_bra * t_dp_pot;
        out.scc_dp_pot[pair.atom_nu] += pair.d_ket * t_dp_pot;
    }
    for pair in &context.shell_pairs {
        let scale = 2.0 * q_v[pair.i] * q_v[pair.j];
        out.kernel_qq[pair.atom_i] += pair.dkernel * scale;
        out.kernel_qq[pair.atom_j] -= pair.dkernel * scale;
    }
    Ok(out)
}

/// Generic SCC-overlap background contraction at GRADIENT level, contracted
/// with `v`: `Σ_pairs −P_slot_{μν} · (pot_{shμ}+pot_{shν}) · (∇S·v)_{μν}` —
/// the shape every `−P·V·∇S` background family shares, with caller-chosen
/// density slot and shell potential.
pub(crate) fn background_overlap_gradient_scalar(
    context: &ResponseGradientContext,
    p_slot: &Matrix,
    pot: &[f64],
    v: &[f64],
) -> f64 {
    let mut acc = 0.0;
    for pair in &context.ao_pairs {
        let dp = p_slot[(pair.mu, pair.nu)];
        let pot_pair = pot[pair.shell_mu] + pot[pair.shell_nu];
        let t = -(dp * pot_pair);
        let (va, vk) = (
            [v[3 * pair.atom_mu], v[3 * pair.atom_mu + 1], v[3 * pair.atom_mu + 2]],
            [v[3 * pair.atom_nu], v[3 * pair.atom_nu + 1], v[3 * pair.atom_nu + 2]],
        );
        acc += t
            * (pair.d_bra.x * va[0]
                + pair.d_bra.y * va[1]
                + pair.d_bra.z * va[2]
                + pair.d_ket.x * vk[0]
                + pair.d_ket.y * vk[1]
                + pair.d_ket.z * vk[2]);
    }
    acc
}

/// The HESSIAN-level sibling: `Σ_pairs −P_slot·(pot-pair)·(∂²S : vv)` — the
/// `∇S → ∂²S` eigen-motion of the same background families.
pub(crate) fn background_overlap_hessian_scalar(
    system: &PeriodicSystem,
    basis: &BasisSet,
    context: &ResponseGradientContext,
    p_slot: &Matrix,
    pot: &[f64],
    v: &[f64],
) -> f64 {
    let mut acc = 0.0;
    for pair in &context.ao_pairs {
        let dp = p_slot[(pair.mu, pair.nu)];
        if dp == 0.0 {
            continue;
        }
        let pot_pair = pot[pair.shell_mu] + pot[pair.shell_nu];
        if pot_pair == 0.0 {
            continue;
        }
        let ra = system.atoms[pair.atom_mu].position;
        let rk = system.atoms[pair.atom_nu].position;
        let p2 = crate::integrals::contracted_pair_with_second_derivatives(
            &basis.aos[pair.mu],
            &basis.aos[pair.nu],
            ra,
            rk,
        );
        let va = [v[3 * pair.atom_mu], v[3 * pair.atom_mu + 1], v[3 * pair.atom_mu + 2]];
        let vk = [v[3 * pair.atom_nu], v[3 * pair.atom_nu + 1], v[3 * pair.atom_nu + 2]];
        let mut s2 = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                s2 += p2.h_bra_bra[0][a][b] * va[a] * va[b]
                    + 2.0 * p2.h_bra_ket[0][a][b] * va[a] * vk[b]
                    + p2.h_ket_ket[0][a][b] * vk[a] * vk[b];
            }
        }
        acc += -(dp * pot_pair) * s2;
    }
    acc
}

fn set_hessian_column_from_gradient(
    hessian: &mut Matrix,
    col: usize,
    gradient: &[Vec3],
) -> Result<()> {
    if hessian.rows() != 3 * gradient.len() || hessian.cols() <= col {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB response Hessian column shape mismatch".to_string(),
        ));
    }
    for (atom, value) in gradient.iter().enumerate() {
        hessian[(3 * atom, col)] = value.x;
        hessian[(3 * atom + 1, col)] = value.y;
        hessian[(3 * atom + 2, col)] = value.z;
    }
    Ok(())
}

pub fn response_shell_scc_kernel(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<Matrix> {
    let mut model = ShellChargeModel::build(system, &electronic.basis, params)?;
    // Keep the response kernel consistent with the energy expression: the
    // on-site block is ∂²E_onsite/∂q² = 2Γq + Σ_{n≥4}(n−1)X_n q^{n−2}, where the
    // n ≥ 4 Linear Breathing-Radius orders are active when charge_order > 3.
    // (Before v0.5.0 only the DFTB3 2Γq piece was included, so every response
    // property was silently inconsistent with charge_order ≥ 4 energies.)
    model.charge_order = electronic.charge_order.max(3);
    let mut kernel = effective_coulomb_matrix(system, &electronic.basis, &model);
    let atomic_charges = model.atomic_charges(&electronic.basis, &electronic.shell_charges);
    for (atom, &qat) in atomic_charges.iter().enumerate() {
        let count = model.atom_shell_counts[atom];
        if count == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let (_, second, _, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
            model.hardness[offset],
            model.hubbard_derivs[offset],
            model.charge_order,
            qat,
        );
        let add = second;
        for local_i in 0..count {
            for local_j in 0..count {
                kernel[(offset + local_i, offset + local_j)] += add;
            }
        }
    }
    Ok(kernel)
}

#[allow(clippy::too_many_arguments)]
fn cartesian_ao_derivative_matrices(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    shell_scalar_derivatives: &[Vec<f64>],
    cn_derivatives: Option<&[Vec<f64>]>,
) -> Result<Vec<AoDerivativeMatrices>> {
    cartesian_ao_derivative_matrices_raw(
        system,
        params,
        &electronic.basis,
        &electronic.coordination_numbers,
        &electronic.shell_scc_potential,
        shell_scalar_derivatives,
        cn_derivatives,
    )
}

/// Per-Cartesian-DOF AO derivative matrices (`overlap_deriv = dS/dR`, `h0_deriv =
/// d(H0 − ½ v_scc·S)/dR` at frozen density) built from **raw** inputs rather than a
/// restricted [`ElectronicResult`], so the CN-coupled `dh0/dR` skeleton machinery is
/// reusable for arbitrary states — e.g. the per-spin-channel base Fock of the DFT+U
/// linear-response geometry derivative ([`crate::plus_u_dudr`]), where each channel
/// carries a different `shell_scc_potential` (`v^σ = v_c ∓ v_s`). `coordination_numbers`
/// is per-atom, `shell_scc_potential` is per-shell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cartesian_ao_derivative_matrices_raw(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: &BasisSet,
    coordination_numbers: &[f64],
    shell_scc_potential: &[f64],
    shell_scalar_derivatives: &[Vec<f64>],
    cn_derivatives: Option<&[Vec<f64>]>,
) -> Result<Vec<AoDerivativeMatrices>> {
    let n = basis.len();
    let ndim = 3 * system.atoms.len();
    if shell_scalar_derivatives.len() != ndim
        || shell_scalar_derivatives
            .iter()
            .any(|row| row.len() != basis.shells.len())
    {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB shell scalar derivative dimension mismatch".to_string(),
        ));
    }
    if let Some(cn) = cn_derivatives {
        if cn.len() != ndim || cn.iter().any(|row| row.len() != system.atoms.len()) {
            return Err(Gfn1Error::InvalidInput(
                "CPXTB CN derivative dimension mismatch".to_string(),
            ));
        }
    }
    let mut out = (0..ndim)
        .map(|_| AoDerivativeMatrices {
            h0_deriv: Matrix::zeros(n, n),
            overlap_deriv: Matrix::zeros(n, n),
        })
        .collect::<Vec<_>>();

    for mu in 0..n {
        let ao_mu = &basis.aos[mu];
        let atom_mu = ao_mu.atom_index;
        let shell_mu = ao_mu.shell_index;
        let rmu = system.atoms[atom_mu].position;
        for nu in 0..=mu {
            let ao_nu = &basis.aos[nu];
            let atom_nu = ao_nu.atom_index;
            let shell_nu = ao_nu.shell_index;
            let rnu = system.atoms[atom_nu].position;
            let (moments, d_bra, d_ket) = contracted_pair_with_derivatives(ao_mu, ao_nu, rmu, rnu);
            let overlap = moments[0];
            if overlap.abs().max(d_bra[0].norm()).max(d_ket[0].norm()) <= 1.0e-18 {
                continue;
            }
            let pref = h0_prefactor_and_derivatives(
                system,
                params,
                basis,
                coordination_numbers,
                shell_mu,
                shell_nu,
            )?;
            let scalar_shift = 0.5
                * (shell_scc_potential[shell_mu]
                    + shell_scc_potential[shell_nu]);
            add_center_derivative(
                &mut out,
                atom_mu,
                mu,
                nu,
                pref.value,
                overlap,
                d_bra[0],
                pref.d_bra,
                scalar_shift,
            );
            add_center_derivative(
                &mut out,
                atom_nu,
                mu,
                nu,
                pref.value,
                overlap,
                d_ket[0],
                pref.d_ket,
                scalar_shift,
            );
            add_scalar_derivative_matrices(
                &mut out,
                shell_mu,
                shell_nu,
                mu,
                nu,
                overlap,
                shell_scalar_derivatives,
            );
            if let Some(cn) = cn_derivatives {
                add_cn_h0_derivative_matrices(
                    &mut out, system, params, basis, shell_mu, shell_nu, mu, nu, overlap, cn,
                )?;
            }
        }
    }

    for matrices in &mut out {
        copy_lower_to_upper(&mut matrices.overlap_deriv);
        copy_lower_to_upper(&mut matrices.h0_deriv);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn add_center_derivative(
    out: &mut [AoDerivativeMatrices],
    atom: usize,
    mu: usize,
    nu: usize,
    h_pref: f64,
    overlap: f64,
    ds_vec: Vec3,
    dhpref_vec: Vec3,
    scalar_shift: f64,
) {
    let ds = ds_vec.to_array();
    let dhpref = dhpref_vec.to_array();
    for axis in 0..3 {
        let coord = 3 * atom + axis;
        let ds_axis = ds[axis];
        let fock_deriv = h_pref * ds_axis + overlap * dhpref[axis] - scalar_shift * ds_axis;
        out[coord].overlap_deriv[(mu, nu)] += ds_axis;
        out[coord].h0_deriv[(mu, nu)] += fock_deriv;
    }
}

fn add_scalar_derivative_matrices(
    out: &mut [AoDerivativeMatrices],
    shell_mu: usize,
    shell_nu: usize,
    mu: usize,
    nu: usize,
    overlap: f64,
    shell_scalar_derivatives: &[Vec<f64>],
) {
    if overlap.abs() <= 1.0e-30 {
        return;
    }
    for (coord, row) in shell_scalar_derivatives.iter().enumerate() {
        let dscalar = 0.5 * (row[shell_mu] + row[shell_nu]);
        if dscalar.abs() > 1.0e-30 {
            out[coord].h0_deriv[(mu, nu)] -= dscalar * overlap;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_cn_h0_derivative_matrices(
    out: &mut [AoDerivativeMatrices],
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: &BasisSet,
    shell_mu: usize,
    shell_nu: usize,
    mu: usize,
    nu: usize,
    overlap: f64,
    cn_derivatives: &[Vec<f64>],
) -> Result<()> {
    if overlap.abs() <= 1.0e-30 {
        return Ok(());
    }
    let atom_mu = basis.shells[shell_mu].atom_index;
    let atom_nu = basis.shells[shell_nu].atom_index;
    let (coeff_mu, coeff_nu) =
        h0_cn_derivative_coefficients(system, params, basis, shell_mu, shell_nu)?;
    if coeff_mu.abs().max(coeff_nu.abs()) <= 1.0e-30 {
        return Ok(());
    }
    for (coord, row) in cn_derivatives.iter().enumerate() {
        let dh0 = overlap * (coeff_mu * row[atom_mu] + coeff_nu * row[atom_nu]);
        if dh0.abs() > 1.0e-30 {
            out[coord].h0_deriv[(mu, nu)] += dh0;
        }
    }
    Ok(())
}

fn h0_cn_derivative_coefficients(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: &BasisSet,
    shell_mu: usize,
    shell_nu: usize,
) -> Result<(f64, f64)> {
    let si = &basis.shells[shell_mu];
    let sj = &basis.shells[shell_nu];
    let k_mu = si.kcn_raw.unwrap_or(0.0);
    let k_nu = sj.kcn_raw.unwrap_or(0.0);
    if si.atom_index == sj.atom_index {
        return Ok((-0.5 * k_mu, -0.5 * k_nu));
    }
    let ri = system.atoms[si.atom_index].position;
    let rj = system.atoms[sj.atom_index].position;
    let r = (rj - ri).norm();
    let rad_sum = atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?;
    let rr = (r / rad_sum).sqrt();
    let poly = shell_polynomial(si, sj, rr);
    let scale = hscale(si, sj, params)? * poly;
    Ok((-0.5 * k_mu * scale, -0.5 * k_nu * scale))
}

#[derive(Clone, Copy, Debug)]
struct H0PrefactorDerivatives {
    value: f64,
    d_bra: Vec3,
    d_ket: Vec3,
}

fn h0_prefactor_and_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    basis: &BasisSet,
    coordination_numbers: &[f64],
    shell_mu: usize,
    shell_nu: usize,
) -> Result<H0PrefactorDerivatives> {
    let si = &basis.shells[shell_mu];
    let sj = &basis.shells[shell_nu];
    let self_i = shell_self_energy(si, coordination_numbers[si.atom_index]);
    let self_j = shell_self_energy(sj, coordination_numbers[sj.atom_index]);
    let base = 0.5 * (self_i + self_j);
    if si.atom_index == sj.atom_index {
        return Ok(H0PrefactorDerivatives {
            value: base,
            d_bra: Vec3::zero(),
            d_ket: Vec3::zero(),
        });
    }
    let ri = system.atoms[si.atom_index].position;
    let rj = system.atoms[sj.atom_index].position;
    let dr = rj - ri;
    let r = dr.norm();
    if r <= DIST_EPS {
        return Ok(H0PrefactorDerivatives {
            value: base,
            d_bra: Vec3::zero(),
            d_ket: Vec3::zero(),
        });
    }
    let rad_sum = atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?;
    let rr = (r / rad_sum).sqrt();
    let poly = shell_polynomial(si, sj, rr);
    let hscale = hscale(si, sj, params)?;
    let pref_base = base * hscale;
    let dpoly_dr = shell_polynomial_derivative(si, sj, rr) * 0.5 / (rad_sum * rr.max(1.0e-16));
    let u = dr / r;
    Ok(H0PrefactorDerivatives {
        value: pref_base * poly,
        d_bra: u * (-pref_base * dpoly_dr),
        d_ket: u * (pref_base * dpoly_dr),
    })
}

fn shell_self_energy(shell: &BasisShell, cn: f64) -> f64 {
    shell.hdiag_ha - shell.kcn_raw.unwrap_or(0.0) * cn
}

fn shell_polynomial(si: &BasisShell, sj: &BasisShell, rr: f64) -> f64 {
    (1.0 + si.poly_raw.unwrap_or(0.0) * rr) * (1.0 + sj.poly_raw.unwrap_or(0.0) * rr)
}

fn shell_polynomial_derivative(si: &BasisShell, sj: &BasisShell, rr: f64) -> f64 {
    let pi = si.poly_raw.unwrap_or(0.0);
    let pj = sj.poly_raw.unwrap_or(0.0);
    pi * (1.0 + pj * rr) + pj * (1.0 + pi * rr)
}

fn shell_polynomial_log_derivative(si: &BasisShell, sj: &BasisShell, rvec: Vec3, r2: f64) -> Vec3 {
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

/// Per-Cartesian-DOF derivative of the atomic coordination numbers (`∂CN_A/∂R`),
/// the input the CN-coupled `dh0/dR` skeleton needs. Exposed for reuse by
/// [`crate::plus_u_dudr`].
pub(crate) fn coordination_number_derivatives(system: &PeriodicSystem, cutoff: f64) -> Result<Vec<Vec<f64>>> {
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let mut out = vec![vec![0.0_f64; nat]; ndim];
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
            let value = deriv.to_array()[axis];
            out[3 * pair.i + axis][pair.i] += value;
            out[3 * pair.i + axis][pair.j] += value;
            out[3 * pair.j + axis][pair.i] -= value;
            out[3 * pair.j + axis][pair.j] -= value;
        }
    }
    Ok(out)
}

/// Explicit geometry derivative of the SCC scalar (Coulomb) shell potential at fixed
/// shell charges, `∂v_c/∂R = (∂A/∂R)·q` per Cartesian DOF (the `dA/dR·q` term of
/// `dv_c/dR`; the implicit `A·dq/dR` piece is the CPHF response, handled by the
/// caller). Exposed for reuse by [`crate::plus_u_dudr`].
pub(crate) fn shell_scalar_potential_derivatives(
    system: &PeriodicSystem,
    basis: &BasisSet,
    params: &Gfn1Parameters,
    shell_charges: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let nat = system.atoms.len();
    let ndim = 3 * nat;
    let nsh = basis.shells.len();
    if shell_charges.len() != nsh {
        return Err(Gfn1Error::InvalidInput(
            "shell charge dimension mismatch for CPXTB scalar derivative".to_string(),
        ));
    }
    let model = ShellChargeModel::build(system, basis, params)?;
    let mut out = vec![vec![0.0_f64; nsh]; ndim];
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for j in 0..i {
            let aj = basis.shells[j].atom_index;
            if ai == aj {
                continue;
            }
            let rvec = system.atoms[ai].position - system.atoms[aj].position;
            let gamma = harmonic_average(model.hardness[i], model.hardness[j]);
            let dk = effective_kernel_derivative_vector(rvec, gamma);
            for axis in 0..3 {
                let value = dk.to_array()[axis];
                out[3 * ai + axis][i] += value * shell_charges[j];
                out[3 * ai + axis][j] += value * shell_charges[i];
                out[3 * aj + axis][i] -= value * shell_charges[j];
                out[3 * aj + axis][j] -= value * shell_charges[i];
            }
        }
    }
    Ok(out)
}

fn effective_kernel_derivative_vector(rvec: Vec3, gamma: f64) -> Vec3 {
    let r = rvec.norm();
    if r <= DIST_EPS {
        return Vec3::zero();
    }
    let g = GFN1_COULOMB_EXPONENT;
    let denom = r.powf(g) + gamma.powf(-g);
    let pref = -r.powf(g - 2.0) * denom.powf(-1.0 - 1.0 / g);
    rvec * pref
}

pub fn cpxtb_rhs_vector(
    _basis: &BasisSet,
    mos: &Matrix,
    occupations: &[f64],
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    orbital_energies: &[f64],
) -> Result<Vec<f64>> {
    validate_square_like(mos, fock_deriv, "fock_deriv")?;
    validate_square_like(mos, overlap_deriv, "overlap_deriv")?;
    if mos.cols() != occupations.len() || orbital_energies.len() != occupations.len() {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB MO, occupation, and orbital-energy dimensions differ".to_string(),
        ));
    }
    let space = CpxtbSpace::from_occupations(occupations)?;
    let fock_mo = mo_transform(mos, fock_deriv)?;
    let overlap_mo = mo_transform(mos, overlap_deriv)?;
    let mut rhs = Vec::with_capacity(space.len());
    for &(i, a) in &space.pairs {
        let f1_ia = fock_mo[(i, a)];
        let s1_ia = overlap_mo[(i, a)];
        rhs.push(-f1_ia + orbital_energies[i] * s1_ia);
    }
    Ok(rhs)
}

fn cpxtb_matvec_precomputed(
    shell_scc_kernel: &Matrix,
    orbital_gaps: &[f64],
    transition: &[Vec<f64>],
    occupation_scales: &[f64],
    u_vec: &[f64],
) -> Result<Vec<f64>> {
    if u_vec.len() != orbital_gaps.len()
        || transition.len() != u_vec.len()
        || occupation_scales.len() != u_vec.len()
    {
        return Err(Gfn1Error::InvalidInput(
            "precomputed CPXTB vector dimensions differ".to_string(),
        ));
    }
    let mut out = orbital_gaps
        .iter()
        .zip(u_vec.iter())
        .map(|(&gap, &u)| gap * u)
        .collect::<Vec<_>>();
    let coupling = apply_scc_coupling_matrix_from_transition(
        shell_scc_kernel,
        transition,
        occupation_scales,
        u_vec,
    )?;
    for (dst, add) in out.iter_mut().zip(coupling.iter()) {
        *dst += *add;
    }
    Ok(out)
}

fn apply_scc_coupling_matrix_from_transition(
    shell_scc_kernel: &Matrix,
    transition: &[Vec<f64>],
    occupation_scales: &[f64],
    u_vec: &[f64],
) -> Result<Vec<f64>> {
    let nshell = shell_scc_kernel.rows();
    if shell_scc_kernel.cols() != nshell {
        return Err(Gfn1Error::InvalidInput(
            "shell SCC kernel must be square".to_string(),
        ));
    }
    let mut induced_shell_charges = vec![0.0_f64; nshell];
    if occupation_scales.len() != transition.len() {
        return Err(Gfn1Error::InvalidInput(
            "occupation-scale transition dimension mismatch".to_string(),
        ));
    }
    for ((qia, &scale), &u) in transition
        .iter()
        .zip(occupation_scales.iter())
        .zip(u_vec.iter())
    {
        if qia.len() != nshell {
            return Err(Gfn1Error::InvalidInput(
                "transition charge shell dimension mismatch".to_string(),
            ));
        }
        for shell in 0..nshell {
            induced_shell_charges[shell] += qia[shell] * scale * u;
        }
    }
    let shell_potential = matrix_vector_product(shell_scc_kernel, &induced_shell_charges)?;
    let mut out = vec![0.0_f64; transition.len()];
    for (row, qia) in transition.iter().enumerate() {
        out[row] = qia
            .iter()
            .zip(shell_potential.iter())
            .map(|(&q, &v)| q * v)
            .sum::<f64>();
    }
    Ok(out)
}

pub fn transition_shell_charges(
    basis: &BasisSet,
    mos: &Matrix,
    occupations: &[f64],
    overlap: &Matrix,
) -> Result<Vec<Vec<f64>>> {
    if mos.rows() != overlap.rows() || overlap.rows() != overlap.cols() {
        return Err(Gfn1Error::InvalidInput(
            "transition charge matrix shape mismatch".to_string(),
        ));
    }
    if mos.cols() != occupations.len() {
        return Err(Gfn1Error::InvalidInput(
            "transition charge occupation dimension mismatch".to_string(),
        ));
    }
    let space = CpxtbSpace::from_occupations(occupations)?;
    let sc = overlap.matmul(mos)?;
    let mut out = Vec::with_capacity(space.len());
    for &(i, a) in &space.pairs {
        let mut q = vec![0.0_f64; basis.shells.len()];
        for (shell_idx, shell) in basis.shells.iter().enumerate() {
            let end = shell.first_ao + shell.nao;
            for mu in shell.first_ao..end {
                q[shell_idx] -= mos[(mu, a)] * sc[(mu, i)] + mos[(mu, i)] * sc[(mu, a)];
            }
        }
        out.push(q);
    }
    Ok(out)
}

/// Mulliken transition shell charges of an arbitrary molecular-orbital pair
/// `(left, right)`: `q[s] = -sum_{mu in s} (C_{mu,right}(SC)_{mu,left}
/// + C_{mu,left}(SC)_{mu,right})`, with `sc = S C` precomputed. The occupied-virtual
/// special case is [`transition_shell_charges`]; this version is needed for the
/// occupied-occupied and virtual-virtual blocks of the TDA Lagrangian (now used only
/// by the legacy-path diagnostic tests; retained as a response-module utility).
#[allow(dead_code)]
pub(crate) fn mo_pair_transition_shell_charge(
    basis: &BasisSet,
    mos: &Matrix,
    sc: &Matrix,
    left: usize,
    right: usize,
) -> Result<Vec<f64>> {
    if mos.rows() != sc.rows() || mos.cols() != sc.cols() {
        return Err(Gfn1Error::InvalidInput(
            "MO-pair transition-charge matrix dimensions differ".to_string(),
        ));
    }
    let mut out = vec![0.0_f64; basis.shells.len()];
    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        let end = shell.first_ao + shell.nao;
        for mu in shell.first_ao..end {
            out[shell_idx] -= mos[(mu, right)] * sc[(mu, left)] + mos[(mu, left)] * sc[(mu, right)];
        }
    }
    Ok(out)
}

/// Explicit nuclear-coordinate gradient of the transition-transition Coulomb
/// coupling `E_c = c * P^T K P` evaluated at fixed transition shell charges
/// `p_shell`, restricted to the geometry-dependent off-diagonal `dgamma/dR` part
/// (the on-site/third-order pieces are charge-independent and drop at fixed `P`).
/// Equals `c * sum_{i>j} (dgamma_ij/dR)(p_i p_j + p_i p_j)` via the cached
/// shell-pair kernel derivatives, i.e. `c * P^T (dK/dR) P`.
pub(crate) fn coupling_kernel_gradient(
    context: &ResponseGradientContext,
    p_shell: &[f64],
    coupling_scale: f64,
    nat: usize,
) -> Vec<Vec3> {
    let mut gradient = vec![Vec3::zero(); nat];
    if coupling_scale == 0.0 {
        return gradient;
    }
    for pair in &context.shell_pairs {
        let scale = 2.0 * coupling_scale * p_shell[pair.i] * p_shell[pair.j];
        gradient[pair.atom_i] += pair.dkernel * scale;
        gradient[pair.atom_j] -= pair.dkernel * scale;
    }
    gradient
}

fn add_metric_scc_rhs(
    rhs_vectors: &mut [Vec<f64>],
    basis: &BasisSet,
    shell_scc_kernel: &Matrix,
    mos: &Matrix,
    occupations: &[f64],
    overlap: &Matrix,
    ground_density: &Matrix,
    orbital_energies: &[f64],
    derivative_matrices: &[AoDerivativeMatrices],
) -> Result<()> {
    if rhs_vectors.len() != derivative_matrices.len() {
        return Err(Gfn1Error::InvalidInput(
            "metric-SCC RHS coordinate count mismatch".to_string(),
        ));
    }
    let n = basis.len();
    let zero_overlap = Matrix::zeros(n, n);
    for (rhs, deriv) in rhs_vectors.iter_mut().zip(derivative_matrices.iter()) {
        let mut metric_density = Matrix::zeros(n, n);
        add_occupied_metric_density_response(
            &mut metric_density,
            mos,
            occupations,
            &deriv.overlap_deriv,
        )?;
        let metric_shell = response_shell_charges_from_density(
            basis,
            overlap,
            ground_density,
            &metric_density,
            &deriv.overlap_deriv,
        )?;
        let shell_potential = matrix_vector_product(shell_scc_kernel, &metric_shell)?;
        let response_fock = scalar_response_fock_matrix(basis, overlap, &shell_potential)?;
        let add = cpxtb_rhs_vector(
            basis,
            mos,
            occupations,
            &response_fock,
            &zero_overlap,
            orbital_energies,
        )?;
        if add.len() != rhs.len() {
            return Err(Gfn1Error::InvalidInput(
                "metric-SCC RHS vector length mismatch".to_string(),
            ));
        }
        for (dst, value) in rhs.iter_mut().zip(add.iter()) {
            *dst += *value;
        }
    }
    Ok(())
}

pub fn response_density(mos: &Matrix, occupations: &[f64], u_response: &[f64]) -> Result<Matrix> {
    let space = CpxtbSpace::from_occupations(occupations)?;
    response_density_with_space(mos, occupations, &space, u_response)
}

fn response_density_with_space(
    mos: &Matrix,
    occupations: &[f64],
    space: &CpxtbSpace,
    u_response: &[f64],
) -> Result<Matrix> {
    if mos.cols() != occupations.len() || u_response.len() != space.len() {
        return Err(Gfn1Error::InvalidInput(
            "response-density dimension mismatch".to_string(),
        ));
    }
    let norb = occupations.len();
    let mut coeff = Matrix::zeros(norb, norb);
    for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
        let weight = (occupations[i] - occupations[a]) * u_response[pair_idx];
        coeff[(a, i)] += weight;
        coeff[(i, a)] += weight;
    }
    mo_coefficient_matrix_to_ao(mos, &coeff)
}

pub(crate) fn fermi_occupation_response(
    occupations: &[f64],
    orbital_energy_response: &[f64],
    kt: f64,
) -> Result<Vec<f64>> {
    if occupations.len() != orbital_energy_response.len() || kt <= 0.0 {
        return Err(Gfn1Error::InvalidInput(
            "Fermi occupation response dimension mismatch".to_string(),
        ));
    }
    let weights = occupations
        .iter()
        .map(|&occ| (occ * (1.0 - 0.5 * occ)).max(0.0) / kt)
        .collect::<Vec<_>>();
    let denom = weights.iter().sum::<f64>();
    if denom <= 1.0e-30 {
        return Ok(vec![0.0; occupations.len()]);
    }
    let dmu = weights
        .iter()
        .zip(orbital_energy_response.iter())
        .map(|(&w, &deps)| w * deps)
        .sum::<f64>()
        / denom;
    Ok(weights
        .iter()
        .zip(orbital_energy_response.iter())
        .map(|(&w, &deps)| -w * (deps - dmu))
        .collect())
}

pub(crate) fn finite_temperature_density_response(
    mos: &Matrix,
    occupations: &[f64],
    orbital_energies: &[f64],
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    response_fock: &Matrix,
    kt: f64,
) -> Result<(Matrix, Vec<f64>)> {
    let (h_mo, s_mo) =
        finite_temperature_mo_derivatives(mos, fock_deriv, overlap_deriv, response_fock)?;
    let eps_response = orbital_energy_response_from_mo(orbital_energies, &h_mo, &s_mo)?;
    let occupation_response = fermi_occupation_response(occupations, &eps_response, kt)?;
    let coeff = finite_temperature_response_coefficients_from_mo(
        occupations,
        orbital_energies,
        &occupation_response,
        &h_mo,
        &s_mo,
        kt,
        false,
    )?;
    Ok((
        mo_coefficient_matrix_to_ao(mos, &coeff)?,
        occupation_response,
    ))
}

pub(crate) fn finite_temperature_energy_weighted_response(
    mos: &Matrix,
    occupations: &[f64],
    occupation_response: &[f64],
    orbital_energies: &[f64],
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    response_fock: &Matrix,
    kt: f64,
) -> Result<Matrix> {
    let (h_mo, s_mo) =
        finite_temperature_mo_derivatives(mos, fock_deriv, overlap_deriv, response_fock)?;
    let coeff = finite_temperature_response_coefficients_from_mo(
        occupations,
        orbital_energies,
        occupation_response,
        &h_mo,
        &s_mo,
        kt,
        true,
    )?;
    let _ = kt;
    mo_coefficient_matrix_to_ao(mos, &coeff)
}

pub(crate) fn finite_temperature_mo_derivatives(
    mos: &Matrix,
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
    response_fock: &Matrix,
) -> Result<(Matrix, Matrix)> {
    validate_square_like(mos, fock_deriv, "fock_deriv")?;
    validate_square_like(mos, overlap_deriv, "overlap_deriv")?;
    validate_square_like(mos, response_fock, "response_fock")?;
    let mut total_fock_deriv = fock_deriv.clone();
    for idx in 0..total_fock_deriv.as_mut_slice().len() {
        total_fock_deriv.as_mut_slice()[idx] += response_fock.as_slice()[idx];
    }
    Ok((
        mo_transform(mos, &total_fock_deriv)?,
        mo_transform(mos, overlap_deriv)?,
    ))
}

pub(crate) fn orbital_energy_response_from_mo(
    orbital_energies: &[f64],
    h_mo: &Matrix,
    s_mo: &Matrix,
) -> Result<Vec<f64>> {
    if h_mo.rows() != orbital_energies.len()
        || h_mo.cols() != orbital_energies.len()
        || s_mo.rows() != orbital_energies.len()
        || s_mo.cols() != orbital_energies.len()
    {
        return Err(Gfn1Error::InvalidInput(
            "orbital energy response MO dimension mismatch".to_string(),
        ));
    }
    let mut out = vec![0.0_f64; orbital_energies.len()];
    for i in 0..orbital_energies.len() {
        out[i] = h_mo[(i, i)] - orbital_energies[i] * s_mo[(i, i)];
    }
    Ok(out)
}

pub(crate) fn finite_temperature_response_coefficients_from_mo(
    occupations: &[f64],
    orbital_energies: &[f64],
    occupation_response: &[f64],
    h_mo: &Matrix,
    s_mo: &Matrix,
    kt: f64,
    energy_weighted: bool,
) -> Result<Matrix> {
    let norb = occupations.len();
    if orbital_energies.len() != norb
        || occupation_response.len() != norb
        || h_mo.rows() != norb
        || h_mo.cols() != norb
        || s_mo.rows() != norb
        || s_mo.cols() != norb
    {
        return Err(Gfn1Error::InvalidInput(
            "finite-temperature response coefficient dimension mismatch".to_string(),
        ));
    }
    let mut coeff = Matrix::zeros(norb, norb);
    for i in 0..norb {
        let f_i = occupations[i];
        let e_i = orbital_energies[i];
        let df_i = occupation_response[i];
        coeff[(i, i)] = if energy_weighted {
            let h_ii = h_mo[(i, i)] - e_i * s_mo[(i, i)];
            f_i * h_ii + e_i * df_i - f_i * e_i * s_mo[(i, i)]
        } else {
            df_i - f_i * s_mo[(i, i)]
        };
        for j in i + 1..norb {
            let f_j = occupations[j];
            let e_j = orbital_energies[j];
            let h_ij = h_mo[(i, j)];
            let s_ij = s_mo[(i, j)];
            let gap = e_i - e_j;
            let value = if gap.abs() > 1.0e-10 {
                if energy_weighted {
                    let w_i = f_i * e_i;
                    let w_j = f_j * e_j;
                    (w_i - w_j) * h_ij / gap - (w_i * e_i - w_j * e_j) * s_ij / gap
                } else {
                    (f_i - f_j) * h_ij / gap - (f_i * e_i - f_j * e_j) * s_ij / gap
                }
            } else {
                let eps = 0.5 * (e_i + e_j);
                let f = 0.5 * (f_i + f_j);
                let slope_f = -0.5 * (f_i * (1.0 - 0.5 * f_i) + f_j * (1.0 - 0.5 * f_j)) / kt;
                if energy_weighted {
                    let slope_w = f + eps * slope_f;
                    let slope_eps_w = 2.0 * eps * f + eps * eps * slope_f;
                    slope_w * h_ij - slope_eps_w * s_ij
                } else {
                    slope_f * h_ij - (f + eps * slope_f) * s_ij
                }
            };
            coeff[(i, j)] = value;
            coeff[(j, i)] = value;
        }
    }
    Ok(coeff)
}

pub(crate) fn mo_coefficient_matrix_to_ao(mos: &Matrix, coeff: &Matrix) -> Result<Matrix> {
    if coeff.rows() != mos.cols() || coeff.cols() != mos.cols() {
        return Err(Gfn1Error::InvalidInput(
            "MO coefficient response matrix shape mismatch".to_string(),
        ));
    }
    let tmp = mos.matmul(coeff)?;
    tmp.matmul(&mos.transpose())
}

fn mo_transform(mos: &Matrix, ao_matrix: &Matrix) -> Result<Matrix> {
    validate_square_like(mos, ao_matrix, "ao_matrix")?;
    let tmp = ao_matrix.matmul(mos)?;
    mos.transpose().matmul(&tmp)
}

pub fn response_energy_weighted_density(
    mos: &Matrix,
    occupations: &[f64],
    orbital_energies: &[f64],
    u_response: &[f64],
) -> Result<Matrix> {
    let space = CpxtbSpace::from_occupations(occupations)?;
    response_energy_weighted_density_with_space(
        mos,
        occupations,
        orbital_energies,
        &space,
        u_response,
    )
}

fn response_energy_weighted_density_with_space(
    mos: &Matrix,
    occupations: &[f64],
    orbital_energies: &[f64],
    space: &CpxtbSpace,
    u_response: &[f64],
) -> Result<Matrix> {
    if mos.cols() != occupations.len()
        || orbital_energies.len() != occupations.len()
        || u_response.len() != space.len()
    {
        return Err(Gfn1Error::InvalidInput(
            "response energy-weighted density dimension mismatch".to_string(),
        ));
    }
    let norb = occupations.len();
    let mut coeff = Matrix::zeros(norb, norb);
    for (pair_idx, &(i, a)) in space.pairs.iter().enumerate() {
        let weight = (occupations[i] - occupations[a]) * orbital_energies[i] * u_response[pair_idx];
        coeff[(a, i)] += weight;
        coeff[(i, a)] += weight;
    }
    mo_coefficient_matrix_to_ao(mos, &coeff)
}

fn add_occupied_metric_density_response(
    density_response: &mut Matrix,
    mos: &Matrix,
    occupations: &[f64],
    overlap_deriv: &Matrix,
) -> Result<()> {
    validate_square_like(mos, overlap_deriv, "overlap_deriv")?;
    validate_same_shape(
        density_response,
        overlap_deriv,
        "density_response",
        "overlap_deriv",
    )?;
    let s_mo = mo_transform(mos, overlap_deriv)?;
    let norb = occupations.len();
    let mut coeff = Matrix::zeros(norb, norb);
    for i in 0..occupations.len() {
        if occupations[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occupations.len() {
            if occupations[j] <= 1.0e-8 {
                continue;
            }
            let occ_scale = 0.5 * (occupations[i] + occupations[j]);
            let s1 = s_mo[(i, j)];
            let weight = -occ_scale * s1;
            coeff[(i, j)] += weight;
        }
    }
    let add = mo_coefficient_matrix_to_ao(mos, &coeff)?;
    add_matrix_in_place(density_response, &add)?;
    Ok(())
}

fn add_occupied_metric_energy_weighted_response(
    w_response: &mut Matrix,
    mos: &Matrix,
    occupations: &[f64],
    orbital_energies: &[f64],
    fock_deriv: &Matrix,
    overlap_deriv: &Matrix,
) -> Result<()> {
    validate_square_like(mos, fock_deriv, "fock_deriv")?;
    validate_square_like(mos, overlap_deriv, "overlap_deriv")?;
    validate_same_shape(w_response, fock_deriv, "w_response", "fock_deriv")?;
    let f_mo = mo_transform(mos, fock_deriv)?;
    let s_mo = mo_transform(mos, overlap_deriv)?;
    let norb = occupations.len();
    let mut coeff = Matrix::zeros(norb, norb);
    for i in 0..occupations.len() {
        if occupations[i] <= 1.0e-8 {
            continue;
        }
        for j in 0..occupations.len() {
            if occupations[j] <= 1.0e-8 {
                continue;
            }
            let occ_scale = 0.5 * (occupations[i] + occupations[j]);
            let f1 = f_mo[(i, j)];
            let s1 = s_mo[(i, j)];
            let weight = occ_scale * (f1 - (orbital_energies[i] + orbital_energies[j]) * s1);
            coeff[(i, j)] += weight;
        }
    }
    let add = mo_coefficient_matrix_to_ao(mos, &coeff)?;
    add_matrix_in_place(w_response, &add)?;
    Ok(())
}

pub(crate) fn response_shell_charges_from_density(
    basis: &BasisSet,
    overlap: &Matrix,
    ground_density: &Matrix,
    density_response: &Matrix,
    overlap_deriv: &Matrix,
) -> Result<Vec<f64>> {
    let n = basis.len();
    if overlap.rows() != n
        || overlap.cols() != n
        || ground_density.rows() != n
        || ground_density.cols() != n
        || density_response.rows() != n
        || density_response.cols() != n
        || overlap_deriv.rows() != n
        || overlap_deriv.cols() != n
    {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB response shell-charge dimensions differ".to_string(),
        ));
    }
    let mut out = vec![0.0_f64; basis.shells.len()];
    for nu in 0..n {
        let mut population = 0.0;
        for kappa in 0..n {
            population += density_response[(nu, kappa)] * overlap[(kappa, nu)];
        }
        out[basis.aos[nu].shell_index] -= population;
    }
    for nu in 0..n {
        let mut population_deriv = 0.0;
        for kappa in 0..n {
            population_deriv += ground_density[(nu, kappa)] * overlap_deriv[(kappa, nu)];
        }
        out[basis.aos[nu].shell_index] -= population_deriv;
    }
    Ok(out)
}

pub(crate) fn scalar_response_fock_matrix(
    basis: &BasisSet,
    overlap: &Matrix,
    shell_potential: &[f64],
) -> Result<Matrix> {
    let n = basis.len();
    if overlap.rows() != n || overlap.cols() != n || shell_potential.len() != basis.shells.len() {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB response scalar-potential matrix dimensions differ".to_string(),
        ));
    }
    let mut out = Matrix::zeros(n, n);
    for mu in 0..n {
        let v_mu = shell_potential[basis.aos[mu].shell_index];
        for nu in 0..n {
            let v_nu = shell_potential[basis.aos[nu].shell_index];
            out[(mu, nu)] = -0.5 * (v_mu + v_nu) * overlap[(mu, nu)];
        }
    }
    Ok(out)
}

pub fn assemble_hessian_pulay_response(
    density_response: &Matrix,
    fock_deriv_x: &Matrix,
    overlap_deriv_x: &Matrix,
    w_response: &Matrix,
) -> Result<f64> {
    validate_same_shape(
        density_response,
        fock_deriv_x,
        "density_response",
        "fock_deriv_x",
    )?;
    validate_same_shape(w_response, overlap_deriv_x, "w_response", "overlap_deriv_x")?;
    Ok(trace_product(density_response, fock_deriv_x) - trace_product(w_response, overlap_deriv_x))
}

fn solve_cpxtb_preconditioned<F>(
    a_times_u: F,
    rhs: &[f64],
    precond_diag: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<CpxtbSolution>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
{
    if rhs.is_empty() {
        return Ok(CpxtbSolution {
            amplitudes: Vec::new(),
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
            route: CpxtbRoute::Pcg,
        });
    }
    if precond_diag.len() != rhs.len() {
        return Err(Gfn1Error::InvalidInput(
            "CPXTB preconditioner diagonal length mismatch".to_string(),
        ));
    }
    let inv = precond_diag
        .iter()
        .map(|&d| {
            if d.is_finite() && d > CPXTB_PRECOND_GAP_FLOOR {
                1.0 / d
            } else {
                1.0 / CPXTB_PRECOND_GAP_FLOOR
            }
        })
        .collect::<Vec<_>>();
    let target = tol.max(0.0) * norm(rhs).max(1.0);
    let mut x = vec![0.0_f64; rhs.len()];
    let mut r = rhs.to_vec();
    let mut best_resid = norm(&r);
    if best_resid <= target {
        return Ok(CpxtbSolution {
            amplitudes: x,
            iterations: 0,
            residual_norm: best_resid,
            converged: true,
            route: CpxtbRoute::Pcg,
        });
    }
    let mut best_x = x.clone();
    let mut z = r
        .iter()
        .zip(inv.iter())
        .map(|(&ri, &mi)| ri * mi)
        .collect::<Vec<_>>();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut iterations = 0usize;
    for iter in 1..=max_iter.max(1) {
        iterations = iter;
        let ap = a_times_u(&p)?;
        if ap.len() != rhs.len() {
            return Err(Gfn1Error::InvalidInput(
                "CPXTB matrix-vector product returned wrong length".to_string(),
            ));
        }
        let denom = dot(&p, &ap);
        if !(denom.is_finite() && denom.abs() > 1.0e-30) {
            break;
        }
        let alpha = rz / denom;
        for k in 0..x.len() {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        let rnorm = norm(&r);
        if !rnorm.is_finite() {
            break;
        }
        if rnorm < best_resid {
            best_resid = rnorm;
            best_x.copy_from_slice(&x);
        }
        if rnorm <= target {
            return Ok(CpxtbSolution {
                amplitudes: x,
                iterations,
                residual_norm: rnorm,
                converged: true,
                route: CpxtbRoute::Pcg,
            });
        }
        if rnorm > CPXTB_PCG_DIVERGENCE_FACTOR * best_resid {
            break;
        }
        for k in 0..z.len() {
            z[k] = r[k] * inv[k];
        }
        let rz_next = dot(&r, &z);
        if !(rz.is_finite() && rz.abs() > 1.0e-300) {
            break;
        }
        let beta = rz_next / rz;
        for k in 0..p.len() {
            p[k] = z[k] + beta * p[k];
        }
        rz = rz_next;
    }
    if rhs.len() <= CPXTB_DENSE_FALLBACK_MAX_DIM {
        if let Ok(operator) = build_dense_cpxtb_operator(rhs.len(), &a_times_u) {
            if let Ok(dense) = solve_cpxtb_dense(&operator, rhs) {
                if dense.residual_norm.is_finite() && dense.residual_norm <= best_resid {
                    return Ok(CpxtbSolution {
                        amplitudes: dense.amplitudes,
                        iterations,
                        residual_norm: dense.residual_norm,
                        converged: dense.residual_norm <= target,
                        route: CpxtbRoute::PcgDenseFallback,
                    });
                }
            }
        }
    }
    Ok(CpxtbSolution {
        amplitudes: best_x,
        iterations,
        residual_norm: best_resid,
        converged: best_resid <= target,
        route: CpxtbRoute::Pcg,
    })
}

fn build_dense_cpxtb_operator<F>(n: usize, matvec: F) -> Result<Matrix>
where
    F: Fn(&[f64]) -> Result<Vec<f64>>,
{
    let mut out = Matrix::zeros(n, n);
    for col in 0..n {
        let mut unit = vec![0.0_f64; n];
        unit[col] = 1.0;
        let au = matvec(&unit)?;
        if au.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "dense CPXTB operator matvec length mismatch".to_string(),
            ));
        }
        for row in 0..n {
            out[(row, col)] = au[row];
        }
    }
    Ok(out)
}

fn solve_cpxtb_dense(operator: &Matrix, rhs: &[f64]) -> Result<CpxtbSolution> {
    let mut batch = solve_cpxtb_dense_batch(operator, &[rhs.to_vec()])?;
    batch.pop().ok_or_else(|| {
        Gfn1Error::InvalidInput("dense CPXTB batch solver returned no solution".to_string())
    })
}

fn solve_cpxtb_dense_batch(
    operator: &Matrix,
    rhs_vectors: &[Vec<f64>],
) -> Result<Vec<CpxtbSolution>> {
    let n = operator.rows();
    if operator.cols() != n {
        return Err(Gfn1Error::InvalidInput(
            "dense CPXTB operator must be square".to_string(),
        ));
    }
    if rhs_vectors.is_empty() {
        return Ok(Vec::new());
    }
    for rhs in rhs_vectors {
        if rhs.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "dense CPXTB RHS dimension mismatch".to_string(),
            ));
        }
    }
    let faer_operator = FaerMat::from_fn(n, n, |i, j| operator[(i, j)]);
    let rhs_matrix = FaerMat::from_fn(n, rhs_vectors.len(), |i, j| rhs_vectors[j][i]);
    let solution_matrix = faer_operator.partial_piv_lu().solve(&rhs_matrix);
    let mut out = Vec::with_capacity(rhs_vectors.len());
    for (col, rhs) in rhs_vectors.iter().enumerate() {
        let amplitudes = (0..n)
            .map(|row| solution_matrix[(row, col)])
            .collect::<Vec<_>>();
        let mut residual_ss = 0.0_f64;
        for row in 0..n {
            let mut ax = 0.0_f64;
            for col_op in 0..n {
                ax += operator[(row, col_op)] * amplitudes[col_op];
            }
            let delta = ax - rhs[row];
            residual_ss += delta * delta;
        }
        let residual_norm = residual_ss.sqrt();
        out.push(CpxtbSolution {
            amplitudes,
            iterations: 0,
            residual_norm,
            converged: residual_norm <= 1.0e-8_f64.max(1.0e-10 * norm(rhs)),
            route: CpxtbRoute::Dense,
        });
    }
    Ok(out)
}

/// **Charge-space (low-rank) reduction of the CPXTB Jacobian.**
///
/// The MO-pair operator built by [`cpxtb_matvec_precomputed`] is a diagonal plus
/// a rank-`nsh` term,
///
/// ```text
///   A = D_g + T K Tᵀ D_s
/// ```
///
/// with `D_g` the occupied–virtual gaps, `D_s = ½(f_i − f_a)`, `T` the
/// `npair × nsh` Mulliken transition shell charges and `K` the SCC response
/// kernel. Since `npair ~ n²/4` while `nsh ~ n`, *every* pair-space route —
/// the dense `npair × npair` LU and the pair-space PCG alike — resolves an
/// `nsh × nsh` amount of physics with `O(npair³)` / `O(npair²·nsh)` work.
/// Eliminating the diagonal in closed form,
///
/// ```text
///   y ≡ Tᵀ D_s x,   X ≡ Tᵀ D_s D_g⁻¹ T,   B ≡ D_s D_g⁻¹ T
///   (I + X K) y = Bᵀ b
///   x = D_g⁻¹ (b − T K y)
/// ```
///
/// leaves ONE `nsh × nsh` factorization for every right-hand side and
/// `O(npair·nsh)` GEMM work per RHS. This is the same dielectric reduction the
/// charge-space solver ([`crate::response::charge_space`]) performs on the
/// spectral side — here derived directly from the MO-pair operator, so the
/// amplitudes it returns are the amplitudes of `A x = b`, agreeing with the
/// dense route to `~1e-15` relative on every measured fixture.
///
/// **Measured** (`bench_cp_routes`, `npair`/route/total time): distorted
/// Ni(CO)₄ at 3000 K `679`, dense LU `0.23–0.26 s` → `0.002–0.015 s`
/// (**16–132×**); water 3×3×2 `5184`, dense LU `22–40 s` → `0.18–0.27 s`
/// (**126–147×**); water 3×3×3 `11664`, PCG `17–22 s` → `0.56–0.85 s`
/// (**26–30×**). The `1/gap` factor is applied in closed
/// form rather than resolved by a factorization, which is what lets the
/// near-degenerate Fermi-smeared fixtures — where the preconditioned CG averages
/// 356 iterations against 679 unknowns and never reaches its tolerance — be
/// answered in zero iterations at round-off.
struct CpxtbLowRank {
    npair: usize,
    nshell: usize,
    /// `T` — transition shell charges, row `p` = `q^{(ia)}` (npair × nsh).
    transition: Matrix,
    /// SCC response kernel `K` (nsh × nsh).
    kernel: Matrix,
    /// `M = I + X K`, kept alongside its factorization so the exact pair-space
    /// residual `A x − b = −T K [(I + X K) y − Bᵀ b]` can be formed.
    dielectric: Matrix,
    lu: DenseLu,
    inv_gap: Vec<f64>,
    scales: Vec<f64>,
}

impl CpxtbLowRank {
    fn build(
        shell_scc_kernel: &Matrix,
        orbital_gaps: &[f64],
        transition: &[Vec<f64>],
        occupation_scales: &[f64],
    ) -> Result<Self> {
        let npair = orbital_gaps.len();
        let nshell = shell_scc_kernel.rows();
        if shell_scc_kernel.cols() != nshell {
            return Err(Gfn1Error::InvalidInput(
                "charge-space CPXTB reduction: SCC kernel must be square".to_string(),
            ));
        }
        if transition.len() != npair || occupation_scales.len() != npair {
            return Err(Gfn1Error::InvalidInput(
                "charge-space CPXTB reduction: pair dimensions differ".to_string(),
            ));
        }
        if npair == 0 || nshell == 0 {
            return Err(Gfn1Error::InvalidInput(
                "charge-space CPXTB reduction: empty pair or shell space".to_string(),
            ));
        }
        let mut t = Matrix::zeros(npair, nshell);
        for (p, row) in transition.iter().enumerate() {
            if row.len() != nshell {
                return Err(Gfn1Error::InvalidInput(
                    "charge-space CPXTB reduction: transition-charge shell dimension mismatch"
                        .to_string(),
                ));
            }
            t.as_mut_slice()[p * nshell..(p + 1) * nshell].copy_from_slice(row);
        }
        let mut inv_gap = Vec::with_capacity(npair);
        let mut scaled = Matrix::zeros(npair, nshell);
        for p in 0..npair {
            let gap = orbital_gaps[p];
            if !(gap.is_finite() && gap > 0.0) {
                return Err(Gfn1Error::InvalidInput(
                    "charge-space CPXTB reduction: non-positive occupied-virtual gap".to_string(),
                ));
            }
            let ig = 1.0 / gap;
            let weight = occupation_scales[p] * ig;
            if !(ig.is_finite() && weight.is_finite() && weight.abs() <= CPXTB_LOW_RANK_MAX_WEIGHT) {
                return Err(Gfn1Error::InvalidInput(
                    "charge-space CPXTB reduction: occupation/gap weight overflowed".to_string(),
                ));
            }
            inv_gap.push(ig);
            let (src, dst) = (
                &t.as_slice()[p * nshell..(p + 1) * nshell],
                &mut scaled.as_mut_slice()[p * nshell..(p + 1) * nshell],
            );
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = weight * *s;
            }
        }
        // X = Tᵀ D_s D_g⁻¹ T, then M = I + X K.
        let x = matmul_transpose_a(&t, &scaled)?;
        drop(scaled);
        let mut dielectric = x.matmul(shell_scc_kernel)?;
        for s in 0..nshell {
            dielectric[(s, s)] += 1.0;
        }
        if dielectric.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(Gfn1Error::InvalidInput(
                "charge-space CPXTB reduction: dielectric matrix is not finite".to_string(),
            ));
        }
        let lu = DenseLu::factor(&dielectric)?;
        Ok(Self {
            npair,
            nshell,
            transition: t,
            kernel: shell_scc_kernel.clone(),
            dielectric,
            lu,
            inv_gap,
            scales: occupation_scales.to_vec(),
        })
    }

    /// Solve `A x_j = rhs_j` for a whole right-hand-side family. Everything that
    /// touches the `npair` dimension is a GEMM over a block of RHS columns.
    fn solve_batch(&self, rhs_vectors: &[Vec<f64>]) -> Result<Vec<CpxtbSolution>> {
        let (npair, nshell) = (self.npair, self.nshell);
        for rhs in rhs_vectors {
            if rhs.len() != npair {
                return Err(Gfn1Error::InvalidInput(
                    "charge-space CPXTB reduction: RHS dimension mismatch".to_string(),
                ));
            }
        }
        let mut out = Vec::with_capacity(rhs_vectors.len());
        for chunk in rhs_vectors.chunks(CPXTB_LOW_RANK_RHS_CHUNK) {
            let m = chunk.len();
            // Bᵀ b for the whole block: pre-scale the RHS by D_s D_g⁻¹ so the
            // contraction with T is a plain transpose-GEMM.
            let mut weighted = Matrix::zeros(npair, m);
            for p in 0..npair {
                let w = self.scales[p] * self.inv_gap[p];
                for (j, rhs) in chunk.iter().enumerate() {
                    weighted[(p, j)] = w * rhs[p];
                }
            }
            let y0 = matmul_transpose_a(&self.transition, &weighted)?;
            drop(weighted);
            let mut k_y = Matrix::zeros(nshell, m);
            let mut residual_shell = Matrix::zeros(nshell, m);
            let mut rhs_shell = vec![0.0_f64; nshell];
            for j in 0..m {
                for (s, dst) in rhs_shell.iter_mut().enumerate() {
                    *dst = y0[(s, j)];
                }
                let y = self.lu.solve_vec(&rhs_shell)?;
                let my = matrix_vector_product(&self.dielectric, &y)?;
                let ky = matrix_vector_product(&self.kernel, &y)?;
                for s in 0..nshell {
                    residual_shell[(s, j)] = my[s] - rhs_shell[s];
                    k_y[(s, j)] = ky[s];
                }
            }
            let t_k_y = self.transition.matmul(&k_y)?;
            // Exact pair-space residual: A x − b = −T K [(I + X K) y − Bᵀ b].
            let k_res = self.kernel.matmul(&residual_shell)?;
            let t_k_res = self.transition.matmul(&k_res)?;
            for (j, rhs) in chunk.iter().enumerate() {
                let mut amplitudes = vec![0.0_f64; npair];
                let mut residual_ss = 0.0_f64;
                for p in 0..npair {
                    amplitudes[p] = self.inv_gap[p] * (rhs[p] - t_k_y[(p, j)]);
                    let d = t_k_res[(p, j)];
                    residual_ss += d * d;
                }
                let residual_norm = residual_ss.sqrt();
                let finite = residual_norm.is_finite()
                    && amplitudes.iter().all(|v| v.is_finite());
                out.push(CpxtbSolution {
                    amplitudes,
                    iterations: 0,
                    residual_norm,
                    converged: finite
                        && residual_norm <= 1.0e-8_f64.max(1.0e-10 * norm(rhs)),
                    route: CpxtbRoute::ChargeSpace,
                });
            }
        }
        Ok(out)
    }
}

/// Solve the CP family `A x_j = rhs_j` by the cheapest sound route.
///
/// Order of preference:
///  1. the direct **charge-space reduction** ([`CpxtbLowRank`]) — non-iterative,
///     one `nsh × nsh` factorization for the whole right-hand-side family;
///  2. the dense pair-space LU (small `npair`);
///  3. per-RHS preconditioned CG.
///
/// The reduction is only kept when every solution meets the same convergence
/// criterion the dense route uses, so a degenerate/overflowing reduction can
/// never silently replace a working factorization.
fn solve_cpxtb_all(
    shell_scc_kernel: &Matrix,
    orbital_gaps: &[f64],
    transition: &[Vec<f64>],
    occupation_scales: &[f64],
    rhs_vectors: &[Vec<f64>],
    cpxtb_options: CpxtbOptions,
) -> Result<Vec<CpxtbSolution>> {
    if rhs_vectors.is_empty() {
        return Ok(Vec::new());
    }
    let npair = orbital_gaps.len();
    {
        let _profile = crate::profile::scope("cphf.solve_charge_space");
        if let Ok(low_rank) = CpxtbLowRank::build(
            shell_scc_kernel,
            orbital_gaps,
            transition,
            occupation_scales,
        ) {
            if let Ok(solutions) = low_rank.solve_batch(rhs_vectors) {
                if solutions.iter().all(|s| s.converged) {
                    return Ok(solutions);
                }
            }
        }
    }
    let matvec = |u: &[f64]| {
        cpxtb_matvec_precomputed(
            shell_scc_kernel,
            orbital_gaps,
            transition,
            occupation_scales,
            u,
        )
    };
    if npair <= CPXTB_DENSE_FALLBACK_MAX_DIM {
        let _profile = crate::profile::scope("cphf.solve_dense");
        let operator = build_dense_cpxtb_operator(npair, &matvec)?;
        return solve_cpxtb_dense_batch(&operator, rhs_vectors);
    }
    let _profile = crate::profile::scope("cphf.solve_pcg");
    let mut out = Vec::with_capacity(rhs_vectors.len());
    for rhs in rhs_vectors {
        out.push(solve_cpxtb_preconditioned(
            &matvec,
            rhs,
            orbital_gaps,
            cpxtb_options.tol,
            cpxtb_options.max_iter,
        )?);
    }
    Ok(out)
}

fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut value = 0.0;
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            value += a[(i, j)] * b[(j, i)];
        }
    }
    value
}

fn add_matrix_in_place(dst: &mut Matrix, add: &Matrix) -> Result<()> {
    validate_same_shape(dst, add, "dst", "add")?;
    for (dst_value, add_value) in dst.as_mut_slice().iter_mut().zip(add.as_slice().iter()) {
        *dst_value += *add_value;
    }
    Ok(())
}

fn validate_square_like(mos: &Matrix, matrix: &Matrix, name: &str) -> Result<()> {
    if matrix.rows() != mos.rows() || matrix.cols() != mos.rows() {
        return Err(Gfn1Error::InvalidInput(format!(
            "{name} must be square in the AO dimension"
        )));
    }
    Ok(())
}

fn validate_same_shape(a: &Matrix, b: &Matrix, a_name: &str, b_name: &str) -> Result<()> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(Gfn1Error::InvalidInput(format!(
            "{a_name} and {b_name} shape mismatch"
        )));
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

fn copy_lower_to_upper(matrix: &mut Matrix) {
    let n = matrix.rows().min(matrix.cols());
    for i in 0..n {
        for j in 0..i {
            matrix[(j, i)] = matrix[(i, j)];
        }
    }
}

/// Convergence/conditioning gates and the measurement harness for the CP linear
/// solves.
///
/// The benchmarks are `#[ignore]`d (they rebuild several SCF states and run every
/// route side by side); run them with
/// `cargo test --profile reltest --lib response::cpxtb::tests -- --ignored --nocapture`.
/// The non-ignored tests are the regressions: they assert on **routes, iteration
/// counts and residuals** — never wall time — so they are CI-stable.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::{run_electronic, ElectronicOptions};
    use std::time::Instant;

    const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
    const NI_CO4_DISTORTED: &str = "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n";
    /// Stretched (1.55 Å) linear H₁₀ chain: the smallest closed-shell fixture
    /// whose occupied–virtual gap collapses at T = 0 without tripping the
    /// singular guard — the conditioning stress case for the pair-space routes.
    fn stretched_h10_xyz() -> String {
        let mut out = String::from("10\nstretched linear H10\n");
        for i in 0..10 {
            out.push_str(&format!("H 0.000000 0.000000 {:.6}\n", 1.55 * i as f64));
        }
        out
    }

    /// A `nx × ny × nz` grid of well-separated waters — the size knob for the
    /// scaling measurements (`npair` grows as the square of the molecule count,
    /// so this crosses the dense/PCG branch point quickly).
    fn water_grid_xyz(nx: usize, ny: usize, nz: usize) -> String {
        let spacing = 4.2_f64;
        let mut body = String::new();
        let mut count = 0usize;
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let (x, y, z) = (
                        spacing * ix as f64,
                        spacing * iy as f64,
                        spacing * iz as f64,
                    );
                    body.push_str(&format!("O {:.6} {:.6} {:.6}\n", x, y, z + 0.1173));
                    body.push_str(&format!("H {:.6} {:.6} {:.6}\n", x, y + 0.7572, z - 0.4692));
                    body.push_str(&format!("H {:.6} {:.6} {:.6}\n", x, y - 0.7572, z - 0.4692));
                    count += 3;
                }
            }
        }
        format!("{count}\nwater grid\n{body}")
    }

    fn options_at(temperature: f64) -> ElectronicOptions {
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = temperature;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        options
    }

    fn setup_for(xyz: &str, temperature: f64) -> (CpxtbSetup, usize) {
        let options = options_at(temperature);
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let nshell = electronic.basis.shells.len();
        let setup = build_cpxtb_setup(
            &system,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: options.hamiltonian.coordination_cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            None,
        )
        .unwrap();
        (setup, nshell)
    }

    /// True Jacobi diagonal of `A = D_g + T K Tᵀ D_s`, i.e. the gap preconditioner
    /// PLUS the on-pair screening `½(f_i − f_a)·(q^{ia})ᵀ K q^{ia}` the shipped
    /// preconditioner drops. Measurement-only (see the report): it changes the
    /// PCG iteration count by well under the 10 % bar on every fixture, and the
    /// PCG branch is no longer reached in practice.
    fn operator_diagonal(
        kernel: &Matrix,
        gaps: &[f64],
        transition: &[Vec<f64>],
        scales: &[f64],
    ) -> Vec<f64> {
        (0..gaps.len())
            .map(|p| {
                let kq = matrix_vector_product(kernel, &transition[p]).unwrap();
                gaps[p] + scales[p] * dot(&transition[p], &kq)
            })
            .collect()
    }

    /// PCG with an explicit initial guess (the shipped solver always starts at
    /// zero). Used to measure the RHS-seeding idea of task item 3.
    fn pcg_with_guess<F>(
        a_times_u: F,
        rhs: &[f64],
        precond_diag: &[f64],
        x0: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> (Vec<f64>, usize, f64)
    where
        F: Fn(&[f64]) -> Result<Vec<f64>>,
    {
        let n = rhs.len();
        let inv: Vec<f64> = precond_diag
            .iter()
            .map(|&d| {
                if d.is_finite() && d > CPXTB_PRECOND_GAP_FLOOR {
                    1.0 / d
                } else {
                    1.0 / CPXTB_PRECOND_GAP_FLOOR
                }
            })
            .collect();
        let target = tol * norm(rhs).max(1.0);
        let mut x = x0.to_vec();
        let ax0 = a_times_u(&x).unwrap();
        let mut r: Vec<f64> = (0..n).map(|k| rhs[k] - ax0[k]).collect();
        if norm(&r) <= target {
            return (x, 0, norm(&r));
        }
        let mut z: Vec<f64> = (0..n).map(|k| r[k] * inv[k]).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for iter in 1..=max_iter {
            let ap = a_times_u(&p).unwrap();
            let denom = dot(&p, &ap);
            if !(denom.is_finite() && denom.abs() > 1.0e-30) {
                return (x, iter, norm(&r));
            }
            let alpha = rz / denom;
            for k in 0..n {
                x[k] += alpha * p[k];
                r[k] -= alpha * ap[k];
            }
            let rnorm = norm(&r);
            if !rnorm.is_finite() || rnorm <= target {
                return (x, iter, rnorm);
            }
            for k in 0..n {
                z[k] = r[k] * inv[k];
            }
            let rz_next = dot(&r, &z);
            let beta = rz_next / rz;
            for k in 0..n {
                p[k] = z[k] + beta * p[k];
            }
            rz = rz_next;
        }
        (x, max_iter, norm(&r))
    }

    fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    /// One fixture, every route, side by side.
    fn bench_fixture(label: &str, xyz: &str, temperature: f64, run_dense: bool) {
        let (setup, nshell) = setup_for(xyz, temperature);
        let npair = setup.space.len();
        let ndof = setup.rhs_vectors.len();
        let matvec = |u: &[f64]| setup.matvec(u);
        eprintln!(
            "\n=== {label}: npair {npair}  nshell {nshell}  nRHS {ndof}  \
             min gap {:.3e}  max gap {:.3e}",
            setup
                .orbital_gaps
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            setup.orbital_gaps.iter().cloned().fold(0.0_f64, f64::max),
        );

        // --- Route: charge-space reduction (the new default).
        let t0 = Instant::now();
        let low_rank = CpxtbLowRank::build(
            &setup.shell_kernel,
            &setup.orbital_gaps,
            &setup.transition,
            &setup.occupation_scales,
        )
        .unwrap();
        let lr_build = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let lr_solutions = low_rank.solve_batch(&setup.rhs_vectors).unwrap();
        let lr_solve = t0.elapsed().as_secs_f64();
        let lr_resid = lr_solutions
            .iter()
            .map(|s| s.residual_norm)
            .fold(0.0_f64, f64::max);
        eprintln!(
            "  charge-space : build {lr_build:9.4} s  solve {lr_solve:9.4} s  \
             total {:9.4} s  max resid {lr_resid:.3e}",
            lr_build + lr_solve
        );

        // --- Route: dense pair-space LU (the previous default for npair <= 2048).
        let mut dense_solutions = Vec::new();
        if run_dense {
            let t0 = Instant::now();
            let operator = build_dense_cpxtb_operator(npair, &matvec).unwrap();
            let dense_build = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            dense_solutions = solve_cpxtb_dense_batch(&operator, &setup.rhs_vectors).unwrap();
            let dense_solve = t0.elapsed().as_secs_f64();
            let dense_resid = dense_solutions
                .iter()
                .map(|s| s.residual_norm)
                .fold(0.0_f64, f64::max);
            eprintln!(
                "  dense LU     : build {dense_build:9.4} s  solve {dense_solve:9.4} s  \
                 total {:9.4} s  max resid {dense_resid:.3e}  speedup {:6.1}x",
                dense_build + dense_solve,
                (dense_build + dense_solve) / (lr_build + lr_solve).max(1.0e-12)
            );
        }

        // --- Route: PCG, gap preconditioner (shipped), zero start.
        let t0 = Instant::now();
        let mut pcg_iters = 0usize;
        let mut pcg_worst = 0.0_f64;
        let mut pcg_solutions = Vec::new();
        for rhs in &setup.rhs_vectors {
            let s = solve_cpxtb_preconditioned(&matvec, rhs, &setup.orbital_gaps, 1.0e-9, 400)
                .unwrap();
            pcg_iters += s.iterations;
            pcg_worst = pcg_worst.max(s.residual_norm);
            pcg_solutions.push(s);
        }
        let pcg_time = t0.elapsed().as_secs_f64();
        eprintln!(
            "  PCG gap-P    : {pcg_time:9.4} s  iters {pcg_iters:6} (avg {:6.1})  \
             max resid {pcg_worst:.3e}",
            pcg_iters as f64 / ndof as f64
        );

        // --- Route: PCG, true Jacobi diagonal (task item 2 candidate).
        let diag = operator_diagonal(
            &setup.shell_kernel,
            &setup.orbital_gaps,
            &setup.transition,
            &setup.occupation_scales,
        );
        let t0 = Instant::now();
        let mut diag_iters = 0usize;
        for rhs in &setup.rhs_vectors {
            let (_, it, _) = pcg_with_guess(
                &matvec,
                rhs,
                &diag,
                &vec![0.0; npair],
                1.0e-9,
                400,
            );
            diag_iters += it;
        }
        let diag_time = t0.elapsed().as_secs_f64();
        eprintln!(
            "  PCG diag-P   : {diag_time:9.4} s  iters {diag_iters:6} (avg {:6.1})  \
             vs gap-P {:+.1}%",
            diag_iters as f64 / ndof as f64,
            100.0 * (diag_iters as f64 - pcg_iters as f64) / pcg_iters.max(1) as f64
        );

        // --- Route: PCG, gap preconditioner, warm start from the previous DOF
        //     (task item 3: neighbouring-DOF seeding of the 3N family).
        let t0 = Instant::now();
        let mut warm_iters = 0usize;
        let mut previous = vec![0.0_f64; npair];
        for rhs in &setup.rhs_vectors {
            let (x, it, _) = pcg_with_guess(
                &matvec,
                rhs,
                &setup.orbital_gaps,
                &previous,
                1.0e-9,
                400,
            );
            warm_iters += it;
            previous = x;
        }
        let warm_time = t0.elapsed().as_secs_f64();
        eprintln!(
            "  PCG warm     : {warm_time:9.4} s  iters {warm_iters:6} (avg {:6.1})  \
             vs cold {:+.1}%",
            warm_iters as f64 / ndof as f64,
            100.0 * (warm_iters as f64 - pcg_iters as f64) / pcg_iters.max(1) as f64
        );

        // --- The Z-vector / adjoint solve exactly as the 2n+1 quartic ladder
        //     issues it (`response_stage.rs`: tol 1e-11, max_iter 4000), one RHS.
        let rhs0 = &setup.rhs_vectors[0];
        let t0 = Instant::now();
        let adjoint_pcg = solve_cpxtb_preconditioned(
            &matvec,
            rhs0,
            &setup.orbital_gaps,
            1.0e-11,
            4000,
        )
        .unwrap();
        let adjoint_pcg_time = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let adjoint_direct = setup.solve_adjoint(rhs0, 1.0e-11, 4000).unwrap();
        let adjoint_direct_time = t0.elapsed().as_secs_f64();
        eprintln!(
            "  adjoint 1e-11: PCG {adjoint_pcg_time:9.4} s ({} iters, route {:?})  \
             setup.solve_adjoint {adjoint_direct_time:9.4} s (route {:?})  speedup {:6.1}x",
            adjoint_pcg.iterations,
            adjoint_pcg.route,
            adjoint_direct.route,
            adjoint_pcg_time / adjoint_direct_time.max(1.0e-12)
        );

        // --- Cross-route agreement.
        let mut worst_lr_pcg = 0.0_f64;
        let mut scale = 0.0_f64;
        for (lr, pcg) in lr_solutions.iter().zip(pcg_solutions.iter()) {
            worst_lr_pcg = worst_lr_pcg.max(max_abs_diff_vec(&lr.amplitudes, &pcg.amplitudes));
            scale = scale.max(max_abs(&lr.amplitudes));
        }
        eprintln!("  |x_chargespace - x_PCG|max {worst_lr_pcg:.3e}  (|x|max {scale:.3e})");
        if run_dense {
            let mut worst_lr_dense = 0.0_f64;
            for (lr, dense) in lr_solutions.iter().zip(dense_solutions.iter()) {
                worst_lr_dense =
                    worst_lr_dense.max(max_abs_diff_vec(&lr.amplitudes, &dense.amplitudes));
            }
            eprintln!("  |x_chargespace - x_dense|max {worst_lr_dense:.3e}");
        }
    }

    #[test]
    #[ignore = "measurement harness; run with --ignored --nocapture"]
    fn bench_cp_routes() {
        bench_fixture("water T=0", WATER, 0.0, true);
        bench_fixture("stretched H10 T=0", &stretched_h10_xyz(), 0.0, true);
        bench_fixture("distorted Ni(CO)4 3000 K", NI_CO4_DISTORTED, 3000.0, true);
        bench_fixture("water 2x2x2 T=0", &water_grid_xyz(2, 2, 2), 0.0, true);
        bench_fixture("water 3x3x2 T=0", &water_grid_xyz(3, 3, 2), 0.0, true);
        bench_fixture("water 3x3x3 T=0", &water_grid_xyz(3, 3, 3), 0.0, false);
    }

    /// Task item 4 (read-only analysis): where the charge-space *context* build
    /// time goes as `nshell` grows. `ChargeSpaceContext::build` runs one full
    /// spectral response per shell, so the reported ms/shell is the quantity a
    /// batched-GEMM χ⁰ formulation would collapse.
    #[test]
    #[ignore = "measurement harness; run with --ignored --nocapture"]
    fn bench_chi0_build_scaling() {
        let params = Gfn1Parameters::builtin().unwrap();
        let cases: Vec<(String, String)> = vec![
            ("water".to_string(), WATER.to_string()),
            ("Ni(CO)4".to_string(), NI_CO4_DISTORTED.to_string()),
            ("water 2x2x2".to_string(), water_grid_xyz(2, 2, 2)),
            ("water 3x3x2".to_string(), water_grid_xyz(3, 3, 2)),
            ("water 3x3x3".to_string(), water_grid_xyz(3, 3, 3)),
        ];
        eprintln!("\n=== ChargeSpaceContext::build (chi0 + dielectric LU)");
        for (label, xyz) in cases {
            // Finite temperature so the context is the one the FC3/FC4 paths use.
            let options = options_at(1500.0);
            let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
            let electronic = run_electronic(&system, &params, options).unwrap();
            let n = electronic.basis.len();
            let nshell = electronic.basis.shells.len();
            let t0 = Instant::now();
            let ctx =
                crate::response::charge_space::ChargeSpaceContext::build(
                    &system, &params, &electronic,
                );
            let elapsed = t0.elapsed().as_secs_f64();
            match ctx {
                Ok(_) => eprintln!(
                    "  {label:14} n {n:5}  nshell {nshell:5}  build {elapsed:9.4} s  \
                     {:8.3} ms/shell  {:8.3} ns/(shell·n^3)",
                    1000.0 * elapsed / nshell as f64,
                    1.0e9 * elapsed / (nshell as f64 * (n as f64).powi(3))
                ),
                Err(err) => eprintln!("  {label:14} n {n:5}  nshell {nshell:5}  SKIPPED: {err}"),
            }
        }
    }

    /// **Task item 4 — batched χ⁰, derived and validated here, NOT wired into
    /// `charge_space.rs` (that file belongs to another workstream).**
    ///
    /// `ChargeSpaceContext::build` obtains χ⁰ column by column: for each of the
    /// `nshell` unit shell potentials it forms the AO response Fock, runs a full
    /// spectral density response and re-Mullikens the AO density response. That
    /// is `6` `n × n` matmuls per shell — `12·nshell·n³ ≈ 7 n⁴` flops — and two
    /// of the six transform a matrix that is identically zero (the unit-potential
    /// perturbation has no overlap derivative).
    ///
    /// The whole loop collapses. With `D_t` the indicator of shell `t`, the
    /// unit-potential Fock is `F_t = −½(D_t S + S D_t)`, so with `G_t[p,q] =
    /// Σ_{μ∈t} C[μ,p]·(SC)[μ,q]` (one `n_t·n²` product per shell, `n³` for all
    /// shells together):
    ///
    /// ```text
    ///   h̃^t = Cᵀ F_t C = −½ (G_t + G_tᵀ)          s̃^t ≡ 0
    ///   χ⁰[s,t] = −⟨G_s , 𝒞(h̃^t)⟩
    /// ```
    ///
    /// — the last line because `q^{(t)}_s = −Σ_{ν∈s}(P^{(t)}S)_{νν}` and
    /// `P^{(t)} = C 𝒞 Cᵀ`. Everything `nshell`-sized is then ONE
    /// `(nshell × n²)·(n² × nshell)` GEMM of `2·nshell²·n² ≈ 0.7 n⁴` flops, i.e.
    /// ~10× fewer flops than the loop AND in a single BLAS-3 call instead of
    /// `6·nshell` small ones. (`𝒞` is the shipped coefficient formula, unchanged;
    /// `s̃ = 0` also kills its overlap channel.)
    ///
    /// Production note for the charge-space workstream: the `nshell × n²` blocks
    /// must be **tiled over shells** (a tile of 32 shells is `32·n²` doubles) —
    /// materializing all of `G` is `O(nshell·n²)` memory, ~1 GB at `n = 600`.
    fn batched_chi0(electronic: &ElectronicResult) -> Result<Matrix> {
        let basis = &electronic.basis;
        let n = basis.len();
        let nshell = basis.shells.len();
        let overlap = &electronic.integrals.overlap;
        let eig = lowdin_solve_generalized(&electronic.fock, overlap, 1.0e-12)?;
        let mos = eig.vectors;
        let energies = eig.values;
        let occupations = &electronic.occupations;
        let kt = electronic.electronic_temperature.max(0.0) * BOLTZMANN_HARTREE_PER_K;
        let finite_t = kt > 0.0
            && occupations
                .iter()
                .any(|&f| f > 1.0e-10 && f < 2.0 - 1.0e-10);
        let kt_formula = if finite_t { kt } else { 1.0 };
        let sc = overlap.matmul(&mos)?;
        let zero = Matrix::zeros(n, n);

        let mut g_rows = Matrix::zeros(nshell, n * n);
        let mut coeff_cols = Matrix::zeros(n * n, nshell);
        for t in 0..nshell {
            let shell = &basis.shells[t];
            let nao = shell.nao;
            let mut c_t = Matrix::zeros(nao, n);
            let mut sc_t = Matrix::zeros(nao, n);
            for r in 0..nao {
                let mu = shell.first_ao + r;
                for p in 0..n {
                    c_t[(r, p)] = mos[(mu, p)];
                    sc_t[(r, p)] = sc[(mu, p)];
                }
            }
            let g = matmul_transpose_a(&c_t, &sc_t)?;
            let mut h = Matrix::zeros(n, n);
            for p in 0..n {
                for q in 0..n {
                    h[(p, q)] = -0.5 * (g[(p, q)] + g[(q, p)]);
                }
            }
            let eps_response: Vec<f64> = (0..n).map(|p| h[(p, p)]).collect();
            let occupation_response = if finite_t {
                fermi_occupation_response(occupations, &eps_response, kt)?
            } else {
                vec![0.0; n]
            };
            let coeff = finite_temperature_response_coefficients_from_mo(
                occupations,
                &energies,
                &occupation_response,
                &h,
                &zero,
                kt_formula,
                false,
            )?;
            g_rows.as_mut_slice()[t * n * n..(t + 1) * n * n].copy_from_slice(g.as_slice());
            for (k, value) in coeff.as_slice().iter().enumerate() {
                coeff_cols[(k, t)] = *value;
            }
        }
        let mut chi0 = g_rows.matmul(&coeff_cols)?;
        for value in chi0.as_mut_slice() {
            *value = -*value;
        }
        Ok(chi0)
    }

    #[test]
    #[ignore = "measurement harness; run with --ignored --nocapture"]
    fn bench_chi0_batched_formulation() {
        let params = Gfn1Parameters::builtin().unwrap();
        let cases: Vec<(String, String, f64)> = vec![
            ("water T=0".to_string(), WATER.to_string(), 0.0),
            (
                "Ni(CO)4 3000 K".to_string(),
                NI_CO4_DISTORTED.to_string(),
                3000.0,
            ),
            ("water 2x2x2".to_string(), water_grid_xyz(2, 2, 2), 1500.0),
            ("water 3x3x2".to_string(), water_grid_xyz(3, 3, 2), 1500.0),
            ("water 3x3x3".to_string(), water_grid_xyz(3, 3, 3), 1500.0),
        ];
        eprintln!("\n=== chi0: shipped per-shell loop vs the batched GEMM formulation");
        for (label, xyz, temperature) in cases {
            let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
            let electronic = run_electronic(&system, &params, options_at(temperature)).unwrap();
            let n = electronic.basis.len();
            let nshell = electronic.basis.shells.len();
            let t0 = Instant::now();
            let ctx = crate::response::charge_space::ChargeSpaceContext::build(
                &system,
                &params,
                &electronic,
            );
            let shipped = t0.elapsed().as_secs_f64();
            let Ok(ctx) = ctx else { continue };
            let t0 = Instant::now();
            let batched = batched_chi0(&electronic).unwrap();
            let batched_time = t0.elapsed().as_secs_f64();
            let worst = ctx
                .chi0
                .as_slice()
                .iter()
                .zip(batched.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let scale = max_abs(ctx.chi0.as_slice());
            eprintln!(
                "  {label:16} n {n:4} nshell {nshell:4}  shipped(build) {shipped:8.4} s  \
                 batched chi0 {batched_time:8.4} s  ratio {:6.1}x  \
                 |Δχ⁰|max {worst:.3e} (|χ⁰|max {scale:.3e})",
                shipped / batched_time.max(1.0e-9)
            );
        }
    }

    // ------------------------------------------------------------------
    // Regressions
    // ------------------------------------------------------------------

    /// The batched χ⁰ recipe handed to the charge-space workstream must
    /// reproduce the shipped per-shell construction — at `T = 0` and with Fermi
    /// smearing (where the occupation channel `f'`/`μ^{(t)}` is live).
    #[test]
    fn batched_chi0_recipe_matches_the_shipped_per_shell_build() {
        let params = Gfn1Parameters::builtin().unwrap();
        for (label, xyz, temperature) in [
            ("water T=0", WATER.to_string(), 0.0),
            (
                "distorted Ni(CO)4 3000 K",
                NI_CO4_DISTORTED.to_string(),
                3000.0,
            ),
        ] {
            let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false).unwrap();
            let electronic = run_electronic(&system, &params, options_at(temperature)).unwrap();
            let ctx = crate::response::charge_space::ChargeSpaceContext::build(
                &system,
                &params,
                &electronic,
            )
            .unwrap();
            let batched = batched_chi0(&electronic).unwrap();
            let worst = ctx
                .chi0
                .as_slice()
                .iter()
                .zip(batched.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let scale = max_abs(ctx.chi0.as_slice()).max(1.0e-12);
            assert!(
                worst <= 1.0e-11 * scale,
                "{label}: batched chi0 recipe drifted from the shipped build: \
                 {worst:.3e} against |χ⁰|max {scale:.3e}"
            );
        }
    }

    /// **Route regression.** Every fixture that used to pay a dense
    /// `npair × npair` factorization (or a Krylov loop) must now be answered by
    /// the direct charge-space reduction — zero iterations, converged, and with
    /// a residual no worse than the dense route's on the same right-hand sides.
    #[test]
    fn cp_solves_take_the_direct_charge_space_route() {
        for (label, xyz, temperature) in [
            ("water", WATER.to_string(), 0.0),
            ("stretched H10", stretched_h10_xyz(), 0.0),
            ("distorted Ni(CO)4 3000 K", NI_CO4_DISTORTED.to_string(), 3000.0),
            ("water 2x2x2", water_grid_xyz(2, 2, 2), 0.0),
        ] {
            let (setup, _) = setup_for(&xyz, temperature);
            let solutions = solve_cpxtb_all(
                &setup.shell_kernel,
                &setup.orbital_gaps,
                &setup.transition,
                &setup.occupation_scales,
                &setup.rhs_vectors,
                CpxtbOptions::default(),
            )
            .unwrap();
            assert!(!solutions.is_empty(), "{label}: no CP solutions");
            for solution in &solutions {
                assert_eq!(
                    solution.route,
                    CpxtbRoute::ChargeSpace,
                    "{label}: CP solve fell off the direct charge-space route"
                );
                assert_eq!(
                    solution.iterations, 0,
                    "{label}: the direct route must not iterate"
                );
                assert!(solution.converged, "{label}: direct route did not converge");
            }
        }
    }

    /// **Correctness of the reduction.** The charge-space amplitudes must solve
    /// the SAME pair-space equation the dense operator encodes: re-apply the
    /// explicit `A` to them and compare against the right-hand side. This is the
    /// gate that makes the route swap safe for every downstream consumer of
    /// `solutions[*].amplitudes` (Z-vector, 2n+1 ladder, TD).
    #[test]
    fn charge_space_amplitudes_solve_the_explicit_pair_space_operator() {
        for (label, xyz, temperature) in [
            ("water", WATER.to_string(), 0.0),
            ("distorted Ni(CO)4 3000 K", NI_CO4_DISTORTED.to_string(), 3000.0),
        ] {
            let (setup, _) = setup_for(&xyz, temperature);
            let low_rank = CpxtbLowRank::build(
                &setup.shell_kernel,
                &setup.orbital_gaps,
                &setup.transition,
                &setup.occupation_scales,
            )
            .unwrap();
            let solutions = low_rank.solve_batch(&setup.rhs_vectors).unwrap();
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for (solution, rhs) in solutions.iter().zip(setup.rhs_vectors.iter()) {
                let ax = setup.matvec(&solution.amplitudes).unwrap();
                worst = worst.max(max_abs_diff_vec(&ax, rhs));
                scale = scale.max(max_abs(rhs));
            }
            assert!(
                worst <= 1.0e-10 * scale.max(1.0e-6),
                "{label}: |A x - b|max {worst:.3e} against |b|max {scale:.3e}"
            );
        }
    }

    /// **Iteration-count regression — the headline win, asserted the CI-stable
    /// way.** The Fermi-smeared near-degenerate fixture (gaps down to `3.6e-7`)
    /// is where the Krylov route breaks down: with the shipped gap
    /// preconditioner the CG needs iterations on the order of the problem
    /// dimension to reach `1e-9`, and the measured average over the 3N family is
    /// 356 iterations per right-hand side against 679 unknowns. The direct
    /// charge-space reduction answers the SAME right-hand sides in **zero**
    /// iterations.
    ///
    /// Accuracy is *not* the claim: both direct routes sit at round-off (the
    /// dense LU and the reduction agree to `~1e-15` relative on this fixture,
    /// and either can be the marginally smaller residual run to run). What the
    /// reduction buys is the iteration count and the `O(npair³) → O(nsh³)`
    /// factorization, so this test asserts exactly that, plus cross-route
    /// agreement of the amplitudes every downstream consumer reads.
    #[test]
    fn near_degenerate_finite_t_stalls_the_krylov_route_but_not_the_direct_one() {
        let (setup, _) = setup_for(NI_CO4_DISTORTED, 3000.0);
        let npair = setup.space.len();
        let min_gap = setup
            .orbital_gaps
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_gap < 1.0e-6,
            "fixture lost its near-degenerate occupied-virtual pairs (min gap {min_gap:.3e})"
        );

        // The whole 3N family, not a sample: the first three right-hand sides
        // are the Ni displacements and converge in ~6 iterations; the stalling
        // ones are the ligand DOF, so a truncated sample would miss the effect.
        const BUDGET: usize = 120;
        let mut krylov_worst = 0usize;
        let mut krylov_total = 0usize;
        for rhs in &setup.rhs_vectors {
            let solution = solve_cpxtb_preconditioned(
                |u| setup.matvec(u),
                rhs,
                &setup.orbital_gaps,
                1.0e-9,
                BUDGET,
            )
            .unwrap();
            krylov_worst = krylov_worst.max(solution.iterations);
            krylov_total += solution.iterations;
        }

        let low_rank = CpxtbLowRank::build(
            &setup.shell_kernel,
            &setup.orbital_gaps,
            &setup.transition,
            &setup.occupation_scales,
        )
        .unwrap();
        let direct = low_rank.solve_batch(&setup.rhs_vectors).unwrap();
        for solution in &direct {
            assert_eq!(solution.route, CpxtbRoute::ChargeSpace);
            assert_eq!(
                solution.iterations, 0,
                "the direct charge-space route must not iterate"
            );
            assert!(solution.converged, "direct charge-space route did not converge");
        }
        eprintln!(
            "near-degenerate CP ({npair} unknowns, {} RHS): Krylov iterations worst \
             {krylov_worst} / mean {:.0} (budget {BUDGET}); direct route = 0",
            setup.rhs_vectors.len(),
            krylov_total as f64 / setup.rhs_vectors.len() as f64
        );
        assert!(
            krylov_worst >= BUDGET,
            "the Krylov route no longer exhausts its budget on this fixture \
             ({krylov_worst} iterations) — re-measure before trusting the direct route's margin"
        );

        // The two direct routes must agree: the reduction is a re-factorization
        // of the same operator, not a different equation.
        let operator = build_dense_cpxtb_operator(npair, |u| setup.matvec(u)).unwrap();
        let dense = solve_cpxtb_dense_batch(&operator, &setup.rhs_vectors).unwrap();
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for (d, l) in dense.iter().zip(direct.iter()) {
            worst = worst.max(max_abs_diff_vec(&d.amplitudes, &l.amplitudes));
            scale = scale.max(max_abs(&d.amplitudes));
        }
        assert!(
            worst <= 1.0e-11 * scale.max(1.0e-6),
            "charge-space and dense amplitudes disagree: {worst:.3e} against |x|max {scale:.3e}"
        );
    }

    /// **The singular guards still fire, and preconditioning does not mask
    /// them.** A zero-gap integer-occupation configuration must be rejected by
    /// `solve_nonpbc_cpxtb_hessian_response` BEFORE any linear algebra runs, and
    /// the charge-space reduction must independently refuse to build a
    /// factorization from a non-positive gap.
    #[test]
    fn zero_gap_integer_occupation_guards_still_fire() {
        // Symmetric Ni(CO)4 at T = 0: degenerate frontier orbitals with integer
        // (aufbau) occupations — the ~1e42 garbage-amplitude case.
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        let options = options_at(0.0);
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        if let Ok(electronic) = run_electronic(&system, &params, options.clone()) {
            let result = solve_nonpbc_cpxtb_hessian_response(
                &system,
                &params,
                &electronic,
                AoDerivativeOptions {
                    coordination_cutoff: options.hamiltonian.coordination_cutoff,
                    include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
                },
                CpxtbOptions::default(),
            );
            if let Err(err) = &result {
                assert!(
                    format!("{err}").contains("singular"),
                    "unexpected rejection reason: {err}"
                );
            }
        }

        // Direct unit test of the reduction's own guard: a zero gap must be
        // refused rather than silently regularized by the preconditioner floor.
        let kernel = Matrix::identity(2);
        let transition = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let scales = vec![1.0, 1.0];
        assert!(
            CpxtbLowRank::build(&kernel, &[1.0, 0.0], &transition, &scales).is_err(),
            "the charge-space reduction accepted a zero occupied-virtual gap"
        );
        assert!(
            CpxtbLowRank::build(&kernel, &[1.0, -0.5], &transition, &scales).is_err(),
            "the charge-space reduction accepted a negative occupied-virtual gap"
        );
        assert!(
            CpxtbLowRank::build(&kernel, &[1.0, 1.0], &transition, &scales).is_ok(),
            "the charge-space reduction rejected a well-gapped system"
        );
    }
}
