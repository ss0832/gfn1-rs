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
use crate::linalg::{lowdin_solve_generalized, matrix_vector_product, Matrix};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;
use faer::linalg::solvers::Solve;
use faer::Mat as FaerMat;

const DIST_EPS: f64 = 1.0e-12;
const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_808_578_545_117e-6;
const CPXTB_PRECOND_GAP_FLOOR: f64 = 1.0e-4;
const CPXTB_DENSE_FALLBACK_MAX_DIM: usize = 2048;
const CPXTB_PCG_DIVERGENCE_FACTOR: f64 = 1.0e3;

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

#[derive(Clone, Debug)]
pub struct CpxtbSolution {
    pub amplitudes: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
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

    let dense_operator = if space.len() <= CPXTB_DENSE_FALLBACK_MAX_DIM {
        let _profile = crate::profile::scope("cphf.dense_operator");
        Some(build_dense_cpxtb_operator(space.len(), |u| {
            cpxtb_matvec_precomputed(
                &shell_kernel,
                &orbital_gaps,
                &transition,
                &coupling_occupation_scales,
                u,
            )
        })?)
    } else {
        None
    };
    let mut solutions = Vec::with_capacity(ndim);
    let mut converged = true;
    let mut max_residual_norm = 0.0_f64;
    {
        let _profile = crate::profile::scope("cphf.solve_linear");
        if let Some(operator) = &dense_operator {
            solutions = solve_cpxtb_dense_batch(operator, &rhs_vectors)?;
            for solution in &solutions {
                converged &= solution.converged;
                max_residual_norm = max_residual_norm.max(solution.residual_norm);
            }
        } else {
            for rhs in &rhs_vectors {
                let solution = solve_cpxtb_preconditioned(
                    |u| {
                        cpxtb_matvec_precomputed(
                            &shell_kernel,
                            &orbital_gaps,
                            &transition,
                            &coupling_occupation_scales,
                            u,
                        )
                    },
                    rhs,
                    &orbital_gaps,
                    cpxtb_options.tol,
                    cpxtb_options.max_iter,
                )?;
                converged &= solution.converged;
                max_residual_norm = max_residual_norm.max(solution.residual_norm);
                solutions.push(solution);
            }
        }
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
    {
        let _profile = crate::profile::scope("cphf.response_densities");
        for coord in 0..ndim {
            let solution = &solutions[coord];
            let orbital_density =
                response_density_with_space(&mos, occupations, &space, &solution.amplitudes)?;
            if finite_temperature_response {
                let mut response_fock = Matrix::zeros(basis.len(), basis.len());
                let mut shell_response = vec![0.0_f64; basis.shells.len()];
                let response_mixing = 0.35_f64;
                for _ in 0..50 {
                    let (next_density, _) = finite_temperature_density_response(
                        &mos,
                        occupations,
                        &orbital_energies,
                        &derivative_matrices[coord].h0_deriv,
                        &derivative_matrices[coord].overlap_deriv,
                        &response_fock,
                        kt,
                    )?;
                    let next_shell = response_shell_charges_from_density(
                        basis,
                        &electronic.integrals.overlap,
                        &electronic.density,
                        &next_density,
                        &derivative_matrices[coord].overlap_deriv,
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
                    let shell_potential = matrix_vector_product(&shell_kernel, &mixed_shell)?;
                    response_fock = scalar_response_fock_matrix(
                        basis,
                        &electronic.integrals.overlap,
                        &shell_potential,
                    )?;
                    shell_response = mixed_shell;
                    if shell_delta < 1.0e-12 {
                        break;
                    }
                }
                let (final_density, final_occupation) = finite_temperature_density_response(
                    &mos,
                    occupations,
                    &orbital_energies,
                    &derivative_matrices[coord].h0_deriv,
                    &derivative_matrices[coord].overlap_deriv,
                    &response_fock,
                    kt,
                )?;
                let shell_response = response_shell_charges_from_density(
                    basis,
                    &electronic.integrals.overlap,
                    &electronic.density,
                    &final_density,
                    &derivative_matrices[coord].overlap_deriv,
                )?;
                let weighted = finite_temperature_energy_weighted_response(
                    &mos,
                    occupations,
                    &final_occupation,
                    &orbital_energies,
                    &derivative_matrices[coord].h0_deriv,
                    &derivative_matrices[coord].overlap_deriv,
                    &response_fock,
                    kt,
                )?;
                density_responses.push(final_density);
                orbital_density_responses.push(orbital_density);
                energy_weighted_density_responses.push(weighted);
                shell_charge_responses.push(shell_response);
                occupation_responses.push(final_occupation);
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
            let mut shell_response = vec![0.0_f64; basis.shells.len()];
            let mut response_fock = Matrix::zeros(basis.len(), basis.len());
            for _ in 0..1 {
                shell_response = response_shell_charges_from_density(
                    basis,
                    &electronic.integrals.overlap,
                    &electronic.density,
                    &density,
                    &derivative_matrices[coord].overlap_deriv,
                )?;
                let shell_potential = matrix_vector_product(&shell_kernel, &shell_response)?;
                response_fock = scalar_response_fock_matrix(
                    basis,
                    &electronic.integrals.overlap,
                    &shell_potential,
                )?;
                break;
            }
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
    pub fn solve_adjoint(
        &self,
        rhs_like: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<CpxtbSolution> {
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
                    continue; // degenerate same-block pair: gauge-arbitrary
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

    let dense_operator = if space.len() <= CPXTB_DENSE_FALLBACK_MAX_DIM {
        Some(build_dense_cpxtb_operator(space.len(), |u| {
            cpxtb_matvec_precomputed(
                &shell_kernel,
                &orbital_gaps,
                &transition,
                &coupling_occupation_scales,
                u,
            )
        })?)
    } else {
        None
    };
    let mut solutions = Vec::with_capacity(3);
    let mut converged = true;
    let mut max_residual_norm = 0.0_f64;
    if let Some(operator) = &dense_operator {
        solutions = solve_cpxtb_dense_batch(operator, &rhs_vectors)?;
    } else {
        for rhs in &rhs_vectors {
            solutions.push(solve_cpxtb_preconditioned(
                |u| {
                    cpxtb_matvec_precomputed(
                        &shell_kernel,
                        &orbital_gaps,
                        &transition,
                        &coupling_occupation_scales,
                        u,
                    )
                },
                rhs,
                &orbital_gaps,
                cpxtb_options.tol,
                cpxtb_options.max_iter,
            )?);
        }
    }
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
    let model = ShellChargeModel::build(system, &electronic.basis, params)?;
    let mut kernel = effective_coulomb_matrix(system, &electronic.basis, &model);
    let atomic_charges = model.atomic_charges(&electronic.basis, &electronic.shell_charges);
    for (atom, &qat) in atomic_charges.iter().enumerate() {
        let count = model.atom_shell_counts[atom];
        if count == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let add = 2.0 * qat * model.hubbard_derivs[offset];
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

fn fermi_occupation_response(
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

fn finite_temperature_mo_derivatives(
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

fn orbital_energy_response_from_mo(
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

fn finite_temperature_response_coefficients_from_mo(
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
            iterations: 1,
            residual_norm,
            converged: residual_norm <= 1.0e-8_f64.max(1.0e-10 * norm(rhs)),
        });
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
