// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic molecular TDA and RPA eigenvalue and manifold derivatives.
use super::*;
use crate::fourth_derivative::assemble::{
    directional_first_order_field, directional_second_order_field,
};
use crate::response::charge_space::ChargeSpaceContext;
use crate::scalar_dual::Dual as D;

#[derive(Clone, Debug)]
pub struct TdaHessianResult {
    pub state_index: usize,
    pub ground_energy: f64,
    pub excitation_energy: f64,
    pub total_energy: f64,
    pub ground_hessian: Matrix,
    pub excitation_hessian: Matrix,
    pub hessian: Matrix,
}

#[derive(Clone, Debug)]
pub struct TdaManifoldHessianResult {
    pub roots: std::ops::Range<usize>,
    pub mean: TdaHessianResult,
}

fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}

/// Molecular total excited-state Hessian, Hartree/bohr², with a 512 MiB budget.
pub fn solve_tda_hessian_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaHessianResult> {
    solve_tda_hessian_analytic_with_budget(system, params, electronic, state, options, 536870912)
}

/// Analytic orbital/SCC second response and complete CI eigenvector response.
/// Cartesian cross derivatives use exact polarization of analytic directional
/// second derivatives. No displaced geometry or finite-difference step is used.
pub fn solve_tda_hessian_analytic_with_budget(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
    memory_budget_bytes: usize,
) -> Result<TdaHessianResult> {
    solve_response_hessian(
        system,
        params,
        electronic_options,
        state,
        options,
        memory_budget_bytes,
        ResponseMethod::Tda,
        false,
    )
    .map(|result| result.mean)
}

/// Analytic total Hessian of the mean molecular TDA manifold surface.
pub fn solve_tda_manifold_hessian_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaManifoldHessianResult> {
    solve_tda_manifold_hessian_analytic_with_budget(
        system, params, electronic, state, options, 536870912,
    )
}

/// Manifold-average Hessian with an explicit workspace budget.
pub fn solve_tda_manifold_hessian_analytic_with_budget(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
    memory_budget_bytes: usize,
) -> Result<TdaManifoldHessianResult> {
    let mut electronic = electronic.clone();
    electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
    electronic.charge_tolerance = electronic.charge_tolerance.min(1e-11);
    solve_response_hessian(
        system,
        params,
        &electronic,
        state,
        options,
        memory_budget_bytes,
        ResponseMethod::Tda,
        true,
    )
}

/// Analytic molecular RPA total Hessian in Hartree/bohr².
pub fn solve_rpa_hessian_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaHessianResult> {
    solve_response_hessian(
        system,
        params,
        electronic_options,
        state,
        options,
        536870912,
        ResponseMethod::Rpa,
        false,
    )
    .map(|result| result.mean)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_response_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
    memory_budget_bytes: usize,
    method: ResponseMethod,
    average: bool,
) -> Result<TdaManifoldHessianResult> {
    let memory_budget_bytes = crate::resources::effective_memory_budget(memory_budget_bytes);
    if system.lattice.is_some() {
        return Err(invalid(
            "analytic TDA excited-state Hessians currently require a molecule",
        ));
    }
    if electronic_options.solvation.is_some()
        || electronic_options.external_field.electric_field.is_some()
    {
        return Err(invalid("analytic molecular excited-state Hessians require a gas-phase reference without an applied field"));
    }
    if electronic_options.ml_model.is_some() {
        return Err(invalid(
            "analytic TDA Hessians do not yet include structural ML response",
        ));
    }
    crate::terms::require_order(
        electronic_options,
        params,
        3,
        "the analytic TDA excited-state Hessian",
    )?;
    let basis = crate::basis::BasisSet::build(
        system,
        params,
        crate::basis::BasisOptions {
            nprim: electronic_options.nprim,
        },
    )?;
    let n = basis.len();
    let ndof = 3 * system.atoms.len();
    let base_estimate = n
        .checked_mul(n)
        .and_then(|n2| {
            ndof.checked_mul(96)
                .and_then(|b| b.checked_add(256))
                .and_then(|b| n2.checked_mul(b))
        })
        .and_then(|bytes| {
            ndof.checked_mul(ndof)
                .and_then(|d| d.checked_mul(40))
                .and_then(|d| bytes.checked_add(d))
        })
        .ok_or_else(|| invalid("TD Hessian ground-response allocation overflow"))?;
    if base_estimate > memory_budget_bytes {
        return Err(invalid(format!(
            "TD Hessian workspace estimate {base_estimate} exceeds budget {memory_budget_bytes} before SCF"
        )));
    }
    let electronic = crate::run_electronic(system, params, electronic_options.clone())?;
    validate_tda_occupations(&electronic.occupations)?;
    let configurations =
        response_pairs(&electronic.orbital_energies, &electronic.occupations)?.len();
    // Use the actual configuration count. The dense primal CI eigensolver is
    // the only transition-square workspace needed by the derivative.
    let estimate = configurations
        .checked_mul(configurations)
        .and_then(|x| x.checked_mul(64))
        .and_then(|x| x.checked_add(base_estimate))
        .ok_or_else(|| invalid("TD Hessian CI allocation overflow"))?;
    if estimate > memory_budget_bytes {
        return Err(invalid(format!(
            "TD Hessian workspace estimate {estimate} exceeds budget {memory_budget_bytes} for {configurations} configurations"
        )));
    }
    let ground = crate::hessian::analytic_hessian_from_result(
        system,
        params,
        Some(electronic),
        crate::AnalyticHessianOptions {
            electronic_options: electronic_options.clone(),
            ..Default::default()
        },
    )?;
    let electronic = ground
        .electronic_result
        .as_ref()
        .ok_or_else(|| invalid("TD Hessian needs its ground-state reference"))?;
    let cphf = ground
        .cpxtb_response
        .as_ref()
        .ok_or_else(|| invalid("TDA Hessian needs ground-state response"))?;
    if !electronic.converged || !cphf.converged {
        return Err(invalid("TDA Hessian requires converged SCC and response"));
    }
    // Every CI root is needed in the second-order eigenvector-response sum.
    let td = solve_molecular_response(
        system,
        params,
        electronic,
        TdaOptions {
            n_states: usize::MAX,
            ..options
        },
        None,
        method,
    )?;

    let roots = td.root_manifold(state)?;
    if roots.len() > 1 && !average {
        return Err(invalid(format!(
            "excited-state Hessian requires an isolated root; state {state} belongs to a {}-fold degenerate manifold",
            roots.len()
        )));
    }
    let count = roots.len() as f64;
    let excitation_energy = td.states[roots.clone()]
        .iter()
        .map(|s| s.excitation_energy)
        .sum::<f64>()
        / count;
    let ctx = ChargeSpaceContext::build(system, params, electronic)?;
    let mo = &cphf.mos;
    let sc = electronic.integrals.overlap.matmul(mo)?;
    let model = crate::coulomb::ShellChargeModel::build(system, &electronic.basis, params)?;

    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    let response = MolecularResponseDerivatives {
        dvdr_q: Some(&dvdr_q),
        system,
        params,
        electronic,
        electronic_options,
        cphf,
        ctx: &ctx,
        pairs: &td.pairs,
        sc: &sc,
        model: &model,
        coupling: options.spin.coupling_scale(),
        method,
    };

    let amplitudes: Vec<_> = td.states[roots.clone()]
        .iter()
        .map(|root| root.amplitudes.as_slice())
        .collect();
    let lambda = |omega: f64| match method {
        ResponseMethod::Tda => omega,
        ResponseMethod::Rpa => omega * omega,
    };
    let directional = |v: &[f64]| -> Result<f64> {
        let response = response.directional(v, &amplitudes, true)?;
        let mut trace = 0.;
        for (local, (action, frozen_second)) in response.iter().enumerate() {
            let omega = td.states[roots.start + local].excitation_energy;
            let mut second = *frozen_second;
            for (index, other) in td.states.iter().enumerate() {
                let mixing = dot(&other.amplitudes, action);
                if !roots.contains(&index) {
                    second +=
                        2. * mixing * mixing / (lambda(omega) - lambda(other.excitation_energy));
                } else if method == ResponseMethod::Rpa {
                    let nu = other.excitation_energy;
                    trace -= mixing * mixing / (2. * omega * nu * (omega + nu));
                }
            }
            trace += match method {
                ResponseMethod::Tda => second,
                ResponseMethod::Rpa => second / (2. * omega),
            };
        }
        Ok(trace / count)
    };
    let scratch_per_worker =
        (n * n * 256).saturating_add(td.pairs.len() * electronic.basis.shells.len() * 72);
    let workers = (memory_budget_bytes.saturating_sub(estimate) / scratch_per_worker.max(1))
        .max(1)
        .min(crate::resources::effective_threads())
        .min(8);
    let excitation_hessian =
        directional_hessian(ndof, workers, |v| directional(v).map(|d2| (0., d2)))?.1;
    let mut hessian = ground.hessian.clone();
    for (value, &add) in hessian
        .as_mut_slice()
        .iter_mut()
        .zip(excitation_hessian.as_slice())
    {
        *value += add;
    }

    Ok(TdaManifoldHessianResult {
        roots,
        mean: TdaHessianResult {
            state_index: state,
            ground_energy: electronic.total_free,
            excitation_energy,
            total_energy: electronic.total_free + excitation_energy,
            ground_hessian: ground.hessian,
            excitation_hessian,
            hessian,
        },
    })
}

/// Analytic total Hessian of a molecular RPA manifold.
pub fn solve_rpa_manifold_hessian_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaManifoldHessianResult> {
    solve_response_hessian(
        system,
        params,
        electronic,
        state,
        options,
        536870912,
        ResponseMethod::Rpa,
        true,
    )
}

/// Analytic molecular RPA total gradient in Hartree/bohr.
pub fn solve_rpa_gradient_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaGradientResult> {
    solve_molecular_manifold_gradient(
        system,
        params,
        electronic_options,
        state,
        options,
        ResponseMethod::Rpa,
        false,
        None,
    )
    .map(|result| result.mean)
}

/// Analytic molecular RPA gradient of the mean excitation surface.
pub fn solve_rpa_manifold_gradient_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
) -> Result<TdaManifoldGradientResult> {
    solve_molecular_manifold_gradient(
        system,
        params,
        electronic,
        state,
        options,
        ResponseMethod::Rpa,
        true,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_molecular_manifold_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    state: usize,
    options: TdaOptions,
    method: ResponseMethod,
    average: bool,
    fixed_roots: Option<std::ops::Range<usize>>,
) -> Result<TdaManifoldGradientResult> {
    if system.lattice.is_some() {
        return Err(invalid(
            "analytic molecular TD nuclear derivatives require a molecule",
        ));
    }
    let ground = analytic_gradient(
        system,
        params,
        AnalyticGradientOptions {
            electronic: electronic.clone(),
            ..Default::default()
        },
    )?;
    let td = solve_molecular_response(
        system,
        params,
        &ground.electronic_result,
        options.full_spectrum(),
        None,
        method,
    )?;
    let roots = if let Some(roots) = fixed_roots {
        roots
    } else if average {
        td.root_manifold(state)?
    } else {
        td.require_isolated_root(state)?;
        state..state + 1
    };
    let mean = molecular_response_gradient(
        system,
        params,
        electronic,
        ground,
        &td,
        state,
        roots.clone(),
        options.spin,
        method,
    )?;
    Ok(TdaManifoldGradientResult {
        roots,
        effective_step_bohr: None,
        mean,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn molecular_response_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic_options: &ElectronicOptions,
    ground: crate::gradient::AnalyticGradientResult,
    td: &TdaResult,
    state: usize,
    roots: std::ops::Range<usize>,
    spin: TdaSpin,
    method: ResponseMethod,
) -> Result<TdaGradientResult> {
    crate::terms::require_order(
        electronic_options,
        params,
        2,
        "the analytic TD-GFN1 gradient",
    )?;
    if electronic_options.ml_model.is_some()
        || electronic_options.solvation.is_some()
        || electronic_options.external_field.electric_field.is_some()
    {
        return Err(invalid("analytic thermal TDA and RPA derivatives require the scalar gas-phase Hamiltonian without structural ML or an applied field"));
    }
    let states = td
        .states
        .get(roots)
        .ok_or_else(|| invalid("TD-GFN1 states are outside the computed spectrum"))?;
    let count = states.len() as f64;
    let omega = states
        .iter()
        .map(|root| root.excitation_energy)
        .sum::<f64>()
        / count;
    let amplitudes: Vec<_> = states
        .iter()
        .map(|root| root.amplitudes.as_slice())
        .collect();
    let electronic = &ground.electronic_result;
    let cphf = solve_nonpbc_cpxtb_hessian_response(
        system,
        params,
        electronic,
        AoDerivativeOptions {
            coordination_cutoff: electronic_options.hamiltonian.coordination_cutoff,
            include_cn_h0: electronic_options.hamiltonian.enable_cn_hamiltonian,
        },
        CpxtbOptions::default(),
    )?;
    if !cphf.converged {
        return Err(invalid("TD-GFN1 gradient requires converged SCC response"));
    }
    let ctx = ChargeSpaceContext::build(system, params, electronic)?;
    let sc = electronic.integrals.overlap.matmul(&cphf.mos)?;
    let model = crate::coulomb::ShellChargeModel::build(system, &electronic.basis, params)?;
    let response = MolecularResponseDerivatives {
        dvdr_q: None,
        system,
        params,
        electronic,
        electronic_options,
        cphf: &cphf,
        ctx: &ctx,
        pairs: &td.pairs,
        sc: &sc,
        model: &model,
        coupling: spin.coupling_scale(),
        method,
    };
    let mut gradient = ground.gradient;
    let mut v = vec![0.0; 3 * gradient.len()];
    for coord in 0..v.len() {
        v[coord] = 1.0;
        let action = response.directional(&v, &amplitudes, false)?;
        v[coord] = 0.0;
        let derivative = states
            .iter()
            .zip(action)
            .map(|(root, (action, _))| {
                let value = dot(&root.amplitudes, &action);
                if method == ResponseMethod::Rpa {
                    value / (2. * root.excitation_energy)
                } else {
                    value
                }
            })
            .sum::<f64>()
            / count;
        match coord % 3 {
            0 => gradient[coord / 3].x += derivative,
            1 => gradient[coord / 3].y += derivative,
            _ => gradient[coord / 3].z += derivative,
        }
    }
    Ok(TdaGradientResult {
        state_index: state,
        excitation_energy: omega,
        total_energy: ground.total_energy + omega,
        forces: gradient.iter().map(|g| -*g).collect(),
        gradient,
    })
}

struct MolecularResponseDerivatives<'a> {
    dvdr_q: Option<&'a Matrix>,
    system: &'a PeriodicSystem,
    params: &'a Gfn1Parameters,
    electronic: &'a ElectronicResult,
    electronic_options: &'a ElectronicOptions,
    cphf: &'a crate::cphf::GammaCartesianCpxtbResult,
    ctx: &'a ChargeSpaceContext,
    pairs: &'a [(usize, usize)],
    sc: &'a Matrix,
    model: &'a crate::coulomb::ShellChargeModel,
    coupling: f64,
    method: ResponseMethod,
}

impl MolecularResponseDerivatives<'_> {
    fn directional(
        &self,
        v: &[f64],
        amplitudes: &[&[f64]],
        second_order: bool,
    ) -> Result<Vec<(Vec<f64>, f64)>> {
        let Self {
            dvdr_q,
            system,
            params,
            electronic,
            electronic_options,
            cphf,
            ctx,
            pairs,
            sc,
            model,
            coupling,
            method,
        } = *self;
        let nshell = electronic.basis.shells.len();
        let mo = &cphf.mos;
        let overlap = &electronic.integrals.overlap;
        let first = directional_first_order_field(electronic, cphf, ctx, v)?;
        let second = if second_order {
            Some(
                directional_second_order_field(
                    system,
                    params,
                    electronic,
                    ctx,
                    &first,
                    electronic_options.hamiltonian.coordination_cutoff,
                    electronic_options.hamiltonian.enable_cn_hamiltonian,
                    v,
                    dvdr_q.expect("second-order geometric potential"),
                    false,
                )?
                .0,
            )
        } else {
            None
        };

        let kt = electronic.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
        let eps = &cphf.orbital_energies;
        // Keep the orbital frame smooth at the displaced geometries used for
        // derivatives. The thermal window also bounds the error of the
        // confluent occupation derivatives inside a block.
        let tolerance = if ctx.is_finite_temperature() {
            (1e-4 * kt).clamp(1e-10, THERMAL_INTERBAND_BLOCK_HARTREE)
        } else {
            THERMAL_INTERBAND_BLOCK_HARTREE
        };
        let orbital_blocks = orbital_blocks(eps, tolerance);
        let (rotation, f1, frame2) = crate::response::charge_space::directional_block_frame(
            &first,
            second.as_ref(),
            eps,
            &orbital_blocks,
        )?;
        let c1 = mo.matmul(&rotation)?;
        let mut sc1 = first.overlap_deriv.matmul(mo)?;
        let metric1 = overlap.matmul(&c1)?;
        for (value, &add) in sc1.as_mut_slice().iter_mut().zip(metric1.as_slice()) {
            *value += add;
        }

        let (c2, sc2) = if let (Some(second), Some((rotation2, _))) = (&second, &frame2) {
            let c2 = mo.matmul(rotation2)?;
            // S'' C = S C (C^T S'' C) in the complete generalized MO basis.
            let mut sc2 = sc.matmul(&second.s_dot_fixed)?;
            let cross = first.overlap_deriv.matmul(&c1)?;
            let metric2 = overlap.matmul(&c2)?;
            for (value, (&cross, &last)) in sc2
                .as_mut_slice()
                .iter_mut()
                .zip(cross.as_slice().iter().zip(metric2.as_slice()))
            {
                *value += 2. * cross + last;
            }
            (c2, sc2)
        } else {
            (
                Matrix::zeros(mo.rows(), mo.cols()),
                Matrix::zeros(sc.rows(), sc.cols()),
            )
        };
        let blocks = transition_blocks(pairs, &orbital_blocks);
        let f2 = frame2.as_ref().map(|(_, f2)| f2);
        let occupations = occupation_jet(
            &cphf.orbital_energies,
            &electronic.occupations,
            electronic.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
            &f1,
            f2,
            tolerance,
            ctx.is_finite_temperature(),
        );
        let mut transition = vec![vec![D::default(); nshell]; pairs.len()];
        for (pair, &(i, a)) in pairs.iter().enumerate() {
            for (shell_index, shell) in electronic.basis.shells.iter().enumerate() {
                for mu in shell.first_ao..shell.first_ao + shell.nao {
                    let coefficient = |p| D::seed_second(mo[(mu, p)], c1[(mu, p)], c2[(mu, p)]);
                    let metric = |p| D::seed_second(sc[(mu, p)], sc1[(mu, p)], sc2[(mu, p)]);
                    transition[pair][shell_index] -=
                        coefficient(i) * metric(a) + coefficient(a) * metric(i);
                }
            }
        }
        let mut weighted = vec![vec![D::default(); nshell]; pairs.len()];
        let mut gap_blocks = Vec::with_capacity(blocks.len());
        for block in blocks {
            let len = block.len();
            let mut gap = vec![vec![D::default(); len]; len];
            let mut weight_squared = gap.clone();
            let fock = |i, j| {
                D::seed_second(
                    if i == j { cphf.orbital_energies[i] } else { 0. },
                    f1[(i, j)],
                    f2.map_or(0., |m| m[(i, j)]),
                )
            };
            for (row, &p) in block.iter().enumerate() {
                let (i, a) = pairs[p];
                for (col, &q) in block.iter().enumerate() {
                    let (j, b) = pairs[q];
                    if i == j {
                        gap[row][col] += fock(a, b);
                        weight_squared[row][col] -= 0.5 * occupations[a][b];
                    }
                    if a == b {
                        gap[row][col] -= fock(i, j);
                        weight_squared[row][col] += 0.5 * occupations[i][j];
                    }
                }
            }
            if method == ResponseMethod::Rpa {
                weight_squared = jet_product(&gap, &weight_squared);
                gap = jet_product(&gap, &gap);
            }
            let weight = diagonal_jet_sqrt(&weight_squared);
            for (row, &p) in block.iter().enumerate() {
                for (col, &q) in block.iter().enumerate() {
                    for (value, &raw) in weighted[p].iter_mut().zip(&transition[q]) {
                        *value += weight[row][col] * raw;
                    }
                }
            }
            gap_blocks.push((block, gap));
        }
        let atomic0 = model.atomic_charges(&electronic.basis, &electronic.shell_charges);
        let atomic1 = model.atomic_charges(&electronic.basis, &first.bundle.shell_charges);
        let atomic2 = second
            .as_ref()
            .map(|s| model.atomic_charges(&electronic.basis, &s.bundle.shell_charges));
        let mut kernel = vec![vec![D::default(); nshell]; nshell];
        for i in 0..nshell {
            let ai = electronic.basis.shells[i].atom_index;
            for j in 0..=i {
                let aj = electronic.basis.shells[j].atom_index;
                let hardness =
                    crate::coulomb::harmonic_average(model.hardness[i], model.hardness[j]);
                let value = if ai == aj {
                    let offset = model.atom_offsets[ai];
                    let q = D::seed_second(
                        atomic0[ai],
                        atomic1[ai],
                        atomic2.as_ref().map_or(0., |q| q[ai]),
                    );
                    let gamma3 = model.hubbard_derivs[offset];
                    let mut value = D::constant(hardness);
                    if electronic.charge_order >= 3 {
                        value += 2. * gamma3 * q;
                    }
                    let eta = model.hardness[offset];
                    if eta.abs() > 1e-8 {
                        for order in 4..=electronic.charge_order {
                            value += eta
                                * (2. * gamma3 / eta).powi(order as i32 - 2)
                                * q.powi(order - 2);
                        }
                    }
                    value
                } else {
                    let delta = system.atoms[ai].position - system.atoms[aj].position;
                    let r2 = (0..3).fold(D::default(), |sum, k| {
                        sum + D::seed(delta.to_array()[k], v[3 * ai + k] - v[3 * aj + k]).powi(2)
                    });
                    (r2 + 1. / hardness.powi(2)).powf(-0.5)
                };
                kernel[i][j] = value;
                kernel[j][i] = value;
            }
        }
        let scale = if method == ResponseMethod::Rpa {
            2. * coupling
        } else {
            coupling
        };
        let mut results = Vec::with_capacity(amplitudes.len());
        for &x in amplitudes {
            let mut projection = vec![D::default(); nshell];
            for (q, &amplitude) in weighted.iter().zip(x) {
                for (p, &charge) in projection.iter_mut().zip(q) {
                    *p += amplitude * charge;
                }
            }
            let potential: Vec<_> = kernel
                .iter()
                .map(|row| dual_dot(row, &projection))
                .collect();
            let mut action = vec![0.; x.len()];
            for (p, q) in weighted.iter().enumerate() {
                action[p] = scale * dual_dot(q, &potential).derivative;
            }
            let mut second = scale * dual_dot(&projection, &potential).second;
            for (block, gap) in &gap_blocks {
                for (row, &p) in block.iter().enumerate() {
                    for (col, &q) in block.iter().enumerate() {
                        action[p] += gap[row][col].derivative * x[q];
                        second += x[p] * gap[row][col].second * x[q];
                    }
                }
            }
            results.push((action, second));
        }
        Ok(results)
    }
}

fn dual_dot(a: &[D], b: &[D]) -> D {
    a.iter()
        .zip(b)
        .fold(D::default(), |sum, (&a, &b)| sum + a * b)
}

fn transition_blocks(pairs: &[(usize, usize)], ids: &[usize]) -> Vec<Vec<usize>> {
    let mut blocks = std::collections::BTreeMap::<_, Vec<_>>::new();
    for (p, &(i, a)) in pairs.iter().enumerate() {
        blocks.entry((ids[i], ids[a])).or_default().push(p);
    }
    blocks.into_values().collect()
}

pub(crate) fn orbital_blocks(eps: &[f64], tol: f64) -> Vec<usize> {
    let mut start = 0;
    (0..eps.len())
        .map(|i| {
            if eps[i] - eps[start] >= tol {
                start = i;
            }
            start
        })
        .collect()
}

fn jet_product(a: &[Vec<D>], b: &[Vec<D>]) -> Vec<Vec<D>> {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).fold(D::default(), |sum, k| sum + a[i][k] * b[k][j]))
                .collect()
        })
        .collect()
}

/// Differentiate W W = B in the basis where W at the reference is diagonal.
pub(crate) fn diagonal_jet_sqrt(b: &[Vec<D>]) -> Vec<Vec<D>> {
    let n = b.len();
    let root: Vec<_> = (0..n).map(|i| b[i][i].value.sqrt()).collect();
    let mut w = vec![vec![D::default(); n]; n];
    for i in 0..n {
        for j in 0..n {
            w[i][j] = D::seed(
                if i == j { root[i] } else { 0. },
                b[i][j].derivative / (root[i] + root[j]),
            );
        }
    }
    for i in 0..n {
        for j in 0..n {
            let cross = (0..n)
                .map(|k| w[i][k].derivative * w[k][j].derivative)
                .sum::<f64>();
            w[i][j].second = (b[i][j].second - 2. * cross) / (root[i] + root[j]);
        }
    }
    w
}

pub(crate) fn occupation_jet(
    eps: &[f64],
    f: &[f64],
    kt: f64,
    first: &Matrix,
    second: Option<&Matrix>,
    tolerance: f64,
    thermal: bool,
) -> Vec<Vec<D>> {
    let n = f.len();
    let mut out = vec![vec![D::default(); n]; n];
    for (i, &value) in f.iter().enumerate() {
        out[i][i] = D::constant(value);
    }
    if !thermal {
        return out;
    }
    let ids = orbital_blocks(eps, tolerance);
    let slope: Vec<_> = f.iter().map(|&f| -f * (1. - 0.5 * f) / kt).collect();
    let sum = slope.iter().sum::<f64>();
    if sum.abs() < 1e-30 {
        return out;
    }
    let mu1 = (0..n).map(|i| slope[i] * first[(i, i)]).sum::<f64>() / sum;
    let f1 = |i, j| first[(i, j)] - if i == j { mu1 } else { 0. };
    for i in 0..n {
        for j in 0..n {
            if ids[i] != ids[j] {
                continue;
            }
            let df = 0.5 * (slope[i] + slope[j]);
            out[i][j].derivative = df * f1(i, j);
            if let Some(second) = second {
                let ddf = -0.5 * (slope[i] * (1. - f[i]) + slope[j] * (1. - f[j])) / kt;
                let square = (ids[i]..n)
                    .take_while(|&k| ids[k] == ids[i])
                    .map(|k| f1(i, k) * f1(k, j))
                    .sum::<f64>();
                out[i][j].second = df * second[(i, j)] + ddf * square;
            }
        }
    }
    if second.is_some() {
        let mu2 = (0..n).map(|i| out[i][i].second).sum::<f64>() / sum;
        for i in 0..n {
            out[i][i].second -= slope[i] * mu2;
        }
    }
    out
}

/// Evaluate independent analytic directions in bounded batches, then polarize.
pub(crate) fn directional_hessian(
    ndof: usize,
    workers: usize,
    evaluate: impl Fn(&[f64]) -> Result<(f64, f64)> + Sync,
) -> Result<(Vec<f64>, Matrix)> {
    use rayon::prelude::*;
    let solve = |&(i, j): &(usize, usize)| {
        let mut v = vec![0.; ndof];
        v[i] = 1.;
        v[j] = 1.;
        let result = evaluate(&v)?;
        if !result.0.is_finite() || !result.1.is_finite() {
            return Err(invalid("non-finite analytic excited-state response"));
        }
        Ok(result)
    };
    let process = |directions: &[(usize, usize)]| -> Result<Vec<(f64, f64)>> {
        if workers == 1 {
            directions.iter().map(&solve).collect()
        } else {
            directions.par_iter().map(&solve).collect()
        }
    };
    let directions: Vec<_> = (0..ndof).map(|i| (i, i)).collect();
    let mut diagonal = Vec::with_capacity(ndof);
    for chunk in directions.chunks(workers) {
        diagonal.extend(process(chunk)?);
    }
    let mut hessian = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        hessian[(i, i)] = diagonal[i].1;
        let directions: Vec<_> = (0..i).map(|j| (i, j)).collect();
        for chunk in directions.chunks(workers) {
            for (&(_, j), (_, d2)) in chunk.iter().zip(process(chunk)?) {
                let mixed = 0.5 * (d2 - diagonal[i].1 - diagonal[j].1);
                hessian[(i, j)] = mixed;
                hessian[(j, i)] = mixed;
            }
        }
    }
    Ok((diagonal.into_iter().map(|(d1, _)| d1).collect(), hessian))
}

#[cfg(test)]
mod thermal_audit {
    use super::*;

    #[test]
    #[ignore = "full Ni(CO)4 TDA Hessian Richardson audit"]
    fn nickel_carbonyl_tda_richardson() {
        use rayon::prelude::*;

        let system =
            PeriodicSystem::from_xyz_str(include_str!("../../examples/ni_co4.xyz"), 0., false)
                .unwrap();
        let params = Gfn1Parameters::builtin().unwrap();
        let warm = std::env::var_os("GFN1_AUDIT_WARM_SCC").is_some();
        let raw = std::env::var_os("GFN1_AUDIT_RAW").is_some();
        let mut electronic = ElectronicOptions {
            electronic_temperature: 300.,
            max_scc: 1500,
            energy_tolerance: 1e-14,
            charge_tolerance: 3e-14,
            ..Default::default()
        };
        if warm {
            let central = crate::run_electronic(&system, &params, electronic.clone()).unwrap();
            assert!(central.converged);
            electronic.scc_initial_shell_charges = Some(central.shell_charges);
        }
        let options = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let n = system.atoms.len() * 3;
        let hessian = solve_response_hessian(
            &system,
            &params,
            &electronic,
            0,
            options,
            536870912,
            ResponseMethod::Tda,
            true,
        )
        .unwrap();
        let roots = hessian.roots;
        let analytic = hessian.mean.hessian;
        let scale = analytic
            .as_slice()
            .iter()
            .fold(0_f64, |acc, v| acc.max(v.abs()));
        println!(
            "warm_scc={warm} roots={roots:?} entries={} analytic maximum={scale:.12e}",
            n * n
        );
        if raw {
            println!("ANALYTIC {:?}", analytic.as_slice());
        }
        let steps = [1.28e-7, 6.4e-8, 3.2e-8, 1.6e-8, 8e-9, 4e-9, 2e-9];
        let mut table: Vec<_> = steps
            .into_par_iter()
            .map(|step| {
                let columns: Vec<Vec<f64>> = (0..n)
                    .into_par_iter()
                    .map(|column| {
                        let gradient = |sign| {
                            let mut displaced = system.clone();
                            shift_atom(&mut displaced, column / 3, column % 3, sign * step);
                            solve_molecular_manifold_gradient(
                                &displaced,
                                &params,
                                &electronic,
                                0,
                                options,
                                ResponseMethod::Tda,
                                true,
                                Some(roots.clone()),
                            )
                            .unwrap()
                            .mean
                            .gradient
                        };
                        gradient(1.)
                            .iter()
                            .zip(gradient(-1.))
                            .flat_map(|(p, m)| ((*p - m) / (2. * step)).to_array())
                            .collect()
                    })
                    .collect();
                let mut matrix = Matrix::zeros(n, n);
                for (col, values) in columns.iter().enumerate() {
                    for (row, &value) in values.iter().enumerate() {
                        matrix[(row, col)] = value;
                    }
                }
                if raw {
                    println!("RAW {step:e} {:?}", matrix.as_slice());
                }
                matrix
            })
            .collect();
        let mut best = f64::INFINITY;
        for order in 0..steps.len() {
            let factor = 4_f64.powi(order as i32 + 1);
            for (first, estimate) in table.iter().enumerate() {
                let (index, error) = estimate
                    .as_slice()
                    .iter()
                    .zip(analytic.as_slice())
                    .enumerate()
                    .map(|(i, (fd, a))| (i, (fd - a).abs()))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                best = best.min(error / scale);
                println!(
                    "order={order} steps={:.2e}..{:.2e} max={error:.12e} rel={:.12e} at=({}, {})",
                    steps[first],
                    steps[first + order],
                    error / scale,
                    index / n,
                    index % n,
                );
            }
            table = table
                .windows(2)
                .map(|pair| {
                    let mut extrapolated = pair[1].clone();
                    for (x, &coarse) in extrapolated
                        .as_mut_slice()
                        .iter_mut()
                        .zip(pair[0].as_slice())
                    {
                        *x = (factor * *x - coarse) / (factor - 1.);
                    }
                    extrapolated
                })
                .collect();
        }
        assert!(
            best < 1e-5,
            "Ni TDA Hessian maximum scaled error {best:.3e}"
        );
    }

    #[test]
    #[ignore = "full Ni(CO)4 finite-temperature Hessian audit"]
    fn nickel_carbonyl_thermal_manifold_all_components() {
        let system =
            PeriodicSystem::from_xyz_str(include_str!("../../examples/ni_co4.xyz"), 0., false)
                .unwrap();
        let params = Gfn1Parameters::builtin().unwrap();
        let electronic = ElectronicOptions {
            electronic_temperature: 300.,
            max_scc: 1500,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-12,
            ..Default::default()
        };
        let options = TdaOptions {
            n_states: 5,
            spin: TdaSpin::Singlet,
        };
        let n = system.atoms.len() * 3;
        for method in [ResponseMethod::Tda, ResponseMethod::Rpa] {
            let reference = solve_molecular_manifold_gradient(
                &system,
                &params,
                &electronic,
                0,
                options,
                method,
                true,
                None,
            )
            .unwrap();
            let hessian = solve_response_hessian(
                &system,
                &params,
                &electronic,
                0,
                options,
                536870912,
                method,
                true,
            )
            .unwrap();
            assert_eq!(reference.roots, hessian.roots);
            let mut max_gradient = 0_f64;
            let mut max_hessian = 0_f64;
            let mut max_scaled = 0_f64;
            let mut hessian_scale = 1_f64;
            for column in 0..n {
                let mut estimates = Vec::new();
                let steps = if method == ResponseMethod::Tda {
                    [8e-9, 4e-9]
                } else {
                    [2e-6, 1e-6]
                };
                for step in steps {
                    let evaluate = |sign| {
                        let mut displaced = system.clone();
                        shift_atom(&mut displaced, column / 3, column % 3, sign * step);
                        solve_molecular_manifold_gradient(
                            &displaced,
                            &params,
                            &electronic,
                            0,
                            options,
                            method,
                            true,
                            Some(reference.roots.clone()),
                        )
                        .unwrap()
                        .mean
                    };
                    let plus = evaluate(1.);
                    let minus = evaluate(-1.);
                    // The ground energy is tens of Hartree. A nanobohr
                    // stencil is needed only for the very close CI roots.
                    let energy_fd =
                        (plus.excitation_energy - minus.excitation_energy) / (2. * step);
                    let grad_fd: Vec<_> = plus
                        .gradient
                        .iter()
                        .zip(&minus.gradient)
                        .flat_map(|(p, m)| ((*p - *m) / (2. * step)).to_array())
                        .collect();
                    estimates.push((energy_fd, grad_fd));
                }
                let ground_energy = |step| {
                    let mut displaced = system.clone();
                    shift_atom(&mut displaced, column / 3, column % 3, step);
                    crate::run_electronic(&displaced, &params, electronic.clone())
                        .unwrap()
                        .total_free
                };
                let large_step = 1e-4;
                let ground_fd =
                    (ground_energy(large_step) - ground_energy(-large_step)) / (2. * large_step);
                let grad_ref = ground_fd + (4. * estimates[1].0 - estimates[0].0) / 3.;
                let grad = reference.mean.gradient[column / 3].to_array()[column % 3];
                max_gradient = max_gradient.max((grad - grad_ref).abs());
                for row in 0..n {
                    let fd = (4. * estimates[1].1[row] - estimates[0].1[row]) / 3.;
                    let analytic = hessian.mean.hessian[(row, column)];
                    let error = (analytic - fd).abs();
                    max_hessian = max_hessian.max(error);
                    hessian_scale = hessian_scale.max(fd.abs());
                    max_scaled = max_scaled.max(error / (1. + fd.abs()));
                }
            }
            let relative = max_hessian / hessian_scale;
            eprintln!("Ni(CO)4 300 K {method:?}, roots {:?}: all {n} gradients max {max_gradient:.6e}; all {} Hessian entries max {max_hessian:.6e}, relative to matrix maximum {relative:.6e}, componentwise scaled max {max_scaled:.6e}", reference.roots, n*n);
            assert!(max_gradient < 1e-5);
            assert!(relative < 3e-3 && max_scaled < 0.1);
        }
    }
}
