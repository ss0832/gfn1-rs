// SPDX-License-Identifier: GPL-3.0-or-later
//! Nuclear derivative of a locally fixed sTDA-xTB CI root, Ω'=XᵀA'X.
//! All orbital-generator, metric, response-kernel and excluded-configuration
//! perturbative contributions are analytic. No displaced calculation is used.
use super::dual::Dual as D;
use super::ground::{ground_directional, ground_state_at_temperature, tables};
#[cfg(test)]
use super::solve_stda_xtb_native;
use super::{NativeStdaOptions, NativeStdaResult};
use crate::error::{Gfn1Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct NativeStdaGradientResult {
    pub state: usize,
    pub excitation_energy: f64,
    pub roots: std::ops::Range<usize>,
    /// dΩ/dR in Hartree/bohr. This is not a ground-plus-excitation PES gradient.
    pub excitation_gradient: Vec<Vec3>,
    pub spectrum: NativeStdaResult,
}
#[derive(Clone, Debug)]
pub struct NativeStdaTotalGradientResult {
    pub total_energy: f64,
    pub ground_energy: f64,
    pub excitation_energy: f64,
    pub roots: std::ops::Range<usize>,
    pub thermal_response: bool,
    pub nonpositive_ci_roots: usize,
    pub lowest_ci_eigenvalue_hartree: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub excitation_gradient: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct NativeStdaHessianResult {
    pub state: usize,
    pub excitation_energy: f64,
    pub roots: std::ops::Range<usize>,
    pub excitation_gradient: Vec<Vec3>,
    pub excitation_hessian: Matrix,
    pub spectrum: NativeStdaResult,
}

#[derive(Clone, Debug)]
pub struct NativeStdaTotalHessianResult {
    pub state: usize,
    pub ground_energy: f64,
    pub excitation_energy: f64,
    pub roots: std::ops::Range<usize>,
    pub thermal_response: bool,
    pub nonpositive_ci_roots: usize,
    pub lowest_ci_eigenvalue_hartree: f64,
    pub total_energy: f64,
    pub ground_hessian: Matrix,
    pub excitation_hessian: Matrix,
    pub hessian: Matrix,
}

/// Analytic isolated excitation-energy Hessian. Directions are differentiated
/// analytically at one geometry, then polarized into Cartesian components.
pub fn solve_stda_xtb_hessian(
    system: &PeriodicSystem,
    state: usize,
    options: &NativeStdaOptions,
) -> Result<NativeStdaHessianResult> {
    let (gradient, hessian) = solve_stda_response(system, state, options, true)?;
    Ok(NativeStdaHessianResult {
        state,
        excitation_energy: gradient.excitation_energy,
        roots: gradient.roots,
        excitation_gradient: gradient.excitation_gradient,
        excitation_hessian: hessian.expect("second-order response requested"),
        spectrum: gradient.spectrum,
    })
}

/// Explicit composite GFN1 free energy plus dedicated sTDA excitation Hessian.
pub fn solve_stda_xtb_total_hessian(
    system: &PeriodicSystem,
    params: &crate::Gfn1Parameters,
    electronic: &crate::ElectronicOptions,
    state: usize,
    options: &NativeStdaOptions,
) -> Result<NativeStdaTotalHessianResult> {
    if (electronic.electronic_temperature - options.electronic_temperature).abs() > 1e-12 {
        return Err(invalid(
            "GFN1+sTDA Hessians require matching electronic temperatures",
        ));
    }
    let excitation = solve_stda_xtb_hessian(system, state, options)?;
    let ground = crate::analytic_hessian(
        system,
        params,
        crate::AnalyticHessianOptions {
            electronic_options: electronic.clone(),
            ..Default::default()
        },
    )?;
    let ground_energy = ground
        .electronic_result
        .as_ref()
        .ok_or_else(|| invalid("sTDA composite Hessian needs ground-state energy"))?
        .total_free;
    let mut hessian = ground.hessian.clone();
    for (value, &add) in hessian
        .as_mut_slice()
        .iter_mut()
        .zip(excitation.excitation_hessian.as_slice())
    {
        *value += add;
    }
    Ok(NativeStdaTotalHessianResult {
        state,
        ground_energy,
        excitation_energy: excitation.excitation_energy,
        roots: excitation.roots,
        thermal_response: excitation.spectrum.thermal_response,
        nonpositive_ci_roots: excitation.spectrum.nonpositive_ci_roots,
        lowest_ci_eigenvalue_hartree: excitation.spectrum.lowest_ci_eigenvalue_hartree,
        total_energy: ground_energy + excitation.excitation_energy,
        ground_hessian: ground.hessian,
        excitation_hessian: excitation.excitation_hessian,
        hessian,
    })
}
/// Explicit composite PES E_GFN1,free + Ω_sTDA-xTB, with analytic gradients of
/// both terms. The dedicated orbital-generator eigenvalue sum is not a fitted
/// ground-state total energy and is never substituted for E_GFN1 here.
pub fn solve_stda_xtb_total_gradient(
    system: &PeriodicSystem,
    params: &crate::Gfn1Parameters,
    electronic: &crate::ElectronicOptions,
    state: usize,
    options: &NativeStdaOptions,
) -> Result<NativeStdaTotalGradientResult> {
    if (electronic.electronic_temperature - options.electronic_temperature).abs() > 1e-12 {
        return Err(invalid(
            "GFN1+sTDA composite gradients require matching electronic temperatures for both terms",
        ));
    }
    let excitation = solve_stda_xtb_gradient(system, state, options)?;
    let ground = crate::analytic_gradient(
        system,
        params,
        crate::AnalyticGradientOptions {
            electronic: electronic.clone(),
            ..Default::default()
        },
    )?;
    let gradient: Vec<_> = ground
        .gradient
        .iter()
        .zip(&excitation.excitation_gradient)
        .map(|(g, e)| *g + *e)
        .collect();
    let forces = gradient.iter().map(|g| -*g).collect();
    Ok(NativeStdaTotalGradientResult {
        total_energy: ground.total_energy + excitation.excitation_energy,
        ground_energy: ground.total_energy,
        excitation_energy: excitation.excitation_energy,
        roots: excitation.roots,
        thermal_response: excitation.spectrum.thermal_response,
        nonpositive_ci_roots: excitation.spectrum.nonpositive_ci_roots,
        lowest_ci_eigenvalue_hartree: excitation.spectrum.lowest_ci_eigenvalue_hartree,
        gradient,
        forces,
        excitation_gradient: excitation.excitation_gradient,
    })
}
fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn pair(i: usize, j: usize) -> usize {
    let (a, b) = if i >= j { (i, j) } else { (j, i) };
    a * (a + 1) / 2 + b
}
fn dot(a: &[D], b: &[D]) -> D {
    a.iter().zip(b).fold(D::default(), |s, (a, b)| s + *a * *b)
}
fn matvec(m: &[Vec<D>], q: &[D]) -> Vec<D> {
    m.iter().map(|r| dot(r, q)).collect()
}
pub(super) fn sqrt_and_derivatives(
    eig: &crate::linalg::EigenDecomposition,
    ds: &Matrix,
    d2s: Option<&Matrix>,
) -> Result<(Matrix, Matrix, Matrix)> {
    let u = &eig.vectors;
    let d = u.transpose().matmul(ds)?.matmul(u)?;
    let n = u.rows();
    let mut sqrt = Matrix::zeros(n, n);
    let mut derivative = Matrix::zeros(n, n);
    for i in 0..n {
        if eig.values[i] <= 1e-10 {
            return Err(invalid("sTDA Lowdin metric is singular"));
        }
        sqrt[(i, i)] = eig.values[i].sqrt();
        for j in 0..n {
            derivative[(i, j)] = d[(i, j)] / (eig.values[i].sqrt() + eig.values[j].sqrt());
        }
    }
    let second = if let Some(d2s) = d2s {
        let d2 = u.transpose().matmul(d2s)?.matmul(u)?;
        let product = derivative.matmul(&derivative)?;
        let mut value = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                value[(i, j)] = (d2[(i, j)] - 2. * product[(i, j)])
                    / (eig.values[i].sqrt() + eig.values[j].sqrt());
            }
        }
        u.matmul(&value)?.matmul(&u.transpose())?
    } else {
        Matrix::zeros(0, 0)
    };
    Ok((
        u.matmul(&sqrt)?.matmul(&u.transpose())?,
        u.matmul(&derivative)?.matmul(&u.transpose())?,
        second,
    ))
}

/// Analytic excitation-energy gradient, with the same zero-based state convention
/// as TD-GFN1. Fitted orbital generation is independent of GFN1 calculator options.
/// A hard configuration boundary or a degenerate selected orbital/root does not
/// define a unique derivative of this discrete, orbital-dependent model.
pub fn solve_stda_xtb_gradient(
    system: &PeriodicSystem,
    state: usize,
    options: &NativeStdaOptions,
) -> Result<NativeStdaGradientResult> {
    solve_stda_response(system, state, options, false).map(|(gradient, _)| gradient)
}

fn root_range(
    spectrum: &NativeStdaResult,
    state: usize,
    options: &NativeStdaOptions,
) -> Result<std::ops::Range<usize>> {
    let root = spectrum
        .states
        .get(state)
        .ok_or_else(|| invalid("sTDA state is outside the computed spectrum"))?;
    if root.excitation_ev > options.energy_window_ev {
        return Err(invalid("sTDA state is outside the requested energy window"));
    }
    let nearby = |index: usize| {
        (spectrum.states[index].excitation_hartree - root.excitation_hartree).abs() < 1e-8
    };
    let mut first = state;
    let mut end = state + 1;
    while first > 0 && nearby(first - 1) {
        first -= 1;
    }
    while end < spectrum.states.len() && nearby(end) {
        end += 1;
    }
    if end - first > 1 && !options.manifold_average {
        return Err(invalid(
            "sTDA excitation is degenerate; use manifold averaging for its mean surface",
        ));
    }
    Ok(first..end)
}

fn solve_stda_response(
    system: &PeriodicSystem,
    state: usize,
    options: &NativeStdaOptions,
    second_order: bool,
) -> Result<(NativeStdaGradientResult, Option<Matrix>)> {
    let mut options = options.clone();
    options.memory_budget_bytes =
        crate::resources::effective_memory_budget(options.memory_budget_bytes);
    if second_order {
        let dimensions = 3 * system.atoms.len();
        let output = dimensions
            .checked_mul(dimensions)
            .and_then(|d| d.checked_mul(64))
            .ok_or_else(|| invalid("sTDA Hessian dimensions overflow"))?;
        options.memory_budget_bytes =
            options
                .memory_budget_bytes
                .checked_sub(output)
                .ok_or_else(|| {
                    invalid("sTDA Hessian output exceeds memory budget before orbital allocation")
                })?;
    }
    options.n_states = options.n_states.max(
        state
            .checked_add(2)
            .ok_or_else(|| invalid("sTDA state overflow"))?,
    );
    let bounded_ao = options
        .max_ao
        .min((options.memory_budget_bytes / 2048).isqrt());
    let orbital = ground_state_at_temperature(system, bounded_ao, options.electronic_temperature)?;
    if super::thermal::use_ensemble(&orbital, &options)? {
        let ensemble = super::thermal::Ensemble::new(system, &orbital, &options)?;
        let spectrum = ensemble.solve(true)?;
        let roots = root_range(&spectrum, state, &options)?;
        let excitation_energy = spectrum.states[roots.clone()]
            .iter()
            .map(|root| root.excitation_hartree)
            .sum::<f64>()
            / roots.len() as f64;
        let (excitation_gradient, hessian) =
            super::thermal::derivatives(&ensemble, roots.clone(), &spectrum, second_order)?;
        return Ok((
            NativeStdaGradientResult {
                state,
                excitation_energy,
                roots,
                excitation_gradient,
                spectrum,
            },
            hessian,
        ));
    }
    // Derivatives follow the signed, energy-ordered CI branch. Absorption's
    // positive-energy filter must never relabel an adiabatic state at zero.
    let mut result = super::native::solve_stda_xtb_native_response(system, &options)?;
    let roots = root_range(&result, state, &options)?;
    if roots.len() > 1 {
        return Err(invalid("legacy sTDA manifold derivatives require the ensemble response; use positive temperature and --stda-thermal"));
    }
    let root = &result.states[state];
    let metric = symmetric_eigen(&orbital.overlap)?;
    let nat = system.atoms.len();
    let n = orbital.aos.len();
    let threshold = options.energy_window_ev / 27.211385050;
    let range = 2. * (1. + 0.8 * 0.5) * threshold;
    let occupied: Vec<_> = (0..orbital.nocc)
        .filter(|&i| orbital.energies[i] > orbital.energies[orbital.nocc] - range)
        .collect();
    let virtuals: Vec<_> = (orbital.nocc..n)
        .filter(|&a| {
            orbital.energies[a] <= 3.
                && orbital.energies[a] < orbital.energies[orbital.nocc - 1] + range
        })
        .collect();
    let active: Vec<_> = occupied.iter().chain(&virtuals).copied().collect();
    let nmo = active.len();
    let npair = nmo * (nmo + 1) / 2;
    // Dual values plus derivatives double the cache size of the spectrum path.
    if npair
        .checked_mul(nat)
        .and_then(|x| x.checked_mul(72))
        .and_then(|x| {
            n.checked_mul(n)
                .and_then(|n2| n2.checked_mul(1024))
                .and_then(|workspace| x.checked_add(workspace))
        })
        .and_then(|x| {
            result
                .states
                .len()
                .checked_mul(result.configurations.len())
                .and_then(|m| m.checked_mul(8))
                .and_then(|m| x.checked_add(m))
        })
        .is_none_or(|x| x > options.memory_budget_bytes)
    {
        return Err(invalid(
            "sTDA gradient charge cache exceeds memory_budget_bytes",
        ));
    }
    let mut map = vec![usize::MAX; n];
    for (i, &a) in active.iter().enumerate() {
        map[a] = i;
    }
    let selected: Vec<_> = result
        .configurations
        .iter()
        .map(|&(i, a)| (map[i], map[a]))
        .collect();
    let mut gradient = vec![Vec3::zero(); nat];
    let ndof = 3 * nat;
    let mut directions = Vec::with_capacity(if second_order {
        ndof * (ndof + 1) / 2
    } else {
        ndof
    });
    for i in 0..ndof {
        directions.push((i, None));
    }
    if second_order {
        for i in 0..ndof {
            for j in 0..i {
                directions.push((i, Some(j)));
            }
        }
    }
    let mut curvatures = Vec::with_capacity(if second_order { directions.len() } else { 0 });
    for (direction, &(first_axis, second_axis)) in directions.iter().enumerate() {
        let mut vector = vec![0.; ndof];
        vector[first_axis] = 1.;
        if let Some(axis) = second_axis {
            vector[axis] = 1.;
        }
        let response = ground_directional(system, &orbital, &vector, second_order)?;
        if response
            .degenerate_pairs
            .iter()
            .any(|&(i, j)| map[i] != usize::MAX || map[j] != usize::MAX)
        {
            return Err(invalid("sTDA analytic gradient has a degenerate active orbital block; the fitted orbital-dependent Kia/selection model requires a specified orbital gauge"));
        }
        let (sqrt, dsqrt, d2sqrt) = sqrt_and_derivatives(
            &metric,
            &response.ds,
            if second_order {
                Some(&response.d2s)
            } else {
                None
            },
        )?;
        let clow = sqrt.matmul(&orbital.coefficients)?;
        let mut dclow = dsqrt.matmul(&orbital.coefficients)?;
        let dc = sqrt.matmul(&response.dc)?;
        for (a, b) in dclow.as_mut_slice().iter_mut().zip(dc.as_slice()) {
            *a += b;
        }
        let d2clow = if second_order {
            let mut value = d2sqrt.matmul(&orbital.coefficients)?;
            let cross = dsqrt.matmul(&response.dc)?;
            let last = sqrt.matmul(&response.d2c)?;
            for (a, (&b, &c)) in value
                .as_mut_slice()
                .iter_mut()
                .zip(cross.as_slice().iter().zip(last.as_slice()))
            {
                *a += 2. * b + c;
            }
            value
        } else {
            Matrix::zeros(0, 0)
        };
        let mut q = vec![vec![D::default(); nat]; npair];
        for i in 0..nmo {
            for j in 0..=i {
                for mu in 0..n {
                    q[pair(i, j)][orbital.aos[mu].atom_index] += D::seed_second(
                        clow[(mu, active[i])],
                        dclow[(mu, active[i])],
                        if second_order {
                            d2clow[(mu, active[i])]
                        } else {
                            0.
                        },
                    ) * D::seed_second(
                        clow[(mu, active[j])],
                        dclow[(mu, active[j])],
                        if second_order {
                            d2clow[(mu, active[j])]
                        } else {
                            0.
                        },
                    );
                }
            }
        }
        let mut gj = vec![vec![D::default(); nat]; nat];
        let mut gk = gj.clone();
        for i in 0..nat {
            for j in 0..=i {
                let eta = 0.5
                    * (tables()["default_hardness"][usize::from(system.atoms[i].z) - 1]
                        .as_f64()
                        .expect("hardness")
                        + tables()["default_hardness"][usize::from(system.atoms[j].z) - 1]
                            .as_f64()
                            .expect("hardness"));
                let r = system.atoms[i].position - system.atoms[j].position;
                let length = r.norm();
                let velocity: [f64; 3] =
                    std::array::from_fn(|k| vector[3 * i + k] - vector[3 * j + k]);
                let dr = if i == j {
                    0.
                } else {
                    (0..3).map(|k| r.to_array()[k] * velocity[k]).sum::<f64>() / length
                };
                let d2r = if i == j {
                    0.
                } else {
                    (velocity.iter().map(|v| v * v).sum::<f64>() - dr * dr) / length
                };
                let distance = D::seed_second(length, dr, if second_order { d2r } else { 0. });
                let jv = (distance.powi(4) + (0.5 * eta).powi(-4)).powf(-0.25);
                let kv = (distance.powi(2) + eta.powi(-2)).powf(-0.5);
                gj[i][j] = jv;
                gj[j][i] = jv;
                gk[i][j] = kv;
                gk[j][i] = kv;
            }
        }
        let jq: Vec<_> = q.iter().map(|v| matvec(&gj, v)).collect();
        let kq: Vec<_> = q.iter().map(|v| matvec(&gk, v)).collect();
        let ak = if options.triplet { 0. } else { 2. };
        let coupling = |i: usize, a: usize, j: usize, b: usize| {
            ak * dot(&q[pair(i, a)], &kq[pair(j, b)]) - dot(&q[pair(i, j)], &jq[pair(a, b)])
        };
        let mut diagonal = vec![D::default(); occupied.len() * virtuals.len()];
        for i in 0..occupied.len() {
            for v in 0..virtuals.len() {
                let a = occupied.len() + v;
                let ek = dot(&q[pair(i, a)], &kq[pair(i, a)]);
                let shift = (0.5 * 0.036749320) / (1. + (ek / (0.1 * 0.036749320)).powi(4));
                diagonal[i * virtuals.len() + v] = D::seed_second(
                    orbital.energies[active[a]] - orbital.energies[active[i]],
                    response.de[active[a]] - response.de[active[i]],
                    if second_order {
                        response.d2e[active[a]] - response.d2e[active[i]]
                    } else {
                        0.
                    },
                ) + coupling(i, a, i, a)
                    + shift;
            }
        }
        let diagonal_at = |i: usize, a: usize| diagonal[i * virtuals.len() + a - occupied.len()];
        let mut selected_diagonals: Vec<_> =
            selected.iter().map(|&(i, a)| diagonal_at(i, a)).collect();
        for (index, &(i, a)) in selected
            .iter()
            .take(result.primary_configuration_count)
            .enumerate()
        {
            if (diagonal_at(i, a).value - threshold).abs() < 1e-9 {
                return Err(invalid(
                    "sTDA configuration lies on the hard primary-selection boundary",
                ));
            }
            for j in 0..occupied.len() {
                for v in 0..virtuals.len() {
                    let b = occupied.len() + v;
                    if selected.contains(&(j, b)) {
                        continue;
                    }
                    let d = diagonal_at(j, b) - diagonal_at(i, a) + 1e-10;
                    selected_diagonals[index] -= coupling(j, b, i, a).powi(2) / d;
                }
            }
        }
        let mut derivative = 0.;
        let mut curvature = 0.;
        let mut action = if second_order {
            vec![0.; selected.len()]
        } else {
            Vec::new()
        };
        for (i, &(oi, va)) in selected.iter().enumerate() {
            derivative += root.amplitudes[i].powi(2) * selected_diagonals[i].derivative;
            if second_order {
                curvature += root.amplitudes[i].powi(2) * selected_diagonals[i].second;
                action[i] += selected_diagonals[i].derivative * root.amplitudes[i];
            }
            for (j, &(oj, vb)) in selected.iter().enumerate().take(i) {
                let entry = coupling(oi, va, oj, vb);
                derivative += 2. * root.amplitudes[i] * root.amplitudes[j] * entry.derivative;
                if second_order {
                    curvature += 2. * root.amplitudes[i] * root.amplitudes[j] * entry.second;
                    action[i] += entry.derivative * root.amplitudes[j];
                    action[j] += entry.derivative * root.amplitudes[i];
                }
            }
        }
        if direction < ndof {
            match direction % 3 {
                0 => gradient[direction / 3].x = derivative,
                1 => gradient[direction / 3].y = derivative,
                _ => gradient[direction / 3].z = derivative,
            }
        }
        if second_order {
            for (index, other) in result.states.iter().enumerate() {
                if index == state {
                    continue;
                }
                let mixing: f64 = other
                    .amplitudes
                    .iter()
                    .zip(&action)
                    .map(|(a, b)| a * b)
                    .sum();
                curvature +=
                    2. * mixing * mixing / (root.excitation_hartree - other.excitation_hartree);
            }
            curvatures.push(curvature);
        }
    }
    let hessian = if second_order {
        let mut hessian = Matrix::zeros(ndof, ndof);
        for i in 0..ndof {
            hessian[(i, i)] = curvatures[i];
        }
        let mut index = ndof;
        for i in 0..ndof {
            for j in 0..i {
                let value = 0.5 * (curvatures[index] - curvatures[i] - curvatures[j]);
                index += 1;
                hessian[(i, j)] = value;
                hessian[(j, i)] = value;
            }
        }
        Some(hessian)
    } else {
        None
    };
    let excitation_energy = root.excitation_hartree;
    // Preserve the ordinary absorption spectrum, while the separate signed
    // eigenvalues retain the derivative branch's identity, including at zero.
    result
        .states
        .retain(|s| s.excitation_hartree > 0. && s.excitation_ev <= options.energy_window_ev);
    result.states.truncate(options.n_states);
    Ok((
        NativeStdaGradientResult {
            state,
            roots,
            excitation_energy,
            excitation_gradient: gradient,
            spectrum: result,
        },
        hessian,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_ci_branch_is_differentiable_across_zero_without_relabeling() {
        let input: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/stda-ci-instability.json"))
                .unwrap();
        let mut xyz = String::from("6\nCI instability regression\n");
        for (number, position) in input["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .zip(input["positions_angstrom"].as_array().unwrap())
        {
            xyz.push_str(&format!(
                "{} {} {} {}\n",
                crate::z_to_symbol(number.as_u64().unwrap() as u8).unwrap(),
                position[0],
                position[1],
                position[2]
            ));
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0., false).unwrap();
        let options = NativeStdaOptions {
            electronic_temperature: 300.,
            energy_window_ev: 20.,
            ..Default::default()
        };
        let spectrum = solve_stda_xtb_native(&system, &options).unwrap();
        assert!(spectrum.nonpositive_ci_roots > 0 && spectrum.lowest_ci_eigenvalue_hartree < 0.);
        assert!(spectrum.states[0].excitation_ev > 1.);
        let response = solve_stda_xtb_gradient(&system, 0, &options).unwrap();
        assert!((response.excitation_energy - spectrum.lowest_ci_eigenvalue_hartree).abs() < 1e-12);
        let mut maximum = 0.0_f64;
        for coordinate in 0..18 {
            let mut differences = Vec::new();
            for step in [2e-4, 1e-4] {
                let mut energies = Vec::new();
                for sign in [1., -1.] {
                    let mut moved = system.clone();
                    let r = &mut moved.atoms[coordinate / 3].position;
                    match coordinate % 3 {
                        0 => r.x += sign * step,
                        1 => r.y += sign * step,
                        _ => r.z += sign * step,
                    }
                    energies.push(
                        super::super::native::solve_stda_xtb_native_response(&moved, &options)
                            .unwrap()
                            .ci_eigenvalues_hartree[0],
                    );
                }
                differences.push((energies[0] - energies[1]) / (2. * step));
            }
            let fd = (4. * differences[1] - differences[0]) / 3.;
            let actual = response.excitation_gradient[coordinate / 3].to_array()[coordinate % 3];
            maximum = maximum.max((actual - fd).abs());
        }
        eprintln!("signed sTDA branch at zero crossing: all18 energy-gradient FD max={maximum:e}");
        assert!(maximum < 2e-8, "signed CI gradient error {maximum:e}");
    }
    #[test]
    fn native_stda_total_gradient_matches_composite_pes() {
        let params = crate::Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO .03 -.02 .01\nH .757 .586 .02\nH -.757 .586 -.01\n",
            0.,
            false,
        )
        .unwrap();
        let electronic = crate::ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            max_scc: 1000,
            ..Default::default()
        };
        let options = NativeStdaOptions::default();
        let result =
            solve_stda_xtb_total_gradient(&system, &params, &electronic, 0, &options).unwrap();
        let mut maximum = 0.0_f64;
        for direction in 0..9 {
            let energy = |step: f64| {
                let mut s = system.clone();
                let r = &mut s.atoms[direction / 3].position;
                match direction % 3 {
                    0 => r.x += step,
                    1 => r.y += step,
                    _ => r.z += step,
                }
                crate::run_electronic(&s, &params, electronic.clone())
                    .unwrap()
                    .total_free
                    + solve_stda_xtb_native(&s, &options).unwrap().states[0].excitation_hartree
            };
            let h = 1e-3;
            let coarse = (energy(h) - energy(-h)) / (2. * h);
            let fine = (energy(h / 2.) - energy(-h / 2.)) / h;
            eprintln!(
                "total PES coordinate={direction}: analytic={} coarse={coarse} fine={fine}",
                result.gradient[direction / 3].to_array()[direction % 3]
            );
            maximum = maximum.max(
                (result.gradient[direction / 3].to_array()[direction % 3]
                    - (4. * fine - coarse) / 3.)
                    .abs(),
            );
        }
        eprintln!("GFN1+sTDA total PES all 9 components max error={maximum:e}");
        assert!(maximum < 1e-8);
    }
    #[test]
    fn native_stda_analytic_gradient_all_components_match_fd() {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/stda-ground-reference.json"))
                .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let atoms = case["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["positions_bohr"].as_array().unwrap())
                .map(|(z, r)| {
                    let x = r[0].as_f64().unwrap();
                    let y = r[1].as_f64().unwrap();
                    let zc = r[2].as_f64().unwrap();
                    // Rotate the original planar geometries: every Cartesian axis
                    // then has nonzero responses, with the same physical spectrum.
                    crate::system::Atom {
                        z: z.as_u64().unwrap() as u8,
                        position: Vec3::new(
                            0.8 * x - 0.6 * zc,
                            0.8 * y - 0.6 * (0.6 * x + 0.8 * zc),
                            0.6 * y + 0.8 * (0.6 * x + 0.8 * zc),
                        ),
                    }
                })
                .collect();
            let system = PeriodicSystem::new(atoms, None);
            for triplet in [false, true] {
                let options = NativeStdaOptions {
                    energy_window_ev: 10.,
                    n_states: 6,
                    triplet,
                    ..Default::default()
                };
                let state = if case["name"] == "ethylene" { 2 } else { 0 };
                let result = solve_stda_xtb_gradient(&system, state, &options).unwrap();
                let mut maximum = 0.0_f64;
                let mut coarse_max = 0.0_f64;
                for direction in 0..3 * system.atoms.len() {
                    let energy = |step: f64| {
                        let mut displaced = system.clone();
                        let r = &mut displaced.atoms[direction / 3].position;
                        match direction % 3 {
                            0 => r.x += step,
                            1 => r.y += step,
                            _ => r.z += step,
                        }
                        solve_stda_xtb_native(&displaced, &options).unwrap().states[state]
                            .excitation_hartree
                    };
                    let h = 1e-3;
                    let coarse = (energy(h) - energy(-h)) / (2. * h);
                    let fine = (energy(h / 2.) - energy(-h / 2.)) / h;
                    let analytic =
                        result.excitation_gradient[direction / 3].to_array()[direction % 3];
                    coarse_max = coarse_max.max((analytic - coarse).abs());
                    maximum = maximum.max((analytic - (4. * fine - coarse) / 3.).abs());
                    assert!(
                        (analytic - (4. * fine - coarse) / 3.).abs() < 2e-7,
                        "{} triplet={triplet} coordinate={direction} analytic={analytic:e} FD={:e}",
                        case["name"],
                        (4. * fine - coarse) / 3.
                    );
                }
                let sum = result
                    .excitation_gradient
                    .iter()
                    .fold(Vec3::zero(), |a, b| a + *b);
                let torque = system
                    .atoms
                    .iter()
                    .zip(&result.excitation_gradient)
                    .fold(Vec3::zero(), |a, (atom, g)| a + atom.position.cross(*g));
                eprintln!("{} sTDA gradient triplet={triplet} all {} components: h error={coarse_max:e}, Richardson error={maximum:e}, translation={:e}, torque={:e}",case["name"],3*system.atoms.len(),sum.norm(),torque.norm());
                assert!(sum.norm() < 1e-8 && torque.norm() < 1e-8);
            }
        }
    }
}
