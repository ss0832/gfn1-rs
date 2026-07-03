// SPDX-License-Identifier: GPL-3.0-or-later
//! Molecular geometry optimization utilities.

use crate::error::{Gfn1Error, Result};
use crate::gradient::{analytic_gradient, AnalyticGradientOptions};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

const DEFAULT_HISTORY: usize = 12;
const DEFAULT_MAX_ITERATIONS: usize = 250;
const DEFAULT_GRADIENT_TOLERANCE: f64 = 1.0e-4;
const DEFAULT_STEP_TOLERANCE: f64 = 1.0e-7;
const DEFAULT_INITIAL_STEP: f64 = 1.0;
const DEFAULT_MAX_ATOM_STEP: f64 = 0.30;

#[derive(Clone, Debug)]
pub struct GeometryOptimizationOptions {
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub history: usize,
    pub initial_step: f64,
    pub max_atom_step: f64,
    pub gradient_options: AnalyticGradientOptions,
    /// Optional path for a **streaming** multi-frame XYZ trajectory: when set, each L-BFGS
    /// step's geometry is appended and flushed as the optimization runs (so it can be watched
    /// live), rather than collected and written only at the end.
    pub trajectory_path: Option<std::path::PathBuf>,
}

impl Default for GeometryOptimizationOptions {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            gradient_tolerance: DEFAULT_GRADIENT_TOLERANCE,
            step_tolerance: DEFAULT_STEP_TOLERANCE,
            history: DEFAULT_HISTORY,
            initial_step: DEFAULT_INITIAL_STEP,
            max_atom_step: DEFAULT_MAX_ATOM_STEP,
            gradient_options: AnalyticGradientOptions::default(),
            trajectory_path: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeometryOptimizationStep {
    pub iteration: usize,
    pub energy: f64,
    pub max_gradient: f64,
    pub step_norm: f64,
    /// Atomic positions (Bohr) at this step, so the full trajectory can be written out.
    pub positions: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct GeometryOptimizationResult {
    pub system: PeriodicSystem,
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub iterations: usize,
    pub converged: bool,
    pub max_gradient: f64,
    pub trajectory: Vec<GeometryOptimizationStep>,
}

#[derive(Clone, Debug)]
struct Evaluation {
    energy: f64,
    gradient: Vec<f64>,
    gradient_vec3: Vec<Vec3>,
    forces: Vec<Vec3>,
    max_gradient: f64,
}

pub fn optimize_geometry(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: GeometryOptimizationOptions,
) -> Result<GeometryOptimizationResult> {
    // Periodic systems optimize the atomic positions at **fixed cell**: `analytic_gradient` already
    // routes to the PBC (Γ / k-point) gradient and `system_with_positions` preserves the lattice, so
    // the L-BFGS machinery is geometry-agnostic. (Variable-cell relaxation is gated separately below.)
    if system.is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "cannot optimize an empty system".to_string(),
        ));
    }

    let mut current = system.clone();
    let mut x = flatten_positions(&current);
    let mut eval = evaluate(&current, params, &options.gradient_options)?;
    let mut trajectory = vec![GeometryOptimizationStep {
        iteration: 0,
        energy: eval.energy,
        max_gradient: eval.max_gradient,
        step_norm: 0.0,
        positions: current.atoms.iter().map(|a| a.position).collect(),
    }];

    // Optional streaming trajectory: write + flush each frame as it is produced.
    let mut traj_writer = match &options.trajectory_path {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => None,
    };
    if let Some(w) = traj_writer.as_mut() {
        stream_xyz_frame(w, &current, &trajectory[0])?;
    }

    let history_len = options.history.max(1);
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut converged = eval.max_gradient <= options.gradient_tolerance;
    let mut iterations = 0usize;

    while !converged && iterations < options.max_iterations {
        iterations += 1;
        let mut direction = lbfgs_direction(&eval.gradient, &s_hist, &y_hist, &rho_hist);
        if dot(&direction, &eval.gradient) >= 0.0 || !all_finite(&direction) {
            direction = eval.gradient.iter().map(|g| -g).collect();
        }
        limit_atom_step(&mut direction, options.max_atom_step);

        let directional_derivative = dot(&eval.gradient, &direction);
        let (next_x, next_eval, step_norm) = line_search(
            &current,
            &x,
            &direction,
            eval.energy,
            directional_derivative,
            params,
            &options,
        )?;

        let s = subtract(&next_x, &x);
        let y = subtract(&next_eval.gradient, &eval.gradient);
        let ys = dot(&y, &s);
        if ys > 1.0e-12 && all_finite(&s) && all_finite(&y) {
            if s_hist.len() == history_len {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / ys);
        }

        x = next_x;
        current = system_with_positions(system, &x);
        eval = next_eval;
        trajectory.push(GeometryOptimizationStep {
            iteration: iterations,
            energy: eval.energy,
            max_gradient: eval.max_gradient,
            step_norm,
            positions: current.atoms.iter().map(|a| a.position).collect(),
        });
        if let Some(w) = traj_writer.as_mut() {
            stream_xyz_frame(w, &current, trajectory.last().unwrap())?;
        }
        converged =
            eval.max_gradient <= options.gradient_tolerance || step_norm <= options.step_tolerance;
    }

    Ok(GeometryOptimizationResult {
        system: current,
        energy: eval.energy,
        gradient: eval.gradient_vec3,
        forces: eval.forces,
        iterations,
        converged,
        max_gradient: eval.max_gradient,
        trajectory,
    })
}

/// Write one XYZ frame (Angstrom) of `step` to `w` and flush, so a streaming trajectory file
/// is updated live. The comment line carries the iteration / energy (Hartree) / max gradient.
fn stream_xyz_frame<W: std::io::Write>(
    w: &mut W,
    system: &PeriodicSystem,
    step: &GeometryOptimizationStep,
) -> std::io::Result<()> {
    let bohr_to_angstrom = 1.0 / crate::system::ANGSTROM_TO_BOHR;
    writeln!(w, "{}", system.atoms.len())?;
    writeln!(
        w,
        "iter {} energy {:.10} Ha max_grad {:.3e}",
        step.iteration, step.energy, step.max_gradient
    )?;
    for (atom, p) in system.atoms.iter().zip(&step.positions) {
        let sym = crate::system::z_to_symbol(atom.z).unwrap_or("X");
        writeln!(
            w,
            "{sym:2} {:18.10} {:18.10} {:18.10}",
            p.x * bohr_to_angstrom,
            p.y * bohr_to_angstrom,
            p.z * bohr_to_angstrom
        )?;
    }
    w.flush()
}

fn evaluate(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &AnalyticGradientOptions,
) -> Result<Evaluation> {
    let result = analytic_gradient(system, params, options.clone())?;
    Ok(Evaluation {
        energy: result.total_energy,
        gradient: flatten_vec3(&result.gradient),
        gradient_vec3: result.gradient,
        forces: result.forces,
        max_gradient: result.max_gradient,
    })
}

fn line_search(
    base_system: &PeriodicSystem,
    x: &[f64],
    direction: &[f64],
    energy: f64,
    directional_derivative: f64,
    params: &Gfn1Parameters,
    options: &GeometryOptimizationOptions,
) -> Result<(Vec<f64>, Evaluation, f64)> {
    let mut alpha = options.initial_step.max(1.0e-8);
    let c1 = 1.0e-4;
    let mut last_error = None;
    for _ in 0..24 {
        let trial = add_scaled(x, direction, alpha);
        let trial_system = system_with_positions(base_system, &trial);
        match evaluate(&trial_system, params, &options.gradient_options) {
            Ok(eval) if eval.energy <= energy + c1 * alpha * directional_derivative => {
                let step_norm = norm(&subtract(&trial, x));
                return Ok((trial, eval, step_norm));
            }
            Ok(_) => {}
            Err(err) => last_error = Some(err),
        }
        alpha *= 0.5;
    }

    if let Some(err) = last_error {
        Err(err)
    } else {
        Err(Gfn1Error::InvalidInput(
            "L-BFGS line search failed to find a downhill step".to_string(),
        ))
    }
}

fn lbfgs_direction(
    g: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
) -> Vec<f64> {
    if s_hist.is_empty() {
        return g.iter().map(|v| -v).collect();
    }
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; s_hist.len()];
    for i in (0..s_hist.len()).rev() {
        alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
        axpy(&mut q, &y_hist[i], -alpha[i]);
    }

    let last = s_hist.len() - 1;
    let yy = dot(&y_hist[last], &y_hist[last]);
    let ys = dot(&y_hist[last], &s_hist[last]);
    let gamma = if yy > 1.0e-18 { ys / yy } else { 1.0 };
    for value in &mut q {
        *value *= gamma;
    }

    for i in 0..s_hist.len() {
        let beta = rho_hist[i] * dot(&y_hist[i], &q);
        axpy(&mut q, &s_hist[i], alpha[i] - beta);
    }
    q.iter().map(|v| -v).collect()
}

fn limit_atom_step(direction: &mut [f64], max_atom_step: f64) {
    if max_atom_step <= 0.0 {
        return;
    }
    let mut max_step = 0.0_f64;
    for chunk in direction.chunks_exact(3) {
        max_step =
            max_step.max((chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt());
    }
    if max_step > max_atom_step {
        let scale = max_atom_step / max_step;
        for value in direction {
            *value *= scale;
        }
    }
}

fn flatten_positions(system: &PeriodicSystem) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * system.atoms.len());
    for atom in &system.atoms {
        out.extend_from_slice(&[atom.position.x, atom.position.y, atom.position.z]);
    }
    out
}

fn flatten_vec3(values: &[Vec3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * values.len());
    for value in values {
        out.extend_from_slice(&[value.x, value.y, value.z]);
    }
    out
}

fn system_with_positions(reference: &PeriodicSystem, x: &[f64]) -> PeriodicSystem {
    let mut system = reference.clone();
    for (atom, xyz) in system.atoms.iter_mut().zip(x.chunks_exact(3)) {
        atom.position = Vec3::new(xyz[0], xyz[1], xyz[2]);
    }
    system
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn add_scaled(a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + scale * y).collect()
}

fn axpy(y: &mut [f64], x: &[f64], alpha: f64) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}
