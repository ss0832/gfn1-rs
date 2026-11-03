// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic structural relaxation, finite-strain equations of state and mechanical
//! tensor algebra. Production forces and stresses are analytic, including ML.
//! The explicitly named seminumerical elastic audit differentiates stresses;
//! it is not an analytic strain-response implementation.
use super::{pbc_stress, pbc_stress_from_scc, PbcOptions};
use crate::lattice::Lattice;
use crate::linalg::{symmetric_eigen, DenseLu, Matrix};
use crate::math::{Mat3, Vec3};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};

/// CODATA conversion of Hartree/bohr³ to GPa.
pub const STRESS_AU_TO_GPA: f64 = 29_421.015_696_522;
pub const VOIGT_ORDER: [&str; 6] = ["xx", "yy", "zz", "yz", "xz", "xy"];
const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn lattice_3d(system: &PeriodicSystem) -> Result<Lattice> {
    let lattice = system
        .lattice
        .ok_or_else(|| invalid("mechanical properties require a periodic cell"))?;
    if system.atoms.is_empty() || lattice.periodic != [true; 3] {
        return Err(invalid(
            "bulk mechanical properties require a nonempty three-dimensional periodic cell",
        ));
    }
    Ok(lattice)
}
fn tight(options: &ElectronicOptions) -> ElectronicOptions {
    let mut out = options.clone();
    out.energy_tolerance = out.energy_tolerance.min(1e-12);
    out.charge_tolerance = out.charge_tolerance.min(1e-10);
    out
}
fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn inverse3(m: &Matrix) -> Result<Matrix> {
    DenseLu::factor(m)?.solve_matrix(&Matrix::identity(3))
}
fn mat_vec(m: &Matrix, v: Vec3) -> Vec3 {
    let x = v.to_array();
    Vec3::new(
        (0..3).map(|j| m[(0, j)] * x[j]).sum(),
        (0..3).map(|j| m[(1, j)] * x[j]).sum(),
        (0..3).map(|j| m[(2, j)] * x[j]).sum(),
    )
}
/// Symmetric deformation `F=I+epsilon`, with engineering shear components.
pub fn strain_deformation(strain: [f64; 6]) -> Result<Matrix> {
    if strain.iter().any(|x| !x.is_finite()) {
        return Err(invalid("strain components must be finite"));
    }
    let mut f = Matrix::identity(3);
    for (s, &(i, j)) in PAIRS.iter().enumerate() {
        let v = strain[s] * if i == j { 1. } else { 0.5 };
        f[(i, j)] += v;
        if i != j {
            f[(j, i)] += v;
        }
    }
    let eigen = symmetric_eigen(&f)?;
    if eigen.values[0] <= 0. {
        return Err(invalid(
            "homogeneous deformation must have positive principal stretches",
        ));
    }
    Ok(f)
}
/// Apply a homogeneous strain at fixed fractional atomic coordinates.
pub fn homogeneous_deformation(
    system: &PeriodicSystem,
    strain: [f64; 6],
) -> Result<PeriodicSystem> {
    deform_matrix(system, &strain_deformation(strain)?)
}
fn deform_matrix(system: &PeriodicSystem, f: &Matrix) -> Result<PeriodicSystem> {
    let old = lattice_3d(system)?;
    let mut out = system.clone();
    out.lattice = Some(Lattice::new(
        Mat3::from_columns(
            mat_vec(f, old.cell.col[0]),
            mat_vec(f, old.cell.col[1]),
            mat_vec(f, old.cell.col[2]),
        ),
        old.periodic,
    )?);
    for atom in &mut out.atoms {
        atom.position = mat_vec(f, atom.position);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct PeriodicRelaxOptions {
    pub relax_cell: bool,
    pub max_iterations: usize,
    /// Maximum physical atomic gradient in Hartree/bohr.
    pub force_tolerance: f64,
    /// Maximum `sigma+P_target*I` component in Hartree/bohr³.
    pub stress_tolerance: f64,
    pub target_pressure: f64,
    pub max_atom_step: f64,
    pub max_strain_step: f64,
    pub history: usize,
}
impl Default for PeriodicRelaxOptions {
    fn default() -> Self {
        Self {
            relax_cell: false,
            max_iterations: 300,
            force_tolerance: 1e-7,
            stress_tolerance: 1e-7,
            target_pressure: 0.,
            max_atom_step: 0.2,
            max_strain_step: 0.025,
            history: 12,
        }
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct PeriodicRelaxStep {
    pub iteration: usize,
    pub energy_hartree: f64,
    pub enthalpy_hartree: f64,
    pub volume_bohr3: f64,
    pub maximum_gradient_hartree_per_bohr: f64,
    pub maximum_stress_residual_hartree_per_bohr3: f64,
    pub scf_iterations: usize,
    pub cumulative_scf_evaluations: usize,
    pub cumulative_scf_iterations: usize,
    pub cumulative_scf_map_evaluations: usize,
    pub cumulative_rejected_trials: usize,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct PeriodicRelaxWork {
    pub scf_evaluations: usize,
    pub scf_iterations: usize,
    pub scf_map_evaluations: usize,
    /// False if an SCC error prevented recovering its map diagnostics.
    pub scf_map_evaluations_complete: bool,
    pub failed_scf_evaluations: usize,
    pub rejected_trials: usize,
    pub geometry_rejected_trials: usize,
    pub scf_seconds: f64,
    pub gradient_seconds: f64,
    pub stress_seconds: f64,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Debug)]
pub struct PeriodicRelaxResult {
    pub system: PeriodicSystem,
    pub energy_hartree: f64,
    pub enthalpy_hartree: f64,
    pub gradient: Vec<Vec3>,
    pub stress: Matrix,
    pub converged: bool,
    pub iterations: usize,
    pub termination: &'static str,
    pub last_rejected_trial: Option<String>,
    pub history: Vec<PeriodicRelaxStep>,
    pub electronic_temperature_kelvin: f64,
    pub scc_energy_tolerance: f64,
    pub scc_charge_tolerance: f64,
    pub work: PeriodicRelaxWork,
    /// Small converged shell-charge state for subsequent calculations on the
    /// final or a nearby geometry. Does not retain dense Bloch matrices.
    pub scf_guess: super::PbcSccGuess,
}
impl PeriodicRelaxResult {
    pub fn as_json(&self) -> serde_json::Value {
        let lattice = self.system.lattice.unwrap();
        serde_json::json!({"energy_hartree":self.energy_hartree,"enthalpy_hartree":self.enthalpy_hartree,
            "numbers":self.system.atoms.iter().map(|a|a.z).collect::<Vec<_>>(),
            "positions_bohr":self.system.atoms.iter().map(|a|a.position.to_array()).collect::<Vec<_>>(),
            "cell_bohr":lattice.cell.col.map(|v|v.to_array()),"gradient_hartree_per_bohr":self.gradient.iter().map(|v|v.to_array()).collect::<Vec<_>>(),
            "stress_hartree_per_bohr3":rows(&self.stress),"pressure_gpa":-(0..3).map(|i|self.stress[(i,i)]).sum::<f64>()/3.*STRESS_AU_TO_GPA,
            "converged":self.converged,"iterations":self.iterations,"termination":self.termination,"last_rejected_trial":self.last_rejected_trial,
            "history":self.history,"electronic_temperature_kelvin":self.electronic_temperature_kelvin,
            "scc_energy_tolerance":self.scc_energy_tolerance,"scc_charge_tolerance":self.scc_charge_tolerance,"work":self.work,
            "shell_charge_guess":self.scf_guess.shell_charges})
    }
}
struct Evaluation {
    system: PeriodicSystem,
    energy: f64,
    objective: f64,
    gradient: Vec<f64>,
    atomic: Vec<Vec3>,
    stress: Matrix,
    max_force: f64,
    max_stress: f64,
    guess: super::PbcSccGuess,
    scf_iterations: usize,
}
fn relaxation_geometry(
    reference: &PeriodicSystem,
    x: &[f64],
    cell: bool,
    scale: f64,
) -> Result<(PeriodicSystem, Matrix)> {
    let n = reference.atoms.len();
    let mut system = reference.clone();
    for (i, a) in system.atoms.iter_mut().enumerate() {
        a.position = Vec3::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    }
    let f = if cell {
        let strain = std::array::from_fn(|i| x[3 * n + i] / scale);
        let f = strain_deformation(strain)?;
        if symmetric_eigen(&f)?.values[0] < 0.25 {
            return Err(invalid("relaxation principal stretch below 0.25"));
        }
        f
    } else {
        Matrix::identity(3)
    };
    Ok((deform_matrix(&system, &f)?, f))
}
fn relaxation_evaluate(
    reference: &PeriodicSystem,
    x: &[f64],
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    pbc: &PbcOptions,
    opts: &PeriodicRelaxOptions,
    scale: f64,
    guess: Option<&super::PbcSccGuess>,
    work: &mut PeriodicRelaxWork,
) -> Result<Evaluation> {
    let (system, f) = relaxation_geometry(reference, x, opts.relax_cell, scale)?;
    let begin = std::time::Instant::now();
    work.scf_evaluations += 1;
    let scf = super::run_pbc_scc_with_guess(&system, params, electronic, pbc, guess);
    work.scf_seconds += begin.elapsed().as_secs_f64();
    let scf = match scf {
        Ok(value) => value,
        Err(error) => {
            work.failed_scf_evaluations += 1;
            work.scf_map_evaluations_complete = false;
            if let Gfn1Error::SccNotConverged { iterations, .. } = &error {
                work.scf_iterations += iterations;
            }
            return Err(error);
        }
    };
    let iterations = scf.iterations;
    work.scf_iterations += iterations;
    work.scf_map_evaluations += scf.scc_diagnostics.map_evaluations;
    let next_guess = super::PbcSccGuess::from(&scf);
    let begin = std::time::Instant::now();
    let g = super::pbc_gradient_from_scc(&system, params, scf, electronic, pbc)?;
    work.gradient_seconds += begin.elapsed().as_secs_f64();
    if !g.scf.converged {
        return Err(invalid("periodic relaxation SCC did not converge"));
    }
    let energy = g.total_energy;
    let atomic = g.gradient;
    let lattice = system.lattice.unwrap();
    let begin = std::time::Instant::now();
    let stress = pbc_stress_from_scc(&system, params, g.scf, electronic, pbc, &lattice)?.stress;
    work.stress_seconds += begin.elapsed().as_secs_f64();
    let volume = lattice.volume();
    let mut total_stress = stress.clone();
    for i in 0..3 {
        total_stress[(i, i)] += opts.target_pressure;
    }
    let max_force = atomic.iter().map(|g| g.norm()).fold(0., f64::max);
    let max_stress = total_stress
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0., f64::max);
    let mut gradient = Vec::with_capacity(x.len());
    let ft = f.transpose();
    for &g in &atomic {
        gradient.extend(mat_vec(&ft, g).to_array());
    }
    if opts.relax_cell {
        let gf = total_stress.matmul(&inverse3(&f)?.transpose())?;
        for &(i, j) in &PAIRS {
            gradient.push(
                volume
                    * if i == j {
                        gf[(i, j)]
                    } else {
                        0.5 * (gf[(i, j)] + gf[(j, i)])
                    }
                    / scale,
            );
        }
    }
    if gradient.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite periodic relaxation derivative"));
    }
    Ok(Evaluation {
        system,
        energy,
        objective: energy + opts.target_pressure * volume,
        gradient,
        atomic,
        stress,
        max_force,
        max_stress,
        guess: next_guess,
        scf_iterations: iterations,
    })
}
fn lbfgs(g: &[f64], s: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<f64> {
    let mut q = g.to_vec();
    let mut alpha = vec![0.; s.len()];
    for i in (0..s.len()).rev() {
        alpha[i] = dot(&s[i], &q) / dot(&s[i], &y[i]);
        for (q, y) in q.iter_mut().zip(&y[i]) {
            *q -= alpha[i] * y;
        }
    }
    let gamma = if let Some(last) = s.len().checked_sub(1) {
        (dot(&s[last], &y[last]) / dot(&y[last], &y[last])).clamp(1e-4, 100.)
    } else {
        0.5
    };
    for q in &mut q {
        *q *= gamma;
    }
    for i in 0..s.len() {
        let beta = dot(&y[i], &q) / dot(&s[i], &y[i]);
        for (q, s) in q.iter_mut().zip(&s[i]) {
            *q += s * (alpha[i] - beta);
        }
    }
    q.into_iter().map(|v| -v).collect()
}
/// Analytic-force/stress L-BFGS relaxation with the caller's exact k mesh and
/// cutoffs. Cell variables are symmetric reference deformations, and ionic
/// variables are reference Cartesian positions, so their analytic chain rule
/// includes `F^T g` and `V sigma F^-T`. Tiny steps never certify convergence.
pub fn pbc_relax(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    opts: &PeriodicRelaxOptions,
) -> Result<PeriodicRelaxResult> {
    pbc_relax_with_guess(system, params, options, pbc, opts, None)
}
/// The same force/stress optimization, initialized from a previous converged
/// charge state. Accepted geometries supply the guess for subsequent trials.
pub fn pbc_relax_with_guess(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    opts: &PeriodicRelaxOptions,
    guess: Option<&super::PbcSccGuess>,
) -> Result<PeriodicRelaxResult> {
    let start = std::time::Instant::now();
    let mut work = PeriodicRelaxWork {
        scf_map_evaluations_complete: true,
        ..Default::default()
    };
    super::electromechanical::require_scalar(system, options)?;
    let lattice = lattice_3d(system)?;
    if opts.max_iterations == 0
        || opts.history == 0
        || [
            opts.force_tolerance,
            opts.stress_tolerance,
            opts.max_atom_step,
            opts.max_strain_step,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
        || !opts.target_pressure.is_finite()
    {
        return Err(invalid(
            "invalid periodic relaxation iteration, tolerance, step or pressure",
        ));
    }
    let n = system.atoms.len();
    let d = 3 * n + if opts.relax_cell { 6 } else { 0 };
    crate::resources::check_dense_workspace(d, opts.history, 4, "periodic L-BFGS history")?;
    let electronic = tight(options);
    // Keep the coordinate/strain metric invariant on repeating a primitive
    // cell: both ionic and cell coordinates then scale as sqrt(atom count).
    let scale = (lattice.volume() / n as f64).cbrt() * (n as f64).sqrt();
    let mut x: Vec<f64> = system
        .atoms
        .iter()
        .flat_map(|a| a.position.to_array())
        .collect();
    x.resize(d, 0.);
    let mut eval = relaxation_evaluate(
        system,
        &x,
        params,
        &electronic,
        pbc,
        opts,
        scale,
        guess,
        &mut work,
    )?;
    let mut history = Vec::new();
    let mut ss = Vec::<Vec<f64>>::new();
    let mut yy = Vec::<Vec<f64>>::new();
    let mut last_error = None;
    let mut termination = "iteration_limit";
    let mut converged = false;
    for iteration in 0..=opts.max_iterations {
        history.push(PeriodicRelaxStep {
            iteration,
            energy_hartree: eval.energy,
            enthalpy_hartree: eval.objective,
            volume_bohr3: eval.system.lattice.unwrap().volume(),
            maximum_gradient_hartree_per_bohr: eval.max_force,
            maximum_stress_residual_hartree_per_bohr3: eval.max_stress,
            scf_iterations: eval.scf_iterations,
            cumulative_scf_evaluations: work.scf_evaluations,
            cumulative_scf_iterations: work.scf_iterations,
            cumulative_scf_map_evaluations: work.scf_map_evaluations,
            cumulative_rejected_trials: work.rejected_trials,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        });
        if let Some(logger) = &electronic.logger {
            logger.event(
                crate::LogLevel::Info,
                "periodic_relaxation_step",
                serde_json::json!(history.last().unwrap()),
            )?;
        }
        if eval.max_force <= opts.force_tolerance
            && (!opts.relax_cell || eval.max_stress <= opts.stress_tolerance)
        {
            termination = "force_and_stress_converged";
            converged = true;
            break;
        }
        if iteration == opts.max_iterations {
            break;
        }
        let mut direction = lbfgs(&eval.gradient, &ss, &yy);
        if dot(&direction, &eval.gradient) >= 0. {
            direction = eval.gradient.iter().map(|g| -g).collect();
            ss.clear();
            yy.clear();
        }
        let max_atom = direction[..3 * n]
            .chunks_exact(3)
            .map(|g| (g.iter().map(|x| x * x).sum::<f64>()).sqrt())
            .fold(0., f64::max);
        let mut factor = (opts.max_atom_step / max_atom.max(1e-30)).min(1.);
        if opts.relax_cell {
            let strain = direction[3 * n..]
                .iter()
                .map(|x| x.abs() / scale)
                .fold(0., f64::max);
            factor = factor.min(opts.max_strain_step / strain.max(1e-30));
        }
        for v in &mut direction {
            *v *= factor;
        }
        let slope = dot(&eval.gradient, &direction);
        let mut next = None;
        for backtrack in 0..32 {
            let alpha = 2.0_f64.powi(-backtrack);
            let trial: Vec<f64> = x
                .iter()
                .zip(&direction)
                .map(|(a, b)| a + alpha * b)
                .collect();
            // Bound non-affine motion in physical coordinates. The separate
            // strain bound controls affine advection; counting F*R here would
            // make convergence depend on which lattice image contains an atom.
            if let Ok((_, f)) = relaxation_geometry(system, &trial, opts.relax_cell, scale) {
                let actual = (0..n)
                    .map(|i| {
                        mat_vec(
                            &f,
                            Vec3::new(
                                trial[3 * i] - x[3 * i],
                                trial[3 * i + 1] - x[3 * i + 1],
                                trial[3 * i + 2] - x[3 * i + 2],
                            ),
                        )
                        .norm()
                    })
                    .fold(0., f64::max);
                if actual > opts.max_atom_step * (1. + 1e-12) {
                    work.geometry_rejected_trials += 1;
                    continue;
                }
            }
            if trial.iter().zip(&x).all(|(a, b)| a == b) {
                continue;
            }
            match relaxation_evaluate(
                system,
                &trial,
                params,
                &electronic,
                pbc,
                opts,
                scale,
                Some(&eval.guess),
                &mut work,
            ) {
                Ok(value) if value.objective <= eval.objective + 1e-4 * alpha * slope => {
                    next = Some((trial, value));
                    break;
                }
                Ok(_) => {}
                Err(error) => last_error = Some(error.to_string()),
            }
            work.rejected_trials += 1;
        }
        let Some((next_x, next_eval)) = next else {
            termination = "line_search_stalled";
            break;
        };
        let s: Vec<_> = next_x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<_> = next_eval
            .gradient
            .iter()
            .zip(&eval.gradient)
            .map(|(a, b)| a - b)
            .collect();
        let ys = dot(&s, &y);
        if ys > 1e-12 * (dot(&s, &s) * dot(&y, &y)).sqrt() && ys > 1e-30 {
            if ss.len() == opts.history {
                ss.remove(0);
                yy.remove(0);
            }
            ss.push(s);
            yy.push(y);
        }
        x = next_x;
        eval = next_eval;
    }
    work.elapsed_seconds = start.elapsed().as_secs_f64();
    Ok(PeriodicRelaxResult {
        system: eval.system,
        energy_hartree: eval.energy,
        enthalpy_hartree: eval.objective,
        gradient: eval.atomic,
        stress: eval.stress,
        converged,
        iterations: history.last().unwrap().iteration,
        termination,
        last_rejected_trial: last_error,
        history,
        electronic_temperature_kelvin: electronic.electronic_temperature,
        scc_energy_tolerance: electronic.energy_tolerance,
        scc_charge_tolerance: electronic.charge_tolerance,
        work,
        scf_guess: eval.guess,
    })
}
fn relaxed(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    opts: &PeriodicRelaxOptions,
) -> Result<PeriodicSystem> {
    let mut fixed = opts.clone();
    fixed.relax_cell = false;
    let result = pbc_relax(system, params, options, pbc, &fixed)?;
    if !result.converged {
        return Err(invalid(format!(
            "internal relaxation failed after {} iterations: {}",
            result.iterations, result.termination
        )));
    }
    Ok(result.system)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct EosFit {
    pub equilibrium_volume_bohr3: f64,
    pub minimum_energy_hartree: f64,
    pub bulk_modulus_hartree_per_bohr3: f64,
    pub bulk_modulus_gpa: f64,
    pub bulk_modulus_pressure_derivative: f64,
    pub rms_energy_error_hartree: f64,
    pub maximum_energy_error_hartree: f64,
    pub reference_volume_bohr3: f64,
    /// E=a0+a1*t+a2*t²+a3*t³, t=(f-f_center)/f_scale,
    /// f=[(Vref/V)^(2/3)-1]/2.
    pub finite_strain_coefficients: [f64; 4],
    pub finite_strain_center: f64,
    pub finite_strain_scale: f64,
    pub sampled_volume_range_bohr3: [f64; 2],
}
/// Third-order Birch-Murnaghan fit, solved as a scaled cubic in Eulerian finite
/// strain with twice-reorthogonalized QR. A minimum outside the sampled volume
/// range is rejected instead of presented as an established equilibrium.
pub fn fit_birch_murnaghan(volumes: &[f64], energies: &[f64]) -> Result<EosFit> {
    if volumes.len() != energies.len()
        || volumes.len() < 4
        || volumes.iter().any(|v| !v.is_finite() || *v <= 0.)
        || energies.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "EOS requires at least four matching finite energies and positive volumes",
        ));
    }
    let mut sorted = volumes.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted
        .windows(2)
        .any(|w| (w[1] - w[0]).abs() < 1e-12 * w[1])
    {
        return Err(invalid("EOS volumes must be distinct"));
    }
    let vref = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let f: Vec<_> = volumes
        .iter()
        .map(|v| 0.5 * ((vref / v).powf(2. / 3.) - 1.))
        .collect();
    let center = f.iter().sum::<f64>() / f.len() as f64;
    let scale = f.iter().map(|v| (v - center).abs()).fold(0., f64::max);
    if scale < 1e-8 {
        return Err(invalid(
            "EOS volume span is too small for a stable cubic fit",
        ));
    }
    let t: Vec<_> = f.iter().map(|v| (v - center) / scale).collect();
    let reference = energies.iter().sum::<f64>() / energies.len() as f64;
    let target: Vec<_> = energies.iter().map(|e| e - reference).collect();
    let mut q = Vec::<Vec<f64>>::new();
    let mut r = Matrix::zeros(4, 4);
    let mut qt = [0.; 4];
    for j in 0..4 {
        let mut col: Vec<_> = t.iter().map(|v| v.powi(j as i32)).collect();
        for _ in 0..2 {
            for i in 0..j {
                let projection = dot(&q[i], &col);
                r[(i, j)] += projection;
                for (a, b) in col.iter_mut().zip(&q[i]) {
                    *a -= projection * b;
                }
            }
        }
        let norm = dot(&col, &col).sqrt();
        if norm < 1e-10 {
            return Err(invalid("EOS sampling matrix is rank deficient"));
        }
        r[(j, j)] = norm;
        for v in &mut col {
            *v /= norm;
        }
        qt[j] = dot(&col, &target);
        q.push(col);
    }
    let mut a = [0.; 4];
    for i in (0..4).rev() {
        a[i] = (qt[i] - (i + 1..4).map(|j| r[(i, j)] * a[j]).sum::<f64>()) / r[(i, i)];
    }
    a[0] += reference;
    let eval = |x: f64| a[0] + x * (a[1] + x * (a[2] + x * a[3]));
    let mut roots = Vec::new();
    let qa = 3. * a[3];
    let qb = 2. * a[2];
    let qc = a[1];
    if qa.abs() < 1e-14 * (qb.abs() + qc.abs()).max(1e-30) {
        if qb.abs() > 1e-30 {
            roots.push(-qc / qb);
        }
    } else {
        let disc = qb * qb - 4. * qa * qc;
        if disc >= 0. {
            let p = -0.5 * (qb + disc.sqrt().copysign(qb));
            if p.abs() > 1e-30 {
                roots.extend([p / qa, qc / p]);
            } else {
                roots.push(-qb / (2. * qa));
            }
        }
    }
    let mut minimum = None;
    for t0 in roots {
        let f0 = center + scale * t0;
        let x = 1. + 2. * f0;
        if x <= 0. {
            continue;
        }
        let v0 = vref / x.powf(1.5);
        let d2 = (2. * a[2] + 6. * a[3] * t0) / (scale * scale);
        if v0 < sorted[0] || v0 > *sorted.last().unwrap() || d2 <= 0. {
            continue;
        }
        let energy = eval(t0);
        if minimum
            .as_ref()
            .is_none_or(|(_, e, _, _): &(f64, f64, f64, f64)| energy < *e)
        {
            minimum = Some((v0, energy, d2, x));
        }
    }
    let Some((v0, e0, d2, x)) = minimum else {
        return Err(invalid(
            "EOS has no stable minimum inside the sampled volume range",
        ));
    };
    let bulk = x * x * d2 / (9. * v0);
    let bp = 4. + x * (6. * a[3] / scale.powi(3)) / (3. * d2);
    let errors: Vec<_> = t.iter().zip(energies).map(|(&t, &e)| eval(t) - e).collect();
    Ok(EosFit {
        equilibrium_volume_bohr3: v0,
        minimum_energy_hartree: e0,
        bulk_modulus_hartree_per_bohr3: bulk,
        bulk_modulus_gpa: bulk * STRESS_AU_TO_GPA,
        bulk_modulus_pressure_derivative: bp,
        rms_energy_error_hartree: (dot(&errors, &errors) / errors.len() as f64).sqrt(),
        maximum_energy_error_hartree: errors.iter().map(|v| v.abs()).fold(0., f64::max),
        reference_volume_bohr3: vref,
        finite_strain_coefficients: a,
        finite_strain_center: center,
        finite_strain_scale: scale,
        sampled_volume_range_bohr3: [sorted[0], *sorted.last().unwrap()],
    })
}
#[derive(Clone, Debug)]
pub struct EquationOfStateResult {
    pub systems: Vec<PeriodicSystem>,
    pub volumes_bohr3: Vec<f64>,
    pub energies_hartree: Vec<f64>,
    pub pressures_gpa: Vec<f64>,
    pub relaxed_ions: bool,
    pub fit: EosFit,
    pub electronic_temperature_kelvin: f64,
}
impl EquationOfStateResult {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({"volumes_bohr3":self.volumes_bohr3,"energies_hartree":self.energies_hartree,"pressures_gpa":self.pressures_gpa,"relaxed_ions":self.relaxed_ions,"fit":self.fit,"electronic_temperature_kelvin":self.electronic_temperature_kelvin,"positions_bohr":self.systems.iter().map(|s|s.atoms.iter().map(|a|a.position.to_array()).collect::<Vec<_>>()).collect::<Vec<_>>()})
    }
}
pub fn pbc_equation_of_state(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    volume_ratios: &[f64],
    relax_ions: bool,
    relax: &PeriodicRelaxOptions,
) -> Result<EquationOfStateResult> {
    pbc_equation_of_state_with_guess(
        system,
        params,
        options,
        pbc,
        volume_ratios,
        relax_ions,
        relax,
        None,
    )
}
/// EOS scan with charge continuation between volumes and, optionally, a seed
/// from the supplied reference structure's converged electronic state.
pub fn pbc_equation_of_state_with_guess(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    volume_ratios: &[f64],
    relax_ions: bool,
    relax: &PeriodicRelaxOptions,
    guess: Option<&super::PbcSccGuess>,
) -> Result<EquationOfStateResult> {
    super::electromechanical::require_scalar(system, options)?;
    lattice_3d(system)?;
    if volume_ratios.len() < 4 || volume_ratios.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(invalid("EOS needs at least four positive volume ratios"));
    }
    let electronic = tight(options);
    let mut systems = Vec::new();
    let mut volumes = Vec::new();
    let mut energies = Vec::new();
    let mut pressures = Vec::new();
    let mut continuation = guess.cloned();
    for &ratio in volume_ratios {
        let a = ratio.cbrt() - 1.;
        let mut trial = homogeneous_deformation(system, [a, a, a, 0., 0., 0.])?;
        let (energy, stress) = if relax_ions {
            let mut fixed = relax.clone();
            fixed.relax_cell = false;
            let relaxed = pbc_relax_with_guess(
                &trial,
                params,
                &electronic,
                pbc,
                &fixed,
                continuation.as_ref(),
            )?;
            if !relaxed.converged {
                return Err(invalid(format!(
                    "EOS internal relaxation failed: {}",
                    relaxed.termination
                )));
            }
            continuation = Some(relaxed.scf_guess);
            trial = relaxed.system;
            (relaxed.energy_hartree, relaxed.stress)
        } else {
            let scf = super::run_pbc_scc_with_guess(
                &trial,
                params,
                &electronic,
                pbc,
                continuation.as_ref(),
            )?;
            if !scf.converged {
                return Err(invalid("EOS SCC did not converge"));
            }
            continuation = Some(super::PbcSccGuess::from(&scf));
            let s = pbc_stress_from_scc(
                &trial,
                params,
                scf,
                &electronic,
                pbc,
                &trial.lattice.unwrap(),
            )?;
            (s.total_energy, s.stress)
        };
        volumes.push(trial.lattice.unwrap().volume());
        energies.push(energy);
        pressures.push(-(0..3).map(|i| stress[(i, i)]).sum::<f64>() / 3. * STRESS_AU_TO_GPA);
        systems.push(trial);
    }
    let fit = fit_birch_murnaghan(&volumes, &energies)?;
    Ok(EquationOfStateResult {
        systems,
        volumes_bohr3: volumes,
        energies_hartree: energies,
        pressures_gpa: pressures,
        relaxed_ions: relax_ions,
        fit,
        electronic_temperature_kelvin: electronic.electronic_temperature,
    })
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ElasticProperties {
    pub compliance_bohr3_per_hartree: Vec<Vec<f64>>,
    pub stiffness_eigenvalues_hartree_per_bohr3: Vec<f64>,
    pub bulk_voigt_gpa: f64,
    pub bulk_reuss_gpa: f64,
    pub bulk_hill_gpa: f64,
    pub shear_voigt_gpa: f64,
    pub shear_reuss_gpa: f64,
    pub shear_hill_gpa: f64,
    pub young_hill_gpa: f64,
    pub poisson_hill: f64,
    pub universal_anisotropy: f64,
    pub transverse_hill_m_per_s: f64,
    pub longitudinal_hill_m_per_s: f64,
    pub mean_sound_m_per_s: f64,
    pub density_kg_m3: f64,
}
fn validate_stiffness(stiffness: &Matrix) -> Result<Matrix> {
    if stiffness.rows() != 6
        || stiffness.cols() != 6
        || stiffness.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "stiffness must be a finite 6x6 engineering-Voigt matrix",
        ));
    }
    let scale = stiffness
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(0., f64::max)
        .max(1e-20);
    if stiffness.max_abs_diff(&stiffness.transpose()) > 1e-7 * scale {
        return Err(invalid(
            "stiffness lacks major symmetry; inspect raw strain/SCF convergence",
        ));
    }
    let eig = symmetric_eigen(stiffness)?;
    if eig.values[0] <= 1e-12 * scale {
        return Err(invalid(
            "elastic stiffness is unstable or singular; positive moduli/compliance are undefined",
        ));
    }
    DenseLu::factor(stiffness)?.solve_matrix(&Matrix::identity(6))
}
/// Zero-stress conventional Voigt-Reuss-Hill polycrystal averages. Supply a
/// stress-free material stiffness, not an uncorrected finite-pressure tangent.
pub fn elastic_properties(stiffness: &Matrix, density_kg_m3: f64) -> Result<ElasticProperties> {
    if !density_kg_m3.is_finite() || density_kg_m3 <= 0. {
        return Err(invalid("density must be positive kg/m3"));
    }
    let compliance = validate_stiffness(stiffness)?;
    let (c, s) = (stiffness, &compliance);
    let diagonal = (0..3).map(|i| c[(i, i)]).sum::<f64>();
    let off = c[(0, 1)] + c[(0, 2)] + c[(1, 2)];
    let shear = c[(3, 3)] + c[(4, 4)] + c[(5, 5)];
    let sd = (0..3).map(|i| s[(i, i)]).sum::<f64>();
    let so = s[(0, 1)] + s[(0, 2)] + s[(1, 2)];
    let ss = s[(3, 3)] + s[(4, 4)] + s[(5, 5)];
    let bv = (diagonal + 2. * off) / 9.;
    let br = 1. / (sd + 2. * so);
    let gv = (diagonal - off + 3. * shear) / 15.;
    let gr = 15. / (4. * sd - 4. * so + 3. * ss);
    let b = (bv + br) / 2.;
    let g = (gv + gr) / 2.;
    let vt = (g * STRESS_AU_TO_GPA * 1e9 / density_kg_m3).sqrt();
    let vl = ((b + 4. * g / 3.) * STRESS_AU_TO_GPA * 1e9 / density_kg_m3).sqrt();
    Ok(ElasticProperties {
        compliance_bohr3_per_hartree: rows(s),
        stiffness_eigenvalues_hartree_per_bohr3: symmetric_eigen(c)?.values,
        bulk_voigt_gpa: bv * STRESS_AU_TO_GPA,
        bulk_reuss_gpa: br * STRESS_AU_TO_GPA,
        bulk_hill_gpa: b * STRESS_AU_TO_GPA,
        shear_voigt_gpa: gv * STRESS_AU_TO_GPA,
        shear_reuss_gpa: gr * STRESS_AU_TO_GPA,
        shear_hill_gpa: g * STRESS_AU_TO_GPA,
        young_hill_gpa: 9. * b * g / (3. * b + g) * STRESS_AU_TO_GPA,
        poisson_hill: (3. * b - 2. * g) / (2. * (3. * b + g)),
        universal_anisotropy: 5. * gv / gr + bv / br - 6.,
        transverse_hill_m_per_s: vt,
        longitudinal_hill_m_per_s: vl,
        mean_sound_m_per_s: ((2. / vt.powi(3) + 1. / vl.powi(3)) / 3.).powf(-1. / 3.),
        density_kg_m3,
    })
}
fn unit(direction: [f64; 3]) -> Result<[f64; 3]> {
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-14 {
        return Err(invalid("direction must be a finite nonzero vector"));
    }
    Ok(direction.map(|v| v / norm))
}
fn voigt_pair(i: usize, j: usize) -> usize {
    match (i.min(j), i.max(j)) {
        (0, 0) => 0,
        (1, 1) => 1,
        (2, 2) => 2,
        (1, 2) => 3,
        (0, 2) => 4,
        (0, 1) => 5,
        _ => unreachable!(),
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct DirectionalElasticResponse {
    pub direction: [f64; 3],
    pub transverse_direction: [f64; 3],
    pub young_gpa: f64,
    pub poisson_ratio: f64,
    pub shear_gpa: f64,
    pub christoffel_gpa: [[f64; 3]; 3],
    pub sound_speeds_m_per_s: [f64; 3],
    pub sound_polarizations: [[f64; 3]; 3],
}
pub fn directional_elastic_response(
    stiffness: &Matrix,
    density_kg_m3: f64,
    direction: [f64; 3],
    transverse: [f64; 3],
) -> Result<DirectionalElasticResponse> {
    if !density_kg_m3.is_finite() || density_kg_m3 <= 0. {
        return Err(invalid("density must be positive kg/m3"));
    }
    let n = unit(direction)?;
    let m = unit(transverse)?;
    if n.iter().zip(m).map(|(a, b)| a * b).sum::<f64>().abs() > 1e-8 {
        return Err(invalid(
            "transverse direction must be perpendicular to loading direction",
        ));
    }
    let s = validate_stiffness(stiffness)?;
    let mut axial = 0.;
    let mut lateral = 0.;
    let mut shear = 0.;
    let mut christoffel = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let a = voigt_pair(i, j);
                    let b = voigt_pair(k, l);
                    let sf =
                        s[(a, b)] / if i == j { 1. } else { 2. } / if k == l { 1. } else { 2. };
                    axial += sf * n[i] * n[j] * n[k] * n[l];
                    lateral += sf * m[i] * m[j] * n[k] * n[l];
                    shear += 4. * sf * n[i] * m[j] * n[k] * m[l];
                    christoffel[(i, k)] += stiffness[(a, b)] * n[j] * n[l];
                }
            }
        }
    }
    let eig = symmetric_eigen(&christoffel)?;
    if eig.values[0] <= 0. {
        return Err(invalid("negative directional acoustic stiffness"));
    }
    Ok(DirectionalElasticResponse {
        direction: n,
        transverse_direction: m,
        young_gpa: STRESS_AU_TO_GPA / axial,
        poisson_ratio: -lateral / axial,
        shear_gpa: STRESS_AU_TO_GPA / shear,
        christoffel_gpa: std::array::from_fn(|i| {
            std::array::from_fn(|j| christoffel[(i, j)] * STRESS_AU_TO_GPA)
        }),
        sound_speeds_m_per_s: std::array::from_fn(|i| {
            (eig.values[i] * STRESS_AU_TO_GPA * 1e9 / density_kg_m3).sqrt()
        }),
        sound_polarizations: std::array::from_fn(|mode| {
            std::array::from_fn(|i| eig.vectors[(i, mode)])
        }),
    })
}

#[derive(Clone, Debug)]
pub struct ElasticOptions {
    pub strain_step: f64,
    pub richardson: bool,
    pub relaxed_ions: bool,
    pub relax: PeriodicRelaxOptions,
}
impl Default for ElasticOptions {
    fn default() -> Self {
        Self {
            strain_step: 1e-3,
            richardson: true,
            relaxed_ions: false,
            relax: PeriodicRelaxOptions::default(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct ElasticResult {
    pub reference_system: PeriodicSystem,
    pub stiffness: Matrix,
    pub raw_stiffness: Matrix,
    pub reference_stress: Matrix,
    pub raw_major_asymmetry: f64,
    pub step_error: f64,
    pub relaxed_ions: bool,
    pub strain_step: f64,
    pub electronic_temperature_kelvin: f64,
}
impl ElasticResult {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({"method":"seminumerical stress derivative; material dS/dGreen_strain","voigt_order":VOIGT_ORDER,"stiffness_hartree_per_bohr3":rows(&self.stiffness),"stiffness_gpa":rows(&self.stiffness).into_iter().map(|r|r.into_iter().map(|x|x*STRESS_AU_TO_GPA).collect::<Vec<_>>()).collect::<Vec<_>>(),"raw_stiffness_hartree_per_bohr3":rows(&self.raw_stiffness),"reference_stress_hartree_per_bohr3":rows(&self.reference_stress),"raw_major_asymmetry_hartree_per_bohr3":self.raw_major_asymmetry,"step_error_hartree_per_bohr3":self.step_error,"relaxed_ions":self.relaxed_ions,"strain_step":self.strain_step,"electronic_temperature_kelvin":self.electronic_temperature_kelvin})
    }
}
/// Independent finite-strain audit/oracle, explicitly NOT an analytic elastic
/// route. Piola pull-back removes geometric prestress from the material tensor.
pub fn pbc_elastic_constants_seminumerical(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    opts: &ElasticOptions,
) -> Result<ElasticResult> {
    lattice_3d(system)?;
    if !opts.strain_step.is_finite() || opts.strain_step <= 0. || opts.strain_step > 0.05 {
        return Err(invalid("elastic strain step must lie in (0,0.05]"));
    }
    if options.external_field.is_active() {
        return Err(invalid("elastic tensor requires zero external fields"));
    }
    let electronic = tight(options);
    let reference = if opts.relaxed_ions {
        relaxed(system, params, &electronic, pbc, &opts.relax)?
    } else {
        system.clone()
    };
    let initial = pbc_stress(&reference, params, &electronic, pbc)?;
    if !initial.scf.converged {
        return Err(invalid("elastic reference SCC did not converge"));
    }
    let mut raw = Matrix::zeros(6, 6);
    let mut error = 0.0_f64;
    for column in 0..6 {
        let evaluate = |h: f64| -> Result<[f64; 6]> {
            let mut strain = [0.; 6];
            strain[column] = h;
            let f = strain_deformation(strain)?;
            let mut trial = deform_matrix(&reference, &f)?;
            if opts.relaxed_ions {
                trial = relaxed(&trial, params, &electronic, pbc, &opts.relax)?;
            }
            let s = pbc_stress(&trial, params, &electronic, pbc)?;
            if !s.scf.converged {
                return Err(invalid("elastic displaced SCC did not converge"));
            }
            let inv = inverse3(&f)?;
            let pulled = inv.matmul(&s.stress)?.matmul(&inv.transpose())?;
            let jac = trial.lattice.unwrap().volume() / reference.lattice.unwrap().volume();
            Ok(std::array::from_fn(|row| {
                let (i, j) = PAIRS[row];
                jac * 0.5 * (pulled[(i, j)] + pulled[(j, i)])
            }))
        };
        let plus = evaluate(opts.strain_step)?;
        let minus = evaluate(-opts.strain_step)?;
        let coarse: [f64; 6] =
            std::array::from_fn(|i| (plus[i] - minus[i]) / (2. * opts.strain_step));
        if opts.richardson {
            let p = evaluate(opts.strain_step / 2.)?;
            let m = evaluate(-opts.strain_step / 2.)?;
            for i in 0..6 {
                let fine = (p[i] - m[i]) / opts.strain_step;
                error = error.max((fine - coarse[i]).abs() / 3.);
                raw[(i, column)] = (4. * fine - coarse[i]) / 3.;
            }
        } else {
            for i in 0..6 {
                raw[(i, column)] = coarse[i];
            }
        }
    }
    let asym = raw.max_abs_diff(&raw.transpose());
    let mut sym = raw.clone();
    for i in 0..6 {
        for j in 0..6 {
            sym[(i, j)] = 0.5 * (raw[(i, j)] + raw[(j, i)]);
        }
    }
    Ok(ElasticResult {
        reference_system: reference,
        stiffness: sym,
        raw_stiffness: raw,
        reference_stress: initial.stress,
        raw_major_asymmetry: asym,
        step_error: error,
        relaxed_ions: opts.relaxed_ions,
        strain_step: opts.strain_step,
        electronic_temperature_kelvin: electronic.electronic_temperature,
    })
}
