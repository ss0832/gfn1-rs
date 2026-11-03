// SPDX-License-Identifier: GPL-3.0-or-later
//! Molecular overlap-dual SOC and physical complex-spinor SCC (Jha2023SOC).
//! Input zeta is in H_atom=zeta L.S, in hartree. This explicit independent
//! convention is not an unverified transcription of the paper's epsilon table.
//! No author spreadsheet is bundled. Noncollinear W and composite models are gated.
use crate::error::{Gfn1Error, Result};
use crate::linalg::{lowdin_orthogonalizer, DenseLu, Matrix};
use crate::params::AngularMomentum;
use crate::pbc::complex::CMatrix;
use crate::{BasisSet, ElectronicOptions, ElectronicResult, Gfn1Parameters, PeriodicSystem};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct SocParameters {
    /// (Z,l) -> zeta L.S in hartree. Missing p/d entries are errors, including
    /// when a caller intends zero; the zero must be explicitly supplied.
    pub zeta: BTreeMap<(u8, usize), f64>,
    pub provenance: String,
}
impl SocParameters {
    /// `units hartree`, `provenance ...`, then rows `Z l zeta`.
    pub fn from_zeta_str(text: &str) -> Result<Self> {
        let mut lines = text
            .lines()
            .filter(|s| !s.trim().is_empty() && !s.trim_start().starts_with('#'));
        if lines.next().map(str::trim) != Some("units hartree") {
            return Err(Gfn1Error::InvalidInput(
                "SOC zeta input must start with `units hartree`".into(),
            ));
        }
        let provenance = lines
            .next()
            .and_then(|s| s.strip_prefix("provenance "))
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| Gfn1Error::InvalidInput("SOC zeta provenance is required".into()))?
            .to_string();
        let mut zeta = BTreeMap::new();
        for line in lines {
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() != 3 {
                return Err(Gfn1Error::InvalidInput(
                    "SOC zeta rows must be Z l zeta".into(),
                ));
            }
            let z = p[0].parse::<u8>().ok().filter(|&x| x > 0 && x <= 86);
            let l = p[1].parse::<usize>().ok().filter(|&x| x <= 2);
            let value = p[2].parse::<f64>().ok().filter(|x| x.is_finite());
            let (Some(z), Some(l), Some(value)) = (z, l, value) else {
                return Err(Gfn1Error::InvalidInput("invalid SOC zeta row".into()));
            };
            if zeta.insert((z, l), value).is_some() {
                return Err(Gfn1Error::InvalidInput("duplicate SOC zeta row".into()));
            }
        }
        Ok(Self { zeta, provenance })
    }
}

/// Lx,Ly,Lz with ordering and dz2 phase matching basis.rs. Constructed by
/// applying -i(r x grad) to the actual real-spherical polynomials.
pub fn shell_angular_momentum(l: AngularMomentum) -> Result<[CMatrix; 3]> {
    let (powers, coeff): (Vec<[usize; 3]>, Vec<Vec<f64>>) = match l {
        AngularMomentum::S => (vec![[0, 0, 0]], vec![vec![1.0]]),
        AngularMomentum::P => (
            vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        AngularMomentum::D => {
            let a = 3.0_f64.sqrt();
            (
                vec![
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                ],
                vec![
                    vec![a / 2.0, -a / 2.0, 0.0, 0.0, 0.0, 0.0],
                    vec![0.5, 0.5, -1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, a, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0, a, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0, 0.0, a],
                ],
            )
        }
        _ => return Err(Gfn1Error::InvalidInput("SOC supports s/p/d shells".into())),
    };
    let n = coeff.len();
    let np = powers.len();
    let m = Matrix::from_vec(
        np,
        n,
        (0..np)
            .flat_map(|k| coeff.iter().map(move |c| c[k]))
            .collect(),
    )?;
    let lu = DenseLu::factor(&m.transpose().matmul(&m)?)?;
    let mut out = std::array::from_fn(|_| CMatrix::zeros(n));
    for axis in 0..3 {
        let b = (axis + 1) % 3;
        let c = (axis + 2) % 3;
        for j in 0..n {
            let mut rotated = vec![0.0; np];
            for (p, &power) in powers.iter().enumerate() {
                for (derivative, multiplier, sign) in [(c, b, 1.0), (b, c, -1.0)] {
                    if power[derivative] == 0 {
                        continue;
                    }
                    let mut target = power;
                    target[derivative] -= 1;
                    target[multiplier] += 1;
                    let k = powers.iter().position(|&x| x == target).unwrap();
                    rotated[k] += sign * coeff[j][p] * power[derivative] as f64;
                }
            }
            let rhs: Vec<f64> = coeff
                .iter()
                .map(|c| c.iter().zip(&rotated).map(|(a, b)| a * b).sum())
                .collect();
            let expansion = lu.solve_vec(&rhs)?;
            for i in 0..n {
                out[axis].im[(i, j)] = -expansion[i];
            }
        }
    }
    Ok(out)
}

fn spin_overlap(s: &Matrix) -> Matrix {
    let n = s.rows();
    let mut out = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = s[(i, j)];
            out[(n + i, n + j)] = s[(i, j)];
        }
    }
    out
}
pub fn onsite_soc(basis: &BasisSet, parameters: &SocParameters) -> Result<CMatrix> {
    if parameters.provenance.trim().is_empty() {
        return Err(Gfn1Error::InvalidInput(
            "SOC parameter provenance is required".into(),
        ));
    }
    let n = basis.len();
    let mut out = CMatrix::zeros(2 * n);
    for shell in &basis.shells {
        let l = shell.angular.as_index();
        if l == 0 {
            continue;
        }
        let zeta = *parameters.zeta.get(&(shell.z, l)).ok_or_else(|| {
            Gfn1Error::InvalidInput(format!("missing SOC zeta for Z={}, l={l}", shell.z))
        })?;
        if !zeta.is_finite() {
            return Err(Gfn1Error::InvalidInput("non-finite SOC coupling".into()));
        }
        let [lx, ly, lz] = shell_angular_momentum(shell.angular)?;
        for a in 0..shell.nao {
            for b in 0..shell.nao {
                let (i, j) = (shell.first_ao + a, shell.first_ao + b);
                let scale = 0.5 * zeta;
                out.accumulate(i, j, 0.0, scale * lz.im[(a, b)]);
                out.accumulate(n + i, n + j, 0.0, -scale * lz.im[(a, b)]);
                out.accumulate(i, n + j, scale * ly.im[(a, b)], scale * lx.im[(a, b)]);
                out.accumulate(n + i, j, -scale * ly.im[(a, b)], scale * lx.im[(a, b)]);
            }
        }
    }
    Ok(out)
}
/// {S, H_atom}/2 includes the off-site dual term; an on-site-only addition
/// would miss both this operator and its overlap/Pulay derivative.
pub fn dual_soc(onsite: &CMatrix, overlap: &Matrix) -> Result<CMatrix> {
    let s = spin_overlap(overlap);
    if s.rows() != onsite.n {
        return Err(Gfn1Error::InvalidInput(
            "SOC/overlap dimension mismatch".into(),
        ));
    }
    let mut out = CMatrix::zeros(onsite.n);
    let sr = s.matmul(&onsite.re)?;
    let rs = onsite.re.matmul(&s)?;
    let si = s.matmul(&onsite.im)?;
    let is = onsite.im.matmul(&s)?;
    for i in 0..onsite.n {
        for j in 0..onsite.n {
            out.re[(i, j)] = 0.5 * (sr[(i, j)] + rs[(i, j)]);
            out.im[(i, j)] = 0.5 * (si[(i, j)] + is[(i, j)]);
        }
    }
    Ok(out)
}

/// Native-complex eigensolution of the PHYSICAL matrix; no real-embedding
/// copies. Columns obey C^dagger S C=I, each physical spinor has max occupation 1.
pub struct SpinorEigen {
    pub values: Vec<f64>,
    pub vectors: CMatrix,
}
pub fn spinor_eigen(h: &CMatrix, overlap: &Matrix, tolerance: f64) -> Result<SpinorEigen> {
    if h.n != overlap.rows() || overlap.rows() != overlap.cols() {
        return Err(Gfn1Error::InvalidInput(
            "spinor eigen dimension mismatch".into(),
        ));
    }
    let orth = lowdin_orthogonalizer(overlap, tolerance)?;
    let re = orth.xt.matmul(&h.re.matmul(&orth.x)?)?;
    let im = orth.xt.matmul(&h.im.matmul(&orth.x)?)?;
    let eig = crate::pbc::complex::native_hermitian_eigen(&CMatrix { n: h.n, re, im })?;
    let values = eig.values;
    let mut u = eig.vectors;
    u.re = orth.x.matmul(&u.re)?;
    u.im = orth.x.matmul(&u.im)?;
    Ok(SpinorEigen { values, vectors: u })
}
pub(crate) fn density(eigen: &SpinorEigen, weights: &[f64]) -> CMatrix {
    let n = eigen.values.len();
    let mut p = CMatrix::zeros(n);
    let c = &eigen.vectors;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                p.re[(i, j)] +=
                    weights[k] * (c.re[(i, k)] * c.re[(j, k)] + c.im[(i, k)] * c.im[(j, k)]);
                p.im[(i, j)] +=
                    weights[k] * (c.im[(i, k)] * c.re[(j, k)] - c.re[(i, k)] * c.im[(j, k)]);
            }
        }
    }
    p
}
fn spin_trace(p: &CMatrix) -> Matrix {
    let n = p.n / 2;
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = p.re[(i, j)] + p.re[(n + i, n + j)];
        }
    }
    out
}
pub struct SocResult {
    pub calculation_id: Option<String>,
    pub total_internal: f64,
    pub total_free: f64,
    pub soc_energy: f64,
    pub atomic_charges: Vec<f64>,
    pub eigen: SpinorEigen,
    pub occupations: Vec<f64>,
    pub density: CMatrix,
    pub iterations: usize,
    pub max_charge_residual: f64,
    pub dipole: crate::math::Vec3,
    pub method_citations: Vec<&'static str>,
    pub parameter_provenance: String,
    scalar_proxy: ElectronicResult,
}
pub fn run_soc_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    soc: &SocParameters,
) -> Result<SocResult> {
    let span=options.logger.as_ref().map(|logger|logger.begin("soc_spinor_scf",serde_json::json!({"atoms":system.atoms.len(),"electronic_temperature_K":options.electronic_temperature}))).transpose()?;
    let mut result = run_soc_scc_impl(system, params, options, soc);
    if let Ok(value) = &mut result {
        value.calculation_id = span.as_ref().map(|s| s.id().to_string());
    }
    crate::logging::finish(
        span,
        result,
        |s| serde_json::json!({"iterations":s.iterations,"total_free_hartree":s.total_free,"soc_energy_hartree":s.soc_energy,"charge_residual_max":s.max_charge_residual}),
    )
}
fn run_soc_scc_impl(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    soc: &SocParameters,
) -> Result<SocResult> {
    if system.lattice.is_some()
        || options.spin_polarization
        || options.plus_u
        || options.multipole
        || options.lr_exchange
        || options.solvation.is_some()
        || options.ml_model.is_some()
        || options.experimental_d4
        || options.external_field.magnetic_field.is_some()
        || options.spin_multiplicity.is_some_and(|m| m != 1)
    {
        return Err(Gfn1Error::InvalidInput("SOC SCC supports molecular spin-independent GFN1 without external fields or composite models".into()));
    }
    let mut reference = crate::run_electronic(system, params, options.clone())?;
    let basis = &reference.basis;
    let overlap = &reference.integrals.overlap;
    let onsite = onsite_soc(basis, soc)?;
    let operator = dual_soc(&onsite, overlap)?;
    let sspin = spin_overlap(overlap);
    let n = basis.len();
    let mut model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let external = crate::field::electric_shell_potential(&options.external_field, system, basis);
    model.charge_order = options.charge_order.max(3);
    let kernel = crate::coulomb::effective_coulomb_matrix(system, basis, &model);
    let mut q = reference.shell_charges.clone();
    let mut mixer = crate::electronic::BroydenMixer::new(
        q.len(),
        options.scc_broyden_size.max(2),
        options.mixing,
    );
    let mut previous = f64::INFINITY;
    for iteration in 0..options.max_scc {
        let coulomb =
            crate::coulomb::coulomb_energy_potential_from_matrix(basis, &model, &q, &kernel)?;
        let mut potential = coulomb.shell_potential;
        if let Some(ext) = &external {
            for (x, y) in potential.iter_mut().zip(ext) {
                *x += y;
            }
        }
        let fock =
            crate::electronic::fock_from_shell_potential(basis, overlap, &reference.h0, &potential);
        let mut h = operator.clone();
        for i in 0..n {
            for j in 0..n {
                h.re[(i, j)] += fock[(i, j)];
                h.re[(n + i, n + j)] += fock[(i, j)];
            }
        }
        let eigen = spinor_eigen(&h, &sspin, options.eigen_tolerance)?;
        let (occ, mu, entropy) = crate::spin::channel_occupations(
            &eigen.values,
            reference.nelec,
            options.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
        );
        let p = density(&eigen, &occ);
        let scalar = spin_trace(&p);
        let new_q = crate::electronic::mulliken_shell_charges(basis, overlap, &scalar);
        let mut final_coulomb =
            crate::coulomb::coulomb_energy_potential_from_matrix(basis, &model, &new_q, &kernel)?;
        let field_energy = external
            .as_ref()
            .map_or(0.0, |ext| crate::field::electric_field_energy(ext, &new_q));
        let soc_energy = p.real_trace_product(&operator);
        let band = crate::electronic::electronic_energy(&reference.h0, &scalar);
        let internal = field_energy
            + band
            + soc_energy
            + final_coulomb.second_order
            + final_coulomb.third_order
            + final_coulomb.higher_order
            + reference.repulsion_energy
            + reference.dispersion_energy
            + reference.halogen_energy;
        let energy = internal + entropy;
        let residual: Vec<f64> = new_q.iter().zip(&q).map(|(a, b)| a - b).collect();
        let max = residual.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if let Some(logger) = &options.logger {
            logger.event(crate::LogLevel::Info,"soc_scf_iteration",serde_json::json!({"iteration":iteration+1,"total_free_hartree":energy,"charge_residual_max":max}))?;
        }
        if max < options.charge_tolerance && (energy - previous).abs() < options.energy_tolerance {
            let weighted: Vec<f64> = eigen.values.iter().zip(&occ).map(|(e, o)| e * o).collect();
            let mut w = spin_trace(&density(&eigen, &weighted));
            // Tr[P dH_SOC]=Tr[K dS], K=spin-trace Re[(O P+P O)/2].
            // W-K gives the full SOC overlap derivative via the existing Pulay driver.
            let op = onsite.re.matmul(&p.re)?;
            let oi = onsite.im.matmul(&p.im)?;
            let po = p.re.matmul(&onsite.re)?;
            let pi = p.im.matmul(&onsite.im)?;
            for i in 0..n {
                for j in 0..n {
                    for offset in [0, n] {
                        w[(i, j)] -= 0.5
                            * (op[(offset + i, offset + j)] - oi[(offset + i, offset + j)]
                                + po[(offset + i, offset + j)]
                                - pi[(offset + i, offset + j)]);
                    }
                }
            }
            if let Some(ext) = &external {
                for (x, y) in final_coulomb.shell_potential.iter_mut().zip(ext) {
                    *x += y;
                }
            }
            reference.fock = crate::electronic::fock_from_shell_potential(
                basis,
                overlap,
                &reference.h0,
                &final_coulomb.shell_potential,
            );
            reference.dipole = crate::field::mulliken_dipole(
                system,
                &final_coulomb.atomic_charges,
                options.external_field.origin,
            );
            reference.external_field_energy = field_energy;
            reference.density = scalar;
            reference.energy_weighted_density = w;
            reference.shell_charges = new_q;
            reference.atomic_charges = final_coulomb.atomic_charges;
            reference.shell_scc_potential = final_coulomb.shell_potential;
            reference.isotropic_scc_energy = final_coulomb.second_order;
            reference.third_order_energy = final_coulomb.third_order;
            reference.higher_order_energy = final_coulomb.higher_order;
            reference.electronic_energy = band + soc_energy;
            reference.total_internal = internal;
            reference.total_free = energy;
            reference.electronic_entropy_term = entropy;
            reference.fermi_level = mu;
            reference.electronic_temperature = options.electronic_temperature.max(0.0);
            reference.method_citations.push("Jha2023SOC");
            return Ok(SocResult {
                calculation_id: None,
                total_internal: internal,
                total_free: energy,
                soc_energy,
                atomic_charges: reference.atomic_charges.clone(),
                eigen,
                occupations: occ,
                density: p,
                iterations: iteration + 1,
                max_charge_residual: max,
                dipole: reference.dipole,
                method_citations: reference.method_citations.clone(),
                parameter_provenance: soc.provenance.clone(),
                scalar_proxy: reference,
            });
        }
        if !energy.is_finite() {
            return Err(Gfn1Error::InvalidInput("non-finite SOC energy".into()));
        }
        previous = energy;
        q = mixer.next(&q, &residual).unwrap_or_else(|| {
            q.iter()
                .zip(&residual)
                .map(|(a, b)| a + options.mixing * b)
                .collect()
        });
    }
    Err(Gfn1Error::SccNotConverged {
        iterations: options.max_scc,
        rms: f64::INFINITY,
    })
}
pub fn soc_analytic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    soc: &SocParameters,
) -> Result<(SocResult, Vec<crate::math::Vec3>)> {
    let result = run_soc_scc(system, params, options, soc)?;
    let gradient = crate::gradient::analytic_gradient_from_result(
        system,
        params,
        result.scalar_proxy.clone(),
        &crate::AnalyticGradientOptions {
            electronic: options.clone(),
            ..Default::default()
        },
    )?;
    Ok((result, gradient.gradient))
}

pub struct SocPolarizability {
    pub tensor: [[f64; 3]; 3],
    pub max_normalized_residual: f64,
}
/// Fixed-electron-number, clamped-ion static spinor CP-SCC response. A single
/// shell-space LU factorization is shared by all three electric-field RHSs.
pub fn soc_static_polarizability(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    result: &SocResult,
) -> Result<SocPolarizability> {
    let e = &result.scalar_proxy;
    let basis = &e.basis;
    let n = basis.len();
    let ns = basis.shells.len();
    let m = 2 * n;
    let kt = e.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
    let bare = |scalar: &Matrix| -> Result<Vec<f64>> {
        let h = CMatrix {
            n: m,
            re: spin_overlap(scalar),
            im: Matrix::zeros(m, m),
        };
        let coeff = &result.eigen.vectors;
        let transformed = coeff.adjoint().matmul(&h.matmul(coeff)?)?;
        let slopes: Vec<f64> = result
            .occupations
            .iter()
            .map(|&f| if kt > 0.0 { -f * (1.0 - f) / kt } else { 0.0 })
            .collect();
        let denom: f64 = slopes.iter().sum();
        let dmu = if denom.abs() > 1e-30 {
            (0..m)
                .map(|i| slopes[i] * transformed.re[(i, i)])
                .sum::<f64>()
                / denom
        } else {
            0.0
        };
        let mut response = CMatrix::zeros(m);
        for i in 0..m {
            response.re[(i, i)] = slopes[i] * (transformed.re[(i, i)] - dmu);
            for j in 0..i {
                let gap = result.eigen.values[i] - result.eigen.values[j];
                let df = result.occupations[i] - result.occupations[j];
                let factor = if gap.abs() > 1e-10 {
                    df / gap
                } else if kt > 0.0 {
                    0.5 * (slopes[i] + slopes[j])
                } else if df.abs() < 1e-10 {
                    0.0
                } else {
                    return Err(Gfn1Error::InvalidInput("spinor response has a degenerate partially occupied branch; use an ensemble".into()));
                };
                response.re[(i, j)] = factor * transformed.re[(i, j)];
                response.re[(j, i)] = response.re[(i, j)];
                response.im[(i, j)] = factor * transformed.im[(i, j)];
                response.im[(j, i)] = -response.im[(i, j)];
            }
        }
        let dp = spin_trace(&coeff.matmul(&response)?.matmul(&coeff.adjoint())?);
        let overlap = &e.integrals.overlap;
        let mut q = vec![0.0; ns];
        for (s, sh) in basis.shells.iter().enumerate() {
            for i in sh.first_ao..sh.first_ao + sh.nao {
                for j in 0..n {
                    q[s] -= dp[(i, j)] * overlap[(j, i)];
                }
            }
        }
        Ok(q)
    };
    let mut chi = Matrix::zeros(ns, ns);
    let zero = Matrix::zeros(n, n);
    for s in 0..ns {
        let mut v = vec![0.0; ns];
        v[s] = 1.0;
        let f =
            crate::electronic::fock_from_shell_potential(basis, &e.integrals.overlap, &zero, &v);
        let q = bare(&f)?;
        for t in 0..ns {
            chi[(t, s)] = q[t];
        }
    }
    let kernel = crate::response::cpxtb::response_shell_scc_kernel(system, params, e)?;
    let product = chi.matmul(&kernel)?;
    let mut jacobian = Matrix::identity(ns);
    for i in 0..ns {
        for j in 0..ns {
            jacobian[(i, j)] -= product[(i, j)];
        }
    }
    let lu = DenseLu::factor(&jacobian)?;
    let mut tensor = [[0.0; 3]; 3];
    let mut residual = 0.0_f64;
    for axis in 0..3 {
        let potential: Vec<f64> = basis
            .shells
            .iter()
            .map(|s| -system.atoms[s.atom_index].position.to_array()[axis])
            .collect();
        let field = crate::electronic::fock_from_shell_potential(
            basis,
            &e.integrals.overlap,
            &zero,
            &potential,
        );
        let rhs = bare(&field)?;
        let q = lu.solve_vec(&rhs)?;
        let norm_a = jacobian
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let norm_q = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_r = (0..ns)
            .map(|i| {
                let r = (0..ns).map(|j| jacobian[(i, j)] * q[j]).sum::<f64>() - rhs[i];
                r * r
            })
            .sum::<f64>()
            .sqrt();
        let scale = (norm_a * norm_q + norm_b).max(1e-300);
        let normalized = norm_r / scale;
        let cp = crate::resources::current_cp();
        if !normalized.is_finite()
            || ((cp.absolute_tolerance.is_some() || cp.relative_tolerance.is_some())
                && norm_r > cp.residual_target(0., 1e-7, scale))
        {
            return Err(Gfn1Error::LinearAlgebra(format!(
                "molecular spinor CP-SCC residual {norm_r:e} (normalized {normalized:e}) exceeds configured tolerance"
            )));
        }
        residual = residual.max(normalized);
        for (s, sh) in basis.shells.iter().enumerate() {
            for a in 0..3 {
                tensor[a][axis] += q[s] * system.atoms[sh.atom_index].position.to_array()[a];
            }
        }
    }
    Ok(SocPolarizability {
        tensor,
        max_normalized_residual: residual,
    })
}
