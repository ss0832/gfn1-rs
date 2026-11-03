// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic affine/nuclear response building blocks. The generalized coordinate
//! convention is R'=F(R+u), A'=FA. Mixed coordinate acceleration is included.
use crate::jets::Jet2;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::Result;

/// Cartesian vector after an affine deformation and nuclear displacements.
/// ndof=9 is pure affine; ndof=3N+9 additionally seeds the two reference nuclei.
pub(crate) fn affine_nuclear_vector_jet(
    vector: Vec3,
    ndof: usize,
    positive: usize,
    negative: usize,
) -> [Jet2; 3] {
    let base = ndof - 9;
    let r = vector.to_array();
    std::array::from_fn(|a| {
        let mut value = Jet2::constant(r[a], ndof);
        for b in 0..3 {
            value.gradient[base + 3 * a + b] = r[b];
        }
        if base > 0 {
            value.gradient[3 * positive + a] += 1.;
            value.gradient[3 * negative + a] -= 1.;
            for b in 0..3 {
                let strain = base + 3 * a + b;
                for (atom, sign) in [(positive, 1.), (negative, -1.)] {
                    let nuclear = 3 * atom + b;
                    value.hessian[nuclear * ndof + strain] += sign;
                    value.hessian[strain * ndof + nuclear] += sign;
                }
            }
        }
        value
    })
}

pub(crate) fn engineering_basis(s: usize) -> [(usize, f64); 2] {
    match s {
        0 => [(0, 1.), (0, 0.)],
        1 => [(4, 1.), (4, 0.)],
        2 => [(8, 1.), (8, 0.)],
        3 => [(5, 0.5), (7, 0.5)],
        4 => [(2, 0.5), (6, 0.5)],
        5 => [(1, 0.5), (3, 0.5)],
        _ => unreachable!(),
    }
}
/// Transform the affine tail of a full nuclear/Cartesian-strain Hessian.
pub fn engineering_hessian(raw: &Matrix, nuclear: usize) -> Result<Matrix> {
    if raw.rows() != nuclear + 9 || raw.cols() != nuclear + 9 {
        return Err(crate::Gfn1Error::InvalidInput(
            "affine Hessian dimension mismatch".into(),
        ));
    }
    let mut out = Matrix::zeros(nuclear + 6, nuclear + 6);
    for x in 0..nuclear + 6 {
        for y in 0..nuclear + 6 {
            let xs = if x < nuclear {
                [(x, 1.), (x, 0.)]
            } else {
                engineering_basis(x - nuclear).map(|(i, w)| (nuclear + i, w))
            };
            let ys = if y < nuclear {
                [(y, 1.), (y, 0.)]
            } else {
                engineering_basis(y - nuclear).map(|(i, w)| (nuclear + i, w))
            };
            for (i, wi) in xs {
                for (j, wj) in ys {
                    out[(x, y)] += wi * wj * raw[(i, j)];
                }
            }
        }
    }
    Ok(out)
}

/// Learned-parameter and correction strain Hessian at fixed level sensitivities.
/// The electronic eigenvector response is added by the full elastic driver.
pub fn ml_frozen_strain_hessian(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    scf: &super::PbcSccResult,
) -> Result<Matrix> {
    match &scf.ml_parameters {
        Some(view) => view
            .affine_fixed_parameter_hessian(&super::ml::level_sensitivities(system, params, scf)?),
        None => Ok(Matrix::zeros(6, 6)),
    }
}

pub fn ml_frozen_affine_nuclear_hessian(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    scf: &super::PbcSccResult,
) -> Result<Matrix> {
    match &scf.ml_parameters {
        Some(view) => view.affine_nuclear_fixed_parameter_hessian(&super::ml::level_sensitivities(
            system, params, scf,
        )?),
        None => Ok(Matrix::zeros(
            3 * system.atoms.len() + 9,
            3 * system.atoms.len() + 9,
        )),
    }
}

/// Complete analytic classical geometry block, before electronic response.
pub fn classical_affine_nuclear_hessian(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    options: &crate::ElectronicOptions,
) -> Result<Matrix> {
    let mut out = crate::repulsion::repulsion_affine_nuclear_hessian(system, params)?;
    if options.enable_dispersion {
        let d3 = crate::dispersion::dispersion_affine_nuclear_hessian(
            system,
            params,
            options.d3_reference_path.as_deref(),
        )?;
        for (a, b) in out.as_mut_slice().iter_mut().zip(d3.as_slice()) {
            *a += b;
        }
    }
    let halogen = crate::halogen::halogen_affine_nuclear_hessian(system, params)?;
    for (a, b) in out.as_mut_slice().iter_mut().zip(halogen.as_slice()) {
        *a += b;
    }
    Ok(out)
}

fn jet_dot(v: &[Jet2; 3]) -> Jet2 {
    v.iter()
        .fold(Jet2::constant(0., v[0].gradient.len()), |a, b| {
            a.add(&b.mul(b))
        })
}
fn affine_metric(ndof: usize) -> (Jet2, Vec<Vec<Jet2>>) {
    let base = ndof - 9;
    let f: Vec<Vec<_>> = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| {
                    let mut v = Jet2::constant(if i == j { 1. } else { 0. }, ndof);
                    v.gradient[base + 3 * i + j] = 1.;
                    v
                })
                .collect()
        })
        .collect();
    let minor = |row: usize, col: usize| {
        let rr: Vec<_> = (0..3).filter(|&i| i != row).collect();
        let cc: Vec<_> = (0..3).filter(|&i| i != col).collect();
        f[rr[0]][cc[0]]
            .mul(&f[rr[1]][cc[1]])
            .sub(&f[rr[0]][cc[1]].mul(&f[rr[1]][cc[0]]))
            .scale(if (row + col) % 2 == 0 { 1. } else { -1. })
    };
    let det = (0..3).fold(Jet2::constant(0., ndof), |a, j| {
        a.add(&f[0][j].mul(&minor(0, j)))
    });
    let inverse = (0..3)
        .map(|i| (0..3).map(|j| minor(j, i).div(&det)).collect())
        .collect();
    (det, inverse)
}
fn phase_structure(
    system: &crate::PeriodicSystem,
    g: Vec3,
    weights: &[f64],
    ndof: usize,
) -> (Jet2, Jet2) {
    let mut re = Jet2::constant(0., ndof);
    let mut im = Jet2::constant(0., ndof);
    let gg = g.to_array();
    for (a, atom) in system.atoms.iter().enumerate() {
        if weights[a] == 0. {
            continue;
        }
        let mut p = Jet2::constant(g.dot(atom.position), ndof);
        for k in 0..3 {
            p.gradient[3 * a + k] = gg[k];
        }
        let c = p
            .compose(p.value.cos(), -p.value.sin(), -p.value.cos())
            .scale(weights[a]);
        let s = p
            .compose(p.value.sin(), p.value.cos(), -p.value.sin())
            .scale(weights[a]);
        re = re.add(&c);
        im = im.add(&s);
    }
    (re, im)
}

/// Exact fixed-charge electrostatic geometry Hessian, including real and
/// reciprocal Ewald, QCore R^-3, short-range KO and background terms. Alpha is
/// held fixed at the reference; the converged Ewald sum is alpha independent.
pub fn electrostatic_affine_nuclear_hessian(
    system: &crate::PeriodicSystem,
    scf: &super::PbcSccResult,
    pbc: &super::PbcOptions,
) -> Result<Matrix> {
    use super::ewald::{
        exp1, qcore_r3_real_value_derivatives, qcore_short_value_derivatives, resolve_alpha,
        QCORE_R3_COEFF, TAU,
    };
    let lattice = system.lattice.ok_or_else(|| {
        crate::Gfn1Error::InvalidInput("affine electrostatics requires a periodic cell".into())
    })?;
    let nat = system.atoms.len();
    let d = 3 * nat + 9;
    crate::resources::check_dense_workspace(d, d, 24, "affine Ewald jets")?;
    let alpha = resolve_alpha(system, &pbc.ewald);
    let cutoff = TAU / alpha;
    let mut total = Jet2::constant(0., d);
    let images = lattice.image_offsets(cutoff.max(pbc.ewald.sr_cutoff));
    let q = &scf.shell_charges;
    let qa = &scf.atomic_charges;
    let model = &scf.shell_model;
    // Atomic 1/R real sum; the diagonal home cell is the only omitted image.
    for a in 0..nat {
        for b in 0..=a {
            let weight = qa[a] * qa[b] * if a == b { 0.5 } else { 1. };
            if weight == 0. {
                continue;
            }
            for &off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let v =
                    system.atoms[a].position - system.atoms[b].position - lattice.translation(off);
                let r = v.norm();
                if r < 1e-12 || r > cutoff {
                    continue;
                }
                let radial = jet_dot(&affine_nuclear_vector_jet(v, d, a, b)).powf(0.5);
                let er = crate::math::erfc(alpha * r);
                let ex = (-alpha * alpha * r * r).exp();
                let c = 2. * alpha / std::f64::consts::PI.sqrt();
                let first = -er / (r * r) - c * ex / r;
                let second =
                    2. * er / r.powi(3) + 2. * c * ex / (r * r) + 2. * c * alpha * alpha * ex;
                total = total.add(&radial.compose(er / r, first, second).scale(weight));
            }
        }
    }
    for i in 0..q.len() {
        for j in 0..=i {
            let weight = q[i] * q[j] * if i == j { 0.5 } else { 1. };
            if weight == 0. {
                continue;
            }
            let a = scf.basis.shells[i].atom_index;
            let b = scf.basis.shells[j].atom_index;
            let eta = crate::coulomb::harmonic_average(model.hardness[i], model.hardness[j]);
            for &off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let v =
                    system.atoms[a].position - system.atoms[b].position - lattice.translation(off);
                let r = v.norm();
                if r < 1e-12 {
                    continue;
                }
                let mut derivatives = [0.; 3];
                if r <= cutoff {
                    let (v, d1, d2) = qcore_r3_real_value_derivatives(r, eta, alpha);
                    derivatives[0] += QCORE_R3_COEFF * v;
                    derivatives[1] += QCORE_R3_COEFF * d1;
                    derivatives[2] += QCORE_R3_COEFF * d2;
                }
                if r <= pbc.ewald.sr_cutoff {
                    let (v, d1, d2) = qcore_short_value_derivatives(r, eta);
                    derivatives[0] += v;
                    derivatives[1] += d1;
                    derivatives[2] += d2;
                }
                if derivatives.iter().all(|&v| v == 0.) {
                    continue;
                }
                let radial = jet_dot(&affine_nuclear_vector_jet(v, d, a, b)).powf(0.5);
                total = total.add(
                    &radial
                        .compose(derivatives[0], derivatives[1], derivatives[2])
                        .scale(weight),
                );
            }
        }
    }
    let (jac, inverse) = affine_metric(d);
    let inv_volume = jac.scale(lattice.volume()).powf(-1.);
    let mut q1 = vec![0.; nat];
    let mut q2 = vec![0.; nat];
    for (i, sh) in scf.basis.shells.iter().enumerate() {
        q1[sh.atom_index] += q[i] / model.hardness[i];
        q2[sh.atom_index] += q[i] / model.hardness[i].powi(2);
    }
    for (_, g) in lattice.reciprocal_vectors_within(2. * alpha * TAU, false) {
        let ga = g.to_array();
        let reciprocal: [Jet2; 3] = std::array::from_fn(|a| {
            (0..3).fold(Jet2::constant(0., d), |s, b| {
                s.add(&inverse[b][a].scale(ga[b]))
            })
        });
        let g2 = jet_dot(&reciprocal);
        let x = g2.scale(1. / (4. * alpha * alpha));
        let (r0, i0) = phase_structure(system, g, qa, d);
        let (r1, i1) = phase_structure(system, g, &q1, d);
        let (r2, i2) = phase_structure(system, g, &q2, d);
        let sf = r0.mul(&r0).add(&i0.mul(&i0));
        total = total.add(
            &inv_volume
                .mul(&x.scale(-1.).exp().div(&g2))
                .mul(&sf)
                .scale(2. * std::f64::consts::PI),
        );
        let sf3 = r2
            .mul(&r0)
            .add(&i2.mul(&i0))
            .add(&r1.mul(&r1))
            .add(&i1.mul(&i1))
            .scale(0.5);
        let e1 = x.compose(
            exp1(x.value),
            -(-x.value).exp() / x.value,
            (-x.value).exp() * (1. / x.value + 1. / x.value.powi(2)),
        );
        total = total.add(
            &inv_volume
                .mul(&e1)
                .mul(&sf3)
                .scale(QCORE_R3_COEFF * std::f64::consts::PI),
        );
    }
    let q0 = qa.iter().sum::<f64>();
    let q1 = q1.iter().sum::<f64>();
    let q2 = q2.iter().sum::<f64>();
    let zero_structure = 0.5 * (q2 * q0 + q1 * q1);
    total = total.add(&inv_volume.scale(
        QCORE_R3_COEFF
            * 2.
            * std::f64::consts::PI
            * super::ewald::qcore_r3_k0_log(alpha)
            * zero_structure
            - std::f64::consts::PI * q0 * q0 / (2. * alpha * alpha),
    ));
    Matrix::from_vec(d, d, total.hessian)
}

fn cn_jets(
    system: &crate::PeriodicSystem,
    options: &crate::ElectronicOptions,
    d: usize,
) -> Result<Vec<Jet2>> {
    let cn = crate::coordination::coordination_with_derivatives(
        system,
        crate::coordination::CoordinationOptions {
            cutoff: options.hamiltonian.coordination_cutoff,
            ..Default::default()
        },
    )?;
    let mut out: Vec<_> = cn.cn.iter().map(|&v| Jet2::constant(v, d)).collect();
    if !options.hamiltonian.enable_cn_hamiltonian {
        return Ok(out);
    }
    let kcn = crate::coordination::CoordinationOptions::default().kcn;
    for pair in cn.pairs {
        let r = pair.r_ij.norm();
        if r < 1e-12 {
            continue;
        }
        let radial = jet_dot(&affine_nuclear_vector_jet(pair.r_ij, d, pair.i, pair.j)).powf(0.5);
        let radius = crate::data_tables::covalent_radius_d3_bohr(system.atoms[pair.i].z)?
            + crate::data_tables::covalent_radius_d3_bohr(system.atoms[pair.j].z)?;
        let value = if pair.i == pair.j {
            pair.value / 2.
        } else {
            pair.value
        };
        let second = pair.dcn_dr * (-2. / r + kcn * radius * (2. * value - 1.) / (r * r));
        let term = radial.compose(0., pair.dcn_dr, second);
        out[pair.i] = out[pair.i].add(&term);
        if pair.i != pair.j {
            out[pair.j] = out[pair.j].add(&term);
        }
    }
    Ok(out)
}
fn image_value(
    densities: &[super::complex::CMatrix],
    scf: &super::PbcSccResult,
    off: crate::lattice::ImageOffset,
    mu: usize,
    nu: usize,
) -> f64 {
    scf.kpoints
        .iter()
        .enumerate()
        .map(|(k, kp)| {
            let (c, s) = super::kpoints::bloch_phase(kp.fractional, off);
            kp.weight * (c * densities[k].re[(mu, nu)] + s * densities[k].im[(mu, nu)])
        })
        .sum()
}
fn generalized_gradient(raw: &[f64], n: usize) -> Vec<f64> {
    (0..n + 6)
        .map(|x| {
            if x < n {
                raw[x]
            } else {
                engineering_basis(x - n)
                    .into_iter()
                    .map(|(i, w)| w * raw[n + i])
                    .sum()
            }
        })
        .collect()
}
fn expand_potential(nuclear: &[Vec<f64>], strain: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = nuclear.to_vec();
    for s in [0, 5, 4, 5, 1, 3, 4, 3, 2] {
        out.push(strain[s].clone());
    }
    out
}
fn overlap_jet(
    pair: &crate::integrals::ContractedPairSecondDerivatives,
    vector: &[Jet2; 3],
    d: usize,
) -> Jet2 {
    let mut out = Jet2::constant(pair.moments[0], d);
    let first = pair.d_bra[0].to_array();
    for x in 0..d {
        out.gradient[x] = (0..3).map(|a| first[a] * vector[a].gradient[x]).sum();
        for y in 0..d {
            out.hessian[x * d + y] = (0..3)
                .map(|a| first[a] * vector[a].hessian[x * d + y])
                .sum::<f64>()
                + (0..3)
                    .map(|a| {
                        (0..3)
                            .map(|b| {
                                pair.h_bra_bra[0][a][b]
                                    * vector[a].gradient[x]
                                    * vector[b].gradient[y]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
        }
    }
    out
}
fn polynomial_jet(
    si: &crate::basis::BasisShell,
    sj: &crate::basis::BasisShell,
    r: &Jet2,
) -> Result<Jet2> {
    let radius = crate::data_tables::atomic_radius_bohr(si.z)?
        + crate::data_tables::atomic_radius_bohr(sj.z)?;
    let rr = r.scale(1. / radius).powf(0.5);
    Ok(rr
        .scale(si.poly_raw.unwrap_or(0.))
        .add_scalar(1.)
        .mul(&rr.scale(sj.poly_raw.unwrap_or(0.)).add_scalar(1.)))
}

#[derive(Clone, Debug)]
pub struct AnalyticElasticOptions {
    /// First reconverge internal coordinates at fixed lattice, using analytic forces.
    pub relax_reference: bool,
    pub relax: super::mechanical::PeriodicRelaxOptions,
    pub force_tolerance: f64,
    pub stability_tolerance: f64,
}
impl Default for AnalyticElasticOptions {
    fn default() -> Self {
        Self {
            relax_reference: false,
            relax: super::mechanical::PeriodicRelaxOptions::default(),
            force_tolerance: 1e-6,
            stability_tolerance: 1e-10,
        }
    }
}
#[derive(Clone, Debug)]
pub struct AnalyticElasticResponse {
    pub reference_system: crate::PeriodicSystem,
    /// Exact reference state for shared electromechanical calculations.
    /// Excluded from the lightweight JSON representation.
    pub scf: super::PbcSccResult,
    pub reference_free_energy_hartree: f64,
    pub reference_stress: Matrix,
    pub reference_max_gradient: f64,
    pub reference_gradient: Vec<Vec3>,
    /// Material d²F/dGreen-strain² per reference volume; engineering Voigt.
    pub clamped_stiffness: Matrix,
    pub internal_strain_coupling: Matrix,
    pub nuclear_hessian: Matrix,
    pub generalized_hessian: Matrix,
    pub raw_generalized_hessian: Matrix,
    pub relaxed_stiffness: Option<Matrix>,
    pub internal_displacement_strain: Option<Matrix>,
    pub internal_curvatures: Vec<f64>,
    pub relaxed_response_error: Option<String>,
    pub raw_max_asymmetry: f64,
    pub translation_coupling_residual: f64,
    pub electronic_temperature_kelvin: f64,
    pub scc_energy_tolerance: f64,
    pub scc_charge_tolerance: f64,
}
impl AnalyticElasticResponse {
    pub fn as_json(&self) -> serde_json::Value {
        let row = |m: &Matrix| {
            (0..m.rows())
                .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        serde_json::json!({"method":"analytic joint nuclear/affine CP-SCC; full ML structural response",
            "scc_diagnostics":self.scf.scc_diagnostics.as_json(),"scc_iterations":self.scf.iterations,
            "reference_shell_charges":self.scf.shell_charges,
            "voigt_order":super::mechanical::VOIGT_ORDER,"reference_free_energy_hartree":self.reference_free_energy_hartree,
            "reference_stress_hartree_per_bohr3":row(&self.reference_stress),"reference_max_gradient_hartree_per_bohr":self.reference_max_gradient,
            "reference_gradient_hartree_per_bohr":self.reference_gradient.iter().map(|g|g.to_array()).collect::<Vec<_>>(),
            "reference_forces_hartree_per_bohr":self.reference_gradient.iter().map(|g|g.to_array().map(|x|-x)).collect::<Vec<_>>(),
            "clamped_stiffness_hartree_per_bohr3":row(&self.clamped_stiffness),"clamped_stiffness_gpa":row(&self.clamped_stiffness).into_iter().map(|r|r.into_iter().map(|x|x*super::mechanical::STRESS_AU_TO_GPA).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "relaxed_stiffness_hartree_per_bohr3":self.relaxed_stiffness.as_ref().map(row),"internal_displacement_strain_bohr":self.internal_displacement_strain.as_ref().map(row),
            "internal_strain_coupling_hartree_per_bohr":row(&self.internal_strain_coupling),"nuclear_hessian_hartree_per_bohr2":row(&self.nuclear_hessian),
            "raw_max_asymmetry":self.raw_max_asymmetry,"translation_coupling_residual":self.translation_coupling_residual,
            "internal_curvatures_hartree_per_bohr2":self.internal_curvatures,"relaxed_response_error":self.relaxed_response_error,
            "electronic_temperature_kelvin":self.electronic_temperature_kelvin,
            "scc_energy_tolerance":self.scc_energy_tolerance,"scc_charge_tolerance":self.scc_charge_tolerance,
            "reference_positions_bohr":self.reference_system.atoms.iter().map(|a|a.position.to_array()).collect::<Vec<_>>(),
            "reference_cell_bohr":self.reference_system.lattice.unwrap().cell.col.map(|v|v.to_array())})
    }
}

/// Full analytic material elasticity and internal-strain coupling of scalar
/// GFN1/ML-xTB. One SCC plus canonical CP response; no displaced geometry.
pub fn pbc_elastic_response(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    options: &crate::ElectronicOptions,
    pbc: &super::PbcOptions,
    opts: &AnalyticElasticOptions,
) -> Result<AnalyticElasticResponse> {
    pbc_elastic_response_with_guess(system, params, options, pbc, opts, None)
}
/// Full analytic response initialized from a previously converged charge state.
pub fn pbc_elastic_response_with_guess(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    options: &crate::ElectronicOptions,
    pbc: &super::PbcOptions,
    opts: &AnalyticElasticOptions,
    guess: Option<&super::PbcSccGuess>,
) -> Result<AnalyticElasticResponse> {
    super::electromechanical::require_scalar(system, options)?;
    if !opts.force_tolerance.is_finite()
        || opts.force_tolerance <= 0.
        || !opts.stability_tolerance.is_finite()
        || opts.stability_tolerance <= 0.
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "invalid analytic elastic tolerance".into(),
        ));
    }
    let mut electronic = options.clone();
    electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
    electronic.charge_tolerance = electronic.charge_tolerance.min(1e-10);
    crate::resources::preflight_response(
        system,
        params,
        &electronic,
        pbc.kmesh.size.into_iter().product(),
        false,
    )?;
    if crate::resources::current().memory_budget_bytes.is_some() {
        let basis = crate::basis::BasisSet::build(
            system,
            params,
            crate::BasisOptions {
                nprim: electronic.nprim,
            },
        )?;
        let atoms = system.atoms.len();
        let dof = atoms
            .checked_mul(3)
            .and_then(|n| n.checked_add(9))
            .ok_or_else(|| crate::Gfn1Error::InvalidInput("elastic workspace overflow".into()))?;
        let estimate = (|| {
            let kpoints = pbc
                .kmesh
                .size
                .into_iter()
                .try_fold(1usize, usize::checked_mul)?;
            let response = basis
                .len()
                .checked_mul(basis.len())?
                .checked_mul(8 * 48)?
                .checked_mul(dof - 2)?
                .checked_mul(kpoints)?;
            let fixed_count = atoms
                .checked_mul(atoms)?
                .checked_add(atoms)?
                .checked_add(basis.shells.len())?
                .checked_add(80)?;
            let fixed = dof
                .checked_mul(dof)?
                .checked_mul(8)?
                .checked_mul(fixed_count)?;
            response.checked_add(fixed)
        })()
        .ok_or_else(|| crate::Gfn1Error::InvalidInput("elastic workspace overflow".into()))?;
        crate::resources::check_workspace(
            estimate,
            "joint nuclear/strain response and fixed-geometry jets",
        )?;
    }
    let mut relaxed_guess = None;
    let reference = if opts.relax_reference {
        let mut relaxation = opts.relax.clone();
        relaxation.relax_cell = false;
        let result = super::mechanical::pbc_relax_with_guess(
            system,
            params,
            &electronic,
            pbc,
            &relaxation,
            guess,
        )?;
        if !result.converged {
            return Err(crate::Gfn1Error::InvalidInput(format!(
                "elastic reference relaxation failed: {}",
                result.termination
            )));
        }
        relaxed_guess = Some(result.scf_guess);
        result.system
    } else {
        system.clone()
    };
    let scf = super::run_pbc_scc_with_guess(
        &reference,
        params,
        &electronic,
        pbc,
        relaxed_guess.as_ref().or(guess),
    )?;
    if !scf.converged {
        return Err(crate::Gfn1Error::InvalidInput(
            "analytic elastic reference SCC did not converge".into(),
        ));
    }
    let n = 3 * reference.atoms.len();
    let d = n + 9;
    let dg = n + 6;
    let count = scf.basis.shells.len() + reference.atoms.len() + 40;
    crate::resources::check_dense_workspace(d, d, count, "analytic affine/nuclear fixed geometry")?;
    let strain =
        super::electromechanical::strain_skeletons(&reference, params, &scf, &electronic, pbc)?;
    let mut all_skeletons = Vec::new();
    for (k, kp) in scf.kpoints.iter().enumerate() {
        let mut nuclear = super::hessian::kpoint_skeleton_derivatives(
            &reference,
            params,
            &scf,
            &electronic,
            pbc,
            kp.fractional,
        )?;
        nuclear.overlap.extend(strain[k].overlap.iter().cloned());
        nuclear.fock.extend(strain[k].fock.iter().cloned());
        nuclear
            .shell_potential
            .extend(strain[k].shell_potential.iter().cloned());
        nuclear
            .coordination
            .extend(strain[k].coordination.iter().cloned());
        all_skeletons.push(nuclear);
    }
    let (dp, dw, dq) = super::hessian::kpoint_cpxtb_from_skeletons(&scf, &all_skeletons, true)?;
    let nuclear_potential = &all_skeletons[0].shell_potential[..n];
    let full_potential = expand_potential(nuclear_potential, &strain[0].shell_potential);
    let kernel = super::hessian::periodic_response_kernel(&scf);
    let response_potential: Vec<Vec<f64>> = dq
        .iter()
        .map(|q| crate::linalg::matrix_vector_product(&kernel, q))
        .collect::<Result<_>>()?;
    let cn = cn_jets(&reference, &electronic, d)?;
    let mut levels: Vec<_> = scf
        .basis
        .shells
        .iter()
        .enumerate()
        .map(|(s, sh)| {
            let mut value = Jet2::constant(scf.bloch.self_energies[s], d);
            if electronic.hamiltonian.enable_cn_hamiltonian {
                for x in 0..d {
                    value.gradient[x] += scf.bloch.dsedcn[s] * cn[sh.atom_index].gradient[x];
                    for y in 0..d {
                        value.hessian[x * d + y] +=
                            scf.bloch.dsedcn[s] * cn[sh.atom_index].hessian[x * d + y];
                    }
                }
            }
            value
        })
        .collect();
    if let Some(view) = &scf.ml_parameters {
        let nuclear = view.level_jacobian()?;
        let affine = view.affine_level_jacobian()?;
        for (s, sh) in scf.basis.shells.iter().enumerate() {
            let a = sh.atom_index;
            let l = sh.angular.as_index();
            for x in 0..n {
                levels[s].gradient[x] += nuclear[x][a][l];
            }
            for (x, source) in [0, 5, 4, 5, 1, 3, 4, 3, 2].into_iter().enumerate() {
                levels[s].gradient[n + x] += affine[source][a][l];
            }
        }
    }
    let mut frozen = Jet2::constant(0., d);
    let mut asymmetric = Matrix::zeros(d, d);
    let mut responsive = Matrix::zeros(dg, dg);
    let basis = &scf.basis;
    let lattice = reference.lattice.unwrap();
    let minimum = crate::basis::atom_min_exponents(basis, reference.atoms.len());
    let mut atom_aos = vec![Vec::new(); reference.atoms.len()];
    for (mu, ao) in basis.aos.iter().enumerate() {
        atom_aos[ao.atom_index].push(mu);
    }
    for off in lattice.image_offsets(pbc.ao_cutoff) {
        let t = lattice.translation(off);
        for a in 0..reference.atoms.len() {
            for b in 0..reference.atoms.len() {
                if off.is_origin() && a == b {
                    continue;
                }
                let ra = reference.atoms[a].position;
                let rb = reference.atoms[b].position + t;
                let vector = ra - rb;
                let r = vector.norm();
                if r < 1e-12
                    || r > pbc.ao_cutoff
                    || crate::basis::overlap_screened(&minimum, a, b, r * r)
                {
                    continue;
                }
                let v = affine_nuclear_vector_jet(vector, d, a, b);
                let radial = jet_dot(&v).powf(0.5);
                for &mu in &atom_aos[a] {
                    for &nu in &atom_aos[b] {
                        let si = basis.aos[mu].shell_index;
                        let sj = basis.aos[nu].shell_index;
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            ra,
                            rb,
                        );
                        let overlap = overlap_jet(&pair, &v, d);
                        let poly = polynomial_jet(&basis.shells[si], &basis.shells[sj], &radial)?
                            .scale(crate::hamiltonian::hscale(
                                &basis.shells[si],
                                &basis.shells[sj],
                                params,
                            )?);
                        let h0 = levels[si]
                            .add(&levels[sj])
                            .scale(0.5)
                            .mul(&poly)
                            .mul(&overlap);
                        let p = image_value(&scf.density_k, &scf, off, mu, nu);
                        let w = image_value(&scf.ew_density_k, &scf, off, mu, nu);
                        let potential =
                            0.5 * (scf.shell_scc_potential[si] + scf.shell_scc_potential[sj]);
                        frozen = frozen
                            .add(&h0.scale(p))
                            .sub(&overlap.scale(w + p * potential));
                        for x in 0..d {
                            for y in 0..d {
                                asymmetric[(x, y)] -= p
                                    * overlap.gradient[x]
                                    * 0.5
                                    * (full_potential[y][si] + full_potential[y][sj]);
                            }
                        }
                        let h0x = generalized_gradient(&h0.gradient, n);
                        let sx = generalized_gradient(&overlap.gradient, n);
                        for y in 0..dg {
                            let py = image_value(&dp[y], &scf, off, mu, nu);
                            let wy = image_value(&dw[y], &scf, off, mu, nu);
                            let shift =
                                0.5 * (response_potential[y][si] + response_potential[y][sj]);
                            for x in 0..dg {
                                responsive[(x, y)] +=
                                    py * h0x[x] - (wy + py * potential + p * shift) * sx[x];
                            }
                        }
                    }
                }
            }
        }
    }
    // Onsite diagonal H0 levels; atomic reference charges and on-site SCC
    // energies have no explicit geometry dependence.
    for (s, sh) in basis.shells.iter().enumerate() {
        for mu in sh.first_ao..sh.first_ao + sh.nao {
            let off = crate::lattice::ImageOffset::origin();
            let p = image_value(&scf.density_k, &scf, off, mu, mu);
            frozen = frozen.add(&levels[s].scale(p));
            let first = generalized_gradient(&levels[s].gradient, n);
            for y in 0..dg {
                let py = image_value(&dp[y], &scf, off, mu, mu);
                for x in 0..dg {
                    responsive[(x, y)] += py * first[x];
                }
            }
        }
    }
    let mut fixed = Matrix::from_vec(d, d, frozen.hessian)?;
    for block in [
        electrostatic_affine_nuclear_hessian(&reference, &scf, pbc)?,
        classical_affine_nuclear_hessian(&reference, params, &electronic)?,
        ml_frozen_affine_nuclear_hessian(&reference, params, &scf)?,
    ] {
        for (a, b) in fixed.as_mut_slice().iter_mut().zip(block.as_slice()) {
            *a += b;
        }
    }
    for (a, b) in fixed.as_mut_slice().iter_mut().zip(asymmetric.as_slice()) {
        *a += b;
    }
    let mut raw = engineering_hessian(&fixed, n)?;
    for x in 0..dg {
        for y in 0..dg {
            raw[(x, y)] += responsive[(x, y)]
                + dq[y]
                    .iter()
                    .zip(&all_skeletons[0].shell_potential[x])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
        }
    }
    let asymmetry = raw.max_abs_diff(&raw.transpose());
    let mut h = raw.clone();
    for x in 0..dg {
        for y in 0..dg {
            h[(x, y)] = 0.5 * (raw[(x, y)] + raw[(y, x)]);
        }
    }
    let gradient = super::pbc_gradient_from_scc(&reference, params, scf, &electronic, pbc)?;
    let reference_max_gradient = gradient.max_gradient;
    let stressed =
        super::pbc_stress_from_scc(&reference, params, gradient.scf, &electronic, pbc, &lattice)?;
    let stress = stressed.stress;
    let scf = stressed.scf;
    let mut clamped = Matrix::zeros(6, 6);
    let mut lambda = Matrix::zeros(n, 6);
    let mut nuclear = Matrix::zeros(n, n);
    for x in 0..n {
        for y in 0..n {
            nuclear[(x, y)] = h[(x, y)];
        }
        for y in 0..6 {
            lambda[(x, y)] = h[(x, n + y)];
        }
    }
    for x in 0..6 {
        for y in 0..6 {
            let mut geom = 0.;
            for (ia, wa) in engineering_basis(x) {
                for (ib, wb) in engineering_basis(y) {
                    let (a, b) = (ia / 3, ia % 3);
                    let (c, e) = (ib / 3, ib % 3);
                    if a == c {
                        geom += wa * wb * stress[(b, e)];
                    }
                }
            }
            clamped[(x, y)] = h[(n + x, n + y)] / lattice.volume() - geom;
        }
    }
    let mut translation = 0.0_f64;
    for axis in 0..3 {
        for j in 0..6 {
            translation = translation.max(
                (0..reference.atoms.len())
                    .map(|a| lambda[(3 * a + axis, j)])
                    .sum::<f64>()
                    .abs(),
            );
        }
    }
    let (relaxed, du, curvatures, error) = relaxed_ionic_response(
        &nuclear,
        &lambda,
        &clamped,
        lattice.volume(),
        reference_max_gradient,
        opts,
    )?;
    Ok(AnalyticElasticResponse {
        reference_system: reference,
        reference_free_energy_hartree: scf.total_free,
        scf,
        reference_stress: stress,
        reference_max_gradient,
        reference_gradient: gradient.gradient,
        clamped_stiffness: clamped,
        internal_strain_coupling: lambda,
        nuclear_hessian: nuclear,
        generalized_hessian: h,
        raw_generalized_hessian: raw,
        relaxed_stiffness: relaxed,
        internal_displacement_strain: du,
        internal_curvatures: curvatures,
        relaxed_response_error: error,
        raw_max_asymmetry: asymmetry,
        translation_coupling_residual: translation,
        electronic_temperature_kelvin: electronic.electronic_temperature,
        scc_energy_tolerance: electronic.energy_tolerance,
        scc_charge_tolerance: electronic.charge_tolerance,
    })
}

type IonicRelaxation = (Option<Matrix>, Option<Matrix>, Vec<f64>, Option<String>);
fn relaxed_ionic_response(
    k: &Matrix,
    lambda: &Matrix,
    clamped: &Matrix,
    volume: f64,
    max_force: f64,
    opts: &AnalyticElasticOptions,
) -> Result<IonicRelaxation> {
    let n = k.rows() / 3;
    if n == 1 {
        return Ok((
            Some(clamped.clone()),
            Some(Matrix::zeros(3, 6)),
            Vec::new(),
            None,
        ));
    }
    let mut basis = Matrix::zeros(3 * n, 3 * (n - 1));
    for j in 0..n - 1 {
        let norm = (((j + 1) * (j + 2)) as f64).sqrt();
        for axis in 0..3 {
            for a in 0..=j {
                basis[(3 * a + axis, 3 * j + axis)] = 1. / norm;
            }
            basis[(3 * (j + 1) + axis, 3 * j + axis)] = -(j as f64 + 1.) / norm;
        }
    }
    let internal = basis.transpose().matmul(k)?.matmul(&basis)?;
    let eigen = crate::linalg::symmetric_eigen(&internal)?;
    if max_force > opts.force_tolerance {
        return Ok((
            None,
            None,
            eigen.values,
            Some(format!(
                "reference maximum gradient {max_force:e} exceeds stationary-atom tolerance"
            )),
        ));
    }
    if eigen.values[0] <= opts.stability_tolerance {
        return Ok((None,None,eigen.values.clone(),Some(format!("internal nuclear curvature {} is unstable or too soft for a static relaxed-ion response",eigen.values[0]))));
    }
    let rhs = basis.transpose().matmul(lambda)?;
    let solution = crate::linalg::DenseLu::factor(&internal)?.solve_matrix(&rhs)?;
    let mut du = basis.matmul(&solution)?;
    for v in du.as_mut_slice() {
        *v = -*v;
    }
    let correction = lambda.transpose().matmul(&du)?;
    let mut relaxed = clamped.clone();
    for x in 0..6 {
        for y in 0..6 {
            relaxed[(x, y)] += correction[(x, y)] / volume;
        }
    }
    Ok((Some(relaxed), Some(du), eigen.values, None))
}
