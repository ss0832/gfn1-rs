// SPDX-License-Identifier: GPL-3.0-or-later
//! Physical 2*nAO complex-spinor periodic SOC SCC. The overlap-dual operator
//! {S(k),O}/2 and its exact Pulay contribution use the same explicit zeta L.S
//! convention as molecular SOC. Noncollinear W and composite models are gated.
use super::complex::{CMatrix, NativeComplexOrthogonalizer};
use super::scf::{fock_at_k, mulliken_shell_charges, spinor_occupations};
use super::{PbcOptions, PbcSccResult};
use crate::coulomb::coulomb_energy_potential_from_matrix;
use crate::electronic::BroydenMixer;
use crate::math::Vec3;
use crate::soc::{onsite_soc, SocParameters, SpinorEigen};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};

pub struct PbcSocResult {
    pub calculation_id: Option<String>,
    pub total_internal: f64,
    pub total_free: f64,
    pub soc_energy: f64,
    pub atomic_charges: Vec<f64>,
    pub eigen_k: Vec<SpinorEigen>,
    pub occupations_k: Vec<Vec<f64>>,
    pub density_k: Vec<CMatrix>,
    pub kpoints: Vec<crate::model::KPoint>,
    pub fermi_level: f64,
    pub electronic_temperature: f64,
    pub iterations: usize,
    pub max_charge_residual: f64,
    pub method_citations: Vec<&'static str>,
    pub parameter_provenance: String,
    pub(crate) scalar_proxy: PbcSccResult,
}
impl PbcSocResult {
    /// Force evaluation reusing this converged spinor state. Geometry, active
    /// parameters and controls must match the reference, as for the scalar
    /// periodic `pbc_gradient_from_scc` API.
    pub fn analytic_gradient(
        &self,
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> Result<Vec<Vec3>> {
        Ok(
            super::pbc_gradient_from_scc(system, params, self.scalar_proxy.clone(), options, pbc)?
                .gradient,
        )
    }
    /// Fixed-fractional-coordinate stress of the same reference state.
    pub fn analytic_stress(
        &self,
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> Result<crate::linalg::Matrix> {
        let lattice = system.lattice.as_ref().ok_or_else(|| {
            Gfn1Error::InvalidInput("SOC stress requires a periodic lattice".into())
        })?;
        Ok(super::pbc_stress_from_scc(
            system,
            params,
            self.scalar_proxy.clone(),
            options,
            pbc,
            lattice,
        )?
        .stress)
    }
}

pub(super) fn lift(scalar: &CMatrix) -> CMatrix {
    let n = scalar.n;
    let mut spin = CMatrix::zeros(2 * n);
    for i in 0..n {
        for j in 0..n {
            for shift in [0, n] {
                spin.re[(shift + i, shift + j)] = scalar.re[(i, j)];
                spin.im[(shift + i, shift + j)] = scalar.im[(i, j)];
            }
        }
    }
    spin
}
pub(super) fn spin_trace(spin: &CMatrix) -> CMatrix {
    let n = spin.n / 2;
    let mut scalar = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            scalar.re[(i, j)] = spin.re[(i, j)] + spin.re[(n + i, n + j)];
            scalar.im[(i, j)] = spin.im[(i, j)] + spin.im[(n + i, n + j)];
        }
    }
    scalar
}
pub(super) fn anticommutator(a: &CMatrix, b: &CMatrix) -> Result<CMatrix> {
    let ab = a.matmul(b)?;
    let ba = b.matmul(a)?;
    let mut result = CMatrix::zeros(a.n);
    for i in 0..a.n {
        for j in 0..a.n {
            result.re[(i, j)] = 0.5 * (ab.re[(i, j)] + ba.re[(i, j)]);
            result.im[(i, j)] = 0.5 * (ab.im[(i, j)] + ba.im[(i, j)]);
        }
    }
    result.hermitianize();
    Ok(result)
}
pub fn run_pbc_soc_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    parameters: &SocParameters,
) -> Result<PbcSocResult> {
    let span = options.logger.as_ref().map(|l| l.begin("periodic_soc_spinor_scf", serde_json::json!({"atoms":system.atoms.len(),"kmesh":pbc.kmesh.size,"electronic_temperature_K":options.electronic_temperature}))).transpose()?;
    let mut result = run_impl(system, params, options, pbc, parameters);
    if let Ok(value) = &mut result {
        value.calculation_id = span.as_ref().map(|s| s.id().to_string());
        value.scalar_proxy.calculation_id = value.calculation_id.clone();
    }
    crate::logging::finish(
        span,
        result,
        |s| serde_json::json!({"iterations":s.iterations,"total_free_hartree":s.total_free,"soc_energy_hartree":s.soc_energy,"charge_residual_max":s.max_charge_residual,"physical_spinors_per_k":s.eigen_k.first().map(|e|e.values.len())}),
    )
}
fn run_impl(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    parameters: &SocParameters,
) -> Result<PbcSocResult> {
    if system.lattice.is_none()
        || options.spin_polarization
        || options.plus_u
        || options.multipole
        || options.lr_exchange
        || options.solvation.is_some()
        || options.ml_model.is_some()
        || options.experimental_d4
        || options.scf_trah
        || options.external_field.magnetic_field.is_some()
        || options.spin_multiplicity.is_some_and(|m| m != 1)
    {
        return Err(Gfn1Error::InvalidInput("periodic SOC supports spin-independent scalar GFN1, explicit zeta L.S and electric fields; composite/noncollinear W models are not implemented".into()));
    }
    if options.max_scc == 0
        || !options.electronic_temperature.is_finite()
        || options.electronic_temperature < 0.
        || !options.energy_tolerance.is_finite()
        || options.energy_tolerance <= 0.
        || !options.charge_tolerance.is_finite()
        || options.charge_tolerance <= 0.
        || !options.mixing.is_finite()
        || options.mixing <= 0.
        || options.mixing > 1.
    {
        return Err(Gfn1Error::InvalidInput(
            "invalid periodic spinor SCC settings".into(),
        ));
    }
    let mut reference = super::run_pbc_scc(system, params, options, pbc)?;
    let n = reference.basis.len();
    let onsite = onsite_soc(&reference.basis, parameters)?;
    let spin_overlaps: Vec<_> = reference.hs_k.iter().map(|(_, s)| lift(s)).collect();
    let operators: Vec<_> = spin_overlaps
        .iter()
        .map(|s| anticommutator(s, &onsite))
        .collect::<Result<_>>()?;
    let orthogonalizers: Vec<_> = spin_overlaps
        .iter()
        .map(|s| NativeComplexOrthogonalizer::build(s, options.eigen_tolerance))
        .collect::<Result<_>>()?;
    let weights: Vec<_> = reference.kpoints.iter().map(|k| k.weight).collect();
    let external =
        crate::field::electric_shell_potential(&options.external_field, system, &reference.basis);
    let mut q = reference.shell_charges.clone();
    let mut mixer = BroydenMixer::new(q.len(), options.scc_broyden_size.max(2), options.mixing);
    let mut previous = f64::INFINITY;
    let mut last_rms = f64::INFINITY;
    for iteration in 0..options.max_scc {
        let input = coulomb_energy_potential_from_matrix(
            &reference.basis,
            &reference.shell_model,
            &q,
            &reference.gamma,
        )?;
        let mut potential = input.shell_potential;
        if let Some(ext) = &external {
            for (v, e) in potential.iter_mut().zip(ext) {
                *v += e;
            }
        }
        let mut vao = vec![0.; n];
        for (s, shell) in reference.basis.shells.iter().enumerate() {
            for ao in shell.first_ao..shell.first_ao + shell.nao {
                vao[ao] = potential[s];
            }
        }
        let eigen: Vec<_> = reference
            .hs_k
            .iter()
            .enumerate()
            .map(|(k, (h0, s))| {
                let mut fock = lift(&fock_at_k(h0, s, &vao));
                for i in 0..2 * n {
                    for j in 0..2 * n {
                        fock.re[(i, j)] += operators[k].re[(i, j)];
                        fock.im[(i, j)] += operators[k].im[(i, j)];
                    }
                }
                let e = orthogonalizers[k].solve(&fock)?;
                Ok(SpinorEigen {
                    values: e.values,
                    vectors: e.vectors,
                })
            })
            .collect::<Result<_>>()?;
        let bands: Vec<_> = eigen.iter().map(|e| e.values.clone()).collect();
        let occupation = spinor_occupations(
            &bands,
            &weights,
            reference.nelec,
            options.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
        )?;
        let density: Vec<_> = eigen
            .iter()
            .zip(&occupation.occupations)
            .map(|(e, o)| crate::soc::density(e, o))
            .collect();
        let scalar: Vec<_> = density.iter().map(spin_trace).collect();
        let new_q = mulliken_shell_charges(&reference.basis, &reference.hs_k, &scalar, &weights);
        let mut coulomb = coulomb_energy_potential_from_matrix(
            &reference.basis,
            &reference.shell_model,
            &new_q,
            &reference.gamma,
        )?;
        let soc_energy: f64 = density
            .iter()
            .zip(&operators)
            .zip(&weights)
            .map(|((p, h), w)| w * p.real_trace_product(h))
            .sum();
        let band: f64 = scalar
            .iter()
            .zip(&reference.hs_k)
            .zip(&weights)
            .map(|((p, (h, _)), w)| w * p.real_trace_product(h))
            .sum();
        let field_energy = external
            .as_ref()
            .map_or(0., |v| crate::field::electric_field_energy(v, &new_q));
        let internal = band
            + soc_energy
            + coulomb.second_order
            + coulomb.third_order
            + coulomb.higher_order
            + reference.repulsion_energy
            + reference.dispersion_energy
            + reference.halogen_energy
            + field_energy;
        let free = internal + occupation.entropy;
        let residual: Vec<_> = new_q.iter().zip(&q).map(|(a, b)| a - b).collect();
        let maximum = residual.iter().map(|x| x.abs()).fold(0., f64::max);
        last_rms = (residual.iter().map(|x| x * x).sum::<f64>() / residual.len() as f64).sqrt();
        if !free.is_finite() || !last_rms.is_finite() {
            return Err(Gfn1Error::InvalidInput(
                "non-finite periodic SOC SCC state".into(),
            ));
        }
        if let Some(logger) = &options.logger {
            logger.event(crate::LogLevel::Info, "periodic_soc_scf_iteration", serde_json::json!({"iteration":iteration+1,"total_free_hartree":free,"charge_residual_max":maximum,"charge_rms":last_rms}))?;
        }
        if maximum < options.charge_tolerance && (free - previous).abs() < options.energy_tolerance
        {
            let mut weighted = Vec::with_capacity(eigen.len());
            for (k, e) in eigen.iter().enumerate() {
                let w: Vec<_> = e
                    .values
                    .iter()
                    .zip(&occupation.occupations[k])
                    .map(|(e, o)| e * o)
                    .collect();
                let mut effective = crate::soc::density(e, &w);
                let overlap_adjoint = anticommutator(&onsite, &density[k])?;
                for i in 0..2 * n {
                    for j in 0..2 * n {
                        effective.re[(i, j)] -= overlap_adjoint.re[(i, j)];
                        effective.im[(i, j)] -= overlap_adjoint.im[(i, j)];
                    }
                }
                weighted.push(spin_trace(&effective));
            }
            if let Some(ext) = &external {
                for (v, e) in coulomb.shell_potential.iter_mut().zip(ext) {
                    *v += e;
                }
            }
            reference.shell_charges = new_q;
            reference.atomic_charges = coulomb.atomic_charges;
            reference.shell_scc_potential = coulomb.shell_potential;
            reference.density_k = scalar;
            reference.ew_density_k = weighted;
            reference.isotropic_scc_energy = coulomb.second_order;
            reference.third_order_energy = coulomb.third_order;
            reference.higher_order_energy = coulomb.higher_order;
            reference.electronic_energy = band + soc_energy;
            reference.external_field_energy = field_energy;
            reference.electronic_entropy_term = occupation.entropy;
            reference.total_internal = internal;
            reference.total_free = free;
            reference.fermi_level = occupation.fermi_level;
            reference.electronic_temperature = options.electronic_temperature;
            reference.dipole = crate::field::mulliken_dipole(
                system,
                &reference.atomic_charges,
                options.external_field.origin,
            );
            reference.iterations = iteration + 1;
            reference.converged = true;
            reference.method_citations.push("Jha2023SOC");
            return Ok(PbcSocResult {
                calculation_id: None,
                total_internal: internal,
                total_free: free,
                soc_energy,
                atomic_charges: reference.atomic_charges.clone(),
                eigen_k: eigen,
                occupations_k: occupation.occupations,
                density_k: density,
                kpoints: reference.kpoints.clone(),
                fermi_level: occupation.fermi_level,
                electronic_temperature: options.electronic_temperature,
                iterations: iteration + 1,
                max_charge_residual: maximum,
                method_citations: reference.method_citations.clone(),
                parameter_provenance: parameters.provenance.clone(),
                scalar_proxy: reference,
            });
        }
        previous = free;
        q = if options.scc_broyden {
            mixer.next(&q, &residual)
        } else {
            None
        }
        .unwrap_or_else(|| {
            q.iter()
                .zip(&residual)
                .map(|(a, b)| a + options.mixing * b)
                .collect()
        });
    }
    Err(Gfn1Error::SccNotConverged {
        iterations: options.max_scc,
        rms: last_rms,
    })
}

pub fn pbc_soc_analytic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    parameters: &SocParameters,
) -> Result<(PbcSocResult, Vec<Vec3>)> {
    let result = run_pbc_soc_scc(system, params, options, pbc, parameters)?;
    let gradient = result.analytic_gradient(system, params, options, pbc)?;
    Ok((result, gradient))
}
pub fn pbc_soc_stress(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    parameters: &SocParameters,
) -> Result<(PbcSocResult, crate::linalg::Matrix)> {
    let result = run_pbc_soc_scc(system, params, options, pbc, parameters)?;
    let stress = result.analytic_stress(system, params, options, pbc)?;
    Ok((result, stress))
}

pub struct PbcSocHessianResult {
    pub soc: PbcSocResult,
    pub hessian: crate::linalg::Matrix,
    pub raw_max_asymmetry: f64,
    pub max_response_residual: f64,
}
pub fn pbc_soc_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    parameters: &SocParameters,
) -> Result<PbcSocHessianResult> {
    crate::resources::preflight_response(
        system,
        params,
        options,
        pbc.kmesh.size.into_iter().product(),
        true,
    )?;
    let span = options
        .logger
        .as_ref()
        .map(|l| {
            l.begin(
                "periodic_soc_dfpt_hessian",
                serde_json::json!({"atoms":system.atoms.len(),"kmesh":pbc.kmesh.size}),
            )
        })
        .transpose()?;
    let result = (|| {
        let soc = run_pbc_soc_scc(system, params, options, pbc, parameters)?;
        let response =
            super::soc_response::nuclear_responses(system, params, options, pbc, parameters, &soc)?;
        let hessian = super::hessian::hessian_from_responses(
            system,
            params,
            options,
            pbc,
            soc.scalar_proxy.clone(),
            (response.density, response.weighted, response.charges),
        )?;
        Ok(PbcSocHessianResult {
            soc,
            hessian: hessian.hessian,
            raw_max_asymmetry: hessian.raw_max_asymmetry,
            max_response_residual: response.max_residual,
        })
    })();
    crate::logging::finish(
        span,
        result,
        |h| serde_json::json!({"degrees_of_freedom":h.hessian.rows(),"raw_max_asymmetry":h.raw_max_asymmetry,"max_response_residual":h.max_response_residual}),
    )
}
