// SPDX-License-Identifier: GPL-3.0-or-later
//! Native physical-spinor CP-SCC, with one shell-space LU for all nuclear RHSs.
//! Nonorthogonal matrix-function derivatives use divided differences of f,
//! epsilon*f and epsilon^2*f. Fixed total electron count uses one GLOBAL dmu
//! over physical k weights; no real-embedding factors or per-k Fermi levels.
use super::complex::CMatrix;
use super::soc::{anticommutator, lift, spin_trace, PbcSocResult};
use crate::linalg::{DenseLu, Matrix};
use crate::soc::SpinorEigen;
use crate::soc::{onsite_soc, SocParameters};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};

struct SpectralKernel {
    f: Matrix,
    ef: Matrix,
    eef: Matrix,
    slopes: Vec<f64>,
    adjoint: CMatrix,
}
#[derive(Clone, Copy)]
struct SpectralReference<'a> {
    eigen_k: &'a [SpinorEigen],
    occupations_k: &'a [Vec<f64>],
    kpoints: &'a [crate::model::KPoint],
    scalar_proxy: &'a super::PbcSccResult,
    electronic_temperature: f64,
    spinors: bool,
    occupation_limit: f64,
}
impl SpectralReference<'_> {
    fn expand(&self, value: &CMatrix) -> CMatrix {
        if self.spinors {
            lift(value)
        } else {
            value.clone()
        }
    }
    fn fold(&self, value: &CMatrix) -> CMatrix {
        if self.spinors {
            spin_trace(value)
        } else {
            value.clone()
        }
    }
}
struct BareContext<'a> {
    result: SpectralReference<'a>,
    kernels: Vec<SpectralKernel>,
    slope_sum: f64,
}
struct Response {
    density: Vec<CMatrix>,
    weighted: Vec<CMatrix>,
    charges: Vec<f64>,
}
impl<'a> BareContext<'a> {
    fn build(result: SpectralReference<'a>) -> Result<Self> {
        let kt = result.electronic_temperature * crate::constants::KB_HARTREE_PER_K;
        if kt <= 0.
            && result
                .occupations_k
                .iter()
                .flatten()
                .any(|&f| f > 1e-10 && f < result.occupation_limit - 1e-10)
        {
            return Err(Gfn1Error::InvalidInput("zero-temperature periodic spinor nuclear response requires gapped integer occupations; use a finite-temperature ensemble".into()));
        }
        let mut kernels = Vec::new();
        let mut slope_sum = 0.;
        for (k, eig) in result.eigen_k.iter().enumerate() {
            let occ = &result.occupations_k[k];
            let n = eig.values.len();
            let slopes: Vec<_> = occ
                .iter()
                .map(|&f| {
                    if kt > 0. {
                        -f * (1. - f / result.occupation_limit) / kt
                    } else {
                        0.
                    }
                })
                .collect();
            slope_sum += result.kpoints[k].weight * slopes.iter().sum::<f64>();
            let mut f = Matrix::zeros(n, n);
            let mut ef = Matrix::zeros(n, n);
            let mut eef = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let gap = eig.values[i] - eig.values[j];
                    let df = occ[i] - occ[j];
                    let divided = if gap.abs() > 1e-10 {
                        df / gap
                    } else if kt > 0. {
                        0.5 * (slopes[i] + slopes[j])
                    } else if df.abs() < 1e-10 {
                        0.
                    } else {
                        return Err(Gfn1Error::InvalidInput(
                            "degenerate spinor response has unequal occupations".into(),
                        ));
                    };
                    let e = 0.5 * (eig.values[i] + eig.values[j]);
                    let mean_f = 0.5 * (occ[i] + occ[j]);
                    f[(i, j)] = divided;
                    ef[(i, j)] = mean_f + e * divided;
                    eef[(i, j)] = 2. * e * mean_f
                        + 0.5 * (eig.values[i].powi(2) + eig.values[j].powi(2)) * divided;
                }
            }
            kernels.push(SpectralKernel {
                f,
                ef,
                eef,
                slopes,
                adjoint: eig.vectors.adjoint(),
            });
        }
        Ok(Self {
            result,
            kernels,
            slope_sum,
        })
    }
    fn apply(&self, fock: &[CMatrix], overlap: &[CMatrix]) -> Result<Response> {
        let result = self.result;
        let reference = result.scalar_proxy;
        let mut transformed = Vec::with_capacity(fock.len());
        let mut numerator = 0.;
        for k in 0..fock.len() {
            let c = &result.eigen_k[k].vectors;
            let kernel = &self.kernels[k];
            let h = kernel.adjoint.matmul(&fock[k].matmul(c)?)?;
            let s = kernel
                .adjoint
                .matmul(&result.expand(&overlap[k]).matmul(c)?)?;
            for i in 0..c.n {
                numerator += result.kpoints[k].weight
                    * kernel.slopes[i]
                    * (h.re[(i, i)] - result.eigen_k[k].values[i] * s.re[(i, i)]);
            }
            transformed.push((h, s));
        }
        let dmu = if self.slope_sum.abs() > 1e-30 {
            numerator / self.slope_sum
        } else {
            0.
        };
        let mut density = Vec::new();
        let mut weighted = Vec::new();
        for (k, (h, s)) in transformed.iter().enumerate() {
            let c = &result.eigen_k[k].vectors;
            let kernel = &self.kernels[k];
            let n = c.n;
            let mut dp = CMatrix::zeros(n);
            let mut dw = CMatrix::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    dp.re[(i, j)] =
                        kernel.f[(i, j)] * h.re[(i, j)] - kernel.ef[(i, j)] * s.re[(i, j)];
                    dp.im[(i, j)] =
                        kernel.f[(i, j)] * h.im[(i, j)] - kernel.ef[(i, j)] * s.im[(i, j)];
                    dw.re[(i, j)] =
                        kernel.ef[(i, j)] * h.re[(i, j)] - kernel.eef[(i, j)] * s.re[(i, j)];
                    dw.im[(i, j)] =
                        kernel.ef[(i, j)] * h.im[(i, j)] - kernel.eef[(i, j)] * s.im[(i, j)];
                }
            }
            for i in 0..n {
                dp.re[(i, i)] -= kernel.slopes[i] * dmu;
                dw.re[(i, i)] -= result.eigen_k[k].values[i] * kernel.slopes[i] * dmu;
            }
            density.push(c.matmul(&dp)?.matmul(&kernel.adjoint)?);
            weighted.push(c.matmul(&dw)?.matmul(&kernel.adjoint)?);
        }
        let scalar: Vec<_> = density.iter().map(|p| result.fold(p)).collect();
        let weights: Vec<_> = result.kpoints.iter().map(|k| k.weight).collect();
        let mut charges = super::scf::mulliken_shell_charges(
            &reference.basis,
            &reference.hs_k,
            &scalar,
            &weights,
        );
        for (s, shell) in reference.basis.shells.iter().enumerate() {
            charges[s] -= shell.reference_occ;
            for k in 0..result.kpoints.len() {
                for mu in shell.first_ao..shell.first_ao + shell.nao {
                    for nu in 0..reference.basis.len() {
                        let p = &reference.density_k[k];
                        let ds = &overlap[k];
                        charges[s] -= weights[k]
                            * (p.re[(mu, nu)] * ds.re[(nu, mu)] - p.im[(mu, nu)] * ds.im[(nu, mu)]);
                    }
                }
            }
        }
        Ok(Response {
            density,
            weighted,
            charges,
        })
    }
}
fn charge_fock(result: SpectralReference<'_>, potential: &[f64]) -> Vec<CMatrix> {
    let scf = result.scalar_proxy;
    let n = scf.basis.len();
    let mut vao = vec![0.; n];
    for (s, shell) in scf.basis.shells.iter().enumerate() {
        for a in shell.first_ao..shell.first_ao + shell.nao {
            vao[a] = potential[s];
        }
    }
    let zero = CMatrix::zeros(n);
    scf.hs_k
        .iter()
        .map(|(_, s)| result.expand(&super::scf::fock_at_k(&zero, s, &vao)))
        .collect()
}
pub(crate) struct NuclearResponses {
    pub(crate) density: Vec<Vec<CMatrix>>,
    pub(crate) weighted: Vec<Vec<CMatrix>>,
    pub(crate) charges: Vec<Vec<f64>>,
    pub(crate) max_residual: f64,
}
pub(crate) fn nuclear_responses(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &super::PbcOptions,
    parameters: &SocParameters,
    result: &PbcSocResult,
) -> Result<NuclearResponses> {
    let state = SpectralReference {
        eigen_k: &result.eigen_k,
        occupations_k: &result.occupations_k,
        kpoints: &result.kpoints,
        scalar_proxy: &result.scalar_proxy,
        electronic_temperature: result.electronic_temperature,
        spinors: true,
        occupation_limit: 1.,
    };
    let onsite = onsite_soc(&result.scalar_proxy.basis, parameters)?;
    nuclear_impl(system, params, options, pbc, state, Some(&onsite))
}

/// Restricted max-two physical scalar bands. Shares the native matrix-function
/// and shell-space LU implementation with spinors, without doubling nAO or
/// forming a dense occupied/virtual-pair response operator.
pub(crate) fn scalar_nuclear_responses(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &super::PbcOptions,
    scf: &super::PbcSccResult,
) -> Result<NuclearResponses> {
    let _profile = crate::profile::scope("pbc.native_charge_space.scalar_responses");
    let mut vao = vec![0.; scf.basis.len()];
    for (s, shell) in scf.basis.shells.iter().enumerate() {
        for a in shell.first_ao..shell.first_ao + shell.nao {
            vao[a] = scf.shell_scc_potential[s];
        }
    }
    let eigen: Vec<_> = scf
        .hs_k
        .iter()
        .map(|(h, s)| {
            let e = super::complex::native_generalized_eigen(
                &super::scf::fock_at_k(h, s, &vao),
                s,
                options.eigen_tolerance,
            )?;
            Ok(SpinorEigen {
                values: e.values,
                vectors: e.vectors,
            })
        })
        .collect::<Result<_>>()?;
    let bands: Vec<_> = eigen.iter().map(|e| e.values.clone()).collect();
    let weights: Vec<_> = scf.kpoints.iter().map(|k| k.weight).collect();
    let occupations = super::scf::spinor_occupations(
        &bands,
        &weights,
        scf.nelec / 2.,
        scf.electronic_temperature * crate::constants::KB_HARTREE_PER_K,
    )?;
    let occupations: Vec<Vec<f64>> = occupations
        .occupations
        .into_iter()
        .map(|o| o.into_iter().map(|f| 2. * f).collect())
        .collect();
    let state = SpectralReference {
        eigen_k: &eigen,
        occupations_k: &occupations,
        kpoints: &scf.kpoints,
        scalar_proxy: scf,
        electronic_temperature: scf.electronic_temperature,
        spinors: false,
        occupation_limit: 2.,
    };
    nuclear_impl(system, params, options, pbc, state, None)
}

fn nuclear_impl(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &super::PbcOptions,
    result: SpectralReference<'_>,
    onsite: Option<&CMatrix>,
) -> Result<NuclearResponses> {
    let bare = BareContext::build(result)?;
    let scf = result.scalar_proxy;
    let ns = scf.basis.shells.len();
    let n = scf.basis.len();
    let nk = scf.kpoints.len();
    let zero_overlap: Vec<_> = (0..nk).map(|_| CMatrix::zeros(n)).collect();
    let mut susceptibility = Matrix::zeros(ns, ns);
    for s in 0..ns {
        let mut potential = vec![0.; ns];
        potential[s] = 1.;
        let response = bare.apply(&charge_fock(result, &potential), &zero_overlap)?;
        for a in 0..ns {
            susceptibility[(a, s)] = response.charges[a];
        }
    }
    let kernel = super::hessian::periodic_response_kernel(scf);
    let chi_kernel = susceptibility.matmul(&kernel)?;
    let mut matrix = Matrix::identity(ns);
    for a in 0..ns {
        for b in 0..ns {
            matrix[(a, b)] -= chi_kernel[(a, b)];
        }
    }
    let lu = DenseLu::factor(&matrix)?;
    let skeletons: Vec<_> = scf
        .kpoints
        .iter()
        .map(|k| {
            super::hessian::kpoint_skeleton_derivatives(
                system,
                params,
                scf,
                options,
                pbc,
                k.fractional,
            )
        })
        .collect::<Result<_>>()?;
    let mut output = NuclearResponses {
        density: Vec::new(),
        weighted: Vec::new(),
        charges: Vec::new(),
        max_residual: 0.,
    };
    for y in 0..3 * system.atoms.len() {
        let overlap: Vec<_> = skeletons.iter().map(|s| s.overlap[y].clone()).collect();
        let mut fock: Vec<_> = skeletons
            .iter()
            .map(|s| result.expand(&s.fock[y]))
            .collect();
        let physical_n = result.eigen_k[0].values.len();
        if let Some(onsite) = onsite {
            for k in 0..nk {
                let extra = anticommutator(&lift(&overlap[k]), onsite)?;
                for i in 0..physical_n {
                    for j in 0..physical_n {
                        fock[k].re[(i, j)] += extra.re[(i, j)];
                        fock[k].im[(i, j)] += extra.im[(i, j)];
                    }
                }
            }
        }
        let initial = bare.apply(&fock, &overlap)?;
        let charges = lu.solve_vec(&initial.charges)?;
        let potential = crate::linalg::matrix_vector_product(&kernel, &charges)?;
        let feedback = charge_fock(result, &potential);
        for k in 0..nk {
            for i in 0..physical_n {
                for j in 0..physical_n {
                    fock[k].re[(i, j)] += feedback[k].re[(i, j)];
                    fock[k].im[(i, j)] += feedback[k].im[(i, j)];
                }
            }
        }
        let total = bare.apply(&fock, &overlap)?;
        let norm = charges.iter().map(|v| v.abs()).fold(0., f64::max);
        let residual = total
            .charges
            .iter()
            .zip(&charges)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max)
            / (1. + norm);
        if !residual.is_finite()
            || residual > crate::resources::current_cp().residual_target(0., 1e-7, 1.)
        {
            return Err(Gfn1Error::InvalidInput(format!(
                "periodic spinor CP-SCC normalized residual {residual:e}"
            )));
        }
        output.max_residual = output.max_residual.max(residual);
        let mut weighted = Vec::with_capacity(nk);
        for k in 0..nk {
            let mut effective = total.weighted[k].clone();
            if let Some(onsite) = onsite {
                let extra = anticommutator(onsite, &total.density[k])?;
                for i in 0..physical_n {
                    for j in 0..physical_n {
                        effective.re[(i, j)] -= extra.re[(i, j)];
                        effective.im[(i, j)] -= extra.im[(i, j)];
                    }
                }
            }
            weighted.push(result.fold(&effective));
        }
        output
            .density
            .push(total.density.iter().map(|p| result.fold(p)).collect());
        output.weighted.push(weighted);
        output.charges.push(charges);
    }
    Ok(output)
}
