// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic derivatives of the discretized Berry phase, Im Tr(M^-1 dM).
//! Occupied-space gauge rotations cancel around closed strings. Structural ML
//! heads, SCC screening, basis-centre and reciprocal-boost chains are included.
use super::{
    pbc_berry_polarization_from_scc, perpendicular_axes, string_base, BerryPolarizationOptions,
    BerryPolarizationResult, POLARIZATION_AU_TO_C_PER_M2,
};
use crate::linalg::{DenseLu, Matrix};
use crate::math::Vec3;
use crate::pbc::complex::CMatrix;
use crate::pbc::electromechanical::{require_scalar, strain_action, strain_skeletons};
use crate::pbc::hessian::{
    kpoint_complex_mos, kpoint_cpxtb_from_skeletons, kpoint_skeleton_derivatives,
    periodic_response_kernel, KpointSkeleton,
};
use crate::pbc::{run_pbc_scc, PbcOptions, PbcSccResult};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};
use std::f64::consts::PI;

pub struct BerryPolarizationDerivatives {
    pub polarization: BerryPolarizationResult,
    /// [atom][polarization Cartesian][atomic displacement Cartesian], units e.
    pub born_effective_charges: Vec<Matrix>,
    /// Proper clamped-ion e[i,J], engineering Voigt xx yy zz yz xz xy, e/bohr².
    pub clamped_piezoelectric: Matrix,
    pub phase_derivatives: Matrix,
    pub born_acoustic_sum_residual: Matrix,
    pub electronic_temperature_kelvin: f64,
    pub ml_model_sha256: Option<String>,
}
impl BerryPolarizationDerivatives {
    pub fn as_json(&self) -> serde_json::Value {
        let rows = |m: &Matrix| {
            (0..m.rows())
                .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let si = (0..3)
            .map(|i| {
                (0..6)
                    .map(|j| self.clamped_piezoelectric[(i, j)] * POLARIZATION_AU_TO_C_PER_M2)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        serde_json::json!({"method":"analytic Berry-phase CP-SCC including structural ML",
            "polarization_au":self.polarization.polarization,"polarization_c_per_m2":self.polarization.polarization.map(|x|x*POLARIZATION_AU_TO_C_PER_M2),
            "born_effective_charges":self.born_effective_charges.iter().map(rows).collect::<Vec<_>>(),
            "clamped_piezoelectric_au":rows(&self.clamped_piezoelectric),"clamped_piezoelectric_c_per_m2":si,
            "phase_derivatives":rows(&self.phase_derivatives),"born_acoustic_sum_residual":rows(&self.born_acoustic_sum_residual),
            "volume_bohr3":self.polarization.volume,"electronic_temperature_kelvin":self.electronic_temperature_kelvin,
            "mesh":self.polarization.mesh,"minimum_band_gap_hartree":self.polarization.min_band_gap,
            "polarization_quantum_au":self.polarization.quantum,
            "ml_model_sha256":self.ml_model_sha256,"voigt_order":["xx","yy","zz","yz","xz","xy"],
            "ionic_response":"clamped ions; no relaxed-ion correction included"})
    }
}

fn add(a: &mut CMatrix, b: &CMatrix, scale: f64) {
    for i in 0..a.n {
        for j in 0..a.n {
            a.re[(i, j)] += scale * b.re[(i, j)];
            a.im[(i, j)] += scale * b.im[(i, j)];
        }
    }
}
fn occupied(c: &CMatrix, m: usize) -> CMatrix {
    let mut out = CMatrix::zeros(c.n);
    for i in 0..c.n {
        for j in 0..m {
            out.re[(i, j)] = c.re[(i, j)];
            out.im[(i, j)] = c.im[(i, j)];
        }
    }
    out
}
fn link(cnext: &CMatrix, b: &CMatrix, c: &CMatrix, m: usize) -> Result<CMatrix> {
    let full = cnext.adjoint().matmul(&b.matmul(c)?)?;
    let mut out = CMatrix::zeros(m);
    for i in 0..m {
        for j in 0..m {
            out.re[(i, j)] = full.re[(i, j)];
            out.im[(i, j)] = full.im[(i, j)];
        }
    }
    Ok(out)
}
fn link_lu(m: &CMatrix) -> Result<DenseLu> {
    let mut real = Matrix::zeros(2 * m.n, 2 * m.n);
    for i in 0..m.n {
        for j in 0..m.n {
            real[(i, j)] = m.re[(i, j)];
            real[(i + m.n, j + m.n)] = m.re[(i, j)];
            real[(i + m.n, j)] = m.im[(i, j)];
            real[(i, j + m.n)] = -m.im[(i, j)];
        }
    }
    DenseLu::factor(&real)
}
fn phase_differential(lu: &DenseLu, dm: &CMatrix) -> Result<f64> {
    let mut rhs = Matrix::zeros(2 * dm.n, dm.n);
    for i in 0..dm.n {
        for j in 0..dm.n {
            rhs[(i, j)] = dm.re[(i, j)];
            rhs[(i + dm.n, j)] = dm.im[(i, j)];
        }
    }
    let solution = lu.solve_matrix(&rhs)?;
    Ok((0..dm.n).map(|i| solution[(i + dm.n, i)]).sum())
}

fn orbital_derivatives(
    scf: &PbcSccResult,
    coefficients: &[CMatrix],
    energies: &[Vec<f64>],
    skeletons: &[KpointSkeleton],
    dq: &[Vec<f64>],
    no: usize,
) -> Result<Vec<Vec<CMatrix>>> {
    let n = scf.basis.len();
    let nd = dq.len();
    let nk = scf.kpoints.len();
    let kernel = periodic_response_kernel(scf);
    let potentials = dq
        .iter()
        .map(|q| crate::linalg::matrix_vector_product(&kernel, q))
        .collect::<Result<Vec<_>>>()?;
    let mut dc = Vec::with_capacity(nk);
    for k in 0..nk {
        let c = &coefficients[k];
        let adj = c.adjoint();
        let sk = &scf.hs_k[k].1;
        let mut batch = Vec::with_capacity(nd);
        for (y, potential) in potentials.iter().enumerate() {
            let mut f = skeletons[k].fock[y].clone();
            for i in 0..n {
                for j in 0..n {
                    let v = -0.5
                        * (potential[scf.basis.aos[i].shell_index]
                            + potential[scf.basis.aos[j].shell_index]);
                    f.accumulate(i, j, v * sk.re[(i, j)], v * sk.im[(i, j)]);
                }
            }
            let fm = adj.matmul(&f.matmul(c)?)?;
            let sm = adj.matmul(&skeletons[k].overlap[y].matmul(c)?)?;
            let mut u = CMatrix::zeros(n);
            for i in 0..no {
                for a in 0..n {
                    if a < no {
                        u.re[(a, i)] = -0.5 * sm.re[(a, i)];
                        u.im[(a, i)] = -0.5 * sm.im[(a, i)];
                    } else {
                        let gap = energies[k][i] - energies[k][a];
                        u.re[(a, i)] = (fm.re[(a, i)] - energies[k][i] * sm.re[(a, i)]) / gap;
                        u.im[(a, i)] = (fm.im[(a, i)] - energies[k][i] * sm.im[(a, i)]) / gap;
                    }
                }
            }
            batch.push(c.matmul(&u)?);
        }
        dc.push(batch);
    }
    Ok(dc)
}

/// Nuclear Born charges and proper clamped piezoelectric tensor. The complete
/// Γ-centred Berry mesh is also the SCC/CP mesh, so no sampling changes occur.
pub fn pbc_polarization_derivatives(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
) -> Result<BerryPolarizationDerivatives> {
    require_scalar(system, options)?;
    validate_mesh(pbc, berry)?;
    let mut electronic = options.clone();
    electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
    electronic.charge_tolerance = electronic.charge_tolerance.min(1e-10);
    let scf = run_pbc_scc(system, params, &electronic, pbc)?;
    pbc_polarization_derivatives_from_scc(system, params, &electronic, pbc, berry, &scf)
}

pub(crate) fn validate_mesh(
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
) -> Result<([usize; 3], usize)> {
    let mesh = berry.mesh;
    let nk = mesh
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| Gfn1Error::InvalidInput("Berry mesh size overflow".into()))?;
    if mesh.contains(&0)
        || mesh != pbc.kmesh.size
        || !berry.directions.iter().all(|x| *x)
        || (nk > 1 && (!pbc.kmesh.gamma_centered || pbc.kmesh.fold_time_reversal))
    {
        return Err(Gfn1Error::InvalidInput("analytic Berry response requires all directions and the same complete Gamma-centred, unfolded SCC/Berry mesh".into()));
    }
    crate::resources::check_dense_workspace(nk, 1, 128, "Berry mesh")?;
    Ok((mesh, nk))
}

pub(crate) fn pbc_polarization_derivatives_from_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
    scf: &PbcSccResult,
) -> Result<BerryPolarizationDerivatives> {
    let (mesh, nk) = validate_mesh(pbc, berry)?;
    let primal = pbc_berry_polarization_from_scc(system, scf, options, pbc, berry)?;
    let n = scf.basis.len();
    let nat = system.atoms.len();
    let nd = 3 * nat + 6;
    let no = primal.occupied_bands;
    crate::resources::check_dense_workspace(
        n,
        n,
        nd.saturating_mul(nk).saturating_mul(10),
        "analytic Berry response",
    )?;
    let strain = strain_skeletons(system, params, scf, options, pbc)?;
    let mut skeletons = Vec::with_capacity(nk);
    let mut coefficients = Vec::with_capacity(nk);
    let mut energies = Vec::with_capacity(nk);
    for (k, ss) in scf.kpoints.iter().zip(strain) {
        let mut s = kpoint_skeleton_derivatives(system, params, scf, options, pbc, k.fractional)?;
        s.overlap.extend(ss.overlap);
        s.fock.extend(ss.fock);
        s.shell_potential.extend(ss.shell_potential);
        s.coordination.extend(ss.coordination);
        skeletons.push(s);
        let mos = kpoint_complex_mos(scf, coefficients.len())?;
        coefficients.push(mos.coeff);
        energies.push(mos.energies);
    }
    let (_, _, dq) = kpoint_cpxtb_from_skeletons(scf, &skeletons, true)?;
    let dc = orbital_derivatives(scf, &coefficients, &energies, &skeletons, &dq, no)?;
    let co: Vec<_> = coefficients.iter().map(|c| occupied(c, no)).collect();
    let grid_index = |frac: [f64; 3]| -> usize {
        let ix: [usize; 3] = std::array::from_fn(|d| {
            ((frac[d].rem_euclid(1.) * mesh[d] as f64).round() as usize) % mesh[d]
        });
        (ix[0] * mesh[1] + ix[1]) * mesh[2] + ix[2]
    };
    let mut lookup = vec![usize::MAX; nk];
    for (i, k) in scf.kpoints.iter().enumerate() {
        let index = grid_index(k.fractional);
        if lookup[index] != usize::MAX {
            return Err(Gfn1Error::InvalidInput(
                "Berry mesh contains duplicate points".into(),
            ));
        }
        lookup[index] = i;
    }
    if lookup.contains(&usize::MAX) {
        return Err(Gfn1Error::InvalidInput(
            "incomplete Berry response mesh".into(),
        ));
    }
    let lattice = system.lattice.unwrap();
    let recip = lattice.reciprocal_vectors_2pi();
    let mut phase = Matrix::zeros(3, nd);
    for d in 0..3 {
        let q = recip[d] / mesh[d] as f64;
        let blocks = derivative_blocks(system, &scf.basis, pbc.ao_cutoff, q);
        let (e1, e2) = perpendicular_axes(d);
        let strings = mesh[e1] * mesh[e2];
        for string in 0..strings {
            let base = string_base(d, string, mesh);
            for j in 0..mesh[d] {
                let mut frac = base;
                frac[d] += j as f64 / mesh[d] as f64;
                let mut next = frac;
                next[d] = ((j + 1) % mesh[d]) as f64 / mesh[d] as f64;
                let k = lookup[grid_index(frac)];
                let next = lookup[grid_index(next)];
                let (b, db) = boost_derivatives_at_k(&blocks, n, nat, frac, q);
                let m = link(&co[next], &b, &co[k], no)?;
                let lu = link_lu(&m)?;
                for y in 0..nd {
                    let mut dm = link(&dc[next][y], &b, &co[k], no)?;
                    add(&mut dm, &link(&co[next], &db[y], &co[k], no)?, 1.);
                    add(&mut dm, &link(&co[next], &b, &dc[k][y], no)?, 1.);
                    phase[(d, y)] -= 2. * phase_differential(&lu, &dm)? / strings as f64;
                }
            }
        }
        for a in 0..nat {
            for axis in 0..3 {
                let mut unit = [0.; 3];
                unit[axis] = 1.;
                let v = lattice.frac_of(Vec3::new(unit[0], unit[1], unit[2]));
                phase[(d, 3 * a + axis)] +=
                    2. * PI * scf.basis.reference_electrons[a] * v.to_array()[d];
            }
        }
    }
    let mut born = vec![Matrix::zeros(3, 3); nat];
    let mut piezo = Matrix::zeros(3, 6);
    let mut sum = Matrix::zeros(3, 3);
    for i in 0..3 {
        for y in 0..nd {
            let derivative = (0..3)
                .map(|d| lattice.cell.col[d].to_array()[i] * phase[(d, y)] / (2. * PI))
                .sum::<f64>();
            if y < 3 * nat {
                born[y / 3][(i, y % 3)] = derivative;
                sum[(i, y % 3)] += derivative;
            } else {
                piezo[(i, y - 3 * nat)] = derivative / lattice.volume();
            }
        }
    }
    Ok(BerryPolarizationDerivatives {
        polarization: primal,
        born_effective_charges: born,
        clamped_piezoelectric: piezo,
        phase_derivatives: phase,
        born_acoustic_sum_residual: sum,
        electronic_temperature_kelvin: scf.electronic_temperature,
        ml_model_sha256: scf.ml_parameters.as_ref().map(|m| m.model_sha256.clone()),
    })
}

pub struct ElectronicDielectric {
    pub relative_dielectric: Matrix,
    pub susceptibility: Matrix,
    pub raw_asymmetry: f64,
    pub volume_bohr3: f64,
    pub mesh: [usize; 3],
    pub electronic_temperature_kelvin: f64,
    pub ml_model_sha256: Option<String>,
    pub minimum_susceptibility_eigenvalue: f64,
}
impl ElectronicDielectric {
    pub fn as_json(&self) -> serde_json::Value {
        let rows = |m: &Matrix| {
            (0..m.rows())
                .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        serde_json::json!({"electronic_relative_dielectric":rows(&self.relative_dielectric),
            "susceptibility_au":rows(&self.susceptibility),"raw_asymmetry":self.raw_asymmetry,
            "minimum_susceptibility_eigenvalue":self.minimum_susceptibility_eigenvalue,
            "volume_bohr3":self.volume_bohr3,"mesh":self.mesh,"electronic_temperature_kelvin":self.electronic_temperature_kelvin,
            "ml_model_sha256":self.ml_model_sha256,"field_coupling":"variational discretized Berry electric enthalpy F-Omega*E.P",
            "ionic_response":"clamped ions"})
    }
}
fn padded_inverse(lu: &DenseLu, m: usize, n: usize) -> Result<CMatrix> {
    let mut rhs = Matrix::zeros(2 * m, m);
    for i in 0..m {
        rhs[(i, i)] = 1.;
    }
    let inverse = lu.solve_matrix(&rhs)?;
    let mut out = CMatrix::zeros(n);
    for i in 0..m {
        for j in 0..m {
            out.re[(i, j)] = inverse[(i, j)];
            out.im[(i, j)] = inverse[(i + m, j)];
        }
    }
    Ok(out)
}
fn add_i(a: &mut CMatrix, b: &CMatrix, scale: f64) {
    for i in 0..a.n {
        for j in 0..a.n {
            a.re[(i, j)] -= scale * b.im[(i, j)];
            a.im[(i, j)] += scale * b.re[(i, j)];
        }
    }
}

/// Analytic clamped-ion susceptibility of F-Ω E.P_Berry. The driving operator
/// is the exact orbital derivative of the same discrete Berry functional, not
/// the reference-cell Mulliken dipole. The shell-space CP kernel includes ML.
pub fn pbc_electronic_dielectric(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
) -> Result<ElectronicDielectric> {
    require_scalar(system, options)?;
    validate_mesh(pbc, berry)?;
    let mut electronic = options.clone();
    electronic.energy_tolerance = electronic.energy_tolerance.min(1e-12);
    electronic.charge_tolerance = electronic.charge_tolerance.min(1e-10);
    let scf = run_pbc_scc(system, params, &electronic, pbc)?;
    pbc_electronic_dielectric_from_scc(system, &electronic, pbc, berry, &scf)
}

pub(crate) fn pbc_electronic_dielectric_from_scc(
    system: &PeriodicSystem,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    berry: &BerryPolarizationOptions,
    scf: &PbcSccResult,
) -> Result<ElectronicDielectric> {
    let (mesh, nk) = validate_mesh(pbc, berry)?;
    let primal = pbc_berry_polarization_from_scc(system, scf, options, pbc, berry)?;
    let n = scf.basis.len();
    let no = primal.occupied_bands;
    let ns = scf.basis.shells.len();
    crate::resources::check_dense_workspace(n, n, 40 * nk, "Berry electronic dielectric")?;
    let mut coefficients = Vec::with_capacity(nk);
    let mut energies = Vec::with_capacity(nk);
    for k in 0..nk {
        let mo = kpoint_complex_mos(scf, k)?;
        coefficients.push(mo.coeff);
        energies.push(mo.energies);
    }
    let co: Vec<_> = coefficients.iter().map(|c| occupied(c, no)).collect();
    let index = |frac: [f64; 3]| {
        let a: [usize; 3] = std::array::from_fn(|d| {
            ((frac[d].rem_euclid(1.) * mesh[d] as f64).round() as usize) % mesh[d]
        });
        (a[0] * mesh[1] + a[1]) * mesh[2] + a[2]
    };
    let mut lookup = vec![usize::MAX; nk];
    for (i, k) in scf.kpoints.iter().enumerate() {
        lookup[index(k.fractional)] = i;
    }
    if lookup.contains(&usize::MAX) {
        return Err(Gfn1Error::InvalidInput(
            "incomplete Berry electric mesh".into(),
        ));
    }
    let lattice = system.lattice.unwrap();
    let recip = lattice.reciprocal_vectors_2pi();
    let mut gradients = vec![vec![CMatrix::zeros(n); nk]; 3];
    for d in 0..3 {
        let q = recip[d] / mesh[d] as f64;
        let blocks = super::boosted_image_blocks(system, &scf.basis, pbc.ao_cutoff, q);
        let (a, b) = perpendicular_axes(d);
        let strings = mesh[a] * mesh[b];
        for string in 0..strings {
            let base = string_base(d, string, mesh);
            for j in 0..mesh[d] {
                let mut frac = base;
                frac[d] = j as f64 / mesh[d] as f64;
                let mut next = base;
                next[d] = ((j + 1) % mesh[d]) as f64 / mesh[d] as f64;
                let k = lookup[index(frac)];
                let l = lookup[index(next)];
                let b = super::boost_at_k(&blocks, n, frac);
                let m = link(&co[l], &b, &co[k], no)?;
                let inverse = padded_inverse(&link_lu(&m)?, no, n)?;
                let ket = b.adjoint().matmul(&co[l].matmul(&inverse.adjoint())?)?;
                let bra = b.matmul(&co[k].matmul(&inverse)?)?;
                for axis in 0..3 {
                    let scale =
                        2. * lattice.cell.col[d].to_array()[axis] / (2. * PI * strings as f64);
                    add_i(&mut gradients[axis][k], &ket, scale);
                    add_i(&mut gradients[axis][l], &bra, -scale);
                }
            }
        }
    }
    let mut skeletons = Vec::with_capacity(nk);
    for k in 0..nk {
        let sco = scf.hs_k[k].1.matmul(&co[k])?;
        let mut fock = Vec::with_capacity(3);
        for gradient in &gradients {
            let mut b = gradient[k].clone();
            let factor = 1. / (4. * scf.kpoints[k].weight);
            for i in 0..n {
                for j in 0..n {
                    b.re[(i, j)] *= factor;
                    b.im[(i, j)] *= factor;
                }
            }
            let occ = co[k].adjoint().matmul(&b)?;
            add(&mut b, &sco.matmul(&occ)?, -1.);
            let mut field = b.matmul(&sco.adjoint())?;
            add(&mut field, &sco.matmul(&b.adjoint())?, 1.);
            fock.push(field);
        }
        skeletons.push(KpointSkeleton {
            overlap: vec![CMatrix::zeros(n); 3],
            fock,
            shell_potential: vec![vec![0.; ns]; 3],
            coordination: vec![vec![0.; system.atoms.len()]; 3],
        });
    }
    let (_, _, dq) = kpoint_cpxtb_from_skeletons(scf, &skeletons, true)?;
    let dc = orbital_derivatives(scf, &coefficients, &energies, &skeletons, &dq, no)?;
    let mut susceptibility = Matrix::zeros(3, 3);
    for d in 0..3 {
        let q = recip[d] / mesh[d] as f64;
        let blocks = super::boosted_image_blocks(system, &scf.basis, pbc.ao_cutoff, q);
        let (a, b) = perpendicular_axes(d);
        let strings = mesh[a] * mesh[b];
        for string in 0..strings {
            let base = string_base(d, string, mesh);
            for j in 0..mesh[d] {
                let mut frac = base;
                frac[d] = j as f64 / mesh[d] as f64;
                let mut next = base;
                next[d] = ((j + 1) % mesh[d]) as f64 / mesh[d] as f64;
                let k = lookup[index(frac)];
                let l = lookup[index(next)];
                let b = super::boost_at_k(&blocks, n, frac);
                let lu = link_lu(&link(&co[l], &b, &co[k], no)?)?;
                for axis in 0..3 {
                    let mut dm = link(&dc[l][axis], &b, &co[k], no)?;
                    add(&mut dm, &link(&co[l], &b, &dc[k][axis], no)?, 1.);
                    let dphi = phase_differential(&lu, &dm)?;
                    for i in 0..3 {
                        susceptibility[(i, axis)] -= 2. * lattice.cell.col[d].to_array()[i] * dphi
                            / (2. * PI * lattice.volume() * strings as f64);
                    }
                }
            }
        }
    }
    let mut eps = Matrix::identity(3);
    let mut asym = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            eps[(i, j)] += 4. * PI * susceptibility[(i, j)];
            asym = asym.max((susceptibility[(i, j)] - susceptibility[(j, i)]).abs());
        }
    }
    let minimum = crate::linalg::symmetric_eigen(&susceptibility)?.values[0];
    Ok(ElectronicDielectric {
        relative_dielectric: eps,
        susceptibility,
        raw_asymmetry: asym,
        volume_bohr3: lattice.volume(),
        mesh,
        electronic_temperature_kelvin: scf.electronic_temperature,
        ml_model_sha256: scf.ml_parameters.as_ref().map(|m| m.model_sha256.clone()),
        minimum_susceptibility_eigenvalue: minimum,
    })
}

struct BoostDerivative {
    mu: usize,
    nu: usize,
    a: usize,
    b: usize,
    offset: crate::lattice::ImageOffset,
    ca: Vec3,
    cb: Vec3,
    value: (f64, f64),
    da: [(f64, f64); 3],
    db: [(f64, f64); 3],
    dq: [(f64, f64); 3],
}
fn derivative_blocks(
    system: &PeriodicSystem,
    basis: &crate::BasisSet,
    cutoff: f64,
    q: Vec3,
) -> Vec<BoostDerivative> {
    let lattice = system.lattice.unwrap();
    let mut out = Vec::new();
    let exponents = crate::basis::atom_min_exponents(basis, system.atoms.len());
    for offset in lattice.image_offsets(cutoff) {
        let t = lattice.translation(offset);
        for (mu, a) in basis.aos.iter().enumerate() {
            for (nu, b) in basis.aos.iter().enumerate() {
                let ca = system.atoms[a.atom_index].position;
                let cb = system.atoms[b.atom_index].position + t;
                let r2 = (ca - cb).norm2();
                if r2 > cutoff * cutoff
                    || crate::basis::overlap_screened(&exponents, a.atom_index, b.atom_index, r2)
                {
                    continue;
                }
                let (value, da, db, dq) =
                    crate::magnetic::boosted_overlap_pair_derivatives(a, b, ca, cb, q);
                out.push(BoostDerivative {
                    mu,
                    nu,
                    a: a.atom_index,
                    b: b.atom_index,
                    offset,
                    ca,
                    cb,
                    value,
                    da,
                    db,
                    dq,
                });
            }
        }
    }
    out
}
fn boost_derivatives_at_k(
    blocks: &[BoostDerivative],
    n: usize,
    nat: usize,
    frac: [f64; 3],
    q: Vec3,
) -> (CMatrix, Vec<CMatrix>) {
    let mut b = CMatrix::zeros(n);
    let mut db = vec![CMatrix::zeros(n); 3 * nat + 6];
    for p in blocks {
        let (c, s) = crate::pbc::kpoints::bloch_phase(frac, p.offset);
        let add = |matrix: &mut CMatrix, value: (f64, f64)| {
            matrix.accumulate(
                p.mu,
                p.nu,
                c * value.0 - s * value.1,
                s * value.0 + c * value.1,
            )
        };
        add(&mut b, p.value);
        for axis in 0..3 {
            add(&mut db[3 * p.a + axis], p.da[axis]);
            add(&mut db[3 * p.b + axis], p.db[axis]);
        }
        for j in 0..6 {
            let a = strain_action(j, p.ca).to_array();
            let b = strain_action(j, p.cb).to_array();
            let q = (-strain_action(j, q)).to_array();
            let mut value = (0., 0.);
            for axis in 0..3 {
                value.0 += a[axis] * p.da[axis].0 + b[axis] * p.db[axis].0 + q[axis] * p.dq[axis].0;
                value.1 += a[axis] * p.da[axis].1 + b[axis] * p.db[axis].1 + q[axis] * p.dq[axis].1;
            }
            add(&mut db[3 * nat + j], value);
        }
    }
    (b, db)
}
