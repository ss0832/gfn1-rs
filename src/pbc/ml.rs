// SPDX-License-Identifier: GPL-3.0-or-later
//! Stationary structural-parameter chain for periodic scalar Hamiltonians.
use crate::data_tables::atomic_radius_bohr;
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::linalg::Matrix;
use crate::pbc::complex::CMatrix;
use crate::pbc::{kpoints::bloch_phase, PbcSccResult};
use crate::{Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};

/// dE/d(delta h_A,l), including all AO images and physical k-point weights.
/// The full ordered Bloch pair list already includes mirror terms, so the
/// symmetric 1/2 level prefactor is not doubled again. No division by H0 or S
/// is used: zero/canceling self energies remain well-defined.
pub fn level_sensitivities(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
) -> Result<Vec<[f64; 3]>> {
    sensitivities_with_density(system, params, scf, &scf.density_k)
}

pub(crate) fn sensitivities_with_density(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    densities: &[CMatrix],
) -> Result<Vec<[f64; 3]>> {
    let lattice = system.lattice.as_ref().ok_or_else(|| {
        Gfn1Error::InvalidInput("periodic ML sensitivity requires a lattice".into())
    })?;
    let mut sensitivity = vec![[0.; 3]; system.atoms.len()];
    for pair in &scf.bloch.pairs {
        let a = &scf.basis.shells[scf.basis.aos[pair.mu].shell_index];
        let b = &scf.basis.shells[scf.basis.aos[pair.nu].shell_index];
        let factor = if pair.offset.is_origin() && a.atom_index == b.atom_index {
            1.
        } else {
            let distance = (system.atoms[b.atom_index].position + lattice.translation(pair.offset)
                - system.atoms[a.atom_index].position)
                .norm();
            let radius = atomic_radius_bohr(a.z)? + atomic_radius_bohr(b.z)?;
            hscale(a, b, params)? * shell_polynomial(a, b, (distance / radius).sqrt())
        };
        let mut density = 0.;
        for (ik, kp) in scf.kpoints.iter().enumerate() {
            let (c, s) = bloch_phase(kp.fractional, pair.offset);
            let p = &densities[ik];
            density += kp.weight * (p.re[(pair.mu, pair.nu)] * c + p.im[(pair.mu, pair.nu)] * s);
        }
        let derivative = 0.5 * density * pair.s * factor;
        sensitivity[a.atom_index][a.angular.as_index()] += derivative;
        sensitivity[b.atom_index][b.angular.as_index()] += derivative;
    }
    Ok(sensitivity)
}

pub(crate) fn require_cubic_response(scf: &PbcSccResult) -> Result<()> {
    if scf.ml_parameters.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "periodic analytic ML FC3 requires third-order structural response, not yet implemented".into(),
        ));
    }
    Ok(())
}

/// Add H_theta theta_y to each fixed-density Fock skeleton. The basis and S
/// do not depend on the heads, so no extra overlap/kernel derivative is needed.
pub(crate) fn add_level_skeleton(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    fractional: [f64; 3],
    mut add: impl FnMut(usize, usize, usize, f64, f64),
) -> Result<()> {
    let Some(view) = &scf.ml_parameters else {
        return Ok(());
    };
    let jac = view.level_jacobian()?;
    let lattice = system.lattice.unwrap();
    for pair in &scf.bloch.pairs {
        let a = &scf.basis.shells[scf.basis.aos[pair.mu].shell_index];
        let b = &scf.basis.shells[scf.basis.aos[pair.nu].shell_index];
        let factor = if pair.offset.is_origin() && a.atom_index == b.atom_index {
            1.
        } else {
            let r = (system.atoms[b.atom_index].position + lattice.translation(pair.offset)
                - system.atoms[a.atom_index].position)
                .norm();
            hscale(a, b, params)?
                * shell_polynomial(
                    a,
                    b,
                    (r / (atomic_radius_bohr(a.z)? + atomic_radius_bohr(b.z)?)).sqrt(),
                )
        };
        let (c, s) = bloch_phase(fractional, pair.offset);
        for (y, row) in jac.iter().enumerate() {
            let value = 0.5
                * pair.s
                * factor
                * (row[a.atom_index][a.angular.as_index()]
                    + row[b.atom_index][b.angular.as_index()]);
            add(y, pair.mu, pair.nu, value * c, value * s);
        }
    }
    Ok(())
}

/// Partial nuclear derivative of dE/dtheta at fixed P and fixed theta. All
/// unscreened AO pairs are differentiated, including S=0/dS!=0 pairs. The
/// overlap-zero filtered Bloch list is sufficient for dE/dtheta itself, but
/// would reproduce the known upstream screening omission in this mixed term.
fn geometric_sensitivity(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    pbc: &crate::pbc::PbcOptions,
) -> Result<Vec<Vec<[f64; 3]>>> {
    let nat = system.atoms.len();
    let n = scf.basis.len();
    let mut out = vec![vec![[0.; 3]; nat]; 3 * nat];
    let mut aos = vec![Vec::new(); nat];
    for (mu, ao) in scf.basis.aos.iter().enumerate() {
        aos[ao.atom_index].push(mu);
    }
    let minimum = crate::basis::atom_min_exponents(&scf.basis, nat);
    for (offset, translation) in crate::model::image_translations(system, pbc.ao_cutoff) {
        let mut density = Matrix::zeros(n, n);
        for (ik, kp) in scf.kpoints.iter().enumerate() {
            let (c, s) = bloch_phase(kp.fractional, offset);
            for mu in 0..n {
                for nu in 0..n {
                    density[(mu, nu)] += kp.weight
                        * (c * scf.density_k[ik].re[(mu, nu)] + s * scf.density_k[ik].im[(mu, nu)]);
                }
            }
        }
        for a in 0..nat {
            for b in 0..nat {
                if a == b {
                    continue;
                }
                let ra = system.atoms[a].position;
                let rb = system.atoms[b].position + translation;
                let rvec = ra - rb;
                let r = rvec.norm();
                if r < 1e-12
                    || r > pbc.ao_cutoff
                    || crate::basis::overlap_screened(&minimum, a, b, r * r)
                {
                    continue;
                }
                let radius =
                    atomic_radius_bohr(system.atoms[a].z)? + atomic_radius_bohr(system.atoms[b].z)?;
                let rr = (r / radius).sqrt();
                for &mu in &aos[a] {
                    for &nu in &aos[b] {
                        let sa = &scf.basis.shells[scf.basis.aos[mu].shell_index];
                        let sb = &scf.basis.shells[scf.basis.aos[nu].shell_index];
                        let (moments, bra, ket) =
                            crate::integrals::contracted_pair_with_derivatives(
                                &scf.basis.aos[mu],
                                &scf.basis.aos[nu],
                                ra,
                                rb,
                            );
                        let hs = hscale(sa, sb, params)?;
                        let factor = hs * shell_polynomial(sa, sb, rr);
                        let pa = sa.poly_raw.unwrap_or(0.);
                        let pb = sb.poly_raw.unwrap_or(0.);
                        let df = rvec * (hs * (pa + pb + 2. * pa * pb * rr) * 0.5 * rr / (r * r));
                        let coefficient = 0.5 * density[(mu, nu)];
                        let ga = ((bra[0] * factor + df * moments[0]) * coefficient).to_array();
                        let gb = ((ket[0] * factor - df * moments[0]) * coefficient).to_array();
                        for (central, l) in [(a, sa.angular.as_index()), (b, sb.angular.as_index())]
                        {
                            for k in 0..3 {
                                out[3 * a + k][central][l] += ga[k];
                                out[3 * b + k][central][l] += gb[k];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub(crate) struct NuclearResponse {
    jacobian: Vec<Vec<[f64; 3]>>,
    frozen: Matrix,
}
impl NuclearResponse {
    pub(crate) fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        scf: &PbcSccResult,
        pbc: &crate::pbc::PbcOptions,
    ) -> Result<Option<Self>> {
        let Some(view) = &scf.ml_parameters else {
            return Ok(None);
        };
        let sensitivity = level_sensitivities(system, params, scf)?;
        let mut frozen = view.fixed_parameter_hessian(&sensitivity)?;
        let jacobian = view.level_jacobian()?;
        let geometric = geometric_sensitivity(system, params, scf, pbc)?;
        let d = jacobian.len();
        for x in 0..d {
            for y in 0..d {
                for i in 0..system.atoms.len() {
                    for l in 0..3 {
                        frozen[(x, y)] += jacobian[x][i][l] * geometric[y][i][l]
                            + jacobian[y][i][l] * geometric[x][i][l];
                    }
                }
            }
        }
        Ok(Some(Self { jacobian, frozen }))
    }
    pub(crate) fn add_frozen(&self, hessian: &mut Matrix) {
        for x in 0..hessian.rows() {
            for y in 0..hessian.cols() {
                hessian[(x, y)] += self.frozen[(x, y)];
            }
        }
    }
    pub(crate) fn response_column(&self, y: usize, sensitivity: &[[f64; 3]], hessian: &mut Matrix) {
        // The ordinary response gradient already contains dP/dW/dq from the
        // total Fock skeleton. Only theta_x * Tr[dP_y H_theta] is additional.
        for (x, row) in self.jacobian.iter().enumerate() {
            for (a, levels) in row.iter().enumerate() {
                for l in 0..3 {
                    hessian[(x, y)] += levels[l] * sensitivity[a][l];
                }
            }
        }
    }
}
