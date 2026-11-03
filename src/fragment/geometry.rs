// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared integral and Ewald geometry, without a supersystem SCF.
use crate::basis::{BasisOptions, BasisSet};
use crate::coordination::{coordination_with_derivatives, CoordinationOptions};
use crate::coulomb::{effective_coulomb_matrix, harmonic_average, ShellChargeModel};
use crate::data_tables::atomic_radius_bohr;
use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::hamiltonian::{hscale, shell_polynomial};
use crate::integrals::contracted_pair_with_derivatives;
use crate::linalg::Matrix;
use crate::math::{erfc, Vec3};
use crate::params::Gfn1Parameters;
use crate::pbc::ewald::{
    exp1, periodic_gamma_matrix, qcore_r3_k0_log, qcore_r3_real_value_derivatives,
    qcore_short_value_derivatives, resolve_alpha, QCORE_R3_COEFF, TAU,
};
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use std::f64::consts::PI;
const GEOMETRY_CACHE_BYTES: usize = 536870912;

struct Pair {
    mu: usize,
    nu: usize,
    a: usize,
    b: usize,
    r: Vec3,
    ds: Vec3,
    dh: Vec3,
    cn_a: f64,
    cn_b: f64,
}
struct KernelPair {
    i: usize,
    j: usize,
    a: usize,
    b: usize,
    gradient: Vec3,
    strain: [f64; 9],
}
struct RepPair {
    a: usize,
    b: usize,
    energy: f64,
    gradient: Vec3,
    r: Vec3,
}

pub(super) struct Geometry {
    pub basis: BasisSet,
    pub model: ShellChargeModel,
    pub h: Matrix,
    pub s: Matrix,
    pub gamma: Matrix,
    pub classical: f64,
    pub repulsion: f64,
    pub classical_derivatives: Vec<f64>,
    pub ndof: usize,
    pub volume: Option<f64>,
    pairs: Vec<Pair>,
    kernel: Vec<KernelPair>,
    cn_derivatives: Vec<Vec<f64>>,
    rep_pairs: Vec<RepPair>,
}

pub(super) struct Direction {
    pub h: Matrix,
    pub s: Matrix,
    pub gamma: Matrix,
    pub classical: f64,
}

impl Geometry {
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        pbc: &PbcOptions,
    ) -> Result<Self> {
        let nat = system.atoms.len();
        let ndof = 3 * nat + if system.lattice.is_some() { 9 } else { 0 };
        let basis = BasisSet::build(
            system,
            params,
            BasisOptions {
                nprim: options.nprim,
            },
        )?;
        let model = ShellChargeModel::build(system, &basis, params)?;
        let cn = coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: options.hamiltonian.coordination_cutoff,
                ..Default::default()
            },
        )?;
        let mut cn_derivatives = vec![vec![0.; nat]; ndof];
        if options.hamiltonian.enable_cn_hamiltonian {
            for pair in &cn.pairs {
                let r = pair.r_ij;
                let g = r * (pair.dcn_dr / r.norm());
                for axis in 0..3 {
                    let d = g.to_array()[axis];
                    cn_derivatives[3 * pair.i + axis][pair.i] += d;
                    cn_derivatives[3 * pair.j + axis][pair.i] -= d;
                    if pair.i != pair.j {
                        cn_derivatives[3 * pair.i + axis][pair.j] += d;
                        cn_derivatives[3 * pair.j + axis][pair.j] -= d;
                    }
                }
                if system.lattice.is_some() {
                    for a in 0..3 {
                        for b in 0..3 {
                            let d = pair.dcn_dr * r.to_array()[a] * r.to_array()[b] / r.norm();
                            cn_derivatives[3 * nat + 3 * a + b][pair.i] += d;
                            if pair.i != pair.j {
                                cn_derivatives[3 * nat + 3 * a + b][pair.j] += d;
                            }
                        }
                    }
                }
            }
        }
        let n = basis.len();
        if n.checked_mul(n)
            .and_then(|x| x.checked_mul(96))
            .is_none_or(|bytes| bytes > GEOMETRY_CACHE_BYTES / 2)
        {
            return Err(Gfn1Error::InvalidInput(
                "fragment reference matrices exceed the geometry cache budget".into(),
            ));
        }
        let mut h = Matrix::zeros(n, n);
        let mut s = Matrix::zeros(n, n);
        let cutoff = if system.lattice.is_some() {
            pbc.ao_cutoff
        } else {
            options.hamiltonian.integral_cutoff
        };
        let se: Vec<_> = basis
            .shells
            .iter()
            .map(|sh| {
                sh.hdiag_ha
                    - if options.hamiltonian.enable_cn_hamiltonian {
                        sh.kcn_raw.unwrap_or(0.) * cn.cn[sh.atom_index]
                    } else {
                        0.
                    }
            })
            .collect();
        let mut pairs = Vec::new();
        for (offset, translation) in crate::model::image_translations(system, cutoff) {
            for mu in 0..n {
                for nu in 0..n {
                    let ai = &basis.aos[mu];
                    let aj = &basis.aos[nu];
                    let a = ai.atom_index;
                    let b = aj.atom_index;
                    let r = system.atoms[a].position - system.atoms[b].position - translation;
                    let distance = r.norm();
                    if cutoff > 0. && distance > cutoff {
                        continue;
                    }
                    let si = &basis.shells[ai.shell_index];
                    let sj = &basis.shells[aj.shell_index];
                    let onsite = a == b && offset.is_origin();
                    let (moments, ds_bra, _) = contracted_pair_with_derivatives(
                        ai,
                        aj,
                        system.atoms[a].position,
                        system.atoms[b].position + translation,
                    );
                    let overlap = moments[0];
                    let ds = ds_bra[0];
                    if overlap == 0. && ds.norm2() == 0. {
                        continue;
                    }
                    let (scale, radial) = if onsite {
                        (1., 0.)
                    } else {
                        let rr = (distance
                            / (atomic_radius_bohr(si.z)? + atomic_radius_bohr(sj.z)?))
                        .sqrt();
                        let k = hscale(si, sj, params)?;
                        let pi = si.poly_raw.unwrap_or(0.);
                        let pj = sj.poly_raw.unwrap_or(0.);
                        (
                            k * shell_polynomial(si, sj, rr),
                            k * (pi + pj + 2. * pi * pj * rr) * rr / (2. * distance),
                        )
                    };
                    let average = 0.5 * (se[ai.shell_index] + se[aj.shell_index]);
                    let pref = average * scale;
                    let dh = ds * pref
                        + if onsite {
                            Vec3::zero()
                        } else {
                            r * (average * radial * overlap / distance)
                        };
                    h[(mu, nu)] += pref * overlap;
                    s[(mu, nu)] += overlap;
                    if pairs.len() == pairs.capacity() {
                        let remaining = GEOMETRY_CACHE_BYTES / 2 / std::mem::size_of::<Pair>();
                        if pairs.len() >= remaining {
                            return Err(Gfn1Error::InvalidInput("fragment AO image cache exceeds 256 MiB; reduce the reference cell or AO cutoff".into()));
                        }
                        pairs
                            .try_reserve_exact((remaining - pairs.len()).min(16384))
                            .map_err(|_| {
                                Gfn1Error::InvalidInput(
                                    "fragment image-cache allocation failed".into(),
                                )
                            })?;
                    }
                    pairs.push(Pair {
                        mu,
                        nu,
                        a,
                        b,
                        r,
                        ds,
                        dh,
                        cn_a: -0.5 * si.kcn_raw.unwrap_or(0.) * scale * overlap,
                        cn_b: -0.5 * sj.kcn_raw.unwrap_or(0.) * scale * overlap,
                    });
                }
            }
        }
        let gamma = if system.lattice.is_some() {
            periodic_gamma_matrix(system, &basis, &model, &pbc.ewald)?
        } else {
            effective_coulomb_matrix(system, &basis, &model)
        };
        let kernel = kernel_pairs(system, &basis, &model, pbc)?;
        let rep = crate::repulsion::repulsion_energy_gradient(system, params)?;
        let mut rep_pairs = Vec::new();
        crate::pairlist::for_each_unique_short_range_pair(
            system,
            crate::model::Cutoffs::default().repulsion,
            |pair| {
                let (energy, prefactor) =
                    crate::repulsion::repulsion_pair_energy_gradient_prefactor(
                        system.atoms[pair.i].z,
                        system.atoms[pair.j].z,
                        pair.r,
                        params,
                    )?;
                rep_pairs.push(RepPair {
                    a: pair.i,
                    b: pair.j,
                    energy,
                    gradient: pair.dr * prefactor,
                    r: pair.dr,
                });
                Ok(())
            },
        )?;
        let hal = crate::halogen::halogen_energy_gradient(system, params)?;
        let disp = if options.enable_dispersion {
            Some(crate::dispersion::dispersion_energy_gradient(
                system,
                params,
                options.d3_reference_path.as_deref(),
            )?)
        } else {
            None
        };
        let classical = rep.energy + hal.energy + disp.as_ref().map_or(0., |d| d.energy);
        let mut classical_derivatives = vec![0.; ndof];
        for a in 0..nat {
            for axis in 0..3 {
                classical_derivatives[3 * a + axis] = (rep.gradient[a]
                    + hal.gradient[a]
                    + disp.as_ref().map_or(Vec3::zero(), |d| d.gradient[a]))
                .to_array()[axis];
            }
        }
        let volume = system.lattice.map(|l| l.volume());
        if let Some(v) = volume {
            let rep = crate::repulsion::repulsion_stress(system, params)?.expect("periodic");
            let hal = crate::halogen::halogen_stress(system, params)?.expect("periodic");
            let disp = if options.enable_dispersion {
                crate::dispersion::dispersion_stress(
                    system,
                    params,
                    options.d3_reference_path.as_deref(),
                )?
            } else {
                None
            };
            for a in 0..3 {
                for b in 0..3 {
                    classical_derivatives[3 * nat + 3 * a + b] =
                        v * (rep[(a, b)] + hal[(a, b)] + disp.as_ref().map_or(0., |d| d[(a, b)]));
                }
            }
        }
        Ok(Self {
            basis,
            model,
            h,
            s,
            gamma,
            classical,
            repulsion: rep.energy,
            classical_derivatives,
            ndof,
            volume,
            pairs,
            kernel,
            cn_derivatives,
            rep_pairs,
        })
    }

    pub fn direction(&self, y: usize) -> Direction {
        let n = self.basis.len();
        let nsh = self.basis.shells.len();
        let mut h = Matrix::zeros(n, n);
        let mut s = Matrix::zeros(n, n);
        let mut gamma = Matrix::zeros(nsh, nsh);
        let ncart = self.ndof - if self.volume.is_some() { 9 } else { 0 };
        for p in &self.pairs {
            let (ds, dh) = if y < ncart {
                let sign = (usize::from(p.a == y / 3) as f64) - (usize::from(p.b == y / 3) as f64);
                (sign * p.ds.to_array()[y % 3], sign * p.dh.to_array()[y % 3])
            } else {
                let a = (y - ncart) / 3;
                let b = (y - ncart) % 3;
                (
                    p.ds.to_array()[a] * p.r.to_array()[b],
                    p.dh.to_array()[a] * p.r.to_array()[b],
                )
            };
            s[(p.mu, p.nu)] += ds;
            h[(p.mu, p.nu)] +=
                dh + p.cn_a * self.cn_derivatives[y][p.a] + p.cn_b * self.cn_derivatives[y][p.b];
        }
        for p in &self.kernel {
            let value = if y < ncart {
                ((usize::from(p.a == y / 3) as f64) - (usize::from(p.b == y / 3) as f64))
                    * p.gradient.to_array()[y % 3]
            } else {
                p.strain[y - ncart]
            };
            gamma[(p.i, p.j)] = value;
            gamma[(p.j, p.i)] = value;
        }
        Direction {
            h,
            s,
            gamma,
            classical: self.classical_derivatives[y],
        }
    }
    pub fn repulsion(&self, scales: &[f64], direction: Option<usize>) -> f64 {
        let ncart = self.ndof - if self.volume.is_some() { 9 } else { 0 };
        self.rep_pairs
            .iter()
            .map(|p| {
                scales[p.a]
                    * scales[p.b]
                    * match direction {
                        None => p.energy,
                        Some(y) if y < ncart => {
                            ((usize::from(p.a == y / 3) as f64)
                                - (usize::from(p.b == y / 3) as f64))
                                * p.gradient.to_array()[y % 3]
                        }
                        Some(y) => {
                            -p.gradient.to_array()[(y - ncart) / 3]
                                * p.r.to_array()[(y - ncart) % 3]
                        }
                    }
            })
            .sum()
    }
}

fn kernel_pairs(
    system: &PeriodicSystem,
    basis: &BasisSet,
    model: &ShellChargeModel,
    pbc: &PbcOptions,
) -> Result<Vec<KernelPair>> {
    let mut output = Vec::new();
    let nsh = basis.shells.len();
    // Lists and reciprocal weights are shared across all shell pairs.
    let alpha = resolve_alpha(system, &pbc.ewald);
    let lattice = system.lattice;
    let real_cut = (TAU / alpha).max(pbc.ewald.sr_cutoff);
    let images = crate::model::image_translations(system, real_cut);
    let reciprocal = lattice
        .map(|l| l.reciprocal_vectors_within(2. * alpha * TAU, false))
        .unwrap_or_default();
    for i in 0..nsh {
        for j in 0..=i {
            let a = basis.shells[i].atom_index;
            let b = basis.shells[j].atom_index;
            let rab = system.atoms[a].position - system.atoms[b].position;
            let eta = harmonic_average(model.hardness[i], model.hardness[j]);
            let mut gradient = Vec3::zero();
            let mut strain = [0.; 9];
            if let Some(lattice) = lattice {
                let volume = lattice.volume();
                let ie2 = 1. / (eta * eta);
                for (_, translation) in &images {
                    let r = rab - *translation;
                    let d = r.norm();
                    if d <= 1e-12 {
                        continue;
                    }
                    let mut radial = 0.;
                    if d <= TAU / alpha {
                        radial += -erfc(alpha * d) / (d * d)
                            - 2. * alpha / PI.sqrt() * (-alpha * alpha * d * d).exp() / d;
                        radial += QCORE_R3_COEFF * qcore_r3_real_value_derivatives(d, eta, alpha).1;
                    }
                    if d <= pbc.ewald.sr_cutoff {
                        radial += qcore_short_value_derivatives(d, eta).1;
                    }
                    gradient += r * (radial / d);
                    for x in 0..3 {
                        for y in 0..3 {
                            strain[3 * x + y] += radial * r.to_array()[x] * r.to_array()[y] / d;
                        }
                    }
                }
                for (_, g) in &reciprocal {
                    let g2 = g.norm2();
                    let x = g2 / (4. * alpha * alpha);
                    let exponential = (-x).exp();
                    let phase = g.dot(rab);
                    let w1 = 4. * PI / volume * exponential / g2;
                    let w3 = QCORE_R3_COEFF * 2. * PI / volume * ie2 * exp1(x);
                    gradient += *g * (-(w1 + w3) * phase.sin());
                    let dw1_dg2 = w1 * (-1. / (4. * alpha * alpha) - 1. / g2);
                    let dw3_dg2 = QCORE_R3_COEFF * 2. * PI / volume * ie2 * (-exponential / x)
                        / (4. * alpha * alpha);
                    for row in 0..3 {
                        for col in 0..3 {
                            let delta = if row == col { 1. } else { 0. };
                            strain[3 * row + col] += phase.cos()
                                * (-2.
                                    * g.to_array()[row]
                                    * g.to_array()[col]
                                    * (dw1_dg2 + dw3_dg2)
                                    - delta * (w1 + w3));
                        }
                    }
                }
                let background = -PI / (alpha * alpha * volume);
                let k0 = QCORE_R3_COEFF * 4. * PI / volume * ie2 * qcore_r3_k0_log(alpha);
                for axis in 0..3 {
                    strain[4 * axis] -= background + k0;
                }
            } else {
                let denominator = (rab.norm2() + 1. / (eta * eta)).powf(1.5);
                gradient = rab * (-1. / denominator);
            }
            output.push(KernelPair {
                i,
                j,
                a,
                b,
                gradient,
                strain,
            });
        }
    }
    if output
        .iter()
        .any(|p| p.strain.iter().any(|v| !v.is_finite()))
    {
        return Err(Gfn1Error::InvalidInput(
            "non-finite fragment Ewald derivative".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn displaced(system: &PeriodicSystem, y: usize, h: f64) -> PeriodicSystem {
        let mut out = system.clone();
        let ncart = 3 * system.atoms.len();
        if y < ncart {
            let mut r = out.atoms[y / 3].position.to_array();
            r[y % 3] += h;
            out.atoms[y / 3].position = Vec3::new(r[0], r[1], r[2]);
        } else {
            let a = (y - ncart) / 3;
            let b = (y - ncart) % 3;
            let strain = |r: Vec3| {
                let mut v = r.to_array();
                v[a] += h * v[b];
                Vec3::new(v[0], v[1], v[2])
            };
            for atom in &mut out.atoms {
                atom.position = strain(atom.position);
            }
            let lattice = out.lattice.unwrap();
            out.lattice = Some(
                crate::lattice::Lattice::new(
                    crate::math::Mat3 {
                        col: lattice.cell.col.map(strain),
                    },
                    lattice.periodic,
                )
                .unwrap(),
            );
        }
        out
    }
    #[test]
    fn fragment_geometry_all_components_match_position_and_strain_fd() {
        let params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions::default();
        let pbc = PbcOptions::default();
        for periodic in [false, true] {
            let comment = if periodic {
                "Lattice=\"13 .4 .2 0 14 .5 .2 .1 15\" pbc=\"T T T\""
            } else {
                "molecule"
            };
            let system = PeriodicSystem::from_xyz_str(
                &format!("2\n{comment}\nH .2 .4 -.1\nF 1.9 .6 .2\n"),
                0.,
                true,
            )
            .unwrap();
            let geometry = Geometry::build(&system, &params, &options, &pbc).unwrap();
            let step = 2e-5;
            let mut maxima = [0.0_f64; 4];
            for y in 0..geometry.ndof {
                let analytic = geometry.direction(y);
                let plus =
                    Geometry::build(&displaced(&system, y, step), &params, &options, &pbc).unwrap();
                let minus = Geometry::build(&displaced(&system, y, -step), &params, &options, &pbc)
                    .unwrap();
                for (which, (a, p, m)) in [
                    (&analytic.h, &plus.h, &minus.h),
                    (&analytic.s, &plus.s, &minus.s),
                    (&analytic.gamma, &plus.gamma, &minus.gamma),
                ]
                .into_iter()
                .enumerate()
                {
                    for ((a, p), m) in a.as_slice().iter().zip(p.as_slice()).zip(m.as_slice()) {
                        maxima[which] = maxima[which].max((a - (p - m) / (2. * step)).abs());
                    }
                }
                maxima[3] = maxima[3].max(
                    (analytic.classical - (plus.classical - minus.classical) / (2. * step)).abs(),
                );
            }
            eprintln!("fragment geometry periodic={periodic}: max H/S/Gamma/classical={maxima:?}");
            assert!(maxima.iter().all(|&e| e < 2e-7), "{maxima:?}");
        }
    }
}
