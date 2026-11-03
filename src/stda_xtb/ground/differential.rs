// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic first derivatives of the complete dedicated orbital generator.
//! Scalar forward differentiation includes fitted exponents and contractions;
//! Fermi density derivatives use a generalized-metric matrix function.
use super::*;
use crate::fragment::spectral::Spectrum;
use crate::stda_xtb::dual::Dual as D;

#[derive(Clone)]
struct Primitive {
    alpha: D,
    coefficient: D,
}
pub(crate) struct DirectionalOrbitals {
    pub ds: Matrix,
    pub dc: Matrix,
    pub de: Vec<f64>,
    pub d2s: Matrix,
    pub d2c: Matrix,
    pub d2e: Vec<f64>,
    pub degenerate_pairs: Vec<(usize, usize)>,
    pub fock_first: Matrix,
    pub fock_second: Matrix,
}
fn coordinate(system: &PeriodicSystem, direction: &[f64]) -> Vec<[D; 3]> {
    system
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let r = a.position.to_array();
            std::array::from_fn(|k| D::seed(r[k], direction[3 * i + k]))
        })
        .collect()
}
fn distance(a: [D; 3], b: [D; 3]) -> D {
    if a.iter().zip(b).all(|(a, b)| a.value == b.value) {
        return D::constant(0.);
    }
    (0..3)
        .fold(D::default(), |r, k| r + (a[k] - b[k]).powi(2))
        .sqrt()
}
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1., |r, i| r * (n - i) as f64 / (i + 1) as f64)
}
fn moment(ia: usize, ib: usize, a: D, b: D, alpha: D, beta: D) -> D {
    let p = alpha + beta;
    let center = (alpha * a + beta * b) / p;
    let mut value = D::default();
    for i in 0..=ia {
        for j in 0..=ib {
            let order = i + j;
            if order % 2 == 1 {
                continue;
            }
            let factor = (1..order).step_by(2).fold(1., |x, n| x * n as f64);
            let gaussian = (std::f64::consts::PI / p).sqrt() * factor / (2. * p).powi(order / 2);
            value += binomial(ia, i)
                * binomial(ib, j)
                * (center - a).powi(ia - i)
                * (center - b).powi(ib - j)
                * gaussian;
        }
    }
    value
}
fn pair_integrals(
    a: &AOBasisFunction,
    pa: &[Primitive],
    b: &AOBasisFunction,
    pb: &[Primitive],
    ra: [D; 3],
    rb: [D; 3],
    kinetic: bool,
) -> (D, D) {
    let mut s = D::default();
    let mut t = D::default();
    for p in pa {
        for q in pb {
            let total = p.alpha + q.alpha;
            let r2 = (0..3).fold(D::default(), |r, k| r + (ra[k] - rb[k]).powi(2));
            let pref = (-p.alpha * q.alpha / total * r2).exp() * p.coefficient * q.coefficient;
            for ac in &a.components {
                for bc in &b.components {
                    let ap = [ac.power.x, ac.power.y, ac.power.z];
                    let bp = [bc.power.x, bc.power.y, bc.power.z];
                    let m: [D; 3] = std::array::from_fn(|k| {
                        moment(ap[k], bp[k], ra[k], rb[k], p.alpha, q.alpha)
                    });
                    let factor = pref * ac.coefficient * bc.coefficient;
                    s += factor * m[0] * m[1] * m[2];
                    if kinetic {
                        for k in 0..3 {
                            let mut tk = q.alpha * (2 * bp[k] + 1) as f64 * m[k]
                                - 2. * q.alpha.powi(2)
                                    * moment(ap[k], bp[k] + 2, ra[k], rb[k], p.alpha, q.alpha);
                            if bp[k] >= 2 {
                                tk -= 0.5
                                    * (bp[k] * (bp[k] - 1)) as f64
                                    * moment(ap[k], bp[k] - 2, ra[k], rb[k], p.alpha, q.alpha);
                            }
                            t += factor * tk * m[(k + 1) % 3] * m[(k + 2) % 3];
                        }
                    }
                }
            }
        }
    }
    (s, t)
}
fn radial_dual(ng: usize, n: u8, l: usize, zeta: D) -> Result<Vec<Primitive>> {
    let key = format!("{ng}:{n}:{l}");
    let rows = tables()["sto"][&key]
        .as_array()
        .ok_or_else(|| invalid(format!("missing STO {key}")))?;
    let df = (1..2 * l).step_by(2).fold(1., |r, k| r * k as f64);
    Ok(rows
        .iter()
        .map(|r| {
            let alpha = r[0].as_f64().expect("STO exponent") * zeta.powi(2);
            let norm = (2. / std::f64::consts::PI * alpha).powf(0.75)
                * (4. * alpha).powf(l as f64 / 2.)
                / df.sqrt();
            Primitive {
                alpha,
                coefficient: r[1].as_f64().expect("STO coefficient") * norm,
            }
        })
        .collect())
}
fn basis_dual(
    system: &PeriodicSystem,
    p: &Parameters,
    q: &[D],
    cn: &[D],
) -> Result<(Vec<AOBasisFunction>, Vec<bool>, Vec<Vec<Primitive>>)> {
    let (aos, rydberg) = basis(
        system,
        p,
        &q.iter().map(|d| d.value).collect::<Vec<_>>(),
        &cn.iter().map(|d| d.value).collect::<Vec<_>>(),
    )?;
    let mut primitives = Vec::new();
    let mut index = 0;
    for (atom, center) in system.atoms.iter().enumerate() {
        let z = center.z;
        let mut references: [Option<(usize, Vec<Primitive>)>; 2] = [None, None];
        for shell in &p.elements[usize::from(z)].shells {
            let diffuse = shell.kind >= 11;
            let angulars = match shell.kind {
                11 => vec![0],
                12 => vec![0, 1],
                13 => vec![0, 1, 2],
                l => vec![l],
            };
            for l in angulars {
                let zeta = if diffuse {
                    D::constant(shell.zeta)
                } else {
                    shell.zeta
                        * (1.
                            + p.get("zqf")
                                * if cn[atom].value > 1e-6 {
                                    q[atom]
                                } else {
                                    D::default()
                                })
                        * (1. + 0.1 * p.get("zcnf") * cn[atom])
                };
                let ng = if diffuse {
                    if z <= 2 && l == 1 {
                        4
                    } else if shell.n > 5 {
                        6
                    } else {
                        3
                    }
                } else if shell.n > 5 {
                    6
                } else {
                    4
                };
                let principal = if z <= 2 && diffuse && l == 1 {
                    2
                } else if l == 2 && diffuse && shell.n >= 5 {
                    5
                } else {
                    shell.n
                };
                let mut radial = radial_dual(ng, principal, l, zeta)?;
                if diffuse && l < 2 && !(z <= 2 && l == 1) {
                    let (reference, ref_radial) = references[l]
                        .as_ref()
                        .ok_or_else(|| invalid("diffuse shell reference"))?;
                    let zero = [D::default(); 3];
                    let projection = pair_integrals(
                        &aos[*reference],
                        ref_radial,
                        &aos[index],
                        &radial,
                        zero,
                        zero,
                        false,
                    )
                    .0;
                    radial.extend(ref_radial.iter().map(|r| Primitive {
                        alpha: r.alpha,
                        coefficient: -projection * r.coefficient,
                    }));
                    let norm = pair_integrals(
                        &aos[index],
                        &radial,
                        &aos[index],
                        &radial,
                        zero,
                        zero,
                        false,
                    )
                    .0
                    .sqrt();
                    for r in &mut radial {
                        r.coefficient = r.coefficient / norm;
                    }
                }
                if !diffuse && l < 2 {
                    references[l] = Some((index, radial.clone()));
                }
                for _ in 0..2 * l + 1 {
                    primitives.push(radial.clone());
                    index += 1;
                }
            }
        }
    }
    Ok((aos, rydberg, primitives))
}
struct DualMatrix {
    n: usize,
    data: Vec<D>,
}
impl DualMatrix {
    fn new(n: usize) -> Self {
        Self {
            n,
            data: vec![D::default(); n * n],
        }
    }
    fn get(&self, i: usize, j: usize) -> D {
        self.data[i * self.n + j]
    }
    fn set(&mut self, i: usize, j: usize, v: D) {
        self.data[i * self.n + j] = v;
        self.data[j * self.n + i] = v;
    }
    fn value(&self) -> Matrix {
        Matrix::from_vec(self.n, self.n, self.data.iter().map(|d| d.value).collect())
            .expect("matrix size")
    }
    fn derivative(&self) -> Matrix {
        Matrix::from_vec(
            self.n,
            self.n,
            self.data.iter().map(|d| d.derivative).collect(),
        )
        .expect("matrix size")
    }
    fn second(&self) -> Matrix {
        Matrix::from_vec(self.n, self.n, self.data.iter().map(|d| d.second).collect())
            .expect("matrix size")
    }
}
fn integrals_dual(
    aos: &[AOBasisFunction],
    primitives: &[Vec<Primitive>],
    r: &[[D; 3]],
) -> (DualMatrix, DualMatrix) {
    let mut s = DualMatrix::new(aos.len());
    let mut t = DualMatrix::new(aos.len());
    for i in 0..aos.len() {
        for j in 0..=i {
            let (overlap, kinetic) = pair_integrals(
                &aos[i],
                &primitives[i],
                &aos[j],
                &primitives[j],
                r[aos[i].atom_index],
                r[aos[j].atom_index],
                true,
            );
            s.set(i, j, overlap);
            t.set(i, j, kinetic * EV_PER_HARTREE);
        }
    }
    (s, t)
}
#[allow(clippy::too_many_arguments)]
fn hamiltonian_dual(
    system: &PeriodicSystem,
    p: &Parameters,
    aos: &[AOBasisFunction],
    rydberg: &[bool],
    s: &DualMatrix,
    t: &DualMatrix,
    q: &[D],
    cn: &[D],
    r: &[[D; 3]],
) -> DualMatrix {
    let mut h = DualMatrix::new(aos.len());
    let k = [p.get("ks"), p.get("kp"), p.get("kd"), p.get("kf")];
    let levels: Vec<_> = aos
        .iter()
        .map(|a| a.hdiag_ev * (1. + 0.01 * p.get("kcn") * cn[a.atom_index]))
        .collect();
    let potential: Vec<_> = system
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            system
                .atoms
                .iter()
                .enumerate()
                .fold(D::default(), |sum, (j, b)| {
                    sum + EV_PER_HARTREE * q[j]
                        / (distance(r[i], r[j])
                            + 1. / (p.hardness[usize::from(a.z) - 1]
                                * p.hardness[usize::from(b.z) - 1])
                                .sqrt())
                })
        })
        .collect();
    for i in 0..aos.len() {
        for j in 0..=i {
            let average = 0.5 * (levels[i] + levels[j]);
            let value = if i == j {
                levels[i]
            } else if rydberg[i] && rydberg[j] {
                s.get(i, j) * p.get("kdiffa") * average
            } else if rydberg[i] || rydberg[j] {
                s.get(i, j) * p.get("kdiffb") * average
            } else {
                -t.get(i, j) * p.get("tscal")
                    + s.get(i, j)
                        * 0.5
                        * (k[aos[i].angular.as_index()] + k[aos[j].angular.as_index()])
                        * average
            };
            let charge = 0.5
                * s.get(i, j)
                * p.get("gscal")
                * (potential[aos[i].atom_index] + potential[aos[j].atom_index]);
            h.set(i, j, value - charge);
        }
    }
    h
}
fn correction_dual(system: &PeriodicSystem, r: &[[D; 3]]) -> Vec<D> {
    let mut correction = vec![D::default(); system.atoms.len()];
    for (i, a) in system.atoms.iter().enumerate() {
        for (j, b) in system.atoms.iter().enumerate() {
            if a.z == b.z {
                continue;
            }
            let pair = match (a.z.min(b.z), a.z.max(b.z)) {
                (1, 6) => Some(0.0502),
                (1, 7) => Some(0.1747),
                (1, 8) => Some(0.1671),
                (6, 7) => Some(0.0556),
                (6, 8) => Some(0.0234),
                (7, 8) => Some(-0.0346),
                _ => None,
            };
            let d = pair.map_or_else(
                || table("cm5_a0", usize::from(a.z) - 1) - table("cm5_a0", usize::from(b.z) - 1),
                |v| if a.z < b.z { v } else { -v },
            );
            correction[i] += d
                * (-2.474
                    * (distance(r[i], r[j]) * 0.52917726
                        - table("cm5_radius_angstrom", usize::from(a.z) - 1)
                        - table("cm5_radius_angstrom", usize::from(b.z) - 1)))
                .exp();
        }
    }
    correction
}

pub(crate) fn ground_directional(
    system: &PeriodicSystem,
    validated: &Orbitals,
    direction: &[f64],
    second_order: bool,
) -> Result<DirectionalOrbitals> {
    let mut p = Parameters::first()?;
    let r = coordinate(system, direction);
    let nat = r.len();
    let reference = system
        .atoms
        .iter()
        .map(|a| valence(a.z))
        .collect::<Result<Vec<_>>>()?;
    let electrons = reference.iter().sum::<f64>() - system.charge;
    let mut cn = vec![D::default(); nat];
    for i in 0..nat {
        for j in i + 1..nat {
            let d = distance(r[i], r[j]);
            if d.value * d.value > 4000. {
                continue;
            }
            let radius = table("covalent_radius_bohr", usize::from(system.atoms[i].z) - 1)
                + table("covalent_radius_bohr", usize::from(system.atoms[j].z) - 1);
            let value = 1. / (1. + (-16. * (radius / d - 1.)).exp());
            cn[i] += value;
            cn[j] += value;
        }
    }
    let en: Vec<_> = system
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| p.en[usize::from(a.z) - 1] - p.get("ken") * cn[i].sqrt())
        .collect();
    let normalization = electrons / reference.iter().sum::<f64>();
    let mut q = vec![D::default(); nat];
    for i in 0..nat {
        let mut population = D::constant(reference[i]);
        for j in 0..nat {
            if i != j {
                let radius = 0.5
                    * (table("covalent_radius_bohr", usize::from(system.atoms[i].z) - 1)
                        + table("covalent_radius_bohr", usize::from(system.atoms[j].z) - 1));
                population += (en[i] - en[j]) / (distance(r[i], r[j]) / radius).powi(6);
            }
        }
        q[i] = reference[i] - normalization * population;
    }
    let correction = correction_dual(system, &r);
    let (aos, rydberg, primitives) = basis_dual(system, &p, &q, &cn)?;
    let (s, t) = integrals_dual(&aos, &primitives, &r);
    let sv = s.value();
    let ds = s.derivative();
    for _ in 0..2 {
        let h = hamiltonian_dual(system, &p, &aos, &rydberg, &s, &t, &q, &cn, &r);
        let spectrum = Spectrum::new(
            &h.value(),
            &sv,
            1e-10,
            validated.electronic_temperature_kelvin * VTB_KB_EV,
            Some(electrons),
            0.,
        )?;
        let dp = spectrum.canonical_response(&h.derivative(), &ds, &sv)?.0;
        let dp2 = if second_order {
            spectrum.canonical_second_density(&h.derivative(), &ds, &h.second(), &s.second())?
        } else {
            Matrix::zeros(sv.rows(), sv.cols())
        };
        q = reference
            .iter()
            .zip(&correction)
            .map(|(a, b)| *a + *b)
            .collect();
        for i in 0..aos.len() {
            for j in 0..aos.len() {
                q[aos[i].atom_index] -=
                    D::seed_second(spectrum.density[(i, j)], dp[(i, j)], dp2[(i, j)]) * s.get(j, i);
            }
        }
    }
    p.read(std::str::from_utf8(crate::stda_xtb::PARAM2).expect("parameter UTF-8"))?;
    let (aos, rydberg, primitives) = basis_dual(system, &p, &q, &cn)?;
    let (s, t) = integrals_dual(&aos, &primitives, &r);
    let sv = s.value();
    let ds = s.derivative();
    let h = hamiltonian_dual(system, &p, &aos, &rydberg, &s, &t, &q, &cn, &r);
    let eig = lowdin_solve_generalized(&h.value(), &sv, 1e-10)?;
    // Differentiate in the primal CI orbital frame. Re-diagonalizing the
    // scalar-AD primal can rotate a near-degenerate block by much more than
    // roundoff; sign alignment alone cannot repair that different frame.
    let c = &validated.coefficients;
    let response_energies: Vec<_> = validated
        .energies
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            e * EV_PER_HARTREE
                - if i >= validated.nocc {
                    p.get("lshift")
                } else {
                    0.
                }
        })
        .collect();
    let f = c.transpose().matmul(&h.derivative())?.matmul(c)?;
    let m = c.transpose().matmul(&ds)?.matmul(c)?;
    let n = c.cols();
    let mut rotation = Matrix::zeros(n, n);
    let mut de = vec![0.; n];
    let mut degenerate_pairs = Vec::new();
    for i in 0..n {
        de[i] = (f[(i, i)] - response_energies[i] * m[(i, i)]) / EV_PER_HARTREE;
        rotation[(i, i)] = -0.5 * m[(i, i)];
        for j in 0..i {
            let gap = response_energies[i] - response_energies[j];
            if gap.abs() < 1e-8 * EV_PER_HARTREE {
                degenerate_pairs.push((i, j));
                rotation[(i, j)] = -0.5 * m[(i, j)];
                rotation[(j, i)] = -0.5 * m[(j, i)];
            } else {
                rotation[(j, i)] = (f[(j, i)] - response_energies[i] * m[(j, i)]) / gap;
                rotation[(i, j)] = (f[(i, j)] - response_energies[j] * m[(i, j)]) / -gap;
            }
        }
    }
    let mut dc = c.matmul(&rotation)?;
    let mut fock_first = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            fock_first[(i, j)] = (f[(i, j)]
                + response_energies[i] * rotation[(i, j)]
                + response_energies[j] * rotation[(j, i)])
                / EV_PER_HARTREE;
        }
    }
    let d2s = if second_order {
        s.second()
    } else {
        Matrix::zeros(0, 0)
    };
    let mut d2c = Matrix::zeros(0, 0);
    let mut d2e = Vec::new();
    let mut fock_second = Matrix::zeros(0, 0);
    if second_order {
        let f2 = c.transpose().matmul(&h.second())?.matmul(c)?;
        let m2 = c.transpose().matmul(&d2s)?.matmul(c)?;
        let mr = m.matmul(&rotation)?;
        let fr = f.matmul(&rotation)?;
        let rt = rotation.transpose();
        let rr = rt.matmul(&rotation)?;
        let mut er = rotation.clone();
        for i in 0..n {
            for j in 0..n {
                er[(i, j)] *= response_energies[i];
            }
        }
        let rer = rt.matmul(&er)?;
        let mut rotation2 = Matrix::zeros(n, n);
        fock_second = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let metric = m2[(i, j)] + 2. * (mr[(i, j)] + mr[(j, i)] + rr[(i, j)]);
                let known = f2[(i, j)] + 2. * (fr[(i, j)] + fr[(j, i)] + rer[(i, j)]);
                let gap = response_energies[j] - response_energies[i];
                if gap.abs() < 1e-8 * EV_PER_HARTREE {
                    rotation2[(i, j)] = -0.5 * metric;
                    fock_second[(i, j)] = (known
                        - 0.5 * (response_energies[i] + response_energies[j]) * metric)
                        / EV_PER_HARTREE;
                } else {
                    rotation2[(i, j)] = (known - response_energies[j] * metric) / gap;
                }
            }
        }
        d2e = (0..n).map(|i| fock_second[(i, i)]).collect();
        d2c = c.matmul(&rotation2)?;
    }
    // The scalar-AD integral contractions can differ in the last few bits.
    // Align arbitrary real MO phases to the primal CI basis before contraction.
    let phase_overlap = c.transpose().matmul(&sv)?.matmul(&validated.coefficients)?;
    for i in 0..n {
        if phase_overlap[(i, i)] < 0. {
            for mu in 0..n {
                dc[(mu, i)] = -dc[(mu, i)];
                if second_order {
                    d2c[(mu, i)] = -d2c[(mu, i)];
                }
            }
        }
    }
    let primal_error = eig
        .values
        .iter()
        .enumerate()
        .map(|(i, e)| {
            ((e + if i >= validated.nocc {
                p.get("lshift")
            } else {
                0.
            }) / EV_PER_HARTREE
                - validated.energies[i])
                .abs()
        })
        .fold(0.0_f64, f64::max);
    if primal_error > 1e-8 {
        return Err(invalid(format!(
            "sTDA analytic orbital primal mismatch {primal_error:e}"
        )));
    }
    Ok(DirectionalOrbitals {
        ds,
        dc,
        de,
        d2s,
        d2c,
        d2e,
        degenerate_pairs,
        fock_first,
        fock_second,
    })
}
