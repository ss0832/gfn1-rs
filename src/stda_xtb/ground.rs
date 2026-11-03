// SPDX-License-Identifier: GPL-3.0-or-later
//! Independent native sTDA-xTB orbital generator following pinned xtb4stda equations.
//! Parameter and universal STO/CM5 tables retain their upstream provenance.
use crate::basis::{spherical_components, AOBasisFunction};
use crate::error::{Gfn1Error, Result};
use crate::integrals::{contracted_kinetic, contracted_pair};
use crate::linalg::{lowdin_solve_generalized, Matrix};
use crate::math::Vec3;
use crate::params::AngularMomentum;
use crate::sto::{primitive_norm, PrimitiveGaussian};
use crate::system::PeriodicSystem;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;
mod differential;
pub(super) use differential::ground_directional;

pub(super) const EV_PER_HARTREE: f64 = 27.21139570;
pub(super) const VTB_KB_EV: f64 = 3.166808578545117E-06 * EV_PER_HARTREE;
pub(super) fn tables() -> &'static Value {
    static TABLES: OnceLock<Value> = OnceLock::new();
    TABLES.get_or_init(|| {
        serde_json::from_str(include_str!("../../third_party/stda_xtb/tables.json"))
            .expect("bundled sTDA-xTB tables")
    })
}
fn table(name: &str, index: usize) -> f64 {
    tables()[name][index]
        .as_f64()
        .expect("bundled numeric table")
}
fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
#[derive(Clone)]
struct Shell {
    n: u8,
    kind: usize,
    level: f64,
    zeta: f64,
    polarization: bool,
}
#[derive(Clone, Default)]
struct Element {
    shells: Vec<Shell>,
}
#[derive(Clone)]
struct Parameters {
    global: HashMap<String, f64>,
    elements: Vec<Element>,
    hardness: Vec<f64>,
    en: Vec<f64>,
}
impl Parameters {
    fn first() -> Result<Self> {
        let mut result = Self {
            global: HashMap::new(),
            elements: vec![Element::default(); 95],
            hardness: (0..94).map(|i| table("default_hardness", i)).collect(),
            en: (0..94).map(|i| table("default_en", i)).collect(),
        };
        result.read(std::str::from_utf8(super::PARAM1).expect("parameter UTF-8"))?;
        Ok(result)
    }
    fn get(&self, key: &str) -> f64 {
        *self.global.get(key).unwrap_or(&0.)
    }
    fn read(&mut self, text: &str) -> Result<()> {
        self.elements.fill(Element::default());
        let mut element = None;
        let mut global = false;
        for line in text.lines().map(str::trim) {
            if line == "$globpar" {
                global = true;
                continue;
            }
            if line == "$end" {
                global = false;
                element = None;
                continue;
            }
            if let Some(z) = line.strip_prefix("$Z=") {
                element = Some(
                    z.split_whitespace()
                        .next()
                        .ok_or_else(|| invalid("sTDA element header"))?
                        .parse::<usize>()
                        .map_err(|_| invalid("sTDA element number"))?,
                );
                continue;
            }
            if global {
                let mut words = line.split_whitespace();
                if let (Some(k), Some(v)) = (words.next(), words.next()) {
                    self.global.insert(
                        k.to_string(),
                        v.parse().map_err(|_| invalid("sTDA global parameter"))?,
                    );
                }
                continue;
            }
            let Some(z) = element else {
                continue;
            };
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            if key == "ao" {
                let chars: Vec<_> = value.chars().filter(|c| !c.is_whitespace()).collect();
                let mut i = 0;
                let mut shells = Vec::new();
                while i < chars.len() {
                    let n = chars[i]
                        .to_digit(10)
                        .ok_or_else(|| invalid("sTDA basis principal quantum number"))?
                        as u8;
                    i += 1;
                    let start = i;
                    while i < chars.len() && !chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let label: String = chars[start..i].iter().collect();
                    let kind = match label.as_str() {
                        "s" => 0,
                        "p" => 1,
                        "d" => 2,
                        "f" => 3,
                        "S" => 11,
                        "sp" => 12,
                        "spd" => 13,
                        _ => return Err(invalid(format!("unsupported sTDA basis label {label}"))),
                    };
                    shells.push(Shell {
                        n,
                        kind,
                        level: 0.,
                        zeta: 0.,
                        polarization: false,
                    });
                }
                self.elements[z].shells = shells;
            } else if key == "lev" || key == "exp" {
                let values = value
                    .split_whitespace()
                    .map(|x| {
                        x.parse::<f64>()
                            .map_err(|_| invalid("sTDA shell parameter"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if values.len() != self.elements[z].shells.len() {
                    return Err(invalid("sTDA shell parameter count mismatch"));
                }
                let polarization_exponent = self.elements[z]
                    .shells
                    .iter()
                    .zip(&values)
                    .filter(|(s, v)| s.kind < 4 && **v > 0. && **v < 4.)
                    .max_by_key(|(s, _)| s.kind)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.)
                    * self.get("fpol");
                for (s, value) in self.elements[z].shells.iter_mut().zip(values) {
                    if key == "lev" {
                        s.level = value;
                    } else {
                        s.polarization = value <= 0.;
                        s.zeta = if s.polarization {
                            polarization_exponent
                        } else {
                            value
                        };
                    }
                }
            } else if key == "EN" {
                self.en[z - 1] = value.parse().map_err(|_| invalid("sTDA EN parameter"))?;
            } else if key == "GAM" {
                self.hardness[z - 1] = value
                    .parse()
                    .map_err(|_| invalid("sTDA hardness parameter"))?;
            }
        }
        Ok(())
    }
}

pub struct Orbitals {
    pub aos: Vec<AOBasisFunction>,
    pub overlap: Matrix,
    pub coefficients: Matrix,
    pub energies: Vec<f64>,
    pub nocc: usize,
    pub vtb_charges: Vec<f64>,
    pub electronic_temperature_kelvin: f64,
    pub maximum_occupation_deviation: f64,
    /// Diagnostic canonical occupations of the final unshifted Hamiltonian.
    /// CI transitions use the fixed closed-shell nocc partition, not this array.
    pub occupations: Vec<f64>,
    pub(crate) unshifted_energies: Vec<f64>,
    pub(crate) virtual_level_shift: f64,
}
fn valence(z: u8) -> Result<f64> {
    let core = match z {
        1..=2 => 0,
        3..=10 => 2,
        11..=18 => 10,
        19..=29 => 18,
        30..=36 => 28,
        37..=47 => 36,
        48..=54 => 46,
        55..=71 => 54,
        72..=79 => 68,
        80..=86 => 78,
        _ => return Err(invalid("sTDA-xTB valence table supports Z=1..86")),
    };
    Ok(f64::from(z) - f64::from(core))
}
fn basis_count(system: &PeriodicSystem, p: &Parameters) -> Result<usize> {
    let mut count = 0usize;
    for atom in &system.atoms {
        let shells = &p.elements[usize::from(atom.z)].shells;
        if shells.is_empty() {
            return Err(invalid(format!(
                "pinned sTDA-xTB parameter set has no basis for Z={}",
                atom.z
            )));
        }
        for shell in shells {
            let functions = match shell.kind {
                11 => 1,
                12 => 4,
                13 => 9,
                l => 2 * l + 1,
            };
            count = count
                .checked_add(functions)
                .ok_or_else(|| invalid("sTDA basis dimension overflow"))?;
        }
    }
    Ok(count)
}
fn radial(ng: usize, n: u8, l: usize, zeta: f64) -> Result<Vec<PrimitiveGaussian>> {
    let key = format!("{ng}:{n}:{l}");
    let data = tables()["sto"][&key].as_array().ok_or_else(|| {
        invalid(format!(
            "the pinned sTDA-xTB source has no supported STO expansion {key}"
        ))
    })?;
    Ok(data
        .iter()
        .map(|p| {
            let exponent = p[0].as_f64().expect("STO exponent") * zeta * zeta;
            PrimitiveGaussian {
                exponent,
                coefficient: p[1].as_f64().expect("STO coefficient") * primitive_norm(exponent, l),
            }
        })
        .collect())
}
fn ao(
    atom: usize,
    z: u8,
    shell: &Shell,
    l: usize,
    primitives: Vec<PrimitiveGaussian>,
    level: f64,
) -> Result<Vec<AOBasisFunction>> {
    let angular = match l {
        0 => AngularMomentum::S,
        1 => AngularMomentum::P,
        2 => AngularMomentum::D,
        3 => AngularMomentum::F,
        _ => return Err(invalid("sTDA angular momentum")),
    };
    Ok(spherical_components(angular)?
        .into_iter()
        .map(|(cart_label, components)| AOBasisFunction {
            atom_index: atom,
            z,
            shell_index: 0,
            shell_param_index: 0,
            shell_label: format!("{}{}", shell.n, cart_label),
            angular,
            cart_label,
            components,
            hdiag_ev: level,
            hdiag_ha: level / EV_PER_HARTREE,
            slater: shell.zeta,
            principal_n: shell.n,
            nprim: primitives.len(),
            reference_occ: 0.,
            is_valence: shell.kind < 11 && !shell.polarization,
            poly_raw: None,
            kcn_raw: None,
            lpar_raw: None,
            primitives: primitives.clone(),
        })
        .collect())
}
fn orthogonalized(
    mut diffuse: Vec<PrimitiveGaussian>,
    reference: &AOBasisFunction,
    template: &AOBasisFunction,
) -> Result<Vec<PrimitiveGaussian>> {
    let mut trial = template.clone();
    trial.primitives = diffuse.clone();
    let overlap = contracted_pair(reference, &trial, Vec3::zero(), Vec3::zero()).0;
    diffuse.extend(reference.primitives.iter().map(|p| PrimitiveGaussian {
        exponent: p.exponent,
        coefficient: -overlap * p.coefficient,
    }));
    trial.primitives = diffuse.clone();
    let norm = contracted_pair(&trial, &trial, Vec3::zero(), Vec3::zero())
        .0
        .sqrt();
    if !(norm.is_finite() && norm > 1e-10) {
        return Err(invalid("sTDA diffuse shell is linearly dependent"));
    }
    for p in &mut diffuse {
        p.coefficient /= norm;
    }
    Ok(diffuse)
}
fn basis(
    system: &PeriodicSystem,
    parameters: &Parameters,
    q: &[f64],
    cn: &[f64],
) -> Result<(Vec<AOBasisFunction>, Vec<bool>)> {
    let mut aos = Vec::new();
    let mut rydberg = Vec::new();
    for (atom, center) in system.atoms.iter().enumerate() {
        let z = center.z;
        let shells = &parameters.elements[usize::from(z)].shells;
        if shells.is_empty() {
            return Err(invalid(format!(
                "pinned sTDA-xTB parameter set has no basis for Z={z}"
            )));
        }
        let mut references: [Option<AOBasisFunction>; 2] = [None, None];
        for shell in shells {
            let diffuse = shell.kind >= 11;
            let angulars: Vec<_> = match shell.kind {
                11 => vec![0],
                12 => vec![0, 1],
                13 => vec![0, 1, 2],
                l => vec![l],
            };
            for l in angulars {
                let zeta = if diffuse {
                    shell.zeta
                } else {
                    shell.zeta
                        * (1. + parameters.get("zqf") * if cn[atom] > 1e-6 { q[atom] } else { 0. })
                        * (1. + 0.1 * parameters.get("zcnf") * cn[atom])
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
                let mut primitives = radial(ng, principal, l, zeta)?;
                let split = if z == 2 {
                    2.25
                } else if z <= 2 {
                    0.
                } else {
                    parameters.get("split")
                };
                let level = shell.level
                    + if diffuse {
                        match l {
                            0 => -split,
                            1 => split,
                            _ => 2. * split,
                        }
                    } else {
                        0.
                    };
                let template = ao(atom, z, shell, l, primitives.clone(), level)?.remove(0);
                if diffuse && l < 2 && !(z <= 2 && l == 1) {
                    let reference = references[l]
                        .as_ref()
                        .ok_or_else(|| invalid("sTDA diffuse shell precedes its valence shell"))?;
                    primitives = orthogonalized(primitives, reference, &template)?;
                }
                let functions = ao(atom, z, shell, l, primitives, level)?;
                if !diffuse && l < 2 {
                    references[l] = Some(functions[0].clone());
                }
                rydberg.extend(std::iter::repeat_n(diffuse, functions.len()));
                aos.extend(functions);
            }
        }
    }
    Ok((aos, rydberg))
}
fn integrals(system: &PeriodicSystem, aos: &[AOBasisFunction]) -> (Matrix, Matrix) {
    let n = aos.len();
    let mut s = Matrix::zeros(n, n);
    let mut t = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let a = system.atoms[aos[i].atom_index].position;
            let b = system.atoms[aos[j].atom_index].position;
            let overlap = contracted_pair(&aos[i], &aos[j], a, b).0;
            let kinetic = contracted_kinetic(&aos[i], &aos[j], a, b) * EV_PER_HARTREE;
            s[(i, j)] = overlap;
            s[(j, i)] = overlap;
            t[(i, j)] = kinetic;
            t[(j, i)] = kinetic;
        }
    }
    (s, t)
}
fn hamiltonian(
    system: &PeriodicSystem,
    parameters: &Parameters,
    aos: &[AOBasisFunction],
    rydberg: &[bool],
    s: &Matrix,
    t: &Matrix,
    q: &[f64],
    cn: &[f64],
) -> Matrix {
    let n = aos.len();
    let mut h = Matrix::zeros(n, n);
    let k = [
        parameters.get("ks"),
        parameters.get("kp"),
        parameters.get("kd"),
        parameters.get("kf"),
    ];
    let levels: Vec<_> = aos
        .iter()
        .map(|a| a.hdiag_ev * (1. + 0.01 * parameters.get("kcn") * cn[a.atom_index]))
        .collect();
    let potential: Vec<_> = system
        .atoms
        .iter()
        .map(|a| {
            system
                .atoms
                .iter()
                .zip(q)
                .map(|(b, &q)| {
                    EV_PER_HARTREE * q
                        / ((a.position - b.position).norm()
                            + 1. / (parameters.hardness[usize::from(a.z) - 1]
                                * parameters.hardness[usize::from(b.z) - 1])
                                .sqrt())
                })
                .sum::<f64>()
        })
        .collect();
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j {
                levels[i]
            } else {
                let average = 0.5 * (levels[i] + levels[j]);
                if rydberg[i] && rydberg[j] {
                    s[(i, j)] * parameters.get("kdiffa") * average
                } else if rydberg[i] || rydberg[j] {
                    s[(i, j)] * parameters.get("kdiffb") * average
                } else {
                    -t[(i, j)] * parameters.get("tscal")
                        + s[(i, j)]
                            * 0.5
                            * (k[aos[i].angular.as_index()] + k[aos[j].angular.as_index()])
                            * average
                }
            };
            // Apply the operator to every AO pair. A component-wise hard
            // |S| cutoff introduces orientation-dependent discontinuities in
            // nuclear derivatives when an off-diagonal integral crosses zero.
            let charge = 0.5
                * s[(i, j)]
                * parameters.get("gscal")
                * (potential[aos[i].atom_index] + potential[aos[j].atom_index]);
            h[(i, j)] = value - charge;
            h[(j, i)] = value - charge;
        }
    }
    h
}
fn cm5(system: &PeriodicSystem) -> Vec<f64> {
    let mut q = vec![0.; system.atoms.len()];
    for (i, a) in system.atoms.iter().enumerate() {
        for b in &system.atoms {
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
            q[i] += d
                * (-2.474
                    * ((a.position - b.position).norm() * 0.52917726
                        - table("cm5_radius_angstrom", usize::from(a.z) - 1)
                        - table("cm5_radius_angstrom", usize::from(b.z) - 1)))
                .exp();
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::column_weighted_gram;
    #[test]
    fn native_stda_orbitals_match_independent_xtb4stda_wavefunctions() {
        let reference: Value =
            serde_json::from_str(include_str!("../../tests/data/stda-ground-reference.json"))
                .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let atoms = case["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["positions_bohr"].as_array().unwrap())
                .map(|(z, r)| crate::system::Atom {
                    z: z.as_u64().unwrap() as u8,
                    position: Vec3::new(
                        r[0].as_f64().unwrap(),
                        r[1].as_f64().unwrap(),
                        r[2].as_f64().unwrap(),
                    ),
                })
                .collect();
            let result = ground_state(&PeriodicSystem::new(atoms, None), 512).unwrap();
            assert_eq!(result.nocc, case["nocc"].as_u64().unwrap() as usize);
            let energies = case["orbital_energies_hartree"].as_array().unwrap();
            assert_eq!(
                result.energies.iter().filter(|&&e| e <= 3.).count(),
                energies.len()
            );
            let e = result
                .energies
                .iter()
                .zip(energies)
                .map(|(a, b)| (a - b.as_f64().unwrap()).abs())
                .fold(0.0_f64, f64::max);
            let occ: Vec<_> = (0..result.aos.len())
                .map(|i| if i < result.nocc { 2. } else { 0. })
                .collect();
            let p = column_weighted_gram(&result.coefficients, &occ).unwrap();
            let mut d = 0.0_f64;
            for i in 0..p.rows() {
                for j in 0..p.cols() {
                    d = d.max((p[(i, j)] - case["density"][i][j].as_f64().unwrap()).abs());
                }
            }
            eprintln!(
                "{} native sTDA ground orbital/density max errors {e:.3e}/{d:.3e}",
                case["name"]
            );
            assert!(
                e < 1e-6 && d < 1e-6,
                "independent orbital generator mismatch"
            );
            let reference_coefficients = Matrix::from_vec(
                result.aos.len(),
                energies.len(),
                case["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
                    .collect(),
            )
            .unwrap();
            let overlap = result
                .coefficients
                .transpose()
                .matmul(&result.overlap)
                .unwrap()
                .matmul(&reference_coefficients)
                .unwrap();
            let error = (0..energies.len())
                .map(|i| (overlap[(i, i)].abs() - 1.).abs())
                .fold(0.0_f64, f64::max);
            eprintln!(
                "{} independent MO column overlap error={error:e}",
                case["name"]
            );
            assert!(error < 1e-6);
            if case["name"] == "water" {
                let dipole: Value = serde_json::from_str(include_str!(
                    "../../tests/data/stda-dipole-reference.json"
                ))
                .unwrap();
                let coordinates = case["positions_bohr"].as_array().unwrap();
                let mut maximum = 0.0_f64;
                for i in 0..result.aos.len() {
                    for j in 0..=i {
                        let center = |a: usize| {
                            Vec3::new(
                                coordinates[a][0].as_f64().unwrap(),
                                coordinates[a][1].as_f64().unwrap(),
                                coordinates[a][2].as_f64().unwrap(),
                            )
                        };
                        let integral = contracted_pair(
                            &result.aos[i],
                            &result.aos[j],
                            center(result.aos[i].atom_index),
                            center(result.aos[j].atom_index),
                        );
                        for (axis, value) in
                            [integral.1, integral.2, integral.3].into_iter().enumerate()
                        {
                            let value = value
                                + center(result.aos[j].atom_index).to_array()[axis] * integral.0;
                            let expected = dipole[axis][i][j].as_f64().unwrap();
                            if (value + expected).abs() > maximum {
                                maximum = (value + expected).abs();
                                if maximum > 1e-5 {
                                    eprintln!("AO dipole largest axis={axis} mu={i} nu={j} ours={value:e} reference={expected:e}");
                                }
                            }
                        }
                    }
                }
                eprintln!("water independent AO dipole max error={maximum:e}");
                assert!(maximum < 1e-7);
            }
        }
    }
}
pub fn ground_state(system: &PeriodicSystem, max_ao: usize) -> Result<Orbitals> {
    ground_state_at_temperature(system, max_ao, 300.)
}
pub fn ground_state_at_temperature(
    system: &PeriodicSystem,
    max_ao: usize,
    temperature: f64,
) -> Result<Orbitals> {
    if !temperature.is_finite() || temperature < 0. {
        return Err(invalid(
            "sTDA electronic temperature must be finite and nonnegative",
        ));
    }
    if system.atoms.is_empty() || system.lattice.is_some() {
        return Err(invalid("native sTDA-xTB requires a nonempty molecule"));
    }
    let mut parameters = Parameters::first()?;
    for a in &system.atoms {
        if a.z == 0 || a.z > 86 || !a.position.norm2().is_finite() {
            return Err(invalid("invalid sTDA-xTB element or coordinates"));
        }
    }
    let mut extended_parameters = parameters.clone();
    extended_parameters.read(std::str::from_utf8(super::PARAM2).expect("parameter UTF-8"))?;
    if basis_count(system, &parameters)? > max_ao
        || basis_count(system, &extended_parameters)? > max_ao
    {
        return Err(invalid(
            "sTDA basis exceeds max_ao/memory budget before integral allocation",
        ));
    }
    let reference = system
        .atoms
        .iter()
        .map(|a| valence(a.z))
        .collect::<Result<Vec<_>>>()?;
    let electrons = reference.iter().sum::<f64>() - system.charge;
    let nocc = (electrons / 2.).round() as usize;
    if !electrons.is_finite() || electrons <= 0. || (electrons - 2. * nocc as f64).abs() > 1e-8 {
        return Err(invalid(
            "native sTDA-xTB currently requires an even closed-shell electron count",
        ));
    }
    let nat = system.atoms.len();
    let mut cn = vec![0.; nat];
    for i in 0..nat {
        for j in i + 1..nat {
            let d = (system.atoms[i].position - system.atoms[j].position).norm();
            if d < 1e-8 {
                return Err(invalid("coincident sTDA-xTB atoms"));
            }
            if d * d > 4000. {
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
        .map(|(i, a)| parameters.en[usize::from(a.z) - 1] - parameters.get("ken") * cn[i].sqrt())
        .collect();
    let mut q = vec![0.; nat];
    let normalization = electrons / reference.iter().sum::<f64>();
    for i in 0..nat {
        let mut population = reference[i];
        for j in 0..nat {
            if i != j {
                let radius = 0.5
                    * (table("covalent_radius_bohr", usize::from(system.atoms[i].z) - 1)
                        + table("covalent_radius_bohr", usize::from(system.atoms[j].z) - 1));
                let d = (system.atoms[i].position - system.atoms[j].position).norm() / radius;
                population += (en[i] - en[j]) / d.powi(6);
            }
        }
        q[i] = reference[i] - population * normalization;
    }
    let correction = cm5(system);
    let (aos, rydberg) = basis(system, &parameters, &q, &cn)?;
    if aos.len() > max_ao || nocc >= aos.len() {
        return Err(invalid(
            "sTDA-xTB basis exceeds max_ao or has no virtual orbitals",
        ));
    }
    let (s, t) = integrals(system, &aos);
    let mut maximum_occupation_deviation = 0.0_f64;
    for _ in 0..2 {
        let mu = crate::fragment::spectral::Spectrum::new(
            &hamiltonian(system, &parameters, &aos, &rydberg, &s, &t, &q, &cn),
            &s,
            1e-10,
            temperature * VTB_KB_EV,
            Some(electrons),
            0.,
        )?;
        for (i, &f) in mu.occupations.iter().enumerate() {
            maximum_occupation_deviation =
                maximum_occupation_deviation.max((f - if i < nocc { 2. } else { 0. }).abs());
        }
        let p = mu.density;
        q = reference
            .iter()
            .zip(&correction)
            .map(|(a, b)| a + b)
            .collect();
        for i in 0..aos.len() {
            for j in 0..aos.len() {
                q[aos[i].atom_index] -= p[(i, j)] * s[(j, i)];
            }
        }
    }
    let vtb_charges = q.clone();
    parameters = extended_parameters;
    let (aos, rydberg) = basis(system, &parameters, &q, &cn)?;
    if aos.len() > max_ao {
        return Err(invalid("extended sTDA-xTB basis exceeds max_ao"));
    }
    let (overlap, t) = integrals(system, &aos);
    let eig = lowdin_solve_generalized(
        &hamiltonian(system, &parameters, &aos, &rydberg, &overlap, &t, &q, &cn),
        &overlap,
        1e-10,
    )?;
    let kt = temperature * VTB_KB_EV;
    let mu = crate::fragment::spectral::canonical_mu(&eig.values, electrons, kt)?;
    let occupations: Vec<_> = eig
        .values
        .iter()
        .map(|&e| {
            if kt > 0. {
                crate::fragment::spectral::fermi(e, mu, kt)
            } else if e < mu {
                2.
            } else {
                0.
            }
        })
        .collect();
    let final_deviation = occupations
        .iter()
        .enumerate()
        .map(|(i, f)| (f - if i < nocc { 2. } else { 0. }).abs())
        .fold(0.0_f64, f64::max);
    // The extended-basis eigenproblem is not a third density/charge update.
    // Thermal tails here are diagnostic and do not enter the fixed-nocc CI.
    // Intermediate VTB densities still use the requested temperature.
    maximum_occupation_deviation = maximum_occupation_deviation.max(final_deviation);
    let energies = eig
        .values
        .iter()
        .enumerate()
        .map(|(i, e)| {
            (e + if i >= nocc {
                parameters.get("lshift")
            } else {
                0.
            }) / EV_PER_HARTREE
        })
        .collect();
    Ok(Orbitals {
        aos,
        overlap,
        coefficients: eig.vectors,
        energies,
        nocc,
        vtb_charges,
        electronic_temperature_kelvin: temperature,
        maximum_occupation_deviation,
        occupations,
        unshifted_energies: eig.values.iter().map(|energy| energy / EV_PER_HARTREE).collect(),
        virtual_level_shift: parameters.get("lshift") / EV_PER_HARTREE,
    })
}
