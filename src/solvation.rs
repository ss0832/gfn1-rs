// SPDX-License-Identifier: GPL-3.0-or-later
// Rust adaptation of xtb solvation, copyright (C) 2019-2020 Sebastian Ehlert;
// grid data copyright (C) 2017-2020 Stefan Grimme, upstream LGPL-3.0-or-later.
// Original files, license texts and hashes: third_party/xtb, assets.json.
//! GFN1-specific equilibrium ALPB/P16 and GBSA/Still (Ehlert2021ALPB).
//! Geometry is cached across SCC iterations. The SAME quadratic form supplies
//! energy, charge potential and response kernel. In GFN1, solvent charges are
//! q_CM5 = q_Mulliken + c_CM5(R), an AFFINE geometry-dependent mapping, as in
//! pinned scf_module.F90. Its linear Fock term and geometric chain are included.
//! Coordinate derivatives use
//! forward-mode analytic differentiation of Born, SASA, H-bond and shape terms;
//! they do not displace atoms. The directional jets feed variational SCC response.

use crate::basis::BasisSet;
use crate::constants::KB_HARTREE_PER_K;
use crate::error::{Gfn1Error, Result};
use crate::jets::Jet1;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::PeriodicSystem;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub const UPSTREAM_COMMIT: &str = "26b28010e805f7d1aeeef39813feb473e69cc4be";
// Model-side conversions from the pinned xtb/mctc implementation.
const AATOAU: f64 = 1.0 / 0.52917726;
const KCAL_TO_AU: f64 = 1.0 / 627.50947428;
const WIDTH: f64 = 0.3 * AATOAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolvationModel {
    Alpb,
    Gbsa,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionState {
    Gsolv,
    Bar1Mol,
    Reference,
}

#[derive(Clone, Debug)]
pub struct SolvationOptions {
    pub model: SolvationModel,
    pub solvent: String,
    pub state: SolutionState,
    /// Thermodynamic solvent temperature, separate from electronic Fermi T.
    pub temperature: f64,
    /// Exact Lebedev grid size; unsupported sizes are errors, not replacements.
    pub grid_points: usize,
}

impl SolvationOptions {
    pub fn new(model: SolvationModel, solvent: impl Into<String>) -> Self {
        Self {
            model,
            solvent: solvent.into(),
            state: SolutionState::Gsolv,
            temperature: 298.15,
            grid_points: 230,
        }
    }
}

#[derive(Clone, Debug)]
struct Parameters {
    dielectric: f64,
    mass: f64,
    density: f64,
    scale: f64,
    probe: f64,
    shift: f64,
    offset: f64,
    tension: Vec<f64>,
    descreen: Vec<f64>,
    hbond: Vec<f64>,
}

macro_rules! solvent_sources {
    ($name:expr, $($s:literal),+ $(,)?) => { match $name {
        $($s => Some(include_str!(concat!("../third_party/xtb/include/param_alpb_", $s, ".fh"))),)+
        _ => None,
    }};
}
macro_rules! gbsa_sources {
    ($name:expr, $($s:literal),+ $(,)?) => { match $name {
        $($s => Some(include_str!(concat!("../third_party/xtb/include/param_gbsa_", $s, ".fh"))),)+
        _ => None,
    }};
}

fn normalized_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "h2o" => "water".into(),
        "dichlormethane" | "dichloromethane" => "ch2cl2".into(),
        "chloroform" => "chcl3".into(),
        "nhexan" | "n-hexan" | "nhexane" | "n-hexane" => "hexane".into(),
        other => other.to_string(),
    }
}

fn fortran_numbers(text: &str) -> Result<Vec<f64>> {
    text.split("_wp")
        .take(text.matches("_wp").count())
        .map(|part| {
            let token: String = part
                .chars()
                .rev()
                .take_while(|c| {
                    c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E' | 'd' | 'D')
                })
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            token.replace(['d', 'D'], "e").parse().map_err(|_| {
                Gfn1Error::InvalidInput("invalid numeric value in bundled solvation data".into())
            })
        })
        .collect()
}

fn parameters(options: &SolvationOptions) -> Result<Parameters> {
    let name = normalized_name(&options.solvent);
    let (text, key) = match options.model {
        SolvationModel::Alpb => (
            solvent_sources!(
                name.as_str(),
                "acetone",
                "acetonitrile",
                "aniline",
                "benzaldehyde",
                "benzene",
                "ch2cl2",
                "chcl3",
                "cs2",
                "dioxane",
                "dmf",
                "dmso",
                "ether",
                "ethylacetate",
                "furane",
                "hexadecane",
                "hexane",
                "nitromethane",
                "octanol",
                "phenol",
                "thf",
                "toluene",
                "water",
                "woctanol",
                "methanol",
                "ethanol"
            ),
            format!("gfn1_alpb_{name}"),
        ),
        SolvationModel::Gbsa => {
            let suffix = if name == "water" { "h2o" } else { &name };
            (
                gbsa_sources!(
                    suffix,
                    "acetone",
                    "acetonitrile",
                    "benzene",
                    "ch2cl2",
                    "chcl3",
                    "cs2",
                    "dmso",
                    "ether",
                    "h2o",
                    "methanol",
                    "thf",
                    "toluene"
                ),
                format!("gfn1_{suffix}"),
            )
        }
    };
    let err = || {
        Gfn1Error::InvalidInput(format!(
            "GFN1 {:?}: solvent `{name}` has no bundled parameter set",
            options.model
        ))
    };
    let text = text.ok_or_else(err)?;
    let block = text
        .split_once(&format!(":: {key} ="))
        .ok_or_else(err)?
        .1
        .split("type(gbsa_parameter)")
        .next()
        .unwrap();
    let p = fortran_numbers(block)?;
    if p.len() != 290 {
        return Err(Gfn1Error::InvalidInput(format!(
            "invalid {key} parameter count {}",
            p.len()
        )));
    }
    Ok(Parameters {
        dielectric: p[0],
        mass: p[1],
        density: p[2],
        scale: p[3],
        probe: p[4] * AATOAU,
        shift: p[5] * KCAL_TO_AU,
        offset: p[6] * 0.1 * AATOAU,
        tension: p[8..102]
            .iter()
            .map(|x| x * 4.0 * std::f64::consts::PI * 1e-5)
            .collect(),
        descreen: p[102..196].to_vec(),
        hbond: p[196..290].iter().map(|x| -x * x * KCAL_TO_AU).collect(),
    })
}

/// Directional derivatives on the reference geometry's local solvent branches.
#[derive(Clone, Copy, Debug, Default)]
struct D(Jet1);

impl D {
    fn c(v: f64) -> Self {
        Self(Jet1::constant(v))
    }
    fn sqrt(self) -> Self {
        self.powf(0.5)
    }
    fn ln(self) -> Self {
        let v = self.0.value;
        let inv = v.recip();
        Self(
            self.0
                .compose(v.ln(), inv, -inv * inv, 2. * inv.powi(3), -6. * inv.powi(4)),
        )
    }
    fn exp(self) -> Self {
        Self(self.0.exp())
    }
    fn tanh(self) -> Self {
        let t = self.0.value.tanh();
        let u = 1. - t * t;
        Self(self.0.compose(
            t,
            u,
            -2. * t * u,
            (6. * t * t - 2.) * u,
            (16. * t - 24. * t.powi(3)) * u,
        ))
    }
    fn powf(self, p: f64) -> Self {
        if self.0.d1 == 0. && self.0.d2 == 0. && self.0.d3 == 0. && self.0.d4 == 0. {
            Self::c(self.0.value.powf(p))
        } else {
            Self(self.0.powf(p))
        }
    }
}
impl Add for D {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0.add(&b.0))
    }
}
impl Sub for D {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0.sub(&b.0))
    }
}
impl Mul for D {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self(self.0.mul(&b.0))
    }
}
impl Div for D {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        self * b.powf(-1.)
    }
}
impl Neg for D {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.scale(-1.))
    }
}
macro_rules! scalar_ops {
    ($trait:ident,$method:ident) => {
        impl $trait<f64> for D {
            type Output = D;
            fn $method(self, b: f64) -> D {
                self.$method(D::c(b))
            }
        }
        impl $trait<D> for f64 {
            type Output = D;
            fn $method(self, b: D) -> D {
                D::c(self).$method(b)
            }
        }
    };
}
scalar_ops!(Add, add);
scalar_ops!(Sub, sub);
scalar_ops!(Mul, mul);
scalar_ops!(Div, div);
fn norm2(r: [D; 3]) -> D {
    r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
}
fn distance(a: [D; 3], b: [D; 3]) -> D {
    norm2([a[0] - b[0], a[1] - b[1], a[2] - b[2]]).sqrt()
}

fn angular_grid(n: usize) -> Result<Vec<([f64; 3], f64)>> {
    let text = include_str!("../third_party/xtb/src/solv/lebedev.f90");
    let block = text
        .split_once(&format!("pure subroutine ld{n:04}("))
        .ok_or_else(|| Gfn1Error::InvalidInput(format!("unsupported Lebedev size {n}")))?
        .1
        .split("end subroutine")
        .next()
        .unwrap();
    let (mut a, mut b, mut weight) = (0.0, 0.0, 0.0);
    let mut out = Vec::with_capacity(n);
    for line in block.lines().map(str::trim) {
        if line.starts_with("a =") {
            a = fortran_numbers(line)?[0];
        }
        if line.starts_with("b =") {
            b = fortran_numbers(line)?[0];
        }
        if line.starts_with("v =") {
            weight = fortran_numbers(line)?[0];
        }
        if let Some(rest) = line.strip_prefix("call genOh") {
            let base = match rest.as_bytes()[0] {
                b'1' => [1.0, 0.0, 0.0],
                b'2' => [0.0, 0.5_f64.sqrt(), 0.5_f64.sqrt()],
                b'3' => [1.0 / 3.0_f64.sqrt(); 3],
                b'4' => [a, a, (1.0 - 2.0 * a * a).sqrt()],
                b'5' => [a, (1.0 - a * a).sqrt(), 0.0],
                b'6' => [a, b, (1.0 - a * a - b * b).sqrt()],
                _ => return Err(Gfn1Error::InvalidInput("invalid angular orbit".into())),
            };
            let mut orbit = Vec::new();
            for p in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                for mask in 0..8 {
                    let point = std::array::from_fn(|k| {
                        if base[p[k]] == 0.0 {
                            0.0
                        } else {
                            base[p[k]] * if mask & (1 << k) == 0 { 1.0 } else { -1.0 }
                        }
                    });
                    if !orbit.contains(&point) {
                        orbit.push(point);
                    }
                }
            }
            out.extend(orbit.into_iter().map(|point| (point, weight)));
        }
    }
    if out.len() != n {
        return Err(Gfn1Error::InvalidInput(
            "Lebedev orbit count mismatch".into(),
        ));
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct SolvationEnergy {
    pub electrostatic: f64,
    pub hydrogen_bond: f64,
    pub sasa: f64,
    pub shift: f64,
}
impl SolvationEnergy {
    pub fn total(&self) -> f64 {
        self.electrostatic + self.hydrogen_bond + self.sasa + self.shift
    }
}

#[derive(Clone, Debug)]
pub struct SolvationCache {
    system: PeriodicSystem,
    options: SolvationOptions,
    params: Parameters,
    radii: Vec<f64>,
    grid: Vec<([f64; 3], f64)>,
    pub matrix: Matrix,
    pub born_radii: Vec<f64>,
    pub charge_correction: Vec<f64>,
    hb: Vec<f64>,
    sasa_energy: f64,
    shift: f64,
}

pub(crate) struct SolvationDirection {
    pub matrix: Vec<Jet1>,
    pub correction: Vec<Jet1>,
    pub sasa: Jet1,
    pub shift: f64,
}

impl SolvationDirection {
    pub fn potential(&self, q: &[f64]) -> Vec<Jet1> {
        let n = q.len();
        (0..n)
            .map(|i| {
                let mut potential = Jet1::default();
                for (j, &charge) in q.iter().enumerate() {
                    potential = potential
                        .add(&self.matrix[i * n + j].mul(&self.correction[j].add_scalar(charge)));
                }
                potential
            })
            .collect()
    }

    pub fn energy(&self, q: &[Jet1]) -> Jet1 {
        let q: Vec<_> = q
            .iter()
            .zip(&self.correction)
            .map(|(q, c)| q.add(c))
            .collect();
        let mut energy = self.sasa.add_scalar(self.shift);
        for (i, qi) in q.iter().enumerate() {
            for (j, qj) in q.iter().enumerate() {
                energy.add_scaled(&qi.mul(qj).mul(&self.matrix[i * q.len() + j]), 0.5);
            }
        }
        energy
    }
}

impl SolvationCache {
    pub fn options(&self) -> &SolvationOptions {
        &self.options
    }
    pub fn build(system: &PeriodicSystem, options: &SolvationOptions) -> Result<Self> {
        if system.lattice.is_some()
            || system.atoms.is_empty()
            || !options.temperature.is_finite()
            || options.temperature <= 0.0
            || system.atoms.iter().any(|a| {
                a.z == 0
                    || a.z > 86
                    || [a.position.x, a.position.y, a.position.z]
                        .iter()
                        .any(|x| !x.is_finite())
            })
        {
            return Err(Gfn1Error::InvalidInput("equilibrium solvation requires a finite molecular geometry, Z=1..86, and positive solvent T".into()));
        }
        let params = parameters(options)?;
        let source = include_str!("../third_party/xtb/src/param/vdwradd3.f90");
        let raw = source
            .split_once("aatoau * [&")
            .unwrap()
            .1
            .split(']')
            .next()
            .unwrap();
        let table = fortran_numbers(raw)?;
        let radii = system
            .atoms
            .iter()
            .map(|a| table[a.z as usize - 1] * AATOAU)
            .collect();
        let grid = angular_grid(options.grid_points)?;
        let shift = params.shift + state_shift(options, &params);
        let n = system.atoms.len();
        let mut cache = Self {
            system: system.clone(),
            options: options.clone(),
            params,
            radii,
            grid,
            matrix: Matrix::zeros(n, n),
            born_radii: vec![],
            charge_correction: vec![],
            hb: vec![],
            sasa_energy: 0.0,
            shift,
        };
        let (matrix, born, hb, sasa, correction) = cache.geometry(None)?;
        cache.matrix = Matrix::from_vec(n, n, matrix.iter().map(|x| x.0.value).collect())?;
        cache.born_radii = born.iter().map(|x| x.0.value).collect();
        cache.charge_correction = correction.iter().map(|x| x.0.value).collect();
        cache.hb = hb.iter().map(|x| x.0.value).collect();
        cache.sasa_energy = sasa.0.value;
        Ok(cache)
    }

    pub fn energy(&self, q: &[f64]) -> Result<SolvationEnergy> {
        let potential = self.potential(q)?;
        let corrected: Vec<f64> = q
            .iter()
            .zip(&self.charge_correction)
            .map(|(q, c)| q + c)
            .collect();
        let hb: f64 = corrected.iter().zip(&self.hb).map(|(q, h)| q * q * h).sum();
        Ok(SolvationEnergy {
            electrostatic: 0.5
                * corrected
                    .iter()
                    .zip(potential)
                    .map(|(q, v)| q * v)
                    .sum::<f64>()
                - hb,
            hydrogen_bond: hb,
            sasa: self.sasa_energy,
            shift: self.shift,
        })
    }
    pub fn potential(&self, q: &[f64]) -> Result<Vec<f64>> {
        let n = self.system.atoms.len();
        if q.len() != n || q.iter().any(|x| !x.is_finite()) {
            return Err(Gfn1Error::InvalidInput(
                "invalid solvation atomic charges".into(),
            ));
        }
        Ok((0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.matrix[(i, j)] * (q[j] + self.charge_correction[j]))
                    .sum()
            })
            .collect())
    }
    pub fn potential_geometry_derivatives(&self, q: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.potential(q)?;
        let n = q.len();
        (0..3 * n)
            .map(|c| {
                let (matrix, _, _, _, correction) = self.geometry(Some(c))?;
                Ok((0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| (matrix[i * n + j] * (q[j] + correction[j])).0.d1)
                            .sum()
                    })
                    .collect())
            })
            .collect()
    }
    /// Shell aggregation A: q_atom=A q_shell; return A^T B A.
    pub fn shell_kernel(&self, basis: &BasisSet) -> Matrix {
        let n = basis.shells.len();
        let mut out = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] = self.matrix[(basis.shells[i].atom_index, basis.shells[j].atom_index)];
            }
        }
        out
    }
    /// Explicit fixed-charge partial derivative. SCC stationarity and Pulay
    /// terms are supplied by the main gradient driver, not duplicated here.
    pub fn gradient(&self, q: &[f64]) -> Result<Vec<Vec3>> {
        self.potential(q)?;
        let n = q.len();
        let mut out = vec![Vec3::zero(); n];
        for c in 0..3 * n {
            let (matrix, _, _, sasa, correction) = self.geometry(Some(c))?;
            let mut d = sasa.0.d1;
            for i in 0..n {
                for j in 0..n {
                    d +=
                        (0.5 * (q[i] + correction[i]) * (q[j] + correction[j]) * matrix[i * n + j])
                            .0
                            .d1;
                }
            }
            match c % 3 {
                0 => out[c / 3].x = d,
                1 => out[c / 3].y = d,
                _ => out[c / 3].z = d,
            }
        }
        Ok(out)
    }

    pub(crate) fn directional_geometry(&self, direction: &[f64]) -> Result<SolvationDirection> {
        let (matrix, _, _, sasa, correction) = self.geometry_along(direction, true)?;
        Ok(SolvationDirection {
            matrix: matrix.into_iter().map(|x| x.0).collect(),
            correction: correction.into_iter().map(|x| x.0).collect(),
            sasa: sasa.0,
            shift: self.shift,
        })
    }

    fn geometry(&self, seed: Option<usize>) -> Result<(Vec<D>, Vec<D>, Vec<D>, D, Vec<D>)> {
        let mut direction = Vec::new();
        if let Some(seed) = seed {
            direction.resize(3 * self.system.atoms.len(), 0.);
            direction[seed] = 1.;
        }
        self.geometry_along(&direction, false)
    }

    fn geometry_along(
        &self,
        direction: &[f64],
        higher: bool,
    ) -> Result<(Vec<D>, Vec<D>, Vec<D>, D, Vec<D>)> {
        let check_branch = |value: D, cutoff: f64| -> Result<()> {
            if higher && (value.0.value - cutoff).abs() <= 1e-10 && value.0.d1 != 0. {
                return Err(Gfn1Error::InvalidInput(
                    "higher solvent derivatives require a smooth local Born/SASA branch at the reference geometry".into(),
                ));
            }
            Ok(())
        };
        let n = self.system.atoms.len();
        let p = &self.params;
        let xyz: Vec<[D; 3]> = self
            .system
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                std::array::from_fn(|k| {
                    D(Jet1 {
                        value: a.position.to_array()[k],
                        d1: direction.get(3 * i + k).copied().unwrap_or(0.),
                        ..Default::default()
                    })
                })
            })
            .collect();
        let mut distances = vec![D::c(0.0); n * n];
        for i in 0..n {
            for j in 0..i {
                let r = distance(xyz[i], xyz[j]);
                if r.0.value < 1e-10 {
                    return Err(Gfn1Error::InvalidInput(
                        "coincident solvation centers".into(),
                    ));
                }
                distances[i * n + j] = r;
                distances[j * n + i] = r;
            }
        }
        let mut born = vec![D::c(0.0); n];
        for i in 0..n {
            let mut psi = D::c(0.0);
            let ri = self.radii[i];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let r = distances[i * n + j];
                check_branch(r, 35.0 * AATOAU)?;
                if r.0.value >= 35.0 * AATOAU {
                    continue;
                }
                let rho = self.radii[j] * p.descreen[self.system.atoms[j].z as usize - 1];
                let ap = r + rho;
                let am = r - rho;
                check_branch(r, ri + rho)?;
                check_branch(ap, ri)?;
                if r.0.value >= ri + rho {
                    psi = psi + rho / (ap * am) + 0.5 * (am / ap).ln() / r;
                } else if ap.0.value > ri {
                    psi = psi + 1.0 / ri - 1.0 / ap
                        + 0.5 / r * (0.5 * am * (1.0 / ap - ap / (ri * ri)) - (ap / ri).ln());
                }
            }
            let sv = ri - p.offset;
            if sv <= 0.0 {
                return Err(Gfn1Error::InvalidInput(
                    "nonpositive offset solvent radius".into(),
                ));
            }
            let b = psi * (0.5 * sv);
            let arg = b * (1.0 + b * (4.85 * b - 0.8));
            born[i] = p.scale / (1.0 / sv - arg.tanh() / ri);
            if !born[i].0.value.is_finite() || born[i].0.value <= 0.0 {
                return Err(Gfn1Error::InvalidInput("nonpositive Born radius".into()));
            }
        }
        let mut hb = vec![D::c(0.0); n];
        let mut sasa = D::c(0.0);
        let sa: Vec<f64> = self.radii.iter().map(|r| r + p.probe).collect();
        let srcut = 2.0 * (WIDTH + sa.iter().copied().fold(0.0, f64::max)) + 2.0 * AATOAU;
        for i in 0..n {
            let r = sa[i];
            let radial = |x: f64| {
                (0.25 / WIDTH
                    - 3.0 / (4.0 * WIDTH.powi(3)) * (0.2 * x * x - 0.5 * x * r + r * r / 3.0))
                    * x.powi(3)
            };
            let wr = radial(r + WIDTH) - radial(r - WIDTH);
            let mut area = D::c(0.0);
            for &(point, weight) in &self.grid {
                let pos: [D; 3] = std::array::from_fn(|k| xyz[i][k] + r * point[k]);
                let mut exposure = D::c(1.0);
                for j in 0..n {
                    if i != j {
                        check_branch(distances[i * n + j], srcut)?;
                    }
                    if i == j
                        || distances[i * n + j].0.value >= srcut
                        || distances[i * n + j].0.value >= 35.0 * AATOAU
                    {
                        continue;
                    }
                    let dr = [pos[0] - xyz[j][0], pos[1] - xyz[j][1], pos[2] - xyz[j][2]];
                    let r2 = norm2(dr);
                    check_branch(r2, (sa[j] - WIDTH).powi(2))?;
                    check_branch(r2, (sa[j] + WIDTH).powi(2))?;
                    if r2.0.value <= (sa[j] - WIDTH).powi(2) {
                        exposure = D::c(0.0);
                        break;
                    }
                    if r2.0.value < (sa[j] + WIDTH).powi(2) {
                        let u = r2.sqrt() - sa[j];
                        exposure = exposure
                            * (0.5 + (3.0 / (4.0 * WIDTH) - u * u / (4.0 * WIDTH.powi(3))) * u);
                    }
                }
                check_branch(exposure, 1e-6)?;
                if exposure.0.value > 1e-6 {
                    area = area + exposure * (wr * weight);
                }
            }
            let z = self.system.atoms[i].z as usize - 1;
            sasa = sasa + area * p.tension[z];
            hb[i] = area * (p.hbond[z] / (r * r));
        }
        let abeta = if self.options.model == SolvationModel::Alpb {
            0.571412 / p.dielectric
        } else {
            0.0
        };
        let keps = (1.0 / p.dielectric - 1.0) / (1.0 + abeta);
        let shape = if abeta > 0.0 {
            let total: f64 = self.radii.iter().map(|r| r.powi(3)).sum();
            let mut center = [D::c(0.0); 3];
            for i in 0..n {
                for k in 0..3 {
                    center[k] = center[k] + xyz[i][k] * (self.radii[i].powi(3) / total);
                }
            }
            let mut inertia = [[D::c(0.0); 3]; 3];
            for i in 0..n {
                let r = std::array::from_fn(|k| xyz[i][k] - center[k]);
                let rr = norm2(r);
                for a in 0..3 {
                    for b in 0..3 {
                        inertia[a][b] = inertia[a][b]
                            + self.radii[i].powi(3)
                                * ((if a == b {
                                    rr + 0.4 * self.radii[i] * self.radii[i]
                                } else {
                                    D::c(0.0)
                                }) - r[a] * r[b]);
                    }
                }
            }
            let m = inertia;
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            keps * abeta / (det.powf(1.0 / 3.0) / (0.4 * total)).sqrt()
        } else {
            D::c(0.0)
        };
        let mut matrix = vec![shape; n * n];
        for i in 0..n {
            matrix[i * n + i] = matrix[i * n + i] + keps / born[i] + 2.0 * hb[i];
            for j in 0..i {
                let r = distances[i * n + j];
                let product = born[i] * born[j];
                let denom = match self.options.model {
                    SolvationModel::Gbsa => {
                        (r * r + product * (-r * r / (4.0 * product)).exp()).sqrt()
                    }
                    SolvationModel::Alpb => {
                        let ab = product.sqrt();
                        let arg = ab / (ab + r * (1.028 / 16.0));
                        r + ab * arg.powf(16.0)
                    }
                };
                matrix[i * n + j] = matrix[i * n + j] + keps / denom;
                matrix[j * n + i] = matrix[i * n + j];
            }
        }
        if matrix
            .iter()
            .any(|x| !x.0.value.is_finite() || !x.0.d1.is_finite())
            || !sasa.0.value.is_finite()
            || !sasa.0.d1.is_finite()
        {
            return Err(Gfn1Error::InvalidInput(
                "non-finite solvation geometry derivative".into(),
            ));
        }
        let correction = self.cm5(&distances)?;
        Ok((matrix, born, hb, sasa, correction))
    }

    fn cm5(&self, distances: &[D]) -> Result<Vec<D>> {
        let text = include_str!("../third_party/xtb/src/solv/cm5.f90");
        let rad = fortran_numbers(
            text.split_once(":: rad(mz) = (/")
                .unwrap()
                .1
                .split("/)")
                .next()
                .unwrap(),
        )?;
        let a0 = fortran_numbers(
            text.split_once(":: a0(mz) = (/")
                .unwrap()
                .1
                .split("/)")
                .next()
                .unwrap(),
        )?;
        let n = self.system.atoms.len();
        let mut correction = vec![D::c(0.0); n];
        for i in 0..n {
            for j in 0..i {
                let zi = self.system.atoms[i].z as usize;
                let zj = self.system.atoms[j].z as usize;
                if zi == zj {
                    continue;
                }
                let (lo, hi) = if zi < zj { (zi, zj) } else { (zj, zi) };
                let dij = match (lo, hi) {
                    (1, 6) => 0.0502,
                    (1, 7) => 0.1747,
                    (1, 8) => 0.1671,
                    (6, 7) => 0.0556,
                    (6, 8) => 0.0234,
                    (7, 8) => -0.0346,
                    _ => a0[lo - 1] - a0[hi - 1],
                };
                let transfer = (-(2.4740 / AATOAU)
                    * (distances[i * n + j] - (rad[zi - 1] + rad[zj - 1]) * AATOAU))
                    .exp()
                    * dij
                    * if zi < zj { 1.0 } else { -1.0 };
                correction[i] = correction[i] + transfer;
                correction[j] = correction[j] - transfer;
            }
        }
        Ok(correction)
    }
}

fn state_shift(options: &SolvationOptions, p: &Parameters) -> f64 {
    let log_volume = (24.79 * options.temperature / 298.15).ln();
    match options.state {
        SolutionState::Gsolv => 0.0,
        SolutionState::Bar1Mol => options.temperature * KB_HARTREE_PER_K * log_volume,
        // Match the pinned state.f90 + model.f90 density convention exactly.
        SolutionState::Reference => {
            options.temperature
                * KB_HARTREE_PER_K
                * (log_volume + (p.density / p.mass * 8.92388e-2 / 1e-3 / p.mass).ln())
        }
    }
}
