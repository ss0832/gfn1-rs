// SPDX-License-Identifier: GPL-3.0-or-later
use crate::error::{Gfn1Error, Result};
use crate::lattice::Lattice;
use crate::math::{Mat3, Vec3};
use std::fs;
use std::path::Path;

pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_625_770_2;

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    pub position: Vec3,
}

#[derive(Clone, Debug)]
pub struct PeriodicSystem {
    pub atoms: Vec<Atom>,
    pub lattice: Option<Lattice>,
    pub charge: f64,
}

impl PeriodicSystem {
    pub fn new(atoms: Vec<Atom>, lattice: Option<Lattice>) -> Self {
        Self {
            atoms,
            lattice,
            charge: 0.0,
        }
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn from_xyz_file(
        path: impl AsRef<Path>,
        charge: f64,
        coordinates_are_bohr: bool,
    ) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?, charge, coordinates_are_bohr)
    }

    pub fn from_xyz_str(text: &str, charge: f64, coordinates_are_bohr: bool) -> Result<Self> {
        let mut lines = text.lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| Gfn1Error::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            Gfn1Error::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let comment = lines.next().unwrap_or_default();
        let mut lattice = parse_extxyz_lattice(comment)?;
        let scale = if coordinates_are_bohr {
            1.0
        } else {
            ANGSTROM_TO_BOHR
        };
        let mut atoms = Vec::with_capacity(natoms);
        for idx in 0..natoms {
            let line_no = idx + 3;
            let line = lines
                .next()
                .ok_or_else(|| Gfn1Error::InvalidInput(format!("XYZ ended before atom {idx}")))?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 4 {
                return Err(Gfn1Error::InvalidInput(format!(
                    "XYZ line {line_no} has fewer than 4 fields"
                )));
            }
            let z = symbol_to_z(parts[0]).ok_or_else(|| {
                Gfn1Error::InvalidInput(format!("unknown element on line {line_no}: {}", parts[0]))
            })?;
            atoms.push(Atom {
                z,
                position: Vec3::new(
                    parse_f64(parts[1], line_no)?,
                    parse_f64(parts[2], line_no)?,
                    parse_f64(parts[3], line_no)?,
                ) * scale,
            });
        }
        if !coordinates_are_bohr {
            if let Some(lat) = lattice {
                let cell = Mat3::from_columns(
                    lat.cell.col[0] * ANGSTROM_TO_BOHR,
                    lat.cell.col[1] * ANGSTROM_TO_BOHR,
                    lat.cell.col[2] * ANGSTROM_TO_BOHR,
                );
                lattice = Some(Lattice::new(cell, lat.periodic)?);
            }
        }
        Ok(Self {
            atoms,
            lattice,
            charge,
        })
    }
    pub fn convert_angstrom_to_bohr(&mut self) {
        for atom in &mut self.atoms {
            atom.position = atom.position * ANGSTROM_TO_BOHR;
        }
        if let Some(lat) = self.lattice {
            let cell = Mat3::from_columns(
                lat.cell.col[0] * ANGSTROM_TO_BOHR,
                lat.cell.col[1] * ANGSTROM_TO_BOHR,
                lat.cell.col[2] * ANGSTROM_TO_BOHR,
            );
            self.lattice =
                Some(Lattice::new(cell, lat.periodic).expect("scaled lattice remains valid"));
        }
    }
    pub fn wrap_positions(&mut self) {
        if let Some(lattice) = &self.lattice {
            for atom in &mut self.atoms {
                atom.position = lattice.wrap_cart(atom.position);
            }
        }
    }
    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

pub type Molecule = PeriodicSystem;

pub fn parse_charges_file(path: impl AsRef<Path>, natoms: usize) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path)?;
    let mut charges = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let token = trimmed.split_whitespace().next().unwrap();
        charges.push(token.parse::<f64>().map_err(|_| Gfn1Error::Parse {
            line: idx + 1,
            message: format!("invalid charge value: {token}"),
        })?);
    }
    if charges.len() != natoms {
        return Err(Gfn1Error::InvalidInput(format!(
            "charge file has {} values, but system has {natoms} atoms",
            charges.len()
        )));
    }
    Ok(charges)
}

pub fn symbol_to_z(sym: &str) -> Option<u8> {
    if let Ok(z) = sym.parse::<u8>() {
        if (1..=86).contains(&z) {
            return Some(z);
        }
    }
    let s = normalize_symbol(sym);
    ELEMENTS
        .iter()
        .position(|&x| x == s.as_str())
        .map(|i| i as u8)
}

pub fn z_to_symbol(z: u8) -> Option<&'static str> {
    ELEMENTS.get(z as usize).copied().filter(|s| !s.is_empty())
}

fn normalize_symbol(sym: &str) -> String {
    let mut chars = sym.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.push(first.to_ascii_uppercase());
            for c in chars {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        None => String::new(),
    }
}
fn parse_f64(token: &str, line: usize) -> Result<f64> {
    token.parse::<f64>().map_err(|_| Gfn1Error::Parse {
        line,
        message: format!("invalid floating point value: {token}"),
    })
}
fn parse_extxyz_lattice(comment: &str) -> Result<Option<Lattice>> {
    let Some(start) = comment.find("Lattice=\"") else {
        return Ok(None);
    };
    let rest = &comment[start + "Lattice=\"".len()..];
    let Some(end) = rest.find('"') else {
        return Err(Gfn1Error::InvalidInput(
            "unterminated Lattice=\"...\" field in XYZ comment".to_string(),
        ));
    };
    let values = rest[..end]
        .split_whitespace()
        .map(|v| v.parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Gfn1Error::InvalidInput("invalid number in XYZ Lattice field".to_string()))?;
    if values.len() != 9 {
        return Err(Gfn1Error::InvalidInput(format!(
            "XYZ Lattice field needs 9 numbers, got {}",
            values.len()
        )));
    }
    let periodic = parse_extxyz_pbc(comment)?.unwrap_or([true, true, true]);
    Ok(Some(Lattice::from_vectors(
        Vec3::new(values[0], values[1], values[2]),
        Vec3::new(values[3], values[4], values[5]),
        Vec3::new(values[6], values[7], values[8]),
        periodic,
    )?))
}

fn parse_extxyz_pbc(comment: &str) -> Result<Option<[bool; 3]>> {
    let marker = comment.find("pbc=\"").or_else(|| comment.find("PBC=\""));
    let Some(start) = marker else {
        return Ok(None);
    };
    let rest = &comment[start + 5..];
    let Some(end) = rest.find('"') else {
        return Err(Gfn1Error::InvalidInput(
            "unterminated pbc=\"...\" field in XYZ comment".to_string(),
        ));
    };
    let tokens = rest[..end].split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 3 {
        return Err(Gfn1Error::InvalidInput(format!(
            "XYZ pbc field needs 3 boolean tokens, got {}",
            tokens.len()
        )));
    }
    let mut periodic = [false; 3];
    for (i, token) in tokens.iter().enumerate() {
        let t = token.to_ascii_lowercase();
        periodic[i] = match t.as_str() {
            "t" | "true" | "1" | ".true." | "yes" | "y" => true,
            "f" | "false" | "0" | ".false." | "no" | "n" => false,
            _ => {
                return Err(Gfn1Error::InvalidInput(format!(
                    "invalid XYZ pbc token: {token}"
                )))
            }
        };
    }
    Ok(Some(periodic))
}

const ELEMENTS: [&str; 87] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
];
