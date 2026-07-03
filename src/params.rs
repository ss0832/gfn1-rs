// SPDX-License-Identifier: GPL-3.0-or-later

use crate::constants::EV_TO_HARTREE;
use crate::error::{Gfn1Error, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

pub const GFN1_PARAM_ENV: &str = "GFN1_XTB_PARAM";
pub const GFN1_D3_REFERENCE_ENV: &str = "GFN1_D3_REFERENCE";

pub fn resolve_param_path(explicit: Option<&str>) -> Result<String> {
    if let Some(path) = explicit {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    env::var(GFN1_PARAM_ENV).map_err(|_| {
        Gfn1Error::InvalidInput(format!(
            "missing parameter file; pass --param FILE or set {GFN1_PARAM_ENV}"
        ))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AngularMomentum {
    S,
    P,
    D,
    F,
    G,
}

impl AngularMomentum {
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_lowercase() {
            's' => Some(Self::S),
            'p' => Some(Self::P),
            'd' => Some(Self::D),
            'f' => Some(Self::F),
            'g' => Some(Self::G),
            _ => None,
        }
    }

    pub fn as_index(self) -> usize {
        match self {
            Self::S => 0,
            Self::P => 1,
            Self::D => 2,
            Self::F => 3,
            Self::G => 4,
        }
    }

    pub fn degeneracy(self) -> usize {
        match self {
            Self::S => 1,
            Self::P => 3,
            Self::D => 5,
            Self::F => 7,
            Self::G => 9,
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::S => "S",
            Self::P => "P",
            Self::D => "D",
            Self::F => "F",
            Self::G => "G",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShellParam {
    pub label: String,
    pub principal_n: u8,
    pub l: AngularMomentum,
    pub level_ev: f64,
    pub slater: f64,
    pub poly_raw: f64,
    pub lpar_raw: f64,
}

impl ShellParam {
    pub fn level_hartree(&self) -> f64 {
        self.level_ev * EV_TO_HARTREE
    }
}

#[derive(Clone, Debug)]
pub struct ElementParam {
    pub z: u8,
    pub ao_raw: String,
    pub shells: Vec<ShellParam>,
    pub gam: f64,
    pub gam3_raw: f64,
    pub repa: f64,
    pub repb: f64,
    pub raw: HashMap<String, Vec<f64>>,
}

impl ElementParam {
    pub fn gam3_model(&self) -> f64 {
        self.gam3_raw * 0.1
    }

    pub fn shell_hardness(&self, l: AngularMomentum) -> f64 {
        let lpar = self
            .shells
            .iter()
            .find(|sh| sh.l == l)
            .map(|sh| sh.lpar_raw)
            .unwrap_or(0.0);
        self.gam * (1.0 + 0.1 * lpar)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gfn1Parameters {
    pub source_path: Option<String>,
    pub info: HashMap<String, String>,
    pub globpar: HashMap<String, f64>,
    pub elements: HashMap<u8, ElementParam>,
    pub pairpar: HashMap<(u8, u8), f64>,
}

impl Gfn1Parameters {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut params = Self::from_str(&fs::read_to_string(path)?)?;
        params.source_path = Some(path.to_string_lossy().to_string());
        Ok(params)
    }

    pub fn from_str(text: &str) -> Result<Self> {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Section {
            None,
            Info,
            Globpar,
            Pairpar,
            Element,
        }

        let mut out = Self::default();
        let mut section = Section::None;
        let mut builder: Option<ElementBuilder> = None;

        for (idx, raw_line) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("$info") {
                finish_element(&mut out, &mut builder, line_no)?;
                section = Section::Info;
                continue;
            }
            if line.starts_with("$globpar") {
                finish_element(&mut out, &mut builder, line_no)?;
                section = Section::Globpar;
                continue;
            }
            if line.starts_with("$pairpar") {
                finish_element(&mut out, &mut builder, line_no)?;
                section = Section::Pairpar;
                continue;
            }
            if line.starts_with("$Z=") {
                finish_element(&mut out, &mut builder, line_no)?;
                builder = Some(ElementBuilder::new(parse_z_header(line, line_no)?));
                section = Section::Element;
                continue;
            }
            if line.starts_with("$end") {
                finish_element(&mut out, &mut builder, line_no)?;
                section = Section::None;
                continue;
            }

            match section {
                Section::Info => parse_info_line(&mut out.info, line),
                Section::Globpar => parse_globpar_line(&mut out.globpar, line, line_no)?,
                Section::Pairpar => parse_pairpar_line(&mut out.pairpar, line, line_no)?,
                Section::Element => builder
                    .as_mut()
                    .ok_or_else(|| Gfn1Error::Parse {
                        line: line_no,
                        message: "element data encountered before $Z header".to_string(),
                    })?
                    .parse_line(line, line_no)?,
                Section::None => {
                    return Err(Gfn1Error::Parse {
                        line: line_no,
                        message: format!("data outside a section: {line}"),
                    });
                }
            }
        }
        finish_element(&mut out, &mut builder, text.lines().count())?;
        out.validate_gfn1()?;
        Ok(out)
    }

    pub fn element(&self, z: u8) -> Result<&ElementParam> {
        self.elements.get(&z).ok_or(Gfn1Error::MissingElement(z))
    }

    pub fn required_global(&self, key: &str) -> Result<f64> {
        self.globpar
            .get(&key.to_ascii_lowercase())
            .copied()
            .ok_or_else(|| Gfn1Error::MissingGlobal(key.to_string()))
    }

    pub fn global(&self, key: &str, default: f64) -> f64 {
        self.globpar
            .get(&key.to_ascii_lowercase())
            .copied()
            .unwrap_or(default)
    }

    pub fn pair_scaling(&self, za: u8, zb: u8) -> f64 {
        let key = if za <= zb { (za, zb) } else { (zb, za) };
        if let Some(value) = self.pairpar.get(&key) {
            return *value;
        }
        default_gfn1_pair_scaling(za, zb)
    }

    pub fn k_shell(&self, l: AngularMomentum) -> f64 {
        match l {
            AngularMomentum::S => self.global("ks", 1.85),
            AngularMomentum::P => self.global("kp", 2.25),
            AngularMomentum::D => self.global("kd", 2.00),
            AngularMomentum::F | AngularMomentum::G => self.global("kd", 2.00),
        }
    }

    pub fn k_scale(&self, li: AngularMomentum, lj: AngularMomentum, zi: u8, zj: u8) -> f64 {
        let base = if (li == AngularMomentum::S && lj == AngularMomentum::P)
            || (li == AngularMomentum::P && lj == AngularMomentum::S)
        {
            self.global("ksp", 2.08)
        } else {
            0.5 * (self.k_shell(li) + self.k_shell(lj))
        };
        base * self.pair_scaling(zi, zj)
    }

    /// Serialize back into the `param_gfn1-xtb.txt` text format.
    ///
    /// Output is deterministic (keys are emitted in a canonical, then sorted,
    /// order) and uses shortest round-trippable float formatting, so
    /// `from_str(&p.to_param_string())` reproduces `p`'s parsed values exactly.
    /// Free-form comments and the date stamps on `$Z=` headers are not part of
    /// [`Gfn1Parameters`] and are therefore not preserved.
    pub fn to_param_string(&self) -> String {
        let mut out = String::new();

        out.push_str("$info\n");
        for key in ["level", "name", "doi"] {
            if let Some(value) = self.info.get(key) {
                out.push_str(&format!("{key} {value}\n"));
            }
        }
        let mut extra_info = self
            .info
            .keys()
            .filter(|k| !matches!(k.as_str(), "level" | "name" | "doi"))
            .cloned()
            .collect::<Vec<_>>();
        extra_info.sort();
        for key in extra_info {
            out.push_str(&format!("{key} {}\n", self.info[&key]));
        }

        out.push_str("$globpar\n");
        for key in GLOBPAR_CANONICAL_ORDER {
            if let Some(value) = self.globpar.get(*key) {
                out.push_str(&format!("{key} {}\n", fmt_param_float(*value)));
            }
        }
        let mut extra_glob = self
            .globpar
            .keys()
            .filter(|k| !GLOBPAR_CANONICAL_ORDER.contains(&k.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        extra_glob.sort();
        for key in extra_glob {
            out.push_str(&format!("{key} {}\n", fmt_param_float(self.globpar[&key])));
        }
        out.push_str("$end\n");

        if !self.pairpar.is_empty() {
            out.push_str("$pairpar\n");
            let mut pairs = self.pairpar.iter().collect::<Vec<_>>();
            pairs.sort_by_key(|((za, zb), _)| (*za, *zb));
            for ((za, zb), value) in pairs {
                out.push_str(&format!("{za} {zb} {}\n", fmt_param_float(*value)));
            }
            out.push_str("$end\n");
        }

        let mut zs = self.elements.keys().copied().collect::<Vec<_>>();
        zs.sort_unstable();
        for z in zs {
            let elem = &self.elements[&z];
            out.push_str(&format!("$Z= {z}\n"));
            out.push_str(&format!(" ao={}\n", elem.ao_raw));
            for key in element_raw_key_order(&elem.raw) {
                let values = &elem.raw[&key];
                // The reference file lowercases `lev`/`exp`; the parser
                // uppercases on read, so either case round-trips.
                let written_key = match key.as_str() {
                    "LEV" => "lev".to_string(),
                    "EXP" => "exp".to_string(),
                    other => other.to_string(),
                };
                let joined = values
                    .iter()
                    .map(|v| fmt_param_float(*v))
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push_str(&format!(" {written_key}= {joined}\n"));
            }
            out.push_str("$end\n");
        }

        out
    }

    /// Write the parameters to `path` in `param_gfn1-xtb.txt` format.
    pub fn write_param_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path.as_ref(), self.to_param_string())?;
        Ok(())
    }

    /// Read the scalar value addressed by `target` (see [`ParameterTarget`]).
    pub fn parameter_value(&self, target: &ParameterTarget) -> Result<f64> {
        match target {
            ParameterTarget::Global(key) => self.required_global(key),
            ParameterTarget::Pair(za, zb) => {
                let key = if za <= zb { (*za, *zb) } else { (*zb, *za) };
                self.pairpar.get(&key).copied().ok_or_else(|| {
                    Gfn1Error::InvalidInput(format!("pair parameter {za}-{zb} is not present"))
                })
            }
            ParameterTarget::Element { z, key, index } => {
                let elem = self.element(*z)?;
                let values = elem.raw.get(key).ok_or_else(|| {
                    Gfn1Error::InvalidInput(format!("element {z} has no `{key}` entry"))
                })?;
                values.get(*index).copied().ok_or_else(|| {
                    Gfn1Error::InvalidInput(format!(
                        "element {z} `{key}` has no index {index} (len {})",
                        values.len()
                    ))
                })
            }
        }
    }

    /// Return a clone with the scalar addressed by `target` set to `value`.
    ///
    /// The clone is re-derived through the text round-trip so that all derived
    /// fields (shell levels, hardness, repulsion, ...) stay consistent with the
    /// raw table — this is what the finite-difference parameter derivatives use.
    pub fn with_parameter(&self, target: &ParameterTarget, value: f64) -> Result<Self> {
        let mut clone = self.clone();
        match target {
            ParameterTarget::Global(key) => {
                let key = key.to_ascii_lowercase();
                if !clone.globpar.contains_key(&key) {
                    return Err(Gfn1Error::InvalidInput(format!(
                        "global parameter `{key}` is not present"
                    )));
                }
                clone.globpar.insert(key, value);
            }
            ParameterTarget::Pair(za, zb) => {
                let key = if za <= zb { (*za, *zb) } else { (*zb, *za) };
                if !clone.pairpar.contains_key(&key) {
                    return Err(Gfn1Error::InvalidInput(format!(
                        "pair parameter {za}-{zb} is not present"
                    )));
                }
                clone.pairpar.insert(key, value);
            }
            ParameterTarget::Element { z, key, index } => {
                let elem = clone
                    .elements
                    .get_mut(z)
                    .ok_or(Gfn1Error::MissingElement(*z))?;
                let values = elem.raw.get_mut(key).ok_or_else(|| {
                    Gfn1Error::InvalidInput(format!("element {z} has no `{key}` entry"))
                })?;
                if *index >= values.len() {
                    return Err(Gfn1Error::InvalidInput(format!(
                        "element {z} `{key}` has no index {index} (len {})",
                        values.len()
                    )));
                }
                values[*index] = value;
            }
        }
        // Re-derive through the canonical text form so derived fields are rebuilt.
        Self::from_str(&clone.to_param_string())
    }

    fn validate_gfn1(&self) -> Result<()> {
        let level = self.info.get("level").map(String::as_str);
        let name = self.info.get("name").map(|s| s.to_ascii_lowercase());
        if level != Some("1") && !name.as_deref().unwrap_or("").contains("gfn1") {
            return Err(Gfn1Error::InvalidInput(
                "parameter file is not GFN1-xTB; expected `$info level 1` or a GFN1 name"
                    .to_string(),
            ));
        }
        for key in [
            "ks",
            "kp",
            "kd",
            "ksp",
            "kdiff",
            "enscale",
            "ipeashift",
            "cns",
            "cnp",
            "cnd1",
            "cnd2",
            "alphaj",
            "a1",
            "a2",
            "s8",
            "s9",
            "kexp",
            "kexplight",
            "xbdamp",
            "xbrad",
        ] {
            self.required_global(key)?;
        }
        Ok(())
    }
}

/// Canonical emission order for global parameters (matches the reference file).
const GLOBPAR_CANONICAL_ORDER: &[&str] = &[
    "ks",
    "kp",
    "kd",
    "ksp",
    "kdiff",
    "enscale",
    "ipeashift",
    "cns",
    "cnp",
    "cnd1",
    "cnd2",
    "alphaj",
    "a1",
    "a2",
    "s8",
    "s9",
    "kexp",
    "kexplight",
    "xbdamp",
    "xbrad",
];

/// Shortest round-trippable float formatting (`from_str` reproduces the bits).
fn fmt_param_float(value: f64) -> String {
    format!("{value}")
}

fn element_raw_key_order(raw: &HashMap<String, Vec<f64>>) -> Vec<String> {
    const CANON: &[&str] = &[
        "LEV", "EXP", "GAM", "GAM3", "REPA", "REPB", "POLYS", "POLYP", "POLYD", "POLYF", "LPARS",
        "LPARP", "LPARD", "LPARF",
    ];
    let mut out = Vec::new();
    for key in CANON {
        if raw.contains_key(*key) {
            out.push((*key).to_string());
        }
    }
    let mut extra = raw
        .keys()
        .filter(|k| !CANON.contains(&k.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    extra.sort();
    out.extend(extra);
    out
}

/// Addresses a single scalar inside a [`Gfn1Parameters`] table.
///
/// String form (used by the CLI and `param_deriv`):
/// - `glob:<key>` — a global parameter (e.g. `glob:ks`).
/// - `pair:<ZA>:<ZB>` — a pair scaling entry (e.g. `pair:1:6`).
/// - `elem:<Z>:<KEY>[:<idx>]` — an element entry; `idx` selects within a vector
///   value such as `LEV`/`EXP` and defaults to 0 (e.g. `elem:6:LEV:1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterTarget {
    Global(String),
    Pair(u8, u8),
    Element { z: u8, key: String, index: usize },
}

impl ParameterTarget {
    pub fn parse(spec: &str) -> Result<Self> {
        let parts = spec.trim().split(':').collect::<Vec<_>>();
        match parts.as_slice() {
            ["glob", key] | ["global", key] => Ok(Self::Global(key.to_ascii_lowercase())),
            ["pair", za, zb] => Ok(Self::Pair(parse_u8_target(za)?, parse_u8_target(zb)?)),
            ["elem", z, key] => Ok(Self::Element {
                z: parse_u8_target(z)?,
                key: key.to_ascii_uppercase(),
                index: 0,
            }),
            ["elem", z, key, idx] => Ok(Self::Element {
                z: parse_u8_target(z)?,
                key: key.to_ascii_uppercase(),
                index: parse_usize_target(idx)?,
            }),
            _ => Err(Gfn1Error::InvalidInput(format!(
                "invalid parameter target `{spec}` (expected glob:KEY, pair:ZA:ZB, or elem:Z:KEY[:idx])"
            ))),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Global(key) => format!("glob:{key}"),
            Self::Pair(za, zb) => format!("pair:{za}:{zb}"),
            Self::Element { z, key, index } => format!("elem:{z}:{key}:{index}"),
        }
    }
}

fn parse_u8_target(text: &str) -> Result<u8> {
    text.trim().parse::<u8>().map_err(|_| {
        Gfn1Error::InvalidInput(format!("invalid element/atomic number `{text}` in target"))
    })
}

fn parse_usize_target(text: &str) -> Result<usize> {
    text.trim()
        .parse::<usize>()
        .map_err(|_| Gfn1Error::InvalidInput(format!("invalid index `{text}` in target")))
}

fn default_gfn1_pair_scaling(za: u8, zb: u8) -> f64 {
    fn d_row(z: u8) -> Option<f64> {
        if (21..30).contains(&z) {
            Some(1.1)
        } else if (39..48).contains(&z) {
            Some(1.2)
        } else if (57..80).contains(&z) {
            Some(1.2)
        } else {
            None
        }
    }
    match (d_row(za), d_row(zb)) {
        (Some(a), Some(b)) => 0.5 * (a + b),
        _ => 1.0,
    }
}

fn finish_element(
    out: &mut Gfn1Parameters,
    builder: &mut Option<ElementBuilder>,
    line_no: usize,
) -> Result<()> {
    if let Some(builder) = builder.take() {
        let element = builder.finish(line_no)?;
        out.elements.insert(element.z, element);
    }
    Ok(())
}

fn parse_z_header(line: &str, line_no: usize) -> Result<u8> {
    let rest = line
        .strip_prefix("$Z=")
        .ok_or_else(|| Gfn1Error::Parse {
            line: line_no,
            message: "invalid element header".to_string(),
        })?
        .trim();
    rest.split_whitespace()
        .next()
        .ok_or_else(|| Gfn1Error::Parse {
            line: line_no,
            message: "missing element number".to_string(),
        })?
        .parse::<u8>()
        .map_err(|_| Gfn1Error::Parse {
            line: line_no,
            message: "invalid element number".to_string(),
        })
}

fn parse_info_line(info: &mut HashMap<String, String>, line: &str) {
    let mut parts = line.splitn(2, char::is_whitespace);
    if let Some(key) = parts.next() {
        let value = parts.next().unwrap_or("").trim();
        info.insert(key.to_ascii_lowercase(), value.to_string());
    }
}

fn parse_globpar_line(
    globpar: &mut HashMap<String, f64>,
    line: &str,
    line_no: usize,
) -> Result<()> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 2 {
        return Err(Gfn1Error::Parse {
            line: line_no,
            message: "global parameter line must be `key value`".to_string(),
        });
    }
    globpar.insert(
        fields[0].to_ascii_lowercase(),
        parse_f64(fields[1], line_no)?,
    );
    Ok(())
}

fn parse_pairpar_line(
    pairpar: &mut HashMap<(u8, u8), f64>,
    line: &str,
    line_no: usize,
) -> Result<()> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 3 {
        return Err(Gfn1Error::Parse {
            line: line_no,
            message: "pair parameter line must be `ZA ZB value`".to_string(),
        });
    }
    let za = parse_u8(fields[0], line_no)?;
    let zb = parse_u8(fields[1], line_no)?;
    let key = if za <= zb { (za, zb) } else { (zb, za) };
    pairpar.insert(key, parse_f64(fields[2], line_no)?);
    Ok(())
}

#[derive(Clone, Debug)]
struct ElementBuilder {
    z: u8,
    ao_raw: Option<String>,
    raw: HashMap<String, Vec<f64>>,
}

impl ElementBuilder {
    fn new(z: u8) -> Self {
        Self {
            z,
            ao_raw: None,
            raw: HashMap::new(),
        }
    }

    fn parse_line(&mut self, line: &str, line_no: usize) -> Result<()> {
        if let Some(rest) = line.strip_prefix("ao=") {
            self.ao_raw = Some(rest.trim().to_string());
            return Ok(());
        }
        let Some((key, values)) = line.split_once('=') else {
            return Err(Gfn1Error::Parse {
                line: line_no,
                message: "element parameter line must contain `=`".to_string(),
            });
        };
        let values = values
            .split_whitespace()
            .map(|v| parse_f64(v, line_no))
            .collect::<Result<Vec<_>>>()?;
        self.raw.insert(key.trim().to_ascii_uppercase(), values);
        Ok(())
    }

    fn finish(self, line_no: usize) -> Result<ElementParam> {
        let ao_raw = self.ao_raw.ok_or_else(|| Gfn1Error::Parse {
            line: line_no,
            message: format!("Z={} is missing ao= entry", self.z),
        })?;
        let labels = parse_ao_labels(&ao_raw, line_no)?;
        let lev = required_vec(&self.raw, "LEV", line_no, self.z)?;
        let exp = required_vec(&self.raw, "EXP", line_no, self.z)?;
        if lev.len() != labels.len() || exp.len() != labels.len() {
            return Err(Gfn1Error::Parse {
                line: line_no,
                message: format!(
                    "Z={} has {} AO labels but {} LEV and {} EXP values",
                    self.z,
                    labels.len(),
                    lev.len(),
                    exp.len()
                ),
            });
        }
        let gam = scalar(&self.raw, "GAM", line_no, self.z)?;
        let gam3_raw = optional_scalar(&self.raw, "GAM3", 0.0);
        let repa = scalar(&self.raw, "REPA", line_no, self.z)?;
        let repb = scalar(&self.raw, "REPB", line_no, self.z)?;

        let mut shells = Vec::with_capacity(labels.len());
        for (idx, (label, principal_n, l)) in labels.into_iter().enumerate() {
            let suffix = l.suffix();
            shells.push(ShellParam {
                label,
                principal_n,
                l,
                level_ev: lev[idx],
                slater: exp[idx],
                poly_raw: optional_scalar(&self.raw, &format!("POLY{suffix}"), 0.0),
                lpar_raw: optional_scalar(&self.raw, &format!("LPAR{suffix}"), 0.0),
            });
        }
        Ok(ElementParam {
            z: self.z,
            ao_raw,
            shells,
            gam,
            gam3_raw,
            repa,
            repb,
            raw: self.raw,
        })
    }
}

fn parse_ao_labels(raw: &str, line_no: usize) -> Result<Vec<(String, u8, AngularMomentum)>> {
    let mut out = Vec::new();
    let chars = raw.trim().chars().collect::<Vec<_>>();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        if !chars[i].is_ascii_digit() {
            return Err(Gfn1Error::Parse {
                line: line_no,
                message: format!("invalid AO label in `{raw}`"),
            });
        }
        let mut n_text = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            n_text.push(chars[i]);
            i += 1;
        }
        if i >= chars.len() {
            return Err(Gfn1Error::Parse {
                line: line_no,
                message: format!("AO label missing angular momentum in `{raw}`"),
            });
        }
        let l_char = chars[i];
        i += 1;
        let l = AngularMomentum::from_char(l_char).ok_or_else(|| Gfn1Error::Parse {
            line: line_no,
            message: format!("unsupported angular momentum `{l_char}`"),
        })?;
        let principal_n = n_text.parse::<u8>().map_err(|_| Gfn1Error::Parse {
            line: line_no,
            message: format!("invalid principal quantum number `{n_text}`"),
        })?;
        out.push((
            format!("{principal_n}{}", l_char.to_ascii_lowercase()),
            principal_n,
            l,
        ));
    }
    Ok(out)
}

fn required_vec<'a>(
    raw: &'a HashMap<String, Vec<f64>>,
    key: &str,
    line_no: usize,
    z: u8,
) -> Result<&'a [f64]> {
    raw.get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| Gfn1Error::Parse {
            line: line_no,
            message: format!("Z={z} is missing {key}"),
        })
}

fn scalar(raw: &HashMap<String, Vec<f64>>, key: &str, line_no: usize, z: u8) -> Result<f64> {
    let values = required_vec(raw, key, line_no, z)?;
    values.first().copied().ok_or_else(|| Gfn1Error::Parse {
        line: line_no,
        message: format!("Z={z} has empty {key}"),
    })
}

fn optional_scalar(raw: &HashMap<String, Vec<f64>>, key: &str, default: f64) -> f64 {
    raw.get(key)
        .and_then(|values| values.first())
        .copied()
        .unwrap_or(default)
}

fn parse_f64(text: &str, line_no: usize) -> Result<f64> {
    text.parse::<f64>().map_err(|_| Gfn1Error::Parse {
        line: line_no,
        message: format!("invalid float `{text}`"),
    })
}

fn parse_u8(text: &str, line_no: usize) -> Result<u8> {
    text.parse::<u8>().map_err(|_| Gfn1Error::Parse {
        line: line_no,
        message: format!("invalid integer `{text}`"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r#"
$info
level 1
name GFN1-xTB
$globpar
ks 1.85
kp 2.25
kd 2.0
ksp 2.08
kdiff 2.85
enscale -0.7
ipeashift 1.78069
cns 0.6
cnp -0.3
cnd1 -0.5
cnd2 0.5
alphaj 2.0
a1 0.63
a2 5.0
s8 2.4
s9 0.0
kexp 1.5
kexplight 1.5
xbdamp 0.44
xbrad 1.3
$end
$pairpar
1 1 0.96
$end
$Z= 1
ao=1s2s
lev= -10.0 -1.0
exp= 1.2 2.0
GAM= 0.47
REPA= 2.2
REPB= 1.1
$end
"#;

    #[test]
    fn parses_minimal_gfn1() {
        let p = Gfn1Parameters::from_str(MINI).unwrap();
        assert_eq!(p.element(1).unwrap().shells.len(), 2);
        assert_eq!(p.pair_scaling(1, 1), 0.96);
    }

    #[test]
    fn rejects_non_gfn1() {
        let bad = MINI
            .replace("level 1", "level 2")
            .replace("GFN1-xTB", "GFN2-xTB");
        assert!(Gfn1Parameters::from_str(&bad).is_err());
    }

    #[test]
    fn round_trips_through_text() {
        let p = Gfn1Parameters::from_str(MINI).unwrap();
        let text = p.to_param_string();
        let q = Gfn1Parameters::from_str(&text).unwrap();
        assert_eq!(p.globpar, q.globpar);
        assert_eq!(p.pairpar, q.pairpar);
        assert_eq!(p.info.get("level"), q.info.get("level"));
        assert_eq!(p.elements.len(), q.elements.len());
        for (z, ep) in &p.elements {
            let eq = q.element(*z).unwrap();
            assert_eq!(ep.ao_raw.trim(), eq.ao_raw.trim());
            assert_eq!(ep.raw, eq.raw);
            assert_eq!(ep.gam, eq.gam);
            assert_eq!(ep.repa, eq.repa);
            assert_eq!(ep.repb, eq.repb);
            assert_eq!(ep.shells.len(), eq.shells.len());
            for (a, b) in ep.shells.iter().zip(&eq.shells) {
                assert_eq!(a.level_ev, b.level_ev);
                assert_eq!(a.slater, b.slater);
                assert_eq!(a.poly_raw, b.poly_raw);
            }
        }
        // Serialization is deterministic and idempotent.
        assert_eq!(text, q.to_param_string());
    }

    #[test]
    fn parameter_target_get_and_set() {
        let p = Gfn1Parameters::from_str(MINI).unwrap();

        let t = ParameterTarget::parse("glob:ks").unwrap();
        assert_eq!(p.parameter_value(&t).unwrap(), 1.85);
        let p2 = p.with_parameter(&t, 2.0).unwrap();
        assert_eq!(p2.parameter_value(&t).unwrap(), 2.0);

        let te = ParameterTarget::parse("elem:1:GAM").unwrap();
        assert_eq!(p.parameter_value(&te).unwrap(), 0.47);
        let p3 = p.with_parameter(&te, 0.5).unwrap();
        assert_eq!(p3.parameter_value(&te).unwrap(), 0.5);
        // The derived shell hardness must track the raw GAM change.
        assert_ne!(
            p.element(1).unwrap().shell_hardness(AngularMomentum::S),
            p3.element(1).unwrap().shell_hardness(AngularMomentum::S)
        );

        let tlev = ParameterTarget::parse("elem:1:lev:1").unwrap();
        assert_eq!(p.parameter_value(&tlev).unwrap(), -1.0);
        let p4 = p.with_parameter(&tlev, -2.0).unwrap();
        assert_eq!(p4.element(1).unwrap().shells[1].level_ev, -2.0);

        let tp = ParameterTarget::parse("pair:1:1").unwrap();
        assert_eq!(p.parameter_value(&tp).unwrap(), 0.96);
        let p5 = p.with_parameter(&tp, 0.9).unwrap();
        assert_eq!(p5.pair_scaling(1, 1), 0.9);
    }

    #[test]
    fn round_trips_real_param_file() {
        let Ok(path) = std::env::var(GFN1_PARAM_ENV) else {
            return; // skip when the external parameter file is unavailable
        };
        let p = Gfn1Parameters::from_file(&path).unwrap();
        let q = Gfn1Parameters::from_str(&p.to_param_string()).unwrap();
        assert_eq!(p.globpar, q.globpar);
        assert_eq!(p.pairpar, q.pairpar);
        assert_eq!(p.elements.len(), q.elements.len());
        for (z, ep) in &p.elements {
            let eq = q.element(*z).unwrap();
            assert_eq!(ep.raw, eq.raw, "raw mismatch for Z={z}");
            assert_eq!(ep.ao_raw.trim(), eq.ao_raw.trim(), "ao mismatch for Z={z}");
        }
        // Idempotent: re-serializing the re-parsed copy is byte-identical.
        assert_eq!(p.to_param_string(), q.to_param_string());
    }
}
