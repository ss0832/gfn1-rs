// SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;

pub type Result<T> = std::result::Result<T, Gfn1Error>;

#[derive(Debug)]
pub enum Gfn1Error {
    Io(std::io::Error),
    Parse { line: usize, message: String },
    InvalidInput(String),
    MissingElement(u8),
    MissingGlobal(String),
    LinearAlgebra(String),
    SingularCell,
    SccNotConverged { iterations: usize, rms: f64 },
}

impl fmt::Display for Gfn1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement(z) => write!(f, "missing GFN1 parameter block for Z={z}"),
            Self::MissingGlobal(key) => write!(f, "missing required $globpar entry `{key}`"),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::SingularCell => write!(f, "lattice vectors form a singular cell"),
            Self::SccNotConverged { iterations, rms } => {
                write!(
                    f,
                    "GFN1 SCC did not converge after {iterations} iterations (rms={rms:.6e})"
                )
            }
        }
    }
}

impl std::error::Error for Gfn1Error {}

impl From<std::io::Error> for Gfn1Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
