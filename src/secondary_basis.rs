// SPDX-License-Identifier: GPL-3.0-or-later
//! Secondary (dual) basis for the **GFN1-xTB-M1** kinetic-energy correction.
//!
//! GFN1-xTB-M1 (Cheng & Wibowo-Teale, *J. Chem. Theory Comput.* **19**, 6226
//! (2023)) evaluates only the `p^2` / `pi^2` kinetic-energy integrals of the
//! magnetic correction over a *secondary* set of basis functions chosen to have
//! the correct nodal structure (the GFN1 minimal AOs lack radial nodes, which
//! spoils the M0 kinetic-energy correction). The secondary set is a node-correct
//! subset of cc-pVDZ (the `$Basis = GFN1-xTB-cc-pVDZ` file in the paper's SI),
//! with one contraction per (element, angular momentum) matched one-to-one to the
//! primary GFN1 shells; hydrogen uses the GFN1-xTB AO.
//!
//! Only the radial part differs from the primary basis — each primary AO `mu` maps
//! to a secondary AO with the **same** centre and Cartesian (angular) component but
//! the secondary contraction's primitives, so the correction stays an `n x n`
//! matrix in the primary AO space.
//!
//! The data is *not* bundled: like `param_gfn1-xtb.txt` it is loaded from a path /
//! string the caller supplies. The numerical exponents/coefficients are cc-pVDZ
//! (freely redistributable, e.g. from the Basis Set Exchange) plus the GFN1 H AO,
//! sign-matched to the GFN1 phase, so a license-clean copy can be regenerated from
//! cc-pVDZ rather than copied from the SI.

use crate::error::{Gfn1Error, Result};
use std::collections::HashMap;

/// One contracted radial function (a list of `(exponent, coefficient)` primitives)
/// for a given angular momentum.
#[derive(Clone, Debug)]
pub struct SecondaryContraction {
    pub primitives: Vec<(f64, f64)>,
}

/// Parsed secondary basis: per element `Z`, per angular momentum `l` (0 = s, 1 = p,
/// 2 = d), an ordered list of contractions (ordered by increasing principal quantum
/// number / node count, as written in the file).
#[derive(Clone, Debug, Default)]
pub struct SecondaryBasis {
    by_element: HashMap<u8, [Vec<SecondaryContraction>; 3]>,
}

impl SecondaryBasis {
    /// Contractions of angular momentum `l` for element `z`, in file order.
    pub fn contractions(&self, z: u8, l: usize) -> &[SecondaryContraction] {
        match self.by_element.get(&z) {
            Some(by_l) if l < 3 => &by_l[l],
            _ => &[],
        }
    }

    /// The `rank`-th contraction (0-based) of angular momentum `l` for element `z`,
    /// i.e. the secondary radial function for the `rank`-th primary GFN1 shell of
    /// that angular momentum on the element.
    pub fn contraction(&self, z: u8, l: usize, rank: usize) -> Option<&SecondaryContraction> {
        self.contractions(z, l).get(rank)
    }

    pub fn elements(&self) -> usize {
        self.by_element.len()
    }
}

fn angular_from_label(line: &str) -> Option<usize> {
    let upper = line.to_ascii_uppercase();
    if upper.contains("S-TYPE") {
        Some(0)
    } else if upper.contains("P-TYPE") {
        Some(1)
    } else if upper.contains("D-TYPE") {
        Some(2)
    } else {
        None
    }
}

/// Parse the `$Basis = GFN1-xTB-cc-pVDZ` secondary-basis text. Format:
/// element blocks `a <Z>`; per-angular-momentum sections `$ X-TYPE FUNCTIONS`; a
/// header `<nprim> <ncontr> 0`; then `nprim` rows of `exponent c_1 ... c_ncontr`.
/// Each column is one contraction (zero-coefficient primitives are dropped).
pub fn parse_secondary_basis(text: &str) -> Result<SecondaryBasis> {
    let mut out = SecondaryBasis::default();
    let mut current_z: Option<u8> = None;
    let mut current_l: Option<usize> = None;
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("a ").or_else(|| line.strip_prefix("A ")) {
            let z: u8 = rest.trim().parse().map_err(|_| {
                Gfn1Error::InvalidInput(format!("secondary basis: bad element line `{line}`"))
            })?;
            current_z = Some(z);
            current_l = None;
            out.by_element
                .entry(z)
                .or_insert_with(|| [Vec::new(), Vec::new(), Vec::new()]);
            continue;
        }
        if line.starts_with('$') {
            if let Some(l) = angular_from_label(line) {
                current_l = Some(l);
            }
            continue;
        }
        // Otherwise this must be a shell header `<nprim> <ncontr> 0`.
        let header: Vec<&str> = line.split_whitespace().collect();
        if header.len() < 2 {
            continue;
        }
        let (Ok(nprim), Ok(ncontr)) = (header[0].parse::<usize>(), header[1].parse::<usize>())
        else {
            continue;
        };
        let z = current_z.ok_or_else(|| {
            Gfn1Error::InvalidInput("secondary basis: shell before any `a <Z>`".to_string())
        })?;
        let l = current_l.ok_or_else(|| {
            Gfn1Error::InvalidInput("secondary basis: shell before any `$ X-TYPE`".to_string())
        })?;
        let mut columns = vec![Vec::<(f64, f64)>::new(); ncontr];
        for _ in 0..nprim {
            let row = lines.next().ok_or_else(|| {
                Gfn1Error::InvalidInput("secondary basis: truncated primitive block".to_string())
            })?;
            let vals: Vec<f64> = row
                .split_whitespace()
                .map(|t| t.parse::<f64>())
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| {
                    Gfn1Error::InvalidInput(format!("secondary basis: bad primitive row `{row}`"))
                })?;
            if vals.len() < 1 + ncontr {
                return Err(Gfn1Error::InvalidInput(format!(
                    "secondary basis: primitive row needs {} columns, got {}",
                    1 + ncontr,
                    vals.len()
                )));
            }
            let exponent = vals[0];
            for (c, col) in columns.iter_mut().enumerate() {
                let coeff = vals[1 + c];
                if coeff != 0.0 {
                    col.push((exponent, coeff));
                }
            }
        }
        if let Some(by_l) = out.by_element.get_mut(&z) {
            for col in columns {
                if !col.is_empty() {
                    by_l[l].push(SecondaryContraction { primitives: col });
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = "$Basis = test\n\
        a 1\n\
        $ HYDROGEN\n\
        $ S-TYPE FUNCTIONS\n\
        3 2 0\n\
        7.6 0.05 0.0\n\
        1.4 0.26 0.0\n\
        0.4 0.0 0.6\n\
        a 8\n\
        $ OXYGEN\n\
        $ S-TYPE FUNCTIONS\n\
        2 1 0\n\
        11.0 -0.1 \n\
        1.0 0.5\n\
        $ P-TYPE FUNCTIONS\n\
        2 1 0\n\
        17.0 0.04\n\
        1.0 0.5\n";

    #[test]
    fn parses_elements_shells_and_contractions() {
        let basis = parse_secondary_basis(MINI).unwrap();
        assert_eq!(basis.elements(), 2);
        // Hydrogen: two s-contractions (the GFN1 1s + a node function).
        let h_s = basis.contractions(1, 0);
        assert_eq!(h_s.len(), 2);
        assert_eq!(h_s[0].primitives.len(), 2); // 7.6, 1.4 (the no-node 1s)
        assert_eq!(h_s[1].primitives.len(), 1); // 0.4 (the second column)
        assert!((h_s[0].primitives[0].0 - 7.6).abs() < 1e-12);
        assert!((h_s[0].primitives[0].1 - 0.05).abs() < 1e-12);
        // Oxygen: one s and one p contraction.
        assert_eq!(basis.contractions(8, 0).len(), 1);
        assert_eq!(basis.contractions(8, 1).len(), 1);
        assert_eq!(basis.contraction(8, 0, 0).unwrap().primitives.len(), 2);
        assert!(basis.contraction(8, 2, 0).is_none()); // no d
    }
}
