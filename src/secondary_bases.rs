// SPDX-License-Identifier: GPL-3.0-or-later
//! Built-in GFN1-xTB-M secondary (dual) bases, **bundled into the binary** via
//! `include_str!` so the GFN1-xTB-M1 kinetic-energy correction and the experimental
//! richer-moment multipole electrostatics work without an external basis file.
//!
//! The data is the Dunning correlation-consistent series (cc-pVDZ/TZ/QZ/5Z), with one
//! GFN1-valence contraction per (element, angular momentum) selected by radial-node
//! matching to the GFN1 primary shell (`scripts/gen_secondary_basis.py`, validated to
//! reproduce the published GFN1-xTB-cc-pVDZ set exactly). The numbers come from the
//! Basis Set Exchange (CC-BY-4.0, attribution in the file headers); see each `.txt`.
//! Covers Z = 1..36 (the published reference set).

use crate::error::Result;
use crate::secondary_basis::{parse_secondary_basis, SecondaryBasis};

/// Bundled GFN1-xTB-cc-pVDZ secondary basis (the GFN1-xTB-M1 dual basis).
pub const CC_PVDZ: &str = include_str!("secondary_bases/gfn1-xtb-cc-pvdz.txt");
/// Bundled GFN1-xTB-cc-pVTZ secondary basis.
pub const CC_PVTZ: &str = include_str!("secondary_bases/gfn1-xtb-cc-pvtz.txt");
/// Bundled GFN1-xTB-cc-pVQZ secondary basis.
pub const CC_PVQZ: &str = include_str!("secondary_bases/gfn1-xtb-cc-pvqz.txt");
/// Bundled GFN1-xTB-cc-pV5Z secondary basis.
pub const CC_PV5Z: &str = include_str!("secondary_bases/gfn1-xtb-cc-pv5z.txt");

/// The bundled secondary-basis text for a name (`"cc-pVDZ"`, `"cc-pVTZ"`, `"cc-pVQZ"`,
/// `"cc-pV5Z"`; case- and `-`-insensitive). `None` for an unknown name.
pub fn builtin_secondary_text(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
        "ccpvdz" => Some(CC_PVDZ),
        "ccpvtz" => Some(CC_PVTZ),
        "ccpvqz" => Some(CC_PVQZ),
        "ccpv5z" => Some(CC_PV5Z),
        _ => None,
    }
}

/// Parse a bundled secondary basis by name. Returns `None` for an unknown name; the
/// inner `Result` reports a parse error (should not occur for the bundled data).
pub fn builtin_secondary(name: &str) -> Option<Result<SecondaryBasis>> {
    builtin_secondary_text(name).map(parse_secondary_basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bundled basis must parse and cover Z=1..86: 35 all-electron elements (Z=1..36,
    /// K/Z=19 omitted) reproducing the published GFN1-xTB-cc-pVDZ set, plus 31 heavy
    /// elements from the pseudopotential basis (best-effort, structure-validated).
    #[test]
    fn bundled_secondary_bases_parse() {
        for name in ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"] {
            let basis = builtin_secondary(name)
                .unwrap_or_else(|| panic!("no bundled basis {name}"))
                .unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
            assert_eq!(basis.elements(), 66, "{name} element count");
            // Oxygen (all-electron) carries an s and a p valence contraction.
            assert_eq!(basis.contractions(8, 0).len(), 1, "{name} O s");
            assert_eq!(basis.contractions(8, 1).len(), 1, "{name} O p");
            // Antimony (Z=51, heavy/PP) carries s, p, d valence contractions (ao=5s5p5d).
            assert_eq!(basis.contractions(51, 0).len(), 1, "{name} Sb s");
            assert_eq!(basis.contractions(51, 2).len(), 1, "{name} Sb d");
        }
        // Name normalization.
        assert!(builtin_secondary_text("CC_PV5Z").is_some());
        assert!(builtin_secondary_text("nonsense").is_none());
    }
}
