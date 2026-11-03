// SPDX-License-Identifier: GPL-3.0-or-later
//! Method references from the user-supplied revision-2 bibliography.
//! A citation describes a method; it is not an asset redistribution license.

use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};

pub const REFERENCES_BIB: &str = include_str!("../references.bib");

/// References applicable to the selected electronic model. Property drivers
/// append their own references (e.g. `Piccardo2015GVPT` for PES-only VPT2).
pub fn electronic_citations(options: &ElectronicOptions, periodic: bool) -> Vec<&'static str> {
    let mut keys = vec!["Grimme2017GFN1"];
    if options.spin_polarization {
        keys.push("Neugebauer2023Spin");
    }
    if options.solvation.is_some() {
        keys.push("Ehlert2021ALPB");
    }
    if options.ml_model.is_some() {
        keys.push("Li2026MLxTB");
    }
    if options
        .ml_model
        .as_ref()
        .is_some_and(|m| m.training_provenance.contains("reference FCI/cc-pVTZ"))
    {
        keys.extend(["Sun2020PySCF", "Dunning1989CC"]);
    }
    if options
        .ml_model
        .as_ref()
        .is_some_and(|m| m.training_provenance.contains("reference PBE0/def2-SVP"))
    {
        keys.extend(["Sun2020PySCF", "Adamo1999PBE0", "Weigend2005Def2"]);
    }
    if options.external_field.magnetic_field.is_some() {
        keys.push("Cheng2023Magnetic");
    }
    if periodic {
        keys.push("Buccheri2025Periodic");
    }
    keys
}

/// Select complete BibTeX entries, preserving the supplied text. No network
/// lookup, inferred equation numbers, or citations for inactive planned methods.
pub fn bibliography(keys: &[&str]) -> Result<String> {
    let mut out = String::new();
    let mut seen = Vec::new();
    for &key in keys {
        if seen.contains(&key) {
            continue;
        }
        let entry = REFERENCES_BIB
            .split("\n@")
            .skip(1)
            .find(|entry| {
                entry
                    .split_once('{')
                    .is_some_and(|(_, rest)| rest.starts_with(&format!("{key},")))
            })
            .ok_or_else(|| Gfn1Error::InvalidInput(format!("unknown method citation `{key}`")))?;
        out.push('@');
        out.push_str(entry.trim_end());
        out.push_str("\n\n");
        seen.push(key);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_references_are_complete_and_deduplicated() {
        let bib = bibliography(&["Grimme2017GFN1", "Piccardo2015GVPT", "Grimme2017GFN1"]).unwrap();
        assert_eq!(bib.matches("@article{").count(), 2);
        assert!(bib.contains("10.1002/qua.24931"));
        assert!(!bib.contains("@article{Ehlert2021ALPB"));
        assert!(bibliography(&["unknown"]).is_err());
    }
}
