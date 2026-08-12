// SPDX-License-Identifier: GPL-3.0-or-later
//! Energy-term registry: the single source of truth for which analytic
//! derivative orders each term implements.
//!
//! Every derivative driver (gradient, Hessian, third/fourth derivative) calls
//! [`require_order`] first, so an option set whose terms are not implemented at
//! the requested order fails fast with a uniform message instead of silently
//! returning derivatives of a *different* energy expression (the pre-v0.5.0
//! failure mode of the analytic Hessian).
//!
//! Update a term's `max_analytic_order` in exactly one place — here — when a
//! new derivative order lands for it.

use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::params::Gfn1Parameters;

/// One energy term (or physics-changing modifier) of the model.
#[derive(Clone, Copy, Debug)]
pub struct TermDescriptor {
    /// Human-readable name used in error messages.
    pub name: &'static str,
    /// Whether the current options/parameters activate this term.
    pub active: bool,
    /// Highest nuclear-coordinate derivative order with a complete analytic
    /// implementation (0 = energy only, 1 = gradient, 2 = Hessian, ...).
    pub max_analytic_order: u8,
}

/// The registry, evaluated for a concrete option/parameter set.
///
/// Orders reflect the state of the NON-PBC drivers (the PBC drivers carry
/// their own coverage and guards).
pub fn active_terms(options: &ElectronicOptions, params: &Gfn1Parameters) -> Vec<TermDescriptor> {
    let d3_atm_active =
        options.enable_dispersion && !options.experimental_d4 && params.global("s9", 0.0) != 0.0;
    vec![
        TermDescriptor {
            name: "repulsion",
            active: true,
            // Radial fourth ladder (`repulsion_fourth_derivative`).
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "H0 band + Pulay + CN chain",
            active: true,
            // Phase-4d frozen fourth blocks + the directional 2n+1 response
            // assembly (`fourth_derivative::directional_fourth_derivative`).
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "isotropic SCC (2nd/3rd/higher charge orders)",
            active: true,
            // Frozen SCC2 fourth + the second-order charge-space response
            // (onsite anharmonic chains included through E⁗(q)).
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "D3(BJ) two-body dispersion",
            active: options.enable_dispersion && !options.experimental_d4,
            // Energy/gradient/Hessian/third + `dispersion_fourth_derivative` (Jet4).
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "D3 ATM (three-body) dispersion",
            active: d3_atm_active,
            // Jet2/Jet3/Jet4 promotions of the same ATM triple energy give the Hessian, third and
            // fourth derivatives; the molecular and lattice-summed loops are both covered.
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "halogen bond",
            active: true,
            // Jet2/Jet3/Jet4 promotions of the same per-triple energy give the Hessian, third
            // and fourth derivatives (`halogen_fourth_derivative`).
            max_analytic_order: 4,
        },
        TermDescriptor {
            name: "experimental D4 dispersion",
            active: options.experimental_d4,
            max_analytic_order: 1,
        },
        TermDescriptor {
            name: "multipole (mDFTB2/CAMM) electrostatics",
            active: options.multipole,
            max_analytic_order: 1,
        },
        TermDescriptor {
            name: "long-range Fock exchange (MFX/OFX)",
            active: options.lr_exchange,
            max_analytic_order: 1,
        },
        TermDescriptor {
            name: "DFT+U/+U+V",
            active: options.plus_u,
            max_analytic_order: 1,
        },
        TermDescriptor {
            name: "spin polarization (spGFN1)",
            active: options.spin_polarization,
            max_analytic_order: 1,
        },
        TermDescriptor {
            name: "external electric field",
            active: options.external_field.electric_field.is_some(),
            max_analytic_order: 1,
        },
    ]
    // NOTE: Fermi smearing is deliberately NOT a registry row. Whether the
    // occupations are actually fractional is only known after the SCC
    // converges (the default 300 K leaves gapped molecules at integer
    // occupations), so the third-derivative driver keeps its runtime
    // occupation-based guard as the authority for that modifier.
}

/// Fail unless every active term implements analytic derivatives of `order`.
pub fn require_order(
    options: &ElectronicOptions,
    params: &Gfn1Parameters,
    order: u8,
    context: &str,
) -> Result<()> {
    let blockers: Vec<String> = active_terms(options, params)
        .into_iter()
        .filter(|t| t.active && t.max_analytic_order < order)
        .map(|t| format!("`{}` (max analytic order {})", t.name, t.max_analytic_order))
        .collect();
    if blockers.is_empty() {
        return Ok(());
    }
    Err(Gfn1Error::InvalidInput(format!(
        "{context} requires analytic order-{order} derivatives, but the active option set \
         includes terms without them: {}. Disable those options or use a lower-order / \
         finite-difference path",
        blockers.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_gfn1_supports_order_three() {
        let params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions::default();
        assert!(require_order(&options, &params, 3, "test").is_ok());
    }

    #[test]
    fn multipole_blocks_order_two() {
        let params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions {
            multipole: true,
            ..ElectronicOptions::default()
        };
        let err = require_order(&options, &params, 2, "test");
        assert!(err.is_err());
        assert!(format!("{}", err.err().unwrap()).contains("multipole"));
    }

    /// The ATM row activates on `s9 != 0`, but it now carries analytic derivatives through
    /// **order 4** (Jet2/Jet3/Jet4 promotions of the ATM triple energy), so activating it must no
    /// longer block orders 2–4. Anything above 4 still has to fail.
    #[test]
    fn atm_activates_on_nonzero_s9() {
        let mut params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions::default();
        // The official GFN1 file carries s9 = 0 → ATM inactive.
        assert!(require_order(&options, &params, 3, "test").is_ok());
        params.globpar.insert("s9".to_string(), 1.0);
        assert!(
            active_terms(&options, &params)
                .iter()
                .any(|t| t.name.contains("ATM") && t.active),
            "s9 != 0 must activate the ATM registry row"
        );
        for order in 2..=3 {
            assert!(
                require_order(&options, &params, order, "test").is_ok(),
                "ATM must not block analytic order {order}"
            );
        }
        // With the directional quartic assembly landed, stock GFN1 (+ATM) supports order 4;
        // order 5 must still fail, and ATM must not be among its blockers.
        assert!(
            require_order(&options, &params, 4, "test").is_ok(),
            "stock GFN1 + ATM must support analytic order 4"
        );
        // Nothing implements order 5 — every core row (ATM included) blocks it.
        assert!(require_order(&options, &params, 5, "test").is_err());
    }

    /// The full stock option set supports the directional analytic quartic.
    #[test]
    fn stock_gfn1_supports_order_four() {
        let params = Gfn1Parameters::builtin().unwrap();
        let options = ElectronicOptions::default();
        assert!(require_order(&options, &params, 4, "test").is_ok());
        assert!(require_order(&options, &params, 5, "test").is_err());
    }
}
