// SPDX-License-Identifier: GPL-3.0-or-later
//! Response (coupled-perturbed) machinery for GFN1-xTB.
//!
//! [`cpxtb`] holds the real non-PBC CPXTB solver and its response helpers; it was
//! previously the crate-root module `cphf`, which now re-exports from here.
//! [`charge_space`] is the v0.5.0 charge-space dielectric solver: one factored
//! `nsh × nsh` linear system serving every first- (and, in Phase 6, second-)
//! order response right-hand side, finite-temperature native.

pub mod charge_space;
pub mod cpxtb;

pub use cpxtb::*;
