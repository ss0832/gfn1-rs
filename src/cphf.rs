// SPDX-License-Identifier: GPL-3.0-or-later
//! Deprecated location: the CPXTB machinery lives in [`crate::response`].
//!
//! This module is kept as a compatibility shim so that existing `crate::cphf::*`
//! and `gfn1_rs::cphf::*` paths keep resolving; see [`crate::response::cpxtb`].

pub use crate::response::cpxtb::*;
