// SPDX-License-Identifier: GPL-3.0-or-later
//! Lightweight opt-in wall-clock profiling.
//!
//! Set `GFN1_PROFILE=1` to emit one line per scoped region to stderr:
//! `gfn1_profile_ms <label> <milliseconds>`.

use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();

#[derive(Debug)]
pub struct ProfileScope {
    label: &'static str,
    start: Option<Instant>,
}

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("GFN1_PROFILE")
            .map(|value| {
                let value = value.trim();
                !(value.is_empty()
                    || value == "0"
                    || value.eq_ignore_ascii_case("false")
                    || value.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(false)
    })
}

#[inline]
pub fn scope(label: &'static str) -> ProfileScope {
    ProfileScope {
        label,
        start: enabled().then(Instant::now),
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("gfn1_profile_ms {} {:.6}", self.label, elapsed_ms);
        }
    }
}
