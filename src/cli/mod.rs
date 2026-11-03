// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared command implementations for native binaries and installed Python scripts.
mod calculation;
mod observables;
pub mod properties;
mod training;

/// Run the shared native training workflow, preserving structured Rust errors.
pub fn train_model(arguments: Vec<String>) -> crate::Result<()> {
    run_with_resources(arguments, true)
}

fn run_with_resources(arguments: Vec<String>, training: bool) -> crate::Result<()> {
    let mut args = arguments.into_iter();
    let mut forwarded = Vec::new();
    let (mut threads, mut memory_mb) = (None, None);
    let mut cp = crate::resources::CpSettings::default();
    while let Some(arg) = args.next() {
        if matches!(
            arg.as_str(),
            "--cp-tolerance" | "--cp-relative-tolerance" | "--cp-max-iter"
        ) {
            let value = args
                .next()
                .ok_or_else(|| crate::Gfn1Error::InvalidInput(format!("{arg} requires a value")))?;
            if arg == "--cp-max-iter" {
                if cp.max_iterations.is_some() {
                    return Err(crate::Gfn1Error::InvalidInput(format!("duplicate {arg}")));
                }
                cp.max_iterations = Some(value.parse().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid {arg}: {value}"))
                })?);
            } else {
                let target = if arg == "--cp-tolerance" {
                    &mut cp.absolute_tolerance
                } else {
                    &mut cp.relative_tolerance
                };
                if target.is_some() {
                    return Err(crate::Gfn1Error::InvalidInput(format!("duplicate {arg}")));
                }
                *target = Some(value.parse().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid {arg}: {value}"))
                })?);
            }
            continue;
        }
        let destination = match arg.as_str() {
            "--threads" => Some(&mut threads),
            "--memory-mb" => Some(&mut memory_mb),
            _ => None,
        };
        if let Some(destination) = destination {
            if destination.is_some() {
                return Err(crate::Gfn1Error::InvalidInput(format!("duplicate {arg}")));
            }
            let value = args.next().ok_or_else(|| {
                crate::Gfn1Error::InvalidInput(format!("{arg} requires a positive integer"))
            })?;
            *destination =
                Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid {arg}: {value}"))
                })?);
        } else {
            forwarded.push(arg);
        }
    }
    let limits = crate::resources::ResourceLimits::from_megabytes(threads, memory_mb)?;
    let report = (threads.is_some() || memory_mb.is_some())
        && !forwarded
            .iter()
            .any(|a| matches!(a.as_str(), "--help" | "-h" | "--version" | "-V"));
    crate::resources::ResourceContext::with_cp(limits, cp)?.run(move || {
        if cp != crate::resources::CpSettings::default() {
            eprintln!("CP settings: {}", cp.as_json());
        }
        if report {
            eprintln!(
                "resources: threads={}, workspace_budget_bytes={}",
                crate::resources::effective_threads(),
                limits
                    .memory_budget_bytes
                    .map_or_else(|| "method default".into(), |n| n.to_string())
            );
        }
        if training {
            self::training::run(forwarded)
        } else if forwarded.first().is_some_and(|a| a == "properties") {
            properties::run(forwarded.into_iter().skip(1).collect())
        } else {
            calculation::run(forwarded)
        }
    })
}

/// Execute a CLI with arguments excluding the executable name.
pub fn main(arguments: Vec<String>, training: bool) -> i32 {
    let result = run_with_resources(arguments, training);
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}
