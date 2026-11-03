// SPDX-License-Identifier: GPL-3.0-or-later
//! Scoped CPU scheduling and estimated calculation-workspace budgets.
//!
//! Memory limits constrain checked workspace plans, not process RSS, Python
//! objects, allocator overhead, or results retained by the caller.
use crate::{Gfn1Error, Result};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Worker count; None uses Rayon's configured default.
    pub threads: Option<usize>,
    /// Upper bound used by checked workspace plans, in bytes.
    pub memory_budget_bytes: Option<usize>,
}

thread_local! {
    static LIMITS: Cell<ResourceLimits> = const { Cell::new(ResourceLimits { threads: None, memory_budget_bytes: None }) };
    static CP_SETTINGS: Cell<CpSettings> = const { Cell::new(CpSettings { absolute_tolerance: None, relative_tolerance: None, max_iterations: None }) };
}

/// Scoped CP/DFPT residual controls. None preserves each route's existing default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CpSettings {
    pub absolute_tolerance: Option<f64>,
    pub relative_tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
}
impl CpSettings {
    pub fn validate(self) -> Result<Self> {
        if [self.absolute_tolerance, self.relative_tolerance]
            .into_iter()
            .flatten()
            .any(|x| !x.is_finite() || x <= 0.)
            || self.max_iterations == Some(0)
        {
            return Err(Gfn1Error::InvalidInput(
                "CP tolerances and iteration limit must be finite and positive".into(),
            ));
        }
        Ok(self)
    }
    fn inherit(self, outer: Self) -> Self {
        Self {
            absolute_tolerance: self.absolute_tolerance.or(outer.absolute_tolerance),
            relative_tolerance: self.relative_tolerance.or(outer.relative_tolerance),
            max_iterations: self.max_iterations.or(outer.max_iterations),
        }
    }
    pub fn residual_target(self, absolute: f64, relative: f64, scale: f64) -> f64 {
        self.absolute_tolerance
            .unwrap_or(absolute)
            .max(self.relative_tolerance.unwrap_or(relative) * scale)
    }
    pub fn iterations(self, default: usize) -> usize {
        self.max_iterations.unwrap_or(default)
    }
    pub fn as_json(self) -> serde_json::Value {
        serde_json::json!({"absolute_tolerance":self.absolute_tolerance,"relative_tolerance":self.relative_tolerance,"max_iterations":self.max_iterations})
    }
    pub fn run<T: Send>(self, operation: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        ResourceContext::with_cp(ResourceLimits::default(), self)?.run(operation)
    }
}
pub fn current_cp() -> CpSettings {
    CP_SETTINGS.with(Cell::get)
}

impl ResourceLimits {
    pub fn validate(self) -> Result<Self> {
        if self.threads == Some(0) || self.memory_budget_bytes == Some(0) {
            return Err(Gfn1Error::InvalidInput(
                "threads and memory budget must be positive".into(),
            ));
        }
        Ok(self)
    }

    pub fn from_megabytes(threads: Option<usize>, memory_mb: Option<usize>) -> Result<Self> {
        let memory_budget_bytes = memory_mb
            .map(|m| {
                m.checked_mul(1024 * 1024)
                    .ok_or_else(|| Gfn1Error::InvalidInput("memory_mb overflows bytes".into()))
            })
            .transpose()?;
        Self {
            threads,
            memory_budget_bytes,
        }
        .validate()
    }

    /// Run once; repeated calculations should reuse a ResourceContext.
    pub fn run<T: Send>(self, operation: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        ResourceContext::new(self)?.run(operation)
    }
}

/// A reusable private Rayon pool; independently configured contexts can coexist.
pub struct ResourceContext {
    limits: ResourceLimits,
    cp: CpSettings,
    pool: Option<rayon::ThreadPool>,
}

impl ResourceContext {
    pub fn new(limits: ResourceLimits) -> Result<Self> {
        Self::with_cp(limits, current_cp())
    }
    pub fn with_cp(limits: ResourceLimits, cp: CpSettings) -> Result<Self> {
        let limits = limits.validate()?;
        let cp = cp.validate()?.inherit(current_cp());
        let pool = if limits == ResourceLimits::default() && cp == CpSettings::default() {
            None
        } else {
            let mut builder = rayon::ThreadPoolBuilder::new()
                .thread_name(|i| format!("gfn1-worker-{i}"))
                .start_handler(move |_| {
                    LIMITS.with(|slot| slot.set(limits));
                    CP_SETTINGS.with(|slot| slot.set(cp));
                });
            if let Some(n) = limits.threads {
                builder = builder.num_threads(n);
            }
            Some(builder.build().map_err(|error| {
                Gfn1Error::InvalidInput(format!("cannot create calculation thread pool: {error}"))
            })?)
        };
        Ok(Self { limits, cp, pool })
    }

    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }
    pub fn cp_settings(&self) -> CpSettings {
        self.cp
    }

    pub fn run<T: Send>(&self, operation: impl FnOnce() -> Result<T> + Send) -> Result<T> {
        let previous = current();
        let effective = ResourceLimits {
            threads: minimum(previous.threads, self.limits.threads),
            memory_budget_bytes: minimum(
                previous.memory_budget_bytes,
                self.limits.memory_budget_bytes,
            ),
        };
        let effective_cp = self.cp.inherit(current_cp());
        if effective != self.limits || effective_cp != self.cp {
            return Self::with_cp(effective, effective_cp)?.run(operation);
        }
        let invoke = || {
            struct Restore(ResourceLimits, CpSettings);
            impl Drop for Restore {
                fn drop(&mut self) {
                    LIMITS.with(|slot| slot.set(self.0));
                    CP_SETTINGS.with(|slot| slot.set(self.1));
                }
            }
            let _restore = Restore(current(), current_cp());
            LIMITS.with(|slot| slot.set(effective));
            CP_SETTINGS.with(|slot| slot.set(effective_cp));
            operation()
        };
        match &self.pool {
            Some(pool) => pool.install(invoke),
            None => invoke(),
        }
    }
}

fn minimum(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

pub fn current() -> ResourceLimits {
    LIMITS.with(Cell::get)
}

/// Explicit local budgets can tighten, but never raise, the calculation budget.
pub fn effective_memory_budget(requested: usize) -> usize {
    current()
        .memory_budget_bytes
        .map_or(requested, |limit| requested.min(limit))
}

/// Check an estimated simultaneous allocation before creating its large arrays.
pub fn check_workspace(bytes: usize, label: &str) -> Result<()> {
    if let Some(limit) = current().memory_budget_bytes {
        if bytes > limit {
            return Err(Gfn1Error::InvalidInput(format!("{label} workspace estimate {bytes} bytes exceeds calculation memory budget {limit} bytes")));
        }
    }
    Ok(())
}

/// Plan dense storage, including the requested number of matrices and scratch.
pub fn check_dense_workspace(rows: usize, cols: usize, matrices: usize, label: &str) -> Result<()> {
    let bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_mul(matrices))
        .ok_or_else(|| Gfn1Error::InvalidInput(format!("{label} workspace size overflows")))?;
    check_workspace(bytes, label)
}

pub fn effective_threads() -> usize {
    rayon::current_num_threads()
}

/// Dense SCF storage, shell kernels and configured mixing histories.
pub fn preflight_scf(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    options: &crate::ElectronicOptions,
    kpoints: usize,
) -> Result<()> {
    if current().memory_budget_bytes.is_none() {
        return Ok(());
    }
    let basis = crate::basis::BasisSet::build(
        system,
        params,
        crate::BasisOptions {
            nprim: options.nprim,
        },
    )?;
    let matrices = (if options.spin_polarization || options.plus_u {
        64usize
    } else {
        32usize
    })
    .checked_add(if options.lr_exchange || options.onsite_exchange {
        options.scc_broyden_size.clamp(2, 20) * 2
    } else {
        0
    })
    .and_then(|n| n.checked_mul(kpoints.max(1)))
    .ok_or_else(|| Gfn1Error::InvalidInput("SCF workspace overflow".into()))?;
    let dense = basis
        .len()
        .checked_mul(basis.len())
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_mul(matrices));
    let shells = basis.shells.len();
    let history = shells
        .checked_mul(options.scc_broyden_size.max(2))
        .and_then(|n| n.checked_mul(8 * 8));
    let estimate = dense
        .and_then(|n| history.and_then(|h| n.checked_add(h)))
        .ok_or_else(|| Gfn1Error::InvalidInput("SCF workspace overflow".into()))?;
    check_workspace(estimate, "SCF dense arrays and mixing history")
}

/// Conservative DFPT response plan, checked before the reference SCF.
pub fn preflight_response(
    system: &crate::PeriodicSystem,
    params: &crate::Gfn1Parameters,
    options: &crate::ElectronicOptions,
    kpoints: usize,
    spinor: bool,
) -> Result<()> {
    if current().memory_budget_bytes.is_none() {
        return Ok(());
    }
    let basis = crate::basis::BasisSet::build(
        system,
        params,
        crate::BasisOptions {
            nprim: options.nprim,
        },
    )?;
    let ndof = system
        .atoms
        .len()
        .checked_mul(3)
        .ok_or_else(|| Gfn1Error::InvalidInput("response DOF overflow".into()))?;
    let factor = 48usize
        .checked_mul(ndof.saturating_add(1))
        .and_then(|n| n.checked_mul(kpoints.max(1)))
        .and_then(|n| n.checked_mul(if spinor { 4 } else { 1 }))
        .ok_or_else(|| Gfn1Error::InvalidInput("response workspace overflow".into()))?;
    check_dense_workspace(basis.len(), basis.len(), factor, "DFPT response")?;
    check_dense_workspace(ndof, ndof, 16, "DFPT Hessian output")
}

/// Local faer control. Crossing into a second pool from a parallel tensor
/// assembly can recursively steal more assembly jobs while waiting, exhausting
/// the worker stack. Pass sequential parallelism directly instead.
pub(crate) fn faer_parallelism() -> faer::Par {
    let sequential = current().threads.is_none()
        && std::env::var("GFN1_FAER_THREADS")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "0" | "1"));
    if sequential {
        faer::Par::Seq
    } else {
        faer::get_global_parallelism()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn cp_settings_propagate_to_parallel_response_and_restore() {
        let cp = CpSettings {
            absolute_tolerance: Some(1e-11),
            relative_tolerance: Some(2e-11),
            max_iterations: Some(321),
        };
        cp.run(|| {
            (0..64).into_par_iter().for_each(|_| {
                assert_eq!(current_cp(), cp);
                let options = crate::cphf::CpxtbOptions::default();
                assert_eq!(options.tol, 1e-11);
                assert_eq!(options.max_iter, 321);
            });
            Ok(())
        })
        .unwrap();
        assert_eq!(current_cp(), CpSettings::default());
        assert!(CpSettings {
            absolute_tolerance: Some(f64::NAN),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(CpSettings {
            relative_tolerance: Some(-1.),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(CpSettings {
            max_iterations: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn configured_cp_residual_is_checked_on_real_direct_hessian() {
        let parameters = crate::Gfn1Parameters::builtin().unwrap();
        let molecule = crate::PeriodicSystem::from_xyz_str(
            "3\nwater\nO .02 -.01 .03\nH .958 .02 -.01\nH -.25 .925 .04\n",
            0.,
            false,
        )
        .unwrap();
        let options = crate::AnalyticHessianOptions::default();
        let baseline = crate::analytic_hessian(&molecule, &parameters, options.clone()).unwrap();
        let accepted = CpSettings {
            absolute_tolerance: Some(1e-10),
            relative_tolerance: Some(1e-10),
            ..Default::default()
        }
        .run(|| crate::analytic_hessian(&molecule, &parameters, options.clone()))
        .unwrap();
        assert!(baseline.hessian.max_abs_diff(&accepted.hessian) < 1e-11);
        let error = CpSettings {
            absolute_tolerance: Some(1e-30),
            relative_tolerance: Some(1e-30),
            ..Default::default()
        }
        .run(|| crate::analytic_hessian(&molecule, &parameters, options))
        .unwrap_err()
        .to_string();
        assert!(error.contains("CP residual"), "{error}");
    }

    #[test]
    fn scoped_workers_propagate_limits_and_do_not_change_other_contexts() {
        let one = ResourceContext::new(ResourceLimits {
            threads: Some(1),
            memory_budget_bytes: Some(16 * 1024 * 1024),
        })
        .unwrap();
        let two = ResourceContext::new(ResourceLimits {
            threads: Some(2),
            memory_budget_bytes: Some(32 * 1024 * 1024),
        })
        .unwrap();
        std::thread::scope(|scope| {
            for context in [&one, &two] {
                scope.spawn(move || {
                    context
                        .run(|| {
                            let limits = context.limits();
                            (0..32).into_par_iter().for_each(|_| {
                                assert_eq!(current(), limits);
                                assert_eq!(effective_threads(), limits.threads.unwrap());
                                crate::linalg::ensure_parallelism();
                                assert_eq!(
                                    faer::get_global_parallelism().degree(),
                                    limits.threads.unwrap()
                                );
                            });
                            // The high-level faer eigensolver must use this same pool,
                            // including after an earlier one-worker calculation.
                            let mut matrix = crate::linalg::Matrix::identity(40);
                            matrix[(0, 0)] = 2.0;
                            let eig = crate::linalg::symmetric_eigen(&matrix)?;
                            assert_eq!(eig.values[39], 2.0);
                            Ok(())
                        })
                        .unwrap()
                });
            }
        });
        assert_eq!(current(), ResourceLimits::default());
        one.run(|| {
            two.run(|| {
                assert_eq!(current(), one.limits());
                assert_eq!(effective_threads(), 1);
                assert_eq!(effective_memory_budget(usize::MAX), 16 * 1024 * 1024);
                Ok(())
            })
        })
        .unwrap();
    }

    #[test]
    fn workspace_budget_rejects_real_calculation_and_cannot_be_overridden_locally() {
        let params = crate::Gfn1Parameters::resolve(None).unwrap();
        let system =
            crate::PeriodicSystem::from_xyz_str("2\nH2\nH 0 0 0\nH 0 0 0.75\n", 0.0, false)
                .unwrap();
        let limits = ResourceLimits {
            threads: Some(1),
            memory_budget_bytes: Some(1),
        };
        limits
            .run(|| {
                let error =
                    crate::run_electronic(&system, &params, crate::ElectronicOptions::default())
                        .unwrap_err();
                assert!(error.to_string().contains("memory budget"));
                assert!(
                    crate::third_derivative::SymmetricThird::try_zeros(12, usize::MAX).is_err()
                );
                assert!(
                    crate::fourth_derivative::SymmetricFourth::try_zeros(12, usize::MAX).is_err()
                );
                let mut options = crate::hessian::AnalyticHessianOptions::default();
                options.electronic_options.max_scc = 0;
                let error = crate::third_derivative::CubicReference::build(
                    &system,
                    &params,
                    &options.electronic_options,
                    usize::MAX,
                )
                .err()
                .unwrap();
                assert!(error.to_string().contains("memory budget"));
                Ok(())
            })
            .unwrap();
        assert!(ResourceLimits::from_megabytes(Some(0), None).is_err());
        assert!(ResourceLimits::from_megabytes(None, Some(0)).is_err());
        assert!(ResourceLimits::from_megabytes(None, Some(usize::MAX)).is_err());
    }
}
