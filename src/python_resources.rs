// SPDX-License-Identifier: GPL-3.0-or-later
//! Generic Python callback scope, shared by every public calculator method.
use crate::resources::{ResourceContext, ResourceLimits};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "_ResourceContext")]
pub(crate) struct PyResourceContext {
    context: ResourceContext,
}

#[pymethods]
impl PyResourceContext {
    #[new]
    #[pyo3(signature=(threads=None,memory_mb=None,cp_tolerance=None,cp_relative_tolerance=None,cp_max_iter=None))]
    fn new(
        threads: Option<usize>,
        memory_mb: Option<usize>,
        cp_tolerance: Option<f64>,
        cp_relative_tolerance: Option<f64>,
        cp_max_iter: Option<usize>,
    ) -> PyResult<Self> {
        let limits = ResourceLimits::from_megabytes(threads, memory_mb)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            context: ResourceContext::with_cp(
                limits,
                crate::resources::CpSettings {
                    absolute_tolerance: cp_tolerance,
                    relative_tolerance: cp_relative_tolerance,
                    max_iterations: cp_max_iter,
                },
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn run(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<Py<PyAny>> {
        py.allow_threads(|| {
            self.context
                .run(|| Ok(Python::with_gil(|gil| callback.call0(gil))))
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?
    }

    fn info(&self) -> (Option<usize>, Option<usize>) {
        let limits = self.context.limits();
        (limits.threads, limits.memory_budget_bytes)
    }
    fn cp_info(&self) -> (Option<f64>, Option<f64>, Option<usize>) {
        let cp = self.context.cp_settings();
        (
            cp.absolute_tolerance,
            cp.relative_tolerance,
            cp.max_iterations,
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResourceContext>()
}
