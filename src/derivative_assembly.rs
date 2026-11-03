// SPDX-License-Identifier: GPL-3.0-or-later
//! Packed Cartesian polarization with bounded worker-local directions.

use crate::error::{Gfn1Error, Result};
use crate::fourth_derivative::SymmetricFourth;
use crate::third_derivative::SymmetricThird;
use rayon::prelude::*;

fn overflow() -> Gfn1Error {
    Gfn1Error::InvalidInput("Cartesian derivative assembly size overflow".into())
}

pub(crate) fn packed_len(n: usize, order: usize) -> Result<usize> {
    let mut count = 1u128;
    for k in 1..=order as u128 {
        count = count.checked_mul(n as u128 + k - 1).ok_or_else(overflow)? / k;
    }
    usize::try_from(count).map_err(|_| overflow())
}

/// Packed output, lower-order directional values, and bounded batch scratch.
pub(crate) fn estimated_bytes(
    order: usize,
    size: usize,
    ndof: usize,
    workers: usize,
    batch: usize,
    packed_output: bool,
) -> Result<usize> {
    if workers == 0 || batch == 0 {
        return Err(Gfn1Error::InvalidInput(
            "Cartesian derivative batch size and parallelism must be positive".into(),
        ));
    }
    let cached_order = if packed_output { order } else { order - 1 };
    let elements = (1..=cached_order).try_fold(0usize, |count, degree| {
        count
            .checked_add(packed_len(size, degree)?)
            .ok_or_else(overflow)
    })?;
    let batch_bytes = packed_len(size, order)?
        .min(batch)
        .checked_mul(order * std::mem::size_of::<usize>() + std::mem::size_of::<f64>())
        .ok_or_else(overflow)?;
    let directions = ndof
        .checked_mul(8)
        .and_then(|n| n.checked_mul(workers.min(batch)))
        .ok_or_else(overflow)?;
    elements
        .checked_mul(8)
        .and_then(|n| n.checked_add(batch_bytes))
        .and_then(|n| n.checked_add(directions))
        .ok_or_else(overflow)
}

pub(crate) struct CartesianAssembly<'a> {
    dofs: &'a [usize],
    ndof: usize,
    budget: usize,
    workers: usize,
    batch: usize,
}

impl<'a> CartesianAssembly<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        order: usize,
        dofs: &'a [usize],
        ndof: usize,
        budget: usize,
        workspace: usize,
        reserve: usize,
        worker_limit: usize,
        batch: usize,
    ) -> Result<Self> {
        Self::new_with_output(
            order,
            dofs,
            ndof,
            budget,
            workspace,
            reserve,
            worker_limit,
            batch,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_output(
        order: usize,
        dofs: &'a [usize],
        ndof: usize,
        budget: usize,
        workspace: usize,
        reserve: usize,
        worker_limit: usize,
        batch: usize,
        packed_output: bool,
    ) -> Result<Self> {
        if dofs.iter().any(|&i| i >= ndof) {
            return Err(Gfn1Error::InvalidInput(
                "Cartesian derivative selected DOF out of range".into(),
            ));
        }
        let budget = crate::resources::effective_memory_budget(budget);
        let batch = if packed_output {
            batch
        } else {
            // A file output can use a smaller batch when the reference and
            // lower-order caches leave little memory for the result buffer.
            let minimum = estimated_bytes(order, dofs.len(), ndof, 1, 1, false)?
                .checked_add(reserve)
                .and_then(|n| n.checked_add(workspace))
                .ok_or_else(overflow)?;
            let bytes_per_entry = order * std::mem::size_of::<usize>() + std::mem::size_of::<f64>();
            batch.min(1 + budget.saturating_sub(minimum) / bytes_per_entry)
        };
        let limit = worker_limit
            .min(crate::resources::effective_threads())
            .min(batch);
        let serial = estimated_bytes(order, dofs.len(), ndof, 1, batch, packed_output)?
            .checked_add(reserve)
            .and_then(|n| n.checked_add(workspace))
            .ok_or_else(overflow)?;
        if serial > budget || limit == 0 {
            return Err(Gfn1Error::InvalidInput(format!(
                "FC{order} packed output/reference estimate {serial} bytes exceeds budget {budget}, or no workers were allowed"
            )));
        }
        let additional = workspace
            .checked_add(ndof.checked_mul(8).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let workers = limit.min(1 + (budget - serial) / additional.max(1));
        Ok(Self {
            dofs,
            ndof,
            budget,
            workers,
            batch,
        })
    }

    fn sweep<const K: usize>(
        &self,
        indices: impl Iterator<Item = [usize; K]>,
        evaluate: impl Fn([usize; K], &mut [f64]) -> Result<f64> + Sync,
        mut store: impl FnMut([usize; K], f64) -> Result<()>,
    ) -> Result<()> {
        let mut indices = indices;
        let count = packed_len(self.dofs.len(), K)?.min(self.batch);
        let mut keys = Vec::with_capacity(count);
        let mut values = vec![0.; count];
        let mut serial_direction = if self.workers == 1 {
            vec![0.; self.ndof]
        } else {
            Vec::new()
        };
        let run = |keys: &[[usize; K]], values: &mut [f64], direction: &mut [f64]| -> Result<()> {
            for (&key, value) in keys.iter().zip(values) {
                *value = evaluate(key, direction)?;
                if !value.is_finite() {
                    return Err(Gfn1Error::InvalidInput(
                        "non-finite analytic Cartesian derivative".into(),
                    ));
                }
            }
            Ok(())
        };
        loop {
            keys.clear();
            keys.extend(indices.by_ref().take(count));
            if keys.is_empty() {
                break;
            }
            let values = &mut values[..keys.len()];
            if self.workers == 1 {
                run(&keys, values, &mut serial_direction)?;
            } else {
                let chunk = keys.len().div_ceil(self.workers);
                keys.par_chunks(chunk)
                    .zip(values.par_chunks_mut(chunk))
                    .try_for_each(|(keys, values)| run(keys, values, &mut vec![0.; self.ndof]))?;
            }
            for (&key, &value) in keys.iter().zip(values.iter()) {
                store(key, value)?;
            }
        }
        Ok(())
    }

    fn direction<const K: usize>(&self, indices: [usize; K], v: &mut [f64]) {
        v.fill(0.);
        for i in indices {
            v[self.dofs[i]] += 1.;
        }
    }

    fn lower_values(
        &self,
        order: u32,
        evaluate: &(impl Fn(&[f64]) -> Result<f64> + Sync),
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.dofs.len();
        let mut singles = vec![0.; n];
        self.sweep(
            (0..n).map(|i| [i]),
            |[i], v| {
                self.direction([i], v);
                evaluate(v)
            },
            |[i], value| {
                singles[i] = value;
                Ok(())
            },
        )?;
        let mut pairs = vec![0.; packed_len(n, 2)?];
        self.sweep(
            (0..n).flat_map(|j| (0..=j).map(move |i| [i, j])),
            |[i, j], v| {
                if i == j {
                    return Ok(2_f64.powi(order as i32) * singles[i]);
                }
                self.direction([i, j], v);
                evaluate(v)
            },
            |[i, j], value| {
                pairs[pair_index(i, j)] = value;
                Ok(())
            },
        )?;
        Ok((singles, pairs))
    }

    pub(crate) fn cubic(
        &self,
        evaluate: impl Fn(&[f64]) -> Result<f64> + Sync,
    ) -> Result<SymmetricThird> {
        let n = self.dofs.len();
        let mut tensor = SymmetricThird::try_zeros(n, self.budget)?;
        self.cubic_into(evaluate, |[a, b, c], value| {
            tensor.add(a, b, c, value);
            Ok(())
        })?;
        Ok(tensor)
    }

    pub(crate) fn cubic_into(
        &self,
        evaluate: impl Fn(&[f64]) -> Result<f64> + Sync,
        store: impl FnMut([usize; 3], f64) -> Result<()>,
    ) -> Result<()> {
        let n = self.dofs.len();
        let (single, pairs) = self.lower_values(3, &evaluate)?;
        let pair = |i, j| pairs[pair_index(i, j)];
        self.sweep(
            triples(n),
            |[a, b, c], v| {
                if a == c {
                    return Ok(single[a]);
                }
                self.direction([a, b, c], v);
                Ok((evaluate(v)? - pair(a, b) - pair(a, c) - pair(b, c)
                    + single[a]
                    + single[b]
                    + single[c])
                    / 6.)
            },
            store,
        )
    }

    pub(crate) fn quartic(
        &self,
        evaluate: impl Fn(&[f64]) -> Result<f64> + Sync,
    ) -> Result<SymmetricFourth> {
        let n = self.dofs.len();
        let mut tensor = SymmetricFourth::try_zeros(n, self.budget)?;
        self.quartic_into(evaluate, |[a, b, c, d], value| {
            tensor.add(a, b, c, d, value);
            Ok(())
        })?;
        Ok(tensor)
    }

    /// Emit the canonical packed sequence without holding its values in RAM.
    pub(crate) fn quartic_into(
        &self,
        evaluate: impl Fn(&[f64]) -> Result<f64> + Sync,
        store: impl FnMut([usize; 4], f64) -> Result<()>,
    ) -> Result<()> {
        let n = self.dofs.len();
        let (single, pairs) = self.lower_values(4, &evaluate)?;
        let pair = |i, j| pairs[pair_index(i, j)];
        let mut third = SymmetricThird::try_zeros(n, self.budget)?;
        self.sweep(
            triples(n),
            |[a, b, c], v| {
                if a == c {
                    return Ok(81. * single[a]);
                }
                self.direction([a, b, c], v);
                evaluate(v)
            },
            |[a, b, c], value| {
                third.add(a, b, c, value);
                Ok(())
            },
        )?;
        let indices = (0..n).flat_map(|d| triples(d + 1).map(move |[a, b, c]| [a, b, c, d]));
        self.sweep(
            indices,
            |[a, b, c, d], v| {
                if a == d {
                    return Ok(single[a]);
                }
                let sum = if a == b && c == d {
                    16. * pair(a, c)
                } else {
                    self.direction([a, b, c, d], v);
                    evaluate(v)?
                };
                Ok((sum
                    - third.get(a, b, c)
                    - third.get(a, b, d)
                    - third.get(a, c, d)
                    - third.get(b, c, d)
                    + pair(a, b)
                    + pair(a, c)
                    + pair(a, d)
                    + pair(b, c)
                    + pair(b, d)
                    + pair(c, d)
                    - single[a]
                    - single[b]
                    - single[c]
                    - single[d])
                    / 24.)
            },
            store,
        )
    }
}

fn pair_index(i: usize, j: usize) -> usize {
    j * (j + 1) / 2 + i
}

fn triples(n: usize) -> impl Iterator<Item = [usize; 3]> {
    (0..n).flat_map(|k| (0..=k).flat_map(move |j| (0..=j).map(move |i| [i, j, k])))
}

pub(crate) fn periodic_cubic_plan(
    dofs: &[usize],
    ndof: usize,
    basis: usize,
    kpoints: usize,
) -> Result<CartesianAssembly<'_>> {
    let workspace = periodic_workspace(ndof, basis, kpoints)?;
    CartesianAssembly::new(3, dofs, ndof, usize::MAX, workspace, 0, 4, 128)
}

pub(crate) fn periodic_workspace(ndof: usize, basis: usize, kpoints: usize) -> Result<usize> {
    // Same per-AO block reserve as the DFPT preflight, including Cartesian
    // responses and complex k-point scratch.
    basis
        .checked_mul(basis)
        .and_then(|n| n.checked_mul(kpoints))
        .and_then(|n| ndof.checked_add(1).and_then(|d| n.checked_mul(d)))
        .and_then(|n| n.checked_mul(48 * 8))
        .ok_or_else(overflow)
}

pub(crate) fn cubic_contraction(
    direction: &[f64],
    mut evaluate: impl FnMut(&[f64]) -> Result<f64>,
) -> Result<crate::linalg::Matrix> {
    use crate::linalg::Matrix;
    let n = direction.len();
    crate::resources::check_dense_workspace(n, n, 1, "FC3 Cartesian contraction")?;
    let fv = evaluate(direction)?;
    let mut singles = Vec::with_capacity(n);
    let mut shifted = Vec::with_capacity(n);
    for i in 0..n {
        let mut e = vec![0.; n];
        e[i] = 1.;
        singles.push(evaluate(&e)?);
        e.copy_from_slice(direction);
        e[i] += 1.;
        shifted.push(evaluate(&e)?);
    }
    let mut out = Matrix::zeros(n, n);
    let mut e = vec![0.; n];
    for i in 0..n {
        for j in i..n {
            e.fill(0.);
            e[i] += 1.;
            e[j] += 1.;
            let pair = if i == j {
                8. * singles[i]
            } else {
                evaluate(&e)?
            };
            for k in 0..n {
                e[k] += direction[k];
            }
            let value =
                (evaluate(&e)? - pair - shifted[i] - shifted[j] + singles[i] + singles[j] + fv)
                    / 6.;
            out[(i, j)] = value;
            out[(j, i)] = value;
        }
    }
    Ok(out)
}
