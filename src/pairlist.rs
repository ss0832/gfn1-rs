// SPDX-License-Identifier: GPL-3.0-or-later
//! Cutoff pair-list construction for short-range real-space terms.
//!
//! The routines in this module enumerate each symmetry-unique atom/image pair
//! once.  They keep the previous exact image semantics, but replace the inner
//! all-pairs scan by a Cartesian cell-list query for each retained lattice
//! image.  This is deliberately conservative: no pair inside the requested
//! cutoff is dropped, even for skew cells or small cells where several images of
//! the same atom pair are within the cutoff.

use crate::error::{Gfn1Error, Result};
use crate::lattice::ImageOffset;
use crate::math::Vec3;
use crate::system::PeriodicSystem;

const DIST_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct ShortRangePair {
    pub i: usize,
    pub j: usize,
    pub offset: ImageOffset,
    pub translation: Vec3,
    /// Displacement from atom i in the reference cell to atom j in `offset`.
    pub dr: Vec3,
    pub r2: f64,
    pub r: f64,
}

impl ShortRangePair {
    #[inline]
    pub fn is_self_image(&self) -> bool {
        self.i == self.j && !self.offset.is_origin()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PairListOptions {
    pub cutoff: f64,
    pub cell_size: f64,
}

impl PairListOptions {
    pub fn new(cutoff: f64) -> Result<Self> {
        if cutoff <= 0.0 || !cutoff.is_finite() {
            return Err(Gfn1Error::InvalidInput(format!(
                "pair-list cutoff must be positive, got {cutoff}"
            )));
        }
        Ok(Self {
            cutoff,
            cell_size: cutoff.max(1.0e-8),
        })
    }
}

fn molecular_all_unique_pairs(system: &PeriodicSystem) -> Vec<ShortRangePair> {
    let mut pairs = Vec::new();
    for i in 0..system.atoms.len() {
        let ri = system.atoms[i].position;
        for j in (i + 1)..system.atoms.len() {
            let dr = system.atoms[j].position - ri;
            let r2 = dr.norm2();
            if r2 <= DIST_EPS * DIST_EPS {
                continue;
            }
            pairs.push(ShortRangePair {
                i,
                j,
                offset: ImageOffset::origin(),
                translation: Vec3::zero(),
                dr,
                r2,
                r: r2.sqrt(),
            });
        }
    }
    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    });
    pairs
}

fn molecular_all_directed_pairs(system: &PeriodicSystem) -> Vec<ShortRangePair> {
    let mut pairs = Vec::new();
    for i in 0..system.atoms.len() {
        let ri = system.atoms[i].position;
        for j in 0..system.atoms.len() {
            if i == j {
                continue;
            }
            let dr = system.atoms[j].position - ri;
            let r2 = dr.norm2();
            if r2 <= DIST_EPS * DIST_EPS {
                continue;
            }
            pairs.push(ShortRangePair {
                i,
                j,
                offset: ImageOffset::origin(),
                translation: Vec3::zero(),
                dr,
                r2,
                r: r2.sqrt(),
            });
        }
    }
    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    });
    pairs
}

fn molecular_all_center_neighbors(system: &PeriodicSystem, center: usize) -> Vec<ShortRangePair> {
    let mut pairs = Vec::new();
    if center >= system.atoms.len() {
        return pairs;
    }
    let ri = system.atoms[center].position;
    for j in 0..system.atoms.len() {
        if j == center {
            continue;
        }
        let dr = system.atoms[j].position - ri;
        let r2 = dr.norm2();
        if r2 <= DIST_EPS * DIST_EPS {
            continue;
        }
        pairs.push(ShortRangePair {
            i: center,
            j,
            offset: ImageOffset::origin(),
            translation: Vec3::zero(),
            dr,
            r2,
            r: r2.sqrt(),
        });
    }
    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.j.cmp(&b.j))
    });
    pairs
}

fn cutoff_disabled_for_molecule(system: &PeriodicSystem, cutoff: f64) -> bool {
    system.lattice.is_none() && (cutoff <= 0.0 || !cutoff.is_finite())
}

/// Enumerate all symmetry-unique atom/image pairs within the cutoff.
///
/// For the origin image only `i < j` pairs are emitted.  For non-origin images,
/// only one vector from the `T`/`-T` pair is emitted and all `(i,j)` pairs are
/// kept.  Pair-energy loops can therefore add `q_i q_j K(r)` or `E_ij(r)` once,
/// while potential loops must update both endpoints; self-image pairs contribute
/// a factor of two to the derivative with respect to the single stored charge.
pub fn unique_short_range_pairs(
    system: &PeriodicSystem,
    cutoff: f64,
) -> Result<Vec<ShortRangePair>> {
    if cutoff_disabled_for_molecule(system, cutoff) {
        return Ok(molecular_all_unique_pairs(system));
    }
    let opts = PairListOptions::new(cutoff)?;
    unique_short_range_pairs_with_options(system, opts)
}

pub fn unique_short_range_pairs_with_options(
    system: &PeriodicSystem,
    options: PairListOptions,
) -> Result<Vec<ShortRangePair>> {
    let cutoff = options.cutoff;
    let cutoff2 = cutoff * cutoff;
    let mut pairs = Vec::new();
    if system.atoms.is_empty() {
        return Ok(pairs);
    }

    let images = image_offsets_for_cutoff(system, cutoff);
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let grid = CartesianCellList::build(&positions, options.cell_size);

    for offset in images {
        if !offset.is_origin() && !canonical_positive_offset(offset) {
            continue;
        }
        let translation = system
            .lattice
            .as_ref()
            .map(|lat| lat.translation(offset))
            .unwrap_or_else(Vec3::zero);
        for (i, &ri) in positions.iter().enumerate() {
            // A site j+T is within cutoff of ri iff the home-cell point j is
            // within cutoff of ri-T.  Querying a single home-cell grid this way
            // avoids rebuilding an identical Cartesian cell list for every
            // retained lattice image.
            let query_point = ri - translation;
            for j in grid.query(query_point, cutoff) {
                if offset.is_origin() && j <= i {
                    continue;
                }
                let dr = positions[j] + translation - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS || r2 > cutoff2 {
                    continue;
                }
                pairs.push(ShortRangePair {
                    i,
                    j,
                    offset,
                    translation,
                    dr,
                    r2,
                    r: r2.sqrt(),
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.n.cmp(&b.offset.n))
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    });
    Ok(pairs)
}

/// Stream symmetry-unique atom/image pairs without materializing the full pair list.
///
/// The cutoff semantics are identical to [`unique_short_range_pairs`].  The
/// enumeration order is intentionally not sorted, which avoids an O(npair log
/// npair) sort and a large temporary allocation for hot real-space Ewald,
/// repulsion, and SCC residual loops.  Callers that require deterministic
/// distance-sorted output should continue to use [`unique_short_range_pairs`].
pub fn for_each_unique_short_range_pair<F>(
    system: &PeriodicSystem,
    cutoff: f64,
    mut f: F,
) -> Result<()>
where
    F: FnMut(ShortRangePair) -> Result<()>,
{
    if cutoff_disabled_for_molecule(system, cutoff) {
        for i in 0..system.atoms.len() {
            let ri = system.atoms[i].position;
            for j in (i + 1)..system.atoms.len() {
                let dr = system.atoms[j].position - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS {
                    continue;
                }
                f(ShortRangePair {
                    i,
                    j,
                    offset: ImageOffset::origin(),
                    translation: Vec3::zero(),
                    dr,
                    r2,
                    r: r2.sqrt(),
                })?;
            }
        }
        return Ok(());
    }
    let opts = PairListOptions::new(cutoff)?;
    let cutoff2 = opts.cutoff * opts.cutoff;
    if system.atoms.is_empty() {
        return Ok(());
    }

    let images = image_offsets_for_cutoff(system, opts.cutoff);
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let grid = CartesianCellList::build(&positions, opts.cell_size);

    for offset in images {
        if !offset.is_origin() && !canonical_positive_offset(offset) {
            continue;
        }
        let translation = system
            .lattice
            .as_ref()
            .map(|lat| lat.translation(offset))
            .unwrap_or_else(Vec3::zero);
        for (i, &ri) in positions.iter().enumerate() {
            let query_point = ri - translation;
            grid.for_each_query(query_point, opts.cutoff, |j| {
                if offset.is_origin() && j <= i {
                    return Ok(());
                }
                let dr = positions[j] + translation - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS || r2 > cutoff2 {
                    return Ok(());
                }
                f(ShortRangePair {
                    i,
                    j,
                    offset,
                    translation,
                    dr,
                    r2,
                    r: r2.sqrt(),
                })
            })?;
        }
    }
    Ok(())
}

/// Stream directed atom/image pairs without materializing the full pair list.
///
/// This preserves the directed semantics of [`directed_short_range_pairs`] and
/// is intended for orientation-sensitive kernels such as AES q--mu terms.
pub fn for_each_directed_short_range_pair<F>(
    system: &PeriodicSystem,
    cutoff: f64,
    mut f: F,
) -> Result<()>
where
    F: FnMut(ShortRangePair) -> Result<()>,
{
    if cutoff_disabled_for_molecule(system, cutoff) {
        for i in 0..system.atoms.len() {
            let ri = system.atoms[i].position;
            for j in 0..system.atoms.len() {
                if i == j {
                    continue;
                }
                let dr = system.atoms[j].position - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS {
                    continue;
                }
                f(ShortRangePair {
                    i,
                    j,
                    offset: ImageOffset::origin(),
                    translation: Vec3::zero(),
                    dr,
                    r2,
                    r: r2.sqrt(),
                })?;
            }
        }
        return Ok(());
    }
    let opts = PairListOptions::new(cutoff)?;
    let cutoff2 = opts.cutoff * opts.cutoff;
    if system.atoms.is_empty() {
        return Ok(());
    }

    let images = image_offsets_for_cutoff(system, opts.cutoff);
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let grid = CartesianCellList::build(&positions, opts.cell_size);

    for offset in images {
        let translation = system
            .lattice
            .as_ref()
            .map(|lat| lat.translation(offset))
            .unwrap_or_else(Vec3::zero);
        for (i, &ri) in positions.iter().enumerate() {
            let query_point = ri - translation;
            grid.for_each_query(query_point, opts.cutoff, |j| {
                if offset.is_origin() && j == i {
                    return Ok(());
                }
                let dr = positions[j] + translation - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS || r2 > cutoff2 {
                    return Ok(());
                }
                f(ShortRangePair {
                    i,
                    j,
                    offset,
                    translation,
                    dr,
                    r2,
                    r: r2.sqrt(),
                })
            })?;
        }
    }
    Ok(())
}

/// Enumerate all directed atom/image pairs within the cutoff using the same
/// cell-list acceleration as [`unique_short_range_pairs`].
///
/// This preserves the legacy directed semantics used by orientation-sensitive
/// kernels such as q--mu AES: origin-image `(i,j)` and `(j,i)` are both present,
/// and non-origin `+T` and `-T` images are both present.  Energy expressions
/// written for directed loops should therefore keep their existing `1/2` factor.
pub fn directed_short_range_pairs(
    system: &PeriodicSystem,
    cutoff: f64,
) -> Result<Vec<ShortRangePair>> {
    if cutoff_disabled_for_molecule(system, cutoff) {
        return Ok(molecular_all_directed_pairs(system));
    }
    let opts = PairListOptions::new(cutoff)?;
    let cutoff2 = opts.cutoff * opts.cutoff;
    let mut pairs = Vec::new();
    if system.atoms.is_empty() {
        return Ok(pairs);
    }

    let images = image_offsets_for_cutoff(system, opts.cutoff);
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let grid = CartesianCellList::build(&positions, opts.cell_size);

    for offset in images {
        let translation = system
            .lattice
            .as_ref()
            .map(|lat| lat.translation(offset))
            .unwrap_or_else(Vec3::zero);
        for (i, &ri) in positions.iter().enumerate() {
            let query_point = ri - translation;
            for j in grid.query(query_point, opts.cutoff) {
                if offset.is_origin() && j == i {
                    continue;
                }
                let dr = positions[j] + translation - ri;
                let r2 = dr.norm2();
                if r2 <= DIST_EPS * DIST_EPS || r2 > cutoff2 {
                    continue;
                }
                pairs.push(ShortRangePair {
                    i,
                    j,
                    offset,
                    translation,
                    dr,
                    r2,
                    r: r2.sqrt(),
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.offset.n.cmp(&b.offset.n))
            .then_with(|| a.i.cmp(&b.i))
            .then_with(|| a.j.cmp(&b.j))
    });
    Ok(pairs)
}

/// Enumerate directed neighbors for every home-cell center atom in one pass.
///
/// This is equivalent to calling [`center_short_range_neighbors`] for every
/// center, but it constructs the directed atom/image pair list only once.  It is
/// therefore the preferred backend for three-body terms such as D4 ATM, where
/// rebuilding a cell list for every center would otherwise add an avoidable
/// O(N_center * N_image * N_atom) setup cost.
pub fn all_center_short_range_neighbors(
    system: &PeriodicSystem,
    cutoff: f64,
) -> Result<Vec<Vec<ShortRangePair>>> {
    let nat = system.atoms.len();
    let mut grouped = vec![Vec::new(); nat];
    if nat == 0 {
        return Ok(grouped);
    }

    if cutoff_disabled_for_molecule(system, cutoff) {
        for center in 0..nat {
            grouped[center] = molecular_all_center_neighbors(system, center);
        }
        return Ok(grouped);
    }

    for_each_directed_short_range_pair(system, cutoff, |pair| {
        if pair.i < nat {
            grouped[pair.i].push(pair);
        }
        Ok(())
    })?;
    for neigh in &mut grouped {
        neigh.sort_by(|a, b| {
            a.r2.partial_cmp(&b.r2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.j.cmp(&b.j))
                .then_with(|| a.offset.n.cmp(&b.offset.n))
        });
    }
    Ok(grouped)
}

/// Enumerate directed neighbors of a single home-cell center atom with a
/// cell-list query for each retained image.
pub fn center_short_range_neighbors(
    system: &PeriodicSystem,
    center: usize,
    cutoff: f64,
) -> Result<Vec<ShortRangePair>> {
    if center >= system.atoms.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "center atom index {center} out of range"
        )));
    }
    if cutoff_disabled_for_molecule(system, cutoff) {
        return Ok(molecular_all_center_neighbors(system, center));
    }
    let opts = PairListOptions::new(cutoff)?;
    let cutoff2 = opts.cutoff * opts.cutoff;
    let mut pairs = Vec::new();
    if system.atoms.is_empty() {
        return Ok(pairs);
    }

    let ri = system.atoms[center].position;
    let positions = system.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
    let grid = CartesianCellList::build(&positions, opts.cell_size);
    for offset in image_offsets_for_cutoff(system, opts.cutoff) {
        let translation = system
            .lattice
            .as_ref()
            .map(|lat| lat.translation(offset))
            .unwrap_or_else(Vec3::zero);
        let query_point = ri - translation;
        for j in grid.query(query_point, opts.cutoff) {
            if offset.is_origin() && j == center {
                continue;
            }
            let dr = positions[j] + translation - ri;
            let r2 = dr.norm2();
            if r2 <= DIST_EPS * DIST_EPS || r2 > cutoff2 {
                continue;
            }
            pairs.push(ShortRangePair {
                i: center,
                j,
                offset,
                translation,
                dr,
                r2,
                r: r2.sqrt(),
            });
        }
    }
    pairs.sort_by(|a, b| {
        a.r2.partial_cmp(&b.r2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.j.cmp(&b.j))
            .then_with(|| a.offset.n.cmp(&b.offset.n))
    });
    Ok(pairs)
}

#[inline]
pub fn image_offsets_for_cutoff(system: &PeriodicSystem, cutoff: f64) -> Vec<ImageOffset> {
    match &system.lattice {
        Some(lattice) => lattice.image_offsets(cutoff),
        None => vec![ImageOffset::origin()],
    }
}

#[inline]
pub fn canonical_positive_offset(offset: ImageOffset) -> bool {
    for n in offset.n {
        if n > 0 {
            return true;
        }
        if n < 0 {
            return false;
        }
    }
    false
}

#[derive(Clone, Debug)]
struct CartesianCellList {
    min: Vec3,
    inv_cell: f64,
    dims: [usize; 3],
    buckets: Vec<Vec<usize>>,
}

impl CartesianCellList {
    fn build(points: &[Vec3], cell_size: f64) -> Self {
        let mut min = points[0];
        let mut max = points[0];
        for &p in points.iter().skip(1) {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        let size = cell_size.max(1.0e-8);
        let inv_cell = 1.0 / size;
        let extent = max - min;
        let dims = [
            ((extent.x * inv_cell).floor() as usize + 1).max(1),
            ((extent.y * inv_cell).floor() as usize + 1).max(1),
            ((extent.z * inv_cell).floor() as usize + 1).max(1),
        ];
        let mut buckets = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        for (idx, &p) in points.iter().enumerate() {
            let c = Self::coord_static(min, inv_cell, dims, p);
            buckets[Self::linear_static(dims, c)].push(idx);
        }
        Self {
            min,
            inv_cell,
            dims,
            buckets,
        }
    }

    fn query(&self, point: Vec3, cutoff: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let _ = self.for_each_query(point, cutoff, |idx| {
            out.push(idx);
            Ok(())
        });
        out
    }

    fn for_each_query<F>(&self, point: Vec3, cutoff: f64, mut f: F) -> Result<()>
    where
        F: FnMut(usize) -> Result<()>,
    {
        let center = self.coord(point);
        let radius = (cutoff * self.inv_cell).ceil() as isize + 1;
        for dx in -radius..=radius {
            let Some(ix) = checked_axis(center[0], dx, self.dims[0]) else {
                continue;
            };
            for dy in -radius..=radius {
                let Some(iy) = checked_axis(center[1], dy, self.dims[1]) else {
                    continue;
                };
                for dz in -radius..=radius {
                    let Some(iz) = checked_axis(center[2], dz, self.dims[2]) else {
                        continue;
                    };
                    let bucket = &self.buckets[Self::linear_static(self.dims, [ix, iy, iz])];
                    for &idx in bucket {
                        f(idx)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn coord(&self, p: Vec3) -> [usize; 3] {
        Self::coord_static(self.min, self.inv_cell, self.dims, p)
    }

    fn coord_static(min: Vec3, inv_cell: f64, dims: [usize; 3], p: Vec3) -> [usize; 3] {
        [
            clamp_cell(((p.x - min.x) * inv_cell).floor() as isize, dims[0]),
            clamp_cell(((p.y - min.y) * inv_cell).floor() as isize, dims[1]),
            clamp_cell(((p.z - min.z) * inv_cell).floor() as isize, dims[2]),
        ]
    }

    #[inline]
    fn linear_static(dims: [usize; 3], c: [usize; 3]) -> usize {
        (c[0] * dims[1] + c[1]) * dims[2] + c[2]
    }
}

#[inline]
fn checked_axis(center: usize, delta: isize, dim: usize) -> Option<usize> {
    let v = center as isize + delta;
    if v < 0 || v >= dim as isize {
        None
    } else {
        Some(v as usize)
    }
}

#[inline]
fn clamp_cell(v: isize, dim: usize) -> usize {
    if v < 0 {
        0
    } else if v >= dim as isize {
        dim - 1
    } else {
        v as usize
    }
}
