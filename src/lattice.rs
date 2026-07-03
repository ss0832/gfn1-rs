// SPDX-License-Identifier: GPL-3.0-or-later
use crate::error::{Gfn1Error, Result};
use crate::math::{reciprocal_vectors_2pi, Mat3, Vec3};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const DET_EPS: f64 = 1.0e-14;
const RANGE_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LatticeListCacheKey {
    cell_bits: [u64; 9],
    periodic: [bool; 3],
    cutoff_bits: u64,
    include_zero: bool,
}

static IMAGE_OFFSET_CACHE: OnceLock<Mutex<HashMap<LatticeListCacheKey, Vec<ImageOffset>>>> =
    OnceLock::new();
static RECIPROCAL_VECTOR_CACHE: OnceLock<
    Mutex<HashMap<LatticeListCacheKey, Vec<([i32; 3], Vec3)>>>,
> = OnceLock::new();

fn cache_key(
    cell: Mat3,
    periodic: [bool; 3],
    cutoff: f64,
    include_zero: bool,
) -> LatticeListCacheKey {
    let c = cell.col;
    LatticeListCacheKey {
        cell_bits: [
            c[0].x.to_bits(),
            c[0].y.to_bits(),
            c[0].z.to_bits(),
            c[1].x.to_bits(),
            c[1].y.to_bits(),
            c[1].z.to_bits(),
            c[2].x.to_bits(),
            c[2].y.to_bits(),
            c[2].z.to_bits(),
        ],
        periodic,
        cutoff_bits: cutoff.to_bits(),
        include_zero,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageOffset {
    pub n: [i32; 3],
}

impl ImageOffset {
    #[inline]
    pub const fn origin() -> Self {
        Self { n: [0, 0, 0] }
    }
    #[inline]
    pub fn is_origin(self) -> bool {
        self.n == [0, 0, 0]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lattice {
    pub cell: Mat3,
    inv_rows: [Vec3; 3],
    pub periodic: [bool; 3],
}

impl Lattice {
    pub fn new(cell: Mat3, periodic: [bool; 3]) -> Result<Self> {
        let a = cell.col[0];
        let b = cell.col[1];
        let c = cell.col[2];
        let det = a.dot(b.cross(c));
        if det.abs() < DET_EPS {
            return Err(Gfn1Error::SingularCell);
        }
        let inv_rows = [b.cross(c) / det, c.cross(a) / det, a.cross(b) / det];
        Ok(Self {
            cell,
            inv_rows,
            periodic,
        })
    }

    pub fn from_vectors(a: Vec3, b: Vec3, c: Vec3, periodic: [bool; 3]) -> Result<Self> {
        Self::new(Mat3::from_columns(a, b, c), periodic)
    }

    pub fn from_lengths_angles(
        a: f64,
        b: f64,
        c: f64,
        alpha_deg: f64,
        beta_deg: f64,
        gamma_deg: f64,
        periodic: [bool; 3],
    ) -> Result<Self> {
        let deg = std::f64::consts::PI / 180.0;
        let alpha = alpha_deg * deg;
        let beta = beta_deg * deg;
        let gamma = gamma_deg * deg;
        let avec = Vec3::new(a, 0.0, 0.0);
        let bvec = Vec3::new(b * gamma.cos(), b * gamma.sin(), 0.0);
        let cx = c * beta.cos();
        let cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin();
        let cz2 = c * c - cx * cx - cy * cy;
        if cz2 <= 0.0 {
            return Err(Gfn1Error::InvalidInput(
                "cell angles produce an invalid triclinic lattice".to_string(),
            ));
        }
        Self::from_vectors(avec, bvec, Vec3::new(cx, cy, cz2.sqrt()), periodic)
    }

    #[inline]
    pub fn volume(&self) -> f64 {
        self.cell.volume()
    }

    #[inline]
    pub fn frac_of(&self, cart: Vec3) -> Vec3 {
        Vec3::new(
            self.inv_rows[0].dot(cart),
            self.inv_rows[1].dot(cart),
            self.inv_rows[2].dot(cart),
        )
    }

    #[inline]
    pub fn inverse_rows(&self) -> [Vec3; 3] {
        self.inv_rows
    }

    #[inline]
    pub fn cart_of(&self, frac: Vec3) -> Vec3 {
        self.cell.mul_vec(frac)
    }

    pub fn wrap_frac(&self, mut frac: Vec3) -> Vec3 {
        if self.periodic[0] {
            frac.x -= frac.x.floor();
        }
        if self.periodic[1] {
            frac.y -= frac.y.floor();
        }
        if self.periodic[2] {
            frac.z -= frac.z.floor();
        }
        frac
    }

    /// Wrap fractional coordinates into [-1/2, 1/2) on periodic axes.  This is
    /// useful for displacement-like quantities and avoids the asymmetric behavior
    /// of wrapping into [0, 1).
    pub fn wrap_frac_centered(&self, mut frac: Vec3) -> Vec3 {
        if self.periodic[0] {
            frac.x -= (frac.x + 0.5).floor();
        }
        if self.periodic[1] {
            frac.y -= (frac.y + 0.5).floor();
        }
        if self.periodic[2] {
            frac.z -= (frac.z + 0.5).floor();
        }
        frac
    }

    pub fn wrap_cart(&self, cart: Vec3) -> Vec3 {
        self.cart_of(self.wrap_frac(self.frac_of(cart)))
    }

    pub fn minimum_image(&self, delta: Vec3) -> Vec3 {
        self.minimum_image_with_offset(delta).0
    }

    /// Return the minimum-image displacement and the integer image offset that was
    /// subtracted from the second point.
    ///
    /// For skew/triclinic cells, rounding each fractional component can miss the
    /// true nearest lattice image.  We therefore inspect the 3x3x3 stencil around
    /// the rounded fractional offset on periodic axes.  This keeps direct PBC pair
    /// loops, CN, H0 polynomial factors, D4 and local AES damping consistent for
    /// non-orthogonal cells.
    pub fn minimum_image_with_offset(&self, delta: Vec3) -> (Vec3, ImageOffset) {
        let frac = self.frac_of(delta);
        let center = [
            if self.periodic[0] {
                frac.x.round() as i32
            } else {
                0
            },
            if self.periodic[1] {
                frac.y.round() as i32
            } else {
                0
            },
            if self.periodic[2] {
                frac.z.round() as i32
            } else {
                0
            },
        ];
        let mut best = delta;
        let mut best2 = delta.norm2();
        let mut best_off = [0_i32; 3];
        let rx = if self.periodic[0] { -1..=1 } else { 0..=0 };
        let ry = if self.periodic[1] { -1..=1 } else { 0..=0 };
        let rz = if self.periodic[2] { -1..=1 } else { 0..=0 };
        for ix in rx {
            for iy in ry.clone() {
                for iz in rz.clone() {
                    let off = [center[0] + ix, center[1] + iy, center[2] + iz];
                    let wrapped = Vec3::new(
                        frac.x - off[0] as f64,
                        frac.y - off[1] as f64,
                        frac.z - off[2] as f64,
                    );
                    let cart = self.cart_of(wrapped);
                    let r2 = cart.norm2();
                    if r2 < best2 {
                        best = cart;
                        best2 = r2;
                        best_off = off;
                    }
                }
            }
        }
        (best, ImageOffset { n: best_off })
    }

    #[inline]
    pub fn translation(&self, offset: ImageOffset) -> Vec3 {
        self.cell.col[0] * offset.n[0] as f64
            + self.cell.col[1] * offset.n[1] as f64
            + self.cell.col[2] * offset.n[2] as f64
    }

    pub fn plane_height(&self, axis: usize) -> f64 {
        basis_plane_height(self.cell.col, axis)
    }

    /// Conservative integer image range for a sphere of radius `cutoff` in the
    /// direct lattice.  The bound uses the height of the lattice planes rather
    /// than the vector lengths, which is required for skew/triclinic cells.
    pub fn image_ranges(&self, cutoff: f64) -> [i32; 3] {
        let mut range = [0_i32; 3];
        if cutoff <= 0.0 {
            return range;
        }
        for axis in 0..3 {
            if self.periodic[axis] {
                let h = self.plane_height(axis).max(RANGE_EPS);
                range[axis] = (cutoff / h).ceil() as i32 + 1;
            }
        }
        range
    }

    pub fn image_offsets(&self, cutoff: f64) -> Vec<ImageOffset> {
        let key = cache_key(self.cell, self.periodic, cutoff, false);
        let cache = IMAGE_OFFSET_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(cached) = cache
            .lock()
            .expect("image offset cache poisoned")
            .get(&key)
            .cloned()
        {
            return cached;
        }

        let range = self.image_ranges(cutoff);
        let mut out = Vec::new();
        for i in -range[0]..=range[0] {
            for j in -range[1]..=range[1] {
                for k in -range[2]..=range[2] {
                    out.push(ImageOffset { n: [i, j, k] });
                }
            }
        }
        out.sort_by(|a, b| {
            let ra = self.translation(*a).norm2();
            let rb = self.translation(*b).norm2();
            ra.partial_cmp(&rb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.n.cmp(&b.n))
        });
        cache
            .lock()
            .expect("image offset cache poisoned")
            .insert(key, out.clone());
        out
    }

    pub fn reciprocal_vectors_2pi(&self) -> [Vec3; 3] {
        reciprocal_vectors_2pi(self.cell)
    }

    /// Conservative Miller-index range for reciprocal vectors |G| <= cutoff.
    /// Using |b_i| for this bound misses vectors in skew cells because reciprocal
    /// basis vectors can cancel.  The plane-height criterion is the correct safe
    /// bound for triclinic cells.
    pub fn reciprocal_index_ranges(&self, cutoff: f64) -> [i32; 3] {
        reciprocal_index_ranges_from_basis(self.reciprocal_vectors_2pi(), cutoff)
    }

    /// Enumerate reciprocal lattice vectors inside a sphere.  The returned list is
    /// sorted by |G| for deterministic floating-point accumulation.
    pub fn reciprocal_vectors_within(
        &self,
        cutoff: f64,
        include_zero: bool,
    ) -> Vec<([i32; 3], Vec3)> {
        let key = cache_key(self.cell, self.periodic, cutoff, include_zero);
        let cache = RECIPROCAL_VECTOR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(cached) = cache
            .lock()
            .expect("reciprocal vector cache poisoned")
            .get(&key)
            .cloned()
        {
            return cached;
        }

        let recip = self.reciprocal_vectors_2pi();
        let range = reciprocal_index_ranges_from_basis(recip, cutoff);
        let cutoff2 = cutoff * cutoff;
        let mut out = Vec::new();
        for h in -range[0]..=range[0] {
            for k in -range[1]..=range[1] {
                for l in -range[2]..=range[2] {
                    if !include_zero && h == 0 && k == 0 && l == 0 {
                        continue;
                    }
                    let g = recip[0] * h as f64 + recip[1] * k as f64 + recip[2] * l as f64;
                    let g2 = g.norm2();
                    if g2 <= cutoff2 + 64.0 * f64::EPSILON * cutoff2.max(1.0) {
                        out.push(([h, k, l], g));
                    }
                }
            }
        }
        out.sort_by(|a, b| {
            let ra = a.1.norm2();
            let rb = b.1.norm2();
            ra.partial_cmp(&rb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        cache
            .lock()
            .expect("reciprocal vector cache poisoned")
            .insert(key, out.clone());
        out
    }

    #[inline]
    pub fn is_fully_periodic(&self) -> bool {
        self.periodic == [true, true, true]
    }
}

pub fn reciprocal_index_ranges_from_basis(basis: [Vec3; 3], cutoff: f64) -> [i32; 3] {
    let mut range = [0_i32; 3];
    if cutoff <= 0.0 {
        return range;
    }
    for axis in 0..3 {
        let h = basis_plane_height(basis, axis).max(RANGE_EPS);
        range[axis] = (cutoff / h).ceil() as i32 + 1;
    }
    range
}

fn basis_plane_height(basis: [Vec3; 3], axis: usize) -> f64 {
    let a = basis[0];
    let b = basis[1];
    let c = basis[2];
    let volume = a.dot(b.cross(c)).abs();
    match axis {
        0 => volume / b.cross(c).norm().max(RANGE_EPS),
        1 => volume / c.cross(a).norm().max(RANGE_EPS),
        2 => volume / a.cross(b).norm().max(RANGE_EPS),
        _ => unreachable!("axis must be 0..3"),
    }
}
