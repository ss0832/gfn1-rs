// SPDX-License-Identifier: GPL-3.0-or-later
//! Experimental **parameter-free multipole electrostatics (mDFTB2)** correction for the
//! non-periodic GFN1-xTB SCC, restricted to the energy and analytic gradient.
//!
//! Reference: V.-Q. Vuong, B. Aradi, A. M. N. Niklasson, Q. Cui, S. Irle,
//! "Multipole Expansion of Atomic Electron Density Fluctuation Interactions in the
//! Density-Functional Tight-Binding Method", *J. Chem. Theory Comput.* **19**, 7592
//! (2023), DOI 10.1021/acs.jctc.3c00778.
//!
//! mDFTB2 extends the second-order term with atomic **dipole** `d_A` and **quadrupole**
//! `Q_A` density-fluctuation moments (eq 15/27/32). It is *parameter-free*: the
//! multipole interaction tensors `f^(mn)_AB` are spatial derivatives of the **same**
//! `γ` profile already used for the monopole electrostatics (eq 18), and the atomic
//! moments come from the density via AO dipole/quadrupole integrals (on-site
//! approximation, eq 29-32). Here we add it as a self-consistent **correction on top of
//! GFN1**: GFN1's existing shell-resolved monopole-monopole term is kept, and only the
//! terms involving a `d` or `Q` are added.
//!
//! This module currently provides the kernel-tensor machinery `f^(mn)` (the only new
//! "physics"); the atomic moments, the energy/Fock self-consistency, and the analytic
//! gradient build on top of it.

use crate::basis::BasisSet;
use crate::integrals::IntegralMatrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use rayon::prelude::*;

/// Atomic density-fluctuation multipole moments (mDFTB2): the dipole `Δd_A` and the
/// **traceless** quadrupole `ΔQ_A` per atom, from the on-site approximation (eq 32).
/// The spherical reference density carries no dipole nor traceless quadrupole, so the
/// fluctuation moments are taken from the full density `P` directly. The quadrupole is
/// stored as a symmetric `3x3` (already trace-removed, eq 20).
#[derive(Clone, Debug)]
pub struct AtomicMoments {
    pub dipole: Vec<Vec3>,
    pub quad: Vec<[[f64; 3]; 3]>,
}

/// Number of multipole mixing-vector components per atom: 3 dipole + 6 unique (symmetric)
/// quadrupole = 9. The SCC mixes these alongside the monopole shell charges (tblite-style
/// multipole SCF), so a quasi-Newton (Broyden) mixer captures the monopole/multipole coupling.
pub const MOMENT_STRIDE: usize = 9;

/// Flatten the atomic moments into `out` (length `MOMENT_STRIDE*nat`): per atom
/// `[dx, dy, dz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]`.
pub fn pack_moments(moments: &AtomicMoments, out: &mut [f64]) {
    for a in 0..moments.dipole.len() {
        let o = &mut out[a * MOMENT_STRIDE..(a + 1) * MOMENT_STRIDE];
        let d = moments.dipole[a];
        let q = &moments.quad[a];
        o[0] = d.x;
        o[1] = d.y;
        o[2] = d.z;
        o[3] = q[0][0];
        o[4] = q[0][1];
        o[5] = q[0][2];
        o[6] = q[1][1];
        o[7] = q[1][2];
        o[8] = q[2][2];
    }
}

/// Inverse of [`pack_moments`]; the quadrupole is symmetrized and re-trace-removed (it stays
/// traceless under linear mixing, but enforce it to be safe).
pub fn unpack_moments(v: &[f64], nat: usize) -> AtomicMoments {
    let mut dipole = vec![Vec3::zero(); nat];
    let mut quad = vec![[[0.0_f64; 3]; 3]; nat];
    for a in 0..nat {
        let o = &v[a * MOMENT_STRIDE..(a + 1) * MOMENT_STRIDE];
        dipole[a] = Vec3::new(o[0], o[1], o[2]);
        let mut q = [[o[3], o[4], o[5]], [o[4], o[6], o[7]], [o[5], o[7], o[8]]];
        let tr = (q[0][0] + q[1][1] + q[2][2]) / 3.0;
        for i in 0..3 {
            q[i][i] -= tr;
        }
        quad[a] = q;
    }
    AtomicMoments { dipole, quad }
}

/// Mixing-vector components per atom for the experimental **octupole** block: the 10
/// unique Cartesian components of the symmetric rank-3 moment. Appended after the
/// monopole/dipole/quadrupole vector when the octupole correction is on.
pub const OCTU_STRIDE: usize = 10;

/// Flatten the atomic octupoles into `out` (length `OCTU_STRIDE*nat`): per atom the 10
/// unique components `[xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz]`.
pub fn pack_octu(octu: &[[[[f64; 3]; 3]; 3]], out: &mut [f64]) {
    for (a, o) in octu.iter().enumerate() {
        let s = &mut out[a * OCTU_STRIDE..(a + 1) * OCTU_STRIDE];
        s[0] = o[0][0][0];
        s[1] = o[0][0][1];
        s[2] = o[0][0][2];
        s[3] = o[0][1][1];
        s[4] = o[0][1][2];
        s[5] = o[0][2][2];
        s[6] = o[1][1][1];
        s[7] = o[1][1][2];
        s[8] = o[1][2][2];
        s[9] = o[2][2][2];
    }
}

/// Inverse of [`pack_octu`]; the octupole is rebuilt symmetric and re-trace-removed (it
/// stays traceless under linear mixing, but enforce it to be safe).
pub fn unpack_octu(v: &[f64], nat: usize) -> Vec<[[[f64; 3]; 3]; 3]> {
    let mut out = vec![[[[0.0_f64; 3]; 3]; 3]; nat];
    for (a, slot) in out.iter_mut().enumerate() {
        let s = &v[a * OCTU_STRIDE..(a + 1) * OCTU_STRIDE];
        let comps = [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]];
        *slot = detrace_octupole(&octu_from_components(&comps));
    }
    out
}

/// Number of mixing-vector components per atom for the **arbitrary-rank** generic multipole SCF:
/// `Σ_{l=1}^{max_rank} (l+1)(l+2)/2` (the unique symmetric Cartesian components of ranks 1..L,
/// excluding the rank-0 monopole, which is mixed as the GFN1 shell charges). For `max_rank=2` this
/// is `3+6=9 = MOMENT_STRIDE`; adding rank 3 gives `+10 = OCTU_STRIDE` — i.e. the generic stride
/// nests the legacy dipole/quad/octupole strides exactly.
pub fn generic_moment_stride(max_rank: usize) -> usize {
    (1..=max_rank).map(|l| (l + 1) * (l + 2) / 2).sum()
}

/// Representative flat (row-major `3^l`) index for the symmetric component with axis multiplicities
/// `(lx,ly,lz)`: the canonical sequence `0…0 1…1 2…2` read as base-3. Since the tensor is fully
/// symmetric, any permutation gives the same value, so this representative is sufficient.
fn rep_flat_index(lx: usize, ly: usize, lz: usize) -> usize {
    let mut idx = 0usize;
    for _ in 0..lx {
        idx *= 3;
    }
    for _ in 0..ly {
        idx = idx * 3 + 1;
    }
    for _ in 0..lz {
        idx = idx * 3 + 2;
    }
    idx
}

/// Flatten the **arbitrary-rank** atomic moments `moments[a][l]` (full `3^l` row-major detraced
/// Cartesian tensors, `l = 1..=max_rank`; the rank-0 monopole is *not* included) into `out`
/// (length `generic_moment_stride(max_rank)·nat`) as the unique symmetric components per rank, in
/// [`crate::integrals::cartesian_rank_components`] order. The inverse is [`unpack_generic_moments`].
pub fn pack_generic_moments(moments: &[Vec<Vec<f64>>], max_rank: usize, out: &mut [f64]) {
    let stride = generic_moment_stride(max_rank);
    for (a, m) in moments.iter().enumerate() {
        let mut off = a * stride;
        for l in 1..=max_rank {
            for (lx, ly, lz) in crate::integrals::cartesian_rank_components(l) {
                out[off] = m[l][rep_flat_index(lx, ly, lz)];
                off += 1;
            }
        }
    }
}

/// Inverse of [`pack_generic_moments`]: rebuild `moments[a][l]` as full `3^l` row-major tensors for
/// `l = 1..=max_rank`, each re-expanded ([`expand_symmetric_cartesian`]) and re-detraced
/// ([`detrace_symmetric`]) so the tensors stay symmetric-traceless under linear mixing (it does,
/// but enforce it — mirrors [`unpack_octu`]). Index `[a][0]` is a placeholder `[0.0]` (the rank-0
/// monopole is supplied separately from the shell charges by the caller).
pub fn unpack_generic_moments(v: &[f64], nat: usize, max_rank: usize) -> Vec<Vec<Vec<f64>>> {
    let stride = generic_moment_stride(max_rank);
    let mut out = Vec::with_capacity(nat);
    for a in 0..nat {
        let mut per: Vec<Vec<f64>> = vec![vec![0.0]]; // rank-0 placeholder (set by caller from q)
        let mut off = a * stride;
        for l in 1..=max_rank {
            let ncomp = (l + 1) * (l + 2) / 2;
            let unique = &v[off..off + ncomp];
            per.push(detrace_symmetric(&expand_symmetric_cartesian(unique, l), l));
            off += ncomp;
        }
        out.push(per);
    }
    out
}

/// Atom index of each AO and the AO list per atom.
fn atom_ao_lists(basis: &BasisSet, nat: usize) -> Vec<Vec<usize>> {
    let mut per_atom = vec![Vec::new(); nat];
    for (i, ao) in basis.aos.iter().enumerate() {
        per_atom[ao.atom_index].push(i);
    }
    per_atom
}

/// On-site atomic dipole `d̄_{μκ}` (3-vector) and traceless-input quadrupole `Q̄_{μκ}`
/// (symmetric `3x3`) for a same-atom AO pair, read from the ket-centred AO moment
/// integrals (`<μ|(r-R_A)|κ>` etc.). For μ,κ on the same atom A the ket centre is `R_A`.
#[inline]
fn onsite_dipole(ints: &IntegralMatrices, mu: usize, ka: usize) -> Vec3 {
    Vec3::new(
        ints.dipole_x[(mu, ka)],
        ints.dipole_y[(mu, ka)],
        ints.dipole_z[(mu, ka)],
    )
}

#[inline]
fn onsite_quad(ints: &IntegralMatrices, mu: usize, ka: usize) -> [[f64; 3]; 3] {
    let xx = ints.quad_xx[(mu, ka)];
    let xy = ints.quad_xy[(mu, ka)];
    let yy = ints.quad_yy[(mu, ka)];
    let xz = ints.quad_xz[(mu, ka)];
    let yz = ints.quad_yz[(mu, ka)];
    let zz = ints.quad_zz[(mu, ka)];
    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
}

/// Atomic mDFTB2 moments `Δd_A`, `ΔQ_A` (traceless) from the on-site approximation
/// (eq 32) and the (real, symmetric) density matrix `P`:
/// `Δd_A = ½ Σ_{μ∈A,κ∈A,ν} d̄_{μκ}(S_{κν}P_{νμ} + P_{μν}S_{νκ})`, similarly `ΔQ_A` with the
/// second-moment integrals, then trace-removed. `S` is the AO overlap.
pub fn atomic_moments(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    density: &Matrix,
) -> AtomicMoments {
    let s = &integrals.overlap;
    let n = basis.len();
    let per_atom = atom_ao_lists(basis, nat);
    // Each atom's on-site moments are independent; compute them in parallel (the inner ν sum
    // order is preserved per atom, so the result is bit-identical to the serial path). The
    // inner `Σ_ν` over the full overlap row is the O(N) factor that makes this O(N²) overall.
    let moments: Vec<(Vec3, [[f64; 3]; 3])> = per_atom
        .par_iter()
        .map(|aos| {
            let mut d = Vec3::zero();
            let mut q = [[0.0_f64; 3]; 3];
            for &mu in aos {
                for &ka in aos {
                    let dbar = onsite_dipole(integrals, mu, ka);
                    let qbar = onsite_quad(integrals, mu, ka);
                    // w_{μκ} = ½ Σ_ν (S_{κν}P_{νμ} + P_{μν}S_{νκ})
                    let mut w = 0.0;
                    for nu in 0..n {
                        w += 0.5
                            * (s[(ka, nu)] * density[(nu, mu)] + density[(mu, nu)] * s[(nu, ka)]);
                    }
                    d += dbar * w;
                    for i in 0..3 {
                        for j in 0..3 {
                            q[i][j] += qbar[i][j] * w;
                        }
                    }
                }
            }
            // Traceless quadrupole (eq 20): Q^traceless_ij = Q_ij - δij/3 Tr(Q).
            let tr = (q[0][0] + q[1][1] + q[2][2]) / 3.0;
            for (i, qi) in q.iter_mut().enumerate() {
                qi[i] -= tr;
            }
            (d, q)
        })
        .collect();
    let mut dipole = vec![Vec3::zero(); nat];
    let mut quad = vec![[[0.0_f64; 3]; 3]; nat];
    for (a, (d, q)) in moments.into_iter().enumerate() {
        dipole[a] = d;
        quad[a] = q;
    }
    AtomicMoments { dipole, quad }
}

/// **Richer secondary-basis moment integrals.** Return a copy of the `primary` integral set in
/// which the **on-site (same-atom) dipole/quadrupole** AO integrals are recomputed over the
/// node-correct **secondary** (GFN1-xTB-cc-pVnZ) AOs `sec_aos` (built by
/// [`crate::magnetic::build_secondary_aos`], 1:1 with the primary AOs, same centre/angular
/// momentum, richer radial shape). The **overlap is kept primary** — so the Mulliken-style
/// density-weighted population `½(SP+PS)` that contracts these moment integrals is unchanged —
/// while the moment *operator* `⟨μ|(r−R_A)|κ⟩` is better resolved. This mirrors how the M1 dual
/// basis enriches the magnetic kinetic-energy integral; here it enriches the field-free mDFTB
/// dipole/quadrupole moments. Off-site dipole/quad blocks (unused by the on-site moments) are
/// left at their primary values. The rank-3 octupole operator is read from the primary AOs and
/// is *not* enriched here. Non-periodic.
pub fn secondary_moment_integrals(
    primary: &IntegralMatrices,
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    sec_aos: &[crate::basis::AOBasisFunction],
) -> IntegralMatrices {
    let mut out = primary.clone();
    let per_atom = atom_ao_lists(basis, nat);
    for (a, aos) in per_atom.iter().enumerate() {
        let ra = atom_pos[a];
        for &i in aos {
            for &j in aos {
                let p = crate::integrals::contracted_pair(&sec_aos[i], &sec_aos[j], ra, ra);
                // p = (overlap, dx, dy, dz, qxx, qxy, qyy, qxz, qyz, qzz); keep `overlap` primary.
                out.dipole_x[(i, j)] = p.1;
                out.dipole_y[(i, j)] = p.2;
                out.dipole_z[(i, j)] = p.3;
                out.quad_xx[(i, j)] = p.4;
                out.quad_xy[(i, j)] = p.5;
                out.quad_yy[(i, j)] = p.6;
                out.quad_xz[(i, j)] = p.7;
                out.quad_yz[(i, j)] = p.8;
                out.quad_zz[(i, j)] = p.9;
            }
        }
    }
    out
}

/// Build the full symmetric rank-3 octupole tensor from the 10 unique Cartesian
/// components `[xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz]`.
fn octu_from_components(c: &[f64; 10]) -> [[[f64; 3]; 3]; 3] {
    fn set(o: &mut [[[f64; 3]; 3]; 3], a: usize, b: usize, d: usize, v: f64) {
        for &(i, j, k) in &[
            (a, b, d),
            (a, d, b),
            (b, a, d),
            (b, d, a),
            (d, a, b),
            (d, b, a),
        ] {
            o[i][j][k] = v;
        }
    }
    let mut o = [[[0.0_f64; 3]; 3]; 3];
    set(&mut o, 0, 0, 0, c[0]);
    set(&mut o, 0, 0, 1, c[1]);
    set(&mut o, 0, 0, 2, c[2]);
    set(&mut o, 0, 1, 1, c[3]);
    set(&mut o, 0, 1, 2, c[4]);
    set(&mut o, 0, 2, 2, c[5]);
    set(&mut o, 1, 1, 1, c[6]);
    set(&mut o, 1, 1, 2, c[7]);
    set(&mut o, 1, 2, 2, c[8]);
    set(&mut o, 2, 2, 2, c[9]);
    o
}

/// Remove the trace of a symmetric rank-3 tensor:
/// `O^tl_{ijk} = O_{ijk} - (1/5)(δ_{ij}T_k + δ_{jk}T_i + δ_{ik}T_j)`, `T_a = Σ_m O_{mma}`.
fn detrace_octupole(o: &[[[f64; 3]; 3]; 3]) -> [[[f64; 3]; 3]; 3] {
    let mut tvec = [0.0_f64; 3];
    for (k, tk) in tvec.iter_mut().enumerate() {
        for m in 0..3 {
            *tk += o[m][m][k];
        }
    }
    let mut tl = [[[0.0_f64; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let mut v = o[i][j][k];
                if i == j {
                    v -= tvec[k] / 5.0;
                }
                if j == k {
                    v -= tvec[i] / 5.0;
                }
                if i == k {
                    v -= tvec[j] / 5.0;
                }
                tl[i][j][k] = v;
            }
        }
    }
    tl
}

/// `(n)!!` for odd `n` (`(-1)!! = 1`).
#[inline]
fn odd_double_factorial(n: i64) -> f64 {
    let mut r = 1.0_f64;
    let mut k = n;
    while k > 0 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// Symmetric-traceless (STF) projection of a fully symmetric rank-`l` Cartesian tensor `s`
/// (flat row-major over `{0,1,2}^l`, `3^l` entries) — the **arbitrary-rank** generalization of
/// [`traceless`] (l=2) and [`detrace_octupole`] (l=3). Standard detracer
/// `T = Σ_{matchings (P,G)} c_{l,|P|} (Π_{(a,b)∈P} δ_{i_a i_b}) Tr^{|P|}(S)`, with
/// `c_{l,k} = (-1)^k (2l−2k−1)!!/(2l−1)!!` and `Tr^k(S)` the k-fold trace evaluated at the
/// singleton indices `G`. Reproduces `traceless`/`detrace_octupole` exactly at l=2,3 (gated).
#[allow(dead_code)] // used by the arbitrary-rank moment path (multipole_order ≥ 4)
fn detrace_symmetric(s: &[f64], l: usize) -> Vec<f64> {
    let n = 3usize.pow(l as u32);
    if l < 2 {
        return s.to_vec(); // rank 0/1 are already traceless
    }
    let ms = matchings(l);
    let dd = odd_double_factorial(2 * l as i64 - 1);
    let mut out = vec![0.0_f64; n];
    let mut idx = vec![0usize; l];
    let mut sidx = vec![0usize; l];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut t = flat;
        for a in (0..l).rev() {
            idx[a] = t % 3;
            t /= 3;
        }
        let mut acc = 0.0;
        for (pairs, singles) in &ms {
            // Kronecker-delta factors: the two output indices of every pair must coincide.
            if pairs.iter().any(|&(a, b)| idx[a] != idx[b]) {
                continue;
            }
            let k = pairs.len();
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let c = sign * odd_double_factorial(2 * l as i64 - 2 * k as i64 - 1) / dd;
            for &g in singles {
                sidx[g] = idx[g];
            }
            // Tr^k(S) at the singleton indices = Σ over the k pair-trace indices.
            let mut tracesum = 0.0;
            for mvec in 0..3usize.pow(k as u32) {
                let mut mm = mvec;
                for &(a, b) in pairs {
                    let mj = mm % 3;
                    mm /= 3;
                    sidx[a] = mj;
                    sidx[b] = mj;
                }
                let mut sf = 0usize;
                for &si in sidx.iter() {
                    sf = sf * 3 + si;
                }
                tracesum += s[sf];
            }
            acc += c * tracesum;
        }
        *slot = acc;
    }
    out
}

/// On-site atomic octupole `Ō_{μκ}` (symmetric `3x3x3`) for a same-atom AO pair,
/// `<μ|(r-R_A)_i(r-R_A)_j(r-R_A)_k|κ>` (operator origin = atom centre `ra`), from the
/// from-scratch octupole AO integral [`crate::integrals::contracted_octupole_pair`].
fn onsite_octupole(basis: &BasisSet, mu: usize, ka: usize, ra: Vec3) -> [[[f64; 3]; 3]; 3] {
    let c = crate::integrals::contracted_octupole_pair(&basis.aos[mu], &basis.aos[ka], ra, ra);
    octu_from_components(&c)
}

/// Geometry-fixed cache of the **raw** on-site octupole AO tensors `Ō_{μκ}` for every same-atom
/// AO pair. The octupole integral (`contracted_octupole_pair`, costly for d-heavy atoms) does
/// **not** change across SCC iterations, yet `atomic_octupole_moments` and the octupole Fock
/// shift recompute it every iteration. Build this **once per geometry** and reuse — the cached
/// value is bit-for-bit `onsite_octupole(...)`, so the energy/Fock/gradient are byte-identical to
/// the recompute path (only the redundant integral work is removed). Per atom, a row-major
/// `nao_a × nao_a` block of raw `[[[f64;3];3];3]` tensors. (v0.2.0 octupole-optimization fix.)
#[derive(Clone, Debug)]
pub struct OnsiteOctupoleCache {
    per_atom: Vec<Vec<[[[f64; 3]; 3]; 3]>>,
    nao: Vec<usize>,
}

impl OnsiteOctupoleCache {
    /// Build the cache for the given geometry (parallel over atoms).
    pub fn build(basis: &BasisSet, nat: usize, atom_pos: &[Vec3]) -> Self {
        let per_atom_aos = atom_ao_lists(basis, nat);
        let nao: Vec<usize> = per_atom_aos.iter().map(|a| a.len()).collect();
        let per_atom = per_atom_aos
            .par_iter()
            .enumerate()
            .map(|(a, aos)| {
                let ra = atom_pos[a];
                let mut block = Vec::with_capacity(aos.len() * aos.len());
                for &mu in aos {
                    for &ka in aos {
                        block.push(onsite_octupole(basis, mu, ka, ra));
                    }
                }
                block
            })
            .collect();
        Self { per_atom, nao }
    }

    /// Raw on-site octupole for atom `a`'s local AO indices `(mu_local, ka_local)`.
    #[inline]
    fn get(&self, a: usize, mu_local: usize, ka_local: usize) -> [[[f64; 3]; 3]; 3] {
        self.per_atom[a][mu_local * self.nao[a] + ka_local]
    }
}

/// Resolve the raw on-site octupole for atom `a`, local pair `(mi, ki)` (global `mu`,`ka`),
/// from the cache when present, else compute it. Byte-identical either way.
#[inline]
fn onsite_octupole_cached(
    cache: Option<&OnsiteOctupoleCache>,
    basis: &BasisSet,
    a: usize,
    mi: usize,
    ki: usize,
    mu: usize,
    ka: usize,
    ra: Vec3,
) -> [[[f64; 3]; 3]; 3] {
    match cache {
        Some(c) => c.get(a, mi, ki),
        None => onsite_octupole(basis, mu, ka, ra),
    }
}

/// Atomic mDFTB octupole moments `ΔO_A` (**traceless** rank-3) from the on-site
/// approximation, mirroring [`atomic_moments`]:
/// `ΔO_A = ½ Σ_{μ∈A,κ∈A,ν} Ō_{μκ}(S_{κν}P_{νμ} + P_{μν}S_{νκ})`, then trace-removed with
/// `O^tl_{ijk} = O_{ijk} - (1/5)(δ_{ij}T_k + δ_{jk}T_i + δ_{ik}T_j)`, `T_a = Σ_m O_{mma}`.
/// `atom_positions[a]` is the centre of atom `a`. Non-periodic.
pub fn atomic_octupole_moments(
    basis: &BasisSet,
    nat: usize,
    atom_positions: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    cache: Option<&OnsiteOctupoleCache>,
) -> Vec<[[[f64; 3]; 3]; 3]> {
    let s = &integrals.overlap;
    let n = basis.len();
    let per_atom = atom_ao_lists(basis, nat);
    // Each atom's traceless octupole is independent (the inner Σ_ν overlap row is the O(N)
    // factor → O(N²) overall, plus the rank-3 octupole AO integral per same-atom pair). Compute
    // per atom in parallel; the per-atom sum order is preserved → bit-identical. Mirrors
    // [`atomic_moments`]; the rank-3 integrals over d-heavy atoms (Pd, Fe) make this costly, so the
    // geometry-fixed `Ō_{μκ}` come from `cache` (built once per geometry) when present.
    per_atom
        .par_iter()
        .enumerate()
        .map(|(a, aos)| {
            let ra = atom_positions[a];
            let mut o = [[[0.0_f64; 3]; 3]; 3];
            for (mi, &mu) in aos.iter().enumerate() {
                for (ki, &ka) in aos.iter().enumerate() {
                    let obar = onsite_octupole_cached(cache, basis, a, mi, ki, mu, ka, ra);
                    let mut w = 0.0;
                    for nu in 0..n {
                        w += 0.5
                            * (s[(ka, nu)] * density[(nu, mu)] + density[(mu, nu)] * s[(nu, ka)]);
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                o[i][j][k] += obar[i][j][k] * w;
                            }
                        }
                    }
                }
            }
            detrace_octupole(&o)
        })
        .collect()
}

/// Expand the `(L+1)(L+2)/2` **unique** symmetric Cartesian components (in
/// [`crate::integrals::cartesian_rank_components`] order — `lx` desc then `ly` desc) of a fully
/// symmetric rank-`l` tensor into the full `3^l` row-major tensor over `{0,1,2}^l`. Each flat
/// index's `(lx,ly,lz)` signature (counts of axes 0/1/2) selects the unique component.
fn expand_symmetric_cartesian(unique: &[f64], l: usize) -> Vec<f64> {
    let comps = crate::integrals::cartesian_rank_components(l);
    debug_assert_eq!(comps.len(), unique.len());
    let n = 3usize.pow(l as u32);
    let mut out = vec![0.0_f64; n];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut t = flat;
        let (mut lx, mut ly, mut lz) = (0usize, 0usize, 0usize);
        for _ in 0..l {
            match t % 3 {
                0 => lx += 1,
                1 => ly += 1,
                _ => lz += 1,
            }
            t /= 3;
        }
        // Find this signature in the canonical list (l is small; linear scan is fine).
        let pos = comps
            .iter()
            .position(|&c| c == (lx, ly, lz))
            .expect("signature in list");
        *slot = unique[pos];
    }
    out
}

/// Detraced (STF) rank-`l` on-site atomic moments from the density `P` — the **arbitrary-rank**
/// generalization of [`atomic_moments`] (the traceless quadrupole, `l=2`) and
/// [`atomic_octupole_moments`] (`l=3`):
/// `M^(l)_A = detrace( Σ_{μ,κ∈A,ν} ⟨μ|(r−R_A)^⊗l|κ⟩ · ½(S_{κν}P_{νμ}+P_{μν}S_{νκ}) )`.
/// Each atom's moment is returned as a full `3^l` row-major Cartesian tensor (already trace-free
/// via [`detrace_symmetric`]). The raw on-site rank-`l` AO integral is
/// [`crate::integrals::contracted_moment_rank`] (which reproduces the hard-coded
/// octupole integral byte-for-byte at `l=3`), so for `l=2,3` this matches the legacy paths
/// (gated); for `l≥4` it is the generic path used by `multipole_order ≥ 4`. Non-periodic.
/// Geometry-fixed cache of the **raw** (undetraced, full `3^l` row-major) on-site rank-`l` AO moment
/// tensors `M̄^(l)_{μκ} = ⟨μ|(r−R_A)^⊗l|κ⟩` for every same-atom AO pair and every rank `l=1..=max_rank`.
/// These integrals ([`crate::integrals::contracted_moment_rank`], the costly McMurchie–Davidson 1D
/// moments) do **not** change across SCC iterations, yet the generic arbitrary-rank moment extraction
/// and Fock shift recompute them every iteration — the dominant per-iteration cost. Building this
/// **once per geometry** collapses the SCC loop's integral cost to a single up-front pass. The cached
/// value is bit-for-bit `expand_symmetric_cartesian(contracted_moment_rank(...))`, so the moments /
/// energy / Fock / gradient are byte-identical to the recompute path. Generalizes
/// [`OnsiteOctupoleCache`] (its `l=3` special case) to arbitrary rank. (v0.2.0 large-scale opt.)
#[derive(Clone, Debug)]
pub struct OnsiteMomentCache {
    /// `per_atom[a][l-1]` = row-major `nao_a × nao_a` block of raw full `3^l` tensors.
    per_atom: Vec<Vec<Vec<Vec<f64>>>>,
    nao: Vec<usize>,
    max_rank: usize,
}

impl OnsiteMomentCache {
    /// Build the cache for the given geometry (parallel over atoms), ranks `1..=max_rank`, over the
    /// **primary** AOs. Byte-identical to `build_with_aos(.., None)`.
    pub fn build(basis: &BasisSet, nat: usize, atom_pos: &[Vec3], max_rank: usize) -> Self {
        Self::build_with_aos(basis, nat, atom_pos, max_rank, None)
    }

    /// Build the cache evaluating the on-site moment integrals over an optional **secondary** AO set
    /// (the Stage-5 richer moments): when `sec_aos` is `Some`, `sec_aos[mu]` (same centre + Cartesian
    /// component as the primary AO, node-correct radial part) replaces `basis.aos[mu]`, so the
    /// arbitrary-rank generic multipole path consumes the secondary basis at **every** rank — the
    /// natural generalisation of the legacy rank-1/2 [`secondary_moment_integrals`]. `None` ⇒ the
    /// primary AOs (byte-identical to the previous behaviour, so the secondary-off path is unchanged).
    pub fn build_with_aos(
        basis: &BasisSet,
        nat: usize,
        atom_pos: &[Vec3],
        max_rank: usize,
        sec_aos: Option<&[crate::basis::AOBasisFunction]>,
    ) -> Self {
        let per_atom_aos = atom_ao_lists(basis, nat);
        let nao: Vec<usize> = per_atom_aos.iter().map(|a| a.len()).collect();
        let per_atom = per_atom_aos
            .par_iter()
            .enumerate()
            .map(|(a, aos)| {
                let ra = atom_pos[a];
                (1..=max_rank)
                    .map(|l| {
                        let mut block = Vec::with_capacity(aos.len() * aos.len());
                        for &mu in aos {
                            for &ka in aos {
                                let aom = match sec_aos {
                                    Some(s) => &s[mu],
                                    None => &basis.aos[mu],
                                };
                                let aok = match sec_aos {
                                    Some(s) => &s[ka],
                                    None => &basis.aos[ka],
                                };
                                let unique =
                                    crate::integrals::contracted_moment_rank(aom, aok, ra, ra, l);
                                block.push(expand_symmetric_cartesian(&unique, l));
                            }
                        }
                        block
                    })
                    .collect()
            })
            .collect();
        Self {
            per_atom,
            nao,
            max_rank,
        }
    }

    /// Raw full `3^l` on-site moment tensor for atom `a`'s local AO pair `(mi, ki)`.
    #[inline]
    fn get(&self, a: usize, l: usize, mi: usize, ki: usize) -> &[f64] {
        debug_assert!(l >= 1 && l <= self.max_rank);
        &self.per_atom[a][l - 1][mi * self.nao[a] + ki]
    }
}

/// Detraced (STF) rank-`l` on-site atomic moments from the density `P` — the **arbitrary-rank**
/// generalization of [`atomic_moments`] (the traceless quadrupole, `l=2`) and
/// [`atomic_octupole_moments`] (`l=3`):
/// `M^(l)_A = detrace( Σ_{μ,κ∈A,ν} ⟨μ|(r−R_A)^⊗l|κ⟩ · ½(S_{κν}P_{νμ}+P_{μν}S_{νκ}) )`.
/// Each atom's moment is returned as a full `3^l` row-major Cartesian tensor (already trace-free
/// via [`detrace_symmetric`]). The raw on-site rank-`l` AO integral is
/// [`crate::integrals::contracted_moment_rank`] (which reproduces the hard-coded octupole integral
/// byte-for-byte at `l=3`) — or, geometry-fixed, the [`OnsiteMomentCache`] when present
/// (byte-identical). For `l=2,3` this matches the legacy paths (gated); for `l≥4` it is the generic
/// path used by `multipole_order ≥ 4`. Non-periodic.
#[allow(dead_code)] // wired in by the arbitrary-rank multipole path (multipole_order ≥ 4)
pub fn atomic_moment_rank_l(
    basis: &BasisSet,
    nat: usize,
    atom_positions: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    l: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Vec<Vec<f64>> {
    let s = &integrals.overlap;
    let n = basis.len();
    let nl = 3usize.pow(l as u32);
    let per_atom = atom_ao_lists(basis, nat);
    // Mirrors atomic_octupole_moments: each atom independent, per-atom sum order preserved
    // (bit-identical to serial). The geometry-fixed raw on-site integral comes from `cache` when
    // present (byte-identical), else is recomputed.
    per_atom
        .par_iter()
        .enumerate()
        .map(|(a, aos)| {
            let ra = atom_positions[a];
            let mut acc = vec![0.0_f64; nl];
            for (mi, &mu) in aos.iter().enumerate() {
                for (ki, &ka) in aos.iter().enumerate() {
                    let full_owned;
                    let full: &[f64] = match cache {
                        Some(c) => c.get(a, l, mi, ki),
                        None => {
                            let unique = crate::integrals::contracted_moment_rank(
                                &basis.aos[mu],
                                &basis.aos[ka],
                                ra,
                                ra,
                                l,
                            );
                            full_owned = expand_symmetric_cartesian(&unique, l);
                            &full_owned
                        }
                    };
                    let mut w = 0.0;
                    for nu in 0..n {
                        w += 0.5
                            * (s[(ka, nu)] * density[(nu, mu)] + density[(mu, nu)] * s[(nu, ka)]);
                    }
                    for (acck, fk) in acc.iter_mut().zip(full.iter()) {
                        *acck += fk * w;
                    }
                }
            }
            detrace_symmetric(&acc, l)
        })
        .collect()
}

/// Radial derivatives `G_p = d^p/d(r^2)^p γ` of the GFN1 Klopman-Ohno kernel
/// `γ(r) = 1/sqrt(r^2 + c)` (with `c = 1/η^2`, `η` the effective atomic hardness),
/// for `p = 0..=nmax`. `G_p = (-1)^p (2p-1)!!/2^p (r^2+c)^{-(2p+1)/2}` (a simple downward
/// recurrence, so this extends to **arbitrary order** — the rank cap is removed; an
/// interaction `f^(mn)` needs `nmax = m+n` (energy) or `m+n+1` (gradient)).
pub(crate) fn radial_derivs(r2: f64, c: f64, nmax: usize) -> Vec<f64> {
    let s = r2 + c;
    let mut g = vec![0.0_f64; nmax + 1];
    g[0] = 1.0 / s.sqrt();
    for p in 1..=nmax {
        // G_p = G_{p-1} * (-(2p-1)/2) / s
        g[p] = g[p - 1] * (-((2 * p - 1) as f64) / 2.0) / s;
    }
    g
}

/// All partial matchings of `k` positions `{0..k}` into unordered disjoint pairs plus
/// singletons. Each entry is `(pairs, singletons)`. Used to assemble the Cartesian
/// gradient tensor of a radial function. `k <= 5` in practice (tiny).
fn matchings(k: usize) -> Vec<(Vec<(usize, usize)>, Vec<usize>)> {
    fn rec(
        rest: &[usize],
        pairs: &mut Vec<(usize, usize)>,
        out: &mut Vec<(Vec<(usize, usize)>, Vec<usize>)>,
    ) {
        if rest.is_empty() {
            out.push((pairs.clone(), Vec::new()));
            // singletons are filled in by the caller chain; handled below instead.
            return;
        }
        let first = rest[0];
        // Option 1: `first` is a singleton.
        {
            let sub: Vec<usize> = rest[1..].to_vec();
            let start = out.len();
            rec(&sub, pairs, out);
            for entry in out.iter_mut().skip(start) {
                entry.1.push(first);
            }
        }
        // Option 2: `first` pairs with some later position `rest[j]`.
        for j in 1..rest.len() {
            let partner = rest[j];
            let sub: Vec<usize> = rest[1..]
                .iter()
                .copied()
                .filter(|&p| p != partner)
                .collect();
            pairs.push((first, partner));
            rec(&sub, pairs, out);
            pairs.pop();
        }
    }
    let positions: Vec<usize> = (0..k).collect();
    let mut out = Vec::new();
    let mut pairs = Vec::new();
    rec(&positions, &mut pairs, &mut out);
    out
}

/// Fully-symmetric rank-`k` Cartesian gradient tensor `T^(k)_{i1..ik} = ∇^k γ(r)` of the
/// radial kernel, stored row-major over `{0,1,2}^k` (`3^k` entries). Built from the
/// radial derivatives `g = [G_0..G_5]` and the displacement `x = R_A - R_B`:
/// `T^(k) = Σ_matchings 2^{k-m} G_{k-m} (Π_pairs δ) (Π_singletons x)`, `m = #pairs`.
/// `g` must hold `G_0..G_k` (length ≥ `k+1`). Arbitrary rank `k` (no cap; the `idx` buffer is a
/// reused `Vec`, so high-rank multipoles are supported — `3^k` grows fast, the practical limit).
pub(crate) fn grad_tensor(x: Vec3, g: &[f64], k: usize) -> Vec<f64> {
    let xa = [x.x, x.y, x.z];
    if k == 0 {
        return vec![g[0]];
    }
    let ms = matchings(k);
    let n = 3usize.pow(k as u32);
    let mut out = vec![0.0_f64; n];
    let mut idx = vec![0usize; k]; // reused index tuple (no rank cap)
    for (flat, slot) in out.iter_mut().enumerate() {
        // Decode the index tuple (i_0 .. i_{k-1}).
        let mut t = flat;
        for a in (0..k).rev() {
            idx[a] = t % 3;
            t /= 3;
        }
        let mut acc = 0.0;
        for (pairs, singles) in &ms {
            let m = pairs.len();
            let mut term = 1.0;
            for &(a, b) in pairs {
                if idx[a] != idx[b] {
                    term = 0.0;
                    break;
                }
            }
            if term == 0.0 {
                continue;
            }
            for &s in singles {
                term *= xa[idx[s]];
            }
            let coeff = (2.0_f64).powi((k - m) as i32) * g[k - m];
            acc += coeff * term;
        }
        *slot = acc;
    }
    out
}

/// The mDFTB interaction tensor `f^(mn)_AB = (1/m!n!) ∂^m_{R_A} ∂^n_{R_B} γ`
/// `= ((-1)^n / (m! n!)) ∇^{m+n} γ` (eq 18), as a flat rank-`(m+n)` tensor (`3^{m+n}`).
/// `x = R_A - R_B`, `c = 1/η^2`. The sign/factorial prefactor distinguishes the bra/ket
/// orders; `f^(00)=γ`, `f^(10)=∇γ`, `f^(01)=-∇γ`, `f^(11)=-∇^2γ`, etc.
pub(crate) fn f_mn(x: Vec3, c: f64, m: usize, n: usize) -> Vec<f64> {
    let g = radial_derivs(x.norm2(), c, m + n);
    let t = grad_tensor(x, &g, m + n);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

/// Spatial gradient `∂_x f^(mn)` (eq 18 differentiated once more), a flat rank-`(m+n+1)`
/// tensor whose **first** index is the gradient direction and whose remaining `m+n` indices
/// are the (fully symmetric) `f^(mn)` indices. Same prefactor as [`f_mn`].
pub(crate) fn f_mn_grad(x: Vec3, c: f64, m: usize, n: usize) -> Vec<f64> {
    let g = radial_derivs(x.norm2(), c, m + n + 1);
    let t = grad_tensor(x, &g, m + n + 1);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

// --- CAMM-on-mDFTB2 (v0.4.2): erf charge-cloud off-site AES kernel ---
//
// The CAMM/GFN2-AES off-site interaction (model `CammOnMdftb2`) replaces the Ohno multipole
// tensors above by the derivatives of the **erf charge-cloud** kernel `γ_cloud(R)=erf(R/σ_AB)/R`
// (`σ_AB = κ·exchange_sigma_pair(η_A,η_B)`, the *range factor* `κ` is the primary calibration
// lever). It is less screened than Ohno at all R (so its multipole tensors are the stronger,
// CAMM-like ones), finite at `R→0`, and → bare `1/R` derivatives at long range — that long-range
// tail is independent of `σ`, so `κ` re-balances the contact region without touching it.

/// Radial derivatives `G_p^cloud = d^p/d(R²)^p [erf(R/σ_ab)/R]` of the erf charge-cloud kernel,
/// `p = 0..=nmax`. Closed form `G_p = (2α/√π)(−α²)^p F_p(α²R²)`, `α = 1/σ_ab`, `F_p` the Boys
/// function (= the `erf` half of `pbc::ewald_multipole::ewald_real_radial_derivs`). Finite at
/// `R→0` (`F_p(0)=1/(2p+1)`). Fed into the same [`grad_tensor`] engine as [`radial_derivs`].
pub(crate) fn erf_cloud_radial_derivs(r2: f64, sigma_ab: f64, nmax: usize) -> Vec<f64> {
    let alpha = 1.0 / sigma_ab;
    let a2 = alpha * alpha;
    let f = crate::nmr::boys(nmax, a2 * r2);
    let pref = 2.0 * alpha / std::f64::consts::PI.sqrt();
    let mut g = vec![0.0_f64; nmax + 1];
    let mut neg_a2_pow = 1.0_f64; // (−α²)^p
    for (p, gp) in g.iter_mut().enumerate() {
        *gp = pref * neg_a2_pow * f[p];
        neg_a2_pow *= -a2;
    }
    g
}

/// erf-cloud interaction tensor `f^(mn) = ((−1)ⁿ/m!n!) ∇^{m+n}[erf(R/σ_ab)/R]` (CAMM-on-mDFTB2
/// off-site AES kernel), a flat rank-`(m+n)` tensor. `x = R_A − R_B`. Same sign/factorial
/// prefactor convention as [`f_mn`]; only the radial ladder differs.
pub(crate) fn f_mn_cloud(x: Vec3, sigma_ab: f64, m: usize, n: usize) -> Vec<f64> {
    let g = erf_cloud_radial_derivs(x.norm2(), sigma_ab, m + n);
    let t = grad_tensor(x, &g, m + n);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

/// Spatial gradient `∂_x f^(mn)` of the erf-cloud tensor (rank `m+n+1`). Mirrors [`f_mn_grad`].
pub(crate) fn f_mn_grad_cloud(x: Vec3, sigma_ab: f64, m: usize, n: usize) -> Vec<f64> {
    let g = erf_cloud_radial_derivs(x.norm2(), sigma_ab, m + n + 1);
    let t = grad_tensor(x, &g, m + n + 1);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

// --- unique-symmetric-component machinery (memory/speed for high rank) ---
//
// The interaction tensors `f^(mn)` are derivatives of a scalar, hence **fully symmetric** rank-`r`
// (`r=m+n`) tensors with only `(r+1)(r+2)/2` distinct components, not `3^r`. For `r=8` that is 45
// vs 6561 (`f^(4,4)`); for `r=9`, 55 vs 19683 (its gradient). The generic arbitrary-rank path
// therefore builds and contracts these tensors in **unique-component space** — exact (the multiset
// of axis multiplicities determines the value), but polynomial in `r` rather than exponential. The
// (small, `3^l≤81`) moment tensors stay full; only the `f` tensors use this path. The legacy rank
// ≤3 routines keep the full-tensor helpers (byte-compatible) and are untouched.

/// Position of the multiplicity `(lx,ly,·)` (with `lx+ly+lz=total`) in the canonical
/// [`crate::integrals::cartesian_rank_components`] order (`lx` desc then `ly` desc), computed in
/// O(total) without building the list.
#[inline]
fn cart_index(lx: usize, ly: usize, total: usize) -> usize {
    let mut idx = 0usize;
    for lxp in (lx + 1)..=total {
        idx += total - lxp + 1; // # components with a larger lx
    }
    idx + ((total - lx) - ly) // within this lx, ly runs (total-lx) down to 0
}

/// Multinomial coefficient `total! / (lx! ly! lz!)` = number of full row-major indices sharing the
/// multiplicity `(lx,ly,lz)`. `total = lx+ly+lz` is small.
#[inline]
fn multinomial(lx: usize, ly: usize, lz: usize) -> f64 {
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    fact(lx + ly + lz) / (fact(lx) * fact(ly) * fact(lz))
}

/// Unique symmetric components of the fully-symmetric rank-`k` gradient tensor `∇^k γ` (the
/// `(k+1)(k+2)/2` distinct values of [`grad_tensor`], in `cartesian_rank_components(k)` order).
/// Evaluates the same matchings formula as [`grad_tensor`] at one representative index per
/// multiplicity, so it never materializes the `3^k` tensor.
pub(crate) fn grad_tensor_unique(x: Vec3, g: &[f64], k: usize) -> Vec<f64> {
    let xa = [x.x, x.y, x.z];
    if k == 0 {
        return vec![g[0]];
    }
    let ms = matchings(k);
    let comps = crate::integrals::cartesian_rank_components(k);
    let mut idx = vec![0usize; k];
    comps
        .iter()
        .map(|&(lx, ly, lz)| {
            // Representative index tuple for this multiplicity: 0…0 1…1 2…2.
            let mut p = 0;
            for _ in 0..lx {
                idx[p] = 0;
                p += 1;
            }
            for _ in 0..ly {
                idx[p] = 1;
                p += 1;
            }
            for _ in 0..lz {
                idx[p] = 2;
                p += 1;
            }
            let mut acc = 0.0;
            for (pairs, singles) in &ms {
                let m = pairs.len();
                let mut term = 1.0;
                for &(a, b) in pairs {
                    if idx[a] != idx[b] {
                        term = 0.0;
                        break;
                    }
                }
                if term == 0.0 {
                    continue;
                }
                for &s in singles {
                    term *= xa[idx[s]];
                }
                acc += (2.0_f64).powi((k - m) as i32) * g[k - m] * term;
            }
            acc
        })
        .collect()
}

/// Unique-component form of [`f_mn`] (rank `m+n`, `(m+n+1)(m+n+2)/2` values).
pub(crate) fn f_mn_unique(x: Vec3, c: f64, m: usize, n: usize) -> Vec<f64> {
    let g = radial_derivs(x.norm2(), c, m + n);
    let t = grad_tensor_unique(x, &g, m + n);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

/// Unique-component form of [`f_mn_grad`] (rank `m+n+1`, fully symmetric — the gradient direction
/// is just one more symmetric index of `∇^{m+n+1}γ`).
fn f_mn_grad_unique(x: Vec3, c: f64, m: usize, n: usize) -> Vec<f64> {
    let g = radial_derivs(x.norm2(), c, m + n + 1);
    let t = grad_tensor_unique(x, &g, m + n + 1);
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    let pref = if n % 2 == 0 { 1.0 } else { -1.0 } / (fact(m) * fact(n));
    t.iter().map(|v| v * pref).collect()
}

/// Contract the **last** `lb` indices of a fully-symmetric rank-`(la+lb)` tensor given by its
/// unique components `f_u` with the (full `3^lb`) symmetric moment `mb`, returning the full `3^la`
/// rank-`la` result — the unique-space equivalent of [`contract_last`].
/// `out_u[α] = Σ_γ multinomial(γ)·f_u[α+γ]·mb_u[γ]`, then expanded to full.
pub(crate) fn contract_last_unique(f_u: &[f64], la: usize, lb: usize, mb: &[f64]) -> Vec<f64> {
    let r = la + lb;
    let comps_a = crate::integrals::cartesian_rank_components(la);
    let comps_b = crate::integrals::cartesian_rank_components(lb);
    let mut o_u = vec![0.0_f64; comps_a.len()];
    for (ia, &(ax, ay, az)) in comps_a.iter().enumerate() {
        let _ = az;
        let mut acc = 0.0;
        for &(gx, gy, gz) in &comps_b {
            let mult = multinomial(gx, gy, gz);
            let mb_g = mb[rep_flat_index(gx, gy, gz)];
            acc += mult * f_u[cart_index(ax + gx, ay + gy, r)] * mb_g;
        }
        o_u[ia] = acc;
    }
    expand_symmetric_cartesian(&o_u, la)
}

/// Contract a fully-symmetric rank-`(1+m+n)` gradient tensor given by its unique components `df_u`
/// with the (full) symmetric moments `ma` (rank `m`) and `mb` (rank `n`), leaving the rank-1
/// gradient vector — the unique-space equivalent of [`kernel_grad`].
/// `g[d] = Σ_{α,γ} multinomial(α)·multinomial(γ)·df_u[e_d+α+γ]·ma_u[α]·mb_u[γ]`.
/// `pub(crate)` so the periodic multipole gradient (`pbc::ewald_multipole`) reuses the exact
/// molecular contraction convention with its Ewald-split gradient tensors.
pub(crate) fn kernel_grad_unique(df_u: &[f64], m: usize, n: usize, ma: &[f64], mb: &[f64]) -> Vec3 {
    let r = 1 + m + n;
    let comps_m = crate::integrals::cartesian_rank_components(m);
    let comps_n = crate::integrals::cartesian_rank_components(n);
    let mut o = [0.0_f64; 3];
    for (d, od) in o.iter_mut().enumerate() {
        let (ex, ey) = match d {
            0 => (1usize, 0usize),
            1 => (0, 1),
            _ => (0, 0),
        };
        let mut acc = 0.0;
        for &(ax, ay, az) in &comps_m {
            let _ = az;
            let mult_a = multinomial(ax, ay, m - ax - ay);
            let ma_a = ma[rep_flat_index(ax, ay, m - ax - ay)];
            for &(gx, gy, gz) in &comps_n {
                let _ = gz;
                let mult_g = multinomial(gx, gy, n - gx - gy);
                let mb_g = mb[rep_flat_index(gx, gy, n - gx - gy)];
                acc +=
                    mult_a * mult_g * df_u[cart_index(ex + ax + gx, ey + ay + gy, r)] * ma_a * mb_g;
            }
        }
        *od = acc;
    }
    Vec3::new(o[0], o[1], o[2])
}

// --- tensor contractions (flat row-major rank-k over {0,1,2}^k) ---

#[inline]
fn dot1(f: &[f64], d: Vec3) -> f64 {
    f[0] * d.x + f[1] * d.y + f[2] * d.z
}

#[inline]
fn dot2_full(f: &[f64], q: &[[f64; 3]; 3]) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += f[i * 3 + j] * q[i][j];
        }
    }
    s
}

/// rank-2 `f` contracted on its second index with a vector -> vector.
#[inline]
fn r2_vec(f: &[f64], d: Vec3) -> Vec3 {
    let da = [d.x, d.y, d.z];
    let mut o = [0.0_f64; 3];
    for (i, oi) in o.iter_mut().enumerate() {
        for (j, &dj) in da.iter().enumerate() {
            *oi += f[i * 3 + j] * dj;
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-2 `f` scaled by a scalar -> 3x3.
#[inline]
fn r2_scaled(f: &[f64], s: f64) -> [[f64; 3]; 3] {
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = f[i * 3 + j] * s;
        }
    }
    o
}

/// rank-3 `f` contracted on its last two indices with a 3x3 -> vector.
#[inline]
fn r3_quad(f: &[f64], q: &[[f64; 3]; 3]) -> Vec3 {
    let mut o = [0.0_f64; 3];
    for (i, oi) in o.iter_mut().enumerate() {
        for j in 0..3 {
            for k in 0..3 {
                *oi += f[(i * 3 + j) * 3 + k] * q[j][k];
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-3 `f` contracted on its last index with a vector -> 3x3.
#[inline]
fn r3_vec(f: &[f64], d: Vec3) -> [[f64; 3]; 3] {
    let da = [d.x, d.y, d.z];
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for (k, &dk) in da.iter().enumerate() {
                o[i][j] += f[(i * 3 + j) * 3 + k] * dk;
            }
        }
    }
    o
}

/// rank-3 `f` (gradient tensor `[g,i,j]`) contracted on its last two indices with two
/// vectors `a`, `b` -> vector over the first (gradient) index: `o[g] = Σ_ij f[g,i,j] a_i b_j`.
#[inline]
fn r3_two_vec(f: &[f64], a: Vec3, b: Vec3) -> Vec3 {
    let aa = [a.x, a.y, a.z];
    let bb = [b.x, b.y, b.z];
    let mut o = [0.0_f64; 3];
    for (g, og) in o.iter_mut().enumerate() {
        for (i, &ai) in aa.iter().enumerate() {
            for (j, &bj) in bb.iter().enumerate() {
                *og += f[(g * 3 + i) * 3 + j] * ai * bj;
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-4 `f` (gradient tensor `[g,i,j,k]`) contracted with a vector `a` (index 1) and a
/// 3x3 `q` (indices 2,3) -> vector over the gradient index: `o[g] = Σ_i a_i Σ_jk f[g,i,j,k] q_jk`.
#[inline]
fn r4_vec_quad(f: &[f64], a: Vec3, q: &[[f64; 3]; 3]) -> Vec3 {
    let aa = [a.x, a.y, a.z];
    let mut o = [0.0_f64; 3];
    for (g, og) in o.iter_mut().enumerate() {
        for (i, &ai) in aa.iter().enumerate() {
            for j in 0..3 {
                for k in 0..3 {
                    *og += f[((g * 3 + i) * 3 + j) * 3 + k] * ai * q[j][k];
                }
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-4 `f` (gradient tensor `[g,i,j,k]`) contracted with a 3x3 `q` (indices 1,2) and a
/// vector `a` (index 3) -> vector over the gradient index: `o[g] = Σ_ij q_ij Σ_k f[g,i,j,k] a_k`.
#[inline]
fn r4_quad_vec(f: &[f64], q: &[[f64; 3]; 3], a: Vec3) -> Vec3 {
    let aa = [a.x, a.y, a.z];
    let mut o = [0.0_f64; 3];
    for (g, og) in o.iter_mut().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                for (k, &ak) in aa.iter().enumerate() {
                    *og += f[((g * 3 + i) * 3 + j) * 3 + k] * q[i][j] * ak;
                }
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-5 `f` (gradient tensor `[g,i,j,k,l]`) contracted with two 3x3 `qa` (indices 1,2)
/// and `qb` (indices 3,4) -> vector: `o[g] = Σ_ijkl qa_ij qb_kl f[g,i,j,k,l]`.
#[inline]
fn r5_quad_quad(f: &[f64], qa: &[[f64; 3]; 3], qb: &[[f64; 3]; 3]) -> Vec3 {
    let mut o = [0.0_f64; 3];
    for (g, og) in o.iter_mut().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        *og += f[(((g * 3 + i) * 3 + j) * 3 + k) * 3 + l] * qa[i][j] * qb[k][l];
                    }
                }
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// rank-4 `f` contracted on its last two indices with a 3x3 -> 3x3.
#[inline]
fn r4_quad(f: &[f64], q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    o[i][j] += f[((i * 3 + j) * 3 + k) * 3 + l] * q[k][l];
                }
            }
        }
    }
    o
}

// --- general flat-tensor contractions (for octupole and beyond) ---
// These generalize the hand-typed helpers above so higher multipoles need no new
// per-rank code; they reproduce the typed helpers exactly (unit-tested).

/// Flatten a symmetric `3x3` into row-major `[f64; 9]`.
#[inline]
#[allow(dead_code)]
fn quad_flat(q: &[[f64; 3]; 3]) -> [f64; 9] {
    let mut f = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            f[i * 3 + j] = q[i][j];
        }
    }
    f
}

/// Flatten a symmetric `3x3x3` into row-major `[f64; 27]`.
#[inline]
#[allow(dead_code)]
fn octu_flat(o: &[[[f64; 3]; 3]; 3]) -> [f64; 27] {
    let mut f = [0.0_f64; 27];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                f[(i * 3 + j) * 3 + k] = o[i][j][k];
            }
        }
    }
    f
}

/// Reshape a flat rank-3 (`[f64; 27]`) back into a `3x3x3`.
#[inline]
#[allow(dead_code)]
fn octu_unflat(f: &[f64]) -> [[[f64; 3]; 3]; 3] {
    let mut o = [[[0.0_f64; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j][k] = f[(i * 3 + j) * 3 + k];
            }
        }
    }
    o
}

/// Contract the **last** `n` indices of a flat row-major rank-`(m+n)` tensor `f` with a
/// flat rank-`n` moment `mb` (length `3^n`), leaving a flat rank-`m` tensor (length
/// `3^m`): `out[i] = Σ_j f[i·3^n + j] · mb[j]`. This is the "field on moment-type m from
/// moment-type n" contraction `f^(mn)·m_B^(n)`.
#[allow(dead_code)]
fn contract_last(f: &[f64], m: usize, n: usize, mb: &[f64]) -> Vec<f64> {
    let dm = 3usize.pow(m as u32);
    let dn = 3usize.pow(n as u32);
    let mut out = vec![0.0_f64; dm];
    for (i, oi) in out.iter_mut().enumerate() {
        for (j, &mbj) in mb.iter().enumerate().take(dn) {
            *oi += f[i * dn + j] * mbj;
        }
    }
    out
}

/// Result of the mDFTB2 multipole correction at a given density: the correction energy
/// and its contribution `F = ∂E/∂P` to the Fock matrix (self-consistent shift).
#[derive(Clone, Debug)]
pub struct MultipoleEnergyFock {
    pub energy: f64,
    pub fock: Matrix,
}

/// Self-consistent mDFTB2 multipole correction energy and Fock shift at the density `P`.
/// Adds only the terms involving an atomic dipole `d` or quadrupole `Q` on top of GFN1's
/// monopole electrostatics. `atom_hardness[A]` is the atomic Klopman-Ohno hardness `η_A`
/// (so the kernel is `γ_AB(r) = 1/sqrt(r^2 + 1/η_AB^2)`, `η_AB = harmonic_average(η_A,η_B)`);
/// `atomic_charges[A]` is the GFN1 atomic Mulliken charge `Δq_A`. Eqs 16/21/33/34 of
/// Vuong et al. The Fock is the exact `∂E/∂P` so the SCC stays variational.
pub fn multipole_energy_fock(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
) -> MultipoleEnergyFock {
    let moments = atomic_moments(basis, nat, integrals, density);
    multipole_fock_from_moments(
        basis,
        nat,
        atom_hardness,
        atom_pos,
        integrals,
        &moments,
        atomic_charges,
    )
}

/// mDFTB2 correction energy + Fock shift from **given** atomic moments (rather than from a
/// density). Used by the self-consistent SCC, where the atomic dipole/quadrupole moments are
/// mixed (Broyden) alongside the monopole charges (the tblite-style multipole SCF); building
/// the Fock from the *mixed* moments is what makes the multipole self-consistency converge on
/// hard, polarizable systems. `atomic_charges[A]` is the mDFTB monopole `Δq_A`.
pub fn multipole_fock_from_moments(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    moments: &AtomicMoments,
    atomic_charges: &[f64],
) -> MultipoleEnergyFock {
    let n = basis.len();
    let (s, vd, vq) = potentials_from_moments(
        nat,
        atom_hardness,
        atom_pos,
        &moments.dipole,
        &moments.quad,
        atomic_charges,
    );
    let energy = multipole_energy_terms(nat, moments, &s, &vd, &vq, atomic_charges);
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd, &vq);
    let fock = assemble_shift(n, &per_atom, &atom_of, &s, &dmat, &mmat, &integrals.overlap);
    MultipoleEnergyFock { energy, fock }
}

/// mDFTB2 correction **energy only** from given atomic moments (no Fock build).
pub fn multipole_energy_from_moments(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    moments: &AtomicMoments,
    atomic_charges: &[f64],
) -> f64 {
    let (s, vd, vq) = potentials_from_moments(
        nat,
        atom_hardness,
        atom_pos,
        &moments.dipole,
        &moments.quad,
        atomic_charges,
    );
    multipole_energy_terms(nat, moments, &s, &vd, &vq, atomic_charges)
}

/// `E = ½ Σ_A [ q_A s_A + d_A·vd_A + Q_A:vQ_A ]` (eqs 16/21).
fn multipole_energy_terms(
    nat: usize,
    moments: &AtomicMoments,
    s: &[f64],
    vd: &[Vec3],
    vq: &[[[f64; 3]; 3]],
    q: &[f64],
) -> f64 {
    let mut energy = 0.0;
    for a in 0..nat {
        energy += 0.5
            * (q[a] * s[a]
                + moments.dipole[a].dot(vd[a])
                + dot2_full_mat(&moments.quad[a], &vq[a]));
    }
    energy
}

/// Atomic moments and the mDFTB2 potentials (the full "fields" `K m`) from a density.
/// Shared by the analytic gradient (which works from the converged density).
#[allow(clippy::type_complexity)]
fn moments_and_potentials(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
) -> (AtomicMoments, Vec<f64>, Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
    let moments = atomic_moments(basis, nat, integrals, density);
    let (s, vd, vq) = potentials_from_moments(
        nat,
        atom_hardness,
        atom_pos,
        &moments.dipole,
        &moments.quad,
        atomic_charges,
    );
    (moments, s, vd, vq)
}

/// The mDFTB2 potentials `s_A` (felt by `q_A`), `vd_A` (felt by `d_A`), `vQ_A` (felt by
/// `Q_A`) from the atomic moments `dip`/`quad` and monopoles `q` (eqs 16/34). Pure function
/// of the moments + geometry; no density/integrals needed.
#[allow(clippy::type_complexity)]
fn potentials_from_moments(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    dip: &[Vec3],
    quad: &[[[f64; 3]; 3]],
    q: &[f64],
) -> (Vec<f64>, Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
    use crate::coulomb::harmonic_average;
    // Each atom `a`'s fields are an independent sum over `b`; compute them in parallel. The
    // per-`a` `Σ_b` order is preserved, so the result is bit-identical to the serial path. This
    // is the dominant per-SCC-iteration multipole cost at scale (O(N²) `f^(mn)` evaluations).
    let fields: Vec<(f64, Vec3, [[f64; 3]; 3])> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut s_a = 0.0_f64;
            let mut vd_a = Vec3::zero();
            let mut vq_a = [[0.0_f64; 3]; 3];
            for b in 0..nat {
                if a == b {
                    // On-site: only the dipole-dipole f^(11)_AA and quad-quad f^(22)_AA survive
                    // (f^(01)_AA = f^(12)_AA = 0; monopole-quad vanishes via the traceless Q).
                    // These are r->0 limits, isotropic, position-independent (no gradient).
                    let c = 1.0 / (atom_hardness[a] * atom_hardness[a]);
                    let x = Vec3::new(1.0e-6, 0.0, 0.0);
                    let f11 = f_mn(x, c, 1, 1);
                    let f22 = f_mn(x, c, 2, 2);
                    vd_a += r2_vec(&f11, dip[a]);
                    let aq = r4_quad(&f22, &quad[a]);
                    for i in 0..3 {
                        for j in 0..3 {
                            vq_a[i][j] += aq[i][j];
                        }
                    }
                    continue;
                }
                let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
                let c = 1.0 / (eta * eta);
                let x = atom_pos[a] - atom_pos[b];
                let f01 = f_mn(x, c, 0, 1);
                let f02 = f_mn(x, c, 0, 2);
                let f10 = f_mn(x, c, 1, 0);
                let f11 = f_mn(x, c, 1, 1);
                let f12 = f_mn(x, c, 1, 2);
                let f20 = f_mn(x, c, 2, 0);
                let f21 = f_mn(x, c, 2, 1);
                let f22 = f_mn(x, c, 2, 2);
                s_a += dot1(&f01, dip[b]) + dot2_full(&f02, &quad[b]);
                vd_a += f10_vec(&f10, q[b]) + r2_vec(&f11, dip[b]) + r3_quad(&f12, &quad[b]);
                let add = add_quads(
                    &r2_scaled(&f20, q[b]),
                    &add_quads(&r3_vec(&f21, dip[b]), &r4_quad(&f22, &quad[b])),
                );
                for i in 0..3 {
                    for j in 0..3 {
                        vq_a[i][j] += add[i][j];
                    }
                }
            }
            (s_a, vd_a, vq_a)
        })
        .collect();
    let mut s = vec![0.0_f64; nat];
    let mut vd = vec![Vec3::zero(); nat];
    let mut vq = vec![[[0.0_f64; 3]; 3]; nat];
    for (a, (s_a, vd_a, vq_a)) in fields.into_iter().enumerate() {
        s[a] = s_a;
        vd[a] = vd_a;
        vq[a] = vq_a;
    }
    (s, vd, vq)
}

/// **Arbitrary-rank** multipole fields — the generic orchestration that unifies the legacy
/// monopole/dipole/quadrupole [`potentials_from_moments`] and the [`octupole_fields`] add-on into a
/// single rank loop. `moments[a][l]` is atom `a`'s detraced rank-`l` Cartesian moment as a flat
/// `3^l` vector for `l = 0..=max_rank` (rank 0 = `[q_a]`, rank 1 = `[dx,dy,dz]`, rank 2 = the 9
/// row-major quadrupole components, …). Returns the same-shaped field container `V[a][l]`
/// (flat `3^l`), where
/// `V^(la)_A = Σ_{B≠A} Σ_{lb} f^(la,lb)_AB ⊗ M^(lb)_B + [la≥1] f^(la,la)_AA ⊗ M^(la)_A`.
///
/// Two conventions match the legacy paths exactly: (1) the **pure monopole–monopole** term
/// `(la=lb=0)` is **excluded off-site** — GFN1 already carries it, mDFTB only adds terms with at
/// least one `d`/`Q`/… moment; (2) **on-site** only the **same-rank** `f^(la,la)_AA` survives
/// (`r→0` isotropic, position-free), as cross-rank on-site contributions vanish against the
/// traceless moments. Energy is then `E = ½ Σ_A Σ_{l≥1?} M^(l)_A · V^(l)_A` with `q_A·V^(0)_A`
/// for the monopole row. Reproduces `potentials_from_moments` for `max_rank=2` (gated).
/// Non-periodic; pure function of moments + geometry.
#[allow(dead_code)] // wired in by the arbitrary-rank multipole path (multipole_order ≥ 4)
fn multipole_fields_generic(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    moments: &[Vec<Vec<f64>>],
    max_rank: usize,
) -> Vec<Vec<Vec<f64>>> {
    use crate::coulomb::harmonic_average;
    // Per-(atom,rank) "active" mask: a rank-`l` source contributes nothing if its moment is ~0.
    // The traceless rank-`l` moment vanishes analytically for atoms whose AOs can't reach rank `l`
    // (e.g. s/p-only atoms have zero rank ≥3), so on a typical (mostly light-atom) system only a
    // few atoms carry high-rank moments. Skipping zero-moment **source** terms turns the high-rank
    // part of this O(N²) field into O(N·n_active) — the large-scale win (drops only ≲ε terms; the
    // FD gates pass). Rank 0 (the monopole) is always active.
    let active = moment_active_mask(moments, nat, max_rank);
    (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut field: Vec<Vec<f64>> = (0..=max_rank)
                .map(|la| vec![0.0_f64; 3usize.pow(la as u32)])
                .collect();
            for b in 0..nat {
                if a == b {
                    // On-site: same-rank only (la = lb ≥ 1), r→0 isotropic, position-free.
                    let c = 1.0 / (atom_hardness[a] * atom_hardness[a]);
                    let x = Vec3::new(1.0e-6, 0.0, 0.0);
                    for la in 1..=max_rank {
                        if !active[a][la] {
                            continue;
                        }
                        let f = f_mn_unique(x, c, la, la);
                        let contrib = contract_last_unique(&f, la, la, &moments[a][la]);
                        for (acc, v) in field[la].iter_mut().zip(contrib.iter()) {
                            *acc += v;
                        }
                    }
                    continue;
                }
                let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
                let c = 1.0 / (eta * eta);
                let x = atom_pos[a] - atom_pos[b];
                // Loop the **source** rank `lb` outer so a zero-moment source skips every `la`.
                for lb in 0..=max_rank {
                    if lb >= 1 && !active[b][lb] {
                        continue;
                    }
                    for la in 0..=max_rank {
                        if la == 0 && lb == 0 {
                            continue; // GFN1 already carries the monopole–monopole term.
                        }
                        let f = f_mn_unique(x, c, la, lb);
                        let contrib = contract_last_unique(&f, la, lb, &moments[b][lb]);
                        for (acc, v) in field[la].iter_mut().zip(contrib.iter()) {
                            *acc += v;
                        }
                    }
                }
            }
            field
        })
        .collect()
}

/// Threshold below which an atomic rank-`l` moment is treated as zero ("inactive") for the
/// large-scale screening in the generic multipole path. Conservative — the traceless high-rank
/// moments of s/p-only atoms detrace to machine ε, well below this.
const MULTIPOLE_ACTIVE_EPS: f64 = 1.0e-10;

/// `active[a][l]` = atom `a`'s rank-`l` moment is non-negligible (rank 0 is always active).
/// `pub(crate)` so the periodic generic field/force builders (`pbc::ewald_multipole`) reuse the
/// same source-rank screening for the large-system `O(N·n_active)` high-rank path.
pub(crate) fn moment_active_mask(
    moments: &[Vec<Vec<f64>>],
    nat: usize,
    max_rank: usize,
) -> Vec<Vec<bool>> {
    (0..nat)
        .map(|a| {
            (0..=max_rank)
                .map(|l| l == 0 || moments[a][l].iter().any(|v| v.abs() > MULTIPOLE_ACTIVE_EPS))
                .collect()
        })
        .collect()
}

/// Full contraction of two equal-length flat tensors: `Σ_i a_i b_i`.
#[inline]
fn dot_flat(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Build the **arbitrary-rank moments container** `M[a][l]` (flat `3^l` detraced Cartesian) for
/// `l = 0..=max_rank` from the density: rank 0 = `[q_a]` (the mDFTB monopole), ranks `l ≥ 1` from
/// [`atomic_moment_rank_l`]. Shared by the generic energy/Fock/gradient. Non-periodic.
pub fn build_generic_moments(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Vec<Vec<Vec<f64>>> {
    let mut moments: Vec<Vec<Vec<f64>>> = (0..nat).map(|a| vec![vec![atomic_charges[a]]]).collect();
    for l in 1..=max_rank {
        let ml = atomic_moment_rank_l(basis, nat, atom_pos, integrals, density, l, cache);
        for (a, m) in ml.into_iter().enumerate() {
            moments[a].push(m);
        }
    }
    moments
}

/// **Arbitrary-rank** mDFTB correction energy + Fock shift from **given** generic moments
/// `M[a][l]` (flat `3^l`, `l=0..=max_rank`; rank 0 = `[q_a]`). The generic counterpart of
/// [`multipole_fock_from_moments`] (rank ≤2) and [`octupole_fock_from_moments`] (rank 3): it uses
/// the unified [`multipole_fields_generic`] for the fields and a single rank-summed moment matrix
/// for the Fock. `E = ½ Σ_A Σ_{l} M^(l)_A · V^(l)_A`; the Fock is the symmetric overlap shift
/// `assemble_shift` of the rank-summed on-site moment operator `Σ_{l≥1} V^(l)_A · detrace(M̄^(l)_{μκ})`
/// (each AO moment integral [`crate::integrals::contracted_moment_rank`] detraced to match the
/// traceless atomic moment, exactly as the legacy quad/octupole shifts do). Reproduces the legacy
/// energy+Fock for `max_rank=2,3` (gated). The exact `∂E/∂P`, so the SCC stays variational.
/// Non-periodic.
#[allow(dead_code)] // wired in by the arbitrary-rank multipole path (multipole_order ≥ 4)
pub fn multipole_fock_generic(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    moments: &[Vec<Vec<f64>>],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> MultipoleEnergyFock {
    let n = basis.len();
    let v = multipole_fields_generic(nat, atom_hardness, atom_pos, moments, max_rank);
    // Energy: ½ Σ_A Σ_l M^(l)_A · V^(l)_A (rank 0 = q_A · s_A).
    let mut energy = 0.0;
    for a in 0..nat {
        let mut e_a = 0.0;
        for l in 0..=max_rank {
            e_a += dot_flat(&moments[a][l], &v[a][l]);
        }
        energy += 0.5 * e_a;
    }
    // Fock: symmetric overlap shift of the rank-summed on-site moment operator.
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, &v, max_rank, cache);
    let fock = assemble_shift(
        n,
        &per_atom,
        &atom_of,
        &s_field,
        &dmat,
        &mmat,
        &integrals.overlap,
    );
    MultipoleEnergyFock { energy, fock }
}

/// Build the arbitrary-rank multipole on-site Fock from **externally supplied** per-atom fields
/// `v[A][l]` (full `3^l` layout) — the `∂E/∂P` operator for `E = ½ Σ_A Σ_l M[A][l]·V[A][l]` with
/// *given* `V`. The periodic SCC ([`crate::pbc::scf`]) calls this with the QCore-Ewald fields from
/// [`crate::pbc::ewald_multipole::periodic_multipole_fields_generic`] instead of the molecular
/// kernel. The rank-0 (charge) potential `v[A][0]` enters via the scalar shell shift; zero it
/// before calling to get a **moment-only** Fock (ranks ≥ 1) when the charge route is handled
/// separately through the Bloch overlap `S(k)`.
pub fn multipole_fock_from_fields(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    v: &[Vec<Vec<f64>>],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Matrix {
    let n = basis.len();
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, v, max_rank, cache);
    assemble_shift(
        n,
        &per_atom,
        &atom_of,
        &s_field,
        &dmat,
        &mmat,
        &integrals.overlap,
    )
}

/// Build the arbitrary-rank multipole **overlap-Pulay weight** `W = ∂E_mp/∂S` from externally
/// supplied per-atom fields `v[A][l]` — the same symmetric shift assembly as
/// [`multipole_fock_from_fields`] but with the *density* `P` as the base instead of the overlap
/// (`Fock = shift(v, S)`, `W = shift(v, P)`; the assembly is the symmetric bilinear conjugate, so
/// `tr[Fock·P] = tr[W·S]`). The periodic analytic gradient ([`crate::pbc::gradient`]) contracts
/// `W` with `dS/dR` for the multipole's overlap-derivative (Pulay) force; the fields are the
/// QCore-Ewald [`crate::pbc::ewald_multipole::periodic_multipole_fields_generic`] (full ranks,
/// including the rank-0 charge route).
pub fn multipole_weight_from_fields(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    density: &Matrix,
    v: &[Vec<Vec<f64>>],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Matrix {
    let n = basis.len();
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, v, max_rank, cache);
    assemble_shift(n, &per_atom, &atom_of, &s_field, &dmat, &mmat, density)
}

/// Energy-only of the arbitrary-rank generic multipole correction: `E = ½ Σ_A Σ_l M^(l)_A · V^(l)_A`
/// — the [`multipole_fock_generic`] energy **without** building the Fock (no O(N²) shift assembly),
/// for the SCC energy at the output density. Byte-identical to `multipole_fock_generic(..).energy`
/// (same fields, same accumulation), but skips the unused Fock pass each iteration.
pub fn multipole_energy_generic(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    moments: &[Vec<Vec<f64>>],
    max_rank: usize,
) -> f64 {
    let v = multipole_fields_generic(nat, atom_hardness, atom_pos, moments, max_rank);
    let mut energy = 0.0;
    for a in 0..nat {
        let mut e_a = 0.0;
        for l in 0..=max_rank {
            e_a += dot_flat(&moments[a][l], &v[a][l]);
        }
        energy += 0.5 * e_a;
    }
    energy
}

/// Build the inputs to [`assemble_shift`] for the generic arbitrary-rank path from the fields
/// `v[a][l]`: the rank-0 field `s_A`, a zero `dmat`, and the **rank-summed on-site moment
/// operator** `mmat_{μκ} = Σ_{l≥1} V^(l)_A · detrace(M̄^(l)_{μκ})` (each AO moment integral
/// detraced to match the traceless atomic moment, as the legacy quad/octupole shifts do). Shared
/// by the generic Fock (`base = S`) and the generic overlap-Pulay weight (`base = P`).
#[allow(clippy::type_complexity)]
fn generic_shift_inputs(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    v: &[Vec<Vec<f64>>],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> (Vec<f64>, Matrix, Matrix, Vec<Vec<usize>>, Vec<usize>) {
    let n = basis.len();
    let per_atom = atom_ao_lists(basis, nat);
    let mut atom_of = vec![0usize; n];
    for (a, aos) in per_atom.iter().enumerate() {
        for &mu in aos {
            atom_of[mu] = a;
        }
    }
    let s_field: Vec<f64> = (0..nat).map(|a| v[a][0][0]).collect();
    let dmat = Matrix::zeros(n, n);
    let mut mmat = Matrix::zeros(n, n);
    for (a, aos) in per_atom.iter().enumerate() {
        let ra = atom_pos[a];
        for (mi, &mu) in aos.iter().enumerate() {
            for (ki, &ka) in aos.iter().enumerate() {
                let mut acc = 0.0;
                for l in 1..=max_rank {
                    // Raw on-site moment operator from the cache (geometry-fixed) or recomputed,
                    // then detraced to match the traceless atomic moment.
                    let raw_owned;
                    let raw: &[f64] = match cache {
                        Some(c) => c.get(a, l, mi, ki),
                        None => {
                            raw_owned = expand_symmetric_cartesian(
                                &crate::integrals::contracted_moment_rank(
                                    &basis.aos[mu],
                                    &basis.aos[ka],
                                    ra,
                                    ra,
                                    l,
                                ),
                                l,
                            );
                            &raw_owned
                        }
                    };
                    let mbar = detrace_symmetric(raw, l);
                    acc += dot_flat(&v[a][l], &mbar);
                }
                mmat[(mu, ka)] = acc;
            }
        }
    }
    (s_field, dmat, mmat, per_atom, atom_of)
}

/// **Arbitrary-rank** kernel-force contribution to the analytic gradient — the generic
/// counterpart of [`multipole_kernel_forces`] (rank ≤2) + [`octupole_kernel_forces`] (rank 3),
/// unified into one `(la,lb)` loop over [`kernel_grad`]. The explicit off-site `∂f^(la,lb)_AB/∂R`
/// derivatives contracted with the fixed moment pairs; on-site terms are position-independent
/// (they enter only the overlap-Pulay weight). The pure monopole–monopole `(0,0)` term is GFN1's
/// own and excluded. `grad[a] = Σ_{b≠a} ½ (g(a,b) − g(b,a))`. Non-periodic.
pub fn multipole_kernel_forces_generic(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    moments: &[Vec<Vec<f64>>],
    max_rank: usize,
) -> Vec<Vec3> {
    use crate::coulomb::harmonic_average;
    // Active-rank screening (see `multipole_fields_generic`): every kernel-force term is bilinear in
    // `M_a^(la)` and `M_b^(lb)`, so a zero moment on **either** side kills the term — skip it.
    let active = moment_active_mask(moments, nat, max_rank);
    let gforce = |a: usize, b: usize| -> Vec3 {
        let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
        let c = 1.0 / (eta * eta);
        let x = atom_pos[a] - atom_pos[b];
        let mut g = Vec3::zero();
        for la in 0..=max_rank {
            if la >= 1 && !active[a][la] {
                continue;
            }
            for lb in 0..=max_rank {
                if la == 0 && lb == 0 {
                    continue; // GFN1 carries the monopole–monopole force.
                }
                if lb >= 1 && !active[b][lb] {
                    continue;
                }
                let df = f_mn_grad_unique(x, c, la, lb);
                g += kernel_grad_unique(&df, la, lb, &moments[a][la], &moments[b][lb]);
            }
        }
        g
    };
    (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut ga = Vec3::zero();
            for b in 0..nat {
                if a == b {
                    continue;
                }
                ga += (gforce(a, b) - gforce(b, a)) * 0.5;
            }
            ga
        })
        .collect()
}

/// **Arbitrary-rank** overlap-Pulay weight `W = ∂E_mp/∂S` for the analytic gradient — the generic
/// counterpart of [`multipole_overlap_weight`] (+ [`octupole_overlap_weight`]). Identical shift
/// assembly to [`multipole_fock_generic`] but contracted with the density `P` rather than the
/// overlap `S`. Non-periodic.
pub fn multipole_overlap_weight_generic(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    density: &Matrix,
    moments: &[Vec<Vec<f64>>],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Matrix {
    let n = basis.len();
    let v = multipole_fields_generic(nat, atom_hardness, atom_pos, moments, max_rank);
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, &v, max_rank, cache);
    assemble_shift(n, &per_atom, &atom_of, &s_field, &dmat, &mmat, density)
}

/// Threshold above which an atomic traceless octupole is treated as nonzero ("active"). The
/// rank-3 moment is analytically zero for s/p-only atoms (pure-trace → detraced to ≈machine ε),
/// so this cleanly separates genuine d/f-atom octupoles (~1e-2) from numerical noise (~1e-16).
const OCTU_ACTIVE_EPS: f64 = 1.0e-8;

#[inline]
fn octu_is_active(o: &[[[f64; 3]; 3]; 3]) -> bool {
    o.iter()
        .flatten()
        .flatten()
        .any(|v| v.abs() > OCTU_ACTIVE_EPS)
}

/// Octupole contributions to the mDFTB fields (gated; only evaluated when the octupole
/// correction is on): the octupole field `vo_A` (felt by `O_A`) plus the octupole's
/// additions to the monopole/dipole/quadrupole fields, via every `f^(m,3)` / `f^(3,n)`
/// interaction. Uses the general [`contract_last`], leaving the dipole+quad
/// [`potentials_from_moments`] untouched. On-site only the same-rank `f^(3,3)_AA`
/// survives (like `f^(1,1)`/`f^(2,2)`; cross-rank on-site terms vanish against the
/// traceless moments). Returns `(extra_s, extra_vd, extra_vq, vo)`.
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
fn octupole_fields(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    q: &[f64],
    dip: &[Vec3],
    quad: &[[[f64; 3]; 3]],
    octu: &[[[[f64; 3]; 3]; 3]],
) -> (
    Vec<f64>,
    Vec<Vec3>,
    Vec<[[f64; 3]; 3]>,
    Vec<[[[f64; 3]; 3]; 3]>,
) {
    use crate::coulomb::harmonic_average;
    // Only atoms with a (numerically) nonzero traceless octupole participate: the rank-3 moment
    // vanishes analytically for s/p-only atoms (it is pure-trace), so on typical molecules only
    // d/f atoms (Pd, Fe, S, …) are "active". Every octupole interaction (`f^(m,3)`/`f^(3,n)`)
    // carries an `O_A` or `O_B`, so contributions from inactive atoms are ~0 (≈machine ε from the
    // detrace). Screening on `active` turns this O(N²) pairwise field into O(N·n_active) — the
    // dominant per-SCC-iteration octupole cost on d-block complexes. (Drops ≲1e-16 terms; the FD
    // gates and cache-consistency test pass.)
    let active: Vec<usize> = (0..nat).filter(|&b| octu_is_active(&octu[b])).collect();
    let fields: Vec<(f64, Vec3, [[f64; 3]; 3], [f64; 27])> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut es_a = 0.0_f64;
            let mut evd_a = Vec3::zero();
            let mut evq_a = [[0.0_f64; 3]; 3];
            let mut vo_a = [0.0_f64; 27];
            // Field on q_A / d_A / Q_A from O_B: only active `b` contribute (O_B ≈ 0 otherwise).
            for &b in &active {
                if b == a {
                    continue;
                }
                let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
                let c = 1.0 / (eta * eta);
                let x = atom_pos[a] - atom_pos[b];
                let ob = octu_flat(&octu[b]);
                es_a += contract_last(&f_mn(x, c, 0, 3), 0, 3, &ob)[0];
                let evda = contract_last(&f_mn(x, c, 1, 3), 1, 3, &ob);
                evd_a += Vec3::new(evda[0], evda[1], evda[2]);
                let evqa = contract_last(&f_mn(x, c, 2, 3), 2, 3, &ob);
                for i in 0..3 {
                    for j in 0..3 {
                        evq_a[i][j] += evqa[i * 3 + j];
                    }
                }
            }
            // The octupole field `vo_A` is only used contracted with `O_A` downstream, so compute
            // it only for active `a`. Its q/d/Q sources span all `b`; its `O_B` source only active.
            if octu_is_active(&octu[a]) {
                for b in 0..nat {
                    if a == b {
                        // On-site: only f^(3,3)_AA (same-rank, r->0 isotropic, position-free).
                        let c = 1.0 / (atom_hardness[a] * atom_hardness[a]);
                        let f33 = f_mn(Vec3::new(1.0e-6, 0.0, 0.0), c, 3, 3);
                        let voa = contract_last(&f33, 3, 3, &octu_flat(&octu[a]));
                        for (acc, v) in vo_a.iter_mut().zip(voa.iter()) {
                            *acc += v;
                        }
                        continue;
                    }
                    let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
                    let c = 1.0 / (eta * eta);
                    let x = atom_pos[a] - atom_pos[b];
                    let qb = [q[b]];
                    let db = [dip[b].x, dip[b].y, dip[b].z];
                    let qbf = quad_flat(&quad[b]);
                    let f30 = contract_last(&f_mn(x, c, 3, 0), 3, 0, &qb);
                    let f31 = contract_last(&f_mn(x, c, 3, 1), 3, 1, &db);
                    let f32 = contract_last(&f_mn(x, c, 3, 2), 3, 2, &qbf);
                    for i in 0..27 {
                        vo_a[i] += f30[i] + f31[i] + f32[i];
                    }
                    if octu_is_active(&octu[b]) {
                        let f33 = contract_last(&f_mn(x, c, 3, 3), 3, 3, &octu_flat(&octu[b]));
                        for i in 0..27 {
                            vo_a[i] += f33[i];
                        }
                    }
                }
            }
            (es_a, evd_a, evq_a, vo_a)
        })
        .collect();
    let mut es = vec![0.0_f64; nat];
    let mut evd = vec![Vec3::zero(); nat];
    let mut evq = vec![[[0.0_f64; 3]; 3]; nat];
    let mut vo = Vec::with_capacity(nat);
    for (a, (es_a, evd_a, evq_a, vo_a)) in fields.into_iter().enumerate() {
        es[a] = es_a;
        evd[a] = evd_a;
        evq[a] = evq_a;
        vo.push(octu_unflat(&vo_a));
    }
    (es, evd, evq, vo)
}

/// Octupole contribution to the experimental multipole correction energy + Fock at a
/// given density (added to the dipole+quad mDFTB2 result). The traceless atomic
/// octupoles are read from the density via [`atomic_octupole_moments`]; the octupole
/// fields ([`octupole_fields`]) drive both the energy `½ Σ_A (q·s + d·vd + Q:vQ + O:vo)`
/// and the Fock shift — the octupole-induced dipole/quad fields reuse the mDFTB
/// [`shift_matrices`]/[`assemble_shift`] machinery, and the octupole field itself
/// contracts with the on-site octupole operator. Mirrors [`multipole_energy_fock`].
pub fn octupole_energy_fock(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
    cache: Option<&OnsiteOctupoleCache>,
) -> MultipoleEnergyFock {
    let m = atomic_moments(basis, nat, integrals, density);
    let octu = atomic_octupole_moments(basis, nat, atom_pos, integrals, density, cache);
    octupole_fock_from_moments(
        basis,
        nat,
        atom_hardness,
        atom_pos,
        integrals,
        atomic_charges,
        &m.dipole,
        &m.quad,
        &octu,
        cache,
    )
}

/// Octupole energy + Fock from **given** atomic moments (the SCC path, where the
/// dipole/quadrupole/octupole moments are Broyden-mixed alongside the monopole charges);
/// see [`octupole_energy_fock`] for the construction and the traceless-operator note.
#[allow(clippy::too_many_arguments)]
pub fn octupole_fock_from_moments(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    q: &[f64],
    dip: &[Vec3],
    quad: &[[[f64; 3]; 3]],
    octu: &[[[[f64; 3]; 3]; 3]],
    cache: Option<&OnsiteOctupoleCache>,
) -> MultipoleEnergyFock {
    let n = basis.len();
    let (es, evd, evq, vo) = octupole_fields(nat, atom_hardness, atom_pos, q, dip, quad, octu);
    let mut energy = 0.0;
    for a in 0..nat {
        let oo: f64 = octu_flat(&octu[a])
            .iter()
            .zip(octu_flat(&vo[a]).iter())
            .map(|(x, y)| x * y)
            .sum();
        energy += 0.5 * (q[a] * es[a] + dip[a].dot(evd[a]) + dot2_full_mat(&quad[a], &evq[a]) + oo);
    }
    let (dmat, mut mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &evd, &evq);
    for (a, aos) in per_atom.iter().enumerate() {
        let ra = atom_pos[a];
        for (mi, &mu) in aos.iter().enumerate() {
            for (ki, &ka) in aos.iter().enumerate() {
                // The moment `O_A` is traceless, so the shift uses the traceless octupole
                // operator (mirroring the traceless quadrupole in `shift_matrices`).
                let obar =
                    detrace_octupole(&onsite_octupole_cached(cache, basis, a, mi, ki, mu, ka, ra));
                let mut s = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            s += vo[a][i][j][k] * obar[i][j][k];
                        }
                    }
                }
                mmat[(mu, ka)] += s;
            }
        }
    }
    let fock = assemble_shift(
        n,
        &per_atom,
        &atom_of,
        &es,
        &dmat,
        &mmat,
        &integrals.overlap,
    );
    MultipoleEnergyFock { energy, fock }
}

/// Contract a flat `f^(mn)` gradient tensor `df` (rank `1+m+n`, first index = gradient
/// direction, then the `m+n` moment indices) with flat moments `ma` (length `3^m`) and
/// `mb` (length `3^n`): `g[d] = Σ_ij df[d,i,j] ma[i] mb[j]`.
fn kernel_grad(df: &[f64], m: usize, n: usize, ma: &[f64], mb: &[f64]) -> Vec3 {
    let dm = 3usize.pow(m as u32);
    let dn = 3usize.pow(n as u32);
    let mut o = [0.0_f64; 3];
    for (d, od) in o.iter_mut().enumerate() {
        for (i, &mai) in ma.iter().enumerate().take(dm) {
            for (j, &mbj) in mb.iter().enumerate().take(dn) {
                *od += df[(d * dm + i) * dn + j] * mai * mbj;
            }
        }
    }
    Vec3::new(o[0], o[1], o[2])
}

/// Octupole contribution to the mDFTB **kernel** gradient: the off-site `df^(m,3)/dR` /
/// `df^(3,n)/dR` derivatives contracted with the fixed moment pairs (every interaction
/// involving an octupole). Added on top of [`multipole_kernel_forces`]. On-site terms
/// translate rigidly (no derivative). Uses the general [`kernel_grad`].
pub fn octupole_kernel_forces(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
    cache: Option<&OnsiteOctupoleCache>,
) -> Vec<Vec3> {
    use crate::coulomb::harmonic_average;
    let q = atomic_charges;
    let m = atomic_moments(basis, nat, integrals, density);
    let octu = atomic_octupole_moments(basis, nat, atom_pos, integrals, density, cache);
    // Octupole kernel force from the ordered pair (a,b) on atom a. As in
    // [`multipole_kernel_forces`], `grad[a] = Σ_{b≠a} ½ (g(a,b) − g(b,a))`, computed per atom in
    // parallel (each ordered kernel recomputed once per endpoint; value-identical up to ~1 ULP).
    let gforce = |a: usize, b: usize| -> Vec3 {
        let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
        let c = 1.0 / (eta * eta);
        let x = atom_pos[a] - atom_pos[b];
        let (oa, ob) = (octu_flat(&octu[a]), octu_flat(&octu[b]));
        let (qa, qb) = ([q[a]], [q[b]]);
        let da = [m.dipole[a].x, m.dipole[a].y, m.dipole[a].z];
        let db = [m.dipole[b].x, m.dipole[b].y, m.dipole[b].z];
        let (qaf, qbf) = (quad_flat(&m.quad[a]), quad_flat(&m.quad[b]));
        let mut g = Vec3::zero();
        g += kernel_grad(&f_mn_grad(x, c, 0, 3), 0, 3, &qa, &ob); // q_A·O_B
        g += kernel_grad(&f_mn_grad(x, c, 1, 3), 1, 3, &da, &ob); // d_A·O_B
        g += kernel_grad(&f_mn_grad(x, c, 2, 3), 2, 3, &qaf, &ob); // Q_A·O_B
        g += kernel_grad(&f_mn_grad(x, c, 3, 0), 3, 0, &oa, &qb); // O_A·q_B
        g += kernel_grad(&f_mn_grad(x, c, 3, 1), 3, 1, &oa, &db); // O_A·d_B
        g += kernel_grad(&f_mn_grad(x, c, 3, 2), 3, 2, &oa, &qbf); // O_A·Q_B
        g += kernel_grad(&f_mn_grad(x, c, 3, 3), 3, 3, &oa, &ob); // O_A·O_B
        g
    };
    // Every term in `gforce` carries an `O_A` or `O_B`, so it vanishes unless `a` or `b` is an
    // active (nonzero-octupole) atom. Screen as in `octupole_fields`: an active `a` sums over all
    // `b`; an inactive `a` only over active `b` (turns the gradient O(N²)→O(N·n_active)).
    let active: Vec<usize> = (0..nat).filter(|&b| octu_is_active(&octu[b])).collect();
    (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut ga = Vec3::zero();
            if octu_is_active(&octu[a]) {
                for b in 0..nat {
                    if a == b {
                        continue;
                    }
                    ga += (gforce(a, b) - gforce(b, a)) * 0.5;
                }
            } else {
                for &b in &active {
                    if a == b {
                        continue;
                    }
                    ga += (gforce(a, b) - gforce(b, a)) * 0.5;
                }
            }
            ga
        })
        .collect()
}

/// Octupole contribution to the overlap-Pulay weight `W = ∂E_octu/∂S` (added to
/// [`multipole_overlap_weight`]); the same shift assembly as [`octupole_fock_from_moments`]
/// but contracted with the density rather than the overlap.
pub fn octupole_overlap_weight(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
    cache: Option<&OnsiteOctupoleCache>,
) -> Matrix {
    let n = basis.len();
    let m = atomic_moments(basis, nat, integrals, density);
    let octu = atomic_octupole_moments(basis, nat, atom_pos, integrals, density, cache);
    let (es, evd, evq, vo) = octupole_fields(
        nat,
        atom_hardness,
        atom_pos,
        atomic_charges,
        &m.dipole,
        &m.quad,
        &octu,
    );
    let (dmat, mut mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &evd, &evq);
    for (a, aos) in per_atom.iter().enumerate() {
        let ra = atom_pos[a];
        for (mi, &mu) in aos.iter().enumerate() {
            for (ki, &ka) in aos.iter().enumerate() {
                let obar =
                    detrace_octupole(&onsite_octupole_cached(cache, basis, a, mi, ki, mu, ka, ra));
                let mut s = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            s += vo[a][i][j][k] * obar[i][j][k];
                        }
                    }
                }
                mmat[(mu, ka)] += s;
            }
        }
    }
    assemble_shift(n, &per_atom, &atom_of, &es, &dmat, &mmat, density)
}

/// Block-diagonal dipole/quadrupole "shift" matrices `D_{μκ} = vd_A·d̄_{μκ}` and
/// `M_{μκ} = vQ_A:Q̄^tl_{μκ}` (μ,κ on the same atom A), plus the AO->atom map. Used by both
/// the Fock (contract with S) and the overlap-Pulay weight (contract with P).
#[allow(clippy::type_complexity)]
fn shift_matrices(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    vd: &[Vec3],
    vq: &[[[f64; 3]; 3]],
) -> (Matrix, Matrix, Vec<Vec<usize>>, Vec<usize>) {
    let n = basis.len();
    let per_atom = atom_ao_lists(basis, nat);
    let mut atom_of = vec![0usize; n];
    for (a, aos) in per_atom.iter().enumerate() {
        for &mu in aos {
            atom_of[mu] = a;
        }
    }
    let mut dmat = Matrix::zeros(n, n);
    let mut mmat = Matrix::zeros(n, n);
    for (a, aos) in per_atom.iter().enumerate() {
        for &mu in aos {
            for &ka in aos {
                let dbar = onsite_dipole(integrals, mu, ka);
                let qbar = traceless(&onsite_quad(integrals, mu, ka));
                dmat[(mu, ka)] = vd[a].dot(dbar);
                mmat[(mu, ka)] = dot2_full_mat(&vq[a], &qbar);
            }
        }
    }
    (dmat, mmat, per_atom, atom_of)
}

/// Assemble the symmetric multipole shift `½(s_A+s_B)·base + ½(D·base + base·D) +
/// ½(M·base + base·M)`. With `base = S` this is the Fock shift (eq 33); with `base = P`
/// it is the overlap-Pulay weight `W = ∂E_mp/∂S` (the same bilinear, S↔P swapped).
fn assemble_shift(
    n: usize,
    per_atom: &[Vec<usize>],
    atom_of: &[usize],
    s: &[f64],
    dmat: &Matrix,
    mmat: &Matrix,
    base: &Matrix,
) -> Matrix {
    // Each row `mu` is independent; build the rows in parallel (bit-identical — every cell is a
    // deterministic contraction) and copy them into the dense output. This O(N²) assembly runs
    // once per SCC iteration per multipole term, so it matters at scale.
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|mu| {
            let am = atom_of[mu];
            let mut row = vec![0.0_f64; n];
            for (nu, slot) in row.iter_mut().enumerate() {
                let an = atom_of[nu];
                let mut f = 0.5 * (s[am] + s[an]) * base[(mu, nu)];
                let mut ds = 0.0;
                let mut sd = 0.0;
                let mut ms = 0.0;
                let mut sm = 0.0;
                for ka in &per_atom[am] {
                    ds += dmat[(mu, *ka)] * base[(*ka, nu)];
                    ms += mmat[(mu, *ka)] * base[(*ka, nu)];
                }
                for kb in &per_atom[an] {
                    sd += base[(mu, *kb)] * dmat[(*kb, nu)];
                    sm += base[(mu, *kb)] * mmat[(*kb, nu)];
                }
                f += 0.5 * (ds + sd) + 0.5 * (ms + sm);
                *slot = f;
            }
            row
        })
        .collect();
    let mut out = Matrix::zeros(n, n);
    for (mu, row) in rows.iter().enumerate() {
        for (nu, &v) in row.iter().enumerate() {
            out[(mu, nu)] = v;
        }
    }
    out
}

/// The overlap-Pulay weight `W_{κν} = ∂E_mp/∂S_{κν}` at fixed density (for the variational
/// gradient: the multipole energy's explicit `dS/dR` term is `Σ_{κν} W_{κν} dS_{κν}/dR`).
/// It is the Fock construction with the overlap `S` replaced by the density `P`.
pub fn multipole_overlap_weight(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
) -> Matrix {
    let n = basis.len();
    let (_moments, s, vd, vq) = moments_and_potentials(
        basis,
        nat,
        atom_hardness,
        atom_pos,
        integrals,
        density,
        atomic_charges,
    );
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd, &vq);
    assemble_shift(n, &per_atom, &atom_of, &s, &dmat, &mmat, density)
}

// =====================================================================================
// CAMM-on-mDFTB2 (v0.4.2): cumulative atomic multipoles + GFN2-style AES off-site term.
// =====================================================================================

/// Squared atom-pair cutoff (bohr²) for the cumulative-moment two-center AO loops. Beyond the
/// GFN1 integral cutoff (30 bohr) every two-center overlap/dipole/quadrupole integral is
/// negligible, so far pairs contribute nothing — skipping them turns the `O(n²)` AO-pair loops
/// (moment build, Fock shift, gradient integral derivatives) into `O(n·neighbors)` for large,
/// spatially-extended systems. Same-atom pairs (distance 0) are always kept (s–p dipole integrals
/// are nonzero at zero overlap). Result-preserving (matches the un-screened FD-gated path).
const CAMM_PAIR_CUTOFF_SQ: f64 = 30.0 * 30.0;

/// Per-AO atom index (`atom_of[μ]` = atom carrying AO `μ`).
fn ao_atom_index(basis: &BasisSet, nat: usize) -> Vec<usize> {
    let mut atom_of = vec![0usize; basis.len()];
    for (a, aos) in atom_ao_lists(basis, nat).into_iter().enumerate() {
        for mu in aos {
            atom_of[mu] = a;
        }
    }
    atom_of
}

/// `⟨bra|(r − R_{atom(bra)})|ket⟩` (3-vector). The stored dipole integral is ket-centred
/// (`D_{ij}=⟨i|(r−R_{atom(j)})|j⟩`), so a shift by `R_{atom(ket)} − R_{atom(bra)}` re-references it
/// to the **bra** atom (the CAMM atom-A index is the bra). `S` is the overlap.
#[inline]
fn referenced_dipole(
    integrals: &IntegralMatrices,
    atom_pos: &[Vec3],
    atom_of: &[usize],
    bra: usize,
    ket: usize,
) -> Vec3 {
    let d = onsite_dipole(integrals, bra, ket);
    let s = integrals.overlap[(bra, ket)];
    let delta = atom_pos[atom_of[ket]] - atom_pos[atom_of[bra]];
    d + delta * s
}

/// `⟨bra|(r−R_A)(r−R_A)|ket⟩` (symmetric 3×3, **raw** — caller detraces), `A = atom(bra)`.
/// `(r−R_A) = (r−R_{ket}) + δ`, `δ = R_{atom(ket)} − R_{atom(bra)}`, so
/// `Q^A = Quad_ket-centred + δ⊗D + D⊗δ + δ⊗δ·S`.
#[inline]
fn referenced_quad(
    integrals: &IntegralMatrices,
    atom_pos: &[Vec3],
    atom_of: &[usize],
    bra: usize,
    ket: usize,
) -> [[f64; 3]; 3] {
    let q = onsite_quad(integrals, bra, ket); // ket-centred second moment
    let d = onsite_dipole(integrals, bra, ket); // ket-centred dipole
    let s = integrals.overlap[(bra, ket)];
    let delta = atom_pos[atom_of[ket]] - atom_pos[atom_of[bra]];
    let dl = [delta.x, delta.y, delta.z];
    let dv = [d.x, d.y, d.z];
    let mut out = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] = q[a][b] + dl[a] * dv[b] + dl[b] * dv[a] + dl[a] * dl[b] * s;
        }
    }
    out
}

/// **Cumulative atomic multipole moments** (CAMM) `μ_A`, `Θ_A` (traceless) from the density `P`:
/// `μ_A = Σ_{κ∈A} Σ_λ P_κλ ⟨κ|(r−R_A)|λ⟩`, `Θ_A = detrace(Σ_{κ∈A} Σ_λ P_κλ ⟨κ|(r−R_A)(r−R_A)|λ⟩)`.
/// One AO index (the bra `κ`) is restricted to atom `A`; the other (`λ`) runs over **all** AOs —
/// the cumulative (Stone/GFN2) partitioning that conserves the molecular dipole/quadrupole, unlike
/// the on-site [`atomic_moments`]. Same positive-density sign convention as [`atomic_moments`].
pub fn camm_atomic_moments(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    density: &Matrix,
    atom_pos: &[Vec3],
) -> AtomicMoments {
    let n = basis.len();
    let per_atom = atom_ao_lists(basis, nat);
    let atom_of = ao_atom_index(basis, nat);
    let moments: Vec<(Vec3, [[f64; 3]; 3])> = per_atom
        .par_iter()
        .map(|aos| {
            let mut d = Vec3::zero();
            let mut q = [[0.0_f64; 3]; 3];
            for &ka in aos {
                let ra = atom_pos[atom_of[ka]];
                for la in 0..n {
                    let p = density[(ka, la)];
                    if p == 0.0 {
                        continue;
                    }
                    if (atom_pos[atom_of[la]] - ra).norm2() > CAMM_PAIR_CUTOFF_SQ {
                        continue;
                    }
                    d += referenced_dipole(integrals, atom_pos, &atom_of, ka, la) * p;
                    let rq = referenced_quad(integrals, atom_pos, &atom_of, ka, la);
                    for i in 0..3 {
                        for j in 0..3 {
                            q[i][j] += rq[i][j] * p;
                        }
                    }
                }
            }
            // Traceless quadrupole (same convention as the on-site moments).
            let tr = (q[0][0] + q[1][1] + q[2][2]) / 3.0;
            for (i, qi) in q.iter_mut().enumerate() {
                qi[i] -= tr;
            }
            (d, q)
        })
        .collect();
    let mut dipole = vec![Vec3::zero(); nat];
    let mut quad = vec![[[0.0_f64; 3]; 3]; nat];
    for (a, (d, q)) in moments.into_iter().enumerate() {
        dipole[a] = d;
        quad[a] = q;
    }
    AtomicMoments { dipole, quad }
}

/// The CAMM/AES potentials `s_A` (felt by `q_A`), `vd_A` (felt by `μ_A`), `vq_A` (felt by `Θ_A`),
/// **already combined** as `s_onsite·(on-site mDFTB penalty) + s_AES·(off-site GFN2-AES)`. Off-site
/// keeps only the GFN2 AES set `q–μ, q–Θ, μ–μ` (the erf-cloud `f^(0,1)/f^(1,0)/f^(0,2)/f^(2,0)/f^(1,1)`;
/// the `μ–Θ`/`Θ–Θ` tensors `f^(1,2)/f^(2,1)/f^(2,2)` are excluded); the on-site (`a==b`) self-energy is
/// the pure-Ohno mDFTB penalty `g_dd`/`g_QQ`, scaled by the **per-atom** `s_onsite`
/// (`onsite_scale[a]`, element-resolved so each σ-hole type gets its own on-site temper). `σ_AB =
/// κ·exchange_sigma_pair(η_A,η_B)`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn camm_aes_potentials(
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    dip: &[Vec3],
    quad: &[[[f64; 3]; 3]],
    q: &[f64],
    kappa: &[f64],
    scale: f64,
    onsite_scale: &[f64],
) -> (Vec<f64>, Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
    use crate::coulomb::{exchange_sigma_pair, harmonic_average};
    let fields: Vec<(f64, Vec3, [[f64; 3]; 3])> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut s_a = 0.0_f64;
            let mut vd_a = Vec3::zero();
            let mut vq_a = [[0.0_f64; 3]; 3];
            for b in 0..nat {
                if a == b {
                    // On-site mDFTB penalty (pure Ohno r→0 self-energy g_dd, g_QQ), scaled by
                    // s_onsite (the lever that tempers the cumulative-moment over-penalization;
                    // s_onsite=1 is byte-identical to the un-scaled penalty).
                    let c = 1.0 / (atom_hardness[a] * atom_hardness[a]);
                    let x = Vec3::new(1.0e-6, 0.0, 0.0);
                    let f11 = f_mn(x, c, 1, 1);
                    let f22 = f_mn(x, c, 2, 2);
                    vd_a += r2_vec(&f11, dip[a]) * onsite_scale[a];
                    let aq = r4_quad(&f22, &quad[a]);
                    for i in 0..3 {
                        for j in 0..3 {
                            vq_a[i][j] += onsite_scale[a] * aq[i][j];
                        }
                    }
                    continue;
                }
                // Off-site GFN2-AES (erf-cloud, q–μ, q–Θ, μ–μ only), scaled by s_AES.
                // Element-specific range factor: σ_AB = √(κ_A·κ_B)·σ^HP (symmetric geometric mean).
                let sigma =
                    (kappa[a] * kappa[b]).sqrt() * exchange_sigma_pair(atom_hardness[a], atom_hardness[b]);
                let x = atom_pos[a] - atom_pos[b];
                let f01 = f_mn_cloud(x, sigma, 0, 1); // q_A felt by μ_B
                let f02 = f_mn_cloud(x, sigma, 0, 2); // q_A felt by Θ_B
                let f10 = f_mn_cloud(x, sigma, 1, 0); // μ_A felt by q_B
                let f11 = f_mn_cloud(x, sigma, 1, 1); // μ_A felt by μ_B
                let f20 = f_mn_cloud(x, sigma, 2, 0); // Θ_A felt by q_B
                let _ = harmonic_average; // (kept off-site kernel = erf-cloud, not Ohno)
                s_a += scale * (dot1(&f01, dip[b]) + dot2_full(&f02, &quad[b]));
                vd_a += (f10_vec(&f10, q[b]) + r2_vec(&f11, dip[b])) * scale;
                let aq = r2_scaled(&f20, q[b]);
                for i in 0..3 {
                    for j in 0..3 {
                        vq_a[i][j] += scale * aq[i][j];
                    }
                }
            }
            (s_a, vd_a, vq_a)
        })
        .collect();
    let mut s = vec![0.0_f64; nat];
    let mut vd = vec![Vec3::zero(); nat];
    let mut vq = vec![[[0.0_f64; 3]; 3]; nat];
    for (a, (s_a, vd_a, vq_a)) in fields.into_iter().enumerate() {
        s[a] = s_a;
        vd[a] = vd_a;
        vq[a] = vq_a;
    }
    (s, vd, vq)
}

/// CAMM Fock/Pulay assembly for cumulative moments. `F_{ρσ} = ½(s_ρ+s_σ)S_{ρσ} +
/// ½(vd_ρ·R^{Aρ}_{ρσ} + vd_σ·R^{Aσ}_{σρ}) + ½(vq_ρ:Q̃^{Aρ}_{ρσ} + vq_σ:Q̃^{Aσ}_{σρ})`, with the
/// **bra-referenced** dipole/quadrupole integrals (`Q̃` traceless) — `∂(s·q + vd·μ + vq·Θ)/∂P`
/// for the cumulative `q,μ,Θ`. Distinct from [`assemble_shift`] (which uses on-site blocks).
fn camm_aes_shift(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    atom_pos: &[Vec3],
    s: &[f64],
    vd: &[Vec3],
    vq: &[[[f64; 3]; 3]],
) -> Matrix {
    let n = basis.len();
    let atom_of = ao_atom_index(basis, nat);
    let ov = &integrals.overlap;
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|rho| {
            let ar = atom_of[rho];
            let mut row = vec![0.0_f64; n];
            let ra = atom_pos[ar];
            for (sigma, slot) in row.iter_mut().enumerate() {
                let as_ = atom_of[sigma];
                // monopole channel (overlap ≈ 0 for far pairs, so this term self-screens)
                let mut f = 0.5 * (s[ar] + s[as_]) * ov[(rho, sigma)];
                // dipole + quadrupole channels are nonzero only within the integral cutoff.
                if (atom_pos[as_] - ra).norm2() <= CAMM_PAIR_CUTOFF_SQ {
                    let rd_rs = referenced_dipole(integrals, atom_pos, &atom_of, rho, sigma);
                    let rd_sr = referenced_dipole(integrals, atom_pos, &atom_of, sigma, rho);
                    f += 0.5 * (vd[ar].dot(rd_rs) + vd[as_].dot(rd_sr));
                    let rq_rs =
                        traceless(&referenced_quad(integrals, atom_pos, &atom_of, rho, sigma));
                    let rq_sr =
                        traceless(&referenced_quad(integrals, atom_pos, &atom_of, sigma, rho));
                    f += 0.5 * (dot2_full_mat(&vq[ar], &rq_rs) + dot2_full_mat(&vq[as_], &rq_sr));
                }
                *slot = f;
            }
            row
        })
        .collect();
    let mut out = Matrix::zeros(n, n);
    for (rho, row) in rows.iter().enumerate() {
        for (sigma, &v) in row.iter().enumerate() {
            out[(rho, sigma)] = v;
        }
    }
    out
}

/// **CAMM-on-mDFTB2 off-site AES correction** energy + Fock shift from given cumulative moments.
/// `E = s_onsite·E_onsite^mDFTB(μ,Θ) + s_AES·(E_qμ + E_qΘ + E_μμ)`; the Fock is the exact `∂E/∂P` for the
/// cumulative moments. The mDFTB off-site Ohno multipole is **not** included (it is disabled in
/// this mode — no double counting). `atomic_charges[A] = Δq_A` (the mixed monopole, code sign).
#[allow(clippy::too_many_arguments)]
pub fn camm_aes_energy_fock(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    moments: &AtomicMoments,
    atomic_charges: &[f64],
    kappa: &[f64],
    scale: f64,
    onsite_scale: &[f64],
) -> MultipoleEnergyFock {
    let (s, vd, vq) = camm_aes_potentials(
        nat,
        atom_hardness,
        atom_pos,
        &moments.dipole,
        &moments.quad,
        atomic_charges,
        kappa,
        scale,
        onsite_scale,
    );
    let energy = multipole_energy_terms(nat, moments, &s, &vd, &vq, atomic_charges);
    let fock = camm_aes_shift(basis, nat, integrals, atom_pos, &s, &vd, &vq);
    MultipoleEnergyFock { energy, fock }
}

/// Analytic **energy gradient** `∂E_camm/∂R` of the CAMM-on-mDFTB2 correction at the converged
/// density (the implicit density response is carried by the base band-energy-weighted-density
/// Pulay term, since the CAMM Fock is in the converged Fock). Two explicit pieces:
/// (1) the **off-site kernel force** `½ m^T (∂K/∂R) m` (erf-cloud `f^(mn)_grad`, the GFN2-AES set
///     `q–μ, q–Θ, μ–μ` only, scaled by `s_AES`); and
/// (2) the **cumulative-moment integral-derivative force** `Σ_A field_A·∂m_A/∂R` (`field = K m =
///     (s,vd,vq)`), where the cumulative `q,μ,Θ` depend on geometry through the two-center
///     dipole/quadrupole/overlap integrals (via [`crate::integrals::contracted_pair_with_derivatives`])
///     and the `R_A` reference shifts. The on-site penalty's geometry dependence is included
///     automatically through `vd`/`vq` (its kernel is position-free, so it adds no (1)).
#[allow(clippy::too_many_arguments)]
pub fn camm_aes_gradient(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
    kappa: &[f64],
    scale: f64,
    onsite_scale: &[f64],
) -> Vec<Vec3> {
    use crate::coulomb::exchange_sigma_pair;
    let moments = camm_atomic_moments(basis, nat, integrals, density, atom_pos);
    let dip = &moments.dipole;
    let quad = &moments.quad;
    let q = atomic_charges;
    let (s, vd, vq) =
        camm_aes_potentials(nat, atom_hardness, atom_pos, dip, quad, q, kappa, scale, onsite_scale);

    // (1) Off-site kernel force (erf-cloud, q–μ/q–Θ/μ–μ only, ×s_AES). `gforce(a,b)` = ∂/∂R_a of
    // the ordered-pair energy; grad_a = Σ_{b≠a} ½(gforce(a,b) − gforce(b,a)) (mirrors
    // `multipole_kernel_forces`, restricted to the GFN2-AES tensors).
    let gforce = |a: usize, b: usize| -> Vec3 {
        let sigma =
            (kappa[a] * kappa[b]).sqrt() * exchange_sigma_pair(atom_hardness[a], atom_hardness[b]);
        let x = atom_pos[a] - atom_pos[b];
        let df01 = f_mn_grad_cloud(x, sigma, 0, 1);
        let df02 = f_mn_grad_cloud(x, sigma, 0, 2);
        let df10 = f_mn_grad_cloud(x, sigma, 1, 0);
        let df11 = f_mn_grad_cloud(x, sigma, 1, 1);
        let df20 = f_mn_grad_cloud(x, sigma, 2, 0);
        let mut g = Vec3::zero();
        g += r2_vec(&df01, dip[b]) * q[a]; // q_A · μ_B
        g += r3_quad(&df02, &quad[b]) * q[a]; // q_A · Θ_B
        g += r2_vec(&df10, dip[a]) * q[b]; // μ_A · q_B
        g += r3_two_vec(&df11, dip[a], dip[b]); // μ_A · μ_B
        g += r3_quad(&df20, &quad[a]) * q[b]; // Θ_A · q_B
        g * scale
    };
    let mut grad: Vec<Vec3> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut ga = Vec3::zero();
            for b in 0..nat {
                if a == b {
                    continue;
                }
                ga += (gforce(a, b) - gforce(b, a)) * 0.5;
            }
            ga
        })
        .collect();

    // (2) Cumulative-moment integral-derivative force. Per AO pair (κ∈A, λ): the cumulative
    // q,μ,Θ on atom A = atom(κ) get contributions `P_κλ·(integral referenced to R_A)`. The
    // gradient `Σ field_A·∂m_A/∂R` is assembled from the two-center integral derivatives
    // (`da = ∂/∂R_A`, `db = ∂/∂R_B`) plus the explicit `∂(R_λ−R_A)` reference-shift terms.
    let atom_of = ao_atom_index(basis, nat);
    let n = basis.len();
    let pair_grads: Vec<(usize, usize, Vec3, Vec3)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|ka| {
            let a = atom_of[ka];
            let sa = s[a];
            let vda = vd[a];
            let vqa = traceless(&vq[a]);
            // contraction coefficients for vqA:Q over the 6 unique (xx,xy,yy,xz,yz,zz) components
            let cq = [
                vqa[0][0],
                2.0 * vqa[0][1],
                vqa[1][1],
                2.0 * vqa[0][2],
                2.0 * vqa[1][2],
                vqa[2][2],
            ];
            let pos_a = atom_pos[a];
            (0..n)
                .filter_map(|la| {
                    let p = density[(ka, la)];
                    if p == 0.0 {
                        return None;
                    }
                    let b = atom_of[la];
                    let pos_b = atom_pos[b];
                    if (pos_b - pos_a).norm2() > CAMM_PAIR_CUTOFF_SQ {
                        return None;
                    }
                    let (m10, da, db) = crate::integrals::contracted_pair_with_derivatives(
                        &basis.aos[ka],
                        &basis.aos[la],
                        pos_a,
                        pos_b,
                    );
                    let ss = m10[0];
                    let dd = Vec3::new(m10[1], m10[2], m10[3]);
                    let delta = pos_b - pos_a;
                    // vqA·δ and vqA·D (3-vectors), wδ scalar, vdA·δ scalar
                    let mul = |mat: &[[f64; 3]; 3], v: Vec3| -> Vec3 {
                        Vec3::new(
                            mat[0][0] * v.x + mat[0][1] * v.y + mat[0][2] * v.z,
                            mat[1][0] * v.x + mat[1][1] * v.y + mat[1][2] * v.z,
                            mat[2][0] * v.x + mat[2][1] * v.y + mat[2][2] * v.z,
                        )
                    };
                    let wv = mul(&vqa, delta);
                    let vqd = mul(&vqa, dd);
                    let wdelta = wv.dot(delta);
                    let vddelta = vda.dot(delta);
                    let mut g_a = Vec3::zero();
                    let mut g_b = Vec3::zero();
                    // q-channel: ∂(s_A q_A)/∂R, q_A = Σ P S
                    g_a += da[0] * (sa * p);
                    g_b += db[0] * (sa * p);
                    // μ-channel: E_μ = p( vdA·D + (vdA·δ)S )
                    g_a += (da[1] * vda.x + da[2] * vda.y + da[3] * vda.z) * p;
                    g_b += (db[1] * vda.x + db[2] * vda.y + db[3] * vda.z) * p;
                    g_a += (da[0] * vddelta - vda * ss) * p;
                    g_b += (db[0] * vddelta + vda * ss) * p;
                    // Θ-channel: E_Θ = p( vqA:Q + 2 wv·D + wδ S )
                    let dqa = da[4] * cq[0]
                        + da[5] * cq[1]
                        + da[6] * cq[2]
                        + da[7] * cq[3]
                        + da[8] * cq[4]
                        + da[9] * cq[5];
                    let dqb = db[4] * cq[0]
                        + db[5] * cq[1]
                        + db[6] * cq[2]
                        + db[7] * cq[3]
                        + db[8] * cq[4]
                        + db[9] * cq[5];
                    g_a += dqa * p;
                    g_b += dqb * p;
                    g_a += ((da[1] * wv.x + da[2] * wv.y + da[3] * wv.z) - vqd) * (2.0 * p);
                    g_b += ((db[1] * wv.x + db[2] * wv.y + db[3] * wv.z) + vqd) * (2.0 * p);
                    g_a += (da[0] * wdelta - wv * (2.0 * ss)) * p;
                    g_b += (db[0] * wdelta + wv * (2.0 * ss)) * p;
                    Some((a, b, g_a, g_b))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for (a, b, g_a, g_b) in pair_grads {
        grad[a] += g_a;
        grad[b] += g_b;
    }
    grad
}

/// Sum of the atomic dipole moments `Σ_A d_A`. Together with the Mulliken monopole dipole
/// `Σ_A q_A R_A` this is the physically complete molecular dipole once the mDFTB2 atomic
/// dipoles are switched on (Stage 3 field–multipole coupling).
pub fn total_atomic_dipole(moments: &AtomicMoments) -> Vec3 {
    let mut d = Vec3::zero();
    for di in &moments.dipole {
        d += *di;
    }
    d
}

/// **Stage 3 — first-order field–dipole coupling.** Under a uniform external electric field
/// `E`, the first-order energy (otherwise absorbed into `H0`) becomes
/// `E_field = −Σ_A [ q_A φ(R_A) + d_A·E + … ]`. The monopole part `−E·Σ_A q_A R_A` is already
/// handled by [`crate::field`]; this adds the **atomic dipole** part `E_field^dip = −E·Σ_A d_A`
/// (the quadrupole/octupole couple only to a field *gradient*, zero for a uniform field). The
/// field is external (constant), so the Fock shift is `∂E_field^dip/∂P = Σ_A (−E)·∂d_A/∂P` —
/// the same dipole shift bilinear as mDFTB2 with the constant "dipole potential" `vd_A = −E`
/// (no `½`, no monopole/quad channel, no on-site self-interaction). The result is a **constant
/// per geometry** (independent of the SCC moments), so build it once and add it to the
/// multipole Fock every iteration. Mirrors [`multipole_fock_from_moments`]'s Fock assembly.
pub fn field_dipole_fock(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    field: Vec3,
) -> Matrix {
    let n = basis.len();
    let vd = vec![-field; nat];
    let vq = vec![[[0.0_f64; 3]; 3]; nat];
    let s = vec![0.0_f64; nat];
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd, &vq);
    assemble_shift(n, &per_atom, &atom_of, &s, &dmat, &mmat, &integrals.overlap)
}

/// **Periodic / site-resolved dipole Fock** from a *per-atom* dipole potential `V_A = ∂E/∂d_A`.
/// Generalizes [`field_dipole_fock`] (a single *uniform* external field) to a site-dependent
/// potential — the SCF Fock route for the **periodic** dipole self-energy, where `V_A` is the
/// periodic Ewald field
/// [`crate::pbc::ewald_multipole::periodic_dipole_field_ko_pairwise`]. The on-site dipole AO
/// operator `⟨μ|(r−R_A)|ν⟩` (intra-atomic) is contracted against `V_A` through the same
/// `shift_matrices`/`assemble_shift` machinery the molecular generic Fock uses, so the result is
/// the exact `∂E/∂P` of `E = ½ Σ_A d_A·V_A` (variational). Being on-site (A=A, T=0) it carries a
/// trivial Bloch phase and is added identically to every `H(k)`. Sign: `vd_A = V_A` (**not**
/// negated) since `V_A` already equals `∂E/∂d_A`; contrast [`field_dipole_fock`], whose uniform
/// field carries `vd = −E` because there `E_field = −E·Σ_A d_A`.
pub fn periodic_dipole_fock(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    fields: &[Vec3],
) -> Matrix {
    let n = basis.len();
    let vq = vec![[[0.0_f64; 3]; 3]; nat];
    let s = vec![0.0_f64; nat];
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, fields, &vq);
    assemble_shift(n, &per_atom, &atom_of, &s, &dmat, &mmat, &integrals.overlap)
}

/// Field–dipole interaction energy `E_field^dip = −E·Σ_A d_A` from the atomic dipole moments
/// (full, no double-counting `½` — the potential is the *external* field, P-independent, exactly
/// like the monopole field energy `Σ_i q_i v_ext_i`). See [`field_dipole_fock`].
pub fn field_dipole_energy(field: Vec3, moments: &AtomicMoments) -> f64 {
    -field.dot(total_atomic_dipole(moments))
}

/// Overlap-Pulay weight `W = ∂E_field^dip/∂S` for the analytic gradient: the only explicit
/// position dependence of `E_field^dip = −E·Σ_A d_A` is through the on-site dipole moments'
/// overlap factor `½(SP+PS)` (the on-site dipole *integral* translates rigidly, and the
/// uniform field has no off-site kernel). The `dS/dR` machinery consumes this exactly like
/// [`multipole_overlap_weight`]. Mirrors that routine with `vd_A = −E`, `s = vQ = 0`.
pub fn field_dipole_overlap_weight(
    basis: &BasisSet,
    nat: usize,
    integrals: &IntegralMatrices,
    density: &Matrix,
    field: Vec3,
) -> Matrix {
    let n = basis.len();
    let vd = vec![-field; nat];
    let vq = vec![[[0.0_f64; 3]; 3]; nat];
    let s = vec![0.0_f64; nat];
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd, &vq);
    assemble_shift(n, &per_atom, &atom_of, &s, &dmat, &mmat, density)
}

/// **Third-order on-site multipole electrostatics.** Generalize the on-site monopole
/// third-order term `(1/3) Γ_A Δq_A³` so the third order also carries the *angular* density
/// fluctuations. The physical origin is the same charge expansion that produces the monopole
/// term: the on-site electrostatic hardness depends on the local charge. For the **multipole**
/// self-interactions, the on-site dipole–dipole and quadrupole–quadrupole "self-hardnesses"
/// `g_dd(η)=f^(1,1)_AA`, `g_QQ(η)=f^(2,2)_AA` are the `r→0` limits of the Klopman–Ohno kernel
/// and scale as `η³`, `η⁵`. With the breathing-radius charge-dependence
/// `η_A(q) = γ_A + 2Γ_A Δq_A` (the same `∂η/∂q = 2Γ` that yields GFN1's `(1/3)ΓΔq³`, with
/// `Γ = gam3`), Taylor-expanding the second-order multipole self-energy `½ g(η(q))(m·m)` to
/// first order in `Δq` gives the leading on-site cross terms
/// ```text
///   E³ = Σ_A [ α_A Δq_A (d_A·d_A) + β_A Δq_A (Q_A:Q_A) ],
///   α_A = g_dd'(γ_A) Γ_A = 3 (Γ_A/γ_A) g_dd(γ_A),   β_A = g_QQ'(γ_A) Γ_A = 5 (Γ_A/γ_A) g_QQ(γ_A)
/// ```
/// (the `½·2Γ` from the expansion cancels into the `g'(γ)Γ` coefficient; `g_dd' = 3g_dd/γ`,
/// `g_QQ' = 5g_QQ/γ` by the `η³`/`η⁵` homogeneity). This is **parameter-free** — `α,β` come
/// from the existing on-site kernels and the existing `γ_A`, `Γ_A` — and reduces to nothing for
/// a spherical atom (`d=Q=0`). All quantities use `g_dd(d·d) = d·r2_vec(f11,d)` and
/// `g_QQ(Q:Q) = Q:r4_quad(f22,Q)`, i.e. the on-site self-fields already built by the second-order
/// [`potentials_from_moments`]. `atomic_charges[A] = Δq_A` (the multipole monopole). The Fock is
/// the exact `∂E³/∂P` (FD-verified), via the on-site potentials
/// `s³_A = ∂E³/∂Δq`, `vd³_A = ∂E³/∂d`, `vq³_A = ∂E³/∂Q` and the shared shift machinery.
///
/// Reference: the third-order DFTB charge expansion — M. Gaus, Q. Cui, M. Elstner,
/// *J. Chem. Theory Comput.* **7**, 931 (2011); generalized to the atomic multipole moments
/// in the spirit of V.-Q. Vuong et al., *J. Chem. Theory Comput.* **19**, 7592 (2023).
#[allow(clippy::too_many_arguments)]
pub fn third_order_fock_from_moments(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    integrals: &IntegralMatrices,
    moments: &AtomicMoments,
    atomic_charges: &[f64],
) -> MultipoleEnergyFock {
    let n = basis.len();
    let (s3, vd3, vq3) = third_order_potentials(nat, atom_hardness, gam3, moments, atomic_charges);
    let energy = (0..nat).map(|a| atomic_charges[a] * s3[a]).sum();
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd3, &vq3);
    let fock = assemble_shift(
        n,
        &per_atom,
        &atom_of,
        &s3,
        &dmat,
        &mmat,
        &integrals.overlap,
    );
    MultipoleEnergyFock { energy, fock }
}

/// Third-order multipole correction **energy only** from given atomic moments (no Fock build),
/// for the SCC energy at the output density. See [`third_order_fock_from_moments`].
pub fn third_order_energy_from_moments(
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    moments: &AtomicMoments,
    atomic_charges: &[f64],
) -> f64 {
    let (s3, _vd3, _vq3) =
        third_order_potentials(nat, atom_hardness, gam3, moments, atomic_charges);
    (0..nat).map(|a| atomic_charges[a] * s3[a]).sum()
}

/// The third-order on-site potentials `s³ = ∂E³/∂Δq`, `vd³ = ∂E³/∂d`, `vq³ = ∂E³/∂Q` (the exact
/// partials that drive the Fock / overlap-Pulay weight). Shared by the SCC Fock and the
/// gradient. See [`third_order_fock_from_moments`].
#[allow(clippy::type_complexity)]
fn third_order_potentials(
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    moments: &AtomicMoments,
    atomic_charges: &[f64],
) -> (Vec<f64>, Vec<Vec3>, Vec<[[f64; 3]; 3]>) {
    let mut s3 = vec![0.0_f64; nat];
    let mut vd3 = vec![Vec3::zero(); nat];
    let mut vq3 = vec![[[0.0_f64; 3]; 3]; nat];
    for a in 0..nat {
        let gamma = atom_hardness[a];
        if gamma.abs() < 1.0e-12 {
            continue;
        }
        let ratio = gam3[a] / gamma; // Γ_A / γ_A
        let c = 1.0 / (gamma * gamma);
        let x = Vec3::new(1.0e-6, 0.0, 0.0);
        let f11 = f_mn(x, c, 1, 1);
        let f22 = f_mn(x, c, 2, 2);
        let vd_self = r2_vec(&f11, moments.dipole[a]); // g_dd · d
        let vq_self = r4_quad(&f22, &moments.quad[a]); // g_QQ · Q
        let dd_e = moments.dipole[a].dot(vd_self); // g_dd (d·d)
        let qq_e = dot2_full_mat(&moments.quad[a], &vq_self); // g_QQ (Q:Q)
        let dq = atomic_charges[a];
        // s³ = ∂E³/∂Δq = α(d·d) + β(Q:Q), with α = 3(Γ/γ)g_dd, β = 5(Γ/γ)g_QQ.
        s3[a] = 3.0 * ratio * dd_e + 5.0 * ratio * qq_e;
        // vd³ = ∂E³/∂d = 2αΔq·d = 6(Γ/γ)Δq·g_dd·d = 6(Γ/γ)Δq·vd_self.
        vd3[a] = vd_self * (6.0 * ratio * dq);
        // vq³ = ∂E³/∂Q = 2βΔq·Q = 10(Γ/γ)Δq·g_QQ·Q = 10(Γ/γ)Δq·vq_self.
        for i in 0..3 {
            for j in 0..3 {
                vq3[a][i][j] = vq_self[i][j] * (10.0 * ratio * dq);
            }
        }
    }
    (s3, vd3, vq3)
}

/// **Generic on-site multipole×charge cross terms** — arbitrary rank `l` × arbitrary charge order.
/// Generalises [`third_order_potentials`] (rank 1/2, charge order 3 only) to the full Taylor
/// expansion of the on-site multipole self-energy `½ g_l(η_A(q))(m_l·m_l)` in the breathing-radius
/// charge dependence `η_A(q)=γ_A+2Γ_A Δq_A` (using `g_l ∝ η^{2l+1}`):
/// ```text
///   E = Σ_A Σ_{l≥1} Σ_{k≥1} c_{l,k} · ½ g_l(γ_A)(m_l·m_l) · Δq_A^k,
///   c_{l,k} = (1/k!)(2Γ/γ)^k Π_{j=0}^{k-1}(2l+1−j)   [recurrence  c_{l,k}=c_{l,k-1}(2Γ/γ)(2l+2−k)/k]
/// ```
/// `max_order_per_rank[l−1]` = max charge order (≥3) coupled to rank `l` (`<3` ⇒ no cross term); `k`
/// runs `1..=order−2` and naturally terminates at `k=2l+1` (the `g_l` polynomial degree, where the
/// recurrence hits 0). `k=1, l∈{1,2}` reproduces [`third_order_potentials`] exactly. Parameter-free
/// (existing `γ_A`, `Γ_A=gam3`, on-site kernels `f^{(l,l)}`). Returns `(energy, v)` with `v[a]` in the
/// generic field layout (`v[a][0][0]=∂E/∂Δq_a`, `v[a][l]=∂E/∂m_l`) for [`generic_shift_inputs`].
pub fn multipole_charge_cross_fields(
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    moments: &[Vec<Vec<f64>>],
    atomic_charges: &[f64],
    max_order_per_rank: &[usize],
    max_rank: usize,
) -> (f64, Vec<Vec<Vec<f64>>>) {
    let mut v: Vec<Vec<Vec<f64>>> = moments
        .iter()
        .map(|m| m.iter().map(|t| vec![0.0; t.len()]).collect())
        .collect();
    let mut energy = 0.0;
    for a in 0..nat {
        let gamma = atom_hardness[a];
        if gamma.abs() < 1.0e-12 || a >= v.len() {
            continue;
        }
        let c = 1.0 / (gamma * gamma);
        let ratio2 = 2.0 * gam3[a] / gamma; // 2Γ/γ
        let dq = atomic_charges[a];
        let x = Vec3::new(1.0e-6, 0.0, 0.0);
        let mut s_a = 0.0;
        for l in 1..=max_rank {
            let max_order = max_order_per_rank.get(l - 1).copied().unwrap_or(2);
            if max_order < 3 || l >= moments[a].len() {
                continue;
            }
            let m_l = &moments[a][l];
            if m_l.iter().all(|&x| x == 0.0) {
                continue;
            }
            let self_field = contract_last(&f_mn(x, c, l, l), l, l, m_l); // g_l · m_l
            let mm = dot_flat(m_l, &self_field); // g_l (m·m)
            let mut c_lk = 1.0; // c_{l,0}
            let mut s_factor = 0.0; // Σ_k c_{l,k} k Δq^{k−1}
            let mut v_factor = 0.0; // Σ_k c_{l,k} Δq^k
            for k in 1..=(max_order - 2) {
                c_lk *= ratio2 * (2 * l + 2 - k) as f64 / k as f64;
                if c_lk == 0.0 {
                    break;
                }
                s_factor += c_lk * k as f64 * dq.powi((k - 1) as i32);
                v_factor += c_lk * dq.powi(k as i32);
            }
            s_a += 0.5 * mm * s_factor;
            energy += 0.5 * mm * v_factor;
            for (vi, &sf) in v[a][l].iter_mut().zip(self_field.iter()) {
                *vi += sf * v_factor;
            }
        }
        if !v[a].is_empty() && !v[a][0].is_empty() {
            v[a][0][0] = s_a;
        }
    }
    (energy, v)
}

/// **Combined** generic multipole Fock + per-rank charge-cross Fock sharing a SINGLE
/// `generic_shift_inputs`/`assemble_shift` pass (perf). The standard and cross on-site field
/// operators are both linear in the overlap shift, so their fields are summed *before* the
/// rank-summed, `detrace`-heavy O(N²) shift assembly — the cross terms then cost no extra overlap
/// pass (vs. separate [`multipole_fock_generic`] + cross calls, which would run that loop twice).
/// `energy` = the standard `½ Σ_A Σ_l M·V` multipole energy ONLY; the cross-term energy is the
/// higher-order polynomial from [`multipole_charge_cross_energy`], added at the output density.
/// An empty `max_order_per_rank` ⇒ byte-identical to [`multipole_fock_generic`] (cross fully gated).
#[allow(clippy::too_many_arguments)]
pub fn multipole_fock_generic_with_cross(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    moments: &[Vec<Vec<f64>>],
    atomic_charges: &[f64],
    max_order_per_rank: &[usize],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> MultipoleEnergyFock {
    let n = basis.len();
    let mut v = multipole_fields_generic(nat, atom_hardness, atom_pos, moments, max_rank);
    // Standard ½ Σ_A Σ_l M·V energy — from the standard fields, *before* adding the cross fields.
    let mut energy = 0.0;
    for a in 0..nat {
        let mut e_a = 0.0;
        for l in 0..=max_rank {
            e_a += dot_flat(&moments[a][l], &v[a][l]);
        }
        energy += 0.5 * e_a;
    }
    if !max_order_per_rank.is_empty() {
        let (_e_cross, vc) = multipole_charge_cross_fields(
            nat,
            atom_hardness,
            gam3,
            moments,
            atomic_charges,
            max_order_per_rank,
            max_rank,
        );
        add_generic_fields(&mut v, &vc);
    }
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, &v, max_rank, cache);
    let fock = assemble_shift(
        n,
        &per_atom,
        &atom_of,
        &s_field,
        &dmat,
        &mmat,
        &integrals.overlap,
    );
    MultipoleEnergyFock { energy, fock }
}

/// Accumulate one generic field layout into another in place (`dst += src`), used to fuse the
/// standard multipole and charge-cross on-site fields before a single shift assembly.
fn add_generic_fields(dst: &mut [Vec<Vec<f64>>], src: &[Vec<Vec<f64>>]) {
    for (da, sa) in dst.iter_mut().zip(src.iter()) {
        for (dl, sl) in da.iter_mut().zip(sa.iter()) {
            for (x, y) in dl.iter_mut().zip(sl.iter()) {
                *x += *y;
            }
        }
    }
}

/// Energy-only of the generic multipole×charge cross terms (SCC energy at the output density).
pub fn multipole_charge_cross_energy(
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    moments: &[Vec<Vec<f64>>],
    atomic_charges: &[f64],
    max_order_per_rank: &[usize],
    max_rank: usize,
) -> f64 {
    multipole_charge_cross_fields(
        nat,
        atom_hardness,
        gam3,
        moments,
        atomic_charges,
        max_order_per_rank,
        max_rank,
    )
    .0
}

/// **Combined** generic multipole + charge-cross overlap-Pulay weight `W = ∂E/∂S`, sharing a single
/// shift assembly (the gradient counterpart of [`multipole_fock_generic_with_cross`]; `S→P`). An
/// empty `max_order_per_rank` ⇒ byte-identical to [`multipole_overlap_weight_generic`].
#[allow(clippy::too_many_arguments)]
pub fn multipole_overlap_weight_generic_with_cross(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    atom_pos: &[Vec3],
    density: &Matrix,
    moments: &[Vec<Vec<f64>>],
    atomic_charges: &[f64],
    max_order_per_rank: &[usize],
    max_rank: usize,
    cache: Option<&OnsiteMomentCache>,
) -> Matrix {
    let n = basis.len();
    let mut v = multipole_fields_generic(nat, atom_hardness, atom_pos, moments, max_rank);
    if !max_order_per_rank.is_empty() {
        let (_e_cross, vc) = multipole_charge_cross_fields(
            nat,
            atom_hardness,
            gam3,
            moments,
            atomic_charges,
            max_order_per_rank,
            max_rank,
        );
        add_generic_fields(&mut v, &vc);
    }
    let (s_field, dmat, mmat, per_atom, atom_of) =
        generic_shift_inputs(basis, nat, atom_pos, &v, max_rank, cache);
    assemble_shift(n, &per_atom, &atom_of, &s_field, &dmat, &mmat, density)
}

/// The third-order overlap-Pulay weight `W = ∂E³/∂S` for the analytic gradient (the
/// `assemble_shift` bilinear with `S→P`). The on-site `f^(1,1)_AA`/`f^(2,2)_AA` are
/// position-independent and the on-site moment integrals translate rigidly, so — like the
/// second-order multipole term — the only explicit derivative is this overlap term (no off-site
/// kernel-force). `atomic_charges[A] = Δq_A`.
#[allow(clippy::too_many_arguments)]
pub fn third_order_overlap_weight(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    gam3: &[f64],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
) -> Matrix {
    let n = basis.len();
    let moments = atomic_moments(basis, nat, integrals, density);
    let (s3, vd3, vq3) = third_order_potentials(nat, atom_hardness, gam3, &moments, atomic_charges);
    let (dmat, mmat, per_atom, atom_of) = shift_matrices(basis, nat, integrals, &vd3, &vq3);
    assemble_shift(n, &per_atom, &atom_of, &s3, &dmat, &mmat, density)
}

/// The mDFTB2 **kernel** contribution to the analytic gradient: the explicit position
/// derivatives `df^(mn)_AB/dR` of the off-site interaction tensors, contracted with the
/// (fixed) atomic moment pairs. The on-site `f^(11)_AA`, `f^(22)_AA` are position-
/// independent and the on-site `d̄`/`Q̄` translate rigidly, so neither contributes here
/// (those enter only the overlap-Pulay term). Returns the per-atom force-gradient
/// `Σ_B ½ ∂_x (moment_A · f^(mn)_AB · moment_B)` summed over ordered pairs.
pub fn multipole_kernel_forces(
    basis: &BasisSet,
    nat: usize,
    atom_hardness: &[f64],
    atom_pos: &[Vec3],
    integrals: &IntegralMatrices,
    density: &Matrix,
    atomic_charges: &[f64],
) -> Vec<Vec3> {
    use crate::coulomb::harmonic_average;
    let q = atomic_charges;
    let (moments, _s, _vd, _vq) =
        moments_and_potentials(basis, nat, atom_hardness, atom_pos, integrals, density, q);
    let dip = &moments.dipole;
    let quad = &moments.quad;
    // Force from the *ordered* pair (a,b) on atom a: ∂/∂R_a of ½ (moment_a · f^(mn)_ab · moment_b).
    let gforce = |a: usize, b: usize| -> Vec3 {
        let eta = harmonic_average(atom_hardness[a], atom_hardness[b]);
        let c = 1.0 / (eta * eta);
        let x = atom_pos[a] - atom_pos[b];
        // ∂_x f^(mn): one rank higher than f^(mn) (first index = gradient direction).
        let df01 = f_mn_grad(x, c, 0, 1); // rank 2
        let df02 = f_mn_grad(x, c, 0, 2); // rank 3
        let df10 = f_mn_grad(x, c, 1, 0); // rank 2
        let df11 = f_mn_grad(x, c, 1, 1); // rank 3
        let df12 = f_mn_grad(x, c, 1, 2); // rank 4
        let df20 = f_mn_grad(x, c, 2, 0); // rank 3
        let df21 = f_mn_grad(x, c, 2, 1); // rank 4
        let df22 = f_mn_grad(x, c, 2, 2); // rank 5
        let mut g = Vec3::zero();
        g += r2_vec(&df01, dip[b]) * q[a]; // q_A · d_B
        g += r3_quad(&df02, &quad[b]) * q[a]; // q_A · Q_B
        g += r2_vec(&df10, dip[a]) * q[b]; // d_A · q_B
        g += r3_two_vec(&df11, dip[a], dip[b]); // d_A · d_B
        g += r4_vec_quad(&df12, dip[a], &quad[b]); // d_A · Q_B
        g += r3_quad(&df20, &quad[a]) * q[b]; // Q_A · q_B
        g += r4_quad_vec(&df21, &quad[a], dip[b]); // Q_A · d_B
        g += r5_quad_quad(&df22, &quad[a], &quad[b]); // Q_A · Q_B
        g
    };
    // The serial loop adds `+½ g(a,b)` to `grad[a]` and `-½ g(a,b)` to `grad[b]` over all ordered
    // pairs, so `grad[a] = Σ_{b≠a} ½ (g(a,b) − g(b,a))`. Computing each atom's row independently
    // makes it embarrassingly parallel (each ordered kernel is recomputed once per endpoint —
    // 2× the `f^(mn)_grad` work, amortized across cores; value-identical up to ~1 ULP).
    (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut ga = Vec3::zero();
            for b in 0..nat {
                if a == b {
                    continue;
                }
                ga += (gforce(a, b) - gforce(b, a)) * 0.5;
            }
            ga
        })
        .collect()
}

#[inline]
fn f10_vec(f: &[f64], scalar: f64) -> Vec3 {
    Vec3::new(f[0] * scalar, f[1] * scalar, f[2] * scalar)
}

#[inline]
fn add_quads(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] + b[i][j];
        }
    }
    o
}

#[inline]
fn traceless(q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let tr = (q[0][0] + q[1][1] + q[2][2]) / 3.0;
    let mut o = *q;
    for i in 0..3 {
        o[i][i] -= tr;
    }
    o
}

#[inline]
fn dot2_full_mat(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma(x: Vec3, c: f64) -> f64 {
        1.0 / (x.norm2() + c).sqrt()
    }

    /// The analytic Cartesian gradient tensors `T^(k) = ∇^k γ` must match repeated
    /// central finite differences of `γ` (the parameter-free `f^(mn)` foundation).
    #[test]
    fn grad_tensors_match_finite_difference() {
        let x0 = Vec3::new(0.7, -0.4, 0.9);
        let c = 1.0 / (0.5_f64 * 0.5); // c = 1/η^2, η = 0.5
        let h = 1.0e-3;
        let unit = |a: usize| match a {
            0 => Vec3::new(h, 0.0, 0.0),
            1 => Vec3::new(0.0, h, 0.0),
            _ => Vec3::new(0.0, 0.0, h),
        };
        // k = 1: ∇_i γ via central difference.
        let g = radial_derivs(x0.norm2(), c, 5);
        let t1 = grad_tensor(x0, &g, 1);
        for a in 0..3 {
            let fd = (gamma(x0 + unit(a), c) - gamma(x0 - unit(a), c)) / (2.0 * h);
            assert!(
                (t1[a] - fd).abs() < 1.0e-6,
                "T1[{a}] {} vs fd {}",
                t1[a],
                fd
            );
        }
        // k = 2: ∇_i∇_j γ via FD of the analytic T1 (one order down).
        let t2 = grad_tensor(x0, &g, 2);
        for a in 0..3 {
            for b in 0..3 {
                let gp = radial_derivs((x0 + unit(b)).norm2(), c, 1);
                let gm = radial_derivs((x0 - unit(b)).norm2(), c, 1);
                let fd = (grad_tensor(x0 + unit(b), &gp, 1)[a]
                    - grad_tensor(x0 - unit(b), &gm, 1)[a])
                    / (2.0 * h);
                assert!((t2[a * 3 + b] - fd).abs() < 1.0e-5, "T2 mismatch");
            }
        }
        // k = 3, 4, 5: FD of the analytic order-(k-1) tensor.
        for k in 3..=5 {
            let tk = grad_tensor(x0, &g, k);
            let lower = 3usize.pow((k - 1) as u32);
            let mut maxdiff = 0.0_f64;
            for low in 0..lower {
                for b in 0..3 {
                    let gp = radial_derivs((x0 + unit(b)).norm2(), c, k - 1);
                    let gm = radial_derivs((x0 - unit(b)).norm2(), c, k - 1);
                    let fd = (grad_tensor(x0 + unit(b), &gp, k - 1)[low]
                        - grad_tensor(x0 - unit(b), &gm, k - 1)[low])
                        / (2.0 * h);
                    let ana = tk[low * 3 + b];
                    maxdiff = maxdiff.max((ana - fd).abs());
                }
            }
            assert!(maxdiff < 1.0e-3, "T{k} max FD mismatch {maxdiff:.3e}");
        }
    }

    /// v0.2.1: the generic per-rank multipole×charge cross-term fields, restricted to charge order
    /// 3 on dipole+quadrupole (`[3, 3]` ⇒ the single `k=1` Taylor term for ranks 1,2), must
    /// reproduce the legacy [`third_order_potentials`] exactly — both the total energy and the
    /// on-site charge potential `s = ∂E/∂Δq` agree to machine precision. Pins the cross-term
    /// recurrence `c_{l,k}=c_{l,k-1}(2Γ/γ)(2l+2−k)/k` to the validated third-order form.
    #[test]
    fn multipole_charge_order_3_equals_legacy_third_order() {
        let nat = 2;
        let hardness = vec![0.45, 0.62];
        let gam3 = vec![0.10, 0.07];
        let charges = vec![-0.33, 0.21]; // Δq
        let d = [Vec3::new(0.12, -0.20, 0.31), Vec3::new(-0.05, 0.18, 0.09)];
        let make_q = |a: f64, b: f64| {
            let mut q = [[0.0_f64; 3]; 3];
            q[0][0] = a;
            q[1][1] = b;
            q[2][2] = -(a + b); // traceless
            q[0][1] = 0.07;
            q[1][0] = 0.07;
            q[0][2] = -0.04;
            q[2][0] = -0.04;
            q[1][2] = 0.11;
            q[2][1] = 0.11;
            q
        };
        let qmat = [make_q(0.15, -0.05), make_q(-0.08, 0.12)];
        let am = AtomicMoments {
            dipole: d.to_vec(),
            quad: qmat.to_vec(),
        };
        let (s3, _vd3, _vq3) = third_order_potentials(nat, &hardness, &gam3, &am, &charges);
        let e_third: f64 = (0..nat).map(|a| charges[a] * s3[a]).sum();

        // Generic moments: [ [q], dipole(3), quad(9 row-major) ] per atom.
        let moments: Vec<Vec<Vec<f64>>> = (0..nat)
            .map(|a| {
                let qf: Vec<f64> = (0..3)
                    .flat_map(|i| (0..3).map(move |j| qmat[a][i][j]))
                    .collect();
                vec![vec![charges[a]], vec![d[a].x, d[a].y, d[a].z], qf]
            })
            .collect();
        let (e_cross, v) =
            multipole_charge_cross_fields(nat, &hardness, &gam3, &moments, &charges, &[3, 3], 2);

        assert!(
            (e_cross - e_third).abs() < 1.0e-12,
            "cross [3,3] energy {e_cross:.15} vs legacy third-order {e_third:.15}"
        );
        for a in 0..nat {
            assert!(
                (v[a][0][0] - s3[a]).abs() < 1.0e-12,
                "atom {a}: cross s {} vs third-order s3 {}",
                v[a][0][0],
                s3[a]
            );
        }
    }

    /// v0.2.1: a rank-`l` cross-term contribution terminates at charge order `2l+3` (the recurrence
    /// factor `(2l+2−k)` reaches zero), so any order above the bound adds exactly nothing. The
    /// dipole (`l=1`, bound 5) therefore yields identical fields for `order=5` and `order=9`.
    #[test]
    fn multipole_charge_cross_terminates_at_bound() {
        let nat = 1;
        let hardness = vec![0.5];
        let gam3 = vec![0.12];
        let charges = vec![0.4];
        let moments = vec![vec![vec![0.4], vec![0.2, -0.1, 0.3], vec![0.0; 9]]];
        let (e5, _) =
            multipole_charge_cross_fields(nat, &hardness, &gam3, &moments, &charges, &[5], 2);
        let (e9, _) =
            multipole_charge_cross_fields(nat, &hardness, &gam3, &moments, &charges, &[9], 2);
        assert!(
            (e5 - e9).abs() < 1.0e-14,
            "dipole cross energy must terminate at order 5: e5={e5:.15} e9={e9:.15}"
        );
    }

    /// v0.2.0 arbitrary-rank gate: `radial_derivs`/`grad_tensor`/`f_mn` now extend **past the old
    /// rank-7 cap** (G0..G7). Verify rank-8 (`f^(4,4)`, hexadecapole–hexadecapole) is finite and
    /// correctly sized, and spot-check a few components of `grad_tensor(8)` against a finite
    /// difference of the analytic `grad_tensor(7)` (cheap — only a few components, not all 3^8).
    #[test]
    fn grad_tensor_extends_past_octupole_cap() {
        let x0 = Vec3::new(0.6, -0.5, 0.8);
        let c = 1.0 / (0.45_f64 * 0.45);
        // f^(4,4): rank-8 interaction tensor (3^8 = 6561 entries), all finite.
        let f44 = f_mn(x0, c, 4, 4);
        assert_eq!(f44.len(), 3usize.pow(8));
        assert!(f44.iter().all(|v| v.is_finite()) && f44.iter().any(|&v| v != 0.0));
        // Spot-check grad_tensor(8) vs FD of grad_tensor(7) for a handful of components.
        let g = radial_derivs(x0.norm2(), c, 9);
        let t8 = grad_tensor(x0, &g, 8);
        let h = 1.0e-3;
        let unit = |a: usize| match a {
            0 => Vec3::new(h, 0.0, 0.0),
            1 => Vec3::new(0.0, h, 0.0),
            _ => Vec3::new(0.0, 0.0, h),
        };
        let n7 = 3usize.pow(7);
        for &low in &[0usize, 137, 1000, n7 - 1] {
            for b in 0..3 {
                let gp = radial_derivs((x0 + unit(b)).norm2(), c, 7);
                let gm = radial_derivs((x0 - unit(b)).norm2(), c, 7);
                let fd = (grad_tensor(x0 + unit(b), &gp, 7)[low]
                    - grad_tensor(x0 - unit(b), &gm, 7)[low])
                    / (2.0 * h);
                let ana = t8[low * 3 + b];
                assert!(
                    (ana - fd).abs() < 1.0e-3 * (1.0 + ana.abs()),
                    "rank-8 component (low={low},b={b}): analytic {ana:.4e} vs FD {fd:.4e}"
                );
            }
        }
    }

    /// v0.2.0 arbitrary-rank gate: the general `detrace_symmetric` must reproduce the specialized
    /// `traceless` (rank 2) and `detrace_octupole` (rank 3) exactly, and produce a genuinely
    /// trace-free rank-4 tensor (all single traces vanish).
    #[test]
    fn detrace_symmetric_matches_low_ranks_and_is_traceless() {
        // rank 2 vs traceless.
        let vals = [1.3, -0.4, 0.7, -0.4, 2.1, 0.5, 0.7, 0.5, -0.9];
        let mut q = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                q[i][j] = vals[i * 3 + j];
            }
        }
        let flat2: Vec<f64> = (0..9).map(|f| q[f / 3][f % 3]).collect();
        let d2 = detrace_symmetric(&flat2, 2);
        let tl = traceless(&q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (d2[i * 3 + j] - tl[i][j]).abs() < 1.0e-13,
                    "rank-2 ({i},{j})"
                );
            }
        }
        // rank 3 vs detrace_octupole.
        let comps = [0.6, -0.2, 0.3, 0.1, -0.15, 0.25, 0.4, -0.05, 0.2, -0.35];
        let o = octu_from_components(&comps);
        let d3 = detrace_symmetric(&octu_flat(&o), 3);
        let ref3 = octu_flat(&detrace_octupole(&o));
        for k in 0..27 {
            assert!(
                (d3[k] - ref3[k]).abs() < 1.0e-13,
                "rank-3 comp {k}: {} vs {}",
                d3[k],
                ref3[k]
            );
        }
        // rank 4: build a fully symmetric tensor (sum of two rank-1 outer products, which is
        // symmetric under every index permutation), detrace, check all single traces
        // Σ_m T_{mmkl} vanish (genuinely trace-free).
        let v = [0.7_f64, -0.5, 0.9];
        let w = [0.2_f64, 1.1, -0.3];
        let mut s4 = vec![0.0_f64; 81];
        for (f, slot) in s4.iter_mut().enumerate() {
            let (i, j, k, l) = (f / 27, (f / 9) % 3, (f / 3) % 3, f % 3);
            *slot = v[i] * v[j] * v[k] * v[l] + 0.4 * w[i] * w[j] * w[k] * w[l];
        }
        let d4 = detrace_symmetric(&s4, 4);
        let mut max_trace = 0.0_f64;
        for k in 0..3 {
            for l in 0..3 {
                let mut tr = 0.0;
                for m in 0..3 {
                    tr += d4[((m * 3 + m) * 3 + k) * 3 + l];
                }
                max_trace = max_trace.max(tr.abs());
            }
        }
        assert!(
            max_trace < 1.0e-12,
            "rank-4 single trace should vanish, got {max_trace:.2e}"
        );
    }

    fn load_params() -> Option<crate::params::Gfn1Parameters> {
        let path = std::env::var("GFN1_XTB_PARAM").ok()?;
        crate::params::Gfn1Parameters::from_file(path).ok()
    }

    /// A free closed-shell atom has a spherical density, so its mDFTB atomic dipole and
    /// traceless quadrupole must vanish.
    #[test]
    fn free_atom_moments_vanish() {
        let Some(params) = load_params() else {
            return;
        };
        let system =
            crate::system::PeriodicSystem::from_xyz_str("1\nNe\nNe 0.0 0.0 0.0\n", 0.0, false)
                .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let result = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let m = atomic_moments(&basis, 1, &ints, &result.density);
        let d = m.dipole[0];
        let dmax = d.x.abs().max(d.y.abs()).max(d.z.abs());
        assert!(dmax < 1.0e-8, "free Ne atomic dipole {dmax:.3e} != 0");
        let qmax = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .fold(0.0_f64, |mx, (i, j)| mx.max(m.quad[0][i][j].abs()));
        assert!(
            qmax < 1.0e-8,
            "free Ne traceless quadrupole {qmax:.3e} != 0"
        );
    }

    /// Stage 2c gate: the on-site atomic octupole moments must be fully symmetric under
    /// index permutation and traceless (`Σ_m O_mmk = 0`), and nonzero (active). NOTE: the
    /// traceless octupole from s-p pairs is pure trace (vanishes), so a minimal s,p basis
    /// (light elements) has zero traceless octupole — it requires **d functions** (p-d
    /// pairs). H2S is used (S carries d in GFN1) so the moment is genuinely nonzero.
    #[test]
    fn octupole_moments_symmetric_and_traceless() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let result = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let positions: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let octu = atomic_octupole_moments(
            &basis,
            system.atoms.len(),
            &positions,
            &ints,
            &result.density,
            None,
        );
        let (mut maxmag, mut maxsym, mut maxtr) = (0.0_f64, 0.0_f64, 0.0_f64);
        for o in &octu {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        maxmag = maxmag.max(o[i][j][k].abs());
                        maxsym = maxsym
                            .max((o[i][j][k] - o[j][i][k]).abs())
                            .max((o[i][j][k] - o[i][k][j]).abs())
                            .max((o[i][j][k] - o[k][j][i]).abs());
                    }
                }
            }
            for k in 0..3 {
                let t: f64 = (0..3).map(|m| o[m][m][k]).sum();
                maxtr = maxtr.max(t.abs());
            }
        }
        assert!(maxmag > 1.0e-6, "octupole moments all ~zero ({maxmag:.3e})");
        assert!(
            maxsym < 1.0e-12,
            "octupole not index-symmetric: {maxsym:.3e}"
        );
        assert!(maxtr < 1.0e-12, "octupole not traceless: {maxtr:.3e}");
    }

    /// Arbitrary-rank gate: the generic `atomic_moment_rank_l` must reproduce the legacy
    /// quadrupole (`l=2`, [`atomic_moments`]) and octupole (`l=3`, [`atomic_octupole_moments`])
    /// on-site atomic moments. H2S (S carries d in GFN1) gives a genuinely nonzero rank-3 moment.
    #[test]
    fn atomic_moment_rank_l_matches_legacy_quad_and_octupole() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let result = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let nat = system.atoms.len();
        let positions: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();

        // l=2 vs legacy traceless quadrupole.
        let legacy_q = atomic_moments(&basis, nat, &ints, &result.density);
        let gen_q = atomic_moment_rank_l(&basis, nat, &positions, &ints, &result.density, 2, None);
        let mut max2 = 0.0_f64;
        for a in 0..nat {
            for i in 0..3 {
                for j in 0..3 {
                    max2 = max2.max((gen_q[a][i * 3 + j] - legacy_q.quad[a][i][j]).abs());
                }
            }
        }
        assert!(
            max2 < 1.0e-10,
            "rank-2 generic vs legacy quad mismatch: {max2:.3e}"
        );

        // l=3 vs legacy traceless octupole.
        let legacy_o =
            atomic_octupole_moments(&basis, nat, &positions, &ints, &result.density, None);
        let gen_o = atomic_moment_rank_l(&basis, nat, &positions, &ints, &result.density, 3, None);
        let mut max3 = 0.0_f64;
        for a in 0..nat {
            let lf = octu_flat(&legacy_o[a]);
            for (k, &lv) in lf.iter().enumerate() {
                max3 = max3.max((gen_o[a][k] - lv).abs());
            }
        }
        assert!(
            max3 < 1.0e-10,
            "rank-3 generic vs legacy octupole mismatch: {max3:.3e}"
        );
    }

    /// Arbitrary-rank gate: the generic `multipole_fields_generic` (rank loop) must reproduce the
    /// legacy `potentials_from_moments` (monopole/dipole/quadrupole) for `max_rank=2`. Pure
    /// function of synthetic moments + geometry — no SCC/params needed.
    #[test]
    fn multipole_fields_generic_matches_legacy_rank2() {
        let nat = 3;
        let hardness = [0.45_f64, 0.30, 0.55];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.4, 0.2, -0.3),
            Vec3::new(-0.5, 1.1, 0.7),
        ];
        let q = [0.2_f64, -0.1, -0.1];
        let dip = [
            Vec3::new(0.10, -0.20, 0.05),
            Vec3::new(-0.05, 0.15, 0.20),
            Vec3::new(0.00, 0.10, -0.10),
        ];
        let quad = [
            [
                [0.30, 0.05, -0.10],
                [0.05, -0.15, 0.02],
                [-0.10, 0.02, -0.15],
            ],
            [
                [-0.20, 0.08, 0.03],
                [0.08, 0.10, -0.04],
                [0.03, -0.04, 0.10],
            ],
            [
                [0.12, -0.06, 0.01],
                [-0.06, -0.05, 0.07],
                [0.01, 0.07, -0.07],
            ],
        ];
        let (s, vd, vq) = potentials_from_moments(nat, &hardness, &pos, &dip, &quad, &q);

        let moments: Vec<Vec<Vec<f64>>> = (0..nat)
            .map(|a| {
                vec![
                    vec![q[a]],
                    vec![dip[a].x, dip[a].y, dip[a].z],
                    quad_flat(&quad[a]).to_vec(),
                ]
            })
            .collect();
        let v = multipole_fields_generic(nat, &hardness, &pos, &moments, 2);

        let mut maxd = 0.0_f64;
        for a in 0..nat {
            maxd = maxd.max((v[a][0][0] - s[a]).abs());
            let vda = [vd[a].x, vd[a].y, vd[a].z];
            for k in 0..3 {
                maxd = maxd.max((v[a][1][k] - vda[k]).abs());
            }
            for i in 0..3 {
                for j in 0..3 {
                    maxd = maxd.max((v[a][2][i * 3 + j] - vq[a][i][j]).abs());
                }
            }
        }
        assert!(
            maxd < 1.0e-10,
            "generic fields vs legacy rank-2 mismatch: {maxd:.3e}"
        );
    }

    /// The general flat-tensor contraction `contract_last` must reproduce every hand-typed
    /// helper used in the multipole energy, so it can safely carry the octupole terms.
    #[test]
    fn general_contraction_matches_typed_helpers() {
        let x = Vec3::new(0.7, -0.4, 0.9);
        let c = 1.0 / (0.6_f64 * 0.6);
        let d = Vec3::new(0.3, -0.2, 0.5);
        let q = [[0.4, 0.1, -0.2], [0.1, -0.3, 0.05], [-0.2, 0.05, -0.1]];
        let dflat = [d.x, d.y, d.z];
        let qflat = quad_flat(&q);
        let f01 = f_mn(x, c, 0, 1);
        assert!((contract_last(&f01, 0, 1, &dflat)[0] - dot1(&f01, d)).abs() < 1e-13);
        let f02 = f_mn(x, c, 0, 2);
        assert!((contract_last(&f02, 0, 2, &qflat)[0] - dot2_full(&f02, &q)).abs() < 1e-13);
        let f11 = f_mn(x, c, 1, 1);
        let cl = contract_last(&f11, 1, 1, &dflat);
        let rv = r2_vec(&f11, d);
        assert!((cl[0] - rv.x).abs() < 1e-13);
        assert!((cl[1] - rv.y).abs() < 1e-13);
        assert!((cl[2] - rv.z).abs() < 1e-13);
        let f12 = f_mn(x, c, 1, 2);
        let cl = contract_last(&f12, 1, 2, &qflat);
        let rq = r3_quad(&f12, &q);
        assert!((cl[0] - rq.x).abs() < 1e-13);
        assert!((cl[1] - rq.y).abs() < 1e-13);
        assert!((cl[2] - rq.z).abs() < 1e-13);
        let f22 = f_mn(x, c, 2, 2);
        let cl = contract_last(&f22, 2, 2, &qflat);
        let r4 = r4_quad(&f22, &q);
        for i in 0..3 {
            for j in 0..3 {
                assert!((cl[i * 3 + j] - r4[i][j]).abs() < 1e-13);
            }
        }
        // round-trip octupole flatten/unflatten.
        let o = [[[1.0; 3]; 3]; 3];
        assert_eq!(octu_unflat(&octu_flat(&o)), o);
    }

    /// `octupole_fields` must accumulate every `f^(m,3)`/`f^(3,n)` term into the right
    /// field, so the field-based octupole energy `½ Σ_A (q_A·s + d_A·vd + Q_A:vQ + O_A:vo)`
    /// equals the direct pairwise double-sum (catches accumulation / index bugs).
    #[test]
    fn octupole_fields_energy_consistent() {
        use crate::coulomb::harmonic_average;
        let nat = 3;
        let hardness = [0.5_f64, 0.45, 0.6];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.1, 0.3, -0.4),
            Vec3::new(-1.0, 1.5, 0.8),
        ];
        let q = [0.2_f64, -0.15, 0.05];
        let dip = [
            Vec3::new(0.1, -0.2, 0.05),
            Vec3::new(-0.1, 0.0, 0.2),
            Vec3::new(0.05, 0.1, -0.1),
        ];
        let quad = [
            [[0.3, 0.1, -0.1], [0.1, -0.2, 0.05], [-0.1, 0.05, -0.1]],
            [[-0.1, 0.05, 0.2], [0.05, 0.2, -0.1], [0.2, -0.1, -0.1]],
            [[0.15, -0.1, 0.0], [-0.1, -0.05, 0.1], [0.0, 0.1, -0.1]],
        ];
        let octu = [
            octu_from_components(&[0.3, 0.1, -0.1, 0.05, 0.2, -0.05, 0.1, 0.0, 0.15, -0.2]),
            octu_from_components(&[-0.1, 0.2, 0.05, -0.1, 0.0, 0.1, 0.2, -0.05, 0.1, 0.05]),
            octu_from_components(&[0.05, -0.1, 0.15, 0.0, 0.1, -0.2, 0.05, 0.1, -0.1, 0.0]),
        ];
        let (es, evd, evq, vo) = octupole_fields(nat, &hardness, &pos, &q, &dip, &quad, &octu);
        let mut e_field = 0.0;
        for a in 0..nat {
            let oo: f64 = octu_flat(&octu[a])
                .iter()
                .zip(octu_flat(&vo[a]).iter())
                .map(|(x, y)| x * y)
                .sum();
            e_field +=
                0.5 * (q[a] * es[a] + dip[a].dot(evd[a]) + dot2_full_mat(&quad[a], &evq[a]) + oo);
        }
        let dot = |va: &[f64], vb: &[f64]| -> f64 { va.iter().zip(vb).map(|(x, y)| x * y).sum() };
        let mut e_direct = 0.0;
        for a in 0..nat {
            for b in 0..nat {
                let oa = octu_flat(&octu[a]);
                let ob = octu_flat(&octu[b]);
                if a == b {
                    let c = 1.0 / (hardness[a] * hardness[a]);
                    let f33 = f_mn(Vec3::new(1.0e-6, 0.0, 0.0), c, 3, 3);
                    e_direct += 0.5 * dot(&oa, &contract_last(&f33, 3, 3, &ob));
                    continue;
                }
                let eta = harmonic_average(hardness[a], hardness[b]);
                let c = 1.0 / (eta * eta);
                let x = pos[a] - pos[b];
                let qb = [q[b]];
                let db = [dip[b].x, dip[b].y, dip[b].z];
                let qbf = quad_flat(&quad[b]);
                let da = [dip[a].x, dip[a].y, dip[a].z];
                let qaf = quad_flat(&quad[a]);
                e_direct += 0.5 * dot(&oa, &contract_last(&f_mn(x, c, 3, 0), 3, 0, &qb));
                e_direct += 0.5 * dot(&oa, &contract_last(&f_mn(x, c, 3, 1), 3, 1, &db));
                e_direct += 0.5 * dot(&oa, &contract_last(&f_mn(x, c, 3, 2), 3, 2, &qbf));
                e_direct += 0.5 * dot(&oa, &contract_last(&f_mn(x, c, 3, 3), 3, 3, &ob));
                e_direct += 0.5 * q[a] * contract_last(&f_mn(x, c, 0, 3), 0, 3, &ob)[0];
                e_direct += 0.5 * dot(&da, &contract_last(&f_mn(x, c, 1, 3), 1, 3, &ob));
                e_direct += 0.5 * dot(&qaf, &contract_last(&f_mn(x, c, 2, 3), 2, 3, &ob));
            }
        }
        assert!(
            (e_field - e_direct).abs() < 1.0e-12,
            "octupole field energy {e_field} vs direct {e_direct}"
        );
    }

    /// Stage 2c gate: the octupole Fock `F = ∂E_octu/∂P` must match a central
    /// finite-difference of the octupole correction energy (recomputing the moments from
    /// the perturbed density). H2S is used so the traceless octupole is nonzero.
    #[test]
    fn octupole_fock_matches_energy_derivative() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let p0 = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let energy_at = |p: &Matrix| -> f64 {
            let q = atomic_population(&basis, &ints.overlap, p, nat);
            octupole_energy_fock(&basis, nat, &eta, &pos, &ints, p, &q, None).energy
        };
        let q0 = atomic_population(&basis, &ints.overlap, &p0, nat);
        let f = octupole_energy_fock(&basis, nat, &eta, &pos, &ints, &p0, &q0, None).fock;
        let mut dp = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (((i * 7 + j * 13) % 17) as f64 - 8.0) * 0.02;
                dp[(i, j)] = v;
                dp[(j, i)] = v;
            }
        }
        let eps = 1.0e-5;
        let (mut pp, mut pm) = (p0.clone(), p0.clone());
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dp[(i, j)];
                pm[(i, j)] -= eps * dp[(i, j)];
            }
        }
        let fd = (energy_at(&pp) - energy_at(&pm)) / (2.0 * eps);
        let mut ana = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += f[(i, j)] * dp[(i, j)];
            }
        }
        assert!(
            (fd - ana).abs() < 1.0e-7 + 1.0e-5 * ana.abs(),
            "octupole Fock vs dE/dP FD: analytic {ana:.3e} vs FD {fd:.3e}"
        );
    }

    /// Arbitrary-rank gate: the generic `multipole_fock_generic` must reproduce the legacy
    /// energy+Fock. Fed the **identical** atomic moments the legacy path uses (so this isolates
    /// the orchestration, not moment extraction — gated separately): `max_rank=2` == the full
    /// rank-2 `multipole_energy_fock`; `max_rank=3` == rank-2 + the octupole increment
    /// `octupole_energy_fock`. H2S (S has d → nonzero octupole).
    #[test]
    fn multipole_fock_generic_matches_legacy_rank2_and_rank3() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let density = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let q = atomic_population(&basis, &ints.overlap, &density, nat);

        // Legacy: full rank-2 + octupole increment.
        let legacy2 = multipole_energy_fock(&basis, nat, &eta, &pos, &ints, &density, &q);
        let legacy3 = octupole_energy_fock(&basis, nat, &eta, &pos, &ints, &density, &q, None);

        // Identical moments fed to the generic path (same atomic_moments / atomic_octupole_moments).
        let am = atomic_moments(&basis, nat, &ints, &density);
        let octu = atomic_octupole_moments(&basis, nat, &pos, &ints, &density, None);
        let mk = |upto: usize| -> Vec<Vec<Vec<f64>>> {
            (0..nat)
                .map(|a| {
                    let mut m = vec![
                        vec![q[a]],
                        vec![am.dipole[a].x, am.dipole[a].y, am.dipole[a].z],
                        quad_flat(&am.quad[a]).to_vec(),
                    ];
                    if upto >= 3 {
                        m.push(octu_flat(&octu[a]).to_vec());
                    }
                    m
                })
                .collect()
        };
        let gen2 = multipole_fock_generic(&basis, nat, &eta, &pos, &ints, &mk(2), 2, None);
        let gen3 = multipole_fock_generic(&basis, nat, &eta, &pos, &ints, &mk(3), 3, None);

        // rank-2: generic == legacy full rank-2.
        assert!(
            (gen2.energy - legacy2.energy).abs() < 1.0e-9,
            "E(rank2) generic {} vs legacy {}",
            gen2.energy,
            legacy2.energy
        );
        let mut maxf2 = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                maxf2 = maxf2.max((gen2.fock[(i, j)] - legacy2.fock[(i, j)]).abs());
            }
        }
        assert!(
            maxf2 < 1.0e-9,
            "Fock(rank2) generic vs legacy mismatch {maxf2:.3e}"
        );

        // rank-3: generic == legacy rank-2 + octupole increment.
        assert!(
            (gen3.energy - (legacy2.energy + legacy3.energy)).abs() < 1.0e-9,
            "E(rank3) generic {} vs legacy {}",
            gen3.energy,
            legacy2.energy + legacy3.energy
        );
        let mut maxf3 = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                maxf3 = maxf3
                    .max((gen3.fock[(i, j)] - (legacy2.fock[(i, j)] + legacy3.fock[(i, j)])).abs());
            }
        }
        assert!(
            maxf3 < 1.0e-9,
            "Fock(rank3) generic vs legacy mismatch {maxf3:.3e}"
        );
    }

    /// Optimization gate: the geometry-fixed `OnsiteMomentCache` must give **byte-identical**
    /// arbitrary-rank moments and Fock to the recompute path (the cached value is bit-for-bit the
    /// raw on-site integral). H2S, ranks up to 4.
    #[test]
    fn onsite_moment_cache_matches_recompute() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let density = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let q = atomic_population(&basis, &ints.overlap, &density, nat);
        let max_rank = 4;
        let cache = OnsiteMomentCache::build(&basis, nat, &pos, max_rank);

        // Moments byte-identical with/without the cache.
        for l in 1..=max_rank {
            let m_no = atomic_moment_rank_l(&basis, nat, &pos, &ints, &density, l, None);
            let m_yes = atomic_moment_rank_l(&basis, nat, &pos, &ints, &density, l, Some(&cache));
            for (a, b) in m_no.iter().zip(&m_yes) {
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "moment rank {l} not byte-identical"
                    );
                }
            }
        }

        // Fock byte-identical with/without the cache.
        let moments = build_generic_moments(&basis, nat, &pos, &ints, &density, &q, max_rank, None);
        let f_no = multipole_fock_generic(&basis, nat, &eta, &pos, &ints, &moments, max_rank, None);
        let f_yes = multipole_fock_generic(
            &basis,
            nat,
            &eta,
            &pos,
            &ints,
            &moments,
            max_rank,
            Some(&cache),
        );
        assert_eq!(f_no.energy.to_bits(), f_yes.energy.to_bits());
        for i in 0..n {
            for j in 0..n {
                assert_eq!(f_no.fock[(i, j)].to_bits(), f_yes.fock[(i, j)].to_bits());
            }
        }
    }

    /// Arbitrary-rank gate: the generic gradient pieces must reproduce the legacy ones (which are
    /// themselves FD-gated): `multipole_kernel_forces_generic` and `multipole_overlap_weight_generic`
    /// at `max_rank=2` == the rank-2 legacy paths, and at `max_rank=3` == rank-2 + the octupole
    /// increment. Fed identical moments (isolating orchestration). H2S (S has d).
    #[test]
    fn multipole_generic_gradient_matches_legacy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let density = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let q = atomic_population(&basis, &ints.overlap, &density, nat);

        let am = atomic_moments(&basis, nat, &ints, &density);
        let octu = atomic_octupole_moments(&basis, nat, &pos, &ints, &density, None);
        let mk = |upto: usize| -> Vec<Vec<Vec<f64>>> {
            (0..nat)
                .map(|a| {
                    let mut m = vec![
                        vec![q[a]],
                        vec![am.dipole[a].x, am.dipole[a].y, am.dipole[a].z],
                        quad_flat(&am.quad[a]).to_vec(),
                    ];
                    if upto >= 3 {
                        m.push(octu_flat(&octu[a]).to_vec());
                    }
                    m
                })
                .collect()
        };

        // --- kernel forces ---
        let leg_kf2 = multipole_kernel_forces(&basis, nat, &eta, &pos, &ints, &density, &q);
        let leg_kf3 = octupole_kernel_forces(&basis, nat, &eta, &pos, &ints, &density, &q, None);
        let gen_kf2 = multipole_kernel_forces_generic(nat, &eta, &pos, &mk(2), 2);
        let gen_kf3 = multipole_kernel_forces_generic(nat, &eta, &pos, &mk(3), 3);
        let mut mkf2 = 0.0_f64;
        let mut mkf3 = 0.0_f64;
        for a in 0..nat {
            let d2 = gen_kf2[a] - leg_kf2[a];
            mkf2 = mkf2.max(d2.x.abs()).max(d2.y.abs()).max(d2.z.abs());
            let s3 = leg_kf2[a] + leg_kf3[a];
            let d3 = gen_kf3[a] - s3;
            mkf3 = mkf3.max(d3.x.abs()).max(d3.y.abs()).max(d3.z.abs());
        }
        assert!(
            mkf2 < 1.0e-9,
            "kernel force(rank2) generic vs legacy {mkf2:.3e}"
        );
        assert!(
            mkf3 < 1.0e-9,
            "kernel force(rank3) generic vs legacy {mkf3:.3e}"
        );

        // --- overlap-Pulay weight ---
        let leg_w2 = multipole_overlap_weight(&basis, nat, &eta, &pos, &ints, &density, &q);
        let leg_w3 = octupole_overlap_weight(&basis, nat, &eta, &pos, &ints, &density, &q, None);
        let gen_w2 =
            multipole_overlap_weight_generic(&basis, nat, &eta, &pos, &density, &mk(2), 2, None);
        let gen_w3 =
            multipole_overlap_weight_generic(&basis, nat, &eta, &pos, &density, &mk(3), 3, None);
        let mut mw2 = 0.0_f64;
        let mut mw3 = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                mw2 = mw2.max((gen_w2[(i, j)] - leg_w2[(i, j)]).abs());
                mw3 = mw3.max((gen_w3[(i, j)] - (leg_w2[(i, j)] + leg_w3[(i, j)])).abs());
            }
        }
        assert!(
            mw2 < 1.0e-9,
            "overlap weight(rank2) generic vs legacy {mw2:.3e}"
        );
        assert!(
            mw3 < 1.0e-9,
            "overlap weight(rank3) generic vs legacy {mw3:.3e}"
        );
    }

    /// Arbitrary-rank SCC-mixing gate: the generic stride nests the legacy ones
    /// (`max_rank=2 → 9 = MOMENT_STRIDE`, `+rank3 → +10 = OCTU_STRIDE`), and pack→unpack is a
    /// round-trip on detraced moments (H2S, S has d → nonzero rank-3).
    #[test]
    fn generic_moment_pack_unpack_round_trip() {
        assert_eq!(generic_moment_stride(2), MOMENT_STRIDE);
        assert_eq!(generic_moment_stride(3), MOMENT_STRIDE + OCTU_STRIDE);
        assert_eq!(generic_moment_stride(4), MOMENT_STRIDE + OCTU_STRIDE + 15);

        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let density = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;

        let max_rank = 3;
        // Detraced moments l=1..=3 (full 3^l), rank-0 placeholder.
        let moments: Vec<Vec<Vec<f64>>> = {
            let mut m: Vec<Vec<Vec<f64>>> = (0..nat).map(|_| vec![vec![0.0]]).collect();
            for l in 1..=max_rank {
                let ml = atomic_moment_rank_l(&basis, nat, &pos, &ints, &density, l, None);
                for (a, t) in ml.into_iter().enumerate() {
                    m[a].push(t);
                }
            }
            m
        };
        let mut packed = vec![0.0_f64; generic_moment_stride(max_rank) * nat];
        pack_generic_moments(&moments, max_rank, &mut packed);
        let back = unpack_generic_moments(&packed, nat, max_rank);

        let mut maxd = 0.0_f64;
        for a in 0..nat {
            for l in 1..=max_rank {
                for (x, y) in moments[a][l].iter().zip(back[a][l].iter()) {
                    maxd = maxd.max((x - y).abs());
                }
            }
        }
        assert!(maxd < 1.0e-12, "pack→unpack round-trip mismatch {maxd:.3e}");
    }

    /// Optimization gate: the unique-symmetric-component kernels must reproduce the full-tensor
    /// helpers exactly (up to FP) for every `(m,n)` up to rank 4 — `f_mn_unique` expands to `f_mn`,
    /// `contract_last_unique` == `contract_last`, `kernel_grad_unique` == `kernel_grad` (on
    /// symmetric moments, which is what the physics produces). This is what lets the generic path
    /// avoid materializing the `3^(m+n)` tensors.
    #[test]
    fn unique_component_kernels_match_full_tensor() {
        let x = Vec3::new(0.7, -0.4, 0.9);
        let c = 1.0 / (0.55 * 0.55);
        // A fully symmetric rank-`l` tensor from arbitrary unique components.
        let sym = |l: usize, seed: f64| -> Vec<f64> {
            let comps = crate::integrals::cartesian_rank_components(l);
            let u: Vec<f64> = (0..comps.len())
                .map(|i| ((i as f64 + 1.0) * seed).sin())
                .collect();
            expand_symmetric_cartesian(&u, l)
        };
        for m in 0..=4 {
            for n in 0..=4 {
                // f_mn_unique expands to the full f_mn.
                let full = f_mn(x, c, m, n);
                let uexp = expand_symmetric_cartesian(&f_mn_unique(x, c, m, n), m + n);
                for (a, b) in full.iter().zip(uexp.iter()) {
                    assert!((a - b).abs() < 1.0e-12, "f_mn_unique({m},{n})");
                }
                // contract_last_unique == contract_last (symmetric moment).
                let mb = sym(n, 0.37 + n as f64);
                let cl_full = contract_last(&full, m, n, &mb);
                let cl_uni = contract_last_unique(&f_mn_unique(x, c, m, n), m, n, &mb);
                for (a, b) in cl_full.iter().zip(cl_uni.iter()) {
                    assert!(
                        (a - b).abs() < 1.0e-10,
                        "contract_last_unique({m},{n}): {a} vs {b}"
                    );
                }
                // kernel_grad_unique == kernel_grad (symmetric moments).
                let ma = sym(m, 0.21 + m as f64);
                let kg_full = kernel_grad(&f_mn_grad(x, c, m, n), m, n, &ma, &mb);
                let kg_uni = kernel_grad_unique(&f_mn_grad_unique(x, c, m, n), m, n, &ma, &mb);
                assert!(
                    (kg_full.x - kg_uni.x).abs() < 1.0e-10
                        && (kg_full.y - kg_uni.y).abs() < 1.0e-10
                        && (kg_full.z - kg_uni.z).abs() < 1.0e-10,
                    "kernel_grad_unique({m},{n})"
                );
            }
        }
    }

    /// v0.2.0 perf fix: the geometry-fixed on-site octupole cache must give **byte-identical**
    /// energy + Fock to the recompute path (the cache value is bit-for-bit `onsite_octupole`).
    /// H2S (S has d) so the octupole is nonzero.
    #[test]
    fn onsite_octupole_cache_matches_recompute() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let density = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let q = atomic_population(&basis, &ints.overlap, &density, nat);
        let cache = OnsiteOctupoleCache::build(&basis, nat, &pos);

        // Moments byte-identical with/without the cache.
        let m_no = atomic_octupole_moments(&basis, nat, &pos, &ints, &density, None);
        let m_yes = atomic_octupole_moments(&basis, nat, &pos, &ints, &density, Some(&cache));
        for (a, b) in m_no.iter().zip(&m_yes) {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        assert_eq!(a[i][j][k], b[i][j][k]);
                    }
                }
            }
        }
        // Energy + Fock byte-identical.
        let no = octupole_energy_fock(&basis, nat, &eta, &pos, &ints, &density, &q, None);
        let yes = octupole_energy_fock(&basis, nat, &eta, &pos, &ints, &density, &q, Some(&cache));
        assert_eq!(no.energy, yes.energy);
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                assert_eq!(no.fock[(i, j)], yes.fock[(i, j)]);
            }
        }
    }

    /// Stage 2d/2e gate: the octupole correction enters the SCC self-consistently — the
    /// run converges and the total energy shifts vs dipole+quad-only mDFTB2 (H2S, S has d).
    #[test]
    fn octupole_scc_converges_and_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt_dq = crate::electronic::ElectronicOptions::default();
        opt_dq.multipole = true;
        let r_dq = crate::electronic::run_electronic(&system, &params, opt_dq).unwrap();
        let mut opt_o = crate::electronic::ElectronicOptions::default();
        opt_o.multipole = true;
        opt_o.multipole_octupole = true;
        let r_o = crate::electronic::run_electronic(&system, &params, opt_o).unwrap();
        eprintln!(
            "mDFTB2 {:.8}  +octupole {:.8}  d={:.3e}",
            r_dq.total_free,
            r_o.total_free,
            r_o.total_free - r_dq.total_free
        );
        assert!(
            (r_o.total_free - r_dq.total_free).abs() > 1.0e-9,
            "octupole SCC had no effect on the energy: {} vs {}",
            r_dq.total_free,
            r_o.total_free
        );
    }

    /// Arbitrary-rank (16-pole and higher) gate: the generic multipole path
    /// (`multipole_order ≥ 4`) puts every rank 1..=L into the joint tblite/GFN2-style Broyden
    /// vector, so the **hexadecapole (rank-4, 16-pole)** SCC must converge and shift the energy
    /// vs the dipole+quadrupole baseline. H2S (S has d-functions ⇒ rank-3/4 on-site moments
    /// are nonzero). This verifies the arbitrary-rank joint moment mixing actually works.
    #[test]
    fn generic_hexadecapole_scc_converges_and_changes_energy() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let baseline = {
            let mut o = crate::electronic::ElectronicOptions::default();
            o.multipole = true; // dipole + quadrupole (legacy path)
            crate::electronic::run_electronic(&system, &params, o).unwrap()
        };
        assert!(
            baseline.converged,
            "dipole+quad multipole SCC did not converge"
        );
        let hexa = {
            let mut o = crate::electronic::ElectronicOptions::default();
            o.multipole = true;
            o.multipole_order = 4; // generic path: ranks 1..=4 (up to 16-pole), jointly mixed
            crate::electronic::run_electronic(&system, &params, o).unwrap()
        };
        assert!(
            hexa.converged,
            "arbitrary-rank (order-4 / 16-pole) multipole SCC did not converge"
        );
        assert!(
            (hexa.total_free - baseline.total_free).abs() > 1.0e-9,
            "rank-3/4 moments had no effect on the energy: {} vs {}",
            baseline.total_free,
            hexa.total_free
        );
    }

    /// Rank-continuation gate: the staged "rank ladder" (converge octupole rank 3, warm-start
    /// rank 4) must converge and reach the **same** SCF solution as a direct rank-4 run — the
    /// staging changes only the convergence path, not the minimum. (Demonstrates the
    /// progressive 8-pole → 16-pole mixing the user asked about.)
    #[test]
    fn rank_ladder_converges_to_same_solution_as_direct() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.961 0.928\nH 0.0 -0.961 0.928\n",
            0.0,
            false,
        )
        .unwrap();
        let direct = {
            let mut o = crate::electronic::ElectronicOptions::default();
            o.multipole = true;
            o.multipole_order = 4;
            crate::electronic::run_electronic(&system, &params, o).unwrap()
        };
        let ladder = crate::electronic::run_electronic_rank_ladder(
            &system,
            &params,
            &crate::electronic::ElectronicOptions::default(),
            3, // base: octupole (8-pole)
            4, // target: hexadecapole (16-pole)
        )
        .unwrap();
        assert!(
            ladder.converged,
            "rank-continuation ladder did not converge"
        );
        assert!(
            (ladder.total_free - direct.total_free).abs() < 1.0e-6,
            "ladder {} vs direct {} (should reach the same SCF solution)",
            ladder.total_free,
            direct.total_free
        );
    }

    /// Raw atomic Mulliken population `q_A = Σ_{μ∈A}(P S)_{μμ}` (linear in `P`; the
    /// constant reference offset is irrelevant to `∂E/∂P`).
    fn atomic_population(basis: &BasisSet, s: &Matrix, p: &Matrix, nat: usize) -> Vec<f64> {
        let n = basis.len();
        let mut q = vec![0.0_f64; nat];
        for (mu, ao) in basis.aos.iter().enumerate() {
            let mut acc = 0.0;
            for nu in 0..n {
                acc += p[(mu, nu)] * s[(nu, mu)];
            }
            q[ao.atom_index] += acc;
        }
        q
    }

    /// The multipole Fock shift must equal `∂E_mp/∂P` (variational consistency): along a
    /// symmetric density perturbation `δP`, the central FD of `E_mp(P)` (with `q`
    /// recomputed from `P`) matches `Σ_{μν} F_{μν} δP_{μν}`.
    #[test]
    fn multipole_fock_matches_energy_derivative() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat]; // arbitrary positive hardness (consistency is kernel-agnostic)
        let p0 = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let energy_at = |p: &Matrix| -> f64 {
            let q = atomic_population(&basis, &ints.overlap, p, nat);
            multipole_energy_fock(&basis, nat, &eta, &atom_pos, &ints, p, &q).energy
        };
        let q0 = atomic_population(&basis, &ints.overlap, &p0, nat);
        let f = multipole_energy_fock(&basis, nat, &eta, &atom_pos, &ints, &p0, &q0).fock;
        // Symmetric pseudo-random perturbation.
        let mut dp = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (((i * 7 + j * 13) % 17) as f64 - 8.0) * 0.02;
                dp[(i, j)] = v;
                dp[(j, i)] = v;
            }
        }
        let eps = 1.0e-5;
        let mut pp = p0.clone();
        let mut pm = p0.clone();
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dp[(i, j)];
                pm[(i, j)] -= eps * dp[(i, j)];
            }
        }
        let fd = (energy_at(&pp) - energy_at(&pm)) / (2.0 * eps);
        let mut ana = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += f[(i, j)] * dp[(i, j)];
            }
        }
        assert!(
            (fd - ana).abs() < 1.0e-6 * ana.abs().max(1.0e-4),
            "multipole Fock != dE/dP: fd {fd:.6e} ana {ana:.6e}"
        );
    }

    /// The erf-cloud radial ladder `G_p = d^p/d(R²)^p [erf(R/σ)/R]` must match the value of
    /// `fr_gamma_exchange` (at p=0) and the central FD in `s=R²` of the previous rank (p≥1).
    #[test]
    fn erf_cloud_radial_derivs_match_fd() {
        let sigma = 1.3_f64;
        let nmax = 5;
        for &r2 in &[0.05_f64, 0.7, 2.5, 9.0] {
            let g = erf_cloud_radial_derivs(r2, sigma, nmax);
            let v0 = crate::coulomb::fr_gamma_exchange(r2.sqrt(), sigma);
            assert!((g[0] - v0).abs() < 1.0e-12, "G0 {} vs fr {}", g[0], v0);
            let h = 1.0e-5;
            for p in 1..=nmax {
                let plus = erf_cloud_radial_derivs(r2 + h, sigma, nmax)[p - 1];
                let minus = erf_cloud_radial_derivs(r2 - h, sigma, nmax)[p - 1];
                let fd = (plus - minus) / (2.0 * h);
                assert!(
                    (g[p] - fd).abs() < 1.0e-6 * g[p].abs().max(1.0e-4),
                    "G{p}({r2}) = {} vs fd {fd}",
                    g[p]
                );
            }
        }
    }

    /// The CAMM cumulative moments must conserve the molecular electron dipole:
    /// `Σ_A (μ_A + q_A^elec R_A) = Σ_κλ P_κλ ⟨κ|r|λ⟩`.
    #[test]
    fn camm_moments_conserve_molecular_dipole() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let p0 = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        let m = camm_atomic_moments(&basis, nat, &ints, &p0, &atom_pos);
        let qel = atomic_population(&basis, &ints.overlap, &p0, nat);
        let mut cam = Vec3::zero();
        for a in 0..nat {
            cam += m.dipole[a] + atom_pos[a] * qel[a];
        }
        let mut refd = Vec3::zero();
        for k in 0..n {
            for l in 0..n {
                let p = p0[(k, l)];
                let d = onsite_dipole(&ints, k, l);
                let s = ints.overlap[(k, l)];
                refd += (d + atom_pos[basis.aos[l].atom_index] * s) * p;
            }
        }
        assert!(
            (cam - refd).norm() < 1.0e-9,
            "CAMM dipole {cam:?} != molecular electron dipole {refd:?}"
        );
    }

    /// The CAMM/AES Fock must equal `∂E/∂P` for the cumulative moments: the central FD of the
    /// CAMM energy (with `q` *and* the CAMM `μ,Θ` recomputed from `P`) matches `Σ F_{μν} δP_{μν}`.
    #[test]
    fn camm_aes_fock_matches_energy_derivative() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = IntegralMatrices::build(&system, &basis).unwrap();
        let atom_pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let p0 = crate::electronic::run_electronic(
            &system,
            &params,
            crate::electronic::ElectronicOptions::default(),
        )
        .unwrap()
        .density;
        // Per-atom (element-specific) κ — distinct O vs H values exercise the √(κ_A·κ_B) path.
        let kappa: Vec<f64> = (0..nat)
            .map(|a| if system.atoms[a].z == 8 { 1.3 } else { 0.8 })
            .collect();
        let scale = 0.7_f64; // exercise the s_AES scaling too
        // Per-element s_onsite (O vs H) exercises the per-atom on-site-penalty path.
        let onsite: Vec<f64> = (0..nat)
            .map(|a| if system.atoms[a].z == 8 { 0.6 } else { 0.4 })
            .collect();
        let energy_at = |p: &Matrix| -> f64 {
            let q = atomic_population(&basis, &ints.overlap, p, nat);
            let m = camm_atomic_moments(&basis, nat, &ints, p, &atom_pos);
            camm_aes_energy_fock(&basis, nat, &eta, &atom_pos, &ints, &m, &q, &kappa, scale, &onsite)
                .energy
        };
        let q0 = atomic_population(&basis, &ints.overlap, &p0, nat);
        let m0 = camm_atomic_moments(&basis, nat, &ints, &p0, &atom_pos);
        let f = camm_aes_energy_fock(&basis, nat, &eta, &atom_pos, &ints, &m0, &q0, &kappa, scale,
            &onsite)
        .fock;
        let mut dp = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (((i * 7 + j * 13) % 17) as f64 - 8.0) * 0.02;
                dp[(i, j)] = v;
                dp[(j, i)] = v;
            }
        }
        let eps = 1.0e-5;
        let mut pp = p0.clone();
        let mut pm = p0.clone();
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dp[(i, j)];
                pm[(i, j)] -= eps * dp[(i, j)];
            }
        }
        let fd = (energy_at(&pp) - energy_at(&pm)) / (2.0 * eps);
        let mut ana = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += f[(i, j)] * dp[(i, j)];
            }
        }
        assert!(
            (fd - ana).abs() < 1.0e-6 * ana.abs().max(1.0e-4),
            "CAMM Fock != dE/dP: fd {fd:.6e} ana {ana:.6e}"
        );
    }

    /// The mDFTB2 correction must be inert when off (== GFN1), converge when on, and
    /// measurably change the energy of a polar molecule.
    #[test]
    fn scc_multipole_off_equals_gfn1_and_on_converges() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let mut off = crate::electronic::ElectronicOptions::default();
        off.multipole = false;
        let e_gfn1 = crate::electronic::run_electronic(&system, &params, off)
            .unwrap()
            .total_internal;
        let mut on = crate::electronic::ElectronicOptions::default();
        on.multipole = true;
        let res = crate::electronic::run_electronic(&system, &params, on).unwrap();
        assert!(res.converged, "mDFTB2 SCC did not converge");
        let de = res.total_internal - e_gfn1;
        eprintln!(
            "water: GFN1 {e_gfn1:.6} Ha, mDFTB2 {:.6} Ha, dE = {de:.6} Ha",
            res.total_internal
        );
        assert!(
            de.abs() > 1.0e-6,
            "multipole correction had no effect: dE {de:.3e}"
        );
    }

    /// CAMM-on-mDFTB2 must converge, differ from both GFN1 and mDFTB2, and respond to both
    /// calibration levers (`camm_aes_scale` and `camm_damp`).
    #[test]
    fn scc_camm_on_mdftb2_converges_and_scales() {
        use crate::electronic::{ElectronicOptions, MultipoleModel};
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "6\nwater dimer\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\nO 0.0 0.0 2.9\nH 0.0 0.94 3.2\nH 0.0 -0.94 3.2\n",
            0.0,
            false,
        )
        .unwrap();
        let run = |o: ElectronicOptions| crate::electronic::run_electronic(&system, &params, o);
        let mut gfn1 = ElectronicOptions::default();
        gfn1.multipole = false;
        let e_gfn1 = run(gfn1).unwrap().total_internal;
        let mut md = ElectronicOptions::default();
        md.multipole = true;
        let e_md = run(md).unwrap().total_internal;
        let mut camm = ElectronicOptions::default();
        camm.multipole = true;
        camm.multipole_model = MultipoleModel::CammOnMdftb2;
        let res = run(camm.clone()).unwrap();
        assert!(res.converged, "CAMM-on-mDFTB2 SCC did not converge");
        let e_camm = res.total_internal;
        assert!((e_camm - e_gfn1).abs() > 1.0e-6, "CAMM ≡ GFN1 (no effect)");
        assert!((e_camm - e_md).abs() > 1.0e-8, "CAMM ≡ mDFTB2 (off-site not replaced)");
        // s_AES scales the off-site AES away.
        let mut camm0 = camm.clone();
        camm0.camm_aes_scale = 0.0;
        let e_camm0 = run(camm0).unwrap().total_internal;
        assert!((e_camm - e_camm0).abs() > 1.0e-7, "camm_aes_scale had no effect");
        // camm_damp (κ) re-balances the contact region (range-selective).
        let mut camm_k = camm.clone();
        camm_k.camm_damp = 1.5;
        let e_camm_k = run(camm_k).unwrap().total_internal;
        assert!((e_camm - e_camm_k).abs() > 1.0e-8, "camm_damp had no effect");
        eprintln!(
            "water dimer: GFN1 {e_gfn1:.6}  mDFTB2 {e_md:.6}  CAMM {e_camm:.6}  CAMM(s=0) {e_camm0:.6}  CAMM(κ=1.5) {e_camm_k:.6}"
        );
    }

    #[test]
    fn matchings_counts() {
        // #partial matchings: k=2 -> 2 (one pair, two singletons), k=3 -> 4, k=4 -> 10,
        // k=5 -> 26 (telephone numbers / involutions).
        assert_eq!(matchings(2).len(), 2);
        assert_eq!(matchings(3).len(), 4);
        assert_eq!(matchings(4).len(), 10);
        assert_eq!(matchings(5).len(), 26);
    }
}
