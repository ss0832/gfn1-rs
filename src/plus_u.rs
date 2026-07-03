// SPDX-License-Identifier: GPL-3.0-or-later
//! DFT+U / DFT+U+V (extended Hubbard) correction on top of GFN1-xTB.
//!
//! Plain GFN1-xTB describes the on-site charge fluctuation only at the monopole
//! (shell-charge) level through the second-order SCC `γ` (the Klopman–Ohno
//! Hubbard/hardness). It carries **no** orbital-resolved penalty against the
//! self-interaction / over-delocalisation of a localised (typically transition-
//! metal `d`) shell. That self-interaction error is the diagnosed root cause of
//! GFN-xTB's poor transition-metal spin-state energetics and geometries.
//!
//! This module adds the rotationally-invariant (Dudarev) extended-Hubbard energy
//!
//! ```text
//! E = Σ_A (U_A/2) Σ_σ Tr[ n^σ_AA (1 − n^σ_AA) ]
//!   − Σ_{A<B} (V_AB) Σ_σ Tr[ n^σ_AB n^σ_BA ]
//! ```
//!
//! built from the **dual (symmetric Mulliken) population matrix** of the
//! correlated subspace in the non-orthogonal AO basis,
//!
//! ```text
//! n^σ_μν = ½ ( P^σ S + S P^σ )_μν ,   μ,ν ∈ correlated AOs ,
//! ```
//!
//! (`P^σ` the spin-channel AO density, `S` the overlap). The on-site `+U` term
//! is the standard fully-localised-limit (FLL) DFTB+U of Hourahine et al.; the
//! inter-site `+V` term (Campo–Cococcioni) penalises the cross-site occupation
//! and restores the metal–ligand hybridisation that bare `+U` over-localises —
//! the piece that matters for covalent molecular complexes.
//!
//! ## Variational consistency
//!
//! The SCC potential (Fock contribution) returned here is the exact functional
//! derivative `∂E/∂P^σ`,
//!
//! ```text
//! G^σ = ½ ( Ṽ^σ S + S Ṽ^σ ) ,   Ṽ^σ = block[ U(½ I − n^σ_AA) ]  (+ V cross blocks),
//! ```
//!
//! so the SCC is stationary and the energy needs no separate double-counting
//! subtraction (it is added directly, exactly like the spin and multipole
//! terms). The on-site identity `G^σ = ∂E_U/∂P^σ` is checked against a central
//! finite difference in the unit tests.
//!
//! Stage 1 (this commit): the correlated-subspace selection, the dual population,
//! the on-site `+U` energy and its variational Fock potential, all FD-verified.
//! The inter-site `+V` energy/potential and the SCC + analytic-gradient wiring
//! follow in later stages.

use crate::basis::BasisSet;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::AngularMomentum;

/// One correlated atom: the AO indices of its correlated shell and the on-site
/// Hubbard `U` (Hartree) to apply there.
#[derive(Clone, Debug)]
pub struct CorrelatedAtom {
    /// Index of the atom in the system.
    pub atom_index: usize,
    /// AO indices (into the global basis) spanning the correlated shell.
    pub aos: Vec<usize>,
    /// On-site Hubbard `U` for this atom's correlated shell, in Hartree.
    pub u: f64,
}

/// Collect the correlated subspace from the basis: every atom carrying a shell
/// of the requested angular momentum (`l = 2`, the `d` shell, by default) whose
/// element appears in `u_by_z` with a non-zero `U`. Atoms with no entry, zero
/// `U`, or no matching shell are skipped.
///
/// If an atom has more than one shell of the requested angular momentum (e.g. a
/// valence + polarisation `d`), all such AOs are pooled into one correlated
/// block — the dual population is taken over the union, matching the
/// shell-agnostic on-site definition.
pub fn correlated_subspace(
    basis: &BasisSet,
    u_by_z: &[(u8, f64)],
    angular: AngularMomentum,
) -> Vec<CorrelatedAtom> {
    let u_of = |z: u8| -> f64 {
        u_by_z
            .iter()
            .find(|(zz, _)| *zz == z)
            .map(|(_, u)| *u)
            .unwrap_or(0.0)
    };
    let mut out: Vec<CorrelatedAtom> = Vec::new();
    for shell in &basis.shells {
        if shell.angular != angular {
            continue;
        }
        let u = u_of(shell.z);
        if u == 0.0 {
            continue;
        }
        let aos: Vec<usize> = (shell.first_ao..shell.first_ao + shell.nao).collect();
        if let Some(existing) = out.iter_mut().find(|c| c.atom_index == shell.atom_index) {
            existing.aos.extend(aos);
        } else {
            out.push(CorrelatedAtom {
                atom_index: shell.atom_index,
                aos,
                u,
            });
        }
    }
    out
}

/// Automatically select the correlated subspace **without any element list**:
/// every atom carrying a `d` shell. With `all_d = false` (default) only **valence**
/// `d` shells (`reference_occ > 0` — the transition metals, whose localised `d`
/// electrons carry the self-interaction error) are taken; main-group empty `d`
/// polarisation shells are excluded. With `all_d = true` **every** `d` shell is
/// included (also the main-group polarisation `d`; note these are nearly empty so
/// the FLL penalty is small, and their linear-response `U` is more prone to the
/// ill-conditioning handled in [`extract_uv_from_response`]). The `u` field is left
/// at `0.0`, to be filled by the linear-response procedure.
pub fn correlated_subspace_auto(basis: &BasisSet, all_d: bool) -> Vec<CorrelatedAtom> {
    let mut out: Vec<CorrelatedAtom> = Vec::new();
    for shell in &basis.shells {
        if shell.angular != AngularMomentum::D || (!all_d && shell.reference_occ <= 0.0) {
            continue;
        }
        let aos: Vec<usize> = (shell.first_ao..shell.first_ao + shell.nao).collect();
        if let Some(existing) = out.iter_mut().find(|c| c.atom_index == shell.atom_index) {
            existing.aos.extend(aos);
        } else {
            out.push(CorrelatedAtom { atom_index: shell.atom_index, aos, u: 0.0 });
        }
    }
    out
}

/// Gauss–Jordan inverse of a small dense matrix (row-major `Vec<Vec<f64>>`).
/// Returns `None` if singular. Used to invert the (correlated-atom × correlated-
/// atom) occupation-response matrices in the linear-response `U`/`V` extraction.
pub fn invert_small(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 || m.iter().any(|r| r.len() != n) {
        return None;
    }
    // Augment [M | I].
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1.0e-14 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for x in a[col].iter_mut() {
            *x /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f != 0.0 {
                for c in 0..2 * n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    Some(a.iter().map(|row| row[n..].to_vec()).collect())
}

/// Extract the non-empirical on-site `U` and inter-site `V` from the bare (`chi0`)
/// and self-consistent (`chi`) occupation-response matrices (Cococcioni–de
/// Gironcoli, full inter-site form):
///
/// ```text
/// K = chi0⁻¹ − chi⁻¹ ,   U_I = K_II ,   V_IJ = −K_IJ  (I ≠ J).
/// ```
///
/// Returns `(u, v)` with `u[i]` the per-atom on-site `U` and `v[i][j]` the
/// inter-site `V`.
///
/// **Robustness** (for the ill-conditioned degenerate-`d¹` case, where the
/// occupation response `χ` is huge / near-singular and the bare difference of two
/// near-singular inverses is numerically unstable):
/// - **Tikhonov regularisation** `(χ + λI)⁻¹` (`λ = REG`) caps the inverse of a
///   near-singular response so the result stays finite;
/// - the extracted `U` is **clamped to the physical range `[0, U_MAX]`** (and `V`
///   to `[−U_MAX, U_MAX]`), and any non-finite entry leaves that site uncorrected.
///
/// `λ` and `U_MAX` are fixed numerical-robustness constants, not fitted parameters.
pub fn extract_uv_from_response(chi0: &[Vec<f64>], chi: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    const REG: f64 = 1.0e-3; // Tikhonov regularisation of the response matrices
    const U_MAX: f64 = 1.0; // physical clamp on |U|, |V| (Hartree, ~27 eV)
    let n = chi0.len();
    let zero = (vec![0.0; n], vec![vec![0.0; n]; n]);
    if n == 0 {
        return zero;
    }
    let reg = |m: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut r: Vec<Vec<f64>> = m.to_vec();
        for (i, row) in r.iter_mut().enumerate() {
            row[i] += REG;
        }
        r
    };
    let (Some(i0), Some(i1)) = (invert_small(&reg(chi0)), invert_small(&reg(chi))) else {
        return zero;
    };
    let mut u = vec![0.0; n];
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let k = i0[i][j] - i1[i][j];
            if !k.is_finite() {
                continue; // ill-conditioned site → left uncorrected
            }
            if i == j {
                u[i] = k.clamp(0.0, U_MAX);
            } else {
                v[i][j] = (-k).clamp(-U_MAX, U_MAX);
            }
        }
    }
    (u, v)
}

/// Build the orbital potential `Ṽ` and the `+U+V` energy working **only in the
/// correlated subspace** (the few `d` AOs), so the cost is `O(|corr|² N)` — linear
/// in system size — instead of the `O(N³)` of a full `P·S` product. Returns
/// `(corr_aos, vtilde_corr, energy)`: the sorted union of correlated AO indices,
/// the dense `|corr|×|corr|` row-major `Ṽ` restricted to them, and `E_{+U+V}`.
///
/// `Ṽ_AA = U(½ I − n_AA)` (on-site), `Ṽ_AB = −V n_AB` (inter-site), with the dual
/// population `n_{ab} = ½(M_{ab}+M_{ba})`, `M = P S` evaluated entry-wise only at
/// correlated `(a,b)` pairs.
fn build_vtilde_corr(
    p: &Matrix,
    s: &Matrix,
    subspace: &[CorrelatedAtom],
    pairs: &[IntersitePair],
) -> (Vec<usize>, Vec<f64>, f64) {
    let n = p.rows();
    let mut corr_aos: Vec<usize> = subspace.iter().flat_map(|a| a.aos.iter().copied()).collect();
    corr_aos.sort_unstable();
    corr_aos.dedup();
    let c = corr_aos.len();
    let mut pos = vec![usize::MAX; n];
    for (ic, &a) in corr_aos.iter().enumerate() {
        pos[a] = ic;
    }
    // M_corr[ic][jc] = (P S)_{corr_aos[ic], corr_aos[jc]}, computed entry-wise.
    let mut mcorr = vec![0.0; c * c];
    for (ic, &a) in corr_aos.iter().enumerate() {
        for (jc, &b) in corr_aos.iter().enumerate() {
            let mut acc = 0.0;
            for k in 0..n {
                acc += p[(a, k)] * s[(k, b)];
            }
            mcorr[ic * c + jc] = acc;
        }
    }
    let dual = |ic: usize, jc: usize| 0.5 * (mcorr[ic * c + jc] + mcorr[jc * c + ic]);
    let mut vtilde = vec![0.0; c * c];
    let mut energy = 0.0;
    // On-site +U (FLL): E_A = (U/2)[Tr n − Tr n²]; Ṽ_AA = U(½ I − n_AA).
    for atom in subspace {
        let mut tr_n = 0.0;
        let mut tr_n2 = 0.0;
        for &a in &atom.aos {
            let ia = pos[a];
            tr_n += dual(ia, ia);
            for &b in &atom.aos {
                let nv = dual(ia, pos[b]);
                tr_n2 += nv * nv;
            }
        }
        energy += 0.5 * atom.u * (tr_n - tr_n2);
        for &a in &atom.aos {
            let ia = pos[a];
            for &b in &atom.aos {
                let ib = pos[b];
                let delta = if a == b { 0.5 } else { 0.0 };
                vtilde[ia * c + ib] += atom.u * (delta - dual(ia, ib));
            }
        }
    }
    // Inter-site +V: E_V = −V Σ (n_AB)²; Ṽ_AB = Ṽ_BA = −V n_AB.
    for pair in pairs {
        let mut sumsq = 0.0;
        for &a in &subspace[pair.a].aos {
            let ia = pos[a];
            for &b in &subspace[pair.b].aos {
                let ib = pos[b];
                let nv = dual(ia, ib);
                sumsq += nv * nv;
                vtilde[ia * c + ib] += -pair.v * nv;
                vtilde[ib * c + ia] += -pair.v * nv;
            }
        }
        energy += -pair.v * sumsq;
    }
    (corr_aos, vtilde, energy)
}

/// Symmetric overlap/density dressing `½(Ṽ X + X Ṽ)` restricted to the correlated
/// subspace: `X = S` gives the Fock potential `G`, `X = P` gives the overlap-Pulay
/// weight `Q`. Because `Ṽ` is non-zero only on correlated rows/columns, this costs
/// `O(|corr|² N)` and the result is the symmetric part of `Ṽ X`.
fn symm_dress_corr(corr_aos: &[usize], vtilde: &[f64], x: &Matrix) -> Matrix {
    let c = corr_aos.len();
    let n = x.rows();
    let mut g = Matrix::zeros(n, n);
    if c == 0 {
        return g;
    }
    // VX[ic][j] = Σ_kc vtilde[ic][kc] · X[corr_aos[kc]][j]  (|corr|×N).
    let mut vx = vec![0.0; c * n];
    for ic in 0..c {
        for (kc, &krow) in corr_aos.iter().enumerate() {
            let v = vtilde[ic * c + kc];
            if v == 0.0 {
                continue;
            }
            for j in 0..n {
                vx[ic * n + j] += v * x[(krow, j)];
            }
        }
    }
    // G_{ij} = ½(VX_{ij} + VX_{ji}); VX has non-zero rows only at correlated AOs.
    for (ic, &i) in corr_aos.iter().enumerate() {
        for j in 0..n {
            let val = 0.5 * vx[ic * n + j];
            g[(i, j)] += val;
            g[(j, i)] += val;
        }
    }
    g
}

/// One inter-site `+V` coupling: the two correlated atoms (as indices into the
/// subspace vector) and the inter-site Hubbard `V_AB` (Hartree).
#[derive(Clone, Debug)]
pub struct IntersitePair {
    /// Index of atom A into the `subspace` slice.
    pub a: usize,
    /// Index of atom B into the `subspace` slice.
    pub b: usize,
    /// Inter-site Hubbard `V_AB`, in Hartree.
    pub v: f64,
}

/// Extended-Hubbard (`+U` on-site + `+V` inter-site) energy and its variational
/// Fock potential for a single spin-channel density `p` (AO basis), overlap `s`.
///
/// ```text
/// E = Σ_A (U_A/2)[Tr n_AA − Tr (n_AA)²] − Σ_pairs V_AB Σ_{a∈A,b∈B} (n_AB)²
/// ```
///
/// with the dual population `n = ½(P S + S P)`. Returns `(E, G)` where `G` is
/// the `N × N` Fock contribution `½(Ṽ S + S Ṽ)` and `Ṽ` is the orbital-space
/// potential: on-site blocks `U(½ I − n_AA)`, cross blocks `−2 V_AB n_AB`. `G`
/// is exactly `∂E/∂p` (FD-verified for both the `U` and `V` parts).
///
/// Pass the **spin-channel** density (`P^α` or `P^β`); each channel carries the
/// same `U`/`V`. `pairs` reference atoms by their position in `subspace`.
pub fn plus_u_v(
    p: &Matrix,
    s: &Matrix,
    subspace: &[CorrelatedAtom],
    pairs: &[IntersitePair],
) -> (f64, Matrix) {
    let n = p.rows();
    if subspace.is_empty() {
        return (0.0, Matrix::zeros(n, n));
    }
    let (corr_aos, vtilde, energy) = build_vtilde_corr(p, s, subspace, pairs);
    let g = symm_dress_corr(&corr_aos, &vtilde, s); // G = ½(Ṽ S + S Ṽ)
    (energy, g)
}

/// Overlap-Pulay weight `Q = ∂E_{+U+V}/∂S = ½(P Ṽ + Ṽ P)` for one spin-channel
/// density `p`. Contracting `Σ_{μν} Q_{μν} dS_{μν}/dR` with the overlap
/// derivatives gives the **explicit** geometry dependence of the `+U+V` energy —
/// the only one, since at the SCC minimum the density response is already carried
/// by the energy-weighted density. FD-verified as `Q = ∂E/∂S`. `O(|corr|² N)`.
pub fn plus_u_v_overlap_weight(
    p: &Matrix,
    s: &Matrix,
    subspace: &[CorrelatedAtom],
    pairs: &[IntersitePair],
) -> Matrix {
    let n = p.rows();
    if subspace.is_empty() {
        return Matrix::zeros(n, n);
    }
    let (corr_aos, vtilde, _) = build_vtilde_corr(p, s, subspace, pairs);
    symm_dress_corr(&corr_aos, &vtilde, p) // Q = ½(Ṽ P + P Ṽ)
}

/// On-site `+U` (FLL) only — thin wrapper over [`plus_u_v`] with no inter-site
/// pairs. Returns `(E_U, G)` with `G = ∂E_U/∂p` (FD-verified).
///
/// Pass the **spin-channel** density (`P^α` or `P^β`); for a restricted closed
/// shell pass `½P` for each of the two channels (or call once per channel).
pub fn onsite_plus_u(p: &Matrix, s: &Matrix, subspace: &[CorrelatedAtom]) -> (f64, Matrix) {
    plus_u_v(p, s, subspace, &[])
}

/// Per-correlated-atom Mulliken occupation `Tr n_I` of the correlated shell,
/// `Tr n_I = Σ_{a∈I} (P S)_aa`. This is the quantity whose response to a
/// localised potential defines the linear-response Hubbard `U`.
pub fn subspace_occupations(p: &Matrix, s: &Matrix, subspace: &[CorrelatedAtom]) -> Vec<f64> {
    // (P S)_aa = Σ_k P[a][k] S[k][a], evaluated only for the correlated AOs — no
    // full P·S product (O(|corr| N) instead of O(N³)).
    let n = p.rows();
    subspace
        .iter()
        .map(|atom| {
            atom.aos
                .iter()
                .map(|&a| (0..n).map(|k| p[(a, k)] * s[(k, a)]).sum::<f64>())
                .sum()
        })
        .collect()
}

/// The FLL on-site penalty value `½ Tr[n_AA (1 − n_AA)] = ½[Tr n_AA − Tr n_AA²]`
/// per correlated atom, and the inter-site overlap `Tr[n_AB n_BA] = Σ (n_AB)²`
/// per pair, from one spin-channel density `p` (dual population `n = ½(PS+SP)`).
///
/// These are exactly the geometry-derivatives of the `+U+V` energy with respect
/// to the **Hubbard parameters** at fixed density:
///
/// ```text
/// ∂E/∂U_A  =  ½[Tr n_AA − Tr n_AA²]   (= `du[A]`, summed over the two spin channels)
/// ∂E/∂V_IJ = −Σ_{a∈I,b∈J} (n_IJ)²     (= `−dv[pair]`)
/// ```
///
/// (`build_vtilde_corr` defines `E_A = (U_A/2)[Tr n_AA − Tr n_AA²]`, so
/// `∂E_A/∂U_A` is the bracket halved; and `E_V = −V_IJ Σ(n_IJ)²`, so
/// `∂E_V/∂V_IJ = −Σ(n_IJ)²`.) `du[i]` is per correlated atom (indexed like
/// `subspace`); `dv[k]` is per pair (indexed like `pairs`, the **positive**
/// `Σ(n_IJ)²` — the caller applies the minus sign of `∂E/∂V`). Pass the
/// spin-channel density; the on-site `∂E/∂U` must be **summed over both spin
/// channels** to match the total energy. FD-verified against the central
/// difference of [`plus_u_v`] in `U`/`V`.
pub fn plus_u_param_derivatives(
    p: &Matrix,
    s: &Matrix,
    subspace: &[CorrelatedAtom],
    pairs: &[IntersitePair],
) -> (Vec<f64>, Vec<f64>) {
    let n = p.rows();
    let mut du = vec![0.0; subspace.len()];
    let mut dv = vec![0.0; pairs.len()];
    if subspace.is_empty() {
        return (du, dv);
    }
    // Correlated-AO index map and the dual population restricted to correlated AOs,
    // computed entry-wise (O(|corr|² N)) exactly as in `build_vtilde_corr`.
    let mut corr_aos: Vec<usize> = subspace.iter().flat_map(|a| a.aos.iter().copied()).collect();
    corr_aos.sort_unstable();
    corr_aos.dedup();
    let c = corr_aos.len();
    let mut pos = vec![usize::MAX; n];
    for (ic, &a) in corr_aos.iter().enumerate() {
        pos[a] = ic;
    }
    let mut mcorr = vec![0.0; c * c];
    for (ic, &a) in corr_aos.iter().enumerate() {
        for (jc, &b) in corr_aos.iter().enumerate() {
            let mut acc = 0.0;
            for k in 0..n {
                acc += p[(a, k)] * s[(k, b)];
            }
            mcorr[ic * c + jc] = acc;
        }
    }
    let dual = |ic: usize, jc: usize| 0.5 * (mcorr[ic * c + jc] + mcorr[jc * c + ic]);
    for (ia_sub, atom) in subspace.iter().enumerate() {
        let mut tr_n = 0.0;
        let mut tr_n2 = 0.0;
        for &a in &atom.aos {
            let ia = pos[a];
            tr_n += dual(ia, ia);
            for &b in &atom.aos {
                let nv = dual(ia, pos[b]);
                tr_n2 += nv * nv;
            }
        }
        du[ia_sub] = 0.5 * (tr_n - tr_n2);
    }
    for (ip, pair) in pairs.iter().enumerate() {
        let mut sumsq = 0.0;
        for &a in &subspace[pair.a].aos {
            let ia = pos[a];
            for &b in &subspace[pair.b].aos {
                let ib = pos[b];
                let nv = dual(ia, ib);
                sumsq += nv * nv;
            }
        }
        dv[ip] = sumsq;
    }
    (du, dv)
}

/// Build the inter-site `+V` pairs from the correlated subspace: every pair of
/// correlated atoms within `cutoff` (bohr) whose element pair has a non-zero
/// entry in `v_by_pair` (matched unordered). The returned pairs index into
/// `subspace`. `positions` / `z` are per-system-atom (indexed by `atom_index`).
pub fn intersite_pairs(
    subspace: &[CorrelatedAtom],
    positions: &[Vec3],
    z: &[u8],
    v_by_pair: &[(u8, u8, f64)],
    cutoff: f64,
) -> Vec<IntersitePair> {
    let v_of = |za: u8, zb: u8| -> f64 {
        v_by_pair
            .iter()
            .find(|(a, b, _)| (*a == za && *b == zb) || (*a == zb && *b == za))
            .map(|(_, _, v)| *v)
            .unwrap_or(0.0)
    };
    let mut pairs = Vec::new();
    for i in 0..subspace.len() {
        for j in (i + 1)..subspace.len() {
            let ai = subspace[i].atom_index;
            let aj = subspace[j].atom_index;
            let v = v_of(z[ai], z[aj]);
            if v == 0.0 {
                continue;
            }
            let (pi, pj) = (&positions[ai], &positions[aj]);
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let dz = pi.z - pj.z;
            if (dx * dx + dy * dy + dz * dz).sqrt() <= cutoff {
                pairs.push(IntersitePair { a: i, b: j, v });
            }
        }
    }
    pairs
}

/// Linear-response (Cococcioni–de Gironcoli) on-site Hubbard parameter from the
/// bare and self-consistent occupation responses of one correlated site:
///
/// ```text
/// U = χ0⁻¹ − χ⁻¹
/// ```
///
/// where `χ0 = dn/dα` is the **bare** (one-shot, fixed-potential) response and
/// `χ = dn/dα` the **self-consistent** (fully re-converged) response of the
/// site's correlated occupation `n` to a localised potential shift `α` on that
/// subspace. Both responses are negative (raising the on-site potential depletes
/// the occupation), and `χ⁻¹ < χ0⁻¹`, so `U > 0`.
pub fn linear_response_u(chi0: f64, chi: f64) -> f64 {
    1.0 / chi0 - 1.0 / chi
}

/// Per-atom linear-response Hubbard `U` from central-difference occupation
/// responses. The four slices give, per correlated atom `I`, the occupation
/// `Tr n_I` after applying a `±delta` localised potential to atom `I`'s own
/// correlated subspace, from a **bare** one-shot solve (`occ_bare_*`, → `χ0`)
/// and a fully **self-consistent** re-converged solve (`occ_scc_*`, → `χ`):
///
/// ```text
/// χ0_I = (occ_bare_plus_I − occ_bare_minus_I) / (2 delta)
/// χ_I  = (occ_scc_plus_I  − occ_scc_minus_I ) / (2 delta)
/// U_I  = χ0_I⁻¹ − χ_I⁻¹
/// ```
///
/// This is the **single-site (diagonal)** approximation: each `U_I` uses only
/// the on-site response. The fully screened parameter additionally inverts the
/// inter-site response matrix `χ_IJ`; that refinement is left for later (it needs
/// the cross-site occupation responses collected the same way). A non-finite or
/// vanishing `χ0`/`χ` yields `U_I = 0` (the site is treated as uncorrected).
pub fn linear_response_u_from_responses(
    occ_bare_plus: &[f64],
    occ_bare_minus: &[f64],
    occ_scc_plus: &[f64],
    occ_scc_minus: &[f64],
    delta: f64,
) -> Vec<f64> {
    let n = occ_bare_plus.len();
    (0..n)
        .map(|i| {
            let chi0 = (occ_bare_plus[i] - occ_bare_minus[i]) / (2.0 * delta);
            let chi = (occ_scc_plus[i] - occ_scc_minus[i]) / (2.0 * delta);
            if chi0 == 0.0 || chi == 0.0 {
                return 0.0;
            }
            let u = linear_response_u(chi0, chi);
            if u.is_finite() {
                u
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(n: usize, vals: &[f64]) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = vals[i * n + j];
            }
        }
        // symmetrise to be safe
        for i in 0..n {
            for j in 0..i {
                let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = avg;
                m[(j, i)] = avg;
            }
        }
        m
    }

    fn frob_dot(a: &Matrix, b: &Matrix) -> f64 {
        let mut s = 0.0;
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                s += a[(i, j)] * b[(i, j)];
            }
        }
        s
    }

    /// Toy non-orthogonal 5-AO system: one correlated atom occupying AOs {1,2,3}
    /// (a 3-function "d-like" block), U = 0.15 Ha. Verify the returned Fock
    /// contribution G equals ∂E_U/∂P by central finite difference:
    /// (E[P+εΔ] − E[P−εΔ]) / 2ε  ==  Tr(G Δ)  for a random symmetric Δ.
    #[test]
    fn onsite_plus_u_potential_matches_finite_difference() {
        let nn = 5;
        // Overlap: SPD, off-diagonal (non-orthogonal) to exercise the S-dressing.
        let s = sym(
            nn,
            &[
                1.00, 0.12, 0.05, 0.00, 0.08, //
                0.12, 1.00, 0.18, 0.06, 0.00, //
                0.05, 0.18, 1.00, 0.20, 0.04, //
                0.00, 0.06, 0.20, 1.00, 0.10, //
                0.08, 0.00, 0.04, 0.10, 1.00, //
            ],
        );
        let p = sym(
            nn,
            &[
                0.90, 0.20, 0.10, 0.05, 0.15, //
                0.20, 0.70, 0.25, 0.10, 0.05, //
                0.10, 0.25, 0.60, 0.30, 0.10, //
                0.05, 0.10, 0.30, 0.50, 0.20, //
                0.15, 0.05, 0.10, 0.20, 0.80, //
            ],
        );
        let subspace = vec![CorrelatedAtom {
            atom_index: 0,
            aos: vec![1, 2, 3],
            u: 0.15,
        }];

        let (e0, g) = onsite_plus_u(&p, &s, &subspace);
        assert!(e0.abs() > 1e-6, "energy should be non-trivial, got {e0}");

        // Random symmetric perturbation direction.
        let delta = sym(
            nn,
            &[
                0.03, -0.02, 0.01, 0.04, -0.01, //
                -0.02, 0.05, -0.03, 0.02, 0.01, //
                0.01, -0.03, 0.04, -0.02, 0.03, //
                0.04, 0.02, -0.02, 0.06, -0.04, //
                -0.01, 0.01, 0.03, -0.04, 0.02, //
            ],
        );
        let eps = 1e-5;
        let mut pp = p.clone();
        let mut pm = p.clone();
        for i in 0..nn {
            for j in 0..nn {
                pp[(i, j)] = p[(i, j)] + eps * delta[(i, j)];
                pm[(i, j)] = p[(i, j)] - eps * delta[(i, j)];
            }
        }
        let (ep, _) = onsite_plus_u(&pp, &s, &subspace);
        let (em, _) = onsite_plus_u(&pm, &s, &subspace);
        let fd = (ep - em) / (2.0 * eps);
        let analytic = frob_dot(&g, &delta);
        assert!(
            (fd - analytic).abs() < 1e-7,
            "∂E/∂P mismatch: FD {fd:.3e} vs Tr(GΔ) {analytic:.3e}"
        );
    }

    /// Empty subspace (or zero U) must be an exact no-op: zero energy, zero Fock.
    #[test]
    fn empty_subspace_is_noop() {
        let s = Matrix::identity(3);
        let p = sym(3, &[1.0, 0.1, 0.0, 0.1, 0.8, 0.05, 0.0, 0.05, 0.6]);
        let (e, g) = onsite_plus_u(&p, &s, &[]);
        assert_eq!(e, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(g[(i, j)], 0.0);
            }
        }
    }

    /// A fully occupied (n = 1) or empty (n = 0) idempotent correlated block has
    /// Tr[n(1−n)] = 0, so the FLL energy must vanish — the hallmark of the FLL
    /// penalty acting only on *fractional* occupations.
    #[test]
    fn integer_occupation_has_zero_energy() {
        // Orthogonal (S = I) so the dual population is just the P block.
        let s = Matrix::identity(3);
        // Idempotent P over the whole space with the correlated block {0,1,2}:
        // a rank-controlled projector P = v vᵀ (v normalised) is idempotent and
        // its block has Tr[n(1−n)] = 0 only if the block is the whole support.
        // Simplest exact case: P diagonal with entries 1,1,0 → integer occ.
        let mut p = Matrix::zeros(3, 3);
        p[(0, 0)] = 1.0;
        p[(1, 1)] = 1.0;
        p[(2, 2)] = 0.0;
        let subspace = vec![CorrelatedAtom {
            atom_index: 0,
            aos: vec![0, 1, 2],
            u: 0.2,
        }];
        let (e, _) = onsite_plus_u(&p, &s, &subspace);
        assert!(e.abs() < 1e-12, "integer-occupation FLL energy must vanish, got {e}");
    }

    /// Combined +U+V: two correlated atoms (AOs {0,1} and {3,4}) with an
    /// inter-site V pair, on a non-orthogonal 6-AO system. Verify the full Fock
    /// contribution G = ∂E/∂P (both U and V parts) by central finite difference.
    #[test]
    fn plus_u_v_potential_matches_finite_difference() {
        let nn = 6;
        let s = sym(
            nn,
            &[
                1.00, 0.15, 0.04, 0.07, 0.02, 0.05, //
                0.15, 1.00, 0.10, 0.03, 0.06, 0.01, //
                0.04, 0.10, 1.00, 0.12, 0.08, 0.03, //
                0.07, 0.03, 0.12, 1.00, 0.16, 0.04, //
                0.02, 0.06, 0.08, 0.16, 1.00, 0.11, //
                0.05, 0.01, 0.03, 0.04, 0.11, 1.00, //
            ],
        );
        let p = sym(
            nn,
            &[
                0.85, 0.22, 0.10, 0.18, 0.06, 0.09, //
                0.22, 0.75, 0.14, 0.12, 0.20, 0.05, //
                0.10, 0.14, 0.65, 0.16, 0.11, 0.07, //
                0.18, 0.12, 0.16, 0.70, 0.24, 0.13, //
                0.06, 0.20, 0.11, 0.24, 0.80, 0.17, //
                0.09, 0.05, 0.07, 0.13, 0.17, 0.60, //
            ],
        );
        let subspace = vec![
            CorrelatedAtom { atom_index: 0, aos: vec![0, 1], u: 0.18 },
            CorrelatedAtom { atom_index: 1, aos: vec![3, 4], u: 0.12 },
        ];
        let pairs = vec![IntersitePair { a: 0, b: 1, v: 0.06 }];

        let (e0, g) = plus_u_v(&p, &s, &subspace, &pairs);
        let (e_u_only, _) = onsite_plus_u(&p, &s, &subspace);
        assert!(
            (e0 - e_u_only).abs() > 1e-6,
            "the +V term must move the energy: E(U+V) {e0:.6} vs E(U) {e_u_only:.6}"
        );

        let delta = sym(
            nn,
            &[
                0.02, -0.03, 0.01, 0.04, -0.02, 0.01, //
                -0.03, 0.05, -0.01, 0.02, 0.03, -0.02, //
                0.01, -0.01, 0.03, -0.02, 0.01, 0.04, //
                0.04, 0.02, -0.02, 0.06, -0.03, 0.01, //
                -0.02, 0.03, 0.01, -0.03, 0.04, -0.01, //
                0.01, -0.02, 0.04, 0.01, -0.01, 0.02, //
            ],
        );
        let eps = 1e-5;
        let mut pp = p.clone();
        let mut pm = p.clone();
        for i in 0..nn {
            for j in 0..nn {
                pp[(i, j)] = p[(i, j)] + eps * delta[(i, j)];
                pm[(i, j)] = p[(i, j)] - eps * delta[(i, j)];
            }
        }
        let (ep, _) = plus_u_v(&pp, &s, &subspace, &pairs);
        let (em, _) = plus_u_v(&pm, &s, &subspace, &pairs);
        let fd = (ep - em) / (2.0 * eps);
        let analytic = frob_dot(&g, &delta);
        assert!(
            (fd - analytic).abs() < 1e-7,
            "∂E/∂P mismatch (U+V): FD {fd:.3e} vs Tr(GΔ) {analytic:.3e}"
        );
    }

    /// With S = I the correlated-shell occupation is just the sum of the block's
    /// diagonal density — the Mulliken population of the shell.
    #[test]
    fn subspace_occupations_orthogonal_is_block_population() {
        let s = Matrix::identity(5);
        let p = sym(
            5,
            &[
                0.90, 0.20, 0.10, 0.05, 0.15, //
                0.20, 0.70, 0.25, 0.10, 0.05, //
                0.10, 0.25, 0.60, 0.30, 0.10, //
                0.05, 0.10, 0.30, 0.50, 0.20, //
                0.15, 0.05, 0.10, 0.20, 0.80, //
            ],
        );
        let subspace = vec![CorrelatedAtom { atom_index: 0, aos: vec![1, 2, 3], u: 0.1 }];
        let occ = subspace_occupations(&p, &s, &subspace);
        let expected = p[(1, 1)] + p[(2, 2)] + p[(3, 3)];
        assert!((occ[0] - expected).abs() < 1e-12, "occ {} vs {expected}", occ[0]);
    }

    /// Linear-response U = χ0⁻¹ − χ⁻¹ from synthetic central-difference responses:
    /// bare χ0 = −2.0, self-consistent χ = −1.0 → U = −0.5 + 1.0 = 0.5 Ha.
    #[test]
    fn linear_response_u_from_synthetic_responses() {
        let delta = 0.01;
        // χ0 = (0.98 − 1.02)/(2·0.01) = −2.0 ; χ = (0.99 − 1.01)/(2·0.01) = −1.0.
        let u = linear_response_u_from_responses(&[0.98], &[1.02], &[0.99], &[1.01], delta);
        assert!((u[0] - 0.5).abs() < 1e-12, "U {} vs 0.5", u[0]);
        // Direct formula must agree.
        assert!((linear_response_u(-2.0, -1.0) - 0.5).abs() < 1e-12);
        // Degenerate (zero) response → uncorrected site (U = 0), no NaN/Inf.
        let z = linear_response_u_from_responses(&[1.0], &[1.0], &[0.99], &[1.01], delta);
        assert_eq!(z[0], 0.0);
    }

    /// Gauss–Jordan inverse: M · M⁻¹ = I for a non-trivial 2×2.
    #[test]
    fn invert_small_is_correct() {
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_small(&m).unwrap();
        // Product M·inv must be the identity.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += m[i][k] * inv[k][j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-12, "({i},{j}) = {acc}");
            }
        }
        // Singular → None.
        assert!(invert_small(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
    }

    /// Full inter-site extraction K = χ0⁻¹ − χ⁻¹, U_I = K_II, V_IJ = −K_IJ.
    /// 1×1 reduces to the scalar linear_response_u; a 2×2 with off-diagonal χ
    /// coupling produces a non-zero V.
    #[test]
    fn extract_uv_scalar_and_pair() {
        // Scalar: χ0 = −2, χ = −1 → U ≈ 0.5 (within the Tikhonov λ=1e-3 shift).
        let (u, v) = extract_uv_from_response(&[vec![-2.0]], &[vec![-1.0]]);
        assert!((u[0] - 0.5).abs() < 3e-3, "U {}", u[0]);
        assert_eq!(v.len(), 1);
        // Pair: χ0 = −2 I, χ = [[-1,0.2],[0.2,-1]] → U ≈ 0.5417, V_01 ≈ −0.2083.
        let chi0 = vec![vec![-2.0, 0.0], vec![0.0, -2.0]];
        let chi = vec![vec![-1.0, 0.2], vec![0.2, -1.0]];
        let (u2, v2) = extract_uv_from_response(&chi0, &chi);
        assert!((u2[0] - 0.5416667).abs() < 3e-3, "U {}", u2[0]);
        assert!((v2[0][1] - (-0.2083333)).abs() < 3e-3, "V {}", v2[0][1]);
    }

    /// ILL-CONDITIONED (degenerate-d¹-like) robustness: a near-singular response χ
    /// must yield a finite, clamped U (no NaN/Inf, within [0, U_MAX]) thanks to the
    /// Tikhonov regularisation + physical clamp — not garbage.
    #[test]
    fn extract_uv_ill_conditioned_is_finite_and_clamped() {
        // χ near-singular (tiny diagonal), χ0 ordinary: the bare difference of
        // inverses would explode; the regularised+clamped result must stay sane.
        let chi0 = vec![vec![-1.0]];
        let chi = vec![vec![-1.0e-9]];
        let (u, v) = extract_uv_from_response(&chi0, &chi);
        assert!(u[0].is_finite() && (0.0..=1.0).contains(&u[0]), "U {}", u[0]);
        assert!(v[0][0].is_finite());
        // A genuinely singular (all-zero) response → uncorrected (zero), not NaN.
        let (uz, _) = extract_uv_from_response(&vec![vec![0.0]], &vec![vec![0.0]]);
        assert!(uz[0].is_finite());
    }

    /// Overlap-Pulay weight Q = ∂E/∂S: perturbing the overlap S along a symmetric
    /// direction Δ must change E_{+U+V} by Tr(Q Δ) (the explicit S-dependence that
    /// becomes the analytic nuclear-gradient term after contracting with dS/dR).
    #[test]
    fn plus_u_v_overlap_weight_matches_finite_difference() {
        let nn = 6;
        let s = sym(
            nn,
            &[
                1.00, 0.15, 0.04, 0.07, 0.02, 0.05, //
                0.15, 1.00, 0.10, 0.03, 0.06, 0.01, //
                0.04, 0.10, 1.00, 0.12, 0.08, 0.03, //
                0.07, 0.03, 0.12, 1.00, 0.16, 0.04, //
                0.02, 0.06, 0.08, 0.16, 1.00, 0.11, //
                0.05, 0.01, 0.03, 0.04, 0.11, 1.00, //
            ],
        );
        let p = sym(
            nn,
            &[
                0.85, 0.22, 0.10, 0.18, 0.06, 0.09, //
                0.22, 0.75, 0.14, 0.12, 0.20, 0.05, //
                0.10, 0.14, 0.65, 0.16, 0.11, 0.07, //
                0.18, 0.12, 0.16, 0.70, 0.24, 0.13, //
                0.06, 0.20, 0.11, 0.24, 0.80, 0.17, //
                0.09, 0.05, 0.07, 0.13, 0.17, 0.60, //
            ],
        );
        let subspace = vec![
            CorrelatedAtom { atom_index: 0, aos: vec![0, 1], u: 0.18 },
            CorrelatedAtom { atom_index: 1, aos: vec![3, 4], u: 0.12 },
        ];
        let pairs = vec![IntersitePair { a: 0, b: 1, v: 0.06 }];
        let q = plus_u_v_overlap_weight(&p, &s, &subspace, &pairs);
        let delta = sym(
            nn,
            &[
                0.00, 0.03, -0.02, 0.01, 0.02, -0.01, //
                0.03, 0.00, 0.04, -0.02, 0.01, 0.03, //
                -0.02, 0.04, 0.00, 0.02, -0.03, 0.01, //
                0.01, -0.02, 0.02, 0.00, 0.04, -0.02, //
                0.02, 0.01, -0.03, 0.04, 0.00, 0.02, //
                -0.01, 0.03, 0.01, -0.02, 0.02, 0.00, //
            ],
        );
        let eps = 1e-5;
        let mut sp = s.clone();
        let mut sm = s.clone();
        for i in 0..nn {
            for j in 0..nn {
                sp[(i, j)] = s[(i, j)] + eps * delta[(i, j)];
                sm[(i, j)] = s[(i, j)] - eps * delta[(i, j)];
            }
        }
        let (ep, _) = plus_u_v(&p, &sp, &subspace, &pairs);
        let (em, _) = plus_u_v(&p, &sm, &subspace, &pairs);
        let fd = (ep - em) / (2.0 * eps);
        let analytic = frob_dot(&q, &delta);
        assert!(
            (fd - analytic).abs() < 1e-7,
            "∂E/∂S mismatch: FD {fd:.3e} vs Tr(QΔ) {analytic:.3e}"
        );
    }

    /// STAGE 1 (Hubbard-parameter derivatives): `∂E/∂U_I` and `∂E/∂V_IJ` from
    /// [`plus_u_param_derivatives`] must equal the central finite difference of the
    /// `+U+V` energy in each `U_I` / `V_IJ`. These are the Hellmann–Feynman partials
    /// that, contracted with `dU/dR` / `dV/dR`, give the consistent-force `F_corr`.
    #[test]
    fn plus_u_param_derivatives_match_finite_difference() {
        let nn = 6;
        let s = sym(
            nn,
            &[
                1.00, 0.15, 0.04, 0.07, 0.02, 0.05, //
                0.15, 1.00, 0.10, 0.03, 0.06, 0.01, //
                0.04, 0.10, 1.00, 0.12, 0.08, 0.03, //
                0.07, 0.03, 0.12, 1.00, 0.16, 0.04, //
                0.02, 0.06, 0.08, 0.16, 1.00, 0.11, //
                0.05, 0.01, 0.03, 0.04, 0.11, 1.00, //
            ],
        );
        let p = sym(
            nn,
            &[
                0.85, 0.22, 0.10, 0.18, 0.06, 0.09, //
                0.22, 0.75, 0.14, 0.12, 0.20, 0.05, //
                0.10, 0.14, 0.65, 0.16, 0.11, 0.07, //
                0.18, 0.12, 0.16, 0.70, 0.24, 0.13, //
                0.06, 0.20, 0.11, 0.24, 0.80, 0.17, //
                0.09, 0.05, 0.07, 0.13, 0.17, 0.60, //
            ],
        );
        let subspace = vec![
            CorrelatedAtom { atom_index: 0, aos: vec![0, 1], u: 0.18 },
            CorrelatedAtom { atom_index: 1, aos: vec![3, 4], u: 0.12 },
        ];
        let pairs = vec![IntersitePair { a: 0, b: 1, v: 0.06 }];
        let (du, dv) = plus_u_param_derivatives(&p, &s, &subspace, &pairs);
        let eps = 1e-6;
        // ∂E/∂U_i for each correlated atom.
        for i in 0..subspace.len() {
            let mut sp_plus = subspace.clone();
            let mut sp_minus = subspace.clone();
            sp_plus[i].u += eps;
            sp_minus[i].u -= eps;
            let (ep, _) = plus_u_v(&p, &s, &sp_plus, &pairs);
            let (em, _) = plus_u_v(&p, &s, &sp_minus, &pairs);
            let fd = (ep - em) / (2.0 * eps);
            assert!(
                (fd - du[i]).abs() < 1e-8,
                "∂E/∂U_{i} mismatch: FD {fd:.3e} vs analytic {:.3e}",
                du[i]
            );
        }
        // ∂E/∂V_IJ = −Σ(n_IJ)²; plus_u_param_derivatives returns +Σ(n_IJ)² in dv.
        for (k, _pair) in pairs.iter().enumerate() {
            let mut pp = pairs.clone();
            let mut pm = pairs.clone();
            pp[k].v += eps;
            pm[k].v -= eps;
            let (ep, _) = plus_u_v(&p, &s, &subspace, &pp);
            let (em, _) = plus_u_v(&p, &s, &subspace, &pm);
            let fd = (ep - em) / (2.0 * eps);
            assert!(
                (fd - (-dv[k])).abs() < 1e-8,
                "∂E/∂V_{k} mismatch: FD {fd:.3e} vs analytic {:.3e}",
                -dv[k]
            );
        }
    }
}
