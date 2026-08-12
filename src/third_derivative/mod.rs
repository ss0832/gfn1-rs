// SPDX-License-Identifier: GPL-3.0-or-later
//! Non-PBC analytic **third nuclear derivative** `T_{abc} = ∂³E/∂R_a∂R_b∂R_c` (cubic force
//! constants), assembled via the **2n+1 rule** so it reuses the *first-order* CPHF response
//! already computed for the analytic Hessian. This module currently provides the shared
//! central rank-3 block helper used by the purely-geometric pair terms (repulsion, the
//! frozen SCC electrostatic kernel, …); the full electronic 2n+1 driver and the
//! Dense/Block/Vector output modes build on top of it.
//!
//! Validation respects stationarity: only the *total* electronic Lagrangian is stationary,
//! so geometric / frozen-response terms may be FD-validated in isolation against their own
//! analytic Hessian block, but the response-carrying electronic terms must be validated as
//! the whole stationary bundle.

use crate::electronic::ElectronicResult;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

/// Symmetric-packed storage for a fully-symmetric third-derivative tensor `T[i][j][k]`
/// (`T_{abc}` is invariant under permutations of its three nuclear DOF indices). Only the
/// `n(n+1)(n+2)/6` canonical entries `i ≤ j ≤ k` are stored, a ~6x memory and
/// accumulation-cost reduction over the dense `n³`. The Dense output mode and the per-pair
/// assemblies use this to exploit the permutation symmetry.
pub struct SymmetricThird {
    n: usize,
    data: Vec<f64>,
}

impl SymmetricThird {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * (n + 1) * (n + 2) / 6],
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Canonical packed index of `(i,j,k)` (order-independent): sort to `a ≤ b ≤ c`, then the
    /// combinatorial-number-system offset `C(c+2,3) + C(b+1,2) + a`.
    #[inline]
    fn index(i: usize, j: usize, k: usize) -> usize {
        let mut t = [i, j, k];
        t.sort_unstable();
        let (a, b, c) = (t[0], t[1], t[2]);
        c * (c + 1) * (c + 2) / 6 + b * (b + 1) / 2 + a
    }

    /// Accumulate `v` into the symmetric slot for `(i,j,k)` (any index order).
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, k: usize, v: f64) {
        let idx = Self::index(i, j, k);
        self.data[idx] += v;
    }

    /// Read the symmetric entry `T[i][j][k]` (any index order).
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[Self::index(i, j, k)]
    }

    /// Scale all stored entries by `factor` in place (packed data, no `n³` traversal). Used to apply
    /// the `1/6` permutation average when a fully-symmetric tensor is accumulated from all `n³` orderings
    /// of an only-partially-symmetric producer (the closed-form 2n+1 slabs).
    pub fn scale(&mut self, factor: f64) {
        for v in self.data.iter_mut() {
            *v *= factor;
        }
    }

    /// Add another store into this one (both must share `n`). Operates directly on the
    /// packed `n(n+1)(n+2)/6` data -- no `n³` traversal, the symmetry paying off again.
    pub fn add_from(&mut self, other: &SymmetricThird) {
        assert_eq!(self.n, other.n, "SymmetricThird size mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// Materialize the full tensor as `n` dense `n×n` slabs (`slab[c][(a,b)] = T_abc`) -- the
    /// **Dense output mode** / interop with the per-slab Hessian-derivative producers.
    pub fn to_dense_slabs(&self) -> Vec<Matrix> {
        let n = self.n;
        let mut slabs = vec![Matrix::zeros(n, n); n];
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    slabs[c][(a, b)] = self.get(a, b, c);
                }
            }
        }
        slabs
    }

    /// The fully-contracted scalar `T[v,v,v] = Σ_abc T_abc v_a v_b v_c` -- the cubic anharmonicity
    /// along a single direction `v` (e.g. a normal mode). The cheapest directional output:
    /// truly a single contracted quantity (and, by 2n+1, the only one needing just `x_v`).
    pub fn contract_vvv(&self, v: &[f64]) -> f64 {
        let n = self.n;
        let mut s = 0.0;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    s += self.get(a, b, c) * v[a] * v[b] * v[c];
                }
            }
        }
        s
    }

    /// **Block output mode**: the `|dofs|³` sub-tensor restricted to the given DOF indices,
    /// returned as `|dofs|` dense slabs (`slab[c][(a,b)] = T[dofs[a]][dofs[b]][dofs[c]]`). For
    /// local anharmonicity over a chosen atom subset.
    pub fn block(&self, dofs: &[usize]) -> Vec<Matrix> {
        let m = dofs.len();
        let mut slabs = vec![Matrix::zeros(m, m); m];
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    slabs[c][(a, b)] = self.get(dofs[a], dofs[b], dofs[c]);
                }
            }
        }
        slabs
    }

    /// Contract one index with a vector `v`: the Hessian-shaped matrix `K[a][b] = Σ_c T_abc v_c`
    /// (the **Vector output mode** -- the directional third derivative along `v`), read directly
    /// from the symmetric store.
    pub fn contract_last(&self, v: &[f64]) -> Matrix {
        let n = self.n;
        let mut m = Matrix::zeros(n, n);
        for a in 0..n {
            for b in 0..n {
                let mut s = 0.0;
                for c in 0..n {
                    s += self.get(a, b, c) * v[c];
                }
                m[(a, b)] = s;
            }
        }
        m
    }
}

/// Add the rank-3 central block of a radial pair function `f(r)` to `tensor` (one
/// `ndof×ndof` slab per third index `c`). `rel = R_i − R_j` is the true relative vector,
/// `g = f''/r − f'/r²`, and `f3 = f'''`. The relative-vector third derivative is
///
/// ```text
///   T_rel[a][b][c] = (f''' − 3g) u_a u_b u_c + g (δ_ab u_c + δ_ac u_b + δ_bc u_a),  u = rel/r
/// ```
///
/// distributed over the two atoms with sign `σ_X σ_Y σ_Z` (`+1` for atom `i`, `−1` for
/// `j`, since `∂rel/∂R_i = +I`). `scale` multiplies the whole block (e.g. `q_i q_j` for the
/// shell-charge electrostatics; `1` for repulsion). Slab `c` then equals `∂(Hessian)/∂R_c`.
pub(crate) fn add_radial_third_block(
    tensor: &mut [Matrix],
    i: usize,
    j: usize,
    rel: Vec3,
    g: f64,
    f3: f64,
    scale: f64,
) {
    let r = rel.norm();
    if r <= 1.0e-12 || scale == 0.0 {
        return;
    }
    let u = (rel / r).to_array();
    let coeff_uuu = (f3 - 3.0 * g) * scale;
    let gs = g * scale;
    let atoms = [i, j];
    let signs = [1.0_f64, -1.0_f64];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let kron = if a == b { u[c] } else { 0.0 }
                    + if a == c { u[b] } else { 0.0 }
                    + if b == c { u[a] } else { 0.0 };
                let t_rel = coeff_uuu * u[a] * u[b] * u[c] + gs * kron;
                if t_rel == 0.0 {
                    continue;
                }
                for (xi, &ax) in atoms.iter().enumerate() {
                    for (yi, &ay) in atoms.iter().enumerate() {
                        for (zi, &az) in atoms.iter().enumerate() {
                            let value = signs[xi] * signs[yi] * signs[zi] * t_rel;
                            tensor[3 * az + c][(3 * ax + a, 3 * ay + b)] += value;
                        }
                    }
                }
            }
        }
    }
}

/// Symmetric-packed counterpart of [`add_radial_third_block`]: accumulates the same central
/// rank-3 block into a [`SymmetricThird`] store, writing each unordered index-triple **once**
/// (the canonical `i ≤ j ≤ k` representative). `store.get(a,b,c)` then equals the dense
/// `tensor[c][(a,b)]`, at ~6x less memory.
#[cfg(test)]
pub(crate) fn add_radial_third_block_sym(
    store: &mut SymmetricThird,
    i: usize,
    j: usize,
    rel: Vec3,
    g: f64,
    f3: f64,
    scale: f64,
) {
    let r = rel.norm();
    if r <= 1.0e-12 || scale == 0.0 {
        return;
    }
    let u = (rel / r).to_array();
    let coeff_uuu = (f3 - 3.0 * g) * scale;
    let gs = g * scale;
    let atoms = [i, j];
    let signs = [1.0_f64, -1.0_f64];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let kron = if a == b { u[c] } else { 0.0 }
                    + if a == c { u[b] } else { 0.0 }
                    + if b == c { u[a] } else { 0.0 };
                let t_rel = coeff_uuu * u[a] * u[b] * u[c] + gs * kron;
                if t_rel == 0.0 {
                    continue;
                }
                for (xi, &ax) in atoms.iter().enumerate() {
                    for (yi, &ay) in atoms.iter().enumerate() {
                        for (zi, &az) in atoms.iter().enumerate() {
                            let (ii, jj, kk) = (3 * ax + a, 3 * ay + b, 3 * az + c);
                            // Each unordered triple is written once (sorted representative),
                            // exploiting the permutation symmetry of the tensor.
                            if ii <= jj && jj <= kk {
                                store.add(ii, jj, kk, signs[xi] * signs[yi] * signs[zi] * t_rel);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Driver: the **geometric (response-free) blocks** of the non-PBC third derivative -- the
/// classical repulsion and halogen terms -- assembled into a symmetric-packed
/// [`SymmetricThird`]. These carry no electronic response, so they FD-validate as a bundle
/// against the sum of their analytic Hessians. The electronic blocks (frozen SCC/Pulay/H0 +
/// CN-H0 + D3 + the CPHF response) are added on top in the full 2n+1 driver.
pub fn third_derivative_geometric(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
) -> Result<SymmetricThird> {
    let rep = crate::repulsion::repulsion_third_derivative(system, params)?;
    let hal = crate::halogen::halogen_third_derivative(system, params)?;
    let n = rep.len();
    let mut store = SymmetricThird::zeros(n);
    // Read each canonical (i ≤ j ≤ k) entry once (the slabs hold the full symmetric tensor).
    for k in 0..n {
        for j in 0..=k {
            for i in 0..=j {
                store.add(i, j, k, rep[k][(i, j)] + hal[k][(i, j)]);
            }
        }
    }
    Ok(store)
}

/// Driver: the **frozen electronic `L_abc` blocks** -- the second-order SCC electrostatics
/// (frozen shell charges) and the band/H0/overlap+SCC-shift Pulay term (frozen density) -- /// assembled into the symmetric-packed store. Frozen => no response, so they FD-validate as a
/// bundle against the sum of their frozen analytic Hessians. The CN-H0/D3 frozen blocks and
/// the CPHF response cross-terms are added on top in the full 2n+1 driver.
pub fn third_derivative_frozen_electronic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<SymmetricThird> {
    let scc = crate::hessian::fixed_shell_charge_scc_third_derivative(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    let pulay = crate::hessian::fixed_density_pulay_third_derivative(system, params, electronic)?;
    let n = scc.len();
    let mut store = SymmetricThird::zeros(n);
    for k in 0..n {
        for j in 0..=k {
            for i in 0..=j {
                store.add(i, j, k, scc[k][(i, j)] + pulay[k][(i, j)]);
            }
        }
    }
    Ok(store)
}

/// Driver: the **complete frozen `L_abc` bundle** so far -- geometric (repulsion + halogen) +
/// frozen-electronic (SCC2 + Pulay) -- merged into one symmetric-packed store. This is the
/// frozen part of the 2n+1 third derivative; the CN-H0/D3 frozen blocks and the CPHF response
/// cross-terms are added on top to complete it.
pub fn third_derivative_frozen(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
) -> Result<SymmetricThird> {
    let mut store = third_derivative_geometric(system, params)?;
    let electronic_blocks = third_derivative_frozen_electronic(system, params, electronic)?;
    store.add_from(&electronic_blocks);
    Ok(store)
}

/// Driver: the **D3-BJ dispersion** frozen geometric block (`L_abc`), packed from the dense Jet3
/// third derivative ([`crate::dispersion::dispersion_third_derivative`]) into the symmetric store.
/// Carries the full many-body `C6(CN(R))` chain rule (forward-AD, no hand-coded Faà di Bruno) but
/// **no electronic response**, so -- like repulsion/halogen -- it FD-isolates against FD of the
/// analytic dispersion Hessian (validated in `dispersion.rs`). The Jet3 tensor is fully
/// permutation-symmetric, so reading the canonical `i ≤ j ≤ k` entries reproduces it exactly.
/// Added to the full 2n+1 bundle when dispersion is enabled.
pub fn third_derivative_dispersion(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    reference_path: Option<&str>,
) -> Result<SymmetricThird> {
    let disp = crate::dispersion::dispersion_third_derivative(system, params, reference_path)?;
    let n = disp.ndof;
    let mut store = SymmetricThird::zeros(n);
    for k in 0..n {
        for j in 0..=k {
            for i in 0..=j {
                store.add(i, j, k, disp.third[(i * n + j) * n + k]);
            }
        }
    }
    Ok(store)
}

/// Driver: the **frozen `L_abc` bundle including dispersion** -- geometric (repulsion + halogen) +
/// frozen-electronic (SCC2 + Pulay) + D3-BJ dispersion -- merged into one symmetric-packed store.
/// Every constituent here has a **fully-symmetric** per-block third derivative, so it FD-validates
/// as a bundle against the sum of their analytic Hessians. (The **CN-H0** frozen block is analytic
/// and FD-gated standalone -- [`crate::hessian::fixed_density_cn_h0_third_derivative`] -- but its
/// per-block tensor is symmetric only in `(b,c)` because it is `∂` of a *partial* Hessian block, so
/// it must be summed as a dense tensor with the other blocks *before* canonical packing; that
/// dense-sum assembly is the remaining bundle step, alongside the CPHF response cross-terms.)
pub fn third_derivative_frozen_full(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    reference_path: Option<&str>,
) -> Result<SymmetricThird> {
    let mut store = third_derivative_frozen(system, params, electronic)?;
    let disp = third_derivative_dispersion(system, params, reference_path)?;
    store.add_from(&disp);
    Ok(store)
}

/// Driver: the **complete frozen `L_abc` bundle including CN-H0** -- every frozen (response-free)
/// block of the non-PBC analytic third derivative: repulsion + halogen + SCC2 + Pulay + D3 +
/// **CN-H0** -- summed as **dense** slabs `slab[c][(a,b)] = ∂(frozen H_ab)/∂R_c`.
///
/// Returned **dense** (not symmetric-packed) on purpose: the frozen bundle is *deliberately*
/// asymmetric. GFN1's analytic Hessian holds the converged self-energy (hence `CN`) **fixed** in
/// the Pulay block, so the frozen blocks sum to a Hessian whose `∂³` is symmetric only in `(b,c)`,
/// not in the slab index -- the missing symmetrizing terms are supplied by the **CPHF response**
/// (the density's response to `CN(R)`). So `frozen (dense) + response (dense)` is the symmetric
/// total third derivative; symmetric-packing the frozen part alone would discard the asymmetry the
/// response must cancel. FD-validates exactly (per slab) against the full frozen Hessian.
///
/// **Dispersion gating (bug fix 2026-07-01):** the D3 dispersion 3rd derivative is included ONLY
/// when `include_dispersion` is true. Previously this always added `dispersion_third_derivative`
/// (which is nonzero even for `reference_path = None` — D3's built-in default reference), so the
/// analytic third derivative carried a spurious D3 term even when dispersion was disabled by the
/// caller, while the seminumerical ground truth (and the analytic Hessian) correctly excluded it.
/// At non-equilibrium/compressed geometries the D3 3rd derivative grows sharply and could dominate
/// the (small) total third derivative — the O(100%) non-EQ error. The Hessian gates dispersion the
/// same way (`include_dispersion && enable_dispersion`); this restores consistency.
pub fn third_derivative_frozen_complete(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    reference_path: Option<&str>,
    coordination_cutoff: f64,
    include_dispersion: bool,
) -> Result<Vec<Matrix>> {
    let ndof = 3 * system.atoms.len();
    let geo = third_derivative_geometric(system, params)?.to_dense_slabs();
    let scc = crate::hessian::fixed_shell_charge_scc_third_derivative(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    let pulay = crate::hessian::fixed_density_pulay_third_derivative(system, params, electronic)?;
    let disp = if include_dispersion {
        Some(crate::dispersion::dispersion_third_derivative(system, params, reference_path)?)
    } else {
        None
    };
    let cn_h0 = crate::hessian::fixed_density_cn_h0_third_derivative(
        system,
        params,
        electronic,
        coordination_cutoff,
    )?;
    let mut total = vec![Matrix::zeros(ndof, ndof); ndof];
    for c in 0..ndof {
        for a in 0..ndof {
            for b in 0..ndof {
                total[c][(a, b)] = geo[c][(a, b)]
                    + scc[c][(a, b)]
                    + pulay[c][(a, b)]
                    + disp.as_ref().map_or(0.0, |d| d.third[(a * ndof + b) * ndof + c])
                    + cn_h0[c][(a, b)];
            }
        }
    }
    Ok(total)
}

/// The metric / explicit-overlap **response-Hessian residual** `M_ab := R^code_ab − R^orb_ab =
/// cphf.hessian_response_ab + rhs_a·x_b`. Defined as the residual (NOT by which functions are "metric"):
/// `R^code = cphf.hessian_response` (the full, metric-inclusive response Hessian) minus the orbital
/// sector `R^orb = −rhs_a·x_b`. `rhs` and `x` come from the SAME solve (CPHF exposes `rhs_vectors`), so
/// the dot product is coordinate-consistent. Stage 1 FDs this across displaced geometries (it re-solves
/// CPHF — the temporary scaffold replaced by analytic `D_c M` in Stage 3).
///
/// Retained as a diagnostic: `third_derivative_analytic` no longer uses it (the response derivative is now
/// the strict-analytic [`closed_form_response_hessian_derivative`]).
#[allow(dead_code)]
fn cphf_metric_residual_hessian(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    ao_opts: crate::cphf::AoDerivativeOptions,
) -> Result<Matrix> {
    let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
        system,
        params,
        electronic,
        ao_opts,
        crate::cphf::CpxtbOptions::default(),
    )?;
    let ndof = 3 * system.atoms.len();
    let mut m = cphf.hessian_response.clone(); // R^code
    for a in 0..ndof {
        for b in 0..ndof {
            let dot: f64 = cphf.rhs_vectors[a]
                .iter()
                .zip(cphf.solutions[b].amplitudes.iter())
                .map(|(r, x)| r * x)
                .sum();
            m[(a, b)] += dot; // M = R^code + rhs_a·x_b
        }
    }
    Ok(m)
}

/// **STRICT closed-form 2n+1 nuclear third derivative** `T_abc = ∂³E/∂R_a∂R_b∂R_c` (cubic force constants).
/// NO finite differences anywhere. `T_abc = D_c H_frozen + D_c(cphf.hessian_response)`:
///   * `D_c H_frozen = L_abc + L_abx·x_c` — the analytic geometric 3rd derivative ([`third_derivative_frozen_complete`]
///     + the scalar-overlap block [`crate::hessian::fixed_density_scalar_overlap_third_derivative`]) plus the
///     strict-analytic density-path [`frozen_hessian_density_path`] (Pulay V-channel fed the TOTAL `dV/dR_c`);
///   * `D_c(hessian_response) = D_c R_static + D_c R_orbital` — the analytic Z-vector assembly
///     [`closed_form_response_hessian_derivative`]. `R_static`/`R_orbital` derivatives share the bundle-gradient
///     derivative `(D_c G_a)[bundle] + G_a[D_c bundle]`; the orbital-amplitude derivative is closed by the 2n+1
///     interchange via the self-adjoint Z-vector `y_a = A⁻¹ L_a` (no second-order CPHF solve).
/// Final tensor averaged over the 6 index permutations.
///
/// **State (v0.5.0, 2026-08-10): production-accurate.** The analytic path matches the seminumerical
/// reference to the FD noise floor (~1e-7 abs at FD step 2e-4) on equilibrium AND non-equilibrium
/// geometries, including charged species, halogen-bonded systems, and symmetric molecules with
/// degenerate orbitals. Three historical bugs, all fixed:
///   1. **[FIXED v0.4.x] dispersion gating** — [`third_derivative_frozen_complete`] added the D3 3rd
///      derivative unconditionally; now gated on `include_dispersion && enable_dispersion`.
///   2. **[FIXED v0.5.0] missing ∂K/∂q kernel chain** — the response kernel `K = γ + 2Γ_A q_A` is
///      charge-dependent, but the derivative of a kernel action on a response-charge vector `u`
///      only carried `(∂γ/∂R_c)·u + K·(D_c u)`; the onsite anharmonicity piece
///      `2Γ_A q_A^{(c)} (Σ_{t∈A} u_t)` was missing at four sites (`bundle_grad`'s `dcf`,
///      `d_sp_o`, `d_sp_s`, and the `(D_c A)x_b` operator derivative). Errors up to ~1e-5 rel,
///      scaling with the charge response (worst for H-bonded/near-degenerate systems).
///   3. **[FIXED v0.5.0] degenerate-orbital handling** — `mo_coefficient_derivatives` left
///      degenerate same-block rotations at zero (violating first-order orthonormality
///      `U_pq + U_qp = −S̃_pq`), and per-orbital `ε^{(c)}_p` (gauge-dependent in a degenerate
///      block) entered four contractions. Fixed by the symmetric gauge `U_pq = −½S̃_pq` and the
///      gauge-invariant in-block matrix `Λ^c_pq = F̃^c_pq − ε S̃^c_pq` (see `block_members`).
///      Errors were ~2e-2 rel for symmetric molecules (NH₃, CH₄); now ~1e-8.
/// The earlier claim that the 2n+1 interchange omits a second-order `∂_s R_rc` term remains WRONG —
/// the 2n+1 route needs only first-order responses. Every component is independently FD-gated
/// (`f_bc_full_matches_fd`, `d_c_fock_mo_derivative_matches_fd`, `d_c_rhs_{nonmetric,metric}_matches_fd`,
/// `d_c_operator_action_matches_fd`, `d_c_orbital_bundle_derivative_matches_fd`, the Group-A
/// `d_c_groupa_*`, `d_c_orbital_adjoint_total_matches_fd`, `d_c_static_sector_matches_fd`,
/// `closed_form_response_matches_hessian_response_fd`). The seminumerical path
/// [`third_derivative_seminumerical_dense`]/`_block`/`_vector` (two analytic-Hessian evals per DOF)
/// remains available as an independent cross-check. Returned dense slabs `slab[c][(a,b)] = T_abc`.
/// For best accuracy use tight SCF in `electronic_options` (`energy_tolerance 1e-11`,
/// `charge_tolerance 1e-9`).
///
/// Strict-analytic density-path of the fixed-density Hessian, `L_abx·x_c` for one nuclear DOF `c`:
/// the directional derivative of `frozen_hess_at` along the first-order response `(P^(c),W^(c),q^(c),V^(c))`
/// at FIXED geometry. NO finite differences. Each frozen block is multilinear in the density fields, so
/// its density-path is the block evaluated with the response densities, with the product rule applied to
/// the bilinear terms:
///   * `cn_h0`, `cross` — linear in `P` → block(density=`P^(c)`);
///   * `s2` — quadratic in `q` → `fixed_shell_charge_scc_hessian_charge_path(q, q^(c))`;
///   * `pulay` — linear in `P`,`W`, bilinear `p·V` → `pulay(P^(c),W^(c),V)` + `[pulay(V+V^(c))−pulay(V)]`;
///   * `scalar_overlap` — bilinear `(P, q)` → `scalar_overlap(P^(c),q)` + `scalar_overlap(P,q^(c))`;
///   * `repulsion`/`halogen`/`dispersion` — no density dependence → 0.
/// `v_c` must be the TOTAL `dV/dR_c = ∂V/∂R_c|_q + E_qq·q^(c)` (geometric + density) — the Pulay block
/// reads `V` and `L_abc` holds it fixed, so the full `dV` flows through this density-path channel (the
/// geometric `∂V/∂R_c|_q` is the `ab·c`-pattern coupling that neither `L_abc` nor a density-only `V_c` carry).
///
/// Visibility: `pub(crate)` so the quartic assembly
/// ([`crate::fourth_derivative::directional::directional_fourth_hessian_path_stage`]) can FD-gate
/// its own `λ`-derivative against this exact object.
#[allow(clippy::too_many_arguments)]
pub(crate) fn frozen_hessian_density_path(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    p_c: &Matrix,
    w_c: &Matrix,
    q_c: &[f64],
    v_c: &[f64],
) -> Result<Matrix> {
    let ndof = 3 * system.atoms.len();
    let nshell = electronic.shell_charges.len();
    let mut out = Matrix::zeros(ndof, ndof);
    let add = |out: &mut Matrix, h: &Matrix| {
        for r in 0..ndof {
            for c in 0..ndof {
                out[(r, c)] += h[(r, c)];
            }
        }
    };
    // cn_h0 + cross: linear in P → evaluate at density = P^(c).
    let elec_pc = {
        let mut e = electronic.clone();
        e.density = p_c.clone();
        e
    };
    add(
        &mut out,
        &crate::hessian::fixed_density_cn_h0_hessian(
            system,
            params,
            &elec_pc,
            coordination_cutoff,
        )?
        .hessian,
    );
    add(
        &mut out,
        &crate::hessian::fixed_density_cn_h0_pulay_cross_hessian(
            system,
            params,
            &elec_pc,
            coordination_cutoff,
        )?,
    );
    // s2: quadratic in q → bilinear charge-path.
    add(
        &mut out,
        &crate::hessian::fixed_shell_charge_scc_hessian_charge_path(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            q_c,
            params,
        )?,
    );
    // pulay: linear(P,W) + bilinear(P,V). path = pulay(P^(c),W^(c),V) + [pulay(V+V^(c)) − pulay(V)].
    {
        let mut e1 = electronic.clone();
        e1.density = p_c.clone();
        e1.energy_weighted_density = w_c.clone();
        let h1 = crate::hessian::fixed_density_pulay_hessian(system, params, &e1)?.hessian;
        let mut e2 = electronic.clone();
        for s in 0..nshell {
            e2.shell_scc_potential[s] += v_c[s];
        }
        let h2 = crate::hessian::fixed_density_pulay_hessian(system, params, &e2)?.hessian;
        let h0 = crate::hessian::fixed_density_pulay_hessian(system, params, electronic)?.hessian;
        for r in 0..ndof {
            for c in 0..ndof {
                out[(r, c)] += h1[(r, c)] + h2[(r, c)] - h0[(r, c)];
            }
        }
    }
    // scalar_overlap: bilinear (P, q). path = scalar_overlap(P^(c), q) + scalar_overlap(P, q^(c)).
    add(
        &mut out,
        &crate::hessian::fixed_density_scalar_overlap_hessian(system, params, &elec_pc)?,
    );
    let elec_qc = {
        let mut e = electronic.clone();
        e.shell_charges = q_c.to_vec();
        e
    };
    add(
        &mut out,
        &crate::hessian::fixed_density_scalar_overlap_hessian(system, params, &elec_qc)?,
    );
    Ok(out)
}

/// **Closed-form** nuclear derivative of the CPHF response Hessian, `∂_c cphf.hessian_response_ab =
/// D_c R_static_ab + D_c R_orbital_ab`, with NO finite differences and NO second-order CPHF solve. This is
/// the strict-analytic replacement for the FD scaffolds (`cphf_metric_residual_hessian`, the displaced
/// `CpxtbSetup` rhs/matvec) in the 2n+1 third-derivative bridge.
///
/// **Z-vector route.** `hessian_response = R_static + R_orbital`, `R_static_ab = G_a[static_b]` (x-independent
/// metric bundle), `R_orbital_ab = G_a[B x_b] = L_a·x_b`. Their nuclear derivatives:
///   `D_c R_static_ab = (D_c G_a)[static_b] + G_a[D_c static_b]`,
///   `D_c R_orbital_ab = (D_c L_a)·x_b + y_a·[D_c rhs_b − (D_c A)x_b]`,  `y_a = A⁻¹ L_a` (self-adjoint solve),
/// where `(D_c L_a)·x_b = (D_c G_a)[B x_b] + G_a[D_c(B x_b)]` (the 2n+1 interchange removes `x_bc`).
/// Both R_static and R_orbital use the SAME bundle-gradient derivative `(D_c G_a)[bundle] + G_a[D_c bundle]`
/// (the `bundle_grad` closure), differing only in the bundle. `(D_c G_a)[bundle]` (Group A) reuses the
/// validated F_bc blocks (`band+poly+cn = Tr[ΔP·(h0_bare_second+cn_block)]`, `scc_kernel` via the shell
/// scalar-potential 2nd derivatives) plus the pulay/scc-overlap overlap-second-derivative loop; `G_a[D_c
/// bundle]` (Group B) is `response_electronic_gradient` on the bundle derivative. Each piece is FD-gated
/// (tests `d_c_*_matches_fd`, `d_c_orbital_adjoint_total_matches_fd`, `d_c_static_sector_matches_fd`).
///
/// Visibility: `pub(crate)` so the directional quartic stage
/// ([`crate::fourth_derivative::response_stage::directional_response_third`]) can gate its
/// `vvv`-contracted specialization against this exact object.
#[allow(clippy::too_many_arguments)]
pub(crate) fn closed_form_response_hessian_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    cphf: &crate::cphf::GammaCartesianCpxtbResult,
    ao_opts: crate::cphf::AoDerivativeOptions,
    cutoff: f64,
) -> Result<Vec<Matrix>> {
    use crate::linalg::Matrix as M;
    use rayon::prelude::*;
    let basis = &electronic.basis;
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let n = basis.len();
    let nshell = basis.shells.len();
    let mos = cphf.mos.clone();
    let occ = &electronic.occupations;
    let eps = &cphf.orbital_energies;
    let s_mat = &electronic.integrals.overlap;
    let p_mat = &electronic.density;
    let v_pot = &electronic.shell_scc_potential;
    let space = crate::cphf::CpxtbSpace::from_occupations(occ)?;
    let npair = space.pairs.len();
    let c_analytic = crate::cphf::mo_coefficient_derivatives(system, params, electronic, cphf)?;
    let cand = crate::cphf::relaxed_fock_derivative_candidates(system, params, electronic, cphf)?;
    // Degenerate ε-block structure. For orbitals inside a degenerate block the
    // per-orbital ε^{(c)}_p is gauge-dependent; the gauge-INVARIANT object is the
    // in-block matrix Λ^c_pq = F̃^c_pq − ε S̃^c_pq (the antisymmetric in-block
    // rotation cancels through first-order orthonormality). Every formula below
    // that used `eps_c[p] · X_p` is generalized to `Σ_{p'∈block(p)} Λ^c_{pp'} X_{p'}`,
    // which reduces exactly to the old expression for singleton (non-degenerate)
    // blocks. Without this, symmetric molecules (NH₃, CH₄, ...) picked up O(1%)
    // errors in the analytic third derivative.
    let occ_flag: Vec<bool> = occ.iter().map(|&o| o > 1.0e-8).collect();
    let block_members: Vec<Vec<usize>> = {
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for p in 0..n {
            let start_new = match blocks.last() {
                Some(block) => {
                    let q = *block.last().unwrap();
                    (eps[p] - eps[q]).abs() >= 1.0e-6 || occ_flag[p] != occ_flag[q]
                }
                None => true,
            };
            if start_new {
                blocks.push(vec![p]);
            } else {
                blocks.last_mut().unwrap().push(p);
            }
        }
        let mut per_orbital = vec![Vec::new(); n];
        for block in &blocks {
            for &p in block {
                per_orbital[p] = block.clone();
            }
        }
        per_orbital
    };
    let pair_of: Vec<usize> = {
        let mut map = vec![usize::MAX; n * n];
        for (idx, &(i, a)) in space.pairs.iter().enumerate() {
            map[i * n + a] = idx;
        }
        map
    };
    let shell_kernel = crate::cphf::response_shell_scc_kernel(system, params, electronic)?;
    // ∂K/∂q chain data. The response kernel `K = γ + 2Γ_A q_A` (same-atom shell blocks)
    // is charge-dependent, so the nuclear derivative of a kernel action on a *fixed*
    // response-charge vector `u` has three pieces:
    //   D_c(K·u)|_u = (∂γ/∂R_c)·u  +  K·(D_c u)  +  2Γ_A q_A^{(c)} (Σ_{t∈A} u_t)|shells-of-A .
    // The third (onsite anharmonicity) piece was historically missing — it is the exact
    // response-potential analogue of the `2Γ q q^{(c)}` term that `kvec(q_c)` already
    // carries for the reference potential derivative `v_c`.
    let shell_model = crate::coulomb::ShellChargeModel::build(system, basis, params)?;
    let charge_order = electronic.charge_order.max(3);
    let shell_atom: Vec<usize> = {
        let mut map = vec![0usize; nshell];
        for atom in 0..nat {
            let offset = shell_model.atom_offsets[atom];
            for local in 0..shell_model.atom_shell_counts[atom] {
                map[offset + local] = atom;
            }
        }
        map
    };
    // Per-atom ∂K_onsite/∂q = ∂³E_onsite/∂q³ at the reference charges: 2Γ for stock
    // GFN1 (charge_order 3), plus the Σ(n−1)(n−2)X_n q^{n−3} Breathing-Radius orders
    // when charge_order > 3 — consistent with the response kernel's ∂²E/∂q².
    let kernel_q_atom: Vec<f64> = (0..nat)
        .map(|atom| {
            if shell_model.atom_shell_counts[atom] == 0 {
                return 0.0;
            }
            let offset = shell_model.atom_offsets[atom];
            let (_, _, third, _) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                shell_model.hardness[offset],
                shell_model.hubbard_derivs[offset],
                charge_order,
                electronic.atomic_charges[atom],
            );
            third
        })
        .collect();
    let ref_ctx = crate::cphf::ResponseGradientContext::new(
        system,
        basis,
        params,
        electronic,
        cutoff,
        ao_opts.include_cn_h0,
    )?;
    let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
        system,
        basis,
        &electronic.shell_charges,
        params,
    )?;
    let f_b_ref: Vec<M> = (0..ndof)
        .map(|b| cphf.derivative_matrices[b].h0_deriv.clone())
        .collect();
    let s_b_ref: Vec<M> = (0..ndof)
        .map(|b| cphf.derivative_matrices[b].overlap_deriv.clone())
        .collect();
    let scale_v: Vec<f64> = space
        .pairs
        .iter()
        .map(|&(i, a)| 0.5 * (occ[i] - occ[a]))
        .collect();
    let q_trans = crate::cphf::transition_shell_charges(basis, &mos, occ, s_mat)?;
    let sc = s_mat.matmul(&mos)?; // S·C (reference)
                                  // y_a = A⁻¹ L_a (self-adjoint Z-vector solve)
    let l_vectors = crate::cphf::density_gradient_adjoint_vectors(
        system, params, electronic, ao_opts, &mos, eps,
    )?;
    let setup = crate::cphf::build_cpxtb_setup(system, params, electronic, ao_opts, Some(&mos))?;
    let mut y_vectors = Vec::with_capacity(ndof);
    for l in &l_vectors {
        y_vectors.push(setup.solve_adjoint(l, 1.0e-11, 4000)?.amplitudes);
    }
    // shared closures
    let motrans = |m: &M, u: &M| -> M { u.transpose().matmul(&m.matmul(u).unwrap()).unwrap() };
    let d_motrans = |cc: &M, m: &M, m_c: &M| -> M {
        let t1 = cc.transpose().matmul(&m.matmul(&mos).unwrap()).unwrap();
        let t2 = motrans(m_c, &mos);
        let t3 = mos.transpose().matmul(&m.matmul(cc).unwrap()).unwrap();
        let mut r = t1;
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] += t2[(i, j)] + t3[(i, j)];
            }
        }
        r
    };
    let population = |dens: &M, ov: &M| -> Vec<f64> {
        let mut out = vec![0.0_f64; nshell];
        for nu in 0..n {
            let mut a = 0.0;
            for k in 0..n {
                a += dens[(nu, k)] * ov[(k, nu)];
            }
            out[basis.aos[nu].shell_index] -= a;
        }
        out
    };
    let kvec = |v: &[f64]| -> Vec<f64> {
        (0..nshell)
            .map(|s| {
                (0..nshell)
                    .map(|t| shell_kernel[(s, t)] * v[t])
                    .sum::<f64>()
            })
            .collect()
    };
    let triple = |cc: &M, coeff: &M, dcoeff: &M| -> M {
        let a1 = cc.matmul(&coeff.matmul(&mos.transpose()).unwrap()).unwrap();
        let a2 = mos
            .matmul(&dcoeff.matmul(&mos.transpose()).unwrap())
            .unwrap();
        let a3 = mos.matmul(&coeff.matmul(&cc.transpose()).unwrap()).unwrap();
        let mut m = a1;
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] += a2[(i, j)] + a3[(i, j)];
            }
        }
        m
    };

    // The `c`-slabs are independent, so compute them in parallel over the shared rayon pool. This is
    // memory-neutral (each task builds only its own ndof×ndof slab) and the dominant wall-clock speedup
    // for the closed-form third derivative on multicore machines.
    let resp: Vec<M> = (0..ndof)
        .into_par_iter()
        .map(|c| -> Result<M> {
            let mut resp_c = M::zeros(ndof, ndof);
            let (atom_c, axis_c) = (c / 3, c % 3);
            let cc = &c_analytic[c];
            // Gauge-invariant in-block orbital-energy derivative matrix (see the
            // block_members comment above); the diagonal lam(p,p) equals the
            // historical per-orbital ε^{(c)}_p.
            let lam = {
                let (h0_mo_c, resp_mo_c, s_tilde_c) = &cand[c];
                move |p: usize, q: usize| -> f64 {
                    h0_mo_c[(p, q)] + resp_mo_c[(p, q)]
                        - 0.5 * (eps[p] + eps[q]) * s_tilde_c[(p, q)]
                }
            };
            let s_c = &s_b_ref[c];
            let q_c = &cphf.shell_charge_responses[c];
            let p_c = &cphf.density_responses[c];
            let v_c: Vec<f64> = {
                let kq = kvec(q_c);
                (0..nshell).map(|s| dvdr_q[(s, c)] + kq[s]).collect()
            };
            // Atomic charge response along `c` and the onsite ∂K/∂q chain action
            // `dk_chain(u)[s] = 2 Γ_{A(s)} q_{A(s)}^{(c)} (Σ_{t∈A(s)} u_t)` — the third
            // piece of D_c(K·u)|_u (see the shell_model comment above).
            let qat_c: Vec<f64> = {
                let mut out = vec![0.0_f64; nat];
                for s in 0..nshell {
                    out[shell_atom[s]] += q_c[s];
                }
                out
            };
            let dk_chain = |u: &[f64]| -> Vec<f64> {
                let mut atom_sum = vec![0.0_f64; nat];
                for s in 0..nshell {
                    atom_sum[shell_atom[s]] += u[s];
                }
                (0..nshell)
                    .map(|s| {
                        let atom = shell_atom[s];
                        kernel_q_atom[atom] * qat_c[atom] * atom_sum[atom]
                    })
                    .collect()
            };
            let dvdr_qc = crate::hessian::shell_scalar_potential_first_derivatives(
                system, basis, q_c, params,
            )?;
            // F_bc^{H0+CN}[a] = h0_bare_second(a,c)+cn_block(a,c), bundle-independent
            let mut fbc_h0cn: Vec<M> = Vec::with_capacity(ndof);
            for a in 0..ndof {
                let h0a = crate::hessian::h0_bare_second_derivative_matrix(
                    system, params, electronic, a, c,
                )?;
                let cna = crate::hessian::h0_cn_block_second_derivative_matrix(
                    system, params, electronic, cutoff, a, c,
                )?;
                let mut m = h0a;
                for i in 0..n {
                    for j in 0..n {
                        m[(i, j)] += cna[(i, j)];
                    }
                }
                fbc_h0cn.push(m);
            }
            // dSC = S_c·C + S·C^(c)  (for transition-charge derivative)
            let dsc = {
                let a = s_c.matmul(&mos)?;
                let b = s_mat.matmul(cc)?;
                let mut m = a;
                for i in 0..n {
                    for j in 0..n {
                        m[(i, j)] += b[(i, j)];
                    }
                }
                m
            };
            // bundle-gradient closure: (D_c G_a)[bundle] + G_a[D_c bundle]
            let bundle_grad = |dp: &M,
                               dw: &M,
                               dq: &[f64],
                               d_dp: &M,
                               d_dw: &M,
                               d_dq: &[f64]|
             -> Result<Vec<f64>> {
                let mut out = vec![0.0_f64; ndof];
                let sp_resp = kvec(dq);
                let dk_dq = crate::hessian::shell_scalar_potential_first_derivatives(
                    system, basis, dq, params,
                )?;
                let chain_dq = dk_chain(dq);
                // Group A: H0+CN reuse + scc_kernel reuse
                for a in 0..ndof {
                    let mut acc = 0.0;
                    for mu in 0..n {
                        for nu in 0..n {
                            acc += dp[(mu, nu)] * fbc_h0cn[a][(mu, nu)];
                        }
                    }
                    let mut kern = 0.0;
                    for s in 0..nshell {
                        kern += dq[s] * (d2vdr_q[s][(a, c)] + dvdr_qc[(s, a)]);
                    }
                    out[a] += acc + kern;
                }
                // Group A: pulay + scc_overlap (ao-pair loop)
                for mu in 0..n {
                    let atom_mu = basis.aos[mu].atom_index;
                    let shell_mu = basis.aos[mu].shell_index;
                    let rmu = system.atoms[atom_mu].position;
                    for nu in 0..mu {
                        let atom_nu = basis.aos[nu].atom_index;
                        if atom_mu == atom_nu {
                            continue;
                        }
                        let shell_nu = basis.aos[nu].shell_index;
                        let rnu = system.atoms[atom_nu].position;
                        if (rmu - rnu).norm2() <= 1.0e-18 {
                            continue;
                        }
                        let pair = crate::integrals::contracted_pair_with_second_derivatives(
                            &basis.aos[mu],
                            &basis.aos[nu],
                            rmu,
                            rnu,
                        );
                        let dw_v = dw[(mu, nu)];
                        let scalar_shift = v_pot[shell_mu] + v_pot[shell_nu];
                        let scalar_response = sp_resp[shell_mu] + sp_resp[shell_nu];
                        let dp_v = dp[(mu, nu)];
                        let p0_v = p_mat[(mu, nu)];
                        let f_scc = -(dp_v * scalar_shift + p0_v * scalar_response);
                        let dcf = -(dp_v * (v_c[shell_mu] + v_c[shell_nu])
                            + p_c[(mu, nu)] * scalar_response
                            + p0_v
                                * (dk_dq[(shell_mu, c)]
                                    + dk_dq[(shell_nu, c)]
                                    + chain_dq[shell_mu]
                                    + chain_dq[shell_nu]));
                        let dbra0 = pair.d_bra[0].to_array();
                        let dket0 = pair.d_ket[0].to_array();
                        for alpha in 0..3 {
                            let mut db = 0.0;
                            let mut dk = 0.0;
                            if atom_c == atom_mu {
                                db += pair.h_bra_bra[0][alpha][axis_c];
                                dk += pair.h_bra_ket[0][axis_c][alpha];
                            }
                            if atom_c == atom_nu {
                                db += pair.h_bra_ket[0][alpha][axis_c];
                                dk += pair.h_ket_ket[0][alpha][axis_c];
                            }
                            out[3 * atom_mu + alpha] +=
                                db * (-2.0 * dw_v) + db * f_scc + dbra0[alpha] * dcf;
                            out[3 * atom_nu + alpha] +=
                                dk * (-2.0 * dw_v) + dk * f_scc + dket0[alpha] * dcf;
                        }
                    }
                }
                // Group B
                let gb = crate::cphf::response_electronic_gradient(
                    system,
                    electronic,
                    &shell_kernel,
                    &ref_ctx,
                    d_dp,
                    d_dp,
                    d_dw,
                    d_dq,
                )?;
                for at in 0..nat {
                    out[3 * at] += gb[at].x;
                    out[3 * at + 1] += gb[at].y;
                    out[3 * at + 2] += gb[at].z;
                }
                Ok(out)
            };
            for b in 0..ndof {
                let s_b = &s_b_ref[b];
                let f_b = &f_b_ref[b];
                let x_b = &cphf.solutions[b].amplitudes;
                let s_tilde_b = motrans(s_b, &mos);
                let f_tilde_b = motrans(f_b, &mos);
                let s_bc = crate::cphf::overlap_second_derivative_matrix(system, basis, b, c)?;
                // F_bc = h0_bare_second + cn_block + scc; the H0+CN part is exactly fbc_h0cn[b] (already built
                // for this slab `c`), so reuse it and only add the SCC-scalar block.
                let f_bc = {
                    let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                        system, params, electronic, &v_c, q_c, b, c,
                    )?;
                    let mut m = fbc_h0cn[b].clone();
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += scc[(i, j)];
                        }
                    }
                    m
                };
                let d_s_tilde = d_motrans(cc, s_b, &s_bc);
                let d_f_tilde = d_motrans(cc, f_b, &f_bc);

                // ===== ORBITAL bundle B x_b and its derivative =====
                let ob = crate::cphf::orbital_response_bundle_from_amplitudes(
                    basis,
                    s_mat,
                    p_mat,
                    &mos,
                    occ,
                    eps,
                    &space,
                    &shell_kernel,
                    x_b,
                )?;
                let (dp_o, dw_o, dq_o) = (
                    ob.density.clone(),
                    ob.weighted.clone(),
                    ob.shell_charges.clone(),
                );
                let mut coeff_po = M::zeros(n, n);
                let mut coeff_w1o = M::zeros(n, n);
                for (pi, &(i, a)) in space.pairs.iter().enumerate() {
                    let w = (occ[i] - occ[a]) * x_b[pi];
                    coeff_po[(a, i)] += w;
                    coeff_po[(i, a)] += w;
                    let w1 = w * eps[i];
                    coeff_w1o[(a, i)] += w1;
                    coeff_w1o[(i, a)] += w1;
                }
                let sp_o = kvec(&dq_o);
                let rf_o = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp_o)?;
                let rf_mo_o = motrans(&rf_o, &mos);
                let mut coeff_w2o = M::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        coeff_w2o[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo_o[(i, j)];
                    }
                }
                let zero = M::zeros(n, n);
                let d_dp_o = triple(cc, &coeff_po, &zero);
                let d_dq_o = {
                    let a = population(&d_dp_o, s_mat);
                    let b2 = population(&dp_o, s_c);
                    (0..nshell).map(|s| a[s] + b2[s]).collect::<Vec<f64>>()
                };
                let dk_dqo = crate::hessian::shell_scalar_potential_first_derivatives(
                    system, basis, &dq_o, params,
                )?;
                let d_sp_o: Vec<f64> = {
                    let kdq = kvec(&d_dq_o);
                    let chain = dk_chain(&dq_o);
                    (0..nshell)
                        .map(|s| dk_dqo[(s, c)] + kdq[s] + chain[s])
                        .collect()
                };
                let d_rf_o = {
                    let t1 = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d_sp_o)?;
                    let mut m = t1;
                    for mu in 0..n {
                        let smu = sp_o[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let snu = sp_o[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (smu + snu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                let d_rf_mo_o = d_motrans(cc, &rf_o, &d_rf_o);
                let mut dcoeff_w1o = M::zeros(n, n);
                let mut dcoeff_w2o = M::zeros(n, n);
                for &(i, a) in space.pairs.iter() {
                    // Λ-covariant ε^{(c)}·x contraction (degenerate-block safe).
                    let mut e_x = 0.0;
                    for &i2 in &block_members[i] {
                        let p2 = pair_of[i2 * n + a];
                        if p2 != usize::MAX {
                            e_x += lam(i, i2) * x_b[p2];
                        }
                    }
                    let dw1 = (occ[i] - occ[a]) * e_x;
                    dcoeff_w1o[(a, i)] += dw1;
                    dcoeff_w1o[(i, a)] += dw1;
                }
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        dcoeff_w2o[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo_o[(i, j)];
                    }
                }
                let d_dw_o = {
                    let a = triple(cc, &coeff_w1o, &dcoeff_w1o);
                    let b2 = triple(cc, &coeff_w2o, &dcoeff_w2o);
                    let mut m = a;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += b2[(i, j)];
                        }
                    }
                    m
                };
                let orb = bundle_grad(&dp_o, &dw_o, &dq_o, &d_dp_o, &d_dw_o, &d_dq_o)?;

                // ===== STATIC bundle and its derivative =====
                let mut bmat = M::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        bmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * s_tilde_b[(i, j)];
                    }
                }
                let dp_s = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &bmat)?;
                let dq_s = crate::cphf::response_shell_charges_from_density(
                    basis, s_mat, p_mat, &dp_s, s_b,
                )?;
                let sp_s = kvec(&dq_s);
                let rf_s = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &sp_s)?;
                let rf_mo_s = motrans(&rf_s, &mos);
                let mut cwa = M::zeros(n, n);
                let mut cwb = M::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        cwa[(i, j)] = 0.5
                            * (occ[i] + occ[j])
                            * (f_tilde_b[(i, j)] - (eps[i] + eps[j]) * s_tilde_b[(i, j)]);
                        cwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * rf_mo_s[(i, j)];
                    }
                }
                let dw_s = {
                    let a = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwa)?;
                    let b2 = crate::cphf::mo_coefficient_matrix_to_ao(&mos, &cwb)?;
                    let mut m = a;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += b2[(i, j)];
                        }
                    }
                    m
                };
                let mut dbmat = M::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        dbmat[(i, j)] = -0.5 * (occ[i] + occ[j]) * d_s_tilde[(i, j)];
                    }
                }
                let d_dp_s = triple(cc, &bmat, &dbmat);
                let d_dq_s = {
                    let a = population(&d_dp_s, s_mat);
                    let b2 = population(&dp_s, s_c);
                    let d = population(p_c, s_b);
                    let e = population(p_mat, &s_bc);
                    (0..nshell)
                        .map(|s| a[s] + b2[s] + d[s] + e[s])
                        .collect::<Vec<f64>>()
                };
                let dk_dqs = crate::hessian::shell_scalar_potential_first_derivatives(
                    system, basis, &dq_s, params,
                )?;
                let d_sp_s: Vec<f64> = {
                    let kdq = kvec(&d_dq_s);
                    let chain = dk_chain(&dq_s);
                    (0..nshell)
                        .map(|s| dk_dqs[(s, c)] + kdq[s] + chain[s])
                        .collect()
                };
                let d_rf_s = {
                    let t1 = crate::cphf::scalar_response_fock_matrix(basis, s_mat, &d_sp_s)?;
                    let mut m = t1;
                    for mu in 0..n {
                        let smu = sp_s[basis.aos[mu].shell_index];
                        for nu in 0..n {
                            let snu = sp_s[basis.aos[nu].shell_index];
                            m[(mu, nu)] += -0.5 * (smu + snu) * s_c[(mu, nu)];
                        }
                    }
                    m
                };
                let d_rf_mo_s = d_motrans(cc, &rf_s, &d_rf_s);
                let mut dcwa = M::zeros(n, n);
                let mut dcwb = M::zeros(n, n);
                for i in 0..n {
                    if occ[i] <= 1e-8 {
                        continue;
                    }
                    for j in 0..n {
                        if occ[j] <= 1e-8 {
                            continue;
                        }
                        // Λ-covariant (ε^{(c)}_i + ε^{(c)}_j)·S̃_b contraction
                        // (degenerate-block safe; singleton blocks reduce to the
                        // historical diagonal form).
                        let mut e_s = 0.0;
                        for &k in &block_members[i] {
                            e_s += lam(i, k) * s_tilde_b[(k, j)];
                        }
                        for &k in &block_members[j] {
                            e_s += s_tilde_b[(i, k)] * lam(k, j);
                        }
                        dcwa[(i, j)] = 0.5
                            * (occ[i] + occ[j])
                            * (d_f_tilde[(i, j)]
                                - e_s
                                - (eps[i] + eps[j]) * d_s_tilde[(i, j)]);
                        dcwb[(i, j)] = 0.5 * (occ[i] + occ[j]) * d_rf_mo_s[(i, j)];
                    }
                }
                let d_dw_s = {
                    let a = triple(cc, &cwa, &dcwa);
                    let b2 = triple(cc, &cwb, &dcwb);
                    let mut m = a;
                    for i in 0..n {
                        for j in 0..n {
                            m[(i, j)] += b2[(i, j)];
                        }
                    }
                    m
                };
                let stat = bundle_grad(&dp_s, &dw_s, &dq_s, &d_dp_s, &d_dw_s, &d_dq_s)?;

                // ===== D_c rhs_b (non-metric + metric-SCC) =====
                //   rhs0[ia] = −(CᵀF_bC)_ia + ε_i(CᵀS_bC)_ia ;  metric m[ia] = −(Cᵀ RF_b^static C)_ia.
                let d_rf_mo_b_oo = &d_rf_mo_s; // metric RF_b derivative = static RF_b derivative
                let mut d_rhs = vec![0.0_f64; npair];
                for (idx, &(i, a)) in space.pairs.iter().enumerate() {
                    // Λ-covariant ε^{(c)}_i·S̃_b contraction (degenerate-block safe).
                    let mut e_s = 0.0;
                    for &i2 in &block_members[i] {
                        e_s += lam(i, i2) * s_tilde_b[(i2, a)];
                    }
                    let drhs0 = -d_f_tilde[(i, a)] + e_s + eps[i] * d_s_tilde[(i, a)];
                    let dmetric = -d_rf_mo_b_oo[(i, a)];
                    d_rhs[idx] = drhs0 + dmetric;
                }
                // ===== (D_c A) x_b  (operator derivative-action) =====
                let mut d_axb = vec![0.0_f64; npair];
                {
                    // D_c transition charges
                    let dqt: Vec<Vec<f64>> = space
                        .pairs
                        .iter()
                        .map(|&(i, a)| {
                            let mut q = vec![0.0_f64; nshell];
                            for (sh, shell) in basis.shells.iter().enumerate() {
                                let end = shell.first_ao + shell.nao;
                                for mu in shell.first_ao..end {
                                    q[sh] -= cc[(mu, a)] * sc[(mu, i)]
                                        + mos[(mu, a)] * dsc[(mu, i)]
                                        + cc[(mu, i)] * sc[(mu, a)]
                                        + mos[(mu, i)] * dsc[(mu, a)];
                                }
                            }
                            q
                        })
                        .collect();
                    let mut g = vec![0.0_f64; nshell];
                    let mut dg = vec![0.0_f64; nshell];
                    for p in 0..npair {
                        for s in 0..nshell {
                            g[s] += q_trans[p][s] * scale_v[p] * x_b[p];
                            dg[s] += dqt[p][s] * scale_v[p] * x_b[p];
                        }
                    }
                    let pot = kvec(&g);
                    let dk_g = crate::hessian::shell_scalar_potential_first_derivatives(
                        system, basis, &g, params,
                    )?;
                    let k_dg = kvec(&dg);
                    let chain_g = dk_chain(&g);
                    let dpot: Vec<f64> = (0..nshell)
                        .map(|s| dk_g[(s, c)] + k_dg[s] + chain_g[s])
                        .collect();
                    for (p, &(i, a)) in space.pairs.iter().enumerate() {
                        // Λ-covariant gap derivative: [x Λ^c_vv − Λ^c_oo x]_(i,a)
                        // (degenerate-block safe; reduces to (ε^c_a − ε^c_i)·x_p
                        // for singleton blocks).
                        let mut v = 0.0;
                        for &a2 in &block_members[a] {
                            let p2 = pair_of[i * n + a2];
                            if p2 != usize::MAX {
                                v += lam(a, a2) * x_b[p2];
                            }
                        }
                        for &i2 in &block_members[i] {
                            let p2 = pair_of[i2 * n + a];
                            if p2 != usize::MAX {
                                v -= lam(i, i2) * x_b[p2];
                            }
                        }
                        for s in 0..nshell {
                            v += dqt[p][s] * pot[s] + q_trans[p][s] * dpot[s];
                        }
                        d_axb[p] = v;
                    }
                }
                // ===== assemble D_c R_static + D_c R_orbital =====
                for a in 0..ndof {
                    let mut zterm = 0.0;
                    for p in 0..npair {
                        zterm += y_vectors[a][p] * (d_rhs[p] - d_axb[p]);
                    }
                    resp_c[(a, b)] = stat[a] + orb[a] + zterm;
                }
            }
            Ok(resp_c)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(resp)
}

/// Shared producer: the **un-symmetrized** closed-form 2n+1 slabs `total[c][(a,b)] = ∂_c H_ab`
/// (`= D_c H_frozen + D_c(cphf.hessian_response)`), BEFORE the 6-permutation average. Each slab is
/// symmetric in `(a,b)` (it differentiates the symmetric Hessian) but the slab index `c` carries the
/// ordered-bridge role, so the full tensor is symmetrized by the output drivers. Building this once and
/// letting [`third_derivative_analytic_dense`] / `_vector` / `_block` consume it avoids materializing
/// both `total` AND a separate symmetric copy.
fn third_derivative_closed_form_total(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<Vec<Matrix>> {
    crate::terms::require_order(
        &options.electronic_options,
        params,
        3,
        "the analytic third derivative",
    )?;
    let electronic =
        crate::electronic::run_electronic(system, params, options.electronic_options.clone())?;
    // TEMPORARY guard (until the finite-temperature response rework): the closed-form
    // response-derivative algebra below assumes integer (0/2) occupations. With Fermi
    // smearing it would silently return wrong numbers, so reject honestly instead.
    // The seminumerical path fully supports fractional occupations today.
    if electronic
        .occupations
        .iter()
        .any(|&f| f > 1.0e-8 && (f - 2.0).abs() > 1.0e-8)
    {
        return Err(crate::error::Gfn1Error::InvalidInput(
            "analytic third derivative with fractional (Fermi-smeared) occupations is not yet \
             supported; use third_derivative_seminumerical_* until the finite-temperature \
             analytic path lands"
                .to_string(),
        ));
    }
    let ndof = 3 * system.atoms.len();
    // CPHF first-order responses (density P^(z), energy-weighted W^(z), shell charges q^(z)).
    let cphf = crate::cphf::solve_nonpbc_cpxtb_hessian_response(
        system,
        params,
        &electronic,
        crate::cphf::AoDerivativeOptions {
            coordination_cutoff,
            include_cn_h0: options.electronic_options.hamiltonian.enable_cn_hamiltonian,
        },
        crate::cphf::CpxtbOptions::default(),
    )?;
    // ===== Strict closed-form 2n+1:  T_abc = D_c H_frozen + D_c(cphf.hessian_response) =====
    // NO finite differences anywhere. D_c H_frozen = L_abc + L_abx·x_c (analytic geometric 3rd derivative
    // + strict-analytic density-path of the fixed-density Hessian). The response part
    // D_c(hessian_response) = D_c R_static + D_c R_orbital is the strict-analytic Z-vector assembly.
    let ao_opts = crate::cphf::AoDerivativeOptions {
        coordination_cutoff,
        include_cn_h0: options.electronic_options.hamiltonian.enable_cn_hamiltonian,
    };
    let mut total = closed_form_response_hessian_derivative(
        system,
        params,
        &electronic,
        &cphf,
        ao_opts,
        coordination_cutoff,
    )?;
    // Dispersion is included only when BOTH the Hessian-level flag and the electronic-level flag are
    // on — exactly the gate `analytic_hessian_from_result` uses — so the analytic third derivative
    // matches the seminumerical FD-of-Hessian ground truth (which excludes D3 when disabled).
    let include_disp =
        options.include_dispersion && options.electronic_options.enable_dispersion;
    let dispersion_ref = if include_disp {
        options.electronic_options.d3_reference_path.as_deref()
    } else {
        None
    };
    let l_abc_geo = third_derivative_frozen_complete(
        system,
        params,
        &electronic,
        dispersion_ref,
        coordination_cutoff,
        include_disp,
    )?;
    let scalar_overlap_3rd =
        crate::hessian::fixed_density_scalar_overlap_third_derivative(system, params, &electronic)?;
    let dscalar = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        &electronic.basis,
        &electronic.shell_charges,
        params,
    )?;
    let shell_kernel = crate::cphf::response_shell_scc_kernel(system, params, &electronic)?;
    let nshell = electronic.shell_charges.len();
    // Pulay CN-response 3rd-derivative term: `h0` in the Pulay overlap-coefficient reads a CN cached in
    // `electronic`, so its geometric CN derivative (`2P·∂h0/∂CN·∂CN/∂R_c` in both pulay channels) is a
    // frozen-density term the density-path (P/W/V response only) omits. Gated on the CN-Hamiltonian flag,
    // like the CPHF's `include_cn_h0`. `cn_grad[at][c] = ∂CN_at/∂R_c` (grad-only, lean). See
    // `fixed_density_pulay_cn_h0_response`; FD-gated in `diag_nonEq_third_derivative_decompose` (STEP5/6).
    let cn_grad = if options.electronic_options.hamiltonian.enable_cn_hamiltonian {
        Some(crate::hessian::cn_gradient_matrix(system, coordination_cutoff)?)
    } else {
        None
    };
    let nat = system.atoms.len();
    // Accumulate the frozen contribution into `total` (which already holds the response slabs).
    for c in 0..ndof {
        let q_c = &cphf.shell_charge_responses[c];
        let v_c: Vec<f64> = (0..nshell)
            .map(|s| {
                dscalar[(s, c)]
                    + (0..nshell)
                        .map(|t| shell_kernel[(s, t)] * q_c[t])
                        .sum::<f64>()
            })
            .collect();
        let l_abx = frozen_hessian_density_path(
            system,
            params,
            &electronic,
            coordination_cutoff,
            &cphf.density_responses[c],
            &cphf.energy_weighted_density_responses[c],
            q_c,
            &v_c,
        )?;
        let l_cn = if let Some(ref grad) = cn_grad {
            let cn_grad_c: Vec<f64> = (0..nat).map(|at| grad[at][c]).collect();
            crate::hessian::fixed_density_pulay_cn_h0_response(
                system,
                params,
                &electronic,
                &cn_grad_c,
            )?
        } else {
            Matrix::zeros(ndof, ndof)
        };
        for a in 0..ndof {
            for b in 0..ndof {
                total[c][(a, b)] += l_abc_geo[c][(a, b)]
                    + scalar_overlap_3rd[c][(a, b)]
                    + l_abx[(a, b)]
                    + l_cn[(a, b)];
            }
        }
    }
    Ok(total)
}

/// **Dense output (backward-compatible).** The strict closed-form nuclear third derivative as `ndof`
/// fully-symmetrized dense slabs (`slab[c][(a,b)] = T_abc`). For large systems prefer the memory-lean
/// [`third_derivative_analytic_vector`] (a single `3N×3N` directional matrix) or
/// [`third_derivative_analytic_block`] (a chosen atom subset). The packed
/// [`third_derivative_analytic_dense`] keeps the same data in `~1/6` the memory.
pub fn third_derivative_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<Vec<Matrix>> {
    Ok(
        third_derivative_analytic_dense(system, params, options, coordination_cutoff)?
            .to_dense_slabs(),
    )
}

/// **Dense (symmetric-packed) output mode.** The full closed-form `T_abc` in a [`SymmetricThird`]
/// (`n(n+1)(n+2)/6` entries — ~`6×` smaller than `ndof` dense slabs). The un-symmetrized slabs are
/// summed over all `n³` index orderings into the canonical store and `1/6`-averaged, recovering the
/// fully-symmetric tensor. Read with `.get(a,b,c)` / `.to_dense_slabs()` / `.block(dofs)` /
/// `.contract_last(v)` / `.contract_vvv(v)`.
pub fn third_derivative_analytic_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
) -> Result<SymmetricThird> {
    let total = third_derivative_closed_form_total(system, params, options, coordination_cutoff)?;
    let ndof = total.len();
    let mut store = SymmetricThird::zeros(ndof);
    // Visit each canonical triple `i ≤ j ≤ k` ONCE and store its explicit 6-permutation average. (Summing
    // over all n³ orderings and dividing by 6 would be wrong for repeated indices, whose orbit is smaller
    // than 6 — the explicit 6-term mean is correct for distinct AND repeated triples alike.)
    for k in 0..ndof {
        for j in 0..=k {
            for i in 0..=j {
                let val = (total[k][(i, j)]
                    + total[j][(i, k)]
                    + total[k][(j, i)]
                    + total[i][(j, k)]
                    + total[j][(k, i)]
                    + total[i][(k, j)])
                    / 6.0;
                store.add(i, j, k, val);
            }
        }
    }
    Ok(store)
}

/// **Vector output mode (memory-lean, recommended for large systems).** The directional third
/// derivative `K[a][b] = Σ_c v_c T_abc` — a single `3N×3N` matrix (the derivative of the Hessian along
/// `v`, e.g. a normal mode), WITHOUT ever returning the full `ndof³` tensor. Built from the
/// un-symmetrized slabs as `K = (1/3)(A + B + Bᵀ)` with `A[a][b] = Σ_c v_c total[c][(a,b)]` and
/// `B[a][b] = (total[a]·v)[b]` (exact `Σ_c v_c T_abc^sym`, since each slab is `(a,b)`-symmetric).
pub fn third_derivative_analytic_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    v: &[f64],
) -> Result<Matrix> {
    let total = third_derivative_closed_form_total(system, params, options, coordination_cutoff)?;
    let ndof = total.len();
    if v.len() != ndof {
        return Err(crate::error::Gfn1Error::InvalidInput(format!(
            "third_derivative_analytic_vector: direction length {} != 3*natoms {}",
            v.len(),
            ndof
        )));
    }
    let mut a_mat = Matrix::zeros(ndof, ndof);
    let mut b_mat = Matrix::zeros(ndof, ndof);
    for s in 0..ndof {
        let vs = v[s];
        for i in 0..ndof {
            let mut row_dot = 0.0;
            for j in 0..ndof {
                let t = total[s][(i, j)];
                a_mat[(i, j)] += vs * t;
                row_dot += t * v[j];
            }
            b_mat[(s, i)] = row_dot; // B[s][i] = (total[s]·v)[i]
        }
    }
    let mut k = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            k[(i, j)] = (a_mat[(i, j)] + b_mat[(i, j)] + b_mat[(j, i)]) / 3.0;
        }
    }
    Ok(k)
}

/// **Block output mode (OOM control for large systems).** The `|dofs|³` sub-tensor restricted to the
/// Cartesian DOFs of the chosen `atoms`, returned as `(dofs, slabs)` with
/// `slabs[ci][(ai,bi)] = T[dofs[ai]][dofs[bi]][dofs[ci]]` (the fully-symmetrized closed-form tensor,
/// same layout as [`SymmetricThird::block`]). The returned tensor is `O(|block|³)` — for local
/// anharmonicity over a reactive subregion.
pub fn third_derivative_analytic_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    coordination_cutoff: f64,
    atoms: &[usize],
) -> Result<(Vec<usize>, Vec<Matrix>)> {
    let total = third_derivative_closed_form_total(system, params, options, coordination_cutoff)?;
    let mut dofs = Vec::with_capacity(3 * atoms.len());
    for &a in atoms {
        for axis in 0..3 {
            dofs.push(3 * a + axis);
        }
    }
    let m = dofs.len();
    let mut slabs = vec![Matrix::zeros(m, m); m];
    for ci in 0..m {
        for ai in 0..m {
            for bi in 0..m {
                let (a, b, c) = (dofs[ai], dofs[bi], dofs[ci]);
                // 6-permutation average of the un-symmetrized slabs.
                slabs[ci][(ai, bi)] = (total[c][(a, b)]
                    + total[b][(a, c)]
                    + total[c][(b, a)]
                    + total[a][(b, c)]
                    + total[b][(c, a)]
                    + total[a][(c, b)])
                    / 6.0;
            }
        }
    }
    Ok((dofs, slabs))
}

/// **Semi-numerical** nuclear third derivative (production path): the directional derivative
/// of the analytic Hessian along `v`, `K_ab(v) = Σ_c v_c T_abc ~= [H_ab(R+h v) − H_ab(R−h v)]/2h`.
/// This is the **Vector output mode** and the cheapest route -- just **two** analytic-Hessian
/// evaluations. Exact to FD precision and reuses the entire (FD-validated) analytic Hessian;
/// the fully-analytic 2n+1 assembly (the frozen `L_abc` blocks + the CPHF response cross-terms)
/// is the experimental alternative. `v` is a full `3N` direction (e.g. a normal mode).
pub fn third_derivative_seminumerical_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    v: &[f64],
    step: f64,
) -> Result<Matrix> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut plus = system.clone();
    let mut minus = system.clone();
    for atom in 0..nat {
        for c in 0..3 {
            let d = step * v[3 * atom + c];
            match c {
                0 => {
                    plus.atoms[atom].position.x += d;
                    minus.atoms[atom].position.x -= d;
                }
                1 => {
                    plus.atoms[atom].position.y += d;
                    minus.atoms[atom].position.y -= d;
                }
                _ => {
                    plus.atoms[atom].position.z += d;
                    minus.atoms[atom].position.z -= d;
                }
            }
        }
    }
    let hp = crate::hessian::analytic_hessian(&plus, params, options.clone())?.hessian;
    let hm = crate::hessian::analytic_hessian(&minus, params, options)?.hessian;
    let mut k = Matrix::zeros(ndof, ndof);
    for a in 0..ndof {
        for b in 0..ndof {
            k[(a, b)] = (hp[(a, b)] - hm[(a, b)]) / (2.0 * step);
        }
    }
    Ok(k)
}

/// **Semi-numerical** full third-derivative tensor (production path), packed into a
/// [`SymmetricThird`]: each slab `c` is `∂(H_ab)/∂R_c` by central FD of the analytic Hessian
/// (`2·ndof` Hessian evaluations). For large systems prefer
/// [`third_derivative_seminumerical_vector`] (2 evaluations) or a `Block` subset.
pub fn third_derivative_seminumerical_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    step: f64,
) -> Result<SymmetricThird> {
    let nat = system.atoms.len();
    let ndof = 3 * nat;
    let mut store = SymmetricThird::zeros(ndof);
    for c in 0..ndof {
        let (atom, axis) = (c / 3, c % 3);
        let mut plus = system.clone();
        let mut minus = system.clone();
        match axis {
            0 => {
                plus.atoms[atom].position.x += step;
                minus.atoms[atom].position.x -= step;
            }
            1 => {
                plus.atoms[atom].position.y += step;
                minus.atoms[atom].position.y -= step;
            }
            _ => {
                plus.atoms[atom].position.z += step;
                minus.atoms[atom].position.z -= step;
            }
        }
        let hp = crate::hessian::analytic_hessian(&plus, params, options.clone())?.hessian;
        let hm = crate::hessian::analytic_hessian(&minus, params, options.clone())?.hessian;
        // Store only the canonical entries this slab owns (i ≤ j ≤ c) -- each unordered triple
        // is written once, from its largest index.
        for a in 0..=c {
            for b in a..=c {
                store.add(a, b, c, (hp[(a, b)] - hm[(a, b)]) / (2.0 * step));
            }
        }
    }
    Ok(store)
}

/// **Semi-numerical Block mode** (production OOM control): the `|dofs|³` sub-tensor restricted to
/// the DOFs of the chosen `atoms`, returned as `(dofs, slabs)` with
/// `slabs[c][(a,b)] = T[dofs[a]][dofs[b]][dofs[c]]` -- the same layout as [`SymmetricThird::block`],
/// but computed **without** materializing the full `ndof³` tensor. Only the Hessian along the
/// in-block axes is finite-differenced (`|dofs|` central pairs vs `ndof` for Dense), and only the
/// in-block `(a,b)` entries are read -- so both memory and compute scale with the subset, not `N`.
///
/// It uses the **same canonical packing** as [`third_derivative_seminumerical_dense`] -- each
/// unordered triple is finite-differenced along its *largest* index (always in-block, since all
/// three indices are) and the rest is filled by permutation symmetry -- so the result is **bit-for-
/// bit** the Dense sub-block (not merely equal to FD-truncation order, which would differ between
/// FD axes). For local anharmonicity over a reactive subregion. `atoms` are atom indices; their 3
/// Cartesian DOFs each enter the block.
pub fn third_derivative_seminumerical_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: crate::hessian::AnalyticHessianOptions,
    atoms: &[usize],
    step: f64,
) -> Result<(Vec<usize>, Vec<Matrix>)> {
    let mut dofs = Vec::with_capacity(3 * atoms.len());
    for &a in atoms {
        for axis in 0..3 {
            dofs.push(3 * a + axis);
        }
    }
    let m = dofs.len();
    // Block-local symmetric store, indexed 0..m over the block DOFs. Canonical packing: slab `ci`
    // owns the triples whose largest block-index is `ci`, finite-differenced along global axis
    // `dofs[ci]` -- identical to the Dense path restricted to these DOFs.
    let mut store = SymmetricThird::zeros(m);
    for ci in 0..m {
        let c = dofs[ci];
        let (atom, axis) = (c / 3, c % 3);
        let mut plus = system.clone();
        let mut minus = system.clone();
        match axis {
            0 => {
                plus.atoms[atom].position.x += step;
                minus.atoms[atom].position.x -= step;
            }
            1 => {
                plus.atoms[atom].position.y += step;
                minus.atoms[atom].position.y -= step;
            }
            _ => {
                plus.atoms[atom].position.z += step;
                minus.atoms[atom].position.z -= step;
            }
        }
        let hp = crate::hessian::analytic_hessian(&plus, params, options.clone())?.hessian;
        let hm = crate::hessian::analytic_hessian(&minus, params, options.clone())?.hessian;
        for ai in 0..=ci {
            for bi in ai..=ci {
                store.add(
                    ai,
                    bi,
                    ci,
                    (hp[(dofs[ai], dofs[bi])] - hm[(dofs[ai], dofs[bi])]) / (2.0 * step),
                );
            }
        }
    }
    Ok((dofs, store.to_dense_slabs()))
}

pub mod finite_t;

#[cfg(test)]
mod tests;
