// SPDX-License-Identifier: GPL-3.0-or-later
//! Experimental **parameter-free long-range Fock exchange** for the GFN1-xTB SCC (non-periodic),
//! the Mulliken-approximated range-separated exchange of **LC-DFTB**.
//!
//! References:
//! - T. A. Niehaus, F. Della Sala, "Range separated functionals in the density functional based
//!   tight-binding method: Formalism", *Phys. Status Solidi B* **249**, 237 (2012).
//! - A. V. Lutsker, B. Aradi, T. A. Niehaus, "Implementation and benchmark of a long-range
//!   corrected functional in the density functional based tight-binding method", *J. Chem. Phys.*
//!   **143**, 184107 (2015) [arXiv:1504.00243].
//!
//! **MFX** (Mulliken Fock eXchange) is the long-range exact-exchange term written in the
//! Mulliken/monopole approximation: the four-centre exchange integral `(μσ|νλ)` is replaced by the
//! overlap-weighted product `¼ S_{μσ} S_{νλ}(γ^lr_{A_μA_ν}+γ^lr_{A_μA_λ}+γ^lr_{A_σA_ν}+γ^lr_{A_σA_λ})`,
//! with the **parameter-free** Gaussian charge-cloud long-range kernel `γ^lr` of
//! [`crate::coulomb`] (derived from the existing chemical hardness `η`; no fitted constant) and the
//! range-separation `ω` from the chosen [`crate::coulomb`] ω-scheme (HardnessPairwise here).
//!
//! The exchange acts on the **density fluctuation** `ΔP = P − P0` (neutral-atom reference), so it
//! adds no new constant or first-order term on top of GFN1's existing electrostatics:
//!   `E_x = ½ Tr[ΔP · K[ΔP]]`,  `K[ΔP]_{μν} = −⅛ Σ_{σλ} ΔP_{σλ} S_{μσ} S_{νλ}
//!         (γ^lr_{μν}+γ^lr_{μλ}+γ^lr_{σν}+γ^lr_{σλ})`.
//! The kernel operator is **self-adjoint** (symmetric under `(μν)↔(σλ)`), so the Fock contribution
//! is exactly `F = ∂E_x/∂P = K[ΔP]` (used by the SCC and FD-gated below).
//!
//! Evaluated by the **GEMM factorization** (no `O(N⁴)` four-index loop):
//!   `K = −⅛ [ Γ∘(SΔPS) + (Γ∘(SΔP))S + S(Γ∘(ΔPS)) + S(Γ∘ΔP)S ]`,
//! where `∘` is the Hadamard (elementwise) product and `Γ_{μν} = γ^lr_{A_μ A_ν}` is the AO×AO
//! long-range kernel. Off by default; non-periodic.

use crate::basis::BasisSet;
use crate::linalg::Matrix;
use crate::math::Vec3;
use std::sync::{Arc, Mutex, OnceLock};

/// Build the AO×AO long-range exchange kernel `Γ_{μν} = γ^lr_{A_μ A_ν}(R_{AB}; ω_{AB})` from the
/// per-atom positions and chemical hardnesses, using the Gaussian charge-cloud `γ^lr`
/// ([`crate::coulomb::lr_gamma_exchange`]) with the parameter-free **HardnessPairwise** ω
/// ([`crate::coulomb::omega_hardness_pairwise`], `ω_A = η_A`). Geometry-fixed (built once per
/// geometry). The kernel is the same for every AO on a given atom pair (Mulliken/monopole picture).
pub fn lr_exchange_gamma_matrix(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    atom_hardness: &[f64],
    omega_scheme: crate::coulomb::OmegaScheme,
) -> Matrix {
    use crate::coulomb::{exchange_sigma_pair, lr_gamma_exchange, omega_pair};
    // Per-atom-pair kernel value (symmetric, nat×nat), then broadcast to AOs.
    let mut gab = vec![0.0_f64; nat * nat];
    for a in 0..nat {
        for b in a..nat {
            let r = (atom_pos[a] - atom_pos[b]).norm();
            let sigma = exchange_sigma_pair(atom_hardness[a], atom_hardness[b]);
            let omega = omega_pair(omega_scheme, atom_hardness[a], atom_hardness[b]);
            let g = lr_gamma_exchange(r, sigma, omega);
            gab[a * nat + b] = g;
            gab[b * nat + a] = g;
        }
    }
    let n = basis.len();
    let atom_of: Vec<usize> = basis.aos.iter().map(|ao| ao.atom_index).collect();
    let mut gamma = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            gamma[(mu, nu)] = gab[atom_of[mu] * nat + atom_of[nu]];
        }
    }
    gamma
}

/// **LocalGeometry** counterpart of [`lr_exchange_gamma_matrix`]: the per-pair ω is the
/// size-modulated [`crate::coulomb::omega_pair_local_geometry`] built from the per-atom size factors
/// `s` (`s_A = (1+CN_A)^(−1/3)`, [`crate::coulomb::local_size_factor_from_cn`]) instead of the bare
/// hardness. `s ≡ 1` reproduces [`lr_exchange_gamma_matrix`] under `HardnessPairwise`. Geometry-fixed
/// per SCC (the `s` change with geometry, hence the analytic `∂ω/∂R` force in the gradient).
pub fn lr_exchange_gamma_matrix_local(
    basis: &BasisSet,
    nat: usize,
    atom_pos: &[Vec3],
    atom_hardness: &[f64],
    s: &[f64],
) -> Matrix {
    use crate::coulomb::{exchange_sigma_pair, lr_gamma_exchange, omega_pair_local_geometry};
    let mut gab = vec![0.0_f64; nat * nat];
    for a in 0..nat {
        for b in a..nat {
            let r = (atom_pos[a] - atom_pos[b]).norm();
            let sigma = exchange_sigma_pair(atom_hardness[a], atom_hardness[b]);
            let omega = omega_pair_local_geometry(atom_hardness[a], s[a], atom_hardness[b], s[b]);
            let g = lr_gamma_exchange(r, sigma, omega);
            gab[a * nat + b] = g;
            gab[b * nat + a] = g;
        }
    }
    let n = basis.len();
    let atom_of: Vec<usize> = basis.aos.iter().map(|ao| ao.atom_index).collect();
    let mut gamma = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            gamma[(mu, nu)] = gab[atom_of[mu] * nat + atom_of[nu]];
        }
    }
    gamma
}

/// Neutral-atom **reference density** `P0` (block-diagonal) for the exchange fluctuation `ΔP = P − P0`:
/// each shell's GFN1 reference occupation spread equally over its (normalized) AOs on the diagonal,
/// off-diagonal zero. This is the minimal-basis superposition-of-neutral-atoms density, so the
/// exchange correction `½Tr[ΔP K[ΔP]]` vanishes at the reference (no double-counting of the
/// isolated-atom exchange already implicit in GFN1's parameters). Non-periodic.
pub fn neutral_atom_reference_density(basis: &BasisSet) -> Matrix {
    let n = basis.len();
    let mut p0 = Matrix::zeros(n, n);
    for shell in &basis.shells {
        if shell.nao == 0 {
            continue;
        }
        let per_ao = shell.reference_occ / shell.nao as f64;
        for mu in shell.first_ao..shell.first_ao + shell.nao {
            p0[(mu, mu)] = per_ao;
        }
    }
    p0
}

/// Hadamard (elementwise) product of two equal-shape matrices.
fn hadamard(a: &Matrix, b: &Matrix) -> Matrix {
    let (r, c) = (a.rows(), a.cols());
    let mut out = Matrix::zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            out[(i, j)] = a[(i, j)] * b[(i, j)];
        }
    }
    out
}

/// Elementwise `a + b` (both symmetric here).
fn add4(a: &Matrix, b: &Matrix, c: &Matrix, d: &Matrix) -> Matrix {
    let (r, cc) = (a.rows(), a.cols());
    let mut out = Matrix::zeros(r, cc);
    for i in 0..r {
        for j in 0..cc {
            out[(i, j)] = a[(i, j)] + b[(i, j)] + c[(i, j)] + d[(i, j)];
        }
    }
    out
}

/// The MFX exchange kernel `K[ΔP]` (a symmetric AO×AO matrix) for the density-fluctuation `ΔP` and
/// overlap `S`, via the GEMM factorization (see module docs). `gamma` is the AO×AO long-range
/// kernel from [`lr_exchange_gamma_matrix`]. Self-adjoint, so this is also the Fock shift
/// `F = ∂E_x/∂P`.
pub fn mfx_kernel(dp: &Matrix, s: &Matrix, gamma: &Matrix) -> Matrix {
    // `S` and `ΔP` are symmetric, and `Γ` is symmetric, so two of the four terms are transposes of
    // products already formed: `ΔPS = (SΔP)ᵀ` and `K3 = S(Γ∘ΔPS) = [(Γ∘SΔP)S]ᵀ = K2ᵀ`. That removes
    // the `ΔP·S` and `K3` matrix products — 5 GEMMs instead of 7 per kernel build (the dominant
    // per-SCC-iteration cost at scale).
    let s_dp = s.matmul(dp).expect("S·ΔP conformable"); // GEMM 1
    let s_dp_s = s_dp.matmul(s).expect("S·ΔP·S conformable"); // GEMM 2
                                                              // K1 = Γ∘(SΔPS)
    let k1 = hadamard(gamma, &s_dp_s);
    // K2 = (Γ∘(SΔP))·S ; K3 = S·(Γ∘(ΔPS)) = K2ᵀ
    let k2 = hadamard(gamma, &s_dp).matmul(s).expect("k2 conformable"); // GEMM 3
    let k3 = k2.transpose();
    // K4 = S·(Γ∘ΔP)·S
    let k4 = s
        .matmul(&hadamard(gamma, dp))
        .expect("k4a conformable") // GEMM 4
        .matmul(s)
        .expect("k4 conformable"); // GEMM 5
    let mut k = add4(&k1, &k2, &k3, &k4);
    let n = k.rows();
    for i in 0..n {
        for j in 0..n {
            k[(i, j)] *= -0.125;
        }
    }
    k
}

/// MFX correction energy and Fock shift at the density `p` with neutral-atom reference `p0`:
/// `ΔP = P − P0`, `E_x = ½ Tr[ΔP · K[ΔP]]`, `F = K[ΔP]`. The kernel is self-adjoint, so `F` is the
/// exact `∂E_x/∂P` (FD-gated). `gamma` is the geometry-fixed AO×AO long-range kernel.
pub struct ExchangeEnergyFock {
    pub energy: f64,
    pub fock: Matrix,
}

/// Compute [`ExchangeEnergyFock`] from `p`, `p0`, `s`, `gamma`.
pub fn mfx_energy_fock(p: &Matrix, p0: &Matrix, s: &Matrix, gamma: &Matrix) -> ExchangeEnergyFock {
    let n = p.rows();
    let mut dp = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            dp[(i, j)] = p[(i, j)] - p0[(i, j)];
        }
    }
    let k = mfx_kernel(&dp, s, gamma);
    // E = ½ Tr[ΔP K] = ½ Σ_ij ΔP_ij K_ij (both symmetric).
    let mut energy = 0.0;
    for i in 0..n {
        for j in 0..n {
            energy += dp[(i, j)] * k[(i, j)];
        }
    }
    energy *= 0.5;
    ExchangeEnergyFock { energy, fock: k }
}

/// Overlap-Pulay weight `W^S_{ab} = ∂E_x/∂S_{ab}` (symmetric AO×AO) for the analytic gradient — the
/// explicit `S`-dependence of `E_x = ½Tr[ΔP K[ΔP]]` (the implicit density response is carried by the
/// base band-structure energy-weighted-density Pulay term, since the exchange Fock is in the
/// converged Fock). GEMM-factored:
/// `W^S = −⅛[ (ΔP∘Γ)SΔP + ((ΔP S)∘Γ)ΔP + ΔP(Γ∘(SΔP)) + (ΔP S)(ΔP∘Γ) ]`.
/// Contract with `dS/dR` to get the overlap-derivative force contribution.
pub fn mfx_overlap_weight(dp: &Matrix, s: &Matrix, gamma: &Matrix) -> Matrix {
    let dps = dp.matmul(s).expect("ΔP·S");
    let sdp = s.matmul(dp).expect("S·ΔP");
    let dpg = hadamard(dp, gamma); // ΔP∘Γ
    let t1 = dpg.matmul(s).expect("a").matmul(dp).expect("b"); // (ΔP∘Γ)SΔP
    let t2 = hadamard(&dps, gamma).matmul(dp).expect("c"); // ((ΔP S)∘Γ)ΔP
    let t3 = dp.matmul(&hadamard(gamma, &sdp)).expect("d"); // ΔP(Γ∘(SΔP))
    let t4 = dps.matmul(&dpg).expect("e"); // (ΔP S)(ΔP∘Γ)
    let mut w = add4(&t1, &t2, &t3, &t4);
    let n = w.rows();
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] *= -0.125;
        }
    }
    w
}

/// Kernel-force weight `W^Γ_{cd} = ∂E_x/∂Γ_{cd}` (symmetric AO×AO) for the analytic gradient — the
/// explicit dependence of `E_x` on the long-range kernel `Γ`. `E_x` is linear in `Γ`, so this does
/// not depend on `Γ` itself. GEMM-factored: `W^Γ = −⅛[ ΔP∘(SΔPS) + (SΔP)∘(ΔPS) ]`. Aggregated to
/// atom pairs and contracted with `dγ^lr_{AB}/dR` it gives the off-site kernel force.
pub fn mfx_gamma_weight(dp: &Matrix, s: &Matrix) -> Matrix {
    let sdp = s.matmul(dp).expect("S·ΔP");
    let dps = dp.matmul(s).expect("ΔP·S");
    let sdps = sdp.matmul(s).expect("S·ΔP·S");
    let t1 = hadamard(dp, &sdps); // ΔP∘(SΔPS)
    let t2 = hadamard(&sdp, &dps); // (SΔP)∘(ΔPS)
    let n = dp.rows();
    let mut w = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = -0.125 * (t1[(i, j)] + t2[(i, j)]);
        }
    }
    w
}

// --- OFX: one-center two-electron integrals (M5, experimental) ---
//
// The onsite Fock exchange correction needs the *real* one-center two-electron exchange integrals
// over the GFN1 STO-nG basis (the Mulliken/`U` approximation cannot reproduce the p/d/f-channel
// integrals). They are built by McMurchie–Davidson, reusing the same `hermite_e` + `boys`
// machinery as the nuclear-attraction integrals (`crate::nmr`): for four AOs on the *same* atom the
// bra product `P` and ket product `Q` both sit at that atom, so `R_PQ = 0`, the Hermite expansions
// use `xpa = xpb = 0` and `K_ab = 1`, and the Hermite–Coulomb integrals carry the reduced exponent
// `α = pq/(p+q)`. Reference for the OFX construction: Rüdenberg, *J. Chem. Phys.* **19**, 1459
// (1951); Domínguez et al. (onsite DFTB). (Feasibility-gated milestone: the integral primitive +
// its analytic gate land first; the exchange contraction + Mulliken-difference correction build on
// it.)

/// One-center two-electron Coulomb integral `(μν|κλ)` for four **primitive** Cartesian Gaussians on
/// the same center (powers `lμ…lλ = [x,y,z]`, exponents `a…d`), via McMurchie–Davidson. Analytic
/// check (all-`s`, exponents all `ζ`): `2π^{5/2}/(p q √(p+q))` with `p=q=2ζ` = `π^{5/2}/(4 ζ^{5/2})`.
#[allow(clippy::too_many_arguments)]
fn onsite_eri_primitive(
    lmu: [usize; 3],
    lnu: [usize; 3],
    lka: [usize; 3],
    lla: [usize; 3],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> f64 {
    onsite_eri_primitive_core(lmu, lnu, lka, lla, a, b, c, d, 1.0)
}

/// **Long-range** (`erf(ωr)/r`-screened) one-center two-electron integral for OFX. Same MMD
/// construction as the full-range integral but the Hermite–Coulomb integrals use the attenuated
/// exponent `α' = αβ` and are scaled by `√β`, with `β = ω²/(ω²+α)`, `α = pq/(p+q)` (the Hermite
/// expansion coefficients of the Gaussian products are unchanged — `β` enters only the operator).
/// `ω→∞` (`β→1`) recovers the full-range integral; `ω→0` → 0; `ω ≤ 0` → 0 (no long-range part).
/// Per-primitive reference (the production path is the factored [`build_onsite_eri_tensor`]); kept
/// as the screened-integral oracle for `onsite_eri_lr_screening`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn onsite_eri_primitive_lr(
    lmu: [usize; 3],
    lnu: [usize; 3],
    lka: [usize; 3],
    lla: [usize; 3],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    omega: f64,
) -> f64 {
    if omega <= 0.0 {
        return 0.0;
    }
    let alpha = (a + b) * (c + d) / (a + b + c + d);
    let beta = omega * omega / (omega * omega + alpha);
    onsite_eri_primitive_core(lmu, lnu, lka, lla, a, b, c, d, beta)
}

/// Shared one-center ERI core. `beta` is the range-separation attenuation (`1` = full-range Coulomb;
/// `β = ω²/(ω²+α)` for the long-range `erf(ωr)/r` operator): the Hermite–Coulomb integrals use the
/// reduced exponent `α·β` and the result is scaled by `√β`.
#[allow(clippy::too_many_arguments)]
fn onsite_eri_primitive_core(
    lmu: [usize; 3],
    lnu: [usize; 3],
    lka: [usize; 3],
    lla: [usize; 3],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    beta: f64,
) -> f64 {
    use crate::nmr::{hermite_coulomb, hermite_e};
    let p = a + b;
    let q = c + d;
    // Same center ⇒ xpa = xpb = 0, K_ab = 1 for both products (β does not affect these).
    let ex_bra = hermite_e(lmu[0], lnu[0], 0.0, 0.0, p, 1.0);
    let ey_bra = hermite_e(lmu[1], lnu[1], 0.0, 0.0, p, 1.0);
    let ez_bra = hermite_e(lmu[2], lnu[2], 0.0, 0.0, p, 1.0);
    let ex_ket = hermite_e(lka[0], lla[0], 0.0, 0.0, q, 1.0);
    let ey_ket = hermite_e(lka[1], lla[1], 0.0, 0.0, q, 1.0);
    let ez_ket = hermite_e(lka[2], lla[2], 0.0, 0.0, q, 1.0);
    let alpha = p * q / (p + q);
    let tmax = lmu[0] + lnu[0] + lka[0] + lla[0];
    let umax = lmu[1] + lnu[1] + lka[1] + lla[1];
    let vmax = lmu[2] + lnu[2] + lka[2] + lla[2];
    // Hermite–Coulomb integrals at coincident centers (R_PQ = 0), attenuated exponent α·β.
    let r = hermite_coulomb(tmax, umax, vmax, alpha * beta, [0.0, 0.0, 0.0]);
    let mut acc = 0.0;
    for (t, &et) in ex_bra.iter().enumerate() {
        for (u, &eu) in ey_bra.iter().enumerate() {
            for (v, &ev) in ez_bra.iter().enumerate() {
                let ebra = et * eu * ev;
                if ebra == 0.0 {
                    continue;
                }
                for (tau, &etk) in ex_ket.iter().enumerate() {
                    for (sig, &euk) in ey_ket.iter().enumerate() {
                        for (phi, &evk) in ez_ket.iter().enumerate() {
                            // (−1)^{τ+σ+φ}: the ket Hermites are differentiated w.r.t. Q.
                            let sign = if (tau + sig + phi) % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            acc += ebra * etk * euk * evk * sign * r[t + tau][u + sig][v + phi];
                        }
                    }
                }
            }
        }
    }
    acc * beta.sqrt() * 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt())
}

/// Contracted one-center two-electron Coulomb integral `(μν|κλ)` for four AOs on the **same** atom
/// (caller guarantees `μ,ν,κ,λ` share a center). Sums [`onsite_eri_primitive`] over the contraction
/// primitives and Cartesian components of all four AOs. Experimental OFX building block.
pub fn onsite_eri(
    mu: &crate::basis::AOBasisFunction,
    nu: &crate::basis::AOBasisFunction,
    kappa: &crate::basis::AOBasisFunction,
    lambda: &crate::basis::AOBasisFunction,
) -> f64 {
    let mut total = 0.0;
    for cm in &mu.components {
        let lm = [cm.power.x, cm.power.y, cm.power.z];
        for cn in &nu.components {
            let ln = [cn.power.x, cn.power.y, cn.power.z];
            for ck in &kappa.components {
                let lk = [ck.power.x, ck.power.y, ck.power.z];
                for cl in &lambda.components {
                    let ll = [cl.power.x, cl.power.y, cl.power.z];
                    let ccoef = cm.coefficient * cn.coefficient * ck.coefficient * cl.coefficient;
                    for pm in &mu.primitives {
                        for pn in &nu.primitives {
                            for pk in &kappa.primitives {
                                for pl in &lambda.primitives {
                                    let pcoef = pm.coefficient
                                        * pn.coefficient
                                        * pk.coefficient
                                        * pl.coefficient;
                                    total += ccoef
                                        * pcoef
                                        * onsite_eri_primitive(
                                            lm,
                                            ln,
                                            lk,
                                            ll,
                                            pm.exponent,
                                            pn.exponent,
                                            pk.exponent,
                                            pl.exponent,
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    total
}

/// Contracted **long-range** one-center two-electron integral `(μν|κλ)^lr` (erf(ωr)/r-screened) for
/// four AOs on the same atom — the screened analogue of [`onsite_eri`].
///
/// **Performance:** the expensive Hermite–Coulomb array `R` depends only on the primitive exponents
/// (`p,q,ω`) and the shells' maximum angular momenta, **not** the individual Cartesian-component
/// powers. So `R` is built **once per primitive quartet** (at the AOs' max ranks) and reused across
/// every component combination — removing a `~component⁴` factor versus calling
/// [`onsite_eri_primitive_lr`] per (component, primitive). The per-component work is then just the
/// cheap `hermite_e` coefficients + the index contraction.
pub fn onsite_eri_lr(
    mu: &crate::basis::AOBasisFunction,
    nu: &crate::basis::AOBasisFunction,
    kappa: &crate::basis::AOBasisFunction,
    lambda: &crate::basis::AOBasisFunction,
    omega: f64,
) -> f64 {
    use crate::nmr::{hermite_coulomb, hermite_e};
    if omega <= 0.0 {
        return 0.0;
    }
    let maxp = |ao: &crate::basis::AOBasisFunction, dir: usize| -> usize {
        ao.components
            .iter()
            .map(|c| [c.power.x, c.power.y, c.power.z][dir])
            .max()
            .unwrap_or(0)
    };
    // R index range: t+τ up to (max lμ_x + max lν_x) + (max lκ_x + max lλ_x), etc.
    let tmax = maxp(mu, 0) + maxp(nu, 0) + maxp(kappa, 0) + maxp(lambda, 0);
    let umax = maxp(mu, 1) + maxp(nu, 1) + maxp(kappa, 1) + maxp(lambda, 1);
    let vmax = maxp(mu, 2) + maxp(nu, 2) + maxp(kappa, 2) + maxp(lambda, 2);
    let pref = 2.0 * std::f64::consts::PI.powf(2.5);
    let mut total = 0.0;
    for pm in &mu.primitives {
        for pn in &nu.primitives {
            let p = pm.exponent + pn.exponent;
            for pk in &kappa.primitives {
                for pl in &lambda.primitives {
                    let q = pk.exponent + pl.exponent;
                    let alpha = p * q / (p + q);
                    let beta = omega * omega / (omega * omega + alpha);
                    // Build the Hermite–Coulomb R-array ONCE for this primitive quartet.
                    let r = hermite_coulomb(tmax, umax, vmax, alpha * beta, [0.0, 0.0, 0.0]);
                    let pscale = beta.sqrt() * pref / (p * q * (p + q).sqrt())
                        * pm.coefficient
                        * pn.coefficient
                        * pk.coefficient
                        * pl.coefficient;
                    for cm in &mu.components {
                        for cn in &nu.components {
                            let exb = hermite_e(cm.power.x, cn.power.x, 0.0, 0.0, p, 1.0);
                            let eyb = hermite_e(cm.power.y, cn.power.y, 0.0, 0.0, p, 1.0);
                            let ezb = hermite_e(cm.power.z, cn.power.z, 0.0, 0.0, p, 1.0);
                            for ck in &kappa.components {
                                for cl in &lambda.components {
                                    let exk = hermite_e(ck.power.x, cl.power.x, 0.0, 0.0, q, 1.0);
                                    let eyk = hermite_e(ck.power.y, cl.power.y, 0.0, 0.0, q, 1.0);
                                    let ezk = hermite_e(ck.power.z, cl.power.z, 0.0, 0.0, q, 1.0);
                                    let ccoef = cm.coefficient
                                        * cn.coefficient
                                        * ck.coefficient
                                        * cl.coefficient;
                                    let mut acc = 0.0;
                                    for (t, &et) in exb.iter().enumerate() {
                                        for (u, &eu) in eyb.iter().enumerate() {
                                            for (v, &ev) in ezb.iter().enumerate() {
                                                let ebra = et * eu * ev;
                                                if ebra == 0.0 {
                                                    continue;
                                                }
                                                for (ta, &etk) in exk.iter().enumerate() {
                                                    for (sg, &euk) in eyk.iter().enumerate() {
                                                        for (ph, &evk) in ezk.iter().enumerate() {
                                                            let sign = if (ta + sg + ph) % 2 == 0 {
                                                                1.0
                                                            } else {
                                                                -1.0
                                                            };
                                                            acc += ebra
                                                                * etk
                                                                * euk
                                                                * evk
                                                                * sign
                                                                * r[t + ta][u + sg][v + ph];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    total += pscale * ccoef * acc;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    total
}

/// Per-atom AO index lists.
fn atom_ao_lists(basis: &BasisSet, nat: usize) -> Vec<Vec<usize>> {
    let mut per = vec![Vec::new(); nat];
    for (i, ao) in basis.aos.iter().enumerate() {
        per[ao.atom_index].push(i);
    }
    per
}

/// Sparse Hermite density of an AO pair for one primitive pair: the list of nonzero
/// `(t, u, v, value)` where `value = Σ_{cm∈μ, cn∈ν} c_m c_n · E_x(l_mx,l_nx;p)[t] E_y[u] E_z[v]`
/// (McMurchie–Davidson Hermite-expansion coefficients of the same-center Gaussian product, `p` the
/// combined exponent). Storing only nonzeros makes the ERI contraction skip the (mostly-zero) dense
/// Hermite grid.
type HermiteDensity = Vec<(usize, usize, usize, f64)>;

/// Build the full one-center long-range ERI tensor `(μν|κλ)^lr` for one atom's AOs `aos`, returned
/// flat row-major over `{0..nao}^4` (`idx = ((μ·nao+ν)·nao+κ)·nao+λ`). This is the performance-
/// critical OFX kernel. Two layers of factorization make it tractable for d/f atoms:
///
/// 1. **Memoized Hermite–Coulomb array `R`** — `R` depends only on the primitive-exponent **pair**
///    `(p,q)` (and the atom's max rank), not on the AOs or their Cartesian powers, so each unique
///    `(p,q)` builds `R` once and reuses it across every AO quartet.
/// 2. **Pre-factored sparse bra/ket Hermite densities** — `(μν|κλ) = Σ_{p-pair,q-pair} radial ·
///    Σ_{tuv,τσφ} B_{μν}[tuv] (−1)^{τ+σ+φ} R[t+τ][u+σ][v+φ] B_{κλ}[τσφ]`, and the density `B_{μν}`
///    depends only on the AO pair `(μ,ν)` and its primitive pair — **not** on `(κ,λ)`. Computing each
///    `B` once (`nao²·prim²` builds) instead of inside the `nao⁴·prim⁴` quartet loop removes a ~`nao²`
///    factor of redundant `hermite_e` work, and storing only the nonzero `(t,u,v)` shrinks the inner
///    contraction. The per-element tensor is geometry-/ω-fixed — cache it (see [`OnsiteExchangeCache`]).
fn build_onsite_eri_tensor(aos: &[&crate::basis::AOBasisFunction], omega: f64) -> Vec<f64> {
    build_onsite_eri_tensor_impl(aos, omega, false)
}

/// Analytic `∂/∂ω` of [`build_onsite_eri_tensor`], in **one** pass (no finite difference, no
/// rebuild-at-`ω±δ`). At coincident centres each integral is a power series `Σ_n A_n β^{(n+1)/2}` in
/// `β = ω²/(ω²+α)` with ω-independent `A_n` (since `R(αβ)[tuv] = R(α)[tuv]·β^{(t+u+v)/2}` at `R=0`),
/// so `∂/∂ω = radial·(∂β/∂ω)/β · ½ · Σ_terms term·(order+1)`. Direct reference for the cached
/// [`OnsiteEriSkeleton::eval_deriv`] used in production (gated equal by `onsite_eri_skeleton_matches_direct`).
#[allow(dead_code)]
fn build_onsite_eri_tensor_omega_deriv(
    aos: &[&crate::basis::AOBasisFunction],
    omega: f64,
) -> Vec<f64> {
    build_onsite_eri_tensor_impl(aos, omega, true)
}

fn build_onsite_eri_tensor_impl(
    aos: &[&crate::basis::AOBasisFunction],
    omega: f64,
    derivative: bool,
) -> Vec<f64> {
    use crate::nmr::{hermite_coulomb, hermite_e};
    use std::collections::HashMap;
    let nao = aos.len();
    let mut tensor = vec![0.0_f64; nao * nao * nao * nao];
    if omega <= 0.0 || nao == 0 {
        return tensor;
    }
    let maxl = |dir: usize| {
        aos.iter()
            .flat_map(|ao| ao.components.iter())
            .map(|c| [c.power.x, c.power.y, c.power.z][dir])
            .max()
            .unwrap_or(0)
    };
    // R indices t+τ run up to 4·max(l) per direction (the worst-case AO quartet).
    let (tmax, umax, vmax) = (4 * maxl(0), 4 * maxl(1), 4 * maxl(2));
    let pref = 2.0 * std::f64::consts::PI.powf(2.5);

    // Pre-factor the sparse bra/ket Hermite density for every (AO pair, primitive pair). Indexed
    // `dens[a*nao+b][im*nprim[b]+in]`; the same table serves bra and ket (the ket sign is applied in
    // the contraction). This hoists every `hermite_e` evaluation out of the `nao⁴·prim⁴` quartet loop.
    let nprim: Vec<usize> = aos.iter().map(|ao| ao.primitives.len()).collect();
    let mut dens: Vec<Vec<HermiteDensity>> = vec![Vec::new(); nao * nao];
    for a in 0..nao {
        for b in 0..nao {
            let (na, nb) = (nprim[a], nprim[b]);
            let mut lists: Vec<HermiteDensity> = Vec::with_capacity(na * nb);
            for (im, pm) in aos[a].primitives.iter().enumerate() {
                let _ = im;
                for pn in aos[b].primitives.iter() {
                    let p = pm.exponent + pn.exponent;
                    // Dense accumulation over Cartesian components, then collect nonzeros.
                    let (tb, ub, vb) = (
                        aos[a]
                            .components
                            .iter()
                            .map(|c| c.power.x)
                            .max()
                            .unwrap_or(0)
                            + aos[b]
                                .components
                                .iter()
                                .map(|c| c.power.x)
                                .max()
                                .unwrap_or(0),
                        aos[a]
                            .components
                            .iter()
                            .map(|c| c.power.y)
                            .max()
                            .unwrap_or(0)
                            + aos[b]
                                .components
                                .iter()
                                .map(|c| c.power.y)
                                .max()
                                .unwrap_or(0),
                        aos[a]
                            .components
                            .iter()
                            .map(|c| c.power.z)
                            .max()
                            .unwrap_or(0)
                            + aos[b]
                                .components
                                .iter()
                                .map(|c| c.power.z)
                                .max()
                                .unwrap_or(0),
                    );
                    let mut grid = vec![0.0_f64; (tb + 1) * (ub + 1) * (vb + 1)];
                    for cm in &aos[a].components {
                        for cn in &aos[b].components {
                            let ex = hermite_e(cm.power.x, cn.power.x, 0.0, 0.0, p, 1.0);
                            let ey = hermite_e(cm.power.y, cn.power.y, 0.0, 0.0, p, 1.0);
                            let ez = hermite_e(cm.power.z, cn.power.z, 0.0, 0.0, p, 1.0);
                            let cc = cm.coefficient * cn.coefficient;
                            for (t, &et) in ex.iter().enumerate() {
                                if et == 0.0 {
                                    continue;
                                }
                                for (u, &eu) in ey.iter().enumerate() {
                                    let etu = et * eu;
                                    if etu == 0.0 {
                                        continue;
                                    }
                                    for (v, &ev) in ez.iter().enumerate() {
                                        let w = cc * etu * ev;
                                        if w != 0.0 {
                                            grid[(t * (ub + 1) + u) * (vb + 1) + v] += w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut list: HermiteDensity = Vec::new();
                    for t in 0..=tb {
                        for u in 0..=ub {
                            for v in 0..=vb {
                                let w = grid[(t * (ub + 1) + u) * (vb + 1) + v];
                                if w != 0.0 {
                                    list.push((t, u, v, w));
                                }
                            }
                        }
                    }
                    lists.push(list);
                }
            }
            dens[a * nao + b] = lists;
        }
    }

    let mut rmemo: HashMap<(u64, u64), Vec<Vec<Vec<f64>>>> = HashMap::new();
    // The one-center ERI `(μν|κλ)` has the full 8-fold permutational symmetry, so only the canonical
    // quartets (`ai≥bi`, `ci≥di`, bra pair ≥ ket pair) are *evaluated*; each fills its ≤8 equivalent
    // tensor entries. This cuts the (dominant) contraction work ~8× for the per-element build.
    let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nao + b) * nao + c) * nao + d;
    for ai in 0..nao {
        for bi in 0..=ai {
            let bralists = &dens[ai * nao + bi];
            let nbi = nprim[bi];
            for ci in 0..=ai {
                let di_max = if ci == ai { bi } else { ci };
                for di in 0..=di_max {
                    let ketlists = &dens[ci * nao + di];
                    let ndi = nprim[di];
                    let mut val = 0.0;
                    for (im, pm) in aos[ai].primitives.iter().enumerate() {
                        for (inn, pn) in aos[bi].primitives.iter().enumerate() {
                            let p = pm.exponent + pn.exponent;
                            let bra = &bralists[im * nbi + inn];
                            if bra.is_empty() {
                                continue;
                            }
                            let cb = pm.coefficient * pn.coefficient;
                            for (ik, pk) in aos[ci].primitives.iter().enumerate() {
                                for (il, pl) in aos[di].primitives.iter().enumerate() {
                                    let ket = &ketlists[ik * ndi + il];
                                    if ket.is_empty() {
                                        continue;
                                    }
                                    let q = pk.exponent + pl.exponent;
                                    let alpha = p * q / (p + q);
                                    let beta = omega * omega / (omega * omega + alpha);
                                    let key = (p.to_bits(), q.to_bits());
                                    let r = rmemo.entry(key).or_insert_with(|| {
                                        hermite_coulomb(tmax, umax, vmax, alpha * beta, [0.0; 3])
                                    });
                                    let radial = beta.sqrt() * pref / (p * q * (p + q).sqrt())
                                        * cb
                                        * pk.coefficient
                                        * pl.coefficient;
                                    let mut sub = 0.0;
                                    let mut sub_order = 0.0; // Σ term·order (derivative mode only)
                                    for &(t, u, v, bv) in bra {
                                        for &(ta, sg, ph, kv) in ket {
                                            let sign =
                                                if (ta + sg + ph) % 2 == 0 { 1.0 } else { -1.0 };
                                            let term = bv * kv * sign * r[t + ta][u + sg][v + ph];
                                            sub += term;
                                            if derivative {
                                                sub_order +=
                                                    term * (t + ta + u + sg + v + ph) as f64;
                                            }
                                        }
                                    }
                                    if derivative {
                                        // ∂β/∂ω / β = 2α/(ω(ω²+α)); ∂val/∂ω = radial·(∂β/∂ω/β)·½·Σ term·(order+1).
                                        let dbeta_over_beta =
                                            2.0 * alpha / (omega * (omega * omega + alpha));
                                        val += radial * dbeta_over_beta * 0.5 * (sub_order + sub);
                                    } else {
                                        val += radial * sub;
                                    }
                                }
                            }
                        }
                    }
                    // Scatter to all 8 permutational images (`(μν|κλ)=(νμ|κλ)=(μν|λκ)=(κλ|μν)=…`).
                    tensor[idx(ai, bi, ci, di)] = val;
                    tensor[idx(bi, ai, ci, di)] = val;
                    tensor[idx(ai, bi, di, ci)] = val;
                    tensor[idx(bi, ai, di, ci)] = val;
                    tensor[idx(ci, di, ai, bi)] = val;
                    tensor[idx(di, ci, ai, bi)] = val;
                    tensor[idx(ci, di, bi, ai)] = val;
                    tensor[idx(di, ci, bi, ai)] = val;
                }
            }
        }
    }
    tensor
}

/// ω-independent **skeleton** of a one-center long-range ERI tensor. At coincident centres each
/// integral is a power series in `β = ω²/(ω²+α)` with ω-independent coefficients (because
/// `R(α·β)[tuv] = R(α)[tuv]·β^{(t+u+v)/2}` at `R=0`): `(μκ|νλ)^lr(ω) = Σ_{(p,q)} Σ_k c_k β^{(2k+1)/2}`.
/// Built **once per element** (the expensive contraction, [`build_onsite_eri_skeleton`]), then
/// [`eval`](Self::eval)'d at any ω in O(quartets·groups·order) — so the dynamic-ω OFX cache rebuilds in
/// ~ms per geometry step instead of rebuilding the d-atom ERIs. **Memory-efficient:** primitive
/// quartets are grouped by their `(p,q)` exponent sums (the bra/ket Hermite density depends only on
/// `p`, so the coefficient is shared and the primitive coefficients aggregate), only the **even**
/// orders are stored (odd vanish by parity), trimmed to the actual max order, and all-zero
/// `(p,q)`/quartet entries are dropped — ~`10`s of MB per d-element rather than `~85`.
struct OnsiteEriSkeleton {
    nao: usize,
    quartets: Vec<QuartetSkel>,
}

struct QuartetSkel {
    abcd: [usize; 4],
    /// Per `(p,q)` group: `(α, [c_0, c_2, c_4, …])` (even-order coefficients of `Σ_k c_k β^{(2k+1)/2}`).
    terms: Vec<(f64, Vec<f64>)>,
}

impl OnsiteEriSkeleton {
    /// Evaluate the full `nao⁴` tensor at screening `ω`.
    fn eval(&self, omega: f64) -> Vec<f64> {
        let nao = self.nao;
        let mut tensor = vec![0.0_f64; nao * nao * nao * nao];
        if omega <= 0.0 {
            return tensor;
        }
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nao + b) * nao + c) * nao + d;
        for qt in &self.quartets {
            let mut val = 0.0;
            for &(alpha, ref ce) in &qt.terms {
                let beta = omega * omega / (omega * omega + alpha);
                let mut bp = beta.sqrt(); // β^{(2k+1)/2}, k=0 ⇒ √β
                for &c in ce {
                    val += c * bp;
                    bp *= beta; // k → k+1
                }
            }
            let [a, b, c, d] = qt.abcd;
            tensor[idx(a, b, c, d)] = val;
            tensor[idx(b, a, c, d)] = val;
            tensor[idx(a, b, d, c)] = val;
            tensor[idx(b, a, d, c)] = val;
            tensor[idx(c, d, a, b)] = val;
            tensor[idx(d, c, a, b)] = val;
            tensor[idx(c, d, b, a)] = val;
            tensor[idx(d, c, b, a)] = val;
        }
        tensor
    }

    /// Evaluate the analytic `∂tensor/∂ω`: `∂/∂ω Σ_k c_k β^{(2k+1)/2} = (∂β/∂ω / β) Σ_k c_k (2k+1)/2 β^{(2k+1)/2}`.
    fn eval_deriv(&self, omega: f64) -> Vec<f64> {
        let nao = self.nao;
        let mut tensor = vec![0.0_f64; nao * nao * nao * nao];
        if omega <= 0.0 {
            return tensor;
        }
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nao + b) * nao + c) * nao + d;
        for qt in &self.quartets {
            let mut val = 0.0;
            for &(alpha, ref ce) in &qt.terms {
                let beta = omega * omega / (omega * omega + alpha);
                let dbeta_over_beta = 2.0 * alpha / (omega * (omega * omega + alpha));
                let mut bp = beta.sqrt();
                let mut s = 0.0;
                for (k, &c) in ce.iter().enumerate() {
                    s += c * bp * (2 * k + 1) as f64 / 2.0;
                    bp *= beta;
                }
                val += s * dbeta_over_beta;
            }
            let [a, b, c, d] = qt.abcd;
            tensor[idx(a, b, c, d)] = val;
            tensor[idx(b, a, c, d)] = val;
            tensor[idx(a, b, d, c)] = val;
            tensor[idx(b, a, d, c)] = val;
            tensor[idx(c, d, a, b)] = val;
            tensor[idx(d, c, a, b)] = val;
            tensor[idx(c, d, b, a)] = val;
            tensor[idx(d, c, b, a)] = val;
        }
        tensor
    }
}

/// Build the ω-independent [`OnsiteEriSkeleton`] for one atom's AOs (see its docs for the math and the
/// memory layout). This carries the per-element contraction cost (the expensive part) exactly once.
fn build_onsite_eri_skeleton(aos: &[&crate::basis::AOBasisFunction]) -> OnsiteEriSkeleton {
    use crate::nmr::{hermite_coulomb, hermite_e};
    use std::collections::HashMap;
    let nao = aos.len();
    if nao == 0 {
        return OnsiteEriSkeleton {
            nao,
            quartets: Vec::new(),
        };
    }
    let maxl = |dir: usize| {
        aos.iter()
            .flat_map(|ao| ao.components.iter())
            .map(|c| [c.power.x, c.power.y, c.power.z][dir])
            .max()
            .unwrap_or(0)
    };
    let (tmax, umax, vmax) = (4 * maxl(0), 4 * maxl(1), 4 * maxl(2));
    let pref = 2.0 * std::f64::consts::PI.powf(2.5);
    // The component-summed Hermite density of AO pair (a,b) at combined exponent p (depends on p only).
    let build_dens = |a: usize, b: usize, p: f64| -> HermiteDensity {
        let (tb, ub, vb) = (
            aos[a]
                .components
                .iter()
                .map(|c| c.power.x)
                .max()
                .unwrap_or(0)
                + aos[b]
                    .components
                    .iter()
                    .map(|c| c.power.x)
                    .max()
                    .unwrap_or(0),
            aos[a]
                .components
                .iter()
                .map(|c| c.power.y)
                .max()
                .unwrap_or(0)
                + aos[b]
                    .components
                    .iter()
                    .map(|c| c.power.y)
                    .max()
                    .unwrap_or(0),
            aos[a]
                .components
                .iter()
                .map(|c| c.power.z)
                .max()
                .unwrap_or(0)
                + aos[b]
                    .components
                    .iter()
                    .map(|c| c.power.z)
                    .max()
                    .unwrap_or(0),
        );
        let mut grid = vec![0.0_f64; (tb + 1) * (ub + 1) * (vb + 1)];
        for cm in &aos[a].components {
            for cn in &aos[b].components {
                let ex = hermite_e(cm.power.x, cn.power.x, 0.0, 0.0, p, 1.0);
                let ey = hermite_e(cm.power.y, cn.power.y, 0.0, 0.0, p, 1.0);
                let ez = hermite_e(cm.power.z, cn.power.z, 0.0, 0.0, p, 1.0);
                let cc = cm.coefficient * cn.coefficient;
                for (t, &et) in ex.iter().enumerate() {
                    if et == 0.0 {
                        continue;
                    }
                    for (u, &eu) in ey.iter().enumerate() {
                        let etu = et * eu;
                        if etu == 0.0 {
                            continue;
                        }
                        for (v, &ev) in ez.iter().enumerate() {
                            let w = cc * etu * ev;
                            if w != 0.0 {
                                grid[(t * (ub + 1) + u) * (vb + 1) + v] += w;
                            }
                        }
                    }
                }
            }
        }
        let mut list: HermiteDensity = Vec::new();
        for t in 0..=tb {
            for u in 0..=ub {
                for v in 0..=vb {
                    let w = grid[(t * (ub + 1) + u) * (vb + 1) + v];
                    if w != 0.0 {
                        list.push((t, u, v, w));
                    }
                }
            }
        }
        list
    };
    // Per AO pair (a,b): the distinct combined exponents p, each with the aggregated primitive
    // coefficient Σ c_m c_n and the (shared) Hermite density.
    let mut pairdens: Vec<Vec<(f64, f64, HermiteDensity)>> = vec![Vec::new(); nao * nao];
    for a in 0..nao {
        for b in 0..nao {
            let mut by_p: HashMap<u64, usize> = HashMap::new();
            let mut list: Vec<(f64, f64, HermiteDensity)> = Vec::new();
            for pm in &aos[a].primitives {
                for pn in &aos[b].primitives {
                    let p = pm.exponent + pn.exponent;
                    let coef = pm.coefficient * pn.coefficient;
                    match by_p.get(&p.to_bits()) {
                        Some(&i) => list[i].1 += coef,
                        None => {
                            by_p.insert(p.to_bits(), list.len());
                            list.push((p, coef, build_dens(a, b, p)));
                        }
                    }
                }
            }
            pairdens[a * nao + b] = list;
        }
    }
    // R(α) memoized by (p,q) (ω-independent, β = 1 ⇒ argument α).
    let mut rmemo: HashMap<(u64, u64), Vec<Vec<Vec<f64>>>> = HashMap::new();
    let mut quartets: Vec<QuartetSkel> = Vec::new();
    for ai in 0..nao {
        for bi in 0..=ai {
            let bral = &pairdens[ai * nao + bi];
            for ci in 0..=ai {
                let di_max = if ci == ai { bi } else { ci };
                for di in 0..=di_max {
                    let ketl = &pairdens[ci * nao + di];
                    let mut terms: Vec<(f64, Vec<f64>)> = Vec::new();
                    for &(p, cb, ref bra) in bral {
                        if bra.is_empty() {
                            continue;
                        }
                        for &(q, cq, ref ket) in ketl {
                            if ket.is_empty() {
                                continue;
                            }
                            let alpha = p * q / (p + q);
                            let r = rmemo.entry((p.to_bits(), q.to_bits())).or_insert_with(|| {
                                hermite_coulomb(tmax, umax, vmax, alpha, [0.0; 3])
                            });
                            let g_base = pref / (p * q * (p + q).sqrt()) * cb * cq;
                            // Bin the contraction by Hermite order (even orders only; odd vanish at R=0).
                            let mut ce: Vec<f64> = Vec::new();
                            for &(t, u, v, bv) in bra {
                                for &(ta, sg, ph, kv) in ket {
                                    let order = t + ta + u + sg + v + ph;
                                    if order % 2 != 0 {
                                        continue;
                                    }
                                    let sign = if (ta + sg + ph) % 2 == 0 { 1.0 } else { -1.0 };
                                    let k = order / 2;
                                    if k >= ce.len() {
                                        ce.resize(k + 1, 0.0);
                                    }
                                    ce[k] += bv * kv * sign * r[t + ta][u + sg][v + ph];
                                }
                            }
                            for c in ce.iter_mut() {
                                *c *= g_base;
                            }
                            while ce.last() == Some(&0.0) {
                                ce.pop();
                            }
                            if !ce.is_empty() {
                                terms.push((alpha, ce));
                            }
                        }
                    }
                    if !terms.is_empty() {
                        quartets.push(QuartetSkel {
                            abcd: [ai, bi, ci, di],
                            terms,
                        });
                    }
                }
            }
        }
    }
    OnsiteEriSkeleton { nao, quartets }
}

/// **Refined onsite long-range exchange kernel** `K_onsite,refined^lr[ΔP]` (the "refined" half of
/// OFX): the exact one-center exchange contraction `K_{μν} = −½ Σ_{κλ∈A} (μκ|νλ)^lr ΔP_{κλ}` for
/// `μ,ν` on atom `A` (and `κ,λ` on the same atom), from the real screened one-center ERIs (the
/// memoized [`build_onsite_eri_tensor`]). The kernel operator is **self-adjoint** (`(μκ|νλ)=(κμ|λν)`
/// by ERI symmetry), so the energy is `½Tr[ΔP·K]` and the Fock contribution is exactly `K[ΔP]`. The
/// full OFX correction subtracts the Mulliken-approximated onsite exchange already in MFX (next
/// step). The per-atom ERI tensor is geometry-/ω-fixed; cache it across the SCC. Non-periodic.
pub fn onsite_refined_exchange_kernel(
    basis: &BasisSet,
    nat: usize,
    dp: &Matrix,
    omega: f64,
) -> Matrix {
    let per_atom = atom_ao_lists(basis, nat);
    let n = basis.len();
    let mut k = Matrix::zeros(n, n);
    for aos_idx in &per_atom {
        let aos: Vec<&crate::basis::AOBasisFunction> =
            aos_idx.iter().map(|&i| &basis.aos[i]).collect();
        let nao = aos.len();
        // Geometry-fixed one-center ERI tensor `(μν|κλ)^lr` (memoized R; the expensive part).
        let tensor = build_onsite_eri_tensor(&aos, omega);
        for (mi, &mu) in aos_idx.iter().enumerate() {
            for (ni, &nu) in aos_idx.iter().enumerate() {
                let mut acc = 0.0;
                for (ki, &ka) in aos_idx.iter().enumerate() {
                    for (li, &la) in aos_idx.iter().enumerate() {
                        // (μκ|νλ)^lr: exchange pairs μ↔κ (electron 1), ν↔λ (electron 2).
                        acc += tensor[((mi * nao + ki) * nao + ni) * nao + li] * dp[(ka, la)];
                    }
                }
                k[(mu, nu)] = -0.5 * acc;
            }
        }
    }
    k
}

/// **Mulliken-approximated onsite long-range exchange kernel** — the same-atom restriction of the
/// MFX kernel: `K_{μν} = −⅛ Σ_{σλ∈A} ΔP_{σλ} S_{μσ}S_{νλ}(Γ_{μν}+Γ_{μλ}+Γ_{σν}+Γ_{σλ})` for `μ,ν` on
/// atom `A`. This is exactly the onsite part that [`mfx_kernel`] already contributes, so subtracting
/// it from [`onsite_refined_exchange_kernel`] yields a pure *correction* (OFX) with no double count.
pub fn onsite_mulliken_exchange_kernel(
    basis: &BasisSet,
    nat: usize,
    s: &Matrix,
    gamma: &Matrix,
    dp: &Matrix,
) -> Matrix {
    let per_atom = atom_ao_lists(basis, nat);
    let n = basis.len();
    let mut k = Matrix::zeros(n, n);
    for aos in &per_atom {
        for &mu in aos {
            for &nu in aos {
                let mut acc = 0.0;
                for &sg in aos {
                    for &la in aos {
                        acc += dp[(sg, la)]
                            * s[(mu, sg)]
                            * s[(nu, la)]
                            * (gamma[(mu, nu)]
                                + gamma[(mu, la)]
                                + gamma[(sg, nu)]
                                + gamma[(sg, la)]);
                    }
                }
                k[(mu, nu)] = -0.125 * acc;
            }
        }
    }
    k
}

/// **OFX onsite Fock-exchange correction** `K_OFX = K_onsite,refined^lr − K_onsite,Mulliken^lr`: the
/// difference between the exact one-center long-range exchange (real ERIs) and the Mulliken
/// approximation MFX already applies onsite. Self-adjoint (both halves are) ⇒ `F = ∂E/∂P = K_OFX[ΔP]`,
/// `E = ½Tr[ΔP·K_OFX]`. `gamma` must be the same AO×AO `γ^lr` (built with the same `ω`) that MFX uses,
/// so the correction is consistent. Experimental; off by default.
pub fn onsite_fock_exchange_kernel(
    basis: &BasisSet,
    nat: usize,
    s: &Matrix,
    gamma: &Matrix,
    dp: &Matrix,
    omega: f64,
) -> Matrix {
    let refined = onsite_refined_exchange_kernel(basis, nat, dp, omega);
    let mulliken = onsite_mulliken_exchange_kernel(basis, nat, s, gamma, dp);
    let n = basis.len();
    let mut k = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            k[(i, j)] = refined[(i, j)] - mulliken[(i, j)];
        }
    }
    k
}

/// Structural signature of one atom's AO basis: the Cartesian powers, primitive exponents, and
/// contraction coefficients of all its AOs (in atom-local order). Two atoms with the same signature
/// have *identical* primary AO bases — i.e. they are the same element — so they share an onsite ERI
/// tensor. (Geometry never enters: one-center ERIs depend only on the basis.)
fn element_signature(aos: &[&crate::basis::AOBasisFunction]) -> Vec<u8> {
    let mut sig = Vec::with_capacity(aos.len() * 16);
    for ao in aos {
        sig.extend_from_slice(&(ao.components.len() as u32).to_le_bytes());
        for c in &ao.components {
            sig.push(c.power.x as u8);
            sig.push(c.power.y as u8);
            sig.push(c.power.z as u8);
            sig.extend_from_slice(&c.coefficient.to_bits().to_le_bytes());
        }
        sig.extend_from_slice(&(ao.primitives.len() as u32).to_le_bytes());
        for p in &ao.primitives {
            sig.extend_from_slice(&p.exponent.to_bits().to_le_bytes());
            sig.extend_from_slice(&p.coefficient.to_bits().to_le_bytes());
        }
    }
    sig
}

/// Geometry-independent **per-element onsite ERI cache** for OFX. The one-center long-range two-
/// electron tensor `(μν|κλ)^lr` of an atom depends only on that atom's AO basis (exponents, powers,
/// coefficients) and `ω` — *never on geometry* — so the expensive [`build_onsite_eri_tensor`] is run
/// **once per unique element** and reused across every atom of that element, every SCC iteration, and
/// every geometry step (e.g. a whole optimisation). This makes OFX practical even for d/f elements,
/// where a single tensor build is the dominant cost. Build once with [`OnsiteExchangeCache::build`]
/// and pass `&cache` to the cached kernels.
pub struct OnsiteExchangeCache {
    /// Per atom: index into `tensors` of that atom's element tensor.
    atom_tensor: Vec<usize>,
    /// Per atom: its global AO indices in atom-local order (the order the tensor is laid out in).
    atom_aos: Vec<Vec<usize>>,
    /// Unique-element flat `nao⁴` onsite ERI tensors (`idx = ((μ·nao+ν)·nao+κ)·nao+λ`), shared
    /// (`Arc`) with the process-global memo so no allocation is duplicated across geometries.
    tensors: Vec<Arc<Vec<f64>>>,
    /// `nao` (AOs on the atom) for each entry of `tensors`.
    tensor_nao: Vec<usize>,
}

/// Upper bound on the number of `(element, ω)` entries retained in the cross-geometry onsite-ERI
/// memo. A static-ω run has only `n_elements` distinct keys (well under this); a **dynamic-ω**
/// optimisation produces a new key every geometry step, so the cap turns an otherwise unbounded
/// per-step leak into bounded reuse (cleared, then rebuilt cheaply from the ω-independent skeleton).
const ONSITE_ERI_MEMO_CAP: usize = 256;

/// Process-global memo of per-element onsite ERI tensors, keyed by `(element_signature, ω.to_bits())`.
/// The one-center ERIs depend only on the element basis and `ω` (never on geometry), and they are
/// deterministic, so this is shared safely across threads and across *every* `run_electronic` call —
/// the expensive d/f-element tensor builds happen **once per process** (not once per geometry step),
/// which is what makes a *static-ω* OFX geometry optimisation tractable. Bounded by
/// [`ONSITE_ERI_MEMO_CAP`] so a *dynamic-ω* run (new ω each step) cannot grow it without limit. `Arc`
/// so a cache borrows the shared allocation rather than copying it.
#[allow(clippy::type_complexity)]
fn global_onsite_eri_memo(
) -> &'static Mutex<std::collections::HashMap<(Vec<u8>, u64), Arc<Vec<f64>>>> {
    static MEMO: OnceLock<Mutex<std::collections::HashMap<(Vec<u8>, u64), Arc<Vec<f64>>>>> =
        OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Process-global memo of per-element **ω-independent ERI skeletons** ([`OnsiteEriSkeleton`]), keyed by
/// the element signature only. The skeleton carries the whole (expensive) one-center contraction; it
/// is built **once per process per element** and then cheaply [`eval`](OnsiteEriSkeleton::eval)'d at
/// each atom's `ω_AA`. This is what makes a **dynamic-ω** OFX optimisation tractable: the screening
/// changes every geometry step, but the skeleton (which the heavy d/f-element work lives in) is
/// reused, so each step's tensor is an O(quartets·groups·order) re-evaluation, not a rebuild.
fn global_onsite_eri_skeleton_memo(
) -> &'static Mutex<std::collections::HashMap<Vec<u8>, Arc<OnsiteEriSkeleton>>> {
    static MEMO: OnceLock<Mutex<std::collections::HashMap<Vec<u8>, Arc<OnsiteEriSkeleton>>>> =
        OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Fetch (or build + globally cache, once per process per element) the ω-independent ERI skeleton.
fn onsite_eri_skeleton_for(aos: &[&crate::basis::AOBasisFunction]) -> Arc<OnsiteEriSkeleton> {
    let sig = element_signature(aos);
    {
        let memo = global_onsite_eri_skeleton_memo().lock().unwrap();
        if let Some(existing) = memo.get(&sig) {
            return existing.clone();
        }
    }
    // Build outside the lock (the build is the expensive part; don't serialise other elements on it).
    let sk = Arc::new(build_onsite_eri_skeleton(aos));
    let mut memo = global_onsite_eri_skeleton_memo().lock().unwrap();
    memo.entry(sig).or_insert_with(|| sk.clone()).clone()
}

impl OnsiteExchangeCache {
    /// Build the per-element onsite ERI cache for `basis` (`nat` atoms). `omega_per_atom[a]` is the
    /// onsite screening `ω_AA` for atom `a` — this **must** be the same onsite `ω` the MFX `γ^lr`
    /// uses (e.g. `η_A` under `HardnessPairwise`) so `K_OFX = refined − Mulliken` is a clean
    /// same-operator correction. Each distinct `(element, ω_AA)` tensor is fetched from (or, on first
    /// sight, added to) the [`global_onsite_eri_memo`], so it is built **once per process** and reused
    /// across all atoms of that element, all SCC iterations, and all geometry steps.
    pub fn build(basis: &BasisSet, nat: usize, omega_per_atom: &[f64]) -> Self {
        use std::collections::HashMap;
        let per_atom = atom_ao_lists(basis, nat);
        let mut tensors: Vec<Arc<Vec<f64>>> = Vec::new();
        let mut tensor_nao: Vec<usize> = Vec::new();
        let mut sig_to_idx: HashMap<(Vec<u8>, u64), usize> = HashMap::new();
        let mut atom_tensor = vec![0usize; nat];
        for (a, aos_idx) in per_atom.iter().enumerate() {
            let aos: Vec<&crate::basis::AOBasisFunction> =
                aos_idx.iter().map(|&i| &basis.aos[i]).collect();
            let omega = omega_per_atom.get(a).copied().unwrap_or(0.0);
            let key = (element_signature(&aos), omega.to_bits());
            let idx = *sig_to_idx.entry(key.clone()).or_insert_with(|| {
                // Fetch the shared per-element tensor (building + caching it globally on first sight).
                let tensor = {
                    let mut memo = global_onsite_eri_memo().lock().unwrap();
                    if let Some(existing) = memo.get(&key) {
                        existing.clone()
                    } else {
                        // Cheap re-evaluation of the (memoized, ω-independent) per-element skeleton at
                        // this atom's ω — the heavy d/f contraction is paid once, not per geometry step.
                        let t = Arc::new(onsite_eri_skeleton_for(&aos).eval(omega));
                        // Bound this cross-geometry (element, ω) memo. Under a **dynamic-ω**
                        // optimisation ω = η/s(CN) changes every geometry step (and per atom), so the
                        // key set is unbounded — an uncapped map would retain a fresh nao⁴ ERI tensor
                        // for every step of every sweep config (kept alive forever by the map) and
                        // exhaust memory (the observed cc-pVQZ/long-optimisation host OOM). A static-ω
                        // run has only a handful of distinct keys and never trips the cap; the per-build
                        // `sig_to_idx` already dedups within a geometry, and the bounded ω-independent
                        // skeleton memo keeps the re-eval after a clear cheap.
                        if memo.len() >= ONSITE_ERI_MEMO_CAP {
                            memo.clear();
                        }
                        memo.insert(key.clone(), t.clone());
                        t
                    }
                };
                tensors.push(tensor);
                tensor_nao.push(aos.len());
                tensors.len() - 1
            });
            atom_tensor[a] = idx;
        }
        Self {
            atom_tensor,
            atom_aos: per_atom,
            tensors,
            tensor_nao,
        }
    }

    /// Number of distinct element tensors referenced by this cache (≤ number of elements present).
    pub fn n_unique_elements(&self) -> usize {
        self.tensors.len()
    }
}

/// Cached counterpart of [`onsite_refined_exchange_kernel`]: the exact one-center long-range exchange
/// `K_{μν} = −½ Σ_{κλ∈A} (μκ|νλ)^lr ΔP_{κλ}` using the pre-built per-element tensors in `cache` (no
/// ERI rebuild — just the cheap `ΔP` contraction). Use inside the SCC loop.
pub fn onsite_refined_exchange_kernel_cached(
    basis: &BasisSet,
    dp: &Matrix,
    cache: &OnsiteExchangeCache,
) -> Matrix {
    use rayon::prelude::*;
    let n = basis.len();
    // Each atom contracts its own (disjoint) AO block independently ⇒ parallelise over atoms, then
    // scatter the per-atom `(μ,ν,value)` triples into the kernel (blocks never overlap).
    let blocks: Vec<Vec<(usize, usize, f64)>> = cache
        .atom_aos
        .par_iter()
        .enumerate()
        .map(|(a, aos_idx)| {
            let ti = cache.atom_tensor[a];
            let tensor = cache.tensors[ti].as_slice();
            let nao = cache.tensor_nao[ti];
            let mut out = Vec::with_capacity(aos_idx.len() * aos_idx.len());
            for (mi, &mu) in aos_idx.iter().enumerate() {
                for (ni, &nu) in aos_idx.iter().enumerate() {
                    let mut acc = 0.0;
                    for (ki, &ka) in aos_idx.iter().enumerate() {
                        for (li, &la) in aos_idx.iter().enumerate() {
                            acc += tensor[((mi * nao + ki) * nao + ni) * nao + li] * dp[(ka, la)];
                        }
                    }
                    out.push((mu, nu, -0.5 * acc));
                }
            }
            out
        })
        .collect();
    let mut k = Matrix::zeros(n, n);
    for block in blocks {
        for (mu, nu, v) in block {
            k[(mu, nu)] = v;
        }
    }
    k
}

/// Cached counterpart of [`onsite_fock_exchange_kernel`]: `K_OFX = K_onsite,refined^lr −
/// K_onsite,Mulliken^lr` using the pre-built per-element ERI tensors (`cache`). The Mulliken half is
/// cheap (no ERIs); only the refined half needs the cache. Self-adjoint ⇒ `F = ∂E/∂P = K_OFX[ΔP]`,
/// `E = ½Tr[ΔP·K_OFX]`. `gamma` must be the same `γ^lr` (same ω) MFX uses.
pub fn onsite_fock_exchange_kernel_cached(
    basis: &BasisSet,
    nat: usize,
    s: &Matrix,
    gamma: &Matrix,
    dp: &Matrix,
    cache: &OnsiteExchangeCache,
) -> Matrix {
    let refined = onsite_refined_exchange_kernel_cached(basis, dp, cache);
    let mulliken = onsite_mulliken_exchange_kernel(basis, nat, s, gamma, dp);
    let n = basis.len();
    let mut k = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            k[(i, j)] = refined[(i, j)] - mulliken[(i, j)];
        }
    }
    k
}

/// Per-atom `∂E_OFX/∂ω_AA` for the **dynamic-ω (LocalGeometry)** gradient. The on-site exchange
/// `E_OFX = ½Tr[ΔP·K_OFX] = E_refined − E_Mulliken` depends on the per-atom screening `ω_AA = ω_A`:
/// the **refined** half `E_ref = −¼ Σ_{μνκλ∈A} ΔP_{μν}ΔP_{κλ}(μκ|νλ)^lr(ω)` through the one-center
/// ERIs (its `∂/∂ω` is the **analytic** ω-derivative tensor [`build_onsite_eri_tensor_omega_deriv`] —
/// ω is factored out of the ERI build, so one pass instead of two `ω±δ` rebuilds), and the **Mulliken** half
/// `E_mull = −¼ γ_AA(ω) Σ_{μνσλ∈A} ΔP_{μν}ΔP_{σλ}S_{μσ}S_{νλ}` (all four γ-terms collapse to `γ_AA`
/// since every AO is on `A`) analytically via `∂γ_AA/∂ω`. Returns `∂E_OFX/∂ω_A` per atom; the caller
/// folds it into `∂E/∂ω_A` and chains to the forces via `∂ω_A/∂CN_A·∂CN_A/∂R`. Non-periodic.
pub fn onsite_exchange_omega_energy_derivs(
    basis: &BasisSet,
    nat: usize,
    s: &Matrix,
    dp: &Matrix,
    hardness: &[f64],
    omega_per_atom: &[f64],
) -> Vec<f64> {
    use crate::coulomb::{exchange_sigma_pair, lr_gamma_exchange_omega_deriv};
    use rayon::prelude::*;
    let per_atom = atom_ao_lists(basis, nat);
    // Atoms are independent; the per-atom ERI-tensor rebuilds (the expensive part) parallelise.
    per_atom
        .par_iter()
        .enumerate()
        .map(|(a, aos_idx)| {
            if aos_idx.is_empty() {
                return 0.0;
            }
            let w = omega_per_atom[a];
            let aos: Vec<&crate::basis::AOBasisFunction> =
                aos_idx.iter().map(|&i| &basis.aos[i]).collect();
            let nao = aos.len();
            // Refined half: ∂E_ref/∂ω = −¼ Σ ΔP_μν ΔP_κλ ∂(μκ|νλ)^lr/∂ω, via the (cached, ω-independent)
            // skeleton's analytic ω-derivative — no per-step rebuild and no FD.
            let dtensor = onsite_eri_skeleton_for(&aos).eval_deriv(w);
            let mut d_refined = 0.0;
            for (mi, &mu) in aos_idx.iter().enumerate() {
                for (ni, &nu) in aos_idx.iter().enumerate() {
                    let mut acc = 0.0;
                    for (ki, &ka) in aos_idx.iter().enumerate() {
                        for (li, &la) in aos_idx.iter().enumerate() {
                            acc += dtensor[((mi * nao + ki) * nao + ni) * nao + li] * dp[(ka, la)];
                        }
                    }
                    d_refined += dp[(mu, nu)] * acc;
                }
            }
            d_refined *= -0.25;
            // Mulliken half: ∂E_mull/∂ω = −¼ (∂γ_AA/∂ω) Σ ΔP ΔP S S.
            let sigma_aa = exchange_sigma_pair(hardness[a], hardness[a]);
            let dgamma = lr_gamma_exchange_omega_deriv(0.0, sigma_aa, w);
            let mut ssc = 0.0;
            for &mu in aos_idx {
                for &nu in aos_idx {
                    for &sg in aos_idx {
                        for &la in aos_idx {
                            ssc += dp[(mu, nu)] * dp[(sg, la)] * s[(mu, sg)] * s[(nu, la)];
                        }
                    }
                }
            }
            let d_mulliken = -0.25 * dgamma * ssc;
            d_refined - d_mulliken
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one-center `(ss|ss)` two-electron integral for a single s-primitive of exponent `ζ`
    /// (unnormalized, coefficient 1) has the exact analytic value `π^{5/2}/(4 ζ^{5/2})` (the
    /// Gaussian self-repulsion). This pins the McMurchie–Davidson prefactor + contraction.
    #[test]
    fn onsite_eri_ss_matches_analytic() {
        for &zeta in &[0.5_f64, 1.0, 2.3] {
            let val = onsite_eri_primitive(
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                zeta,
                zeta,
                zeta,
                zeta,
            );
            let analytic = std::f64::consts::PI.powf(2.5) / (4.0 * zeta.powf(2.5));
            assert!(
                (val - analytic).abs() < 1.0e-12 * analytic,
                "(ss|ss) at ζ={zeta}: {val:.10e} vs analytic {analytic:.10e}"
            );
        }
    }

    /// One-center ERIs obey the 8-fold permutational symmetry: `(μν|κλ) = (νμ|κλ) = (κλ|μν)`
    /// (swapping powers *and* exponents together). Mixed angular momenta + exponents.
    #[test]
    fn onsite_eri_permutational_symmetry() {
        // (p_x p_y | p_x p_y): every Cartesian direction has even total power (x:2, y:2), so it is
        // parity-allowed (nonzero) — a quadrupole–quadrupole one-center integral.
        let (lm, ln, lk, ll) = ([1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]);
        let (a, b, c, d) = (0.8_f64, 1.3, 0.6, 1.1);
        let base = onsite_eri_primitive(lm, ln, lk, ll, a, b, c, d);
        assert!(base.abs() > 1.0e-12, "test integral should be nonzero");
        let bra_swap = onsite_eri_primitive(ln, lm, lk, ll, b, a, c, d);
        let braket_swap = onsite_eri_primitive(lk, ll, lm, ln, c, d, a, b);
        assert!(
            (base - bra_swap).abs() < 1.0e-12 * (1.0 + base.abs()),
            "bra swap"
        );
        assert!(
            (base - braket_swap).abs() < 1.0e-12 * (1.0 + base.abs()),
            "bra↔ket swap"
        );
    }

    /// The long-range (`erf(ωr)/r`) screened one-center ERI: `ω→∞` recovers the full-range integral,
    /// `ω→0` (and `ω≤0`) vanish, and the screened `(ss|ss)` equals `√(ω²/(ω²+ζ))·` full (α=ζ for s).
    #[test]
    fn onsite_eri_lr_screening() {
        let z = 1.3_f64;
        let s = [0usize, 0, 0];
        let full = onsite_eri_primitive(s, s, s, s, z, z, z, z);
        let big = onsite_eri_primitive_lr(s, s, s, s, z, z, z, z, 1.0e6);
        assert!(
            (big - full).abs() < 1.0e-6 * full,
            "ω→∞ recovers full-range: {big} vs {full}"
        );
        // √β ∝ ω as ω→0, so the screened integral vanishes (linearly) — far below the full value.
        let small = onsite_eri_primitive_lr(s, s, s, s, z, z, z, z, 1.0e-6);
        assert!(
            small > 0.0 && small < 1.0e-4 * full,
            "ω→0 vanishes: {small}"
        );
        assert_eq!(onsite_eri_primitive_lr(s, s, s, s, z, z, z, z, 0.0), 0.0);
        let omega = 0.7_f64;
        let val = onsite_eri_primitive_lr(s, s, s, s, z, z, z, z, omega);
        let beta = omega * omega / (omega * omega + z); // α = ζ for all-s
        let analytic = beta.sqrt() * full;
        assert!(
            (val - analytic).abs() < 1.0e-12 * full,
            "screened (ss|ss): {val} vs {analytic}"
        );
    }

    /// The refined onsite long-range exchange kernel is self-adjoint, so its Fock contribution is
    /// the exact `∂E/∂P` of `E = ½Tr[ΔP·K[ΔP]]`: `Σ_ij F_ij δP_ij` matches a central finite
    /// difference of `E`. Water (O = s+p onsite block) keeps the gate fast; the integral math for
    /// higher l is pinned by the primitive gates (`onsite_eri_*`). (d/f-atom performance needs the
    /// shell-driven ERI engine — a separate optimization.)
    #[test]
    fn onsite_refined_exchange_fock_matches_energy_derivative() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let omega = 0.5;
        let dp = sym(n, 0.7);
        let dpert = sym(n, 1.9);
        let energy = |p: &Matrix| -> f64 {
            let k = onsite_refined_exchange_kernel(&basis, nat, p, omega);
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += p[(i, j)] * k[(i, j)];
                }
            }
            0.5 * e
        };
        let f = onsite_refined_exchange_kernel(&basis, nat, &dp, omega);
        let mut ana = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += f[(i, j)] * dpert[(i, j)];
            }
        }
        let eps = 1.0e-5;
        let (mut pp, mut pm) = (dp.clone(), dp.clone());
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dpert[(i, j)];
                pm[(i, j)] -= eps * dpert[(i, j)];
            }
        }
        let fd = (energy(&pp) - energy(&pm)) / (2.0 * eps);
        assert!(
            (ana - fd).abs() < 1.0e-7 + 1.0e-6 * ana.abs(),
            "onsite refined exchange Fock vs FD: {ana:.6e} vs {fd:.6e}"
        );
    }

    /// The **analytic** ω-derivative tensor [`build_onsite_eri_tensor_omega_deriv`] (ω factored out of
    /// the ERI build) matches a central finite difference of [`build_onsite_eri_tensor`] over ω,
    /// element-wise. The O onsite block (s+p) exercises nonzero Hermite orders (`β^{(order+1)/2}`).
    #[test]
    fn onsite_eri_tensor_omega_deriv_matches_fd() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system =
            crate::system::PeriodicSystem::from_xyz_str("1\nO\nO 0.0 0.0 0.0\n", 0.0, false)
                .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let aos: Vec<&crate::basis::AOBasisFunction> = basis.aos.iter().collect();
        let omega = 0.6;
        let dw = 1.0e-5;
        let ana = build_onsite_eri_tensor_omega_deriv(&aos, omega);
        let tp = build_onsite_eri_tensor(&aos, omega + dw);
        let tm = build_onsite_eri_tensor(&aos, omega - dw);
        let mut maxdiff = 0.0_f64;
        for k in 0..ana.len() {
            let fd = (tp[k] - tm[k]) / (2.0 * dw);
            maxdiff = maxdiff.max((ana[k] - fd).abs());
        }
        assert!(maxdiff < 1.0e-6, "∂tensor/∂ω vs FD: max diff {maxdiff:.3e}");
    }

    /// The memory-efficient ω-skeleton ([`build_onsite_eri_skeleton`], (p,q)-grouped + even-order)
    /// reproduces **both** the direct value tensor and its analytic ω-derivative, to round-off, at
    /// several ω. O's onsite block (s+p) exercises nonzero Hermite orders.
    #[test]
    fn onsite_eri_skeleton_matches_direct() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system =
            crate::system::PeriodicSystem::from_xyz_str("1\nO\nO 0.0 0.0 0.0\n", 0.0, false)
                .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let aos: Vec<&crate::basis::AOBasisFunction> = basis.aos.iter().collect();
        let skel = build_onsite_eri_skeleton(&aos);
        for &omega in &[0.35_f64, 0.7, 1.3] {
            let direct = build_onsite_eri_tensor(&aos, omega);
            let sk = skel.eval(omega);
            let ddir = build_onsite_eri_tensor_omega_deriv(&aos, omega);
            let dsk = skel.eval_deriv(omega);
            let mut mv = 0.0_f64;
            let mut md = 0.0_f64;
            for k in 0..direct.len() {
                mv = mv.max((direct[k] - sk[k]).abs());
                md = md.max((ddir[k] - dsk[k]).abs());
            }
            assert!(
                mv < 1.0e-10,
                "skeleton value vs direct @ω={omega}: {mv:.3e}"
            );
            assert!(
                md < 1.0e-10,
                "skeleton deriv vs direct @ω={omega}: {md:.3e}"
            );
        }
    }

    /// Performance check: a single refined-exchange kernel build for a **d-atom** (H2S, S has a
    /// 9-AO s+p+d onsite block) must complete quickly with the memoized one-center ERI tensor
    /// (R built once per unique primitive-exponent pair). Sanity-checks symmetry of the result.
    #[test]
    fn onsite_refined_exchange_feasible_for_d_atom() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
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
        let dp = sym(n, 0.5);
        let k = onsite_refined_exchange_kernel(&basis, nat, &dp, 0.5);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[(i, j)] - k[(j, i)]).abs() < 1.0e-9 * (1.0 + k[(i, j)].abs()),
                    "refined kernel not symmetric at ({i},{j})"
                );
            }
        }
    }

    /// The per-element [`OnsiteExchangeCache`] reproduces the direct refined kernel **bit-for-bit**
    /// (same ERI tensor, same contraction order) while building each element's tensor only once. On
    /// `H2O2` the two H share one tensor and the two O share another — so 4 atoms ⇒ 2 unique element
    /// tensors — and the cached kernel must equal the uncached one.
    #[test]
    fn onsite_cache_matches_uncached() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "4\nH2O2\nO 0.0 0.0 0.0\nO 1.45 0.0 0.0\nH -0.3 0.9 0.2\nH 1.75 -0.9 0.2\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let omega = 0.5;
        let dp = sym(n, 0.6);
        let cache = OnsiteExchangeCache::build(&basis, nat, &vec![omega; nat]);
        assert_eq!(cache.n_unique_elements(), 2, "H2O2 has 2 distinct elements");
        let direct = onsite_refined_exchange_kernel(&basis, nat, &dp, omega);
        let cached = onsite_refined_exchange_kernel_cached(&basis, &dp, &cache);
        // The cached path evaluates the ω-skeleton (`Σ_k c_k β^{(2k+1)/2}`, summed in a different order
        // than the direct build), so it agrees to round-off rather than bit-for-bit.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (direct[(i, j)] - cached[(i, j)]).abs()
                        < 1.0e-10 * (1.0 + direct[(i, j)].abs()),
                    "cached refined kernel differs at ({i},{j}): {} vs {}",
                    direct[(i, j)],
                    cached[(i, j)]
                );
            }
        }
    }

    /// The OFX correction `K_OFX = refined − Mulliken-onsite` is self-adjoint, so `F = ∂E/∂P` (FD),
    /// and is a genuine (nonzero) correction. Water keeps the gate fast (the integral math for
    /// higher l is pinned by the primitive `onsite_eri_*` gates). ω = 0.5 (= ω_AA for η = 0.5,
    /// matching the γ^lr matrix the Mulliken half uses).
    #[test]
    fn onsite_fock_exchange_self_adjoint_fd() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let n = basis.len();
        let ints = crate::integrals::IntegralMatrices::build(&system, &basis).unwrap();
        let s = &ints.overlap;
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let gamma = lr_exchange_gamma_matrix(
            &basis,
            nat,
            &pos,
            &eta,
            crate::coulomb::OmegaScheme::HardnessPairwise,
        );
        let omega = 0.5;
        let dp = sym(n, 0.6);
        let dpert = sym(n, 1.4);
        let energy = |p: &Matrix| -> f64 {
            let k = onsite_fock_exchange_kernel(&basis, nat, s, &gamma, p, omega);
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += p[(i, j)] * k[(i, j)];
                }
            }
            0.5 * e
        };
        let f = onsite_fock_exchange_kernel(&basis, nat, s, &gamma, &dp, omega);
        let mut ana = 0.0;
        let mut fnorm = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += f[(i, j)] * dpert[(i, j)];
                fnorm += f[(i, j)].abs();
            }
        }
        assert!(
            fnorm > 1.0e-6,
            "OFX correction (refined − Mulliken) should be nonzero"
        );
        let eps = 1.0e-5;
        let (mut pp, mut pm) = (dp.clone(), dp.clone());
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dpert[(i, j)];
                pm[(i, j)] -= eps * dpert[(i, j)];
            }
        }
        let fd = (energy(&pp) - energy(&pm)) / (2.0 * eps);
        assert!(
            (ana - fd).abs() < 1.0e-7 + 1.0e-6 * ana.abs(),
            "OFX Fock vs FD: {ana:.6e} vs {fd:.6e}"
        );
    }

    /// A symmetric `n×n` matrix from a deterministic pseudo-random seed.
    fn sym(n: usize, seed: f64) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (((i * 7 + j * 13 + 1) as f64) * seed).sin() * 0.3;
                m[(i, j)] = v;
                m[(j, i)] = v;
            }
        }
        m
    }

    /// The GEMM-factorized [`mfx_kernel`] must reproduce the explicit `O(N⁴)` four-index Mulliken
    /// exchange kernel `K_{μν} = −⅛ Σ_{σλ} ΔP_{σλ} S_{μσ} S_{νλ}(Γ_{μν}+Γ_{μλ}+Γ_{σν}+Γ_{σλ})`.
    #[test]
    fn mfx_kernel_matches_brute_force() {
        let n = 7;
        let s = sym(n, 0.7);
        let gamma = sym(n, 1.3);
        let dp = sym(n, 2.1);
        let k = mfx_kernel(&dp, &s, &gamma);
        let mut brute = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for sg in 0..n {
                    for la in 0..n {
                        acc += dp[(sg, la)]
                            * s[(mu, sg)]
                            * s[(nu, la)]
                            * (gamma[(mu, nu)]
                                + gamma[(mu, la)]
                                + gamma[(sg, nu)]
                                + gamma[(sg, la)]);
                    }
                }
                brute[(mu, nu)] = -0.125 * acc;
            }
        }
        let mut maxd = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                maxd = maxd.max((k[(i, j)] - brute[(i, j)]).abs());
            }
        }
        assert!(maxd < 1.0e-12, "GEMM MFX vs brute force: {maxd:.3e}");
    }

    /// The kernel must be symmetric (it is a Fock contribution).
    #[test]
    fn mfx_kernel_is_symmetric() {
        let n = 6;
        let s = sym(n, 0.9);
        let gamma = sym(n, 1.7);
        let dp = sym(n, 0.4);
        let k = mfx_kernel(&dp, &s, &gamma);
        let mut maxd = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                maxd = maxd.max((k[(i, j)] - k[(j, i)]).abs());
            }
        }
        assert!(maxd < 1.0e-13, "MFX kernel not symmetric: {maxd:.3e}");
    }

    /// The MFX Fock must be the exact `∂E_x/∂P`: `Σ_ij F_ij δP_ij` matches a central
    /// finite-difference of `E_x(P)` for an arbitrary symmetric perturbation `δP`. (Pure
    /// linear-algebra gate: the self-adjointness of the kernel operator.)
    #[test]
    fn mfx_fock_matches_energy_derivative() {
        let n = 6;
        let s = sym(n, 0.55);
        let gamma = sym(n, 1.1);
        let p = sym(n, 0.8);
        let p0 = sym(n, 0.2);
        let dpert = sym(n, 1.9);
        let ef = mfx_energy_fock(&p, &p0, &s, &gamma);
        let mut ana = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana += ef.fock[(i, j)] * dpert[(i, j)];
            }
        }
        let eps = 1.0e-5;
        let mut pp = p.clone();
        let mut pm = p.clone();
        for i in 0..n {
            for j in 0..n {
                pp[(i, j)] += eps * dpert[(i, j)];
                pm[(i, j)] -= eps * dpert[(i, j)];
            }
        }
        let ep = mfx_energy_fock(&pp, &p0, &s, &gamma).energy;
        let em = mfx_energy_fock(&pm, &p0, &s, &gamma).energy;
        let fd = (ep - em) / (2.0 * eps);
        assert!(
            (ana - fd).abs() < 1.0e-7 + 1.0e-6 * ana.abs(),
            "MFX Fock vs FD: {ana:.6e} vs {fd:.6e}"
        );
    }

    /// The gradient weights `∂E_x/∂S` and `∂E_x/∂Γ` must match central finite-differences of `E_x`
    /// (at fixed `ΔP`), validating the overlap-Pulay and kernel-force contributions.
    #[test]
    fn mfx_gradient_weights_match_energy_derivative() {
        let n = 6;
        let s = sym(n, 0.55);
        let gamma = sym(n, 1.1);
        let p = sym(n, 0.8);
        let p0 = sym(n, 0.2);
        let mut dp = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                dp[(i, j)] = p[(i, j)] - p0[(i, j)];
            }
        }
        let dpert = sym(n, 1.7);
        let eps = 1.0e-5;
        // --- ∂E_x/∂S ---
        let ws = mfx_overlap_weight(&dp, &s, &gamma);
        let mut ana_s = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana_s += ws[(i, j)] * dpert[(i, j)];
            }
        }
        let (mut sp, mut sm) = (s.clone(), s.clone());
        for i in 0..n {
            for j in 0..n {
                sp[(i, j)] += eps * dpert[(i, j)];
                sm[(i, j)] -= eps * dpert[(i, j)];
            }
        }
        let fd_s = (mfx_energy_fock(&p, &p0, &sp, &gamma).energy
            - mfx_energy_fock(&p, &p0, &sm, &gamma).energy)
            / (2.0 * eps);
        assert!(
            (ana_s - fd_s).abs() < 1.0e-7 + 1.0e-6 * ana_s.abs(),
            "∂E_x/∂S: {ana_s:.6e} vs {fd_s:.6e}"
        );
        // --- ∂E_x/∂Γ ---
        let wg = mfx_gamma_weight(&dp, &s);
        let mut ana_g = 0.0;
        for i in 0..n {
            for j in 0..n {
                ana_g += wg[(i, j)] * dpert[(i, j)];
            }
        }
        let (mut gp, mut gm) = (gamma.clone(), gamma.clone());
        for i in 0..n {
            for j in 0..n {
                gp[(i, j)] += eps * dpert[(i, j)];
                gm[(i, j)] -= eps * dpert[(i, j)];
            }
        }
        let fd_g = (mfx_energy_fock(&p, &p0, &s, &gp).energy
            - mfx_energy_fock(&p, &p0, &s, &gm).energy)
            / (2.0 * eps);
        assert!(
            (ana_g - fd_g).abs() < 1.0e-7 + 1.0e-6 * ana_g.abs(),
            "∂E_x/∂Γ: {ana_g:.6e} vs {fd_g:.6e}"
        );
    }

    /// The AO×AO long-range kernel must be symmetric, finite, and (with `ΔP=0` reference) give zero
    /// exchange. A two-atom H2-like geometry with finite hardness.
    #[test]
    fn lr_gamma_matrix_properties_and_zero_dp() {
        let params = match std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        {
            Some(p) => p,
            None => return,
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let nat = system.atoms.len();
        let pos: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        let eta = vec![0.5_f64; nat];
        let gamma = lr_exchange_gamma_matrix(
            &basis,
            nat,
            &pos,
            &eta,
            crate::coulomb::OmegaScheme::HardnessPairwise,
        );
        let n = basis.len();
        for i in 0..n {
            for j in 0..n {
                assert!(gamma[(i, j)].is_finite());
                assert!((gamma[(i, j)] - gamma[(j, i)]).abs() < 1.0e-14);
                assert!(gamma[(i, j)] > 0.0, "γ^lr must be positive");
            }
        }
        // ΔP = 0 → no exchange energy/Fock.
        let s = crate::integrals::IntegralMatrices::build(&system, &basis)
            .unwrap()
            .overlap;
        let zero = Matrix::zeros(n, n);
        let ef = mfx_energy_fock(&zero, &zero, &s, &gamma);
        assert!(ef.energy.abs() < 1.0e-15);
    }

    /// The neutral-atom reference density `P0` is diagonal and its Mulliken populations equal the
    /// shell reference occupations (so the exchange fluctuation `ΔP = P − P0` vanishes for a
    /// neutral-atom density). H2O.
    #[test]
    fn reference_density_is_neutral() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let basis =
            crate::basis::BasisSet::build(&system, &params, crate::basis::BasisOptions::default())
                .unwrap();
        let ints = crate::integrals::IntegralMatrices::build(&system, &basis).unwrap();
        let p0 = neutral_atom_reference_density(&basis);
        // Off-diagonal P0 is zero.
        let n = basis.len();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_eq!(p0[(i, j)], 0.0);
                }
            }
        }
        // Mulliken shell populations of P0 = the reference occupations (neutral atom).
        let qsh = crate::electronic::mulliken_shell_charges(&basis, &ints.overlap, &p0);
        for (ish, q) in qsh.iter().enumerate() {
            // qsh = reference_occ − population; for the neutral reference it must be ~0 (up to AO
            // self-overlap normalization roundoff, S_μμ ≈ 1 ± ε).
            assert!(
                q.abs() < 1.0e-7,
                "shell {ish} reference charge {q:.3e} != 0"
            );
        }
    }

    /// MFX SCC integration gate: with `lr_exchange` on, the SCC must converge to a finite total
    /// energy that **differs** from stock GFN1 (the exchange is active) but stays sane (not blown
    /// up). Off ≡ the default, so the off path is the plain GFN1 baseline. Water.
    #[test]
    fn mfx_scc_changes_energy_and_converges() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = crate::electronic::ElectronicOptions::default();
        let base = crate::electronic::run_electronic(&system, &params, opt.clone()).unwrap();
        opt.lr_exchange = true;
        let ex = crate::electronic::run_electronic(&system, &params, opt).unwrap();
        assert!(ex.total_free.is_finite(), "MFX total energy not finite");
        let de = ex.total_free - base.total_free;
        assert!(
            de.abs() > 1.0e-6,
            "MFX did not change the energy ({de:.3e})"
        );
        assert!(
            de.abs() < 1.0,
            "MFX energy shift implausibly large ({de:.3e})"
        );
    }

    /// SCF convergence robustness: a polar/asymmetric water geometry (which oscillates under the
    /// raw dual self-consistency) must converge **out-of-the-box** — `lr_exchange` auto-caps the
    /// charge mixing, so the user does not have to tune it. Uses default options.
    #[test]
    fn mfx_scc_converges_on_polar_geometry() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.78 0.55 -0.05\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let mut opt = crate::electronic::ElectronicOptions::default();
        opt.lr_exchange = true;
        let ex = crate::electronic::run_electronic(&system, &params, opt)
            .expect("MFX SCF should converge on the polar geometry (auto-capped mixing)");
        assert!(ex.total_free.is_finite());
    }

    /// The robust density-matrix SCF must give a **mixing-independent** result: with exchange on,
    /// the full-Fock-CDIIS driver does not use the charge-mixing parameter, so the same molecule at
    /// two different `mixing` values must converge to the **same** energy (the erratic
    /// mixing-dependence of the old charge-Broyden path is gone). Uses the previously-erratic polar
    /// water at default options otherwise.
    #[test]
    fn mfx_robust_scf_mixing_independent() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.02 0.01 0.10\nH 0.78 0.55 -0.05\nH -0.74 0.58 0.03\n",
            0.0,
            false,
        )
        .unwrap();
        let mut o1 = crate::electronic::ElectronicOptions::default();
        o1.lr_exchange = true;
        o1.mixing = 0.1;
        let mut o2 = o1.clone();
        o2.mixing = 0.4;
        let e1 = crate::electronic::run_electronic(&system, &params, o1)
            .unwrap()
            .total_free;
        let e2 = crate::electronic::run_electronic(&system, &params, o2)
            .unwrap()
            .total_free;
        assert!(
            (e1 - e2).abs() < 1.0e-8,
            "robust exchange SCF must be mixing-independent: {e1:.10} vs {e2:.10}"
        );
    }

    /// Size consistency: two well-separated H₂ fragments must give `E(dimer) = 2·E(monomer)` with
    /// the exchange on. Although `γ^lr` has a `1/R` tail, the Mulliken exchange is overlap-weighted
    /// (`S_{μσ}S_{νλ}`), so the inter-fragment kernel vanishes with the (exponentially decaying)
    /// overlap — the parameter-free HardnessPairwise ω (geometry-independent, pairwise-local) keeps
    /// the method size-consistent. Dispersion off to isolate the electronic size-consistency.
    #[test]
    fn mfx_size_consistent_dissociation() {
        let Some(params) = std::env::var("GFN1_XTB_PARAM")
            .ok()
            .and_then(|p| crate::params::Gfn1Parameters::from_file(p).ok())
        else {
            return;
        };
        let mk = |xyz: &str| crate::system::PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let monomer = mk("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n");
        let dimer =
            mk("4\nH2 H2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\nH 0.0 0.0 15.0\nH 0.0 0.0 15.74\n");
        let mut opt = crate::electronic::ElectronicOptions::default();
        opt.lr_exchange = true;
        opt.enable_dispersion = false;
        let e1 = crate::electronic::run_electronic(&monomer, &params, opt.clone())
            .unwrap()
            .total_free;
        let e2 = crate::electronic::run_electronic(&dimer, &params, opt)
            .unwrap()
            .total_free;
        assert!(
            (e2 - 2.0 * e1).abs() < 1.0e-5,
            "MFX not size-consistent: E(dimer)={e2:.8} vs 2·E(monomer)={:.8}",
            2.0 * e1
        );
    }
}
