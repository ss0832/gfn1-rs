// SPDX-License-Identifier: GPL-3.0-or-later

use crate::basis::BasisSet;
use crate::error::Result;
use crate::linalg::{matrix_vector_product, Matrix};
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

pub const GFN1_COULOMB_EXPONENT: f64 = 2.0;

#[derive(Clone, Debug)]
pub struct ShellChargeModel {
    pub atom_offsets: Vec<usize>,
    pub atom_shell_counts: Vec<usize>,
    pub hardness: Vec<f64>,
    pub hubbard_derivs: Vec<f64>,
    /// Maximum on-site charge-expansion order. GFN1's 2nd-order Klopman–Ohno + 3rd-order
    /// DFTB3 terms are `charge_order = 3` (the default ≡ stock GFN1). Orders `n ≥ 4` are
    /// the experimental, parameter-free **Linear Breathing-Radius** extension
    /// (see [`coulomb_energy_potential_from_matrix`]); set from
    /// `ElectronicOptions::charge_order`.
    pub charge_order: usize,
}

#[derive(Clone, Debug)]
pub struct CoulombEnergy {
    pub second_order: f64,
    pub third_order: f64,
    /// Experimental on-site charge energy of orders `4..=charge_order` (0 unless
    /// `charge_order > 3`); the Linear Breathing-Radius extension.
    pub higher_order: f64,
    pub shell_potential: Vec<f64>,
    pub atomic_charges: Vec<f64>,
}

impl ShellChargeModel {
    pub fn build(
        system: &PeriodicSystem,
        basis: &BasisSet,
        params: &Gfn1Parameters,
    ) -> Result<Self> {
        let nat = system.atoms.len();
        let mut atom_shell_counts = vec![0usize; nat];
        for shell in &basis.shells {
            atom_shell_counts[shell.atom_index] += 1;
        }

        let mut atom_offsets = vec![0usize; nat];
        let mut offset = 0usize;
        for atom in 0..nat {
            atom_offsets[atom] = offset;
            offset += atom_shell_counts[atom];
        }

        let mut hardness = vec![0.0; basis.shells.len()];
        let mut hubbard_derivs = vec![0.0; basis.shells.len()];
        for (ish, shell) in basis.shells.iter().enumerate() {
            let elem = params.element(shell.z)?;
            hardness[ish] = elem.shell_hardness(shell.angular);
            hubbard_derivs[ish] = elem.gam3_model();
        }

        Ok(Self {
            atom_offsets,
            atom_shell_counts,
            hardness,
            hubbard_derivs,
            charge_order: 3,
        })
    }

    pub fn atomic_charges(&self, basis: &BasisSet, shell_charges: &[f64]) -> Vec<f64> {
        let nat = self.atom_shell_counts.len();
        let mut out = vec![0.0; nat];
        for (ish, shell) in basis.shells.iter().enumerate() {
            out[shell.atom_index] += shell_charges[ish];
        }
        out
    }
}

pub fn effective_coulomb_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    model: &ShellChargeModel,
) -> Matrix {
    let nsh = basis.shells.len();
    let mut amat = Matrix::zeros(nsh, nsh);
    for i in 0..nsh {
        let ai = basis.shells[i].atom_index;
        for j in 0..=i {
            let aj = basis.shells[j].atom_index;
            let value = if ai == aj {
                harmonic_average(model.hardness[i], model.hardness[j])
            } else {
                let ri = system.atoms[ai].position;
                let rj = system.atoms[aj].position;
                let r = (rj - ri).norm();
                let gamma = harmonic_average(model.hardness[i], model.hardness[j]);
                effective_kernel_0d(r, gamma, GFN1_COULOMB_EXPONENT)
            };
            amat[(i, j)] = value;
            amat[(j, i)] = value;
        }
    }
    amat
}

pub fn coulomb_energy_potential(
    system: &PeriodicSystem,
    basis: &BasisSet,
    shell_charges: &[f64],
    params: &Gfn1Parameters,
) -> Result<CoulombEnergy> {
    let model = ShellChargeModel::build(system, basis, params)?;
    let amat = effective_coulomb_matrix(system, basis, &model);
    coulomb_energy_potential_from_matrix(basis, &model, shell_charges, &amat)
}

/// GFN1 isotropic electrostatics from shell charges: 2nd-order Klopman–Ohno
/// `E₂ = ½ q·A·q`, the DFTB3 on-site 3rd-order `E₃ = Σ_A (1/3) Γ_A Δq_A³` (Gaus, Cui &
/// Elstner, *J. Chem. Theory Comput.* **7**, 931 (2011); GFN1-xTB convention, Grimme,
/// Bannwarth & Shushkov, *J. Chem. Theory Comput.* **13**, 1989 (2017)), and — for
/// `model.charge_order > 3` — experimental on-site terms of orders `4..=charge_order`.
///
/// **Arbitrary-order on-site terms (Linear Breathing-Radius Model).** GFN1's on-site
/// hardness is the inverse effective electrostatic radius, `η_A(q) = γ_AA(q) = 1/R_A(q)`.
/// Assume the radius responds linearly to excess charge, `R_A(q) = R_A⁰ + λ_A Δq_A` (the
/// simplest fitting-free, Padé-free closure for hardness saturation). Then `η` is a
/// geometric series, `η(q) = γ Σ_{k≥0} (−λγ q)^k`, and matching GFN1's `(1/n)` series —
/// where `η = ∂²E/∂q² = Σ_n (n−1) X_n q^{n−2}`, so `η'(0)=2Γ ⇒ λ = −2Γ/γ²` — gives a
/// **closed form for every order**, with no new parameters:
/// ```text
///   X_n = (γ_A/(n−1)) (2Γ_A/γ_A)^{n−2},   E_n = Σ_A (1/n) X_n Δq_A^n,   ∂E_n/∂q = X_n Δq^{n−1}
/// ```
/// This reduces to `X_2 = γ`, `X_3 = Γ` (stock GFN1) and `X_4 = (4/3)Γ²/γ` (the original
/// 4th-order term, equivalently `2Γ²/γ` in the `(1/n!)` convention). At order 4 the
/// quartic `E = ½γq²[1 + ⅔x + ⅔x²]` (`x = (Γ/γ)q`) is convex/bounded (no spurious
/// inflection); higher orders are the truncated breathing-radius series.
pub fn coulomb_energy_potential_from_matrix(
    basis: &BasisSet,
    model: &ShellChargeModel,
    shell_charges: &[f64],
    amat: &Matrix,
) -> Result<CoulombEnergy> {
    let aq = matrix_vector_product(amat, shell_charges)?;
    let second_order = 0.5
        * shell_charges
            .iter()
            .zip(aq.iter())
            .map(|(q, v)| q * v)
            .sum::<f64>();

    let mut shell_potential = aq;
    let atomic_charges = model.atomic_charges(basis, shell_charges);
    let mut third_order = 0.0;
    let mut higher_order = 0.0;
    for (atom, &qat) in atomic_charges.iter().enumerate() {
        if model.atom_shell_counts[atom] == 0 {
            continue;
        }
        let offset = model.atom_offsets[atom];
        let gam3 = model.hubbard_derivs[offset];
        third_order += qat * qat * qat * gam3 / 3.0;
        let mut potential = qat * qat * gam3;
        // Experimental on-site orders n = 4..=charge_order (Linear Breathing-Radius
        // Model): X_n = (γ/(n-1))(2Γ/γ)^(n-2), E_n = (1/n) X_n Δq^n, ∂E_n/∂q = X_n Δq^(n-1).
        if model.charge_order > 3 {
            let gamma = model.hardness[offset];
            if gamma.abs() > 1.0e-8 {
                let ratio = 2.0 * gam3 / gamma; // 2Γ/γ
                for n in 4..=model.charge_order {
                    let ni = n as i32;
                    let xn = gamma / ((n - 1) as f64) * ratio.powi(ni - 2);
                    let qn1 = qat.powi(ni - 1); // Δq^(n-1)
                    higher_order += xn / (n as f64) * qn1 * qat;
                    potential += xn * qn1;
                }
            }
        }
        for local in 0..model.atom_shell_counts[atom] {
            shell_potential[offset + local] += potential;
        }
    }

    Ok(CoulombEnergy {
        second_order,
        third_order,
        higher_order,
        shell_potential,
        atomic_charges,
    })
}

/// On-site charge **anharmonicity** derivatives `(∂E/∂q, ∂²E/∂q², ∂³E/∂q³, ∂⁴E/∂q⁴)` beyond the
/// harmonic `½γΔq²` — i.e. the DFTB3 `(1/3)Γq³` and the Linear Breathing-Radius orders
/// `4..=charge_order` (`E_n = (1/n)X_n q^n`, `X_n = (γ/(n−1))(2Γ/γ)^{n−2}`). The second/third
/// derivatives are the on-site charge kernels the 2n+1 `L_axx`/`L_xxx` terms contract with the
/// charge responses; the harmonic part contributes nothing to the third.
///
/// ```text
///   ∂³E/∂q³ = 2Γ    + Σ_{n≥4}(n−1)(n−2)X_n q^{n−3}
///   ∂⁴E/∂q⁴ =         Σ_{n≥4}(n−1)(n−2)(n−3)X_n q^{n−4}
/// ```
///
/// The fourth derivative vanishes identically for stock GFN1 (`charge_order = 3`, `E = ⅓Γq³`) —
/// it is the `λ`-derivative of the `∂K/∂q` chain kernel that the directional QUARTIC response
/// stage needs, and it only becomes non-zero with the Breathing-Radius orders `n ≥ 4`.
///
/// Production consumers: the CPXTB response kernel (`∂²E/∂q²` on the same-atom
/// block, keeping Hessian/response properties consistent with `charge_order ≥ 4`
/// energies), the third-derivative ∂K/∂q chain (`∂³E/∂q³`) and the fourth-derivative
/// `D_λ(∂K/∂q)` chain (`∂⁴E/∂q⁴`).
pub(crate) fn onsite_charge_anharmonic_derivatives(
    gamma: f64,
    gam3: f64,
    charge_order: usize,
    q: f64,
) -> (f64, f64, f64, f64) {
    let mut first = gam3 * q * q; // ∂[(1/3)Γq³]
    let mut second = 2.0 * gam3 * q; // ∂²
    let mut third = 2.0 * gam3; // ∂³
    let mut fourth = 0.0; // ∂⁴ — the cubic term is exhausted at third order
    if charge_order > 3 && gamma.abs() > 1.0e-8 {
        let ratio = 2.0 * gam3 / gamma; // 2Γ/γ
        for n in 4..=charge_order {
            let ni = n as i32;
            let xn = gamma / ((n - 1) as f64) * ratio.powi(ni - 2);
            first += xn * q.powi(ni - 1);
            second += (n - 1) as f64 * xn * q.powi(ni - 2);
            third += (n - 1) as f64 * (n - 2) as f64 * xn * q.powi(ni - 3);
            fourth += (n - 1) as f64 * (n - 2) as f64 * (n - 3) as f64 * xn * q.powi(ni - 4);
        }
    }
    (first, second, third, fourth)
}

#[inline]
pub fn harmonic_average(gi: f64, gj: f64) -> f64 {
    2.0 / (1.0 / gi + 1.0 / gj)
}

#[inline]
pub fn effective_kernel_0d(r: f64, gamma: f64, _gexp: f64) -> f64 {
    let r2 = r * r;
    let inv_g2 = 1.0 / (gamma * gamma);
    1.0 / (r2 + inv_g2).sqrt()
}

// --- Range-separated exchange kernel (v0.2.0 OFX/MFX foundation) -----------------------------
//
// The Mulliken-approximated Fock exchange (LC-DFTB: V. Lutsker, B. Aradi, T. A. Niehaus,
// J. Chem. Phys. 143, 184107 (2015); concept T. A. Niehaus & F. Della Sala) needs a *long-range*
// two-electron kernel γ^lr_AB(R;ω) defined by screening the electron–electron interaction
// `erf(ω r)/r` over the finite-width atomic charge clouds. A naive `γ_KO(R)·erf(ωR)` is NOT a
// consistent screened integral (it vanishes at R=0). We model each atom's charge cloud as a
// normalized Gaussian of width `σ_A` set by the GFN1 chemical hardness so the on-site full-range
// kernel reproduces the hardness, `γ^fr_AA(0) = η_A` ⇒ `σ_A = sqrt(2/π)/η_A`. Then the Coulomb
// (full-range) kernel between two clouds is the standard `erf(R/σ_AB)/R`, σ_AB² = σ_A² + σ_B²
// (finite at R=0), and the long-range (erf(ωr)-screened) kernel is obtained by adding the
// screening Gaussian width `1/ω` in quadrature:
//
//   γ^fr_AB(R)    = erf(R / σ_AB) / R,                       σ_AB² = σ_A² + σ_B²
//   γ^lr_AB(R;ω)  = erf(R / τ)    / R,   τ = sqrt(σ_AB² + 1/ω²)
//   γ^sr_AB(R;ω)  = γ^fr_AB(R) − γ^lr_AB(R;ω)
//
// This satisfies the physical limits exactly (verified in tests), for both R>0 and R=0:
//   γ^lr(R;0) = 0           (ω→0: τ→∞, no long-range exchange),
//   lim_{ω→∞} γ^lr(R;ω) = γ^fr(R)   (τ→σ_AB).
// Everything derives from the existing hardness η (no fitted constant); ω comes from the
// `OmegaScheme` (M3). All R=0 values use the analytic small-R limit `erf(R/w)/R → 2/(√π w)`.

/// Gaussian charge-cloud width `σ_A = sqrt(2/π)/η_A` from the atomic chemical hardness `η_A`,
/// matched so the on-site full-range kernel equals the hardness (`γ^fr_AA(0) = η_A`).
#[inline]
pub fn exchange_sigma(eta: f64) -> f64 {
    (2.0 / std::f64::consts::PI).sqrt() / eta
}

/// Combined two-cloud width `σ_AB = sqrt(σ_A² + σ_B²)` from the per-atom hardnesses.
#[inline]
pub fn exchange_sigma_pair(eta_a: f64, eta_b: f64) -> f64 {
    let sa = exchange_sigma(eta_a);
    let sb = exchange_sigma(eta_b);
    (sa * sa + sb * sb).sqrt()
}

/// `erf(R/w)/R` with the analytic `R→0` limit `2/(√π w)` (Coulomb of two Gaussian clouds of
/// combined width `w`).
#[inline]
fn erf_over_r(r: f64, w: f64) -> f64 {
    if r < 1.0e-12 {
        2.0 / (std::f64::consts::PI.sqrt() * w)
    } else {
        crate::math::erf(r / w) / r
    }
}

/// Full-range Gaussian-cloud exchange kernel `γ^fr_AB(R) = erf(R/σ_AB)/R` (finite at R=0).
#[inline]
pub fn fr_gamma_exchange(r: f64, sigma_ab: f64) -> f64 {
    erf_over_r(r, sigma_ab)
}

/// Long-range (range-separated) Gaussian-cloud exchange kernel
/// `γ^lr_AB(R;ω) = erf(R/τ)/R`, `τ = sqrt(σ_AB² + 1/ω²)`. `ω ≤ 0` returns 0 (no long-range part).
#[inline]
pub fn lr_gamma_exchange(r: f64, sigma_ab: f64, omega: f64) -> f64 {
    if omega <= 0.0 {
        return 0.0;
    }
    let tau = (sigma_ab * sigma_ab + 1.0 / (omega * omega)).sqrt();
    erf_over_r(r, tau)
}

/// Short-range complement `γ^sr_AB(R;ω) = γ^fr_AB(R) − γ^lr_AB(R;ω)`.
#[inline]
pub fn sr_gamma_exchange(r: f64, sigma_ab: f64, omega: f64) -> f64 {
    fr_gamma_exchange(r, sigma_ab) - lr_gamma_exchange(r, sigma_ab, omega)
}

/// Radial derivative `dγ^lr_AB/dR` of the long-range exchange kernel (for the analytic gradient):
/// `d/dR[erf(R/τ)/R] = [ (2/(√π τ)) e^{−R²/τ²} R − erf(R/τ) ] / R²`, `τ = sqrt(σ_AB²+1/ω²)`.
/// `0` for `ω ≤ 0` (no long-range part) and at `R → 0` (the kernel is even in `R`, so the slope
/// vanishes at the origin).
#[inline]
pub fn lr_gamma_exchange_deriv(r: f64, sigma_ab: f64, omega: f64) -> f64 {
    if omega <= 0.0 || r < 1.0e-12 {
        return 0.0;
    }
    let tau = (sigma_ab * sigma_ab + 1.0 / (omega * omega)).sqrt();
    let x = r / tau;
    let gaussian = (2.0 / (std::f64::consts::PI.sqrt() * tau)) * (-x * x).exp();
    (gaussian * r - crate::math::erf(x)) / (r * r)
}

/// Derivative `∂γ^lr_AB/∂ω` of the long-range exchange kernel with respect to the range-separation
/// `ω` (for the dynamic-ω, `LocalGeometry`, analytic force). With `τ = √(σ²+1/ω²)`,
/// `∂γ/∂ω = (∂γ/∂τ)(∂τ/∂ω) = [−(2/(√π τ²)) e^{−R²/τ²}]·[−1/(ω³ τ)] = (2/(√π τ³ ω³)) e^{−R²/τ²}`,
/// valid at `R = 0` too (the kernel `2/(√π τ)` there has the same `∂/∂τ`). `0` for `ω ≤ 0`.
#[inline]
pub fn lr_gamma_exchange_omega_deriv(r: f64, sigma_ab: f64, omega: f64) -> f64 {
    if omega <= 0.0 {
        return 0.0;
    }
    let tau = (sigma_ab * sigma_ab + 1.0 / (omega * omega)).sqrt();
    let x = r / tau;
    (2.0 / (std::f64::consts::PI.sqrt() * tau.powi(3) * omega.powi(3))) * (-x * x).exp()
}

/// Enforce the one **physically necessary** bound on a range-separation `ω`: non-negativity
/// (`ω ≥ 0`). A negative ω flips the sign of `erf(ωr)` and is meaningless; `ω = 0` is the valid
/// "no long-range exchange" limit (pure GFN1). There is deliberately **no upper bound**: ω has no
/// hard physical ceiling — `ω → ∞` merely converges (smoothly) to the finite full-range kernel
/// `γ^fr` (`τ = √(σ²+1/ω²) → σ`), so every `ω ≥ 0` is numerically well-conditioned. (An earlier
/// `≤ 1` cap was dropped as weakly justified; hardness-derived ω is ~0.2–0.5 bohr⁻¹ anyway.)
#[inline]
pub fn sanitize_omega(omega: f64) -> f64 {
    omega.max(0.0)
}

/// **HardnessPairwise** ω (parameter-free, geometry-independent): `ω_A = η_A` (the Klopman–Ohno
/// inverse radius = chemical hardness), combined for a pair as the **harmonic mean**
/// `ω_AB = 2 ω_A ω_B/(ω_A+ω_B)` (= arithmetic mean of the lengths `1/ω`; symmetric, recovers the
/// homonuclear limit, lies between the two atomic values), guaranteed `≥ 0`. Local to the pair ⇒
/// size-consistent; depends only on atomic constants ⇒ no `∂ω/∂R`.
#[inline]
pub fn omega_hardness_pairwise(eta_a: f64, eta_b: f64) -> f64 {
    let denom = eta_a + eta_b;
    let w = if denom > 0.0 {
        2.0 * eta_a * eta_b / denom
    } else {
        0.0
    };
    sanitize_omega(w)
}

/// Range-separation **ω scheme** for the long-range exchange (M3). Selects how the per-atom-pair
/// `ω_AB` is constructed; all variants are **parameter-free** (no fitted constant) except the
/// reference `Fixed`. Default = [`OmegaScheme::HardnessPairwise`] (size-consistent, geometry-
/// independent). The geometry-dependent `LocalGeometry` and the diagnostic-only
/// `GlobalAdaptiveDiagnostic` are reserved for later (they need `∂ω/∂R` / are non-size-consistent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OmegaScheme {
    /// A single user-supplied ω for every pair (comparison/reference baseline; not parameter-free).
    Fixed(f64),
    /// `ω_A = η_A`, combined as the harmonic mean `ω_AB = 2ω_Aω_B/(ω_A+ω_B)`
    /// ([`omega_hardness_pairwise`]). Parameter-free, size-consistent, geometry-independent (the
    /// default).
    HardnessPairwise,
    /// **Experimental, geometry-dependent.** The hardness radius `1/η_A` is modulated by a smooth,
    /// dimensionless local size factor `s_A(R)` ([`omega_local_geometry`]): `ω_A = η_A/s_A`, combined
    /// pairwise as the harmonic mean ([`omega_pair_local_geometry`]). Parameter-free when `s_A` is
    /// built from existing GFN1/D3 quantities (the cube root of the CN-interpolated atomic
    /// polarizability ratio). Size-consistent (the GFN1 CN has a finite cutoff, so a remote fragment
    /// leaves a local `s_A` unchanged), but its analytic gradient needs the extra `∂ω/∂R = −η ∂s/∂R / s²`
    /// term (via `dCN/dR`). The pairwise `omega_pair` below cannot evaluate it (it has no `s_A`); use
    /// [`omega_pair_local_geometry`] where the per-atom size factors are available.
    LocalGeometry,
}

impl Default for OmegaScheme {
    fn default() -> Self {
        OmegaScheme::HardnessPairwise
    }
}

/// The pairwise range-separation `ω_AB` for the chosen [`OmegaScheme`] from the atomic hardnesses
/// `η_A,η_B`. Always `≥ 0` ([`sanitize_omega`]).
#[inline]
pub fn omega_pair(scheme: OmegaScheme, eta_a: f64, eta_b: f64) -> f64 {
    match scheme {
        OmegaScheme::Fixed(w) => sanitize_omega(w),
        OmegaScheme::HardnessPairwise => omega_hardness_pairwise(eta_a, eta_b),
        // LocalGeometry needs per-atom size factors `s_A` (use `omega_pair_local_geometry`); with
        // only the hardnesses available it reduces to the unmodulated `s_A = 1` limit = HardnessPairwise.
        OmegaScheme::LocalGeometry => omega_hardness_pairwise(eta_a, eta_b),
    }
}

/// **LocalGeometry** atomic ω (M3, experimental): the hardness radius `ℓ_A^0 = 1/η_A` modulated by a
/// dimensionless, environment-dependent size factor `s_A` (`s_A = 1` = isolated-atom reference;
/// larger `s_A` = a larger/softer atom ⇒ longer range ⇒ smaller ω), `ω_A = η_A / s_A`. Parameter-free
/// when `s_A` is built from existing GFN1/D3 quantities — e.g. `s_A = (α_A(CN)/α_A(0))^{1/3}` with the
/// D3 CN-interpolated atomic polarizability. Guaranteed `≥ 0`; reduces to `η_A` at `s_A = 1`.
#[inline]
pub fn omega_local_geometry(eta: f64, s: f64) -> f64 {
    if s <= 0.0 {
        return sanitize_omega(eta);
    }
    sanitize_omega(eta / s)
}

/// Pairwise **LocalGeometry** ω: the harmonic mean of the size-modulated atomic ω
/// ([`omega_local_geometry`]) — same length-averaging as [`omega_hardness_pairwise`], to which it
/// reduces when both `s = 1`. Symmetric; `≥ 0`; size-consistent (a remote fragment leaves each local
/// `s_A` unchanged because the GFN1 CN has a finite cutoff).
#[inline]
pub fn omega_pair_local_geometry(eta_a: f64, s_a: f64, eta_b: f64, s_b: f64) -> f64 {
    let wa = omega_local_geometry(eta_a, s_a);
    let wb = omega_local_geometry(eta_b, s_b);
    let denom = wa + wb;
    let w = if denom > 0.0 {
        2.0 * wa * wb / denom
    } else {
        0.0
    };
    sanitize_omega(w)
}

/// Parameter-free **LocalGeometry size factor** from the GFN1-Hamiltonian coordination number:
/// `s_A = (1 + CN_A)^(−1/3)`. A more-coordinated atom is more compressed (effective volume
/// `∝ 1/(1+CN)`, length `∝ volume^(1/3)`) ⇒ smaller `s` ⇒ larger `ω_A = η_A/s` ([`omega_local_geometry`])
/// ⇒ shorter-range exchange. The free atom (`CN = 0`) gives `s = 1`, so the scheme reduces exactly to
/// [`omega_hardness_pairwise`]. No fitted constant (the only input is the existing GFN1 CN, whose
/// finite cutoff keeps the scheme size-consistent: a remote fragment leaves each local `s_A`
/// unchanged). Returns `(s, ds/dCN)`; the derivative drives the analytic `∂ω/∂R` exchange force.
#[inline]
pub fn local_size_factor_from_cn(cn: f64) -> (f64, f64) {
    let base = (1.0 + cn).max(1.0e-12);
    let s = base.powf(-1.0 / 3.0);
    let ds_dcn = (-1.0 / 3.0) * base.powf(-4.0 / 3.0);
    (s, ds_dcn)
}

#[cfg(test)]
mod charge_anharmonic_tests {
    use super::onsite_charge_anharmonic_derivatives;

    // The on-site charge derivative ladder (the L_ax/L_axx/L_xxx charge factors): each order is
    // the central FD of the previous — first→second→third→fourth — for stock GFN1 (order 3) and
    // the higher-order Breathing-Radius extension. The fourth order is what the directional
    // QUARTIC response stage's `D_λ(∂K/∂q)` chain contracts with; for order 3 it must be exactly
    // zero (`E = ⅓Γq³` ⇒ `E'''' ≡ 0`), which the last assertion pins.
    #[test]
    fn onsite_charge_derivative_ladder_matches_finite_difference() {
        let (gamma, gam3) = (0.5_f64, 0.12_f64);
        let h = 1.0e-6;
        for &order in &[3_usize, 4, 6] {
            for &q in &[-0.4_f64, -0.1, 0.25, 0.6] {
                let (_, second, third, fourth) =
                    onsite_charge_anharmonic_derivatives(gamma, gam3, order, q);
                let fd2 = (onsite_charge_anharmonic_derivatives(gamma, gam3, order, q + h).0
                    - onsite_charge_anharmonic_derivatives(gamma, gam3, order, q - h).0)
                    / (2.0 * h);
                let fd3 = (onsite_charge_anharmonic_derivatives(gamma, gam3, order, q + h).1
                    - onsite_charge_anharmonic_derivatives(gamma, gam3, order, q - h).1)
                    / (2.0 * h);
                let fd4 = (onsite_charge_anharmonic_derivatives(gamma, gam3, order, q + h).2
                    - onsite_charge_anharmonic_derivatives(gamma, gam3, order, q - h).2)
                    / (2.0 * h);
                assert!(
                    (second - fd2).abs() < 1.0e-5 * (1.0 + second.abs()),
                    "∂²E/∂q² order {order} q {q}: {second} vs FD {fd2}"
                );
                assert!(
                    (third - fd3).abs() < 1.0e-5 * (1.0 + third.abs()),
                    "∂³E/∂q³ order {order} q {q}: {third} vs FD {fd3}"
                );
                assert!(
                    (fourth - fd4).abs() < 1.0e-5 * (1.0 + fourth.abs()),
                    "∂⁴E/∂q⁴ order {order} q {q}: {fourth} vs FD {fd4}"
                );
                if order == 3 {
                    assert_eq!(
                        fourth, 0.0,
                        "stock GFN1 (charge_order 3) must have ∂⁴E/∂q⁴ ≡ 0"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod omega_tests {
    use super::*;

    /// `s_A = (1+CN)^(−1/3)` is 1 at CN=0 (⇒ LocalGeometry reduces to HardnessPairwise there),
    /// decreases monotonically, and its returned derivative matches a finite difference.
    #[test]
    fn local_size_factor_properties() {
        assert!((local_size_factor_from_cn(0.0).0 - 1.0).abs() < 1.0e-12);
        assert!(local_size_factor_from_cn(4.0).0 < local_size_factor_from_cn(1.0).0);
        let h = 1.0e-6;
        for &cn in &[0.5_f64, 2.0, 5.3] {
            let (_, d) = local_size_factor_from_cn(cn);
            let fd = (local_size_factor_from_cn(cn + h).0 - local_size_factor_from_cn(cn - h).0)
                / (2.0 * h);
            assert!((d - fd).abs() < 1.0e-7, "ds/dCN at {cn}: {d} vs {fd}");
        }
    }

    /// ω must be **non-negative** (the only physical bound; no upper cap). The harmonic mean is
    /// symmetric, recovers the homonuclear limit, and lies between the two atomic values.
    #[test]
    fn omega_is_nonnegative() {
        for &(ea, eb) in &[
            (0.30, 0.45),
            (0.47, 0.47),
            (0.20, 0.60),
            (2.0, 3.0),
            (1.5, 0.10),
            (0.0, 0.5),
        ] {
            let w = omega_hardness_pairwise(ea, eb);
            assert!(w >= 0.0, "ω_AB({ea},{eb}) = {w} is negative");
        }
        // homonuclear limit ω_AA = η_A (no spurious cap); symmetric.
        assert!((omega_hardness_pairwise(0.4, 0.4) - 0.4).abs() < 1.0e-12);
        assert!((omega_hardness_pairwise(2.0, 3.0) - 2.4).abs() < 1.0e-12); // no upper clamp
        assert!(
            (omega_hardness_pairwise(0.3, 0.5) - omega_hardness_pairwise(0.5, 0.3)).abs() < 1.0e-15
        );
        // lower bound only.
        assert_eq!(sanitize_omega(5.0), 5.0);
        assert_eq!(sanitize_omega(-2.0), 0.0);
    }

    /// The `OmegaScheme` dispatcher: `Fixed` returns the (sanitized) constant for any pair;
    /// `HardnessPairwise` reproduces `omega_hardness_pairwise`; the default is `HardnessPairwise`.
    #[test]
    fn omega_scheme_dispatch() {
        assert_eq!(OmegaScheme::default(), OmegaScheme::HardnessPairwise);
        // Fixed: constant regardless of η; negative sanitized to 0.
        assert_eq!(omega_pair(OmegaScheme::Fixed(0.3), 0.5, 0.9), 0.3);
        assert_eq!(omega_pair(OmegaScheme::Fixed(-1.0), 0.5, 0.9), 0.0);
        // HardnessPairwise matches the dedicated function.
        for &(ea, eb) in &[(0.30, 0.45), (0.47, 0.47), (0.5, 0.9)] {
            assert_eq!(
                omega_pair(OmegaScheme::HardnessPairwise, ea, eb),
                omega_hardness_pairwise(ea, eb)
            );
        }
    }

    /// LocalGeometry ω: reduces to HardnessPairwise at the isolated-atom reference (`s = 1`),
    /// monotonically decreases with the size factor (larger/softer atom ⇒ longer range ⇒ smaller ω),
    /// the pairwise combination is symmetric and `≥ 0`, and `omega_pair(LocalGeometry,…)` falls back
    /// to HardnessPairwise (no per-atom `s` available there).
    #[test]
    fn omega_local_geometry_properties() {
        // s = 1 ⇒ ω_A = η_A.
        assert!((omega_local_geometry(0.45, 1.0) - 0.45).abs() < 1.0e-15);
        // monotone decreasing in s.
        assert!(omega_local_geometry(0.45, 1.5) < omega_local_geometry(0.45, 1.0));
        assert!(omega_local_geometry(0.45, 0.8) > omega_local_geometry(0.45, 1.0));
        assert!(omega_local_geometry(0.45, 2.0) >= 0.0);
        // degenerate s ≤ 0 → falls back to η.
        assert_eq!(omega_local_geometry(0.45, 0.0), sanitize_omega(0.45));
        // pairwise reduces to HardnessPairwise at s = 1; symmetric.
        for &(ea, eb) in &[(0.30, 0.45), (0.5, 0.9)] {
            assert!(
                (omega_pair_local_geometry(ea, 1.0, eb, 1.0) - omega_hardness_pairwise(ea, eb))
                    .abs()
                    < 1.0e-15
            );
        }
        assert!(
            (omega_pair_local_geometry(0.3, 1.2, 0.5, 0.9)
                - omega_pair_local_geometry(0.5, 0.9, 0.3, 1.2))
            .abs()
                < 1.0e-15
        );
        // dispatcher fallback.
        assert_eq!(
            omega_pair(OmegaScheme::LocalGeometry, 0.5, 0.9),
            omega_hardness_pairwise(0.5, 0.9)
        );
    }
}

#[cfg(test)]
mod exchange_kernel_tests {
    use super::*;

    /// `γ^lr(R;0) = 0` and `lim_{ω→∞} γ^lr(R;ω) = γ^fr(R)`, for both R>0 and R=0 — the physical
    /// range-separation limits the naive `γ_KO·erf` form fails.
    #[test]
    fn lr_gamma_limits() {
        let sigma = exchange_sigma_pair(0.45, 0.30); // two distinct hardnesses
        for &r in &[0.0_f64, 0.5, 1.7, 4.0, 12.0] {
            // ω → 0: long-range vanishes.
            assert!(
                lr_gamma_exchange(r, sigma, 0.0).abs() < 1.0e-14,
                "γ^lr({r};0) should be 0"
            );
            assert!(
                lr_gamma_exchange(r, sigma, 1.0e-8).abs() < 1.0e-6,
                "γ^lr({r};0+) → 0"
            );
            // ω → ∞: long-range → full-range.
            let fr = fr_gamma_exchange(r, sigma);
            let lr_big = lr_gamma_exchange(r, sigma, 1.0e6);
            assert!(
                (fr - lr_big).abs() < 1.0e-6 * (1.0 + fr.abs()),
                "γ^lr({r};∞) = {lr_big:.6} should equal γ^fr = {fr:.6}"
            );
        }
    }

    /// All kernels are finite (and positive) at R=0 — the key property the screened integral has
    /// but `γ_KO·erf` lacks. On-site `γ^fr_AA(0)` reproduces the hardness η.
    #[test]
    fn kernels_finite_at_origin() {
        let eta = 0.42;
        let sigma_aa = exchange_sigma_pair(eta, eta);
        let fr0 = fr_gamma_exchange(0.0, sigma_aa);
        assert!(fr0.is_finite() && fr0 > 0.0);
        assert!(
            (fr0 - eta).abs() < 1.0e-10,
            "on-site γ^fr_AA(0) = {fr0:.6} should equal η = {eta}"
        );
        let lr0 = lr_gamma_exchange(0.0, sigma_aa, 0.4);
        assert!(lr0.is_finite() && lr0 > 0.0 && lr0 < fr0);
        // γ^fr = γ^sr + γ^lr partition holds everywhere.
        for &r in &[0.0_f64, 0.7, 2.3] {
            let split = sr_gamma_exchange(r, sigma_aa, 0.4) + lr_gamma_exchange(r, sigma_aa, 0.4);
            assert!((split - fr_gamma_exchange(r, sigma_aa)).abs() < 1.0e-13);
        }
    }

    /// Large-R both kernels approach the bare `1/R` (clouds look point-like).
    #[test]
    fn large_r_tends_to_coulomb() {
        let sigma = exchange_sigma_pair(0.5, 0.5);
        let r = 25.0;
        assert!((fr_gamma_exchange(r, sigma) - 1.0 / r).abs() < 1.0e-6);
        assert!((lr_gamma_exchange(r, sigma, 0.3) - 1.0 / r).abs() < 1.0e-6);
    }

    /// `lr_gamma_exchange_deriv` must match a central finite-difference of `lr_gamma_exchange`
    /// across a range of `R`, and vanish for `ω ≤ 0`.
    #[test]
    fn lr_gamma_deriv_matches_fd() {
        let sigma = exchange_sigma_pair(0.45, 0.55);
        let omega = omega_hardness_pairwise(0.45, 0.55);
        let h = 1.0e-6;
        for &r in &[0.5, 1.0, 2.0, 3.5, 6.0] {
            let fd = (lr_gamma_exchange(r + h, sigma, omega)
                - lr_gamma_exchange(r - h, sigma, omega))
                / (2.0 * h);
            let ana = lr_gamma_exchange_deriv(r, sigma, omega);
            assert!(
                (ana - fd).abs() < 1.0e-7,
                "dγ^lr/dR at R={r}: analytic {ana:.3e} vs FD {fd:.3e}"
            );
        }
        assert_eq!(lr_gamma_exchange_deriv(1.5, sigma, 0.0), 0.0);
    }
}
