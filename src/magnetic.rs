// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform **magnetic field** support: the **GFN1-xTB-M0 / GFN1-xTB-M1** method.
//!
//! A uniform magnetic field `B` enters a tight-binding model through the vector
//! potential `A(r) = 1/2 B x (r - O)` (`O` = gauge origin). Because `A` grows with
//! `r`, the Hamiltonian must be built from **London atomic orbitals** (LAOs / GIAOs)
//! `omega_nu = phi_nu exp(-i A_nu . r)`, `A_nu = 1/2 B x (R_nu - O)`, to keep the
//! energy gauge-origin independent. The closed-shell GFN1-xTB-M Hamiltonian (eq 12 /
//! 18-20 of Cheng & Wibowo-Teale) is the ordinary GFN1 `H0` evaluated over the LAO
//! overlap `S(B)` plus a field-dependent kinetic-energy correction
//! `(K^KE/2)(<omega_mu|pi^2|omega_nu> - e^{i f_munu} <phi_mu|p^2|phi_nu>)`,
//! `pi = p + A`, `f_munu = 1/2 (A_mu - A_nu).(R_mu + R_nu)` (London's midpoint phase,
//! eq 11). `K^KE = K^SZ = 1` for both M0 and M1, and for a closed shell the
//! spin-Zeeman term cancels, so the only new physics over field-free GFN1 is that
//! kinetic-energy correction. M0 evaluates it over the primary GFN1 minimal basis;
//! **M1** evaluates the `p^2`/`pi^2` integrals over a node-correct secondary
//! (dual) basis (eqs 21-23, see [`crate::secondary_basis`]), which the minimal
//! nodeless STO-NG AOs cannot describe. Everything reduces to field-free GFN1 as
//! `B -> 0`.
//!
//! What is implemented here: the exact complex LAO overlap [`lao_overlap_matrix`]
//! and kinetic integral [`lao_kinetic_matrix`] (`<omega|1/2 pi^2|omega>`), the
//! M0/M1 SCC ([`run_magnetic_scc`] / [`run_magnetic_scc_m1`]), the isotropic
//! magnetizability ([`magnetizability_isotropic`], eq 26) and the magnetic nuclear
//! gradient ([`magnetic_gradient`]). Non-periodic, closed shell.
//!
//! # Conventions and scope
//!
//! - **Magnetizability sign.** `xi_ab = -d^2 E / dB_a dB_b`, equivalently
//!   `E(B) = E(0) - 1/2 xi_ab B_a B_b`. A diamagnetic closed-shell molecule is
//!   pushed *out* of the field, so `d^2E/dB^2 > 0` and its isotropic
//!   magnetizability is **negative** (water lands near `-170 x 10^-30 J T^-2`
//!   here, methane near `-250`). Every magnetizability entry point in this module
//!   — [`magnetizability_isotropic`], [`magnetizability_diagonal_analytic`],
//!   [`magnetizability_tensor_analytic`] — uses that one sign.
//! - **Gauge origin.** `options.external_field.origin` does **not** enter `S(B)`
//!   or `H0(B)` at all: London orbitals make them depend on the field only through
//!   the difference `A_mu - A_nu = 1/2 B x (R_mu - R_nu)`, and the individually
//!   origin-dependent pieces of the `pi^2` integral cancel exactly. The energy is
//!   therefore gauge-origin invariant *structurally*, not approximately, and is
//!   invariant for charged systems too. The origin is still used by the electric
//!   field, the Mulliken dipole and the (common-gauge-origin, hence genuinely
//!   origin-dependent) NMR shielding [`nmr_shielding_tensor`].
//! - **Translation / rotation.** Rigidly translating the molecule multiplies
//!   `H0(B)` and `S(B)` by the same diagonal unitary `exp(i A_mu.d)`, so the energy
//!   is translation invariant; rotating the molecule and `B` together is likewise
//!   exact. Both are gated in `tests/magnetic.rs`.
//! - **Occupations.** The magnetic SCC is strictly zero-temperature: it fills the
//!   lowest `2 * nocc` states with integer occupations and **ignores**
//!   `ElectronicOptions::electronic_temperature`. It also ignores the anisotropic
//!   multipole (AES) options — only the isotropic shell-charge Coulomb model is
//!   used, matching the field-free path at its defaults.
//!
//! # References
//!
//! - **GFN1-xTB-M0/M1 method:** C. Y. Cheng and A. M. Wibowo-Teale, "Semiempirical
//!   Methods for Molecular Systems in Strong Magnetic Fields", *J. Chem. Theory
//!   Comput.* **19**, 6226-6241 (2023). DOI: 10.1021/acs.jctc.3c00671. (Eq 8 LAOs;
//!   eq 11 midpoint phase; eq 12/18-20 Hamiltonian; eqs 21-23 dual basis; eq 26
//!   magnetizability `xi_ab = -d^2 E / dB_a dB_b` by finite field.)
//! - **LAO integral evaluation:** T. J. P. Irons, J. Zemen, A. M. Teale, "Efficient
//!   Calculation of Molecular Integrals over London Atomic Orbitals", *J. Chem.
//!   Theory Comput.* **13**, 3636-3649 (2017). DOI: 10.1021/acs.jctc.7b00540.
//!   (Complex Gaussian product theorem: complex product centre `Pbar = P - (i/2 zeta)
//!   chi`, prefactor `K_P = exp(-chi.chi/4 zeta - i P.chi)`; multipole/differential
//!   recursions for `<omega|pi^2|omega>` used in [`lao_kinetic_pair`].)
//! - **LAO integral derivatives (analytic gradient):** T. J. P. Irons, A. David,
//!   A. M. Teale, "Optimizing Molecular Geometries in Strong Magnetic Fields",
//!   *J. Chem. Theory Comput.* **17**, 2166-2185 (2021). DOI: 10.1021/acs.jctc.0c01297.
//! - **London atomic orbitals:** F. London, *J. Phys. Radium* **8**, 397 (1937).
//! - **Parent GFN1-xTB:** S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory
//!   Comput.* **13**, 1989-2009 (2017). DOI: 10.1021/acs.jctc.7b00118.
//! - **Secondary-basis source (cc-pVDZ):** T. H. Dunning Jr., *J. Chem. Phys.*
//!   **90**, 1007-1023 (1989).

use crate::basis::{AOBasisFunction, BasisOptions, BasisSet};
use crate::coulomb::{
    coulomb_energy_potential_from_matrix, effective_coulomb_matrix, ShellChargeModel,
};
use crate::dispersion::dispersion_energy;
use crate::electronic::{BroydenMixer, ElectronicOptions};
use crate::error::{Gfn1Error, Result};
use crate::field::{electric_field_energy, electric_shell_potential, ExternalFieldOptions};
use crate::halogen::halogen_energy;
use crate::hamiltonian::build_h0;
use crate::linalg::{lowdin_solve_generalized, Matrix};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::pbc::complex::{hermitian_generalized_eigen, weighted_density, CMatrix};
use crate::repulsion::repulsion_energy;
use crate::secondary_basis::SecondaryBasis;
use crate::sto::PrimitiveGaussian;
use crate::system::PeriodicSystem;
use rayon::prelude::*;
use std::collections::HashMap;

/// London (GIAO) phase angle for an orbital pair centred at `r_a` (bra) and
/// `r_b` (ket) in a uniform field `b` with gauge origin `origin`:
///
/// ```text
/// theta_ab = 1/2 * B · [ (r_a - origin) x (r_b - origin) ].
/// ```
///
/// The magnetically-dressed overlap/Hamiltonian block picks up the phase factor
/// `exp(i * theta_ab)`. The angle is antisymmetric (`theta_ab = -theta_ba`),
/// vanishes for `B = 0` and for collinear centres, and is the geometric core of
/// the Peierls substitution.
///
/// **Caveat.** Unlike the exact London overlap this angle *does* depend on
/// `origin`: `theta_ab` is only the leading (s-type, equal-exponent, midpoint)
/// phase of `<omega_a|omega_b>`, and it reproduces it only for `origin = 0`. The
/// true LAO overlap [`lao_overlap_matrix`] is gauge-origin free. Use this for
/// Peierls-style model work, never as a stand-in for the real integrals — see
/// [`london_dress_ao_matrix`].
pub fn london_phase_angle(b: Vec3, origin: Vec3, r_a: Vec3, r_b: Vec3) -> f64 {
    let ra = r_a - origin;
    let rb = r_b - origin;
    0.5 * b.dot(ra.cross(rb))
}

/// `exp(i * theta_ab)` as a `(real, imag)` pair, ready to multiply a real
/// `H0`/`S` block when the complex magnetic Hamiltonian is assembled.
pub fn london_phase_factor(b: Vec3, origin: Vec3, r_a: Vec3, r_b: Vec3) -> (f64, f64) {
    let theta = london_phase_angle(b, origin, r_a, r_b);
    (theta.cos(), theta.sin())
}

/// Exact nuclear-coordinate gradient of the London phase
/// `theta_{ab} = 1/2 B . [(R_a - O) x (R_b - O)]` with respect to the bra-centre
/// `R_a` and the ket-centre `R_b`:
///
/// ```text
/// d theta_ab / d R_a = 1/2 (R_b - O) x B,
/// d theta_ab / d R_b = 1/2 B x (R_a - O).
/// ```
///
/// (Both follow from the cyclic invariance of the scalar triple product.) This is
/// parameter-free and exact; it is the geometric building block of the analytic
/// magnetic gradient — `dH0(B)_{munu}/dR = (dH0_{munu}/dR) e^{i theta} +
/// H0_{munu} (i d theta_{munu}/dR) e^{i theta}`, and likewise for `S(B)` — to be
/// contracted with the complex density / energy-weighted (Pulay) density once the
/// complex AO-derivative assembly is wired in.
pub fn london_phase_gradient(b: Vec3, origin: Vec3, r_a: Vec3, r_b: Vec3) -> (Vec3, Vec3) {
    let ra = r_a - origin;
    let rb = r_b - origin;
    let d_r_a = rb.cross(b) * 0.5;
    let d_r_b = b.cross(ra) * 0.5;
    (d_r_a, d_r_b)
}

/// Uniform-field orbital angular-momentum prefactor `1/2 (B x (R - origin))`,
/// the velocity-gauge coupling that a future orbital-Zeeman term would contract
/// with the momentum integrals. Provided as a building block; not yet used in the
/// Hamiltonian.
pub fn orbital_zeeman_prefactor(b: Vec3, origin: Vec3, position: Vec3) -> Vec3 {
    b.cross(position - origin) * 0.5
}

/// Marker error for magnetic-field features that are scaffolded but not yet
/// physically implemented.
pub fn magnetic_not_implemented(feature: &str) -> Gfn1Error {
    Gfn1Error::InvalidInput(format!(
        "external magnetic field: {feature} is a foothold only and not yet implemented \
         (see the `magnetic` module roadmap)"
    ))
}

/// A complex AO-basis matrix, `re + i*im`, used by the magnetic (London) builders.
#[derive(Clone, Debug)]
pub struct ComplexAoMatrix {
    pub re: Matrix,
    pub im: Matrix,
}

/// London (GIAO) dressing of a real AO matrix `M` (e.g. `H0` or `S`):
///
/// ```text
/// M(B)_{mu nu} = M_{mu nu} * exp(i * theta_{mu nu}),
/// theta_{mu nu} = 1/2 B . [(R_mu - O) x (R_nu - O)].
/// ```
///
/// This is a Peierls-style *model* dressing of the zero-field Hamiltonian /
/// overlap, i.e. the leading s-type phase of the London overlap only. It is **not**
/// what the GFN1-xTB-M SCC uses and it is **not** interchangeable with it:
///
/// - it omits the Gaussian damping `exp(-chi.chi/4 zeta)` and every angular /
///   derivative correction of the exact complex Gaussian product theorem, and
/// - it is **gauge-origin dependent** (`theta_{mu nu}` moves with `options.origin`,
///   see [`london_phase_angle`]), whereas the exact LAO overlap
///   [`lao_overlap_matrix`] does not depend on the origin at all.
///
/// The production path is [`lao_overlap_matrix`] plus the `pi^2` kinetic-energy
/// correction; this function is kept as the Peierls-substitution reference point
/// and as the negative control in
/// `tests/magnetic.rs::lao_overlap_is_gauge_origin_free_but_phase_dressing_is_not`.
/// Non-periodic only. Cheng & Wibowo-Teale, *J. Chem. Theory Comput.* **19**, 6226
/// (2023), eq 12.
pub fn london_dress_ao_matrix(
    real: &Matrix,
    system: &PeriodicSystem,
    basis: &BasisSet,
    options: &ExternalFieldOptions,
) -> Result<ComplexAoMatrix> {
    let field = options.magnetic_field.unwrap_or(Vec3::zero());
    let n = basis.len();
    if real.rows() != n || real.cols() != n {
        return Err(Gfn1Error::InvalidInput(
            "London dressing requires an AO-sized (n x n) matrix".to_string(),
        ));
    }
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    for mu in 0..n {
        let ra = system.atoms[basis.aos[mu].atom_index].position;
        for nu in 0..n {
            let rb = system.atoms[basis.aos[nu].atom_index].position;
            let theta = london_phase_angle(field, options.origin, ra, rb);
            let value = real[(mu, nu)];
            re[(mu, nu)] = value * theta.cos();
            im[(mu, nu)] = value * theta.sin();
        }
    }
    Ok(ComplexAoMatrix { re, im })
}

// Tiny complex-scalar helpers (re, im) for the London integral recursions.
#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
#[inline]
fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn cscale(a: (f64, f64), s: f64) -> (f64, f64) {
    (a.0 * s, a.1 * s)
}

/// Complex 1D Obara-Saika overlap `[la|lb]` over London atomic orbitals: the
/// product centre is complex, `Pbar_x = P_x - i chi_x/(2 zeta)`, so the bra/ket
/// displacements `X_PA = Pbar_x - a`, `X_PB = Pbar_x - b` are complex; the
/// recursion is otherwise the standard overlap OS recursion. `base` is the per-axis
/// base `[0|0]` (e.g. `(pi/zeta)^{1/2}`).
fn lao_overlap_1d(
    la: usize,
    lb: usize,
    p_axis: f64,
    chi_axis: f64,
    a: f64,
    b: f64,
    inv2zeta: f64,
    base: (f64, f64),
) -> (f64, f64) {
    let xpa = (p_axis - a, -chi_axis * inv2zeta);
    let xpb = (p_axis - b, -chi_axis * inv2zeta);
    let w = lb + 1;
    let mut s = vec![(0.0_f64, 0.0_f64); (la + 1) * w];
    s[0] = base;
    for i in 1..=la {
        let mut v = cmul(xpa, s[(i - 1) * w]);
        if i >= 2 {
            v = cadd(v, cscale(s[(i - 2) * w], (i - 1) as f64 * inv2zeta));
        }
        s[i * w] = v;
    }
    for j in 1..=lb {
        for i in 0..=la {
            let mut v = cmul(xpb, s[i * w + (j - 1)]);
            if i >= 1 {
                v = cadd(v, cscale(s[(i - 1) * w + (j - 1)], i as f64 * inv2zeta));
            }
            if j >= 2 {
                v = cadd(v, cscale(s[i * w + (j - 2)], (j - 1) as f64 * inv2zeta));
            }
            s[i * w + j] = v;
        }
    }
    s[la * w + lb]
}

/// Exact London (GIAO) overlap `S(B)_{mu nu} = <omega_mu|omega_nu>` of one AO pair
/// via the complex Gaussian product theorem (Irons, Zemen & Teale, *J. Chem. Theory
/// Comput.* **13**, 3636 (2017)).
fn lao_overlap_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    field: Vec3,
) -> (f64, f64) {
    // chi = 1/2 B x (B - A): geometry/field only, independent of the gauge origin.
    complex_boost_overlap_pair(a, b, ca, cb, field.cross(cb - ca) * 0.5)
}

/// Complex-Gaussian-product overlap for a *general* complex boost vector `chi`:
///
/// ```text
///   <a| exp(-i chi . r) |b>,   Pbar = P - (i/2 zeta) chi,
///   K_P = exp(-chi.chi/(4 zeta) - i P.chi).
/// ```
///
/// This is the one kernel behind both consumers of the complex Gaussian product
/// theorem: the London (GIAO) overlap sets `chi = 1/2 B x (R_b - R_a)` (see
/// [`lao_overlap_matrix`]), the periodic Berry-phase boost sets `chi = -q` (see
/// [`boosted_overlap_pair`]).
fn complex_boost_overlap_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    chi: Vec3,
) -> (f64, f64) {
    let r2 = (ca - cb).norm2();
    let chi2 = chi.dot(chi);
    let mut acc = (0.0_f64, 0.0_f64);
    for pa in &a.primitives {
        for pb in &b.primitives {
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let zeta = alpha + beta;
            let inv2zeta = 0.5 / zeta;
            let pref = pa.coefficient * pb.coefficient;
            let kab = (-alpha * beta / zeta * r2).exp();
            let p = (ca * alpha + cb * beta) * (1.0 / zeta);
            // K_P = exp(-chi.chi/(4 zeta) - i P.chi).
            let kp_mag = (-chi2 / (4.0 * zeta)).exp();
            let angle = -p.dot(chi);
            let kp = (kp_mag * angle.cos(), kp_mag * angle.sin());
            let base = ((std::f64::consts::PI / zeta).sqrt(), 0.0);
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let coeff = pref * ca_term.coefficient * cb_term.coefficient * kab;
                    let ap = ca_term.power;
                    let bp = cb_term.power;
                    let sx = lao_overlap_1d(ap.x, bp.x, p.x, chi.x, ca.x, cb.x, inv2zeta, base);
                    let sy = lao_overlap_1d(ap.y, bp.y, p.y, chi.y, ca.y, cb.y, inv2zeta, base);
                    let sz = lao_overlap_1d(ap.z, bp.z, p.z, chi.z, ca.z, cb.z, inv2zeta, base);
                    let prod = cmul(cmul(sx, sy), cmul(sz, kp));
                    acc.0 += coeff * prod.0;
                    acc.1 += coeff * prod.1;
                }
            }
        }
    }
    acc
}

/// **Boosted AO overlap** `<chi_a| e^{i q . r} |chi_b>` for one contracted AO pair
/// centred at `ca` (bra) and `cb` (ket), returned as `(re, im)`.
///
/// This is the plane-wave-boosted (momentum-shifted) overlap the periodic
/// Berry-phase polarization needs — see
/// [`crate::pbc::polarization::pbc_berry_polarization`]. It is *exactly* the same
/// complex Gaussian product theorem the London/GIAO overlap already uses: with
/// `exp(-zeta |r - P|^2 + i q . r) = exp(-zeta |r - Pbar|^2) exp(i P.q - q.q/(4 zeta))`
/// and `Pbar = P + (i/2 zeta) q`, the LAO kernel [`complex_boost_overlap_pair`]
/// reproduces it verbatim at `chi = -q`. No new integral code: only the complex
/// boost vector differs (LAO: `chi = 1/2 B x (R_b - R_a)`; Berry: `chi = -q`).
///
/// `q` is a **Cartesian** wave vector in `bohr^-1`, and the operator uses the
/// absolute `r`, so the result depends on the choice of coordinate origin exactly
/// as `e^{i q . r}` does. Reduces to the real AO overlap at `q = 0`.
pub fn boosted_overlap_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    q: Vec3,
) -> (f64, f64) {
    complex_boost_overlap_pair(a, b, ca, cb, -q)
}

/// Exact London (GIAO) AO overlap matrix `S(B)_{mu nu} = <omega_mu|omega_nu>` via
/// the complex Gaussian product theorem: a complex product centre
/// `Pbar = P - (i/2 zeta) chi`, `chi = 1/2 B x (R_nu - R_mu)`, and complex prefactor
/// `Utilde = U_P exp(-chi.chi/(4 zeta) - i P.chi)` fed through the standard
/// Obara-Saika overlap recursion in complex arithmetic. Reduces to the real AO
/// overlap at `B = 0` and is Hermitian. Non-periodic. This is the exact LAO overlap
/// behind the GFN1-xTB-M band term (replacing the simple phase dressing of
/// [`london_dress_ao_matrix`]).
pub fn lao_overlap_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    options: &ExternalFieldOptions,
) -> ComplexAoMatrix {
    let field = options.magnetic_field.unwrap_or(Vec3::zero());
    let n = basis.len();
    let ao_min_exp: Vec<f64> = basis
        .aos
        .iter()
        .map(|ao| {
            ao.primitives
                .iter()
                .map(|p| p.exponent)
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let centers: Vec<Vec3> = basis
        .aos
        .iter()
        .map(|ao| system.atoms[ao.atom_index].position)
        .collect();
    // Each row independent; parallelize over `mu` with Gaussian distance screening
    // (overlap < ~e^-40 dropped).
    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|mu| {
            let a_ao = &basis.aos[mu];
            let ca = centers[mu];
            let ea = ao_min_exp[mu];
            let mut rre = vec![0.0_f64; n];
            let mut rim = vec![0.0_f64; n];
            for nu in 0..n {
                let cb = centers[nu];
                let r2 = (ca - cb).norm2();
                let eb = ao_min_exp[nu];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue;
                }
                let (sr, si) = lao_overlap_pair(a_ao, &basis.aos[nu], ca, cb, field);
                rre[nu] = sr;
                rim[nu] = si;
            }
            (rre, rim)
        })
        .collect();
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    for (mu, (rre, rim)) in rows.into_iter().enumerate() {
        for nu in 0..n {
            re[(mu, nu)] = rre[nu];
            im[(mu, nu)] = rim[nu];
        }
    }
    ComplexAoMatrix { re, im }
}

/// Per-axis 1D London integral blocks `(S, D1, D2, M1, M2)` for one Cartesian
/// direction: the overlap `[a|b]`, the first/second differential integrals
/// `[a|d^n|b]` (eq 21, carrying the ket London phase `k_b`), and the first/second
/// multipole integrals `[a|x_o^n|b]` (eq 20, `x_o = x - O`). All complex, built from
/// the complex-centre 1D overlap `lao_overlap_1d`. `bo = B_axis - O_axis`,
/// `kb_axis` = ket London wave-vector component.
#[allow(clippy::too_many_arguments)]
fn lao_axis_blocks(
    la: usize,
    lb: usize,
    p_axis: f64,
    chi_axis: f64,
    a: f64,
    b: f64,
    inv2zeta: f64,
    beta: f64,
    kb_axis: f64,
    bo: f64,
    base: (f64, f64),
) -> ((f64, f64), (f64, f64), (f64, f64), (f64, f64), (f64, f64)) {
    let s_of = |lbp: i64| -> (f64, f64) {
        if lbp < 0 {
            (0.0, 0.0)
        } else {
            lao_overlap_1d(la, lbp as usize, p_axis, chi_axis, a, b, inv2zeta, base)
        }
    };
    let ikb = (0.0_f64, -kb_axis);
    // [a|d_x^1|(ket power lbp)> = lbp S(lbp-1) - i k_b S(lbp) - 2 beta S(lbp+1).
    let d1_of = |lbp: i64| -> (f64, f64) {
        let c = if lbp > 0 { lbp as f64 } else { 0.0 };
        cadd(
            cadd(cscale(s_of(lbp - 1), c), cmul(ikb, s_of(lbp))),
            cscale(s_of(lbp + 1), -2.0 * beta),
        )
    };
    let lbi = lb as i64;
    let s0 = s_of(lbi);
    let sp1 = s_of(lbi + 1);
    let sp2 = s_of(lbi + 2);
    let m1 = cadd(cscale(s0, bo), sp1);
    let m2 = cadd(cscale(m1, bo), cadd(cscale(sp1, bo), sp2));
    let d1 = d1_of(lbi);
    let d2 = cadd(
        cadd(cscale(d1_of(lbi - 1), lb as f64), cmul(ikb, d1)),
        cscale(d1_of(lbi + 1), -2.0 * beta),
    );
    (s0, d1, d2, m1, m2)
}

/// London (GIAO) kinetic-energy integral `<omega_a|1/2 pi^2|omega_b>`
/// (`pi = p + A`, `A = 1/2 B x (r - O)`) for one AO pair, via the complex Gaussian
/// product theorem and the operator decomposition (Irons-Zemen-Teale eqs 14-22):
/// `1/2 pi^2_x = -1/2 d_x^2 - (i/2) B_y d_x z_o + (i/2) B_z d_x y_o + 1/8 B_y^2 z_o^2
/// + 1/8 B_z^2 y_o^2 - 1/4 B_y B_z y_o z_o` (and cyclic). Reduces to the real
/// `<a|-1/2 nabla^2|b>` at `B = 0`. Returns `(re, im)`.
fn lao_kinetic_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    field: Vec3,
    origin: Vec3,
) -> (f64, f64) {
    let r2 = (ca - cb).norm2();
    let chi = field.cross(cb - ca) * 0.5;
    let chi2 = chi.dot(chi);
    let kb = field.cross(cb - origin) * 0.5; // ket London wave vector
    let bo = cb - origin; // ket centre relative to the gauge origin
    let (bx, by, bz) = (field.x, field.y, field.z);
    let mut acc = (0.0_f64, 0.0_f64);
    for pa in &a.primitives {
        for pb in &b.primitives {
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let zeta = alpha + beta;
            let inv2zeta = 0.5 / zeta;
            let pref = pa.coefficient * pb.coefficient;
            let kab = (-alpha * beta / zeta * r2).exp();
            let p = (ca * alpha + cb * beta) * (1.0 / zeta);
            let kp_mag = (-chi2 / (4.0 * zeta)).exp();
            let angle = -p.dot(chi);
            let kp = (kp_mag * angle.cos(), kp_mag * angle.sin());
            let base = ((std::f64::consts::PI / zeta).sqrt(), 0.0);
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let coeff = pref * ca_term.coefficient * cb_term.coefficient * kab;
                    let ap = ca_term.power;
                    let bp = cb_term.power;
                    let (sx, d1x, d2x, m1x, m2x) = lao_axis_blocks(
                        ap.x, bp.x, p.x, chi.x, ca.x, cb.x, inv2zeta, beta, kb.x, bo.x, base,
                    );
                    let (sy, d1y, d2y, m1y, m2y) = lao_axis_blocks(
                        ap.y, bp.y, p.y, chi.y, ca.y, cb.y, inv2zeta, beta, kb.y, bo.y, base,
                    );
                    let (sz, d1z, d2z, m1z, m2z) = lao_axis_blocks(
                        ap.z, bp.z, p.z, chi.z, ca.z, cb.z, inv2zeta, beta, kb.z, bo.z, base,
                    );
                    let mut t = (0.0_f64, 0.0_f64);
                    let mut term =
                        |c: (f64, f64), p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)| {
                            let prod = cmul(cmul(p1, p2), cmul(p3, c));
                            t.0 += prod.0;
                            t.1 += prod.1;
                        };
                    // 1/2 pi^2_x (differential on x; multipoles z_o, y_o):
                    term((-0.5, 0.0), d2x, sy, sz);
                    term((0.0, -0.5 * by), d1x, sy, m1z);
                    term((0.0, 0.5 * bz), d1x, m1y, sz);
                    term((0.125 * by * by, 0.0), sx, sy, m2z);
                    term((0.125 * bz * bz, 0.0), sx, m2y, sz);
                    term((-0.25 * by * bz, 0.0), sx, m1y, m1z);
                    // 1/2 pi^2_y (differential on y; multipoles x_o, z_o):
                    term((-0.5, 0.0), sx, d2y, sz);
                    term((0.0, -0.5 * bz), m1x, d1y, sz);
                    term((0.0, 0.5 * bx), sx, d1y, m1z);
                    term((0.125 * bz * bz, 0.0), m2x, sy, sz);
                    term((0.125 * bx * bx, 0.0), sx, sy, m2z);
                    term((-0.25 * bz * bx, 0.0), m1x, sy, m1z);
                    // 1/2 pi^2_z (differential on z; multipoles y_o, x_o):
                    term((-0.5, 0.0), sx, sy, d2z);
                    term((0.0, -0.5 * bx), sx, m1y, d1z);
                    term((0.0, 0.5 * by), m1x, sy, d1z);
                    term((0.125 * bx * bx, 0.0), sx, m2y, sz);
                    term((0.125 * by * by, 0.0), m2x, sy, sz);
                    term((-0.25 * bx * by, 0.0), m1x, m1y, sz);
                    let tk = cmul(t, kp);
                    acc.0 += coeff * tk.0;
                    acc.1 += coeff * tk.1;
                }
            }
        }
    }
    acc
}

/// London (GIAO) kinetic-energy matrix `<omega_mu|1/2 pi^2|omega_nu>` (complex,
/// non-periodic). At `B = 0` this is the real `<mu|-1/2 nabla^2|nu>`
/// ([`crate::integrals::kinetic_energy_matrix`]); the difference from the
/// phase-dressed zero-field kinetic is the GFN1-xTB-M kinetic-energy correction.
pub fn lao_kinetic_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    options: &ExternalFieldOptions,
) -> ComplexAoMatrix {
    let field = options.magnetic_field.unwrap_or(Vec3::zero());
    let origin = options.origin;
    let n = basis.len();
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    for mu in 0..n {
        let a_ao = &basis.aos[mu];
        let ca = system.atoms[a_ao.atom_index].position;
        for nu in 0..n {
            let b_ao = &basis.aos[nu];
            let cb = system.atoms[b_ao.atom_index].position;
            let (kr, ki) = lao_kinetic_pair(a_ao, b_ao, ca, cb, field, origin);
            re[(mu, nu)] = kr;
            im[(mu, nu)] = ki;
        }
    }
    ComplexAoMatrix { re, im }
}

/// Spin-Zeeman one-electron contributions `<mu|H_SZ|nu> = +/- 1/2 |B| S_{mu nu}`
/// for the alpha (`+`) and beta (`-`) spin blocks (Cheng & Wibowo-Teale eq 13).
/// Returns `(alpha_block, beta_block)`. These are the open-shell spin terms a full
/// magnetic UHF would add to the per-spin effective Hamiltonian; for a closed shell
/// they cancel, which is why the SCC here never calls this.
///
/// Convention: `H_SZ = B . s` in atomic units (`g_e/2 ~ 1`) with the spin quantised
/// **along `B`**, so the coupling is the field *magnitude* `|B|` and alpha (spin
/// parallel to `B`) is raised. Consequently the blocks do not change sign under
/// `B -> -B` — the quantisation axis flips with the field.
pub fn spin_zeeman_blocks(overlap: &Matrix, options: &ExternalFieldOptions) -> (Matrix, Matrix) {
    let b = options.magnetic_field.unwrap_or(Vec3::zero()).norm();
    let half = 0.5 * b;
    let n = overlap.rows();
    let mut alpha = Matrix::zeros(n, n);
    let mut beta = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let s = overlap[(i, j)] * half;
            alpha[(i, j)] = s;
            beta[(i, j)] = -s;
        }
    }
    (alpha, beta)
}

/// Orbital angular-momentum AO integrals `L_a` about gauge origin `origin`,
/// `L = (r - O) x p`, `p = -i grad`. The physical matrix element is
/// `<mu|L_a|nu> = -i * out[a][(mu,nu)]`, so the returned real matrices `out[a]`
/// (a = x, y, z) are **antisymmetric with zero diagonal** (because `L` is Hermitian
/// and the `-i` makes `<mu|L|nu>` purely imaginary). The orbital magnetic-dipole
/// operator is `m = -1/2 L` (atomic units, electron charge `-1`); rotatory strengths
/// and the optical-rotation tensor use the magnetic transition/response dipole built
/// from these. Non-periodic, field-free integrals (no London phase).
///
/// `L_z = (x-O_x) p_y - (y-O_y) p_x`, etc.; in real Cartesian-Gaussian 1D blocks
/// `<mu|L_z|nu> = -i * sum (S_z [M1_x D1_y - M1_y D1_x])`, with the 1D overlap `S`,
/// multipole `M1 = <a|(x-O)|b>` and differential `D1 = <a|d/dx|b>` per axis.
pub fn angular_momentum_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    origin: Vec3,
) -> [Matrix; 3] {
    let n = basis.len();
    let mut lx = Matrix::zeros(n, n);
    let mut ly = Matrix::zeros(n, n);
    let mut lz = Matrix::zeros(n, n);
    for mu in 0..n {
        let a_ao = &basis.aos[mu];
        let ca = system.atoms[a_ao.atom_index].position;
        for nu in 0..n {
            let b_ao = &basis.aos[nu];
            let cb = system.atoms[b_ao.atom_index].position;
            let r2 = (ca - cb).norm2();
            let (mut cx, mut cy, mut cz) = (0.0_f64, 0.0_f64, 0.0_f64);
            for pa in &a_ao.primitives {
                for pb in &b_ao.primitives {
                    let alpha = pa.exponent;
                    let beta = pb.exponent;
                    let zeta = alpha + beta;
                    let inv2zeta = 0.5 / zeta;
                    let kab = (-alpha * beta / zeta * r2).exp();
                    let p = (ca * alpha + cb * beta) * (1.0 / zeta);
                    let base = ((std::f64::consts::PI / zeta).sqrt(), 0.0);
                    // Real 1D overlap [la|lb] on a given axis (chi = 0 -> real).
                    let s1d = |la: usize, lb: i64, pax: f64, a: f64, b: f64| -> f64 {
                        if lb < 0 {
                            0.0
                        } else {
                            lao_overlap_1d(la, lb as usize, pax, 0.0, a, b, inv2zeta, base).0
                        }
                    };
                    // Per-axis S, D1 = <a|d/dx|b>, M1 = <a|(x-O)|b>.
                    let blocks = |la: usize,
                                  lb: usize,
                                  pax: f64,
                                  a: f64,
                                  b: f64,
                                  o: f64|
                     -> (f64, f64, f64) {
                        let lbi = lb as i64;
                        let s0 = s1d(la, lbi, pax, a, b);
                        let d1 = lb as f64 * s1d(la, lbi - 1, pax, a, b)
                            - 2.0 * beta * s1d(la, lbi + 1, pax, a, b);
                        let m1 = (b - o) * s0 + s1d(la, lbi + 1, pax, a, b);
                        (s0, d1, m1)
                    };
                    for ca_term in &a_ao.components {
                        for cb_term in &b_ao.components {
                            let coeff = pa.coefficient
                                * pb.coefficient
                                * ca_term.coefficient
                                * cb_term.coefficient
                                * kab;
                            let ap = ca_term.power;
                            let bp = cb_term.power;
                            let (sx, dx, mx) = blocks(ap.x, bp.x, p.x, ca.x, cb.x, origin.x);
                            let (sy, dy, my) = blocks(ap.y, bp.y, p.y, ca.y, cb.y, origin.y);
                            let (sz, dz, mz) = blocks(ap.z, bp.z, p.z, ca.z, cb.z, origin.z);
                            // L_x = (y-O)d_z - (z-O)d_y ; L_y = (z-O)d_x - (x-O)d_z ;
                            // L_z = (x-O)d_y - (y-O)d_x  (real coefficient of -i).
                            cx += coeff * sx * (my * dz - mz * dy);
                            cy += coeff * sy * (mz * dx - mx * dz);
                            cz += coeff * sz * (mx * dy - my * dx);
                        }
                    }
                }
            }
            lx[(mu, nu)] = cx;
            ly[(mu, nu)] = cy;
            lz[(mu, nu)] = cz;
        }
    }
    [lx, ly, lz]
}

/// London (GIAO) electric-dipole integral `<omega_a|(r_c - O)|omega_b>` per Cartesian
/// component `c` (`O` = the gauge origin) for one AO pair, built from the complex 1D
/// London blocks: the operator axis carries the multipole `M1 = <a|(x-O)|b>`, the
/// other two the overlap `S`, all dressed by the London prefactor `K_P`. Returns
/// `[(re, im); 3]`. Reduces to the real dipole integral `<a|(r_c - O)|b>` at `B = 0`.
fn lao_dipole_pair(
    a: &AOBasisFunction,
    b: &AOBasisFunction,
    ca: Vec3,
    cb: Vec3,
    field: Vec3,
    origin: Vec3,
) -> [(f64, f64); 3] {
    let r2 = (ca - cb).norm2();
    let chi = field.cross(cb - ca) * 0.5;
    let chi2 = chi.dot(chi);
    let kb = field.cross(cb - origin) * 0.5; // ket London wave vector
    let bo = cb - origin; // ket centre relative to the gauge/dipole origin
    let mut acc = [(0.0_f64, 0.0_f64); 3];
    for pa in &a.primitives {
        for pb in &b.primitives {
            let alpha = pa.exponent;
            let beta = pb.exponent;
            let zeta = alpha + beta;
            let inv2zeta = 0.5 / zeta;
            let pref = pa.coefficient * pb.coefficient;
            let kab = (-alpha * beta / zeta * r2).exp();
            let p = (ca * alpha + cb * beta) * (1.0 / zeta);
            let kp_mag = (-chi2 / (4.0 * zeta)).exp();
            let angle = -p.dot(chi);
            let kp = (kp_mag * angle.cos(), kp_mag * angle.sin());
            let base = ((std::f64::consts::PI / zeta).sqrt(), 0.0);
            for ca_term in &a.components {
                for cb_term in &b.components {
                    let coeff = pref * ca_term.coefficient * cb_term.coefficient * kab;
                    let ap = ca_term.power;
                    let bp = cb_term.power;
                    let (sx, _, _, m1x, _) = lao_axis_blocks(
                        ap.x, bp.x, p.x, chi.x, ca.x, cb.x, inv2zeta, beta, kb.x, bo.x, base,
                    );
                    let (sy, _, _, m1y, _) = lao_axis_blocks(
                        ap.y, bp.y, p.y, chi.y, ca.y, cb.y, inv2zeta, beta, kb.y, bo.y, base,
                    );
                    let (sz, _, _, m1z, _) = lao_axis_blocks(
                        ap.z, bp.z, p.z, chi.z, ca.z, cb.z, inv2zeta, beta, kb.z, bo.z, base,
                    );
                    let dx = cmul(cmul(m1x, sy), cmul(sz, kp));
                    let dy = cmul(cmul(sx, m1y), cmul(sz, kp));
                    let dz = cmul(cmul(sx, sy), cmul(m1z, kp));
                    acc[0].0 += coeff * dx.0;
                    acc[0].1 += coeff * dx.1;
                    acc[1].0 += coeff * dy.0;
                    acc[1].1 += coeff * dy.1;
                    acc[2].0 += coeff * dz.0;
                    acc[2].1 += coeff * dz.1;
                }
            }
        }
    }
    acc
}

/// London (GIAO) electric-dipole integral matrices `D_c(B)_{mu nu} =
/// <omega_mu|(r_c - O)|omega_nu>` (`c = x, y, z`; `O` = the gauge origin in
/// `options.origin`), built from the complex 1D London blocks ([`lao_axis_blocks`]).
/// Reduces to the real dipole integrals at `B = 0` and is Hermitian. Non-periodic.
/// This is the field-dependent electric-dipole operator behind the LAO optical-
/// rotation G-tensor (the magnetic-field derivative of the electric dipole) and other
/// London-orbital response properties.
pub fn lao_dipole_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    options: &ExternalFieldOptions,
) -> [CMatrix; 3] {
    let field = options.magnetic_field.unwrap_or(Vec3::zero());
    let origin = options.origin;
    let n = basis.len();
    let mut d = [CMatrix::zeros(n), CMatrix::zeros(n), CMatrix::zeros(n)];
    for mu in 0..n {
        let a_ao = &basis.aos[mu];
        let ca = system.atoms[a_ao.atom_index].position;
        for nu in 0..n {
            let b_ao = &basis.aos[nu];
            let cb = system.atoms[b_ao.atom_index].position;
            let dip = lao_dipole_pair(a_ao, b_ao, ca, cb, field, origin);
            for (c, dc) in dip.iter().enumerate() {
                d[c].re[(mu, nu)] = dc.0;
                d[c].im[(mu, nu)] = dc.1;
            }
        }
    }
    d
}

/// Result of a closed-shell magnetic (GFN1-xTB-M0) SCC.
#[derive(Clone, Debug)]
pub struct MagneticSccResult {
    /// Total energy (Hartree).
    pub energy: f64,
    pub band_energy: f64,
    pub scc_second_order: f64,
    pub scc_third_order: f64,
    pub repulsion_energy: f64,
    pub dispersion_energy: f64,
    pub halogen_energy: f64,
    pub shell_charges: Vec<f64>,
    /// Mulliken (monopole) dipole `mu = sum_shell q_shell (R_atom - origin)` in the
    /// combined electric+magnetic field (atomic units). The electric polarizability in
    /// a magnetic field, `alpha(B) = d mu / d E`, is the building block of MCD and the
    /// Cotton-Mouton effect.
    pub dipole: Vec3,
    /// Converged complex one-particle density `P(B)` in the London (GIAO) AO basis.
    pub density: CMatrix,
    /// Converged complex energy-weighted density `W(B) = C f eps C^H` (London AO
    /// basis), the Pulay weight for the overlap-derivative term of the gradient.
    pub energy_weighted_density: CMatrix,
    /// Converged AO-resolved SCC potential `vao` (`shell_potential` broadcast to
    /// AOs). The frozen-charge effective Hamiltonian is `F_munu = H0_munu -
    /// 1/2(vao_mu + vao_nu) S_munu`; exposed for the analytic magnetic response.
    pub shell_potential_ao: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

fn magnetic_mulliken(basis: &BasisSet, density: &CMatrix, overlap: &CMatrix) -> Vec<f64> {
    let n = basis.len();
    let mut pop = vec![0.0_f64; n];
    for mu in 0..n {
        let mut acc = 0.0;
        for nu in 0..n {
            // Re[(P S)_mu mu] = sum_nu (P.re S.re - P.im S.im)_{mu nu, nu mu}.
            acc += density.re[(mu, nu)] * overlap.re[(nu, mu)]
                - density.im[(mu, nu)] * overlap.im[(nu, mu)];
        }
        pop[mu] = acc;
    }
    let mut qsh = vec![0.0_f64; basis.shells.len()];
    for (ish, shell) in basis.shells.iter().enumerate() {
        let mut population = 0.0;
        for iao in shell.first_ao..shell.first_ao + shell.nao {
            population += pop[iao];
        }
        qsh[ish] = shell.reference_occ - population;
    }
    qsh
}

fn complex_trace_product(p: &CMatrix, h: &CMatrix) -> f64 {
    let n = p.n;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            acc += p.re[(i, j)] * h.re[(j, i)] - p.im[(i, j)] * h.im[(j, i)];
        }
    }
    acc
}

fn charge_rms(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let ss: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (ss / a.len() as f64).sqrt()
}

/// Closed-shell, zero-temperature **magnetic (GFN1-xTB-M0) SCC** for a uniform
/// magnetic field `options.external_field.magnetic_field`. The zero-field GFN1
/// Build the **secondary (dual) basis AOs** for GFN1-xTB-M1: one secondary AO per
/// primary GFN1 AO, with the same centre and Cartesian (angular) component but the
/// secondary contraction's radial primitives (matched by `(element, l, shell-rank)`),
/// renormalised to the primary AO's self-overlap. AOs without a secondary entry keep
/// their primary primitives (so they reduce to M0).
pub(crate) fn build_secondary_aos(
    basis: &BasisSet,
    system: &PeriodicSystem,
    secondary: &SecondaryBasis,
) -> Vec<AOBasisFunction> {
    // Rank of each shell among same-(atom, l) shells, so the k-th GFN1 shell of an
    // angular momentum maps to the k-th secondary contraction.
    let mut shell_rank = vec![0usize; basis.shells.len()];
    let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
    for (si, sh) in basis.shells.iter().enumerate() {
        let key = (sh.atom_index, sh.angular.as_index());
        let r = seen.entry(key).or_insert(0);
        shell_rank[si] = *r;
        *r += 1;
    }
    let mut out = Vec::with_capacity(basis.aos.len());
    for ao in &basis.aos {
        let mut sec = ao.clone();
        let rank = shell_rank[ao.shell_index];
        let l = ao.angular.as_index();
        if let Some(contraction) = secondary.contraction(ao.z, l, rank) {
            // The secondary-basis file lists contraction coefficients for
            // *normalized* primitives; fold in the same per-primitive
            // normalization `slater_to_gauss` uses for the primary GFN1 AOs so the
            // two bases share one convention (the shared Cartesian `components`
            // then carry the angular part identically).
            sec.primitives = contraction
                .primitives
                .iter()
                .map(|&(exponent, coefficient)| PrimitiveGaussian {
                    exponent,
                    coefficient: coefficient * crate::sto::primitive_norm(exponent, l),
                })
                .collect();
            let center = system.atoms[ao.atom_index].position;
            let s_self = crate::integrals::contracted_pair(&sec, &sec, center, center).0;
            let s_primary = crate::integrals::contracted_pair(ao, ao, center, center).0;
            if s_self > 1.0e-30 && s_primary > 0.0 {
                let scale = (s_primary / s_self).sqrt();
                for p in &mut sec.primitives {
                    p.coefficient *= scale;
                }
            }
        }
        out.push(sec);
    }
    out
}

/// GFN1-xTB-M kinetic-energy correction matrix over the AOs `ke_aos` (the primary
/// GFN1 AOs for M0, the secondary AOs for M1), indexed by the primary AO pair:
/// `corr_munu = <ke_mu|1/2 pi^2|ke_nu> - e^{i f_munu} <ke_mu|-1/2 nabla^2|ke_nu>`,
/// `f_munu = 1/2 (A_mu - A_nu).(R_mu + R_nu)`, `A_x = 1/2 B x (R_x - O)`. Complex;
/// zero at `B = 0`.
fn magnetic_ke_correction(
    ke_aos: &[AOBasisFunction],
    centers: &[Vec3],
    field: Vec3,
    origin: Vec3,
) -> ComplexAoMatrix {
    let n = ke_aos.len();
    // Per-AO minimum primitive exponent for Gaussian distance screening: the KE
    // correction <om|1/2 pi^2|om> - e^{if}<phi|1/2 p^2|phi> decays with the AO-pair
    // overlap, so pairs with `r^2 ea eb > 40(ea+eb)` (overlap < ~e^-40) are skipped.
    let ao_min_exp: Vec<f64> = ke_aos
        .iter()
        .map(|ao| {
            ao.primitives
                .iter()
                .map(|p| p.exponent)
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    // Each row is independent; parallelize over `mu` (deterministic per-element writes).
    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|mu| {
            let rmu = centers[mu];
            let a_mu = field.cross(rmu - origin) * 0.5;
            let ea = ao_min_exp[mu];
            let mut rre = vec![0.0_f64; n];
            let mut rim = vec![0.0_f64; n];
            for nu in 0..n {
                let rnu = centers[nu];
                let r2 = (rmu - rnu).norm2();
                let eb = ao_min_exp[nu];
                if r2 * ea * eb > 40.0 * (ea + eb) {
                    continue; // negligible KE pair
                }
                let a_nu = field.cross(rnu - origin) * 0.5;
                let (kr, ki) = lao_kinetic_pair(&ke_aos[mu], &ke_aos[nu], rmu, rnu, field, origin);
                let t = crate::integrals::contracted_kinetic(&ke_aos[mu], &ke_aos[nu], rmu, rnu);
                let f = 0.5 * (a_mu - a_nu).dot(rmu + rnu);
                rre[nu] = kr - f.cos() * t;
                rim[nu] = ki - f.sin() * t;
            }
            (rre, rim)
        })
        .collect();
    let mut re = Matrix::zeros(n, n);
    let mut im = Matrix::zeros(n, n);
    for (mu, (rre, rim)) in rows.into_iter().enumerate() {
        for nu in 0..n {
            re[(mu, nu)] = rre[nu];
            im[(mu, nu)] = rim[nu];
        }
    }
    ComplexAoMatrix { re, im }
}

/// The field-free GFN1 `H0` **shell-pair** prefactors `hij[(I, J)]`, i.e. the factor
/// in `H0_{mu nu} = hij(I, J) S_{mu nu}` for `mu` in shell `I` and `nu` in shell `J`
/// ([`crate::hamiltonian::build_h0_from_overlap`]: `hij` is built from the shell
/// self-energies, `hscale` and the distance polynomial only, so it is constant across
/// the AO block).
///
/// Recovered as `H0_{mu nu} / S_{mu nu}` at the AO pair with the **largest** `|S|` in
/// the block. Taking that ratio per AO pair instead is unsound: inside a block the
/// field-free overlap can vanish *exactly* by symmetry — `<O 2p_z|H 1s>` for a
/// molecule lying in the `xy` plane — while the London overlap `S(B)_{mu nu}` at an
/// in-plane field does not. A per-pair ratio then silently drops the whole band
/// contribution `hij S(B)_{mu nu}` for those pairs, and whether the denominator is a
/// hard zero or a `~1e-16` rounding artefact depends on where the molecule sits and
/// how it is oriented, so the energy stops being rotation / translation invariant.
/// A block whose overlap vanishes identically (beyond the integral cutoff) has
/// `H0 = 0` too, so `hij = 0` there is exact.
fn shell_pair_h0_prefactors(basis: &BasisSet, core: &crate::hamiltonian::HamiltonianCore) -> Matrix {
    let nsh = basis.shells.len();
    let mut hij = Matrix::zeros(nsh, nsh);
    for (ish, shi) in basis.shells.iter().enumerate() {
        for (jsh, shj) in basis.shells.iter().enumerate() {
            let (mut best, mut value) = (0.0_f64, 0.0_f64);
            for mu in shi.first_ao..shi.first_ao + shi.nao {
                for nu in shj.first_ao..shj.first_ao + shj.nao {
                    let s = core.integrals.overlap[(mu, nu)];
                    if s.abs() > best {
                        best = s.abs();
                        value = core.h0[(mu, nu)] / s;
                    }
                }
            }
            hij[(ish, jsh)] = value;
        }
    }
    hij
}

/// Assemble the closed-shell GFN1-xTB-M LAO Hamiltonian `H0(B)` and overlap `S(B)`
/// in the primary AO basis for the magnetic field / gauge origin in `field`
/// (Cheng & Wibowo-Teale eq 12/20). `secondary = Some(..)` evaluates the
/// kinetic-energy correction over the M1 dual basis. `hij` is the (`B`-independent)
/// field-free GFN1 `H0` shell-pair prefactor ([`shell_pair_h0_prefactors`]), so
/// `hij S(B)` is the band term over LAOs and the bracket
/// `<om|1/2 pi^2|om> - e^{i f}<phi|1/2 p^2|phi>` is the kinetic-energy correction.
/// Reduces to `(H0_real, S_real)` at `B = 0`. Shared by [`run_magnetic_scc`]'s SCC
/// loop and the analytic magnetic response so both see identical matrices.
fn assemble_magnetic_matrices(
    system: &PeriodicSystem,
    basis: &BasisSet,
    core: &crate::hamiltonian::HamiltonianCore,
    field: &ExternalFieldOptions,
    secondary: Option<&SecondaryBasis>,
) -> (CMatrix, CMatrix) {
    let n = basis.len();
    let bvec = field.magnetic_field.unwrap_or(Vec3::zero());
    let origin = field.origin;
    let s_lao = {
        let _p = crate::profile::scope("magnetic.assemble.lao_overlap");
        lao_overlap_matrix(system, basis, field)
    };
    let s_b = CMatrix {
        n,
        re: s_lao.re,
        im: s_lao.im,
    };
    let ke_aos = match secondary {
        Some(sec) => build_secondary_aos(basis, system, sec),
        None => basis.aos.clone(),
    };
    let centers: Vec<Vec3> = basis
        .aos
        .iter()
        .map(|ao| system.atoms[ao.atom_index].position)
        .collect();
    let corr = {
        let _p = crate::profile::scope("magnetic.assemble.ke");
        magnetic_ke_correction(&ke_aos, &centers, bvec, origin)
    };
    let hpair = shell_pair_h0_prefactors(basis, core);
    let mut h0_b = CMatrix::zeros(n);
    for mu in 0..n {
        let ish = basis.aos[mu].shell_index;
        for nu in 0..n {
            let hij = hpair[(ish, basis.aos[nu].shell_index)];
            h0_b.re[(mu, nu)] = hij * s_b.re[(mu, nu)] + corr.re[(mu, nu)];
            h0_b.im[(mu, nu)] = hij * s_b.im[(mu, nu)] + corr.im[(mu, nu)];
        }
    }
    (h0_b, s_b)
}

/// Build the GFN1-xTB-M LAO Hamiltonian `H0(B)` and overlap `S(B)` (primary AO
/// basis) for the magnetic field / gauge origin in `options.external_field`.
/// `secondary = Some(..)` selects the M1 dual basis for the kinetic-energy
/// correction. This is the public entry point used by the analytic magnetic
/// response (magnetizability / gradient) to take field derivatives of the
/// Hamiltonian; see [`assemble_magnetic_matrices`].
pub fn magnetic_h0_overlap(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
) -> Result<(CMatrix, CMatrix)> {
    let basis = {
        let _p = crate::profile::scope("magnetic.h0.basis");
        BasisSet::build(
            system,
            params,
            BasisOptions {
                nprim: options.nprim,
            },
        )?
    };
    let core = {
        let _p = crate::profile::scope("magnetic.h0.core");
        build_h0(system, &basis, params, &options.hamiltonian)?
    };
    Ok(assemble_magnetic_matrices(
        system,
        &basis,
        &core,
        &options.external_field,
        secondary,
    ))
}

/// Closed-shell **GFN1-xTB-M0** magnetic SCC (single basis): the kinetic-energy
/// correction is evaluated over the primary GFN1 AOs. See
/// [`run_magnetic_scc_m1`] for the dual-basis M1 variant. Non-periodic; reduces
/// exactly to the field-free GFN1 energy at `B = 0`. Cheng & Wibowo-Teale,
/// *J. Chem. Theory Comput.* **19**, 6226 (2023).
pub fn run_magnetic_scc(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> Result<MagneticSccResult> {
    run_magnetic_scc_inner(system, params, options, None)
}

/// Closed-shell **GFN1-xTB-M1** magnetic SCC: the kinetic-energy correction is
/// evaluated over the node-correct secondary (dual) basis (see
/// [`crate::secondary_basis`]), which substantially improves magnetic properties
/// (magnetizabilities, NMR shieldings) over M0 for heavier elements.
pub fn run_magnetic_scc_m1(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: &SecondaryBasis,
) -> Result<MagneticSccResult> {
    run_magnetic_scc_inner(system, params, options, Some(secondary))
}

/// Geometry-only (field-independent) inputs to the magnetic SCC: the GFN1 basis, the
/// field-free `H0` core, the shell-charge model and its Coulomb matrix, and the
/// classical (repulsion / dispersion / halogen) energies. These do not depend on the
/// magnetic or electric field, so the field-derivative / finite-difference response
/// routines (magnetizability, MCD, Cotton-Mouton, polarizability) build this **once**
/// and reuse it across all field evaluations instead of rebuilding per call.
struct MagneticGeom {
    basis: BasisSet,
    core: crate::hamiltonian::HamiltonianCore,
    shell_model: ShellChargeModel,
    amat: Matrix,
    repulsion: f64,
    dispersion: f64,
    halogen: f64,
}

fn magnetic_geom(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> Result<MagneticGeom> {
    if system.lattice.is_some() {
        return Err(Gfn1Error::InvalidInput(
            "magnetic SCC is implemented for non-periodic systems only".to_string(),
        ));
    }
    let basis = BasisSet::build(
        system,
        params,
        BasisOptions {
            nprim: options.nprim,
        },
    )?;
    let core = build_h0(system, &basis, params, &options.hamiltonian)?;
    let mut shell_model = ShellChargeModel::build(system, &basis, params)?;
    shell_model.charge_order = options.charge_order.max(3);
    let amat = effective_coulomb_matrix(system, &basis, &shell_model);
    let repulsion = repulsion_energy(system, params)?;
    let dispersion = if options.enable_dispersion {
        dispersion_energy(system, params, options.d3_reference_path.as_deref())?
    } else {
        0.0
    };
    let halogen = halogen_energy(system, params)?;
    Ok(MagneticGeom {
        basis,
        core,
        shell_model,
        amat,
        repulsion,
        dispersion,
        halogen,
    })
}

/// `H0`/`S` are evaluated over London orbitals (exact LAO overlap + the
/// kinetic-energy correction), and the SCC is solved with the complex Hermitian
/// generalized eigensolver. `secondary = Some(..)` selects the M1 dual basis for
/// the kinetic-energy correction; `None` is single-basis M0.
fn run_magnetic_scc_inner(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
) -> Result<MagneticSccResult> {
    let geom = magnetic_geom(system, params, options)?;
    run_magnetic_scc_with_geom(&geom, system, params, options, secondary, None)
}

/// Magnetic SCC over a prebuilt [`MagneticGeom`] (the geometry-only inputs). Only the
/// field-dependent LAO `H0(B)`/`S(B)` assembly and the SCC loop are done here, so a
/// caller varying only the field reuses one `MagneticGeom`.
fn run_magnetic_scc_with_geom(
    geom: &MagneticGeom,
    system: &PeriodicSystem,
    _params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    extra_h0: Option<&CMatrix>,
) -> Result<MagneticSccResult> {
    let _profile = crate::profile::scope("magnetic.scf.total");
    let field = options.external_field;
    if field.magnetic_field.is_none() {
        return Err(Gfn1Error::InvalidInput(
            "run_magnetic_scc requires options.external_field.magnetic_field".to_string(),
        ));
    }

    let basis = &geom.basis;
    let core = &geom.core;
    let n = basis.len();
    let nsh = basis.shells.len();

    // Exact London (GIAO) overlap S(B) and the GFN1-xTB-M0/M1 magnetic H0(B)
    // (`assemble_magnetic_matrices`); at B = 0 these are (H0_real, S_real) so the SCC
    // reduces to field-free GFN1. `secondary = Some` selects the M1 dual basis for the
    // kinetic-energy correction.
    let (mut h0_b, s_b) = assemble_magnetic_matrices(system, basis, core, &field, secondary);
    // Optional caller-supplied one-electron operator added to `H0(B)` (Hermitian).
    // Used by the NMR-shielding FD gate to inject the orbital-Zeeman / nuclear
    // magnetic-dipole couplings (`extra_h0 = B.L_O + m.L_A/r_A^3 + B.m.d`) so the
    // mixed second derivative `d^2E/dB dm` can be differenced; `None` leaves the SCC
    // unchanged (every magnetizability / polarizability caller passes `None`).
    if let Some(extra) = extra_h0 {
        for i in 0..n {
            for j in 0..n {
                h0_b.re[(i, j)] += extra.re[(i, j)];
                h0_b.im[(i, j)] += extra.im[(i, j)];
            }
        }
    }

    let shell_model = &geom.shell_model;
    let amat = &geom.amat;

    let charge = options.charge.unwrap_or(system.charge);
    let nelec = basis.total_reference_electrons - charge;
    let nocc = (nelec / 2.0).round() as usize;
    if (nelec - 2.0 * nocc as f64).abs() > 1.0e-6 {
        return Err(Gfn1Error::InvalidInput(
            "magnetic SCC currently supports closed-shell (even electron) systems only".to_string(),
        ));
    }
    // Embedded occupations: the lowest `2*nocc` of the `2n` paired states.
    let mut occ = vec![0.0_f64; 2 * n];
    for o in occ.iter_mut().take(2 * nocc) {
        *o = 1.0;
    }

    let (repulsion, dispersion, halogen) = (geom.repulsion, geom.dispersion, geom.halogen);

    // Optional uniform electric field, coupled to the GFN1 Mulliken monopoles exactly
    // as in the field-free path: a real per-shell site potential v_ext_i = -E·(R_i-O)
    // added to the SCC shift, with energy E_field = sum_i q_i v_ext_i. This enables the
    // combined electric+magnetic response (MCD = dalpha/dB, Cotton-Mouton = d2alpha/dB2).
    let elec_shell = electric_shell_potential(&field, system, basis);
    let elec_ao: Vec<f64> = {
        let mut v = vec![0.0_f64; n];
        if let Some(es) = &elec_shell {
            for (ish, shell) in basis.shells.iter().enumerate() {
                for iao in shell.first_ao..shell.first_ao + shell.nao {
                    v[iao] = es[ish];
                }
            }
        }
        v
    };

    let mixing = options.mixing.clamp(0.01, 1.0);
    let mut broyden = BroydenMixer::new(nsh, options.scc_broyden_size.max(2), mixing);
    let mut q = vec![0.0_f64; nsh];
    let mut converged = false;
    let mut iterations = 0usize;
    let mut last_energy: Option<f64> = None;
    let mut final_q = q.clone();
    let (mut final_band, mut final_scc2, mut final_scc3) = (0.0, 0.0, 0.0);
    let mut final_elec = 0.0_f64;
    // Converged complex density P(B) and energy-weighted density W(B) = C f eps C^H,
    // exposed for the (future) analytic magnetic gradient's Pulay contraction.
    let mut final_density: Option<CMatrix> = None;
    let mut final_ew_density: Option<CMatrix> = None;
    // Converged AO-resolved SCC potential `vao` (the `1/2(v_mu+v_nu) S` shift's per-AO
    // part), exposed so the analytic magnetic response can rebuild the frozen-charge
    // effective Hamiltonian `F = H0 - vao(.)S` and take its field derivatives.
    let mut final_vao = vec![0.0_f64; n];

    for iter in 1..=options.max_scc {
        iterations = iter;
        let scc = coulomb_energy_potential_from_matrix(basis, shell_model, &q, amat)?;
        let mut vao = vec![0.0_f64; n];
        for (ish, shell) in basis.shells.iter().enumerate() {
            for iao in shell.first_ao..shell.first_ao + shell.nao {
                vao[iao] = scc.shell_potential[ish];
            }
        }
        final_vao.clone_from(&vao);
        let mut fock = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                // SCC shift plus the (geometry-fixed) external electric-field site shift.
                let shift = 0.5 * (vao[i] + vao[j]) + 0.5 * (elec_ao[i] + elec_ao[j]);
                fock.re[(i, j)] = h0_b.re[(i, j)] - shift * s_b.re[(i, j)];
                fock.im[(i, j)] = h0_b.im[(i, j)] - shift * s_b.im[(i, j)];
            }
        }
        let eig = hermitian_generalized_eigen(&fock, &s_b, options.eigen_tolerance)?;
        let p = weighted_density(&eig, &occ)?;
        let new_q = magnetic_mulliken(basis, &p, &s_b);
        let band = complex_trace_product(&p, &h0_b);
        let elec = elec_shell
            .as_deref()
            .map(|es| electric_field_energy(es, &new_q))
            .unwrap_or(0.0);
        let energy = band + scc.second_order + scc.third_order + elec;
        let rms = charge_rms(&new_q, &q);
        let err = last_energy
            .map(|e| (energy - e).abs())
            .unwrap_or(f64::INFINITY);
        final_q = new_q.clone();
        final_band = band;
        final_scc2 = scc.second_order;
        final_scc3 = scc.third_order;
        final_elec = elec;
        let ew_weights: Vec<f64> = occ
            .iter()
            .zip(eig.values.iter())
            .map(|(g, e)| g * e)
            .collect();
        final_ew_density = Some(weighted_density(&eig, &ew_weights)?);
        final_density = Some(p);
        if err < options.energy_tolerance && rms < options.charge_tolerance {
            converged = true;
            break;
        }
        let residual: Vec<f64> = new_q.iter().zip(&q).map(|(a, b)| a - b).collect();
        q = broyden
            .next(&q, &residual)
            .filter(|c| c.iter().all(|v| v.is_finite() && v.abs() < 10.0))
            .unwrap_or_else(|| {
                q.iter()
                    .zip(&residual)
                    .map(|(qq, r)| qq + mixing * r)
                    .collect()
            });
        last_energy = Some(energy);
    }
    if !converged {
        return Err(Gfn1Error::SccNotConverged {
            iterations,
            rms: 0.0,
        });
    }
    let energy =
        final_band + final_scc2 + final_scc3 + final_elec + repulsion + dispersion + halogen;
    // Mulliken (monopole) dipole mu = sum_shell q_shell (R_atom - origin).
    let mut dipole = Vec3::zero();
    for (ish, shell) in basis.shells.iter().enumerate() {
        dipole += (system.atoms[shell.atom_index].position - field.origin) * final_q[ish];
    }
    Ok(MagneticSccResult {
        energy,
        band_energy: final_band,
        scc_second_order: final_scc2,
        scc_third_order: final_scc3,
        repulsion_energy: repulsion,
        dispersion_energy: dispersion,
        halogen_energy: halogen,
        shell_charges: final_q,
        dipole,
        density: final_density.expect("converged magnetic SCC stored no density"),
        energy_weighted_density: final_ew_density
            .expect("converged magnetic SCC stored no energy-weighted density"),
        shell_potential_ao: final_vao,
        iterations,
        converged,
    })
}

/// Isotropic magnetizability `xi_iso = (1/3) Tr xi`, `xi_ab = -d^2 E / dB_a dB_b`
/// (eq 26 of Cheng & Wibowo-Teale), by central finite difference of the magnetic
/// SCC energy along each Cartesian field direction. Returns atomic units
/// (Hartree / (atomic field unit)^2); multiply by `MAGNETIZABILITY_AU_TO_SI` for
/// `10^-30 J T^-2`. `secondary = Some(..)` selects M1, `None` selects M0. The field
/// in `options.external_field` is overridden. Non-periodic.
///
/// Sign: with `E(B) = E(0) - 1/2 xi_ab B_a B_b`, a diamagnetic closed-shell molecule
/// returns a **negative** `xi_iso` (gated in
/// `tests/magnetic.rs::diamagnetic_molecules_have_negative_isotropic_magnetizability`).
pub fn magnetizability_isotropic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    step: f64,
) -> Result<f64> {
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "magnetizability step must be positive".to_string(),
        ));
    }
    let energy_at = |b: Vec3| -> Result<f64> {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        let result = match secondary {
            Some(sec) => run_magnetic_scc_m1(system, params, &opt, sec)?,
            None => run_magnetic_scc(system, params, &opt)?,
        };
        Ok(result.energy)
    };
    let e0 = energy_at(Vec3::zero())?;
    let mut trace = 0.0;
    for axis in 0..3 {
        let mut bp = Vec3::zero();
        let mut bm = Vec3::zero();
        match axis {
            0 => {
                bp.x = step;
                bm.x = -step;
            }
            1 => {
                bp.y = step;
                bm.y = -step;
            }
            _ => {
                bp.z = step;
                bm.z = -step;
            }
        }
        let ep = energy_at(bp)?;
        let em = energy_at(bm)?;
        let d2 = (ep - 2.0 * e0 + em) / (step * step);
        trace += -d2;
    }
    Ok(trace / 3.0)
}

/// Dense complex matrix product `A * B` for `n x n` [`CMatrix`] operands.
fn cmatmul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let n = a.n;
    let mut o = CMatrix::zeros(n);
    for i in 0..n {
        for k in 0..n {
            let (ar, ai) = (a.re[(i, k)], a.im[(i, k)]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            for j in 0..n {
                o.re[(i, j)] += ar * b.re[(k, j)] - ai * b.im[(k, j)];
                o.im[(i, j)] += ar * b.im[(k, j)] + ai * b.re[(k, j)];
            }
        }
    }
    o
}

/// `Re Tr(A B)` for complex `A`, `B` ([`CMatrix`]).
fn re_trace_cc(a: &CMatrix, b: &CMatrix) -> f64 {
    let n = a.n;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            acc += a.re[(i, j)] * b.re[(j, i)] - a.im[(i, j)] * b.im[(j, i)];
        }
    }
    acc
}

/// First-order density response `P^a` and energy-weighted-density response `W^a` of
/// the closed-shell magnetic SCC for the AO field derivatives `h0_a = dH0/dB_a`,
/// `s_a = dS/dB_a` (anti-Hermitian at `B = 0`). `c`/`eps` are the real `B = 0`
/// generalized eigenvectors/values, `p0` the density, `f0c` the converged Fock (as a
/// [`CMatrix`]), `vao` the per-AO SCC potential. The response is uncoupled
/// (`dq/dB = 0`): the occ-virt block of the orbital response is the canonical CP-SCC
/// driven by the Fock derivative `F^a = H0^a - vao(.)S^a`, the occ-occ block is the
/// reorthonormalization `-1/2 S^a_mo` (degeneracy-safe), and the energy-weighted
/// density uses the McWeeny identity `W = 1/2 P F P` (no occ-occ orbital ambiguity).
#[allow(clippy::too_many_arguments)]
fn magnetic_first_order_response(
    n: usize,
    nocc: usize,
    c: &Matrix,
    ct: &Matrix,
    eps: &[f64],
    vao: &[f64],
    p0: &CMatrix,
    f0c: &CMatrix,
    h0_a: &CMatrix,
    s_a: &CMatrix,
) -> (CMatrix, CMatrix) {
    let mut f_a = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (vao[i] + vao[j]);
            f_a.re[(i, j)] = h0_a.re[(i, j)] - v * s_a.re[(i, j)];
            f_a.im[(i, j)] = h0_a.im[(i, j)] - v * s_a.im[(i, j)];
        }
    }
    let mo = |m: &CMatrix| -> CMatrix {
        CMatrix {
            n,
            re: ct.matmul(&m.re).unwrap().matmul(c).unwrap(),
            im: ct.matmul(&m.im).unwrap().matmul(c).unwrap(),
        }
    };
    let fmo = mo(&f_a);
    let smo = mo(s_a);
    let col = |m: &Matrix, k: usize| -> Vec<f64> { (0..n).map(|r| m[(r, k)]).collect() };
    let mut pa = CMatrix::zeros(n);
    for i in 0..nocc {
        let mut u_re = vec![0.0; n];
        let mut u_im = vec![0.0; n];
        for p in 0..n {
            if p < nocc {
                u_re[p] = -0.5 * smo.re[(p, i)];
                u_im[p] = -0.5 * smo.im[(p, i)];
            } else {
                let denom = eps[i] - eps[p];
                u_re[p] = (fmo.re[(p, i)] - eps[i] * smo.re[(p, i)]) / denom;
                u_im[p] = (fmo.im[(p, i)] - eps[i] * smo.im[(p, i)]) / denom;
            }
        }
        let mut cia_re = vec![0.0; n];
        let mut cia_im = vec![0.0; n];
        for p in 0..n {
            let cp = col(c, p);
            for r in 0..n {
                cia_re[r] += u_re[p] * cp[r];
                cia_im[r] += u_im[p] * cp[r];
            }
        }
        let ci = col(c, i);
        for r in 0..n {
            for s in 0..n {
                pa.re[(r, s)] += 2.0 * (cia_re[r] * ci[s] + ci[r] * cia_re[s]);
                pa.im[(r, s)] += 2.0 * (cia_im[r] * ci[s] - ci[r] * cia_im[s]);
            }
        }
    }
    let t1 = cmatmul(&cmatmul(&pa, f0c), p0);
    let t2 = cmatmul(&cmatmul(p0, &f_a), p0);
    let t3 = cmatmul(&cmatmul(p0, f0c), &pa);
    let mut wa = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            wa.re[(i, j)] = 0.5 * (t1.re[(i, j)] + t2.re[(i, j)] + t3.re[(i, j)]);
            wa.im[(i, j)] = 0.5 * (t1.im[(i, j)] + t2.im[(i, j)] + t3.im[(i, j)]);
        }
    }
    (pa, wa)
}

/// Rigidly move a molecule so that its atom centroid sits at the coordinate
/// origin, carrying the field/gauge origin along with it.
///
/// This is an exact **gauge transformation** for the London basis: translating
/// the molecule by `d` multiplies `S(B)` and `H0(B)` by the same diagonal
/// unitary `exp(i A_mu . d)`, so every observable is unchanged. The *finite
/// differences* below are not invariant under it, though, because they are
/// taken along fixed global field axes: their effective expansion parameter is
/// `step` times the LAO phase area `1/2 |B x R_mu . R_nu|`, which grows with
/// the molecule's distance from the coordinate origin. Measured on non-eq water
/// (`step = 4e-3`), the tensor moves by rel `6.3e-6` under a 2-bohr rigid shift
/// and by rel `1.2e-3` under a 9.4-bohr one — a `|d|^4` growth of an error that
/// has nothing to do with the physics, only with where the caller happened to
/// put the molecule.
///
/// Recentring removes that dependence at the root: the FD parameter is then set
/// by the molecule's own extent, and the same molecule always produces the same
/// tensor. Measured residual under a rigid translation afterwards: rel `1.8e-12`
/// (2 bohr) and `7.5e-12` (9.4 bohr), i.e. SCC-convergence noise only.
///
/// Translating `external_field.origin` by the same vector keeps the electric
/// site potential `-E.(R_A - origin)` and the Mulliken dipole identical, so the
/// recentring is invisible to a simultaneous electric field. Periodic inputs are
/// returned untouched (the LAO path is molecular).
fn recentred_for_field_derivatives(
    system: &PeriodicSystem,
    options: &ElectronicOptions,
) -> (PeriodicSystem, ElectronicOptions) {
    if system.lattice.is_some() || system.atoms.is_empty() {
        return (system.clone(), options.clone());
    }
    let mut centroid = Vec3::zero();
    for atom in &system.atoms {
        centroid = centroid + atom.position;
    }
    let centroid = centroid * (1.0 / system.atoms.len() as f64);
    let mut recentred = system.clone();
    for atom in &mut recentred.atoms {
        atom.position = atom.position - centroid;
    }
    let mut shifted = options.clone();
    shifted.external_field.origin = shifted.external_field.origin - centroid;
    (recentred, shifted)
}

/// Richardson combination `(4 D(h/2) - D(h)) / 3` of the same central-difference
/// matrix evaluated at two steps, elementwise on the real and imaginary parts.
///
/// The central differences of the LAO builder are exactly `O(h^2)` — the
/// residual ladder in `tests/magnetizability_frame_invariance.rs` measures a
/// ratio of 4.00 per halving across a decade of steps — so this leaves `O(h^4)`
/// and is legitimate at every step in the truncation-dominated regime.
fn richardson_pair(coarse: &CMatrix, fine: &CMatrix) -> CMatrix {
    let n = coarse.n;
    let mut out = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            out.re[(i, j)] = (4.0 * fine.re[(i, j)] - coarse.re[(i, j)]) / 3.0;
            out.im[(i, j)] = (4.0 * fine.im[(i, j)] - coarse.im[(i, j)]) / 3.0;
        }
    }
    out
}

/// Diagonal magnetizability tensor `xi_aa = -d^2 E / dB_a^2` (atomic units;
/// multiply by [`MAGNETIZABILITY_AU_TO_SI`] for `10^-30 J T^-2`), evaluated by the
/// **analytic McWeeny density-matrix CP-SCC response** instead of differencing the
/// energy. Only the LAO `H0(B)` / `S(B)` *integrals* are differenced (`magnetic_h0_
/// overlap` at `±step`, no SCF); the orbital response is analytic, so this costs one
/// magnetic SCC plus cheap builder evaluations rather than the `6+1` full SCCs of
/// [`magnetizability_isotropic`].
///
/// **What `step` means.** The integral derivatives are Richardson extrapolated:
/// each is built at `step` *and* `step / 2` and combined as `(4 D(h/2) - D(h))/3`,
/// so `step` is the **coarse** node of the pair and the truncation is `O(step^4)`,
/// not `O(step^2)`. That costs twice the builder evaluations but still only one SCC
/// (measured on the full tensor: water `68 -> 112 ms`, methane `73 -> 142 ms`), and
/// it is what makes the result independent of the coordinate frame — see
/// [`recentred_for_field_derivatives`] for the other half of the fix and
/// `tests/magnetizability_frame_invariance.rs` for the ladders behind both. Useful
/// steps are `4e-3 .. 1.6e-2`; the extrapolation is at its best (frame residual
/// rel `~5e-11`) around `4e-3` to `8e-3` and degrades below `2e-3`, where the
/// `1/h^2` amplification of builder rounding takes over.
///
/// For each field direction `a` (closed shell, at `B = 0`, real reference orbitals):
/// ```text
/// xi_aa = -[ Tr(P0 H0^aa) - Tr(W0 S^aa)                         (diamagnetic)
///          + Tr(P^a H0^a) - Tr(W^a S^a)                         (paramagnetic response)
///          - sum_mu vao_mu (Re(P^a S^a)_mu,mu + Re(P0 S^aa)_mu,mu) ]  (charge-overlap)
/// ```
/// where the first-order density response `P^a` has occ-virt block from the canonical
/// CP-SCC (driven by the Fock derivative `F^a = H0^a - vao(.)S^a`) and occ-occ block
/// `-1/2 S^a_mo` (reorthonormalization; degeneracy-safe), and the energy-weighted
/// density response uses the density-matrix identity `W = 1/2 P F P`:
/// `W^a = 1/2 (P^a F0 P0 + P0 F^a P0 + P0 F0 P^a)` (no occ-occ orbital ambiguity).
/// The magnetic response is uncoupled (`dq/dB = 0` at first order by time reversal),
/// but the Mulliken charges still depend on `B` through `S(B)`, giving the per-AO
/// charge-overlap term (the symmetric `1/2(vao_mu+vao_nu)` Pulay form vanishes for the
/// anti-Hermitian `S^a`). `secondary = Some(..)` selects M1, `None` M0. Non-periodic.
///
/// Refs: M. Malagoli, density-matrix CPHF (McWeeny purification, GIAO); Cheng &
/// Wibowo-Teale, *J. Chem. Theory Comput.* **19**, 6226 (2023).
pub fn magnetizability_diagonal_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    step: f64,
) -> Result<[f64; 3]> {
    let _profile = crate::profile::scope("magnetic.analytic_magnetizability.total");
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "magnetizability step must be positive".to_string(),
        ));
    }
    // Frame fix: differentiate in the molecule's own centroid frame, so the FD
    // truncation error cannot depend on where the caller placed the molecule.
    let (system, options) = recentred_for_field_derivatives(system, options);
    let (system, options) = (&system, &options);
    let with_field = |b: Vec3| -> ElectronicOptions {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        opt
    };
    let unit = |axis: usize, s: f64| -> Vec3 {
        match axis {
            0 => Vec3::new(s, 0.0, 0.0),
            1 => Vec3::new(0.0, s, 0.0),
            _ => Vec3::new(0.0, 0.0, s),
        }
    };

    // Geometry-only inputs built once and reused across every field evaluation below.
    let geom = magnetic_geom(system, params, options)?;
    let h0s = |opt: &ElectronicOptions| -> (CMatrix, CMatrix) {
        assemble_magnetic_matrices(
            system,
            &geom.basis,
            &geom.core,
            &opt.external_field,
            secondary,
        )
    };
    // Field-free reference: converged density / EW density / SCC potential, and the
    // real generalized eigenbasis of the converged Fock F0 = H0 - vao(.)S.
    let opt0 = with_field(Vec3::zero());
    let scc0 = run_magnetic_scc_with_geom(&geom, system, params, &opt0, secondary, None)?;
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = h0s(&opt0);
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, options.eigen_tolerance)?;
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let p0 = &scc0.density;
    let mut f0c = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            f0c.re[(i, j)] = f0[(i, j)];
        }
    }
    // MO transform of a complex AO matrix M -> C^T M C (real C).
    let mo = |m: &CMatrix| -> CMatrix {
        CMatrix {
            n,
            re: ct.matmul(&m.re).unwrap().matmul(c).unwrap(),
            im: ct.matmul(&m.im).unwrap().matmul(c).unwrap(),
        }
    };

    // First/second field derivatives of H0(B), S(B) along one axis, by central FD
    // of the LAO builder at one step: `[H0^a, S^a, H0^aa, S^aa]`.
    let axis_derivatives = |axis: usize, h: f64| -> [CMatrix; 4] {
        let (h0p, sp) = h0s(&with_field(unit(axis, h)));
        let (h0m, sm) = h0s(&with_field(unit(axis, -h)));
        let inv = 1.0 / (2.0 * h);
        let inv2 = 1.0 / (h * h);
        let mut h0_a = CMatrix::zeros(n);
        let mut s_a = CMatrix::zeros(n);
        let mut h0_aa = CMatrix::zeros(n);
        let mut s_aa = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                h0_a.re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) * inv;
                h0_a.im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) * inv;
                s_a.re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) * inv;
                s_a.im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) * inv;
                h0_aa.re[(i, j)] = (h0p.re[(i, j)] - 2.0 * h00.re[(i, j)] + h0m.re[(i, j)]) * inv2;
                h0_aa.im[(i, j)] = (h0p.im[(i, j)] - 2.0 * h00.im[(i, j)] + h0m.im[(i, j)]) * inv2;
                s_aa.re[(i, j)] = (sp.re[(i, j)] - 2.0 * s00.re[(i, j)] + sm.re[(i, j)]) * inv2;
                s_aa.im[(i, j)] = (sp.im[(i, j)] - 2.0 * s00.im[(i, j)] + sm.im[(i, j)]) * inv2;
            }
        }
        [h0_a, s_a, h0_aa, s_aa]
    };
    let mut diag = [0.0_f64; 3];
    for axis in 0..3 {
        // Richardson over (step, step/2): removes the leading O(step^2) FD
        // truncation, which is what makes the result frame independent.
        let coarse = axis_derivatives(axis, step);
        let fine = axis_derivatives(axis, 0.5 * step);
        let h0_a = richardson_pair(&coarse[0], &fine[0]);
        let s_a = richardson_pair(&coarse[1], &fine[1]);
        let h0_aa = richardson_pair(&coarse[2], &fine[2]);
        let s_aa = richardson_pair(&coarse[3], &fine[3]);
        let h0mo_aa = mo(&h0_aa);
        let smo_aa = mo(&s_aa);

        // Diamagnetic (no response): Tr(P0 H0^aa) - Tr(W0 S^aa), MO-diagonal over occ.
        let mut dia = 0.0;
        for i in 0..nocc {
            dia += 2.0 * (h0mo_aa.re[(i, i)] - eps[i] * smo_aa.re[(i, i)]);
        }

        // First-order density / EW-density response P^a, W^a (uncoupled CP-SCC).
        let (pa, wa) =
            magnetic_first_order_response(n, nocc, c, &ct, eps, vao, p0, &f0c, &h0_a, &s_a);
        let para = re_trace_cc(&pa, &h0_a) - re_trace_cc(&wa, &s_a);
        // Charge-overlap response (per-AO): -sum_mu vao_mu (Re(P^a S^a) + Re(P0 S^aa))_mu,mu.
        let mut chargeov = 0.0;
        for mu in 0..n {
            let mut t = 0.0;
            for nu in 0..n {
                t += pa.re[(mu, nu)] * s_a.re[(nu, mu)] - pa.im[(mu, nu)] * s_a.im[(nu, mu)];
                t += p0.re[(mu, nu)] * s_aa.re[(nu, mu)] - p0.im[(mu, nu)] * s_aa.im[(nu, mu)];
            }
            chargeov -= vao[mu] * t;
        }
        diag[axis] = -(dia + para + chargeov);
    }
    Ok(diag)
}

/// Isotropic magnetizability `xi_iso = (1/3) Tr xi` from the analytic CP-SCC
/// diagonal ([`magnetizability_diagonal_analytic`]); atomic units (multiply by
/// [`MAGNETIZABILITY_AU_TO_SI`] for `10^-30 J T^-2`). Validated to match
/// [`magnetizability_isotropic`] (the finite-field reference) to <0.1%.
pub fn magnetizability_isotropic_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    step: f64,
) -> Result<f64> {
    let d = magnetizability_diagonal_analytic(system, params, options, secondary, step)?;
    Ok((d[0] + d[1] + d[2]) / 3.0)
}

/// Full symmetric magnetizability tensor `xi_ab = -d^2 E / dB_a dB_b` (atomic units;
/// multiply by [`MAGNETIZABILITY_AU_TO_SI`] for `10^-30 J T^-2`) from the analytic
/// McWeeny density-matrix CP-SCC response. The diagonal matches
/// [`magnetizability_diagonal_analytic`]; the off-diagonals use the mixed second field
/// derivative `H0^ab`/`S^ab` (cross finite difference of the LAO builder) and the
/// symmetrized cross response `1/2[Tr(P^a H0^b) + Tr(P^b H0^a)] - ...`. `secondary =
/// Some(..)` selects M1. Non-periodic. See [`magnetizability_diagonal_analytic`] for
/// the term-by-term derivation and for what `step` means (it is the coarse node of a
/// Richardson pair, not a bare central-difference step).
///
/// **Frame independence.** The cross finite difference runs along the *global* field
/// axes, so its truncation error is not a tensor: at a bare `step = 4e-3` the tensor
/// moved by rel `6.3e-6` under a 2-bohr rigid translation (rel `1.2e-3` at 9.4 bohr)
/// and broke `xi(R r) = R xi(r) R^T` by rel `2.7e-6`. Recentring the molecule on its
/// centroid kills the translation dependence structurally and the Richardson pair
/// removes the `O(step^2)` term the rotation residual is made of; both are gated to
/// rel `1e-9` in `tests/magnetizability_frame_invariance.rs`. Shrinking `step`
/// instead cannot get there: the bare rotation residual bottoms out at rel `~1e-8`
/// near `step = 2.5e-4` and then rises again on builder rounding.
pub fn magnetizability_tensor_analytic(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    step: f64,
) -> Result<[[f64; 3]; 3]> {
    let _profile = crate::profile::scope("magnetic.analytic_magnetizability.tensor");
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "magnetizability step must be positive".to_string(),
        ));
    }
    // Frame fix: differentiate in the molecule's own centroid frame, so the FD
    // truncation error cannot depend on where the caller placed the molecule.
    let (system, options) = recentred_for_field_derivatives(system, options);
    let (system, options) = (&system, &options);
    let with_field = |b: Vec3| -> ElectronicOptions {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        opt
    };
    let axis_vec = |axis: usize, s: f64| -> Vec3 {
        match axis {
            0 => Vec3::new(s, 0.0, 0.0),
            1 => Vec3::new(0.0, s, 0.0),
            _ => Vec3::new(0.0, 0.0, s),
        }
    };
    // Two-axis field s_a*h on axis a plus s_b*h on axis b (b != a).
    let pair_vec = |a: usize, sa: f64, b: usize, sb: f64, h: f64| -> Vec3 {
        let mut v = axis_vec(a, sa * h);
        match b {
            0 => v.x += sb * h,
            1 => v.y += sb * h,
            _ => v.z += sb * h,
        }
        v
    };

    let geom = magnetic_geom(system, params, options)?;
    let h0s = |opt: &ElectronicOptions| -> (CMatrix, CMatrix) {
        assemble_magnetic_matrices(
            system,
            &geom.basis,
            &geom.core,
            &opt.external_field,
            secondary,
        )
    };
    let opt0 = with_field(Vec3::zero());
    let scc0 = run_magnetic_scc_with_geom(&geom, system, params, &opt0, secondary, None)?;
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = h0s(&opt0);
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, options.eigen_tolerance)?;
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let p0 = &scc0.density;
    let mut f0c = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            f0c.re[(i, j)] = f0[(i, j)];
        }
    }
    let mo = |m: &CMatrix| -> CMatrix {
        CMatrix {
            n,
            re: ct.matmul(&m.re).unwrap().matmul(c).unwrap(),
            im: ct.matmul(&m.im).unwrap().matmul(c).unwrap(),
        }
    };
    // `[H0^a, S^a, H0^aa, S^aa]` from the central FD of the LAO builder at step `h`.
    let axis_derivatives = |axis: usize, h: f64| -> [CMatrix; 4] {
        let (h0p, sp) = h0s(&with_field(axis_vec(axis, h)));
        let (h0m, sm) = h0s(&with_field(axis_vec(axis, -h)));
        let inv = 1.0 / (2.0 * h);
        let inv2 = 1.0 / (h * h);
        let mut da = CMatrix::zeros(n);
        let mut dsa = CMatrix::zeros(n);
        let mut h0_aa = CMatrix::zeros(n);
        let mut s_aa = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                da.re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) * inv;
                da.im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) * inv;
                dsa.re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) * inv;
                dsa.im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) * inv;
                h0_aa.re[(i, j)] = (h0p.re[(i, j)] - 2.0 * h00.re[(i, j)] + h0m.re[(i, j)]) * inv2;
                h0_aa.im[(i, j)] = (h0p.im[(i, j)] - 2.0 * h00.im[(i, j)] + h0m.im[(i, j)]) * inv2;
                s_aa.re[(i, j)] = (sp.re[(i, j)] - 2.0 * s00.re[(i, j)] + sm.re[(i, j)]) * inv2;
                s_aa.im[(i, j)] = (sp.im[(i, j)] - 2.0 * s00.im[(i, j)] + sm.im[(i, j)]) * inv2;
            }
        }
        [da, dsa, h0_aa, s_aa]
    };

    // Per-axis first derivatives and responses, plus the diagonal second derivatives.
    let mut h0_a: Vec<CMatrix> = Vec::with_capacity(3);
    let mut s_a: Vec<CMatrix> = Vec::with_capacity(3);
    let mut pa: Vec<CMatrix> = Vec::with_capacity(3);
    let mut wa: Vec<CMatrix> = Vec::with_capacity(3);
    let mut tensor = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        // Richardson over (step, step/2): removes the leading O(step^2) FD
        // truncation, which is what makes the tensor frame independent.
        let coarse = axis_derivatives(a, step);
        let fine = axis_derivatives(a, 0.5 * step);
        let da = richardson_pair(&coarse[0], &fine[0]);
        let dsa = richardson_pair(&coarse[1], &fine[1]);
        let h0_aa = richardson_pair(&coarse[2], &fine[2]);
        let s_aa = richardson_pair(&coarse[3], &fine[3]);
        let (p_a, w_a) =
            magnetic_first_order_response(n, nocc, c, &ct, eps, vao, p0, &f0c, &da, &dsa);
        // Diagonal element xi_aa.
        let h0mo_aa = mo(&h0_aa);
        let smo_aa = mo(&s_aa);
        let mut dia = 0.0;
        for i in 0..nocc {
            dia += 2.0 * (h0mo_aa.re[(i, i)] - eps[i] * smo_aa.re[(i, i)]);
        }
        let para = re_trace_cc(&p_a, &da) - re_trace_cc(&w_a, &dsa);
        let mut chargeov = 0.0;
        for mu in 0..n {
            let mut t = 0.0;
            for nu in 0..n {
                t += p_a.re[(mu, nu)] * dsa.re[(nu, mu)] - p_a.im[(mu, nu)] * dsa.im[(nu, mu)];
                t += p0.re[(mu, nu)] * s_aa.re[(nu, mu)] - p0.im[(mu, nu)] * s_aa.im[(nu, mu)];
            }
            chargeov -= vao[mu] * t;
        }
        tensor[a][a] = -(dia + para + chargeov);
        h0_a.push(da);
        s_a.push(dsa);
        pa.push(p_a);
        wa.push(w_a);
    }

    // Off-diagonal elements xi_ab (a < b), symmetrized.
    for a in 0..3 {
        for b in (a + 1)..3 {
            let cross_derivatives = |h: f64| -> [CMatrix; 2] {
                let (pp, spp) = h0s(&with_field(pair_vec(a, 1.0, b, 1.0, h)));
                let (pm, spm) = h0s(&with_field(pair_vec(a, 1.0, b, -1.0, h)));
                let (mp, smp) = h0s(&with_field(pair_vec(a, -1.0, b, 1.0, h)));
                let (mm, smm) = h0s(&with_field(pair_vec(a, -1.0, b, -1.0, h)));
                let mut h0_ab = CMatrix::zeros(n);
                let mut s_ab = CMatrix::zeros(n);
                let q = 1.0 / (4.0 * h * h);
                for i in 0..n {
                    for j in 0..n {
                        h0_ab.re[(i, j)] =
                            (pp.re[(i, j)] - pm.re[(i, j)] - mp.re[(i, j)] + mm.re[(i, j)]) * q;
                        h0_ab.im[(i, j)] =
                            (pp.im[(i, j)] - pm.im[(i, j)] - mp.im[(i, j)] + mm.im[(i, j)]) * q;
                        s_ab.re[(i, j)] =
                            (spp.re[(i, j)] - spm.re[(i, j)] - smp.re[(i, j)] + smm.re[(i, j)]) * q;
                        s_ab.im[(i, j)] =
                            (spp.im[(i, j)] - spm.im[(i, j)] - smp.im[(i, j)] + smm.im[(i, j)]) * q;
                    }
                }
                [h0_ab, s_ab]
            };
            // Same Richardson pair as the diagonal blocks; the cross difference
            // carries the same O(h^2) truncation.
            let coarse = cross_derivatives(step);
            let fine = cross_derivatives(0.5 * step);
            let h0_ab = richardson_pair(&coarse[0], &fine[0]);
            let s_ab = richardson_pair(&coarse[1], &fine[1]);
            // Mixed diamagnetic Tr(P0 H0^ab) - Tr(W0 S^ab) (MO-diagonal over occ).
            let h0mo_ab = mo(&h0_ab);
            let smo_ab = mo(&s_ab);
            let mut dia = 0.0;
            for i in 0..nocc {
                dia += 2.0 * (h0mo_ab.re[(i, i)] - eps[i] * smo_ab.re[(i, i)]);
            }
            // Symmetrized cross response.
            let para = 0.5
                * (re_trace_cc(&pa[a], &h0_a[b]) + re_trace_cc(&pa[b], &h0_a[a])
                    - re_trace_cc(&wa[a], &s_a[b])
                    - re_trace_cc(&wa[b], &s_a[a]));
            let mut chargeov = 0.0;
            for mu in 0..n {
                let mut t = 0.0;
                for nu in 0..n {
                    t += 0.5
                        * (pa[a].re[(mu, nu)] * s_a[b].re[(nu, mu)]
                            - pa[a].im[(mu, nu)] * s_a[b].im[(nu, mu)]
                            + pa[b].re[(mu, nu)] * s_a[a].re[(nu, mu)]
                            - pa[b].im[(mu, nu)] * s_a[a].im[(nu, mu)]);
                    t += p0.re[(mu, nu)] * s_ab.re[(nu, mu)] - p0.im[(mu, nu)] * s_ab.im[(nu, mu)];
                }
                chargeov -= vao[mu] * t;
            }
            let xi = -(dia + para + chargeov);
            tensor[a][b] = xi;
            tensor[b][a] = xi;
        }
    }
    Ok(tensor)
}

/// Atomic unit of magnetizability in `10^-30 J T^-2` (`e^2 a_0^2 / m_e =
/// 7.8910366008e-29 J T^-2`, CODATA 2018), for comparing
/// [`magnetizability_isotropic`] with the SI-unit literature/benchmark values (e.g.
/// Cheng & Wibowo-Teale Figure 2 / SI). Gated against the underlying SI constants by
/// `tests/magnetic.rs::magnetizability_au_to_si_matches_codata`; before that gate the
/// literal was `78.9103832`, wrong in the 7th significant figure.
pub const MAGNETIZABILITY_AU_TO_SI: f64 = 78.910_366_008;

fn cartesian_unit(axis: usize, s: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    }
}

/// Electric dipole polarizability `alpha_ij(B) = d mu_i / d E_j` (atomic units,
/// `e^2 a_0^2 / E_h`) in the uniform magnetic field set in `options.external_field`,
/// by central finite field of the combined electric+magnetic SCC Mulliken (monopole)
/// dipole. At `B = 0` this reduces to the field-free GFN1 monopole polarizability.
/// `secondary = Some(..)` selects M1; `e_step` is the electric-field step. This is the
/// building block of MCD (`d alpha / d B`) and the Cotton-Mouton effect
/// (`d^2 alpha / d B^2`). Non-periodic.
pub fn magnetic_polarizability(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    e_step: f64,
) -> Result<[[f64; 3]; 3]> {
    if !(e_step.is_finite() && e_step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "polarizability electric-field step must be positive".to_string(),
        ));
    }
    let geom = magnetic_geom(system, params, options)?;
    magnetic_polarizability_with_geom(&geom, system, params, options, secondary, e_step)
}

/// [`magnetic_polarizability`] over a prebuilt [`MagneticGeom`] (the field-independent
/// geometry data). The six combined-SCC electric-field evaluations reuse one `geom`;
/// the MCD / Cotton-Mouton routines build `geom` once and reuse it across all field
/// points.
fn magnetic_polarizability_with_geom(
    geom: &MagneticGeom,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    e_step: f64,
) -> Result<[[f64; 3]; 3]> {
    let dipole_at = |e: Vec3| -> Result<Vec3> {
        let mut opt = options.clone();
        opt.external_field.electric_field = Some(e);
        Ok(run_magnetic_scc_with_geom(geom, system, params, &opt, secondary, None)?.dipole)
    };
    let mut alpha = [[0.0_f64; 3]; 3];
    for j in 0..3 {
        let plus = dipole_at(cartesian_unit(j, e_step))?;
        let minus = dipole_at(cartesian_unit(j, -e_step))?;
        let d = (plus - minus) * (1.0 / (2.0 * e_step));
        let arr = d.to_array();
        for (i, &val) in arr.iter().enumerate() {
            alpha[i][j] = val;
        }
    }
    Ok(alpha)
}

/// Magnetic circular dichroism / Faraday tensor `d alpha_ij / d B_k` (atomic units),
/// the magnetic-field derivative of the electric polarizability, by central finite
/// difference in `B` of [`magnetic_polarizability`] about the field `b0` (usually
/// zero). Indexed `[k][i][j]`.
///
/// **Note (GFN1 monopole model):** in GFN1 the electric field couples only to the
/// Mulliken monopoles and `dq/dB = 0` by time reversal, so the monopole dipole has no
/// first-order `B` response and this tensor is **identically zero**. A nonzero orbital-
/// current MCD/Faraday rotation requires the length-gauge electric dipole (the LAO
/// dipole integrals, [`lao_dipole_matrix`]) rather than the point-charge dipole — the
/// same physics that makes the static optical-rotation `G`-tensor vanish. The routine
/// is the correct general `d alpha / d B` and would be nonzero for a dipole-coupled
/// model. `e_step`/`b_step` are the electric/magnetic finite-difference steps.
/// Non-periodic.
pub fn mcd_tensor(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    b0: Vec3,
    e_step: f64,
    b_step: f64,
) -> Result<[[[f64; 3]; 3]; 3]> {
    if !(b_step.is_finite() && b_step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "MCD magnetic-field step must be positive".to_string(),
        ));
    }
    let geom = magnetic_geom(system, params, options)?;
    let alpha_at = |b: Vec3| -> Result<[[f64; 3]; 3]> {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        magnetic_polarizability_with_geom(&geom, system, params, &opt, secondary, e_step)
    };
    let mut mcd = [[[0.0_f64; 3]; 3]; 3];
    for k in 0..3 {
        let ap = alpha_at(b0 + cartesian_unit(k, b_step))?;
        let am = alpha_at(b0 - cartesian_unit(k, b_step))?;
        for i in 0..3 {
            for j in 0..3 {
                mcd[k][i][j] = (ap[i][j] - am[i][j]) / (2.0 * b_step);
            }
        }
    }
    Ok(mcd)
}

/// Cotton-Mouton tensor `d^2 alpha_ij / d B_k^2` (atomic units), the second magnetic-
/// field derivative of the electric polarizability along each Cartesian field
/// direction, by central second difference in `B` of [`magnetic_polarizability`] about
/// `B = 0`. Indexed `[k][i][j]`. Even in `B`, so the symmetric (in `i,j`) part
/// survives; it drives the magnetic-field-induced birefringence. `e_step`/`b_step` are
/// the electric/magnetic finite-difference steps. Non-periodic.
pub fn cotton_mouton_tensor(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    e_step: f64,
    b_step: f64,
) -> Result<[[[f64; 3]; 3]; 3]> {
    if !(b_step.is_finite() && b_step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "Cotton-Mouton magnetic-field step must be positive".to_string(),
        ));
    }
    let geom = magnetic_geom(system, params, options)?;
    let alpha_at = |b: Vec3| -> Result<[[f64; 3]; 3]> {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        magnetic_polarizability_with_geom(&geom, system, params, &opt, secondary, e_step)
    };
    let a0 = alpha_at(Vec3::zero())?;
    let mut cm = [[[0.0_f64; 3]; 3]; 3];
    for k in 0..3 {
        let ap = alpha_at(cartesian_unit(k, b_step))?;
        let am = alpha_at(cartesian_unit(k, -b_step))?;
        for i in 0..3 {
            for j in 0..3 {
                cm[k][i][j] = (ap[i][j] - 2.0 * a0[i][j] + am[i][j]) / (b_step * b_step);
            }
        }
    }
    Ok(cm)
}

/// Result of a magnetic (GFN1-xTB-M0) nuclear gradient.
#[derive(Clone, Debug)]
pub struct MagneticGradientResult {
    /// Total magnetic (M0) energy at the input geometry (Hartree).
    pub energy: f64,
    /// `dE/dR` per atom (Hartree/Bohr).
    pub gradient: Vec<Vec3>,
    /// Forces (`-dE/dR`).
    pub forces: Vec<Vec3>,
}

/// Nuclear gradient of the closed-shell magnetic (GFN1-xTB-M0) SCC energy by
/// central finite difference (non-periodic).
///
/// The fully analytic magnetic gradient requires the complex London-phase
/// derivatives of `H0(B)`/`S(B)` (`d/dR [H0_munu exp(i theta_munu)]` with
/// `d theta_munu / dR` from the `1/2 B·((R_a-O)x(R_b-O))` phase) contracted with
/// the complex density and a complex energy-weighted (Pulay) density. That
/// machinery is the documented next step; this finite-difference gradient is the
/// exact derivative of the converged M0 energy to the displacement order and is
/// the working magnetic-field force. Each component re-runs the complex SCC, so
/// it costs `6N + 1` magnetic SCCs.
pub fn magnetic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    step: f64,
) -> Result<MagneticGradientResult> {
    let _profile = crate::profile::scope("magnetic.gradient.total");
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "magnetic gradient step must be positive".to_string(),
        ));
    }
    let base = run_magnetic_scc(system, params, options)?;
    let nat = system.atoms.len();
    let inv = 1.0 / (2.0 * step);
    let mut gradient = vec![Vec3::zero(); nat];
    for atom in 0..nat {
        for axis in 0..3 {
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
            let ep = run_magnetic_scc(&plus, params, options)?.energy;
            let em = run_magnetic_scc(&minus, params, options)?.energy;
            let d = (ep - em) * inv;
            match axis {
                0 => gradient[atom].x = d,
                1 => gradient[atom].y = d,
                _ => gradient[atom].z = d,
            }
        }
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(MagneticGradientResult {
        energy: base.energy,
        gradient,
        forces,
    })
}

/// SCC second/third-order Coulomb plus the classical repulsion / dispersion /
/// halogen energy at a geometry, with the shell charges held **fixed** at `q`. The
/// Hellmann-Feynman magnetic gradient differentiates this term directly; the charge
/// response is carried separately by the energy-weighted-density (Pulay) term.
fn scc_classical_energy_fixed_charges(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    q: &[f64],
) -> Result<f64> {
    let basis = BasisSet::build(
        system,
        params,
        BasisOptions {
            nprim: options.nprim,
        },
    )?;
    let shell_model = ShellChargeModel::build(system, &basis, params)?;
    let amat = effective_coulomb_matrix(system, &basis, &shell_model);
    let scc = coulomb_energy_potential_from_matrix(&basis, &shell_model, q, &amat)?;
    let rep = repulsion_energy(system, params)?;
    let disp = if options.enable_dispersion {
        dispersion_energy(system, params, options.d3_reference_path.as_deref())?
    } else {
        0.0
    };
    let hal = halogen_energy(system, params)?;
    Ok(scc.second_order + scc.third_order + rep + disp + hal)
}

/// Analytic (Hellmann-Feynman) nuclear gradient of the closed-shell magnetic
/// (GFN1-xTB-M0/M1) SCC energy. With the converged complex density `P`, energy-
/// weighted density `W` and shell charges `q` held fixed, the gradient is
///
/// ```text
/// dE/dR = Re Tr(P dH0(B)/dR) - Re Tr(W dS(B)/dR)        (band + Pulay)
///       + d/dR [ E_scc(q) + E_rep + E_disp + E_hal ].   (Coulomb + classical)
/// ```
///
/// Only the integral / Coulomb **builders** are differenced (central finite
/// difference of the matrices and the fixed-charge energy) — the self-consistent
/// energy is *not* re-converged at the displaced geometries, so this costs one
/// magnetic SCC plus cheap builder evaluations instead of the `6N+1` full SCCs of
/// [`magnetic_gradient`]. Reduces to the field-free analytic gradient at `B = 0`.
/// `secondary = Some(..)` selects the GFN1-xTB-M1 dual basis; `step` is the nuclear
/// finite-difference step for the integral derivatives. Non-periodic.
pub fn magnetic_analytic_gradient(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    step: f64,
) -> Result<MagneticGradientResult> {
    let _profile = crate::profile::scope("magnetic.analytic_gradient.total");
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(
            "magnetic gradient step must be positive".to_string(),
        ));
    }
    let base = run_magnetic_scc_inner(system, params, options, secondary)?;
    let p = &base.density;
    let w = &base.energy_weighted_density;
    let q = &base.shell_charges;
    let vao = &base.shell_potential_ao; // converged SCC potential per AO (the shift)
    let n = p.n;
    let nat = system.atoms.len();
    let inv = 1.0 / (2.0 * step);
    // Re Tr(M . D) for complex M = (re, im) and complex D supplied as (re, im).
    let re_trace = |m: &CMatrix, d_re: &Matrix, d_im: &Matrix| -> f64 {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                acc += m.re[(i, j)] * d_re[(j, i)] - m.im[(i, j)] * d_im[(j, i)];
            }
        }
        acc
    };
    let displaced = |atom: usize, axis: usize, s: f64| -> PeriodicSystem {
        let mut sys = system.clone();
        match axis {
            0 => sys.atoms[atom].position.x += s,
            1 => sys.atoms[atom].position.y += s,
            _ => sys.atoms[atom].position.z += s,
        }
        sys
    };
    // Each (atom, axis) DOF is an independent finite-difference of the LAO builder at a
    // displaced geometry (no `MagneticGeom` reuse since the geometry moves); evaluate
    // them in parallel (deterministic: results indexed by DOF, no cross-iteration sum).
    let per_dof = |atom: usize, axis: usize| -> Result<f64> {
        let plus = displaced(atom, axis, step);
        let minus = displaced(atom, axis, -step);
        // dH0(B)/dR, dS(B)/dR by central FD of the LAO matrix builder (this captures
        // the AO-centre movement, the GFN1 H0 prefactor `hij(R)` and the kinetic-energy
        // correction, all consistently).
        let (h0p, sp) = magnetic_h0_overlap(&plus, params, options, secondary)?;
        let (h0m, sm) = magnetic_h0_overlap(&minus, params, options, secondary)?;
        let mut dh0_re = Matrix::zeros(n, n);
        let mut dh0_im = Matrix::zeros(n, n);
        let mut ds_re = Matrix::zeros(n, n);
        let mut ds_im = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                dh0_re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) * inv;
                dh0_im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) * inv;
                ds_re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) * inv;
                ds_im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) * inv;
            }
        }
        let band_pulay = re_trace(p, &dh0_re, &dh0_im) - re_trace(w, &ds_re, &ds_im);
        // SCC shift-Pulay: the Mulliken charges depend on the LAO overlap, so dq/dR
        // couples to the SCC potential `V` (= vao). This is the
        // `-P . (1/2(V_mu+V_nu) dS/dR)` term (cf. the field-free gradient's
        // `-P*shift*dS'`); the explicit `1/2 q dgamma/dR q` is in the fixed-charge
        // energy difference below.
        let mut shift_pulay = 0.0;
        for i in 0..n {
            for j in 0..n {
                let wgt = 0.5 * (vao[i] + vao[j]);
                shift_pulay -= wgt * (p.re[(i, j)] * ds_re[(j, i)] - p.im[(i, j)] * ds_im[(j, i)]);
            }
        }
        let e_other_p = scc_classical_energy_fixed_charges(&plus, params, options, q)?;
        let e_other_m = scc_classical_energy_fixed_charges(&minus, params, options, q)?;
        Ok(band_pulay + shift_pulay + (e_other_p - e_other_m) * inv)
    };
    let vals: Vec<Result<f64>> = (0..nat * 3)
        .into_par_iter()
        .map(|idx| per_dof(idx / 3, idx % 3))
        .collect();
    let mut gradient = vec![Vec3::zero(); nat];
    for (idx, v) in vals.into_iter().enumerate() {
        let d = v?;
        let (atom, axis) = (idx / 3, idx % 3);
        match axis {
            0 => gradient[atom].x = d,
            1 => gradient[atom].y = d,
            _ => gradient[atom].z = d,
        }
    }
    let forces = gradient.iter().map(|g| -*g).collect::<Vec<_>>();
    Ok(MagneticGradientResult {
        energy: base.energy,
        gradient,
        forces,
    })
}

/// Speed of light in atomic units, i.e. the inverse fine-structure constant
/// `1/alpha = 137.035999177` (CODATA 2022; the 2018 recommendation was
/// `137.035999084`, a `7e-10` relative difference that is far below the accuracy of
/// any shielding computed here). The NMR shielding prefactor is
/// `alpha^2/2 = 1/(2 c^2)`.
pub const SPEED_OF_LIGHT_AU: f64 = 137.035_999_177;

/// NMR nuclear magnetic shielding tensor of one nucleus, `sigma_{ab} = d^2 E /
/// dB_a dm_b` (closed-shell, non-periodic), split into the diamagnetic
/// (ground-state expectation) and paramagnetic (CP-SCC magnetic response) parts.
/// Dimensionless atomic units; multiply by `1e6` for the chemist's ppm scale.
#[derive(Debug, Clone)]
pub struct NmrShielding {
    /// Index of the shielded nucleus in `system.atoms`.
    pub nucleus: usize,
    /// Full shielding tensor `sigma_{ab}` (row `a` = external-field axis, column `b`
    /// = nuclear-moment axis), atomic units.
    pub sigma: [[f64; 3]; 3],
    /// Diamagnetic part `sigma^dia_{ab} = (alpha^2/2) Tr(P0 . d_{ba})`, with `d_{ba}`
    /// the bare bracket `[delta_ab (r_O.r_A) - r_{A,a} r_{O,b}]/r_A^3`.
    pub diamagnetic: [[f64; 3]; 3],
    /// Paramagnetic part `sigma^para_{ab} = alpha^2 Tr((dP/dB_a) . L_{A,b}/r_A^3)`,
    /// evaluated as `(alpha^2/2) Tr(P^a . L_{A,b}/r_A^3)` with `P^a = 2 dP/dB_a`, the
    /// response to the *bare* angular momentum — see [`nmr_shielding_tensor`].
    pub paramagnetic: [[f64; 3]; 3],
}

impl NmrShielding {
    /// Isotropic shielding `sigma_iso = (1/3) Tr(sigma)` (atomic units; `x1e6` = ppm).
    pub fn isotropic(&self) -> f64 {
        (self.sigma[0][0] + self.sigma[1][1] + self.sigma[2][2]) / 3.0
    }
}

/// Analytic GFN1 NMR nuclear magnetic shielding tensor of nucleus `nucleus` with the
/// common gauge origin `gauge_origin` (CGO), `sigma_{A,ab} = d^2 E / dB_a dm_{A,b}`.
///
/// With `A = A_B + A_m`, `A_B = 1/2 B x r_O` (`O` = gauge origin) and
/// `A_m = alpha^2 (m x r_A)/r_A^3`, the exact derivatives of `H = 1/2 (p + A)^2` are
/// ```text
/// dH/dB_a      = 1/2 L_{O,a}                = 1/2 (-i)(r_O x grad)_a,
/// dH/dm_b      = alpha^2 L_{A,b}/r_A^3      = alpha^2 (-i)(r_A x grad)_b / r_A^3,
/// d2H/dB_a dm_b = (alpha^2/2) [delta_ab (r_O.r_A) - r_{A,a} r_{O,b}] / r_A^3.
/// ```
/// **Bookkeeping note:** the code below perturbs with the *bare* angular momentum
/// `(-i)(r_O x grad)_a`, i.e. `2 dH/dB_a`, so the density response `pa` it obtains is
/// `2 dP/dB_a`; the single shared prefactor `alpha^2/2` then lands the paramagnetic
/// term on `alpha^2 Tr(dP/dB_a . L_{A,b}/r_A^3)` — the physical value. The factor 2
/// is deliberate and cancels; do not "fix" it without re-deriving the prefactor.
/// (`crate::nmr::diamagnetic_operator_matrix` likewise returns the bracket without
/// its `alpha^2/2`.) Then
/// ```text
/// sigma_ab = (alpha^2/2) [ Tr(P0 . d_{ba})              (diamagnetic, no response)
///                        + Tr( (dP/dB_a) . L_{A,b}/r_A^3 ) ] (paramagnetic response)
/// ```
/// where `dP/dB_a` is the closed-shell CP-SCC density response to the orbital-Zeeman
/// perturbation ([`magnetic_first_order_response`], reused from the magnetizability),
/// purely imaginary/antisymmetric for the real reference, and `alpha = 1/c` is the
/// fine-structure constant. The bare-operator second derivative is validated against
/// the operator-injected magnetic-SCC energy (`tests::nmr_shielding_matches_energy_fd`);
/// the `alpha^2/2` prefactor is the standard (Ramsey) NMR conversion, anchored by the
/// diamagnetic term. Closed-shell, non-periodic. `secondary = Some(..)` selects the M1
/// dual basis for the kinetic-energy correction.
pub fn nmr_shielding_tensor(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    secondary: Option<&SecondaryBasis>,
    nucleus: usize,
    gauge_origin: Vec3,
) -> Result<NmrShielding> {
    let _profile = crate::profile::scope("magnetic.nmr_shielding.total");
    if nucleus >= system.atoms.len() {
        return Err(Gfn1Error::InvalidInput(format!(
            "nmr_shielding_tensor: nucleus index {nucleus} out of range ({} atoms)",
            system.atoms.len()
        )));
    }
    let with_field = |b: Vec3| -> ElectronicOptions {
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(b);
        opt
    };
    // Field-free reference (mirrors `magnetizability_diagonal_analytic`): converged
    // density `P0` / SCC potential `vao`, and the real generalized eigenbasis of the
    // converged Fock `F0 = H0 - vao(.)S`.
    let geom = magnetic_geom(system, params, options)?;
    let opt0 = with_field(Vec3::zero());
    let scc0 = run_magnetic_scc_with_geom(&geom, system, params, &opt0, secondary, None)?;
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = assemble_magnetic_matrices(
        system,
        &geom.basis,
        &geom.core,
        &opt0.external_field,
        secondary,
    );
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, options.eigen_tolerance)?;
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let p0 = &scc0.density;
    let mut f0c = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            f0c.re[(i, j)] = f0[(i, j)];
        }
    }

    // Magnetic-dipole AO operators (real). `lo[a]` = orbital angular momentum
    // (r_O x grad)_a about the gauge origin; `mop[b]` = nuclear paramagnetic operator
    // (r_A x grad)_b / r_A^3; `dmat[adir][bdir]` = diamagnetic operator
    // [delta(r_O.r_A) - r_{O,adir} r_{A,bdir}] / r_A^3.
    let nuc_pos = system.atoms[nucleus].position;
    let lo = angular_momentum_matrix(system, &geom.basis, gauge_origin);
    let mop = crate::nmr::paramagnetic_operator_matrix(system, &geom.basis, nuc_pos.to_array());
    let dmat = crate::nmr::diamagnetic_operator_matrix(
        system,
        &geom.basis,
        nuc_pos.to_array(),
        gauge_origin.to_array(),
    );

    let pref = 0.5 / (SPEED_OF_LIGHT_AU * SPEED_OF_LIGHT_AU);
    let mut diamagnetic = [[0.0_f64; 3]; 3];
    let mut paramagnetic = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        // Orbital-Zeeman perturbation `h0_a = dH/dB_a = -i L_{O,a}` (purely imaginary);
        // `s_a = 0` (the common gauge origin carries no London phase).
        let mut h0_a = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                h0_a.im[(i, j)] = -lo[a][(i, j)];
            }
        }
        let s_a = CMatrix::zeros(n);
        let (pa, _wa) =
            magnetic_first_order_response(n, nocc, c, &ct, eps, vao, p0, &f0c, &h0_a, &s_a);
        for b in 0..3 {
            // sigma^para_ab = Tr(P^a . (-i M_b)) = re_trace_cc(P^a, {re: 0, im: -M_b}).
            let mut mb = CMatrix::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    mb.im[(i, j)] = -mop[b][(i, j)];
                }
            }
            paramagnetic[a][b] = pref * re_trace_cc(&pa, &mb);
            // sigma^dia_ab = Tr(P0 . d[b][a]); P0 real-symmetric, the operator real.
            let cross = &dmat[b][a];
            let mut t = 0.0;
            for i in 0..n {
                for j in 0..n {
                    t += p0.re[(i, j)] * cross[(j, i)];
                }
            }
            diamagnetic[a][b] = pref * t;
        }
    }
    let mut sigma = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            sigma[a][b] = diamagnetic[a][b] + paramagnetic[a][b];
        }
    }
    Ok(NmrShielding {
        nucleus,
        sigma,
        diamagnetic,
        paramagnetic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // Cross-module items used by the (formerly integration) magnetic property tests.
    use crate::{
        analytic_gradient, parse_secondary_basis, run_electronic, tda_rotatory_strengths,
        AnalyticGradientOptions, AngularMomentum, TdaOptions, TdaSpin,
    };

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";

    fn load_params() -> Option<crate::params::Gfn1Parameters> {
        let params = crate::params::Gfn1Parameters::resolve(None)
            .expect("GFN1 parameter resolution failed");
        Some(params)
    }

    /// Load the GFN1-xTB-M1 secondary basis from the path in `GFN1_M1_BASIS` (the
    /// paper's `$Basis = GFN1-xTB-cc-pVDZ` file). Tests no-op when it is absent.
    fn load_m1_basis() -> Option<SecondaryBasis> {
        let path = std::env::var("GFN1_M1_BASIS").ok()?;
        let text = std::fs::read_to_string(path).ok()?;
        parse_secondary_basis(&text).ok()
    }

    fn opts_with_field(b: Vec3) -> ElectronicOptions {
        ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(b),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        }
    }

    /// End-to-end FD gate for the NMR shielding assembly: the analytic
    /// `nmr_shielding_tensor` (CP-SCC magnetic response + ground-state expectation)
    /// must equal `d^2 E / dB_a dm_b` of the magnetic SCC with the *full* CGO B-m
    /// coupling injected via the new `extra_h0` hook. This independently validates the
    /// paramagnetic response contraction, the diamagnetic operator, and their relative
    /// sign/scale; the `alpha^2/2` prefactor cancels (applied to both sides).
    #[test]
    fn nmr_shielding_matches_energy_fd() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        // Tight SCC: the mixed-derivative energy FD amplifies noise by 1/(4 h^2).
        let options = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            max_scc: 300,
            ..ElectronicOptions::default()
        };
        let nucleus = 0; // oxygen at the origin
        let gauge = system.atoms[nucleus].position; // common gauge origin at the nucleus
        let sh = nmr_shielding_tensor(&system, &params, &options, None, nucleus, gauge).unwrap();

        // Independent FD reference. extra_h0(B,m) = B_a(-i L_O,a) + m_b(-i M_b)
        // + B_a m_b d[b][a]; the energy second derivative is the bare (pre-prefactor)
        // shielding sigma_int = sigma / (alpha^2/2).
        let geom = magnetic_geom(&system, &params, &options).unwrap();
        let n = geom.basis.len();
        let nuc_pos = system.atoms[nucleus].position;
        let lo = angular_momentum_matrix(&system, &geom.basis, gauge);
        let mop =
            crate::nmr::paramagnetic_operator_matrix(&system, &geom.basis, nuc_pos.to_array());
        let dmat = crate::nmr::diamagnetic_operator_matrix(
            &system,
            &geom.basis,
            nuc_pos.to_array(),
            gauge.to_array(),
        );
        let mut opt = options.clone();
        opt.external_field.magnetic_field = Some(Vec3::zero()); // CGO entirely via extra_h0
        let energy = |a: usize, b: usize, db: f64, dm: f64| -> f64 {
            let mut extra = CMatrix::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    extra.im[(i, j)] = -db * lo[a][(i, j)] - dm * mop[b][(i, j)];
                    extra.re[(i, j)] = db * dm * dmat[b][a][(i, j)];
                }
            }
            run_magnetic_scc_with_geom(&geom, &system, &params, &opt, None, Some(&extra))
                .unwrap()
                .energy
        };
        let h = 6.0e-3;
        let pref = 0.5 / (SPEED_OF_LIGHT_AU * SPEED_OF_LIGHT_AU);
        let mut max_abs = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let fd = (energy(a, b, h, h) - energy(a, b, h, -h) - energy(a, b, -h, h)
                    + energy(a, b, -h, -h))
                    / (4.0 * h * h);
                let fd_phys = pref * fd;
                let analytic = sh.sigma[a][b];
                max_abs = max_abs.max((analytic - fd_phys).abs());
                eprintln!(
                    "sigma[{a}][{b}] analytic={analytic:.6e} fd={fd_phys:.6e} d={:.2e}",
                    analytic - fd_phys
                );
            }
        }
        eprintln!("NMR shielding: max |analytic - FD| = {max_abs:.3e}");
        // sigma ~ O(1e-4); a structural error (sign/factor/wrong operator) is O(1e-4+).
        assert!(
            max_abs < 5.0e-6,
            "NMR shielding analytic vs energy FD: max abs diff {max_abs:.3e}"
        );
    }

    /// Correlation of our (common-gauge-origin) NMR shielding against the published
    /// GFN1-xTB-M0 / M1 isotropic shieldings (Cheng & Wibowo-Teale, J. Chem. Theory
    /// Comput. 2023, 19, 6226; SI Table 3, ppm). Our direct `d^2E/dB dm` route with a
    /// common gauge origin is a *different* approximation than the paper's GIAO/London
    /// current-density + Biot-Savart formulation, so the absolute values differ
    /// (gauge-dependent, by ~1-2x, worst for 1H). This test therefore checks that the
    /// two methods are strongly *correlated* across the -420..+60 ppm range (and agree
    /// in sign), not that they match numerically. Standard geometries are used (the
    /// published ones are not reproduced exactly), so some scatter is expected.
    #[test]
    fn nmr_shielding_correlates_with_published_gfn1m() {
        let Some(params) = load_params() else {
            return;
        };
        let hf = "2\nHF\nF 0.0 0.0 0.0\nH 0.917 0.0 0.0\n";
        let co = "2\nCO\nC 0.0 0.0 0.0\nO 1.128 0.0 0.0\n";
        let n2 = "2\nN2\nN 0.0 0.0 0.0\nN 1.098 0.0 0.0\n";
        let ch4 = "5\nCH4\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\n\
                   H 0.6276 -0.6276 -0.6276\nH -0.6276 0.6276 -0.6276\n\
                   H -0.6276 -0.6276 0.6276\n";
        let nh3 = "4\nNH3\nN 0.0 0.0 0.1173\nH 0.0 0.9377 -0.2737\n\
                   H 0.8120 -0.4689 -0.2737\nH -0.8120 -0.4689 -0.2737\n";
        // (xyz, nucleus index, published GFN1-xTB-M0, published GFN1-xTB-M1) in ppm.
        let data: &[(&str, usize, f64, f64)] = &[
            (hf, 0, 61.51, -27.14),
            (hf, 1, 36.42, 32.50),
            (co, 0, -303.01, -208.37),
            (co, 1, -240.75, -333.07),
            (n2, 0, -422.92, -307.68),
            (WATER, 0, 33.35, -2.38),
            (WATER, 1, 30.25, 31.23),
            (ch4, 0, -22.93, -5.76),
            (ch4, 1, 30.76, 36.16),
            (nh3, 0, -34.96, -18.36),
            (nh3, 1, 29.51, 34.22),
        ];
        let opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-9,
            charge_tolerance: 1.0e-8,
            ..ElectronicOptions::default()
        };
        let (mut mine, mut m0, mut m1) = (Vec::new(), Vec::new(), Vec::new());
        for &(xyz, nuc, pm0, pm1) in data {
            let system = crate::system::PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let gauge = system.atoms[nuc].position;
            let sh = nmr_shielding_tensor(&system, &params, &opts, None, nuc, gauge).unwrap();
            let iso = sh.isotropic() * 1.0e6;
            eprintln!(
                "{:>4} nuc{nuc} mine={iso:8.2}  M0={pm0:8.2}  M1={pm1:8.2}",
                xyz.lines().nth(1).unwrap_or("")
            );
            mine.push(iso);
            m0.push(pm0);
            m1.push(pm1);
        }
        // Pearson R^2 and least-squares slope of `paper = slope * mine`.
        let stats = |x: &[f64], y: &[f64]| -> (f64, f64) {
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for k in 0..x.len() {
                sxy += (x[k] - mx) * (y[k] - my);
                sxx += (x[k] - mx).powi(2);
                syy += (y[k] - my).powi(2);
            }
            (sxy * sxy / (sxx * syy), sxy / sxx)
        };
        let (r2_m0, sl_m0) = stats(&mine, &m0);
        let (r2_m1, sl_m1) = stats(&mine, &m1);
        eprintln!("corr vs GFN1-xTB-M0: R^2={r2_m0:.4} slope={sl_m0:.3}");
        eprintln!("corr vs GFN1-xTB-M1: R^2={r2_m1:.4} slope={sl_m1:.3}");
        // Strong positive correlation with the published GFN1-xTB-M trends.
        assert!(
            r2_m0 > 0.9 && sl_m0 > 0.0,
            "weak/wrong correlation with published GFN1-xTB-M0: R^2={r2_m0:.3} slope={sl_m0:.3}"
        );
        assert!(
            r2_m1 > 0.8 && sl_m1 > 0.0,
            "weak/wrong correlation with published GFN1-xTB-M1: R^2={r2_m1:.3} slope={sl_m1:.3}"
        );
    }

    #[test]
    fn lao_overlap_reduces_to_real_at_zero_field_and_is_hermitian() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let n = basis.len();
        let positions: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        // B = 0: must reproduce the real AO overlap with vanishing imaginary part.
        let zero = ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        };
        let s0 = lao_overlap_matrix(&system, &basis, &zero);
        let (mut re_err, mut im0) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let ri = positions[basis.aos[i].atom_index];
            for j in 0..n {
                let rj = positions[basis.aos[j].atom_index];
                let s_real =
                    crate::integrals::contracted_pair(&basis.aos[i], &basis.aos[j], ri, rj).0;
                re_err = re_err.max((s0.re[(i, j)] - s_real).abs());
                im0 = im0.max(s0.im[(i, j)].abs());
            }
        }
        assert!(
            re_err < 1.0e-10,
            "LAO overlap re vs real overlap {re_err:.3e}"
        );
        assert!(im0 < 1.0e-12, "LAO overlap imaginary part at B=0 {im0:.3e}");
        // Finite field: Hermitian (re symmetric, im antisymmetric) and genuinely complex.
        let opts = ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(0.0, 0.0, 0.06)),
            ..ExternalFieldOptions::default()
        };
        let s = lao_overlap_matrix(&system, &basis, &opts);
        let (mut herm, mut im_max) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            for j in 0..n {
                herm = herm
                    .max((s.re[(i, j)] - s.re[(j, i)]).abs())
                    .max((s.im[(i, j)] + s.im[(j, i)]).abs());
                im_max = im_max.max(s.im[(i, j)].abs());
            }
        }
        assert!(herm < 1.0e-12, "LAO overlap not Hermitian: {herm:.3e}");
        assert!(
            im_max > 1.0e-6,
            "finite field produced no imaginary overlap"
        );
    }

    #[test]
    fn lao_kinetic_reduces_to_real_at_zero_field_and_is_hermitian() {
        let Some(params) = load_params() else {
            return;
        };
        let system = crate::system::PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let n = basis.len();
        let positions: Vec<Vec3> = system.atoms.iter().map(|a| a.position).collect();
        // B = 0: <1/2 pi^2> = <-1/2 nabla^2> (real kinetic), vanishing imaginary part.
        let zero = ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        };
        let k0 = lao_kinetic_matrix(&system, &basis, &zero);
        let (mut re_err, mut im0) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let ri = positions[basis.aos[i].atom_index];
            for j in 0..n {
                let rj = positions[basis.aos[j].atom_index];
                let t_real =
                    crate::integrals::contracted_kinetic(&basis.aos[i], &basis.aos[j], ri, rj);
                re_err = re_err.max((k0.re[(i, j)] - t_real).abs());
                im0 = im0.max(k0.im[(i, j)].abs());
            }
        }
        assert!(
            re_err < 1.0e-9,
            "LAO kinetic re vs real kinetic {re_err:.3e}"
        );
        assert!(im0 < 1.0e-12, "LAO kinetic imaginary part at B=0 {im0:.3e}");
        // Finite field with an off-centre gauge origin: Hermitian and genuinely complex.
        let opts = ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(0.0, 0.0, 0.06)),
            origin: Vec3::new(0.3, -0.2, 0.1),
            ..ExternalFieldOptions::default()
        };
        let k = lao_kinetic_matrix(&system, &basis, &opts);
        let (mut herm, mut im_max) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            for j in 0..n {
                herm = herm
                    .max((k.re[(i, j)] - k.re[(j, i)]).abs())
                    .max((k.im[(i, j)] + k.im[(j, i)]).abs());
                im_max = im_max.max(k.im[(i, j)].abs());
            }
        }
        assert!(herm < 1.0e-9, "LAO kinetic not Hermitian: {herm:.3e}");
        assert!(
            im_max > 1.0e-6,
            "finite field produced no imaginary kinetic term"
        );
    }

    #[test]
    fn london_phase_is_antisymmetric_and_zero_without_field() {
        let b = Vec3::new(0.0, 0.0, 0.3);
        let o = Vec3::zero();
        let ra = Vec3::new(1.0, 0.0, 0.0);
        let rb = Vec3::new(0.0, 1.0, 0.0);
        let ab = london_phase_angle(b, o, ra, rb);
        let ba = london_phase_angle(b, o, rb, ra);
        assert!((ab + ba).abs() < 1.0e-14, "phase must be antisymmetric");
        // B x: 1/2 * Bz * (ra x rb)_z = 1/2 * 0.3 * 1 = 0.15.
        assert!((ab - 0.15).abs() < 1.0e-12);
        assert!(london_phase_angle(Vec3::zero(), o, ra, rb).abs() < 1.0e-14);
    }

    #[test]
    fn collinear_centres_have_zero_phase() {
        let b = Vec3::new(0.1, 0.2, 0.3);
        let o = Vec3::zero();
        let ra = Vec3::new(1.0, 1.0, 1.0);
        let rb = Vec3::new(2.0, 2.0, 2.0); // parallel to ra
        assert!(london_phase_angle(b, o, ra, rb).abs() < 1.0e-14);
    }

    #[test]
    fn phase_factor_is_unit_modulus() {
        let (re, im) = london_phase_factor(
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );
        assert!((re * re + im * im - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn spin_zeeman_blocks_are_plus_minus_half_b_overlap() {
        let mut s = Matrix::zeros(2, 2);
        s[(0, 0)] = 1.0;
        s[(1, 1)] = 1.0;
        let opts = ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(0.0, 0.0, 0.4)),
            ..ExternalFieldOptions::default()
        };
        let (alpha, beta) = spin_zeeman_blocks(&s, &opts);
        assert!((alpha[(0, 0)] - 0.2).abs() < 1.0e-14);
        assert!((beta[(0, 0)] + 0.2).abs() < 1.0e-14);
    }

    #[test]
    fn magnetic_scc_reduces_to_field_free_and_responds_to_field() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

        // Field-free GFN1 reference (T = 0 internal energy: band + SCC + rep/disp/hal).
        let ref_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let e_ref = run_electronic(&system, &params, ref_opts)
            .unwrap()
            .total_internal;

        // B = 0 magnetic SCC must reproduce the field-free energy exactly.
        let m0 = run_magnetic_scc(&system, &params, &opts_with_field(Vec3::zero())).unwrap();
        assert!(m0.converged);
        assert!(
            (m0.energy - e_ref).abs() < 1.0e-8,
            "B=0 magnetic energy {} vs field-free {} (diff {:.3e})",
            m0.energy,
            e_ref,
            (m0.energy - e_ref).abs()
        );

        // A finite field perpendicular to the molecular plane: real, finite, converged,
        // and with a measurable effect on the energy. (The sign of the M0 magnetic
        // response is not guaranteed without the kinetic-energy correction of the
        // dual-basis M1 variant, so only the magnitude is checked here.)
        let mb = run_magnetic_scc(
            &system,
            &params,
            &opts_with_field(Vec3::new(0.0, 0.0, 0.05)),
        )
        .unwrap();
        assert!(mb.converged && mb.energy.is_finite());
        assert!(
            (mb.energy - m0.energy).abs() > 1.0e-7,
            "the magnetic field has no effect on the energy: B {} vs 0 {}",
            mb.energy,
            m0.energy
        );

        // The field couples through the gauge-origin-dependent London phase but the
        // energy is gauge-origin invariant for a neutral closed shell: shifting the
        // origin must not change the energy.
        let shifted = ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(0.0, 0.0, 0.05)),
            origin: Vec3::new(1.3, -0.7, 0.4),
            ..ExternalFieldOptions::default()
        };
        let opts_shift = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            external_field: shifted,
            ..ElectronicOptions::default()
        };
        let ms = run_magnetic_scc(&system, &params, &opts_shift).unwrap();
        assert!(
            (ms.energy - mb.energy).abs() < 1.0e-7,
            "magnetic energy is not gauge-origin invariant: {} vs {}",
            ms.energy,
            mb.energy
        );
    }

    #[test]
    fn m1_reduces_to_field_free_differs_from_m0_and_is_gauge_invariant() {
        let Some(params) = load_params() else {
            return;
        };
        let Some(secondary) = load_m1_basis() else {
            return; // needs GFN1_M1_BASIS
        };
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();

        // B = 0: M1 reproduces the field-free GFN1 energy (the KE correction vanishes).
        let ref_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let e_ref = run_electronic(&system, &params, ref_opts)
            .unwrap()
            .total_internal;
        let m1_0 =
            run_magnetic_scc_m1(&system, &params, &opts_with_field(Vec3::zero()), &secondary)
                .unwrap();
        assert!(
            (m1_0.energy - e_ref).abs() < 1.0e-8,
            "M1 B=0 energy {} vs field-free {}",
            m1_0.energy,
            e_ref
        );

        // Finite field: M1 must differ from M0 (the secondary basis changes the KE term).
        let bz = Vec3::new(0.0, 0.0, 0.08);
        let m0_b = run_magnetic_scc(&system, &params, &opts_with_field(bz)).unwrap();
        let m1_b = run_magnetic_scc_m1(&system, &params, &opts_with_field(bz), &secondary).unwrap();
        assert!(
            (m1_b.energy - m0_b.energy).abs() > 1.0e-7,
            "M1 energy equals M0 — secondary basis appears inactive ({} vs {})",
            m1_b.energy,
            m0_b.energy
        );

        // Gauge-origin invariance of the M1 energy.
        let shifted = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(bz),
                origin: Vec3::new(1.3, -0.7, 0.4),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        };
        let m1_s = run_magnetic_scc_m1(&system, &params, &shifted, &secondary).unwrap();
        assert!(
            (m1_s.energy - m1_b.energy).abs() < 1.0e-7,
            "M1 energy is not gauge-origin invariant: {} vs {}",
            m1_s.energy,
            m1_b.energy
        );

        // Isotropic magnetizabilities (printed for comparison with the paper / literature).
        let base = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(Vec3::zero()),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        };
        let xi_m0 = magnetizability_isotropic(&system, &params, &base, None, 0.02).unwrap();
        let xi_m1 =
            magnetizability_isotropic(&system, &params, &base, Some(&secondary), 0.02).unwrap();
        eprintln!(
            "H2O isotropic magnetizability (10^-30 J/T^2): M0 = {:.3}, M1 = {:.3}",
            xi_m0 * MAGNETIZABILITY_AU_TO_SI,
            xi_m1 * MAGNETIZABILITY_AU_TO_SI
        );
        assert!(xi_m0.is_finite() && xi_m1.is_finite());
        assert!(
            (xi_m1 - xi_m0).abs() > 1.0e-12,
            "M1 magnetizability equals M0"
        );
    }

    /// The full magnetizability **tensor** must be symmetric, have a diagonal matching
    /// [`magnetizability_diagonal_analytic`], and reproduce the off-diagonal mixed energy
    /// second derivative `-d^2E/dB_a dB_b`. Uses a low-symmetry geometry so the off-
    /// diagonals are nonzero.
    #[test]
    fn analytic_magnetizability_tensor_matches_finite_field() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.62 0.51 0.30\nH -0.51 0.59 0.22\n",
            0.0,
            false,
        )
        .unwrap();
        let base = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(Vec3::zero()),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        };
        let step = 0.004;
        let xi = magnetizability_tensor_analytic(&system, &params, &base, None, step).unwrap();
        // Symmetric.
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (xi[a][b] - xi[b][a]).abs() < 1.0e-10,
                    "tensor not symmetric"
                );
            }
        }
        // Diagonal consistent with the dedicated diagonal routine.
        let diag = magnetizability_diagonal_analytic(&system, &params, &base, None, step).unwrap();
        for a in 0..3 {
            assert!(
                (xi[a][a] - diag[a]).abs() < 1.0e-9,
                "tensor diagonal {} != diagonal routine {}",
                xi[a][a],
                diag[a]
            );
        }
        // Off-diagonal xy against the mixed energy finite difference -d^2E/dBx dBy.
        let energy = |b: Vec3| -> f64 {
            let mut o = base.clone();
            o.external_field.magnetic_field = Some(b);
            run_magnetic_scc(&system, &params, &o).unwrap().energy
        };
        let f = |sa: f64, sb: f64| Vec3::new(sa * step, sb * step, 0.0);
        let d2 = (energy(f(1.0, 1.0)) - energy(f(1.0, -1.0)) - energy(f(-1.0, 1.0))
            + energy(f(-1.0, -1.0)))
            / (4.0 * step * step);
        let xy_fd = -d2;
        eprintln!(
            "tilted-water xi_xy: analytic = {:.5}, FD = {:.5} (atomic units)",
            xi[0][1], xy_fd
        );
        assert!(
            (xi[0][1] - xy_fd).abs() < 5.0e-3 * xy_fd.abs().max(0.01),
            "xi_xy analytic {} != finite-field {}",
            xi[0][1],
            xy_fd
        );
    }

    /// The analytic McWeeny density-matrix CP-SCC magnetizability must reproduce the
    /// finite-field energy second derivative (`magnetizability_isotropic`) for both a
    /// polar lone-pair molecule (water) and a degenerate-orbital one (methane), to high
    /// accuracy — the analytic response is exact; only the LAO integral derivatives are
    /// differenced. Validates the diamagnetic + paramagnetic + charge-overlap terms.
    #[test]
    fn analytic_magnetizability_matches_finite_field() {
        let Some(params) = load_params() else {
            return;
        };
        let base = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(Vec3::zero()),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        };
        let cases = [
            ("water", WATER),
            (
                "methane",
                "5\nCH4\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH 0.6276 -0.6276 -0.6276\nH -0.6276 0.6276 -0.6276\nH -0.6276 -0.6276 0.6276\n",
            ),
        ];
        for (name, xyz) in cases {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let xi_fd = magnetizability_isotropic(&system, &params, &base, None, 0.004).unwrap();
            let xi_an =
                magnetizability_isotropic_analytic(&system, &params, &base, None, 0.004).unwrap();
            // The diagonal must also be consistent (isotropic = mean of the diagonal).
            let diag =
                magnetizability_diagonal_analytic(&system, &params, &base, None, 0.004).unwrap();
            let iso_from_diag = (diag[0] + diag[1] + diag[2]) / 3.0;
            eprintln!(
                "{name} isotropic magnetizability (10^-30 J/T^2): analytic = {:.4}, FD = {:.4}",
                xi_an * MAGNETIZABILITY_AU_TO_SI,
                xi_fd * MAGNETIZABILITY_AU_TO_SI
            );
            assert!(
                (xi_an - iso_from_diag).abs() < 1.0e-12,
                "{name}: isotropic helper disagrees with the diagonal mean"
            );
            assert!(
                (xi_an - xi_fd).abs() < 1.0e-3 * xi_fd.abs().max(1.0e-3),
                "{name}: analytic magnetizability {} != finite-field {} (atomic units)",
                xi_an,
                xi_fd
            );
        }
    }

    #[test]
    fn magnetic_gradient_matches_field_free_at_zero_and_responds_to_field() {
        let Some(params) = load_params() else {
            return;
        };
        // Off-equilibrium water so the gradient is non-trivial.
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
            0.0,
            false,
        )
        .unwrap();
        let step = 1.0e-3;

        // At B = 0 the magnetic (M0) gradient is the finite difference of the
        // field-free internal energy and must reproduce the field-free analytic
        // nuclear gradient.
        let g0 = magnetic_gradient(&system, &params, &opts_with_field(Vec3::zero()), step).unwrap();
        let ref_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let ana = analytic_gradient(
            &system,
            &params,
            AnalyticGradientOptions {
                electronic: ref_opts,
                ..AnalyticGradientOptions::default()
            },
        )
        .unwrap();
        let mut max_diff = 0.0_f64;
        for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
            max_diff = max_diff
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
        assert!(
            max_diff < 5.0e-5,
            "B=0 magnetic gradient vs field-free analytic gradient: max diff {max_diff:.3e}"
        );

        // forces = -gradient.
        for (g, f) in g0.gradient.iter().zip(g0.forces.iter()) {
            assert!((g.x + f.x).abs() < 1.0e-14 && (g.y + f.y).abs() < 1.0e-14);
        }

        // A finite field changes the forces (real, finite) and the gradient remains
        // step-consistent between two finite-difference steps.
        let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
        let gb1 = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
        let gb2 = magnetic_gradient(&system, &params, &field, 2.0e-3).unwrap();
        let mut step_diff = 0.0_f64;
        let mut field_diff = 0.0_f64;
        for ((a, b), z) in gb1
            .gradient
            .iter()
            .zip(gb2.gradient.iter())
            .zip(g0.gradient.iter())
        {
            assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite());
            step_diff = step_diff
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
            field_diff = field_diff
                .max((a.x - z.x).abs())
                .max((a.y - z.y).abs())
                .max((a.z - z.z).abs());
        }
        assert!(
            step_diff < 1.0e-4,
            "magnetic gradient not step-consistent: {step_diff:.3e}"
        );
        assert!(
            field_diff > 1.0e-7,
            "the field did not change the forces: {field_diff:.3e}"
        );
    }

    #[test]
    fn magnetic_analytic_gradient_matches_field_free_and_finite_difference() {
        let Some(params) = load_params() else {
            return;
        };
        // Off-equilibrium water so the gradient is non-trivial.
        let system = PeriodicSystem::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.05\nH 0.79 0.59 0.0\nH -0.74 0.58 0.0\n",
            0.0,
            false,
        )
        .unwrap();

        // At B = 0 the analytic (Hellmann-Feynman) magnetic gradient must reproduce the
        // field-free GFN1 analytic nuclear gradient.
        let g0 = magnetic_analytic_gradient(
            &system,
            &params,
            &opts_with_field(Vec3::zero()),
            None,
            1.0e-3,
        )
        .unwrap();
        let ref_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let ana = analytic_gradient(
            &system,
            &params,
            AnalyticGradientOptions {
                electronic: ref_opts,
                ..AnalyticGradientOptions::default()
            },
        )
        .unwrap();
        let mut max_b0 = 0.0_f64;
        for (a, b) in g0.gradient.iter().zip(ana.gradient.iter()) {
            max_b0 = max_b0
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
        assert!(
            max_b0 < 5.0e-5,
            "B=0 analytic magnetic gradient vs field-free analytic gradient: {max_b0:.3e}"
        );

        // At a finite field the analytic gradient must match the finite-difference of the
        // converged magnetic energy ([`magnetic_gradient`]) — the key correctness check
        // of the Hellmann-Feynman density/Pulay contraction.
        let field = opts_with_field(Vec3::new(0.0, 0.0, 0.08));
        let ga = magnetic_analytic_gradient(&system, &params, &field, None, 1.0e-3).unwrap();
        let gfd = magnetic_gradient(&system, &params, &field, 1.0e-3).unwrap();
        let mut max_fd = 0.0_f64;
        for (a, b) in ga.gradient.iter().zip(gfd.gradient.iter()) {
            max_fd = max_fd
                .max((a.x - b.x).abs())
                .max((a.y - b.y).abs())
                .max((a.z - b.z).abs());
        }
        assert!(
            max_fd < 1.0e-4,
            "analytic vs finite-difference magnetic gradient at finite B: {max_fd:.3e}"
        );

        // forces = -gradient.
        for (g, f) in ga.gradient.iter().zip(ga.forces.iter()) {
            assert!((g.x + f.x).abs() < 1.0e-14 && (g.z + f.z).abs() < 1.0e-14);
        }
    }

    #[test]
    fn angular_momentum_is_antisymmetric_and_matches_p_orbital_value() {
        let Some(params) = load_params() else {
            return;
        };
        // Single carbon atom at the gauge origin: the orbital angular momentum about
        // its own centre has the exact p-shell value L_z p_y = -i p_x, so the real
        // coefficient c_z[px,py] = <px|px> (and cyclic).
        let system = PeriodicSystem::from_xyz_str("1\nC\nC 0.0 0.0 0.0\n", 0.0, false).unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let n = basis.len();
        let zero = ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        };
        let s = lao_overlap_matrix(&system, &basis, &zero).re;
        let l = angular_momentum_matrix(&system, &basis, Vec3::zero());

        // L_a are antisymmetric with zero diagonal (since <mu|L|nu> = -i c is imaginary).
        let (mut asym, mut diag) = (0.0_f64, 0.0_f64);
        for axis in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    asym = asym.max((l[axis][(i, j)] + l[axis][(j, i)]).abs());
                    if i == j {
                        diag = diag.max(l[axis][(i, i)].abs());
                    }
                }
            }
        }
        assert!(
            asym < 1.0e-10,
            "angular momentum not antisymmetric: {asym:.3e}"
        );
        assert!(
            diag < 1.0e-10,
            "angular momentum diagonal not zero: {diag:.3e}"
        );

        // Locate the Cartesian p AOs by their single component power.
        let find_p = |px: usize, py: usize, pz: usize| -> usize {
            basis
                .aos
                .iter()
                .position(|a| {
                    a.angular == AngularMomentum::P
                        && a.components.len() == 1
                        && a.components[0].power.x == px
                        && a.components[0].power.y == py
                        && a.components[0].power.z == pz
                })
                .expect("carbon 2p AO not found")
        };
        let (ipx, ipy, ipz) = (find_p(1, 0, 0), find_p(0, 1, 0), find_p(0, 0, 1));
        // c_z[px,py] = <px|px>, c_x[py,pz] = <py|py>, c_y[pz,px] = <pz|pz>.
        assert!(
            (l[2][(ipx, ipy)] - s[(ipx, ipx)]).abs() < 1.0e-9,
            "L_z[px,py] {} vs <px|px> {}",
            l[2][(ipx, ipy)],
            s[(ipx, ipx)]
        );
        assert!(
            (l[0][(ipy, ipz)] - s[(ipy, ipy)]).abs() < 1.0e-9,
            "L_x[py,pz] {} vs <py|py> {}",
            l[0][(ipy, ipz)],
            s[(ipy, ipy)]
        );
        assert!(
            (l[1][(ipz, ipx)] - s[(ipz, ipz)]).abs() < 1.0e-9,
            "L_y[pz,px] {} vs <pz|pz> {}",
            l[1][(ipz, ipx)],
            s[(ipz, ipz)]
        );
    }

    #[test]
    fn tda_rotatory_strengths_vanish_for_achiral_water() {
        let Some(params) = load_params() else {
            return;
        };
        // Water is achiral (C2v) -> every electronic-CD rotatory strength must vanish,
        // even though the electric transition dipoles are nonzero.
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let ref_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-10,
            charge_tolerance: 1.0e-9,
            ..ElectronicOptions::default()
        };
        let electronic = run_electronic(&system, &params, ref_opts).unwrap();
        let states = tda_rotatory_strengths(
            &system,
            &params,
            &electronic,
            TdaOptions {
                n_states: 6,
                spin: TdaSpin::Singlet,
            },
            Vec3::zero(),
        )
        .unwrap();
        assert!(!states.is_empty());
        let max_r = states
            .iter()
            .fold(0.0_f64, |m, s| m.max(s.rotatory_strength.abs()));
        assert!(
            max_r < 1.0e-8,
            "achiral water has a non-vanishing rotatory strength: {max_r:.3e}"
        );
    }

    #[test]
    fn lao_dipole_reduces_to_real_and_shifts_with_origin() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
        let n = basis.len();
        // B = 0: the dipole integral matrices are real and symmetric.
        let zero = ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        };
        let d0 = lao_dipole_matrix(&system, &basis, &zero);
        let s = lao_overlap_matrix(&system, &basis, &zero).re;
        let (mut im0, mut sym) = (0.0_f64, 0.0_f64);
        for dc in d0.iter() {
            for i in 0..n {
                for j in 0..n {
                    im0 = im0.max(dc.im[(i, j)].abs());
                    sym = sym.max((dc.re[(i, j)] - dc.re[(j, i)]).abs());
                }
            }
        }
        assert!(im0 < 1.0e-12, "B=0 dipole has imaginary part {im0:.3e}");
        assert!(sym < 1.0e-10, "B=0 dipole not symmetric {sym:.3e}");

        // Shifting the gauge/dipole origin by `o` shifts D_c by o_c * S (at B = 0):
        // <mu|(x-0)|nu> - <mu|(x-o)|nu> = o * <mu|nu>.
        let o = Vec3::new(0.37, -0.21, 0.44);
        let opts_o = ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            origin: o,
            ..ExternalFieldOptions::default()
        };
        let d_o = lao_dipole_matrix(&system, &basis, &opts_o);
        let mut shift_err = 0.0_f64;
        for (c, (dc0, dco)) in d0.iter().zip(d_o.iter()).enumerate() {
            let oc = match c {
                0 => o.x,
                1 => o.y,
                _ => o.z,
            };
            for i in 0..n {
                for j in 0..n {
                    let expected = oc * s[(i, j)];
                    shift_err = shift_err.max((dc0.re[(i, j)] - dco.re[(i, j)] - expected).abs());
                }
            }
        }
        assert!(
            shift_err < 1.0e-9,
            "dipole origin shift inconsistent: {shift_err:.3e}"
        );

        // Finite field: Hermitian (re symmetric, im antisymmetric) and genuinely complex.
        let opts_b = ExternalFieldOptions {
            magnetic_field: Some(Vec3::new(0.0, 0.0, 0.07)),
            ..ExternalFieldOptions::default()
        };
        let db = lao_dipole_matrix(&system, &basis, &opts_b);
        let (mut herm, mut im_max) = (0.0_f64, 0.0_f64);
        for dc in db.iter() {
            for i in 0..n {
                for j in 0..n {
                    herm = herm
                        .max((dc.re[(i, j)] - dc.re[(j, i)]).abs())
                        .max((dc.im[(i, j)] + dc.im[(j, i)]).abs());
                    im_max = im_max.max(dc.im[(i, j)].abs());
                }
            }
        }
        assert!(
            herm < 1.0e-10,
            "dipole not Hermitian at finite field: {herm:.3e}"
        );
        assert!(im_max > 1.0e-6, "finite field produced no imaginary dipole");
    }

    /// The combined electric+magnetic SCC polarizability `alpha(B) = dmu/dE` reduces to
    /// the field-free GFN1 polarizability at `B = 0`; its magnetic-field derivatives
    /// obey the Onsager symmetry: `dalpha_ij/dB` is antisymmetric in `(i,j)` (MCD) and
    /// `d2alpha_ij/dB2` is symmetric in `(i,j)` (Cotton-Mouton).
    #[test]
    fn magnetic_polarizability_reduces_and_mcd_cotton_mouton_symmetry() {
        let Some(params) = load_params() else {
            return;
        };
        let system = PeriodicSystem::from_xyz_str(WATER, 0.0, false).unwrap();
        let base = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            external_field: ExternalFieldOptions {
                magnetic_field: Some(Vec3::zero()),
                ..ExternalFieldOptions::default()
            },
            ..ElectronicOptions::default()
        };
        let e_step = 0.002;

        // alpha(B=0) must match the field-free GFN1 monopole polarizability.
        let mut alpha0 = magnetic_polarizability(&system, &params, &base, None, e_step).unwrap();
        for i in 0..3 {
            for j in (i + 1)..3 {
                let m = 0.5 * (alpha0[i][j] + alpha0[j][i]);
                alpha0[i][j] = m;
                alpha0[j][i] = m;
            }
        }
        let ff_opts = ElectronicOptions {
            electronic_temperature: 0.0,
            energy_tolerance: 1.0e-11,
            charge_tolerance: 1.0e-10,
            ..ElectronicOptions::default()
        };
        let ff = crate::properties::static_polarizability_finite_field(
            &system, &params, &ff_opts, e_step,
        )
        .unwrap();
        let mut max_diff = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                max_diff = max_diff.max((alpha0[i][j] - ff.tensor[i][j]).abs());
            }
        }
        assert!(
            max_diff < 1.0e-6,
            "alpha(B=0) != field-free polarizability: max diff {max_diff:.3e}"
        );

        // MCD = d alpha/d B vanishes identically in the GFN1 monopole electric model:
        // dq/dB = 0 (time reversal) => the Mulliken dipole has no first-order B response,
        // so alpha(B) is even in B. (A nonzero orbital-current MCD needs the length-gauge
        // LAO dipole; see `lao_dipole_matrix`.)
        let mcd = mcd_tensor(&system, &params, &base, None, Vec3::zero(), e_step, 0.01).unwrap();
        let max_mag = mcd
            .iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        eprintln!("MCD  max|dalpha/dB| = {max_mag:.3e} (monopole model: identically zero)");
        assert!(
            max_mag < 1.0e-7,
            "monopole MCD should vanish (dq/dB=0): {max_mag:.3e}"
        );

        // Cotton-Mouton = d^2 alpha/d B^2: symmetric in (i,j), even in B, finite.
        let cm = cotton_mouton_tensor(&system, &params, &base, None, e_step, 0.02).unwrap();
        let (mut max_cm, mut max_asym) = (0.0_f64, 0.0_f64);
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(cm[k][i][j].is_finite());
                    max_cm = max_cm.max(cm[k][i][j].abs());
                    max_asym = max_asym.max((cm[k][i][j] - cm[k][j][i]).abs());
                }
            }
        }
        eprintln!(
            "CM   max|d2alpha/dB2| = {max_cm:.3e}, residual antisymmetric part = {max_asym:.3e}"
        );
        assert!(
            max_asym < 0.02 * max_cm.max(1.0e-4),
            "Cotton-Mouton tensor not symmetric in (i,j): asym {max_asym:.3e} vs mag {max_cm:.3e}"
        );
    }
}
