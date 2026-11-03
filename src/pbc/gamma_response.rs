//! **Gamma-point response fields via the molecular charge-space solver.**
//!
//! The molecular [`ChargeSpaceContext`] machinery (first/second/third-order
//! CP-SCC response) is representation-generic: every solver step reads only
//! `{basis, S, P₀, C, ε, f, kernel, χ⁰/LU, onsite E'''/E''''}` plus the
//! per-perturbation inputs (skeleton Fock/overlap derivatives and geometric
//! γ-derivative vectors). At the Gamma point of a periodic system all of the
//! reference objects are real matrices, and the periodicity enters ONLY
//! through the inputs — the image-summed integrals inside the skeletons and
//! the Ewald-split periodic γ inside the response kernel. Building the
//! context from the Bloch Γ reference with
//! [`crate::pbc::hessian::periodic_response_kernel`] injected therefore
//! reuses the entire validated molecular solver stack for the analytic
//! periodic third derivative.
//!
//! The result is the canonical free-energy derivative at the reference's
//! electronic temperature and fixed electron number. Thermal density and
//! chemical-potential responses use matrix-function divided differences,
//! including confluent limits at exactly degenerate fractional occupations.
//!
//! Gate: at integer filling the context's first-order field agrees with the independent
//! occ-virt PCG route ([`crate::pbc::hessian::gamma_cpxtb_response_directional`]),
//! which is itself pinned against reconverged-SCC finite differences.

use crate::coordination::CoordinationDerivatives;
use crate::error::{Gfn1Error, Result};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::params::Gfn1Parameters;
use crate::pbc::gamma_third::FrozenDensityImages;
use crate::pbc::hessian::{
    periodic_response_kernel, GammaMos, GammaSkeletonDerivatives, ResponseBandPair,
};
use crate::pbc::scf::PbcSccResult;
use crate::response::charge_space::ChargeSpaceContext;
use crate::system::PeriodicSystem;
use crate::third_derivative::SymmetricThird;

/// Build a [`ChargeSpaceContext`] on the Gamma-point Bloch reference.
///
/// Requirements: a Γ-only k-mesh (the reference matrices must be real) and a
/// converged periodic SCC. `charge_order` is the SCC charge-expansion order
/// the reference was run with (`ElectronicOptions::charge_order`).
pub(crate) fn gamma_charge_space_context(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    scf: &PbcSccResult,
    mos: &GammaMos,
    charge_order: usize,
) -> Result<ChargeSpaceContext> {
    if scf.kpoints.len() != 1 {
        return Err(Gfn1Error::InvalidInput(format!(
            "gamma_charge_space_context requires a Gamma-only mesh (got {} k-points)",
            scf.kpoints.len()
        )));
    }
    let n = scf.basis.len();
    let mut density0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            density0[(i, j)] = scf.density_k[0].re[(i, j)];
        }
    }
    let kernel = periodic_response_kernel(scf);
    ChargeSpaceContext::from_raw_parts(
        system,
        params,
        scf.basis.clone(),
        mos.overlap.clone(),
        density0,
        mos.coeff.clone(),
        mos.energies.clone(),
        mos.occupations.clone(),
        scf.electronic_temperature,
        &scf.atomic_charges,
        charge_order,
        kernel,
    )
}

/// **Shared, direction-INDEPENDENT reference state** for the analytic
/// Gamma-point third derivative: one periodic SCC, one Gamma MO solve, one
/// skeleton-derivative build, one charge-space factorization, plus the
/// real-space ground densities, the response band-pair table, the screening
/// kernel and the coordination derivatives.
///
/// Every one of these depends only on the reference geometry, so the dense /
/// block polarization drivers pay for them ONCE and reuse them across all
/// ~`C(n+2,3)` directions. What genuinely cannot be hoisted is listed on
/// [`pbc_gamma_third_with_reference`].
pub struct GammaThirdReference {
    scf: PbcSccResult,
    mos: GammaMos,
    sk: GammaSkeletonDerivatives,
    ctx: ChargeSpaceContext,
    dens0: FrozenDensityImages,
    band_pairs: Vec<ResponseBandPair>,
    kernel: Matrix,
    cn: Option<CoordinationDerivatives>,
    lattice: Lattice,
    /// Derived caches (basis size, atom count, DOF count).
    n: usize,
    nat: usize,
    ndof: usize,
}

impl GammaThirdReference {
    /// Build the shared reference, rejecting every option set the analytic
    /// Gamma assembly does not cover.
    ///
    /// Guards, in cheap-to-expensive order: the term registry at analytic
    /// order 3 (which is what rejects multipole / long-range exchange /
    /// DFT+U / spin polarization / external fields / experimental D4, all
    /// capped at order 1), a Gamma-only k-mesh, a lattice, and SCC convergence.
    /// The converged reference's canonical occupations may be integer or
    /// fractional; both use the same analytic response assembly.
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &crate::electronic::ElectronicOptions,
        pbc: &crate::pbc::PbcOptions,
    ) -> Result<Self> {
        use crate::pbc::gamma_third::gamma_realspace_densities;
        use crate::pbc::hessian::{
            build_response_band_pairs, gamma_mos, gamma_skeleton_derivatives,
        };

        crate::terms::require_order(
            options,
            params,
            3,
            "the analytic Gamma-point periodic third derivative",
        )?;
        if !pbc.kmesh.is_gamma_only() {
            return Err(Gfn1Error::InvalidInput(format!(
                "the analytic Gamma-point periodic third derivative requires a Gamma-only \
                 k-mesh (got a {:?} Monkhorst-Pack grid); use \
                 pbc_kpoint_third_derivative_seminumerical_* for k-point sampling",
                pbc.kmesh.size
            )));
        }
        let lattice = *system
            .lattice
            .as_ref()
            .ok_or_else(|| Gfn1Error::InvalidInput("periodic third needs a lattice".into()))?;
        let scf = crate::pbc::scf::run_pbc_scc(system, params, options, pbc)?;
        crate::pbc::ml::require_cubic_response(&scf)?;
        if !scf.converged {
            return Err(Gfn1Error::InvalidInput(
                "periodic SCC did not converge for the analytic third derivative".into(),
            ));
        }
        let mos = gamma_mos(&scf, scf.nelec)?;
        let sk = gamma_skeleton_derivatives(system, params, &scf, options, pbc)?;
        let ctx = gamma_charge_space_context(system, params, &scf, &mos, options.charge_order)?;
        let dens0 = gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff);
        let band_pairs = build_response_band_pairs(system, params, &scf, &dens0.p, pbc)?;
        let kernel = periodic_response_kernel(&scf);
        let cn = if options.hamiltonian.enable_cn_hamiltonian {
            Some(crate::coordination::coordination_with_derivatives(
                system,
                crate::coordination::CoordinationOptions {
                    cutoff: options.hamiltonian.coordination_cutoff,
                    ..crate::coordination::CoordinationOptions::default()
                },
            )?)
        } else {
            None
        };
        let n = scf.basis.len();
        let nat = system.atoms.len();
        Ok(Self {
            scf,
            mos,
            sk,
            ctx,
            dens0,
            band_pairs,
            kernel,
            cn,
            lattice,
            n,
            nat,
            ndof: 3 * nat,
        })
    }

    /// The converged periodic SCC the reference was built on.
    pub fn scc(&self) -> &PbcSccResult {
        &self.scf
    }

    /// The Gamma-point MOs (coefficients, energies, occupations, overlap).
    pub fn mos(&self) -> &GammaMos {
        &self.mos
    }
}

/// **Analytic Gamma-point directional third derivative** `e³[v]` against a
/// shared [`GammaThirdReference`], assembled as
///
/// ```text
///   e³[v] = frozen third                      (GammaFrozenThird, all blocks)
///         + density path                      (∂frozen²/∂X₀ · X¹)
///         + g(X²)·v                           (response gradient, second-order slots)
///         + B6(X¹)[v,v] + background_complete (gamma_response_path_directional)
/// ```
///
/// with `X¹` from the directional first-order charge-space solve and `X²`
/// from the molecular second-order solver on the Gamma context. Gated against
/// [`crate::pbc::third_derivative`]'s seminumerical FD of the production
/// periodic Hessian.
///
/// **Cost note — what stays per-direction.** Everything below is a genuine
/// function of `v`: the `F¹/S¹` skeleton contractions, the first-order field,
/// the `F^vv/S^vv` directional second-order skeleton matrices, the
/// second-order field, the frozen third, the response path and the density
/// path. The geometric kernel motion at response charges uses the borrowed
/// charge-slot potential builder. It avoids cloning the SCC record and
/// rebuilding AO skeleton integrals for each direction. The reference also
/// reuses its immutable spectral divided-difference tables.
pub fn pbc_gamma_third_with_reference(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    pbc: &crate::pbc::PbcOptions,
    reference: &GammaThirdReference,
    v: &[f64],
) -> Result<f64> {
    use crate::pbc::gamma_third::{
        gamma_directional_second_matrices, gamma_response_path_directional,
        pbc_gamma_frozen_third_directional, shell_potential_first_directional,
        shell_potential_second_directional, uniform_density_images,
    };
    use crate::pbc::hessian::{
        response_gradient, shell_potential_derivatives_for_charges, DensityLookup,
    };

    let GammaThirdReference {
        scf,
        mos: _,
        sk,
        ctx,
        dens0,
        band_pairs,
        kernel,
        cn,
        lattice,
        n,
        nat,
        ndof,
    } = reference;
    let (n, nat, ndof) = (*n, *nat, *ndof);
    let lattice = *lattice;
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "pbc_gamma_third_with_reference: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let enable_cn = options.hamiltonian.enable_cn_hamiltonian;
    let coordination_cutoff = options.hamiltonian.coordination_cutoff;

    // ---- frozen third (all blocks, incl. the Ewald SCC2 sums) ----
    let frozen = pbc_gamma_frozen_third_directional(
        system,
        params,
        scf,
        sk,
        pbc,
        coordination_cutoff,
        enable_cn,
        options.enable_dispersion,
        options.d3_reference_path.as_deref(),
        v,
    )?;

    // ---- X¹: directional first-order field on the Gamma context ----
    let mut f1 = Matrix::zeros(n, n);
    let mut s1 = Matrix::zeros(n, n);
    for (y, &vy) in v.iter().enumerate() {
        if vy == 0.0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                f1[(i, j)] += vy * sk.fock[y][(i, j)];
                s1[(i, j)] += vy * sk.overlap[y][(i, j)];
            }
        }
    }
    let field = ctx.first_order_field(f1, s1)?;
    let q1 = field.bundle.shell_charges.clone();

    // ---- geometric kernel motion at the response charges: (∂γ/∂R·v)·q¹ ----
    let dgamma_v_q1 = {
        let mut atom_q = vec![0.0_f64; nat];
        for (ish, shell) in scf.basis.shells.iter().enumerate() {
            atom_q[shell.atom_index] += q1[ish];
        }
        let derivatives =
            shell_potential_derivatives_for_charges(system, &lattice, scf, pbc, &q1, &atom_q)?;
        let mut value = vec![0.; q1.len()];
        for (coordinate, &weight) in v.iter().enumerate() {
            for (shell, result) in value.iter_mut().enumerate() {
                *result += weight * derivatives[coordinate][shell];
            }
        }
        value
    };

    // ---- X²: the molecular second-order solver on the periodic inputs ----
    let v1_pot = shell_potential_first_directional(sk, v);
    let v2_pot = shell_potential_second_directional(system, &lattice, scf, pbc, v);
    let (f_vv, s_vv) =
        gamma_directional_second_matrices(system, params, scf, options, pbc, &v1_pot, &v2_pot, v)?;
    let second =
        ctx.second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_q1, &dgamma_v_q1)?;

    // ---- g(X²)·v ----
    let g2_grad = response_gradient(
        system,
        params,
        scf,
        band_pairs,
        DensityLookup::Uniform(&second.bundle.density),
        DensityLookup::Uniform(&second.bundle.energy_weighted),
        &second.bundle.shell_charges,
        kernel,
        pbc,
        cn.as_ref(),
    )?;
    let g2: f64 = g2_grad
        .iter()
        .enumerate()
        .map(|(at, g)| g.x * v[3 * at] + g.y * v[3 * at + 1] + g.z * v[3 * at + 2])
        .sum();

    // ---- B6 + complete geometric/reference background ----
    let dens1 = uniform_density_images(
        &lattice,
        pbc.ao_cutoff,
        &field.bundle.density,
        &field.bundle.energy_weighted,
    );
    let path = gamma_response_path_directional(
        system, params, scf, options, pbc, dens0, &dens1, &q1, &v1_pot, v,
    )?;

    // ---- density path: ∂frozen2/∂X₀ · X¹ ----
    // The periodic response gradient carries no potential legs (B6 above is
    // its geometric motion only), so the frozen Hessian's own X₀ motion is a
    // separate contribution: the Hessian-shaped blocks with X¹ in the frozen
    // slots — band/Pulay WITH the frozen-charge potential legs, the CN block
    // in its two-sided Hessian convention, the SCC2 charge-path bilinear,
    // and the V(q₀)-cache motion (value + geometric legs at q¹) via the
    // Δ-potential trick.
    let density_path = {
        use crate::pbc::gamma_third::{
            pbc_band_pulay_third_directional, pbc_cn_third_directional,
            pbc_scc2_bilinear_second_directional,
        };
        let dbg = std::env::var("GFN1_G3_DEBUG").is_ok();
        let mut dp = 0.0;
        let d_bp = pbc_band_pulay_third_directional(
            system, params, scf, pbc, &dens1, &v1_pot, &v2_pot, v,
        )?
        .second;
        dp += d_bp;
        let d_cn = if enable_cn {
            pbc_cn_third_directional(system, params, scf, pbc, coordination_cutoff, &dens1.p, v)?
                .second
        } else {
            0.0
        };
        dp += d_cn;
        let d_bl = 2.0
            * pbc_scc2_bilinear_second_directional(
                system,
                &lattice,
                scf,
                pbc,
                &scf.shell_charges,
                &q1,
                v,
            );
        dp += d_bl;
        // V(q₀)-cache motion: value shift K·q¹ plus its geometric legs at q¹.
        let kq1 = crate::linalg::matrix_vector_product(kernel, &q1)?;
        let (dgamma2_v_q1, doctored) = {
            let mut d = scf.clone();
            d.shell_charges = q1.clone();
            let mut atom_q = vec![0.0_f64; nat];
            for (ish, shell) in scf.basis.shells.iter().enumerate() {
                atom_q[shell.atom_index] += q1[ish];
            }
            d.atomic_charges = atom_q;
            let v2q1 = shell_potential_second_directional(system, &lattice, &d, pbc, v);
            (v2q1, d)
        };
        let _ = &doctored;
        let mut scf_shift = scf.clone();
        for (s, dv) in scf_shift.shell_scc_potential.iter_mut().zip(&kq1) {
            *s += dv;
        }
        let v1_shift: Vec<f64> = v1_pot
            .iter()
            .zip(&dgamma_v_q1)
            .map(|(a, b)| a + b)
            .collect();
        let v2_shift: Vec<f64> = v2_pot
            .iter()
            .zip(&dgamma2_v_q1)
            .map(|(a, b)| a + b)
            .collect();
        let shifted = pbc_band_pulay_third_directional(
            system, params, &scf_shift, pbc, dens0, &v1_shift, &v2_shift, v,
        )?
        .second;
        let base =
            pbc_band_pulay_third_directional(system, params, scf, pbc, dens0, &v1_pot, &v2_pot, v)?
                .second;
        dp += shifted - base;
        // The reference self-energy value's geometric motion belongs to the
        // complete frozen CN cubic, not this density-slot derivative. Its
        // response-gradient counterpart is the second CN1*E1(P1) term in B6.
        if dbg {
            // The X¹ band/Pulay block WITHOUT potential legs, for the
            // dpath-inventory bisection against the FD target.
            let d_bp0 = pbc_band_pulay_third_directional(
                system,
                params,
                scf,
                pbc,
                &dens1,
                &vec![0.0; scf.basis.shells.len()],
                &vec![0.0; scf.basis.shells.len()],
                v,
            )?
            .second;
            eprintln!(
                "dpath components: bp(X1;v1,v2) {d_bp:+.10e}  bp(X1;0,0) {d_bp0:+.10e}  cn2 \
                 {d_cn:+.10e}  2bil {d_bl:+.10e}  dV-cache {:+.10e}",
                shifted - base
            );
        }
        dp
    };

    if std::env::var("GFN1_G3_DEBUG").is_ok() {
        eprintln!(
            "g3 components: frozen {:+.10e}  dpath {density_path:+.10e}  g2 {g2:+.10e}  b6 \
             {:+.10e}  background {:+.10e}\n  b6 blocks {:?}\n  raw background families {:?}\n  frozen thirds: bp \
             {:+.10e}  cn {:+.10e}  scc2r {:+.10e}  scc2e {:+.10e}",
            frozen.total().third,
            path.b6,
            path.background_complete,
            path.b6_blocks,
            path.bg4_families,
            frozen.band_pulay.third,
            frozen.coordination.third,
            frozen.scc2_realspace.third,
            frozen.scc2_ewald.third
        );
    }
    Ok(frozen.total().third + density_path + g2 + path.b6 + path.background_complete)
}

/// **Analytic Gamma-point periodic third derivative, Vector mode**: the single
/// contracted scalar `e³[v] = Σ_abc T_abc v_a v_b v_c` along one direction.
///
/// The cheapest output mode — one shared-reference build plus one directional
/// evaluation. See [`GammaThirdReference`] for the option coverage (Gamma-only
/// mesh, canonical integer/fractional occupations, analytic order 3 terms) and
/// [`pbc_gamma_third_with_reference`] for the assembly.
pub fn pbc_gamma_third_analytic_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    pbc: &crate::pbc::PbcOptions,
    v: &[f64],
) -> Result<f64> {
    let reference = GammaThirdReference::build(system, params, options, pbc)?;
    pbc_gamma_third_with_reference(system, params, options, pbc, &reference, v)
}

/// **Dense mode**: the full packed `T_abc` recovered from directional
/// evaluations by the cubic polarization identity
/// `T(x₁,x₂,x₃) = (1/6) Σ_{∅≠S⊆{1,2,3}} (−1)^{3−|S|} e³[Σ_{i∈S} x_i]`,
/// exactly mirroring the molecular
/// [`crate::third_derivative::finite_t::third_derivative_finite_t_dense`]
/// driver: the subset directions of every canonical triple are deduplicated,
/// evaluated once each in parallel against ONE shared
/// [`GammaThirdReference`], and recombined.
///
/// Cost is `~C(n+2,3)` directional evaluations (56 for a 2-atom cell, 816 for
/// 8 atoms) and grows as `n³` — prefer [`pbc_gamma_third_analytic_block`] or
/// the vector mode for anything but small cells.
pub fn pbc_gamma_third_analytic_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    pbc: &crate::pbc::PbcOptions,
) -> Result<SymmetricThird> {
    let dofs: Vec<usize> = (0..3 * system.atoms.len()).collect();
    gamma_third_polarized(system, params, options, pbc, &dofs)
}

/// **Block mode**: the `|dofs|³` sub-tensor of the dense analytic Gamma third
/// derivative, indexed by POSITION in `dofs`, via the same polarization driver
/// — only the directions the requested triples actually need are evaluated.
pub fn pbc_gamma_third_analytic_block(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    pbc: &crate::pbc::PbcOptions,
    dofs: &[usize],
) -> Result<SymmetricThird> {
    let ndof = 3 * system.atoms.len();
    for &d in dofs {
        if d >= ndof {
            return Err(Gfn1Error::InvalidInput(format!(
                "pbc_gamma_third_analytic_block: dof {d} out of range (ndof {ndof})"
            )));
        }
    }
    gamma_third_polarized(system, params, options, pbc, dofs)
}

/// The shared polarization driver behind the dense and block modes.
fn gamma_third_polarized(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &crate::electronic::ElectronicOptions,
    pbc: &crate::pbc::PbcOptions,
    dofs: &[usize],
) -> Result<SymmetricThird> {
    let reference = GammaThirdReference::build(system, params, options, pbc)?;
    let plan = crate::derivative_assembly::periodic_cubic_plan(
        dofs, reference.ndof, reference.n, 1,
    )?;
    plan.cubic(|v| pbc_gamma_third_with_reference(system, params, options, pbc, &reference, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::ElectronicOptions;
    use crate::pbc::hessian::{
        gamma_cpxtb_response_directional, gamma_mos, gamma_skeleton_derivatives,
    };
    use crate::pbc::scf::run_pbc_scc;
    use crate::pbc::{EwaldOptions, KMesh, PbcOptions};

    fn params() -> Gfn1Parameters {
        Gfn1Parameters::builtin().unwrap()
    }

    #[test]
    #[ignore = "finite-temperature response-ladder diagnostic"]
    fn gamma_thermal_response_ladder_diagnostic() {
        use crate::pbc::gamma_third::{
            gamma_directional_second_matrices, shell_potential_first_directional,
            shell_potential_second_directional,
        };
        use crate::pbc::hessian::{pbc_gamma_hessian, shell_potential_derivatives_for_charges};
        use crate::response::charge_space::FirstOrderField;
        let params = params();
        let system=PeriodicSystem::from_xyz_str("2\nLattice=\"5.13 0 0 0 5.13 0 0 0 5.13\" pbc=\"T T T\"\nH .11 -.07 .03\nH .93 .12 -.06\n",0.,false).unwrap();
        let pbc = PbcOptions {
            kmesh: KMesh::gamma(),
            ao_cutoff: 12.,
            ewald: EwaldOptions {
                real_cutoff: 20.,
                sr_cutoff: 8.,
                ..Default::default()
            },
            ..Default::default()
        };
        let v = vec![1., 0., 0., 0., 0., 0.];
        for temperature in [0., 30000.] {
            for cn in [false, true] {
                let mut options = ElectronicOptions {
                    electronic_temperature: temperature,
                    enable_dispersion: false,
                    energy_tolerance: 1e-13,
                    charge_tolerance: 1e-11,
                    max_scc: 1000,
                    ..Default::default()
                };
                options.hamiltonian.enable_cn_hamiltonian = cn;
                let field_at =
                    |step: f64| -> (PeriodicSystem, GammaThirdReference, FirstOrderField) {
                        let mut moved = system.clone();
                        moved.atoms[0].position.x += step;
                        let reference =
                            GammaThirdReference::build(&moved, &params, &options, &pbc).unwrap();
                        let field = reference
                            .ctx
                            .first_order_field(
                                reference.sk.fock[0].clone(),
                                reference.sk.overlap[0].clone(),
                            )
                            .unwrap();
                        (moved, reference, field)
                    };
                let (_, reference, field) = field_at(0.);
                let scf = &reference.scf;
                let q1 = &field.bundle.shell_charges;
                let mut aq = vec![0.; 2];
                for (shell, &q) in scf.basis.shells.iter().zip(q1) {
                    aq[shell.atom_index] += q;
                }
                let dgamma = shell_potential_derivatives_for_charges(
                    &system,
                    &reference.lattice,
                    scf,
                    &pbc,
                    q1,
                    &aq,
                )
                .unwrap()[0]
                    .clone();
                let v1 = shell_potential_first_directional(&reference.sk, &v);
                let v2 =
                    shell_potential_second_directional(&system, &reference.lattice, scf, &pbc, &v);
                let (f2, s2) = gamma_directional_second_matrices(
                    &system, &params, scf, &options, &pbc, &v1, &v2, &v,
                )
                .unwrap();
                let second = reference
                    .ctx
                    .second_order_field(&field, &field, &f2, &s2, &dgamma, &dgamma)
                    .unwrap();
                for h in [1e-3, 5e-4] {
                    let (sp, rp, fp) = field_at(h);
                    let (sm, rm, fm) = field_at(-h);
                    let worst = |a: &Matrix, p: &Matrix, m: &Matrix| {
                        a.as_slice()
                            .iter()
                            .zip(p.as_slice())
                            .zip(m.as_slice())
                            .map(|((&a, &p), &m)| (a - (p - m) / (2. * h)).abs())
                            .fold(0., f64::max)
                    };
                    let p1 = worst(
                        &field.bundle.density,
                        &rp.scf.density_k[0].re,
                        &rm.scf.density_k[0].re,
                    );
                    let w1 = worst(
                        &field.bundle.energy_weighted,
                        &rp.scf.ew_density_k[0].re,
                        &rm.scf.ew_density_k[0].re,
                    );
                    let p2 = worst(
                        &second.bundle.density,
                        &fp.bundle.density,
                        &fm.bundle.density,
                    );
                    let w2 = worst(
                        &second.bundle.energy_weighted,
                        &fp.bundle.energy_weighted,
                        &fm.bundle.energy_weighted,
                    );
                    let hp = pbc_gamma_hessian(&sp, &params, &options, &pbc)
                        .unwrap()
                        .hessian;
                    let hm = pbc_gamma_hessian(&sm, &params, &options, &pbc)
                        .unwrap()
                        .hessian;
                    let analytic = pbc_gamma_third_with_reference(
                        &system, &params, &options, &pbc, &reference, &v,
                    )
                    .unwrap();
                    println!("thermal ladder T={temperature} CN={cn} h={h}: P1={p1:e},W1={w1:e},P2={p2:e},W2={w2:e}; FC3={analytic:e}, FD={:e}",(hp[(0,0)]-hm[(0,0)])/(2.*h));
                }
            }
        }
    }

    fn electronic() -> ElectronicOptions {
        let mut o = ElectronicOptions::default();
        o.enable_dispersion = false;
        o.energy_tolerance = 1.0e-12;
        o.charge_tolerance = 1.0e-10;
        o
    }

    fn pbc_opts() -> PbcOptions {
        PbcOptions {
            kmesh: KMesh::gamma(),
            ao_cutoff: 12.0,
            ewald: EwaldOptions {
                sr_cutoff: 8.0,
                ..EwaldOptions::default()
            },
            ..PbcOptions::default()
        }
    }

    // Same distorted fixtures as `pbc::gamma_third`'s SCF gates.
    const FIXTURES: &[(&str, &str)] = &[
        (
            "diamond-skew",
            "2\n\
Lattice=\"0.06 1.83 1.75 1.75 0.04 1.81 1.82 1.76 0.03\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.930000 0.880000 0.905000\n",
        ),
        (
            "BN-skew",
            "2\n\
Lattice=\"0.06 1.86 1.78 1.78 0.04 1.84 1.85 1.79 0.03\" pbc=\"T T T\"\n\
B 0.000000 0.000000 0.000000\n\
N 0.940000 0.890000 0.920000\n",
        ),
    ];

    fn direction(ndof: usize, seed: u64) -> Vec<f64> {
        (0..ndof)
            .map(|k| {
                let x = ((k as u64 + 1) * (seed + 7)) % 13;
                0.31 - 0.05 * (x as f64) + 0.01 * ((k % 3) as f64)
            })
            .collect()
    }

    #[test]
    #[ignore = "paired timing of the borrowed charge-slot optimization against the former full skeleton build"]
    fn charge_slot_potential_benchmark() {
        use crate::pbc::hessian::shell_potential_derivatives_for_charges;
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0., false).unwrap();
            let params = params();
            let options = electronic();
            let pbc = pbc_opts();
            let scf = run_pbc_scc(&system, &params, &options, &pbc).unwrap();
            assert!(scf.converged);
            let q: Vec<_> = (0..scf.basis.shells.len())
                .map(|s| 0.03 - 0.02 * s as f64)
                .collect();
            let mut atomic = vec![0.; system.atoms.len()];
            for (shell, &value) in scf.basis.shells.iter().zip(&q) {
                atomic[shell.atom_index] += value;
            }
            let legacy = || {
                let mut copied = scf.clone();
                copied.shell_charges = q.clone();
                copied.atomic_charges = atomic.clone();
                gamma_skeleton_derivatives(&system, &params, &copied, &options, &pbc)
                    .unwrap()
                    .shell_potential
            };
            let borrowed = || {
                shell_potential_derivatives_for_charges(
                    &system,
                    system.lattice.as_ref().unwrap(),
                    &scf,
                    &pbc,
                    &q,
                    &atomic,
                )
                .unwrap()
            };
            let expected = legacy();
            let actual = borrowed();
            let error = expected
                .iter()
                .flatten()
                .zip(actual.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(error < 1e-13);
            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for repetition in 0..5 {
                let mut sample = |old: bool| {
                    let start = std::time::Instant::now();
                    std::hint::black_box(if old { legacy() } else { borrowed() });
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    if old {
                        old_ms.push(elapsed);
                    } else {
                        new_ms.push(elapsed);
                    }
                };
                sample(repetition % 2 == 0);
                sample(repetition % 2 != 0);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            let avoided_payload =
                2 * 3 * system.atoms.len() * scf.basis.len().pow(2) * std::mem::size_of::<f64>();
            println!("charge_slot_benchmark fixture={name} legacy_median_ms={} borrowed_median_ms={} max_error={error:e} avoided_ao_matrix_payload_bytes={avoided_payload}",old_ms[2],new_ms[2]);
        }
    }

    /// FD split diagnostic: the true `D_v[r2]` (reconverged first-order
    /// response gradient contraction at displaced geometries) vs the
    /// assembly's response side `g(X²) + B6 + background_complete`, and by subtraction the
    /// frozen side vs `GammaFrozenThird.total().third`.
    #[test]
    #[ignore = "diagnostic"]
    fn gamma_response_split_diagnostic() {
        use crate::pbc::gamma_third::gamma_realspace_densities;
        use crate::pbc::hessian::{
            build_response_band_pairs, gamma_cpxtb_response_directional, gamma_mos,
            gamma_skeleton_derivatives, response_gradient, DensityLookup,
        };
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let ndof = 3 * system.atoms.len();
            let v = direction(ndof, 41);
            let r2_at = |lam: f64| -> f64 {
                let mut sys = system.clone();
                for (atom, a) in sys.atoms.iter_mut().enumerate() {
                    a.position.x += lam * v[3 * atom];
                    a.position.y += lam * v[3 * atom + 1];
                    a.position.z += lam * v[3 * atom + 2];
                }
                let lattice = *sys.lattice.as_ref().unwrap();
                let scf = crate::pbc::scf::run_pbc_scc(&sys, &params, &opts, &pbc).unwrap();
                let mos = gamma_mos(&scf, scf.nelec).unwrap();
                let sk = gamma_skeleton_derivatives(&sys, &params, &scf, &opts, &pbc).unwrap();
                let n = scf.basis.len();
                let mut f1 = Matrix::zeros(n, n);
                let mut s1 = Matrix::zeros(n, n);
                for (y, &vy) in v.iter().enumerate() {
                    for i in 0..n {
                        for j in 0..n {
                            f1[(i, j)] += vy * sk.fock[y][(i, j)];
                            s1[(i, j)] += vy * sk.overlap[y][(i, j)];
                        }
                    }
                }
                let (p1, w1, q1) = gamma_cpxtb_response_directional(&scf, &mos, &f1, &s1).unwrap();
                let dens0 = gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff);
                let band_pairs =
                    build_response_band_pairs(&sys, &params, &scf, &dens0.p, &pbc).unwrap();
                let kernel = periodic_response_kernel(&scf);
                let cn = crate::coordination::coordination_with_derivatives(
                    &sys,
                    crate::coordination::CoordinationOptions {
                        cutoff: opts.hamiltonian.coordination_cutoff,
                        ..crate::coordination::CoordinationOptions::default()
                    },
                )
                .unwrap();
                let g = response_gradient(
                    &sys,
                    &params,
                    &scf,
                    &band_pairs,
                    DensityLookup::Uniform(&p1),
                    DensityLookup::Uniform(&w1),
                    &q1,
                    &kernel,
                    &pbc,
                    Some(&cn),
                )
                .unwrap();
                g.iter()
                    .enumerate()
                    .map(|(at, gg)| gg.x * v[3 * at] + gg.y * v[3 * at + 1] + gg.z * v[3 * at + 2])
                    .sum()
            };
            let h = 1.0e-3;
            let fd_r2 = (r2_at(h) - r2_at(-h)) / (2.0 * h);
            let fd_r2_half = (r2_at(0.5 * h) - r2_at(-0.5 * h)) / h;
            let rich = (4.0 * fd_r2_half - fd_r2) / 3.0;
            println!("split/{name}: D_v[r2] FD {fd_r2:+.10e}  richardson {rich:+.10e}");
            assert!(fd_r2.is_finite());
        }
    }

    /// Frozen-side block split: FD each frozen block's `.second` at the
    /// RECONVERGED scf(λ) — true `block_third + block_density_path` — and
    /// print against the assembly attribution, pinning the density-path
    /// inventory block by block.
    #[test]
    #[ignore = "diagnostic"]
    fn gamma_frozen_block_split_diagnostic() {
        use crate::pbc::gamma_third::{
            gamma_realspace_densities, pbc_band_pulay_third_directional, pbc_cn_third_directional,
            pbc_scc2_realspace_third_directional, shell_potential_first_directional,
            shell_potential_second_directional,
        };
        use crate::pbc::hessian::gamma_skeleton_derivatives;
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let ndof = 3 * system.atoms.len();
            let v = direction(ndof, 41);
            let blocks_at = |lam: f64| -> [f64; 3] {
                let mut sys = system.clone();
                for (atom, a) in sys.atoms.iter_mut().enumerate() {
                    a.position.x += lam * v[3 * atom];
                    a.position.y += lam * v[3 * atom + 1];
                    a.position.z += lam * v[3 * atom + 2];
                }
                let lattice = *sys.lattice.as_ref().unwrap();
                let scf = crate::pbc::scf::run_pbc_scc(&sys, &params, &opts, &pbc).unwrap();
                let sk = gamma_skeleton_derivatives(&sys, &params, &scf, &opts, &pbc).unwrap();
                let dens = gamma_realspace_densities(&scf, &lattice, pbc.ao_cutoff);
                let v1 = shell_potential_first_directional(&sk, &v);
                let v2 = shell_potential_second_directional(&sys, &lattice, &scf, &pbc, &v);
                let bp = pbc_band_pulay_third_directional(
                    &sys, &params, &scf, &pbc, &dens, &v1, &v2, &v,
                )
                .unwrap()
                .second;
                let cn = pbc_cn_third_directional(
                    &sys,
                    &params,
                    &scf,
                    &pbc,
                    opts.hamiltonian.coordination_cutoff,
                    &dens.p,
                    &v,
                )
                .unwrap()
                .second;
                let s2 =
                    pbc_scc2_realspace_third_directional(&sys, &lattice, &scf, &pbc, &v).second;
                [bp, cn, s2]
            };
            let h = 1.0e-3;
            let (p, m) = (blocks_at(h), blocks_at(-h));
            let names = ["band_pulay", "cn", "scc2_real"];
            for i in 0..3 {
                println!(
                    "fblock/{name}/{}: D_v(true) {:+.10e}",
                    names[i],
                    (p[i] - m[i]) / (2.0 * h)
                );
            }
        }
    }

    /// X² isolation gate: the second-order shell charges `q^vv` from the
    /// molecular solver on periodic inputs vs the SECOND central difference
    /// of reconverged SCC shell charges along `v`.
    #[test]
    #[ignore = "diagnostic"]
    fn gamma_second_order_charges_match_reconverged_fd() {
        use crate::pbc::gamma_third::{
            gamma_directional_second_matrices, shell_potential_first_directional,
            shell_potential_second_directional,
        };
        use crate::pbc::hessian::{gamma_mos, gamma_skeleton_derivatives};
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let params = params();
            let mut opts = electronic();
            opts.energy_tolerance = 1.0e-13;
            opts.charge_tolerance = 1.0e-12;
            let pbc = pbc_opts();
            let lattice = *system.lattice.as_ref().unwrap();
            let scf = crate::pbc::scf::run_pbc_scc(&system, &params, &opts, &pbc).unwrap();
            let mos = gamma_mos(&scf, scf.nelec).unwrap();
            let sk = gamma_skeleton_derivatives(&system, &params, &scf, &opts, &pbc).unwrap();
            let ndof = 3 * system.atoms.len();
            let nat = system.atoms.len();
            let v = direction(ndof, 41);
            let n = scf.basis.len();
            let mut f1 = Matrix::zeros(n, n);
            let mut s1 = Matrix::zeros(n, n);
            for (y, &vy) in v.iter().enumerate() {
                for i in 0..n {
                    for j in 0..n {
                        f1[(i, j)] += vy * sk.fock[y][(i, j)];
                        s1[(i, j)] += vy * sk.overlap[y][(i, j)];
                    }
                }
            }
            let ctx = gamma_charge_space_context(&system, &params, &scf, &mos, opts.charge_order)
                .unwrap();
            let field = ctx.first_order_field(f1, s1).unwrap();
            let q1 = field.bundle.shell_charges.clone();
            let dgamma_v_q1 = {
                let mut d = scf.clone();
                d.shell_charges = q1.clone();
                let mut atom_q = vec![0.0_f64; nat];
                for (ish, shell) in scf.basis.shells.iter().enumerate() {
                    atom_q[shell.atom_index] += q1[ish];
                }
                d.atomic_charges = atom_q;
                let sk1 = gamma_skeleton_derivatives(&system, &params, &d, &opts, &pbc).unwrap();
                shell_potential_first_directional(&sk1, &v)
            };
            let v1_pot = shell_potential_first_directional(&sk, &v);
            let v2_pot = shell_potential_second_directional(&system, &lattice, &scf, &pbc, &v);
            let (f_vv, s_vv) = gamma_directional_second_matrices(
                &system, &params, &scf, &opts, &pbc, &v1_pot, &v2_pot, &v,
            )
            .unwrap();
            let second = ctx
                .second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_q1, &dgamma_v_q1)
                .unwrap();
            let charges_at = |lam: f64| -> Vec<f64> {
                let mut sys = system.clone();
                for (atom, a) in sys.atoms.iter_mut().enumerate() {
                    a.position.x += lam * v[3 * atom];
                    a.position.y += lam * v[3 * atom + 1];
                    a.position.z += lam * v[3 * atom + 2];
                }
                crate::pbc::scf::run_pbc_scc(&sys, &params, &opts, &pbc)
                    .unwrap()
                    .shell_charges
            };
            let h = 2.0e-3;
            let (cp, c0, cm) = (charges_at(h), charges_at(0.0), charges_at(-h));
            let mut worst = 0.0_f64;
            for s in 0..q1.len() {
                let fd = (cp[s] - 2.0 * c0[s] + cm[s]) / (h * h);
                worst = worst.max((fd - second.bundle.shell_charges[s]).abs());
            }
            println!("gamma_q2/{name}: worst |analytic q^vv - FD| {worst:.3e}");
            assert!(worst.is_finite());
        }
    }

    /// **The Phase 8.3 total gate**: the assembled analytic directional third
    /// against the seminumerical FD of the production periodic Hessian
    /// (`pbc_third_derivative_seminumerical_vector`), on both distorted
    /// fixtures.
    #[test]
    #[ignore = "periodic total gate: one analytic assembly + 2 Hessian FDs per fixture"]
    fn gamma_analytic_third_matches_seminumerical() {
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let ndof = 3 * system.atoms.len();
            let v = direction(ndof, 41);
            let analytic = pbc_gamma_third_analytic_vector(&system, &params, &opts, &pbc, &v)
                .expect("analytic third");
            let ref_at = |h: f64| -> f64 {
                let dv_h = crate::pbc::third_derivative::pbc_third_derivative_seminumerical_vector(
                    &system, &params, &opts, &pbc, h, &v,
                )
                .expect("seminumerical");
                let mut acc = 0.0;
                for a in 0..ndof {
                    for b in 0..ndof {
                        acc += v[a] * v[b] * dv_h[(a, b)];
                    }
                }
                acc
            };
            let ref_h = ref_at(1.0e-3);
            let ref_h2 = ref_at(5.0e-4);
            let reference = (4.0 * ref_h2 - ref_h) / 3.0;
            println!(
                "gamma_total/{name}: analytic {analytic:+.10e} vs seminumerical richardson \
                 {reference:+.10e}  |delta| {:.3e}  (fd(h) delta {:.3e}, ladder ratio {:.2})",
                (analytic - reference).abs(),
                (analytic - ref_h).abs(),
                (ref_h - analytic) / (ref_h2 - analytic)
            );
            assert!(
                (analytic - reference).abs() < 1.0e-6 * (1.0 + reference.abs()),
                "{name}: analytic Gamma third vs seminumerical: {analytic:.10e} vs \
                 {reference:.10e}"
            );
        }
    }

    /// The charge-space first-order field (dielectric route) must agree with
    /// the occ-virt PCG route on `(P¹, W¹, q¹)` — two mathematically
    /// equivalent solvers of the same fixed point, implemented independently.
    #[test]
    #[ignore = "periodic response gate: context build + two first-order solves per fixture"]
    fn gamma_context_first_order_matches_pcg_route() {
        for (name, xyz) in FIXTURES {
            let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
            let params = params();
            let opts = electronic();
            let pbc = pbc_opts();
            let scf = run_pbc_scc(&system, &params, &opts, &pbc).expect("periodic SCC");
            assert!(scf.converged, "fixture SCC did not converge");
            let mos = gamma_mos(&scf, scf.nelec).expect("gamma mos");
            let sk =
                gamma_skeleton_derivatives(&system, &params, &scf, &opts, &pbc).expect("skeleton");
            let ndof = 3 * system.atoms.len();
            let v = direction(ndof, 37);
            let n = scf.basis.len();
            let mut f1 = Matrix::zeros(n, n);
            let mut s1 = Matrix::zeros(n, n);
            for (y, &vy) in v.iter().enumerate() {
                for i in 0..n {
                    for j in 0..n {
                        f1[(i, j)] += vy * sk.fock[y][(i, j)];
                        s1[(i, j)] += vy * sk.overlap[y][(i, j)];
                    }
                }
            }
            let (p_ref, w_ref, q_ref) =
                gamma_cpxtb_response_directional(&scf, &mos, &f1, &s1).expect("PCG route");
            let ctx = gamma_charge_space_context(&system, &params, &scf, &mos, opts.charge_order)
                .expect("gamma context");
            let field = ctx
                .first_order_field(f1.clone(), s1.clone())
                .expect("charge-space route");
            let mut worst_p = 0.0_f64;
            let mut worst_w = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    worst_p = worst_p.max((field.bundle.density[(i, j)] - p_ref[(i, j)]).abs());
                    worst_w =
                        worst_w.max((field.bundle.energy_weighted[(i, j)] - w_ref[(i, j)]).abs());
                }
            }
            let mut worst_q = 0.0_f64;
            for s in 0..q_ref.len() {
                worst_q = worst_q.max((field.bundle.shell_charges[s] - q_ref[s]).abs());
            }
            println!(
                "gamma_ctx_first/{name}: worst dP {worst_p:.3e}  dW {worst_w:.3e}  dq \
                 {worst_q:.3e}"
            );
            assert!(
                worst_p < 1.0e-7 && worst_w < 1.0e-7 && worst_q < 1.0e-8,
                "{name}: charge-space vs PCG first-order response disagree: dP {worst_p:.3e} \
                 dW {worst_w:.3e} dq {worst_q:.3e}"
            );
        }
    }
}
