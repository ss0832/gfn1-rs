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
//! Gate: the context's first-order field must agree with the independent
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

/// Occupations closer than this to 0 or 2 count as integer (the Fermi fill at
/// the default 300 K leaves a gapped insulator at exactly integer filling).
/// Mirrors `pbc::hessian`'s own fractional-occupation epsilon.
const FRACTIONAL_OCC_EPS: f64 = 1.0e-10;

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
    /// capped at order 1), a Gamma-only k-mesh, a lattice, SCC convergence,
    /// and finally — only knowable after the SCC — integer occupations.
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &crate::electronic::ElectronicOptions,
        pbc: &crate::pbc::PbcOptions,
    ) -> Result<Self> {
        use crate::pbc::gamma_third::gamma_realspace_densities;
        use crate::pbc::hessian::{build_response_band_pairs, gamma_mos, gamma_skeleton_derivatives};

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
        if !scf.converged {
            return Err(Gfn1Error::InvalidInput(
                "periodic SCC did not converge for the analytic third derivative".into(),
            ));
        }
        let mos = gamma_mos(&scf, scf.nelec)?;
        if mos
            .occupations
            .iter()
            .any(|&f| f > FRACTIONAL_OCC_EPS && f < 2.0 - FRACTIONAL_OCC_EPS)
        {
            return Err(Gfn1Error::InvalidInput(
                "the analytic Gamma-point periodic third derivative requires integer (gapped) \
                 occupations, but the periodic SCC converged to a Fermi-smeared filling; \
                 set ElectronicOptions::electronic_temperature = 0, or use \
                 pbc_third_derivative_seminumerical_* which supports fractional occupations"
                    .into(),
            ));
        }
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
///         + B6(X¹)[v,v] + 2·bg4(X¹, X¹)·v     (gamma_response_path_directional)
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
/// path. One item is per-direction and NOT cheap: `dγ_v_q1` needs the
/// geometric kernel motion evaluated AT the response charges `q¹`, which the
/// skeleton builder only exposes through a full `gamma_skeleton_derivatives`
/// call on an SCC record doctored to carry `q¹`. That is a second O(N²)
/// image-summed skeleton build per direction — unavoidable without a
/// charge-slot-generic skeleton entry point, and it is why the dense driver
/// costs ~2x a naive per-direction skeleton count.
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
    use crate::pbc::hessian::{gamma_skeleton_derivatives, response_gradient, DensityLookup};

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
        let mut doctored = scf.clone();
        doctored.shell_charges = q1.clone();
        let mut atom_q = vec![0.0_f64; nat];
        for (ish, shell) in scf.basis.shells.iter().enumerate() {
            atom_q[shell.atom_index] += q1[ish];
        }
        doctored.atomic_charges = atom_q;
        let sk1 = gamma_skeleton_derivatives(system, params, &doctored, options, pbc)?;
        shell_potential_first_directional(&sk1, v)
    };

    // ---- X²: the molecular second-order solver on the periodic inputs ----
    let v1_pot = shell_potential_first_directional(sk, v);
    let v2_pot = shell_potential_second_directional(system, &lattice, scf, pbc, v);
    let (f_vv, s_vv) =
        gamma_directional_second_matrices(system, params, scf, options, pbc, &v1_pot, &v2_pot, v)?;
    let second = ctx.second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_q1, &dgamma_v_q1)?;

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

    // ---- B6 + bg4 ----
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
        let v1_shift: Vec<f64> =
            v1_pot.iter().zip(&dgamma_v_q1).map(|(a, b)| a + b).collect();
        let v2_shift: Vec<f64> =
            v2_pot.iter().zip(&dgamma2_v_q1).map(|(a, b)| a + b).collect();
        let shifted = pbc_band_pulay_third_directional(
            system, params, &scf_shift, pbc, dens0, &v1_shift, &v2_shift, v,
        )?
        .second;
        let base = pbc_band_pulay_third_directional(
            system, params, scf, pbc, dens0, &v1_pot, &v2_pot, v,
        )?
        .second;
        dp += shifted - base;
        // NOT ADDED — the self-energy cache motion. `pbc_band_pulay_*` carries
        // the H0 prefactor's ½(se_i + se_j) and `se` moves with the
        // coordination numbers, and the cache probe measures that channel at
        // −5.18e-8 (diamond) / −3.02e-8 (BN). Adding it via the affine trick
        // (`se := dsedcn ⊙ CN¹` minus `se := 0`) moves the total gate by
        // exactly those amounts: diamond improves 8.53e-8 → 3.36e-8, but BN
        // degrades 8.03e-8 → 1.10e-7. So the channel is real yet a
        // compensating piece of the opposite sign is still missing on the
        // heteronuclear fixture — most likely the CN block's third already
        // absorbs part of the same physics (its cross term is built from
        // `dE/dCN`, which is the se-mediated coupling), so the two must be
        // apportioned rather than summed. Left out until that boundary is
        // derived; the residual it would fix is the documented ~1e-7 tail.
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
             {:+.10e}  bg4 {:+.10e}\n  b6 blocks {:?}\n  bg4 families {:?}\n  frozen thirds: bp \
             {:+.10e}  cn {:+.10e}  scc2r {:+.10e}  scc2e {:+.10e}",
            frozen.total().third,
            path.b6,
            path.bg4,
            path.b6_blocks,
            path.bg4_families,
            frozen.band_pulay.third,
            frozen.coordination.third,
            frozen.scc2_realspace.third,
            frozen.scc2_ewald.third
        );
    }
    // bg4 enters twice: the response gradient's bilinear self-terms
    // differentiate into BOTH slots (e.g. d/dλ[−P¹(Kq¹)∇S] feeds the mixed
    // (X²,X¹) pairs from either side), while g(X²) supplies only the
    // diagonal — the two mixed completions are exactly the bg4 families.
    Ok(frozen.total().third + density_path + g2 + path.b6 + 2.0 * path.bg4)
}

/// **Analytic Gamma-point periodic third derivative, Vector mode**: the single
/// contracted scalar `e³[v] = Σ_abc T_abc v_a v_b v_c` along one direction.
///
/// The cheapest output mode — one shared-reference build plus one directional
/// evaluation. See [`GammaThirdReference`] for the option coverage (Gamma-only
/// mesh, integer occupations, analytic order 3 terms) and
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
    use rayon::prelude::*;
    use std::collections::{BTreeMap, HashMap};

    let reference = GammaThirdReference::build(system, params, options, pbc)?;
    let ndof = 3 * system.atoms.len();
    let m = dofs.len();

    // Phase 1: deduplicate the subset directions of every canonical triple.
    let mut key_index: HashMap<Vec<(usize, u8)>, usize> = HashMap::new();
    let mut keys: Vec<Vec<(usize, u8)>> = Vec::new();
    // Per canonical triple: the 7 (sign, key) polarization terms.
    let mut plan: Vec<((usize, usize, usize), Vec<(f64, usize)>)> = Vec::new();
    for k in 0..m {
        for j in 0..=k {
            for i in 0..=j {
                let idxs = [dofs[i], dofs[j], dofs[k]];
                let mut terms = Vec::with_capacity(7);
                for mask in 1u8..8 {
                    let mut dir: BTreeMap<usize, u8> = BTreeMap::new();
                    for (bit, &dof) in idxs.iter().enumerate() {
                        if mask & (1 << bit) != 0 {
                            *dir.entry(dof).or_insert(0) += 1;
                        }
                    }
                    let key: Vec<(usize, u8)> = dir.into_iter().collect();
                    let sign = if (3 - mask.count_ones()) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let idx = *key_index.entry(key.clone()).or_insert_with(|| {
                        keys.push(key);
                        keys.len() - 1
                    });
                    terms.push((sign, idx));
                }
                plan.push(((i, j, k), terms));
            }
        }
    }

    // Phase 2: evaluate each distinct direction once, in parallel, against
    // the shared reference.
    let values: Result<Vec<f64>> = keys
        .par_iter()
        .map(|key| {
            let mut v = vec![0.0_f64; ndof];
            for &(dof, weight) in key {
                v[dof] = weight as f64;
            }
            pbc_gamma_third_with_reference(system, params, options, pbc, &reference, &v)
        })
        .collect();
    let values = values?;

    // Phase 3: assemble the packed tensor.
    let mut store = SymmetricThird::zeros(m);
    for ((i, j, k), terms) in plan {
        let mut t = 0.0;
        for (sign, idx) in terms {
            t += sign * values[idx];
        }
        store.add(i, j, k, t / 6.0);
    }
    Ok(store)
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

    /// FD split diagnostic: the true `D_v[r2]` (reconverged first-order
    /// response gradient contraction at displaced geometries) vs the
    /// assembly's response side `g(X²) + B6 + bg4`, and by subtraction the
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
                .map(|(at, gg)| {
                    gg.x * v[3 * at] + gg.y * v[3 * at + 1] + gg.z * v[3 * at + 2]
                })
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
            gamma_realspace_densities, pbc_band_pulay_third_directional,
            pbc_cn_third_directional, pbc_scc2_realspace_third_directional,
            shell_potential_first_directional, shell_potential_second_directional,
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
            let ctx =
                gamma_charge_space_context(&system, &params, &scf, &mos, opts.charge_order)
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
                let sk1 =
                    gamma_skeleton_derivatives(&system, &params, &d, &opts, &pbc).unwrap();
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
                let dv_h =
                    crate::pbc::third_derivative::pbc_third_derivative_seminumerical_vector(
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
            let sk = gamma_skeleton_derivatives(&system, &params, &scf, &opts, &pbc)
                .expect("skeleton");
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
                    worst_w = worst_w
                        .max((field.bundle.energy_weighted[(i, j)] - w_ref[(i, j)]).abs());
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
