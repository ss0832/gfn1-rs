// SPDX-License-Identifier: GPL-3.0-or-later
//! Charge-space ("AO-side") CP-SCC solver.
//!
//! The SCC self-consistency couples the density only through the `nsh` Mulliken
//! shell charges, so any response splits into a **bare** part (the density
//! response at frozen SCC potential, evaluated spectrally from the reference
//! MOs) and a **screening** part mediated by the response kernel `K`. With the
//! static shell-charge susceptibility `χ⁰` (charges induced by a unit shell
//! potential at frozen skeleton), the self-consistent shell-charge response of
//! ANY perturbation solves the `nsh × nsh` linear system
//!
//! ```text
//!   (I − χ⁰ K) δq  =  δq_bare ,
//! ```
//!
//! whose dielectric matrix is LU-factored ONCE and reused for every right-hand
//! side — all `3N` first-order geometric perturbations and, in the quartic
//! force-constant work, all `O((3N)²)` second-order ones. This replaces both
//! the MO-pair-space iterative/dense CPXTB solves and the 50-iteration
//! fixed-point loop of the finite-temperature branch with one exact solve.
//!
//! Finite temperature is native: the unified response-coefficient formula in
//! [`super::cpxtb`] covers fractional occupations (orbital rotations with
//! `(f_p − f_q)/(ε_p − ε_q)` weights, the metric channel, and the
//! grand-canonical occupation channel with the chemical-potential shift from
//! particle-number conservation). At `T = 0` with integer occupations the
//! occupation channel vanishes and the same formulas reduce exactly to the
//! classic CPXTB response; a (near-)zero occupied–virtual gap with integer
//! occupations is rejected at build time (singular response — use Fermi
//! smearing), mirroring the CPXTB guard.

use crate::basis::BasisSet;
use crate::error::{Gfn1Error, Result};
use crate::linalg::{lowdin_solve_generalized, matrix_vector_product, DenseLu, Matrix};
use crate::params::Gfn1Parameters;
use crate::system::PeriodicSystem;

use super::cpxtb::{
    fermi_occupation_response, finite_temperature_mo_derivatives,
    finite_temperature_response_coefficients_from_mo, mo_coefficient_matrix_to_ao,
    orbital_energy_response_from_mo, response_shell_charges_from_density,
    response_shell_scc_kernel, scalar_response_fock_matrix,
};
use crate::constants::KB_HARTREE_PER_K;
use crate::electronic::ElectronicResult;

/// One perturbation's fully screened first-order response bundle.
#[derive(Clone, Debug)]
pub struct FirstOrderBundle {
    /// Density response `P^{(x)}` (AO, n×n).
    pub density: Matrix,
    /// Energy-weighted density response `W^{(x)}` (AO, n×n).
    pub energy_weighted: Matrix,
    /// Self-consistent shell-charge response `q^{(x)}` (nsh).
    pub shell_charges: Vec<f64>,
    /// Occupation response `f^{(x)}` (norb; zero at integer occupations).
    pub occupation_response: Vec<f64>,
    /// Screening potential `K q^{(x)}` (nsh) — the SCC part of the response
    /// Fock actually applied.
    pub screened_potential: Vec<f64>,
}

/// Reference-state data + the factored dielectric matrix.
pub struct ChargeSpaceContext {
    basis: BasisSet,
    overlap: Matrix,
    density0: Matrix,
    mos: Matrix,
    orbital_energies: Vec<f64>,
    occupations: Vec<f64>,
    /// Response kernel `K = γ + ∂²E_onsite/∂q²` (nsh×nsh).
    pub kernel: Matrix,
    /// Static shell-charge susceptibility `χ⁰` (nsh×nsh).
    pub chi0: Matrix,
    dielectric: DenseLu,
    kt: f64,
    finite_t: bool,
    /// Shell → atom map (for the onsite ∂K/∂q chain).
    shell_atom: Vec<usize>,
    /// Per-atom `∂³E_onsite/∂q³` at the reference charges (2Γ for stock GFN1,
    /// plus the Breathing-Radius orders when `charge_order > 3`).
    kernel_q_atom: Vec<f64>,
    /// Per-atom `∂⁴E_onsite/∂q⁴` (zero for stock GFN1; nonzero Breathing-Radius
    /// orders when `charge_order ≥ 4`) — the third-order solve's `E''''` chain.
    kernel_q2_atom: Vec<f64>,
    /// Use the frame-free Daleckii–Krein (resolvent) form for the
    /// second-order density/energy-weighted response instead of the
    /// coefficient/rotation algebra. Equality-gated against the frame path on
    /// non-degenerate finite-T systems; the only form valid inside exactly
    /// degenerate fractionally occupied blocks.
    dk_second: bool,
}

/// One perturbation's first-order data cached for second-order solves: the
/// screened bundle plus the AO/MO representations the second-order algebra
/// contracts with.
#[derive(Clone, Debug)]
pub struct FirstOrderField {
    pub bundle: FirstOrderBundle,
    /// Skeleton Fock derivative `d(H0 − ½vS)/dλ` at frozen density (AO).
    pub fock_skeleton: Matrix,
    /// Overlap derivative `dS/dλ` (AO).
    pub overlap_deriv: Matrix,
    /// MO representation `C†(F_skel + RF(Kq))C` of the TOTAL Fock derivative.
    pub h_tilde: Matrix,
    /// MO representation `C†(dS/dλ)C`.
    pub s_tilde: Matrix,
    /// Per-orbital orbital-energy response `ε^{(λ)}_p = h̃_pp − ε_p s̃_pp`.
    pub eps_response: Vec<f64>,
    /// MO rotation `U^{(λ)}` (so `C^{(λ)} = C U`): diagonal `−½s̃_pp`, ov/general
    /// `(h̃_pq − ε_q s̃_pq)/(ε_q − ε_p)`, symmetric gauge `−½s̃_pq` inside
    /// (near-)degenerate blocks.
    pub u_rotation: Matrix,
}

/// The screened second-order response bundle for a perturbation pair `(x, y)`.
#[derive(Clone, Debug)]
pub struct SecondOrderBundle {
    pub density: Matrix,
    pub energy_weighted: Matrix,
    pub shell_charges: Vec<f64>,
    /// Occupation second response `f^{(xy)}` (zero at integer occupations).
    pub occupation_response: Vec<f64>,
}

/// The screened second-order bundle for a pair `(x, y)` PLUS the second-order
/// MO-representation objects the quartic assembly contracts with — the exact
/// second-order siblings of the [`FirstOrderField`] members.
///
/// Every member is a **total** `λ_y`-derivative of the corresponding
/// first-order object of the `x` field, i.e. `D_y[·]`, with the frame rotation
/// `U^{(y)}` of the MO basis already folded in.
#[derive(Clone, Debug)]
pub struct SecondOrderField {
    /// The screened second-order response bundle (identical to what
    /// [`ChargeSpaceContext::solve_second_order`] returns).
    pub bundle: SecondOrderBundle,
    /// `ḣ = D_y[h̃^{(x)}]` — MO representation of the second derivative of the
    /// TOTAL Fock operator, INCLUDING the `RF(K q^{(xy)})` screening add.
    pub h_dot: Matrix,
    /// `ṡ = D_y[s̃^{(x)}]` — MO representation of `d²S/dλ_x dλ_y`.
    pub s_dot: Matrix,
    /// `D_y[ε^{(x)}_p]` — the second-order per-orbital energies
    /// `ḣ_pp − ε^{(y)}_p s̃^{(x)}_pp − ε_p ṡ_pp`.
    pub eps_second: Vec<f64>,
    /// `D_y[U^{(x)}]` — the second-order MO rotation (same gauge convention as
    /// [`FirstOrderField::u_rotation`]: `−½ṡ_pp` on the diagonal and `−½ṡ_pq`
    /// inside (near-)degenerate blocks).
    pub u_second: Matrix,
    /// FIXED-basis part of `ḣ` — `C†(d²F_total/dλ²)C` WITHOUT the `U`-frame
    /// transport (the third-order solve's `mo_transform(F^{vv}_AO)` leg,
    /// including the `RF(K q^{(xy)})` screening add).
    pub h_dot_fixed: Matrix,
    /// FIXED-basis part of `ṡ` — `C†(d²S/dλ²)C`.
    pub s_dot_fixed: Matrix,
}

/// The screened THIRD-order directional response bundle (`x = y = z = v`).
#[derive(Clone, Debug)]
pub struct ThirdOrderBundle {
    pub density: Matrix,
    pub energy_weighted: Matrix,
    pub shell_charges: Vec<f64>,
    /// Occupation third response `f^{(vvv)}` (zero at integer occupations).
    pub occupation_response: Vec<f64>,
}

/// Orbital-energy separation below which a divided difference switches to its
/// confluent (derivative) limit.
const DK_CONFLUENT_EPS: f64 = 1.0e-9;

/// Reference-spectrum divided differences of the grand-canonical Fermi
/// function, for the Daleckii–Krein (resolvent) response forms.
struct DkTables {
    eps: Vec<f64>,
    f: Vec<f64>,
    fp: Vec<f64>,
    fpp: Vec<f64>,
    f1: Matrix,
    fp1: Matrix,
}

impl DkTables {
    /// Second divided difference `f^{[2]}(ε_p, ε_r, ε_q)` with both confluent
    /// branches: `p ≈ q` (pinched) and the fully confluent `p ≈ r ≈ q`.
    #[inline]
    fn f2(&self, p: usize, r: usize, q: usize) -> f64 {
        let dpq = self.eps[p] - self.eps[q];
        if dpq.abs() > DK_CONFLUENT_EPS {
            return (self.f1[(p, r)] - self.f1[(r, q)]) / dpq;
        }
        let drp = self.eps[r] - self.eps[p];
        if drp.abs() > DK_CONFLUENT_EPS {
            return (self.f1[(p, r)] - self.fp[p]) / drp;
        }
        0.5 * self.fpp[p]
    }
}

/// Per-block data of the Λ-covariant degenerate occupation channel (all
/// matrices are in-block, `k × k` over `members`).
struct BlockSecondData {
    members: Vec<usize>,
    eps0: f64,
    /// Shared Fermi weight `w = f(1 − f/2)/kT = −f'`.
    w: f64,
    fbar: f64,
    lx: Matrix,
    ly: Matrix,
    dlx: Matrix,
    dly: Matrix,
    ldot: Matrix,
    sx: Matrix,
    sdot: Matrix,
    /// `(f''/2){δΛ^y, δΛ^x}`.
    prod: Matrix,
    f_xy: Matrix,
}

impl ChargeSpaceContext {
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        electronic: &ElectronicResult,
    ) -> Result<Self> {
        if system.lattice.is_some() {
            return Err(Gfn1Error::InvalidInput(
                "the charge-space response solver is non-PBC only".to_string(),
            ));
        }
        let basis = electronic.basis.clone();
        let overlap = electronic.integrals.overlap.clone();

        let eig = lowdin_solve_generalized(&electronic.fock, &overlap, 1.0e-12)?;
        let mos = eig.vectors;
        let orbital_energies = eig.values;
        let occupations = electronic.occupations.clone();

        let kernel = response_shell_scc_kernel(system, params, electronic)?;
        Self::from_raw_parts(
            system,
            params,
            basis,
            overlap,
            electronic.density.clone(),
            mos,
            orbital_energies,
            occupations,
            electronic.electronic_temperature,
            &electronic.atomic_charges,
            electronic.charge_order,
            kernel,
        )
    }

    /// Assemble a context from raw reference-state parts with an **injected
    /// response kernel** — the shared tail of [`Self::build`], exposed so the
    /// Gamma-point periodic driver can hand in the Bloch reference (real Γ
    /// matrices) together with `periodic_response_kernel`. Everything below is
    /// representation-generic: χ⁰, the dielectric LU, and the onsite
    /// `∂K/∂q` chain read only the arguments.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_raw_parts(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        basis: BasisSet,
        overlap: Matrix,
        density0: Matrix,
        mos: Matrix,
        orbital_energies: Vec<f64>,
        occupations: Vec<f64>,
        electronic_temperature: f64,
        atomic_charges: &[f64],
        charge_order: usize,
        kernel: Matrix,
    ) -> Result<Self> {
        let n = basis.len();
        let nshell = basis.shells.len();
        if occupations.len() != n {
            return Err(Gfn1Error::InvalidInput(
                "charge-space solver: occupation/basis dimension mismatch".to_string(),
            ));
        }
        let kt = electronic_temperature.max(0.0) * KB_HARTREE_PER_K;
        let finite_t = kt > 0.0
            && occupations
                .iter()
                .any(|&f| f > 1.0e-10 && f < 2.0 - 1.0e-10);
        if !finite_t {
            // Integer occupations: a vanishing occupied-virtual gap makes the
            // response singular (same guard as the MO-space CPXTB solver).
            for (i, &fi) in occupations.iter().enumerate() {
                if fi <= 1.0e-8 {
                    continue;
                }
                for (a, &fa) in occupations.iter().enumerate() {
                    if fa > 1.0e-8 {
                        continue;
                    }
                    let gap = orbital_energies[a] - orbital_energies[i];
                    if gap < 1.0e-6 {
                        return Err(Gfn1Error::InvalidInput(format!(
                            "charge-space response is singular: occupied orbital {i} and \
                             virtual orbital {a} are (near-)degenerate (gap {gap:.3e} Ha) \
                             with integer occupations; enable Fermi smearing"
                        )));
                    }
                }
            }
        }

        // χ⁰: shell charges induced by a unit potential on each shell, at
        // frozen skeleton (no geometric derivative, no screening).
        let zero = Matrix::zeros(n, n);
        let mut chi0 = Matrix::zeros(nshell, nshell);
        let helper = ResponseHelper {
            mos: &mos,
            orbital_energies: &orbital_energies,
            occupations: &occupations,
            kt,
            finite_t,
        };
        for t in 0..nshell {
            let mut unit = vec![0.0_f64; nshell];
            unit[t] = 1.0;
            let response_fock = scalar_response_fock_matrix(&basis, &overlap, &unit)?;
            let (dp, _df) = helper.density_response(&response_fock, &zero, &zero)?;
            let dq = response_shell_charges_from_density(&basis, &overlap, &density0, &dp, &zero)?;
            for s in 0..nshell {
                chi0[(s, t)] = dq[s];
            }
        }

        // Dielectric D = I − χ⁰ K, factored once.
        let chi0_k = chi0.matmul(&kernel)?;
        let mut dielectric = Matrix::zeros(nshell, nshell);
        for s in 0..nshell {
            for t in 0..nshell {
                dielectric[(s, t)] = if s == t { 1.0 } else { 0.0 } - chi0_k[(s, t)];
            }
        }
        let dielectric_lu = DenseLu::factor(&dielectric)?;

        // Onsite ∂K/∂q chain data (see the FC3 kernel-chain fix): shell → atom
        // map and per-atom ∂³E_onsite/∂q³ at the reference charges.
        let shell_model = crate::coulomb::ShellChargeModel::build(system, &basis, params)?;
        let nat = system.atoms.len();
        let charge_order = charge_order.max(3);
        let mut shell_atom = vec![0usize; nshell];
        for atom in 0..nat {
            let offset = shell_model.atom_offsets[atom];
            for local in 0..shell_model.atom_shell_counts[atom] {
                shell_atom[offset + local] = atom;
            }
        }
        let mut kernel_q_atom = vec![0.0_f64; nat];
        let mut kernel_q2_atom = vec![0.0_f64; nat];
        for atom in 0..nat {
            if shell_model.atom_shell_counts[atom] == 0 {
                continue;
            }
            let offset = shell_model.atom_offsets[atom];
            let (_, _, third, fourth) = crate::coulomb::onsite_charge_anharmonic_derivatives(
                shell_model.hardness[offset],
                shell_model.hubbard_derivs[offset],
                charge_order,
                atomic_charges[atom],
            );
            kernel_q_atom[atom] = third;
            kernel_q2_atom[atom] = fourth;
        }

        let mut ctx = Self {
            basis,
            overlap,
            density0,
            mos,
            orbital_energies,
            occupations,
            kernel,
            chi0,
            dielectric: dielectric_lu,
            kt,
            finite_t,
            shell_atom,
            kernel_q_atom,
            kernel_q2_atom,
            dk_second: false,
        };
        // Exactly degenerate fractionally occupied orbitals make the
        // coefficient/rotation algebra basis-dependent (the in-block second
        // order is not a function of the arbitrary intra-block eigenbasis).
        // The resolvent form is; switch to it exactly there.
        ctx.dk_second = !ctx.fractional_degenerate_blocks().is_empty();
        Ok(ctx)
    }

    /// Switch the second-order density/energy-weighted response to the
    /// frame-free Daleckii–Krein form (see [`Self::dk_second_order_mo`]).
    /// [`Self::build`] already selects it for degenerate references; this is
    /// the equality gate's handle for forcing it on a non-degenerate one.
    #[cfg(test)]
    pub(crate) fn set_dk_second(&mut self, on: bool) {
        self.dk_second = on;
    }

    #[inline]
    pub fn nshell(&self) -> usize {
        self.basis.shells.len()
    }

    #[inline]
    pub fn is_finite_temperature(&self) -> bool {
        self.finite_t
    }

    fn helper(&self) -> ResponseHelper<'_> {
        ResponseHelper {
            mos: &self.mos,
            orbital_energies: &self.orbital_energies,
            occupations: &self.occupations,
            kt: self.kt,
            finite_t: self.finite_t,
        }
    }

    /// Solve the fully screened first-order response for one perturbation,
    /// specified by its skeleton Fock derivative (`d(H0 − ½v·S)/dλ` at frozen
    /// density) and overlap derivative `dS/dλ`.
    pub fn solve_first_order(
        &self,
        fock_skeleton: &Matrix,
        overlap_deriv: &Matrix,
    ) -> Result<FirstOrderBundle> {
        let n = self.basis.len();
        let zero = Matrix::zeros(n, n);
        let helper = self.helper();

        // Bare response → bare shell charges.
        let (p_bare, _f_bare) = helper.density_response(fock_skeleton, overlap_deriv, &zero)?;
        let q_bare = response_shell_charges_from_density(
            &self.basis,
            &self.overlap,
            &self.density0,
            &p_bare,
            overlap_deriv,
        )?;

        // Screened charges via the factored dielectric, then one exact
        // evaluation at the total (skeleton + screening) perturbation.
        let shell_charges = self.dielectric.solve_vec(&q_bare)?;
        let screened_potential = matrix_vector_product(&self.kernel, &shell_charges)?;
        let response_fock =
            scalar_response_fock_matrix(&self.basis, &self.overlap, &screened_potential)?;
        let (density, occupation_response) =
            helper.density_response(fock_skeleton, overlap_deriv, &response_fock)?;
        let energy_weighted = helper.energy_weighted_response(
            fock_skeleton,
            overlap_deriv,
            &response_fock,
            &occupation_response,
        )?;

        Ok(FirstOrderBundle {
            density,
            energy_weighted,
            shell_charges,
            occupation_response,
            screened_potential,
        })
    }
}

impl ChargeSpaceContext {
    /// Solve the first order and cache the MO-side representations needed by
    /// [`Self::solve_second_order`].
    pub fn first_order_field(
        &self,
        fock_skeleton: Matrix,
        overlap_deriv: Matrix,
    ) -> Result<FirstOrderField> {
        let bundle = self.solve_first_order(&fock_skeleton, &overlap_deriv)?;
        let response_fock =
            scalar_response_fock_matrix(&self.basis, &self.overlap, &bundle.screened_potential)?;
        let mut fock_total = fock_skeleton.clone();
        for (dst, src) in fock_total
            .as_mut_slice()
            .iter_mut()
            .zip(response_fock.as_slice())
        {
            *dst += *src;
        }
        let h_tilde = self.mo_transform(&fock_total)?;
        let s_tilde = self.mo_transform(&overlap_deriv)?;
        let n = self.occupations.len();
        let mut eps_response = vec![0.0_f64; n];
        for p in 0..n {
            eps_response[p] = h_tilde[(p, p)] - self.orbital_energies[p] * s_tilde[(p, p)];
        }
        let mut u_rotation = Matrix::zeros(n, n);
        // Degenerate-branch threshold: at finite temperature the coefficient
        // formula switches branches at 1e-10, and branch CONSISTENCY between
        // the rotation and the coefficients is what makes the second-order
        // frame-rotation/coefficient cancellation exact for near-degenerate
        // pairs — so the rotation must use the same threshold there. At T = 0
        // same-occupation near-degenerate pairs are rotation-invariant in
        // every consumer, so the wider 1e-6 window (numerically gentler) is
        // kept.
        let tol_deg = if self.finite_t { 1.0e-10 } else { 1.0e-6 };
        for p in 0..n {
            u_rotation[(p, p)] = -0.5 * s_tilde[(p, p)];
            for q in 0..n {
                if p == q {
                    continue;
                }
                let de = self.orbital_energies[q] - self.orbital_energies[p];
                if de.abs() < tol_deg {
                    // (Near-)degenerate block: symmetric gauge fixed by
                    // first-order orthonormality (see the FC3 degenerate fix).
                    u_rotation[(p, q)] = -0.5 * s_tilde[(p, q)];
                } else {
                    u_rotation[(p, q)] =
                        (h_tilde[(p, q)] - self.orbital_energies[q] * s_tilde[(p, q)]) / de;
                }
            }
        }
        Ok(FirstOrderField {
            bundle,
            fock_skeleton,
            overlap_deriv,
            h_tilde,
            s_tilde,
            eps_response,
            u_rotation,
        })
    }

    fn mo_transform(&self, ao: &Matrix) -> Result<Matrix> {
        let tmp = ao.matmul(&self.mos)?;
        self.mos.transpose().matmul(&tmp)
    }

    /// Onsite anharmonic kernel-chain potential `(∂K/∂q · q^{(y)}) · u`:
    /// per shell `s`, `E'''_{A(s)} · q^{(y)}_{A(s)} · Σ_{t∈A(s)} u_t`.
    pub(crate) fn kernel_chain_potential(&self, u: &[f64], q_y: &[f64]) -> Vec<f64> {
        let nat = self.kernel_q_atom.len();
        let mut atom_u = vec![0.0_f64; nat];
        let mut atom_qy = vec![0.0_f64; nat];
        for s in 0..u.len() {
            atom_u[self.shell_atom[s]] += u[s];
            atom_qy[self.shell_atom[s]] += q_y[s];
        }
        (0..u.len())
            .map(|s| {
                let a = self.shell_atom[s];
                self.kernel_q_atom[a] * atom_qy[a] * atom_u[a]
            })
            .collect()
    }

    /// **Daleckii–Krein response tables** at the reference spectrum: the
    /// divided differences that weight every term of the contour-integral
    /// (resolvent) form of the response.
    ///
    /// With `G(z) = (zS − H)^{-1}` and `P = (2πi)^{-1} ∮ f(z) G dz`, the
    /// derivatives of `G` are pure resolvent products, so in the MO
    /// representation every response element is a divided difference of `f`
    /// against the reference orbital energies — no frame rotation `U`, no
    /// gauge choice, and no in-block special case: degeneracies are the
    /// confluent limits of the same smooth divided differences.
    fn dk_tables(&self) -> DkTables {
        let n = self.orbital_energies.len();
        let eps = self.orbital_energies.clone();
        let f = self.occupations.clone();
        let fp: Vec<f64> = f
            .iter()
            .map(|&fp| -(fp * (1.0 - 0.5 * fp)).max(0.0) / self.kt)
            .collect();
        let fpp: Vec<f64> = f
            .iter()
            .zip(&fp)
            .map(|(&fq, &fpq)| -fpq * (1.0 - fq) / self.kt)
            .collect();
        let mut f1 = Matrix::zeros(n, n);
        let mut fp1 = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                let d = eps[p] - eps[q];
                if d.abs() > DK_CONFLUENT_EPS {
                    f1[(p, q)] = (f[p] - f[q]) / d;
                    fp1[(p, q)] = (fp[p] - fp[q]) / d;
                } else {
                    f1[(p, q)] = fp[p];
                    fp1[(p, q)] = fpp[p];
                }
            }
        }
        DkTables {
            eps,
            f,
            fp,
            fpp,
            f1,
            fp1,
        }
    }

    /// Second-order MO-representation response of the matrix function
    /// `z^L f(z)` (`L = 0` density, `L = 1` energy-weighted density), from the
    /// second derivative of the resolvent:
    ///
    /// ```text
    ///   d²G = G Bˣ G Bʸ G + G Bʸ G Bˣ G − G Bˣʸ G ,  B = z dS − dH
    /// ```
    ///
    /// Contour-integrating term by term turns `z^k` into the divided
    /// differences of `z^{L+k} f`, built by the Leibniz recursion
    /// `w_{k+1}^{[m]}(p, …, q) = ε_p w_k^{[m]}(p, …, q) + w_k^{[m−1]}(…, q)`.
    /// The chemical-potential response enters as `∂_μ` chains of the same
    /// object (`∂_μ z^L f(z−μ) = −z^L f′`). Returns the `μ^{xy}`-independent
    /// part; the caller adds the diagonal `μ^{xy}` term.
    #[allow(clippy::too_many_arguments)]
    fn dk_second_order_mo(
        &self,
        t: &DkTables,
        level: usize,
        a_x: &Matrix,
        s_x: &Matrix,
        a_y: &Matrix,
        s_y: &Matrix,
        a_xy: &Matrix,
        s_xy: &Matrix,
        mu_x: f64,
        mu_y: f64,
    ) -> Matrix {
        let n = t.eps.len();
        let e = &t.eps;
        // Level-lifted divided differences (Leibniz with the linear factor z).
        // w^{[0]}_k(q) = ε_q^k f_q ; w^{[1]}_k(p, q) ; w^{[2]}_k(p, r, q).
        let pow = |x: f64, k: usize| -> f64 { (0..k).fold(1.0, |acc, _| acc * x) };
        let w0 = |k: usize, q: usize| -> f64 { pow(e[q], k) * t.f[q] };
        let w1 = |k: usize, p: usize, q: usize| -> f64 {
            let mut acc = t.f1[(p, q)];
            for j in 0..k {
                acc = e[p] * acc + w0(j, q);
            }
            acc
        };
        let w2 = |k: usize, p: usize, r: usize, q: usize| -> f64 {
            let mut acc = t.f2(p, r, q);
            for j in 0..k {
                acc = e[p] * acc + w1(j, r, q);
            }
            acc
        };
        // ∂_μ chains: u = −z^k f′.
        let u0 = |k: usize, q: usize| -> f64 { -pow(e[q], k) * t.fp[q] };
        let u1 = |k: usize, p: usize, q: usize| -> f64 {
            let mut acc = -t.fp1[(p, q)];
            for j in 0..k {
                acc = e[p] * acc + u0(j, q);
            }
            acc
        };

        let mut out = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                // Second-order skeleton term: w^{[1]}_L A^{xy} − w^{[1]}_{L+1} S^{xy}.
                let mut acc =
                    w1(level, p, q) * a_xy[(p, q)] - w1(level + 1, p, q) * s_xy[(p, q)];
                // Bilinear resolvent term, both orderings.
                for r in 0..n {
                    let f2 = w2(level, p, r, q);
                    let g2 = w2(level + 1, p, r, q);
                    let k2 = w2(level + 2, p, r, q);
                    acc += f2 * (a_x[(p, r)] * a_y[(r, q)] + a_y[(p, r)] * a_x[(r, q)]);
                    acc -= g2
                        * (s_x[(p, r)] * a_y[(r, q)]
                            + a_x[(p, r)] * s_y[(r, q)]
                            + s_y[(p, r)] * a_x[(r, q)]
                            + a_y[(p, r)] * s_x[(r, q)]);
                    acc += k2 * (s_x[(p, r)] * s_y[(r, q)] + s_y[(p, r)] * s_x[(r, q)]);
                }
                // μ cross chains.
                acc += mu_y * (u1(level, p, q) * a_x[(p, q)] - u1(level + 1, p, q) * s_x[(p, q)]);
                acc += mu_x * (u1(level, p, q) * a_y[(p, q)] - u1(level + 1, p, q) * s_y[(p, q)]);
                out[(p, q)] = acc;
            }
        }
        // μ^x μ^y curvature (diagonal): ∂²_μ z^L f = +z^L f''.
        for p in 0..n {
            out[(p, p)] += mu_x * mu_y * pow(e[p], level) * t.fpp[p];
        }
        out
    }

    /// First-order MO-representation response of `z^L f(z)` — the same
    /// resolvent form one order down, used for the particle-number condition
    /// that fixes `μ^{xy}` and as the first-order consistency check.
    fn dk_first_order_mo(
        &self,
        t: &DkTables,
        level: usize,
        a_x: &Matrix,
        s_x: &Matrix,
        mu_x: f64,
    ) -> Matrix {
        let n = t.eps.len();
        let e = &t.eps;
        let pow = |x: f64, k: usize| -> f64 { (0..k).fold(1.0, |acc, _| acc * x) };
        let w0 = |k: usize, q: usize| -> f64 { pow(e[q], k) * t.f[q] };
        let w1 = |k: usize, p: usize, q: usize| -> f64 {
            let mut acc = t.f1[(p, q)];
            for j in 0..k {
                acc = e[p] * acc + w0(j, q);
            }
            acc
        };
        let mut out = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                out[(p, q)] =
                    w1(level, p, q) * a_x[(p, q)] - w1(level + 1, p, q) * s_x[(p, q)];
            }
            out[(p, p)] += mu_x * (-pow(e[p], level) * t.fp[p]);
        }
        out
    }

    /// The T = 0 response-coefficient matrix `𝒞(h̃, s̃)` (density variant) and
    /// its energy-weighted sibling, split out so the second-order code can
    /// apply the same formula to derivative inputs.
    fn coeff_from_mo(&self, h_tilde: &Matrix, s_tilde: &Matrix, energy_weighted: bool) -> Matrix {
        let n = self.occupations.len();
        let f = &self.occupations;
        let e = &self.orbital_energies;
        let mut c = Matrix::zeros(n, n);
        for p in 0..n {
            c[(p, p)] = if energy_weighted {
                // f_p(h̃_pp − ε_p s̃_pp) − f_p ε_p s̃_pp
                f[p] * (h_tilde[(p, p)] - e[p] * s_tilde[(p, p)]) - f[p] * e[p] * s_tilde[(p, p)]
            } else {
                -f[p] * s_tilde[(p, p)]
            };
            for q in 0..n {
                if p == q {
                    continue;
                }
                let gap = e[p] - e[q];
                let value = if gap.abs() > 1.0e-6 {
                    if energy_weighted {
                        let wp = f[p] * e[p];
                        let wq = f[q] * e[q];
                        ((wp - wq) * h_tilde[(p, q)] - (wp * e[p] - wq * e[q]) * s_tilde[(p, q)])
                            / gap
                    } else {
                        ((f[p] - f[q]) * h_tilde[(p, q)]
                            - (f[p] * e[p] - f[q] * e[q]) * s_tilde[(p, q)])
                            / gap
                    }
                } else {
                    // Degenerate same-occupation block at integer occupations:
                    // the h̃ slope carries f(1−f/2) = 0, leaving the metric term.
                    let fb = 0.5 * (f[p] + f[q]);
                    let eb = 0.5 * (e[p] + e[q]);
                    if energy_weighted {
                        -(2.0 * eb * fb) * s_tilde[(p, q)] + fb * h_tilde[(p, q)]
                    } else {
                        -fb * s_tilde[(p, q)]
                    }
                };
                c[(p, q)] = value;
            }
        }
        c
    }

    /// ε-derivative correction `Δ𝒞` of the coefficient formula along a
    /// perturbation with per-orbital orbital-energy responses `eps_y`
    /// (T = 0: occupations fixed). Degenerate blocks carry no explicit-ε term
    /// in `𝒞`, so they contribute nothing here.
    fn coeff_eps_correction(
        &self,
        h_tilde_x: &Matrix,
        s_tilde_x: &Matrix,
        eps_y: &[f64],
        energy_weighted: bool,
    ) -> Matrix {
        let n = self.occupations.len();
        let f = &self.occupations;
        let e = &self.orbital_energies;
        let mut c = Matrix::zeros(n, n);
        for p in 0..n {
            if energy_weighted {
                // c_pp = f_p h̃_pp − 2 f_p ε_p s̃_pp → Δ = −2 f_p ε^y_p s̃_pp
                c[(p, p)] = -2.0 * f[p] * eps_y[p] * s_tilde_x[(p, p)];
            }
            for q in 0..n {
                if p == q {
                    continue;
                }
                let gap = e[p] - e[q];
                if gap.abs() <= 1.0e-6 {
                    // Degenerate block: the density coefficient carries no
                    // explicit ε; the energy-weighted one is f̄h̃ − 2ε̄f̄s̃, whose
                    // ε̄ derivative is the (gauge-invariant) in-block average.
                    if energy_weighted {
                        let fb = 0.5 * (f[p] + f[q]);
                        let eps_avg = 0.5 * (eps_y[p] + eps_y[q]);
                        c[(p, q)] = -2.0 * fb * eps_avg * s_tilde_x[(p, q)];
                    }
                    continue;
                }
                let dgap = eps_y[p] - eps_y[q];
                let value = if energy_weighted {
                    let wp = f[p] * e[p];
                    let wq = f[q] * e[q];
                    let base = ((wp - wq) * h_tilde_x[(p, q)]
                        - (wp * e[p] - wq * e[q]) * s_tilde_x[(p, q)])
                        / gap;
                    let dwp = f[p] * eps_y[p];
                    let dwq = f[q] * eps_y[q];
                    ((dwp - dwq) * h_tilde_x[(p, q)]
                        - (2.0 * f[p] * e[p] * eps_y[p] - 2.0 * f[q] * e[q] * eps_y[q])
                            * s_tilde_x[(p, q)])
                        / gap
                        - base * dgap / gap
                } else {
                    let base = ((f[p] - f[q]) * h_tilde_x[(p, q)]
                        - (f[p] * e[p] - f[q] * e[q]) * s_tilde_x[(p, q)])
                        / gap;
                    (-(f[p] * eps_y[p] - f[q] * eps_y[q]) * s_tilde_x[(p, q)]) / gap
                        - base * dgap / gap
                };
                c[(p, q)] = value;
            }
        }
        c
    }

    /// Finite-temperature reference-motion correction `Δ𝒞_T` of the response
    /// coefficient formula: the derivative of
    /// [`finite_temperature_response_coefficients_from_mo`] with respect to its
    /// REFERENCE inputs (occupations `f` and orbital energies `ε`) along the
    /// `y` perturbation, holding the linear inputs `(f^{(x)}, h̃^{(x)}, s̃^{(x)})`
    /// fixed. Together with the base formula applied to the derivative inputs
    /// `(f^{(xy)}, ḣ, ṡ)` this is the exact `λ_y`-derivative of the
    /// finite-temperature first-order coefficient matrix.
    ///
    /// Branch structure (and the `1e-10` degeneracy threshold) matches the base
    /// formula exactly, so base + correction differentiates the branch actually
    /// taken. The occupation motion enters as `f^{(y)}_p` (from the bundle, μ
    /// shift folded in) and, in the degenerate slope branch, as the Fermi
    /// second derivative `f''_p = w_p(1 − f_p)/kT` chained with
    /// `ε^{(y)}_p − μ^{(y)}`.
    /// The finite-temperature reference-motion correction with EXPLICIT linear
    /// inputs `(h_lin, s_lin, occ_lin)` (the first-order use passes the `x`
    /// field's members; the third-order assembly reuses the same chains with
    /// the DOTTED linear inputs `(ḣ, ṡ, f^{(vv)})`).
    #[allow(clippy::too_many_arguments)]
    fn coeff_finite_t_reference_correction_lin(
        &self,
        h_lin: &Matrix,
        s_lin: &Matrix,
        occ_lin: &[f64],
        eps_y: &[f64],
        occ_y_in: &[f64],
        energy_weighted: bool,
    ) -> Matrix {
        let n = self.occupations.len();
        let f = &self.occupations;
        let e = &self.orbital_energies;
        let occ_x = occ_lin;
        let occ_y = occ_y_in;
        let ey = eps_y;
        let w: Vec<f64> = f
            .iter()
            .map(|&fp| (fp * (1.0 - 0.5 * fp)).max(0.0) / self.kt)
            .collect();
        let wsum: f64 = w.iter().sum();
        let mu_y = if wsum > 1.0e-30 {
            w.iter().zip(ey).map(|(&wp, &dep)| wp * dep).sum::<f64>() / wsum
        } else {
            0.0
        };
        // f''_p = w_p (1 − f_p)/kT  (with f' = −w).
        let d2f: Vec<f64> = w
            .iter()
            .zip(f)
            .map(|(&wp, &fp)| wp * (1.0 - fp) / self.kt)
            .collect();
        let hx = h_lin;
        let sx = s_lin;
        let mut c = Matrix::zeros(n, n);
        for i in 0..n {
            c[(i, i)] = if energy_weighted {
                // D_y[f_i ε^{(x)}_i + ε_i f^{(x)}_i − f_i ε_i s̃^{(x)}_ii] minus the
                // base formula on the derivative inputs.
                occ_y[i] * (hx[(i, i)] - 2.0 * e[i] * sx[(i, i)])
                    + ey[i] * occ_x[i]
                    - 2.0 * f[i] * ey[i] * sx[(i, i)]
            } else {
                -occ_y[i] * sx[(i, i)]
            };
            for j in i + 1..n {
                let gap = e[i] - e[j];
                let h_ij = hx[(i, j)];
                let s_ij = sx[(i, j)];
                let value = if gap.abs() > 1.0e-10 {
                    let dgap = ey[i] - ey[j];
                    if energy_weighted {
                        let w_i = f[i] * e[i];
                        let w_j = f[j] * e[j];
                        let base = (w_i - w_j) * h_ij / gap - (w_i * e[i] - w_j * e[j]) * s_ij / gap;
                        let wy_i = occ_y[i] * e[i] + f[i] * ey[i];
                        let wy_j = occ_y[j] * e[j] + f[j] * ey[j];
                        let dwe_i = occ_y[i] * e[i] * e[i] + 2.0 * f[i] * e[i] * ey[i];
                        let dwe_j = occ_y[j] * e[j] * e[j] + 2.0 * f[j] * e[j] * ey[j];
                        ((wy_i - wy_j) * h_ij - (dwe_i - dwe_j) * s_ij) / gap - base * dgap / gap
                    } else {
                        let base = (f[i] - f[j]) * h_ij / gap
                            - (f[i] * e[i] - f[j] * e[j]) * s_ij / gap;
                        ((occ_y[i] - occ_y[j]) * h_ij
                            - (occ_y[i] * e[i] + f[i] * ey[i] - occ_y[j] * e[j] - f[j] * ey[j])
                                * s_ij)
                            / gap
                            - base * dgap / gap
                    }
                } else {
                    // Degenerate slope branch: differentiate the averaged slopes.
                    let ebar = 0.5 * (e[i] + e[j]);
                    let fbar = 0.5 * (f[i] + f[j]);
                    let slope_f = -0.5 * (w[i] + w[j]);
                    let fbar_y = 0.5 * (occ_y[i] + occ_y[j]);
                    let ebar_y = 0.5 * (ey[i] + ey[j]);
                    let slope_f_y =
                        0.5 * (d2f[i] * (ey[i] - mu_y) + d2f[j] * (ey[j] - mu_y));
                    if energy_weighted {
                        let slope_w_y = fbar_y + ebar_y * slope_f + ebar * slope_f_y;
                        let slope_eps_w_y = 2.0 * (ebar_y * fbar + ebar * fbar_y)
                            + 2.0 * ebar * ebar_y * slope_f
                            + ebar * ebar * slope_f_y;
                        slope_w_y * h_ij - slope_eps_w_y * s_ij
                    } else {
                        slope_f_y * h_ij
                            - (fbar_y + ebar_y * slope_f + ebar * slope_f_y) * s_ij
                    }
                };
                c[(i, j)] = value;
                c[(j, i)] = value;
            }
        }
        c
    }

    /// Reference-state coefficient matrix of a first-order field: the T = 0
    /// formula at integer occupations, the unified finite-temperature formula
    /// (occupation channel included) otherwise.
    fn ref_coeff(&self, field: &FirstOrderField, energy_weighted: bool) -> Result<Matrix> {
        if self.finite_t {
            finite_temperature_response_coefficients_from_mo(
                &self.occupations,
                &self.orbital_energies,
                &field.bundle.occupation_response,
                &field.h_tilde,
                &field.s_tilde,
                self.kt,
                energy_weighted,
            )
        } else {
            Ok(self.coeff_from_mo(&field.h_tilde, &field.s_tilde, energy_weighted))
        }
    }

    /// The base coefficient formula applied to the DERIVATIVE inputs
    /// `(f^{(xy)}, ḣ, ṡ)` — the linear part of the coefficient's `λ_y`
    /// derivative.
    fn dot_coeff(
        &self,
        h_dot: &Matrix,
        s_dot: &Matrix,
        occ_xy: &[f64],
        energy_weighted: bool,
    ) -> Result<Matrix> {
        if self.finite_t {
            finite_temperature_response_coefficients_from_mo(
                &self.occupations,
                &self.orbital_energies,
                occ_xy,
                h_dot,
                s_dot,
                self.kt,
                energy_weighted,
            )
        } else {
            Ok(self.coeff_from_mo(h_dot, s_dot, energy_weighted))
        }
    }

    /// The reference-motion part of the coefficient's `λ_y` derivative:
    /// ε-denominator chains at T = 0, ε AND occupation chains at finite T.
    fn deriv_correction(
        &self,
        x: &FirstOrderField,
        y: &FirstOrderField,
        energy_weighted: bool,
    ) -> Matrix {
        self.deriv_correction_lin(
            &x.h_tilde,
            &x.s_tilde,
            &x.bundle.occupation_response,
            &y.eps_response,
            &y.bundle.occupation_response,
            energy_weighted,
        )
    }

    /// [`Self::deriv_correction`] with explicit linear inputs AND explicit
    /// reference-motion vectors — the third-order assembly applies the same
    /// chains both to the dotted linear inputs (with the first-order motion)
    /// and to the first-order inputs (with the SECOND-order motion
    /// `ε^{(vv)}/f^{(vv)}`, the `∂𝒞/∂ref · ref''` half of `Δ²𝒞`).
    fn deriv_correction_lin(
        &self,
        h_lin: &Matrix,
        s_lin: &Matrix,
        occ_lin: &[f64],
        eps_y: &[f64],
        occ_y: &[f64],
        energy_weighted: bool,
    ) -> Matrix {
        if self.finite_t {
            self.coeff_finite_t_reference_correction_lin(
                h_lin,
                s_lin,
                occ_lin,
                eps_y,
                occ_y,
                energy_weighted,
            )
        } else {
            self.coeff_eps_correction(h_lin, s_lin, eps_y, energy_weighted)
        }
    }

    /// Contiguous clusters of EXACTLY degenerate (|Δε| < `threshold`),
    /// fractionally occupied (w > 1e-8) orbitals. Orbital energies are sorted
    /// ascending, so transitive clustering is a linear scan. Members of one
    /// cluster share (to `threshold`) `f̄`, `w`, `f''`.
    ///
    /// The production threshold is 1e-10 (matching the first-order
    /// degenerate-branch gauge); the consistency gate widens it artificially
    /// to force NEAR-degenerate pairs through the covariant block channel and
    /// compare against the regular scalar path.
    fn fractional_degenerate_blocks_with(&self, threshold: f64) -> Vec<Vec<usize>> {
        let e = &self.orbital_energies;
        let n = e.len();
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < n {
            let mut end = start;
            while end + 1 < n && (e[end + 1] - e[end]).abs() < threshold {
                end += 1;
            }
            if end > start {
                let w = (self.occupations[start] * (1.0 - 0.5 * self.occupations[start])).max(0.0)
                    / self.kt;
                if w > 1.0e-8 {
                    blocks.push((start..=end).collect());
                }
            }
            start = end + 1;
        }
        blocks
    }

    fn fractional_degenerate_blocks(&self) -> Vec<Vec<usize>> {
        self.fractional_degenerate_blocks_with(1.0e-10)
    }

    /// Spin-summed Fermi occupation `f(ε) = 2/(1 + e^{(ε−μ)/kT})` with
    /// overflow guards, for the divided-difference helpers below.
    fn fermi_at(&self, e: f64) -> f64 {
        let x = (e - self.mu0()) / self.kt;
        if x > 80.0 {
            0.0
        } else if x < -80.0 {
            2.0
        } else {
            2.0 / (1.0 + x.exp())
        }
    }

    /// First divided difference of the grand-canonical Fermi occupation,
    /// `f^[1](a, b) = (f(a) − f(b))/(a − b)`, with the confluent limit
    /// `f'(a) = −f(a)(1 − f(a)/2)/kT` (spin-summed convention `f ∈ [0, 2]`).
    fn fermi_first_divided(&self, ea: f64, eb: f64) -> f64 {
        let fa = self.fermi_at(ea);
        if (ea - eb).abs() < 1.0e-9 {
            return -(fa * (1.0 - 0.5 * fa)).max(0.0) / self.kt;
        }
        let fb = self.fermi_at(eb);
        (fa - fb) / (ea - eb)
    }

    /// The "pinched" second divided difference `f^[2](ε₀, ε_r, ε₀)
    /// = (f^[1](ε₀, ε_r) − f'(ε₀)) / (ε_r − ε₀)`, the weight of an
    /// out-of-block intermediate `r` in the degenerate-block occupation
    /// second derivative. Confluent limit (`ε_r → ε₀`): `½f''(ε₀)`.
    fn fermi_second_divided_pinched(&self, eps0: f64, er: f64) -> f64 {
        let f0 = self.fermi_at(eps0);
        let w0 = (f0 * (1.0 - 0.5 * f0)).max(0.0) / self.kt;
        let d2f0 = w0 * (1.0 - f0) / self.kt;
        if (er - eps0).abs() < 1.0e-7 {
            // Confluent limit ½f''(ε₀), with f' = −w and
            // f'' = −dw/dε = +w(1−f)/kT = +d2f in the code's convention.
            return 0.5 * d2f0;
        }
        (self.fermi_first_divided(eps0, er) - (-w0)) / (er - eps0)
    }

    /// Reference chemical potential of the converged occupations (the Fermi
    /// level the SCC solved for): recovered from any fractionally occupied
    /// orbital; falls back to the HOMO/LUMO midpoint.
    fn mu0(&self) -> f64 {
        for (p, &fp) in self.occupations.iter().enumerate() {
            if fp > 1.0e-6 && fp < 2.0 - 1.0e-6 {
                // f = 2/(1+exp((ε−μ)/kT)) → μ = ε − kT ln(2/f − 1).
                let arg = (2.0 / fp - 1.0).max(1.0e-300);
                return self.orbital_energies[p] - self.kt * arg.ln();
            }
        }
        let mut homo = f64::NEG_INFINITY;
        let mut lumo = f64::INFINITY;
        for (p, &fp) in self.occupations.iter().enumerate() {
            if fp > 1.0 {
                homo = homo.max(self.orbital_energies[p]);
            } else {
                lumo = lumo.min(self.orbital_energies[p]);
            }
        }
        0.5 * (homo + lumo)
    }

    /// Second-order occupation response with the Λ-covariant in-block channel.
    ///
    /// Outside degenerate blocks the scalar chain
    /// `f^{(xy)}_p = f''δε^x_pδε^y_p + f'(ε^{(xy)}_p − μ^{(xy)})` applies. Inside
    /// an exactly degenerate fractionally occupied block the occupation
    /// response is MATRIX-valued (Daleckii–Krein second divided difference —
    /// the in-block products `Λ^x_{pr}Λ^y_{rq}` a scalar chain cannot see):
    ///
    /// ```text
    ///   F^{(xy)}_B = (f''/2){δΛ^y, δΛ^x} + f'(Λ̇^{xy} − μ^{(xy)} I),
    ///   δΛ^z = Λ̃^z − μ^{(z)} I,   Λ̃^z = [h̃^z − ε₀ s̃^z]_B,
    ///   Λ̇^{xy} = [ḣ − ½{Λ̃^y, s̃^x} − ε₀ ṡ]_B ,
    /// ```
    ///
    /// with `μ^{(xy)}` fixed by particle-number conservation over regular AND
    /// block orbitals (block traces are gauge-invariant). Returns the
    /// occupation vector (block members carry `F_B`'s diagonal; the full block
    /// coefficient overwrite happens in
    /// [`Self::apply_covariant_block_coefficients`]) plus the per-block data.
    #[allow(clippy::too_many_arguments)]
    fn occupation_second_with_blocks(
        &self,
        blocks: &[Vec<usize>],
        x: &FirstOrderField,
        y: &FirstOrderField,
        h_dot_fixed: &Matrix,
        s_dot_fixed: &Matrix,
        s_dot_metric: &Matrix,
        eps_xy: &[f64],
    ) -> Result<(Vec<f64>, Vec<BlockSecondData>)> {
        let n = self.occupations.len();
        if !self.finite_t {
            return Ok((vec![0.0; n], Vec::new()));
        }
        let f = &self.occupations;
        let w: Vec<f64> = f
            .iter()
            .map(|&fp| (fp * (1.0 - 0.5 * fp)).max(0.0) / self.kt)
            .collect();
        let wsum: f64 = w.iter().sum();
        if wsum <= 1.0e-30 {
            return Ok((vec![0.0; n], Vec::new()));
        }
        let d2f: Vec<f64> = w
            .iter()
            .zip(f)
            .map(|(&wp, &fp)| wp * (1.0 - fp) / self.kt)
            .collect();
        let ex = &x.eps_response;
        let ey = &y.eps_response;
        let mu_x = w.iter().zip(ex).map(|(&wp, &d)| wp * d).sum::<f64>() / wsum;
        let mu_y = w.iter().zip(ey).map(|(&wp, &d)| wp * d).sum::<f64>() / wsum;

        let mut in_block = vec![false; n];
        for b in blocks {
            for &p in b {
                in_block[p] = true;
            }
        }
        let sub = |m: &Matrix, mem: &[usize]| -> Matrix {
            let k = mem.len();
            let mut o = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    o[(i, j)] = m[(mem[i], mem[j])];
                }
            }
            o
        };
        let anti = |a: &Matrix, b: &Matrix| -> Result<Matrix> {
            let mut ab = a.matmul(b)?;
            let ba = b.matmul(a)?;
            for (dst, src) in ab.as_mut_slice().iter_mut().zip(ba.as_slice()) {
                *dst += *src;
            }
            Ok(ab)
        };

        // Stage 1: block matrices + the μ^{xy}-independent pieces.
        let mut data = Vec::with_capacity(blocks.len());
        for members in blocks {
            let k = members.len();
            let eps0 = self.orbital_energies[members[0]];
            let wp = w[members[0]];
            let d2fp = d2f[members[0]];
            let hx = sub(&x.h_tilde, members);
            let sxm = sub(&x.s_tilde, members);
            let hy = sub(&y.h_tilde, members);
            let sym = sub(&y.s_tilde, members);
            // Operator (Daleckii–Krein) derivatives need the FIXED-basis dots
            // (mo_transform of the second-derivative operators, WITHOUT the
            // U-frame transport, which the caller adds separately).
            let hdot_b = sub(h_dot_fixed, members);
            let sdot_b = sub(s_dot_fixed, members);
            let sdot_metric_b = sub(s_dot_metric, members);
            let mut lx = hx.clone();
            for (dst, src) in lx.as_mut_slice().iter_mut().zip(sxm.as_slice()) {
                *dst -= eps0 * *src;
            }
            let mut ly = hy.clone();
            for (dst, src) in ly.as_mut_slice().iter_mut().zip(sym.as_slice()) {
                *dst -= eps0 * *src;
            }
            let mut dlx = lx.clone();
            let mut dly = ly.clone();
            for i in 0..k {
                dlx[(i, i)] -= mu_x;
                dly[(i, i)] -= mu_y;
            }
            // Λ̇^{xy} = ḣ − ½{Λ̃^y, s̃^x} − ε₀ ṡ, with the anti-commutator
            // taken over ALL orbitals (the in-block-only product was the
            // measured 4.5e-2 failure): Λ̃^y is the full symmetrized
            // Löwdin-frame argument derivative, `Λ̃^y_pr = h̃^y_pr −
            // ½(ε_p+ε_r)s̃^y_pr`, which reduces to `h̃ − ε₀s̃` inside the
            // block.
            let e = &self.orbital_energies;
            let lam_full = |field: &FirstOrderField, p: usize, r: usize| -> f64 {
                field.h_tilde[(p, r)] - 0.5 * (e[p] + e[r]) * field.s_tilde[(p, r)]
            };
            let mut ldot = hdot_b.clone();
            for (i, &p) in members.iter().enumerate() {
                for (j, &q) in members.iter().enumerate() {
                    let mut half = 0.0;
                    for r in 0..n {
                        half += lam_full(y, p, r) * x.s_tilde[(r, q)]
                            + x.s_tilde[(p, r)] * lam_full(y, r, q);
                    }
                    ldot[(i, j)] -= 0.5 * half + eps0 * sdot_b[(i, j)];
                }
            }
            // Occupation-curvature products: in-block intermediates carry the
            // confluent weight ½f''(ε₀) (the anti-commutator of the
            // μ-shifted δΛ), out-of-block intermediates the pinched second
            // divided difference f^[2](ε₀, ε_r, ε₀) — the coupling a purely
            // in-block chain cannot represent.
            let mut prod = anti(&dly, &dlx)?;
            for v in prod.as_mut_slice() {
                *v *= 0.5 * d2fp;
            }
            for (i, &p) in members.iter().enumerate() {
                for (j, &q) in members.iter().enumerate() {
                    let mut cross = 0.0;
                    for r in 0..n {
                        if in_block[r] {
                            continue;
                        }
                        let f2 = self.fermi_second_divided_pinched(eps0, e[r]);
                        cross += f2
                            * (lam_full(x, p, r) * lam_full(y, r, q)
                                + lam_full(y, p, r) * lam_full(x, r, q));
                    }
                    prod[(i, j)] += cross;
                }
            }
            data.push(BlockSecondData {
                members: members.clone(),
                eps0,
                w: wp,
                fbar: f[members[0]],
                lx,
                ly,
                dlx,
                dly,
                ldot,
                sx: sxm,
                sdot: sdot_metric_b,
                prod,
                f_xy: Matrix::zeros(k, k),
            });
        }

        // Stage 2: block-aware μ^{xy} from Σ f^{(xy)} = 0 (f' = −w).
        let mut numer = 0.0;
        for p in 0..n {
            if in_block[p] {
                continue;
            }
            numer += w[p] * eps_xy[p] - d2f[p] * (ex[p] - mu_x) * (ey[p] - mu_y);
        }
        for d in &data {
            let k = d.members.len();
            let tr_ldot: f64 = (0..k).map(|i| d.ldot[(i, i)]).sum();
            let tr_prod: f64 = (0..k).map(|i| d.prod[(i, i)]).sum();
            numer += d.w * tr_ldot - tr_prod;
        }
        let mu_xy = numer / wsum;

        // Stage 3: F^{(xy)}_B and the occupation vector.
        let mut occ = vec![0.0_f64; n];
        for p in 0..n {
            if in_block[p] {
                continue;
            }
            occ[p] = d2f[p] * (ex[p] - mu_x) * (ey[p] - mu_y) - w[p] * (eps_xy[p] - mu_xy);
        }
        for d in &mut data {
            let k = d.members.len();
            let mut f_xy = d.prod.clone();
            for i in 0..k {
                for j in 0..k {
                    f_xy[(i, j)] -= d.w * d.ldot[(i, j)];
                }
                f_xy[(i, i)] += d.w * mu_xy;
            }
            for (i, &p) in d.members.iter().enumerate() {
                occ[p] = f_xy[(i, i)];
            }
            d.f_xy = f_xy;
        }
        Ok((occ, data))
    }

    /// Overwrite the (exactly degenerate, fractionally occupied) block entries
    /// of the second-order coefficient matrices with the Λ-covariant forms:
    ///
    /// ```text
    ///   density: 𝒞_B = F^{(xy)}_B − ½{F^y, s̃^x} − f̄ ṡ ,   F^y = −w δΛ^y ;
    ///   EW:      𝒲_B = f̄ Λ̇^{xy} + ½{Λ̃^x, F^y} + ½{Λ̃^y, F^x} + ε₀ F^{(xy)}
    ///                  − ½{f̄ Λ̃^y + ε₀ F^y, s̃^x} − ε₀ f̄ ṡ .
    /// ```
    fn apply_covariant_block_coefficients(
        &self,
        data: &[BlockSecondData],
        mut c_dot: Option<&mut Matrix>,
        mut cw_dot: Option<&mut Matrix>,
    ) -> Result<()> {
        let anti = |a: &Matrix, b: &Matrix| -> Result<Matrix> {
            let mut ab = a.matmul(b)?;
            let ba = b.matmul(a)?;
            for (dst, src) in ab.as_mut_slice().iter_mut().zip(ba.as_slice()) {
                *dst += *src;
            }
            Ok(ab)
        };
        for d in data {
            let k = d.members.len();
            let mut f_y = d.dly.clone();
            for v in f_y.as_mut_slice() {
                *v *= -d.w;
            }
            // Density block.
            if let Some(c_dot) = c_dot.as_deref_mut() {
                let half_fs = anti(&f_y, &d.sx)?;
                let mut c_b = d.f_xy.clone();
                for idx in 0..k * k {
                    c_b.as_mut_slice()[idx] -=
                        0.5 * half_fs.as_slice()[idx] + d.fbar * d.sdot.as_slice()[idx];
                }
                for (i, &p) in d.members.iter().enumerate() {
                    for (j, &q) in d.members.iter().enumerate() {
                        c_dot[(p, q)] = c_b[(i, j)];
                    }
                }
            }
            // Energy-weighted block.
            if let Some(cw) = cw_dot.as_deref_mut() {
                let mut f_x = d.dlx.clone();
                for v in f_x.as_mut_slice() {
                    *v *= -d.w;
                }
                let t1 = anti(&d.lx, &f_y)?;
                let t2 = anti(&d.ly, &f_x)?;
                let mut inner = d.ly.clone();
                for idx in 0..k * k {
                    inner.as_mut_slice()[idx] =
                        d.fbar * inner.as_slice()[idx] + d.eps0 * f_y.as_slice()[idx];
                }
                let t3 = anti(&inner, &d.sx)?;
                let mut w_b = Matrix::zeros(k, k);
                for idx in 0..k * k {
                    w_b.as_mut_slice()[idx] = d.fbar * d.ldot.as_slice()[idx]
                        + 0.5 * (t1.as_slice()[idx] + t2.as_slice()[idx])
                        + d.eps0 * d.f_xy.as_slice()[idx]
                        - 0.5 * t3.as_slice()[idx]
                        - d.eps0 * d.fbar * d.sdot.as_slice()[idx];
                }
                for (i, &p) in d.members.iter().enumerate() {
                    for (j, &q) in d.members.iter().enumerate() {
                        cw[(p, q)] = w_b[(i, j)];
                    }
                }
            }
        }
        Ok(())
    }

    /// `U^T m + m U` — the frame-rotation part of the derivative of an
    /// MO-representation matrix.
    fn frame_rotate(u: &Matrix, m: &Matrix) -> Result<Matrix> {
        let a = u.transpose().matmul(m)?;
        let b = m.matmul(u)?;
        let mut out = a;
        for (dst, src) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
            *dst += *src;
        }
        Ok(out)
    }

    /// Screened second-order response for the perturbation pair `(x, y)`,
    /// given the skeleton second derivatives `F^{xy}_skel` (frozen density AND
    /// frozen charges) and `S^{xy}`, plus the geometric kernel derivative
    /// `dgamma_y_qx[s] = [(∂γ/∂λ_y) q^{(x)}]_s`.
    ///
    /// Finite temperature is native: the occupation channel enters through the
    /// second-order Fermi response `f^{(xy)}` (an `f''` chain with the
    /// second-order chemical-potential shift `μ^{(xy)}` from particle-number
    /// conservation) plus the occupation motion `Δ𝒞_T` of the coefficient
    /// formula. At integer occupations both channels vanish and the T = 0 path
    /// is used unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_second_order(
        &self,
        x: &FirstOrderField,
        y: &FirstOrderField,
        fock_skeleton_xy: &Matrix,
        overlap_xy: &Matrix,
        dgamma_y_qx: &[f64],
        dgamma_x_qy: &[f64],
    ) -> Result<SecondOrderBundle> {
        Ok(self
            .second_order_field(
                x,
                y,
                fock_skeleton_xy,
                overlap_xy,
                dgamma_y_qx,
                dgamma_x_qy,
            )?
            .bundle)
    }

    /// [`Self::solve_second_order`] plus the second-order MO-representation
    /// objects (`ḣ`, `ṡ`, `ε^{(xy)}`, `U^{(xy)}`) — see [`SecondOrderField`].
    ///
    /// The three extras are the exact `λ_y`-derivatives of the corresponding
    /// [`FirstOrderField`] members of `x`:
    ///
    /// ```text
    ///   ε^{(xy)}_p = ḣ_pp − ε^{(y)}_p s̃^{(x)}_pp − ε_p ṡ_pp
    ///   U^{(xy)}_pq = [ḣ_pq − ε^{(y)}_q s̃^{(x)}_pq − ε_q ṡ_pq]/(ε_q − ε_p)
    ///                 − U^{(x)}_pq (ε^{(y)}_q − ε^{(y)}_p)/(ε_q − ε_p)
    /// ```
    ///
    /// with `−½ṡ_pp` on the diagonal and `−½ṡ_pq` inside (near-)degenerate
    /// blocks, mirroring the first-order gauge. By construction they satisfy
    /// the second-order orthonormality identity `U^{(xy)} + U^{(xy)T} = −ṡ`
    /// (the `λ`-derivative of `U + Uᵀ = −s̃`).
    #[allow(clippy::too_many_arguments)]
    pub fn second_order_field(
        &self,
        x: &FirstOrderField,
        y: &FirstOrderField,
        fock_skeleton_xy: &Matrix,
        overlap_xy: &Matrix,
        dgamma_y_qx: &[f64],
        dgamma_x_qy: &[f64],
    ) -> Result<SecondOrderField> {
        // Exactly degenerate fractionally occupied orbitals are handled by the
        // frame-free resolvent (Daleckii–Krein) form, which `build` switches
        // on for exactly those references — see [`Self::dk_second_order_mo`].
        // The historical coefficient/rotation variants all failed here
        // (in-block-only 4.5e-2, frame-included dots 4.5e-2, full-orbital
        // f^[2] completion P 2.2e1) because the in-block second order is not
        // a function of the arbitrary intra-block eigenbasis in that algebra.
        self.second_order_field_with_blocks(
            x,
            y,
            fock_skeleton_xy,
            overlap_xy,
            dgamma_y_qx,
            dgamma_x_qy,
            Vec::new(),
        )
    }

    /// [`Self::second_order_field`] with the covariant degenerate blocks
    /// injected by the caller — the diagnostics drive the (still
    /// gate-failing) matrix-valued occupation channel through here without
    /// tripping the production guard.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn second_order_field_with_blocks(
        &self,
        x: &FirstOrderField,
        y: &FirstOrderField,
        fock_skeleton_xy: &Matrix,
        overlap_xy: &Matrix,
        dgamma_y_qx: &[f64],
        dgamma_x_qy: &[f64],
        blocks: Vec<Vec<usize>>,
    ) -> Result<SecondOrderField> {
        let n = self.basis.len();
        let nshell = self.basis.shells.len();

        // ---- external (q^{xy}-independent) part of dF^x_tot/dy ----
        // Screening derivative: d/dy[RF_S(K q^x)] =
        //   RF_S((∂_y γ)q^x + ∂K/∂q-chain + K q^{xy}) + RF_{S^y}(K q^x).
        // Reference-potential response inside the x-skeleton: the frozen-charge
        // F^{xy}_skel misses the y-response of the reference potential, whose
        // two channels are the (x ↔ y) mirrors of the screening terms above:
        //   RF_S((∂_x γ)q^y) + RF_{S^x}(K q^y).
        let chain = self.kernel_chain_potential(&x.bundle.shell_charges, &y.bundle.shell_charges);
        let mut dv_ext = vec![0.0_f64; nshell];
        for s in 0..nshell {
            dv_ext[s] = dgamma_y_qx[s] + dgamma_x_qy[s] + chain[s];
        }
        let rf_dv_ext = scalar_response_fock_matrix(&self.basis, &self.overlap, &dv_ext)?;
        let rf_sy_vx = scalar_response_fock_matrix(
            &self.basis,
            &y.overlap_deriv,
            &x.bundle.screened_potential,
        )?;
        let rf_sx_vy = scalar_response_fock_matrix(
            &self.basis,
            &x.overlap_deriv,
            &y.bundle.screened_potential,
        )?;
        // dF_ext = F^{xy}_skel + RF_S(dv_ext) + RF_{S^y}(v^x_scr) + RF_{S^x}(v^y_scr)
        let mut df_ext = fock_skeleton_xy.clone();
        for (dst, ((a, b), c)) in df_ext.as_mut_slice().iter_mut().zip(
            rf_dv_ext
                .as_slice()
                .iter()
                .zip(rf_sy_vx.as_slice())
                .zip(rf_sx_vy.as_slice()),
        ) {
            *dst += a + b + c;
        }

        // ---- derivative MO representations (q^{xy}-independent parts) ----
        // ḣ = Uᵀh̃ + h̃U + C†(dF_ext)C  (+ C†RF(Kq^{xy})C handled via the solve)
        // ṡ = Uᵀs̃ + s̃U + C†S^{xy}C
        let h_dot_fixed_ext = self.mo_transform(&df_ext)?;
        let s_dot_fixed = self.mo_transform(overlap_xy)?;
        let h_dot_ext = {
            let mut m = Self::frame_rotate(&y.u_rotation, &x.h_tilde)?;
            for (dst, src) in m.as_mut_slice().iter_mut().zip(h_dot_fixed_ext.as_slice()) {
                *dst += *src;
            }
            m
        };
        let s_dot = {
            let mut m = Self::frame_rotate(&y.u_rotation, &x.s_tilde)?;
            for (dst, src) in m.as_mut_slice().iter_mut().zip(s_dot_fixed.as_slice()) {
                *dst += *src;
            }
            m
        };

        // ---- second-order orbital energies and occupations (ext pass) ----
        // ε^{(xy)}_p = ḣ_pp − ε^{(y)}_p s̃^{(x)}_pp − ε_p ṡ_pp, evaluated here
        // with the q^{xy}-independent ḣ_ext (the K q^{xy} screening add joins in
        // the final pass); f^{(xy)} follows by the grand-canonical chain.
        let norb = self.occupations.len();
        let eps_first = &y.eps_response;

        // ---- Daleckii–Krein (resolvent) second-order response ----
        // A frame-free alternative to the coefficient/rotation algebra below.
        // Because it is built from divided differences of `f` against the
        // reference spectrum, exactly degenerate orbitals are the confluent
        // limits of the same expressions — no in-block special case. Gated
        // against the frame path on non-degenerate finite-T systems.
        let dk_tables = if self.finite_t {
            Some(self.dk_tables())
        } else {
            None
        };
        let (mu_x, mu_y, w_sum) = if let Some(t) = dk_tables.as_ref() {
            let w: Vec<f64> = t.fp.iter().map(|&v| -v).collect();
            let ws: f64 = w.iter().sum();
            let mx = if ws > 0.0 {
                w.iter().zip(&x.eps_response).map(|(&a, &b)| a * b).sum::<f64>() / ws
            } else {
                0.0
            };
            let my = if ws > 0.0 {
                w.iter().zip(&y.eps_response).map(|(&a, &b)| a * b).sum::<f64>() / ws
            } else {
                0.0
            };
            (mx, my, ws)
        } else {
            (0.0, 0.0, 0.0)
        };
        let dk_response = |a_xy: &Matrix| -> (Matrix, Matrix) {
            let t = dk_tables.as_ref().expect("DK tables");
            let mut p_mo = self.dk_second_order_mo(
                t,
                0,
                &x.h_tilde,
                &x.s_tilde,
                &y.h_tilde,
                &y.s_tilde,
                a_xy,
                &s_dot_fixed,
                mu_x,
                mu_y,
            );
            let mut w_mo = self.dk_second_order_mo(
                t,
                1,
                &x.h_tilde,
                &x.s_tilde,
                &y.h_tilde,
                &y.s_tilde,
                a_xy,
                &s_dot_fixed,
                mu_x,
                mu_y,
            );
            // Particle number: d²Tr[SP] = 0 fixes μ^{xy}.
            let p1x = self.dk_first_order_mo(t, 0, &x.h_tilde, &x.s_tilde, mu_x);
            let p1y = self.dk_first_order_mo(t, 0, &y.h_tilde, &y.s_tilde, mu_y);
            let mut target = 0.0;
            for p in 0..norb {
                target -= self.occupations[p] * s_dot_fixed[(p, p)];
                for q in 0..norb {
                    target -= x.s_tilde[(p, q)] * p1y[(q, p)] + y.s_tilde[(p, q)] * p1x[(q, p)];
                }
            }
            let tr: f64 = (0..norb).map(|p| p_mo[(p, p)]).sum();
            let mu_xy = if w_sum > 0.0 {
                (target - tr) / w_sum
            } else {
                0.0
            };
            for p in 0..norb {
                p_mo[(p, p)] += mu_xy * (-t.fp[p]);
                w_mo[(p, p)] += mu_xy * (-self.orbital_energies[p] * t.fp[p]);
            }
            (p_mo, w_mo)
        };
        let use_dk = dk_tables.is_some() && self.dk_second;
        let eps2_of = |h_dot_m: &Matrix, s_dot_m: &Matrix| -> Vec<f64> {
            (0..norb)
                .map(|p| {
                    h_dot_m[(p, p)]
                        - eps_first[p] * x.s_tilde[(p, p)]
                        - self.orbital_energies[p] * s_dot_m[(p, p)]
                })
                .collect()
        };
        let eps2_ext = eps2_of(&h_dot_ext, &s_dot);
        let (occ_xy_ext, blocks_ext) = self.occupation_second_with_blocks(
            &blocks,
            x,
            y,
            &h_dot_fixed_ext,
            &s_dot_fixed,
            &s_dot,
            &eps2_ext,
        )?;

        // ---- coefficient matrix pieces (q^{xy}-independent) ----
        let c_x = self.ref_coeff(x, false)?;
        let mut c_dot_ext = self.dot_coeff(&h_dot_ext, &s_dot, &occ_xy_ext, false)?;
        {
            let corr = self.deriv_correction(x, y, false);
            for (dst, src) in c_dot_ext.as_mut_slice().iter_mut().zip(corr.as_slice()) {
                *dst += *src;
            }
        }
        self.apply_covariant_block_coefficients(&blocks_ext, Some(&mut c_dot_ext), None)?;
        // P^{xy}_ext = C [U^y c_x + c_x U^{yT} + ċ_ext] C†
        let mut inner_ext = y.u_rotation.matmul(&c_x)?;
        {
            let t = c_x.matmul(&y.u_rotation.transpose())?;
            for (dst, src) in inner_ext.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *dst += *src;
            }
            for (dst, src) in inner_ext
                .as_mut_slice()
                .iter_mut()
                .zip(c_dot_ext.as_slice())
            {
                *dst += *src;
            }
        }
        let p_xy_ext = if use_dk {
            let (p_mo, _) = dk_response(&h_dot_fixed_ext);
            mo_coefficient_matrix_to_ao(&self.mos, &p_mo)?
        } else {
            mo_coefficient_matrix_to_ao(&self.mos, &inner_ext)?
        };

        // ---- dielectric solve for q^{xy} ----
        // q^{xy} = −Tr_s(P^{xy}S) − Tr_s(P^x S^y) − Tr_s(P^y S^x) − Tr_s(P0 S^{xy}),
        // with P^{xy} = P^{xy}_ext + Λ[K q^{xy}] → (I − χ⁰K) q^{xy} = q̃^{xy}.
        let zero = Matrix::zeros(n, n);
        let mut q_tilde = response_shell_charges_from_density(
            &self.basis,
            &self.overlap,
            &self.density0,
            &p_xy_ext,
            overlap_xy,
        )?;
        // Cross terms −Tr_s(P^x S^y) and −Tr_s(P^y S^x): reuse the helper with a
        // zero response density so only the "P·δS" channel fires.
        let cross_xy = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &x.bundle.density,
            &zero,
            &y.overlap_deriv,
        )?;
        let cross_yx = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &y.bundle.density,
            &zero,
            &x.overlap_deriv,
        )?;
        for s in 0..nshell {
            q_tilde[s] += cross_xy[s] + cross_yx[s];
        }
        let q_xy = self.dielectric.solve_vec(&q_tilde)?;

        // ---- final evaluation with the full dv (including K q^{xy}) ----
        let kq_xy = matrix_vector_product(&self.kernel, &q_xy)?;
        let rf_kq = scalar_response_fock_matrix(&self.basis, &self.overlap, &kq_xy)?;
        let rf_kq_mo = self.mo_transform(&rf_kq)?;
        let mut h_dot = h_dot_ext;
        for (dst, src) in h_dot.as_mut_slice().iter_mut().zip(rf_kq_mo.as_slice()) {
            *dst += *src;
        }
        let eps_second = eps2_of(&h_dot, &s_dot);
        let h_dot_fixed = {
            let mut m = h_dot_fixed_ext.clone();
            for (dst, src) in m.as_mut_slice().iter_mut().zip(rf_kq_mo.as_slice()) {
                *dst += *src;
            }
            m
        };
        let (occ_xy, blocks_full) = self.occupation_second_with_blocks(
            &blocks,
            x,
            y,
            &h_dot_fixed,
            &s_dot_fixed,
            &s_dot,
            &eps_second,
        )?;
        let mut c_dot = self.dot_coeff(&h_dot, &s_dot, &occ_xy, false)?;
        {
            let corr = self.deriv_correction(x, y, false);
            for (dst, src) in c_dot.as_mut_slice().iter_mut().zip(corr.as_slice()) {
                *dst += *src;
            }
        }
        self.apply_covariant_block_coefficients(&blocks_full, Some(&mut c_dot), None)?;
        let mut inner = y.u_rotation.matmul(&c_x)?;
        {
            let t = c_x.matmul(&y.u_rotation.transpose())?;
            for (dst, src) in inner.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *dst += *src;
            }
            for (dst, src) in inner.as_mut_slice().iter_mut().zip(c_dot.as_slice()) {
                *dst += *src;
            }
        }
        let dk_final = if use_dk {
            Some(dk_response(&h_dot_fixed))
        } else {
            None
        };
        let density = if let Some((p_mo, _)) = dk_final.as_ref() {
            mo_coefficient_matrix_to_ao(&self.mos, p_mo)?
        } else {
            mo_coefficient_matrix_to_ao(&self.mos, &inner)?
        };

        // ---- energy-weighted second response ----
        let cw_x = self.ref_coeff(x, true)?;
        let mut cw_dot = self.dot_coeff(&h_dot, &s_dot, &occ_xy, true)?;
        {
            let corr = self.deriv_correction(x, y, true);
            for (dst, src) in cw_dot.as_mut_slice().iter_mut().zip(corr.as_slice()) {
                *dst += *src;
            }
        }
        self.apply_covariant_block_coefficients(&blocks_full, None, Some(&mut cw_dot))?;
        let mut inner_w = y.u_rotation.matmul(&cw_x)?;
        {
            let t = cw_x.matmul(&y.u_rotation.transpose())?;
            for (dst, src) in inner_w.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *dst += *src;
            }
            for (dst, src) in inner_w.as_mut_slice().iter_mut().zip(cw_dot.as_slice()) {
                *dst += *src;
            }
        }
        let energy_weighted = if let Some((_, w_mo)) = dk_final.as_ref() {
            mo_coefficient_matrix_to_ao(&self.mos, w_mo)?
        } else {
            mo_coefficient_matrix_to_ao(&self.mos, &inner_w)?
        };

        // ---- second-order MO-representation extras ----
        // Plain λ_y-derivatives of the FirstOrderField members of `x`, taken
        // element-wise in the (λ-dependent) MO basis — the frame rotation is
        // already inside ḣ/ṡ, so no further U-transport is needed here.
        // (`eps_second` was computed above from the full ḣ.)
        let mut u_second = Matrix::zeros(norb, norb);
        // Same branch threshold as `u_rotation` (see first_order_field): the
        // finite-T path needs branch consistency down to the coefficient
        // formula's 1e-10 window.
        let tol_deg = if self.finite_t { 1.0e-10 } else { 1.0e-6 };
        for p in 0..norb {
            u_second[(p, p)] = -0.5 * s_dot[(p, p)];
            for q in 0..norb {
                if p == q {
                    continue;
                }
                let gap = self.orbital_energies[q] - self.orbital_energies[p];
                if gap.abs() < tol_deg {
                    // Same symmetric gauge as the first order: U_pq = −½s̃_pq
                    // holds identically across the (near-)degenerate block, so
                    // its λ-derivative is −½ṡ_pq.
                    u_second[(p, q)] = -0.5 * s_dot[(p, q)];
                } else {
                    let num = h_dot[(p, q)]
                        - eps_first[q] * x.s_tilde[(p, q)]
                        - self.orbital_energies[q] * s_dot[(p, q)];
                    u_second[(p, q)] = num / gap
                        - x.u_rotation[(p, q)] * (eps_first[q] - eps_first[p]) / gap;
                }
            }
        }

        Ok(SecondOrderField {
            bundle: SecondOrderBundle {
                density,
                energy_weighted,
                shell_charges: q_xy,
                occupation_response: occ_xy,
            },
            h_dot,
            s_dot,
            eps_second,
            u_second,
            h_dot_fixed,
            s_dot_fixed,
        })
    }

    /// **Directional THIRD-order screened response** (`x = y = z = v`): the
    /// exact `λ`-derivative of the second-order field along the same `v`,
    /// solved with the SAME factored dielectric.
    ///
    /// Ingredient recursion (all `frame(U, M) = UᵀM + MU`):
    ///
    /// ```text
    ///   s̈ = C†S^{vvv}C + frame(U, C†S^{vv}C) + frame(U̇, s̃) + frame(U, ṡ)
    ///   ḧ = C†F^{vvv}C + frame(U, C†F^{vv}C) + frame(U̇, h̃) + frame(U, ḣ)
    ///   ε^{vvv}_p = ḧ_pp − ε^{vv}_p s̃_pp − 2ε^{v}_p ṡ_pp − ε_p s̈_pp
    ///   V^{vvv}  = geo³ + 3(∂²_vγ)q^v + 3(∂_vγ)q^{vv} + 3E'''q^v∘q^{vv}
    ///            + E''''(q^v)³ + K q^{vvv}
    /// ```
    ///
    /// The coefficient third derivative is assembled as
    /// `base(ḧ, s̈, f^{vvv}) + 2·Δ𝒞(ḣ, ṡ, f^{vv}) + Δ𝒞(h̃, s̃, f^v; ref'' = ε^{vv}/f^{vv})
    /// + Δ²𝒞_quad`, where the LAST term (the second derivative of the
    /// coefficient formula contracted with `(ref')²`) is currently the
    /// [`Self::coeff_second_reference_quadratic`] hook — see its doc for the
    /// completion status. The `fock_skeleton_vvv` input is the FROZEN
    /// (density- and charge-held) directional third AO skeleton with the
    /// GEOMETRIC potential legs folded in (the response-potential crosses are
    /// added here, mirroring [`Self::solve_second_order`]).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_third_order_directional(
        &self,
        v_field: &FirstOrderField,
        vv_field: &SecondOrderField,
        fock_skeleton_vvv: &Matrix,
        overlap_vvv: &Matrix,
        overlap_vv: &Matrix,
        v_pot_geo: &[f64],
        dgamma_v_qv: &[f64],
        dgamma_v_qvv: &[f64],
        d2gamma_vv_qv: &[f64],
    ) -> Result<ThirdOrderBundle> {
        // The resolvent (Daleckii–Krein) rewrite that made the SECOND order
        // basis-independent inside exactly degenerate fractionally occupied
        // blocks stops here: this assembly is still frame-based
        // (`frame_rotate`, `u_second`, the U-transported coefficient chains),
        // and a frame is not well defined when an intra-block rotation leaves
        // the reference invariant. Measured on T_d Ni(CO)₄ at 3000 K the
        // third-order FD gate lands at 3.5e3. Refuse rather than return it.
        if self.finite_t {
            let e = &self.orbital_energies;
            for p in 0..e.len() {
                let wp = (self.occupations[p] * (1.0 - 0.5 * self.occupations[p])).max(0.0);
                if wp < 1.0e-8 * self.kt {
                    continue;
                }
                for q in (p + 1)..e.len() {
                    if (e[p] - e[q]).abs() < 1.0e-10 {
                        return Err(Gfn1Error::InvalidInput(format!(
                            "third-order finite-temperature response: orbitals {p} and {q} are \
                             exactly degenerate (gap {:.1e}) with fractional occupation — the \
                             third-order assembly is still frame-based and the frame is not \
                             defined inside such a block (the second order takes the \
                             frame-free resolvent form instead); break the symmetry or use a \
                             seminumerical path",
                            (e[p] - e[q]).abs()
                        )));
                    }
                }
            }
        }
        let n = self.basis.len();
        let nshell = self.basis.shells.len();
        let norb = self.occupations.len();
        let nat = self.kernel_q_atom.len();

        // ---- onsite chains ----
        let mut atom_qv = vec![0.0_f64; nat];
        let mut atom_qvv = vec![0.0_f64; nat];
        for s in 0..nshell {
            atom_qv[self.shell_atom[s]] += v_field.bundle.shell_charges[s];
            atom_qvv[self.shell_atom[s]] += vv_field.bundle.shell_charges[s];
        }
        let chain2: Vec<f64> = (0..nshell)
            .map(|s| {
                let a = self.shell_atom[s];
                self.kernel_q_atom[a] * atom_qv[a] * atom_qv[a]
            })
            .collect();
        let chain3: Vec<f64> = (0..nshell)
            .map(|s| {
                let a = self.shell_atom[s];
                self.kernel_q_atom[a] * atom_qv[a] * atom_qvv[a]
            })
            .collect();
        let chain4: Vec<f64> = (0..nshell)
            .map(|s| {
                let a = self.shell_atom[s];
                self.kernel_q2_atom[a] * atom_qv[a] * atom_qv[a] * atom_qv[a]
            })
            .collect();

        // ---- external (q^{vvv}-independent) screening pieces ----
        let dv_ext: Vec<f64> = (0..nshell)
            .map(|s| {
                3.0 * dgamma_v_qvv[s] + 3.0 * d2gamma_vv_qv[s] + 3.0 * chain3[s] + chain4[s]
            })
            .collect();
        let rf_dv_ext = scalar_response_fock_matrix(&self.basis, &self.overlap, &dv_ext)?;
        // Mirror metric terms: 3 RF_{S^v}(V^{vv}_resp) + 3 RF_{S^{vv}}(K q^v).
        let kq_vv = matrix_vector_product(&self.kernel, &vv_field.bundle.shell_charges)?;
        let v_resp_vv: Vec<f64> = (0..nshell)
            .map(|s| 2.0 * dgamma_v_qv[s] + chain2[s] + kq_vv[s])
            .collect();
        let rf_sv = scalar_response_fock_matrix(&self.basis, &v_field.overlap_deriv, &v_resp_vv)?;
        let rf_svv = scalar_response_fock_matrix(
            &self.basis,
            overlap_vv,
            &v_field.bundle.screened_potential,
        )?;
        // The geometric skeleton (geo legs, zero charge legs) carries only
        // TWO of the three symmetric-D³ copies of `RF_{S^{vv}}(V^v_geo)` (the
        // A- and G-channels of the scc-scalar third builder coincide there) —
        // supply the third copy explicitly.
        let rf_svv_geo = scalar_response_fock_matrix(&self.basis, overlap_vv, v_pot_geo)?;
        let mut df_ext = fock_skeleton_vvv.clone();
        for (dst, (((a, b), c), d)) in df_ext.as_mut_slice().iter_mut().zip(
            rf_dv_ext
                .as_slice()
                .iter()
                .zip(rf_sv.as_slice())
                .zip(rf_svv.as_slice())
                .zip(rf_svv_geo.as_slice()),
        ) {
            *dst += a + 3.0 * b + 3.0 * c + d;
        }

        // ---- third-order MO representations (recursion) ----
        let u1 = &v_field.u_rotation;
        let u2 = &vv_field.u_second;
        let add3 = |m: &mut Matrix, a: &Matrix, b: &Matrix, c: &Matrix| {
            for idx in 0..m.as_slice().len() {
                m.as_mut_slice()[idx] +=
                    a.as_slice()[idx] + b.as_slice()[idx] + c.as_slice()[idx];
            }
        };
        let s_ddot = {
            let mut m = self.mo_transform(overlap_vvv)?;
            let t1 = Self::frame_rotate(u1, &vv_field.s_dot_fixed)?;
            let t2 = Self::frame_rotate(u2, &v_field.s_tilde)?;
            let t3 = Self::frame_rotate(u1, &vv_field.s_dot)?;
            add3(&mut m, &t1, &t2, &t3);
            m
        };
        let h_ddot_ext = {
            let mut m = self.mo_transform(&df_ext)?;
            let t1 = Self::frame_rotate(u1, &vv_field.h_dot_fixed)?;
            let t2 = Self::frame_rotate(u2, &v_field.h_tilde)?;
            let t3 = Self::frame_rotate(u1, &vv_field.h_dot)?;
            add3(&mut m, &t1, &t2, &t3);
            m
        };

        // ---- third-order orbital energies / occupations ----
        let eps3_of = |h_ddot: &Matrix| -> Vec<f64> {
            (0..norb)
                .map(|p| {
                    h_ddot[(p, p)]
                        - vv_field.eps_second[p] * v_field.s_tilde[(p, p)]
                        - 2.0 * v_field.eps_response[p] * vv_field.s_dot[(p, p)]
                        - self.orbital_energies[p] * s_ddot[(p, p)]
                })
                .collect()
        };
        let occ3_of = |eps3: &[f64]| -> Vec<f64> {
            if !self.finite_t {
                return vec![0.0; norb];
            }
            let f = &self.occupations;
            let w: Vec<f64> = f
                .iter()
                .map(|&fp| (fp * (1.0 - 0.5 * fp)).max(0.0) / self.kt)
                .collect();
            let wsum: f64 = w.iter().sum();
            if wsum <= 1.0e-30 {
                return vec![0.0; norb];
            }
            let d2f: Vec<f64> = w
                .iter()
                .zip(f)
                .map(|(&wp, &fp)| wp * (1.0 - fp) / self.kt)
                .collect();
            // f''' = −w((1−f)² − w·kT)/kT².
            let d3f: Vec<f64> = w
                .iter()
                .zip(f)
                .map(|(&wp, &fp)| {
                    -wp * ((1.0 - fp) * (1.0 - fp) - wp * self.kt) / (self.kt * self.kt)
                })
                .collect();
            let e1 = &v_field.eps_response;
            let e2 = &vv_field.eps_second;
            let mu1 = w.iter().zip(e1).map(|(&wp, &d)| wp * d).sum::<f64>() / wsum;
            // μ^{vv} from Σ f^{vv} = 0 given the stored bundle values.
            let mu2 = {
                let prod: f64 = (0..norb)
                    .map(|p| d2f[p] * (e1[p] - mu1) * (e1[p] - mu1))
                    .sum();
                (w.iter().zip(e2).map(|(&wp, &d)| wp * d).sum::<f64>() - prod) / wsum
            };
            let pre: Vec<f64> = (0..norb)
                .map(|p| {
                    let dx = e1[p] - mu1;
                    let d2 = e2[p] - mu2;
                    d3f[p] * dx * dx * dx + 3.0 * d2f[p] * dx * d2
                })
                .collect();
            let mu3 = (w.iter().zip(eps3).map(|(&wp, &d)| wp * d).sum::<f64>()
                - pre.iter().sum::<f64>())
                / wsum;
            (0..norb)
                .map(|p| pre[p] - w[p] * (eps3[p] - mu3))
                .collect()
        };

        // ---- coefficient pieces (shared by ext and final passes) ----
        // c₁ and ċ (the FINAL second-order coefficient, screening included).
        let build_c1_cdot = |energy_weighted: bool| -> Result<(Matrix, Matrix)> {
            let c1 = self.ref_coeff(v_field, energy_weighted)?;
            let mut cdot = self.dot_coeff(
                &vv_field.h_dot,
                &vv_field.s_dot,
                &vv_field.bundle.occupation_response,
                energy_weighted,
            )?;
            let corr = self.deriv_correction(v_field, v_field, energy_weighted);
            for (dst, src) in cdot.as_mut_slice().iter_mut().zip(corr.as_slice()) {
                *dst += *src;
            }
            Ok((c1, cdot))
        };
        // c̈ from a given ḧ: base + 2Δ𝒞(dotted) + Δ𝒞(ref'') + Δ²𝒞_quad.
        let build_cddot = |h_ddot: &Matrix,
                           occ3: &[f64],
                           energy_weighted: bool|
         -> Result<Matrix> {
            let mut m = self.dot_coeff(h_ddot, &s_ddot, occ3, energy_weighted)?;
            let d1 = self.deriv_correction_lin(
                &vv_field.h_dot,
                &vv_field.s_dot,
                &vv_field.bundle.occupation_response,
                &v_field.eps_response,
                &v_field.bundle.occupation_response,
                energy_weighted,
            );
            let d2 = self.deriv_correction_lin(
                &v_field.h_tilde,
                &v_field.s_tilde,
                &v_field.bundle.occupation_response,
                &vv_field.eps_second,
                &vv_field.bundle.occupation_response,
                energy_weighted,
            );
            let dq = self.coeff_second_reference_quadratic(v_field, energy_weighted);
            for idx in 0..m.as_slice().len() {
                m.as_mut_slice()[idx] +=
                    2.0 * d1.as_slice()[idx] + d2.as_slice()[idx] + dq.as_slice()[idx];
            }
            Ok(m)
        };
        // I₃ = U I₂ + I₂ Uᵀ + U̇c₁ + c₁U̇ᵀ + Uċ + ċUᵀ + c̈.
        let build_inner3 = |c1: &Matrix, cdot: &Matrix, cddot: &Matrix| -> Result<Matrix> {
            let mut i2 = u1.matmul(c1)?;
            {
                let t = c1.matmul(&u1.transpose())?;
                for (dst, src) in i2.as_mut_slice().iter_mut().zip(t.as_slice()) {
                    *dst += *src;
                }
                for (dst, src) in i2.as_mut_slice().iter_mut().zip(cdot.as_slice()) {
                    *dst += *src;
                }
            }
            let mut i3 = u1.matmul(&i2)?;
            let t1 = i2.matmul(&u1.transpose())?;
            let t2 = u2.matmul(c1)?;
            let t3 = c1.matmul(&u2.transpose())?;
            let t4 = u1.matmul(cdot)?;
            let t5 = cdot.matmul(&u1.transpose())?;
            for idx in 0..i3.as_slice().len() {
                i3.as_mut_slice()[idx] += t1.as_slice()[idx]
                    + t2.as_slice()[idx]
                    + t3.as_slice()[idx]
                    + t4.as_slice()[idx]
                    + t5.as_slice()[idx]
                    + cddot.as_slice()[idx];
            }
            Ok(i3)
        };

        // ---- ext pass: solve for q^{vvv} ----
        let (c1, cdot) = build_c1_cdot(false)?;
        let eps3_ext = eps3_of(&h_ddot_ext);
        let occ3_ext = occ3_of(&eps3_ext);
        let cddot_ext = build_cddot(&h_ddot_ext, &occ3_ext, false)?;
        let i3_ext = build_inner3(&c1, &cdot, &cddot_ext)?;
        let p_ext = mo_coefficient_matrix_to_ao(&self.mos, &i3_ext)?;
        let zero = Matrix::zeros(n, n);
        let mut q_tilde = response_shell_charges_from_density(
            &self.basis,
            &self.overlap,
            &self.density0,
            &p_ext,
            overlap_vvv,
        )?;
        let cross_vv_v = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &vv_field.bundle.density,
            &zero,
            &v_field.overlap_deriv,
        )?;
        let cross_v_vv = response_shell_charges_from_density(
            &self.basis,
            &zero,
            &v_field.bundle.density,
            &zero,
            overlap_vv,
        )?;
        for s in 0..nshell {
            q_tilde[s] += 3.0 * cross_vv_v[s] + 3.0 * cross_v_vv[s];
        }
        let q_vvv = self.dielectric.solve_vec(&q_tilde)?;

        // ---- final pass with the full screening ----
        let kq3 = matrix_vector_product(&self.kernel, &q_vvv)?;
        let rf_kq3 = scalar_response_fock_matrix(&self.basis, &self.overlap, &kq3)?;
        let rf_kq3_mo = self.mo_transform(&rf_kq3)?;
        let mut h_ddot = h_ddot_ext;
        for (dst, src) in h_ddot.as_mut_slice().iter_mut().zip(rf_kq3_mo.as_slice()) {
            *dst += *src;
        }
        let eps3 = eps3_of(&h_ddot);
        let occ3 = occ3_of(&eps3);
        let cddot = build_cddot(&h_ddot, &occ3, false)?;
        let i3 = build_inner3(&c1, &cdot, &cddot)?;
        let density = mo_coefficient_matrix_to_ao(&self.mos, &i3)?;

        let (cw1, cwdot) = build_c1_cdot(true)?;
        let cwddot = build_cddot(&h_ddot, &occ3, true)?;
        let i3w = build_inner3(&cw1, &cwdot, &cwddot)?;
        let energy_weighted = mo_coefficient_matrix_to_ao(&self.mos, &i3w)?;

        Ok(ThirdOrderBundle {
            density,
            energy_weighted,
            shell_charges: q_vvv,
            occupation_response: occ3,
        })
    }

    /// The QUADRATIC second reference-motion correction `∂Δ𝒞/∂ref · ref'` of
    /// the coefficient formula at FIXED linear inputs and FIXED motion
    /// arguments — the last piece of the third-order coefficient derivative.
    ///
    /// Computed EXACTLY by dual-number lifting: the reference `(f_p, ε_p)` is
    /// seeded with `(f^{(v)}_p, ε^{(v)}_p)` and the correction formula is
    /// re-evaluated in dual arithmetic; the derivative part is the quadratic
    /// correction. (T = 0 closed-form cross-check for the density quotient
    /// branch: `Δ² = −2·(δε^v_{pq}/δε_{pq})·Δ𝒞_{pq}` — pinned by a unit test.)
    fn coeff_second_reference_quadratic(
        &self,
        v_field: &FirstOrderField,
        energy_weighted: bool,
    ) -> Matrix {
        let n = self.occupations.len();
        let e_seed = &v_field.eps_response;
        let f_seed = &v_field.bundle.occupation_response;
        let e_dual: Vec<Dual> = (0..n)
            .map(|p| Dual::new(self.orbital_energies[p], e_seed[p]))
            .collect();
        if self.finite_t {
            let f_dual: Vec<Dual> = (0..n)
                .map(|p| Dual::new(self.occupations[p], f_seed[p]))
                .collect();
            finite_t_reference_correction_dual(
                &f_dual,
                &e_dual,
                self.kt,
                &v_field.bundle.occupation_response,
                &v_field.h_tilde,
                &v_field.s_tilde,
                &v_field.eps_response,
                &v_field.bundle.occupation_response,
                energy_weighted,
            )
        } else {
            eps_correction_dual(
                &self.occupations,
                &e_dual,
                &v_field.h_tilde,
                &v_field.s_tilde,
                &v_field.eps_response,
                energy_weighted,
            )
        }
    }
}

/// The caller-side inputs of [`ChargeSpaceContext::solve_third_order_directional`]:
/// the frozen directional third skeleton (geo legs + the CN cache motion of
/// the bare-H0 second), the overlap ladders, and the γ-derivative legs.
pub struct ThirdOrderInputs {
    pub fock_skeleton_vvv: Matrix,
    pub overlap_vvv: Matrix,
    pub overlap_vv: Matrix,
    pub v_pot_geo: Vec<f64>,
    pub dgamma_v_qv: Vec<f64>,
    pub dgamma_v_qvv: Vec<f64>,
    pub d2gamma_vv_qv: Vec<f64>,
}

/// Build [`ThirdOrderInputs`] at the reference geometry (shared by the
/// production fourth-order assembly and the third-order FD gate).
pub fn directional_third_order_inputs(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicResult,
    coordination_cutoff: f64,
    field: &FirstOrderField,
    second: &SecondOrderField,
    v: &[f64],
) -> Result<ThirdOrderInputs> {
    let basis = &electronic.basis;
    let n = basis.len();
    let nshell = basis.shells.len();
    let ndof = 3 * system.atoms.len();
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
    let v_pot_geo: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
        .collect();
    let v_geo2: Vec<f64> = (0..nshell)
        .map(|s| {
            (0..ndof)
                .map(|c| {
                    (0..ndof)
                        .map(|d| v[c] * v[d] * d2vdr_q[s][(c, d)])
                        .sum::<f64>()
                })
                .sum()
        })
        .collect();
    let zeros = vec![0.0_f64; nshell];
    let mut fock_skeleton_vvv =
        crate::hessian::directional_h0_bare_third_matrix(system, params, electronic, v)?;
    {
        let cn = crate::hessian::directional_h0_cn_block_third_matrix(
            system,
            params,
            electronic,
            coordination_cutoff,
            v,
        )?;
        let scc = crate::hessian::directional_h0_scc_scalar_third_matrix(
            system, params, electronic, v, &v_pot_geo, &v_geo2, &zeros, &zeros,
        )?;
        // CN cache motion of h0_bare² (affine self-energy trick).
        let cn_v_resp: Vec<f64> = {
            let nat = system.atoms.len();
            let cn_grad = crate::hessian::cn_gradient_matrix(system, coordination_cutoff)?;
            (0..nat)
                .map(|at| (0..ndof).map(|c| v[c] * cn_grad[at][c]).sum())
                .collect()
        };
        let cn_cache = {
            let mut el_cnv = electronic.clone();
            el_cnv.coordination_numbers = cn_v_resp;
            let a = crate::hessian::directional_h0_bare_second_matrix(
                system, params, &el_cnv, v,
            )?;
            let mut el_cn0 = electronic.clone();
            el_cn0.coordination_numbers = vec![0.0; system.atoms.len()];
            let b = crate::hessian::directional_h0_bare_second_matrix(
                system, params, &el_cn0, v,
            )?;
            let mut m = a;
            for k in 0..n * n {
                m.as_mut_slice()[k] -= b.as_slice()[k];
            }
            m
        };
        for k in 0..n * n {
            fock_skeleton_vvv.as_mut_slice()[k] +=
                cn.as_slice()[k] + scc.as_slice()[k] + cn_cache.as_slice()[k];
        }
    }
    let overlap_vvv = crate::hessian::directional_overlap_third_matrix(system, basis, v)?;
    let overlap_vv = crate::hessian::directional_overlap_second_matrix(system, basis, v)?;
    let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &field.bundle.shell_charges,
        params,
    )?;
    let dgamma_v_qv: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
        .collect();
    let dgamma_qvv = crate::hessian::shell_scalar_potential_first_derivatives(
        system,
        basis,
        &second.bundle.shell_charges,
        params,
    )?;
    let dgamma_v_qvv: Vec<f64> = (0..nshell)
        .map(|s| (0..ndof).map(|c| v[c] * dgamma_qvv[(s, c)]).sum())
        .collect();
    let d2vdr_qv = crate::hessian::shell_scalar_potential_second_derivatives(
        system,
        basis,
        &field.bundle.shell_charges,
        params,
    )?;
    let d2gamma_vv_qv: Vec<f64> = (0..nshell)
        .map(|s| {
            (0..ndof)
                .map(|c| {
                    (0..ndof)
                        .map(|d| v[c] * v[d] * d2vdr_qv[s][(c, d)])
                        .sum::<f64>()
                })
                .sum()
        })
        .collect();
    Ok(ThirdOrderInputs {
        fock_skeleton_vvv,
        overlap_vvv,
        overlap_vv,
        v_pot_geo,
        dgamma_v_qv,
        dgamma_v_qvv,
        d2gamma_vv_qv,
    })
}

/// Minimal forward-mode dual number for the quadratic reference-motion
/// correction (value + directional derivative).
#[derive(Clone, Copy, Debug)]
struct Dual {
    v: f64,
    d: f64,
}

impl Dual {
    #[inline]
    fn new(v: f64, d: f64) -> Self {
        Self { v, d }
    }
    #[inline]
    fn c(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
}

impl core::ops::Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual::new(self.v + o.v, self.d + o.d)
    }
}
impl core::ops::Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual::new(self.v - o.v, self.d - o.d)
    }
}
impl core::ops::Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        Dual::new(self.v * o.v, self.d * o.v + self.v * o.d)
    }
}
impl core::ops::Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        Dual::new(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v * o.v))
    }
}

/// Dual-lifted [`ChargeSpaceContext::coeff_eps_correction`] (T = 0): the
/// derivative part w.r.t. the seeded orbital energies. Branches follow the
/// VALUE parts, matching the original's branch structure exactly.
fn eps_correction_dual(
    occupations: &[f64],
    e: &[Dual],
    h_tilde_x: &Matrix,
    s_tilde_x: &Matrix,
    eps_y: &[f64],
    energy_weighted: bool,
) -> Matrix {
    let n = occupations.len();
    let f = occupations;
    let mut c = Matrix::zeros(n, n);
    for p in 0..n {
        if energy_weighted {
            // −2 f_p ε^y_p s̃_pp: ε-independent → derivative 0. (No entry.)
        }
        for q in 0..n {
            if p == q {
                continue;
            }
            let gap = e[p] - e[q];
            if gap.v.abs() <= 1.0e-6 {
                if energy_weighted {
                    // −2 f̄ ε̄^y s̃: reference-independent → 0.
                }
                continue;
            }
            let dgap = Dual::c(eps_y[p] - eps_y[q]);
            let h_pq = Dual::c(h_tilde_x[(p, q)]);
            let s_pq = Dual::c(s_tilde_x[(p, q)]);
            let value = if energy_weighted {
                let wp = Dual::c(f[p]) * e[p];
                let wq = Dual::c(f[q]) * e[q];
                let base = ((wp - wq) * h_pq - (wp * e[p] - wq * e[q]) * s_pq) / gap;
                let dwp = Dual::c(f[p] * eps_y[p]);
                let dwq = Dual::c(f[q] * eps_y[q]);
                ((dwp - dwq) * h_pq
                    - (Dual::c(2.0 * f[p] * eps_y[p]) * e[p]
                        - Dual::c(2.0 * f[q] * eps_y[q]) * e[q])
                        * s_pq)
                    / gap
                    - base * dgap / gap
            } else {
                let base = (Dual::c(f[p] - f[q]) * h_pq
                    - (Dual::c(f[p]) * e[p] - Dual::c(f[q]) * e[q]) * s_pq)
                    / gap;
                (Dual::c(0.0) - Dual::c(f[p] * eps_y[p] - f[q] * eps_y[q]) * s_pq) / gap
                    - base * dgap / gap
            };
            c[(p, q)] = value.d;
        }
    }
    c
}

/// Dual-lifted [`ChargeSpaceContext::coeff_finite_t_reference_correction_lin`]:
/// the derivative part w.r.t. the seeded reference occupations AND orbital
/// energies (the Fermi weights, `f''`, and the μ^{(y)} shift all inherit the
/// motion through the dual arithmetic).
#[allow(clippy::too_many_arguments)]
fn finite_t_reference_correction_dual(
    f: &[Dual],
    e: &[Dual],
    kt: f64,
    _occ_ref: &[f64],
    h_tilde_x: &Matrix,
    s_tilde_x: &Matrix,
    eps_y: &[f64],
    occ_y: &[f64],
    energy_weighted: bool,
) -> Matrix {
    let n = f.len();
    let ktd = Dual::c(kt);
    let half = Dual::c(0.5);
    let one = Dual::c(1.0);
    let two = Dual::c(2.0);
    let w: Vec<Dual> = f
        .iter()
        .map(|&fp| {
            let raw = fp * (one - half * fp) / ktd;
            if raw.v < 0.0 {
                Dual::c(0.0)
            } else {
                raw
            }
        })
        .collect();
    let mut wsum = Dual::c(0.0);
    for &wp in &w {
        wsum = wsum + wp;
    }
    let mu_y = if wsum.v > 1.0e-30 {
        let mut acc = Dual::c(0.0);
        for p in 0..n {
            acc = acc + w[p] * Dual::c(eps_y[p]);
        }
        acc / wsum
    } else {
        Dual::c(0.0)
    };
    let d2f: Vec<Dual> = (0..n).map(|p| w[p] * (one - f[p]) / ktd).collect();
    let mut c = Matrix::zeros(n, n);
    for i in 0..n {
        let value_diag = if energy_weighted {
            Dual::c(occ_y[i])
                * (Dual::c(h_tilde_x[(i, i)]) - two * e[i] * Dual::c(s_tilde_x[(i, i)]))
                + Dual::c(eps_y[i] * occ_y[i])
                - two * f[i] * Dual::c(eps_y[i] * s_tilde_x[(i, i)])
        } else {
            Dual::c(0.0) - Dual::c(occ_y[i] * s_tilde_x[(i, i)])
        };
        c[(i, i)] = value_diag.d;
        for j in i + 1..n {
            let gap = e[i] - e[j];
            let h_ij = Dual::c(h_tilde_x[(i, j)]);
            let s_ij = Dual::c(s_tilde_x[(i, j)]);
            let value = if gap.v.abs() > 1.0e-10 {
                let dgap = Dual::c(eps_y[i] - eps_y[j]);
                if energy_weighted {
                    let w_i = f[i] * e[i];
                    let w_j = f[j] * e[j];
                    let base =
                        ((w_i - w_j) * h_ij - (w_i * e[i] - w_j * e[j]) * s_ij) / gap;
                    let wy_i = Dual::c(occ_y[i]) * e[i] + f[i] * Dual::c(eps_y[i]);
                    let wy_j = Dual::c(occ_y[j]) * e[j] + f[j] * Dual::c(eps_y[j]);
                    let dwe_i = Dual::c(occ_y[i]) * e[i] * e[i]
                        + two * f[i] * e[i] * Dual::c(eps_y[i]);
                    let dwe_j = Dual::c(occ_y[j]) * e[j] * e[j]
                        + two * f[j] * e[j] * Dual::c(eps_y[j]);
                    ((wy_i - wy_j) * h_ij - (dwe_i - dwe_j) * s_ij) / gap - base * dgap / gap
                } else {
                    let base = ((f[i] - f[j]) * h_ij
                        - (f[i] * e[i] - f[j] * e[j]) * s_ij)
                        / gap;
                    ((Dual::c(occ_y[i] - occ_y[j])) * h_ij
                        - (Dual::c(occ_y[i]) * e[i] + f[i] * Dual::c(eps_y[i])
                            - Dual::c(occ_y[j]) * e[j]
                            - f[j] * Dual::c(eps_y[j]))
                            * s_ij)
                        / gap
                        - base * dgap / gap
                }
            } else {
                let ebar = half * (e[i] + e[j]);
                let fbar = half * (f[i] + f[j]);
                let slope_f = Dual::c(0.0) - half * (w[i] + w[j]);
                let fbar_y = Dual::c(0.5 * (occ_y[i] + occ_y[j]));
                let ebar_y = Dual::c(0.5 * (eps_y[i] + eps_y[j]));
                let slope_f_y = half
                    * (d2f[i] * (Dual::c(eps_y[i]) - mu_y)
                        + d2f[j] * (Dual::c(eps_y[j]) - mu_y));
                if energy_weighted {
                    let slope_w_y = fbar_y + ebar_y * slope_f + ebar * slope_f_y;
                    let slope_eps_w_y = two * (ebar_y * fbar + ebar * fbar_y)
                        + two * ebar * ebar_y * slope_f
                        + ebar * ebar * slope_f_y;
                    slope_w_y * h_ij - slope_eps_w_y * s_ij
                } else {
                    slope_f_y * h_ij - (fbar_y + ebar_y * slope_f + ebar * slope_f_y) * s_ij
                }
            };
            c[(i, j)] = value.d;
            c[(j, i)] = value.d;
        }
    }
    c
}

/// The unified (T = 0 and finite-T) spectral response evaluation at a FIXED
/// total perturbation — thin composition of the [`super::cpxtb`] primitives.
struct ResponseHelper<'a> {
    mos: &'a Matrix,
    orbital_energies: &'a [f64],
    occupations: &'a [f64],
    kt: f64,
    finite_t: bool,
}

impl ResponseHelper<'_> {
    /// `kt` fed to the coefficient formula: only its degenerate-pair branch
    /// divides by `kt`, and at integer occupations those slope terms carry the
    /// factor `f(1 − f/2) = 0`, so any positive dummy value is exact there.
    #[inline]
    fn kt_for_formula(&self) -> f64 {
        if self.finite_t {
            self.kt
        } else {
            1.0
        }
    }

    fn density_response(
        &self,
        fock_deriv: &Matrix,
        overlap_deriv: &Matrix,
        response_fock: &Matrix,
    ) -> Result<(Matrix, Vec<f64>)> {
        let (h_mo, s_mo) = finite_temperature_mo_derivatives(
            self.mos,
            fock_deriv,
            overlap_deriv,
            response_fock,
        )?;
        let eps_resp = orbital_energy_response_from_mo(self.orbital_energies, &h_mo, &s_mo)?;
        let occ_resp = if self.finite_t {
            fermi_occupation_response(self.occupations, &eps_resp, self.kt)?
        } else {
            vec![0.0; self.occupations.len()]
        };
        let coeff = finite_temperature_response_coefficients_from_mo(
            self.occupations,
            self.orbital_energies,
            &occ_resp,
            &h_mo,
            &s_mo,
            self.kt_for_formula(),
            false,
        )?;
        Ok((mo_coefficient_matrix_to_ao(self.mos, &coeff)?, occ_resp))
    }

    fn energy_weighted_response(
        &self,
        fock_deriv: &Matrix,
        overlap_deriv: &Matrix,
        response_fock: &Matrix,
        occupation_response: &[f64],
    ) -> Result<Matrix> {
        let (h_mo, s_mo) = finite_temperature_mo_derivatives(
            self.mos,
            fock_deriv,
            overlap_deriv,
            response_fock,
        )?;
        let coeff = finite_temperature_response_coefficients_from_mo(
            self.occupations,
            self.orbital_energies,
            occupation_response,
            &h_mo,
            &s_mo,
            self.kt_for_formula(),
            true,
        )?;
        mo_coefficient_matrix_to_ao(self.mos, &coeff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electronic::{run_electronic, ElectronicOptions};
    use crate::response::cpxtb::{
        solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions,
    };

    fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn compare_against_cpxtb(xyz: &str, options: ElectronicOptions, tol: f64) {
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        let ndof = 3 * system.atoms.len();
        let mut worst_p = 0.0_f64;
        let mut worst_w = 0.0_f64;
        let mut worst_q = 0.0_f64;
        for x in 0..ndof {
            let bundle = ctx
                .solve_first_order(
                    &cpxtb.derivative_matrices[x].h0_deriv,
                    &cpxtb.derivative_matrices[x].overlap_deriv,
                )
                .unwrap();
            worst_p = worst_p.max(max_abs_diff(&bundle.density, &cpxtb.density_responses[x]));
            worst_w = worst_w.max(max_abs_diff(
                &bundle.energy_weighted,
                &cpxtb.energy_weighted_density_responses[x],
            ));
            let dq = bundle
                .shell_charges
                .iter()
                .zip(&cpxtb.shell_charge_responses[x])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            worst_q = worst_q.max(dq);
        }
        assert!(
            worst_p < tol && worst_w < tol && worst_q < tol,
            "charge-space vs CPXTB first order: P {worst_p:.3e}  W {worst_w:.3e}  q {worst_q:.3e} (tol {tol:.1e})"
        );
    }

    /// T = 0 (gapped molecule at the default 300 K → integer occupations): the
    /// direct dielectric solve must reproduce the MO-pair CPXTB bundles.
    #[test]
    fn first_order_matches_cpxtb_water_t0() {
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-11;
        options.charge_tolerance = 1.0e-9;
        compare_against_cpxtb(
            "3\nwater\nO 0.0 0.0 0.119262\nH 0.0 0.763239 -0.477047\nH 0.0 -0.763239 -0.477047\n",
            options,
            2.0e-9,
        );
    }

    /// Second-order gate: `P^{xy}`, `W^{xy}`, `q^{xy}` must match the central
    /// finite difference of the screened FIRST-order fields along `y`
    /// (both the perturbation definition and the reference state move with the
    /// geometry — exactly the total derivative the quartic assembly needs).
    #[test]
    fn second_order_matches_first_order_finite_difference() {
        let xyz = "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
        };
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();

        // Geometric first derivative of the reference SCC potential (frozen q).
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nshell = electronic.basis.shells.len();

        let field_for = |dof: usize| -> FirstOrderField {
            ctx.first_order_field(
                cpxtb.derivative_matrices[dof].h0_deriv.clone(),
                cpxtb.derivative_matrices[dof].overlap_deriv.clone(),
            )
            .unwrap()
        };

        let h = 1.0e-4;
        let mut worst_p = 0.0_f64;
        let mut worst_w = 0.0_f64;
        let mut worst_q = 0.0_f64;
        for &(dof_x, dof_y) in &[(0usize, 0usize), (1, 4), (2, 7), (5, 5), (4, 1)] {
            let fx = field_for(dof_x);
            let fy = field_for(dof_y);

            // Skeleton second derivatives (frozen density AND frozen charges).
            let mut f_xy = crate::hessian::h0_bare_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                dof_x,
                dof_y,
            )
            .unwrap();
            let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                cutoff,
                dof_x,
                dof_y,
            )
            .unwrap();
            let v_geo_y: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, dof_y)]).collect();
            let scc_block = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                &v_geo_y,
                &vec![0.0; nshell],
                dof_x,
                dof_y,
            )
            .unwrap();
            for (dst, (a, b)) in f_xy
                .as_mut_slice()
                .iter_mut()
                .zip(cn_block.as_slice().iter().zip(scc_block.as_slice()))
            {
                *dst += a + b;
            }
            let s_xy = crate::response::cpxtb::overlap_second_derivative_matrix(
                &system,
                &electronic.basis,
                dof_x,
                dof_y,
            )
            .unwrap();
            let dgamma_y_qx = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fx.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_y)]).collect::<Vec<f64>>()
            };

            let dgamma_x_qy = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fy.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_x)]).collect::<Vec<f64>>()
            };
            let second = ctx
                .solve_second_order(&fx, &fy, &f_xy, &s_xy, &dgamma_y_qx, &dgamma_x_qy)
                .unwrap();

            // FD reference: displace along dof_y, rebuild EVERYTHING, resolve x.
            let displace = |sign: f64| -> FirstOrderBundle {
                let mut sys = system.clone();
                let (atom, axis) = (dof_y / 3, dof_y % 3);
                match axis {
                    0 => sys.atoms[atom].position.x += sign * h,
                    1 => sys.atoms[atom].position.y += sign * h,
                    _ => sys.atoms[atom].position.z += sign * h,
                }
                let el = run_electronic(&sys, &params, options.clone()).unwrap();
                let cp = solve_nonpbc_cpxtb_hessian_response(
                    &sys,
                    &params,
                    &el,
                    ao_opts,
                    CpxtbOptions::default(),
                )
                .unwrap();
                let c = ChargeSpaceContext::build(&sys, &params, &el).unwrap();
                c.solve_first_order(
                    &cp.derivative_matrices[dof_x].h0_deriv,
                    &cp.derivative_matrices[dof_x].overlap_deriv,
                )
                .unwrap()
            };
            let plus = displace(1.0);
            let minus = displace(-1.0);
            for i in 0..second.density.as_slice().len() {
                let fd =
                    (plus.density.as_slice()[i] - minus.density.as_slice()[i]) / (2.0 * h);
                worst_p = worst_p.max((second.density.as_slice()[i] - fd).abs());
                let fdw = (plus.energy_weighted.as_slice()[i]
                    - minus.energy_weighted.as_slice()[i])
                    / (2.0 * h);
                worst_w = worst_w.max((second.energy_weighted.as_slice()[i] - fdw).abs());
            }
            for s in 0..nshell {
                let fd = (plus.shell_charges[s] - minus.shell_charges[s]) / (2.0 * h);
                worst_q = worst_q.max((second.shell_charges[s] - fd).abs());
            }
        }
        eprintln!(
            "second-order vs FD(first-order): P {worst_p:.3e}  W {worst_w:.3e}  q {worst_q:.3e}"
        );
        assert!(
            worst_p < 1.0e-6 && worst_w < 1.0e-6 && worst_q < 1.0e-6,
            "second-order bundle vs FD: P {worst_p:.3e}  W {worst_w:.3e}  q {worst_q:.3e}"
        );
    }

    // ------------------------------------------------------------------
    // Directional second-order MO-representation gates (stage 5b inputs).
    // ------------------------------------------------------------------

    /// Non-equilibrium water, tight SCF, and a generic (non-symmetric)
    /// direction `v` — the fixture shared by the directional gates below.
    fn directional_gate_fixture() -> (PeriodicSystem, ElectronicOptions, Vec<f64>) {
        let system = PeriodicSystem::from_xyz_str(
            "3\nnon-eq water\nO 0.0 0.0 0.0\nH 1.05 0.55 0.0\nH -0.60 0.95 0.10\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();
        (system, options, v)
    }

    /// `R → R + step·v` (the λ-family the directional derivatives live on).
    fn displaced_along(system: &PeriodicSystem, v: &[f64], step: f64) -> PeriodicSystem {
        let mut sys = system.clone();
        for (atom, a) in sys.atoms.iter_mut().enumerate() {
            a.position.x += step * v[3 * atom];
            a.position.y += step * v[3 * atom + 1];
            a.position.z += step * v[3 * atom + 2];
        }
        sys
    }

    /// Reconverge the SCF at `system` and build the screened directional
    /// FIRST-order field along `v`.
    fn directional_first_order_at(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        v: &[f64],
    ) -> (
        crate::electronic::ElectronicResult,
        ChargeSpaceContext,
        FirstOrderField,
    ) {
        let electronic = run_electronic(system, params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            system,
            params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(system, params, &electronic).unwrap();
        let n = electronic.basis.len();
        let mut f_skel = Matrix::zeros(n, n);
        let mut s_dir = Matrix::zeros(n, n);
        for (c, &vc) in v.iter().enumerate() {
            for k in 0..n * n {
                f_skel.as_mut_slice()[k] +=
                    vc * cpxtb.derivative_matrices[c].h0_deriv.as_slice()[k];
                s_dir.as_mut_slice()[k] +=
                    vc * cpxtb.derivative_matrices[c].overlap_deriv.as_slice()[k];
            }
        }
        let field = ctx.first_order_field(f_skel, s_dir).unwrap();
        (electronic, ctx, field)
    }

    /// The directional SECOND-order field `D_v[·]` of the first-order field
    /// along the same `v` (skeleton second derivatives assembled block-wise
    /// with weights `v_c v_d`).
    fn directional_second_order_at(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        electronic: &crate::electronic::ElectronicResult,
        ctx: &ChargeSpaceContext,
        field: &FirstOrderField,
        v: &[f64],
    ) -> SecondOrderField {
        let cutoff = options.hamiltonian.coordination_cutoff;
        let basis = &electronic.basis;
        let n = basis.len();
        let nshell = basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &electronic.shell_charges,
            params,
        )
        .unwrap();
        let zeros = vec![0.0_f64; nshell];
        let mut f_vv = Matrix::zeros(n, n);
        let mut s_vv = Matrix::zeros(n, n);
        for c in 0..ndof {
            if v[c] == 0.0 {
                continue;
            }
            for d in 0..ndof {
                let w = v[c] * v[d];
                if w == 0.0 {
                    continue;
                }
                let bare = crate::hessian::h0_bare_second_derivative_matrix(
                    system, params, electronic, c, d,
                )
                .unwrap();
                let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                    system, params, electronic, cutoff, c, d,
                )
                .unwrap();
                let v_geo_d: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, d)]).collect();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    system, params, electronic, &v_geo_d, &zeros, c, d,
                )
                .unwrap();
                let sov =
                    crate::response::cpxtb::overlap_second_derivative_matrix(system, basis, c, d)
                        .unwrap();
                for k in 0..n * n {
                    f_vv.as_mut_slice()[k] +=
                        w * (bare.as_slice()[k] + cn_block.as_slice()[k] + scc.as_slice()[k]);
                    s_vv.as_mut_slice()[k] += w * sov.as_slice()[k];
                }
            }
        }
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            basis,
            &field.bundle.shell_charges,
            params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        ctx.second_order_field(field, field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)
            .unwrap()
    }

    /// `eps_second` and `u_second` must be the honest λ-derivatives of the
    /// first-order field's `eps_response` / `u_rotation` along the same `v`.
    ///
    /// `eps_response` is phase-free (per orbital, and water's orbitals are all
    /// gapped so their ORDER is stable under an `O(h)` displacement), so its
    /// central difference is a clean gate with the usual `h²` signature.
    ///
    /// `u_rotation` is NOT phase-free: each displaced geometry re-diagonalizes
    /// through `lowdin_solve_generalized`, whose eigenvector signs may flip.
    /// A flip `C_p → σ_p C_p` scales `h̃_pq` and `s̃_pq` by `σ_p σ_q`, hence
    /// `U_pq → σ_p σ_q U_pq` (while `ε_p` and `eps_response[p]` are invariant).
    /// The comparison is therefore phase-fixed by reading each displaced
    /// context's MO columns (private field, same module) against the reference
    /// ones and undoing the sign pattern before differencing.
    #[test]
    fn second_order_field_matches_first_order_field_finite_difference() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = directional_gate_fixture();
        let (electronic, ctx, field) =
            directional_first_order_at(&system, &params, &options, &v);
        let second =
            directional_second_order_at(&system, &params, &options, &electronic, &ctx, &field, &v);

        let n = electronic.basis.len();
        let norb = ctx.occupations.len();

        // Central FD of (eps_response, U) at step `h`, phase-fixed against the
        // reference MOs. Returns (eps_error, u_error) against the analytic
        // second-order objects.
        let fd_errors = |h: f64| -> (f64, f64) {
            let mut eps_fd = vec![0.0_f64; norb];
            let mut u_fd = Matrix::zeros(norb, norb);
            for &sign in &[1.0_f64, -1.0] {
                let sys = displaced_along(&system, &v, sign * h);
                let (_el, ctx_d, field_d) =
                    directional_first_order_at(&sys, &params, &options, &v);
                // Sign pattern of the displaced MOs relative to the reference.
                let sigma: Vec<f64> = (0..norb)
                    .map(|p| {
                        let dot: f64 =
                            (0..n).map(|mu| ctx.mos[(mu, p)] * ctx_d.mos[(mu, p)]).sum();
                        assert!(
                            dot.abs() > 0.5,
                            "MO {p} lost its identity under the h={h:.1e} displacement \
                             (|overlap| {:.3e}) — orbital tracking failed",
                            dot.abs()
                        );
                        if dot < 0.0 {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let w = sign / (2.0 * h);
                for p in 0..norb {
                    eps_fd[p] += w * field_d.eps_response[p];
                    for q in 0..norb {
                        u_fd[(p, q)] +=
                            w * sigma[p] * sigma[q] * field_d.u_rotation[(p, q)];
                    }
                }
            }
            let eps_err = (0..norb)
                .map(|p| (second.eps_second[p] - eps_fd[p]).abs())
                .fold(0.0_f64, f64::max);
            let u_err = second
                .u_second
                .as_slice()
                .iter()
                .zip(u_fd.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            (eps_err, u_err)
        };

        let h = 1.0e-3;
        let (eps_h, u_h) = fd_errors(h);
        let (eps_2h, u_2h) = fd_errors(2.0 * h);
        eprintln!(
            "second-order field vs FD(first-order field): \
             eps {eps_h:.3e} (2h {eps_2h:.3e}, ratio {:.2})  \
             U {u_h:.3e} (2h {u_2h:.3e}, ratio {:.2})",
            eps_2h / eps_h.max(1.0e-300),
            u_2h / u_h.max(1.0e-300)
        );
        assert!(
            eps_h < 1.0e-6,
            "eps_second vs central FD of eps_response: {eps_h:.3e}"
        );
        assert!(
            u_h < 1.0e-6,
            "u_second vs phase-fixed central FD of u_rotation: {u_h:.3e}"
        );
        // h² signature: doubling the step must quadruple the truncation error.
        // A missing term would leave an h-independent floor (ratio → 1).
        let eps_ratio = eps_2h / eps_h;
        let u_ratio = u_2h / u_h;
        assert!(
            (2.5..6.0).contains(&eps_ratio),
            "eps_second FD error does not scale as h²: {eps_h:.3e} → {eps_2h:.3e} \
             (ratio {eps_ratio:.2}, expected ≈4)"
        );
        assert!(
            (2.5..6.0).contains(&u_ratio),
            "u_second FD error does not scale as h²: {u_h:.3e} → {u_2h:.3e} \
             (ratio {u_ratio:.2}, expected ≈4)"
        );
    }

    /// Second-order orthonormality identity — a phase-free internal gate.
    ///
    /// Derivation. MO orthonormality `C(λ)† S(λ) C(λ) = I` holds at every λ.
    /// With `dC/dλ = C U` its first λ-derivative reads
    ///
    /// ```text
    ///   Uᵀ (C†SC) + C†(dS/dλ)C + (C†SC) U = 0   ⟹   U + Uᵀ = −s̃ ,
    /// ```
    ///
    /// where `s̃ = C†(dS/dλ)C` — the relation the first-order gauge already
    /// satisfies identically (diagonal `−½s̃_pp`, degenerate `−½s̃_pq`, and for
    /// a non-degenerate pair `U_pq + U_qp = [(ε_p−ε_q)s̃_pq]/(ε_q−ε_p) = −s̃_pq`).
    ///
    /// Because that identity holds at EVERY λ, differentiating it once more
    /// along λ — element-wise in the moving MO basis, which is exactly the
    /// derivative `u_second`/`s_dot` represent — gives
    ///
    /// ```text
    ///   U^(λλ) + U^(λλ)ᵀ = −ṡ ,      ṡ = D_λ[s̃] = Uᵀs̃ + s̃U + C†(d²S/dλ²)C .
    /// ```
    ///
    /// Expanding `d²(C†SC)/dλ² = 0` directly with `C̈ = C(U² + U̇)` gives the
    /// equivalent, longer form
    ///
    /// ```text
    ///   U̇ + U̇ᵀ = −ṡ − (Uᵀs̃ + s̃U) − U² − (Uᵀ)² − 2UᵀU ,
    /// ```
    ///
    /// whose extra four terms cancel identically once `Uᵀ = −U − s̃` is
    /// substituted, confirming the compact form.
    ///
    /// The identity is algebraically exact for the implemented gauge, so it is
    /// asserted at 1e-10 — it pins down `u_second`'s antisymmetric-plus-metric
    /// structure without depending on MO phases at all.
    #[test]
    fn second_order_rotation_satisfies_orthonormality_identity() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = directional_gate_fixture();
        let (electronic, ctx, field) =
            directional_first_order_at(&system, &params, &options, &v);
        let second =
            directional_second_order_at(&system, &params, &options, &electronic, &ctx, &field, &v);
        let norb = ctx.occupations.len();

        // First order: U + Uᵀ = −s̃.
        let mut worst_first = 0.0_f64;
        for p in 0..norb {
            for q in 0..norb {
                let r = field.u_rotation[(p, q)]
                    + field.u_rotation[(q, p)]
                    + field.s_tilde[(p, q)];
                worst_first = worst_first.max(r.abs());
            }
        }
        // Second order: U^(vv) + U^(vv)ᵀ = −ṡ.
        let mut worst_second = 0.0_f64;
        for p in 0..norb {
            for q in 0..norb {
                let r = second.u_second[(p, q)]
                    + second.u_second[(q, p)]
                    + second.s_dot[(p, q)];
                worst_second = worst_second.max(r.abs());
            }
        }
        // ḣ and ṡ inherit the symmetry of their AO parents.
        let mut worst_sym = 0.0_f64;
        for p in 0..norb {
            for q in 0..norb {
                worst_sym = worst_sym
                    .max((second.s_dot[(p, q)] - second.s_dot[(q, p)]).abs())
                    .max((second.h_dot[(p, q)] - second.h_dot[(q, p)]).abs());
            }
        }
        eprintln!(
            "orthonormality residuals: first {worst_first:.3e}  second {worst_second:.3e}  \
             symmetry {worst_sym:.3e}"
        );
        assert!(
            worst_first < 1.0e-10,
            "first-order orthonormality U + Uᵀ + s̃ = 0 violated: {worst_first:.3e}"
        );
        assert!(
            worst_second < 1.0e-10,
            "second-order orthonormality U^(vv) + U^(vv)ᵀ + ṡ = 0 violated: {worst_second:.3e}"
        );
        assert!(
            worst_sym < 1.0e-10,
            "ḣ/ṡ are not symmetric: {worst_sym:.3e}"
        );
    }

    /// `solve_second_order` must stay a pure projection of `second_order_field`
    /// — bit-for-bit, so the existing quartic-assembly gates are untouched.
    #[test]
    fn solve_second_order_is_bit_identical_to_second_order_field() {
        let params = Gfn1Parameters::builtin().unwrap();
        let (system, options, v) = directional_gate_fixture();
        let (electronic, ctx, field) =
            directional_first_order_at(&system, &params, &options, &v);
        let full =
            directional_second_order_at(&system, &params, &options, &electronic, &ctx, &field, &v);

        // Rebuild the same inputs and go through the legacy entry point.
        let basis = &electronic.basis;
        let n = basis.len();
        let nshell = basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let zeros = vec![0.0_f64; nshell];
        let mut f_vv = Matrix::zeros(n, n);
        let mut s_vv = Matrix::zeros(n, n);
        for c in 0..ndof {
            for d in 0..ndof {
                let w = v[c] * v[d];
                if w == 0.0 {
                    continue;
                }
                let bare = crate::hessian::h0_bare_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    c,
                    d,
                )
                .unwrap();
                let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    cutoff,
                    c,
                    d,
                )
                .unwrap();
                let v_geo_d: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, d)]).collect();
                let scc = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                    &system,
                    &params,
                    &electronic,
                    &v_geo_d,
                    &zeros,
                    c,
                    d,
                )
                .unwrap();
                let sov =
                    crate::response::cpxtb::overlap_second_derivative_matrix(&system, basis, c, d)
                        .unwrap();
                for k in 0..n * n {
                    f_vv.as_mut_slice()[k] +=
                        w * (bare.as_slice()[k] + cn_block.as_slice()[k] + scc.as_slice()[k]);
                    s_vv.as_mut_slice()[k] += w * sov.as_slice()[k];
                }
            }
        }
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            basis,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let legacy = ctx
            .solve_second_order(&field, &field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)
            .unwrap();

        assert_eq!(legacy.density.as_slice(), full.bundle.density.as_slice());
        assert_eq!(
            legacy.energy_weighted.as_slice(),
            full.bundle.energy_weighted.as_slice()
        );
        assert_eq!(legacy.shell_charges, full.bundle.shell_charges);
        assert_eq!(
            legacy.occupation_response,
            full.bundle.occupation_response
        );
    }

    /// Finite temperature: the ground truth is the SCC itself. Compare BOTH
    /// solvers' shell-charge responses against the central finite difference
    /// of the reconverged SCC shell charges on selected columns, and check the
    /// dielectric solve satisfies its own fixed point exactly.
    #[test]
    fn first_order_finite_t_ni_co4_vs_scc_finite_difference() {
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        options.energy_tolerance = 1.0e-12;
        options.charge_tolerance = 1.0e-10;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        assert!(ctx.is_finite_temperature());

        let h = 1.0e-4;
        let mut worst_mine = 0.0_f64;
        let mut worst_theirs = 0.0_f64;
        let mut worst_selfc = 0.0_f64;
        for &col in &[0usize, 13, 26] {
            let bundle = ctx
                .solve_first_order(
                    &cpxtb.derivative_matrices[col].h0_deriv,
                    &cpxtb.derivative_matrices[col].overlap_deriv,
                )
                .unwrap();
            // Internal fixed-point residual: shells(P_final) must equal q.
            let n = electronic.basis.len();
            let _ = n;
            let q_of_p = response_shell_charges_from_density(
                &electronic.basis,
                &electronic.integrals.overlap,
                &electronic.density,
                &bundle.density,
                &cpxtb.derivative_matrices[col].overlap_deriv,
            )
            .unwrap();
            let selfc = q_of_p
                .iter()
                .zip(&bundle.shell_charges)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            worst_selfc = worst_selfc.max(selfc);

            // SCC finite difference (reconverged at displaced geometries).
            let (atom, axis) = (col / 3, col % 3);
            let mut plus = system.clone();
            let mut minus = system.clone();
            match axis {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let qp = run_electronic(&plus, &params, options.clone())
                .unwrap()
                .shell_charges;
            let qm = run_electronic(&minus, &params, options.clone())
                .unwrap()
                .shell_charges;
            for s in 0..qp.len() {
                let fd = (qp[s] - qm[s]) / (2.0 * h);
                worst_mine = worst_mine.max((bundle.shell_charges[s] - fd).abs());
                worst_theirs =
                    worst_theirs.max((cpxtb.shell_charge_responses[col][s] - fd).abs());
            }
        }
        eprintln!(
            "finite-T q^x vs SCC FD: charge-space {worst_mine:.3e}  fixed-point branch {worst_theirs:.3e}  self-consistency {worst_selfc:.3e}"
        );
        assert!(
            worst_selfc < 1.0e-10,
            "dielectric solution violates its own fixed point: {worst_selfc:.3e}"
        );
        assert!(
            worst_mine < 5.0e-6,
            "charge-space finite-T response vs SCC FD: {worst_mine:.3e}"
        );
    }

    /// **Finite-temperature second-order gate body.** `solve_second_order` on
    /// a Fermi-smeared fixture must match the central finite difference of the
    /// FIRST-order bundle along the `y` displacement with EVERYTHING
    /// reconverged (SCF, CPXTB skeletons, charge-space context) — exercising
    /// the `f''`/`μ^{(xy)}` occupation channel and the finite-T coefficient
    /// motion `Δ𝒞_T`, both absent at T = 0. P/W/q are AO-basis observables, so
    /// the FD is free of MO-phase ambiguity. Also pins second-order
    /// particle-number conservation `Σ_p f^{(xy)}_p = 0`.
    fn run_finite_t_second_order_gate(xyz: &str, label: &str) {
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        // Extra-tight SCF: the FD noise floor of the reconverged first-order
        // bundles scales as charge_tolerance / (2h) ≈ 5e-7 at 1e-10 — tighten
        // so the gate measures the analytic assembly, not SCC convergence.
        options.energy_tolerance = 1.0e-14;
        options.charge_tolerance = 1.0e-12;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
        };
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        assert!(ctx.is_finite_temperature());
        {
            // Both fixtures must keep their (near-)degeneracy coverage: the
            // branch-consistent rotation/coefficient algebra is exactly what
            // this gate exists to pin, so a geometry edit that silently
            // removes the sub-1e-6 pair gaps would gut the test.
            let e = &ctx.orbital_energies;
            let mut min_gap = f64::INFINITY;
            for p in 0..e.len() {
                for q in (p + 1)..e.len() {
                    min_gap = min_gap.min((e[p] - e[q]).abs());
                }
            }
            let blocks = ctx.fractional_degenerate_blocks();
            eprintln!(
                "{label}: kt {:.3e}, min pair gap {min_gap:.3e}, fractional degenerate blocks: {:?} (w of first members: {:?})",
                ctx.kt,
                blocks.iter().map(|b| b.len()).collect::<Vec<_>>(),
                blocks
                    .iter()
                    .map(|b| (ctx.occupations[b[0]] * (1.0 - 0.5 * ctx.occupations[b[0]])).max(0.0)
                        / ctx.kt)
                    .collect::<Vec<_>>()
            );
            assert!(
                min_gap < 1.0e-6,
                "{label}: fixture lost its (near-)degeneracy coverage (min gap {min_gap:.3e})"
            );
        }

        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let nshell = electronic.basis.shells.len();
        let field_for = |dof: usize| -> FirstOrderField {
            ctx.first_order_field(
                cpxtb.derivative_matrices[dof].h0_deriv.clone(),
                cpxtb.derivative_matrices[dof].overlap_deriv.clone(),
            )
            .unwrap()
        };

        let h = 2.0e-4;
        let mut worst_p = 0.0_f64;
        let mut worst_w = 0.0_f64;
        let mut worst_q = 0.0_f64;
        let mut worst_fsum = 0.0_f64;
        for &(dof_x, dof_y) in &[(0usize, 0usize), (4, 13), (10, 22)] {
            let fx = field_for(dof_x);
            let fy = field_for(dof_y);

            let mut f_xy = crate::hessian::h0_bare_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                dof_x,
                dof_y,
            )
            .unwrap();
            let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                cutoff,
                dof_x,
                dof_y,
            )
            .unwrap();
            let v_geo_y: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, dof_y)]).collect();
            let scc_block = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                &v_geo_y,
                &vec![0.0; nshell],
                dof_x,
                dof_y,
            )
            .unwrap();
            for (dst, (a, b)) in f_xy
                .as_mut_slice()
                .iter_mut()
                .zip(cn_block.as_slice().iter().zip(scc_block.as_slice()))
            {
                *dst += a + b;
            }
            let s_xy = crate::response::cpxtb::overlap_second_derivative_matrix(
                &system,
                &electronic.basis,
                dof_x,
                dof_y,
            )
            .unwrap();
            let dgamma_y_qx = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fx.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_y)]).collect::<Vec<f64>>()
            };
            let dgamma_x_qy = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fy.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_x)]).collect::<Vec<f64>>()
            };
            let second = ctx
                .solve_second_order(&fx, &fy, &f_xy, &s_xy, &dgamma_y_qx, &dgamma_x_qy)
                .unwrap();
            worst_fsum = worst_fsum.max(
                second
                    .occupation_response
                    .iter()
                    .sum::<f64>()
                    .abs(),
            );

            let displace = |sign: f64| -> FirstOrderBundle {
                let mut sys = system.clone();
                let (atom, axis) = (dof_y / 3, dof_y % 3);
                match axis {
                    0 => sys.atoms[atom].position.x += sign * h,
                    1 => sys.atoms[atom].position.y += sign * h,
                    _ => sys.atoms[atom].position.z += sign * h,
                }
                let el = run_electronic(&sys, &params, options.clone()).unwrap();
                let cp = solve_nonpbc_cpxtb_hessian_response(
                    &sys,
                    &params,
                    &el,
                    ao_opts,
                    CpxtbOptions::default(),
                )
                .unwrap();
                let c = ChargeSpaceContext::build(&sys, &params, &el).unwrap();
                c.solve_first_order(
                    &cp.derivative_matrices[dof_x].h0_deriv,
                    &cp.derivative_matrices[dof_x].overlap_deriv,
                )
                .unwrap()
            };
            let plus = displace(1.0);
            let minus = displace(-1.0);
            for i in 0..second.density.as_slice().len() {
                let fd = (plus.density.as_slice()[i] - minus.density.as_slice()[i]) / (2.0 * h);
                worst_p = worst_p.max((second.density.as_slice()[i] - fd).abs());
                let fdw = (plus.energy_weighted.as_slice()[i]
                    - minus.energy_weighted.as_slice()[i])
                    / (2.0 * h);
                worst_w = worst_w.max((second.energy_weighted.as_slice()[i] - fdw).abs());
            }
            for s in 0..nshell {
                let fd = (plus.shell_charges[s] - minus.shell_charges[s]) / (2.0 * h);
                worst_q = worst_q.max((second.shell_charges[s] - fd).abs());
            }
        }
        eprintln!(
            "{label} finite-T second-order vs FD(first-order): P {worst_p:.3e}  W {worst_w:.3e}  \
             q {worst_q:.3e}  |Σf_xy| {worst_fsum:.3e}"
        );
        assert!(
            worst_fsum < 1.0e-12,
            "{label}: second-order occupation response violates particle-number conservation: \
             {worst_fsum:.3e}"
        );
        assert!(
            worst_p < 1.0e-6 && worst_w < 1.0e-6 && worst_q < 1.0e-6,
            "{label} finite-T second-order bundle vs FD: P {worst_p:.3e}  W {worst_w:.3e}  \
             q {worst_q:.3e}"
        );
    }

    /// Distorted Ni(CO)₄ (Td broken): fractionally occupied non-degenerate
    /// levels PLUS accidental near-degenerate pairs (gaps 1.9e-8 … 3.6e-7) —
    /// the branch-consistency window the threshold alignment exists for.
    #[test]
    fn second_order_finite_t_matches_first_order_finite_difference() {
        run_finite_t_second_order_gate(
            "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n",
            "distorted",
        );
    }

    /// Build the DIRECTIONAL second-order field plus the third-order inputs at
    /// `system`, reconverging everything — the shared harness of the
    /// third-order gate.
    fn third_order_setup(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: &ElectronicOptions,
        v: &[f64],
    ) -> (
        crate::electronic::ElectronicResult,
        ChargeSpaceContext,
        FirstOrderField,
        SecondOrderField,
    ) {
        let electronic = run_electronic(system, params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let include_cn_h0 = options.hamiltonian.enable_cn_hamiltonian;
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            system,
            params,
            &electronic,
            AoDerivativeOptions {
                coordination_cutoff: cutoff,
                include_cn_h0,
            },
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(system, params, &electronic).unwrap();
        let n = electronic.basis.len();
        let nshell = electronic.basis.shells.len();
        let ndof = 3 * system.atoms.len();
        let mut f_skel = Matrix::zeros(n, n);
        let mut s_dir = Matrix::zeros(n, n);
        for (c, &vc) in v.iter().enumerate() {
            for k in 0..n * n {
                f_skel.as_mut_slice()[k] += vc * cpxtb.derivative_matrices[c].h0_deriv.as_slice()[k];
                s_dir.as_mut_slice()[k] +=
                    vc * cpxtb.derivative_matrices[c].overlap_deriv.as_slice()[k];
            }
        }
        let field = ctx.first_order_field(f_skel, s_dir).unwrap();
        // Directional second-order skeletons (one-pass builders).
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            &electronic.basis,
            &electronic.shell_charges,
            params,
        )
        .unwrap();
        let v_geo: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
            .collect();
        let zeros = vec![0.0_f64; nshell];
        let mut f_vv = crate::hessian::directional_h0_bare_second_matrix(
            system, params, &electronic, v,
        )
        .unwrap();
        {
            let cn = crate::hessian::directional_h0_cn_block_second_matrix(
                system, params, &electronic, cutoff, v,
            )
            .unwrap();
            let scc = crate::hessian::directional_h0_scc_scalar_second_matrix(
                system, params, &electronic, v, &v_geo, &zeros,
            )
            .unwrap();
            for k in 0..n * n {
                f_vv.as_mut_slice()[k] += cn.as_slice()[k] + scc.as_slice()[k];
            }
        }
        let s_vv =
            crate::hessian::directional_overlap_second_matrix(system, &electronic.basis, v)
                .unwrap();
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            system,
            &electronic.basis,
            &field.bundle.shell_charges,
            params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let second = ctx
            .second_order_field(&field, &field, &f_vv, &s_vv, &dgamma_v_qv, &dgamma_v_qv)
            .unwrap();
        (electronic, ctx, field, second)
    }

    /// **Third-order directional gate.** `solve_third_order_directional` vs
    /// the central FD of the second-order bundle along `v` with everything
    /// reconverged (SCF, CPXTB, context, directional fields), `h²` ladder.
    /// Exercises the full inventory: the dual-lifted `Δ²𝒞_quad`, the
    /// symmetric-D³ `RF_{S^{vv}}(V^v_geo)` completion, and the CN cache
    /// motion of the bare-H0 skeleton.
    fn run_third_order_gate(
        system: PeriodicSystem,
        options: ElectronicOptions,
        label: &str,
        h_base: f64,
        tol: f64,
    ) {
        let params = Gfn1Parameters::builtin().unwrap();
        let ndof = 3 * system.atoms.len();
        let v: Vec<f64> = (0..ndof)
            .map(|k| 0.23 - 0.11 * ((k % 4) as f64) + 0.05 * ((k * 7 % 5) as f64))
            .collect();

        let (electronic, ctx, field, second) =
            third_order_setup(&system, &params, &options, &v);
        let nshell = electronic.basis.shells.len();
        let n = electronic.basis.len();
        let cutoff = options.hamiltonian.coordination_cutoff;

        // Third-order inputs.
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let d2vdr_q = crate::hessian::shell_scalar_potential_second_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let v_geo: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|d| v[d] * dvdr_q[(s, d)]).sum())
            .collect();
        let v_geo2: Vec<f64> = (0..nshell)
            .map(|s| {
                (0..ndof)
                    .map(|c| {
                        (0..ndof)
                            .map(|d| v[c] * v[d] * d2vdr_q[s][(c, d)])
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        let zeros = vec![0.0_f64; nshell];
        let mut f3 = crate::hessian::directional_h0_bare_third_matrix(
            &system, &params, &electronic, &v,
        )
        .unwrap();
        {
            let cn = crate::hessian::directional_h0_cn_block_third_matrix(
                &system, &params, &electronic, cutoff, &v,
            )
            .unwrap();
            let scc = crate::hessian::directional_h0_scc_scalar_third_matrix(
                &system, &params, &electronic, &v, &v_geo, &v_geo2, &zeros, &zeros,
            )
            .unwrap();
            // CN cache motion of h0_bare²: the affine self-energy trick — the
            // bare-second builder at CN = CN^v minus at CN = 0 is exactly the
            // missing third `c₁·P₂` copy (the cn-block third emits only two).
            let cn_v_resp = {
                let nat = system.atoms.len();
                let cn_grad =
                    crate::hessian::cn_gradient_matrix(&system, cutoff).unwrap();
                (0..nat)
                    .map(|at| (0..ndof).map(|c| v[c] * cn_grad[at][c]).sum::<f64>())
                    .collect::<Vec<f64>>()
            };
            let cn_cache = {
                let mut el_cnv = electronic.clone();
                el_cnv.coordination_numbers = cn_v_resp;
                let a = crate::hessian::directional_h0_bare_second_matrix(
                    &system, &params, &el_cnv, &v,
                )
                .unwrap();
                let mut el_cn0 = electronic.clone();
                el_cn0.coordination_numbers = vec![0.0; system.atoms.len()];
                let b = crate::hessian::directional_h0_bare_second_matrix(
                    &system, &params, &el_cn0, &v,
                )
                .unwrap();
                let mut m = a;
                for k in 0..n * n {
                    m.as_mut_slice()[k] -= b.as_slice()[k];
                }
                m
            };
            for k in 0..n * n {
                f3.as_mut_slice()[k] +=
                    cn.as_slice()[k] + scc.as_slice()[k] + cn_cache.as_slice()[k];
            }
        }
        let s3 = crate::hessian::directional_overlap_third_matrix(
            &system,
            &electronic.basis,
            &v,
        )
        .unwrap();
        let s2m =
            crate::hessian::directional_overlap_second_matrix(&system, &electronic.basis, &v)
                .unwrap();
        let dgamma_qv = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let dgamma_v_qv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qv[(s, c)]).sum())
            .collect();
        let dgamma_qvv = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &second.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let dgamma_v_qvv: Vec<f64> = (0..nshell)
            .map(|s| (0..ndof).map(|c| v[c] * dgamma_qvv[(s, c)]).sum())
            .collect();
        let d2vdr_qv = crate::hessian::shell_scalar_potential_second_derivatives(
            &system,
            &electronic.basis,
            &field.bundle.shell_charges,
            &params,
        )
        .unwrap();
        let d2gamma_vv_qv: Vec<f64> = (0..nshell)
            .map(|s| {
                (0..ndof)
                    .map(|c| {
                        (0..ndof)
                            .map(|d| v[c] * v[d] * d2vdr_qv[s][(c, d)])
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();

        let third = ctx
            .solve_third_order_directional(
                &field,
                &second,
                &f3,
                &s3,
                &s2m,
                &v_geo,
                &dgamma_v_qv,
                &dgamma_v_qvv,
                &d2gamma_vv_qv,
            )
            .unwrap();

        // FD reference: second-order bundle at R ± h·v, everything reconverged.
        let bundle_at = |step: f64| -> SecondOrderBundle {
            let sys = displaced_along(&system, &v, step);
            let (_el, _ctx, _f, sec) = third_order_setup(&sys, &params, &options, &v);
            sec.bundle
        };
        let compare = |h: f64| -> (f64, f64, f64) {
            let plus = bundle_at(h);
            let minus = bundle_at(-h);
            let mut wp = 0.0_f64;
            let mut ww = 0.0_f64;
            let mut wq = 0.0_f64;
            for i in 0..third.density.as_slice().len() {
                let fd = (plus.density.as_slice()[i] - minus.density.as_slice()[i]) / (2.0 * h);
                wp = wp.max((third.density.as_slice()[i] - fd).abs());
                let fdw = (plus.energy_weighted.as_slice()[i]
                    - minus.energy_weighted.as_slice()[i])
                    / (2.0 * h);
                ww = ww.max((third.energy_weighted.as_slice()[i] - fdw).abs());
            }
            for s in 0..nshell {
                let fd = (plus.shell_charges[s] - minus.shell_charges[s]) / (2.0 * h);
                wq = wq.max((third.shell_charges[s] - fd).abs());
            }
            (wp, ww, wq)
        };
        let (p1, w1, q1) = compare(h_base);
        let (p2, w2, q2) = compare(0.5 * h_base);
        eprintln!(
            "{label} third-order vs FD(second-order): h={h_base:.1e} P {p1:.3e} W {w1:.3e} \
             q {q1:.3e} | h/2 P {p2:.3e} W {w2:.3e} q {q2:.3e} | ratios P {:.2} W {:.2} q {:.2}",
            p1 / p2.max(1.0e-300),
            w1 / w2.max(1.0e-300),
            q1 / q2.max(1.0e-300)
        );
        assert!(
            p2 < tol && w2 < tol && q2 < tol,
            "{label} third-order bundle vs FD: P {p2:.3e} W {w2:.3e} q {q2:.3e} (tol {tol:.1e})"
        );
        // h² scaling on the largest channel (the smaller ones may sit on the
        // reconvergence noise floor, which GROWS as 1/h).
        let r_max = (p1 / p2.max(1.0e-300))
            .max(w1 / w2.max(1.0e-300))
            .max(q1 / q2.max(1.0e-300));
        assert!(
            r_max > 3.0,
            "{label}: no channel shows h² truncation scaling (best ratio {r_max:.2})"
        );
    }

    #[test]
    fn third_order_directional_matches_second_order_fd() {
        let (system, options, _v) = directional_gate_fixture();
        run_third_order_gate(system, options, "T=0 water", 1.0e-3, 1.0e-7);
    }

    /// Fermi-smeared variant: distorted Ni(CO)₄ at 3000 K (fractional
    /// occupations, accidental near-degenerate pairs) — the `f'''`/`μ^{vvv}`
    /// occupation chains and the finite-T dual-lifted `Δ²𝒞` go live.
    #[test]
    fn third_order_directional_matches_second_order_fd_finite_t() {
        let system = PeriodicSystem::from_xyz_str(
            "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n",
            0.0,
            false,
        )
        .unwrap();
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        options.energy_tolerance = 1.0e-14;
        options.charge_tolerance = 1.0e-12;
        // Larger FD step than the T=0 gate: the reconvergence noise of the
        // second-order bundle (amplified by the near-degenerate pairs) grows
        // as 1/h and sits near 5e-8 at h = 5e-4.
        run_third_order_gate(system, options, "finite-T Ni(CO)4", 4.0e-3, 1.0e-6);
    }

    /// Symmetric Td Ni(CO)₄ at 3000 K — EXACTLY degenerate fractionally
    /// occupied blocks. The resolvent form must reproduce the finite
    /// difference of the reconverged first-order response.
    #[test]
    fn second_order_finite_t_exact_degenerate_matches_fd() {
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        run_finite_t_second_order_gate(xyz, "Td Ni(CO)4 exact-degenerate");
    }

    /// The THIRD order is still frame-based, so it must refuse an exactly
    /// degenerate fractional reference rather than return a number (measured
    /// 3.5e3 against the FD gate if forced). Pins the guard that replaced the
    /// second-order one when the resolvent form landed.
    #[test]
    fn third_order_finite_t_exact_degenerate_is_rejected() {
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        assert!(!ctx.fractional_degenerate_blocks().is_empty());
        let n = electronic.basis.len();
        let nshell = electronic.basis.shells.len();
        let zero = Matrix::zeros(n, n);
        let fx = ctx.first_order_field(zero.clone(), zero.clone()).unwrap();
        // The second order must SUCCEED (resolvent path) …
        let second = ctx
            .second_order_field(
                &fx,
                &fx,
                &zero,
                &zero,
                &vec![0.0; nshell],
                &vec![0.0; nshell],
            )
            .expect("second order takes the frame-free resolvent path");
        // … and the third must refuse.
        let err = ctx
            .solve_third_order_directional(
                &fx,
                &second,
                &zero,
                &zero,
                &zero,
                &vec![0.0; nshell],
                &vec![0.0; nshell],
                &vec![0.0; nshell],
                &vec![0.0; nshell],
            )
            .unwrap_err();
        assert!(
            format!("{err}").contains("exactly degenerate"),
            "expected the third-order degeneracy guard, got: {err}"
        );
    }

    /// The reference state must actually carry exactly degenerate fractional
    /// blocks (otherwise the gate above is vacuous) and the solver must be on
    /// the resolvent path for them.
    #[test]
    fn exact_degenerate_reference_selects_the_resolvent_path() {
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        options.energy_tolerance = 1.0e-14;
        options.charge_tolerance = 1.0e-12;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        assert!(ctx.is_finite_temperature());
        assert!(
            !ctx.fractional_degenerate_blocks().is_empty(),
            "fixture lost its fractional degenerate blocks"
        );
        assert!(
            ctx.dk_second,
            "an exactly degenerate fractional reference must select the resolvent path"
        );
    }

    /// **Equality gate for the Daleckii–Krein second order.** On a
    /// NON-degenerate finite-T system the frame-free resolvent form and the
    /// validated coefficient/rotation algebra must agree element-wise — the
    /// strongest available check that the divided-difference derivation is
    /// right before it is trusted inside degenerate blocks.
    #[test]
    fn dk_second_order_matches_frame_path_non_degenerate() {
        let xyz = "9\ndistorted Ni(CO)4\nNi 0.020000 -0.030000 0.010000\nC 1.960000 1.750000 1.820000\nO 2.640000 2.400000 2.480000\nC -1.820000 -1.870000 1.760000\nO -2.480000 -2.540000 2.400000\nC -1.750000 1.820000 -1.900000\nO -2.400000 2.480000 -2.560000\nC 1.820000 -1.760000 -1.820000\nO 2.480000 -2.420000 -2.480000\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        options.energy_tolerance = 1.0e-14;
        options.charge_tolerance = 1.0e-12;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
        };
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        let mut ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        assert!(ctx.is_finite_temperature());
        assert!(
            ctx.fractional_degenerate_blocks().is_empty(),
            "equality gate needs a NON-degenerate fixture"
        );
        let n = electronic.basis.len();
        let nshell = electronic.basis.shells.len();
        let (dof_x, dof_y) = (4usize, 13usize);
        let build_second = |ctx: &ChargeSpaceContext| -> SecondOrderBundle {
            let fx = ctx
                .first_order_field(
                    cpxtb.derivative_matrices[dof_x].h0_deriv.clone(),
                    cpxtb.derivative_matrices[dof_x].overlap_deriv.clone(),
                )
                .unwrap();
            let fy = ctx
                .first_order_field(
                    cpxtb.derivative_matrices[dof_y].h0_deriv.clone(),
                    cpxtb.derivative_matrices[dof_y].overlap_deriv.clone(),
                )
                .unwrap();
            let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
                &system,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap();
            let mut f_xy = crate::hessian::h0_bare_second_derivative_matrix(
                &system, &params, &electronic, dof_x, dof_y,
            )
            .unwrap();
            let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
                &system, &params, &electronic, cutoff, dof_x, dof_y,
            )
            .unwrap();
            let v_geo_y: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, dof_y)]).collect();
            let scc_block = crate::hessian::h0_scc_scalar_second_derivative_matrix(
                &system,
                &params,
                &electronic,
                &v_geo_y,
                &vec![0.0; nshell],
                dof_x,
                dof_y,
            )
            .unwrap();
            for (dst, (a, b)) in f_xy
                .as_mut_slice()
                .iter_mut()
                .zip(cn_block.as_slice().iter().zip(scc_block.as_slice()))
            {
                *dst += a + b;
            }
            let s_xy = crate::response::cpxtb::overlap_second_derivative_matrix(
                &system,
                &electronic.basis,
                dof_x,
                dof_y,
            )
            .unwrap();
            let dg_y_qx = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fx.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_y)]).collect::<Vec<f64>>()
            };
            let dg_x_qy = {
                let m = crate::hessian::shell_scalar_potential_first_derivatives(
                    &system,
                    &electronic.basis,
                    &fy.bundle.shell_charges,
                    &params,
                )
                .unwrap();
                (0..nshell).map(|s| m[(s, dof_x)]).collect::<Vec<f64>>()
            };
            ctx.solve_second_order(&fx, &fy, &f_xy, &s_xy, &dg_y_qx, &dg_x_qy)
                .unwrap()
        };
        let frame = build_second(&ctx);
        ctx.set_dk_second(true);
        let dk = build_second(&ctx);
        let mut worst_p = 0.0_f64;
        let mut worst_w = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                worst_p = worst_p.max((frame.density[(i, j)] - dk.density[(i, j)]).abs());
                worst_w =
                    worst_w.max((frame.energy_weighted[(i, j)] - dk.energy_weighted[(i, j)]).abs());
            }
        }
        let worst_q = frame
            .shell_charges
            .iter()
            .zip(&dk.shell_charges)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "DK vs frame second order (non-degenerate): P {worst_p:.3e}  W {worst_w:.3e}  q \
             {worst_q:.3e}"
        );
        assert!(
            worst_p < 1.0e-8 && worst_w < 1.0e-8 && worst_q < 1.0e-9,
            "DK second-order response disagrees with the frame path: P {worst_p:.3e} W \
             {worst_w:.3e} q {worst_q:.3e}"
        );
    }

    /// Shell/orbital-wise dissection of the (still failing) covariant block
    /// channel on Td Ni(CO)₄: drive `second_order_field_with_blocks` with the
    /// forced degenerate blocks and print WHERE the q/P errors live — the
    /// production gate only reports the worst norms.
    #[test]
    #[ignore = "diagnostic"]
    fn covariant_block_channel_shellwise_diagnostic() {
        let xyz = "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n";
        let mut options = ElectronicOptions::default();
        options.enable_dispersion = false;
        options.electronic_temperature = 3000.0;
        options.energy_tolerance = 1.0e-14;
        options.charge_tolerance = 1.0e-12;
        let params = Gfn1Parameters::builtin().unwrap();
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let electronic = run_electronic(&system, &params, options.clone()).unwrap();
        let cutoff = options.hamiltonian.coordination_cutoff;
        let ao_opts = AoDerivativeOptions {
            coordination_cutoff: cutoff,
            include_cn_h0: options.hamiltonian.enable_cn_hamiltonian,
        };
        let cpxtb = solve_nonpbc_cpxtb_hessian_response(
            &system,
            &params,
            &electronic,
            ao_opts,
            CpxtbOptions::default(),
        )
        .unwrap();
        let ctx = ChargeSpaceContext::build(&system, &params, &electronic).unwrap();
        let blocks = ctx.fractional_degenerate_blocks();
        assert!(!blocks.is_empty());
        let nshell = electronic.basis.shells.len();
        let dof = 0usize;
        let fx = ctx
            .first_order_field(
                cpxtb.derivative_matrices[dof].h0_deriv.clone(),
                cpxtb.derivative_matrices[dof].overlap_deriv.clone(),
            )
            .unwrap();
        let dvdr_q = crate::hessian::shell_scalar_potential_first_derivatives(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let mut f_xy = crate::hessian::h0_bare_second_derivative_matrix(
            &system, &params, &electronic, dof, dof,
        )
        .unwrap();
        let cn_block = crate::hessian::h0_cn_block_second_derivative_matrix(
            &system, &params, &electronic, cutoff, dof, dof,
        )
        .unwrap();
        let v_geo: Vec<f64> = (0..nshell).map(|s| dvdr_q[(s, dof)]).collect();
        let scc_block = crate::hessian::h0_scc_scalar_second_derivative_matrix(
            &system,
            &params,
            &electronic,
            &v_geo,
            &vec![0.0; nshell],
            dof,
            dof,
        )
        .unwrap();
        for (dst, (a, b)) in f_xy
            .as_mut_slice()
            .iter_mut()
            .zip(cn_block.as_slice().iter().zip(scc_block.as_slice()))
        {
            *dst += a + b;
        }
        let s_xy = crate::response::cpxtb::overlap_second_derivative_matrix(
            &system,
            &electronic.basis,
            dof,
            dof,
        )
        .unwrap();
        let dgamma = {
            let m = crate::hessian::shell_scalar_potential_first_derivatives(
                &system,
                &electronic.basis,
                &fx.bundle.shell_charges,
                &params,
            )
            .unwrap();
            (0..nshell).map(|s| m[(s, dof)]).collect::<Vec<f64>>()
        };
        let second = ctx
            .second_order_field_with_blocks(&fx, &fx, &f_xy, &s_xy, &dgamma, &dgamma, blocks.clone())
            .unwrap();

        // FD of the reconverged first-order bundle along the same dof.
        let h = 2.0e-4;
        let bundle_at = |sign: f64| -> FirstOrderBundle {
            let mut sys = system.clone();
            sys.atoms[dof / 3].position.x += sign * h;
            let el = run_electronic(&sys, &params, options.clone()).unwrap();
            let cp = solve_nonpbc_cpxtb_hessian_response(
                &sys, &params, &el, ao_opts, CpxtbOptions::default(),
            )
            .unwrap();
            let c2 = ChargeSpaceContext::build(&sys, &params, &el).unwrap();
            c2.solve_first_order(
                &cp.derivative_matrices[dof].h0_deriv,
                &cp.derivative_matrices[dof].overlap_deriv,
            )
            .unwrap()
        };
        let bp = bundle_at(1.0);
        let bm = bundle_at(-1.0);
        println!("shell |  q^xy analytic |  q^xy FD | diff");
        for s in 0..nshell {
            let fd = (bp.shell_charges[s] - bm.shell_charges[s]) / (2.0 * h);
            let an = second.bundle.shell_charges[s];
            if (an - fd).abs() > 1.0e-4 {
                println!("{s:5} | {an:+.6e} | {fd:+.6e} | {:+.3e}", an - fd);
            }
        }
        let n = electronic.basis.len();
        let mut dp = Matrix::zeros(n, n);
        let mut worst = (0.0_f64, 0usize, 0usize);
        for i in 0..n {
            for j in 0..n {
                let fd = (bp.density[(i, j)] - bm.density[(i, j)]) / (2.0 * h);
                dp[(i, j)] = second.bundle.density[(i, j)] - fd;
                let d = dp[(i, j)].abs();
                if d > worst.0 {
                    worst = (d, i, j);
                }
            }
        }
        println!(
            "worst P element: {:.3e} at AO ({}, {})  blocks(orbital indices): {:?}",
            worst.0, worst.1, worst.2, blocks
        );
        // Project the error into the MO basis: D = Cᵀ S ΔP S C. If the
        // failure is the block-coefficient overwrite, the large entries sit
        // exactly on in-block (p, q) pairs.
        let sc = ctx.overlap.matmul(&ctx.mos).unwrap();
        let tmp = dp.matmul(&sc).unwrap();
        let dmo = sc.transpose().matmul(&tmp).unwrap();
        let mut in_block = vec![false; n];
        for b in &blocks {
            for &p in b {
                in_block[p] = true;
            }
        }
        let mut entries: Vec<(f64, usize, usize)> = Vec::new();
        for p in 0..n {
            for q in 0..n {
                entries.push((dmo[(p, q)].abs(), p, q));
            }
        }
        entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("top MO-basis error entries (|D|, p, q, in-block flags):");
        for &(d, p, q) in entries.iter().take(10) {
            println!(
                "  {d:.3e}  ({p:2}, {q:2})  [{}, {}]  eps ({:+.6}, {:+.6})",
                in_block[p], in_block[q], ctx.orbital_energies[p], ctx.orbital_energies[q]
            );
        }

        // FIRST-order hypothesis check: the ground density is gauge/basis
        // invariant, so dP₀/dλ is an unambiguous reference for P¹. If the
        // degenerate branch misses the matrix-valued in-block occupation
        // response F¹_B, the error shows up here, in-block, at O(f'Λ).
        let p0_at = |sign: f64| -> Matrix {
            let mut sys = system.clone();
            sys.atoms[dof / 3].position.x += sign * h;
            run_electronic(&sys, &params, options.clone()).unwrap().density
        };
        let (p0p, p0m) = (p0_at(1.0), p0_at(-1.0));
        let mut dp1 = Matrix::zeros(n, n);
        let mut worst1 = (0.0_f64, 0usize, 0usize);
        for i in 0..n {
            for j in 0..n {
                let fd = (p0p[(i, j)] - p0m[(i, j)]) / (2.0 * h);
                dp1[(i, j)] = fx.bundle.density[(i, j)] - fd;
                if dp1[(i, j)].abs() > worst1.0 {
                    worst1 = (dp1[(i, j)].abs(), i, j);
                }
            }
        }
        let tmp1 = dp1.matmul(&sc).unwrap();
        let dmo1 = sc.transpose().matmul(&tmp1).unwrap();
        let mut entries1: Vec<(f64, usize, usize)> = Vec::new();
        for p in 0..n {
            for q in 0..n {
                entries1.push((dmo1[(p, q)].abs(), p, q));
            }
        }
        entries1.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!(
            "FIRST-order P1 vs dP0/dlambda: worst AO {:.3e} at ({}, {}); top MO entries:",
            worst1.0, worst1.1, worst1.2
        );
        for &(d, p, q) in entries1.iter().take(8) {
            println!(
                "  {d:.3e}  ({p:2}, {q:2})  [{}, {}]",
                in_block[p], in_block[q]
            );
        }
    }
}
