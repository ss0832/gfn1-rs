// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic damped-multipole Ewald engine (plan A1).
//!
//! The periodic multipole electrostatics needs the Cartesian derivative tensors of the
//! Ewald-partitioned Klopman–Ohno kernel. Following the existing monopole code
//! ([`crate::pbc::ewald`]) the KO γ is split as
//! `γ_KO = [1/R − 1/(2η²R³)]_Ewald + [KO − 1/R + 1/(2η²R³)]_short-range`, and each piece's
//! `∇^{l+m}` is built from a **radial-derivative array** fed into the molecular
//! symmetric-Cartesian engine [`crate::multipole::grad_tensor_unique`] — exactly the role of
//! the molecular [`crate::multipole::radial_derivs`], but for the Ewald-screened kernels.
//!
//! This increment provides the foundational real-space radial-derivative array for the
//! screened `1/R` (`erfc`) kernel. The remaining pieces (the damped `1/R³` term, the
//! reciprocal-space sum, the four G=0/boundary corrections, and the SCF/gradient/stress
//! assembly) build on top of it.

use crate::lattice::{ImageOffset, Lattice};
use crate::math::Vec3;
use crate::nmr::boys;
use crate::system::PeriodicSystem;
use rayon::prelude::*;

const SQRT_PI: f64 = 1.772_453_850_905_516;

/// Contract a unique rank-2 Cartesian tensor (`[xx,xy,xz,yy,yz,zz]`) with two vectors:
/// `Σ_{ij} a_i M_ij b_j`.
fn contract_rank2(unique: &[f64], a: Vec3, b: Vec3) -> f64 {
    let m = [
        [unique[0], unique[1], unique[2]],
        [unique[1], unique[3], unique[4]],
        [unique[2], unique[4], unique[5]],
    ];
    let av = [a.x, a.y, a.z];
    let bv = [b.x, b.y, b.z];
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += av[i] * m[i][j] * bv[j];
        }
    }
    s
}

#[inline]
fn negated(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

/// Matrix–vector product of a unique rank-2 tensor (`[xx,xy,xz,yy,yz,zz]`) with a vector.
fn matvec_rank2(unique: &[f64], v: Vec3) -> Vec3 {
    let m = [
        [unique[0], unique[1], unique[2]],
        [unique[1], unique[3], unique[4]],
        [unique[2], unique[4], unique[5]],
    ];
    let vv = [v.x, v.y, v.z];
    Vec3::new(
        m[0][0] * vv[0] + m[0][1] * vv[1] + m[0][2] * vv[2],
        m[1][0] * vv[0] + m[1][1] * vv[1] + m[1][2] * vv[2],
        m[2][0] * vv[0] + m[2][1] * vv[1] + m[2][2] * vv[2],
    )
}

/// Periodic **monopole (charge–charge)** energy `½ Σ_{A,B} q_A q_B γ_KO(R_AB)` (single hardness
/// `eta`) assembled from the multipole machinery at rank 0, **including the two charge-sector
/// G=0 corrections**: the Coulomb neutralizing background `−π/(α²V)` and the KO-`R⁻³` `r3_k0`
/// completion `½η⁻²·(4π/V)·log-term` — both present even for a neutral cell, and the background
/// is what makes a **non-neutral** cell α-independent. Reproduces `periodic_gamma_matrix`'s
/// scalar role; validated here by α-independence on a charged cell.
pub fn periodic_monopole_energy(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    eta: f64,
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_monopole_energy requires a periodic system");
    let inv_eta2 = 1.0 / (eta * eta);
    let volume = lattice.volume();
    // The α-dependent `erfc`/`q/r³` parts converge by `TAU/α`, but the α-*independent* KO
    // short-range residual decays only as `r⁻⁵` at rank 0, so the real sum needs a **fixed**
    // floor cutoff for α-independence (the A2 energy assembly must split the residual onto its
    // own fixed cutoff; here a generous floor suffices for the scalar validation).
    let real_cut = (crate::pbc::ewald::TAU / alpha).max(45.0);
    let images = lattice.image_offsets(real_cut);
    let gws = ko_recip_weights(lattice, alpha, inv_eta2);
    let self0 = ewald_self_tensor_ko(alpha, eta, 0)[0];
    let background = -std::f64::consts::PI / (alpha * alpha * volume); // Coulomb 1/R G=0
    let r3_k0 = crate::pbc::ewald::QCORE_R3_COEFF
        * (4.0 * std::f64::consts::PI * crate::pbc::ewald::qcore_r3_k0_log(alpha) / volume)
        * inv_eta2; // KO R⁻³ G=0 completion
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            let mut gamma = background + r3_k0 + ewald_recip_potential(rab, &gws);
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                gamma += periodic_multipole_real_fmn(r, alpha, eta, 0, 0)[0];
            }
            if a == b {
                gamma -= self0; // remove the spurious smooth self the reciprocal includes
            }
            energy += 0.5 * charges[a] * charges[b] * gamma;
        }
    }
    energy
}

/// Full contraction of a unique rank-2 tensor (`[xx,xy,xz,yy,yz,zz]`) with a symmetric `3×3`:
/// `Σ_{ij} M_ij Q_ij`.
fn contract_rank2_full(unique: &[f64], q: &[[f64; 3]; 3]) -> f64 {
    let m = [
        [unique[0], unique[1], unique[2]],
        [unique[1], unique[3], unique[4]],
        [unique[2], unique[4], unique[5]],
    ];
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += m[i][j] * q[i][j];
        }
    }
    s
}

/// Periodic **charge–quadrupole** cross interaction `E = Σ_{A,B} q_A · f^(02)_AB : Q_B` for the
/// full KO kernel (`f^(02)` is rank 2; prefactor `+½` from `(−1)²/(0!2!)`). Real-space screened
/// `f^(02)` over images (skip the `A=B` home cell) + reciprocal (`1/R` + QCore `R⁻³`) + the
/// rank-2 self correction. With a **charge-neutral** set the energy is α-independent —
/// validating the monopole↔quadrupole channel.
pub fn periodic_charge_quad_energy(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    quads: &[[[f64; 3]; 3]],
    eta: f64,
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_charge_quad_energy requires a periodic system");
    let inv_eta2 = 1.0 / (eta * eta);
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let gws = ko_recip_weights(lattice, alpha, inv_eta2);
    let self2 = ewald_self_tensor_ko(alpha, eta, 2); // f^(02) self = ½·this
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                // f^(02)_real already carries the +½ prefactor.
                let f02 = periodic_multipole_real_fmn(r, alpha, eta, 0, 2);
                energy += charges[a] * contract_rank2_full(&f02, &quads[b]);
            }
            let recip = ewald_recip_tensor(rab, &gws, 2);
            energy += charges[a] * 0.5 * contract_rank2_full(&recip, &quads[b]);
        }
        energy -= charges[a] * 0.5 * contract_rank2_full(&self2, &quads[a]);
    }
    energy
}

/// Dot a unique rank-1 Cartesian tensor (`[x,y,z]`) with a vector.
#[inline]
fn contract_rank1(unique: &[f64], v: Vec3) -> f64 {
    unique[0] * v.x + unique[1] * v.y + unique[2] * v.z
}

/// Periodic **charge–dipole** cross interaction `E = Σ_{A,B} q_A · F^(01)_AB · d_B` for the full
/// KO kernel (`f^(01)` is rank 1, odd ⇒ no self correction). Real-space screened `f^(01)`
/// summed over images (skip the `A=B` home cell) + reciprocal (`1/R` + QCore `R⁻³`). With a
/// **charge-neutral** set `Σ q_A = 0` no background/G=0 term is needed; the energy is then
/// α-independent — validating the mixed-rank (monopole↔dipole) periodic tensor the SCF uses.
pub fn periodic_charge_dipole_energy(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    eta: f64,
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_charge_dipole_energy requires a periodic system");
    let inv_eta2 = 1.0 / (eta * eta);
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let gws = ko_recip_weights(lattice, alpha, inv_eta2);
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f01 = periodic_multipole_real_fmn(r, alpha, eta, 0, 1);
                energy += charges[a] * contract_rank1(&f01, dipoles[b]);
            }
            let recip = negated(&ewald_recip_tensor(rab, &gws, 1));
            energy += charges[a] * contract_rank1(&recip, dipoles[b]);
        }
    }
    energy
}

/// Periodic **charge–dipole** cross interaction with **per-atom hardnesses** (the heteroatomic
/// monopole↔dipole coupling the joint SCF mixer couples): `η_AB = harmonic_mean(η_A, η_B)`.
/// `f^(01)` is rank 1 (odd ⇒ no self). With a charge-neutral set the energy is α-independent;
/// reduces to [`periodic_charge_dipole_energy`] when all hardnesses are equal.
pub fn periodic_charge_dipole_energy_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_charge_dipole_energy_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f01 = periodic_multipole_real_fmn(r, alpha, eta_ab, 0, 1);
                energy += charges[a] * contract_rank1(&f01, dipoles[b]);
            }
            let recip = negated(&ewald_recip_tensor(rab, &gws, 1));
            energy += charges[a] * contract_rank1(&recip, dipoles[b]);
        }
    }
    energy
}

/// Both SCF potentials of the periodic **charge–dipole** cross energy
/// [`periodic_charge_dipole_energy_pairwise`]: the charge potential `V_q[A] = ∂E/∂q_A` (folds into
/// the shell-charge Fock route) and the dipole field `V_d[B] = ∂E/∂d_B` (folds into the on-site
/// moment Fock route) — the two routes the joint SCF mixer couples. With `K_AB = F^(01)(R_AB)`
/// (the rank-1 charge→dipole kernel vector, real images + reciprocal), `V_q[A] = Σ_B K_AB·d_B` and
/// `V_d[B] = Σ_A q_A K_AB`. The energy is **bilinear** (degree 1 in `q`, degree 1 in `d`), so the
/// Euler identities `Σ_A q_A V_q[A] = Σ_B d_B·V_d[B] = E` hold exactly (a no-FD cross-check).
pub fn periodic_charge_dipole_fields_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> (Vec<f64>, Vec<Vec3>) {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_charge_dipole_fields_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut v_q = vec![0.0_f64; nat];
    let mut v_d = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            // Charge→dipole kernel vector K_AB = F^(01)(R_AB): real images + reciprocal.
            let mut kab = Vec3::zero();
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f01 = periodic_multipole_real_fmn(r, alpha, eta_ab, 0, 1);
                kab += Vec3::new(f01[0], f01[1], f01[2]);
            }
            let recip = negated(&ewald_recip_tensor(rab, &gws, 1));
            kab += Vec3::new(recip[0], recip[1], recip[2]);
            // E_AB = q_A (K_AB · d_B).
            v_q[a] += kab.dot(dipoles[b]);
            v_d[b] += kab * charges[a];
        }
    }
    (v_q, v_d)
}

/// The complete **dipole-rank** periodic multipole SCF field bundle: the total per-atom charge
/// potential `V_q[A]` and dipole field `V_d[A]` from the dipole–dipole self-interaction **plus**
/// the charge–dipole cross term — the exact two Fock routes the joint SCF mixer injects at dipole
/// rank (the charge–charge monopole γ stays in [`crate::pbc::ewald::periodic_gamma_matrix`]). The
/// dipole field is `V_d = V_d^{dd} + V_d^{cd}` and the charge potential `V_q = V_q^{cd}` (the
/// monopole side carries no extra multipole charge potential). With the combined multipole energy
/// `E_mp = E_dd + E_cd` the **mixed Euler identity** holds exactly:
/// `Σ_A d_A·V_d[A] + Σ_A q_A·V_q[A] = 2 E_mp` (the degree-2 dipole–dipole part gives `2 E_dd`; the
/// bilinear cross gives `E_cd` from each of the `q` and `d` sides). This is the single call the
/// Γ/k-point multipole SCC loop makes per iteration.
pub fn periodic_dipole_rank_fields(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> (Vec<f64>, Vec<Vec3>) {
    let v_d_dd = periodic_dipole_field_ko_pairwise(system, alpha, dipoles, hardnesses);
    let (v_q, v_d_cd) =
        periodic_charge_dipole_fields_pairwise(system, alpha, charges, dipoles, hardnesses);
    let nat = system.atoms.len();
    let v_d: Vec<Vec3> = (0..nat).map(|a| v_d_dd[a] + v_d_cd[a]).collect();
    (v_q, v_d)
}

/// **Arbitrary-rank** periodic multipole field `V[A][l]` (per atom, per rank `l = 0..=max_rank`),
/// the periodic analog of the molecular [`crate::multipole::multipole_fields_generic`] and the
/// generalization of [`periodic_dipole_rank_fields`] to quadrupole and beyond. For every ordered
/// pair `(A rank la, B rank lb)` it assembles the full Ewald-split KO tensor `f^(la,lb)` — real
/// images ([`periodic_multipole_real_fmn`], skip the `A=B` home cell) + reciprocal
/// `pref(la,lb)·`[`ewald_recip_tensor`] — and contracts the source moment `M[B][lb]` with it
/// (via [`crate::multipole::contract_last_unique`], the molecular convention), accumulating into
/// `V[A][la]`. The rank-diagonal **self** correction (`la = lb`, the atom's own hardness) is
/// subtracted. The monopole–monopole `(0,0)` block is omitted (carried by
/// [`crate::pbc::ewald::periodic_gamma_matrix`]). `moments[A][l]` and the returned `V[A][l]` are
/// in the full `3^l` Cartesian layout (matching [`crate::multipole::build_generic_moments`]); the
/// energy is `E = ½ Σ_A Σ_l M[A][l]·V[A][l]`. α-independent at every rank.
pub fn periodic_multipole_fields_generic(
    system: &PeriodicSystem,
    alpha: f64,
    moments: &[Vec<Vec<f64>>],
    hardnesses: &[f64],
    max_rank: usize,
) -> Vec<Vec<Vec<f64>>> {
    periodic_multipole_fields_generic_direct(system, alpha, moments, hardnesses, max_rank)
}

/// Geometry-only linear operator for [`periodic_multipole_fields_generic`].
///
/// The periodic multipole SCC repeatedly evaluates the same Ewald-split tensor
/// blocks for a fixed geometry while only the source moments change. Caching
/// those blocks once per SCC step removes the expensive real/reciprocal lattice
/// tensor reconstruction from every charge-mixing iteration. The stored blocks
/// map one source atom/rank moment to one target atom/rank field in the same
/// full Cartesian layout used by `build_generic_moments`.
#[derive(Clone, Debug)]
pub struct PeriodicMultipoleFieldKernel {
    nat: usize,
    max_rank: usize,
    dims: Vec<usize>,
    tensors: Vec<Option<Vec<f64>>>,
}

const DEFAULT_PERIODIC_MULTIPOLE_FIELD_KERNEL_LIMIT_BYTES: usize = 256 * 1024 * 1024;

impl PeriodicMultipoleFieldKernel {
    pub fn try_build(
        system: &PeriodicSystem,
        alpha: f64,
        hardnesses: &[f64],
        max_rank: usize,
    ) -> Option<Self> {
        let estimate = Self::estimated_bytes(system.atoms.len(), max_rank);
        if estimate <= periodic_multipole_field_kernel_cache_limit_bytes() {
            Some(Self::build(system, alpha, hardnesses, max_rank))
        } else {
            None
        }
    }

    pub fn estimated_bytes(nat: usize, max_rank: usize) -> usize {
        let mut per_pair_components = 0usize;
        for lb in 0..=max_rank {
            for la in 0..=max_rank {
                if la == 0 && lb == 0 {
                    continue;
                }
                per_pair_components = per_pair_components
                    .saturating_add(crate::integrals::cartesian_rank_components(la + lb).len());
            }
        }
        let nslot = nat
            .saturating_mul(max_rank + 1)
            .saturating_mul(nat)
            .saturating_mul(max_rank + 1);
        nat.saturating_mul(nat)
            .saturating_mul(per_pair_components)
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_add(nslot.saturating_mul(std::mem::size_of::<Option<Vec<f64>>>()))
    }

    pub fn build(system: &PeriodicSystem, alpha: f64, hardnesses: &[f64], max_rank: usize) -> Self {
        let lattice = system
            .lattice
            .as_ref()
            .expect("PeriodicMultipoleFieldKernel requires a periodic system");
        let real_cut = crate::pbc::ewald::TAU / alpha;
        let images = lattice.image_offsets(real_cut);
        let nat = system.atoms.len();
        let dims: Vec<usize> = (0..=max_rank).map(full_rank_len).collect();
        let mut recip_cache: std::collections::HashMap<u64, Vec<(Vec3, f64)>> =
            std::collections::HashMap::new();
        let nslot = nat * (max_rank + 1) * nat * (max_rank + 1);
        let mut tensors = vec![None; nslot];

        for a in 0..nat {
            for b in 0..nat {
                let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
                let inv_eta2 = 1.0 / (eta_ab * eta_ab);
                let gws = recip_cache
                    .entry(inv_eta2.to_bits())
                    .or_insert_with(|| ko_recip_weights(lattice, alpha, inv_eta2));
                let rab = system.atoms[a].position - system.atoms[b].position;
                let mut pair_tensors: Vec<Option<Vec<f64>>> = vec![None; 2 * max_rank + 1];
                for lb in 0..=max_rank {
                    for la in 0..=max_rank {
                        if la == 0 && lb == 0 {
                            continue; // monopole-monopole is carried by periodic_gamma_matrix
                        }
                        let k = la + lb;
                        let base = pair_tensors[k].get_or_insert_with(|| {
                            periodic_multipole_pair_base_tensor(
                                lattice,
                                &images,
                                rab,
                                alpha,
                                eta_ab,
                                gws,
                                a == b,
                                k,
                            )
                        });
                        let pref = multipole_rank_prefactor(la, lb);
                        let mut f: Vec<f64> = base.iter().map(|v| pref * v).collect();
                        if a == b && la == lb && la >= 1 {
                            let self_t = ewald_self_tensor_ko(alpha, hardnesses[a], 2 * la);
                            for (acc, v) in f.iter_mut().zip(self_t.iter()) {
                                *acc -= pref * v;
                            }
                        }
                        let idx = field_block_index(nat, max_rank, a, la, b, lb);
                        tensors[idx] = Some(f);
                    }
                }
            }
        }

        Self {
            nat,
            max_rank,
            dims,
            tensors,
        }
    }

    pub fn apply(&self, moments: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        debug_assert_eq!(moments.len(), self.nat);
        let active = crate::multipole::moment_active_mask(moments, self.nat, self.max_rank);
        let mut field: Vec<Vec<Vec<f64>>> = (0..self.nat)
            .map(|_| {
                (0..=self.max_rank)
                    .map(|l| vec![0.0_f64; self.dims[l]])
                    .collect()
            })
            .collect();
        for a in 0..self.nat {
            for b in 0..self.nat {
                for lb in 0..=self.max_rank {
                    if lb >= 1 && !active[b][lb] {
                        continue;
                    }
                    let source = &moments[b][lb];
                    debug_assert_eq!(source.len(), self.dims[lb]);
                    for la in 0..=self.max_rank {
                        if la == 0 && lb == 0 {
                            continue;
                        }
                        let idx = field_block_index(self.nat, self.max_rank, a, la, b, lb);
                        let Some(tensor) = self.tensors[idx].as_ref() else {
                            continue;
                        };
                        let contrib =
                            crate::multipole::contract_last_unique(tensor, la, lb, source);
                        for (acc, v) in field[a][la].iter_mut().zip(contrib.iter()) {
                            *acc += v;
                        }
                    }
                }
            }
        }
        field
    }
}

#[inline]
fn full_rank_len(l: usize) -> usize {
    3usize.pow(l as u32)
}

#[inline]
fn field_block_index(
    nat: usize,
    max_rank: usize,
    a: usize,
    la: usize,
    b: usize,
    lb: usize,
) -> usize {
    (((a * (max_rank + 1) + la) * nat + b) * (max_rank + 1)) + lb
}

fn periodic_multipole_field_kernel_cache_limit_bytes() -> usize {
    std::env::var("GFN1_PBC_MULTIPOLE_FIELD_CACHE_MB")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|mb| mb.saturating_mul(1024 * 1024))
        .unwrap_or(DEFAULT_PERIODIC_MULTIPOLE_FIELD_KERNEL_LIMIT_BYTES)
}

fn periodic_multipole_pair_base_tensor(
    lattice: &Lattice,
    images: &[ImageOffset],
    rab: Vec3,
    alpha: f64,
    eta: f64,
    gws: &[(Vec3, f64)],
    same_atom: bool,
    k: usize,
) -> Vec<f64> {
    let mut f = vec![0.0_f64; crate::integrals::cartesian_rank_components(k).len()];
    for off in images {
        if same_atom && off.is_origin() {
            continue;
        }
        let r = rab - lattice.translation(*off);
        let fr = periodic_multipole_real_base_tensor(r, alpha, eta, k);
        for (acc, v) in f.iter_mut().zip(fr.iter()) {
            *acc += v;
        }
    }
    let recip = ewald_recip_tensor(rab, gws, k);
    for (acc, v) in f.iter_mut().zip(recip.iter()) {
        *acc += v;
    }
    f
}

fn periodic_multipole_pair_base_grad_tensor(
    lattice: &Lattice,
    images: &[ImageOffset],
    rab: Vec3,
    alpha: f64,
    eta: f64,
    gws: &[(Vec3, f64)],
    same_atom: bool,
    k: usize,
) -> Vec<f64> {
    let mut gt = vec![0.0_f64; crate::integrals::cartesian_rank_components(k).len()];
    for off in images {
        if same_atom && off.is_origin() {
            continue;
        }
        let r = rab - lattice.translation(*off);
        let gr = periodic_multipole_real_grad_base_tensor(r, alpha, eta, k);
        for (acc, v) in gt.iter_mut().zip(gr.iter()) {
            *acc += v;
        }
    }
    let recip = ewald_recip_tensor(rab, gws, k);
    for (acc, v) in gt.iter_mut().zip(recip.iter()) {
        *acc += v;
    }
    gt
}

#[allow(dead_code)]
fn periodic_multipole_fields_generic_direct(
    system: &PeriodicSystem,
    alpha: f64,
    moments: &[Vec<Vec<f64>>],
    hardnesses: &[f64],
    max_rank: usize,
) -> Vec<Vec<Vec<f64>>> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_multipole_fields_generic requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let full = |l: usize| 3usize.pow(l as u32);
    let fact = |n: usize| (1..=n).product::<usize>().max(1) as f64;
    let pref_mn = |m: usize, n: usize| (if n % 2 == 0 { 1.0 } else { -1.0 }) / (fact(m) * fact(n));
    // Source-rank screening: a rank-`lb` source whose moment is ~0 contributes nothing to any
    // field, and a rank-`l` self term with a ~0 moment is ~0. Skipping them makes the high-rank
    // part `O(N·n_active)` (few atoms carry high-rank traceless moments) — the large-system win.
    let active = crate::multipole::moment_active_mask(moments, nat, max_rank);
    let mut recip_cache: std::collections::HashMap<u64, Vec<(Vec3, f64)>> =
        std::collections::HashMap::new();
    let mut field: Vec<Vec<Vec<f64>>> = (0..nat)
        .map(|_| (0..=max_rank).map(|l| vec![0.0_f64; full(l)]).collect())
        .collect();
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let inv_eta2 = 1.0 / (eta_ab * eta_ab);
            let gws = recip_cache
                .entry(inv_eta2.to_bits())
                .or_insert_with(|| ko_recip_weights(lattice, alpha, inv_eta2));
            let rab = system.atoms[a].position - system.atoms[b].position;
            let mut pair_tensors: Vec<Option<Vec<f64>>> = vec![None; 2 * max_rank + 1];
            for lb in 0..=max_rank {
                if lb >= 1 && !active[b][lb] {
                    continue; // zero source moment ⇒ no field contribution
                }
                for la in 0..=max_rank {
                    if la == 0 && lb == 0 {
                        continue; // monopole–monopole carried by periodic_gamma_matrix
                    }
                    let k = la + lb;
                    let base = pair_tensors[k].get_or_insert_with(|| {
                        periodic_multipole_pair_base_tensor(
                            lattice,
                            &images,
                            rab,
                            alpha,
                            eta_ab,
                            gws,
                            a == b,
                            k,
                        )
                    });
                    let pref = multipole_rank_prefactor(la, lb);
                    let f: Vec<f64> = base.iter().map(|v| pref * v).collect();
                    let contrib =
                        crate::multipole::contract_last_unique(&f, la, lb, &moments[b][lb]);
                    for (acc, v) in field[a][la].iter_mut().zip(contrib.iter()) {
                        *acc += v;
                    }
                }
            }
        }
        // Rank-diagonal self correction (atom's own hardness): − pref(l,l)·self_tensor · M[A][l].
        for l in 1..=max_rank {
            if !active[a][l] {
                continue; // zero moment ⇒ zero self contribution
            }
            let pref = pref_mn(l, l);
            let self_t = ewald_self_tensor_ko(alpha, hardnesses[a], 2 * l);
            let scaled: Vec<f64> = self_t.iter().map(|v| pref * v).collect();
            let contrib = crate::multipole::contract_last_unique(&scaled, l, l, &moments[a][l]);
            for (acc, v) in field[a][l].iter_mut().zip(contrib.iter()) {
                *acc -= v;
            }
        }
    }
    field
}

/// **Arbitrary-rank** periodic multipole kernel forces `F_A = −∂E/∂R_A` at **fixed** moments — the
/// inter-atomic (Hellmann–Feynman) force of `E = ½ Σ_A Σ_l M[A][l]·V[A][l]` the analytic gradient
/// adds once the moments are SCF-converged. Generalizes [`periodic_dipole_rank_forces`]: for every
/// ordered rank pair `(la, lb)` it builds the rank-`(la+lb+1)` gradient tensor — real images
/// ([`periodic_multipole_real_grad_fmn`]) + `pref(la,lb)·`[`ewald_recip_tensor`] one rank higher —
/// and contracts it with `M[A][la]`, `M[B][lb]` via [`crate::multipole::kernel_grad_unique`]. The
/// `½` with both `(la,lb)` orderings reproduces the full cross-term force; the rank-diagonal
/// **self** term is atom-centred ⇒ no force. The monopole–monopole `(0,0)` force stays in
/// [`crate::pbc::gradient`]. `moments[A][l]` in full `3^l` layout (as `build_generic_moments`).
pub fn periodic_multipole_forces_generic(
    system: &PeriodicSystem,
    alpha: f64,
    moments: &[Vec<Vec<f64>>],
    hardnesses: &[f64],
    max_rank: usize,
) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_multipole_forces_generic requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    // Both moments are fixed (converged) here, so a force term vanishes if EITHER rank's moment is
    // ~0 — screen both indices for the large-system `O(N·n_active)` high-rank path.
    let active = crate::multipole::moment_active_mask(moments, nat, max_rank);
    (0..nat)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![Vec3::zero(); nat],
                    std::collections::HashMap::<u64, Vec<(Vec3, f64)>>::new(),
                )
            },
            |(mut local_forces, mut recip_cache), a| {
                for b in 0..nat {
                    let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
                    let inv_eta2 = 1.0 / (eta_ab * eta_ab);
                    let gws = recip_cache
                        .entry(inv_eta2.to_bits())
                        .or_insert_with(|| ko_recip_weights(lattice, alpha, inv_eta2));
                    let rab = system.atoms[a].position - system.atoms[b].position;
                    let mut pair_grad_tensors: Vec<Option<Vec<f64>>> = vec![None; 2 * max_rank + 2];
                    for lb in 0..=max_rank {
                        if lb >= 1 && !active[b][lb] {
                            continue;
                        }
                        for la in 0..=max_rank {
                            if la == 0 && lb == 0 {
                                continue;
                            }
                            if la >= 1 && !active[a][la] {
                                continue;
                            }
                            let k = la + lb + 1;
                            let base = pair_grad_tensors[k].get_or_insert_with(|| {
                                periodic_multipole_pair_base_grad_tensor(
                                    lattice,
                                    &images,
                                    rab,
                                    alpha,
                                    eta_ab,
                                    gws,
                                    a == b,
                                    k,
                                )
                            });
                            let pref = multipole_rank_prefactor(la, lb);
                            let gt: Vec<f64> = base.iter().map(|v| pref * v).collect();
                            let g = crate::multipole::kernel_grad_unique(
                                &gt,
                                la,
                                lb,
                                &moments[a][la],
                                &moments[b][lb],
                            );
                            local_forces[a] -= g * 0.5;
                            local_forces[b] += g * 0.5;
                        }
                    }
                }
                (local_forces, recip_cache)
            },
        )
        .map(|(forces, _)| forces)
        .reduce(
            || vec![Vec3::zero(); nat],
            |mut acc, forces| {
                for (a, f) in acc.iter_mut().zip(forces.iter()) {
                    *a += *f;
                }
                acc
            },
        )
}

/// Per-atom dipole **field** for the full KO kernel with **per-atom hardnesses** — the
/// heteroatomic SCF Fock shift, `V_A = ∂E/∂d_A` of [`periodic_dipole_dipole_energy_ko_pairwise`].
pub fn periodic_dipole_field_ko_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_field_ko_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut field = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f11 = periodic_multipole_real_fmn(r, alpha, eta_ab, 1, 1);
                field[a] += matvec_rank2(&f11, dipoles[b]);
            }
            let recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            field[a] += matvec_rank2(&recip, dipoles[b]);
        }
        let k_self = negated(&ewald_self_tensor_ko(alpha, hardnesses[a], 2));
        field[a] -= matvec_rank2(&k_self, dipoles[a]);
    }
    field
}

/// Per-atom **dipole field** for the **full GFN1 KO** kernel (single hardness `eta`):
/// `V_A = ∂E/∂d_A` of [`periodic_dipole_dipole_energy_ko`] — the multipole Fock shift the
/// joint-mixing SCF actually needs. Real-space full-KO `f^(11)` over images + reciprocal
/// (`1/R` + QCore `R⁻³`) + full-KO self.
pub fn periodic_dipole_field_ko(
    system: &PeriodicSystem,
    alpha: f64,
    dipoles: &[Vec3],
    eta: f64,
) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_field_ko requires a periodic system");
    let inv_eta2 = 1.0 / (eta * eta);
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let gws = ko_recip_weights(lattice, alpha, inv_eta2);
    let k_self = negated(&ewald_self_tensor_ko(alpha, eta, 2));
    let nat = system.atoms.len();
    let mut field = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f11 = periodic_multipole_real_fmn(r, alpha, eta, 1, 1);
                field[a] += matvec_rank2(&f11, dipoles[b]);
            }
            let recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            field[a] += matvec_rank2(&recip, dipoles[b]);
        }
        field[a] -= matvec_rank2(&k_self, dipoles[a]);
    }
    field
}

/// Per-atom **dipole field** `V_A = ∂E/∂d_A = Σ_B F^(11)_AB · d_B − F^(11)_self · d_A` for the
/// periodic dipole–dipole interaction ([`periodic_dipole_dipole_energy`]). This is the
/// quantity the joint-mixing multipole SCF needs: the field conjugate to each atomic moment
/// (the multipole Fock shift). It is the exact gradient of the energy w.r.t. the dipoles.
pub fn periodic_dipole_field(system: &PeriodicSystem, alpha: f64, dipoles: &[Vec3]) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_field requires a periodic system");
    let volume = lattice.volume();
    let tau = crate::pbc::ewald::TAU;
    let real_cut = tau / alpha;
    let g_cut = 2.0 * alpha * tau;
    let images = lattice.image_offsets(real_cut);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * std::f64::consts::PI / volume;
    let gws: Vec<(Vec3, f64)> = lattice
        .reciprocal_vectors_within(g_cut, false)
        .into_iter()
        .map(|(_, g)| {
            let g2 = g.norm2();
            (g, four_pi_v * (-g2 * inv_4a2).exp() / g2)
        })
        .collect();
    let k_self = negated(&ewald_self_tensor_1r(alpha, 2));
    let nat = system.atoms.len();
    let mut field = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let k_real = negated(&ewald_real_tensor(r, alpha, 2));
                field[a] += matvec_rank2(&k_real, dipoles[b]);
            }
            let k_recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            field[a] += matvec_rank2(&k_recip, dipoles[b]);
        }
        field[a] -= matvec_rank2(&k_self, dipoles[a]);
    }
    field
}

/// Fully contract a unique rank-`k` symmetric Cartesian tensor with `w^{⊗k}`:
/// `Σ_{lx+ly+lz=k} multinomial(lx,ly,lz)·T[lx,ly,lz]·w_x^{lx} w_y^{ly} w_z^{lz}`.
fn contract_w(unique: &[f64], k: usize, w: Vec3) -> f64 {
    let wv = [w.x, w.y, w.z];
    let fact = |n: usize| (1..=n).product::<usize>().max(1);
    crate::integrals::cartesian_rank_components(k)
        .iter()
        .enumerate()
        .map(|(i, &(lx, ly, lz))| {
            let mult = (fact(k) / (fact(lx) * fact(ly) * fact(lz))) as f64;
            mult * unique[i] * wv[0].powi(lx as i32) * wv[1].powi(ly as i32) * wv[2].powi(lz as i32)
        })
        .sum()
}

/// Arbitrary-rank generalization of the periodic multipole-tensor energy, in the pure-`1/R`
/// tinfoil limit, with every atom carrying the same rank-`K` "moment direction" `w^{⊗K}`:
/// `E = ½ Σ_{A,B}[Σ_T' ∇^K(erfc/r)(R_AB+T) + ∇^K γ_recip(R_AB)] : w^K − ½ Σ_A ∇^K(erf/r)|₀ : w^K`.
/// The full Ewald sum (real + reciprocal + self) is α-independent at every (even) `K`, which
/// validates the assembly's rank-generality and the higher-rank self correction.
pub fn periodic_multipole_w_energy(system: &PeriodicSystem, alpha: f64, k: usize, w: Vec3) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_multipole_w_energy requires a periodic system");
    let volume = lattice.volume();
    let tau = crate::pbc::ewald::TAU;
    let real_cut = tau / alpha;
    let g_cut = 2.0 * alpha * tau;
    let images = lattice.image_offsets(real_cut);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * std::f64::consts::PI / volume;
    let gws: Vec<(Vec3, f64)> = lattice
        .reciprocal_vectors_within(g_cut, false)
        .into_iter()
        .map(|(_, g)| {
            let g2 = g.norm2();
            (g, four_pi_v * (-g2 * inv_4a2).exp() / g2)
        })
        .collect();
    let self_t = ewald_self_tensor_1r(alpha, k);
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                energy += 0.5 * contract_w(&ewald_real_tensor(r, alpha, k), k, w);
            }
            energy += 0.5 * contract_w(&ewald_recip_tensor(rab, &gws, k), k, w);
        }
        energy -= 0.5 * contract_w(&self_t, k, w);
    }
    energy
}

/// Periodic dipole–dipole energy for the full KO kernel with **per-atom hardnesses** (the
/// heteroatomic case the SCF needs): each pair uses `η_AB = harmonic_mean(η_A, η_B)`, the
/// on-site self uses `η_A`. Because the QCore `R⁻³` reciprocal weight depends on the pairwise
/// `η`, the reciprocal is evaluated per pair (as in the monopole `periodic_gamma_matrix`).
/// α-independent for any hardness set; reduces to [`periodic_dipole_dipole_energy_ko`] when all
/// hardnesses are equal.
pub fn periodic_dipole_dipole_energy_ko_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_dipole_energy_ko_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let f11 = periodic_multipole_real_fmn(r, alpha, eta_ab, 1, 1);
                energy += 0.5 * contract_rank2(&f11, dipoles[a], dipoles[b]);
            }
            let k_recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            energy += 0.5 * contract_rank2(&k_recip, dipoles[a], dipoles[b]);
        }
        let k_self = negated(&ewald_self_tensor_ko(alpha, hardnesses[a], 2));
        energy -= 0.5 * contract_rank2(&k_self, dipoles[a], dipoles[a]);
    }
    energy
}

/// Inter-atomic **forces** `F_A = −∂E/∂R_A` of the periodic dipole–dipole energy
/// [`periodic_dipole_dipole_energy_ko_pairwise`] at **fixed** dipoles — the kernel (Hellmann–
/// Feynman) force the analytic gradient adds once the atomic moments are SCF-converged. The
/// **self** term is atom-centred (position-independent at fixed `α`/`η`) ⇒ contributes no force.
/// The real-space (`∇f^(11)`) and reciprocal (`∇` recip tensor = the rank-3 recip tensor)
/// gradients are contracted with the two dipoles via [`crate::multipole::kernel_grad_unique`];
/// `∂R_AB/∂R_A = +1`, `∂R_AB/∂R_B = −1`, and `force = −∂E/∂R`.
pub fn periodic_dipole_dipole_forces_ko_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_dipole_forces_ko_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut forces = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            let da = [dipoles[a].x, dipoles[a].y, dipoles[a].z];
            let db = [dipoles[b].x, dipoles[b].y, dipoles[b].z];
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let gf = periodic_multipole_real_grad_fmn(r, alpha, eta_ab, 1, 1);
                let grad = crate::multipole::kernel_grad_unique(&gf, 1, 1, &da, &db);
                // E term is ½ d_A·f^(11)·d_B ⇒ ∂E/∂R_A = +½ grad, ∂E/∂R_B = −½ grad.
                forces[a] -= grad * 0.5;
                forces[b] += grad * 0.5;
            }
            // Reciprocal: ∇ of −recip_2 is −recip_3 (the rank-3 reciprocal tensor).
            let recip3 = negated(&ewald_recip_tensor(rab, &gws, 3));
            let grad_recip = crate::multipole::kernel_grad_unique(&recip3, 1, 1, &da, &db);
            forces[a] -= grad_recip * 0.5;
            forces[b] += grad_recip * 0.5;
        }
    }
    forces
}

/// Inter-atomic **forces** `F_A = −∂E/∂R_A` of the periodic **charge–dipole** cross energy
/// [`periodic_charge_dipole_energy_pairwise`] at fixed charges + dipoles. Same machinery as the
/// dipole–dipole force one rank lower: `∇f^(01)` is rank 2, contracted with the charge (rank 0)
/// and the source dipole (rank 1) via [`crate::multipole::kernel_grad_unique`]. No `½` (the cross
/// loop runs every ordered `(charge_A, dipole_B)` pair once). `f^(01)` is odd ⇒ no self term.
pub fn periodic_charge_dipole_forces_pairwise(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> Vec<Vec3> {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_charge_dipole_forces_pairwise requires a periodic system");
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let nat = system.atoms.len();
    let mut forces = vec![Vec3::zero(); nat];
    for a in 0..nat {
        for b in 0..nat {
            let eta_ab = crate::coulomb::harmonic_average(hardnesses[a], hardnesses[b]);
            let gws = ko_recip_weights(lattice, alpha, 1.0 / (eta_ab * eta_ab));
            let rab = system.atoms[a].position - system.atoms[b].position;
            let qa = [charges[a]];
            let db = [dipoles[b].x, dipoles[b].y, dipoles[b].z];
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let gf = periodic_multipole_real_grad_fmn(r, alpha, eta_ab, 0, 1);
                let grad = crate::multipole::kernel_grad_unique(&gf, 0, 1, &qa, &db);
                forces[a] -= grad;
                forces[b] += grad;
            }
            let recip2 = negated(&ewald_recip_tensor(rab, &gws, 2));
            let grad_recip = crate::multipole::kernel_grad_unique(&recip2, 0, 1, &qa, &db);
            forces[a] -= grad_recip;
            forces[b] += grad_recip;
        }
    }
    forces
}

/// The complete **dipole-rank** periodic multipole kernel force at fixed moments: dipole–dipole +
/// charge–dipole cross, the inter-atomic `F_A = −∂E_mp/∂R_A` the analytic SCC gradient adds once
/// the moments are converged (the self term is atom-centred ⇒ no force; the charge–charge monopole
/// force stays in [`crate::pbc::gradient`]). `charges` are the mDFTB monopole moments
/// `qm = −(atomic charge)` used in the SCC.
pub fn periodic_dipole_rank_forces(
    system: &PeriodicSystem,
    alpha: f64,
    charges: &[f64],
    dipoles: &[Vec3],
    hardnesses: &[f64],
) -> Vec<Vec3> {
    let dd = periodic_dipole_dipole_forces_ko_pairwise(system, alpha, dipoles, hardnesses);
    let cd = periodic_charge_dipole_forces_pairwise(system, alpha, charges, dipoles, hardnesses);
    (0..system.atoms.len()).map(|a| dd[a] + cd[a]).collect()
}

/// Periodic **dipole–dipole** electrostatic energy for fixed atomic dipoles, in the pure
/// Coulomb (`1/R`) tinfoil-boundary limit — the canonical hard-convergence case validating the
/// QCore multipole Ewald assembly. `E = ½ Σ_{A,B} d_A · F^(11) · d_B`, `F^(11) = −∇⊗∇γ`:
/// real-space screened `erfc/r` (skip the `A=B` home image) + reciprocal (`G≠0`; tinfoil ⇒ no
/// surface/G=0 term) + the smooth self correction ([`ewald_self_tensor_1r`]). α-independent by
/// construction (the defining correctness property).
pub fn periodic_dipole_dipole_energy(system: &PeriodicSystem, alpha: f64, dipoles: &[Vec3]) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_dipole_energy requires a periodic system");
    let volume = lattice.volume();
    let tau = crate::pbc::ewald::TAU;
    let real_cut = tau / alpha;
    let g_cut = 2.0 * alpha * tau;
    let images = lattice.image_offsets(real_cut);
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * std::f64::consts::PI / volume;
    let gws: Vec<(Vec3, f64)> = lattice
        .reciprocal_vectors_within(g_cut, false)
        .into_iter()
        .map(|(_, g)| {
            let g2 = g.norm2();
            (g, four_pi_v * (-g2 * inv_4a2).exp() / g2)
        })
        .collect();
    let k_self = negated(&ewald_self_tensor_1r(alpha, 2)); // f^(11) form of the smooth self
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let k_real = negated(&ewald_real_tensor(r, alpha, 2));
                energy += 0.5 * contract_rank2(&k_real, dipoles[a], dipoles[b]);
            }
            let k_recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            energy += 0.5 * contract_rank2(&k_recip, dipoles[a], dipoles[b]);
        }
        // Remove the spurious smooth self-interaction the reciprocal sum includes at r=0.
        energy -= 0.5 * contract_rank2(&k_self, dipoles[a], dipoles[a]);
    }
    energy
}

/// Radial s-derivatives `g[p] = (d/d(r²))^p [erfc(αr)/r]`, `p = 0..=nmax`, of the Ewald
/// real-space screened `1/R` kernel.
///
/// Built as the bare `1/R` derivatives minus the smooth `erf(αr)/r` part. With
/// `erf(αr)/r = (2α/√π) F₀(α²r²)` and the Boys recurrence `Fₙ'(x) = −Fₙ₊₁(x)`,
/// `(d/ds)^p [erf(α√s)/√s] = (2α/√π)(−α²)^p F_p(α²s)`. The result feeds
/// [`crate::multipole::grad_tensor_unique`] just like the molecular `radial_derivs`.
///
/// Valid for `r > 0` (the on-site `r → 0` value is the *separate* self correction; the bare
/// `1/R` array diverges there but cancels against `erf/r` in the true `erfc/r`).
pub fn ewald_real_radial_derivs(r2: f64, alpha: f64, nmax: usize) -> Vec<f64> {
    let bare = crate::multipole::radial_derivs(r2, 0.0, nmax); // (d/ds)^p (1/r)
    let f = boys(nmax, alpha * alpha * r2);
    let pref = 2.0 * alpha / SQRT_PI;
    let mut g = vec![0.0_f64; nmax + 1];
    let mut neg_alpha2_pow = 1.0_f64; // (−α²)^p
    for p in 0..=nmax {
        g[p] = bare[p] - pref * neg_alpha2_pow * f[p];
        neg_alpha2_pow *= -(alpha * alpha);
    }
    g
}

/// Radial s-derivatives `g[p] = (d/d(r²))^p [q(r)/r³]`, `p = 0..=nmax`, of the Ewald
/// real-space screened `1/R³` kernel, where `q(r) = erfc(αr) + (2αr/√π) e^{−α²r²}` is the
/// QCore generalized-Ewald `R⁻³` numerator ([`crate::pbc::ewald::qcore_r3_real_value_derivatives`]).
///
/// As with the `1/R` case it is the bare `1/r³` minus a smooth screened complement
/// `C(r) = [erf(αr) − (2αr/√π)e^{−α²r²}]/r³ = (4α³/√π) F₁(α²r²)` (Boys recurrence
/// `F₀ − e^{−x} = 2x F₁`), whose `s`-derivatives are `(4α³/√π)(−α²)^p F_{p+1}(α²s)`. The
/// `−1/(2η²)` prefactor of the binomial split is applied at assembly, not here. Valid `r > 0`.
pub fn ewald_r3_radial_derivs(r2: f64, alpha: f64, nmax: usize) -> Vec<f64> {
    let s = r2;
    let mut bare = vec![0.0_f64; nmax + 1];
    bare[0] = s.powf(-1.5); // (d/ds)^0 s^{-3/2} = 1/r³
    for p in 1..=nmax {
        bare[p] = bare[p - 1] * (-((2 * p + 1) as f64) / 2.0) / s;
    }
    let f = boys(nmax + 1, alpha * alpha * r2);
    let pref = 4.0 * alpha * alpha * alpha / SQRT_PI;
    let mut g = vec![0.0_f64; nmax + 1];
    let mut neg_alpha2_pow = 1.0_f64; // (−α²)^p
    for p in 0..=nmax {
        g[p] = bare[p] - pref * neg_alpha2_pow * f[p + 1];
        neg_alpha2_pow *= -(alpha * alpha);
    }
    g
}

/// Unique symmetric-Cartesian components of `∇^k [erfc(αr)/r]` (rank `k`), in
/// [`crate::integrals::cartesian_rank_components`] order. This is the screened real-space
/// `1/R` contribution to the periodic multipole interaction tensor `f^(mn)` — the screened
/// analogue of the molecular `f_mn_unique`, built by feeding [`ewald_real_radial_derivs`]
/// into the same [`crate::multipole::grad_tensor_unique`] engine. Valid for `r > 0`.
pub fn ewald_real_tensor(x: Vec3, alpha: f64, k: usize) -> Vec<f64> {
    let g = ewald_real_radial_derivs(x.norm2(), alpha, k);
    crate::multipole::grad_tensor_unique(x, &g, k)
}

/// Bare `1/r³` radial s-derivatives `g[p] = (d/ds)^p s^{-3/2}`, the un-screened counterpart
/// used in the binomial-split residual.
fn bare_r3_radial_derivs(r2: f64, nmax: usize) -> Vec<f64> {
    let mut g = vec![0.0_f64; nmax + 1];
    g[0] = r2.powf(-1.5);
    for p in 1..=nmax {
        g[p] = g[p - 1] * (-((2 * p + 1) as f64) / 2.0) / r2;
    }
    g
}

/// **Screened real-space `f^(mn)` interaction tensor** between two atoms at relative position
/// `x = R_A − R_B` (one image), in QCore generalized-Ewald form: the binomial-split sum of
/// the short-range residual `[KO − 1/r + ½η⁻²/r³]`, the Ewald-screened `1/R` (`erfc/r`), and
/// the screened QCore `R⁻³` (`−½η⁻² q/r³`). `eta` is the pairwise KO hardness. Returns unique
/// rank-`(m+n)` components (molecular `f_mn_unique` convention).
///
/// As `α → 0` (screening off) every screened piece relaxes to its bare form and the sum
/// collapses to the bare KO kernel, so the tensor → the molecular bare `f^(mn)` — the
/// correctness gate for the binomial-split coefficients.
pub fn periodic_multipole_real_fmn(x: Vec3, alpha: f64, eta: f64, m: usize, n: usize) -> Vec<f64> {
    let k = m + n;
    let t = periodic_multipole_real_base_tensor(x, alpha, eta, k);
    let pref = multipole_rank_prefactor(m, n);
    t.iter().map(|v| v * pref).collect()
}

fn periodic_multipole_real_base_tensor(x: Vec3, alpha: f64, eta: f64, k: usize) -> Vec<f64> {
    let r2 = x.norm2();
    let inv_eta2 = 1.0 / (eta * eta);
    let ko = crate::multipole::radial_derivs(r2, inv_eta2, k);
    let bare1 = crate::multipole::radial_derivs(r2, 0.0, k);
    let bare3 = bare_r3_radial_derivs(r2, k);
    let screened1 = ewald_real_radial_derivs(r2, alpha, k);
    let screened3 = ewald_r3_radial_derivs(r2, alpha, k);
    let mut g = vec![0.0_f64; k + 1];
    for p in 0..=k {
        // residual + screened 1/R − ½η⁻²·(screened q/r³)
        g[p] = ko[p] - bare1[p] + 0.5 * inv_eta2 * bare3[p] + screened1[p]
            - 0.5 * inv_eta2 * screened3[p];
    }
    crate::multipole::grad_tensor_unique(x, &g, k)
}

/// Position **gradient** of [`periodic_multipole_real_fmn`]: the rank-`(m+n+1)` Cartesian tensor
/// `∇_x f^(mn)(x)`. Since `f^(mn) = pref(m,n)·∇^{m+n}φ` for the Ewald-split KO potential `φ`, its
/// gradient is `pref(m,n)·∇^{m+n+1}φ` — the **same** radial array carried one derivative order
/// higher (`k = m+n+1`), with the same `pref(m,n)`. Contracted with the two atomic moments (ranks
/// `m`, `n`) via [`crate::multipole::kernel_grad_unique`] it yields the real-space inter-atomic
/// force. Unique components in `cartesian_rank_components(m+n+1)` order.
pub fn periodic_multipole_real_grad_fmn(
    x: Vec3,
    alpha: f64,
    eta: f64,
    m: usize,
    n: usize,
) -> Vec<f64> {
    let k = m + n + 1;
    let t = periodic_multipole_real_grad_base_tensor(x, alpha, eta, k);
    let pref = multipole_rank_prefactor(m, n);
    t.iter().map(|v| v * pref).collect()
}

fn periodic_multipole_real_grad_base_tensor(x: Vec3, alpha: f64, eta: f64, k: usize) -> Vec<f64> {
    let r2 = x.norm2();
    let inv_eta2 = 1.0 / (eta * eta);
    let ko = crate::multipole::radial_derivs(r2, inv_eta2, k);
    let bare1 = crate::multipole::radial_derivs(r2, 0.0, k);
    let bare3 = bare_r3_radial_derivs(r2, k);
    let screened1 = ewald_real_radial_derivs(r2, alpha, k);
    let screened3 = ewald_r3_radial_derivs(r2, alpha, k);
    let mut g = vec![0.0_f64; k + 1];
    for p in 0..=k {
        g[p] = ko[p] - bare1[p] + 0.5 * inv_eta2 * bare3[p] + screened1[p]
            - 0.5 * inv_eta2 * screened3[p];
    }
    crate::multipole::grad_tensor_unique(x, &g, k)
}

#[inline]
fn multipole_rank_prefactor(m: usize, n: usize) -> f64 {
    let fact = |k: usize| (1..=k).product::<usize>().max(1) as f64;
    (if n % 2 == 0 { 1.0 } else { -1.0 }) / (fact(m) * fact(n))
}

/// Reciprocal-space contribution to the rank-`k` periodic multipole interaction tensor at
/// displacement `r = R_A − R_B`, in **QCore generalized-Ewald** form. The caller supplies the
/// reciprocal vectors and their scalar QCore weights `W(G)` — `(4π/V) e^{−G²/4α²}/G²` for the
/// `1/R` sector, `(2π/V) E₁(G²/4α²) η⁻²` for the QCore `R⁻³` sector ([`crate::pbc::ewald`]).
/// The contribution is `∇^k` of the reciprocal potential `Σ_G W(G) cos(G·r)`, i.e.
/// `Σ_G W(G) Re[(iG)^{⊗k} e^{iG·r}]`, whose unique component `(lx,ly,lz)` is
/// `Σ_G W(G) G_x^{lx} G_y^{ly} G_z^{lz} cos(G·r + kπ/2)`. Returns unique components in
/// [`crate::integrals::cartesian_rank_components`] order.
pub fn ewald_recip_tensor(r: Vec3, gws: &[(Vec3, f64)], k: usize) -> Vec<f64> {
    let comps = crate::integrals::cartesian_rank_components(k);
    let phase_shift = (k as f64) * std::f64::consts::FRAC_PI_2;
    let mut out = vec![0.0_f64; comps.len()];
    for &(g, w) in gws {
        let ga = [g.x, g.y, g.z];
        let phase = (g.dot(r) + phase_shift).cos();
        for (ci, &(lx, ly, lz)) in comps.iter().enumerate() {
            let gp = ga[0].powi(lx as i32) * ga[1].powi(ly as i32) * ga[2].powi(lz as i32);
            out[ci] += w * gp * phase;
        }
    }
    out
}

/// Rank-`k` **self-interaction tensor** of the Ewald `1/R` sector: the `r → 0` limit of
/// `∇^k[erf(αr)/r]`, the smooth part the reciprocal sum spuriously includes for an atom with
/// itself. With `erf(αr)/r = (2α/√π) F₀(α²r²)` and `F_p(0) = 1/(2p+1)`, the origin radial
/// s-derivatives are `g[p] = (2α/√π)(−α²)^p/(2p+1)`, and the tensor is
/// `grad_tensor_unique(0, g, k)` — which keeps only the all-paired isotropic terms, so it is
/// **nonzero only for even `k`** (rank-diagonal for STF moments). At `k=0` it equals `2α/√π`
/// (the scalar `self_term` of [`crate::pbc::ewald`] is its negative). The QCore `R⁻³`-sector
/// self is the analogous `r3_self` term, handled separately.
pub fn ewald_self_tensor_1r(alpha: f64, k: usize) -> Vec<f64> {
    let pref = 2.0 * alpha / SQRT_PI;
    let mut g = vec![0.0_f64; k + 1];
    let mut neg_alpha2_pow = 1.0_f64;
    for p in 0..=k {
        g[p] = pref * neg_alpha2_pow / ((2 * p + 1) as f64);
        neg_alpha2_pow *= -(alpha * alpha);
    }
    crate::multipole::grad_tensor_unique(Vec3::zero(), &g, k)
}

/// Rank-`k` **self tensor for the full KO kernel**: the `r→0` smooth-self of `γ_KO`'s long-
/// range part `[erf(αr)/r] − ½η⁻²·C(r)`, `C(r) = (4α³/√π)F₁(α²r²)`. Origin radials:
/// `g[p] = (2α/√π)(−α²)^p/(2p+1) − ½η⁻²·(4α³/√π)(−α²)^p/(2p+3)`. Reduces to
/// [`ewald_self_tensor_1r`] as `η→∞`.
pub fn ewald_self_tensor_ko(alpha: f64, eta: f64, k: usize) -> Vec<f64> {
    let inv_eta2 = 1.0 / (eta * eta);
    let p1 = 2.0 * alpha / SQRT_PI;
    let p3 = 4.0 * alpha * alpha * alpha / SQRT_PI;
    let mut g = vec![0.0_f64; k + 1];
    let mut neg_alpha2_pow = 1.0_f64;
    for p in 0..=k {
        g[p] = neg_alpha2_pow
            * (p1 / ((2 * p + 1) as f64) - 0.5 * inv_eta2 * p3 / ((2 * p + 3) as f64));
        neg_alpha2_pow *= -(alpha * alpha);
    }
    crate::multipole::grad_tensor_unique(Vec3::zero(), &g, k)
}

/// QCore reciprocal weights for the **full KO** kernel: the `1/R` term `(4π/V)e^{−G²/4α²}/G²`
/// plus the QCore `R⁻³` term `−½η⁻²·(2π/V)·E₁(G²/4α²)` (the `exp1` integral).
fn ko_recip_weights(
    lattice: &crate::lattice::Lattice,
    alpha: f64,
    inv_eta2: f64,
) -> Vec<(Vec3, f64)> {
    let volume = lattice.volume();
    let g_cut = 2.0 * alpha * crate::pbc::ewald::TAU;
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let four_pi_v = 4.0 * std::f64::consts::PI / volume;
    let two_pi_v = 2.0 * std::f64::consts::PI / volume;
    lattice
        .reciprocal_vectors_within(g_cut, false)
        .into_iter()
        .map(|(_, g)| {
            let g2 = g.norm2();
            let w_1r = four_pi_v * (-g2 * inv_4a2).exp() / g2;
            let w_r3 = -0.5 * inv_eta2 * two_pi_v * crate::pbc::ewald::exp1(g2 * inv_4a2);
            (g, w_1r + w_r3)
        })
        .collect()
}

/// Periodic **dipole–dipole** energy for the **full GFN1 Klopman–Ohno** kernel (single hardness
/// `eta`), the η-dependent generalization of [`periodic_dipole_dipole_energy`]: real-space full-KO
/// `f^(11)` (screened, [`periodic_multipole_real_fmn`]) summed over images + reciprocal (`1/R`
/// `e^{−G²/4α²}/G²` **and** QCore `R⁻³` `E₁`) + the full-KO smooth self. α-independent — the
/// correctness gate for the QCore `R⁻³` multipole sector. Reduces to the pure-`1/R` form as `η→∞`.
pub fn periodic_dipole_dipole_energy_ko(
    system: &PeriodicSystem,
    alpha: f64,
    dipoles: &[Vec3],
    eta: f64,
) -> f64 {
    let lattice = system
        .lattice
        .as_ref()
        .expect("periodic_dipole_dipole_energy_ko requires a periodic system");
    let inv_eta2 = 1.0 / (eta * eta);
    let real_cut = crate::pbc::ewald::TAU / alpha;
    let images = lattice.image_offsets(real_cut);
    let gws = ko_recip_weights(lattice, alpha, inv_eta2);
    let k_self = negated(&ewald_self_tensor_ko(alpha, eta, 2));
    let nat = system.atoms.len();
    let mut energy = 0.0;
    for a in 0..nat {
        for b in 0..nat {
            let rab = system.atoms[a].position - system.atoms[b].position;
            for off in &images {
                if a == b && off.is_origin() {
                    continue;
                }
                let r = rab - lattice.translation(*off);
                let k_real = periodic_multipole_real_fmn(r, alpha, eta, 1, 1);
                energy += 0.5 * contract_rank2(&k_real, dipoles[a], dipoles[b]);
            }
            let k_recip = negated(&ewald_recip_tensor(rab, &gws, 2));
            energy += 0.5 * contract_rank2(&k_recip, dipoles[a], dipoles[b]);
        }
        energy -= 0.5 * contract_rank2(&k_self, dipoles[a], dipoles[a]);
    }
    energy
}

/// Reciprocal potential `Σ_G W(G) cos(G·r)` (the rank-0 case of [`ewald_recip_tensor`]).
pub fn ewald_recip_potential(r: Vec3, gws: &[(Vec3, f64)]) -> f64 {
    gws.iter().map(|&(g, w)| w * g.dot(r).cos()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::cartesian_rank_components;
    use crate::math::erfc;

    /// `g[0]` is exactly `erfc(αr)/r`, and each higher order is the `s = r²` derivative of the
    /// previous one (central FD), which validates the whole Boys-built array against itself
    /// and the closed-form anchor.
    #[test]
    fn ewald_real_radial_derivs_match_s_derivative_fd() {
        let alpha = 0.42_f64;
        for &r in &[0.7_f64, 1.5, 3.0, 5.5] {
            let r2 = r * r;
            let nmax = 5;
            let g = ewald_real_radial_derivs(r2, alpha, nmax);
            // Anchor: g[0] = erfc(αr)/r.
            assert!(
                (g[0] - erfc(alpha * r) / r).abs() < 1.0e-12,
                "g[0] at r={r}: {} vs {}",
                g[0],
                erfc(alpha * r) / r
            );
            // g[p] = d/ds g[p-1].
            let h = 1.0e-6 * r2;
            for p in 1..=nmax {
                let plus = ewald_real_radial_derivs(r2 + h, alpha, nmax)[p - 1];
                let minus = ewald_real_radial_derivs(r2 - h, alpha, nmax)[p - 1];
                let fd = (plus - minus) / (2.0 * h);
                assert!(
                    (g[p] - fd).abs() < 1.0e-6 * (1.0 + g[p].abs()),
                    "g[{p}] at r={r}: analytic {} vs FD {fd}",
                    g[p]
                );
            }
        }
    }

    fn scalar(x: Vec3, alpha: f64) -> f64 {
        let r = x.norm();
        erfc(alpha * r) / r
    }

    fn shift(mut x: Vec3, axis: usize, d: f64) -> Vec3 {
        match axis {
            0 => x.x += d,
            1 => x.y += d,
            _ => x.z += d,
        }
        x
    }

    fn unique_index(k: usize, a: usize, b: usize) -> usize {
        let mut m = [0usize; 3];
        m[a] += 1;
        m[b] += 1;
        let _ = k;
        cartesian_rank_components(2)
            .iter()
            .position(|&(lx, ly, lz)| [lx, ly, lz] == m)
            .unwrap()
    }

    // Two-link FD chain validating the radial→Cartesian integration (`grad_tensor_unique`
    // fed the screened radials): rank-1 (gradient) vs FD of the scalar `erfc(αr)/r`, then
    // rank-2 vs FD of the analytic rank-1 tensor.
    #[test]
    fn ewald_real_tensor_matches_finite_difference() {
        let alpha = 0.42_f64;
        let x = Vec3::new(0.9, -0.6, 0.4);
        let h = 1.0e-6;
        // Link 1: rank-1 vs FD of scalar.
        let g1 = ewald_real_tensor(x, alpha, 1);
        for b in 0..3 {
            let fd = (scalar(shift(x, b, h), alpha) - scalar(shift(x, b, -h), alpha)) / (2.0 * h);
            assert!((g1[b] - fd).abs() < 1.0e-6, "rank1[{b}]: {} vs {fd}", g1[b]);
        }
        // Link 2: rank-2 component (a,b) vs FD of analytic rank-1 component b along axis a.
        let g2 = ewald_real_tensor(x, alpha, 2);
        for a in 0..3 {
            for b in 0..3 {
                let fd = (ewald_real_tensor(shift(x, a, h), alpha, 1)[b]
                    - ewald_real_tensor(shift(x, a, -h), alpha, 1)[b])
                    / (2.0 * h);
                let idx = unique_index(2, a, b);
                assert!(
                    (g2[idx] - fd).abs() < 1.0e-6 * (1.0 + g2[idx].abs()),
                    "rank2 ({a},{b}) idx {idx}: {} vs {fd}",
                    g2[idx]
                );
            }
        }
    }

    /// `g[0]` is exactly `q(r)/r³` (the QCore R⁻³ numerator over r³), and each higher order
    /// is the `s`-derivative of the previous — validates the Boys-`F_{p+1}` array.
    #[test]
    fn ewald_r3_radial_derivs_match_s_derivative_fd() {
        let alpha = 0.5_f64;
        for &r in &[0.8_f64, 1.7, 3.5] {
            let r2 = r * r;
            let nmax = 5;
            let g = ewald_r3_radial_derivs(r2, alpha, nmax);
            let ar = alpha * r;
            let q = erfc(ar) + (2.0 * ar / SQRT_PI) * (-(ar * ar)).exp();
            assert!(
                (g[0] - q / (r * r * r)).abs() < 1.0e-12,
                "g[0] at r={r}: {} vs {}",
                g[0],
                q / (r * r * r)
            );
            let h = 1.0e-6 * r2;
            for p in 1..=nmax {
                let plus = ewald_r3_radial_derivs(r2 + h, alpha, nmax)[p - 1];
                let minus = ewald_r3_radial_derivs(r2 - h, alpha, nmax)[p - 1];
                let fd = (plus - minus) / (2.0 * h);
                assert!(
                    (g[p] - fd).abs() < 1.0e-6 * (1.0 + g[p].abs()),
                    "g[{p}] at r={r}: {} vs FD {fd}",
                    g[p]
                );
            }
        }
    }

    // The QCore reciprocal multipole tensor (∇^k of the reciprocal potential, with the
    // (iG)^k structure) validated by the two-link FD chain against the scalar potential.
    #[test]
    fn ewald_recip_tensor_matches_finite_difference() {
        // A few reciprocal vectors with QCore-1/R-like weights exp(-G²/4α²)/G².
        let alpha = 0.6_f64;
        let gws: Vec<(Vec3, f64)> = [
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(0.0, 1.1, 0.0),
            Vec3::new(0.5, -0.4, 0.9),
            Vec3::new(-0.8, 0.3, 0.2),
        ]
        .iter()
        .map(|&g| {
            let g2 = g.norm2();
            (g, (-g2 / (4.0 * alpha * alpha)).exp() / g2)
        })
        .collect();
        let r = Vec3::new(0.6, -0.5, 0.3);
        let h = 1.0e-6;
        let shift = |x: Vec3, axis: usize, d: f64| {
            let mut v = x;
            match axis {
                0 => v.x += d,
                1 => v.y += d,
                _ => v.z += d,
            }
            v
        };
        // Link 1: rank-1 vs FD of the scalar potential.
        let g1 = ewald_recip_tensor(r, &gws, 1);
        for b in 0..3 {
            let fd = (ewald_recip_potential(shift(r, b, h), &gws)
                - ewald_recip_potential(shift(r, b, -h), &gws))
                / (2.0 * h);
            assert!(
                (g1[b] - fd).abs() < 1.0e-6,
                "recip rank1[{b}]: {} vs {fd}",
                g1[b]
            );
        }
        // Link 2: rank-2 vs FD of rank-1.
        let g2 = ewald_recip_tensor(r, &gws, 2);
        for a in 0..3 {
            for b in 0..3 {
                let fd = (ewald_recip_tensor(shift(r, a, h), &gws, 1)[b]
                    - ewald_recip_tensor(shift(r, a, -h), &gws, 1)[b])
                    / (2.0 * h);
                let idx = unique_index(2, a, b);
                assert!(
                    (g2[idx] - fd).abs() < 1.0e-6 * (1.0 + g2[idx].abs()),
                    "recip rank2 ({a},{b}): {} vs {fd}",
                    g2[idx]
                );
            }
        }
    }

    // The QCore binomial-split screened real-space f^(mn) must recover the molecular bare KO
    // f^(mn) as α → 0 (screening off). A wrong split coefficient would leave an O(1) residual
    // that does NOT vanish, so the small-α agreement is a sharp correctness gate.
    #[test]
    fn periodic_multipole_real_fmn_recovers_bare_ko_at_small_alpha() {
        let eta = 0.5_f64;
        let inv_eta2 = 1.0 / (eta * eta);
        let x = Vec3::new(0.9, -0.6, 0.4);
        let alpha = 1.0e-6;
        for &(m, n) in &[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (2, 1),
            (1, 2),
            (2, 2),
        ] {
            let screened = periodic_multipole_real_fmn(x, alpha, eta, m, n);
            let bare = crate::multipole::f_mn_unique(x, inv_eta2, m, n);
            assert_eq!(screened.len(), bare.len());
            for (a, b) in screened.iter().zip(bare.iter()) {
                assert!(
                    (a - b).abs() < 1.0e-4,
                    "f^({m}{n}) component: screened {a} vs bare {b}"
                );
            }
        }
    }

    // The defining QCore-Ewald correctness property: the periodic dipole–dipole energy must
    // be independent of the splitting parameter α (real + reciprocal + self all combine).
    #[test]
    fn periodic_dipole_dipole_energy_is_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        // General case: the dipoles do not sum to zero.
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let e1 = periodic_dipole_dipole_energy(&system, 0.20, &dipoles);
        let e2 = periodic_dipole_dipole_energy(&system, 0.35, &dipoles);
        let e3 = periodic_dipole_dipole_energy(&system, 0.50, &dipoles);
        assert!(
            (e1 - e2).abs() < 1.0e-6 && (e1 - e3).abs() < 1.0e-6,
            "dipole-dipole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // A4 keystone: the periodic dipole–dipole kernel FORCE at fixed dipoles equals −dE/dR of the
    // energy (the rank-3 gradient tensor `∇f^(11)` + the rank-3 reciprocal tensor, contracted with
    // the two dipoles). FD vs the analytic force, per atom per axis.
    #[test]
    fn periodic_dipole_dipole_forces_match_energy_fd() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let forces = periodic_dipole_dipole_forces_ko_pairwise(&system, alpha, &dipoles, &hard);
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut sp = system.clone();
                let mut sm = system.clone();
                match c {
                    0 => {
                        sp.atoms[a].position.x += h;
                        sm.atoms[a].position.x -= h;
                    }
                    1 => {
                        sp.atoms[a].position.y += h;
                        sm.atoms[a].position.y -= h;
                    }
                    _ => {
                        sp.atoms[a].position.z += h;
                        sm.atoms[a].position.z -= h;
                    }
                }
                let ep = periodic_dipole_dipole_energy_ko_pairwise(&sp, alpha, &dipoles, &hard);
                let em = periodic_dipole_dipole_energy_ko_pairwise(&sm, alpha, &dipoles, &hard);
                let fd = -(ep - em) / (2.0 * h); // force = −∂E/∂R
                let an = match c {
                    0 => forces[a].x,
                    1 => forces[a].y,
                    _ => forces[a].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-6 * (1.0 + an.abs()),
                    "force[{a}][{c}] analytic {an:.10} vs FD {fd:.10}"
                );
            }
        }
    }

    // A4: the complete dipole-rank kernel force (dipole–dipole + charge–dipole cross) equals
    // −d(E_dd + E_cd)/dR at fixed moments — validating the cross force and the SCF-facing bundle.
    #[test]
    fn periodic_dipole_rank_forces_match_energy_fd() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let charges = [0.35_f64, -0.35];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let forces = periodic_dipole_rank_forces(&system, alpha, &charges, &dipoles, &hard);
        let energy = |s: &PeriodicSystem| {
            periodic_dipole_dipole_energy_ko_pairwise(s, alpha, &dipoles, &hard)
                + periodic_charge_dipole_energy_pairwise(s, alpha, &charges, &dipoles, &hard)
        };
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut sp = system.clone();
                let mut sm = system.clone();
                match c {
                    0 => {
                        sp.atoms[a].position.x += h;
                        sm.atoms[a].position.x -= h;
                    }
                    1 => {
                        sp.atoms[a].position.y += h;
                        sm.atoms[a].position.y -= h;
                    }
                    _ => {
                        sp.atoms[a].position.z += h;
                        sm.atoms[a].position.z -= h;
                    }
                }
                let fd = -(energy(&sp) - energy(&sm)) / (2.0 * h);
                let an = match c {
                    0 => forces[a].x,
                    1 => forces[a].y,
                    _ => forces[a].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-6 * (1.0 + an.abs()),
                    "dipole-rank force[{a}][{c}] analytic {an:.10} vs FD {fd:.10}"
                );
            }
        }
    }

    // The monopole (charge-charge) energy must be α-independent even for a NON-neutral cell —
    // the decisive test of the charge-sector G=0 corrections (Coulomb background + KO-R⁻³ r3_k0).
    #[test]
    fn periodic_monopole_energy_alpha_independent_charged() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let charges = [0.4_f64, 0.1]; // deliberately net-charged (Σ = 0.5 ≠ 0)
        let eta = 0.45;
        let e1 = periodic_monopole_energy(&system, 0.20, &charges, eta);
        let e2 = periodic_monopole_energy(&system, 0.35, &charges, eta);
        let e3 = periodic_monopole_energy(&system, 0.50, &charges, eta);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "charged monopole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // The per-pair-η charge-dipole energy must be α-independent for distinct hardnesses, and
    // reduce to the single-η charge-dipole energy when hardnesses are equal.
    #[test]
    fn periodic_charge_dipole_energy_pairwise_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let charges = [0.3_f64, -0.3]; // neutral
        let dipoles = [Vec3::new(0.10, -0.04, 0.02), Vec3::new(-0.03, 0.07, 0.05)];
        let hard = [0.40_f64, 0.55];
        let e1 = periodic_charge_dipole_energy_pairwise(&system, 0.20, &charges, &dipoles, &hard);
        let e2 = periodic_charge_dipole_energy_pairwise(&system, 0.35, &charges, &dipoles, &hard);
        let e3 = periodic_charge_dipole_energy_pairwise(&system, 0.50, &charges, &dipoles, &hard);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "pairwise charge-dipole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
        let eta = 0.45;
        let uniform =
            periodic_charge_dipole_energy_pairwise(&system, 0.35, &charges, &dipoles, &[eta, eta]);
        let single = periodic_charge_dipole_energy(&system, 0.35, &charges, &dipoles, eta);
        assert!(
            (uniform - single).abs() < 1.0e-9 * (1.0 + single.abs()),
            "uniform {uniform:.10} != single-η {single:.10}"
        );
    }

    // The charge-quadrupole (mixed rank-0/2) periodic energy must be α-independent for a
    // charge-neutral cell — validating the f^(02) tensor (monopole↔quadrupole) and its self.
    #[test]
    fn periodic_charge_quad_energy_is_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let charges = [0.3_f64, -0.3]; // neutral
                                       // Traceless symmetric quadrupoles.
        let quads = [
            [[0.10, 0.03, 0.0], [0.03, -0.04, 0.02], [0.0, 0.02, -0.06]],
            [[-0.05, 0.0, 0.01], [0.0, 0.08, -0.02], [0.01, -0.02, -0.03]],
        ];
        let eta = 0.45;
        let e1 = periodic_charge_quad_energy(&system, 0.20, &charges, &quads, eta);
        let e2 = periodic_charge_quad_energy(&system, 0.35, &charges, &quads, eta);
        let e3 = periodic_charge_quad_energy(&system, 0.50, &charges, &quads, eta);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "charge-quad α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // The charge-dipole (mixed rank-0/1) periodic energy must be α-independent for a
    // charge-neutral cell — validating the f^(01) tensor (the monopole↔dipole coupling).
    #[test]
    fn periodic_charge_dipole_energy_is_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let charges = [0.3_f64, -0.3]; // neutral
        let dipoles = [Vec3::new(0.10, -0.04, 0.02), Vec3::new(-0.03, 0.07, 0.05)];
        let eta = 0.45;
        let e1 = periodic_charge_dipole_energy(&system, 0.20, &charges, &dipoles, eta);
        let e2 = periodic_charge_dipole_energy(&system, 0.35, &charges, &dipoles, eta);
        let e3 = periodic_charge_dipole_energy(&system, 0.50, &charges, &dipoles, eta);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "charge-dipole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // The per-pair-η dipole-dipole energy must be α-independent for DISTINCT hardnesses (the
    // heteroatomic case), and reduce to the single-η full-KO energy when hardnesses are equal.
    #[test]
    fn periodic_dipole_dipole_energy_ko_pairwise_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let hardnesses = [0.40_f64, 0.55]; // distinct (Na vs Cl-like)
        let e1 = periodic_dipole_dipole_energy_ko_pairwise(&system, 0.20, &dipoles, &hardnesses);
        let e2 = periodic_dipole_dipole_energy_ko_pairwise(&system, 0.35, &dipoles, &hardnesses);
        let e3 = periodic_dipole_dipole_energy_ko_pairwise(&system, 0.50, &dipoles, &hardnesses);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "pairwise-η dipole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
        // Equal hardnesses ⇒ the single-η full-KO energy.
        let eta = 0.45;
        let uniform =
            periodic_dipole_dipole_energy_ko_pairwise(&system, 0.35, &dipoles, &[eta, eta]);
        let single = periodic_dipole_dipole_energy_ko(&system, 0.35, &dipoles, eta);
        assert!(
            (uniform - single).abs() < 1.0e-9 * (1.0 + single.abs()),
            "uniform pairwise {uniform:.10} != single-η {single:.10}"
        );
    }

    // The full-KO (η-dependent) dipole-dipole energy must be α-independent — validating the
    // QCore R⁻³ multipole sector (reciprocal E₁ + R⁻³ self) — and must reduce to the pure-1/R
    // form as η→∞ (the R⁻³ binomial term ∝ η⁻² vanishes).
    #[test]
    fn periodic_dipole_dipole_energy_ko_is_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let eta = 0.45;
        let e1 = periodic_dipole_dipole_energy_ko(&system, 0.20, &dipoles, eta);
        let e2 = periodic_dipole_dipole_energy_ko(&system, 0.35, &dipoles, eta);
        let e3 = periodic_dipole_dipole_energy_ko(&system, 0.50, &dipoles, eta);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "full-KO dipole-dipole α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
        // η → ∞ ⇒ R⁻³ sector off ⇒ pure-1/R form.
        let e_large_eta = periodic_dipole_dipole_energy_ko(&system, 0.35, &dipoles, 1.0e6);
        let e_pure = periodic_dipole_dipole_energy(&system, 0.35, &dipoles);
        assert!(
            (e_large_eta - e_pure).abs() < 1.0e-6 * (1.0 + e_pure.abs()),
            "η→∞ KO {e_large_eta:.10} should match pure-1/R {e_pure:.10}"
        );
    }

    // The arbitrary-rank periodic multipole-tensor energy must be α-independent at every even
    // rank K (real + reciprocal + the rank-K self correction combine) — extending the
    // dipole-dipole keystone to the quadrupole rank (K=4).
    #[test]
    fn periodic_multipole_w_energy_alpha_independent_higher_rank() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let w = Vec3::new(0.7, -0.4, 0.5);
        for &k in &[2_usize, 4] {
            let e1 = periodic_multipole_w_energy(&system, 0.20, k, w);
            let e2 = periodic_multipole_w_energy(&system, 0.35, k, w);
            let e3 = periodic_multipole_w_energy(&system, 0.50, k, w);
            assert!(
                (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                    && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
                "rank-{k} α dependence: {e1:.10} {e2:.10} {e3:.10}"
            );
        }
    }

    // Euler relation for the quadratic (degree-2 homogeneous) periodic dipole energy:
    // Σ_A d_A·V_A = 2E. A consistency cross-check between the validated per-pair-η energy and
    // field with no finite difference.
    #[test]
    fn pairwise_dipole_field_euler_relation() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let e = periodic_dipole_dipole_energy_ko_pairwise(&system, alpha, &dipoles, &hard);
        let field = periodic_dipole_field_ko_pairwise(&system, alpha, &dipoles, &hard);
        let sum: f64 = dipoles
            .iter()
            .zip(field.iter())
            .map(|(d, f)| d.dot(*f))
            .sum();
        assert!(
            (sum - 2.0 * e).abs() < 1.0e-9 * (1.0 + e.abs()),
            "Euler: Σ d·V = {sum:.10} should be 2E = {:.10}",
            2.0 * e
        );
    }

    // The periodic charge–dipole cross fields are the exact potentials ∂E/∂q and ∂E/∂d: the
    // bilinear Euler identities Σ q·V_q = Σ d·V_d = E (no FD), plus a finite-difference check of
    // BOTH the dipole field V_d (∂E/∂d) and the charge potential V_q (∂E/∂q). These are the two
    // SCF Fock routes the joint mixer couples for the monopole↔dipole cross term.
    #[test]
    fn pairwise_charge_dipole_fields_match_energy() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let charges = [0.3_f64, -0.3];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let e = periodic_charge_dipole_energy_pairwise(&system, alpha, &charges, &dipoles, &hard);
        let (vq, vd) =
            periodic_charge_dipole_fields_pairwise(&system, alpha, &charges, &dipoles, &hard);
        // Euler (degree 1 in q and in d separately ⇒ each sum equals E, not 2E).
        let sq: f64 = charges.iter().zip(&vq).map(|(q, v)| q * v).sum();
        let sd: f64 = dipoles.iter().zip(&vd).map(|(d, v)| d.dot(*v)).sum();
        assert!(
            (sq - e).abs() < 1.0e-9 * (1.0 + e.abs()),
            "Euler: Σ q·V_q = {sq:.10} should be E = {e:.10}"
        );
        assert!(
            (sd - e).abs() < 1.0e-9 * (1.0 + e.abs()),
            "Euler: Σ d·V_d = {sd:.10} should be E = {e:.10}"
        );
        // FD of the dipole field V_d and the charge potential V_q.
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut dp = dipoles;
                let mut dm = dipoles;
                match c {
                    0 => {
                        dp[a].x += h;
                        dm[a].x -= h;
                    }
                    1 => {
                        dp[a].y += h;
                        dm[a].y -= h;
                    }
                    _ => {
                        dp[a].z += h;
                        dm[a].z -= h;
                    }
                }
                let ep =
                    periodic_charge_dipole_energy_pairwise(&system, alpha, &charges, &dp, &hard);
                let em =
                    periodic_charge_dipole_energy_pairwise(&system, alpha, &charges, &dm, &hard);
                let fd = (ep - em) / (2.0 * h);
                let an = match c {
                    0 => vd[a].x,
                    1 => vd[a].y,
                    _ => vd[a].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-6 * (1.0 + an.abs()),
                    "V_d[{a}][{c}] analytic {an:.10} vs FD {fd:.10}"
                );
            }
            let mut qp = charges;
            let mut qm = charges;
            qp[a] += h;
            qm[a] -= h;
            let ep = periodic_charge_dipole_energy_pairwise(&system, alpha, &qp, &dipoles, &hard);
            let em = periodic_charge_dipole_energy_pairwise(&system, alpha, &qm, &dipoles, &hard);
            let fd = (ep - em) / (2.0 * h);
            assert!(
                (vq[a] - fd).abs() < 1.0e-6 * (1.0 + vq[a].abs()),
                "V_q[{a}] analytic {:.10} vs FD {fd:.10}",
                vq[a]
            );
        }
    }

    // The generic arbitrary-rank field builder reduces EXACTLY to the validated dipole-rank bundle
    // at max_rank=1: V[A][0] (charge potential from dipoles) == V_q, and V[A][1] (dipole field)
    // == V_d. This pins the generic machinery against the FD-gated dipole path before trusting it
    // at quadrupole rank.
    #[test]
    fn periodic_multipole_fields_generic_reduces_to_dipole_rank() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let charges = [0.35_f64, -0.35];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let (v_q, v_d) = periodic_dipole_rank_fields(&system, alpha, &charges, &dipoles, &hard);
        // moments[A] = [[q_A], [dx,dy,dz]].
        let moments: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|a| {
                vec![
                    vec![charges[a]],
                    vec![dipoles[a].x, dipoles[a].y, dipoles[a].z],
                ]
            })
            .collect();
        let field = periodic_multipole_fields_generic(&system, alpha, &moments, &hard, 1);
        for a in 0..2 {
            assert!(
                (field[a][0][0] - v_q[a]).abs() < 1.0e-10 * (1.0 + v_q[a].abs()),
                "V[{a}][0] generic {:.10} vs V_q {:.10}",
                field[a][0][0],
                v_q[a]
            );
            let vd = [v_d[a].x, v_d[a].y, v_d[a].z];
            for c in 0..3 {
                assert!(
                    (field[a][1][c] - vd[c]).abs() < 1.0e-10 * (1.0 + vd[c].abs()),
                    "V[{a}][1][{c}] generic {:.10} vs V_d {:.10}",
                    field[a][1][c],
                    vd[c]
                );
            }
        }
    }

    // Quadrupole correctness: the generic energy E = ½ Σ_A Σ_l M·V at rank 2 (dipoles + quadrupoles)
    // is α-independent — the defining Ewald property, now exercised at QUADRUPOLE rank through the
    // generic field builder (real images + reciprocal + rank-diagonal self for l=1 AND l=2).
    #[test]
    fn periodic_multipole_fields_generic_rank2_alpha_independent() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let hard = [0.40_f64, 0.55];
        // Detraced (traceless) quadrupoles in full 3×3=9 layout, plus dipoles; charges 0 (the
        // dipole/quad sectors are absolutely-convergent ⇒ α-independent without neutralization).
        let q0 = [
            [0.20, 0.05, -0.10, 0.05, -0.12, 0.03, -0.10, 0.03, -0.08],
            [-0.15, 0.04, 0.02, 0.04, 0.09, -0.06, 0.02, -0.06, 0.06],
        ];
        let dip = [[0.12, -0.05, 0.03], [-0.04, 0.09, 0.07]];
        let moments: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|a| vec![vec![0.0], dip[a].to_vec(), q0[a].to_vec()])
            .collect();
        let energy = |alpha: f64| -> f64 {
            let field = periodic_multipole_fields_generic(&system, alpha, &moments, &hard, 2);
            let mut e = 0.0;
            for a in 0..2 {
                for l in 0..=2 {
                    e += 0.5
                        * moments[a][l]
                            .iter()
                            .zip(field[a][l].iter())
                            .map(|(m, v)| m * v)
                            .sum::<f64>();
                }
            }
            e
        };
        let e1 = energy(0.25);
        let e2 = energy(0.40);
        let e3 = energy(0.55);
        assert!(
            (e1 - e2).abs() < 1.0e-6 * (1.0 + e1.abs())
                && (e1 - e3).abs() < 1.0e-6 * (1.0 + e1.abs()),
            "rank-2 generic energy α dependence: {e1:.10} {e2:.10} {e3:.10}"
        );
    }

    // The arbitrary-rank kernel FORCE at QUADRUPOLE rank equals −dE/dR of the generic energy
    // ½ Σ M·V (charges + dipoles + quadrupoles fixed): FD per atom per axis. Confirms the
    // generic force (rank-(la+lb+1) gradient tensors, all rank pairs) for quadrupole.
    #[test]
    fn periodic_multipole_field_kernel_matches_direct_rank2() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let q0 = [
            [0.20, 0.05, -0.10, 0.05, -0.12, 0.03, -0.10, 0.03, -0.08],
            [-0.15, 0.04, 0.02, 0.04, 0.09, -0.06, 0.02, -0.06, 0.06],
        ];
        let dip = [[0.12, -0.05, 0.03], [-0.04, 0.09, 0.07]];
        let moments: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|a| vec![vec![0.25 - 0.5 * a as f64], dip[a].to_vec(), q0[a].to_vec()])
            .collect();
        let direct = periodic_multipole_fields_generic_direct(&system, alpha, &moments, &hard, 2);
        let cached = PeriodicMultipoleFieldKernel::build(&system, alpha, &hard, 2).apply(&moments);
        for a in 0..2 {
            for l in 0..=2 {
                for i in 0..direct[a][l].len() {
                    let d = direct[a][l][i];
                    let c = cached[a][l][i];
                    assert!(
                        (d - c).abs() < 1.0e-12 * (1.0 + d.abs()),
                        "field[{a}][{l}][{i}] direct {d:.12e} cached {c:.12e}"
                    );
                }
            }
        }
    }

    #[test]
    fn periodic_multipole_forces_generic_match_energy_fd() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let q = [0.30_f64, -0.30];
        let dip = [[0.12, -0.05, 0.03], [-0.04, 0.09, 0.07]];
        let quad = [
            [0.20, 0.05, -0.10, 0.05, -0.12, 0.03, -0.10, 0.03, -0.08],
            [-0.15, 0.04, 0.02, 0.04, 0.09, -0.06, 0.02, -0.06, 0.06],
        ];
        let moments: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|a| vec![vec![q[a]], dip[a].to_vec(), quad[a].to_vec()])
            .collect();
        let forces = periodic_multipole_forces_generic(&system, alpha, &moments, &hard, 2);
        let energy = |s: &PeriodicSystem| -> f64 {
            let field = periodic_multipole_fields_generic(s, alpha, &moments, &hard, 2);
            let mut e = 0.0;
            for a in 0..2 {
                for l in 0..=2 {
                    e += 0.5
                        * moments[a][l]
                            .iter()
                            .zip(field[a][l].iter())
                            .map(|(m, v)| m * v)
                            .sum::<f64>();
                }
            }
            e
        };
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut sp = system.clone();
                let mut sm = system.clone();
                match c {
                    0 => {
                        sp.atoms[a].position.x += h;
                        sm.atoms[a].position.x -= h;
                    }
                    1 => {
                        sp.atoms[a].position.y += h;
                        sm.atoms[a].position.y -= h;
                    }
                    _ => {
                        sp.atoms[a].position.z += h;
                        sm.atoms[a].position.z -= h;
                    }
                }
                let fd = -(energy(&sp) - energy(&sm)) / (2.0 * h);
                let an = match c {
                    0 => forces[a].x,
                    1 => forces[a].y,
                    _ => forces[a].z,
                };
                assert!(
                    (an - fd).abs() < 1.0e-6 * (1.0 + an.abs()),
                    "generic force[{a}][{c}] analytic {an:.10} vs FD {fd:.10}"
                );
            }
        }
    }

    // The combined dipole-rank SCF field bundle (dipole–dipole + charge–dipole cross) satisfies
    // the mixed Euler identity Σ d·V_d + Σ q·V_q = 2 E_mp, where E_mp = E_dd + E_cd. A no-FD
    // cross-check that the consolidated SCF-loop field call is consistent with both pair energies.
    #[test]
    fn periodic_dipole_rank_fields_mixed_euler_relation() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hard = [0.40_f64, 0.55];
        let charges = [0.35_f64, -0.35];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let e_dd = periodic_dipole_dipole_energy_ko_pairwise(&system, alpha, &dipoles, &hard);
        let e_cd =
            periodic_charge_dipole_energy_pairwise(&system, alpha, &charges, &dipoles, &hard);
        let e_mp = e_dd + e_cd;
        let (vq, vd) = periodic_dipole_rank_fields(&system, alpha, &charges, &dipoles, &hard);
        let sd: f64 = dipoles.iter().zip(&vd).map(|(d, v)| d.dot(*v)).sum();
        let sq: f64 = charges.iter().zip(&vq).map(|(q, v)| q * v).sum();
        assert!(
            (sd + sq - 2.0 * e_mp).abs() < 1.0e-9 * (1.0 + e_mp.abs()),
            "mixed Euler: Σd·V_d + Σq·V_q = {:.10} should be 2 E_mp = {:.10}",
            sd + sq,
            2.0 * e_mp
        );
    }

    // The per-pair-η dipole field must equal ∂E/∂d_A of the per-pair-η energy (the heteroatomic
    // SCF Fock shift).
    #[test]
    fn periodic_dipole_field_ko_pairwise_matches_energy_gradient() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let hardnesses = [0.40_f64, 0.55];
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let field = periodic_dipole_field_ko_pairwise(&system, alpha, &dipoles, &hardnesses);
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut dp = dipoles;
                let mut dm = dipoles;
                match c {
                    0 => {
                        dp[a].x += h;
                        dm[a].x -= h;
                    }
                    1 => {
                        dp[a].y += h;
                        dm[a].y -= h;
                    }
                    _ => {
                        dp[a].z += h;
                        dm[a].z -= h;
                    }
                }
                let fd =
                    (periodic_dipole_dipole_energy_ko_pairwise(&system, alpha, &dp, &hardnesses)
                        - periodic_dipole_dipole_energy_ko_pairwise(
                            &system,
                            alpha,
                            &dm,
                            &hardnesses,
                        ))
                        / (2.0 * h);
                let analytic = match c {
                    0 => field[a].x,
                    1 => field[a].y,
                    _ => field[a].z,
                };
                assert!(
                    (analytic - fd).abs() < 1.0e-6,
                    "pairwise field atom {a} comp {c}: {analytic} vs FD {fd}"
                );
            }
        }
    }

    // The full-KO dipole field must equal ∂E/∂d_A of the full-KO dipole energy (the actual
    // SCF Fock shift for the GFN1 KO kernel).
    #[test]
    fn periodic_dipole_field_ko_matches_energy_gradient() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let eta = 0.45;
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let field = periodic_dipole_field_ko(&system, alpha, &dipoles, eta);
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut dp = dipoles;
                let mut dm = dipoles;
                match c {
                    0 => {
                        dp[a].x += h;
                        dm[a].x -= h;
                    }
                    1 => {
                        dp[a].y += h;
                        dm[a].y -= h;
                    }
                    _ => {
                        dp[a].z += h;
                        dm[a].z -= h;
                    }
                }
                let fd = (periodic_dipole_dipole_energy_ko(&system, alpha, &dp, eta)
                    - periodic_dipole_dipole_energy_ko(&system, alpha, &dm, eta))
                    / (2.0 * h);
                let analytic = match c {
                    0 => field[a].x,
                    1 => field[a].y,
                    _ => field[a].z,
                };
                assert!(
                    (analytic - fd).abs() < 1.0e-6,
                    "KO dipole field atom {a} comp {c}: {analytic} vs FD {fd}"
                );
            }
        }
    }

    // The per-atom dipole field must equal the gradient ∂E/∂d_A of the dipole-dipole energy —
    // this is the SCF-facing quantity (the multipole Fock shift the joint mixer consumes).
    #[test]
    fn periodic_dipole_field_matches_energy_gradient() {
        let system = PeriodicSystem::from_xyz_str(
            "2\nLattice=\"5.6 0 0 0 5.6 0 0 0 5.6\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.8 2.8 2.8\n",
            0.0,
            false,
        )
        .unwrap();
        let alpha = 0.35;
        let dipoles = [Vec3::new(0.12, -0.05, 0.03), Vec3::new(-0.04, 0.09, 0.07)];
        let field = periodic_dipole_field(&system, alpha, &dipoles);
        let h = 1.0e-6;
        for a in 0..2 {
            for c in 0..3 {
                let mut dp = dipoles;
                let mut dm = dipoles;
                match c {
                    0 => {
                        dp[a].x += h;
                        dm[a].x -= h;
                    }
                    1 => {
                        dp[a].y += h;
                        dm[a].y -= h;
                    }
                    _ => {
                        dp[a].z += h;
                        dm[a].z -= h;
                    }
                }
                let fd = (periodic_dipole_dipole_energy(&system, alpha, &dp)
                    - periodic_dipole_dipole_energy(&system, alpha, &dm))
                    / (2.0 * h);
                let analytic = match c {
                    0 => field[a].x,
                    1 => field[a].y,
                    _ => field[a].z,
                };
                assert!(
                    (analytic - fd).abs() < 1.0e-6,
                    "dipole field atom {a} comp {c}: analytic {analytic} vs FD {fd}"
                );
            }
        }
    }

    // The full-KO self tensor reduces to the pure-1/R self tensor as η→∞ (the R⁻³ ∝ η⁻² part
    // vanishes) — a cross-check between the two self-correction functions.
    #[test]
    fn ewald_self_tensor_ko_reduces_to_1r_at_large_eta() {
        let alpha = 0.5_f64;
        for &k in &[0_usize, 2, 4] {
            let ko = ewald_self_tensor_ko(alpha, 1.0e6, k);
            let r1 = ewald_self_tensor_1r(alpha, k);
            assert_eq!(ko.len(), r1.len());
            for (a, b) in ko.iter().zip(r1.iter()) {
                assert!(
                    (a - b).abs() < 1.0e-6 * (1.0 + b.abs()),
                    "η→∞ KO self {a} vs 1/R self {b} (rank {k})"
                );
            }
        }
    }

    // The 1/R self tensor: rank-0 = 2α/√π (anchor); odd ranks vanish; rank-2 is isotropic
    // with diagonal −4α³/(3√π) and zero off-diagonal (rank-diagonal STF self).
    #[test]
    fn ewald_self_tensor_1r_values() {
        let alpha = 0.6_f64;
        assert!((ewald_self_tensor_1r(alpha, 0)[0] - 2.0 * alpha / SQRT_PI).abs() < 1.0e-14);
        for v in ewald_self_tensor_1r(alpha, 1) {
            assert!(v.abs() < 1.0e-14, "rank-1 self must vanish: {v}");
        }
        for v in ewald_self_tensor_1r(alpha, 3) {
            assert!(v.abs() < 1.0e-13, "rank-3 self must vanish: {v}");
        }
        let t2 = ewald_self_tensor_1r(alpha, 2);
        let diag = -4.0 * alpha * alpha * alpha / (3.0 * SQRT_PI);
        for (idx, &(lx, ly, lz)) in cartesian_rank_components(2).iter().enumerate() {
            let expect = if lx == 2 || ly == 2 || lz == 2 {
                diag
            } else {
                0.0
            };
            assert!(
                (t2[idx] - expect).abs() < 1.0e-14,
                "rank-2 self ({lx},{ly},{lz}): {} vs {expect}",
                t2[idx]
            );
        }
    }

    /// Large `αr` ⇒ the screened kernel and all its radial derivatives vanish (erfc → 0);
    /// small `αr` stays finite (the erf part cancels the bare-1/R divergence in `g[0]`).
    #[test]
    fn ewald_real_radial_derivs_screening_limits() {
        let alpha = 1.0_f64;
        let g_far = ewald_real_radial_derivs(100.0, alpha, 4);
        for &v in &g_far {
            assert!(
                v.abs() < 1.0e-6,
                "screened kernel should vanish far out: {v}"
            );
        }
        let g_near = ewald_real_radial_derivs(0.01, alpha, 4);
        assert!(g_near.iter().all(|v| v.is_finite()));
    }
}
