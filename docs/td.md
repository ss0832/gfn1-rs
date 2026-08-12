# TD-GFN1: TDA excited states and their nuclear gradients

Everything in this page is in **atomic units** — excitation energies and total
energies in Hartree, gradients in Hartree/bohr, transition dipoles in `e·a0`,
positions in bohr. Oscillator strengths are dimensionless.

Owning module: [`src/td.rs`](../src/td.rs). Gates: [`tests/td.rs`](../tests/td.rs)
plus the in-module `src/td.rs` diagnostic tests.

---

## 1. The model

TD-GFN1 is the Tamm-Dancoff approximation (TDA) of the transition-charge
TD-DFTB response model (Niehaus et al., *Phys. Rev. B* **63**, 085108 (2001))
built on the GFN1 Mulliken transition shell charges and the GFN1 SCC response
kernel. For a converged, **gapped, closed-shell** ground state the TDA matrix
over occupied→virtual pairs `(i,a)` is

```text
A_{ia,jb} = (eps_a - eps_i) delta_ij delta_ab  +  c * sum_{s,t} q^{ia}_s K_st q^{jb}_t
```

with

| symbol | meaning |
| --- | --- |
| `eps_p` | ground-state MO (or Bloch band) energy |
| `q^{ia}_s = -sum_{mu in s} (C_{mu i}(SC)_{mu a} + C_{mu a}(SC)_{mu i})` | Mulliken **transition shell charge** of pair `(i,a)` on shell `s` |
| `K_st = d^2 E_SCC / dq_s dq_t` | the SCC response kernel: the second-order Klopman-Ohno `gamma_st` **plus** the on-site anharmonic block `d^2E_onsite/dq_A^2 = 2 Gamma_A q_A + …` (DFTB3 third order, and the Linear Breathing-Radius orders when `charge_order > 3`) |
| `c` | spin factor: `2` for singlets, `0` for triplets |

The excitation energies are the eigenvalues of `A`, the amplitudes `X` its
(orthonormal) eigenvectors. Oscillator strengths use the Mulliken monopole
transition dipole `mu = sqrt(2) sum_{ia} X_ia sum_s q^{ia}_s R_{A(s)}`, consistent
with the GFN1 point-charge electrostatics, so `f = (2/3) omega |mu|^2`.

### What is *not* in the kernel

* **Triplets get no magnetic kernel.** `K` is spin independent in GFN1, so
  `TdaSpin::Triplet` sets `c = 0` and the triplet spectrum is *exactly* the sorted
  bare orbital-energy gaps (gated: `tda_triplet_matches_orbital_gaps_and_singlet_is_higher`,
  agreement `< 1e-9` Hartree). This is spin-restricted TD-DFTB without the
  spin-constant `W` term — it is a real model gap, not an implementation gap.
* **No full response beyond the monopole channel.** The response kernel is the
  same shell-charge kernel the SCC uses. Any experimental model layer that is not
  part of that kernel (multipole/CAMM electrostatics, Fock exchange, +U, spin
  polarization, external field) is **not** in the TDA coupling. The TDA entry
  points do not reject those option sets — they solve the plain GFN1 TDA on top of
  whatever ground state you converged.
* **Dispersion, repulsion and the halogen correction are state independent.** They
  cancel exactly in `d omega/dR` and are present, in full, in the *total*
  excited-state gradient (see §3).

### Requirements

`CpxtbSpace::from_occupations` demands integer closed-shell occupations, so the
TDA and every TDA gradient need `electronic_temperature = 0` and a real HOMO-LUMO
gap. A non-positive occupied→virtual gap is rejected:

> `TD-GFN1 requires a positive occupied-virtual gap (gapped closed shell)`

---

## 2. API map

All names below are re-exported at the crate root (`gfn1_rs::…`).

### Energies

| Entry point | Boundary | What it solves |
| --- | --- | --- |
| `solve_tda` | non-periodic | Dense TDA from a converged `ElectronicResult`. |
| `solve_tda_pbc_gamma` | periodic, Γ only | Runs its own Γ-point PBC SCC, builds the Ewald Klopman-Ohno kernel, then the same dense TDA on the real Γ Bloch MOs. |
| `solve_tda_kpoint` | periodic, Monkhorst-Pack | Diagonalises the complex `F(k)` at every k-point and assembles the optical (`q = 0`) TDA over all occ→virt band pairs in the BZ; transition charges carry `sqrt(w_k)`. Reduces **exactly** to `solve_tda_pbc_gamma` on a Γ-only mesh (gated, `< 1e-9`). |
| `tda_frozen_excitation_energy` | non-periodic | The Rayleigh quotient `X^T A(R) X / X^T X` at a **fixed** amplitude vector. See the gauge warning in §4. |
| `tda_rotatory_strengths`, `tda_optical_rotation` | non-periodic | ECD rotatory strengths and the Rosenfeld `beta(omega)`. |

`TdaResult.pairs` is the amplitude ordering: `(i, a)` MO indices for the molecular
and Γ paths, and `(ik, i*n + a)` k-point/band labels for `solve_tda_kpoint`.

### Gradients

Every gradient entry point returns `TdaGradientResult`, whose `gradient` field is
the gradient of the **total excited-state energy**

```text
E_exc(R) = E_ground_free(R) + omega_state(R)
```

i.e. it *includes* repulsion, D3 dispersion and the halogen correction through the
ground-state part. `total_energy` is that same sum, and `forces = -gradient`. To
get `d omega/dR` alone, subtract the ground-state gradient
(`analytic_gradient(...).gradient`) — that is exactly what the gate
`tda_analytic_gradient_matches_fd_with_h_ladder` does.

| Entry point | Boundary | Method | Cost |
| --- | --- | --- | --- |
| `solve_tda_gradient_analytic` | non-periodic **and** periodic Γ | Fully analytic (direct CPHF), see §3 | 1 SCC + 1 CPHF over `3N` |
| `solve_tda_kpoint_gradient_analytic` | periodic, any k-mesh | Fully analytic, BZ-summed | 1 SCC + per-k CPHF over `3N` |
| `solve_tda_gradient_seminumerical` | non-periodic only | Analytic ground gradient + central FD of the frozen-amplitude `omega` | `6N` SCC |
| `solve_tda_gradient` | non-periodic **and** periodic Γ | Full central FD of the re-diagonalised, root-tracked total energy | `6N` (SCC + TDA) |
| `solve_tda_kpoint_gradient` | periodic, any k-mesh | Full central FD on the k-mesh | `6N` (SCC + TDA) |
| `solve_tda_gradient_method` | dispatch | Selects `Analytic` / `SemiNumerical` / `FiniteDifference` | — |

`TdaGradientMethod::parse` accepts `analytic` / `lagrangian` / `z_vector`,
`semi_numerical` / `semi`, and `finite_difference` / `fd` / `numerical`
(case- and hyphen-insensitive). `SemiNumerical` is the `Default`.

### Options that are honored, and options that are not

* **Honored:** `TdaOptions { n_states, spin }`; the whole `ElectronicOptions` of the
  ground-state SCC (convergence thresholds, `charge_order`, the CN-dependent `H0`
  switch — the CPHF chain follows `hamiltonian.enable_cn_hamiltonian`); the `KMesh`
  on the k-point entry points.
* **Silently overridden:** every periodic TD entry point constructs its own
  `PbcOptions` as `PbcOptions { kmesh, ..Default::default() }`. There is no
  `PbcOptions` parameter, so Ewald settings, cutoffs and the boundary condition are
  **not** configurable from the TD API, and `ElectronicOptions::boundary` has no
  effect on which periodic path runs — that is decided by `system.lattice`.
* **Not in the coupling kernel** (but not rejected either): the experimental model
  layers listed in §1.
* **State independent, so they cancel in `d omega/dR` but *are* in the returned
  total gradient:** D3 dispersion, repulsion, the halogen correction.

`solve_tda_gradient_method(..., SemiNumerical)` on a periodic system errors:

> `solve_tda_gradient_seminumerical is non-periodic; use solve_tda_gradient (finite difference) for periodic (Gamma-point) systems`

`solve_tda_gradient_analytic` silently *routes* a periodic system to the
Γ-point analytic path — it does not error.

---

## 3. The analytic gradient

At the eigenvector the excitation energy is stationary in the amplitudes
(`d omega/dX = 0`), so the `2n+1` rule applies and no amplitude response is
needed:

```text
d omega/dR = sum_{ia} X_ia^2  d(eps_a - eps_i)/dR              (gap term)
           + c * P^T (dK/dR) P                                  (kernel term)
           + 2c * (dP/dR)^T K P                                 (transition-charge term)
```

with the state transition shell charges `P_s = sum_{ia} X_ia q^{ia}_s`.

**Gap term.** `d eps_p/dR = (C^T (dH0/dR + dF^scc/dR) C)_pp - eps_p (C^T dS/dR C)_pp`.
`dF^scc/dR` is the SCC response Fock built from the full ground-state shell-charge
response — the CPHF density response `dD/dR` *plus* the explicit `dS/dR` metric
piece of the Mulliken populations. The CPHF is
`solve_nonpbc_cpxtb_hessian_response` (Γ PBC: `pbc::hessian`), the same solver the
analytic Hessian uses, so the CN-dependent `H0` chain is included whenever
`enable_cn_hamiltonian` is on.

**Kernel term.** `K` splits into a purely geometric piece and a charge-dependent
on-site piece, and *both* must be differentiated:

* `c P^T (dgamma/dR) P` — the explicit position derivative of the second-order
  Klopman-Ohno kernel (`coupling_kernel_gradient` non-PBC /
  `transition_kernel_gamma_gradient` PBC).
* `c sum_A (d^3 E_onsite/dq_A^3) (dq_A/dR) P_A^2` — the on-site anharmonic block
  `d^2E_onsite/dq_A^2 = 2 Gamma_A q_A + …` has *no* explicit position dependence,
  but it is evaluated at the **ground-state** charge `q_A`, which moves with the
  nuclei. Folding the atomic weight onto the atom's shells makes this a linear
  functional of the shell-charge response the CPHF already produces
  (`onsite_third_order_coupling_weights`).

The second piece was missing before this revision. It is invisible for triplets
(`c = 0`), for dark states (`P = 0`) and for `Gamma = 0`, so only a bright state
with third-order electrostatics exposes it. Measured on the jittered-formaldehyde
`S3` root it was a `3.35e-6` Hartree/bohr **constant** offset that did not shrink
with the step (`3.44e-6 -> 3.35e-6` for `h = 2e-3 -> 1e-3`, ladder ratio 1.03);
zeroing `Gamma` in both the analytic path *and* the FD oracle collapsed it to
`6.29e-7 -> 1.57e-7`, ratio 4.00, which identified the term. With it added the
unmodified-parameter residual is `1.45e-6 -> 3.62e-7`, ratio **4.00**.

**Transition-charge term.** `dq^{ia}/dR` needs the full first-order
orbital-rotation matrix `U` (`C^(R) = C U`): occ-virt from the converged CPHF,
occ-occ / virt-virt from Brillouin/eigenvalue stationarity
`U_pq = (F^(R)_pq - eps_q S^(R)_pq)/(eps_q - eps_p)` (falling back to the symmetric
metric `-1/2 S_pq` for degenerate pairs), and the diagonal from the orthonormality
metric `-1/2 S_pp`.

The k-mesh generalisation adds the BZ sum with `sqrt(w_k)`-weighted transition
charges and complex per-k orbital rotations; the discrete max-AO phase fixing that
pins each Bloch band is never differentiated, because the gauge-covariant `<i|.|a>`
products are all that enter.

There is **no Z-vector / relaxed-difference-density step**, and none is needed:
the relaxed density is the machinery you need to differentiate a *variational*
excitation energy through the ground-state orbitals, and the direct-CPHF route
does that differentiation explicitly instead. The earlier Lagrangian port is
retained only as `#[cfg(test)]` diagnostics in `src/td.rs`; it carried a
`~7e-3 Hartree/bohr` residual (`tda_zvector_scale_sweep`).

---

## 4. The MO phase gauge (fixed in this revision)

A symmetric eigensolver fixes each eigenvector only up to a sign, and the sign it
returns is **not** a continuous function of geometry. The TDA *spectrum* is gauge
invariant — flipping MO `p` is a diagonal similarity transform `D A D`,
`D = diag(+-1)`, which leaves the eigenvalues alone — but a **frozen-amplitude**
Rayleigh quotient at fixed `X` is not: `q^{ia}` flips sign with either orbital, so
the transition-charge Coulomb coupling picks up sign-flipped cross terms and the
frozen excitation energy acquires a **step discontinuity** along the geometry.

Consequences before the fix, measured on jittered water:

| root | `f` | `max abs(analytic - FD(omega))`, `h = 2e-3` | `h = 1e-3` |
| --- | --- | --- | --- |
| `S0` (dark, transition dipole `~1e-15`) | 3.6e-31 | 1.18e-7 | 2.95e-8 |
| `S1` | 3.7e-4 | **2.16** | **4.31** |
| `S3` | 1.9e-2 | **7.16** | **14.3** |
| `S4` | 3.0e-4 | **4.9e-2** | **1.9e-2** |

The error **grows as `1/h`** (a fixed energy step divided by `2h`), and the
`SemiNumerical` gradient — the **default** `TdaGradientMethod` — inherited it
exactly. Dark roots, whose transition charges vanish identically, were accidentally
immune, which is why every pre-existing gate (all of which used water `S0` or a
dark formaldehyde root) passed.

The fix pins the MO phase gauge of every displaced solve to the reference geometry
(`phase_align_mos_to_reference`: flip column `p` when `<C^ref_p | S | C_p> < 0`).
After it, the same rows are

| root | `h = 2e-3` | `h = 1e-3` | ratio |
| --- | --- | --- | --- |
| `S1` | 5.80e-7 | 1.45e-7 | 4.00 |
| `S3` | 2.65e-6 | 6.63e-7 | 4.00 |
| `S4` | 1.99e-7 | 4.98e-8 | 4.00 |

i.e. pure `O(h^2)` finite-difference truncation. (These are the final numbers, with
the §3 third-order kernel term also in place.)

**API consequence.** `tda_frozen_excitation_energy` now takes a
`reference: Option<&ElectronicResult>` argument. Pass `Some(&reference_scc)` for
**any** cross-geometry use (every finite difference). `None` reproduces the old
behaviour and is only correct at the reference geometry itself. The same alignment
is applied inside `solve_tda_gradient` (non-periodic branch) so that its
amplitude-overlap root tracking compares amplitudes in one gauge.

The **periodic** `solve_tda_gradient` branch re-diagonalises without that
alignment; it is a finite-difference reference only and the periodic analytic
paths are gated directly against it (§5), but a periodic frozen-amplitude route
does not exist and should not be added without the same alignment.

---

## 5. Gates and measured numbers

Fixtures are deliberately **jittered off their symmetry**: on symmetric water the
lowest TDA roots carry *exactly zero* transition charge (`|mu| ~ 1e-15`), which
switches off the entire coupling-derivative machinery and makes a gate on those
roots vacuous for two of the three gradient terms. Each gate therefore asserts
that at least one gated root is bright.

| Gate (`tests/td.rs`) | What it pins | Measured |
| --- | --- | --- |
| `tda_analytic_gradient_matches_fd_with_h_ladder` | molecular analytic vs central FD of **both** the total excited energy (re-diagonalised, root-tracked) and the excitation energy alone (frozen amplitudes), `h = 2e-3 -> 1e-3` | see the per-root table below |
| `tda_gradient_translational_invariance` | `sum_A dE/dR_A = 0` for the molecular, semi-numerical, Γ-PBC and k-mesh analytic gradients | `<= 7.7e-17` molecular, `<= 6.7e-16` Γ-PBC and k-mesh |
| `tda_gradient_method_dispatch_and_cross_consistency` | `solve_tda_gradient_method` is bit-identical to each concrete entry point; semi-numerical vs analytic vs full FD | dispatch exact (`0.0`); semi vs analytic `1.4e-7`, full FD vs analytic `1.7e-7` |
| `pbc_gamma_tda_gradient_analytic_matches_fd` | Γ-PBC analytic vs Γ-PBC FD, 3 roots | `< 1e-5` |
| `pbc_gamma_tda_gradient_matches_molecular_limit` | large-box Γ-PBC → molecular limit for excitation **energies and gradients**, `L = 11 A` then `16 A`, residual must shrink | `1.15e-4 / 9.09e-5` at `L = 11`, `3.57e-5 / 3.10e-5` at `L = 16` (Hartree, Hartree/bohr) |
| `kmesh_tda_gradient_analytic_matches_fd` | k-mesh analytic vs k-mesh FD (Γ, `2x1x1`, dispersive HF `1x2x1`) | `< 1e-5` |
| `kmesh_gamma_reduces_to_pbc_gamma_gradient_and_matches_fd_on_a_chain` | Γ-only k-mesh == Γ path (energy and gradient); 1D chain `3x1x1` analytic vs FD | `3.7e-16` / `1.7e-16` (reduction), `5.32e-7` (chain vs FD) |
| `tda_near_degenerate_root_gradient` | the near-degenerate formaldehyde `S2/S3` pair (`Delta omega = 1.07e-3` Hartree) | documented below |
| `tda_frozen_excitation_energy_reproduces_tda_energy` | `X^T A X == omega` at the reference geometry, both gauges, singlet and triplet | `< 1e-8` |

Per-root numbers of the molecular ladder gate (`max |analytic - FD|`, Hartree/bohr,
`h = 2e-3 -> 1e-3`; `_total` is the re-diagonalised total-energy oracle, `_omega`
the frozen-amplitude excitation-energy oracle):

| fixture / root | `f` | `_total` | `_omega` |
| --- | --- | --- | --- |
| water `S0` | 3.6e-31 | 7.06e-7 → 1.77e-7 | 1.18e-7 → 2.95e-8 |
| water `S1` | 3.7e-4 | 6.74e-7 → 1.68e-7 | 5.80e-7 → 1.45e-7 |
| water `S2` | 5.0e-31 | 4.46e-7 → 1.12e-7 | 1.42e-7 → 3.55e-8 |
| water `S3` | 1.9e-2 | 5.60e-7 → 1.40e-7 | 2.65e-6 → 6.63e-7 |
| water `S4` | 3.0e-4 | 4.49e-7 → 1.12e-7 | 1.99e-7 → 4.98e-8 |
| formaldehyde `S0` | 5.1e-6 | 2.26e-6 → 5.65e-7 | 2.65e-7 → 6.63e-8 |
| formaldehyde `S1` | 3.4e-7 | 2.24e-6 → 5.59e-7 | 7.22e-6 → 1.81e-6 |

Every ratio is in `[3.3, 4.5]`, i.e. `O(h^2)` finite-difference truncation with no
constant analytic residual.

### Near-degeneracy and root tracking

Formaldehyde has a near-degenerate singlet pair, `S2 = 0.314852` and
`S3 = 0.315925` Hartree (`Delta = 1.07e-3`). Two things degrade there, and they
degrade differently:

* The **re-diagonalised FD oracle** (`solve_tda_gradient`) follows the root by
  amplitude overlap. Inside a near-degenerate pair the two eigenvectors are an
  essentially arbitrary mixture of the same two-dimensional subspace, so the
  tracking is ill-posed: `|analytic - FD|` at `h = 1e-3` is `1.91e-5` on `S2` and
  `2.02e-5` on `S3`, against the `~1.5e-7` it reaches on well-separated roots.
* The **frozen-amplitude oracle** is immune to root flipping by construction (it
  never re-diagonalises) and stays clean through the pair: `6.03e-7 -> 1.51e-7` on
  `S2` and `1.45e-6 -> 3.62e-7` on `S3`, both ratio `4.00`.

The gates therefore skip roots closer than `5e-3` Hartree to a neighbour when
asserting the `h^2` ladder, and cover the near-degenerate pair in a dedicated gate
against the frozen oracle instead. **For a user this means: near a conical
intersection or an avoided crossing, use `solve_tda_gradient_analytic` (which
never re-diagonalises at a displaced geometry) and treat the state label as
belonging to the amplitude vector, not to an energy ordering.**

---

## 6. Limitations

* **Integer occupations only.** No Fermi smearing anywhere in the TDA or its
  gradients. Periodic finite-T response is separately known-broken
  ([limitations.md](limitations.md)); the TDA never goes near it.
* **Triplets have no magnetic kernel** — the triplet spectrum is the bare
  orbital-gap spectrum (§1).
* **Only the monopole (shell-charge) response channel** is in the coupling kernel;
  experimental model layers are not, and are not rejected.
* **`tda_frozen_excitation_energy` with `reference = None` is only valid at the
  reference geometry** (§4).
* The **periodic** branch of `solve_tda_gradient` does not phase-align its
  displaced solves, so its amplitude-overlap root tracking is less robust than the
  non-periodic branch.
* **No excited-state properties beyond energies, oscillator strengths, rotatory
  strengths and gradients** — no excited-state Hessian, no non-adiabatic couplings,
  no state-specific density.
* `solve_tda_kpoint` requires an integer closed-shell **band** filling:

  > `k-point TD-GFN1 requires an integer closed-shell band filling (gapped insulator)`

  and rejects a metallic mesh:

  > `k-point TD-GFN1 found no positive-gap occupied->virtual transitions (metallic or non-integer occupations are not supported)`
