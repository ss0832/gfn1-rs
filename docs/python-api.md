# Python / ASE API

Build the extension (Python 3.9–3.14):

```bash
python -m pip install maturin "ase>=3.22" "numpy>=1.22"
maturin develop --release --features python
```

For a Python newer than PyO3's support window, set
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `cargo`/`maturin`.
`default_param_path()` returns `$GFN1_XTB_PARAM`, or pass `param_path=...`.

> **Periodic systems are supported through Python.** The native calculator's
> `calculate_periodic(numbers, positions, cell, pbc, kgrid, ...)` runs the Gamma/k-point PBC path
> (energy, forces, stress); the ASE wrapper auto-dispatches to it whenever `any(atoms.pbc)`. The
> experimental **angular multipole** correction (`multipole=True`, arbitrary `multipole_order`) now
> runs fully under PBC — **energy, analytic forces, and analytic stress** — so variable-cell
> relaxation (ASE `ExpCellFilter`) works with the multipole on.

## Low-level: `Gfn1NativeCalculator`

Accepts the same model controls as the CLI. Constructor keywords and defaults:

```python
from gfn1_rs import Gfn1NativeCalculator, default_param_path

calc = Gfn1NativeCalculator(
    param_path=default_param_path(),
    charge=0.0, multiplicity=None,
    spin_polarization=False,   # spGFN1: collinear spin term for open-shell (non-PBC, energy+forces);
                               # closed-shell singlet == GFN1. Set a multiplicity to use it.
    max_scc=250, energy_tolerance=1.0e-6, charge_tolerance=2.0e-5,
    mixing=0.4, scc_broyden=True, scc_broyden_size=250,
    electronic_temperature=300.0,
    nprim=0, eigen_tolerance=1.0e-12,
    enable_dispersion=True, d3_reference_path=None,
    experimental_d4=False, d4_cutoff=None, d4_cn_cutoff=None,
    d4_atm=True, d4_atm_cutoff=None, d4_s9=None,
    enable_cn_hamiltonian=True,
    multipole=False,                   # experimental mDFTB2 dipole+quadrupole correction
    multipole_octupole=False,          # + atomic rank-3 octupole (requires multipole; needs d fns)
    field_multipole=False,             # + first-order field-dipole coupling (requires multipole + field)
    multipole_third_order=False,       # + on-site charge*dipole^2 / charge*quad^2 terms (requires multipole)
    multipole_secondary_basis=None,    # evaluate the moments over a cc-pVnZ basis name/path (requires multipole)
    charge_order=3,                    # on-site charge expansion order: 3 = stock GFN1, n>=4 experimental
    multipole_order=0,                 # arbitrary-rank multipole path: n>=4 (requires multipole); <=3 = legacy
    multipole_charge_order=[],         # per-rank multipole x charge cross terms, e.g. [6,4,2] (requires multipole)
    lr_exchange=False,                 # experimental long-range Mulliken Fock exchange (MFX)
    onsite_exchange=False,             # + exact one-center Fock exchange (OFX) on top of MFX (requires lr_exchange)
    dynamic_omega=False,               # + geometry-adaptive omega (LocalGeometry: omega_A=eta_A/(1+CN_A)^(-1/3))
    scf_trah=False,                    # force the TRAH second-order SCF for the exchange path (else AutoTRAH)
    multipole_rank_ladder_base=None,   # v0.2.2: rank-continuation base for the high-rank multipole SCC
)

numbers = [1, 1]
positions = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]

result = calc.calculate(numbers, positions, unit="angstrom", compute_gradient=True)
print(result.energy_ev)                        # finite-T (Mermin) free energy E - T*S_elec, eV
print(dict(result.energy_terms_hartree()))     # also energy_terms_ev(); has total_internal / total_free
print(result.forces_ev_per_angstrom)           # None unless compute_gradient=True
print(result.charges, result.converged, result.iterations)
```

`CalculationResult` exposes `energy_hartree` / `energy_ev` (the **finite-temperature
Mermin free energy** `E − T·S_elec`, = the internal energy at `T_elec = 0`; the
plain internal energy is `energy_terms_*()["total_internal"]`),
`gradient_hartree_per_bohr`, `forces_ev_per_angstrom`, `charges`, `iterations`,
`converged`, and the `energy_terms_hartree()` / `energy_terms_ev()` dictionaries.
(There is no `free_energy` field — that name is ambiguous with the vibrational free
energy from a frequency analysis.)

The Rust-native L-BFGS optimizer is a separate method:

```python
opt = calc.optimize(
    numbers, positions, unit="angstrom",
    max_iterations=250, gradient_tolerance=1.0e-4,
    step_tolerance=1.0e-7, history=12, max_atom_step=0.30,
    trajectory_path="opt_traj.xyz",   # optional: stream the trajectory live (see below)
)
print(opt.converged, opt.iterations, opt.max_gradient)
print(opt.positions_angstrom)                  # relaxed geometry
print(opt.forces_ev_per_angstrom)
```

`trajectory_path=` **streams** a multi-frame XYZ to that file as the optimization runs — one
flushed frame per L-BFGS step — so the path can be watched live in a trajectory viewer.
Alternatively, the result also carries the whole trajectory in memory, to write out at the end:

```python
print(opt.trajectory_energies_hartree)         # per-step energies (frame 0 = input)
print(len(opt.trajectory_positions_angstrom))  # frames x atoms x 3
open("opt_final.xyz", "w").write(opt.to_xyz())          # final geometry (XYZ)
open("opt_traj.xyz", "w").write(opt.trajectory_xyz())   # multi-frame trajectory (XYZ)
```

Unit / conversion constants are re-exported from the module:
`HARTREE_TO_EV`, `BOHR_TO_ANGSTROM`, `ANGSTROM_TO_BOHR`,
`FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM`,
`HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2`.

The native surface lives in `gfn1_rs.native` (`Gfn1NativeCalculator`, the result types,
`default_param_path`) and is re-exported at the package top level, so
`from gfn1_rs import Gfn1NativeCalculator` and `from gfn1_rs.native import Gfn1NativeCalculator`
are equivalent. The ASE wrapper (`gfn1_rs.ase`) is a thin convenience layer over this class.

`experimental_d4=True` replaces D3 with the experimental non-PBC,
self-consistent D4 path including the ATM three-body term. The GFN1 damping
constants `a1`, `a2`, and `s8` are read from `param_gfn1-xtb.txt`; `d4_s9`
defaults to the GFN2-xTB value `5.0` only when D4 is active, while ordinary
non-D4 calculations resolve to `s9 = 0`. `d4_cutoff`, `d4_cn_cutoff`,
`d4_atm_cutoff`, `d4_atm`, and `d4_s9` override the defaults. The ASE wrapper
exposes the same keywords.

The experimental **`multipole=True`** flag turns on the parameter-free mDFTB2 atomic
dipole+quadrupole electrostatics correction (self-consistent energy + analytic forces; off by
default ≡ GFN1). It flows through `calculate`, `calculate_periodic`, and `optimize`, and through the
ASE wrapper (`GFN1RSCalculator(multipole=True)`). It works **both non-periodically and under PBC**
(v0.2.2): for a cell, the moments (ranks `1..=multipole_order`, default dipole+quadrupole) are mixed
**jointly** with the shell charges in the k-point SCC, with the QCore generalized-Ewald field; the
periodic energy, analytic forces, **and analytic stress** are α/FD-validated. The general
`scripts/optimize.py`
optimizes **any** XYZ with the correction flags (`--multipole`, `--lr-exchange`, `--onsite-exchange`,
`--multipole-order N`, `--charge-order N`); add `--compare` for an on-vs-off comparison.

The further **experimental, off-by-default, FD-gated** electrostatics/exchange knobs (all
non-PBC, self-consistent energy + analytic forces, threaded through `calculate`, `optimize`,
and the ASE wrapper) are:

- **`multipole_order=n`** — generalises the moment ladder to **arbitrary rank**. `n >= 4`
  self-consistently mixes the atomic moments of ranks `1..n` (parameter-free generic path);
  `n <= 3` keeps the byte-compatible legacy dipole/quadrupole/octupole paths. Requires
  `multipole=True`. High rank is experimental and slower (the cost grows with rank).
- **`multipole_charge_order=[o1, o2, ...]`** — per-rank **multipole × charge cross terms**: entry
  `l-1` is the highest on-site charge order coupled to the rank-`l` (2^l-pole) atomic multipole,
  via the breathing-radius Taylor expansion of `½ g_l(η(q))(m_l·m_l)`. `[]` (default) = no cross
  terms. Example `[6, 4, 2]` = dipole→6th, quadrupole→4th, octupole→off. Each entry must satisfy
  `order <= 2l+3` (the rank-`l` self-energy terminates there; a higher value **raises an error**,
  it is never silently truncated). Requires `multipole=True` and `multipole_order >=` the highest
  rank carrying a cross term (it forces the generic path on). Generalises `multipole_third_order`
  (which is the `l in {1,2}`, order-3 special case). Parameter-free; self-consistent energy +
  analytic forces.

  ```python
  # per-rank cross terms: dipole→4th-order charge, quadrupole→4th-order charge.
  calc = Gfn1NativeCalculator(
      param_path=default_param_path(),
      multipole=True, multipole_order=2, multipole_charge_order=[4, 4], charge_order=4,
  )
  res = calc.calculate(numbers=[8, 1, 1], positions=pos, unit="angstrom")
  # (ASE: GFN1RSCalculator(multipole=True, multipole_order=2, multipole_charge_order=[4, 4], ...))
  ```
- **`multipole_rank_ladder_base=b`** (v0.2.2) — **rank-continuation** for a robust high-rank
  multipole SCC. A cold direct 16-pole+ (`multipole_order >= 4`) SCC can struggle: the
  monopole↔high-multipole coupling oscillates. With this set to a low base rank `b`, the SCC is
  converged one rank at a time — `b`, `b+1`, …, up to `multipole_order` — warm-starting each
  rank's shell charges from the previous (lower-rank) converged result, so only the newly added
  rank has to relax. It reaches the **same** SCF solution as a direct run, just more reliably.
  `None` (default) = direct. Non-periodic.
  ```python
  # octupole -> 16-pole -> 32-pole, staged:
  calc = Gfn1NativeCalculator(
      param_path=default_param_path(),
      multipole=True, multipole_order=5, multipole_rank_ladder_base=3,
  )
  # (ASE: GFN1RSCalculator(multipole=True, multipole_order=5, multipole_rank_ladder_base=3))
  ```
- **`lr_exchange=True`** — parameter-free **long-range Mulliken Fock exchange** (MFX,
  LC-DFTB style): `E_x = ½Tr[ΔP·K[ΔP]]` over `ΔP = P − P0`, with a hardness-derived
  range-separated `γ^lr` and `HardnessPairwise` `ω`, added self-consistently. Size-consistent
  (overlap-weighted); the exchange path uses a density-matrix DIIS solver.
- **`onsite_exchange=True`** — **exact one-center (Fock) exchange** layered on MFX (requires
  `lr_exchange`): replaces the same-atom Mulliken exchange with the real one-center
  two-electron integrals (`K_OFX = refined − Mulliken-onsite`, no double count). The ERIs are
  geometry-independent and cached per element. OFX adds no explicit force (static `ω`).
- **`dynamic_omega=True`** — experimental **geometry-adaptive range separation** (requires
  `lr_exchange`): `ω_A = η_A / s_A`, `s_A = (1+CN_A)^(−1/3)` from the GFN1 coordination number, so a
  more-coordinated atom screens at shorter range (reduces to the default `HardnessPairwise` at
  `CN = 0`). The analytic gradient adds the `∂ω/∂R` reorganisation force for both MFX and OFX; the
  one-center ERIs are factored into an ω-independent, memory-efficient per-element skeleton so a
  dynamic-ω optimisation re-evaluates (not rebuilds) the d/f-element integrals each step. On dppf-PdCl₂
  it changes the optimised energy by ≈ −4 Ha and the geometry by ≈ 0.12 Å RMSD vs the static `ω`.
- **`scf_trah=True`** — force the matrix-free **Trust-Region Augmented Hessian** second-order
  SCF for the exchange path (minimises the energy over orbital rotations; robust where DIIS
  stalls). Left `False`, the exchange path uses **AutoTRAH**: DIIS first, TRAH automatically
  if it does not converge. Closed-shell / gapped systems.

```python
calc = Gfn1NativeCalculator(param_path=default_param_path(),
                            lr_exchange=True, onsite_exchange=True)  # MFX + OFX
res = calc.calculate([8, 1, 1],
                     [[0,0,0], [0.757,0.586,0], [-0.757,0.586,0]],
                     unit="angstrom", compute_gradient=True)
print(res.energy_hartree, res.converged)
```

### All property methods (non-ASE)

`Gfn1NativeCalculator` exposes **every** property the ASE calculator does — call them directly
with `numbers` (atomic numbers) and `positions` (nested list, `unit="angstrom"` by default).
There is no ASE dependency. Most return plain Python lists / dicts (wrap with `numpy.asarray`
as needed). Grouped:

```python
import gfn1_rs
calc = gfn1_rs.Gfn1NativeCalculator(param_path=gfn1_rs.default_param_path())
nums = [8, 1, 1]
pos = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]

# --- single point / geometry ---
res  = calc.calculate(nums, pos, unit="angstrom", compute_gradient=True)  # CalculationResult
opt  = calc.optimize(nums, pos, unit="angstrom")                          # OptimizationResult

# --- Hessian & vibrations (fully analytic; non-periodic AND periodic) ---
H    = calc.hessian(nums, pos, unit="angstrom")            # 3N x 3N analytic Hessian (Hartree/bohr^2)
vib  = calc.vibrational_frequencies(nums, pos)             # {"wavenumbers" cm^-1, "modes"}
# periodic (fixed-cell): Gamma when kgrid=None/[1,1,1], else the k-point mesh Hessian:
Hpbc = calc.hessian_periodic(nums, pos, cell=[[10.0,0,0],[0,10.0,0],[0,0,10.0]],
                             pbc=(True, True, True), kgrid=(2, 2, 2))   # 3N x 3N

# --- electric response / spectroscopy ---
alpha = calc.polarizability(nums, pos)              # 3x3 static polarizability (a0^3)
ddip  = calc.dipole_derivatives(nums, pos)          # dmu/dR (IR tensor)
ir    = calc.ir_spectrum(nums, pos)                 # harmonic wavenumbers + IR intensities
raman = calc.raman_spectrum(nums, pos)              # + Raman activities

# --- cubic force constants (nuclear third derivative) ---
# STRICT CLOSED FORM T_abc = d^3E/dR_a dR_b dR_c (v0.3.0): fully analytic 2n+1 assembly -- NO finite
# differences anywhere. Three output modes (v0.3.1) so large systems never hold the full 3N^3 tensor:
import numpy as np
#  (a) Dense  -- list of 3N dense slabs, slab[c][a][b] = T_abc (Hartree / bohr^3):
t3 = calc.third_derivative(nums, pos)                      # list of 3N (3N x 3N) slabs
#  (b) Vector -- the directional contraction K[a][b] = sum_c v_c T_abc returned as ONE 3N x 3N matrix
#      (returns only the contraction, not the 3N^3 tensor; recommended when you need a direction):
v = np.zeros(3 * len(nums)); v[0] = 1.0
kv = calc.third_derivative_vector(nums, pos, v.tolist())   # 3N x 3N matrix
#  (c) Block  -- the O(|block|^3) sub-tensor over chosen atoms (local anharmonicity):
dofs, tblk = calc.third_derivative_block(nums, pos, [0, 2])  # (dofs, slabs); slabs[ci][ai][bi]
# Cheaper SEMI-NUMERICAL directional alternative (two analytic-Hessian evaluations, FD precision):
k = calc.third_derivative_along(nums, pos, v.tolist())     # 3N x 3N matrix (= sum_c v_c T_abc)
# The strict closed form matches FD(analytic Hessian) to ~7e-5 at equilibrium and, since v0.4.4
# (the Pulay coordination-number-response term), to ~1e-4 at strongly non-equilibrium geometries.

# --- excited states (TD-GFN1 / TDA) ---
exc   = calc.tda(nums, pos, n_states=8, spin="singlet")
cd    = calc.rotatory_strengths(nums, pos, n_states=6)  # R_n + magnetic_transition_dipoles
beta  = calc.optical_rotation(nums, pos, frequencies_ev=[0.0, 2.1])

# --- magnetic / magneto-optical (non-periodic; m1_basis_path= selects M1) ---
em    = calc.magnetic_energy(nums, pos, b_field=(0.0, 0.0, 0.05))
xi    = calc.magnetizability(nums, pos)             # analytic=True default; 1e-30 J/T^2
xit   = calc.magnetizability_tensor(nums, pos)      # also magnetizability_diagonal(...)
aB    = calc.magnetic_polarizability(nums, pos)     # alpha_ij(B)
cm    = calc.cotton_mouton(nums, pos)               # d^2 alpha / dB^2  [k][i][j]
mcd   = calc.mcd(nums, pos)                         # d alpha / dB (Faraday) [k][i][j]
sig   = calc.nmr_shielding(nums, pos, nucleus=0)    # NMR shielding sigma_ab, 3x3, ppm
L     = calc.angular_momentum(nums, pos)            # raw <mu|L_a|nu> = -i*L[a]
ldip  = calc.lao_dipole(nums, pos, b_field=(0,0,0)) # dict {"re","im"} of LAO dipole integrals
mf    = calc.magnetic_forces(nums, pos, b_field=(0.0, 0.0, 0.05))  # dict energy/gradient/forces

# --- parameters (finite-difference derivatives + PyTorch interop) ---
dEdp  = calc.parameter_derivatives(nums, pos, targets=[...])
```

Constructor keywords match the example above (and `ase.py`'s `_make_native_calculator`): a
GFN1 parameter file plus the SCC / dispersion / CN controls. The same model options
(`charge`, `multiplicity`, tolerances, `enable_dispersion`, …) apply to every method. See the
method docstrings (`help(gfn1_rs.Gfn1NativeCalculator.<method>)`) for the per-method signatures
(step sizes, origins, spin, frequencies, `m1_basis_path`, etc.).

### CAMM presets and DFT+U / DFT+U+V

Both apply to `Gfn1NativeCalculator` **and** `GFN1RSCalculator` (same keyword names):

```python
# CAMM-on-mDFTB2 named preset (fills per-element κ + s_onsite; implies multipole + camm_on_mdftb2).
# "sigma-hole" (v0.4.4) is the unified σ-hole preset with per-element s_onsite. See sigma_hole_preset.md.
calc = gfn1_rs.Gfn1NativeCalculator(param_path=..., camm_preset="sigma-hole")

# DFT+U / +U+V (non-periodic; rides the spin path -> open-shell). Fixed U:
calc = gfn1_rs.Gfn1NativeCalculator(param_path=..., spin_polarization=True,
                                    plus_u=True, hubbard_u=[(28, 0.12)])          # Ni d, U=0.12 Ha
# NON-EMPIRICAL U (+V) by linear response -- no fitted parameters (auto-selects TM d shells):
calc = gfn1_rs.Gfn1NativeCalculator(param_path=..., spin_polarization=True,
                                    plus_u=True, hubbard_u_linear_response=True,
                                    plus_u_v=True, hubbard_v=[(28, 7, 0.04)])     # + Ni-N inter-site V
# plus_u_all_d=True extends +U to every d-shell atom (incl. main-group d polarisation).
```

Forces (and geometry optimization) are analytic in every mode, including the geometry response
`dU/dR` of the self-consistently-determined `U` in the non-empirical path (FD-verified).

## ASE: `GFN1RSCalculator`

One calculator class. ASE-standard properties are `energy`, `forces`, and
`charges`; units are ASE-standard (Angstrom, eV, eV/Angstrom).

```python
from ase import Atoms
from ase.optimize import LBFGS
from gfn1_rs.ase import GFN1RSCalculator

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
atoms.calc = GFN1RSCalculator(charge=0.0, multiplicity=None,
                              max_scc=250, energy_tolerance=1.0e-6)

print(atoms.get_potential_energy())            # eV
print(atoms.get_forces())                      # eV/Angstrom
print(atoms.calc.results["native_energy_terms_hartree"])
print(atoms.calc.results["native_converged"], atoms.calc.results["native_iterations"])

# ASE-driven optimization
LBFGS(atoms, logfile="ase.log").run(fmax=0.05)
```

The calculator also exposes the Rust-native L-BFGS, which updates the attached
`Atoms` in place. It is **fixed-cell**: a periodic `Atoms` (any `pbc=True`) relaxes
the atomic positions at the Gamma point with the lattice held fixed (the gradient
routes through the PBC path; non-periodic `Atoms` are optimized as molecules):

```python
result = atoms.calc.optimize_native(
    atoms, max_iterations=250, gradient_tolerance=1.0e-4,
    step_tolerance=1.0e-7, history=12, max_atom_step=0.30,
)
print(result.converged, result.max_gradient)
```

### Periodic cells, stress, and variable-cell dynamics (NPT)

A periodic `Atoms` (any `pbc=True`) routes single points through the Gamma /
k-point PBC path (set the Monkhorst–Pack mesh with the `kgrid=(a, b, c)`
parameter). The periodic **stress** tensor is computed, so ASE's variable-cell
drivers — including the **NPT ensemble** — work directly; the calculator supplies
energy/forces/stress each step and ASE integrates the cell:

```python
import numpy as np
from ase import units
from ase.md.npt import NPT

atoms.calc = GFN1RSCalculator()          # 'stress' is advertised; 'energy' is force-consistent
print(atoms.get_potential_energy())      # finite-T (Mermin) free energy E - T*S_elec, eV
print(atoms.get_stress())                # eV/Angstrom^3, ASE Voigt 6-vector

# Constant-pressure (NPT) MD. (ASE's NPT needs an upper-triangular cell.)
dyn = NPT(atoms, timestep=1.0 * units.fs, temperature_K=300,
          externalstress=1.01325 * units.bar, ttime=25 * units.fs,
          pfactor=(75 * units.fs) ** 2 * 100 * units.GPa)
dyn.run(100)

# Variable-cell *relaxation* (also stress-driven) via ASE's cell filter:
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGS
LBFGS(ExpCellFilter(atoms)).run(fmax=0.05)
```

The native `optimize_native` is fixed-cell; for variable-cell relaxation use ASE's
`ExpCellFilter`/`UnitCellFilter` (above), which drive the lattice from the stress
this calculator provides.

`GFN1RSCalculator` accepts the full default-parameter set (`param_path`,
`charge`, `multiplicity`, `max_scc`, `energy_tolerance`, `charge_tolerance`,
`mixing`, `scc_broyden`, `scc_broyden_size`, `electronic_temperature`, `nprim`,
`eigen_tolerance`, `enable_dispersion`, `d3_reference_path`, `experimental_d4`,
`d4_cutoff`, `d4_cn_cutoff`, `d4_atm`, `d4_atm_cutoff`, `d4_s9`,
`enable_cn_hamiltonian`, plus the experimental multipole/exchange knobs above);
unset values fall back to the native defaults.

See [`scripts/compare_optimizations.py`](../scripts/compare_optimizations.py) for a
worked ASE example that cross-checks the native optimizer against the `tblite` CLI.

## External field, spectroscopy, and SCC controls (v0.1.2)

The external electric field and the SCC convergence controls are constructor
keywords on both calculators; the dipole is reported alongside every result:

```python
calc = GFN1RSCalculator(
    electric_field=(0.0, 0.01, 0.0),   # atomic units (Hartree/(e a0)), or None
    level_shift=0.1,                   # virtual level shift (Hartree)
    scc_accelerator="cdiis",           # "broyden" | "linear" | "cdiis" | "newton"
)
atoms.calc = calc
atoms.get_potential_energy()
print(atoms.calc.results["native_dipole_au"])      # Mulliken dipole (a.u.)
```

The analytic dipole derivatives, polarizability, and IR/Raman spectra are explicit
methods (non-periodic only). They return plain dicts / NumPy arrays:

```python
alpha = atoms.calc.get_polarizability()            # {"tensor", "isotropic", "anisotropy"}
ddip  = atoms.calc.get_dipole_derivatives()        # {"dipole", "ddipole_dr"}
ir    = atoms.calc.get_ir_spectrum()               # {"wavenumbers", "intensities_km_per_mol", ...}
raman = atoms.calc.get_raman_spectrum()            # {"wavenumbers", "activities", ...}
H     = atoms.calc.get_hessian()                    # (3N,3N) analytic Hessian (Ha/bohr^2); PBC-aware:
                                                    #   a periodic Atoms (any pbc=True) returns the fixed-cell
                                                    #   Gamma (or k-point, if kgrid set) Hessian automatically
vib   = atoms.calc.get_vibrational_frequencies()    # {"wavenumbers" cm^-1, "modes"} (non-periodic)
t3    = atoms.calc.get_third_derivative()           # closed-form cubic force constants, Dense T[c,a,b] (3N,3N,3N) ndarray
kv    = atoms.calc.get_third_derivative_vector(v)   # closed-form Vector mode K[a,b]=sum_c v_c T_abc (3N x 3N), memory-lean
dofs, tblk = atoms.calc.get_third_derivative_block([0, 2])  # closed-form Block mode over an atom subset (|block|^3)
k3    = atoms.calc.get_third_derivative_along(v)    # directional (semi-numerical) cubic constants along `v` (3N x 3N)
dp    = atoms.calc.get_parameter_derivatives(["glob:ks", "elem:1:GAM"])  # [{"target","value","energy_derivative"}, ...]
```

The same methods exist on the low-level `Gfn1NativeCalculator`
(`polarizability`, `dipole_derivatives`, `ir_spectrum`, `raman_spectrum`,
`parameter_derivatives`), taking `numbers`/`positions` directly. The module-level
`gfn1_rs.roundtrip_param_file(in_path, out_path)` writes the canonical, value-exact
`param_gfn1-xtb.txt` serialization.

## Periodic systems and TD-GFN1 (v0.1.3)

Periodic `Atoms` now run through the ASE calculator (energy, forces, stress); pass
the Monkhorst-Pack mesh with `kgrid` (`None`/`(1,1,1)` = Gamma):

```python
from ase.build import bulk
atoms = bulk("C", "diamond", a=3.567)
atoms.calc = GFN1RSCalculator(kgrid=(2, 2, 2))
print(atoms.get_potential_energy(), atoms.get_stress())   # eV, eV/Angstrom^3 (Voigt)
```

The low-level `Gfn1NativeCalculator.calculate_periodic(numbers, positions, cell,
pbc, kgrid=..., compute_gradient=..., compute_stress=...)` returns the same
`CalculationResult` (with a 3x3 `stress` in atomic units, Hartree/bohr^3).

The **higher-order on-site charge expansion** `charge_order` (the radial, isotropic Δq
expansion — `4` = quartic Breathing-Radius, `5+` = higher) works for **periodic** systems too:
it is a per-atom local term, so it adds to the k-point SCC energy/potential with no
extra lattice sum. Set `charge_order=4` on either calculator and it applies to both molecular
and periodic single points.

### PBC multipole extension (periodic mDFTB2, arbitrary rank)

The **angular multipole correction** (`multipole=True`) is **also periodic** (v0.3.0): the atomic
moments (ranks `1..=multipole_order`, default dipole+quadrupole) are converged self-consistently
through a **damped generalized-Ewald** field — every rank-pair gets a real-space + reciprocal +
self term — and the same flags carry analytic **forces and stress**. Set the flags on the
calculator exactly as for a molecule; they flow through `calculate_periodic` (energy / forces /
stress) automatically, so the ASE wrapper and `Gfn1NativeCalculator.calculate_periodic`
both honour them:

```python
from ase.build import bulk
atoms = bulk("C", "diamond", a=3.567)
# dipole+quadrupole periodic multipole SCC (+ optional richer expansions):
atoms.calc = GFN1RSCalculator(
    kgrid=(2, 2, 2),
    multipole=True,                 # periodic mDFTB2 multipole SCC (rank 1..multipole_order)
    multipole_order=2,              # 1 = dipole only, 2 = +quadrupole (default), n>=3 = higher rank
    charge_order=4,                 # (optional) quartic on-site monopole expansion, also periodic
)
print(atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())

# Low-level native calculator (numbers/positions/cell directly):
calc = gfn1_rs.Gfn1NativeCalculator(param_path, multipole=True, multipole_order=2)
res  = calc.calculate_periodic(
    nums, pos, cell, pbc=(True, True, True), kgrid=(2, 2, 2),
    compute_gradient=True, compute_stress=True,
)
print(res.energy, res.stress)       # stress in atomic units (Hartree/bohr^3)
```

The full molecular multipole vocabulary applies under PBC too: `multipole_order=n` (arbitrary
rank, n≥3 needs `d` functions for rank 3), `multipole_charge_order=[...]` (per-rank multipole×charge
cross terms), and `multipole_secondary_basis=` (evaluate the moments over a cc-pVnZ basis). High
rank is experimental and slower; the cost grows with rank and with the Ewald range.

TD-GFN1 (TDA) excited states (non-periodic) are available as `get_tda` on the ASE
calculator and `tda` on the native calculator:

```python
exc = atoms.calc.get_tda(n_states=5, spin="singlet")  # or "triplet"
print(exc["excitation_energies_ev"], exc["oscillator_strengths"])

# Excited-state gradient (Hartree/bohr). `method` selects the algorithm:
#   "semi_numerical" (default) = analytic ground gradient + finite difference of
#       the frozen-amplitude excitation energy (exact for a tracked state, fast,
#       non-periodic);
#   "fd"       = full finite difference with root tracking (robust across state
#       crossings; the only option for periodic Gamma-point Atoms);
#   "analytic" = direct-CPHF analytic excitation-energy gradient (FD-verified,
#       non-periodic).
g = atoms.calc.get_tda_gradient(state=0, spin="singlet", method="semi_numerical")
print(g["total_energy_hartree"], g["gradient"])

# Periodic (k-point) TDA over a Monkhorst-Pack mesh (requires a cell + pbc):
kexc = atoms.calc.get_tda_kpoint(kmesh=(2, 2, 2), n_states=5, spin="singlet")
print(kexc["excitation_energies_ev"])

# Periodic excited-state gradient: analytic Gamma + k-mesh (gauge-invariant via the
# natural CPHF gauge), FD-verified.
kg = atoms.calc.get_tda_kpoint_gradient(kmesh=(2, 2, 2), state=0, spin="singlet")
print(kg["gradient"])
```

Periodic stress parameter derivatives and target chunking are available through
the CLI (`--param-stress`, `--target-chunk i/n`); the native
`parameter_derivatives` method covers `dE/dp`.

### Magnetic field (GFN1-xTB-M0 / M1)

The closed-shell magnetic SCC, isotropic magnetizability, and analytic magnetic
forces are exposed on the ASE calculator. Set `m1_basis_path` (here or as a
calculator parameter) to a `GFN1-xTB-cc-pVDZ` secondary-basis file to select the
node-correct **M1** variant; omit it for single-basis **M0**. `b_field` is in
atomic units; everything is non-periodic.

```python
calc = GFN1RSCalculator(param_path=..., m1_basis_path="GFN1-xTB-cc-pVDZ.txt")
b = (0.0, 0.0, 0.05)
e   = calc.get_magnetic_energy(b, atoms=mol)             # Hartree
xi  = calc.get_magnetizability(atoms=mol)                # 1e-30 J/T^2 (analytic CP-SCC by default)
xid = calc.get_magnetizability_diagonal(atoms=mol)       # [xi_xx, xi_yy, xi_zz]
xit = calc.get_magnetizability_tensor(atoms=mol)         # full symmetric 3x3
f   = calc.get_magnetic_forces(b, atoms=mol)             # analytic (Hellmann-Feynman) by default
```

`get_magnetizability(analytic=True)` (default) is the McWeeny density-matrix CP-SCC
response (one SCC + analytic response); pass `analytic=False` for the finite-field
reference. The low-level `Gfn1NativeCalculator.magnetic_energy / magnetizability /
magnetic_forces(...)` take the same arguments (`m1_basis_path=None` -> M0).

### Magneto-optical raw tensors

The differential tensors behind circular dichroism, optical rotation, the
Cotton-Mouton effect and the Faraday/MCD effect are all exposed (non-periodic,
atomic units unless noted):

```python
cd  = calc.get_rotatory_strengths(atoms=mol, n_states=6)  # R_n + raw magnetic transition dipoles m_n0
b0  = calc.get_optical_rotation(atoms=mol, frequencies_ev=[0.0, 2.1])  # Rosenfeld beta(omega)
a   = calc.get_magnetic_polarizability(atoms=mol, b_field=(0,0,0))     # alpha_ij(B), 3x3
cm  = calc.get_cotton_mouton(atoms=mol)                   # d^2 alpha / dB^2, [k,i,j]
mcd = calc.get_mcd(atoms=mol)                             # d alpha / dB (Faraday), [k,i,j]
L   = calc.get_angular_momentum(atoms=mol)               # raw <mu|L_a|nu> = -i*L[a], (3,n,n)
re, im = calc.get_lao_dipole(atoms=mol, b_field=(0,0,0.06))  # raw LAO dipole integrals
```

In the GFN1 point-charge electric model the static optical-rotation `G`-tensor and
the MCD `d alpha/d B` (`get_mcd`) are identically zero by time reversal; the
length-gauge `get_lao_dipole` integrals are the route to their orbital-current
versions. The Cotton-Mouton `d^2 alpha/d B^2` is a nonzero observable.

### NMR nuclear magnetic shielding

The shielding tensor `sigma_ab = d^2 E / dB_a dm_b` of a chosen nucleus (closed-shell,
non-periodic) from analytic magnetic-dipole integrals built from scratch (Boys /
McMurchie-Davidson `1/r` and `1/r^3` operators) plus the McWeeny CP-SCC magnetic-field
response. The common gauge origin sits at the shielded nucleus.

```python
sigma = calc.get_nmr_shielding(nucleus=0, atoms=mol)   # 3x3 tensor in ppm
iso   = float(np.trace(sigma) / 3.0)                   # isotropic shielding (ppm)
```

The paramagnetic part comes from the CP-SCC response `dP/dB` (the same response used by
the magnetizability) contracted with the nuclear paramagnetic operator `(r_A x grad)_b /
r_A^3`; the diamagnetic part is the ground-state expectation of `[(r_O.r_A) delta_ab -
r_{A,a} r_{O,b}] / r_A^3`. The assembly is finite-difference-validated against
`d^2 E / dB dm` of the operator-injected magnetic SCC. Note that the GFN1 valence-only
basis omits core electrons, so absolute shieldings are not comparable to all-electron
references; use the values for within-method trends. `m1_basis_path=` selects the M1
kinetic-energy basis.

## Parameter derivatives of more observables + PyTorch (v0.1.4)

Beyond `parameter_derivatives` (`dE/dp`), the native calculator differentiates the
**dipole** and the **Hessian** with respect to parameters:

```python
dmu = calc.dipole_parameter_derivatives(numbers, positions, ["glob:ks", "elem:8:GAM"])
dh  = calc.hessian_parameter_derivatives(numbers, positions, ["glob:ks"])
print(dmu[0]["dipole_derivative"], len(dh[0]["hessian_derivative"]))
```

PyTorch interop keeps the GFN1 total energy differentiable w.r.t. a set of model
parameters **without making torch a dependency** (it is imported lazily):

```python
import torch
from gfn1_rs.torch_interop import parameter_energy_function

targets = ["glob:ks", "elem:1:GAM"]
energy_fn = parameter_energy_function(calc, numbers, positions, targets)
p = torch.tensor(calc.parameter_values(targets), dtype=torch.float64, requires_grad=True)
energy = energy_fn(p)     # Hartree, differentiable
energy.backward()
print(p.grad)             # dE/dp (the analytic/finite-difference parameter gradient)
```

The backward pass uses `Gfn1NativeCalculator.parameter_energy_and_gradient`, which
evaluates the energy and `dE/dp` at arbitrary parameter values without mutating the
parameter file.
