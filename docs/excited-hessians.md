# Analytic molecular excited-state Hessians

TD-GFN1 TDA/RPA and independent Rust sTDA-xTB expose second nuclear derivatives
at a single geometry. Neither production implementation displaces nuclei or
chooses a finite-difference step. State indices are zero based. These APIs are
for molecules with locally unchanged configuration selection. A single-root
Hessian requires an isolated root. TDA, RPA and ensemble sTDA support
the mean Hessian of a spectrally separated degenerate manifold.

The returned total surface is `F_GFN1 + Omega`. For sTDA this is explicitly a
composite surface: its dedicated VTB orbital generator does not define a fitted
ground-state total energy. Both contributions use the same requested electronic
temperature. Molecular TDA and RPA differentiate fractional occupation
matrices at fixed electron number, including degenerate orbitals. Ensemble
sTDA uses canonical fractional occupations of the final dedicated VTB
Hamiltonian. Its kernel and configuration rules are documented separately.
See [td.md](td.md) and [stda-xtb.md](stda-xtb.md). These requirements do not restrict the canonical
smearing support of ground-state FC3/FC4.

## CLI

`--tda-hessian`, `--rpa-hessian` and `--stda-hessian` print the total
Cartesian Hessian in Hartree/bohr². Select the zero-based root with
`--tda-state`, `--rpa-state` or `--stda-state` respectively. Use
`--tda-manifold-average`, `--rpa-manifold-average` or `--stda-manifold-average`
to select the mean energy surface. A one-root selection reports
`isolated_root` and returns its ordinary derivative. Python and ASE accept
`manifold_average=True`.

## Rust

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem,
              TdaOptions, solve_tda_hessian_analytic};
use gfn1_rs::stda_xtb::{NativeStdaOptions, solve_stda_xtb_hessian,
                       solve_stda_xtb_total_hessian};

let molecule = PeriodicSystem::from_xyz_str(
    "3\nwater\nO 0 0 0\nH .757 .586 0\nH -.757 .586 0\n", 0.0, false)?;
let parameters = Gfn1Parameters::builtin()?;
let electronic = ElectronicOptions {
    electronic_temperature: 300.0, ..Default::default()
};
let tda = solve_tda_hessian_analytic(
    &molecule, &parameters, &electronic, 0, TdaOptions::default())?;
let options = NativeStdaOptions {
    electronic_temperature: 300.0, energy_window_ev: 20.0,
    ..Default::default()
};
let stda = solve_stda_xtb_total_hessian(
    &molecule, &parameters, &electronic, 0, &options)?;
let excitation_only = solve_stda_xtb_hessian(&molecule, 0, &options)?;
println!("{} {}", tda.hessian[(0, 0)], stda.hessian[(0, 0)]);
```

`hessian`, `ground_hessian` and `excitation_hessian` are `3N` by `3N`
matrices in Hartree/bohr², ordered `(atom0.x, atom0.y, atom0.z, atom1.x, ...)`.
The excitation-only sTDA result contains `excitation_hessian` and its matching
gradient. Energies are Hartree. TD-GFN1 also offers
`solve_tda_hessian_analytic_with_budget(..., memory_budget_bytes)`; sTDA uses
`NativeStdaOptions.memory_budget_bytes`. The scoped [resource budget](resources.md)
can tighten either method's budget.

## Python native

```python
from gfn1_rs import Gfn1NativeCalculator

calc = Gfn1NativeCalculator(electronic_temperature=300.0,
                            threads=4, memory_mb=2048)
nums = [8, 1, 1]
pos = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]  # angstrom
tda = calc.tda_hessian(nums, pos, state=0, spin="singlet")
stda = calc.stda_xtb_hessian(nums, pos, state=0, energy_window_ev=20.0)
print(tda["total_energy_hartree"])
print(stda["hessian"])  # Hartree/bohr²
```

Both calls return the total `hessian` and separate `ground_hessian` and
`excitation_hessian`. Explicit `*_hessian_hartree_per_bohr2` keys preserve the
same native matrices. The `method` and `model` metadata identify the analytic
implementation and surface. `n_states` controls the usual requested spectrum;
the second-order response internally retains all CI eigenvectors, including
roots above the sTDA reporting window. This does not enlarge that window or
change the selected configuration space.

## ASE

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

atoms = Atoms("OH2", positions=[[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]])
atoms.calc = GFN1RSCalculator(electronic_temperature=300.0,
                             threads=4, memory_mb=2048)
tda = atoms.calc.get_tda_hessian(atoms=atoms, state=0)
stda = atoms.calc.get_stda_xtb_hessian(
    atoms=atoms, state=0, energy_window_ev=20.0)
print(stda["hessian"])  # eV/angstrom²
print(stda["hessian_hartree_per_bohr2"])  # native units
```

Unsuffixed ASE Hessians use eV/angstrom²; suffixed native arrays remain in
Hartree/bohr². Energies follow the same eV/native-key convention. These advanced
calls do not install an excited-state ASE potential automatically. For an
optimizer, supply energy and analytic forces from the matching excited-state
gradient API, as in [the ethylene calculations](ethylene-excited-optimization.md).

## Response equations and boundaries

For a normalized isolated eigenpair `A X_n = Omega_n X_n`, an analytic
directional derivative satisfies

```text
Omega_n'' = X_n^T A'' X_n
          + 2 sum_(m != n) (X_m^T A' X_n)^2 / (Omega_n - Omega_m).
```

The sum covers the complete selected CI matrix. TD-GFN1 includes first/second
SCC and generalized-MO response, metric normalization, transition-charge
response and the charge-dependent on-site kernel. Structural ML response in
this excited Hessian is explicitly rejected. Unsupported ground-model terms
are checked before the response is assembled.

sTDA differentiates both VTB updates, their canonical Fermi densities and
chemical potentials, Gaussian exponents/integrals, final generalized orbitals,
Lowdin charges, kernels, Kia shift and excluded-configuration PT denominators
through second order. The square-root overlap derivative obeys
`L L'' + L'' L = S'' - 2 L' L'`. Cross components follow exact polarization
of directional second derivatives. Intermediate VTB thermal degeneracy is
handled by confluent matrix-function divided differences.

An individual adiabatic Hessian is undefined at a CI degeneracy. A changing hard
orbital/configuration selection also needs a different model before derivatives
across that boundary are meaningful. sTDA reports active final-orbital degeneracy
explicitly. Its gradient/Hessian follow signed energy-ordered CI roots,
including through zero when the root is isolated. Filtering these roots and
relabelling the next positive root as S1 would silently change the surface.
The spectrum API retains positive absorption-line reporting and separately
exposes all signed `ci_eigenvalues_hartree`. Derivative results include
`nonpositive_ci_roots` and `lowest_ci_eigenvalue_hartree` as diagnostics.
A negative excitation indicates reference instability, even if the explicit
composite surface has a mathematical stationary point.

## Independent checks

`tests/td.rs` compares every Cartesian Hessian entry with differences of analytic
total gradients for water S1/S3 and ethylene S1 at 0/300 K: 486 entries, maximum
error `3.07e-11 Hartree/bohr²`. `tests/excited_hessian.rs` independently tests
sTDA singlet/triplet Hessians, selected/PT spaces, temperature,
translation, composite consistency and pre-allocation budget refusal. Python
tests check installed native/ASE composition, units and signed-root identity.
The eight sTDA conditions cover 1620 entries with maximum Richardson difference
`3.93e-9 Hartree/bohr²`. The wide 200 eV selection is separately diagnosed:
strong curvature and numerical-difference noise around `1e-6` prevent assigning
it the tighter 10 eV validation bound.

For gas-phase Ni(CO)4 at 300 K, the first three TDA roots are separated from
the next root by about 1.27e-8 Hartree. The audit fixes the reference root
range for both signs of every Cartesian displacement. Near-degenerate
orbital blocks are transported together to avoid dividing orbital-motion
terms by their small internal splittings. With SCC energy and charge
tolerances of 1e-14 Hartree and 3e-14 e, a Richardson ladder from 1.28e-7
through 1.6e-8 bohr gave a maximum difference of 0.187 Hartree/bohr² across
all 729 entries, or 2.61e-6 of the maximum absolute analytic matrix entry.
Using the central SCC charges as the initial guess for both signs gave
0.173 Hartree/bohr² and a ratio of 2.42e-6. Smaller steps increased the
difference. Set `GFN1_AUDIT_WARM_SCC=1` to use the common initial guess.
The following command runs the audit.

~~~sh
cargo test --profile reltest --lib nickel_carbonyl_tda_richardson -- --ignored --nocapture
~~~

Finite differences are used only in these verification programs.
