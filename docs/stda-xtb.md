# TDA and sTDA-xTB

These are distinct models. The existing `tda` implementation is TD-GFN1 in the
Tamm–Dancoff approximation, with GFN1 orbitals, shell transition charges and the
GFN1 SCC kernel. sTDA-xTB is the model of Grimme and Bannwarth, with a dedicated
ground-state procedure and two separately fitted Hamiltonian/basis parameter
sets followed by the simplified excitation kernel. Applying that kernel to
ordinary GFN1 orbitals would be a different model and must be named accordingly.

## Primary sources and implementation references

- S. Grimme and C. Bannwarth, *Ultra-fast computation of electronic spectra for
  large systems by tight-binding based simplified Tamm-Dancoff approximation
  (sTDA-xTB)*, J. Chem. Phys. **145**, 054103 (2016),
  [doi:10.1063/1.4959605](https://doi.org/10.1063/1.4959605).
- S. Grimme, *A simplified Tamm-Dancoff density functional approach for the
  electronic excitation spectra of very large molecules*, J. Chem. Phys.
  **138**, 244104 (2013),
  [doi:10.1063/1.4811331](https://doi.org/10.1063/1.4811331).
- Official ground-state implementation:
  [grimme-lab/xtb4stda](https://github.com/grimme-lab/xtb4stda), reference commit
  `238db6c7df81d5a2ed829f4f682f00aaa87e137d`.
  Relevant files are `.param_stda1.xtb`, `.param_stda2.xtb`, `src/main.f`,
  `src/xtb.f`, `src/xbasis.f`, `src/iniq.f` and `src/cm5.f`.
- Official excitation implementation:
  [grimme-lab/stda](https://github.com/grimme-lab/stda) (currently redirected to
  `Theoretical-Chemistry-Group-UCLouvain/std2`), reference commit
  `8d50f9f2a4b66b534e19bb8a99524be6b7153a47`.
  Relevant files are `main.f`, `stda.f`, `readxtb.f`, and `std2_manual.pdf`.
  The `-xtb` defaults and the excitation-space selection/correction rules belong
  to this source version and must be identified separately from fitted
  Hamiltonian parameters.

The referenced repositories were actually downloaded and retained in the
validation workspace. Their LGPL-3.0-or-later license notices and parameter-file
hashes must accompany any incorporated source or data. The publisher PDF of the
2016 paper is presently unavailable to this runner (HTTP 403); Europe PMC lists
PMID 27497535 without a PMCID. This is recorded as an unavailable primary text,
not as a downloaded paper. The public reference implementations remain the
source for implementation review while the publisher text is sought.

## Native Rust implementation and API

`calc.stda_xtb` and `gfn1-rs --stda-xtb` perform the complete calculation in Rust.
Rust, Python native and ASE expose this independent implementation only; the
package contains no official-program driver API or external-executable option.
They require no `xtb4stda`, `std2`, Fortran runtime or libcint installation.
Rust builds the dedicated VTB/xTB orbitals, Gaussian overlap/kinetic/dipole
integrals, Lowdin transition charges, simplified response matrix, perturbative
configuration selection and correction, and solves the excitation eigenproblem.
The fitted parameters and universal tables are bundled with source provenance
and license notices; they are not downloaded at runtime.

```python
from gfn1_rs import Gfn1NativeCalculator
calc = Gfn1NativeCalculator(charge=0)
spectrum = calc.stda_xtb(
    [8, 1, 1], [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]],
    n_states=10, energy_window_ev=20.0, triplet=False,
)
assert spectrum["implementation"] == "native-rust"
print(spectrum["excitation_energies_ev"], spectrum["oscillator_strengths"])
print(spectrum["parameter_sha256"])
```

```sh
gfn1-rs water.xyz --stda-xtb --stda-window-ev 20 --n-states 10
```

Rust exposes `stda_xtb::solve_stda_xtb` with `NativeStdaOptions` and returns
`NativeStdaResult`. Energies and response contractions use double precision.
The result includes excitation energies in Hartree and eV, length-gauge
oscillator strengths, transition dipoles in atomic units, selected MO-pair
configurations and CI amplitudes. `max_ao`, `max_configurations` and
`memory_budget_bytes` bound allocations.

The two parameter SHA-256 values identify the original, unmodified
`.param_stda1.xtb` and `.param_stda2.xtb` bytes. The source commits identify the
orbital-generator parameter provenance and response equations respectively.
No GFN1 parameters are substituted for missing sTDA-xTB parameters.

This dedicated model uses its own parameterization. Calculator electronic
temperature is applied to both VTB updates and the final canonical occupations;
300 K is the default. `NativeStdaOptions.electronic_temperature` controls this
in Rust; Python and ASE use the calculator's `electronic_temperature`, and the
CLI accepts `--electronic-temperature`. Dispersion, SOC, solvation and other GFN1 model modifiers
do not alter the dedicated excitation model. Ground-state charge
is passed explicitly. The current implementation supports closed-shell
molecules; periodic sTDA-xTB is not implemented. Existing `tda` calls continue
to select TD-GFN1. See [the same-molecule Rust comparison](tda-stda-comparison.md)
for actual TD-GFN1 TDA and native sTDA-xTB results.

## Analytic nuclear gradients

The Rust implementation differentiates its entire dedicated orbital generator
and response problem. It does not call a reference executable or use nuclear
finite differences. State indices are zero based, as for TD-GFN1. Derivative
calls use signed, energy-ordered CI eigenvalues before absorption filtering,
so a root is not relabelled when it crosses zero. `stda_xtb` retains positive
absorption lines and additionally returns all `ci_eigenvalues_hartree`.

```python
gradient = calc.stda_xtb_gradient(
    [8, 1, 1], [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]],
    state=0, n_states=5, energy_window_ev=20.0)
print(gradient["excitation_energy_hartree"])
print(gradient["excitation_gradient"])  # dOmega/dR, Hartree/bohr
print(gradient["total_energy_hartree"], gradient["gradient"], gradient["forces"])
```

`excitation_gradient` differentiates the sTDA excitation energy alone. To match
the existing TD-GFN1 gradient API, `gradient`, `forces` and `total_energy_hartree`
refer to the explicitly named composite surface **GFN1 free energy + sTDA-xTB
excitation energy**. The dedicated VTB orbital eigenvalue sum is not a fitted
ground-state total energy. The result labels this choice in `model` and
`ground_reference`; calculator modifiers apply to the GFN1 reference term.

Rust exposes both surfaces separately:

```rust
use gfn1_rs::stda_xtb::{NativeStdaOptions, solve_stda_xtb_gradient,
                       solve_stda_xtb_total_gradient};
let options = NativeStdaOptions { energy_window_ev: 20.0, ..Default::default() };
let excitation = solve_stda_xtb_gradient(&molecule, 0, &options)?;
let total = solve_stda_xtb_total_gradient(
    &molecule, &gfn1_parameters, &electronic_options, 0, &options)?;
```

ASE exposes `calculator.get_stda_xtb(atoms=atoms, ...)` and
`calculator.get_stda_xtb_gradient(atoms=atoms, state=0, ...)`. In the latter,
unsuffixed energies are eV and unsuffixed gradients/forces are eV/angstrom;
`excitation_gradient_hartree_per_bohr` preserves the native excitation gradient.

### Equations and scope

For a normalized isolated CI root, `A X = Omega X`,
`Omega' = X^T A' X`. Scalar analytic forward differentiation propagates CN,
initial EN charges, CM5 corrections, charge-dependent Gaussian exponents,
contraction normalization/orthogonalization, and all overlap/kinetic integrals.
The two VTB updates differentiate their canonical Fermi density matrices
at fixed electron number, including the overlap metric and chemical potential.
Confluent divided differences remain finite at intermediate VTB degeneracies
at positive temperature. The original fixed-partition sTDA CI uses a closed-shell
occupied/virtual partition from the even electron count. Canonical occupations
of the final extended-basis Hamiltonian are diagnostics, not another charge
update or thermal weights in that CI matrix. Their fractional tails therefore
do not invalidate its nuclear derivative. Active final-MO degeneracy still
requires a specified orbital gauge and is rejected. This is the fixed-partition
sTDA model, not a general finite-temperature ensemble excitation theory.
The finite-temperature ensemble below also weights the CI by the final canonical occupations.
The native charge potential is applied to every AO pair. The upstream
component-wise `abs(S_mu_nu)>1e-6` screen is omitted: it changes with orbital
orientation and causes discontinuous derivatives near a zero overlap component.
Removing it preserves the independently checked spectra within their reported
precision and makes the nuclear derivative smooth at those components.

For the final generalized eigenproblem, define `F = C^T H' C` and
`M = C^T S' C`. Then `epsilon_i' = F_ii - epsilon_i M_ii`,
`C' = C U`, `U_ii = -M_ii/2`, and
`U_ij = (F_ij - epsilon_j M_ij)/(epsilon_j-epsilon_i)` for unequal eigenvalues.
For Lowdin charges, differentiate `L = S^(1/2)` via the Sylvester equation
`L L' + L' L = S'`, then differentiate `L C`, both distance kernels and the
Kia diagonal shift. Excluded secondary configurations contribute the derivative
of the full second-order primary-diagonal correction, including denominators.
The CI amplitude derivative cancels in the eigenvalue contraction.

The derivative is local to fixed orbital/configuration selection. A hard
selection boundary or a degenerate individual adiabatic CI state does not
define a unique single-state derivative. The orbital-dependent Kia/selection
model also needs a specified gauge for an exactly degenerate active final-MO
block; the API reports that case explicitly. This restriction does not apply
to the canonical intermediate VTB density derivative or finite-temperature
GFN1 FC3/FC4 matrix-function response.

The regression rotates three molecules so all Cartesian axes participate, then
checks all 78 nuclear components across singlet and triplet calculations against
two central-difference step sizes with Richardson extrapolation. The largest
excitation-gradient difference is 1.4e-9 Hartree/bohr. Net translation and
rotation derivatives vanish to about 1e-12 or better. See
`src/stda_xtb/gradient.rs` and `src/stda_xtb/ground/differential.rs` for the
derivation's implementation and executable checks.

## Independent validation

The signed composite PES has a measured direction-dependent limit at final
HOMO/LUMO degeneracy. The ordinary signed-sheet ethylene minimum remains
unverified. A separate explicit upper-envelope crossing calculation is
documented, with full derivative checks, in [stda-pes.md](stda-pes.md).

Analytic second derivatives, their Rust/native/ASE usage and the complete CI
eigenvector response are documented in [excited-hessians.md](excited-hessians.md).
Gradient and Hessian calls retain a nondegenerate signed CI branch across zero
rather than relabelling the next positive excitation as S1. Results expose
`nonpositive_ci_roots` and `lowest_ci_eigenvalue_hartree` for this diagnosis.
A negative excitation signals an instability of this approximate reference;
it is not a positive absorption line. Mathematical convergence of the explicit
composite surface does not establish a physically stable excited state.

The official Fortran programs were downloaded and built separately to generate
independent regression fixtures. They are validation tools, not dependencies of
the native calculation. Tests compare the Rust orbital energies, AO density,
AO dipole integrals and all 105 singlet excitations for water, ethylene and
formaldehyde below 20 eV. Reference files record the upstream revisions and
parameter hashes. The native implementation does not read their wavefunctions
or launch them when calculating a spectrum.


## Finite-temperature ensemble response in v0.7.2

At positive electronic temperature, fractional final occupations or
degenerate active final VTB orbitals select the ensemble CI automatically. The CLI
reports `stda_response_model thermal_ensemble`. Use `--stda-thermal` to
select it explicitly or `--stda-legacy` to select the original fixed-partition
model. The Rust option `thermal_response: Option<bool>` and the Python/ASE
keyword `thermal_response` use the same selection rules.

The ensemble takes the final unshifted VTB Fock spectrum and canonical
Fermi occupations. It uses complete interband products of orbital-energy
blocks of width `1e-6 Hartree` inside the active gap window. Each kernel
leg carries `sqrt((f_i-f_a)/2)`. The fitted virtual level shift is weighted
by `(f_i-f_a)/2`. The diagonal Kia shift uses the mean exchange contraction
over each complete block product. These rules make the model invariant to
rotations within exactly degenerate VTB blocks. Perturbative selection and
its diagonal corrections belong to the original fixed-partition model.
The ensemble uses its complete active configuration space and applies the
same memory and configuration limits. The original fit is described by
[Grimme and Bannwarth](https://doi.org/10.1063/1.4959605).

Ensemble gradients and Hessians include first and second canonical occupation
response, orbital motion, the kernel, and the block Kia shift. The gradient
and Hessian select signed CI roots using the same energy ordering. With
`--stda-manifold-average` they return the mean surface, with `isolated_root`
for a one-root selection. These derivatives are local to a fixed band and
configuration partition.

```sh
gfn1-rs examples/ni_co4.xyz --stda-grad --stda-manifold-average
gfn1-rs examples/ni_co4.xyz --stda-hessian --stda-manifold-average
```
