# Rust TDA and Rust sTDA-xTB at the same 300 K

All spectra and diagnostics here were computed by native Rust, with the same
Cartesian coordinates, atom order, neutral charge, singlet channel and **300 K**
electronic temperature. Both VTB updates and the final sTDA occupation diagnostic
use that requested temperature. No official executable produced either column.

The two models retain their own basis, Hamiltonian and response kernel. This is
a comparison of distinct models, not a test with an identical Hamiltonian.

TDA strengths are shown for the unchanged legacy Mulliken operator and the
optional AO length-gauge operator. Both use identical TDA energies/amplitudes.
The AO operator retains intra-atomic dipoles that monopoles omit. sTDA uses its
own basis and AO length-gauge dipoles. Rows are independently energy-sorted;
an equal row index does not prove that the states have the same character.

Energies are eV and strengths are dimensionless. Printed zeros below 1e-10
retain their full values in the accompanying JSON. Geometries are explicit
representative inputs, not optimized structures or experimental benchmarks.

## Water

| Energy order | TD-GFN1 TDA energy | TDA f (Mulliken) | TDA f (AO length) | Rust sTDA-xTB energy | sTDA f (AO length) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.330132 | 0.000000 | 0.000636 | 7.513431 | 0.042099 |
| 2 | 10.684755 | 0.000410 | 0.012812 | 8.957503 | 0.000000 |
| 3 | 12.265107 | 0.000000 | 0.000000 | 9.606034 | 0.003448 |
| 4 | 12.505583 | 0.018472 | 0.057928 | 10.014359 | 0.006349 |
| 5 | 13.584324 | 0.000331 | 0.004671 | 10.791272 | 0.108034 |

```python
numbers = [8, 1, 1]
positions = [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]
```

## Ethylene

| Energy order | TD-GFN1 TDA energy | TDA f (Mulliken) | TDA f (AO length) | Rust sTDA-xTB energy | sTDA f (AO length) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.386288 | 0.000000 | 0.000000 | 7.031339 | 0.000051 |
| 2 | 7.275667 | 0.000000 | 0.000000 | 8.004215 | 0.000000 |
| 3 | 8.447745 | 0.000000 | 0.000000 | 8.055571 | 0.525019 |
| 4 | 9.198932 | 0.000000 | 0.000000 | 8.186528 | 0.000000 |
| 5 | 9.235380 | 0.000000 | 0.006089 | 8.193730 | 0.000000 |

```python
numbers = [6, 6, 1, 1, 1, 1]
positions = [[-0.6695, 0, 0], [0.6695, 0, 0], [-1.232, -0.928, 0], [-1.232, 0.928, 0], [1.232, -0.928, 0], [1.232, 0.928, 0]]
```

## Formaldehyde

| Energy order | TD-GFN1 TDA energy | TDA f (Mulliken) | TDA f (AO length) | Rust sTDA-xTB energy | sTDA f (AO length) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.104306 | 0.000000 | 0.000000 | 4.292654 | 0.000000 |
| 2 | 7.231006 | 0.000000 | 0.000322 | 6.481243 | 0.031555 |
| 3 | 8.534792 | 0.000000 | 0.000000 | 7.750876 | 0.068308 |
| 4 | 8.599018 | 0.000540 | 0.007864 | 8.127300 | 0.000000 |
| 5 | 9.663098 | 0.000000 | 0.171959 | 8.417355 | 0.029571 |

```python
numbers = [6, 8, 1, 1]
positions = [[0, 0, 0], [1.208, 0, 0], [-0.587, 0.939, 0], [-0.587, -0.939, 0]]
```

## Cause of the differences

The first-root CI expectation values decompose the excitation as
`Omega = sum_ia |X_ia|^2 (epsilon_a-epsilon_i) + response/selection`:

| Molecule | TDA AOs / sTDA AOs | TDA gap | TDA response | sTDA shifted gap | sTDA response + selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| water | 8 / 18 | 9.330132 | -0.000000 | 14.347342 | -6.833911 |
| ethylene | 16 / 36 | 6.386288 | 0.000000 | 12.217479 | -5.186140 |
| formaldehyde | 12 / 26 | 4.104306 | 0.000000 | 10.348364 | -6.055710 |

The sTDA gap includes its fitted **+3.1 eV virtual-level shift**. Its response
contains simplified Coulomb/exchange terms, Kia shift and excluded-configuration
perturbative correction. TD-GFN1 retains its existing monopole SCC kernel.
A vanishing transition charge leaves no monopole response correction for these
first TDA roots. Changing the dipole observable improves their strengths but
does not change that excitation kernel. Dedicated orbitals, extra diffuse basis
functions and different response terms account for the large model differences.

Temperature effects were checked separately by recomputing both models at 0 K:

| Molecule | Max TDA 0→300 K change (eV) | Max sTDA change (eV) | Max TDA occupation deviation | Max sTDA occupation deviation |
| --- | ---: | ---: | ---: | ---: |
| water | 7.553e-14 | 4.032e-13 | 4.441e-16 | 8.882e-16 |
| ethylene | 4.229e-14 | 2.464e-12 | 1.332e-15 | 1.332e-15 |
| formaldehyde | 6.948e-14 | 1.098e-12 | 8.882e-16 | 1.332e-15 |

These three molecules are gapped at 300 K: occupation deviations are roundoff
scale and the temperature changes are negligible. This does not assert the
same result for a small-gap system with fractional occupations; the current
closed-shell excitation APIs explicitly reject unsupported fractional fillings.

## Reproduction

```sh
cargo test --profile reltest --test tda_stda_analysis > comparison-raw.json
```

The equivalent installed Python calls are:

```python
from gfn1_rs import Gfn1NativeCalculator
calc = Gfn1NativeCalculator(electronic_temperature=300., energy_tolerance=1e-13,
                           charge_tolerance=1e-11, max_scc=1000)
tda = calc.tda(numbers, positions, n_states=10)
tda_ao = calc.tda(numbers, positions, n_states=10, transition_dipole='ao_length')
stda = calc.stda_xtb(numbers, positions, n_states=100, energy_window_ev=20.)
assert tda['electronic_temperature_kelvin'] == stda['electronic_temperature_kelvin'] == 300.
```

Full numeric results, coordinates, temperature diagnostics and decomposition
are in `tests/data/tda-stda-native-comparison.json`. The separate independent
official-code fixtures compare all 105 singlet roots below 20 eV. Reference
energy rounding is 0.0001 eV and the maximum difference is 0.000059 eV. Near
degenerate roots are also checked by summed subspace strengths. Official programs
are offline validation tools and are not available through the runtime API.

References and parameter provenance: [stda-xtb.md](stda-xtb.md). TD-GFN1 working
equations and compatibility conventions: [td.md](td.md).
