# Native sTDA PES and an explicitly defined crossing result

The original signed composite branch `F_GFN1 + Omega_sTDA` has **no verified
smooth force-zero minimum in these ethylene searches**. A separate **upper
envelope of two uncoupled composite surfaces** has an audited crossing
stationary point. These are different convergence statements.

All results use native Rust, neutral ethylene, singlet CI, **300 K**, unchanged
bundled parameters and no nuclear geometric constraints. No official binary
or refitting was used.

## Original-sheet force-zero searches

Unconstrained nuclear stationary-point searches now find force-zero points on
the original signed `F_GFN1 + Omega_sTDA` surface. They minimize the squared
gradient using the analytic nuclear Hessian, retaining signed curvatures in
the Newton direction. Only rigid translation/rotation directions are removed;
there is no planar restraint, gap constraint or replacement potential.

| Initial twist | Iterations | Final maximum force (eV/angstrom) | Negative internal modes |
| --- | ---: | ---: | ---: |
| 5 degrees | 4 | 5.3000e-8 | 1 |
| 30 degrees | 19 | 4.7112e-8 | 3 |
| 150 degrees | 28 | 6.0663e-9 | 3 |

These are **saddles, not minima**. The 5-degree search ends at a planar point
without imposing planarity: the acute CH2-plane angle is zero and signed
H–C–C–H torsions are approximately zero/180 degrees. Its energy is
`-165.78802073045 eV`; the negative internal curvature is
`-0.01710650 Hartree/bohr²`. Independent, fully reconverged differences check
all 18 gradient components and all 324 Hessian components at two steps with
Richardson extrapolation. Maximum errors are `1.760e-9 Hartree/bohr` and
`1.025e-8 Hartree/bohr²`. The selected CI configuration sets remain identical
over those stencils. The other two stationary points have analytic curvature
classifications; they are not claimed to have separate full-component audits.

Displacing the audited point in **both signs of its negative mode** and then
minimizing the original energy lowers it to about `-171.16 eV`, but both
searches stall near the frontier-degeneracy region with large ordinary forces.
The starts at 60/120 degrees also fail the force criterion. Thus the original
request for a force-zero point is achieved at a saddle; a smooth minimum on
that same model surface remains unverified. These measured results and the
complete independent audit are in
[ethylene-stda-original-stationary.json](../tests/data/ethylene-stda-original-stationary.json).

The squared-gradient idea is mathematically related to
[Hait and Head-Gordon's orbital SGM](https://arxiv.org/abs/1911.04709),
[DOI: 10.1021/acs.jctc.9b01127](https://doi.org/10.1021/acs.jctc.9b01127).
Here it is applied to **nuclear** stationary equations, rather than their
orbital optimization problem; a zero gradient does not certify a minimum.

## Direction-dependent orbital-degeneracy limit

The stalled geometry has an unshifted final HOMO-LUMO gap near `0.000233 eV`.
The overlap matrix is positive definite: minimum eigenvalue `5.14e-4`,
condition number `1.11e4`. Reported orbital gaps include the fitted +3.1 eV
virtual shift; subtract it to diagnose the unshifted frontier gap.

Transporting the two-frontier-orbital Hamiltonian into a fixed frame and
zeroing its diagonal difference/off-diagonal element locates a degeneracy
with gap about `4e-13 Hartree`. The signed CI energies approach different
values along two branching directions. Rows average the two displacement
signs; raw arrays retain both.

| Radius (angstrom) | Direction a (eV) | Direction b (eV) | Difference (eV) |
| --- | ---: | ---: | ---: |
| 0.0001 | -1.91090149 | -1.08948622 | 0.82141527 |
| 1e-06 | -2.05278308 | -1.08869490 | 0.96408818 |
| 1e-08 | -2.05420230 | -1.08870179 | 0.96550051 |

At a 200 eV window, all 120 allowed configurations are primary and no PT
selection correction remains. The directional difference still equals
`0.959977 eV` at `1e-8 angstrom`. Thus discrete configuration selection
alone is not the cause. The fixed occupied/virtual partition and
orbital-dependent model lose a unique limit at frontier degeneracy.
CP thresholds and optimizer tuning cannot make that limit unique.

The distributed Rust probe was independently rebuilt in the release profile
and repeated all 40 directional measurements. Its complete results and
source fingerprints are in
[stda-pes-release-reproduction.json](../tests/data/stda-pes-release-reproduction.json).

The numerical implementation now uses the primal CI orbital frame for
derivatives, follows signed isolated roots through zero without relabelling
positive absorption lines, and retains the actual temperature in both VTB
updates. Final diagnostic occupations do not truncate the fixed-partition CI.
These fixes do not remove the model's orbital-degeneracy limitation.

The [local-topology follow-up](stda-pes-local-topology.md) resolves the narrow
angular feature with gap-scaled sampling, compares analytic angular derivatives
and explains why the fixed-rank CI projector retains a singular limit even
though the intermediate densities use finite temperature.

## Explicit upper-envelope calculation

Define `F0=F_GFN1`, `F1=F0+Omega` and the additional analysis convention
`U=max(F0,F1)`. Near the first zero crossing, with the next CI root higher,
this is the upper level of an **uncoupled** reference-plus-CI model. Standard
gradient APIs still return `dF1/dR`; their surface has not been redefined.
No derivative coupling between the different reference models is supplied.

At `Omega=0`, `U` has a cusp. Its stationarity condition is

```text
g0 + lambda * dOmega/dR = 0,     0 < lambda < 1
H_L = H0 + lambda * H_Omega
```

The multiplier is solved, not fitted. All nuclear coordinates remain free.
The optimizer uses native analytic gradients/Hessians and follows negative
curvature in the electronic-crossing tangent space when necessary.

| Quantity | Measured value |
| --- | ---: |
| Iterations | 24 |
| Upper-envelope energy (eV) | -169.7236959981 |
| C-C distance (angstrom) | 1.38941447 |
| State gap (eV) | -4.145103e-09 |
| Multiplier | 0.4746444686 |
| KKT maximum force (eV/angstrom) | 1.005375e-07 |
| Individual F1 maximum force (eV/angstrom) | 2.364200e+00 |
| Smallest of 11 tangent curvatures (Hartree/bohr²) | 0.01486939 |
| H2-C0-C1-H4 torsion (degrees) | 41.307108 |
| H3-C0-C1-H5 torsion (degrees) | 113.230088 |
| Acute CH2-plane angle (degrees) | 66.451743 |

The individual-sheet force is **not zero**. The ordinary Cartesian Hessian
of the nonsmooth upper envelope is undefined at the crossing. The positive
curvatures belong to `H_L`, after removing six rigid directions and the
gap normal. Interior lambda, positive tangent curvature and independent
energy checks support a local minimum of this explicitly stated model.
They do not certify a physical conical intersection or global S1 minimum.

## Independent checks

Both sheets were independently rediagonalized at displaced geometries.
All 18 ground-gradient and 18 excitation-gradient components, and all 324
Lagrangian-Hessian entries, were checked with `1e-4`/`5e-5` bohr central
differences and Richardson extrapolation. Configuration sets are asserted
identical. A coarser stencil crossed a selection boundary and was not used
as a smooth derivative oracle.

| Check | Maximum difference |
| --- | ---: |
| Ground gradient (Hartree/bohr) | 2.143e-10 |
| Excitation gradient (Hartree/bohr) | 2.162e-10 |
| Lagrangian Hessian (Hartree/bohr²) | 8.867e-10 |

All 48 energy perturbations along the normal and eleven tangent modes,
using both signs and two steps, raised `U`. The smallest measured increase
was `1.028e-07 eV`.
The original signed branch has separately audited force-zero saddles, as
described above. This crossing calculation does not certify its smooth minimum.

## Reproduce

From the source root with the native wheel, ASE and NumPy:

```sh
python tests/validation/optimize_stda_crossing.py tests/data/ethylene-stda-crossing-seed.json --output target/validation/stda-crossing
python tests/validation/validate_stda_crossing.py target/validation/stda-crossing/result.json --output target/validation/stda-crossing/audit.json
python tests/validation/probe_stda_pes_degeneracy.py --output target/validation/stda-pes-limits.json
python tests/validation/search_stda_stationary.py --output target/validation/stda-original-stationary
python tests/validation/validate_stda_stationary.py target/validation/stda-original-stationary/twist-5.json --output target/validation/stda-original-stationary/audit-5.json --seeds target/validation/stda-original-negative-seeds
```

The degeneracy probe builds this repository's Rust validation target through
Cargo; it does not invoke an official xTB/sTDA program. Coordinates, complete
derivative arrays, directional limits and fingerprints are retained in
[stda-pes-optimization.json](../tests/data/stda-pes-optimization.json) and
[stda-pes-degeneracy.json](../tests/data/stda-pes-degeneracy.json).
