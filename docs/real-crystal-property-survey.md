# Property-by-property executions from sourced crystal structures

Every newly added property family has now been executed on a sourced crystal
or its explicitly model-relaxed descendant. The [complete measured receipt](../tests/data/real-crystal-property-survey.json)
contains all arrays, errors, conditions, original/relaxed geometries, timings
and actual native-extension hashes. A completed calculation is not a claim of
experimental accuracy or Brillouin-zone convergence.

## Structures and provenance

The [input fixture](../tests/data/real-crystal-survey-inputs.json) was extracted
from actual attributed COD CIF snapshots, not by relabelling the earlier ideal
prototypes. Reported symmetry operations were expanded directly and an exact
FCC primitive basis change reduced each conventional eight-atom cell to two
atoms, without idealization or geometry optimization during extraction.

| Crystal | COD record | Source | Declared source lattice a (angstrom) |
| --- | --- | --- | ---: |
| Diamond | [9008564](https://www.crystallography.net/cod/9008564.html) | R. W. G. Wyckoff, *Crystal Structures*, second edition, volume 1, pp. 7-83 (1963) | 3.56679 |
| Synthetic sphalerite ZnS | [9000107](https://www.crystallography.net/cod/9000107.html) | B. J. Skinner, *American Mineralogist* 46, 1399-1411 (1961), “Unit-cell edges of natural and synthetic sphalerites” | 5.4093 |
| Cubic beta SiC | [9008856](https://www.crystallography.net/cod/9008856.html) | Wyckoff, volume 1, pp. 85-237 (1963) | 4.348 |
| Halite NaCl | [9008678](https://www.crystallography.net/cod/9008678.html) | Wyckoff, volume 1, pp. 85-237 (1963) | 5.64056 |

ZnS is explicitly a reported experimental synthetic sample. The three handbook
records are published crystal-structure data; no unreported measurement
temperature or pressure is invented. COD's direct endpoint was unavailable
during this retrieval, so fixed Git commits of the COD tools/Avogadro mirrors
were used. Each exact mirror URL, original raw-CIF SHA and publication citation
is retained. Raw CIF headers remain unmodified in the validation cache; the
distributed input contains numerical fields and provenance.

The larger actual neutron Na2MoO4 structure, COD2240655, remains a separate
experimental counterexample in [the earlier crystal results](crystal-properties-results.md).

## Conditions and source-to-model structural changes

All electronic calculations use builtin scalar GFN1, no ML model, charge zero,
300 K, SCC thresholds 1e-13 Hartree/1e-11 shell charge, up to 1000 SCC iterations,
two workers and a 2 GB checked workspace budget. Primary runs use a Gamma-centred
2x2x2 SCF mesh and24 bohr AO cutoff. Joint atomic/cell relaxation uses 1e-7
Hartree/bohr force and 1e-7 Hartree/bohr cubed stress tolerances. Fixed-volume
EOS nodes independently relax the ions. Only the additional diamond311 run
uses the explicitly labelled3x1x1 electronic mesh.

| Execution | Source primitive V (angstrom cubed) | Model V (angstrom cubed) | Change | B Hill (GPa) | G Hill (GPa) | E Hill (GPa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diamond | 11.344167 | 11.948391 | +5.326% | 482.672 | 453.777 | 1036.510 |
| zns | 39.569741 | 51.426589 | +29.964% | 38.561 | 9.624 | 26.655 |
| sic | 20.549848 | 23.050336 | +12.168% | 200.565 | 162.945 | 384.665 |
| nacl | 44.864897 | 42.491772 | -5.289% | 22.282 | 13.612 | 33.926 |
| diamond-consistent-311 | 11.344167 | 15.600036 | +37.516% | 224.638 | 183.634 | 432.933 |

The relaxed cells, internal coordinates and full force/stress residuals remain
in the receipt. The 3x1x1 diamond descendant also changes cell shape; its volume
is not a refined experimental lattice constant. The large sampling-dependent
change is evidence that sparse-mesh thermodynamics must not be presented as
a converged material prediction.

## Executed property coverage

Each row points to actual successful executions; failed or unavailable cases
are retained separately. `diamond-consistent-311` is a different electronic
sampling condition from `diamond`, not an interchangeable retry.

| Property family | Successful recorded executions |
| --- | --- |
| energy forces stress pressure | diamond, zns, sic, nacl, diamond-consistent-311 |
| atomic cell relaxation equilibrium geometry | diamond, zns, sic, nacl, diamond-consistent-311 |
| equation of state bulk modulus | diamond, zns, sic, nacl, diamond-consistent-311 |
| analytic elasticity internal strain | diamond, zns, sic, nacl, diamond-consistent-311 |
| compliance moduli poisson sound speeds | diamond, zns, sic, nacl, diamond-consistent-311 |
| directional moduli christoffel speeds | diamond, zns, sic, nacl, diamond-consistent-311 |
| gamma modes and finite q phonons | diamond, zns, sic, nacl, diamond-consistent-311 |
| phonon dos | diamond, zns, sic, nacl, diamond-consistent-311 |
| zero point free energy entropy cv | sic, diamond-consistent-311 |
| berry polarization and born charges | diamond, sic, nacl, diamond-consistent-311 |
| proper clamped piezoelectricity | diamond, sic, nacl, diamond-consistent-311 |
| relaxed ion piezoelectricity | diamond, sic, nacl, diamond-consistent-311 |
| piezoelectric strain d | diamond, sic, nacl, diamond-consistent-311 |
| electronic dielectric | diamond, sic, nacl, diamond-consistent-311 |
| static dielectric fixed strain and free stress | diamond, sic, nacl, diamond-consistent-311 |
| infrared mode charges oscillator tensors | diamond, sic, nacl, diamond-consistent-311 |
| directional lo to | diamond, sic, nacl, diamond-consistent-311 |
| electronic bands and sampled gap | diamond, zns, sic, diamond-consistent-311 |
| electronic dos | diamond, zns, sic, diamond-consistent-311 |
| quasiharmonic volume thermal expansion cp | diamond-full-222-internal-relaxed-qha, diamond-internal-relaxed-qha-tol1e-7 |

The finite-q mesh was initially2x1x1 (Gamma of a four-atom supercell). It folds
a primitive2x1x1 electronic mesh, which differs from the primary2x2x2 reference.
That difference is explicit in corrected condition metadata and the original
driver fields are preserved. The matching311 and full222 QHA runs independently
relax each fixed-volume node on exactly the corresponding primitive mesh and
take static energies from the same supercell SCC used for its phonons.

## A nonzero polar example: cubic SiC

For the model-relaxed COD9008856 descendant on2x2x2 sampling, the full Born,
proper piezoelectric and dielectric calculations return nonzero response.
The shear entries are approximately `e14=0.346535 C/m²` and
`d14=1.710865 pC/N`. Static relative dielectric diagonals are approximately
8.05869 at fixed strain and 8.12577 at free stress. These values are coarse-mesh
model outputs, not validated experimental constants. Nonzero nominally
cubic-forbidden entries reveal finite Berry-mesh discretization and are not
silently projected away.

All mode effective charges and oscillator tensors are retained, together with
explicit directional LO–TO matrices/frequencies along100,110 and111. The three
directions are actual computations, not inserted empirical Born/dielectric
parameters. The sampled model orbital gap on a uniform4x4x4 evaluation grid is
about 1.75130 eV; that evaluation uses the separately converged2x2x2 SCC density
and is not a quasiparticle or optical gap.

## Harmonic and quasi-harmonic thermodynamics

Phonon DOS retains signed imaginary frequencies. Harmonic ZPE/free energy,
entropy and Cv reject genuinely imaginary modes; they never replace them by
absolute frequencies. Lattice temperatures0,100,300 and600 K are distinct from
the fixed300 K electronic temperature.

The successful internally relaxed QHA executions give at lattice300 K:

| Execution | V equilibrium (bohr cubed) | B (GPa) | alpha V (1/K) | Cv (J/mol primitive cells/K) | Cp (J/mol primitive cells/K) |
| --- | ---: | ---: | ---: | ---: | ---: |
| diamond-full-222-internal-relaxed-qha | 81.340334 | 473.487 | 8.59677e-07 | 11.548884 | 11.549646 |
| diamond-internal-relaxed-qha-tol1e-7 | 106.768193 | 224.036 | 5.52654e-07 | 14.743732 | 14.743928 |

The primary full 2x2x2 result uses a 16-atom supercell and all eight commensurate
q points. At all five volume nodes the cubic special-position atoms already
have force norms about 2.1e-13 to 3.1e-13 Hartree/bohr; zero ionic iterations are
needed. There is no evidence for large physical internal relaxation in cubic
diamond. Every node has zero imaginary modes below minus 1e-3cm^-1; the tiny
acoustic residuals are retained. The 3x1x1 comparisons instead have a distorted
cell and sparse anisotropic electronic sampling. Their alpha sign and small
differences are workflow/sampling/shape/volume-fit/force-tolerance sensitivity,
not an experimental claim about internal relaxation or thermal expansion.

These are fixed-shape isotropic volume fits over ratios 0.88,0.94,1,1.06,1.12,
with internal ions relaxed at every node. They are not anisotropic thermal
expansion tensors. The 3x1x1 q mesh is anisotropic and coarse;2x2x2 is a 3D
commensurate mesh but remains coarse. Neither is labelled a converged 3D
thermodynamic prediction. Volume-fit RMS errors and all node frequencies are
retained; the frozen-internal-coordinate311 calculation is kept as a distinct
approximation rather than mixed with the internally relaxed data.

A separately requested stricter 1e-8 force test stalled at the central node
with an atomic gradient norm around 1.2e-8 Hartree/bohr, near energy resolution.
It is not marked converged. The successful internal-relaxation run uses the
same explicitly recorded 1e-7 force target as the main survey; electronic SCC
temperature and thresholds are unchanged.

## Failures and explicit numerical comparisons

- ZnS reaches a force/stress-stationary model cell but has a metallic manifold.
  Its Berry/Born/piezo route is rejected at 300 K, retaining that physical guard.
- The original diamond2x2x2-relaxed reference has imaginary finite-q modes
  under the different2x1x1 electronic folding; its harmonic/QHA failures remain.
- SiC's initial frozen-coordinate2x1x1 QHA has no stable fitted minimum inside
  that volume window. It is not called a converged equilibrium.
- NaCl has a non-positive overlap on off-mesh band points at 24 bohr cutoff.
  At the exact same geometry,300 K,2x2x2 SCC grid and4x4x4 evaluation grid,
 30,40 and50bohr complete. This explicit cutoff experiment does not overwrite
  the failed24 bohr band/DOS run. Its adjacent-cutoff differences are:

| AO cutoffs (bohr) | Max all-band difference (eV) | Max occupied-band difference (eV) | Gap difference (eV) |
| --- | ---: | ---: | ---: |
| 30 to 40 | 1262.47 | 0.00125666 | 0.000854902 |
| 40 to 50 | 52.6647 | 2.61188e-05 | 7.39011e-06 |

The revised cutoff changes only that diagnostic's AO image sum; no temperature,
electronic sampling, geometry or eigenvalue clipping is used to conceal the
failure. Restoring positive overlap does **not** establish convergence of the
full virtual manifold: even 40 to 50 bohr changes the highest bands by about
52.7 eV. The occupied bands and sampled gap converge much more closely in that
comparison. NaCl's full virtual spectrum and broad-range electronic DOS are
therefore not quantitative predictions. The primary diamond/SiC/ZnS DOS grids
instead integrate to their expected 16/26/26 states per primitive cell.
Converge cutoffs, k/q sampling and volume windows before using these
outputs as quantitative material properties.

## Reproduce

With an installed native wheel, ASE and NumPy, run from the source root:

```sh
python tests/validation/real_crystal_properties.py --case sic-cod9008856 --output target/validation/real-survey/sic-raw.json --stage all
python tests/validation/real_crystal_properties.py --case diamond-cod9008564 --output target/validation/real-survey/diamond-consistent-311-raw.json --stage all --kmesh 3 1 1 --repetitions 3 1 1
python tests/validation/real_crystal_relaxed_qha.py target/validation/real-survey/diamond-consistent-311-raw.json --force-tolerance 1e-7 --output target/validation/real-survey/diamond-internal-relaxed-qha-tol1e-7.json
python tests/validation/real_crystal_cutoff_convergence.py target/validation/real-survey/nacl-raw.json --output target/validation/real-survey/nacl-cutoff-comparison.json
```

Use the same main command for `diamond-cod9008564`, `zns-cod9000107` and
`nacl-cod9008678`. For full222 internally relaxed QHA, pass the primary
`diamond-raw.json` to the QHA driver. The attributed numerical inputs, source
URLs and original CIF hashes are distributed; one-off source-capture and report
formatting helpers are excluded. These calculations read the retained inputs.

API definitions and units: [mechanics](mechanical-properties.md),
[electromechanics](electromechanical.md), [spectra/thermodynamics](phonon-band-properties.md),
and [native Rust/pip crystal CLI](crystal-cli.md).

## Same-crystal CLI cross-check

The supplied calculation input uses the explicitly model-relaxed COD9008856
SiC descendant, with exactly the survey's 300 K, 2x2x2 mesh, 24 bohr cutoff,
isotope masses and SCC thresholds:

```sh
gfn1-rs properties --request examples/crystal-sic-cod9008856-relaxed-request.json --output sic-response.json --threads 2 --memory-mb 2048
python -m pytest tests/python/test_crystal_cli.py -k sourced_sic
```

The installed CLI reproduces every clamped-elastic/Born/proper-piezo,
relaxed e/d and electronic/static dielectric component from the native survey.
Largest tensor differences are below 2.5e-14 in their native units. The LO-TO
dynamical matrix agrees within 7e-17; optical frequencies agree within
1.4e-12 cm^-1. Raw acoustic frequencies near zero retain their signed numerical
square-root amplification and are not silently replaced with zero.
The [cross-check receipt](../tests/data/real-crystal-cli-validation.json)
retains both measured binary identities and exact input/result hashes.
