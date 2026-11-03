# Ethylene S1: unconstrained optimization with Rust analytic gradients

**TD-GFN1 reaches a nonplanar unconstrained local minimum at the tested resolution.**
After two rounds of negative-curvature restarts, both displacement signs meet
the force criterion. The positive branch has all twelve internal curvatures
positive. **The original signed sTDA branch has no verified smooth minimum.**
A separate explicit upper-envelope model has an audited electronic-crossing
stationary point; see [stda-pes.md](stda-pes.md).
Every primary optimization and every curvature check is free of geometric
constraints; no planar constraint is required.

Subsequent squared-gradient searches find force-zero points on the **original
signed sTDA surface**, without geometric constraints. The independently
audited point has maximum force `5.3000e-8 eV/angstrom`, a zero acute CH2-plane
angle and one negative internal Hessian mode. It is a saddle. Both signs of
its negative mode were followed by ordinary energy minimization; both lower
the energy and then stall near frontier degeneracy. Full 18-component
gradient/324-component Hessian checks and the other starts are recorded in
[stda-pes.md](stda-pes.md#original-sheet-force-zero-searches) and
[the measured original-sheet receipt](../tests/data/ethylene-stda-original-stationary.json).

## Conditions and surfaces

All runs use neutral ethylene, singlet `state=0`, **300 K electronic temperature**,
native Rust analytic forces, SCC energy/charge tolerances `1e-13` Hartree /
`1e-11`, and at most 1000 SCC iterations. The restarts use inverse BFGS with
Armijo backtracking, invalid-trial step halving, maximum atomic step
0.04 angstrom, at most 400 steps and a maximum atomic force convergence
threshold of `0.001 eV/angstrom`. sTDA uses a 20 eV selection window.

TD-GFN1 minimizes `F_GFN1 + Omega_TDA` with its legacy TDA kernel. The sTDA
surface is the explicitly defined composite `F_GFN1 + Omega_sTDA-xTB`; its
dedicated VTB orbital generator is not a fitted ground-state energy functional.
The two excited-state models are distinct. Energy-ordered `state=0` does not
imply equal diabatic state character between them. The closed-shell excitation
response domain is separate from finite-temperature ground-state FC3/FC4.

## TD-GFN1 result and dihedrals

| Route | Optimizer status | Steps | C-C (angstrom) | Excitation (eV) | Total energy (eV) | Max force (eV/angstrom) |
| --- | --- | --- | --- | --- | --- | --- |
| TD-GFN1 round 2, positive | converged | 38 | 1.38365630 | 3.6446746 | -166.8087255023 | 8.804e-04 |
| TD-GFN1 round 2, negative | converged | 37 | 1.38364286 | 3.6442945 | -166.8087248692 | 6.599e-04 |

Atom indices are zero based: C0, C1, then H2/H3 on C0 and H4/H5 on C1.

| TD-GFN1 branch | H2-C0-C1-H4 (degrees) | H3-C0-C1-H5 (degrees) | Acute CH2-plane angle (degrees) |
| --- | --- | --- | --- |
| positive | 115.34492 | 1.03193 | 37.960086 |
| negative | -1.05354 | -115.35043 | 37.974105 |

The CH2-plane value is the acute angle between C0-H2-H3 and C1-H4-H5.
The carbons can pyramidize, so that value is not a unique H-C-C-H torsion;
the two signed torsions are reported explicitly. The JSON retains all four
H-C-C-H combinations, coordinates and analytic gradients.

At the positive-branch final point, independent analytic-gradient differences
at `0.001` and `0.0005` bohr with Richardson extrapolation give all twelve
internal eigenvalues positive after removing six rigid translations/rotations:
`0.00034159753` to `1.09639171 Hartree/bohr^2`.
All 18 gradient components also agree with whole-energy differences to
`2.345e-11 Hartree/bohr`.
This supports a local minimum at the force and FD resolution used; it is not
a global-minimum search. The negative branch is force converged, but its
curvature was not independently audited.

## Why two negative-curvature restarts were needed

The first free relaxation from each sign reached a force-stationary TD point
near `-166.7367786847 eV`, but its lowest internal curvature remained
`-0.000725801043 Hartree/bohr^2`.
Both signs of that remaining negative mode were displaced again, with maximum
atomic displacement 0.05 angstrom, and optimized without constraints to the
round-two structures above. The earlier planar saddle is retained only as
seed provenance for the first negative-mode experiment, not as the physical
result or a required optimization constraint.

## sTDA-xTB: unconverged boundary diagnostics

| Historical sTDA route | Optimizer status | Steps | C-C (angstrom) | Excitation (eV) | Total energy (eV) | Max force (eV/angstrom) |
| --- | --- | --- | --- | --- | --- | --- |
| positive | domain_boundary | 28 | 1.31243622 | 4.173439e-09 | -169.4088070176 | 1.574e+00 |
| negative | domain_boundary | 29 | 1.31243622 | 6.071699e-10 | -169.4088070187 | 1.574e+00 |

| Historical sTDA branch | H2-C0-C1-H4 (degrees) | H3-C0-C1-H5 (degrees) | Acute CH2-plane angle (degrees) |
| --- | --- | --- | --- |
| positive | 79.16673 | 79.00506 | 79.086029 |
| negative | -79.16673 | -79.00506 | 79.086029 |

A separate earlier direct free attempt stopped with `line_search_stalled` at a CH2-plane angle of **79.351641 degrees** and maximum force `8.144e+00 eV/angstrom`. That angle is a last accepted geometry, not an optimized minimum.

The negative-mode runs also retain large forces, and their selected excitation
approaches zero. Their historical error strings are preserved in the JSON;
the occupation-guard messages belong to rejected sTDA trials. The accepted
frames above remain unconverged.

There was a separate sTDA state-selection defect: filtering out a nonpositive
lowest CI root could make index zero refer to the next positive root. Crossing
zero would then change the followed surface. The analytic response now follows
signed isolated CI branches without relabelling positive absorption roots.
A near-zero root or failed optimization alone does not establish a minimum.

These historical boundary points remain unconverged. The separate crossing
calculation uses `max(F_GFN1,F_GFN1+Omega)` and its nonsmooth stationarity
criterion; it does not assert that the original signed-sheet force vanished.
Its independent derivative and energy checks are in [stda-pes.md](stda-pes.md).

## Gradient verification

The repaired TD-GFN1 analytic gradients were additionally tested at six actual
trajectory geometries for S1/S2/S3: **324 Cartesian components**,
with independently rediagonalized whole-energy differences and two-step
Richardson extrapolation. Maximum error: **3.861e-11 Hartree/bohr**.
The full arrays and the final-point curvature audit are retained in
[ethylene-excited-optimization.json](../tests/data/ethylene-excited-optimization.json).
Its input SHA-256 values identify the measured logs. Source hashes labeled
`report_generation_source_sha256` describe the working tree when the report
was written; they are not claimed to identify the older trajectory binary.

## Reproduce

Install the native wheel and ASE/NumPy. A direct free run uses no constraints:

```sh
python tests/validation/ethylene_excited_optimization.py --optimizer domain_bfgs --steps 400 --output target/validation/v070/ethylene-free-rerun
```

To reproduce a measured round-two TD branch, export its stored initial seed
from the report and optimize it with the same free optimizer:

```python
import json
from pathlib import Path
report = json.loads(Path("tests/data/ethylene-excited-optimization.json").read_text())
seed = report["unconstrained"]["td_gfn1"]["positive"]["seed_metadata"]
Path("target/ethylene-td-seed.json").write_text(json.dumps(seed))
```

```sh
python tests/validation/ethylene_excited_optimization.py --seed-file target/ethylene-td-seed.json --model td_gfn1 --optimizer domain_bfgs --steps 400 --output target/validation/v070/ethylene-td-rerun
python tests/validation/verify_ethylene_optimized_points.py target/validation/v070/ethylene-td-rerun --model td_gfn1 --output target/validation/v070/ethylene-td-rerun-audit.json
```

The second negative-mode restart can also be regenerated directly from the
measured first unconstrained TD stationary point and its audit:

```sh
python tests/validation/ethylene_negative_mode_restart.py --reference target/validation/v070/ethylene-negative-modes/positive --audit target/validation/v070/ethylene-negative-modes/positive-audit.json --output target/validation/v070/ethylene-negative-modes-round2-rerun
```

The distributed receipt preserves measured optimizer status. The commands
above produce new optimization and curvature records; a curvature audit is
required before calling a new stationary point a minimum. No planar run is
a prerequisite. The one-off report formatter is not distributed.

API and model definitions: [td.md](td.md), [stda-xtb.md](stda-xtb.md).
