# Crystal relaxation, equations of state and analytic elasticity

The same operations are exposed by the shared Rust/pip CLI; see
[crystal-cli.md](crystal-cli.md) for complete JSON requests and the
Rust/native/ASE/CLI mapping.

The periodic scalar GFN1 and ML-xTB Hamiltonians share these Rust calculations.
Python native and ASE call the same implementation. All force, stress and
elastic-response production routes are analytic, including learned structural
level heads, correction heads, descriptor images and SCC/occupation response.
An EOS samples volumes and fits an explicitly stated thermodynamic model;
it is not a finite difference implementation of the elastic tensor.

## Definitions and conditions

The coordinate map is `R'=F(R+u)`, `A'=FA`. Engineering Voigt order is
`[xx, yy, zz, yz, xz, xy]`; a shear entry is twice the Cartesian symmetric
strain. The output is the material tensor
`C = (1/V0) d²F_electronic/dE_Green²`, with the geometric reference-stress
term removed from the affine energy Hessian. This is also `dS/dE_Green` for
second Piola-Kirchhoff stress. At zero reference stress it agrees with the
usual infinitesimal stress-strain stiffness. Definitions differ at nonzero
stress; the reference stress is returned explicitly. See the
[ABINIT elasticity notes](https://docs.abinit.org/theory/documents/elasticity-oganov.pdf).

The analytic joint Hessian provides nuclear curvature `K`, nuclear/strain
coupling `Lambda` and the clamped-ion material tensor. The internal response is

```text
du/deta = -K_internal^-1 Lambda
C_relaxed = C_clamped - Lambda^T K_internal^-1 Lambda / V0
```

Exactly three rigid translations are projected out; rotations are not removed
from a periodic crystal. Relaxed-ion tensors require stationary internal atoms
and positive internal curvature. Otherwise `relaxed_stiffness` and
`internal_displacement_strain` are absent/null, and `relaxed_response_error`
records the reason. Negative curvature is not clipped or converted into a
stable static response.

Bulk calls require a nonempty 3D periodic scalar system at zero external field.
SCF thresholds are tightened to at most `1e-12` Hartree and `1e-10` shell charge;
stricter supplied values are preserved and the actual thresholds are returned.
Electronic temperature is preserved, including metallic/Fermi-smeared response.
The examples use **300 K electronic temperature**; this is not a vibrational
lattice temperature. Vibrational free energies and thermal expansion are in
[phonon-band-properties.md](phonon-band-properties.md).

Relaxation minimizes `F_electronic + P_target V`, with positive target pressure
compressive. Convergence requires the requested atomic-force and, for a moving
cell, stress residuals. A small step, iteration limit or failed line search is
never reported as convergence. Fractional coordinates deform consistently with
the cell; a custom k mesh and cutoff are retained for every trial.

## Rust

From an application depending on this source tree:

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::pbc::mechanical::{
    pbc_relax, pbc_equation_of_state, elastic_properties,
    directional_elastic_response, PeriodicRelaxOptions,
};
use gfn1_rs::pbc::elastic_response::{pbc_elastic_response, AnalyticElasticOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"0 2.82 2.82 2.82 0 2.82 2.82 2.82 0\" pbc=\"T T T\"\nNa 0 0 0\nCl 2.82 0 0\n",
        0.0, false)?;
    let params = Gfn1Parameters::builtin()?;
    let electronic = ElectronicOptions {
        electronic_temperature: 300.0, energy_tolerance: 1e-13,
        charge_tolerance: 1e-11, max_scc: 1000, ..Default::default()
    };
    let pbc = PbcOptions {
        kmesh: KMesh { size: [2,2,2], gamma_centered: true,
            fold_time_reversal: true }, ..Default::default()
    };
    let relaxed = pbc_relax(&system, &params, &electronic, &pbc,
        &PeriodicRelaxOptions { relax_cell: true, ..Default::default() })?;
    if !relaxed.converged { return Err(relaxed.termination.into()); }
    let response = pbc_elastic_response(&relaxed.system, &params,
        &electronic, &pbc, &AnalyticElasticOptions::default())?;
    let stiffness = response.relaxed_stiffness.as_ref()
        .ok_or("no stable relaxed-ion response; inspect relaxed_response_error")?;
    let volume = relaxed.system.lattice.unwrap().volume();
    let density = (22.98976928+35.45)*1.66053906660e-27
        / (volume*(gfn1_rs::constants::BOHR_TO_ANGSTROM*1e-10).powi(3));
    let properties = elastic_properties(stiffness, density)?;
    let directional = directional_elastic_response(stiffness, density,
        [1.,0.,0.], [0.,1.,0.])?;
    let eos = pbc_equation_of_state(&relaxed.system, &params, &electronic,
        &pbc, &[0.94,0.97,1.,1.03,1.06], true,
        &PeriodicRelaxOptions::default())?;
    println!("B_Hill={} GPa, E_100={} GPa, EOS B={} GPa",
        properties.bulk_hill_gpa, directional.young_gpa, eos.fit.bulk_modulus_gpa);
    Ok(())
}
```

The model may predict a different equilibrium or an instability. Inspect the
returned convergence/stability values before interpreting a tensor. Use
`resources::ResourceContext` to reuse explicit CPU/memory and CP settings;
see [resources.md](resources.md) and [cp-settings.md](cp-settings.md).

## Python native

Geometry defaults to angstrom; property inputs/outputs below use explicitly
named atomic-unit keys. Tolerances, target pressure and optimization step
limits are in atomic units, independently of the geometry `unit` argument.

```python
import numpy as np
from gfn1_rs import Gfn1NativeCalculator

calc = Gfn1NativeCalculator(electronic_temperature=300.0,
    energy_tolerance=1e-13, charge_tolerance=1e-11, max_scc=1000,
    threads=2, memory_mb=2048)
nums = [11, 17]
pos = [[0.,0.,0.], [2.82,0.,0.]]
cell = [[0.,2.82,2.82], [2.82,0.,2.82], [2.82,2.82,0.]]
relax = calc.relax_periodic(nums, pos, cell, kmesh=(2,2,2),
    gamma_centered=True, relax_cell=True, force_tolerance=1e-7,
    stress_tolerance=1e-7)
assert relax["converged"], relax["termination"]
pos, cell = relax["positions_bohr"], relax["cell_bohr"]
elastic = calc.elastic_constants(nums, pos, cell, unit="bohr",
    kmesh=(2,2,2), gamma_centered=True)
C = elastic["relaxed_stiffness_hartree_per_bohr3"]
if C is None:
    raise RuntimeError(elastic["relaxed_response_error"])
volume = abs(np.linalg.det(cell))
density = (22.98976928+35.45)*1.66053906660e-27 / (volume*(5.29177210903e-11)**3)
moduli = calc.elastic_properties(C, density)
direction = calc.directional_elastic_properties(C, density, (1,0,0), (0,1,0))
eos = calc.equation_of_state(nums, pos, cell, [.94,.97,1.,1.03,1.06],
    unit="bohr", kmesh=(2,2,2), gamma_centered=True, relax_ions=True)
print(moduli["bulk_hill_gpa"], direction["young_gpa"], eos["fit"]["bulk_modulus_gpa"])
```

`calc.fit_equation_of_state(volumes_bohr3, energies_hartree)` fits existing
data without running electronic calculations. At least four distinct volumes
are required. Scanning more volumes and comparing fit windows is preferable
to treating an exact four-point fit as an uncertainty estimate.

## ASE

ASE unsuffixed geometry/energy/stress/stiffness use angstrom, eV and eV/angstrom³.
The explicit native-unit keys remain available. Density is always kg/m³;
the `*_gpa` modulus outputs and sound speeds in m/s are unchanged SI quantities.

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

atoms = Atoms("NaCl", positions=[[0,0,0],[2.82,0,0]],
    cell=[[0,2.82,2.82],[2.82,0,2.82],[2.82,2.82,0]], pbc=True)
atoms.calc = GFN1RSCalculator(electronic_temperature=300.0,
    energy_tolerance=1e-13, charge_tolerance=1e-11, max_scc=1000,
    threads=2, memory_mb=2048)
relax = atoms.calc.get_relax_periodic(atoms=atoms, kmesh=(2,2,2),
    gamma_centered=True, relax_cell=True, update_atoms=True,
    force_tolerance=5e-6, stress_tolerance=2e-6)
elastic = atoms.calc.get_elastic_constants(atoms=atoms,
    kmesh=(2,2,2), gamma_centered=True)
C = elastic["relaxed_stiffness"]  # eV/angstrom³; None if no static ionic response
if C is None:
    raise RuntimeError(elastic["relaxed_response_error"])
density = atoms.get_masses().sum()*1.66053906660e-27 / (atoms.get_volume()*1e-30)
moduli = atoms.calc.get_elastic_properties(C, density)
direction = atoms.calc.get_directional_elastic_properties(C, density, [1,0,0], [0,1,0])
eos = atoms.calc.get_equation_of_state([.94,.97,1.,1.03,1.06], atoms=atoms,
    kmesh=(2,2,2), gamma_centered=True, relax_ions=True)
print(moduli["young_hill_gpa"], direction["sound_speeds_m_per_s"], eos["fit"]["bulk_modulus_gpa"])
```

`update_atoms=True` applies a result only after convergence; otherwise it raises
and leaves the supplied Atoms unchanged. With `False`, the returned structure
can be inspected independently. `get_fit_equation_of_state(volumes, energies)`
accepts angstrom³/eV samples.

## Moduli, EOS and validation

`elastic_properties` returns compliance, Voigt/Reuss/Hill bulk and shear moduli,
Hill Young modulus/Poisson ratio, universal anisotropy and aggregate sound-speed
estimates. These are conventional zero-stress, untextured aggregate estimates,
not measured single-crystal observables. `directional_elastic_properties`
normalizes a loading and perpendicular transverse direction and solves the
Christoffel matrix. Its three velocities are sorted by stiffness eigenvalue;
polarizations are returned rather than assuming isotropic longitudinal modes.

The EOS is a third-order Birch-Murnaghan finite-strain polynomial, solved by
scaled, twice-reorthogonalized QR. It returns the fit error and the sampled
range. A minimum outside that range is rejected. The fitted bulk modulus and
the corresponding analytic elastic combination should be compared after
volume-window and k-mesh convergence.

`tests/pbc_mechanical.rs` checks independently generated BM3 parameters,
isotropic directional moduli/speeds, real force-based relaxation and every
elastic/internal-strain/nuclear Hessian component for scalar, nonzero ML,
multik-point and fractional-temperature cases. The initial coarse strain
stencil encountered image-cutoff changes; the retained fine stencil is checked
for identical pair sets. Finite differences are numerical oracles only.
`pbc_elastic_constants_seminumerical` is explicitly named as an audit route;
the installed `elastic_constants` API uses the analytic implementation.

The independent ML SiC relaxed-ion test actually optimizes both signs of all
six imposed strains. All 36 internal-displacement derivatives agree with the
analytic result within `4.45e-8 bohr`, and all 36 relaxed stiffness entries within
`7.15e-10 Hartree/bohr³`. The artificial, nonzero ML heads exercise structural
parameter derivatives; they are not fitted SiC parameters or accuracy evidence.

## Optimization work and continuation

Every trial uses one SCC, followed by the analytic gradient and stress from
that same state. Accepted shell charges initialize subsequent trials. Rejected
trials do not overwrite the accepted guess. Only the small charge vector is
retained between trials; dense electronic matrices are not accumulated.
`history` and `work` report SCC calls, iterations and actual Fock-map evaluations, rejected trials,
elapsed time, and separate SCC/gradient/stress timings. They distinguish an
expensive electronic solve from a geometry line-search problem.
The map-count completeness flag is false if a failed SCC call did not return
its detailed diagnostics; such a partial count is not a total work estimate.

The atomic step limit controls physical non-affine motion. The strain limit
controls affine cell deformation separately, so translating an atom by a
lattice vector cannot artificially restrict the allowed cell step. The
coordinate metric scales consistently when a cell is replicated.

Rust also provides `pbc_relax_with_guess`,
`pbc_elastic_response_with_guess` and `pbc_equation_of_state_with_guess`; their
last argument is `Option<&PbcSccGuess>`. A relaxation result supplies
`result.scf_guess` for the next calculation. The original entry points retain
their signatures. EOS calculations continue the electronic state between
volumes and reuse the final force/stress evaluation when ions are relaxed.
No SCC tolerance or electronic temperature is changed to obtain convergence.

An earlier cold-start survey was interrupted during LiF relaxation. Its
incomplete relaxation is not a converged result. The subsequent executions
below keep the same 300 K, Gamma mesh, cutoffs, mixing `0.1` and SCC thresholds
(`1e-13 Hartree`, `1e-11` charge). The final Auto SCC includes the analytic
charge-Newton fallback described in [the periodic guide](pbc.md).

| Execution | Geometry steps | SCC calls | SCC iterations | Fock maps | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Charge continuation before Newton fallback | 27 | 28 | 825 | not recorded | 9.51 |
| Final Auto SCC and origin-independent coordinate metric | 27 | 28 | 452 | 471 | 9.94 |

These wall times were measured with other jobs present; they are not a strict
speed benchmark. No numerical speedup factor is inferred from the interrupted
cold-start baseline. The final calculation reaches force/stress convergence,
post-relaxation analytic elasticity and all five ±6% EOS volume samples.
Historical failed SCC/EOS evaluations remain in separate optimizer receipts.

The stationary Gamma LiF cell still has negative internal curvature
(`-6.1695e-4 Hartree/bohr²`). It is a saddle rather than a stable crystal
minimum, so relaxed-ion stiffness is absent. Its BM3 value `21.80794 GPa`
describes the sampled stationary branch and is not a stable relaxed-ion
bulk-modulus prediction. The independent origin-invariance regression shifts
the full crystal by `[1000, -800, 500] bohr`; it reaches the same energy and
volume in the same 27 steps.

The strain-skeleton builder shares its geometry, learned descriptors, Ewald
derivatives and Gaussian pair derivatives across k points. A matched benchmark
uses one converged nonzero-ML SiC electronic state per mesh, 13 AOs, a 20 bohr
AO cutoff, 300 K, two workers and a 512 MB workspace limit. It repeats both
assembly patterns three times with alternating timing order.

| Electronic mesh | Retained k points | Single-k loop median (s) | Shared assembly median (s) |
| --- | ---: | ---: | ---: |
| 2×2×2 | 8 | 4.957696 | 0.552759 |
| 4×4×4 | 36 | 26.622331 | 0.881968 |

Every real/imaginary overlap and Fock derivative, shell-potential entry and
coordination entry agrees exactly. These timings cover strain assembly only;
SCC and the rest of the response calculation are excluded. The measured
[receipt](../tests/data/strain-skeleton-performance.json) retains all timings,
the original linked-library/source hashes and the exact measured source text.
The permanent [reproduction source](../tests/profiling/strain_skeleton.rs)
was formatted after measurement; its separate path/hash is recorded without
remeasuring or relabeling the historical timings. Reproduce the
calculation with
`cargo test --offline --profile reltest --test profile_strain_skeleton`;
a new execution produces new timing data rather than updating that receipt.

## Crystal execution results

[The mechanical receipt](../tests/data/crystal-mechanical-properties.json)
contains complete input cells, source records, all forces/stresses, elastic
and mixed tensors, stability diagnostics, SCC retry histories and raw input
hashes. The reusable
[execution driver](../tests/validation/crystal_mechanical_validation.rs)
produces new measurements; the one-off report formatter is not distributed.
Source hashes labeled `report_generation` describe the source at report
generation, not an unrecorded earlier compiled binary.

The six archetypes below are specified ideal representatives. Their analytic
reference pressures already show why a converged calculation at a single mesh
does not establish sampling convergence or agreement with experiment. Both
meshes use 300 K electronic occupations, Gamma centering and the same 30 bohr
AO cutoff; the nuclear temperature is not set by the electronic temperature.

| Representative input | Pressure, Gamma (GPa) | Pressure, 2×2×2 (GPa) |
| --- | ---: | ---: |
| C diamond | 360.391 | 27.483 |
| Si diamond | 37.698 | 16.125 |
| SiC zincblende | 168.998 | 26.726 |
| LiF rock salt | 54.380 | 16.514 |
| NaCl rock salt | 11.476 | -1.192 |
| MgO rock salt | 99.016 | 15.637 |

For three model-relaxed Gamma cells, both the force and stress criteria were
satisfied. The moduli are calculated from the analytic relaxed-ion tensor;
these are model results, not experimental material constants.

| Model-relaxed cell | Volume (bohr³/primitive cell) | Hill B (GPa) | Hill G (GPa) | BM3 B, ±6% volume (GPa) |
| --- | ---: | ---: | ---: | ---: |
| C diamond | 145.125727 | 140.376 | 79.044 | 144.770 |
| Si diamond | 361.163030 | 106.003 | 60.984 | 101.389 |
| SiC zincblende | 243.190397 | 124.764 | 34.007 | 136.684 |

The EOS window changes retained image pairs and spans nonlinear curvature;
its fit bulk modulus has not converged to the local elastic value in this
survey. Reduce the volume window and converge all cutoffs and the k mesh before
using those numbers as physical predictions.

The separate experimental test uses the pinned Na2MoO4 structure from COD
2240655, documented in the [real-structure fixture](../tests/data/real_crystals.json)
and [Fortes' crystallographic report](https://doi.org/10.1107/S2056989015008774).
At its experimental coordinates, the model gives pressure `2.4610 GPa`, maximum
atomic gradient `0.015920 Hartree/bohr`, and six negative internal curvatures
(minimum `-0.013008 Hartree/bohr²`). The full analytic clamped tensor is computed
(`C11=116.594`, `C12=83.987`, `C44=98.234 GPa`), but relaxed-ion moduli are
unavailable because this input is neither stationary nor internally stable in
the model. This is a documented model/structure limitation, not a successful
prediction of the experimental crystal's elastic constants.

References: [Birch, finite elastic strain](https://doi.org/10.1103/PhysRev.71.809),
[Hill, crystalline aggregates](https://doi.org/10.1088/0370-1298/65/5/307),
[Liu and Nocedal, limited-memory optimization](https://doi.org/10.1007/BF01589116),
and the [ABINIT stress/strain conventions](https://docs.abinit.org/theory/documents/elasticity-oganov.pdf).
