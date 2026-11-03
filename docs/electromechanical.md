# Analytic bulk electrical and electromechanical response

For the shared Rust/pip CLI, see [crystal-cli.md](crystal-cli.md): it wires
the same Born/dielectric/coupled/IR/LO–TO backends and documents the complete
Rust/native/ASE/CLI mapping with runnable result-reference requests.

The scalar periodic GFN1 and structural ML Hamiltonians expose Born effective
charges, proper clamped-ion piezoelectricity and electronic dielectric response.
The combined response shares a single converged SCC state with analytic
elasticity and the nuclear Hessian. Stable, stationary references additionally
give relaxed-ion piezoelectricity, static dielectric response and, at zero
stress, piezoelectric strain constants. There are no displaced nuclei, strained
cells or finite fields in these production derivative paths.

These are model predictions. Availability does not imply experimental accuracy;
see [actual crystal results](crystal-properties-results.md), including strong
sampling effects and rejected nonstationary or fractionally occupied cases.

## Definitions and conditions

All three directions must be periodic. Berry strings use the same complete,
unfolded Gamma-centred mesh as the reference SCC. A single Gamma point is also
accepted, but does not establish Brillouin-zone convergence. Occupied-space
Berry polarization requires an insulator with effectively integer occupations;
the occupation/gap guard is enforced at the **requested** electronic
temperature. A metallic or appreciably smeared state is rejected, not cooled.
This is not a general mixed-state polarization theory for thermal metals.
Multipole, spin-polarized, +U, long-range-exchange, solvation and externally
biased composite Hamiltonians are outside this scalar response API.

With cell volume Ω and Cartesian polarization P,

```text
Z*[atom][i][j] = Ω dP_i/dR_atom,j
e_clamped[i][J] = proper dP_i/deta_J
epsilon_infinity = I + 4 pi chi_electronic
K = d²F/dR²; Lambda = d²F/(dR deta)
du/deta = -K_internal^-1 Lambda
e_relaxed = e_clamped + Z* (du/deta) / Ω
d = e_relaxed C_relaxed^-1
epsilon_static,free-stress = epsilon_static,fixed-strain
                            + 4 pi e_relaxed C_relaxed^-1 e_relaxed^T
```

Voigt order is `xx, yy, zz, yz, xz, xy` with engineering shear strain.
Proper piezoelectricity differentiates the reduced Berry polarization phases;
it removes rigid-rotation and volume-dilution artifacts instead of returning
the branch-dependent improper Cartesian derivative. The electronic field
operator is the derivative of the **discrete Berry electric enthalpy**
`F - Ω E·P`; it is not a reference-cell Mulliken dipole approximation.

Nuclear/strain overlap and Fock skeletons include Gaussian centre derivatives,
reciprocal-vector changes, coordination dependence, Ewald charge potentials,
all active structural ML heads and canonical SCC response. A polarization
quantum still exists: changes on a finite path require consistent branch
tracking; a local analytic derivative does not select a global branch.

Internal response projects out three rigid translations only. Unstable or
nonstationary ions return `None` for the unavailable relaxed properties with a
reason. Piezoelectric `d` and free-stress dielectric response also require
stable elasticity and a zero-stress reference. Increasing these guard
tolerances changes the admission policy, not the physical stationary state.

## Rust API

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::pbc::polarization::BerryPolarizationOptions;
use gfn1_rs::pbc::polarization::response::{
    pbc_polarization_derivatives, pbc_electronic_dielectric,
};
use gfn1_rs::pbc::electromechanical::{
    pbc_electromechanical_response, ElectromechanicalOptions, lo_to_modes,
};

fn calculate(system: &PeriodicSystem) -> gfn1_rs::Result<()> {
    let parameters = Gfn1Parameters::builtin()?;
    let electronic = ElectronicOptions {
        electronic_temperature: 300.0,
        energy_tolerance: 1e-13, charge_tolerance: 1e-11,
        ..Default::default()
    };
    let pbc = PbcOptions {
        kmesh: KMesh { size: [2, 2, 2], gamma_centered: true,
                      fold_time_reversal: false },
        ..Default::default()
    };
    let berry = BerryPolarizationOptions::from_kmesh(pbc.kmesh);
    let z = pbc_polarization_derivatives(system, &parameters,
        &electronic, &pbc, &berry)?;
    let eps = pbc_electronic_dielectric(system, &parameters,
        &electronic, &pbc, &berry)?;
    println!("Born: {}; dielectric: {}", z.as_json(), eps.as_json());

    let masses: Vec<_> = system.atoms.iter().map(|a|
        gfn1_rs::data_tables::relative_atomic_mass(a.z)).collect();
    let settings = ElectromechanicalOptions { berry, ..Default::default() };
    let response = pbc_electromechanical_response(system, &parameters,
        &electronic, &pbc, &masses, &settings)?;
    let modes = lo_to_modes(&response.elastic.nuclear_hessian,
        &response.berry.born_effective_charges,
        &response.electronic.relative_dielectric,
        response.elastic.reference_system.lattice.unwrap().volume(),
        &masses, [1.0, 0.0, 0.0])?;
    println!("{}", modes.as_json());
    Ok(())
}
```

Use the combined function when multiple properties are needed; the separate
functions each run their own SCC. Set `electronic.ml_model` to an `Arc` of a
loaded `MlHamiltonian` to include its structural derivatives. Both the model
hash and actual reference conditions are retained. `settings.elastic` accepts
the same optional ionic reference relaxation as analytic elasticity; variable
cell relaxation is a separate preceding operation.

## Python native

```python
from gfn1_rs import Gfn1NativeCalculator

numbers = [11, 17]
positions = [[0, 0, 0], [2.82, 0, 0]]
cell = [[0, 2.82, 2.82], [2.82, 0, 2.82], [2.82, 2.82, 0]]
calc = Gfn1NativeCalculator(electronic_temperature=300.0,
    energy_tolerance=1e-13, charge_tolerance=1e-11,
    threads=2, memory_mb=2048)  # add ml_model_path="user.model" if applicable
args = dict(numbers=numbers, positions=positions, cell=cell,
            kmesh=(2, 2, 2), gamma_centered=True)
born = calc.born_effective_charges(**args)
electronic = calc.dielectric_tensor(**args)
response = calc.electromechanical_response(**args, masses_amu=[22.989769, 35.45])
print(born["born_effective_charges"])
print(electronic["electronic_relative_dielectric"])
print(response["relaxed_response_error"])  # inspect before using e/d/static epsilon
lo = calc.lo_to_modes(
    response["elastic"]["nuclear_hessian_hartree_per_bohr2"],
    response["berry"]["born_effective_charges"],
    response["electronic"]["electronic_relative_dielectric"],
    abs(__import__("numpy").linalg.det(cell)) / (0.529177210903**3),
    [22.989769, 35.45], direction=(1, 0, 0))
```

Positions/cell default to angstrom input; `unit="bohr"` is also supported.
Native Hessians are Hartree/bohr², volume for the pure mode functions is bohr³,
Born charges are e, proper piezoelectricity is e/bohr² (also explicit C/m²),
`d` is e·bohr/Hartree (also pC/N) and dielectric tensors are relative,
dimensionless tensors. `polar_mode_properties` takes the same Hessian/Born
inputs plus masses and volume and needs no extra SCC. Pure postprocessing
does not fabricate an ML identity for arrays supplied by a caller.

## ASE

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
atoms.set_masses([22.989769, 35.45])
atoms.calc = GFN1RSCalculator(electronic_temperature=300.0,
    energy_tolerance=1e-13, charge_tolerance=1e-11,
    kgrid=(2, 2, 2), threads=2, memory_mb=2048)
z = atoms.calc.get_born_effective_charges(atoms)
eps = atoms.calc.get_dielectric_tensor(atoms)
r = atoms.calc.get_electromechanical_response(atoms)
lo = atoms.calc.get_lo_to_modes(
    r["elastic"]["nuclear_hessian"], r["berry"]["born_effective_charges"],
    r["electronic"]["electronic_relative_dielectric"],
    atoms.get_volume(), atoms.get_masses(), (1, 0, 0))
```

ASE mode functions accept Hessians in eV/angstrom² and volume in angstrom³.
Explicit atomic-unit keys are preserved; additional ASE-unit fields use ASE's
unit constants. Born charges and dielectric tensors need no unit conversion.
The direct ASE polarization/piezoelectric fields are e/angstrom² and `d` is
e·angstrom/eV. These wrappers preserve unavailable-result reasons and never
replace them with zeros. See the returned keys and the tests for unit-specific
fields rather than mixing native CODATA and ASE constants.

## IR and LO–TO

For mass-normalized Gamma eigenvectors the mode effective charge is
`Z_mode,i = sum_atom,j Z*[atom][i][j] e_mode,atom,j / sqrt(M_atom)`.
The output retains its oscillator tensor and the harmonic ionic dielectric
increment. Imaginary modes are retained as signed wavenumbers and prevent an
unjustified stable static dielectric tensor; charged zero modes are also
rejected in the ionic sum.

The directional nonanalytic correction is

```text
D_NA[atom j, atom' l] = (4 pi / Ω)
  (q_i Z*[atom][i][j]) (q_k Z*[atom'][k][l])
  / (sqrt(M_atom M_atom') (q^T epsilon_infinity q))
```

`lo_to_modes` requires explicit Born charges, positive-definite electronic
dielectric tensor, isotope masses and Cartesian direction. It computes the
directional Gamma limit, not a general finite-q dynamical matrix or an
experimental dielectric fit. Finite-q phonons use the separate commensurate
supercell API. The bare Gamma acoustic residual is reported; no silent tensor
symmetrization or acoustic-sum projection hides convergence errors.

## Verification and references

All Born/proper-piezo components are compared to reconverged Berry differences
for scalar and nonzero structural-ML cells, including self-image descriptors.
Every electronic susceptibility component is checked against an independent
finite-field variational orbital minimization of the discrete Berry enthalpy.
The scalar diatomic LO–TO/ionic dielectric relation supplies an independent
mode-level check. Numerical differences are validation oracles only.

The full k-mesh strain skeleton now shares geometry, Ewald/CN/ML preparation
and Gaussian AO derivatives across k points. A matched nonzero-ML SiC
benchmark compares the former single-k loop with the bulk assembly using the
same SCC state, three paired repetitions and alternating order:

| Mesh / retained k points | Single-k loop median (s) | Shared assembly median (s) |
| --- | ---: | ---: |
| 2³ / 8 | 4.958 | 0.553 |
| 4³ / 36 | 26.622 | 0.882 |

Every compared complex matrix and shell/CN vector entry is exactly unchanged.
These are **assembly-only** timings, excluding SCC and the remaining response
work. The [performance receipt](../tests/data/strain-skeleton-performance.json)
retains all measurements, original measured source/library hashes and the
formatted reproduction source mapping. This folded timing workload is
separate from the unfolded meshes required by the Berry response itself.

- [Vanderbilt, proper piezoelectric response](https://arxiv.org/abs/cond-mat/9903137),
  [DOI: 10.1016/S0022-3697(99)00273-5](https://doi.org/10.1016/S0022-3697(99)00273-5).
- [Souza, Iniguez and Vanderbilt, finite-electric-field Berry enthalpy](https://arxiv.org/abs/cond-mat/0205633),
  [DOI: 10.1103/PhysRevLett.89.117602](https://doi.org/10.1103/PhysRevLett.89.117602).
- [Wu, Vanderbilt and Hamann, coupled strain and electric-field response](https://arxiv.org/abs/cond-mat/0501548),
  [DOI: 10.1103/PhysRevB.72.035105](https://doi.org/10.1103/PhysRevB.72.035105).
- [Phonopy formulation: nonanalytic dynamical-matrix correction](https://phonopy.github.io/phonopy/formulation.html).
