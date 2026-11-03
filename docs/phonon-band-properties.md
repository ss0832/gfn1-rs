# Crystal spectra, heat capacities and thermal expansion

The shared Rust/pip CLI also exposes these calculations and postprocessors.
[crystal-cli.md](crystal-cli.md) gives runnable requests, unit-tagged arguments,
result chaining and a four-interface coverage table.

These APIs extend the existing analytic supercell phonon and periodic SCC
engines. The Rust calculations retain the active ML Hamiltonian, canonical
electronic temperature and resource/CP settings. No bare-GFN1 fallback is used.
The electronic temperature is independent of the lattice temperatures supplied
to harmonic or quasi-harmonic thermodynamics.

## Quantities and normalization

| Output | Definition and units |
| --- | --- |
| Phonon mesh | All `q=(i/nx,j/ny,k/nz)` commensurate with the chosen supercell; frequencies in `cm^-1`. |
| Phonon DOS | Normalized Gaussian broadening; states per `cm^-1` per primitive cell. Imaginary modes retain a negative signed coordinate. |
| Vibrational thermodynamics | ZPE, Helmholtz free energy and internal energy in Hartree/cell; entropy and `Cv` in Hartree/(K cell). |
| QHA | Zero-pressure equilibrium volume in bohr³, isothermal bulk modulus, volumetric expansion in `1/K`, and `Cp`. |
| Band structure | One eigenvalue per physical scalar band, in Hartree and eV; path distance in bohr^-1. |
| Electronic DOS | States/(eV cell), including scalar spin degeneracy two. The real embedding's duplicate eigenvalues are removed first. |
| Sampled band gap | Signed `CBM-VBM` on the supplied k points, plus the smallest direct gap; negative indirect values mean overlap. |

Every q/k integration weight must be finite, nonnegative, and sum to one.
The DOS reports both the expected state count and its trapezoidal integral on
the requested grid. Finite-grid Gaussian tails are retained as a convergence
diagnostic, rather than rescaled to conceal a truncated range.

The phonon engine samples **Gamma of the supercell**. This folds a primitive-cell
electronic grid; it is distinct from running only primitive-cell Gamma.
Changing the supercell can therefore change both electronic integration and
the accessible phonon q mesh. Converge those effects before interpreting DOS
or thermodynamic differences as material properties.

## Rust

Use a client crate depending on the local `gfn1-rs` source, as described in
[ml-training.md](ml-training.md). This example uses the representative diamond
cell also retained in [crystal-dfpt-results.md](crystal-dfpt-results.md).

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::pbc::phonon::{AnalyticSupercellPhonons, PhononSupercellOptions};
use gfn1_rs::pbc::spectra::{
    sample_phonon_mesh, phonon_dos, harmonic_thermodynamics,
    pbc_band_structure, electronic_dos,
};

fn main() -> gfn1_rs::Result<()> {
    let crystal = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"0 1.7835 1.7835 1.7835 0 1.7835 1.7835 1.7835 0\" pbc=\"T T T\"\nC 0 0 0\nC .89175 .89175 .89175\n",
        0.0, false)?;
    let parameters = Gfn1Parameters::builtin()?;
    let options = ElectronicOptions {
        electronic_temperature: 300.0,
        energy_tolerance: 1e-13, charge_tolerance: 1e-11,
        ..Default::default()
    };
    let reference = AnalyticSupercellPhonons::build(
        &crystal, &parameters, &options, &PbcOptions::default(),
        &[12.011, 12.011], PhononSupercellOptions {
            repetitions: [2, 2, 2], ..Default::default()
        })?;
    let mesh = sample_phonon_mesh(&reference)?;
    let grid: Vec<_> = (0..2101).map(|i| -100.0 + 2.0*i as f64).collect();
    let dos = phonon_dos(&mesh.wavenumbers_cm1, &mesh.weights, &grid, 10.0)?;
    let thermal = harmonic_thermodynamics(
        &mesh.wavenumbers_cm1, &mesh.weights, &[0.0, 100.0, 300.0], 1e-3)?;
    println!("{}", mesh.as_json());
    println!("{}", dos.as_json());
    println!("{}", thermal[2].as_json());

    let mut electronic_mesh = PbcOptions::default();
    electronic_mesh.kmesh = KMesh {
        size: [4, 4, 4], gamma_centered: true, fold_time_reversal: true,
    };
    let path = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]];
    let bands = pbc_band_structure(
        &crystal, &parameters, &options, &electronic_mesh, &path)?;
    println!("{}", bands.as_json());

    // For a bulk electronic DOS, use a BZ integration grid and its weights.
    let points = gfn1_rs::pbc::kpoints::monkhorst_pack(
        [4, 4, 4], [true, true, true], true);
    let kpoints: Vec<_> = points.iter().map(|p| p.fractional).collect();
    let weights: Vec<_> = points.iter().map(|p| p.weight).collect();
    let sampled = pbc_band_structure(
        &crystal, &parameters, &options, &electronic_mesh, &kpoints)?;
    let energy_grid: Vec<_> = (0..5001).map(|i| -100.0 + 0.025*i as f64).collect();
    let electronic = electronic_dos(&sampled, &weights, &energy_grid, 0.15)?;
    println!("{}", electronic.as_json());
    Ok(())
}
```

For repeated k paths, call `run_pbc_scc` once and use
`band_structure_from_scc(&crystal, &scf, &options, &kpoints)`. The SCF sampling
sets the density; the requested path only resamples its converged Hamiltonian.

To use user-trained weights, set `options.ml_model` before building either
reference:

```rust
let options = ElectronicOptions {
    ml_model: Some(std::sync::Arc::new(
        gfn1_rs::ml::MlHamiltonian::resolve("trained.model")?)),
    ..options
};
```

The binding uses the same model loader as [ml-training.md](ml-training.md).
`ml_model_sha256` identifies the model retained by the band/phonon reference.
An ML correction head enters the scalar total energy and QHA volume curve;
it is not added as a uniform shift to orbital eigenvalues. Learned s/p/d
level heads enter the Bloch Hamiltonian before SCC and remain in its bands.

## Python native

```python
import numpy as np
from gfn1_rs import Gfn1NativeCalculator

numbers = [6, 6]
positions = [[0, 0, 0], [.89175, .89175, .89175]]
cell = [[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]]
calc = Gfn1NativeCalculator(
    electronic_temperature=300.0, energy_tolerance=1e-13,
    charge_tolerance=1e-11, threads=4, memory_mb=2048,
    # ml_model_path="trained.model", log_file="crystal.jsonl", log_console=True,
)
mesh = calc.phonon_mesh(numbers, positions, cell, repetitions=(2, 2, 2),
                        masses_amu=[12.011, 12.011])
dos = calc.phonon_dos(mesh["wavenumbers_cm1"], mesh["weights"],
                      np.linspace(-100, 4100, 2101).tolist(), sigma_cm1=10.0)
thermal = calc.harmonic_thermodynamics(
    mesh["wavenumbers_cm1"], mesh["weights"], [0.0, 100.0, 300.0],
    zero_tolerance_cm1=1e-3)

path = [(0., 0., 0.), (.5, 0., 0.), (.5, .5, .5)]
bands = calc.band_structure(numbers, positions, cell, path,
                            kmesh=(4, 4, 4), gamma_centered=True)
print(bands["energies_ev"], bands["sampled_gap"])

kpoints = [(i/4, j/4, k/4) for i in range(4) for j in range(4) for k in range(4)]
sampled = calc.band_structure(numbers, positions, cell, kpoints,
                              kmesh=(4, 4, 4), gamma_centered=True)
ed = calc.electronic_dos(sampled["energies_hartree"], [1/64]*64,
                         np.linspace(-100, 25, 5001).tolist(), sigma_ev=0.15)
print(ed["expected_states"], ed["integrated_states"])
```

All postprocessors run Rust code. Raw energy and entropy keys explicitly retain
atomic units; frequency/DOS keys retain the units in the table above. Invalid
weights, unsupported states, nonconverged SCC references and imaginary phonons
raise `ValueError` with the failed condition.
The electronic DOS accepts Hartree input by default; `energy_unit="ev"`
selects already converted eV rows.

## ASE

```python
import numpy as np
from ase.build import bulk
from gfn1_rs import GFN1RSCalculator

atoms = bulk("C", "diamond", a=3.567)
atoms.calc = GFN1RSCalculator(electronic_temperature=300.0,
                              threads=4, memory_mb=2048)
mesh = atoms.calc.get_phonon_mesh((2, 2, 2), atoms=atoms)
dos = atoms.calc.get_phonon_dos(
    mesh["wavenumbers_cm1"], mesh["weights"],
    np.linspace(-100, 4100, 2101).tolist(), sigma_cm1=10.0)
thermal = atoms.calc.get_harmonic_thermodynamics(
    mesh["wavenumbers_cm1"], mesh["weights"], [0.0, 100.0, 300.0])
bands = atoms.calc.get_band_structure(
    [(0., 0., 0.), (.5, 0., 0.), (.5, .5, .5)], atoms=atoms,
    kmesh=(4, 4, 4), gamma_centered=True)
```

ASE isotope masses are passed to the phonon engine. Unit-labelled dictionary
keys keep the same meaning in all interfaces; in particular, a key ending in
`_hartree` remains Hartree. The model and worker configuration follows the
calculator used by `atoms`.
Unsuffixed energies, volumes, entropy and heat capacities use eV, angstrom³,
eV/K and eV/K respectively. `get_electronic_dos` accepts the unsuffixed eV
`bands["energies"]`. The supercell phonon method requires calculator
`kgrid=None` or `(1,1,1)`; it reports the actual primitive electronic folding
implied by the supercell repetitions.

## QHA and thermal expansion

At least five volumes are required. Keep the primitive-cell contents, isotope
masses, electronic temperature, sampling and ionic-relaxation convention
consistent across the volume curve. Each harmonic mesh exposes
`reference_energy_hartree_per_primitive_cell`, taken from the same converged
supercell SCC as its Hessian, divided by the number of primitive copies.
This gives a matching static reference for the quasi-harmonic fit.

The Rust entry point is

```rust
gfn1_rs::pbc::spectra::quasiharmonic_zero_pressure(
    &volumes_bohr3, &static_energies_hartree,
    &phonons_by_volume_cm1, &q_weights, &temperatures_kelvin, 1e-3)?;
```

The phonon array layout is `[volume][q][mode]`. Python native accepts the same
arrays through `calc.quasiharmonic_zero_pressure(...)`; ASE uses
`atoms.calc.get_quasiharmonic_zero_pressure(...)`.
The ASE entry point takes volumes in angstrom³ and static energies in eV;
its explicit native-unit output keys are retained alongside ASE-unit keys.

This Python example prepares the required matching arrays. Choose the volume
range around the model-relaxed equilibrium structure; refine it until fit
residuals and the requested properties converge.

```python
from ase.units import Bohr

ratios = np.linspace(0.92, 1.08, 9)
volumes, static, phonons = [], [], []
base_volume = abs(np.linalg.det(cell)) / Bohr**3
for ratio in ratios:
    scale = ratio**(1/3)
    value = calc.phonon_mesh(
        numbers, (np.asarray(positions)*scale).tolist(),
        (np.asarray(cell)*scale).tolist(), repetitions=(2, 2, 2),
        masses_amu=[12.011, 12.011])
    volumes.append(base_volume*ratio)
    static.append(value["reference_energy_hartree_per_primitive_cell"])
    phonons.append(value["wavenumbers_cm1"])
qha = calc.quasiharmonic_zero_pressure(
    volumes, static, phonons, value["weights"], [0., 100., 300., 600.],
    zero_tolerance_cm1=1e-3)
print(qha)
```

The third-order Eulerian finite-strain fit minimizes
`F_static(V) + F_vib(V,T)` inside the sampled volume interval. The response uses

```text
alpha_V = (dS/dV)_T / B_T
Cp = Cv + T * V * B_T * alpha_V^2
```

The entropy and heat capacity use the same finite-strain interpolation basis
as the free energy. No numerical temperature derivative is required. The zero
lattice-temperature entropy, expansion and heat capacity vanish explicitly.
Volume dependence remains a fitted QHA approximation built from analytic
pointwise Hessians. Electronic entropy/heat-capacity changes with lattice
temperature, intrinsic phonon renormalization and finite-pressure QHA are
outside this entry point. A fixed electronic smearing of 300 K stays fixed.
See the [Phonopy QHA formulation](https://phonopy.github.io/phonopy/qha.html)
for the thermodynamic definitions.

## Stability and physical interpretation

Harmonic thermodynamics rejects frequencies below `-zero_tolerance_cm1`.
Modes within the explicitly supplied zero window are counted and omitted;
the returned weight makes the finite-mesh translation-mode omission visible.
This procedure does not stabilize an imaginary optical branch. Refining q
sampling reduces the weight of the three Gamma translations. Gamma-only
thermodynamics is a mode sum, not a converged bulk heat capacity.
The oscillator expressions follow the
[harmonic phonon partition function](https://phonopy.github.io/phonopy/formulation.html#thermodynamic-properties).

The band gap is a gap of the active GFN1/ML orbital Hamiltonian on the supplied
sample. It is distinct from a quasiparticle gap and from the TD-GFN1/sTDA
optical excitation energies. A positive gap on a high-symmetry path alone does
not establish an insulating bulk. `thermally_integer_occupations` checks the
supplied sample against the reference global Fermi level; use a converged BZ
grid for that diagnosis. The result retains signed band overlap.

Scalar bands support structural ML, including caller-trained weights. SOC,
spin polarization, on-site multipoles, exchange, +U, solvent and active external
fields require different Hamiltonian reconstruction and are rejected by this
band entry point. Existing SOC supercell phonons can still feed the generic
phonon postprocessors.

`tests/pbc_spectra.rs` checks oscillator identities and low/high-temperature
limits, DOS state counts/tails, imaginary-mode rejection, k periodicity,
weighted band-energy traces, nonzero ML heads and correction-head separation,
analytic ML supercell mesh reuse, diamond mode symmetry and a synthetic QHA
Gruneisen identity. Experimental structures and model accuracy must be checked
separately from these numerical correctness tests.
