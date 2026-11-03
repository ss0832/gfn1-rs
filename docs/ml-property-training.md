# Fine-tuning ML Hamiltonian weights on physical properties

`fit-properties` warm-starts the actual native `.model` supplied by the user.
It changes only explicitly selected s/p/d level-head or scalar correction-head
coordinates. It does not replace the model by a builtin model or refit the
global GFN1 parameter tables. The ordinary energy/force/dipole/reaction trainer
is described in [ml-training.md](ml-training.md).

The physical predictions below use the analytic Rust production backends,
including derivatives of the learned radial features. General property-loss
gradients are **central differences of model parameters** at fixed nuclear
coordinates, cell, field, charge, electronic temperature and k mesh. They are
not finite differences of positions or strains, and are not claimed to be
analytic third-order mixed parameter derivatives. For unmasked molecular
energy/force objectives, `gradient_method="auto"` reuses the existing stationary
energy and implicit CP-SCC analytic parameter-gradient implementation.

## Available objectives and exact units

All targets use the same explicitly tagged units in Rust, Python native, ASE
and the CLI. The ASE training method does not guess or convert a dataset's
units. These are fixed-geometry targets; an optimization trajectory is not
silently inserted into an objective evaluation.

| `observable` | `unit` | Shape and definition |
| --- | --- | --- |
| `energy` | `hartree` | Scalar electronic Mermin free energy plus model classical/correction terms |
| `forces` | `hartree/bohr` | N×3, minus the analytic energy gradient |
| `stress` | `hartree/bohr^3` | 3×3 periodic analytic stress |
| `hessian` | `hartree/bohr^2` | 3N×3N analytic nuclear Hessian |
| `phonon_eigenvalues` | `hartree/bohr^2/amu` | 3N sorted mass-weighted Γ eigenvalues |
| `phonon_wavenumbers` | `cm^-1` | 3N signed Γ frequencies; negative denotes imaginary |
| `elastic_clamped`, `elastic_relaxed` | `hartree/bohr^3` | 6×6 material Green-strain stiffness; engineering Voigt xx yy zz yz xz xy |
| `born_effective_charges` | `e` | N×3×3, atom/polarization/displacement |
| `piezoelectric_clamped`, `piezoelectric_relaxed` | `e/bohr^2` | 3×6 proper piezoelectric stress tensor |
| `dielectric_electronic` | `dimensionless` | 3×3 clamped-ion relative tensor from Berry electric enthalpy |
| `piezoelectric_strain` | `e*bohr/hartree` | 3×6, e times relaxed elastic compliance |
| `dielectric_static_fixed_strain`, `dielectric_static_free_stress` | `dimensionless` | 3×3 coupled electronic/ionic response with the stated boundary condition |
| `band_energies` | `hartree` | requested k-point×physical scalar-band eigenvalues of the converged ML Hamiltonian |
| `band_gap` | `hartree` | scalar sampled indirect model orbital gap; an overlap may be negative |

These are exactly **17 registered objective names**. DOS, harmonic/QHA
thermodynamics, EOS fits, lattice constants and matching a relaxed geometry
are not implicit additional objectives in this trainer. It does not run an
inner structural optimization or differentiate an optimization trajectory.
Those quantities can be computed with the separate property APIs, but a
request to train one must use a deliberately added objective rather than an
unrecognized JSON name or a fabricated zero label.

Stress, elasticity, Berry, dielectric and band objectives require a periodic
sample. Molecular Hessians and their mass-weighted eigenvalues are also
supported. These phonon labels are Γ quantities, not a Brillouin-zone integral.
Sorted eigenvalues at exact crossings can be nonsmooth with respect to model
weights; degenerate eigenvectors are not fit targets. A sampled orbital gap is
not an experimental optical or quasiparticle gap.

The current training registry accepts the scalar GFN1/structural-ML Hamiltonian
without multipoles, exchange, spin, +U, solvation, D4 or external fields. Each
underlying backend retains its physical admission checks. Berry labels require
an insulating occupied subspace at the declared electronic temperature and a
matching complete Γ-centred unfolded SCC/Berry mesh. Relaxed-ion targets require
stationary internal coordinates and stable internal modes. d and free-stress
dielectric targets additionally require stable zero-stress elasticity. Missing,
unstable or unsupported responses cause a descriptive error; no zero tensor is
substituted. Keep **300 K** electronic conditions consistently when comparing
or fitting the crystal calculations in this release.

Correction heads affect energy and its nuclear/strain derivatives. They do not
change the fixed-geometry electronic eigenvectors, Born tensors, clamped proper
piezoelectricity, electronic dielectric or band energies/gaps. Selecting only
correction-head parameters for any of those electronic training objectives is
rejected. Correction biases also have no force/Hessian sensitivity; a general
parameter-difference run records zero-sensitive selected coordinates and a
`zero_selected_parameter_sensitivity` termination when all selected coordinates
are insensitive. Choose level-head coordinates when changing electronic
response is the goal.

## Dataset, scales, masks and held-out data

Use schema `gfn1-rs-property-training-v1`. Top-level required fields are
`coordinate_units` (`angstrom` or `bohr`), `provenance`, `license_spdx`,
`reference_method`, `independent_reference` and `samples`. Every sample requires
unique `id`, explicit `split` (`train` or `validation`), inline `xyz`, `charge`,
`electronic_temperature_kelvin` and `targets`. At least one positively weighted
target must exist in each split. Split by distinct structures before fitting.
The held-out split does not enter gradients, line search or energy alignment.
There is no implicit energy-offset alignment in this trainer.

Each target has `observable`, exact `unit`, `values`, positive `scale` and an
optional nonnegative `weight` (default 1). An optional boolean `mask` must match
the complete tensor shape. It selects entries to fit; it must select at least
one entry. Unselected entries do not affect loss or its gradient. The loss is

\[
 L=\sum_t\frac{w_t}{2N_t}\sum_{a\in\mathrm{mask}_t}
 \left[\frac{y_{t,a}(\theta)-y^{\rm ref}_{t,a}}{s_t}\right]^2
 +\frac{\lambda}{2}\sum_{p\in\mathrm{selected}}(\theta_p-\theta_{p,0})^2.
\]

Use scales meaningful for the accuracy desired in each unit. Equal weights
alone do not balance quantities with different dimensions. The regularizer is
to the actual initial weights, not to zero. Reports retain targets, masks,
scales, per-property losses/RMSE, exact conditions, selected coordinate changes,
initial/final model SHA-256, dataset SHA-256/provenance and held-out metrics.

Periodic samples additionally require explicit `kmesh: [nx,ny,nz]`,
`gamma_centered`, `fold_time_reversal` and `ao_cutoff_bohr`. Supply `masses_amu`
for phonon/coupled objectives and `band_kpoints` for bands/gaps. These grids
have separate purposes: SCC `kmesh` fixes the density, while `band_kpoints`
samples that converged Hamiltonian. No mesh is inferred from a fitted label.
SCC controls can be supplied as `scc_energy_tolerance_hartree`,
`scc_charge_tolerance_e`, `scc_max_iterations` and `mixing`; defaults are
1e−12 Ha, 1e−10 e and the ordinary SCC iteration/mixer defaults. Dispersion can
be selected with `enable_dispersion`. All objectives use tolerances at least
as tight as 1e−12 Ha and 1e−10 e; stricter caller values are retained. The report
records both requested/effective controls, the actual solver selection and
physical conditions. Temperature is never tightened or changed with a solver
tolerance.

## Runnable synthetic property fit

Install the current source with `python -m pip install .`; use `.[ase]` for the
ASE example. The following script creates a nonzero user warm-start model, a
known synthetic teacher and analytic E/force/Hessian labels at 300 K. This is a
reproducible implementation check, **not independent reference chemistry**.
Replace its labels and provenance with consistently computed reference data
before drawing conclusions about predictive accuracy.

Save as `prepare-property-data.py` in your own working directory:

```python
import json
from pathlib import Path
from gfn1_rs import Gfn1NativeCalculator, create_ml_model

create_ml_model("initial.model", [1], [1.3, 2.2], cutoff_bohr=5.0)

def replace_head(text, head, bias, weights):
    prefix = f"head 1 {head} "
    return "\n".join(
        f"{prefix}{bias} {' '.join(map(str, weights))}" if line.startswith(prefix)
        else line for line in text.splitlines()
    ) + "\n"

text = Path("initial.model").read_text()
text = replace_head(text, 0, 0.0012, [0.003, -0.002])
text = replace_head(text, 3, 0.0003, [0.001, -0.0007])
text = text.replace("provenance untrained zero initialization",
                    "provenance Nonzero independent user warm-start teaching example")
Path("initial.model").write_text(text, encoding="utf-8")
Path("teacher.model").write_text(
    replace_head(text, 3, 0.0003, [0.009, -0.0007]).replace(
        "provenance Nonzero independent user warm-start teaching example",
        "provenance Known synthetic teacher for property-fitting implementation check"),
    encoding="utf-8")
teacher = Gfn1NativeCalculator(ml_model_path="teacher.model", charge=0,
    electronic_temperature=300.0, energy_tolerance=1e-12, charge_tolerance=1e-10)
samples = []
for split, distances in [("train", [0.67, 0.92]), ("validation", [0.79])]:
    for r in distances:
        pos = [[0., 0., 0.], [r, 0., 0.]]
        result = teacher.calculate([1, 1], pos, compute_gradient=True)
        forces = [[-x for x in row] for row in result.gradient_hartree_per_bohr]
        targets = [
            dict(observable="energy", unit="hartree", values=result.energy_hartree, scale=0.1),
            dict(observable="forces", unit="hartree/bohr", values=forces, scale=0.1),
            dict(observable="hessian", unit="hartree/bohr^2",
                 values=teacher.hessian([1, 1], pos), scale=0.1),
        ]
        samples.append(dict(id=f"{split}-{r}", split=split, charge=0,
            electronic_temperature_kelvin=300.0,
            xyz=f"2\nsynthetic analytic H2\nH 0 0 0\nH {r} 0 0\n", targets=targets))
data = dict(schema="gfn1-rs-property-training-v1", coordinate_units="angstrom",
    provenance="Known nonzero native ML teacher; implementation check",
    license_spdx="GPL-3.0-or-later", reference_method="synthetic native ML-xTB",
    independent_reference=False, samples=samples)
Path("properties.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
```

Run `python prepare-property-data.py`, then fit through the installed shared CLI:

```sh
gfn1-rs-train fit-properties --model initial.model --data properties.json --parameters H:correction:1 --output trained.model --report property-report.json --epochs 50 --learning-rate 0.1 --gradient-tolerance 1e-10 --loss-tolerance 1e-18 --threads 2 --memory-mb 2048
```

The selection syntax is `Z:HEAD:COORD`, comma-separated for multiple weights.
Element symbols or atomic numbers are accepted. HEAD is `s`, `p`, `d`,
`correction`, or 0/1/2/3; COORD 0 is the bias, COORD 1 is radial weight 0,
COORD 2 is radial weight 1, and so on. Coordinates outside the actual loaded
model or duplicate selections are rejected. For this example, only correction
radial weight 0 changes; the nonzero electronic heads and other weights remain.

This E/force/Hessian recipe was executed through the installed pip CLI: the
training loss decreased from 0.02710312 to 2.79e−19 and held-out loss to
1.51e−19, terminating at `loss_tolerance`. The Rust regression separately checks
the same nonzero model, unchanged unselected heads and saved-model reload.
Native/ASE training and model identity checks are also exercised. This verifies
an analytic-property round trip against a known synthetic model, not accuracy
against independent electronic-structure or experimental measurements.

Inspect `initial_train`, `final_train`, `initial_validation`, `final_validation`
and `termination` in the report. A completed short fit is not automatically
converged: possible reasons include `loss_tolerance`, `gradient_tolerance`,
`maximum_epochs`, `line_search_no_improvement` and
`zero_selected_parameter_sensitivity`. Trial failures are retained in the
epoch diagnostics, with no temperature/geometry change. `--parameter-step`
controls the central parameter step (default 1e−5 times max(1,|weight|));
`--regularization-to-initial` sets λ. `--maximum-parameter-step` bounds each
coordinate update. Backtracking accepts only a lower training objective.

## Rust API

```rust
use gfn1_rs::{Gfn1Parameters, ml::MlHamiltonian};
use gfn1_rs::property_training::{
    PropertyDataset, PropertyTrainingOptions, fit_properties,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut model, identity) = MlHamiltonian::resolve_with_identity("initial.model")?;
    println!("Warm-start weights: {}", identity.description());
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("properties.json")?)?;
    let dataset = PropertyDataset::from_json(&data)?;
    let settings = PropertyTrainingOptions::from_json(&serde_json::json!({
        "parameters": ["H:correction:1"], "max_epochs": 50,
        "learning_rate": 0.1, "gradient_tolerance": 1e-10, "loss_tolerance": 1e-18
    }))?;
    let context = gfn1_rs::resources::ResourceContext::new(
        gfn1_rs::resources::ResourceLimits::from_megabytes(Some(2), Some(2048))?)?;
    let report = context.run(|| fit_properties(
        &mut model, &Gfn1Parameters::builtin()?, &dataset, &settings))?;
    std::fs::write("trained.model", model.to_model_string()?)?;
    std::fs::write("property-report.json", serde_json::to_string_pretty(&report.as_json())?)?;
    println!("Saved weights: {}", model.canonical_sha256()?);
    Ok(())
}
```

`PropertySample`, `PropertyTarget`, `PropertyObservable` and
`TrainableParameter` also have typed Rust constructors/fields. The public
`evaluate_property_sample` and `property_registry` expose predicted target
dictionaries and the exact observable/backend/unit/JSON-pointer mapping.

## Python native and ASE

The two Python methods call the same Rust trainer in process. They load the
already selected user ML model and save a new model. Dataset conditions govern
each sample; attached ASE atoms and their default units do not override labels.
Load the saved model explicitly for subsequent inference.

```python
import json
from pathlib import Path
from gfn1_rs import Gfn1NativeCalculator

data = json.loads(Path("properties.json").read_text())
settings = dict(parameters=["H:correction:1"], max_epochs=50,
    learning_rate=0.1, gradient_tolerance=1e-10, loss_tolerance=1e-18)
calc = Gfn1NativeCalculator(ml_model_path="initial.model", threads=2, memory_mb=2048)
report = calc.fine_tune_properties(data, settings, output_model="native-trained.model")
print(report["final_train"], report["final_validation"])
trained = Gfn1NativeCalculator(ml_model_path="native-trained.model")
print(trained.ml_model_info())
```

```python
from gfn1_rs import GFN1RSCalculator

ase_calc = GFN1RSCalculator(ml_model_path="initial.model", threads=2, memory_mb=2048)
report = ase_calc.fine_tune_properties(data, settings, output_model="ase-trained.model")
```

As for inference, model load messages identify the path, original-file SHA-256,
canonical SHA-256, provenance and license. The training report links the parent
model identity and the saved model. These identities include the explicit
training provenance; a change in that metadata changes the canonical hash even
when a run terminates without a coordinate update.

## Executed periodic C and dielectric fit

The self-contained Rust regression also performs a **periodic** property fit,
using every component of `elastic_clamped` (36 entries) and
`dielectric_electronic` (nine entries). A nonzero synthetic H2 teacher changes
only s-head radial weight 0 from 0.003 to 0.0037 Ha. Its two fixed 5 Å cubic
cells have bond-coordinate x values 0.75 Å for training and 0.83 Å for held-out
validation, with y = 0.1 Å, Γ sampling, AO cutoff 12 bohr and electronic
temperature 300 K. Training-derived per-property scales are reused unchanged
for the held-out geometry. Neither geometry is relaxed inside the objective.

The unchanged central-parameter optimizer recovers 0.0037 Ha in five accepted
updates. Initial training RMSEs are 1.03646e−6 Ha/bohr³ for C and 2.40034e−5
for the relative dielectric; held-out RMSEs are 7.47791e−7 Ha/bohr³ and
2.62314e−5. Both tensor families on both geometries return to numerical
roundoff at the recovered known model. Saved-model reload gives identical
predictions, while unselected heads remain unchanged. This is a synthetic
model round trip, not a validation of hydrogen-crystal chemical accuracy.

Reproduce from the source root:

```sh
cargo test --profile reltest --test property_training periodic_clamped_elastic_and_dielectric_objectives_fine_tune_one_level_weight -- --nocapture
```

The [distributed measured report](../tests/data/ml-periodic-property-training.json)
retains complete conditions, losses, selected parameter changes and model
identities. Reproduction writes
`target/validation/v070/periodic-property-training-report.json`. All C/ε
predictions use analytic Rust physical response; the only finite differences
are of the selected model weight for the optimizer gradient.
