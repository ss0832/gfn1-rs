# Training and loading independent ML Hamiltonian weights

This page provides a complete path from reference data to a saved model and
inference through Rust, Python native and ASE. User-trained `.model` files are
supported by all three interfaces, using the same parser as builtin models.
Training and inference execute native Rust code; Python training helpers call
the shared Rust trainer in process and do not spawn a CLI executable.

The implemented model consists of smooth species/radial features, atomic
s/p/d Hamiltonian-level heads before SCC, and a separate energy-correction
head. It is this project's independent small architecture, not MACE-OFF or
the published ML-xTB authors' trained network. Arbitrary PyTorch/MACE checkpoint
files are not compatible with the native `.model` format. Global GFN1 tables
and basis exponents remain fixed; analytic forces include feature derivatives.

## Install and choose a data directory

Run the examples from the distributed source root, which contains `models/`.
For Python native and the installed CLI, install the source or a supplied wheel:

```sh
python -m pip install .
gfn1-rs --version
gfn1-rs-train --version
```

Building from source requires a Rust toolchain and a working native linker.
For ASE add `python -m pip install ".[ase]"`; native inference/training does not
require ASE or NumPy. Rust-only installation is `cargo install --path .`.
The command names after either installation are `gfn1-rs` and `gfn1-rs-train`.
CLI redirected output is UTF-8. When capturing it with Python on Windows, use
`subprocess.run(..., capture_output=True, text=True, encoding="utf-8")` rather
than the system's legacy code page. Native Windows canonical paths may carry
the `\\?\` long-path prefix; `Path(source).samefile(path)` checks file identity.

The runnable training example below uses the bundled, independent
`models/h2-fci-reference.json`: FCI/cc-pVTZ H2 energies, with held-out validation
and extrapolation structures. No external quantum-chemistry executable is
needed to fit those existing labels. Generating new reference labels is a
separate task; a tight-binding prediction is not an independent reference.

## Reference-data schema and units

The accepted schemas are `gfn1-rs-energy-training-v1` and
`gfn1-rs-observable-training-v2`. A dataset must contain:

| Field | Meaning |
| --- | --- |
| `schema` | One of the two identifiers above. |
| `coordinate_units` | `angstrom` or `bohr`, applying to inline XYZ coordinates. |
| `energy_units`, `energy_definition` | `hartree`, `mermin_free`. |
| `electronic_temperature_K` | Nonnegative electronic temperature used for predictions. |
| `reference_method` | Actual reference method and basis. |
| `provenance`, `license_spdx` | Origin and permitted use of the labels. |
| `independent_reference` | Explicit boolean; do not label synthetic GFN1 targets independent. |
| `samples` | Array of the structures and labels described below. |

Each sample contains inline `xyz`, `split` and optional unique `id`, `group`,
`charge`, plus one or more target labels. Splits are `train`, `validation`,
`extrapolation`, or `reference`. A sample's `group` defaults to `default`.
`energy_hartree` is one scalar; `gradient_hartree_per_bohr` is N by 3;
`forces_hartree_per_bohr` is the alternative N by 3 force label, with opposite
sign to the gradient. Supply only one nuclear-label representation.
`dipole_e_bohr` has three components. `dipole_origin_bohr`, if used, sets a
common origin for dipole labels. Those label units remain atomic units even
when `coordinate_units` is `angstrom`.

For a runnable small custom file, preserve the metadata and select actual
labelled samples from the supplied independent dataset:

```python
import json
from pathlib import Path
data = json.loads(Path("models/h2-fci-reference.json").read_text())
data["samples"] = ([s for s in data["samples"] if s["split"] == "train"][:3]
                   + [s for s in data["samples"] if s["split"] == "validation"][:1])
Path("small-h2.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
```

For your own system, replace the geometries, labels, method and provenance by
your own consistent reference calculation. Split by physically distinct
structures before training. Do not fit on validation/extrapolation labels.
An explicit `relative_energies` list supports reactions or interaction energies:
each entry has `split`, `group`, `energy_hartree` and `terms` containing sample
`id` and `coefficient`. Shared monomer anchors belong to the `reference` split;
training cases cannot reference held-out samples. The complete
`models/water-clusters-reference.json` illustrates this observable schema.

## Fit through the installed CLI

The following fits the full H2 dataset with energy labels only. Setting both
unused observable weights to zero is explicit; it avoids accidentally training
on gradients when reproducing the energy-only fit.

```sh
gfn1-rs-train new --species H --centers 1.2,2.5 --cutoff 6 --output my-h2-initial.model
gfn1-rs-train fit --model my-h2-initial.model --data models/h2-fci-reference.json --output my-h2.model --report my-h2-report.json --epochs 2000 --optimizer adamw --learning-rate 0.0007 --weight-decay 0.00001 --train-correction --energy-weight 1 --gradient-weight 0 --dipole-weight 0 --relative-energy-weight 0 --log-file my-h2-training.jsonl
gfn1-rs examples/h2.xyz --ml-model my-h2.model --gradient --log-file my-h2-calculation.jsonl
```

If `examples/h2.xyz` is not present, create it as:

```text
2
H2; coordinates in angstrom
H 0 0 0
H 0.74 0 0
```

The model output contains the learned heads, radial basis settings, units,
license and training provenance including the dataset SHA-256. The JSON report
records optimizer settings, input/output hashes, loss history and separate
train/validation/extrapolation metrics. Check `status == "complete"`. A short
run with `--epochs 5` is an API smoke test, not evidence of a useful fit.

For observable training, choose nonzero `--gradient-weight`, `--dipole-weight`
and/or `--relative-energy-weight` and provide matching labels. The water recipe
uses energy weight 0, relative-energy weight 10, gradient weight 50 and dipole
weight 0.02 with the bundled water dataset. Loss units differ, so those weights
are not universal. Each target family is normalized by its active sample/group
count and nuclear losses by 3N components. `--group-ratios NAME:FACTOR,...`
changes subset weighting. AdamW and SGD are available; `--energy-loss` accepts
`squared`, `absolute` or `huber` with `--huber-delta` in Hartree. `--wtmad2`
requires explicit relative-energy cases; it is not for raw atomic total energies.
`--max-model-bytes` bounds the serialized output. Gradient/dipole fitting uses
gapped integer-occupation molecular CP-SCC; finite-T fractional/open-shell and
periodic training are not supplied by this dataset workflow.

## Rust: training, saving, loading and identification

From the distributed source root, create a small client project:

```sh
cargo new --bin --vcs none ml-training-example
```

Add this dependency to its `Cargo.toml` under `[dependencies]`:

```toml
gfn1-rs = { path = ".." }
```

Replace `ml-training-example/src/main.rs` with the following code. Run it from
the **distributed source root** so the dataset path resolves:
`cargo run --release --manifest-path ml-training-example/Cargo.toml`.
The learned files and logs are written in that current directory.

The public shared training workflow accepts structured argument strings,
returns `Result`, and executes directly in the calling process:

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, run_electronic};
use gfn1_rs::ml::MlHamiltonian;
use std::sync::Arc;

fn main() -> gfn1_rs::Result<()> {
    gfn1_rs::cli::train_model([
        "new", "--species", "H", "--centers", "1.2,2.5", "--cutoff", "6",
        "--output", "my-h2-initial.model",
    ].into_iter().map(str::to_string).collect())?;
    gfn1_rs::cli::train_model([
        "fit", "--model", "my-h2-initial.model", "--data", "models/h2-fci-reference.json",
        "--output", "my-h2.model", "--report", "my-h2-report.json",
        "--epochs", "2000", "--learning-rate", "0.0007", "--weight-decay", "0.00001",
        "--train-correction", "--energy-weight", "1", "--gradient-weight", "0",
        "--dipole-weight", "0", "--relative-energy-weight", "0",
    ].into_iter().map(str::to_string).collect())?;
    let (model, identity) = MlHamiltonian::resolve_with_identity("my-h2.model")?;
    println!("ML weights: {}", identity.description());
    let logger = gfn1_rs::CalculationLogger::append(
        "my-h2-rust.jsonl", gfn1_rs::LogLevel::Info, true)?;
    logger.event(gfn1_rs::LogLevel::Info, "ml_model_loaded", identity.as_json())?;
    let options = ElectronicOptions {
        ml_model: Some(Arc::new(model)), logger: Some(logger),
        electronic_temperature: 0.0, ..Default::default()
    };
    let system = PeriodicSystem::from_xyz_str("2\nH2\nH 0 0 0\nH .74 0 0\n", 0.0, false)?;
    let result = run_electronic(&system, &Gfn1Parameters::builtin()?, options)?;
    println!("{} Hartree", result.total_free);
    Ok(())
}
```

For typed in-memory training, use `MlHamiltonian::zero`,
`training::{TrainingExample, TrainingConfig, fit_observables}` and save using
`model.to_model_string()`. `fit_observables` optimizes the supplied **training**
examples; manage held-out batches explicitly. The shared file workflow above
also performs schema validation, split checks and the validation report.

## Python native: training and loading user weights

```python
from gfn1_rs.native import Gfn1NativeCalculator, create_ml_model, train_ml_model

create_ml_model("my-h2-initial.model", species=["H"], centers_bohr=[1.2, 2.5], cutoff_bohr=6.)
report = train_ml_model(
    "my-h2-initial.model", "models/h2-fci-reference.json", "my-h2.model", "my-h2-report.json",
    epochs=2000, learning_rate=.0007, optimizer="adamw", weight_decay=.00001,
    train_correction=True, energy_weight=1, gradient_weight=0, dipole_weight=0,
    relative_energy_weight=0, log_file="my-h2-training.jsonl")
assert report["status"] == "complete"
calc = Gfn1NativeCalculator(ml_model_path="my-h2.model", electronic_temperature=0.,
                           log_file="my-h2-native.jsonl", log_console=True)
print(calc.ml_model_info())
result = calc.calculate([1, 1], [[0, 0, 0], [.74, 0, 0]], compute_gradient=True)
print(result.energy_hartree, result.gradient_hartree_per_bohr)
```

The model path is read at calculator construction. To use newly trained weights,
construct a new calculator; modifying a file does not silently alter an existing
calculator's Hamiltonian. Unknown builtin names, unreadable paths, incompatible
formats and missing species are errors; there is no fallback to builtin weights.

## ASE: training and loading user weights

ASE uses the same in-process training helper and the same saved model. It does
not define a second optimizer or silently convert reference-data units:

```python
import logging
from ase import Atoms
from gfn1_rs import GFN1RSCalculator, create_ml_model, train_ml_model

# Skip these two calls if my-h2.model was already trained above.
create_ml_model("my-h2-initial.model", species=["H"],
                centers_bohr=[1.2, 2.5], cutoff_bohr=6.)
report = train_ml_model(
    "my-h2-initial.model", "models/h2-fci-reference.json", "my-h2.model", "my-h2-report.json",
    epochs=2000, learning_rate=.0007, optimizer="adamw", weight_decay=.00001,
    train_correction=True, energy_weight=1, gradient_weight=0, dipole_weight=0,
    relative_energy_weight=0, log_file="my-h2-training.jsonl")
assert report["status"] == "complete"
logging.basicConfig(level=logging.INFO)
atoms = Atoms("H2", positions=[[0, 0, 0], [.74, 0, 0]])
atoms.calc = GFN1RSCalculator(ml_model_path="my-h2.model", electronic_temperature=0.,
                             log_file="my-h2-ase.jsonl", log_console=True)
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/angstrom
print(atoms.calc._native.ml_model_info())  # file/builtin source and both hashes
print(atoms.calc.results["ml_model_sha256"])
```

## Verify which weights were used

The calculation CLI prints `ml_model` followed by the resolved path or builtin
identifier, source-byte SHA-256, canonical-weight SHA-256, provenance and license.
With file logging enabled, `ml_model_loaded` records the same information as
structured JSON. Python native exposes `ml_model_info()` and
`ml_model_source()`; `log_console=True` also prints the identity. The ASE adapter
logs it through the `gfn1_rs` Python logger and forwards native logging options.

The CLI displays the identity on every ML calculation. In Python, choose
`log_console=True` for immediate stderr output, `log_file=...` for a persistent
JSONL record, or call `ml_model_info()` explicitly. A JSONL load event has
`event="ml_model_loaded"`, with `data.source`, `data.source_sha256`,
`data.canonical_sha256`, `data.training_provenance` and `data.license_spdx`.
Console logging is opt-in for library calls so applications can control their
own output. The examples above enable it.

`source_sha256` identifies the exact loaded bytes. `canonical_sha256` identifies
the parsed model serialized in canonical form; formatting-only changes can alter
the former without altering the latter. `result.ml_model_sha256` and ASE's
`results["ml_model_sha256"]` contain the canonical-weight hash. When ML is inactive,
the native identity is `None`. A trained file and a builtin identifier can load
identical weights; their source identity still makes that choice explicit.

Builtin identifiers remain `builtin:h2-demo`, `builtin:h2-fci` and
`builtin:water-clusters`. Their domains and independently reproduced validation
metrics are in [parameters.md](parameters.md). Fitting a model does not establish
accuracy outside its training domain; read the held-out and extrapolation metrics.

CPU worker counts and checked workspace budgets also apply to training:
`gfn1-rs-train fit ... --threads 4 --memory-mb 2048`, or
`train_ml_model(..., threads=4, memory_mb=2048)`.
Inference uses the matching calculator options; see [resources.md](resources.md).
