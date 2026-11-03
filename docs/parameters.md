# Parameters

## The official parametrization is bundled

Since v0.5.0 the official GFN1-xTB parameter files ship **inside the crate** — no
download, no environment variable, nothing to configure. They are compiled in with
`include_str!` from `third_party/xtb/`:

| Constant (`gfn1_rs::params`) | File | Parametrization |
| --- | --- | --- |
| `BUILTIN_GFN1_PARAM_TEXT` | `third_party/xtb/param_gfn1-xtb.txt` | GFN1-xTB |
| `BUILTIN_GFN1_SI_PARAM_TEXT` | `third_party/xtb/param_gfn1-si-xtb.txt` | GFN1(Si)-xTB (silicon reparametrization) |

Both are **verbatim** (LF-normalized) copies from
[`grimme-lab/xtb`](https://github.com/grimme-lab/xtb)`@2b5cd48`, LGPL-3.0-or-later.
Redistribution inside this GPL-3.0-or-later project is per LGPLv3. Provenance and
licence texts live in `third_party/xtb/PROVENANCE.md`, `COPYING`, `COPYING.LESSER`,
and in the top-level [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### References — and where the lanthanoids ("Ln-xTB") live

| Parametrization | Reference |
| --- | --- |
| GFN1-xTB | S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**, 1989 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118) |
| Ln f-in-core (La–Lu) | M. Bursch, A. Hansen, S. Grimme, *Inorg. Chem.* **56**, 12485 (2017), DOI [10.1021/acs.inorgchem.7b01950](https://doi.org/10.1021/acs.inorgchem.7b01950) |
| GFN1(Si)-xTB | *J. Chem. Inf. Model.*, DOI [10.1021/acs.jcim.1c01170](https://doi.org/10.1021/acs.jcim.1c01170) |

There is **no separate Ln-xTB parameter file** upstream: the f-in-core
lanthanoid parameters were merged into the standard `param_gfn1-xtb.txt`
(`$Z=57`…`$Z=71` blocks are present in the bundled copy), so lanthanoid
complexes work out of the box with the default builtin set. The same DOIs are
carried at runtime by `Gfn1Parameters::references()` and appear in the
provenance banner (`source_description()`, the CLI startup line, and the
Python `param_source()` / `__repr__`).

The one-line provenance tag is the public constant

```rust
pub const BUILTIN_PARAM_PROVENANCE: &str = "grimme-lab/xtb@2b5cd48, LGPL-3.0-or-later";
```

## Resolution order

`Gfn1Parameters::resolve` is the single entry point every front end uses:

```rust
use gfn1_rs::Gfn1Parameters;

let params = Gfn1Parameters::resolve(None)?;                     // bundled GFN1-xTB
let params = Gfn1Parameters::resolve(Some("builtin:si"))?;       // bundled GFN1(Si)-xTB
let params = Gfn1Parameters::resolve(Some("my_param.txt"))?;     // explicit file
```

Precedence, highest first:

1. the **explicit spec** (`--param` / `param_path=` / the `explicit` argument);
2. the **`GFN1_XTB_PARAM`** environment variable (constant `GFN1_PARAM_ENV`);
3. the **builtin** GFN1-xTB set.

The explicit spec is either a file path or one of the builtin specifiers, matched
case-insensitively:

| Spec | Selects |
| --- | --- |
| `builtin`, `builtin:gfn1` | bundled GFN1-xTB |
| `builtin:si`, `builtin:gfn1-si`, `builtin-si` | bundled GFN1(Si)-xTB |
| anything else | treated as a file path (`Gfn1Parameters::from_file`) |

Lower-level constructors remain public: `Gfn1Parameters::builtin()`,
`Gfn1Parameters::builtin_si()`, `Gfn1Parameters::from_file(path)`, and
`Gfn1Parameters::from_str(text)`.

> **`resolve_param_path` is not the same function.** It is the legacy *path-only*
> resolver (explicit string, else `GFN1_XTB_PARAM`) and it still **errors** when
> neither is set — it has no builtin fallback. New code should call
> `Gfn1Parameters::resolve`.

GFN2 parameter files are rejected at parse time, deliberately.

## Provenance reporting

Every loaded parameter set records where it came from in `ParamSource`:

```rust
pub enum ParamSource {
    Builtin,            // bundled GFN1-xTB
    BuiltinSi,          // bundled GFN1(Si)-xTB
    EnvVar(String),     // file named by GFN1_XTB_PARAM
    Explicit(String),   // file named on the command line / API
    Inline,             // parsed from an in-memory string
}
```

`Gfn1Parameters::source_description()` renders the one-liner used by the front ends:

- **CLI** prints it at startup: `parameters: GFN1-xTB (builtin, grimme-lab/xtb@2b5cd48, LGPL-3.0-or-later)`
- **Python**: `calc.param_source()`, and it appears in `repr(calc)`.
- **ASE**: `GFN1RSCalculator` logs it through the `gfn1_rs` logger.

Python's `default_param_path()` returns `$GFN1_XTB_PARAM` when set and the literal
string `"builtin"` otherwise, so it is always a valid `param_path=` argument and
never raises.

## Parameter round-trip and derivatives

`Gfn1Parameters::to_param_string()` / `write_param_file()` emit the
`param_gfn1-xtb.txt` format deterministically with value-exact round-trip.
Individual scalars are addressed with `ParameterTarget` (`glob:<key>`,
`elem:<Z>:<KEY>[:idx]`, `pair:<ZA>:<ZB>`); see
[rust-api.md](rust-api.md#parameters-round-trip-and-finite-difference-derivatives).

Since v0.5.0 the **halogen-bond term reads its constants from the parameter file**
(`xbdamp`, `xbrad` globals and the per-element `CXB`, scaled by 0.1) instead of
hardcoding them. Editing those parameters previously had no effect and their
parameter derivatives were silently zero. Physics is unchanged with the official
parametrization — the old hardcoded constants equal the builtin values.

## Reference data for the dispersion terms

| Data | Bundled at | Override |
| --- | --- | --- |
| D3(BJ) `s-dftd3` reference table | `third_party/simple-dftd3` (LGPL-3.0-or-later) | `--d3-reference PATH` / `GFN1_D3_REFERENCE` (`GFN1_D3_REFERENCE_ENV`) |
| DFT-D4 reference data (experimental D4 path) | `third_party/dftd4` | — |
| tblite atomic spin constants `W` (spGFN1) | `third_party/tblite/spin.f90` (LGPL-3.0-or-later) | — |
| Dunning-derived node-selected secondary bases | `secondary_bases` (BSE 0.12 BSD-3-Clause plus GFN1 H STO LGPL-3.0-or-later) | `--multipole-secondary-basis <name\|file>` |

D4 damping constants `a1`, `a2`, `s8` are read from the **active** GFN1 parameter
file; `s9` is an API/CLI option that defaults to the GFN2-xTB value `5.0` only when
D4 is active (ordinary non-D4 runs use `s9 = 0`).

Secondary-basis numerical provenance is recorded in
`tests/data/secondary-basis-provenance.json`: every distributed contraction was
matched at its eight-decimal precision, with original numerical body bytes
preserved. The TZ/QZ/5Z subsets retain seven cc-pVDZ d-polarization functions
for Ne and Al-Ar; their labels do not denote complete unmodified Dunning basis
sets. Hydrogen uses reconstructed GFN1 primary STO functions. Heavy PP subsets
remain model extensions; a provenance check is not a new M1 accuracy calibration.

## Unit constants (reporting vs model)

`constants::HARTREE_TO_EV` / `EV_TO_HARTREE` are CODATA-2018
(`27.211386245988`) and are used for **unit reporting only**. The model-side
conversion that reads the parameter file (`basis::EV_TO_HARTREE`) deliberately
stays on the legacy `1/27.21138505`: GFN1 was parametrized against it, and
switching it shifts caffeine by ~2e-6 Eh and breaks tblite parity. The split is
documented on both constants.

`constants::KB_HARTREE_PER_K = 3.1668115634556e-6` (exact SI `k_B` over the
CODATA-2018 Hartree) is the **single** Boltzmann constant for every
finite-temperature path — restricted SCC, spin-polarized SCC, CPXTB, periodic
SCC/Hessian, and the Grüneisen heat capacities. Before v0.5.0 the restricted and
spin paths used two slightly different values, breaking their byte-identity at
`T > 0`.

## Independent structural ML models

See [the complete training and loading guide](ml-training.md) for executable
Rust, Python native and ASE workflows, including user-trained weights and
load-time model identity logging.

`MlHamiltonian::resolve` accepts a model file or `builtin:h2-demo`,
`builtin:h2-fci`, `builtin:water-clusters`. Native smooth radial heads shift
structure-local s/p/d levels **before** SCC and supply a separate correction
energy. Global GFN1 tables and basis exponents remain fixed. All structure,
force and response chains include the geometry dependence of these heads.
The architecture, data and weights are independent of MACE-OFF; the models are
small domain demonstrations. Exact byte hashes and data/weight licenses are in
`third_party/assets.json` and `THIRD_PARTY_NOTICES.md`.

### Independent H2 reference reproduction

All 23 H2 FCI/cc-pVTZ energies and analytic gradients were
regenerated with PySCF 2.14.0, and the inherited `builtin:h2-fci` model was
reevaluated without refitting. On the nine held-out structures, absolute-energy
RMSE was `0.13793698 -> 0.00324967` Eh. After aligning each method using only
its mean training residual, shape RMSE was `0.01052147 -> 0.00326639` Eh.
The four extrapolation structures have worse aligned shape error for the model
(`0.05579471` versus baseline `0.04453734` Eh); the training domain remains
0.50–1.90 angstrom. Exact reproduction errors, hashes and all split metrics are
in [`tests/data/h2-reference-reproduction.json`](../tests/data/h2-reference-reproduction.json).

### Water-cluster example (2,869 bytes)

```bash
cargo run --profile reltest --bin gfn1-rs -- examples/water.xyz --ml-model builtin:water-clusters --gradient
```

`models/water-clusters-reference.json` contains original constructed neutral
monomers, dimers, cyclic trimers and tetramers, independently calculated using
PySCF 2.14.0 **PBE0/def2-SVP**. Labels are analytic nuclear gradients including
DFT grid response, total electronic-plus-nuclear dipoles at the zero origin,
and explicit distortion/interaction energies relative to one isolated rigid
water anchor. Full all-electron energies are retained as metadata rather than
fitted against a tight-binding energy origin. Interactions are finite-basis
supermolecular values without counterpoise or dispersion corrections.

The deterministic split has 14 training, 8 validation, 5 extrapolation
structures and 1 shared reference anchor. **Every tetramer is held out.**
The 250-epoch AdamW model is embedded as `builtin:water-clusters`; its text is
2,869 bytes, well within a 5,000,000-byte model budget. Reference data and the
training report are separate distribution assets.

All 28 PBE0/def2-SVP structures were independently regenerated,
including analytic DFT-grid-response gradients and dipoles. Reevaluation of the
unchanged 2,869-byte model on the eight held-out structures again improved all
three observables. The reproduction errors, dataset/model hashes and every
split's metrics are recorded in
[`tests/data/water-reference-reproduction.json`](../tests/data/water-reference-reproduction.json).

Measured held-out validation RMSE over the eight structures:

| Quantity | Standard GFN1 | Trained model | Units |
| --- | ---: | ---: | --- |
| Distortion/interaction energy | 0.01378675 | 0.00757978 | Eh |
| Nuclear-gradient components | 0.00530555 | 0.00297584 | Eh/bohr |
| Dipole components | 0.05062715 | 0.02957005 | e bohr |

`models/water-clusters-trained-report.json` includes metrics for each cluster
size and extrapolation split, plus effective loss weights. The entirely unseen
validation tetramers have energy RMSE 0.02487889 -> 0.01514205 Eh and gradient
RMSE 0.00540023 -> 0.00349566 Eh/bohr. Their dipoles vanish by the constructed
symmetry. Extrapolated dimer/trimer energies and the compressed/stretched
tetramer gradient do not uniformly improve; the report retains these results.
These are initial-domain acceptance measurements, not general liquid-water,
crystalline-ice or complete-basis binding accuracy.

Reproduce the reference data (requires PySCF, NumPy and SciPy) and training:

```bash
python scripts/generate_water_clusters.py --output water-reference.json --threads 2
gfn1-rs-train new --species H,O --centers 1.6,2.0,2.8,3.6,4.6,5.4,6.2,7.2,8.0 --width 0.8 --cutoff 9 --output water-initial.model
gfn1-rs-train fit --model water-initial.model --data models/water-clusters-reference.json --output water-trained.model --report water-report.json --epochs 250 --learning-rate 0.0002 --weight-decay 0.00001 --train-correction --energy-weight 0 --relative-energy-weight 10 --gradient-weight 50 --dipole-weight 0.02 --max-model-bytes 5000000
```

The distributed data hash is
`071e215e35a7dc48a63e708558d10ad780db5cb227203187f173f2dafe15e2ca`;
the model hash is
`5450b55436e10608a22064cfeb03541865d7efd8d9ab4f56d450b36514bbf819`.
Floating-point differences between reference engines/toolchains can change
regenerated bytes; use the distributed references to reproduce fitting.

### Observable training CLI and schema

Build/install the separate `gfn1-rs-train` binary. `new` creates zero heads;
`fit` trains a model; `demo-h2` generates synthetic Hamiltonian targets.
Energy-only compatibility schema `gfn1-rs-energy-training-v1` and observable
schema `gfn1-rs-observable-training-v2` both require explicit energy/coordinate
units, electronic temperature, reference method, provenance, license and
independence declaration. The distributed water dataset is a complete example.

Each sample has an ID, inline XYZ, split and optional group, charge, absolute
`energy_hartree`, `gradient_hartree_per_bohr` **or**
`forces_hartree_per_bohr`, and `dipole_e_bohr`. Nuclear labels have one 3-vector
per atom; forces are negated to gradients. `dipole_origin_bohr` sets the common
origin. Splits are `train`, `validation`, `extrapolation`, `reference`.
`relative_energies` cases contain their own split/group, `energy_hartree`, and
`terms` with sample IDs and coefficients. Shared anchors belong to `reference`;
their direct labels are stripped from every measured batch. Training reactions
cannot reference held-out validation/extrapolation structures.

- Observable weights: `--energy-weight`, `--gradient-weight`,
  `--dipole-weight`, `--relative-energy-weight`. Each target family is
  normalized by its active sample group-ratio sum; gradients average over
  3N components and dipoles over 3 components.
- Optimizer: `--optimizer adamw|sgd`, `--epochs`, `--learning-rate`,
  `--weight-decay`; AdamW defaults beta1=0.9, beta2=0.999, epsilon=1e-8.
  `--train-correction` updates the correction head; `--decay-bias` includes
  biases in decoupled weight decay. Loss gradients use stationary/implicit
  CP-SCC derivatives, without storing SCF iteration histories.
- Energy loss: `--energy-loss squared|absolute|huber` and
  `--huber-delta` in Hartree. `--group-ratios small:0.25,large:2` controls
  relative contributions; zero removes a subset from the loss.
- `--wtmad2` (or `--weighting wtmad2`) selects relative/reaction scaling
  `56.84 / mean(abs(reference energies) in kcal/mol)`. It defaults to absolute
  loss and disables absolute total-energy loss. Cases must be explicitly
  defined; this is not meaningful for raw all-electron total energies.
  Optional `groups.NAME.mean_abs_reference_energy_kcal_mol` supplies fixed
  reference subset means; otherwise fitting derives them from training cases.
  WTMAD-2 metrics use the provided means or the measured split's own reference
  means. These demonstration subsets do not constitute a GMTKN55 benchmark.
- `--max-model-bytes` checks the serialized trained model. `--report` records
  input/model/parameter hashes, settings, loss components, effective weights,
  target counts, validation, extrapolation and per-sample-group diagnostics.
  Training history is evaluated before each optimizer update; final model
  accuracy is measured separately. Absolute energy-shape alignment, when
  applicable, uses training residuals only.

Gradient/dipole training currently requires gapped integer occupations through
the molecular CP-SCC reference. Actual fallback-temperature changes are errors.
For the distributed **energy-only** H2 FCI recipe use `--gradient-weight 0
--dipole-weight 0`, AdamW 2,000 epochs, learning rate 0.0007, decay 0.00001,
`--train-correction`, and a zero H model with centers 1.2,2.5 and cutoff 6.

### Operational file logging

Calculation and training CLIs accept `--log-file calculation.jsonl` and
`--log-level error|info|debug|trace`. Append-only JSONL records include session,
operation and parent IDs, SCF convergence, fallbacks, property boundaries,
training epoch losses, elapsed time and failures. Rust uses
`ElectronicOptions.logger = Some(CalculationLogger::append(...))`; Python uses
`log_file`, `log_level`, `log_console`. Calculation results expose
`calculation_id`. Default logging is disabled. Operational error messages redact
path/private tokens; application-authored custom event payloads remain the
caller's responsibility.
