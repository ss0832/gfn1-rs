# CPU workers and calculation workspace

Rust, installed commands, Python native and ASE accept a worker count and a
workspace budget. A reusable private Rayon pool controls calculation workers,
including real/complex faer eigensolvers and matrix products. Independent
calculators can use different pools. No process environment variable is changed.
`GFN1_FAER_THREADS=1` remains a sequential default for dense algebra when no
explicit thread count is supplied.

CP residual tolerances and iterative limits can share this same calculation
scope; see [cp-settings.md](cp-settings.md).

The memory setting constrains **checked calculation-workspace estimates**, not
process RSS. Python/ASE objects, allocator overhead, model files and results
retained by the caller are outside those estimates. Checks cover dense AO
arrays, eigensolver/GEMM/LU workspaces, molecular/periodic DFPT response, packed
FC3/FC4 outputs and reference plans, commensurate phonons and native sTDA caches.
Not every small temporary is individually tracked. Conservative estimates can
reject a calculation whose eventual resident memory would be smaller.

Existing method-specific `memory_budget_bytes` combine with the calculation
budget by taking their minimum. A larger per-call budget cannot override the
calculator budget. Use blocks or directional contractions when a full tensor
does not fit. Resource selection never changes temperatures or tolerances.

## Rust

```rust
use gfn1_rs::resources::{ResourceContext, ResourceLimits};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, run_electronic};

let params = Gfn1Parameters::builtin()?;
let system = PeriodicSystem::from_xyz_str(
    "2\nH2\nH 0 0 0\nH 0 0 0.75\n", 0.0, false)?;
let context = ResourceContext::new(ResourceLimits::from_megabytes(
    Some(4), Some(2048))?)?;
let result = context.run(|| run_electronic(
    &system, &params, ElectronicOptions::default()))?;
println!("{} Hartree", result.total_free);
```

Reuse the context for repeated geometries. `ResourceLimits::run` is the one-shot
equivalent. Nested scopes can tighten outer limits. `None` uses the method/Rayon
default; counts and budgets must be positive. `resources::current()` reports
active limits and `effective_threads()` the visible worker count.

## Installed commands

```console
gfn1-rs water.xyz --gradient --threads 4 --memory-mb 2048
gfn1-rs-train fit --model initial.model --data dataset.json --output trained.model --threads 4 --memory-mb 2048
```

Both Cargo binaries and pip console scripts use these options. Memory uses
binary MiB (`1024²` bytes). Explicit configuration prints effective workers and
workspace bytes to stderr. Invalid options or an insufficient checked budget
produce an error exit status.

The in-process Python training helper accepts the same settings as keywords:
`train_ml_model(..., threads=4, memory_mb=2048)`. They use the shared Rust
training scope without starting an external calculation program.

## Python native

```python
from gfn1_rs import Gfn1NativeCalculator

calc = Gfn1NativeCalculator(threads=4, memory_mb=2048,
                           electronic_temperature=300.0,
                           log_file="calculation.jsonl", log_console=True)
print(calc.resource_limits())
nums = [8, 1, 1]
pos = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]
result = calc.calculate(nums, pos)
hessian = calc.hessian(nums, pos)
```

Every public calculator method uses the same Rust scope, including advanced
magnetic, molecular/periodic derivative and excitation APIs. Existing method
signatures remain available to `inspect.signature`. The `gfn1_rs` Python logger
reports configuration; `log_console=True` also prints it to stderr. Electronic
calculation JSONL entries include worker count and workspace bytes.

## ASE

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator

atoms = Atoms("OH2", positions=[[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]])
atoms.calc = GFN1RSCalculator(threads=4, memory_mb=2048,
                             electronic_temperature=300.0, log_console=True)
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # eV/Angstrom
print(atoms.calc._native.resource_limits())
```

Property and advanced calls delegate to the native scope. Changing `threads`
or `memory_mb` through `atoms.calc.set(...)` rebuilds the native context and
invalidates ASE's cached results.

## Verification

Rust resource tests exercise concurrent one/two-worker pools, propagation to
worker tasks, nested limits, faer eigensolves after differently configured
calls, and budget rejection before reference SCF. Python tests compare real
energies/forces between worker counts, native/ASE forwarding, preserved method
signatures, JSONL metadata, budget rejection and installed command behavior.
