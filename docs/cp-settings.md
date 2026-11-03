# CP equation convergence controls

Rust, both installed commands, Python native and ASE configure coupled-
perturbed (CP/CPXTB/DFPT) response tolerances. Settings follow the calculation's
private worker pool, so calculators can use different thresholds concurrently.

| Python keyword / CLI flag | Rust `CpSettings` field | Meaning |
| --- | --- | --- |
| `cp_tolerance` / `--cp-tolerance` | `absolute_tolerance` | Absolute linear-equation residual bound. |
| `cp_relative_tolerance` / `--cp-relative-tolerance` | `relative_tolerance` | Bound relative to the equation's RHS/response scale. |
| `cp_max_iter` / `--cp-max-iter` | `max_iterations` | Maximum Krylov iterations on iterative routes. |

Supplied values must be finite and positive; the iteration limit is an integer.
`None` preserves each route's defaults. The acceptance bound is
`max(absolute_tolerance, relative_tolerance * scale)`. Set both tolerances when
imposing a tighter combined criterion. Pair-space solves use an L2 residual;
dielectric solves use a maximum component norm; complex periodic solvers use
their Brillouin-zone weighted norm. Molecular spinor CP-SCC uses an L2
residual with relative scale `||A||_F*||q||_2 + ||rhs||_2`.

Molecular direct pair-space defaults are `max(1e-8, 1e-10*||rhs||)`. Molecular
PCG defaults to `1e-8*max(1,||rhs||)` and 100 iterations. Periodic complex CP
uses `1e-10*max(1,||rhs||)` with a dimension-dependent iteration limit; periodic
TDA uses `1e-11`. The periodic thermal dielectric uses `1e-9*scale`.

Direct solves have zero Krylov iterations. Their residual checks use the
configured tolerances; `cp_max_iter` applies to iterative solves. Molecular
first-, second- and third-order charge-space responses apply configured
residual checks. Periodic CP/TDA and spinor CP-SCC check their own equations.
An unsatisfied configured tolerance produces an error instead of false
convergence.

## Rust

```rust
use gfn1_rs::{Gfn1Parameters, PeriodicSystem, AnalyticHessianOptions, analytic_hessian};
use gfn1_rs::resources::{CpSettings, ResourceContext, ResourceLimits};

let molecule = PeriodicSystem::from_xyz_str(
    "3\nwater\nO 0 0 0\nH .757 .586 0\nH -.757 .586 0\n", 0.0, false)?;
let parameters = Gfn1Parameters::builtin()?;
let cp = CpSettings {
    absolute_tolerance: Some(1e-10), relative_tolerance: Some(1e-10),
    max_iterations: Some(500),
};
let context = ResourceContext::with_cp(
    ResourceLimits::from_megabytes(Some(4), Some(2048))?, cp)?;
let result = context.run(|| analytic_hessian(
    &molecule, &parameters, AnalyticHessianOptions::default()))?;
println!("{}", result.hessian[(0, 0)]);
```

Reuse the context for repeated geometries and higher derivatives. `cp.run(||
...)` is a one-shot shorthand without explicit CPU/memory settings.
`resources::current_cp()` reports the active overrides. Existing Rust
`CpxtbOptions { tol, max_iter }` calls remain available; scoped settings also
reach entry points that construct response options internally.

## Python native

```python
from gfn1_rs import Gfn1NativeCalculator
calc = Gfn1NativeCalculator(
    electronic_temperature=300.0, energy_tolerance=1e-13,
    charge_tolerance=1e-11, cp_tolerance=1e-10,
    cp_relative_tolerance=1e-10, cp_max_iter=500,
    threads=4, memory_mb=2048, log_file="response.jsonl", log_console=True)
print(calc.cp_settings())
hessian = calc.hessian([8, 1, 1], [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]])
```

The calculator settings apply to its molecular/periodic response and higher
derivative calls. Console output identifies overrides. Electronic JSONL entries
retain them under `data.configuration.cp_settings`.

## ASE

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator
atoms = Atoms("OH2", positions=[[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]])
atoms.calc = GFN1RSCalculator(cp_tolerance=1e-10,
    cp_relative_tolerance=1e-10, cp_max_iter=500, electronic_temperature=300.0)
hessian = atoms.calc.get_hessian(atoms=atoms)  # eV/angstrom²
atoms.calc.set(cp_max_iter=800)  # rebuilds context and invalidates caches
print(atoms.calc._native.cp_settings())
```

## Commands and scope

```console
gfn1-rs water.xyz --hessian --cp-tolerance 1e-10 --cp-relative-tolerance 1e-10 --cp-max-iter 500
gfn1-rs-train fit --model initial.model --data dataset.json --output trained.model --cp-tolerance 1e-10 --cp-relative-tolerance 1e-10 --cp-max-iter 500
```

Flags are shared by Cargo binaries and pip scripts. Tight response accuracy
also requires a converged reference SCF. These controls do not change SCF
tolerances, electronic temperature or orbital/configuration selection.
Dedicated sTDA orbital derivatives are explicit generalized-eigenproblem
derivatives; they have no Krylov iteration. The GFN1 reference CP response of
a composite sTDA Hessian uses these settings.
Uncoupled closed-form magnetic orbital response also has no Krylov iteration.

Tests exercise a known CP equation's exact solution and iteration failure,
real water Hessians with normal/unattainable thresholds, worker propagation,
logging, ASE and both installed commands.
