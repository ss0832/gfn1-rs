# Periodic SCC convergence

The default `auto` accelerator starts with charge Broyden mixing and switches
to a globalized analytic charge Newton method when progress stalls. Geometry,
electronic temperature, charge, k mesh and convergence tolerances remain fixed.
This also applies to structural ML Hamiltonian heads. Explicit `linear`,
`broyden`, `cdiis` and `newton` select their actual periodic algorithms.

`auto` preserves the existing molecular Broyden driver and its safeguards.
The legacy `scc_broyden=False` / `--no-broyden` selects linear mixing when the
accelerator is `auto` or `broyden`. An explicitly selected Newton or CDIIS
method takes precedence over that legacy switch.

## Newton equation and acceptance

At fixed geometry define `r(q)=q_out(q)-q`. The solver differentiates the
generalized Bloch eigenproblem and solves

```text
chi0 = dq_out / dV
K    = dV / dq
(I - chi0 K) delta_q = r(q)
```

`K` contains the Ewald shell kernel and every enabled on-site charge order.
The canonical finite-temperature susceptibility uses one chemical potential
over the complete weighted k mesh. Fermi divided differences have an analytic
coincident-energy limit, so exact fractional degeneracies are supported.
The constant-potential mode is removed by the fixed-electron-number response;
trial charges preserve the requested total charge.

Newton steps are capped at 0.5 electron per shell before backtracking. A trial
is accepted only when the actual recomputed squared charge residual decreases.
If a Newton direction cannot decrease it, the solver tries the analytic
negative gradient of the residual norm. A conservative damped fixed-point step
is the final fallback. Neither an extrapolated residual nor a small step alone
can declare convergence: the requested charge and energy tests must both pass.
The reported iteration energy is evaluated consistently at the output density.

In `auto`, stagnation, a large rebound from the best residual, or a long
Broyden tail triggers Newton. The switch restarts from the best charge iterate.
Easy calculations retain the inexpensive Broyden path. Newton allocates a
dense shell susceptibility only when needed and checks its workspace budget.
This is an adaptive default, not a guarantee that one solver is fastest for
every material or that multiple electronic stationary states cannot exist.

Joint multipole SCC and unequal constrained-spin populations currently retain
charge/moment mixing in `auto`; explicitly requesting charge-only Newton for
them is rejected. Zero-temperature fractional occupation discontinuities are
also reported rather than regularized by silently raising the temperature.
Use the intended physical finite temperature when that response is needed.

## Rust

```rust
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, SccAccelerator};
use gfn1_rs::pbc::{PbcOptions, scf::run_pbc_scc};

fn calculate(cell: &PeriodicSystem) -> gfn1_rs::Result<()> {
    let options = ElectronicOptions {
        electronic_temperature: 300.0,
        scc_accelerator: SccAccelerator::Auto,
        energy_tolerance: 1e-13, charge_tolerance: 1e-11,
        max_scc: 250,
        ..Default::default()
    };
    let result = run_pbc_scc(cell, &Gfn1Parameters::builtin()?,
        &options, &PbcOptions::default())?;
    println!("{}", result.scc_diagnostics.as_json());
    Ok(())
}
```

`run_pbc_scc_with_guess` and `PbcSccGuess::from(&previous_result)` reuse a
previous converged charge state. Cell/ionic relaxation and EOS continuation
use accepted-state guesses; failed trials cannot overwrite them.

## Python native and ASE

```python
from gfn1_rs import Gfn1NativeCalculator, GFN1RSCalculator

native = Gfn1NativeCalculator(
    electronic_temperature=300.0, scc_accelerator="auto",
    energy_tolerance=1e-13, charge_tolerance=1e-11, max_scc=250,
    threads=2, memory_mb=2048,
    log_file="periodic-scf.jsonl", log_console=True)
result = native.calculate_periodic(numbers, positions, cell,
    kgrid=(2, 2, 2), gamma_centered=True)

ase_calc = GFN1RSCalculator(
    electronic_temperature=300.0, scc_accelerator="auto",
    energy_tolerance=1e-13, charge_tolerance=1e-11, max_scc=250,
    kgrid=(2, 2, 2), threads=2, memory_mb=2048,
    log_file="ase-periodic-scf.jsonl", log_console=True)
atoms.calc = ase_calc  # atoms has its periodic cell and pbc=True
energy_ev = atoms.get_potential_energy()
```

The JSONL events `periodic_scf_iteration`, `periodic_scf_solver_switch` and
the completed `periodic_scf` span record the actual method, residual,
temperature and work. `map_evaluations` counts trial Fock diagonalizations,
including line-search rejections; `iterations` counts accepted outer-driver
visits, so the two numbers need not match. Elastic/electromechanical response
dictionaries also retain the reference SCC diagnostics.

The CLI accepts `gfn1-rs crystal.xyz --scc-accel auto` (or `newton`, `broyden`,
`cdiis`, `linear`), together with temperature, mixing, iteration-limit
and JSONL logging options. Rust/native/ASE additionally expose explicit SCC
energy/charge tolerances. See the actual CLI help for the complete options.

## Verification and references

The charge Jacobian is independently checked in every shell-potential
component at zero temperature, 300 K and a genuinely fractionally occupied
30000 K case on an unfolded complex k mesh. A separate exactly degenerate
half-occupied case checks invariance under orbital rotations. Full on-site
charge-kernel components are checked through charge order six.

Cold LiF at the previously optimized volume and volumes ±6% now converges
with unchanged 300 K and strict tolerances. Direct Newton and `auto` agree with
the independently reconverged warm Broyden state in free energy, forces and
stress. See [the measured SCC benchmark](../tests/data/pbc-scc-results.json)
and `tests/pbc_scc_solver.rs`; benchmark failures of explicitly chosen legacy
methods are retained as failures. Timings include geometry setup as well as
the SCC loop, and should not be compared to a different machine or workload.

The same-condition benchmark contains six representative crystals on Gamma
and centred 2³ meshes, plus the three expanded LiF volumes, all at 300 K,
mixing 0.4, `1e-13` Hartree energy and `1e-11` electron charge tolerances:

| Method | Converged / tested |
| --- | ---: |
| auto | 15 / 15 |
| Newton | 15 / 15 |
| Broyden | 11 / 15 |
| charge CDIIS | 6 / 15 |
| linear | 6 / 15 |

Where an earlier independent reference energy is available, the maximum
`auto` difference is `6.03e-13 Hartree`. The expanded LiF cases converge in
16–17 outer iterations / 21–22 actual maps under `auto`; direct Newton uses
10 outer iterations / 23 maps. A separate mixing-0.1 regression uses 18–23
`auto` iterations. These are different explicitly recorded mixer conditions.

The dielectric-Jacobian interpretation and the need to compare difficult SCF
problems, rather than assume a universal best mixer, are discussed by
[Woods, Payne and Hasnip, *Computing the self-consistent field in Kohn–Sham
density functional theory*](https://arxiv.org/abs/1905.02332),
[DOI: 10.1088/1361-648X/ab31c0](https://doi.org/10.1088/1361-648X/ab31c0).
The present implementation differentiates this project's SCC shell map;
it does not import a Kohn–Sham solver or change GFN1 parameters.
