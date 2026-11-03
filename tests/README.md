# Tests, validation and profiling

`examples/` contains user calculation inputs. Developer checks live here:

- `tests/*.rs`: ordinary Rust integration tests, registered in `Cargo.toml`.
- `tests/python/`: installed Python native and ASE API regression tests.
- `tests/validation/`: reusable numerical audits, diagnostics and
  source/asset checks. The standalone Rust drivers use their own `main`.
- `tests/profiling/`: standalone latency and memory investigation drivers.
- `tests/data/`: inputs, independent reference results and retained receipts.

Standalone Rust drivers are Cargo test targets with `harness = false` and
`test = false`. They run only when explicitly selected, so the normal test
suite does not unexpectedly execute lengthy profiling or audit workloads:

```sh
cargo test --profile reltest --test crystal_dfpt_validation -- crystal-raw.json
cargo test --profile reltest --test tda_stda_analysis > comparison-raw.json
cargo test --profile reltest --test profile_fc4 -- dir
python tests/validation/audit_assets.py --release --summary
cargo test --profile reltest --test pbc_scc_validation -- target/validation/pbc-scc-results.json
```

Run these commands from the crate root. Arguments after `--` are driver
arguments, not libtest filters. Some diagnostics print measurements without
asserting an acceptance threshold; use their documented numerical audit or
integration test when a pass/fail gate is required. Profiling timings depend
on the host and are not correctness assertions.

Historical JSON receipts preserve the paths and hashes of the source that was
actually measured, including paths under the former `examples/` layout. New
reports record the current paths; historical measurements are not relabelled.
One-off recording, report-formatting, input-capture and packaging helpers are
excluded from the source distribution. Their historical paths/hashes in receipts
identify the measured workflow and are not commands required to reproduce the
physical calculations. Reusable drivers write new results to user-selected paths.

The ethylene crossing workflow uses `optimize_stda_crossing.py` and
`validate_stda_crossing.py`. It checks an explicitly defined uncoupled upper
envelope, rather than claiming force-zero stationarity of one signed CI sheet.
`probe_stda_pes_degeneracy.py` builds the Rust `stda_pes_orbital_probe` target
and measures directional limits. See [docs/stda-pes.md](../docs/stda-pes.md) for
the equations, limitations, fixed seed and complete reproduction commands.

