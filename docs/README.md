# Documentation index

The docs are organised **one page per subsystem** so a new feature is appended to
the page that owns it rather than bolted onto a growing monolith.

| Page | Covers |
| --- | --- |
| [scope.md](scope.md) | The feature matrix — what exists at all. |
| [parameters.md](parameters.md) | Bundled official parameters, resolution order, provenance, reference data. |
| [derivatives.md](derivatives.md) | Nuclear derivative ladder: gradient → Hessian → FC3 → FC4, the term registry, and the verification-gate philosophy. |
| [finite-temperature.md](finite-temperature.md) | Charge-space dielectric response solver (1st + 2nd order), Fermi smearing, finite-T FC3. |
| [pbc.md](pbc.md) | Periodic path: SCC, gradient, stress, Hessian, **analytic FC3 (Γ and k-point)**, seminumerical FC3, `dH/dlnV`, Grüneisen, Berry-phase bulk polarization. |
| [td.md](td.md) | TD-GFN1 (TDA) excited states: the working equations, the analytic excited-state gradients (molecular, Γ-PBC, k-mesh), the MO phase gauge, and the gate numbers. |
| [limitations.md](limitations.md) | **Known gaps and honest failure modes**, with the exact error messages. |
| [rust-api.md](rust-api.md) | Rust library API reference. |
| [python-api.md](python-api.md) | Python (`Gfn1NativeCalculator`) and ASE (`GFN1RSCalculator`) API. |
| [sigma_hole_preset.md](sigma_hole_preset.md) | The `sigma-hole` CAMM preset. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Project discipline — **docs are updated in the same commit as the feature.** |

Version note: the crate version in `Cargo.toml` and the Python distribution
version in `pyproject.toml` are both `0.5.0`, so everything marked *v0.5.0* in
these pages is in the released tree.
