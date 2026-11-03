# Documentation index

The docs are organised **one page per subsystem** so a new feature is appended to
the page that owns it rather than bolted onto a growing monolith.

| Page | Covers |
| --- | --- |
| [api-overview.md](api-overview.md) | Rust/native/ASE/CLI entry points, units, compatibility, controls and supported domains. |
| [scope.md](scope.md) | The feature matrix — what exists at all. |
| [parameters.md](parameters.md) | Bundled parameters, provenance, independent ML models, observable training and operational logs. |
| [ml-training.md](ml-training.md) | Complete reference-data, training, user-weight loading and identity logging recipes for Rust, Python native and ASE. |
| [ml-property-training.md](ml-property-training.md) | Warm-start property objectives, explicit conditions, units, masks, held-out validation and user-weight identity through all three APIs. |
| [derivatives.md](derivatives.md) | Nuclear derivative ladder, contracted FC3/FC4, term registry and verification gates. |
| [finite-temperature.md](finite-temperature.md) | Charge-space dielectric response, Fermi smearing and analytic finite-T FC3/FC4, including exact fractional degeneracy. |
| [pbc.md](pbc.md) | Periodic SCC, gradients, stress, Hessian/FC3, SOC/ML response, commensurate phonons, Grüneisen and Berry polarization. |
| [pbc-scc.md](pbc-scc.md) | Adaptive periodic Broyden/Newton SCC, canonical finite-temperature Jacobian, solver selection and actual-work diagnostics. |
| [mechanical-properties.md](mechanical-properties.md) | Improved periodic relaxation, EOS, analytic elasticity/internal strain, compliance, moduli and acoustic speeds. |
| [phonon-band-properties.md](phonon-band-properties.md) | Commensurate phonon meshes/DOS/thermal properties, volume-sampled QHA, native bands and electronic DOS. |
| [electromechanical.md](electromechanical.md) | Analytic Born charges, proper piezoelectricity, electronic/ionic dielectric response, IR and LO–TO limits. |
| [crystal-cli.md](crystal-cli.md) | Property-by-property Rust/native/ASE/CLI coverage, strict JSON requests and composed calculation workflows. |
| [crystal-properties-results.md](crystal-properties-results.md) | Actual crystal calculations, source structures, sampling effects and mechanical/electrical stability limits. |
| [real-crystal-property-survey.md](real-crystal-property-survey.md) | Twenty property families executed on sourced CIF descendants, including full 2x2x2 diamond QHA and retained failures. |
| [td.md](td.md) | TD-GFN1 (TDA) excited states: the working equations, the analytic excited-state gradients (molecular, Γ-PBC, k-mesh), the MO phase gauge, and the gate numbers. |
| [excited-hessians.md](excited-hessians.md) | Analytic molecular TDA/sTDA excited-state Hessians, response equations and Rust/native/ASE recipes. |
| [ethylene-excited-optimization.md](ethylene-excited-optimization.md) | Actual unconstrained S1 optimizations, negative-curvature restarts, dihedrals and model failures. |
| [limitations.md](limitations.md) | **Known gaps and honest failure modes**, with the exact error messages. |
| [rust-api.md](rust-api.md) | Rust library API reference. |
| [python-api.md](python-api.md) | Python (`Gfn1NativeCalculator`) and ASE (`GFN1RSCalculator`) API. |
| [resources.md](resources.md) | Scoped CPU worker counts, checked workspace budgets, Rust/CLI/native/ASE usage and logs. |
| [cp-settings.md](cp-settings.md) | CP/DFPT residual tolerances and iterative limits through every interface. |
| [soc-dfpt.md](soc-dfpt.md) | Explicit SOC parameters, physical spinors, nuclear CP-SCC/DFPT, units and commensurate phonon examples. |
| [crystal-dfpt-results.md](crystal-dfpt-results.md) | Six crystals at 300 K, Gamma and centred 2x2x2 electronic meshes, actual modes and all-component force-difference checks. |
| [stda-xtb.md](stda-xtb.md) | Distinction between TD-GFN1 TDA and sTDA-xTB, primary references and pinned official implementations. |
| [stda-pes.md](stda-pes.md) | Measured orbital-degeneracy limit and a separately defined, audited upper-envelope crossing stationary point. |
| [stda-pes-local-topology.md](stda-pes-local-topology.md) | Shrinking-neighborhood angular response, finite-temperature projector limit and unconstrained trust-region follow-up. |
| [tda-stda-comparison.md](tda-stda-comparison.md) | Same-molecule spectra computed by both native Rust implementations, with independent validation. |
| [fragments.md](fragments.md) | FMO2/FMO3, covalent HOP, overlapping DC, analytic gradients and periodic stress. |
| [sigma_hole_preset.md](sigma_hole_preset.md) | The `sigma-hole` CAMM preset. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Project discipline — **docs are updated in the same commit as the feature.** |

Version note: Rust and Python distribution versions are `0.7.0`.
Sections marked *v0.5.0* describe inherited baseline functionality.
The implemented scope and further research work
are recorded in [scope.md](scope.md) and [limitations.md](limitations.md).
The current API/CLI and sourced-crystal validation rollup is
[v070-properties-survey-validation.json](../tests/data/v070-properties-survey-validation.json).
The preceding [SCC/ML rollup](../tests/data/v070-crystal-scc-ml-validation.json)
retains its earlier measured binary identities.
