# Changelog

## v0.7.2

- Differentiate molecular TDA and RPA with fractional occupations and degenerate
  orbitals using block Fock and canonical occupation matrices.
- Add RPA manifold derivatives and a rotation-invariant finite-temperature sTDA
  ensemble model with analytic gradients and Hessians.
- Apply directional TD derivatives directly to selected eigenvectors and
  bound parallel Hessian work by the calculation's workspace budget.
- Transport near-degenerate orbitals together in thermal TDA/RPA gradients and
  Hessians to reduce cancellation in their nuclear response.
- Use stationary variational molecular FC3/FC4 and bounded packed Cartesian
  polarization where applicable. Stream CLI tensor components from packed storage.
- Add optional file-backed FC3 and FC4 output in CLI, Rust, Python native and
  ASE. File-backed FC4 supports Cartesian sizes beyond the expanded-output cap.

## v0.7.1

- Default to analytic TDA gradients across the CLI, Rust selector, Python
  native and ASE. Report degenerate roots in all interfaces and provide separately
  selected gradients and Hessians of the mean TDA surface.
- Add fractional-occupation molecular and Γ-point TDA, molecular full-TD RPA
  spectra and analytic nuclear derivatives through second order, and optical
  periodic TD-DFPT excitation energies with optional explicit SOC.
- Add analytic ALPB/GBSA Hessians, FC3 and FC4 for molecular canonical free
  energies, including solvated PES-only VPT2.
- Print cubic and quartic tensor components directly. Expose the periodic DFPT
  Hessian, molecular excited Hessians, magnetic tensors, chiroptical,
  electromechanical, vibrational and fragment properties in the shared CLI.
- List embedded ML-xTB weights, including the water-cluster model and its
  canonical identifier.

## v0.7.0

- Organize the Rust/native/ASE/CLI API entry map, units, compatibility spellings
  and supported domains. Exclude one-off recording/report/capture helpers from
  source packages while retaining reusable physical audits and measured data.
- Wire 21 crystal calculation/postprocessing operations to the shared Rust/pip
  `gfn1-rs properties` CLI, with strict units/conditions, result references,
  user-weight identity and nonzero exits for failed/nonconverged tasks. Add a
  Rust/native/ASE/CLI coverage table and executed request examples.
- Execute twenty property families on attributed diamond, ZnS, SiC and NaCl
  CIF descendants. Retain source/model geometry changes, metallic-Berry and
  overlap-cutoff failures, and both sparse and full 2x2x2 diamond QHA records.
- Extend original-surface sTDA diagnostics with 48 branching-plane samples,
  angular derivative checks and six unconstrained exact-Hessian trust-region
  searches. Preserve their nonconvergence and the measured direction-dependent
  projector limit; no unvalidated smoothing model replaces original sTDA.

- Add adaptive periodic SCC with canonical finite-temperature analytic charge
  Newton steps, residual backtracking, actual-work diagnostics and working
  explicit Broyden/CDIIS/linear/Newton selections. Resolve cold LiF convergence
  without changing its temperature, Hamiltonian or convergence thresholds.
- Improve periodic atomic/cell relaxation with accepted-state SCC guesses,
  replication-scaled strain coordinates, origin-independent atomic step limits
  and bounded geometry caches. Add EOS and analytic material elasticity,
  internal-strain response, compliance, moduli and sound speeds.
- Add analytic structural-ML Born charges, proper piezoelectricity, electronic
  dielectric response and coupled relaxed-ion properties, with explicit
  stationarity/stability guards, IR oscillator strengths and LO–TO corrections.
- Add commensurate phonon mesh/DOS and harmonic thermodynamics, volume-sampled
  QHA, electronic bands/DOS, and full Rust/native/ASE guides and crystal results.
- Add user-model property-objective fine-tuning with explicit physical
  conditions, weighted component masks/scales, held-out validation and saved
  model identity. General objectives use explicitly named parameter differences
  around analytic physical backends; existing molecular E/F objectives reuse
  analytic model-parameter gradients when admitted.
- Find and independently audit unconstrained force-zero stationary points on
  the original signed sTDA composite surface. Classify their negative internal
  Hessian modes as saddles; subsequent downhill searches still approach the
  frontier-degeneracy model singularity rather than a verified smooth minimum.

- Support smearing through all standard molecular FC3/FC4 and periodic Γ/k-mesh
  FC3 APIs, including exact fractional degeneracy. Preserve canonical analytic
  free-energy response and correct omitted periodic CN/SCC reference chains.
- Add scoped CPU worker counts and checked workspace budgets through Rust,
  both CLIs, Python native and ASE, with configuration logging.
- Add scoped absolute/relative CP residual tolerances and iterative limits
  through all interfaces, with explicit direct/iterative residual checks.
- Pass local parallelism directly to Faer eigensolvers and LU solves, fixing
  recursive worker-pool entry and the resulting higher-derivative stack overflow.
- Explicitly include upstream hidden sTDA parameter files in Cargo source
  packages, exclude generated Python bytecode from wheels, and verify a build
  from the extracted Cargo archive.
- Mix complete spin densities for +U/CAMM feedback, conserving the requested
  electronic temperature and resolving the combined CAMM/linear-response-U
  convergence failure. Check the analytic gradient against all Cartesian
  components of reconverged finite differences at 300 K.
- Add analytic molecular TD-GFN1/sTDA excited-state Hessians through Rust,
  native Python and ASE, including complete CI eigenvector response. Follow
  signed nondegenerate sTDA CI branches across zero without silently relabelling
  filtered roots, and report reference-instability diagnostics explicitly.
- Diagnose the native sTDA frontier-degeneracy PES limit. Retain the unresolved
  ordinary signed-sheet minimum explicitly and add a separately defined
  uncoupled upper-envelope crossing optimization with all-component audits.
- Use the supplied prototype ZIP as the development base, preserving its edits.
- Share the Rust calculation and training drivers with Python console scripts.
  Command names are `gfn1-rs` and `gfn1-rs-train` for both Cargo and pip installs.
- Route positive-temperature Python FC3/FC4 calls to analytic free-energy
  response. Add a quadratic-cost finite-temperature cubic vector contraction and
  a molecular `--fourth-derivative` CLI option.
- Support exactly degenerate fractional occupations in the third density
  response used by finite-temperature analytic molecular FC4.
- Implement independent native Rust sTDA-xTB with its dedicated VTB/xTB
  orbitals, sourced parameters, configuration selection and dipole integrals.
  Add analytic excitation gradients and an explicitly named composite
  GFN1-free-energy plus sTDA-excitation total gradient.
- Remove the official-program driver API and executable-path options from
  Rust, Python and the CLI. Official sources are cited and used only for
  independent offline validation fixtures.
- Preserve legacy TD-GFN1 TDA calls and add an optional AO length-gauge
  transition dipole. Document actual native TDA/sTDA spectra with identical
  geometries, charge, spin and 300 K electronic temperature.
- Repair TDA fractional-occupation admission: molecular, Gamma and k-point
  paths reject unsupported thermal fillings before generic CP pair building
  can introduce unweighted intravalence transitions. Keep gapped 300 K spectra
  and analytic gradients unchanged; add whole-energy derivative regressions.
- Add embedded FMO2/FMO3, covalent hybrid-orbital-projection boundaries and
  overlapping density-matrix divide-and-conquer, with molecular/periodic
  analytic forces and Gamma-periodic analytic stress. Bound local allocation
  and reuse the embedding response factorization.
- Review magnetic SCC stationarity and input validation; cache the unchanged
  overlap orthogonalization per SCC solve. Verify magnetic forces, all tensor
  components and mixed-energy Cotton–Mouton differences. Clarify the static
  MCD and common-gauge-origin NMR model boundaries.
- Add complete Rust, native Python and ASE guides for SOC/DFPT, FMO/DC and
  user ML training, weight loading and explicit source/hash logging.
- Report 300 K DFPT results for C, Si, SiC, LiF, NaCl and MgO with all 432
  Hessian entries checked, actual imaginary modes and convergence conditions.
  Add a compatible explicit Gamma-centred Python/ASE Hessian sampling option.
- Reconstruct and verify the inherited secondary-basis subsets against BSE
  and the original GFN1 hydrogen construction; preserve their numerical bodies
  and correct the provenance/license descriptions.
- Move developer diagnostics, numerical validation and profiling drivers out
  of `examples/` into `tests/validation/` and `tests/profiling/`, with explicit
  Cargo test targets and updated reproduction commands.

## v0.6.0

- Add molecular GFN1 ALPB/GBSA, explicit-parameter physical-spinor SOC, periodic
  SOC/ML response, commensurate-supercell phonons and PES-only VPT2.
- Add independent embedded H2/water reference models, a separate training CLI,
  observable training and JSONL operational logs.
- Repair high-rank PH3 convergence at unchanged zero electronic temperature.
- Validate implemented nuclear derivatives with exhaustive Cartesian component
  differences, and real molecular VPT2 and static Cotton–Mouton energy checks.
- Update `private-gemm-x86` to a non-yanked version and resolve the validated
  Python dependency/fixture warnings.

## v0.5.1 inherited baseline

The supplied project originated from this version. It already contained the
molecular/periodic GFN1 engine, analytic derivative infrastructure, spectroscopy,
magnetic M0/M1 models, optimization and Python/ASE interfaces. No separate
v0.5.1 commit history was supplied, so patch-level changes are not reconstructed.

## v0.5.0 (documented source milestones)

- Add analytic molecular quartic force constants and remove the directional
  quartic degree-of-freedom cap.
- Add periodic analytic cubic response at Gamma and k meshes.
- Deduplicate polarization directions and share reference data when expanding
  symmetric higher-derivative tensors.
