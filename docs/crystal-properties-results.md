# Executed crystal-property calculations and physical applicability

The new analytic periodic response can evaluate real crystallographic cells,
but a numerically completed tensor is not a guarantee that builtin GFN1-xTB
describes that material quantitatively. The executions below demonstrate both
usable model responses and explicit limitations. They do not use an external
electronic-structure executable or an imposed dielectric/Born parameter set.

The standalone diagnostic is
`tests/validation/crystal_electromechanical_validation.rs`. The distributed
measurement receipt is
[`tests/data/crystal-electromechanical-properties.json`](../tests/data/crystal-electromechanical-properties.json), with the complete
Hessians, elasticity, Born/piezoelectric tensors, dielectric tensors, signed
modes, IR oscillator tensors, three directional nonanalytic matrices and every
error. Entries marked `analytic_response_completed` only mean that the analytic
calculation returned. Stability/stationarity/availability are separate fields.
The receipt preserves the exact raw-byte hash, measured input-fixture hashes
and previous LiF attempts. One-off report-formatting helpers are not distributed;
the reusable diagnostic above produces new raw measurements.
The development raw path `target/validation/v070/crystal-electromechanical-raw.json`
is identified for provenance, without pretending that the current Rust source
hash identifies all earlier runs. The receipt has 11 current cases: nine
completed responses and two explicit Berry-admission failures, plus two
preserved previous LiF runs.

## Inputs and common conditions

The representative C-diamond, zincblende SiC, rocksalt LiF and rocksalt NaCl
cells are ideal two-atom prototypes at declared representative conventional
lattice constants, respectively 3.567, 4.360, 4.026 and 5.640 Å. They are the
same geometries/masses as `tests/data/crystal-dfpt-results.json`, also used by
the earlier all-component Hessian checks. These four cells are not newly
acquired experimental CIFs. They remain at their supplied geometries here.

The additional 14-atom primitive Na2MoO4 cell is an actual experimental
structure from COD 2240655, with full provenance in
`tests/data/real_crystals.json`. The source is Fortes' neutron powder structure,
reported at 298 K, conventional a = 9.10888 Å. Its measurement pressure is not
specified in the fixture and is not assumed to be ambient. The pinned COD-tools
snapshot, source-file SHA-256, symmetry expansion and primitive-cell conversion
without idealization are recorded. Dummy centroids were removed, with no
geometry relaxation or artificial atoms. [Original structure paper and
supporting data](https://journals.iucr.org/e/issues/2015/06/00/wm5152/index.html),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4459306/).

Every electronic calculation here uses scalar builtin GFN1-xTB at **300 K**,
neutral charge, full Γ-centred **unfolded** SCC/Berry meshes, SCC energy/charge
tolerances 1e−13 Ha/1e−11 e and up to 1000 SCC iterations. Only a failed SCC
mixer is retried, in the recorded sequence 0.4, 0.1, 0.05. No failure causes a
temperature change. The AO cutoff is 30 bohr for the prototypes and the
fixture's 24 bohr for experimental Na2MoO4. Ewald and CP controls retain the
recorded route defaults. Checked workspace budget is 2 GiB with two workers;
this is not a hard process-RSS cap.

Electronic temperature is a Fermi-occupation condition. The source structure's
measurement temperature and phonon thermodynamic temperature are different
quantities. This survey computes electronic responses and Γ modes, not a bulk
phonon free-energy integral or an electron/lattice finite-temperature fit.

All components of a combined response reuse **one converged SCC state**.
Electronic dielectric response differentiates the variational discretized
Berry electric enthalpy `F − Ω E·P_Berry`; it does not substitute the
reference-cell Mulliken dipole field. Born tensors and proper piezoelectricity
use analytic derivatives of the same Berry functional. See the primary
[finite-field formulation](https://arxiv.org/abs/cond-mat/0205633),
[proper piezoelectric response](https://arxiv.org/abs/cond-mat/9903137), and
[coupled strain/displacement/field response](https://arxiv.org/abs/cond-mat/0501548).

## Representative 2×2×2 results

The table reports the first atom's diagonal Born charge for the binary
prototypes, ε∞,xx, the largest absolute clamped e component, the triply
degenerate analytic Γ optic frequency and the highest optic frequency after
the directional [100] nonanalytic correction. Units are e, dimensionless,
C/m², cm⁻¹ and GPa. All three Cartesian acoustic branches are retained near
zero; no acoustic or crystal-symmetry projection was imposed.

| Cell | Z* first atom | ε∞,xx | max \|e clamped\| | Γ optic | Γ+[100] highest optic | Reference pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C diamond | approximately 0 | 3.615519 | below 1e−14 | 1594.5831 | 1594.5831 | 27.4832 |
| SiC | +2.807017 | 3.636328 | 0.676007 | 802.3168 | 1120.4256 | 26.7260 |
| NaCl | +0.993798 | 1.927503 | below 1e−14 | 140.9497 | 245.3073 | −1.1918 |
| LiF | unavailable | unavailable | unavailable | unavailable in combined API | unavailable | see failure below |

The binary prototypes which complete have ion forces below 2.5e−16 Ha/bohr,
Born acoustic-sum residuals below 4e−15 e and no negative Γ curvature below
the declared tolerance. The clamped dielectric matrices retain their actual
off-diagonal entries: approximately 0.112457 for C, 0.0392234 for SiC and
0.00539331 for NaCl. Those entries must be examined with mesh convergence;
they are not overwritten with the infinite-sampling cubic form.

For SiC, relaxed-ion e has a shear component about 0.902650 C/m² at 2³,
distinct from the clamped value. NaCl and diamond have essentially zero proper
piezoelectricity in these calculations. Fixed-strain total dielectric diagonals
are 7.091488 for SiC and 5.838319 for NaCl; C's ionic increment vanishes with
its Born tensors. All supplied prototype cells have appreciable residual
stress, so piezoelectric d and the free-stress static dielectric are
**unavailable** at these references. The API records the stress/stability
reason instead of reporting an equilibrium coefficient at a stressed cell.

For a polar insulator the supplied LO–TO postprocessor adds the explicit
rank-one `4π/Ω` nonanalytic term using the calculated Born tensors, ε∞ and
specified Cartesian direction. This is a directional Γ limit. It is not an
arbitrary-q long-range interpolation or evidence that a coarse q mesh suffices
for thermal properties. The raw report also retains [110] and [111] limits.

LiF 2³ reaches a converged SCC but fails the insulating occupied-subspace guard
at 300 K: one occupation is about 0.999995558, deviating by 4.442e−6 from an
integer against the default 1e−6 Berry tolerance. The Γ-only mixer retry reaches
a genuinely fractional occupation about 0.956002. These are recorded combined
response failures, not passes or zero responses. Repeating both meshes with
the improved automatic analytic charge-Newton SCC solver converges on the
first mixer but retains those same occupation failures; the raw report keeps
the previous runs. The calculation was not
rerun at zero temperature to conceal the condition mismatch. Other finite-T
energy/force/Hessian routes remain separate from insulating Berry response.

## Measured sampling dependence

At the same NaCl geometry, three complete meshes give:

| Electronic/Berry mesh | Na Z*xx | ε∞,xx | ε∞,xy | Γ optic / cm⁻¹ | Pressure / GPa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Γ | 0.98396590 | 1.84376045 | 0.00001137 | 128.00392 | 11.47619 |
| 2³ | 0.99379777 | 1.92750261 | 0.00539331 | 140.94966 | −1.19180 |
| 4³ | 0.98437390 | 2.03831541 | 0.00224857 | 134.42066 | −1.84965 |

The off-diagonal dielectric decreases by about 58% from 2³ to 4³, while the
diagonal still changes by about 5.7%. Thus 4³ has **not** established numerical
convergence. The accidentally small Γ off-diagonal is not an accuracy test.
Finite primitive-direction Berry discretization and SCC sampling must be
converged before interpreting a cubic continuum tensor; this short sequence
does not isolate all discretization and model errors.

Other coarse-sampling changes are substantial: SiC Born charge changes from
3.707863 at Γ to 2.807017 at 2³, clamped e from 1.214213 to 0.676007 C/m² and
optic frequency from 975.8746 to 802.3168 cm⁻¹. Diamond changes from
2292.5208 to 1594.5831 cm⁻¹, and its reference pressure from 360.39 to
27.48 GPa. Numerically exact derivatives of a coarse-mesh energy do not remove
that coarse-mesh model bias.

## Actual experimental Na2MoO4 geometry

Both meshes complete the 14-atom combined analytic calculation at 300 K.

| Quantity | Γ | 2³ |
| --- | ---: | ---: |
| Sampled occupied-subspace minimum gap / eV | 4.343002 | 4.465487 |
| ε∞,xx | 1.557272 | 1.932445 |
| ε∞,xy | 0.149567 | 0.0331844 |
| Maximum atomic gradient / Ha bohr⁻¹ | 0.0159198 | 0.0140028 |
| Hydrostatic reference pressure / GPa | +2.46102 | −0.815391 |

At 2³ the two Mo diagonal Born components are approximately +3.797789 e,
Na diagonal components about +1.0162 to +1.0164 e, and O tensors are anisotropic
with representative diagonals −1.3608 to −1.5544 e. The maximum Born acoustic
sum residual is 1.13e−9 e. Effective charges are response coefficients and need
not equal formal oxidation states. Proper clamped piezoelectricity is below
1e−13 C/m², consistent numerically with the centrosymmetric input.

There are three nearzero Γ translations and 39 positive internal Γ curvatures
at 2³, with internal frequencies from approximately 45.43 to 662.19 cm⁻¹.
The Γ-only electronic reference instead has **six negative internal modes**,
with lowest signed frequency −123.39 cm⁻¹; its stable ionic dielectric is
undefined. Changing only electronic sampling changes this local stability
diagnostic. Neither is a full-Brillouin-zone stability test. The experimental positions
are also **not stationary for this model**, as the forces show. Raw IR mode
charges, oscillator tensors, the local Hessian and directional nonanalytic
limits can be calculated there; interpreting them as equilibrium phonons or
a measured static dielectric is unjustified. The combined API therefore
withholds relaxed e, d and static equilibrium responses, recording the
stationary-atom failure explicitly. Model relaxation at a converged sampling
and subsequent stability checks are necessary before those quantities apply.

## Mechanical relaxation and equation-of-state diagnostics

The separate analytic mechanics driver is
`tests/validation/crystal_mechanical_validation.rs`, with incremental raw
results in `target/validation/v070/representative-mechanics-raw.json`.
The complete promoted mechanical receipt is
[`tests/data/crystal-mechanical-properties.json`](../tests/data/crystal-mechanical-properties.json):
13 original-reference responses, including the remaining LiF/NaCl/MgO
prototypes and experimental Na2MoO4, matched analytic force supplements,
complete stiffness/internal-strain arrays and all errors. Its input raw reports
and hash scope are identified in that receipt; source hashes describe report
generation, not a fabricated historical build fingerprint.
The Γ-only C, Si and SiC cells can be relaxed using analytic forces/stress;
these coarse-mesh minima are numerical/model diagnostics.

For diamond, variable-cell relaxation converges in 14 steps to
V = 145.125727 bohr³ per two-atom primitive cell and free energy
−3.5795229615 Ha. A five-volume ±6% BM3 fit gives V0 = 145.154428 bohr³,
B0 = 144.769929 GPa, B′ = 6.61457 and energy RMS residual 3.31e−6 Ha.
The fitted and optimized volumes differ by about 0.02%, within the finite
window's fitting error. The implied conventional lattice parameter is about
4.414 Å, markedly different from the supplied 3.567 Å. This illustrates why
such a result must not be presented as an experimentally validated lattice
prediction. Fit minima are approximate and need not have exactly zero analytic
pressure.

Si's analogous coarse Γ EOS gives V0 = 361.253134 bohr³, B0 = 101.388990 GPa
and RMS residual 7.58e−6 Ha. The SiC ±6% fit is considerably less trustworthy:
RMS residual 1.98e−4 Ha, B′ = −7.29634 and fitted V0 = 248.463614 bohr³
versus optimized V = 243.190397 bohr³. Inspect the fit and energy samples;
successful polynomial fitting alone is not evidence for a reliable modulus.
At the respective Γ model-relaxed cells, the independent analytic Hill
bulk/shear moduli are C: 140.37594/79.04401 GPa, Si: 106.00330/60.98373 GPa
and SiC: 124.76421/34.00681 GPa. Their differences from finite-window EOS fits
must be considered with EOS residuals and coarse sampling. In particular,
force/stress convergence alone does not prove a minimum: a separately relaxed
LiF stationary structure retains negative internal curvature and is a saddle.
See [mechanical-properties.md](mechanical-properties.md) for the analytic
clamped/relaxed stiffness and stability contract.

## What can be inferred from a structure

For a supported scalar GFN1/ML cell one can obtain model energy, forces,
stress, relaxation, EOS, analytic elasticity, sampled bands/DOS, analytic
Γ/commensurate-supercell phonons, phonon DOS and stable-mode thermodynamics.
Insulating stationary cells additionally admit Born charges, proper
piezoelectricity, Berry electric-enthalpy ε∞, IR mode charges and directional
LO–TO splitting. d and free-stress dielectric require a stable zero-stress
reference. Volume-sampled QHA adds a fit-based thermal-expansion/Cp route,
with electronic temperature held fixed and explicit stability checks.

The same production routes include actual loaded structural ML weights; no
fallback to base GFN1 is used. This particular applicability survey uses the
builtin scalar parameters. A trained model requires its own sampling and
out-of-sample physical validation. [Property-objective fine-tuning](ml-property-training.md)
can target these quantities while retaining analytic physical predictions;
synthetic round trips establish implementation consistency, not chemical
accuracy. The largest practical uncertainties here are unconverged sampling,
nonstationary experimental geometries within the model and the empirical
Hamiltonian's transferability to each solid.
