# Third-Party Notices

## Revision-2 implementation asset ledger

`third_party/assets.json` records exact-byte SHA-256 hashes, upstream identifiers,
SPDX expressions, units, element scope, inherited changes and adoption status
for the inherited and newly adopted numerical/source assets. Run
`python tests/validation/audit_assets.py` to check the distribution; `--release` also
requires all original-source adoption reviews to be complete. A checksum pass
means byte integrity, not a completed upstream/license re-audit. The current work
uses the user-supplied **0.6.0 prototype ZIP**, SHA-256
`bcd36224b930e6adfa5888f11dd77e9567fc1bf0215942719964d62442eeed40`,
recorded as `supplied_base_archive_sha256`. The inherited historical 0.5.1
ledger hash `631b65fa2504fec42c0f6b7302bd84132a59b70cd7edf25919143f701ca27312`
is retained as provenance; it is not the development source used for this work.

The secondary-basis numerical bodies from the supplied prototype are preserved
byte-for-byte. All 720 contractions have now been independently accounted for
against BSE 0.12 (its BSD-3-Clause distribution) and native GFN1 hydrogen STO
functions (LGPL-3.0-or-later). The unverified inherited CC-BY-4.0 header claim was
corrected; Dunning/ccRepo attribution is retained. There are 691 selected BSE
contractions, 21 inherited cc-pVDZ d-polarization fallbacks in the TZ/QZ/5Z
subsets (Ne and Al-Ar), and 8 hydrogen contractions reconstructed from STO
tables, GFN1 exponents and analytic same-centre projection. The original BSE
snapshot was not recorded in the supplied prototype: this is a new documented
reconstruction, not a claim to have recovered that historical version or
revalidated the publisher's SI. See `tests/data/secondary-basis-provenance.json`
and `third_party/basis_set_exchange/LICENSE`. Strict M1 checks shell coverage
before calculation. No author SOC
spreadsheets or MACE-OFF weights are distributed. The small H2 demonstration
weights are independently trained by `gfn1-rs-train demo-h2`, GPL-3.0-or-later,
using synthetic GFN1 Hamiltonian targets; they do not reproduce the paper's
trained model or chemical accuracy. `references.bib` is the
user-supplied revision-2 bibliography, extended with references for independently
generated quantum-chemistry targets and WTMAD-2 weighting.

### Adopted ALPB/GBSA source snapshot

`third_party/xtb/src/{solv,param,mctc}` and `third_party/xtb/include/param_{alpb,gbsa}_*.fh`
were exported verbatim from xtb v6.7.1, commit
`26b28010e805f7d1aeeef39813feb473e69cc4be`. The source headers identify
Sebastian Ehlert (2019–2020) and Stefan Grimme (2017–2020), and license the code
LGPL-3.0-or-later. The included parameter files are the tables referenced by
`solv/model.f90`, under the upstream project's LGPL distribution. Original GNU
license texts remain in `third_party/xtb/COPYING{,.LESSER}`. Grid attribution to
Lebedev/Laikov and the Burkardt/DFTB+ adaptation is retained in `lebedev.f90`.

`src/solvation.rs` is a Rust adaptation: immutable caches, explicit GFN1 CM5
charge mapping, and forward analytic differentiation replace the Fortran state
and handwritten gradient storage. Parameter values, Still/P16 kernels, surface
polynomial, surface threshold and reference-state conventions are retained.
Only GFN1 parameter blocks are selected. Salt screening and periodic solvation
are not part of this adaptation. GFN2/FF blocks retained in original include
files are provenance material and are never used as GFN1 fallbacks.

## Independent H2 quantum-chemistry reference and trained model

`models/h2-fci-reference.json` was independently generated using **PySCF 2.14.0**
by `scripts/generate_h2_fci.py`. It contains 23 nonrelativistic Born-Oppenheimer
H2 geometries: 10 training, 9 held-out validation and 4 extrapolation points.
The method is FCI/cc-pVTZ, evaluated as full-active-space CASCI over every RHF
orbital, with two active electrons and no frozen core. The analytic nuclear
gradients include the finite Gaussian basis Pulay contributions. These are
finite-basis FCI results, not a claim of an exact complete-basis H2 potential.

The generated numerical reference data are dedicated to CC0-1.0. No PySCF
program code or third-party pretrained weights are embedded in the trained
model. PySCF itself is available under Apache-2.0 from
https://github.com/pyscf/pyscf . The independent native Hamiltonian model
`models/h2-fci-trained.model` is GPL-3.0-or-later and was fitted with
`gfn1-rs-train` (AdamW). Its complete optimizer settings, input hashes and
measurements are in `models/h2-fci-trained-report.json`.

Scientific references are centralized in `references.bib`:

- `Sun2020PySCF`: Q. Sun et al., "Recent developments in the PySCF program
  package", J. Chem. Phys. **153**, 024109 (2020), DOI 10.1063/5.0006074.
- `Dunning1989CC`: T. H. Dunning, "Gaussian basis sets for use in correlated
  molecular calculations. I. The atoms boron through neon and hydrogen",
  J. Chem. Phys. **90**, 1007-1023 (1989), DOI 10.1063/1.456153.

The energy-only fitted model reduced held-out curve-shape RMSE from
0.010521471410482257 Eh (standard GFN1-xTB) to 0.0032663928291222933 Eh.
Each predictor's constant energy alignment was determined from training points
only. This is an H2-domain initial acceptance result, not a general molecular
or periodic accuracy claim.

## Independent water-cluster references and trained model

`scripts/generate_water_clusters.py` constructs original neutral water-cluster
geometries and calculates PBE0/def2-SVP energies, analytic gradients with DFT
grid response, and total dipoles using PySCF 2.14.0. The 28 structures comprise
14 training, 8 validation, 5 extrapolation and 1 shared isolated-water anchor;
all four tetramers are held out. No third-party coordinate set or pretrained
weights were copied. Generated coordinates, numerical labels and the numerical
training report are CC0-1.0. Generator/training software and original fitted
weights are GPL-3.0-or-later.

`models/water-clusters-trained.model` is a 2,869-byte native radial Hamiltonian
model trained by `gfn1-rs-train` with AdamW for 250 epochs. The complete loss
settings, independent metrics and hashes are in
`models/water-clusters-trained-report.json`; the references are in
`models/water-clusters-reference.json`. Fitting uses monomer distortion or
uncorrected supermolecular interaction energies rather than mismatched absolute
energy origins. The basis is finite, with no counterpoise or dispersion
correction. Validation improvement and extrapolation limitations are described
in `docs/parameters.md` and the disaggregated report.

Scientific references in `references.bib`: `Sun2020PySCF`, `Adamo1999PBE0`
(DOI 10.1063/1.478522), `Weigend2005Def2` (DOI 10.1039/B508541A).
`Goerigk2017GMTKN55` (DOI 10.1039/C7CP04913G) supplies the WTMAD-2 definition
used for explicitly relative/reaction targets. These small demonstration
subsets are not the GMTKN55 benchmark dataset.

## Real-material crystal fixtures

`scripts/generate_crystal_fixtures.py` exports seven independently constructed
ideal prototypes at declared representative lattice parameters (NaCl, MgO,
diamond, Si, ZnS, SiC, SrTiO3) and three experimental COD-derived geometries:
spinel-type Na2MoO4, benzene I and a ferrocenyl(diphenyl)methanol/DMSO adduct.
The latter has a 110-atom primitive cell. Original hydrogen positions are
retained; dummy Cp centroids are removed. Space-group expansion and primitive
reduction without idealization are recorded, along with source hashes and
measurement temperatures. These are unrelaxed GFN1 model inputs; residual
forces and imaginary modes are measured rather than presumed absent.

The source snapshot is `cod-developers/cod-tools` commit
`0060d4652a5a262fbda2276801f1244060d67128`. The export includes cell/site data and
scientific citations, not upstream contacts, local paths or unrelated CIF text.
Original CIF snapshots are read outside the project; their exact LF-normalized
source hashes and repository paths are retained in `tests/data/real_crystals.json`.
COD records: 2240655 (Fortes, 2015, DOI 10.1107/S2056989015008774), 2100348
(Budzianowski and Katrusiak, 2006, Acta Cryst. B 62, 94-101), 2009668
(Glidewell et al., 1994, DOI 10.1107/S0108270194002180).

The COD data public-domain dedication is recorded as
`LicenseRef-COD-Public-Domain`, documented in `third_party/cod/LICENSE.txt` and
the COD deposition policy described by Grazulis et al. (2012), DOI
10.1093/nar/gkr900. This concerns database structural data; it does not apply
the source article's text/figure license to the data. Independent prototype
coordinates and new numerical validation results are CC0-1.0; generator and
validation code are GPL-3.0-or-later. The ideal prototypes are not presented as
verbatim experimental CIFs. No restricted Springer/Pauling File test records
were adopted from external software repositories.

## xtb GFN1 parameter files (inherited snapshot)

This repository bundles the official GFN1-xTB parametrizations, embedded into
the compiled library (`include_str!` in `src/params.rs`) so the crate, CLI, and
Python wheel work without a separate download:

- `third_party/xtb/param_gfn1-xtb.txt` — GFN1-xTB
  (S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
  1989 (2017), DOI 10.1021/acs.jctc.7b00118), including the merged f-in-core
  lanthanoid ("Ln-xTB") La–Lu parameters
  (M. Bursch, A. Hansen, S. Grimme, *Inorg. Chem.* **56**, 12485 (2017),
  DOI 10.1021/acs.inorgchem.7b01950)
- `third_party/xtb/param_gfn1-si-xtb.txt` — GFN1(Si)-xTB silicon
  reparametrization (DOI 10.1021/acs.jcim.1c01170)

Source: <https://github.com/grimme-lab/xtb>, commit
`2b5cd4829290775e575807daee21560f851ff7e1`. See
`third_party/xtb/PROVENANCE.md` for checksums. Both files are line-for-line
identical to upstream (line endings normalized to LF).

The upstream project licenses these files LGPL-3.0-or-later; the license texts
are included at:

- `third_party/xtb/COPYING`
- `third_party/xtb/COPYING.LESSER`

The main crate is licensed `GPL-3.0-or-later`; LGPL-3.0-or-later code/data can
be distributed as part of this GPL-3.0-or-later work (LGPLv3 §2b). The files
remain available under LGPL-3.0-or-later from the upstream repository.

## simple-dftd3 D3 reference data

This repository includes the minimal D3 reference data needed by `src/dispersion.rs`:

- `third_party/simple-dftd3/src/dftd3/reference.f90`
- `third_party/simple-dftd3/src/dftd3/data/r4r2.f90`

Source: <https://github.com/dftd3/simple-dftd3>

The copied files are marked `SPDX-Identifier: LGPL-3.0-or-later` by the
upstream project. The upstream license texts are included at:

- `third_party/simple-dftd3/COPYING`
- `third_party/simple-dftd3/COPYING.LESSER`

The main crate is licensed `GPL-3.0-or-later`; LGPL-3.0-or-later code/data can
be distributed as part of this GPL-3.0-or-later work.

## dftd4 D4 reference data

This repository includes an upstream DFT-D4 provenance snapshot under
`third_party/dftd4`:

- `assets/parameters.toml`
- `src/dftd4/reference.f90`
- `src/dftd4/reference.inc`
- `src/dftd4/data/r4r2.f90`
- `src/dftd4/data/wfpair.f90`
- `src/dftd4/data/hardness.f90`
- `src/dftd4/data/zeff.f90`

Source: <https://github.com/dftd4/dftd4>, commit
`3f40e365c856b75a2c82adf8fa2f5459e6b8aa44`. See
`third_party/dftd4/PROVENANCE.md` for details and checksums.

The copied files are licensed LGPL-3.0-or-later by the upstream project. The
upstream license texts are included at:

- `third_party/dftd4/COPYING`
- `third_party/dftd4/COPYING.LESSER`

The GFN1-specific D4 damping constants `a1`, `a2`, `s8`, and `s9` are not
hard-coded from the DFT-D4 parameter table. They are read at runtime from the
active GFN1 parameter file (the bundled `third_party/xtb/param_gfn1-xtb.txt`
by default, or a user-supplied file; see "xtb GFN1 parameter files" above).

## tblite GFN1 spin constants

This repository includes the GFN spin constants (`W`) needed by the
spin-polarized GFN1-xTB ("spGFN1") feature (`src/spin.rs`; values transcribed
into `src/data_tables.rs` as `GFN_SPIN_CONSTANTS`):

- `third_party/tblite/spin.f90` (verbatim upstream `src/tblite/data/spin.f90`)

Source: <https://github.com/tblite/tblite>, repository HEAD
`eb50bbfbe1c0869e2e18c9b7cc13144e5130b6df` (main); the spin table
itself was last updated upstream in commit
`66175172f1e80b899fd67f79128ee27b2be8dafa`. See
`third_party/tblite/PROVENANCE.md` for details and the method references.

The upstream file is marked `SPDX-Identifier: LGPL-3.0-or-later`. The main
crate is licensed `GPL-3.0-or-later`; LGPL-3.0-or-later code/data can be
distributed as part of this GPL-3.0-or-later work.

## Pinned upstream byte review

`third_party/upstream-byte-review.json` records successful fresh comparisons
of the inherited xtb, tblite, simple-dftd3 and DFT-D4 source/data bytes and retained license
texts against their existing full upstream commit IDs. Source CRLF-to-LF
normalization is explicit. No numerical data was changed. This review does
not certify the mathematical Rust transcriptions or reconstruct the secondary
basis. The secondary-basis source reconstruction remains
open in `third_party/assets.json`.

The inherited `tests/data/real-crystal-phonons-report-v051.json` is tracked in that
ledger with its original package version and failed case intact. A correct
checksum does not make a numerical validation pass.

The current `tests/data/real-crystal-phonons-report.json` records the fresh
0.6.0-dev.1 ten-crystal pass, with exact source fingerprints and rejected cutoff
displacements retained. The historical failed result is preserved separately.
