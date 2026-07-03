# Third-Party Notices

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
user-supplied `param_gfn1-xtb.txt`. The canonical upstream copy of that
external parameter file is in <https://github.com/grimme-lab/xtb> at
`param_gfn1-xtb.txt`.

## tblite GFN1 spin constants

This repository includes the GFN spin constants (`W`) needed by the
spin-polarized GFN1-xTB ("spGFN1") feature (`src/spin.rs`; values transcribed
into `src/data_tables.rs` as `GFN_SPIN_CONSTANTS`):

- `third_party/tblite/spin.f90` (verbatim upstream `src/tblite/data/spin.f90`)

Source: <https://github.com/tblite/tblite>, repository HEAD
`eb50bbfbe1c0869e2e18c9b7cc13144e5130b6df` (main, 2026-06-28); the spin table
itself was last updated upstream in commit
`66175172f1e80b899fd67f79128ee27b2be8dafa` (2025-07-22). See
`third_party/tblite/PROVENANCE.md` for details and the method references.

The upstream file is marked `SPDX-Identifier: LGPL-3.0-or-later`. The main
crate is licensed `GPL-3.0-or-later`; LGPL-3.0-or-later code/data can be
distributed as part of this GPL-3.0-or-later work.
