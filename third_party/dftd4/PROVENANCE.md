# DFT-D4 Reference Data Provenance

This directory stores the upstream DFT-D4 reference data used to generate and
audit `src/d4_reference.rs`.

- Upstream repository: https://github.com/dftd4/dftd4
- Upstream commit: `3f40e365c856b75a2c82adf8fa2f5459e6b8aa44`
- Commit date: 2026-06-18T08:01:55Z
- License: LGPL-3.0-or-later; see `COPYING` and `COPYING.LESSER`

Copied files:

- `assets/parameters.toml`
- `src/dftd4/reference.f90`
- `src/dftd4/reference.inc`
- `src/dftd4/data/r4r2.f90`
- `src/dftd4/data/wfpair.f90`
- `src/dftd4/data/hardness.f90`
- `src/dftd4/data/zeff.f90`

Selected SHA-256 checksums:

- `assets/parameters.toml`: `8254bfc673e763f7589b9be20506438b202a59ce795dc175d6ea9ba623e23e1f`
- `src/dftd4/reference.inc`: `0cc90b5f5d8b81dcd5f7a87acb44f575ee3d9e75df6e903f7f1a3ecc728e162f`
- `src/dftd4/data/wfpair.f90`: `ffc7f9f48fd0fc01afc9e4e21be79e36569be2a5293d4147be27a4b655cd642d`

GFN1-specific D4 damping constants are not hard-coded from this directory. At
runtime, `a1`, `a2`, and `s8` are read from the user-supplied
`param_gfn1-xtb.txt` through `Gfn1Parameters`. The D4 ATM scale `s9` is an
API/CLI option; when D4 is active and `s9` is not specified, this prototype uses
the GFN2-xTB value `5.0` from `param_gfn2-xtb.txt`, while non-D4 calculations
resolve to `s9 = 0`. The canonical upstream copy of the GFN1 parameter file is
in the xtb repository:

- Upstream repository: https://github.com/grimme-lab/xtb
- Path: `param_gfn1-xtb.txt`
- Upstream commit checked for provenance: `2b5cd4829290775e575807daee21560f851ff7e1`
- Commit date: 2026-05-16T16:57:44Z

GFN2-derived default used for the experimental D4 ATM scale:

- Upstream repository: https://github.com/grimme-lab/xtb
- Path: `param_gfn2-xtb.txt`
- Source line: `s9          5.00000`
- Upstream commit checked for provenance: `2b5cd4829290775e575807daee21560f851ff7e1`
- Local SHA-256 of checked `param_gfn2-xtb.txt`: `5694f387f02a5d9c233cbf4cedabe8d4a7a863970afa05e4e07c0b4944223cb3`
