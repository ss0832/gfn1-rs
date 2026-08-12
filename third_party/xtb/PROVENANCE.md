# Provenance: GFN1-xTB parameter files

Source repository: https://github.com/grimme-lab/xtb
Commit: `2b5cd4829290775e575807daee21560f851ff7e1` (2026-05-16)
License: LGPL-3.0-or-later (see `COPYING` and `COPYING.LESSER` in this directory,
copied verbatim from the same commit).

## Files

| File | Upstream path | SHA-256 (as bundled, LF-normalized) |
|---|---|---|
| `param_gfn1-xtb.txt` | `/param_gfn1-xtb.txt` | `81a9d7afdee31f5d8fd67a4d8ec966179317c7b100261c9c82d724bafb460399` |
| `param_gfn1-si-xtb.txt` | `/param_gfn1-si-xtb.txt` | `f878fd7f8eea16f326aff283fbc7ca23800045992c0f86536e18ad91816cf7ca` |

Both files are line-for-line identical to the upstream commit; the only
normalization applied is conversion of line endings to LF.

## What they are

- `param_gfn1-xtb.txt` — the official GFN1-xTB parametrization.
  Reference: S. Grimme, C. Bannwarth, P. Shushkov,
  *J. Chem. Theory Comput.* **13**, 1989 (2017), DOI 10.1021/acs.jctc.7b00118.
  The file also carries the **f-in-core lanthanoid parameters ("Ln-xTB")**
  for La–Lu (`$Z=57`…`$Z=71`) — upstream merged them into the standard GFN1
  file; there is no separate Ln parameter file.
  Reference: M. Bursch, A. Hansen, S. Grimme, *Inorg. Chem.* **56**,
  12485 (2017), DOI 10.1021/acs.inorgchem.7b01950.
- `param_gfn1-si-xtb.txt` — the GFN1(Si)-xTB reparametrization for silicon.
  References: DOI 10.1021/acs.jctc.7b00118 (base method) and
  *J. Chem. Inf. Model.* DOI 10.1021/acs.jcim.1c01170 (the Si
  reparametrization).

## Licensing note

These files are redistributed under the terms of the GNU Lesser General Public
License v3.0 or later, as published by the upstream `grimme-lab/xtb` project.
LGPL-3.0-or-later material may be conveyed as part of this GPL-3.0-or-later
project (LGPLv3 §2b). The combined work is distributed under
GPL-3.0-or-later; the files in this directory remain available under
LGPL-3.0-or-later from the upstream repository.

They are embedded into the compiled `gfn1-rs` library via `include_str!`
(see `src/params.rs`, `Gfn1Parameters::builtin()` / `Gfn1Parameters::builtin_si()`)
so that the crate and the Python wheel work out of the box.
