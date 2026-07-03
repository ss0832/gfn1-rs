# tblite spin-constant provenance

This directory contains an upstream provenance snapshot of the GFN spin
constants (`W`) used by the spin-polarized GFN1-xTB ("spGFN1") implementation
in this crate (`src/spin.rs`, table transcribed into `src/data_tables.rs`).

## Snapshot

- File: `spin.f90`
- Upstream path: `src/tblite/data/spin.f90`
- Upstream project: tblite — <https://github.com/tblite/tblite>
- Repository HEAD at snapshot time: `eb50bbfbe1c0869e2e18c9b7cc13144e5130b6df`
  (main, 2026-06-28).
- Last upstream commit touching `spin.f90`:
  `66175172f1e80b899fd67f79128ee27b2be8dafa` (2025-07-22,
  "GFN lanthanide spin constants & minor adjustments").

The verbatim Fortran source is preserved here as `spin.f90`. The numeric values
in `src/data_tables.rs` (`GFN_SPIN_CONSTANTS`) are a 1:1 transcription of the
`spin_constants(6, 86)` array in this file; the angular-momentum index map
(`lidx`) is reproduced as `gfn_spin_constant(l1, l2, z)`.

## What the values are

`spin_constants(6, 86)`: 86 elements (Z = 1..86), six values per element ordered
`[ss, sp, pp, sd, pd, dd]`. These are the universal atomic spin constants `W_{ll'}`
(shell-resolved by angular momentum l, l' in {s, p, d}); each is the second
derivative of the atomic (spin-)DFT energy with respect to the shell
magnetization, i.e. the magnetic analogue of the Hubbard/hardness parameter.
Units: Hartree (atomic units), consistent with the rest of the GFN1 electronic
energy. The `lidx` map gives the symmetric index:

```
lidx(s,s)=1  lidx(s,p)=lidx(p,s)=2  lidx(p,p)=3
lidx(s,d)=lidx(d,s)=4  lidx(p,d)=lidx(d,p)=5  lidx(d,d)=6
```

## License

`spin.f90` is marked `SPDX-Identifier: LGPL-3.0-or-later` by the upstream
project. This crate is licensed `GPL-3.0-or-later`; LGPL-3.0-or-later
code/data is compatible with and may be distributed as part of a
GPL-3.0-or-later work. The upstream LGPL/GPL license texts are the standard GNU
texts (see <https://www.gnu.org/licenses/>); tblite ships them in its own
`COPYING` / `COPYING.LESSER`.

## References (spin-polarized DFTB / spGFN1 method)

- C. Köhler, G. Seifert, T. Frauenheim, "Spin polarization in SCC-DFTB",
  *Chem. Phys.* **309**, 23–31 (2005); see also arXiv:1605.01360 for the
  collinear spin-DFTB formulation.
- xtb spin-polarization documentation (grimme-lab xtb-docs, `spgfn`):
  shell-resolved magnetization `m_{A,l} = n^α_{A,l} − n^β_{A,l}`, spin energy
  `E_spin = ½ Σ_A Σ_{l,l'} W_{A,ll'} m_{A,l} m_{A,l'}`, spin potential
  `V^σ_{A,l} = ±Σ_{l'} W_{A,ll'} m_{A,l'}` (+ for α, − for β).
- Re-optimized GFN1 spin constants: arXiv:2405.05761. NOTE: this crate uses the
  **canonical tblite-distributed** GFN1 spin constants (the `spin.f90` snapshot
  above), NOT the arXiv:2405.05761 re-optimized set.
