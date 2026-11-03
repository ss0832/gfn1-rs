# sTDA-xTB parameter provenance

`.param_stda1.xtb` and `.param_stda2.xtb` are unmodified parameter files from
[grimme-lab/xtb4stda](https://github.com/grimme-lab/xtb4stda) at commit
`238db6c7df81d5a2ed829f4f682f00aaa87e137d`, LGPL-3.0-or-later.
The original GPL and LGPL notices are retained in COPYING and COPYING.LESSER.
No GFN1 parameters are substituted and no values have been refitted.

The first set generates the valence-basis VTB charge density, and the second
generates dedicated extended-basis xTB orbitals for sTDA-xTB. Kernel and
configuration-selection defaults are supplied by the separate official std2
program; their reviewed source commit is
`8d50f9f2a4b66b534e19bb8a99524be6b7153a47`.

References, API usage, dependency and output precision details are in
`docs/stda-xtb.md`. Exact-byte SHA-256 hashes are recorded in
`third_party/assets.json` and calculated again for each API result.
