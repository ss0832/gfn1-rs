# The `sigma-hole` CAMM preset (v0.4.4)

`sigma-hole` is a named **CAMM-on-mDFTB2** parameter preset — the unified σ-hole preset baked into
the crate. It fills, in one flag, the per-element CAMM range factor **κ** *and* a **per-element**
on-site penalty scale **s_onsite** from the multi-regime fit in coordinate descent over all
12 tuned elements' κ + per-element s_onsite, GFN1-normalized across HAL59 / A24 / S66 / SSI
interaction energies and MOR41 / NBPRC gradients; objective 6.54.

It resolves a split a single global s_onsite scalar cannot express: **halogen σ-holes want
`s_onsite ≈ 0`** (temper the on-site cumulative-moment over-penalty) while **tetrel centers (Si)
want `s_onsite ≈ 1`**. All other CAMM presets (`polar`, `halogen`, `halogen-v1`,
`halogen-allgrad`) carry only a single global s_onsite; `sigma-hole` is the first with per-element
s_onsite.

## What it is (and isn't) for

- **For:** σ-hole non-covalent interactions — halogen bonds (HAL59) and tetrel σ-holes — where the
  anisotropic CAMM/AES electrostatics on cumulative atomic multipoles improve on GFN1's isotropic
  point charges.
- **Not for:** transition-metal or covalent-framework *geometry*. Like the other CAMM presets it is
  geometry-neutral-to-modestly-worse there. On the 42-atom Ni complex `test.xyz` it reproduces plain
  GFN1 to the digit (RMSD to wB97MV/def2SVP 0.4475 Å for both; the residual is CF₃/ring torsion,
  outside CAMM's σ-hole lever). Across **50 metal-balanced tmQM complexes** (RMSD to g-xTB) it
  averages **0.093 Å vs plain GFN1's 0.078 Å** (median 0.067 vs 0.057; closer on only 13/49) — i.e.
  slightly *worse* on TM geometry, between GFN1 and the `polar` preset (0.127) and near
  `halogen-allgrad` (0.085). For TM geometry / reaction energies use plain GFN1. This is the
  CAMM-domain finding, not a regression.

## Parameters (Z → value)

Global κ = 1.0, s_AES = 1.0, **global s_onsite = 0.05** (the fallback for off-list elements, e.g.
transition metals). Per-element overrides:

| element | H | B | C | N | O | F | Si | P | S | Cl | Br | I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **κ** | 1.9 | 4.9 | 1.2 | 2.1 | 2.575 | 7.375 | 1.775 | 0.225 | 7.75 | 2.475 | 5.625 | 4.25 |
| **s_onsite** | 0.0 | 0.0 | 0.2 | 0.15 | 1.22 | 0.02 | 1.23 | 0.24 | 0.52 | 0.11 | 0.02 | 0.0 |

Baked in [`camm_preset`](../src/electronic.rs) (`electronic::camm_preset("sigma-hole")` returns
`(global_κ, element_κ, s_AES, global_s_onsite, element_s_onsite)`); implies `multipole = true` +
`multipole_model = camm_on_mdftb2`.

## Usage

**CLI** — single point / gradient / optimization:

```bash
gfn1_rs_cli --param param_gfn1-xtb.txt --camm-preset sigma-hole mol.xyz
gfn1_rs_cli --param param_gfn1-xtb.txt --camm-preset sigma-hole --optimize --opt-output mol_opt.xyz mol.xyz
```

Explicit `--camm-damp* / --camm-aes-scale / --camm-onsite-scale*` still override the preset (the
preset only fills the knobs you did not set).

**Python (native)** — `Gfn1NativeCalculator` (no ASE):

```python
import gfn1_rs
calc = gfn1_rs.Gfn1NativeCalculator(param_path=gfn1_rs.default_param_path(), camm_preset="sigma-hole")
res = calc.calculate(numbers=[9, 6, ...], positions=[[...], ...], unit="angstrom")
opt = calc.optimize(numbers=[...], positions=[...], unit="angstrom", gradient_tolerance=1e-3)
```

**Python (ASE)** — `GFN1RSCalculator`:

```python
from gfn1_rs.ase import GFN1RSCalculator
atoms.calc = GFN1RSCalculator(param_path="param_gfn1-xtb.txt", camm_preset="sigma-hole")
e = atoms.get_potential_energy(); f = atoms.get_forces()
```
