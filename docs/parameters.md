# Parameters

## The official parametrization is bundled

Since v0.5.0 the official GFN1-xTB parameter files ship **inside the crate** — no
download, no environment variable, nothing to configure. They are compiled in with
`include_str!` from `third_party/xtb/`:

| Constant (`gfn1_rs::params`) | File | Parametrization |
| --- | --- | --- |
| `BUILTIN_GFN1_PARAM_TEXT` | `third_party/xtb/param_gfn1-xtb.txt` | GFN1-xTB |
| `BUILTIN_GFN1_SI_PARAM_TEXT` | `third_party/xtb/param_gfn1-si-xtb.txt` | GFN1(Si)-xTB (silicon reparametrization) |

Both are **verbatim** (LF-normalized) copies from
[`grimme-lab/xtb`](https://github.com/grimme-lab/xtb)`@2b5cd48`, LGPL-3.0-or-later.
Redistribution inside this GPL-3.0-or-later project is per LGPLv3. Provenance and
licence texts live in `third_party/xtb/PROVENANCE.md`, `COPYING`, `COPYING.LESSER`,
and in the top-level [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### References — and where the lanthanoids ("Ln-xTB") live

| Parametrization | Reference |
| --- | --- |
| GFN1-xTB | S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**, 1989 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118) |
| Ln f-in-core (La–Lu) | M. Bursch, A. Hansen, S. Grimme, *Inorg. Chem.* **56**, 12485 (2017), DOI [10.1021/acs.inorgchem.7b01950](https://doi.org/10.1021/acs.inorgchem.7b01950) |
| GFN1(Si)-xTB | *J. Chem. Inf. Model.*, DOI [10.1021/acs.jcim.1c01170](https://doi.org/10.1021/acs.jcim.1c01170) |

There is **no separate Ln-xTB parameter file** upstream: the f-in-core
lanthanoid parameters were merged into the standard `param_gfn1-xtb.txt`
(`$Z=57`…`$Z=71` blocks are present in the bundled copy), so lanthanoid
complexes work out of the box with the default builtin set. The same DOIs are
carried at runtime by `Gfn1Parameters::references()` and appear in the
provenance banner (`source_description()`, the CLI startup line, and the
Python `param_source()` / `__repr__`).

The one-line provenance tag is the public constant

```rust
pub const BUILTIN_PARAM_PROVENANCE: &str = "grimme-lab/xtb@2b5cd48, LGPL-3.0-or-later";
```

## Resolution order

`Gfn1Parameters::resolve` is the single entry point every front end uses:

```rust
use gfn1_rs::Gfn1Parameters;

let params = Gfn1Parameters::resolve(None)?;                     // bundled GFN1-xTB
let params = Gfn1Parameters::resolve(Some("builtin:si"))?;       // bundled GFN1(Si)-xTB
let params = Gfn1Parameters::resolve(Some("my_param.txt"))?;     // explicit file
```

Precedence, highest first:

1. the **explicit spec** (`--param` / `param_path=` / the `explicit` argument);
2. the **`GFN1_XTB_PARAM`** environment variable (constant `GFN1_PARAM_ENV`);
3. the **builtin** GFN1-xTB set.

The explicit spec is either a file path or one of the builtin specifiers, matched
case-insensitively:

| Spec | Selects |
| --- | --- |
| `builtin`, `builtin:gfn1` | bundled GFN1-xTB |
| `builtin:si`, `builtin:gfn1-si`, `builtin-si` | bundled GFN1(Si)-xTB |
| anything else | treated as a file path (`Gfn1Parameters::from_file`) |

Lower-level constructors remain public: `Gfn1Parameters::builtin()`,
`Gfn1Parameters::builtin_si()`, `Gfn1Parameters::from_file(path)`, and
`Gfn1Parameters::from_str(text)`.

> **`resolve_param_path` is not the same function.** It is the legacy *path-only*
> resolver (explicit string, else `GFN1_XTB_PARAM`) and it still **errors** when
> neither is set — it has no builtin fallback. New code should call
> `Gfn1Parameters::resolve`.

GFN2 parameter files are rejected at parse time, deliberately.

## Provenance reporting

Every loaded parameter set records where it came from in `ParamSource`:

```rust
pub enum ParamSource {
    Builtin,            // bundled GFN1-xTB
    BuiltinSi,          // bundled GFN1(Si)-xTB
    EnvVar(String),     // file named by GFN1_XTB_PARAM
    Explicit(String),   // file named on the command line / API
    Inline,             // parsed from an in-memory string
}
```

`Gfn1Parameters::source_description()` renders the one-liner used by the front ends:

- **CLI** prints it at startup: `parameters: GFN1-xTB (builtin, grimme-lab/xtb@2b5cd48, LGPL-3.0-or-later)`
- **Python**: `calc.param_source()`, and it appears in `repr(calc)`.
- **ASE**: `GFN1RSCalculator` logs it through the `gfn1_rs` logger.

Python's `default_param_path()` returns `$GFN1_XTB_PARAM` when set and the literal
string `"builtin"` otherwise, so it is always a valid `param_path=` argument and
never raises.

## Parameter round-trip and derivatives

`Gfn1Parameters::to_param_string()` / `write_param_file()` emit the
`param_gfn1-xtb.txt` format deterministically with value-exact round-trip.
Individual scalars are addressed with `ParameterTarget` (`glob:<key>`,
`elem:<Z>:<KEY>[:idx]`, `pair:<ZA>:<ZB>`); see
[rust-api.md](rust-api.md#parameters-round-trip-and-finite-difference-derivatives).

Since v0.5.0 the **halogen-bond term reads its constants from the parameter file**
(`xbdamp`, `xbrad` globals and the per-element `CXB`, scaled by 0.1) instead of
hardcoding them. Editing those parameters previously had no effect and their
parameter derivatives were silently zero. Physics is unchanged with the official
parametrization — the old hardcoded constants equal the builtin values.

## Reference data for the dispersion terms

| Data | Bundled at | Override |
| --- | --- | --- |
| D3(BJ) `s-dftd3` reference table | `third_party/simple-dftd3` (LGPL-3.0-or-later) | `--d3-reference PATH` / `GFN1_D3_REFERENCE` (`GFN1_D3_REFERENCE_ENV`) |
| DFT-D4 reference data (experimental D4 path) | `third_party/dftd4` | — |
| tblite atomic spin constants `W` (spGFN1) | `third_party/tblite/spin.f90` (LGPL-3.0-or-later) | — |
| Dunning cc-pVnZ secondary bases | hardcoded in `secondary_bases` (Basis Set Exchange, CC-BY-4.0) | `--multipole-secondary-basis <name\|file>` |

D4 damping constants `a1`, `a2`, `s8` are read from the **active** GFN1 parameter
file; `s9` is an API/CLI option that defaults to the GFN2-xTB value `5.0` only when
D4 is active (ordinary non-D4 runs use `s9 = 0`).

## Unit constants (reporting vs model)

`constants::HARTREE_TO_EV` / `EV_TO_HARTREE` are CODATA-2018
(`27.211386245988`) and are used for **unit reporting only**. The model-side
conversion that reads the parameter file (`basis::EV_TO_HARTREE`) deliberately
stays on the legacy `1/27.21138505`: GFN1 was parametrized against it, and
switching it shifts caffeine by ~2e-6 Eh and breaks tblite parity. The split is
documented on both constants.

`constants::KB_HARTREE_PER_K = 3.1668115634556e-6` (exact SI `k_B` over the
CODATA-2018 Hartree) is the **single** Boltzmann constant for every
finite-temperature path — restricted SCC, spin-polarized SCC, CPXTB, periodic
SCC/Hessian, and the Grüneisen heat capacities. Before v0.5.0 the restricted and
spin paths used two slightly different values, breaking their byte-identity at
`T > 0`.
