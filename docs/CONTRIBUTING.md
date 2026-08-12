# Contributing notes

## Docs are updated when the feature lands — same commit, no exceptions

This is a standing project requirement, not a suggestion.

A feature is **not done** when the code and its gates are green. It is done when
the documentation says what it does, how to call it, and where it breaks. Practically:

1. **Same commit.** The doc edit ships in the commit that lands the feature. Not a
   follow-up, not "later in the phase". A commit that adds a public entry point
   and does not touch `docs/` should not exist.
2. **Pick the owning page.** The docs are organised one page per subsystem so new
   material is *appended*, not bolted onto a growing monolith:

   | Page | Owns |
   | --- | --- |
   | [parameters.md](parameters.md) | parameter sourcing, provenance, reference data, unit constants |
   | [derivatives.md](derivatives.md) | the nuclear derivative ladder + the term registry + gate philosophy |
   | [finite-temperature.md](finite-temperature.md) | the response solvers and Fermi smearing |
   | [pbc.md](pbc.md) | everything periodic |
   | [limitations.md](limitations.md) | known gaps and silent failure modes |
   | [scope.md](scope.md) | the one-line feature-matrix row |
   | [rust-api.md](rust-api.md) / [python-api.md](python-api.md) | the callable surface |

   If a new feature does not fit any page, add a page and register it in
   [README.md](README.md) — do not wedge it into an unrelated one.
3. **Quote API names exactly.** Copy the identifier from the source, including the
   module path. If a name is not re-exported at the crate root, say so and give
   the path the user must actually type. Verify with `grep`, not memory.
4. **Document the gaps in the same breath.** Every explicit `Err` that a user can
   hit belongs in [limitations.md](limitations.md), **quoted verbatim**, so the
   error text they see is searchable in the docs. Anything that can be *silently*
   wrong is more important still — flag it loudly, with the measured error if one
   exists.
5. **Update the feature matrix row.** [scope.md](scope.md) is the index a reader
   scans first; a stale row there is worse than a missing page.
6. **Delete what is no longer true.** Stale claims are the expensive kind of doc
   debt. When a limitation is lifted, remove it from
   [limitations.md](limitations.md) in the same commit that lifts it — do not
   leave it hedged.

## What "gated" means

New analytic derivative work is expected to arrive with the gates described in
[derivatives.md](derivatives.md#5-the-verification-gate-philosophy): analytic
order `n` against a central FD of the validated analytic order `n−1`, with the
`h²` ladder demonstrated (residual dropping by ×4 when `h` is halved), plus the
exact invariances (translational / acoustic sum rule, permutation symmetry,
contraction identities). Quote the measured numbers in the docs — they are the
evidence, and they let the next person recognise a regression.

## Term registry

When a term gains a new analytic derivative order, update its
`max_analytic_order` in `src/terms.rs` — **exactly one place** — and update the
table in [derivatives.md](derivatives.md#0-the-term-registry-require_order).

## Build and test conventions

```bash
cargo build --release                # CLI -> target/release/gfn1_rs_cli
cargo test                           # parameters resolve to the bundled set; no env var needed
cargo test -- --ignored              # long-running gates (FD probes, finite-T Hessian ladders)
```

- Tests must use `Gfn1Parameters::resolve(None)` so they run against the bundled
  parametrization instead of silently no-opping when `GFN1_XTB_PARAM` is unset.
- `GFN1_FD_TIGHT=1` tightens the finite-difference probe; `GFN1_PROFILE=1` emits
  per-scope native timings; `GFN1_FAER_THREADS=1` forces single-threaded `faer`
  for reproducible benchmarking.
- `cargo clippy --all-targets` is expected to be clean, and the tree builds with
  zero compiler warnings across all targets.
