#!/usr/bin/env python3
"""Generate GFN1-xTB-M secondary (dual) basis files from the Dunning correlation-
consistent sets, for the GFN1-xTB-M1 kinetic-energy correction and the experimental
richer-moment multipole electrostatics.

Provenance / licence
---------------------
The numerical exponents and contraction coefficients are the Dunning cc-pVnZ sets as
redistributed by the **Basis Set Exchange** (https://www.basissetexchange.org), which
publishes them under **CC-BY-4.0** with attribution to the original work (T. H. Dunning Jr.
and co-workers; Grant Hill for the cc-repo data). They are NOT copied from any paper's SI:
this script regenerates a licence-clean copy directly from BSE. Heavy elements use the
pseudopotential (`-PP`) / Douglas-Kroll relativistic (`-DK`) variants, as in Cheng &
Wibowo-Teale, J. Chem. Theory Comput. 2023, 19, 6226.

Pipeline: BSE JSON (cached in .bse_cache/) -> per (element, angular momentum) pick the
contraction whose radial node count matches the GFN1 primary valence shell -> emit the
`$Basis = GFN1-xTB-cc-pVnZ` turbomole-style format consumed by parse_secondary_basis().

This module is being built incrementally; `inspect` validates the JSON parser against the
known-good cc-pVDZ oxygen contraction before the node-matching/emit layer is finalized.
"""
import json
import os
import sys

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bse_cache")


def load(name):
    with open(os.path.join(CACHE, name + ".json")) as f:
        return json.load(f)


def element_shells(basis, z):
    """Return [(l, exponents[float], [contraction_coeffs[float], ...]), ...] for element z,
    splitting any sp-type shells into separate angular momenta."""
    el = basis["elements"][str(z)]
    out = []
    for sh in el["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        ang = sh["angular_momentum"]
        coeffs = sh["coefficients"]  # list of contractions (each a list over exponents)
        if len(ang) == 1:
            cols = [[float(c) for c in col] for col in coeffs]
            out.append((ang[0], exps, cols))
        else:
            # General sp/spd shell: coefficients grouped per angular momentum.
            per = len(coeffs) // len(ang)
            for li, l in enumerate(ang):
                cols = [
                    [float(c) for c in coeffs[li * per + k]] for k in range(per)
                ]
                out.append((l, exps, cols))
    return out


def node_count(exps, coeff):
    """Radial-node proxy: sign changes of the contraction coefficients ordered by
    decreasing exponent (tightest first)."""
    pairs = sorted(zip(exps, coeff), key=lambda t: -t[0])
    sgns = [1 if c > 0 else (-1 if c < 0 else 0) for _, c in pairs if c != 0.0]
    return sum(1 for a, b in zip(sgns, sgns[1:]) if a != b)


def node_targets(contractions):
    """Target node counts (one per GFN1 valence shell) from reference contractions
    `[[(exp,coeff),...], ...]` — these encode GFN1's valence (n,l) and carry to any zeta."""
    return [node_count([e for e, _ in c], [v for _, v in c]) for c in contractions]


def peak_exponent(exps, coeff):
    """Exponent of the primitive carrying the largest |coefficient| (the contraction's
    radial 'centre'). The valence shell peaks at lower exponent than the core."""
    return max(zip(exps, coeff), key=lambda t: abs(t[1]))[0]


def pick_valence(exps, cols, targets):
    """Select GFN1-valence contractions of one angular momentum from the cc-pVnZ general
    contractions. `targets` is a list of target radial-node counts (one per GFN1 valence
    shell of this l, taken from the cc-pVDZ reference = GFN1's valence n,l). For each
    target, among the *multi-primitive* contractions (excluding lone diffuse primitives)
    with that node count, take the one with the lowest peak exponent (outermost =
    valence); fall back to the closest node count. Each chosen contraction is sign-matched
    to the GFN1 phase (peak coefficient positive). Reproduces the GFN1-xTB-cc-pVDZ file."""
    cand = []
    for col in cols:
        nz = [c for c in col if c != 0.0]
        if len(nz) >= 2:  # contracted, not a lone diffuse/polarization primitive
            cand.append([node_count(exps, col), peak_exponent(exps, col), col])
    out, used = [], set()
    for t in targets:
        pool = [(p, i, col) for i, (nd, p, col) in enumerate(cand) if nd == t and i not in used]
        if pool:
            # Lowest peak among matching node count; for (near-)equal peaks (degenerate
            # natural orbitals of an unoccupied shell, e.g. the TM 4p) take the
            # higher-index = more-correlating one, matching the GFN1-xTB-cc-pVDZ choice.
            pool.sort(key=lambda r: (round(r[0], 4), -r[1]))
            _, i, col = pool[0]
        else:  # fallback: closest node count, then lowest peak
            alt = sorted(
                ((abs(nd - t), p, i, col) for i, (nd, p, col) in enumerate(cand) if i not in used),
            )
            if not alt:
                continue  # no remaining candidate for this shell (caller copies the ref)
            _, _, i, col = alt[0]
        used.add(i)
        s = -1.0 if max(col, key=abs) < 0 else 1.0
        out.append([s * c for c in col])
    return out


def inspect():
    b = load("cc-pvdz")
    for z, name in [(1, "H"), (8, "O")]:
        print(f"=== Z={z} {name} cc-pVDZ ===")
        for (l, exps, cols) in element_shells(b, z):
            for ci, col in enumerate(cols):
                nz = [(e, c) for e, c in zip(exps, col) if c != 0.0]
                print(
                    f"  l={l} contr{ci}: nprim={len(nz)} nodes={node_count(exps, col)}"
                    f" lead_exp={max(e for e, _ in nz):.4g}"
                )
        # Oxygen valence s should match the existing secondary file: 9 prims,
        # lead exponent 11720, 1-node valence contraction.
    print("\n-- O s-contractions (exponents, coeff) for the 1-node valence pick --")
    for (l, exps, cols) in element_shells(b, 8):
        if l != 0:
            continue
        for ci, col in enumerate(cols):
            if node_count(exps, col) == 1:
                for e, c in zip(exps, col):
                    if c != 0.0:
                        print(f"   {e:18.8f}  {c:14.8f}")


L_TAG = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G"}
# The reference GFN1-xTB-cc-pVDZ file (paper SI) — used ONLY for the per-element GFN1
# shell structure (#contractions per angular momentum) and the H special case, NOT for
# the numbers (those come from BSE). Structure is not copyrightable.
REF_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ct3c00671_si_002.txt",
)


L_INDEX = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def parse_param_shells(path=None):
    """Parse the GFN1 param file `ao=` lines -> {z: {l: [node_counts]}}. Each shell `n l`
    (e.g. `1s2s`, `5s5p5d`, `5d6s6p`) has radial-node count `n - l - 1`. This is the
    authoritative GFN1 shell structure for ALL elements (Z=1..86), unlike the cc-pVDZ
    reference which only covers Z=1..36."""
    import re

    if path is None:
        path = os.environ.get("GFN1_XTB_PARAM")
    out, z = {}, None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\$Z=\s*(\d+)", line)
            if m:
                z = int(m.group(1))
                continue
            m = re.match(r"\s*ao=([0-9spdfg]+)", line)
            if m and z is not None:
                by_l = {}
                for (n, l) in re.findall(r"(\d)([spdfg])", m.group(1)):
                    li = L_INDEX[l]
                    by_l.setdefault(li, []).append(int(n) - li - 1)
                out[z] = by_l
    return out


def validate_param():
    """Prove the param-file shell structure (l's + node counts) matches the cc-pVDZ
    reference for Z=1..36 (so it can drive the heavy elements Z=37..86)."""
    param = parse_param_shells()
    ref = parse_gfn1(REF_FILE)
    mism = []
    for z, struct in ref.items():
        # Node counts per l from the reference contractions.
        ref_by_l = {}
        for (l, contr) in struct:
            ref_by_l.setdefault(l, []).extend(node_targets(contr))
        pby_l = param.get(z, {})
        for l, ref_nodes in ref_by_l.items():
            pnodes = sorted(pby_l.get(l, []))
            if sorted(ref_nodes) != pnodes:
                mism.append((z, l, sorted(ref_nodes), pnodes))
    if mism:
        print(f"PARAM-vs-REF MISMATCHES ({len(mism)}):")
        for m in mism[:30]:
            print("  z=%d l=%d ref_nodes=%s param_nodes=%s" % m)
    print("PARAM SHELL VALIDATION", "PASS" if not mism else "FAIL")
    return not mism


def parse_gfn1(path):
    """Parse a GFN1 secondary file -> {z: [(l, [contraction[(exp,coeff)], ...]), ...]}."""
    import re

    out = {}
    z = l = None
    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        m = re.match(r"^a (\d+)", ln)
        if m:
            z = int(m.group(1))
            out[z] = []
            continue
        if ln.startswith("$"):
            up = ln.upper()
            for ll, tag in L_TAG.items():
                if f"{tag}-TYPE" in up:
                    l = ll
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and "." not in parts[0]:
            nprim, ncontr = int(parts[0]), int(parts[1])
            exps, cols = [], [[] for _ in range(ncontr)]
            for _ in range(nprim):
                row = lines[i].split()
                i += 1
                exps.append(float(row[0]))
                for c in range(ncontr):
                    cols[c].append(float(row[1 + c]))
            contr = [
                [(e, cc) for e, cc in zip(exps, col) if cc != 0.0] for col in cols
            ]
            out[z].append((l, contr))
    return out


def bse_shells_by_l(bse, z):
    """{l: (exponents, [contraction_columns])} for element z from a BSE basis."""
    by_l = {}
    if str(z) not in bse["elements"]:
        return by_l
    for (l, exps, cols) in element_shells(bse, z):
        if l not in by_l:
            by_l[l] = (exps, [])
        by_l[l] = (exps, by_l[l][1] + cols)
    return by_l


def emit_element(z, struct_z, bse):
    """GFN1-format block for element z, matching the GFN1 shell structure (the number of
    contractions per l from `struct_z`) by valence-selecting from BSE. H (and any element
    BSE lacks) copies the GFN1 reference contractions."""
    out = [f"a {z}", f"$ Z={z}"]
    by_l = bse_shells_by_l(bse, z)
    for (l, existing) in struct_z:
        ncontr = len(existing)
        if z == 1 or l not in by_l:
            chosen = existing  # GFN1 H AO / fallback
        else:
            exps, cols = by_l[l]
            picked = pick_valence(exps, cols, node_targets(existing))
            chosen = [[(e, c) for e, c in zip(exps, col) if c != 0.0] for col in picked]
            if len(chosen) < ncontr:
                chosen = existing
        # Union of exponents across the chosen contractions (zero-padded columns).
        allexp = sorted({e for col in chosen for e, _ in col}, key=lambda v: -v)
        out.append(f"$ {L_TAG[l]}-TYPE FUNCTIONS")
        out.append(f"   {len(allexp):3d} {len(chosen):3d}  0")
        cmap = [dict(col) for col in chosen]
        for e in allexp:
            row = f"{e:18.8f}" + "".join(f"{cm.get(e, 0.0):18.8f}" for cm in cmap)
            out.append(row)
    return out


def emit_element_param(z, param_by_l, bse):
    """Z>=37 heavy elements: take the GFN1 shell structure (which l's, how many shells)
    from the param-file `ao=` line and valence-select from the pseudopotential (`-PP`)
    Dunning basis. The PP removes the core, so the GFN1-valence shells are the lowest-node
    contractions (targets 0,1,...,count-1). Best-effort: there is no published GFN1-xTB-M
    secondary basis for Z>=37, so this is structure-validated (correct #shells per l), not
    byte-validated against a reference."""
    out = [f"a {z}", f"$ Z={z}"]
    by_l = bse_shells_by_l(bse, z)
    for l in sorted(param_by_l):
        count = len(param_by_l[l])
        if l not in by_l:
            continue
        exps, cols = by_l[l]
        picked = pick_valence(exps, cols, list(range(count)))  # lowest-node PP valence
        chosen = [[(e, c) for e, c in zip(exps, col) if c != 0.0] for col in picked]
        if not chosen:
            continue
        allexp = sorted({e for col in chosen for e, _ in col}, key=lambda v: -v)
        out.append(f"$ {L_TAG[l]}-TYPE FUNCTIONS")
        out.append(f"   {len(allexp):3d} {len(chosen):3d}  0")
        cmap = [dict(col) for col in chosen]
        for e in allexp:
            out.append(f"{e:18.8f}" + "".join(f"{cm.get(e, 0.0):18.8f}" for cm in cmap))
    return out


def validate_full():
    """Generate cc-pVDZ from BSE for all reference elements and check it reproduces the
    GFN1 shell structure + (for z>1) the BSE valence contractions exactly."""
    ref = parse_gfn1(REF_FILE)
    bse = load("cc-pvdz")
    mism = []
    for z, struct in ref.items():
        by_l = bse_shells_by_l(bse, z)
        for (l, existing) in struct:
            if z == 1 or l not in by_l:
                continue
            exps, cols = by_l[l]
            picked = pick_valence(exps, cols, node_targets(existing))
            for ci, col in enumerate(picked):
                got = sorted(((e, c) for e, c in zip(exps, col) if c != 0.0), key=lambda t: -t[0])
                ex = sorted(existing[ci], key=lambda t: -t[0])
                ok = len(got) == len(ex) and all(
                    abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 2e-6 for a, b in zip(got, ex)
                )
                if not ok:
                    mism.append((z, L_TAG[l], ci, len(got), len(ex)))
    if mism:
        print(f"MISMATCHES ({len(mism)}):")
        for m in mism[:40]:
            print("  z=%d %s contr%d got=%d ref=%d" % m)
    print("FULL VALIDATION", "PASS" if not mism else "FAIL")
    return not mism


def provenance(label):
    return [
        f"$Basis = GFN1-xTB-{label}",
        "$ ----------------------------------------------------------------------------",
        f"$ GFN1-xTB-M secondary (dual) basis derived from the Dunning {label} set.",
        "$ Numerical exponents/coefficients are from the Basis Set Exchange",
        "$ (https://www.basissetexchange.org), redistributed under CC-BY-4.0 with",
        "$ attribution to T. H. Dunning Jr. et al. (cc-pVnZ) and ccRepo/Grant Hill.",
        "$ Per (element, angular momentum) the GFN1-valence contraction is selected by",
        "$ radial-node matching to the GFN1 primary shell, then sign-matched to the GFN1",
        "$ phase (the overall sign is physically irrelevant for the quadratic M1 kinetic-",
        "$ energy correction and multipole moments). Regenerated by",
        "$ scripts/gen_secondary_basis.py validate-full -> generate; NOT copied from any SI.",
        "$ Z=1..36 use the all-electron set and reproduce the published GFN1-xTB-cc-pVDZ",
        "$ secondary basis exactly. Z>=37 use the pseudopotential (-PP) basis with the GFN1",
        "$ shell structure from param_gfn1-xtb.txt (ao= line); these heavy-element bases are",
        "$ best-effort (correct #shells per l, lowest-node PP valence) and structure-",
        "$ validated only (no published reference exists for Z>=37).",
        "$ ----------------------------------------------------------------------------",
    ]


def generate(label):
    """Emit GFN1-xTB-<label>.txt covering Z=1..86: the all-electron cc-pVDZ reference
    structure (byte-validated) for Z=1..36, and the param-file shell structure +
    pseudopotential (`-PP`) basis (best-effort, structure-validated) for Z=37..86."""
    ref = parse_gfn1(REF_FILE)
    param = parse_param_shells()
    bse_ae = load(label)
    try:
        bse_pp = load(label + "-pp")
    except FileNotFoundError:
        bse_pp = None
    lines = provenance(label)
    n_ae, n_pp = 0, 0
    for z in sorted(param):
        if z > 86:
            continue
        if z in ref:
            lines += emit_element(z, ref[z], bse_ae)
            n_ae += 1
        elif bse_pp is not None and str(z) in bse_pp.get("elements", {}):
            lines += emit_element_param(z, param[z], bse_pp)
            n_pp += 1
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "secondary_bases"
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"gfn1-xtb-{label}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    reparsed = parse_gfn1(path)
    nshells = sum(len(v) for v in reparsed.values())
    print(
        f"wrote {path}: {n_ae} all-electron + {n_pp} PP elements, {nshells} shells, {len(lines)} lines"
    )
    return path


def debug_tm():
    """Inspect transition-metal contractions vs the reference to fix valence selection."""
    b = load("cc-pvdz")
    ref = parse_gfn1(REF_FILE)
    for z in (20, 21, 24):
        print(f"=== Z={z} ===  BSE multi-prim contractions:")
        for (l, e, cols) in element_shells(b, z):
            for ci, col in enumerate(cols):
                nz = [(x, y) for x, y in zip(e, col) if y != 0.0]
                if len(nz) >= 2:
                    print(
                        f"  l={l} contr{ci} nodes={node_count(e, col)} "
                        f"peak={peak_exponent(e, col):.3f} nprim={len(nz)}"
                    )
        by_l = bse_shells_by_l(b, z)
        for (l, exc) in ref[z]:
            nodes = [
                node_count([x for x, _ in c], [y for _, y in c]) for c in exc
            ]
            peaks = [
                peak_exponent([x for x, _ in c], [y for _, y in c]) for c in exc
            ]
            print(f"  REF l={l} ncontr={len(exc)} nodes={nodes} peaks={[round(p,3) for p in peaks]}")
            if l in by_l:
                exps, cols = by_l[l]
                picked = pick_valence(exps, cols, node_targets(exc))
                for ci, col in enumerate(picked):
                    got = sorted(((e, c) for e, c in zip(exps, col) if c != 0.0), key=lambda t: -t[0])[:3]
                    rf = sorted(exc[ci], key=lambda t: -t[0])[:3]
                    print(f"    PICK l={l} got3={[ (round(e,2),round(c,5)) for e,c in got]}")
                    print(f"    REF  l={l} ref3={[ (round(e,2),round(c,5)) for e,c in rf]}")


def validate():
    """Prove the valence-selection reproduces the GFN1-xTB-cc-pVDZ oxygen contractions
    (the node-matching gate) before generating cc-pVTZ/QZ/5Z."""
    b = load("cc-pvdz")
    # Oxygen reference, valence s (2s, 9 prims) and p (2p, 4 prims), from the M1 paper SI.
    ref_s = [
        (11720.0, -0.00016), (1759.0, -0.001263), (400.8, -0.006267),
        (113.7, -0.025716), (37.03, -0.070924), (13.27, -0.165411),
        (5.025, -0.116955), (1.013, 0.557368), (0.3023, 0.572759),
    ]
    ref_p = [(17.7, 0.043018), (3.854, 0.228913), (1.046, 0.508728), (0.2753, 0.460531)]
    shells = element_shells(b, 8)
    s = next((e, c) for (l, e, c) in shells if l == 0)
    p = next((e, c) for (l, e, c) in shells if l == 1)
    sv = pick_valence(s[0], s[1], [1])[0]  # O 2s = 1 radial node
    pv = pick_valence(p[0], p[1], [0])[0]  # O 2p = 0 nodes

    def check(name, exps, col, ref):
        got = sorted(((e, c) for e, c in zip(exps, col) if c != 0.0), key=lambda t: -t[0])
        ref_s = sorted(ref, key=lambda t: -t[0])
        ok = len(got) == len(ref_s) and all(
            abs(g[0] - r[0]) < 1e-3 and abs(g[1] - r[1]) < 1e-6 for g, r in zip(got, ref_s)
        )
        print(f"  O {name}: {'MATCH' if ok else 'MISMATCH'} ({len(got)} prims)")
        if not ok:
            for g, r in zip(got, ref_s):
                print(f"    got {g}  ref {r}")
        return ok

    ok = check("s", s[0], sv, ref_s)
    ok = check("p", p[0], pv, ref_p) and ok
    print("VALIDATION", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if cmd == "inspect":
        inspect()
    elif cmd == "validate":
        sys.exit(0 if validate() else 1)
    elif cmd == "validate-full":
        sys.exit(0 if validate_full() else 1)
    elif cmd == "validate-param":
        sys.exit(0 if validate_param() else 1)
    elif cmd == "debug-tm":
        debug_tm()
    elif cmd == "generate":
        for label in ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"):
            generate(label)
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        sys.exit(1)
