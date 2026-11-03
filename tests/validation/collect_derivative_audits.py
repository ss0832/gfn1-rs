"""Collect complete Cartesian audit groups and retain their raw numerical arrays.

Later --input directories replace an entire group. Diagnostic subsets never
replace a full group. Failed earlier measurements are retained separately.
"""
import argparse
import gzip
import hashlib
import json
import math
import tomllib
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_order(order, ndof):
    size = ndof ** order["order"]
    analytic = order["analytic"]
    arrays = [step["finite_difference"] for step in order["steps"]]
    arrays.append(order["richardson_finite_difference"])
    assert order["components"] == size and len(analytic) == size
    assert len(order["steps"]) == 3
    assert all(len(values) == size for values in arrays)
    assert all(math.isfinite(x) for values in [analytic, *arrays] for x in values)
    for values in arrays[-2:]:
        assert all(abs(a - b) <= order["absolute_tolerance"]
                   + order["relative_tolerance"] * max(abs(a), abs(b))
                   for a, b in zip(analytic, values))
    assert order["status"] == "pass"
    return {"order": order["order"], "components": size,
            "absolute_tolerance": order["absolute_tolerance"],
            "relative_tolerance": order["relative_tolerance"],
            "steps_bohr": [step["step_bohr"] for step in order["steps"]],
            "fine_metrics": order["steps"][-1]["metrics"],
            "richardson_metrics": order["richardson_metrics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=Path("tests/data/nuclear-derivatives-report.json"))
    parser.add_argument("--raw", type=Path, default=Path("tests/data/nuclear-derivatives-raw.json.gz"))
    parser.add_argument("--require-current-lockfile", action="store_true")
    args = parser.parse_args()
    groups = {}
    retained = []
    for directory in args.input:
        for path in sorted(directory.glob("group-*.json")):
            group = read(path)
            if group["scope"] != "full_group":
                continue
            for case in group["cases"]:
                detail = read(directory / case["details_file"])
                if case["status"] != "pass":
                    retained.append({"group": path.stem, "measurement": detail})
            groups[path.stem] = (directory, group)
    assert len(groups) == 4, "need molecular, classical, extended and periodic full groups"
    cases = []
    raw_cases = []
    versions = set()
    counts = {str(rank): 0 for rank in range(1, 5)}
    for name, (directory, group) in sorted(groups.items()):
        assert group["status"] == "pass", f"latest full group failed: {name}"
        assert group["executed_cases"] == group["passed_cases"] == len(group["cases"])
        for entry in group["cases"]:
            data = read(directory / entry["details_file"])
            assert data["status"] == "pass" and data["case"] == entry["case"]
            if args.require_current_lockfile:
                assert data.get("dependency_lockfile_sha256") == digest((Path(__file__).resolve().parents[2]/"Cargo.lock").read_bytes()), data["case"]
            assert len(data["orders"]) == data["analytic_orders_requested"]
            assert data["ndof"] == 3 * len(data["atoms"])
            assert [order["order"] for order in data["orders"]] == list(range(1, data["analytic_orders_requested"] + 1))
            versions.add(data["package_version"])
            orders = [verify_order(order, data["ndof"]) for order in data["orders"]]
            for order in orders:
                counts[str(order["order"])] += order["components"]
            cases.append({"case": data["case"], "group": name, "ndof": data["ndof"], "tested_package_version":data["package_version"], "dependency_lockfile_sha256":data.get("dependency_lockfile_sha256"),
                          "status": "pass", "electronic_temperature_K": data["electronic_temperature_K"],
                          "unsupported_higher_orders": data["unsupported_higher_orders"],
                          "audit_driver_sha256": data["audit_driver_sha256"],
                          "seconds": data["seconds"], "orders": orders})
            raw_cases.append(data)
    assert len({case["case"] for case in cases}) == len(cases)
    package_version = tomllib.loads((Path(__file__).resolve().parents[2] / "Cargo.toml").read_text())["package"]["version"]
    raw = {"schema": "gfn1-rs-full-cartesian-raw-v1", "cases": raw_cases,
           "retained_failed_measurements": retained}
    raw_bytes = (json.dumps(raw, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    compressed = gzip.compress(raw_bytes, mtime=0)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.write_bytes(compressed)
    summary = {
        "schema": "gfn1-rs-full-cartesian-summary-v1", "status": "pass_with_documented_scope",
        "package_version": package_version, "tested_package_versions":sorted(versions), "executed_cases": len(cases), "passed_cases": len(cases),
        "cartesian_components_by_order": counts, "total_cartesian_components": sum(counts.values()),
        "finite_difference_step_count": 3,
        "acceptance": "Every finest-step and Richardson component must pass its local absolute-plus-relative tolerance; zeros and permutations are included; no numerical-reference projection or symmetrization",
        "reference": "Central nuclear difference of the next lower derivative, with energy differences for gradients; each lower rank is independently checked in the same ladder",
        "validation_sequence": "All four groups are rerun and their dependency hashes checked against the current lockfile." if args.require_current_lockfile else "Earlier exhaustive molecular/classical and repaired-periodic arrays retain their tested versions. Changed high-rank models and all extended gradients are rerun for this release. All ordinary Rust and installed-wheel Python regressions are rerun on the release source.",
        "coverage_limits": [
            "Representative geometries, not proof for arbitrary chemical space or combinations",
            "Only implemented derivative orders are counted; unsupported higher orders remain listed per case",
            "PH3 rank-4 now converges at unchanged 0 K to a restricted integer-occupation determinant minimum; canonicalization retains potentially non-Aufbau occupation labels. This is not a fractional-occupation ensemble result. Earlier 3000 K fallback failures remain recorded separately.",
            "Periodic scalar FC3 is tested at integer occupations; finite-temperature periodic scalar, periodic SOC and periodic ML are tested through Hessians",
            "Molecular ML is tested through FC3; magnetic, solvent, spin and experimental models in the extended group through gradients",
        ],
        "repairs": [
            "Integer-degenerate second-order density response uses the basis-independent spectral resolvent, including the zero-temperature projector limit",
            "Physical complex k-point MOs use a native complex eigensolver instead of alternating columns of a real embedding",
            "CN-disabled analytic FC4 fails before SCF rather than returning an unsupported value",
            "High-rank 0 K density minimization includes stable multipole Newton responses and preserves the variational occupied/empty subspaces during canonicalization",
        ],
        "retained_failed_measurements": [{"group": entry["group"],
            "case": entry["measurement"]["case"], "status": entry["measurement"]["status"],
            "error": entry["measurement"].get("error")} for entry in retained],
        "raw_data": {"file": args.raw.name, "sha256": digest(compressed), "bytes": len(compressed),
                     "uncompressed_sha256": digest(raw_bytes), "contains": "all analytic, three finite-difference and Richardson arrays, plus earlier failed measurements"},
        "cases": cases,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps({key: summary[key] for key in ["status", "executed_cases", "total_cartesian_components", "cartesian_components_by_order"]}))


if __name__ == "__main__":
    main()
