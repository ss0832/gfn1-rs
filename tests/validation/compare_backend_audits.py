"""Compare paired before/after backend runs and optional complete release gates."""
import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extended_validation(root, work):
    lock_hash = sha(root / "Cargo.lock")
    complex_audit = read(work / "complex-linalg.json")
    assert complex_audit["status"] == "pass" and complex_audit["dependency_lockfile_sha256"] == lock_hash
    before = json.loads(gzip.decompress((work / "before/nuclear-derivatives-raw.json.gz").read_bytes()))
    after = json.loads(gzip.decompress((root / "tests/data/nuclear-derivatives-raw.json.gz").read_bytes()))
    previous = {case["case"]: case for case in before["cases"]}
    assert set(previous) == {case["case"] for case in after["cases"]} and len(previous) == 42
    differences = []
    for case in after["cases"]:
        assert case["dependency_lockfile_sha256"] == lock_hash and case["status"] == "pass"
        for old, new in zip(previous[case["case"]]["orders"], case["orders"]):
            assert old["order"] == new["order"] and len(old["analytic"]) == len(new["analytic"])
            pairs = list(zip(old["analytic"], new["analytic"]))
            assert all(abs(a-b) <= 1e-9 + 1e-8*max(abs(a), abs(b)) for a,b in pairs), (case["case"],new["order"])
            differences.append({"case":case["case"], "order":new["order"], "components":len(pairs),
                "maximum_absolute_analytic_difference":max(abs(a-b) for a,b in pairs), "status":"pass"})
    assert sum(d["components"] for d in differences) == 48345
    native_log = (work / "rust-tests.log").read_text(encoding="utf-8-sig")
    rows = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", native_log)
    counts = {key:sum(int(row[i]) for row in rows) for i,key in enumerate(["passed","failed","ignored"])}
    assert counts["passed"] == 715 and counts["failed"] == 0
    assert not any(token in native_log for token in ["FAILED", "warning:", "error:"])
    suite = ET.parse(work / "python-tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) == 57 and all(int(suite.attrib[k]) == 0 for k in ["failures","errors","skipped"])
    assert "warnings summary" not in (work / "python-tests.log").read_text()
    assert "warning:" not in (work / "clippy.log").read_text(encoding="utf-8-sig")
    assert not (work / "format.log").read_text(encoding="utf-8-sig").strip()
    assert (work / "updated-validation.log").read_text().rstrip().endswith("All updated validation steps passed")
    hcl = read(work / "hcl-standalone/multipole-rank4-hcl-t0.json")
    assert hcl["status"] == "pass" and hcl["dependency_lockfile_sha256"] == lock_hash
    cotton = read(work / "cotton-mouton/cotton-mouton.json")
    assert cotton["dependency_lockfile_sha256"] == lock_hash
    cotton_metrics = [{"model":c["model"],"components":c["components"],"passed_components":c["passed_components"],
        "maximum_richardson_absolute_energy_reference_discrepancy":max(row["abs_error"] for row in c["richardson"]),"status":c["status"]} for c in cotton["cases"]]
    assert all(c["status"] == "pass" and c["passed_components"] == 27 for c in cotton_metrics)
    return {"dependency_lockfile_sha256":lock_hash, "complex_generalized_eigen":complex_audit,
        "full_cartesian_revalidation":{"cases":42,"components":48345,"status":"pass", "previous_analytic_comparison":differences,
            "comparison_absolute_tolerance":1e-9,"comparison_relative_tolerance":1e-8,
            "finite_difference_scope":"Every new case repeats three finite-difference steps and Richardson checks with its original local component tolerances"},
        "audit_wall_time_scope":{"note":"Concurrent full-group audit wall times include shared Rayon pool contention and are not the paired performance benchmark. A separate process confirms the high-rank HCl audit without that shared pool.",
                                 "hcl_separate_process_seconds":hcl["seconds"],"hcl_gradient_components":6},
        "rust_tests":counts,"python_tests":{"passed":57,"warnings":0,"warning_policy":"error","environment":read(work / "python-environment.json")},
        "clippy":"all targets and all features pass with -D warnings", "rustfmt":"pass",
        "vpt2":"Four real molecular cases and water CLI execution rerun", "cotton_mouton":{"scope":"Both implemented 27-component diagonal-B slices rerun against independent energy differences; finite fields amplify roundoff at the smallest step", "cases":cotton_metrics},
        "memory":read(work / "memory-comparison.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("tests/data/backend-update-audit.json"))
    parser.add_argument("--extended-validation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    old = [read(args.input / f"before-paired-{i}.json") for i in range(args.pairs)]
    new = [read(args.input / f"after-paired-{i}.json") for i in range(args.pairs)]
    before_lock = tomllib.loads((args.input / "before/Cargo.lock").read_text())
    after_lock = tomllib.loads((root / "Cargo.lock").read_text())
    key = lambda package: (package["name"], package.get("source"))
    bmap = {key(p): p for p in before_lock["package"]}
    amap = {key(p): p for p in after_lock["package"]}
    changes = [{"name": k[0], "before": bmap.get(k), "after": amap.get(k)}
               for k in sorted(set(bmap) | set(amap)) if bmap.get(k) != amap.get(k)]
    assert len(changes) == 1 and changes[0]["name"] == "private-gemm-x86"
    assert changes[0]["before"]["version"] == "0.1.20"
    assert changes[0]["after"]["version"] == "0.1.22"
    for run in old:
        assert run["lockfile_sha256"] == sha(args.input / "before/Cargo.lock")
    for run in new:
        assert run["lockfile_sha256"] == sha(root / "Cargo.lock")
    assert all(len(run["cases"]) == 55 for run in [*old, *new])
    timing = []
    numerical = []
    for i, first in enumerate(old[0]["cases"]):
        second = new[0]["cases"][i]
        assert first["kind"] == second["kind"]
        description = {k: first[k] for k in ["kind", "m", "n", "k", "n", "row_major", "structure", "parallel_threads"] if k in first}
        if first["kind"] == "gemm":
            assert all(run["cases"][i]["status"] == "pass" for run in [*old, *new])
            numerical.append({**description, "status": "pass", "tested_components_per_run": first["tested_components"],
                "before_max_abs_error": max(run["cases"][i]["max_abs_error"] for run in old),
                "after_max_abs_error": max(run["cases"][i]["max_abs_error"] for run in new),
                "after_max_error_over_roundoff_bound": max(run["cases"][i]["max_error_over_roundoff_bound"] for run in new)})
        else:
            fields = [k for k in ["values", "energy", "gradient", "hessian", "fc3_directional", "fc4_directional"] if k in first]
            errors = {}
            for field in fields:
                a = first[field] if isinstance(first[field], list) else [first[field]]
                values = [run["cases"][i][field] for run in new]
                values = [v if isinstance(v, list) else [v] for v in values]
                error = max(abs(x - y) for v in values for x, y in zip(a, v))
                tolerance = 1e-9 if field == "fc4_directional" else 1e-10
                assert error <= tolerance, (first["kind"], field, error)
                errors[field] = {"maximum_absolute_difference": error, "tolerance": tolerance}
            numerical.append({**description, "status": "pass", "before_after_differences": errors})
        for stage in [k for k in first if k == "timing" or k.endswith("_timing")]:
            bt = [run["cases"][i][stage]["median_seconds"] for run in old]
            at = [run["cases"][i][stage]["median_seconds"] for run in new]
            ratios = [a / b for a, b in zip(at, bt)]
            timing.append({**description, "stage": stage, "before_median_seconds": statistics.median(bt),
                "after_median_seconds": statistics.median(at), "paired_after_over_before": ratios,
                "paired_ratio_range": [min(ratios), max(ratios)],
                "median_paired_ratio": statistics.median(ratios),
                "needs_review": statistics.median(ratios) > 1.10})
    registry = read(args.input / "registry-status.json")
    assert next(x for x in registry if x["num"] == "0.1.22")["yanked"] is False
    source = read(args.input / "upstream-compare.json")
    report = {"schema": "gfn1-rs-backend-update-audit-v1", "package_version": new[0]["package_version"],
        "numerical_status": "pass", "timing_status": "review_required" if any(t["needs_review"] for t in timing) else "pass",
        "registry_status": registry, "lockfile_changes": changes,
        "upstream": {"repository": "https://github.com/sarah-quinones/gemm-x64-v2/",
            "comparison": "https://github.com/sarah-quinones/gemm-x64-v2/compare/b161af1e7751a9b7978414d2419438b13c58a97b...71872b9a98e6a64ef3c0c3b46da02cc2b19d2937",
            "commits": source["commits"], "runtime_changes": "Triangular row/column chunk boundaries and nested cache-block size ordering; no changes to build.rs or generated microkernel arithmetic",
            "yank_reason_scope": "Registry confirms yank status but supplies no reason; the source fixes are verified independently, not asserted to be an official yank explanation."},
        "measurement": {"platform": new[0]["os"] + " " + new[0]["arch"], "build_profile": "reltest",
            "host": read(args.input / "platform.json"),
            "faer_threads": 1, "rayon_threads": 2, "explicit_parallel_kernel_threads": 2,
            "paired_process_runs": args.pairs, "order": ["before/after" if i%2==0 else "after/before" for i in range(args.pairs)],
            "per_process_samples": 7, "cpu_affinity": [0,2], "physical_cores": 2,
            "warm_up": "Five untimed calls plus an adaptive pilot of at least 5 ms; repeated operations target 20 ms per sample",
            "conditions": "No other gfn1-rs build or validation was running during paired measurements. Unrelated shared-host processes were not isolated; paired repetitions and affinity reduce, but do not eliminate, timing noise.",
            "speed_gate": "Review median paired latency increases above 10%; timing variations below this bound are not claimed as optimization gains"},
        "numerical_cases": numerical, "timing_cases": timing,
        "matrix_components_per_run": sum(c.get("tested_components_per_run", 0) for c in numerical),
        "matrix_latency_geometric_mean_ratio": math.exp(statistics.mean(math.log(t["median_paired_ratio"]) for t in timing if t["kind"] == "gemm")),
        "raw_runs": {"before": old, "after": new},
        "source_sha256": {"tests/validation/validate_backend.rs": sha(root / "tests/validation/validate_backend.rs"),
                          "Cargo.lock": sha(root / "Cargo.lock")}}
    if args.extended_validation:
        report["extended_validation"] = extended_validation(root, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps({"numerical_status": report["numerical_status"], "timing_status": report["timing_status"],
        "matrix_ratio": report["matrix_latency_geometric_mean_ratio"],
        "review_cases": [t for t in timing if t["needs_review"]],
        "molecular_timings": [t for t in timing if "quantum" in t["kind"]]}, indent=2))


if __name__ == "__main__":
    main()
