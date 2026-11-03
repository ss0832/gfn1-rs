#!/usr/bin/env python3
"""Audit exact distributed asset bytes. JSON to stdout; no asset modifications.

--release additionally requires completed source adoption. A passing checksum
alone does not close the P0 original-source/licensing review.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Show failures, untracked assets and counts rather than all passing records")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "third_party/assets.json").read_text(encoding="utf-8"))
    records = []
    expected = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in expected:
            raise ValueError(f"unsafe or duplicate manifest path: {relative}")
        expected.add(relative)
        source = manifest["sources"][entry["source"]]
        content = path.read_bytes() if path.is_file() else None
        actual = hashlib.sha256(content).hexdigest() if content is not None else None
        records.append({**entry, "actual_sha256": actual,
                        "size_bytes": len(content) if content is not None else None,
                        "status": "pass" if actual == entry["sha256"] else "fail",
                        "adoption_status": source["adoption_status"]})
    discovered = set()
    for pattern in ("third_party/**/*.f90", "third_party/**/*.inc",
                    "third_party/**/parameters.toml", "third_party/xtb/param*.txt",
                    "third_party/xtb/include/*.fh", "models/*.model", "models/*.json", "tests/data/*.json",
                    "third_party/cod/LICENSE.txt", "src/secondary_bases/*.txt"):
        discovered.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    untracked = sorted(discovered - expected)
    untracked_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in untracked}
    pending = sorted(key for key, src in manifest["sources"].items()
                     if src["adoption_status"] != "verified")
    archive_ok = None
    if args.archive:
        archive_ok = (hashlib.sha256(args.archive.read_bytes()).hexdigest()
                      == manifest.get("supplied_base_archive_sha256", manifest["baseline_archive_sha256"]))
    ok = (bool(records) and all(r["status"] == "pass" for r in records)
          and not untracked and archive_ok is not False and not (args.release and pending))
    print(json.dumps({"status": "pass" if ok else "fail", "files": [r for r in records if r["status"] != "pass"] if args.summary else records,
                      "verified_file_count": sum(r["status"] == "pass" for r in records), "manifest_file_count": len(records),
                      "untracked_assets": untracked, "untracked_asset_sha256": untracked_hashes,
                      "pending_sources": pending,
                      "baseline_archive_verified": archive_ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
