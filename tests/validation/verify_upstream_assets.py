#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recheck inherited asset bytes against an exact upstream Git commit.

Read-only for distributed assets. Output contains repository-relative paths,
source URLs and hashes, never local machine paths or environment details.
LF normalization is explicit; a matching source file does not by itself certify
the numerical Rust transcription or original auxiliary-basis reconstruction.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPOSITORIES = {"xtb": "grimme-lab/xtb", "tblite": "tblite/tblite",
                "dftd4": "dftd4/dftd4", "simple-dftd3": "dftd3/simple-dftd3"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "gfn1-rs asset provenance audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(REPOSITORIES), required=True)
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "third_party/assets.json").read_text(encoding="utf-8"))
    source = manifest["sources"][args.source]
    revision = args.commit or source.get("commit")
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        parser.error("an exact 40-hex-digit source commit is required")
    records = []
    for entry in manifest["files"]:
        if entry["source"] != args.source:
            continue
        relative = entry["path"]
        upstream = relative.removeprefix(f"third_party/{args.source}/")
        if args.source == "tblite":
            upstream = "src/tblite/data/spin.f90"
        url = f"https://raw.githubusercontent.com/{REPOSITORIES[args.source]}/{revision}/{upstream}"
        original = fetch(url)
        normalized = original.replace(b"\r\n", b"\n")
        local = (root / relative).read_bytes()
        equal = local == normalized
        records.append({"path": relative, "upstream_path": upstream, "url": url,
                        "source_sha256": digest(original), "source_sha256_lf": digest(normalized),
                        "distributed_sha256": digest(local), "byte_comparison": "upstream CRLF to LF only",
                        "status": "pass" if equal else "fail"})
    licenses = []
    for name in ["COPYING", "COPYING.LESSER"]:
        path = root / "third_party" / args.source / name
        if not path.is_file() and args.source == "tblite":
            path = root / "third_party" / "xtb" / name
        if not path.is_file():
            continue
        url = f"https://raw.githubusercontent.com/{REPOSITORIES[args.source]}/{revision}/{name}"
        original = fetch(url).replace(b"\r\n", b"\n")
        equal = path.read_bytes().replace(b"\r\n", b"\n") == original
        licenses.append({"path": path.relative_to(root).as_posix(), "shared_standard_text": path.parent.name != args.source, "upstream_url": url,
                         "source_sha256_lf": digest(original), "status": "pass" if equal else "fail"})
    passed = bool(records) and bool(licenses) and all(r["status"] == "pass" for r in records + licenses)
    report = {"schema": "gfn1-rs-upstream-byte-review-v1", "source": args.source, "commit": revision,
              "spdx": source["spdx"], "status": "pass" if passed else "fail", "files": records, "license_files": licenses,
              "scope": "Original upstream source bytes and retained license texts; separate mathematical/transcription tests remain necessary"}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
