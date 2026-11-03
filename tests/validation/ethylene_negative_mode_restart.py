#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Displace a stationary point along both signs of its most negative mode."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from ase.units import Bohr, Hartree
from ethylene_excited_optimization import ExcitedCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", "--planar", dest="reference", type=Path,
                        default=Path("target/validation/v070/ethylene-planar-final"),
                        help="Reference stationary-point directory; --planar is a legacy alias, not a constraint")
    parser.add_argument("--audit", type=Path, default=Path("target/validation/v070/ethylene-planar-audit.json"))
    parser.add_argument("--output", type=Path, default=Path("target/validation/v070/ethylene-negative-modes"))
    parser.add_argument("--displacement", type=float, default=.05, help="Maximum atomic displacement, angstrom")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    audit = json.loads(args.audit.read_text())
    report = {"schema":"gfn1-rs-ethylene-negative-mode-restart-v1", "package_version":audit["package_version"],
              "protocol":"Both signs of each model's lowest internal Cartesian Hessian eigenvector; no geometric constraints; same 300 K and analytic forces; Armijo backtracking shrinks invalid trials",
              "reference_audit_sha256":hashlib.sha256(args.audit.read_bytes()).hexdigest(),"runs":[]}
    for point in audit["cases"]:
        model = point["model"]
        run = json.loads((args.reference/f"{model}.json").read_text())
        positions = np.asarray(run["accepted_frames"][-1]["positions_angstrom"])
        centered = positions/Bohr-positions.mean(axis=0)/Bohr
        rigid = []
        for unit in np.eye(3):
            rigid.append(np.tile(unit,(len(positions),1)).ravel())
            rigid.append(np.cross(np.tile(unit,(len(positions),1)),centered).ravel())
        _, s, vh = np.linalg.svd(np.asarray(rigid),full_matrices=True)
        rank = np.count_nonzero(s>s[0]*1e-10)
        internal = vh[rank:].T
        hessian = np.asarray(point["richardson_hessian_hartree_per_bohr2"])
        eigenvalues, vectors = np.linalg.eigh(internal.T @ ((hessian+hessian.T)/2) @ internal)
        if eigenvalues[0] >= -1e-5:
            raise ValueError("No reliably negative mode to follow")
        direction = (internal @ vectors[:,0]).reshape(-1,3)
        direction /= np.linalg.norm(direction,axis=1).max()
        calculator = ExcitedCalculator(model,run["stda_energy_window_ev"])
        base,_ = calculator.evaluate(run["numbers"],positions.tolist())
        for sign,label in ((1,"positive"),(-1,"negative")):
            moved = positions+sign*args.displacement*direction
            seed_result,_ = calculator.evaluate(run["numbers"],moved.tolist())
            delta = (seed_result["total_energy_hartree"]-base["total_energy_hartree"])*Hartree
            if delta >= 0:
                raise ValueError(f"{model} {label} displacement did not lower energy")
            seed = {"numbers":run["numbers"],"positions_angstrom":moved.tolist(),
                    "provenance":f"{model} reference stationary point displaced {label} along its most negative internal mode",
                    "curvature_hartree_per_bohr2":float(eigenvalues[0]),"sign":sign,
                    "direction_max_atomic_norm_one":direction.tolist(),
                    "maximum_displacement_angstrom":args.displacement,"seed_minus_reference_energy_ev":delta}
            seed_path = args.output/f"{model}-{label}-seed.json"
            seed_path.write_text(json.dumps(seed,indent=2)+"\n")
            command = [sys.executable,str(Path(__file__).with_name("ethylene_excited_optimization.py")),
                       "--seed-file",str(seed_path),"--model",model,"--optimizer","domain_bfgs",
                       "--steps",str(args.steps),"--output",str(args.output/label)]
            print("SEED",model,label,delta,flush=True)
            subprocess.run(command,check=True)
            result = json.loads((args.output/label/f"{model}.json").read_text())
            report["runs"].append({"model":model,"direction_sign":sign,"seed":seed,"optimization":result})
            (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
