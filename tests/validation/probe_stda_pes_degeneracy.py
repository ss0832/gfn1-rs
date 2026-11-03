#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Locate a native final-MO degeneracy and compare directional CI limits.

The subprocess is this repository's Rust validation target, not an upstream
xTB/sTDA program. No production calculation API launches an executable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=root/"tests/data/stda-pes-degeneracy.json")
    parser.add_argument("--output", type=Path, default=root/"target/validation/stda-pes-degeneracy.json")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    base = reference["reference_geometry"]
    process = subprocess.Popen(["cargo", "test", "--release", "--test",
        "stda_pes_orbital_probe", "--", str(args.reference.resolve())],
        cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def query(positions, window=None):
        request = dict(numbers=base["numbers"], positions_angstrom=positions.tolist(), spectrum=window is not None)
        if window is not None:
            request["energy_window_ev"] = window
        process.stdin.write(json.dumps(request)+"\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("The native validation probe terminated")
        return json.loads(line)

    def effective(positions):
        value = query(positions)
        u, _, vh = np.linalg.svd(value["overlap"])
        rotation = u@vh
        h = rotation@np.diag(value["energies_hartree"])@rotation.T
        return np.array([h[0,0]-h[1,1],2*h[0,1]])

    positions = np.asarray(base["positions_angstrom"],dtype=float)
    iterations = []
    cases = []
    try:
        for iteration in range(12):
            gap = effective(positions)
            jacobian = np.zeros((2,positions.size))
            for coordinate in range(positions.size):
                plus, minus = positions.copy(), positions.copy()
                plus.ravel()[coordinate] += 1e-8
                minus.ravel()[coordinate] -= 1e-8
                jacobian[:,coordinate] = (effective(plus)-effective(minus))/2e-8
            iterations.append(dict(iteration=iteration,gap_hartree=float(np.linalg.norm(gap)),positions_angstrom=positions.tolist()))
            if np.linalg.norm(gap)<2e-12:
                break
            step = -np.linalg.pinv(jacobian,rcond=1e-8)@gap
            maximum = np.linalg.norm(step.reshape(-1,3),axis=1).max()
            positions += (step*min(1.,1e-5/max(maximum,1e-30))).reshape(-1,3)
        _, _, directions = np.linalg.svd(jacobian,full_matrices=False)
        for window in (20.,200.):
            for radius in (1e-4,1e-5,1e-6,1e-7,1e-8):
                for label, direction in zip(("a","b"),directions):
                    for sign in (1,-1):
                        value = query(positions+(sign*radius*direction).reshape(-1,3),window)
                        cases.append(dict(window_ev=window,radius_angstrom=radius,direction=label,sign=sign,**value))
        report = dict(schema="gfn1-rs-native-pes-limit-probe-v1",temperature_kelvin=300.,
            reference_geometry=base,seam_positions_angstrom=positions.tolist(),iterations=iterations,
            directions=directions.tolist(),cases=cases,input_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    finally:
        active_exception = sys.exc_info()[0] is not None
        process.stdin.close()
        code = process.wait()
        if code and not active_exception:
            raise RuntimeError(f"Native probe exited with status {code}")
    print(f"Wrote {len(cases)} directional-limit measurements to {args.output}")


if __name__=="__main__":main()
