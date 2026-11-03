#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent complete Cartesian energy differences along an ethylene trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gfn1_rs import Gfn1NativeCalculator, ANGSTROM_TO_BOHR, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="0,4,5,6")
    parser.add_argument("--states", default="0,1,2")
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text())
    calculator = Gfn1NativeCalculator(electronic_temperature=300., max_scc=1000,
                                     energy_tolerance=1e-13, charge_tolerance=1e-11)
    numbers = trajectory["numbers"]
    cases = []
    report = {"schema": "gfn1-rs-td-ethylene-gradient-audit-v1", "package_version": __version__,
              "electronic_temperature_kelvin": 300., "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
              "method": "analytic native Rust total excited-state gradient vs independently rediagonalized whole-energy differences",
              "cases": cases}
    for index in map(int, args.frames.split(",")):
        positions = np.asarray(trajectory["accepted_frames"][index]["positions_angstrom"])*ANGSTROM_TO_BOHR
        for state in map(int, args.states.split(",")):
            case = {"trajectory_step": index, "state": state, "positions_bohr": positions.tolist()}
            cases.append(case)
            try:
                result = calculator.tda_gradient(numbers, positions.tolist(), unit="bohr", state=state,
                                                 n_states=5, method="analytic")
                analytic = np.asarray(result["gradient"]).ravel()
                differences = []
                for h in (1e-3, 5e-4):
                    fd = []
                    for coordinate in range(positions.size):
                        def energy(shift):
                            moved = positions.copy()
                            moved.ravel()[coordinate] += shift
                            ground = calculator.calculate(numbers, moved.tolist(), unit="bohr")
                            if not ground.converged:
                                raise ValueError("Displaced ground-state SCC did not converge")
                            spectrum = calculator.tda(numbers, moved.tolist(), unit="bohr", n_states=5)
                            return ground.energy_hartree+spectrum["excitation_energies_hartree"][state]
                        fd.append((energy(h)-energy(-h))/(2*h))
                    differences.append(np.asarray(fd))
                richardson = (4*differences[1]-differences[0])/3
                error = float(np.max(np.abs(analytic-richardson)))
                case.update(status="pass" if error < 5e-7 else "fail", analytic=analytic.tolist(),
                            fd=[d.tolist() for d in differences], richardson=richardson.tolist(),
                            max_error_hartree_per_bohr=error, tested_components=len(analytic))
                print(index,state,case["status"],error,flush=True)
            except (ValueError, RuntimeError) as error:
                case.update(status="calculation_error", error=str(error))
                print(index,state,case["status"],str(error),flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    report["tested_components"] = sum(c.get("tested_components", 0) for c in cases)
    report["status"] = "pass" if all(c["status"] == "pass" for c in cases) else "fail_or_rejected"
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
