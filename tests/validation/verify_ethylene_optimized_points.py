#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify all energy-gradient components and classify constrained stationary points."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.units import Bohr
from ethylene_excited_optimization import ExcitedCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["both","td_gfn1","stda_xtb"], default="both")
    args = parser.parse_args()
    cases = []
    for model in (("td_gfn1", "stda_xtb") if args.model == "both" else (args.model,)):
        run = json.loads((args.directory/f"{model}.json").read_text())
        if run["status"] != "converged":
            raise ValueError(f"Cannot classify unconverged {model}")
        positions = np.asarray(run["accepted_frames"][-1]["positions_angstrom"])/Bohr
        numbers = run["numbers"]
        calculator = ExcitedCalculator(model, run["stda_energy_window_ev"])
        result, spectrum = calculator.evaluate(numbers, (positions*Bohr).tolist())
        gradient = np.asarray(result["gradient"]).ravel()
        fd, hessians = [], []
        def evaluate(pos):
            return calculator.evaluate(numbers, (pos*Bohr).tolist())[0]
        for h in (1e-3, 5e-4):
            derivatives = np.zeros(positions.size)
            hessian = np.zeros((positions.size,positions.size))
            for coordinate in range(positions.size):
                plus, minus = positions.copy(), positions.copy()
                plus.ravel()[coordinate] += h
                minus.ravel()[coordinate] -= h
                p, m = evaluate(plus), evaluate(minus)
                derivatives[coordinate] = (p["total_energy_hartree"]-m["total_energy_hartree"])/(2*h)
                hessian[:,coordinate] = (np.asarray(p["gradient"]).ravel()-np.asarray(m["gradient"]).ravel())/(2*h)
            fd.append(derivatives)
            hessians.append(hessian)
        richardson = (4*fd[1]-fd[0])/3
        hessian = (4*hessians[1]-hessians[0])/3
        centered = positions-positions.mean(axis=0)
        rigid = []
        for unit in np.eye(3):
            rigid.append(np.tile(unit,(len(numbers),1)).ravel())
            rigid.append(np.cross(np.tile(unit,(len(numbers),1)),centered).ravel())
        _, singular, vh = np.linalg.svd(np.asarray(rigid),full_matrices=True)
        rank = int(np.count_nonzero(singular > singular[0]*1e-10))
        internal = vh[rank:].T
        eigenvalues = np.linalg.eigvalsh(internal.T @ ((hessian+hessian.T)/2) @ internal)
        error = float(np.max(np.abs(gradient-richardson)))
        case = {"model": model, "status": "pass" if error < 5e-7 else "fail",
                "gradient_hartree_per_bohr": gradient.tolist(), "fd_steps_bohr": [1e-3,5e-4],
                "energy_fd_gradients_hartree_per_bohr": [v.tolist() for v in fd],
                "richardson_gradient_hartree_per_bohr": richardson.tolist(),
                "max_gradient_error_hartree_per_bohr": error, "tested_gradient_components": positions.size,
                "raw_hessian_asymmetry_hartree_per_bohr2": float(np.max(np.abs(hessian-hessian.T))),
                "fd_hessians_hartree_per_bohr2": [h.tolist() for h in hessians],
                "richardson_hessian_hartree_per_bohr2": hessian.tolist(),
                "internal_hessian_eigenvalues_hartree_per_bohr2": eigenvalues.tolist(),
                "classification": "saddle of the unconstrained surface" if eigenvalues[0] < -1e-5 else "no negative internal curvature found at this finite-difference resolution",
                "constraints_during_optimization": run["constraints"], "rigid_rank": rank}
        cases.append(case)
        print(model, case["status"], error, eigenvalues[0], case["classification"], flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({"schema":"gfn1-rs-ethylene-stationary-point-audit-v1", "package_version":run["package_version"],
                                          "hessian_method":"independent central differences of analytic gradients, two-step Richardson; unconstrained Cartesian perturbations", "cases":cases}, indent=2)+"\n")
    if any(c["status"] != "pass" for c in cases):
        raise ValueError("Optimized-point gradient audit failed")


if __name__ == "__main__":
    main()
