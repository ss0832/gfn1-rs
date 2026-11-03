#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual matched 300 K S1 optimizations using native Rust analytic gradients."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixCartesian
from ase.optimize import BFGSLineSearch
from ase.units import Bohr, Hartree
from gfn1_rs import Gfn1NativeCalculator, __version__


class ExcitedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, window):
        super().__init__()
        self.model, self.window = model, window
        self.native = Gfn1NativeCalculator(charge=0, electronic_temperature=300.,
                                          energy_tolerance=1e-13, charge_tolerance=1e-11,
                                          max_scc=1000)
        self.evaluations = 0
        self.last = None

    def evaluate(self, numbers, positions):
        if self.model == "td_gfn1":
            result = self.native.tda_gradient(numbers, positions, state=0, n_states=5,
                                               spin="singlet", method="analytic")
            spectrum = self.native.tda(numbers, positions, n_states=5,
                                        spin="singlet", transition_dipole="ao_length")
        else:
            result = self.native.stda_xtb_gradient(numbers, positions, state=0, n_states=5,
                                                   energy_window_ev=self.window)
            spectrum = self.native.stda_xtb(numbers, positions, n_states=5,
                                            energy_window_ev=self.window)
        roots = spectrum["ci_eigenvalues_hartree"] if self.model == "stda_xtb" else spectrum["excitation_energies_hartree"]
        if abs(roots[0] - result["excitation_energy_hartree"]) > 1e-9:
            raise ValueError("Gradient and spectrum selected different roots")
        if roots[1]-roots[0] < 1e-7:
            raise ValueError("S1/S2 degeneracy: no unique individual adiabatic gradient")
        self.evaluations += 1
        return result, spectrum

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        result, spectrum = self.evaluate(self.atoms.numbers.tolist(), self.atoms.positions.tolist())
        self.results = {"energy": result["total_energy_hartree"]*Hartree,
                        "forces": np.asarray(result["forces"])*Hartree/Bohr}
        self.last = (result, spectrum)


def geometry(positions):
    r = np.asarray(positions)
    return {"cc_angstrom": float(np.linalg.norm(r[1]-r[0])),
            "ch_angstrom": [float(min(np.linalg.norm(h-c) for c in r[:2])) for h in r[2:]],
            "positions_angstrom": r.tolist()}


def write_xyz(path, numbers, frames):
    symbols = Atoms(numbers=numbers).get_chemical_symbols()
    text = ""
    for frame in frames:
        text += f"{len(numbers)}\nS1 E_eV={frame['energy_ev']:.12f} fmax_eV_A={frame['fmax_ev_per_angstrom']:.9e}\n"
        text += "".join(f"{s} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}\n"
                        for s, r in zip(symbols, frame["positions_angstrom"]))
    path.write_text(text, encoding="utf-8")


def domain_bfgs(atoms, capture, fmax, steps):
    """Backtrack rejected domain trials; never invent forces outside the model."""
    inverse = np.eye(3*len(atoms))*.04
    capture()
    for _ in range(steps):
        positions = atoms.positions.copy()
        energy = atoms.get_potential_energy()
        gradient = -atoms.get_forces().ravel()
        if np.linalg.norm(gradient.reshape(-1,3),axis=1).max() < fmax:
            return "converged", None
        direction = -inverse @ gradient
        if np.dot(direction,gradient) >= 0:
            inverse = np.eye(gradient.size)*.04
            direction = -inverse @ gradient
        largest = np.linalg.norm(direction.reshape(-1,3),axis=1).max()
        if largest > .04:
            direction *= .04/largest
        slope = float(np.dot(gradient,direction))
        alpha = 1.
        error = None
        accepted = False
        for _ in range(28):
            atoms.positions[:] = positions+alpha*direction.reshape(-1,3)
            try:
                trial = atoms.get_potential_energy()
                if trial <= energy+1e-4*alpha*slope:
                    new_gradient = -atoms.get_forces().ravel()
                    accepted = True
                    break
            except (ValueError, RuntimeError) as exception:
                error = str(exception)
            alpha *= .5
        if not accepted:
            atoms.positions[:] = positions
            atoms.get_potential_energy()
            return ("domain_boundary" if error else "line_search_stalled"), error
        displacement = (atoms.positions-positions).ravel()
        change = new_gradient-gradient
        curvature = float(np.dot(displacement,change))
        if curvature > 1e-12*np.linalg.norm(displacement)*np.linalg.norm(change):
            rho = 1./curvature
            update = np.eye(gradient.size)-rho*np.outer(displacement,change)
            inverse = update @ inverse @ update.T+rho*np.outer(displacement,displacement)
        capture()
        if np.linalg.norm(displacement.reshape(-1,3),axis=1).max() < 1e-9:
            return ("domain_boundary" if error else "line_search_stalled"), error
    return "step_limit", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/validation/v070/ethylene-optimization"))
    parser.add_argument("--fmax", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-ev", type=float, default=20.)
    parser.add_argument("--model", choices=["td_gfn1", "stda_xtb", "both"], default="both")
    parser.add_argument("--planar", action="store_true", help="Additional fixed-z planar stationary-point experiment")
    parser.add_argument("--seed-file", type=Path, help="Explicit numbers/positions_angstrom JSON; no added perturbation")
    parser.add_argument("--optimizer", choices=["ase", "domain_bfgs"], default="ase")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root/"tests/data/tda-stda-native-comparison.json").read_text())
    case = next(c for c in source["cases"] if c["name"] == "ethylene")
    positions = np.asarray(case["positions_angstrom"], dtype=float)
    axis = positions[1]-positions[0]
    axis /= np.linalg.norm(axis)
    angle = 0. if args.planar else np.deg2rad(5.)
    for i in range(2, len(positions)):
        if np.linalg.norm(positions[i]-positions[1]) < np.linalg.norm(positions[i]-positions[0]):
            v = positions[i]-positions[1]
            positions[i] = positions[1]+v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+axis*np.dot(axis,v)*(1-np.cos(angle))
    jitter = np.random.default_rng(173).normal(scale=.003, size=positions.shape)
    if args.planar:
        positions[:, 2] = 0.
        jitter[:, 2] = 0.
    positions += jitter-jitter.mean(axis=0)
    if args.seed_file:
        if args.planar:
            parser.error("--seed-file and --planar are separate experiments")
        seed = json.loads(args.seed_file.read_text())
        case = {"numbers":seed["numbers"]}
        positions = np.asarray(seed["positions_angstrom"], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    models = ["td_gfn1", "stda_xtb"] if args.model == "both" else [args.model]
    for model in models:
        start = time.perf_counter()
        atoms = Atoms(numbers=case["numbers"], positions=positions.copy())
        if args.planar:
            atoms.set_constraint(FixCartesian(range(len(atoms)), mask=(False,False,True)))
        calculator = ExcitedCalculator(model, args.window_ev)
        atoms.calc = calculator
        frames = []
        report = {"schema": "gfn1-rs-ethylene-s1-optimization-v1", "package_version": __version__,
                  "model": model, "electronic_temperature_kelvin": 300., "charge": 0,
                  "spin": "singlet", "state": 0, "state_tracking": "energy-ordered adiabatic S1, not a fixed diabatic character",
                  "potential": "GFN1 free energy + dedicated sTDA-xTB excitation" if model == "stda_xtb" else "GFN1 free energy + legacy TD-GFN1 TDA excitation",
                  "seed": "comparison ethylene; NumPy RNG seed 173 normal 0.003 angstrom perturbation, mean translation removed; planar xy-only" if args.planar else "comparison ethylene; 5 degree C-C twist plus NumPy RNG seed 173 normal 0.003 angstrom perturbation, mean translation removed",
                  "constraints": "all z coordinates fixed to zero" if args.planar else "none",
                  "numbers": case["numbers"], "initial_positions_angstrom": positions.tolist(),
                  "optimizer": "ASE BFGSLineSearch, alpha=10, maxstep=0.06 angstrom", "fmax_ev_per_angstrom": args.fmax,
                  "max_steps": args.steps, "stda_energy_window_ev": args.window_ev,
                  "status": "running", "accepted_frames": frames}
        if args.seed_file:
            report["seed"] = seed["provenance"]
            report["seed_metadata"] = seed
        if args.optimizer == "domain_bfgs":
            report["optimizer"] = "inverse BFGS with Armijo backtracking, rejected-domain step halving, maximum atomic step 0.04 angstrom"
        def save():
            report["elapsed_seconds"] = time.perf_counter()-start
            report["gradient_evaluations"] = calculator.evaluations
            (args.output/f"{model}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            write_xyz(args.output/f"{model}-trajectory.xyz", case["numbers"], frames)
            if frames:
                write_xyz(args.output/f"{model}-final.xyz", case["numbers"], frames[-1:])
        def capture():
            forces = atoms.get_forces()
            result, spectrum = calculator.last
            roots = spectrum["ci_eigenvalues_hartree"] if model == "stda_xtb" else spectrum["excitation_energies_hartree"]
            frame = {"step": len(frames), "energy_ev": atoms.get_potential_energy(),
                     "fmax_ev_per_angstrom": float(np.linalg.norm(forces,axis=1).max()),
                     "raw_fmax_ev_per_angstrom": float(np.linalg.norm(np.asarray(result["forces"])*Hartree/Bohr,axis=1).max()),
                     "gradient_hartree_per_bohr": result["gradient"],
                     "excitation_ev": roots[0]*Hartree, "s1_s2_gap_ev": (roots[1]-roots[0])*Hartree,
                     "oscillator_strength_ao_length": spectrum["oscillator_strengths"][0] if roots[0] > 0. else None,
                     "nonpositive_ci_roots": spectrum.get("nonpositive_ci_roots", 0),
                     "configuration_count": len(spectrum["configurations"]) if model == "stda_xtb" else None,
                     **geometry(atoms.positions)}
            frames.append(frame)
            save()
            print(model, frame["step"], frame["energy_ev"], frame["fmax_ev_per_angstrom"], frame["cc_angstrom"], flush=True)
        try:
            if args.optimizer == "domain_bfgs":
                report["status"], error = domain_bfgs(atoms,capture,args.fmax,args.steps)
                if error:
                    report["error"] = error
            else:
                optimizer = BFGSLineSearch(atoms, maxstep=.06, alpha=10., logfile=str(args.output/f"{model}.log"))
                optimizer.attach(capture, interval=1)
                converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
                report["status"] = "converged" if converged else "step_limit"
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            report["status"] = "calculation_error"
            report["error"] = str(error)
            report["failed_trial_positions_angstrom"] = atoms.positions.tolist()
        save()
        print(model, report["status"], report.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
