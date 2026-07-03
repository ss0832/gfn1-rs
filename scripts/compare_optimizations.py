# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare ASE geometry optimizations from gfn1-rs-python and tblite CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import LBFGS
from ase.units import Bohr, Hartree

from gfn1_rs.ase import GFN1RSCalculator


@dataclass(frozen=True)
class Molecule:
    name: str
    xyz: str


MOLECULES = {
    "hydrogen": Molecule(
        "hydrogen",
        "2\nhydrogen\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n",
    ),
    "hydrogen-fluoride": Molecule(
        "hydrogen-fluoride",
        "2\nhydrogen fluoride\nH 0.000000 0.000000 0.000000\nF 0.917000 0.000000 0.000000\n",
    ),
    "water": Molecule(
        "water",
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
    ),
    "phosphine": Molecule(
        "phosphine",
        "4\nphosphine\nP 0.000000 0.000000 0.000000\nH 1.193000 0.000000 0.768000\nH -0.596500 1.033300 0.768000\nH -0.596500 -1.033300 0.768000\n",
    ),
    "ferrocene": Molecule(
        "ferrocene",
        "21\nferrocene staggered test geometry\nFe 0.000000 0.000000 0.000000\nC 1.430000 0.000000 1.650000\nC 0.441908 1.360370 1.650000\nC -1.156908 0.840788 1.650000\nC -1.156908 -0.840788 1.650000\nC 0.441908 -1.360370 1.650000\nH 2.510000 0.000000 1.650000\nH 0.775615 2.386978 1.650000\nH -2.030615 1.475161 1.650000\nH -2.030615 -1.475161 1.650000\nH 0.775615 -2.386978 1.650000\nC 1.156908 0.840788 -1.650000\nC -0.441908 1.360370 -1.650000\nC -1.430000 0.000000 -1.650000\nC -0.441908 -1.360370 -1.650000\nC 1.156908 -0.840788 -1.650000\nH 2.030615 1.475161 -1.650000\nH -0.775615 2.386978 -1.650000\nH -2.510000 0.000000 -1.650000\nH -0.775615 -2.386978 -1.650000\nH 2.030615 -1.475161 -1.650000\n",
    ),
    "borane": Molecule(
        "borane",
        "4\nborane\nB 0.000000 0.000000 0.000000\nH 1.190000 0.000000 0.000000\nH -0.595000 1.030570 0.000000\nH -0.595000 -1.030570 0.000000\n",
    ),
    "caffeine": Molecule(
        "caffeine",
        "24\ncaffeine fixed test geometry\nN 0.000000 0.000000 0.000000\nC 1.250000 0.000000 0.000000\nN 2.000000 1.100000 0.000000\nC 1.250000 2.200000 0.000000\nC 0.000000 2.200000 0.000000\nC -0.700000 1.100000 0.000000\nN 1.750000 3.350000 0.000000\nC 0.750000 4.250000 0.000000\nN -0.350000 3.350000 0.000000\nO 1.900000 -1.050000 0.000000\nO -1.950000 1.100000 0.000000\nC -0.800000 -1.200000 0.250000\nH -1.830000 -0.880000 0.250000\nH -0.550000 -1.780000 1.140000\nH -0.550000 -1.820000 -0.620000\nC 3.450000 1.100000 0.250000\nH 3.800000 2.130000 0.250000\nH 3.780000 0.580000 1.150000\nH 3.850000 0.540000 -0.600000\nC 3.100000 3.900000 0.250000\nH 3.060000 4.990000 0.250000\nH 3.640000 3.580000 1.140000\nH 3.700000 3.520000 -0.580000\nH 0.780000 5.330000 0.000000\n",
    ),
}


class TbliteCliCalculator(Calculator):
    """ASE calculator using the external tblite CLI for validation only."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, tblite_bin: Path, workdir: Path, *, charge: float = 0.0) -> None:
        super().__init__()
        self.tblite_bin = tblite_bin
        self.workdir = workdir
        self.charge = charge
        self.counter = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        if any(self.atoms.get_pbc()):
            raise ValueError("TbliteCliCalculator supports non-PBC only")

        self.counter += 1
        run_dir = self.workdir / f"step_{self.counter:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        xyz = run_dir / "input.xyz"
        write(xyz, self.atoms, format="xyz")
        args = [
            str(self.tblite_bin),
            "run",
            str(xyz),
            "--method",
            "gfn1",
            "--charge",
            f"{self.charge:.16g}",
            "--no-restart",
            "--json",
            "tblite.json",
            "--grad",
            "tblite.grad",
        ]
        output = subprocess.run(
            args,
            cwd=run_dir,
            env=tblite_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        if output.returncode != 0:
            raise RuntimeError(
                f"tblite failed for {xyz}\nstdout:\n{output.stdout}\nstderr:\n{output.stderr}"
            )
        with (run_dir / "tblite.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        gradient = np.asarray(data["gradient"], dtype=float)
        self.results["energy"] = float(data["energy"]) * Hartree
        self.results["forces"] = -gradient * Hartree / Bohr


def tblite_env() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    entries = [root / ".tblite_alias", Path(r"C:\TDM-GCC-64\bin")]
    path = os.environ.get("PATH", "")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(str(p) for p in entries) + os.pathsep + path
    return env


def atoms_from_xyz(xyz: str):
    return read(StringIO(xyz), format="xyz")


def optimize(atoms, label: str, workdir: Path, fmax: float, max_steps: int):
    workdir.mkdir(parents=True, exist_ok=True)
    dyn = LBFGS(atoms, logfile=str(workdir / f"{label}.log"), trajectory=None)
    converged = bool(dyn.run(fmax=fmax, steps=max_steps))
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    write(workdir / f"{label}.xyz", atoms, format="xyz")
    return {
        "converged": converged,
        "steps": int(dyn.nsteps),
        "energy": energy,
        "fmax": float(np.abs(forces).max()),
        "positions": np.asarray(atoms.get_positions(), dtype=float),
    }


def aligned_deltas(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    ref = reference - reference.mean(axis=0)
    act = actual - actual.mean(axis=0)
    cov = act.T @ ref
    u, _s, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    diff = act @ rot - ref
    distances = np.linalg.norm(diff, axis=1)
    return float(np.sqrt(np.mean(distances * distances))), float(np.max(distances))


def run_one(name: str, args: argparse.Namespace) -> dict[str, Any]:
    molecule = MOLECULES[name]
    out_dir = args.output_dir / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rust_atoms = atoms_from_xyz(molecule.xyz)
    rust_atoms.calc = GFN1RSCalculator(
        param_path=args.param,
        d3_reference_path=args.d3_reference,
        charge=args.charge,
    )
    tblite_atoms = atoms_from_xyz(molecule.xyz)
    tblite_atoms.calc = TbliteCliCalculator(args.tblite_bin, out_dir / "tblite", charge=args.charge)

    rust = optimize(rust_atoms, "gfn1_rs", out_dir, args.fmax, args.max_steps)
    tblite = optimize(tblite_atoms, "tblite", out_dir, args.fmax, args.max_steps)
    rmsd, max_delta = aligned_deltas(tblite["positions"], rust["positions"])
    return {
        "name": name,
        "gfn1_converged": rust["converged"],
        "tblite_converged": tblite["converged"],
        "gfn1_steps": rust["steps"],
        "tblite_steps": tblite["steps"],
        "gfn1_energy_ev": rust["energy"],
        "tblite_energy_ev": tblite["energy"],
        "delta_energy_ev": rust["energy"] - tblite["energy"],
        "gfn1_fmax": rust["fmax"],
        "tblite_fmax": tblite["fmax"],
        "rmsd_angstrom": rmsd,
        "max_delta_angstrom": max_delta,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param",
        type=Path,
        default=Path(os.environ.get("GFN1_XTB_PARAM", "")),
        help="path to external param_gfn1-xtb.txt",
    )
    parser.add_argument(
        "--d3-reference",
        type=Path,
        default=Path(os.environ.get("GFN1_D3_REFERENCE", "")),
        help="path to s-dftd3 reference.f90",
    )
    parser.add_argument(
        "--tblite-bin",
        type=Path,
        default=Path(os.environ.get("GFN1_TBLITE_BIN", root / ".tblite_build_static" / "app" / "tblite.exe")),
    )
    parser.add_argument("--output-dir", type=Path, default=root / ".tblite_runs" / "opt_compare")
    parser.add_argument("--molecule", action="append", choices=sorted(MOLECULES))
    parser.add_argument("--fmax", type=float, default=0.05, help="ASE LBFGS force tolerance in eV/A")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--charge", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.param:
        raise SystemExit("Set --param or GFN1_XTB_PARAM")
    if not args.d3_reference:
        raise SystemExit("Set --d3-reference or GFN1_D3_REFERENCE")
    if not args.tblite_bin.exists():
        raise SystemExit(f"tblite binary not found: {args.tblite_bin}")

    names = args.molecule or list(MOLECULES)
    print(
        "name,gfn1_converged,tblite_converged,gfn1_steps,tblite_steps,"
        "gfn1_energy_ev,tblite_energy_ev,delta_energy_ev,gfn1_fmax,tblite_fmax,"
        "rmsd_angstrom,max_delta_angstrom"
    )
    for name in names:
        row = run_one(name, args)
        print(
            "{name},{gfn1_converged},{tblite_converged},{gfn1_steps},{tblite_steps},"
            "{gfn1_energy_ev:.12f},{tblite_energy_ev:.12f},{delta_energy_ev:.6e},"
            "{gfn1_fmax:.6e},{tblite_fmax:.6e},{rmsd_angstrom:.6e},{max_delta_angstrom:.6e}".format(
                **row
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
