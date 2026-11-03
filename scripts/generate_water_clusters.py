#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate independent PBE0/def2-SVP water-cluster training references.

Geometries are constructed here, without copied benchmark coordinates. The
finite-basis, uncorrected supermolecular interaction energies are demonstration
targets, not counterpoise-corrected or complete-basis binding benchmarks. Full
electronic energies are retained as metadata; fitting uses explicit distortion
or interaction energies to eliminate element-dependent energy origins. The
tetramers are entirely held out from training. Requires PySCF, NumPy and SciPy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib


def monomer(bond=0.958, angle=104.52, second_bond=None):
    theta = np.deg2rad(angle)
    return np.array([[0., 0., 0.], [bond, 0., 0.],
                     [(bond if second_bond is None else second_bond) * np.cos(theta),
                      (bond if second_bond is None else second_bond) * np.sin(theta), 0.]])


def cluster(size, separation, tilt=0.):
    if size == 2:
        donor = monomer() + [-separation / 2, 0., 0.]
        theta = np.deg2rad(104.52 / 2)
        acceptor = np.array([[0., 0., 0.], [np.cos(theta), np.sin(theta), 0.],
                             [np.cos(theta), -np.sin(theta), 0.]])
        acceptor *= 0.958
        phi = np.deg2rad(tilt)
        rotate = np.array([[np.cos(phi), 0., np.sin(phi)], [0., 1., 0.],
                           [-np.sin(phi), 0., np.cos(phi)]])
        return np.vstack([donor, acceptor @ rotate.T + [separation / 2, 0., 0.]])
    radius = separation / (2 * np.sin(np.pi / size))
    points = np.array([[radius * np.cos(2 * np.pi * i / size),
                        radius * np.sin(2 * np.pi * i / size), 0.] for i in range(size)])
    atoms = []
    for i, oxygen in enumerate(points):
        direction = points[(i + 1) % size] - oxygen
        direction /= np.linalg.norm(direction)
        # One hydrogen donates along the ring; the free hydrogen points out
        # of its plane. Tilting the molecular plane preserves internal bonds.
        normal = np.array([0., 0., (-1.) ** i])
        radial = oxygen / np.linalg.norm(oxygen)
        phi = np.deg2rad(tilt)
        transverse = np.cos(phi) * normal + np.sin(phi) * radial
        transverse -= np.dot(transverse, direction) * direction
        transverse /= np.linalg.norm(transverse)
        theta = np.deg2rad(104.52)
        atoms.extend([oxygen, oxygen + 0.958 * direction,
                      oxygen + 0.958 * (np.cos(theta) * direction + np.sin(theta) * transverse)])
    return np.array(atoms)


def geometries():
    yield "water-anchor", "reference", 1, monomer()
    for split, points in {
        "train": [(0.90, 104.52, 0.958), (0.94, 99., 0.98), (0.98, 108., 0.94),
                  (1.02, 104.52, 1.02), (0.958, 112., 0.958), (0.958, 101., 0.958)],
        "validation": [(0.925, 106., 0.975), (0.995, 102., 0.958)],
        "extrapolation": [(1.08, 115., 0.958)],
    }.items():
        for i, (bond, angle, second) in enumerate(points):
            yield f"monomer-{split}-{i}", split, 1, monomer(bond, angle, second)
    for size in [2, 3, 4]:
        grid = {
            "train": [] if size == 4 else [(2.65, -12.), (2.85, 0.), (3.05, 12.), (3.35, 0.)],
            "validation": [(2.75, 7.), (3.20, -6.)],
            "extrapolation": [(2.50, 18.), (3.70, -15.)] if size == 4 else [(3.70, -15.)],
        }
        for split, points in grid.items():
            for i, (distance, tilt) in enumerate(points):
                yield f"cluster-{size}-{split}-{i}", split, size, cluster(size, distance, tilt)


def reference(coordinates, basis, grid_level):
    symbols = ["O", "H", "H"] * (len(coordinates) // 3)
    molecule = gto.M(atom=list(zip(symbols, coordinates)), unit="Angstrom", basis=basis,
                     charge=0, spin=0, symmetry=False, verbose=0)
    solver = dft.RKS(molecule)
    solver.xc = "PBE0"
    solver.grids.level = grid_level
    solver.conv_tol = 1e-11
    solver.conv_tol_grad = 1e-7
    solver.max_cycle = 150
    solver.max_memory = 1500
    solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent water-cluster RKS did not converge")
    derivative = solver.nuc_grad_method()
    derivative.grid_response = True
    gradient = derivative.kernel()
    dipole = solver.dip_moment(unit="AU", verbose=0)
    if not np.isfinite(gradient).all() or not np.isfinite(dipole).all():
        raise RuntimeError("non-finite independent water-cluster labels")
    if np.max(np.abs(gradient.sum(axis=0))) > 1e-7:
        raise RuntimeError("independent gradient violates translational invariance")
    return float(solver.e_tot), gradient.tolist(), dipole.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--basis", default="def2-SVP")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0 or args.grid_level < 0:
        parser.error("positive thread count and nonnegative grid level required")
    lib.num_threads(args.threads)
    samples, relative = [], []
    anchor_energy = None
    for name, split, size, coordinates in geometries():
        coordinates = np.round(coordinates - coordinates.mean(axis=0), 12)
        energy, gradient, dipole = reference(coordinates, args.basis, args.grid_level)
        if anchor_energy is None:
            anchor_energy = energy
        symbols = ["O", "H", "H"] * size
        xyz = f"{3 * size}\nindependently constructed water cluster\n" + "".join(
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}\n" for symbol, (x, y, z) in zip(symbols, coordinates))
        samples.append({"id": name, "split": split, "group": f"water-{size}", "xyz": xyz,
                        "charge": 0, "water_molecules": size,
                        "reference_total_energy_hartree": energy,
                        "gradient_hartree_per_bohr": gradient, "dipole_e_bohr": dipole})
        if split != "reference":
            relative.append({"id": name, "split": split, "group": "distortion" if size == 1 else "interaction",
                             "energy_hartree": energy - size * anchor_energy,
                             "terms": [{"sample": name, "coefficient": 1.},
                                       {"sample": "water-anchor", "coefficient": -float(size)}]})
        print(f"{name} waters={size} E={energy:.14f}", flush=True)
    data = {
        "schema": "gfn1-rs-observable-training-v2", "energy_units": "hartree", "coordinate_units": "angstrom",
        "energy_definition": "mermin_free", "electronic_temperature_K": 0., "dipole_origin_bohr": [0., 0., 0.],
        "provenance": "Independently generated by scripts/generate_water_clusters.py; constructed neutral monomers, dimers, cyclic trimers and tetramers; analytic hybrid-DFT gradients including grid response; no xTB or pretrained targets",
        "reference_method": f"PBE0/{args.basis}", "independent_reference": True, "license_spdx": "CC0-1.0",
        "reference_citations": ["10.1063/5.0006074", "10.1063/1.478522", "10.1039/B508541A"],
        "target_convention": "Monomer distortions or finite-basis uncorrected supermolecular interactions relative to the rigid isolated water anchor; no counterpoise or dispersion correction; absolute all-electron energies are metadata only",
        "split_policy": "Deterministic disjoint constructed geometries; all tetramers held out; shared monomer anchor has no direct training labels",
        "generator_versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ["pyscf", "numpy", "scipy"]}},
        "numerical_settings": {"rks_energy_tolerance_hartree": 1e-11, "rks_gradient_tolerance": 1e-7,
                               "dft_grid_level": args.grid_level, "analytic_grid_response": True, "threads": args.threads},
        "samples": samples, "relative_energies": relative,
    }
    encoded = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"dataset_sha256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
