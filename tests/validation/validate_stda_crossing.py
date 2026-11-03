#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independently check an uncoupled-composite crossing, not a smooth PES minimum."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ase.units import Bohr, Hartree
from gfn1_rs import Gfn1NativeCalculator, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.result.read_text(encoding="utf-8-sig"))
    measured = measured.get("optimization", measured)
    if measured["status"] != "crossing_stationary":
        raise ValueError("The input must actually satisfy the crossing stationarity criterion")
    numbers = measured["numbers"]
    frame = measured["frames"][-1]
    positions = np.asarray(frame["positions_angstrom"]) / Bohr
    multiplier = frame["lambda"]
    calc = Gfn1NativeCalculator(electronic_temperature=300., max_scc=1000,
        energy_tolerance=1e-13, charge_tolerance=1e-11)

    def evaluate(pos):
        coordinates = (pos * Bohr).tolist()
        reference = calc.calculate(numbers, coordinates, compute_gradient=True)
        response = calc.stda_xtb_gradient(numbers, coordinates, state=0, energy_window_ev=20.)
        spectrum = calc.stda_xtb(numbers, coordinates, n_states=400, energy_window_ev=20.)
        if abs(response["excitation_energy_hartree"] - spectrum["ci_eigenvalues_hartree"][0]) > 1e-10:
            raise ValueError("The signed CI branch changed identity")
        return dict(e0=reference.energy_hartree,
            g0=np.asarray(reference.gradient_hartree_per_bohr).ravel(),
            gap=response["excitation_energy_hartree"],
            gg=np.asarray(response["excitation_gradient"]).ravel(),
            configurations=set(map(tuple, spectrum["configurations"])),
            ci_gap=spectrum["ci_eigenvalues_hartree"][1]-spectrum["ci_eigenvalues_hartree"][0])

    base = evaluate(positions)
    native = calc.stda_xtb_hessian(numbers, (positions*Bohr).tolist(), energy_window_ev=20.)
    h0, hx = np.asarray(native["ground_hessian"]), np.asarray(native["excitation_hessian"])
    hl = h0 + multiplier * hx
    gkkt = base["g0"] + multiplier * base["gg"]
    fd = []
    for step in (1e-4, 5e-5):
        e0_gradient, gap_gradient = np.zeros(positions.size), np.zeros(positions.size)
        hl_fd = np.zeros_like(hl)
        for column in range(positions.size):
            plus, minus = positions.copy(), positions.copy()
            plus.ravel()[column] += step
            minus.ravel()[column] -= step
            p, m = evaluate(plus), evaluate(minus)
            if p["configurations"] != base["configurations"] or m["configurations"] != base["configurations"]:
                raise ValueError("A discrete configuration selection boundary crosses the FD stencil")
            e0_gradient[column] = (p["e0"]-m["e0"])/(2*step)
            gap_gradient[column] = (p["gap"]-m["gap"])/(2*step)
            hl_fd[:,column] = (p["g0"]+multiplier*p["gg"]-m["g0"]-multiplier*m["gg"])/(2*step)
        fd.append(dict(step_bohr=step, ground_gradient=e0_gradient.tolist(),
            excitation_gradient=gap_gradient.tolist(), lagrangian_hessian=hl_fd.tolist()))
    rich_ground = (4*np.asarray(fd[1]["ground_gradient"])-fd[0]["ground_gradient"])/3
    rich_gap = (4*np.asarray(fd[1]["excitation_gradient"])-fd[0]["excitation_gradient"])/3
    rich_hessian = (4*np.asarray(fd[1]["lagrangian_hessian"])-fd[0]["lagrangian_hessian"])/3
    centered = positions-positions.mean(axis=0)
    rigid = []
    for axis in np.eye(3):
        rigid.extend([np.tile(axis,(len(numbers),1)).ravel(),
            np.cross(np.tile(axis,(len(numbers),1)),centered).ravel()])
    _, singular, vh = np.linalg.svd(np.asarray(rigid),full_matrices=True)
    rank = int(np.count_nonzero(singular>singular[0]*1e-10))
    internal = vh[rank:].T
    normal = internal.T@base["gg"]
    _, _, vh_gap = np.linalg.svd(normal.reshape(1,-1),full_matrices=True)
    tangent = internal@vh_gap[1:].T
    curvature, eigenvectors = np.linalg.eigh(tangent.T@((hl+hl.T)/2)@tangent)
    directions = np.column_stack([internal@normal/np.linalg.norm(normal),tangent@eigenvectors])
    upper = base["e0"]+max(0.,base["gap"])
    perturbations = []
    for step in (1e-3,5e-4):
        for direction in range(directions.shape[1]):
            for sign in (1,-1):
                value = evaluate(positions+(sign*step*directions[:,direction]).reshape(-1,3))
                delta = (value["e0"]+max(0.,value["gap"])-upper)*Hartree
                perturbations.append(dict(step_bohr=step,direction=direction,sign=sign,delta_upper_energy_ev=delta))
    errors = dict(ground_gradient=float(np.max(np.abs(base["g0"]-rich_ground))),
        excitation_gradient=float(np.max(np.abs(base["gg"]-rich_gap))),
        lagrangian_hessian=float(np.max(np.abs(hl-rich_hessian))))
    passed = (all(errors[name]<5e-8 for name in ("ground_gradient","excitation_gradient"))
        and errors["lagrangian_hessian"]<2e-7 and rank==6 and len(curvature)==11
        and float(curvature[0])>0 and 0<multiplier<1
        and max(p["delta_upper_energy_ev"] for p in perturbations)>0
        and min(p["delta_upper_energy_ev"] for p in perturbations)>-1e-8)
    report = dict(schema="gfn1-rs-uncoupled-crossing-audit-v1",package_version=__version__,
        status="pass" if passed else "fail", temperature_kelvin=300.,energy_window_ev=20.,
        surface="upper envelope max(F_GFN1, F_GFN1+Omega_sTDA); an uncoupled composite model, not a physical conical-intersection claim",
        numbers=numbers,positions_angstrom=frame["positions_angstrom"],multiplier=multiplier,
        gap_ev=base["gap"]*Hartree,next_ci_gap_ev=base["ci_gap"]*Hartree,
        ground_gradient_hartree_per_bohr=base["g0"].tolist(),
        excitation_gradient_hartree_per_bohr=base["gg"].tolist(),
        lagrangian_gradient_hartree_per_bohr=gkkt.tolist(),
        kkt_fmax_ev_per_angstrom=float(np.linalg.norm(gkkt.reshape(-1,3),axis=1).max()*Hartree/Bohr),
        tested_ground_gradient_components=positions.size,tested_excitation_gradient_components=positions.size,
        tested_lagrangian_hessian_components=int(hl.size),errors=errors,fd=fd,
        analytic_lagrangian_hessian_hartree_per_bohr2=hl.tolist(),
        richardson_lagrangian_hessian_hartree_per_bohr2=rich_hessian.tolist(),
        rigid_rank=rank,seam_tangent_curvatures_hartree_per_bohr2=curvature.tolist(),
        upper_envelope_energy_checks=perturbations,
        input_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=report["status"],errors=errors,kkt_fmax=report["kkt_fmax_ev_per_angstrom"],
        min_curvature=float(curvature[0]),minimum_energy_increase_ev=min(p["delta_upper_energy_ev"] for p in perturbations)),indent=2))
    if not passed:raise ValueError("The independent crossing checks failed")


if __name__=="__main__":main()
