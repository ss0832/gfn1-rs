#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit the original signed PES on shrinking circles about a frontier seam.

No surface replacement or nuclear constraint is used in the production API.
The circles are diagnostic displacements, and angular differences are only
independent verification of the analytic Cartesian gradient/Hessian.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ase.units import Bohr, Hartree
from gfn1_rs import Gfn1NativeCalculator
import gfn1_rs._native as extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('tests/data/stda-pes-degeneracy.json'))
    parser.add_argument('--output', type=Path, default=Path('target/validation/stda-angular-response.json'))
    parser.add_argument('--angles', type=int, default=16)
    parser.add_argument('--metric', choices=('branch-gap','nuclear'), default='branch-gap')
    args = parser.parse_args()
    source = json.loads(args.reference.read_text())
    centre = np.asarray(source['seam_positions_angstrom'])
    directions = np.asarray(source['directions']).reshape(2, *centre.shape)
    numbers = source['reference_geometry']['numbers']
    assert np.max(np.abs(directions.reshape(2, -1) @ directions.reshape(2, -1).T - np.eye(2))) < 1e-8
    calc = Gfn1NativeCalculator(electronic_temperature=300., energy_tolerance=1e-13,
        charge_tolerance=1e-11, max_scc=1000, threads=2, memory_mb=1024)
    slopes = []
    for direction in directions:
        gaps = []
        for sign in (-1,1):
            spectrum = calc.stda_xtb(numbers,(centre+sign*1e-6*direction).tolist(),energy_window_ev=200.)
            no = min(pair[1] for pair in spectrum['configurations'])
            energy = spectrum['orbital_energies_hartree']
            gaps.append((energy[no]-energy[no-1])*27.21139570-3.1)
        slopes.append(float(np.mean(gaps)/1e-6))
    assert min(slopes)>0
    if args.metric=='branch-gap':
        # Equalize the first-order electronic gap in the two branching axes.
        # A uniform nuclear circle misses the very narrow angular feature when
        # their sensitivities differ strongly; this remains the same PES.
        directions[1] *= slopes[0]/slopes[1]
    cases = []
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(radius, theta):
        offset = radius * (np.cos(theta)*directions[0] + np.sin(theta)*directions[1])
        position = centre + offset
        result = calc.stda_xtb_gradient(numbers, position.tolist(), state=0, energy_window_ev=200.)
        return position, result, np.asarray(result['gradient'])*Hartree/Bohr

    def save():
        report = dict(schema='gfn1-rs-original-stda-angular-response-v1', package_version='0.7.0',
            model='unchanged signed F_GFN1+Omega_sTDA state0; all120 configurations primary',
            electronic_temperature_kelvin=300., nuclear_constraints='none; circles are diagnostic displacements',
            seam_positions_angstrom=centre.tolist(), directions=directions.tolist(),
            sampling_metric=args.metric, directional_gap_slopes_ev_per_angstrom=slopes,
            input_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            measured_extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-start, cases=cases)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    for radius in ((1e-5,1e-6,1e-7) if args.metric=='branch-gap' else (1e-3,1e-4,1e-5,1e-6)):
        for i in range(args.angles):
            theta = 2*np.pi*i/args.angles
            position, result, gradient = evaluate(radius, theta)
            tangent = radius*(-np.sin(theta)*directions[0] + np.cos(theta)*directions[1])
            radial = (position-centre)/radius
            analytic = float(np.sum(gradient*tangent))
            fd = []
            for step in (0.01,0.005):
                _, plus, _ = evaluate(radius, theta+step)
                _, minus, _ = evaluate(radius, theta-step)
                fd.append((plus['total_energy_hartree']-minus['total_energy_hartree'])*Hartree/(2*step))
            richardson = (4*fd[1]-fd[0])/3
            spectrum = calc.stda_xtb(numbers, position.tolist(), n_states=10, energy_window_ev=200.)
            assert abs(spectrum['ci_eigenvalues_hartree'][0]-result['excitation_energy_hartree'])<1e-10
            assert spectrum['primary_configuration_count']==120 and spectrum['promoted_configuration_count']==0
            # The dedicated model uses valence electrons, not sum(Z).
            # At this full window every occupied/virtual pair is present.
            no = min(pair[1] for pair in spectrum['configurations'])
            orbital = spectrum['orbital_energies_hartree']
            unshifted_gap = (orbital[no]-orbital[no-1])*27.21139570-3.1
            row = dict(radius_angstrom=radius, angle_radians=float(theta),
                total_energy_ev=result['total_energy_hartree']*Hartree,
                excitation_ev=result['excitation_energy_hartree']*Hartree,
                fmax_ev_per_angstrom=float(np.linalg.norm(gradient, axis=1).max()),
                angular_gradient_ev=analytic, angular_gradient_fd_ev=richardson,
                angular_gradient_error_ev=abs(analytic-richardson),
                reference_radius_derivative_ev_per_angstrom=float(np.sum(gradient*radial)),
                unshifted_frontier_gap_ev=unshifted_gap,
                primary_configuration_count=spectrum['primary_configuration_count'],
                promoted_configuration_count=spectrum['promoted_configuration_count'])
            if i in (0, args.angles//4):
                hessian = np.asarray(calc.stda_xtb_hessian(numbers, position.tolist(), state=0,
                    energy_window_ev=200.)['hessian'])*Hartree/Bohr**2
                curvature = float(tangent.ravel() @ hessian @ tangent.ravel()
                                  - np.sum(gradient*(position-centre)))
                angular_fd = []
                for step in (0.01,0.005):
                    pp, _, gp = evaluate(radius, theta+step)
                    pm, _, gm = evaluate(radius, theta-step)
                    tp = radius*(-np.sin(theta+step)*directions[0]+np.cos(theta+step)*directions[1])
                    tm = radius*(-np.sin(theta-step)*directions[0]+np.cos(theta-step)*directions[1])
                    angular_fd.append(float((np.sum(gp*tp)-np.sum(gm*tm))/(2*step)))
                check = (4*angular_fd[1]-angular_fd[0])/3
                row.update(angular_curvature_ev=curvature, angular_curvature_fd_ev=check,
                           angular_curvature_error_ev=abs(curvature-check))
            cases.append(row)
            save()
        group = cases[-args.angles:]
        print(radius, 'angular energy spread', np.ptp([c['excitation_ev'] for c in group]),
              'max angular gradient FD error', max(c['angular_gradient_error_ev'] for c in group), flush=True)
    save()


if __name__ == '__main__':
    main()
