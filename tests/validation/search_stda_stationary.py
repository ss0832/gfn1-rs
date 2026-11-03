#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unconstrained roots of the original native GFN1+sTDA nuclear gradient.

Minimizes the squared gradient with analytic gradient/Hessian information.
Negative Hessian eigenvalues are retained in the Newton step. A force-zero
point is classified by its internal Hessian, never automatically called a
minimum. Neither a crossing condition nor a planar constraint is imposed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ase.units import Bohr, Hartree
from gfn1_rs import Gfn1NativeCalculator, __version__


def internal_space(r):
    centered = r-r.mean(axis=0)
    rigid = []
    for axis in np.eye(3):
        rigid.extend((np.tile(axis, (len(r), 1)).ravel(), np.cross(axis, centered).ravel()))
    _, s, vh = np.linalg.svd(rigid, full_matrices=True)
    return vh[np.count_nonzero(s>s[0]*1e-10):].T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angles', default='5,30,60,120,150')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--fmax', type=float, default=1e-5)
    parser.add_argument('--output', type=Path, default=Path('target/validation/stda-stationary'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root/'tests/data/tda-stda-native-comparison.json'
    case = next(c for c in json.loads(source.read_text())['cases'] if c['name']=='ethylene')
    args.output.mkdir(parents=True, exist_ok=True)
    calc = Gfn1NativeCalculator(electronic_temperature=300., energy_tolerance=1e-13,
                                 charge_tolerance=1e-11, max_scc=1000)
    def evaluate(r):
        value = calc.stda_xtb_gradient(case['numbers'], (r*Bohr).tolist(), state=0, energy_window_ev=20.)
        return value, np.asarray(value['gradient']).ravel()
    for angle in map(float, args.angles.split(',')):
        r = np.asarray(case['positions_angstrom'], dtype=float)/Bohr
        axis = r[1]-r[0]; axis /= np.linalg.norm(axis)
        theta = np.deg2rad(angle)
        for i in range(2, len(r)):
            if np.linalg.norm(r[i]-r[1]) < np.linalg.norm(r[i]-r[0]):
                v = r[i]-r[1]
                r[i] = r[1]+v*np.cos(theta)+np.cross(axis,v)*np.sin(theta)+axis*(axis@v)*(1-np.cos(theta))
        jitter = np.random.default_rng(173).normal(scale=.001/Bohr, size=r.shape)
        r += jitter-jitter.mean(axis=0)
        report = dict(schema='gfn1-rs-stda-gradient-root-v1', package_version=__version__,
            numbers=case['numbers'], electronic_temperature_kelvin=300., state=0,
            model='original signed F_GFN1+Omega_sTDA', geometry_constraints='none',
            initial_twist_degrees=angle, initial_positions_angstrom=(r*Bohr).tolist(),
            algorithm='damped signed-Newton / squared-gradient line search', frames=[],
            source_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status='running')
        path = args.output/f'twist-{angle:g}.json'
        start = time.monotonic()
        def save():
            report['elapsed_seconds'] = time.monotonic()-start
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        for iteration in range(args.steps):
            value, g = evaluate(r)
            basis = internal_space(r)
            h = np.asarray(calc.stda_xtb_hessian(case['numbers'], (r*Bohr).tolist(), state=0,
                                                energy_window_ev=20.)['hessian'])
            h = basis.T@((h+h.T)*.5)@basis
            eig, u = np.linalg.eigh(h)
            gi = basis.T@g
            fmax = float(np.linalg.norm(g.reshape(-1,3),axis=1).max()*Hartree/Bohr)
            frame = dict(iteration=iteration,positions_angstrom=(r*Bohr).tolist(),
                energy_ev=value['total_energy_hartree']*Hartree,
                excitation_ev=value['excitation_energy_hartree']*Hartree,
                raw_fmax_ev_per_angstrom=fmax, gradient_hartree_per_bohr=g.reshape(-1,3).tolist(),
                internal_hessian_eigenvalues_hartree_per_bohr2=eig.tolist(),
                negative_modes=int(np.count_nonzero(eig < -1e-6)))
            report['frames'].append(frame); save()
            print(angle,iteration,frame['energy_ev'],fmax,frame['negative_modes'],flush=True)
            if fmax < args.fmax:
                report['status']='force_zero_minimum' if frame['negative_modes']==0 and eig[0]>1e-6 else 'force_zero_saddle_or_soft_point'
                save();break
            merit = .5*float(gi@gi)
            accepted = False
            for damping in (0., 1e-6, 1e-4, .01, 1.):
                denominator = eig*eig+damping
                if np.min(denominator)<1e-24:
                    continue
                step = basis@(-u@((eig/denominator)*(u.T@gi)))
                maximum = np.linalg.norm(step.reshape(-1,3),axis=1).max()
                step *= min(1.,.04/Bohr/max(maximum,1e-30))
                slope = float(gi@h@(basis.T@step))
                if slope>=0:continue
                for backtrack in range(24):
                    alpha = 2.**(-backtrack)
                    trial = r+alpha*step.reshape(-1,3)
                    try:
                        _, trial_g = evaluate(trial)
                        trial_merit = .5*float(trial_g@trial_g)
                        if trial_merit <= merit+1e-4*alpha*slope:
                            r=trial;accepted=True;break
                    except (RuntimeError,ValueError) as error:
                        frame['last_rejected_error']=str(error)
                if accepted:break
            frame['accepted_step']=accepted;save()
            if not accepted:
                report['status']='gradient_root_line_search_stalled';save();break
        else:
            report['status']='step_limit';save()
        print(angle,report['status'],flush=True)


if __name__=='__main__':main()
